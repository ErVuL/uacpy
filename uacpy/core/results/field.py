"""The unified :class:`Field` result, the :class:`ResultStack`, and the
broadband-to-time-series IFFT synthesis helpers (kept with Field because
they construct it)."""

from __future__ import annotations

import warnings
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union

from uacpy.core._carrier_validate import _DeepCopyMixin, _require_finite, _reject_complex
from uacpy.core.constants import (DEFAULT_SOUND_SPEED,
                                  REFERENCE_PRESSURE_WATER)
from uacpy.core.exceptions import ConfigurationError
from uacpy.core._grid import _nearest_index_on_axis, collapse_axis
from uacpy.core.environment import Bathymetry, Environment
from uacpy.core._warn_frames import USER_FRAME_SKIP

from uacpy.core.acoustics.levels import (
    peak_level as _peak_level,
    sound_exposure_level as _sound_exposure_level,
)
from uacpy.core.results import quantities as _quantities
from uacpy.core.constants import PRESSURE_FLOOR
from uacpy.core.results._base import PhaseReference, Result, _complex_to_dB

# Auto-sized IFFT length is ~sample_rate/df rounded up to a power of two, so a
# too-high sample_rate (or a too-fine frequency grid) can silently demand a
# multi-GB buffer and OOM the process. Cap the *auto* size at 2**26 ≈ 67 M
# samples (~1 GB complex) and raise instead; an explicit ``nfft=`` bypasses it.
_MAX_SYNTHESIS_NFFT = 1 << 26


class Field(Result):
    """Generic gridded result. One container for every spatially or
    spectrally gridded uacpy output.

    The dtype of :attr:`data` plus the keys in :attr:`coords` tell you
    what the field represents:

    =========================  ================================  =====================================
    dtype                      ``coords`` keys                    Physical meaning
    =========================  ================================  =====================================
    complex                    ``{depth, range}``                Narrowband pressure ``p(d, r)``
    real                       ``{depth, range}``                TL in dB
    complex                    ``{depth, range, frequency}``     Broadband ``H(d, r, f)``
    real                       ``{depth, range, time}``          Time-domain ``p(d, r, t)``
    real                       ``{time}``                        Single-point trace
    complex                    ``{source_depth, depth, range}``  Multi-source complex pressure (``.kind == 'pressure'``; ``.dB`` derives dB)
    =========================  ================================  =====================================

    ``data.shape`` matches the insertion order of :attr:`coords`, which
    **is** the axis order — reordering ``coords`` after construction
    desynchronises it from ``data``. The canonical order is
    ``source_depth → depth → range → frequency`` (or ``time``).

    Axis units and signs: ``depth`` and ``source_depth`` in metres below
    the sea surface (positive down, as in the ``.env`` sound-speed profile
    every wrapped model reads), ``range`` in metres from the source,
    ``time`` in seconds, ``frequency`` in Hz.

    Payload and derived views
    -------------------------
    :attr:`data` is the payload, and it is **writeable**: the attribute is
    the stored array itself, so ``field.data *= k`` rescales the result in
    place. The derived views refuse that — :attr:`p` and the real branch of
    :attr:`dB` (and of :attr:`tl`, which is ``.dB`` under the quantity's
    name on pressure fields) hand back arrays with ``writeable=False`` —
    but ``.p`` is a
    view of *this same buffer*, so its read-only flag protects the accessor
    rather than the field: a write through ``.data`` changes what ``.p`` and
    ``.dB`` return afterwards. What the copy-on-ingest in the constructor
    guarantees is the other direction — the stored array never aliases the
    caller's, so mutating the array you passed in cannot reach the Field.

    Slicing
    -------
    :meth:`at` (label) and :meth:`isel` (index) collapse a named axis
    to a single sample. The axis is **dropped** from :attr:`coords` and
    the selected coordinate value is recorded in :attr:`pinned`::

        narrow = tf.at(frequency=200)
        narrow.coords        # {'depth': ..., 'range': ...}
        narrow.pinned        # {'frequency': 198.4}    nearest sample

    :meth:`max` does the same for every axis at once (picking the
    argmax of ``|data|``) — returns a scalar Field with empty
    ``coords`` and every axis pinned.

    :attr:`pinned` is therefore the record of what was dropped:
    ``{axis_name: label}`` in that axis's own units, never an index. It
    accumulates over successive slices and every derived Field inherits it,
    so a consumer can always recover which cell a reduced result came from
    — that is what the plotters put in the subtitle and what
    :meth:`plot_impulse_response` reads to place its time window. A pinned
    value is one of the stored samples for :meth:`at` / :meth:`isel` /
    :meth:`max`, but the *requested* value clamped into range for
    :meth:`eval`, so a consumer must not assume it appears in any coord
    array. Pinning ``frequency`` or ``source_depth`` also narrows the
    identity fields ``frequencies`` / ``source_depths`` to the pinned value.
    """

    field_type = "field"

    def __init__(
        self,
        *,
        data: np.ndarray,
        coords: Dict[str, np.ndarray],
        pinned: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not isinstance(coords, dict):
            raise ConfigurationError(
                "Field.coords: must be a dict of axis_name → 1-D array"
            )
        normalised: Dict[str, np.ndarray] = {}
        for name, v in coords.items():
            # Ahead of the float64 cast below, which discards an imaginary
            # part — see _reject_complex for the two ways it does it. A
            # complex coordinate is the axis, not the data: :attr:`data` is
            # complex on every pressure field, but at() and the slicers read
            # the coords as real distances.
            _reject_complex(v, f"Field.coords[{name!r}]")
            # np.array (not asarray) so each Field owns its coord vectors —
            # slices/derived Fields never alias a parent's (or caller's) arrays.
            arr = np.atleast_1d(np.array(v, dtype=float))
            if arr.ndim != 1:
                raise ConfigurationError(
                    f"Field.coords[{name!r}]: must be 1-D; got shape {arr.shape}"
                )
            # A NaN/inf coordinate makes every |axis - label| distance on
            # this axis NaN at that sample, so at()'s argmin can land on it
            # and hand back a sample no label ever named.
            _require_finite(arr, f"Field.coords[{name!r}]")
            normalised[name] = arr
        self.coords: Dict[str, np.ndarray] = normalised

        # Copy on ingest so the stored field never aliases the caller's (or a
        # model's scratch) array — external mutation of the source must not
        # silently corrupt the result.
        data = np.array(data)
        expected = tuple(normalised[name].size for name in normalised)
        if data.shape != expected:
            raise ConfigurationError(
                f"Field.data: shape {data.shape} does not match coord sizes "
                f"{expected} (axes: {list(normalised)})"
            )
        self.data = data
        self.pinned: Dict[str, float] = (
            {k: float(v) for k, v in pinned.items()} if pinned else {}
        )
        # Validate a quantity tag where it enters, not where it is read: a
        # typo'd kind that survives construction resurfaces as a wrong colour
        # scale or a wrong argmax direction, with nothing pointing back here.
        # The untagged path — every slice, every model default — skips the
        # lookup entirely.
        meta = self.metadata or {}
        if 'kind' in meta or 'unit' in meta:
            _quantities.label(self.kind, self.unit)

    # ── shape / dtype ─────────────────────────────────────────────────

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def axes(self) -> List[str]:
        return list(self.coords)

    @property
    def is_complex(self) -> bool:
        return bool(np.iscomplexobj(self.data))

    @property
    def kind(self) -> str:
        """What the field physically **is** — the quantity it carries.

        ``'pressure'`` by default; models producing something else tag it via
        ``metadata['kind']`` (e.g. ``'reverberation'``). This is one of three
        independent axes, and asking the wrong one is how consumers break:

        ==============  =========================  ========================
        axis            question it answers        ask it for
        ==============  =========================  ========================
        ``kind``        *what* is this?            is comparing these two
                                                   fields meaningful at all
        ``unit``        what is it *measured in*?  which direction is louder
        ``data.dtype``  how is it *stored*?        is there phase to work with
        ==============  =========================  ========================

        There is no ``Field.dtype``; ask :attr:`data` for its ``dtype``, or
        :attr:`is_complex` for the boolean.

        Transmission loss is **not** a separate kind: ``-20·log10|p|`` is the
        same pressure field written in dB, which is the ``unit`` axis's job.
        That is why a RAM TL field and a Kraken complex field compare — same
        kind — while reverberation, which shares TL's representation exactly,
        does not.

        The **domain** is not a fourth axis either: it is already in
        :attr:`coords`, as a ``'time'`` or ``'frequency'`` entry.
        """
        tagged = (self.metadata or {}).get('kind')
        return str(tagged) if tagged else 'pressure'

    @property
    def unit(self) -> str:
        """What :attr:`data` is measured in — ``'Pa'`` or ``'dB'``.

        Derived unless a model tags ``metadata['unit']``: complex data and
        time-domain traces are linear pressure, real frequency-domain data is
        a level in dB. Consumers that need to know which way is louder must
        ask **this** and never :attr:`kind`, or every new dB quantity silently
        inverts them — see :meth:`max`.
        """
        tagged = (self.metadata or {}).get('unit')
        if tagged:
            return str(tagged)
        units = _quantities.quantity(self.kind).units
        if len(units) == 1:
            return next(iter(units))
        # Pressure alone carries two units, and which one is a *storage*
        # question: phase surviving (complex) or a time trace means linear Pa;
        # a real frequency-domain grid is already a level.
        return 'Pa' if (self.is_complex or 'time' in self.coords) else 'dB'

    # ── persistence ───────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialise this field to a plain dict for caching / round-trip.

        Values are numpy arrays and Python scalars (data preserves its
        real/complex dtype), so the result is directly picklable and
        ``np.savez``-able; convert the arrays to lists yourself for JSON.
        ``coords`` insertion order matches the data axes. ``kind`` and
        ``unit`` are included for inspection but are recomputed by
        :meth:`from_dict` (both derive from ``metadata`` and the data).
        Reconstruct with ``Field.from_dict(d)``.
        """
        return {
            'kind': self.kind,
            'unit': self.unit,
            'data': self.data.copy(),
            'coords': {k: v.copy() for k, v in self.coords.items()},
            'pinned': dict(self.pinned),
            'model': self.model,
            'backend': self.backend,
            'source_depths': self.source_depths.copy(),
            'frequencies': (None if self.frequencies is None
                            else self.frequencies.copy()),
            'phase_reference': self.phase_reference,
            'model_source': self.model_source,
            'metadata': dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Field':
        """Reconstruct a :class:`Field` from :meth:`to_dict` output."""
        return cls(
            data=np.asarray(d['data']),
            coords={k: np.asarray(v) for k, v in d['coords'].items()},
            pinned=d.get('pinned') or None,
            model=d.get('model', ''),
            backend=d.get('backend'),
            source_depths=d.get('source_depths'),
            frequencies=d.get('frequencies'),
            phase_reference=d.get('phase_reference'),
            model_source=d.get('model_source'),
            metadata=d.get('metadata'),
        )

    def __repr__(self) -> str:
        bits = [f"kind={self.kind!r}", f"unit={self.unit!r}"]
        if self.model:
            bits.append(f"model={self.model!r}")
        # One frequency prints as a frequency, several as a count — the
        # branch Result.__repr__ makes and this override had dropped, so a
        # field carrying a whole band in its identity but no frequency axis
        # (a synthesised time series, say) printed the band's FIRST sample
        # as though that were the field's frequency.
        if 'frequency' not in self.coords:
            band = self.frequencies
            if band is not None and len(band) > 1:
                bits.append(f"n_f={len(band)}")
            elif self.f0 is not None:
                bits.append(f"f={self.f0:.3g} Hz")
        bits.append(f"axes=({', '.join(self.coords) or 'scalar'})")
        return f"Field({', '.join(bits)})"

    # ── value accessors ───────────────────────────────────────────────

    @property
    def dB(self) -> np.ndarray:
        """This field's dB view, at ``data.shape``.

        ``-20·log10(|data|)`` if data is complex — for pressure that is
        transmission loss (a fresh array) — otherwise ``data`` **itself**
        (real data outside the time domain is already a level), as a
        **read-only view**: the dB values are the field, so mutating them in
        place would corrupt the result.

        The real branch converts nothing, so the array carries the field's
        own **dtype** — float32 for a ``.shd``-backed result, since
        :meth:`to_dB` of a complex64 field is float32 — and it **aliases**
        :attr:`data` whatever that dtype is, so a later write through
        ``data`` shows up in an array taken earlier. A caller that needs
        float64 regardless of the engine that produced the field asks for it:
        ``np.asarray(field.dB, dtype=float)``.

        Named for the *unit*, not the quantity: on a reverberation or
        signal-excess field this returns that quantity's dB values, and
        calling it ``.tl`` would have been the same misnomer the ``kind``
        axis removed. :attr:`tl` is the pressure-only spelling and refuses
        every other ``kind``. Which **direction** those dB values run is a
        separate question the unit cannot answer — see :meth:`max`.

        Raises :class:`AttributeError` for a time-domain field (one
        carrying a ``'time'`` axis) — a time trace is linear pressure, not a
        level; use ``.data`` to read raw samples or ``.extract_tone(f)`` to
        recover a complex narrowband field first."""
        if 'time' in self.coords:
            raise AttributeError(
                "Field.dB: a time-domain trace is linear pressure, not a "
                "level; use .data for raw samples or .extract_tone(f) to "
                "recover a complex narrowband field first"
            )
        if self.is_complex:
            return _complex_to_dB(self.data)
        # Real data is handed back as-is, which is only a level if the field
        # says it is one. A Field carrying a dimensionless quantity — e.g.
        # `sonar_equation`'s probability-of-detection field, kind=
        # 'probability_of_detection', unit='1' — otherwise had its raw values
        # returned as though they were dB. `Field.max()` already consults
        # `self.unit` to pick its direction, so the two accessors disagreed.
        # ``to_dB()`` is not the remedy to offer here: it returns ``self`` for
        # any real field (its first statement), and this branch is reachable
        # only for real data — the complex branch above returns before it. The
        # set of fields that can see this message is exactly the set on which
        # ``to_dB()`` does nothing, so the message names the arithmetic
        # instead.
        if self.unit != 'dB':
            raise AttributeError(
                f"Field.dB: this field is in {self.unit!r}, not dB, so its "
                f"values are not a level; use .data for the raw values. "
                f"to_dB() returns a real field unchanged, so if a dB view of "
                f"{self.unit!r} is meaningful, take "
                f"20*np.log10(np.abs(field.data)) yourself and tag the result "
                f"unit='dB'."
            )
        view = self.data.view()
        view.flags.writeable = False
        return view

    @property
    def tl(self) -> np.ndarray:
        """Transmission loss in dB — :attr:`dB` restricted to pressure fields.

        The values are exactly :attr:`dB`'s (``-20·log10(|data|)`` for
        complex pressure; the same read-only view for a real dB pressure
        field), under the name the quantity carries in the literature, so
        a reader of ``result.tl`` knows the field is pressure-derived
        without consulting :attr:`kind`.

        Raises :class:`AttributeError` for any other ``kind``: a
        reverberation or detection field has its own dB view in :attr:`dB`,
        and returning it here would label that quantity a transmission
        loss. Reverberation is a loss too, and runs the same direction, but
        it is a different loss — scattering, not one-way propagation — so
        the two do not share a scale."""
        if self.kind != 'pressure':
            raise AttributeError(
                f"Field.tl: this field's kind is {self.kind!r}, not "
                f"'pressure', so its values are not a transmission loss; "
                f"its level view is .dB."
            )
        return self.dB

    @property
    def p(self) -> np.ndarray:
        """Complex pressure / transfer-function values, as a read-only view.

        Raises when :attr:`data` is real — phase has been discarded.

        The read-only flag is on this view, not on the buffer: :attr:`data`
        is the same memory and is writeable, so ``field.data *= k`` rescales
        what this returns. See "Payload and derived views" in the class
        docstring."""
        if not self.is_complex:
            raise AttributeError(
                "Field.p: data is real; complex pressure unavailable"
            )
        # Hand back a read-only view: callers must not mutate the field's
        # internal pressure array in place (``p = field.p; p *= k`` would
        # otherwise silently corrupt the result).
        view = self.data.view()
        view.flags.writeable = False
        return view

    @property
    def magnitude(self) -> np.ndarray:
        """Element-wise amplitude ``|data|`` (complex fields only)."""
        if not self.is_complex:
            raise AttributeError(
                "Field.magnitude: requires complex data"
            )
        return np.abs(self.data)

    @property
    def phase(self) -> np.ndarray:
        """Element-wise phase angle in radians, ``angle(data)`` (complex fields only)."""
        if not self.is_complex:
            raise AttributeError("Field.phase: requires complex data")
        return np.angle(self.data)

    # ── coord-axis conveniences ───────────────────────────────────────

    @property
    def depths(self) -> Optional[np.ndarray]:
        return self.coords.get('depth')

    @property
    def ranges(self) -> Optional[np.ndarray]:
        return self.coords.get('range')

    @property
    def times(self) -> Optional[np.ndarray]:
        return self.coords.get('time')

    @property
    def n_depths(self) -> int:
        z = self.coords.get('depth')
        return int(z.size) if z is not None else 0

    @property
    def n_ranges(self) -> int:
        r = self.coords.get('range')
        return int(r.size) if r is not None else 0

    @property
    def n_times(self) -> int:
        t = self.coords.get('time')
        return int(t.size) if t is not None else 0

    @property
    def n_frequencies(self) -> int:
        """Number of frequencies, from the identity list or the
        ``'frequency'`` coord under the same length guard :attr:`f0` uses;
        0 for time-domain results."""
        if self.frequencies is not None and len(self.frequencies):
            return int(len(self.frequencies))
        f = self.coords.get('frequency')
        return int(f.size) if f is not None else 0

    @property
    def f0(self) -> Optional[float]:
        """First / centre frequency (Hz), from the identity list or the
        ``'frequency'`` coord; ``None`` for time-domain results."""
        if self.frequencies is not None and len(self.frequencies):
            return float(self.frequencies[0])
        f = self.coords.get('frequency')
        if f is not None and f.size:
            return float(f[0])
        return None

    def _warn_if_frequency_axis_undersamples(self, wanted, where: str) -> None:
        """Warn when the FREQUENCY axis is too coarse to interpolate coherently.

        The frequency axis carries the same rotating carrier as depth and
        range, but its condition is different: a transfer function holds
        ``exp(-2*pi*i*f*r/c)``, so the phase advance between two stored bins is
        ``2*pi*df*r/c`` and the quarter-cycle limit is ``df < c/(4*r)`` at the
        field's FARTHEST range. That is usually the tightest axis on the whole
        field — at 5 km and df = 1 Hz the carrier turns 3.3 whole cycles
        between bins. Measured on ``H(f) = exp(-2*pi*i*f*r/c)/r`` with 1 Hz
        bins, interpolating to a half-bin frequency: -10.96 dB at r = 613 m,
        and 180-degree phase errors at 1877 m, 5000 m and 31.7 km, all silent.
        """
        if 'frequency' not in tuple(wanted):
            return
        freqs = self.coords.get('frequency')
        # After ``.at(range=…)`` the axis is gone and the cell's own range sits
        # in ``pinned`` — the single-cell spectrum interpolates the identical
        # carrier, so the coord spelling and the pinned one both feed the
        # limit. The pinned value is the cell's own range, which is the right
        # one for it: reading the collapsed axis instead would judge every cell
        # by the far end of the field.
        ranges = self.coords.get('range')
        if ranges is None and 'range' in self.pinned:
            ranges = np.asarray([self.pinned['range']], dtype=float)
        if freqs is None or np.size(freqs) < 2 or ranges is None \
                or not np.size(ranges):
            return
        # |diff|: a descending axis stores the same bins and ``eval`` returns
        # the same values, so the spacing it is judged by is the same too.
        df = float(np.max(np.abs(np.diff(np.asarray(freqs, dtype=float)))))
        r_max = float(np.max(np.abs(np.asarray(ranges, dtype=float))))
        if r_max <= 0.0 or df <= 0.0:
            return
        df_limit = DEFAULT_SOUND_SPEED / (4.0 * r_max)
        if df <= df_limit:
            return
        where_r = ("this cell's range" if 'range' not in self.coords
                   else "the field's farthest range")
        warnings.warn(
            f"{where}: frequency samples are {df:g} Hz apart, over the "
            f"{df_limit:.3g} Hz quarter-cycle limit c/(4r) at {where_r} "
            f"r = {r_max:g} m (nominal "
            f"c={DEFAULT_SOUND_SPEED:g} m/s), so the carrier turns "
            f"{df * r_max / DEFAULT_SOUND_SPEED:.2f} cycles between stored "
            f"bins. Interpolating across it cuts the carrier: the level and "
            f"the phase are both unreliable. Re-run the model on the target "
            f"frequencies instead, or take .dB first if only the level is "
            f"wanted.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)

    def _highest_frequency(self) -> Optional[float]:
        """Highest frequency (Hz) the field carries, or ``None``.

        Resolved from the same two sources as :attr:`f0` and in the same
        order. The undersampling guard wants this rather than ``f0`` because
        the quarter-wavelength condition binds at the shortest wavelength
        present, which is the top of the band.
        """
        if self.frequencies is not None and len(self.frequencies):
            return float(np.max(np.asarray(self.frequencies, dtype=float)))
        f = self.coords.get('frequency')
        if f is not None and f.size:
            return float(np.max(np.asarray(f, dtype=float)))
        return None

    @property
    def dt(self) -> float:
        """Time-axis sample spacing in seconds (``0.0`` if not time-resolved)."""
        t = self.coords.get('time')
        if t is None or t.size < 2:
            return 0.0
        return float(t[1] - t[0])

    @property
    def sample_rate(self) -> float:
        """Sampling rate in Hz (``1/dt``; ``0.0`` if not time-resolved)."""
        dt = self.dt
        return 1.0 / dt if dt > 0 else 0.0

    # ── slicing ────────────────────────────────────────────────────────

    def at(self, **kwargs) -> "Field":
        """Label-based slice. Each kwarg names a coord axis; nearest
        sample is picked and the axis is **dropped** from :attr:`coords`
        (its selected value lands in :attr:`pinned`).

        The label must be finite. ``argmin`` over ``|coord − label|`` has no
        nearest sample to find when every distance is ``NaN`` or ``inf``, and
        an over-large label loses the coord to float cancellation the same
        way; all three land on index 0, which is a real sample and so reads
        as a successful slice."""
        self._check_axes(kwargs)
        # The label guards (finite, not axis-absorbing) and the nearest rule
        # are the ones every non-blendable carrier shares.
        labels = {name: _nearest_index_on_axis(self.coords[name], v, name)
                  for name, v in kwargs.items()}
        return self._slice(labels)

    def isel(self, **kwargs) -> "Field":
        """Integer-index slice. Same semantics as :meth:`at` but the
        value is a positional index into the coord array."""
        self._check_axes(kwargs)
        return self._slice({name: int(i) for name, i in kwargs.items()})

    def window(self, **bounds) -> "Field":
        """Label-based axis window: narrow an axis and **keep** it.

        :meth:`at` and :meth:`isel` collapse an axis to one sample. This
        narrows one instead: each kwarg names a coord axis and an inclusive
        ``(lo, hi)`` pair in that axis's own units, samples outside it are
        dropped, and the axis survives with what remains — so the result is
        still a field over that axis rather than a slice through it. ``None``
        for either end leaves that end alone, so ``window(time=(0.0, None))``
        trims a pre-roll and nothing else.

        Several models put the same scene on different spans — a time-marching
        solver integrating from a negative pre-roll while an IFFT one starts at
        zero — and comparing them means cutting both to one window.

        Raises when a window selects no sample: an empty axis is not a smaller
        field but a field with nothing in it, and every later slice of it would
        fail somewhere less obvious.

        Examples
        --------
        >>> import numpy as np
        >>> from uacpy.core.results import Field
        >>> f = Field(data=np.arange(5.0).reshape(1, 5),
        ...           coords={'depth': np.array([10.0]),
        ...                   'range': np.linspace(0.0, 400.0, 5)})
        >>> f.window(range=(100.0, 300.0)).ranges
        array([100., 200., 300.])
        """
        self._check_axes(bounds)
        data = self.data
        coords = dict(self.coords)
        axis_of = {name: i for i, name in enumerate(self.coords)}
        for name, pair in bounds.items():
            try:
                low, high = pair
            except (TypeError, ValueError):
                raise ConfigurationError(
                    f"Field.window: {name}={pair!r} is not a (lo, hi) pair.",
                    remediation="Pass two bounds, either of which may be None "
                                "to leave that end where it is.") from None
            if (low is not None and high is not None
                    and float(low) > float(high)):
                raise ConfigurationError(
                    f"Field.window: {name}=({low}, {high}) is inverted.",
                    remediation="Give the bounds low end first.")
            axis = coords[name]
            keep = np.ones(axis.size, dtype=bool)
            if low is not None:
                keep &= axis >= float(low)
            if high is not None:
                keep &= axis <= float(high)
            if not keep.any():
                raise ConfigurationError(
                    f"Field.window: {name}=({low}, {high}) keeps no sample of "
                    f"an axis spanning {axis.min():g} to {axis.max():g}.",
                    remediation="Widen the window, or check it is in the "
                                "axis's own units (metres, seconds, Hz).")
            data = np.compress(keep, data, axis=axis_of[name])
            coords[name] = axis[keep]
        # Narrowing an identity-bearing axis narrows the identity with it, as
        # _slice does when it pins one: the lists behind f0 / n_frequencies
        # are what the field HOLDS, not what the run that produced it swept.
        # Left alone, a field windowed to 1400-1600 Hz answers f0 = 1000 —
        # a legal frequency, and the wrong one, with nothing to flag it.
        id_kwargs = self.id_kwargs()
        for name, key in (('frequency', 'frequencies'),
                          ('source_depth', 'source_depths')):
            if name in bounds and id_kwargs.get(key) is not None:
                id_kwargs[key] = np.asarray(coords[name], dtype=float)
        return Field(data=data, coords=coords, pinned=dict(self.pinned),
                     **id_kwargs)

    def shift(self, **offsets) -> "Field":
        """Translate a coordinate axis by a constant. The data is untouched.

        Each kwarg names a coord axis and an offset in that axis's own units.
        Use it to move an origin: a transfer function synthesised against a
        source waveform carries that waveform's own peak offset into its time
        axis, and ``shift(time=-peak)`` puts the emission at ``t=0`` so it
        lines up with a solver that marches from the emission itself.

        Examples
        --------
        >>> import numpy as np
        >>> from uacpy.core.results import Field
        >>> f = Field(data=np.zeros((1, 3)),
        ...           coords={'depth': np.array([10.0]),
        ...                   'time': np.array([0.0, 0.1, 0.2])})
        >>> f.shift(time=-0.1).times
        array([-0.1,  0. ,  0.1])
        """
        self._check_axes(offsets)
        coords = dict(self.coords)
        for name, offset in offsets.items():
            delta = float(offset)
            if not np.isfinite(delta):
                raise ConfigurationError(
                    f"Field.shift: {name}={offset!r} is not finite.",
                    remediation="A non-finite offset would put the whole axis "
                                "at NaN, losing the coordinate entirely.")
            coords[name] = coords[name] + delta
        return Field(data=self.data, coords=coords, pinned=dict(self.pinned),
                     **self.id_kwargs())

    def remove_delay(self, seconds: Optional[float] = None, *,
                     sound_speed: Optional[float] = None) -> "Field":
        """Advance a transfer function: ``H(f) · exp(+2πi·f·τ)``.

        The frequency-domain counterpart of :meth:`shift` — shifting a time
        axis by ``-τ`` and removing a delay ``τ`` from ``H(f)`` are the same
        operation on the two representations.

        Give **either** ``seconds`` (a signed delay; negative *adds* delay) or
        ``sound_speed``, which takes the delay from this field's own range as
        ``r/c``. There is no default: ``r/c`` is the delay worth removing, and
        a Field carries ``r`` but not ``c`` — the sound speed belongs to the
        Environment that produced it, and guessing 1500 m/s would be the
        package inventing a number the caller did not supply.

        With ``sound_speed`` on a field that still has a range axis, **each
        range is advanced by its own** ``r/c`` — the reduced-time convention,
        which lines every trace up on its geometric arrival. On a single range
        (sliced, or pinned by :meth:`at`) that is just the one delay.

        Why it matters: the phase of a delay wraps at ``1/τ`` in frequency, so
        on a grid of spacing ``Δf`` it is unambiguous only for
        ``τ < 1/(2·Δf)``. A 3.3 s travel time on a 1 Hz grid is aliased beyond
        reading, and two models sampled on *different* grids alias differently
        and appear to disagree when they do not. Removing the bulk delay leaves
        the multipath residual, which the grid does resolve.

        The magnitude is untouched — this multiplies by a unit-modulus factor —
        so ``|H|`` and any TL derived from it are unchanged.

        Parameters
        ----------
        seconds : float, optional
            Delay to remove, signed. Mutually exclusive with ``sound_speed``.
        sound_speed : float, optional
            Reference speed in m/s; the delay becomes ``r/c`` from this field's
            own range. Mutually exclusive with ``seconds``.

        Returns
        -------
        Field
            Same coords and identity; only the phase moves.

        Raises
        ------
        ConfigurationError
            Neither or both arguments, no ``frequency`` axis (a time-domain
            field wants :meth:`shift`), real data (no phase to move), no range
            to take ``r/c`` from, or a non-finite / non-positive value.

        Examples
        --------
        >>> import numpy as np
        >>> from uacpy.core.results import Field
        >>> f = np.array([100.0, 200.0])
        >>> H = Field(data=np.exp(-2j * np.pi * f * 0.01).reshape(1, 1, 2),
        ...           coords={'depth': np.array([50.0]),
        ...                   'range': np.array([15.0]), 'frequency': f})
        >>> np.round(np.angle(H.remove_delay(0.01).data).ravel(), 6)
        array([0., 0.])

        The same thing from the geometry, since 15 m at 1500 m/s is 0.01 s:

        >>> np.round(np.angle(H.remove_delay(sound_speed=1500.0).data).ravel(), 6)
        array([0., 0.])
        """
        if (seconds is None) == (sound_speed is None):
            raise ConfigurationError(
                "Field.remove_delay: give exactly one of seconds= or "
                "sound_speed=.",
                remediation="seconds= removes a delay you already know; "
                            "sound_speed= takes it from this field's range as "
                            "r/c.")
        if not self.is_complex:
            raise ConfigurationError(
                "Field.remove_delay: the data is real, so it carries no "
                "phase to move.",
                remediation="Apply it to the complex pressure or transfer "
                            "function, before to_dB() throws the phase away.")

        def _broadcast(name, values):
            """``values`` shaped to broadcast along this field's ``name`` axis."""
            shape = [1] * self.data.ndim
            shape[list(self.coords).index(name)] = np.size(values)
            return np.asarray(values, dtype=float).reshape(shape)

        if 'frequency' in self.coords:
            hertz = _broadcast('frequency', self.coords['frequency'])
        elif 'frequency' in self.pinned:
            # Collapsed by at(frequency=…); the value survives in pinned, so
            # the operation is still well defined — a constant phase.
            hertz = float(self.pinned['frequency'])
        else:
            raise ConfigurationError(
                f"Field.remove_delay: no frequency axis; this field is over "
                f"{list(self.coords)}.",
                remediation="A delay lives in the phase of H(f). For a "
                            "time-domain field, move its axis instead: "
                            "shift(time=-seconds).")

        if seconds is not None:
            delay = float(seconds)
            if not np.isfinite(delay):
                raise ConfigurationError(
                    f"Field.remove_delay: seconds={seconds!r} is not finite.",
                    remediation="Pass the delay to remove in seconds, usually "
                                "the geometric travel time r/c.")
        else:
            speed = float(sound_speed)
            if not np.isfinite(speed) or speed <= 0:
                raise ConfigurationError(
                    f"Field.remove_delay: sound_speed={sound_speed!r} is not a "
                    f"positive, finite speed.",
                    remediation="Pass the reference speed in m/s, e.g. the "
                                "water column's own c.")
            if 'range' in self.coords:
                delay = _broadcast('range', self.coords['range']) / speed
            elif 'range' in self.pinned:
                delay = float(self.pinned['range']) / speed
            else:
                raise ConfigurationError(
                    f"Field.remove_delay: no range to take r/c from; this "
                    f"field is over {list(self.coords)}.",
                    remediation="Pass the delay directly with seconds=.")

        return Field(data=self.data * np.exp(2j * np.pi * hertz * delay),
                     coords=dict(self.coords), pinned=dict(self.pinned),
                     **self.id_kwargs())

    def eval(self, **kwargs) -> "Field":
        """Interpolated slice — the interpolating counterpart of :meth:`at`.

        Each kwarg names a coord axis and a value; the data is interpolated
        along that axis (constant extrapolation past the ends) and the axis
        collapsed into :attr:`pinned`. ``method=`` picks the scheme —
        ``'linear'`` (default), ``'nearest'``, or ``'cubic'``. Use :meth:`at`
        for the nearest stored sample when you must not fabricate values. Note
        that interpolating a real **TL (dB)** field happens in dB and smooths
        sharp interference nulls; slice complex pressure (or use ``at``) for
        null-critical work.
        """
        method = kwargs.pop('method', 'linear')
        self._check_axes(kwargs)
        if method != 'nearest':      # 'nearest' fabricates nothing
            self._warn_if_undersampled('Field.eval', axes=set(kwargs))
        data = self.data
        coords = dict(self.coords)
        pinned = dict(self.pinned)
        order = list(self.coords)
        for name, value in kwargs.items():
            ax = order.index(name)
            data, vq = collapse_axis(data, coords[name], value, method,
                                     axis=ax, name=name)
            pinned[name] = vq
            del coords[name]
            order.remove(name)
        pinned_now = set(kwargs)
        new_frequencies = (
            np.array([pinned['frequency']], dtype=float)
            if 'frequency' in pinned_now else self.frequencies)
        new_source_depths = (
            np.array([pinned['source_depth']], dtype=float)
            if 'source_depth' in pinned_now else self.source_depths)
        id_kwargs = self.id_kwargs()
        id_kwargs['frequencies'] = new_frequencies
        id_kwargs['source_depths'] = new_source_depths
        return Field(data=data, coords=coords, pinned=pinned, **id_kwargs)

    def max(self) -> "Field":
        """Slice at the loudest field point.

        Linear data (``unit='Pa'``): global argmax of ``|data|``.

        Two quantities run backwards, and it takes **both** axes to identify
        them: transmission loss (``kind='pressure'`` in ``unit='dB'``) and
        OASS reverberation are *losses*, so the least of either is the
        loudest. The remaining dB quantities are **levels** — signal excess,
        and any future one — and more of a level is more, so dB alone must
        not decide the direction.

        Reverberation reads as a loss because that is what OASES writes:
        ``-10·log10 E[|p_scat|²]``, from ``CVMAGS`` → ``VALG10`` →
        ``VSMUL(-5E0)`` in ``REVINT`` (``oassun26.f:853-858``, the routine on
        option ``'r'``'s path), which uacpy stores unchanged and tags
        ``oass_quantity='reverberation_loss_dB'``. Read as a level it made
        this method return the *quietest* cell of a reverberation grid.

        ``NaN`` no-data cells (e.g. Bellhop cells no ray reached) are
        excluded. Every axis collapses to a pinned scalar; the returned
        Field has empty :attr:`coords`, 0-D :attr:`data`, and every
        original axis recorded in :attr:`pinned`."""
        if self.data.size == 0:
            raise ConfigurationError(
                f"Field.max: data is empty — coords {list(self.coords)} "
                f"give shape {self.data.shape}. An axis was sliced to "
                f"nothing; widen the .sel/.at selection that produced this "
                f"Field.")
        if self.is_complex:
            strength = np.abs(self.data)  # complex is linear: loudest |p|
        elif self.unit == 'dB' and self.kind in ('pressure', 'reverberation'):
            strength = -np.asarray(self.dB, dtype=float)  # least loss = loudest
        elif self.unit == 'dB':
            strength = np.asarray(self.data, dtype=float)  # a level: more is more
        else:
            strength = np.abs(self.data)
        if not np.isfinite(strength).any():
            raise ConfigurationError(
                "Field.max: no finite samples (all NaN no-data cells)")
        flat = int(np.nanargmax(strength))
        idx = np.unravel_index(flat, self.data.shape)
        idx_map = {name: int(i) for name, i in zip(self.coords, idx)}
        return self._slice(idx_map)

    def _check_axes(self, kwargs: Dict[str, Any]) -> None:
        for name in kwargs:
            if name not in self.coords:
                raise ConfigurationError(
                    f"Field: unknown axis {name!r}; available: "
                    f"{list(self.coords)}"
                )
            if self.coords[name].size == 0:
                raise ConfigurationError(
                    f"Field: axis {name!r} has no samples (size 0) to select "
                    f"from; widen the selection that sliced it to nothing.")

    def _slice(self, idx_map: Dict[str, int]) -> "Field":
        slicers: List[Any] = []
        new_coords: Dict[str, np.ndarray] = {}
        new_pinned: Dict[str, float] = dict(self.pinned)
        for ax_pos, name in enumerate(self.coords):
            if name in idx_map:
                i = idx_map[name]
                size = self.coords[name].size
                if -size <= i < 0:
                    i += size
                if not (0 <= i < size):
                    raise IndexError(
                        f"Field: index {idx_map[name]} out of range for axis "
                        f"{name!r} (size {size})"
                    )
                slicers.append(i)
                new_pinned[name] = float(self.coords[name][i])
            else:
                slicers.append(slice(None))
                new_coords[name] = self.coords[name]
        new_data = self.data[tuple(slicers)]
        # Pinning an identity-bearing axis narrows the identity to the
        # pinned value, so f0 / n_frequencies / repr reflect the slice.
        new_frequencies = (
            np.array([new_pinned['frequency']], dtype=float)
            if 'frequency' in idx_map else self.frequencies
        )
        new_source_depths = (
            np.array([new_pinned['source_depth']], dtype=float)
            if 'source_depth' in idx_map else self.source_depths
        )
        id_kwargs = self.id_kwargs()
        id_kwargs['frequencies'] = new_frequencies
        id_kwargs['source_depths'] = new_source_depths
        return Field(
            data=new_data,
            coords=new_coords,
            pinned=new_pinned,
            **id_kwargs,
        )

    def at_source_level(self, source_level_dB: Optional[float] = None) -> "Field":
        """This field as an absolute received level: ``SL - TL``.

        The propagation term of the sonar equation. Transmission loss is
        referenced to a unit source at 1 m, so a TL field says how much
        quieter a cell is than the source, not how loud it is; naming the
        level the source was driven at turns it into what a hydrophone
        there would measure.

        Reads the loss the field already carries: ``-20·log10|p|`` for
        complex pressure, and the stored numbers themselves for a real dB
        result (Kraken's ``INCOHERENT_TL``, OAST's TL, or an incoherent
        :meth:`ResultStack.superpose`). The result is ``kind='level'``, which
        is what keeps a plotter from captioning it "TL (dB)" and from
        running a 1-D cut through it backwards — a level is not a loss.

        A superposed multi-source field carries its array's gain inside the
        loss (the reference is still one unit source at 1 m), so the level
        this returns is the array's, driven at ``source_level_dB`` per unit
        source. Scale the ``Source`` weights to normalise the array instead.

        Raises :class:`ConfigurationError` for a time-domain trace, which is
        linear pressure rather than a loss — there is nothing to subtract.
        """
        if not _quantities.is_loss(self.kind):
            already = " it is already a level" if self.kind == 'level' else ""
            raise ConfigurationError(
                f"Field.at_source_level: this field's kind is "
                f"{self.kind!r}, which is not a transmission loss, so a "
                f"source level has nothing to subtract from —{already or ' a '
                'residual, a normalised power and a signal excess are all dB '
                'and none of them is a propagation loss'}. Apply the level to "
                f"the loss the run returned, once."
            )
        if 'time' in self.coords:
            raise ConfigurationError(
                "Field.at_source_level: a time-domain trace is linear "
                "pressure, not a transmission loss, so a source level has "
                "nothing to subtract from. Take .extract_tone(f) for a "
                "narrowband field first, or scale .data by the source "
                "amplitude directly."
            )
        if source_level_dB is None:
            source_level_dB = (self.metadata or {}).get('source_level_dB')
        if source_level_dB is None:
            raise ConfigurationError(
                "Field.at_source_level: no source level to apply. Give one "
                "here, or set Source(source_level_dB=...) on the run so "
                "every result it produces carries it."
            )
        sl = float(source_level_dB)
        if not np.isfinite(sl):
            raise ConfigurationError(
                f"Field.at_source_level: source_level_dB must be finite; "
                f"got {source_level_dB!r}.")
        loss = np.asarray(self.dB, dtype=float)
        id_kwargs = self.id_kwargs()
        meta = id_kwargs['metadata']
        meta['kind'] = 'level'
        meta['unit'] = 'dB'
        meta['source_level_dB'] = sl
        return Field(data=sl - loss, coords=self.coords,
                     pinned=dict(self.pinned), **id_kwargs)

    def to_dB(self) -> "Field":
        """Return a real-dB Field via ``-20·log10(|data|)``.

        No-op when ``data`` is already real — including a real field whose
        unit is not dB, which is returned unchanged and whose :attr:`dB` still
        refuses it. There is no linear-to-dB conversion here for real data:
        the sign convention above is the *transmission-loss* one, and applying
        it to an arbitrary real quantity would invent a level the field does
        not carry.

        A ``metadata['unit']`` tag describes the *data*, so it is rewritten
        to ``'dB'`` rather than carried across: the untagged path derives
        ``'dB'`` from the real dtype anyway, and a tag left saying ``'Pa'``
        on dB data sends :meth:`max` down its linear branch, where the
        largest ``|TL|`` is the quietest point rather than the loudest."""
        if not self.is_complex:
            return self
        id_kwargs = self.id_kwargs()
        meta = dict(id_kwargs.get('metadata') or {})
        if 'unit' in meta:
            meta['unit'] = 'dB'
            id_kwargs['metadata'] = meta
        return Field(
            data=_complex_to_dB(self.data),
            coords=dict(self.coords),
            pinned=dict(self.pinned),
            **id_kwargs,
        )

    # ── (depth, range) operations ─────────────────────────────────────

    def mask_below_seafloor(self, bathymetry) -> "Field":
        """Return a copy with samples below the seafloor set to NaN.

        Requires exactly the canonical 2-D layout
        ``coords == {'depth': ..., 'range': ...}``."""
        if list(self.coords) != ['depth', 'range']:
            raise ConfigurationError(
                "Field.mask_below_seafloor: requires canonical "
                f"['depth', 'range'] coords; got {list(self.coords)}"
            )
        if isinstance(bathymetry, Environment):
            bathymetry = bathymetry.bathymetry
        if not isinstance(bathymetry, Bathymetry):
            arr = np.asarray(bathymetry, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ConfigurationError(
                    f"Field.mask_below_seafloor: bathymetry must be shape "
                    f"(N, 2) or an Environment; got array shape {arr.shape}"
                )
            # np.interp below takes its xp on trust: a range column that does
            # not increase interpolates against a broken axis and masks the
            # wrong cells with no error (a two-point profile handed in
            # reversed masked 28 cells where the sorted one masks 24).
            # Bathymetry is where that axis is checked, so the raw array is
            # routed through it rather than checked a second time here.
            bathymetry = Bathymetry.coerce(arr)
        bathy = bathymetry.to_pairs()
        ranges = self.coords['range']
        depths = self.coords['depth']
        seafloor = np.interp(ranges, bathy[:, 0], bathy[:, 1])
        # An inexact payload keeps its width (a .shd-backed float32 result
        # stays float32); only an integer payload, which cannot hold NaN,
        # is widened.
        dtype = (self.data.dtype
                 if np.issubdtype(self.data.dtype, np.inexact) else np.float64)
        new_data = self.data.astype(dtype, copy=True)
        for j, sf in enumerate(seafloor):
            mask = depths > sf
            new_data[mask, j] = np.nan
        return Field(
            data=new_data,
            coords=dict(self.coords),
            pinned=dict(self.pinned),
            **self.id_kwargs(),
        )

    def _warn_if_undersampled(self, where: str, axes=None) -> None:
        """Warn when either axis is too coarse to interpolate coherently.

        Sample spacing against a quarter wavelength, on **both** axes: the
        depth and range biases compound (+2.6 and +1.4 dB alone, +4.9 dB
        together), and `resample_to` always interpolates both. The vertical
        wavenumber never exceeds the total wavenumber, so the same bound is
        conservative on depth — it can warn where depth alone would have been
        adequate, but it cannot stay silent on a grid that is not.

        The wrapped phase step is deliberately **not** used as a fallback:
        it misses 47.8 % of undersampled grids and is blind by construction
        wherever the spacing nears a whole wavelength (``wrap(2*pi*n) = 0``),
        so "sample every wavelength" — maximally aliased — reads as perfect.
        A field with no frequency is reported as unverifiable instead, since
        a silence that reads as a pass is worse than an admission.
        """
        if not self.is_complex:
            return
        # ``source_depth`` is the same physical coordinate as ``depth`` and
        # obeys the same quarter-wavelength condition; scoping the guard to
        # ('depth', 'range') by NAME meant interpolating along it skipped the
        # check entirely. Measured on two source depths 5 m apart at 200 Hz
        # (quarter wavelength 1.875 m): eval(source_depth=...) returned -6.02 dB
        # with the phase inverted and said nothing, while the identical numbers
        # with the axis renamed 'depth' did warn.
        wanted = ('depth', 'range', 'source_depth') if axes is None \
            else tuple(axes)
        axes = [(name, np.asarray(self.coords[name], dtype=float))
                for name in ('depth', 'range', 'source_depth')
                if name in wanted and self.coords.get(name) is not None
                and self.coords[name].size > 1]
        self._warn_if_frequency_axis_undersamples(wanted, where)
        if not axes:
            return
        # The criterion has to be applied at the SHORTEST wavelength the field
        # carries, i.e. its highest frequency: a grid that resolves the bottom
        # of a band aliases the top of it. Taking the first frequency instead
        # made the guard most permissive exactly where the field is most
        # aliased — measured on a 2 m depth grid carrying 100-1000 Hz, eval()
        # was silent while the 1000 Hz slab came back 19.62 dB low with its
        # phase inverted, and the identical grid presented as narrowband at
        # 1000 Hz did warn.
        f_hi = self._highest_frequency()
        if not f_hi:
            warnings.warn(
                f"{where}: this Field carries no frequency, so the "
                f"quarter-wavelength condition that decides whether a coherent "
                f"field may be interpolated cannot be checked. The result may "
                f"carry an unreported level bias; take .dB first if only the "
                f"level is wanted.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
            return
        quarter = DEFAULT_SOUND_SPEED / (4.0 * float(f_hi))
        # |diff|: a descending axis is the same physical grid stored the other
        # way round, and ``eval`` walks it in reverse for the same values, so
        # the spacing it is judged by must not depend on the orientation.
        coarse = [(name, float(np.max(np.abs(np.diff(a))))) for name, a in axes
                  if float(np.max(np.abs(np.diff(a)))) > quarter]
        if not coarse:
            return
        detail = ' and '.join(f"{name} samples are {d:g} m apart"
                              for name, d in coarse)
        warnings.warn(
            f"{where}: {detail}, over the {quarter:.3g} m quarter wavelength at "
            f"{float(f_hi):g} Hz (nominal c={DEFAULT_SOUND_SPEED:g} m/s), so "
            f"interpolating this coherent field cuts across the carrier. It can "
            f"bias the level upward by several dB (measured +2.6 in range, +1.4 "
            f"in depth, +4.9 with both) and it corrupts the phase — the two peak "
            f"at different spacings, so a small level error does not imply a "
            f"usable phase. Re-run the model on the target grid instead, or take "
            f".dB first if only the level is wanted.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)

    def _warn_if_phase_view_aliases(self, where: str) -> None:
        """Warn when a phase-sensitive VIEW of this field is spatially aliased.

        A wrapped phase resolves the carrier only while it turns less than
        half a cycle between neighbouring samples, so the bound here is the
        **half** wavelength, not the quarter wavelength of
        :meth:`_warn_if_undersampled`. The two numbers must not be shared
        because the two guards answer different questions: that one asks
        whether uacpy may interpolate *between* stored samples, where a
        quarter wavelength is what keeps the interpolant off the
        opposite-phase lobe; this one asks whether the stored samples
        themselves resolve the carrier, and that is Nyquist. Between the two
        bounds a phase map is coarse but honest, and warning there would cry
        wolf on grids that draw correctly.

        Nothing is interpolated and no level is biased -- the heatmap draws
        flat cells (``shading='nearest'``) and a line cut joins stored
        samples. What goes wrong is that the picture is a faithful drawing of
        an aliased signal, and an aliased phase reads as smooth large-scale
        structure rather than as noise, so it does not look wrong. Measured on
        the 100 m Pekeris guide at 200 Hz (lambda 7.5 m) over 1000-1300 m: at
        dr = 15 m the panel shows broad diagonal bands that are entirely an
        artefact of the grid, while dr = 0.94 m over the same window shows the
        one-wrap-per-wavelength fringes the field actually has. The two agree
        exactly at the ranges they share (max |dphase| = 0), so the field is
        right and only the view is wrong -- which is why this warns instead of
        raising, and why ``value='dB'`` of the same coarse field is left
        alone: |p| varies on the interference scale, not on the carrier.

        Only the spatial axes are judged. The frequency axis carries the same
        carrier, but against range rather than wavelength, and
        :meth:`plot_transfer_function` draws a phase panel along it on
        purpose; :meth:`_warn_if_frequency_axis_undersamples` is that axis's
        guard and this one must not fire on it.
        """
        if not self.is_complex:
            return
        axes = [(name, np.asarray(self.coords[name], dtype=float))
                for name in ('depth', 'range', 'source_depth')
                if self.coords.get(name) is not None
                and self.coords[name].size > 1]
        if not axes:
            return
        f_hi = self._highest_frequency()
        if not f_hi:
            warnings.warn(
                f"{where}: this Field carries no frequency, so whether its "
                f"spatial grid resolves the carrier cannot be checked. A "
                f"phase view of an undersampled grid draws structure that is "
                f"not in the field; plot value='dB' if only the level is "
                f"wanted.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
            return
        # |diff|: a descending axis is the same physical grid stored the other
        # way round, and it is drawn from the same samples.
        half = DEFAULT_SOUND_SPEED / (2.0 * float(f_hi))
        # ``>=``, not ``>``: at exactly half a wavelength the carrier advances
        # exactly pi between samples, and +pi and -pi are the same wrapped
        # value, so the direction of rotation is already unrecoverable.
        # Nyquist is the first aliased spacing, not the last good one.
        coarse = [(name, float(np.max(np.abs(np.diff(a))))) for name, a in axes
                  if float(np.max(np.abs(np.diff(a)))) >= half]
        if not coarse:
            return
        detail = ' and '.join(f"{name} samples are {d:g} m apart"
                              for name, d in coarse)
        turns = max(d for _, d in coarse) / (2.0 * half)
        warnings.warn(
            f"{where}: {detail}, at or over the {half:.3g} m half "
            f"wavelength at {float(f_hi):g} Hz (nominal "
            f"c={DEFAULT_SOUND_SPEED:g} m/s), so "
            f"the carrier turns up to {turns:.2f} cycles between neighbouring "
            f"samples and this view is aliased. The large-scale pattern it "
            f"draws belongs to the grid, not to the field. Re-run on a grid "
            f"finer than the half wavelength, or plot value='dB' if only the "
            f"level is wanted.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)

    def resample_to(
        self,
        *,
        depths: np.ndarray,
        ranges: np.ndarray,
        method: str = 'linear',
    ) -> "Field":
        """Linearly resample onto a new ``(depth, range)`` grid.

        Requires the canonical 2-D layout ``coords == {'depth', 'range'}``.
        Complex data is interpolated component-wise. Out-of-bound queries
        return NaN.

        Keyword-only and depth-first, like every other axis pair on a
        :class:`Field`: passing the two vectors the other way round would
        otherwise resample onto a transposed grid that is mostly NaN.

        Interpolating a **coherent** field only works while the carrier is
        resolved: samples must be under a quarter wavelength apart **on both
        axes**, or the interpolant cuts across opposite-phase lobes and biases
        the level upward — +2.6 dB from range alone, +1.4 dB from depth alone,
        +4.9 dB from the two together. Both spacings are checked against the
        wavelength at the highest frequency the field carries (its shortest
        wavelength) and a coarse one warns — except under
        ``method='nearest'``, which returns a stored sample and fabricates
        nothing, the same exemption :meth:`eval` makes. Take :attr:`dB` first
        if only the level is wanted; a real field carries no carrier and
        interpolates freely."""
        if list(self.coords) != ['depth', 'range']:
            raise ConfigurationError(
                "Field.resample_to: requires canonical ['depth', 'range'] "
                f"coords; got {list(self.coords)}"
            )
        if method != 'nearest':      # 'nearest' fabricates nothing
            self._warn_if_undersampled('Field.resample_to')
        from scipy.interpolate import RegularGridInterpolator
        new_depths = np.atleast_1d(np.asarray(depths, dtype=float))
        new_ranges = np.atleast_1d(np.asarray(ranges, dtype=float))
        DD, RR = np.meshgrid(new_depths, new_ranges, indexing='ij')
        query = np.stack([DD.ravel(), RR.ravel()], axis=-1)
        if self.is_complex:
            interp_re = RegularGridInterpolator(
                (self.coords['depth'], self.coords['range']), self.data.real,
                method=method, bounds_error=False, fill_value=np.nan,
            )
            interp_im = RegularGridInterpolator(
                (self.coords['depth'], self.coords['range']), self.data.imag,
                method=method, bounds_error=False, fill_value=np.nan,
            )
            vals = interp_re(query) + 1j * interp_im(query)
        else:
            interp = RegularGridInterpolator(
                (self.coords['depth'], self.coords['range']), self.data,
                method=method, bounds_error=False, fill_value=np.nan,
            )
            vals = interp(query)
        new_data = vals.reshape(len(new_depths), len(new_ranges))
        return Field(
            data=new_data,
            coords={'depth': new_depths, 'range': new_ranges},
            pinned=dict(self.pinned),
            **self.id_kwargs(),
        )

    # ── broadband-only (requires 'frequency' coord) ───────────────────

    def to_time_trace(
        self,
        depth: Optional[float] = None,
        range: Optional[float] = None,
        *,
        source_spectrum: Optional[np.ndarray] = None,
        waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        window: str = "hann",
        nfft: Optional[int] = None,
        t_start: Optional[float] = None,
    ) -> "Field":
        """What one signal looks like at one receiver.

        Single-trace IFFT of ``H(d, r, :)`` at a chosen ``(depth, range)``.
        Requires ``coords == {'depth', 'range', 'frequency'}``. Returns
        a single-point ``Field`` with ``coords={'time': ...}``.

        With ``waveform``, this is the received signal: hand it the
        transmitted waveform and the receiver's position and it returns
        ``p(t)`` there::

            trace = H.to_time_trace(depth=50, range=5000,
                                    waveform=chirp, sample_rate=fs)

        With neither ``waveform`` nor ``source_spectrum`` it is the
        band-limited impulse response instead.
        :meth:`synthesize_time_series` does the same convolution for EVERY
        cell at once; use that for a grid, this for a receiver.

        Parameters
        ----------
        depth, range : float, optional
            Cell to synthesise. Matched to the **nearest** stored
            coordinate — never interpolated — and recorded in the returned
            Field's :attr:`pinned`. Defaults are the middle depth
            (``depths[n_d // 2]``) and the first range. A label that is
            given must be a finite scalar, as for :meth:`at`.
        source_spectrum : ndarray, optional
            Continuous source spectrum ``S(f)`` already sampled at
            ``coords['frequency']``. ``None``, with no ``waveform``,
            synthesises the band-limited impulse response.
        waveform : ndarray, optional
            The transmitted signal, in place of ``source_spectrum`` — its
            spectrum is evaluated on this field's axis by
            :func:`_source_spectrum_at`, the exact DTFT. Prefer this:
            resampling an ``rfft`` onto the axis by interpolation is a
            triangular-kernel convolution rather than a resampling. The
            error runs to tens of per cent in ``S(f)`` for ordinary
            waveforms — 43-75 % measured — and reaches the 100 % that
            :func:`_source_spectrum_at` quotes for an unwindowed tone
            sitting on a bin.
            Requires ``sample_rate``. Pass the 1-D signal, not the
            ``(time, signal)`` pair the generators return.
        sample_rate : float, optional
            Rate (Hz) ``waveform`` is sampled at.
        window : str
            Band-edge taper applied to ``H(f)`` before the IFFT: ``'hann'``,
            ``'hamming'``, ``'blackman'``, ``'tukey'`` or ``'none'``.
        nfft : int, optional
            IFFT length. ``None`` sizes it automatically; an explicit value
            that would put the highest data bin at or above Nyquist is
            rejected rather than allowed to alias.
        t_start : float, optional
            Time of the first output sample (s). ``None`` estimates it from
            the range and the fastest sound speed the model reported.

        Warns
        -----
        UserWarning
            When ``depth`` or ``range`` falls outside the grid. The match is
            to the nearest stored coordinate, so a receiver beyond the
            panel's edge silently becomes the edge cell — a trace of the
            wrong place that looks like a trace of the right one."""
        who = "Field.to_time_trace"
        if waveform is not None:
            if source_spectrum is not None:
                raise ConfigurationError(
                    f"{who}: pass either source_spectrum= (already on this "
                    f"field's frequency axis) or waveform= (sampled in "
                    f"time), not both.")
            if sample_rate is None:
                raise ConfigurationError(
                    f"{who}: waveform= needs sample_rate= to have a "
                    f"spectrum at all.")
            if isinstance(waveform, tuple):
                raise ConfigurationError(
                    f"{who}: waveform must be the 1-D signal, not a "
                    f"(time, signal) pair — pass lfm_chirp(...)[1].")
            from uacpy.acoustic_signal.system import (
                waveform_spectrum_at as _source_spectrum_at)
            source_spectrum = _source_spectrum_at(
                waveform, sample_rate,
                np.asarray(self.coords.get('frequency', []), dtype=float))
        for name, label in (('depth', depth), ('range', range)):
            axis = self.coords.get(name)
            if label is None or axis is None or np.size(axis) == 0:
                continue
            lo, hi = float(np.min(axis)), float(np.max(axis))
            if not (lo <= float(label) <= hi):
                warnings.warn(
                    f"{who}: {name}={float(label):g} is outside the grid "
                    f"({lo:g} to {hi:g}); the nearest stored "
                    f"{name} is used instead, so this trace is of a "
                    f"different place than asked for.",
                    UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
        if list(self.coords) != ['depth', 'range', 'frequency']:
            raise ConfigurationError(
                "Field.to_time_trace: requires canonical "
                "['depth', 'range', 'frequency'] coords; got "
                f"{list(self.coords)}"
            )
        return _ifft_to_trace(
            self, depth=depth, range=range,
            source_spectrum=source_spectrum,
            window=window, nfft=nfft, t_start=t_start,
        )

    def truncate_response(self, duration: float, *,
                          origin: Union[str, float] = 'peak',
                          window: str = 'boxcar') -> "Field":
        """``H(f)`` with its impulse response cut to ``duration`` — the
        transfer function a pulse that long actually sees.

        Requires a ``frequency`` axis with a uniform spacing and complex
        data. Every cell is transformed to its own impulse response over the
        band, windowed, and transformed back; the coords, the axis order and
        the identity are unchanged.

        **Why a transfer function has a pulse length in it at all.** ``H(f)``
        as a model returns it is the continuous-wave answer: every path
        present at once, interfering. A pulse of duration ``T`` does not meet
        that channel. Two copies of it interfere only where they overlap, so
        arrivals further apart than ``T`` land as separate, non-interfering
        echoes — "we choose individual arrivals and measure their travel
        times, amplitudes, and waveforms **when the signals are separable in
        the time domain**. If the multiple arrivals are not separable, both
        the phases and amplitudes of the components determine how they
        interfere" (Medwin and Clay, *Fundamentals of Acoustical
        Oceanography*, sect. 3.4.5, "Sum of multiple arrivals"). Jensen et
        al. give the test as an operation rather than a rule: "filter these
        results within a specified bandwidth in order to obtain the pulse
        structure that indicates whether the arrivals are actually separated
        in time" (*Computational Ocean Acoustics*, sect. 2.4.4.1, pointing on
        to sect. 8.3.1). Ainslie names multipath first among the causes of
        coherence loss and prices two replicas in Table 6.9: the input
        carries ``a^2 + b^2`` and the matched filter's output only ``a^2``,
        worst case 3 dB for equal amplitudes (*Sonar Performance Modeling*,
        sect. 6.2.6).

        **The recipe is not new.** Transforming a *windowed* impulse response
        is standard practice, and named: "time windows can also be used in
        separating various components of a transient signal from each other
        ('gating'), say, separating an impulse that is due to a direct sound
        wave from other pulses that are due to reflected waves" (Jacobsen and
        Juhl, *Fundamentals of General Linear Acoustics*, sect. B.3.2 "Time
        Windows"). Room acoustics does this exact thing and calls the result
        the **short-term spectrum** — "a Fourier transform of the first
        64 msec of the impulse response after the direct sound has arrived
        ... windowed using a quarter period cosine squared window ... The
        windowing is necessary to prevent the sudden cutoff of the impulse
        producing spurious effects in the spectrum", with the length set by
        "the integration time of the ear" (Everest and Pohlmann, *Master
        Handbook of Acoustics*, "Prediction of room response"). Everything
        here is that, with the pulse length in place of the ear's
        integration time, and ``window='hann'`` in place of the cosine
        squared and for the reason they give.

        The equivalence to smoothing ``H`` is the convolution theorem, and
        the taper's price is its own main lobe: "applying a window ``w[n]``
        to a signal ``x[n]`` is the same as convolving the Fourier transform
        of the window ``W`` with the signal's Fourier transform ``X`` ...
        de-emphasizing the data near the window edges has the effect of
        shortening the RMS duration and therefore broadening the RMS
        bandwidth" (Abraham, sect. 4.10 "Windowing and window functions") —
        which is why ``'hann'`` removes ``'boxcar'``'s skirt and attenuates a
        path part-way out instead. And the record must hold the response
        before any of this means anything: it "must be selected large enough
        that it contains the entire transient response at each receiver so as
        to eliminate the aliasing" (Jensen et al., sect. 8.2.1.3 "Time
        windowing and sampling"), which is what the warning below measures.

        **How this relates to propagation loss.** Abraham defines the
        propagation loss of a pulse twice over (sect. 3.2.4.2 "Propagation
        loss and the channel frequency response"), and the two definitions
        part company exactly here. In time, ``L_p`` is the source's
        mean-square over the pulse divided by the received mean-square over a
        window of the SAME duration ``T`` — energy landing outside the window
        is not counted. In frequency, by Parseval, ``L_p = int |U_o|^2 df /
        int |H U_o|^2 df``, which runs over ALL time and counts every path.
        Parseval needs the whole signal, so the two agree only while the
        received pulse fits in the window.

        Measured on two paths of amplitude 1 and 0.7 with a 20 ms burst, the
        gap varied (decibels, less is more loss; the ``T``-wide gate is shown
        anchored both on the onset and on the peak, which changes nothing).
        **The band, ``df`` and taper of this run were not recorded with it**,
        so the table is indicative of the ORDERING and the crossover, which
        are what the prose below draws on, and is not reproducible value by
        value; no test pins it.

        =======  ==========  ==========  =========  =======  =====
        gap      freq form   gate@onset  gate@peak  boxcar   hann
        =======  ==========  ==========  =========  =======  =====
        5 ms     -3.813      -3.796      -3.808     -3.813   -3.301
        15 ms    -1.748      -0.152      -0.112     -1.747   -0.035
        30 ms    -1.715      +0.017      +0.017     +0.017   +0.017
        =======  ==========  ==========  =========  =======  =====

        ``window='boxcar'`` reproduces the frequency form **exactly** while
        the copies can overlap, and the gate **exactly** once they cannot —
        it is the criterion the other two only bracket. The frequency form
        never discards: at 30 ms it still adds the late path's energy, and
        ``10*log10(1 + 0.7^2)`` = 1.73 dB is the whole of its -1.715. It
        cannot tell "interferes" from "arrives separately", because averaging
        the fringe away under ``|U_o|^2`` leaves the energy behind. The
        ``T``-wide gate discards from about half a duration out, being ``T``
        wide where overlap needs ``2T``.

        ``'hann'`` costs 0.5 dB on a path a quarter of the way out and 1.7 dB
        at three quarters, and nothing at all once a path is beyond. On a
        channel whose paths sit part-way out, prefer the rectangle and pay
        its skirt.

        **Which duration.** Ainslie's table is for a separation *small*
        against the transmitted pulse ``T`` and *large* against the
        compressed one ``1/B``, so what a receiver resolves is ``1/B`` and
        not ``T``. For a pulse with ``BT`` near 1 — a plain tone burst, an
        unshaped symbol — the two coincide and ``duration = T`` is right.
        For a chirp or any waveform the receiver compresses, ``BT >> 1`` and
        the separable unit is ``1/B``: pass that instead, or this removes
        paths the receiver would still have resolved.

        In the frequency domain it is the same statement: a signal of
        duration ``T`` has ``1/T`` of spectral resolution, so structure in
        ``H`` finer than ``1/T`` — which is what an arrival ``T`` or more
        late puts there — is not something it can see. This method removes
        exactly that structure.

        **What it is not.** With paths in hand the exact receiver-side answer
        is :meth:`~uacpy.core.results.Arrivals.channel_taps`, which applies
        the receiver's own pulse at its decision instants and returns the
        far echoes as separate taps rather than discarding them. This is the
        version for a model that has no paths — a wave model returns a field,
        and the only way to ask it which arrivals are separable is to look at
        its response.

        **Two copies overlap when their delays differ by less than the pulse
        length**, so the window reaches ``duration`` EITHER SIDE of the
        origin — it is ``2 * duration`` wide. A path further out than that
        cannot overlap the one at the origin however they are aligned.

        **The band sets a floor on this.** A band ``B`` wide localises a path
        no better than ``1/B``, so each arrival appears in the response as a
        kernel that wide with skirts around it, and a window cannot separate
        two arrivals closer than that however short it is. That is the
        temporal resolution cell, not a defect of the cut: refine the band,
        not the window.

        Parameters
        ----------
        duration : float
            Pulse length in seconds. The window reaches this far either side
            of the origin, so ``2 * duration`` must fit inside ``1/df``.
        origin : {'peak'} or float, default 'peak'
            Where the window is centred. ``'peak'`` puts it on each cell's
            own ``argmax |h|``, which is the copy a receiver synchronises
            to. A float is a time in seconds from the start of the ``1/df``
            record, shared by every cell.
        window : {'boxcar', 'hann'}, default 'boxcar'
            ``'boxcar'`` is the separability criterion stated plainly —
            inside interferes, outside does not — and puts the rectangle's
            own sinc skirt on ``H``. ``'hann'`` tapers the cut instead,
            trading a wider effective window for a skirt 31 dB down.

        Returns
        -------
        Field
            Same coords and identity; only ``data`` changes.

        Warns
        -----
        UserWarning
            When the response has not decayed by the ends of the ``1/df``
            record. The record is one period of a DFT, so an arrival later
            than ``1/df`` is not absent from it — it has folded back onto the
            early part, where no window can tell it from an early arrival.
            Refine the frequency grid until the ends are quiet.

        Raises
        ------
        ConfigurationError
            No ``frequency`` axis, a non-uniform one, fewer than two
            frequencies, real data, an unknown ``window``, a non-finite or
            out-of-range ``duration``, or an ``origin`` outside the record.
        """
        who = "Field.truncate_response"
        if 'frequency' not in self.coords:
            raise ConfigurationError(
                f"{who}: needs a frequency axis — an impulse response is "
                f"what a transfer function has, and a time-domain field is "
                f"already one (use .window(time=...)). Axes: "
                f"{list(self.coords)}.")
        if not self.is_complex:
            raise ConfigurationError(
                f"{who}: needs complex H(f). A dB or TL field has no phase, "
                f"so it has no impulse response to cut.")
        freqs = np.asarray(self.coords['frequency'], dtype=float)
        axis = list(self.coords).index('frequency')
        # The whole cut — the two transforms, the circular window, the
        # origin handling and the fold warning — is gate_transfer_function's.
        # What this method adds is the coord check above and the re-wrap.
        # Deferred: acoustic_signal pulls scipy, and uacpy's public
        # surface is imported without it (test_lazy_imports).
        from uacpy.acoustic_signal.system import gate_transfer_function
        out = gate_transfer_function(
            self.data, freqs, duration, origin=origin, window=window,
            axis=axis, who=who)
        return Field(data=out,

                     coords=dict(self.coords), pinned=dict(self.pinned),
                     **self.id_kwargs())

    def broadband_loss(self, spectrum=None, *, waveform=None,
                       sample_rate=None) -> "Field":
        """Propagation loss averaged over this field's band — the level a
        signal of finite bandwidth reaches, as a map.

        The transmission loss a model returns is the **continuous-wave**
        answer: one frequency, every path interfering at once. A signal with
        bandwidth does not see that. It sees the band average, and the
        quantity has a name — Ainslie defines **broadband propagation loss**
        for a white source as the frequency average of the coherent
        propagation factor over the sonar bandwidth ``B`` about its centre
        ``f_m`` (*Sonar Performance Modeling*, sect. 11.3.3, Eq. 11.46), and
        gives the coloured-spectrum generalisation in its footnote 13.
        Abraham writes the same ratio as the propagation loss of a pulse,
        ``L_p = int |U_o|^2 df / int |H|^2 |U_o|^2 df`` (sect. 3.2.4.2), and
        Ainslie again as the energy propagation factor behind total path
        loss (sect. 3.3.2.1). They are one formula, which is what this
        computes::

            10 log10( sum_f w(f) / sum_f w(f) |H(f)|^2 ),  w = |spectrum|^2

        **Why not just run an incoherent model.** Because that is the
        approximation, not the quantity. The KRAKEN manual offers it as one
        — "if one is comparing to measured data which has been taken by
        averaging over frequency one can often simulate the resulting
        smoothed result by an incoherent TL" — and Ainslie's Eq. 11.47 is
        the same step, dropping the relative phase of the ray arrivals. He
        marks where it fails: the step "neglects coherent interference
        effects such as cancellation between direct and surface-reflected
        paths. This coherent effect is not negligible, even for incoherent
        broadband processing, if the distance between the sonar (or target)
        and the sea surface is a few wavelengths or less at the center
        frequency, in which case use of Equation (11.46) is required."
        A Lloyd mirror is exactly that case. This is Eq. 11.46, so it keeps
        the interference the band is too narrow to wash out and averages
        away only what the band can reach — and it runs on any model that
        returns ``H(f)``, where ``RunMode.INCOHERENT_TL`` is Bellhop's.

        Jensen's **semicoherent** loss (sect. 3.3.5.4) is a third thing
        again: a shading function applied to an incoherent sum, which he
        introduces as one of a "variety of techniques" that "all tend to be
        somewhat informal and partially empirically based". Prefer this
        where the band is known.

        **What it does not do** is gate. Averaging over the band removes the
        *interference* between paths further apart than ``1/B``, and leaves
        their energy in the sum — Ainslie's energy definition takes "time
        intervals chosen to contain the whole of the transmitted pulse"
        (sect. 3.3.2.1). Discarding a late path instead is
        :meth:`truncate_response`, which answers a receiver-side question
        about one cell, not a propagation quantity over a grid.

        **Where the field comes from.** This needs a complex ``H(f)`` over a
        ``frequency`` axis, which is what ``RunMode.BROADBAND`` returns —
        Bellhop, Kraken, Scooter and RAM all offer it, and ``COHERENT_TL``
        takes a single source frequency by construction::

            f = np.arange(800.0, 1201.0, 20.0)
            H = model.run(env, Source(depths=20.0, frequencies=f), rcv,
                          run_mode=RunMode.BROADBAND, frequencies=f)

        Parameters
        ----------
        spectrum : array_like, optional
            The source's spectrum already sampled on this field's
            ``frequency`` axis, real or complex; only ``|spectrum|^2`` is
            used, and any overall scale cancels in the ratio. ``None`` with
            no ``waveform`` weights the band uniformly, which is Eq. 11.46's
            white source.
        waveform : array_like, optional
            A transmitted waveform, in place of ``spectrum`` — the loss for
            THAT signal, whatever it is. Its spectrum is evaluated on this
            field's axis by :func:`_source_spectrum_at`, the same DTFT
            ``synthesize_time_series`` uses, so the answer satisfies
            ``SEL = ESL - loss`` exactly. Do not pre-interpolate an ``rfft``
            onto the axis and pass it as ``spectrum`` instead: the two grids
            rarely coincide, and interpolation is a triangular-kernel
            convolution rather than a resampling — tens of per cent of error
            in ``S(f)`` for ordinary waveforms (43-75 % measured), reaching
            the 100 % :func:`_source_spectrum_at` quotes for an unwindowed
            tone on a bin. Requires ``sample_rate``.
        sample_rate : float, optional
            Rate (Hz) ``waveform`` is sampled at.

        Returns
        -------
        Field
            The loss in dB, ``kind='pressure'`` and ``unit='dB'`` so
            :attr:`tl` and :meth:`plot` treat it as the transmission-loss
            map it is. The ``frequency`` axis is gone; the band it averaged
            is kept in ``metadata['band_hz']`` as ``(first, last)`` — the
            identity narrows to a single value, so ``.frequencies`` is NOT
            where to look for it — and :attr:`pinned['frequency']` records
            the spectrum-weighted centroid of the axis's samples — the
            signal's own centre for a ``waveform`` (a 500 Hz burst over a
            25 Hz-4 kHz axis pins 500 Hz, not the axis's 2 kHz midpoint,
            which is what a plotter would otherwise caption the map with),
            and the mean of the samples for a white source, which is the
            band centre on the uniform axes these runs produce.
            Cells no path reached stay ``NaN``.

        Raises
        ------
        ConfigurationError
            No ``frequency`` axis, real data, a ``spectrum`` whose length is
            not the frequency axis's, a non-finite or zero-energy
            ``spectrum``, both ``spectrum`` and ``waveform``, or a
            ``waveform`` without a ``sample_rate``.
        """
        who = "Field.broadband_loss"
        if 'frequency' not in self.coords:
            raise ConfigurationError(
                f"{who}: needs a frequency axis to average over; a "
                f"single-frequency field is already the continuous-wave "
                f"answer. Axes: {list(self.coords)}. Run the model with "
                f"run_mode=RunMode.BROADBAND and a frequencies= grid "
                f"(Bellhop, Kraken, Scooter and RAM all offer it); "
                f"COHERENT_TL takes one frequency by construction.")
        if not self.is_complex:
            raise ConfigurationError(
                f"{who}: needs complex H(f). A dB or TL field has lost the "
                f"phase, so averaging it is an incoherent sum and not "
                f"Eq. 11.46 — see the note on RunMode.INCOHERENT_TL.")
        freqs = np.asarray(self.coords['frequency'], dtype=float)
        axis = list(self.coords).index('frequency')
        if waveform is not None:
            if spectrum is not None:
                raise ConfigurationError(
                    f"{who}: pass either spectrum= (already on this field's "
                    f"frequency axis) or waveform= (sampled in time), not "
                    f"both — they are two spellings of one weight.")
            if sample_rate is None:
                raise ConfigurationError(
                    f"{who}: waveform= needs sample_rate= to have a "
                    f"spectrum at all.")
            if isinstance(waveform, tuple):
                raise ConfigurationError(
                    f"{who}: waveform must be the 1-D signal, not a "
                    f"(time, signal) pair — pass tone_burst(...)[1] (the "
                    f"generators return both).")
            from uacpy.acoustic_signal.system import (
                waveform_spectrum_at as _source_spectrum_at)
            spectrum = _source_spectrum_at(waveform, sample_rate, freqs)
        if spectrum is None:
            weights = np.ones(freqs.size, dtype=float)
        else:
            supplied = np.asarray(spectrum)
            if supplied.ndim > 1:
                # Checked on SHAPE before ravel(), which otherwise turns a
                # (3, 4) array into a legal-looking 12-sample weight on a
                # 12-frequency axis and returns the same number as the
                # (12,) case — a wrong spectrum that cannot be told from a
                # right one by its answer.
                raise ConfigurationError(
                    f"{who}: spectrum must be 1-D, one weight per frequency; "
                    f"got shape {supplied.shape}. Flattening it would pair "
                    f"weights with frequencies in an order you did not "
                    f"choose.")
            weights = np.abs(supplied).astype(float).ravel() ** 2
            if weights.size != freqs.size:
                raise ConfigurationError(
                    f"{who}: spectrum has {weights.size} samples but the "
                    f"frequency axis has {freqs.size}; it is the source "
                    f"spectrum ON this field's grid.")
            if not np.all(np.isfinite(weights)):
                raise ConfigurationError(
                    f"{who}: spectrum must be finite.")
            if weights.sum() <= 0.0:
                raise ConfigurationError(
                    f"{who}: spectrum carries no energy, so the weighted "
                    f"average is undefined.")
        # The weighted average and its dB step are
        # broadband_propagation_loss's, including the accumulate-per-map
        # memory behaviour; what this method adds is turning a waveform
        # into w(f), the weighted centroid pin and the band metadata below.
        # Deferred: acoustic_signal pulls scipy, and uacpy's public
        # surface is imported without it (test_lazy_imports).
        from uacpy.acoustic_signal.system import broadband_propagation_loss
        loss = broadband_propagation_loss(self.data, weights, axis=axis,
                                          who=who)
        coords = {name: v for name, v in self.coords.items()
                  if name != 'frequency'}
        pinned = dict(self.pinned)
        # The WEIGHTED centroid, not the axis midpoint: a 500 Hz burst
        # weighted onto a 25 Hz-4 kHz axis is a map of 500 Hz, and labelling
        # it 2 kHz is the same defect Field.window's identity narrowing
        # exists to prevent — a legal frequency, and the wrong one. Uniform
        # weights still give the band centre.
        pinned['frequency'] = float(np.sum(weights * freqs) / weights.sum())
        meta = dict(self.metadata or {})
        meta.update({'kind': 'pressure', 'unit': 'dB'})
        # What was averaged, kept. The identity narrows to the pinned
        # centroid below, so without this a 50 Hz average and a 2 kHz one
        # over the same centre are indistinguishable afterwards — and a
        # plotter captioning the map has only the single frequency, which
        # the map is not.
        meta['band_hz'] = (float(freqs[0]), float(freqs[-1]))
        id_kwargs = self.id_kwargs()
        id_kwargs['metadata'] = meta
        # Collapsing the axis narrows the identity to the value pinned for
        # it, exactly as _slice does when it pins one. Left alone, a 500 Hz
        # burst's map pinned at 500 Hz still reprs as "f=25 Hz" — the band's
        # first sample — which is the disagreement Field.window's own
        # narrowing exists to prevent.
        id_kwargs['frequencies'] = np.array([pinned['frequency']], dtype=float)
        return Field(data=loss, coords=coords, pinned=pinned, **id_kwargs)

    def synthesize_time_series(
        self,
        source_waveform: np.ndarray,
        sample_rate: float,
        *,
        t_start: Optional[float] = None,
        window: str = "hann",
        nfft: Optional[int] = None,
    ) -> "Field":
        """Convolve every grid trace with ``source_waveform`` to obtain a
        time-domain Field shaped ``(n_d, n_r, n_t)``.

        Requires ``coords == {'depth', 'range', 'frequency'}``.

        The record is ``1/Δf`` long and circular: an arrival later than
        that after the record's start lands back on its early part, at a
        fixed place, with nothing at the record's end to show for it — so
        whether the grid holds the channel is knowable only from the
        arrivals (:meth:`Arrivals.synthesis_band`, or the notice a Bellhop
        BROADBAND run gives on its default grid). This checks only that the
        record holds the source pulse.

        Parameters
        ----------
        source_waveform : ndarray
            The 1-D signal only — the waveform generators return a
            ``(time, signal)`` pair, so pass ``lfm_chirp(...)[1]``.
        sample_rate : float
            Sample rate (Hz) the waveform is sampled at. It sets the
            spectrum ``S(f)`` and the *lower bound* on the output rate; the
            realised rate is ``nfft·Δf`` (see :func:`_synthesize_time_series`).
        t_start : float, optional
            Start of the single time window every cell shares. ``None``
            anchors it on the nearest cell (``depths[0]``, ``ranges[0]``).
        window, nfft
            As on :meth:`to_time_trace`, applied to every cell."""
        if list(self.coords) != ['depth', 'range', 'frequency']:
            raise ConfigurationError(
                "Field.synthesize_time_series: requires canonical "
                "['depth', 'range', 'frequency'] coords; got "
                f"{list(self.coords)}"
            )
        # Waveform generators (lfm_chirp, tone_burst, …) return a (t, x) pair;
        # passing the whole pair would silently flatten/misuse it. Catch the
        # common mistake with a clear hint.
        if isinstance(source_waveform, tuple) or (
            np.ndim(source_waveform) == 2 and 2 in np.shape(source_waveform)
        ):
            raise ConfigurationError(
                "Field.synthesize_time_series: source_waveform must be the 1-D "
                "waveform array, not a (time, signal) pair — pass the signal "
                "only, e.g. lfm_chirp(...)[1] (the generators return (time, signal))."
            )
        return _synthesize_time_series(
            self,
            source_waveform=source_waveform,
            sample_rate=sample_rate,
            t_start=t_start, window=window, nfft=nfft,
        )

    def sound_exposure_level(
        self,
        source_waveform: np.ndarray,
        sample_rate: float,
        *,
        reference: float = REFERENCE_PRESSURE_WATER,
        window: str = "none",
        nfft: Optional[int] = None,
        t_start: Optional[float] = None,
    ) -> "Field":
        """Sound exposure level of one transmission of ``source_waveform``,
        cell by cell — the energy a transient delivers, as a map.

        A pulse's currency is energy, not mean-square pressure. Abraham
        introduces it for exactly this population — "many acoustic signals
        are short duration and have varying amplitudes (e.g., a marine
        mammal acoustic emissions, an active sonar echo, or a communications
        packet). In practice such signals are called transient signals;
        however, in a mathematical sense they are energy signals because
        their total energy is finite" — and defines the energy flux density
        as ``(1/rho c) int p^2 dt`` (*Underwater Acoustic Signal
        Processing*, sect. 3.2.1.5). This returns the time integral itself,
        in dB, which is **sound exposure level** as ISO 18405 defines it and
        as the marine-mammal exposure criteria are written in (Southall et
        al. 2019, where it is the weighted metric paired with peak sound
        pressure level). Ainslie builds the active sonar equation on the
        same integral, as the energy propagation factor behind total path
        loss and energy source level (*Sonar Performance Modeling*,
        sect. 3.3.2.1).

        Each cell's trace is synthesised by
        :meth:`synthesize_time_series` and integrated::

            SEL = 10 log10( sum_t p(t)^2 * dt / reference^2 )

        The integral runs over the **whole** record, which is what the
        definition asks for: Ainslie's time intervals are "chosen to contain
        the whole of the transmitted pulse". Nothing is gated away, so a
        late arrival contributes its energy even where it is too late to
        interfere. Whether it interferes is :meth:`broadband_loss`'s
        question, and whether a receiver that opens for ``T`` would see it
        at all is :meth:`truncate_response`'s.

        **A fold corrupts this, and neither method can tell you.**
        Sampling ``H(f)`` every ``df`` periodises the impulse response at
        ``1/df``, so an arrival later than that lands back on the early part
        and adds **coherently** to what is there. Parseval preserves the
        energy of the *aliased* record, which is not the energy of the true
        response, and the cross term is signed: measured over 400 wrapped
        delays on a two-path channel the error ran from **-9.27 dB to
        +2.75 dB**, exceeding half a decibel in 42.8 % of cases, with no
        warning in any of them. :meth:`broadband_loss` reads the same
        undersampled ``H`` and carries the identical bias, so
        ``SEL = ESL - TPL`` still closes while both sides are wrong
        together — that identity pins the two routes' consistency, not
        either one's correctness.

        This is Jensen et al.'s aliasing term: the time window "must be
        selected large enough that it contains the entire transient response
        at each receiver so as to eliminate the aliasing", and the duration
        "is not only controlled by the source signal, but also by the
        dispersive nature of the waveguide" (*Computational Ocean
        Acoustics*, sect. 8.2.1.3) — which is why a check against the pulse
        length cannot stand in for one against the channel.

        It cannot be detected from ``H(f)``: a folded arrival sitting on top
        of the direct one is signature-identical to a clean single arrival.
        The grid has to be sized before the run, from the arrivals, which is
        what :meth:`~uacpy.core.results.Arrivals.synthesis_band` is for.

        **The record must beat twice the delay spread, not once.** Keep
        ``dtau * df < 0.5`` for a path pair ``dtau`` apart — a record longer
        than ``2 * dtau`` — not merely long enough to hold the arrivals.
        Validated on two band configurations: inside the rule the error
        stayed within 0.034 dB over 25 Hz-4 kHz and 0.111 dB over
        900-1100 Hz, against 0.81 / 2.86 dB beyond it and -9.29 dB at the
        worst point found. It bounds the error rather than removing it; on a
        narrow band a tenth of a decibel survives.

        The size of the error beyond the rule is **not** a function of
        ``dtau * df`` alone — it also turns on ``frac(f0 * dtau)``, the
        fringe's phase at the band's first sample. Holding the product at
        1.5 and moving only the band start gave 0.000002, 0.049917 and
        0.028191 dB for ``f0`` = 25, 40 and 55 Hz. So no spot check settles
        it: a single delay on a single axis can land anywhere from a null to
        the maximum.

        **There is no source-level argument: the level rides on the
        waveform's amplitude.** ``synthesize_time_series`` reproduces the
        source waveform where ``H`` is unity, so ``source_waveform`` is the
        source's pressure at the range this field's ``H`` is referenced to —
        1 m for a transmission-loss field. Scale it and the map scales with
        it. For a source level ``SL`` in dB re 1 µPa at 1 m, the waveform's
        rms pressure is ``1e-6 * 10**(SL/20)`` Pa::

            unit = waveform / np.sqrt(np.mean(waveform ** 2))
            sel = H.sound_exposure_level(unit * 1e-6 * 10 ** (SL / 20), fs)

        Passing the unit waveform instead returns the propagation term
        alone, and the source level goes on afterwards as an **energy**
        source level, ``ESL = SL + 10 log10(T)`` for constant power over the
        pulse (Ainslie Eq. 3.155) — the two routes agree exactly. Worked
        through: ``SL`` = 190 dB re 1 µPa at 1 m, a 10 ms burst, 100 m of
        spherical spreading. Received level is 190 - 40 = 150 dB re 1 µPa,
        so the exposure is 150 + 10 log10(0.01) = **130 dB re 1 µPa²·s**,
        which is what both routes return.

        Parameters
        ----------
        source_waveform : ndarray
            The 1-D transmitted waveform; the generators return a
            ``(time, signal)`` pair, so pass ``tone_burst(...)[1]``.
        sample_rate : float
            Rate (Hz) the waveform is sampled at.
        reference : float, default 1e-6
            Reference pressure (Pa). The exposure reference is its square,
            so the default gives dB re 1 µPa²·s.
        window : str, default 'none'
            Band taper, passed to :meth:`synthesize_time_series`, which
            tapers with ``'hann'`` when nothing says otherwise. A flat band
            is what this method asks for, because a taper removes energy
            the integral is defined to
            count, and how much it removes depends on where in the band the
            signal sits rather than on anything physical: a 5-cycle 500 Hz
            burst synthesised over a 25 Hz-4 kHz band reads 52.675 dB flat
            — exact against ``10 log10(int u^2 dt / r^2 / ref^2)`` — and
            35.676 dB under ``'hann'``, 17.0 dB light, because 500 Hz sits
            an eighth of the way up the band where the taper stands at
            0.135. A taper shapes a waveform; it must not set an energy.
        nfft, t_start
            Passed to :meth:`synthesize_time_series`.

        Returns
        -------
        Field
            ``(depth, range)`` in dB, ``kind='sound_exposure'`` — a level,
            so it reads upward and never shares a colorbar with a TL map.
            Cells no path reached stay ``NaN``.

        Raises
        ------
        ConfigurationError
            Real data (a dB or TL field), non-canonical coords
            (:meth:`synthesize_time_series` requires
            ``['depth', 'range', 'frequency']``), or a non-positive
            ``reference``.

        Warns
        -----
        UserWarning
            Through :meth:`synthesize_time_series`, when the ``1/df`` record
            cannot hold the **pulse**. Nothing warns about the **channel**
            outlasting it, and that gap is real — see the note above on
            folds.
        """
        who = "Field.sound_exposure_level"
        if not self.is_complex:
            raise ConfigurationError(
                f"{who}: needs complex H(f). A dB or TL field has no phase, "
                f"so it has no impulse response to integrate — synthesising "
                f"one inverse-transforms the decibels AS pressure and "
                f"overstates the level by tens of dB, silently. Start from "
                f"the BROADBAND field; for an absolute level, scale the "
                f"waveform to the source level rather than calling "
                f"at_source_level() first (see above).")
        if not np.isfinite(reference) or reference <= 0.0:
            raise ConfigurationError(
                f"{who}: reference must be a positive pressure in Pa; got "
                f"{reference!r}.")
        traces = self.synthesize_time_series(
            source_waveform, sample_rate,
            window=window, nfft=nfft, t_start=t_start)
        # The synthesised trace is a real pressure history; taking .real of a
        # complex-typed one drops a numerically-zero imaginary part rather
        # than an analytic signal's quadrature, which would double the energy.
        pressure = np.asarray(traces.data)
        pressure = pressure.real if np.iscomplexobj(pressure) else pressure
        # The integral and its dB step are acoustics.sound_exposure_level's;
        # what this method adds is the synthesis above and the band metadata
        # below. It floors a silent cell at -180 dB rather than -inf, which
        # would poison any mean taken over the map.
        sel = _sound_exposure_level(pressure, traces.dt, reference)
        coords = {name: v for name, v in traces.coords.items()
                  if name != 'time'}
        meta = dict(self.metadata or {})
        meta.update({'kind': 'sound_exposure', 'unit': 'dB'})
        # Recorded here for the same reason as in :meth:`broadband_loss`:
        # this collapses a band and narrows the identity to one centroid, so
        # without it two exposure maps over different bands about the same
        # centre are indistinguishable afterwards.
        meta['band_hz'] = (float(np.asarray(self.coords['frequency'])[0]),
                           float(np.asarray(self.coords['frequency'])[-1]))
        id_kwargs = self.id_kwargs()
        id_kwargs['metadata'] = meta
        # The frequency axis collapses here too, so it is pinned and the
        # identity narrowed the same way :meth:`broadband_loss` does —
        # otherwise the two reducers disagree on the same input, one
        # reprising the signal's centre and the other the band's first
        # sample. The centroid is the waveform's, weighted by its own
        # spectrum on this axis.
        band = np.asarray(self.coords['frequency'], dtype=float)
        from uacpy.acoustic_signal.system import (
            waveform_spectrum_at as _source_spectrum_at)
        weights = np.abs(_source_spectrum_at(
            source_waveform, sample_rate, band)) ** 2
        centre = (float(np.sum(weights * band) / weights.sum())
                  if weights.sum() > 0 else float(np.mean(band)))
        pinned = dict(self.pinned)
        pinned['frequency'] = centre
        id_kwargs['frequencies'] = np.array([centre], dtype=float)
        return Field(data=sel, coords=coords, pinned=pinned, **id_kwargs)

    def peak_sound_pressure_level(
        self,
        source_waveform: np.ndarray,
        sample_rate: float,
        *,
        reference: float = REFERENCE_PRESSURE_WATER,
        window: str = "none",
        nfft: Optional[int] = None,
        t_start: Optional[float] = None,
    ) -> "Field":
        """Peak sound pressure level of one transmission, cell by cell —
        the other half of the impulsive dual metric.

        ``20 log10( max|p(t)| / reference )`` on each cell's synthesised
        trace. Exposure criteria for impulsive sound are stated as a PAIR —
        "frequency-weighted sound exposure level (SEL) and unweighted peak
        sound pressure level", with "exceeding either threshold by the
        specified level ... sufficient to result in the predicted TTS or
        PTS" (Southall et al., *Marine Mammal Noise Exposure Criteria*,
        2019). :meth:`sound_exposure_level` is the first half; this is the
        second, and neither substitutes for the other: SEL integrates the
        whole transmission while this reads its single loudest excursion.

        **It is not recoverable from any band average.** A peak is a
        property of the waveform in time, so no reduction of ``|H(f)|``
        yields it — which is why the criteria name both metrics rather than
        one, and why this needs the synthesis that
        :meth:`broadband_loss` does not.

        **Unlike SEL, it depends on the output sample rate.** A maximum is
        a sample, not an integral, so a coarse grid can miss the true crest
        between samples. Measured on a two-path channel with a 5-cycle
        500 Hz burst, sweeping ``nfft`` from 320 to 65536 (8 kHz to 1.6 MHz
        of output rate): the peak moved **0.0615 dB** and the SEL of the
        same traces moved **0.0000 dB**, Parseval holding it exactly. The
        peak is converged by about ``nfft`` = 4096; the automatic size lands
        within 0.004 dB of that here. Raise ``nfft`` if a fraction of a
        decibel matters, and note the cost is only in the transform length.

        ``window`` defaults to ``'none'`` for the same reason it does on
        :meth:`sound_exposure_level`, and more sharply: a band taper
        reshapes the waveform, and a peak is exactly the part of a waveform
        a taper moves.

        **The source level rides on the waveform's amplitude**, as it does
        for SEL — see that method. Scale the waveform; do not call
        :meth:`at_source_level` first, which this refuses.

        Parameters
        ----------
        source_waveform : ndarray
            The 1-D transmitted waveform; the generators return a
            ``(time, signal)`` pair, so pass ``tone_burst(...)[1]``.
        sample_rate : float
            Rate (Hz) ``source_waveform`` is sampled at.
        reference : float, default 1e-6
            Reference pressure (Pa); the default gives dB re 1 µPa.
        window, nfft, t_start
            Passed to :meth:`synthesize_time_series`.

        Returns
        -------
        Field
            ``(depth, range)`` in dB, ``kind='peak_pressure'`` — a level,
            so it reads upward and never shares a colorbar with a TL map,
            and distinct from ``'level'`` because a peak map and a
            mean-square received-level map are both dB re 1 µPa and are not
            the same reading. The band it collapsed is kept in
            ``metadata['band_hz']``.

        Raises
        ------
        ConfigurationError
            Real data (a dB or TL field), non-canonical coords, or a
            non-positive ``reference``.
        """
        who = "Field.peak_sound_pressure_level"
        if not self.is_complex:
            raise ConfigurationError(
                f"{who}: needs complex H(f). A dB or TL field has no phase, "
                f"so it has no waveform to take a peak of — synthesising "
                f"one inverse-transforms the decibels AS pressure. Start "
                f"from the BROADBAND field and scale the waveform to the "
                f"source level rather than calling at_source_level().")
        if not np.isfinite(reference) or reference <= 0.0:
            raise ConfigurationError(
                f"{who}: reference must be a positive pressure in Pa; got "
                f"{reference!r}.")
        traces = self.synthesize_time_series(
            source_waveform, sample_rate,
            window=window, nfft=nfft, t_start=t_start)
        pressure = np.asarray(traces.data)
        pressure = pressure.real if np.iscomplexobj(pressure) else pressure
        # acoustics.peak_level's, floored the same way, so a silent cell
        # is -180 dB and not -inf.
        peak = _peak_level(pressure, reference, axis=-1)
        coords = {name: v for name, v in traces.coords.items()
                  if name != 'time'}
        band = np.asarray(self.coords['frequency'], dtype=float)
        from uacpy.acoustic_signal.system import (
            waveform_spectrum_at as _source_spectrum_at)
        weights = np.abs(_source_spectrum_at(
            source_waveform, sample_rate, band)) ** 2
        centre = (float(np.sum(weights * band) / weights.sum())
                  if weights.sum() > 0 else float(np.mean(band)))
        meta = dict(self.metadata or {})
        meta.update({'kind': 'peak_pressure', 'unit': 'dB'})
        meta['band_hz'] = (float(band[0]), float(band[-1]))
        id_kwargs = self.id_kwargs()
        id_kwargs['metadata'] = meta
        id_kwargs['frequencies'] = np.array([centre], dtype=float)
        pinned = dict(self.pinned)
        pinned['frequency'] = centre
        return Field(data=peak, coords=coords, pinned=pinned, **id_kwargs)

    def to_transfer_function(self, *, band=None) -> "Field":
        """``H(f)`` from a time-domain Field — the inverse of
        :meth:`to_time_trace`.

        The forward direction has several routes (:meth:`to_time_trace`,
        :meth:`synthesize_time_series`,
        :func:`~uacpy.acoustic_signal.impulse_response_from_transfer_function`);
        this is the way back, as a carrier rather than as raw arrays.

        The transform is::

            H(f) = rfft(h) * dt * exp(-2 pi i f t0)

        The final rotation is what makes it an inverse rather than merely a
        spectrum. A record starting at ``t0`` carries that offset in every
        sample, so a bare ``rfft`` returns ``H`` multiplied by
        ``exp(+2 pi i f t0)`` — correct in magnitude and wrong in phase,
        which is invisible until something interferes two of them. Verified
        against a two-path ``H``: with the rotation the round trip
        reproduces it to ``max|err| = 0.0000``; without it, 3.16.

        **The band is restricted, not extended.** An ``rfft`` of an
        ``N``-sample record returns bins from 0 to the Nyquist frequency,
        but a trace synthesised from a 100-995 Hz field supports nothing
        outside that — the other bins are the synthesis's own edges, and
        returning them would invent data. The band comes from the Field's
        identity when it has one (every trace this package synthesises
        does), or from ``band``.

        **Compared with the two narrower tools.**
        :meth:`get_spectrum` is the raw ``rfft`` — every bin, no rotation,
        arrays not a Field — and is right when that is what is wanted.
        :meth:`extract_tone` is the careful single-frequency answer,
        evaluated AT the frequency rather than at the nearest bin. This is
        the broadband carrier-level one.

        Parameters
        ----------
        band : (float, float), optional
            ``(low, high)`` in Hz to keep. ``None`` uses the identity's
            band, and falls back to every positive bin when the Field
            carries none — a time-marched result read from a file, say.

        Returns
        -------
        Field
            Complex, with ``time`` replaced by ``frequency`` and the other
            axes unchanged, so :meth:`plot_transfer_function`,
            :meth:`broadband_loss` and :meth:`truncate_response` all take
            it.

        Raises
        ------
        ConfigurationError
            No ``time`` axis, fewer than two samples, a non-uniform one, or
            a ``band`` that keeps no bin.
        """
        who = "Field.to_transfer_function"
        if 'time' not in self.coords:
            raise ConfigurationError(
                f"{who}: needs a time axis to transform; this Field has "
                f"{list(self.coords)}. A frequency-domain field is already "
                f"a transfer function.")
        times = np.asarray(self.coords['time'], dtype=float)
        if times.size < 2:
            raise ConfigurationError(
                f"{who}: needs at least two time samples; got {times.size}.")
        steps = np.diff(times)
        dt = float(np.mean(steps))
        if not np.allclose(steps, dt, rtol=1e-9, atol=0.0):
            raise ConfigurationError(
                f"{who}: the time axis is not uniformly spaced, so an FFT "
                f"of it would place every bin wrongly. Resample first.")
        axis = list(self.coords).index('time')
        if band is None:
            identity = self.frequencies
            if identity is not None and len(identity) > 1:
                band = (float(np.min(identity)), float(np.max(identity)))
        # The transform, the t0 rotation and the band cut are
        # transfer_function_from_impulse_response's; what a Field adds is
        # the axis bookkeeping, the identity-derived band above and the
        # metadata below. A pressure history is real, and that function
        # drops an analytic signal's quadrature rather than double the
        # positive-frequency content.
        from uacpy.acoustic_signal.system import (
            transfer_function_from_impulse_response)
        freqs, spectrum = transfer_function_from_impulse_response(
            self.data, 1.0 / dt, t0=float(times[0]), band=band, axis=axis,
            who=who)
        # That function is unscaled, matching the plain irfft of its own
        # counterpart; a Field's H is a spectral density, the convention
        # to_time_trace's `ifft * fs` produces, so dt carries it across.
        spectrum = spectrum * dt
        spectrum = np.moveaxis(spectrum, axis, -1)
        coords = {name: v for name, v in self.coords.items() if name != 'time'}
        coords['frequency'] = freqs
        meta = dict(self.metadata or {})
        meta.update({'kind': 'pressure', 'unit': 'Pa'})
        id_kwargs = self.id_kwargs()
        id_kwargs['metadata'] = meta
        id_kwargs['frequencies'] = freqs
        return Field(
            data=np.moveaxis(spectrum, -1, list(coords).index('frequency')),
            coords=coords, pinned=dict(self.pinned), **id_kwargs)

    def _reduce_to_spectrum(self, method: str) -> "Field":
        """Reduce a broadband Field to a single ``['frequency']`` spectrum.

        Singleton ``depth`` / ``range`` axes are squeezed automatically (so a
        single-receiver field needs no ``.at()``); any remaining non-frequency
        axis means the caller must pick a cell first. Used by the
        transfer-function / impulse-response plot helpers."""
        if 'frequency' not in self.coords:
            raise ConfigurationError(
                f"Field.{method}: needs a broadband field with a 'frequency' "
                f"axis; got coords {list(self.coords)}."
            )
        f = self
        for axis in ('source_depth', 'depth', 'range'):
            if axis in f.coords and f.coords[axis].size == 1:
                f = f.isel(**{axis: 0})
        if list(f.coords) != ['frequency']:
            raise ConfigurationError(
                f"Field.{method}: reduce to one (depth, range) cell first, "
                f"e.g. H.at(depth=…, range=…) — after squeezing singleton axes "
                f"the remaining coords are {list(f.coords)}."
            )
        return f

    def plot_transfer_function(
        self, *, axes=None, ax=None, title=None, figsize=(8, 6), **kwargs,
    ):
        """Plot the transfer function ``H(f)`` at one receiver cell as two
        stacked panels: modulus in dB (``20·log10|H|``, top) over phase
        (bottom), sharing the frequency axis.

        Reduce-then-plot: call on a field already sliced to one ``(depth,
        range)`` cell (``H.at(depth=…, range=…).plot_transfer_function()``); a
        single-receiver field plots directly (singleton axes are squeezed).
        Pass ``axes=(ax_mag, ax_phase)`` — or ``ax=``, the spelling every
        other uacpy plot method uses — to draw into existing axes. This one
        draws two panels, so either name takes a **pair**: anything that
        unpacks into two Axes, including the ndarray ``plt.subplots(2, 1)``
        returns. Returns ``(fig, (ax_mag, ax_phase))``.

        What the panels show depends on the frequency grid. Each pair of
        paths ``dtau`` apart puts a fringe of period ``1/dtau`` on
        ``|H(f)|``, and a grid from ``Arrivals.synthesis_band`` places
        ``margin`` samples on it — 1.2 by default, so the modulus is drawn
        critically sampled: a dense oscillation under a beat envelope, real
        multipath interference but interpolated by the plotter between
        samples. Raise ``margin`` to see the fringes. The phase is drawn
        wrapped, and a bulk delay ``tau`` turns it once every ``1/tau`` Hz,
        so over a band far wider than that it fills the panel; take the
        delay out (``H * exp(2j*pi*f*tau)``) to see what remains."""
        import matplotlib.pyplot as plt
        # ``ax`` is the name every sibling uses, and left to ``**kwargs`` it
        # reached ``spec.plot(..., ax=ax_mag, **kwargs)`` below as a duplicate
        # keyword — a TypeError naming Result.plot, a method the caller never
        # invoked.
        if ax is not None:
            if axes is not None:
                raise ConfigurationError(
                    "Field.plot_transfer_function: pass axes= or ax=, not "
                    "both — they name the same argument.",
                    remediation="Drop one; both take (ax_mag, ax_phase).",
                )
            axes = ax
        if axes is not None:
            # The acceptance test is the two-target unpack this function
            # performs on ``axes`` further down, so it admits exactly what
            # that admits and cannot narrow it: a tuple, a list, the ndarray
            # ``plt.subplots(2, 1)`` actually returns, ``axs.ravel()``,
            # ``axs.flat``. A type test would have to enumerate those.
            try:
                ax_mag, ax_phase = axes
            except (TypeError, ValueError) as exc:
                try:
                    given = len(axes)
                except TypeError:
                    given = 1
                raise ConfigurationError(
                    f"Field.plot_transfer_function: draws two stacked panels, "
                    f"so it needs a pair of Axes; got {given}.",
                    remediation="Pass ax=(ax_mag, ax_phase) — the second "
                                "return value of "
                                "plt.subplots(2, 1, sharex=True) is one.",
                ) from exc
            axes = (ax_mag, ax_phase)
        spec = self._reduce_to_spectrum('plot_transfer_function')
        if not spec.is_complex:
            raise ConfigurationError(
                "Field.plot_transfer_function: needs a complex H(f) (a real "
                "dB spectrum has no phase panel) — plot it with "
                ".plot(value='dB') instead."
            )
        owns_fig = axes is None
        if owns_fig:
            fig, (ax_mag, ax_phase) = plt.subplots(
                2, 1, sharex=True, figsize=figsize)
        else:
            ax_mag, ax_phase = axes
            fig = ax_mag.figure
        spec.plot(value='mag_dB', ax=ax_mag, title=title, **kwargs)
        spec.plot(value='phase', ax=ax_phase, **kwargs)
        ax_phase.set_title('')       # keep the title/pinned subtitle on top only
        ax_mag.set_xlabel('')        # shared axis: label only the bottom panel
        if owns_fig:
            # plot_field skips its credit when handed an ``ax``; draw the
            # model-source footnote once, from the (attributed) source Field.
            # Deferred into the body: ``uacpy.visualization`` imports
            # ``uacpy.core`` at module scope, so this line at file scope makes
            # ``import uacpy`` raise ImportError. docs/DEV.md section 7 records
            # the inversion.
            from uacpy.visualization.plots._common import _draw_result_credit
            _draw_result_credit(fig, self)
        return fig, (ax_mag, ax_phase)

    def plot_impulse_response(
        self, *, ax=None, title=None, window: str = 'hann',
        nfft: Optional[int] = None, t_start: Optional[float] = None,
        figsize=(8, 4), **kwargs,
    ):
        """Plot the band-limited impulse response ``p(t)`` at one receiver cell.

        Reduce-then-plot counterpart of :meth:`plot_transfer_function`: IFFTs
        the single-cell spectrum (``H.at(depth=…, range=…)
        .plot_impulse_response()``; a single-receiver field works directly).
        For the response to a specific source pulse use
        :meth:`synthesize_time_series` instead. Returns ``(fig, ax)``."""
        import matplotlib.pyplot as plt
        spec = self._reduce_to_spectrum('plot_impulse_response')
        if 'range' not in spec.pinned:
            raise ConfigurationError(
                "Field.plot_impulse_response: the spectrum carries no pinned "
                "range — the IFFT needs it for t_start and demodulation. "
                "Slice a canonical broadband grid (H.at(depth=…, range=…)), "
                "or use to_time_trace on the grid directly."
            )
        # Rebuild the canonical (depth, range, frequency) cell so the existing
        # IFFT path applies; the pinned depth/range come from the reduction.
        grid = Field(
            data=spec.data.reshape(1, 1, -1),
            coords={'depth': np.array([spec.pinned.get('depth', 0.0)]),
                    'range': np.array([spec.pinned['range']]),
                    'frequency': spec.coords['frequency']},
            pinned={k: v for k, v in spec.pinned.items()
                    if k not in ('depth', 'range')},
            **spec.id_kwargs(),
        )
        # ``nfft``/``t_start`` belong to the synthesis, not to the line:
        # the sampling warning tells the caller to pass ``t_start=``, so
        # it has to arrive here rather than at matplotlib.
        trace = grid.to_time_trace(window=window, nfft=nfft,
                                   t_start=t_start)
        owns_fig = ax is None
        if owns_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        trace.plot(ax=ax, title=title, **kwargs)
        if owns_fig:
            # Draw the model-source footnote from the (attributed) source Field
            # — the IFFT trace does not carry the model provenance.
            # Deferred into the body: ``uacpy.visualization`` imports
            # ``uacpy.core`` at module scope, so this line at file scope makes
            # ``import uacpy`` raise ImportError. docs/DEV.md section 7 records
            # the inversion.
            from uacpy.visualization.plots._common import _draw_result_credit
            _draw_result_credit(fig, self)
        return fig, ax

    # ── time-domain only (requires 'time' coord) ──────────────────────

    def get_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """Real FFT along the time axis. Returns ``(freqs, X)``.

        Requires a ``'time'`` axis."""
        if 'time' not in self.coords:
            raise ConfigurationError(
                f"Field.get_spectrum: requires a 'time' axis; "
                f"got {list(self.coords)}"
            )
        time_ax = list(self.coords).index('time')
        X = np.fft.rfft(self.data, axis=time_ax)
        freqs = np.fft.rfftfreq(self.n_times, self.dt)
        return freqs, X

    def extract_tone(
        self,
        frequency: float,
        *,
        window: str = 'hann',
    ) -> "Field":
        """Extract steady-state complex pressure at one frequency from a
        time-domain Field. Requires ``coords == {'depth', 'range', 'time'}``.

        The transform is evaluated **at** ``frequency``, not at the nearest
        rfft bin, so a tone that does not land on the record's bin grid is
        recovered correctly; on a bin it reproduces the rfft to ~1e-15. The
        returned Field's ``frequencies``/``pinned['frequency']`` therefore
        carry the frequency asked for.

        The ``2·X/Σwin`` tone estimator assumes a non-DC, non-Nyquist
        frequency; at exactly 0 Hz or the Nyquist frequency the doubling
        overestimates the amplitude by 2×.

        The returned value is the phasor ``A`` of
        ``p(t) = Re{A·e^{+2πift}}`` — the same sign convention the IFFT
        synthesis consumes, so a tone extracted here and an ``H(f)`` bin
        handed to :meth:`to_time_trace` carry phase the same way.
        """
        if list(self.coords) != ['depth', 'range', 'time']:
            raise ConfigurationError(
                "Field.extract_tone: requires canonical "
                "['depth', 'range', 'time'] coords; got "
                f"{list(self.coords)}"
            )
        # The estimator is tone_phasor's; what this method adds is the
        # coord check above and the Field re-wrap below.
        # Evaluate the transform AT the requested frequency rather than
        # sampling the nearest rfft bin, exactly as `_source_spectrum_at`
        # does and for the same reason: `2*X[k]/sum(win)` recovers the phasor
        # only when the tone sits on a bin, and off-bin `X[k]` is a leakage
        # sample of the window transform — neither the phasor at `frequency`
        # nor the one at `freqs[k]`. A model-produced trace picks its own `nt`
        # and `fs`, so the source frequency is essentially never on a bin:
        # half a bin off, this returned 0.84890 for a unit tone (-1.423 dB)
        # with its phase 89.98 deg out, silently. On a bin the sum below
        # reproduces the rfft bin it replaces to ~1.5e-15 — the two differ
        # only in summation order.
        # Deferred: acoustic_signal pulls scipy, and uacpy's public
        # surface is imported without it (test_lazy_imports).
        from uacpy.acoustic_signal.system import tone_phasor
        amp = tone_phasor(
            self.data, np.asarray(self.coords['time'], dtype=float),
            frequency, window=window, who='Field.extract_tone')
        # The recovered tone is the identity of the returned Field, not the
        # time-domain parent's frequency list.
        id_kwargs = self.id_kwargs()
        id_kwargs['frequencies'] = np.array([float(frequency)])
        return Field(
            data=amp,
            coords={'depth': self.coords['depth'], 'range': self.coords['range']},
            pinned={**self.pinned, 'frequency': float(frequency)},
            **id_kwargs,
        )


# ─────────────────────────────────────────────────────────────────────────────
# ResultStack — for non-Field stacks (e.g. multi-source Rays / Arrivals)
# ─────────────────────────────────────────────────────────────────────────────


_RESULTSTACK_VARYING_ATTR = {
    'source_depth': 'source_depths',
    'frequency':    'frequencies',
}


def _check_superpose_grids(slabs) -> None:
    """Refuse slabs that are not sampled at the same points.

    Either kind of sum adds cell to cell, so this belongs to both: a
    coherent one would add pressures from different places, an incoherent
    one their intensities, and the answer would carry the first slab's axes
    whichever it was.
    """
    first = slabs[0]
    for i, slab in enumerate(slabs[1:], start=1):
        same_axes = list(slab.coords) == list(first.coords) and all(
            np.array_equal(slab.coords[k], first.coords[k])
            for k in first.coords)
        if not same_axes or slab.data.shape != first.data.shape:
            sizes = {k: (first.coords[k].size, slab.coords[k].size)
                     for k in first.coords
                     if k in slab.coords
                     and first.coords[k].size != slab.coords[k].size}
            detail = (
                f"axes {list(first.coords)} vs {list(slab.coords)}"
                if list(slab.coords) != list(first.coords) else
                f"shape {first.data.shape} vs {slab.data.shape}"
                + (f", axis lengths {sizes}" if sizes else
                   "; same lengths, different coordinate values")
            )
            raise ConfigurationError(
                f"ResultStack.superpose: slabs[{i}] is on a different "
                f"grid from slabs[0] ({detail}); a sum needs every slab "
                f"sampled at the same points. A TIME_SERIES pair gets one "
                f"time axis from run(output_duration=…)."
            )


def _check_stack_weightable(field: 'Field', weights, *, where: str) -> None:
    """The weaker check a *stack* has to pass: refuse only what no sum of
    it could ever use.

    A stack is not weighted when it is built — :meth:`ResultStack.superpose`
    applies the weights, and the caller has not yet said which sum they
    mean. A dB-only stack cannot add coherently but adds perfectly well in
    intensity, so refusing it here would close the only route to "N mutually
    incoherent sources at these levels". What is unusable either way is a
    complex weight on data with no phase to rotate; that is refused now,
    while the rest waits for :meth:`ResultStack.superpose` to judge against
    the sum actually asked for.
    """
    w = np.atleast_1d(np.asarray(weights, dtype=np.complex128))
    if not np.any(w.imag != 0.0):
        return
    if 'time' in field.coords or not field.is_complex:
        _check_field_weightable(field, w, where=where)
    elif field.phase_reference is None:
        _check_field_weightable(field, w, where=where)


def _check_field_weightable(field: 'Field', weights, *, where: str) -> None:
    """Raise :class:`ConfigurationError` unless ``weights`` can scale
    ``field``. The one rule behind the n = 1 weight
    :meth:`PropagationModel.run` applies and the n-slab sum
    :meth:`ResultStack.superpose` forms:

    * a real frequency-domain field (dB) has lost its phase and takes no
      weight at all — multiplying decibels is not a scaling;
    * a real time-domain trace takes a real weight (a sign flip is -1),
      never a complex one;
    * complex data takes a complex weight only when it carries a
      ``phase_reference``. Bellhop's incoherent and semicoherent beam sums
      are stored complex with none (``bellhop.py`` stamps
      ``phase_reference=None`` outside the coherent run type), so their
      phase is an artefact of AT's storage rather than a propagation
      phase; rotating it would record a source phase the field cannot
      carry. A real weight still scales such a field's level.
    """
    w = np.atleast_1d(np.asarray(weights, dtype=np.complex128))
    time_domain = 'time' in field.coords
    if not time_domain and not field.is_complex:
        raise ConfigurationError(
            f"{where}: the field is real {field.unit!r} values, not "
            f"complex pressure, so its phase is gone and a weight cannot "
            f"scale it (multiplying dB is not a coherent sum). Weight the "
            f"complex field the run returned (before to_dB(), or a mode "
            f"that keeps phase — phase_reference is not None); for the "
            f"level of one source at magnitude |w|, subtract "
            f"20*log10(|w|) from the dB view yourself."
        )
    if not field.is_complex and np.any(w.imag != 0.0):
        raise ConfigurationError(
            f"{where}: the data are real time-domain traces, which a "
            f"complex weight cannot scale; give a real weight (a sign flip "
            f"is -1), or run BROADBAND and weight the transfer function "
            f"before synthesising."
        )
    if (field.is_complex and field.phase_reference is None
            and np.any(w.imag != 0.0)):
        raise ConfigurationError(
            f"{where}: the data are complex but carry no phase reference — "
            f"an incoherent or semicoherent beam sum, whose phase is an "
            f"artefact of the engine's storage and not a propagation "
            f"phase. Rotating it by a complex weight would record a source "
            f"phase the field cannot carry; give a real weight to scale its "
            f"level, or run a coherent mode, which stamps phase_reference."
        )


class ResultStack(_DeepCopyMixin):
    """Stack of typed :class:`Result` slabs along one coordinate.

    Bundles a list of slabs together with the coordinate vector along
    which they are stacked. The coordinate can be a :class:`Result`
    field (``source_depth``, ``frequency``) or an external parameter
    the user varied. Every slab carries the same concrete type,
    ``model``, and ``backend``, and the same identification along
    every axis *except* the stacking axis.

    This is what a multi-source run returns, for gridded (``Field``) and
    sparse (``Rays`` / ``Arrivals``) results alike. Consumers that need one
    dense array — matched-field processing, say — accept either this stack or
    a single :class:`Field` carrying the varying axis in ``coords`` (e.g.
    ``coords={'source_depth', 'depth', 'range'}``).

    Construction
    ------------
    ``ResultStack(slabs, coordinate, coordinate_name='source_depth')``

    Access
    ------
    ``stack[i]``                              i-th slab
    ``for c, slab in stack: …``               iterate ``(coordinate, slab)`` pairs
    ``stack.at(<coordinate_name>=value)``     nearest-label lookup
    ``len(stack)``                            number of slabs
    ``stack.superpose(weights)``              coherent sum ``Σ wᵢ·pᵢ`` → Field
    """

    field_type = 'stack'

    def __init__(
        self,
        slabs: List[Result],
        coordinate: Union[List[float], np.ndarray],
        *,
        coordinate_name: str = 'source_depth',
    ):
        if len(slabs) == 0:
            raise ConfigurationError("ResultStack: requires at least one slab")
        coord = np.atleast_1d(np.asarray(coordinate, dtype=float))
        if coord.size != len(slabs):
            raise ConfigurationError(
                f"ResultStack: coordinate length ({coord.size}) does not "
                f"match number of slabs ({len(slabs)})"
            )
        types = {type(s) for s in slabs}
        if len(types) != 1:
            raise ConfigurationError(
                f"ResultStack: every slab must have the same concrete "
                f"type; got {sorted(t.__name__ for t in types)}"
            )

        varying_attr = _RESULTSTACK_VARYING_ATTR.get(str(coordinate_name))
        shared_attrs = ['model', 'backend']
        for attr in ('frequencies', 'source_depths'):
            if attr != varying_attr:
                shared_attrs.append(attr)

        first = slabs[0]

        def _equal(a, b):
            if a is None or b is None:
                return a is None and b is None
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                a = np.asarray(a)
                b = np.asarray(b)
                return a.shape == b.shape and np.array_equal(a, b)
            return bool(a == b)

        for attr in shared_attrs:
            ref = getattr(first, attr, None)
            for i, s in enumerate(slabs[1:], start=1):
                val = getattr(s, attr, None)
                if not _equal(ref, val):
                    raise ConfigurationError(
                        f"ResultStack: slabs[0].{attr}={ref!r} but "
                        f"slabs[{i}].{attr}={val!r} — every slab must "
                        f"share the same {attr} (stacking axis is "
                        f"{coordinate_name!r})"
                    )

        self.slabs: List[Result] = list(slabs)
        self.coordinate: np.ndarray = coord
        self.coordinate_name: str = str(coordinate_name)

    @property
    def slab_type(self) -> type:
        return type(self.slabs[0])

    @property
    def n_slabs(self) -> int:
        return int(self.coordinate.size)

    @property
    def model(self) -> str:
        return self.slabs[0].model

    @property
    def backend(self) -> str:
        return self.slabs[0].backend

    @property
    def model_source(self):
        """Engine provenance of the first slab — what the plotters read for the
        model-credit footnote."""
        return self.slabs[0].model_source

    @property
    def phase_reference(self) -> Optional[str]:
        return self.slabs[0].phase_reference

    @property
    def frequencies(self) -> Optional[np.ndarray]:
        """Frequencies (Hz) the stack covers: the stacking coordinate when
        stacking by frequency, else the value every slab agrees on."""
        return self._identity_axis('frequencies')

    @property
    def source_depths(self) -> np.ndarray:
        """Source depths (m) the stack covers: the stacking coordinate when
        stacking by source depth, else the value every slab agrees on."""
        return self._identity_axis('source_depths')

    def _identity_axis(self, attr: str):
        if _RESULTSTACK_VARYING_ATTR.get(self.coordinate_name) == attr:
            return self.coordinate.copy()
        return getattr(self.slabs[0], attr)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Metadata of the first slab, as a copy. Slabs are not required to
        agree on metadata — read a specific slab's dict via ``stack[i]``."""
        return dict(self.slabs[0].metadata)

    def __len__(self) -> int:
        return self.n_slabs

    def __getitem__(self, index: int) -> Result:
        return self.slabs[int(index)]

    def __iter__(self):
        for c, slab in zip(self.coordinate, self.slabs):
            yield float(c), slab

    def at(self, **kwargs) -> Result:
        """Select the slab nearest a value on the stacking axis.

        Pass exactly the stacking-axis keyword (``<coordinate_name>=<value>``);
        returns the slab whose coordinate is closest to ``value``. The value
        must be a finite scalar, the same label contract :meth:`Field.at`
        applies.
        """
        if len(kwargs) != 1 or self.coordinate_name not in kwargs:
            raise ConfigurationError(
                f"ResultStack.at(): pass exactly the stacking-axis "
                f"keyword ({self.coordinate_name}=<value>); got "
                f"{list(kwargs)}"
            )
        idx = _nearest_index_on_axis(
            self.coordinate, kwargs[self.coordinate_name],
            self.coordinate_name)
        return self.slabs[idx]

    def isel(self, **kwargs) -> Result:
        """Select a slab by integer position on the stacking axis.

        Pass exactly the stacking-axis keyword (``<coordinate_name>=<index>``);
        the positional counterpart of :meth:`at` (and of ``stack[index]``),
        mirroring :meth:`Field.isel`.
        """
        if len(kwargs) != 1 or self.coordinate_name not in kwargs:
            raise ConfigurationError(
                f"ResultStack.isel(): pass exactly the stacking-axis "
                f"keyword ({self.coordinate_name}=<index>); got {list(kwargs)}"
            )
        return self.slabs[int(kwargs[self.coordinate_name])]

    @property
    def dB(self) -> np.ndarray:
        """Every slab's dB view stacked along the coordinate axis — shape
        ``(n_slabs, *slab.dB.shape)`` — so generic code can read ``result.dB``
        whether one or many source depths were requested. Requires Field slabs.
        """
        first = self.slabs[0]
        if not isinstance(first, Field):
            raise ConfigurationError(
                f"ResultStack.dB: slabs are {self.slab_type.__name__}, not "
                f"Field — no dB view. Pick a slab with stack[i] or "
                f"stack.at({self.coordinate_name}=...)."
            )
        self._warn_if_weights_unapplied('dB')
        if 'time' in first.coords:
            raise ConfigurationError(
                "ResultStack.dB: time-domain slabs are linear pressure, not "
                "a level; read the samples via stack[i].data, or recover a "
                "complex narrowband field first with stack[i].extract_tone(f)."
            )
        # Complex slabs derive their dB view (unit 'Pa', -20*log10|data|);
        # a real slab's data IS its dB view only when its unit says so, and
        # Field.dB refuses any other unit — pre-check it here so the stack
        # raises the same typed error as the time-domain case above.
        if not first.is_complex and first.unit != 'dB':
            raise ConfigurationError(
                f"ResultStack.dB: slabs are in {first.unit!r}, not dB, so "
                f"their values are not a level; read them via stack[i].data. "
                f"to_dB() returns a real slab unchanged, so if a dB view of "
                f"{first.unit!r} is meaningful, take "
                f"20*np.log10(np.abs(stack[i].data)) yourself and tag the "
                f"result unit='dB'."
            )
        return np.stack([s.dB for s in self.slabs], axis=0)

    @property
    def tl(self) -> np.ndarray:
        """Every slab's transmission loss stacked along the coordinate
        axis — :attr:`dB` restricted to pressure slabs, mirroring
        :attr:`Field.tl` in values. The refusal type follows each class's
        own accessors: ``Field`` accessors raise :class:`AttributeError`,
        stack accessors raise ConfigurationError."""
        first = self.slabs[0]
        if isinstance(first, Field) and first.kind != 'pressure':
            raise ConfigurationError(
                f"ResultStack.tl: the slabs' kind is {first.kind!r}, not "
                f"'pressure', so their values are not a transmission loss; "
                f"their level view is stack.dB."
            )
        return self.dB

    def superpose(self, weights=None, *, coherent: bool = True) -> 'Field':
        """Coherent sum of the slabs: one :class:`Field` holding
        ``Σ wᵢ·pᵢ`` over the stack on the shared receiver grid.

        Adding the complex pressure of each source is how a multi-source
        array is driven: the engines are linear in the source amplitude, so
        the field of a weighted array is the weighted sum of the unit-source
        fields the stack holds. Every slab must therefore carry complex
        pressure (``phase_reference`` intact) or a time-domain trace — a
        dB-only slab has lost its phase and is refused.

        Parameters
        ----------
        weights : array-like, optional
            One coefficient per slab. ``None`` reads the weights the
            ``Source`` that produced the stack carried
            (``metadata['source_weights']``, stamped by
            :meth:`PropagationModel.run`), else unit weights. Complex on a
            complex-pressure stack; real on a time-domain stack, whose
            traces are real samples. ``coherent=False`` uses only ``|w|``.
        coherent : bool, keyword-only
            How the sources combine, which is a statement about the sources
            and not about the arithmetic:

            ``True`` (default) adds complex pressure, ``Σ wᵢ·pᵢ`` — the
            sources are driven together with a fixed relative phase, as the
            elements of one array are.

            ``False`` adds intensity, ``√Σ|wᵢ·pᵢ|²`` — the sources are
            mutually incoherent (separate platforms, unrelated tones,
            random relative phase), so their phases carry no information and
            only ``|w|`` is read. N identical sources then give
            ``10·log10(N)`` where a coherent sum gives ``20·log10(N)``.
            The result has no phase, so it comes back the way every engine
            returns its own incoherent mode: real dB, ``phase_reference``
            cleared. A dB-only stack, which cannot add coherently, adds this
            way.

        Returns
        -------
        Field
            Same grid and identity as the slabs, with ``source_depths``
            widened to the stacking coordinate and
            ``metadata['superposed_sources']`` recording the ``depths``, the
            ``weights`` and whether the sum was ``coherent``. A coherent sum
            keeps the slabs' ``phase_reference``; an incoherent one clears
            it and returns real dB.

        Raises
        ------
        ConfigurationError
            Non-Field slabs; slabs on different grids; a weight vector of
            the wrong length or with a non-finite entry. A coherent sum also
            refuses a real frequency-domain (dB) stack, a complex weight on
            a time-domain stack, and a complex weight on data carrying no
            phase reference; an incoherent sum refuses time-domain traces,
            which do not add in intensity sample by sample.
        """
        first = self.slabs[0]
        if not isinstance(first, Field):
            raise ConfigurationError(
                f"ResultStack.superpose: slabs are "
                f"{self.slab_type.__name__}, not Field — only gridded "
                f"pressure adds. Pick a slab with stack[i]."
            )
        if weights is None:
            weights = first.metadata.get('source_weights')
        if weights is None:
            weights = np.ones(self.n_slabs)
        w = np.atleast_1d(np.asarray(weights, dtype=np.complex128))
        if w.ndim != 1 or w.size != self.n_slabs:
            raise ConfigurationError(
                f"ResultStack.superpose: {self.n_slabs} slabs but "
                f"{w.size} weight(s) (shape {w.shape}); give one weight "
                f"per slab."
            )
        if not np.all(np.isfinite(w)):
            bad = int(np.flatnonzero(~np.isfinite(w))[0])
            raise ConfigurationError(
                f"ResultStack.superpose: weights must be finite; "
                f"weights[{bad}] = {w[bad]}"
            )
        _check_superpose_grids(self.slabs)
        if not coherent:
            return self._superpose_incoherent(w)
        _check_field_weightable(first, w, where="ResultStack.superpose")
        real_data = not first.is_complex
        if real_data:
            w = w.real

        total = np.zeros(first.data.shape,
                         dtype=np.result_type(first.data.dtype, w.dtype))
        for wi, slab in zip(w, self.slabs):
            total += wi * slab.data

        id_kwargs = first.id_kwargs()
        id_kwargs['source_depths'] = self.coordinate.copy()
        meta = id_kwargs['metadata']
        meta.pop('source_weights', None)
        meta['superposed_sources'] = {
            'depths': self.coordinate.tolist(),
            'weights': w.tolist(),
            'coherent': True,
        }
        pinned = {k: v for k, v in first.pinned.items()
                  if k != 'source_depth'}
        return Field(data=total, coords=first.coords, pinned=pinned,
                     **id_kwargs)

    def _superpose_incoherent(self, w) -> 'Field':
        """``√Σ|wᵢ·pᵢ|²`` over the slabs, as a real dB ``Field``.

        Reads magnitudes only, so it serves a dB-only stack as well as a
        complex one: ``|p| = 10**(-dB/20)`` recovers the magnitude a level
        already is. Time-domain traces are refused — intensity does not add
        sample by sample."""
        first = self.slabs[0]
        if 'time' in first.coords:
            raise ConfigurationError(
                "ResultStack.superpose(coherent=False): the slabs are "
                "time-domain traces, and intensity does not add sample by "
                "sample. Sum the traces coherently, or superpose the "
                "BROADBAND transfer function and synthesise afterwards."
            )
        if np.any(w.imag != 0.0):
            warnings.warn(
                "ResultStack.superpose(coherent=False): an intensity sum "
                "carries no phase, so only the weight magnitudes are used "
                f"({np.abs(w).tolist()}); the phases of {w.tolist()} are "
                "dropped. Pass coherent=True to drive the sources with a "
                "fixed relative phase.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
        amp = np.abs(w)

        # A loss counts DOWN from the source and a level counts UP, so the
        # stored numbers turn into an amplitude with opposite signs. Reading
        # every real dB field as a loss inverted a stack of levels: two
        # incoherent 120 dB sources came back 116.99 dB instead of 123.01.
        loss = _quantities.is_loss(first.kind)
        sign = -1.0 if loss else 1.0

        def magnitude(slab):
            data = np.asarray(slab.data)
            if slab.is_complex:
                return np.abs(data)
            return np.power(10.0, sign * np.asarray(data, dtype=float) / 20.0)

        total = np.zeros(np.asarray(first.data).shape, dtype=float)
        for a, slab in zip(amp, self.slabs):
            total += (a * magnitude(slab)) ** 2
        # Clamped like every other dB view (``_complex_to_dB``), so a cell no
        # energy reached reads the package's floor instead of ``inf``.
        # A loss counts down (-20log10|p|), a level counts up (+20log10|p|),
        # which is the same ``sign`` the magnitudes were recovered with.
        level = sign * 20.0 * np.log10(
            np.maximum(np.sqrt(total), PRESSURE_FLOOR))

        id_kwargs = first.id_kwargs()
        id_kwargs['source_depths'] = self.coordinate.copy()
        id_kwargs['phase_reference'] = None
        meta = id_kwargs['metadata']
        meta.pop('source_weights', None)
        meta['unit'] = 'dB'
        meta['kind'] = first.kind
        meta['superposed_sources'] = {
            'depths': self.coordinate.tolist(),
            'weights': amp.tolist(),
            'coherent': False,
        }
        pinned = {k: v for k, v in first.pinned.items()
                  if k != 'source_depth'}
        return Field(data=level, coords=first.coords, pinned=pinned,
                     **id_kwargs)

    def _warn_if_weights_unapplied(self, view: str) -> None:
        """Every slab is the unit-amplitude field of one source; when the
        ``Source`` carried other weights, the level view or panel plot of
        the slabs shows a field those weights never touched. Say so once,
        naming :meth:`superpose`, which is where they apply."""
        weights = self.slabs[0].metadata.get('source_weights')
        if weights is None or np.all(np.asarray(weights) == 1.0):
            return
        warnings.warn(
            f"ResultStack.{view}: the slabs are unit-amplitude fields; the "
            f"Source weights {np.asarray(weights).tolist()} this stack "
            f"carries are not applied to them. Call stack.superpose() for "
            f"the weighted field.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)

    def plot(self, **kwargs):
        """Plot every slab as a labelled panel grid (Field stacks), delegating
        to :func:`uacpy.visualization.plot_result`."""
        if isinstance(self.slabs[0], Field):
            self._warn_if_weights_unapplied('plot')
        # Deferred into the body: ``uacpy.visualization`` imports
        # ``uacpy.core`` at module scope, so this line at file scope makes
        # ``import uacpy`` raise ImportError. docs/DEV.md section 7 records
        # the inversion.
        from uacpy.visualization import plots
        return plots.plot_result(self, **kwargs)

    def __repr__(self) -> str:
        return (
            f"ResultStack[{self.slab_type.__name__}]"
            f"(n_slabs={self.n_slabs}, "
            f"{self.coordinate_name}={self.coordinate.tolist()})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sparse / non-grid results
# ─────────────────────────────────────────────────────────────────────────────


def _synthesis_plan(
    tf: "Field",
    *,
    window: str,
    nfft: Optional[int],
    sample_rate: Optional[float],
    who: str,
) -> Tuple[np.ndarray, float, np.ndarray, float, int, np.ndarray]:
    """Validate ``tf``'s frequency axis and size the synthesis grid that every
    cell of the Field shares: returns ``(freqs, df, bin_indices,
    bin_offset_hz, nfft, win)``.

    The single home for the per-Field half of the IFFT synthesis —
    ``_ifft_to_trace`` (one cell) and ``_synthesize_time_series`` (every cell)
    both build on it, so the two paths cannot drift apart. ``who`` is the
    public entry point's name and prefixes every diagnostic (like
    :func:`_taper`).
    """
    freqs = np.asarray(tf.coords['frequency'], dtype=float)
    n_freq = freqs.size

    if n_freq < 2:
        raise ConfigurationError(
            f"{who}: need at least 2 frequencies for IFFT; got {n_freq}"
        )

    if tf.phase_reference == 'time_domain_native':
        raise ConfigurationError(
            f"{who}: phase_reference='time_domain_native' is not a "
            "frequency-domain transfer function; the producing model "
            "(SPARC) returned p(t) directly — read the time-domain Field "
            "from RunMode.TIME_SERIES instead of synthesising via IFFT"
        )

    # The transfer function is sampled at df_data, so the trace it can
    # represent without aliasing is exactly 1/df_data long. Refining df below
    # that (to force a longer window) has to invent the samples in between,
    # and linear interpolation of the spectrum is a convolution with a
    # triangular kernel — i.e. a sinc^2(pi df_data t) taper in time, which
    # progressively attenuates arrivals away from the anchor it is centred on.
    # Return the honest extent instead; a longer record needs a finer
    # frequency grid, which means more model runs.
    from uacpy.acoustic_signal.system import uniform_frequency_step
    df_data = uniform_frequency_step(freqs, who)
    df = df_data

    # A DFT of spacing df can only carry frequencies at integer multiples of
    # df, so each model frequency lands at bin round(f/df). When f[0] is not
    # itself a multiple of df the whole band is placed offset by a common
    # ``bin_offset_hz`` (|offset| <= df/2); ``_synthesize_traces`` removes it
    # exactly by de-rotating the complex sum, so the trace is the band the
    # caller asked for rather than a frequency-shifted copy of it.
    bin_indices = np.floor(freqs / df + 0.5).astype(int)
    bin_offset_hz = float(bin_indices[0] * df - freqs[0])
    # That the offset is COMMON is an assumption, and it fails on a knife
    # edge: when freqs[0]/df sits on the .5 boundary that np.floor(x + 0.5)
    # breaks, the first sample rounds one way and the rest the other, and
    # part of the band is placed a whole bin from where it belongs. The
    # result is silently wrong, not obviously broken — on a 25 Hz-4 kHz band
    # with df = 2/3 (freqs[0]/df = 37.5) a two-path SEL read 52.34 dB
    # against a true 54.01, on a 1500 ms record where nothing folds. Nudging
    # df by 0.005 Hz either way is exact, so it cannot be left to the
    # caller to notice. Checked rather than assumed.
    residual = np.abs(bin_indices * df - freqs - bin_offset_hz)
    if residual.size and float(np.max(residual)) > 1e-6 * df:
        raise ConfigurationError(
            f"{who}: this frequency grid cannot be placed on a DFT of "
            f"spacing {df:g} Hz — freqs[0]/df = {freqs[0] / df:g} lands on "
            f"the rounding boundary, so the band would be split across two "
            f"bin offsets and the result would be wrong by decibels without "
            f"looking wrong. Shift the band start or df by a fraction of a "
            f"bin (a few parts in 1e3 of {df:g} Hz is enough), or build the "
            f"grid so freqs[0] is a multiple of df.")
    max_bin = int(bin_indices[-1])
    explicit_nfft = nfft is not None

    if nfft is None:
        # Floor the auto length at 4 bins per model frequency, and never below
        # a time-sample count the model already reported. On a baseband grid
        # the anti-aliasing minimum below is 2*max_bin + 2 = 2*n_freq, so the
        # floor leaves the trace time-oversampled ~2x rather than critically
        # sampled — the extra bins are zero-padding, which interpolates the
        # trace without changing its band.
        nfft_min = max(int(tf.metadata.get('n_samples', 0)) or 0, 4 * n_freq)
        nfft_target = max(nfft_min, 2 * max_bin + 2)
        if sample_rate is not None:
            nfft_target = max(nfft_target, int(np.ceil(sample_rate / df)))
        nfft = 1
        while nfft < nfft_target:
            nfft *= 2
        if nfft > _MAX_SYNTHESIS_NFFT:
            raise ConfigurationError(
                f"{who}: the requested grid implies an "
                f"{nfft:,}-sample output (~{nfft * 16 / 1e9:.1f} GB), above the "
                f"{_MAX_SYNTHESIS_NFFT:,}-sample safety cap. This is driven by "
                f"sample_rate={sample_rate!r} Hz against a frequency resolution "
                f"df={df:.4g} Hz (length ~ sample_rate/df). Lower sample_rate, "
                f"widen df (coarser frequency grid / shorter window), or pass an "
                f"explicit nfft= if you really need an output this large.",
                remediation="A typical fix is a smaller sample_rate.",
            )

    if explicit_nfft and max_bin >= nfft // 2:
        raise ConfigurationError(
            f"{who}: nfft={nfft} puts the highest data bin "
            f"({max_bin}, {freqs[-1]:.6g} Hz at Δf = {df:.6g} Hz) at or above "
            f"Nyquist (bin {nfft // 2}); those bins fold into the "
            f"negative-frequency half and alias onto the wrong frequencies. "
            f"Use nfft >= {2 * max_bin + 2}, or drop nfft= to size it "
            f"automatically.",
            remediation=f"Pass nfft={2 * max_bin + 2} or larger.",
        )

    from uacpy.acoustic_signal.system import _taper
    win = _taper(window, n_freq, who=who)

    return freqs, df, bin_indices, bin_offset_hz, int(nfft), win


def _warn_unsolved_bins(
    spectra: np.ndarray,
    *,
    cell_depths: np.ndarray,
    cell_ranges: np.ndarray,
    who: str,
) -> None:
    """Warn, per cell, about NaN bins in a batch of cell spectra ``(M, n_f)``.

    A NaN bin is a frequency the model did not solve, not one carrying no
    energy, so it is never zeroed: filling it would put a spectral notch the
    model never produced into a trace that then looks finite and ordinary.
    The NaNs are kept and propagate through the IFFT, which makes the whole
    trace no-data — a trace cannot be synthesised from a spectrum with holes
    in it — and each affected cell is named. An all-NaN cell (one masked
    below the seafloor, say) gets its own wording, since nothing about it was
    solved. ``cell_depths`` / ``cell_ranges`` give each row's coordinates for
    the warning text. ``who`` is the public entry point's name and prefixes
    the diagnostic (like :func:`_synthesis_plan` and :func:`_taper`), since
    both entry points share this path.
    """
    nan_bins = np.isnan(spectra)
    all_nan = np.all(nan_bins, axis=1)
    n_f = spectra.shape[1]
    for i in np.flatnonzero(np.any(nan_bins, axis=1)):
        where = (f"H(f) at depth {float(cell_depths[i]):g} m, range "
                 f"{float(cell_ranges[i]):g} m")
        if all_nan[i]:
            detail = ("is entirely NaN (no valid model output at this "
                      "cell); the synthesised trace is NaN, not silence.")
        else:
            detail = (f"has {int(np.count_nonzero(nan_bins[i]))} of {n_f} "
                      f"bins the model did not solve; the synthesised trace "
                      f"is NaN rather than carrying a notch at those "
                      f"frequencies. Re-run them, or narrow the band to the "
                      f"bins that solved.")
        warnings.warn(f"{who}: {where} {detail}",
                      UserWarning, skip_file_prefixes=USER_FRAME_SKIP)


def _synthesize_traces(
    spectra: np.ndarray,
    *,
    freqs: np.ndarray,
    win: np.ndarray,
    source_spectrum: Optional[np.ndarray],
    bin_indices: np.ndarray,
    bin_offset_hz: float,
    nfft: int,
    df: float,
    t_start: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fourier-synthesize a batch of cell spectra ``(M, n_f)`` into time
    traces ``(M, nfft)`` sharing one window anchored at ``t_start``; returns
    ``(traces, time)``. One ``np.fft.ifft`` over the batch computes every
    cell's transform in a single call. See ``_ifft_to_trace`` for the
    synthesis contract and ``_synthesis_plan`` for the grid inputs."""
    dt = 1.0 / (nfft * df)
    spectra = spectra * win
    if source_spectrum is not None:
        spectra = spectra * np.asarray(source_spectrum)

    # Advance the record to t_start. The synthesis below evaluates
    # sum H(f) e^{+2*pi*i*f*t}, so pre-rotating by e^{+2*pi*i*f*t_start} puts
    # ifft sample n at t = t_start + n*dt instead of at n*dt.
    spectra = spectra * np.exp(1j * 2.0 * np.pi * freqs * t_start)

    # Only the positive-frequency half is physical here: 2·Re(ifft) folds
    # anything at or above Nyquist onto the wrong frequency.
    padded = np.zeros((spectra.shape[0], nfft), dtype=complex)
    valid = (bin_indices >= 0) & (bin_indices < nfft // 2)
    padded[:, bin_indices[valid]] = spectra[:, valid]

    # ifft carries 1/nfft; ×(nfft·df) turns the bin sum into ∫…df
    analytic = np.fft.ifft(padded, axis=-1) * (nfft * df)
    elapsed = np.arange(nfft) * dt
    if bin_offset_hz != 0.0:
        analytic = analytic * np.exp(-2j * np.pi * bin_offset_hz * elapsed)
    return 2.0 * np.real(analytic), t_start + elapsed


def _estimate_t_start(tf: "Field", actual_range: float, T_window: float,
                      who: str) -> float:
    """Start of a synthesis window ``T_window`` seconds long for a cell at
    ``actual_range``: the estimated first arrival, centred in the record.

    Half a window of lead absorbs an arrival earlier than the estimate, the
    other half holds the multipath tail. The estimate anchors on the fastest
    PHYSICAL speed the producing model stamped; it warns when the model
    stamped none, or only a surface speed while the window is short against
    the travel time. Shared by :func:`_ifft_to_trace` (one cell) and
    :func:`_synthesize_time_series` (one window for every cell).
    """
    # The earliest arrival travels at the FASTEST speed in the waveguide,
    # so r/c_fast bounds it from below. Candidates are the physical
    # speeds producers stamp: 'c_max' (Kraken, Scooter, RAM, OASES: the fastest speed anywhere
    # in the waveguide) and 'c0' (Bellhop, the sea-surface water speed).
    # Anchoring on a speed above c_fast opens the window too early: once
    # the excess lead exceeds the half-window margin, the late multipath
    # tail falls past the end of the record and wraps to the beginning
    # — so no algorithmic speed (e.g. a PE expansion point) may enter
    # this max, and c_min never binds it.
    c_max = float(tf.metadata.get('c_max') or 0.0)
    c0 = float(tf.metadata.get('c0') or 0.0)
    # The 1500 m/s default is a fallback for when nothing physical was
    # stamped, never a candidate beside a stamped speed: a stamped speed
    # BELOW it (cold or fresh water) must win, or the window opens early by
    # r·(1/c − 1/1500) and the arrival wraps a whole record with the time
    # axis mislabelled and nothing to show for it.
    stamped = [speed for speed in (c_max, c0) if speed > 0.0]
    anchor_speed = max(stamped) if stamped else DEFAULT_SOUND_SPEED
    travel = actual_range / anchor_speed
    # Centre the estimated first arrival in the record: half a window of
    # lead absorbs an arrival earlier than the estimate, the other half
    # holds the multipath tail behind it.
    lead = 0.5 * T_window
    t_start = max(0.0, travel - lead)
    if not c_max and not c0 and t_start > 0.0:
        # With no stamped speed at all the anchor is the 1500 m/s
        # default, which bounds nothing: a fast seabed (head waves at
        # 2-6 km/s) puts the earliest arrival well before r/1500, past
        # any lead the window can offer.
        warnings.warn(
            f"{who}: the model stamped no sound speed ('c_max'/"
            f"'c0' absent from metadata), so the window is anchored on "
            f"the {DEFAULT_SOUND_SPEED:g} m/s default at t_start="
            f"{t_start:.3g}s. Any path faster than that (e.g. a head "
            f"wave in a fast seabed) arrives before the window and "
            f"wraps to the end of the record. Pass t_start= to pin the "
            f"window start.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    # With only a c0 the anchor carries the fast/slow path spread as
    # error. A 5 % spread is representative of an ocean waveguide; when
    # it exceeds the lead the first arrival can fall before the window
    # and wrap to the end of the record.
    #
    # Where the 5 % lives, exactly: c0 is a SURFACE sample, so it is the
    # fastest water speed only on a downward-refracting profile. The
    # window wraps when r(1/c0 - 1/c_fast) > T/2, and this test fires when
    # 0.05·r/c0 > T/2, so it covers the wrap iff c0 >= 0.95·c_fast — i.e.
    # while the surface sample is within 5 % of the profile maximum. An
    # upward-refracting column with a wider spread (a cold surface over a
    # deep sound channel) leaves a band of window lengths uncovered.
    elif not c_max and t_start > 0.0 and 0.05 * travel > lead:
        warnings.warn(
            f"{who}: the {T_window:.3g}s synthesis window is "
            f"short against the {travel:.3g}s travel time at "
            f"{actual_range:.0f} m, and the model reported no maximum "
            f"sound speed, so the window start is an estimate — the "
            f"earliest arrival may fall before it and wrap to the end of "
            f"the record. Pass t_start= to pin it, or refine the "
            f"frequency grid (the window is 1/Δf).",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    return t_start


def _ifft_to_trace(
    tf: "Field",
    *,
    depth: Optional[float],
    range: Optional[float],
    source_spectrum: Optional[np.ndarray],
    window: str,
    nfft: Optional[int],
    t_start: Optional[float],
    sample_rate: Optional[float] = None,
    who: str = 'to_time_trace',
) -> "Field":
    """IFFT one (depth, range) cell of a broadband Field → time-domain trace Field.

    Evaluates the Fourier synthesis ``p(t) = 2·Re Σ H(f_k)·S(f_k)·
    e^{2πi f_k t}·df`` — a Riemann sum of the continuous inverse
    transform, so the amplitude is independent of ``nfft`` and of the
    bin grid. ``source_spectrum`` must therefore be the *continuous*
    source spectrum sampled at the Field frequencies (a raw DFT times
    the source sampling interval); ``None`` synthesizes the
    band-limited impulse response.

    Places each model frequency at bin ``round(f / Δf)`` with
    ``Δf = f[1] - f[0]``, so the record length is exactly ``1/Δf`` — a longer
    record requires a finer frequency grid, not a larger ``nfft``. The
    frequency axis must therefore be uniformly spaced and ascending. A grid
    whose first bin is not itself a multiple of ``Δf`` lands offset by a
    common ``|δ| <= Δf/2``; the synthesis de-rotates the complex sum by
    ``exp(-2πiδt)``, which recovers the requested band exactly rather than a
    frequency-shifted copy of it. An auto-sized ``nfft`` always keeps the
    largest data bin below Nyquist; an explicit ``nfft`` that would not is
    rejected.
    """
    data = tf.data                                # (n_d, n_r, n_f)
    depths = tf.coords['depth']
    ranges = tf.coords['range']
    n_d, n_r, _ = data.shape

    freqs, df, bin_indices, bin_offset_hz, nfft, win = _synthesis_plan(
        tf, window=window, nfft=nfft, sample_rate=sample_rate, who=who)

    d_idx = (_nearest_index_on_axis(depths, depth, 'depth')
             if depth is not None else n_d // 2)
    r_idx = (_nearest_index_on_axis(ranges, range, 'range')
             if range is not None else 0)
    actual_depth = float(depths[d_idx])
    actual_range = float(ranges[r_idx])

    spectra = data[d_idx, r_idx, :][None, :]
    _warn_unsolved_bins(spectra, cell_depths=np.array([actual_depth]),
                        cell_ranges=np.array([actual_range]), who=who)

    dt = 1.0 / (nfft * df)

    if t_start is None:
        t_start = _estimate_t_start(tf, actual_range, nfft * dt, who)

    traces, time = _synthesize_traces(
        spectra, freqs=freqs, win=win, source_spectrum=source_spectrum,
        bin_indices=bin_indices, bin_offset_hz=bin_offset_hz, nfft=nfft,
        df=df, t_start=t_start)

    return Field(
        data=traces[0],
        coords={'time': time},
        # The parent's pinned axes carry through (the accumulation contract
        # in the class doc), with this cell's coordinates added on top.
        pinned={**dict(tf.pinned),
                'depth': actual_depth, 'range': actual_range},
        model=tf.model,
        backend=tf.backend,
        source_depths=tf.source_depths,
        frequencies=tf.frequencies,
        # The payload is p(t) from here on, whatever convention H(f) carried.
        phase_reference=PhaseReference.TIME_DOMAIN_NATIVE,
        model_source=tf.model_source,
        # Carry the source Field's metadata forward (output paths attached
        # under a pinned work_dir, c0/c_min, …) — every other derived-Field
        # path (slices, id_kwargs clones) preserves it; synthesis must too.
        metadata={**dict(tf.metadata),
                  'window': window, 'source_model': tf.model},
    )


def _synthesize_time_series(
    tf: "Field",
    *,
    source_waveform: np.ndarray,
    sample_rate: float,
    t_start: Optional[float],
    window: str,
    nfft: Optional[int],
) -> "Field":
    """Convolve every grid cell of a broadband Field with a source waveform.

    Output: a time-domain Field with ``coords={'depth', 'range', 'time'}``.
    ``nfft`` is sized so the output sample rate ``1/dt = nfft·df`` is at
    least ``sample_rate`` (rounded up to a power of two, so up to 2×
    finer); read the actual grid from ``coords['time']``. Amplitude is
    grid-independent: a flat ``H ≡ 1`` reproduces the (band-limited)
    source waveform.
    """
    wf = np.asarray(source_waveform, dtype=float).ravel()
    n_src = len(wf)
    if n_src < 2:
        raise ConfigurationError(
            f"_synthesize_time_series: source_waveform must have at least "
            f"2 samples; got {n_src}"
        )
    # NaN-closed (``not (sr > 0)``, not ``sr <= 0``): nan compares False
    # against both bounds, so the plain inequality passes it through to
    # int(nfft), which raises a raw ValueError instead of this typed one.
    if not np.isfinite(sample_rate) or not (sample_rate > 0):
        raise ConfigurationError(
            f"_synthesize_time_series: sample_rate must be positive and "
            f"finite; got {sample_rate}"
        )

    tf_freqs = np.asarray(tf.coords['frequency'], dtype=float)
    if tf_freqs.size > 1:
        df_tf = float(np.diff(tf_freqs).mean())
        t_dft = 1.0 / df_tf if df_tf > 0 else float('inf')
        t_dur = n_src / float(sample_rate)
        # Catches a caller-supplied grid coarser than the pulse itself: the
        # record 1/Δf then cannot even hold the source waveform. It cannot
        # fire on a grid derived from the waveform (Δf = fs/n makes 1/Δf the
        # pulse length exactly), and it says nothing about the CHANNEL: how
        # long the arrivals ring is not visible in H(f) sampled every Δf,
        # and a late arrival folds to a fixed place mid-record, leaving the
        # record's end silent — so that check lives where the arrivals are
        # (``Arrivals.synthesis_band``; Bellhop's BROADBAND run on its
        # default grid), by count and level. One-sample tolerance: float
        # roundoff in Δf can make t_dft and t_dur evaluate as < when they
        # should be ==.
        if t_dft < t_dur - 1.0 / float(sample_rate):
            warnings.warn(
                f"synthesize_time_series: DFT period 1/Δf = {t_dft:.4f}s "
                f"is shorter than the source-waveform duration "
                f"{t_dur:.4f}s — the late-time response wraps back into "
                f"early bins. Refine the frequency grid to Δf ≤ "
                f"{1.0/t_dur:.4g} Hz, or shorten the waveform.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )

    # Refuse a non-uniform axis HERE, not where the plan below reaches it:
    # the source spectrum is evaluated on this axis first, and its chirp-z
    # contour assumes exactly the grid the plan's guard describes. (With
    # fewer than 2 frequencies there is no spacing to check; the plan raises
    # on the count itself.)
    if tf_freqs.size > 1:
        from uacpy.acoustic_signal.system import uniform_frequency_step
        uniform_frequency_step(tf_freqs, 'synthesize_time_series')

    freqs = tf.coords['frequency']
    from uacpy.acoustic_signal.system import (
        waveform_spectrum_at as _source_spectrum_at)
    source_spectrum = _source_spectrum_at(wf, sample_rate, freqs)

    n_d, n_r, n_f = tf.data.shape
    depths = np.asarray(tf.coords['depth'])
    ranges = np.asarray(tf.coords['range'])

    plan_freqs, df, bin_indices, bin_offset_hz, nfft, win = _synthesis_plan(
        tf, window=window, nfft=nfft, sample_rate=sample_rate,
        who='synthesize_time_series')

    if t_start is None:
        # One window for every cell, anchored on the nearest cell's range;
        # the record is 1/Δf long whatever nfft is.
        t_start = _estimate_t_start(tf, float(ranges[0]), 1.0 / df,
                                    'synthesize_time_series')

    # Every cell shares one synthesis grid, so one batched ifft per chunk of
    # cells replaces a per-cell transform. Chunk over the flattened
    # (depth, range) cell axis so the (cells × nfft) complex scratch stays
    # bounded (~4M elements ≈ 64 MB) however large the field is.
    n_cells = n_d * n_r
    spectra = tf.data.reshape(n_cells, n_f)
    out = np.empty((n_cells, nfft), dtype=np.float64)
    time_vec = None
    chunk = max(1, 4_000_000 // nfft)
    for a in range(0, n_cells, chunk):
        idx = np.arange(a, min(a + chunk, n_cells))
        _warn_unsolved_bins(
            spectra[idx],
            cell_depths=depths[idx // n_r],
            cell_ranges=ranges[idx % n_r],
            who='synthesize_time_series',
        )
        traces, time_vec = _synthesize_traces(
            spectra[idx], freqs=plan_freqs, win=win,
            source_spectrum=source_spectrum, bin_indices=bin_indices,
            bin_offset_hz=bin_offset_hz, nfft=nfft, df=df, t_start=t_start)
        out[idx] = traces
    out = out.reshape(n_d, n_r, nfft)

    # All cells share one time window anchored at (depths[0], ranges[0]);
    # arrivals for ranges further out than the window can hold wrap back
    # into early bins (DFT periodicity) — flag it rather than alias silently.
    if n_r > 1 and time_vec is not None and time_vec.size > 1:
        c0 = float(tf.metadata.get('c0') or DEFAULT_SOUND_SPEED)
        span_s = float(ranges.max() - ranges.min()) / c0
        window_s = float(time_vec[-1] - time_vec[0])
        if span_s > window_s:
            warnings.warn(
                f"synthesize_time_series: the receiver range span "
                f"({ranges.max() - ranges.min():.0f} m ≈ {span_s:.2f}s of "
                f"travel time) exceeds the {window_s:.2f}s synthesis window "
                f"— far-range arrivals wrap back into early bins. The window "
                f"is 1/Δf, so widen it with a frequency grid of "
                f"Δf ≤ {1.0/span_s:.3g} Hz; on a TIME_SERIES run "
                f"output_duration ≥ {span_s:.2f}s sets that grid for you "
                f"(BROADBAND takes the grid from frequencies= and ignores "
                f"output_duration).",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )

    return Field(
        data=out,
        coords={'depth': depths, 'range': ranges, 'time': time_vec},
        # The parent's pinned axes carry through (the accumulation contract
        # in the class doc); no axis collapses here, so nothing is added.
        pinned=dict(tf.pinned),
        model=tf.model,
        backend=tf.backend,
        source_depths=tf.source_depths,
        frequencies=tf.frequencies,
        # The payload is p(t) from here on, whatever convention H(f) carried.
        phase_reference=PhaseReference.TIME_DOMAIN_NATIVE,
        model_source=tf.model_source,
        # Carry the source Field's metadata forward (see _ifft_to_trace).
        metadata={**dict(tf.metadata),
                  'source_waveform_sample_rate': sample_rate,
                  'window': window,
                  'source_model': tf.model},
    )
