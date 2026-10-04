"""The unified :class:`Field` result."""

from __future__ import annotations

import json
import warnings
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union

from uacpy.core._validate import (
    require_finite, reject_complex, steps_are_uniform,
)
from uacpy.acoustic_signal._synthesis import (check_source_spectrum,
                                              require_source_waveform,
                                              waveform_spectrum_on)
from uacpy.core.constants import (DEFAULT_SOUND_SPEED,
                                  REFERENCE_PRESSURE_WATER)
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, NumericsWarning,
)
from uacpy.core._plotting import plotter
from uacpy.core._grid import coarse_axes, nearest_index_on_axis, collapse_axis
from uacpy.core.bathymetry import mask_below_seafloor
from uacpy.core.environment import Bathymetry, Environment
from uacpy.core._warn_frames import USER_FRAME_SKIP

from uacpy.core.acoustics.levels import (
    peak_level as _peak_level,
    received_level_dB as _received_level_dB,
    sound_exposure_level as _sound_exposure_level,
    transmission_loss_dB,
)
from uacpy.core.results import quantities as _quantities
from uacpy.core._export import (units_attrs, encode_attrs, join_complex,
                                read_only)
from uacpy.core.results._base import (Result, _integer_index,
                                     _window_pair, axis_match_tolerance,
                                     coordinate_axis)
from uacpy.core._repr import build, qty
from uacpy.core.results.speeds import SoundSpeeds



def _least_sample_count(n) -> int:
    """The least whole number of samples that covers ``n``.

    A producer may report its time-sample count as a real number (mpiramS
    writes ``Nsam = fs*T``, ``peramx.f90:356``), so ``int()`` could drop a
    sample from a count computed a few ULP below a whole number. The ceiling
    keeps every sample; the relative tolerance keeps a count computed a few
    ULP above a whole number (``1018.000000000011``) at that number.
    """
    x = float(n)
    return int(np.ceil(x - 1e-9 * max(1.0, abs(x))))

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

    What the numbers are — the quantity and its unit — is given by
    ``kind=`` and ``unit=`` at construction (``Field(data=…, coords=…,
    kind='level', unit='dB')``), one of the names registered in
    :mod:`uacpy.core.results.quantities`. They are attributes of the Field,
    with :attr:`coherent` and, for a surface in dB re one of its own values,
    :attr:`reference` / :attr:`reference_unit`: set when the Field is built
    and carried by :meth:`replace` to every Field derived from it, never
    re-read from its storage. A ``metadata`` carrying one of them is refused. The unit is
    the ``unit=`` given, or read off the table above.

    Auxiliary coordinates
    ---------------------
    :attr:`aux_coords` holds labels that run along an axis without being one:
    ``aux_coords={'receiver_depth': ('range', depths)}`` gives each sample
    of an irregular receiver line its depth. An entry follows its axis:
    :meth:`window` and :meth:`reindex` narrow or widen it with the axis, and
    a slice or reduction that drops the axis, or relabels it, drops it.
    :meth:`to_xarray` writes each as a non-dimension coordinate.

    Derivation record
    -----------------
    What a derivation records about how the Field was made, and a
    reader of the Field acts on, is held as attributes:
    :attr:`band_hz` (the band a reducer collapsed onto one pinned
    centroid, or the pulse band SPARC marched), :attr:`synthesis_window`
    (the band window a trace was synthesised through),
    :attr:`sub_cutoff_bins` (the leading bins below a normal-mode
    model's cutoff), :attr:`sonar_budget` (the budget a signal-excess
    map was built with) and :attr:`sigma_dB` (the fluctuation spread
    of a detection-probability map). They are set when the Field is
    built and carried by :meth:`replace`; a ``metadata`` carrying one
    of them (or ``window``, the synthesis window's metadata spelling)
    is refused.

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

    #: The quantity attributes: what the data is, and how to read it. Set
    #: once in ``__init__``, carried by :meth:`id_kwargs`, never metadata.
    _QUANTITY = ('kind', 'unit', 'coherent', 'reference', 'reference_unit')
    #: What the time-series synthesis reads off the Field: the medium's
    #: named speeds and the FFT length its producer floors it at.
    _SYNTHESIS = ('speeds', 'synthesis_floor')
    #: The metadata spellings of the :attr:`speeds` members and of
    #: :attr:`synthesis_floor`: a file that keeps them in its metadata loads
    #: them, and a ``metadata=`` carrying one is refused.
    _METADATA_SPEEDS = {'c0': 'surface', 'c_max': 'water_max',
                        'tdelay_speed': 'water_min',
                        'waveguide_c_min': 'waveguide_min',
                        'waveguide_c_max': 'waveguide_max'}
    _METADATA_FLOOR = 'n_time_samples'
    #: The derivation record (see the class doc): set once in
    #: ``__init__``, carried by :meth:`replace`, never metadata.
    _DERIVATION = ('band_hz', 'synthesis_window', 'sub_cutoff_bins',
                   'sonar_budget', 'sigma_dB')
    #: The metadata spelling of a derivation attribute whose name
    #: differs from it: a file that keeps it in its metadata loads it,
    #: and a ``metadata=`` carrying it is refused.
    _METADATA_DERIVATION = {'window': 'synthesis_window'}

    def __init__(
        self,
        *,
        data: np.ndarray,
        coords: Dict[str, np.ndarray],
        pinned: Optional[Dict[str, float]] = None,
        aux_coords: Optional[Dict[str, Tuple[str, np.ndarray]]] = None,
        kind: Optional[str] = None,
        unit: Optional[str] = None,
        coherent: Optional[bool] = None,
        reference: Optional[float] = None,
        reference_unit: Optional[str] = None,
        speeds: Optional[SoundSpeeds] = None,
        synthesis_floor: Optional[int] = None,
        band_hz: Optional[Tuple[float, float]] = None,
        synthesis_window: Optional[str] = None,
        sub_cutoff_bins: Optional[int] = None,
        sonar_budget: Optional[Dict[str, Any]] = None,
        sigma_dB: Optional[float] = None,
        **kwargs,
    ):
        # The quantity, the synthesis inputs and the derivation record are
        # this Field's own attributes, decided here once; a metadata entry
        # naming one would be a second decider.
        meta_keys = set(kwargs.get('metadata') or {})
        carried = sorted(meta_keys & set(self._QUANTITY + self._SYNTHESIS
                                         + self._DERIVATION))
        if carried:
            raise ConfigurationError(
                f"Field: metadata carries {carried}, which are attributes "
                f"of the Field, not metadata.",
                remediation="Pass them as keywords: Field(..., "
                            + ", ".join(f"{t}=..." for t in carried) + ").")
        spelled = sorted(meta_keys & (set(self._METADATA_SPEEDS)
                                      | {self._METADATA_FLOOR}))
        if spelled:
            members = [self._METADATA_SPEEDS[t] for t in spelled
                       if t in self._METADATA_SPEEDS]
            raise ConfigurationError(
                f"Field: metadata carries {spelled}, which the time-series "
                f"synthesis reads from the Field's speeds and "
                "synthesis_floor, not from metadata.",
                remediation="Pass them as keywords: Field(..., "
                            + ", ".join(
                                (["speeds=SoundSpeeds("
                                  + ", ".join(f"{m}=..." for m in members)
                                  + ")"] if members else [])
                                + (["synthesis_floor=..."]
                                   if self._METADATA_FLOOR in spelled
                                   else [])) + ").")
        renamed = sorted(meta_keys & set(self._METADATA_DERIVATION))
        if renamed:
            raise ConfigurationError(
                f"Field: metadata carries {renamed}, which the Field holds "
                f"as its own attributes, not as metadata.",
                remediation="Pass them as keywords: Field(..., "
                            + ", ".join(
                                f"{self._METADATA_DERIVATION[t]}=..."
                                for t in renamed) + ").")
        if band_hz is not None:
            band_hz = tuple(float(v) for v in np.ravel(band_hz))
            if len(band_hz) != 2:
                raise ConfigurationError(
                    f"Field: band_hz={band_hz!r} is not a (low, high) "
                    f"pair of frequencies in Hz.")
        if speeds is not None and not isinstance(speeds, SoundSpeeds):
            raise ConfigurationError(
                f"Field: speeds={speeds!r} is not a SoundSpeeds record.",
                remediation="Pass speeds=SoundSpeeds(surface=..., ...).")
        super().__init__(**kwargs)
        self._speeds = speeds
        self._synthesis_floor = (None if synthesis_floor is None
                                 else _least_sample_count(synthesis_floor))
        self._band_hz = band_hz
        self._synthesis_window = (None if synthesis_window is None
                                  else str(synthesis_window))
        self._sub_cutoff_bins = (None if sub_cutoff_bins is None
                                 else int(sub_cutoff_bins))
        # Copied on ingest, so the caller's dict never aliases it.
        self._sonar_budget = (None if sonar_budget is None
                              else dict(sonar_budget))
        self._sigma_dB = None if sigma_dB is None else float(sigma_dB)
        self._kind = str(kind) if kind else 'pressure'
        if not isinstance(coords, dict):
            raise ConfigurationError(
                "Field.coords: must be a dict of axis_name → 1-D array."
            )
        normalised: Dict[str, np.ndarray] = {}
        for name, v in coords.items():
            # Ahead of the float64 cast below, which discards an imaginary
            # part — see reject_complex for the two ways it does it. A
            # complex coordinate is the axis, not the data: :attr:`data` is
            # complex on every pressure field, but at() and the slicers read
            # the coords as real distances.
            reject_complex(v, f"Field.coords[{name!r}]")
            # np.array (not asarray) so each Field owns its coord vectors —
            # slices/derived Fields never alias a parent's (or caller's) arrays.
            arr = np.atleast_1d(np.array(v, dtype=float))
            if arr.ndim != 1:
                raise ConfigurationError(
                    f"Field.coords[{name!r}]: must be 1-D; got shape {arr.shape}."
                )
            # A NaN/inf coordinate makes every |axis - label| distance on
            # this axis NaN at that sample, so at()'s argmin can land on it
            # and hand back a sample no label ever named.
            require_finite(arr, f"Field.coords[{name!r}]")
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
        self.aux_coords: Dict[str, Tuple[str, np.ndarray]] = (
            self._normalise_aux_coords(aux_coords))
        # The unit is decided here, once: the one given, else the storage
        # rule applied to the data and axes this Field is built with. Every
        # derived Field is given it (:meth:`replace`), so a slice inherits
        # its parent's unit instead of re-deriving it from what the slice has
        # left — a time trace sliced at one instant has no time axis, and
        # re-derived it read as TL in dB.
        self._unit = str(unit) if unit else self._storage_unit()
        # Coherence is decided here too, once, for a pressure field: the
        # producer's answer when it gave one (a model run stamps its declared
        # ``outputs[mode]`` answer), else the one this Field's axes and
        # identity give now. A slice inherits the answer: a snapshot of a
        # trace has no time axis left and, re-decided, read as a coherent
        # frequency-domain field. Every other kind has no answer.
        if self._kind != 'pressure':
            self._coherent = None
        elif coherent is not None:
            self._coherent = bool(coherent)
        else:
            self._coherent = self._storage_coherence()
        self._reference = None if reference is None else float(reference)
        self._reference_unit = (None if reference_unit is None
                                else str(reference_unit))
        # Validate the quantity pair where it enters, not where it is read: a
        # typo'd kind that survives construction resurfaces as a wrong colour
        # scale or a wrong argmax direction, with nothing pointing back here.
        _quantities.label(self.kind, self.unit)

    def _normalise_aux_coords(self, aux_coords):
        """``aux_coords`` checked against the axes: each entry ``name ->
        (dim, values)`` is a real 1-D label array along the axis ``dim``, one
        value per sample of it, under a name that is neither an axis nor
        pinned. A label may be NaN (not known at that sample, as after
        :meth:`reindex`) but never infinite. Stored as its own float copy."""
        out: Dict[str, Tuple[str, np.ndarray]] = {}
        for name, entry in (aux_coords or {}).items():
            where = f"Field.aux_coords[{name!r}]"
            try:
                dim, values = entry
            except (TypeError, ValueError):
                raise ConfigurationError(
                    f"{where}: must be a (dim, values) pair; got {entry!r}."
                ) from None
            if dim not in self.coords:
                raise ConfigurationError(
                    f"{where}: runs along {dim!r}, which is not an axis of "
                    f"this field ({list(self.coords)}).")
            if name in self.coords or name in self.pinned:
                raise ConfigurationError(
                    f"{where}: {name!r} already names an axis or a pinned "
                    f"coordinate of this field.")
            reject_complex(values, where)
            arr = np.atleast_1d(np.array(values, dtype=float))
            if arr.shape != self.coords[dim].shape:
                raise ConfigurationError(
                    f"{where}: has shape {arr.shape} but the {dim!r} axis has "
                    f"{self.coords[dim].size} samples; it is one label per "
                    f"sample of that axis.")
            if np.any(np.isinf(arr)):
                raise ConfigurationError(
                    f"{where}: holds an infinite label; a label is a finite "
                    f"value, or NaN where it is not known.")
            out[str(name)] = (str(dim), arr)
        return out

    def _aux_coords_on(self, coords, keep=None):
        """The auxiliary coordinates that stay valid on ``coords``: those
        whose axis is still there with the same labels, plus those along an
        axis narrowed by the boolean mask ``keep[dim]`` (sliced with it)."""
        keep = keep or {}
        out = {}
        for name, (dim, values) in self.aux_coords.items():
            if dim in keep:
                out[name] = (dim, values[keep[dim]])
            elif dim in coords and np.array_equal(coords[dim],
                                                  self.coords[dim]):
                out[name] = (dim, values)
        return out

    def replace(self, **changes) -> "Field":
        """A new Field like this one, with ``changes`` applied.

        ``changes`` names any of ``data``, ``coords``, ``pinned``,
        ``aux_coords``, the quantity (``kind``, ``unit``, ``coherent``,
        ``reference``, ``reference_unit``; :meth:`_quantity`), the
        synthesis inputs (``speeds``, ``synthesis_floor``), the derivation
        record (``band_hz``, ``synthesis_window``, ``sub_cutoff_bins``,
        ``sonar_budget``, ``sigma_dB``) and the identity
        fields (``model``, ``backend``, ``source_depths``, ``frequencies``,
        ``phase_reference``, ``model_source``, ``run_mode``,
        ``source_level_dB``, ``source_weights``, ``metadata``,
        ``run_settings``) and ``components``; everything not named carries
        over, :attr:`components` only while the quantity's ``kind`` is
        unchanged. The result
        is built by the constructor, so it passes every check a new Field
        does; ``coherent=None`` decides the coherence anew. Auxiliary
        coordinates follow their axis: one whose axis ``coords=`` drops or
        relabels is dropped, unless ``aux_coords=`` states the new set.

        Parameters
        ----------
        **changes
            The fields to change (see above).
        """
        fields = {'data': self.data, 'coords': self.coords,
                  'pinned': self.pinned, **self.id_kwargs(),
                  **self._quantity(), 'speeds': self._speeds,
                  'synthesis_floor': self._synthesis_floor,
                  **self._derivation(),
                  'components': None}
        allowed = set(fields) | {'aux_coords'}
        unknown = sorted(set(changes) - allowed)
        if unknown:
            raise ConfigurationError(
                f"Field.replace: {unknown} name no field of a Field; the "
                f"names are {sorted(allowed)}.")
        fields.update(changes)
        # The same quantity keeps the results it was built from, as the same
        # objects; a derivation to another quantity starts without them.
        if 'components' not in changes and fields['kind'] == self.kind:
            fields['components'] = self.components
        if 'aux_coords' not in changes:
            fields['aux_coords'] = self._aux_coords_on(fields['coords'])
        return Field(**fields)

    def _quantity(self) -> dict:
        """The quantity (``kind``, ``unit``, ``coherent``, ``reference``,
        ``reference_unit``) as constructor keywords: what :meth:`replace`
        carries to a derived Field, so it inherits the quantity; a derivation
        that changes it states the new one."""
        return dict(kind=self._kind, unit=self._unit, coherent=self._coherent,
                    reference=self._reference,
                    reference_unit=self._reference_unit)

    def _derivation(self) -> dict:
        """The derivation record (``band_hz``, ``synthesis_window``,
        ``sub_cutoff_bins``, ``sonar_budget``, ``sigma_dB``) as constructor
        keywords: what :meth:`replace` carries to a derived Field."""
        return {tag: getattr(self, f'_{tag}') for tag in self._DERIVATION}

    def _storage_unit(self) -> str:
        """The unit this Field's storage implies, for a Field built with no
        unit tag: a kind with one registered unit has that unit; pressure —
        the one kind with two — is linear Pa when the data is complex or
        carries a ``time`` axis, and a level in dB otherwise."""
        units = _quantities.quantity(self.kind).units
        if len(units) == 1:
            return next(iter(units))
        return 'Pa' if (self.is_complex or 'time' in self.coords) else 'dB'

    def _storage_coherence(self) -> Optional[bool]:
        """The coherence a pressure Field built with no ``coherent`` tag
        implies: no answer for a time-domain trace (a ``time`` axis, or one
        pinned), else ``True`` when it carries a :attr:`phase_reference` or
        came from a ``COHERENT_TL`` run, and ``False`` otherwise."""
        if 'time' in self.coords or 'time' in self.pinned:
            return None
        return (self.phase_reference is not None
                or self.run_mode == 'coherent_tl')

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

        ``'pressure'`` by default; a producer of something else builds the
        Field with ``kind=`` (e.g. ``'reverberation'``). This is one of three
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
        return self._kind

    @property
    def unit(self) -> str:
        """What :attr:`data` is measured in — ``'Pa'`` or ``'dB'`` for
        pressure, the registered unit for every other kind.

        Decided once, when the Field is built: the ``unit=`` given, else
        read off the storage —
        complex data and time-domain traces are linear pressure, real
        frequency-domain data is a level in dB. Every Field derived from this
        one (a slice, :meth:`max`, :meth:`window`, a stack sum) inherits it,
        so a pressure trace sliced at one instant is still in Pa. Real linear
        magnitudes built by hand therefore need ``unit='Pa'`` at
        construction; untagged, a real map without a time axis is read as TL.

        Consumers that need to know which way is louder must ask **this** and
        never :attr:`kind`, or every new dB quantity silently inverts them —
        see :meth:`max`.
        """
        return self._unit

    @property
    def reference(self) -> Optional[float]:
        """The linear value 0 dB refers to, when this Field's dB values
        are relative to one of its own — ``None`` otherwise.

        An ambiguity surface is in dB re its peak, and the peak is a power
        the processor computed (a ``Covariance`` surface in the
        covariance's unit, a trace-normalised Bartlett surface a number up
        to 1). It is kept here, in :attr:`reference_unit`, so
        ``reference * 10**(data / 10)`` is the linear surface again.
        Recorded when the Field is built
        (:func:`~uacpy.core.results.ambiguity_field`) and inherited by every
        Field derived from this one, as :attr:`unit` is.
        """
        return self._reference

    @property
    def reference_unit(self) -> Optional[str]:
        """The unit :attr:`reference` is in (``'1'`` for a normalised
        surface, ``''`` for a covariance that records no unit), or
        ``None`` when there is no reference."""
        return self._reference_unit

    @property
    def coherent(self) -> Optional[bool]:
        """Whether this pressure field is a coherent sum over paths — the
        field whose TL carries interference fringes.

        Decided once, when the Field is built, as :attr:`unit` is: the
        producer's answer
        (a model run stamps the ``coherent`` its ``outputs[mode]``
        declares), else, for a frequency-domain pressure field, ``True`` when
        it carries a :attr:`phase_reference` or came from a ``COHERENT_TL``
        run, and ``False`` otherwise. OAST's coherent TL is stored as real dB
        with the run mode as its only record, and a ``BROADBAND`` result
        sliced at one frequency is the same coherent pressure under another
        run mode, so neither the dtype nor the run mode alone answers it.
        Every Field derived from this one (a slice, :meth:`max`,
        :meth:`to_dB`) inherits the stored answer. ``None`` for a
        time-domain trace and every snapshot of one, and for every kind
        other than pressure, where the question does not arise.
        """
        return self._coherent

    @property
    def speeds(self) -> Optional[SoundSpeeds]:
        """The named sound speeds of the medium this Field was computed in
        (:class:`~uacpy.core.results.SoundSpeeds`): those its producer
        stated, with ``waveguide_min`` / ``waveguide_max`` taken from
        :attr:`run_settings` where it states none. ``None`` when neither
        states any. :meth:`to_time_trace` and
        :meth:`synthesize_time_series` place their window with them."""
        settings = self.run_settings
        waveguide = (None if settings is None
                     else getattr(settings, 'waveguide', None))
        if self._speeds is None:
            return (None if waveguide is None
                    else SoundSpeeds().with_waveguide(waveguide))
        return self._speeds.with_waveguide(waveguide)

    @property
    def synthesis_floor(self) -> Optional[int]:
        """The FFT length the time-series synthesis floors its own at: the
        least whole number of samples covering the time-sample count the
        producer's own transform used (mpiramS's ``Nsam``, OASP's ``NX``,
        OASSP's ``NT``), or ``None``."""
        return self._synthesis_floor

    @property
    def band_hz(self) -> Optional[Tuple[float, float]]:
        """The frequency band ``(low, high)`` in Hz the field stands for
        where its frequency identity cannot say it: the band a reducer
        (:meth:`broadband_loss`, :meth:`sound_exposure_level`, ...)
        collapsed onto one pinned centroid, or the pulse band a
        time-marching run (SPARC) marched; ``None`` otherwise. A plotter
        captions a pinned frequency with it as a band average."""
        return self._band_hz

    @property
    def synthesis_window(self) -> Optional[str]:
        """The spectral window the time-series synthesis applied across
        the band of ``H(f)`` before the IFFT (``'hann'``, ``'hamming'``,
        ``'blackman'``, ``'tukey'``), or ``None``: no window, or not a
        synthesised trace. :meth:`to_transfer_function` warns when it is
        set, since the spectrum it returns still carries the window."""
        return self._synthesis_window

    @property
    def sub_cutoff_bins(self) -> Optional[int]:
        """The count of leading frequency bins below a normal-mode
        model's lowest modal cutoff (Kraken's band route), which the
        field holds as NaN, or ``None`` when its producer states none.
        The synthesis names the cutoff in its warning, and a
        ``TIME_SERIES`` run takes those bins as zero."""
        return self._sub_cutoff_bins

    @property
    def sonar_budget(self) -> Optional[Dict[str, Any]]:
        """The sonar budget a signal-excess map was built with, as
        ``SonarBudget.to_dict()`` writes it (``mode`` and every term),
        carried onto the maps derived from it; ``None`` on any other
        field. ``SonarBudget.from_dict(field.sonar_budget)`` rebuilds
        the budget. A copy: the field's own record cannot be edited
        through it."""
        return None if self._sonar_budget is None else dict(
            self._sonar_budget)

    @property
    def sigma_dB(self) -> Optional[float]:
        """The fluctuation spread sigma (dB) a detection-probability map
        was computed with
        (:func:`~uacpy.sonar.transition_probability_field`); ``None`` on
        any other field."""
        return self._sigma_dB

    @classmethod
    def _synthesis_from_metadata(cls, metadata, speeds, floor):
        """``(speeds, synthesis_floor)``: those given, else the ones
        ``metadata`` holds under their metadata spellings
        (:attr:`_METADATA_SPEEDS`, ``n_time_samples``), which are popped
        from it either way."""
        named = {}
        for key, member in cls._METADATA_SPEEDS.items():
            value = metadata.pop(key, None)
            if value is not None:
                named[member] = float(value)
        stated_floor = metadata.pop(cls._METADATA_FLOOR, None)
        if speeds is None and named:
            speeds = SoundSpeeds(**named)
        if floor is None:
            floor = stated_floor
        return speeds, floor

    @classmethod
    def _derivation_from(cls, metadata, stated):
        """The derivation keywords: each one ``stated`` holds, else the
        one ``metadata`` keeps under its own name or its metadata
        spelling (:attr:`_METADATA_DERIVATION`), which are popped from
        it either way. A budget held as a JSON string (the form
        :meth:`to_xarray` writes) is parsed."""
        spelling = {tag: key
                    for key, tag in cls._METADATA_DERIVATION.items()}
        out = {}
        for tag in cls._DERIVATION:
            kept = metadata.pop(tag, None)
            if tag in spelling:
                legacy = metadata.pop(spelling[tag], None)
                kept = legacy if kept is None else kept
            value = stated.get(tag)
            out[tag] = kept if value is None else value
        if isinstance(out['sonar_budget'], str):
            out['sonar_budget'] = json.loads(out['sonar_budget'])
        return out

    # ── persistence ───────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialise this field to a plain dict for caching / round-trip.

        Values are numpy arrays, Python scalars and three plain dicts
        (``coords``, ``pinned``, ``metadata``); data preserves its
        real/complex dtype. The result pickles directly. ``np.savez(f,
        **d)`` stores the dicts and ``None`` entries as pickled object
        arrays, so read it back with ``np.load(f, allow_pickle=True)`` and
        pass that mapping to :meth:`from_dict`, which unwraps them. Convert
        the arrays to lists yourself for JSON. ``coords`` insertion order
        matches the data axes. The quantity (``kind``, ``unit``,
        ``coherent``, ``reference``, ``reference_unit``), the synthesis
        inputs (``speeds`` as its plain dict, ``synthesis_floor``) and the
        derivation record (``band_hz``, ``synthesis_window``,
        ``sub_cutoff_bins``, ``sonar_budget``, ``sigma_dB``) are
        written at the top level. A field with auxiliary coordinates adds
        ``aux_coords``
        (``name -> (dim, values)``). Reconstruct with ``Field.from_dict(d)``.
        """
        out = {
            'kind': self.kind,
            'unit': self.unit,
            'coherent': self.coherent,
            'reference': self.reference,
            'reference_unit': self.reference_unit,
            'speeds': (None if self._speeds is None
                       else self._speeds.to_dict()),
            'synthesis_floor': self._synthesis_floor,
            **self._derivation(),
            'data': self.data.copy(),
            'coords': {k: v.copy() for k, v in self.coords.items()},
            'pinned': dict(self.pinned),
            **self._identity_dict(),
        }
        if self.aux_coords:
            out['aux_coords'] = {name: (dim, values.copy()) for name, (dim, values)
                                 in self.aux_coords.items()}
        return out

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Field':
        """Reconstruct a :class:`Field` from :meth:`to_dict` output.

        Also takes the mapping ``np.load(f, allow_pickle=True)`` returns for
        a file written with ``np.savez(f, **field.to_dict())``: every entry
        but ``data`` arrives there as a 0-d array (a pickled dict, ``None``
        or string) and is unwrapped to the value it holds. ``data`` is left
        alone, since a fully pinned field's data is itself 0-d.

        Parameters
        ----------
        d : mapping
            :meth:`to_dict` output, or the mapping ``np.load`` returns for it.
        """
        d = cls._unwrap_saved(d, payload=('data',))
        identity = cls._identity_from_dict(d)
        # The quantity the field held, not one re-read from the data: read
        # from the top level, or from the metadata of a file that keeps it
        # there, which the constructor would refuse.
        metadata = dict(identity['metadata'] or {})
        quantity = {tag: metadata.pop(tag, d.get(tag))
                    for tag in cls._QUANTITY}
        speeds = d.get('speeds')
        speeds, floor = cls._synthesis_from_metadata(
            metadata, None if speeds is None else SoundSpeeds.from_dict(speeds),
            d.get('synthesis_floor'))
        derivation = cls._derivation_from(metadata, d)
        identity['metadata'] = metadata or None
        return cls(
            data=np.asarray(d['data']),
            coords={k: np.asarray(v) for k, v in d['coords'].items()},
            pinned=d.get('pinned') or None,
            aux_coords=d.get('aux_coords') or None,
            speeds=speeds, synthesis_floor=floor,
            **derivation,
            **quantity,
            **identity,
        )

    #: ``attrs`` keys :meth:`to_xarray` writes for the identity surface;
    #: every other attr is read back into ``metadata``.
    _XARRAY_IDENTITY_ATTRS = ('kind', 'units', 'unit', 'coherent',
                              'reference', 'reference_unit',
                              *Result._IDENTITY_ATTRS)

    def to_xarray(self):
        """This field as an ``xarray.DataArray`` (optional extra
        ``uacpy[xarray]``).

        ``dims`` are the :attr:`coords` names in axis order and each axis is
        a coordinate; every :attr:`pinned` axis becomes a scalar coordinate,
        which is how xarray records a selected label, and every
        :attr:`aux_coords` entry a non-dimension coordinate along its axis.
        ``attrs`` carry the
        quantity (``kind``, ``unit``, and ``coherent``, ``reference`` and
        ``reference_unit`` where they are set), the identity (``model``,
        ``backend``,
        ``phase_reference``, ``run_mode``, ``frequencies``, ``source_depths``, and the
        producing engine's id as ``model_source_id``; the settings the run
        used as the JSON record ``run_settings``, which reads back as
        :attr:`run_settings`, beside the one-line ``run_settings_summary``,
        which reads back as metadata), the synthesis inputs (each stated member of
        :attr:`speeds`, the run settings' waveguide included, as
        ``speeds_<member>``, and ``synthesis_floor``), the derivation record
        under its own names (:attr:`band_hz` as a real array named in
        ``attrs['tuple_attrs']``, :attr:`sonar_budget` as a JSON string,
        the others as themselves; each where it is set), the source identity
        (``source_level_dB``, and :attr:`source_weights` written as a complex
        entry is) and every
        :attr:`metadata` entry a NetCDF attribute can hold: a string,
        number, bool or real numeric array as itself; a complex number or
        array as a
        ``<key>_real`` / ``<key>_imag`` pair named in ``attrs['complex_attrs']``;
        a tuple or list of numbers as a real array
        named in ``attrs['tuple_attrs']``. :meth:`from_xarray` restores all
        three. Any other entry (a dict, an array of more than one dimension)
        is left out, with a warning naming it. The DataArray's
        ``name`` is the ``kind``, so ``to_dataset()`` works directly;
        :meth:`to_netcdf` writes complex data as real and imaginary parts
        any netCDF backend stores.
        """
        try:
            import xarray as xr
        except ImportError as exc:
            raise ConfigurationError(
                "Field.to_xarray: xarray is not installed.",
                remediation="pip install 'uacpy[xarray]'") from exc
        coords = {name: (name, np.asarray(v),
                         units_attrs(_quantities.coordinate_unit(name)))
                  for name, v in self.coords.items()}
        coords.update({name: (dim, np.asarray(values),
                              units_attrs(_quantities.coordinate_unit(name)))
                       for name, (dim, values) in self.aux_coords.items()})
        coords.update({name: ((), float(v),
                              units_attrs(_quantities.coordinate_unit(name)))
                       for name, v in self.pinned.items()})
        attrs = {'kind': self.kind, 'units': self.unit}
        for tag in ('coherent', 'reference', 'reference_unit'):
            if getattr(self, tag) is not None:
                attrs[tag] = getattr(self, tag)
        attrs.update(self._identity_attrs())
        # The speeds the synthesis readers need, the run settings' waveguide
        # included, which the summary string cannot give back.
        speeds = self.speeds
        if speeds is not None:
            for name, value in speeds.to_dict().items():
                if value is not None:
                    attrs[f'speeds_{name}'] = value
        if self._synthesis_floor is not None:
            attrs['synthesis_floor'] = self._synthesis_floor
        # The source identity is written as a metadata entry of the same
        # name would be, so from_xarray reads either layout back.
        source = {tag: getattr(self, tag) for tag in self._SOURCE_IDENTITY}
        # The derivation record is written under its own names too; the
        # budget, a dict no attribute holds, as its JSON.
        derivation = self._derivation()
        if derivation['sonar_budget'] is not None:
            derivation['sonar_budget'] = json.dumps(
                derivation['sonar_budget'])
        encode_attrs({**(self.metadata or {}), **source, **derivation},
                     attrs,
                     skip=self._XARRAY_IDENTITY_ATTRS, who='Field.to_xarray')
        return xr.DataArray(np.asarray(self.data), coords=coords,
                            dims=list(self.coords), name=self.kind,
                            attrs=attrs)

    @classmethod
    def from_xarray(cls, array) -> 'Field':
        """A :class:`Field` from an ``xarray.DataArray``, such as
        :meth:`to_xarray` writes or ``xarray.open_dataarray`` reads back.

        Each dimension must carry a 1-D coordinate of the same name; scalar
        coordinates become :attr:`pinned`, and 1-D non-dimension ones
        :attr:`aux_coords`. ``kind`` / ``unit`` and the
        identity attrs are restored; the remaining attrs become
        :attr:`metadata`, the complex and tuple entries :meth:`to_xarray`
        encoded put back together. The producing engine is recorded only by
        id (``metadata['model_source_id']``), since the engine object itself
        does not travel through a file.

        A pinned ``frequency`` or ``source_depth`` with no ``frequencies`` /
        ``source_depths`` attr — a slab ``da.isel(source_depth=i)`` of a
        :meth:`ResultStack.to_xarray`, whose varying identity the stack does
        not write — narrows that identity to the pinned value, as
        :meth:`at` does.

        Parameters
        ----------
        array : xarray.DataArray
            An array as :meth:`to_xarray` writes it.
        """
        # A file's complex data, stored as real and imaginary parts.
        array = join_complex(array)
        missing = [d for d in array.dims if d not in array.coords]
        if missing:
            raise ConfigurationError(
                f"Field.from_xarray: dimension(s) {missing} carry no "
                f"coordinate; a Field axis needs its labels.")
        coords = {d: np.asarray(array.coords[d].values, dtype=float)
                  for d in array.dims}
        pinned = {name: float(c.values) for name, c in array.coords.items()
                  if name not in array.dims and np.ndim(c.values) == 0}
        aux_coords = {name: (c.dims[0], np.asarray(c.values, dtype=float))
                      for name, c in array.coords.items()
                      if name not in array.dims and np.ndim(c.values) == 1}
        attrs = dict(array.attrs)
        identity = cls._identity_from_attrs(attrs,
                                            cls._XARRAY_IDENTITY_ATTRS)
        metadata = dict(identity.pop('metadata') or {})
        source = {tag: metadata.pop(tag, None)
                  for tag in cls._SOURCE_IDENTITY}
        named = {key[len('speeds_'):]: float(metadata.pop(key))
                 for key in list(metadata) if key.startswith('speeds_')}
        speeds, floor = cls._synthesis_from_metadata(
            metadata, SoundSpeeds(**named) if named else None,
            metadata.pop('synthesis_floor', None))
        derivation = cls._derivation_from(metadata, {})
        _narrowed_identity(
            identity, [axis for axis, key in _RESULTSTACK_VARYING_ATTR.items()
                       if identity[key] is None and axis in pinned],
            coords=coords, pinned=pinned)
        return cls(
            data=np.asarray(array.values), coords=coords,
            pinned=pinned or None, aux_coords=aux_coords or None,
            # CF's ``units``; a file written before it carries ``unit``.
            kind=attrs.get('kind'), unit=attrs.get('units', attrs.get('unit')),
            coherent=(None if attrs.get('coherent') is None
                      else bool(attrs['coherent'])),
            reference=attrs.get('reference'),
            reference_unit=attrs.get('reference_unit'),
            metadata=metadata or None,
            speeds=speeds, synthesis_floor=floor,
            **derivation,
            **identity, **source,
        )

    def __repr__(self) -> str:
        bits = [self.model or None,
                ' '.join(str(v) for v in (self.kind, self.unit) if v)]
        # A field carrying a whole band in its identity but no frequency
        # axis (a synthesised time series, say) states the band, not its
        # first sample.
        if 'frequency' not in self.coords:
            band = self.frequencies
            if band is not None and len(band) > 1:
                bits.append(coordinate_axis('frequency', band))
            elif self.f0 is not None:
                bits.append(coordinate_axis('frequency', self.f0))
        bits.append(' × '.join(coordinate_axis(name, values)
                               for name, values in self.coords.items())
                    or 'scalar')
        if self.pinned:
            bits.append("at " + ", ".join(
                f"{name}="
                + (qty(value, _quantities.coordinate_unit(name))
                   if isinstance(value, (int, float, np.number))
                   else repr(value))
                for name, value in self.pinned.items()))
        return build('Field', bits)

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
                "recover a complex narrowband field first."
            )
        if self.is_complex:
            return transmission_loss_dB(self.data)
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
                "Field.p: data is real; complex pressure unavailable."
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
                "Field.magnitude: requires complex data."
            )
        return np.abs(self.data)

    @property
    def phase(self) -> np.ndarray:
        """Element-wise phase angle in radians, ``angle(data)`` (complex fields only)."""
        if not self.is_complex:
            raise AttributeError("Field.phase: requires complex data.")
        return np.angle(self.data)

    # ── coord-axis conveniences ───────────────────────────────────────

    @property
    def depths(self) -> Optional[np.ndarray]:
        """The ``depth`` axis (m) as a read-only view, or ``None``."""
        return self._axis_view('depth')

    @property
    def ranges(self) -> Optional[np.ndarray]:
        """The ``range`` axis (m) as a read-only view, or ``None``."""
        return self._axis_view('range')

    @property
    def times(self) -> Optional[np.ndarray]:
        """The ``time`` axis (s) as a read-only view, or ``None``."""
        return self._axis_view('time')

    def _axis_view(self, name: str) -> Optional[np.ndarray]:
        """Coordinate ``name`` as a read-only view: a write through an
        accessor would otherwise change the field's own axis."""
        values = self.coords.get(name)
        return None if values is None else read_only(values)

    # ── the export protocol ───────────────────────────────────────────

    def _payload(self):
        return {'data': (self.data, tuple(self.coords), self.unit)}

    def _coords(self):
        coords = {name: (values, _quantities.coordinate_unit(name))
                  for name, values in self.coords.items()}
        coords.update({name: (values, _quantities.coordinate_unit(name), dim)
                       for name, (dim, values) in self.aux_coords.items()})
        return coords

    #: The views :meth:`view` computes, by name.
    VIEWS = ('dB', 'level', 'magnitude', 'phase', 'real', 'imag')

    def view(self, value: str) -> np.ndarray:
        """One derived view of :attr:`data`, by name: what a plotter draws
        for its ``value=`` argument, without plotting.

        ``'dB'`` is :attr:`dB` (TL for complex pressure, the stored level
        for a real dB field); ``'level'`` is ``20*log10|data|``, the
        modulus as a level (``-dB``, complex data only); ``'magnitude'`` and
        ``'phase'`` are :attr:`magnitude` and :attr:`phase`; ``'real'`` and
        ``'imag'`` the parts of the data (``'real'`` of real data is the data
        itself). Every view is read-only.

        Parameters
        ----------
        value : {'dB', 'level', 'magnitude', 'phase', 'real', 'imag'}
            The view to return.

        Raises
        ------
        ConfigurationError
            An unknown ``value``, or a view the data cannot give: a level
            of a time trace, ``'dB'`` of real data not in dB, or a
            complex-only view of real data.
        """
        who = 'Field.view'
        if value not in self.VIEWS:
            raise ConfigurationError(
                f"{who}: value={value!r} is not a view; one of "
                f"{list(self.VIEWS)}.")
        if value in ('dB', 'level') and 'time' in self.coords:
            raise ConfigurationError(
                f"{who}: value={value!r} has no meaning on a time-domain "
                f"field — a trace is linear pressure, not a level.",
                remediation="Use 'real' for the samples, or "
                            ".extract_tone(f) for a complex narrowband "
                            "field with a dB view.")
        if value == 'dB':
            if not self.is_complex and self.unit != 'dB':
                raise ConfigurationError(
                    f"{who}: value='dB' has no meaning on this field — its "
                    f"data are real and in {self.unit!r}, not a level.",
                    remediation="Use value='real' for the values "
                                "themselves.")
            return self.dB
        if value == 'real':
            return read_only(self.data.real if self.is_complex
                             else self.data)
        if not self.is_complex:
            raise ConfigurationError(
                f"{who}: value={value!r} requires complex data; this field "
                f"is real{' and already a level, whose view is dB' if value == 'level' else ''}.")
        if value == 'level':
            return read_only(-self.dB)
        return read_only({'magnitude': self.magnitude, 'phase': self.phase,
                          'imag': self.data.imag}[value])

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

    def _frequency_values(self) -> Optional[np.ndarray]:
        """The frequencies this Field holds: its ``'frequency'`` axis when it
        has one, else the identity list ``frequencies`` (a narrowband field,
        or the band a time trace was synthesised from), else ``None``."""
        f = self.coords.get('frequency')
        if f is not None and f.size:
            return f
        if self.frequencies is not None and len(self.frequencies):
            return self.frequencies
        return None

    @property
    def n_frequencies(self) -> int:
        """Number of frequencies: the length of the ``'frequency'`` axis when
        there is one, else of the identity list :attr:`frequencies`. A time
        trace carries the band it was synthesised from as that list, so it
        counts that band's samples; 0 only when neither is present."""
        f = self._frequency_values()
        return 0 if f is None else int(np.size(f))

    @property
    def f0(self) -> Optional[float]:
        """First frequency (Hz) — the only one for a narrowband result, the
        lowest sample of an ascending band otherwise, never its centre. Read
        off the ``'frequency'`` axis when there is one, else the identity list
        :attr:`frequencies`, which on a time trace is the band it was
        synthesised from; ``None`` when neither is present."""
        f = self._frequency_values()
        return None if f is None else float(f[0])

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
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)

    def _highest_frequency(self) -> Optional[float]:
        """Highest frequency (Hz) the field carries, or ``None``.

        Resolved from the same two sources as :attr:`f0` and in the same
        order. The undersampling guard wants this rather than ``f0`` because
        the quarter-wavelength condition binds at the shortest wavelength
        present, which is the top of the band.
        """
        f = self._frequency_values()
        return None if f is None else float(np.max(np.asarray(f, dtype=float)))

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
        as a successful slice.

        Parameters
        ----------
        **kwargs
            ``axis=label`` per axis to slice, each label finite.
        """
        self._check_axes(kwargs)
        # The label guards (finite, not axis-absorbing) and the nearest rule
        # are the ones every non-blendable carrier shares.
        labels = {name: nearest_index_on_axis(self.coords[name], v, name)
                  for name, v in kwargs.items()}
        for name, v in kwargs.items():
            self._warn_label_off_axis(name, float(v))
        return self._slice(labels)

    def _warn_label_off_axis(self, name: str, label: float) -> None:
        """Warn when ``label`` lies more than half the edge sample step past
        either end of the ``name`` axis: the nearest sample is then the edge
        one, a different place from the one asked for. A one-sample axis has
        no step to measure against and is left alone."""
        axis = np.asarray(self.coords[name], dtype=float)
        if axis.size < 2:
            return
        order = np.sort(axis)
        low_tol = 0.5 * (order[1] - order[0])
        high_tol = 0.5 * (order[-1] - order[-2])
        if order[0] - low_tol <= label <= order[-1] + high_tol:
            return
        edge = order[0] if label < order[0] else order[-1]
        warnings.warn(
            f"Field.at: {name}={label:g} lies outside the {name!r} axis "
            f"[{order[0]:g}, {order[-1]:g}] by more than half a sample step; "
            f"the edge sample {name}={edge:g} is returned instead.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)

    def _warn_eval_label_past_axis(self, name: str, label: float) -> None:
        """Warn when ``label`` lies past either end of the ``name`` axis,
        where :meth:`eval` holds the edge value constant rather than
        interpolating between samples."""
        axis = np.asarray(self.coords[name], dtype=float)
        low, high = float(np.min(axis)), float(np.max(axis))
        if low <= label <= high:
            return
        edge = low if label < low else high
        warnings.warn(
            f"Field.eval: {name}={label:g} lies outside the {name!r} axis "
            f"[{low:g}, {high:g}]; eval holds the edge value {name}={edge:g} "
            f"constant past the end, so the number returned is that sample's, "
            f"not an estimate at {name}={label:g}.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)

    def isel(self, **kwargs) -> "Field":
        """Integer-index slice. Same semantics as :meth:`at` but the
        value is a positional index into the coord array, an integer: a
        float such as ``1.7`` is refused rather than truncated to ``1``.

        Parameters
        ----------
        **kwargs
            ``axis=index`` per axis to slice, each an integer.
        """
        self._check_axes(kwargs)
        return self._slice({name: _integer_index(i, f"Field.isel: {name}")
                            for name, i in kwargs.items()})

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

        Parameters
        ----------
        **bounds
            ``axis=(lo, hi)`` per axis to narrow, inclusive; ``None`` leaves that
            end alone.

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
        kept = {}
        axis_of = {name: i for i, name in enumerate(self.coords)}
        for name, pair in bounds.items():
            low, high = _window_pair('Field.window', name, pair)
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
            kept[name] = keep
        # Narrowing an identity-bearing axis narrows the identity it carries.
        id_kwargs = self.id_kwargs()
        narrowed = [name for name in bounds
                    if id_kwargs.get(_RESULTSTACK_VARYING_ATTR.get(name))
                    is not None]
        return self.replace(
            data=data, coords=coords, aux_coords=self._aux_coords_on(
                coords, keep=kept),
            **_narrowed_identity(id_kwargs, narrowed, coords=coords,
                                 pinned=self.pinned))

    def reindex(self, fill_value: float = np.nan, **axes) -> "Field":
        """This field on wider axes: each kwarg names a coord axis and the
        labels it should run over, which must include every stored label.

        Stored samples land at their own labels (matched exactly); the
        labels added hold ``fill_value``, NaN by default — no data, the
        package's marker for a cell a model did not solve, rather than a
        silent zero. The inverse of :meth:`window`: restoring the receiver
        depths a model could not resolve, or the requested ranges an engine
        reported a subset of. An identity-bearing axis (``frequency``,
        ``source_depth``) carries its identity along, as in :meth:`window`;
        auxiliary coordinates along a reindexed axis take ``fill_value`` at
        the added labels too.

        Parameters
        ----------
        fill_value : float, optional
            Value at the added labels. Default NaN.
        **axes
            ``axis=labels`` per axis to widen, the labels including every stored
            one.

        Examples
        --------
        >>> import numpy as np
        >>> from uacpy.core.results import Field
        >>> f = Field(data=np.array([[1.0, 2.0]]),
        ...           coords={'depth': np.array([10.0]),
        ...                   'range': np.array([100.0, 300.0])})
        >>> f.reindex(range=[100.0, 200.0, 300.0]).data
        array([[ 1., nan,  2.]])
        """
        self._check_axes(axes)
        data = self.data
        coords = dict(self.coords)
        aux = dict(self.aux_coords)
        dtype = np.result_type(data, np.asarray(fill_value))
        for name, labels in axes.items():
            reject_complex(labels, f"Field.reindex: {name}")
            target = np.atleast_1d(np.array(labels, dtype=float))
            if target.ndim != 1:
                raise ConfigurationError(
                    f"Field.reindex: {name} must be a 1-D label array; got "
                    f"shape {target.shape}.")
            require_finite(target, f"Field.reindex: {name}")
            if np.unique(target).size != target.size:
                raise ConfigurationError(
                    f"Field.reindex: {name} repeats a label; each label "
                    f"names one sample.")
            stored = coords[name]
            position = {float(v): i for i, v in enumerate(target)}
            missing = [float(v) for v in stored if float(v) not in position]
            if missing:
                raise ConfigurationError(
                    f"Field.reindex: {name} lacks the stored label(s) "
                    f"{missing}; reindexing adds labels, never drops a "
                    f"stored sample.",
                    remediation="Use window() or at() to drop samples.")
            where = np.array([position[float(v)] for v in stored], dtype=int)
            ax = list(coords).index(name)
            shape = list(data.shape)
            shape[ax] = target.size
            wider = np.full(shape, fill_value, dtype=dtype)
            index = [slice(None)] * data.ndim
            index[ax] = where
            wider[tuple(index)] = data
            data = wider
            coords[name] = target
            for aux_name, (dim, values) in self.aux_coords.items():
                if dim == name:
                    filled = np.full(target.size, fill_value, dtype=float)
                    filled[where] = values
                    aux[aux_name] = (dim, filled)
        id_kwargs = self.id_kwargs()
        widened = [name for name in axes
                   if id_kwargs.get(_RESULTSTACK_VARYING_ATTR.get(name))
                   is not None]
        return self.replace(
            data=data, coords=coords, aux_coords=aux,
            **_narrowed_identity(id_kwargs, widened, coords=coords,
                                 pinned=self.pinned))

    def shift(self, **offsets) -> "Field":
        """Translate a coordinate axis by a constant. The data is untouched.

        Each kwarg names a coord axis and an offset in that axis's own units.
        Use it to move an origin: a transfer function synthesised against a
        source waveform carries that waveform's own peak offset into its time
        axis, and ``shift(time=-peak)`` puts the emission at ``t=0`` so it
        lines up with a solver that marches from the emission itself.

        ``frequency`` and ``source_depth`` are refused: each also stands in
        the Field's identity (:attr:`frequencies`, :attr:`source_depths`), and
        a translated axis would leave :attr:`f0` naming a frequency the field
        no longer holds. A frequency-domain delay is :meth:`remove_delay`.

        Parameters
        ----------
        **offsets
            ``axis=offset`` per axis to translate, in the axis's units.

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
        identity = [name for name in offsets
                    if name in _RESULTSTACK_VARYING_ATTR]
        if identity:
            raise ConfigurationError(
                f"Field.shift: {identity[0]!r} is an identity axis — "
                f"translating it would leave "
                f"{_RESULTSTACK_VARYING_ATTR[identity[0]]!r} naming values "
                f"the field no longer holds.",
                remediation="Shift time, range or depth. To move a delay in "
                            "a transfer function, use remove_delay().")
        coords = dict(self.coords)
        for name, offset in offsets.items():
            delta = float(offset)
            if not np.isfinite(delta):
                raise ConfigurationError(
                    f"Field.shift: {name}={offset!r} is not finite.",
                    remediation="A non-finite offset would put the whole axis "
                                "at NaN, losing the coordinate entirely.")
            coords[name] = coords[name] + delta
        # A shift relabels the axis, not its samples, so every auxiliary
        # coordinate along it still belongs to the same sample.
        return self.replace(coords=coords, aux_coords=self.aux_coords)

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
        so ``|H|`` and any TL derived from it are unchanged. On plain arrays
        this is :func:`~uacpy.acoustic_signal.remove_delay`.

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
            hertz = self.coords['frequency']
            axis = list(self.coords).index('frequency')
        elif 'frequency' in self.pinned:
            # Collapsed by at(frequency=…); the value survives in pinned, so
            # the operation is still well defined — a constant phase.
            hertz, axis = float(self.pinned['frequency']), -1
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

        # The phase factor is acoustic_signal.remove_delay's; this method
        # reads the delay from its arguments or its own range.
        from uacpy.acoustic_signal.channel import remove_delay
        return self.replace(data=remove_delay(self.data, hertz, delay,
                                              axis=axis))

    def eval(self, **kwargs) -> "Field":
        """Interpolated slice — the interpolating counterpart of :meth:`at`.

        Each kwarg names a coord axis and a value; the data is interpolated
        along that axis (constant extrapolation past the ends, with a
        warning naming the edge value held) and the axis collapsed into
        :attr:`pinned`. ``method=`` picks the scheme —
        ``'linear'`` (default), ``'nearest'``, or ``'cubic'``. Use :meth:`at`
        for the nearest stored sample when you must not fabricate values. Note
        that interpolating a real **TL (dB)** field happens in dB and smooths
        sharp interference nulls; slice complex pressure (or use ``at``) for
        null-critical work.

        Parameters
        ----------
        **kwargs
            ``axis=value`` per axis to interpolate, and ``method=``:
            ``'linear'`` (default), ``'nearest'`` or ``'cubic'``.
        """
        method = kwargs.pop('method', 'linear')
        self._check_axes(kwargs)
        if method != 'nearest':      # 'nearest' fabricates nothing
            self._warn_if_undersampled('Field.eval', axes=set(kwargs))
        for name, value in kwargs.items():
            self._warn_eval_label_past_axis(name, float(value))
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
        return self.replace(
            data=data, coords=coords, pinned=pinned,
            **_narrowed_identity(self.id_kwargs(), kwargs, coords=coords,
                                 pinned=pinned))

    def max(self) -> "Field":
        """Slice at the loudest field point; **where** it is is
        :attr:`pinned`.

        Every axis collapses to a pinned scalar: the returned Field has empty
        :attr:`coords`, 0-D :attr:`data` (the value), and each original
        axis's coordinate in :attr:`pinned` — ``{'depth': 62.0,
        'range': 3200.0}`` for the peak of a matched-field ambiguity surface,
        which its repr also shows. ``NaN`` no-data cells (e.g. Bellhop cells
        no ray reached) are excluded.

        Linear data (``unit='Pa'``): global argmax of ``|data|``. Two dB
        quantities are losses, so the least of either is the loudest:
        transmission loss (``kind='pressure'`` in ``unit='dB'``) and OASS
        reverberation, which OASES writes as ``-10·log10 E[|p_scat|²]``. The
        other dB quantities are levels (signal excess, an ambiguity surface),
        where more is more."""
        # OASES's reverberation is a loss: CVMAGS -> VALG10 -> VSMUL(-5E0) in
        # REVINT (oassun26.f:853-858, option 'r'), stored unchanged and tagged
        # oass_quantity='reverberation_loss_dB'. Read as a level, max returned
        # the quietest cell of a reverberation grid.
        if self.data.size == 0:
            raise ConfigurationError(
                f"Field.max: data is empty — coords {list(self.coords)} "
                f"give shape {self.data.shape}. An axis was sliced to "
                f"nothing; widen the .at/.isel/.window selection that produced this "
                f"Field.")
        if self.is_complex:
            strength = np.abs(self.data)  # complex is linear: loudest |p|
        elif self.unit == 'dB' and _quantities.is_loss(self.kind):
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
                    f"{list(self.coords)}."
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
        return self.replace(
            data=new_data, coords=new_coords, pinned=new_pinned,
            **_narrowed_identity(self.id_kwargs(), idx_map,
                                 coords=new_coords, pinned=new_pinned))

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

        A cell no energy reached (the loss carries the no-energy marker)
        stays marked: it comes back at ``-NO_ENERGY_DB``, which
        :func:`~uacpy.core.acoustics.no_energy_mask` recognises, not at a
        finite ``SL - 600``.

        Raises :class:`ConfigurationError` for a time-domain trace, or real
        data in any unit but dB (a trace snapshot, linear magnitudes), which
        are linear pressure rather than a loss — there is nothing to
        subtract.

        On plain arrays this is
        :func:`~uacpy.core.acoustics.received_level_dB`.

        Parameters
        ----------
        source_level_dB : float, optional
            Source level (dB re 1 µPa at 1 m); ``None`` is the field's own
            :attr:`source_level_dB`.
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
        if 'time' in self.coords or (not self.is_complex
                                     and self.unit != 'dB'):
            what = ("a time-domain trace" if 'time' in self.coords
                    else f"real data in {self.unit!r} (a time-trace snapshot, "
                         f"or linear magnitudes)")
            raise ConfigurationError(
                f"Field.at_source_level: {what} is linear pressure, not a "
                f"transmission loss, so a source level has nothing to "
                f"subtract from. Take .extract_tone(f) of the trace for a "
                f"narrowband field first, or scale .data by the source "
                f"amplitude directly."
            )
        if source_level_dB is None:
            source_level_dB = self.source_level_dB
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
        # SL - TL is received_level_dB's, which keeps a no-energy cell at
        # the level view of the marker rather than a finite SL - 600.
        return self.replace(data=_received_level_dB(sl, loss), kind='level',
                            unit='dB', source_level_dB=sl)

    def to_dB(self) -> "Field":
        """Return a real-dB Field via ``-20·log10(|data|)``.

        No-op when ``data`` is already real — including a real field whose
        unit is not dB, which is returned unchanged and whose :attr:`dB` still
        refuses it. There is no linear-to-dB conversion here for real data:
        the sign convention above is the *transmission-loss* one, and applying
        it to an arbitrary real quantity would invent a level the field does
        not carry.

        :attr:`unit` describes the *data*, so it becomes ``'dB'`` rather
        than carried across: a unit left saying ``'Pa'`` on dB data sends
        :meth:`max` down its linear branch, where the largest ``|TL|`` is the
        quietest point rather than the loudest."""
        if not self.is_complex:
            return self
        return self.replace(data=transmission_loss_dB(self.data), unit='dB')

    # ── (depth, range) operations ─────────────────────────────────────

    def mask_below_seafloor(self, bathymetry) -> "Field":
        """Return a copy with samples below the seafloor set to NaN.

        Requires ``depth`` and ``range`` as the first two axes; any
        trailing axis (frequency, time) takes the mask of its cell. The
        computation is :func:`uacpy.core.bathymetry.mask_below_seafloor`.

        Parameters
        ----------
        bathymetry : Bathymetry, Environment or array_like
            The seafloor: a carrier, an environment's, or ``(N, 2)``
            ``(range_m, depth_m)`` rows.
        """
        if list(self.coords)[:2] != ['depth', 'range']:
            raise ConfigurationError(
                "Field.mask_below_seafloor: requires 'depth' and 'range' as "
                f"the first two axes; got {list(self.coords)}."
            )
        if isinstance(bathymetry, Environment):
            bathymetry = bathymetry.bathymetry
        if not isinstance(bathymetry, Bathymetry):
            arr = np.asarray(bathymetry, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ConfigurationError(
                    f"Field.mask_below_seafloor: bathymetry must be shape "
                    f"(N, 2) or an Environment; got array shape {arr.shape}."
                )
            # A linear interpolation takes its range axis on trust: a range
            # column that does not increase interpolates against a broken
            # axis and masks the wrong cells with no error (a two-point
            # profile handed in reversed masked 28 cells where the sorted
            # one masks 24).
            # Bathymetry is where that axis is checked, so the raw array is
            # routed through it rather than checked a second time here.
            bathymetry = Bathymetry.coerce(arr)
        return self.replace(data=mask_below_seafloor(
            self.data, self.coords['depth'], self.coords['range'],
            bathymetry))

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
                NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
            return
        quarter = DEFAULT_SOUND_SPEED / (4.0 * float(f_hi))
        # |diff|: a descending axis is the same physical grid stored the other
        # way round, and ``eval`` walks it in reverse for the same values, so
        # the spacing it is judged by must not depend on the orientation.
        coarse = coarse_axes(axes, quarter)
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
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)

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
                NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
            return
        # |diff|: a descending axis is the same physical grid stored the other
        # way round, and it is drawn from the same samples.
        half = DEFAULT_SOUND_SPEED / (2.0 * float(f_hi))
        # ``>=``, not ``>``: at exactly half a wavelength the carrier advances
        # exactly pi between samples, and +pi and -pi are the same wrapped
        # value, so the direction of rotation is already unrecoverable.
        # Nyquist is the first aliased spacing, not the last good one.
        coarse = coarse_axes(axes, half, inclusive=True)
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
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)

    #: The sampling question each :meth:`check_sampling` view asks.
    _SAMPLING_VIEWS = {'interpolate': '_warn_if_undersampled',
                       'phase': '_warn_if_phase_view_aliases'}

    def check_sampling(self, view: str = 'interpolate', *,
                       where: Optional[str] = None) -> None:
        """Warn when this field's grid is too coarse for ``view``.

        ``'interpolate'`` asks whether a coherent field may be interpolated
        between its samples: every spatial axis against a quarter wavelength
        at the highest frequency carried, and the frequency axis against the
        range it reaches — what :meth:`eval` and :meth:`resample_to` check.
        ``'phase'`` asks whether a phase-sensitive view (a phase, real or
        imaginary map) resolves the carrier: the spatial axes against half a
        wavelength (Nyquist) — what the field plotters check. A real field
        carries no carrier and passes both; a complex field with no
        frequency warns that it cannot be checked. ``where`` names the
        caller in the warning.

        Parameters
        ----------
        view : {'interpolate', 'phase'}, optional
            The question asked (see above). Default ``'interpolate'``.
        where : str, optional
            The caller named in the warning.
        """
        try:
            check = self._SAMPLING_VIEWS[view]
        except (KeyError, TypeError):
            raise ConfigurationError(
                f"Field.check_sampling: view={view!r} is not one of "
                f"{sorted(self._SAMPLING_VIEWS)}.") from None
        getattr(self, check)(where or f"Field.check_sampling(view={view!r})")

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
        interpolates freely.

        Parameters
        ----------
        depths : ndarray
            The new depth axis (m).
        ranges : ndarray
            The new range axis (m).
        method : str, optional
            Interpolation scheme. Default ``'linear'``.
        """
        if list(self.coords) != ['depth', 'range']:
            raise ConfigurationError(
                "Field.resample_to: requires canonical ['depth', 'range'] "
                f"coords; got {list(self.coords)}."
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
        return self.replace(data=new_data,
                            coords={'depth': new_depths, 'range': new_ranges})

    # ── broadband-only (requires 'frequency' coord) ───────────────────

    def to_time_trace(
        self,
        *,
        depth: Optional[float] = None,
        range: Optional[float] = None,
        source_spectrum: Optional[np.ndarray] = None,
        source_waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        window: Optional[str] = 'auto',
        nfft: Optional[int] = None,
        t_start: Optional[float] = None,
    ) -> "Field":
        """What one signal looks like at one receiver.

        Single-trace IFFT of ``H(d, r, :)`` at a chosen ``(depth, range)``.
        Takes the ``(depth, range, frequency)`` grid a broadband run returns,
        or a cell already sliced out of one (``H.at(depth=…, range=…)``,
        whose pinned range places the record). Returns a single-point
        ``Field`` with ``coords={'time': ...}``.

        With ``source_waveform``, this is the received signal: hand it the
        transmitted waveform and the receiver's position and it returns
        ``p(t)`` there::

            trace = H.to_time_trace(depth=50, range=5000,
                                    source_waveform=chirp, sample_rate=fs)

        With neither ``source_waveform`` nor ``source_spectrum`` it is the
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
            ``coords['frequency']``. ``None``, with no ``source_waveform``,
            synthesises the band-limited impulse response.
        source_waveform : ndarray, optional
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
            Rate (Hz) ``source_waveform`` is sampled at.
        window : str or None, default 'auto'
            Spectral window applied across the whole band of ``H(f)``
            before the IFFT: ``'hann'``, ``'hamming'``, ``'blackman'``,
            ``'tukey'``, or ``None`` (``'boxcar'``) for none. It spans the
            band, so it is a filter, not an edge taper: it reshapes a pulse
            and removes energy (a Hann costs 0.8-7.7 dB on ordinary
            pulses). ``'auto'`` picks ``None`` when ``source_waveform`` or
            ``source_spectrum`` is given —
            the received signal is then ``S(f)·H(f)`` with no extra filter,
            as Jensen et al. synthesise it (*Computational Ocean
            Acoustics*, sect. 8.2.1.1) — and ``'hann'`` for the bare
            impulse response, whose hard band edges would otherwise ring
            (sidelobes -31 dB untapered against -72 dB with the Hann).
        nfft : int, optional
            IFFT length. ``None`` sizes it automatically; an explicit value
            that would put the highest data bin at or above Nyquist is
            rejected rather than allowed to alias.
        t_start : float, optional
            Time of the first output sample (s). ``None`` estimates it from
            the range and the fastest sound speed the model reported.

        Warns
        -----
        FallbackWarning
            When ``depth`` or ``range`` falls outside the grid. The match is
            to the nearest stored coordinate, so a receiver beyond the
            panel's edge silently becomes the edge cell — a trace of the
            wrong place that looks like a trace of the right one."""
        who = "Field.to_time_trace"
        source_spectrum = waveform_spectrum_on(
            self.coords.get('frequency', []), source_waveform, sample_rate,
            source_spectrum, who)
        grid, collapse, placeholder = self._on_synthesis_axes(who)
        if source_spectrum is not None:
            check_source_spectrum(source_spectrum,
                                  grid.coords['frequency'].size, who)
        for name, label in (('depth', depth), ('range', range)):
            axis = grid.coords.get(name)
            if label is None or name in placeholder:
                continue
            lo, hi = float(np.min(axis)), float(np.max(axis))
            tol = axis_match_tolerance(axis, label)
            if not (lo - tol <= float(label) <= hi + tol):
                warnings.warn(
                    f"{who}: {name}={float(label):g} is outside the grid "
                    f"({lo:g} to {hi:g}); the nearest stored "
                    f"{name} is used instead, so this trace is of a "
                    f"different place than asked for.",
                    FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
        if window == 'auto':
            window = 'hann' if source_spectrum is None else None
        # How long the transmitted signal is, for the record-edge notice:
        # the waveform's own duration, 0 for the bare impulse response, and
        # unknown for a raw source spectrum.
        if source_waveform is not None:
            pulse_s = np.size(source_waveform) / float(sample_rate)
        elif source_spectrum is None:
            pulse_s = 0.0
        else:
            pulse_s = None
        # Deferred: _field_synthesis builds Fields, so it imports this
        # module at load time.
        from uacpy.core.results._field_synthesis import _ifft_to_trace
        trace = _ifft_to_trace(
            grid, depth=depth, range=range,
            source_spectrum=source_spectrum,
            window=window, nfft=nfft, t_start=t_start, pulse_s=pulse_s,
        )
        for name in placeholder:
            trace.pinned.pop(name, None)
        return trace

    #: The axes, in order, the IFFT synthesis runs on.
    _SYNTHESIS_AXES = ('depth', 'range', 'frequency')

    def _on_synthesis_axes(self, who: str):
        """This Field on the ``(depth, range, frequency)`` axes the synthesis
        runs on: ``(grid, collapse, placeholder)``.

        A cell already sliced out of a grid (``H.at(depth=…, range=…)``, or
        :meth:`to_transfer_function` of one trace) has lost its ``depth`` and
        ``range`` axes to :attr:`pinned`; each is re-inflated as a
        one-sample axis at its pinned value, and named in ``collapse`` so the
        caller can slice it back off. Other one-sample axes are sliced away
        first. A ``depth`` that is neither an axis nor pinned is a
        one-sample placeholder at 0 m — the synthesis reads no depth — and is
        named in ``placeholder`` so the caller can drop it from the result's
        :attr:`pinned` rather than report a depth nobody gave. A missing
        ``range`` is refused, since the record's start is estimated from
        it."""
        f = self
        for axis in list(f.coords):
            if (axis not in self._SYNTHESIS_AXES
                    and f.coords[axis].size == 1):
                f = f.isel(**{axis: 0})
        present = [a for a in self._SYNTHESIS_AXES if a in f.coords]
        if 'frequency' not in f.coords or list(f.coords) != present:
            raise ConfigurationError(
                f"{who}: needs a 'frequency' axis, with any 'depth' and "
                f"'range' axes before it in that order; got "
                f"{list(self.coords)}. Slice every other axis to one sample "
                f"first, e.g. .at(...).")
        if list(f.coords) == list(self._SYNTHESIS_AXES):
            return f, (), ()
        if 'range' not in f.coords and 'range' not in f.pinned:
            raise ConfigurationError(
                f"{who}: this field has no range axis and no pinned range, "
                f"and the synthesis needs one to place its record. Slice a "
                f"grid with .at(range=…), or build the field over a 'range' "
                f"axis.")
        coords, data, collapse, placeholder = {}, np.asarray(f.data), [], []
        for position, name in enumerate(self._SYNTHESIS_AXES[:2]):
            if name in f.coords:
                coords[name] = f.coords[name]
                continue
            value = f.pinned.get(name)
            if value is None:
                value = 0.0
                placeholder.append(name)
            coords[name] = np.array([value], dtype=float)
            data = np.expand_dims(data, axis=position)
            collapse.append(name)
        coords['frequency'] = f.coords['frequency']
        pinned = {k: v for k, v in f.pinned.items() if k not in collapse}
        grid = f.replace(data=data, coords=coords, pinned=pinned)
        return grid, tuple(collapse), tuple(placeholder)

    def truncate_response(self, duration: float, *,
                          origin: Union[str, float] = 'peak',
                          window: Optional[str] = None) -> "Field":
        """``H(f)`` with its impulse response cut to ``duration`` — the
        transfer function a pulse that long actually sees.

        Requires a ``frequency`` axis with a uniform spacing and complex
        data. Every cell is transformed to its own impulse response over the
        band, windowed, and transformed back; the coords, the axis order and
        the identity are unchanged.

        The reasoning, the literature and the measurements behind this method
        are in ``docs/theory/broadband_products.md``, section "Cutting a
        transfer function to a pulse length".

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
        window : {None, 'hann'}, default None
            ``None`` (a rectangular gate) is the separability criterion stated plainly —
            inside interferes, outside does not — and puts the rectangle's
            own sinc skirt on ``H``. ``'hann'`` tapers the cut instead,
            trading a wider effective window for a skirt 31 dB down.

        Returns
        -------
        Field
            Same coords and identity; only ``data`` changes.

        Warns
        -----
        NumericsWarning
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

        On plain arrays this is
        :func:`~uacpy.acoustic_signal.gate_transfer_function`; what the
        method adds is the coord check and the Field re-wrap.
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
        from uacpy.acoustic_signal.channel import _gate_transfer_function
        out = _gate_transfer_function(
            self.data, frequencies=freqs, duration=duration, origin=origin, window=window,
            axis=axis, who=who)
        return self.replace(data=out)

    def broadband_loss(self, source_spectrum=None, *, source_waveform=None,
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

            10 log10( sum_f w(f) / sum_f w(f) |H(f)|^2 ),  w = |source_spectrum|^2

        The reasoning, the literature and the measurements behind this method
        are in ``docs/theory/broadband_products.md``, section "Broadband
        propagation loss".

        **Where the field comes from.** This needs a complex ``H(f)`` over a
        ``frequency`` axis, which is what ``RunMode.BROADBAND`` returns —
        Bellhop, Kraken, Scooter and RAM all offer it, and ``COHERENT_TL``
        takes a single source frequency by construction::

            f = np.arange(800.0, 1201.0, 20.0)
            H = model.run(env, Source(depths=20.0, frequencies=f), rcv,
                          run_mode=RunMode.BROADBAND, frequencies=f)

        Parameters
        ----------
        source_spectrum : array_like, optional
            The source's spectrum already sampled on this field's
            ``frequency`` axis, real or complex; only ``|source_spectrum|^2`` is
            used, and any overall scale cancels in the ratio. ``None`` with
            no ``source_waveform`` weights the band uniformly, which is Eq. 11.46's
            white source.
        source_waveform : array_like, optional
            A transmitted waveform, in place of ``source_spectrum`` — the loss for
            THAT signal, whatever it is. Its spectrum is evaluated on this
            field's axis by :func:`_source_spectrum_at`, the same DTFT
            ``synthesize_time_series`` uses, so the answer satisfies
            ``SEL = ESL - loss`` exactly. Do not pre-interpolate an ``rfft``
            onto the axis and pass it as ``source_spectrum`` instead: the two grids
            rarely coincide, and interpolation is a triangular-kernel
            convolution rather than a resampling — tens of per cent of error
            in ``S(f)`` for ordinary waveforms (43-75 % measured), reaching
            the 100 % :func:`_source_spectrum_at` quotes for an unwindowed
            tone on a bin. Requires ``sample_rate``.
        sample_rate : float, optional
            Rate (Hz) ``source_waveform`` is sampled at.

        Returns
        -------
        Field
            The loss in dB, ``kind='pressure'`` and ``unit='dB'`` so
            :attr:`tl` and :meth:`plot` treat it as the transmission-loss
            map it is. The ``frequency`` axis is gone; the band it averaged
            is kept in :attr:`band_hz` as ``(first, last)`` — the
            identity narrows to a single value, so ``.frequencies`` is NOT
            where to look for it — and :attr:`pinned['frequency']` records
            the spectrum-weighted centroid of the axis's samples — the
            signal's own centre for a ``source_waveform`` (a 500 Hz burst over a
            25 Hz-4 kHz axis pins 500 Hz, not the axis's 2 kHz midpoint,
            which is what a plotter would otherwise caption the map with),
            and the mean of the samples for a white source, which is the
            band centre on the uniform axes these runs produce.
            Cells no path reached stay ``NaN``.

        Raises
        ------
        ConfigurationError
            No ``frequency`` axis, real data, a ``source_spectrum`` whose length is
            not the frequency axis's, a non-finite or zero-energy
            ``source_spectrum``, both ``source_spectrum`` and ``source_waveform``, or a
            ``source_waveform`` without a ``sample_rate``.

        On plain arrays this is
        :func:`~uacpy.acoustic_signal.broadband_propagation_loss`; what the
        method adds is turning ``source_waveform`` into ``w(f)``, the weighted
        centroid pin and the band metadata.
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
        source_spectrum = waveform_spectrum_on(
            freqs, source_waveform, sample_rate, source_spectrum, who)
        if source_spectrum is None:
            weights = np.ones(freqs.size, dtype=float)
        else:
            supplied = np.asarray(source_spectrum)
            if supplied.ndim > 1:
                # Checked on SHAPE before ravel(), which otherwise turns a
                # (3, 4) array into a legal-looking 12-sample weight on a
                # 12-frequency axis and returns the same number as the
                # (12,) case — a wrong spectrum that cannot be told from a
                # right one by its answer.
                raise ConfigurationError(
                    f"{who}: source_spectrum must be 1-D, one weight per frequency; "
                    f"got shape {supplied.shape}. Flattening it would pair "
                    f"weights with frequencies in an order you did not "
                    f"choose.")
            weights = np.abs(supplied).astype(float).ravel() ** 2
            if weights.size != freqs.size:
                raise ConfigurationError(
                    f"{who}: source_spectrum has {weights.size} samples but the "
                    f"frequency axis has {freqs.size}; it is the source "
                    f"spectrum ON this field's grid.")
            if not np.all(np.isfinite(weights)):
                raise ConfigurationError(
                    f"{who}: source_spectrum must be finite.")
            if weights.sum() <= 0.0:
                raise ConfigurationError(
                    f"{who}: source_spectrum carries no energy, so the weighted "
                    f"average is undefined.")
        # The weighted average and its dB step are
        # broadband_propagation_loss's, including the accumulate-per-map
        # memory behaviour; what this method adds is turning a waveform
        # into w(f), the weighted centroid pin and the band metadata below.
        # Deferred: acoustic_signal pulls scipy, and uacpy's public
        # surface is imported without it (test_lazy_imports).
        from uacpy.acoustic_signal.channel import _broadband_propagation_loss
        from uacpy.acoustic_signal.spectral import spectral_centroid
        loss = _broadband_propagation_loss(self.data, weights, axis=axis,
                                          who=who)
        coords = {name: v for name, v in self.coords.items()
                  if name != 'frequency'}
        # The WEIGHTED centroid, not the axis midpoint: a 500 Hz burst
        # weighted onto a 25 Hz-4 kHz axis is a map of 500 Hz, and labelling
        # it 2 kHz is the same defect Field.window's identity narrowing
        # exists to prevent — a legal frequency, and the wrong one. Uniform
        # weights still give the band centre.
        return self._collapse_band(loss, coords,
                                   spectral_centroid(freqs, weights),
                                   'pressure')

    def synthesize_time_series(
        self,
        source_waveform: np.ndarray,
        sample_rate: float,
        *,
        t_start: Optional[float] = None,
        window: Optional[str] = None,
        nfft: Optional[int] = None,
    ) -> "Field":
        """Convolve every grid trace with ``source_waveform`` to obtain a
        time-domain Field shaped ``(n_d, n_r, n_t)``.

        Takes the ``(depth, range, frequency)`` grid a broadband run returns,
        or a slice of one whose ``depth`` / ``range`` went to :attr:`pinned`;
        the result then keeps the slice's axes, with ``time`` in place of
        ``frequency``.

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
            anchors it on the nearest range to the source, the smallest
            ``|range|`` on the axis, whichever way the axis runs.
        window : str
            Spectral window across the whole band, as on
            :meth:`to_time_trace`. Default ``None``: the received signal
            is ``S(f)·H(f)``, and a window here would filter the pulse and
            bias its level low. A warning names the case where ``None``
            rings — a band that cuts through the waveform's spectrum.
        nfft : int, optional
            As on :meth:`to_time_trace`, applied to every cell.

        On plain arrays this is
        :func:`~uacpy.acoustic_signal.synthesize_time_series`; what the method
        adds is the start time estimated from the range and the model's sound
        speed, per-cell warnings named by depth and range, and the Field
        re-wrap."""
        grid, collapse, placeholder = self._on_synthesis_axes(
            "Field.synthesize_time_series")
        source_waveform = require_source_waveform(
            source_waveform, "Field.synthesize_time_series")
        # Deferred: _field_synthesis builds Fields, so it imports this
        # module at load time.
        from uacpy.core.results._field_synthesis import _synthesize_time_series
        traces = _synthesize_time_series(
            grid,
            source_waveform=source_waveform,
            sample_rate=sample_rate,
            t_start=t_start, window=window, nfft=nfft,
        )
        if collapse:
            traces = traces.isel(**{name: 0 for name in collapse})
            for name in placeholder:
                traces.pinned.pop(name, None)
        return traces

    def sound_exposure_level(
        self,
        source_waveform: np.ndarray,
        sample_rate: float,
        *,
        ref: float = REFERENCE_PRESSURE_WATER,
        window: Optional[str] = None,
        nfft: Optional[int] = None,
        t_start: Optional[float] = None,
    ) -> "Field":
        """Sound exposure level of one transmission of ``source_waveform``,
        cell by cell — the energy a transient delivers, as a map.

        The reasoning, the literature and the measurements behind this method
        are in ``docs/theory/broadband_products.md``, section "Sound exposure
        level and the record length".

        Each cell's trace is synthesised by
        :meth:`synthesize_time_series` and integrated::

            SEL = 10 log10( sum_t p(t)^2 * dt / ref^2 )

        The integral runs over the **whole** record, which is what the
        definition asks for: Ainslie's time intervals are "chosen to contain
        the whole of the transmitted pulse". Nothing is gated away, so a
        late arrival contributes its energy even where it is too late to
        interfere. Whether it interferes is :meth:`broadband_loss`'s
        question, and whether a receiver that opens for ``T`` would see it
        at all is :meth:`truncate_response`'s.

        **There is no source-level argument: the level rides on the
        waveform's amplitude.** ``synthesize_time_series`` reproduces the
        source waveform where ``H`` is unity, so ``source_waveform`` is the
        source's pressure at the range this field's ``H`` is referenced to —
        1 m for a transmission-loss field. Scale it and the map scales with
        it. For a source level ``SL`` in dB re 1 µPa at 1 m, the waveform's
        rms pressure is ``1e-6 * 10**(SL/20)`` Pa::

            unit = waveform / np.sqrt(np.mean(waveform ** 2))
            sel = H.sound_exposure_level(unit * 1e-6 * 10 ** (SL / 20), fs)

        Passing a waveform of unit **energy** in the ref units instead,
        ``∫u² dt = (1 µPa)²·1 s``::

            u = waveform / np.sqrt(np.sum(waveform ** 2) / fs) * 1e-6
            loss = H.sound_exposure_level(u, fs)      # -TL_E, in dB

        returns the propagation term alone, and the source level goes on
        afterwards as an **energy** source level, ``ESL = SL + 10 log10(T)``
        for constant power over the pulse (Ainslie Eq. 3.155): ``SEL = ESL +
        loss``, and the two routes agree exactly. The rms-normalised
        ``unit`` above does not serve here: its result already holds the
        pulse's own ``10 log10(T)`` and the 120 dB between Pa and µPa, so
        adding ESL to it counts both twice — ``120 + 10 log10(T)`` dB high,
        100 dB for a 10 ms pulse. Worked through: ``SL`` = 190 dB re 1 µPa
        at 1 m, a 10 ms burst, 100 m of spherical spreading. Received level
        is 190 - 40 = 150 dB re 1 µPa, so the exposure is 150 + 10 log10(0.01)
        = **130 dB re 1 µPa²·s**, which is what both routes return.

        Parameters
        ----------
        source_waveform : ndarray
            The 1-D transmitted waveform; the generators return a
            ``(time, signal)`` pair, so pass ``tone_burst(...)[1]``.
        sample_rate : float
            Rate (Hz) the waveform is sampled at.
        ref : float, default 1e-6
            Reference pressure (Pa). The exposure ref is its square,
            so the default gives dB re 1 µPa²·s.
        window : str or None, default None
            Band taper, passed to :meth:`synthesize_time_series`. A flat band
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
            ``ref``.

        Warns
        -----
        NumericsWarning
            Through :meth:`synthesize_time_series`, when the ``1/df`` record
            cannot hold the **pulse**. Nothing warns about the **channel**
            outlasting it, and that gap is real — see the note above on
            folds.

        On plain arrays the level itself is
        :func:`~uacpy.core.acoustics.sound_exposure_level`.
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
        # The integral and its dB step are acoustics.sound_exposure_level's;
        # what this method adds is the synthesis and the band metadata. It
        # floors a silent cell at -180 dB rather than -inf, which would
        # poison any mean taken over the map.
        return self._transient_level(
            lambda pressure, rate: _sound_exposure_level(pressure, rate,
                                                         ref),
            'sound_exposure', who, ref, source_waveform, sample_rate,
            window=window, nfft=nfft, t_start=t_start)

    def peak_sound_pressure_level(
        self,
        source_waveform: np.ndarray,
        sample_rate: float,
        *,
        ref: float = REFERENCE_PRESSURE_WATER,
        window: Optional[str] = None,
        nfft: Optional[int] = None,
        t_start: Optional[float] = None,
    ) -> "Field":
        """Peak sound pressure level of one transmission, cell by cell —
        the other half of the impulsive dual metric.

        ``20 log10( max|p(t)| / ref )`` on each cell's synthesised
        trace. Exposure criteria for impulsive sound are stated as a PAIR —
        "frequency-weighted sound exposure level (SEL) and unweighted peak
        sound pressure level", with "exceeding either threshold by the
        specified level ... sufficient to result in the predicted TTS or
        PTS" (Southall et al., *Marine Mammal Noise Exposure Criteria*,
        2019). :meth:`sound_exposure_level` is the first half; this is the
        second, and neither substitutes for the other: SEL integrates the
        whole transmission while this reads its single loudest excursion.

        The reasoning, the literature and the measurements behind this method
        are in ``docs/theory/broadband_products.md``, section "Peak sound
        pressure level".

        ``window`` defaults to ``None`` for the same reason it does on
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
        ref : float, default 1e-6
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
            :attr:`band_hz`.

        Raises
        ------
        ConfigurationError
            Real data (a dB or TL field), non-canonical coords, or a
            non-positive ``ref``.

        On plain arrays the level itself is
        :func:`~uacpy.core.acoustics.peak_level`.
        """
        who = "Field.peak_sound_pressure_level"
        if not self.is_complex:
            raise ConfigurationError(
                f"{who}: needs complex H(f). A dB or TL field has no phase, "
                f"so it has no waveform to take a peak of — synthesising "
                f"one inverse-transforms the decibels AS pressure. Start "
                f"from the BROADBAND field and scale the waveform to the "
                f"source level rather than calling at_source_level().")
        # acoustics.peak_level's, floored the same way, so a silent cell
        # is -180 dB and not -inf.
        return self._transient_level(
            lambda pressure, rate: _peak_level(pressure, ref, axis=-1),
            'peak_pressure', who, ref, source_waveform, sample_rate,
            window=window, nfft=nfft, t_start=t_start)

    def _transient_level(self, level, kind: str, who: str, reference: float,
                         source_waveform, sample_rate, *, window, nfft,
                         t_start) -> "Field":
        """The per-cell level ``level(pressure, output_rate)`` of the
        waveform received through this field (:meth:`synthesize_time_series`),
        as the band-collapsed map of ``kind`` (:meth:`_collapse_band`),
        pinned at the waveform's spectral centroid on this band. The shared
        body of :meth:`sound_exposure_level` and
        :meth:`peak_sound_pressure_level`."""
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
        values = level(pressure, 1.0 / traces.dt)
        coords = {name: v for name, v in traces.coords.items()
                  if name != 'time'}
        # The centroid is the waveform's, weighted by its own spectrum on
        # this axis, so the map reprises the signal's centre.
        band = np.asarray(self.coords['frequency'], dtype=float)
        return self._collapse_band(
            values, coords, _waveform_centroid(band, source_waveform,
                                               sample_rate), kind)

    def _collapse_band(self, values, coords, centre: float,
                       kind: str) -> "Field":
        """The dB map a band-collapsing reducer returns: ``values`` of
        ``kind`` on ``coords``, the frequency axis pinned at ``centre`` and
        the identity narrowed to it (:func:`_narrowed_identity`).

        The band collapsed is kept in :attr:`band_hz`: the identity
        narrows to one centroid, so without it a 50 Hz average and a 2 kHz
        one over the same centre are indistinguishable afterwards — and a
        plotter captioning the map has only the single frequency, which the
        map is not."""
        band = np.asarray(self.coords['frequency'], dtype=float)
        pinned = dict(self.pinned)
        pinned['frequency'] = centre
        return self.replace(
            data=values, coords=coords, pinned=pinned,
            band_hz=(float(band[0]), float(band[-1])),
            kind=kind, unit='dB',
            **_narrowed_identity({}, ['frequency'], coords=coords,
                                 pinned=pinned))

    def to_transfer_function(self, *, band=None) -> "Field":
        """``H(f)`` from a time-domain Field — the inverse of
        :meth:`to_time_trace` with ``window=None``.

        A trace synthesised through a band window (``to_time_trace``'s
        default ``'hann'`` for a bare impulse response) comes back as
        ``H(f)·w(f)``, the window still on it — measured on a two-path ``H``,
        the default round trip is off by up to 100 % at the band edges where
        ``window=None``'s closes to rounding. A trace whose
        :attr:`synthesis_window` records a window other than ``None`` warns.

        The forward direction has several routes (:meth:`to_time_trace`,
        :meth:`synthesize_time_series`,
        :func:`~uacpy.acoustic_signal.impulse_response_from_transfer_function`);
        this is the way back, as a carrier rather than as raw arrays.

        The transform is::

            H(f) = rfft(h) * dt * exp(-2 pi i f t0)

        The reasoning, the literature and the measurements behind this method
        are in ``docs/theory/broadband_products.md``, section "From a trace
        back to H(f)".

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
            it. It keeps the trace's :attr:`phase_reference`: a trace this
            package synthesised is ``TIME_DOMAIN_NATIVE``, and so is its
            transform, because the spectrum still carries the source
            spectrum (or band window) the trace was made with.
            :meth:`to_time_trace` and :meth:`synthesize_time_series` refuse
            it for that reason — synthesising it again would apply that
            spectrum a second time.

        Raises
        ------
        ConfigurationError
            No ``time`` axis, fewer than two samples, a non-uniform one, or
            a ``band`` that keeps no bin.

        On plain arrays this is
        :func:`~uacpy.acoustic_signal.transfer_function_from_impulse_response`;
        what the method adds is the axis bookkeeping, the band read off its
        identity, and the ``dt`` that carries the result into the density
        convention ``to_time_trace`` produces.
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
        if not steps_are_uniform(steps, dt):
            raise ConfigurationError(
                f"{who}: the time axis is not uniformly spaced, so an FFT "
                f"of it would place every bin wrongly. Resample first.")
        axis = list(self.coords).index('time')
        band_window = self._synthesis_window
        if band_window is not None:
            warnings.warn(
                f"{who}: this trace was synthesised through a "
                f"{band_window!r} band window, so the spectrum returned is "
                f"H(f)·w(f), the window still on it, not H(f). Synthesise "
                f"with window=None for a trace this inverts exactly.",
                NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
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
        from uacpy.acoustic_signal.channel import (
            _transfer_function_from_impulse_response,
        )
        freqs, spectrum = _transfer_function_from_impulse_response(
            self.data, 1.0 / dt, t0=float(times[0]), band=band, axis=axis,
            who=who)
        # That function is unscaled, matching the plain irfft of its own
        # counterpart; a Field's H is a spectral density, the convention
        # to_time_trace's `ifft * fs` produces, so dt carries it across.
        spectrum = spectrum * dt
        spectrum = np.moveaxis(spectrum, axis, -1)
        coords = {name: v for name, v in self.coords.items() if name != 'time'}
        coords['frequency'] = freqs
        # H(f) is another quantity than the trace: its coherence is decided
        # anew, as a frequency-domain pressure field's.
        return self.replace(
            data=np.moveaxis(spectrum, -1, list(coords).index('frequency')),
            coords=coords, kind='pressure', unit='Pa', coherent=None,
            **_narrowed_identity({}, ['frequency'], coords=coords,
                                 pinned=self.pinned))

    def plot_transfer_function(
        self, *, ax=None, title=None, figsize=(8, 6), **kwargs,
    ):
        """Plot the transfer function ``H(f)`` at one receiver cell as two
        stacked panels: modulus in dB (``20·log10|H|``, top) over phase
        (bottom), sharing the frequency axis.

        Reduce-then-plot: call on a field already sliced to one ``(depth,
        range)`` cell (``H.at(depth=…, range=…).plot_transfer_function()``); a
        single-receiver field plots directly (singleton axes are squeezed).
        Pass ``ax=(ax_mag, ax_phase)`` to draw into existing axes. This one
        draws two panels, so ``ax`` takes a **pair**: anything that unpacks
        into two Axes, including the ndarray ``plt.subplots(2, 1)`` returns.
        Returns ``(fig, (ax_mag, ax_phase))``. Draws through
        :func:`uacpy.plot.plot_transfer_function`.

        What the panels show depends on the frequency grid. Each pair of
        paths ``dtau`` apart puts a fringe of period ``1/dtau`` on
        ``|H(f)|``, and a grid from ``Arrivals.synthesis_band`` places
        ``margin`` samples on it — 1.2 by default, so the modulus is drawn
        critically sampled: a dense oscillation under a beat envelope, real
        multipath interference but interpolated by the plotter between
        samples. Raise ``margin`` to see the fringes. The phase is drawn
        wrapped, and a bulk delay ``tau`` turns it once every ``1/tau`` Hz,
        so over a band far wider than that it fills the panel; take the
        delay out (``H * exp(2j*pi*f*tau)``) to see what remains.

        Parameters
        ----------
        ax : (Axes, Axes), optional
            The ``(ax_mag, ax_phase)`` pair (see above).
        title : str, optional
            Title of the modulus panel. ``None`` draws the default caption.
        figsize : tuple, optional
            Size (inches) of the new figure. Default ``(8, 6)``.
        **kwargs
            Keywords of :meth:`plot` for both panels.
        """
        return plotter('plot_transfer_function')(
            self, ax=ax, title=title, figsize=figsize, **kwargs)

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
        :meth:`synthesize_time_series` instead. Returns ``(fig, ax)``.
        Draws through :func:`uacpy.plot.plot_impulse_response`.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Existing axes; a new figure is made when omitted.
        title : str, optional
            Axes title. ``None`` draws the default caption.
        window, nfft, t_start : optional
            Passed to :meth:`to_time_trace`. Default ``window='hann'``.
        figsize : tuple, optional
            Size (inches) of the new figure. Default ``(8, 4)``.
        **kwargs
            Keywords of the time trace's :meth:`plot`.
        """
        return plotter('plot_impulse_response')(
            self, ax=ax, title=title, window=window, nfft=nfft,
            t_start=t_start, figsize=figsize, **kwargs)

    # ── time-domain only (requires 'time' coord) ──────────────────────

    def extract_tone(
        self,
        frequency: float,
        *,
        window: str = 'hann',
    ) -> "Field":
        """Extract steady-state complex pressure at one frequency from a
        time-domain Field. Needs a ``time`` axis; every other axis is kept,
        so a ``(depth, range, time)`` grid gives a ``(depth, range)`` map and
        the single ``(time,)`` trace :meth:`to_time_trace` returns gives a
        0-d phasor.

        The transform is evaluated **at** ``frequency``, not at the nearest
        rfft bin, so a tone that does not land on the record's bin grid is
        recovered correctly; on a bin it reproduces the rfft to ~1e-15. The
        returned Field's ``frequencies``/``pinned['frequency']`` therefore
        carry the frequency asked for.

        The ``2·X/Σwin`` tone estimator assumes a non-DC, non-Nyquist
        frequency; at exactly 0 Hz or the Nyquist frequency the doubling
        overestimates the amplitude by 2×.

        It measures a tone that is steady over the record. A transient's
        energy is finite, so on a pulse response the same sum returns
        ``2·Δf·H(f)`` or less, depending on the window and on where the pulse
        sits; a band-limited impulse response (``kind='impulse_response'``,
        what :meth:`to_time_trace` returns without a source) is therefore
        refused, and ``trace.to_transfer_function().at(frequency=f)`` is its
        ``H(f)``.

        The returned value is the phasor ``A`` of
        ``p(t) = Re{A·e^{+2πift}}`` — the same sign convention the IFFT
        synthesis consumes, so a tone extracted here and an ``H(f)`` bin
        handed to :meth:`to_time_trace` carry phase the same way.

        On plain arrays this is :func:`~uacpy.acoustic_signal.tone_phasor`.

        Parameters
        ----------
        frequency : float
            The tone's frequency (Hz).
        window : str, optional
            Taper over the record. Default ``'hann'``.
        """
        if 'time' not in self.coords:
            raise ConfigurationError(
                "Field.extract_tone: needs a 'time' axis; got "
                f"{list(self.coords)}."
            )
        if self.kind == 'impulse_response':
            # The estimator recovers a tone that fills the record. An impulse
            # response is a transient: 2·Σh·e^{-2πift}/Σw is 2·Δf·H(f) under
            # a rectangular window and a record-dependent fraction of it under
            # any other, so what it returned was not H.
            raise ConfigurationError(
                "Field.extract_tone: this trace is a band-limited impulse "
                "response, a transient, and the tone estimator measures a "
                "tone that fills the record — on a transient it returns "
                "2·Δf·H(f) (window=None) or a record-dependent fraction of "
                "it, not H(f).",
                remediation="Take H at that frequency from the transform: "
                            "trace.to_transfer_function().at(frequency=f).")
        time_axis = list(self.coords).index('time')
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
        from uacpy.acoustic_signal.spectrum_at import _tone_phasor
        amp = _tone_phasor(
            self.data, np.asarray(self.coords['time'], dtype=float),
            frequency, window=window, axis=time_axis,
            who='Field.extract_tone')
        # The recovered tone is the identity of the returned Field, not the
        # time-domain parent's frequency list.
        coords = {k: v for k, v in self.coords.items() if k != 'time'}
        pinned = {**self.pinned, 'frequency': float(frequency)}
        return self.replace(
            data=amp, coords=coords, pinned=pinned,
            **_narrowed_identity({}, ['frequency'], coords=coords,
                                 pinned=pinned))


def _narrowed_identity(id_kwargs: Dict[str, Any], axes, *,
                       coords: Dict[str, np.ndarray],
                       pinned: Dict[str, float]) -> Dict[str, Any]:
    """``id_kwargs`` with the identity of each changed ``axes`` entry that
    carries one (``frequency`` -> ``frequencies``, ``source_depth`` ->
    ``source_depths``) set to what the field now holds: the axis's labels
    when it is still in ``coords``, else ``[pinned value]``.

    The identity lists are what the field HOLDS, not what the run that
    produced it swept: left alone, a field windowed to 1400-1600 Hz answers
    ``f0 = 1000``, and a map pinned at a 500 Hz centroid reprs as the band's
    first sample — legal values, and the wrong ones, with nothing to flag
    them."""
    for axis in axes:
        key = _RESULTSTACK_VARYING_ATTR.get(axis)
        if key is None:
            continue
        if axis in coords:
            id_kwargs[key] = np.asarray(coords[axis], dtype=float)
        else:
            id_kwargs[key] = np.array([pinned[axis]], dtype=float)
    return id_kwargs


def _waveform_centroid(band: np.ndarray, source_waveform,
                       sample_rate: float) -> float:
    """The :func:`~uacpy.acoustic_signal.spectral_centroid` of ``band``
    under the waveform's own energy spectrum ``|S(f)|²`` evaluated on it."""
    # Deferred: acoustic_signal pulls scipy, and uacpy's public surface is
    # imported without it (test_lazy_imports).
    from uacpy.acoustic_signal.spectral import spectral_centroid
    from uacpy.acoustic_signal.spectrum_at import waveform_spectrum_at
    weights = np.abs(waveform_spectrum_at(source_waveform, sample_rate,
                                          band)) ** 2
    return spectral_centroid(band, weights)


_RESULTSTACK_VARYING_ATTR = {
    'source_depth': 'source_depths',
    'frequency':    'frequencies',
}
