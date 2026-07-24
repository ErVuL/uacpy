"""The unified :class:`Field` result, the :class:`ResultStack`, and the
broadband-to-time-series IFFT synthesis helpers (kept with Field because
they construct it)."""

from __future__ import annotations

import copy as _copy
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union

from uacpy.core.constants import DEFAULT_SOUND_SPEED
from uacpy.core.exceptions import ConfigurationError

from uacpy.core.results._base import Result, _complex_to_db

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
    complex                    ``{source_depth, depth, range}``  Multi-source complex pressure (``.kind == 'pressure'``; ``.tl`` derives dB)
    =========================  ================================  =====================================

    ``data.shape`` matches the insertion order of :attr:`coords`. The
    canonical order is ``source_depth → depth → range → frequency``
    (or ``time``).

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
            # np.array (not asarray) so each Field owns its coord vectors —
            # slices/derived Fields never alias a parent's (or caller's) arrays.
            arr = np.atleast_1d(np.array(v, dtype=float))
            if arr.ndim != 1:
                raise ConfigurationError(
                    f"Field.coords[{name!r}]: must be 1-D; got shape {arr.shape}"
                )
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
        """Physical-quantity classification from ``(dtype, coords)``.

        One of ``'tl'`` (real, no time/frequency axis), ``'pressure'``
        (complex, no frequency axis), ``'transfer_function'``
        (complex, ``'frequency'`` axis present), or ``'time_series'``
        (real, ``'time'`` axis present). Independent of dimensionality —
        ``tl.at(depth=20)`` is still ``'tl'``."""
        axes = set(self.coords)
        if 'frequency' in axes and self.is_complex:
            return 'transfer_function'
        if 'time' in axes and not self.is_complex:
            return 'time_series'
        if self.is_complex:
            return 'pressure'
        return 'tl'

    # ── persistence ───────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialise this field to a plain dict for caching / round-trip.

        Values are numpy arrays and Python scalars (data preserves its
        real/complex dtype), so the result is directly picklable and
        ``np.savez``-able; convert the arrays to lists yourself for JSON.
        ``coords`` insertion order matches the data axes. ``kind`` is
        included for inspection but is recomputed by :meth:`from_dict`.
        Reconstruct with ``Field.from_dict(d)``.
        """
        return {
            'kind': self.kind,
            'data': self.data,
            'coords': {k: v for k, v in self.coords.items()},
            'pinned': dict(self.pinned),
            'model': self.model,
            'backend': self.backend,
            'source_depths': self.source_depths,
            'frequencies': self.frequencies,
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
        bits = [f"kind={self.kind!r}"]
        if self.model:
            bits.append(f"model={self.model!r}")
        f0 = self.f0
        if f0 is not None and 'frequency' not in self.coords:
            bits.append(f"f={f0:.3g} Hz")
        bits.append(f"axes=({', '.join(self.coords) or 'scalar'})")
        return f"Field({', '.join(bits)})"

    # ── value accessors ───────────────────────────────────────────────

    @property
    def tl(self) -> np.ndarray:
        """Transmission loss in dB at ``data.shape``.

        ``-20·log10(|data|)`` if data is complex, otherwise ``data``
        returned as-is (real data is taken to be already in dB).

        Raises :class:`AttributeError` for ``kind='time_series'`` —
        a time-domain trace is not transmission loss; use ``.data`` to
        read raw samples or ``.extract_tone(f)`` to recover a complex
        narrowband field first."""
        if self.kind == 'time_series':
            raise AttributeError(
                "Field.tl: time-domain trace is not transmission loss; "
                "use .data for raw samples or .extract_tone(f) to "
                "recover a complex narrowband field first"
            )
        if self.is_complex:
            return _complex_to_db(self.data)
        return np.asarray(self.data, dtype=float)

    @property
    def finite_tl(self) -> np.ndarray:
        """:attr:`tl` with the AT "no data" sentinel masked to ``NaN``.

        Bellhop fills cells that received no ray arrivals (the r=0 column and
        honest shadow zones) with zero pressure, which reads as
        ``NO_DATA_TL_DB`` (~600 dB). Reach for this accessor when *reducing*
        TL — ``field.finite_tl.mean()``, ``np.nanmax(field.finite_tl)``,
        colormaps — so the sentinel does not silently poison the statistic;
        :attr:`tl` returns the raw values (including the sentinel) unchanged."""
        from uacpy.core.constants import NO_DATA_TL_DB
        tl = np.array(self.tl, dtype=float)
        tl[tl >= NO_DATA_TL_DB - 1.0] = np.nan
        return tl

    @property
    def p(self) -> np.ndarray:
        """Complex pressure / transfer-function values.

        Raises when :attr:`data` is real — phase has been discarded."""
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
        if self.frequencies is not None:
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
        (its selected value lands in :attr:`pinned`)."""
        self._check_axes(kwargs)
        idx_map = {
            name: int(np.argmin(np.abs(self.coords[name] - float(v))))
            for name, v in kwargs.items()
        }
        return self._slice(idx_map)

    def isel(self, **kwargs) -> "Field":
        """Integer-index slice. Same semantics as :meth:`at` but the
        value is a positional index into the coord array."""
        self._check_axes(kwargs)
        return self._slice({name: int(i) for name, i in kwargs.items()})

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
        from uacpy.core._grid import collapse_axis
        method = kwargs.pop('method', 'linear')
        self._check_axes(kwargs)
        data = self.data
        coords = dict(self.coords)
        pinned = dict(self.pinned)
        order = list(self.coords)
        for name, value in kwargs.items():
            ax = order.index(name)
            data, vq = collapse_axis(data, coords[name], value, method, axis=ax)
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
        if coords:
            id_kwargs = self.id_kwargs()
            id_kwargs['frequencies'] = new_frequencies
            id_kwargs['source_depths'] = new_source_depths
            return Field(data=data, coords=coords, pinned=pinned, **id_kwargs)
        return self._spawn_scalar(
            data, pinned, new_frequencies, new_source_depths)

    def max(self) -> "Field":
        """Slice at the loudest field point.

        Complex / time-domain data: global argmax of ``|data|``. Real dB
        (``kind='tl'``): the *minimum* finite TL — smaller dB is louder —
        with ``NaN`` and the AT no-data sentinel excluded (via
        :attr:`finite_tl`). Every axis collapses to a pinned scalar; the
        returned Field has empty :attr:`coords`, 0-D :attr:`data`, and
        every original axis recorded in :attr:`pinned`."""
        if self.data.size == 0:
            raise ConfigurationError("Field.max: data is empty")
        if self.kind == 'tl':
            strength = -self.finite_tl          # loudest = smallest dB
        else:
            strength = np.abs(self.data)
        if not np.isfinite(strength).any():
            raise ConfigurationError(
                "Field.max: no finite samples (all NaN / no-data sentinel)")
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
        if new_coords:
            id_kwargs = self.id_kwargs()
            id_kwargs['frequencies'] = new_frequencies
            id_kwargs['source_depths'] = new_source_depths
            return Field(
                data=new_data,
                coords=new_coords,
                pinned=new_pinned,
                **id_kwargs,
            )
        return self._spawn_scalar(
            new_data, new_pinned, new_frequencies, new_source_depths,
        )

    def _spawn_scalar(
        self, new_data, new_pinned, frequencies=None, source_depths=None,
    ) -> "Field":
        # Scalar Field: data is 0-D, coords empty. Re-enter via __init__
        # by re-adding a phantom singleton coord, then immediately
        # dropping it — simpler: bypass the dict size check by allowing
        # empty coords here. We do so by constructing a Field via a
        # private path.
        f = Field.__new__(Field)
        Result.__init__(
            f,
            model=self.model,
            backend=self.backend,
            source_depths=(
                source_depths if source_depths is not None
                else self.source_depths
            ),
            frequencies=frequencies,
            phase_reference=self.phase_reference,
            model_source=self.model_source,
            metadata=dict(self.metadata),
        )
        f.coords = {}
        f.data = np.asarray(new_data)
        f.pinned = new_pinned
        return f

    def to_tl(self) -> "Field":
        """Return a real-dB Field via ``-20·log10(|data|)``.

        No-op when ``data`` is already real."""
        if not self.is_complex:
            return self
        return Field(
            data=_complex_to_db(self.data),
            coords=dict(self.coords),
            pinned=dict(self.pinned),
            **self.id_kwargs(),
        )

    def id_kwargs(self) -> dict:
        """Identification fields (model, backend, source depths, frequencies,
        phase reference, metadata) as a kwargs dict, for cloning them onto a
        :class:`Field` derived from this one. Public so downstream toolkits
        (e.g. :mod:`uacpy.sonar`) can carry provenance without hand-copying."""
        return dict(
            model=self.model,
            backend=self.backend,
            source_depths=self.source_depths,
            frequencies=self.frequencies,
            phase_reference=self.phase_reference,
            model_source=self.model_source,
            metadata=dict(self.metadata),
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
        from uacpy.core.environment import Environment, Bathymetry
        if isinstance(bathymetry, Environment):
            bathymetry = bathymetry.bathymetry
        if isinstance(bathymetry, Bathymetry):
            bathymetry = bathymetry.to_pairs()
        bathy = np.asarray(bathymetry, dtype=float)
        if bathy.ndim != 2 or bathy.shape[1] != 2:
            raise ConfigurationError(
                f"Field.mask_below_seafloor: bathymetry must be shape "
                f"(N, 2) or an Environment; got array shape {bathy.shape}"
            )
        ranges = self.coords['range']
        depths = self.coords['depth']
        seafloor = np.interp(ranges, bathy[:, 0], bathy[:, 1])
        new_data = self.data.astype(
            np.complex128 if self.is_complex else np.float64, copy=True,
        )
        for j, sf in enumerate(seafloor):
            mask = depths > sf
            new_data[mask, j] = np.nan
        return Field(
            data=new_data,
            coords=dict(self.coords),
            pinned=dict(self.pinned),
            **self.id_kwargs(),
        )

    def resample_to(
        self,
        ranges: np.ndarray,
        depths: np.ndarray,
        *,
        method: str = 'linear',
    ) -> "Field":
        """Linearly resample onto a new ``(depth, range)`` grid.

        Requires the canonical 2-D layout ``coords == {'depth', 'range'}``.
        Complex data is interpolated component-wise. Out-of-bound queries
        return NaN."""
        if list(self.coords) != ['depth', 'range']:
            raise ConfigurationError(
                "Field.resample_to: requires canonical ['depth', 'range'] "
                f"coords; got {list(self.coords)}"
            )
        from scipy.interpolate import RegularGridInterpolator
        new_ranges = np.atleast_1d(np.asarray(ranges, dtype=float))
        new_depths = np.atleast_1d(np.asarray(depths, dtype=float))
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
        window: str = "hann",
        nfft: Optional[int] = None,
        t_start: Optional[float] = None,
    ) -> "Field":
        """Single-trace IFFT of ``H(d, r, :)`` at a chosen ``(depth, range)``.

        Requires ``coords == {'depth', 'range', 'frequency'}``. Returns
        a single-point ``Field`` with ``coords={'time': ...}``."""
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

        Requires ``coords == {'depth', 'range', 'frequency'}``."""
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
                "only, e.g. lfm_chirp(...)[1]."
            )
        return _synthesize_time_series(
            self,
            source_waveform=source_waveform,
            sample_rate=sample_rate,
            t_start=t_start, window=window, nfft=nfft,
        )

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
        self, *, axes=None, title=None, figsize=(8, 6), **kwargs,
    ):
        """Plot the transfer function ``H(f)`` at one receiver cell as two
        stacked panels: modulus in dB (``20·log10|H|``, top) over phase
        (bottom), sharing the frequency axis.

        Reduce-then-plot: call on a field already sliced to one ``(depth,
        range)`` cell (``H.at(depth=…, range=…).plot_transfer_function()``); a
        single-receiver field plots directly (singleton axes are squeezed).
        Pass ``axes=(ax_mag, ax_phase)`` to draw into existing axes.
        Returns ``(fig, (ax_mag, ax_phase))``."""
        import matplotlib.pyplot as plt
        spec = self._reduce_to_spectrum('plot_transfer_function')
        if not spec.is_complex:
            raise ConfigurationError(
                "Field.plot_transfer_function: needs a complex H(f) (a real "
                "dB spectrum has no phase panel) — plot it with "
                ".plot(value='tl') instead."
            )
        owns_fig = axes is None
        if owns_fig:
            fig, (ax_mag, ax_phase) = plt.subplots(
                2, 1, sharex=True, figsize=figsize)
        else:
            ax_mag, ax_phase = axes
            fig = ax_mag.figure
        spec.plot(value='mag_db', ax=ax_mag, title=title, **kwargs)
        spec.plot(value='phase', ax=ax_phase, **kwargs)
        ax_phase.set_title('')       # keep the title/pinned subtitle on top only
        ax_mag.set_xlabel('')        # shared axis: label only the bottom panel
        if owns_fig:
            # plot_field skips its credit when handed an ``ax``; draw the
            # model-source footnote once, from the (attributed) source Field.
            from uacpy.visualization.plots._common import _draw_result_credit
            _draw_result_credit(fig, self)
        return fig, (ax_mag, ax_phase)

    def plot_impulse_response(
        self, *, ax=None, title=None, window: str = 'hann',
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
        trace = grid.to_time_trace(window=window)
        owns_fig = ax is None
        if owns_fig:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        trace.plot(ax=ax, title=title, **kwargs)
        if owns_fig:
            # Draw the model-source footnote from the (attributed) source Field
            # — the IFFT trace does not carry the model provenance.
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

        The ``2·X/Σwin`` tone estimator assumes a non-DC, non-Nyquist
        bin; at exactly 0 Hz or the Nyquist frequency the doubling
        overestimates the amplitude by 2×.
        """
        if list(self.coords) != ['depth', 'range', 'time']:
            raise ConfigurationError(
                "Field.extract_tone: requires canonical "
                "['depth', 'range', 'time'] coords; got "
                f"{list(self.coords)}"
            )
        if window == 'hann':
            win = np.hanning(self.n_times)
        elif window == 'hamming':
            win = np.hamming(self.n_times)
        elif window == 'blackman':
            win = np.blackman(self.n_times)
        elif window == 'none':
            win = np.ones(self.n_times)
        else:
            raise ConfigurationError(
                f"Field.extract_tone: unknown window={window!r}"
            )
        windowed = self.data * win
        spec = np.fft.rfft(windowed, axis=-1)
        freqs = np.fft.rfftfreq(self.n_times, self.dt)
        k = int(np.argmin(np.abs(freqs - frequency)))
        amp = 2.0 * spec[..., k] / np.sum(win)
        return Field(
            data=amp,
            coords={'depth': self.coords['depth'], 'range': self.coords['range']},
            pinned={**self.pinned, 'frequency': float(freqs[k])},
            **self.id_kwargs(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# ResultStack — for non-Field stacks (e.g. multi-source Rays / Arrivals)
# ─────────────────────────────────────────────────────────────────────────────


_RESULTSTACK_VARYING_ATTR = {
    'source_depth': 'source_depths',
    'frequency':    'frequencies',
}


class ResultStack:
    """Stack of typed :class:`Result` slabs along one coordinate.

    Bundles a list of slabs together with the coordinate vector along
    which they are stacked. The coordinate can be a :class:`Result`
    field (``source_depth``, ``frequency``) or an external parameter
    the user varied. Every slab carries the same concrete type,
    ``model``, and ``backend``, and the same identification along
    every axis *except* the stacking axis.

    For gridded results, prefer adding the varying axis to
    :class:`Field` ``coords`` instead (e.g. multi-source TL as a Field
    with ``coords={'source_depth', 'depth', 'range'}``); this stack is
    intended for non-Field results (multi-source ``Rays`` /
    ``Arrivals``).

    Construction
    ------------
    ``ResultStack(slabs, coordinate, coordinate_name='source_depth')``

    Access
    ------
    ``stack[i]``                              i-th slab
    ``for c, slab in stack: …``               iterate ``(coordinate, slab)`` pairs
    ``stack.at(<coordinate_name>=value)``     nearest-label lookup
    ``len(stack)``                            number of slabs
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

        def _arrays_equal(a, b):
            if a is None and b is None:
                return True
            if a is None or b is None:
                return False
            a = np.asarray(a)
            b = np.asarray(b)
            return a.shape == b.shape and np.array_equal(a, b)

        for attr in shared_attrs:
            ref = getattr(first, attr, None)
            eq = _arrays_equal if isinstance(ref, np.ndarray) else (lambda a, b: a == b)
            for i, s in enumerate(slabs[1:], start=1):
                val = getattr(s, attr, None)
                if not eq(ref, val):
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
    def metadata(self) -> Dict[str, Any]:
        return self.slabs[0].metadata

    def __len__(self) -> int:
        return self.n_slabs

    def copy(self) -> "ResultStack":
        """Deep copy (symmetric with :class:`Result` and the carriers)."""
        return _copy.deepcopy(self)

    def __getitem__(self, index: int) -> Result:
        return self.slabs[int(index)]

    def __iter__(self):
        for c, slab in zip(self.coordinate, self.slabs):
            yield float(c), slab

    def at(self, **kwargs) -> Result:
        """Select the slab nearest a value on the stacking axis.

        Pass exactly the stacking-axis keyword (``<coordinate_name>=<value>``);
        returns the slab whose coordinate is closest to ``value``.
        """
        if len(kwargs) != 1 or self.coordinate_name not in kwargs:
            raise ConfigurationError(
                f"ResultStack.at(): pass exactly the stacking-axis "
                f"keyword ({self.coordinate_name}=<value>); got "
                f"{list(kwargs)}"
            )
        target = float(kwargs[self.coordinate_name])
        idx = int(np.argmin(np.abs(self.coordinate - target)))
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
    def tl(self) -> np.ndarray:
        """Transmission loss stacked along the coordinate axis — shape
        ``(n_slabs, *slab.tl.shape)`` — so generic code can read ``result.tl``
        whether one or many source depths were requested. Requires Field slabs.
        """
        if not hasattr(self.slabs[0], 'tl'):
            raise ConfigurationError(
                f"ResultStack.tl: slabs are {self.slab_type.__name__}, not "
                f"Field — no transmission loss. Pick a slab with stack[i] or "
                f"stack.at({self.coordinate_name}=...)."
            )
        return np.stack([s.tl for s in self.slabs], axis=0)

    def plot(self, **kwargs):
        """Plot every slab as a labelled panel grid (Field stacks), delegating
        to :func:`uacpy.visualization.plot_result`."""
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
) -> "Field":
    """IFFT one (depth, range) cell of a broadband Field → time-domain trace Field.

    Evaluates the Fourier synthesis ``p(t) = 2·Re Σ H(f_k)·S(f_k)·
    e^{2πi f_k t}·df`` — a Riemann sum of the continuous inverse
    transform, so the amplitude is independent of ``nfft`` and of the
    bin grid. ``source_spectrum`` must therefore be the *continuous*
    source spectrum sampled at the Field frequencies (a raw DFT times
    the source sampling interval); ``None`` synthesizes the
    band-limited impulse response.

    Places each model frequency at bin ``round(f/df)`` (with df capped at
    1 Hz for a ≥ 1-second window); when ``df`` is finer than the data
    spacing, demodulates by ``r/c0`` so the spectrum can be interpolated
    in baseband without ghost echoes, then re-modulates to land the
    arrival at the requested ``t_start``. Always sizes ``nfft`` so the
    largest frequency bin sits below Nyquist.
    """
    data = tf.data                                # (n_d, n_r, n_f)
    freqs = np.asarray(tf.coords['frequency'], dtype=float)
    depths = tf.coords['depth']
    ranges = tf.coords['range']
    n_d, n_r, n_freq = data.shape

    if n_freq < 2:
        raise ConfigurationError(
            f"_ifft_to_trace: need at least 2 frequencies for IFFT; got {n_freq}"
        )

    if tf.phase_reference == 'time_domain_native':
        raise ConfigurationError(
            "_ifft_to_trace: phase_reference='time_domain_native' is not a "
            "frequency-domain transfer function; the producing model "
            "(SPARC) returned p(t) directly — read the time-domain Field "
            "from RunMode.TIME_SERIES instead of synthesising via IFFT"
        )

    d_idx = (
        int(np.argmin(np.abs(depths - depth))) if depth is not None
        else n_d // 2
    )
    r_idx = (
        int(np.argmin(np.abs(ranges - range))) if range is not None
        else 0
    )
    actual_depth = float(depths[d_idx])
    actual_range = float(ranges[r_idx])

    spectrum = data[d_idx, r_idx, :].copy()
    spectrum = np.nan_to_num(spectrum, nan=0.0)

    df_data = float(freqs[1] - freqs[0])
    df = min(df_data, 1.0)               # cap at 1 Hz for ≥ 1-second window

    bin_indices = np.floor(freqs / df + 0.5).astype(int)
    max_bin = int(bin_indices[-1])

    if nfft is None:
        nfft_min = max(int(tf.metadata.get('n_samples', 0)) or 0, 4 * n_freq)
        nfft_target = max(nfft_min, 2 * max_bin + 2)
        if sample_rate is not None:
            nfft_target = max(nfft_target, int(np.ceil(sample_rate / df)))
        nfft = 1
        while nfft < nfft_target:
            nfft *= 2
        if nfft > _MAX_SYNTHESIS_NFFT:
            raise ConfigurationError(
                f"synthesize_time_series: the requested grid implies an "
                f"{nfft:,}-sample output (~{nfft * 16 / 1e9:.1f} GB), above the "
                f"{_MAX_SYNTHESIS_NFFT:,}-sample safety cap. This is driven by "
                f"sample_rate={sample_rate!r} Hz against a frequency resolution "
                f"df={df:.4g} Hz (length ~ sample_rate/df). Lower sample_rate, "
                f"widen df (coarser frequency grid / shorter window), or pass an "
                f"explicit nfft= if you really need an output this large.",
                remediation="A typical fix is a smaller sample_rate.",
            )

    if window == 'hann':
        win = np.hanning(n_freq)
    elif window == 'hamming':
        win = np.hamming(n_freq)
    elif window == 'blackman':
        win = np.blackman(n_freq)
    elif window == 'tukey':
        from scipy.signal import windows
        win = windows.tukey(n_freq, alpha=0.5)
    elif window == 'none':
        win = np.ones(n_freq)
    else:
        raise ConfigurationError(
            f"_ifft_to_trace: unknown window={window!r}; "
            "valid: 'hann', 'hamming', 'blackman', 'tukey', 'none'"
        )

    dt = 1.0 / (nfft * df)

    if t_start is None:
        T_window = nfft * dt
        lead = min(0.5 * T_window, 0.25)
        anchor_speed = float(tf.metadata.get(
            'c_min',
            tf.metadata.get('c0', DEFAULT_SOUND_SPEED),
        ))
        t_start = max(0.0, actual_range / anchor_speed - lead)

    spectrum = spectrum * win
    if source_spectrum is not None:
        spectrum = spectrum * np.asarray(source_spectrum)

    padded = np.zeros(nfft, dtype=complex)
    min_bin = int(bin_indices[0])
    max_bin_fill = int(bin_indices[-1])

    if df < df_data * 0.99 and n_freq >= 4:
        c0 = tf.metadata.get('c0', DEFAULT_SOUND_SPEED)
        t_demod = actual_range / c0
        demod = np.exp(1j * 2.0 * np.pi * freqs * t_demod)
        spec_demod = spectrum * demod

        from scipy.interpolate import interp1d
        fill_bins = np.arange(min_bin, min(max_bin_fill + 1, nfft))
        fill_freqs = fill_bins * df
        re_interp = interp1d(freqs, spec_demod.real, kind='linear',
                             bounds_error=False, fill_value=0.0)
        im_interp = interp1d(freqs, spec_demod.imag, kind='linear',
                             bounds_error=False, fill_value=0.0)
        spec_interp = re_interp(fill_freqs) + 1j * im_interp(fill_freqs)
        remod = np.exp(1j * 2.0 * np.pi * fill_freqs * (t_start - t_demod))
        padded[fill_bins] = spec_interp * remod
    else:
        spectrum = spectrum * np.exp(1j * 2.0 * np.pi * freqs * t_start)
        valid = (bin_indices >= 0) & (bin_indices < nfft)
        padded[bin_indices[valid]] = spectrum[valid]

    # ifft carries 1/nfft; ×(nfft·df) turns the bin sum into ∫…df
    result = 2.0 * np.real(np.fft.ifft(padded)) * (nfft * df)
    time = t_start + np.arange(nfft) * dt

    return Field(
        data=result,
        coords={'time': time},
        pinned={'depth': actual_depth, 'range': actual_range},
        model=tf.model,
        backend=tf.backend,
        source_depths=tf.source_depths,
        frequencies=tf.frequencies,
        phase_reference=tf.phase_reference,
        model_source=tf.model_source,
        metadata={'window': window, 'source_model': tf.model},
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
    if sample_rate <= 0:
        raise ConfigurationError(
            f"_synthesize_time_series: sample_rate must be positive; "
            f"got {sample_rate}"
        )

    tf_freqs = np.asarray(tf.coords['frequency'], dtype=float)
    if tf_freqs.size > 1:
        df_tf = float(np.diff(tf_freqs).mean())
        t_dft = 1.0 / df_tf if df_tf > 0 else float('inf')
        t_dur = n_src / float(sample_rate)
        # One-sample tolerance: float roundoff in Δf can make t_dft and
        # t_dur evaluate as < when they should be ==. Real wraparound
        # has t_dft short by *many* samples.
        if t_dft < t_dur - 1.0 / float(sample_rate):
            import warnings
            warnings.warn(
                f"synthesize_time_series: DFT period 1/Δf = {t_dft:.4f}s "
                f"is shorter than the source-waveform duration "
                f"{t_dur:.4f}s — the late-time response wraps back into "
                f"early bins. Refine the frequency grid to Δf ≤ "
                f"{1.0/t_dur:.4g} Hz, or shorten the waveform.",
                UserWarning, stacklevel=3,
            )

    # ×dt_src: raw DFT → continuous spectrum S(f), the unit _ifft_to_trace expects
    src_fft = np.fft.rfft(wf) / float(sample_rate)
    src_freqs = np.fft.rfftfreq(n_src, 1.0 / sample_rate)

    from scipy.interpolate import interp1d
    re_interp = interp1d(src_freqs, src_fft.real, bounds_error=False, fill_value=0.0)
    im_interp = interp1d(src_freqs, src_fft.imag, bounds_error=False, fill_value=0.0)
    freqs = tf.coords['frequency']
    source_spectrum = re_interp(freqs) + 1j * im_interp(freqs)

    n_d, n_r, _ = tf.data.shape
    depths = np.asarray(tf.coords['depth'])
    ranges = np.asarray(tf.coords['range'])

    if t_start is None:
        t0_trace = _ifft_to_trace(
            tf, depth=float(depths[0]), range=float(ranges[0]),
            source_spectrum=source_spectrum,
            window=window, nfft=nfft, t_start=None,
            sample_rate=sample_rate,
        )
        t_start = float(t0_trace.coords['time'][0]) if t0_trace.n_times else 0.0

    out = None
    time_vec = None
    for di in range(n_d):
        for ri in range(n_r):
            tr = _ifft_to_trace(
                tf, depth=float(depths[di]), range=float(ranges[ri]),
                source_spectrum=source_spectrum,
                window=window, nfft=nfft, t_start=t_start,
                sample_rate=sample_rate,
            )
            if out is None:
                time_vec = tr.coords['time']
                out = np.zeros((n_d, n_r, tr.n_times), dtype=tr.data.dtype)
            out[di, ri, :] = tr.data

    # All cells share one time window anchored at (depths[0], ranges[0]);
    # arrivals for ranges further out than the window can hold wrap back
    # into early bins (DFT periodicity) — flag it rather than alias silently.
    if n_r > 1 and time_vec is not None and time_vec.size > 1:
        c0 = float(tf.metadata.get('c0', DEFAULT_SOUND_SPEED) or
                   DEFAULT_SOUND_SPEED)
        span_s = float(ranges.max() - ranges.min()) / c0
        window_s = float(time_vec[-1] - time_vec[0])
        if span_s > window_s:
            import warnings
            warnings.warn(
                f"synthesize_time_series: the receiver range span "
                f"({ranges.max() - ranges.min():.0f} m ≈ {span_s:.2f}s of "
                f"travel time) exceeds the {window_s:.2f}s synthesis window "
                f"— far-range arrivals wrap back into early bins. Pass "
                f"output_duration ≥ {span_s:.2f}s to run() (or refine the "
                f"frequency grid to Δf ≤ {1.0/span_s:.3g} Hz).",
                UserWarning, stacklevel=3,
            )

    return Field(
        data=out,
        coords={'depth': depths, 'range': ranges, 'time': time_vec},
        model=tf.model,
        backend=tf.backend,
        source_depths=tf.source_depths,
        frequencies=tf.frequencies,
        phase_reference=tf.phase_reference,
        model_source=tf.model_source,
        metadata={'source_waveform_sample_rate': sample_rate, 'window': window},
    )
