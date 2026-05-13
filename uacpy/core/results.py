"""Typed result hierarchy for uacpy model outputs.

Hierarchy
---------
::

    Result
    ├── Field                       (all gridded outputs: pressure / TL / H(f) / p(t) / trace)
    ├── Arrivals                    (per-receiver list of arrivals)
    ├── Rays                        (per-source list of ray paths)
    ├── Modes                       (Kraken normal modes — depth eigenfunctions)
    ├── Covariance                  (OASN hydrophone × hydrophone covariance)
    ├── Replicas                    (OASN MFP frequency-domain Green's-function templates)
    └── ReflectionCoefficient       (R(theta) at a boundary)

:class:`Field` is the single container for everything gridded. The
dtype + which keys are in :attr:`Field.coords` tell you the physical
meaning of a field instance:

* ``complex`` + ``{depth, range}``             — narrowband pressure
* ``real``    + ``{depth, range}``             — TL in dB
* ``complex`` + ``{depth, range, frequency}``  — broadband ``H(d, r, f)``
* ``real``    + ``{depth, range, time}``       — ``p(d, r, t)``
* ``real``    + ``{time}``                     — single-point trace

Slicing (:meth:`Field.at` / :meth:`Field.isel` / :meth:`Field.max`)
collapses named axes by dropping them from :attr:`coords` and
recording the chosen sample in :attr:`Field.pinned`.

Identification (``model``, ``backend``, ``source_depths``,
``frequencies``, ``phase_reference``) is on every :class:`Result`;
read directly via attributes (not via :attr:`Result.metadata`).

Ad-hoc bag — Result.metadata
----------------------------
``Result.metadata`` is a free-form dict for genuinely model-specific
extras that don't have a typed attribute:

* :class:`Bounce` ``ReflectionCoefficient`` — ``'brc_file'``,
  ``'irc_file'``, ``'c_low'``, ``'c_high'``, ``'rmax'``.
* :class:`OAST` field — ``'oast_native_ranges'``, ``'interpolated'``.
* :class:`RAM` results — ``'c0'``, ``'dr'``, ``'dz'``, ``'zmax'``,
  ``'Q'``, ``'T'``.
* :class:`KrakenField` results — ``'mode_coupling'``, ``'n_profiles'``.
"""

from __future__ import annotations

from enum import Enum
import numpy as np
from typing import TYPE_CHECKING, Optional, Dict, Any, List, Tuple, Union

if TYPE_CHECKING:
    from uacpy.core.absorption import Absorption  # noqa: F401

from uacpy.core.constants import PRESSURE_FLOOR, DEFAULT_SOUND_SPEED
from uacpy.core.exceptions import ConfigurationError


def _complex_to_db(data: np.ndarray) -> np.ndarray:
    """``-20·log10(|data|)`` with ``|data|`` clamped to :data:`PRESSURE_FLOOR`.

    Canonical TL conversion used by :attr:`Field.tl` and the metrics in
    :mod:`uacpy.core.metrics`. Preserves shape — no squeeze.
    """
    return -20.0 * np.log10(np.maximum(np.abs(data), PRESSURE_FLOOR))


class PhaseReference(str, Enum):
    """Phase convention of a complex transfer function ``H(f)``.

    Every uacpy wrapper normalises its native phase convention before
    storing data on a broadband :class:`Field`; downstream consumers
    (IFFT, time-series synthesis) only need to know whether the payload
    is in the engineering travelling-wave form or whether it lives in
    the time domain already. Inherits from ``str`` so
    ``ref == 'travelling_wave'`` works directly.

    Members
    -------
    TRAVELLING_WAVE
        ``H(f)`` carries the engineering propagator ``exp(-i k0 r)``;
        ``2*Re[ifft(H)]`` lands the causal arrival at ``t = r/c0``.
        Used by Bellhop, Scooter, OASES OAST/OASP, KrakenField, and RAM
        (mpiramS / Collins backends bake the carrier into the data).
    TIME_DOMAIN_NATIVE
        SPARC writes ``p(t)`` directly. ``H(f)`` is the FFT of the
        already-time-domain trace; consumers that want a time series
        should read the time-domain :class:`Field` from
        ``RunMode.TIME_SERIES`` instead.
    """
    TRAVELLING_WAVE = 'travelling_wave'
    TIME_DOMAIN_NATIVE = 'time_domain_native'


# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────


class Result:
    """Common base for every model output.

    Carries identification (``model``, ``backend``), the source context
    (``source_depths``, ``frequencies``), and a free-form ``metadata``
    dict for model-specific extras. Subclasses add the shape-specific
    payload and methods.

    Parameters
    ----------
    model : str
        Name of the wrapper class that produced this result (e.g. ``'RAM'``,
        ``'Bellhop'``, ``'KrakenField'``).
    backend : str, optional
        Concrete binary that ran (e.g. ``'mpiramS'``, ``'kraken.exe'``,
        ``'bellhop'``). Defaults to ``model.lower()`` when the wrapper is
        not a dispatcher.
    source_depths : array-like, optional
        Source depths used in the run (m). Stored as a 1-D ndarray.
    frequencies : array-like, optional
        Frequency vector in Hz, always stored as 1-D ndarray; length-1 for
        narrowband. Use ``result.f0`` to access the centre/single frequency
        as a scalar.
    metadata : dict, optional
        Model-specific extras (Q, T, dr, dz, n_modes, …).
    """

    # Lower-case tag used by ``visualization.plots.plot_result`` to dispatch
    # rendering and by callers that want a string discriminator. ``isinstance``
    # is the preferred check.
    field_type: str = ""

    def __init__(
        self,
        *,
        model: str = "",
        backend: Optional[str] = None,
        source_depths: Optional[np.ndarray] = None,
        frequencies: Optional[Union[float, np.ndarray]] = None,
        phase_reference: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.backend = backend if backend is not None else (model.lower() if model else "")
        self.source_depths = (
            np.atleast_1d(np.asarray(source_depths, dtype=float))
            if source_depths is not None else np.array([], dtype=float)
        )
        # Plural-only rule: ``frequencies`` is always a 1-D ndarray of length
        # ≥ 1, or ``None`` for results that have no frequency axis (e.g.
        # SPARC native time-domain). Scalar input auto-wraps to length 1.
        if frequencies is not None:
            self.frequencies: Optional[np.ndarray] = np.atleast_1d(
                np.asarray(frequencies, dtype=float)
            )
        else:
            self.frequencies = None
        self.phase_reference: Optional[str] = phase_reference
        self.metadata: Dict[str, Any] = dict(metadata) if metadata else {}

    # Convenience ------------------------------------------------------------

    @property
    def n_frequencies(self) -> int:
        return 0 if self.frequencies is None else int(len(self.frequencies))

    @property
    def f0(self) -> Optional[float]:
        """First / centre frequency in Hz, or ``None`` for time-domain results."""
        if self.frequencies is None or len(self.frequencies) == 0:
            return None
        return float(self.frequencies[0])

    def __repr__(self) -> str:
        cls = type(self).__name__
        bits = [f"model={self.model!r}"] if self.model else []
        if self.frequencies is not None and len(self.frequencies):
            if len(self.frequencies) == 1:
                bits.append(f"f={float(self.frequencies[0]):.3g} Hz")
            else:
                bits.append(f"n_f={len(self.frequencies)}")
        return f"{cls}({', '.join(bits)})"

    # Stub plot() — concrete subclasses override.
    def plot(self, **kwargs):
        from uacpy.visualization import plots
        return plots.plot_result(self, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Field — unified gridded result
# ─────────────────────────────────────────────────────────────────────────────


_CANONICAL_AXIS_ORDER = ('source_depth', 'depth', 'range', 'frequency', 'time')


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
    complex                    ``{source_depth, depth, range}``  Multi-source TL field
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
            arr = np.atleast_1d(np.asarray(v, dtype=float))
            if arr.ndim != 1:
                raise ConfigurationError(
                    f"Field.coords[{name!r}]: must be 1-D; got shape {arr.shape}"
                )
            normalised[name] = arr
        self.coords: Dict[str, np.ndarray] = normalised

        data = np.asarray(data)
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

    # ── value accessors ───────────────────────────────────────────────

    @property
    def tl(self) -> np.ndarray:
        """Transmission loss in dB at ``data.shape``.

        ``-20·log10(|data|)`` if data is complex, otherwise ``data``
        returned as-is (real data is taken to be already in dB)."""
        if self.is_complex:
            return _complex_to_db(self.data)
        return np.asarray(self.data, dtype=float)

    @property
    def p(self) -> np.ndarray:
        """Complex pressure / transfer-function values.

        Raises when :attr:`data` is real — phase has been discarded."""
        if not self.is_complex:
            raise AttributeError(
                "Field.p: data is real; complex pressure unavailable"
            )
        return self.data

    @property
    def magnitude(self) -> np.ndarray:
        if not self.is_complex:
            raise AttributeError(
                "Field.magnitude: requires complex data"
            )
        return np.abs(self.data)

    @property
    def phase(self) -> np.ndarray:
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
    def dt(self) -> float:
        t = self.coords.get('time')
        if t is None or t.size < 2:
            return 0.0
        return float(t[1] - t[0])

    @property
    def fs(self) -> float:
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

    def max(self) -> "Field":
        """Slice at the global argmax of ``|data|``.

        Every axis collapses to a pinned scalar; the returned Field has
        empty :attr:`coords`, 0-D :attr:`data`, and every original axis
        recorded in :attr:`pinned`."""
        if self.data.size == 0:
            raise ValueError("Field.max: data is empty")
        flat = int(np.argmax(np.abs(self.data)))
        idx = np.unravel_index(flat, self.data.shape)
        idx_map = {name: int(i) for name, i in zip(self.coords, idx)}
        return self._slice(idx_map)

    def _check_axes(self, kwargs: Dict[str, Any]) -> None:
        for name in kwargs:
            if name not in self.coords:
                raise ValueError(
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
                if not (0 <= i < size):
                    raise IndexError(
                        f"Field: index {i} out of range for axis "
                        f"{name!r} (size {size})"
                    )
                slicers.append(i)
                new_pinned[name] = float(self.coords[name][i])
            else:
                slicers.append(slice(None))
                new_coords[name] = self.coords[name]
        new_data = self.data[tuple(slicers)]
        return Field(
            data=new_data,
            coords=new_coords if new_coords else {},
            pinned=new_pinned,
            **self._id_kwargs(),
        ) if new_coords else self._spawn_scalar(new_data, new_pinned)

    def _spawn_scalar(self, new_data, new_pinned) -> "Field":
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
            source_depths=self.source_depths,
            frequencies=self.frequencies,
            phase_reference=self.phase_reference,
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
            **self._id_kwargs(),
        )

    def _id_kwargs(self) -> dict:
        return dict(
            model=self.model,
            backend=self.backend,
            source_depths=self.source_depths,
            frequencies=self.frequencies,
            phase_reference=self.phase_reference,
            metadata=dict(self.metadata),
        )

    # ── (depth, range) operations ─────────────────────────────────────

    def mask_below_seafloor(self, bathymetry) -> "Field":
        """Return a copy with samples below the seafloor set to NaN.

        Requires exactly the canonical 2-D layout
        ``coords == {'depth': ..., 'range': ...}``."""
        if list(self.coords) != ['depth', 'range']:
            raise ValueError(
                "Field.mask_below_seafloor: requires canonical "
                f"['depth', 'range'] coords; got {list(self.coords)}"
            )
        from uacpy.core.environment import Environment
        if isinstance(bathymetry, Environment):
            bathymetry = bathymetry.bathymetry
        bathy = np.asarray(bathymetry, dtype=float)
        if bathy.ndim != 2 or bathy.shape[1] != 2:
            raise ValueError(
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
            **self._id_kwargs(),
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
            raise ValueError(
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
            **self._id_kwargs(),
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
            raise ValueError(
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
            raise ValueError(
                "Field.synthesize_time_series: requires canonical "
                "['depth', 'range', 'frequency'] coords; got "
                f"{list(self.coords)}"
            )
        return _synthesize_time_series(
            self,
            source_waveform=source_waveform,
            sample_rate=sample_rate,
            t_start=t_start, window=window, nfft=nfft,
        )

    # ── time-domain only (requires 'time' coord) ──────────────────────

    def get_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """Real FFT along the time axis. Returns ``(freqs, X)``.

        Requires a ``'time'`` axis."""
        if 'time' not in self.coords:
            raise ValueError(
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
        time-domain Field. Requires ``coords == {'depth', 'range', 'time'}``."""
        if list(self.coords) != ['depth', 'range', 'time']:
            raise ValueError(
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
            raise ValueError(
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
            **self._id_kwargs(),
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
            raise ValueError("ResultStack: requires at least one slab")
        coord = np.atleast_1d(np.asarray(coordinate, dtype=float))
        if coord.size != len(slabs):
            raise ValueError(
                f"ResultStack: coordinate length ({coord.size}) does not "
                f"match number of slabs ({len(slabs)})"
            )
        types = {type(s) for s in slabs}
        if len(types) != 1:
            raise TypeError(
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
                    raise ValueError(
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

    def __getitem__(self, index: int) -> Result:
        return self.slabs[int(index)]

    def __iter__(self):
        for c, slab in zip(self.coordinate, self.slabs):
            yield float(c), slab

    def at(self, **kwargs) -> Result:
        if len(kwargs) != 1 or self.coordinate_name not in kwargs:
            raise TypeError(
                f"ResultStack.at(): pass exactly the stacking-axis "
                f"keyword ({self.coordinate_name}=<value>); got "
                f"{list(kwargs)}"
            )
        target = float(kwargs[self.coordinate_name])
        idx = int(np.argmin(np.abs(self.coordinate - target)))
        return self.slabs[idx]

    def __repr__(self) -> str:
        return (
            f"ResultStack[{self.slab_type.__name__}]"
            f"(n_slabs={self.n_slabs}, "
            f"{self.coordinate_name}={self.coordinate.tolist()})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Sparse / non-grid results
# ─────────────────────────────────────────────────────────────────────────────


def _arrival_kind(n_top: int, n_bot: int) -> str:
    if n_top >= 1 and n_bot >= 1:
        return 'both'
    if n_bot >= 1:
        return 'bottom'
    if n_top >= 1:
        return 'surface'
    return 'direct'


_BOUNCE_KINDS = ('direct', 'surface', 'bottom', 'both')


def _bounce_in_bounds(value: int, spec) -> bool:
    """Match ``value`` against an int (exact) or ``(lo, hi)`` tuple
    (closed range, ``None`` = unbounded). Shared with :class:`Rays`."""
    if spec is None:
        return True
    if isinstance(spec, int):
        return value == spec
    lo, hi = spec
    if lo is not None and value < lo:
        return False
    if hi is not None and value > hi:
        return False
    return True


def _bounce_predicate(kind, top, bot):
    """Build a predicate ``(n_top, n_bot) -> bool`` for bounce filtering.

    Shared by :class:`Arrivals` and :class:`Rays`. ``kind`` is one of
    :data:`_BOUNCE_KINDS` (or ``None``); ``top`` / ``bot`` are
    int / (lo, hi) / None specs matching :func:`_bounce_in_bounds`.
    """
    if kind is not None and kind not in _BOUNCE_KINDS:
        raise ValueError(
            f"bounce filter: kind={kind!r} not in {_BOUNCE_KINDS}"
        )

    def predicate(n_top: int, n_bot: int) -> bool:
        if kind is not None and _arrival_kind(n_top, n_bot) != kind:
            return False
        return (_bounce_in_bounds(n_top, top)
                and _bounce_in_bounds(n_bot, bot))

    return predicate


class Arrivals(Result):
    """Ray arrivals from Bellhop — a flat list of arrival events.

    Each arrival is a dict with: ``delay``, ``amplitude``, ``phase``,
    ``n_top_bounces``, ``n_bot_bounces``, ``src_angle``, ``rcv_angle``,
    ``kind`` ('direct' / 'surface' / 'bottom' / 'both'), plus the cell
    of origin (``src_idx``, ``depth_idx``, ``range_idx``) so multi-cell
    runs can be filtered back to one cell if needed.

    Mirrors the :class:`Rays` API surface: filter / chain / table.
    """
    field_type = "arrivals"

    def __init__(
        self,
        *,
        arrivals: Optional[List[Dict[str, Any]]] = None,
        by_receiver: Any = None,
        receiver_depths: np.ndarray,
        receiver_ranges: np.ndarray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.receiver_depths = np.atleast_1d(np.asarray(receiver_depths, dtype=float))
        self.receiver_ranges = np.atleast_1d(np.asarray(receiver_ranges, dtype=float))
        # Nested ``[src][depth][range] -> dict`` form that Bellhop's IO
        # produces and the broadband delay-and-sum path needs.
        self.by_receiver = by_receiver
        if arrivals is not None:
            self.arrivals = list(arrivals)
        else:
            self.arrivals = self._flatten_by_receiver(by_receiver)

    @staticmethod
    def _flatten_by_receiver(by_receiver: Any) -> List[Dict[str, Any]]:
        """Flatten Bellhop's ``arrivals_data[src][depth][range] -> dict``
        nesting into a single per-arrival list. Each emitted record
        carries its source/cell indices so callers can filter back."""
        if by_receiver is None:
            return []
        out: List[Dict[str, Any]] = []
        for s_idx, by_src in enumerate(by_receiver if isinstance(by_receiver, list) else []):
            for d_idx, by_depth in enumerate(by_src if isinstance(by_src, list) else []):
                for r_idx, cell in enumerate(by_depth if isinstance(by_depth, list) else []):
                    if not isinstance(cell, dict):
                        continue
                    delays = np.asarray(cell.get('delays', []))
                    if len(delays) == 0:
                        continue
                    amps = np.asarray(cell.get('amplitudes', np.zeros_like(delays)))
                    phs = np.asarray(cell.get('phases', np.zeros_like(delays)))
                    nt = np.asarray(cell.get('n_top_bounces', np.zeros(len(delays), int)))
                    nb = np.asarray(cell.get('n_bot_bounces', np.zeros(len(delays), int)))
                    sa = np.asarray(cell.get('src_angles', np.zeros_like(delays)))
                    ra = np.asarray(cell.get('rcv_angles', np.zeros_like(delays)))
                    for i in range(len(delays)):
                        n_top, n_bot = int(nt[i]), int(nb[i])
                        out.append({
                            'delay': float(delays[i]),
                            'amplitude': float(amps[i]),
                            'phase': float(phs[i]),
                            'n_top_bounces': n_top,
                            'n_bot_bounces': n_bot,
                            'src_angle': float(sa[i]),
                            'rcv_angle': float(ra[i]),
                            'kind': _arrival_kind(n_top, n_bot),
                            'src_idx': s_idx,
                            'depth_idx': d_idx,
                            'range_idx': r_idx,
                        })
        return out

    # Plot helpers — :func:`uacpy.visualization.plots.plot_arrivals` uses these.
    @property
    def depths(self) -> np.ndarray:
        return self.receiver_depths

    @property
    def ranges(self) -> np.ndarray:
        return self.receiver_ranges

    def __len__(self) -> int:
        return len(self.arrivals)

    def __iter__(self):
        return iter(self.arrivals)

    # Per-field bulk views ---------------------------------------------------

    @property
    def delays(self) -> np.ndarray:
        """Travel times (s) of every arrival in the list."""
        return np.asarray([a['delay'] for a in self.arrivals], dtype=float)

    @property
    def amplitudes(self) -> np.ndarray:
        """Amplitudes (linear) of every arrival in the list."""
        return np.asarray([a['amplitude'] for a in self.arrivals], dtype=float)

    @property
    def phases(self) -> np.ndarray:
        """Phases (rad) of every arrival in the list."""
        return np.asarray([a['phase'] for a in self.arrivals], dtype=float)

    # Filter / chain / sort --------------------------------------------------

    def _spawn(self, arrivals: List[Dict[str, Any]]) -> 'Arrivals':
        return Arrivals(
            arrivals=arrivals,
            receiver_depths=self.receiver_depths,
            receiver_ranges=self.receiver_ranges,
            model=self.model,
            backend=self.backend,
            source_depths=self.source_depths,
            frequencies=self.frequencies,
            metadata={k: v for k, v in self.metadata.items()
                      if k != 'arrivals_by_receiver'},
        )

    def filter(self, predicate) -> 'Arrivals':
        """Return a new ``Arrivals`` keeping arrivals for which
        ``predicate(arrival_dict)`` returns true."""
        return self._spawn([a for a in self.arrivals if predicate(a)])

    def filter_by_bounces(
        self,
        kind: Optional[str] = None,
        top: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None,
        bot: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None,
    ) -> 'Arrivals':
        """Subset by multipath component — same semantics as
        :meth:`Rays.filter_by_bounces`. ``kind`` ∈
        ``{'direct', 'surface', 'bottom', 'both'}``; ``top`` / ``bot`` are
        an int (exact) or ``(lo, hi)`` tuple (closed range, ``None`` =
        unbounded)."""
        pred = _bounce_predicate(kind, top, bot)
        return self.filter(
            lambda a: pred(int(a['n_top_bounces']), int(a['n_bot_bounces']))
        )

    def in_delay_window(
        self,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
    ) -> 'Arrivals':
        """Keep arrivals whose ``delay`` falls inside ``[t_min, t_max]``
        (each bound optional)."""
        def pred(a):
            d = a['delay']
            if t_min is not None and d < t_min:
                return False
            if t_max is not None and d > t_max:
                return False
            return True
        return self.filter(pred)

    def sorted_by_amplitude(self, descending: bool = True) -> 'Arrivals':
        """Return a copy sorted by ``amplitude`` (descending by default)."""
        return self._spawn(sorted(self.arrivals,
                                  key=lambda a: a['amplitude'],
                                  reverse=bool(descending)))

    def top_n_by_amplitude(self, n: int) -> 'Arrivals':
        """Keep the ``n`` loudest arrivals."""
        return self._spawn(self.sorted_by_amplitude(descending=True)
                           .arrivals[:int(n)])


class Rays(Result):
    """Ray paths from Bellhop / BellhopCUDA.

    Pure data container: a list of ray polylines plus the geometric
    context of the run. Filtering helpers return new ``Rays`` objects;
    none of them call back into a solver. To compute "rays at a
    receiver" use :meth:`uacpy.models.PropagationModel.compute_eigenrays`, which
    runs Bellhop's eigenray solver (``RunType='E'``).

    Attributes
    ----------
    rays : list
        Ray dicts with ``r``, ``z``, ``alpha``, ``n_top_bounces``,
        ``n_bot_bounces``. **Polyline coordinates ``r`` (range) and
        ``z`` (depth) are in metres**; ``alpha`` is the launch angle
        in degrees. The Bellhop reader
        (:func:`uacpy.io.oalib_reader.read_ray_file`) preserves
        Bellhop's native metre output, so downstream helpers such as
        :meth:`filter_by_miss_distance` work in metres without any
        unit detection.
    is_eigen : bool
        ``True`` for output of Bellhop's eigenray solver (``RunType='E'``),
        ``False`` for a regular ray fan (``RunType='R'``). Set by the
        wrapper from the run type, not by post-processing.
    receiver_depths, receiver_ranges : ndarray or None
        Receiver geometry the run targeted, when available. ``None``
        when the ``Rays`` came from a standalone reader call without
        receiver context.
    """
    field_type = "rays"

    def __init__(
        self,
        *,
        rays: List[Any],
        is_eigen: bool = False,
        receiver_depths: Optional[np.ndarray] = None,
        receiver_ranges: Optional[np.ndarray] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rays = list(rays)
        self.is_eigen = bool(is_eigen)
        self.receiver_depths = (
            np.atleast_1d(np.asarray(receiver_depths, dtype=float))
            if receiver_depths is not None else None
        )
        self.receiver_ranges = (
            np.atleast_1d(np.asarray(receiver_ranges, dtype=float))
            if receiver_ranges is not None else None
        )

    # ------------------------------------------------------------------
    # Filtering helpers — pure data subsets. ``is_eigen`` is preserved
    # (a subset of a fan stays a fan; a subset of eigenrays stays
    # eigenrays). None of these accept receiver coordinates: geometric
    # "rays at a receiver" is what ``PropagationModel.compute_eigenrays`` is for.
    # ------------------------------------------------------------------

    def filter(self, predicate) -> 'Rays':
        """Return a new ``Rays`` keeping rays for which ``predicate(ray)`` is true."""
        kept = [r for r in self.rays if predicate(r)]
        return self._spawn(kept)

    def filter_by_bounces(
        self,
        kind: Optional[str] = None,
        top: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None,
        bot: Optional[Union[int, Tuple[Optional[int], Optional[int]]]] = None,
    ) -> 'Rays':
        """Subset by multipath component.

        ``kind`` ∈ ``{'direct', 'surface', 'bottom', 'both'}`` keeps a
        qualitative bounce class.

        ``top`` / ``bot`` further constrain the exact bounce count on
        each boundary:

        * ``None``       — any count
        * ``int``        — exact match (e.g. ``top=2``)
        * ``(lo, hi)``   — closed range; ``None`` on either end is
                           unbounded. ``bot=(1, None)`` keeps rays with
                           at least one bottom bounce; ``top=(0, 1)``
                           keeps 0–1 surface bounces.
        """
        pred = _bounce_predicate(kind, top, bot)
        return self.filter(
            lambda r: pred(
                int(r.get('n_top_bounces', 0) or 0),
                int(r.get('n_bot_bounces', 0) or 0),
            )
        )

    def filter_by_launch_angle(
        self,
        min_deg: Optional[float] = None,
        max_deg: Optional[float] = None,
    ) -> 'Rays':
        """Keep rays whose launch angle ``alpha`` is within ``[min_deg, max_deg]``."""
        def pred(ray):
            a = ray.get('alpha')
            if a is None:
                return False
            if min_deg is not None and a < min_deg:
                return False
            if max_deg is not None and a > max_deg:
                return False
            return True
        return self.filter(pred)

    def filter_nfirst(
        self,
        n: int = 10
    ) -> 'Rays':
        """Keep only the first ``n`` rays."""
        return self._spawn(self.rays[:n])

    def _miss_distance_to(
        self, ray, target_range_m: float, target_depth_m: float,
    ) -> Tuple[float, int]:
        """Closest-approach miss distance and its index along the polyline.

        Ray polylines are required to carry ``r`` / ``z`` in **metres**
        (see :class:`Rays` docstring). The Bellhop reader in
        :mod:`uacpy.io.oalib_reader` already preserves Bellhop's native
        metres, so no unit-detection heuristic is needed here.
        """
        r = np.asarray(ray.get('r', []))
        z = np.asarray(ray.get('z', []))
        if len(r) == 0:
            return float('inf'), 0
        d2 = (r - target_range_m) ** 2 + (z - target_depth_m) ** 2
        k = int(np.argmin(d2))
        return float(np.sqrt(d2[k])), k

    def _resolve_target(
        self,
        target_range_m: Optional[float],
        target_depth_m: Optional[float],
    ) -> Tuple[float, float]:
        """Default target to the receiver context when this Rays was built
        from a single-point eigenray query."""
        if target_range_m is None:
            if self.receiver_ranges is None or len(self.receiver_ranges) != 1:
                raise ValueError(
                    "Rays.miss-distance helpers: target_range_m must be "
                    "supplied unless this Rays carries a single-point "
                    "receiver context."
                )
            target_range_m = float(self.receiver_ranges[0])
        if target_depth_m is None:
            if self.receiver_depths is None or len(self.receiver_depths) != 1:
                raise ValueError(
                    "Rays.miss-distance helpers: target_depth_m must be "
                    "supplied unless this Rays carries a single-point "
                    "receiver context."
                )
            target_depth_m = float(self.receiver_depths[0])
        return target_range_m, target_depth_m

    def filter_by_miss_distance(
        self,
        max_miss: float,
        target_range_m: Optional[float] = None,
        target_depth_m: Optional[float] = None,
    ) -> 'Rays':
        """Keep rays whose closest approach to the target is ``≤ max_miss``.

        Each kept ray gets a ``miss_distance_m`` entry attached. Target
        defaults to the single-point receiver this ``Rays`` was built for.
        """
        tr, td = self._resolve_target(target_range_m, target_depth_m)
        kept = []
        for ray in self.rays:
            miss, _ = self._miss_distance_to(ray, tr, td)
            if miss <= max_miss:
                ray = dict(ray)
                ray['miss_distance_m'] = miss
                kept.append(ray)
        return self._spawn(kept)

    def sorted_by_miss(
        self,
        target_range_m: Optional[float] = None,
        target_depth_m: Optional[float] = None,
    ) -> 'Rays':
        """Return rays sorted by ascending miss-distance to the target.

        Each ray gets ``miss_distance_m`` attached. Target defaults to
        the single-point receiver this ``Rays`` was built for. Compose
        with ``filter_nfirst`` to cap, or ``truncate_at_receiver`` to
        clip polylines.
        """
        tr, td = self._resolve_target(target_range_m, target_depth_m)
        scored = []
        for ray in self.rays:
            miss, _ = self._miss_distance_to(ray, tr, td)
            ray = dict(ray)
            ray['miss_distance_m'] = miss
            scored.append((miss, ray))
        scored.sort(key=lambda t: t[0])
        return self._spawn([r for _, r in scored])

    def top_n_by_miss(
        self,
        n: int,
        target_range_m: Optional[float] = None,
        target_depth_m: Optional[float] = None,
    ) -> 'Rays':
        """Return the ``n`` rays with smallest miss-distance to the target.

        Equivalent to ``self.sorted_by_miss(...).filter_nfirst(n)``.
        Target defaults to the single-point receiver this ``Rays`` was
        built for.
        """
        return self.sorted_by_miss(target_range_m, target_depth_m).filter_nfirst(n)

    def truncate_at_receiver(
        self,
        target_range_m: Optional[float] = None,
        target_depth_m: Optional[float] = None,
    ) -> 'Rays':
        """Clip each ray polyline at its closest-approach index.

        Target defaults to the single-point receiver this ``Rays`` was
        built for. Useful before plotting eigenrays so each path stops
        at the receiver instead of running off to its full extent.
        """
        tr, td = self._resolve_target(target_range_m, target_depth_m)
        clipped = []
        for ray in self.rays:
            miss, k = self._miss_distance_to(ray, tr, td)
            ray = dict(ray)
            ray['miss_distance_m'] = miss
            r = np.asarray(ray.get('r', []))
            z = np.asarray(ray.get('z', []))
            if k + 1 < len(r):
                ray['r'] = r[:k + 1]
                ray['z'] = z[:k + 1]
            clipped.append(ray)
        return self._spawn(clipped)

    def _spawn(self, rays: List[Any]) -> 'Rays':
        """Build a new ``Rays`` from a subset, preserving identification."""
        return Rays(
            rays=rays,
            is_eigen=self.is_eigen,
            receiver_depths=self.receiver_depths,
            receiver_ranges=self.receiver_ranges,
            model=self.model,
            backend=self.backend,
            source_depths=self.source_depths,
            frequencies=self.frequencies,
            metadata=dict(self.metadata),
        )


class Modes(Result):
    """Kraken normal modes — depth eigenfunctions of the Helmholtz operator.

    Attributes
    ----------
    k : ndarray, shape ``(n_modes,)`` complex
        Modal horizontal wavenumbers.
    phi : ndarray, shape ``(n_depths, n_modes)``
        Mode shapes sampled at ``depths``.
    depths : ndarray, shape ``(n_depths,)``
        Sampling depths.
    """
    field_type = "modes"

    def __init__(
        self,
        *,
        k: np.ndarray,
        phi: np.ndarray,
        depths: np.ndarray,
        n_modes: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.k = np.asarray(k)
        self.phi = np.asarray(phi)
        self.depths = np.atleast_1d(np.asarray(depths, dtype=float))
        if self.phi.shape != (len(self.depths), len(self.k)):
            raise ValueError(
                f"Modes.phi: shape {self.phi.shape} must equal "
                f"(len(depths), len(k)) = ({len(self.depths)}, {len(self.k)})"
            )
        self.n_modes = int(n_modes if n_modes is not None else len(self.k))

    @property
    def data(self) -> np.ndarray:      # alias for plot helpers
        return self.phi

    def first_n(self, n: int) -> "Modes":
        """Return a new :class:`Modes` containing only the first ``n`` modes.

        No-op when ``n >= self.n_modes``. ``k`` is sliced as ``k[:n]`` and
        ``phi`` as ``phi[:, :n]``; depths and identification metadata are
        preserved.
        """
        if n >= len(self.k):
            return self
        new_k = self.k[:n]
        new_phi = self.phi[:, :n]
        return Modes(
            k=new_k,
            phi=new_phi,
            depths=self.depths,
            n_modes=len(new_k),
            model=self.model, backend=self.backend,
            source_depths=self.source_depths,
            frequencies=self.frequencies,
            metadata=dict(self.metadata),
        )

    def compute_phase_speeds(self) -> np.ndarray:
        """Mode phase speeds ``v_p = ω / Re(k_r)`` in m/s.

        Raises
        ------
        ValueError
            If this :class:`Modes` instance has no frequency context
            (``self.f0 is None``); without a frequency the phase speed
            is undefined. Pass ``frequencies=…`` to the wrapper that
            built this object, or set it on the instance, before calling.
        """
        if self.f0 is None:
            raise ValueError(
                "Modes.compute_phase_speeds requires frequencies; got None"
            )
        omega = 2.0 * np.pi * self.f0
        return omega / np.real(self.k)

    def compute_group_velocity(self, other: "Modes") -> np.ndarray:
        """Approximate group velocity ``v_g = dω/dk`` using a second
        :class:`Modes` instance at a nearby frequency.

        Parameters
        ----------
        other : Modes
            Modes computed at a slightly different frequency.

        Returns
        -------
        v_g : ndarray, shape ``(min(self.n_modes, other.n_modes),)``
            Mode-by-mode group velocity in m/s. Modes that exist in only
            one of the two results are dropped (the array is truncated to
            the shared count).
        """
        f0_self, f0_other = self.f0, other.f0
        if f0_self is None or f0_other is None:
            raise ValueError(
                "Modes.compute_group_velocity: both Modes instances must "
                "have a frequency"
            )
        if f0_self == f0_other:
            raise ValueError(
                "Modes.compute_group_velocity: requires Modes at two distinct "
                "frequencies"
            )
        n = min(self.n_modes, other.n_modes)
        if n == 0:
            return np.array([])
        domega = 2.0 * np.pi * (f0_other - f0_self)
        dk = np.real(other.k[:n] - self.k[:n])
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(dk != 0, domega / dk, np.nan)

    def with_attenuation(
        self,
        alpha_db_per_m: Union[float, np.ndarray],
        *,
        sound_speed_z: Union[float, np.ndarray] = 1500.0,
        density_z: Union[float, np.ndarray] = 1.0,
        bottom=None,
    ) -> "Modes":
        """First-order modal attenuation perturbation.

        For mode ``m`` with horizontal wavenumber ``k_rm`` and depth
        eigenfunction ``ψ_m``, the imaginary part picks up

        ``α_m = (ω / k_rm) · ∫ α(z)/(c(z) ρ(z)) · |ψ_m|² dz / ∫ |ψ_m|² / ρ(z) dz``

        replacing any prior ``k.imag``. The ``1/ρ``-weighted denominator
        matches the Kraken-class normalisation ``∫|ψ|²/ρ dz = 1``.

        Parameters
        ----------
        alpha_db_per_m : float or ndarray
            Per-depth volume attenuation in dB/m, sampled on
            :attr:`depths`. Scalar broadcasts to every depth. Build one
            from an :class:`~uacpy.core.absorption.Absorption` via
            ``absorption.alpha_db_per_m(modes.f0, modes.depths)``.
        sound_speed_z : float or ndarray
            ``c(z)`` in m/s. Defaults to 1500.
        density_z : float or ndarray
            ``ρ(z)`` in **g/cm³** (matches :class:`BoundaryProperties`).
            Defaults to 1.0 (fresh-water reference; 1.025 for seawater).
        bottom : BoundaryProperties, optional
            Half-space below the water column. When supplied, adds an
            evanescent-tail bottom-attenuation contribution proportional
            to ``ψ²(D)``; ``bottom.attenuation`` is read in dB/λ_p and
            ``bottom.density`` in g/cm³.

        Returns
        -------
        Modes
            New :class:`Modes` instance with updated complex ``k``.
        """
        omega = 2.0 * np.pi * float(self.f0 or 0.0)
        if omega == 0.0:
            raise ValueError(
                "Modes.with_attenuation: requires Modes.f0 to be set."
            )
        a = np.asarray(alpha_db_per_m, dtype=float).ravel()
        if a.size == 1:
            a = np.full_like(self.depths, float(a.item()))
        elif a.shape != self.depths.shape:
            raise ValueError(
                f"Modes.with_attenuation: alpha shape {a.shape} "
                f"must match depths {self.depths.shape} (or scalar)."
            )
        c_arr = np.asarray(sound_speed_z, dtype=float).ravel()
        if c_arr.size == 1:
            c_arr = np.full_like(self.depths, float(c_arr.item()))
        elif c_arr.shape != self.depths.shape:
            raise ValueError(
                f"Modes.with_attenuation: sound_speed_z shape "
                f"{c_arr.shape} must match depths {self.depths.shape}"
            )
        rho_g = np.asarray(density_z, dtype=float).ravel()
        if rho_g.size == 1:
            rho_g = np.full_like(self.depths, float(rho_g.item()))
        elif rho_g.shape != self.depths.shape:
            raise ValueError(
                f"Modes.with_attenuation: density_z shape "
                f"{rho_g.shape} must match depths {self.depths.shape}"
            )
        rho_arr = rho_g * 1000.0  # g/cm³ → kg/m³
        a_neper = a * (np.log(10.0) / 20.0)
        phi_re = np.asarray(self.phi).real
        weight = phi_re ** 2
        norm = np.trapezoid(weight / rho_arr[:, None], self.depths, axis=0)
        norm = np.where(norm > 0, norm, 1.0)
        integrand = (a_neper / (c_arr * rho_arr))[:, None] * weight
        kr = np.real(self.k)
        kr_safe = np.where(kr > 0, kr, 1.0)
        alpha_m = (omega / kr_safe) * np.trapezoid(integrand, self.depths, axis=0) / norm
        if bottom is not None:
            from uacpy.core.environment import BoundaryProperties as _BP
            if not isinstance(bottom, _BP):
                raise TypeError(
                    "Modes.with_attenuation: bottom must be a "
                    f"BoundaryProperties; got {type(bottom).__name__}"
                )
            cb = float(bottom.sound_speed)
            rho_b = float(bottom.density) * 1000.0
            ab_neper_per_m = (
                float(bottom.attenuation) * np.log(10.0) / 20.0
                * float(self.f0) / cb
            )
            psi_D = phi_re[-1, :]
            kb = omega / cb
            gamma_m = np.sqrt(np.maximum(kr ** 2 - kb ** 2, 0.0))
            gamma_safe = np.where(gamma_m > 0, gamma_m, 1.0)
            alpha_bottom = (
                psi_D ** 2 * ab_neper_per_m * omega
                / (2.0 * kr_safe * gamma_safe * cb * rho_b)
            )
            alpha_m = alpha_m + alpha_bottom / norm
        new_k = kr + 1j * alpha_m
        return Modes(
            k=new_k, phi=self.phi, depths=self.depths,
            n_modes=self.n_modes,
            model=self.model, backend=self.backend,
            source_depths=self.source_depths,
            frequencies=self.frequencies,
            metadata=dict(self.metadata),
        )

    def modal_propagation_loss(
        self,
        *,
        source_depth: float,
        receiver_depths: np.ndarray,
        ranges_m: np.ndarray,
        source_density: float = 1.0,
    ) -> "Field":
        """Coherent complex pressure field built from the modal sum.

        Asymptotic far-field form of the cylindrical-source modal
        expansion (large ``k_m·r``):

        ``P(r, z_r) ≈ i·exp(−iπ/4) / (ρ_s·√(8πr)) · Σ_m
        ψ_m(z_s)·ψ_m(z_r) · exp(i k_m r) / √|k_m|``

        consistent with the ``∫|ψ|²/ρ dz = 1`` normalisation that Kraken
        and the analytic Pekeris helper use. Honors any imaginary
        ``k.imag`` set via :meth:`with_attenuation`.

        Parameters
        ----------
        source_depth, receiver_depths, ranges_m
            Source location and the target sample grid (m, m, m).
        source_density : float
            Water density at the source depth in **g/cm³** (matches
            :class:`BoundaryProperties`). Defaults to 1.0.

        Returns
        -------
        Field
            Complex narrowband ``Field`` with
            ``coords={'depth': receiver_depths, 'range': ranges_m}``.
        """
        z_s = float(source_depth)
        z_r = np.atleast_1d(np.asarray(receiver_depths, dtype=float))
        r = np.atleast_1d(np.asarray(ranges_m, dtype=float))
        phi = np.asarray(self.phi)
        is_complex = np.iscomplexobj(phi)
        if is_complex:
            phi_zs = np.array([
                np.interp(z_s, self.depths, phi[:, m].real)
                + 1j * np.interp(z_s, self.depths, phi[:, m].imag)
                for m in range(self.n_modes)
            ])
            phi_zr = np.column_stack([
                np.interp(z_r, self.depths, phi[:, m].real)
                + 1j * np.interp(z_r, self.depths, phi[:, m].imag)
                for m in range(self.n_modes)
            ])
        else:
            phi_zs = np.array([
                float(np.interp(z_s, self.depths, phi[:, m]))
                for m in range(self.n_modes)
            ])
            phi_zr = np.column_stack([
                np.interp(z_r, self.depths, phi[:, m])
                for m in range(self.n_modes)
            ])
        k = np.asarray(self.k)
        inv_sqrt_k = 1.0 / np.sqrt(np.abs(k))
        weights = phi_zs * inv_sqrt_k
        expikr = np.exp(1j * k[:, None] * r[None, :])
        contribution = (phi_zr * weights)[:, :, None] * expikr[None, :, :]
        P = contribution.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            sqrt_r = np.sqrt(r)
            sqrt_r = np.where(sqrt_r > 0, sqrt_r, 1.0)
        rho_s = float(source_density) * 1000.0  # g/cm³ → kg/m³
        pref = 1j * np.exp(-1j * np.pi / 4.0) / (rho_s * np.sqrt(8.0 * np.pi))
        P = pref * P / sqrt_r[None, :]
        return Field(
            data=P,
            coords={'depth': z_r, 'range': r},
            model=self.model, backend='modal_sum',
            source_depths=np.array([z_s]),
            frequencies=self.frequencies,
            metadata=dict(self.metadata),
        )


class Covariance(Result):
    """OASN spatial covariance matrix ``C(f, i, j)``.

    Hydrophone × hydrophone correlation per frequency, written by OASN with
    option ``N`` to a ``.xsm`` file. The eigenvectors of ``C[ifreq]`` are
    matched-field-processing replica vectors used for signal-subspace
    detection and localization.

    Attributes
    ----------
    covariance : ndarray, shape ``(n_frequencies, n_receivers, n_receivers)``
        Complex covariance matrices.
    receiver_positions : ndarray, optional, shape ``(n_receivers, 3)``
        ``(x, y, z)`` positions in metres.

    Notes
    -----
    To extract MFP signal-subspace eigenvectors call
    ``np.linalg.eigh(cov.covariance[ifreq])`` directly.
    """
    field_type = "covariance"

    def __init__(
        self,
        *,
        covariance: np.ndarray,
        receiver_positions: Optional[np.ndarray] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        cov = np.asarray(covariance)
        if cov.ndim != 3 or cov.shape[1] != cov.shape[2]:
            raise ValueError(
                f"Covariance.covariance: must be 3-D (n_freq, n_rcv, n_rcv); "
                f"got shape {cov.shape}"
            )
        self.covariance = cov
        if receiver_positions is not None:
            rp = np.asarray(receiver_positions, dtype=float)
            if rp.ndim != 2 or rp.shape[1] != 3 or rp.shape[0] != cov.shape[1]:
                raise ValueError(
                    f"Covariance.receiver_positions: must have shape "
                    f"(n_receivers={cov.shape[1]}, 3); got {rp.shape}"
                )
            self.receiver_positions = rp
        else:
            self.receiver_positions = None

    @property
    def n_frequencies(self) -> int:
        return int(self.covariance.shape[0])

    @property
    def n_receivers(self) -> int:
        return int(self.covariance.shape[1])

    def _replica_grid(self, replicas: "Replicas") -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Validate and reshape the replica field to ``(n_f, n_pts, n_rcv)``."""
        if replicas.replicas.shape[0] != self.n_frequencies:
            raise ValueError(
                f"Covariance MFP: frequency mismatch — "
                f"covariance has {self.n_frequencies} freq, "
                f"replicas has {replicas.replicas.shape[0]}."
            )
        if replicas.replicas.shape[-1] != self.n_receivers:
            raise ValueError(
                f"Covariance MFP: receiver-count mismatch — "
                f"covariance has {self.n_receivers}, "
                f"replicas has {replicas.replicas.shape[-1]}."
            )
        n_f = replicas.replicas.shape[0]
        nz, nx, ny = replicas.replicas.shape[1:4]
        flat = replicas.replicas.reshape(n_f, nz * nx * ny, self.n_receivers)
        return flat, (n_f, nz, nx, ny)

    @staticmethod
    def _normalise_weights(w: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(w, axis=-1, keepdims=True)
        norm = np.where(norm > 0, norm, 1.0)
        return w / norm

    def bartlett(self, replicas: "Replicas") -> np.ndarray:
        """Conventional Bartlett MFP ambiguity surface.

        ``B(z, x, y; f) = w(z, x, y; f)ᴴ · C(f) · w(z, x, y; f)``

        with ``w`` the replica vector at each candidate point, normalised
        to unit length.

        Returns
        -------
        ndarray, shape ``(n_freq, n_zr, n_xr, n_yr)``
            Real-valued ambiguity power. Argmax over the last three axes
            is the source-localisation peak.
        """
        flat, (n_f, nz, nx, ny) = self._replica_grid(replicas)
        out = np.empty((n_f, nz * nx * ny), dtype=float)
        for f in range(n_f):
            W = self._normalise_weights(flat[f])  # (n_pts, n_rcv)
            CW = self.covariance[f] @ W.T          # (n_rcv, n_pts)
            out[f] = np.real(np.einsum('pr,rp->p', W.conj(), CW))
        return out.reshape(n_f, nz, nx, ny)

    def mvdr(
        self,
        replicas: "Replicas",
        *,
        diagonal_loading: float = 1e-6,
    ) -> np.ndarray:
        """Minimum-Variance Distortionless-Response (Capon) MFP.

        ``M(z, x, y; f) = 1 / (wᴴ · (C(f) + δ·I)⁻¹ · w)`` with
        ``δ = diagonal_loading · trace(C(f))/N``. Small loading
        (~1e-6) stabilises rank-deficient covariance for sharp Capon
        peaks; larger loading (~0.1+) flattens the surface toward
        Bartlett for mismatch robustness. This is *not* the
        Cox/Zeskind/Owen white-noise-constrained processor (that
        requires per-replica Lagrange-multiplier bisection).
        """
        flat, (n_f, nz, nx, ny) = self._replica_grid(replicas)
        out = np.empty((n_f, nz * nx * ny), dtype=float)
        for f in range(n_f):
            C = self.covariance[f]
            tr = float(np.real(np.trace(C))) / max(C.shape[0], 1)
            Cload = C + diagonal_loading * tr * np.eye(C.shape[0])
            Cinv = np.linalg.inv(Cload)
            W = self._normalise_weights(flat[f])
            denom = np.einsum('pr,rs,ps->p', W.conj(), Cinv, W)
            out[f] = np.real(1.0 / np.where(np.abs(denom) > 0, denom, 1.0))
        return out.reshape(n_f, nz, nx, ny)


class Replicas(Result):
    """OASN matched-field-processing replicas.

    Frequency-domain Green's-function samples at every array element for
    every candidate source position. Written by OASN with option ``R`` to
    a ``.rpo`` file.

    Attributes
    ----------
    replicas : ndarray, shape ``(n_frequencies, n_zr, n_xr, n_yr, n_receivers)``
        Complex array responses per candidate source ``(z, x, y)``.
    replica_z, replica_x, replica_y : ndarray
        Coordinate axes of the candidate-source grid (m, m, m).
    receiver_positions : ndarray, optional, shape ``(n_receivers, 3)``
        ``(x, y, z)`` positions in metres.

    Notes
    -----
    To compute a Bartlett MFP ambiguity surface, contract a covariance
    estimate against the replica field across the array index.
    """
    field_type = "replicas"

    def __init__(
        self,
        *,
        replicas: np.ndarray,
        replica_z: np.ndarray,
        replica_x: np.ndarray,
        replica_y: np.ndarray,
        receiver_positions: Optional[np.ndarray] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        rep = np.asarray(replicas)
        if rep.ndim != 5:
            raise ValueError(
                f"Replicas.replicas: must be 5-D "
                f"(n_freq, n_zr, n_xr, n_yr, n_rcv); got shape {rep.shape}"
            )
        self.replicas = rep
        self.replica_z = np.atleast_1d(np.asarray(replica_z, dtype=float))
        self.replica_x = np.atleast_1d(np.asarray(replica_x, dtype=float))
        self.replica_y = np.atleast_1d(np.asarray(replica_y, dtype=float))
        expected = (
            len(self.replica_z), len(self.replica_x), len(self.replica_y),
        )
        if rep.shape[1:4] != expected:
            raise ValueError(
                f"Replicas.replicas: axes 1-3 {rep.shape[1:4]} must match "
                f"(n_zr, n_xr, n_yr) = {expected}"
            )
        if receiver_positions is not None:
            rp = np.asarray(receiver_positions, dtype=float)
            if rp.ndim != 2 or rp.shape[1] != 3 or rp.shape[0] != rep.shape[4]:
                raise ValueError(
                    f"Replicas.receiver_positions: must have shape "
                    f"(n_receivers={rep.shape[4]}, 3); got {rp.shape}"
                )
            self.receiver_positions = rp
        else:
            self.receiver_positions = None

    @property
    def n_frequencies(self) -> int:
        return int(self.replicas.shape[0])

    @property
    def n_receivers(self) -> int:
        return int(self.replicas.shape[4])

    @property
    def n_replica_points(self) -> int:
        return int(self.replicas.shape[1] * self.replicas.shape[2] * self.replicas.shape[3])


class ReflectionCoefficient(Result):
    """Angle-dependent reflection coefficient ``R(theta[, f])``.

    Unifies what Bounce and OASR produce. Used for both bottom (BRC) and
    top (TRC) reflection coefficients. ``R`` and ``phi`` may be 1-D
    (single-frequency) or 2-D (frequency-resolved); ``theta`` is always 1-D.

    Attributes
    ----------
    theta : ndarray, shape ``(n_angles,)``  — grazing angles in degrees
    R     : ndarray, shape ``(n_angles,)`` or ``(n_angles, n_frequencies)``
            — magnitude in [0, 1]
    phi   : ndarray, same shape as ``R`` — phase in radians
    frequencies : ndarray, optional, shape ``(n_frequencies,)`` — Hz
        Required when ``R`` is 2-D.
    is_broadband : bool — True iff ``R.ndim == 2``.
    """
    field_type = "reflection_coefficients"

    def __init__(
        self,
        *,
        theta: np.ndarray,
        R: np.ndarray,
        phi: np.ndarray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.theta = np.atleast_1d(np.asarray(theta, dtype=float))
        self.R = np.asarray(R, dtype=float)
        self.phi = np.asarray(phi, dtype=float)
        if self.R.ndim == 1:
            self.R = self.R.reshape(-1)
            self.phi = self.phi.reshape(-1)
            if not (len(self.theta) == len(self.R) == len(self.phi)):
                raise ValueError(
                    f"ReflectionCoefficient: theta/R/phi length mismatch "
                    f"({len(self.theta)}, {len(self.R)}, {len(self.phi)})"
                )
        elif self.R.ndim == 2:
            if self.R.shape != self.phi.shape:
                raise ValueError(
                    f"ReflectionCoefficient: R.shape {self.R.shape} != "
                    f"phi.shape {self.phi.shape}"
                )
            if self.R.shape[0] != len(self.theta):
                raise ValueError(
                    f"ReflectionCoefficient.R: axis 0 ({self.R.shape[0]}) "
                    f"must equal len(theta) ({len(self.theta)})"
                )
            if self.frequencies is None:
                raise ValueError(
                    "ReflectionCoefficient: 2-D R requires frequencies="
                )
            if self.R.shape[1] != len(self.frequencies):
                raise ValueError(
                    f"ReflectionCoefficient.R: axis 1 ({self.R.shape[1]}) "
                    f"must equal len(frequencies) ({len(self.frequencies)})"
                )
        else:
            raise ValueError(
                f"ReflectionCoefficient.R: must be 1-D or 2-D; "
                f"got shape {self.R.shape}"
            )

    @property
    def n_angles(self) -> int:
        return len(self.theta)

    @property
    def is_broadband(self) -> bool:
        return self.R.ndim == 2

    def at(
        self,
        *,
        angle: Optional[float] = None,
        frequency: Optional[float] = None,
    ) -> "ReflectionCoefficient":
        """Label-based slice along the angle and/or frequency axis.

        Either kwarg picks the nearest grid sample. ``frequency=`` is
        valid only for broadband (2-D) reflection coefficients; passing
        it on a narrowband instance raises ``ValueError``.
        """
        if frequency is not None and not self.is_broadband:
            raise ValueError(
                "ReflectionCoefficient.at: frequency= requires a broadband "
                "(2-D) reflection coefficient"
            )
        R = self.R
        phi = self.phi
        theta = self.theta
        freqs = self.frequencies
        if angle is not None:
            ai = int(np.argmin(np.abs(self.theta - angle)))
            theta = self.theta[ai:ai + 1]
            R = R[ai:ai + 1, ...] if R.ndim == 2 else R[ai:ai + 1]
            phi = phi[ai:ai + 1, ...] if phi.ndim == 2 else phi[ai:ai + 1]
        if frequency is not None:
            fi = int(np.argmin(np.abs(self.frequencies - frequency)))
            R = R[:, fi]
            phi = phi[:, fi]
            freqs = float(self.frequencies[fi])
        return ReflectionCoefficient(
            theta=theta, R=R, phi=phi,
            model=self.model, backend=self.backend,
            source_depths=self.source_depths,
            frequencies=freqs,
            metadata=dict(self.metadata),
        )

    @property
    def data(self) -> np.ndarray:
        return self.R

    @property
    def ranges(self) -> np.ndarray:   # convenience alias — angles double as the abscissa for plot helpers
        return self.theta

    @property
    def depths(self) -> np.ndarray:
        return np.array([0.0])


# ─────────────────────────────────────────────────────────────────────────────
# IFFT helpers used by Field
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
        raise ValueError(
            f"_ifft_to_trace: need at least 2 frequencies for IFFT; got {n_freq}"
        )

    if tf.phase_reference == 'time_domain_native':
        raise ValueError(
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
        nfft_min = max(int(tf.metadata.get('Nsam', 0)) or 0, 4 * n_freq)
        nfft_target = max(nfft_min, 2 * max_bin + 2)
        if sample_rate is not None:
            nfft_target = max(nfft_target, int(np.ceil(sample_rate / df)))
        nfft = 1
        while nfft < nfft_target:
            nfft *= 2

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
        raise ValueError(
            f"_ifft_to_trace: unknown window={window!r}; "
            "valid: 'hann', 'hamming', 'blackman', 'tukey', 'none'"
        )

    dt = 1.0 / (nfft * df)

    if t_start is None:
        T_window = nfft * dt
        lead = min(0.5 * T_window, 0.25)
        anchor_speed = float(tf.metadata.get(
            'cmin',
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

    result = 2.0 * np.real(np.fft.ifft(padded))
    time = t_start + np.arange(nfft) * dt

    return Field(
        data=result,
        coords={'time': time},
        pinned={'depth': actual_depth, 'range': actual_range},
        model=tf.model,
        backend=tf.backend,
        source_depths=tf.source_depths,
        frequencies=tf.frequencies,
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
    ``nfft`` is sized so the IFFT sample rate equals ``sample_rate``
    (rounded up to a power of two), so the returned trace is on the same
    sampling grid as the source pulse.
    """
    wf = np.asarray(source_waveform, dtype=float).ravel()
    n_src = len(wf)
    if n_src < 2:
        raise ValueError(
            f"_synthesize_time_series: source_waveform must have at least "
            f"2 samples; got {n_src}"
        )
    if sample_rate <= 0:
        raise ValueError(
            f"_synthesize_time_series: sample_rate must be positive; "
            f"got {sample_rate}"
        )

    src_fft = np.fft.rfft(wf)
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

    return Field(
        data=out,
        coords={'depth': depths, 'range': ranges, 'time': time_vec},
        model=tf.model,
        backend=tf.backend,
        source_depths=tf.source_depths,
        frequencies=tf.frequencies,
        metadata={'source_waveform_fs': sample_rate, 'window': window},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────────────


__all__ = [
    "Result",
    "PhaseReference",
    "Field",
    "ResultStack",
    "Arrivals", "Rays", "Modes",
    "Covariance", "Replicas",
    "ReflectionCoefficient",
]
