"""Typed result hierarchy for uacpy model outputs.

One class per output category — ``isinstance`` is the contract.

Convention
----------
**Spatial axes come first, the variable axis is trailing.** That means:

* ``PressureField.data`` — ``(n_d, n_r)`` or ``(n_d, n_r, n_f)``; real (dB TL)
  when ``units='dB'``, complex when ``units='complex'``
* ``TransferFunction.data`` — ``(n_d, n_r, n_f)`` complex
* ``TimeSeriesField.data``  — ``(n_d, n_r, n_t)`` real
* ``TimeTrace.data``        — ``(n_t,)`` real

The trailing-axis convention matches numpy's FFT default (``axis=-1``) so
``np.fft.ifft(H)`` on a ``TransferFunction`` directly produces the time-domain
counterpart with the same axis layout — no ``axis=`` argument or
``moveaxis`` shuffling needed.

For every grid-based result, ``data.shape[:2] == (len(depths), len(ranges))``
holds and is enforced in the constructor.

Hierarchy
---------
::

    Result
    ├── _GridResult                 (private mixin; spatial accessors)
    │   ├── PressureField           (units='dB' for TL, 'complex' for pressure)
    │   ├── TransferFunction
    │   └── TimeSeriesField
    ├── TimeTrace                   (single point, no spatial grid)
    ├── Arrivals                    (per-receiver list of arrivals)
    ├── Rays                        (per-source list of ray paths)
    ├── Modes                       (Kraken normal modes — depth eigenfunctions)
    ├── Covariance                  (OASN hydrophone × hydrophone covariance)
    ├── Replicas                    (OASN MFP frequency-domain Green's-function templates)
    └── ReflectionCoefficient       (R(theta) at a boundary)

Identification fields (``model``, ``backend``, ``source_depths``,
``frequency``/``frequencies``) are also mirrored into ``metadata`` so
callers can use either typed attributes or dict-style access.
"""

from __future__ import annotations

from enum import Enum
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union

from uacpy.core.constants import PRESSURE_FLOOR, DEFAULT_SOUND_SPEED


class PhaseReference(str, Enum):
    """Phase convention of a complex transfer function ``H(f)``.

    Every uacpy wrapper normalises its native phase convention before
    handing data to :class:`TransferFunction`; downstream consumers
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
        (mpiramS / Collins backends; the wrapper bakes the carrier into
        the data before tagging).
    TIME_DOMAIN_NATIVE
        SPARC writes ``p(t)`` directly. ``H(f)`` is the FFT of the
        already-time-domain trace; consumers that want a time series
        should take it from ``TimeSeriesField`` instead.
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
        self.metadata: Dict[str, Any] = dict(metadata) if metadata else {}
        if self.model:
            self.metadata.setdefault('model', self.model)
        if self.backend:
            self.metadata.setdefault('backend', self.backend)
        if self.source_depths is not None and len(self.source_depths):
            self.metadata.setdefault('source_depths', self.source_depths)
        if self.frequencies is not None:
            self.metadata.setdefault('frequencies', self.frequencies)

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

    def copy(self):
        """Deep copy of the result. Preserves type."""
        import copy as _copy
        return _copy.deepcopy(self)

    def tag(
        self,
        *,
        model: Optional[str] = None,
        backend: Optional[str] = None,
        source_depths: Optional[np.ndarray] = None,
        frequencies: Optional[Union[float, np.ndarray]] = None,
        phase_reference: Optional[str] = None,
        **extra_metadata,
    ) -> 'Result':
        """Attach harmonized identification to a Result built by a reader.

        Each model wrapper calls ``result.tag(model=…, backend=…,
        source_depths=…, frequencies=…, phase_reference=…, **extras)`` after
        the I/O reader returns. Scalar ``frequencies`` auto-wraps to a
        length-1 ndarray. Returns ``self`` for chaining.
        """
        if model is not None:
            self.model = model
            self.metadata['model'] = model
        if backend is not None:
            self.backend = backend
            self.metadata['backend'] = backend
        if source_depths is not None:
            self.source_depths = np.atleast_1d(np.asarray(source_depths, dtype=float))
            self.metadata['source_depths'] = self.source_depths
        if frequencies is not None:
            self.frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
            self.metadata['frequencies'] = self.frequencies
        if phase_reference is not None:
            self.metadata['phase_reference'] = phase_reference
        for k, v in extra_metadata.items():
            self.metadata[k] = v
        return self

    # Stub plot() — concrete subclasses override.
    def plot(self, **kwargs):
        from uacpy.visualization import plots
        return plots.plot_result(self, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Grid mixin: shared API for spatially-gridded results
# ─────────────────────────────────────────────────────────────────────────────


class _GridResult(Result):
    """Mixin for results that live on a regular ``(depth, range)`` grid.

    Provides ``depths``, ``ranges`` and the shape-checking + slicing
    helpers (``get_at_range``, ``get_at_depth``, ``get_value``, ``get_max``).
    Subclasses define the dtype/units of ``data`` and may add a third
    trailing axis (``frequencies`` or ``time``).
    """

    # Filled in by subclasses to give clearer error messages.

    def __init__(
        self,
        *,
        data: np.ndarray,
        depths: np.ndarray,
        ranges: np.ndarray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        data = np.asarray(data)
        depths = np.atleast_1d(np.asarray(depths, dtype=float))
        ranges = np.atleast_1d(np.asarray(ranges, dtype=float))
        if data.ndim < 2:
            raise ValueError(
                f"{type(self).__name__}.data: must be at least 2-D; "
                f"got shape {data.shape}"
            )
        if data.shape[:2] != (len(depths), len(ranges)):
            raise ValueError(
                f"{type(self).__name__}.data: first two axes {data.shape[:2]} "
                f"must equal (n_depths, n_ranges) = ({len(depths)}, {len(ranges)})"
            )
        self.data = data
        self.depths = depths
        self.ranges = ranges

    # Shape ------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def n_depths(self) -> int:
        return len(self.depths)

    @property
    def n_ranges(self) -> int:
        return len(self.ranges)

    # Slicing ----------------------------------------------------------------

    def get_at_range(self, range_m: float) -> np.ndarray:
        """Slice at the nearest range. Returns shape ``data.shape`` minus
        the range axis (axis 1)."""
        r_idx = int(np.argmin(np.abs(self.ranges - range_m)))
        if self.data.ndim == 2:
            return self.data[:, r_idx]
        return self.data[:, r_idx, ...]

    def get_at_depth(self, depth: float) -> np.ndarray:
        """Slice at the nearest depth. Returns shape ``data.shape`` minus
        the depth axis (axis 0)."""
        d_idx = int(np.argmin(np.abs(self.depths - depth)))
        return self.data[d_idx, ...]

    def get_value(
        self,
        range_m: float,
        depth: float,
        *,
        frequency: Optional[float] = None,
        time: Optional[float] = None,
    ) -> Any:
        """Return the value at the nearest ``(depth, range)`` (and, if the
        result has a third axis, the nearest ``frequency`` or ``time``)."""
        d_idx = int(np.argmin(np.abs(self.depths - depth)))
        r_idx = int(np.argmin(np.abs(self.ranges - range_m)))
        if self.data.ndim == 2:
            return self.data[d_idx, r_idx]
        if frequency is not None and self.frequencies is not None:
            k = int(np.argmin(np.abs(self.frequencies - frequency)))
            return self.data[d_idx, r_idx, k]
        if time is not None:
            t_axis = self.metadata.get('time')
            if t_axis is None:
                raise ValueError(
                    f"{type(self).__name__}.get_value: time= requested but "
                    "result has no time axis"
                )
            k = int(np.argmin(np.abs(np.asarray(t_axis) - time)))
            return self.data[d_idx, r_idx, k]
        raise ValueError(
            f"{type(self).__name__}.get_value: 3-D result requires "
            "frequency= or time="
        )

    def get_max(self) -> Tuple[float, float, float]:
        """Return ``(max_value, range_at_max, depth_at_max)`` over the
        spatial grid. For 3-D results, max is taken over all axes."""
        idx = np.unravel_index(np.argmax(self.data), self.data.shape)
        d_idx, r_idx = idx[0], idx[1]
        return (
            float(self.data[idx]),
            float(self.ranges[r_idx]),
            float(self.depths[d_idx]),
        )



# ─────────────────────────────────────────────────────────────────────────────
# Spatial gridded results
# ─────────────────────────────────────────────────────────────────────────────


class PressureField(_GridResult):
    """Acoustic pressure on a ``(depth, range)`` grid.

    Carries either complex pressure (``units='complex'``) or transmission
    loss in dB (``units='dB'``).

    ``data`` shape:
      * narrowband: ``(n_depths, n_ranges)``
      * broadband : ``(n_depths, n_ranges, n_frequencies)``

    Units
    -----
    ``units='complex'`` — model-native **source-normalized** complex pressure
    (dimensionless; equivalently Pa @ 1 m for a unit-amplitude monopole),
    such that ``-20·log10(|p|)`` is TL in dB re 1 m. Every uacpy model
    (Bellhop / Kraken family / Scooter / SPARC / Bounce / RAM / OASES)
    delivers this same convention. To convert to absolute Pa at the
    receiver, multiply by the source amplitude — uacpy does not carry one
    (``Source`` only has frequencies and depths, no level).

    ``units='dB'`` — transmission loss, real, dB re 1 m.

    Use :meth:`to_tl` to get a ``units='dB'`` view, or read the dB array
    on the fly via :attr:`tl` (available for both unit modes).
    """

    def __init__(
        self,
        *,
        data: np.ndarray,
        depths: np.ndarray,
        ranges: np.ndarray,
        units: str = 'complex',
        **kwargs,
    ):
        if units not in ('complex', 'dB'):
            raise ValueError(
                f"PressureField.units: must be 'complex' or 'dB', got {units!r}"
            )
        self.units = units
        super().__init__(
            data=data, depths=depths, ranges=ranges, **kwargs,
        )

    @property
    def field_type(self) -> str:
        return 'tl' if self.units == 'dB' else 'pressure'

    @property
    def is_broadband(self) -> bool:
        return self.data.ndim == 3

    def at_frequency(self, frequency: float) -> "PressureField":
        """For a broadband field, return the narrowband slice at the
        frequency nearest ``frequency``."""
        if not self.is_broadband:
            raise ValueError(
                "PressureField.at_frequency: only valid for broadband data"
            )
        if self.frequencies is None:
            raise ValueError(
                "PressureField.at_frequency: broadband data missing self.frequencies"
            )
        k = int(np.argmin(np.abs(self.frequencies - frequency)))
        return PressureField(
            data=self.data[..., k],
            depths=self.depths,
            ranges=self.ranges,
            units=self.units,
            model=self.model, backend=self.backend,
            source_depths=self.source_depths,
            frequencies=float(self.frequencies[k]),
            metadata=dict(self.metadata),
        )

    def to_tl(self) -> "PressureField":
        """Return a ``units='dB'`` view via ``-20·log10(|p|)``.

        No-op when ``self.units == 'dB'``.
        """
        if self.units == 'dB':
            return self
        p_abs = np.maximum(np.abs(self.data), PRESSURE_FLOOR)
        return PressureField(
            data=-20.0 * np.log10(p_abs),
            depths=self.depths,
            ranges=self.ranges,
            units='dB',
            model=self.model, backend=self.backend,
            source_depths=self.source_depths,
            frequencies=self.frequencies,
            metadata=dict(self.metadata),
        )

    @property
    def tl(self) -> np.ndarray:
        """Transmission loss in dB.

        ``self.data`` directly when ``units='dB'``; computed on the fly via
        ``-20·log10(|p|)`` when ``units='complex'``.
        """
        if self.units == 'dB':
            return self.data
        p_abs = np.maximum(np.abs(self.data), PRESSURE_FLOOR)
        return -20.0 * np.log10(p_abs)

    @property
    def magnitude(self) -> np.ndarray:
        if self.units == 'dB':
            raise AttributeError(
                "PressureField.magnitude: only valid when units='complex'"
            )
        return np.abs(self.data)

    @property
    def phase(self) -> np.ndarray:
        if self.units == 'dB':
            raise AttributeError(
                "PressureField.phase: only valid when units='complex'"
            )
        return np.angle(self.data)


class TransferFunction(_GridResult):
    """Complex broadband transfer function ``H(d, r, f)``.

    ``data`` shape: ``(n_depths, n_ranges, n_frequencies)`` complex.

    The ``phase_reference`` field is required and tells consumers (notably
    :meth:`to_time_trace` / :meth:`synthesize_time_series`) how to interpret
    the stored phase. See ``DOCUMENTATION.md §5.13``.
    """
    field_type = "transfer_function"

    def __init__(
        self,
        *,
        data: np.ndarray,
        depths: np.ndarray,
        ranges: np.ndarray,
        frequencies: np.ndarray,
        phase_reference: Union[PhaseReference, str],
        **kwargs,
    ):
        if frequencies is None or len(frequencies) == 0:
            raise ValueError(
                "TransferFunction: requires a non-empty frequencies vector"
            )
        super().__init__(
            data=data, depths=depths, ranges=ranges,
            frequencies=frequencies, **kwargs,
        )
        if data.ndim != 3:
            raise ValueError(
                f"TransferFunction.data: must be 3-D (n_d, n_r, n_f); "
                f"got shape {data.shape}"
            )
        if data.shape[2] != len(self.frequencies):
            raise ValueError(
                f"TransferFunction.data: axis 2 ({data.shape[2]}) does not "
                f"match len(frequencies) ({len(self.frequencies)})"
            )
        if phase_reference is None or phase_reference == "":
            raise ValueError(
                f"TransferFunction: requires a phase_reference; "
                f"valid: {[m.value for m in PhaseReference]}"
            )
        try:
            ref = PhaseReference(phase_reference)
        except ValueError:
            raise ValueError(
                f"TransferFunction: unknown phase_reference={phase_reference!r}; "
                f"valid: {[m.value for m in PhaseReference]}"
            ) from None
        self.phase_reference = ref
        # Mirror the string form into ``metadata`` so dict-style consumers
        # see the same value they wrote (``ref.value`` round-trips).
        self.metadata.setdefault('phase_reference', ref.value)

    # Time-domain conversion ------------------------------------------------
    def to_time_trace(
        self,
        depth: Optional[float] = None,
        range_m: Optional[float] = None,
        *,
        source_spectrum: Optional[np.ndarray] = None,
        window: str = "hann",
        nfft: Optional[int] = None,
        t_start: Optional[float] = None,
    ) -> "TimeTrace":
        """Single-trace IFFT of ``H(d, r, :)`` at a chosen ``(depth, range)``.

        Returns a :class:`TimeTrace` with ``data`` shape ``(n_t,)``.
        """
        return _ifft_to_trace(
            self, depth=depth, range_m=range_m,
            source_spectrum=source_spectrum,
            window=window, nfft=nfft, t_start=t_start,
        )

    def to_tl(self, frequency: Optional[float] = None) -> "PressureField":
        """Magnitude-in-dB view of this transfer function at one frequency.

        Picks the frequency-axis sample whose value is closest to
        ``frequency`` (or the centre frequency if ``None``) and returns a
        :class:`PressureField` (``units='dB'``) of shape ``(n_depth, n_range)``
        carrying ``-20 * log10(|H|)`` clamped to ``PRESSURE_FLOOR``.
        """
        if self.data.ndim < 3:
            raise ValueError(
                f"TransferFunction.to_tl: requires a 3-D (depth, range, freq) "
                f"array; got shape {self.data.shape}"
            )
        from uacpy.core.constants import PRESSURE_FLOOR
        freqs = np.asarray(self.frequencies, dtype=float)
        target = float(frequency) if frequency is not None else float(np.median(freqs))
        i_f = int(np.argmin(np.abs(freqs - target)))
        H = self.data[..., i_f]
        tl = -20.0 * np.log10(np.maximum(np.abs(H), PRESSURE_FLOOR))
        return PressureField(
            data=tl, depths=self.depths, ranges=self.ranges,
            units='dB',
            model=getattr(self, 'model', None),
            backend=getattr(self, 'backend', None),
            source_depths=getattr(self, 'source_depths', None),
            frequencies=float(freqs[i_f]),
            metadata=dict(self.metadata),
        )

    def synthesize_time_series(
        self,
        source_waveform: np.ndarray,
        sample_rate: float,
        *,
        t_start: Optional[float] = None,
        window: str = "hann",
        nfft: Optional[int] = None,
    ) -> "TimeSeriesField":
        """Convolve every grid trace with ``source_waveform`` to obtain a
        :class:`TimeSeriesField` shaped ``(n_d, n_r, n_t)``."""
        return _synthesize_time_series(
            self, source_waveform=source_waveform, sample_rate=sample_rate,
            t_start=t_start, window=window, nfft=nfft,
        )


class TimeSeriesField(_GridResult):
    """Real time-domain pressure on a ``(depth, range)`` grid.

    ``data`` shape: ``(n_depths, n_ranges, n_t)`` real.

    Time axis lives in ``self.time`` and ``self.metadata['time']`` (also
    ``dt``, ``fs``, ``nt``, ``t_start``).
    """
    field_type = "time_series"

    def __init__(
        self,
        *,
        data: np.ndarray,
        depths: np.ndarray,
        ranges: np.ndarray,
        time: np.ndarray,
        **kwargs,
    ):
        super().__init__(
            data=data, depths=depths, ranges=ranges, **kwargs,
        )
        if data.ndim != 3:
            raise ValueError(
                f"TimeSeriesField.data: must be 3-D (n_d, n_r, n_t); "
                f"got shape {data.shape}"
            )
        time = np.asarray(time, dtype=float)
        if data.shape[2] != len(time):
            raise ValueError(
                f"TimeSeriesField.data: axis 2 ({data.shape[2]}) does not "
                f"match len(time) ({len(time)})"
            )
        self.time = time
        # Mirror time-axis info into ``metadata`` for dict-style consumers.
        self.metadata.setdefault('time', time)
        self.metadata.setdefault('nt', int(len(time)))
        if len(time) >= 2:
            self.metadata.setdefault('dt', float(time[1] - time[0]))
            self.metadata.setdefault('fs', 1.0 / float(time[1] - time[0]))
        self.metadata.setdefault('t_start', float(time[0]) if len(time) else 0.0)

    @property
    def n_t(self) -> int:
        return self.data.shape[2]

    @property
    def dt(self) -> float:
        return float(self.time[1] - self.time[0]) if len(self.time) >= 2 else 0.0

    @property
    def fs(self) -> float:
        return 1.0 / self.dt if self.dt > 0 else 0.0

    def get_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(freqs, X)`` from a real FFT along the time axis.

        Mirrors :meth:`TimeTrace.spectrum` but operates on the full
        ``(n_d, n_r, n_t)`` grid: ``X.shape == (n_d, n_r, n_freq)`` with
        ``n_freq == n_t // 2 + 1`` and ``freqs == np.fft.rfftfreq(n_t, dt)``.
        """
        X = np.fft.rfft(self.data, axis=-1)
        freqs = np.fft.rfftfreq(self.n_t, self.dt)
        return freqs, X

    def get_trace(self, depth: float, range_m: float) -> "TimeTrace":
        """Extract the single-point :class:`TimeTrace` at the nearest
        ``(depth, range)``."""
        d_idx = int(np.argmin(np.abs(self.depths - depth)))
        r_idx = int(np.argmin(np.abs(self.ranges - range_m)))
        return TimeTrace(
            data=self.data[d_idx, r_idx, :].copy(),
            time=self.time,
            depth=float(self.depths[d_idx]),
            range_m=float(self.ranges[r_idx]),
            model=self.model, backend=self.backend,
            source_depths=self.source_depths,
            frequencies=self.frequencies,
            metadata=dict(self.metadata),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Time-domain — single point
# ─────────────────────────────────────────────────────────────────────────────


class TimeTrace(Result):
    """Real pressure ``p(t)`` at a single ``(depth, range)`` point.

    ``data`` shape: ``(n_t,)`` real.
    """
    field_type = "time_trace"

    def __init__(
        self,
        *,
        data: np.ndarray,
        time: np.ndarray,
        depth: float,
        range_m: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        data = np.asarray(data)
        if data.ndim != 1:
            raise ValueError(
                f"TimeTrace.data: must be 1-D; got shape {data.shape}"
            )
        time = np.asarray(time, dtype=float)
        if len(time) != len(data):
            raise ValueError(
                f"TimeTrace: len(time)={len(time)} does not match "
                f"len(data)={len(data)}"
            )
        self.data = data
        self.time = time
        self.depth = float(depth)
        self.range_m = float(range_m)
        self.metadata.setdefault('time', time)
        self.metadata.setdefault('nt', int(len(time)))
        self.metadata.setdefault('depth', self.depth)
        self.metadata.setdefault('range_m', self.range_m)
        if len(time) >= 2:
            self.metadata.setdefault('dt', float(time[1] - time[0]))
            self.metadata.setdefault('fs', 1.0 / float(time[1] - time[0]))
        self.metadata.setdefault('t_start', float(time[0]) if len(time) else 0.0)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def n_t(self) -> int:
        return int(len(self.data))

    @property
    def dt(self) -> float:
        return float(self.time[1] - self.time[0]) if len(self.time) >= 2 else 0.0

    @property
    def fs(self) -> float:
        return 1.0 / self.dt if self.dt > 0 else 0.0

    def spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(freqs, X)`` from a real FFT of ``self.data``."""
        X = np.fft.rfft(self.data)
        freqs = np.fft.rfftfreq(self.n_t, self.dt)
        return freqs, X

    # 1-element ndarrays for plotting helpers that expect a vector.
    @property
    def depths(self) -> np.ndarray:
        return np.array([self.depth])

    @property
    def ranges(self) -> np.ndarray:
        return np.array([self.range_m])


# ─────────────────────────────────────────────────────────────────────────────
# Sparse / non-grid results
# ─────────────────────────────────────────────────────────────────────────────


class Arrivals(Result):
    """Ray arrivals from Bellhop.

    Attributes
    ----------
    by_receiver : dict
        Nested dict keyed by ``(isz, ird, irr)`` returning a list of arrival
        records (each with ``amplitude``, ``phase``, ``delay``,
        ``src_angle``, ``rcv_angle``, ``n_top_bounces``,
        ``n_bot_bounces``).
        The structure mirrors what Bellhop's ``.arr`` reader produces.
    receiver_depths, receiver_ranges : ndarray
        Receiver grid.
    """
    field_type = "arrivals"

    def __init__(
        self,
        *,
        by_receiver: Any,
        receiver_depths: np.ndarray,
        receiver_ranges: np.ndarray,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.by_receiver = by_receiver
        self.receiver_depths = np.atleast_1d(np.asarray(receiver_depths, dtype=float))
        self.receiver_ranges = np.atleast_1d(np.asarray(receiver_ranges, dtype=float))
        # Mirror into ``metadata`` for dict-style access.
        self.metadata.setdefault('arrivals_by_receiver', self.by_receiver)
        self.metadata.setdefault('receiver_depths', self.receiver_depths)
        self.metadata.setdefault('receiver_ranges', self.receiver_ranges)

    # Plotting-helper accessors used by visualization.plots.plot_arrivals.
    @property
    def depths(self) -> np.ndarray:
        return self.receiver_depths

    @property
    def ranges(self) -> np.ndarray:
        return self.receiver_ranges

    # Convenience accessor.
    @property
    def arrivals_data(self) -> Any:
        """The per-receiver arrivals payload. Same as ``self.by_receiver``."""
        return self.by_receiver

    @property
    def delays(self) -> np.ndarray:
        """Travel times of every arrival at the (src=0, depth=0, range=0) cell."""
        return np.asarray(self.at().get('delays', []), dtype=float)

    @property
    def amplitudes(self) -> np.ndarray:
        """Amplitudes of every arrival at the (src=0, depth=0, range=0) cell."""
        return np.asarray(self.at().get('amplitudes', []), dtype=float)

    @property
    def phases(self) -> np.ndarray:
        """Phases (rad) of every arrival at the (src=0, depth=0, range=0) cell."""
        return np.asarray(self.at().get('phases', []), dtype=float)

    def at(self, range_idx: int = 0, depth_idx: int = 0,
           src_idx: int = 0) -> Dict[str, np.ndarray]:
        """Return the flat arrivals dict for one (source, depth, range) cell.

        Bellhop's ``.arr`` payload is nested as
        ``arrivals_data[src][depth][range] -> dict``; with a single-point
        receiver the leading dimensions all have length 1. This walks the
        nesting and returns the dict so users can access ``delays``,
        ``amplitudes``, ``phases``, ``n_top_bounces``,
        ``n_bot_bounces``, ``src_angles``, ``rcv_angles`` directly.
        """
        node = self.by_receiver
        for idx in (src_idx, depth_idx, range_idx):
            if isinstance(node, list):
                if not node:
                    return {}
                idx = min(idx, len(node) - 1)
                node = node[idx]
            elif isinstance(node, dict):
                break
        while isinstance(node, list) and node:
            node = node[0]
        return node if isinstance(node, dict) else {}

    def to_table(self, range_idx: int = 0, depth_idx: int = 0,
                 src_idx: int = 0) -> List[Dict[str, float]]:
        """Return a flat list of per-arrival records (one dict per arrival).

        Each record has keys: ``delay``, ``amplitude``, ``phase``,
        ``n_top_bounces``, ``n_bot_bounces``, ``src_angle``,
        ``rcv_angle``, ``kind`` ('direct' / 'surface' / 'bottom' / 'both').
        """
        d = self.at(range_idx=range_idx, depth_idx=depth_idx, src_idx=src_idx)
        if not d:
            return []
        delays = np.asarray(d['delays'])
        amps = np.asarray(d['amplitudes'])
        phs = np.asarray(d.get('phases', np.zeros_like(delays)))
        nt = np.asarray(d.get('n_top_bounces', np.zeros(len(delays), int)))
        nb = np.asarray(d.get('n_bot_bounces', np.zeros(len(delays), int)))
        sa = np.asarray(d.get('src_angles', np.zeros_like(delays)))
        ra = np.asarray(d.get('rcv_angles', np.zeros_like(delays)))
        out = []
        for i in range(len(delays)):
            ns, nbi = int(nt[i]), int(nb[i])
            if ns >= 1 and nbi >= 1:
                kind = 'both'
            elif nbi >= 1:
                kind = 'bottom'
            elif ns >= 1:
                kind = 'surface'
            else:
                kind = 'direct'
            out.append({
                'delay': float(delays[i]),
                'amplitude': float(amps[i]),
                'phase': float(phs[i]),
                'n_top_bounces': ns,
                'n_bot_bounces': nbi,
                'src_angle': float(sa[i]),
                'rcv_angle': float(ra[i]),
                'kind': kind,
            })
        return out


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
        ``n_bot_bounces``.
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
        # Mirror into ``metadata`` for dict-style access.
        self.metadata.setdefault('rays', self.rays)
        if self.receiver_depths is not None:
            self.metadata.setdefault('receiver_depths', self.receiver_depths)
        if self.receiver_ranges is not None:
            self.metadata.setdefault('receiver_ranges', self.receiver_ranges)

    # Convenience accessor.
    @property
    def ray_data(self) -> List[Any]:
        """The ray list. Same as ``self.rays``."""
        return self.rays

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
        kind_map = {
            'direct':  lambda r: int(r.get('n_top_bounces', 0) or 0) == 0
                              and int(r.get('n_bot_bounces', 0) or 0) == 0,
            'surface': lambda r: int(r.get('n_top_bounces', 0) or 0) > 0
                              and int(r.get('n_bot_bounces', 0) or 0) == 0,
            'bottom':  lambda r: int(r.get('n_bot_bounces', 0) or 0) > 0
                              and int(r.get('n_top_bounces', 0) or 0) == 0,
            'both':    lambda r: int(r.get('n_top_bounces', 0) or 0) > 0
                              and int(r.get('n_bot_bounces', 0) or 0) > 0,
        }
        if kind is not None and kind not in kind_map:
            raise ValueError(
                f"Arrivals.filter: kind={kind!r} not in {sorted(kind_map)}"
            )

        def _in_bounds(value, spec):
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

        def pred(ray):
            if kind is not None and not kind_map[kind](ray):
                return False
            n_top = int(ray.get('n_top_bounces', 0) or 0)
            n_bot = int(ray.get('n_bot_bounces', 0) or 0)
            return _in_bounds(n_top, top) and _in_bounds(n_bot, bot)

        return self.filter(pred)

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
        self._n_modes = int(n_modes if n_modes is not None else len(self.k))
        # Mirror modal data into ``metadata`` for dict-style access.
        self.metadata.setdefault('k', self.k)
        self.metadata.setdefault('phi', self.phi)
        self.metadata.setdefault('depths', self.depths)
        self.metadata.setdefault('n_modes', self._n_modes)

    @property
    def n_modes(self) -> int:
        return self._n_modes

    @property
    def data(self) -> np.ndarray:      # alias for plot helpers
        return self.phi

    def compute_phase_speeds(self) -> np.ndarray:
        omega = 2.0 * np.pi * (self.f0 or 0.0)
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
        # Mirror into ``metadata`` for dict-style access.
        self.metadata.setdefault('covariance', self.covariance)
        if self.receiver_positions is not None:
            self.metadata.setdefault('receiver_positions', self.receiver_positions)

    @property
    def n_frequencies(self) -> int:
        return int(self.covariance.shape[0])

    @property
    def n_receivers(self) -> int:
        return int(self.covariance.shape[1])


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
        self.metadata.setdefault('replicas', self.replicas)
        self.metadata.setdefault('replica_z', self.replica_z)
        self.metadata.setdefault('replica_x', self.replica_x)
        self.metadata.setdefault('replica_y', self.replica_y)
        if self.receiver_positions is not None:
            self.metadata.setdefault('receiver_positions', self.receiver_positions)

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
        # Mirror into ``metadata`` for dict-style access.
        self.metadata.setdefault('theta', self.theta)
        self.metadata.setdefault('R', self.R)
        self.metadata.setdefault('phi', self.phi)

    @property
    def n_angles(self) -> int:
        return len(self.theta)

    @property
    def is_broadband(self) -> bool:
        return self.R.ndim == 2

    def at_frequency(self, frequency: float) -> "ReflectionCoefficient":
        """Single-frequency slice of a broadband reflection coefficient."""
        if not self.is_broadband:
            raise ValueError(
                "ReflectionCoefficient.at_frequency: only valid for "
                "broadband ReflectionCoefficient"
            )
        k = int(np.argmin(np.abs(self.frequencies - frequency)))
        return ReflectionCoefficient(
            theta=self.theta,
            R=self.R[:, k],
            phi=self.phi[:, k],
            model=self.model, backend=self.backend,
            source_depths=self.source_depths,
            frequencies=float(self.frequencies[k]),
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
# IFFT helpers used by TransferFunction
# ─────────────────────────────────────────────────────────────────────────────


def _ifft_to_trace(
    tf: "TransferFunction",
    *,
    depth: Optional[float],
    range_m: Optional[float],
    source_spectrum: Optional[np.ndarray],
    window: str,
    nfft: Optional[int],
    t_start: Optional[float],
    sample_rate: Optional[float] = None,
) -> "TimeTrace":
    """IFFT one (depth, range) cell of a TransferFunction → TimeTrace.

    Places each model frequency at bin ``round(f/df)`` (with df capped at
    1 Hz for a ≥ 1-second window); when ``df`` is finer than the data
    spacing, demodulates by ``r/c0`` so the spectrum can be interpolated
    in baseband without ghost echoes, then re-modulates to land the
    arrival at the requested ``t_start``. Always sizes ``nfft`` so the
    largest frequency bin sits below Nyquist.
    """
    data = tf.data                                # (n_d, n_r, n_f)
    freqs = np.asarray(tf.frequencies, dtype=float)
    n_d, n_r, n_freq = data.shape

    if n_freq < 2:
        raise ValueError(
            f"_ifft_to_trace: need at least 2 frequencies for IFFT; got {n_freq}"
        )

    if tf.phase_reference == 'time_domain_native':
        raise ValueError(
            "_ifft_to_trace: phase_reference='time_domain_native' is not a "
            "frequency-domain transfer function; the producing model "
            "(SPARC) returned p(t) directly — use the TimeSeriesField result "
            "from RunMode.TIME_SERIES instead of synthesising via IFFT"
        )

    d_idx = (
        int(np.argmin(np.abs(tf.depths - depth))) if depth is not None
        else n_d // 2
    )
    r_idx = (
        int(np.argmin(np.abs(tf.ranges - range_m))) if range_m is not None
        else 0
    )
    actual_depth = float(tf.depths[d_idx])
    actual_range = float(tf.ranges[r_idx])

    spectrum = data[d_idx, r_idx, :].copy()
    spectrum = np.nan_to_num(spectrum, nan=0.0)

    df_data = float(freqs[1] - freqs[0])
    df = min(df_data, 1.0)               # cap at 1 Hz for ≥ 1-second window

    bin_indices = np.floor(freqs / df + 0.5).astype(int)
    max_bin = int(bin_indices[-1])

    if nfft is None:
        nfft_min = max(int(tf.metadata.get('Nsam', 0)) or 0, 4 * n_freq)
        # fs >= 2*fmax → nfft*df >= 2*max_bin*df, so nfft >= 2*max_bin+2
        # so the highest populated bin sits strictly below Nyquist.
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
        # Slowest mode (cmin) when a model knows it; otherwise reference
        # speed c0; otherwise the package default. Anchors the IFFT
        # window so the earliest arrival isn't clipped by the lead-in.
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

    return TimeTrace(
        data=result,
        time=time,
        depth=actual_depth,
        range_m=actual_range,
        model=tf.model,
        backend=tf.backend,
        source_depths=tf.source_depths,
        frequencies=tf.frequencies,
        metadata={
            'window': window,
            'source_model': tf.model,
        },
    )


def _synthesize_time_series(
    tf: "TransferFunction",
    *,
    source_waveform: np.ndarray,
    sample_rate: float,
    t_start: Optional[float],
    window: str,
    nfft: Optional[int],
) -> "TimeSeriesField":
    """Convolve every grid cell of a TransferFunction with a source waveform.

    Output shape: ``(n_d, n_r, n_t)``. ``nfft`` is sized so the IFFT
    sample rate equals ``sample_rate`` (rounded up to a power of two), so
    the returned trace is on the same sampling grid as the source pulse.
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
    source_spectrum = re_interp(tf.frequencies) + 1j * im_interp(tf.frequencies)

    n_d, n_r, _ = tf.data.shape
    depths = np.asarray(tf.depths)
    ranges = np.asarray(tf.ranges)

    if t_start is None:
        t0_trace = _ifft_to_trace(
            tf, depth=float(depths[0]), range_m=float(ranges[0]),
            source_spectrum=source_spectrum,
            window=window, nfft=nfft, t_start=None,
            sample_rate=sample_rate,
        )
        t_start = float(t0_trace.metadata['t_start'])

    out = None
    time_vec = None
    for di in range(n_d):
        for ri in range(n_r):
            tr = _ifft_to_trace(
                tf, depth=float(depths[di]), range_m=float(ranges[ri]),
                source_spectrum=source_spectrum,
                window=window, nfft=nfft, t_start=t_start,
                sample_rate=sample_rate,
            )
            if out is None:
                time_vec = tr.time
                out = np.zeros((n_d, n_r, len(tr.data)), dtype=tr.data.dtype)
            out[di, ri, :] = tr.data

    return TimeSeriesField(
        data=out,
        depths=depths,
        ranges=ranges,
        time=time_vec,
        model=tf.model,
        backend=tf.backend,
        source_depths=tf.source_depths,
        frequencies=tf.frequencies,
        metadata={
            'source_waveform_fs': sample_rate,
            'window': window,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────────────


__all__ = [
    "Result",
    "PhaseReference",
    # Spatial
    "PressureField", "TransferFunction",
    # Time-domain
    "TimeSeriesField", "TimeTrace",
    # Sparse / non-grid
    "Arrivals", "Rays", "Modes",
    "Covariance", "Replicas",
    "ReflectionCoefficient",
]
