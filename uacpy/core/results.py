"""Typed result hierarchy for uacpy model outputs.

One class per output category — ``isinstance`` is the contract.

Convention
----------
**Spatial axes come first, the variable axis is trailing.** That means:

* ``TLField.data``       — ``(n_d, n_r)`` narrowband, ``(n_d, n_r, n_f)`` broadband, dB
* ``PressureField.data`` — ``(n_d, n_r)`` complex
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
    │   ├── TLField
    │   ├── PressureField
    │   ├── TransferFunction
    │   └── TimeSeriesField
    ├── TimeTrace                   (single point, no spatial grid)
    ├── Arrivals                    (per-receiver list of arrivals)
    ├── Rays                        (per-source list of ray paths)
    ├── Modes                       (Kraken normal modes)
    ├── OASNCovariance              (OASN replica/covariance modes)
    └── ReflectionCoefficient       (R(theta) at a boundary)

Identification fields (``model``, ``backend``, ``source_depths``,
``frequency``/``frequencies``) are also mirrored into ``metadata`` so
callers can use either typed attributes or dict-style access.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Union

from uacpy.core.constants import DEFAULT_SOUND_SPEED, PRESSURE_FLOOR


# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────


class Result:
    """Common base for every model output.

    Carries identification (``model``, ``backend``), the source context
    (``source_depths``, ``frequency`` or ``frequencies``), and a free-form
    ``metadata`` dict for model-specific extras. Subclasses add the shape-
    specific payload and methods.

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
    frequency : float, optional
        Single (centre) frequency in Hz, when applicable.
    frequencies : array-like, optional
        Frequency vector in Hz, when broadband.
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
        frequency: Optional[float] = None,
        frequencies: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.backend = backend if backend is not None else (model.lower() if model else "")
        self.source_depths = (
            np.atleast_1d(np.asarray(source_depths, dtype=float))
            if source_depths is not None else np.array([], dtype=float)
        )
        if frequency is not None:
            self.frequency: Optional[float] = float(frequency)
        else:
            self.frequency = None
        if frequencies is not None:
            self.frequencies: Optional[np.ndarray] = np.asarray(frequencies, dtype=float)
        else:
            self.frequencies = None
        self.metadata: Dict[str, Any] = dict(metadata) if metadata else {}
        # Mirror identification fields into ``metadata`` so users who prefer
        # dict-style access (``r.metadata['model']`` etc.) get the same
        # values that the typed attributes carry.
        if self.model:
            self.metadata.setdefault('model', self.model)
        if self.backend:
            self.metadata.setdefault('backend', self.backend)
        if self.source_depths is not None and len(self.source_depths):
            self.metadata.setdefault('source_depths', self.source_depths)
        if self.frequency is not None:
            self.metadata.setdefault('frequency', self.frequency)
        if self.frequencies is not None:
            self.metadata.setdefault('frequencies', self.frequencies)

    # Convenience ------------------------------------------------------------

    @property
    def n_frequencies(self) -> int:
        return 0 if self.frequencies is None else int(len(self.frequencies))

    def __repr__(self) -> str:
        cls = type(self).__name__
        bits = [f"model={self.model!r}"] if self.model else []
        if self.frequency is not None:
            bits.append(f"f={self.frequency:.3g} Hz")
        elif self.frequencies is not None and len(self.frequencies):
            bits.append(f"n_f={len(self.frequencies)}")
        return f"{cls}({', '.join(bits)})"

    def copy(self):
        """Deep copy of the result. Preserves type."""
        import copy as _copy
        return _copy.deepcopy(self)

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
    _data_role: str = "data"

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
                f"{type(self).__name__}.data must be at least 2-D "
                f"(got shape {data.shape})"
            )
        if data.shape[:2] != (len(depths), len(ranges)):
            raise ValueError(
                f"{type(self).__name__}.data first two axes "
                f"{data.shape[:2]} must equal (n_depths, n_ranges) = "
                f"({len(depths)}, {len(ranges)})"
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
                raise ValueError("time= requested but result has no time axis")
            k = int(np.argmin(np.abs(np.asarray(t_axis) - time)))
            return self.data[d_idx, r_idx, k]
        raise ValueError(
            f"{type(self).__name__} has 3 axes; specify frequency= or time="
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

    # Generic plotting hook (subclasses delegate)
    def plot(self, env=None, **kwargs):
        from uacpy.visualization import plots
        return plots.plot_result(self, env=env, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Spatial gridded results
# ─────────────────────────────────────────────────────────────────────────────


class TLField(_GridResult):
    """Transmission loss on a ``(depth, range)`` grid (dB, real).

    ``data`` shape:
      * narrowband: ``(n_depths, n_ranges)``
      * broadband : ``(n_depths, n_ranges, n_frequencies)``
    """
    field_type = "tl"
    _data_role = "TL (dB)"

    @property
    def is_broadband(self) -> bool:
        return self.data.ndim == 3

    def at_frequency(self, freq: float) -> "TLField":
        """For broadband ``TLField``, return the narrowband TLField at the
        nearest frequency."""
        if not self.is_broadband:
            raise ValueError("at_frequency() only valid for broadband TLField")
        if self.frequencies is None:
            raise ValueError("broadband TLField missing self.frequencies")
        k = int(np.argmin(np.abs(self.frequencies - freq)))
        return TLField(
            data=self.data[..., k],
            depths=self.depths,
            ranges=self.ranges,
            model=self.model, backend=self.backend,
            source_depths=self.source_depths,
            frequency=float(self.frequencies[k]),
            metadata=self.metadata,
        )


class PressureField(_GridResult):
    """Complex pressure on a ``(depth, range)`` grid.

    ``data`` shape: ``(n_depths, n_ranges)`` complex.

    Use :meth:`to_tl` to obtain the dB transmission-loss view.
    """
    field_type = "pressure"
    _data_role = "complex pressure"

    def to_tl(self) -> TLField:
        """Convert to ``TLField`` via ``-20·log10(|p|)``."""
        p_abs = np.maximum(np.abs(self.data), PRESSURE_FLOOR)
        tl = -20.0 * np.log10(p_abs)
        return TLField(
            data=tl,
            depths=self.depths,
            ranges=self.ranges,
            model=self.model, backend=self.backend,
            source_depths=self.source_depths,
            frequency=self.frequency,
            metadata=self.metadata,
        )

    @property
    def magnitude(self) -> np.ndarray:
        return np.abs(self.data)

    @property
    def phase(self) -> np.ndarray:
        return np.angle(self.data)


class TransferFunction(_GridResult):
    """Complex broadband transfer function ``H(d, r, f)``.

    ``data`` shape: ``(n_depths, n_ranges, n_frequencies)`` complex.

    The ``phase_reference`` field is required and tells consumers (notably
    :meth:`to_time_trace` / :meth:`synthesize_time_series`) how to interpret
    the stored phase. See ``DOCUMENTATION.md §5.13``.
    """
    field_type = "transfer_function"
    _data_role = "complex transfer function"

    def __init__(
        self,
        *,
        data: np.ndarray,
        depths: np.ndarray,
        ranges: np.ndarray,
        frequencies: np.ndarray,
        phase_reference: str,
        **kwargs,
    ):
        if frequencies is None or len(frequencies) == 0:
            raise ValueError("TransferFunction requires a non-empty frequencies vector")
        super().__init__(
            data=data, depths=depths, ranges=ranges,
            frequencies=frequencies, **kwargs,
        )
        if data.ndim != 3:
            raise ValueError(
                f"TransferFunction.data must be 3-D (n_d, n_r, n_f), got {data.shape}"
            )
        if data.shape[2] != len(self.frequencies):
            raise ValueError(
                f"TransferFunction.data axis 2 ({data.shape[2]}) does not "
                f"match len(frequencies) ({len(self.frequencies)})"
            )
        if not phase_reference:
            raise ValueError(
                "TransferFunction requires a non-empty phase_reference "
                "('travelling_wave', 'porter_negated', 'psif_envelope', ...)"
            )
        self.phase_reference = phase_reference
        # Mirror into ``metadata`` for dict-style access.
        self.metadata.setdefault('phase_reference', phase_reference)

    # Time-domain conversion ------------------------------------------------
    # Implementation lives in core.field for now (heavy IFFT machinery); we
    # delegate to it.
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
        from uacpy.core.field import _ifft_to_trace
        return _ifft_to_trace(
            self, depth=depth, range_m=range_m,
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
    ) -> "TimeSeriesField":
        """Convolve every grid trace with ``source_waveform`` to obtain a
        :class:`TimeSeriesField` shaped ``(n_d, n_r, n_t)``."""
        from uacpy.core.field import _synthesize_time_series
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
    _data_role = "p(t)"

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
                f"TimeSeriesField.data must be 3-D (n_d, n_r, n_t), got {data.shape}"
            )
        time = np.asarray(time, dtype=float)
        if data.shape[2] != len(time):
            raise ValueError(
                f"TimeSeriesField.data axis 2 ({data.shape[2]}) does not "
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
            frequency=self.frequency,
            metadata=dict(self.metadata),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Time-domain — single point
# ─────────────────────────────────────────────────────────────────────────────


class TimeTrace(Result):
    """Real pressure ``p(t)`` at a single ``(depth, range)`` point.

    ``data`` shape: ``(n_t,)`` real.
    """
    field_type = "time_series"   # legacy field_type for visualization dispatch

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
                f"TimeTrace.data must be 1-D, got shape {data.shape}"
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
        self.range = float(range_m)
        # Mirror time-axis info into ``metadata`` for dict-style consumers.
        self.metadata.setdefault('time', time)
        self.metadata.setdefault('nt', int(len(time)))
        self.metadata.setdefault('depth', self.depth)
        self.metadata.setdefault('range', self.range)
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

    def plot(self, **kwargs):
        from uacpy.visualization import plots
        return plots.plot_time_trace(self, **kwargs)

    # 1-element ndarrays for plotting helpers that expect a vector.
    @property
    def depths(self) -> np.ndarray:
        return np.array([self.depth])

    @property
    def ranges(self) -> np.ndarray:
        return np.array([self.range])


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
        ``src_angle``, ``rcv_angle``, ``n_top_bounces``, ``n_bot_bounces``).
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

    # Legacy accessors expected by visualization.plots.plot_arrivals -------
    @property
    def depths(self) -> np.ndarray:
        return self.receiver_depths

    @property
    def ranges(self) -> np.ndarray:
        return self.receiver_ranges

    def plot(self, **kwargs):
        from uacpy.visualization import plots
        return plots.plot_arrivals(self, **kwargs)

    # Convenience accessor.
    @property
    def arrivals_data(self) -> Any:
        """The per-receiver arrivals payload. Same as ``self.by_receiver``."""
        return self.by_receiver

    def extract_arrivals(self) -> Any:
        return self.by_receiver

    def at(self, range_idx: int = 0, depth_idx: int = 0,
           src_idx: int = 0) -> Dict[str, np.ndarray]:
        """Return the flat arrivals dict for one (source, depth, range) cell.

        Bellhop's ``.arr`` payload is nested as
        ``arrivals_data[src][depth][range] -> dict``; with a single-point
        receiver the leading dimensions all have length 1. This walks the
        nesting and returns the dict so users can access ``delays``,
        ``amplitudes``, ``phases``, ``n_top_bounces``, ``n_bot_bounces``,
        ``src_angles``, ``rcv_angles`` directly.
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
        ``n_top_bounces``, ``n_bot_bounces``, ``src_angle``, ``rcv_angle``,
        ``kind`` ('direct' / 'surface' / 'bottom' / 'both').
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

    Attributes
    ----------
    rays : list
        List of ray dicts with at least ``r``, ``z`` arrays. Eigenrays are
        the same shape; ``is_eigen`` indicates which.
    """
    field_type = "rays"

    def __init__(self, *, rays: List[Any], is_eigen: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.rays = list(rays)
        self.is_eigen = bool(is_eigen)
        # Mirror into ``metadata`` for dict-style access.
        self.metadata.setdefault('rays', self.rays)

    def plot(self, env=None, **kwargs):
        from uacpy.visualization import plots
        return plots.plot_rays(self, env=env, **kwargs)

    # Convenience accessor.
    @property
    def ray_data(self) -> List[Any]:
        """The ray list. Same as ``self.rays``."""
        return self.rays

    def extract_rays(self) -> List[Any]:
        return self.rays

    def at_receiver(
        self,
        range_m: float,
        depth_m: float,
        tolerance_m: Optional[float] = None,
        max_rays: Optional[int] = None,
        truncate: bool = True,
    ) -> 'Rays':
        """Return rays that arrive at the given (range, depth) within tolerance.

        For every ray, the closest-approach distance to the target point is
        computed in 2-D (range/depth, with depth in metres). Rays farther
        than ``tolerance_m`` are dropped; the remainder are sorted by miss
        distance ascending and trimmed to ``max_rays`` if given. When
        ``truncate=True`` (default), each kept ray is also truncated at its
        closest-approach point so it visibly terminates on the receiver
        instead of continuing past it.

        ``tolerance_m`` defaults to one acoustic wavelength at the source
        frequency (``c0/f`` with ``c0=1500 m/s``); pass ``None`` to keep
        every ray (only sort + trim).
        """
        if not self.rays:
            return Rays(rays=[], is_eigen=self.is_eigen,
                        model=self.model, backend=self.backend,
                        metadata=dict(self.metadata))

        if tolerance_m is None:
            f = float(np.atleast_1d(self.frequency or 0.0)[0])
            if f > 0:
                tolerance_m = 1500.0 / f
            else:
                tolerance_m = float('inf')

        rr_km = range_m / 1000.0
        rd_m = depth_m
        scored = []
        for ray in self.rays:
            r_km = np.asarray(ray.get('r', [])) / 1000.0
            z = np.asarray(ray.get('z', []))
            if len(r_km) == 0:
                continue
            d2 = (r_km - rr_km) ** 2 + ((z - rd_m) / 1000.0) ** 2
            k = int(np.argmin(d2))
            miss_m = float(np.sqrt(d2[k]) * 1000.0)
            if miss_m > tolerance_m:
                continue
            if truncate and k + 1 < len(r_km):
                ray = dict(ray)
                ray['r'] = np.asarray(ray['r'])[: k + 1]
                ray['z'] = np.asarray(ray['z'])[: k + 1]
            ray = dict(ray)
            ray['miss_distance_m'] = miss_m
            scored.append((miss_m, ray))

        scored.sort(key=lambda t: t[0])
        if max_rays is not None:
            scored = scored[:max_rays]

        meta = dict(self.metadata)
        meta['receiver_range_m'] = range_m
        meta['receiver_depth_m'] = depth_m
        meta['tolerance_m'] = tolerance_m
        return Rays(rays=[r for _, r in scored], is_eigen=True,
                    model=self.model, backend=self.backend,
                    metadata=meta)


class Modes(Result):
    """Kraken normal modes.

    Attributes
    ----------
    k : ndarray, shape ``(n_modes,)`` complex
        Modal wavenumbers.
    phi : ndarray, shape ``(n_z, n_modes)``
        Mode shapes sampled at ``z``.
    z : ndarray, shape ``(n_z,)``
        Sampling depths.
    """
    field_type = "modes"

    def __init__(
        self,
        *,
        k: np.ndarray,
        phi: np.ndarray,
        z: np.ndarray,
        n_modes: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.k = np.asarray(k)
        self.phi = np.asarray(phi)
        self.z = np.atleast_1d(np.asarray(z, dtype=float))
        self.n_modes = int(n_modes if n_modes is not None else len(self.k))
        # Mirror modal data into ``metadata`` for dict-style access.
        self.metadata.setdefault('k', self.k)
        self.metadata.setdefault('phi', self.phi)
        self.metadata.setdefault('z', self.z)
        self.metadata.setdefault('n_modes', self.n_modes)

    @property
    def depths(self) -> np.ndarray:    # scalar (depth, range) → 1-D arrays for the plotting helpers
        return self.z

    @property
    def data(self) -> np.ndarray:      # alias for plot helpers
        return self.phi

    def phase_speeds(self) -> np.ndarray:
        omega = 2.0 * np.pi * (self.frequency or 0.0)
        return omega / np.real(self.k)

    def plot(self, **kwargs):
        from uacpy.visualization import plots
        return plots.plot_modes(self, **kwargs)


class OASNCovariance(Result):
    """OASN replica vectors / spatial covariance modes.

    Different physics from Kraken's :class:`Modes` (these are eigenvectors
    of a covariance matrix, not eigenfunctions of the depth-separated wave
    equation). Kept distinct so downstream code can't accidentally treat
    them as Kraken modes.

    Attributes
    ----------
    phi : ndarray
        Replica field samples (depth × mode-index).
    z : ndarray
        Sampling depths.
    covariance : ndarray, optional
        Spatial covariance matrix when present.
    """
    field_type = "modes"   # legacy dispatch — same plot helper

    def __init__(
        self,
        *,
        phi: np.ndarray,
        z: np.ndarray,
        covariance: Optional[np.ndarray] = None,
        n_modes: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.phi = np.asarray(phi)
        self.z = np.atleast_1d(np.asarray(z, dtype=float))
        self.covariance = np.asarray(covariance) if covariance is not None else None
        self.n_modes = int(n_modes if n_modes is not None else self.phi.shape[-1] if self.phi.ndim else 0)

    @property
    def depths(self) -> np.ndarray:
        return self.z

    @property
    def data(self) -> np.ndarray:
        return self.phi


class ReflectionCoefficient(Result):
    """Angle-dependent reflection coefficient ``R(theta)``.

    Unifies what Bounce and OASR produce. Use this for both bottom (BRC)
    and top (TRC) reflection coefficients.

    Attributes
    ----------
    theta : ndarray, shape ``(n_angles,)``  — grazing angles in degrees
    R     : ndarray, shape ``(n_angles,)``  — magnitude in [0, 1]
    phi   : ndarray, shape ``(n_angles,)``  — phase in radians
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
        self.R = np.atleast_1d(np.asarray(R, dtype=float))
        self.phi = np.atleast_1d(np.asarray(phi, dtype=float))
        if not (len(self.theta) == len(self.R) == len(self.phi)):
            raise ValueError(
                f"ReflectionCoefficient: theta/R/phi length mismatch "
                f"({len(self.theta)}, {len(self.R)}, {len(self.phi)})"
            )
        # Mirror into ``metadata`` for dict-style access.
        self.metadata.setdefault('theta', self.theta)
        self.metadata.setdefault('R', self.R)
        self.metadata.setdefault('phi', self.phi)

    @property
    def n_angles(self) -> int:
        return len(self.theta)

    
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
# Discovery
# ─────────────────────────────────────────────────────────────────────────────


__all__ = [
    "Result",
    # Spatial
    "TLField", "PressureField", "TransferFunction",
    # Time-domain
    "TimeSeriesField", "TimeTrace",
    # Sparse / non-grid
    "Arrivals", "Rays", "Modes", "OASNCovariance",
    "ReflectionCoefficient",
]
