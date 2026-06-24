"""Sound-speed-profile shape carrier and the rough sea-surface generator.
Split out of :mod:`uacpy.core.environment`; re-exported from there for stable
import paths.
"""

import copy as _copy
import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

from uacpy.core.exceptions import ConfigurationError
from uacpy.core._carrier_validate import (
    _require_positive, _require_strictly_increasing,
)


_VALID_SSP_SHAPES = (
    'measured', 'isovelocity', 'munk', 'analytic', 'n2linear',
)


@dataclass
class SoundSpeedProfile:
    """
    Unified sound-speed profile (1-D or 2-D).

    Stores the full grid as a 2-D array ``data[n_depth, n_range]``.
    Range-independent profiles use ``n_range = 1`` and ``ranges = None``;
    range-dependent profiles set ``ranges`` to a monotonically-increasing
    metres vector of length ``n_range``.

    Attributes
    ----------
    depths : ndarray, shape (N,)
        Depth axis in metres, monotonically increasing.
    data : ndarray, shape (N, M)
        Sound speed in m/s. ``M = 1`` for 1-D profiles.
    ranges : ndarray, shape (M,), optional
        Range axis in **metres**, monotonically increasing. ``None`` for 1-D.
    shape : str
        Declaration of what the data represents:
        ``'measured'`` (default), ``'isovelocity'``, ``'munk'``,
        ``'analytic'`` or ``'n2linear'``. Only ``'isovelocity'``
        actually overrides ``TopOpt(1)`` (forces ``'C'`` — any connection
        scheme over constant data is constant). The other values are
        informational metadata; the model's ``interp_ssp`` kwarg drives
        the AT character.
    """
    depths: np.ndarray
    data: np.ndarray
    ranges: Optional[np.ndarray] = None
    shape: str = 'measured'
    data_sources: tuple = ()

    def __post_init__(self):
        # Provenance of a fetched profile (tuple of DataSource/DataProvenance);
        # empty for a literal/hand-built one. Physics-agnostic metadata —
        # transforms that return a new profile (extend_to/collapse/eval slices)
        # carry it forward; the fresh-construction classmethods do not.
        self.data_sources = tuple(self.data_sources)
        self.depths = np.array(self.depths, dtype=float).reshape(-1)
        self.data = np.array(self.data, dtype=float)
        if self.data.ndim == 1:
            self.data = self.data.reshape(-1, 1)
        if self.data.ndim != 2:
            raise ConfigurationError(
                f"SoundSpeedProfile: data must be 1-D or 2-D; got {self.data.ndim}-D"
            )
        if self.data.shape[0] != self.depths.size:
            raise ConfigurationError(
                f"SoundSpeedProfile: data rows ({self.data.shape[0]}) must match "
                f"depths length ({self.depths.size})"
            )
        if self.depths.size == 0:
            raise ConfigurationError(
                "SoundSpeedProfile: needs at least one depth/sound-speed sample"
            )
        _require_positive(self.data, "SoundSpeedProfile sound speeds", hint="m/s")
        _require_strictly_increasing(self.depths, "SoundSpeedProfile.depths")
        if self.ranges is not None:
            self.ranges = np.array(self.ranges, dtype=float).reshape(-1)
            if self.ranges.size != self.data.shape[1]:
                raise ConfigurationError(
                    f"SoundSpeedProfile: ranges length ({self.ranges.size}) must "
                    f"match data columns ({self.data.shape[1]})"
                )
            _require_strictly_increasing(
                self.ranges, "SoundSpeedProfile.ranges",
            )
        elif self.data.shape[1] != 1:
            raise ConfigurationError(
                f"SoundSpeedProfile: ranges=None requires single-column data; "
                f"got shape {self.data.shape}"
            )
        self.shape = str(self.shape).lower()
        if self.shape not in _VALID_SSP_SHAPES:
            raise ConfigurationError(
                f"SoundSpeedProfile: shape={self.shape!r} not in "
                f"{_VALID_SSP_SHAPES}"
            )

    def __repr__(self) -> str:
        c_lo = float(np.min(self.data))
        c_hi = float(np.max(self.data))
        bits = [
            f"shape={self.shape!r}",
            f"n_z={self.depths.size}",
            f"z=[{float(self.depths[0]):g}, {float(self.depths[-1]):g}] m",
        ]
        if self.is_range_dependent:
            r_lo = float(self.ranges[0]) / 1000
            r_hi = float(self.ranges[-1]) / 1000
            bits.append(f"n_r={self.data.shape[1]}")
            bits.append(f"range=[{r_lo:g}, {r_hi:g}] km")
        bits.append(f"c=[{c_lo:g}, {c_hi:g}] m/s")
        return f"SoundSpeedProfile({', '.join(bits)})"

    @property
    def is_range_dependent(self) -> bool:
        return self.ranges is not None and self.data.shape[1] > 1

    @property
    def n_depths(self) -> int:
        return int(self.depths.size)

    @property
    def n_ranges(self) -> int:
        return int(self.data.shape[1])

    def to_pairs(self) -> np.ndarray:
        """Return ``(N, 2)`` ``(depth, c)`` view of the 1-D form.

        For range-dependent profiles, returns the range-0 column. Use
        ``at(range=)`` / ``eval(range=)`` for an explicit slice or
        ``collapse`` for a chosen reduction.
        """
        return np.column_stack([self.depths, self.data[:, 0]])

    def at(
        self, *, depth: Optional[float] = None, range: Optional[float] = None,
    ) -> 'SoundSpeedProfile':
        """Nearest-sample slice at the requested depth and/or range.

        Returns the closest stored grid sample on each axis — **never
        fabricates** a value (the grid-library invariant shared with
        ``Field.at`` et al.). For interpolated evaluation use :meth:`eval`;
        for an integer-index slice use :meth:`isel`.
        """
        return self._slice(depth=depth, range=range, interp='nearest')

    def eval(
        self, *, depth: Optional[float] = None, range: Optional[float] = None,
        method: str = 'linear',
    ) -> 'SoundSpeedProfile':
        """Interpolated slice at the requested depth and/or range.

        ``method`` is the interpolation scheme — ``'linear'`` (default),
        ``'nearest'``, or ``'cubic'`` — with constant extrapolation outside
        ``[ranges[0], ranges[-1]]``. The interpolating counterpart of
        :meth:`at` (which is always nearest).
        """
        return self._slice(depth=depth, range=range, interp=method)

    def isel(
        self, *, depth: Optional[int] = None, range: Optional[int] = None,
    ) -> 'SoundSpeedProfile':
        """Integer-index slice on the depth and/or range axis — the positional
        counterpart of :meth:`at`."""
        sliced = self
        if range is not None:
            ridx = int(range)
            if not -self.data.shape[1] <= ridx < self.data.shape[1]:
                raise IndexError(
                    f"SoundSpeedProfile.isel: range index {ridx} out of range "
                    f"for {self.data.shape[1]} column(s)")
            sliced = SoundSpeedProfile(
                depths=self.depths.copy(),
                data=self.data[:, [ridx]].copy(), ranges=None, shape=self.shape,
                data_sources=self.data_sources)
        if depth is not None:
            didx = int(depth)
            if not -sliced.depths.size <= didx < sliced.depths.size:
                raise IndexError(
                    f"SoundSpeedProfile.isel: depth index {didx} out of range "
                    f"for {sliced.depths.size} depth(s)")
            sliced = SoundSpeedProfile(
                depths=np.array([float(sliced.depths[didx])]),
                data=sliced.data[[didx], :].copy(), ranges=None,
                shape=sliced.shape, data_sources=self.data_sources)
        return sliced

    def _slice(
        self, *, depth: Optional[float], range: Optional[float], interp: str,
    ) -> 'SoundSpeedProfile':
        from uacpy.core._grid import collapse_axis
        data = self.data
        if range is not None:
            if not self.is_range_dependent:
                col = data[:, 0]
            else:
                col, _ = collapse_axis(data, self.ranges, range, interp, axis=1)
            data = col.reshape(-1, 1)
        if depth is None:
            if range is None:
                return self
            return SoundSpeedProfile(
                depths=self.depths.copy(), data=data.copy(),
                ranges=None, shape=self.shape, data_sources=self.data_sources)
        c, dv = collapse_axis(data[:, 0], self.depths, depth, interp, axis=0)
        return SoundSpeedProfile(
            depths=np.array([float(dv)]), data=np.array([[float(c)]]),
            ranges=None, shape=self.shape, data_sources=self.data_sources)

    @property
    def value(self) -> float:
        """The single sound speed this profile represents, when unambiguous.

        Valid for an isovelocity profile (every sample equal) or one
        collapsed to a single ``(depth, range)`` cell via
        ``at(depth=, range=)``. Raises if the profile actually varies."""
        if self.data.size > 1 and np.ptp(self.data) > 0:
            raise ConfigurationError(
                f"SoundSpeedProfile.value: profile varies (shape "
                f"{self.data.shape}); slice with at(depth=, range=) first"
            )
        return float(self.data.flat[0])

    def copy(self) -> 'SoundSpeedProfile':
        """Deep copy (symmetric with the other carriers)."""
        return _copy.deepcopy(self)

    def collapse(self, method: str = 'r0') -> 'SoundSpeedProfile':
        """Collapse a 2-D profile to 1-D using ``method``.

        Methods
        -------
        ``'r0'``     : keep the range-0 column.
        ``'mean'``   : depth-wise mean across all ranges.
        ``'median'`` : depth-wise median across all ranges.
        ``'rmax'``   : keep the last (deepest range) column.
        """
        if not self.is_range_dependent:
            return self
        if method == 'r0':
            col = self.data[:, 0]
        elif method == 'rmax':
            col = self.data[:, -1]
        elif method == 'mean':
            col = self.data.mean(axis=1)
        elif method == 'median':
            col = np.median(self.data, axis=1)
        else:
            raise ConfigurationError(
                f"SoundSpeedProfile.collapse: unknown method={method!r}; "
                "valid: 'r0', 'rmax', 'mean', 'median'"
            )
        return SoundSpeedProfile(
            depths=self.depths.copy(),
            data=col.reshape(-1, 1),
            ranges=None,
            shape=self.shape,
            data_sources=self.data_sources,
        )

    def extend_to(self, depth_max: float) -> 'SoundSpeedProfile':
        """Return a copy with the deepest sample sitting exactly at
        ``depth_max``.

        Three cases:

        * ``depth_max == depths[-1]`` — return ``self`` unchanged.
        * ``depth_max > depths[-1]`` — append a new sample at
          ``depth_max`` carrying the deepest existing sound speed
          (constant extrapolation, the AT writer convention).
        * ``depth_max < depths[-1]`` — truncate samples below
          ``depth_max`` and interpolate a final sample exactly at
          ``depth_max`` so writers that require ``ssp[-1] == env.depth``
          (Bellhop / Kraken) round-trip without manual alignment.
        """
        # Tolerant float compare — caller may pass in e.g. ``env.depth``
        # that's been round-tripped through I/O and differs by a few
        # ulps from ``self.depths[-1]``.
        last = float(self.depths[-1])
        if np.isclose(depth_max, last, rtol=1e-9, atol=1e-9):
            return self
        if depth_max > last:
            new_depths = np.append(self.depths, depth_max)
            new_data = np.vstack([self.data, self.data[-1:, :]])
        else:
            keep = self.depths < depth_max
            kept_depths = self.depths[keep]
            kept_data = self.data[keep]
            interp_row = np.array([
                np.interp(depth_max, self.depths, self.data[:, j])
                for j in range(self.data.shape[1])
            ])
            new_depths = np.append(kept_depths, depth_max)
            new_data = np.vstack([kept_data, interp_row[None, :]])
        return SoundSpeedProfile(
            depths=new_depths,
            data=new_data,
            ranges=(self.ranges.copy() if self.ranges is not None else None),
            shape=self.shape,
            data_sources=self.data_sources,
        )

    @classmethod
    def coerce(
        cls, value, *, depth_max: float,
    ) -> 'SoundSpeedProfile':
        """Coerce the user-facing ``ssp=`` shorthand into a profile.

        Mirrors :meth:`Bathymetry.coerce` / :meth:`Altimetry.coerce` so
        :class:`~uacpy.core.environment.Environment` delegates instead of
        hand-rolling the dispatch:

        * ``None`` — isovelocity 1500 m/s spanning ``0..depth_max``.
        * scalar (m/s) — isovelocity at that speed spanning ``0..depth_max``.
        * ``(depth, c)`` pairs — linear profile via :meth:`from_pairs`.
        * a :class:`SoundSpeedProfile` — returned as-is (by reference).

        ``depth_max`` (m) sets the column extent for the isovelocity cases.
        """
        if value is None:
            return cls.from_isovelocity(depth_max, 1500.0)
        if isinstance(value, SoundSpeedProfile):
            return value
        if isinstance(value, (int, float, np.integer, np.floating)):
            return cls.from_isovelocity(depth_max, float(value))
        if isinstance(value, (list, tuple, np.ndarray)):
            return cls.from_pairs(value)
        raise ConfigurationError(
            f"Environment: ssp must be a scalar (m/s), a list of (depth, "
            f"sound_speed) pairs, or a SoundSpeedProfile; got "
            f"{type(value).__name__}"
        )

    @classmethod
    def from_isovelocity(
        cls, depth_max: float, sound_speed: float = 1500.0
    ) -> 'SoundSpeedProfile':
        """Constant-``sound_speed`` (m/s) profile spanning 0 to ``depth_max`` (m)."""
        return cls(
            depths=np.array([0.0, float(depth_max)]),
            data=np.full((2, 1), float(sound_speed)),
            ranges=None,
            shape='isovelocity',
        )

    @classmethod
    def from_pairs(
        cls,
        pairs: Union[List[Tuple[float, float]], np.ndarray],
        shape: str = 'measured',
    ) -> 'SoundSpeedProfile':
        """Build a 1-D profile from ``[(depth, c), …]`` pairs.

        ``shape`` is informational metadata (``'measured'`` default);
        see :class:`SoundSpeedProfile`. The model's ``interp_ssp`` kwarg
        drives the sample-connection scheme.
        """
        arr = np.asarray(pairs, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ConfigurationError(
                f"SoundSpeedProfile.from_pairs: pairs must have shape (N, 2) "
                f"as (depth, sound_speed); got shape {arr.shape}"
            )
        return cls(
            depths=arr[:, 0],
            data=arr[:, 1].reshape(-1, 1),
            ranges=None,
            shape=shape,
        )

    @classmethod
    def from_2d(
        cls,
        depths: np.ndarray,
        ranges: np.ndarray,
        matrix: np.ndarray,
        shape: str = 'measured',
    ) -> 'SoundSpeedProfile':
        """Build a 2-D profile from a depth axis, range axis (metres),
        and ``c(depth, range)`` matrix of shape ``(n_depth, n_range)``.

        For Bellhop, pair with ``Bellhop(interp_ssp='quad')`` to enable
        the external ``.ssp`` (quad) file format.
        """
        return cls(
            depths=np.asarray(depths, dtype=float),
            data=np.asarray(matrix, dtype=float),
            ranges=np.asarray(ranges, dtype=float),
            shape=shape,
        )

    @classmethod
    def from_munk(
        cls, depth_max: float, n_points: int = 101
    ) -> 'SoundSpeedProfile':
        """Munk canonical profile with axis at 1300 m, c_min = 1500 m/s."""
        depths = np.linspace(0.0, float(depth_max), int(n_points))
        z_axis = 1300.0
        epsilon = 0.00737
        c_min = 1500.0
        eta = 2.0 * (depths - z_axis) / z_axis
        c = c_min * (1.0 + epsilon * (eta - 1.0 + np.exp(-eta)))
        return cls(
            depths=depths,
            data=c.reshape(-1, 1),
            ranges=None,
            shape='munk',
        )

    @classmethod
    def from_mackenzie(
        cls,
        depths: np.ndarray,
        temperature_c: np.ndarray,
        salinity_psu: np.ndarray,
    ) -> 'SoundSpeedProfile':
        """Build a profile from in-situ ``T(z)`` and ``S(z)`` via Mackenzie's
        nine-term seawater sound-speed equation.

        ``depths``, ``temperature_c``, ``salinity_psu`` must be 1-D arrays
        of equal length sampled at the same depth grid. Use
        ``np.full_like(depths, T_const)`` if the column is isothermal/
        isohaline. Valid range: ``T ∈ [−2, 30] °C``,
        ``S ∈ [25, 40] PSU``, ``z ∈ [0, 8000] m`` (Mackenzie 1981).
        """
        from uacpy.core.acoustics import soundspeed
        z = np.asarray(depths, dtype=float).ravel()
        T = np.asarray(temperature_c, dtype=float).ravel()
        S = np.asarray(salinity_psu, dtype=float).ravel()
        if not (T.shape == S.shape == z.shape):
            raise ConfigurationError(
                "from_mackenzie: depths, temperature_c, salinity_psu must "
                f"share shape; got {z.shape}, {T.shape}, {S.shape}"
            )
        c = soundspeed(temperature=T, salinity=S, depth=z)
        return cls(
            depths=z, data=np.asarray(c).reshape(-1, 1),
            ranges=None,
        )


def generate_sea_surface(
    max_range: float,
    wind_speed_ms: float = 10.0,
    n_points: int = 500,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a random sea surface realization from the Pierson-Moskowitz spectrum.

    Parameters
    ----------
    max_range : float
        Maximum range in meters.
    wind_speed_ms : float
        Wind speed at 19.5 m height in m/s (Pierson-Moskowitz
        convention). The fully developed significant wave height is
        Hs = 4*sqrt(alpha/beta)*U^2/(2g) = 0.021*U^2:
        - 5 m/s: Hs ~ 0.5 m
        - 10 m/s: Hs ~ 2.1 m
        - 15 m/s: Hs ~ 4.8 m
        - 20 m/s: Hs ~ 8.5 m
    n_points : int
        Number of range points in the output altimetry array.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    altimetry : ndarray, shape (n_points, 2)
        Column 0: range (m), Column 1: surface height (m, positive up).
        Suitable for passing directly to ``Environment(altimetry=...)``.
    """
    if not np.isfinite(max_range) or max_range <= 0:
        raise ConfigurationError(
            f"generate_sea_surface: max_range must be a positive distance (m); "
            f"got {max_range}."
        )
    if not np.isfinite(wind_speed_ms) or wind_speed_ms <= 0:
        raise ConfigurationError(
            f"generate_sea_surface: wind_speed_ms must be a positive m/s value; "
            f"got {wind_speed_ms}."
        )
    if n_points < 2:
        raise ConfigurationError(
            f"generate_sea_surface: n_points must be >= 2; got {n_points}."
        )
    g = 9.81
    rng = np.random.default_rng(seed)

    ranges = np.linspace(0, max_range, n_points)
    dx = ranges[1] - ranges[0]

    # Spatial frequency grid (cycles/m)
    n_fft = n_points
    dk = 1.0 / (n_fft * dx)  # spatial freq resolution
    k = np.arange(1, n_fft // 2 + 1) * dk  # positive frequencies
    omega = np.sqrt(g * 2 * np.pi * k)  # deep-water dispersion: omega^2 = g*k_wave

    # Pierson-Moskowitz spectrum S(omega)
    # S(omega) = (alpha * g^2 / omega^5) * exp(-beta * (omega_p / omega)^4)
    alpha_pm = 8.1e-3
    beta_pm = 0.74
    omega_p = g / wind_speed_ms  # peak angular frequency
    S_omega = (alpha_pm * g**2 / omega**5) * np.exp(-beta_pm * (omega_p / omega)**4)

    # Convert to spatial spectrum S(k) via S(k) = S(omega) * domega/dk
    # with k in cycles/m: omega = sqrt(2*pi*g*k) so domega/dk = pi*g/omega
    domega_dk = np.pi * g / omega
    S_k = S_omega * domega_dk

    # Generate random amplitudes from spectrum
    amplitude = np.sqrt(2 * S_k * dk)
    phase = rng.uniform(0, 2 * np.pi, len(k))

    surface = (
        amplitude[None, :]
        * np.cos(2 * np.pi * np.outer(ranges, k) + phase[None, :])
    ).sum(axis=1)

    return np.column_stack([ranges, surface])
