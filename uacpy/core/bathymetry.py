"""Bathymetry shape carrier: seafloor depth as a function of range.

A 1-D profile (``depth`` vs ``range``), the seafloor analogue of
:class:`uacpy.core.ssp.SoundSpeedProfile`. Split out of
:mod:`uacpy.core.environment`; re-exported from there for stable import paths.
"""

import copy as _copy
import numpy as np
from dataclasses import dataclass

from uacpy.core.exceptions import ConfigurationError
from uacpy.core._carrier_validate import (
    _require_positive, _require_non_negative, _require_strictly_increasing,
)


@dataclass
class Bathymetry:
    """Seafloor depth (m, positive down) as a function of range (m).

    A 1-D grid library carrier mirroring :class:`SoundSpeedProfile`: select a
    depth with :meth:`at` (nearest), :meth:`isel` (positional) or :meth:`eval`
    (interpolated). Because bathymetry has a single axis (range), those
    selectors collapse it and return the **depth value(s)** directly (a scalar
    for a scalar range, an array for an array of ranges).

    Attributes
    ----------
    ranges : ndarray, shape (N,)
        Range axis in metres, monotonically increasing (``[0.0]`` for a flat
        bottom).
    depths : ndarray, shape (N,)
        Seafloor depth in metres at each range (positive down, > 0).
    """

    ranges: np.ndarray
    depths: np.ndarray

    def __post_init__(self):
        self.ranges = np.asarray(self.ranges, dtype=float).reshape(-1)
        self.depths = np.asarray(self.depths, dtype=float).reshape(-1)
        if self.ranges.size != self.depths.size:
            raise ConfigurationError(
                f"Bathymetry: ranges ({self.ranges.size}) and depths "
                f"({self.depths.size}) must have the same length.")
        if self.ranges.size == 0:
            raise ConfigurationError("Bathymetry: needs at least one point.")
        _require_non_negative(self.ranges, "Bathymetry ranges", hint="metres")
        _require_positive(self.depths, "Bathymetry depths", hint="metres, down")
        if self.ranges.size > 1:
            _require_strictly_increasing(self.ranges, "Bathymetry.ranges")

    # ── constructors ────────────────────────────────────────────────────────
    @classmethod
    def coerce(cls, value) -> 'Bathymetry':
        """Coerce ``Bathymetry`` / scalar depth / ``(N, 2)`` ``(range, depth)``
        pairs into a :class:`Bathymetry`."""
        if isinstance(value, Bathymetry):
            return value
        try:
            if np.ndim(value) == 0:
                return cls(ranges=np.array([0.0]),
                           depths=np.array([float(value)]))
            arr = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            raise ConfigurationError(
                f"Bathymetry: must be a positive scalar depth or shape (N, 2) "
                f"as [(range, depth), ...]; got non-numeric {value!r}.")
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ConfigurationError(
                f"Bathymetry: must be a positive scalar or shape (N, 2) as "
                f"[(range, depth), ...]; got shape {arr.shape} "
                f"(example: [(0, 100), (5000, 200)]).")
        return cls(ranges=arr[:, 0], depths=arr[:, 1])

    def to_pairs(self) -> np.ndarray:
        """``(N, 2)`` ``(range, depth)`` view."""
        return np.column_stack([self.ranges, self.depths])

    def copy(self) -> 'Bathymetry':
        """Deep copy (symmetric with the other carriers)."""
        return _copy.deepcopy(self)

    # ── derived ─────────────────────────────────────────────────────────────
    @property
    def n_ranges(self) -> int:
        return int(self.ranges.size)

    @property
    def depth(self) -> float:
        """Maximum seafloor depth (m) — the deepest point of the profile."""
        return float(np.max(self.depths))

    @property
    def range_max(self) -> float:
        return float(np.max(self.ranges))

    @property
    def is_range_dependent(self) -> bool:
        """True when the seafloor depth varies with range."""
        return self.ranges.size > 1 and bool(np.ptp(self.depths) > 0)

    # ── grid-library selectors ──────────────────────────────────────────────
    def at(self, *, range):
        """Nearest seafloor depth to ``range`` (m) — never fabricates. Returns
        a float for a scalar range, an array for an array of ranges. See
        :meth:`eval` (interpolate) and :meth:`isel` (positional)."""
        return self._query(range, 'nearest')

    def eval(self, *, range, method: str = 'linear'):
        """Interpolated seafloor depth at ``range`` (m). ``method`` is
        ``'linear'`` (default), ``'nearest'`` or ``'cubic'``; constant
        extrapolation past the ends. The interpolating counterpart of
        :meth:`at`."""
        return self._query(range, method)

    def isel(self, *, range):
        """Seafloor depth at integer index ``range`` — the positional
        counterpart of :meth:`at`."""
        idx = np.asarray(range)
        n = self.depths.size
        if idx.ndim == 0:
            i = int(idx)
            if not -n <= i < n:
                raise IndexError(
                    f"Bathymetry.isel: range index {i} out of range for "
                    f"{n} point(s)")
        out = self.depths[idx.astype(int)]
        return float(out) if idx.ndim == 0 else out

    def _query(self, range, method):
        from uacpy.core._grid import query_profile
        return query_profile(self.ranges, self.depths, range, method)
