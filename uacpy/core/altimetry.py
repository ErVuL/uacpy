"""Altimetry shape carrier: sea-surface height as a function of range.

The top-surface analogue of :class:`uacpy.core.bathymetry.Bathymetry` — a 1-D
profile (``height`` vs ``range``). Heights are positive **up** (z = 0 at mean
sea level), so a crest is positive and a trough negative. Re-exported from
:mod:`uacpy.core.environment` for stable import paths.
"""

import copy as _copy
import numpy as np
from dataclasses import dataclass

from uacpy.core.exceptions import ConfigurationError
from uacpy.core._carrier_validate import (
    _require_finite, _require_non_negative, _require_strictly_increasing,
)


@dataclass
class Altimetry:
    """Sea-surface height (m, positive up) as a function of range (m).

    A 1-D grid library carrier mirroring :class:`Bathymetry`: select a height
    with :meth:`at` (nearest), :meth:`isel` (positional) or :meth:`eval`
    (interpolated). The single range axis is collapsed, so those return the
    **height value(s)** directly (a scalar for a scalar range, an array for an
    array of ranges).

    Attributes
    ----------
    ranges : ndarray, shape (N,)
        Range axis in metres, monotonically increasing.
    heights : ndarray, shape (N,)
        Surface height in metres at each range (positive up; any sign).
    """

    ranges: np.ndarray
    heights: np.ndarray

    def __post_init__(self):
        self.ranges = np.array(self.ranges, dtype=float).reshape(-1)
        self.heights = np.array(self.heights, dtype=float).reshape(-1)
        if self.ranges.size != self.heights.size:
            raise ConfigurationError(
                f"Altimetry: ranges ({self.ranges.size}) and heights "
                f"({self.heights.size}) must have the same length.")
        if self.ranges.size == 0:
            raise ConfigurationError("Altimetry: needs at least one point.")
        _require_non_negative(self.ranges, "Altimetry ranges", hint="metres")
        _require_finite(self.heights, "Altimetry heights", hint="metres, positive up")
        if self.ranges.size > 1:
            _require_strictly_increasing(self.ranges, "Altimetry.ranges")

    # ── constructors ────────────────────────────────────────────────────────
    @classmethod
    def coerce(cls, value):
        """Coerce ``None`` / ``Altimetry`` / ``(N, 2)`` ``(range, height)``
        pairs into an :class:`Altimetry` (``None`` passes through — a flat
        z = 0 surface)."""
        if value is None or isinstance(value, Altimetry):
            return value
        try:
            arr = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            raise ConfigurationError(
                f"Altimetry: must be shape (N, 2) as [(range, height_m), ...]; "
                f"got non-numeric {value!r}.")
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ConfigurationError(
                f"Altimetry: must have shape (N, 2) as [(range, height_m), "
                f"...]; got shape {arr.shape}.")
        return cls(ranges=arr[:, 0], heights=arr[:, 1])

    def to_pairs(self) -> np.ndarray:
        """``(N, 2)`` ``(range, height)`` view."""
        return np.column_stack([self.ranges, self.heights])

    def copy(self) -> 'Altimetry':
        """Deep copy (symmetric with the other carriers)."""
        return _copy.deepcopy(self)

    # ── derived ─────────────────────────────────────────────────────────────
    @property
    def n_ranges(self) -> int:
        return int(self.ranges.size)

    @property
    def range_max(self) -> float:
        return float(np.max(self.ranges))

    @property
    def is_range_dependent(self) -> bool:
        """True when the surface height varies with range."""
        return self.ranges.size > 1 and bool(np.ptp(self.heights) > 0)

    # ── grid-library selectors ──────────────────────────────────────────────
    def at(self, *, range):
        """Nearest surface height to ``range`` (m) — never fabricates. Float
        for a scalar range, array for an array. See :meth:`eval` (interpolate)
        and :meth:`isel` (positional)."""
        return self._query(range, 'nearest')

    def eval(self, *, range, method: str = 'linear'):
        """Interpolated surface height at ``range`` (m). ``method`` is
        ``'linear'`` (default), ``'nearest'`` or ``'cubic'``; constant
        extrapolation past the ends. The interpolating counterpart of
        :meth:`at`."""
        return self._query(range, method)

    def isel(self, *, range):
        """Surface height at integer index ``range`` — the positional
        counterpart of :meth:`at`."""
        idx = np.asarray(range)
        n = self.heights.size
        if idx.ndim == 0:
            i = int(idx)
            if not -n <= i < n:
                raise IndexError(
                    f"Altimetry.isel: range index {i} out of range for "
                    f"{n} point(s)")
        out = self.heights[idx.astype(int)]
        return float(out) if idx.ndim == 0 else out

    def _query(self, range, method):
        from uacpy.core._grid import query_profile
        return query_profile(self.ranges, self.heights, range, method)
