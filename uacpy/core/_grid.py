"""Shared 1-D axis-collapse used by the grid-library ``eval``/``at`` selectors.

One implementation of "evaluate along an axis and drop it" so ``Field``,
``SoundSpeedProfile`` and ``ReflectionCoefficient`` share identical
interpolation semantics. ``at`` is ``method='nearest'``; ``eval`` defaults to
``'linear'`` and accepts ``'cubic'`` too.

:class:`_RangeProfile` is the companion base for the two single-axis
``value(range)`` carriers (``Bathymetry``, ``Altimetry``).
"""

from __future__ import annotations

import copy as _copy

import numpy as np

from uacpy.core.exceptions import ConfigurationError
from uacpy.core.constants import DECK_RANGE_RESOLUTION_M
from uacpy.core._carrier_validate import (
    _require_non_negative, _require_strictly_increasing,
)

INTERP_METHODS = ('linear', 'nearest', 'cubic')


def collapse_axis(arr, axis_values, value, method='linear', *, axis=0):
    """Collapse ``arr`` along ``axis`` at coordinate ``value``.

    Returns ``(reduced_array, coord)`` where ``reduced_array`` has ``axis``
    removed and ``coord`` is the coordinate the slice sits at (the nearest
    stored value for ``'nearest'``, else the requested value clamped into
    range). ``axis_values`` is the 1-D, increasing coordinate vector. Values
    outside ``[axis_values[0], axis_values[-1]]`` use constant extrapolation.
    """
    if method not in INTERP_METHODS:
        raise ConfigurationError(
            f"interpolation method must be one of {INTERP_METHODS}; "
            f"got {method!r}"
        )
    x = np.asarray(axis_values, dtype=float)
    n = x.size
    if n <= 1:
        return np.take(arr, 0, axis=axis), float(x[0])
    if method == 'nearest':
        i = int(np.argmin(np.abs(x - float(value))))
        return np.take(arr, i, axis=axis), float(x[i])

    # searchsorted and the clamp below both read the axis as ascending, so a
    # descending one is walked in reverse together with the data it indexes.
    if x[0] > x[-1]:
        x = x[::-1]
        arr = np.flip(arr, axis=axis)

    v = float(min(max(float(value), float(x[0])), float(x[-1])))   # clamp
    if method == 'linear':
        j = int(np.clip(np.searchsorted(x, v), 1, n - 1))
        x0, x1 = float(x[j - 1]), float(x[j])
        w = 0.0 if x1 == x0 else (v - x0) / (x1 - x0)
        lo = np.take(arr, j - 1, axis=axis)
        hi = np.take(arr, j, axis=axis)
        return lo * (1.0 - w) + hi * w, v

    # cubic
    if n < 4:
        raise ConfigurationError(
            f"method='cubic' needs at least 4 samples on the axis; got {n}. "
            f"Use method='linear' (or 'nearest')."
        )
    from scipy.interpolate import interp1d
    moved = np.moveaxis(np.asarray(arr), axis, -1)

    def _interp1d(real):
        f = interp1d(x, real, kind='cubic', axis=-1, bounds_error=False,
                     fill_value=(real[..., 0], real[..., -1]))
        return f(v)
    if np.iscomplexobj(moved):
        out = _interp1d(moved.real) + 1j * _interp1d(moved.imag)
    else:
        out = _interp1d(moved)
    return out, v


class _RangeProfile:
    """Base for the 1-D ``value(range)`` shape carriers.

    :class:`~uacpy.core.bathymetry.Bathymetry` (seafloor depth, positive down)
    and :class:`~uacpy.core.altimetry.Altimetry` (sea-surface height, positive
    up) differ only in the name of their value vector and the guard it must
    satisfy; the range-axis validation, the ``at``/``eval``/``isel`` selectors
    and the derived properties all live here.

    Subclasses are ``@dataclass(eq=False, repr=False)`` carrying ``ranges``
    plus the vector named by ``_VALUE_FIELD``. They call
    :meth:`_init_range_profile` from ``__post_init__`` and implement
    :meth:`_validate_values`.
    """

    # Declaration hooks: a subclass states what it holds, so ``__repr__`` and
    # ``_plot_range_profile`` tell the carriers apart by declaration rather
    # than by type-sniffing a concrete field name. ``_VALUE_FIELD`` names the
    # value vector; ``_VALUE_LABEL`` is the noun for it, capitalised into the
    # plot's y-label; ``_AXIS_DOWN`` is True when the value grows downward
    # (depth), which inverts that axis.
    _VALUE_FIELD: str = ''
    _VALUE_LABEL: str = ''
    _VALUE_UNIT: str = 'm'
    _AXIS_DOWN: bool = False

    @property
    def _values(self) -> np.ndarray:
        """The value vector, whatever ``_VALUE_FIELD`` names it."""
        return getattr(self, self._VALUE_FIELD)

    def _validate_values(self) -> None:
        """Subclass hook: guard the value vector (sign / finiteness)."""
        raise NotImplementedError

    def _init_range_profile(self) -> None:
        cls = type(self).__name__
        self.ranges = np.array(self.ranges, dtype=float).reshape(-1)
        setattr(self, self._VALUE_FIELD,
                np.array(self._values, dtype=float).reshape(-1))
        if self.ranges.size != self._values.size:
            raise ConfigurationError(
                f"{cls}: ranges ({self.ranges.size}) and {self._VALUE_FIELD} "
                f"({self._values.size}) must have the same length.")
        if self.ranges.size == 0:
            raise ConfigurationError(f"{cls}: needs at least one point.")
        _require_non_negative(self.ranges, f"{cls} ranges", hint="metres")
        self._validate_values()
        if self.ranges.size > 1:
            _require_strictly_increasing(
                self.ranges, f"{cls}.ranges", min_step=DECK_RANGE_RESOLUTION_M)

    def __repr__(self) -> str:
        v = self._values
        r_lo = float(self.ranges[0]) / 1000
        r_hi = float(self.ranges[-1]) / 1000
        return (f"{type(self).__name__}(n_r={self.ranges.size}, "
                f"range=[{r_lo:g}, {r_hi:g}] km, "
                f"{self._VALUE_LABEL}=[{float(np.min(v)):g}, "
                f"{float(np.max(v)):g}] {self._VALUE_UNIT})")

    def to_pairs(self) -> np.ndarray:
        """``(N, 2)`` ``(range, value)`` array."""
        return np.column_stack([self.ranges, self._values])

    def copy(self):
        """Deep copy (symmetric with the other carriers)."""
        return _copy.deepcopy(self)

    @property
    def n_ranges(self) -> int:
        return int(self.ranges.size)

    @property
    def range_max(self) -> float:
        return float(np.max(self.ranges))

    @property
    def is_range_dependent(self) -> bool:
        """True when the value varies with range."""
        return self.ranges.size > 1 and bool(np.ptp(self._values) > 0)

    def at(self, *, range):
        """Nearest value to ``range`` (m) — never fabricates. Float for a
        scalar range, array for an array of ranges. See :meth:`eval`
        (interpolate) and :meth:`isel` (positional)."""
        return self._query(range, 'nearest')

    def eval(self, *, range, method: str = 'linear'):
        """Interpolated value at ``range`` (m). ``method`` is ``'linear'``
        (default), ``'nearest'`` or ``'cubic'``; constant extrapolation past
        the ends. The interpolating counterpart of :meth:`at`."""
        return self._query(range, method)

    def isel(self, *, range):
        """Value at integer index ``range`` — the positional counterpart of
        :meth:`at`."""
        idx = np.asarray(range)
        v = self._values
        n = v.size
        if idx.ndim == 0:
            i = int(idx)
            if not -n <= i < n:
                raise IndexError(
                    f"{type(self).__name__}.isel: range index {i} out of "
                    f"range for {n} point(s)")
        out = v[idx.astype(int)]
        return float(out) if idx.ndim == 0 else out

    def _query(self, range, method):
        return _query_profile(self.ranges, self._values, range, method)

    def plot(self, ax=None, *, title=None, figsize=(10, 4), **mpl_kw):
        """Plot the profile against range.

        Depth-valued profiles (:class:`~uacpy.core.bathymetry.Bathymetry`)
        point their axis downward; height-valued ones
        (:class:`~uacpy.core.altimetry.Altimetry`) keep it upward."""
        from uacpy.visualization.plots.environment import _plot_range_profile
        return _plot_range_profile(self, ax=ax, title=title,
                                   figsize=figsize, **mpl_kw)


def _query_profile(ranges, values, query, method='linear'):
    """Sample a 1-D ``(range -> value)`` profile at ``query`` range(s).

    Shared by the :class:`Bathymetry` / :class:`Altimetry` carriers (seafloor
    depth and sea-surface height vs range). ``'nearest'`` / ``'linear'`` /
    ``'cubic'`` with constant extrapolation past the ends. Returns a float for
    a scalar ``query``, an ndarray for an array of queries.
    """
    if method not in INTERP_METHODS:
        raise ConfigurationError(
            f"method must be one of {INTERP_METHODS}; got {method!r}")
    x = np.asarray(ranges, dtype=float)
    y = np.asarray(values, dtype=float)
    rq = np.asarray(query, dtype=float)
    scalar = rq.ndim == 0
    if x.size == 1:
        out = np.full(rq.shape, float(y[0]))
    elif method == 'nearest':
        idx = np.abs(x[:, None] - rq.ravel()[None, :]).argmin(0)
        out = y[idx].reshape(rq.shape)
    elif method == 'linear':
        out = np.interp(rq, x, y)                    # constant extrapolation
    else:  # cubic
        if x.size < 4:
            raise ConfigurationError(
                f"method='cubic' needs >= 4 ranges; got {x.size}. Use 'linear'.")
        from scipy.interpolate import interp1d
        f = interp1d(x, y, kind='cubic', bounds_error=False,
                     fill_value=(float(y[0]), float(y[-1])))
        out = f(np.clip(rq, x[0], x[-1]))
    return float(out) if scalar else np.asarray(out, dtype=float)
