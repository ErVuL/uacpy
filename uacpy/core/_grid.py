"""Shared 1-D axis-collapse used by the grid-library ``eval``/``at`` selectors.

One implementation of "evaluate along an axis and drop it" so ``Field``,
``SoundSpeedProfile`` and ``ReflectionCoefficient`` share identical
interpolation semantics. ``at`` is ``method='nearest'``; ``eval`` defaults to
``'linear'`` and accepts ``'cubic'`` too.
"""

from __future__ import annotations

import numpy as np

from uacpy.core.exceptions import ConfigurationError

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


def query_profile(ranges, values, query, method='linear'):
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
