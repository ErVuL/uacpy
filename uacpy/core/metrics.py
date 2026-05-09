"""Quantitative agreement metrics between TL fields.

Stand-alone helpers used by tests, examples, and end-user comparison
scripts. Keeps numeric-comparison logic out of plotting and IO modules.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from uacpy.core.results import PressureField


def _resolve_window(
    coords: np.ndarray, window: Optional[Tuple[float, float]]
) -> np.ndarray:
    """Boolean mask selecting ``coords`` inside ``window`` (inclusive).

    ``window=None`` returns an all-True mask of the right length.
    """
    if window is None:
        return np.ones_like(coords, dtype=bool)
    lo, hi = window
    return (coords >= lo) & (coords <= hi)


def tl_rmse(
    field_a: PressureField,
    field_b: PressureField,
    range_window: Optional[Tuple[float, float]] = None,
    depth_window: Optional[Tuple[float, float]] = None,
) -> float:
    """Root-mean-square TL difference between two TL fields.

    Both fields must share the same ``depths`` and ``ranges`` axes; the
    caller is responsible for ensuring they're sampled compatibly (e.g.
    by passing identical ``Receiver`` objects to both models).

    Parameters
    ----------
    field_a, field_b : PressureField (units='dB')
        Narrowband TL fields. Broadband ``(n_d, n_r, n_f)`` fields raise.
    range_window : (float, float), optional
        ``(rmin_m, rmax_m)`` inclusive. Defaults to all ranges.
    depth_window : (float, float), optional
        ``(zmin_m, zmax_m)`` inclusive. Defaults to all depths.

    Returns
    -------
    float
        RMSE in dB over the windowed grid, ignoring non-finite cells.

    Raises
    ------
    TypeError
        If either input is not a dB-units :class:`PressureField`.
    ValueError
        If shapes/axes don't match, the fields are broadband, or the
        window selects no finite cells.
    """
    for label, f in (('field_a', field_a), ('field_b', field_b)):
        if not isinstance(f, PressureField) or f.units != 'dB':
            raise TypeError(
                f"tl_rmse: {label} must be a PressureField with units='dB' "
                f"(TL); got {type(f).__name__}"
                + (f" (units={f.units!r})" if isinstance(f, PressureField) else "")
            )
    if field_a.data.ndim != 2 or field_b.data.ndim != 2:
        raise ValueError(
            "tl_rmse: broadband TL field (3-D data) is not supported; "
            "extract a single frequency via .at_frequency(f) first."
        )

    da = np.asarray(field_a.data)
    db = np.asarray(field_b.data)
    if da.shape != db.shape:
        raise ValueError(
            f"tl_rmse: shape mismatch — field_a {da.shape} vs field_b {db.shape}"
        )

    depths = np.asarray(field_a.depths)
    ranges = np.asarray(field_a.ranges)
    if depths.shape != np.asarray(field_b.depths).shape or ranges.shape != np.asarray(field_b.ranges).shape:
        raise ValueError("tl_rmse: depth/range axes must have matching shapes")

    rmask = _resolve_window(ranges, range_window)
    zmask = _resolve_window(depths, depth_window)
    region_mask = zmask[:, None] & rmask[None, :]

    diff = da - db
    finite = np.isfinite(diff) & region_mask
    if not np.any(finite):
        raise ValueError(
            "tl_rmse: window contains no finite cells "
            f"(range_window={range_window}, depth_window={depth_window})"
        )
    return float(np.sqrt(np.mean(diff[finite] ** 2)))


__all__ = ["tl_rmse"]
