"""Quantitative agreement metrics between TL fields.

Stand-alone helpers used by tests, examples, and end-user comparison
scripts. Keeps numeric-comparison logic out of plotting and IO modules.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from uacpy.core.results import TLField


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
    field_a: TLField,
    field_b: TLField,
    range_window: Optional[Tuple[float, float]] = None,
    depth_window: Optional[Tuple[float, float]] = None,
) -> float:
    """Root-mean-square TL difference between two ``TLField`` objects.

    Both fields must share the same ``depths`` and ``ranges`` axes; the
    caller is responsible for ensuring they're sampled compatibly (e.g.
    by passing identical ``Receiver`` objects to both models).

    Parameters
    ----------
    field_a, field_b : TLField
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
        If either input is not a :class:`TLField`.
    ValueError
        If shapes/axes don't match, the fields are broadband, or the
        window selects no finite cells.
    """
    if not isinstance(field_a, TLField) or not isinstance(field_b, TLField):
        raise TypeError(
            "tl_rmse: both inputs must be TLField; got "
            f"{type(field_a).__name__} and {type(field_b).__name__}"
        )
    if field_a.data.ndim != 2 or field_b.data.ndim != 2:
        raise ValueError(
            "tl_rmse: broadband TLField (3-D data) is not supported; "
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
