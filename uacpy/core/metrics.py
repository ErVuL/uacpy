"""Quantitative agreement metrics between TL fields.

Stand-alone helpers used by tests, examples, and end-user comparison
scripts. Keeps numeric-comparison logic out of plotting and IO modules.

Public helpers: :func:`tl_rmse`, :func:`tl_max_error`, :func:`tl_bias`.
All accept a pair of 2-D :class:`~uacpy.Field` instances. Read TL via
``field.tl`` regardless of whether the field stores complex pressure or
real dB — :class:`Field` handles the conversion.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from uacpy.core.results import Field


def _resolve_window(
    coords: np.ndarray, window: Optional[Tuple[float, float]]
) -> np.ndarray:
    """Boolean mask selecting ``coords`` inside ``window`` (inclusive).
    ``window=None`` returns an all-True mask."""
    if window is None:
        return np.ones_like(coords, dtype=bool)
    lo, hi = window
    return (coords >= lo) & (coords <= hi)


def _validate_tl_pair_and_window(
    field_a: Field,
    field_b: Field,
    range_window: Optional[Tuple[float, float]],
    depth_window: Optional[Tuple[float, float]],
    fname: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared validation for TL-pair metrics.

    Both inputs must be 2-D ``(depth, range)`` fields. TL is pulled from
    ``.tl`` (handles complex → dB conversion). Returns
    ``(da, db, region_mask, finite)``.
    """
    for label, f in (('field_a', field_a), ('field_b', field_b)):
        if not isinstance(f, Field):
            raise TypeError(
                f"{fname}: {label} must be a Field; got {type(f).__name__}"
            )
        if list(f.coords) != ['depth', 'range']:
            raise ValueError(
                f"{fname}: {label} must be a 2-D (depth, range) Field; "
                f"got coords {list(f.coords)}"
            )

    da = np.asarray(field_a.tl)
    db = np.asarray(field_b.tl)
    if da.shape != db.shape:
        raise ValueError(
            f"{fname}: shape mismatch — field_a {da.shape} vs field_b {db.shape}"
        )

    depths = field_a.coords['depth']
    ranges = field_a.coords['range']
    if (depths.shape != field_b.coords['depth'].shape
            or ranges.shape != field_b.coords['range'].shape):
        raise ValueError(f"{fname}: depth/range axes must have matching shapes")

    rmask = _resolve_window(ranges, range_window)
    zmask = _resolve_window(depths, depth_window)
    region_mask = zmask[:, None] & rmask[None, :]

    diff = da - db
    finite = np.isfinite(diff) & region_mask
    if not np.any(finite):
        raise ValueError(
            f"{fname}: window contains no finite cells "
            f"(range_window={range_window}, depth_window={depth_window})"
        )
    return da, db, region_mask, finite


def tl_rmse(
    field_a: Field,
    field_b: Field,
    range_window: Optional[Tuple[float, float]] = None,
    depth_window: Optional[Tuple[float, float]] = None,
) -> float:
    """Root-mean-square TL difference between two TL fields.

    Both fields must share the same ``depths`` and ``ranges`` axes; the
    caller is responsible for ensuring they're sampled compatibly (e.g.
    by passing identical ``Receiver`` objects to both models).

    Parameters
    ----------
    field_a, field_b : Field
        2-D ``(depth, range)`` fields. Broadband / time-domain fields
        raise.
    range_window : (float, float), optional
        ``(rmin_m, rmax_m)`` inclusive. Defaults to all ranges.
    depth_window : (float, float), optional
        ``(zmin_m, zmax_m)`` inclusive. Defaults to all depths.

    Returns
    -------
    float
        RMSE in dB over the windowed grid, ignoring non-finite cells.
    """
    da, db, _, finite = _validate_tl_pair_and_window(
        field_a, field_b, range_window, depth_window, fname='tl_rmse'
    )
    diff = da - db
    return float(np.sqrt(np.mean(diff[finite] ** 2)))


def tl_max_error(
    field_a: Field,
    field_b: Field,
    range_window: Optional[Tuple[float, float]] = None,
    depth_window: Optional[Tuple[float, float]] = None,
) -> float:
    """Maximum absolute TL difference between two TL fields."""
    da, db, _, finite = _validate_tl_pair_and_window(
        field_a, field_b, range_window, depth_window, fname='tl_max_error'
    )
    diff = da - db
    return float(np.max(np.abs(diff[finite])))


def tl_bias(
    field_a: Field,
    field_b: Field,
    range_window: Optional[Tuple[float, float]] = None,
    depth_window: Optional[Tuple[float, float]] = None,
) -> float:
    """Mean signed TL difference (bias) between two TL fields.

    Positive values mean ``field_a`` reports higher TL (more attenuation)
    than ``field_b`` on average."""
    da, db, _, finite = _validate_tl_pair_and_window(
        field_a, field_b, range_window, depth_window, fname='tl_bias'
    )
    diff = da - db
    return float(np.mean(diff[finite]))


__all__ = ["tl_rmse", "tl_max_error", "tl_bias"]
