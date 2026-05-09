"""Quantitative agreement metrics between TL fields.

Stand-alone helpers used by tests, examples, and end-user comparison
scripts. Keeps numeric-comparison logic out of plotting and IO modules.

Public helpers: :func:`tl_rmse`, :func:`tl_max_error`, :func:`tl_bias`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from uacpy.core.constants import PRESSURE_FLOOR
from uacpy.core.results import PressureField


def _to_db(field: PressureField) -> np.ndarray:
    """Return TL in dB at ``field.data.shape`` — does not squeeze, so
    metrics can pair this with the 2-D ``(depth, range)`` window mask
    even when the field has a singleton axis. Mirrors :attr:`PressureField.tl`
    for the dB conversion but skips the user-facing auto-squeeze."""
    if field.units == 'dB':
        return np.asarray(field.data)
    p_abs = np.maximum(np.abs(field.data), PRESSURE_FLOOR)
    return -20.0 * np.log10(p_abs)


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


def _validate_tl_pair_and_window(
    field_a: PressureField,
    field_b: PressureField,
    range_window: Optional[Tuple[float, float]],
    depth_window: Optional[Tuple[float, float]],
    fname: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared validation for TL-pair metrics.

    Checks both fields are narrowband :class:`PressureField` instances on
    a common grid, builds the window mask, and returns
    ``(da, db, region_mask, finite)``. ``da`` / ``db`` are TL in dB —
    pulled from ``.tl`` so either ``units='complex'`` or ``units='dB'``
    storage works. Raises if the window contains no finite cells.
    """
    for label, f in (('field_a', field_a), ('field_b', field_b)):
        if not isinstance(f, PressureField):
            raise TypeError(
                f"{fname}: {label} must be a PressureField; got {type(f).__name__}"
            )
    if field_a.data.ndim != 2 or field_b.data.ndim != 2:
        raise ValueError(
            f"{fname}: broadband TL field (3-D data) is not supported; "
            "extract a single frequency via .at_frequency(f) first."
        )

    da = _to_db(field_a)
    db = _to_db(field_b)
    if da.shape != db.shape:
        raise ValueError(
            f"{fname}: shape mismatch — field_a {da.shape} vs field_b {db.shape}"
        )

    depths = np.asarray(field_a.depths)
    ranges = np.asarray(field_a.ranges)
    if depths.shape != np.asarray(field_b.depths).shape or ranges.shape != np.asarray(field_b.ranges).shape:
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
    da, db, _, finite = _validate_tl_pair_and_window(
        field_a, field_b, range_window, depth_window, fname='tl_rmse'
    )
    diff = da - db
    return float(np.sqrt(np.mean(diff[finite] ** 2)))


def tl_max_error(
    field_a: PressureField,
    field_b: PressureField,
    range_window: Optional[Tuple[float, float]] = None,
    depth_window: Optional[Tuple[float, float]] = None,
) -> float:
    """Maximum absolute TL difference between two TL fields.

    Same input contract as :func:`tl_rmse`.

    Returns
    -------
    float
        ``max(|field_a - field_b|)`` in dB over the windowed grid,
        ignoring non-finite cells.
    """
    da, db, _, finite = _validate_tl_pair_and_window(
        field_a, field_b, range_window, depth_window, fname='tl_max_error'
    )
    diff = da - db
    return float(np.max(np.abs(diff[finite])))


def tl_bias(
    field_a: PressureField,
    field_b: PressureField,
    range_window: Optional[Tuple[float, float]] = None,
    depth_window: Optional[Tuple[float, float]] = None,
) -> float:
    """Mean signed TL difference (bias) between two TL fields.

    Same input contract as :func:`tl_rmse`. Positive values mean
    ``field_a`` reports higher TL (more attenuation) than ``field_b``
    on average.

    Returns
    -------
    float
        ``mean(field_a - field_b)`` in dB over the windowed grid,
        ignoring non-finite cells.
    """
    da, db, _, finite = _validate_tl_pair_and_window(
        field_a, field_b, range_window, depth_window, fname='tl_bias'
    )
    diff = da - db
    return float(np.mean(diff[finite]))


__all__ = ["tl_rmse", "tl_max_error", "tl_bias"]
