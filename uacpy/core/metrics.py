"""Quantitative agreement metrics between TL fields.

Stand-alone helpers used by tests, examples, and end-user comparison
scripts. Keeps numeric-comparison logic out of plotting and IO modules.

Public helpers: :func:`tl_rmse`, :func:`tl_max_error`, :func:`tl_bias`.
All accept a pair of 2-D :class:`~uacpy.Field` instances. Read TL via
``field.db`` regardless of whether the field stores complex pressure or
real dB — :class:`Field` handles the conversion.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from uacpy.core.results import Field
from uacpy.core.exceptions import ConfigurationError


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
) -> Tuple[np.ndarray, np.ndarray]:
    """Shared validation for TL-pair metrics.

    Both inputs must be 2-D ``(depth, range)`` fields carrying the same
    :attr:`~uacpy.Field.kind`. TL is pulled from ``.db`` (handles complex → dB
    conversion). Returns ``(diff, finite)`` —
    the signed TL difference and the boolean mask of finite cells inside the
    requested window.
    """
    for label, f in (('field_a', field_a), ('field_b', field_b)):
        if not isinstance(f, Field):
            raise ConfigurationError(
                f"{fname}: {label} must be a Field; got {type(f).__name__}"
            )
        if list(f.coords) != ['depth', 'range']:
            raise ConfigurationError(
                f"{fname}: {label} must be a 2-D (depth, range) Field; "
                f"got coords {list(f.coords)}"
            )
    # Compare the QUANTITY, not the representation: complex pressure and real
    # TL are the same quantity written two ways and ``.db`` reconciles them,
    # while reverberation shares TL's dB representation exactly and is a
    # different quantity. Same rule ``compare_models`` applies before it puts
    # two fields on one colour scale.
    if field_a.kind != field_b.kind:
        raise ConfigurationError(
            f"{fname}: field_a is a {field_a.kind!r} field but field_b is "
            f"{field_b.kind!r} — these are different physical quantities and "
            f"their difference is not an agreement metric.",
            remediation="Compare like with like (two TL fields, or two "
                        "reverberation fields).",
        )

    # ``Field.db`` refuses a real field whose unit is not dB, and raises
    # AttributeError doing it. Matching the kind above is not enough to make
    # the pair a TL pair — a probability-of-detection field passes it — so the
    # unit is checked here, the way ``ResultStack.db`` pre-checks its slabs.
    for label, f in (('field_a', field_a), ('field_b', field_b)):
        if not f.is_complex and f.unit != 'dB':
            raise ConfigurationError(
                f"{fname}: {label} is in {f.unit!r}, not dB, so its values "
                f"are not a level and their difference is not a TL error.",
                remediation="Compare two TL (or complex pressure) fields; "
                            "read the raw values via field.data.",
            )

    # dtype=float, not the field's own: ``Field.db`` hands back the stored
    # dtype, and a ``.shd``-backed result is float32. The differences below
    # are reduced to one RMSE / bias scalar, and that accumulation is done in
    # float64 whichever engine produced either side.
    da = np.asarray(field_a.db, dtype=float)
    db = np.asarray(field_b.db, dtype=float)
    if da.shape != db.shape:
        raise ConfigurationError(
            f"{fname}: shape mismatch — field_a {da.shape} vs field_b {db.shape}"
        )

    depths = field_a.coords['depth']
    ranges = field_a.coords['range']
    depths_b = field_b.coords['depth']
    ranges_b = field_b.coords['range']
    if depths.shape != depths_b.shape or ranges.shape != ranges_b.shape:
        raise ConfigurationError(f"{fname}: depth/range axes must have matching shapes")
    # Two tolerance terms: atol=1e-3 admits sub-millimetre unit-conversion
    # rounding near the origin, and rtol=1e-5 scales the allowance with the
    # coordinate (it dominates beyond 100 m — 1 mm at 100 m, 1 m at 100 km).
    # Genuinely different grids differ by whole grid steps and still raise.
    if not np.allclose(depths, depths_b, rtol=1e-5, atol=1e-3):
        raise ConfigurationError(
            f"{fname}: depth axes differ — sample-cells are not aligned. "
            "Resample one field onto the other's grid before comparing."
        )
    if not np.allclose(ranges, ranges_b, rtol=1e-5, atol=1e-3):
        raise ConfigurationError(
            f"{fname}: range axes differ — sample-cells are not aligned. "
            "Resample one field onto the other's grid before comparing."
        )

    rmask = _resolve_window(ranges, range_window)
    zmask = _resolve_window(depths, depth_window)
    region_mask = zmask[:, None] & rmask[None, :]

    diff = da - db
    finite = np.isfinite(diff) & region_mask
    if not np.any(finite):
        raise ConfigurationError(
            f"{fname}: window contains no finite cells "
            f"(range_window={range_window}, depth_window={depth_window})"
        )
    return diff, finite


def tl_rmse(
    field_a: Field,
    field_b: Field,
    range_window: Optional[Tuple[float, float]] = None,
    depth_window: Optional[Tuple[float, float]] = None,
) -> float:
    """Root-mean-square TL difference between two TL fields.

    Both fields must be sampled on the same ``depths`` and ``ranges``
    grid. Agreement is checked with a mixed tolerance — 1 mm absolute plus
    1e-5 relative, the latter dominating beyond 100 m (models interpolate
    onto the requested receiver grid, so two runs of the same grid match
    within it); grids differing by more raise — resample one onto the
    other first.

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
    diff, finite = _validate_tl_pair_and_window(
        field_a, field_b, range_window, depth_window, fname='tl_rmse'
    )
    return float(np.sqrt(np.mean(diff[finite] ** 2)))


def tl_max_error(
    field_a: Field,
    field_b: Field,
    range_window: Optional[Tuple[float, float]] = None,
    depth_window: Optional[Tuple[float, float]] = None,
) -> float:
    """Maximum absolute TL difference between two TL fields."""
    diff, finite = _validate_tl_pair_and_window(
        field_a, field_b, range_window, depth_window, fname='tl_max_error'
    )
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
    diff, finite = _validate_tl_pair_and_window(
        field_a, field_b, range_window, depth_window, fname='tl_bias'
    )
    return float(np.mean(diff[finite]))


__all__ = ["tl_rmse", "tl_max_error", "tl_bias"]
