"""Quantitative agreement metrics between TL fields.

Stand-alone helpers used by tests, examples, and end-user comparison
scripts. Keeps numeric-comparison logic out of plotting and IO modules.

Public helpers: :func:`tl_rmse`, :func:`tl_max_error`, :func:`tl_bias`,
:func:`tl_rmse_on_shared_ranges`. Every one takes ``(field, reference)``
and every signed difference is ``field - reference``.
:func:`tl_rmse_on_shared_ranges` takes two Fields read at one depth, or two
``(ranges, tl_dB)`` curves (a measurement, another code's output).

:func:`tl_rmse`, :func:`tl_max_error` and :func:`tl_bias` take either two
plain dB arrays of one shape (a measured TL curve, a table from another
code) or two :class:`~uacpy.Field` instances on the same ``depth`` and/or
``range`` axes — a 2-D ``(depth, range)`` grid, or a 1-D cut such as
``field.at(depth=z)``. The Field form checks the pair is one quantity on one
grid, reads TL via ``field.dB`` (complex pressure or real dB alike) and
hands the two arrays to the same computation the array form runs.

A cell either field reports as carrying **no energy** (the 600 dB marker,
:func:`~uacpy.core.acoustics.no_energy_mask`) is no data, not a 600 dB loss,
and is left out of every metric the way a NaN is; a ``FallbackWarning`` counts
the cells dropped. A receiver row at the pressure-release surface still
compares genuine nulls — a model's 76 dB against another's 325 dB of
round-off — so window it out (``depth_window``) when the models' null
depths are not the question.
"""

from __future__ import annotations

from typing import Optional, Tuple

import warnings

import numpy as np

from uacpy.core.acoustics.levels import no_energy_mask
from uacpy.core.results import Field
from uacpy.core.exceptions import ConfigurationError, FallbackWarning
from uacpy.core._warn_frames import USER_FRAME_SKIP


def _warn_no_energy_cells(n: int, fname: str) -> None:
    """Say how many no-energy cells a metric left out, so an agreement is
    never silently computed over fewer cells than the caller thinks."""
    if n:
        warnings.warn(
            f"{fname}: {n} cell(s) where a field reports no energy (the "
            f"600 dB marker) are left out, as NaN cells are; the metric is "
            f"over the remaining cells.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)


def _resolve_window(
    coords: np.ndarray, window: Optional[Tuple[float, float]]
) -> np.ndarray:
    """Boolean mask selecting ``coords`` inside ``window`` (inclusive).
    ``window=None`` returns an all-True mask."""
    if window is None:
        return np.ones_like(coords, dtype=bool)
    lo, hi = window
    return (coords >= lo) & (coords <= hi)


# The coordinate axes a TL metric compares over. A Field on any other axis
# (frequency, time, receiver) is not one TL value per cell of a grid.
_TL_AXES = ('depth', 'range')


def _refuse_axes_beyond_tl(f, label: str, fname: str) -> None:
    """Refuse a Field ``f`` with an axis besides depth and range, naming
    the step that removes it: one source, or the sources' sum, for the
    ``source_depth`` axis of a multi-depth run; one value of any other."""
    axes = list(f.coords)
    extra = [a for a in axes if a not in _TL_AXES]
    if axes and not extra:
        return
    others = ', '.join(f'{a}=...' for a in extra if a != 'source_depth')
    raise ConfigurationError(
        f"{fname}: {label} must be a Field on depth and/or range axes (a "
        f"(depth, range) grid or a 1-D cut); got coords {axes}.",
        remediation=(
            "Compare one source with field.at(source_depth=...), or the "
            "sources driven together with field.superpose()."
            + (f" Select one value of each other axis with field.at("
               f"{others})." if others else "")
            if 'source_depth' in extra else
            f"Select one value of each other axis with field.at({others})."
            if others else "Pass a Field on depth and/or range axes."))


def _tl_difference(
    tl_field: np.ndarray,
    tl_reference: np.ndarray,
    region: np.ndarray,
    fname: str,
    window_note: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """``(diff, kept)``: the signed difference ``tl_field - tl_reference``
    and the mask of cells inside ``region`` where both are finite and
    neither is the no-energy marker. The one computation both the array
    and the Field forms of the metrics run."""
    diff = tl_field - tl_reference
    marker = no_energy_mask(tl_field) | no_energy_mask(tl_reference)
    _warn_no_energy_cells(int(np.count_nonzero(marker & region)), fname)
    kept = np.isfinite(diff) & ~marker & region
    if not np.any(kept):
        raise ConfigurationError(
            f"{fname}: the comparison contains no finite cells{window_note}."
        )
    return diff, kept


def _tl_arrays(field, reference, fname):
    """Two plain dB arrays as float64 of one shape; complex refused."""
    arrays = []
    for label, x in (('field', field), ('reference', reference)):
        if np.iscomplexobj(x):
            raise ConfigurationError(
                f"{fname}: {label} is a complex array; the metrics compare "
                f"dB values, and a complex number has no one dB reading.",
                remediation="Pass the loss, e.g. "
                            "uacpy.acoustics.transmission_loss_dB(p), or pass "
                            "two Fields.")
        try:
            arrays.append(np.asarray(x, dtype=float))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"{fname}: {label} must be a Field or an array of dB values; "
                f"got {type(x).__name__} ({exc})") from None
    tl_field, tl_reference = arrays
    if tl_field.shape != tl_reference.shape:
        raise ConfigurationError(
            f"{fname}: shape mismatch — field {tl_field.shape} vs reference "
            f"{tl_reference.shape}.")
    if tl_field.size == 0:
        raise ConfigurationError(f"{fname}: the arrays are empty.")
    return tl_field, tl_reference


def _tl_fields(field, reference, range_window, depth_window, fname):
    """The Field form: check the pair is one dB quantity on one grid, and
    return its two TL arrays and the window's region mask."""
    for label, f in (('field', field), ('reference', reference)):
        if not isinstance(f, Field):
            raise ConfigurationError(
                f"{fname}: field and reference must both be Fields or both "
                f"be dB arrays; {label} is a {type(f).__name__}.")
        _refuse_axes_beyond_tl(f, label, fname)
    if list(field.coords) != list(reference.coords):
        raise ConfigurationError(
            f"{fname}: field is on axes {list(field.coords)} but reference "
            f"is on {list(reference.coords)}; the cells do not correspond.")
    # Compare the QUANTITY, not the representation: complex pressure and real
    # TL are the same quantity written two ways and ``.dB`` reconciles them,
    # while reverberation shares TL's dB representation exactly and is a
    # different quantity. Same rule ``compare_models`` applies before it puts
    # two fields on one colour scale.
    if field.kind != reference.kind:
        raise ConfigurationError(
            f"{fname}: field is a {field.kind!r} field but reference is "
            f"{reference.kind!r} — these are different physical quantities "
            f"and their difference is not an agreement metric.",
            remediation="Compare like with like (two TL fields, or two "
                        "reverberation fields).",
        )

    # ``Field.dB`` refuses a real field whose unit is not dB, and raises
    # AttributeError doing it. Matching the kind above is not enough to make
    # the pair a TL pair — a probability-of-detection field passes it — so the
    # unit is checked here, before ``Field.dB`` is read.
    for label, f in (('field', field), ('reference', reference)):
        if not f.is_complex and f.unit != 'dB':
            raise ConfigurationError(
                f"{fname}: {label} is in {f.unit!r}, not dB, so its values "
                f"are not a level and their difference is not a TL error.",
                remediation="Compare two TL (or complex pressure) fields; "
                            "read the raw values via field.data.",
            )

    # dtype=float, not the field's own: ``Field.dB`` hands back the stored
    # dtype, and a ``.shd``-backed result is float32. The differences are
    # reduced to one RMSE / bias scalar, and that accumulation is done in
    # float64 whichever engine produced either side.
    tl_field = np.asarray(field.dB, dtype=float)
    tl_reference = np.asarray(reference.dB, dtype=float)
    if tl_field.shape != tl_reference.shape:
        _refuse_different_grids(
            f"{fname}: shape mismatch — field {tl_field.shape} vs reference "
            f"{tl_reference.shape}.")

    region = np.ones(tl_field.shape, dtype=bool)
    windows = {'depth': depth_window, 'range': range_window}
    for i, axis in enumerate(field.coords):
        values = np.asarray(field.coords[axis], dtype=float)
        values_ref = np.asarray(reference.coords[axis], dtype=float)
        if values.shape != values_ref.shape:
            raise ConfigurationError(
                f"{fname}: {axis} axes must have matching shapes.")
        # Two tolerance terms: atol=1e-3 admits sub-millimetre
        # unit-conversion rounding near the origin, and rtol=1e-5 scales the
        # allowance with the coordinate (it dominates beyond 100 m — 1 mm at
        # 100 m, 1 m at 100 km). Genuinely different grids differ by whole
        # grid steps and still raise.
        if not np.allclose(values, values_ref, rtol=1e-5, atol=1e-3):
            _refuse_different_grids(
                f"{fname}: {axis} axes differ — sample-cells are not "
                f"aligned.")
        shape = [1] * tl_field.ndim
        shape[i] = values.size
        region = region & _resolve_window(
            values, windows[axis]).reshape(shape)
    for axis, window in windows.items():
        if window is not None and axis not in field.coords:
            raise ConfigurationError(
                f"{fname}: {axis}_window was given but the fields have no "
                f"{axis} axis (coords {list(field.coords)}).")
    return tl_field, tl_reference, region


def _refuse_different_grids(message):
    """Raise ``message`` for two Fields on different grids, with the one
    remedy every such refusal shares."""
    raise ConfigurationError(
        message,
        remediation="Resample one field onto the other's grid "
                    "(Field.resample_to) before comparing, or use "
                    "tl_rmse_on_shared_ranges to compare two models at one "
                    "depth over the ranges both reach.")


def _tl_pair(field, reference, range_window, depth_window, fname):
    """Dispatch to the Field or the array form; returns ``(diff, kept)``."""
    as_fields = isinstance(field, Field) or isinstance(reference, Field)
    if as_fields:
        tl_field, tl_reference, region = _tl_fields(
            field, reference, range_window, depth_window, fname)
    else:
        if range_window is not None or depth_window is not None:
            raise ConfigurationError(
                f"{fname}: plain arrays carry no depth or range axis, so "
                f"range_window/depth_window cannot select from them.",
                remediation="Slice the arrays to the window first, or pass "
                            "two Fields.")
        tl_field, tl_reference = _tl_arrays(field, reference, fname)
        region = np.ones(tl_field.shape, dtype=bool)
    window_note = ("" if range_window is None and depth_window is None else
                   f" (range_window={range_window}, "
                   f"depth_window={depth_window})")
    return _tl_difference(tl_field, tl_reference, region, fname, window_note)


def tl_rmse(
    field,
    reference,
    range_window: Optional[Tuple[float, float]] = None,
    depth_window: Optional[Tuple[float, float]] = None,
) -> float:
    """Root-mean-square TL difference between ``field`` and ``reference``.

    Takes two plain dB arrays of one shape, or two Fields on the same
    ``depth`` and/or ``range`` axes (a 2-D grid or a 1-D cut). Two Fields'
    axes must agree within a mixed tolerance — 1 mm absolute plus 1e-5
    relative, the latter dominating beyond 100 m (models interpolate onto
    the requested receiver grid, so two runs of the same grid match within
    it); grids differing by more raise — resample one onto the other first,
    or use :func:`tl_rmse_on_shared_ranges`.

    Parameters
    ----------
    field, reference : Field or array_like
        Two TL Fields (complex pressure or real dB, same ``kind``), or two
        real dB arrays of one shape. A Field on a frequency or time axis
        raises; a Field and an array together raise.
    range_window : (float, float), optional
        ``(rmin_m, rmax_m)`` inclusive. Defaults to all ranges. Fields only.
    depth_window : (float, float), optional
        ``(zmin_m, zmax_m)`` inclusive. Defaults to all depths. Fields only.

    Returns
    -------
    float
        RMSE in dB over the windowed cells, ignoring non-finite and
        no-energy cells (see the module notes).
    """
    diff, kept = _tl_pair(field, reference, range_window, depth_window,
                          fname='tl_rmse')
    return float(np.sqrt(np.mean(diff[kept] ** 2)))


def tl_max_error(
    field,
    reference,
    range_window: Optional[Tuple[float, float]] = None,
    depth_window: Optional[Tuple[float, float]] = None,
) -> float:
    """Maximum absolute TL difference between ``field`` and ``reference``.

    Same inputs, windows and cell rules as :func:`tl_rmse`.

    Parameters
    ----------
    field, reference : Field or array_like
        Two TL Fields of one ``kind``, or two real dB arrays of one shape
        (see :func:`tl_rmse`).
    range_window : (float, float), optional
        ``(rmin_m, rmax_m)`` inclusive; Fields only.
    depth_window : (float, float), optional
        ``(zmin_m, zmax_m)`` inclusive; Fields only.
    """
    diff, kept = _tl_pair(field, reference, range_window, depth_window,
                          fname='tl_max_error')
    return float(np.max(np.abs(diff[kept])))


def tl_bias(
    field,
    reference,
    range_window: Optional[Tuple[float, float]] = None,
    depth_window: Optional[Tuple[float, float]] = None,
) -> float:
    """Mean signed difference ``field - reference`` (the bias), in dB.

    Positive values mean ``field`` reports a higher value than
    ``reference`` on average: more loss for TL, a louder cell for a
    ``kind='level'`` field. Same inputs, windows and cell rules as
    :func:`tl_rmse`.

    Parameters
    ----------
    field, reference : Field or array_like
        Two TL Fields of one ``kind``, or two real dB arrays of one shape
        (see :func:`tl_rmse`).
    range_window : (float, float), optional
        ``(rmin_m, rmax_m)`` inclusive; Fields only.
    depth_window : (float, float), optional
        ``(zmin_m, zmax_m)`` inclusive; Fields only.
    """
    diff, kept = _tl_pair(field, reference, range_window, depth_window,
                          fname='tl_bias')
    return float(np.mean(diff[kept]))


def _rmse_on_coarser_shared_axis(r_a, tl_a, r_b, tl_b, fname: str) -> float:
    """RMS of ``tl_a - tl_b`` (dB) on the coarser of two increasing range
    axes clipped to their shared span, each curve read there by
    ``np.interp``; no-energy cells are NaN (counted by a FallbackWarning)
    and drop out with the other non-finite cells. ``nan`` when the axes
    share no range or no finite cell."""
    n_marker = int(np.count_nonzero(no_energy_mask(tl_a))
                   + np.count_nonzero(no_energy_mask(tl_b)))
    _warn_no_energy_cells(n_marker, fname)
    tl_a = np.where(no_energy_mask(tl_a), np.nan, tl_a)
    tl_b = np.where(no_energy_mask(tl_b), np.nan, tl_b)

    def _spacing(r):
        return float(np.median(np.diff(r))) if r.size > 1 else float('inf')
    common = r_a if _spacing(r_a) >= _spacing(r_b) else r_b
    common = common[(common >= max(r_a[0], r_b[0]))
                    & (common <= min(r_a[-1], r_b[-1]))]
    if common.size == 0:
        return float('nan')
    residual = np.interp(common, r_a, tl_a) - np.interp(common, r_b, tl_b)
    finite = np.isfinite(residual)
    return (float(np.sqrt(np.mean(residual[finite] ** 2)))
            if finite.any() else float('nan'))


def _range_curve(curve, label: str, fname: str):
    """A ``(ranges, tl_dB)`` pair as two float64 1-D arrays of one length,
    the ranges finite and strictly increasing."""
    try:
        ranges, tl = curve
        ranges = np.asarray(ranges, dtype=float)
        tl = np.asarray(tl, dtype=float)
    except (TypeError, ValueError):
        raise ConfigurationError(
            f"{fname}: {label} must be a Field or a (ranges, tl_dB) pair of "
            f"1-D arrays; got {type(curve).__name__}.") from None
    if ranges.ndim != 1 or tl.shape != ranges.shape or ranges.size == 0:
        raise ConfigurationError(
            f"{fname}: {label}'s ranges and tl_dB must be 1-D arrays of one "
            f"non-zero length; got shapes {ranges.shape} and {tl.shape}.")
    if not (np.all(np.isfinite(ranges)) and np.all(np.diff(ranges) > 0)):
        raise ConfigurationError(
            f"{fname}: {label}'s ranges must be finite and strictly "
            f"increasing (m).")
    return ranges, tl


def tl_rmse_on_shared_ranges(field, reference, *,
                             depth: Optional[float] = None) -> float:
    """RMS dB difference at one depth, over the ranges both fields reach,
    **resampling** onto the coarser of the two axes.

    The companion to :func:`tl_rmse`, which requires the two fields to
    already share a grid and raises otherwise. That refusal is right when
    the question is "do these two runs agree" — a silent interpolation
    would hide a grid mismatch that matters. It is the wrong answer when
    the question is "how far apart are these two models", because models
    run on different range axes by nature, and that comparison is the
    whole point of the pairwise table
    :func:`~uacpy.visualization.plots.plot_field_statistics` draws.

    Two functions rather than a flag on one: the refusal and the
    resampling answer different questions, and a caller who picks the
    wrong one should be picking a name, not a keyword.

    The common grid is the **coarser** axis — the larger median spacing —
    clipped to the shared span. ``np.interp`` reproduces a node exactly, NaN
    included, so an already aligned pair is untouched; the clip keeps
    ``np.interp``'s flat extrapolation past the ends of its own domain out of
    the number. No-energy cells become NaN first and drop out with them.

    Each field is read at its own depth sample nearest ``depth``; when the
    two nearest samples differ, a ``FallbackWarning`` names both, because near a
    null TL changes tens of dB over metres and the residual would then be a
    depth offset rather than a model disagreement. A ``depth`` more than half
    a depth step outside a field's depth axis is refused: its nearest sample
    would be the axis end, a depth nobody asked for.

    The same computation takes two ``(ranges, tl_dB)`` curves — a
    measured TL curve, a table from another code — with no Field built:
    ``tl_rmse_on_shared_ranges((r_meas, tl_meas), (r_model, tl_model))``.

    Parameters
    ----------
    field, reference : Field or (ranges, tl_dB) pair
        TL-like fields with ``ranges`` and a ``dB`` view; or two pairs of
        1-D arrays, ranges in metres (finite, strictly increasing) and TL in
        dB. Both Fields or both pairs.
    depth : float
        The receiver depth to compare at, on both fields (m). Required for
        Fields; a pair is one depth's curve already and takes none.

    Returns
    -------
    float
        RMSE in dB, or ``nan`` when the two share no range at all — which
        is not the same as agreeing.

    Raises
    ------
    ConfigurationError
        The two fields are different physical quantities (``kind``), so
        their difference is not an agreement metric, a real field is not
        in dB, a field has an axis besides depth and range (the
        ``source_depth`` axis of a multi-depth run: pass one source,
        ``field.at(source_depth=…)``, or the sum, ``field.superpose()``), or
        ``depth`` lies outside a field's depth axis.
    """
    fname = 'tl_rmse_on_shared_ranges'
    are_fields = [isinstance(f, Field) for f in (field, reference)]
    if not any(are_fields):
        if depth is not None:
            raise ConfigurationError(
                f"{fname}: depth= selects a depth of a Field; a (ranges, "
                f"tl_dB) pair is one depth's curve already.",
                remediation="Drop depth=, or pass two Fields.")
        r_a, tl_a = _range_curve(field, 'field', fname)
        r_b, tl_b = _range_curve(reference, 'reference', fname)
        return _rmse_on_coarser_shared_axis(r_a, tl_a, r_b, tl_b, fname)
    if not all(are_fields):
        raise ConfigurationError(
            f"{fname}: field and reference must both be Fields or both be "
            f"(ranges, tl_dB) pairs.")
    if depth is None:
        raise ConfigurationError(
            f"{fname}: two Fields are compared at one depth; pass depth= "
            f"(m).")
    for label, f in (('field', field), ('reference', reference)):
        _refuse_axes_beyond_tl(f, label, fname)
        if not f.is_complex and f.unit != 'dB':
            raise ConfigurationError(
                f"{fname}: {label} is in {f.unit!r}, not dB, so its values "
                f"are not a level and their difference is not a TL error.",
                remediation="Compare two TL (or complex pressure) fields.")
    if field.kind != reference.kind:
        raise ConfigurationError(
            f"tl_rmse_on_shared_ranges: a {field.kind!r} field and a "
            f"{reference.kind!r} field are different physical quantities, "
            f"so their RMS difference is not an agreement metric.")
    for label, f in (('field', field), ('reference', reference)):
        z = np.asarray(f.coords.get('depth', ()), dtype=float)
        if z.size < 2:
            continue
        half_step = 0.5 * float(np.median(np.abs(np.diff(z))))
        if not (z.min() - half_step <= depth <= z.max() + half_step):
            raise ConfigurationError(
                f"{fname}: depth={depth:g} m is outside {label}'s depth axis "
                f"[{z.min():g}, {z.max():g}] m, so its nearest sample would "
                f"be {z[np.argmin(np.abs(z - depth))]:g} m.",
                remediation="Pass a depth inside both fields' depth axes "
                            "(in metres).")
    cut_a, cut_b = field.at(depth=depth), reference.at(depth=depth)
    z_a = float(cut_a.pinned.get('depth', depth))
    z_b = float(cut_b.pinned.get('depth', depth))
    if not np.isclose(z_a, z_b, rtol=1e-6, atol=1e-6):
        warnings.warn(
            f"{fname}: depth={depth:g} m lands on {z_a:g} m in `field` and "
            f"{z_b:g} m in `reference` (each field's nearest sample); the RMS "
            f"compares those two depths. Put both on a common depth grid "
            f"(Field.resample_to) to compare one depth.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
    return _rmse_on_coarser_shared_axis(
        np.asarray(field.ranges, dtype=float),
        np.asarray(cut_a.dB, dtype=float),
        np.asarray(reference.ranges, dtype=float),
        np.asarray(cut_b.dB, dtype=float), fname)


__all__ = ["tl_rmse", "tl_max_error", "tl_bias",
           "tl_rmse_on_shared_ranges"]
