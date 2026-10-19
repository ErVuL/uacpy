"""Quantitative agreement metrics between TL fields.

Stand-alone helpers used by tests, examples, and end-user comparison
scripts. Keeps numeric-comparison logic out of plotting and IO modules.

Public helpers: :func:`tl_rmse`, :func:`tl_max_error`, :func:`tl_bias`,
:func:`tl_rmse_on_shared_ranges`, :func:`tl_difference` (the difference map
as a Field), :func:`field_statistics` (per-field statistics and the pairwise
RMS table of several fields at one depth) and :func:`same_grid` (the one
"same grid" rule they all apply). Every one takes ``(field, reference)``
(``field_statistics`` a set of fields) and every signed difference is
``field - reference``.
:func:`tl_rmse_on_shared_ranges` takes two Fields, read at one depth or
already cut there, or two ``(ranges, tl_dB)`` curves (a measurement, another
code's output).

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

from dataclasses import dataclass
from typing import Optional, Tuple

import warnings

import numpy as np

from uacpy.core.acoustics.levels import no_energy_mask
from uacpy.core.results.field import Field
from uacpy.core.exceptions import (ConfigurationError, FallbackWarning,
                                   ValidityWarning)
from uacpy.core._export import ExportRecord
from uacpy.core._plotting import plotter
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


#: Fraction of an axis' own smallest cell within which two samples are one
#: position (uacpy guidance): the sub-millimetre rounding two engines leave on
#: one requested receiver grid agrees, while a shift of a whole fraction of a
#: cell does not, whatever the coordinate's magnitude.
_GRID_AGREEMENT_FRACTION = 1e-3


def _axes_agree(a, b) -> bool:
    """Whether two coordinate axes sample the same positions: one length,
    and every pair of samples within :data:`_GRID_AGREEMENT_FRACTION` of the
    axis' smallest cell. A length-1 axis has no cell and compares with
    numpy's default tolerance."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return False
    if a.size < 2:
        return bool(np.allclose(a, b))
    cell = float(np.min(np.abs(np.diff(a))))
    return bool(np.allclose(a, b, rtol=0.0,
                            atol=_GRID_AGREEMENT_FRACTION * cell))


def same_grid(field, reference) -> bool:
    """Whether two Fields sample the same positions: the same axes in the
    same order, each of one length, and every coordinate within a thousandth
    of that axis' smallest cell of the other's (uacpy guidance). A 0.4 mm
    offset on a 40 m range grid is one grid; a 5 cm one is not.

    The one rule the package applies: :func:`tl_rmse`, :func:`tl_max_error`,
    :func:`tl_bias` and :func:`tl_difference` refuse a pair it rejects, and
    :func:`uacpy.plot.compare_models` warns. Two fields on different grids
    are compared with :func:`tl_rmse_on_shared_ranges`, or after
    ``Field.resample_to``.

    Parameters
    ----------
    field, reference : Field
        The two fields.

    Returns
    -------
    bool
    """
    for label, f in (('field', field), ('reference', reference)):
        if not isinstance(f, Field):
            raise ConfigurationError(
                f"same_grid: {label} must be a Field; got "
                f"{type(f).__name__}.")
    if list(field.coords) != list(reference.coords):
        return False
    return all(_axes_agree(field.coords[name], reference.coords[name])
               for name in field.coords)


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


def _refuse_unlike_tl_fields(field, reference, fname: str) -> None:
    """Refuse two Fields that are not one TL quantity: an axis besides depth
    and range, two different ``kind``s, or a real field not in dB. The one
    check every Field form of the metrics makes, in one order."""
    for label, f in (('field', field), ('reference', reference)):
        _refuse_axes_beyond_tl(f, label, fname)
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
    _refuse_unlike_tl_fields(field, reference, fname)
    if list(field.coords) != list(reference.coords):
        raise ConfigurationError(
            f"{fname}: field is on axes {list(field.coords)} but reference "
            f"is on {list(reference.coords)}; the cells do not correspond.")

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
        if not _axes_agree(values, values_ref):
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
    axes must be one grid by :func:`same_grid` — every coordinate within a
    thousandth of its axis' smallest cell of the other's (models interpolate
    onto the requested receiver grid, so two runs of the same grid match
    within it); grids differing by more raise — resample one onto the other
    first, or use :func:`tl_rmse_on_shared_ranges`.

    Parameters
    ----------
    field, reference : Field or array_like
        Two TL Fields (complex pressure or real dB, same ``kind``), or two
        real dB arrays of one shape. A Field on a frequency or time axis
        raises; a Field and an array together raise.
    range_window : (float, float), optional
        ``(rmin, rmax)`` (m) inclusive. Defaults to all ranges. Fields only.
    depth_window : (float, float), optional
        ``(zmin, zmax)`` (m) inclusive. Defaults to all depths. Fields only.

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
        ``(rmin, rmax)`` (m) inclusive; Fields only.
    depth_window : (float, float), optional
        ``(zmin, zmax)`` (m) inclusive; Fields only.
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
        ``(rmin, rmax)`` (m) inclusive; Fields only.
    depth_window : (float, float), optional
        ``(zmin, zmax)`` (m) inclusive; Fields only.
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
    whole point of the pairwise table :func:`field_statistics` returns.

    Two functions rather than a flag on one: the refusal and the
    resampling answer different questions, and a caller who picks the
    wrong one should be picking a name, not a keyword.

    The common grid is the **coarser** axis — the larger median spacing —
    clipped to the shared span. ``np.interp`` reproduces a node exactly, NaN
    included, so an already aligned pair is untouched; the clip keeps
    ``np.interp``'s flat extrapolation past the ends of its own domain out of
    the number. No-energy cells become NaN first and drop out with them.

    A Field on a depth axis is read at its own depth sample nearest
    ``depth``; a range cut (``field.at(depth=z)``) is used as it is, at the
    depth it was read at. When the two depths differ, a ``FallbackWarning``
    names both, because near a null TL changes tens of dB over metres and the
    residual would then be a depth offset rather than a model disagreement. A
    ``depth`` more than half a depth step outside a field's depth axis is
    refused: its nearest sample would be the axis end, a depth nobody asked
    for.

    The same computation takes two ``(ranges, tl_dB)`` curves — a
    measured TL curve, a table from another code — with no Field built:
    ``tl_rmse_on_shared_ranges((r_meas, tl_meas), (r_model, tl_model))``.

    Parameters
    ----------
    field, reference : Field or (ranges, tl_dB) pair
        TL-like Fields on a range axis — a ``(depth, range)`` grid or a
        range cut — with a ``dB`` view; or two pairs of 1-D arrays, ranges
        in metres (finite, strictly increasing) and TL in dB. Both Fields or
        both pairs.
    depth : float
        The receiver depth to compare at (m). Required when either Field has
        a depth axis; two range cuts and two pairs are one depth's curve
        already and take none.

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
        in dB, a Field has no range axis, a field has an axis besides depth
        and range (the
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
    _refuse_unlike_tl_fields(field, reference, fname)
    for label, f in (('field', field), ('reference', reference)):
        if 'range' not in f.coords:
            raise ConfigurationError(
                f"{fname}: {label} has no range axis (coords "
                f"{list(f.coords)}); the RMS is taken over range.",
                remediation="Pass a (depth, range) Field with depth=, or a "
                            "range cut such as field.at(depth=z).")
    with_depth = ['depth' in f.coords for f in (field, reference)]
    if any(with_depth) and depth is None:
        raise ConfigurationError(
            f"{fname}: a Field on a depth axis is compared at one depth; "
            f"pass depth= (m).")
    if not any(with_depth) and depth is not None:
        raise ConfigurationError(
            f"{fname}: depth= selects a depth of a Field; two range cuts "
            f"are one depth's curve already.",
            remediation="Drop depth=, or pass the (depth, range) Fields.")
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
    # A range cut is used as it is: the depth it was read at, when it
    # records one, is the depth it compares.
    cut_a, cut_b = (f.at(depth=depth) if 'depth' in f.coords else f
                    for f in (field, reference))
    z_a = cut_a.pinned.get('depth', depth)
    z_b = cut_b.pinned.get('depth', depth)
    if (z_a is not None and z_b is not None
            and not np.isclose(float(z_a), float(z_b), rtol=1e-6, atol=1e-6)):
        warnings.warn(
            f"{fname}: the RMS compares {float(z_a):g} m in `field` and "
            f"{float(z_b):g} m in `reference` (each field's nearest sample "
            f"to the depth asked, or the depth its range cut was read at). "
            f"Put both on a common depth grid (Field.resample_to) to compare "
            f"one depth.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
    return _rmse_on_coarser_shared_axis(
        np.asarray(cut_a.coords['range'], dtype=float),
        np.asarray(cut_a.dB, dtype=float),
        np.asarray(cut_b.coords['range'], dtype=float),
        np.asarray(cut_b.dB, dtype=float), fname)


def tl_difference(field, reference) -> Field:
    """The signed difference ``field - reference`` in dB, cell by cell, as a
    Field: the difference map of two models.

    Same inputs and cell rules as :func:`tl_rmse`: two Fields of one quantity
    (``kind``) on one grid (:func:`same_grid`), on ``depth`` and/or ``range``
    axes, read through their ``dB`` view. A cell either field reports as
    carrying no energy (the 600 dB marker) holds no level, so its difference
    is NaN, as the metrics leave it out; a NaN cell of either field is NaN.
    The map shows those cells blank, so no warning counts them (the scalar
    metrics warn because their one number hides the cells it left out).

    Parameters
    ----------
    field, reference : Field
        The two fields; positive values are where ``field`` reads higher —
        more loss for TL, a louder cell for a level.

    Returns
    -------
    Field
        ``kind='difference'``, ``unit='dB'``, on ``field``'s axes and pinned
        coordinates. ``.plot()`` draws it with
        :func:`uacpy.plot.plot_field_difference` (a diverging map
        symmetric about zero).
    """
    fname = 'tl_difference'
    for label, f in (('field', field), ('reference', reference)):
        if not isinstance(f, Field):
            raise ConfigurationError(
                f"{fname}: field and reference must both be Fields; {label} "
                f"is a {type(f).__name__}.",
                remediation="For two dB arrays of one shape, subtract them: "
                            "tl_field - tl_reference.")
    tl_field, tl_reference, _region = _tl_fields(field, reference, None,
                                                 None, fname)
    marker = no_energy_mask(tl_field) | no_energy_mask(tl_reference)
    return Field(data=np.where(marker, np.nan, tl_field - tl_reference),
                 coords=dict(field.coords), pinned=dict(field.pinned or {}),
                 kind='difference', unit='dB')


@dataclass(frozen=True, eq=False)
class FieldStatistics(ExportRecord):
    """Several fields compared at one depth: each field's mean and standard
    deviation of its dB view along range, and the pairwise RMS difference —
    what :func:`field_statistics` returns.

    ``.plot()`` draws it with :func:`uacpy.plot.plot_field_statistics`;
    ``to_dict`` / ``to_xarray`` export it.

    Attributes
    ----------
    labels : tuple of str
        One name per field, in order.
    mean_dB, std_dB : ndarray, shape ``(n,)``
        Mean and standard deviation (dB) of each field's range cut, over
        its finite cells that carry energy.
    rms_dB : ndarray, shape ``(n, n)``
        ``rms_dB[i, j]`` is :func:`tl_rmse_on_shared_ranges` of field ``i``
        against field ``j`` (dB); ``0`` on the diagonal; NaN for a pair
        sharing no range or of two different quantities.
    depth : float
        The depth (m) every field was cut at.
    kinds : tuple of str
        Each field's quantity (``Field.kind``), so the values are read as
        what they are (``'pressure'``: transmission loss).
    """

    labels: Tuple[str, ...]
    mean_dB: np.ndarray
    std_dB: np.ndarray
    rms_dB: np.ndarray
    depth: float
    kinds: Tuple[str, ...]

    _ARRAY_FIELDS = ('mean_dB', 'std_dB', 'rms_dB')
    _XARRAY_FIELDS = {'mean_dB': 'mean_dB', 'std_dB': 'std_dB',
                      'rms_dB': 'rms_dB'}

    def __post_init__(self):
        object.__setattr__(self, 'labels', tuple(str(s) for s in self.labels))
        object.__setattr__(self, 'kinds', tuple(str(s) for s in self.kinds))
        object.__setattr__(self, 'depth', float(self.depth))
        for name in self._ARRAY_FIELDS:
            object.__setattr__(self, name,
                               np.asarray(getattr(self, name), dtype=float))
        n = len(self.labels)
        if (self.mean_dB.shape != (n,) or self.std_dB.shape != (n,)
                or self.rms_dB.shape != (n, n) or len(self.kinds) != n):
            raise ConfigurationError(
                f"FieldStatistics: {n} labels need mean_dB and std_dB of "
                f"shape ({n},), rms_dB of shape ({n}, {n}) and {n} kinds; "
                f"got {self.mean_dB.shape}, {self.std_dB.shape}, "
                f"{self.rms_dB.shape} and {len(self.kinds)}.")
        super().__post_init__()

    def _payload(self):
        return {'mean_dB': (self.mean_dB, ('field',), 'dB'),
                'std_dB': (self.std_dB, ('field',), 'dB'),
                'rms_dB': (self.rms_dB, ('field', 'other_field'), 'dB')}

    def _coords(self):
        return {'field': (np.asarray(self.labels), ''),
                'other_field': (np.asarray(self.labels), '')}

    def plot(self, **kwargs):
        """Draw these statistics with
        :func:`uacpy.plot.plot_field_statistics`; ``kwargs`` reach it.
        Returns ``(fig, axes)``."""
        return plotter('plot_field_statistics')(self, **kwargs)


def field_statistics(fields, labels=None, *, depth: float) -> FieldStatistics:
    """Several fields compared at one depth: each one's mean and standard
    deviation along range, and the pairwise RMS difference between them.

    Each field is cut at ``depth`` (``field.at(depth=depth)``); the mean
    and standard deviation are over the cut's finite cells that carry
    energy (a no-energy cell, the 600 dB marker, is no level). Each
    off-diagonal entry is :func:`tl_rmse_on_shared_ranges` — over the ranges
    both fields computed, on the coarser range axis — and is NaN for a pair
    sharing no range. A pair of two different quantities (``kind``) has no
    agreement number: its entry is NaN, with one ``ValidityWarning`` per
    pair.

    Parameters
    ----------
    fields : sequence of Field, or {label: Field} dict
        The fields; a ``None`` entry (a model that did not run) is dropped
        with its label.
    labels : sequence of str, optional
        One name per field; a dict's keys when it is one; otherwise each
        field's ``model`` (``#i`` without one), numbered where two share it.
    depth : float
        The receiver depth (m) every field is cut at.

    Returns
    -------
    FieldStatistics
        ``.plot()`` draws the two panels.
    """
    fname = 'field_statistics'
    if isinstance(fields, dict):
        if labels is None:
            labels = list(fields.keys())
        fields = list(fields.values())
    fields = list(fields)
    if labels is not None and len(labels) != len(fields):
        raise ConfigurationError(
            f"{fname}: got {len(labels)} labels for {len(fields)} fields; "
            f"one label per field.")
    if labels is None:
        labels = [getattr(f, 'model', '') or f"#{i}"
                  for i, f in enumerate(fields)]
        for name in set(labels):
            group = [i for i, label in enumerate(labels) if label == name]
            if len(group) > 1:
                for n, i in enumerate(group, start=1):
                    labels[i] = f"{name} ({n})"
    kept = [(str(lbl), f) for lbl, f in zip(labels, fields) if f is not None]
    if not kept:
        raise ConfigurationError(
            f"{fname}: no fields to compare — pass one Field per model, as a "
            f"list or as a {{label: field}} dict.",
            remediation="Pass at least one Field; entries that are None are "
                        "dropped as models that did not run.")
    for name, f in kept:
        if not isinstance(f, Field):
            raise ConfigurationError(
                f"{fname}: {name!r} is a {type(f).__name__}, not a Field.")
    means, stds = [], []
    for name, f in kept:
        cut = f.at(depth=depth)
        spread = [axis for axis, values in cut.coords.items()
                  if np.size(values) > 1 and axis != 'range']
        if spread:
            raise ConfigurationError(
                f"{fname}: {name!r} cut at depth={depth:g} m spans {spread}; "
                f"the statistics are over range only.",
                remediation=f"Select one value of each first, e.g. "
                            f"field.at({spread[0]}=...).")
        level = np.asarray(cut.dB, dtype=float)
        level = level[np.isfinite(level) & ~no_energy_mask(level)]
        means.append(float(np.mean(level)) if level.size else float('nan'))
        stds.append(float(np.std(level)) if level.size else float('nan'))
    n = len(kept)
    rms = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            (name_i, f_i), (name_j, f_j) = kept[i], kept[j]
            if f_i.kind != f_j.kind:
                warnings.warn(
                    f"{fname}: {name_i!r} is a {f_i.kind!r} field and "
                    f"{name_j!r} a {f_j.kind!r} field, different physical "
                    f"quantities, so their RMS difference is not an "
                    f"agreement metric; that entry is NaN.",
                    ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP)
                rms[i, j] = rms[j, i] = np.nan
                continue
            rms[i, j] = rms[j, i] = tl_rmse_on_shared_ranges(
                f_i, f_j, depth=depth)
    return FieldStatistics(labels=tuple(name for name, _ in kept),
                           mean_dB=np.array(means), std_dB=np.array(stds),
                           rms_dB=rms, depth=float(depth),
                           kinds=tuple(f.kind for _, f in kept))


__all__ = ["tl_rmse", "tl_max_error", "tl_bias",
           "tl_rmse_on_shared_ranges", "tl_difference", "same_grid",
           "field_statistics", "FieldStatistics"]
