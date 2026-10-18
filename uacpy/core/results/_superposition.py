"""The fields of several sources as one :class:`~uacpy.core.results.Field`:
the per-source fields stacked on a ``source_depth`` axis
(:func:`stack_fields`, public; :func:`_stack_fields` names its caller in a
refusal), split back (:func:`split_source_depths`), their sum
(:func:`superpose_field`, what :meth:`Field.superpose` runs), and the weight
checks a run and that sum share."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Tuple

import numpy as np

from uacpy.core.acoustics.levels import sum_levels_dB
from uacpy.core.constants import PRESSURE_FLOOR
from uacpy.core.exceptions import ConfigurationError, FallbackWarning
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core._validate import real_array_or_none
from uacpy.core.results import quantities as _quantities
from uacpy.core.results._base import PhaseReference


#: The kinds whose values are the power behind a loss or a level, so that
#: mutually incoherent sources add them in intensity: a propagation loss and
#: OASS reverberation (losses), a received level and a sound exposure
#: (levels). A residual, a probability, a signal excess, a peak, a
#: normalised ambiguity power and an impulse response do not add this way.
_INTENSITY_SUM_KINDS = ('pressure', 'reverberation', 'level', 'sound_exposure')

#: The axis the fields of several sources are stacked on.
SOURCE_AXIS = 'source_depth'


def _dB_weight_refusal(unit, *, where: str,
                       sources: bool = False) -> ConfigurationError:
    """The refusal of a weight on a real frequency-domain field of
    ``unit`` (dB): its phase is gone, so no weight scales it. ``sources``
    marks the coherent sum of a field's several sources, which can still
    add in intensity, and names that sum."""
    return ConfigurationError(
        f"{where}: the field is real {unit!r} values, not "
        f"complex pressure, so its phase is gone and a weight cannot "
        f"scale it (multiplying dB is not a coherent sum). Weight the "
        f"complex field the run returned (before to_dB(), or a mode "
        f"that keeps phase — phase_reference is not None); for the "
        f"level of one source at magnitude |w|, subtract "
        f"20*log10(|w|) from the dB view yourself.",
        **({'remediation': "Add the sources in intensity, which needs no "
                           "phase: field.superpose(coherent=False)."}
           if sources else {})
    )


def _check_field_weightable(field, weights, *, where: str,
                            sources: bool = False) -> None:
    """Raise :class:`ConfigurationError` unless ``weights`` can scale
    ``field``. The one rule behind the n = 1 weight
    :meth:`PropagationModel.run` applies and the coherent sum
    :meth:`Field.superpose` forms:

    * a real frequency-domain field (dB) has lost its phase and takes no
      weight at all — multiplying decibels is not a scaling;
    * a real time-domain trace takes a real weight (a sign flip is -1),
      never a complex one;
    * complex data takes a complex weight only when it carries a
      ``phase_reference``. Complex data without one — a field built by
      hand, say — states no phase convention, so its phase cannot be told
      from an artefact of how it was stored; rotating it would record a
      source phase the field cannot carry. A real weight still scales such
      a field's level.

    ``sources`` marks the sum of a field's several sources, whose dB
    refusal names the intensity sum (:func:`_dB_weight_refusal`).
    """
    w = np.atleast_1d(np.asarray(weights, dtype=np.complex128))
    time_domain = 'time' in field.coords
    if not time_domain and not field.is_complex:
        raise _dB_weight_refusal(field.unit, where=where, sources=sources)
    if not field.is_complex and np.any(w.imag != 0.0):
        raise ConfigurationError(
            f"{where}: the data are real time-domain traces, which a "
            f"complex weight cannot scale; give a real weight (a sign flip "
            f"is -1), or run BROADBAND and weight the transfer function "
            f"before synthesising."
        )
    if (field.is_complex and field.phase_reference is None
            and np.any(w.imag != 0.0)):
        raise ConfigurationError(
            f"{where}: the data are complex but carry no phase reference, "
            f"so nothing says their phase is a propagation phase rather "
            f"than an artefact of how it was stored. Rotating it by a "
            f"complex weight would record a source phase the field cannot "
            f"carry; give a real weight to scale its level, or build the "
            f"field with a phase_reference (every coherent model mode "
            f"stamps one)."
        )


def _check_sources_weightable(field, weights, *, where: str) -> None:
    """The weaker check the per-source fields of a run have to pass:
    refuse only what no sum of them could ever use.

    The fields are not weighted when the run returns them —
    :meth:`Field.superpose` applies the weights, and the caller has not yet
    said which sum they mean. A dB-only field cannot add coherently but adds
    perfectly well in intensity, so refusing it here would close the only
    route to "N mutually incoherent sources at these levels". What is
    unusable either way is a complex weight on data with no phase to
    rotate; that is refused now, while the rest waits for
    :meth:`Field.superpose` to judge against the sum actually asked for.
    """
    w = np.atleast_1d(np.asarray(weights, dtype=np.complex128))
    if np.any(w.imag != 0.0):
        _check_field_weightable(field, w, where=where)


def _weighted_sum(field, planes, weights, *, where: str,
                  sources: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """``(Σ wᵢ·planesᵢ, w)``: the coherent weighted sum of the arrays
    ``planes`` (one per source) of ``field``, and the weights as applied —
    real for real data, complex otherwise — after
    :func:`_check_field_weightable` on ``field``. The one sum behind
    :meth:`Field.superpose` and the n = 1 weight :meth:`PropagationModel.run`
    applies to a one-depth ``Source``; ``sources`` marks the former."""
    w = np.atleast_1d(np.asarray(weights, dtype=np.complex128))
    _check_field_weightable(field, w, where=where, sources=sources)
    if not field.is_complex:
        w = w.real
    first = np.asarray(planes[0])
    total = np.zeros(first.shape, dtype=np.result_type(first.dtype, w.dtype))
    for wi, plane in zip(w, planes):
        total += wi * plane
    return total, w


def superpose_field(field, weights=None, *, coherent: bool = True,
                    who: str = 'Field.superpose'):
    """The sum over the ``source_depth`` axis of ``field``: see
    :meth:`Field.superpose`, which this implements. ``who`` names the
    caller in a refusal."""
    if SOURCE_AXIS not in field.coords:
        raise ConfigurationError(
            f"{who}: this field has no {SOURCE_AXIS!r} axis (its axes are "
            f"{list(field.coords)}), so it holds one source and there is "
            f"nothing to sum.",
            remediation="Run a Source with several depths "
                        "(Source(depths=[...], weights=[...])): its field "
                        "carries one plane per source depth on a leading "
                        "'source_depth' axis. A one-depth Source takes its "
                        "weight in the run itself.")
    depths = np.asarray(field.coords[SOURCE_AXIS], dtype=float).copy()
    n = depths.size
    if weights is None:
        weights = field.source_weights
    if weights is None:
        weights = np.ones(n)
    w = np.atleast_1d(np.asarray(weights, dtype=np.complex128))
    if w.ndim != 1 or w.size != n:
        raise ConfigurationError(
            f"{who}: {n} source depths but {w.size} weight(s) (shape "
            f"{w.shape}); give one weight per source depth.")
    if not np.all(np.isfinite(w)):
        bad = int(np.flatnonzero(~np.isfinite(w))[0])
        raise ConfigurationError(
            f"{who}: weights must be finite; weights[{bad}] = {w[bad]}.")
    axis = list(field.coords).index(SOURCE_AXIS)
    planes = np.moveaxis(np.asarray(field.data), axis, 0)
    coords = {k: v for k, v in field.coords.items() if k != SOURCE_AXIS}
    id_kwargs = field.id_kwargs()
    id_kwargs['source_depths'] = depths
    id_kwargs['source_weights'] = None
    if coherent:
        total, w = _weighted_sum(field, planes, w, where=who, sources=True)
        return field.replace(
            data=total, coords=coords,
            superposed_sources={'depths': depths.tolist(),
                                'weights': w.tolist(), 'coherent': True},
            **id_kwargs)

    # An intensity sum: √Σ|wᵢ·pᵢ|² as a real dB field. It reads magnitudes
    # only, so a dB-only field adds as well as a complex one.
    if 'time' in field.coords or 'time' in field.pinned:
        raise ConfigurationError(
            f"{who}(coherent=False): the field holds time-domain traces or "
            f"snapshots of one, and intensity does not add sample by "
            f"sample. Sum the traces coherently, or superpose the BROADBAND "
            f"transfer function and synthesise afterwards.")
    if field.kind not in _INTENSITY_SUM_KINDS:
        raise ConfigurationError(
            f"{who}(coherent=False): the field is a {field.kind!r} field, "
            f"which does not add as intensities; an intensity sum adds the "
            f"power behind a loss or a level "
            f"({', '.join(_INTENSITY_SUM_KINDS)}).",
            remediation="Superpose the propagation field the run returned, "
                        "then derive this quantity from the sum.")
    if np.any(w.imag != 0.0):
        warnings.warn(
            f"{who}(coherent=False): an intensity sum carries no phase, so "
            f"only the weight magnitudes are used "
            f"({np.abs(w).tolist()}); the phases of {w.tolist()} are "
            f"dropped. Pass coherent=True to drive the sources with a fixed "
            f"relative phase.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
    amp = np.abs(w)
    # A loss counts DOWN from the source and a level counts UP, so the
    # stored numbers turn into a level with opposite signs: a stack of
    # levels read as losses sums two incoherent 120 dB sources to 116.99 dB
    # instead of 123.01.
    sign = -1.0 if _quantities.is_loss(field.kind) else 1.0
    with np.errstate(divide='ignore'):
        if field.is_complex or field.unit != 'dB':
            # Complex pressure, or real values in 'Pa' read as linear
            # magnitudes: the level of each source counting up.
            up = 20.0 * np.log10(np.abs(planes))
        else:
            up = sign * np.asarray(planes, dtype=float)
        up = up + (20.0 * np.log10(amp)).reshape((n,) + (1,) * (up.ndim - 1))
    # The power sum of the per-source levels, clamped like every other dB
    # view (``transmission_loss_dB``) so a cell no energy reached reads the
    # package's no-energy marker instead of ``inf``.
    summed = sum_levels_dB(up, axis=0)
    level = sign * np.maximum(summed, 20.0 * np.log10(PRESSURE_FLOOR))
    id_kwargs['phase_reference'] = None
    # An intensity sum is incoherent whatever run mode produced the fields;
    # the run mode it inherits would say otherwise.
    return field.replace(
        data=np.asarray(level, dtype=float), coords=coords, unit='dB',
        coherent=False,
        superposed_sources={'depths': depths.tolist(),
                            'weights': amp.tolist(), 'coherent': False},
        **id_kwargs)


# ── one Field from the fields of several sources ───────────────────────────


def _same_value(a, b) -> bool:
    """Whether two metadata values are the same value: arrays by shape and
    content (NaN equal to NaN), mappings key by key, anything else by
    ``==``."""
    if a is b:
        return True
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same_value(a[k], b[k])
                                            for k in a)
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            x, y = np.asarray(a), np.asarray(b)
        except Exception:
            return False
        if x.shape != y.shape:
            return False
        try:
            return bool(np.array_equal(x, y, equal_nan=True))
        except TypeError:
            return bool(np.array_equal(x, y))
    try:
        return bool(a == b)
    except Exception:
        return False


def _merged_metadata(slabs) -> Dict[str, Any]:
    """The metadata of the stacked field: an entry every slab holds with the
    same value keeps it; any other entry holds one value per slab, in
    stacking order (``None`` where a slab has none) — each source depth's
    own scratch paths, for one."""
    keys: List[str] = []
    for slab in slabs:
        keys.extend(k for k in slab.metadata if k not in keys)
    merged: Dict[str, Any] = {}
    for key in keys:
        values = [slab.metadata.get(key) for slab in slabs]
        held = all(key in slab.metadata for slab in slabs)
        if held and all(_same_value(values[0], v) for v in values[1:]):
            merged[key] = values[0]
        else:
            merged[key] = values
    return merged


def _merged_components(slabs, coordinate, coordinate_name, who):
    """The components of the stacked field: one every slab holds as the
    same object is kept; otherwise the slabs' components are stacked along
    the same axis (Fields into one Field, other results into a
    :class:`~uacpy.core.results.ResultStack`). A Field component stacked
    over the source depths of fields that carry the run's Source weights
    carries them too, so its :meth:`Field.superpose` drives the same
    sources."""
    from uacpy.core.results.field import Field
    from uacpy.core.results.stack import ResultStack
    names: List[str] = []
    for slab in slabs:
        names.extend(k for k in slab.components if k not in names)
    merged = {}
    for name in names:
        values = [slab.components.get(name) for slab in slabs]
        if any(v is None for v in values):
            raise ConfigurationError(
                f"{who}: component {name!r} is held by some of the fields "
                f"and not by others; the fields of one run carry the same "
                f"components.",
                remediation="Stack fields that carry the same components, "
                            "or drop them from every field first: "
                            "field.replace(components=None).")
        if all(v is values[0] for v in values[1:]):
            merged[name] = values[0]
        elif all(isinstance(v, Field) for v in values):
            joined = _stack_fields(values, coordinate,
                                   coordinate_name=coordinate_name, who=who)
            weights = slabs[0].source_weights
            if (coordinate_name == SOURCE_AXIS
                    and joined.source_weights is None
                    and weights is not None
                    and np.size(weights) == np.size(coordinate)):
                joined.source_weights = np.array(weights)
            merged[name] = joined
        else:
            merged[name] = ResultStack(values, coordinate,
                                       coordinate_name=coordinate_name)
    return merged


def _check_labels(slabs, coordinate, coordinate_name: str, who: str) -> None:
    """Refuse a non-finite label, a label used twice (on an axis that
    lists an identity, two within a deck's rounding of each other), and a
    label that contradicts the field it labels: on an axis that lists an
    identity (``source_depth``,
    ``frequency``), a field whose own identity there is one value — a
    one-depth run's ``source_depths``, a one-frequency field's
    ``frequencies``, or that coordinate pinned — is the field of that value,
    so its label must be it (within a deck's rounding,
    :func:`~uacpy.core.results._base.axis_match_tolerance`)."""
    from uacpy.core.results._base import axis_match_tolerance
    from uacpy.core.results.field import _AXIS_IDENTITY_ATTR
    if not np.all(np.isfinite(coordinate)):
        raise ConfigurationError(
            f"{who}: the {coordinate_name} labels {coordinate.tolist()} hold "
            f"a non-finite value; a label is a position on the new axis.",
            remediation="Give each field a finite label.")
    attr = _AXIS_IDENTITY_ATTR.get(coordinate_name)
    # On an identity axis two labels within a deck's rounding of each other
    # are one position, as the identity check below reads them; a sweep's
    # labels, in any unit, repeat only when equal.
    order = np.sort(coordinate)
    repeated = [float(a) for a, b in zip(order[:-1], order[1:])
                if b - a <= (axis_match_tolerance(coordinate, b)
                             if attr is not None else 0.0)]
    if repeated:
        raise ConfigurationError(
            f"{who}: the {coordinate_name} labels "
            f"{coordinate.tolist()} repeat {sorted(set(repeated))}; "
            f"each field is one position on the new axis.",
            remediation="Give each field its own label.")
    if attr is None:
        return
    for i, (slab, label) in enumerate(zip(slabs, coordinate)):
        own = slab.pinned.get(coordinate_name)
        if own is None:
            held = getattr(slab, attr, None)
            if held is None or np.size(held) != 1:
                continue
            own = float(np.asarray(held).ravel()[0])
        if abs(own - label) > axis_match_tolerance(coordinate, label):
            raise ConfigurationError(
                f"{who}: fields[{i}] is the field of {coordinate_name} "
                f"{own:g} but is labelled {label:g}.",
                remediation=f"Label each field with its own {coordinate_name}"
                            f", in the order of the fields: "
                            f"[f.{attr}[0] for f in fields].")


def _check_one_grid(slabs, who: str) -> None:
    """Refuse fields that are not sampled at the same points: one Field
    holds them on one set of axes."""
    first = slabs[0]
    for i, slab in enumerate(slabs[1:], start=1):
        same_axes = list(slab.coords) == list(first.coords) and all(
            np.array_equal(slab.coords[k], first.coords[k])
            for k in first.coords)
        if not same_axes or slab.data.shape != first.data.shape:
            sizes = {k: (first.coords[k].size, slab.coords[k].size)
                     for k in first.coords
                     if k in slab.coords
                     and first.coords[k].size != slab.coords[k].size}
            detail = (
                f"axes {list(first.coords)} vs {list(slab.coords)}"
                if list(slab.coords) != list(first.coords) else
                f"shape {first.data.shape} vs {slab.data.shape}"
                + (f", axis lengths {sizes}" if sizes else
                   "; same lengths, different coordinate values")
            )
            raise ConfigurationError(
                f"{who}: fields[{i}] is on a different grid from fields[0] "
                f"({detail}); one Field holds every field on the same "
                f"points.",
                remediation="Give every source the same receiver grid; a "
                            "TIME_SERIES run gets one time axis from "
                            "run(t_start=..., output_duration=...).")
        same_aux = (slab.aux_coords.keys() == first.aux_coords.keys()
                    and all(slab.aux_coords[k][0] == first.aux_coords[k][0]
                            and np.array_equal(slab.aux_coords[k][1],
                                               first.aux_coords[k][1],
                                               equal_nan=True)
                            for k in first.aux_coords))
        if not same_aux:
            raise ConfigurationError(
                f"{who}: fields[{i}] labels its axes differently from "
                f"fields[0] (aux_coords {sorted(slab.aux_coords)} vs "
                f"{sorted(first.aux_coords)}); one Field carries one set of "
                f"labels.",
                remediation="Give every field the same auxiliary labels, or "
                            "drop them from every field first: "
                            "field.replace(aux_coords={}).")


def _check_one_quantity(slabs, who: str) -> None:
    """Refuse fields that do not hold the same quantity the same way: the
    stacked field has one storage, kind, unit and phase convention."""
    def signature(slab):
        reference = slab.phase_reference
        return {'storage': 'complex' if slab.is_complex else 'real',
                'kind': slab.kind, 'unit': slab.unit,
                'coherent': slab.coherent,
                'reference': slab.reference,
                'reference_unit': slab.reference_unit,
                'phase_reference': (None if reference is None
                                    else PhaseReference(reference).value)}
    first = signature(slabs[0])
    for i, slab in enumerate(slabs[1:], start=1):
        this = signature(slab)
        differ = [k for k in first if this[k] != first[k]]
        if differ:
            detail = ', '.join(f"{k} {first[k]!r} vs {this[k]!r}"
                               for k in differ)
            raise ConfigurationError(
                f"{who}: fields[{i}] holds a different quantity from "
                f"fields[0] ({detail}); one Field is one quantity stored "
                f"one way.",
                remediation="Convert the fields to one form first (e.g. "
                            "to_dB() on every one).")


def _check_one_identity(slabs, coordinate_name: str, who: str) -> None:
    """Refuse fields from different producers, or that disagree on an
    identity attribute other than the one the new axis lists."""
    from uacpy.core.results.field import _AXIS_IDENTITY_ATTR
    varying = _AXIS_IDENTITY_ATTR.get(coordinate_name)
    shared = ['model', 'backend', 'run_mode', 'source_level_dB',
              'source_weights']
    shared += [a for a in ('frequencies', 'source_depths') if a != varying]
    first = slabs[0]
    for attr in shared:
        ref = getattr(first, attr, None)
        for i, slab in enumerate(slabs[1:], start=1):
            if not _same_value(ref, getattr(slab, attr, None)):
                raise ConfigurationError(
                    f"{who}: fields[0].{attr}={ref!r} but "
                    f"fields[{i}].{attr}={getattr(slab, attr, None)!r}; "
                    f"one Field has one {attr} (the new axis is "
                    f"{coordinate_name!r}).",
                    remediation=f"Stack only fields that share {attr}.")
    for i, slab in enumerate(slabs[1:], start=1):
        theirs = {k: v for k, v in slab.pinned.items()
                  if k != coordinate_name}
        ours = {k: v for k, v in first.pinned.items()
                if k != coordinate_name}
        if theirs != ours:
            raise ConfigurationError(
                f"{who}: fields[{i}] is pinned at {theirs} and fields[0] at "
                f"{ours}; one Field records one set of pinned coordinates.",
                remediation="Select every field at the same values first "
                            "(field.at(...) with the same labels), or stack "
                            "only fields pinned alike.")
    # The derivation record, every attribute of it (Field._DERIVATION).
    from uacpy.core.results.field import Field
    for attr in ('speeds', *Field._DERIVATION):
        ref = getattr(first, attr)
        for i, slab in enumerate(slabs[1:], start=1):
            if not _same_value(ref, getattr(slab, attr)):
                raise ConfigurationError(
                    f"{who}: fields[0].{attr}={ref!r} but "
                    f"fields[{i}].{attr}={getattr(slab, attr)!r}; one Field "
                    f"carries one {attr}.",
                    remediation=f"Keep fields of different {attr} as "
                                f"separate Fields and iterate them (a sweep "
                                f"that changes the waveguide's sound speeds: "
                                f"ParallelResult.results).")


#: The axes a stack places last, after the fields' own: the canonical
#: order is ``source_depth → depth → range → frequency`` (or ``time``)
#: (DOCUMENTATION §8), the layout the synthesis and every producer use.
_TRAILING_AXES = ('frequency', 'time')


def stack_fields(fields, coordinate, *,
                 coordinate_name: str = SOURCE_AXIS):
    """One :class:`~uacpy.core.results.Field` holding ``fields`` along a
    new axis: the fields of several separately computed sources as one
    multi-depth field, which :meth:`Field.superpose
    <uacpy.core.results.Field.superpose>` sums.

    The new axis takes its place in the canonical order every producer
    uses, ``source_depth → depth → range → frequency`` (or ``time``): a
    ``frequency`` or ``time`` axis goes last, after the fields' own axes,
    so a stack of single-frequency fields is the ``(depth, range,
    frequency)`` grid the synthesis takes; any other axis goes first.

    It is how a run joins the field of each depth of a multi-depth
    ``Source``, and how the fields of sources run separately (two runs, a
    ``run_parallel`` sweep over source depths) become that one field. The
    fields must lie on one grid, hold one quantity (storage, kind, unit,
    coherence, phase convention) and share their identity — model,
    backend, run mode, source level, frequencies — except the identity
    the new axis lists (``source_depths`` for a ``source_depth`` axis,
    ``frequencies`` for a ``frequency`` axis), which becomes the axis
    labels. A metadata entry the fields disagree on holds one value per
    field; components are kept when every field holds the same object and
    stacked along the axis otherwise. The run record is the fields' shared
    one, the multi-depth record when they are one-depth runs that differ
    only in their source depth, and ``None`` when they ran under different
    settings (each field keeps its own). Nothing is resampled: fields on
    different grids are refused.

    Parameters
    ----------
    fields : sequence of Field
        The fields to join, one per label, each without a
        ``coordinate_name`` axis of its own.
    coordinate : array_like of float
        The label of each field on the new axis, in the order of
        ``fields`` (the source depths in m for ``'source_depth'``).
    coordinate_name : str, keyword-only
        The new axis, ``'source_depth'`` by default; ``'frequency'``
        stacks the fields of several frequencies, any other name a sweep.

    Returns
    -------
    Field
        The fields over a new ``coordinate_name`` axis: last for
        ``frequency`` / ``time``, leading for any other.

    Raises
    ------
    ConfigurationError
        No fields, labels that are not one flat list of real numbers (a
        bool, a complex value or a string is not one), a label
        count that is not one per field, a non-finite label, a label used
        twice (on ``source_depth`` and ``frequency``, two within a deck's
        rounding of each other — 1e-6 of the largest of the axis span, the
        label and 1 — are one), a label
        that contradicts its field's own one value on that
        axis (a 30 m run labelled 60), a field that is not a Field or
        already has the axis, or fields on different grids, of different
        quantities or of different identities.

    Examples
    --------
    >>> import numpy as np
    >>> from uacpy.core.results import Field, stack_fields
    >>> def one_source(z, p):
    ...     return Field(data=np.full((1, 2), p, dtype=complex),
    ...                  coords={'depth': np.array([50.0]),
    ...                          'range': np.array([1000.0, 2000.0])},
    ...                  source_depths=[z], frequencies=100.0)
    >>> pair = stack_fields([one_source(10.0, 1.0), one_source(30.0, 0.5j)],
    ...                     [10.0, 30.0])
    >>> list(pair.coords)
    ['source_depth', 'depth', 'range']
    >>> pair.superpose([1.0, -1.0]).data
    array([[1.-0.5j, 1.-0.5j]])
    """
    return _stack_fields(fields, coordinate, coordinate_name=coordinate_name,
                         who='stack_fields')


def _stack_fields(slabs, coordinate, *, coordinate_name: str = SOURCE_AXIS,
                  who: str):
    """:func:`stack_fields` of ``slabs``, naming ``who`` (the call the user
    made: a model's ``run``, ``ParallelResult.stack``) in a refusal.

    The rules are :func:`stack_fields`'s; besides them, the largest
    ``synthesis_floor`` is kept, so the record a synthesis sizes from it
    holds every field's response, and the run record follows
    :func:`_stacked_run_settings`.
    """
    from uacpy.core.results.field import Field, _AXIS_IDENTITY_ATTR
    slabs = list(slabs)
    if not slabs:
        raise ConfigurationError(f"{who}: no fields to stack.",
                                 remediation="Pass at least one Field.")
    # Real numbers only, by the package's one rule, element by element:
    # read as labels, bools would be 1 and 0 (also among numbers), complex
    # values their real parts.
    labels = real_array_or_none(coordinate)
    if labels is None:
        raise ConfigurationError(
            f"{who}: the {coordinate_name} labels must be real numbers; got "
            f"{coordinate!r}.",
            remediation="Pass one number per field, in the order of the "
                        "fields.")
    coordinate = np.atleast_1d(labels)
    if coordinate.ndim != 1:
        raise ConfigurationError(
            f"{who}: the {coordinate_name} labels must be one 1-D list; got "
            f"shape {coordinate.shape}.",
            remediation="Pass one label per field as a flat list, in the "
                        "order of the fields.")
    if coordinate.size != len(slabs):
        raise ConfigurationError(
            f"{who}: {len(slabs)} field(s) but {coordinate.size} "
            f"{coordinate_name} value(s); give one value per field.",
            remediation="Pass one label per field, in the order of the "
                        "fields.")
    wrong = sorted({type(s).__name__ for s in slabs
                    if not isinstance(s, Field)})
    if wrong:
        raise ConfigurationError(
            f"{who}: only Fields stack into one Field; got {wrong}.",
            remediation="Keep results that are not Fields (rays, arrivals) "
                        "in a ResultStack.")
    first = slabs[0]
    if coordinate_name in first.coords:
        raise ConfigurationError(
            f"{who}: the fields already carry a {coordinate_name!r} axis; "
            f"a second one cannot be added along it.",
            remediation=f"Stack along another coordinate_name, or take one "
                        f"plane of each first: field.at({coordinate_name}=...).")
    _check_labels(slabs, coordinate, coordinate_name, who)
    _check_one_grid(slabs, who)
    _check_one_quantity(slabs, who)
    _check_one_identity(slabs, coordinate_name, who)
    if coordinate_name in _TRAILING_AXES:
        data = np.stack([np.asarray(s.data) for s in slabs], axis=-1)
        coords = {**first.coords, coordinate_name: coordinate}
    else:
        data = np.stack([np.asarray(s.data) for s in slabs], axis=0)
        coords = {coordinate_name: coordinate, **first.coords}
    pinned = {k: v for k, v in first.pinned.items() if k != coordinate_name}
    floors = [s.synthesis_floor for s in slabs
              if s.synthesis_floor is not None]
    changes = dict(
        data=data, coords=coords, pinned=pinned,
        aux_coords=dict(first.aux_coords),
        metadata=_merged_metadata(slabs),
        components=(_merged_components(slabs, coordinate, coordinate_name,
                                       who) or None),
        synthesis_floor=max(floors) if floors else None,
        **_stacked_run_settings(slabs, coordinate, coordinate_name))
    varying = _AXIS_IDENTITY_ATTR.get(coordinate_name)
    if varying is not None:
        changes[varying] = coordinate.copy()
    return first.replace(**changes)


def _stacked_record(slabs, coordinate, coordinate_name: str):
    """The run record of the results ``slabs`` stacked along
    ``coordinate_name`` (a Field's, a :class:`ResultStack`'s):

    * the first slab's, when every slab holds the same record (the planes
      of one run, split and joined again);
    * the whole stack's record, when the records differ only in
      ``source_depths`` and the axis is ``source_depth`` — the one-depth runs
      of one multi-depth run (an OASSP mean-field sub-run per depth, a
      ``run_parallel`` sweep over source depths): ``source_depths`` the
      axis and ``depth_loop`` ``'per_depth'``, what the multi-depth run's
      own record states;
    * ``None`` otherwise — slabs that ran under different settings (a sweep
      over a knob, an environment) have no one record, and any one of them
      would misstate the other planes. Each plane's record stays on the
      result it came from (``ParallelResult.results``).
    """
    records = [slab.run_settings for slab in slabs]
    first = records[0]
    if all(record == first for record in records[1:]):
        return first

    def without_depths(record):
        return {k: v for k, v in record.to_dict().items()
                if k != 'source_depths'}

    if (coordinate_name == SOURCE_AXIS
            and all(record is not None for record in records)
            and all(without_depths(record) == without_depths(first)
                    for record in records[1:])):
        return first._replace(source_depths=coordinate.copy(),
                              depth_loop='per_depth')
    return None


def _stacked_run_settings(slabs, coordinate,
                          coordinate_name: str) -> Dict[str, Any]:
    """The :meth:`Field.replace` changes that give the field stacking
    ``slabs`` along ``coordinate_name`` its run record
    (:func:`_stacked_record`): none when it is the first slab's, else that
    record. Without one record the sound speeds the records stated, which
    every slab shares (:func:`_check_one_identity`), are kept on
    :attr:`Field.speeds`."""
    record = _stacked_record(slabs, coordinate, coordinate_name)
    if record is slabs[0].run_settings:
        return {}
    if record is None:
        return {'run_settings': None, 'speeds': slabs[0].speeds}
    return {'run_settings': record}


def components_at_source_depths(components, depths, *,
                                keeps_axis: bool = False) -> Dict[str, Any]:
    """``components`` narrowed to the source ``depths`` a selection of a
    multi-depth field keeps. A component that runs over source depth — a
    :class:`~uacpy.core.results.ResultStack` over ``source_depth``, a Field
    with a ``source_depth`` axis — keeps its entries at those depths: a
    selection that drops the axis (one depth) gets the entry itself (the
    slab; the plane, its depth pinned), one that ``keeps_axis`` gets the
    stack or field of them. A component that holds no entry
    at a kept depth (an :meth:`Field.eval` between source depths) is dropped,
    since no part of it belongs to the selection. A component no source depth
    singles out (one table every depth shares) is kept as it is."""
    from uacpy.core.results.field import Field
    from uacpy.core.results.stack import ResultStack
    depths = np.atleast_1d(np.asarray(depths, dtype=float))
    out = {}
    for name, component in components.items():
        if isinstance(component, ResultStack) \
                and component.coordinate_name == SOURCE_AXIS:
            axis = component.coordinate
        elif isinstance(component, Field) and SOURCE_AXIS in component.coords:
            axis = component.coords[SOURCE_AXIS]
        else:
            out[name] = component
            continue
        index = []
        for z in depths:
            hit = np.flatnonzero(np.isclose(axis, z, rtol=0.0,
                                            atol=1e-9 * max(1.0, abs(z))))
            if hit.size:
                index.append(int(hit[0]))
        if len(index) != depths.size:
            continue
        one = len(index) == 1 and not keeps_axis
        if isinstance(component, ResultStack):
            out[name] = (component.slabs[index[0]] if one
                         else ResultStack([component.slabs[i] for i in index],
                                          axis[index],
                                          coordinate_name=SOURCE_AXIS))
        elif one:
            out[name] = component.isel(**{SOURCE_AXIS: index[0]})
        else:
            out[name] = component.window(
                **{SOURCE_AXIS: (float(axis[index].min()),
                                 float(axis[index].max()))})
    return out


def split_source_depths(field) -> List[Any]:
    """The field of each source depth of ``field``, in axis order: its
    plane along the ``source_depth`` axis, ``source_depths`` that one
    depth, no ``source_weights``, the axis neither kept nor pinned, its
    components narrowed to that depth (:func:`components_at_source_depths`)
    and its coherence answer kept — what a one-depth run of that source
    returns, for a step that treats each source on its own before
    :func:`restack_source_depths` joins them again. A field without the axis
    is its own one source."""
    if SOURCE_AXIS not in field.coords:
        return [field]
    axis = list(field.coords).index(SOURCE_AXIS)
    coords = {k: v for k, v in field.coords.items() if k != SOURCE_AXIS}
    planes = []
    for i, z in enumerate(field.coords[SOURCE_AXIS]):
        plane = field.replace(
            data=np.take(np.asarray(field.data), i, axis=axis),
            coords=coords, source_depths=np.array([float(z)]),
            source_weights=None,
            components=components_at_source_depths(field.components, [z])
            or None)
        planes.append(field._as_selection(plane))
    return planes


def restack_source_depths(slabs, field, *, who: str):
    """The inverse of :func:`split_source_depths` for ``field``: ``slabs``
    (one per source depth, as it returned them) joined on ``field``'s
    ``source_depth`` axis again, with ``field``'s ``source_weights`` and
    those of each of its components joined with it, or the lone slab of a
    one-source field."""
    if SOURCE_AXIS not in field.coords:
        (slab,) = slabs
        return slab
    joined = _stack_fields(slabs, field.coords[SOURCE_AXIS],
                           coordinate_name=SOURCE_AXIS, who=who)
    if field.source_weights is not None:
        joined.source_weights = field.source_weights.copy()
    for name, component in joined.components.items():
        original = field.components.get(name)
        weights = getattr(original, 'source_weights', None)
        if (component is not original and weights is not None
                and SOURCE_AXIS in getattr(component, 'coords', {})):
            component.source_weights = np.array(weights)
    return joined
