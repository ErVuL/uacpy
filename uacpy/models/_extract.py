"""Stages 5-6 of a run: the identity every result carries
(:func:`result_kwargs`, :func:`stamp_result`), the output contract it is
held to (:func:`check_output_contract`), the cells no engine can fill
(:func:`mask_source_axis`, :func:`mask_unresolvable_depths`), the output
paths a surviving work directory leaves on it, what it records of how it
was computed (the frequency grid the engine propagated), an axis restored
to the values the call asked for, and the end of every IFFT-based
broadband route (:func:`finish_broadband`)."""

import warnings
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np

from uacpy.core._records import array_summary
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.exceptions import (ModelExecutionError, NumericsWarning,
                                   OutputContractError, ValidityWarning)
from uacpy.core.results import PhaseReference, Result
from uacpy.core.run_settings import RunMode, RunSettings
from uacpy.core.source import Source
from uacpy.models._stacking import _slabs_of


def _settings_as_run(settings: RunSettings, result) -> RunSettings:
    """``settings`` with the frequency grid ``result`` was computed on.

    An engine whose ``_to_result`` records a grid of its own (a lattice of
    its own, a realised bin) propagated a grid other than the one the base
    rules resolved. The result's ``frequencies`` are what the engine used, so
    the stamped settings carry those and a note names what had been
    resolved.
    """
    realised = getattr(result, 'frequencies', None)
    if realised is None:
        return settings
    realised = np.atleast_1d(np.asarray(realised, dtype=float))
    resolved = settings.frequencies
    if resolved is not None and np.array_equal(resolved, realised):
        return settings
    used = array_summary(realised, 'Hz')
    if resolved is None:
        note = f"frequencies: {used}, chosen by {settings.model}'s run"
    else:
        was = array_summary(resolved, 'Hz')
        note = (f"frequencies: {settings.model}'s run propagated {used} in "
                f"place of the resolved {was}")
        if used == was and realised.shape == resolved.shape:
            # Two grids that print alike differ past the summary's digits.
            note += (f" (largest difference "
                     f"{float(np.max(np.abs(realised - resolved))):.3g} Hz)")
    return settings._replace(frequencies=realised,
                            notes=settings.notes + (note,))


def add_run_note(result, note: str) -> None:
    """Hand stage 6 one line on how the run went, for
    ``run_settings.notes``: what the engine decided at run time and wrote to
    its own output (a beam count it chose), or a step the read took (a bin
    solved again on a deck of its own) — what the settings, resolved before
    the launch, cannot hold. :meth:`PropagationModel._stamp_run_settings
    <uacpy.models.base.PropagationModel._stamp_run_settings>` moves it into
    the record it stamps."""
    for slab in _slabs_of(result):
        slab._run_notes = tuple(getattr(slab, '_run_notes', ())) + (note,)


def run_notes_of(slabs) -> tuple:
    """The notes :func:`add_run_note` left on ``slabs``, each once, in
    the order given, taken off the slabs."""
    notes = []
    for slab in slabs:
        for note in slab.__dict__.pop('_run_notes', ()):
            if note not in notes:
                notes.append(note)
    return tuple(notes)


def _restore_requested_axis(read_back: np.ndarray, requested: np.ndarray,
                            rtol: float) -> np.ndarray:
    """``requested`` when the axis an engine's output file gave back,
    ``read_back``, holds the same values to within ``rtol`` times the axis's
    largest magnitude; ``read_back`` otherwise.

    A file carries an axis at the precision its writer used (a band written
    ``%.12g`` into the deck, ranges stored REAL*4), so an axis read back
    differs from the requested one by that rounding alone; restored, the
    result carries the values the call asked for. ``rtol`` is that precision.
    An axis of another length, or one that differs by more, is a different
    grid and comes back as read."""
    read = np.asarray(read_back, dtype=float)
    want = np.asarray(requested, dtype=float)
    if read.shape != want.shape or want.size == 0:
        return read_back
    bound = rtol * float(np.max(np.abs(want)))
    if np.all(np.abs(read - want) <= bound):
        return want.copy()
    return read_back


def attach_output_paths(
    result: 'Result',
    work_dir: Path,
    base_name: str,
    *,
    cleanup: bool,
    primary_files: tuple = (),
) -> None:
    """Attach work-dir output paths to ``result.metadata``.

    The paths are recorded iff the run's scratch survives, i.e. iff
    ``cleanup`` is False. With ``cleanup=True`` the work dir is wiped
    immediately after ``run()`` returns, so no keys are written: the
    absence of a ``*_file`` / ``prt_file`` key is the documented signal
    that the directory has been cleaned up (DOCUMENTATION.md §8).

    ``primary_files`` must name only the outputs *this* run produces —
    passing every suffix a model can emit would re-attach an earlier
    run's leftovers in a pinned work dir.

    Otherwise set ``result.metadata['work_dir']`` to the directory, and
    for each ``(key, suffix)`` in ``primary_files``
    ``result.metadata[key] = str(work_dir / f'{base_name}{suffix}')``
    when the file exists. Also set ``'prt_file'`` from the binary's
    diagnostic log when present.
    """
    if cleanup:
        return
    result.metadata['work_dir'] = str(work_dir)
    for key, suffix in primary_files:
        path = work_dir / f'{base_name}{suffix}'
        if path.exists():
            result.metadata[key] = str(path)
    prt_path = work_dir / f'{base_name}.prt'
    if prt_path.exists():
        result.metadata['prt_file'] = str(prt_path)


def result_kwargs(
    model_name: str,
    provenance,
    source: 'Source',
    *,
    backend: Optional[str] = None,
    frequencies: Optional[Union[float, np.ndarray]] = None,
    phase_reference: Optional[str] = None,
    components: Optional[Mapping[str, Any]] = None,
    **extra,
) -> dict:
    """Pre-built kwargs for any :mod:`uacpy.core.results` constructor.

    ``frequencies`` is auto-wrapped to a 1-D ndarray (length ≥ 1) when
    scalar; every wrapper passes a value (broadband / time-series
    results carry the full frequency grid their synthesis derived,
    SPARC's p(t) the edges ``[freq_min, freq_max]`` of the band it marched),
    so ``None`` —
    passed through unchanged — occurs only for a caller that opts out
    of the stamp. ``components`` (the results this one was built from,
    by :data:`~uacpy.core.results.Result.COMPONENT_NAMES` name) goes to the
    constructor's ``components``; anything in ``extra`` is stored on the
    result's ``metadata`` ad-hoc bag.

    ``backend`` names the engine that ran, and the default lowercases
    the class name rather than mixing conventions (``'scooter'``,
    ``'oast'``, …). A dispatcher that passes its own string keeps the
    binary's spelling (``'mpirams'``; Kraken stamps the stem of the
    executable it ran, ``'kraken'`` / ``'krakenc'``, which a renamed
    binary changes), so a caller comparing ``result.backend`` should fold
    the case.

    Identification only: this stamps who produced the result, never what
    it means numerically. Nothing here — nor anywhere else in
    :class:`PropagationModel` — rescales, normalises or re-references the
    payload, so the amplitude a wrapper stores is whatever its reader
    returned. The one cross-model *convention* the base class carries is
    ``phase_reference`` (see
    :class:`~uacpy.core.results._base.PhaseReference`), which is about
    phase, not level; the absolute dB reference behind ``Field.dB`` is a
    per-model property and is not asserted here.
    """
    kw = dict(
        model=model_name,
        backend=backend or model_name.lower(),
        source_depths=np.atleast_1d(np.asarray(
            getattr(source, 'depths', []), dtype=float
        )),
        frequencies=(np.atleast_1d(np.asarray(frequencies, dtype=float))
                     if frequencies is not None else None),
        model_source=provenance,
        metadata=dict(extra),
    )
    # The drive level rides with the result so ``Field.at_source_level()``
    # needs no argument: it is the Source's, not the plot call's.
    level = getattr(source, 'source_level_dB', None)
    if level is not None:
        kw['source_level_dB'] = float(level)
    if phase_reference is not None:
        # Coerced to the PhaseReference enum member, so wrapper-built
        # results carry the same type as the synthesis helpers stamp.
        kw['phase_reference'] = PhaseReference(phase_reference)
    if components:
        kw['components'] = dict(components)
    return kw


def stamp_result(model_name: str, provenance, result, source: 'Source', *,
                 backend: Optional[str] = None,
                 frequencies: Optional[Union[float, np.ndarray]] = None,
                 phase_reference: Optional[str] = None):
    """Stamp the cross-model identification fields onto a ``Result`` that an
    io reader already constructed (so it couldn't take them as kwargs).
    Mirrors :func:`result_kwargs` for the reader-built path (Scooter,
    SPARC); leaves ``result.metadata`` untouched. Returns ``result``."""
    kw = result_kwargs(model_name, provenance, source, backend=backend,
                             frequencies=frequencies,
                             phase_reference=phase_reference)
    for attr in ('model', 'backend', 'source_depths', 'frequencies',
                 'model_source'):
        setattr(result, attr, kw[attr])
    if result.source_level_dB is None:
        result.source_level_dB = kw.get('source_level_dB')
    if phase_reference is not None:
        result.phase_reference = PhaseReference(phase_reference)
    return result


def mask_unresolvable_depths(model_name: str, result, receiver, media_depth):
    """Restore the caller's depth axis on ``result``, NaN below
    ``media_depth``.

    The finite-element / finite-difference mesh stops at the deepest
    modelled interface, so the binary clamps any deeper receiver onto it.
    Handing those cells back under the depth that was asked for would
    misreport where the field was evaluated — they are no-data.

    Used by the full-waveguide spectral solvers (Scooter, SPARC), which
    mesh through the sediment stack the same way.
    """
    depths = np.atleast_1d(np.asarray(receiver.depths, dtype=float))
    data = np.array(result.data)
    if data.shape[0] != depths.size:
        raise ModelExecutionError(
            model_name, return_code=0, stdout=None,
            stderr=(f"{model_name} returned {data.shape[0]} depth "
                    f"rows for {depths.size} requested receiver depths; "
                    f"the depth axis cannot be reattached."),
        )
    data[depths > media_depth, ...] = np.nan
    # Everything else the field holds carries over, the results it was
    # built from (its components) as the same objects.
    return result.replace(data=data,
                          coords={**result.coords, 'depth': depths})


def mask_zero_range_columns(model_name: str, data, ranges,
                            singular_term: str, *, warn: bool = True):
    """Return ``data`` with the range-axis (axis 1) columns at ``r = 0``
    set to NaN, warning once with the model prefix.

    The zero test mirrors ``abs( Rr ) < realmin`` from ``fieldsco.m:69``;
    :class:`~uacpy.core.receiver.Receiver` rejects negative ranges, so it
    is equivalent to ``r == 0`` for any constructible receiver.
    ``singular_term`` names the 1/√r-type factor that has no value there
    (it completes "... at r = 0, where <singular_term> is singular").
    ``data`` is returned unchanged — and no warning is emitted — when no
    column sits on the source axis. :func:`mask_source_axis` applies it
    to a point-source field; SPARC's 'R'/'D' branches call it with
    their own singular term. Every masked engine warns on every run.
    """
    ranges = np.atleast_1d(np.asarray(ranges, dtype=float))
    zero = np.abs(ranges) < np.finfo(float).tiny
    if not zero.any():
        return data
    if warn:
        warn_zero_range(model_name, int(zero.sum()), singular_term)
    masked = np.array(data, copy=True)
    if not np.issubdtype(masked.dtype, np.inexact):
        masked = masked.astype(float)
    masked[:, zero, ...] = np.nan
    return masked


def warn_zero_range(model_name: str, n_zero: int, singular_term: str) -> None:
    """The ``r = 0`` notice, once per run. A model that masks each slab
    of a native multi-depth stack separately would otherwise repeat one
    binary launch's warning per source depth."""
    warnings.warn(
        f"{model_name}: {n_zero} receiver range(s) at "
        f"r = 0, where {singular_term} is singular; those cells are "
        f"returned as NaN (no data). Move the receiver off the source "
        f"axis (e.g. r = 1 m) to get a field value.",
        ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )


#: The end of a time record measured for a response still ringing there (a
#: fraction of the record), and the level, re the record's peak (-20 dB), at
#: which that end counts as still ringing: uacpy guidance, no source names a
#: threshold.
RECORD_TAIL_FRACTION = 0.1
RECORD_TAIL_LEVEL = 0.1


def _peak_ignoring_nan(data: np.ndarray) -> float:
    """``np.nanmax(data)`` as a float, returning NaN for an all-NaN input.

    ``np.nanmax`` reports that case by raising a RuntimeWarning, and the only
    way to mute one is a change to the warnings filters — a stack that is
    process-global, so the change would also swallow warnings other threads
    raise while it holds. Masking the NaNs decides the same thing without
    touching global state. ``where=``/``initial=`` keep ``nanmax``'s treatment
    of the infinities: they are values, not gaps, and propagate to the result.
    """
    present = ~np.isnan(data)
    if not present.any():
        return float('nan')
    return float(np.max(data, where=present, initial=-np.inf))


def record_tail_ratio(times, data) -> Optional[float]:
    """The peak of ``|data|`` over the last :data:`RECORD_TAIL_FRACTION` of
    a time record against its peak over the whole record (time on the last
    axis of ``data``, NaN samples ignored). ``None`` when there is nothing
    to measure: fewer than ten samples, a last axis that is not ``times``,
    or an all-NaN or all-zero record."""
    time = np.asarray(times, dtype=float)
    magnitude = np.abs(np.asarray(data))
    if (time.size < 10 or magnitude.ndim == 0
            or magnitude.shape[-1] != time.size):
        return None
    n_tail = max(1, int(round(RECORD_TAIL_FRACTION * time.size)))
    peak = _peak_ignoring_nan(magnitude)
    tail = _peak_ignoring_nan(magnitude[..., -n_tail:])
    if not (np.isfinite(peak) and np.isfinite(tail) and peak > 0.0):
        return None
    return tail / peak


def warn_if_circular_record_rings(model_name: str, result, *,
                                  late_arrival=None,
                                  range_span=None,
                                  record_edge=None) -> None:
    """Warn, once, when a TIME_SERIES trace synthesised by inverse FFT has
    folded: when it is still ringing at the end of its record
    (:func:`record_tail_ratio` at or above :data:`RECORD_TAIL_LEVEL` on any
    slab), when ``late_arrival`` says the record ends before an arrival,
    when ``range_span`` says the receivers' range span outlasts the record,
    or when ``record_edge`` says an auto-placed record holds energy at an
    edge.

    The record is one period of the inverse FFT of a transfer function
    sampled every Δf, so it is ``1/Δf`` long and circular: the response
    after its end lands back on its early part (JKPS §8.2.1.2, eqs.
    (8.9)-(8.11): sampling in frequency makes the time response periodic
    in ``T = 1/Δf``, and the last sum of (8.11) is that wrap-around). A
    record whose end
    still rings is the precondition of that fold; one whose end has decayed
    holds the response, unless a late isolated echo falls past it, which
    only the arrivals can show. ``late_arrival`` and ``range_span`` are the
    synthesis's own findings of that kind
    (:func:`~uacpy.core.results._field_synthesis._synthesize_time_series`'s
    ``fold_report``): ``(start, end, arrival, speed, range)``, the record
    ending before ``range/speed``, the arrival at the slowest stated water
    speed; ``(span_m, span_s, record_s)``, the receivers' range span and its
    travel time against the record; ``(level_dB, edge, start, end)``, the
    energy at the record's ``'end'`` or ``'start'`` in dB re the trace
    maximum. One condition, one warning: the measured ringing with the
    findings named, or the findings alone."""
    worst = None
    for slab in _slabs_of(result):
        time = np.asarray(slab.coords.get('time', ()), dtype=float)
        ratio = record_tail_ratio(time, slab.data)
        if ratio is not None and (worst is None or ratio > worst[0]):
            worst = (ratio, float(time[0]), float(time[-1]))
    findings = []
    if late_arrival is not None:
        start, end, arrival, speed, range_m = late_arrival
        findings.append(
            f"the [{start:.4g}, {end:.4g}] s record ends before r/c = "
            f"{arrival:.4g} s, the arrival at the slowest water speed "
            f"({speed:g} m/s) at {range_m:.0f} m")
    if record_edge is not None:
        level_dB, edge, start, end = record_edge
        findings.append(
            f"the [{start:.4g}, {end:.4g}] s record holds energy at its "
            f"{edge}, {level_dB:.1f} dB re the trace maximum, so it cuts "
            f"through the arrival train")
    longest = None
    if range_span is not None:
        span_m, longest, record_s = range_span
        findings.append(
            f"the receiver range span ({span_m:.0f} m ≈ {longest:.2f}s of "
            f"travel time) exceeds the {record_s:.2f}s record, so far-range "
            f"arrivals wrap back into early bins")
    at_least = (f", at least {longest:.2f}s for the range span"
                if longest is not None else "")
    remedy = (f"Lengthen the record (output_duration={at_least}, or a "
              f"frequencies= grid with a finer step: the record is 1/Δf) or "
              f"place it with t_start=.")
    if longest is not None:
        remedy += (" Or run BROADBAND and give each range its own record: "
                   "Field.synthesize_time_series(..., t_start='reduced').")
    if worst is not None and worst[0] >= RECORD_TAIL_LEVEL:
        ratio, t0, t1 = worst
        also = "".join(f" And {finding}." for finding in findings)
        message = (
            f"{model_name}.run(run_mode=TIME_SERIES): p(t) is still at "
            f"{100.0 * ratio:.0f}% of its peak over the last "
            f"{100.0 * RECORD_TAIL_FRACTION:.0f}% of the [{t0:.4g}, "
            f"{t1:.4g}] s record. The record is one period of an inverse "
            f"FFT, so the response after {t1:.4g} s has folded back onto its "
            f"early part, where it reads as extra early arrivals.{also} "
            f"{remedy}")
    elif findings:
        message = (
            f"{model_name}.run(run_mode=TIME_SERIES): "
            f"{'; and '.join(findings)}: the later paths fold onto the "
            f"record's start, where they read as early arrivals. {remedy}")
    else:
        return
    warnings.warn(message, NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)


def mask_source_axis(model_name: str, field, source, *, warn: bool = True):
    """NaN the ``r = 0`` column of a point-source ``field`` and warn once.

    Every engine that evaluates ``1/sqrt(r)`` cylindrical spreading
    returns a number on the source axis that belongs to no range —
    Kraken's field.exe (``EvaluateMod.f90:71-73`` skips the factor under
    a ``TINY`` test), OASES's asymptotic Hankel carrier (measured on a
    100 m Pekeris case: OASP |p| = 1.23 at r = 0 against 5e-3 at 500 m),
    RAM's PE (``psi/sqrt(r)``, scaled at a substitute range) — so each
    wrapper hands its assembled field here and the family's grids stay
    comparable cell by cell. ``'line'`` and ``'scaled'`` sources carry
    no ``1/sqrt(r)`` and are returned as they are. Column masking and
    the warning come from :func:`mask_zero_range_columns`, reading the
    field's own range axis.
    """
    if source.source_type != 'point':
        return field
    masked = mask_zero_range_columns(
        model_name, field.data, field.coords['range'],
        'the point-source cylindrical-spreading factor 1/sqrt(r)',
        warn=warn)
    if masked is not field.data:
        field.data = masked
    return field


def check_output_contract(model_name: str, result,
                          settings: RunSettings) -> None:
    """Raise :class:`~uacpy.core.exceptions.OutputContractError` when
    ``result`` (every slab of a stack) is
    not what ``settings.output`` declares: another result class, or a
    ``kind`` / ``unit`` / ``phase_reference`` other than a declared one.
    A mismatch is a defect in the engine's ``_to_result``, never in
    the call. ``coherent`` is not checked: stage 6 assigns the declared
    one (``PropagationModel._stamp_run_settings``)."""
    spec = settings.output
    if spec is None:
        return
    for slab in _slabs_of(result):
        problems = []
        if type(slab).__name__ != spec.result_type:
            problems.append(f"result class {type(slab).__name__}, "
                            f"declared {spec.result_type}")
        for name in ('kind', 'unit', 'phase_reference'):
            declared = getattr(spec, name)
            if declared is None:
                continue
            actual = getattr(slab, name, None)
            if name == 'phase_reference' and actual is not None:
                actual = PhaseReference(actual).value
            if actual != declared:
                problems.append(f"{name} {actual!r}, declared "
                                f"{declared!r}")
        if problems:
            raise OutputContractError(
                f"{model_name}._to_result broke the output contract "
                f"of {settings.mode.name}: {'; '.join(problems)}.")


def finish_broadband(model_name: str, result, settings: RunSettings):
    """Return the BROADBAND transfer function ``result`` of engine
    ``model_name`` as computed or, for TIME_SERIES, its synthesis with the
    pulse of ``settings.time`` (validated in stage 2,
    :meth:`~uacpy.models.base.PropagationModel._check_time_series_keywords`).
    Every IFFT-based wrapper ends its broadband route here.

    The transfer function carries ``settings`` while it is synthesised,
    so the synthesis reads the waveguide the run resolved
    (``run_settings.waveguide``) as every Field does; stage 6 stamps
    the returned result. The synthesised record is circular, so a trace
    still ringing at its end, or a record ending before the slowest
    arrival, warns once (:func:`warn_if_circular_record_rings`), whoever
    chose the record.
    """
    if settings.mode != RunMode.TIME_SERIES:
        return result
    for slab in _slabs_of(result):
        slab._run_settings = settings
    time = settings.time
    # Field.synthesize_time_series, its refusals and notices naming the
    # run the user called. A record that ends before the slowest
    # arrival, a range span longer than the record, energy at an
    # auto-placed record's edge and a record still ringing at its end
    # are one condition, a fold: the synthesis reports the first three
    # and the ringing check says them as one warning.
    fold_report: dict = {}
    trace = result._synthesized(
        time.source_waveform, time.sample_rate, t_start=time.t_start,
        window=None, placement_checks=True,
        who=f"{model_name}.run", fold_report=fold_report)
    warn_if_circular_record_rings(
        model_name, trace,
        late_arrival=fold_report.get('late_arrival'),
        range_span=fold_report.get('range_span'),
        record_edge=fold_report.get('record_edge'))
    return trace
