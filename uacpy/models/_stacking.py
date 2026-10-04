"""The source-depth stack of a run: one engine run per source depth
(:func:`run_per_source_depth`), the weight of a one-depth source
(:func:`apply_single_source_weight`), the slabs of a result, the scratch
subdirectory of each depth, the weights stamp, and the common time axis of
time-domain slabs."""

from typing import List

import numpy as np

from uacpy._log import log_message
from uacpy.core.results import Field, ResultStack
from uacpy.core.run_settings import RunSettings
from uacpy.core.results.stack import (_check_stack_weightable,
                                      _dB_weight_refusal,
                                      _weighted_slab_sum)
from uacpy.core.source import Source


def _slabs_of(result):
    """The slabs of a ``ResultStack``, or ``[result]`` for a single result —
    so a wrapper's post-assembly steps (masking, stamping, output paths)
    run per slab whether the engine returned one field or a stack."""
    return result.slabs if isinstance(result, ResultStack) else [result]


def _source_depth_subdirs(depths) -> List[str]:
    """One scratch subdirectory name per source depth,
    ``source_depth_<z>m``. ``<z>`` is the short ``:g`` form (six significant
    digits); when two depths share that form (1000.001 and 1000.004) every
    name takes the full ``repr`` of its depth, so no two depths share a
    directory and no slab's ``*_file`` paths point at another's files."""
    depths = [float(z) for z in np.atleast_1d(depths)]
    names = [f"source_depth_{z:g}m" for z in depths]
    if len(set(names)) < len(names):
        names = [f"source_depth_{z!r}m" for z in depths]
    return names


def _stamp_source_weights(slabs, source: 'Source') -> None:
    """Record ``source.weights`` on every slab of a source-depth stack so
    ``ResultStack.superpose()`` finds them without the ``Source``."""
    for slab in slabs:
        slab.source_weights = source.weights.copy()


def _pad_time_slabs_to_common_axis(slabs):
    """Zero-pad time-domain ``Field`` slabs at the end to the longest time
    axis. An engine that sizes its window from the arrivals (Bellhop's
    delay-and-sum) gives each source depth its own record length while
    every record starts at the same instant and rate, so padding the
    shorter tails with silence puts the slabs on one axis; a cell with no
    data (all NaN) is padded with NaN. Slabs that are not time-domain
    fields, or whose clocks differ, are returned unchanged."""
    if not all(isinstance(s, Field) and 'time' in s.coords
               and s.coords['time'].size > 1 for s in slabs):
        return slabs
    axes = [s.coords['time'] for s in slabs]
    lengths = [t.size for t in axes]
    if len(set(lengths)) == 1:
        return slabs
    t0 = axes[0][0]
    dt = axes[0][1] - axes[0][0]
    for t in axes[1:]:
        if (abs(t[0] - t0) > 1e-9 * max(abs(t0), dt)
                or abs((t[1] - t[0]) - dt) > 1e-9 * dt):
            return slabs
    n_max = max(lengths)
    # The longest record's own axis is the common one: every record starts
    # at the same instant and rate, so the shorter axes are its prefixes.
    common_time = axes[int(np.argmax(lengths))]
    padded = []
    for slab in slabs:
        n = slab.coords['time'].size
        if n == n_max:
            padded.append(slab)
            continue
        axis = list(slab.coords).index('time')
        data = np.asarray(slab.data)
        pad_shape = list(data.shape)
        pad_shape[axis] = n_max - n
        tail = np.zeros(pad_shape, dtype=data.dtype)
        no_data = np.all(np.isnan(data), axis=axis, keepdims=True)
        tail = np.where(no_data, np.nan, tail)
        coords = dict(slab.coords)
        coords['time'] = common_time
        metadata = dict(slab.metadata)
        if 'nt' in metadata:
            metadata['nt'] = n_max
        padded.append(slab.replace(
            data=np.concatenate([data, tail], axis=axis), coords=coords,
            metadata=metadata, aux_coords=None))
    return padded


def run_per_source_depth(source, run_one, *, pad_time_axes: bool,
                         model_name: str, verbose,
                         scratch_subdir) -> ResultStack:
    """One single-depth engine run, ``run_one(single_depth_source)``,
    per ``source.depths`` entry, stacked over ``source_depth``. The same
    env, receiver, mode and keywords go to every depth, and each slab's
    ``source_weights`` records ``source.weights`` for
    :meth:`ResultStack.superpose` to read as its default. A pinned
    ``work_dir`` gets one ``source_depth_<z>m`` subdirectory per depth,
    so the ``*_file`` paths each slab carries stay that slab's. Time-
    domain slabs whose window the engine sized from each depth's own
    arrivals are zero-padded at the end to one common axis
    (``pad_time_axes``, set when the caller gave no
    ``output_duration``), so the stack always superposes.
    ``scratch_subdir(name)`` is the model's redirect of its pinned work
    directory (``PropagationModel._scratch_subdir``); ``model_name`` and
    ``verbose`` name the model in the log and the refusal."""
    n = int(source.depths.size)
    log_message(model_name,
                f"multi-depth Source: {n} single-depth runs, one per "
                f"depth {source.depths.tolist()}",
                verbose=verbose, level='info')
    slabs = []
    subdirs = _source_depth_subdirs(source.depths)
    for i in range(n):
        single = source.at_depth(i)
        with scratch_subdir(subdirs[i]):
            slabs.append(run_one(single))
        # The weights are applied by ``superpose``, not here — but a
        # field they could never scale (a dB-only mode, or a real trace
        # under a complex weight) is refused now rather than after the
        # remaining n-1 runs, so the one-source and many-source paths
        # refuse the same input at the same point.
        if i == 0 and not source.has_unit_weights and isinstance(
                slabs[0], Field):
            _check_stack_weightable(
                slabs[0], source.weights,
                where=f"{model_name}: Source(weights="
                      f"{source.weights.tolist()})")
    if pad_time_axes:
        slabs = _pad_time_slabs_to_common_axis(slabs)
    _stamp_source_weights(slabs, source)
    return ResultStack.from_slabs(slabs, source.depths,
                                  coordinate_name='source_depth')


def apply_single_source_weight(result, source: 'Source', *, model_name: str):
    """A one-depth ``Source`` with a non-unit weight scales its
    ``Field`` by that weight — the ``n = 1`` case of
    :meth:`ResultStack.superpose` — and records it in
    ``metadata['superposed_sources']``. Unit weight leaves the result
    untouched; a complex weight on a real time-domain trace is refused
    for the reason ``superpose`` gives."""
    if not isinstance(result, Field):
        return result
    w = complex(source.weights[0])
    result.data, _ = _weighted_slab_sum(
        [result], [w], where=f"{model_name}: Source(weights={w})")
    result.metadata['superposed_sources'] = {
        'depths': source.depths.tolist(),
        'weights': [w.real if not result.is_complex else w],
    }
    return result


def refuse_a_one_depth_weight_on_dB(source: 'Source',
                                    settings: RunSettings, *,
                                    model_name: str) -> None:
    """Refuse, before the engine runs, the weight of a one-depth ``Source``
    that :func:`apply_single_source_weight` would apply to a field the run
    declares as real dB (``settings.output.unit``), with the reason
    ``superpose`` gives such a field."""
    if (source.depths.size == 1 and settings.source_weights is not None
            and settings.weights_applied and settings.output is not None
            and settings.output.unit == 'dB'):
        raise _dB_weight_refusal(
            unit='dB', where=f"{model_name}: Source(weights="
                        f"{complex(source.weights[0])})")


def run_depth_loop(source, settings: RunSettings, run_one, *,
                   pad_time_axes: bool, model_name: str, verbose,
                   scratch_subdir):
    """The source-depth loop and the weights of stages 4-5 around
    ``run_one(source) -> result``, which launches the engine for the
    ``Source`` it is handed: once per depth into a ``ResultStack``
    (``settings.depth_loop == 'per_depth'``, see
    :func:`run_per_source_depth`), once with the unit copy of a
    weighted one-depth source (:func:`apply_single_source_weight`), or once
    with the whole source. ``model_name``, ``verbose`` and
    ``scratch_subdir`` are the model's (see :func:`run_per_source_depth`)."""
    if settings.depth_loop == 'per_depth':
        return run_per_source_depth(
            source, run_one, pad_time_axes=pad_time_axes,
            model_name=model_name, verbose=verbose,
            scratch_subdir=scratch_subdir)
    n_depths = int(source.depths.size)
    if (n_depths == 1 and settings.source_weights is not None
            and settings.weights_applied):
        # The engine sees a unit source — as it does for every slab of
        # a multi-depth run — and the weight is applied here, once,
        # whatever the engine's launches do with the source they are
        # handed.
        result = run_one(source.at_depth(0))
        return apply_single_source_weight(result, source,
                                          model_name=model_name)
    result = run_one(source)
    # A stack the engine built itself (Bellhop's .shd / .ray / .arr
    # readers, Kraken's and Scooter's TL) carries the same weight stamp —
    # and the same refusal — as a looped one.
    if (n_depths > 1 and isinstance(result, ResultStack)
            and result.coordinate_name == 'source_depth'
            and result.n_slabs == n_depths):
        if settings.source_weights is not None and isinstance(
                result.slabs[0], Field):
            _check_stack_weightable(
                result.slabs[0], source.weights,
                where=f"{model_name}: Source(weights="
                      f"{source.weights.tolist()})")
        _stamp_source_weights(result.slabs, source)
    return result
