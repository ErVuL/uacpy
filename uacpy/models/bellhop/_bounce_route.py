"""The route of a seabed through BOUNCE: whether a call takes it, the
:class:`~uacpy.models.bounce.Bounce` producer that runs it, the settings and
notices of that run, and the inputs and output of its launch, whose ``.brc``
table the Bellhop deck then carries in place of the seabed."""

import copy
from pathlib import Path
from typing import NamedTuple, Optional

from uacpy.core.environment import Bottom, BoundaryProperties, Environment
from uacpy.core.run_settings import RunMode
from uacpy.core.surface import Surface
from uacpy.models.base import PropagationModel, StageInputs
from uacpy.models.bellhop._plan import launch_source
from uacpy.core.exceptions import FallbackWarning, NumericsWarning
from uacpy.models._notices import message_notice


class _BounceKnobs(NamedTuple):
    """The BOUNCE constructor knobs a routed call passes on; ``None`` leaves
    each to :class:`~uacpy.models.bounce.Bounce`'s own default."""
    c_low: Optional[float] = None
    c_high: Optional[float] = None
    rmax_m: Optional[float] = None


class _BounceTable(NamedTuple):
    """What the BOUNCE launch of a routed call returns: the table as Bounce's
    own ``run`` returns it, and the ``.brc`` the Bellhop deck reads."""
    result: object
    brc_file: Path


class _BounceLaunch(NamedTuple):
    """What the BOUNCE launch of a routed call reads: the producer, and the
    environment it tabulates, projected onto what BOUNCE reads."""
    producer: PropagationModel
    env: Environment


def bounce_route(env, request, *, auto_bounce):
    """``(origin, knobs)`` when this call's seabed goes through BOUNCE,
    else ``None``: ``'run_with_bounce'`` with the knobs that call passed
    (``request.engine_request``; ``request`` is the checked call, or
    ``None``), or ``'auto_bounce'`` for a layered bottom (sediment layers
    anywhere along range), which Bellhop's single-halfspace ``.env`` cannot
    carry. A non-layered bottom — elastic included — stays on the deck:
    ``bellhop.f90:694-712`` computes the exact acousto-elastic halfspace
    reflection coefficient natively (per range node on a range-dependent
    bottom), which a BOUNCE pass would only degrade by collapsing the range
    axis to one column."""
    knobs = None if request is None else request.engine_request
    if knobs is not None:
        return 'run_with_bounce', knobs
    if auto_bounce and env.bottom.is_layered:
        return 'auto_bounce', _BounceKnobs()
    return None


def bounce_route_notices(env, mode, fc, bounce_origin, *, model_name):
    """The stage-3 notices of a seabed routed through BOUNCE: a band whose
    every bin reuses the one reflection table computed at ``fc``, and a
    layered bottom routed there by ``auto_bounce`` (``bounce_origin``)."""
    notices = []
    if (mode in (RunMode.BROADBAND, RunMode.TIME_SERIES)
            and env.bottom.is_layered):
        # BOUNCE tabulates R(theta) at the one frequency it is
        # handed (fc). A layer stack's R(theta) changes with
        # frequency (its nulls move with f*sin(theta)), so every bin
        # of the band then carries fc's reflection loss.
        notices.append(
            message_notice(f"{model_name}: {mode.name} over a layered bottom "
                        f"reuses one BOUNCE reflection table computed at fc = "
                        f"{fc:g} Hz for every frequency of the band, but a "
                        f"layered seabed's R(theta) changes with frequency. "
                        f"Levels away from fc carry fc's reflection loss; run "
                        f"narrower sub-bands at several fc and stitch them where "
                        f"the band edges matter.", NumericsWarning))
    if bounce_origin == 'auto_bounce':
        kind = ('layered bottom (elastic)' if env.bottom.is_elastic
                else 'layered bottom')
        notices.append(
            message_notice(f"{model_name}: env.bottom is a {kind}; "
                        f"auto-routing through BOUNCE to derive a "
                        f"reflection-coefficient table. BOUNCE is "
                        f"range-independent — Bounce's collapse policy reduces "
                        f"the env (default: bottom_range='median', layer stack "
                        f"kept). Pass ``Bellhop(auto_bounce=False)`` to skip the "
                        f"auto-route (Bellhop will then collapse the layer stack "
                        f"to a halfspace via its own collapse policy).",
                        FallbackWarning))
    return notices


def bounce_producer(knobs: _BounceKnobs, *, verbose, user_collapse,
                    timeout, cleanup):
    """The :class:`~uacpy.models.bounce.Bounce` a routed call runs, with
    the route's knobs and the calling model's collapse policy
    (``user_collapse``), verbosity, timeout and ``cleanup``. Its stage
    hooks run in the calling model's work directory, so its table
    records its file paths exactly when that model's results do."""
    from uacpy.models.bounce import Bounce
    return Bounce(
        verbose=verbose,
        c_low=knobs.c_low,
        c_high=knobs.c_high,
        rmax_m=knobs.rmax_m,
        collapse=dict(user_collapse) or None,
        timeout=timeout,
        cleanup=cleanup,
    )


def bounce_view(env):
    """``env`` as BOUNCE is handed it: without its surface and
    altimetry, which BOUNCE never reads — its deck omits the water
    column and puts the water's seafloor sound speed in the top
    half-space (``oalib_writer.write_bounce_input_file``) — so it has
    nothing of them to drop."""
    view = env.copy()
    view.altimetry = None
    view.surface = Surface.coerce(None)
    return view


def resolve_bounce_settings(env, source, receiver, *, producer,
                            model_name):
    """``(settings, notices)`` of the BOUNCE run a routed call makes:
    the stages 1-3 of ``producer`` (the
    :class:`~uacpy.models.bounce.Bounce` of the route) for its
    REFLECTION mode on :func:`bounce_view`, and the notices of Bounce's
    projection, which the calling model's
    :meth:`~uacpy.models.base.PropagationModel._announce_engine_settings`
    says once per call.

    BOUNCE tabulates one seabed column against the water sound speed at
    one seafloor depth, so Bounce's notices about collapsing a
    range-dependent bathymetry or SSP describe the table only; a notice
    of Bellhop's says so first, since the ray trace keeps both as
    given."""
    view = bounce_view(env)
    settings, projection_notices = producer._producer_settings(
        view, source.at_depth(0), receiver, RunMode.REFLECTION)
    kept = [what for what, flag in (
        ('bathymetry', view.bathymetry.varies_with_range),
        ('SSP', view.ssp.is_range_dependent)) if flag]
    notices = []
    if kept and projection_notices:
        what = ' and '.join(kept)
        notices.append(
            message_notice(f"{model_name}: BOUNCE tabulates the seabed reflection "
                        f"coefficient against the water sound speed at one seafloor "
                        f"depth, so Bounce's notices about collapsing the "
                        f"range-dependent {what} describe that table only; the ray "
                        f"trace keeps the {what} as given.", FallbackWarning))
    notices.extend(message_notice(n, FallbackWarning)
                   for n in projection_notices)
    return settings, notices


def prepare_bounce_launch(env, producer) -> _BounceLaunch:
    """What the BOUNCE launch of a routed seabed reads: ``producer`` and the
    environment it tabulates, :func:`bounce_view` of ``env`` projected as
    Bounce projects it (its notices were said in stage 3)."""
    bounce_env, _ = producer._environment_projection(bounce_view(env))
    return _BounceLaunch(producer=producer, env=bounce_env)


def is_bounce_launch(inputs: StageInputs) -> bool:
    """Whether ``inputs`` are launch 0 of a seabed routed through BOUNCE:
    the table, not the Bellhop run that reads it."""
    return inputs.settings.engine.bounce is not None and inputs.launch == 0


def bounce_inputs(inputs: StageInputs) -> StageInputs:
    """The BOUNCE launch's stage inputs: this call's work directory and
    receiver, the environment :meth:`_prepare_launches` projected for
    BOUNCE, and a single-depth copy of the launch's carrier source."""
    source = launch_source(inputs).at_depth(0)
    return StageInputs(
        work_dir=inputs.work_dir, env=inputs.prepared.env,
        source=source, receiver=inputs.receiver,
        settings=inputs.settings.engine.bounce._replace(
            source_depths=source.depths))


def tabulated_seabed(env, brc_file):
    """A copy of ``env`` whose seabed is the BOUNCE table ``brc_file``
    (``acoustic_type='file'``): the environment the Bellhop deck of a routed
    seabed is written from."""
    env = copy.deepcopy(env)
    env.bottom = Bottom.from_halfspace(BoundaryProperties(
        acoustic_type='file',
        reflection_file=brc_file,
    ))
    return env


def read_bounce_output(inputs: StageInputs, deck: Path) -> _BounceTable:
    """The BOUNCE launch's :class:`ReflectionCoefficient`, checked
    against Bounce's output contract and stamped with its settings, and
    its ``.brc`` (Bounce's reader refuses a run that wrote none)."""
    table = inputs.prepared.producer._run_producer_launch(
        bounce_inputs(inputs), deck)
    return _BounceTable(result=table,
                        brc_file=inputs.work_dir / f'{deck.stem}.brc')
