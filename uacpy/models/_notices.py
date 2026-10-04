"""How a run states what it will do: the :class:`Notice` a stage-3 resolver
returns (what the settings record, and what the run warns), the sink a
resolver gives its notices to (:func:`give_notice`), and the notices a run
states once — the lossless-water notice, which only the run the caller made
gives, and the licence notice of a restricted engine, given once per
process."""

import contextvars
import warnings

import numpy as np

from uacpy.core.absorption import Thorp
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.exceptions import ValidityWarning
from uacpy.core.run_settings import Notice


# Source ``id``s already warned about this process, so a licence-restricted
# engine (OASES) emits its one-time ProvenanceWarning once, not per instance.
_WARNED_MODEL_PROVENANCE: set = set()


#: dB of volume absorption that may go unmentioned. Below a decibel over the
#: whole track the omission cannot change a level anyone acts on, and warning
#: about it would only teach users to ignore the notice.
_ABSORPTION_NOTICE_DB = 1.0

#: True while a :meth:`PropagationModel.run` call is in progress in this
#: context (thread or task). Set once the call's own arguments are checked,
#: so a ``run()`` or ``run_settings()`` started inside it (a model a hook
#: runs) knows it is not the call the user made.
_INSIDE_RUN: contextvars.ContextVar = contextvars.ContextVar(
    'uacpy_inside_run', default=False)


def _warn_if_volume_absorption_is_missing(env, source, receiver) -> None:
    """Say so when a run is about to propagate through lossless water.

    The Acoustics Toolbox adds volume attenuation only when asked:
    ``misc/AttenMod.f90:35-38`` makes the second attenuation-unit letter
    (``T`` Thorp, ``F`` Francois-Garrison, ``B`` biological) the one that
    adds it, and the ``SELECT CASE`` at ``:84`` has no default branch. So
    ``Environment(absorption=None)`` is lossless water in every model, not
    just in one — which is the right default (it is what the analytic
    benchmarks compare against, and it keeps a uacpy run reproducing the
    engine's own answer for the same deck) but is easy to leave in place by
    accident. At 40 kHz over a kilometre Thorp puts it at 12.9 dB, comparable
    to the whole bottom-loss budget of such a link.

    Thorp is the yardstick because it takes no parameters, so the size of
    what is being dropped can be estimated without inventing a water column.
    """
    if env.absorption is not None:
        return
    # Said once, for the run the caller asked for. A run or run_settings
    # started inside it (a model one of its hooks runs) calls back with a
    # frequency or range of its own, and a notice from it would quote a band
    # the caller never asked for. Such a call happens inside the outermost
    # ``run()``, which :data:`_INSIDE_RUN` marks, so it is skipped here. The
    # producers the engines run themselves (Bellhop's BOUNCE route, the OASES
    # mean field) check their calls with ``announce=False`` and never reach
    # this notice. Nothing is written
    # to the caller's Environment: the next run on it, or on a copy of it, is
    # a new run and is told again.
    if _INSIDE_RUN.get():
        return
    frequencies = np.atleast_1d(np.asarray(
        getattr(source, 'frequencies', ()), dtype=float))
    ranges = np.atleast_1d(np.asarray(
        getattr(receiver, 'ranges', ()), dtype=float)) if receiver is not None \
        else np.array([])
    if not frequencies.size or not ranges.size:
        return
    freq_max = float(np.max(frequencies))
    r_max = float(np.max(np.abs(ranges)))
    if not (np.isfinite(freq_max) and np.isfinite(r_max)) or freq_max <= 0.0 \
            or r_max <= 0.0:
        return
    alpha = float(np.atleast_1d(Thorp().alpha_dB_per_m(freq_max, 0.0))[0])
    omitted = alpha * r_max
    if not np.isfinite(omitted) or omitted < _ABSORPTION_NOTICE_DB:
        return
    # "Valid at frequencies below 50 kHz": Etter, Underwater Acoustic
    # Modeling and Simulation, on absorption — which also puts the field
    # measurements these laws rest on at 20 Hz-60 kHz.
    warnings.warn(
        f"env.absorption is None, so the water column is lossless: this run "
        f"drops about {omitted:.1f} dB of volume absorption at "
        f"{freq_max:g} Hz over {r_max:g} m (Thorp's estimate). That is "
        f"deliberate for a benchmark against a lossless solution, and wrong "
        f"for anything meant to be realistic — pass absorption=Thorp() (no "
        f"parameters, valid below 50 kHz) or "
        f"absorption=FrancoisGarrison(temperature=..., salinity=..., "
        f"pH=...) for the general case.",
        ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP)



def message_notice(message, category):
    """``message`` as a :class:`~uacpy.core.run_settings.Notice` of
    ``category`` with no note; ``None`` (no notice) stays ``None``."""
    return None if message is None else Notice(None, message, category)


def give_notice(sink, message, category, *,
                skip_file_prefixes=USER_FRAME_SKIP) -> None:
    """Say ``message`` about the run being resolved, as a ``category``
    warning (a :class:`~uacpy.core.exceptions.UACPYWarning` subclass):
    appended to ``sink``, the list stage 3 collects a run's notices in (its
    settings then record them and the run announces them), or, with no sink
    (a resolver called on its own), a warning at once."""
    if sink is None:
        warnings.warn(message, category,
                      skip_file_prefixes=skip_file_prefixes)
    else:
        sink.append(Notice(None, message, category))
