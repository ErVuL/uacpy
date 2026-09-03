"""Base class for acoustic propagation models.

:class:`PropagationModel` holds the contracts every wrapper in
:mod:`uacpy.models` inherits. The file reads in this order:

* :class:`RunMode` — the run-mode vocabulary;
* environment-projection policy — :data:`DEFAULT_COLLAPSE`,
  :data:`VALID_COLLAPSE_METHODS`, :data:`_CAPABILITY_FLAGS`, and the
  roughness helpers those drive;
* :class:`ModelSpec` — the declarative per-model manifest, validated at
  class-definition time by ``__init_subclass__``;
* :class:`PropagationModel`, grouped as: construction + spec application,
  run-mode / run-kwarg resolution, broadband and time-series grid
  derivation, work-dir setup and logging, input validation, the
  ``compute_*`` convenience wrappers, executable lookup and subprocess
  launch, environment projection, output-path and ``.prt`` bookkeeping,
  result stamping, depth policy, ``__repr__``;
* module-private introspection helpers behind ``copy`` / ``__repr__``.

Every concrete ``run()`` follows the same recipe, in this order::

    run_mode = self._resolve_run_mode(run_mode)   # + optional-kwarg guards
    env      = self._project_environment(env)     # collapse what we lack
    self.validate_inputs(env, source, receiver, run_mode=run_mode)
    fm = self._setup_file_manager()
    try:      # write the deck, launch the binary, read it back, build Result
        ...
        self._attach_output_paths(result, fm.work_dir, base_name, ...)
    finally:
        fm.finish()          # wipes iff cleanup=True; always releases the claim
"""

import copy as _copy
import errno
import gc
import os
import re
import shutil
import signal
import subprocess
import threading
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Self, Union

import numpy as np

import uacpy._stack  # noqa: F401 — side-effect: raise RLIMIT_STACK
from uacpy.core.environment import Environment, Bathymetry
from uacpy.core.exceptions import (
    ConfigurationError,
    ExecutableNotFoundError,
    InvalidDepthError,
    ModelExecutionError,
    UnsupportedFeatureError,
)
from uacpy.core.receiver import Receiver
from uacpy.core.results import PhaseReference, Result, ResultStack
from uacpy.core.source import Source
from uacpy.io.file_manager import FileManager
from uacpy.io.oalib_reader import read_prt

# ``warnings.warn(..., skip_file_prefixes=USER_FRAME_SKIP)`` reports the first
# frame outside the uacpy library — a warning raised from a nested helper in a
# model, or in an io / core layer a model delegates to, still points at the
# user's ``run()`` / constructor call. Hand-counted ``stacklevel`` cannot do
# that: it breaks the moment a check moves one frame deeper, and collapses
# distinct call sites onto one uacpy line in the warnings module's dedup key.
# The set is defined in ``core`` so io and core layers warn the same way
# without importing a model; :mod:`uacpy.core._warn_frames` carries the rest —
# why ``tests`` / ``examples`` are excluded, and why each directory entry ends
# at a separator. Re-exported here because ``uacpy.models`` imports it from
# ``base``.
from uacpy.core._warn_frames import USER_FRAME_SKIP

_PACKAGE_DIR = Path(__file__).parent.parent

#: How much of a captured child stream to quote back in an error. The
#: binaries echo their whole deck and every progress block to stdout, so only
#: the tail is useful; the abort reason is always last.
_STREAM_TAIL = 2000


def _tail(text: Optional[str], n_chars: int = _STREAM_TAIL) -> Optional[str]:
    """Last ``n_chars`` of a captured child stream, or ``None`` if empty."""
    if not text:
        return None
    return text if len(text) <= n_chars else '…' + text[-n_chars:]


def _stream_block(label: str, text: Optional[str]) -> str:
    """Render a captured child stream as a labelled block, or nothing."""
    trimmed = _tail(text)
    return f"\n\n{label}:\n{trimmed}" if trimmed else ''


def _is_runnable(path: Path) -> bool:
    """Whether ``path`` is a file this process can actually exec.

    ``exists()`` alone also accepts a directory, the text file an unexpanded
    LFS pointer leaves behind, and a build whose exec bit was lost in transit
    — the realistic broken installs. Each of those reaches ``Popen`` and comes
    back as a raw ``PermissionError`` or ``OSError [Errno 8]`` instead of a
    typed install error, so the executability test belongs at resolve time.
    """
    return path.is_file() and os.access(path, os.X_OK)


class RunMode(Enum):
    """
    Standard run modes for acoustic propagation models.

    Models may support a subset of these modes.
    """
    COHERENT_TL = 'coherent_tl'          # Coherent transmission loss
    INCOHERENT_TL = 'incoherent_tl'      # Incoherent (averaged) TL
    # Incoherent beam sum (Bellhop/influence.f90:139-141, same as 'I') with a
    # Lloyd-mirror source directivity folded into the launch amplitude
    # (Bellhop/bellhop.f90:276-278).
    SEMICOHERENT_TL = 'semicoherent_tl'

    RAYS = 'rays'                        # Ray paths only
    EIGENRAYS = 'eigenrays'              # Eigenrays (specific paths)
    ARRIVALS = 'arrivals'                # Arrival structure

    MODES = 'modes'                      # Normal modes (Kraken depth eigenfunctions)

    # Frequency-domain array products. COVARIANCE → C(f, i, j) hydrophone ×
    # hydrophone matrix, declared by two OASES sub-models: OASN builds it from
    # the noise sources, OASS from the reverberant field (REVCOV, product
    # letter 'a'); OASES.for_mode picks between them on reverberation=.
    # REPLICA → OASN alone: Green's-function samples at the array elements per
    # candidate source position. See core/results.Covariance and
    # core/results.Replicas.
    COVARIANCE = 'covariance'
    REPLICA = 'replica'

    # Time-domain pressure p(t) at the receiver(s). Models that compute a
    # broadband transfer function natively (Bellhop, RAM, Scooter,
    # Kraken, OASES) require ``source_waveform=`` + ``sample_rate=``;
    # SPARC computes p(t) directly from its source pulse and ignores them.
    TIME_SERIES = 'time_series'

    # Broadband complex transfer function H(f).
    BROADBAND = 'broadband'

    REFLECTION = 'reflection'            # Plane-wave reflection coefficients (Bounce, OASR)

    # Scattered / reverberant field from rough interfaces (OASS; OASSP emits
    # its scattered field through BROADBAND / TIME_SERIES instead). Unlike
    # every other mode this is a two-stage run: the scattering kernel is a
    # post-processor over a .rhs written by a preceding OAST/OASR mean-field
    # run with option 's'.
    REVERBERATION = 'reverberation'


DEFAULT_COLLAPSE: Dict[str, str] = {
    'bathymetry': 'max',
    'ssp': 'r0',
    'bottom_range': 'r0',
    'bottom_layers': 'halfspace',
    'altimetry': 'drop',
    'surface': 'r0',
    'elastic': 'fluid',
}

# Allowed method strings per collapse key (validated at construction in
# ``PropagationModel.__init__`` so bad values fail loudly rather than deep
# inside a writer at ``run()``-time).
VALID_COLLAPSE_METHODS: Dict[str, frozenset] = {
    'bathymetry':        frozenset({'max', 'median', 'mean', 'min', 'initial'}),
    'ssp':               frozenset({'r0', 'rmax', 'mean', 'median'}),
    'bottom_range':      frozenset({'r0', 'rmax', 'mean', 'median'}),
    'bottom_layers':     frozenset({'halfspace', 'top_layer', 'volume_average'}),
    'altimetry':         frozenset({'drop'}),
    'surface':           frozenset({'r0', 'rmax', 'mean', 'median'}),
    'elastic':           frozenset({'fluid', 'vacuum'}),
}
# Dev invariants on the collapse-policy constants (raise, not assert, so they
# survive `python -O`).
if set(VALID_COLLAPSE_METHODS) != set(DEFAULT_COLLAPSE):
    raise RuntimeError("VALID_COLLAPSE_METHODS keys must match DEFAULT_COLLAPSE keys")
if not all(DEFAULT_COLLAPSE[k] in VALID_COLLAPSE_METHODS[k] for k in DEFAULT_COLLAPSE):
    raise RuntimeError("DEFAULT_COLLAPSE values must satisfy VALID_COLLAPSE_METHODS")


# Source geometries a model may declare via ``ModelSpec.source_types``.
# 'point'  -> AT 'R', cylindrical spreading applied
# 'line'   -> AT 'X', Cartesian spreading
# 'scaled' -> AT 'S', point source with cylindrical spreading removed
VALID_SOURCE_TYPES: frozenset = frozenset({'point', 'line', 'scaled'})

# Fewest frequencies an auto-derived TIME_SERIES grid may carry. Δf is
# 1/waveform-duration, so a short pulse over a narrow band can derive 2-3 bins
# — below what the band-edge taper in ``_ifft_to_trace`` can act on, and too
# few to represent an arrival. Costs one model run per extra bin.
# Odd on purpose: RAM re-parameterises a uniform grid as an (fc, Q, T) sweep
# that is symmetric about a bin, so an odd count round-trips exactly while an
# even one marches a superset (ram.py _resolve_broadband_grid).
_MIN_TIMESERIES_FREQS = 9


def _max_roughness(boundaries) -> float:
    """Largest interfacial sigma over a list of ``BoundaryProperties``."""
    return max(
        (float(b.roughness) for b in boundaries if b is not None),
        default=0.0,
    )


def _smooth_surface(surface):
    """``surface`` with every node's roughness zeroed.

    Writes each node in ``surface.properties`` directly: the ``Surface``
    delegated write reaches every node too, but warns on a multi-node
    surface that the broadcast flattens range dependence — advice aimed at
    users, not at this deliberate all-nodes write. Assigned rather than
    rebuilt: ``dataclasses.replace`` re-runs ``__post_init__``, which
    rejects explicit acoustic parameters on a vacuum / rigid boundary even
    when they are the values it filled in.
    """
    smoothed = _copy.deepcopy(surface)
    for node in smoothed.properties:
        node.roughness = 0.0
    return smoothed


def _smooth_bottom(bottom):
    """``bottom`` with every interface roughness zeroed — layers and half-space."""
    smoothed = _copy.deepcopy(bottom)
    for column in smoothed.columns:
        column.halfspace.roughness = 0.0
        for layer in column.layers:
            layer.roughness = 0.0
    return smoothed


def _bottom_roughness(bottom) -> float:
    """Largest interfacial sigma anywhere in a ``Bottom``.

    A sediment layer's roughness is the interface at its top, so the seafloor
    of a layered column lives on ``layers[0]``, not on the half-space.
    """
    if bottom is None:
        return 0.0
    boundaries = []
    for column in bottom.columns:
        boundaries.append(column.halfspace)
        boundaries.extend(column.layers)
    return _max_roughness(boundaries)


# Capability-flag names a model may advertise. Each maps to a
# ``_supports_<name>`` instance attribute; the question each answers is "does
# this env *shape* — or this source feature — work with this model?".
# Declaring one is a promise: ``_project_environment`` then leaves that
# feature in the env and emits no warning, so the model's deck writer has to
# carry it — nothing downstream re-checks.
# Keep in lockstep with the ``self._supports_*`` block in
# ``PropagationModel.__init__``.
_CAPABILITY_FLAGS: frozenset = frozenset({
    'altimetry',
    'range_dependent_bathymetry',
    'range_dependent_ssp',
    'range_dependent_bottom',
    'layered_bottom',
    'elastic_media',
    'multi_source_depth',
    'source_beam_pattern',
    'rough_surface',
    'rough_bottom',
})


# Source ``id``s already warned about this process, so a licence-restricted
# engine (OASES) emits its one-time UserWarning once, not per instance.
_WARNED_MODEL_SOURCES: set = set()


# Pinned work dirs currently claimed by a running model: resolved path ->
# [owning thread, claim depth]. The models write fixed scratch filenames
# (``bounce.env``, ``field.flp``, ``tl.grid``), so two runs sharing one
# directory overwrite each other's decks and both read back whatever the last
# binary left: two Bounce runs on different bottoms, started from two threads
# on one pinned work_dir, returned bit-identical answers with nothing raised.
# ``run_parallel`` refuses that configuration across its own jobs; this is the
# same refusal for every other way two runs can start at once.
#
# The owner is the Thread OBJECT, not its ident. CPython hands a dead thread's
# ident straight to the next thread started — measured here, two threads in a
# row both got the ident of one that had already exited — so an ident-keyed
# registry reads an unrelated later thread as the owner and waves it through
# the collision this exists to catch.
_PINNED_WORK_DIRS: dict = {}
_PINNED_WORK_DIRS_LOCK = threading.Lock()


def _claim_work_dir(work_dir, model_name: str):
    """Claim a pinned ``work_dir`` for the calling thread.

    Returns the claim to hand to :func:`_release_work_dir`: the resolved path
    plus the thread that took it. Resolving first makes two names for one
    directory (a symlink, ``./out`` vs an absolute path) collide — the same
    rule ``run_parallel``'s pre-check uses.

    Re-entrant per thread: one thread running twice into a directory, in
    sequence or nested, is the sequential reuse a pinned work_dir is *for*, so
    it is counted rather than refused. Only a second *thread* is a collision.
    """
    key = str(Path(work_dir).resolve())
    # current_thread() rather than get_ident(): it is reuse-proof, and it
    # registers a Thread object for a thread created outside Python, which is
    # what the liveness test below needs.
    me = threading.current_thread()
    with _PINNED_WORK_DIRS_LOCK:
        if _take_claim(key, me):
            return key, me

    # Refusal path only, so its cost buys back a directory nobody is using and
    # is charged to nothing else. The manager holding the claim is normally a
    # local of the run that took it and is released the moment that run
    # returns; one caught in a reference cycle instead waits for the cyclic
    # collector, which ordinary allocation churn does not reliably run (still
    # held after 200k allocations). Collect outside the lock — the finalizer
    # this frees calls _release_work_dir, which takes it.
    gc.collect()
    with _PINNED_WORK_DIRS_LOCK:
        if _take_claim(key, me):
            return key, me
        owner = _PINNED_WORK_DIRS[key]
        raise ConfigurationError(
            f"{model_name}: work_dir {str(work_dir)!r} is already in use by "
            f"thread {owner[0].name!r}, which is still running; concurrent "
            "runs would collide in the same scratch directory and return "
            "each other's results.",
            remediation="Give each run its own work_dir, or leave "
                        "work_dir=None to allocate a fresh tempdir per run.",
        )


def _take_claim(key: str, me) -> bool:
    """Record ``me`` as the owner of ``key`` if it is free; call under the lock.

    Free means unclaimed, claimed by ``me`` already (the re-entrant case), or
    claimed by a thread that has since exited. A dead owner cannot be running
    a model, so its claim is stale — it outlived the run only because whatever
    was going to release it never got the chance.
    """
    owner = _PINNED_WORK_DIRS.get(key)
    if owner is None or not owner[0].is_alive():
        _PINNED_WORK_DIRS[key] = [me, 1]
        return True
    if owner[0] is me:
        owner[1] += 1
        return True
    return False


def _release_work_dir(claim) -> None:
    """Give back one claim taken by :func:`_claim_work_dir`.

    The claim carries the thread that took it, because the release does not
    always run there: the finalizer backing it fires wherever the garbage
    collector happens to be, which was measured running on ``MainThread`` for
    a claim taken on a worker. Matching the *calling* thread instead made that
    release a silent no-op and leaked the claim.

    A claim the registry no longer holds for that owner is ignored, so a
    double release (``cleanup_work_dir`` and then the finalizer) cannot free a
    directory a later run has since claimed.
    """
    key, owner_thread = claim
    with _PINNED_WORK_DIRS_LOCK:
        owner = _PINNED_WORK_DIRS.get(key)
        if owner is None or owner[0] is not owner_thread:
            return
        owner[1] -= 1
        if owner[1] <= 0:
            del _PINNED_WORK_DIRS[key]


def _reset_work_dir_claims() -> None:
    """Drop every claim in a freshly forked child.

    Only the forking thread survives a fork, so the parent's claims describe
    runs the child is not doing — and the lock is inherited in whatever state
    it was in, which would deadlock the child if another thread held it. Both
    are reset here. ``run_parallel(start_method='fork')`` is the path that
    reaches this.
    """
    global _PINNED_WORK_DIRS_LOCK
    _PINNED_WORK_DIRS_LOCK = threading.Lock()
    _PINNED_WORK_DIRS.clear()


if hasattr(os, 'register_at_fork'):        # POSIX only
    os.register_at_fork(after_in_child=_reset_work_dir_claims)


@dataclass(frozen=True)
class ModelSpec:
    """Declarative per-model metadata read by :class:`PropagationModel`.

    Consolidates the *static* facts about a model — the run modes it
    emits, the environment shapes it handles natively, its physics-aware
    collapse defaults, and how to locate its binary — into one block the
    base class reads and **validates at class-definition time**, instead
    of scattering them across ``__init__``. The model stays a
    ``PropagationModel`` subclass, so all generic machinery (collapse
    application, validation, file manager, subprocess, ``copy()``) is
    inherited unchanged; the spec only supplies metadata.

    Precedence is preserved: ``collapse`` here layers on top of
    :data:`DEFAULT_COLLAPSE` but never overrides an explicit
    ``Model(collapse={...})`` user value (same rule as
    :meth:`PropagationModel._set_collapse_defaults`). A subclass may still
    set an instance-dependent flag in ``__init__`` *after* ``super().__init__``
    for the rare case a capability depends on a constructor argument.

    Fields
    ------
    modes : sequence of RunMode
        Run modes the model emits. Becomes ``self._supported_modes``; the
        first entry is the default when ``run_mode=None``.
    supports : iterable of str
        Capability-flag names (subset of :data:`_CAPABILITY_FLAGS`) the
        model honours natively. Every flag not listed defaults ``False``
        and its env feature is collapsed by ``_project_environment``.
    source_types : frozenset of str
        Source geometries (subset of :data:`VALID_SOURCE_TYPES`) the model
        honours. Becomes ``self._supported_source_types``; a ``Source``
        carrying anything else is rejected by ``validate_inputs``.
    collapse : dict
        Per-model collapse defaults overriding :data:`DEFAULT_COLLAPSE`.

    Provenance/licence is intentionally *not* here: it is an orthogonal axis
    (who wrote the engine, under what licence) from the run-behaviour fields
    above, and the codebase already keeps dataset provenance in a separate
    catalogue (:mod:`uacpy.data.sources`) rather than folding it into the
    carriers. Models declare it the same way — a ``source`` class attribute
    referencing :data:`uacpy.models.sources.MODEL_SOURCES`.

    Binary resolution is intentionally *not* here either: nothing generic reads it,
    its shape differs per model (single name vs. list of search dirs vs. the
    OASES helper vs. Bellhop's backend dispatch), and multi-binary models
    (Kraken's krakenc, RAM's Collins backends) pick the real executable at
    ``run()`` time. Each model resolves ``self._exe`` in its own ``__init__``.
    """

    modes: tuple = ()
    supports: frozenset = frozenset()
    source_types: frozenset = frozenset({'point'})
    collapse: dict = field(default_factory=dict)

    def validate(self, model_name: str) -> None:
        """Fail loudly at class-definition time on a malformed spec."""
        for m in self.modes:
            if not isinstance(m, RunMode):
                raise TypeError(
                    f"{model_name}.spec.modes must contain RunMode members; "
                    f"got {m!r}."
                )
        bad_flags = set(self.supports) - _CAPABILITY_FLAGS
        if bad_flags:
            raise ConfigurationError(
                f"{model_name}.spec.supports has unknown capability flags: "
                f"{sorted(bad_flags)}. Valid: {sorted(_CAPABILITY_FLAGS)}."
            )
        bad_types = set(self.source_types) - VALID_SOURCE_TYPES
        if bad_types:
            raise ConfigurationError(
                f"{model_name}.spec.source_types has unknown geometries: "
                f"{sorted(bad_types)}. Valid: {sorted(VALID_SOURCE_TYPES)}."
            )
        if not self.source_types:
            raise ConfigurationError(
                f"{model_name}.spec.source_types is empty; every model must "
                f"accept at least one source geometry."
            )
        unknown = set(self.collapse) - set(DEFAULT_COLLAPSE)
        if unknown:
            raise ConfigurationError(
                f"{model_name}.spec.collapse has unknown keys: "
                f"{sorted(unknown)}. Valid keys: {sorted(DEFAULT_COLLAPSE)}."
            )
        for key, value in self.collapse.items():
            if value not in VALID_COLLAPSE_METHODS[key]:
                raise ConfigurationError(
                    f"{model_name}.spec.collapse[{key!r}] = {value!r} is "
                    f"invalid. Valid: {sorted(VALID_COLLAPSE_METHODS[key])}."
                )


#: dB of volume absorption that may go unmentioned. Below a decibel over the
#: whole track the omission cannot change a level anyone acts on, and warning
#: about it would only teach users to ignore the notice.
_ABSORPTION_NOTICE_DB = 1.0


def _line_source_unit_at_1m(c_source: float, frequencies) -> np.ndarray:
    """Factor ``√k0``, ``k0 = 2πf / c(z_s)``, per frequency, that brings an
    engine's 2-D line-source field normalised as ``Σ ψψ e^{ikx} / k_x``
    (Kraken ``KrakenField/EvaluateMod.f90:36``, Scooter's ``'X'`` branch of
    ``TransformG.f90``) to UNIT AMPLITUDE AT 1 m in free space — JKPS
    §5.2.2's ``p/p0(1)`` reference, the same convention the package uses for
    a point source (``TL(1 m) = 0``). JKPS states that no line-source
    normalisation is established; this one is chosen so every engine reports
    one level. Bellhop's raw line field is ``4√π/√R`` (``influence.f90:784``,
    ``ArrMod.f90:104``) and takes ``1/(4√π)`` instead (:data:`Bellhop
    ._LINE_SOURCE_LEVEL`)."""
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    return np.sqrt(2.0 * np.pi * f / float(c_source))


def _source_sound_speed(env, source) -> float:
    """Sound speed at the (first) source depth — the ``c(z_s)`` in the
    line-source reference wavenumber ``k0 = 2πf / c(z_s)``."""
    depth = float(np.atleast_1d(np.asarray(source.depths, dtype=float))[0])
    return float(np.atleast_1d(env.get_sound_speed(depth))[0])


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
    # Say it once for the run the caller asked for. A wrapper re-runs itself
    # internally — a broadband Bellhop re-runs ARRIVALS at its carrier, the
    # routing path spawns a Bounce, a multi-depth eigenray run loops over
    # source depths — and each of those calls back through here with a
    # different frequency or range. Marking the environment the user handed in
    # keeps the second notice, which quotes a band the caller never asked for,
    # from contradicting the first.
    if getattr(env, '_absorption_notice_given', False):
        return
    frequencies = np.atleast_1d(np.asarray(
        getattr(source, 'frequencies', ()), dtype=float))
    ranges = np.atleast_1d(np.asarray(
        getattr(receiver, 'ranges', ()), dtype=float)) if receiver is not None \
        else np.array([])
    if not frequencies.size or not ranges.size:
        return
    f_max = float(np.max(frequencies))
    r_max = float(np.max(np.abs(ranges)))
    if not (np.isfinite(f_max) and np.isfinite(r_max)) or f_max <= 0.0 \
            or r_max <= 0.0:
        return
    from uacpy.core.absorption import Thorp
    alpha = float(np.atleast_1d(Thorp().alpha_db_per_m(f_max, 0.0))[0])
    omitted = alpha * r_max
    if not np.isfinite(omitted) or omitted < _ABSORPTION_NOTICE_DB:
        return
    try:
        env._absorption_notice_given = True
    except AttributeError:          # a carrier that refuses stray attributes
        pass
    # "Valid at frequencies below 50 kHz": Etter, Underwater Acoustic
    # Modeling and Simulation, on absorption — which also puts the field
    # measurements these laws rest on at 20 Hz-60 kHz.
    warnings.warn(
        f"env.absorption is None, so the water column is lossless: this run "
        f"drops about {omitted:.1f} dB of volume absorption at "
        f"{f_max:g} Hz over {r_max:g} m (Thorp's estimate). That is "
        f"deliberate for a benchmark against a lossless solution, and wrong "
        f"for anything meant to be realistic — pass absorption=Thorp() (no "
        f"parameters, valid below 50 kHz) or "
        f"absorption=FrancoisGarrison(temperature_c=..., salinity_psu=..., "
        f"pH=..., z_bar_m=...) for the general case.",
        UserWarning, skip_file_prefixes=USER_FRAME_SKIP)


class PropagationModel(ABC):
    """
    Abstract base class for acoustic propagation models.

    Provides the common interface and shared utilities (subprocess runner,
    executable lookup, input validation, range-dependent handling) for all
    propagation models.

    Parameters
    ----------
    use_tmpfs : bool, optional
        Use a RAM-backed filesystem for I/O. Default is False.
    verbose : bool or str, optional
        Status-output gate. ``False`` (default) prints only ``WARN`` and
        ``ERROR``. ``True`` or ``'info'`` also prints ``INFO``. ``'debug'``
        additionally prints ``DEBUG`` (per-subprocess command lines,
        grid-resolution choices, etc.). See :mod:`uacpy._log`.
    work_dir : str or Path, optional
        Working directory for files. If ``None``, a temporary directory is
        created per run.
    cleanup : bool, optional
        Delete the run's scratch files when ``run()`` returns. ``None``
        (default) resolves to ``work_dir is None``, i.e. uacpy removes only
        directories it created. ``cleanup=False`` with an unpinned
        ``work_dir`` keeps the temp directory (and the ``*_file`` metadata
        paths that point into it); the caller then owns its removal.

    Attributes
    ----------
    model_name : str
        Name of the model (class name).
    use_tmpfs : bool
        Whether tmpfs is used.
    verbose : bool or str
        Verbose-output gate (see constructor).
    """

    #: Whether this wrapper carries ``env.absorption`` through to its engine.
    #: Off by default so a wrapper that ignores volume attenuation cannot
    #: advise a user to set it — RAM models none at all
    #: (``_warn_on_dropped_absorption``) and the OASES family substitutes its
    #: own empirical law instead (``oaseun31.f:1516-1521``), so for those the
    #: advice would contradict the warning they already emit.
    _consumes_volume_absorption: bool = False

    # The leading positional run() parameters every wrapper must carry, in
    # order. Anything a wrapper adds beyond these must be keyword-only (after
    # a bare ``*``) and no wrapper may use ``**kwargs`` — an unknown keyword
    # has to fail with TypeError at the call site, not be silently swallowed.
    _RUN_POSITIONAL = ('self', 'env', 'source', 'receiver', 'run_mode')

    # Declarative metadata. When a subclass declares a :class:`ModelSpec`, the
    # base validates it at class-definition time and applies it in
    # ``__init__``. A subclass without one keeps the base defaults and sets
    # ``_supported_modes`` / ``_supports_*`` / collapse by hand in ``__init__``.
    spec: Optional['ModelSpec'] = None

    # Provenance/licence: ``id`` of this engine's entry in
    # :data:`uacpy.models.sources.MODEL_SOURCES`. Kept off :class:`ModelSpec`
    # (which is about run behaviour) because provenance is an orthogonal axis —
    # the same separation the data layer keeps between its carriers and
    # :mod:`uacpy.data.sources`. Surfaced via :attr:`provenance` / :attr:`citation`;
    # a ``commercial_use=False`` engine (OASES) warns once at construction.
    source: Optional[str] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        spec = cls.__dict__.get('spec')
        if spec is not None:
            if not isinstance(spec, ModelSpec):
                raise TypeError(
                    f"{cls.__name__}.spec must be a ModelSpec, got "
                    f"{type(spec).__name__}."
                )
            spec.validate(cls.__name__)
        if cls.source is not None:
            from uacpy.models.sources import MODEL_SOURCES
            if cls.source not in MODEL_SOURCES:
                raise ConfigurationError(
                    f"{cls.__name__}.source = {cls.source!r} is not a known "
                    f"model source. Valid: {sorted(MODEL_SOURCES)}."
                )
        run = cls.__dict__.get('run')
        if run is None:
            # Abstract: an intermediate base (OASES) that leaves ``run`` to
            # its own subclasses. Nothing below applies, and the two
            # declarations required of a concrete wrapper are its subclasses'
            # to make.
            return

        # A subclass that defines ``run`` is instantiable, so it is a model a
        # user can hold — and both declarations below are load-bearing at that
        # point. Without ``spec`` the class silently takes the base defaults
        # (COHERENT_TL only, no env-shape support, point sources), which are
        # nobody's real answer; without ``source`` the licence and citation
        # machinery is skipped entirely, so an engine that must warn on use
        # would not (``_warn_restricted_source`` returns on ``source is
        # None``). Both are checked here rather than at ``__init__`` so a
        # missing declaration fails on import.
        missing = [name for name in ('spec', 'source')
                   if getattr(cls, name, None) is None]
        if missing:
            raise TypeError(
                f"{cls.__name__} defines run() but declares no "
                f"{' or '.join(missing)}. A concrete model must set both: "
                f"``spec = ModelSpec(...)`` for its run modes, env-shape "
                f"support and source geometries, and ``source = '<id>'`` "
                f"naming its engine in MODEL_SOURCES so licence and citation "
                f"metadata reach the user."
            )

        import inspect

        params = list(inspect.signature(run).parameters.values())
        for p in params:
            if p.kind is inspect.Parameter.VAR_KEYWORD:
                raise TypeError(
                    f"{cls.__name__}.run() must not use **kwargs: an unknown "
                    "keyword has to raise TypeError, not be swallowed. Declare "
                    "the accepted extras as keyword-only after a bare '*'."
                )

        leading = params[:len(cls._RUN_POSITIONAL)]
        names = tuple(p.name for p in leading)
        if names != cls._RUN_POSITIONAL:
            raise TypeError(
                f"{cls.__name__}.run() must begin with "
                f"{cls._RUN_POSITIONAL!r}; got {names!r}."
            )
        for p in leading:
            if p.kind is inspect.Parameter.KEYWORD_ONLY:
                raise TypeError(
                    f"{cls.__name__}.run() parameter {p.name!r} must be "
                    "positional-or-keyword, not keyword-only."
                )

        for p in params[len(cls._RUN_POSITIONAL):]:
            if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                raise TypeError(
                    f"{cls.__name__}.run() parameter {p.name!r} must be "
                    "keyword-only (place it after a bare '*') so unknown "
                    "positional args cannot reach it."
                )

    def __init__(
        self,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        timeout: float = 600.0,
        collapse: Optional[Dict[str, str]] = None,
    ):
        from uacpy._log import _resolve_threshold
        _resolve_threshold(verbose)  # validate up front
        self.model_name = self.__class__.__name__
        self.use_tmpfs = use_tmpfs
        self.verbose = verbose
        self.work_dir = work_dir
        # cleanup defaults to True only when uacpy owns the work dir. Whether
        # the caller passed it is kept so ``copy()`` can re-resolve against a
        # new work_dir instead of carrying an auto-derived True onto a
        # caller-supplied directory and deleting it.
        self._cleanup_explicit = cleanup is not None
        self.cleanup = (work_dir is None) if cleanup is None else bool(cleanup)
        self.timeout = float(timeout)
        # Per-feature collapse policies applied by ``_project_environment``
        # when an env contains a feature this model doesn't support. Pass
        # ``collapse={'bathymetry': 'min', 'ssp': 'mean', ...}`` to override
        # any subset; missing keys keep the defaults.
        #
        # 'bathymetry'    : 'max'|'median'|'mean'|'min'|'initial'
        # 'ssp'           : 'r0'|'rmax'|'mean'|'median'
        # 'bottom_range'  : 'r0'|'rmax'|'mean'|'median' — reduce a range-
        #                   dependent bottom to one column (mean/median
        #                   numeric only for an all-half-space bottom)
        # 'bottom_layers' : 'halfspace'|'top_layer'|'volume_average' — flatten
        #                   each column's layer stack to a half-space; see
        #                   SeabedColumn.collapse for what each method keeps
        # 'altimetry'     : 'drop'
        # 'surface'       : 'r0'|'rmax'|'mean'|'median' — reduce a range-
        #                   dependent Surface to a single boundary
        # 'elastic'       : 'fluid' (zero shear) | 'vacuum'
        #
        # ``self._collapse`` is the resolved policy (defaults ← spec ← user);
        # ``self.collapse`` keeps the constructor argument verbatim, because
        # ``copy()`` and ``__repr__`` read every knob off ``self.<param>``.
        self._collapse: Dict[str, str] = dict(DEFAULT_COLLAPSE)
        self._user_collapse: Dict[str, str] = {}
        self.collapse = dict(collapse) if collapse else None
        if collapse:
            unknown = set(collapse) - set(DEFAULT_COLLAPSE)
            if unknown:
                raise ConfigurationError(
                    f"Unknown collapse keys: {sorted(unknown)}. "
                    f"Valid keys: {sorted(DEFAULT_COLLAPSE)}"
                )
            for key, value in collapse.items():
                if value not in VALID_COLLAPSE_METHODS[key]:
                    raise ConfigurationError(
                        f"Invalid collapse value for {key!r}: {value!r}. "
                        f"Valid values: {sorted(VALID_COLLAPSE_METHODS[key])}"
                    )
            self._collapse.update(collapse)
            self._user_collapse = dict(collapse)

        # Subclasses override to declare the run modes they support.
        self._supported_modes: List[RunMode] = [RunMode.COHERENT_TL]

        # Capability flags — one per axis of ``Environment`` shape. Subclasses
        # flip True for each feature they honour natively; anything left False
        # that's present in env on ``run()`` is collapsed by
        # ``_project_environment`` and triggers one ``UserWarning`` per dropped
        # feature.
        #
        # The flag list is intentionally bounded. Add a flag ONLY for a
        # question of the form "does this env shape work with this model?".
        # Niche numerical-method requirements (3-D, broadband, specific SSP
        # interp scheme, volume-attenuation formula) belong in run()-time
        # asserts, not here.
        self._supports_altimetry: bool = False
        self._supports_range_dependent_bathymetry: bool = False
        self._supports_range_dependent_ssp: bool = False
        self._supports_range_dependent_bottom: bool = False
        self._supports_layered_bottom: bool = False
        self._supports_elastic_media: bool = False
        # Bellhop is the only model that runs one source-depth grid in a
        # single binary call. Nothing loops in Python: the ten models that
        # read source geometry raise from ``_validate_geometry`` and tell the
        # caller to loop over single-depth ``Source``s. Bounce accepts a
        # multi-depth ``Source`` without raising because it reads no source
        # geometry at all and overrides ``_validate_geometry`` to a no-op
        # (``bounce.py``); the extra depths reach no deck.
        self._supports_multi_source_depth: bool = False
        self._supports_source_beam_pattern: bool = False
        # Surface sigma(1). SPARC's GetPar (Scooter/sparc.f90:177) and
        # Bounce's elastic branch (Kraken/bounce.f90:104) ERROUT on a
        # non-zero SSP%sigma; Kraken consumes it (Kraken/kraken.f90:902) and
        # Scooter in its vacuum-boundary impedance (Scooter/scooter.f90:309).
        self._supports_rough_surface: bool = False
        # Seabed sigma(NMedia+1). Kraken/KrakenC feed it to the
        # Kuperman-Ingenito interfacial-roughness perturbation
        # (Kraken/Scattering.f90:8, Kraken/kraken.f90:902); Bellhop's solver
        # ignores the value and RAM's PE format has nowhere to put it.
        self._supports_rough_bottom: bool = False
        self._supported_source_types: frozenset = frozenset({'point'})

        # When the subclass declares a ModelSpec, apply it now (after the
        # defaults above and after ``_user_collapse`` is populated, so the
        # collapse precedence DEFAULT ← spec ← user override holds). A
        # subclass may still override an individual flag afterward for the
        # rare instance-dependent capability.
        if self.spec is not None:
            self._apply_spec()
        # Independent of spec: a model may declare ``source`` without one.
        self._warn_restricted_source()

    @property
    def provenance(self):
        """The engine's :class:`~uacpy.models.sources.ModelSource` (authorship
        + licence + citation), or ``None`` if the class declares no
        :attr:`source`."""
        from uacpy.models.sources import model_source
        return model_source(self.source)

    @property
    def citation(self) -> str:
        """The engine's bibliographic citation string (``''`` if unknown)."""
        src = self.provenance
        return src.citation if src is not None else ''

    def _warn_restricted_source(self) -> None:
        """Emit a one-time ``UserWarning`` for a licence-restricted engine.

        Mirrors the non-commercial fetch warning in ``data/`` (the audit's
        CRUST1.0 fix): a ``commercial_use=False`` source — OASES — must never
        be used silently. Deduplicated per source ``id`` per process so a
        parameter sweep warns once, not on every instance.
        """
        src = self.provenance
        if src is None or src.commercial_use or src.id in _WARNED_MODEL_SOURCES:
            return
        _WARNED_MODEL_SOURCES.add(src.id)
        warnings.warn(
            f"{self.model_name} uses {src.name} ({src.license}). {src.note} "
            f"Cite: {src.citation}",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )

    def _apply_spec(self) -> None:
        """Install :attr:`spec`'s metadata onto this instance.

        Sets ``_supported_modes`` and every ``_supports_<flag>`` from the
        declarative manifest and layers the spec's collapse defaults via
        :meth:`_set_collapse_defaults` (so user overrides still win). The
        spec is already validated in ``__init_subclass__``.
        """
        spec = self.spec
        if spec.modes:
            self._supported_modes = list(spec.modes)
        for flag in _CAPABILITY_FLAGS:
            setattr(self, f'_supports_{flag}', flag in spec.supports)
        self._supported_source_types = frozenset(spec.source_types)
        if spec.collapse:
            self._set_collapse_defaults(spec.collapse)

    def _set_collapse_defaults(self, defaults: Dict[str, str]) -> None:
        """Subclass hook: install model-specific collapse defaults.

        Each ``(key, value)`` is applied only when the user did not pass
        an explicit value for ``key`` in ``Model(collapse={...})``, so
        user overrides always win. Use it to express physics-aware
        defaults that differ from the global ``DEFAULT_COLLAPSE``.
        """
        for key, value in defaults.items():
            if key not in self._user_collapse:
                self._collapse[key] = value

    def _resolve_run_mode(
        self,
        run_mode: Optional[Union['RunMode', str]],
        *,
        default: Optional['RunMode'] = None,
    ) -> 'RunMode':
        """Default ``None`` to the model's first supported mode and then
        validate that ``run_mode`` is in ``_supported_modes``. Raises
        :class:`UnsupportedFeatureError` otherwise.

        Strings matching a :class:`RunMode` value (e.g. ``'coherent_tl'``)
        are coerced to the corresponding enum member.

        Pass ``default=`` to override the auto-pick when the model has a
        smarter rule (e.g. Kraken picks BROADBAND when a frequency
        vector is supplied).
        """
        if run_mode is None:
            run_mode = default if default is not None else self._supported_modes[0]
        if isinstance(run_mode, str):
            try:
                run_mode = RunMode(run_mode)
            except ValueError:
                raise UnsupportedFeatureError(
                    self.model_name, repr(run_mode),
                    alternatives=[str(m) for m in self._supported_modes],
                    alternatives_label='run modes',
                )
        if not self.supports_mode(run_mode):
            raise UnsupportedFeatureError(
                self.model_name, str(run_mode),
                alternatives=[str(m) for m in self._supported_modes],
                alternatives_label='run modes',
            )
        return run_mode

    def _reject_unsupported_run_kwargs(self, **kwargs):
        """Guard the optional ``run()`` keywords a model does not consume.

        Every engine declares the full contract signature (``frequencies``,
        ``source_waveform``, ``sample_rate``, ``output_duration``) so a
        polymorphic ``model.run(...)`` never raises ``TypeError``. Frequency-
        domain engines (Bounce, OAST, OASN) ignore the broadband/waveform
        keywords — but ignoring them *silently* would hide a caller mistake, so
        any of these passed a non-``None`` value raises here instead."""
        supplied = sorted(name for name, value in kwargs.items() if value is not None)
        if supplied:
            raise UnsupportedFeatureError(
                self.model_name,
                f"run parameter(s): {', '.join(supplied)}",
                alternatives_label='run parameters',
            )

    def _warn_ignored_run_kwargs(self, run_mode, reason=None, **named_values):
        """Warn when the resolved ``run_mode`` will not consume some of the
        optional ``run()`` keywords the caller supplied.

        Complements :meth:`_reject_unsupported_run_kwargs`: that helper is
        for models that *never* consume a keyword (hard error); this one is
        for keywords the model does consume, just not on the resolved run
        mode's path. Only non-``None`` values are reported, in a single
        ``UserWarning``. Consuming paths (BROADBAND/TIME_SERIES, …) must not
        call it."""
        ignored = [
            f'{name}=' for name, value in named_values.items()
            if value is not None
        ]
        if not ignored:
            return
        if reason is None:
            reason = 'these apply to BROADBAND/TIME_SERIES only'
        warnings.warn(
            f"{self.model_name}.run(run_mode="
            f"{getattr(run_mode, 'name', run_mode)}): ignoring "
            f"{', '.join(ignored)} — {reason}.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )

    @property
    def supported_modes(self) -> List[RunMode]:
        """List of run modes supported by this model."""
        return self._supported_modes

    def supports_mode(self, mode: RunMode) -> bool:
        """Return True if the model supports ``mode``."""
        return mode in self._supported_modes

    @property
    def supported_features(self) -> List[str]:
        """Environment-shape features this *instance* carries into its deck.

        The env-shape twin of :attr:`supported_modes`. Names are the
        ``_CAPABILITY_FLAGS`` vocabulary (``'range_dependent_ssp'``,
        ``'layered_bottom'``, …); a feature listed here survives
        ``_project_environment`` untouched, one that is not is collapsed or
        dropped with a warning.

        Read from the instance, not from ``spec.supports``: a model may
        resolve a flag from its own constructor arguments (``Bellhop`` turns
        ``range_dependent_ssp`` off for ``interp_ssp='c-linear'``), and the
        class-level declaration cannot know that.
        """
        return sorted(name for name in _CAPABILITY_FLAGS
                      if getattr(self, f'_supports_{name}'))

    def supports_feature(self, name: str) -> bool:
        """Return True if this instance carries env-shape feature ``name``.

        Raises
        ------
        ValueError
            For a name outside ``_CAPABILITY_FLAGS`` — a typo would
            otherwise answer ``False``, which reads as a real "no".
        """
        if name not in _CAPABILITY_FLAGS:
            raise ValueError(
                f"unknown capability {name!r}; expected one of "
                f"{sorted(_CAPABILITY_FLAGS)}"
            )
        return bool(getattr(self, f'_supports_{name}'))

    def copy(self, **overrides) -> Self:
        """Return a new instance with the same configuration plus ``overrides``.

        Model configuration is constructor-only by design, which means
        every parameter sweep boils down to "instantiate the model again
        with one knob changed." This helper does that without forcing
        the caller to re-type every other argument::

            base = RAM(dr=2.0, dz=0.5, np_pade=8)
            for dr in (1.0, 2.0, 4.0):
                run_one(base.copy(dr=dr), env, source, receiver)

        Implementation: walks ``__init__`` along the MRO (so parameters
        defined on a parent constructor are included, since subclasses
        forward via ``super().__init__(**kwargs)``), pulls each parameter's
        current value off the instance (uacpy models store every
        constructor arg as ``self.<name>``), merges ``overrides``, and
        instantiates. ``**kwargs``-only sinks on the constructor are ignored.

        Parameters
        ----------
        **overrides
            Keyword arguments to override on the new instance.

        Returns
        -------
        Self
            A fresh instance of the same concrete class. ``Self`` rather than
            ``PropagationModel``: ``OASP(...).copy()`` is an ``OASP``, and
            declaring the base class discards that at every call site.

        Raises
        ------
        ConfigurationError
            If ``overrides`` includes a key that isn't a parameter of
            the constructor.
        """
        kwargs: Dict[str, object] = {}
        valid_names = set()
        for name, _default in _collect_init_params(type(self)):
            valid_names.add(name)
            if hasattr(self, name):
                kwargs[name] = getattr(self, name)

        unknown = set(overrides) - valid_names
        if unknown:
            raise ConfigurationError(
                f"{type(self).__name__}.copy: unknown override(s) "
                f"{sorted(unknown)}; valid parameters are "
                f"{sorted(valid_names)}."
            )

        kwargs.update(overrides)

        # An auto-derived ``cleanup`` describes *this* instance's work_dir, not
        # the clone's. Hand back the ``None`` sentinel so the new instance
        # re-resolves; otherwise ``model.copy(work_dir=d)`` off an unpinned
        # model would arrive at ``cleanup=True`` and rmtree the caller's ``d``.
        # An explicitly passed value is the caller's choice and is preserved.
        if (not self._cleanup_explicit and 'cleanup' not in overrides):
            kwargs['cleanup'] = None

        return type(self)(**kwargs)

    @abstractmethod
    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[RunMode] = None,
        *,
        frequencies: Optional[np.ndarray] = None,
        source_waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        output_duration: Optional[float] = None,
    ) -> Union[Result, ResultStack]:
        """Run the propagation model.

        Every wrapper takes the same first four parameters in the same
        order: the common ``Environment`` / ``Source`` / ``Receiver``
        triple, followed by the optional ``run_mode``.

        Model configuration is **constructor-only** — every model knob
        (e.g. ``RAM(dr=2.0, dz=0.5, np_pade=8)``,
        ``Bellhop(beam_type='B', n_beams=500)``) is set when the model
        instance is created. To sweep parameters, instantiate one model
        per parameter set.

        ``run()`` accepts a fixed keyword-only set: ``frequencies``,
        ``source_waveform``, ``sample_rate``, ``output_duration``. Every
        TIME_SERIES-capable wrapper (Bellhop, Scooter, Kraken,
        OASP, RAM) consumes ``source_waveform`` and ``sample_rate``;
        SPARC warns that they are ignored (it uses its constructor
        ``pulse_type``). ``output_duration`` is the desired output time
        window (seconds); when given, the IFFT-based wrappers zero-pad
        the source waveform internally so the auto-derived broadband
        grid is tight enough (``Δf = 1/output_duration``), and Bellhop
        maps it to ``time_window`` for delay-and-sum synthesis. Models
        with a broadband path consume ``frequencies`` as an explicit
        override for ``source.frequencies``. No other kwargs are accepted —
        passing one raises :class:`TypeError`.

        ``frequencies`` passed with a single-frequency ``run_mode`` (the
        TL modes, ``RAYS`` / ``EIGENRAYS`` / ``ARRIVALS``, ``MODES``) is
        handled three ways across the family: OASP consumes it — the
        value pins the frequency sweep its solver always runs; Kraken,
        RAM, Scooter and Bellhop emit a ``UserWarning`` and ignore it;
        OAST, Bounce and OASN raise :class:`UnsupportedFeatureError`.

        Parameters
        ----------
        env : Environment
            Ocean environment.
        source : Source
            Acoustic source.
        receiver : Receiver
            Receiver grid.
        run_mode : RunMode, optional
            Output type to compute. ``None`` selects the model's natural
            default (typically ``RunMode.COHERENT_TL``). Each wrapper's
            :attr:`supported_modes` lists what it accepts.
        frequencies : array-like, optional
            Keyword-only. Explicit broadband frequency grid (Hz), overriding
            ``source.frequencies`` for models with a broadband path.
        source_waveform : array-like, optional
            Keyword-only. Source time series for ``RunMode.TIME_SERIES``
            synthesis (consumed with ``sample_rate``).
        sample_rate : float, optional
            Keyword-only. Sample rate (Hz) of ``source_waveform``.
        output_duration : float, optional
            Keyword-only. Desired output time-window length (s); IFFT-based
            wrappers zero-pad so ``Δf = 1/output_duration``. A TIME_SERIES
            result's ``frequencies`` stamp is that same grid, so it changes
            when ``output_duration`` does — it records the frequencies the
            engine actually propagated for this call, not a property of the
            environment.

        Returns
        -------
        result : Result or ResultStack
            One of the typed :mod:`uacpy.core.results` subclasses
            (``Field``, ``Arrivals``, ``Modes``, …) determined
            by ``run_mode`` and the model — or a ``ResultStack`` of them
            when one run covers several source depths that the model cannot
            stack itself (``Bellhop`` in ``EIGENRAYS`` mode over a
            multi-depth ``Source``; see DOCUMENTATION.md §ResultStack).
            ``ResultStack`` is not a ``Result`` subclass, so a caller that
            annotates the result has to name both.
        """
        # Validates the (env, source, receiver) triple. No wrapper reaches it
        # today — every override opens with its own _require_run_triple call
        # and none delegates to super().run() — but this is the correct
        # prelude for one that ever does, and dropping it would make such a
        # super().run() a silent no-op.
        self._require_run_triple(env, source, receiver)

    # Modes that consume exactly one source frequency. Multi-frequency
    # Source passed to one of these is a configuration error — the user
    # should pick BROADBAND/TIME_SERIES, or REFLECTION/COVARIANCE/REPLICA
    # for the OASES family that genuinely supports multi-freq sweeps.
    _SINGLE_FREQUENCY_MODES: 'frozenset[RunMode]' = frozenset({
        RunMode.COHERENT_TL, RunMode.INCOHERENT_TL, RunMode.SEMICOHERENT_TL,
        RunMode.RAYS, RunMode.EIGENRAYS, RunMode.ARRIVALS, RunMode.MODES,
    })

    def _require_timeseries_signal(
        self,
        run_mode: 'RunMode',
        source_waveform,
        sample_rate,
    ):
        """Validate the TIME_SERIES signal pair and return the waveform the
        run should carry.

        Raises :class:`ConfigurationError` when the caller asked for a
        :attr:`RunMode.TIME_SERIES` result but did not supply both
        ``source_waveform`` and ``sample_rate``, when the waveform is not a
        real finite pressure pulse, or when ``sample_rate`` is not a
        positive finite number.

        Returns ``source_waveform`` itself, except that an accepted complex
        waveform — one whose imaginary part is everywhere ~0, which the
        realness check deliberately admits — comes back as its ``float64``
        real part. Every downstream consumer casts with ``dtype=float``
        (``_pad_waveform_to_duration``, ``_resolve_time_series_frequencies``,
        Bellhop's delay-and-sum), a cast that raises ``TypeError`` on a
        complex Python list and emits ``ComplexWarning`` on a complex
        ndarray; callers therefore run with the returned waveform, never the
        one they passed in.

        Used by every wrapper that synthesises p(t) from a broadband
        transfer function (Bellhop, RAM, Scooter, Kraken, OASP).
        SPARC has its own pulse mechanism (``pulse_type``) and does not
        call this helper.
        """
        if run_mode == RunMode.TIME_SERIES and (
            source_waveform is None or sample_rate is None
        ):
            raise ConfigurationError(
                f"{self.model_name}.run(run_mode=TIME_SERIES) requires "
                f"source_waveform and sample_rate. For the broadband "
                f"transfer function H(f), use run_mode=RunMode.BROADBAND."
            )
        if run_mode == RunMode.TIME_SERIES and sample_rate is not None:
            # Refuse a sample rate that is not a positive finite number. The
            # test is NaN-closed (``not (sr > 0)`` rather than ``sr <= 0``)
            # because nan compares False both ways and would slip through:
            # nothing downstream stops it either — _pad_waveform_to_duration
            # and _resolve_time_series_frequencies no-op on a
            # non-positive rate — so it surfaces as a ZeroDivisionError in
            # the delay-and-sum, a raw ValueError from the deep guard in
            # core/results/field.py, or a trace on a descending time axis.
            try:
                sr = float(sample_rate)
            except (TypeError, ValueError):
                sr = float('nan')
            if not np.isfinite(sr) or not (sr > 0.0):
                raise ConfigurationError(
                    f"{self.model_name}.run(run_mode=TIME_SERIES): "
                    f"sample_rate must be a positive finite number of Hz; "
                    f"got {sample_rate!r}."
                )
        if run_mode == RunMode.TIME_SERIES and source_waveform is not None:
            wf = np.asarray(source_waveform)
            if not np.all(np.isfinite(wf)):
                raise ConfigurationError(
                    f"{self.model_name}.run(run_mode=TIME_SERIES): "
                    "source_waveform contains non-finite values (NaN/inf)."
                )
            if np.iscomplexobj(wf):
                if not np.allclose(wf.imag, 0.0):
                    raise ConfigurationError(
                        f"{self.model_name}.run(run_mode=TIME_SERIES): "
                        "source_waveform must be a real pressure pulse; got a "
                        "complex array with a non-zero imaginary part."
                    )
                return wf.real.astype(np.float64)
        return source_waveform

    def _require_run_triple(self, env, source, receiver, *,
                            allow_none_receiver=False) -> None:
        """Raise :class:`ConfigurationError` unless ``run()``'s three
        positional carriers are an :class:`Environment`, a :class:`Source`
        and a :class:`Receiver` (subclasses accepted), in that order.

        The first statement of every wrapper's ``run()``: without it a
        swapped pair — ``run(source, env, receiver)`` — surfaces as a raw
        ``AttributeError`` deep inside deck assembly, far from the call
        that caused it. ``allow_none_receiver=True`` lets a wrapper whose
        own contract accepts ``receiver=None`` (Bounce, where receivers
        are inert and ``rmax=`` can stand in) apply that contract itself.
        """
        wrong = [
            f"{name}={type(value).__name__}"
            for name, expected, value in (
                ('env', Environment, env),
                ('source', Source, source),
                ('receiver', Receiver, receiver),
            )
            if not isinstance(value, expected)
            and not (allow_none_receiver
                     and name == 'receiver' and value is None)
        ]
        if wrong:
            raise ConfigurationError(
                f"{self.model_name}.run takes (env: Environment, "
                f"source: Source, receiver: Receiver, ...) in that "
                f"order; got {', '.join(wrong)}."
            )
        # Every wrapper opens with this call, so hanging the absorption
        # notice here is what makes it model-independent — but only the
        # wrappers that actually carry ``env.absorption`` to their engine may
        # advise setting it.
        if self._consumes_volume_absorption:
            _warn_if_volume_absorption_is_missing(env, source, receiver)

    def _pad_waveform_to_duration(
        self, source_waveform, sample_rate, output_duration,
    ):
        """Zero-pad ``source_waveform`` so its duration is at least
        ``output_duration`` seconds. Returns the (possibly padded) array
        unchanged when ``output_duration`` is ``None`` or already met.

        Used by every IFFT-based TIME_SERIES wrapper so the user can
        request a longer output than the source pulse without having to
        pre-pad: ``Field.synthesize_time_series`` sets output duration
        = waveform duration, and the auto-derived broadband grid uses
        ``Δf = 1 / waveform_duration``.
        """
        if (
            output_duration is None
            or source_waveform is None
            or sample_rate is None
            or sample_rate <= 0
        ):
            return source_waveform
        wf = np.asarray(source_waveform, dtype=float).ravel()
        n_needed = int(np.ceil(float(output_duration) * float(sample_rate)))
        if wf.size >= n_needed:
            return source_waveform
        pad = np.zeros(n_needed - wf.size, dtype=wf.dtype)
        return np.concatenate([wf, pad])

    def _resolve_broadband_frequencies(
        self,
        source: 'Source',
        frequencies,
        *,
        n_freqs: Optional[int] = None,
        bandwidth_factor: Optional[float] = None,
    ) -> np.ndarray:
        """Resolve the BROADBAND frequency grid.

        Explicit ``frequencies=`` wins. Otherwise a multi-element
        ``source.frequencies`` *is* the band and is used as-is. A single
        centre frequency expands to ``n_freqs`` bins spanning
        ``fc·(1 ± bandwidth_factor/2)``.
        """
        from uacpy.core.constants import (
            DEFAULT_BROADBAND_N_FREQS,
            DEFAULT_BROADBAND_BANDWIDTH_FACTOR,
        )
        if frequencies is not None:
            return np.asarray(frequencies, dtype=float)
        src_f = np.atleast_1d(np.asarray(source.frequencies, dtype=float))
        if src_f.size > 1:
            return src_f
        if n_freqs is None:
            n_freqs = DEFAULT_BROADBAND_N_FREQS
        n_freqs = int(n_freqs)
        # ``np.linspace`` degenerates below two points: 1 returns the lower
        # band edge alone — a grid silently mislabelled as the band — and 0
        # an empty grid, so neither can span fc·(1 ± bandwidth_factor/2).
        if n_freqs < 2:
            raise ConfigurationError(
                f"{self.model_name} broadband: n_freqs = {n_freqs} cannot "
                f"span a frequency band — the expanded grid needs at least "
                f"its two edges.",
                remediation=(
                    "Use n_freqs >= 2, or pass frequencies=[fc] to run a "
                    "single bin."
                ),
            )
        if bandwidth_factor is None:
            bandwidth_factor = DEFAULT_BROADBAND_BANDWIDTH_FACTOR
        fc = float(src_f[0])
        half_bw = 0.5 * float(bandwidth_factor)
        # The lower edge goes to zero or negative once bandwidth_factor >= 2;
        # a 0 Hz bin is not a runnable model frequency, so floor it at 1 Hz —
        # and say so, since the band is then neither centred on fc nor the
        # requested width.
        lo = fc * (1.0 - half_bw)
        hi = fc * (1.0 + half_bw)
        if lo < 1.0:
            warnings.warn(
                f"{self.model_name} broadband: bandwidth_factor="
                f"{bandwidth_factor:g} puts the lower band edge at "
                f"{lo:.4g} Hz; floored at 1 Hz, so the band is no longer "
                f"centred on fc = {fc:g} Hz nor the requested width.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
            lo = 1.0
        # A sub-1 Hz fc can put the floored lower edge at or above the upper
        # edge; a descending or duplicated frequency axis is a silently
        # mislabelled Field, so refuse it.
        if hi <= lo:
            raise ConfigurationError(
                f"{self.model_name} broadband: the band "
                f"[{lo:g}, {hi:g}] Hz is empty after the 1 Hz floor "
                f"(fc = {fc:g} Hz, bandwidth_factor = {bandwidth_factor:g}). "
                f"Sub-1 Hz centre frequencies need an explicit frequencies= "
                f"grid."
            )
        return np.linspace(lo, hi, int(n_freqs))

    def _resolve_time_series_frequencies(
        self,
        run_mode: 'RunMode',
        frequencies,
        source_waveform,
        sample_rate,
        threshold_db: float = -40.0,
        announce: bool = True,
    ):
        """Resolve the broadband frequency grid for TIME_SERIES dispatch.

        When ``run_mode == TIME_SERIES`` and the caller did not pass
        ``frequencies=``, derive one from the source waveform: Δf =
        ``sample_rate / n_samples`` (= 1 / waveform duration), band edges
        from the spectral support above ``threshold_db`` below the peak
        (default −40 dB). Pinning ``frequencies=`` skips derivation.
        Other run-modes pass through unchanged.

        Returns the resolved ndarray (uniformly spaced, Hz), or ``None``
        when no override applies. Emits a single ``UserWarning`` so the
        user sees what band/Δf were picked; ``announce=False`` skips it —
        used when the grid only labels a result's frequency axis (Bellhop's
        delay-and-sum p(t)) rather than driving a solver run.
        """
        if (
            run_mode != RunMode.TIME_SERIES
            or frequencies is not None
            or source_waveform is None
            or sample_rate is None
        ):
            return frequencies
        wf = np.asarray(source_waveform, dtype=float).ravel()
        n = wf.size
        if n < 2 or sample_rate <= 0:
            return frequencies
        fs = float(sample_rate)
        df = fs / n
        spectrum = np.abs(np.fft.rfft(wf))
        src_freqs = np.fft.rfftfreq(n, 1.0 / fs)
        peak = spectrum.max()
        if peak <= 0:
            raise ConfigurationError(
                f"{self.model_name}.run(run_mode=TIME_SERIES): "
                f"source_waveform is identically zero."
            )
        threshold = peak * 10.0 ** (threshold_db / 20.0)
        significant = spectrum >= threshold
        if not significant.any():
            raise ConfigurationError(
                f"{self.model_name}.run(run_mode=TIME_SERIES): "
                f"source_waveform has no spectral content above "
                f"{threshold_db} dB."
            )
        i_lo = int(np.argmax(significant))
        i_hi = len(significant) - 1 - int(np.argmax(significant[::-1]))
        # A pulse with DC content puts ``i_lo`` on bin 0, i.e. f = 0 Hz, which
        # no model can be run at; the first non-zero bin is Δf.
        f_min = max(float(src_freqs[i_lo]), df)
        f_max = float(src_freqs[i_hi])
        if f_max <= f_min:
            f_max = f_min + df
        n_freqs = int(round((f_max - f_min) / df)) + 1
        # A short pulse gives a coarse Δf = 1/duration, so a narrow band can
        # derive only 2-3 bins — too few for the band-edge taper to leave an
        # interior, and too few to represent an arrival at all. Subdivide Δf
        # (equivalent to zero-padding the pulse) rather than hand back a grid
        # the synthesis cannot use.
        refined = max(n_freqs, _MIN_TIMESERIES_FREQS)
        derived = np.linspace(f_min, f_max, refined)
        # Quote the spacing of the grid actually returned; the waveform's own
        # Δf = 1/duration goes in the refinement note so the two never
        # contradict each other.
        df_grid = (f_max - f_min) / (refined - 1)
        note = ""
        if refined != n_freqs:
            note = (f" (waveform Δf = {df:.4g} Hz subdivided so the "
                    f"{n_freqs}-bin band resolves an arrival)")
        if announce:
            # Name the record as well as the spacing. They are one number
            # written two ways — a synthesised trace is 1/Δf long — but the
            # spacing alone leaves the consequence to be derived: this Δf
            # comes from the SOURCE pulse, and a channel whose multipath
            # outlasts it folds the tail onto the early trace, where it reads
            # as extra early arrivals rather than as a mistake.
            record = 1.0 / df_grid if df_grid > 0 else float('inf')
            warnings.warn(
                f"{self.model_name}.run(run_mode=TIME_SERIES): no "
                f"`frequencies=` passed; auto-derived {refined} freqs from "
                f"the source waveform ({f_min:.2f}-{f_max:.2f} Hz, "
                f"Δf={df_grid:.4g} Hz, threshold {threshold_db:.0f} dB){note}. "
                f"That Δf makes the record {record:.4g} s long, and it is set "
                f"by the pulse, not by the channel: any arrival later than "
                f"{record:.4g} s after the first folds back onto the early "
                f"trace. Pass `output_duration=` to buy a longer record, or "
                f"`frequencies=` to set the grid yourself and silence this.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
        return derived

    def _prepare_timeseries(
        self, run_mode, source, frequencies, source_waveform, sample_rate,
        output_duration=None,
    ):
        """Common TIME_SERIES/BROADBAND dispatch preamble for the IFFT-based
        synthesizers (RAM, Kraken, Scooter, OASP): validate the source
        pulse, zero-pad it to ``output_duration``, and derive the broadband
        frequency grid. Returns ``(source_waveform, frequencies)``.

        Bellhop validates the pulse directly and resolves its own grid, and
        SPARC uses ``pulse_type`` instead, so neither calls this.

        ``sample_rate`` / ``output_duration`` size the synthesised time axis
        and are read only on the TIME_SERIES branch, so a BROADBAND run that
        was handed either warns here rather than dropping it in silence.
        """
        if run_mode == RunMode.BROADBAND:
            self._warn_ignored_run_kwargs(
                run_mode,
                reason=('BROADBAND returns the transfer function H(f); the '
                        'source pulse and time-axis keywords apply to '
                        'TIME_SERIES only'),
                source_waveform=source_waveform,
                sample_rate=sample_rate,
                output_duration=output_duration,
            )
        source_waveform = self._require_timeseries_signal(
            run_mode, source_waveform, sample_rate)
        source_waveform = self._pad_waveform_to_duration(
            source_waveform, sample_rate, output_duration,
        )
        frequencies = self._resolve_time_series_frequencies(
            run_mode, frequencies, source_waveform, sample_rate,
        )
        return source_waveform, frequencies

    def _finish_broadband(self, result, run_mode, source_waveform,
                          sample_rate):
        """Return the BROADBAND transfer function as computed or, for
        TIME_SERIES, its synthesis with the pulse ``_prepare_timeseries``
        validated. Every IFFT-based wrapper ends its broadband route here.
        """
        if run_mode == RunMode.TIME_SERIES:
            return result.synthesize_time_series(
                source_waveform=source_waveform, sample_rate=sample_rate)
        return result

    def _setup_file_manager(self) -> FileManager:
        """Build the FileManager. ``self.work_dir`` is used as-is (not a
        parent); when ``None``, a fresh temp dir is created.

        Auto-creates the user-pinned ``work_dir`` if it doesn't exist
        yet, so callers can construct ``Model(work_dir='./out')`` without
        a separate ``mkdir`` step.

        ``self.cleanup`` drives the manager on both branches, so it stays the
        single decision :meth:`_attach_output_paths` keys the ``*_file``
        metadata on: a directory that survives the run always has valid paths
        attached, and a wiped one never does.

        A pinned ``work_dir`` is also *claimed* for this thread (see
        :func:`_claim_work_dir`), so a second thread pointed at the same
        directory raises instead of silently trading results with this one.
        The claim comes back through the manager's release hook.
        """
        if self.work_dir is not None:
            claim = _claim_work_dir(self.work_dir, self.model_name)
            try:
                # base_dir is the *parent*: FileManager validates that it
                # exists, while ``adopt_work_dir`` keys ownership on whether
                # work_dir itself already existed, so it has to do that mkdir
                # itself.
                parent = Path(self.work_dir).parent
                parent.mkdir(parents=True, exist_ok=True)
                if self.use_tmpfs:
                    warnings.warn(
                        f"{self.model_name}(use_tmpfs=True) is ignored for the "
                        f"pinned work_dir {self.work_dir} — a named directory "
                        f"cannot be relocated to /dev/shm. Drop work_dir= for "
                        f"RAM-backed I/O, or point work_dir at a tmpfs mount.",
                        UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
                    )
                fm = FileManager(
                    use_tmpfs=False,
                    base_dir=parent,
                    prefix=f'{self.model_name.lower()}_',
                    cleanup=self.cleanup,
                )
                # Adopted, not owned: cleanup may remove only what this run adds.
                fm.adopt_work_dir(self.work_dir)
                # FileManager's own writability check validated ``base_dir``,
                # i.e. the PARENT; the directory the decks actually go in is
                # this one, and a read-only one otherwise surfaced as a raw
                # PermissionError from the first writer.
                if not os.access(fm.work_dir, os.W_OK):
                    raise ConfigurationError(
                        f"work_dir is not writable: {fm.work_dir}",
                        remediation="Pass a writable work_dir= (or fix its "
                                    "permissions); uacpy writes the model's "
                                    "input and output files inside it.")
            except BaseException:
                # Nothing downstream can release a claim whose manager was
                # never built.
                _release_work_dir(claim)
                raise
            fm.on_release(lambda: _release_work_dir(claim))
        else:
            fm = FileManager(
                use_tmpfs=self.use_tmpfs,
                base_dir=None,
                prefix=f'{self.model_name.lower()}_',
                cleanup=self.cleanup,
            )
            fm.create_work_dir()

        return fm

    def _log(self, message: str, level: str = "info"):
        """Emit a tagged line through :func:`uacpy._log.log_message`.
        ``WARN`` / ``ERROR`` always print; ``INFO`` / ``DEBUG`` only when
        ``self.verbose``."""
        from uacpy._log import log_message
        log_message(
            self.model_name, message,
            verbose=self.verbose, level=level,
        )

    def validate_inputs(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional['RunMode'] = None,
    ):
        """
        Validate inputs against the environment + the resolved run mode.

        Parameters
        ----------
        env : Environment
            Environment to validate against.
        source : Source
            Source to validate.
        receiver : Receiver
            Receiver to validate.
        run_mode : RunMode, optional
            Resolved run mode. When passed, single-frequency modes
            (``COHERENT_TL``, ``RAYS``, ``MODES``, …) refuse a Source
            with more than one frequency.

        Raises
        ------
        InvalidDepthError
            If source depths exceed the model's resolvable depth.
        ConfigurationError
            If source/receiver depths are negative, if ``run_mode`` is a
            single-frequency mode and ``source`` carries multiple
            frequencies, if ``source.source_type`` is outside
            ``_supported_source_types``, or if ``source.beam_pattern`` is set
            on a model that does not read one.

        Notes
        -----
        The source/receiver geometry checks — including the source's angular
        geometry, i.e. its beam pattern — live in :meth:`_validate_geometry`,
        which a model that reads no geometry
        (:class:`~uacpy.models.bounce.Bounce`) overrides to a no-op.
        """
        if (run_mode is not None
                and run_mode in self._SINGLE_FREQUENCY_MODES
                and len(source.frequencies) > 1):
            raise ConfigurationError(
                f"{self.model_name}.run(run_mode={run_mode.name}) takes a "
                f"single source frequency; got {len(source.frequencies)}: "
                f"{list(source.frequencies)}. For broadband H(f) use "
                f"RunMode.BROADBAND, and for time-domain p(t) use "
                f"RunMode.TIME_SERIES."
            )

        if source.source_type not in self._supported_source_types:
            # ``Source`` has already validated the geometry name, so what
            # fails here is this model's coverage of a legal Source — the
            # "try another model" case ``UnsupportedFeatureError`` names.
            raise UnsupportedFeatureError(
                self.model_name,
                f"Source(source_type={source.source_type!r})",
                alternatives=[repr(t)
                              for t in sorted(self._supported_source_types)],
                alternatives_label='source geometries',
            )

        self._validate_geometry(env, source, receiver, run_mode)

    def _validate_geometry(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional['RunMode'] = None,
    ) -> None:
        """Check the source/receiver geometry against what the model resolves.

        Split out of :meth:`validate_inputs` so a model that reads no
        geometry at all can opt out wholesale. The source's angular geometry
        (its beam pattern) belongs here for that reason: a reflection-only
        engine never launches a ray fan, so rejecting a pattern it simply
        ignores would break reusing one ``Source`` across models.
        """
        if (source.beam_pattern is not None
                and not self._supports_source_beam_pattern):
            raise ConfigurationError(
                f"{self.model_name} does not read a source beam pattern; "
                f"drop Source(beam_pattern=...) or use Bellhop or Kraken."
            )

        if (not self._supports_multi_source_depth
                and len(np.atleast_1d(source.depths)) > 1):
            raise ConfigurationError(
                f"{self.model_name} takes a single source depth per run; "
                f"got {len(source.depths)}: {list(source.depths)}. Loop "
                f"over Sources externally for multi-depth runs."
            )

        resolvable_depth = self._max_receiver_depth(env)

        # The source injects energy into the medium, so it must sit within
        # what the model resolves — placing it below is a hard error.
        # Receivers are outputs: ones below the resolvable depth are
        # accepted and return the model's below-domain value (transmitted /
        # evanescent field, or NaN inside a PE absorbing layer); the
        # warning helpers below surface that rather than rejecting it.
        if np.any(source.depths > resolvable_depth):
            exc = InvalidDepthError(
                float(source.depths.max()), resolvable_depth, "Source",
            )
            note = self._source_below_domain_note(env, resolvable_depth)
            if note:
                # Message enrichment only. ``InvalidDepthError.__reduce__``
                # rebuilds from the three constructor arguments, so a copy
                # that crosses a process boundary (``run_parallel``) carries
                # the base remediation.
                exc.remediation = f"{exc.remediation}.\n\n{note}"
            raise exc

        if np.any(source.depths < 0):
            raise ConfigurationError("Source depths must be non-negative")

        # A source exactly at z = 0 sits on the pressure-release sea surface,
        # where the boundary forces p ≈ 0: a field run then returns a
        # degenerate result (a null / saturated-TL sentinel, or — in RAM — an
        # unphysical normalisation) that a valid-looking ``Source(depths=0)``
        # hides. Reflection coefficients and mode shapes don't propagate a
        # source field, so the warning doesn't apply to those.
        if (run_mode not in (RunMode.REFLECTION, RunMode.MODES)
                and np.any(np.asarray(source.depths) == 0.0)):
            warnings.warn(
                f"{self.model_name}: a source at depth 0 m is on the "
                f"pressure-release sea surface, where the field is ~0 — the "
                f"result is degenerate (null / saturated TL, model-dependent). "
                f"Use a small positive depth (e.g. 1 m).",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )

        if receiver.depth_min < 0:
            raise ConfigurationError("Receiver depths must be non-negative")

        self._warn_receiver_below_resolvable(env, receiver, resolvable_depth)
        self._check_per_range_receiver_depth(env, receiver)
        self._warn_on_range_coverage(env, receiver)

    def _warn_receiver_below_resolvable(
        self, env: 'Environment', receiver: 'Receiver',
        resolvable_depth: float,
    ) -> None:
        """Flat-bathymetry counterpart to
        :meth:`_check_per_range_receiver_depth`: warn — never raise — when a
        receiver lies below the depth this model resolves the field at.
        Such receivers are accepted; what comes back is per-engine:
        Bellhop, Scooter, SPARC and RAM return NaN there (their solvers
        clamp the receiver onto the domain or stop meshing, so no field is
        evaluated at the asked depth), while Kraken and the OASES models
        compute a physical transmitted / evanescent field through the
        sediment they mesh. The range-dependent case is handled per-range
        by :meth:`_check_per_range_receiver_depth`.
        """
        if env.has_range_dependent_bathymetry:
            return
        if receiver.depth_max > resolvable_depth:
            warnings.warn(
                f"{self.model_name}: receiver depth "
                f"{float(receiver.depth_max):.1f} m is below the model's "
                f"resolvable depth ({resolvable_depth:.1f} m). It is "
                f"accepted; the result there reflects the model's "
                f"below-domain behaviour (a physical transmitted / "
                f"evanescent field from Kraken and OASES; NaN from "
                f"Bellhop, Scooter, SPARC and RAM).",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )

    def _receiver_grid_is_paired(self, receiver: 'Receiver') -> bool:
        """Whether this model's deck pairs ``receiver.depths[i]`` with
        ``receiver.ranges[i]`` instead of spanning their Cartesian product.

        Returns ``False`` here because every wrapper but one writes a
        rectilinear grid; Bellhop overrides it for ``grid_type='I'``
        (``RunType(5:5)='I'``). The paired flag lives on the model rather
        than on :class:`Receiver` because the same Receiver written to a
        rectilinear deck genuinely does span the product.
        """
        return False

    def _check_per_range_receiver_depth(
        self, env: 'Environment', receiver: 'Receiver',
    ) -> None:
        """Emit a ``UserWarning`` if any receiver sits below the local
        seafloor in a range-dependent bathymetry. Below-seafloor receivers
        are accepted, not rejected; the cells come back NaN from the
        engines that evaluate no field there (Bellhop, Scooter, SPARC,
        RAM) and as a physical transmitted field from the ones that mesh
        the sediment (Kraken, OASES). The flat-bathy case is handled by
        :meth:`_warn_receiver_below_resolvable`.

        Which (depth, range) pairs the deck actually carries comes from
        :meth:`_receiver_grid_is_paired`.
        """
        if not env.has_range_dependent_bathymetry:
            return
        depths = np.atleast_1d(receiver.depths).astype(float)
        ranges = np.atleast_1d(receiver.ranges).astype(float)
        seafloor = np.asarray(env.bathymetry.eval(range=ranges), dtype=float)

        if self._receiver_grid_is_paired(receiver) and depths.size == ranges.size:
            # Paired deck: depths[i] is evaluated only at ranges[i], so the
            # Cartesian product below would report (depth, range) pairs that
            # carry no receiver at all.
            grid_depths, grid_ranges, grid_floors = depths, ranges, seafloor
        else:
            shape = (depths.size, ranges.size)
            grid_depths = np.broadcast_to(depths[:, None], shape)
            grid_ranges = np.broadcast_to(ranges[None, :], shape)
            grid_floors = np.broadcast_to(seafloor[None, :], shape)

        mask = grid_depths > grid_floors

        if np.any(mask):
            flat = int(np.argmax(mask))
            r = float(grid_ranges.ravel()[flat])
            z = float(grid_depths.ravel()[flat])
            sf = float(grid_floors.ravel()[flat])
            warnings.warn(
                f"{self.model_name}: receiver at "
                f"(range={r:.1f} m, depth={z:.1f} m) sits below the local "
                f"seafloor ({sf:.1f} m). Results at that point will "
                f"reflect the model's below-bottom behaviour (e.g. "
                f"infinite TL, PE absorbing layer).",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )

    def _warn_on_range_coverage(
        self, env: 'Environment', receiver: 'Receiver',
    ) -> None:
        """Emit one ``UserWarning`` per range-dependent axis whose extent
        falls short of ``receiver.range_max``. Constant extrapolation is
        what every downstream writer / interpolator does in that case;
        this surfaces it instead of leaving it silent.
        """
        r_target = float(receiver.range_max)
        if r_target <= 0:
            return

        def _check(axis_name: str, axis_max: float) -> None:
            if axis_max < r_target:
                warnings.warn(
                    f"{self.model_name}: {axis_name} extent "
                    f"({axis_max:.1f} m) is shorter than receiver.range_max "
                    f"({r_target:.1f} m); values beyond {axis_max:.1f} m are "
                    f"constant-extrapolated from the last sample.",
                    UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
                )

        if env.has_range_dependent_bathymetry:
            _check("env.bathymetry", float(env.bathymetry.ranges[-1]))
        if env.ssp.is_range_dependent:
            _check("env.ssp.ranges", float(env.ssp.ranges[-1]))
        if env.bottom.is_range_dependent:
            _check("env.bottom.ranges", float(env.bottom.ranges[-1]))
        if env.altimetry is not None and env.altimetry.n_ranges > 1:
            _check("env.altimetry ranges", float(env.altimetry.ranges[-1]))

    def compute_tl(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        *,
        run_mode: Optional['RunMode'] = None,
    ) -> Union[Result, ResultStack]:
        """Compute transmission loss (thin wrapper around ``run``).

        Parameters
        ----------
        env : Environment
            Ocean environment.
        source : Source
            Acoustic source.
        receiver : Receiver
            Receiver grid. Required — depth/range resolution is a physical
            decision and is not auto-generated.
        run_mode : RunMode, optional
            ``COHERENT_TL`` (default), ``INCOHERENT_TL`` or
            ``SEMICOHERENT_TL``. Other modes raise — call ``model.run()``
            directly for those.

        Returns
        -------
        result : Result or ResultStack
            Transmission loss field, or a stack of them over source depth.
            The stacking is done by the readers, not by the caller: a multi-depth
        ``.shd`` / ``.arr`` / ``.ray`` is split into one slab per source
        depth and bundled (``uacpy/io/oalib_reader.py``), so a run over a
        ``Source`` with more than one depth hands back a ``ResultStack``
        over ``source_depth``. ``ResultStack`` is not a ``Result``
        subclass, so a caller that annotates the result has to name both.

        Examples
        --------
        >>> bellhop = Bellhop()
        >>> rcv = uacpy.Receiver(depths=np.linspace(0, env.depth, 50),
        ...                       ranges=np.linspace(100, 10_000, 100))
        >>> tl = bellhop.compute_tl(env, source, rcv)
        """
        # Every ``compute_*`` names the models that declare its run mode.
        # The list cannot be derived here — the mapping is mode -> models,
        # and reading it needs every wrapper imported, which this module is
        # imported *by*. It is instead checked against ``spec.modes`` by
        # ``test_supported_modes.py``, sorted the way that gate sorts, so a
        # model that gains or loses a mode fails there instead of quietly
        # dropping out of the advice.
        if not self.supports_mode(RunMode.COHERENT_TL):
            raise UnsupportedFeatureError(
                self.model_name,
                "transmission loss computation",
                alternatives=['Bellhop', 'Kraken', 'OASP', 'OAST', 'RAM',
                              'Scooter'],
            )
        if run_mode is None:
            run_mode = RunMode.COHERENT_TL
        if run_mode not in (
            RunMode.COHERENT_TL, RunMode.INCOHERENT_TL, RunMode.SEMICOHERENT_TL,
        ):
            raise ConfigurationError(
                f"compute_tl() got run_mode={run_mode}; only COHERENT_TL / "
                f"INCOHERENT_TL / SEMICOHERENT_TL are accepted. Call "
                f"{self.model_name}.run(run_mode=…) for other modes."
            )
        return self.run(env, source, receiver, run_mode=run_mode)

    def compute_rays(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
    ) -> Union[Result, ResultStack]:
        """Compute ray paths (thin wrapper around ``run``).

        ``receiver`` is required — the receiver grid defines the ray-box
        extent and recording locations and is not auto-generated.

        Returns
        -------
        result : Result or ResultStack
            Ray paths, or a stack of them over source depth.
            The stacking is done by the readers, not by the caller: a multi-depth
        ``.shd`` / ``.arr`` / ``.ray`` is split into one slab per source
        depth and bundled (``uacpy/io/oalib_reader.py``), so a run over a
        ``Source`` with more than one depth hands back a ``ResultStack``
        over ``source_depth``. ``ResultStack`` is not a ``Result``
        subclass, so a caller that annotates the result has to name both.

        Examples
        --------
        >>> bellhop = Bellhop()
        >>> rcv = uacpy.Receiver(depths=np.array([env.depth / 2]),
        ...                       ranges=np.linspace(0, 10_000, 50))
        >>> rays = bellhop.compute_rays(env, source, rcv)
        """
        if not self.supports_mode(RunMode.RAYS):
            raise UnsupportedFeatureError(
                self.model_name,
                "ray path computation",
                alternatives=['Bellhop'],
            )
        return self.run(env, source, receiver, run_mode=RunMode.RAYS)

    def compute_arrivals(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
    ) -> Union[Result, ResultStack]:
        """
        Compute the arrival structure (convenience wrapper around ``run``).

        Parameters
        ----------
        env : Environment
            Ocean environment.
        source : Source
            Acoustic source.
        receiver : Receiver
            Receiver array.

        Returns
        -------
        result : Result or ResultStack
            Arrival data, or a stack of it over source depth.
            The stacking is done by the readers, not by the caller: a multi-depth
        ``.shd`` / ``.arr`` / ``.ray`` is split into one slab per source
        depth and bundled (``uacpy/io/oalib_reader.py``), so a run over a
        ``Source`` with more than one depth hands back a ``ResultStack``
        over ``source_depth``. ``ResultStack`` is not a ``Result``
        subclass, so a caller that annotates the result has to name both.

        Raises
        ------
        UnsupportedFeatureError
            If the model does not support arrival computation.

        Examples
        --------
        >>> bellhop = Bellhop()
        >>> arrivals = bellhop.compute_arrivals(env, source, receiver)
        """
        if not self.supports_mode(RunMode.ARRIVALS):
            raise UnsupportedFeatureError(
                self.model_name,
                "arrival computation",
                alternatives=['Bellhop']
            )
        return self.run(env, source, receiver, run_mode=RunMode.ARRIVALS)

    def compute_modes(
        self,
        env: Environment,
        source: Source,
        n_modes: Optional[int] = None,
    ) -> Result:
        """
        Compute normal modes (convenience wrapper around ``run``).

        Parameters
        ----------
        env : Environment
            Ocean environment. A range-dependent env is collapsed to range-
            independent (with a warning) before the mode solve.
        source : Source
            Acoustic source (used for frequency).
        n_modes : int, optional
            Number of modes to compute. If ``None``, all modes are computed.

        Returns
        -------
        result : Result
            :class:`Modes` instance.

        Raises
        ------
        UnsupportedFeatureError
            If the model does not support mode computation.

        Notes
        -----
        This is the one ``compute_*`` wrapper that takes no ``receiver``:
        normal modes are receiver-independent depth eigenfunctions, so there is
        nothing for the caller to position. A placeholder receiver is fabricated
        internally only to satisfy the shared ``run()`` recipe (the Kraken
        backend overrides it with its own dense depth grid).

        Examples
        --------
        >>> kraken = Kraken()
        >>> modes = kraken.compute_modes(env, source, n_modes=50)
        >>> wavenumbers = modes.k
        >>> mode_shapes = modes.phi
        """
        if not self.supports_mode(RunMode.MODES):
            raise UnsupportedFeatureError(
                self.model_name,
                "normal mode computation",
                alternatives=['Kraken']
            )

        if n_modes is not None and (
                isinstance(n_modes, bool)
                or not isinstance(
                    n_modes, (int, float, np.integer, np.floating))):
            raise ConfigurationError(
                f"compute_modes: n_modes must be an int or None; got "
                f"{type(n_modes).__name__}. Unlike the other compute_* "
                f"wrappers, compute_modes takes no receiver (normal modes are "
                f"receiver-independent) — its third argument is n_modes. Pass "
                f"it by keyword: compute_modes(env, source, n_modes=...).")
        # Refuse a cap that is not a whole number. The backend applies it as
        # int(n_modes) — Kraken._compute_modes_impl copies the model with it —
        # which truncates toward zero, so a fractional value would silently
        # run a different cap than asked for and a non-finite one would raise
        # from int() with no context.
        if n_modes is not None and not (
                np.isfinite(n_modes) and float(n_modes).is_integer()):
            raise ConfigurationError(
                f"compute_modes: n_modes must be a whole number of modes or "
                f"None; got {n_modes!r}. The cap is applied as int(n_modes), "
                f"which truncates toward zero.")

        # The env is passed through as-is: the implementation's run() path
        # projects it exactly once (Kraken's MODES path reduces a range-
        # dependent env to its r=0 profile and then projects), so a second
        # projection here would collapse and warn twice.
        return self._compute_modes_impl(env, source, n_modes)

    def _compute_modes_impl(self, env, source, n_modes):
        """
        Model-specific mode computation implementation.

        Caller (compute_modes) already guaranteed RunMode.MODES is supported,
        so a model reaching this base implementation declares the mode without
        implementing the hook. There is no usable generic fallback: normal
        modes are receiver-independent, so the depth grid is the solver's to
        choose, and ``n_modes`` is not part of the ``run()`` contract
        signature :meth:`__init_subclass__` enforces — a mode solver written
        to that signature could not receive it.
        """
        raise NotImplementedError(
            f"{self.model_name} declares RunMode.MODES but does not override "
            f"_compute_modes_impl(env, source, n_modes). Implement it: pick "
            f"the depth grid the mode solver needs and dispatch to the "
            f"binary from there."
        )

    def compute_eigenrays(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
    ) -> Union[Result, ResultStack]:
        """Compute eigenrays — rays that arrive at the receiver(s).

        Thin wrapper around ``run(run_mode=RunMode.EIGENRAYS)``. Returns
        the raw :class:`Rays` from the solver — or, for a multi-depth
        ``Source`` on a model that loops the modes in Python rather than
        stacking them itself (``Bellhop``), a ``ResultStack`` of them over
        ``source_depth``. This is the one ``compute_*`` mode that stacks.
        For a single-point target build a 1-point ``Receiver`` first:

        >>> receiver = uacpy.Receiver(depths=[30.0], ranges=[2000.0])
        >>> rays = bellhop.compute_eigenrays(env, source, receiver)
        >>> close = rays.top_n_by_miss(8).truncate_at_receiver()
        >>> direct = rays.filter_by_bounces(kind='direct')
        >>> within = rays.filter_by_miss_distance(max_miss=15.0)
        """
        if not self.supports_mode(RunMode.EIGENRAYS):
            raise UnsupportedFeatureError(
                self.model_name,
                "eigenray computation",
                alternatives=['Bellhop'],
            )
        return self.run(env, source, receiver, run_mode=RunMode.EIGENRAYS)

    def compute_reflection(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
    ) -> Result:
        """Compute plane-wave reflection coefficients.

        Dispatches to ``run(run_mode=RunMode.REFLECTION)``. Models that
        do not declare ``RunMode.REFLECTION`` in ``supported_modes``
        (everything except Bounce, OASR, OASES) raise
        :class:`UnsupportedFeatureError`.
        """
        if not self.supports_mode(RunMode.REFLECTION):
            raise UnsupportedFeatureError(
                self.model_name,
                "reflection coefficient computation",
                alternatives=['Bounce', 'OASR'],
            )
        return self.run(env, source, receiver, run_mode=RunMode.REFLECTION)

    def compute_time_series(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        *,
        source_waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        output_duration: Optional[float] = None,
    ) -> Result:
        """Compute time-domain pressure p(t) at the receiver(s).

        Forwards ``source_waveform``, ``sample_rate`` and
        ``output_duration`` to ``run(run_mode=RunMode.TIME_SERIES)``.
        ``output_duration`` (seconds) sets the synthesised time window for
        the broadband synthesizers (Bellhop / RAM / Scooter / Kraken /
        OASP) — e.g. the length of an animation; SPARC ignores all three
        (it builds p(t) from its native ``pulse_type`` and ``t_max``), and
        every other TIME_SERIES model requires the waveform/rate.
        """
        if not self.supports_mode(RunMode.TIME_SERIES):
            raise UnsupportedFeatureError(
                self.model_name,
                "time-series computation",
                alternatives=['Bellhop', 'Kraken', 'OASP', 'OASSP', 'RAM',
                              'SPARC', 'Scooter'],
            )
        return self.run(
            env, source, receiver,
            run_mode=RunMode.TIME_SERIES,
            source_waveform=source_waveform,
            sample_rate=sample_rate,
            output_duration=output_duration,
        )

    def compute_transfer_function(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        *,
        frequencies: Optional[np.ndarray] = None,
    ) -> Result:
        """Compute broadband complex transfer function H(f).

        Dispatches to ``run(run_mode=RunMode.BROADBAND)``. Pass
        ``frequencies=`` to override ``source.frequencies`` for the
        sweep.
        """
        if not self.supports_mode(RunMode.BROADBAND):
            raise UnsupportedFeatureError(
                self.model_name,
                "broadband transfer-function computation",
                alternatives=['Bellhop', 'Kraken', 'OASP', 'OASSP', 'RAM',
                              'Scooter'],
            )
        return self.run(
            env, source, receiver,
            run_mode=RunMode.BROADBAND,
            frequencies=frequencies,
        )

    def compute_covariance(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
    ) -> Result:
        """Compute hydrophone-array covariance matrix C(f, i, j).

        Dispatches to ``run(run_mode=RunMode.COVARIANCE)``. Two models
        declare this mode: OASN returns the noise-field covariance, OASS the
        reverberant-field covariance.
        """
        if not self.supports_mode(RunMode.COVARIANCE):
            raise UnsupportedFeatureError(
                self.model_name,
                "covariance-matrix computation",
                alternatives=['OASN', 'OASS'],
            )
        return self.run(env, source, receiver, run_mode=RunMode.COVARIANCE)

    def compute_replicas(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
    ) -> Result:
        """Compute replica fields at the array elements per candidate
        source position (matched-field-processing templates).

        Dispatches to ``run(run_mode=RunMode.REPLICA)``. Currently
        OASN is the only model declaring this mode.
        """
        if not self.supports_mode(RunMode.REPLICA):
            raise UnsupportedFeatureError(
                self.model_name,
                "replica-field computation",
                alternatives=['OASN'],
            )
        return self.run(env, source, receiver, run_mode=RunMode.REPLICA)

    def _resolve_executable(self, executable, find, *,
                            label: Optional[str] = None) -> Path:
        """Store the user's ``executable`` arg verbatim and return the binary.

        Keeps ``self.executable`` exactly as passed (``None`` when
        auto-detected) so ``copy()`` / ``__repr__`` — which read every
        constructor knob off ``self.<param>`` — round-trip the *intent*: a
        clone re-resolves the binary instead of re-pinning an
        already-resolved absolute path.

        Parameters
        ----------
        executable : str or Path, optional
            The user's constructor argument, verbatim.
        find : callable
            Zero-argument callable that locates the binary when
            ``executable`` is ``None`` (typically a
            :meth:`_find_executable_in_paths` closure).
        label : str, optional
            Model label for :class:`ExecutableNotFoundError`. Defaults to
            ``self.model_name``.
        """
        self.executable = Path(executable) if executable is not None else None
        exe = self.executable if self.executable is not None else find()
        if not _is_runnable(exe):
            raise ExecutableNotFoundError(
                label or self.model_name, str(exe),
                reason=(None if not exe.exists() else
                        'not an executable file' if not exe.is_file() else
                        'the execute permission is not set'),
            )
        return exe

    def _find_executable_in_paths(
        self,
        names,
        bin_subdirs=None,
        dev_subdir: Optional[str] = None,
        try_exe_suffix: bool = True,
    ) -> Path:
        """
        Find a model executable by searching standard locations.

        Search order:
            1. uacpy/bin/<bin_subdir>/<name>[+.exe] for each combination
            2. uacpy/third_party/<dev_subdir>/bin (development location)
            3. System PATH

        Parameters
        ----------
        names : str or list of str
            Executable name(s) to try, in preference order.
        bin_subdirs : str or list of str, optional
            Subdirectory/ies under uacpy/bin/. Default 'oalib'.
        dev_subdir : str, optional
            Subdirectory under uacpy/third_party/ (e.g. 'Acoustics-Toolbox/Kraken',
            'oases'). If given, also checks <dev_subdir>/bin and <dev_subdir>/.
        try_exe_suffix : bool, optional
            If True, also try "<name>.exe". Default True.

        Raises
        ------
        ExecutableNotFoundError
        """
        if isinstance(names, str):
            names = [names]
        if bin_subdirs is None:
            bin_subdirs = ['oalib']
        elif isinstance(bin_subdirs, str):
            bin_subdirs = [bin_subdirs]

        base_dir = _PACKAGE_DIR
        candidates = []
        for name in names:
            variants = [name]
            if try_exe_suffix and not name.endswith('.exe'):
                variants.append(name + '.exe')
            for v in variants:
                for sd in bin_subdirs:
                    candidates.append(base_dir / 'bin' / sd / v)
                if dev_subdir:
                    candidates.append(base_dir / 'third_party' / dev_subdir / 'bin' / v)
                    candidates.append(base_dir / 'third_party' / dev_subdir / v)

        # A dud candidate is skipped rather than selected: an earlier search
        # location holding an unexpanded LFS pointer or a half-extracted file
        # must not shadow the working build further down the list.
        for path in candidates:
            if _is_runnable(path):
                return path

        for name in names:
            variants = [name]
            if try_exe_suffix and not name.endswith('.exe'):
                variants.append(name + '.exe')
            for v in variants:
                found = shutil.which(v)
                if found:
                    return Path(found)

        raise ExecutableNotFoundError(
            self.model_name,
            names[0],
            search_paths=[str(p) for p in candidates],
        )

    def _run_subprocess(
        self,
        cmd,
        cwd,
        timeout: Optional[float] = None,
        stdin_input: Optional[str] = None,
        env: Optional[dict] = None,
        check: bool = True,
    ):
        """
        Run an external binary and raise ModelExecutionError on failure.

        All Fortran acoustic binaries are spawned through this helper so that
        failures surface as ``ModelExecutionError`` with stdout/stderr
        attached, and so every child inherits a raised ``RLIMIT_STACK``.
        Several Acoustics-Toolbox binaries (notably SPARC, whose ``MARCH``
        declares twelve automatic ``COMPLEX(NTot1)`` arrays at
        ``Scooter/sparc.f90:354-355``) put large working arrays on the stack;
        the default 8 MB Linux soft stack segfaults them on first use.

        Parameters
        ----------
        cmd : list
            Command argv (str-able elements).
        cwd : path-like
            Working directory for the subprocess.
        timeout : float, optional
            Max seconds before raising.
        stdin_input : str, optional
            Text fed to the subprocess's stdin.
        env : dict, optional
            Environment variables for the subprocess.
        check : bool, optional
            If True (default), raise on non-zero return code.

        Returns
        -------
        subprocess.CompletedProcess
        """
        if timeout is None:
            timeout = getattr(self, 'timeout', 600.0)
        cmd_str = ' '.join(str(c) for c in cmd)
        self._log(f"Running: {cmd_str}", level='debug')

        # start_new_session puts the child in its own process group so a
        # timeout can SIGTERM the whole tree, not just the direct child.
        proc = None
        try:
            proc = subprocess.Popen(
                [str(c) for c in cmd],
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE if stdin_input is not None else None,
                text=True,
                env=env,
                start_new_session=(os.name == 'posix'),
            )
            stdout, stderr = proc.communicate(input=stdin_input, timeout=timeout)
            result = subprocess.CompletedProcess(
                proc.args, proc.returncode, stdout, stderr,
            )
        except OSError as e:
            # ``Popen`` reports a failed chdir with ``filename`` set to the
            # cwd, so a work directory deleted mid-run arrives here naming that
            # directory. Matching on it keeps the failure from being announced
            # as "Executable not found" quoting a path that was never a binary.
            if e.filename is not None and Path(e.filename) == Path(cwd):
                raise ModelExecutionError(
                    self.model_name, return_code=-1, stdout=None,
                    stderr=(f"Work directory is no longer usable: {e}. It was "
                            f"deleted, moved or made inaccessible while the "
                            f"run was in progress."),
                ) from e
            if isinstance(e, FileNotFoundError):
                raise ModelExecutionError(
                    self.model_name, return_code=-1,
                    stdout=None, stderr=f"Executable not found: {e}",
                ) from e
            # A binary that is present but will not exec — permission denied,
            # or ENOEXEC from an unexpanded LFS pointer, a wrong-architecture
            # build or a half-extracted archive — is a broken install, and its
            # remediation is the install script's, so it joins the
            # missing-binary case instead of escaping as a bare OSError.
            if isinstance(e, PermissionError) or e.errno == errno.ENOEXEC:
                raise ExecutableNotFoundError(
                    self.model_name, str(cmd[0]), reason=e.strerror,
                ) from e
            raise
        except subprocess.TimeoutExpired as e:
            # Kill the whole process group, not just the direct child.
            if proc is not None:
                self._terminate_process_group(proc)
            raise ModelExecutionError(
                self.model_name, return_code=-1,
                stdout=(e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout),
                stderr=f"Timed out after {timeout}s",
                timed_out=True,
            ) from e
        except BaseException:
            # start_new_session detaches the child from the terminal's signals,
            # so an interrupt reaches this process alone and the binary would
            # keep running. The wrapper owns the whole tree, so the group is
            # reaped before the exception continues on its way.
            if proc is not None and proc.poll() is None:
                self._terminate_process_group(proc)
            raise

        if check and result.returncode != 0:
            raise ModelExecutionError(
                self.model_name,
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        return result

    @staticmethod
    def _has_shear(boundary) -> bool:
        """True if ``boundary`` carries any non-zero shear speed. Accepts a
        :class:`Bottom` or a surface :class:`BoundaryProperties`."""
        from uacpy.core.bottom import Bottom
        from uacpy.core.surface import Surface
        if boundary is None:
            return False
        if isinstance(boundary, (Bottom, Surface)):
            return boundary.is_elastic
        return getattr(boundary, 'shear_speed', 0.0) > 0

    @staticmethod
    def _collapse_elastic_boundary(boundary, method: str):
        """Collapse elastic shear on ``boundary`` per ``method``.

        ``'fluid'``  : zero shear_speed and shear_attenuation; keep cp / ρ / α.
        ``'vacuum'`` : replace with a vacuum boundary.

        Accepts a :class:`Bottom` (every column's layers + half-space) or a
        surface :class:`BoundaryProperties`, returning the same kind.
        """
        from uacpy.core.bottom import BoundaryProperties, Bottom
        from uacpy.core.surface import Surface

        def _zero_shear(b):
            b.shear_speed = 0.0
            b.shear_attenuation = 0.0

        if method not in ('fluid', 'vacuum'):
            raise ConfigurationError(
                f"Unknown elastic collapse method {method!r}. Use "
                "'fluid' or 'vacuum'."
            )
        if isinstance(boundary, Bottom):
            if method == 'vacuum':
                return Bottom.from_halfspace(
                    BoundaryProperties(acoustic_type='vacuum'))
            b = _copy.deepcopy(boundary)
            for col in b.columns:
                for layer in col.layers:
                    _zero_shear(layer)
                _zero_shear(col.halfspace)
            return b
        if isinstance(boundary, Surface):
            if method == 'vacuum':
                return Surface(properties=[
                    BoundaryProperties(acoustic_type='vacuum')])
            b = _copy.deepcopy(boundary)
            for p in b.properties:
                _zero_shear(p)
            return b
        if method == 'vacuum':
            return BoundaryProperties(acoustic_type='vacuum')
        b = _copy.deepcopy(boundary)
        _zero_shear(b)
        return b

    def _project_environment(self, env: 'Environment') -> 'Environment':
        """Return a copy of ``env`` with every unsupported feature collapsed.

        Each per-feature axis is checked against the matching
        ``_supports_*`` flag and reduced via the matching key in the
        configured ``collapse={…}`` dict. Emits one ``UserWarning`` per
        dropped feature.

        Notes
        -----
        The caller's ``env`` is never touched — everything happens on
        ``env.copy()``, so a user can reuse one ``Environment`` across models
        that project it differently. Only ``altimetry``, ``surface``,
        ``bathymetry``, ``ssp`` and ``bottom`` are rewritten;
        ``env.absorption`` and everything else pass through.

        Every branch *narrows* — it removes range dependence, layering or
        shear that the model cannot represent — with one deliberate
        exception: collapsing the bathymetry can deepen the seafloor, and the
        SSP is then extended to match (see below).

        Order matters (the bottom's range axis is collapsed before its layer
        axis), so a subclass that overrides this must call ``super()`` —
        Kraken does — rather than reimplement the sequence.
        """
        e = env.copy()

        if e.altimetry is not None and not self._supports_altimetry:
            method = self._collapse["altimetry"]
            # Defensive: unreachable through either entry point, since a
            # user's ``collapse=`` and a subclass's ``ModelSpec.collapse`` are
            # both checked against ``VALID_COLLAPSE_METHODS['altimetry']``,
            # which holds only 'drop'. Only mutating the private
            # ``_collapse`` gets here. Kept because it becomes live the day a
            # second altimetry method is added, and raising (not asserting)
            # keeps it under ``python -O``.
            if method != 'drop':
                raise ConfigurationError(
                    f"Unknown collapse['altimetry']={method!r}. "
                    "Currently only 'drop' is supported."
                )
            e.altimetry = None
            warnings.warn(
                f"{self.model_name} does not support sea-surface altimetry; "
                f"using flat surface (collapse['altimetry']={method!r}).",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )

        # No model consumes a range-dependent surface deck (the AT family
        # takes one TopOpt boundary, RAM one attenuator, OASES one top
        # half-space), so the collapse is unconditional; ``collapse['surface']``
        # picks the reduction method.
        if e.surface.is_range_dependent:
            method = self._collapse["surface"]
            e.surface = e.surface.collapse(method)
            warnings.warn(
                f"{self.model_name} does not support range-dependent surface "
                f"properties (e.g. a marginal ice zone); collapsed to a single "
                f"boundary (collapse['surface']={method!r}).",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )

        surf_sigma = _max_roughness(e.surface.properties)
        if surf_sigma and not self._supports_rough_surface:
            e.surface = _smooth_surface(e.surface)
            warnings.warn(
                f"{self.model_name} does not carry a rough sea surface into "
                f"its deck; env.surface.roughness={surf_sigma:g} m dropped. "
                f"Use Kraken with a vacuum, rigid or half-space surface, an "
                f"OASES model, or Scooter with a vacuum surface, to keep it.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )

        bot_sigma = _bottom_roughness(e.bottom)
        if bot_sigma and not self._supports_rough_bottom:
            e.bottom = _smooth_bottom(e.bottom)
            warnings.warn(
                f"{self.model_name} does not carry seabed interfacial "
                f"roughness into its deck; env.bottom roughness="
                f"{bot_sigma:g} m dropped. Use Kraken or an OASES model "
                f"to keep it.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )

        if e.has_range_dependent_bathymetry and not self._supports_range_dependent_bathymetry:
            method = self._collapse["bathymetry"]
            new_depth = e.get_representative_depth(method)
            min_d = float(e.bathymetry.depths.min())
            max_d = float(e.bathymetry.depths.max())
            e.bathymetry = Bathymetry(
                ranges=np.array([0.0]), depths=np.array([new_depth]))
            # A collapse method that picks a deeper column than the profile
            # was tabulated for (``'max'``, the default, on a sloping bottom)
            # leaves the SSP short of the new seafloor, which the AT writers
            # reject. Constant-extrapolate the deepest sound speed down to it.
            if e.ssp.depths[-1] < new_depth:
                e.ssp = e.ssp.extend_to(new_depth)
            warnings.warn(
                f"{self.model_name} does not support range-dependent "
                f"bathymetry; collapsed to {new_depth:.1f} m "
                f"(method={method!r}, range {min_d:.1f}–{max_d:.1f} m). "
                f"Override via `collapse={{'bathymetry': "
                f"'min'|'median'|'mean'|'max'|'initial'}}`.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )

        if e.has_range_dependent_ssp and not self._supports_range_dependent_ssp:
            method = self._collapse["ssp"]
            e.ssp = e.ssp.collapse(method)
            warnings.warn(
                f"{self.model_name} does not support range-dependent SSP; "
                f"collapsed to 1-D (collapse['ssp']={method!r}).",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )

        # Bottom: two orthogonal axes. Collapse the range axis first (to a
        # single column, keeping its layers), then flatten the layer axis if
        # the model can't take layers — leaving, for a model that supports RD
        # but not layers (Bellhop), a range-dependent half-space bottom.
        if e.bottom.is_range_dependent and not self._supports_range_dependent_bottom:
            method = self._collapse["bottom_range"]
            e.bottom = e.bottom.select_range(method)
            warnings.warn(
                f"{self.model_name} does not support range-dependent bottoms; "
                f"reduced to a single column "
                f"(collapse['bottom_range']={method!r}).",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )

        if e.bottom.is_layered and not self._supports_layered_bottom:
            method = self._collapse["bottom_layers"]
            e.bottom = e.bottom.collapse(layers=method)
            warnings.warn(
                f"{self.model_name} does not support layered (depth-dependent) "
                f"bottoms; flattened each column to a half-space "
                f"(collapse['bottom_layers']={method!r}).",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )

        if not self._supports_elastic_media:
            collapsed_at = []
            if e.surface is not None and self._has_shear(e.surface):
                e.surface = self._collapse_elastic_boundary(
                    e.surface, self._collapse["elastic"],
                )
                collapsed_at.append('surface')
            if e.bottom.is_elastic:
                e.bottom = self._collapse_elastic_boundary(
                    e.bottom, self._collapse["elastic"],
                )
                collapsed_at.append('bottom')
            if collapsed_at:
                method = self._collapse["elastic"]
                where = '/'.join(collapsed_at)
                warnings.warn(
                    f"{self.model_name} does not support elastic media; "
                    f"collapsed shear properties on {where} "
                    f"(collapse['elastic']={method!r}).",
                    UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
                )

        return e

    def _attach_output_paths(
        self,
        result: 'Result',
        work_dir: Path,
        base_name: str,
        *,
        primary_files: tuple = (),
    ) -> None:
        """Attach work-dir output paths to ``result.metadata``.

        The paths are recorded iff the run's scratch survives, i.e. iff
        ``self.cleanup`` is False. With ``cleanup=True`` the work dir is wiped
        immediately after ``run()`` returns, so no keys are written: the
        absence of a ``*_file`` / ``prt_file`` key is the documented signal
        that the directory has been cleaned up (DOCUMENTATION.md §8).

        ``primary_files`` must name only the outputs *this* run produces —
        passing every suffix a model can emit would re-attach an earlier
        run's leftovers in a pinned work dir.

        Otherwise, for each ``(key, suffix)`` in ``primary_files`` set
        ``result.metadata[key] = str(work_dir / f'{base_name}{suffix}')``
        when the file exists. Also set ``'prt_file'`` from the binary's
        diagnostic log when present.
        """
        if self.cleanup:
            return
        for key, suffix in primary_files:
            path = work_dir / f'{base_name}{suffix}'
            if path.exists():
                result.metadata[key] = str(path)
        prt_path = work_dir / f'{base_name}.prt'
        if prt_path.exists():
            result.metadata['prt_file'] = str(prt_path)

    def _require_output(self, candidates, *, what: str, process=None,
                        hint: str = '', prt_base: Optional[str] = None,
                        work_dir=None) -> Path:
        """Return the first of ``candidates`` that exists and is non-empty.

        None of them means the binary failed without a usable exit status —
        the OALIB/OASES/RAM engines routinely exit 0 after writing nothing.
        The raised :class:`ModelExecutionError` names every checked path and
        carries whatever diagnostics the engine family has:

        * ``process`` — the binary's :class:`subprocess.CompletedProcess`;
          its stdout/stderr tails are quoted (OASES and the RAM family write
          no print file, so the streams are the only record of the failure).
        * ``prt_base`` + ``work_dir`` — appends the tail of
          ``<work_dir>/<prt_base>.prt`` (the Acoustics-Toolbox models log
          fatal errors there rather than on their streams).
        """
        for candidate in candidates:
            path = Path(candidate)
            if path.exists() and path.stat().st_size > 0:
                return path
        checked = ', '.join(str(c) for c in candidates)
        exc = ModelExecutionError(
            self.model_name,
            return_code=getattr(process, 'returncode', 0),
            stdout=_tail(getattr(process, 'stdout', None)),
            stderr=(
                f"{self.model_name} did not produce {what}. Checked: "
                f"{checked}." + (f" {hint}" if hint else "")
                + _stream_block('binary stderr',
                                getattr(process, 'stderr', None))
            ),
        )
        if prt_base is not None and work_dir is not None:
            self._attach_prt_tail(exc, work_dir, prt_base)
        raise exc

    @staticmethod
    def _attach_prt_tail(exc, work_dir, base_name, tail_bytes: int = 2000):
        """Append the tail of the binary's ``<base>.prt`` log to ``exc``.

        Acoustics-Toolbox binaries dump fatal errors (``*** FATAL ERROR ***``)
        to ``.prt`` instead of stderr; surface that tail on the raised
        ``ModelExecutionError`` so the user sees the actual cause, not just a
        "check the .prt file" pointer.

        Updates ``exc.message`` (what ``UACPYError.__str__`` renders) **and**
        ``exc.stderr`` (a constructor arg, so the tail survives the pickle
        round-trip that ``run_parallel`` relies on — the rebuilt message is
        re-derived from ``stderr``), keeping ``exc.args`` in sync.
        """
        tail = read_prt(Path(work_dir) / f"{base_name}.prt",
                        tail_bytes=tail_bytes)
        if tail is None:
            return
        block = f"\n\n.prt tail:\n{tail}"
        if getattr(exc, 'message', None) is not None:
            exc.message += block
        if hasattr(exc, 'stderr'):
            exc.stderr = (exc.stderr + block) if exc.stderr else f".prt tail:\n{tail}"
        head = getattr(exc, 'message', None) or (
            f"{exc.args[0]}{block}" if exc.args else f"{exc}{block}")
        exc.args = (head,) + exc.args[1:]

    # ERROUT messages that describe a physical outcome uacpy models
    # explicitly, rather than a run failure. Empty here: which messages are
    # benign is solver-specific, so each model declares its own.
    _BENIGN_FORTRAN_FATALS: tuple = ()

    def _raise_on_fortran_fatal(self, result, work_dir, base_name):
        """Raise when a binary reported a fatal error but exited 0.

        Acoustics-Toolbox errors funnel through ``ERROUT``
        (``misc/FatalError.f90:18,30``), which writes ``*** FATAL ERROR ***``
        to the ``.prt`` and then ends in ``STOP '<string>'`` — a *character*
        stop code, which gfortran exits **0** for. A return-code test therefore
        never fires, and the binary leaves its output file untouched: with a
        pinned ``work_dir`` the previous run's ``.mod`` / ``.shd`` is still on
        disk and would be read as this run's answer.

        stderr is the authoritative signal because it belongs to this process;
        the ``.prt`` is checked too since it names the actual cause.

        The test is on the *form* of the stop, not on any banner text. A
        character stop code is how both toolchains report an abnormal end, and
        their banners do not share a marker: AT ends at
        ``STOP 'Fatal Error: …'`` via ``ERROUT`` but also stops directly with
        ``STOP 'ERROR IN KRAKENC: …'`` and ``STOP 'FATAL ERROR in BandPass: …'``,
        while OASES uses 46 distinct banners of which only 19 carry ``***`` —
        26 use ``>>> … <<<`` and one is bare ``'INVALID INPATCH'``. Matching
        ``***`` caught well under half of them, and a missed stop is read as
        success off whatever output file happens to be on disk.
        """
        stderr = result.stderr or ''
        prt = read_prt(Path(work_dir) / f"{base_name}.prt")
        fatal_stderr = bool(re.search(r'^\s*STOP\s+\S', stderr, re.MULTILINE))
        if not fatal_stderr and (
                prt is None or '*** FATAL ERROR ***' not in prt):
            return
        if prt and any(m in prt for m in self._BENIGN_FORTRAN_FATALS):
            return
        exc = ModelExecutionError(
            self.model_name, return_code=0, stdout=result.stdout,
            stderr=stderr or "binary reported a fatal error and exited 0",
        )
        self._attach_prt_tail(exc, work_dir, base_name)
        raise exc

    @staticmethod
    def _terminate_process_group(proc) -> None:
        """Reap ``proc`` and everything it spawned.

        Binaries are launched with ``start_new_session``, so they sit in their
        own process group and survive a signal delivered only to this process.
        SIGTERM to the group first, escalating to SIGKILL if it does not exit.
        """
        if os.name == 'posix':
            try:
                pgid = os.getpgid(proc.pid)
                os.killpg(pgid, signal.SIGTERM)
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(pgid, signal.SIGKILL)
                    proc.wait()
                return
            except (ProcessLookupError, PermissionError):
                pass
        proc.kill()
        proc.wait()

    def _run_and_attach_prt(self, cmd, work_dir, base_name, *,
                            timeout: Optional[float] = None,
                            env: Optional[dict] = None,
                            stale_outputs: tuple = ()):
        """Run a Fortran/AT binary via :meth:`_run_subprocess`, appending the
        ``<base>.prt`` tail to a :class:`ModelExecutionError` on failure and
        logging stdout when verbose. Returns the ``CompletedProcess``. Shared
        by every model's binary-launch wrapper.

        The AT engines hold the file root in ``CHARACTER(LEN=80)`` buffers,
        so a root longer than 80 characters is silently truncated; launching
        with ``work_dir`` as cwd and a short relative ``base_name`` keeps
        every root far inside that limit.

        ``stale_outputs`` lists the suffixes (``'.shd'``, ``'.arr'``, …) this
        binary may write under ``base_name``. They are removed before launch
        so a pinned work dir cannot hand an earlier run's output back as this
        run's answer — the same reason the ``.prt`` is cleared.
        """
        # Anything left under base_name by an earlier run in a pinned work_dir
        # would be read as this run's, so clear it before launching.
        for suffix in ('.prt',) + tuple(stale_outputs):
            Path(work_dir).joinpath(f"{base_name}{suffix}").unlink(missing_ok=True)
        try:
            result = self._run_subprocess(cmd, cwd=work_dir, timeout=timeout, env=env)
        except ModelExecutionError as exc:
            self._attach_prt_tail(exc, work_dir, base_name)
            raise
        self._raise_on_fortran_fatal(result, work_dir, base_name)
        self._warn_on_prt_warnings(work_dir, base_name)
        if result.stdout:
            self._log(f"{self.model_name} output:\n{result.stdout}", level='debug')
        return result

    def _warn_on_prt_warnings(self, work_dir, base_name) -> None:
        """Surface the binary's own non-fatal ``Warning in ...`` lines.

        The AT binaries write both fatals and *non-fatal* diagnoses to the
        ``.prt``. Only the fatals were read (``_attach_prt_tail``, on the
        exception path), so a run the solver itself diagnosed came back at
        exit 0 with a full-size result and nothing said — measured, BELLHOP
        writes ``Warning in BELLHOP : Too few beams`` and uacpy emitted zero
        warnings while returning a TL field the binary had just called
        under-sampled.

        These are the solver's words, not uacpy's, so they are passed through
        verbatim rather than reinterpreted.
        """
        text = read_prt(Path(work_dir) / f"{base_name}.prt", tail_bytes=200000)
        if not text:
            return
        seen, lines = set(), []
        for raw in text.splitlines():
            line = raw.strip()
            if line.lower().startswith('warning in') and line not in seen:
                seen.add(line)
                lines.append(line)
        if lines:
            joined = "\n  ".join(lines)
            warnings.warn(
                f"{self.model_name} reported {len(lines)} non-fatal "
                f"warning(s) in its .prt log:\n  {joined}",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP)

    def _result_kwargs(
        self,
        source: 'Source',
        *,
        backend: Optional[str] = None,
        frequencies: Optional[Union[float, np.ndarray]] = None,
        phase_reference: Optional[str] = None,
        **extra,
    ) -> dict:
        """Pre-built kwargs for any :mod:`uacpy.core.results` constructor.

        ``frequencies`` is auto-wrapped to a 1-D ndarray (length ≥ 1) when
        scalar; every wrapper passes a value (broadband / time-series
        results carry the full frequency grid their synthesis derived,
        SPARC's native p(t) its pulse centre frequency), so ``None`` —
        passed through unchanged — occurs only for a caller that opts out
        of the stamp. Anything in ``extra`` is stored on the result's
        ``metadata`` ad-hoc bag.

        ``backend`` names the concrete binary that ran and is lowercase
        across the package (``'scooter'``, ``'oast'``, ``'mpiramS'``, …), so
        the default lowercases the class name rather than mixing conventions.

        Identification only: this stamps who produced the result, never what
        it means numerically. Nothing here — nor anywhere else in
        :class:`PropagationModel` — rescales, normalises or re-references the
        payload, so the amplitude a wrapper stores is whatever its reader
        returned. The one cross-model *convention* the base class carries is
        ``phase_reference`` (see
        :class:`~uacpy.core.results._base.PhaseReference`), which is about
        phase, not level; the absolute dB reference behind ``Field.db`` is a
        per-model property and is not asserted here.
        """
        kw = dict(
            model=self.model_name,
            backend=backend or self.model_name.lower(),
            source_depths=np.atleast_1d(np.asarray(
                getattr(source, 'depths', []), dtype=float
            )),
            frequencies=(np.atleast_1d(np.asarray(frequencies, dtype=float))
                         if frequencies is not None else None),
            model_source=self.provenance,
            metadata=dict(extra),
        )
        if phase_reference is not None:
            # Coerced to the PhaseReference enum member, so wrapper-built
            # results carry the same type as the synthesis helpers stamp.
            kw['phase_reference'] = PhaseReference(phase_reference)
        return kw

    def _stamp_result(self, result, source: 'Source', *,
                      backend: Optional[str] = None,
                      frequencies: Optional[Union[float, np.ndarray]] = None,
                      phase_reference: Optional[str] = None):
        """Stamp the cross-model identification fields onto a ``Result`` that an
        io reader already constructed (so it couldn't take them as kwargs).
        Mirrors :meth:`_result_kwargs` for the reader-built path (Scooter,
        SPARC); leaves ``result.metadata`` untouched. Returns ``result``."""
        kw = self._result_kwargs(source, backend=backend,
                                 frequencies=frequencies,
                                 phase_reference=phase_reference)
        for attr in ('model', 'backend', 'source_depths', 'frequencies',
                     'model_source'):
            setattr(result, attr, kw[attr])
        if phase_reference is not None:
            result.phase_reference = PhaseReference(phase_reference)
        return result

    @staticmethod
    def _speed_bounds(env: 'Environment'):
        """Slowest / fastest compressional speeds (m/s) anywhere in ``env``.

        Water column plus every bottom layer and half-space that carries
        geoacoustics. An Environment always carries an SSP (``ssp=None``
        resolves to the isovelocity default), so the list is never empty.
        """
        # Every profile, not just the one at r=0: a range-dependent SSP can
        # hold its extremes in any column.
        speeds = [float(c) for c in np.asarray(env.ssp.data).ravel()
                  if np.isfinite(c)]
        speeds.extend(c for c in env.bottom.all_sound_speeds() if c)
        return float(min(speeds)), float(max(speeds))

    def _resolve_c_max(self, env: 'Environment') -> Optional[float]:
        """Fastest compressional speed (m/s) anywhere in ``env`` — water
        column plus sediment layers and geoacoustic half-spaces.

        Wrappers stamp it on broadband / complex-pressure results as
        ``c_max``: at long range the earliest arrival is the
        bottom-refracted path travelling at the fastest *seabed* speed, and
        ``Field.to_time_trace`` anchors its synthesis window at
        ``r / c_max``, ahead of that arrival. The anchor contract there
        admits only physical speeds — never an algorithmic reference such
        as a PE expansion point.

        Always a speed, never ``None``: :meth:`_speed_bounds` returns a
        2-tuple on every path, and its own docstring records why the list it
        reduces is never empty (an Environment always carries an SSP). The
        return type stays ``Optional`` for subclasses — ``RAM`` overrides
        this and keeps a fallback of its own.
        """
        return float(self._speed_bounds(env)[1])

    #: Whether the engine resolves the field THROUGH the sediment layers it
    #: is given (Kraken, Scooter, SPARC, OASES mesh them as media), so a
    #: receiver — or a buried source — inside the seabed is a supported
    #: geometry down to the deepest interface. Ray models stop at the
    #: seafloor; RAM keeps the seafloor by its own override even though its
    #: PE marches deeper (see :meth:`RAM._max_receiver_depth`).
    _receivers_reach_sediment: bool = False

    def _max_receiver_depth(self, env: 'Environment') -> float:
        """Deepest receiver depth this model can resolve the field at: the
        seafloor, or the deepest modelled interface when
        ``_receivers_reach_sediment`` (see :meth:`_total_media_depth`).

        The value gates source and receiver depths asymmetrically in
        :meth:`_validate_geometry`: a source below it raises, a receiver
        below it only warns. Changing it therefore changes which source
        placements are legal, not just which receivers warn.
        """
        if self._receivers_reach_sediment:
            return self._total_media_depth(env)
        return float(env.depth)

    def _source_below_domain_note(self, env: 'Environment',
                                  resolvable_depth: float):
        """Extra paragraph for the source-below-``_max_receiver_depth``
        error, or ``None``.

        The generic message states the depth and the limit; a model whose
        limit sits above what its engine can compute overrides this to say
        which of the two it is, so the user is not left reading a numerical
        limit as a physical one.
        """
        return None

    def _mask_unresolvable_depths(self, result, receiver, media_depth):
        """Restore the caller's depth axis on ``result``, NaN below
        ``media_depth``.

        The finite-element / finite-difference mesh stops at the deepest
        modelled interface, so the binary clamps any deeper receiver onto it.
        Handing those cells back under the depth that was asked for would
        misreport where the field was evaluated — they are no-data.

        Used by the full-waveguide spectral solvers (Scooter, SPARC), which
        mesh through the sediment stack the same way.
        """
        from uacpy.core.results import Field
        depths = np.atleast_1d(np.asarray(receiver.depths, dtype=float))
        d = result.to_dict()
        data = d['data']
        if data.shape[0] != depths.size:
            raise ModelExecutionError(
                self.model_name, return_code=0, stdout=None,
                stderr=(f"{self.model_name} returned {data.shape[0]} depth "
                        f"rows for {depths.size} requested receiver depths; "
                        f"the depth axis cannot be reattached."),
            )
        data[depths > media_depth, ...] = np.nan
        d['coords'] = {**d['coords'], 'depth': depths}
        return Field.from_dict(d)

    def _mask_zero_range_columns(self, data, ranges, singular_term: str):
        """Return ``data`` with the range-axis (axis 1) columns at ``r = 0``
        set to NaN, warning once with the model prefix.

        The zero test mirrors ``abs( Rr ) < realmin`` from ``fieldsco.m:69``;
        :class:`~uacpy.core.receiver.Receiver` rejects negative ranges, so it
        is equivalent to ``r == 0`` for any constructible receiver.
        ``singular_term`` names the 1/√r-type factor that has no value there
        (it completes "... at r = 0, where <singular_term> is singular").
        ``data`` is returned unchanged — and no warning is emitted — when no
        column sits on the source axis. :meth:`_mask_source_axis` applies it
        to a point-source field; SPARC's 'R'/'D' branches call it with
        their own singular term. Every masked engine warns on every run.
        """
        ranges = np.atleast_1d(np.asarray(ranges, dtype=float))
        zero = np.abs(ranges) < np.finfo(float).tiny
        if not zero.any():
            return data
        warnings.warn(
            f"{self.model_name}: {int(zero.sum())} receiver range(s) at "
            f"r = 0, where {singular_term} is singular; those cells are "
            f"returned as NaN (no data). Move the receiver off the source "
            f"axis (e.g. r = 1 m) to get a field value.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
        masked = np.array(data, copy=True)
        if not np.issubdtype(masked.dtype, np.inexact):
            masked = masked.astype(float)
        masked[:, zero, ...] = np.nan
        return masked

    def _mask_source_axis(self, field, source):
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
        the warning come from :meth:`_mask_zero_range_columns`, reading the
        field's own range axis.
        """
        if source.source_type != 'point':
            return field
        masked = self._mask_zero_range_columns(
            field.data, field.coords['range'],
            'the point-source cylindrical-spreading factor 1/sqrt(r)')
        if masked is not field.data:
            field.data = masked
        return field

    def _reject_malformed_irc_bottom(self, env: 'Environment') -> None:
        """Refuse a ``'precalc'`` seabed whose table is not in ``.irc`` layout.

        A ``'precalc'`` bottom is staged verbatim as ``<root>.irc``, which
        ``misc/RefCoef.f90:94-107`` reads as: line 1 ``Title freq``, line 2
        the record count ``NkTab``, then ``NkTab`` records written
        ``(5G15.7,I5)`` — tangential wavenumber ``x``, the complex impedance
        pair ``f`` / ``g``, and a power-of-ten exponent: five reals and an
        integer per record. That is a different format from the
        ``.brc`` / ``.trc`` angle tables (a bare count line, then
        ``theta |R| phase`` rows), and the binary answers the mismatch with a
        bare Fortran I/O abort. Checked on the wrappers that stage the file
        (Kraken, Scooter) before any binary is launched; a missing or unset
        ``reflection_file`` is left to the staging step's own typed errors.
        """
        def _numeric(token: str) -> bool:
            try:
                float(token.replace('D', 'E').replace('d', 'e'))
            except ValueError:
                return False
            return True

        def _reason(lines) -> Optional[str]:
            if len(lines) < 3:
                return (f"only {len(lines)} non-blank line(s); the layout is "
                        f"a Title/freq line, a record count, and the records")
            head = lines[0].split()
            if all(_numeric(t) for t in head):
                return ("line 1 is all-numeric — an .irc starts with "
                        "'Title freq', while a bare record count or a "
                        "theta/|R|/phase row starts a .brc/.trc angle table")
            if not _numeric(head[-1]):
                return ("line 1 carries no trailing frequency — an .irc "
                        "starts with 'Title freq'")
            count = lines[1].split()
            if len(count) != 1 or not count[0].lstrip('+-').isdigit() \
                    or int(count[0]) < 1:
                return (f"line 2 is {lines[1].strip()!r} where the .irc "
                        f"record count NkTab (a single positive integer) "
                        f"belongs")
            record = lines[2].split()
            if len(record) < 6 or not all(_numeric(t) for t in record[:6]):
                return (f"the first record {lines[2].strip()!r} does not "
                        f"carry the (5G15.7,I5) x/f/g/iPow fields — a "
                        f"3-column row is a theta/|R|/phase angle table")
            return None

        for column in env.bottom.columns:
            hs = column.halfspace
            if hs.acoustic_type != 'precalc':
                continue
            table = getattr(hs, 'reflection_file', None)
            if not table or not Path(table).exists():
                continue
            with open(table, 'r', errors='replace') as fh:
                lines = [ln for ln in (fh.readline() for _ in range(64))
                         if ln.strip()][:3]
            reason = _reason(lines)
            if reason:
                raise ConfigurationError(
                    f"{self.model_name}: acoustic_type='precalc' stages "
                    f"{table} as the .irc internal-reflection table, but "
                    f"{reason}. An .irc (BOUNCE's f/g impedance format, "
                    f"misc/RefCoef.f90:94-107) is not a .brc/.trc "
                    f"angle-magnitude-phase table; the binary aborts with a "
                    f"bare Fortran backtrace on the mismatch.",
                    remediation=(
                        "Pass a BOUNCE result.metadata['irc_file'] as "
                        "reflection_file=, or use acoustic_type='file' for "
                        "a theta/|R|/phase angle table (.brc/.trc)."
                    ),
                )

    def _total_media_depth(self, env: 'Environment') -> float:
        """
        Deepest modelled interface (m): the water column plus all sediment
        layer thicknesses. Below this lies the semi-infinite halfspace.

        Full-waveguide spectral solvers (Scooter, SPARC) resolve the field
        through the water and every fluid/elastic sediment layer, so the
        valid receiver range extends to this depth — not merely to
        ``env.depth`` (the seafloor).
        """
        depth = float(env.depth)
        if env.bottom.is_layered:
            depth += env.bottom.max_total_thickness()
        return depth

    def __repr__(self) -> str:
        """``ClassName(arg=val, …)`` showing only constructor params whose
        current value differs from the constructor default.

        Walks ``__init__`` along the MRO (subclasses forward to
        ``super().__init__(**kwargs)``, so the union of named parameters
        across the chain is the full configuration surface). Reads each
        param off ``self.<name>`` — the same contract that powers
        ``model.copy``. Ndarrays and long sequences are summarised so
        the result stays one-line-readable even when a model has many
        knobs.
        """
        bits: List[str] = []
        for name, default in _collect_init_params(type(self)):
            if not hasattr(self, name):
                continue
            # Resolved binary paths are machine-specific and not copy-paste-
            # portable — hide them regardless of value.
            if name in ('executable', 'field_executable'):
                continue
            value = getattr(self, name)
            # ``cleanup`` resolves to ``work_dir is None`` when left at its None
            # default; hide it while it carries that auto value, matching
            # ``copy()``, which hands the ``None`` sentinel back so the clone
            # re-resolves against its own work_dir.
            if name == 'cleanup' and not self._cleanup_explicit:
                continue
            if default is not _NO_DEFAULT and _values_equal(value, default):
                continue
            bits.append(f"{name}={_short_repr(value)}")
        return f"{type(self).__name__}({', '.join(bits)})"


_NO_DEFAULT = object()


def _collect_init_params(cls) -> List[tuple]:
    """Walk ``cls.__mro__`` for every ``__init__`` and collect named
    parameters (excluding ``self`` and ``**kwargs``) in declaration order,
    deduplicated by name (subclass declaration wins).

    Returns a list of ``(name, default_or_NO_DEFAULT)`` pairs. Used by
    :meth:`PropagationModel.__repr__` and parallels what
    :meth:`PropagationModel.copy` introspects.
    """
    import inspect as _inspect

    seen: Dict[str, object] = {}
    order: List[str] = []
    for klass in cls.__mro__:
        if klass is object:
            continue
        init = klass.__dict__.get('__init__')
        if init is None:
            continue
        try:
            sig = _inspect.signature(init)
        except (TypeError, ValueError):
            continue
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            if param.kind == _inspect.Parameter.VAR_KEYWORD:
                continue
            if param.kind == _inspect.Parameter.VAR_POSITIONAL:
                continue
            if name in seen:
                continue
            seen[name] = (
                param.default if param.default is not _inspect.Parameter.empty
                else _NO_DEFAULT
            )
            order.append(name)
    return [(name, seen[name]) for name in order]


def _values_equal(a, b) -> bool:
    """Compare two configuration values, tolerating ndarray equality."""
    if a is b:
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(a), np.asarray(b)))
        except Exception:
            return False
    try:
        return bool(a == b)
    except Exception:
        return False


def _short_repr(value, *, list_limit: int = 6) -> str:
    """Compact ``repr`` for a constructor value, summarising big arrays.

    Used by :meth:`PropagationModel.__repr__` so ``print(model)`` stays
    short even when a knob holds a large ndarray or list.
    """
    if isinstance(value, np.ndarray):
        if value.size <= list_limit:
            return repr(value.tolist())
        return f"ndarray(shape={tuple(value.shape)}, dtype={value.dtype})"
    if isinstance(value, (list, tuple)):
        if len(value) <= list_limit:
            return repr(value)
        return f"{type(value).__name__}(len={len(value)})"
    if isinstance(value, Path):
        return repr(str(value))
    return repr(value)
