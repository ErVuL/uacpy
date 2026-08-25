"""Run propagation-model jobs in parallel.

Every uacpy model run is an independent, subprocess-bound computation, so a
batch of runs is embarrassingly parallel. The core primitive is
:func:`run_parallel`, which takes a list of fully-specified :class:`Job`\\ s —
each carrying its own ``model`` / ``env`` / ``source`` / ``receiver`` /
``run_mode`` / run-kwargs — and runs them across a process pool. Because each
job is self-contained and pickled to its own worker, heterogeneous batches
(different models, scenarios, or run modes per job) need no special handling:
cross-model comparison and parameter sweeps are the same operation. To sweep
one model's knobs, build the jobs with ``model.copy(**overrides)``.

Results (``Field`` / ``Rays`` / ``Modes`` / ``Arrivals`` / …) carry their full
numerical content as in-memory arrays, so pickling them back from a worker
loses nothing — including ray paths, eigenrays, and mode shapes. A worker that
owns its work dir (``cleanup=True``, the default) does drop the on-disk scratch
files and the ``result.metadata`` paths that point at them; build the job's
model with a pinned ``work_dir`` (``cleanup=False``) to keep those.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import warnings
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from uacpy.core.results import Result, ResultStack
from uacpy.core.exceptions import ConfigurationError
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.io.file_manager import FileManager


@dataclass
class Job:
    """One self-contained model run.

    A job fully describes a single ``model.run(env, source, receiver,
    run_mode=…, **run_kwargs)`` call. Different jobs may use entirely different
    models, scenarios, or run modes — that is what makes cross-model batches
    work without special casing.

    Attributes
    ----------
    model : PropagationModel
        The (already configured) model instance to run. To keep on-disk
        artifacts, build it with a pinned ``work_dir`` and ``cleanup=False``.
    env, source, receiver
        The scenario for this job. To run several models on the same scenario,
        pass the same objects to each ``Job``.
    run_mode : RunMode, optional
        Mode for this run.
    run_kwargs : dict, optional
        Extra keyword arguments for this model's ``run()``: ``frequencies``,
        ``source_waveform``, ``sample_rate``, ``output_duration``. Model
        configuration belongs on the constructor, so vary it by building one
        model per job (or ``model.copy(**overrides)``).
    label : any, optional
        Identifier for this job (used as the stacking coordinate; defaults to
        the job's index).
    """

    model: Any
    env: Any
    source: Any
    receiver: Any
    run_mode: Any = None
    run_kwargs: Dict[str, Any] = field(default_factory=dict)
    label: Any = None


def _worker_init(started_counter, scratch_root: str) -> None:
    """Pool initializer, run once per worker as it boots.

    Increments ``started_counter`` — proof at least one worker survived
    bootstrap, which is what separates a ``__main__`` re-import crash from a
    native binary dying mid-run — and points ``tempfile`` at a parent-owned
    scratch root so a SIGKILLed worker's model tempdirs are reaped by the
    parent instead of accumulating in /tmp.
    """
    with started_counter.get_lock():
        started_counter.value += 1
    tempfile.tempdir = scratch_root


# Where FileManager puts a use_tmpfs=True work dir (see its ``_tmpfs_available``).
_TMPFS_ROOT = Path('/dev/shm')


def _kept_tmpfs_dirs(results) -> List[str]:
    """The RAM-backed work dirs this batch's results name, deduplicated.

    A work dir is created inside its worker process, so the only record of its
    path that reaches the parent is the ``*_file`` metadata a kept run attaches
    to its result — which is attached iff ``cleanup=False``, i.e. iff the
    directory survived. A job that raised before writing an output names
    nothing, so this can come back shorter than the count of kept jobs.
    """
    dirs = set()
    for result in results:
        metadata = getattr(result, 'metadata', None) or {}
        for key, value in metadata.items():
            if not (key.endswith('_file') and isinstance(value, str)):
                continue
            work_dir = Path(value).parent
            if _TMPFS_ROOT in work_dir.parents:
                dirs.add(str(work_dir))
    return sorted(dirs)


def _reap_scratch_root(scratch_root: str, jobs, results=()) -> bool:
    """Remove the parent-owned scratch root, or keep the work dirs a job asked
    for. Returns True when the root was removed.

    An entry under the root is either a work dir a job was ASKED to keep —
    ``cleanup=False`` with an unpinned ``work_dir``, so the model allocated its
    tempdir here and left it behind — or debris: a killed worker's tempdir after
    a broken pool, or one a model could not unwind after a per-job exception
    (which ``raise_on_error=False`` collects, and which then arrives here).

    The decision is all-or-nothing, not per entry: an entry carries nothing
    that ties it back to the job that made it, so when ANY job was configured
    to keep its work dir the whole root is retained — debris from other jobs
    included — and the warning names the directory so the caller can clear it.
    Erring the other way would delete files a caller explicitly asked to keep.
    When no job asked, every entry is debris by elimination and the root is
    removed, which is what stops it leaking for the life of the machine.

    ``use_tmpfs=True`` dirs live in /dev/shm, outside this root, so they are
    neither reaped nor even visible here — which is why the decision below
    reads the *jobs* for them rather than the root's contents. Left to the
    listing alone, a batch of kept tmpfs work dirs looked like an empty root
    and the caller was never told RAM was still held. ``results`` is read only
    to name those dirs in the warning: /dev/shm is a shared system directory,
    so unlike the scratch root it cannot simply be handed over for removal."""
    try:
        leftovers = os.listdir(scratch_root)
    except OSError:
        leftovers = []
    keepers = [
        job for job in jobs
        if getattr(job.model, 'cleanup', True) is False
        and getattr(job.model, 'work_dir', None) is None
    ]
    # A tmpfs request falls back to disk where /dev/shm is missing or
    # unwritable, and the dir then lands under scratch_root like any other —
    # covered by the leftovers warning, so it must not also be reported as
    # RAM-backed.
    kept_on_tmpfs = [job for job in keepers
                     if getattr(job.model, 'use_tmpfs', False)]
    if kept_on_tmpfs and FileManager._tmpfs_available():
        # Name the individual dirs, the way the leftovers warning below names
        # its directory: /dev/shm holds every process's RAM-backed scratch, so
        # naming the root alone leaves the caller to work out which entries
        # are this batch's. Falls back to "them" for a batch whose results
        # carry no paths to name.
        named = _kept_tmpfs_dirs(results)
        removable = ', '.join(named) if named else 'them'
        warnings.warn(
            f"run_parallel: {len(kept_on_tmpfs)} job work dir(s) were kept "
            f"(cleanup=False, use_tmpfs=True) under {_TMPFS_ROOT}; they hold "
            f"RAM until removed, and are outside {scratch_root} so this call "
            f"cannot reap them. Remove {removable} when done with the files.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
    if leftovers and keepers:
        warnings.warn(
            f"run_parallel: {len(leftovers)} job work dir(s) were kept "
            f"(cleanup=False) under {scratch_root}; remove that "
            f"directory when done with the files.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
        return False
    shutil.rmtree(scratch_root, ignore_errors=True)
    return True


def _job_worker(job: Job):
    """Execute one job. Runs in a worker process; the job (and its model)
    arrives by pickle, so the worker holds its own isolated copy."""
    return job.model.run(
        job.env, job.source, job.receiver,
        run_mode=job.run_mode, **job.run_kwargs,
    )


@dataclass
class ParallelResult:
    """Outcome of :func:`run_parallel`, aligned to the jobs.

    Attributes
    ----------
    results : list
        Per-job result (``None`` where that job raised).
    errors : dict
        ``{job_index: exception}`` for the jobs that failed.
    labels : list
        Per-job label (defaults to the job index).
    coordinate_name : str
        Label for the stacking axis.
    """

    results: List[Optional[Result]]
    errors: Dict[int, BaseException]
    labels: List[Any]
    coordinate_name: str = 'case'

    @property
    def ok(self) -> bool:
        """True when every job succeeded."""
        return not self.errors

    def stack(self, coordinate_name: Optional[str] = None) -> ResultStack:
        """Bundle the successful results into a :class:`ResultStack`.

        Skips failed jobs (and their labels). Uses the job labels as the
        coordinate when they are numeric, else the job index. ``ResultStack``
        requires the slabs to share a concrete type *and* the same ``model`` /
        ``backend`` — so this is for single-model sweeps. For a cross-model
        batch, iterate ``results`` instead of stacking.
        """
        keep = [i for i, r in enumerate(self.results) if r is not None]
        if not keep:
            raise ConfigurationError(
                f"stack: no successful results to stack ({len(self.errors)} failed)."
            )
        try:
            coord = np.array([float(self.labels[i]) for i in keep], dtype=float)
        except (TypeError, ValueError):
            warnings.warn(
                "ParallelResult.stack: job labels are non-numeric; using "
                "the successful-job indices as the stack coordinate "
                "instead of the labels.",
                UserWarning, stacklevel=2,
            )
            coord = np.array(keep, dtype=float)
        return ResultStack(
            [self.results[i] for i in keep],
            coord,
            coordinate_name=coordinate_name or self.coordinate_name,
        )

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, i: int) -> Optional[Result]:
        return self.results[i]

    def __iter__(self):
        return iter(self.results)


def _main_is_importable() -> bool:
    """Whether ``__main__`` can be re-imported by a spawned worker.

    The ``spawn`` / ``forkserver`` start methods re-import the parent's
    ``__main__`` module in every worker. That needs an importable main module
    — a ``.py`` file run as a script. Interactive sessions (REPL, Jupyter,
    ``python -c``, piped stdin) have no importable ``__main__``, so spawned
    workers crash on import with a cryptic ``BrokenProcessPool``.
    """
    f = getattr(sys.modules.get('__main__'), '__file__', None)
    return bool(f) and os.path.isfile(f)


def run_parallel(
    jobs: Sequence[Job],
    *,
    n_workers: Optional[int] = None,
    raise_on_error: bool = True,
    start_method: str = 'spawn',
    coordinate_name: str = 'case',
) -> ParallelResult:
    """Run a list of fully-specified :class:`Job`\\ s in parallel.

    This is the generic primitive: each job carries its own model, scenario,
    run mode, and run-kwargs, so heterogeneous batches — different models,
    different scenarios, cross-model comparisons — all run the same way.

    Parameters
    ----------
    jobs : sequence of Job
        The runs to execute.
    n_workers : int, optional
        Pool size. Defaults to ``min(len(jobs), os.cpu_count())``.
    raise_on_error : bool, default True
        If True, the first failing job re-raises; jobs not yet started are
        cancelled, but jobs already running finish before the pool shuts down.
        If False, **clean** per-job failures (any ``Exception`` raised in a
        worker — bad env, unsupported mode, a model that exits non-zero) are
        collected in ``ParallelResult.errors`` and the other jobs still return.
        A **hard** worker crash (a native binary segfaulting or being OOM-killed)
        is different: it breaks the whole ``ProcessPoolExecutor``, so the
        remaining jobs cannot complete — that case always raises a typed error
        regardless of ``raise_on_error`` (it cannot be isolated to one slot).
    start_method : str, default 'spawn'
        Pool start method (``'spawn'`` / ``'forkserver'`` / ``'fork'``).
        Defaults to ``'spawn'``: ``'fork'`` is unsafe here because uacpy is
        multi-threaded (numpy/BLAS) and forking a multi-threaded process can
        deadlock the child on a copied lock. ``'spawn'``/``'forkserver'``
        re-import ``__main__`` in each worker, so from an interactive session
        (REPL, Jupyter, ``python -c``, piped stdin) run your code from a ``.py``
        script — otherwise ``run_parallel`` raises a clear error telling you to
        (or to pass ``start_method='fork'``).
    coordinate_name : str, default 'case'
        Label for the stacking axis of the returned ``ParallelResult``.

    Returns
    -------
    ParallelResult
        Per-job results and errors; call ``.stack()`` for a ``ResultStack``.
    """
    jobs = list(jobs)
    if not jobs:
        raise ConfigurationError(
            "run_parallel: jobs is empty. Pass at least one Job(model, env, "
            "source, receiver, run_mode=…, run_kwargs=…) — one per model run "
            "you want executed.")

    # A pinned (cleanup=False) work_dir must be unique per job: concurrent
    # workers sharing one dir collide on the models' fixed scratch filenames.
    # work_dir=None lets each worker allocate its own fresh tempdir.
    seen_dirs = set()
    for job in jobs:
        wd = getattr(job.model, 'work_dir', None)
        if wd is None:
            continue
        key = str(Path(wd).resolve())
        if key in seen_dirs:
            raise ConfigurationError(
                f"run_parallel: work_dir {wd!r} is pinned on more than one "
                "job; concurrent jobs would collide in the same scratch "
                "directory. Give each job its own work_dir, or leave "
                "work_dir=None to allocate a fresh tempdir per job."
            )
        seen_dirs.add(key)

    if n_workers is None:
        n_workers = min(len(jobs), os.cpu_count() or 1)
    # Typed errors for the two knobs the pool would otherwise reject with a
    # bare ValueError/TypeError from concurrent.futures / multiprocessing.
    try:
        n_workers = int(n_workers)
    except (TypeError, ValueError):
        raise ConfigurationError(
            f"run_parallel: n_workers must be an integer >= 1, got "
            f"{n_workers!r}.") from None
    if n_workers < 1:
        raise ConfigurationError(
            f"run_parallel: n_workers must be >= 1, got {n_workers}.")
    if start_method not in ('fork', 'spawn', 'forkserver'):
        raise ConfigurationError(
            f"run_parallel: start_method must be 'fork', 'spawn' or "
            f"'forkserver', got {start_method!r}.")

    results: List[Optional[Result]] = [None] * len(jobs)
    errors: Dict[int, BaseException] = {}

    ctx = mp.get_context(start_method)
    # started.value > 0 once any worker's initializer ran: the discriminator
    # between "workers crashed on bootstrap" and "a worker died mid-run".
    started = ctx.Value('i', 0)
    # Parent-owned scratch root the workers point tempfile at, removed in the
    # finally below — so model tempdirs of a SIGKILLed/OOM-killed worker are
    # reaped instead of leaking in /tmp.
    scratch_root = tempfile.mkdtemp(prefix='uacpy_parallel_')
    try:
        with ProcessPoolExecutor(
            max_workers=n_workers, mp_context=ctx,
            initializer=_worker_init, initargs=(started, scratch_root),
        ) as executor:
            future_to_idx = {
                executor.submit(_job_worker, job): i for i, job in enumerate(jobs)
            }
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                try:
                    results[i] = future.result()
                except BrokenProcessPool:
                    # A worker died hard (native segfault / OOM-kill) — the pool
                    # is now broken and every outstanding job is unrecoverable.
                    # Unlike a clean per-job exception this cannot be isolated to
                    # one slot, so re-raise to the handler below instead of
                    # quietly stuffing the opaque error into every errors[i].
                    # BrokenProcessPool derives from RuntimeError, hence from
                    # Exception: this clause has to stay above the general one
                    # or ``raise_on_error=False`` would swallow it as a per-job
                    # failure.
                    raise
                except Exception as exc:  # noqa: BLE001 — surface or collect per policy
                    if raise_on_error:
                        for f in future_to_idx:
                            f.cancel()
                        raise
                    errors[i] = exc
    except BrokenProcessPool as exc:
        # The worker pool died before returning. The usual cause in interactive
        # use: spawn/forkserver workers re-import __main__, but an interactive
        # session (REPL, Jupyter, ``python -c``, piped stdin) has none, so the
        # children crash on bootstrap. Turn the opaque error into a fix. A
        # genuine in-worker crash (segfault/OOM in a native binary) keeps the
        # original BrokenProcessPool.
        if start_method in ('spawn', 'forkserver') and not _main_is_importable():
            raise ConfigurationError(
                f"run_parallel: the {start_method!r} worker pool died on "
                f"startup — most likely because it cannot re-import __main__ "
                f"from an interactive session (REPL, Jupyter, `python -c`, or "
                f"piped stdin). Run from a .py script (guarded with "
                f"`if __name__ == '__main__':`), or pass start_method='fork' "
                f"(note: fork can deadlock a heavily-threaded process)."
            ) from exc
        # A spawned worker re-imports __main__. If the calling script runs
        # run_parallel at module level (no ``if __name__ == '__main__':``), the
        # import re-enters here and the workers die on bootstrap — before any
        # initializer runs. ``started`` distinguishes that from a worker that
        # booted fine and was then killed mid-run (the old "did any job
        # finish" test could not: an early OOM-kill also finishes nothing,
        # and misblamed a __main__ guard that was already correct).
        if (start_method in ('spawn', 'forkserver') and started.value == 0):
            raise ConfigurationError(
                f"run_parallel: the {start_method!r} workers died on "
                f"bootstrap, before any began running. If the calling script "
                f"runs run_parallel at module level, wrap it in "
                f"`if __name__ == '__main__':` — each worker re-imports "
                f"__main__, so an unguarded call re-enters run_parallel on "
                f"import. Otherwise a native library crashed while "
                f"initialising in the spawned process."
            ) from exc
        # Otherwise a worker died mid-run (a native binary segfaulted or was
        # OOM-killed). ProcessPoolExecutor cannot isolate this — the whole pool
        # is gone — so the batch aborts regardless of raise_on_error, with a
        # clear typed error rather than an opaque BrokenProcessPool in every slot.
        raise ConfigurationError(
            "run_parallel: a worker process died mid-run — most likely a native "
            "model binary segfaulted or was OOM-killed. This breaks the whole "
            "pool, so the remaining jobs could not complete. Re-run the "
            "suspect job on its own to see its error, lower n_workers if "
            "memory-bound, or check that job's model inputs."
        ) from exc

    finally:
        _reap_scratch_root(scratch_root, jobs, results)

    labels = [job.label if job.label is not None else i for i, job in enumerate(jobs)]
    return ParallelResult(results, errors, labels, coordinate_name)
