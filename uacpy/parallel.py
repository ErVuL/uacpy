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
from concurrent.futures import (CancelledError, ProcessPoolExecutor,
                                TimeoutError as FuturesTimeoutError,
                                as_completed)
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from uacpy.core.results import Result, ResultStack
from uacpy.core._export import _resolve_class, saved_class_path
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, IOWarning, ModelExecutionError,
)
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core._repr import build, count, value_bit
from uacpy.models._workspace import FileManager, ScratchPolicy

__all__ = ['run_parallel', 'Job', 'ParallelResult']


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
        ``source_waveform``, ``sample_rate``, ``output_duration``,
        ``t_start``. Model
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

    def __repr__(self) -> str:
        return build('Job', [type(self.model).__name__,
                             value_bit('run_mode', self.run_mode),
                             value_bit('label', self.label)])


def _worker_init(started_counter, scratch_root: str) -> None:
    """Pool initializer, run once per worker as it boots.

    Increments ``started_counter`` — proof at least one worker survived
    bootstrap, which is what separates a ``__main__`` re-import crash from a
    native binary dying mid-run — points ``tempfile`` at a parent-owned
    scratch root so a SIGKILLed worker's model tempdirs are reaped by the
    parent instead of accumulating in /tmp, and makes every engine launch
    single-threaded: the pool supplies the parallelism, and an engine that
    also started one thread per core would run N x N threads on N cores.
    """
    with started_counter.get_lock():
        started_counter.value += 1
    tempfile.tempdir = scratch_root
    from uacpy.models._launch import set_launch_threads
    set_launch_threads(True)


def _scratch_policy(model) -> ScratchPolicy:
    """``model.scratch_policy()``. A job's model may be any object with a
    ``run``; one that is not a
    :class:`~uacpy.models.base.PropagationModel` keeps no scratch of its
    own."""
    policy = getattr(model, 'scratch_policy', None)
    if policy is None:
        return ScratchPolicy(pinned_dir=None, keeps_files=False,
                             on_tmpfs=False)
    return policy()


def _kept_tmpfs_dirs(results) -> List[str]:
    """The RAM-backed work dirs this batch's results name, deduplicated.

    A work dir is created inside its worker process, so the only record of its
    path that reaches the parent is the ``work_dir`` a kept run stamps on its
    result's metadata — stamped iff ``cleanup=False``, i.e. iff the directory
    survived. A job that raised before producing a result names nothing, so
    this can come back shorter than the count of kept jobs.
    """
    dirs = set()
    for result in results:
        metadata = getattr(result, 'metadata', None) or {}
        work_dir = metadata.get('work_dir')
        if (isinstance(work_dir, str)
                and FileManager.DEV_SHM_ROOT in Path(work_dir).parents):
            dirs.add(work_dir)
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
    policies = [_scratch_policy(job.model) for job in jobs]
    keepers = [policy for policy in policies
               if policy.keeps_files and policy.pinned_dir is None]
    # A tmpfs request falls back to disk where /dev/shm is missing or
    # unwritable, and the dir then lands under scratch_root like any other —
    # covered by the leftovers warning, so it must not also be reported as
    # RAM-backed.
    kept_on_tmpfs = [policy for policy in keepers if policy.on_tmpfs]
    if kept_on_tmpfs and FileManager.dev_shm_usable():
        # Name the individual dirs, the way the leftovers warning below names
        # its directory: /dev/shm holds every process's RAM-backed scratch, so
        # naming the root alone leaves the caller to work out which entries
        # are this batch's. Falls back to "them" for a batch whose results
        # carry no paths to name.
        named = _kept_tmpfs_dirs(results)
        removable = ', '.join(named) if named else 'them'
        warnings.warn(
            f"run_parallel: {len(kept_on_tmpfs)} job work dir(s) were kept "
            f"(cleanup=False, use_tmpfs=True) under "
            f"{FileManager.DEV_SHM_ROOT}; they hold RAM until removed, and are "
            f"outside {scratch_root} so this call cannot reap them. Remove "
            f"{removable} when done with the files.",
            IOWarning, skip_file_prefixes=USER_FRAME_SKIP)
    if leftovers and keepers:
        warnings.warn(
            f"run_parallel: {len(leftovers)} job work dir(s) were kept "
            f"(cleanup=False) under {scratch_root}; remove that "
            f"directory when done with the files.",
            IOWarning, skip_file_prefixes=USER_FRAME_SKIP)
        return False
    shutil.rmtree(scratch_root, ignore_errors=True)
    return True


def _worker_crash_error(cause: BaseException) -> ModelExecutionError:
    """The typed error for a worker that died mid-run.

    Built through :class:`ModelExecutionError`'s constructor, then given a
    message of its own: the pool reports no exit code and no model name for
    the dead worker, so ``return_code`` is ``None`` (unknown) — not
    ``NEVER_LAUNCHED``, since the worker did run — and the constructor's
    "exit code" sentence, which would state a number nobody measured, is
    replaced.
    """
    err = ModelExecutionError('run_parallel worker', None)
    err.message = (
        "run_parallel: a worker process died mid-run — most likely a native "
        "model binary segfaulted or was OOM-killed. This breaks the whole "
        f"pool, so the jobs it cut off could not complete ({cause}).")
    err.remediation = (
        "Re-run the suspect job on its own to see its error, lower n_workers "
        "if memory-bound, or check that job's model inputs. With "
        "raise_on_error=False the finished jobs' results are returned.")
    err.args = (err.message,)
    return err


def _job_worker(job: Job):
    """Execute one job and return ``(result, warnings)``. Runs in a worker
    process; the job (and its model) arrives by pickle, so the worker holds
    its own isolated copy.

    A warning raised in a worker never reaches the parent's filters or the
    caller's notebook cell, and the package announces NaN cells,
    collapses and solver diagnoses by warning, so every warning the run
    emits is recorded here as ``(category, message)`` and handed back."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = job.model.run(
            job.env, job.source, job.receiver,
            run_mode=job.run_mode, **job.run_kwargs,
        )
    return result, [(w.category, str(w.message)) for w in caught]


def _reemit_job_warnings(job_warnings, job: Job, index: int) -> None:
    """Raise each warning a worker recorded again in this process, under
    its own category, prefixed with the job's label, so the caller's
    filters, ``pytest.warns`` and notebook see it."""
    label = job.label if job.label is not None else index
    for category, message in job_warnings:
        warnings.warn(f"run_parallel job {label!r}: {message}", category,
                      skip_file_prefixes=USER_FRAME_SKIP)


def _saved_result(saved) -> Any:
    """The result a :meth:`ParallelResult.to_dict` entry holds, rebuilt
    by the ``from_dict`` of the class it names (a :class:`Result` or a
    :class:`ResultStack`)."""
    saved = dict(saved)
    path = saved.pop('__class__')
    klass: Any = (ResultStack if path == saved_class_path(ResultStack)
                  else _resolve_class(path, Result))
    return klass.from_dict(saved)


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
    warnings : dict
        ``{job_index: [(category, message), ...]}`` for the jobs that
        finished and warned. :func:`run_parallel` also re-emits each one in
        the calling process, under its own category, with the message
        prefixed by the job's label.

    Saves and loads through :meth:`to_dict` / :meth:`from_dict`, each
    result nested as its own ``to_dict``.
    """

    results: List[Optional[Any]]
    errors: Dict[int, BaseException]
    labels: List[Any]
    coordinate_name: str = 'case'
    warnings: Dict[int, List[Tuple[type, str]]] = field(default_factory=dict)

    def __repr__(self) -> str:
        return build('ParallelResult', [
            count(len(self.results), 'result'),
            value_bit(self.coordinate_name, list(self.labels)),
            count(len(self.errors), 'error') if self.errors else None,
            f"{len(self.warnings)} warned" if self.warnings else None])

    @property
    def ok(self) -> bool:
        """True when every job succeeded."""
        return not self.errors

    def to_dict(self) -> Dict[str, Any]:
        """This outcome as a dict: ``results``, each result as its own
        ``to_dict`` under ``'__class__'``, its public class path
        (``None`` where the job raised), then ``errors``, ``labels``,
        ``coordinate_name`` and ``warnings``, the exceptions and warning
        categories as the objects they are (they pickle).
        ``np.savez(f, **d)`` stores it; read it back with
        ``np.load(f, allow_pickle=True)`` into :meth:`from_dict`."""
        return {
            'results': [None if r is None else
                        {'__class__': saved_class_path(type(r)),
                         **r.to_dict()} for r in self.results],
            'errors': dict(self.errors),
            'labels': list(self.labels),
            'coordinate_name': self.coordinate_name,
            'warnings': {i: list(w) for i, w in self.warnings.items()},
        }

    @classmethod
    def from_dict(cls, d) -> 'ParallelResult':
        """The outcome :meth:`to_dict` wrote, each result rebuilt by
        the ``from_dict`` of the class it names; ``d`` may be the
        mapping ``np.load(f, allow_pickle=True)`` returns.

        Parameters
        ----------
        d : mapping
            :meth:`to_dict` output, or the mapping ``np.load`` returns
            for it.
        """
        d = {k: (v.item() if isinstance(v, np.ndarray) and v.ndim == 0
                 else v) for k, v in dict(d).items()}
        return cls(results=[None if saved is None else _saved_result(saved)
                            for saved in d['results']],
                   errors=dict(d['errors']), labels=list(d['labels']),
                   coordinate_name=d['coordinate_name'],
                   warnings={i: list(w)
                             for i, w in dict(d['warnings']).items()})

    def stack(self, coordinate_name: Optional[str] = None) -> ResultStack:
        """Bundle the successful results into a :class:`ResultStack`.

        Skips failed jobs (and their labels). Uses the job labels as the
        coordinate when they are numeric, else the job index. ``ResultStack``
        requires the slabs to share a concrete type *and* the same ``model`` /
        ``backend`` — so this is for single-model sweeps. For a cross-model
        batch, iterate ``results`` instead of stacking.

        Parameters
        ----------
        coordinate_name : str, optional
            The stacking axis's name.
        """
        kept = [(i, r) for i, r in enumerate(self.results) if r is not None]
        keep = [i for i, _ in kept]
        if not keep:
            raise ConfigurationError(
                f"stack: no successful results to stack ({len(self.errors)} failed)."
            )
        nested = [i for i, r in kept if isinstance(r, ResultStack)]
        if nested:
            raise ConfigurationError(
                f"ParallelResult.stack: job(s) {nested} returned a "
                f"ResultStack (a multi-depth Source), and a stack of stacks "
                f"is neither superposable nor plottable.",
                remediation=("Stack the jobs' results yourself along the "
                             "axis you need, e.g. one ResultStack per source "
                             "depth from result.slabs[k], or run single-depth "
                             "Sources."))
        try:
            coord = np.array([float(self.labels[i]) for i in keep], dtype=float)
        except (TypeError, ValueError):
            warnings.warn(
                "ParallelResult.stack: job labels are non-numeric; using "
                "the successful-job indices as the stack coordinate "
                "instead of the labels.",
                FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
            coord = np.array(keep, dtype=float)
        return ResultStack(
            [r for _, r in kept],
            coord,
            coordinate_name=coordinate_name or self.coordinate_name,
        )

    def summary(self):
        """One row per job as a ``pandas.DataFrame`` (optional extra
        ``uacpy[xarray]``): ``label``, ``ok``, ``error_type`` (the
        exception class, ``''`` for a job that finished) and ``n_warnings``,
        in job order."""
        from uacpy.core._export import require_extra
        pandas = require_extra('pandas', 'ParallelResult.summary')
        n = len(self.results)
        return pandas.DataFrame({
            'label': list(self.labels),
            'ok': [i not in self.errors for i in range(n)],
            'error_type': [type(self.errors[i]).__name__
                           if i in self.errors else '' for i in range(n)],
            'n_warnings': [len(self.warnings.get(i, ())) for i in range(n)],
        })

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, i: int) -> Optional[Result]:
        return self.results[i]

    def __iter__(self):
        return iter(self.results)


def _main_is_interactive() -> bool:
    """Whether ``__main__`` names neither a file nor a module spec — a REPL, a
    Jupyter kernel or ``python -c`` — so a spawned worker does not re-run it
    and nothing defined in it exists in the worker."""
    main = sys.modules.get('__main__')
    if getattr(getattr(main, '__spec__', None), 'name', None):
        return False
    return not getattr(main, '__file__', None)


def _main_cannot_be_reimported() -> bool:
    """Whether a spawned worker would fail to re-run the parent's ``__main__``.

    The ``spawn`` / ``forkserver`` start methods re-run the parent's
    ``__main__`` in each worker only when it names a file (or a module spec).
    A REPL, a Jupyter kernel and ``python -c`` name neither, so their workers
    skip that step and boot normally. The failing shape is a ``__file__`` that
    is not a readable file — piped stdin, whose ``__file__`` is ``'<stdin>'``
    — which every worker then tries, and fails, to re-read.
    """
    main = sys.modules.get('__main__')
    if getattr(getattr(main, '__spec__', None), 'name', None):
        return False
    f = getattr(main, '__file__', None)
    return isinstance(f, str) and bool(f) and not os.path.isfile(f)


def _check_jobs(jobs) -> list:
    """``jobs`` as a list, refused when empty or when two jobs pin one
    ``work_dir``."""
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
        wd = _scratch_policy(job.model).pinned_dir
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
    return jobs


def _check_pool_args(jobs, n_workers, start_method) -> int:
    """The worker count, and a refusal for a start method the pool does not
    have or one that cannot load a job's model class."""
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
    if start_method in ('spawn', 'forkserver') and _main_is_interactive():
        # A class defined in an interactive session pickles by the reference
        # ``__main__.<name>``, which a spawned worker's own ``__main__`` does
        # not define: unpickling the job kills the worker, and the pool would
        # report it as a crashed binary.
        for i, job in enumerate(jobs):
            if type(job.model).__module__ == '__main__':
                raise ConfigurationError(
                    f"run_parallel: job {i}'s model class "
                    f"{type(job.model).__name__!r} is defined in this "
                    f"interactive session's __main__, which a {start_method!r} "
                    f"worker cannot import.",
                    remediation="Define the class in an importable module "
                                "(a .py file on the path) and import it, or "
                                "pass start_method='fork'.")
    return n_workers


def _raise_if_bootstrap_death(exc, start_method, started) -> None:
    """A broken pool whose workers never started (``started`` counts the
    initializers that ran) is a bootstrap death: raise the refusal that
    names its cause. A pool that broke mid-run returns."""
    if start_method in ('spawn', 'forkserver') and started.value == 0:
        if _main_cannot_be_reimported():
            raise ConfigurationError(
                f"run_parallel: the {start_method!r} workers died on "
                f"bootstrap: each one re-runs __main__ from its file, and "
                f"this __main__ was read from piped stdin, which a worker "
                f"cannot re-read. Save the code to a .py file (with the "
                f"`if __name__ == '__main__':` guard), run it from a REPL "
                f"or Jupyter, or pass start_method='fork' (note: fork can "
                f"deadlock a heavily-threaded process)."
            ) from exc
        # A spawned worker re-runs a script's __main__. If the script calls
        # run_parallel at module level (no ``if __name__ == '__main__':``),
        # that re-run re-enters here and the workers die on bootstrap.
        raise ConfigurationError(
            f"run_parallel: the {start_method!r} workers died on "
            f"bootstrap, before any began running. If the calling script "
            f"runs run_parallel at module level, wrap it in "
            f"`if __name__ == '__main__':` — each worker re-imports "
            f"__main__, so an unguarded call re-enters run_parallel on "
            f"import. Otherwise a native library crashed while "
            f"initialising in the spawned process."
        ) from exc


def _salvage_after_crash(future_to_idx, jobs, results, errors, warned,
                         crash) -> None:
    """After a worker crash broke the pool: keep the results of the jobs
    that finished, and record ``crash`` for every other job."""
    # The whole pool is gone, but the jobs that finished before it broke
    # hold their results; every other job is cut off by the crash.
    for future, i in future_to_idx.items():
        if results[i] is not None or i in errors:
            continue
        try:
            results[i], job_warnings = future.result(timeout=0)
            _reemit_job_warnings(job_warnings, jobs[i], i)
            if job_warnings:
                warned[i] = job_warnings
        except (BrokenProcessPool, CancelledError, FuturesTimeoutError):
            errors[i] = crash
        except Exception as job_exc:  # noqa: BLE001 — that job's own failure
            errors[i] = job_exc


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
        breaks the whole ``ProcessPoolExecutor``, so the jobs still running or
        queued cannot complete. It is a :class:`ModelExecutionError`: raised
        when ``raise_on_error`` is True; with False the jobs that finished keep
        their results and every job the crash cut off carries that error in
        ``ParallelResult.errors``.
    start_method : str, default 'spawn'
        Pool start method (``'spawn'`` / ``'forkserver'`` / ``'fork'``).
        Defaults to ``'spawn'``: ``'fork'`` is unsafe here because uacpy is
        multi-threaded (numpy/BLAS) and forking a multi-threaded process can
        deadlock the child on a copied lock. ``'spawn'``/``'forkserver'``
        re-run a script's ``__main__`` in each worker, so a ``.py`` script needs
        the ``if __name__ == '__main__':`` guard. A REPL, a Jupyter kernel and
        ``python -c`` work as they are; code piped on stdin cannot be re-read
        by the workers, and ``run_parallel`` raises an error saying so. In any
        session, a model class defined in an interactive ``__main__`` cannot
        be loaded by a worker — define it in an importable module.
    coordinate_name : str, default 'case'
        Label for the stacking axis of the returned ``ParallelResult``.

    Returns
    -------
    ParallelResult
        Per-job results and errors; call ``.stack()`` for a ``ResultStack``.
    """
    jobs = _check_jobs(jobs)

    n_workers = _check_pool_args(jobs, n_workers, start_method)

    results: List[Optional[Result]] = [None] * len(jobs)
    errors: Dict[int, BaseException] = {}
    warned: Dict[int, List[Tuple[type, str]]] = {}

    ctx = mp.get_context(start_method)
    # started.value > 0 once any worker's initializer ran: the discriminator
    # between "workers crashed on bootstrap" and "a worker died mid-run".
    started = ctx.Value('i', 0)
    # Parent-owned scratch root the workers point tempfile at, removed in the
    # finally below — so model tempdirs of a SIGKILLed/OOM-killed worker are
    # reaped instead of leaking in /tmp.
    scratch_root = tempfile.mkdtemp(prefix='uacpy_parallel_')
    future_to_idx = {}
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
                    results[i], job_warnings = future.result()
                    _reemit_job_warnings(job_warnings, jobs[i], i)
                    if job_warnings:
                        warned[i] = job_warnings
                except BrokenProcessPool:
                    # A worker died hard (native segfault / OOM-kill) — the pool
                    # is now broken and every outstanding job is unrecoverable.
                    # Re-raised to the handler below, which tells a bootstrap
                    # death from a mid-run one. BrokenProcessPool derives from
                    # RuntimeError, hence from Exception: this clause has to
                    # stay above the general one or ``raise_on_error=False``
                    # would swallow it as a per-job failure.
                    raise
                except Exception as exc:  # noqa: BLE001 — surface or collect per policy
                    if raise_on_error:
                        for f in future_to_idx:
                            f.cancel()
                        raise
                    errors[i] = exc
    except BrokenProcessPool as exc:
        # ``started`` counts workers whose initializer ran, so 0 means the
        # pool died on bootstrap and anything more means a worker booted and
        # was then killed mid-run (a native segfault or OOM-kill) — a death no
        # property of ``__main__`` can explain.
        _raise_if_bootstrap_death(exc, start_method, started)
        # Otherwise a worker died mid-run (a native binary segfaulted or was
        # OOM-killed): a solver that died, so ModelExecutionError.
        crash = _worker_crash_error(exc)
        if raise_on_error:
            raise crash from exc
        _salvage_after_crash(future_to_idx, jobs, results, errors, warned,
                             crash)

    finally:
        _reap_scratch_root(scratch_root, jobs, results)

    labels = [job.label if job.label is not None else i for i, job in enumerate(jobs)]
    return ParallelResult(results, errors, labels, coordinate_name, warned)
