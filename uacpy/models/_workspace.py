"""The work directory of a run: the :class:`FileManager` that allocates,
adopts and cleans it, the claim a pinned directory takes against a second
thread or process, and the per-depth redirect of a pinned directory into its
subdirectory."""

import gc
import os
import shutil
import tempfile
import threading
import warnings
import weakref
from pathlib import Path
from typing import NamedTuple, Optional, Union

from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.exceptions import ConfigurationError, FallbackWarning


# Pinned work dirs currently claimed by a running model: resolved path ->
# [owning thread, claim depth, directory fd holding the cross-process lock
# (or None)]. The models write fixed scratch filenames
# (``bounce.env``, ``field.flp``, ``tl.grid``), so two runs sharing one
# directory overwrite each other's decks and both read back whatever the last
# binary left: two Bounce runs on different bottoms, started from two threads
# on one pinned work_dir, returned bit-identical answers with nothing raised.
# ``run_parallel`` refuses that configuration across its own jobs; this is the
# same refusal for every other way two runs can start at once. A second
# PROCESS on the same directory is refused through an advisory ``flock`` on
# the directory itself (:func:`_lock_work_dir_across_processes`), which this
# in-memory registry cannot see.
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
        if owner is not None:
            _close_work_dir_lock(owner)
        _PINNED_WORK_DIRS[key] = [me, 1, None]
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
            _close_work_dir_lock(owner)
            del _PINNED_WORK_DIRS[key]


def _close_work_dir_lock(entry) -> None:
    """Release the cross-process lock a registry entry holds, if any."""
    fd = entry[2]
    entry[2] = None
    if fd is not None:
        try:
            os.close(fd)            # closing the descriptor drops the flock
        except OSError:
            pass


def _lock_work_dir_across_processes(claim, model_name: str) -> None:
    """Hold an advisory lock on the claimed directory for the claim's life,
    so a run in ANOTHER process pinned to the same directory is refused
    rather than trading scratch files with this one.

    ``flock`` on the directory's own descriptor: nothing is written into a
    directory the caller owns. Taken once per claim (a nested or sequential
    re-entry by the owning thread already holds it) and released by
    :func:`_release_work_dir` with the last claim. Where the platform or the
    filesystem offers no ``flock`` (Windows, some network mounts) the
    in-process guard is all there is, and the run proceeds.
    """
    key, owner_thread = claim
    try:
        import fcntl
    except ImportError:                    # Windows
        return
    with _PINNED_WORK_DIRS_LOCK:
        entry = _PINNED_WORK_DIRS.get(key)
        if entry is None or entry[0] is not owner_thread or entry[2] is not None:
            return
        try:
            fd = os.open(key, os.O_RDONLY)
        except OSError:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            raise ConfigurationError(
                f"{model_name}: work_dir {key!r} is in use by another "
                f"process; concurrent runs would collide in the same scratch "
                f"directory and return each other's results.",
                remediation="Give each run its own work_dir, or leave "
                            "work_dir=None to allocate a fresh tempdir per "
                            "run.",
            ) from None
        except OSError:
            os.close(fd)                   # a filesystem without flock
            return
        entry[2] = fd


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
    # The child's copies of the parent's lock descriptors go too. The parent
    # keeps its own, so a directory a parent thread is still running in
    # stays locked against the child — that is a real collision.
    for entry in _PINNED_WORK_DIRS.values():
        _close_work_dir_lock(entry)
    _PINNED_WORK_DIRS.clear()


if hasattr(os, 'register_at_fork'):        # POSIX only
    os.register_at_fork(after_in_child=_reset_work_dir_claims)


#: Per-depth scratch redirects in flight, keyed by ``id(model)``. Thread-local
#: because two threads may drive one model instance, and module-level (not an
#: instance attribute) because a model is pickled to ``run_parallel`` workers
#: and :class:`threading.local` is not picklable.
_SCRATCH_REDIRECT = threading.local()


def _effective_work_dir(model) -> Optional[Path]:
    """The pinned directory a run writes into: ``model.work_dir``, or its
    per-depth subdirectory while :meth:`PropagationModel.run` is inside one
    depth of a per-depth loop (see
    :meth:`PropagationModel._scratch_subdir`). Read by
    :meth:`PropagationModel._setup_file_manager` in place of the attribute,
    so the redirect never touches what the user configured.

    Module-level rather than a method because ``_setup_file_manager`` is
    borrowed as a plain function by stand-ins that are not
    ``PropagationModel`` subclasses (``test_concurrency``'s ``_StubModel``),
    which is what keeps that test driving the real recipe."""
    if model.work_dir is None:
        return None
    sub = getattr(_SCRATCH_REDIRECT, 'by_model', {}).get(id(model))
    return Path(model.work_dir) / sub if sub else Path(model.work_dir)


def _run_release_hooks(hooks):
    """Run every pending release hook and drop it, so each fires exactly once.

    Module-level rather than a method: it is also the target of the
    ``weakref.finalize`` in :meth:`FileManager.on_release`, and a bound method
    there would keep the manager alive and stop that finalizer ever running.

    A hook that raises is swallowed for the same reason cleanup failures are —
    letting go of a scratch directory must never mask the exception a caller
    is trying to surface.
    """
    while hooks:
        hook = hooks.pop()
        try:
            hook()
        except Exception:
            pass


class ScratchPolicy(NamedTuple):
    """Where a model's scratch files go and what happens to them after a
    run: :meth:`PropagationModel.scratch_policy
    <uacpy.models.base.PropagationModel.scratch_policy>`."""
    #: The ``work_dir`` the caller named, or ``None`` for a fresh temporary
    #: directory per run.
    pinned_dir: Optional[Path]
    #: Whether the run leaves its scratch files in place (``cleanup=False``).
    keeps_files: bool
    #: Whether the model asked for a RAM-backed directory under
    #: :attr:`FileManager.DEV_SHM_ROOT` (``use_tmpfs=True``); it falls back to
    #: disk where :meth:`FileManager.dev_shm_usable` is False.
    on_tmpfs: bool


class FileManager:
    """
    Manage temporary files for acoustic models, optionally on tmpfs.

    Provides automatic cleanup of temporary files and optional placement in
    a RAM-based filesystem for improved I/O performance.

    Parameters
    ----------
    use_tmpfs : bool, optional
        Use RAM-based tmpfs filesystem. Default is False. On Linux, uses
        ``/dev/shm`` if available; when ``/dev/shm`` is unavailable or
        ``base_dir`` is given, files go to the directory actually chosen
        and the ``use_tmpfs`` attribute reads False.
    base_dir : str or Path, optional
        Base directory for file operations. If ``None``, uses the system
        temp directory.
    prefix : str, optional
        Prefix for temporary directory names. Default is ``'uacpy_'``.
    cleanup : bool, optional
        Whether :meth:`finish` removes the scratch files. Default is True.
        Both ``__exit__`` and the model recipe (in ``models/base.py``'s module
        docstring) go through :meth:`finish` from a ``finally``, so a run lets
        go of its directory the same way whether it succeeded or raised. What
        *this* class decides is how much gets removed; see
        :meth:`cleanup_work_dir`.

    Attributes
    ----------
    work_dir : Path
        Current working directory for model files.
    use_tmpfs : bool
        Whether the files are actually placed on ``/dev/shm`` — False when
        the constructor fell back to disk or ``base_dir`` was given, whatever
        was requested.
    cleanup : bool
        Whether automatic cleanup is enabled.

    Examples
    --------
    Basic usage with automatic cleanup:

    >>> with FileManager(use_tmpfs=True) as fm:
    ...     env_file = fm.get_path('env.env')
    ...     # Write files, run model. Files cleaned up on exit.

    Manual management:

    >>> fm = FileManager(cleanup=False)
    >>> work_dir = fm.create_work_dir()
    >>> # ... do work ...
    >>> fm.cleanup_work_dir()
    """

    #: Where a ``use_tmpfs=True`` work dir is created: the RAM-backed
    #: filesystem Linux mounts at ``/dev/shm``.
    DEV_SHM_ROOT = Path('/dev/shm')

    def __init__(
        self,
        use_tmpfs: bool = False,
        base_dir: Optional[Union[str, Path]] = None,
        prefix: str = 'uacpy_',
        cleanup: bool = True,
    ):
        self.use_tmpfs = False
        self.prefix = prefix
        self.cleanup = cleanup
        self.work_dir = None
        # Set when uacpy created the work dir itself, so cleanup may remove it
        # whole. A caller-pinned directory is adopted via ``adopt_work_dir``,
        # which records what was already there so cleanup spares it.
        self._owns_work_dir = False
        self._preexisting = None
        # Callables to run when this manager lets go of its work dir; see
        # :meth:`on_release`.
        self._release_hooks = []

        if base_dir is not None:
            self.base_dir = Path(base_dir)
        elif use_tmpfs and self.dev_shm_usable():
            self.base_dir = self.DEV_SHM_ROOT
            self.use_tmpfs = True
        else:
            self.base_dir = Path(tempfile.gettempdir())

        if not self.base_dir.exists():
            raise ConfigurationError(
                f"Base directory does not exist: {self.base_dir}.",
                remediation="Create it, or pass a writable base_dir=.")
        if not os.access(self.base_dir, os.W_OK):
            raise ConfigurationError(
                f"Base directory not writable: {self.base_dir}.",
                remediation="Pass a writable base_dir= (or fix its permissions).")

    @classmethod
    def dev_shm_usable(cls) -> bool:
        """Whether :attr:`DEV_SHM_ROOT` exists and is writable, i.e. whether a
        ``use_tmpfs=True`` work dir lands on it rather than on disk."""
        return cls.DEV_SHM_ROOT.exists() and os.access(cls.DEV_SHM_ROOT, os.W_OK)

    def create_work_dir(self) -> Path:
        """
        Create a uniquely-named scratch directory under ``base_dir``.

        The directory is uacpy's, so :meth:`cleanup_work_dir` removes it whole.
        Use :meth:`adopt_work_dir` for a directory the caller names.

        Refuses while a working directory is already live: this manager tracks
        exactly one, so a second call would drop the only reference to the
        first and leave it on disk for good — ``cleanup_work_dir`` can no
        longer reach it. Release the current one first. (:meth:`__enter__`
        reuses a live directory rather than tripping this, so ``with`` still
        works on a manager that already has one.)

        Returns
        -------
        work_dir : Path
            Path to the working directory.
        """
        if self.work_dir is not None and self.work_dir.exists():
            raise ConfigurationError(
                f"FileManager already holds work_dir {self.work_dir}; creating "
                f"a second would abandon it with nothing left to remove it.",
                remediation="Call cleanup_work_dir() first, or use a separate "
                            "FileManager for the second directory.")
        self.work_dir = Path(tempfile.mkdtemp(
            prefix=self.prefix,
            dir=str(self.base_dir)
        ))
        self._owns_work_dir = True
        self._preexisting = None
        return self.work_dir

    def adopt_work_dir(self, work_dir: Union[str, Path]) -> Path:
        """Use a caller-named directory, taking ownership only if we create it.

        A directory that did not exist is uacpy's to remove whole. One the
        caller already had is not: its prior entries are recorded so
        :meth:`cleanup_work_dir` removes only what this run adds, and the
        directory itself survives. Snapshotting is used rather than tracking
        :meth:`get_path` calls because the model binaries also write files
        uacpy never names (``tl.grid``, ``.prt``, ``.shd``).

        A path that exists but is not a directory is a caller mistake, not a
        filesystem accident — ``Bellhop(work_dir='some_file.txt')`` reaches
        here through ``models/base.py`` — so it is reported as a typed
        :class:`~uacpy.core.exceptions.ConfigurationError` rather than the
        bare ``FileExistsError`` ``mkdir`` raises.
        """
        work_dir = Path(work_dir)
        if work_dir.exists() and not work_dir.is_dir():
            raise ConfigurationError(
                f"work_dir {work_dir} exists but is not a directory.",
                remediation="Pass a directory path (existing or not); uacpy "
                            "writes the model's input and output files inside "
                            "it.")
        self.work_dir = work_dir
        existed = self.work_dir.exists()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._owns_work_dir = not existed
        self._preexisting = (
            {p.name for p in self.work_dir.iterdir()} if existed else None)
        return self.work_dir

    def get_path(self, filename: str) -> Path:
        """
        Return the full path for a file in the working directory.

        Parameters
        ----------
        filename : str
            Filename.

        Returns
        -------
        path : Path
            Full path to the file (working directory is created on demand).
        """
        if self.work_dir is None:
            self.create_work_dir()

        return self.work_dir / filename

    def on_release(self, callback):
        """Register a zero-argument callable to run when this manager lets go
        of its work dir.

        For state a *caller* attaches to the run — the process-wide claim on a
        pinned ``work_dir`` that :mod:`uacpy.models.base` takes, so two threads
        cannot drive two models through one scratch directory. Such a claim has
        to come back on every exit path, which is what :meth:`finish` is for.
        The weakref finalizer stays as the backstop for a manager nobody
        finished — collection is not prompt, so it is a safety net, not the
        mechanism.

        Each callback runs at most once.
        """
        if not self._release_hooks:
            weakref.finalize(self, _run_release_hooks, self._release_hooks)
        self._release_hooks.append(callback)

    def finish(self):
        """Let go of the work directory at the end of a run.

        The single exit point every model's ``finally:`` uses, because the two
        halves of letting go are decided differently: whether to *remove* the
        scratch files is the caller's ``cleanup`` choice, while handing back
        the claim registered through :meth:`on_release` is unconditional.
        A pinned ``work_dir`` (where ``cleanup`` defaults to False) therefore
        releases its claim here too, rather than when the manager is
        collected — a failed run's manager is a local the traceback keeps
        alive, so a caller holding the exception would otherwise pin the
        claim indefinitely and refuse the next thread that uses the
        directory.
        """
        if self.cleanup:
            self.cleanup_work_dir()
        else:
            _run_release_hooks(self._release_hooks)

    def cleanup_work_dir(self):
        """Remove uacpy's scratch files.

        A directory uacpy created is removed whole. A caller-supplied one
        (see :meth:`adopt_work_dir`) keeps the directory and everything that
        was already in it — only entries this run added are removed, so
        ``cleanup=True`` on ``work_dir='.'`` cannot take the caller's tree.

        Cleanup failures (stale NFS lock, Windows file-handle held by a
        Fortran subprocess that hasn't reaped, etc.) are swallowed: a
        failed cleanup must never mask the original ``run()`` exception
        a caller is trying to surface.
        """
        # Ahead of the early return: a manager whose directory is already gone
        # still has to hand back whatever :meth:`on_release` registered. Also
        # ahead of the walk below, so an unreadable directory cannot cost the
        # hooks their run.
        _run_release_hooks(self._release_hooks)
        # ``exists()`` and ``iterdir()`` both raise on an unreadable
        # directory, which is exactly the "stale NFS lock" case the docstring
        # promises to swallow.
        try:
            if self.work_dir is None or not self.work_dir.exists():
                return
            if self._owns_work_dir:
                shutil.rmtree(self.work_dir, ignore_errors=True)
            else:
                preexisting = self._preexisting or set()
                for path in self.work_dir.iterdir():
                    if path.name in preexisting:
                        continue
                    if path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                    else:
                        try:
                            path.unlink()
                        except OSError:
                            pass
        except OSError:
            pass
        self.work_dir = None
        self._owns_work_dir = False
        self._preexisting = None

    def __enter__(self):
        # Reuse a directory the caller already created or adopted; creating a
        # second one here would strand the first (see :meth:`create_work_dir`).
        if self.work_dir is None or not self.work_dir.exists():
            self.create_work_dir()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

    def __repr__(self) -> str:
        tmpfs_str = "tmpfs" if self.use_tmpfs else "disk"
        work_str = str(self.work_dir) if self.work_dir else "not created"
        return f"FileManager({tmpfs_str}, work_dir={work_str})"


def setup_file_manager(work_dir, *, model_name: str, use_tmpfs: bool,
                       cleanup: bool) -> FileManager:
    """Build the FileManager of a run. ``work_dir`` (the run's pinned
    directory, :func:`_effective_work_dir`) is used as-is (not a parent);
    when ``None``, a fresh temp dir is created.

    Auto-creates the user-pinned ``work_dir`` if it doesn't exist
    yet, so callers can construct ``Model(work_dir='./out')`` without
    a separate ``mkdir`` step.

    ``cleanup`` drives the manager on both branches, so it stays the
    single decision :func:`~uacpy.models._extract.attach_output_paths` keys the ``*_file``
    metadata on: a directory that survives the run always has valid paths
    attached, and a wiped one never does.

    A pinned ``work_dir`` is also *claimed* for this thread (see
    :func:`_claim_work_dir`), so a second thread pointed at the same
    directory raises instead of silently trading results with this one.
    The claim comes back through the manager's release hook.
    """
    if work_dir is not None:
        claim = _claim_work_dir(work_dir, model_name)
        try:
            # base_dir is the *parent*: FileManager validates that it
            # exists, while ``adopt_work_dir`` keys ownership on whether
            # work_dir itself already existed, so it has to do that mkdir
            # itself.
            parent = Path(work_dir).parent
            parent.mkdir(parents=True, exist_ok=True)
            if use_tmpfs:
                warnings.warn(
                    f"{model_name}(use_tmpfs=True) is ignored for the "
                    f"pinned work_dir {work_dir} — a named directory "
                    f"cannot be relocated to /dev/shm. Drop work_dir= for "
                    f"RAM-backed I/O, or point work_dir at a tmpfs mount.",
                    FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
                )
            fm = FileManager(
                use_tmpfs=False,
                base_dir=parent,
                prefix=f'{model_name.lower()}_',
                cleanup=cleanup,
            )
            # Adopted, not owned: cleanup may remove only what this run adds.
            fm.adopt_work_dir(work_dir)
            _lock_work_dir_across_processes(claim, model_name)
            # FileManager's own writability check validated ``base_dir``,
            # i.e. the PARENT; the directory the decks actually go in is
            # this one, and a read-only one otherwise surfaced as a raw
            # PermissionError from the first writer.
            if not os.access(fm.work_dir, os.W_OK):
                raise ConfigurationError(
                    f"work_dir is not writable: {fm.work_dir}.",
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
            use_tmpfs=use_tmpfs,
            base_dir=None,
            prefix=f'{model_name.lower()}_',
            cleanup=cleanup,
        )
        fm.create_work_dir()

    return fm
