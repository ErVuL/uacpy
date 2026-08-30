"""File manager for acoustic model I/O with optional RAM-disk support."""

import os
import tempfile
import shutil
import weakref
from pathlib import Path
from typing import Optional, Union

from uacpy.core.exceptions import ConfigurationError


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
        elif use_tmpfs and self._tmpfs_available():
            self.base_dir = Path('/dev/shm')
            self.use_tmpfs = True
        else:
            self.base_dir = Path(tempfile.gettempdir())

        if not self.base_dir.exists():
            raise ConfigurationError(
                f"Base directory does not exist: {self.base_dir}",
                remediation="Create it, or pass a writable base_dir=.")
        if not os.access(self.base_dir, os.W_OK):
            raise ConfigurationError(
                f"Base directory not writable: {self.base_dir}",
                remediation="Pass a writable base_dir= (or fix its permissions).")

    @staticmethod
    def _tmpfs_available() -> bool:
        """Return True if tmpfs (``/dev/shm``) is available and writable."""
        shm_path = Path('/dev/shm')
        return shm_path.exists() and os.access(shm_path, os.W_OK)

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
        ``finally: if fm.cleanup: fm.cleanup_work_dir()`` conflated them, so a
        pinned ``work_dir`` (where ``cleanup`` defaults to False) kept its
        claim until the manager was collected — and a failed run's manager is
        a local kept alive by the traceback, so any caller holding the
        exception pins it indefinitely and the next thread to use that
        directory is refused.
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
