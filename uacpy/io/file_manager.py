"""File manager for acoustic model I/O with optional RAM-disk support."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Union

from uacpy.core.exceptions import ConfigurationError


class FileManager:
    """
    Manage temporary files for acoustic models, optionally on tmpfs.

    Provides automatic cleanup of temporary files and optional placement in
    a RAM-based filesystem for improved I/O performance.

    Parameters
    ----------
    use_tmpfs : bool, optional
        Use RAM-based tmpfs filesystem. Default is False. On Linux, uses
        ``/dev/shm`` if available.
    base_dir : str or Path, optional
        Base directory for file operations. If ``None``, uses the system
        temp directory.
    prefix : str, optional
        Prefix for temporary directory names. Default is ``'uacpy_'``.
    cleanup : bool, optional
        Cleanup on ``__exit__``. Default is True. Callers driving the
        manager without ``with`` read the flag themselves — the model recipe
        (``models/base.py:29-31``) calls :meth:`cleanup_work_dir` from a
        ``finally``, so a run cleans up the same way whether it succeeded or
        raised. What *this* class decides is how much gets removed; see
        :meth:`cleanup_work_dir`.

    Attributes
    ----------
    work_dir : Path
        Current working directory for model files.
    use_tmpfs : bool
        Whether tmpfs is being used.
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
        self.use_tmpfs = use_tmpfs
        self.prefix = prefix
        self.cleanup = cleanup
        self.work_dir = None
        self._temp_dir = None
        # Set when uacpy created the work dir itself, so cleanup may remove it
        # whole. A caller-pinned directory is adopted via ``adopt_work_dir``,
        # which records what was already there so cleanup spares it.
        self._owns_work_dir = False
        self._preexisting = None

        if base_dir is not None:
            self.base_dir = Path(base_dir)
        elif use_tmpfs and self._tmpfs_available():
            self.base_dir = Path('/dev/shm')
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

        Returns
        -------
        work_dir : Path
            Path to the working directory.
        """
        self._temp_dir = tempfile.mkdtemp(
            prefix=self.prefix,
            dir=str(self.base_dir)
        )
        self.work_dir = Path(self._temp_dir)
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
        """
        self.work_dir = Path(work_dir)
        existed = self.work_dir.exists()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._owns_work_dir = not existed
        self._temp_dir = None
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
        self.work_dir = None
        self._temp_dir = None

    def __enter__(self):
        self.create_work_dir()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup:
            self.cleanup_work_dir()

    def __repr__(self) -> str:
        tmpfs_str = "tmpfs" if self.use_tmpfs else "disk"
        work_str = str(self.work_dir) if self.work_dir else "not created"
        return f"FileManager({tmpfs_str}, work_dir={work_str})"
