"""File manager for acoustic model I/O with optional RAM-disk support."""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Union
from contextlib import contextmanager


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
        Automatically cleanup files on exit. Default is True.

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

        if base_dir is not None:
            self.base_dir = Path(base_dir)
        elif use_tmpfs and self._tmpfs_available():
            self.base_dir = Path('/dev/shm')
        else:
            self.base_dir = Path(tempfile.gettempdir())

        if not self.base_dir.exists():
            raise ValueError(f"Base directory does not exist: {self.base_dir}")
        if not os.access(self.base_dir, os.W_OK):
            raise ValueError(f"Base directory not writable: {self.base_dir}")

    @staticmethod
    def _tmpfs_available() -> bool:
        """Return True if tmpfs (``/dev/shm``) is available and writable."""
        shm_path = Path('/dev/shm')
        return shm_path.exists() and os.access(shm_path, os.W_OK)

    def create_work_dir(self, name: Optional[str] = None) -> Path:
        """
        Create a working directory for model files.

        Parameters
        ----------
        name : str, optional
            Directory name. If ``None``, a unique name is generated.

        Returns
        -------
        work_dir : Path
            Path to the working directory.
        """
        if name is not None:
            self.work_dir = self.base_dir / name
            self.work_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._temp_dir = tempfile.mkdtemp(
                prefix=self.prefix,
                dir=str(self.base_dir)
            )
            self.work_dir = Path(self._temp_dir)

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
        """Remove the working directory and all of its contents."""
        if self.work_dir is not None and self.work_dir.exists():
            shutil.rmtree(self.work_dir)
            self.work_dir = None
            self._temp_dir = None

    def list_files(self, pattern: str = '*') -> list:
        """
        List files in the working directory.

        Parameters
        ----------
        pattern : str, optional
            Glob pattern for filtering. Default is ``'*'`` (all files).

        Returns
        -------
        files : list of Path
            List of matching file paths.
        """
        if self.work_dir is None or not self.work_dir.exists():
            return []

        return list(self.work_dir.glob(pattern))

    def copy_file(self, src: Union[str, Path], dst_name: Optional[str] = None) -> Path:
        """
        Copy a file into the working directory.

        Parameters
        ----------
        src : str or Path
            Source file path.
        dst_name : str, optional
            Destination filename. If ``None``, uses the source filename.

        Returns
        -------
        dst_path : Path
            Path to the copied file in the working directory.
        """
        src = Path(src)
        if dst_name is None:
            dst_name = src.name

        dst = self.get_path(dst_name)
        shutil.copy2(src, dst)
        return dst

    def __enter__(self):
        self.create_work_dir()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cleanup:
            self.cleanup_work_dir()

    def __del__(self):
        if self.cleanup and hasattr(self, 'work_dir'):
            try:
                self.cleanup_work_dir()
            except Exception:
                pass

    def __repr__(self) -> str:
        tmpfs_str = "tmpfs" if self.use_tmpfs else "disk"
        work_str = str(self.work_dir) if self.work_dir else "not created"
        return f"FileManager({tmpfs_str}, work_dir={work_str})"


@contextmanager
def temporary_directory(use_tmpfs: bool = False, prefix: str = 'uacpy_'):
    """
    Yield a temporary directory that is cleaned up on exit.

    Parameters
    ----------
    use_tmpfs : bool, optional
        Use RAM-based filesystem. Default is False.
    prefix : str, optional
        Directory name prefix.

    Yields
    ------
    path : Path
        Path to the temporary directory.

    Examples
    --------
    >>> with temporary_directory(use_tmpfs=True) as tmpdir:
    ...     file_path = tmpdir / 'test.txt'
    ...     file_path.write_text('data')
    """
    fm = FileManager(use_tmpfs=use_tmpfs, prefix=prefix, cleanup=True)
    try:
        yield fm.create_work_dir()
    finally:
        fm.cleanup_work_dir()
