"""Single output channel for uacpy status / debug / warning text.

Every module that needs to emit a tagged line — models, writers,
readers — calls :func:`log_message`. The ``verbose`` argument controls
which severity levels reach stdout:

``False`` / ``None`` / ``'off'`` / ``'silent'``
    Only ``WARN`` and ``ERROR`` print. Default.
``True`` / ``'info'``
    ``INFO`` + ``WARN`` + ``ERROR`` print. ``DEBUG`` is suppressed.
``'debug'``
    Everything prints, including ``DEBUG``.

Format on stdout:

``[YYYY/MM/DD HH:MM:SS UTC] [LEVEL] [source] message``

Genuine user-facing problems still go through :mod:`warnings` (typed
``UserWarning``) or a typed exception in :mod:`uacpy.core.exceptions`.
``WARN`` / ``ERROR`` here are for status banners that don't fit
either of those (e.g. "field.exe exited non-zero but the .shd is
readable — continuing").
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Union


_LEVEL_VALUE = {
    'debug': 10,
    'info': 20,
    'warn': 30,
    'warning': 30,
    'error': 40,
}

_VERBOSE_THRESHOLD = {
    'off': 30,
    'silent': 30,
    'info': 20,
    'debug': 10,
}


def _resolve_threshold(verbose: Union[bool, str, None]) -> int:
    """Map a ``verbose=`` argument to the minimum level value that
    prints. Levels strictly below the threshold are dropped.

    ``False`` / ``None`` / ``'off'`` / ``'silent'`` → 30 (WARN+ only).
    ``True`` / ``'info'``                          → 20 (INFO+).
    ``'debug'``                                    → 10 (everything).
    """
    if verbose is None or verbose is False:
        return 30
    if verbose is True:
        return 20
    key = str(verbose).lower()
    if key not in _VERBOSE_THRESHOLD:
        raise ValueError(
            f"verbose={verbose!r} not recognized. "
            f"Valid: False/True/'off'/'silent'/'info'/'debug'."
        )
    return _VERBOSE_THRESHOLD[key]


def log_message(
    source: str,
    message: str,
    *,
    verbose: Union[bool, str, None] = False,
    level: str = "info",
) -> None:
    """Level-tagged ``print`` shared by all uacpy modules.

    Parameters
    ----------
    source : str
        Short tag for the caller (e.g. ``'Bellhop'``, ``'bellhop_writer'``,
        ``'bathy_io'``).
    message : str
        Free-form text. The function prepends a UTC timestamp, level
        label, and source tag.
    verbose : bool or str, optional
        Minimum-severity gate. ``False`` / ``True`` / ``'off'`` /
        ``'info'`` / ``'debug'`` — see module docstring. Default
        ``False`` (only ``WARN`` / ``ERROR`` print).
    level : {'info', 'debug', 'warn', 'warning', 'error'}, optional
        Severity of *this* message. Default ``'info'``.
    """
    lvl = level.lower()
    if lvl not in _LEVEL_VALUE:
        raise ValueError(
            f"log_message: unknown level={level!r}. "
            f"Valid: {sorted(_LEVEL_VALUE)}."
        )
    if _LEVEL_VALUE[lvl] < _resolve_threshold(verbose):
        return
    label = {'warning': 'WARN'}.get(lvl, lvl.upper())
    ts = datetime.now(timezone.utc).strftime("%Y/%m/%d %H:%M:%S UTC")
    print(f"[{ts}] [{label}] [{source}] {message}")


def _source_from_filename(filename: str) -> str:
    """Map a Python source path to a dotted module-ish tag for warnings.

    ``/.../uacpy/models/bellhop.py`` → ``'uacpy.models.bellhop'``;
    paths outside the package are returned as the bare file stem.
    """
    try:
        parts = Path(filename).resolve().parts
        if 'uacpy' in parts:
            i = len(parts) - 1 - parts[::-1].index('uacpy')
            tail = parts[i:]
            return '.'.join([*tail[:-1], Path(tail[-1]).stem])
    except (ValueError, OSError):
        pass
    return Path(filename).stem


def _uacpy_format_warning(
    message, category, filename, lineno, line=None,  # noqa: ARG001
) -> str:
    """Custom :func:`warnings.formatwarning` that matches :func:`log_message`.

    Output: ``[YYYY/MM/DD HH:MM:SS UTC] [<CATEGORY>] [<module>:<lineno>] message``.
    Installed by :func:`install_warning_formatter`.
    """
    ts = datetime.now(timezone.utc).strftime("%Y/%m/%d %H:%M:%S UTC")
    cat_name = getattr(category, '__name__', str(category))
    label = 'WARN' if cat_name == 'UserWarning' else cat_name.replace(
        'Warning', '',
    ).upper() or 'WARN'
    source = f"{_source_from_filename(filename)}:{lineno}"
    return f"[{ts}] [{label}] [{source}] {message}\n"


def install_warning_formatter() -> None:
    """Replace :data:`warnings.formatwarning` with the uacpy-styled version.

    Idempotent. Called once at package import time so every
    ``warnings.warn(...)`` raised by uacpy code (and anywhere else in
    the process) renders in the same ``[ts] [LEVEL] [source] msg`` shape
    as :func:`log_message`. Python's filtering, ``pytest.warns``,
    ``simplefilter('error')`` and friends keep working unchanged — only
    the rendered string is replaced.
    """
    warnings.formatwarning = _uacpy_format_warning
