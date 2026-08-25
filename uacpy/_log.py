"""Single output channel for uacpy status / debug / warning text.

Every module that needs to emit a tagged line — models, writers,
readers — calls :func:`log_message`. The ``verbose`` argument controls
which severity levels print:

``False`` / ``None`` / ``'off'`` / ``'silent'``
    Only ``WARN`` and ``ERROR`` print. Default.
``True`` / ``'info'``
    ``INFO`` + ``WARN`` + ``ERROR`` print. ``DEBUG`` is suppressed.
``'debug'``
    Everything prints, including ``DEBUG``.

``DEBUG`` / ``INFO`` go to stdout and ``WARN`` / ``ERROR`` to stderr, so a
script's own stdout stays separable from uacpy's problem reports by plain
shell redirection.

Format:

``[YYYY/MM/DD HH:MM:SS UTC] [LEVEL] [source] message``

Genuine user-facing problems still go through :mod:`warnings` (typed
``UserWarning``) or a typed exception in :mod:`uacpy.core.exceptions`.
``WARN`` / ``ERROR`` here are for status banners that don't fit
either of those (e.g. "field.exe exited non-zero but the .shd is
readable — continuing").
"""

from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

from uacpy.core.exceptions import ConfigurationError


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
        raise ConfigurationError(
            f"verbose={verbose!r} not recognized.",
            remediation="Use False/True/'off'/'silent'/'info'/'debug'.",
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
        raise ConfigurationError(
            f"log_message: unknown level={level!r}.",
            remediation=f"Use one of {sorted(_LEVEL_VALUE)}.",
        )
    if _LEVEL_VALUE[lvl] < _resolve_threshold(verbose):
        return
    label = {'warning': 'WARN'}.get(lvl, lvl.upper())
    ts = datetime.now(timezone.utc).strftime("%Y/%m/%d %H:%M:%S UTC")
    # WARN / ERROR go to stderr, the stream a caller can separate from a
    # script's real output by redirection; DEBUG / INFO stay on stdout.
    stream = sys.stderr if _LEVEL_VALUE[lvl] >= 30 else sys.stdout
    print(f"[{ts}] [{label}] [{source}] {message}", file=stream)


def _source_from_filename(filename: str) -> str:
    """Map a Python source path to a dotted module-ish tag for warnings.

    ``/.../uacpy/models/bellhop.py`` → ``'uacpy.models.bellhop'``; an
    ``__init__.py`` is tagged as its package (``/.../uacpy/__init__.py`` →
    ``'uacpy'``); paths outside the package are returned as the bare file
    stem.
    """
    try:
        resolved = Path(filename).resolve()
        pkg_root = Path(__file__).resolve().parent   # the uacpy package dir
        rel = resolved.relative_to(pkg_root)
        parts = ['uacpy', *rel.parts[:-1]]
        if rel.stem != '__init__':
            parts.append(rel.stem)
        return '.'.join(parts)
    except (ValueError, OSError):
        # Outside the package (site-packages, <stdin>, user scripts): keep
        # the bare stem — never dress third-party code as uacpy.
        pass
    return Path(filename).stem


def _uacpy_format_warning(
    message, category, filename, lineno, line=None,  # noqa: ARG001
) -> str:
    """Custom :func:`warnings.formatwarning` that matches :func:`log_message`.

    Output: ``[YYYY/MM/DD HH:MM:SS UTC] [<CATEGORY>] [<module>:<lineno>] message``.
    Installed by :func:`install_warning_formatter`.

    Guarded against interpreter-shutdown teardown: when matplotlib (or
    any other library) emits a warning during ``__del__`` after Python
    has started tearing down ``sys.modules``, ``datetime`` /
    ``_source_from_filename`` may be unavailable. Fall back to a plain
    str(message) in that case so the warning isn't lost.
    """
    try:
        ts = datetime.now(timezone.utc).strftime("%Y/%m/%d %H:%M:%S UTC")
        cat_name = getattr(category, '__name__', str(category))
        label = 'WARN' if cat_name == 'UserWarning' else cat_name.replace(
            'Warning', '',
        ).upper() or 'WARN'
        source = f"{_source_from_filename(filename)}:{lineno}"
        return f"[{ts}] [{label}] [{source}] {message}\n"
    except Exception:
        return f"{filename}:{lineno}: {category.__name__}: {message}\n"


def install_warning_formatter() -> None:
    """Replace :data:`warnings.formatwarning` with the uacpy-styled version.

    Idempotent. Called once at package import time so every
    ``warnings.warn(...)`` raised by uacpy code (and anywhere else in
    the process) renders in the same ``[ts] [LEVEL] [source] msg`` shape
    as :func:`log_message`. Python's filtering, ``pytest.warns``,
    ``simplefilter('error')`` and friends keep working unchanged — only
    the rendered string is replaced.

    ``warnings.formatwarning`` is process-global, so this reformats the
    warnings of every library in the process, not just uacpy's — the price of
    one consistent rendering for a package used as an application. Two escapes,
    in the order they are checked: ``UACPY_NO_WARNING_FORMAT=1`` keeps Python's
    own rendering (the same truthy opt-out spelling :mod:`uacpy._stack` uses
    for its process-global RLIMIT change), and a host application that
    installed its own formatter first keeps it — only the stdlib default (or a
    previous install of this one) is replaced.
    """
    # Truthy opt-out only: '0'/'false'/'no' keep the default behaviour
    # (installing), since someone setting 0 means "do not disable".
    if os.environ.get('UACPY_NO_WARNING_FORMAT', '').strip().lower() not in (
            '', '0', 'false', 'no'):
        return
    current = warnings.formatwarning
    # A formatter that is neither the stdlib's nor a previous install of this
    # one was set by the host application first — respect it rather than
    # clobbering process-wide rendering. Both module spellings count as the
    # stdlib default: CPython defines `formatwarning` in the pure-Python
    # `warnings` module, but the C accelerator `_warnings` shadows parts of it,
    # so a build that sourced the default from there must still be replaced.
    # An `endswith('warnings')` test would be wrong here — a host application
    # with its own `myapp.warnings` submodule would match and get clobbered.
    if (current is not _uacpy_format_warning
            and getattr(current, '__module__', '') not in (
                'warnings', '_warnings')):
        return
    warnings.formatwarning = _uacpy_format_warning
