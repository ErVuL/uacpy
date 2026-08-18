"""Tests for ``uacpy._log``: warning-source attribution, the styled
``warnings.formatwarning`` replacement, and its install policy.

The formatter is process-global state (installed at ``import uacpy``), so
every test that touches ``warnings.formatwarning`` restores it."""

import re
import warnings
from pathlib import Path

import pytest

import uacpy._log as _log
from uacpy._log import (
    _source_from_filename,
    _uacpy_format_warning,
    install_warning_formatter,
)

_PKG_ROOT = Path(_log.__file__).resolve().parent


@pytest.fixture
def restore_formatwarning():
    saved = warnings.formatwarning
    yield
    warnings.formatwarning = saved


# ─── _source_from_filename ─────────────────────────────────────────────────


def test_paths_inside_the_package_become_dotted_modules():
    assert _source_from_filename(
        str(_PKG_ROOT / 'models' / 'bellhop.py')) == 'uacpy.models.bellhop'
    assert _source_from_filename(str(_PKG_ROOT / '_log.py')) == 'uacpy._log'


def test_site_packages_paths_keep_the_bare_stem():
    # Third-party code must never be dressed as uacpy — a uacpy-prefixed
    # source here would route a dependency's warning into the pyproject
    # deprecation gate, inverting its scope.
    source = _source_from_filename(
        '/opt/venv/lib/python3.12/site-packages/numpy/_core/numeric.py')
    assert source == 'numeric'
    assert not source.startswith('uacpy')


def test_user_scripts_keep_the_bare_stem():
    assert _source_from_filename('/home/someone/run_survey.py') == 'run_survey'


def test_stdin_keeps_its_pseudo_filename():
    source = _source_from_filename('<stdin>')
    assert source == '<stdin>'
    assert not source.startswith('uacpy')


# ─── formatter output shape ────────────────────────────────────────────────


def test_formatter_renders_the_log_message_shape():
    out = _uacpy_format_warning(
        'grid too coarse', UserWarning,
        str(_PKG_ROOT / 'models' / 'ram.py'), 42)
    assert re.fullmatch(
        r"\[\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2} UTC\] \[WARN\] "
        r"\[uacpy\.models\.ram:42\] grid too coarse\n",
        out,
    ), out


def test_formatter_labels_by_category():
    def label(category):
        out = _uacpy_format_warning('m', category, '/x/script.py', 1)
        return re.match(r"\[[^]]+\] \[([^]]+)\]", out).group(1)

    # UserWarning matches log_message's WARN; other categories keep their
    # name minus the 'Warning' suffix; the bare Warning base class (empty
    # after the strip) falls back to WARN.
    assert label(UserWarning) == 'WARN'
    assert label(DeprecationWarning) == 'DEPRECATION'
    assert label(RuntimeWarning) == 'RUNTIME'
    assert label(Warning) == 'WARN'


# ─── interpreter-shutdown fallback ─────────────────────────────────────────


def test_torn_down_datetime_falls_back_to_the_plain_shape(monkeypatch):
    # During interpreter shutdown a __del__-time warning can arrive after
    # sys.modules teardown has nulled this module's globals; the formatter
    # must still return a line rather than raise and lose the warning.
    monkeypatch.setattr(_log, 'datetime', None)
    out = _uacpy_format_warning('late warning', UserWarning, 'plot.py', 7)
    assert out == 'plot.py:7: UserWarning: late warning\n'


def test_torn_down_path_falls_back_too(monkeypatch):
    # Same guard, torn one level deeper: _source_from_filename itself
    # raising (Path nulled) is caught by the formatter's fallback.
    monkeypatch.setattr(_log, 'Path', None)
    out = _uacpy_format_warning('late warning', UserWarning, 'plot.py', 7)
    assert out == 'plot.py:7: UserWarning: late warning\n'


# ─── install_warning_formatter policy ──────────────────────────────────────


def test_install_replaces_the_stdlib_default(restore_formatwarning):
    def stdlib_like(message, category, filename, lineno, line=None):
        return 'stdlib'

    stdlib_like.__module__ = 'warnings'
    warnings.formatwarning = stdlib_like
    install_warning_formatter()
    assert warnings.formatwarning is _uacpy_format_warning


def test_install_is_idempotent(restore_formatwarning):
    warnings.formatwarning = _uacpy_format_warning
    install_warning_formatter()
    assert warnings.formatwarning is _uacpy_format_warning


def test_install_respects_a_host_applications_formatter(restore_formatwarning):
    def host_formatter(message, category, filename, lineno, line=None):
        return 'host'

    warnings.formatwarning = host_formatter
    install_warning_formatter()
    assert warnings.formatwarning is host_formatter
