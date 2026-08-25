"""Tests for ``uacpy._log``: warning-source attribution, the styled
``warnings.formatwarning`` replacement, its install policy, and which stream
``log_message`` writes each severity to.

The formatter is process-global state (installed at ``import uacpy``), so
every test that touches ``warnings.formatwarning`` restores it. The stream
tests run each ``log_message`` in a fresh interpreter for the same reason —
the channel is decided at import, so an in-process check would read whatever
an earlier test left behind."""

import json
import os
import re
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

import uacpy
import uacpy._log as _log
from uacpy._log import (
    _source_from_filename,
    _uacpy_format_warning,
    install_warning_formatter,
)

_PKG_ROOT = Path(_log.__file__).resolve().parent
_REPO_ROOT = Path(uacpy.__file__).resolve().parents[1]


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


def test_importing_uacpy_replaces_the_genuine_stdlib_formatwarning():
    # The real article, in a pristine interpreter. Every other test in this
    # section hands `install_warning_formatter` a stand-in whose `__module__`
    # is set by hand, which pins the guard's mechanism but not the object it
    # exists to recognise; this one captures CPython's own
    # `warnings.formatwarning` before uacpy is imported and asserts the
    # import-time install replaced it.
    probe = (
        'import json, warnings\n'
        'stdlib_default = warnings.formatwarning\n'
        'import uacpy\n'
        'from uacpy._log import _uacpy_format_warning\n'
        'print(json.dumps({\n'
        '    "default_module": getattr(stdlib_default, "__module__", None),\n'
        '    "default_was_uacpys": stdlib_default is _uacpy_format_warning,\n'
        '    "installed": warnings.formatwarning is _uacpy_format_warning,\n'
        '}))\n'
    )
    env = dict(os.environ)
    env.pop('UACPY_NO_WARNING_FORMAT', None)
    run = subprocess.run([sys.executable, '-c', probe], env=env,
                         capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    seen = json.loads(run.stdout)
    # A pristine interpreter must hand back something that is not already ours,
    # otherwise the assertion below would pass without the install running.
    assert seen['default_was_uacpys'] is False, seen
    assert seen['installed'] is True, seen


def test_install_replaces_a_formatter_defined_in_the_warnings_module(
        restore_formatwarning):
    # Stand-in for the stdlib default, faked so the case runs in-process:
    # CPython defines `formatwarning` in the pure-Python `warnings` module.
    def stdlib_like(message, category, filename, lineno, line=None):
        return 'stdlib'

    stdlib_like.__module__ = 'warnings'
    warnings.formatwarning = stdlib_like
    install_warning_formatter()
    assert warnings.formatwarning is _uacpy_format_warning


def test_install_replaces_a_formatter_defined_in_the_c_accelerator(
        restore_formatwarning):
    # The other spelling the guard accepts as "stdlib": a build that sourced
    # the default from the `_warnings` accelerator rather than `warnings.py`.
    def accelerator_like(message, category, filename, lineno, line=None):
        return 'stdlib'

    accelerator_like.__module__ = '_warnings'
    warnings.formatwarning = accelerator_like
    install_warning_formatter()
    assert warnings.formatwarning is _uacpy_format_warning


def test_install_respects_a_host_formatter_from_a_module_named_warnings(
        restore_formatwarning):
    # A host application's own `myapp.warnings` submodule is not the stdlib.
    # Matching module names by suffix would clobber it, breaking the
    # host-installs-first contract `install_warning_formatter` documents.
    def host_formatter(message, category, filename, lineno, line=None):
        return 'host'

    host_formatter.__module__ = 'myapp.warnings'
    warnings.formatwarning = host_formatter
    install_warning_formatter()
    assert warnings.formatwarning is host_formatter


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


def _emit(level, verbose):
    """Run one ``log_message`` in a fresh interpreter; return (stdout, stderr)."""
    code = (
        "from uacpy._log import log_message\n"
        f"log_message('probe', 'the message', verbose={verbose!r}, "
        f"level={level!r})\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        timeout=120, cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    return result.stdout, result.stderr


@pytest.mark.parametrize("level", ["warn", "warning", "error"])
def test_problem_levels_go_to_stderr(level):
    """A warning interleaved into stdout cannot be separated from a script's
    real output by redirection, which is the whole reason a script redirects."""
    out, err = _emit(level, False)
    assert "the message" in err
    assert "the message" not in out


@pytest.mark.parametrize("level, verbose", [("info", True), ("debug", "debug")])
def test_status_levels_stay_on_stdout(level, verbose):
    out, err = _emit(level, verbose)
    assert "the message" in out
    assert "the message" not in err


def test_both_streams_emit_the_timestamp_level_source_message_shape():
    """``[ts] [LEVEL] [source] message`` — the format the warning formatter
    matches, and what any log scraper keys on."""
    pattern = re.compile(
        r"^\[\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2} UTC\] \[(WARN|INFO)\] "
        r"\[probe\] the message$"
    )
    assert pattern.match(_emit("warn", False)[1].strip())
    assert pattern.match(_emit("info", True)[0].strip())


def _formatter_module(env_value):
    """The module ``warnings.formatwarning`` belongs to after ``import uacpy``
    in a fresh interpreter, with ``UACPY_NO_WARNING_FORMAT`` set as given."""
    import os
    env = dict(os.environ)
    if env_value is None:
        env.pop('UACPY_NO_WARNING_FORMAT', None)
    else:
        env['UACPY_NO_WARNING_FORMAT'] = env_value
    result = subprocess.run(
        [sys.executable, "-c",
         "import warnings, uacpy; print(warnings.formatwarning.__module__)"],
        capture_output=True, text=True, timeout=120, env=env,
        cwd=str(_REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_importing_uacpy_installs_the_warning_formatter():
    assert _formatter_module(None) == 'uacpy._log'


def test_the_warning_formatter_can_be_declined():
    """``warnings.formatwarning`` is process-global — installing it reformats
    every other library's warnings too — so a host gets the same opt-out
    spelling :mod:`uacpy._stack` gives its RLIMIT change."""
    assert _formatter_module('1') == 'warnings'


@pytest.mark.parametrize("value", ['0', 'false', 'no', ''])
def test_a_falsey_opt_out_installs_it(value):
    """Truthy opt-out only: setting 0 means "do not disable"."""
    assert _formatter_module(value) == 'uacpy._log'


class TestWarningSourceTags:
    """``_source_from_filename`` tags an ``__init__.py`` as its package and
    any other in-package module by its dotted path; a path outside the
    package keeps its bare stem."""

    def test_package_init_is_tagged_as_the_package(self):
        import uacpy
        assert _source_from_filename(uacpy.__file__) == 'uacpy'

    def test_subpackage_init_is_tagged_as_the_subpackage(self):
        import uacpy.models
        assert _source_from_filename(uacpy.models.__file__) == 'uacpy.models'

    def test_module_keeps_its_dotted_path(self):
        import uacpy._log
        assert _source_from_filename(uacpy._log.__file__) == 'uacpy._log'

    def test_path_outside_the_package_keeps_its_stem(self):
        assert _source_from_filename('/somewhere/else/script.py') == 'script'
