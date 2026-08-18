"""The lazy-import promise of ``import uacpy`` (PEP 562).

``uacpy/__init__`` eagerly loads only :mod:`uacpy.core`; the model
wrappers, plotting, DSP and data subpackages — and with them scipy and
matplotlib — are paid for on first attribute access. Every test here runs
in a fresh subprocess because the promise is about a cold interpreter:
the pytest process itself has long since imported everything.

The resolution test walks ``_LAZY_ATTRS`` / ``_LAZY_SUBMODULES`` from the
live module, so a typo'd table entry — which otherwise explodes only at a
user's first attribute access — fails here instead.
"""

import os
import subprocess
import sys
from pathlib import Path

import uacpy

_REPO_ROOT = Path(uacpy.__file__).parent.parent


def _run_python(code: str) -> subprocess.CompletedProcess:
    """Run ``code`` in a cold interpreter with the in-tree uacpy first."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO_ROOT), env.get("PYTHONPATH", "")]
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=120, env=env,
    )


def _assert_clean_exit(result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, (
        f"subprocess failed (rc={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )


def test_import_uacpy_leaves_scipy_and_matplotlib_unloaded():
    """``import uacpy`` must not register scipy or matplotlib in
    ``sys.modules`` — importing any of their submodules would register the
    top-level name, so the two keys cover the whole families."""
    result = _run_python(
        "import sys\n"
        "import uacpy\n"
        "for heavy in ('scipy', 'matplotlib'):\n"
        "    assert heavy not in sys.modules, (\n"
        "        f'{heavy} was imported eagerly by import uacpy'\n"
        "    )\n"
    )
    _assert_clean_exit(result)


def test_import_uacpy_io_leaves_scipy_and_matplotlib_unloaded():
    """The io layer reads and writes decks with numpy alone; its one
    scipy-adjacent dependency (acoustic_signal.waveforms in the SPARC
    readers) is function-local, so ``import uacpy.io`` stays light."""
    result = _run_python(
        "import sys\n"
        "import uacpy.io\n"
        "for heavy in ('scipy', 'matplotlib'):\n"
        "    assert heavy not in sys.modules, (\n"
        "        f'{heavy} was imported eagerly by import uacpy.io'\n"
        "    )\n"
    )
    _assert_clean_exit(result)


def test_every_lazy_table_entry_resolves():
    """Every name in ``_LAZY_ATTRS`` and ``_LAZY_SUBMODULES`` resolves via
    ``getattr(uacpy, name)``, and resolution caches the value so
    ``__getattr__`` is not hit twice for the same name."""
    result = _run_python(
        "import uacpy\n"
        "failures = []\n"
        "names = list(uacpy._LAZY_SUBMODULES) + list(uacpy._LAZY_ATTRS)\n"
        "assert names, 'lazy tables are empty — the surface moved?'\n"
        "for name in names:\n"
        "    try:\n"
        "        value = getattr(uacpy, name)\n"
        "    except Exception as exc:\n"
        "        failures.append(f'{name}: {type(exc).__name__}: {exc}')\n"
        "        continue\n"
        "    if vars(uacpy).get(name) is not value:\n"
        "        failures.append(f'{name}: resolved but not cached')\n"
        "if failures:\n"
        "    raise SystemExit('unresolvable lazy entries:\\n'\n"
        "                     + '\\n'.join(failures))\n"
    )
    _assert_clean_exit(result)


def test_lazy_names_are_advertised():
    """The lazy surface is discoverable: every table entry appears in
    ``dir(uacpy)``, so tab completion and ``__all__`` agree with PEP 562
    resolution. Runs in-process — it inspects tables, not import order."""
    advertised = set(dir(uacpy))
    lazy = set(uacpy._LAZY_SUBMODULES) | set(uacpy._LAZY_ATTRS)
    missing = lazy - advertised
    assert not missing, f"lazy names absent from dir(uacpy): {sorted(missing)}"


def test_bare_absorption_formulas_stay_off_the_top_level():
    """Only the class wrappers (Thorp, FrancoisGarrison, ...) are re-exported
    at the top level; the bare per-km formula functions stay addressed as
    ``uacpy.core.absorption.*``. Runs in-process — it inspects the surface,
    not import order."""
    assert not hasattr(uacpy, 'thorp_db_per_km')
    assert not hasattr(uacpy, 'francois_garrison_db_per_km')
    # The class spellings and the fully-qualified functions remain available.
    assert hasattr(uacpy, 'Thorp') and hasattr(uacpy, 'FrancoisGarrison')
    from uacpy.core.absorption import thorp_db_per_km          # noqa: F401
    from uacpy.core.absorption import francois_garrison_db_per_km  # noqa: F401


def test_uacpy_plot_is_an_attribute_alias_not_a_module_path():
    """docs/guide/plotting.md §1: ``uacpy.plot`` aliases
    ``uacpy.visualization.plots`` (so ``from uacpy.plot import ...`` raises
    ``ModuleNotFoundError``), and exactly four conveniences are re-exported
    at the top level as the same objects."""
    result = _run_python(
        "import uacpy\n"
        "import uacpy.visualization.plots as plots\n"
        "assert uacpy.plot is plots\n"
        "for name in ('plot_result', 'plot_field', 'plot_overview',\n"
        "             'compare_models'):\n"
        "    assert getattr(uacpy, name) is getattr(plots, name), name\n"
        "try:\n"
        "    from uacpy.plot import plot_field  # noqa: F401\n"
        "except ModuleNotFoundError:\n"
        "    pass\n"
        "else:\n"
        "    raise SystemExit('from uacpy.plot import ... did not raise')\n"
    )
    _assert_clean_exit(result)


def test_importing_visualization_leaves_rcparams_untouched():
    """docs/guide/plotting.md: importing the plotting surface must not
    modify ``matplotlib.rcParams`` — the user's own style sheet survives.
    Cold subprocess: snapshot rcParams, resolve ``uacpy.plot``, diff."""
    result = _run_python(
        "import matplotlib\n"
        "before = dict(matplotlib.rcParams)\n"
        "import uacpy\n"
        "uacpy.plot  # resolves the alias -> imports uacpy.visualization\n"
        "changed = [k for k, v in matplotlib.rcParams.items()\n"
        "           if before.get(k) != v]\n"
        "assert not changed, f'rcParams touched: {changed}'\n"
    )
    _assert_clean_exit(result)
