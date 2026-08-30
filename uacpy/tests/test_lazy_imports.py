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

import pytest

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


_INIT_PATH = Path(uacpy.__file__).resolve()


def _statically_re_imported_targets():
    """``{bound name: (module, attribute)}`` for every ``from ... import ...``
    inside the ``if TYPE_CHECKING:`` block of ``uacpy/__init__.py``.

    ``from uacpy import models`` yields ``('uacpy', 'models')``, which names
    the module ``uacpy.models``; ``from uacpy.models import Bellhop`` yields
    ``('uacpy.models', 'Bellhop')``, which is the ``_LAZY_ATTRS`` entry
    verbatim. One rule reads both tables' spellings."""
    import ast

    tree = ast.parse(_INIT_PATH.read_text(encoding='utf-8'))
    targets = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        guard = (test.id if isinstance(test, ast.Name)
                 else test.attr if isinstance(test, ast.Attribute) else '')
        if not guard.endswith('TYPE_CHECKING'):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.ImportFrom):
                for alias in child.names:
                    targets[alias.asname or alias.name] = (
                        child.module or '', alias.name)
    return targets


_STATIC_MIRROR = _statically_re_imported_targets()


def test_the_lazy_surface_has_a_static_mirror_for_type_checkers():
    """The sweep itself, so an empty block cannot pass the gate below.

    If the ``if TYPE_CHECKING:`` block is ever deleted, this says so rather
    than letting the comparison succeed against two empty sets."""
    assert _STATIC_MIRROR, (
        "uacpy/__init__.py has no `if TYPE_CHECKING:` block re-importing the "
        "lazy names; without it `from uacpy import Bellhop; Bellhop()` is a "
        "pyright error on a py.typed package (PEP 562 + PEP 561)")


def test_every_lazy_name_is_statically_re_imported_for_type_checkers():
    """``__getattr__`` is what resolves the lazy tables at runtime, and a
    checker reading its two return paths infers ``ModuleType | Any`` — so a
    downstream ``Bellhop()`` is reported as calling a module, and 33 of the 77
    exported names reveal as ``Any``. The ``if TYPE_CHECKING:`` block in
    ``uacpy/__init__.py`` restates the same names as ordinary imports, which
    is what a checker reads instead.

    Both directions and the targets are compared: a name added to a table and
    not to the block is untyped downstream, a name in the block and not in a
    table is a promise nothing resolves, and a block entry pointing at the
    wrong module types the name as the wrong object with nothing failing.

    mypy does not report the underlying error — it treats an unannotated
    ``__getattr__`` as untyped and hands back ``Any`` — so a mypy-only check
    cannot stand in for this gate.
    """
    lazy = {name: (target.rsplit('.', 1)[0], target.rsplit('.', 1)[1])
            for name, target in uacpy._LAZY_SUBMODULES.items()}
    lazy.update(uacpy._LAZY_ATTRS)

    missing = sorted(set(lazy) - set(_STATIC_MIRROR))
    assert not missing, (
        f"lazy name(s) {missing} are absent from the `if TYPE_CHECKING:` "
        f"block in uacpy/__init__.py, so uacpy.<name> resolves to "
        f"`ModuleType | Any` for a downstream type checker")

    extra = sorted(set(_STATIC_MIRROR) - set(lazy))
    assert not extra, (
        f"the `if TYPE_CHECKING:` block re-imports {extra}, which no lazy "
        f"table resolves — uacpy.<name> raises AttributeError at runtime "
        f"while a type checker says it exists")

    mismatched = {name: (_STATIC_MIRROR[name], lazy[name])
                  for name in sorted(lazy)
                  if _STATIC_MIRROR[name] != lazy[name]}
    assert not mismatched, (
        f"the `if TYPE_CHECKING:` block points at a different target than "
        f"the lazy table for {mismatched} (static, runtime)")


def test_every_model_wrapper_is_reachable_from_the_top_level():
    """The reverse direction of ``test_every_lazy_table_entry_resolves``.

    That test walks ``_LAZY_ATTRS`` and asks whether each entry resolves; it
    cannot see a wrapper that never got an entry. Such a model is importable
    as ``uacpy.models.NewModel`` and raises ``AttributeError`` as
    ``uacpy.NewModel``, with nothing failing. Runs in-process: importing
    ``uacpy.models`` here does not affect the cold-import tests above, which
    each run in their own subprocess."""
    from uacpy.tests.conftest import concrete_model_classes

    wrappers = set(concrete_model_classes())
    reachable = set(uacpy._LAZY_ATTRS) | set(vars(uacpy))
    missing = sorted(wrappers - reachable)
    assert not missing, (
        f"model wrapper(s) {missing} are exported by uacpy.models but absent "
        f"from _LAZY_ATTRS in uacpy/__init__.py, so uacpy.<name> raises "
        f"AttributeError (docs/DEV.md section 3, step 4)")

    for name in sorted(wrappers):
        assert getattr(uacpy, name) is concrete_model_classes()[name], name


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


def test_importing_the_plotting_surface_leaves_the_comms_toolkit_unloaded():
    """``uacpy/__init__`` advertises a lazy-cost design, and the plotters keep
    their compute-side imports inside the functions that use them. A single
    module-scope ``from uacpy.comms.metrics import ...`` in the comms plotter
    pulled the whole toolkit — and scipy.signal behind it — into every
    ``import uacpy.visualization``."""
    result = _run_python(
        "import sys\n"
        "import uacpy.visualization\n"
        "comms = [m for m in sys.modules if m.startswith('uacpy.comms')]\n"
        "assert not comms, f'comms toolkit loaded eagerly: {sorted(comms)}'\n"
        "assert 'scipy.signal' not in sys.modules, (\n"
        "    'scipy.signal loaded eagerly by import uacpy.visualization'\n"
        ")\n"
    )
    _assert_clean_exit(result)


def test_importing_the_plotting_surface_leaves_the_io_layer_unloaded():
    """The plotters label axes in km, and ``km_to_m``/``m_to_km``/``deg_to_rad``
    live in :mod:`uacpy.core.units` precisely so a layer above ``io`` can reach
    them without importing a sibling package for arithmetic. Four plot modules
    spelling those three lines ``from uacpy.io.units import ...`` pulled all 18
    ``uacpy.io`` modules — every reader and writer in the tree — into every
    ``import uacpy.visualization``."""
    result = _run_python(
        "import sys\n"
        "import uacpy.visualization\n"
        "io_modules = [m for m in sys.modules if m.startswith('uacpy.io')]\n"
        "assert not io_modules, (\n"
        "    f'uacpy.io loaded eagerly by import uacpy.visualization: '\n"
        "    f'{len(io_modules)} module(s), {sorted(io_modules)}'\n"
        ")\n"
    )
    _assert_clean_exit(result)


_VISUALIZATION_DIR = Path(uacpy.__file__).resolve().parent / 'visualization'


def _visualization_imports_of_io():
    """``(relative path, lineno, module)`` for every module-scope
    ``uacpy.io`` import under ``uacpy/visualization``."""
    import ast

    found = []
    for path in sorted(_VISUALIZATION_DIR.rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.iter_child_nodes(tree):
            names = []
            if isinstance(node, ast.ImportFrom) and (node.module or '').startswith(
                    'uacpy.io'):
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names
                         if a.name.startswith('uacpy.io')]
            for name in names:
                found.append((str(path.relative_to(_VISUALIZATION_DIR.parent)),
                              node.lineno, name))
    return found


def test_no_visualization_module_imports_the_io_layer_at_module_scope():
    """The sweep behind the subprocess gate above: it names the offending line
    instead of only reporting that something in the package reached ``io``.
    A plotter that genuinely needs a reader defers the import into the function
    that calls it, as the comms plotter does."""
    offenders = _visualization_imports_of_io()
    assert not offenders, (
        "module-scope uacpy.io imports under uacpy/visualization (each one "
        "loads all 18 io modules into import uacpy.visualization): "
        + '; '.join(f'{p}:{n} imports {m}' for p, n, m in offenders))


def test_both_unit_helper_spellings_reach_the_same_functions():
    """Moving the plotters' import does not retire ``uacpy.io.units``: it
    re-exports the three names for the writers and readers, and both spellings
    must keep resolving to the one definition."""
    from uacpy.core.units import deg_to_rad, km_to_m, m_to_km
    from uacpy.io.units import deg_to_rad as io_deg, km_to_m as io_km
    from uacpy.io.units import m_to_km as io_m
    assert (io_deg, io_km, io_m) == (deg_to_rad, km_to_m, m_to_km)
    import uacpy.core.units as core_units
    import uacpy.io.units as io_units
    for name in ('km_to_m', 'm_to_km', 'deg_to_rad'):
        assert getattr(io_units, name) is getattr(core_units, name)


def test_the_comms_plotter_draws_its_theory_overlay():
    """The deferred import has to resolve when the overlay is actually asked
    for — a plotter that only fails at call time is worse than an eager one."""
    result = _run_python(
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "from uacpy.visualization.plots.comms import plot_ber_curve\n"
        "fig, ax = plot_ber_curve([0, 5, 10], [1e-1, 1e-2, 1e-3],\n"
        "                         scheme='bpsk')\n"
        "assert len(ax.lines) == 2, 'theory curve missing'\n"
        "plt.close(fig)\n"
    )
    _assert_clean_exit(result)


_CORE_DIR = Path(uacpy.__file__).resolve().parent / 'core'

#: The word each deferred ``core -> visualization`` import carries above it.
#: The comment is the whole remedy for an inversion that is otherwise
#: invisible at the site: nothing in ``core`` reads as though it depends on
#: the plotting stack until the method runs.
_DEFERRAL_MARKER = 'deferred'


def _core_imports_of_visualization():
    """``(relative path, lineno, enclosing function or None)`` for every
    ``uacpy.visualization`` import under ``uacpy/core``."""
    import ast

    found = []
    for path in sorted(_CORE_DIR.rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))

        def walk(node, enclosing):
            for child in ast.iter_child_nodes(node):
                inner = (child.name
                         if isinstance(child, (ast.FunctionDef,
                                               ast.AsyncFunctionDef))
                         else enclosing)
                walk(child, inner)
                module = (child.module or '') if isinstance(
                    child, ast.ImportFrom) else ''
                if module.startswith('uacpy.visualization'):
                    found.append((str(path.relative_to(_CORE_DIR.parent)),
                                  child.lineno, enclosing))

        walk(tree, None)
    return found


_CORE_VISUALIZATION_IMPORTS = _core_imports_of_visualization()


def test_core_reaches_up_into_visualization():
    """The sweep itself, so an empty collection cannot pass as a green gate.

    If these edges are ever removed the two tests below become vacuous, and
    this one says so rather than staying quiet."""
    assert _CORE_VISUALIZATION_IMPORTS, (
        "no core -> visualization import found; drop these gates and the "
        "docs/DEV.md section 7 paragraph that records the inversion")


@pytest.mark.parametrize(
    'relative_path,lineno,enclosing', _CORE_VISUALIZATION_IMPORTS,
    ids=[f'{p}:{n}' for p, n, _ in _CORE_VISUALIZATION_IMPORTS])
def test_each_core_import_of_visualization_sits_in_a_function_body(
        relative_path, lineno, enclosing):
    """``uacpy/__init__`` eagerly loads ``uacpy.core`` and
    ``uacpy.visualization.plots`` imports ``uacpy.core`` at module scope, so
    one of these hoisted to file scope makes ``import uacpy`` raise
    ``ImportError`` from a partially initialised module."""
    assert enclosing is not None, (
        f"{relative_path}:{lineno} imports uacpy.visualization at module "
        f"scope; import uacpy raises ImportError on that")


@pytest.mark.parametrize(
    'relative_path,lineno,enclosing', _CORE_VISUALIZATION_IMPORTS,
    ids=[f'{p}:{n}' for p, n, _ in _CORE_VISUALIZATION_IMPORTS])
def test_each_core_import_of_visualization_says_why_it_is_deferred(
        relative_path, lineno, enclosing):
    """A reader of ``core`` meets a lone import inside a method with no reason
    given, and the reason is a cycle they cannot see from there."""
    lines = (_CORE_DIR.parent / relative_path).read_text(
        encoding='utf-8').splitlines()
    above = lines[max(0, lineno - 7):lineno - 1]
    comments = [ln.strip() for ln in above if ln.strip().startswith('#')]
    assert any(_DEFERRAL_MARKER in ln.lower() for ln in comments), (
        f"{relative_path}:{lineno} has no comment saying the import is "
        f"deferred to break the cycle; comments above it: {comments}")
