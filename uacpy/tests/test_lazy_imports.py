"""The lazy-import promise of ``import uacpy`` (PEP 562).

``uacpy/__init__`` eagerly loads only the core carriers; the result
types, the model wrappers, plotting, DSP and data subpackages — and with them scipy and
matplotlib — are paid for on first attribute access. Every test here runs
in a fresh subprocess because the promise is about a cold interpreter:
the pytest process itself has long since imported everything.

The resolution test walks ``_LAZY_ATTRS`` / ``_LAZY_SUBMODULES`` from the
live module, so a typo'd table entry — which otherwise explodes only at a
user's first attribute access — fails here instead.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import uacpy

#: Every test here pins a repo convention (sources, docs, packaging), not
#: runtime behaviour: ``-m "not convention"`` deselects the module.
pytestmark = pytest.mark.convention

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
    scipy-adjacent dependency (acoustic_signal.generate in the SPARC
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


def _statically_re_imported_targets(init_path=_INIT_PATH):
    """``{bound name: (module, attribute)}`` for every ``from ... import ...``
    inside the ``if TYPE_CHECKING:`` block of ``init_path``
    (``uacpy/__init__.py`` by default).

    ``from uacpy import models`` yields ``('uacpy', 'models')``, which names
    the module ``uacpy.models``; ``from uacpy.models import Bellhop`` yields
    ``('uacpy.models', 'Bellhop')``, which is the ``_LAZY_ATTRS`` entry
    verbatim. One rule reads both tables' spellings."""
    import ast

    tree = ast.parse(init_path.read_text(encoding='utf-8'))
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
    from uacpy.models._registry import engine_classes

    classes = {cls.__name__: cls for cls in engine_classes().values()}
    wrappers = set(classes)
    reachable = set(uacpy._LAZY_ATTRS) | set(vars(uacpy))
    missing = sorted(wrappers - reachable)
    assert not missing, (
        f"model wrapper(s) {missing} are exported by uacpy.models but absent "
        f"from _LAZY_ATTRS in uacpy/__init__.py, so uacpy.<name> raises "
        f"AttributeError (docs/DEV.md section 3, step 4)")

    for name in sorted(wrappers):
        assert getattr(uacpy, name) is classes[name], name


#: The subpackages whose ``__init__`` resolves its names on first access
#: from an ``_EXPORTS`` table (``{name: defining module}``).
_LAZY_PACKAGES = ('uacpy.core', 'uacpy.models')


@pytest.mark.parametrize('package', _LAZY_PACKAGES)
def test_each_lazy_package_restates_its_table_for_type_checkers(package):
    """The ``if TYPE_CHECKING:`` block of a lazy subpackage imports exactly
    the names its ``_EXPORTS`` table resolves, each from the module the table
    names, and the submodules its ``__all__`` lists: what a checker reads in
    place of ``__getattr__``."""
    import importlib
    module = importlib.import_module(package)
    table = {name: (target, name) for name, target in module._EXPORTS.items()}
    table.update({name: (package, name)
                  for name in getattr(module, '_LISTED_SUBMODULES', ())})
    mirror = _statically_re_imported_targets(Path(module.__file__))
    assert mirror == table, (
        f"{package}: static mirror and lazy table differ; only in the "
        f"mirror: {sorted(set(mirror.items()) - set(table.items()))}, only "
        f"in the table: {sorted(set(table.items()) - set(mirror.items()))}")


@pytest.mark.parametrize('package', _LAZY_PACKAGES)
def test_every_name_a_lazy_package_lists_resolves_and_is_cached(package):
    """Every ``__all__`` name of a lazy subpackage resolves in a cold
    interpreter and is bound on the package afterwards, so its
    ``__getattr__`` runs once per name."""
    result = _run_python(
        "import importlib\n"
        f"package = importlib.import_module({package!r})\n"
        "failures = []\n"
        "for name in package.__all__:\n"
        "    try:\n"
        "        value = getattr(package, name)\n"
        "    except Exception as exc:\n"
        "        failures.append(f'{name}: {type(exc).__name__}: {exc}')\n"
        "        continue\n"
        "    if vars(package).get(name) is not value:\n"
        "        failures.append(f'{name}: resolved but not cached')\n"
        "if failures:\n"
        "    raise SystemExit('\\n'.join(failures))\n"
    )
    _assert_clean_exit(result)


def test_importing_the_models_package_loads_no_engine():
    """``import uacpy.models`` reads the engine registry and nothing else;
    naming one engine loads that engine's module, not the others."""
    result = _run_python(
        "import sys\n"
        "import uacpy.models\n"
        "loaded = sorted(m for m in sys.modules\n"
        "                if m.startswith('uacpy.models.'))\n"
        "assert loaded == ['uacpy.models._registry'], loaded\n"
        "uacpy.models.Scooter\n"
        "assert 'uacpy.models.scooter' in sys.modules\n"
        "assert 'uacpy.models.bellhop' not in sys.modules\n"
    )
    _assert_clean_exit(result)


def test_the_absorption_formulas_are_array_functions_at_uacpy_acoustics():
    """An absorption is asked for at two levels, one computation each:
    ``uacpy.acoustics`` holds the formulas on plain arrays, named
    ``absorption_<model>`` like ``sound_speed_mackenzie`` and returning dB/km,
    and a law's ``table`` returns a carrier whose unit is a value, written on
    those formulas. The formulas are ``uacpy.acoustics`` names, not top-level
    ones; the laws (one object per model, which an Environment holds) are
    top-level. Runs in-process — it inspects the surface, not import
    order."""
    import uacpy.acoustics as acoustics
    from uacpy.core.acoustics import attenuation
    for name in ('absorption_thorp', 'absorption_francois_garrison',
                 'absorption_biological'):
        assert getattr(acoustics, name) is getattr(attenuation, name), name
        assert not hasattr(uacpy, name), name
    assert hasattr(uacpy, 'Thorp') and hasattr(uacpy, 'FrancoisGarrison')
    for name in ('thorp_dB_per_km', 'francois_garrison_dB_per_km',
                 'biological_dB_per_km'):
        assert not hasattr(acoustics, name) and not hasattr(uacpy, name), name


def test_uacpy_plot_is_a_module_re_exporting_every_plotter():
    """docs/guide/plotting.md §1: ``uacpy.plot`` is a module, so both
    ``uacpy.plot.plot_field`` and ``from uacpy.plot import plot_field``
    resolve, each to the object ``uacpy.visualization.plots`` defines, and it
    is the plotters' one public path: none is reachable as ``uacpy.<name>``."""
    result = _run_python(
        "import uacpy\n"
        "import uacpy.visualization.plots as plots\n"
        "from uacpy.plot import plot_field\n"
        "assert plot_field is plots.plot_field\n"
        "assert list(uacpy.plot.__all__) == list(plots.__all__)\n"
        "for name in plots.__all__:\n"
        "    assert getattr(uacpy.plot, name) is getattr(plots, name), name\n"
        "for name in plots.__all__:\n"
        "    assert getattr(uacpy, name, None) is not getattr(plots, name), name\n"
    )
    _assert_clean_exit(result)


def test_import_uacpy_leaves_the_plot_module_unloaded():
    """``uacpy.plot`` loads matplotlib, so ``import uacpy`` must not import
    it; the first ``uacpy.plot`` access does."""
    result = _run_python(
        "import sys, uacpy\n"
        "assert 'uacpy.plot' not in sys.modules\n"
        "uacpy.plot\n"
        "assert 'uacpy.plot' in sys.modules\n"
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
        "uacpy.plot  # imports uacpy.plot -> uacpy.visualization\n"
        "changed = [k for k, v in matplotlib.rcParams.items()\n"
        "           if before.get(k) != v]\n"
        "assert not changed, f'rcParams touched: {changed}'\n"
    )
    _assert_clean_exit(result)


def test_importing_the_plotting_surface_leaves_the_comms_toolkit_unloaded():
    """``uacpy/__init__`` advertises a lazy-cost design, and the plotters keep
    their compute-side imports inside the functions that use them. A single
    module-scope ``from uacpy.comms.<module> import ...`` in the comms plotter
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
    spelling those three lines ``from uacpy.core.units import ...`` pulled all 18
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
                names = ([alias.name for alias in child.names]
                         if isinstance(child, ast.ImportFrom) else [])
                if (module.startswith('uacpy.visualization')
                        or (module == 'uacpy'
                            and 'visualization' in names)):
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
    """``uacpy/__init__`` eagerly loads the core carriers and
    ``uacpy.visualization.plots`` imports them at module scope, so a
    carrier's one of these hoisted to file scope makes ``import uacpy`` raise
    ``ImportError`` from a partially initialised module; a result module's
    one would load the plotting stack with the first result type."""
    assert enclosing is not None, (
        f"{relative_path}:{lineno} imports uacpy.visualization at module "
        f"scope")


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


def test_the_restated_export_lists_stay_in_sync():
    """The plotters have one public path, ``uacpy.plot``, which exposes
    ``uacpy.visualization.plots`` object for object; ``uacpy.visualization``
    holds no plotter (it holds the style and the coastline backdrop). And
    ``uacpy`` restates ``uacpy.core``'s names deliberately, so this pins that
    every core name the top level exports is the core object."""
    import uacpy
    import uacpy.core as core
    import uacpy.plot
    import uacpy.visualization as viz
    import uacpy.visualization.plots as plots
    assert set(uacpy.plot.__all__) == set(plots.__all__)
    drifted = [n for n in plots.__all__
               if getattr(uacpy.plot, n) is not getattr(plots, n)]
    assert drifted == [], f"uacpy.plot exposes a different object: {drifted}"
    third = sorted(set(viz.__all__) & set(plots.__all__))
    assert third == [], f"uacpy.visualization re-exports plotters: {third}"
    leaked = [n for n in plots.__all__ if callable(getattr(plots, n, None))
              and getattr(viz, n, None) is getattr(plots, n)]
    assert leaked == [], f"plotters reachable as uacpy.visualization.<name>: {leaked}"
    import types
    # ``uacpy.metrics`` is the root shim MODULE over ``uacpy.core.metrics``:
    # two module objects with one content, not a drift — compare objects only.
    core_drift = [n for n in core.__all__
                  if n in uacpy.__all__
                  and not isinstance(getattr(core, n), types.ModuleType)
                  and getattr(uacpy, n) is not getattr(core, n)]
    assert core_drift == [], f"uacpy re-exports a different core object: {core_drift}"



def test_uacpy_acoustics_is_an_importable_module_over_core_acoustics():
    """The docs name ``uacpy.acoustics``; ``from uacpy.acoustics import spl``
    must work as well as attribute access, and the facade carries exactly the
    core package's names, each the same object."""
    import uacpy
    import uacpy.core.acoustics as core_acoustics
    from uacpy.acoustics import spl
    assert spl is core_acoustics.spl
    assert sorted(uacpy.acoustics.__all__) == sorted(core_acoustics.__all__)
    drifted = [n for n in core_acoustics.__all__
               if getattr(uacpy.acoustics, n) is not getattr(core_acoustics, n)]
    assert drifted == [], drifted


def test_the_acoustics_docstring_lists_every_subject_module_it_re_exports():
    """``help(uacpy.acoustics)`` describes the namespace: every sub-module a
    public name comes from is one of the listed subjects."""
    import uacpy.core.acoustics as core_acoustics
    sources = {getattr(core_acoustics, n).__module__.rsplit('.', 1)[-1]
               for n in core_acoustics.__all__}
    listed = set(re.findall(r'^\* ``(\w+)``', core_acoustics.__doc__, re.M))
    assert sources <= listed, sorted(sources - listed)


def test_the_root_namespace_offers_no_stdlib_helper():
    """``dir(uacpy)`` is documented as the full index; the stdlib module the
    lazy loader uses is not part of it."""
    import uacpy
    assert 'importlib' not in dir(uacpy)


def test_parallel_declares_its_public_names():
    import uacpy.parallel as parallel
    assert sorted(parallel.__all__) == ['Job', 'ParallelResult', 'run_parallel']


def test_the_child_stack_limit_and_its_opt_out_are_documented():
    """Model binaries run with a raised RLIMIT_STACK (the child alone); the
    opt-out variable ``_stack`` reads is in the user manual."""
    import uacpy._stack as stack
    names = set(re.findall(r"'(UACPY_\w+)'", Path(stack.__file__).read_text()))
    assert names == {'UACPY_NO_STACK_RAISE'}
    manual = (Path(uacpy.__file__).parents[1] / 'DOCUMENTATION.md').read_text()
    assert 'UACPY_NO_STACK_RAISE' in manual and 'RLIMIT_STACK' in manual


def test_every_dotted_uacpy_name_in_the_core_docstrings_resolves():
    """A ``uacpy.a.b`` spelled in a core docstring is a line a user copies;
    each must resolve (``uacpy.plots`` did not — the namespace is
    ``uacpy.plot``)."""
    import importlib
    root = Path(uacpy.__file__).parent
    files = list((root / 'core').rglob('*.py')) + [
        root / name for name in ('metrics.py', 'parallel.py', 'analytic.py',
                                 'acoustics.py', '__init__.py')]
    unresolved = []
    for f in files:
        for m in re.finditer(r'``(uacpy(?:\.\w+)+)', f.read_text()):
            parts = m.group(1).split('.')
            obj = uacpy
            for k, part in enumerate(parts[1:], 1):
                try:
                    obj = getattr(obj, part)
                except AttributeError:
                    try:
                        obj = importlib.import_module('.'.join(parts[:k + 1]))
                    except ModuleNotFoundError:
                        unresolved.append(f"{f.name}: {m.group(1)}")
                        break
    assert unresolved == [], unresolved


class TestPublicReexports:
    """Public namespace contract."""

    def test_sound_speed_profile_at_top_level(self):
        from uacpy import SoundSpeedProfile
        assert SoundSpeedProfile is uacpy.core.environment.SoundSpeedProfile

    def test_environment_helpers_at_core(self):
        from uacpy.core import SoundSpeedProfile, generate_sea_surface
        assert SoundSpeedProfile is uacpy.core.environment.SoundSpeedProfile
        assert generate_sea_surface is uacpy.core.altimetry.generate_sea_surface

    def test_acoustic_signal_is_importable_submodule(self):
        import uacpy.acoustic_signal as sig
        assert sig is uacpy.acoustic_signal

    def test_signal_analysis_classes_reachable(self):
        sig = uacpy.acoustic_signal
        # Estimators/transforms are free functions; FRF is a class. Each
        # estimator is named for the statistic it returns, so the doors are
        # what a call site reaches for.
        for name in ('welch', 'welch',
                     'sound_exposure', 'constant_q',
                     'constant_q',
                     'probabilistic_welch',
                     'probabilistic_welch',
                     'probabilistic_sound_exposure',
                     'probabilistic_constant_q',
                     'probabilistic_constant_q',
                     'FRF', 'fk_transform', 'spectrogram'):
            assert hasattr(sig, name), f"uacpy.acoustic_signal.{name} not reachable"
            assert name in sig.__all__, f"{name} missing from __all__"
        for retired in RETIRED_SPECTRAL_NAMES:
            assert not hasattr(sig, retired), (
                f"{retired} is back: there is one function per statistic, "
                f"each carrying only the parameters it can honour — no "
                f"alias that hides which statistic it is, and no general "
                f"estimator whose arguments have to be policed against one "
                f"another")

    def test_metrics_is_importable_submodule(self):
        import uacpy.metrics as m
        assert m is uacpy.metrics
        assert hasattr(m, 'tl_rmse')
        assert hasattr(m, 'tl_max_error')
        assert hasattr(m, 'tl_bias')

    @pytest.mark.parametrize('name, core', [('materials', 'uacpy.core.materials'),
                                            ('units', 'uacpy.core.units')])
    def test_materials_and_units_are_importable_modules(self, name, core):
        import importlib
        module = importlib.import_module(f'uacpy.{name}')
        assert module is getattr(uacpy, name)
        assert module.__name__ == f'uacpy.{name}'
        source = importlib.import_module(core)
        assert module.__all__ == source.__all__
        assert all(getattr(module, n) is getattr(source, n)
                   for n in source.__all__)


#: Every name the spectral reshape retired. The attribute check below sees
#: only the package namespace; the text sweep sees docstrings, comments,
#: error messages, guides and examples — which is where all of them actually
#: survived a rename, because a ``psd(``-shaped search does not match
#: ``from uacpy.acoustic_signal import ..., psd`` on its own line.
RETIRED_SPECTRAL_NAMES = (
    'psd', 'ppsd', 'sel', 'spectral_estimate',
    'probabilistic_spectral_estimate', 'power_spectral_density',
    'power_spectrum', 'constant_q_psd', 'constant_q_spectrum',
    'constant_q_spectral_density', 'probabilistic_power_spectral_density',
    'probabilistic_power_spectrum', 'probabilistic_constant_q_spectrum',
    'PSDResult', 'PPSDResult', 'CQPSDResult', 'CQPPSDResult', 'SELResult',
    'ConstantQEstimate', 'SPECTRAL_METHODS', 'SPECTRAL_SCALINGS',
)


#: The retired names that are also ordinary short words: ``psd`` is a
#: parameter of ``decidecade_band_levels``, ``sel`` a local in half a dozen
#: readers, ``mode='psd'`` scipy's own. For these, only an API-SHAPED
#: reference counts — a code span, an import, a Sphinx role or a call.
_SHORT_RETIRED = ('psd', 'ppsd', 'sel')


#: Spellings that legitimately contain a retired name: the live functions
#: whose names embed one, and the one place the guide explains the short
#: names are gone.
_RETIREMENT_ALLOWED = (
    'synthesize_noise_from_psd', 'plot_psd', 'plot_ppsd', 'plot_sel',
    'plot_constant_q_psd', 'plot_constant_q_ppsd', 'as_psd', 'integrate_psd',
    'sound_exposure', 'isel',
    # xarray's own selector, which several readers document by name
    "sel(method='nearest')", '.sel(',
    'RETIRED_SPECTRAL_NAMES', '_RETIREMENT_ALLOWED', '_SHORT_RETIRED',
    'no `psd` / `ppsd`',
)


def _api_shaped(name):
    """Patterns that mean "the function ``name``", not a word spelled that way.

    A code span, a docstring literal, an import, a Sphinx role or a call —
    the shapes an API reference takes. A bare ``psd`` in
    ``if np.any(psd < 0)`` is a parameter and not a finding.
    """
    import re
    escaped = re.escape(name)
    return [re.compile(p.format(n=escaped)) for p in (
        r'import\s+[^\n]*\b{n}\b', r':func:`[^`]*{n}`',
        r'(?<![\w.`]){n}\(',
    )]


def test_no_retired_spectral_name_survives_in_text():
    """The rename has to reach prose, not just the namespace.

    Every defect this sweep would have caught lived in a docstring, a comment,
    a runtime error message, a guide or an example — places a ``hasattr``
    check cannot see and a call-shaped grep does not match.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    targets = sorted(
        [p for p in (root / 'uacpy').rglob('*.py')
         if 'third_party' not in str(p) and '/build/' not in str(p)]
        + list((root / 'docs').rglob('*.md'))
        + [root / 'DOCUMENTATION.md'])
    assert len(targets) > 100, f"sweep found only {len(targets)} files"

    long_names = [n for n in RETIRED_SPECTRAL_NAMES if n not in _SHORT_RETIRED]
    short_patterns = {n: _api_shaped(n) for n in _SHORT_RETIRED}
    hits = []
    for path in targets:
        if path.name == Path(__file__).name:
            continue
        for lineno, line in enumerate(path.read_text(encoding='utf-8')
                                      .splitlines(), 1):
            if any(ok in line for ok in _RETIREMENT_ALLOWED):
                continue
            if line.lstrip().startswith(('def ', 'async def ')):
                continue        # a local helper may share an ordinary word
            found = next((n for n in long_names
                          if re.search(rf'(?<![\w.]){re.escape(n)}(?![\w])',
                                       line)), None)
            if found is None:
                found = next((n for n, pats in short_patterns.items()
                              if any(p.search(line) for p in pats)), None)
            if found is not None:
                hits.append(f"{path.relative_to(root)}:{lineno}: {found}"
                            f" -> {line.strip()[:90]}")
    assert not hits, "retired spectral names still in the text:\n  " + \
        "\n  ".join(hits[:25])
