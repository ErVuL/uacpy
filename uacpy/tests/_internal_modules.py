"""The package-internal API modules: the ``_``-prefixed modules another
top-level package of uacpy may import, each with the reason it is shared.

A leading underscore on a module says "not public API". It does not say which
of uacpy's own packages may import it, so without this table a private module
reached from another package is invisible to the private-name gates, which
read only ``_``-prefixed *names* (``from uacpy.data._http import http_get``
imports a public name from a private module). The table is that answer, one
entry per module, and it only shrinks: a new entry is a new cross-package
dependency on a private module, and one no longer imported is deleted.
``test_packaging`` holds every cross-package import of a private module to it;
the names such a module is imported for are covered by its entry, so the
name-level lists do not repeat them.

Not collected by pytest (no ``test_`` prefix).
"""

import ast
from pathlib import Path

#: ``module: reason``. The module is the first ``_``-prefixed component of
#: the import path (``uacpy.core._validate``), whatever is imported from it.
PACKAGE_INTERNAL_MODULES = {
    'uacpy._log':
        "the package's one logger (log_message and the verbose= threshold), "
        "so one setting governs the output of every package",
    'uacpy._stack':
        "the child-only stack-limit and parent-death argv prefixes every "
        "binary launch runs under",
    'uacpy._version':
        "the generated version file the package root reads __version__ from",
    'uacpy.acoustic_signal._results':
        "the result record types (ResultTuple, PlottedResult, POWER_UNITS) "
        "the signal results share, which WenzNoise, the comms results "
        "(JanusReception, BerCurve) and the signal plotters build and read",
    'uacpy.acoustic_signal._synthesis':
        "the spectrum-to-record synthesis that Field's time series and the "
        "models' pulse bands run",
    'uacpy.io._parsers':
        "the raw output-file parsers the engines read; users read the "
        "records the public readers build from the same files",
    'uacpy.core._beamforming':
        "the quadratic-form kernel of Bartlett and MVDR, shared by the array "
        "spectra and matched-field processing",
    'uacpy.core._export':
        "the export protocol engine (Exportable, the attribute encoding, the "
        "optional-extra import) that the signal result tuples, "
        "ParallelResult and the io and data records (ExportRecord) build "
        "on, as the results and carriers do",
    'uacpy.core._finite_difference':
        "the float32 resolution rule for differencing stored wavenumbers, "
        "applied by the modal dispersion estimate",
    'uacpy.core._host':
        "the host's memory and CPU budget the engines size their grids "
        "against",
    'uacpy.core._plotting':
        "the one import that reaches up into uacpy.visualization: every "
        "carrier and result .plot() draws through the public plotter it "
        "names (docs/DEV.md section 7)",
    'uacpy.core._provenance':
        "the one de-duplication rule for DataProvenance records, shared by "
        "the carriers and the fetchers' assembled profiles",
    'uacpy.core._records':
        "the frozen-record base every settings and result record builds on",
    'uacpy.core._repr':
        "the one-line repr formatting (axis, quantity, record and tool "
        "helpers) every public class prints with, one form across packages",
    'uacpy.core._validate':
        "the argument validators every package refuses bad input with, one "
        "wording per rule",
    'uacpy.core._warn_frames':
        "the frame-skip prefix that attributes every warning to the user's "
        "own line",
    'uacpy.data._cache':
        "the offline dataset cache: the basemap renderer reads and writes "
        "its coastline there, with the same atomic write",
    'uacpy.data._http':
        "map-tile download reuses the data layer's HTTP transport (curl, "
        "then urllib, with the same timeout rule)",
    'uacpy.models._launch':
        "the launch-thread policy a run_parallel worker sets for the engines "
        "it runs",
    'uacpy.models._registry':
        "the engine registry the package root builds its lazy model names "
        "from",
    'uacpy.models._workspace':
        "the work-directory manager the parallel runner allocates each job's "
        "directory with",
}


def private_module(module):
    """The first ``_``-prefixed component of ``module`` with the path above
    it (``uacpy.core._validate.x`` -> ``uacpy.core._validate``), or ``None``.
    A dunder component (``__init__``) is not private."""
    parts = module.split('.')
    for i, part in enumerate(parts):
        if part.startswith('_') and not part.startswith('__'):
            return '.'.join(parts[:i + 1])
    return None


def owner(relative):
    """The top-level package of a file path relative to the package root;
    ``'<top>'`` for a root module."""
    return relative.split('/')[0] if '/' in relative else '<top>'


def module_owner(module):
    """The top-level package a dotted ``uacpy.*`` module belongs to: its
    second component (a root module such as ``uacpy._log`` is its own)."""
    parts = module.split('.')
    return parts[1] if len(parts) > 1 else '<top>'


_PACKAGE = Path(__file__).resolve().parent.parent


def _is_module(module, package):
    path = package.joinpath(*module.split('.')[1:])
    return path.with_suffix('.py').is_file() or (path / '__init__.py').is_file()


def private_module_imports(tree, relative, package=_PACKAGE):
    """``(relative, private module)`` for every import in ``tree`` (the file
    at ``relative`` under the package root) that reaches a ``_``-prefixed
    module of ANOTHER top-level package: ``from <private module> import x``,
    ``from <package> import <_private module>`` and ``import <private
    module>``."""
    found = set()
    for node in ast.walk(tree):
        modules = []
        if (isinstance(node, ast.ImportFrom) and node.module
                and node.level == 0 and node.module.startswith('uacpy')):
            for alias in node.names:
                full = f'{node.module}.{alias.name}'
                named_module = (alias.name.startswith('_')
                                and not alias.name.startswith('__')
                                and _is_module(full, package))
                modules.append(full if named_module else node.module)
        elif isinstance(node, ast.Import):
            modules = [a.name for a in node.names
                       if a.name.startswith('uacpy')]
        for module in modules:
            private = private_module(module)
            if private and module_owner(private) != owner(relative):
                found.add((relative, private))
    return found
