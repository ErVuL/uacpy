"""The free names of a function, and what each one resolves to, compared
across a move.

A function moved into another module keeps its body but not its globals:
every name it reads without binding it (a *free* name) resolves in the
module it sits in afterwards. A same-named constant, helper or import there
takes the place of the one it read before, with no error: ``_BASE_NAME`` is
``'bounce_run'`` in ``models/bounce/_model.py`` and ``'model'`` in
``models/scooter/_model.py``, so ``Bounce._write_input`` moved next to Scooter's
code would write ``model.env``. This module lists a function's free names,
resolves each one to the module-level binding it reads, and reports every
name whose binding differs between two sites.

Command line, run from the repository root (the directory holding
``uacpy/``)::

    python -m uacpy.tests._free_names OLD.py:QUALNAME NEW.py:QUALNAME

prints one line per differing name and exits 1, or exits 0 when every free
name resolves alike. ``QUALNAME`` is dotted for a method
(``Bounce._write_input``). Run it on the function before and after a move,
with the old tree's file kept aside for the first argument.

What a binding is:

* a ``def`` or ``class`` at module level: its kind and a hash of its AST, so
  two same-named helpers with different bodies differ;
* an assignment at module level: the source text of the value;
* an import: followed into the imported module when that module is in the
  tree (``uacpy.*``), until it reaches a ``def``, ``class`` or assignment, so
  ``from uacpy.models.base import X`` and the ``X = …`` it names are one
  binding; otherwise the imported module and name;
* a builtin, when nothing at module level binds the name;
* unbound, otherwise.

A name bound in more than one place at module level (under an ``if`` or a
``try``) resolves to the tuple of its bindings, in source order.

Not collected by pytest (not named ``test_*``); ``test_free_names.py``
exercises it.
"""

import ast
import builtins
import hashlib
import sys
import symtable
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

#: The directory holding the ``uacpy`` package.
_ROOT = Path(__file__).resolve().parents[2]


def free_names(source: str, qualname: str) -> Dict[str, str]:
    """The names the function ``qualname`` in ``source`` reads without
    binding them, in its body and every scope nested in it (a lambda, a
    comprehension, an inner ``def``), each mapped to ``'global'`` (read from
    the module) or ``'enclosing'`` (read from an enclosing function)."""
    table = symtable.symtable(source, '<source>', 'exec')
    for part in qualname.split('.'):
        matches = [child for child in table.get_children()
                   if child.get_name() == part
                   and child.get_type() in ('function', 'class')]
        if not matches:
            raise KeyError(f"{qualname!r}: no scope named {part!r}")
        table = matches[-1]
    if table.get_type() != 'function':
        raise KeyError(f"{qualname!r} is not a function")
    found: Dict[str, str] = {}

    def collect(scope, depth):
        for symbol in scope.get_symbols():
            name = symbol.get_name()
            if symbol.is_global() and symbol.is_referenced():
                found.setdefault(name, 'global')
            elif (symbol.is_free() and depth == 0
                  and symbol.is_referenced()):
                found.setdefault(name, 'enclosing')
        for child in scope.get_children():
            # A nested function's own free names are bound in this one or
            # read from the module; the second kind shows up as global.
            collect(child, depth + 1)

    collect(table, 0)
    return found


def _module_name(path: Path) -> Optional[str]:
    """The dotted module name of ``path`` under the tree, or ``None``."""
    try:
        rel = path.resolve().relative_to(_ROOT)
    except ValueError:
        return None
    parts = list(rel.with_suffix('').parts)
    if parts[-1] == '__init__':
        parts.pop()
    return '.'.join(parts)


def _module_source(name: str) -> Optional[str]:
    """The source of the tree module ``name``, or ``None`` outside it."""
    if not (name == 'uacpy' or name.startswith('uacpy.')):
        return None
    base = _ROOT.joinpath(*name.split('.'))
    for path in (base.with_suffix('.py'), base / '__init__.py'):
        if path.is_file():
            return path.read_text(encoding='utf-8')
    return None


def _absolute(module: Optional[str], level: int, package: str) -> str:
    """The absolute module name of a ``from`` import."""
    if level == 0:
        return module or ''
    base = package.split('.')
    if level > 1:
        base = base[:-(level - 1)]
    return '.'.join(base + ([module] if module else []))


def _module_level(tree: ast.Module):
    """Every statement run at module level: the body, and the bodies of the
    ``if``, ``try``, ``with`` and loop statements in it, never a ``def`` or
    ``class`` body."""
    stack = list(reversed(tree.body))
    while stack:
        node = stack.pop()
        yield node
        nested = []
        for field in ('body', 'orelse', 'finalbody'):
            nested.extend(getattr(node, field, []) or [])
        for handler in getattr(node, 'handlers', []) or []:
            nested.extend(handler.body)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            continue
        stack.extend(reversed(nested))


def _digest(node: ast.AST) -> str:
    return hashlib.sha256(ast.dump(node).encode()).hexdigest()[:12]


def resolve(name: str, source: str, module: str,
            module_source: Callable[[str], Optional[str]] = _module_source,
            _seen=None) -> Tuple[str, ...]:
    """The binding(s) ``name`` resolves to at the top level of ``source``,
    the source of module ``module`` (see the module docstring), following
    imports through ``module_source(dotted_name) -> source or None``."""
    seen = set() if _seen is None else _seen
    if (module, name) in seen:
        return (f'cycle {module}.{name}',)
    seen.add((module, name))
    package = module if _is_package(module) else module.rpartition('.')[0]
    found = []
    for node in _module_level(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)) and node.name == name:
            kind = 'class' if isinstance(node, ast.ClassDef) else 'def'
            found.append(f'{kind} {_digest(node)}')
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target])
            names = {n.id for t in targets for n in ast.walk(t)
                     if isinstance(n, ast.Name)}
            if name in names and node.value is not None:
                found.append(f'= {ast.unparse(node.value)}')
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.partition('.')[0]
                if bound == name:
                    found.append(f'module {alias.name if alias.asname else bound}')
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if (alias.asname or alias.name) != name:
                    continue
                origin = _absolute(node.module, node.level, package)
                target = module_source(origin)
                if target is None:
                    found.append(f'from {origin} import {alias.name}')
                else:
                    found.extend(resolve(alias.name, target, origin,
                                         module_source, seen))
    if not found:
        if hasattr(builtins, name):
            return (f'builtin {name}',)
        return ('unbound',)
    return tuple(found)


def _is_package(module: str) -> bool:
    """Whether the tree module ``module`` is a package (an ``__init__``)."""
    if not (module == 'uacpy' or module.startswith('uacpy.')):
        return False
    return _ROOT.joinpath(*module.split('.'), '__init__.py').is_file()


def diff_bindings(old_source: str, old_module: str, old_qualname: str,
                  new_source: str, new_module: str, new_qualname: str,
                  module_source=_module_source) -> Dict[str, Tuple]:
    """``{name: (old binding, new binding)}`` for every free name of either
    function whose binding differs between the two sites (a name only one
    of them reads counts as unbound in the other)."""
    old = free_names(old_source, old_qualname)
    new = free_names(new_source, new_qualname)
    differ = {}
    for name in sorted(set(old) | set(new)):
        before = (resolve(name, old_source, old_module, module_source)
                  if name in old else ('not read',))
        after = (resolve(name, new_source, new_module, module_source)
                 if name in new else ('not read',))
        if old.get(name) == 'enclosing' or new.get(name) == 'enclosing':
            before = (old.get(name, 'not read'),)
            after = (new.get(name, 'not read'),)
        if before != after:
            differ[name] = (before, after)
    return differ


def _site(arg: str):
    path, _, qualname = arg.rpartition(':')
    path = Path(path)
    module = _module_name(path) or path.stem
    return path.read_text(encoding='utf-8'), module, qualname


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__.split('Command line', 1)[1].split('What a binding',
                                                          1)[0])
        return 2
    differ = diff_bindings(*_site(argv[0]), *_site(argv[1]))
    for name, (before, after) in differ.items():
        print(f"{name}: {' | '.join(before)}  ->  {' | '.join(after)}")
    return 1 if differ else 0


if __name__ == '__main__':
    sys.exit(main())
