"""Every uacpy call a reader can copy out of the documentation binds to the
package's current signature.

No test executes the guides' fenced Python blocks, the README, DOCUMENTATION.md
or most docstring examples, so a rename that a word sweep misses leaves a call
there that raises on first use: a keyword the callee no longer takes, a
positional argument it has moved behind ``*``. This gate reads each such call
and binds it, by name, against the signatures read from the package source.

A call is judged only when its callee is a uacpy name: a function or class
called by its bare name, a chain rooted at ``uacpy`` or at a name imported
from it, or a method whose name only uacpy spells (``channel_taps``,
``compute_modes``). A method name another library shares (``plot``,
``max``) is not attributed to uacpy unless its receiver is a uacpy name.
A name several uacpy callables share is refused only when every one of
them refuses the call.
"""
import ast
import builtins
import re
from collections import defaultdict
from pathlib import Path

import pytest

_PACKAGE = Path(__file__).resolve().parents[1]
_REPO = _PACKAGE.parent

#: Method names other libraries the documentation calls also use: a call on
#: one of them is judged only when its receiver is a uacpy name.
_FOREIGN_METHODS = frozenset({
    'plot', 'set', 'get', 'run', 'update', 'copy', 'close', 'save', 'write', 'read',
    'append', 'extend', 'items', 'keys', 'values', 'format', 'replace', 'split', 'join',
    'mean', 'sum', 'max', 'min', 'fill', 'text', 'legend', 'grid', 'imshow', 'scatter',
    'bar', 'hist', 'loglog', 'semilogx', 'semilogy', 'subplots', 'figure', 'axhline',
    'axvline', 'annotate', 'contour', 'contourf', 'pcolormesh', 'colorbar', 'savefig',
    'tight_layout', 'reshape', 'astype', 'at', 'squeeze', 'ravel', 'conj', 'dot', 'all',
    'any', 'clip', 'round', 'transpose', 'cumsum', 'argmax', 'argmin', 'std', 'var',
    'flatten', 'tolist', 'item', 'pop', 'add', 'remove', 'sort', 'index', 'count',
    'strip', 'startswith', 'endswith', 'lower', 'upper', 'load', 'apply', 'filter',
    'interp', 'window', 'shift', 'spectrum', 'power', 'transform', 'fit', 'to_dict',
    'from_dict', 'invert_yaxis', 'fill_between', 'step', 'errorbar', 'twinx'})

#: Calls the gate cannot bind and why; each entry is (file, line text
#: fragment) -> the reason. Kept short: an entry is a decision, not a skip list.
KNOWN = {}

#: A fence line; its info string is the language, written straight after the
#: backticks or after a space (README writes "``` python").
_FENCE = re.compile(r'^```[ \t]*(\S*)[ \t]*$', re.M)
_PYTHON_FENCES = {'', 'python', 'py', 'pycon'}
_PROMPT = re.compile(r'(?m)^(\s*)(>>>|\.\.\.) ?')


def _signatures():
    sigs = defaultdict(list)

    def add(name, fn, drop_self):
        a = fn.args
        pos = [x.arg for x in a.posonlyargs + a.args]
        if drop_self and pos:
            pos = pos[1:]
        sigs[name].append(dict(pos=pos, kwonly=[x.arg for x in a.kwonlyargs],
                               varargs=a.vararg is not None, varkw=a.kwarg is not None))

    for f in _PACKAGE.rglob('*.py'):
        if {'tests', 'third_party', '__pycache__'} & set(f.relative_to(_PACKAGE).parts):
            continue
        tree = ast.parse(f.read_text(encoding='utf-8'))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                add(node.name, node, False)
            elif isinstance(node, ast.ClassDef):
                has_init = any(isinstance(s, ast.FunctionDef) and s.name == '__init__'
                               for s in node.body)
                if not has_init and any(w in ast.unparse(d) for d in node.decorator_list
                                        for w in ('dataclass', 'carrier')):
                    fields = [s.target.id for s in node.body
                              if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
                              and 'ClassVar' not in ast.unparse(s.annotation)]
                    sigs[node.name].append(dict(pos=fields, kwonly=[], varargs=False,
                                                varkw=False))
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef):
                        if sub.name == '__init__':
                            add(node.name, sub, True)
                        elif not sub.name.startswith('_'):
                            static = any(isinstance(d, ast.Name) and d.id == 'staticmethod'
                                         for d in sub.decorator_list)
                            add(sub.name, sub, not static)
    return sigs


def _fits(sig, n_pos, keywords):
    if n_pos > len(sig['pos']) and not sig['varargs']:
        return False
    every = set(sig['pos']) | set(sig['kwonly'])
    if sig['varkw']:
        # **kwargs swallows any keyword, but a parameter's stem (dynamic_range
        # for dynamic_range_dB) is that parameter misspelt
        return not any(k not in every and any(n.startswith(k + '_') for n in every)
                       for k in keywords)
    allowed = set(sig['pos'][n_pos:]) | set(sig['kwonly'])
    return all(k in allowed for k in keywords)


def _python_blocks(text):
    """(offset, code) of each Python fenced block, fences paired in order so
    a closing fence never opens a block."""
    fences = list(_FENCE.finditer(text))
    for opening, closing in zip(fences[::2], fences[1::2]):
        if opening.group(1).lower() in _PYTHON_FENCES:
            start = opening.end() + 1
            yield start, text[start:closing.start()]


def _units():
    """(path relative to the repo, first line, code) for every unit read."""
    docs = [_REPO / 'README.md', _REPO / 'DOCUMENTATION.md',
            *sorted((_REPO / 'docs').rglob('*.md'))]
    for md in docs:
        if not md.is_file():
            continue
        text = md.read_text(encoding='utf-8')
        for start, code in _python_blocks(text):
            yield md.relative_to(_REPO), text.count('\n', 0, start) + 1, code
    for f in sorted(_PACKAGE.rglob('*.py')):
        if {'tests', 'third_party', '__pycache__'} & set(f.relative_to(_PACKAGE).parts):
            continue
        tree = ast.parse(f.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef,
                                 ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc and '>>>' in doc:
                    lines = [ln for ln in doc.split('\n')
                             if ln.lstrip().startswith(('>>>', '...'))]
                    start = (node.body[0].lineno if not isinstance(node, ast.Module)
                             else 1)
                    yield f.relative_to(_REPO), start, '\n'.join(lines)


def _calls(code):
    code = _PROMPT.sub(r'\1', code)
    code = re.sub(r'=\s*(?=[,)])', '=None', code)     # a `name=` placeholder
    lines = code.split('\n')
    lines = [ln[min(len(ln) - len(ln.lstrip()) for ln in lines if ln.strip()):]
             if ln.strip() else ln for ln in lines]
    i = 0
    while i < len(lines):
        for j in range(len(lines), i, -1):
            try:
                tree = ast.parse('\n'.join(lines[i:j]))
            except SyntaxError:
                continue
            roots = {'uacpy'}
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or '').startswith('uacpy'):
                    roots |= {a.asname or a.name for a in node.names}
                elif isinstance(node, ast.Import):
                    roots |= {a.asname or a.name.split('.')[0] for a in node.names
                              if a.name.startswith('uacpy')}
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    yield i + node.lineno - 1, node, roots, lines[i + node.lineno - 1]
            i = j
            break
        else:
            i += 1


def _root(func):
    while isinstance(func, ast.Attribute):
        func = func.value
    return func.id if isinstance(func, ast.Name) else None


def _judged(func, roots):
    if isinstance(func, ast.Name):
        return func.id not in dir(builtins)
    if isinstance(func, ast.Attribute):
        if _root(func) in roots:
            return True
        name = func.attr
        return (name not in _FOREIGN_METHODS and '_' in name
                and not name.startswith(('set_', 'get_', 'add_', '_')))
    return False


def unbound_calls(units=None):
    """``[(path, line, callee, code line)]`` for every documented uacpy call
    that fits no signature of its name."""
    sigs = _signatures()
    out = []
    for path, start, code in (units if units is not None else _units()):
        for off, call, roots, text in _calls(code):
            func = call.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', None)
            if name not in sigs or not _judged(func, roots):
                continue
            if any(isinstance(a, ast.Starred) for a in call.args):
                continue
            if any(isinstance(a, ast.Constant) and a.value is Ellipsis for a in call.args):
                continue
            keywords = [k.arg for k in call.keywords if k.arg is not None]
            if any(_fits(s, len(call.args), keywords) for s in sigs[name]):
                continue
            if any(str(path) == f and frag in text for (f, frag) in KNOWN):
                continue
            out.append((str(path), start + off, name, text.strip()))
    return out


def test_every_documented_uacpy_call_binds_to_its_current_signature():
    bad = unbound_calls()
    assert not bad, (
        "documented call(s) that no longer bind (a renamed keyword, an argument "
        "moved behind *):\n" + "\n".join(f"  {p}:{n}: {name}: {t}" for p, n, name, t in bad))


@pytest.mark.parametrize('code, callee', [
    ('from uacpy.noise import WenzNoise\nWenzNoise(f, wind_speed_mps=10.0)\n', 'WenzNoise'),
    ('import uacpy\nuacpy.generate_sea_surface(1000.0, 10.0, 64, 3)\n', 'generate_sea_surface'),
    ('rc = arr.channel_taps(2000.0, carrier=3000.0)\n', 'channel_taps'),
])
def test_the_gate_refuses_a_stale_call(code, callee):
    """A renamed keyword, one positional too many and a uacpy-only method
    name: each is the case a word sweep misses and each must be reported."""
    bad = unbound_calls([('docs/probe.md', 1, code)])
    assert [b[2] for b in bad] == [callee]


@pytest.mark.parametrize('code', [
    'import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nax.plot(x, y, lw=2)\n',
    'best = (np.abs(p) ** 2).max(axis=0)\n',
    'from uacpy.noise import WenzNoise\nWenzNoise(f, wind_speed_kn=10.0)\n',
])
def test_the_gate_leaves_a_binding_or_foreign_call_alone(code):
    assert unbound_calls([('docs/probe.md', 1, code)]) == []


@pytest.mark.parametrize('text, blocks', [
    ('``` python\nf(1)\n```\n', ['f(1)\n']),
    ('```python\nf(1)\n```\n', ['f(1)\n']),
    ('```\nf(1)\n```\n', ['f(1)\n']),
    ('``` bash\nls\n```\n', []),
    ('```bash\nls\n```\nprose\n```python\nf(1)\n```\n', ['f(1)\n']),
])
def test_a_python_fence_is_read_with_or_without_a_space(text, blocks):
    """README writes "``` python"; a closing fence never opens a block, so
    the prose after a bash block is not read as code."""
    assert [code for _, code in _python_blocks(text)] == blocks


def test_the_readme_python_blocks_are_read():
    readme = (_REPO / 'README.md').read_text(encoding='utf-8')
    fences = len(re.findall(r'(?m)^```[ \t]*python[ \t]*$', readme))
    assert fences >= 2
    assert sum(str(path) == 'README.md' for path, _, _ in _units()) == fences
