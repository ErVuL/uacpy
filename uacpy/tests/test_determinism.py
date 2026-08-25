"""Gates on the determinism claims uacpy makes, and on the draws behind them.

Three separate things go wrong when a result is not reproducible, and only one
of them is a bug:

* **A draw with no seed behind it.** ``np.random.randn`` and friends read the
  legacy process-global ``RandomState``, so a script that calls one has no
  ``rng`` a caller could pin, and no seed the script itself can record. That is
  a defect in the code, and the source-scanning gate below fails on it.
* **A seed the caller must pass and is not told about.** ``uacpy.comms``'s
  drawing entry points all take ``rng`` and all default to an *unseeded*
  ``default_rng()``. That is the right default for a Monte-Carlo sweep, so the
  fix is documentation, not a signature change — hence the docstring and page
  gates here.
* **A set turned into an ordered output.** Python's set iteration order follows
  the hash seed, so ``list(some_set)`` is a run-to-run difference with no cause
  the code exposes and no seed the caller can pin. The second source-scanning
  gate below fails on it.
* **Float arithmetic that is not associative.** Backend choice, GPU reduction
  order and BLAS thread count all move the last bits. Nothing can be gated into
  agreement, so what is gated is that the pages *say so*.

The prose gates read normalised text (whitespace collapsed) so re-wrapping a
paragraph cannot fire them; they check that a measured statement is present and
that a measured-false one is absent, not how either is worded.

The page gates that belong to a *specific* document — what the Bellhop page
says about backends, what the arrays page says about BLAS threads — live in
``test_documentation.py`` beside the rest of that tree's gates. What is here is
the seed story itself: the source scan, the comms docstrings, and the pages
that introduce the link call.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

import uacpy
from uacpy.comms import awgn, ber_sweep, fading_taps, simulate_link


PKG_DIR = Path(uacpy.__file__).resolve().parent


REPO_ROOT = PKG_DIR.parent


DOCS_DIR = REPO_ROOT / "docs"


_needs_docs = pytest.mark.skipif(
    not DOCS_DIR.is_dir(),
    reason="the docs tree is not present in an installed layout")


def _normalised(path: Path) -> str:
    """One-line form of a markdown page, so a re-wrap cannot move a phrase."""
    return " ".join(path.read_text(encoding="utf-8").split())


# ``numpy.random`` names that construct or type a *generator*: each one hands
# the caller an object whose stream is theirs to seed. Everything else on that
# module is a draw off the process-global ``RandomState``.
_GENERATOR_NAMES = frozenset({
    "default_rng", "Generator", "BitGenerator", "SeedSequence",
    "PCG64", "PCG64DXSM", "Philox", "SFC64", "MT19937",
})


# ``uacpy/tests`` seeds the global generator deliberately (``conftest.py``) and
# draws off it in a handful of fixtures, which is fine: a test is allowed to be
# a caller. ``uacpy/third_party`` is vendored upstream code — ``bellhopcuda``'s
# ``find_failing_ray.py`` is a developer bisection tool, not a shipped module —
# and rewriting it would fork the vendored tree.
_EXCLUDED_PARTS = ("tests", "third_party")


def _scanned_sources():
    """Every ``.py`` file this gate holds: the shipped package minus the
    excluded subtrees, plus the figure scripts that draw the docs' figures."""
    for path in sorted(PKG_DIR.rglob("*.py")):
        if not any(part in _EXCLUDED_PARTS for part in path.parts):
            yield path
    if DOCS_DIR.is_dir():
        yield from sorted((DOCS_DIR / "figure_scripts").glob("*.py"))


def _numpy_aliases(tree: ast.AST) -> set:
    """Names this module binds to the ``numpy`` module itself."""
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "numpy":
                    aliases.add(alias.asname or "numpy")
    return aliases


def _attribute_path(node: ast.AST):
    """``a.b.c`` as ``['a', 'b', 'c']``, or ``None`` for anything else."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return parts[::-1]


def _global_rng_uses(path: Path):
    """``(lineno, what)`` for every legacy global-generator use in one file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases = _numpy_aliases(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "random" or alias.name.startswith("random."):
                    yield node.lineno, f"import {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "random" or module.startswith("random."):
                yield node.lineno, f"from {module} import ..."
            elif module in ("numpy.random", "numpy.random.mtrand"):
                for alias in node.names:
                    if alias.name not in _GENERATOR_NAMES:
                        yield node.lineno, f"from {module} import {alias.name}"
        elif isinstance(node, ast.Attribute):
            parts = _attribute_path(node)
            if (parts is not None and len(parts) == 3
                    and parts[0] in aliases and parts[1] == "random"
                    and parts[2] not in _GENERATOR_NAMES):
                yield node.lineno, ".".join(parts)


def test_no_shipped_module_draws_from_numpys_legacy_global_generator() -> None:
    """No shipped module, example or figure script draws off the legacy
    process-global generator.

    ``np.random.randn(n)`` reads the global ``RandomState``: the caller cannot
    pin it and the script cannot record what it drew, so the output is
    different every run for no reason the code exposes. The replacement is a
    ``Generator`` the code owns — ``rng = np.random.default_rng(seed)`` — or an
    ``rng`` parameter the caller supplies.

    ``uacpy/tests`` and ``uacpy/third_party`` are excluded; see
    ``_EXCLUDED_PARTS``. The examples are deliberately *not*: their integration
    test runs each script in a subprocess, where ``conftest.py``'s autouse seed
    fixture does not reach, and it asserts only that the process exited 0 and
    wrote a PNG — it never compares two runs of the same script. Nothing else
    in the suite can see an unseeded draw in an example.
    """
    problems = []
    scanned = 0
    for path in _scanned_sources():
        scanned += 1
        for lineno, what in _global_rng_uses(path):
            problems.append(
                f"{path.relative_to(REPO_ROOT)}:{lineno}: {what}")

    # Silence must mean "every file agreed", never "the walk found no files".
    # 215 files were in scope when this was written: 195 shipped modules and
    # examples, plus the 20 figure scripts.
    assert scanned > 180, (
        f"only {scanned} source files were scanned — the package layout has "
        f"changed and this gate needs updating"
    )
    assert not problems, (
        "legacy global-RNG use, which no caller can seed:\n"
        + "\n".join(problems)
        + "\n\nUse a Generator the code owns (np.random.default_rng(seed)) "
          "or take an rng parameter."
    )


# ``set`` and ``frozenset`` are the two calls that build one from anything.
_SET_CONSTRUCTORS = frozenset({"set", "frozenset"})


# The calls that freeze an iterable's order into a value someone downstream can
# index, print or compare. ``sorted`` is deliberately absent: it *is* the fix.
_ORDERING_CALLS = frozenset({"list", "tuple"})


# Set methods returning another set, so an ordering call applied to the result
# inherits the receiver's hash-dependent order. ``keys`` is not here: a dict
# view iterates in insertion order, which does not move with the hash seed.
_SET_RETURNING_METHODS = frozenset({
    "union", "intersection", "difference", "symmetric_difference", "copy",
})


# The set operators, all of which return a set when either side is one.
_SET_OPERATORS = (ast.BitOr, ast.BitAnd, ast.Sub, ast.BitXor)


def _is_set_expression(node: ast.AST, set_names: set) -> bool:
    """Whether this expression evaluates to a set, as far as a syntax walk can
    tell: a literal, a comprehension, a ``set()`` call, a name bound to one of
    those, set algebra, or a set method that returns a set."""
    if isinstance(node, (ast.Set, ast.SetComp)):
        return True
    if isinstance(node, ast.Name):
        return node.id in set_names
    if isinstance(node, ast.BinOp) and isinstance(node.op, _SET_OPERATORS):
        return (_is_set_expression(node.left, set_names)
                or _is_set_expression(node.right, set_names))
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            return node.func.id in _SET_CONSTRUCTORS
        if isinstance(node.func, ast.Attribute):
            return (node.func.attr in _SET_RETURNING_METHODS
                    and _is_set_expression(node.func.value, set_names))
    return False


def _set_bound_names(tree: ast.AST) -> set:
    """Every name this module assigns a set to, anywhere in the file.

    Run to a fixed point because ``a = {1}`` may be read by ``b = a | c`` that
    the walk reaches first; two rounds settle any chain a real module has, and
    the loop stops as soon as a round adds nothing.
    """
    names: set = set()
    while True:
        before = len(names)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets = [node.target]
            else:
                continue
            if not _is_set_expression(node.value, names):
                continue
            names.update(t.id for t in targets if isinstance(t, ast.Name))
        if len(names) == before:
            return names


def _unsorted_set_orderings(path: Path):
    """``(lineno, what)`` for every set frozen into an order in one file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    set_names = _set_bound_names(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if not _is_set_expression(node.args[0], set_names):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in _ORDERING_CALLS:
            yield node.lineno, f"{node.func.id}(<set>)"
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "join":
            yield node.lineno, "join(<set>)"


def test_no_shipped_module_freezes_a_set_into_an_order_without_sorting() -> None:
    """No shipped module, example or figure script turns a set into a list,
    tuple or joined string without ``sorted()``.

    ``list(names)`` on a set hands the caller whatever order this process's
    hash seed produced. Two runs of the same script then disagree in a way no
    seed the caller can set will fix, and the difference reaches anything that
    prints, writes or compares the result. ``sorted(names)`` costs nothing at
    these sizes and removes the dependence entirely.

    The gate fires at the *conversion*, not on every set: a set that stays a
    set is order-free by construction, and a function returning one leaves the
    ordering decision — and this gate's verdict — with its caller. The three
    places the package iterates a set directly are covered by the same
    argument: ``_oases_option_chars`` builds one for character membership, and
    ``_apply_spec`` and ``_close_figures_since`` loop over one for a
    per-element side effect, none of which is an ordered output.

    Scope matches the global-RNG gate above: the shipped package and examples
    minus ``_EXCLUDED_PARTS``, plus the figure scripts.
    """
    problems = []
    scanned = 0
    for path in _scanned_sources():
        scanned += 1
        for lineno, what in _unsorted_set_orderings(path):
            problems.append(
                f"{path.relative_to(REPO_ROOT)}:{lineno}: {what}")

    # Silence must mean "every file agreed", never "the walk found no files".
    assert scanned > 180, (
        f"only {scanned} source files were scanned — the package layout has "
        f"changed and this gate needs updating"
    )
    assert not problems, (
        "set(s) frozen into a hash-seed-dependent order:\n"
        + "\n".join(problems)
        + "\n\nWrap the set in sorted(), or keep it a set."
    )


def test_the_set_ordering_scan_flags_each_way_a_set_reaches_an_order(tmp_path):
    """The scan above finds ``list``, ``tuple`` and ``join`` over a set — by
    literal, by comprehension, by name and by set algebra — and passes
    ``sorted()`` and a bare set.

    Without this, a scan that had quietly stopped recognising set expressions
    would report the same clean sweep as one that works.
    """
    source = tmp_path / "sample.py"
    source.write_text(
        "seen = {'b', 'a'}\n"
        "other = {'c'}\n"
        "def by_name():\n"
        "    return list(seen)\n"
        "def by_comprehension():\n"
        "    return tuple({c for c in 'abc'})\n"
        "def by_algebra():\n"
        "    return list(seen | other)\n"
        "def by_join():\n"
        "    return ', '.join(set('ab'))\n"
        "def by_method():\n"
        "    return list(seen.difference(other))\n"
        "def sorted_is_fine():\n"
        "    return list(sorted(seen))\n"
        "def a_set_is_fine():\n"
        "    return seen | other\n",
        encoding="utf-8")

    flagged = {what for _, what in _unsorted_set_orderings(source)}
    assert flagged == {"list(<set>)", "tuple(<set>)", "join(<set>)"}
    assert len(list(_unsorted_set_orderings(source))) == 5


_DRAWING_COMMS_CALLABLES = (awgn, fading_taps, simulate_link, ber_sweep)


_PARAM_ENTRY = re.compile(r"^(\*{0,2}[A-Za-z_]\w*)\s*:\s*(.*)$")


def _parameters_entries(doc: str) -> dict:
    """Parameter name -> description text, for one numpydoc docstring."""
    lines = doc.expandtabs().splitlines()
    start = next((i + 2 for i, line in enumerate(lines[:-1])
                  if line.strip() == "Parameters"
                  and set(lines[i + 1].strip()) == {"-"}), None)
    if start is None:
        return {}
    end = len(lines)
    for i in range(start, len(lines) - 1):
        title, rule = lines[i].strip(), lines[i + 1].strip()
        if title and set(rule) == {"-"} and len(rule) >= len(title):
            end = i
            break
    body = lines[start:end]
    indents = [len(l) - len(l.lstrip()) for l in body if l.strip()]
    if not indents:
        return {}
    base = min(indents)

    entries, name, desc = {}, None, []
    for line in body:
        stripped = line.strip()
        head = (_PARAM_ENTRY.match(stripped)
                if stripped and len(line) - len(line.lstrip()) == base
                else None)
        if head:
            if name:
                entries[name] = " ".join(desc)
            name, desc = head.group(1), []
            continue
        desc.append(stripped)
    if name:
        entries[name] = " ".join(desc)
    return entries


def test_every_comms_call_that_draws_documents_its_rng_parameter() -> None:
    """``awgn``, ``fading_taps``, ``simulate_link`` and ``ber_sweep`` each give
    ``rng`` a numpydoc ``Parameters`` entry that points the reader at seeding.

    All four default ``rng`` to an unseeded ``np.random.default_rng()``, so
    calling them without one is the non-reproducible form and the signature
    alone does not say so. This is the only place the caller can be told.
    """
    problems = []
    for fn in _DRAWING_COMMS_CALLABLES:
        entries = _parameters_entries(fn.__doc__ or "")
        if not entries:
            problems.append(
                f"{fn.__module__}.{fn.__name__}: no Parameters section")
            continue
        if "rng" not in entries:
            problems.append(
                f"{fn.__module__}.{fn.__name__}: Parameters documents "
                f"{sorted(entries)} but not 'rng'")
            continue
        if "seed" not in entries["rng"].lower():
            problems.append(
                f"{fn.__module__}.{fn.__name__}: the rng entry never mentions "
                f"seeding: {entries['rng']!r}")
    assert not problems, (
        "comms draw(s) whose rng is undocumented:\n" + "\n".join(problems))


@_needs_docs
def test_the_comms_pages_seed_the_link_call_they_introduce() -> None:
    """The first ``simulate_link`` call on each comms page passes an ``rng``.

    Both pages lead with that call, so it is the form a reader copies. Left
    unseeded it prints a different BER every run — the one number the snippet
    exists to show.
    """
    problems = []
    for path in (REPO_ROOT / "DOCUMENTATION.md",
                 DOCS_DIR / "guide" / "comms.md"):
        text = _normalised(path)
        first = re.search(r"simulate_link\([^)]*\)", text)
        assert first is not None, f"{path.name}: no simulate_link call to check"
        if "rng=" not in first.group(0):
            problems.append(f"{path.name}: {first.group(0)}")
        if "default_rng(" not in text:
            problems.append(
                f"{path.name}: never shows a seeded default_rng() to pass")
    assert not problems, (
        "unseeded introductory simulate_link call(s):\n" + "\n".join(problems))
