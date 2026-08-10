"""
Gate tests for the ``docs/`` tree.

The two checkers under ``docs/`` are the single source of truth; this module
imports and runs them so a developer gets the same verdict from ``pytest`` as
from the command line, and so CI enforces them without extra workflow steps.

Both checks are pure-Python and take about a second — no binaries, no network,
no ``slow`` marker. Figure regeneration is deliberately *not* tested here: it
needs the native binaries and runs for several minutes, so it stays a manual
step (``python docs/generate_model_figures.py``).

What these catch that the rest of the suite does not: a renamed page leaving a
dead cross-reference, and the wreckage of an interrupted or twice-applied edit
— duplicated blocks, unbalanced fences, code samples that fail to parse.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

import uacpy

DOCS_DIR = Path(uacpy.__file__).resolve().parent.parent / "docs"

pytestmark = pytest.mark.skipif(
    not DOCS_DIR.is_dir(),
    reason="docs/ is not present (source checkout only)",
)


def _load(name: str) -> ModuleType:
    """Import a checker from ``docs/`` by path; it is not an installed module."""
    path = DOCS_DIR / f"{name}.py"
    if not path.is_file():
        pytest.skip(f"{path} is missing")
    spec = importlib.util.spec_from_file_location(f"_docs_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_broken_relative_links() -> None:
    """Every relative link and image target in the docs resolves on disk."""
    checker = _load("check_links")

    broken = [
        f"{path.relative_to(checker.REPO_ROOT)}:{line}: -> {target}"
        for path in checker.markdown_files(DOCS_DIR)
        for line, target in checker.broken_links(path)
    ]

    assert not broken, "broken relative link(s):\n" + "\n".join(broken)


def test_documentation_is_structurally_sound() -> None:
    """No unbalanced fences, duplicated blocks, dead figures or bad tables."""
    checker = _load("check_structure")

    problems = [
        f"{path.relative_to(checker.REPO_ROOT)}: {kind}: {detail}"
        for path in checker.markdown_files(DOCS_DIR)
        for kind, detail in checker.check_page(path)
    ]

    assert not problems, "structural problem(s):\n" + "\n".join(problems)


def test_every_documented_code_sample_parses() -> None:
    """Python samples and REPL transcripts are syntactically valid.

    Separated from the structural test so a broken sample names itself in the
    failure rather than hiding among unrelated formatting problems.
    """
    checker = _load("check_structure")

    broken = [
        f"{path.relative_to(checker.REPO_ROOT)}: {kind}: {detail}"
        for path in checker.markdown_files(DOCS_DIR)
        for kind, detail in checker.check_page(path)
        if kind.startswith("SYNTAX_")
    ]

    assert not broken, "unparseable code sample(s):\n" + "\n".join(broken)
