#!/usr/bin/env python3
"""Verify the documentation's structural integrity.

Where ``check_links.py`` asks whether a target resolves, this asks whether a
page is well-formed: fences balanced, no block applied twice, tables aligned,
code samples parseable. It targets the damage signature of an edit that was
interrupted or applied twice — the failure a link check and a figure build both
pass straight through.

Checks
------
* unbalanced ``` fences
* headings, paragraphs or sentences repeated verbatim within a page
* image targets that do not exist on disk
* header/separator column mismatch in tables
* heading level jumps (``##`` straight to ``####``), a removed-section tell
* placeholder markers (TODO / FIXME / TBD / XXX)
* every ``python`` block parses, REPL (``>>>``) transcripts included

Signature displays (``f(a, *, b)``) and error transcripts (``SomeError: ...``)
are not executable and are skipped by design.

Usage
-----
    python docs/check_structure.py            # whole docs tree
    python docs/check_structure.py docs/guide # a subtree

Exit status is non-zero if any problem is found. CI does not call this script:
``uacpy/tests/test_documentation.py`` imports it and runs the same check over
``docs/``, so ``pytest`` and the command line give the same verdict.
"""

from __future__ import annotations

import ast
import collections
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_HEADING = re.compile(r"^(#{1,6}) \S")
_TABLE_SEP = re.compile(r"^\s*\|[-: |]+\|\s*$")
_IMAGE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
_CODE_BLOCK = re.compile(r"```(?:python|py)\n(.*?)```", re.S)
_PLACEHOLDER = re.compile(r"\bTODO\b|\bFIXME\b|\bTBD\b|XXX|<placeholder")
_SIGNATURE = re.compile(r"^\s*\w+\([^)]*\*")
_TRACEBACK = re.compile(r"^\w*Error:")
_SKIP_DIRS = {".git", "uacpy_venv", "node_modules", "__pycache__",
              "third_party", "data_cache", ".pytest_cache",
              # Working notes, not published pages: plans and specs carry
              # deliberately partial snippets and repeated fixtures.
              "superpowers", "other"}

# A parse error a signature illustration produces but real code cannot: a bare
# `*` keyword-only marker written inside what looks like a call.
_SIGNATURE_ERRORS = ("Invalid star expression",
                     "named arguments must follow bare *",
                     "iterable argument unpacking follows keyword argument")


def markdown_files(root: Path):
    for path in sorted(root.rglob("*.md")):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        yield path


def _repl_source(lines: list[str]) -> str:
    """Extract the input lines from a ``>>>`` transcript, dropping output."""
    out = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(">>> "):
            out.append(stripped[4:])
        elif stripped.startswith("... "):
            out.append(stripped[4:])
    return "\n".join(out)


def check_page(path: Path) -> list[tuple[str, str]]:
    """Return ``(kind, detail)`` for every structural problem on one page."""
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    found: list[tuple[str, str]] = []

    fences = [i for i, ln in enumerate(lines, 1) if ln.lstrip().startswith("```")]
    if len(fences) % 2:
        found.append(("UNBALANCED_FENCE",
                      f"{len(fences)} fences, last at line {fences[-1]}"))

    heads = collections.Counter(ln.strip() for ln in lines if _HEADING.match(ln))
    for head, n in heads.items():
        if n > 1:
            found.append(("DUP_HEADING", f"{n}x {head[:70]!r}"))

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text)]

    prose = collections.Counter(
        p for p in paragraphs
        if len(p) >= 80 and not p.startswith(("|", "```", "#", "-", "*", ">"))
    )
    for para, n in prose.items():
        if n > 1:
            found.append(("DUP_PARAGRAPH", f"{n}x {para[:70]!r}..."))

    for para in paragraphs:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para)
                     if len(s.strip()) > 45]
        for sentence, n in collections.Counter(sentences).items():
            if n > 1:
                found.append(("DUP_SENTENCE", f"{n}x {sentence[:70]!r}..."))

    for match in _IMAGE.finditer(text):
        if not (path.parent / match.group(1)).resolve().exists():
            found.append(("DEAD_FIGURE", match.group(1)))

    in_fence = False
    for i, ln in enumerate(lines, 1):
        if ln.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence and _PLACEHOLDER.search(ln):
            found.append(("PLACEHOLDER", f"line {i}: {ln.strip()[:60]!r}"))

    for i, ln in enumerate(lines):
        if _TABLE_SEP.match(ln) and i and lines[i - 1].strip().startswith("|"):
            if ln.count("|") != lines[i - 1].count("|"):
                found.append(("TABLE_MISMATCH",
                              f"line {i+1}: header {lines[i-1].count('|')} "
                              f"vs separator {ln.count('|')} pipes"))

    # Fenced blocks hold `#` comments, so track fence state before reading
    # heading levels or every code comment reads as an h1.
    prev = 0
    in_fence = False
    for i, ln in enumerate(lines, 1):
        if ln.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _HEADING.match(ln)
        if match:
            level = len(match.group(1))
            if prev and level > prev + 1:
                found.append(("HEADING_JUMP", f"line {i}: h{prev} -> h{level}"))
            prev = level

    for match in _CODE_BLOCK.finditer(text):
        code = match.group(1)
        body = [ln for ln in code.splitlines() if ln.strip()]
        if not body:
            continue
        line0 = text[:match.start()].count("\n") + 1
        if any(ln.lstrip().startswith(">>>") for ln in body):
            source, kind = _repl_source(body), "REPL"
        elif _SIGNATURE.match(body[0]) or _TRACEBACK.match(body[0]):
            continue
        else:
            source, kind = code, "BLOCK"
        try:
            ast.parse(source)
        except SyntaxError as exc:
            if exc.msg in _SIGNATURE_ERRORS:
                continue
            found.append((f"SYNTAX_{kind}", f"line {line0}: {exc.msg}"))

    return found


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    root = (REPO_ROOT / args[0]).resolve() if args else REPO_ROOT / "docs"

    n_files = 0
    problems: list[tuple[Path, str, str]] = []
    for path in markdown_files(root):
        n_files += 1
        for kind, detail in check_page(path):
            problems.append((path.relative_to(REPO_ROOT), kind, detail))

    for rel, kind, detail in problems:
        print(f"{kind:18s} {rel}\n{'':18s}  {detail}")

    print(f"\n{n_files} file(s), {len(problems)} structural problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
