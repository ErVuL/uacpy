#!/usr/bin/env python3
"""Verify every relative link and image in the documentation resolves on disk.

Markdown rots quietly: a page gets renamed, a figure stem changes, and the
link stays green-looking until a reader clicks it. This walks every ``.md``
under the repo, resolves each relative target against the containing file, and
reports the ones that do not exist.

The ``#fragment`` half of a link is checked too (:func:`broken_anchors`): the
pages cross-reference each other by section number, so renumbering a heading
breaks every ``[§4](#4-…)`` pointing at it while leaving the file path — and
therefore :func:`broken_links` — perfectly happy.

External links (``http://``, ``https://``, ``mailto:``) are not fetched — this
is an offline check by design. Links inside fenced code blocks are sample text
rather than navigation, so both checks skip them (:func:`link_targets`).

Usage
-----
    python docs/check_links.py            # whole repo
    python docs/check_links.py docs/      # a subtree

Exit status is non-zero if any link is broken. CI does not call this script:
``uacpy/tests/test_documentation.py`` imports it and runs the same functions
over the ``docs/`` tree plus the repo-root ``DOCUMENTATION.md`` and
``README.md``; the no-argument command line here walks the **whole repo**, a
superset of that gate scope.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parent.parent

# [text](target) and ![alt](target) — the target group stops at ')' or whitespace
_LINK = re.compile(r'!?\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)')
_SKIP_SCHEME = ('http://', 'https://', 'mailto:', 'ftp://', '#')
_HEADING = re.compile(r'^(#{1,6})\s+(.*?)\s*#*\s*$')
_HTML_ANCHOR = re.compile(r'<a\s+(?:id|name)=["\']([^"\']+)["\']', re.I)
_CODE_SPAN = re.compile(r'`([^`]*)`')
_INLINE_LINK = re.compile(r'!?\[([^\]]*)\]\([^)]*\)')
_FENCE = re.compile(r'^\s*```')
_SKIP_DIRS = {'.git', 'uacpy_venv', 'node_modules', '__pycache__',
              'third_party', 'data_cache', '.pytest_cache'}


def markdown_files(root: Path):
    for path in sorted(root.rglob('*.md')):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        yield path


def link_targets(text: str):
    """Yield ``(line_no, target)`` for every relative link outside a fenced
    code block.

    A fenced block is sample text, not navigation: a page showing
    ``[report](out/run.md)`` in a usage example is documenting a command, not
    pointing at a file that has to exist. :func:`broken_anchors` has always
    tracked fences; this applies the same rule to link targets."""
    in_fence = False
    for line_no, line in enumerate(text.splitlines(), start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for target in _LINK.findall(line):
            if not target.startswith(_SKIP_SCHEME):
                yield line_no, target


def broken_links(path: Path):
    """Yield ``(line_no, target)`` for each relative link that does not exist."""
    text = path.read_text(encoding='utf-8', errors='replace')
    for line_no, target in link_targets(text):
        # Strip any in-page anchor: docs/foo.md#section
        file_part = unquote(target.split('#', 1)[0])
        if not file_part:
            continue
        if not (path.parent / file_part).resolve().exists():
            yield line_no, target


def heading_slug(text: str) -> str:
    """GitHub's heading anchor for ``text``.

    github-slugger lowercases, drops every character that is not a letter,
    digit, ``-``, ``_``, space or a combining mark, then turns spaces into
    hyphens. It does **not** trim, which is why a heading opening with an
    emoji anchors as ``#-licensing``: the space the emoji left behind becomes
    the leading hyphen.
    """
    text = _CODE_SPAN.sub(r'\1', text)
    text = _INLINE_LINK.sub(r'\1', text)
    text = unicodedata.normalize('NFC', text)
    kept = [
        ch for ch in text.lower()
        if ch.isalnum() or ch in '-_ ' or unicodedata.category(ch)[0] == 'M'
    ]
    return ''.join(kept).replace(' ', '-')


def anchors(path: Path) -> set:
    """Every fragment ``path`` defines: one per heading, plus explicit HTML
    anchors. Repeats take github-slugger's ``-1``, ``-2`` … suffixes."""
    found, seen = set(), {}
    in_fence = False
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        found.update(_HTML_ANCHOR.findall(line))
        match = _HEADING.match(line)
        if not match:
            continue
        base = heading_slug(match.group(2))
        n = seen.get(base, 0)
        seen[base] = n + 1
        found.add(base if n == 0 else f'{base}-{n}')
    return found


def broken_anchors(path: Path, cache: dict = None):
    """Yield ``(line_no, target)`` for each link whose ``#fragment`` names no
    heading in the file it points at.

    ``cache`` maps resolved path -> anchor set across calls; pass one when
    checking a whole tree so each page is slugged once.
    """
    cache = {} if cache is None else cache
    in_fence = False
    text = path.read_text(encoding='utf-8', errors='replace')
    for line_no, line in enumerate(text.splitlines(), start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for target in _LINK.findall(line):
            if target.startswith(('http://', 'https://', 'mailto:', 'ftp://')):
                continue
            file_part, _, fragment = target.partition('#')
            fragment = unquote(fragment)
            if not fragment:
                continue
            dest = (path if not file_part
                    else (path.parent / unquote(file_part)).resolve())
            if dest.suffix != '.md' or not dest.is_file():
                continue        # a missing file is broken_links' report
            if dest not in cache:
                cache[dest] = anchors(dest)
            if fragment not in cache[dest]:
                yield line_no, target


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    root = (REPO_ROOT / args[0]).resolve() if args else REPO_ROOT

    n_files = n_links = n_broken = 0
    cache: dict = {}
    for path in markdown_files(root):
        n_files += 1
        text = path.read_text(encoding='utf-8', errors='replace')
        n_links += sum(1 for _ in link_targets(text))
        rel = path.relative_to(REPO_ROOT)
        for line_no, target in broken_links(path):
            n_broken += 1
            print(f'{rel}:{line_no}: broken link -> {target}')
        for line_no, target in broken_anchors(path, cache):
            n_broken += 1
            print(f'{rel}:{line_no}: dead anchor -> {target}')

    print(f'\n{n_files} file(s), {n_links} relative link(s), '
          f'{n_broken} broken')
    return 1 if n_broken else 0


if __name__ == '__main__':
    raise SystemExit(main())
