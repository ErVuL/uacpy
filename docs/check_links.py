#!/usr/bin/env python3
"""Verify every relative link and image in the documentation resolves on disk.

Markdown rots quietly: a page gets renamed, a figure stem changes, and the
link stays green-looking until a reader clicks it. This walks every ``.md``
under the repo, resolves each relative target against the containing file, and
reports the ones that do not exist.

External links (``http://``, ``https://``, ``mailto:``) are not fetched — this
is an offline check by design.

Usage
-----
    python docs/check_links.py            # whole repo
    python docs/check_links.py docs/      # a subtree

Exit status is non-zero if any link is broken, so it works as a CI gate.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parent.parent

# [text](target) and ![alt](target) — the target group stops at ')' or whitespace
_LINK = re.compile(r'!?\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)')
_SKIP_SCHEME = ('http://', 'https://', 'mailto:', 'ftp://', '#')
_SKIP_DIRS = {'.git', 'uacpy_venv', 'node_modules', '__pycache__',
              'third_party', 'data_cache', '.pytest_cache'}


def markdown_files(root: Path):
    for path in sorted(root.rglob('*.md')):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        yield path


def broken_links(path: Path):
    """Yield ``(line_no, target)`` for each relative link that does not exist."""
    text = path.read_text(encoding='utf-8', errors='replace')
    for line_no, line in enumerate(text.splitlines(), start=1):
        for target in _LINK.findall(line):
            if target.startswith(_SKIP_SCHEME):
                continue
            # Strip any in-page anchor: docs/foo.md#section
            file_part = unquote(target.split('#', 1)[0])
            if not file_part:
                continue
            if not (path.parent / file_part).resolve().exists():
                yield line_no, target


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    root = (REPO_ROOT / args[0]).resolve() if args else REPO_ROOT

    n_files = n_links = n_broken = 0
    for path in markdown_files(root):
        n_files += 1
        text = path.read_text(encoding='utf-8', errors='replace')
        n_links += sum(1 for t in _LINK.findall(text)
                       if not t.startswith(_SKIP_SCHEME))
        for line_no, target in broken_links(path):
            n_broken += 1
            rel = path.relative_to(REPO_ROOT)
            print(f'{rel}:{line_no}: broken link -> {target}')

    print(f'\n{n_files} file(s), {n_links} relative link(s), '
          f'{n_broken} broken')
    return 1 if n_broken else 0


if __name__ == '__main__':
    raise SystemExit(main())
