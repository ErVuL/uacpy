#!/usr/bin/env python3
"""Regenerate every figure used by the documentation under ``docs/``.

Each page has one module in ``docs/figure_scripts/`` exposing a ``FIGURES``
mapping of ``stem -> callable`` returning a matplotlib ``Figure``. This driver
imports them all, runs them, and writes ``docs/models/figures/<stem>.png`` (or
``docs/guide/figures/`` for the guide pages).

Usage
-----
    python docs/generate_model_figures.py              # everything
    python docs/generate_model_figures.py bellhop ram  # only these modules
    python docs/generate_model_figures.py --list       # show what exists

Exit status is non-zero if any figure fails, so this doubles as a check that
every documented example still runs. Requires the native binaries
(``./install.sh``) for the model pages.
"""

from __future__ import annotations

import argparse
import importlib
import os
import pkgutil
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')

DOCS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DOCS_DIR.parent
sys.path.insert(0, str(DOCS_DIR))
sys.path.insert(0, str(REPO_ROOT))


def discover(filters):
    """Module name -> imported module, for every figure_scripts submodule."""
    import figure_scripts

    found = {}
    for info in pkgutil.iter_modules(figure_scripts.__path__):
        if info.name.startswith('_'):
            continue
        if filters and not any(f in info.name for f in filters):
            continue
        found[info.name] = importlib.import_module(
            f'figure_scripts.{info.name}')
    return dict(sorted(found.items()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('filters', nargs='*',
                    help='substrings selecting figure modules (e.g. bellhop)')
    ap.add_argument('--list', action='store_true',
                    help='list modules and their figures, generate nothing')
    args = ap.parse_args(argv)

    modules = discover(args.filters)
    if not modules:
        print('No matching figure modules.', file=sys.stderr)
        return 1

    if args.list:
        for name, mod in modules.items():
            figures = getattr(mod, 'FIGURES', {})
            print(f'{name} ({len(figures)})')
            for stem in figures:
                print(f'    {stem}')
        return 0

    from figure_scripts import _common

    total = failed = 0
    t_all = time.time()
    for name, mod in modules.items():
        figures = getattr(mod, 'FIGURES', {})
        guide = bool(getattr(mod, 'GUIDE', False))
        print(f'\n[{name}] {len(figures)} figure(s)')
        for stem, builder in figures.items():
            total += 1
            t0 = time.time()
            try:
                fig = builder()
                path = _common.save(fig, stem, guide=guide)
                rel = path.relative_to(REPO_ROOT)
                print(f'  ok   {rel}  ({time.time() - t0:.1f}s)')
            except Exception:
                failed += 1
                print(f'  FAIL {stem}  ({time.time() - t0:.1f}s)')
                traceback.print_exc()

    print(f'\n{total - failed}/{total} figures generated '
          f'in {time.time() - t_all:.0f}s')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
