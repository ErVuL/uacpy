#!/usr/bin/env python3
"""Run every uacpy example end-to-end and print a pass/fail summary.

Unlike the pytest smoke tests (which gate on markers — ``slow`` /
``requires_binary`` / ``requires_network``), this runs **all** examples in
this directory unconditionally, so it's the "exercise everything on my
machine" tool. Each example runs as an isolated subprocess with a headless
matplotlib backend; figures/animations land in ``output/`` (each example's
own ``OUTPUT_DIR``).

Usage
-----
    python run_all_examples.py                 # run every example
    python run_all_examples.py 26 ram          # only stems matching 26 / ram
    python run_all_examples.py --timeout 1200  # per-example timeout (s)

Exit status is non-zero if any example fails (so it doubles as a local
gate). Requires the native binaries (``./install.sh``) for the model
examples and the data cache / network for ``example_37`` — a failure there
just means those prerequisites are absent on this machine.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_DIR.parent.parent          # holds the importable `uacpy`


def discover(filters):
    examples = sorted(EXAMPLES_DIR.glob("example_*.py"))
    if filters:
        examples = [p for p in examples
                    if any(f in p.stem for f in filters)]
    return examples


def run_one(path: Path, timeout: float):
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")          # headless
    # Make the in-tree package importable even without `pip install -e`.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env.get("PYTHONPATH", "")]
    )
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(EXAMPLES_DIR),
            capture_output=True, text=True, timeout=timeout, env=env,
        )
        ok = proc.returncode == 0
        detail = "" if ok else (
            (proc.stderr.strip().splitlines() or ["(no stderr)"])[-1])
    except subprocess.TimeoutExpired:
        ok, detail = False, f"timeout after {timeout:.0f}s"
    return ok, time.time() - t0, detail


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Run all uacpy examples and report pass/fail.")
    ap.add_argument("filters", nargs="*",
                    help="substrings selecting example stems (e.g. 26 ram)")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="per-example timeout in seconds (default 900)")
    args = ap.parse_args(argv)

    examples = discover(args.filters)
    if not examples:
        print("No matching examples.", file=sys.stderr)
        return 1

    print(f"Running {len(examples)} example(s) from {EXAMPLES_DIR}\n")
    failures = []
    for path in examples:
        print(f"  {path.stem:<46}", end=" ", flush=True)
        ok, dt, detail = run_one(path, args.timeout)
        print(f"{'ok  ' if ok else 'FAIL'} {dt:7.1f}s"
              + (f"   {detail}" if detail else ""))
        if not ok:
            failures.append(path.stem)

    passed = len(examples) - len(failures)
    print(f"\n{passed}/{len(examples)} passed.")
    if failures:
        print("Failed: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
