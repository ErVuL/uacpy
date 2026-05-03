"""
Auto-discovered smoke tests for uacpy/examples/.

Examples 01 and 05 are the cheap introductory walkthroughs that finish in
seconds and run on every PR (60-second subprocess timeout). The rest are
gated behind ``slow`` and run on the nightly / on-demand path with a
generous 1200-second timeout, since the Lytaev-optimized PE grids can
push deep-ocean / multi-model examples to several minutes each.

Documentation-style checks of the examples (e.g. "examples 11+ must not import
example_helpers") are NOT here — those are static lints, see
``scripts/check_example_helpers.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import uacpy

EXAMPLES_DIR = Path(uacpy.__file__).parent / "examples"

# Slow examples (>=30 s on the reference machine) are gated behind ``slow``;
# everything else runs on every PR.
_SLOW_STEMS = {
    "example_02_sound_speed_profiles",
    "example_17_boundary_conditions_layered",
    "example_19_broadband_comparison",
    "example_22_ram_lytaev_grid",
}
FAST_EXAMPLES = sorted(
    p for p in EXAMPLES_DIR.glob("example_*.py")
    if p.stem not in _SLOW_STEMS and p.name != "example_helpers.py"
)
SLOW_EXAMPLES = sorted(
    p for p in EXAMPLES_DIR.glob("example_*.py")
    if p.stem in _SLOW_STEMS
)


def _run(example: Path, timeout: int) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    # Make sure the in-tree `uacpy` package is importable when `pip install -e`
    # was not used. This mirrors what the example would do when run by hand.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(EXAMPLES_DIR.parent.parent), env.get("PYTHONPATH", "")]
    )
    # Force matplotlib non-interactive in case the example imports pyplot.
    env.setdefault("MPLBACKEND", "Agg")
    return subprocess.run(
        [sys.executable, str(example)],
        cwd=str(EXAMPLES_DIR),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


@pytest.mark.requires_binary
@pytest.mark.parametrize("example", FAST_EXAMPLES, ids=lambda p: p.stem)
def test_fast_example_runs(example):
    """Fast examples (<30s on the reference machine) run on every PR."""
    result = _run(example, timeout=120)
    assert result.returncode == 0, (
        f"{example.name} failed (rc={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )


@pytest.mark.requires_binary
@pytest.mark.slow
@pytest.mark.parametrize("example", SLOW_EXAMPLES, ids=lambda p: p.stem)
def test_slow_example_runs(example):
    """Examples >=30 s run end-to-end on the nightly path."""
    result = _run(example, timeout=240)
    assert result.returncode == 0, (
        f"{example.name} failed (rc={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )
