"""
Auto-discovered smoke tests for uacpy/examples/.

Every example runs end-to-end as a subprocess with a generous timeout.
All examples are tagged ``slow`` so the default ``pytest -m "not slow"``
run skips them; the on-demand / nightly path runs them via plain
``pytest`` or ``pytest -m slow``.

Documentation-style checks of the examples (e.g. "examples 11+ must not
import example_helpers") are NOT here — those are static lints, see
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

# Examples that import (and run) at least one OASES wrapper — must be
# de-selectable with `-m "not requires_oases"` when the OASES binaries are
# absent. Docstring-only mentions of OASES don't count.
_OASES_STEMS = {
    "example_02_sound_speed_profiles",
    "example_03_multi_frequency",
    "example_07_all_models_comparison",
    "example_08_long_range",
    "example_13_oases_suite",
}

# Examples that need a noticeably longer subprocess timeout (deep-ocean /
# multi-model / Lytaev-grid runs may take several minutes each).
_LONG_TIMEOUT_STEMS = {
    "example_02_sound_speed_profiles",
    "example_17_boundary_conditions_layered",
    "example_19_broadband_comparison",
    "example_22_ram_lytaev_grid",
}

ALL_EXAMPLES = sorted(
    p for p in EXAMPLES_DIR.glob("example_*.py")
    if p.name != "example_helpers.py"
)


def _example_marks(example: Path):
    marks = [pytest.mark.requires_binary, pytest.mark.slow]
    if example.stem in _OASES_STEMS:
        marks.append(pytest.mark.requires_oases)
    return marks


def _params(examples):
    return [
        pytest.param(p, marks=_example_marks(p), id=p.stem)
        for p in examples
    ]


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


@pytest.mark.parametrize("example", _params(ALL_EXAMPLES))
def test_example_runs(example):
    """Run an example end-to-end and assert it exits cleanly."""
    timeout = 240 if example.stem in _LONG_TIMEOUT_STEMS else 120
    result = _run(example, timeout=timeout)
    assert result.returncode == 0, (
        f"{example.name} failed (rc={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )
