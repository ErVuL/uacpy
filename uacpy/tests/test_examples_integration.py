"""
Auto-discovered smoke tests for uacpy/examples/.

Every example runs end-to-end as a subprocess with a generous timeout.
Examples that drive a native binary (Bellhop, Kraken, RAM, …) are
additionally tagged ``slow`` so they're skipped by the default
``pytest -m "not slow"`` run; pure-Python examples (signal processing,
canonical presets, ambient noise) run on the fast path.

The marker assignment is derived statically from each example's
``from uacpy.models import ...`` line so it can't drift away from the
example's actual dependencies.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path
from typing import Set

import pytest

import uacpy

EXAMPLES_DIR = Path(uacpy.__file__).parent / "examples"

# Model classes whose ``.run(...)`` spawns one of the OALIB / RAM
# Fortran/C++ binaries shipped by ``install.sh``.
_BINARY_MODEL_CLASSES = frozenset({
    "Bellhop", "BellhopCUDA",
    "Kraken", "KrakenC", "KrakenField",
    "Scooter", "SPARC", "Bounce",
    "RAM",
})

# Sub-classes of OASES — academic-licensed, downloaded by
# ``install.sh --oases yes``. ``OASES`` is the factory.
_OASES_MODEL_CLASSES = frozenset({
    "OAST", "OASN", "OASR", "OASP", "OASES",
})

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


def _imported_names(path: Path) -> Set[str]:
    """Names brought into an example's namespace via ``from X import Y``.

    Module-level only (no ``import inside-a-function`` parsing); covers
    every model-class import pattern actually used by examples/.
    """
    tree = ast.parse(path.read_text())
    names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def _example_marks(example: Path):
    """Derive (requires_binary, slow, requires_oases?) from imports."""
    imported = _imported_names(example)
    needs_oases = bool(imported & _OASES_MODEL_CLASSES)
    needs_binary = needs_oases or bool(imported & _BINARY_MODEL_CLASSES)
    marks = []
    if needs_binary:
        marks.extend([pytest.mark.requires_binary, pytest.mark.slow])
    if needs_oases:
        marks.append(pytest.mark.requires_oases)
    return marks


def _params(examples):
    return [
        pytest.param(p, marks=_example_marks(p), id=p.stem)
        for p in examples
    ]


def _run(
    example: Path, timeout: int, cwd: Path | None = None,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    # Make sure the in-tree `uacpy` package is importable when `pip install -e`
    # was not used.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(EXAMPLES_DIR.parent.parent), env.get("PYTHONPATH", "")]
    )
    env.setdefault("MPLBACKEND", "Agg")
    return subprocess.run(
        [sys.executable, str(example)],
        cwd=str(cwd) if cwd is not None else str(EXAMPLES_DIR),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _check_pngs_well_formed(example_dir: Path) -> None:
    """Every PNG the example wrote in its working directory must carry a
    valid PNG signature and be at least 1 KiB. A 0-byte PNG, or a binary
    that doesn't start with the magic, almost always means a silent
    matplotlib regression that ``returncode == 0`` would miss.
    """
    for png in example_dir.glob("*.png"):
        # Ignore tiny PNGs (icons, etc.) — generated figures from
        # examples are typically 50-500 KiB.
        size = png.stat().st_size
        assert size >= 1024, (
            f"{png.name}: {size} bytes is too small to be a real plot"
        )
        with png.open("rb") as fh:
            header = fh.read(8)
        assert header == _PNG_SIG, (
            f"{png.name}: missing PNG signature (got {header!r})"
        )


@pytest.mark.parametrize("example", _params(ALL_EXAMPLES))
def test_example_runs(example, tmp_path):
    """Run an example end-to-end, verify clean exit + any PNG output."""
    timeout = 240 if example.stem in _LONG_TIMEOUT_STEMS else 120
    # Run inside a per-test scratch dir so PNG droppings stay isolated
    # and we can assert on them without polluting examples/.
    workdir = tmp_path / example.stem
    workdir.mkdir()
    result = _run(example, timeout=timeout, cwd=workdir)
    assert result.returncode == 0, (
        f"{example.name} failed (rc={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout[-2000:]}\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )
    _check_pngs_well_formed(workdir)
