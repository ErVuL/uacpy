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
import re
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
    "Bellhop",
    "Kraken",
    "Scooter", "SPARC", "Bounce",
    "RAM",
})

# Sub-classes of OASES — academic-licensed, downloaded by
# ``install.sh --oases yes``. ``OASES`` is the factory.
_OASES_MODEL_CLASSES = frozenset({
    "OAST", "OASN", "OASR", "OASP", "OASES",
})

# Examples that need a noticeably longer subprocess timeout (deep-ocean /
# multi-model / Lytaev-grid / live-fetch runs may take several minutes each).
_LONG_TIMEOUT_STEMS = {
    "example_02_sound_speed_profiles",
    "example_17_boundary_conditions_layered",
    "example_19_broadband_comparison",
    "example_22_ram_lytaev_grid",
    "example_37_realworld_environment",
}

# Encoding five per-solver GIFs is the heaviest example by far — give it lots
# of headroom so a slow runner doesn't time out mid-render.
_EXTRA_LONG_TIMEOUT_STEMS = {
    "example_26_wave_propagation",
}

# Examples that source live ocean databases. These are *cache-first*: with the
# install-time cache (``install.sh --data all``) they run fully offline, so they
# only need the network as a fallback when the cache is absent. ``_example_marks``
# therefore tags them ``requires_network`` ONLY when the cache cannot satisfy
# them; with the cache present they run in the default suite, offline.
_NETWORK_STEMS = {
    "example_37_realworld_environment",
}

# Datasets example_37 needs to assemble its environment offline (GEBCO bathy,
# WOA23 SSP, EMODnet seabed for the North Sea transect).
_EXAMPLE_37_DATASETS = ("gebco", "woa23", "emodnet")


def _offline_cache_ready(datasets):
    """True when every named dataset is present in the install-time cache."""
    try:
        from uacpy.data import _cache
        for ds in datasets:
            _cache.require(ds)
        return True
    except Exception:
        return False

# Every example runs — no example is silently excluded from the suite. The
# marks below gate WHEN each runs (binary/oases/network availability, slow
# path), not WHETHER it exists as a test.
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
    # Every example is an end-to-end integration test that spawns a full
    # Python + matplotlib subprocess, so all are ``slow`` — otherwise the
    # pure-Python compute-heavy examples (noise / signal / comms, which import
    # no binary model) leak into the fast ``-m "not slow"`` feedback subset.
    marks = [pytest.mark.slow]
    if needs_binary:
        marks.append(pytest.mark.requires_binary)
    if needs_oases:
        marks.append(pytest.mark.requires_oases)
    if example.stem in _NETWORK_STEMS and not _offline_cache_ready(
        _EXAMPLE_37_DATASETS
    ):
        # No usable cache → the example would fall back to the live services,
        # so it genuinely needs the network here.
        marks.append(pytest.mark.requires_network)
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
    """Run an example end-to-end, verify clean exit + any PNG output.

    The three tiers below are wall-clock subprocess timeouts, so they measure
    contention as much as work: an example sized against the default 120 s on
    an idle machine can exceed it when the suite runs under ``-n`` and every
    worker is spawning its own binary. A timeout here is therefore evidence
    about the machine first and the example second — re-run the case alone
    before treating it as a real failure, and do not raise the bound to make a
    contended run go green.
    """
    if example.stem in _EXTRA_LONG_TIMEOUT_STEMS:
        timeout = 900
    elif example.stem in _LONG_TIMEOUT_STEMS:
        timeout = 360
    else:
        timeout = 120
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
    _check_no_swallowed_failure(example, result)


# Examples wrap optional sections in broad ``except Exception`` handlers so a
# missing binary or absent network degrades gracefully instead of killing the
# script. That also means a stale API call cannot fail the run — it prints a
# warning, skips its figure, and still exits 0. Catch that here.
_SWALLOW_MARKERS = ("! Warning: Could not", "✗ ", "Traceback (most recent call last)")

# Unmarked degradation reports: handlers that print "RAM error: {e}" /
# "{label} ERROR: {e}" carry no ✗ marker at all. The lookbehind keeps
# CamelCase exception-class names out — example 20 prints
# "→ UnsupportedFeatureError: ..." every run as a deliberate gap
# demonstration, and that must not read as a marker.
_ERRORISH_LINE = re.compile(r"(?i)(?<![a-z0-9_])error\s*:|\[error\]")

_TRACEBACK_HEADER = "Traceback (most recent call last)"

# Exception class names, plus the message shapes a handler prints when it
# renders ``str(exc)`` instead of the type — which is the common case and
# is what let a Fortran fatal hide behind "✗ Kraken error: ...".
_REAL_DEFECT_SIGNATURES = (
    "TypeError", "AttributeError", "ValueError", "KeyError",
    "ModelExecutionError", "ConfigurationError", "UnboundLocalError",
    "IndexError", "NameError", "ZeroDivisionError",
    "execution failed", "unexpected keyword", "has no attribute",
    "object is not", "not enough values", "too many values",
    # ConfigurationError's deck-refusal shape, as rendered by str(exc)
    # (no class name on the line) in "RAM error: {e}"-style handlers.
    "the deck cannot express",
)


def _traceback_exception_lines(text: str):
    """The ``SomeError: message`` line terminating each traceback in ``text``.

    ``traceback.print_exc()`` indents every frame line; the first
    non-indented, non-blank line after the header is the exception itself,
    the only line of the block that carries the class name."""
    lines = text.splitlines()
    found = []
    for i, line in enumerate(lines):
        if _TRACEBACK_HEADER not in line:
            continue
        for follow in lines[i + 1:]:
            if not follow.strip() or follow.startswith((" ", "\t")):
                continue
            found.append(follow)
            break
    return found


def _check_no_swallowed_failure(example: Path, result) -> None:
    """Fail if an example degraded silently instead of doing its work.

    A handler that reports a *precondition* (no binary, no network, no
    licence) is legitimate; one reporting a ``TypeError`` / ``AttributeError``
    / ``ModelExecutionError`` is a real defect the handler is hiding.

    Both streams are scanned: ``traceback.print_exc()`` inside a handler
    writes to STDERR while the handler's own status line goes to stdout, so
    a stdout-only scan let example 05's swallowed ConfigurationError sail
    through."""
    for text in (result.stdout, result.stderr):
        for line in text.splitlines():
            marked = (any(m in line for m in _SWALLOW_MARKERS)
                      or _ERRORISH_LINE.search(line))
            if not marked:
                continue
            if any(sig in line for sig in _REAL_DEFECT_SIGNATURES):
                raise AssertionError(
                    f"{example.name}: a broad handler swallowed a real "
                    f"defect, so the example exited 0 without doing its "
                    f"work:\n  {line.strip()}"
                )
        # A printed traceback carries its class name only on the block's
        # final exception line, never on the marker line itself.
        for exc_line in _traceback_exception_lines(text):
            if any(sig in exc_line for sig in _REAL_DEFECT_SIGNATURES):
                raise AssertionError(
                    f"{example.name}: a broad handler printed a traceback "
                    f"for a real defect and exited 0 without doing its "
                    f"work:\n  {exc_line.strip()}"
                )


# ---------------------------------------------------------------------------
# The detector's own contract, pinned on synthetic subprocess results.
# ---------------------------------------------------------------------------

class _FakeResult:
    """Stand-in for subprocess.CompletedProcess with just the two streams."""

    def __init__(self, stdout: str = "", stderr: str = ""):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = 0


_EXAMPLE = EXAMPLES_DIR / "example_05_ram_advanced.py"

# Example 05's output before its handler was de-silenced: the status line
# went to stdout with no ✗ marker and no class name, and print_exc() put the
# traceback on stderr — the old stdout-only, marker-gated scan passed both.
_OLD_SWALLOWED_STDOUT = """\
2. Running RAM (parabolic equation)...
  RAM error: mpiramS sediment file: cs profile(s) start with a negative \
value (min -21.32), which the binary's profile counter reads as a '-1 \
range' header sentinel (peramx.f90:120-123) — the deck cannot express it.
3. Running Kraken for comparison...
  ✓ Kraken completed (using range-independent approximation)
✓ Example 05 complete
"""

_OLD_SWALLOWED_STDERR = """\
Traceback (most recent call last):
  File "example_05_ram_advanced.py", line 163, in <module>
    result_ram = ram.run(env, source, receiver)
  File "uacpy/models/ram.py", line 1603, in run
    return self._run_tl(env, source, receiver)
uacpy.core.exceptions.ConfigurationError: mpiramS sediment file: cs \
profile(s) start with a negative value (min -21.32), which the binary's \
profile counter reads as a '-1 range' header sentinel (peramx.f90:120-123) \
— the deck cannot express it.
"""


def test_detector_fails_example_05s_old_swallowed_output():
    """The exact output that once sailed through must now fail — via the
    stderr traceback and via the unmarked stdout "RAM error:" line, each on
    its own, so removing either print path cannot re-open the hole."""
    with pytest.raises(AssertionError, match="swallowed|traceback"):
        _check_no_swallowed_failure(
            _EXAMPLE,
            _FakeResult(_OLD_SWALLOWED_STDOUT, _OLD_SWALLOWED_STDERR),
        )
    with pytest.raises(AssertionError):
        _check_no_swallowed_failure(
            _EXAMPLE, _FakeResult(stderr=_OLD_SWALLOWED_STDERR),
        )
    with pytest.raises(AssertionError):
        _check_no_swallowed_failure(
            _EXAMPLE, _FakeResult(stdout=_OLD_SWALLOWED_STDOUT),
        )


def test_detector_accepts_example_20s_gap_demonstration():
    """Example 20 prints an UnsupportedFeatureError line every run as a
    deliberate demonstration of the rams/ramsurf capability gap; the
    CamelCase class name must not read as an 'error:' marker."""
    _check_no_swallowed_failure(_EXAMPLE, _FakeResult(stdout=(
        "  elastic + altimetry → UnsupportedFeatureError: RAM does not "
        "support: elastic bottom + sea-surface altimetry\n"
        "✓ Example 20 complete\n"
    )))


def test_detector_accepts_precondition_reports():
    """A handler that reports a missing binary / network is a legitimate
    graceful degradation, on either stream, marked or not."""
    _check_no_swallowed_failure(_EXAMPLE, _FakeResult(
        stdout=(
            "  ✗ Kraken not available: [Errno 2] No such file or directory\n"
            "! Warning: Could not fetch bathymetry (network unreachable) — "
            "using flat default\n"
            "  Scooter error: scooter binary not found in uacpy/bin — run "
            "install.sh\n"
        ),
        stderr=(
            "Traceback (most recent call last):\n"
            '  File "example.py", line 10, in <module>\n'
            "    model.run(env, source, receiver)\n"
            "uacpy.core.exceptions.ExecutableNotFoundError: scooter binary "
            "not found in uacpy/bin — run install.sh\n"
        ),
    ))


def test_detector_accepts_the_suites_warning_stream():
    """The [WARN] banners a green run writes to stderr (grid raises,
    below-seafloor receivers, 'predicted error is …' accuracy notes) carry
    error-ish words without the error: shape and must pass."""
    _check_no_swallowed_failure(_EXAMPLE, _FakeResult(stderr=(
        "[2026/08/17 20:46:10 UTC] [WARN] [uacpy.examples.example_05:171] "
        "RAM:mpiramS: raised dz from 0.385 m to 0.935 m for mpiramS runtime "
        "cap (λ_p / 16). The Lytaev accuracy budget ε=1e-01 is not met on "
        "this grid — its predicted error is 4.30e-01.\n"
        "[2026/08/17 20:46:11 UTC] [WARN] [uacpy.examples.example_05:189] "
        "Kraken reported 2 non-fatal warning(s) in its .prt log:\n"
        "  Warning in KRAKENC - RootFinderSecant : Failure to converge\n"
    )))


def test_detector_still_fails_marked_defects():
    """The original stdout contract is preserved: a ✗-marked line whose
    message shape betrays a stale API call is a real defect."""
    with pytest.raises(AssertionError, match="swallowed"):
        _check_no_swallowed_failure(_EXAMPLE, _FakeResult(stdout=(
            "  ✗ Kraken error: run() got an unexpected keyword argument "
            "'beam_type'\n"
        )))


def test_detector_fails_unmarked_error_lines_with_defect_signatures():
    """An 'ERROR: {e}'-style line (examples 17/18) whose message carries a
    defect class name fails even with no marker and no traceback."""
    with pytest.raises(AssertionError, match="swallowed"):
        _check_no_swallowed_failure(_EXAMPLE, _FakeResult(stdout=(
            "    KrakenField       ERROR: 'Field' object has no attribute "
            "'to_db'\n"
        )))
