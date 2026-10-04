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

import numpy as np
import pytest

import uacpy
from uacpy.models._registry import ENGINES

EXAMPLES_DIR = Path(uacpy.__file__).parent / "examples"

# The engines whose ``.run(...)`` spawns a binary shipped by ``install.sh``,
# and the OASES ones among them (academic-licensed, downloaded by
# ``install.sh --oases yes``), read from the engine registry. ``OASES`` is
# their common base, which an example can name too.
_BINARY_MODEL_CLASSES = frozenset(
    entry.class_name for entry in ENGINES.values()
    if entry.requires == ('binary',))
_OASES_MODEL_CLASSES = frozenset(
    entry.class_name for entry in ENGINES.values()
    if 'oases' in entry.requires) | {"OASES"}

# Examples that need a noticeably longer subprocess timeout (deep-ocean /
# multi-model / Lytaev-grid / live-fetch runs may take several minutes each).
_LONG_TIMEOUT_STEMS = {
    "example_02_sound_speed_profiles",
    "example_17_boundary_conditions_layered",
    "example_22_ram_lytaev_grid",
    # Three Bellhop runs over a 401-bin broadband grid plus two synthesis
    # passes: measured 178 s standalone on an IDLE machine, so it is over
    # the 120 s tier before any contention. The tier follows that
    # measurement, as the note above says — it is not a contended run being
    # waved through.
    "example_45_broadband_and_exposure_maps",
}

# The heaviest examples: GIF encoding across five solvers, and a broadband
# three-model comparison measured at ~133 s NOMINAL on an idle 2-worker run —
# under a fully loaded `-n logical` session (~3x contention) that already
# overruns the 360 s tier, so the tier assignment follows the measurement.
_EXTRA_LONG_TIMEOUT_STEMS = {
    "example_19_broadband_comparison",
    "example_26_wave_propagation",
    # Measured 193 s alone on an idle machine from the installed cache, 120 s
    # of it one Bellhop run (800 Hz over 374 km at Bellhop's own beam count,
    # about 26 000 beams; the receiver grid does not move it: 122 s at
    # 150x350, 118 s at 75x175). 10 000 beams take 20 s but move the coherent
    # field by 2 dB median per cell, so the example keeps the beam count and
    # this tier follows the measurement, as for the two above.
    "example_37_realworld_environment",
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
ALL_EXAMPLES = sorted(EXAMPLES_DIR.glob("example_*.py"))


def _guarded_nodes(tree) -> set:
    """Ids of every node inside a ``try`` body whose handlers catch
    ``ExecutableNotFoundError``: a model constructed and run there degrades
    to a printed precondition line when its binary is absent, so it is not a
    requirement of the example."""
    guarded = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        caught = []
        for handler in node.handlers:
            types = handler.type
            if types is None:
                continue
            for t in (types.elts if isinstance(types, ast.Tuple) else [types]):
                caught.append(t.attr if isinstance(t, ast.Attribute)
                              else getattr(t, "id", None))
        if "ExecutableNotFoundError" not in caught:
            continue
        for stmt in node.body:
            guarded.update(id(n) for n in ast.walk(stmt))
    return guarded


def _referenced_names(path: Path, *, unguarded_only: bool = False) -> Set[str]:
    """Names an example can construct a model through: ``from X import Y``
    bindings plus attribute references rooted at ``uacpy`` / ``uacpy.models``
    (``uacpy.OASS(...)``, ``uacpy.models.RAM(...)``). Covers every model-class
    reference pattern actually used by examples/. With ``unguarded_only``,
    references inside an ``ExecutableNotFoundError`` guard (see
    :func:`_guarded_nodes`) are left out.
    """
    tree = ast.parse(path.read_text())
    guarded = _guarded_nodes(tree) if unguarded_only else set()
    names: Set[str] = set()
    for node in ast.walk(tree):
        if id(node) in guarded:
            continue
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Attribute):
            root = node.value
            if isinstance(root, ast.Attribute) and root.attr == "models":
                root = root.value
            if isinstance(root, ast.Name) and root.id == "uacpy":
                names.add(node.attr)
    return names


def _example_marks(example: Path):
    """Derive (requires_binary, slow, requires_oases?) from the names the
    example references."""
    referenced = _referenced_names(example)
    # An OASES model built and run inside an ExecutableNotFoundError guard
    # (examples 03, 07, 08, 19) prints a precondition line and the example
    # carries on with its other models, so it does not require OASES.
    unguarded = _referenced_names(example, unguarded_only=True)
    needs_oases = bool(unguarded & _OASES_MODEL_CLASSES)
    needs_binary = needs_oases or bool(referenced & _BINARY_MODEL_CLASSES)
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
    if cwd is not None:
        # Examples honour UACPY_EXAMPLE_OUTPUT, so their PNGs land in the
        # per-test working directory instead of the shared examples/output/.
        env["UACPY_EXAMPLE_OUTPUT"] = str(cwd)
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
    # Each example runs in a per-test scratch dir that UACPY_EXAMPLE_OUTPUT
    # (set by _run) also names as its output directory, so its PNGs land
    # here — isolated from examples/output/ and visible to the checks below.
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
# demonstration, and that must not read as a marker. "skipped" is a marker
# too: the "SKIPPED: {e}" / "{name} skipped: {e}" / "[skipped] {e}" handlers
# of examples 19, 26 and 37 catch broad exceptions, and a precondition report
# ("needs ./install.sh", "executable not found") carries no defect signature.
_ERRORISH_LINE = re.compile(
    r"(?i)(?<![a-z0-9_])error\s*:|\[error\]|\bskipped\b")

_TRACEBACK_HEADER = "Traceback (most recent call last)"

# Exception class names, plus the message shapes a handler prints when it
# renders ``str(exc)`` instead of the type — which is the common case and
# is what let a Fortran fatal hide behind "✗ Kraken error: ...".
_REAL_DEFECT_SIGNATURES = (
    "TypeError", "AttributeError", "ValueError", "KeyError",
    "ModelExecutionError", "ConfigurationError", "UnboundLocalError",
    "IndexError", "NameError", "ZeroDivisionError",
    "execution failed", "output is unusable", "unexpected keyword",
    "has no attribute",
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
range' header sentinel (peramx.f90:128-131) — the deck cannot express it.
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
profile counter reads as a '-1 range' header sentinel (peramx.f90:128-131) \
— the deck cannot express it.
"""


def test_detector_fails_on_a_traceback_or_an_unmarked_ram_error_line():
    """The exact output that once sailed through must now fail — via the
    stderr traceback and via the unmarked stdout "RAM error:" line, each on
    its own, so removing either print path cannot re-open the hole."""
    with pytest.raises(AssertionError, match="swallowed|traceback"):
        _check_no_swallowed_failure(
            _EXAMPLE,
            _FakeResult(_OLD_SWALLOWED_STDOUT, _OLD_SWALLOWED_STDERR),
        )
    with pytest.raises(AssertionError,
                       match='printed a traceback for a real defect'):
        _check_no_swallowed_failure(
            _EXAMPLE, _FakeResult(stderr=_OLD_SWALLOWED_STDERR),
        )
    with pytest.raises(AssertionError,
                       match='a broad handler swallowed a real defect'):
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
        "RAM:mpirams: raised dz from 0.385 m to 0.935 m for mpiramS runtime "
        "cap (λ_p / 16). The Lytaev accuracy budget ε=1e-01 is not met on "
        "this grid — its predicted error is 4.30e-01.\n"
        "[2026/08/17 20:46:11 UTC] [WARN] [uacpy.examples.example_05:189] "
        "Kraken reported 2 non-fatal warning(s) in its .prt log:\n"
        "  Warning in KRAKENC - RootFinderSecant : Failure to converge\n"
    )))


def test_detector_fails_marked_defects():
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
            "'to_dB'\n"
        )))


@pytest.mark.parametrize("line", [
    "  SKIPPED: 'Field' object has no attribute 'to_dB'",        # example 19
    "  Kraken skipped: run() got an unexpected keyword argument 'x'",  # ex 26
    "  TL: [skipped] AttributeError: 'Bellhop' object has no attribute 'go'",
])
def test_detector_fails_a_skipped_line_carrying_a_defect_signature(line):
    """'skipped' is a marker like 'error:': a handler that prints
    'SKIPPED: {e}' / '{name} skipped: {e}' / '[skipped] {e}' (examples 19,
    26, 37) is scanned for a defect signature like any other degradation
    report, so a caught API drift fails the run."""
    with pytest.raises(AssertionError, match="swallowed"):
        _check_no_swallowed_failure(_EXAMPLE, _FakeResult(stdout=line + "\n"))


def test_detector_accepts_a_skipped_precondition_report():
    """Example 37's '[skipped] needs ./install.sh --data seaice' and a
    'skipped: <binary> not found' line name a precondition, not a defect."""
    _check_no_swallowed_failure(_EXAMPLE, _FakeResult(stdout=(
        "  sea ice: [skipped] needs ./install.sh --data seaice\n"
        "  Scooter skipped: scooter executable not found: install.sh\n"
        "  OAST skipped: OASES executable not found (./install.sh --oases "
        "yes)\n"
    )))


# ---------------------------------------------------------------------------
# The marker derivation's own contract, pinned on synthetic example files.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("construction", [
    "uacpy.OASS(correlation_length=10.0)",
    "uacpy.models.OASSP(correlation_length=10.0)",
])
def test_attribute_constructed_oases_models_mark_requires_oases(
        construction, tmp_path):
    """``uacpy.OASS(...)`` / ``uacpy.models.OASSP(...)`` constructions mark an
    example requires_oases exactly as a ``from uacpy import OASS`` does."""
    path = tmp_path / "example_synthetic.py"
    path.write_text(f"import uacpy\n\n\ndef main():\n    m = {construction}\n")
    marks = {m.name for m in _example_marks(path)}
    assert {"requires_oases", "requires_binary"} <= marks


def test_pure_python_attribute_references_carry_no_binary_marks(tmp_path):
    """``uacpy.Environment(...)`` names no model class, so the example keeps
    only the blanket ``slow`` mark."""
    path = tmp_path / "example_synthetic.py"
    path.write_text(
        "import uacpy\n\n\ndef main():\n"
        "    env = uacpy.Environment(bathymetry=100.0, ssp=1500.0)\n"
    )
    marks = {m.name for m in _example_marks(path)}
    assert marks == {"slow"}


def test_example_39_is_marked_requires_oases():
    marks = {m.name for m in _example_marks(
        EXAMPLES_DIR / "example_39_oass_reverberation.py")}
    assert {"requires_oases", "requires_binary"} <= marks


@pytest.mark.parametrize("guard, expect_oases", [
    ("except uacpy.ExecutableNotFoundError:", False),
    ("except ExecutableNotFoundError:", False),
    ("except (RuntimeError, uacpy.ExecutableNotFoundError):", False),
    ("except RuntimeError:", True),
])
def test_an_oases_model_guarded_by_its_missing_binary_does_not_require_oases(
        guard, expect_oases, tmp_path):
    """A ``try`` that catches ExecutableNotFoundError around the OASES
    construction makes OASES optional; any other handler leaves the
    requirement in place. Bellhop outside the guard keeps requires_binary."""
    path = tmp_path / "example_synthetic.py"
    path.write_text(
        "import uacpy\n\nfields = {'Bellhop': uacpy.Bellhop().run(env)}\n"
        "try:\n    fields['OAST'] = uacpy.OAST().run(env)\n"
        f"{guard}\n    print('  OAST skipped: OASES executable not found')\n"
    )
    marks = {m.name for m in _example_marks(path)}
    assert ("requires_oases" in marks) is expect_oases
    assert "requires_binary" in marks


@pytest.mark.parametrize("stem", ["example_03", "example_07", "example_08",
                                  "example_19"])
def test_the_optional_oast_examples_are_not_marked_requires_oases(stem):
    example, = EXAMPLES_DIR.glob(f"{stem}_*.py")
    marks = {m.name for m in _example_marks(example)}
    assert "requires_oases" not in marks
    assert "requires_binary" in marks


def test_every_example_puts_the_repo_root_on_sys_path():
    """The bootstrap line has to name the directory that HOLDS the ``uacpy``
    package, not the package itself.

    ``Path(__file__).parent.parent`` from ``uacpy/examples/x.py`` is
    ``uacpy/`` — so it never made ``import uacpy`` work from a checkout, and
    what it did do was publish ``core``, ``models``, ``visualization`` and
    ``io`` as top-level importable names (``io`` shadowing the stdlib module).
    """
    repo_root = EXAMPLES_DIR.parent.parent
    assert (repo_root / "uacpy" / "__init__.py").is_file(), repo_root
    offenders = []
    for path in sorted(EXAMPLES_DIR.glob("*.py")):
        for line in path.read_text().splitlines():
            if "sys.path.insert" not in line:
                continue
            if "Path(__file__).parents[2]" not in line:
                offenders.append(f"{path.name}: {line.strip()}")
    assert not offenders, (
        "examples put a directory other than the repo root on sys.path:\n"
        + "\n".join(offenders)
    )


def test_the_path_expression_resolves_to_the_repo_root():
    """Evaluated, not just pattern-matched: ``parents[2]`` has to be the
    directory holding the package for any of the above to mean anything."""
    example = next(EXAMPLES_DIR.glob("example_*.py"))
    assert example.resolve().parents[2] == EXAMPLES_DIR.parent.parent.resolve()
    assert (example.resolve().parents[2] / "uacpy" / "__init__.py").is_file()


# ---------------------------------------------------------------------------
# The PNG gate's own contract: the workdir glob sees real files.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_harness_env_var_lands_example_pngs_in_the_workdir(tmp_path):
    """``UACPY_EXAMPLE_OUTPUT`` points the example at the per-test workdir,
    so ``_check_pngs_well_formed`` inspects the files the example wrote:
    the workdir glob is non-empty for an example that saves a figure."""
    example = EXAMPLES_DIR / "example_25_canonical_presets.py"
    workdir = tmp_path / example.stem
    workdir.mkdir()
    result = _run(example, timeout=240, cwd=workdir)
    assert result.returncode == 0, (
        f"{example.name} failed (rc={result.returncode}):\n"
        f"--- stderr ---\n{result.stderr[-2000:]}"
    )
    assert list(workdir.glob("*.png")), (
        "the example exited 0 but wrote no PNG into the harness workdir — "
        "the PNG well-formedness gate would pass on an empty glob"
    )
    _check_pngs_well_formed(workdir)


# ---------------------------------------------------------------------------
# The examples prologue: UACPY_EXAMPLE_OUTPUT at an arbitrary depth.
# ---------------------------------------------------------------------------


def _mkdir_calls():
    """``(file, line, keyword names)`` for every ``mkdir`` in the examples."""
    calls = []
    for example in sorted(EXAMPLES_DIR.rglob("*.py")):
        for node in ast.walk(ast.parse(example.read_text())):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "mkdir"):
                calls.append((example, node.lineno,
                              {kw.arg for kw in node.keywords}))
    return calls


def _prologue_source(example):
    """The example's module-level statements up to and including its ``mkdir``
    — everything that runs before it imports anything from uacpy."""
    lines = example.read_text().splitlines()
    end = next(i for i, line in enumerate(lines) if ".mkdir(" in line)
    return "\n".join(lines[:end + 1])


def test_every_example_creates_the_parents_of_its_output_directory():
    """``UACPY_EXAMPLE_OUTPUT`` has no documented contract, so a path whose
    parent does not exist yet is an ordinary thing to point it at. Without
    ``parents=True`` that ``mkdir`` raises ``FileNotFoundError`` from a
    module-level statement and the example dies before importing anything.
    The harness cannot see it: it pre-creates the workdir it sets the variable
    to."""
    calls = _mkdir_calls()
    assert calls, "no mkdir call found — this gate is measuring nothing"
    offenders = [f"{path.name}:{line}" for path, line, kwargs in calls
                 if "parents" not in kwargs]
    assert offenders == []


@pytest.mark.parametrize("relative, parent_preexists", [
    ("out", True),                  # one new level under a directory that exists
    ("runs/today/out", False),      # the level that has to be created too
])
def test_an_example_prologue_creates_its_output_directory_at_any_depth(
        tmp_path, relative, parent_preexists):
    """Both sides of the boundary the ``mkdir`` sits on: whether the target's
    parent already exists."""
    example = EXAMPLES_DIR / "example_09_ambient_noise.py"
    nested = tmp_path / relative
    assert nested.parent.exists() is parent_preexists
    stub = tmp_path / example.name          # a real file, so __file__ resolves
    stub.write_text(_prologue_source(example))
    result = subprocess.run(
        [sys.executable, str(stub)], capture_output=True, text=True,
        env={**os.environ, "UACPY_EXAMPLE_OUTPUT": str(nested)})
    assert result.returncode == 0, result.stderr[-800:]
    assert nested.is_dir()


# ---------------------------------------------------------------------------
# Figure lifetime in the examples.
# ---------------------------------------------------------------------------


def _figure_balance(example):
    """``(figures opened with pyplot, plt.close calls)`` in one example."""
    opened = closed = 0
    for node in ast.walk(ast.parse(example.read_text())):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if getattr(node.func.value, "id", None) != "plt":
            continue
        if node.func.attr in ("subplots", "figure"):
            opened += 1
        elif node.func.attr == "close":
            closed += 1
    return opened, closed


def test_an_example_opening_several_figures_closes_them():
    """pyplot keeps a reference to every figure it makes, so a script that
    opens several and closes none holds them all until it exits — and
    matplotlib starts warning about the leak at twenty.

    The rule starts above one figure deliberately: a script that opens exactly
    one, saves it and exits accumulates nothing. ``example_20``, ``example_21``,
    ``example_22`` and ``example_24`` are in that shape and are outside this
    gate; they are not evidence that leaving several open is fine."""
    offenders = []
    for example in sorted(EXAMPLES_DIR.glob("example_*.py")):
        opened, closed = _figure_balance(example)
        if opened > 1 and closed < opened:
            offenders.append(f"{example.name}: opened {opened}, closed {closed}")
    assert offenders == []


def test_every_example_saves_through_the_figure_it_bound():
    """Every save names the figure the plotter returned, never pyplot's
    *current* one.

    ``plt.savefig()`` writes whichever figure pyplot considers current, so
    inserting a panel between a plotter call and its save silently writes the
    wrong figure — and in a script that builds several, the wrong figure is
    usually the last one touched. Binding the return value makes that
    impossible to express. Checked across every example rather than in the one
    that first got it wrong.
    """
    offenders = []
    for example in sorted(EXAMPLES_DIR.glob("example_*.py")):
        for node in ast.walk(ast.parse(example.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "savefig"
                    and getattr(node.func.value, "id", None) in ("plt",
                                                                 "pyplot",
                                                                 "matplotlib")):
                offenders.append(f"{example.name}:{node.lineno}")
    assert offenders == [], (
        "these saves go through pyplot's current figure instead of a bound "
        f"one: {offenders}")


def test_the_bound_figure_check_can_see_a_pyplot_save(tmp_path):
    """The sweep above passes trivially if the AST walk finds nothing."""
    probe = tmp_path / "example_99_probe.py"
    probe.write_text("import matplotlib.pyplot as plt\nplt.savefig('x.png')\n",
                     encoding="utf-8")
    found = [node for node in ast.walk(ast.parse(probe.read_text()))
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr == "savefig"
             and getattr(node.func.value, "id", None) == "plt"]
    assert len(found) == 1


def test_the_ambiguity_surfaces_are_drawn_through_the_library():
    """Example 38 hands its ``sonar.bartlett`` / ``sonar.mvdr`` surfaces to
    ``plot_matched_field`` rather than drawing them itself.

    It used to build an ``imshow`` extent by hand, and an extent taken from the
    outer grid CENTRES rather than the cell edges shifts the whole surface half
    a cell — the peak is then reported at a position the processor never
    scanned. It then routed each surface through a hand-built ``Field`` and
    hand-drew the truth star and the estimate circle, which is exactly what
    ``plot_matched_field`` draws from the candidate axes.
    """
    source = (EXAMPLES_DIR / 'example_38_matched_field.py').read_text(
        encoding='utf-8')
    assert 'plot_matched_field(' in source and 'true_position=' in source
    for hand_rolled in ('imshow(', 'extent=', 'Field(', "'w*'"):
        assert hand_rolled not in source, (
            f"example 38 is back to placing the surface or its markers itself "
            f"({hand_rolled}); plot_matched_field draws both from the "
            f"candidate positions.")


def test_the_comparison_examples_use_the_librarys_tl_difference_renderer():
    """The TL-difference panel is ``uacpy.plot.plot_field_difference``, not a
    copy inside an example.

    It was ``examples/plotting_utils._plot_tl_difference`` until it was
    promoted into the library; a local reimplementation pasted back into one
    example is the regression this catches, because it would drift from the
    tagging that keeps a signed residual off the TL colour scale.
    """
    for stem in ('example_05_ram_advanced',
                 'example_15_elastic_boundaries_comparison',
                 'example_16_bellhop_bounce_integration',
                 'example_18_rd_bottom_krakenfield_vs_ram',
                 'example_22_ram_lytaev_grid'):
        source = (EXAMPLES_DIR / f'{stem}.py').read_text(encoding='utf-8')
        assert 'plot_field_difference' in source, stem

    local = []
    for path in sorted(EXAMPLES_DIR.glob('example_*.py')):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if (isinstance(node, ast.FunctionDef)
                    and 'difference' in node.name.lower()):
                local.append(f'{path.name}:{node.name}')
    assert not local, (
        f"an example defines its own difference renderer: {local}. Use "
        f"uacpy.plot.plot_field_difference.")


def _module_constants(tree):
    """Module-level ``NAME = literal`` and ``a, b = 1, 2`` assignments."""
    consts = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        try:
            value = ast.literal_eval(node.value)
        except ValueError:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            consts[target.id] = value
        elif (isinstance(target, ast.Tuple)
                and all(isinstance(e, ast.Name) for e in target.elts)):
            consts.update(zip((e.id for e in target.elts), value))
    return consts


def _first_call(tree, name):
    """The first ``name(...)`` / ``x.name(...)`` call in ``tree``."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if ((isinstance(f, ast.Name) and f.id == name)
                    or (isinstance(f, ast.Attribute) and f.attr == name)):
                return node
    raise AssertionError(f"no call to {name}")


def _literal_kwargs(call, consts):
    """The keyword arguments of ``call`` that are literals or module
    constants; the others (an axes, an array) are left out."""
    out = {}
    for kw in call.keywords:
        v = kw.value
        if isinstance(v, ast.Name) and v.id in consts:
            out[kw.arg] = consts[v.id]
        else:
            try:
                out[kw.arg] = ast.literal_eval(v)
            except ValueError:
                continue
    return out


def test_example_06_marks_the_profile_ranges_its_run_held():
    """The dashed lines on the shelf TL panel sit at the ranges the adiabatic
    run held its profiles at — the ``n_segments`` linspace
    ``segment_environment_by_range`` builds over the same bathymetry — and
    nowhere else. Lines at the midpoints between profiles annotate the figure
    with switches the run never made."""
    from uacpy.core.units import m_to_km
    from uacpy.models.kraken._segments import segment_environment_by_range

    source = (EXAMPLES_DIR / 'example_06_kraken_advanced.py').read_text(
        encoding='utf-8')
    tree = ast.parse(source)
    n_segments = _module_constants(tree)['N_SEGMENTS']
    bathymetry = next(
        ast.literal_eval(node.value.args[0]) for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == 'bathymetry')
    env = uacpy.Environment(bathymetry=np.array(bathymetry, dtype=float))
    profiles_km = m_to_km(np.array(
        [r for r, _ in segment_environment_by_range(env, n_segments=n_segments)]))

    marker_loop = next(
        node for node in ast.walk(tree) if isinstance(node, ast.For)
        and any(isinstance(c, ast.Call) and getattr(c.func, 'attr', '') == 'axvline'
                for c in ast.walk(node)))
    drawn_km = np.asarray(eval(ast.get_source_segment(source, marker_loop.iter),
                               {'np': np, 'N_SEGMENTS': n_segments}), dtype=float)
    np.testing.assert_allclose(drawn_km, profiles_km[1:-1], err_msg=(
        f"the markers sit at {drawn_km} km; the run's profiles sit at "
        f"{profiles_km} km"))


def test_example_10_band_power_window_holds_the_sweep():
    """The constant-Q band-power panel's y window contains the level of the
    chirp it is titled for: the bulk of the in-band bins — their interquartile
    band — sits inside it. A window under the sweep shows only the leakage
    skirts, the curve climbing off the top edge between them."""
    from uacpy.acoustic_signal import constant_q
    from uacpy.acoustic_signal.generate import lfm_chirp
    from uacpy.core.acoustics import power_to_dB

    source = (EXAMPLES_DIR / 'example_10_signal_processing.py').read_text(
        encoding='utf-8')
    tree = ast.parse(source)
    consts = _module_constants(tree)
    chirp = _literal_kwargs(_first_call(tree, 'lfm_chirp'), consts)
    _, lfm = lfm_chirp(**chirp)
    # The panel is drawn from the example's own
    # ``constant_q(..., scaling='spectrum', ...)`` call.
    cq_call = _literal_kwargs(_first_call(tree, 'constant_q'), consts)
    cq = constant_q(lfm, chirp['sample_rate'], **cq_call)
    window = _literal_kwargs(_first_call(tree, 'plot_constant_q_psd'), consts)
    in_band = (cq.frequencies >= chirp['freq_start']) & (cq.frequencies <= chirp['freq_end'])
    levels = power_to_dB(cq.power, 1e-6)[in_band]     # as the plotter scales it
    q1, q3 = np.percentile(levels, [25, 75])
    assert window['ymin'] <= q1 and q3 <= window['ymax'], (
        f"the in-band interquartile band {q1:.0f}..{q3:.0f} dB is not inside "
        f"the panel's {window['ymin']}..{window['ymax']} dB window")


#: Hand-rolled spellings of a computation the package exports, and the call
#: that replaces each. An example is read as the way to use uacpy, so one that
#: re-derives an exported formula teaches the reader to bypass it.
_HAND_ROLLED = {
    "/ 10 ** (snr_dB / 10)) * rng.standard_normal": "comms.awgn",
    "np.arccos(": "uacpy.acoustics.critical_angle / Modes.grazing_angles",
    ".k.real": "Modes.phase_speeds",
    "record['delay']": "Arrivals.delays",
    "marker='*'": "plot_field(source=)",
    "np.asarray(f.data)": "ResultStack.p",
}

#: example_34 refers its SNR to the clean beacon rather than to the echo-laden
#: record it adds the noise to, which ``comms.awgn`` cannot express.
_HAND_ROLLED_ALLOWED = {
    ("example_34_janus_beacon", "/ 10 ** (snr_dB / 10)) * rng.standard_normal"),
}


def test_the_examples_call_the_package_for_what_it_exports():
    """No example re-implements AWGN, the critical/modal grazing angle, modal
    phase speeds, the arrival-delay list or the source marker.

    Examples 32/33 hand-rolled AWGN beside example 31's ``comms.awgn``;
    examples 15 and 41 spelled ``arccos(c/v)``, which the package keeps in one
    home; 06 rebuilt ``phase_speeds`` and ``excitation``; 11 rebuilt
    ``Arrivals.delays``; 04 and 06 drew source stars that
    ``plot_field(source=)`` draws; 42, 43 and 44 stacked slab pressure by
    hand where ``ResultStack.p`` returns it.
    """
    offences = []
    for path in sorted(EXAMPLES_DIR.glob("example_*.py")):
        text = " ".join(path.read_text(encoding="utf-8").split())
        for pattern, api in _HAND_ROLLED.items():
            if (" ".join(pattern.split()) in text
                    and (path.stem, pattern) not in _HAND_ROLLED_ALLOWED):
                offences.append(f"{path.name}: {pattern!r} -> use {api}")
    assert offences == [], "\n".join(offences)


#: Deep imports an example may keep because the name has no public export
#: yet, as ``(stem, module, name)``, each with the reason it stays.
_PRIVATE_IMPORT_ALLOWED: set = {
    # The example reads the JANUS standard's version and initial-band
    # constants (JANUS_VERSION, FC_INITIAL, BW_INITIAL) from their module.
    ("example_34_janus_beacon", "uacpy.comms", "janus"),
}


#: A public namespace deeper than ``uacpy.<package>``: the closed-form
#: physics functions are documented as ``uacpy.core.acoustics``.
_PUBLIC_NAMESPACE_EXTRAS = {"uacpy.core.acoustics"}


def test_the_examples_import_from_the_public_namespaces():
    """Every ``from uacpy… import name`` in an example imports from a public
    namespace (``uacpy`` or ``uacpy.<package>``, plus
    ``uacpy.core.acoustics``) whose ``__all__`` exports ``name``. A defining
    module's own ``__all__`` does not count: ``core.exceptions`` and
    ``core.metrics`` have one and are still not where a reader imports from.

    Fifteen examples reached into defining modules (``acoustic_signal.
    generate``, ``core.exceptions``, ``core.metrics``, ``comms.constellations`` …)
    for names the public namespaces export, teaching readers import paths
    that move whenever the code is reorganised."""
    import importlib
    offences = []
    for path in sorted(EXAMPLES_DIR.glob("example_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ImportFrom) and node.module
                    and node.module.split(".")[0] == "uacpy"):
                continue
            public = (node.module in _PUBLIC_NAMESPACE_EXTRAS
                      or (node.module.count(".") <= 1
                          and node.module != "uacpy.core"))
            exported = getattr(importlib.import_module(node.module),
                               "__all__", ()) if public else ()
            for alias in node.names:
                if (alias.name not in exported and (
                        path.stem, node.module, alias.name)
                        not in _PRIVATE_IMPORT_ALLOWED):
                    offences.append(f"{path.name}:{node.lineno} "
                                    f"from {node.module} import {alias.name}")
    assert offences == [], "\n".join(offences)


# ── docs/figure_scripts: every documentation figure builds ─────────────────
# docs/generate_model_figures.py regenerates the documentation figures and
# exits non-zero when one fails. It runs here once per page module, with
# UACPY_FIGURE_OUTPUT pointing the PNGs at the test's tmp_path, so the run
# never rewrites docs/. A source checkout carries docs/; an installed package
# does not, and then there is nothing to collect.
DOCS_DIR = EXAMPLES_DIR.parent.parent / "docs"
FIGURE_DRIVER = DOCS_DIR / "generate_model_figures.py"
FIGURE_MODULES = sorted(
    p for p in (DOCS_DIR / "figure_scripts").glob("*.py")
    if not p.name.startswith("_"))

# Wall clock measured on the 2026-10-03 figure baseline: the sonar page's
# replica banks take 246 s and the SPARC page's six transients 300 s; every
# other page takes under 60 s.
_LONG_FIGURE_MODULES = {"sonar", "sparc"}

# The datasets the data page reads from the install-time cache. Each of its
# figures pins source='local', so the network cannot stand in for them.
_DATA_FIGURE_DATASETS = ("gebco", "woa23", "emodnet", "sediment", "diesing",
                         "graw", "crust1", "globsed", "seaice", "coastline")


def _figure_module_marks(module: Path):
    """``slow`` always; ``requires_binary`` / ``requires_oases`` from the
    engines the module names, as for an example; the data page is skipped
    when the install-time cache lacks one of its datasets."""
    referenced = _referenced_names(module)
    needs_oases = bool(referenced & _OASES_MODEL_CLASSES)
    marks = [pytest.mark.slow]
    if needs_oases or referenced & _BINARY_MODEL_CLASSES:
        marks.append(pytest.mark.requires_binary)
    if needs_oases:
        marks.append(pytest.mark.requires_oases)
    if module.stem == "data" and not _offline_cache_ready(
            _DATA_FIGURE_DATASETS):
        marks.append(pytest.mark.skip(
            reason="the data page reads the install-time cache "
                   "(./install.sh --data all)"))
    return marks


@pytest.mark.parametrize("module", [
    pytest.param(p, marks=_figure_module_marks(p), id=p.stem)
    for p in FIGURE_MODULES])
def test_every_documentation_figure_builds(module, tmp_path):
    """Every figure of one docs/figure_scripts page builds through the
    driver and writes a well-formed PNG."""
    stems = _figure_stems(module)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(EXAMPLES_DIR.parent.parent), env.get("PYTHONPATH", "")])
    env.setdefault("MPLBACKEND", "Agg")
    env["UACPY_FIGURE_OUTPUT"] = str(tmp_path)
    result = subprocess.run(
        [sys.executable, str(FIGURE_DRIVER), module.stem],
        cwd=str(tmp_path), capture_output=True, text=True, env=env,
        timeout=900 if module.stem in _LONG_FIGURE_MODULES else 300)
    assert result.returncode == 0, (
        f"{module.name}: a figure failed (rc={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout[-3000:]}\n"
        f"--- stderr ---\n{result.stderr[-3000:]}")
    assert f"[{module.stem}]" in result.stdout
    written = {p.stem for p in tmp_path.glob("*.png")}
    assert written == stems, (
        f"{module.name}: FIGURES names {sorted(stems)}, the driver wrote "
        f"{sorted(written)}")
    _check_pngs_well_formed(tmp_path)


def _figure_stems(module: Path) -> set:
    """The stems a page's ``FIGURES`` mapping names, read from its source:
    the keys of the dict literal assigned to ``FIGURES``."""
    tree = ast.parse(module.read_text(encoding="utf-8"))
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "FIGURES"
                        for t in node.targets)
                and isinstance(node.value, ast.Dict)):
            return {k.value for k in node.value.keys
                    if isinstance(k, ast.Constant)}
    raise AssertionError(f"{module.name}: no FIGURES dict literal")
