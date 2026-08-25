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
    "OAST", "OASN", "OASR", "OASP", "OASS", "OASSP", "OASES",
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
ALL_EXAMPLES = sorted(EXAMPLES_DIR.glob("example_*.py"))


def _referenced_names(path: Path) -> Set[str]:
    """Names an example can construct a model through: ``from X import Y``
    bindings plus attribute references rooted at ``uacpy`` / ``uacpy.models``
    (``uacpy.OASS(...)``, ``uacpy.models.RAM(...)``). Covers every model-class
    reference pattern actually used by examples/.
    """
    tree = ast.parse(path.read_text())
    names: Set[str] = set()
    for node in ast.walk(tree):
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
    needs_oases = bool(referenced & _OASES_MODEL_CLASSES)
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


def test_detector_fails_on_a_traceback_or_an_unmarked_ram_error_line():
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
            "'to_db'\n"
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
# plotting_utils: the shared report panels.
# ---------------------------------------------------------------------------


def _plotting_utils():
    """``uacpy/examples`` carries no ``__init__.py``, so the shared helper is
    loaded straight from its path rather than imported by package name."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "uacpy_examples_plotting_utils", EXAMPLES_DIR / "plotting_utils.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tl_field(ranges, level_db):
    """A Field whose TL at every depth is ``level_db + 20·log10(r)``."""
    import numpy as np
    from uacpy.core.results import Field

    depths = np.linspace(0.0, 200.0, 21)
    tl = np.tile(level_db + 20.0 * np.log10(np.maximum(ranges, 1.0)),
                 (depths.size, 1))
    return Field(data=tl, coords={'depth': depths, 'range': ranges},
                 model='test')


def _rms_tiles(results):
    """The numbers the RMS-error panel prints, keyed by ``(row, column)``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = _plotting_utils().plot_model_statistics(results, source_depth=100.0)
    tiles = {(round(t.get_position()[1]), round(t.get_position()[0])):
             t.get_text() for t in fig.axes[1].texts}
    plt.close(fig)
    return tiles


def test_the_rms_panel_compares_only_ranges_both_models_computed():
    """The panel is labelled "RMS Error (dB)". Two models on the same *count*
    of ranges over different spans are a pairing ``uacpy.metrics.tl_rmse``
    refuses outright, and differencing them cell-by-cell publishes a number for
    ranges that never met."""
    import numpy as np
    import pytest as _pytest

    a = _tl_field(np.linspace(50.0, 3000.0, 200), 60.0)
    b = _tl_field(np.linspace(500.0, 8000.0, 200), 66.0)
    with _pytest.raises(Exception):                 # the library's own metric
        __import__('uacpy').metrics.tl_rmse(a, b)
    # 500-3000 m is the shared span, where the two differ by exactly 6 dB
    assert _rms_tiles({'A': a, 'B': b}) == {(0, 1): '6.0', (1, 0): '6.0'}


def test_the_rms_panel_leaves_an_aligned_pair_untouched():
    """The common-grid step must be an identity on a pair already sharing one
    axis, and on the unequal-length pair the panel already interpolated."""
    import numpy as np

    span = np.linspace(50.0, 3000.0, 200)
    assert _rms_tiles({'C': _tl_field(span, 60.0),
                       'D': _tl_field(span, 70.5)}) == {(0, 1): '10.5',
                                                        (1, 0): '10.5'}
    assert _rms_tiles(
        {'E': _tl_field(np.linspace(50.0, 3000.0, 40), 60.0),
         'F': _tl_field(np.linspace(50.0, 3000.0, 25), 70.5)}) == {
            (0, 1): '10.5', (1, 0): '10.5'}


def test_the_rms_panel_reports_no_number_for_models_sharing_no_range():
    """Two spans that do not touch have nothing to compare, so the tile carries
    no figure and does not take the diagonal's zero-error colour."""
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    import matplotlib.pyplot as plt

    results = {'G': _tl_field(np.linspace(50.0, 500.0, 30), 60.0),
               'H': _tl_field(np.linspace(5000.0, 8000.0, 30), 60.0)}
    assert _rms_tiles(results) == {(0, 1): 'n/a', (1, 0): 'n/a'}

    fig = _plotting_utils().plot_model_statistics(results, source_depth=100.0)
    im = fig.axes[1].get_images()[0]
    rgba = im.cmap(im.norm(np.ma.filled(im.get_array(), np.nan)))
    plt.close(fig)
    assert tuple(rgba[0, 0]) == plt.get_cmap('RdYlGn_r')(0.0)   # the diagonal
    assert tuple(rgba[0, 1]) != plt.get_cmap('RdYlGn_r')(0.0)


@pytest.mark.parametrize("start_b, tile", [
    (3000.0, '6.0'),      # the spans meet on one shared range: comparable
    (3000.001, 'n/a'),    # a millimetre further apart and they share nothing
])
def test_the_smallest_overlap_the_rms_panel_will_compare(start_b, tile):
    """Both sides of the boundary between a comparison and no comparison: the
    two spans touching at a single range is still a range both models
    computed."""
    import numpy as np

    a = _tl_field(np.linspace(50.0, 3000.0, 200), 60.0)
    b = _tl_field(np.linspace(start_b, 8000.0, 200), 66.0)
    assert _rms_tiles({'A': a, 'B': b})[(0, 1)] == tile


def test_the_rms_panel_diagonal_keeps_the_colormaps_zero_tile():
    """The diagonal is a true zero and must stay visually distinct from a
    not-comparable tile."""
    import matplotlib
    matplotlib.use("Agg")
    import numpy as np
    import matplotlib.pyplot as plt

    span = np.linspace(50.0, 3000.0, 200)
    fig = _plotting_utils().plot_model_statistics(
        {'C': _tl_field(span, 60.0), 'D': _tl_field(span, 70.5)},
        source_depth=100.0)
    im = fig.axes[1].get_images()[0]
    diagonal = im.cmap(im.norm(np.ma.filled(im.get_array(), np.nan)))[0, 0]
    plt.close(fig)
    assert tuple(diagonal) == plt.get_cmap('RdYlGn_r')(0.0)


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


def test_example_04_saves_through_the_figure_it_bound():
    """Every save in this example names the figure the plotter returned rather
    than pyplot's current one, so inserting a panel between a plotter call and
    its save cannot silently write the wrong figure."""
    source = (EXAMPLES_DIR / "example_04_bellhop_advanced.py").read_text()
    saves = [node for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute)
             and node.func.attr == "savefig"]
    assert len(saves) == 5, len(saves)
    assert [getattr(node.func.value, "id", None) for node in saves] == [
        "fig1", "fig2", "fig3", "fig4", "fig5"]

