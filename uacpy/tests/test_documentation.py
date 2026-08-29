"""
Gate tests for the ``docs/`` tree.

The two checkers under ``docs/`` are the single source of truth; this module
imports and runs them so a developer gets the same verdict from ``pytest`` as
from the command line, and so CI enforces them without extra workflow steps.

Both checks are pure-Python and take about a second — no binaries, no network,
no ``slow`` marker. Figure regeneration is deliberately *not* tested here: it
needs the native binaries and runs for several minutes, so it stays a manual
step (``python docs/generate_model_figures.py``).

What these catch that the rest of the suite does not: a renamed page leaving a
dead cross-reference, and the wreckage of an interrupted or twice-applied edit
— duplicated blocks, unbalanced fences, code samples that fail to parse.

The second half of this module gates ``DOCUMENTATION.md`` against the *code*:
the §7 capability matrix is parsed and compared against each model's
``ModelSpec`` / ``_supports_*`` flags, and every §18 parameter-reference
default is compared against the live constructor / ``run()`` signature via
``inspect.signature``. Parsing is structural (tables found by heading, cells
read by column name, mode names and yes/no tokens extracted — never exact
prose), so wording polish does not break it; renaming a model, dropping a
column, or drifting a default does.

Three further gates hold the doc tree to its own promises: the §17 examples
index must list exactly the ``uacpy/examples/example_NN_*.py`` scripts on
disk, the ``models/README.md`` run-mode matrix must agree with every model's
``ModelSpec.modes``, and each model page's worked-example code must be
verbatim figure-script code (the pages claim "the code below is that code").
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

import uacpy

# The Markdown-citation machinery these gates exercise, shared with
# test_packaging.py which runs it against the real tree.
from uacpy.tests._doc_gate import (
    _DOC_ANCHOR,
    _MIN_DOC_ANCHORS_RESOLVED,
    _MIN_QUOTED_DOC_ANCHORS,
    _NOT_DOCUMENTATION,
    _OWN_DOC_LINE_PIN,
    _headings,
    _line_pin_verdict,
    _repo_markdown,
    _resolve_markdown,
    _section_span,
)

DOCS_DIR = Path(uacpy.__file__).resolve().parent.parent / "docs"
REPO_ROOT = Path(uacpy.__file__).resolve().parent.parent

#: Applied to the tests that actually READ ``docs/``. It is deliberately
#: NOT a module-level ``pytestmark``: most of this file gates DOCUMENTATION.md,
#: the shipped sources and synthetic documents, and those must keep running in
#: a checkout without ``docs/`` -- the one place where a silently skipped doc
#: gate would matter most.
requires_docs = pytest.mark.skipif(
    not DOCS_DIR.is_dir(),
    reason="docs/ is not present (source checkout only)",
)


def _load(name: str) -> ModuleType:
    """Import a checker from ``docs/`` by path; it is not an installed module."""
    path = DOCS_DIR / f"{name}.py"
    if not path.is_file():
        pytest.skip(f"{path} is missing")
    spec = importlib.util.spec_from_file_location(f"_docs_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The repo-root pages sit outside docs/ and would otherwise escape every gate.
REPO_ROOT_MD_PAGES = ("DOCUMENTATION.md", "README.md")


def _gate_pages(checker):
    """The pages the four docs gates hold: the ``docs/`` tree plus the
    repo-root ``DOCUMENTATION.md`` and ``README.md``."""
    yield from checker.markdown_files(DOCS_DIR)
    for name in REPO_ROOT_MD_PAGES:
        path = DOCS_DIR.parent / name
        if path.is_file():
            yield path


@requires_docs
def test_no_broken_relative_links() -> None:
    """Every relative link and image target in the docs resolves on disk."""
    checker = _load("check_links")

    broken = [
        f"{path.relative_to(checker.REPO_ROOT)}:{line}: -> {target}"
        for path in _gate_pages(checker)
        for line, target in checker.broken_links(path)
    ]

    assert not broken, "broken relative link(s):\n" + "\n".join(broken)


@requires_docs
def test_every_section_cross_reference_names_a_real_heading() -> None:
    """Every ``#fragment`` in the docs resolves to a heading in the file it
    points at.

    The pages cross-reference each other by section number — ``[§4](#4-…)`` —
    so renumbering or retitling a heading breaks every pointer at it while the
    file path stays valid, which is all :func:`broken_links` looks at.
    """
    checker = _load("check_links")
    cache = {}

    dead = [
        f"{path.relative_to(checker.REPO_ROOT)}:{line}: -> {target}"
        for path in _gate_pages(checker)
        for line, target in checker.broken_anchors(path, cache)
    ]

    assert not dead, "dead section anchor(s):\n" + "\n".join(dead)


@requires_docs
def test_documentation_is_structurally_sound() -> None:
    """No unbalanced fences, duplicated blocks, dead figures or bad tables."""
    checker = _load("check_structure")

    problems = [
        f"{path.relative_to(checker.REPO_ROOT)}: {kind}: {detail}"
        for path in _gate_pages(checker)
        for kind, detail in checker.check_page(path)
    ]

    assert not problems, "structural problem(s):\n" + "\n".join(problems)


@requires_docs
def test_every_documented_code_sample_parses() -> None:
    """Python samples and REPL transcripts are syntactically valid.

    Separated from the structural test so a broken sample names itself in the
    failure rather than hiding among unrelated formatting problems.
    """
    checker = _load("check_structure")

    broken = [
        f"{path.relative_to(checker.REPO_ROOT)}: {kind}: {detail}"
        for path in _gate_pages(checker)
        for kind, detail in checker.check_page(path)
        if kind.startswith("SYNTAX_")
    ]

    assert not broken, "unparseable code sample(s):\n" + "\n".join(broken)


# ═══════════════════════════════════════════════════════════════════════════
# DOCUMENTATION.md ↔ code consistency
# ═══════════════════════════════════════════════════════════════════════════

import ast      # noqa: E402
import pkgutil  # noqa: E402
import inspect  # noqa: E402
import re       # noqa: E402

DOCUMENTATION_MD = Path(uacpy.__file__).resolve().parent.parent / "DOCUMENTATION.md"


def _documentation_text() -> str:
    if not DOCUMENTATION_MD.is_file():
        pytest.skip("DOCUMENTATION.md is not present")
    return DOCUMENTATION_MD.read_text(encoding="utf-8")


def _model_classes() -> dict:
    """The models the DOCUMENTATION.md tables describe, by name.

    Derived from ``uacpy.models.__all__``. The gates below compare the
    documented rows against *this* set, so a hand-written list here would let
    a newly-added wrapper ship with no matrix row and a green suite — it
    would simply never enter the comparison."""
    from uacpy.tests.conftest import concrete_model_classes
    return concrete_model_classes()


# ── structural markdown-table extraction ──────────────────────────────────


def _split_row(row: str) -> list:
    """Cells of one ``| a | b |`` line, splitting on *unescaped* pipes only
    (Meaning cells legitimately carry ``\\|`` inside math like ``\\|RG\\|``)."""
    cells = [c.strip() for c in re.split(r"(?<!\\)\|", row)]
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return cells


def _iter_heading_tables(text: str, level: str = "###"):
    """Yield ``(heading, header_cells, row_cells)`` for the first table under
    each ``level`` heading. Fenced code blocks are ignored, so a ``#``-comment
    inside an example can never masquerade as a heading."""
    lines = text.splitlines()
    heading = None
    table: list = []
    in_fence = False

    def emit():
        if heading is not None and len(table) >= 2:
            return (heading, _split_row(table[0]),
                    [_split_row(r) for r in table[2:]])
        return None

    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if re.match(rf"{re.escape(level)} ", line):
            out = emit()
            if out:
                yield out
            heading, table = line[len(level) + 1:].strip(), []
            continue
        if heading is None:
            continue
        if line.lstrip().startswith("|"):
            table.append(line.strip())
        elif table:
            # first non-table line after the table: the section's table is done
            out = emit()
            if out:
                yield out
            heading = None
            table = []
    out = emit()
    if out:
        yield out


def _column(header: list, keyword: str) -> int:
    for i, cell in enumerate(header):
        if keyword in cell.lower():
            return i
    pytest.fail(
        f"DOCUMENTATION.md table header {header} lost its {keyword!r} column "
        f"— the structural contract this test parses by has changed."
    )


# ── §7 capability matrix ──────────────────────────────────────────────────


def _capability_matrix():
    """(header, rows) of the ``### Capability matrix`` table."""
    for heading, header, rows in _iter_heading_tables(_documentation_text()):
        if "capability matrix" in heading.lower():
            return header, rows
    pytest.fail("DOCUMENTATION.md has no 'Capability matrix' section")


def _matrix_model_name(cell: str) -> str:
    return re.sub(r"[*`]", "", cell).strip()


def _documented_modes(cell: str) -> set:
    """RunMode names a matrix cell claims: literal enum-name tokens, plus the
    ``TL (coh/incoh/semicoh)`` shorthand."""
    from uacpy.models import RunMode
    out = {m.name for m in RunMode if re.search(rf"\b{m.name}\b", cell)}
    low = cell.lower()
    if re.search(r"\bTL\b", cell) and "coh" in low:
        if re.search(r"(?<!in)(?<!semi)coh", low):
            out.add("COHERENT_TL")
        if "incoh" in low:
            out.add("INCOHERENT_TL")
        if "semicoh" in low:
            out.add("SEMICOHERENT_TL")
    return out


def _yes_no(cell: str):
    """True/False for a plain yes/no cell; None for a qualified cell like
    ``(multi-freq sweep)`` — those carry nuance the flags cannot express, so
    they are deliberately not compared."""
    c = cell.strip().lower()
    if c.startswith("yes"):
        return True
    if c == "no" or c.startswith("no "):
        return False
    return None


def test_capability_matrix_run_modes_match_model_specs() -> None:
    """Every capability-matrix row lists exactly the ``RunMode``s the model's
    ``ModelSpec.modes`` declares — no documented mode a model lacks, no
    supported mode the docs hide."""
    classes = _model_classes()
    header, rows = _capability_matrix()
    ci_model, ci_modes = _column(header, "model"), _column(header, "run modes")

    problems = []
    seen = set()
    for row in rows:
        name = _matrix_model_name(row[ci_model])
        cls = classes.get(name)
        if cls is None:
            problems.append(f"{name}: matrix row has no model class")
            continue
        seen.add(name)
        doc = _documented_modes(row[ci_modes])
        spec = {m.name for m in cls.spec.modes}
        if doc != spec:
            problems.append(
                f"{name}: documented modes {sorted(doc)} != spec.modes "
                f"{sorted(spec)} (cell: {row[ci_modes]!r})"
            )
    missing = set(classes) - seen
    if missing:
        problems.append(f"models absent from the matrix: {sorted(missing)}")
    assert not problems, "capability-matrix mode drift:\n" + "\n".join(problems)


def test_capability_matrix_broadband_column_matches_specs() -> None:
    """A plain yes/no in the Broadband column must agree with ``BROADBAND``
    membership in ``spec.modes``; qualified cells (e.g. a multi-frequency
    sweep that is not a BROADBAND run mode) are skipped by design."""
    classes = _model_classes()
    header, rows = _capability_matrix()
    ci_model, ci_bb = _column(header, "model"), _column(header, "broadband")

    problems = []
    for row in rows:
        cls = classes.get(_matrix_model_name(row[ci_model]))
        if cls is None:
            continue  # reported by the run-modes test
        documented = _yes_no(row[ci_bb])
        if documented is None:
            continue
        actual = any(m.name == "BROADBAND" for m in cls.spec.modes)
        if documented != actual:
            problems.append(
                f"{cls.__name__}: Broadband column says {documented}, "
                f"spec.modes says {actual}"
            )
    assert not problems, "Broadband-column drift:\n" + "\n".join(problems)


@pytest.mark.requires_binary
def test_capability_matrix_feature_flags_match_models() -> None:
    """Range-dep./Elastic/Altimetry columns match the ``_supports_*`` flags of
    a default-constructed instance (instances, not the class spec, because
    Bellhop resolves ``_supports_range_dependent_ssp`` per-instance from
    ``interp_ssp`` — bellhop.py). Needs binaries: constructors resolve their
    executable."""
    from uacpy.core.exceptions import ExecutableNotFoundError

    classes = _model_classes()
    header, rows = _capability_matrix()
    ci_model = _column(header, "model")
    ci_rd = _column(header, "range")
    ci_el = _column(header, "elastic")
    ci_alt = _column(header, "altimetry")

    problems = []
    for row in rows:
        name = _matrix_model_name(row[ci_model])
        cls = classes.get(name)
        if cls is None:
            continue  # reported by the run-modes test
        kwargs = {"verbose": False}
        # OASS/OASSP refuse construction without the roughness statistics;
        # any other constructor failure propagates as the defect it is.
        if "correlation_length" in inspect.signature(cls).parameters:
            kwargs["correlation_length"] = 10.0
        try:
            model = cls(**kwargs)
        except ExecutableNotFoundError:
            pytest.skip(f"{name} binary not installed")

        rd_cell = row[ci_rd].lower()
        for token, flag in (("bathy", "range_dependent_bathymetry"),
                            ("ssp", "range_dependent_ssp"),
                            ("bottom", "range_dependent_bottom")):
            documented = token in rd_cell
            actual = getattr(model, f"_supports_{flag}")
            if documented != actual:
                problems.append(
                    f"{name}: Range-dep. cell {row[ci_rd]!r} implies "
                    f"{flag}={documented}, instance says {actual}"
                )
        for ci, flag in ((ci_el, "elastic_media"), (ci_alt, "altimetry")):
            documented = _yes_no(row[ci])
            if documented is None:
                continue
            actual = getattr(model, f"_supports_{flag}")
            if documented != actual:
                problems.append(
                    f"{name}: {flag}: documented {documented}, "
                    f"instance says {actual}"
                )
    assert not problems, "feature-flag drift:\n" + "\n".join(problems)


# ── §18 parameter reference ───────────────────────────────────────────────


# (owner, parameter) pairs the generic default-comparison must not flag:
#   - SPARC.timeout: the common-plumbing row itself documents the exception
#     ("SPARC default 180.0") in its Meaning cell — prose this parser
#     deliberately does not read.
_KNOWN_DEFAULT_EXCEPTIONS = {
    ("SPARC", "timeout"),
}


def _section_18_tables():
    """``(heading, header, rows)`` for every table in §18."""
    text = _documentation_text()
    match = re.search(r"^## 18\..*$", text, re.M) or re.search(
        r"^##.*Parameter Reference.*$", text, re.M)
    if match is None:
        pytest.fail("DOCUMENTATION.md has no §18 / Parameter Reference section")
    section = text[match.end():]
    nxt = re.search(r"^## ", section, re.M)
    if nxt:
        section = section[: nxt.start()]
    return list(_iter_heading_tables(section))


def _doc_default(cell: str):
    """``(kind, value)`` for a Default cell: ``('literal', v)`` for a
    backticked Python literal, ``('required', None)`` for *required*,
    ``('opaque', None)`` for anything else (``—``, prose)."""
    tokens = re.findall(r"`([^`]+)`", cell)
    if not tokens:
        return ("required" if "required" in cell.lower() else "opaque"), None
    try:
        return "literal", ast.literal_eval(tokens[0])
    except (ValueError, SyntaxError):
        try:
            return "literal", float(tokens[0])
        except ValueError:
            return "opaque", None


def _check_documented_default(owner: str, sig, param: str, cell: str) -> list:
    """Problems for one (signature, parameter, Default-cell) triple."""
    if (owner.split(".")[0], param) in _KNOWN_DEFAULT_EXCEPTIONS:
        return []
    if param not in sig.parameters:
        return [f"{owner}: documented parameter {param!r} is not in the "
                f"signature"]
    actual = sig.parameters[param].default
    kind, documented = _doc_default(cell)
    if kind == "opaque":
        return []
    if kind == "required":
        if actual is not inspect.Parameter.empty:
            return [f"{owner}.{param}: documented *required* but the "
                    f"signature default is {actual!r}"]
        return []
    if actual is inspect.Parameter.empty:
        return [f"{owner}.{param}: documented default {documented!r} but the "
                f"signature has no default (required)"]
    matches = (actual == documented
               or (isinstance(actual, tuple)
                   and tuple(actual) == tuple(documented)))
    if not matches:
        return [f"{owner}.{param}: documented default {documented!r} != "
                f"signature default {actual!r}"]
    return []


def _section_18_targets(heading: str, classes: dict):
    """Signatures a §18 table documents: ``[(owner_name, signature), ...]``,
    ``'dotted'`` for the Source/Receiver table, or None for an unmapped
    section (ignored — new prose sections must not break the parser)."""
    low = heading.lower()
    if "common plumbing" in low:
        return [(n, inspect.signature(c.__init__)) for n, c in classes.items()]
    if "run()" in low:
        return [(f"{n}.run", inspect.signature(c.run))
                for n, c in classes.items()]
    if "source" in low and "receiver" in low:
        return "dotted"
    oases = re.match(r"OASES\s*[—–-]+\s*(\w+)", heading)
    name = oases.group(1) if oases else heading.split()[0]
    cls = classes.get(name)
    if cls is None:
        return None
    return [(name, inspect.signature(cls.__init__))]


def test_parameter_reference_defaults_match_signatures() -> None:
    """Every backticked literal in a §18 Default column equals the live
    default of that parameter in the documented signature, and every
    documented parameter exists there. Opaque cells (``—``, prose) and the
    two annotated exceptions are skipped; *required* rows must have no
    default."""
    classes = _model_classes()
    tables = _section_18_tables()
    assert tables, "§18 contains no parsable tables"

    problems = []
    checked = 0
    for heading, header, rows in tables:
        targets = _section_18_targets(heading, classes)
        if targets is None:
            continue
        ci_param = 0
        for key in ("param", "argument"):
            if any(key in c.lower() for c in header):
                ci_param = _column(header, key)
                break
        ci_default = _column(header, "default")
        for row in rows:
            if len(row) <= max(ci_param, ci_default):
                continue
            for param in re.findall(r"`([^`]+)`", row[ci_param]):
                param = param.strip()
                if targets == "dotted":
                    if "." not in param:
                        continue
                    owner, attr = param.split(".", 1)
                    cls = getattr(uacpy, owner, None)
                    if cls is None:
                        problems.append(f"[{heading}] {param}: uacpy has no "
                                        f"{owner!r}")
                        continue
                    problems += _check_documented_default(
                        param, inspect.signature(cls.__init__), attr,
                        row[ci_default])
                    checked += 1
                else:
                    for owner_name, sig in targets:
                        problems += _check_documented_default(
                            owner_name, sig, param, row[ci_default])
                        checked += 1
    # Structural self-check: silence here must mean "everything agreed",
    # never "the parser matched nothing".
    assert checked > 100, (
        f"§18 parser only checked {checked} (owner, parameter) pairs — the "
        f"section structure has changed and the test needs updating"
    )
    assert not problems, (
        "§18 default drift:\n" + "\n".join(problems)
    )


def test_oasn_white_noise_level_default_is_none_sentinel() -> None:
    """OASN's ``white_noise_level`` defaults to ``None`` (written as -200 dB
    — a 1e-20 linear power, numerically off), because OASES itself has no off
    switch and an explicit ``0.0`` means a literal 0 dB per sensor. §18 once
    documented ``0.0`` / "0 disables" here; this pins the sentinel so neither
    side regresses."""
    classes = _model_classes()
    sig = inspect.signature(classes["OASN"].__init__)
    assert sig.parameters["white_noise_level"].default is None


# ── §17 examples index ────────────────────────────────────────────────────

EXAMPLES_DIR = Path(uacpy.__file__).resolve().parent / "examples"


def _section_17_rows():
    """``(number_cell, topic_cell)`` per data row of the §17 table."""
    text = _documentation_text()
    match = re.search(r"^## .*Examples Index.*$", text, re.M)
    if not match:
        pytest.fail("DOCUMENTATION.md has no §17 / Examples Index section")
    section = text[match.end():]
    nxt = re.search(r"^## ", section, re.M)
    if nxt:
        section = section[: nxt.start()]
    rows = [_split_row(line) for line in section.splitlines()
            if line.lstrip().startswith("|")]
    # Header + separator lead the table; data rows have a numeric first cell.
    return section, [(r[0], r[1]) for r in rows
                     if len(r) >= 2 and re.fullmatch(r"\d+", r[0])]


def test_examples_index_lists_exactly_the_scripts_on_disk() -> None:
    """§17 has one row per ``uacpy/examples/example_NN_*.py`` and no orphan
    rows, and its "All N runnable scripts" claim states the real count."""
    if not EXAMPLES_DIR.is_dir():
        pytest.skip("uacpy/examples/ is not present")
    section, rows = _section_17_rows()
    assert rows, "§17 contains no parsable table rows"

    on_disk = {}
    for path in sorted(EXAMPLES_DIR.glob("example_*.py")):
        m = re.match(r"example_(\d+)_", path.name)
        if m:
            on_disk[int(m.group(1))] = path.name
    assert on_disk, f"no example_NN_*.py scripts found under {EXAMPLES_DIR}"

    documented = [int(num) for num, _ in rows]
    problems = []
    dupes = {n for n in documented if documented.count(n) > 1}
    if dupes:
        problems.append(f"§17 lists example number(s) more than once: "
                        f"{sorted(dupes)}")
    for n in sorted(set(documented) - set(on_disk)):
        problems.append(f"§17 row {n:02d} has no matching script on disk")
    for n in sorted(set(on_disk) - set(documented)):
        problems.append(f"{on_disk[n]} has no §17 row")

    claim = re.search(r"All (\d+) runnable scripts", section)
    if claim and int(claim.group(1)) != len(on_disk):
        problems.append(
            f"§17 claims 'All {claim.group(1)} runnable scripts' but "
            f"{len(on_disk)} example_NN_*.py scripts exist"
        )
    assert not problems, "§17 examples-index drift:\n" + "\n".join(problems)


# ── models/README.md run-mode matrix ──────────────────────────────────────

MODELS_README = DOCS_DIR / "models" / "README.md"


def _readme_run_mode_matrix():
    """(header, rows) of the ``## Run modes`` table in models/README.md,
    which the page itself says is generated from ``model.supported_modes``."""
    if not MODELS_README.is_file():
        pytest.skip("docs/models/README.md is not present")
    text = MODELS_README.read_text(encoding="utf-8")
    match = re.search(r"^## Run modes\s*$", text, re.M)
    if not match:
        pytest.fail("models/README.md has no '## Run modes' section")
    section = text[match.end():]
    nxt = re.search(r"^## ", section, re.M)
    if nxt:
        section = section[: nxt.start()]
    table = [line for line in section.splitlines()
             if line.lstrip().startswith("|")]
    if len(table) < 3:
        pytest.fail("the '## Run modes' section has no parsable table")
    return _split_row(table[0]), [_split_row(r) for r in table[2:]]


def _readme_matrix_supported_modes() -> dict:
    """Mode-name sets per README column. The OASES column documents the
    union across its instantiable sub-models, as the page states.

    Built from :func:`_model_classes`, so a wrapper added to
    ``uacpy.models.__all__`` and left out of the README fails
    ``test_readme_run_mode_matrix_matches_model_specs`` below rather than
    passing unnoticed."""
    from uacpy.models.oases import OASES
    out = {}
    oases_modes = set()
    for name, cls in _model_classes().items():
        modes = {m.name for m in cls.spec.modes}
        if issubclass(cls, OASES):
            oases_modes |= modes
        else:
            out[name] = modes
    out["OASES"] = oases_modes
    return out


@requires_docs
def test_readme_run_mode_matrix_matches_model_specs() -> None:
    """Every ✅/✗ in the README run-mode matrix agrees with the models'
    ``ModelSpec.modes`` (the source ``supported_modes`` is built from): a ✅
    cell's mode(s) are all supported, a ✗ cell's are all absent — and every
    ``RunMode`` member appears in some row, so a new mode cannot ship
    undocumented."""
    from uacpy.models import RunMode

    header, rows = _readme_run_mode_matrix()
    supported = _readme_matrix_supported_modes()
    columns = [(_matrix_model_name(cell), i)
               for i, cell in enumerate(header) if _matrix_model_name(cell)]
    for name, _ in columns:
        if name not in supported:
            pytest.fail(f"README run-mode matrix has an unknown model "
                        f"column {name!r}")
    # Both directions: a column naming a model that no longer exists fails
    # above, and a model with no column fails here. Without the second half a
    # newly-added wrapper ships with no README row and a green suite, because
    # it simply never enters the comparison.
    uncolumned = sorted(set(supported) - {name for name, _ in columns})
    if uncolumned:
        pytest.fail(f"model(s) {uncolumned} have no column in the README "
                    f"run-mode matrix (docs/DEV.md section 3, step 7)")

    problems = []
    covered = set()
    for row in rows:
        modes = {m.name for m in RunMode
                 if re.search(rf"\b{m.name}\b", row[0])}
        if not modes:
            problems.append(f"row {row[0]!r} names no RunMode member")
            continue
        covered |= modes
        for name, ci in columns:
            cell = row[ci].strip()
            if cell.startswith("✅"):
                missing = modes - supported[name]
                if missing:
                    problems.append(
                        f"{row[0]!r} × {name}: ✅ but spec lacks "
                        f"{sorted(missing)}")
            elif cell.startswith("✗"):
                extra = modes & supported[name]
                if extra:
                    problems.append(
                        f"{row[0]!r} × {name}: ✗ but spec supports "
                        f"{sorted(extra)}")
            else:
                problems.append(f"{row[0]!r} × {name}: unreadable cell "
                                f"{cell!r}")
    undocumented = {m.name for m in RunMode} - covered
    if undocumented:
        problems.append(f"RunMode member(s) absent from the matrix: "
                        f"{sorted(undocumented)}")
    assert not problems, ("README run-mode matrix drift:\n"
                          + "\n".join(problems))


# ── worked example ↔ figure script containment ────────────────────────────

#: Model pages promising "the code below is that code" about their figure
#: script (docs/models/README.md makes the same claim for every page).
#: oases.md is exempt: it walks its six sub-models section by section and
#: carries no single worked-example block.
FIGURE_SCRIPT_PAGES = {
    "bellhop.md": "bellhop.py",
    "bounce.md": "bounce.py",
    "kraken.md": "kraken.py",
    "ram.md": "ram.py",
    "scooter.md": "scooter.py",
    "sparc.md": "sparc.py",
}


def _strip_inline_comment(line: str) -> str:
    """``line`` without any trailing ``#`` comment (quote-aware), stripped."""
    out, quote = [], None
    for ch in line:
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "#":
            break
        out.append(ch)
    return "".join(out).strip()


def _code_line_set(text: str) -> set:
    """Every non-blank line of ``text``, comment-stripped and whitespace-
    normalized — the unit the containment gate compares at."""
    return {s for s in map(_strip_inline_comment, text.splitlines()) if s}


def _worked_example_section(page_text: str, page: str) -> str:
    match = re.search(r"^## .*Worked example.*$", page_text, re.M)
    if not match:
        pytest.fail(f"{page} has no Worked example section")
    section = page_text[match.end():]
    nxt = re.search(r"^## ", section, re.M)
    return section[: nxt.start()] if nxt else section


@requires_docs
def test_worked_examples_are_drawn_from_the_figure_scripts() -> None:
    """Each model page's worked-example code is verbatim figure-script code.

    The six pages above (and ``models/README.md``) promise that the worked
    example *is* the page's figure code, so this gate makes drift between
    them fail the suite: every code line of every ```python block in the
    page's Worked-example section must appear verbatim in the named
    ``docs/figure_scripts/`` module or in ``_common.py`` (the scripts import
    their shared scenarios from there; the pages inline them).

    Comparison is per line, whitespace-normalized, with trailing ``#``
    comments stripped on both sides. Three doc-side line kinds are exempt,
    because the pages legitimately condense the scripts there: ``import`` /
    ``from`` lines (scripts also import matplotlib and ``_common``), pure
    comments, and ``...`` elision markers.
    """
    common = _code_line_set(
        (DOCS_DIR / "figure_scripts" / "_common.py").read_text(
            encoding="utf-8"))

    problems = []
    for page, script in FIGURE_SCRIPT_PAGES.items():
        page_path = DOCS_DIR / "models" / page
        script_path = DOCS_DIR / "figure_scripts" / script
        page_text = page_path.read_text(encoding="utf-8")
        if "is that code" not in " ".join(page_text.split()):
            problems.append(
                f"{page}: the 'the code below is that code' claim is gone — "
                f"if the page no longer promises containment, update "
                f"FIGURE_SCRIPT_PAGES")
            continue
        corpus = _code_line_set(
            script_path.read_text(encoding="utf-8")) | common
        section = _worked_example_section(page_text, page)
        blocks = re.findall(r"```python\n(.*?)```", section, re.S)
        checked = 0
        for bi, block in enumerate(blocks):
            for line in block.splitlines():
                s = _strip_inline_comment(line)
                if not s or s.startswith(("import ", "from ", "...")):
                    continue
                checked += 1
                if s not in corpus:
                    problems.append(
                        f"{page} (block {bi}): not in {script} or "
                        f"_common.py: {s!r}")
        # Structural self-check, as for §18: silence must mean "every line
        # agreed", never "the parser matched nothing".
        if len(blocks) < 4 or checked < 30:
            problems.append(
                f"{page}: only {len(blocks)} python block(s) / {checked} "
                f"checked line(s) parsed from the Worked-example section — "
                f"the page structure has changed and the gate needs updating")
    assert not problems, ("worked-example ↔ figure-script drift:\n"
                          + "\n".join(problems))


# ── guide page ↔ figure script containment ────────────────────────────────

#: Guide pages promising their snippets are verbatim figure-script code
#: ("the snippets above are that code"). The other guide pages mix figure
#: code with condensed API illustrations, and their prose says so instead
#: of promising containment.
GUIDE_FIGURE_SCRIPT_PAGES = {
    "plotting.md": "plotting.py",
}

_GUIDE_CLAIM_TOKENS = ("is that code", "are that code")

# A signature display's first line — ``f(a, *, b=1, ...)`` — the same shape
# ``docs/check_structure.py`` exempts from parsing: reference material, not
# figure code.
_SIGNATURE_DISPLAY = re.compile(r"^\s*\w+\([^)]*\*")


@requires_docs
def test_guide_page_snippets_are_drawn_from_the_figure_scripts() -> None:
    """Every ```python block on a guide page that promises containment is
    verbatim figure-script code, page-wide.

    Same per-line comparison and doc-side exemptions as the model-page gate
    (imports, pure comments, ``...`` elisions), plus two block-level
    exemptions for reference material that is not figure code: REPL
    transcripts (any ``>>>`` line) and signature displays.
    """
    common = _code_line_set(
        (DOCS_DIR / "figure_scripts" / "_common.py").read_text(
            encoding="utf-8"))

    problems = []
    for page, script in GUIDE_FIGURE_SCRIPT_PAGES.items():
        page_text = (DOCS_DIR / "guide" / page).read_text(encoding="utf-8")
        if not any(tok in " ".join(page_text.split())
                   for tok in _GUIDE_CLAIM_TOKENS):
            problems.append(
                f"{page}: the containment claim is gone — if the page no "
                f"longer promises its snippets verbatim, update "
                f"GUIDE_FIGURE_SCRIPT_PAGES")
            continue
        corpus = _code_line_set(
            (DOCS_DIR / "figure_scripts" / script).read_text(
                encoding="utf-8")) | common
        blocks = re.findall(r"```python\n(.*?)```", page_text, re.S)
        checked = 0
        for bi, block in enumerate(blocks):
            lines = block.splitlines()
            if any(ln.lstrip().startswith(">>>") for ln in lines):
                continue
            first = next((ln for ln in lines if ln.strip()), "")
            if _SIGNATURE_DISPLAY.match(first):
                continue
            for line in lines:
                s = _strip_inline_comment(line)
                if not s or s.startswith(("import ", "from ", "...")):
                    continue
                checked += 1
                if s not in corpus:
                    problems.append(
                        f"{page} (block {bi}): not in {script} or "
                        f"_common.py: {s!r}")
        # Structural self-check, as for the model pages: silence must mean
        # "every line agreed", never "the parser matched nothing".
        if len(blocks) < 4 or checked < 30:
            problems.append(
                f"{page}: only {len(blocks)} python block(s) / {checked} "
                f"checked line(s) parsed — the page structure has changed "
                f"and the gate needs updating")
    assert not problems, ("guide snippet ↔ figure-script drift:\n"
                          + "\n".join(problems))


@requires_docs
def test_guide_pages_state_containment_only_when_gated() -> None:
    """A guide page claims "is/are that code" exactly when the containment
    gate covers it, so a page cannot promise verbatim snippets the suite
    does not hold it to."""
    problems = []
    for path in sorted((DOCS_DIR / "guide").glob("*.md")):
        text = " ".join(path.read_text(encoding="utf-8").split())
        claims = any(tok in text for tok in _GUIDE_CLAIM_TOKENS)
        gated = path.name in GUIDE_FIGURE_SCRIPT_PAGES
        if claims and not gated:
            problems.append(
                f"{path.name}: promises verbatim snippets but is not in "
                f"GUIDE_FIGURE_SCRIPT_PAGES")
        elif gated and not claims:
            problems.append(
                f"{path.name}: gated but no longer states the claim — "
                f"remove it from GUIDE_FIGURE_SCRIPT_PAGES or restore the "
                f"claim")
    assert not problems, ("guide containment-claim drift:\n"
                          + "\n".join(problems))


# ═══════════════════════════════════════════════════════════════════════════
# Per-page constructor tables ↔ signatures
# ═══════════════════════════════════════════════════════════════════════════

#: Model page -> the classes its ``| Name | Default | … |`` tables document.
#: ``oases.md`` walks six sub-models, each table introduced by a ``**`OAST`**``
#: marker; the scope marker narrows a table to one of them.
_PAGE_CLASSES = {
    "bellhop.md": ("Bellhop",),
    "bounce.md": ("Bounce",),
    "kraken.md": ("Kraken",),
    "ram.md": ("RAM",),
    "scooter.md": ("Scooter",),
    "sparc.md": ("SPARC",),
    "oases.md": ("OAST", "OASN", "OASR", "OASP", "OASS", "OASSP"),
}

_SCOPE_MARKER = re.compile(r"^\*\*`(\w+)`\*\*$")
_TABLE_RULE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


def _pipe_tables(text: str):
    """``(line_no, scope, header, rows)`` for each pipe table in ``text``.

    ``scope`` is the class named by the nearest preceding ``**`Name`**`` line,
    or None.
    """
    known = {n for names in _PAGE_CLASSES.values() for n in names}
    lines = text.splitlines()
    scope = None
    i = 0
    while i < len(lines):
        marker = _SCOPE_MARKER.match(lines[i].strip())
        if marker and marker.group(1) in known:
            scope = marker.group(1)
        if (lines[i].lstrip().startswith("|") and i + 1 < len(lines)
                and _TABLE_RULE.match(lines[i + 1])):
            header = _split_row(lines[i])
            rows = []
            j = i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append(_split_row(lines[j]))
                j += 1
            yield i + 1, scope, header, rows
            i = j
        else:
            i += 1


@requires_docs
def test_model_page_constructor_defaults_match_signatures() -> None:
    """Every backticked literal in a model page's Default column equals the
    live constructor default, and every documented name is a real parameter.

    The same claim §18 makes, on the pages a reader actually lands on. Two
    shapes §18 never had to parse appear here: a table scoped to one class by
    a preceding ``**`OAST`**`` marker, and a row naming several parameters
    against a matching list of defaults (```freq_min`, `freq_max``` |
    ```0.0`, `None```).

    Cells that are prose rather than a literal are skipped, as in §18. A cell
    that states the *effective* value where the signature default is a ``None``
    sentinel fails: on OAST/OASR passing the effective value explicitly
    conflicts with ``options=``, and on RAM naming ``accuracy`` promotes a log
    line to a warning — so the two are not interchangeable.
    """
    problems = []
    checked = 0
    for page, page_classes in _PAGE_CLASSES.items():
        path = DOCS_DIR / "models" / page
        if not path.is_file():
            problems.append(f"{page} is missing")
            continue
        for line, scope, header, rows in _pipe_tables(
                path.read_text(encoding="utf-8")):
            low = [h.lower() for h in header]
            if not low or low[0] not in ("name", "knob", "parameter",
                                         "argument"):
                continue
            if "default" not in low:
                continue
            ci_default = low.index("default")
            targets = (scope,) if scope else page_classes
            for row in rows:
                if len(row) <= ci_default:
                    continue
                names = [n.strip() for n in re.findall(r"`([^`]+)`", row[0])]
                cells = re.findall(r"`([^`]+)`", row[ci_default])
                if not names or not cells:
                    continue
                if len(cells) == 1:
                    pairs = [(n, cells[0]) for n in names]
                elif len(cells) == len(names):
                    pairs = list(zip(names, cells))
                else:
                    continue
                for param, token in pairs:
                    kind, documented = _doc_default(f"`{token}`")
                    if kind != "literal":
                        continue
                    owners = [
                        (name, inspect.signature(
                            getattr(uacpy.models, name).__init__))
                        for name in targets
                    ]
                    present = [(n, s) for n, s in owners
                               if param in s.parameters]
                    if not present:
                        problems.append(
                            f"{page}:{line}: `{param}` is documented but is "
                            f"not a constructor parameter of "
                            f"{list(targets)}")
                        continue
                    for name, sig in present:
                        checked += 1
                        problems += [
                            f"{page}:{line}: {p}" for p in
                            _check_documented_default(
                                name, sig, param, f"`{token}`")
                        ]
    # Structural self-check, as for §18.
    assert checked > 100, (
        f"the model-page parser only checked {checked} (class, parameter) "
        f"pairs — the page structure has changed and the gate needs updating"
    )
    assert not problems, ("model-page default drift:\n" + "\n".join(problems))


# ═══════════════════════════════════════════════════════════════════════════
# Documented samples and example scripts ↔ the live API
# ═══════════════════════════════════════════════════════════════════════════

#: ``docs/superpowers/`` holds dated implementation plans and design specs —
#: records of what was proposed on a given day, not claims about today's API.
_API_SCAN_SKIP_DIRS = ("superpowers",)


def _uacpy_bindings(tree: ast.AST) -> dict:
    """``local name -> dotted uacpy path`` for every uacpy import in ``tree``."""
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "uacpy" or alias.name.startswith("uacpy."):
                    out[alias.asname or alias.name.split(".")[0]] = (
                        alias.name if alias.asname
                        else alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level or not (module == "uacpy"
                                  or module.startswith("uacpy.")):
                continue
            for alias in node.names:
                if alias.name != "*":
                    out[alias.asname or alias.name] = f"{module}.{alias.name}"
    return out


def _resolve_dotted(dotted: str):
    """``(object, None)`` or ``(None, first unresolvable prefix)``."""
    parts = dotted.split(".")
    obj = importlib.import_module(parts[0])
    seen = parts[0]
    for part in parts[1:]:
        seen += "." + part
        nxt = getattr(obj, part, None)
        if nxt is None:
            try:
                nxt = importlib.import_module(seen)
            except ImportError:
                return None, seen
        obj = nxt
    return obj, None


def _attribute_chain(node: ast.AST):
    """``Name.a.b`` -> ``['Name', 'a', 'b']``; None for anything else."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return list(reversed(parts))


def _method_on_fresh_instance(call: ast.Call, bindings: dict):
    """``(display chain, dotted path)`` for ``Class(...).method(...)``.

    Returns None unless the receiver is a direct construction of a uacpy class
    — a factory or classmethod call returns something this walker cannot name.
    """
    func = call.func
    if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Call)):
        return None
    ctor = _attribute_chain(func.value.func)
    if not ctor or ctor[0] not in bindings:
        return None
    dotted = ".".join([bindings[ctor[0]]] + ctor[1:])
    obj, _ = _resolve_dotted(dotted)
    if obj is None or not inspect.isclass(obj):
        return None
    return [".".join(ctor) + "()", func.attr], f"{dotted}.{func.attr}"


def _constructed_instances(tree: ast.AST, bindings: dict) -> dict:
    """``variable -> dotted class path`` for names assigned exactly once from
    a direct class construction. A classmethod or factory call returns
    something this walker cannot name, so only real classes are recorded."""
    counts, candidates = {}, {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            counts[target.id] = counts.get(target.id, 0) + 1
            if not isinstance(node.value, ast.Call):
                continue
            chain = _attribute_chain(node.value.func)
            if not chain or chain[0] not in bindings:
                continue
            dotted = ".".join([bindings[chain[0]]] + chain[1:])
            obj, missing = _resolve_dotted(dotted)
            if obj is not None and inspect.isclass(obj):
                candidates[target.id] = dotted
    return {k: v for k, v in candidates.items() if counts.get(k) == 1}


def _api_problems(label: str, tree: ast.AST, inherited: dict = None) -> list:
    """Names and keyword arguments in ``tree`` that the live API cannot take.

    ``inherited`` carries the imports the rest of the page already made: a
    doc page imports once near the top and the later blocks use the names
    without repeating the import.
    """
    bindings = dict(inherited or {})
    bindings.update(_uacpy_bindings(tree))
    if not bindings:
        return []
    instances = _constructed_instances(tree, bindings)
    found, seen = [], set()

    def note(detail):
        if detail not in seen:
            seen.add(detail)
            found.append(f"{label}: {detail}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            chain = _attribute_chain(node)
            if chain and len(chain) > 1 and chain[0] in bindings:
                dotted = ".".join([bindings[chain[0]]] + chain[1:])
                if _resolve_dotted(dotted)[0] is None:
                    note(f"{'.'.join(chain)} does not resolve "
                         f"({_resolve_dotted(dotted)[1]})")
        if not isinstance(node, ast.Call):
            continue
        chain = _attribute_chain(node.func)
        if chain is None:
            # ``Bellhop().supports_mode(...)`` — the receiver is a fresh
            # instance rather than a name, so read the method off the class
            # the constructor call names.
            inner = _method_on_fresh_instance(node, bindings)
            if inner is None:
                continue
            chain, dotted = inner
        elif chain[0] in bindings:
            dotted = ".".join([bindings[chain[0]]] + chain[1:])
        elif chain[0] in instances and len(chain) == 2:
            # One hop only: ``model.run(...)`` names a method of the class,
            # while a deeper chain walks through instance attributes that are
            # invisible on the class object.
            dotted = ".".join([instances[chain[0]]] + chain[1:])
        else:
            continue
        obj, missing = _resolve_dotted(dotted)
        if obj is None:
            note(f"{'.'.join(chain)} does not resolve ({missing})")
            continue
        if not (inspect.isclass(obj) or inspect.isfunction(obj)
                or inspect.ismethod(obj)):
            continue
        try:
            sig = inspect.signature(
                obj.__init__ if inspect.isclass(obj) else obj)
        except (TypeError, ValueError):
            continue
        params = dict(sig.parameters)
        first = next(iter(params), None)
        if first in ("self", "cls"):
            params.pop(first)
        if any(p.kind is inspect.Parameter.VAR_KEYWORD
               for p in params.values()):
            continue
        accepted = {n for n, p in params.items()
                    if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                  inspect.Parameter.KEYWORD_ONLY)}
        for keyword in node.keywords:
            if keyword.arg is not None and keyword.arg not in accepted:
                note(f"{'.'.join(chain)}({keyword.arg}=…) is not a parameter; "
                     f"it takes {sorted(accepted)}")
    return found


def _page_code_blocks(path: Path):
    """``(first line number, tree)`` for each python block in a markdown page."""
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        fence = re.match(r"^\s*```(\w*)\s*$", lines[i])
        if not fence:
            i += 1
            continue
        start = i + 1
        j = start
        while j < len(lines) and not re.match(r"^\s*```\s*$", lines[j]):
            j += 1
        body, i = lines[start:j], j + 1
        if fence.group(1).lower() not in ("python", "py", "pycon"):
            continue
        source = "\n".join(body)
        if any(ln.startswith(">>> ") for ln in body):
            source = "\n".join(
                ln[4:] for ln in body
                if ln.startswith(">>> ") or ln.startswith("... "))
        try:
            yield start + 1, ast.parse(source)
        except SyntaxError:
            continue            # already reported by the parse gate above


def _documented_python_units():
    """``(label, tree, page imports)`` for every python sample in the docs and
    every example script — the two places the package shows code it claims a
    user can run. A page's blocks share the imports made anywhere on it, since
    a page imports once near the top and the later blocks just use the names.
    """
    root = Path(uacpy.__file__).resolve().parent.parent
    for directory in ((root / "uacpy" / "examples"),
                      (DOCS_DIR / "figure_scripts")):
        for path in sorted(directory.glob("*.py")):
            yield (str(path.relative_to(root)),
                   ast.parse(path.read_text(encoding="utf-8")), {})

    pages = [p for p in sorted(DOCS_DIR.rglob("*.md"))
             if not any(d in p.parts for d in _API_SCAN_SKIP_DIRS)]
    for extra in ("DOCUMENTATION.md", "README.md"):
        if (root / extra).is_file():
            pages.append(root / extra)
    for path in pages:
        blocks = list(_page_code_blocks(path))
        page_bindings = {}
        for _, tree in blocks:
            page_bindings.update(_uacpy_bindings(tree))
        for line, tree in blocks:
            yield f"{path.relative_to(root)}:{line}", tree, page_bindings


@requires_docs
def test_documented_samples_use_names_the_package_has() -> None:
    """Every ``uacpy.…`` name and keyword argument used by a documented sample
    or an example script exists in the live API.

    The parse gate above proves a sample is syntactically valid; this proves
    it would still bind. A renamed attribute, a dropped keyword or a moved
    submodule turns a documented example into a `AttributeError`/`TypeError`
    that nothing else in the suite notices — the examples' own smoke tests
    only cover ``uacpy/examples/``, not the pages.

    Resolution is deliberately conservative: chains rooted at a uacpy import,
    plus one hop off a variable assigned directly from a uacpy class. Anything
    it cannot name for certain it leaves alone, so a pass means "nothing
    provably broken", never "everything checked".
    """
    problems = []
    units = 0
    for label, tree, page_bindings in _documented_python_units():
        units += 1
        problems += _api_problems(label, tree, page_bindings)
    assert units > 200, (
        f"only {units} code units were discovered — the docs layout has "
        f"changed and the gate needs updating"
    )
    assert not problems, ("documented API drift:\n" + "\n".join(problems))


# ── docstring prose against the signature it is written on ──────────────────
#
# Everything above gates the `docs/` tree and the DOCUMENTATION.md tables, and
# every one of those checks reads a *table*. Nothing above checks a sentence.
#
# The gates below add the one prose-to-code direction that can be checked
# without guessing: a numpydoc `Parameters` entry names a parameter and sits on
# the function whose signature defines it, so the subject of the sentence is
# known exactly rather than inferred. Two claims are compared against that
# signature: every documented parameter name exists in it, and every documented
# default written as a literal equals the real one. Parsing is structural (the
# section found by its underline, entries by their indentation, the default by
# a literal token) and reads no other prose, so rewording a description cannot
# fire either — only renaming a parameter, or changing a default without
# changing the sentence that states it, does.
#
# Scope of the evidence behind them. The survey that chose this measured the
# *defaults* category and found zero drift across 173 markdown sentences and 21
# hand-checked claims. That is not evidence that doc prose is accurate in
# general: the two documented cases of real drift this project has seen — a
# `UserWarning` described after its deadband was removed, and a fixed default
# described after it became derived — were NOT in the defaults category, and
# both lived in the warns/threshold prose that was scanned and then excluded as
# mechanically unresolvable. Those categories remain unmeasured and ungated.
#
# What these deliberately do NOT catch, measured on this tree when they were
# written. The first five are the skip rules the parser applies; each costs
# coverage, and the counts are what they cost:
#
# * Markdown prose — the whole `docs/` tree. A sentence there has no signature
#   attached to it, and the same identifier is a parameter of several unrelated
#   callables with different defaults (`nperseg` is None on the FRF methods and
#   8192 on `psd`; `scaling` is 'density' on the linear estimators and
#   'spectrum' on the constant-Q family; `vmin` differs across three plotters).
#   Resolving the subject from the page is guesswork, and guessing produced
#   false positives on every disagreement it found — prose states non-defaults
#   (`merge=False` "returns the raw records") in the shape it states defaults.
# * Entries heading more than one parameter (`c_low, c_high : float`) — 64
#   entries covering 151 parameters. One stated default cannot have two
#   subjects, and in that very example it belongs to `c_high` alone.
# * `**kwargs` entries — 4 entries holding 27 default mentions, all of them
#   defaults of the keys being forwarded rather than of the parameter.
# * Derived defaults. When the signature default is None the docstring
#   describes what None resolves to, not the literal, so those are skipped:
#   `plot_field`'s `value` "Defaults to 'real' for a time-series field" is
#   correct prose about a None sentinel. A derived default that drifts is
#   invisible here.
# * Descriptions stating two different defaults, and undelimited numbers
#   followed by a unit. The first has no single subject (`fetch_environment`'s
#   `with_absorption` gives pH's 8.1 before its own False); the second states a
#   physical quantity that need not be the stored value (`power_to_db`'s
#   "1 µPa" is REFERENCE_PRESSURE_WATER, 1e-6 Pa). These two rules are what let
#   the claim pattern accept the connective-less "Default 10." that most of
#   this tree writes.
# * "X raises Y" and "X warns". Whether a function raises or warns is a
#   property of everything it calls, not of its own body, so neither is decided
#   by anything these can read.
# * Numbers quoted from a computation or a measurement — the dB figures,
#   tolerances and counts the guide pages quote from their figure scripts.
# * Defaults stated anywhere other than a `Parameters` entry: a module
#   docstring, a `Notes` section, or narrative in the body.
#
# NOTE: these two read `uacpy/` sources, not `docs/`, but the module-level
# `pytestmark` above skips the whole file when `docs/` is absent, so they now
# skip in that case as well. `docs/` is present in every source checkout and
# `uacpy.tests` is excluded from the wheel, so nothing reaches them without it.


PKG_DIR = Path(uacpy.__file__).resolve().parent

# ``name : type`` opening an entry, or ``a, b : type`` for a shared one.
_PARAM_HEAD = re.compile(
    r"^(\*{0,2}[A-Za-z_]\w*(?:\s*,\s*\*{0,2}[A-Za-z_]\w*)*)\s*:\s*")

# "Default is X" / "Defaults to X" / "Default: X" / "Default X", where X is a
# single token. The connective is optional because most of this tree writes
# "Default 10." with none; requiring one removed 86 of the comparisons below.
# The number pattern stops before a sentence's full stop, so "Default is 1e-9."
# yields ``1e-9`` and "(default: 1.4 for air)" yields ``1.4``.
_NUMBER = r"-?\d+(?:_\d+)*(?:\.\d*)?(?:[eE][+-]?\d+)?"
_DELIMITED = r"``[^`]+``|`[^`]+`|\([^)]*\)|'[^']*'|\"[^\"]*\""
# A unit directly after an undelimited number means the sentence states a
# physical quantity, which need not be the stored value: ``power_to_db``'s
# "(default: 1 µPa, water)" is ``REFERENCE_PRESSURE_WATER``, i.e. 1e-6 Pa.
_UNIT = (r"(?:m/s|[µu]?Pa|[kMG]?Hz|dB|km|kg|ms|degrees?|deg|ppt|psu|°?C"
         r"|[ms])\b")
_DEFAULT_CLAIM = re.compile(
    r"\bdefaults?\b\s*(?:\bis\b|\bto\b|[:=])?\s*"
    r"(?:(" + _DELIMITED + r")"
    r"|(" + _NUMBER + r")(?!\.?\d)(?!\s*" + _UNIT + r")"
    r"|(True|False|None)\b)",
    re.I)


def _python_sources():
    """Every shipped ``.py`` file — the test suite itself is not documentation."""
    for path in sorted(PKG_DIR.rglob("*.py")):
        if "tests" not in path.parts:
            yield path


def _documented_units(tree: ast.AST):
    """``(label, docstring, signature_node)`` for each unit whose docstring
    documents parameters. A class documents ``__init__``'s signature."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn = node
        elif isinstance(node, ast.ClassDef):
            fn = next((n for n in node.body
                       if isinstance(n, ast.FunctionDef)
                       and n.name == "__init__"), None)
            if fn is None:
                continue
        else:
            continue
        doc = ast.get_docstring(node)
        if doc:
            yield node.name, doc, fn


def _parameters_section(doc: str):
    """``(names, description)`` for every entry under a numpydoc
    ``Parameters`` heading."""
    lines = doc.expandtabs().splitlines()
    start = next((i + 2 for i, line in enumerate(lines[:-1])
                  if line.strip() == "Parameters"
                  and set(lines[i + 1].strip()) == {"-"}), None)
    if start is None:
        return
    end = len(lines)
    for i in range(start, len(lines) - 1):
        title, rule = lines[i].strip(), lines[i + 1].strip()
        if title and set(rule) == {"-"} and len(rule) >= len(title):
            end = i
            break
    body = [line for line in lines[start:end]]
    indents = [len(l) - len(l.lstrip()) for l in body if l.strip()]
    if not indents:
        return
    base = min(indents)

    names, desc = None, []
    for line in body:
        stripped = line.strip()
        if stripped and len(line) - len(line.lstrip()) == base:
            head = _PARAM_HEAD.match(stripped)
            if head:
                if names:
                    yield names, "\n".join(desc)
                names = [n.strip() for n in head.group(1).split(",")]
                desc = []
                continue
        desc.append(stripped)
    if names:
        yield names, "\n".join(desc)


def _signature_names(fn: ast.AST) -> list:
    args = fn.args
    names = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
    if args.vararg:
        names.append(args.vararg.arg)
    if args.kwarg:
        names.append(args.kwarg.arg)
    return names


_CONSTANT_NAME = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+")


def _package_constants() -> dict:
    """``UPPER_CASE`` module-level name -> value, for names the package binds
    to one literal everywhere. A name assigned different values in different
    modules is dropped: it cannot be resolved from a docstring alone."""
    found = {}
    for path in _python_sources():
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            else:
                continue
            for target in targets:
                if not (isinstance(target, ast.Name)
                        and _CONSTANT_NAME.fullmatch(target.id)):
                    continue
                try:
                    literal = ast.literal_eval(value)
                except (ValueError, SyntaxError, TypeError):
                    continue
                found.setdefault(target.id, []).append(literal)
    return {name: values[0] for name, values in found.items()
            if all(v == values[0] for v in values)}


_CONSTANTS = None


def _constants() -> dict:
    global _CONSTANTS
    if _CONSTANTS is None:
        _CONSTANTS = _package_constants()
    return _CONSTANTS


def _resolve(node: ast.AST):
    """``(resolved, value)`` for a default expression: a literal, or a bare
    name the package binds to one constant."""
    try:
        return True, ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        pass
    if isinstance(node, ast.Name) and node.id in _constants():
        return True, _constants()[node.id]
    return False, None


def _signature_defaults(fn: ast.AST) -> dict:
    """Parameter name -> ``(resolved, value)`` for every defaulted argument."""
    args = fn.args
    positional = args.posonlyargs + args.args
    defaults = dict(zip(positional[len(positional) - len(args.defaults):],
                        args.defaults))
    defaults.update({a: d for a, d in zip(args.kwonlyargs, args.kw_defaults)
                     if d is not None})
    return {arg.arg: _resolve(node) for arg, node in defaults.items()}


def _claimed_default(desc: str):
    """``(found, resolved, value)`` for the default a description states.

    A description stating more than one default has no single subject and is
    not read: ``fetch_environment``'s ``with_absorption`` gives pH's 8.1
    before its own ``False``, and only the second one is about the parameter.
    """
    values = []
    for match in _DEFAULT_CLAIM.finditer(desc):
        token = next(g for g in match.groups() if g)
        token = token.strip().strip("`").strip()
        try:
            values.append((True, ast.literal_eval(token)))
            continue
        except (ValueError, SyntaxError):
            pass
        if token in _constants():
            values.append((True, _constants()[token]))
        else:
            values.append((False, token))

    if not values:
        return False, False, None
    resolved = [value for ok, value in values if ok]
    if not resolved:
        return True, False, values[0][1]
    first = repr(resolved[0])
    if any(repr(value) != first for value in resolved):
        return True, False, None
    return True, True, resolved[0]


def test_every_documented_parameter_exists_in_its_signature() -> None:
    """Every name given a numpydoc ``Parameters`` entry is an argument of the
    function that entry is written on.

    A parameter renamed in the signature but not in the docstring leaves prose
    describing an argument the caller cannot pass. Units taking ``**kwargs``
    forward arguments they do not name, so a documented name they lack is
    allowed there.
    """
    problems = []
    checked = 0
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for label, doc, fn in _documented_units(tree):
            names = _signature_names(fn)
            forwards = fn.args.kwarg is not None
            for entry, _ in _parameters_section(doc):
                for param in entry:
                    param = param.lstrip("*")
                    checked += 1
                    if param in names or forwards:
                        continue
                    problems.append(
                        f"{path.relative_to(PKG_DIR.parent)}:{fn.lineno}: "
                        f"{label} documents {param!r}, which is not in "
                        f"({', '.join(names)})")

    # Silence here must mean "every entry agreed", never "the section parser
    # matched nothing". 1246 entries parsed when this was written.
    assert checked > 1000, (
        f"the Parameters parser only found {checked} entries — the docstring "
        f"format has changed and this gate needs updating"
    )
    assert not problems, (
        "documented parameter(s) absent from the signature:\n"
        + "\n".join(problems)
    )


def test_every_documented_default_equals_the_signature_default() -> None:
    """Every "Default is X" / "Defaults to X" written as a literal in a
    numpydoc ``Parameters`` entry equals that argument's real default.

    Descriptions stating no literal are not read, and a ``None`` default is
    skipped: ``None`` is this package's derive-it sentinel, so the sentence
    describes the derivation rather than the value in the signature.
    """
    problems = []
    compared = 0
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for label, doc, fn in _documented_units(tree):
            defaults = _signature_defaults(fn)
            names = _signature_names(fn)
            for entry, desc in _parameters_section(doc):
                # An entry heading two parameters — ``c_low, c_high : float``
                # — states one default that belongs to only one of them, so
                # the subject is no longer known and the claim is not read.
                # A ``**kwargs`` entry lists the defaults of the keys it
                # forwards, none of which is its own.
                if len(entry) != 1 or entry[0].startswith("*"):
                    continue
                param, = entry
                found, resolved, claimed = _claimed_default(desc)
                if not found or not resolved or param not in names:
                    continue
                if param not in defaults:
                    problems.append(
                        f"{path.relative_to(PKG_DIR.parent)}:{fn.lineno}: "
                        f"{label}.{param} documents default "
                        f"{claimed!r} but the signature has none")
                    continue
                actual_resolved, actual = defaults[param]
                if not actual_resolved or (actual is None
                                           and claimed is not None):
                    continue
                # ``None`` documented against a ``None`` default still catches
                # a default that stops being None, but it compares no value,
                # so the floor below is not allowed to count it.
                if actual is not None or claimed is not None:
                    compared += 1
                if actual != claimed:
                    problems.append(
                        f"{path.relative_to(PKG_DIR.parent)}:{fn.lineno}: "
                        f"{label}.{param} documents default {claimed!r} "
                        f"but the signature default is {actual!r}")

    # Silence here must mean "every stated default agreed", never "the claim
    # parser matched nothing". 180 value-bearing comparisons were available
    # when this was written (216 including the ``None``-to-``None`` pairs this
    # count excludes), against 308 documented parameters whose signature
    # default is a resolvable non-``None`` value.
    assert compared > 150, (
        f"only {compared} documented defaults were compared — the claim "
        f"parser has stopped matching and this gate needs updating"
    )
    assert not problems, (
        "documented default(s) disagreeing with the signature:\n"
        + "\n".join(problems)
    )


# ── constants and docstrings against the source they cite ──────────────────
#
# Two claims that live in code rather than in `docs/`, and that no table gate
# above can see: a constant whose comment records WHY it is what it is, and a
# docstring that has to name the band its guide page documents.



def test_c_high_factor_records_which_models_require_the_pad():
    """The pad cannot be tuned for Kraken alone: Scooter/SPARC use c_high as
    the lower limit of a wavenumber integral, where it keeps the branch point
    k = omega/c_bottom inside the window. Without this note a later round
    'simplifies' the constant and breaks the model users are told to fall back
    to."""
    src = (Path(uacpy.__file__).parent / 'core' / 'constants.py').read_text()
    block = src.split('C_LOW_FACTOR = 0.95')[0].split('C_HIGH_FACTOR')[-1]
    assert 'branch point' in block
    assert 'scooter.f90:67,123' in block
    assert "SPARC._write_sparc_env" in block


def test_thorp_docstring_points_at_its_frequency_band():
    """``help(thorp_db_per_km)`` gave T/S/pH/depth at length and said nothing
    about frequency, while the guide has it —
    ``docs/guide/environment.md §6 "Two things the curve does not tell you"``.
    """
    from uacpy.core.absorption import thorp_db_per_km
    doc = thorp_db_per_km.__doc__
    assert 'docs/guide/environment.md §6 "Two things the curve does not tell you"' in doc
    assert '10 Hz' in doc


_COLLAPSE_GUIDE_LINK = 'docs/guide/environment.md#7-collapse-policy'


@requires_docs
def test_documentation_collapse_section_points_at_the_guide():
    documentation = REPO_ROOT / 'DOCUMENTATION.md'
    guide = REPO_ROOT / 'docs' / 'guide' / 'environment.md'
    if not documentation.is_file() or not guide.is_file():
        pytest.skip('docs are not present in this install')

    text = documentation.read_text(encoding='utf-8')
    section = text.partition('### Environment feature support and collapse')[2]
    assert section, 'DOCUMENTATION.md lost its collapse section'
    section = section.partition('\n### ')[0]
    assert _COLLAPSE_GUIDE_LINK in section
    assert 'ConfigurationError' in section     # the raise half of the policy

    # The anchor the link resolves to (docs/check_links.py gates this too).
    assert '## 7. Collapse policy' in guide.read_text(encoding='utf-8')


# Shared with test_determinism.py: the prose gates read normalised text
# (whitespace collapsed) so re-wrapping a paragraph cannot fire them.
# `_needs_docs` is redundant beside this module's own pytestmark and is
# kept so the gates read the same in both files.
_needs_docs = pytest.mark.skipif(
    not DOCS_DIR.is_dir(),
    reason="the docs tree is not present in an installed layout")


def _normalised(path: Path) -> str:
    """One-line form of a markdown page, so a re-wrap cannot move a phrase."""
    return " ".join(path.read_text(encoding="utf-8").split())


@requires_docs
@_needs_docs
def test_the_bellhop_page_denies_that_arrivals_match_across_backends() -> None:
    """``docs/models/bellhop.md`` says the ``.arr`` merge rule normalises the
    record *set* and not the values in it.

    Measured with ``beam_type='G'``, fortran against cuda: the record count
    matched at 52, and amplitude differed in 52 of 52 (max 2.09e-08), delay in
    39 of 52 (max 1.05e-06 s), receiver angle in 25 of 52 (max 7.28e-05 deg),
    source angle in 8 of 52 and phase in 1 of 52. The page previously closed
    the paragraph by promising the opposite, which is the claim a
    reproducibility statement would rest on.
    """
    text = _normalised(DOCS_DIR / "models" / "bellhop.md")

    assert "is the same whichever backend produced the file" not in text, (
        "the .arr paragraph promises backend-independent Arrivals again — "
        "measured false: the merge rule normalises the record set, not the "
        "float32 values in it"
    )
    for phrase in ("same *set* of records",
                   "does not hold the same",
                   "float32-scale tolerance, never for equality"):
        assert phrase in text, (
            f"the .arr paragraph no longer states {phrase!r} — the page has "
            f"to say what the merge rule does and does not normalise"
        )


@requires_docs
@_needs_docs
def test_the_bellhop_page_states_the_default_backend_is_not_reproducible() -> None:
    """``docs/models/bellhop.md`` says the auto-picked backend gives a
    different field on a second run, and names the one that does not.

    Two default ``Bellhop()`` runs (both auto-picked to ``'cuda'``) wrote a
    byte-identical ``model.env`` and ``model.prt`` but a differing
    ``model.shd``: TL differed in 57 of 500 elements, by up to 1.53e-05 dB.
    ``backend='fortran'`` is the reproducible choice for the ``.shd`` only —
    its ``.prt`` carries a CPU-time stamp — so the page must not promise
    byte-identical *output* for it.
    """
    text = _normalised(DOCS_DIR / "models" / "bellhop.md")

    for phrase in ("The default backend is not run-to-run reproducible",
                   "1.53e-05 dB",
                   "`Bellhop(backend='fortran')`",
                   "`CPU Time"):
        assert phrase in text, (
            f"the backend-reproducibility gotcha no longer states {phrase!r}"
        )
    gotcha = text.split("The default backend is not run-to-run reproducible")[1]
    gotcha = gotcha.split("## 8. References")[0]
    assert "`.shd` is byte-identical" in gotcha, (
        "the gotcha has to scope the fortran promise to the .shd: its .prt is "
        "not byte-identical"
    )


@requires_docs
@_needs_docs
def test_the_reference_states_what_run_with_bounce_derives_c_low_from() -> None:
    """``DOCUMENTATION.md``'s ``run_with_bounce`` section states the rule that
    resolves ``c_low=None``, and the speed it quotes is ``DEFAULT_C_MIN``.

    The derived value is user-visible: ``Bounce`` resolves it as
    ``min(DEFAULT_C_MIN, min(env.ssp))`` — bounce.htm's "lowest speed in the
    problem" — so a cold layer or a fresh surface layer tabulates a different
    reflection table than a rule reading the seafloor water speed alone
    (measured: 1300 and 1320 m/s against 1400 over six profiles, with the four
    ordinary-water decks byte-identical either way). The section showed only an
    explicit ``c_low=1400.0`` call and never said what ``None`` resolves to.

    The quoted speed is compared against the constant rather than read as
    prose, so moving ``DEFAULT_C_MIN`` and leaving the sentence behind fails
    here.
    """
    from uacpy.core.constants import DEFAULT_C_MIN

    text = _normalised(REPO_ROOT / "DOCUMENTATION.md")
    section = text.split("run_with_bounce")[-1].split("### Kraken")[0]

    quoted = re.search(r"`min\((\d+(?:\.\d*)?), min\(env\.ssp\)\)`", section)
    assert quoted is not None, (
        "the run_with_bounce section no longer states how c_low=None is "
        "resolved — it is a derived value that differs from the explicit "
        "c_low=1400.0 the section demonstrates"
    )
    assert float(quoted.group(1)) == float(DEFAULT_C_MIN), (
        f"the section quotes min({quoted.group(1)}, min(env.ssp)) but "
        f"DEFAULT_C_MIN is {DEFAULT_C_MIN}"
    )
    assert "below 1400 m/s" in section, (
        "the section states the rule but not its consequence: in cold or "
        "brackish water the derived c_low falls below the reference speed"
    )


@requires_docs
@_needs_docs
def test_the_arrays_page_states_the_blas_threading_caveat() -> None:
    """``docs/guide/arrays.md`` names the two environment variables that
    change its results and says what changing them costs.

    OpenBLAS partitions a large GEMM by thread count and the summation order
    follows, so the spectra move by 1e-16 to 1e-13 relative between one thread
    and eight. Every discrete output measured was unchanged — DOA peak
    bearings at both problem sizes, and the JANUS detection index in 750 of
    750 real detections — so the page has to give the reassurance alongside
    the caveat, or a reader concludes their bearings are unreliable.
    """
    text = _normalised(DOCS_DIR / "guide" / "arrays.md")

    for phrase in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                   "1e-16 to 1e-13 relative",
                   "bit-identity problem, not a correctness one",
                   "750 of 750"):
        assert phrase in text, (
            f"the BLAS-threading gotcha no longer states {phrase!r}"
        )


def _check_links():
    path = REPO_ROOT / "docs" / "check_links.py"
    if not path.is_file():
        pytest.skip(f"{path} is missing")
    spec = importlib.util.spec_from_file_location("_r21_check_links", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@requires_docs
def test_the_link_checker_reads_links_and_not_code_samples(tmp_path):
    """A fenced block is sample text: a page showing a command's output path
    was reported as a broken link, which is the doc gate crying wolf."""
    checker = _check_links()
    page = tmp_path / "page.md"
    page.write_text(
        "# Probe\n"
        "\n"
        "[outside](gone.md)\n"
        "\n"
        "```bash\n"
        "See [report](out/run.md) and ![fig](out/plot.png)\n"
        "```\n",
        encoding="utf-8",
    )
    assert [t for _, t in checker.broken_links(page)] == ["gone.md"]
    assert [t for _, t in checker.link_targets(page.read_text())] == ["gone.md"]


@requires_docs
@pytest.mark.skipif(not (DOCS_DIR / 'guide' / 'io.md').is_file(),
                    reason='docs/ is not present (source checkout only)')
class TestIoGuidePageMatchesTheExportSurface:

    def test_header_public_name_count_equals_len_of_all(self):
        import uacpy.io
        text = (DOCS_DIR / 'guide' / 'io.md').read_text()
        m = re.search(r'`uacpy\.io` · (\d+) public names', text)
        assert m is not None
        assert int(m.group(1)) == len(uacpy.io.__all__)

    def test_reference_section_lists_every_non_submodule_name_in_all(self):
        import uacpy.io
        text = (DOCS_DIR / 'guide' / 'io.md').read_text()
        assert '## 9. Reference' in text and '## 10.' in text
        section = text.split('## 9. Reference')[1].split('## 10.')[0]
        listed = set(re.findall(r'^\| `([A-Za-z_]\w*)[`(]', section, re.M))
        expected = {n for n in uacpy.io.__all__
                    if not inspect.ismodule(getattr(uacpy.io, n))}
        assert listed == expected


_SYNTHETIC_DOC = """# A document

## 1. First

Some prose about the first thing.

```python
# 2. This is a comment inside a fence, not a heading
model = Thing()
```

Prose after the fence, still inside section 1.

## 2. Second

The claim the citation leans on.

### 2.1 A numbered subsection

A deeper claim.

## 3. Third

Something else entirely.
"""


class TestSectionSpan:

    @staticmethod
    def _body():
        return _SYNTHETIC_DOC.splitlines()

    def test_a_section_runs_to_the_next_heading_of_its_own_level(self):
        body = self._body()
        first, last = _section_span(body, '2')
        assert body[first] == '## 2. Second'
        assert body[last + 1] == '## 3. Third'
        # §2 contains its numbered subsection.
        assert '### 2.1 A numbered subsection' in body[first:last + 1]

    def test_a_numbered_subsection_is_addressable_on_its_own(self):
        body = self._body()
        first, last = _section_span(body, '2.1')
        assert body[first] == '### 2.1 A numbered subsection'
        assert 'A deeper claim.' in body[first:last + 1]
        assert 'The claim the citation leans on.' not in body[first:last + 1]

    def test_a_hash_comment_inside_a_fence_does_not_end_a_section(self):
        # DOCUMENTATION.md is full of Python examples whose comments open with
        # ``#``. Read as a level-1 heading, one of them ended §7 at line 762
        # instead of 1058 and put the phrase cited from line 857 "outside" the
        # section that contains it.
        body = self._body()
        first, last = _section_span(body, '1')
        assert 'Prose after the fence, still inside section 1.' in \
            body[first:last + 1]

    def test_an_absent_section_number_resolves_to_nothing(self):
        assert _section_span(self._body(), '9') is None
        assert _section_span(self._body(), '2.7') is None


class TestDocAnchorPattern:

    def test_it_reads_the_file_the_section_and_the_quote(self):
        match = _DOC_ANCHOR.search(
            'see kraken.md §6.5 "asking for leaky ones" for the count')
        assert match.groups() == ('kraken.md', '6.5', 'asking for leaky ones')

    def test_a_bare_section_reference_is_read_without_a_quote(self):
        match = _DOC_ANCHOR.search('as ram.md §5 explains')
        assert match.groups() == ('ram.md', '5', None)

    def test_a_quote_that_wrapped_leaves_a_bare_section_reference(self):
        # The known blind spot, stated here rather than left to be discovered:
        # the anchor still resolves, the quote is simply not read. The quoted
        # -anchor floor in the gate is what notices this happening in bulk.
        match = _DOC_ANCHOR.search('kraken.md §7 "a phrase that runs on')
        assert match.group(3) is None


# The fixture addresses below are assembled at runtime rather than written
# out. Both gates read this file like any other, and a literal example of the
# form they reject is indistinguishable from the real thing: spelled out, the
# three line pins were flagged as offences and the two anchors as citations
# into a document that does not exist. Split, the patterns see nothing here
# and the fixtures still exercise the shape they are about.
_SAMPLE_DOC = 'round26-sample' + '.md'


class TestOwnDocLinePinPattern:

    @pytest.mark.parametrize('address', ['313', '316-318', '998'])
    def test_it_finds_a_line_pin(self, address):
        assert _OWN_DOC_LINE_PIN.search(
            f'see docs/models/{_SAMPLE_DOC}:{address} for the rest') is not None

    @pytest.mark.parametrize('tail', [' §6.5 "a quote"', ' §5',
                                      ' alone'])
    def test_it_leaves_a_section_anchor_alone(self, tail):
        assert _OWN_DOC_LINE_PIN.search(_SAMPLE_DOC + tail) is None


@requires_docs
def test_the_gate_reads_a_useful_number_of_anchors_from_the_real_tree():
    """An anti-vacuity check on the gate's own floors.

    The floors in ``test_packaging.py`` say "at least this much"; this says
    the real tree is comfortably above them, so a floor raised by a later
    round has a measured number to be raised against rather than a guess.
    """
    package = Path(REPO_ROOT) / 'uacpy'
    by_name = _repo_markdown()
    resolved = quoted = 0
    for path in package.rglob('*.py'):
        if 'third_party' in path.parts:
            continue
        for line in path.read_text(encoding='utf-8').splitlines():
            for match in _DOC_ANCHOR.finditer(line):
                cited, section, quote = match.groups()
                target = _resolve_markdown(by_name, cited)
                if target is None:
                    continue
                body = target.read_text(encoding='utf-8').splitlines()
                if _section_span(body, section) is None:
                    continue
                resolved += 1
                quoted += quote is not None
    assert resolved >= _MIN_DOC_ANCHORS_RESOLVED
    assert quoted >= _MIN_QUOTED_DOC_ANCHORS


class TestHeadingWalkerSkipsWhatOnlyLooksLikeAHeading:
    """Three shapes that put a ``#`` at the start of a line without meaning a
    heading. None occurs in this repo's Markdown today, which is exactly why
    they need pinning here rather than being left to the real tree."""

    def test_a_four_backtick_fence_is_not_closed_by_an_inner_three(self):
        body = [
            '## 1. First',
            '````markdown',
            '```',
            '# 9. Not a heading, it is inside the outer fence',
            '```',
            '````',
            'Still section 1.',
            '## 2. Second',
        ]
        numbers = [n for _, _, n in _headings(body)]
        assert numbers == ['1', '2']
        first, last = _section_span(body, '1')
        assert 'Still section 1.' in body[first:last + 1]

    def test_a_hash_inside_an_html_comment_is_not_a_heading(self):
        body = [
            '## 1. First',
            '<!--',
            '## 9. A heading someone commented out',
            '-->',
            'Still section 1.',
            '## 2. Second',
        ]
        numbers = [n for _, _, n in _headings(body)]
        assert numbers == ['1', '2']
        first, last = _section_span(body, '1')
        assert 'Still section 1.' in body[first:last + 1]

    def test_a_tilde_fence_is_not_closed_by_a_backtick_fence(self):
        # CommonMark, and the reason an unbalanced fence is a disclosed blind
        # spot rather than a bug: the ``~~~`` block runs to the end of file,
        # so §2 is invisible and a citation to it fails loudly with "no
        # heading numbered §2" instead of resolving to the wrong lines.
        body = [
            '## 1. First',
            '~~~',
            '# Inside the tilde fence',
            '```',
            '## 2. Second',
        ]
        numbers = [n for _, _, n in _headings(body)]
        assert numbers == ['1']
        assert _section_span(body, '2') is None


class TestRepoMarkdownIsDocumentationOnly:

    def test_the_walk_applies_its_own_exclusion_set(self):
        # Deliberately relative to ``_NOT_DOCUMENTATION`` rather than to a
        # list of directory names: this pins that the filter runs at all, and
        # the test below pins what belongs in it. Asserting both here would
        # be one tautology wearing two names.
        found = [p for paths in _repo_markdown().values() for p in paths]
        assert len(found) > 20, "the Markdown walk has stopped finding files"
        for path in found:
            assert not set(path.parts) & _NOT_DOCUMENTATION, path

    def test_the_pytest_cache_readme_is_excluded_by_name(self):
        # ``.pytest_cache/README.md`` is a tool artefact that ships with every
        # run, and it was one of the copies that made ``README.md`` resolve to
        # nothing at all.
        assert '.pytest_cache' in _NOT_DOCUMENTATION
        readmes = _repo_markdown().get('README.md', ())
        assert readmes, "no README.md found — the walk is broken"
        assert not any('.pytest_cache' in p.parts for p in readmes)


class TestLinePinVerdict:
    """A line pin is forbidden on its own terms, not on whether it resolves."""

    @staticmethod
    def _by_name():
        return _repo_markdown()

    def test_an_ambiguous_basename_is_an_offence_not_a_pass(self):
        # The hole this replaced: ``README.md`` names four documents here, so
        # it resolves to none of them, and the earlier draft read that as
        # "not ours" and waved the pin through.
        assert _resolve_markdown(self._by_name(), 'README.md') is None
        verdict = _line_pin_verdict(self._by_name(), False, 'README.md')
        assert verdict is not None
        assert 'no single file' in verdict

    def test_a_resolvable_own_doc_is_an_offence(self):
        verdict = _line_pin_verdict(
            self._by_name(), False, 'docs/models/kraken.md')
        assert verdict == 'docs/models/kraken.md'

    def test_vendored_markdown_keeps_its_line_numbers(self):
        assert _line_pin_verdict(
            self._by_name(), False,
            'bellhopcuda/doc/accuracy.md') is None

    def test_an_external_marked_address_is_let_through(self):
        assert _line_pin_verdict(
            self._by_name(), True, 'README.md') is None
        assert _line_pin_verdict(
            self._by_name(), True, 'docs/models/kraken.md') is None

    def test_the_pattern_captures_the_external_marker_apart_from_the_path(self):
        match = _OWN_DOC_LINE_PIN.search(
            'external:' + _SAMPLE_DOC + ':42')
        assert match.group(1) == 'external:'
        assert match.group(2) == _SAMPLE_DOC


# ─────────────────────────────────────────────────────────────────────────────
# Runnable ``>>>`` examples in uacpy/io
# ─────────────────────────────────────────────────────────────────────────────

import doctest       # noqa: E402

_IO_DIR = Path(uacpy.__file__).resolve().parent / 'io'

#: An example that calls a ``write_*`` function builds every input it needs —
#: there is no user file to stand in for. So it is runnable by construction,
#: and the first thing a reader of that writer copies. Reader examples in the
#: same package are deliberately not collected: they open the caller's own
#: model output, which no example can supply.
_WRITER_CALL = re.compile(r'\bwrite_\w+\s*\(')


def _io_writer_examples():
    """``(module, qualname)`` for every ``uacpy/io`` docstring whose ``>>>``
    block calls a writer."""
    parser = doctest.DocTestParser()
    found = []
    for path in sorted(_IO_DIR.rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        module = f"uacpy.io.{path.stem}"
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef)):
                continue
            text = ast.get_docstring(node)
            if not text or '>>>' not in text:
                continue
            source = ''.join(e.source for e in parser.get_examples(text))
            if _WRITER_CALL.search(source):
                found.append((module, node.name))
    return found


_IO_WRITER_EXAMPLES = _io_writer_examples()


def test_the_writer_example_sweep_finds_the_known_writers():
    """The sweep itself, so an empty collection cannot pass as a green gate.

    ``write_ssp`` is the named member; the count is a floor, not a pin, so
    documenting another writer does not fail this."""
    assert ('uacpy.io.oalib_writer', 'write_ssp') in _IO_WRITER_EXAMPLES
    assert len(_IO_WRITER_EXAMPLES) >= 5


@pytest.mark.parametrize('module_name,qualname', _IO_WRITER_EXAMPLES,
                         ids=[f'{m.rsplit(".", 1)[-1]}.{q}'
                              for m, q in _IO_WRITER_EXAMPLES])
def test_every_io_writer_docstring_example_runs(module_name, qualname,
                                                tmp_path, monkeypatch):
    """Run each writer's own ``>>>`` block, in an empty directory.

    Nothing else in the suite executes a docstring: ``pyproject.toml`` sets no
    ``--doctest-modules``, and ``test_every_documented_code_sample_parses``
    reads the Markdown pages and only parses. So an example that calls the
    function with arguments the function's own guards reject is green
    everywhere until a user copies it.

    The empty ``tmp_path`` is the second half of the check: an example that
    writes into the caller's working directory leaves a file behind here, and
    the convention these follow (``_fortran_helpers.read_vector``) is a
    ``tempfile.TemporaryDirectory``."""
    import numpy as np

    module = importlib.import_module(module_name)
    target = getattr(module, qualname)
    globs = dict(vars(module))
    globs.setdefault('np', np)
    monkeypatch.chdir(tmp_path)

    runner = doctest.DocTestRunner(
        optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    output = []
    tests = doctest.DocTestFinder(recurse=False).find(
        target, name=f"{module_name}.{qualname}", globs=globs)
    assert tests, f"no doctest collected for {module_name}.{qualname}"
    for test in tests:
        assert test.examples, f"no examples in {module_name}.{qualname}"
        result = runner.run(test, out=output.append)
        assert result.failed == 0, ''.join(output)

    assert list(tmp_path.iterdir()) == [], (
        f"{module_name}.{qualname}'s example wrote into the caller's working "
        f"directory: {[p.name for p in tmp_path.iterdir()]}")


# ── README.md's documentation-page count ──────────────────────────────────


#: The phrase README.md's documentation bullet uses for each ``docs/guide/``
#: page. A new guide page has to be added here *and* to that sentence.
_GUIDE_PAGE_PHRASES = {
    'environment': 'environments',
    'source-receiver': 'sources and receivers',
    'results': 'results',
    'plotting': 'plotting',
    'signal': 'signal processing',
    'arrays': 'arrays',
    'comms': 'communications',
    'noise': 'noise',
    'sonar': 'sonar',
    'data': 'external data',
    'io': 'I/O',
    'utilities': 'utilities',
    'reproducibility': 'reproducibility',
}


def _docs_bullet() -> str:
    """README.md's ``docs/`` bullet, up to the start of the next bullet, with
    its runs of whitespace collapsed — the bullet is wrapped, so a phrase like
    "external data" is split across two source lines."""
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    start = text.index("the guided documentation.")
    return re.sub(r"\s+", " ", text[start:text.index("\n- **", start)])


def _content_pages():
    """``(guide stems, model stems, index READMEs)``.

    ``docs/DEV.md`` is deliberately outside the count: README.md bullets it
    separately, two bullets below the one this gate reads."""
    guide = sorted(p.stem for p in (DOCS_DIR / "guide").glob("*.md"))
    models = sorted(p.stem for p in (DOCS_DIR / "models").glob("*.md")
                    if p.name != "README.md")
    indexes = [DOCS_DIR / "README.md", DOCS_DIR / "models" / "README.md"]
    return guide, models, indexes


@requires_docs
def test_the_readme_documentation_page_count_matches_the_docs_tree() -> None:
    """README.md publishes both a page count and the count including the two
    index READMEs; a page added under ``docs/guide/`` moves both."""
    guide, models, indexes = _content_pages()
    assert all(p.is_file() for p in indexes), [str(p) for p in indexes]
    match = re.search(r"(\d+) pages \((\d+)\s+counting the two index READMEs\)",
                      _docs_bullet())
    assert match is not None, "README.md's page-count phrasing changed"
    assert int(match.group(1)) == len(guide) + len(models)
    assert int(match.group(2)) == len(guide) + len(models) + len(indexes)


@requires_docs
def test_the_readme_names_every_guide_page() -> None:
    """The same bullet enumerates the guides in prose. A page present on disk
    but absent from the sentence is invisible to a reader of the front page."""
    guide, _, _ = _content_pages()
    unmapped = sorted(set(guide) - set(_GUIDE_PAGE_PHRASES))
    assert unmapped == [], f"add these to _GUIDE_PAGE_PHRASES: {unmapped}"
    bullet = _docs_bullet()
    missing = sorted(stem for stem in guide
                     if _GUIDE_PAGE_PHRASES[stem] not in bullet)
    assert missing == []


# ── reproducibility.md §2's enumeration of every drawing function ─────────


_NUMBER_WORDS = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
                 'eleven': 11, 'twelve': 12}

#: Public functions whose draw happens inside a private helper they call, so
#: the AST sweep does not see it at their own definition.
_DRAWS_THROUGH_A_HELPER = {'make_bandlimited_noise'}


def _reproducibility_table():
    """``{function name: (seeding argument, default column)}`` from §2."""
    text = (DOCS_DIR / "guide" / "reproducibility.md").read_text(encoding="utf-8")
    table = {}
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) != 4 or not cells[0].startswith('`uacpy'):
            continue
        knob = re.findall(r"`(\w+)=`", cells[2])
        if not knob:
            continue
        for name in re.findall(r"`(\w+)`", cells[1]):
            table[name] = (knob[0], cells[3].strip('* '))
    return table


def _public_draw_sites():
    """Every public, exported function whose own body calls ``default_rng`` or
    constructs a ``Generator`` — an AST sweep, so a docstring example that
    seeds a generator is not counted as a draw."""
    package = Path(uacpy.__file__).resolve().parent
    exported = set(getattr(uacpy, '__all__', ()))
    for module in pkgutil.iter_modules([str(package)]):
        try:
            exported |= set(getattr(
                importlib.import_module(f"uacpy.{module.name}"), '__all__', ()))
        except Exception:                          # noqa: BLE001 — optional deps
            continue
    names = set()
    for path in sorted(package.rglob("*.py")):
        parts = path.relative_to(package).parts
        if parts and parts[0] in {'tests', 'examples'}:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith('_') or node.name not in exported:
                continue
            for call in (c for c in ast.walk(node) if isinstance(c, ast.Call)):
                func = call.func
                attr = (func.attr if isinstance(func, ast.Attribute)
                        else getattr(func, 'id', None))
                if attr in ('default_rng', 'Generator'):
                    names.add(node.name)
                    break
    return names


@requires_docs
def test_the_reproducibility_page_enumerates_every_public_draw() -> None:
    """§2 is the one page whose job is to list what a caller must pin, and it
    claims to be exhaustive. A public drawing function missing from it is a
    stream a reader has no way to know to look for — whether or not that
    function happens to be seeded by default."""
    swept = _public_draw_sites() | _DRAWS_THROUGH_A_HELPER
    assert swept, "the draw-site sweep found nothing — this gate is blind"
    assert swept == set(_reproducibility_table())


@requires_docs
def test_each_default_column_matches_the_live_signature() -> None:
    """§2's ``Default`` column is the load-bearing half: it says which of the
    listed functions a reproducible run has to pin. ``unseeded`` means the
    seeding argument defaults to ``None``; ``fixed seed`` means it does not."""
    for name, (knob, default_column) in _reproducibility_table().items():
        function = _resolve_public(name)
        default = inspect.signature(function).parameters[knob].default
        if default_column == 'unseeded':
            assert default is None, (name, default)
        else:
            assert default_column == 'fixed seed', (name, default_column)
            assert default is not None, name


@requires_docs
def test_the_page_counts_the_functions_a_run_has_to_pin() -> None:
    """Both numbers §2 states in words: how many draw at all, and how many of
    those are unseeded by default."""
    text = (DOCS_DIR / "guide" / "reproducibility.md").read_text(encoding="utf-8")
    table = _reproducibility_table()
    unseeded = [n for n, (_, default) in table.items() if default == 'unseeded']
    total = re.search(r"\*\*Exactly (\w+) public functions draw at all\*\*", text)
    pinned = re.search(r"\*\*Nine of the ten draw afresh unless you seed them\*\*",
                       text)
    assert total is not None and pinned is not None, "§2 phrasing changed"
    assert _NUMBER_WORDS[total.group(1)] == len(table)
    assert len(unseeded) == 9, len(unseeded)


def test_a_fixed_default_seed_really_repeats_without_being_asked() -> None:
    """The claim §2 makes about its one exception, measured rather than read
    off the signature."""
    import numpy as np

    first = uacpy.comms.schmidl_cox_preamble(64, 16)
    assert np.array_equal(first, uacpy.comms.schmidl_cox_preamble(64, 16))


def _resolve_public(name):
    """The public function ``name`` refers to, from the surfaces §2 names."""
    for module in (uacpy, uacpy.acoustic_signal, uacpy.comms):
        function = getattr(module, name, None)
        if function is not None:
            return function
    raise AssertionError(f"{name} is on no public surface")


@requires_docs
def test_the_checklist_counts_the_functions_it_sends_a_reader_to() -> None:
    """§5 item 1 states the same pinned-function count as §2, in words."""
    text = (DOCS_DIR / "guide" / "reproducibility.md").read_text(encoding="utf-8")
    table = _reproducibility_table()
    unseeded = [n for n, (_, default) in table.items() if default == 'unseeded']
    checklist = re.search(r"on any of the (\w+)\s*\n?\s*unseeded-by-default", text)
    assert checklist is not None, "§5 item 1 phrasing changed"
    assert _NUMBER_WORDS[checklist.group(1)] == len(unseeded)


@requires_docs
def test_every_enumerated_function_takes_the_seeding_argument_named() -> None:
    """A row that names ``rng=`` for a function taking ``seed=`` would send a
    reader to a keyword that raises."""
    for name, (knob, _default) in _reproducibility_table().items():
        function = _resolve_public(name)
        assert knob in inspect.signature(function).parameters, (name, knob)


def test_an_unseeded_sea_surface_draws_afresh_and_a_seeded_one_repeats() -> None:
    """The property §2's new row claims: the ninth function is unseeded by
    default, and its ``seed=`` is effective."""
    import numpy as np

    assert not np.array_equal(uacpy.generate_sea_surface(1000.0, n_points=16),
                              uacpy.generate_sea_surface(1000.0, n_points=16))
    assert np.array_equal(
        uacpy.generate_sea_surface(1000.0, n_points=16, seed=7),
        uacpy.generate_sea_surface(1000.0, n_points=16, seed=7))


# ── docs/guide/data.md against uacpy.data's export surface ───────────────


@requires_docs
@pytest.mark.skipif(not (DOCS_DIR / 'guide' / 'data.md').is_file(),
                    reason='docs/ is not present (source checkout only)')
class TestDataGuidePageMatchesTheExportSurface:
    """The same pair of gates ``io.md`` carries. ``data.md`` publishes a public
    name count in its header and teaches the cache API in §9; neither was held
    to the module, and the §9 block taught an underscore import."""

    @staticmethod
    def _page():
        return (DOCS_DIR / 'guide' / 'data.md').read_text(encoding='utf-8')

    def test_header_public_name_count_equals_len_of_all(self):
        import uacpy.data
        match = re.search(r'`uacpy\.data` · (\d+) public names', self._page())
        assert match is not None
        assert int(match.group(1)) == len(uacpy.data.__all__)

    def test_the_cache_section_teaches_the_public_names(self):
        """A user-facing guide reaching into ``uacpy.data._cache`` teaches an
        underscore import as the supported way to answer "where is my cache?"."""
        import uacpy.data
        page = self._page()
        section = page.split('## 9. The offline cache')[1].split('## 10.')[0]
        assert 'from uacpy.data import _cache' not in section
        assert '_cache.' not in section
        for name in ('cache_root', 'dataset_root', 'is_installed'):
            assert f'data.{name}(' in section, name
            assert name in uacpy.data.__all__, name


def test_the_cache_introspection_names_answer_without_an_exception():
    """What the guide's §9 block claims of them: a directory, a dataset
    directory, and a boolean that needs no ``try``/``except``."""
    from pathlib import Path

    import uacpy.data

    assert isinstance(uacpy.data.cache_root(), Path)
    assert (uacpy.data.dataset_root('woa23')
            == uacpy.data.cache_root() / 'woa23')
    assert isinstance(uacpy.data.is_installed('woa23'), bool)


def test_an_unknown_dataset_name_is_a_typo_not_an_uninstalled_dataset():
    """``is_installed`` answers ``False`` for a real dataset that is absent and
    raises for a name no dataset has — the two are different questions."""
    import uacpy.data
    from uacpy.core.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match='unknown dataset'):
        uacpy.data.is_installed('nope')


def test_is_installed_follows_the_cache_root_it_is_pointed_at(tmp_path,
                                                              monkeypatch):
    """Both sides of the boundary the predicate sits on: the dataset directory
    present, and absent."""
    import uacpy.data

    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path))
    assert uacpy.data.is_installed('woa23') is False
    (tmp_path / 'woa23').mkdir()
    assert uacpy.data.is_installed('woa23') is True
    assert uacpy.data.is_installed('woa23', 'absent.nc') is False
    (tmp_path / 'woa23' / 'absent.nc').write_bytes(b'')
    assert uacpy.data.is_installed('woa23', 'absent.nc') is True


def test_no_example_reaches_into_the_private_cache_module():
    """The shipped examples are what a user copies. One hand-rolling an
    ``is_installed`` predicate out of ``_cache.require`` and ``try``/``except``
    teaches that shape as the supported one."""
    examples = Path(uacpy.__file__).resolve().parent / 'examples'
    offenders = [path.name for path in sorted(examples.rglob('*.py'))
                 if 'uacpy.data._cache' in path.read_text(encoding='utf-8')
                 or 'from uacpy.data import _cache' in path.read_text(encoding='utf-8')]
    assert offenders == []


@requires_docs
def test_a_broken_spaced_language_fence_fails_the_structure_gate(tmp_path):
    gate = _load("check_structure")
    page = tmp_path / 'page.md'
    page.write_text('# Title\n\n``` python\ndef broken(:\n```\n',
                    encoding='utf-8')
    assert any(kind == 'SYNTAX_BLOCK' for kind, _ in gate.check_page(page))


@requires_docs
def test_a_valid_spaced_language_fence_passes_the_structure_gate(tmp_path):
    gate = _load("check_structure")
    page = tmp_path / 'page.md'
    page.write_text('# Title\n\n``` python\nx = 1\n```\n', encoding='utf-8')


def test_every_public_sonar_name_appears_in_the_reference_manual():
    import uacpy.sonar as sonar
    text = _documentation_text()
    missing = [name for name in sonar.__all__
               if not inspect.ismodule(getattr(sonar, name))
               and not re.search(rf'\b{re.escape(name)}\b', text)]
    assert missing == []
