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
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

import uacpy

DOCS_DIR = Path(uacpy.__file__).resolve().parent.parent / "docs"

pytestmark = pytest.mark.skipif(
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


def test_no_broken_relative_links() -> None:
    """Every relative link and image target in the docs resolves on disk."""
    checker = _load("check_links")

    broken = [
        f"{path.relative_to(checker.REPO_ROOT)}:{line}: -> {target}"
        for path in checker.markdown_files(DOCS_DIR)
        for line, target in checker.broken_links(path)
    ]

    assert not broken, "broken relative link(s):\n" + "\n".join(broken)


def test_documentation_is_structurally_sound() -> None:
    """No unbalanced fences, duplicated blocks, dead figures or bad tables."""
    checker = _load("check_structure")

    problems = [
        f"{path.relative_to(checker.REPO_ROOT)}: {kind}: {detail}"
        for path in checker.markdown_files(DOCS_DIR)
        for kind, detail in checker.check_page(path)
    ]

    assert not problems, "structural problem(s):\n" + "\n".join(problems)


def test_every_documented_code_sample_parses() -> None:
    """Python samples and REPL transcripts are syntactically valid.

    Separated from the structural test so a broken sample names itself in the
    failure rather than hiding among unrelated formatting problems.
    """
    checker = _load("check_structure")

    broken = [
        f"{path.relative_to(checker.REPO_ROOT)}: {kind}: {detail}"
        for path in checker.markdown_files(DOCS_DIR)
        for kind, detail in checker.check_page(path)
        if kind.startswith("SYNTAX_")
    ]

    assert not broken, "unparseable code sample(s):\n" + "\n".join(broken)


# ═══════════════════════════════════════════════════════════════════════════
# DOCUMENTATION.md ↔ code consistency
# ═══════════════════════════════════════════════════════════════════════════

import ast      # noqa: E402
import inspect  # noqa: E402
import re       # noqa: E402

DOCUMENTATION_MD = Path(uacpy.__file__).resolve().parent.parent / "DOCUMENTATION.md"


def _documentation_text() -> str:
    if not DOCUMENTATION_MD.is_file():
        pytest.skip("DOCUMENTATION.md is not present")
    return DOCUMENTATION_MD.read_text(encoding="utf-8")


def _model_classes() -> dict:
    """The twelve models the DOCUMENTATION.md tables describe, by name."""
    from uacpy.models import RAM, SPARC, Bellhop, Bounce, Kraken, Scooter
    from uacpy.models.oases import OASN, OASP, OASR, OASS, OASSP, OAST
    return {
        "Bellhop": Bellhop, "Kraken": Kraken, "Scooter": Scooter, "RAM": RAM,
        "SPARC": SPARC, "Bounce": Bounce, "OAST": OAST, "OASN": OASN,
        "OASR": OASR, "OASP": OASP, "OASS": OASS, "OASSP": OASSP,
    }


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
        try:
            model = cls(verbose=False)
        except ExecutableNotFoundError:
            pytest.skip(f"{name} binary not installed")
        except Exception:
            # OASS/OASSP require correlation_length at construction.
            model = cls(verbose=False, correlation_length=10.0)

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
