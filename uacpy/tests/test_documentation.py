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
    union across its six instantiable sub-models, as the page states."""
    from uacpy.models import RAM, SPARC, Bellhop, Bounce, Kraken, Scooter
    from uacpy.models.oases import OASN, OASP, OASR, OASS, OASSP, OAST
    out = {name: {m.name for m in cls.spec.modes} for name, cls in (
        ("Bellhop", Bellhop), ("Kraken", Kraken), ("Scooter", Scooter),
        ("SPARC", SPARC), ("RAM", RAM), ("Bounce", Bounce))}
    out["OASES"] = {m.name for cls in (OAST, OASN, OASR, OASP, OASS, OASSP)
                    for m in cls.spec.modes}
    return out


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
