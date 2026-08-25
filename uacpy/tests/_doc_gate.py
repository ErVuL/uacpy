"""The Markdown-citation gate's machinery, shared by more than one test module.

``test_packaging.py`` runs these against the real tree and ``test_documentation.py``
exercises them on synthetic documents where the answer is known in advance. It
lived in ``test_packaging.py`` and the second caller reached it by loading that
file by path with ``importlib`` — which re-executed a test module, including its
module-level ``importorskip``, to borrow four helpers. A plain module both can
import removes the need.

Not named ``test_*``, so pytest does not collect it; the tests that exercise
these live in ``test_documentation.py``.
"""

import re
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]


# Any ``file.md:NNN``, with the ``external:`` marker DEV.md §10 defines for
# source not vendored here. Every one of these is an offence unless the file is
# vendored Markdown (which keeps its line numbers) or carries that marker.
# Deciding on *resolution* would be backwards for a ban: a basename nothing
# resolves is not a reason to permit a line pin, it is a reason to flag one.
# ``README.md`` is the case that made this concrete — it exists in four places
# in this repo, so it resolves to nothing unambiguous and a resolution-based
# rule would let it through.
_OWN_DOC_LINE_PIN = re.compile(
    r"(?<![\w./-])(external:)?([\w./-]+\.md):\d")


# ``file.md §N`` / ``file.md §N.M``, optionally carrying the phrase the citing
# comment leans on. The bare section is the form the package already used in
# some seventy places; the quote is what an address converted off a line number
# carries, and it is read against a single line of the document, so a phrase
# must not span one of the document's line breaks.
_DOC_ANCHOR = re.compile(
    r"(?<![\w./-])([\w./-]+\.md)\s+§(\d+(?:\.\d+)*)(?![\d.])"
    r"(?:[ \t]*\"([^\"\n]+)\")?")


# A Markdown ATX heading, split into its ``#``s and its first word, so a
# numbered one (``## 6. Worked example``, ``### 6.5 Leaky modes``) can be told
# from a prose one (``### The modes``).
_MD_HEADING = re.compile(r"^(#{1,6})[ \t]+(\S+)")


# Floors on coverage, on the same reasoning as the vendored gate's: a regex
# that stops matching would otherwise leave the gate silently green. The
# second is the sharper of the two — a quote that wrapped onto the next source
# line stops being read *as a quote* while the anchor around it still
# resolves, so only a count of the quoted ones can see that happen.
_MIN_DOC_ANCHORS_RESOLVED = 60


_MIN_QUOTED_DOC_ANCHORS = 34


# Trees whose Markdown is not documentation: build products, dependencies, and
# tool caches. ``.pytest_cache`` earns its place here twice over — a stale
# artefact has no business in a gate's file set, and its ``README.md`` was one
# of the copies that made that basename ambiguous.
_NOT_DOCUMENTATION = frozenset({
    "uacpy_venv", "build", ".git", ".tox", "node_modules", ".pytest_cache",
    "__pycache__", ".mypy_cache", ".ruff_cache", "dist", "site-packages",
})


def _repo_markdown() -> dict:
    """Every Markdown file in the repo, indexed by basename."""
    by_name: dict = {}
    for path in _REPO_ROOT.rglob("*.md"):
        if set(path.parts) & _NOT_DOCUMENTATION:
            continue
        by_name.setdefault(path.name, []).append(path)
    return by_name


def _resolve_markdown(by_name: dict, cited: str):
    """The one Markdown file ``cited`` names, or ``None`` if it is not ours or
    the basename is ambiguous."""
    wanted = Path(cited).parts
    hits = [p for p in by_name.get(wanted[-1], ())
            if p.parts[-len(wanted):] == wanted]
    return hits[0] if len(hits) == 1 else None


def _line_pin_verdict(by_name: dict, external: bool, cited: str):
    """Why ``cited.md:NNN`` is an offence, or ``None`` if it is allowed.

    Split out of the gate so the rule can be read and tested on its own: the
    two exemptions are deliberate, and the third case — a name nothing
    resolves — is the one an earlier draft got backwards.
    """
    if external:
        return None                              # marked unreadable, DEV.md §10
    target = _resolve_markdown(by_name, cited)
    if target is None:
        return "no single file in this repo resolves that name"
    if "third_party" in target.parts:
        return None                              # vendored, the other gate's
    return str(target.relative_to(_REPO_ROOT))


def _headings(body):
    """``(index, depth, leading token)`` for every ATX heading in ``body``.

    Two kinds of line that merely *look* like headings are skipped, because
    these documents are full of both:

    * a ``#`` comment inside a fenced code block — reading ``# Force Scooter
      to use the near-source SSP`` as a level-1 heading ended the section
      containing it about 300 lines early;
    * a ``#`` inside an HTML comment.

    Fences follow CommonMark on the two points that matter here: a closing
    fence must use the same character as the opener and be at least as long,
    so a ```` ```` ```` block is not closed by an inner ``` and a ``~~~``
    block is not closed by a ``` at all.

    An **unbalanced** fence therefore runs to the end of the file and hides
    every heading after it — correct per CommonMark, and a real blind spot for
    the section walker rather than a bug in it. No document in this repo has
    one today; if one ever does, its citations will fail with "no heading
    numbered §N", which is the visible failure and not a silent pass.
    """
    fence = None
    in_html_comment = False
    for i, line in enumerate(body):
        stripped = line.lstrip()
        if in_html_comment:
            if "-->" in stripped:
                in_html_comment = False
            continue
        if fence is None and stripped.startswith("<!--"):
            if "-->" not in stripped:
                in_html_comment = True
            continue
        if stripped[:1] in ("`", "~"):
            char = stripped[0]
            run = len(stripped) - len(stripped.lstrip(char))
            if run >= 3:
                if fence is None:
                    fence = (char, run)
                elif fence[0] == char and run >= fence[1]:
                    fence = None
                continue
        if fence is not None:
            continue
        match = _MD_HEADING.match(line)
        if match is not None:
            yield i, len(match.group(1)), match.group(2).rstrip(".")


def _section_span(body, section: str):
    """``(first, last)`` 0-based line bounds of the ``§section`` heading's
    body, or ``None`` when no heading carries that number.

    A section runs to the next heading at the same or a shallower level, so
    ``§6`` contains ``§6.1`` and its unnumbered subsections.
    """
    start = level = None
    for i, depth, number in _headings(body):
        if start is None:
            if number == section:
                start, level = i, depth
            continue
        if depth <= level:
            return start, i - 1
    return None if start is None else (start, len(body) - 1)
