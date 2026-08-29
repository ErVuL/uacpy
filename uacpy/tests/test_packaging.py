"""Package-discovery guard for pyproject.toml.

Pins the "12 packages, not 222" regression: before ``namespaces = false``
and the explicit ``include``/``exclude`` lists, flat-layout discovery with
a bare ``uacpy*`` glob swept ``uacpy/bin``, ``uacpy/examples``, the whole
of ``uacpy/third_party`` and even ``uacpy_venv`` into the wheel.

The test calls ``setuptools.find_packages`` with EXACTLY the settings
parsed from ``[tool.setuptools.packages.find]`` — not a re-hardcoded
copy — so any drift in the config itself is what fails here. No build is
performed.

The file has since taken on the rest of the project-configuration gates that
read the repo rather than run it: the version/licence/``py.typed`` declarations,
the lint invocation CI uses, the line-pin and doc-anchor citation gates, and
the ``filterwarnings`` policy at the end."""

import ast
import re
import subprocess
import tomllib
import warnings
from pathlib import Path

import numpy as np
import pytest

setuptools = pytest.importorskip(
    "setuptools", reason="packaging guard needs the build backend"
)

# The Markdown-citation machinery is shared with test_documentation.py.
from uacpy.tests._doc_gate import (            # noqa: E402
    _DOC_ANCHOR,
    _MIN_DOC_ANCHORS_RESOLVED,
    _MIN_QUOTED_DOC_ANCHORS,
    _NOT_DOCUMENTATION,
    _OWN_DOC_LINE_PIN,
    _REPO_ROOT,
    _line_pin_verdict,
    _repo_markdown,
    _resolve_markdown,
    _section_span,
)
from uacpy.tests import conftest  # noqa: E402


# The complete importable surface the wheel ships. A new subpackage must be
# added here deliberately — discovery growing on its own is the regression
# this file exists to catch.
_EXPECTED_PACKAGES = [
    "uacpy",
    "uacpy.acoustic_signal",
    "uacpy.comms",
    "uacpy.core",
    "uacpy.core.results",
    "uacpy.data",
    "uacpy.io",
    "uacpy.models",
    "uacpy.noise",
    "uacpy.sonar",
    "uacpy.visualization",
    "uacpy.visualization.plots",
]

# Directory trees that live inside the repo (and inside uacpy/) but must
# never be discovered as packages.
_MUST_NOT_SHIP = ("uacpy_venv", "third_party", "bin", "examples", "tests")


def _find_config():
    pyproject = _REPO_ROOT / "pyproject.toml"
    with pyproject.open("rb") as fh:
        cfg = tomllib.load(fh)
    return cfg["tool"]["setuptools"]["packages"]["find"]


def _discover():
    cfg = _find_config()
    # ``namespaces`` selects the finder exactly as setuptools.build_meta
    # does: true (the setuptools default) walks every directory, false
    # requires an __init__.py chain.
    if cfg.get("namespaces", True):
        finder = setuptools.find_namespace_packages
    else:
        finder = setuptools.find_packages
    return sorted(finder(
        where=str(_REPO_ROOT),
        include=cfg.get("include", ("*",)),
        exclude=cfg.get("exclude", ()),
    ))


def test_discovery_yields_exactly_the_shipped_packages():
    assert _discover() == _EXPECTED_PACKAGES


def test_every_discovered_package_is_under_the_uacpy_root():
    for pkg in _discover():
        assert pkg == "uacpy" or pkg.startswith("uacpy."), pkg


def test_no_forbidden_tree_is_discovered():
    packages = _discover()
    for forbidden in _MUST_NOT_SHIP:
        hits = [p for p in packages
                if forbidden in p.split(".")]
        assert not hits, (
            f"discovery swept {forbidden!r} into the wheel: {hits}"
        )


def test_the_forbidden_trees_actually_exist():
    """The exclusion assertions above are only meaningful while the trees
    they guard against are present to be swept.

    Only git-tracked trees are asserted: ``uacpy_venv`` and ``uacpy/bin``
    exist where a developer or CI's build step has created them, a fresh
    clone has neither, and discovery has nothing to sweep in an absent
    tree — so their exclusions are exercised on every machine that has
    them and vacuous on the machines that do not."""
    for tracked in ("uacpy/third_party", "uacpy/examples", "uacpy/tests"):
        assert (_REPO_ROOT / tracked).is_dir(), tracked


def test_the_config_carries_the_load_bearing_keys():
    """``namespaces = false`` is what stops discovery at directories with
    no __init__.py; the include/exclude pair is what keeps uacpy_venv and
    uacpy.tests out. Losing any key silently reverts to setuptools
    defaults (namespaces=true, include everything)."""
    cfg = _find_config()
    assert cfg.get("namespaces") is False
    assert "include" in cfg and "exclude" in cfg
    assert "uacpy.tests" in cfg["exclude"]


def test_version_exists_and_matches_pyproject():
    """``uacpy.__version__`` is importable and is the same string the wheel
    declares: pyproject marks ``version`` dynamic and reads it from
    ``uacpy._version.__version__``, so this pins the whole chain — the
    public attribute, the single-source module, and the pyproject
    directive that ties them together. No build is performed."""
    import uacpy
    from uacpy._version import __version__ as source_version

    assert isinstance(uacpy.__version__, str) and uacpy.__version__
    assert uacpy.__version__ == source_version
    assert '__version__' in uacpy.__all__

    with (_REPO_ROOT / "pyproject.toml").open("rb") as fh:
        cfg = tomllib.load(fh)
    assert "version" in cfg["project"]["dynamic"]
    assert (cfg["tool"]["setuptools"]["dynamic"]["version"]["attr"]
            == "uacpy._version.__version__")


def test_the_licence_is_declared_as_an_spdx_expression():
    """PEP 639: ``license`` is a bare SPDX string with the licence text listed
    in ``license-files``. The old ``{ text = ... }`` table is deprecated with a
    removal date, and pairing an SPDX string with a ``License ::`` trove
    classifier is a hard ``InvalidConfigError`` — so the two move together."""
    with (_REPO_ROOT / "pyproject.toml").open("rb") as fh:
        cfg = tomllib.load(fh)
    project = cfg["project"]
    assert project["license"] == "GPL-3.0-or-later"
    assert project["license-files"] == ["LICENSE"]
    assert (_REPO_ROOT / "LICENSE").is_file()
    assert not [c for c in project["classifiers"] if c.startswith("License ::")]
    # PEP 639 support landed in setuptools 77; an older backend rejects both
    # fields, so the build requirement has to keep pace with them.
    requires = cfg["build-system"]["requires"]
    assert any(r.replace(" ", "").startswith("setuptools>=")
               and int(r.split(">=")[1].split(".")[0].strip()) >= 77
               for r in requires), requires


def test_the_config_setuptools_reads_matches_the_file():
    """The declaration has to survive setuptools' own parser, not just
    ``tomllib``: an SPDX ``license`` string that the installed backend cannot
    read fails at build time, long after this file is edited."""
    from setuptools.config.pyprojecttoml import read_configuration
    project = read_configuration(str(_REPO_ROOT / "pyproject.toml"))["project"]
    assert project["license"] == "GPL-3.0-or-later"
    assert project["license-files"] == ["LICENSE"]


def test_the_py_typed_marker_ships():
    """PEP 561: without the marker a downstream type checker ignores every
    annotation in the package, however completely it is annotated.

    Two declarations, neither of which is the artifact:
    ``test_a_built_wheel_carries_the_py_typed_marker`` is what reads the zip.
    """
    import uacpy
    assert (Path(uacpy.__file__).parent / "py.typed").is_file()
    with (_REPO_ROOT / "pyproject.toml").open("rb") as fh:
        cfg = tomllib.load(fh)
    package_data = cfg["tool"]["setuptools"]["package-data"]
    assert "py.typed" in package_data["uacpy"]


# ── the artifact this config actually produces ──────────────────────────────

# Trees that exist in a real checkout and must stay out of the wheel, mapped
# to the files each one carries. Recreated as stubs in the staged copy below:
# copying the real ones is 1.2 GB, and a copy without them present cannot show
# the "222 packages instead of 12" regression coming back.
#
# Which guard rejects which tree is what the file lists reproduce. The first
# group has no ``__init__.py`` — ``namespaces = false`` is what keeps them
# out, and the setuptools default (true) is what swept them in. ``uacpy/tests``
# is an importable package, so only the ``exclude`` list keeps it out.
_STUBBED_TREES = {
    "uacpy/third_party": ("aliases.py",),
    "uacpy/third_party/Acoustics-Toolbox": ("misc.py",),
    "uacpy/bin": ("bellhopcuda",),
    "uacpy/examples": ("example_01_stub.py",),
    "uacpy_venv": ("activate_this.py",),
    "uacpy_venv/site_packages_stub": ("shim.py",),
    "uacpy/tests": ("__init__.py", "test_stub.py"),
    "uacpy/tests/sub": ("__init__.py", "test_stub.py"),
}


def _staged_tree(root):
    """A buildable copy of the repo under ``root``: the shipped package
    verbatim, the build config beside it, and a stub where each gitignored
    tree sits. Built here rather than in place because a build writes
    ``build/`` and ``uacpy.egg-info/`` next to the sources."""
    import shutil

    staged = root / "repo"
    staged.mkdir(parents=True)
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(_REPO_ROOT / name, staged / name)
    shutil.copytree(
        _REPO_ROOT / "uacpy", staged / "uacpy",
        ignore=shutil.ignore_patterns("third_party", "bin", "examples",
                                      "tests", "__pycache__"))
    for relative, files in _STUBBED_TREES.items():
        directory = staged / relative
        directory.mkdir(parents=True, exist_ok=True)
        for name in files:
            (directory / name).write_text("", encoding="utf-8")
    return staged


def _build_wheel(staged, outdir):
    """Drive the project's own PEP 517 backend over ``staged``; return the
    wheel path. The backend is run in a subprocess so the build's own
    ``build/`` and ``uacpy.egg-info/`` land in the staged copy, and the
    result is read off ``outdir`` rather than off the call: the wheel name
    the backend returns is discarded by the ``SystemExit`` distutils raises
    on the way out of a successful build."""
    import sys

    outdir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys\n"
         "import setuptools.build_meta as backend\n"
         "backend.build_wheel(sys.argv[1])\n",
         str(outdir)],
        cwd=str(staged), capture_output=True, text=True, timeout=600,
    )
    built = sorted(outdir.glob("*.whl"))
    assert proc.returncode == 0 and len(built) == 1, (
        f"the project's own build backend produced {len(built)} wheel(s) "
        f"(rc={proc.returncode}):\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    return built[0]


@pytest.fixture(scope="module")
def wheel_members(tmp_path_factory):
    """``ZipFile.namelist()`` of a freshly built wheel, plus its METADATA.

    Every gate above this one reads the config and stops there; this is the
    one that reads the artifact. Built once for the whole module."""
    import zipfile

    root = tmp_path_factory.mktemp("wheelgate")
    wheel = _build_wheel(_staged_tree(root), root / "out")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata = archive.read(
            f"uacpy-{uacpy_version()}.dist-info/METADATA").decode("utf-8")
    return names, metadata


def uacpy_version():
    import uacpy
    return uacpy.__version__


def test_a_built_wheel_carries_the_py_typed_marker(wheel_members):
    """``test_the_py_typed_marker_ships`` reads the declaration; this reads
    the artifact, which is what a downstream checker sees.

    The two are further apart than they look: emptying
    ``[tool.setuptools.package-data]`` leaves ``uacpy/py.typed`` in the wheel
    anyway, because a pyproject-configured setuptools defaults
    ``include-package-data`` to true and sweeps the file in regardless.
    Deleting the marker itself is what empties it out of the zip, and only
    this gate sees that."""
    names, _ = wheel_members
    assert "uacpy/py.typed" in names, (
        "the built wheel has no uacpy/py.typed, so a downstream type checker "
        f"ignores every annotation in it; non-.py members: "
        f"{[n for n in names if not n.endswith('.py')]}")


def test_a_built_wheel_ships_exactly_the_expected_packages(wheel_members):
    """The "12 packages, not 222" pin, read off the artifact. The discovery
    test calls ``find_packages`` with the parsed settings; this one asks what
    the build actually put in the zip, which is the thing a user installs."""
    names, _ = wheel_members
    # Every directory the wheel puts a module in, not every ``__init__.py``:
    # ``namespaces = true`` sweeps directories that have no ``__init__.py``,
    # so keying on the marker file would miss exactly the regression.
    shipped = sorted({
        name.rsplit("/", 1)[0].replace("/", ".")
        for name in names if name.endswith(".py")})
    assert shipped == _EXPECTED_PACKAGES, (
        f"the built wheel ships {len(shipped)} packages, not the "
        f"{len(_EXPECTED_PACKAGES)} declared: "
        f"unexpected={sorted(set(shipped) - set(_EXPECTED_PACKAGES))} "
        f"missing={sorted(set(_EXPECTED_PACKAGES) - set(shipped))}")


def test_a_built_wheel_carries_nothing_from_a_gitignored_tree(wheel_members):
    """``uacpy/third_party`` is ~150 directories of vendored Fortran,
    ``uacpy/bin`` is machine-specific binaries, and both are gitignored: a
    member from either means discovery swept the checkout rather than the
    package."""
    names, _ = wheel_members
    swept = sorted(n for n in names
                   if any(f"/{tree}/" in f"/{n}" for tree in _MUST_NOT_SHIP))
    assert not swept, (
        f"the built wheel carries {len(swept)} member(s) from a gitignored "
        f"tree: {swept[:10]}")


def test_a_built_wheel_is_marked_private_so_it_cannot_be_uploaded(
        wheel_members):
    """No binary can ship in a wheel — they are machine-specific and OASES is
    non-redistributable — so an uploaded artifact would install cleanly and
    then raise ``ExecutableNotFoundError`` from every model, naming an
    ``./install.sh`` it does not contain. PyPI refuses an upload carrying an
    unregistered classifier, which is what this one is for."""
    _, metadata = wheel_members
    assert "Classifier: Private :: Do Not Upload" in metadata, (
        "the built wheel's METADATA has no `Private :: Do Not Upload` "
        "classifier, so `twine upload` of it would succeed")


# ── CI parity: the lint step ────────────────────────────────────────────────

_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _ci_flake8_argv():
    """The flake8 arguments CI runs, read out of the workflow itself.

    Read rather than re-declared for the same reason the discovery test parses
    ``pyproject.toml``: a copied argument list drifts from CI silently, and the
    whole value of the check is that it runs what CI runs. The workflow's step
    is a literal block scalar, so this joins its backslash continuations and
    splits on whitespace.
    """
    if not _WORKFLOW.is_file():
        pytest.skip(f"{_WORKFLOW} is not present (source checkout only)")
    lines = _WORKFLOW.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("- name: Lint"):
            break
    else:
        pytest.fail("ci.yml has no 'Lint' step — update this gate")
    body, indent = [], None
    for line in lines[i + 1:]:
        if not line.strip():
            continue
        current = len(line) - len(line.lstrip())
        if line.strip().startswith(("- name:", "#")) and current <= (indent or 0):
            break
        if line.strip() in ("run: |", "run:|"):
            indent = current
            continue
        if indent is None:
            continue
        if current <= indent:
            break
        body.append(line.strip())
    command = " ".join(body).replace("\\", " ")
    tokens = command.split()
    assert tokens and tokens[0] == "flake8", (
        f"the Lint step no longer starts with flake8: {command!r}")
    return tokens[1:]


def test_the_tree_passes_the_lint_ci_runs():
    """``flake8`` with CI's own select-list reports nothing.

    ``E9`` / ``F63`` / ``F7`` / ``F82`` are syntax and undefined-name errors;
    ``F401`` / ``F811`` are unused and redefined imports. Nothing else in the
    suite runs flake8, so before this gate the only enforcement was the GitHub
    workflow — and the tree had drifted to five F401s under it.
    """
    import subprocess
    import sys

    pytest.importorskip("flake8", reason="the lint gate needs flake8")
    argv = _ci_flake8_argv()
    proc = subprocess.run(
        [sys.executable, "-m", "flake8", *argv],
        cwd=str(_REPO_ROOT), capture_output=True, text=True,
    )
    assert proc.returncode == 0, (
        "flake8 (CI's select-list) reported problems:\n"
        + (proc.stdout or proc.stderr)
    )


# ── CI parity: the source conventions DEV.md states ─────────────────────────

_INTRA_PACKAGE_LINE_PIN = re.compile(r"\b([a-z_][a-z0-9_]*\.py):\d")


def test_no_comment_pins_a_line_number_in_another_python_file():
    """No comment or docstring cites ``some_module.py:NNN``.

    DEV.md §10: "Do not pin to current line numbers in nearby files; cite
    source-of-truth files (``AttenMod.f90:78``) instead." Vendored Fortran and
    C are the source of truth and change only on a re-vendor, so a line there
    is a stable address; uacpy's own files move every time anyone edits above
    the citation, and four of the seven such pins had already drifted onto
    unrelated code by the time this gate was written — one pointed at a
    function that did not exist when the comment was written.

    Naming the symbol instead (``write_ssp_section``, ``Bellhop.run``) is
    both stable and easier to follow.

    ``tests/`` is covered on the same terms as the package: a test comment
    pinning a package line rots exactly as fast, and a reader who follows the
    address to unrelated code concludes the test's premise is false.
    """
    package = Path(_REPO_ROOT) / "uacpy"
    offenders = []
    for path in sorted(package.rglob("*.py")):
        parts = set(path.parts)
        if "third_party" in parts:
            continue
        for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1):
            for match in _INTRA_PACKAGE_LINE_PIN.finditer(line):
                cited = match.group(1)
                if not list(package.rglob(cited)):
                    continue        # not one of ours — a third-party address
                offenders.append(
                    f"{path.relative_to(_REPO_ROOT)}:{lineno}: {match.group(0)}")

    assert not offenders, (
        "comment(s) pinning a line number inside the package (DEV.md §10 — "
        "cite the symbol instead):\n" + "\n".join(offenders)
    )


# The suffixes uacpy cites vendored source by. ``.py`` is deliberately absent:
# an intra-package address is the gate above's business, not this one's.
# ``md`` is here for the Markdown that ships *inside* ``third_party`` (the
# bellhopcuda docs). An address into uacpy's *own* Markdown matches this
# pattern too; it is handed to
# ``test_no_comment_pins_a_line_number_in_the_projects_own_docs`` below, which
# bans that form outright, rather than failed here as a vendored file the tree
# does not carry.
_VENDORED_SUFFIXES = ("f90", "f", "c", "h", "m", "tex", "md")

# The comma-separated list of single lines and ``NNN-NNN`` ranges an address
# may carry. Captured as one group, and split by the loop below, so that an
# address is either read in full or not matched at all: a pattern anchored on
# a lone ``\d+`` matches no part of ``RefCoef.f90:139-140,146-147`` and lets
# the whole thing through unread.
_ADDRESS = r"\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*"

# ``file.ext:ADDR``, optionally prefixed by the ``external:`` marker DEV.md
# §10 defines for source not vendored here.
_VENDORED_CITATION = re.compile(
    r"(?<![\w./-])(external:)?"
    r"([\w./-]+\.(?:" + "|".join(_VENDORED_SUFFIXES) + r")):"
    r"(" + _ADDRESS + r")(?![-\d])"
)

# A bare ``:NNN`` continuing the address beside it — ``oasvun31.f:421-448,
# :724`` names two places in one file, and neither the gate nor a reader
# should have to re-read the filename to know it.
#
# Only punctuation may stand between the two: a comma or slash, and the RST
# literal backticks a citation is usually wrapped in. A bare address that
# prose has separated from a filename stays unbound and is counted as skipped
# rather than bound by guess, because the nearest filename above it is often
# the wrong one — ``bellhop_writer.write_bellhop_env``'s record-order list
# names its file in prose ("``ReadEnvironmentBell.f90`` unless another file is
# named") and cites two other files along the way.
_BARE_CONTINUATION = re.compile(
    r"`{0,2}[ \t]*[,/][ \t]*`{0,2}:(" + _ADDRESS + r")(?![-\d])")

# The same separator left dangling at the end of a line, and the bare address
# that resumes on the next one — a continuation that wrapped, which in a
# comment block carries the next line's indent and ``#`` in front of it.
_CONTINUATION_OPENER = re.compile(r"`{0,2}[ \t]*[,/][ \t]*`{0,2}[ \t]*$")
_WRAPPED_CONTINUATION = re.compile(
    r"[ \t]*(?:#[ \t]*)?`{0,2}:(" + _ADDRESS + r")(?![-\d])")

# A bare address nothing bound. Counted, never failed. The lookbehind keeps
# Python slices (``data[:5]``) and format specs (``f"{x:9.5g}"``) out of the
# count, which would otherwise swamp it.
_BARE_ADDRESS = re.compile(r"(?<=[\s(`]):(" + _ADDRESS + r")(?![-\d])")

# A floor on coverage: below it the walk or the regex has stopped reaching the
# tree, and a green gate would mean nothing.
_MIN_CITATIONS_CHECKED = 900

# The same, for the bare continuations alone, so the binding above cannot
# lapse back to reading only the addresses that carry a filename of their own.
_MIN_BARE_CONTINUATIONS_BOUND = 90

# And for the Markdown side alone, so a walk that covers both file kinds
# cannot lapse back to Python-only and stay green on the Python count.
_MIN_MARKDOWN_CITATIONS_CHECKED = 40


def _citing_sources():
    """Every file of uacpy's own that may carry a vendored address.

    The package's Python, plus uacpy's own Markdown: ``DOCUMENTATION.md``, the
    ``README.md`` beside it, the guide / model / dev pages under ``docs/``, and
    the ``MODIFICATIONS.md`` that records the local patches to the vendored
    tree. Prose cites ``ram.f90:169`` exactly the way a docstring does, and the
    address resolves — or fails to — by the same rule, so reading only the
    Python left those addresses unchecked.

    ``docs/superpowers`` is *included*, unlike in ``test_documentation.py``'s
    API scan: a page of working notes may use an API that has since moved, but
    a line number into frozen vendored source is either right or wrong whenever
    it was written. Vendored Markdown is excluded on the same ground as
    vendored Python — it is not uacpy's citation.
    """
    package = Path(_REPO_ROOT) / "uacpy"
    paths = [path for path in sorted(package.rglob("*.py"))
             if "third_party" not in path.parts]
    markdown = [Path(_REPO_ROOT) / "DOCUMENTATION.md",
                Path(_REPO_ROOT) / "README.md",
                package / "third_party" / "MODIFICATIONS.md"]
    markdown += sorted((Path(_REPO_ROOT) / "docs").rglob("*.md"))
    paths += [path for path in markdown
              if path.is_file() and not set(path.parts) & _NOT_DOCUMENTATION]
    return paths


def test_vendored_citations_resolve_and_single_line_targets_carry_code():
    """Every element of an ``AttenMod.f90:78``-style address is read.

    A single-line target that is blank is the cheap end of citation drift: the
    address still resolves, so nothing complains, but the evidence it was
    written for has moved a line or two away. A range is exempt from that
    check — a range spanning a blank line is ordinary — but it is still
    resolved, so its ``external:`` marking is checked and its end is required
    to fall inside the file.

    Each element of a comma-continued address (``RefCoef.f90:139-140,146-147``
    cites two ``IF`` arms) is checked on its own terms; a range in the list
    does not exempt the single lines beside it.

    A bare ``:NNN`` that continues the address beside it (``oasvun31.f:421-448,
    :724``) is read against that same file, on the same line or wrapped onto
    the next line of the same comment. Binding reaches no further than the
    punctuation between the two, so a bare address that prose has separated
    from a filename is counted as skipped instead — see ``_BARE_CONTINUATION``.

    The walk covers uacpy's Markdown as well as its Python — see
    :func:`_citing_sources`. Prose cites the Fortran the way a docstring does,
    and 76 of the single-line targets read here live in ``DOCUMENTATION.md``,
    the pages under ``docs/`` and ``third_party/MODIFICATIONS.md``.

    Four groups are skipped rather than guessed at, and the counts ride along
    in every failure message so the coverage stays visible:

    * ``external:``-marked addresses — the CMRE janus-c reference the JANUS
      receiver is transcribed from is not vendored, so nothing here can read
      them. Fifteen at the time of writing, in ``comms/janus.py``,
      ``tests/test_comms.py`` and ``docs/DEV.md``.
    * basenames that resolve to more than one vendored file — ``sspMod.f90``
      (Bellhop's and misc's), ``compar.f`` (``src`` and ``pulsplot``) and
      ``ram1.5.f`` ship more than once each, so a bare basename does not say
      which. Seven at the time of writing; qualify the citation with a
      directory to have them checked.
    * bare ``:NNN`` addresses no citation beside them could bind, and those
      continuing a citation that was itself skipped. Around 550 at the time of
      writing, most of them prose (``:113 scales by c²/ω``); name the file to
      have one checked.
    * addresses into uacpy's own Markdown, which are banned rather than
      resolved — ``test_no_comment_pins_a_line_number_in_the_projects_own_docs``
      owns that population.

    An address that resolves to nothing *without* the marker fails, because it
    is indistinguishable from a typo. One that carries the marker but *does*
    resolve fails too: the marker asserts unverifiability, and the tree can
    verify it.
    """
    package = Path(_REPO_ROOT) / "uacpy"
    by_name: dict = {}
    for path in (package / "third_party").rglob("*"):
        if path.is_file():
            by_name.setdefault(path.name, []).append(path)

    body_cache: dict = {}
    counts = {'checked': 0, 'spanned': 0, 'bound_bare': 0,
              'skipped_bare': 0, 'markdown': 0}
    external, ambiguous, unmarked, offenders = [], [], [], []
    own_doc = []

    def read(address, vendored, where):
        """Read every element of ``address`` against ``vendored``'s body."""
        if vendored not in body_cache:
            body_cache[vendored] = vendored.read_text(
                encoding="utf-8", errors="replace").splitlines()
        body = body_cache[vendored]
        for element in address.split(","):
            if "-" in element:
                counts['spanned'] += 1
                end = int(element.split("-")[1])
                if end > len(body):
                    offenders.append(
                        f"{where} — the range {element} ends past the "
                        f"file's last line {len(body)}")
                continue
            counts['checked'] += 1
            target = int(element)
            if target > len(body):
                offenders.append(
                    f"{where} — line {target} is past the file's last "
                    f"line {len(body)}")
            elif not body[target - 1].strip():
                offenders.append(f"{where} — line {target} is blank")

    def read_continuations(line, at, vendored, name, prefix, bound):
        """Read the bare addresses continuing the one that ended at ``at``.

        ``vendored`` is ``None`` when the citation they continue was itself
        skipped, and they are then counted as skipped too. Returns the
        position past the last one, so the caller can see whether the line
        ends mid-address.
        """
        while True:
            more = _BARE_CONTINUATION.match(line, at)
            if more is None:
                return at
            bound.append(more.span())
            if vendored is None:
                counts['skipped_bare'] += 1
            else:
                counts['bound_bare'] += 1
                read(more.group(1), vendored,
                     f"{prefix} :{more.group(1)} (continuing {name})")
            at = more.end()

    for path in _citing_sources():
        markdown_at = counts['checked'] if path.suffix == ".md" else None
        # The citation a bare address at the start of the next line continues,
        # or ``None`` when the line before it ended no address mid-list.
        wrapping = None
        for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1):
            prefix = f"{path.relative_to(_REPO_ROOT)}:{lineno}:"
            bound = [m.span() for m in _VENDORED_CITATION.finditer(line)]
            carry = None

            resumed = wrapping and _WRAPPED_CONTINUATION.match(line)
            if resumed:
                vendored, name = wrapping
                bound.append(resumed.span())
                counts['bound_bare'] += 1
                read(resumed.group(1), vendored,
                     f"{prefix} :{resumed.group(1)} (continuing {name})")
                at = read_continuations(line, resumed.end(), vendored, name,
                                        prefix, bound)
                if _CONTINUATION_OPENER.search(line[at:]):
                    carry = wrapping
            wrapping = None

            for match in _VENDORED_CITATION.finditer(line):
                where = f"{prefix} {match.group(0)}"
                wanted = Path(match.group(2)).parts
                hits = [p for p in by_name.get(wanted[-1], ())
                        if p.parts[-len(wanted):] == wanted]
                vendored = None
                if match.group(1):
                    external.append(where)
                    if hits:
                        offenders.append(
                            f"{where} — marked external, but "
                            f"{hits[0].relative_to(_REPO_ROOT)} is vendored")
                elif not hits and match.group(2).endswith(".md"):
                    # An address into uacpy's own Markdown, which is banned
                    # outright rather than resolved — see
                    # ``test_no_comment_pins_a_line_number_in_the_projects_own_docs``,
                    # which owns that population. Handing it there keeps one
                    # gate per rule: this one reads addresses into frozen
                    # vendored source, and own Markdown is not frozen.
                    own_doc.append(where)
                elif not hits:
                    unmarked.append(where)
                elif len(hits) > 1:
                    ambiguous.append(where)
                else:
                    vendored = hits[0]
                    read(match.group(3), vendored, where)
                at = read_continuations(line, match.end(), vendored,
                                        match.group(2), prefix, bound)
                if vendored is not None and _CONTINUATION_OPENER.search(
                        line[at:]):
                    carry = (vendored, match.group(2))
            wrapping = carry

            for stray in _BARE_ADDRESS.finditer(line):
                if not any(a <= stray.start() < b for a, b in bound):
                    counts['skipped_bare'] += 1

        if markdown_at is not None:
            counts['markdown'] += counts['checked'] - markdown_at

    checked = counts['checked']
    coverage = (f"{checked} single lines checked ({counts['markdown']} of "
                f"them in Markdown), {counts['spanned']} ranges "
                f"resolved, {counts['bound_bare']} bare continuations bound, "
                f"{counts['skipped_bare']} bare addresses skipped as unbound, "
                f"{len(ambiguous)} skipped as ambiguous, {len(external)} "
                f"skipped as external, {len(own_doc)} handed to the own-doc "
                f"line-pin gate")
    assert not offenders, (
        "citation(s) into the vendored tree whose target carries no code — "
        f"the evidence has drifted ({coverage}):\n" + "\n".join(offenders)
    )
    assert not unmarked, (
        "citation(s) naming a file that is not under uacpy/third_party. Give "
        "an address into un-vendored source the ``external:`` prefix (DEV.md "
        f"§10) so its unverifiability is visible ({coverage}):\n"
        + "\n".join(unmarked)
    )
    assert checked >= _MIN_CITATIONS_CHECKED, (
        f"the citation gate reached only {checked} citations, under the "
        f"{_MIN_CITATIONS_CHECKED} floor — it has stopped covering the tree "
        f"({coverage})"
    )
    assert counts['bound_bare'] >= _MIN_BARE_CONTINUATIONS_BOUND, (
        f"the gate bound only {counts['bound_bare']} bare continuations, "
        f"under the {_MIN_BARE_CONTINUATIONS_BOUND} floor — it has stopped "
        f"reading the addresses that carry no filename of their own "
        f"({coverage})"
    )
    assert counts['markdown'] >= _MIN_MARKDOWN_CITATIONS_CHECKED, (
        f"the gate read only {counts['markdown']} citations in Markdown, "
        f"under the {_MIN_MARKDOWN_CITATIONS_CHECKED} floor — it has stopped "
        f"covering uacpy's own prose and is checking the Python alone "
        f"({coverage})"
    )


def test_the_citation_walk_reads_the_projects_own_markdown():
    """The gates above read uacpy's prose as well as its Python.

    Every file kind that carries a vendored address has to be in the walk, and
    each of the four Markdown homes is here for its own reason:
    ``DOCUMENTATION.md`` is the parameter reference, the pages under ``docs/``
    are the guide and the per-model notes, ``MODIFICATIONS.md`` records the
    local patches address by address, and the working notes under
    ``docs/superpowers`` argue from the Fortran the same way — that tree is
    an untracked dev-checkout companion, so its participation is asserted
    only where it exists. Vendored Markdown is excluded — it is not uacpy's
    citation.
    """
    sources = _citing_sources()
    relative = {str(path.relative_to(_REPO_ROOT)) for path in sources}

    assert "DOCUMENTATION.md" in relative
    assert "uacpy/third_party/MODIFICATIONS.md" in relative
    assert "docs/models/kraken.md" in relative
    if (_REPO_ROOT / "docs" / "superpowers").is_dir():
        assert any(name.startswith("docs/superpowers/")
                   for name in relative)
    assert "uacpy/models/ram.py" in relative

    vendored = [path for path in sources
                if path.suffix == ".md" and "third_party" in path.parts
                and path.name != "MODIFICATIONS.md"]
    assert not vendored, f"vendored Markdown in the walk: {vendored}"


# ── the third drift signature: a target that only closes a block ───────────
#
# The gate above sees two of them — a blank target and one past the file's
# last line. A third is legible without reading the claim: a line that carries
# no expression at all. ``end if``, ``continue``, a bare ``end`` assert
# nothing, so nothing a citing comment says can be evidenced there, and the
# statement the claim names is a line or two away. That is precisely what an
# insertion above the citation leaves behind.
#
# The BROAD form of the rule — reject every target carrying no executable
# statement, comments included — was measured over the single-line citations
# that resolve into the vendored tree and rejected: 56 flags, around 50 of
# them legitimate. Citing a comment is often the *most* authoritative thing to
# do (``kraken.f90:212`` cites a commented-out line, which IS the evidence;
# ``bellhop.f90:50`` cites ``! NPts, Sigma not used by BELLHOP``), and 22 more
# were ``ELSE IF (OPT(I).EQ.'G')`` option-parsing arms, where the cited
# statement is itself the claim. A ~50-entry whitelist bought a handful of
# catches, which is the maintenance tax this suite's precedent refuses.
#
# The NARROW form below measured 6: four genuine drifts and the two exempted
# addresses. Three of the four cited line 100 of ``oaseun31.f``, an ``end if``
# two lines above the ``clen(m)=0e0`` they claimed; the fourth cited line 233
# of ``unoasp22.f``, an ``END IF`` one line above the ``NX=MIN0(NX,2*NP)`` it
# claimed. Those four are written out here rather than in address form on
# purpose: they were drifted, they have since been re-resolved, and a gate
# that spelled them as live citations would flag its own commentary.

# A line that is nothing but the end of a block, with the F77 statement label
# and the trailing inline comment a real one may carry. ``ELSE`` and ``GO TO
# n`` are deliberately absent: they say which arm and where control goes, so
# they can be the evidence, and ``oaseun31.f:2310`` is cited for exactly that.
_BARE_BLOCK_ENDER = re.compile(
    r"[ \t]*(?:\d+[ \t]+)?"
    r"(?:end(?:[ \t]*(?:if|do|while|select|subroutine|function|program"
    r"|module|type|block|interface)(?:[ \t]+\w+)?)?"
    r"|contains|return|continue)"
    r"[ \t]*(?:!.*)?",
    re.IGNORECASE)

# The addresses where the block-ender IS the evidence, keyed by the vendored
# address rather than by the citing site: the Python moves every time anyone
# edits above the comment, while the vendored line moves only on a re-vendor,
# which invalidates every citation into that file anyway. The one entry below
# licenses the two sites that make the same claim — the writer's option-letter
# table and the test that pins it.
_STRUCTURAL_TARGET_IS_THE_EVIDENCE = {
    'unoasn22.f:727': "the claim is that OASN's GETOPT ladder ends with a "
                      "bare END IF, dropping an unknown option letter in "
                      "silence — the empty block-ender is what a reader has "
                      "to see",
}


def _single_line_vendored_targets():
    """Yield ``(where, vendored, lineno, text)`` per single-line target.

    The traversal is the resolve gate's, over the same module-level patterns,
    reduced to what a content check needs: the addresses that name one line of
    one unambiguously resolved vendored file. What that gate counts and
    reports on — ``external:`` markings, ambiguous basenames, ranges, the bare
    addresses nothing binds — is dropped here rather than re-classified, so
    the two are not two accounts of one census that could disagree about it.
    """
    package = Path(_REPO_ROOT) / "uacpy"
    by_name: dict = {}
    for path in (package / "third_party").rglob("*"):
        if path.is_file():
            by_name.setdefault(path.name, []).append(path)

    body_cache: dict = {}

    def elements(address, vendored, where):
        if vendored not in body_cache:
            body_cache[vendored] = vendored.read_text(
                encoding="utf-8", errors="replace").splitlines()
        body = body_cache[vendored]
        for element in address.split(","):
            if "-" in element:
                continue            # a range spans structure legitimately
            target = int(element)
            if target <= len(body):
                yield where, vendored, target, body[target - 1]

    def continuations(line, at, vendored, name, prefix):
        """The bare ``:NNN`` addresses continuing the one that ended at ``at``.

        Returns the position past the last of them, so the caller can see
        whether the line ends mid-address, and the targets they name — none
        when the citation they continue was itself skipped.
        """
        found = []
        while True:
            more = _BARE_CONTINUATION.match(line, at)
            if more is None:
                return at, found
            if vendored is not None:
                found.extend(elements(
                    more.group(1), vendored,
                    f"{prefix} :{more.group(1)} (continuing {name})"))
            at = more.end()

    for path in _citing_sources():
        wrapping = None
        for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1):
            prefix = f"{path.relative_to(_REPO_ROOT)}:{lineno}:"
            carry = None

            resumed = wrapping and _WRAPPED_CONTINUATION.match(line)
            if resumed:
                vendored, name = wrapping
                yield from elements(
                    resumed.group(1), vendored,
                    f"{prefix} :{resumed.group(1)} (continuing {name})")
                at, found = continuations(line, resumed.end(), vendored,
                                          name, prefix)
                yield from found
                if _CONTINUATION_OPENER.search(line[at:]):
                    carry = wrapping
            wrapping = None

            for match in _VENDORED_CITATION.finditer(line):
                where = f"{prefix} {match.group(0)}"
                wanted = Path(match.group(2)).parts
                hits = [p for p in by_name.get(wanted[-1], ())
                        if p.parts[-len(wanted):] == wanted]
                vendored = None
                if not match.group(1) and len(hits) == 1:
                    vendored = hits[0]
                    yield from elements(match.group(3), vendored, where)
                at, found = continuations(line, match.end(), vendored,
                                          match.group(2), prefix)
                yield from found
                if vendored is not None and _CONTINUATION_OPENER.search(
                        line[at:]):
                    carry = (vendored, match.group(2))
            wrapping = carry


def test_no_citation_resolves_to_a_bare_block_ender():
    """A single-line target must carry an expression, not just close a block.

    ``end if``, ``end do``, ``continue``, ``return``, ``CONTAINS``, a bare
    ``end``: a line that states nothing cannot be what a comment's claim rests
    on, so an address that lands on one has drifted off the statement it was
    written for. Ranges are exempt — a range spanning a block-ender is
    ordinary — and so are the addresses in
    ``_STRUCTURAL_TARGET_IS_THE_EVIDENCE``, where the empty block IS the
    claim.

    **What this cannot see, which is the point.** No address check can tell
    that a citation resolves to the wrong *statement*. A blank target, a
    target past the file's end and a bare block-ender are the entire set of
    drift signatures a machine can read without understanding the claim;
    everything else lands on ordinary code and passes. Measured when this rule
    was written: of 26 addresses shifted by one patch to a vendored file, the
    resolve gate above caught 1 — the one that happened to land on a blank
    line — and this rule would have caught 1 more. The other 24 pointed at
    real code that had nothing to do with their claims, and nothing here can
    find them. That is why ``docs/DEV.md`` §9.1 requires a patch to a vendored
    file to be followed by re-resolving every citation into it *from the
    claim*, and why these gates are a floor rather than the check.
    """
    examined, flagged, exempt_seen = 0, [], set()
    for where, vendored, lineno, text in _single_line_vendored_targets():
        examined += 1
        if not _BARE_BLOCK_ENDER.fullmatch(text):
            continue
        address = f"{vendored.name}:{lineno}"
        if address in _STRUCTURAL_TARGET_IS_THE_EVIDENCE:
            exempt_seen.add(address)
            continue
        flagged.append(f"{where} — line {lineno} is {text.strip()!r}, which "
                       f"closes a block and states nothing")

    assert not flagged, (
        "citation(s) whose target only closes a block — the statement the "
        "claim names is a line or two away, which is what an insertion above "
        "the citation does to an address. Re-resolve from what the comment "
        "claims (DEV.md §9.1), not by adding the offset "
        f"({examined} single-line targets examined):\n" + "\n".join(flagged)
    )
    assert examined >= _MIN_CITATIONS_CHECKED, (
        f"the rule reached only {examined} single-line targets, under the "
        f"{_MIN_CITATIONS_CHECKED} floor the resolve gate uses over the same "
        f"population — the walk or the patterns have stopped covering the tree"
    )
    stale = set(_STRUCTURAL_TARGET_IS_THE_EVIDENCE) - exempt_seen
    assert not stale, (
        "exemption(s) nothing cites any more, or that no longer resolve to a "
        "block-ender — drop them rather than leaving a licence standing over "
        f"an address: {sorted(stale)}"
    )


# ── Markdown citations ─────────────────────────────────────────────────────
#
# Vendored Fortran is frozen between re-vendors, so a line number into it is a
# stable address. uacpy's own Markdown is not: it is edited whenever anything
# it describes changes, and every insertion above a citation moves it. When
# these two gates were written, 16 of the 35 ``*.md:NNN`` citations in the
# package pointed somewhere other than their evidence — the citing comment for
# line 1745 of ``DOCUMENTATION.md`` claimed a bandwidth formula that line had
# not carried for a long time, and two addresses had drifted onto blank lines.
#
# So an own-doc citation names a numbered section and quotes the sentence it
# leans on: ``docs/models/kraken.md §6.5 "asking for leaky ones finds 27"``.
# The quote is the address — it moves with the text instead of away from it —
# and the section is how a reader gets there. This is the Markdown form of the
# rule ``test_no_comment_pins_a_line_number_in_another_python_file`` applies to
# Python (DEV.md §10: name the symbol, not the line).


def test_no_comment_pins_a_line_number_in_the_projects_own_docs():
    """No comment or docstring addresses ``docs/models/kraken.md`` by line.

    uacpy's own Markdown moves under its citations the same way its Python
    does, and the measurement that prompted this gate says it moves *faster*.
    Reading every pin against the current documents, 16 of 35 pointed at text
    that did not support the citing comment; re-derived independently against
    git history — was this address right when it was *written*? — the rate came
    out higher still, 18 of 30. Either number dwarfs the 4 of 7 that prompted
    the sibling gate for Python. One pin addressed a Markdown table separator
    ``|---|---|---|``, which was useless from birth. Cite
    ``file.md §N "a phrase from it"`` instead — checked by
    :func:`test_every_doc_section_anchor_resolves_and_carries_its_quote`.

    Exactly two addresses are let through, and neither is let through for
    being unreadable. Markdown under ``uacpy/third_party`` keeps its line
    numbers — it is vendored, frozen between re-vendors, and read by
    :func:`test_vendored_citations_resolve_and_single_line_targets_carry_code`
    like the Fortran beside it. And an ``external:``-marked address is the
    DEV.md §10 escape hatch for Markdown that lives outside this repo, where
    nothing here could check a section either.

    Everything else is an offence **whether or not the path resolves**. An
    earlier draft skipped what it could not resolve, borrowing that rule from
    the Python gate, where an unresolvable name really does mean "not ours".
    For Markdown it means the opposite: ``README.md`` names four files in this
    repo, so it resolved to none of them and its line pins were waved through.
    Resolution is what it takes to *check* an address, never what it takes to
    *forbid* one.

    Does NOT catch: an own-doc line pin written *inside* a Markdown page,
    since only ``.py`` is walked here. The vendored-citation gate above hands
    such an address to this rule rather than failing on it, so what the pair
    leaves unchecked is exactly that case — one at the time of writing, a
    "Files: Modify" edit list in a finished plan under ``docs/superpowers``,
    where a line number is a target rather than a claim.
    """
    package = Path(_REPO_ROOT) / "uacpy"
    by_name = _repo_markdown()
    assert by_name, "no Markdown found in the repo — the walk is broken"

    offenders = []
    for path in sorted(package.rglob("*.py")):
        if "third_party" in path.parts:
            continue
        for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1):
            for match in _OWN_DOC_LINE_PIN.finditer(line):
                verdict = _line_pin_verdict(
                    by_name, bool(match.group(1)), match.group(2))
                if verdict is None:
                    continue
                offenders.append(
                    f"{path.relative_to(_REPO_ROOT)}:{lineno}: "
                    f"{match.group(0)}… → {verdict}")

    assert not offenders, (
        "citation(s) pinning a line number in Markdown. Those lines move on "
        "every edit above them; name the section and quote the sentence "
        "instead — ``kraken.md §6.5 \"asking for leaky ones finds 27\"``. An "
        "address into Markdown outside this repo takes the ``external:`` "
        "prefix (DEV.md §10) so its unverifiability is visible:\n"
        + "\n".join(offenders)
    )


def test_every_doc_section_anchor_resolves_and_carries_its_quote():
    """Every ``file.md §N "phrase"`` address points at that phrase.

    Three things are read: the file resolves inside the repo, a heading
    numbered ``N`` exists in it, and the quoted phrase appears on a line
    *inside that section*. The third is the one that makes the address an
    address — when a doc is rewritten, the phrase stops matching and the gate
    names the citing comment whose premise just changed. An insertion
    somewhere above, which is what broke the line pins this replaced, moves
    nothing.

    The quote is optional — a bare ``ram.md §5`` gets the first two checks
    only — but ``_MIN_QUOTED_DOC_ANCHORS`` keeps the third from quietly
    emptying out. It has to sit on one source line here *and* match within one
    line of the document, so keep the phrase short and expect an inline
    ``**bold**`` or ``` `code` ``` marker to sit between two words that read
    as adjacent.

    Does NOT catch, in rough order of how likely each is to bite:

    * **A quote taken from the wrong place in the right section.** The sharpest
      hole, and the one that produced a real defect here: converting a *stale*
      line pin by quoting whatever now sits at that line number launders the
      drift into a form the gate cannot see, because the quote really is in
      §N. ``test_kraken.py``'s range-dependent-BROADBAND test was converted
      that way and ended up citing a sentence about sub-cutoff bins. Convert
      from the **claim**, never from the address.
    * **Coarse sections.** ``DOCUMENTATION.md §7`` is 373 lines and ``§18`` is
      322, so "the phrase is somewhere in §N" is a weak constraint there. It
      is the quote, not the section, doing the work in those files.
    * **A quote that wrapped** onto the next source line: it degrades to a
      bare section reference, silently. That is what ``_MIN_QUOTED_DOC_ANCHORS``
      watches for in aggregate — it cannot name the one that slipped.
    * **A quote matched inside a fenced code block.** The section span is not
      filtered, so an example line can stand in as evidence for prose.
    * A phrase that survives verbatim while the argument around it is
      rewritten; a claim that was never true; a section renumbered so ``§N``
      still exists but now covers something else *and* still contains the
      phrase; a document with an unbalanced fence (see :func:`_headings`).
    * Citations inside ``.md`` files, since only ``.py`` is walked; and a
      phrase against Markdown outside this repo, or one whose basename is
      ambiguous — both are skipped here, though a *line pin* to either is
      still banned by the gate above.
    """
    package = Path(_REPO_ROOT) / "uacpy"
    by_name = _repo_markdown()
    assert by_name, "no Markdown found in the repo — the walk is broken"

    body_cache: dict = {}
    resolved = quoted = 0
    unresolved, offenders = [], []

    for path in sorted(package.rglob("*.py")):
        if "third_party" in path.parts:
            continue
        for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1):
            for match in _DOC_ANCHOR.finditer(line):
                cited, section, quote = match.groups()
                where = (f"{path.relative_to(_REPO_ROOT)}:{lineno}: "
                         f"{cited} §{section}"
                         + (f" \"{quote}\"" if quote else ""))
                target = _resolve_markdown(by_name, cited)
                if target is None:
                    unresolved.append(where)
                    continue
                if target not in body_cache:
                    body_cache[target] = target.read_text(
                        encoding="utf-8").splitlines()
                body = body_cache[target]
                span = _section_span(body, section)
                if span is None:
                    offenders.append(
                        f"{where} — no heading numbered §{section} in "
                        f"{target.relative_to(_REPO_ROOT)}")
                    continue
                first, last = span
                resolved += 1
                if quote is None:
                    continue
                quoted += 1
                if not any(quote in row for row in body[first:last + 1]):
                    hit = next((i + 1 for i, row in enumerate(body)
                                if quote in row), None)
                    moved = (f"; it is on line {hit}, outside §{section} "
                             f"(lines {first + 1}-{last + 1})" if hit else
                             "; it is nowhere in the file")
                    offenders.append(
                        f"{where} — the quoted phrase is not in "
                        f"§{section} of "
                        f"{target.relative_to(_REPO_ROOT)}{moved}")

    coverage = (f"{resolved} anchors resolved, {quoted} of them quoted")
    assert not offenders, (
        "doc citation(s) whose evidence has moved or been rewritten — reread "
        "the section and requote it, or move the citation to the section that "
        f"now carries the claim ({coverage}):\n" + "\n".join(offenders)
    )
    assert not unresolved, (
        "doc citation(s) naming a Markdown file this repo does not contain "
        f"({coverage}):\n" + "\n".join(unresolved)
    )
    assert resolved >= _MIN_DOC_ANCHORS_RESOLVED, (
        f"the doc-anchor gate resolved only {resolved} citations, under the "
        f"{_MIN_DOC_ANCHORS_RESOLVED} floor — the citation form or the walk "
        f"has drifted and the gate is checking almost nothing ({coverage})"
    )
    assert quoted >= _MIN_QUOTED_DOC_ANCHORS, (
        f"only {quoted} doc anchors carried a checkable quote, under the "
        f"{_MIN_QUOTED_DOC_ANCHORS} floor — the strong half of this gate has "
        f"lapsed into checking that section numbers exist ({coverage})"
    )


# ── the `filterwarnings` policy declared in pyproject.toml ─────────────
#
# Every test below probes the *live* filter stack. Inside a pytest item that
# stack is exactly the ini's entries (pytest applies them over a
# `simplefilter('always')` base), so `warnings.warn_explicit` — which takes
# the attributed module name as an argument instead of deriving it from the
# calling frame — can ask "what would this filter set do to a warning raised
# *there*?" without having to make the real third-party call.
#
# Consequence: these fail if run against a different ini, and they are meant
# to. They are the only thing standing between a scoped filter and a future
# change quietly widening it back to a blanket ignore.


# The exact text NumPy 2.5 emits from the deprecated `ndarray.shape` setter;
# the pyproject entries match on its leading clause.
_SHAPE_MESSAGE = (
    'Setting the shape on a NumPy array has been deprecated in NumPy 2.5.\n'
    'As an alternative, you can create a new view using np.reshape '
    '(with copy=False if needed).'
)


def _emit(category, module, message='probe'):
    """Raise ``category`` attributed to ``module`` and return what got out.

    Returns the list of recorded warnings — empty when the filter stack
    ignored it. Propagates the warning as an exception when the stack says
    ``error``. A private ``registry`` keeps the once/default bookkeeping out
    of any real module's ``__warningregistry__``.
    """
    with warnings.catch_warnings(record=True) as seen:
        warnings.warn_explicit(message, category, f'{module}.py', 1,
                               module=module, registry={})
    return seen


# ─── netCDF4 `.shape` deprecation: scope of the ignore ──────────────────────


def test_shape_deprecation_from_a_uacpy_test_module_is_ignored():
    # The exemption's whole purpose: uacpy's tests write synthetic .nc caches
    # through netCDF4's Cython writer, which NumPy attributes to the calling
    # test frame.
    assert _emit(DeprecationWarning, 'uacpy.tests.test_data_offline',
                 _SHAPE_MESSAGE) == []


def test_shape_deprecation_from_shipped_uacpy_code_is_an_error():
    # The hole the module scope closes. Shipped uacpy never assigns
    # `ndarray.shape`; if it starts, the deprecation gate must fire rather
    # than the netCDF4 exemption swallowing it.
    with pytest.raises(DeprecationWarning):
        _emit(DeprecationWarning, 'uacpy.core.ssp', _SHAPE_MESSAGE)


def test_shape_deprecation_from_the_numpy_masked_read_path_is_ignored():
    # Second emitter of the same NumPy deprecation: reading a masked netCDF
    # variable reaches `MaskedArray.shape.__set__` in numpy/ma/core.py, which
    # NumPy attributes to numpy's own frame. Not uacpy's to fix, and report
    # noise without its own entry.
    assert _emit(DeprecationWarning, 'numpy.ma.core', _SHAPE_MESSAGE) == []


# ─── VisibleDeprecationWarning: escaping the UserWarning ignore ─────────────


def test_visible_deprecation_warning_subclasses_user_warning():
    # The premise the explicit entry exists for. If NumPy ever reparents the
    # category, `ignore::UserWarning` stops covering it and the entry — plus
    # the comment explaining it — needs rewriting.
    assert issubclass(np.exceptions.VisibleDeprecationWarning, UserWarning)


def test_visible_deprecation_from_uacpy_is_an_error():
    # Later ini entries outrank earlier ones, so the explicit error entry
    # beats `ignore::UserWarning` for uacpy-attributed frames.
    with pytest.raises(np.exceptions.VisibleDeprecationWarning):
        _emit(np.exceptions.VisibleDeprecationWarning, 'uacpy.data.woa')


def test_visible_deprecation_from_a_dependency_stays_non_fatal():
    # Same scoping as the rest of the deprecation gate: a third-party
    # library's own visible deprecation is not uacpy's to fail CI over.
    assert _emit(np.exceptions.VisibleDeprecationWarning, 'scipy.signal') == []


_INSTALL_SH = _REPO_ROOT / "install.sh"


def _source_function(name, extra=""):
    """Bash source text for one ``install.sh`` function plus the colour stubs
    every one of them writes through."""
    body = _INSTALL_SH.read_text()
    match = re.search(rf"^{name}\(\) \{{.*?^\}}", body, re.S | re.M)
    assert match, f"{name}() not found in install.sh"
    return (
        "RED=''; YELLOW=''; GREEN=''; BLUE=''; NC=''\n"
        f"{extra}\n{match.group(0)}\n"
    )


def _run_bash(script):
    return subprocess.run(["bash", "-c", script], capture_output=True,
                          text=True, timeout=120)


def test_install_sh_is_syntactically_valid():
    result = subprocess.run(["bash", "-n", str(_INSTALL_SH)],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


#: Printed by the harness below after install.sh's own call line. The script
#: reaches its summary block and ``exit 0`` from there, so a run that does not
#: print it is a run the build record aborted.
_SUMMARY_REACHED = "SUMMARY-REACHED"


def _build_record_harness(tmp_path, bin_root):
    """install.sh's build-record writer, its two helpers, and **its own call
    line**, under the script's own ``set -euo pipefail``.

    The call line is extracted rather than retyped: dropping the guard on it
    is one of the two ways this comes back, and a harness that spelled the
    call itself could not see that."""
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "oalib_build.log").write_text("make output\n", encoding="utf-8")
    setup = (
        f'SCRIPT_DIR="{tmp_path}"\n'
        f'BIN_ROOT="{bin_root}"\n'
        f'BUILD_LOG_DIR="{logs}"\n'
        'BUILD_RECORD_WRITTEN=""; BUILD_LOGS_COPIED=""\n'
        'OVERALL=ok\n'
        'STATUS_OALIB=ok; STATUS_BELLHOPCUDA=ok; STATUS_MPIRAMS=ok\n'
        'STATUS_RAMSURF=ok; STATUS_RAMGEO=ok; STATUS_OASES=skipped\n'
        'STATUS_DATA=skipped\n'
        'OALIB_FFLAGS="-O1 -ffast-math -funroll-loops"\n'
        'command_exists() { command -v "$1" >/dev/null 2>&1; }\n'
    )
    return (
        "set -euo pipefail\n"
        + _source_function("ensure_dir", extra=setup)
        + _extract(r"^tool_version\(\) \{.*?^\}") + "\n"
        + _extract(r"^write_build_record\(\) \{.*?^\}") + "\n"
        + _extract(r"^write_build_record(?!\()[^\n]*$") + "\n"
        + f'echo "{_SUMMARY_REACHED}"\n'
    )


def test_the_build_record_lands_next_to_the_binaries(tmp_path):
    """``uacpy/bin/`` is gitignored and machine-specific, and the per-run build
    logs live under a ``mktemp`` dir — deliberately, since a fixed ``/tmp``
    name collides between users on a shared machine and can be pre-created as
    a symlink (install.sh's own note) — which ``$TMPDIR`` clears on reboot.

    So the question a numerical result that will not reproduce on another
    machine asks first — which compiler, which flags, which commit built these
    binaries — needs an answer written where the binaries are. This drives the
    writer over a temporary ``BIN_ROOT`` and reads what it left.
    """
    bin_root = tmp_path / "bin"
    result = _run_bash(_build_record_harness(tmp_path, bin_root))
    assert result.returncode == 0, result.stderr

    record = bin_root / "BUILD_INFO.txt"
    assert record.is_file(), (
        f"install.sh wrote no BUILD_INFO.txt under BIN_ROOT; "
        f"{sorted(p.name for p in tmp_path.rglob('*'))}")
    text = record.read_text(encoding="utf-8")
    for expected in ("gfortran", "OALIB FFLAGS   : -O1 -ffast-math",
                     "outcome        : ok", "submodules:",
                     "MODIFICATIONS.md"):
        assert expected in text, (
            f"BUILD_INFO.txt does not record {expected!r}:\n{text}")

    copied = bin_root / "build_logs" / "oalib_build.log"
    assert copied.is_file() and copied.read_text() == "make output\n", (
        "the per-run build logs were not copied next to the binaries, so they "
        "live only under $TMPDIR and do not survive a reboot")


def test_an_unwritable_bin_root_leaves_a_successful_install_successful(
        tmp_path):
    """The record is strictly less important than the build it records.

    ``install.sh`` runs under ``set -euo pipefail`` and writes this after every
    component has been built, so an unwritable ``uacpy/bin/`` — root-owned by
    an earlier ``sudo ./install.sh``, a read-only checkout, a full disk —
    would abort the run there: no summary, no ``exit 0``, and a bare
    "Permission denied" as the only thing the user is told about a build that
    completely succeeded.

    Note the shape the guard has to take: ``if ! { … } > "$file"; then`` does
    **not** see a refused output redirect on a compound command (measured:
    the group reports the error and the ``if`` takes the false branch), so the
    status is captured with ``|| rc=$?`` instead.
    """
    bin_root = tmp_path / "bin"
    bin_root.mkdir()
    script = _build_record_harness(tmp_path, bin_root)
    bin_root.chmod(0o500)
    try:
        result = _run_bash(script)
    finally:
        bin_root.chmod(0o700)

    assert result.returncode == 0, (
        "an unwritable BIN_ROOT failed a run in which every component built:\n"
        f"rc={result.returncode}\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}")
    assert _SUMMARY_REACHED in result.stdout, (
        "the build record aborted before install.sh's summary block, so the "
        f"user sees no component table and no exit 0:\n{result.stdout}\n"
        f"{result.stderr}")
    assert not (bin_root / "BUILD_INFO.txt").exists()
    assert "denied" not in result.stderr.lower(), (
        "a bare shell error reached the user instead of the explanatory line; "
        f"stderr: {result.stderr!r}")


def _data_ids():
    """The dataset-id list install.sh validates ``--data`` against."""
    body = _INSTALL_SH.read_text()
    match = re.search(r'^DATA_IDS="([^"]+)"', body, re.M)
    assert match, "DATA_IDS not found in install.sh"
    return match.group(1).split()


def _validate_data(selection):
    """Run ``validate_data_selection`` on one ``--data`` value.

    Returns ``(rc, INSTALL_DATA, STATUS_DATA)`` as the script would leave them.
    """
    script = _source_function(
        "validate_data_selection",
        extra=(f'DATA_IDS="{" ".join(_data_ids())}"\n'
               f'INSTALL_DATA="{selection}"\n'
               'STATUS_DATA="skipped"\nNOTE_DATA=""\n'),
    ) + (
        "validate_data_selection; rc=$?\n"
        'echo "RC=$rc DATA=$INSTALL_DATA STATUS=$STATUS_DATA"\n'
    )
    result = _run_bash(script)
    assert result.returncode == 0, result.stderr
    line = [ln for ln in result.stdout.splitlines() if ln.startswith("RC=")][-1]
    rc, data, status = re.match(
        r"RC=(\d+) DATA=(\S*) STATUS=(\S+)", line).groups()
    return int(rc), data, status


def test_every_download_function_has_a_dataset_id():
    """``DATA_IDS`` is what ``--data`` is validated against and what ``--data
    all`` expands to, so a dataset whose id is missing from it can never be
    requested."""
    body = _INSTALL_SH.read_text()
    implemented = set(re.findall(r"^download_(\w+)\(\) \{", body, re.M))
    assert implemented, "no download_* functions found"
    assert implemented == set(_data_ids())


def test_a_typo_in_data_is_rejected_rather_than_reported_installed():
    """Unvalidated, ``--data gebko`` matched no ``data_requested`` test, left
    every download short-circuited, and still printed a green 'installed' row
    for a cache it had never filled."""
    rc, data, status = _validate_data("gebko")
    assert rc != 0
    assert data == "no", "an unknown id must not reach the download loop"
    assert status == "failed"


def test_a_valid_list_passes_through_untouched():
    rc, data, status = _validate_data("gebco,woa23")
    assert (rc, data, status) == (0, "gebco,woa23", "skipped")


def test_a_partly_valid_list_keeps_the_known_ids():
    rc, data, _ = _validate_data("gebco,gebko,woa23")
    assert rc == 0
    assert data == "gebco,woa23"


def test_the_gebco_digest_is_checked_before_the_grid_is_installed():
    """``-C -`` resumes onto whatever ``.part`` a previous run left, so a
    remote file that changed between runs would otherwise be spliced from two
    versions and moved in as a valid cached grid. The OASES tarball has been
    pinned this way all along."""
    body = _INSTALL_SH.read_text()
    assert re.search(r'^GEBCO_MD5="[0-9a-f]{32}"', body, re.M)
    gebco = re.search(r"^download_gebco\(\) \{.*?^\}", body, re.S | re.M).group(0)
    check = gebco.index("GEBCO_MD5")
    promote = gebco.index('mv -f "$tmp"')
    assert check < promote, "the digest is checked after the file is promoted"
    # A mismatch must drop the .part: resuming it again can only rebuild the
    # same bad file.
    assert 'rm -f "$tmp"' in gebco


def test_robust_curl_stops_on_a_permanent_http_error():
    """curl exit 22 with ``-f`` means the server answered >= 400.
    ``--retry-all-errors`` has already re-sent it 4 times inside curl, so the
    20-iteration outer loop turns one dead URL into ~80 requests — and
    ``download_woa23`` loops 26 files."""
    body = _INSTALL_SH.read_text()
    fn = re.search(r"^robust_curl\(\) \{.*?^\}", body, re.S | re.M).group(0)
    assert "-w '%{http_code}'" in fn, "the HTTP status is never captured"
    assert re.search(r"rc == 22", fn), "curl's HTTP-error exit is not tested"
    # Rate limiting is transient and must still go round the outer loop.
    assert '"429"' in fn and '"408"' in fn


@pytest.mark.slow
def test_robust_curl_returns_immediately_on_a_404():
    """Executed, not just read: a local server answers 404 and the call must
    come back without spending the outer retry budget."""
    import http.server
    import socketserver
    import threading
    import time

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_error(404)

        def log_message(self, *args):
            pass

    with socketserver.TCPServer(("127.0.0.1", 0), _Handler) as srv:
        port = srv.server_address[1]
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        try:
            script = _source_function("robust_curl") + (
                f'robust_curl "http://127.0.0.1:{port}/missing" /dev/null; '
                'echo "RC=$?"\n'
            )
            started = time.monotonic()
            result = _run_bash(script)
            elapsed = time.monotonic() - started
        finally:
            srv.shutdown()
            thread.join(timeout=5)
    assert "RC=1" in result.stdout
    # curl's own --retry 3 --retry-delay 5 costs ~15 s; a single outer
    # iteration adds 5 more, and the full 20-iteration budget ~400 s.
    assert elapsed < 60, f"the outer retry loop ran anyway ({elapsed:.0f}s)"


def test_ci_header_names_only_python_versions_the_project_supports():
    """The workflow header lists the configurations it does NOT cover but that
    are advertised as supported. It named 3.10 and 3.11, which ``pyproject``'s
    ``requires-python = ">=3.12"`` does not support at all."""
    workflow = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
    if not workflow.is_file():
        pytest.skip(f"{workflow} is not present (source checkout only)")
    header = []
    for line in workflow.read_text(encoding="utf-8").splitlines():
        if not line.startswith("#"):
            break
        header.append(line)
    named = set(re.findall(r"\b3\.\d+\b", "\n".join(header)))
    with (_REPO_ROOT / "pyproject.toml").open("rb") as fh:
        cfg = tomllib.load(fh)
    supported = {c.rsplit(" :: ", 1)[1]
                 for c in cfg["project"]["classifiers"]
                 if c.startswith("Programming Language :: Python :: 3.")}
    assert named <= supported, (
        f"ci.yml names Python {sorted(named - supported)}, which pyproject "
        f"does not advertise (classifiers: {sorted(supported)})")


_PYPROJECT = _REPO_ROOT / "pyproject.toml"


def _pyproject():
    return tomllib.loads(_PYPROJECT.read_text(encoding='utf-8'))


def _floor(specs, name):
    """The ``>=`` floor declared for ``name`` as a version tuple."""
    for spec in specs:
        match = re.match(rf"{re.escape(name)}>=([0-9.]+)", spec)
        if match:
            return tuple(int(p) for p in match.group(1).split('.'))
    pytest.fail(f"{name} is not pinned in {specs}")


def test_the_xdist_floor_covers_the_scheduler_addopts_asks_for():
    """``--dist=worksteal`` is the scheduler pytest-xdist added in 3.2.0; a
    resolver landing on 3.0/3.1 fails on every single invocation."""
    config = _pyproject()
    addopts = config['tool']['pytest']['ini_options']['addopts']
    if '--dist=worksteal' not in addopts:
        pytest.skip("addopts no longer selects the worksteal scheduler")
    floor = _floor(config['project']['optional-dependencies']['test'],
                   'pytest-xdist')
    assert floor >= (3, 2)


def test_the_pytest_floor_can_run_on_the_supported_pythons():
    """pytest gained Python 3.12 support in 7.3.2, and requires-python is
    >= 3.12: a 6.x floor declares an environment that cannot run the suite."""
    config = _pyproject()
    assert config['project']['requires-python'] == '>=3.12'
    assert _floor(config['project']['optional-dependencies']['test'],
                  'pytest') >= (7, 4)


# Each family's first release whose wheels cover every interpreter in
# ``classifiers``, a band below it that does not, and why. Facts about
# published artifacts, so the table does not drift;
# ``test_the_recorded_wheel_floors_are_the_first_that_cover_every_python``
# re-derives both directions from PyPI when a run asks for the network.
_FIRST_WHEEL_ON_EVERY_ADVERTISED_PYTHON = {
    'pyproj': ((3, 7, 0), '>=3.6.1,<3.7',
               "pyproj publishes no cp312 wheel before 3.6.1 and no cp313 "
               "wheel before 3.7.0"),
    'pillow': ((10, 4), '>=10.1,<10.2',
               "pillow 9.5's only cp312 wheels are win32 and win_amd64 "
               "(10.0 is the first with a Linux cp312 wheel), and no release "
               "before 10.4 publishes a cp313 wheel"),
}


def test_every_recorded_floor_has_a_wheel_on_every_advertised_python():
    """A floor is a claim about a configuration someone can build. Two of them
    named releases that cannot be installed from wheels on *either*
    interpreter in ``classifiers``: pinning to the declared minimum to
    reproduce a result, or running a lower-bound CI job, falls back to a
    source build of a compiled geodesy/imaging library or fails outright.

    The bands still resolve — a resolver walks upward to a current release —
    so nothing about ``pip install uacpy`` reports this. Only the floor is
    wrong, and only a gate that reads it says so."""
    core = _pyproject()['project']['dependencies']
    for name, (first, _below, why) in (
            _FIRST_WHEEL_ON_EVERY_ADVERTISED_PYTHON.items()):
        declared = _floor(core, name)
        # Pad the declared floor to the reference's length rather than
        # truncating the reference to the declared one: ``pillow>=10`` is
        # ``(10,)``, and a truncated ``(10,)`` reference would admit it —
        # while pillow 10.0 through 10.3 publish no cp313 wheel at all.
        padded = declared + (0,) * (len(first) - len(declared))
        assert padded >= first, (
            f"{name} floor is {'.'.join(map(str, declared))}, below "
            f"{'.'.join(map(str, first))}: {why}")


def _advertised_pythons():
    """Every ``Programming Language :: Python :: 3.x`` in ``classifiers``, as
    version tuples, lowest first."""
    return sorted(
        tuple(int(part) for part in c.rsplit(" :: ", 1)[1].split("."))
        for c in _pyproject()['project']['classifiers']
        if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", c))


def _resolves_from_wheels(band, python, tmp_path):
    """Whether pip can resolve ``band`` for ``python`` on manylinux, wheels
    only. pip owns the PEP 425 tag matching, abi3 and the glibc rules, so it
    is asked rather than reimplemented."""
    import shutil
    import sys

    target = tmp_path / f"{band}-{python}".replace("/", "_")
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "download", "--no-deps",
         "--only-binary=:all:", "--implementation", "cp",
         "--python-version", ".".join(map(str, python)),
         "--platform", "manylinux_2_17_x86_64", "-d", str(target), band],
        capture_output=True, text=True, timeout=600)
    shutil.rmtree(target, ignore_errors=True)
    return proc.returncode == 0


def _floor_band(spec, name):
    """``name>=floor,<next-minor`` — the floor's own minor series, which is
    what "pin to the declared minimum" resolves to."""
    floor = _floor([spec], name)
    return (f"{name}>={'.'.join(map(str, floor))},"
            f"<{floor[0]}.{floor[1] + 1}")


@pytest.mark.requires_network
@pytest.mark.slow
def test_every_declared_floor_resolves_from_wheels_on_the_lowest_python(
        tmp_path):
    """Every declared floor's own minor series must resolve from wheels on the
    lowest interpreter in ``classifiers``. A family that fails here declares a
    minimum environment nobody can build on the primary supported platform —
    which ``pip install uacpy`` never reports, because a resolver walks upward
    to a current release."""
    lowest = _advertised_pythons()[0]
    unbuildable = [
        band for band in (
            _floor_band(spec, re.split(r"[<>=!~\[]", spec, maxsplit=1)[0])
            for spec in _pyproject()['project']['dependencies'])
        if not _resolves_from_wheels(band, lowest, tmp_path)]
    assert not unbuildable, (
        f"declared floor band(s) with no wheel for Python "
        f"{'.'.join(map(str, lowest))} on manylinux: " + "; ".join(unbuildable))


@pytest.mark.requires_network
@pytest.mark.slow
def test_the_recorded_wheel_floors_are_the_first_that_cover_every_python(
        tmp_path):
    """The measurement behind ``_FIRST_WHEEL_ON_EVERY_ADVERTISED_PYTHON``,
    run against PyPI, in both directions.

    Upward: the recorded release resolves on **every** interpreter in
    ``classifiers``, not just the lowest — the table's claim is about all of
    them, and a gate that only asked about the lowest could not re-derive the
    half that made 3.7.0 and 10.4 the answer rather than 3.6.1 and 10.1.

    Downward: the recorded band below it fails on at least one of them, so a
    floor cannot be raised past what the evidence supports and still pass."""
    pythons = _advertised_pythons()
    assert len(pythons) >= 2, pythons
    problems = []
    for name, (first, below, why) in (
            _FIRST_WHEEL_ON_EVERY_ADVERTISED_PYTHON.items()):
        band = (f"{name}>={'.'.join(map(str, first))},"
                f"<{first[0]}.{first[1] + 1}")
        missing = [p for p in pythons
                   if not _resolves_from_wheels(band, p, tmp_path)]
        if missing:
            problems.append(
                f"{band} has no wheel for Python "
                f"{', '.join('.'.join(map(str, p)) for p in missing)} — "
                f"the recorded floor is too low")
        covered = [p for p in pythons
                   if _resolves_from_wheels(f"{name}{below}", p, tmp_path)]
        if len(covered) == len(pythons):
            problems.append(
                f"{name}{below} resolves on every advertised Python, so the "
                f"recorded floor {'.'.join(map(str, first))} is higher than "
                f"the evidence for it: {why}")
    assert not problems, "\n".join(problems)


def test_copernicusmarine_is_an_extra_not_a_core_dependency():
    """It is credential-gated, imported at the point of use, and drags
    xarray/dask/zarr/boto3 in behind it — so it is not paid for by an install
    that will never call a Copernicus fetcher."""
    config = _pyproject()
    core = config['project']['dependencies']
    assert not any(spec.startswith('copernicusmarine') for spec in core)
    extras = config['project']['optional-dependencies']
    assert any(spec.startswith('copernicusmarine')
               for spec in extras['copernicus'])


def _extract(pattern):
    match = re.search(pattern, _INSTALL_SH.read_text(), re.S | re.M)
    assert match, f"{pattern} not found in install.sh"
    return match.group(0)


def _parse_args(*args):
    """Run install.sh's argument loop alone; return ``(rc, settings)``."""
    script = (
        "set -euo pipefail\n"
        "RED=''; YELLOW=''; GREEN=''; BLUE=''; NC=''\n"
        "AUTO_YES=0; FORCE=0; BUILD_MODELS=1\n"
        "BELLHOP_VERSION=fortran; INSTALL_OASES=ask; INSTALL_DATA=no\n"
        "print_help() { echo '(help)'; }\n"
        + _extract(r"^require_value\(\) \{.*?^\}") + "\n"
        + _extract(r"^while \[\[ \$# -gt 0 \]\]; do.*?^done$") + "\n"
        'echo "YES=$AUTO_YES BELLHOP=$BELLHOP_VERSION OASES=$INSTALL_OASES '
        'DATA=$INSTALL_DATA MODELS=$BUILD_MODELS"\n'
    )
    result = subprocess.run(["bash", "-c", script, "install.sh", *args],
                            capture_output=True, text=True, timeout=120)
    settings = dict(
        pair.split('=', 1)
        for line in result.stdout.splitlines() if line.startswith('YES=')
        for pair in line.split()
    )
    return result.returncode, settings


def test_a_flag_shaped_token_is_not_swallowed_as_a_value():
    """``--data`` consumed the next token unconditionally, so ``--data --yes``
    set INSTALL_DATA='--yes' and silently dropped the -y — committing a
    multi-hour run to choices the user never made."""
    rc, _ = _parse_args("--data", "--yes")
    assert rc != 0


def test_an_unknown_flag_stops_the_run():
    """It only warned and carried on into a build measured in hours."""
    rc, _ = _parse_args("--recompile-everythng")
    assert rc != 0


def test_the_documented_invocations_parse():
    rc, settings = _parse_args("-y", "--data", "all", "--bellhop", "cuda")
    assert rc == 0
    assert settings['YES'] == '1'
    assert settings['DATA'] == 'all'
    assert settings['BELLHOP'] == 'cuda'

    rc, settings = _parse_args("--no-models", "--oases", "no")
    assert rc == 0
    assert (settings['MODELS'], settings['OASES']) == ('0', 'no')


def _overall(*statuses):
    """The verdict install.sh reaches for seven component statuses."""
    assignments = "\n".join(
        f'{name}="{value}"' for name, value in zip(
            ("STATUS_OALIB", "STATUS_BELLHOPCUDA", "STATUS_OASES",
             "STATUS_MPIRAMS", "STATUS_RAMSURF", "STATUS_RAMGEO",
             "STATUS_DATA"), statuses))
    script = (
        "RED=''; YELLOW=''; GREEN=''; NC=''\n" + assignments + "\n"
        + _extract(r'^OVERALL="ok"\n.*?^fi$') + "\n"
        'echo "OVERALL=$OVERALL"\n'
    )
    result = subprocess.run(["bash", "-c", script], capture_output=True,
                            text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    banner = [ln for ln in result.stdout.splitlines() if 'UACPY' in ln]
    overall = [ln for ln in result.stdout.splitlines()
               if ln.startswith('OVERALL=')][-1]
    return overall.split('=', 1)[1], banner[0]


def test_a_hard_failure_is_not_reported_as_a_warning():
    """The verdict assigned "partial" for a failed row too, so a component that
    never built printed 'finished with warnings'."""
    verdict, banner = _overall("failed", "ok", "ok", "ok", "ok", "ok", "ok")
    assert verdict == "failed"
    assert "FAILED" in banner


def test_a_partial_row_is_a_warning():
    verdict, banner = _overall("ok", "partial", "ok", "ok", "ok", "ok", "ok")
    assert verdict == "partial"
    assert "finished with warnings" in banner


def test_a_failure_outranks_a_partial():
    verdict, _ = _overall("partial", "failed", "ok", "ok", "ok", "ok", "ok")
    assert verdict == "failed"


def test_a_clean_run_is_reported_clean():
    verdict, banner = _overall(*["ok"] * 6, "skipped")
    assert verdict == "ok"
    assert "completed" in banner


def test_every_function_keeps_its_locals_to_itself():
    """A bare assignment in a function writes the caller's scope; the script
    runs under ``set -u`` with dozens of globals named for what they hold."""
    body = _INSTALL_SH.read_text()
    leaked = []
    for match in re.finditer(r"^(\w+)\(\) \{\n(.*?)^\}", body, re.S | re.M):
        name, fn_body = match.group(1), match.group(2)
        for line in fn_body.splitlines():
            assignment = re.match(r"    (\w+)=\"\$[1-9]\"", line)
            if assignment:
                leaked.append(f"{name}: {line.strip()}")
    assert not leaked, f"positional args assigned without local: {leaked}"


# ── a docstring the interpreter cannot see ──────────────────────────────────
#
# A module whose first statement is an import and whose "docstring" comes after
# it has no docstring at all: the string is a bare expression statement,
# `__doc__` is None, and `help()`, a doc gate and every other tool see nothing.
# It reads as documentation to a human and does not exist to anything else.
#
# Seven files in `uacpy/tests/` were in exactly this shape, all from one past
# edit that inserted a line at position 1. The cost of never repeating it is
# this walk.

def _shadowed_module_docstring(tree):
    """Line number of a module-level bare string in a module that has no
    docstring, or ``None`` when the module is well formed.

    A module with a real docstring is fine however many strings follow it, and
    a module with no strings at all is fine; the defect is specifically a
    string sitting at module level with nothing claiming it.
    """
    if not tree.body or ast.get_docstring(tree) is not None:
        return None
    for node in tree.body:
        if (isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            return node.lineno
    return None


def test_no_module_hides_its_docstring_behind_an_import():
    """Every ``.py`` uacpy ships states its docstring where Python reads it.

    Vendored ``third_party/`` is excluded: it is upstream code and rewriting it
    would fork the tree.
    """
    offenders, scanned = [], 0
    for path in sorted((_REPO_ROOT / "uacpy").rglob("*.py")):
        if "third_party" in path.parts:
            continue
        scanned += 1
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:                       # pragma: no cover - not ours
            continue
        lineno = _shadowed_module_docstring(tree)
        if lineno is not None:
            offenders.append(
                f"{path.relative_to(_REPO_ROOT)}:{lineno}: module-level string "
                f"with no docstring above it — __doc__ is None")

    # Silence must mean "every file was well formed", never "the walk found
    # nothing". 322 files were in scope when this was written.
    assert scanned > 250, (
        f"only {scanned} modules were scanned — the package layout has changed "
        f"and this gate needs updating")
    assert not offenders, (
        "module docstring(s) the interpreter cannot see — move the import below "
        "the docstring:\n" + "\n".join(offenders))


def test_the_shadowed_docstring_detector_fires_on_the_shape_it_hunts():
    """Anti-vacuity: the gate above passes on a clean tree, so the detector
    itself is pinned against the broken shape and the two well-formed ones it
    must not confuse with it."""
    broken = ast.parse("import warnings\n'''Looks like a docstring.'''\n")
    assert _shadowed_module_docstring(broken) == 2

    proper = ast.parse("'''A real docstring.'''\nimport warnings\n")
    assert _shadowed_module_docstring(proper) is None

    # A real docstring followed by other strings is still well formed.
    trailing = ast.parse("'''Real.'''\nimport warnings\n'''incidental'''\n")
    assert _shadowed_module_docstring(trailing) is None

    # No strings at all is well formed.
    assert _shadowed_module_docstring(ast.parse("import warnings\n")) is None


# ─────────────────────────────────────────────────────────────────────────────
# Private names crossing a package boundary
# ─────────────────────────────────────────────────────────────────────────────

#: ``(importer, source module, name)`` for every underscore-prefixed name
#: imported across a top-level package boundary, with the reason it is not a
#: public one. A leading underscore in uacpy means "not public API", not
#: "module-private" — so what the convention cannot say is *which* privates
#: other packages depend on. This list says it, and renaming any of these
#: names is a cross-package change.
_CROSS_PACKAGE_PRIVATES = {
    # The packaging convention: a dunder read from the generated file.
    ('__init__.py', 'uacpy._version', '__version__'),
    # The core -> visualization inversion (docs/DEV.md section 7). Every one
    # of these sits inside a .plot() body; hoisting one to module scope makes
    # ``import uacpy`` raise ImportError.
    ('core/_grid.py', 'uacpy.visualization.plots.environment',
     '_plot_range_profile'),
    ('core/environment.py', 'uacpy.visualization.plots.environment',
     '_plot_environment'),
    ('core/ssp.py', 'uacpy.visualization.plots.environment', '_plot_ssp'),
    ('core/results/field.py', 'uacpy.visualization.plots._common',
     '_draw_result_credit'),
    # Hamilton-Bachman grain-size coefficients: one definition, in the
    # sediment model that owns the relation, read by the dataset reader that
    # applies it.
    ('data/graw_local.py', 'uacpy.core.sediment', '_HB_PHI'),
    ('data/graw_local.py', 'uacpy.core.sediment', '_HB_RHO'),
    # The set of acoustic_type values that carry no geoacoustic parameters.
    # Defined beside the carrier that validates them; every writer and every
    # wrapper that skips a geoacoustic block reads the same set.
    ('io/oalib_writer.py', 'uacpy.core.bottom', '_NON_GEOACOUSTIC_TYPES'),
    ('models/kraken.py', 'uacpy.core.bottom', '_NON_GEOACOUSTIC_TYPES'),
    ('models/ram.py', 'uacpy.core.bottom', '_NON_GEOACOUSTIC_TYPES'),
    # Carrier validators applied at the file boundary so a deck written from
    # raw arguments gets the same guard a carrier would have applied.
    ('io/oalib_writer.py', 'uacpy.core._carrier_validate', '_sanitize_title'),
    ('io/refl_io.py', 'uacpy.core._carrier_validate',
     '_require_strictly_increasing'),
    ('acoustic_signal/_signal_validate.py', 'uacpy.core._carrier_validate',
     '_require_strictly_increasing'),
    # The union-and-dedupe of ``data_sources`` over several carriers, whose
    # docstring declares itself the single home for it: Bottom aggregates over
    # its columns, Surface over its nodes, Environment over its five carriers,
    # and the range-dependent SSP assembly over its columns.
    ('data/sound_speed.py', 'uacpy.core._carrier_validate',
     '_dedupe_provenance'),
    # The verbosity-threshold resolver behind ``verbose=``, shared so a model
    # and the logger agree on what a level means.
    ('models/base.py', 'uacpy._log', '_resolve_threshold'),
    # The on-demand dataset cache, read by the basemap renderer to draw from
    # what is already downloaded rather than fetching again.
    ('visualization/basemap.py', 'uacpy.data', '_cache'),
}


def _cross_package_private_imports():
    """Every ``from <other package> import _name`` in the shipped tree."""
    import uacpy
    package = Path(uacpy.__file__).resolve().parent

    def owner(relative: str) -> str:
        head = relative.split('/')[0]
        return head if '/' in relative else '<top>'

    found = set()
    for path in sorted(package.rglob('*.py')):
        relative = path.relative_to(package).as_posix()
        if relative.split('/')[0] in ('third_party', 'tests', 'examples'):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            if not node.module.startswith('uacpy'):
                continue
            source = node.module[len('uacpy.'):] if node.module != 'uacpy' \
                else '<top>'
            source_owner = source.split('.')[0] if source != '<top>' \
                else '<top>'
            if source_owner == owner(relative):
                continue
            for alias in node.names:
                if alias.name.startswith('_'):
                    found.add((relative, node.module, alias.name))
    return found


def test_no_undocumented_private_name_crosses_a_package_boundary():
    """A leading underscore says "not public API"; it says nothing about
    which package may import the name.

    So the convention cannot answer the question a maintainer renaming a
    private actually has — who depends on this? — and the answer is spread
    across five packages. This list is that answer, one entry per dependency
    with its reason.

    BLIND SPOT, stated because a green run here is not coverage: only
    ``from … import _name`` is read. A private reached as an attribute
    (``file_manager._tmpfs_available()``, which ``parallel.py`` does) never
    appears in an ``ImportFrom`` node and is invisible here."""
    found = _cross_package_private_imports()
    added = sorted(found - _CROSS_PACKAGE_PRIVATES)
    dropped = sorted(_CROSS_PACKAGE_PRIVATES - found)
    assert not added, (
        f"private name(s) newly crossing a package boundary: {added}. Add "
        f"each to _CROSS_PACKAGE_PRIVATES with the reason it is not public, "
        f"or import a public name instead.")
    assert not dropped, (
        f"_CROSS_PACKAGE_PRIVATES names dependencies that no longer exist: "
        f"{dropped}. Drop them.")


def test_dev_md_points_at_the_list_rather_than_restating_it():
    """``docs/DEV.md``'s underscore convention sends the reader to the list
    above instead of summarising it.

    A count or a set of directions written into that prose is a second copy
    that nothing checks — the bullet said "Fifteen" and named six of the eight
    package pairs, both already wrong. So the prose is held to naming the two
    identifiers a reader needs to find, and to carrying no total."""
    dev_md = _REPO_ROOT / 'docs' / 'DEV.md'
    if not dev_md.is_file():
        pytest.skip('docs/DEV.md is not present')
    text = dev_md.read_text(encoding='utf-8')

    bullets = [b for b in text.split('\n- ')
               if 'leading underscore' in b.split('\n')[0]]
    assert len(bullets) == 1, 'expected exactly one underscore-convention bullet'
    bullet = bullets[0]

    assert '_CROSS_PACKAGE_PRIVATES' in bullet
    assert 'test_no_undocumented_private_name_crosses_a_package_boundary' in bullet

    # A cardinal applied to the things the list holds is a claim about the
    # list's size that only the list can settle. ``one`` is excluded from the
    # alternation: in this bullet it is an article ("in one place", "the same
    # edit"), not a total.
    counted = re.findall(
        r'\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|'
        r'thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|'
        r'twenty|\d+)\b[^.]{0,60}?\b(names?|imports?|privates?|entries|'
        r'pairs?|directions?|dependencies|sites?)\b',
        bullet, flags=re.IGNORECASE)
    assert not counted, (
        f"docs/DEV.md's underscore bullet states a count {counted} that "
        f"nothing gates; point at _CROSS_PACKAGE_PRIVATES instead")


def test_the_private_name_sweep_reads_the_shipped_tree():
    """The sweep itself, so an empty collection cannot pass as a green gate."""
    found = _cross_package_private_imports()
    assert len(found) >= 10
    assert ('core/ssp.py', 'uacpy.visualization.plots.environment',
            '_plot_ssp') in found


# ── MODIFICATIONS.md diff blocks vs the vendored source they describe ──────

#: ``## <heading>`` in ``MODIFICATIONS.md`` -> the ``third_party/`` directory
#: it documents. A heading not listed here has no vendored tree to check its
#: blocks against, and its diff blocks are counted as skipped rather than
#: guessed at.
_MODIFICATION_ROOTS = {
    'Acoustics Toolbox': 'Acoustics-Toolbox',
    'ramsurf': 'ramsurf',
    'ramgeo': 'ramgeo',
    'mpiramS': 'mpiramS',
}


def _modification_diff_blocks():
    """``(target_path, block_start_line, added_lines, removed_lines)`` for each
    ```diff`` block whose section names a vendored file that exists.

    A ``###`` heading carries the path in backticks, relative to the tree its
    ``##`` heading names. Blocks under a heading with no such path — the
    'Build system' / 'Dispatcher' / 'Enlarged array dimensions' sections —
    resolve to nothing and are returned as skips.
    """
    import re

    third_party = _REPO_ROOT / "uacpy" / "third_party"
    text = (third_party / "MODIFICATIONS.md").read_text(encoding="utf-8")
    blocks, skipped = [], 0
    root = target = None
    in_diff = False
    block, start = [], 0
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.startswith('## '):
            head = line[3:].split('(')[0].split('—')[0].split('--')[0].strip()
            root, target = _MODIFICATION_ROOTS.get(head), None
        elif line.startswith('### '):
            match = re.search(r'`([^`]+)`', line[4:])
            target = None
            if match and root:
                candidate = third_party / root / match.group(1)
                target = candidate if candidate.is_file() else None
        elif line.startswith('```diff'):
            in_diff, block, start = True, [], lineno
        elif in_diff and line.startswith('```'):
            in_diff = False
            if target is None:
                skipped += 1
            else:
                added = [ln[1:].strip() for ln in block if ln.startswith('+')]
                removed = {ln[1:].strip() for ln in block
                           if ln.startswith('-')}
                blocks.append((target, start, added, removed))
        elif in_diff:
            block.append(line)
    return blocks, skipped


def test_every_added_line_in_modifications_md_is_in_the_vendored_source():
    """The ``+`` side of each diff block is the patched source, so every one
    of those lines has to be findable in the file the block names.

    Lines that are also on the ``-`` side are skipped: a block that shows a
    line moving, or a context line the diff quotes both ways, is not a claim
    about the final state. Nothing else re-checks these blocks — the
    citation-drift gate reads ``MODIFICATIONS.md`` as a *citing source*
    (do its ``file:line`` addresses resolve), never as a diff to re-apply, so
    a ``+`` line that drifted from the code it claims to show is invisible.
    """
    blocks, _ = _modification_diff_blocks()
    stale = []
    for target, start, added, removed in blocks:
        source = {ln.strip() for ln
                  in target.read_text(encoding='utf-8',
                                      errors='replace').splitlines()}
        for line in added:
            if line and line not in removed and line not in source:
                stale.append(f"{target.name} (block at "
                             f"MODIFICATIONS.md line {start}): {line!r}")
    assert not stale, (
        "diff block(s) in MODIFICATIONS.md show a patched line that is not in "
        "the vendored source:\n  " + "\n  ".join(stale))


def test_the_modifications_diff_sweep_reads_real_blocks():
    """A resolver that matched nothing would pass the gate above silently."""
    blocks, skipped = _modification_diff_blocks()
    assert len(blocks) >= 20, f"only {len(blocks)} diff blocks resolved"
    checked = sum(len([a for a in added if a and a not in removed])
                  for _, _, added, removed in blocks)
    assert checked >= 200, f"only {checked} added lines checked"
    assert skipped <= 5, (
        f"{skipped} diff blocks resolve to no vendored file; name the file in "
        f"the ### heading or extend _MODIFICATION_ROOTS")


# ── how much a single try block in models/ may hold ───────────────────────

#: Longest ``try`` block permitted in ``uacpy/models/``, in source lines.
#: The three that exceeded it were binary round-trips whose ``try`` held the
#: deck write, the launch, the read *and* the whole assembly of the result —
#: 136, 140 and 163 lines, against a package median well under 20. Each now
#: ends by delegating to a named assembly helper, and the longest remaining
#: block is 125. The ceiling sits above that with room to edit, and below the
#: smallest of the three, so restoring any of them fails here.
_MAX_TRY_SPAN_IN_MODELS = 135


def _try_block_spans():
    """``(span, filename, lineno)`` for every ``try`` in ``uacpy/models/``."""
    import ast

    spans = []
    for path in sorted((_REPO_ROOT / "uacpy" / "models").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                spans.append((node.end_lineno - node.lineno, path.name,
                              node.lineno))
    return sorted(spans, reverse=True)


def test_no_try_block_in_models_swallows_a_whole_run():
    """A ``try`` that spans the deck write, the launch, the read and the
    assembly puts a hundred lines of array algebra inside the scope whose
    ``finally`` deletes the work dir — and hides which of them the ``finally``
    is actually protecting.

    ``Kraken.run`` is the counter-example the finding cited: it dispatches
    more run modes than any of the three and ends by delegating.
    """
    over = [f"{name}:{line} spans {span} lines"
            for span, name, line in _try_block_spans()
            if span > _MAX_TRY_SPAN_IN_MODELS]
    assert not over, (
        f"try block(s) longer than {_MAX_TRY_SPAN_IN_MODELS} lines in "
        f"uacpy/models/: {over}. End the block by delegating to an assembly "
        f"helper, as _assemble_tl_field / _assemble_broadband_field / "
        f"_assemble_oasp_result / _read_and_assemble do.")


def test_the_try_span_sweep_reads_real_blocks():
    """A walk that found nothing would pass the ceiling silently."""
    spans = _try_block_spans()
    assert len(spans) >= 20, f"only {len(spans)} try blocks found"
    assert spans[0][0] >= 50, (
        f"the longest try block is {spans[0][0]} lines — the sweep is either "
        f"reading the wrong tree or the ceiling is no longer meaningful")


def test_each_binary_round_trip_ends_by_delegating():
    """The four assembly helpers exist and are reached from the ``run`` path.

    Named individually so deleting one and inlining it back — which the line
    ceiling alone would not necessarily catch, since each is under it — fails
    here instead.
    """
    from uacpy.models.bellhop import Bellhop
    from uacpy.models.oases import OASP
    from uacpy.models.ram import RAM

    assert hasattr(RAM, '_assemble_tl_field')
    assert hasattr(RAM, '_assemble_broadband_field')
    assert hasattr(OASP, '_assemble_oasp_result')
    assert hasattr(Bellhop, '_read_and_assemble')


# ── names state what the code does ──────────────────────────────────────────

#: Tokens that place a name in time rather than describing what it asserts.
#: ``..._still_runs`` / ``..._is_scored_as_before`` / ``..._no_longer_raises``
#: each describe a *diff*: they go stale the moment the history stops
#: mattering, and they tell a reader who never saw the bug nothing about the
#: contract being pinned. The measured before/after belongs in the body or the
#: docstring as evidence, never in the name.
_HISTORICAL_TOKENS = frozenset({'still', 'previously', 'anymore', 'again'})

#: The same idea spelled across two tokens.
_HISTORICAL_BIGRAMS = frozenset({('as', 'before'), ('no', 'longer'),
                                 ('used', 'to')})

#: Names where the word is a *physical* state rather than a diff. Each entry
#: is checked to still exist, so a stale exemption fails rather than widens
#: the gate silently.
_PHYSICAL_STATE_NAMES = {
    # A SPARC trace that has not decayed by t_max: "still ringing" is what the
    # trace is doing, not what the code used to do.
    'test_a_still_ringing_trace_is_reported',
}

#: ``unchanged`` / ``untouched`` are deliberately absent from the token set;
#: the gate's own docstring below carries the reason.


def _identifier_words(name: str):
    """The lowercase words of a ``snake_case`` or ``CamelCase`` identifier."""
    return [w.lower() for w in
            re.findall(r'[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|\d+', name)]


def _historical_words(name: str):
    """The banned tokens and bigrams ``name`` carries, as whole words."""
    words = _identifier_words(name)
    return (sorted(set(words) & _HISTORICAL_TOKENS)
            + [f'{a} {b}' for a, b in zip(words, words[1:])
               if (a, b) in _HISTORICAL_BIGRAMS])


def _defined_names():
    """``(relative path, lineno, name)`` for every ``def``/``class`` under
    ``uacpy/``, vendored sources excluded."""
    package = _REPO_ROOT / "uacpy"
    found = []
    for path in sorted(package.rglob("*.py")):
        if "third_party" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                found.append((str(path.relative_to(_REPO_ROOT)), node.lineno,
                              node.name))
    return found


_DEFINED_NAMES = _defined_names()


def test_the_name_sweep_reads_the_whole_package():
    """A walk that found nothing would pass the gate below silently."""
    assert len(_DEFINED_NAMES) >= 5000, (
        f"only {len(_DEFINED_NAMES)} def/class names found under uacpy/ — "
        f"the walk is reading the wrong tree")
    files = {path for path, _, _ in _DEFINED_NAMES}
    assert len(files) >= 200, f"only {len(files)} files walked"


def test_no_name_places_itself_in_time_instead_of_stating_the_behaviour():
    """``still`` / ``as before`` / ``no longer`` / ``used to`` / ``previously``
    / ``anymore`` / ``again`` in a ``def`` or ``class`` name.

    Rename to the behaviour or the invariant the name is about:
    ``test_a_track_that_stays_above_two_cells_runs``, not
    ``test_a_track_still_runs``; ``test_oast_writes_the_rounded_count_not_the
    _one_asked_for``, not ``test_oast_no_longer_writes_the_wrong_count``.
    A word that names a *physical* state goes in ``_PHYSICAL_STATE_NAMES``
    with the reason, not into the general vocabulary.

    **``unchanged`` and ``untouched`` are deliberately NOT banned here**, and
    adding them would make this gate worse rather than stricter. The test is
    what the word is *about*: in this tree these two overwhelmingly name the
    **pass-through contract** — "a real half-space is left untouched", "valid
    samples pass through unchanged", "TL at a range is unchanged *by* a finer
    output grid". That is what the code does to its input, not a claim about
    an earlier revision, and a reader who never saw the bug learns the
    contract from the name. Banning the words would flag every one of those to
    reach the few whose subject is really the code's earlier state, and the
    allowlist would end up longer than the set it protects — a shape this
    project has rejected before. Those few were found and renamed by reading
    each name against its assertions, which is not something a sweep can do.

    Matching is on identifier **words**, not substrings, and
    :func:`test_the_matcher_reads_whole_words_not_substrings` below owns that
    claim rather than this paragraph: ``against`` contains ``again``, and a
    substring test would flag every ``..._measured_against_...`` in the tree
    and teach the next reader to widen the exemptions instead of fix a name.
    """
    offenders = []
    for path, lineno, name in _DEFINED_NAMES:
        if name in _PHYSICAL_STATE_NAMES:
            continue
        hit = _historical_words(name)
        if hit:
            offenders.append(f"{path}:{lineno}: {name} — {', '.join(hit)}")
    assert not offenders, (
        f"{len(offenders)} name(s) describe a diff against an earlier state "
        f"instead of the behaviour they pin:\n" + "\n".join(offenders))


@pytest.mark.parametrize('name,expected', [
    # The substring traps. Each of these is a real shape in the tree, and a
    # naive ``'again' in name`` would flag every one of them.
    ('test_broadside_gain_against_isotropic_noise', []),
    ('TestAgainstPublishedExpressions', []),
    ('test_a_wavenumber_is_measured_against_float64', []),
    ('test_the_used_topology_is_the_resolved_one', []),
    # …and the shapes that must be caught, snake_case and CamelCase alike.
    # A physics use of the word is caught here too — the matcher only reads
    # words; ``_PHYSICAL_STATE_NAMES`` is what lets one through, in the gate.
    ('test_a_still_ringing_trace_is_reported', ['still']),
    ('test_a_track_still_runs', ['still']),
    ('TestTheDeckStillWrites', ['still']),
    ('test_it_is_scored_as_before', ['as before']),
    ('test_it_no_longer_raises', ['no longer']),
    ('test_it_used_to_clamp_to_one', ['used to']),
    ('test_it_previously_returned_nan', ['previously']),
    ('test_it_does_not_warn_anymore', ['anymore']),
    ('test_a_second_geometry_warns_again', ['again']),
])
def test_the_matcher_reads_whole_words_not_substrings(name, expected):
    """The claim the gate's docstring makes, owned by an assertion instead.

    ``against`` / ``Against`` carry ``again`` as a substring and are ordinary
    English; ``used_topology`` carries ``used`` without the bigram. A
    substring matcher would flag all three and push the next reader towards
    widening ``_PHYSICAL_STATE_NAMES`` rather than fixing a name — which is
    how an exemption list grows past the set it protects.
    """
    assert _historical_words(name) == expected


def test_every_physical_state_exemption_names_a_live_definition():
    """A renamed or deleted test must not leave its exemption behind widening
    the gate for whatever takes the name next."""
    defined = {name for _, _, name in _DEFINED_NAMES}
    stale = sorted(_PHYSICAL_STATE_NAMES - defined)
    assert not stale, (
        f"exemption(s) in _PHYSICAL_STATE_NAMES name nothing in the tree: "
        f"{stale}. Drop them, or fix the name they were meant to cover.")


_EVENT_NAMED_TEST_FILE = re.compile(r"audit|20\d{6}|round\d|batch",
                                    re.IGNORECASE)


@pytest.mark.convention
def test_no_test_file_is_named_after_the_work_session_that_produced_it():
    """Test files are named by the module or behaviour they pin, never by
    the audit, date, round, or batch that produced them: an event name says
    nothing about what breaks when the file goes red, and the tests inside
    it drift away from the per-module file the next reader searches.

    ``round\\d`` rather than ``round`` keeps roundtrip names out of the
    sweep."""
    tests_dir = Path(__file__).resolve().parent
    offenders = sorted(
        path.name for path in tests_dir.glob("test_*.py")
        if _EVENT_NAMED_TEST_FILE.search(path.name))
    assert not offenders, (
        f"test file(s) named after a work session rather than what they "
        f"pin: {offenders}. Move each test into the per-module file that "
        f"covers the same code.")


# ── the py.typed promise, held where it is already kept ─────────────────────

#: Subpackages held at zero type-checker errors. ``uacpy/noise`` is the one
#: that is already there, so gating it costs nothing and stops it drifting
#: back. The rest of the package is deliberately ungated: a frozen count over
#: the whole tree is a number nobody reads and a rubber stamp on every bump,
#: and the errors it would freeze are mostly inference friction rather than
#: wrong annotations. What the contract actually promises is measured by the
#: gates on the public surface — the lazy-import static mirror
#: (test_lazy_imports.py) and the carrier field annotations
#: (test_core_classes.py) — not by a total.
_TYPE_CLEAN_SUBPACKAGES = ("uacpy/noise",)


@pytest.mark.parametrize("subpackage", _TYPE_CLEAN_SUBPACKAGES)
def test_a_type_clean_subpackage_stays_at_zero_checker_errors(subpackage,
                                                              tmp_path):
    """``py.typed`` tells a downstream checker to believe these annotations,
    and nothing in the suite or in ``ci.yml`` ran one — so the promise had
    never been measured from inside the project.

    ``--follow-imports=silent`` keeps the scope to this subpackage: its
    imports are analysed for types but their own errors are not reported,
    which is what makes a per-subpackage zero meaningful while the rest of
    the tree is ungated.
    """
    import sys

    pytest.importorskip(
        "mypy", reason="the type-contract gate needs mypy (the `dev` extra)")
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--no-incremental",
         "--cache-dir", str(tmp_path / "mypy-cache"),
         "--follow-imports=silent", subpackage],
        cwd=str(_REPO_ROOT), capture_output=True, text=True, timeout=600)
    reported = [line for line in proc.stdout.splitlines()
                if line.startswith(f"{subpackage}/") and ": error:" in line]
    assert not reported, (
        f"{subpackage} no longer type-checks clean:\n" + "\n".join(reported))
    assert proc.returncode == 0, (
        f"mypy exited {proc.returncode} without naming a {subpackage} error:\n"
        f"{proc.stdout}\n{proc.stderr}")


# ── readers whose declared return type is a TypedDict ───────────────────────

#: Directories under ``uacpy/`` that are not shipped code.
_UNSHIPPED_DIRS = {'tests', 'examples', 'third_party', 'bin', '__pycache__'}

#: The lowest number of ``TypedDict``-annotated returns the sweep below has to
#: find. Two readers carry one today; a sweep that found none would report no
#: mismatches and pass.
_MIN_TYPED_DICT_RETURNS = 3


def _typed_dict_returns():
    """``[(site, declared keys, returned keys)]`` for every ``return {...}``
    literal in a shipped function whose return annotation names a
    ``TypedDict`` defined in the same module."""
    package = _REPO_ROOT / "uacpy"
    found = []
    for path in sorted(package.rglob("*.py")):
        if set(path.relative_to(package).parts) & _UNSHIPPED_DIRS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        declared = {
            node.name: [statement.target.id for statement in node.body
                        if isinstance(statement, ast.AnnAssign)]
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and any(getattr(base, 'id', '') == 'TypedDict'
                    or getattr(base, 'attr', '') == 'TypedDict'
                    for base in node.bases)}
        if not declared:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and isinstance(node.returns, ast.Name)
                    and node.returns.id in declared):
                continue
            for statement in ast.walk(node):
                if not (isinstance(statement, ast.Return)
                        and isinstance(statement.value, ast.Dict)):
                    continue
                keys = [key.value for key in statement.value.keys
                        if isinstance(key, ast.Constant)]
                found.append((
                    f"{path.relative_to(_REPO_ROOT)}:{statement.lineno} "
                    f"({node.name} -> {node.returns.id})",
                    declared[node.returns.id], keys))
    return found


_TYPED_DICT_RETURNS = _typed_dict_returns()


def test_the_typed_dict_return_sweep_finds_the_readers_it_checks():
    """The sweep itself, so an empty result cannot pass as agreement."""
    assert len(_TYPED_DICT_RETURNS) >= _MIN_TYPED_DICT_RETURNS, (
        f"the sweep found {len(_TYPED_DICT_RETURNS)} TypedDict-annotated "
        f"return literals, under the {_MIN_TYPED_DICT_RETURNS} this gate "
        f"requires: it is no longer reading the readers it checks")


def test_every_typed_dict_reader_returns_exactly_the_keys_it_declares():
    """``read_reflection_coefficient`` was declared ``Dict[str, np.ndarray]``
    and returns an ``int`` under ``"n_pts"``, so
    ``read_reflection_coefficient(f)["n_pts"] + 1`` is arithmetic on something
    the annotation calls an array. ``read_oast_tl`` was declared a 4-tuple and
    returns a dict, which its caller in ``uacpy/models/oases.py`` subscripts
    with ``'depths'`` and ``'metadata'``.

    A ``TypedDict`` states the keys and their value types, and this holds the
    declaration to the literal on both sides: a key added to the return and
    not to the class is invisible to a caller, and a key declared and not
    returned is a ``KeyError`` a checker says cannot happen.
    """
    mismatched = [
        f"{site}: declares {sorted(declared)}, returns {sorted(returned)}"
        for site, declared, returned in _TYPED_DICT_RETURNS
        if set(declared) != set(returned)]
    assert not mismatched, (
        "TypedDict return type(s) disagreeing with the dict the function "
        "builds:\n" + "\n".join(mismatched))


# ── parameters declared non-optional and defaulted to None ──────────────────

#: The lowest number of annotated parameters the sweep below has to reach for
#: its clean result to mean anything. A sweep that stopped reading after two
#: files would report zero offenders and pass.
_MIN_ANNOTATED_PARAMETERS_SWEPT = 2400


def _annotation_admits_none(node):
    """Whether an annotation AST node accepts ``None``.

    ``Optional[X]``, ``Union[X, None]``, ``X | None``, ``Any`` and ``object``
    all do. A string annotation (a forward reference) is parsed and asked the
    same question; one that will not parse is treated as accepting, because
    the gate is not a type checker and must not guess."""
    if node is None:
        return True
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            node = ast.parse(node.value, mode='eval').body
        except SyntaxError:
            return True
    if isinstance(node, ast.Constant) and node.value is None:
        return True
    name = (node.attr if isinstance(node, ast.Attribute)
            else node.id if isinstance(node, ast.Name) else '')
    if name in ('Any', 'object'):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return (_annotation_admits_none(node.left)
                or _annotation_admits_none(node.right))
    if isinstance(node, ast.Subscript):
        base = node.value
        generic = (base.attr if isinstance(base, ast.Attribute)
                   else getattr(base, 'id', ''))
        if generic == 'Optional':
            return True
        if generic == 'Union':
            arguments = (node.slice.elts if isinstance(node.slice, ast.Tuple)
                         else [node.slice])
            return any(_annotation_admits_none(a) for a in arguments)
    return False


def _parameters_defaulted_to_none():
    """``(offenders, annotated parameters swept, files swept)`` over the
    shipped tree: every parameter that is annotated, defaults to ``None``,
    and whose annotation does not admit ``None``."""
    package = _REPO_ROOT / "uacpy"
    offenders, annotated, files = [], 0, 0
    for path in sorted(package.rglob("*.py")):
        if set(path.relative_to(package).parts) & _UNSHIPPED_DIRS:
            continue
        files += 1
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            positional = node.args.posonlyargs + node.args.args
            padding = [None] * (len(positional) - len(node.args.defaults))
            pairs = list(zip(positional, padding + list(node.args.defaults)))
            pairs += list(zip(node.args.kwonlyargs, node.args.kw_defaults))
            for argument, default in pairs:
                if argument.annotation is None:
                    continue
                annotated += 1
                if (isinstance(default, ast.Constant) and default.value is None
                        and not _annotation_admits_none(argument.annotation)):
                    offenders.append(
                        f"{path.relative_to(_REPO_ROOT)}:{argument.lineno}: "
                        f"{node.name}({argument.arg}: "
                        f"{ast.unparse(argument.annotation)} = None)")
    return offenders, annotated, files


_NONE_DEFAULTS, _ANNOTATED_PARAMETERS, _SWEPT_FILES = (
    _parameters_defaulted_to_none())


def test_the_none_default_sweep_reads_the_whole_shipped_package():
    """The sweep itself, so a clean result cannot come from reading nothing.

    ``_EXPECTED_PACKAGES`` fixes what ships; this fixes that the sweep saw it.
    """
    assert _SWEPT_FILES >= len(_EXPECTED_PACKAGES), _SWEPT_FILES
    assert _ANNOTATED_PARAMETERS >= _MIN_ANNOTATED_PARAMETERS_SWEPT, (
        f"the sweep reached only {_ANNOTATED_PARAMETERS} annotated parameters "
        f"across {_SWEPT_FILES} files, under the "
        f"{_MIN_ANNOTATED_PARAMETERS_SWEPT} this gate requires: it is reading "
        f"less than the package it claims to")


def test_no_parameter_defaults_to_none_under_an_annotation_that_refuses_it():
    """``def __init__(self, message: str, remediation: str = None)`` declares
    that ``remediation`` is a ``str`` and then hands it ``None`` — the
    declaration is wrong at the definition, before any caller is involved.

    It matters most where it sat: six of these were on the public exception
    hierarchy, whose ``remediation``/``stdout``/``stderr`` are ``None`` in the
    common case. Every caller passing the documented ``None`` was then
    reported as passing the wrong type, so the class showed up as a pile of
    call-site errors rather than as one wrong signature.
    """
    assert not _NONE_DEFAULTS, (
        f"{len(_NONE_DEFAULTS)} parameter(s) default to None under an "
        f"annotation that does not admit it:\n" + "\n".join(_NONE_DEFAULTS))


class _CollectedItem:
    """Stand-in for a collected pytest item: a bag of marker names exposing
    the two methods ``pytest_collection_modifyitems`` calls."""

    def __init__(self, *marker_names):
        self.marker_names = set(marker_names)

    def get_closest_marker(self, name):
        return name if name in self.marker_names else None

    def add_marker(self, marker):
        self.marker_names.add(marker.name)


def _registered_marker_names():
    cfg = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    markers = cfg["tool"]["pytest"]["ini_options"]["markers"]
    return {entry.split(":", 1)[0].strip() for entry in markers}


@pytest.mark.convention
class TestConventionMarkerIsRegistered:
    """``convention`` is a registered marker, so ``-m convention`` /
    ``-m "not convention"`` select without ``--strict-markers`` warnings."""

    def test_convention_is_a_registered_marker(self):
        assert "convention" in _registered_marker_names()

    def test_the_prior_markers_are_registered_beside_it(self):
        assert {"slow", "requires_binary", "requires_oases",
                "requires_network", "benchmark"} <= _registered_marker_names()


@pytest.mark.convention
class TestRequiresOasesImpliesRequiresBinary:
    """The conftest collection hook attaches ``requires_binary`` to every
    ``requires_oases`` item — the wiring that makes
    ``-m "not requires_binary and not slow"`` exclude OASES tests."""

    def test_an_oases_item_gains_the_binary_marker(self):
        item = _CollectedItem("requires_oases")
        conftest.pytest_collection_modifyitems([item])
        assert "requires_binary" in item.marker_names

    def test_an_unmarked_item_gains_no_markers(self):
        item = _CollectedItem()
        conftest.pytest_collection_modifyitems([item])
        assert item.marker_names == set()


@pytest.mark.convention
class TestTheComposedDevTierIsDocumented:
    """README.md and docs/DEV.md both name the composed pure-Python tier
    command, and DEV.md names the gate reporting flags that keep skips
    identifiable."""

    def test_the_readme_names_the_composed_tier_command(self):
        text = (_REPO_ROOT / "README.md").read_text()
        assert '-m "not requires_binary and not slow"' in text

    def test_dev_md_names_the_composed_tier_command(self):
        text = (_REPO_ROOT / "docs" / "DEV.md").read_text()
        assert '-m "not requires_binary and not slow"' in text

    def test_dev_md_names_the_gate_reporting_flags(self):
        text = (_REPO_ROOT / "docs" / "DEV.md").read_text()
        assert "-rs --durations=50" in text


@pytest.mark.convention
class TestDevMdStatesTheMatchAnchoringRule:
    """docs/DEV.md states the ``match=`` convention: patterns cover the
    load-bearing fragment of a message, not the full sentence."""

    def test_the_rule_names_the_load_bearing_fragment(self):
        text = (_REPO_ROOT / "docs" / "DEV.md").read_text()
        assert "load-bearing fragment" in text
