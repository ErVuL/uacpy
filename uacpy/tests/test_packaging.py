"""Package-discovery guard for pyproject.toml.

Pins the "12 packages, not 222" regression: before ``namespaces = false``
and the explicit ``include``/``exclude`` lists, flat-layout discovery with
a bare ``uacpy*`` glob swept ``uacpy/bin``, ``uacpy/examples``, the whole
of ``uacpy/third_party`` and even ``uacpy_venv`` into the wheel.

The test calls ``setuptools.find_packages`` with EXACTLY the settings
parsed from ``[tool.setuptools.packages.find]`` — not a re-hardcoded
copy — so any drift in the config itself is what fails here. No build is
performed."""

import tomllib
from pathlib import Path

import pytest

setuptools = pytest.importorskip(
    "setuptools", reason="packaging guard needs the build backend"
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

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
    they guard against are present to be swept."""
    for present in ("uacpy_venv", "uacpy/third_party", "uacpy/bin",
                    "uacpy/examples", "uacpy/tests"):
        assert (_REPO_ROOT / present).is_dir(), present


def test_the_config_still_carries_the_load_bearing_keys():
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
