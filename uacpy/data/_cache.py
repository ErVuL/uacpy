"""Locator for the offline data cache (install-time-downloaded datasets).

Public datasets — the GEBCO bathymetry grid, the WOA23 climatology grids, the
DECK41 + grain-size sediment samples, the EMODnet seabed-substrate polygons and
the Natural Earth coastline — are **not bundled**. Exactly like OASES, they are
downloaded by ``install.sh`` into a gitignored cache and sampled locally; this
module only *locates* them. A missing dataset raises a typed
:class:`ConfigurationError` naming the exact install flag to run.

Cache location: ``$UACPY_DATA_CACHE`` if set, else ``<repo>/data_cache``.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from uacpy.core.exceptions import ConfigurationError

__all__ = ['cache_root', 'dataset_root', 'require', 'cached_grid', 'cached_grid_at',
           'register_cache', 'invalidate_grids', 'DATASETS']

# uacpy/uacpy/data/_cache.py → parents[2] is the repo root (editable install).
_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class _Dataset:
    """One install-time dataset: where it lives and how to install it."""
    id: str
    subdir: str
    install_flag: str
    description: str


DATASETS = {
    'gebco': _Dataset(
        'gebco', 'gebco', './install.sh --data gebco',
        'GEBCO 2025 global bathymetry grid (~7.5 GB)'),
    'woa23': _Dataset(
        'woa23', 'woa23', './install.sh --data woa23',
        'World Ocean Atlas 2023 temperature/salinity grids'),
    'sediment': _Dataset(
        'sediment', 'sediment', './install.sh --data sediment',
        'NCEI seafloor grain-size samples (grainsize.csv; the reader also picks '
        'up a hand-placed deck41.csv)'),
    'emodnet': _Dataset(
        'emodnet', 'emodnet', './install.sh --data emodnet',
        'EMODnet Geology seabed substrate polygons (Folk 5cl, European seas)'),
    'coastline': _Dataset(
        'coastline', 'coastline', './install.sh --data coastline',
        'Natural Earth land polygons (coastline backdrop, public domain)'),
    'globsed': _Dataset(
        'globsed', 'globsed', './install.sh --data globsed',
        'GlobSed v3 total sediment thickness grid (NOAA NCEI)'),
    'crust1': _Dataset(
        'crust1', 'crust1', './install.sh --data crust1',
        'CRUST1.0 layered crustal model — Vp/Vs/density (Laske et al. 2013)'),
    'diesing': _Dataset(
        'diesing', 'diesing', './install.sh --data diesing',
        'Diesing 2020 global deep-sea seafloor lithology map (CC-BY)'),
    'seaice': _Dataset(
        'seaice', 'seaice', './install.sh --data seaice',
        'NSIDC sea-ice concentration monthly climatology (public domain)'),
    'glodap': _Dataset(
        'glodap', 'glodap', './install.sh --data glodap',
        'GLODAPv2.2016b mapped seawater pH climatology (CC-BY)'),
    'wind': _Dataset(
        'wind', 'wind', './install.sh --data wind',
        'NBS 10 m wind-speed monthly climatology (NOAA, public domain)'),
    'graw': _Dataset(
        'graw', 'graw', './install.sh --data graw',
        'Graw 2021 predicted seabed bulk-density grid (CC-BY 4.0)'),
}


def cache_root() -> Path:
    """Root of the offline data cache (``$UACPY_DATA_CACHE`` or repo default)."""
    env = os.environ.get('UACPY_DATA_CACHE')
    return Path(env).expanduser() if env else _REPO_ROOT / 'data_cache'


def dataset_root(name: str) -> Path:
    """Directory a given dataset is expected to live in (may not exist yet)."""
    return cache_root() / DATASETS[name].subdir


def require(name: str, *relative: str) -> Path:
    """Resolve ``cache_root()/<subdir>/<relative...>``, or raise if absent.

    Use with no ``relative`` parts to require the dataset directory itself, or
    with parts to require a specific file inside it (e.g. a WOA23 field). The
    error names the ``install.sh`` flag that downloads the dataset.
    """
    ds = DATASETS[name]
    path = dataset_root(name).joinpath(*relative)
    if not path.exists():
        target = f"{ds.description} ({path})" if relative else ds.description
        raise ConfigurationError(
            f"Offline {name} data not found: {target}.",
            remediation=f"Run `{ds.install_flag}` to download it, or set "
                        f"$UACPY_DATA_CACHE to a directory that has it.",
        )
    return path


_GRIDS: dict = {}   # resolved path -> opened grid object


def cached_grid(name: str, filename: str, factory):
    """Open ``cache_root()/<name>/<filename>`` through ``factory`` once.

    The offline raster/NetCDF backends each wrap one large file that is opened
    once and sampled many times; this is the single memo they share, keyed on
    the resolved path so a changed ``$UACPY_DATA_CACHE`` reopens.
    """
    return cached_grid_at(require(name, filename), factory)


def cached_grid_at(path, factory):
    """Memoise ``factory(path)`` on an already-resolved path.

    For backends that locate their file themselves (a glob, a versioned name)
    rather than by a fixed filename.
    """
    key = str(path)
    if key not in _GRIDS:
        _GRIDS[key] = factory(path)
    return _GRIDS[key]


_CLEARERS: list = []   # extra drop-my-memo callables, one per backend


def register_cache(clear) -> None:
    """Register a backend's own memo so :func:`invalidate_grids` drops it too.

    Several backends memoise something that is not a grid — a KD-tree, an
    STRtree, a stack of netCDF handles — keyed on the cache root rather than on
    a file path. Each calls this at import with its own drop function so callers
    have one invalidation entry point instead of seven.
    """
    _CLEARERS.append(clear)


def invalidate_grids() -> None:
    """Drop every memo in the data layer so a freshly downloaded file is reopened.

    Covers the :func:`cached_grid` / :func:`cached_grid_at` memo (GEBCO,
    GlobSed, Graw, GLODAP) plus everything registered through
    :func:`register_cache`. A backend that has not been imported has nothing
    memoised, so an unregistered module is not a gap.
    """
    _GRIDS.clear()
    for clear in _CLEARERS:
        clear()
