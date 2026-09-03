"""Locator for the offline data cache (install-time-downloaded datasets).

Public datasets — the GEBCO bathymetry grid, the WOA23 climatology grids, the
DECK41 + grain-size sediment samples, the EMODnet seabed-substrate polygons and
the Natural Earth coastline — are **not bundled**. Exactly like OASES, they are
downloaded by ``install.sh`` into a gitignored cache and sampled locally; this
module only *locates* them. A missing dataset raises a typed
:class:`ConfigurationError` naming the exact install flag to run.

Cache location: ``$UACPY_DATA_CACHE`` if set; else ``<repo>/data_cache``
when running from a checkout; else the per-user cache directory
(``$XDG_CACHE_HOME/uacpy/data_cache`` or ``~/.cache/uacpy/data_cache``).
"""

import contextlib
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from uacpy._log import log_message
from uacpy.core.exceptions import ConfigurationError, DataFetchError

__all__ = ['cache_root', 'dataset_root', 'prepare_download', 'require',
           'require_npz', 'cached_grid',
           'cached_grid_at', 'register_cache', 'invalidate_grids', 'atomic_write',
           'staging_path', 'reading', 'memoize', 'memo_lock', 'DATASETS']

# uacpy/uacpy/data/_cache.py → parents[2] is the repo root in the documented
# clone + `pip install -e .` layout; under a plain wheel install it is
# site-packages, which must not grow a data_cache.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_cache_root() -> Path:
    """Repo ``data_cache/`` from a checkout, else the per-user cache dir.

    A checkout is recognised by ``install.sh`` sitting at the repo root —
    the script that populates the cache. Anywhere else (a wheel install
    resolving ``_REPO_ROOT`` to site-packages) the default goes to
    ``$XDG_CACHE_HOME/uacpy/data_cache`` or ``~/.cache/uacpy/data_cache``
    instead of writing into site-packages.
    """
    if (_REPO_ROOT / 'install.sh').exists():
        return _REPO_ROOT / 'data_cache'
    xdg = os.environ.get('XDG_CACHE_HOME')
    base = Path(xdg).expanduser() if xdg else Path.home() / '.cache'
    return base / 'uacpy' / 'data_cache'


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
    """Root of the offline data cache.

    ``$UACPY_DATA_CACHE`` if set, else the repo's ``data_cache/`` in a
    checkout, else the per-user cache directory (wheel installs).
    """
    env = os.environ.get('UACPY_DATA_CACHE')
    return Path(env).expanduser() if env else _default_cache_root()


def dataset_root(name: str) -> Path:
    """Directory a given dataset is expected to live in (may not exist yet)."""
    return cache_root() / DATASETS[name].subdir


def prepare_download(name: str, banner: str, *, cache_dir=None, verbose=False,
                     log_tag: Optional[str] = None) -> Path:
    """Resolve, create and announce a dataset's cache directory.

    ``cache_dir`` when given, else the dataset's own directory under the cache
    root. Created if absent, announced with ``banner``, and returned. Raises
    ``KeyError`` for a ``name`` the catalogue does not hold.
    """
    dest = Path(cache_dir) if cache_dir else dataset_root(name)
    dest.mkdir(parents=True, exist_ok=True)
    # ``log_tag`` where the channel differs from the dataset: the EMODnet
    # substrate download reports on 'seabed', the channel its reader uses.
    log_message(log_tag or name, banner, verbose=verbose)
    return dest


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


def is_installed(name: str, *relative: str) -> bool:
    """Whether a cached dataset (or one file inside it) is present.

    The predicate behind :func:`require`'s raise, for code that picks a source
    rather than failing on a missing one. An unknown dataset name is a caller
    error, not an uninstalled dataset, so it raises rather than answering
    ``False``.
    """
    if name not in DATASETS:
        raise ConfigurationError(
            f"is_installed: unknown dataset {name!r}.",
            remediation=f"Known datasets: {', '.join(sorted(DATASETS))}.")
    try:
        require(name, *relative)
    except ConfigurationError:
        return False
    return True


def require_npz(name: str, filename: str, superseding: str) -> Path:
    """Resolve a cached ``.npz`` index, refusing the pickle it replaced.

    The EMODnet polygon index and the NSIDC sea-ice climatology used to be
    cached as pickles, so anything that could write the cache directory could
    run arbitrary code in every later reader. Both are ``.npz`` now, loaded
    with ``allow_pickle=False``.

    A cache still holding the old ``superseding`` file is refused rather than
    converted: converting it would have to unpickle it first, which is exactly
    the execution this format change removes. The error names the file to
    delete and the flag that rebuilds the dataset.
    """
    root = dataset_root(name)
    stale = root / superseding
    if not (root / filename).exists() and stale.exists():
        raise ConfigurationError(
            f"Offline {name} data at {stale} is in the retired pickle format. "
            f"This is a security migration, not a damaged file: the cache used "
            f"to be unpickled on every read, which runs whatever code the file "
            f"contains, so the index is now a data-only .npz ({filename}) and "
            f"the pickle is refused instead of converted — converting it would "
            f"have to unpickle it.",
            remediation=f"Delete {stale} and run "
                        f"`{DATASETS[name].install_flag}` to rebuild it as "
                        f"{filename}.",
        )
    return require(name, filename)


# mkstemp creates its file 0600, but the cached datasets are public reference
# grids and $UACPY_DATA_CACHE may point at a directory a group shares, so the
# staging file is widened to what a plain create would have produced. The umask
# can only be read by setting it, so it is sampled once here at import — while
# the import lock is held — rather than on every write.
_UMASK = os.umask(0o022)
os.umask(_UMASK)
_CACHE_FILE_MODE = 0o666 & ~_UMASK


def staging_path(out) -> Path:
    """Create an empty staging sibling of ``out`` and return its path.

    Every install-time writer stages here first and ``os.replace``s onto the
    destination, so an interrupted build never leaves a partial file where
    :func:`require` will accept it.

    The name comes from :func:`tempfile.mkstemp`, not from a predictable
    sibling such as ``<out>.part``. A predictable sibling lets anything that
    can write the cache directory pre-place a symlink there: the build then
    writes through it to wherever the link points, and ``os.replace`` moves the
    *link* onto ``out``, so every later write to that dataset redirects as
    well. mkstemp creates the file ``O_CREAT|O_EXCL`` under a name that cannot be guessed, so
    there is no name to pre-place a link at. Callers re-open the path by name
    rather than being handed the descriptor, which leaves a race in principle —
    but only for an attacker who can guess an unguessable name.

    The trailing ``.tmp`` keeps a staging file out of the readers that locate
    their grid by extension (:func:`uacpy.data.gebco_local._grid` globs
    ``*.nc``).
    """
    out = Path(out)
    fd, name = tempfile.mkstemp(dir=out.parent, prefix=out.name + '.',
                                suffix='.tmp')
    os.close(fd)
    part = Path(name)
    os.chmod(part, _CACHE_FILE_MODE)
    return part


@contextlib.contextmanager
def atomic_write(out):
    """Yield a private staging path to write, and move it onto ``out`` on success.

    An install-time writer that opens the destination directly leaves a
    truncated file behind when the transfer is interrupted (Ctrl-C during a
    multi-minute paging loop) or the disk fills, and :func:`require` accepts it
    forever after — it tests only that the path exists. The staging file is a
    sibling in the same directory, so ``os.replace`` is atomic; it is removed
    if the body raises. Same contract as
    :func:`uacpy.data._http.curl_download`, for the writers that build their
    file locally rather than downloading it whole.

    The staging file is :func:`staging_path`'s, whose docstring records why the
    name is unguessable rather than ``<out>.part``.
    """
    part = staging_path(out)
    try:
        yield part
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    os.replace(part, out)


@contextlib.contextmanager
def reading(name: str, path):
    """Re-raise whatever a cached file's parser throws as a typed error.

    A file the cache accepts can still be unreadable — truncated by an
    interrupted install predating :func:`atomic_write`, or damaged on disk —
    and each reader then fails with its own parser's exception
    (``OSError``, ``ValueError``, ``UnicodeDecodeError``). The source-fallback
    chains in :mod:`uacpy.data.environment` catch ``DataFetchError`` and
    ``ConfigurationError`` only, so an untyped failure aborts the chain
    instead of falling through to the next source — and ``'auto'``, documented
    never to fail, never reaches its pelagic terminus.
    """
    try:
        yield
    except (ConfigurationError, DataFetchError):
        raise
    except Exception as exc:
        # The remediation names the install flag when `name` is a known
        # dataset. Looking it up unguarded put a KeyError on the error path
        # itself — an untyped failure raised from the helper whose whole
        # purpose is to stop untyped failures escaping.
        entry = DATASETS.get(name)
        remedy = (f"Delete {path} and re-run `{entry.install_flag}`."
                  if entry is not None else
                  f"Delete {path} and re-fetch it.")
        raise DataFetchError(
            f"Cached {name} data at {path} is present but unreadable "
            f"({type(exc).__name__}: {exc}).",
            remediation=remedy,
        ) from exc


_GRIDS: dict = {}   # resolved path -> opened grid object

# One lock behind every memo in the data layer. A memo written as a bare
# ``if key not in store: store[key] = factory()`` lets N threads racing a cold
# entry all run the factory: the array-backed readers then build N copies of
# one grid and keep the last (8 threads took EMODnet from 620 MB to 4.2 GB of
# peak RSS, and the 7 losers are unreachable from invalidate_grids), while the
# netCDF-backed ones open one HDF5 file N times at once, which the
# non-thread-safe HDF5 build under netCDF4 answers with a SIGSEGV.
#
# One shared lock rather than one per entry: the netCDF opens have to
# serialise against each other anyway (see uacpy.data._netcdf), and the warm
# path never takes it, so the only cost is that two cold loads of *different*
# datasets queue instead of running together. Reentrant because one backend's
# factory may consult another backend's memo on the same thread.
_MEMO_LOCK = threading.RLock()


def memo_lock():
    """The lock guarding every data-layer memo; hold it to mutate one directly."""
    return _MEMO_LOCK


def memoize(store, key, factory):
    """Return ``store[key]``, calling ``factory()`` at most once across threads.

    The lookup that hits a warm entry takes no lock — a dict read is atomic
    under the GIL, and an entry is published only after its factory has
    returned. A miss takes :func:`memo_lock` and re-checks under it, so N
    threads arriving on a cold key produce one object and one factory call
    however slow that call is.
    """
    if key in store:
        return store[key]
    with _MEMO_LOCK:
        if key not in store:
            store[key] = factory()
        return store[key]


def cached_grid(name: str, filename: str, factory):
    """Open ``cache_root()/<name>/<filename>`` through ``factory`` once.

    The offline raster/NetCDF backends each wrap one large file that is opened
    once and sampled many times; this is the single memo they share, keyed on
    the resolved path so a changed ``$UACPY_DATA_CACHE`` reopens.
    """
    return cached_grid_at(require(name, filename), factory, name)


def cached_grid_at(path, factory, name: str = 'cached'):
    """Memoise ``factory(path)`` on an already-resolved path.

    For backends that locate their file themselves (a glob, a versioned name)
    rather than by a fixed filename.

    The open runs under :func:`reading`, so a truncated or damaged NetCDF file
    raises this package's ``DataFetchError`` rather than netCDF4's bare
    ``OSError``. Without it the largest, most download-interruptible files in
    the cache — GEBCO at 7.5 GB and WOA23 at 1.5 GB — aborted the
    source-fallback chains that catch only ``DataFetchError`` and
    ``ConfigurationError``: a corrupt GLODAP file took down an entire
    ``fetch_environment`` whose pH axis is documented as best-effort, and a
    corrupt Graw or GlobSed file stopped ``'auto'`` from reaching the pelagic
    terminus it is documented never to fail at.

    The open runs through :func:`memoize`, so threads racing a cold path open
    the file once between them rather than once each.
    """
    def open_grid():
        with reading(name, path):
            return factory(path)

    return memoize(_GRIDS, str(path), open_grid)


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

    Runs under :func:`memo_lock`, so it cannot interleave with a memo *fill*
    (the miss path re-checks under the lock). The warm :func:`memoize` read is
    lock-free, so a concurrent reader can still be handed a handle while this
    is closing it; the ``_netcdf`` process lock keeps that from segfaulting a
    read already in flight, and the reader's *next* use of the stale handle
    raises netCDF4's closed-handle error rather than crashing. Invalidate
    between fetch batches, not under concurrent readers.
    """
    with _MEMO_LOCK:
        for grid in _GRIDS.values():
            close = getattr(grid, 'close', None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass    # a torn handle must not block the invalidation
        _GRIDS.clear()
        for clear in _CLEARERS:
            clear()
