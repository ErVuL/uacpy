"""Offline EMODnet seabed substrate (European seas) → ``BoundaryProperties``.

The offline, attribution-clean (**CC-BY**) European seabed backend — the local
counterpart of the live :mod:`uacpy.data.seabed` WFS fetch, parallel to the
global grain-size DB (:mod:`uacpy.data.sediment_db`). ``install.sh --data
emodnet`` downloads the harmonised EMODnet Geology seabed-substrate polygons
(Folk 5-class, 1:1M) once from the public WFS and stores them as a compact WKB
index in the cache; this module then does **offline point-in-polygon** lookups
(shapely STRtree) and maps the Folk class to a bottom with the very same
relations used by the live backend.

Coverage is European seas only (like the live source): a point outside the
mapped polygons raises ``DataFetchError`` so a caller's 'auto' bottom can fall
through to the global grain-size DB.
"""

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

from uacpy._log import log_message
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.core.environment import BoundaryProperties, Bottom
from uacpy.data import _cache
from uacpy.data._geo import Coordinate, as_coordinate, normalize_lon
from uacpy.data._http import http_get
from uacpy.data.seabed import (
    EMODNET_WFS_URL, EMODNET_LAYER, _bottom_from_folk5,
)
from uacpy.data.sediment import range_dependent_bottom_along, water_sound_speed_at

__all__ = ['download_emodnet_db', 'fetch_seabed_local', 'fetch_bottom_local',
           'fetch_bottom_local_transect']

INDEX_FILE = 'seabed_substrate.npz'
#: The pre-npz pickled index, refused by :func:`uacpy.data._cache.require_npz`.
RETIRED_INDEX_FILE = 'seabed_substrate.pkl'
_PAGE = 5000                        # WFS GetFeature page size (startIndex/count)
# The layer has no primary key, so GeoServer needs an explicit sort to page.
_SORT_BY = 'objectid'
_INDEX = {}                         # cache_root -> (STRtree, codes ndarray)
_cache.register_cache(_INDEX.clear)


def _shapely():
    """Import shapely lazily, raising a typed install hint if it is absent."""
    try:
        import shapely
        return shapely
    except ImportError as exc:                                  # pragma: no cover
        raise ConfigurationError(
            "The offline EMODnet seabed backend needs 'shapely'.",
            remediation="shapely ships with the default uacpy install; reinstall "
                        "with `pip install -e .`, or `pip install shapely`.",
        ) from exc


def download_emodnet_db(cache_dir=None, *, timeout=300.0, verbose=False):
    """Download the EMODnet seabed-substrate polygons into a local WKB index.

    Pages through the public EMODnet Geology WFS (Folk 5-class, 1:1M, EPSG:4326),
    keeps each polygon's geometry + ``folk_5cl`` code, and writes
    ``<cache>/emodnet/seabed_substrate.npz`` — the file the offline backend
    reads. Returns the written path.

    The index is three plain arrays, so it loads with ``allow_pickle=False``:
    ``codes`` (int32, one Folk class per polygon), ``wkb`` (uint8, every
    polygon's WKB concatenated) and ``offsets`` (int64, ``n + 1`` cut points
    into ``wkb``). WKB is already the compact serialisation shapely reads, so
    only the container changed; the bytes per polygon are the same ones the
    pickled index held.

    ``cache_dir`` defaults to the offline cache's ``emodnet`` directory.
    """
    shapely = _shapely()
    dest = Path(cache_dir) if cache_dir else _cache.dataset_root('emodnet')
    dest.mkdir(parents=True, exist_ok=True)
    log_message('seabed', "downloading EMODnet seabed substrate (Folk 5cl, 1:1M)",
                verbose=verbose)

    codes, wkb = [], []
    start = 0
    while True:
        url = (f"{EMODNET_WFS_URL}?service=WFS&version=2.0.0&request=GetFeature"
               f"&typeNames={EMODNET_LAYER}&outputFormat=application/json"
               f"&srsName=EPSG:4326&sortBy={_SORT_BY}&count={_PAGE}"
               f"&startIndex={start}")
        fc = json.loads(http_get(url, timeout=timeout, verbose=verbose,
                                 source='seabed'))
        feats = fc.get('features') or []
        if not feats:
            break
        for ft in feats:
            try:
                code = int(ft['properties']['folk_5cl'])
            except (KeyError, TypeError, ValueError):
                continue
            codes.append(code)
            wkb.append(shapely.to_wkb(shapely.geometry.shape(ft['geometry'])))
        start += len(feats)
        log_message('seabed', f"EMODnet seabed: {start} polygons fetched",
                    verbose=verbose)
        if len(feats) < _PAGE:
            break

    if not codes:
        raise DataFetchError(
            "EMODnet WFS returned no seabed polygons.",
            remediation="Retry; the upstream WFS layout may have changed.",
        )
    out = dest / INDEX_FILE
    with _cache.atomic_write(out) as part:
        # A file object, not the path: np.savez_compressed appends '.npz' to a
        # name that lacks it, and the staging name does not end in '.npz'.
        with open(part, 'wb') as fh:
            np.savez_compressed(
                fh,
                codes=np.asarray(codes, dtype=np.int32),
                wkb=np.frombuffer(b''.join(wkb), dtype=np.uint8),
                offsets=np.cumsum([0] + [len(b) for b in wkb], dtype=np.int64),
            )
    _INDEX.clear()                            # force rebuild of the spatial index
    log_message('seabed', f"EMODnet seabed substrate: {len(codes)} polygons → {out}",
                verbose=verbose)
    return out


def _build_index():
    """Read the cached polygons into an ``(STRtree, Folk-code array)`` pair."""
    shapely = _shapely()
    # raises if not installed, or if the retired pickled index is still there
    path = _cache.require_npz('emodnet', INDEX_FILE, RETIRED_INDEX_FILE)
    with _cache.reading('emodnet', path):
        # allow_pickle=False is passed rather than left to numpy's default:
        # this file is read straight out of the cache directory, and under
        # allow_pickle an object array in it would execute on load.
        with np.load(path, allow_pickle=False) as z:
            blob, offsets = z['wkb'], z['offsets']
            codes = np.asarray(z['codes'], dtype=int)
            # Slicing the blob is a view; only the per-polygon tobytes() copies,
            # so the 203 MB index is never held twice over.
            wkb = [blob[a:b].tobytes()
                   for a, b in zip(offsets[:-1], offsets[1:])]
        geoms = shapely.from_wkb(np.asarray(wkb, dtype=object))
    return (shapely.STRtree(geoms), codes)


def _index():
    """Build (or reuse) the STRtree + Folk-code array from the local index.

    Built through :func:`uacpy.data._cache.memoize`: the polygons cost ~620 MB
    to load, and threads racing the unguarded memo used to build one copy each
    and keep the last.
    """
    return _cache.memoize(_INDEX, str(_cache.cache_root()), _build_index)


def fetch_seabed_local(point: Coordinate) -> dict:
    """Offline EMODnet seabed-substrate record at a ``(lat, lon)`` point.

    Returns ``{'folk_5cl', 'source'}``. Raises ``DataFetchError`` outside the
    mapped European-seas polygons.
    """
    shapely = _shapely()
    lat, lon = as_coordinate(point)
    tree, codes = _index()
    hits = tree.query(shapely.Point(normalize_lon(lon), lat), predicate='intersects')
    if len(hits) == 0:
        raise DataFetchError(
            f"EMODnet has no seabed substrate at {lat:.3f}, {lon:.3f} "
            "(coverage is European seas only).",
            remediation="Outside European seas, use bottom_sources='grainsize' "
                        "(global) or pass an explicit grain size (ϕ) / class via bottom=.",
        )
    # STRtree.query returns matches in no guaranteed order (it varies across
    # shapely versions), so a point on a shared polygon boundary must resolve
    # by a deterministic rule: the lowest polygon index wins.
    return {'folk_5cl': int(codes[hits.min()]),
            'source': 'EMODnet Geology seabed substrate 1:1M (offline)'}


def fetch_bottom_local(point: Coordinate, *, roughness: float = 0.0,
                       water_sound_speed: Optional[float] = None,
                       timeout=None, verbose: Union[bool, str] = False
                       ) -> BoundaryProperties:
    """Model-ready bottom from the offline EMODnet polygon at ``(lat, lon)``.

    This is the EMODnet seabed-substrate provider of the ``fetch_bottom_local``
    protocol name that ``fetch_environment`` resolves per provider module
    (``bottom_sources='emodnet'``); the package-level
    ``uacpy.data.fetch_bottom_local`` is :mod:`uacpy.data.sediment_db`'s
    grain-size provider, not this function.

    ``timeout`` is accepted (and ignored — this backend is offline) for signature
    uniformity with the network bottom fetchers. ``water_sound_speed`` (m/s)
    scales the grain-size velocity ratio to the in-situ near-seabed water;
    ``None`` uses the Hamilton reference.
    """
    lat, lon = as_coordinate(point)
    sub = fetch_seabed_local(point)
    bottom = _bottom_from_folk5(sub['folk_5cl'], lat, lon, roughness=roughness,
                                water_sound_speed=water_sound_speed)
    log_message(
        'seabed', f"EMODnet (offline) folk_5cl={sub['folk_5cl']} at "
        f"{lat:.3f}, {lon:.3f} → {bottom.acoustic_type} "
        f"c_p={bottom.sound_speed:.0f} m/s", verbose=verbose,
    )
    return bottom


def fetch_bottom_local_transect(start: Coordinate, end: Coordinate, *,
                                n_points=6, max_points=None,
                                roughness: float = 0.0,
                                water_sound_speed: Optional[float] = None,
                                timeout=None, verbose: Union[bool, str] = False
                                ) -> Bottom:
    """Range-dependent bottom from the offline EMODnet polygons along a transect.

    The EMODnet provider of the ``fetch_bottom_local_transect`` protocol name;
    the package-level ``uacpy.data.fetch_bottom_local_transect`` is
    :mod:`uacpy.data.sediment_db`'s. ``water_sound_speed`` also takes a
    ``(lat, lon) -> m/s`` callable, so each column scales to the water over
    its own seafloor. ``timeout``/``verbose`` are accepted (and ignored —
    this backend is offline) for signature uniformity with the network
    bottom fetchers.
    """
    return range_dependent_bottom_along(
        lambda la, lo: fetch_bottom_local(
            (la, lo), roughness=roughness,
            water_sound_speed=water_sound_speed_at(water_sound_speed, la, lo)),
        start, end, n_points, source_label='EMODnet (offline)',
        max_points=max_points,
    )
