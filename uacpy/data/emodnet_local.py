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
import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np

from uacpy._log import log_message
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.core.environment import BoundaryProperties, Bottom
from uacpy.data import _cache
from uacpy.data._geo import Coordinate, as_coordinate, normalize_lon
from uacpy.data._http import http_get
from uacpy.data.seabed import EMODNET_WFS_URL, EMODNET_LAYER, _FOLK5_TO_BOTTOM
from uacpy.data.sediment import (
    bottom_from_class, bottom_from_grain_size, range_dependent_bottom_along,
    water_sound_speed_at,
)

__all__ = ['download_emodnet_db', 'fetch_seabed_local', 'fetch_bottom_local',
           'fetch_bottom_local_transect']

INDEX_FILE = 'seabed_substrate.pkl'
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
    ``<cache>/emodnet/seabed_substrate.pkl`` (a pickled ``{'codes', 'wkb'}``
    index) — the file the offline backend reads. Returns the written path.

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
    with open(out, 'wb') as fh:
        pickle.dump({'codes': codes, 'wkb': wkb}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)
    _INDEX.clear()                            # force rebuild of the spatial index
    log_message('seabed', f"EMODnet seabed substrate: {len(codes)} polygons → {out}",
                verbose=verbose)
    return out


def _index():
    """Build (or reuse) the STRtree + Folk-code array from the local index."""
    root = str(_cache.cache_root())
    if root in _INDEX:
        return _INDEX[root]
    shapely = _shapely()
    path = _cache.require('emodnet', INDEX_FILE)         # raises if not installed
    with open(path, 'rb') as fh:
        data = pickle.load(fh)
    geoms = shapely.from_wkb(np.asarray(data['wkb'], dtype=object))
    codes = np.asarray(data['codes'], dtype=int)
    result = (shapely.STRtree(geoms), codes)
    _INDEX[root] = result
    return result


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
    return {'folk_5cl': int(codes[hits[0]]),
            'source': 'EMODnet Geology seabed substrate 1:1M (offline)'}


def fetch_bottom_local(point: Coordinate, *, roughness: float = 0.0,
                       water_sound_speed: Optional[float] = None,
                       timeout=None, verbose: Union[bool, str] = False
                       ) -> BoundaryProperties:
    """Model-ready bottom from the offline EMODnet polygon at ``(lat, lon)``.

    ``timeout`` is accepted (and ignored — this backend is offline) for signature
    uniformity with the network bottom fetchers. ``water_sound_speed`` (m/s)
    scales the grain-size velocity ratio to the in-situ near-seabed water;
    ``None`` uses the Hamilton reference.
    """
    lat, lon = as_coordinate(point)
    sub = fetch_seabed_local(point)
    if sub['folk_5cl'] not in _FOLK5_TO_BOTTOM:
        raise DataFetchError(
            f"EMODnet returned an unrecognised Folk-5 class "
            f"{sub['folk_5cl']!r} at {lat:.3f}, {lon:.3f}; "
            "refusing to fabricate a default bottom.",
            remediation="Pass an explicit grain size (ϕ) or sediment class, or "
                        "let the 'auto' bottom chain fall through to another "
                        "source.",
        )
    kind, value = _FOLK5_TO_BOTTOM[sub['folk_5cl']]
    if kind == 'phi':
        bottom = bottom_from_grain_size(
            value, roughness=roughness, water_sound_speed=water_sound_speed)
    else:
        bottom = bottom_from_class(value, roughness=roughness)
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

    ``water_sound_speed`` also takes a ``(lat, lon) -> m/s`` callable,
    so each column scales to the water over its own seafloor.
    """
    return range_dependent_bottom_along(
        lambda la, lo: fetch_bottom_local(
            (la, lo), roughness=roughness,
            water_sound_speed=water_sound_speed_at(water_sound_speed, la, lo)),
        start, end, n_points, source_label='EMODnet (offline)',
        max_points=max_points,
    )
