"""Coastline backdrop for map plots — Natural Earth land polygons.

A geographic backdrop (land vs sea) for a fetched field, from the **Natural
Earth** dataset: **public domain** — no licence, no attribution, no usage
limits. Fetched as GeoJSON with the data layer's :func:`http_get` (stdlib
``urllib`` + retry); no extra dependency. Drawn in plain lon/lat.

The install-time offline cache (``./install.sh --data coastline``) is used first
when present, so maps keep their coastline without network access; otherwise the
GeoJSON is fetched live, falling back to a sea-only map if unreachable.
"""

import json
from pathlib import Path

import numpy as np

from uacpy._log import log_message
from uacpy.data import _cache
from uacpy.data._http import http_get

__all__ = ['land_polygons', 'download_coastline', 'NATURAL_EARTH_URL',
           'COASTLINE_RESOLUTIONS']

# Natural Earth land polygons (public domain). nvkelso/natural-earth-vector.
NATURAL_EARTH_URL = (
    'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/'
    'master/geojson/ne_{resolution}_land.geojson'
)
COASTLINE_RESOLUTIONS = ('110m', '50m', '10m')
_USER_AGENT = 'uacpy (+https://github.com/ErVuL/uacpy)'


def _rings(data):
    """Exterior ``(N, 2)`` ``(lon, lat)`` rings from a land GeoJSON, or ``None``."""
    rings = []
    for feat in data.get('features', []):
        geom = feat.get('geometry') or {}
        coords = geom.get('coordinates')
        if not coords:
            continue
        polys = coords if geom.get('type') == 'MultiPolygon' else [coords]
        for poly in polys:
            rings.append(np.asarray(poly[0], dtype=float))   # exterior ring
    return rings or None


def _local_path(resolution):
    return _cache.dataset_root('coastline') / f'ne_{resolution}_land.geojson'


def land_polygons(resolution='50m', *, url=None, timeout=40.0, verbose=False):
    """Natural Earth land polygons (**public domain**) for a coastline backdrop.

    Returns a list of ``(N, 2)`` ``(lon, lat)`` exterior rings, or ``None`` if
    no source is available. ``resolution`` is ``'110m'`` (coarse), ``'50m'``
    (default) or ``'10m'`` (detailed, large). Reads the install-time cache first,
    then the live GeoJSON. No attribution is required.
    """
    local = _local_path(resolution)
    if url is None and local.exists():
        try:
            return _rings(json.loads(local.read_text()))
        except Exception:          # noqa: BLE001 — fall back to the live source
            pass
    src = (url or NATURAL_EARTH_URL).format(resolution=resolution)
    try:
        data = json.loads(http_get(src, timeout=timeout, verbose=verbose,
                                   source='coastline', user_agent=_USER_AGENT))
    except Exception:          # noqa: BLE001 — backdrop is optional
        return None
    return _rings(data)


def download_coastline(cache_dir=None, *, resolutions=COASTLINE_RESOLUTIONS,
                       timeout=120.0, verbose=False):
    """Download Natural Earth land polygons into the offline cache.

    Saves ``ne_<res>_land.geojson`` for each requested ``resolutions`` into
    ``<cache>/coastline/`` (public domain — no attribution). Returns the list of
    written paths.
    """
    dest = Path(cache_dir) if cache_dir else _cache.dataset_root('coastline')
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for res in resolutions:
        log_message('coastline', f"downloading Natural Earth land ({res})",
                    verbose=verbose)
        blob = http_get(NATURAL_EARTH_URL.format(resolution=res), timeout=timeout,
                        verbose=verbose, source='coastline', user_agent=_USER_AGENT)
        out = dest / f'ne_{res}_land.geojson'
        out.write_bytes(blob)
        written.append(out)
    log_message('coastline', f"coastline cached: {len(written)} file(s) → {dest}",
                verbose=verbose)
    return written
