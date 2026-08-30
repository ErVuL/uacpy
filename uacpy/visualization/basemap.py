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
from uacpy.core.exceptions import DataFetchError
from uacpy.data._http import http_get
from uacpy.visualization.plots._common import _plot_warn

__all__ = ['land_polygons', 'download_coastline', 'NATURAL_EARTH_URL',
           'COASTLINE_RESOLUTIONS']

# Natural Earth land polygons (public domain). nvkelso/natural-earth-vector.
#
# This is the one floating ref in the project: every other remote artefact is
# pinned (``install.sh`` checks out bellhopcuda by commit SHA and verifies the
# OASES tarball's sha256; GEBCO_2025.nc / GlobSed-v3.nc / WOA23 carry their
# version in the path), while ``master`` here resolves to whatever the branch
# holds on the day of the fetch. Pinning it to a release tag would make the
# backdrop reproducible, but the tag has to be READ OFF the upstream repo — a
# guessed one turns a working optional download into a permanent 404. The
# cached GeoJSON in ``coastline/`` cannot settle it either: its only metadata
# are ``name``/``crs``/``bbox``, with no version stamp to match against.
# To do it, with network: list the repo's tags, confirm
# ``geojson/ne_{110m,50m,10m}_land.geojson`` exist at the chosen one, swap
# ``master`` for it, and re-run ``./install.sh --data coastline``.
# Until then the exposure stays small: the cache is read first
# (:func:`land_polygons`), an unreachable or malformed source only warns and
# drops the land layer, and ``download_coastline`` parses before it writes so a
# portal error page never reaches the cache.
NATURAL_EARTH_URL = (
    'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/'
    'master/geojson/ne_{resolution}_land.geojson'
)
COASTLINE_RESOLUTIONS = ('110m', '50m', '10m')
_USER_AGENT = 'uacpy (+https://github.com/ErVuL/uacpy)'


def _rings(data):
    """Exterior ``(N, 2)`` ``(lon, lat)`` rings from a land GeoJSON, or ``None``.

    Only ``poly[0]``, the exterior ring, is kept; a polygon's interior rings —
    the holes Natural Earth uses for inland water — are dropped, so a lake
    inside a landmass paints as land rather than showing the sea colour through
    the hole. The trade-off is deliberate: the maps here are marine, the land
    layer is a backdrop drawn over the bathymetry to hide coastal grid-cell
    bleed (``maps.py``), and matplotlib's ``fill`` takes one closed path per
    call — punching the holes back out needs a compound ``Path`` with
    per-ring ``CLOSEPOLY`` codes and correct winding. Revisit if a lake ever
    has to read as water."""
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
    cache_fault = None
    if url is None and local.exists():
        try:
            return _rings(json.loads(local.read_text(encoding='utf-8')))
        except Exception as exc:   # noqa: BLE001 — fall back to the live source
            # A present-but-corrupt cache is a different state from an absent
            # one and takes a different remedy: re-running the install flag
            # over a file that already exists changes nothing until the
            # damaged one is deleted. Keep the diagnosis, and still try the
            # live source — a cache damaged on disk self-heals when online.
            cache_fault = (f"the cached coastline at {local} is present but "
                           f"unreadable ({type(exc).__name__}: {exc})")
            _plot_warn(
                f"{cache_fault}; reading the live source instead. Delete that "
                f"file and re-run `./install.sh --data coastline` to repair "
                f"the offline cache.")
    src = (url or NATURAL_EARTH_URL).format(resolution=resolution)
    try:
        data = json.loads(http_get(src, timeout=timeout, verbose=verbose,
                                   source='coastline', user_agent=_USER_AGENT))
    except Exception as exc:   # noqa: BLE001 — backdrop is optional
        # The map still renders (sea only); say why the land layer is
        # missing instead of degrading silently after a multi-retry stall.
        # When the cache is the reason there is nothing to fall back to, the
        # remedy is to delete it — telling that user to cache the dataset
        # offline names the state they are already in.
        # The warning names the user's own line — where `basemap=` and
        # `coastline_resolution=` are — whether they called a map plotter or
        # this function directly, which a frame count could not do for both.
        remedy = (f"{cache_fault}, so there is nothing to fall back to: "
                  f"delete that file and re-run `./install.sh --data "
                  f"coastline`, or pass basemap=False."
                  if cache_fault else
                  f"Run `./install.sh --data coastline` to cache it offline, "
                  f"or pass basemap=False.")
        _plot_warn(
            f"coastline backdrop unavailable ({type(exc).__name__}); the "
            f"map renders without land. {remedy}")
        return None
    return _rings(data)


def download_coastline(cache_dir=None, *, resolutions=COASTLINE_RESOLUTIONS,
                       timeout=120.0, verbose=False):
    """Download Natural Earth land polygons into the offline cache.

    Saves ``ne_<res>_land.geojson`` for each requested ``resolutions`` directly
    into ``cache_dir`` when given, else into the offline cache's ``coastline/``
    dataset root (public domain — no attribution). Returns the list of written
    paths.
    """
    dest = Path(cache_dir) if cache_dir else _cache.dataset_root('coastline')
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for res in resolutions:
        log_message('coastline', f"downloading Natural Earth land ({res})",
                    verbose=verbose)
        blob = http_get(NATURAL_EARTH_URL.format(resolution=res), timeout=timeout,
                        verbose=verbose, source='coastline', user_agent=_USER_AGENT)
        # A captive portal or proxy error page arrives as HTTP 200 HTML;
        # caching it would poison every later offline read. Parse before
        # writing, and fail with the data layer's typed error.
        try:
            json.loads(blob)
        except ValueError as exc:
            raise DataFetchError(
                f"coastline download for {res!r} is not valid GeoJSON — a "
                f"captive portal or proxy error page was likely returned.",
                remediation="Check connectivity (log in to the network "
                            "portal if one appeared) and re-run "
                            "./install.sh --data coastline.",
            ) from exc
        out = dest / f'ne_{res}_land.geojson'
        # Staged like every other cache writer: a write that fails part-way
        # through leaves a truncated GeoJSON that every later existence check
        # accepts, and `land_polygons` then falls through to the network on
        # every map it draws.
        with _cache.atomic_write(out) as part:
            part.write_bytes(blob)
        written.append(out)
    log_message('coastline', f"coastline cached: {len(written)} file(s) → {dest}",
                verbose=verbose)
    return written
