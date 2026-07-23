"""EMODnet Bathymetry DTM live fetch (European seas + Caribbean, high-res).

A higher-resolution regional alternative to GEBCO/GMRT: the EMODnet Digital
Terrain Model is a ~1/16 arc-minute (~115 m) hydrographic-survey compilation of
the European seas, with a separate Caribbean tile. Licence **CC-BY-4.0**
(commercial use OK, attribution required); served live from the EMODnet
**ERDDAP** griddap service (no auth) — no download.

Coverage is regional: a point outside the European/Caribbean tiles raises
``DataFetchError`` so a fallback source (GMRT, GEBCO) can take over. Elevation is
referenced to **Lowest Astronomical Tide (LAT)**, not mean sea level, so a
returned depth is a few tenths of a metre to a couple of metres deeper than the
MSL-referenced GEBCO/GMRT depth in shallow water — negligible offshore.
"""

import urllib.parse

import numpy as np

from uacpy.core.exceptions import DataFetchError
from uacpy.data._geo import as_coordinate, normalize_lon
from uacpy.data._http import http_get

__all__ = ['point_depth', 'depths_along', 'region_grid', 'ERDDAP_URL',
           'DATASETS']

ERDDAP_URL = 'https://erddap.emodnet.eu/erddap/griddap'
#: The European DTM then the Caribbean tile — tried in order so a point in
#: either is resolved and one outside both falls through.
DATASETS = ('bathymetry_dtm_2024', 'bathymetry_dtm_carib_2024')
_USER_AGENT = 'uacpy (+https://github.com/ErVuL/uacpy)'
#: DTM native grid spacing (1/16 arc-minute ≈ 0.001041667°) — used to size the
#: strided ``region_grid`` subset.
EMODNET_NATIVE_DEG = 1.0 / 960.0


def _griddap_url(dataset, constraint):
    """ERDDAP griddap CSV URL for a variable ``constraint`` expression."""
    query = urllib.parse.quote(f"elevation{constraint}", safe='[]():.,-')
    return f"{ERDDAP_URL}/{dataset}.csv?{query}"


def _elevation(lat, lon, dataset, *, timeout, verbose):
    """Elevation (m, +up, LAT datum) at one cell of ``dataset``, or ``NaN``.

    Returns ``NaN`` where the service reports no value; raises
    ``DataFetchError`` (out-of-range coordinate) via :func:`http_get` when the
    point lies outside the tile — the caller treats both as "no coverage here".
    """
    url = _griddap_url(dataset, f"[({lat})][({normalize_lon(lon)})]")
    body = http_get(url, timeout=timeout, verbose=verbose, source='bathymetry',
                    user_agent=_USER_AGENT).decode('utf-8', 'replace')
    return _parse_point_csv(body)


def _parse_point_csv(body):
    """Last elevation value from an ERDDAP griddap ``.csv`` point response.

    The body is ``latitude,longitude,elevation`` then a units row then one data
    row; return the elevation, or ``NaN`` if it is missing / ``NaN``.
    """
    rows = [ln for ln in body.splitlines() if ln.strip()]
    if len(rows) < 3:
        return np.nan
    try:
        return float(rows[-1].split(',')[-1])
    except (ValueError, IndexError):
        return np.nan


def _elevation_any(lat, lon, *, timeout, verbose):
    """First finite elevation across the DTM tiles, or raise if none covers."""
    for dataset in DATASETS:
        try:
            elev = _elevation(lat, lon, dataset, timeout=timeout, verbose=verbose)
        except DataFetchError:
            continue                        # out of this tile's range — try next
        if np.isfinite(elev):
            return elev
    raise DataFetchError(
        f"EMODnet DTM has no coverage at ({lat:.4f}, {lon:.4f}) "
        "(European seas + Caribbean only).",
        remediation="Use bathymetry_sources='gmrt' or 'gebco' for global "
                    "coverage.",
    )


def _depth(elev, lat, lon):
    if elev >= 0.0:
        raise DataFetchError(
            f"EMODnet DTM reports land (elevation {elev:.0f} m) at "
            f"({lat:.4f}, {lon:.4f}); no water column.",
            remediation="Pick a location offshore, or supply a depth directly.",
        )
    return -elev


def point_depth(point, *, timeout=30.0, verbose=False):
    """Water depth (m, positive down) at a single point from the EMODnet DTM."""
    lat, lon = as_coordinate(point)
    return _depth(_elevation_any(lat, lon, timeout=timeout, verbose=verbose),
                  lat, lon)


def depths_along(lats, lons, *, timeout=30.0, verbose=False):
    """Depths (m) at paired ``lats``/``lons`` waypoints (one griddap call each)."""
    out = np.empty(len(lats))
    for k, (la, lo) in enumerate(zip(lats, lons)):
        out[k] = _depth(
            _elevation_any(la, lo, timeout=timeout, verbose=verbose), la, lo)
    return out


def region_grid(lat_range, lon_range, n_lat, n_lon, *, timeout=120.0,
                verbose=False):
    """``(lats, lons, depth)`` over a region; land / uncovered cells are NaN.

    Requests a strided ERDDAP griddap subset (sized so the native DTM is
    down-sampled to roughly ``n_lat × n_lon`` nodes), then nearest-samples it
    onto the requested mesh so the return shape matches
    :func:`uacpy.data.fetch_bathy_grid`.
    """
    la0, la1 = sorted((float(lat_range[0]), float(lat_range[1])))
    lo0, lo1 = sorted((normalize_lon(lon_range[0]), normalize_lon(lon_range[1])))
    stride_lat = _stride(la1 - la0, n_lat)
    stride_lon = _stride(lo1 - lo0, n_lon)
    constraint = (f"[({la0}):{stride_lat}:({la1})]"
                  f"[({lo0}):{stride_lon}:({lo1})]")
    glat, glon, gz = _subset(constraint, timeout=timeout, verbose=verbose)

    lats = np.linspace(la0, la1, n_lat)
    lons = np.linspace(lo0, lo1, n_lon)
    ri = _nearest_indices(glat, lats)
    ci = _nearest_indices(glon, lons)
    block = gz[np.ix_(ri, ci)]
    depth = np.where(block < 0.0, -block, np.nan)
    return lats, lons, depth


def _stride(span_deg, n):
    """Index stride down-sampling the native DTM to ~``n`` nodes over ``span``."""
    native_nodes = span_deg / EMODNET_NATIVE_DEG
    return max(1, int(round(native_nodes / max(int(n) - 1, 1))))


def _subset(constraint, *, timeout, verbose):
    """Fetch a griddap subset and pivot it to ``(lats, lons, elev_2d)``.

    Tries each DTM tile in turn; the first that returns a grid wins, so a region
    in the Caribbean tile resolves after the European tile reports out-of-range.
    """
    errors = []
    for dataset in DATASETS:
        url = _griddap_url(dataset, constraint)
        try:
            body = http_get(url, timeout=timeout, verbose=verbose,
                            source='bathymetry', user_agent=_USER_AGENT)
        except DataFetchError as exc:
            errors.append(exc)
            continue
        return _parse_grid_csv(body.decode('utf-8', 'replace'))
    raise DataFetchError(
        "EMODnet DTM has no coverage over the requested region "
        "(European seas + Caribbean only).",
        remediation="Use bathymetry_sources='gmrt' or 'gebco' for global "
                    "coverage.",
    )


def _parse_grid_csv(body):
    """Pivot an ERDDAP griddap ``.csv`` grid response to ``(lats, lons, z)``.

    Rows are ``latitude,longitude,elevation`` combinations (after two header
    rows); collect the unique sorted axes and lay each elevation into its cell
    (missing cells stay ``NaN``).
    """
    lat_lon_z = []
    for ln in body.splitlines()[2:]:
        if not ln.strip():
            continue
        parts = ln.split(',')
        if len(parts) < 3:
            continue
        try:
            lat_lon_z.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError:
            continue                                 # NaN / malformed cell
    if not lat_lon_z:
        raise DataFetchError(
            "EMODnet DTM subset returned no parseable cells.",
            remediation="Retry, or use bathymetry_sources='gmrt' / 'gebco'.",
        )
    lats = np.array(sorted({t[0] for t in lat_lon_z}))
    lons = np.array(sorted({t[1] for t in lat_lon_z}))
    lat_idx = {v: i for i, v in enumerate(lats)}
    lon_idx = {v: i for i, v in enumerate(lons)}
    z = np.full((lats.size, lons.size), np.nan)
    for la, lo, elev in lat_lon_z:
        z[lat_idx[la], lon_idx[lo]] = elev
    return lats, lons, z


def _nearest_indices(axis, queries):
    """Nearest-node index into ``axis`` for each query, any axis orientation."""
    axis = np.asarray(axis, dtype=float)
    order = np.argsort(axis)
    sorted_axis = axis[order]
    pos = np.searchsorted(sorted_axis, queries)
    lo = np.clip(pos - 1, 0, sorted_axis.size - 1)
    hi = np.clip(pos, 0, sorted_axis.size - 1)
    pick_hi = np.abs(sorted_axis[hi] - queries) < np.abs(queries - sorted_axis[lo])
    nearest_sorted = np.where(pick_hi, hi, lo)
    return order[nearest_sorted]
