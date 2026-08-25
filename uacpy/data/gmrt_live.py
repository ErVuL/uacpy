"""GMRT live bathymetry (Global Multi-Resolution Topography synthesis).

A higher-resolution alternative to the OpenTopoData/GEBCO point API: GMRT merges
edited multibeam sonar (down to ~100 m where surveyed) over GEBCO/SRTM15+ where
it isn't, so it is never coarser than GEBCO and often much finer near coasts,
ridges and survey corridors. Licence **CC-BY-4.0** (commercial use OK,
attribution required); served live (no download).

Endpoints (no auth):
  * ``PointServer`` — elevation (m, +up) at one lon/lat.
  * ``GridServer?format=coards`` — a ``lon``/``lat``/``altitude`` NetCDF subset.
"""

import numpy as np

from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data._geo import (
    as_coordinate, nearest_indices, normalize_lon,
)
from uacpy.data._http import http_get

__all__ = ['point_depth', 'depths_along', 'region_grid',
           'POINT_URL', 'GRID_URL']

POINT_URL = 'https://www.gmrt.org/services/PointServer'
GRID_URL = 'https://www.gmrt.org/services/GridServer'
_USER_AGENT = 'uacpy (+https://github.com/ErVuL/uacpy)'


def _elevation(lat, lon, *, timeout, verbose):
    """Elevation (m, positive up) at one point from the GMRT PointServer."""
    lon = normalize_lon(lon)
    url = f"{POINT_URL}?longitude={lon}&latitude={lat}&format=text/plain"
    body = http_get(url, timeout=timeout, verbose=verbose, source='bathymetry',
                    user_agent=_USER_AGENT).decode('utf-8', 'replace').strip()
    try:
        return float(body.split()[0])
    except (ValueError, IndexError):
        raise DataFetchError(
            f"GMRT returned an unparseable elevation at ({lat:.4f}, {lon:.4f}): "
            f"{body[:80]!r}.",
            remediation="Retry, or use source='api' (OpenTopoData).",
        )


def _depth(elev, lat, lon):
    if elev >= 0.0:
        raise DataFetchError(
            f"GMRT reports land (elevation {elev:.0f} m) at "
            f"({lat:.4f}, {lon:.4f}); no water column.",
            remediation="Pick a location offshore, or supply a depth directly.",
        )
    return -elev


def point_depth(point, *, timeout=30.0, verbose=False):
    """Water depth (m, positive down) at a single point from GMRT."""
    lat, lon = as_coordinate(point)
    return _depth(_elevation(lat, lon, timeout=timeout, verbose=verbose), lat, lon)


def depths_along(lats, lons, *, timeout=30.0, verbose=False):
    """Depths (m) at paired ``lats``/``lons`` waypoints (one PointServer call each)."""
    out = np.empty(len(lats))
    for k, (la, lo) in enumerate(zip(lats, lons)):
        out[k] = _depth(_elevation(la, lo, timeout=timeout, verbose=verbose), la, lo)
    return out


def region_grid(lat_range, lon_range, n_lat, n_lon, *, timeout=120.0, verbose=False):
    """``(lats, lons, depth)`` over a region; land cells are NaN.

    Fetches a GMRT GridServer COARDS subset, then samples it (nearest node) onto
    the requested ``n_lat × n_lon`` mesh so the return shape matches
    :func:`uacpy.data.fetch_bathy_grid`.
    """
    import contextlib
    import tempfile
    from pathlib import Path

    from uacpy.data._netcdf import netcdf_lock, open_netcdf

    if normalize_lon(lon_range[1]) < normalize_lon(lon_range[0]):
        raise ConfigurationError(
            f"gmrt_live: lon_range {lon_range} crosses the antimeridian "
            f"(end < start means an eastward crossing on the GEBCO/local "
            f"path), but this service takes plain min<max boxes — the "
            f"sorted request would silently sweep the long way around.",
            remediation="Use bathymetry source 'local'/'gebco', or split "
                        "the request at 180 deg.")
    lo0, lo1 = sorted((normalize_lon(lon_range[0]), normalize_lon(lon_range[1])))
    la0, la1 = sorted((float(lat_range[0]), float(lat_range[1])))
    url = (f"{GRID_URL}?minlongitude={lo0}&maxlongitude={lo1}"
           f"&minlatitude={la0}&maxlatitude={la1}&format=coards&resolution=high")
    blob = http_get(url, timeout=timeout, verbose=verbose, source='bathymetry',
                    user_agent=_USER_AGENT)
    # Unique, race-safe scratch file (hash(url) is salted per-process and
    # collides across concurrent same-URL calls).
    with tempfile.NamedTemporaryFile(suffix='.nc', prefix='uacpy_gmrt_',
                                     delete=False) as fh:
        fh.write(blob)
        tmp = Path(fh.name)
    try:
        # closing(), not a bare close() after the reads: a grid whose variable
        # names do not match raises between the two and would leave the handle
        # open on the file the finally is about to unlink.
        #
        # netcdf_lock spans the whole statement, so the slices below and the
        # close() that ends it are inside it as well as the open — netCDF4 is
        # not thread-safe here, and a live download on one thread would
        # otherwise read and close alongside another thread's grid read.
        with netcdf_lock, contextlib.closing(open_netcdf(tmp)) as ds:
            names = {n.lower(): n for n in ds.variables}
            try:
                glon = np.asarray(ds.variables[names['lon']][:], dtype=float)
                glat = np.asarray(ds.variables[names['lat']][:], dtype=float)
                elev_var = names.get('altitude') or names.get('z') or 'altitude'
                # netCDF4 returns a masked array where the grid declares a
                # _FillValue; np.asarray would drop the mask and expose the raw
                # fill as a real elevation (a large positive one reads as land,
                # a large negative one as a kilometres-deep basin). Fill
                # *through* the mask to NaN, the no-data value the rest of this
                # module already uses.
                gz = np.ma.filled(
                    np.ma.asarray(ds.variables[elev_var][:], dtype=float),
                    np.nan)
            except KeyError as exc:
                raise DataFetchError(
                    f"GMRT COARDS grid is missing an expected variable "
                    f"({exc}): the schema is 'lon'/'lat' axes with an "
                    f"'altitude' (or 'z') elevation; this file has "
                    f"{sorted(names)}. The GridServer response format may "
                    f"have changed.",
                    remediation="Retry, or use bathymetry source "
                                "'gebco'/'local'.",
                ) from exc
    finally:
        tmp.unlink(missing_ok=True)

    lats = np.linspace(la0, la1, n_lat)
    lons = np.linspace(lo0, lo1, n_lon)
    ri = nearest_indices(glat, lats)
    ci = nearest_indices(glon, lons)
    block = gz[np.ix_(ri, ci)]
    depth = np.where(block < 0.0, -block, np.nan)
    return lats, lons, depth
