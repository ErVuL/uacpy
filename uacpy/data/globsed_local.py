"""Offline GlobSed total sediment thickness (NCEI, public domain).

``install.sh --data globsed`` downloads the GlobSed v3 grid — a global
5-arc-minute ``z(lat, lon)`` field of **total sediment thickness (m)** (Straume
et al. 2019, NOAA NCEI) — into the cache. This module samples it locally.

Sediment thickness is the low-frequency complement to a surficial grain size:
at tens of Hz the field penetrates the whole sediment column, so whether it
reaches basement (and how thick the column is) matters more than the top-cm
texture. Pair it with :mod:`uacpy.data.crust1_local` for a layered bottom.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np

from uacpy._log import log_message
from uacpy.core.exceptions import DataFetchError
from uacpy.data import _cache
from uacpy.data._geo import as_coordinate, normalize_lon
from uacpy.data._http import http_get
from uacpy.data._netcdf import open_netcdf

__all__ = ['download_globsed_db', 'fetch_sediment_thickness',
           'fetch_sediment_thickness_transect']

GLOBSED_FILE = 'GlobSed-v3.nc'
GLOBSED_URL = ('https://www.ncei.noaa.gov/data/oceans/archive/arc0231/0305030/'
               '1.1/data/0-data/GlobSed/GlobSed_package3/GlobSed-v3.nc')

_GRID = {}   # path -> _GlobSedGrid


def _curl_download(url, out, *, timeout, verbose):
    """Fetch ``url`` → ``out`` with curl; ``True`` on success, ``False`` if curl
    is absent or fails. NCEI/Akamai throttles Python urllib to a trickle but
    serves curl at full speed, so this is the preferred path for the grid."""
    curl = shutil.which('curl')
    if not curl:
        return False
    try:
        subprocess.run(
            [curl, '-fL', '--retry', '3', '--max-time', str(int(timeout)),
             '-o', str(out), url],
            check=True, capture_output=not verbose)
    except (subprocess.SubprocessError, OSError):
        out.unlink(missing_ok=True)
        return False
    return out.exists() and out.stat().st_size > 0


def download_globsed_db(cache_dir=None, *, timeout=300.0, verbose=False):
    """Download the GlobSed v3 sediment-thickness grid into the cache.

    Writes ``<cache>/globsed/GlobSed-v3.nc`` and returns the path. Uses curl when
    available (NCEI throttles Python urllib), falling back to the urllib fetcher.
    """
    dest = Path(cache_dir) if cache_dir else _cache.dataset_root('globsed')
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / GLOBSED_FILE
    log_message('globsed', "downloading GlobSed v3 sediment thickness (~11 MB)",
                verbose=verbose)
    if not _curl_download(GLOBSED_URL, out, timeout=timeout, verbose=verbose):
        out.write_bytes(http_get(GLOBSED_URL, timeout=timeout, verbose=verbose,
                                 source='globsed'))
    _GRID.clear()
    log_message('globsed', f"GlobSed grid cached → {out}", verbose=verbose)
    return out


class _GlobSedGrid:
    """Nearest-cell accessor over the GlobSed ``z(lat, lon)`` thickness grid."""

    def __init__(self, path):
        self._ds = open_netcdf(path)
        names = {n.lower(): n for n in self._ds.variables}
        try:
            self._lat = self._ds.variables[names['lat']]
            self._lon = self._ds.variables[names['lon']]
            self._z = self._ds.variables[names['z']]
        except KeyError as exc:
            raise DataFetchError(
                f"GlobSed NetCDF {Path(path).name} is missing an expected "
                f"variable ({exc}); its schema may have changed.",
                remediation="Re-run ./install.sh --data globsed.",
            ) from exc
        self._lat0 = float(self._lat[0])
        self._lon0 = float(self._lon[0])
        self._dlat = float(self._lat[1] - self._lat[0])
        self._dlon = float(self._lon[1] - self._lon[0])
        self._nlat = self._lat.shape[0]
        self._nlon = self._lon.shape[0]

    def _row(self, lat):
        return int(np.clip(round((lat - self._lat0) / self._dlat),
                           0, self._nlat - 1))

    def _col(self, lon):
        return int(np.clip(round((normalize_lon(lon) - self._lon0) / self._dlon),
                           0, self._nlon - 1))

    def thickness(self, lat, lon):
        v = self._z[self._row(lat), self._col(lon)]
        if v is None or np.ma.is_masked(v) or not np.isfinite(v):
            return np.nan
        return float(v)


def _grid():
    path = _cache.require('globsed', GLOBSED_FILE)
    key = str(path)
    if key not in _GRID:
        _GRID[key] = _GlobSedGrid(path)
    return _GRID[key]


def fetch_sediment_thickness(point):
    """Total sediment thickness (m) at a ``(lat, lon)`` point from GlobSed.

    Raises ``DataFetchError`` where GlobSed has no value (land / unmapped).
    """
    lat, lon = as_coordinate(point)
    thk = _grid().thickness(lat, lon)
    if not np.isfinite(thk):
        raise DataFetchError(
            f"GlobSed has no sediment thickness at {lat:.3f}, {lon:.3f} "
            "(land or unmapped).",
            remediation="Pick an offshore point, or supply a thickness directly.",
        )
    return thk


def fetch_sediment_thickness_transect(start, end, n_points=6):
    """``(ranges_m, thickness_m)`` sampled along ``start`` → ``end``.

    ``thickness_m`` is ``NaN`` at any waypoint GlobSed does not cover.
    """
    from uacpy.data.bathymetry import _geodesic_waypoints
    lats, lons, ranges_m = _geodesic_waypoints(start, end, n_points)
    g = _grid()
    thk = np.array([g.thickness(la, lo) for la, lo in zip(lats, lons)])
    return np.asarray(ranges_m), thk
