"""Offline GlobSed total sediment thickness (NCEI, public domain).

``install.sh --data globsed`` downloads the GlobSed v3 grid — a global
5-arc-minute ``z(lat, lon)`` field of **total sediment thickness (m)** (Straume
et al. 2019, NOAA NCEI) — into the cache. This module samples it locally.

Sediment thickness is the low-frequency complement to a surficial grain size:
at tens of Hz the field penetrates the whole sediment column, so whether it
reaches basement (and how thick the column is) matters more than the top-cm
texture. Pair it with :mod:`uacpy.data.crust1_local` for a layered bottom.
"""

import os
from pathlib import Path

import numpy as np

from uacpy._log import log_message
from uacpy.core.exceptions import DataFetchError
from uacpy.data import _cache
from uacpy.data._geo import as_coordinate
from uacpy.data._http import curl_download, http_get
from uacpy.data._netcdf import NetcdfGrid

__all__ = ['download_globsed_db', 'fetch_sediment_thickness',
           'fetch_sediment_thickness_transect']

GLOBSED_FILE = 'GlobSed-v3.nc'
GLOBSED_URL = ('https://www.ncei.noaa.gov/data/oceans/archive/arc0231/0305030/'
               '1.1/data/0-data/GlobSed/GlobSed_package3/GlobSed-v3.nc')



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
    if not curl_download(GLOBSED_URL, out, timeout=timeout, verbose=verbose):
        part = Path(str(out) + '.part')
        part.write_bytes(http_get(GLOBSED_URL, timeout=timeout, verbose=verbose,
                                  source='globsed'))
        os.replace(part, out)
    _cache.invalidate_grids()
    log_message('globsed', f"GlobSed grid cached → {out}", verbose=verbose)
    return out


class _GlobSedGrid(NetcdfGrid):
    """Nearest-cell accessor over the GlobSed ``z(lat, lon)`` thickness grid."""

    def __init__(self, path):
        try:
            super().__init__(path)
            self._z = self.var('z')
        except KeyError as exc:
            raise DataFetchError(
                f"GlobSed NetCDF {Path(path).name} is missing an expected "
                f"variable ({exc}); its schema may have changed.",
                remediation="Re-run ./install.sh --data globsed.",
            ) from exc

    def thickness(self, lat, lon):
        return self.cell(self._z, self.row(lat), self.col(lon))


def _grid():
    return _cache.cached_grid('globsed', GLOBSED_FILE, _GlobSedGrid)


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
    from uacpy.data._geo import geodesic_waypoints
    lats, lons, ranges_m = geodesic_waypoints(start, end, n_points)
    g = _grid()
    thk = np.array([g.thickness(la, lo) for la, lo in zip(lats, lons)])
    return np.asarray(ranges_m), thk
