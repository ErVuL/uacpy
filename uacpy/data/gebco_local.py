"""Sample the locally-cached GEBCO grid (offline bathymetry backend).

Reads the install-time-downloaded GEBCO NetCDF (``elevation(lat, lon)`` on a
regular 15-arc-second global grid, positive up) and returns water depth
(positive down), with the same land-cell semantics as the live OpenTopoData
backend. Unlike the API, there is no per-request rate limit, so arbitrarily
large grids and transects sample instantly once downloaded.
"""

import numpy as np

from uacpy.core.exceptions import DataFetchError
from uacpy.data import _cache
from uacpy.data._geo import as_coordinate, lon_linspace
from uacpy.data._netcdf import NetcdfGrid

__all__ = ['point_depth', 'depths_along', 'region_grid']

_GRID = {}   # path -> _GebcoGrid (open once, sample many)


class _GebcoGrid(NetcdfGrid):
    """Nearest-cell accessor over the GEBCO ``elevation`` variable."""

    def __init__(self, path):
        super().__init__(path)
        self._elev = self.var('elevation', 'z')

    def elevation(self, lat, lon):
        elev = self.cell(self._elev, self.row(lat), self.col(lon))
        if not np.isfinite(elev):
            raise DataFetchError(
                f"GEBCO has no elevation at ({lat:.4f}, {lon:.4f}) "
                "(fill / masked cell).",
                remediation="Pick a coordinate inside the GEBCO grid.",
            )
        return elev

    def region(self, lat_range, lon_range, n_lat, n_lon):
        # Preserve the caller's axis order (matching the api path), instead of
        # min/max which silently re-sorts. Lat is bounded so it's a plain
        # linspace; lon goes eastward so a range crossing the antimeridian
        # (end west of start, e.g. (179, -179)) samples the short strip over
        # 180°, not the long way through 0°.
        lats = np.linspace(lat_range[0], lat_range[1], n_lat)
        lons = lon_linspace(lon_range[0], lon_range[1], n_lon)
        rows = [self.row(v) for v in lats]
        cols = [self.col(v) for v in lons]
        block = np.asarray(self._elev[min(rows):max(rows) + 1,
                                      min(cols):max(cols) + 1], dtype=float)
        ri = [r - min(rows) for r in rows]
        ci = [c - min(cols) for c in cols]
        return lats, lons, block[np.ix_(ri, ci)]


def _grid():
    path = _cache.require('gebco')
    nc = next((p for p in sorted(path.glob('*.nc'))), None)
    if nc is None:
        # No .nc in the (existing) cache dir → require() raises the typed
        # ConfigurationError naming the install flag; it never returns here.
        _cache.require('gebco', 'GEBCO_2025.nc')
    key = str(nc)
    if key not in _GRID:
        _GRID[key] = _GebcoGrid(nc)
    return _GRID[key]


def _depth_from_elevation(elev, lat, lon):
    if elev >= 0.0:
        raise DataFetchError(
            f"GEBCO reports land (elevation {elev:.0f} m) at "
            f"({lat:.4f}, {lon:.4f}); no water column.",
            remediation="Pick a location offshore, or supply a depth directly.",
        )
    return -elev


def point_depth(point):
    """Water depth (m, positive down) at a single point from the GEBCO grid."""
    lat, lon = as_coordinate(point)
    return _depth_from_elevation(_grid().elevation(lat, lon), lat, lon)


def depths_along(lats, lons):
    """Depths (m, positive down) at paired ``lats``/``lons`` waypoints."""
    g = _grid()
    out = np.empty(len(lats))
    for k, (la, lo) in enumerate(zip(lats, lons)):
        out[k] = _depth_from_elevation(g.elevation(la, lo), la, lo)
    return out


def region_grid(lat_range, lon_range, n_lat, n_lon):
    """``(lats, lons, depth)`` over a region; land cells are NaN.

    Mirrors :func:`uacpy.data.fetch_bathy_grid` but with no request cap.
    """
    lats, lons, elev = _grid().region(lat_range, lon_range, n_lat, n_lon)
    depth = np.where(elev < 0.0, -elev, np.nan)
    return lats, lons, depth
