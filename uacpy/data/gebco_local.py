"""Sample the locally-cached GEBCO grid (offline bathymetry backend).

Reads the install-time-downloaded GEBCO NetCDF (``elevation(lat, lon)`` on a
regular 15-arc-second global grid, positive up) and returns water depth
(positive down), with the same land-cell semantics as the live OpenTopoData
backend. Unlike the API, there is no per-request rate limit, so arbitrarily
large grids and transects sample instantly once downloaded.
"""

from pathlib import Path

import numpy as np

from uacpy.core.exceptions import DataFetchError
from uacpy.data import _cache
from uacpy.data._geo import as_coordinate, lon_linspace
from uacpy.data._netcdf import NetcdfGrid

__all__ = ['point_depth', 'depths_along', 'region_grid']



class _GebcoGrid(NetcdfGrid):
    """Nearest-cell accessor over the GEBCO ``elevation`` variable."""

    def __init__(self, path):
        try:
            super().__init__(path)
            self._elev = self.var('elevation', 'z')
        except KeyError as exc:
            raise DataFetchError(
                f"GEBCO NetCDF {Path(path).name} is missing an expected "
                f"variable ({exc}); its schema may have changed.",
                remediation="Re-run ./install.sh --data gebco.",
            ) from exc

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
        # Latitude is emitted ascending whatever order the caller gave —
        # every fetch_bathy_grid source shares that axis order. Lon goes
        # eastward so a range crossing the antimeridian (end west of start,
        # e.g. (179, -179)) samples the short strip over 180°, not the long
        # way through 0°.
        lats = np.linspace(min(lat_range), max(lat_range), n_lat)
        lons = lon_linspace(lon_range[0], lon_range[1], n_lon)
        rows = [self.row(v) for v in lats]
        cols = [self.col(v) for v in lons]
        r0, r1 = min(rows), max(rows)
        slab_rows = [r - r0 for r in rows]      # row indices within the slab
        # Eastward longitudes give non-decreasing column indices with at most
        # one wrap back to 0; slicing min..max across that wrap would read all
        # 86 400 columns, so read each contiguous run and stitch.
        wrap = next((i for i in range(1, len(cols)) if cols[i] < cols[i - 1]),
                    None)
        runs = [cols] if wrap is None else [cols[:wrap], cols[wrap:]]
        blocks = []
        for run in runs:
            c0 = min(run)
            slab = np.asarray(self._elev[r0:r1 + 1, c0:max(run) + 1],
                              dtype=float)
            slab_cols = [c - c0 for c in run]
            blocks.append(slab[np.ix_(slab_rows, slab_cols)])
        return lats, lons, np.hstack(blocks)


def _grid():
    path = _cache.require('gebco')
    nc = next((p for p in sorted(path.glob('*.nc'))), None)
    if nc is None:
        # No .nc in the (existing) cache dir → require() raises the typed
        # ConfigurationError naming the install flag; it never returns here.
        _cache.require('gebco', 'GEBCO_2025.nc')
    return _cache.cached_grid_at(nc, _GebcoGrid)


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
    ``lats`` is ascending regardless of the order of ``lat_range``.
    """
    lats, lons, elev = _grid().region(lat_range, lon_range, n_lat, n_lon)
    depth = np.where(elev < 0.0, -elev, np.nan)
    return lats, lons, depth
