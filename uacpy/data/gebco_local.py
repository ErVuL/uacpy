"""Sample the locally-cached GEBCO grid (offline bathymetry backend).

Reads the install-time-downloaded GEBCO NetCDF (``elevation(lat, lon)`` on a
regular 15-arc-second global grid, positive up) and returns water depth
(positive down), with the same land-cell semantics as the live OpenTopoData
backend. Unlike the API, there is no per-request rate limit, so arbitrarily
large grids and transects sample instantly once downloaded.
"""

import warnings
from pathlib import Path

import numpy as np

from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.data import _cache
from uacpy.data._geo import as_coordinate, lon_linspace
from uacpy.data._netcdf import NetcdfGrid, netcdf_lock

__all__ = ['point_depth', 'depths_along', 'region_grid']



class _GebcoGrid(NetcdfGrid):
    """Nearest-cell accessor over the GEBCO ``elevation`` variable."""

    dataset_name = 'gebco'

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
        # Read only the requested rows/cols (unique, ascending) via netCDF4
        # orthogonal indexing, never the min..max bounding slab: a coarse
        # request over a wide box (e.g. 50×50 over the globe) needs ~50 of the
        # 43 200 rows, while the bounding slab materialises the whole
        # 43 200 × 86 400 grid (tens of GB) to keep 2 500 cells of it.
        u_rows = np.unique(rows)
        pos_rows = np.searchsorted(u_rows, rows)    # row positions in the block
        # Eastward longitudes give non-decreasing column indices with at most
        # one wrap back to 0; reading unique columns across that wrap in one
        # ascending sequence would splice the west of the antimeridian onto
        # the east out of order, so read each contiguous run and stitch.
        wrap = next((i for i in range(1, len(cols)) if cols[i] < cols[i - 1]),
                    None)
        runs = [cols] if wrap is None else [cols[:wrap], cols[wrap:]]
        blocks = []
        for run in runs:
            u_cols = np.unique(run)
            # Same lock and typing as NetcdfGrid.cell: this reads the netCDF
            # variable directly rather than cell by cell.
            with netcdf_lock, _cache.reading('gebco', self.path):
                block = np.asarray(self._elev[u_rows, u_cols], dtype=float)
            pos_cols = np.searchsorted(u_cols, run)
            blocks.append(block[np.ix_(pos_rows, pos_cols)])
        return lats, lons, np.hstack(blocks)


def _grid_path():
    """The cached GEBCO ``.nc`` this process samples: the newest by name."""
    path = _cache.require('gebco')
    # Newest grid first: the files are named GEBCO_<year>.nc, so descending
    # name order ranks releases; names are unique within the directory, and
    # the warning below names the file chosen.
    grids = sorted(path.glob('*.nc'), key=lambda p: p.name, reverse=True)
    nc = grids[0] if grids else None
    if len(grids) > 1:
        warnings.warn(
            f"gebco: {len(grids)} grids are cached in {path} "
            f"({', '.join(p.name for p in grids)}); sampling {nc.name}.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
    if nc is None:
        # The cache dir exists but holds no .nc grid: raise the same typed
        # error require() gives for the canonical missing file.
        ds = _cache.DATASETS['gebco']
        raise ConfigurationError(
            f"Offline gebco data not found: {ds.description} "
            f"({path / 'GEBCO_2025.nc'}).",
            remediation=f"Run `{ds.install_flag}` to download it, or set "
                        f"$UACPY_DATA_CACHE to a directory that has it.",
        )
    return nc


def grid_name() -> str:
    """Vintage of the local grid, from its file name (``'GEBCO_2025'``) — what
    a provenance record cites, since the grid DOI is per release."""
    return _grid_path().stem


def _grid():
    return _cache.cached_grid_at(_grid_path(), _GebcoGrid, 'gebco')


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
