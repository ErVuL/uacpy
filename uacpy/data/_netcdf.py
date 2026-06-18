"""Shared netCDF4 access for the offline grid readers (GEBCO, GlobSed, WOA23)."""

import numpy as np

from uacpy.core.exceptions import ConfigurationError
from uacpy.data._geo import normalize_lon

__all__ = ['open_netcdf', 'NetcdfGrid']


def open_netcdf(path):
    """Open a local NetCDF file, or raise a typed error if netCDF4 is absent.

    netCDF4 ships with the default install (the offline grid backend); the typed
    error only fires in a stripped-down environment that dropped it.
    """
    try:
        from netCDF4 import Dataset
    except ImportError:
        raise ConfigurationError(
            "Reading local GEBCO/WOA23 grids requires the netCDF4 package.",
            remediation="netCDF4 ships with the default uacpy install; reinstall "
                        "with `pip install -e .`, or `pip install netCDF4`.",
        )
    return Dataset(str(path))


class NetcdfGrid:
    """Nearest-cell accessor over a regular lat/lon NetCDF grid.

    Resolves the ``lat``/``lon`` coordinate variables case-insensitively and
    caches the grid origin/step/extent, so :meth:`row` / :meth:`col` map a
    coordinate to its nearest cell. Subclasses bind whatever data variable(s)
    they expose (GEBCO elevation, GlobSed thickness, …) via :meth:`var`.
    """

    def __init__(self, path):
        self.ds = open_netcdf(path)
        self.names = {n.lower(): n for n in self.ds.variables}
        self._lat = self.ds.variables[self.names['lat']]
        self._lon = self.ds.variables[self.names['lon']]
        self._lat0 = float(self._lat[0])
        self._lon0 = float(self._lon[0])
        self._dlat = float(self._lat[1] - self._lat[0])
        self._dlon = float(self._lon[1] - self._lon[0])
        self._nlat = self._lat.shape[0]
        self._nlon = self._lon.shape[0]

    def var(self, *candidates):
        """Resolve a data variable by case-insensitive name (first match)."""
        for c in candidates:
            if c.lower() in self.names:
                return self.ds.variables[self.names[c.lower()]]
        raise KeyError(candidates)

    def row(self, lat):
        return int(np.clip(round((lat - self._lat0) / self._dlat),
                           0, self._nlat - 1))

    def col(self, lon):
        return int(np.clip(round((normalize_lon(lon) - self._lon0) / self._dlon),
                           0, self._nlon - 1))
