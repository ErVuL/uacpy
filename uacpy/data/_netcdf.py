"""Shared netCDF4 access for the offline grid readers (GEBCO, GlobSed, WOA23)."""

import numpy as np

from uacpy.core.exceptions import ConfigurationError

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

    Origin and step are read from the file's own axis variables, so the class is
    registration-agnostic: it snaps to the nearest **stored node** whether those
    nodes are cell centres (GEBCO 2025 starts at -89.99791667, half a 15" cell
    inside the pole) or cell edges (GlobSed v3 starts at exactly -90.0). The
    step is signed, so a north-up (descending-latitude) file needs no special
    case. Uniform spacing is assumed — the step is taken from the first two
    nodes.
    """

    def close(self) -> None:
        """Close the underlying netCDF handle (``invalidate_grids`` calls
        this on every memoised grid before dropping it)."""
        ds = getattr(self, 'ds', None)
        if ds is not None:
            try:
                ds.close()
            except Exception:
                pass        # a torn handle must not block the invalidation

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

    @staticmethod
    def cell(variable, row, col):
        """Scalar at ``[row, col]`` with ``_FillValue``/masked → ``NaN``.

        netCDF4 returns a masked array when ``_FillValue`` is set, and
        ``float(masked)`` raises or yields garbage; map any masked / non-finite
        cell to ``NaN`` so every reader handles no-data uniformly.
        """
        v = variable[row, col]
        if v is None or np.ma.is_masked(v) or not np.isfinite(v):
            return np.nan
        return float(v)

    def row(self, lat):
        """Index of the node nearest ``lat``; latitude is bounded, so a value
        past either pole clamps to the first/last row rather than wrapping."""
        return int(np.clip(round((lat - self._lat0) / self._dlat),
                           0, self._nlat - 1))

    def col(self, lon):
        """Index of the node nearest ``lon``, wrapped into the stored axis."""
        # Wrap the query into this grid's OWN longitude convention so every
        # stored origin resolves: [-180, 180) (GEBCO), [0, 360), or an offset
        # one (GLODAP's mapped product starts at 20.5°E). After the wrap the
        # query is inside one period of the axis, and clip only guards a grid
        # that does not span the full 360°.
        # Only a full-360 axis is periodic, under either registration:
        # cell/pixel-registered nodes span nlon*dlon degrees (GEBCO, GLODAP),
        # while a gridline-registered axis stores both meridian ends and spans
        # (nlon-1)*dlon (GlobSed: 4321 nodes at 5'). Wrapping a
        # partial-coverage grid would silently send a just-west-of-edge query
        # to the east edge, so those clip on the raw longitude instead.
        if (abs(self._nlon * self._dlon - 360.0) < 1e-6
                or abs((self._nlon - 1) * self._dlon - 360.0) < 1e-6):
            lon = self._lon0 + ((float(lon) - self._lon0) % 360.0)
            return int(round((lon - self._lon0) / self._dlon)) % self._nlon
        col = int(round((float(lon) - self._lon0) / self._dlon))
        return int(np.clip(col, 0, self._nlon - 1))
