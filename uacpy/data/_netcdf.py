"""Shared netCDF4 access for the offline grid readers (GEBCO, GlobSed, WOA23)."""

import threading
import warnings

import numpy as np

from uacpy.core.exceptions import ConfigurationError
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.data import _cache

__all__ = ['open_netcdf', 'netcdf_lock', 'NetcdfGrid']

# netCDF4 here sits on an HDF5 build without thread safety, so two threads
# inside the library at once corrupt its shared state: 8 threads reading one
# already-open GlobSed handle raise ``RuntimeError: NetCDF: HDF error`` or take
# the interpreter down with a SIGSEGV, and 8 threads opening one file at once
# segfault outright.
#
# Three call kinds cross out of Python into that library, and this lock is held
# across every one of them package-wide: ``Dataset(...)`` opens, ``Variable``
# slicing (``v[i, j]``, ``v[:]``), and ``Dataset.close()`` — a close landing
# inside another thread's read exits 139/135. Naming a variable is not one of
# them: ``Dataset.variables`` is a plain dict built at open time (verified — the
# same ``dict`` object on every access, and still readable after ``close()``),
# so ``ds.variables[name]`` never leaves Python and is safe outside the lock.
#
# Process-wide rather than per grid because the unsafe state belongs to the
# library, not the handle: two threads reading two DIFFERENT already-open files
# collide the same way two reading one do — 8 threads over two files, unlocked,
# exit 139 on 3 of 3 runs. Reentrant so an open that reads its own axis
# variables can nest inside itself.
netcdf_lock = threading.RLock()

# Slack (degrees) on the half-cell coverage test, so a global cell-registered
# grid queried at exactly a pole reads as in-coverage: GEBCO's outermost centre
# sits half a 15" cell inside 90 deg, and that half-cell divides to 0.5000000393
# rather than 0.5 in float. 1e-6 deg is ~0.1 m — far below any grid's step.
_COVERAGE_TOL_DEG = 1e-6


def open_netcdf(path):
    """Open a local NetCDF file, or raise a typed error if netCDF4 is absent.

    netCDF4 ships with the default install (the offline grid backend); the typed
    error only fires in a stripped-down environment that dropped it.

    The open itself holds :data:`netcdf_lock`; concurrent opens of one file
    segfault the interpreter without it.
    """
    try:
        from netCDF4 import Dataset
    except ImportError:
        raise ConfigurationError(
            "Reading local GEBCO/WOA23 grids requires the netCDF4 package.",
            remediation="netCDF4 ships with the default uacpy install; reinstall "
                        "with `pip install -e .`, or `pip install netCDF4`.",
        )
    with netcdf_lock:
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
        this on every memoised grid before dropping it).

        Holds :data:`netcdf_lock` like every other call into the library, so a
        close cannot land inside another thread's read.
        """
        ds = getattr(self, 'ds', None)
        if ds is not None:
            try:
                with netcdf_lock:
                    ds.close()
            except Exception:
                pass        # a torn handle must not block the invalidation

    # Dataset this grid belongs to, for the typed error :meth:`cell` raises on
    # an unreadable cell; subclasses name their own so the remediation can name
    # the install flag that re-downloads it.
    dataset_name = 'cached'

    def __init__(self, path):
        self.path = path
        self.ds = open_netcdf(path)
        with netcdf_lock:
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
        """Resolve a data variable by case-insensitive name (first match).

        A lookup in the handle's variable dict, not a read: it hands back the
        ``Variable`` object without entering netCDF4, so it is the one call
        site here that takes no :data:`netcdf_lock` (see the module comment).
        Slicing what it returns does enter, and every such slice is locked.
        """
        for c in candidates:
            if c.lower() in self.names:
                return self.ds.variables[self.names[c.lower()]]
        raise KeyError(candidates)

    def cell(self, variable, row, col):
        """Scalar at ``[row, col]`` with ``_FillValue``/masked → ``NaN``.

        netCDF4 returns a masked array when ``_FillValue`` is set, and
        ``float(masked)`` raises or yields garbage; map any masked / non-finite
        cell to ``NaN`` so every reader handles no-data uniformly.

        The read holds :data:`netcdf_lock` and runs under
        :func:`uacpy.data._cache.reading`. The lock is what makes a concurrent
        read safe at all — without it eight threads on one warm handle raise
        ``NetCDF: HDF error`` or segfault. ``reading`` covers what the lock
        cannot: a damaged chunk in a file the cache accepts still throws from
        inside HDF5, and an untyped throw aborts the ``environment``
        source-fallback chains, which catch ``DataFetchError`` and
        ``ConfigurationError`` only.
        """
        with netcdf_lock, _cache.reading(self.dataset_name, self.path):
            v = variable[row, col]
        if np.ma.is_masked(v) or not np.isfinite(v):
            return np.nan
        return float(v)

    def _bounded(self, axis, value, first, step, n):
        """Index of the node nearest ``value`` on a bounded (non-wrapping) axis.

        Warns when the query lies outside the axis' coverage — more than half a
        step past the first or last node — because the clamp then hands back an
        edge node's value as if it had been sampled at the query, which no
        caller can tell from a real reading. Every grid this class serves today
        spans its axis globally, so the warning is the first regional grid's.
        """
        lo, hi = sorted((first, first + (n - 1) * step))
        half = abs(step) / 2.0 + _COVERAGE_TOL_DEG
        if not lo - half <= value <= hi + half:
            warnings.warn(
                f"{type(self).__name__}: {axis} {value:g} is outside this "
                f"grid's coverage ({lo:g} to {hi:g}); returning the nearest "
                f"edge node's value, which was not sampled at the query.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
        return int(np.clip(round((value - first) / step), 0, n - 1))

    def row(self, lat):
        """Index of the node nearest ``lat``; latitude is bounded, so a value
        past either pole clamps to the first/last row rather than wrapping.
        A query outside the grid's own latitude coverage warns first."""
        return self._bounded('latitude', float(lat), self._lat0, self._dlat,
                             self._nlat)

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
        # to the east edge, so those clip on the raw longitude instead — and
        # warn when the query is off the grid altogether.
        if (abs(self._nlon * self._dlon - 360.0) < 1e-6
                or abs((self._nlon - 1) * self._dlon - 360.0) < 1e-6):
            lon = self._lon0 + ((float(lon) - self._lon0) % 360.0)
            return int(round((lon - self._lon0) / self._dlon)) % self._nlon
        return self._bounded('longitude', float(lon), self._lon0, self._dlon,
                             self._nlon)
