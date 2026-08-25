"""Offline NBS 10 m wind-speed monthly climatology.

``install.sh --data wind`` builds a **monthly climatology** of NBS 10 m wind
speed: for each calendar month it averages the NBS monthly-mean global fields
over a period of years into a 0.25° ``speed(12, lat, lon)`` grid, cached as an
``.npz``. This module then returns the climatological wind speed at a point and
month — the offline, reproducible analogue of the live NBS fetch (also the
mean-state fallback, so it understates day-to-day sea state; the live
:func:`uacpy.data.wind_live.fetch_wind` is preferred for a specific date).

NBS is a U.S. Government work — **public domain**.
"""

import contextlib
import tempfile
from pathlib import Path

import numpy as np

from uacpy._log import log_message
from uacpy.core.exceptions import DataFetchError
from uacpy.data import _cache
from uacpy.data._geo import as_coordinate, normalize_lon
from uacpy.data._http import http_get
from uacpy.data._time import parse_date

__all__ = ['download_wind_db', 'wind_speed']

WIND_FILE = 'wind_climatology.npz'
_MONTHLY_DATASET = 'noaacwBlendedWindsMonthly'
_ERDDAP = 'https://coastwatch.noaa.gov/erddap/griddap'
_SPEED_VARS = ('windspeed', 'wind_speed', 'w')
_DEFAULT_YEARS = tuple(range(2013, 2023))            # a recent decade
_USER_AGENT = 'uacpy (+https://github.com/ErVuL/uacpy)'

_CLIM = {}   # path -> _Climatology
_cache.register_cache(_CLIM.clear)


def download_wind_db(cache_dir=None, *, years=_DEFAULT_YEARS, timeout=120.0,
                     verbose=False):
    """Build the NBS monthly wind-speed climatology and cache it.

    Averages the NBS monthly-mean global fields over ``years`` per calendar
    month, writing ``<cache>/wind/wind_climatology.npz`` (arrays ``lat``,
    ``lon``, ``speed`` of shape ``(12, nlat, nlon)``). Missing months are
    skipped. Returns the path.
    """
    dest = Path(cache_dir) if cache_dir else _cache.dataset_root('wind')
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / WIND_FILE
    log_message('wind', f"building NBS wind climatology over {years[0]}-"
                f"{years[-1]} (~{len(years) * 12} monthly grids)", verbose=verbose)
    lat = lon = None
    accum, count = None, None
    for month in range(1, 13):
        for year in years:
            grid = _fetch_monthly_grid(year, month, timeout=timeout,
                                       verbose=verbose)
            if grid is None:
                continue
            glat, glon, speed = grid
            if accum is None:
                lat, lon = glat, glon
                accum = np.zeros((12, lat.size, lon.size))
                count = np.zeros((12, lat.size, lon.size))
            valid = np.isfinite(speed)
            accum[month - 1][valid] += speed[valid]
            count[month - 1][valid] += 1
    if accum is None:
        raise DataFetchError(
            "NBS wind climatology build fetched no monthly grids.",
            remediation="Check network / the CoastWatch ERDDAP dataset id.",
        )
    with np.errstate(invalid='ignore'):
        speed = np.where(count > 0, accum / np.maximum(count, 1), np.nan)
    with _cache.atomic_write(out) as part:
        # A file object, not the path: np.savez_compressed appends '.npz' to a
        # name that lacks it, which would write '<out>.part.npz'.
        with open(part, 'wb') as fh:
            np.savez_compressed(fh, lat=lat, lon=lon, speed=speed)
    _CLIM.clear()
    log_message('wind', f"wind climatology cached → {out}", verbose=verbose)
    return out


def _fetch_monthly_grid(year, month, *, timeout, verbose):
    """One NBS monthly-mean global wind-speed field, or ``None`` on failure."""
    import urllib.parse
    from uacpy.data._netcdf import netcdf_lock, open_netcdf
    iso = f"{year}-{month:02d}-01T00:00:00Z"
    for var in _SPEED_VARS:
        constraint = f"{var}[({iso})][(10.0)][][]"
        query = urllib.parse.quote(constraint, safe='[]():.,-TZ')
        url = f"{_ERDDAP}/{_MONTHLY_DATASET}.nc?{query}"
        try:
            blob = http_get(url, timeout=timeout, verbose=verbose, source='wind',
                            user_agent=_USER_AGENT)
        except DataFetchError:
            continue
        with tempfile.NamedTemporaryFile(suffix='.nc', prefix='uacpy_nbs_',
                                         delete=False) as fh:
            fh.write(blob)
            tmp = Path(fh.name)
        try:
            # closing(), not a bare close() after the reads: the KeyError the
            # next clause catches is raised between the two, once per month a
            # 120-grid climatology build cannot name a variable in — each one
            # leaking a handle on the file the finally is about to unlink.
            #
            # netcdf_lock spans the whole statement, so the slices below and
            # the close() that ends it are inside it as well as the open —
            # netCDF4 is not thread-safe here, and a 120-file climatology
            # build would otherwise read and close alongside another thread's
            # grid read.
            with netcdf_lock, contextlib.closing(open_netcdf(tmp)) as ds:
                names = {n.lower(): n for n in ds.variables}
                lat = np.asarray(ds.variables[names['latitude']][:], float)
                lon = np.asarray(ds.variables[names['longitude']][:], float)
                # np.asarray() first would strip the netCDF4 mask before
                # filled() ever sees it, letting a _FillValue cell (land / no
                # retrieval) into the monthly mean as a real wind speed.
                # Convert *as* a masked array, then fill through the mask.
                speed = np.ma.filled(np.ma.asarray(
                    ds.variables[names[var.lower()]][:], float).squeeze(),
                    np.nan)
        except (KeyError, OSError):
            continue
        finally:
            tmp.unlink(missing_ok=True)
        return lat, lon, speed
    return None


class _Climatology:
    """Nearest-cell accessor over the cached ``speed(12, lat, lon)`` grid."""

    def __init__(self, path):
        with _cache.reading('wind', path):
            # ``with``, like every sibling reader: each member access can
            # raise on a damaged cache file, and the error's own remediation
            # invites a retry — so an unclosed NpzFile leaks one descriptor
            # per attempt. allow_pickle=False is passed rather than left to
            # numpy's default: this file is read straight out of the cache
            # directory, and under allow_pickle an object array in it would
            # execute on load.
            with np.load(path, allow_pickle=False) as data:
                self.lat = data['lat']
                self.lon = data['lon']
                self.speed = data['speed']
        self._lat0 = float(self.lat[0])
        self._lon0 = float(self.lon[0])
        self._dlat = float(self.lat[1] - self.lat[0])
        self._dlon = float(self.lon[1] - self.lon[0])

    def at(self, lat, lon, month):
        """Climatological speed (m/s) at the nearest cell for ``month`` (1-12)."""
        row = int(np.clip(round((lat - self._lat0) / self._dlat),
                          0, self.lat.size - 1))
        # NBS serves longitude on [0, 360) (0 → 359.75 at 0.25°) and the cache
        # keeps that axis, so wrap the query into it before indexing (the same
        # modulo _netcdf.NetcdfGrid.col applies). Latitude is south-up.
        lon = self._lon0 + ((normalize_lon(lon) - self._lon0) % 360.0)
        # The axis spans a full 360 deg, so the index wraps: the half-cell
        # west of the origin belongs to column 0, not the last column
        # (same rule as _netcdf.NetcdfGrid.col).
        col = int(round((lon - self._lon0) / self._dlon)) % self.lon.size
        return float(self.speed[month - 1, row, col])


def _clim():
    """Load (or reuse) the monthly climatology.

    Built through :func:`uacpy.data._cache.memoize`, so threads racing a cold
    memo read the ``.npz`` once between them rather than once each.
    """
    path = _cache.require('wind', WIND_FILE)
    return _cache.memoize(_CLIM, str(path), lambda: _Climatology(path))


def wind_speed(point, *, date):
    """Climatological 10 m wind speed (m/s) at a ``(lat, lon)`` point and month.

    Raises ``DataFetchError`` where the climatology has no value (land).
    """
    lat, lon = as_coordinate(point)
    month = parse_date(date).month
    speed = _clim().at(lat, lon, month)
    if not np.isfinite(speed):
        raise DataFetchError(
            f"NBS wind climatology has no value at {lat:.3f}, {lon:.3f} (land).",
            remediation="Pick an ocean location, or supply a wind speed directly.",
        )
    return speed
