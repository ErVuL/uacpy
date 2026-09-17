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

import numpy as np

from uacpy._log import log_message
from uacpy.core.exceptions import DataFetchError
from uacpy.data import _cache
from uacpy.data._geo import as_coordinate, normalize_lon
from uacpy.data._http import curl_download
from uacpy.data._time import parse_date

__all__ = ['download_wind_db', 'wind_speed', 'climatology_period']

WIND_FILE = 'wind_climatology.npz'
#: The published climatology this fetcher caches: NOAA/NCEI's own blended
#: monthly means over the 1991-2020 WMO reference period, on a 0.25 deg global
#: grid. One request for a file that is already averaged.
NBS_CLIMATOLOGY_URL = (
    'https://www.ncei.noaa.gov/data/blended-global-sea-surface-wind-products/'
    'access/climatology/NBS_v02_wind_climmonthly_s1991_e2020_c20221206.nc')
#: Reference period of :data:`NBS_CLIMATOLOGY_URL`, recorded in the cache so
#: :func:`climatology_period` reports it.
NBS_CLIMATOLOGY_YEARS = (1991, 2020)
#: Name the published file is staged under while its wind speed is extracted.
_NBS_RAW_FILE = 'nbs_climatology.nc'

_CLIM = {}   # path -> _Climatology
_cache.register_cache(_CLIM.clear)


def download_wind_db(cache_dir=None, *, timeout=300.0, verbose=False):
    """Cache NOAA/NCEI's published wind climatology and return its path.

    Downloads :data:`NBS_CLIMATOLOGY_URL` and writes
    ``<cache>/wind/wind_climatology.npz`` (arrays ``lat``, ``lon``, ``speed``
    of shape ``(12, nlat, nlon)``, and ``years``). One request for a file NOAA
    has already averaged over 1991-2020.

    The file carries ``u_wind``, ``v_wind``, ``mask`` and ``windspeed`` on a
    ``(month, zlev, lat, lon)`` grid; only ``windspeed`` is cached, and the
    ``zlev`` axis (one level, 10 m) is dropped, so the cache keeps
    ``(12, nlat, nlon)``. The download itself is deleted afterwards: it is
    237 MB against the ~28 MB the cache keeps.
    """
    from uacpy.data._netcdf import netcdf_lock, open_netcdf

    dest = _cache.prepare_download(
        'wind', "downloading NBS monthly wind climatology "
        f"({NBS_CLIMATOLOGY_YEARS[0]}-{NBS_CLIMATOLOGY_YEARS[1]}, ~237 MB)",
        cache_dir=cache_dir, verbose=verbose)
    out = dest / WIND_FILE
    raw = dest / _NBS_RAW_FILE
    if not curl_download(NBS_CLIMATOLOGY_URL, raw, timeout=timeout,
                         verbose=verbose):
        raise DataFetchError(
            f"Could not download {NBS_CLIMATOLOGY_URL}.",
            remediation="Retry — the transfer resumes — or pass years= to "
                        "build the climatology from monthly fields instead.",
        )
    try:
        with netcdf_lock, contextlib.closing(open_netcdf(str(raw))) as ds:
            lat = np.asarray(ds['lat'][:], dtype=np.float64)
            lon = np.asarray(ds['lon'][:], dtype=np.float64)
            # Masked cells become NaN, which is what every reader of this
            # cache already treats as "no value here".
            speed = np.ma.filled(
                np.ma.masked_invalid(ds['windspeed'][:]).astype(np.float32),
                np.nan)
    finally:
        raw.unlink(missing_ok=True)
    if speed.ndim == 4:                      # (month, zlev, lat, lon)
        speed = speed[:, 0]
    if speed.shape != (12, lat.size, lon.size):
        raise DataFetchError(
            f"Published climatology has shape {speed.shape}, expected "
            f"(12, {lat.size}, {lon.size}).",
            remediation="The upstream file layout changed; pass years= to "
                        "build from monthly fields while this is fixed.",
        )
    with _cache.atomic_write(out) as part:
        # A file object, not the path: np.savez_compressed appends '.npz' to a
        # name that lacks it.
        with open(part, 'wb') as fh:
            np.savez_compressed(
                fh, lat=lat, lon=lon, speed=speed,
                years=np.asarray(NBS_CLIMATOLOGY_YEARS, dtype=np.int32))
    _CLIM.clear()
    log_message('wind', f"wind climatology cached → {out}", verbose=verbose)
    return out


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
                # Optional: caches built before the key existed, and the
                # synthetic ones the tests write, carry no reference period.
                # Absent is not an error — the vintage is simply unstated.
                self.years = ([int(y) for y in data['years']]
                              if 'years' in data.files else None)
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


def climatology_period():
    """Reference period of the installed climatology, or ``None``.

    A string such as ``'2013-2022 (climatology)'`` for the provenance
    ``data_date``. ``None`` where the cache predates the ``years`` key or was
    written without one — the grid is still usable, its vintage is simply not
    recorded.
    """
    years = _clim().years
    if not years:
        return None
    return f"{min(years)}-{max(years)} (climatology)"


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
