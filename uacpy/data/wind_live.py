"""NOAA/NCEI Blended Seawinds (NBS) live 10 m wind (public domain).

NBS blends all available scatterometers over an ERA5 background into a global
0.25°, 6-hourly 10 m wind field spanning 1987→present — so it serves any
historical date, not just recent ones. Served live from the NOAA CoastWatch
**ERDDAP** griddap service (no auth), like the Argo / EMODnet fetchers.

The 10 m wind speed feeds two consumers: the Wenz ambient-noise wind term
(:class:`uacpy.WenzNoise`, whose ``wind_speed`` is in **knots** — multiply the
m/s returned here by ``1.9438``) and the Pierson-Moskowitz sea surface
(:func:`uacpy.data.fetch_sea_surface`, when no wave source is available).

NBS is a U.S. Government work — **public domain**.
"""

import numpy as np

from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data._geo import as_coordinate, normalize_lon
from uacpy.data._http import http_get
from uacpy.data._time import parse_date

__all__ = ['fetch_wind', 'fetch_wind_transect', 'ERDDAP_URL', 'WIND_SOURCES']

ERDDAP_URL = 'https://coastwatch.noaa.gov/erddap/griddap'
DATASET = 'noaacwBlendedWinds6hr'
#: Wind-speed variable candidates, then the (u, v) components to combine when no
#: scalar speed variable is exposed.
_SPEED_VARS = ('windspeed', 'wind_speed', 'w')
_COMPONENT_VARS = ('u', 'v')
_USER_AGENT = 'uacpy (+https://github.com/ErVuL/uacpy)'

WIND_SOURCES = ('erddap', 'local')


def _check_source(source):
    if source not in WIND_SOURCES:
        raise ConfigurationError(
            f"wind source must be one of {WIND_SOURCES}; got {source!r}.",
            remediation="Use 'erddap' (NBS live) or 'local' (cached monthly "
                        "climatology; see install.sh --data wind).",
        )


def _griddap_url(var, when, lat, lon):
    """ERDDAP griddap CSV URL for ``var`` at the nearest time/lat/lon cell.

    NBS carries a singleton 10 m ``zlev`` axis between time and latitude; the
    ``[(...)]`` value selectors snap each axis to its nearest node.
    """
    import urllib.parse
    iso = f"{parse_date(when)}T00:00:00Z"
    constraint = (f"{var}[({iso})][(10.0)][({lat})][({normalize_lon(lon)})]")
    query = urllib.parse.quote(constraint, safe='[]():.,-TZ')
    return f"{ERDDAP_URL}/{DATASET}.csv?{query}"


def _last_value(body):
    """Last numeric field of an ERDDAP griddap ``.csv`` point response."""
    rows = [ln for ln in body.splitlines() if ln.strip()]
    if len(rows) < 3:
        return np.nan
    try:
        return float(rows[-1].split(',')[-1])
    except (ValueError, IndexError):
        return np.nan


def _fetch_var(var, when, lat, lon, *, timeout, verbose):
    url = _griddap_url(var, when, lat, lon)
    body = http_get(url, timeout=timeout, verbose=verbose, source='wind',
                    user_agent=_USER_AGENT).decode('utf-8', 'replace')
    return _last_value(body)


def _wind_speed(lat, lon, when, *, timeout, verbose):
    """10 m wind speed (m/s) at a cell: a scalar speed var, else √(u²+v²)."""
    for var in _SPEED_VARS:
        try:
            speed = _fetch_var(var, when, lat, lon, timeout=timeout,
                               verbose=verbose)
        except DataFetchError:
            continue                              # variable absent — try next
        if np.isfinite(speed):
            return abs(speed)
    try:
        u = _fetch_var(_COMPONENT_VARS[0], when, lat, lon, timeout=timeout,
                       verbose=verbose)
        v = _fetch_var(_COMPONENT_VARS[1], when, lat, lon, timeout=timeout,
                       verbose=verbose)
    except DataFetchError as exc:
        raise DataFetchError(
            f"NBS returned no wind speed at ({lat:.4f}, {lon:.4f}): {exc.message}",
            remediation="Retry, or use source='local' (cached climatology).",
        ) from exc
    if not (np.isfinite(u) and np.isfinite(v)):
        raise DataFetchError(
            f"NBS has no wind at ({lat:.4f}, {lon:.4f}) on {parse_date(when)}.",
            remediation="Pick an ocean location/date in range, or source='local'.",
        )
    return float(np.hypot(u, v))


def fetch_wind(point, *, date, source='erddap', timeout=60.0, verbose=False):
    """10 m wind speed (m/s) at a ``(lat, lon)`` point and date.

    ``source='erddap'`` queries NBS live (date-specific); ``source='local'``
    reads the cached monthly climatology (``install.sh --data wind``). Raises
    ``DataFetchError`` where no wind is available (land / out of range).
    """
    _check_source(source)
    lat, lon = as_coordinate(point)
    if source == 'local':
        from uacpy.data import wind_local
        return wind_local.wind_speed(point, date=date)
    return _wind_speed(lat, lon, date, timeout=timeout, verbose=verbose)


def fetch_wind_transect(start, end, *, date, n_points=6, source='erddap',
                        timeout=60.0, verbose=False):
    """``(ranges_m, wind_speed_ms)`` sampled along ``start`` → ``end``."""
    from uacpy.data._geo import geodesic_waypoints
    _check_source(source)
    lats, lons, ranges_m = geodesic_waypoints(start, end, n_points)
    speeds = np.array([
        fetch_wind((la, lo), date=date, source=source, timeout=timeout,
                   verbose=verbose)
        for la, lo in zip(lats, lons)])
    return np.asarray(ranges_m), speeds
