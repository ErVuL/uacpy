"""NOAA/NCEI Blended Seawinds (NBS) live 10 m wind (public domain).

NBS blends all available scatterometers over an ERA5 background into a global
0.25°, 6-hourly 10 m wind field spanning 1987→present — so it serves any
historical date, not just recent ones. Served live from the NOAA CoastWatch
**ERDDAP** griddap service (no auth), like the Argo / EMODnet fetchers.

The 10 m wind speed feeds two consumers: the Wenz ambient-noise wind term
(:class:`uacpy.noise.WenzNoise`, whose ``wind_speed_kn`` is in **knots** — multiply the
m/s returned here by ``1.9438``) and the Pierson-Moskowitz sea surface
(:func:`uacpy.data.fetch_sea_surface`, when no wave source is available).

NBS is a U.S. Government work — **public domain**.
"""

import numpy as np

from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data._geo import as_coordinate
from uacpy.data._http import (erddap_griddap_url, erddap_last_value,
                              http_get)
from uacpy.data._time import parse_date

__all__ = ['fetch_wind', 'fetch_wind_transect', 'ERDDAP_URL', 'WIND_SOURCES']

ERDDAP_URL = 'https://coastwatch.noaa.gov/erddap/griddap'
DATASET = 'noaacwBlendedWinds6hr'
#: (u, v) component pairs to combine — ``noaacwBlendedWinds6hr`` carries
#: ``u_wind`` / ``v_wind`` — then scalar speed candidates as schema tolerance.
_COMPONENT_VARS = (('u_wind', 'v_wind'), ('u', 'v'))
_SPEED_VARS = ('windspeed', 'wind_speed')
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
    return erddap_griddap_url(ERDDAP_URL, DATASET, var, when, lat, lon,
                              level=10.0)


def _fetch_var(var, when, lat, lon, *, timeout, verbose):
    url = _griddap_url(var, when, lat, lon)
    body = http_get(url, timeout=timeout, verbose=verbose, source='wind',
                    user_agent=_USER_AGENT).decode('utf-8', 'replace')
    return erddap_last_value(body)


def _wind_speed(lat, lon, when, *, timeout, verbose):
    """10 m wind speed (m/s) at a cell: √(u²+v²), else a scalar speed var."""
    last_exc = None
    for u_var, v_var in _COMPONENT_VARS:
        try:
            u = _fetch_var(u_var, when, lat, lon, timeout=timeout,
                           verbose=verbose)
            v = _fetch_var(v_var, when, lat, lon, timeout=timeout,
                           verbose=verbose)
        except DataFetchError as exc:
            last_exc = exc
            continue
        if not (np.isfinite(u) and np.isfinite(v)):
            raise DataFetchError(
                f"NBS has no wind at ({lat:.4f}, {lon:.4f}) on {parse_date(when)}.",
                remediation="Pick an ocean location/date in range, or "
                            "source='local'.",
            )
        return float(np.hypot(u, v))
    for var in _SPEED_VARS:
        try:
            speed = _fetch_var(var, when, lat, lon, timeout=timeout,
                               verbose=verbose)
        except DataFetchError as exc:
            last_exc = exc
            continue
        if np.isfinite(speed):
            return abs(speed)
    raise DataFetchError(
        f"NBS returned no wind speed at ({lat:.4f}, {lon:.4f}): "
        f"{last_exc.message}",
        remediation="Retry, or use source='local' (cached climatology).",
    ) from last_exc


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
    """``(ranges_m, wind_speed_mps)`` sampled along ``start`` → ``end``."""
    from uacpy.data._geo import checked_n_points, geodesic_waypoints
    n_points = checked_n_points(n_points, 'fetch_wind_transect')
    _check_source(source)
    lats, lons, ranges_m = geodesic_waypoints(start, end, n_points)
    speeds = np.array([
        fetch_wind((la, lo), date=date, source=source, timeout=timeout,
                   verbose=verbose)
        for la, lo in zip(lats, lons)])
    return np.asarray(ranges_m), speeds
