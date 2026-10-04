"""NOAA/NCEI Blended Seawinds (NBS) live 10 m wind (public domain).

NBS blends all available scatterometers over an ERA5 background into a global
0.25°, 6-hourly 10 m wind field spanning 1987→present — so it serves any
historical date, not just recent ones. Served live from the NOAA CoastWatch
**ERDDAP** griddap service (no auth), like the Argo / EMODnet fetchers.

The 10 m wind speed feeds two consumers: the Wenz ambient-noise wind term
(:class:`uacpy.noise.WenzNoise`, whose ``wind_speed_kn`` is in **knots**, the unit
returned here) and the Pierson-Moskowitz sea surface
(:func:`uacpy.data.fetch_sea_surface`, when no wave source is available).

NBS is a U.S. Government work — **public domain**.
"""

import numpy as np

from uacpy.core.exceptions import DataFetchError
from uacpy.core.units import ms_to_knots
from uacpy.core.geo import as_coordinate, geodesic_waypoints
from uacpy.data._geo import (AlongTrack, cell_half_diagonal_km, checked_max_distance,
                             checked_n_points, checked_offset, require_source)
from uacpy.data._http import (erddap_griddap_url, erddap_point,
                              http_get)
from uacpy.core.geo import parse_date
from uacpy.data.sources import SOURCES, DataProvenance
from uacpy.data._provenance_notice import one_provenance_notice

__all__ = ['fetch_wind', 'fetch_wind_transect', 'ERDDAP_URL', 'WIND_SOURCES']

ERDDAP_URL = 'https://coastwatch.noaa.gov/erddap/griddap'
DATASET = 'noaacwBlendedWinds6hr'
#: NBS's grid spacing (deg), the cell the offset rule measures against.
NBS_GRID_DEG = 0.25
#: (u, v) component pairs to combine — ``noaacwBlendedWinds6hr`` carries
#: ``u_wind`` / ``v_wind`` — then scalar speed candidates as schema tolerance.
_COMPONENT_VARS = (('u_wind', 'v_wind'), ('u', 'v'))
_SPEED_VARS = ('windspeed', 'wind_speed')
_USER_AGENT = 'uacpy (+https://github.com/ErVuL/uacpy)'

#: The live NBS product by its catalogue id (the id its provenance records
#: carry), and ``'local'`` for its cached monthly climatology.
WIND_SOURCES = ('nbs', 'local')


def _check_source(source):
    require_source(source, WIND_SOURCES, 'wind source',
                   "Use 'nbs' (NBS live) or 'local' (cached monthly "
                   "climatology; see install.sh --data wind).")


def _griddap_url(var, when, lat, lon):
    return erddap_griddap_url(ERDDAP_URL, DATASET, var, when, lat, lon,
                              level=10.0)


def _fetch_var(var, when, lat, lon, *, timeout, verbose):
    """``(value, node)``: ``var`` at the grid node NBS selected for the
    query, and that node's ``(lat, lon)`` (``None`` when unnamed)."""
    url = _griddap_url(var, when, lat, lon)
    body = http_get(url, timeout=timeout, verbose=verbose, source='wind',
                    user_agent=_USER_AGENT).decode('utf-8', 'replace')
    node_lat, node_lon, value = erddap_point(body)
    return value, (None if node_lat is None else (node_lat, node_lon))


def _wind_speed(lat, lon, when, *, timeout, verbose):
    """``(speed, node)``: the 10 m wind speed (m/s) at a cell, √(u²+v²), else
    a scalar speed var, and the ``(lat, lon)`` of the node read."""
    last_exc = None
    for u_var, v_var in _COMPONENT_VARS:
        try:
            u, node = _fetch_var(u_var, when, lat, lon, timeout=timeout,
                                 verbose=verbose)
            v, _ = _fetch_var(v_var, when, lat, lon, timeout=timeout,
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
        return float(np.hypot(u, v)), node
    for var in _SPEED_VARS:
        try:
            speed, node = _fetch_var(var, when, lat, lon, timeout=timeout,
                                     verbose=verbose)
        except DataFetchError as exc:
            last_exc = exc
            continue
        # A speed cannot be negative: a negative number is a fill or a schema
        # change, and reads as no value rather than as its magnitude.
        if np.isfinite(speed) and speed >= 0.0:
            return speed, node
    raise DataFetchError(
        f"NBS returned no wind speed at ({lat:.4f}, {lon:.4f}): "
        f"{last_exc.message}",
        remediation="Retry, or use source='local' (cached climatology).",
    ) from last_exc


def wind_at(point, *, date, source='nbs', timeout=60.0, verbose=False,
            max_distance_km=None, who='fetch_wind'):
    """``(speed in knots, the 'nbs' DataProvenance of the cell read)`` at a
    ``(lat, lon)`` point: the reading :func:`fetch_wind` returns and the
    record :func:`uacpy.data.fetch_sea_surface` stamps, through the offset
    rule (a warning once the cell read is not the point's own 0.25° cell, a
    refusal past ``max_distance_km``)."""
    _check_source(source)
    lat, lon = as_coordinate(point)
    if source == 'local':
        from uacpy.data import wind_local
        speed = wind_local.wind_speed(point, date=date)
        node = wind_local.wind_cell(point)
    else:
        speed, node = _wind_speed(lat, lon, date, timeout=timeout,
                                  verbose=verbose)
    prov = checked_offset(
        DataProvenance(source=SOURCES['nbs'], data_point=node,
                       requested_point=(lat, lon), point_kind='cell',
                       cell_size_deg=NBS_GRID_DEG),
        who=who,
        warn_km=cell_half_diagonal_km(lat if node is None else node[0],
                                      NBS_GRID_DEG, NBS_GRID_DEG),
        max_distance_km=max_distance_km)
    return float(ms_to_knots(speed)), prov


def fetch_wind(point, *, date, source='nbs', timeout=60.0, verbose=False,
               max_distance_km=None):
    """10 m wind speed (knots) at a ``(lat, lon)`` point and date: the
    source's m/s, converted here.

    ``source='nbs'`` queries NBS live over ERDDAP (date-specific);
    ``source='local'``
    reads the cached monthly climatology (``install.sh --data wind``). Raises
    ``DataFetchError`` where no wind is available (land / out of range).

    The value is raw and carries no provenance;
    :func:`uacpy.data.fetch_sea_surface` returns the carrier that records
    it in ``.data_sources``.

    Parameters
    ----------
    point : (lat, lon)
        Site coordinates in decimal degrees.
    date : str or datetime.date
        Calendar date.
    source : {'nbs', 'local'}, optional
        NBS live (default) or the cached monthly climatology.
    timeout : float, optional
        Per-request network timeout in seconds.
    verbose : bool or str, optional
        Logging gate passed through to ``log_message``.
    max_distance_km : float, optional
        Refuse a cell standing farther than this (km) from ``point`` (the
        offset rule); ``None`` (default) sets no limit beyond its warning.
    """
    speed, _prov = wind_at(point, date=date, source=source, timeout=timeout,
                           verbose=verbose, max_distance_km=checked_max_distance(
                               max_distance_km, 'fetch_wind'))
    return speed


@one_provenance_notice(subject='the samples',
                       record="the result's .provenance")
def fetch_wind_transect(start, end, *, date, n_points=6, source='nbs',
                        timeout=60.0, verbose=False, max_distance_km=None):
    """The 10 m wind speed (knots) sampled along ``start`` → ``end``, as an
    :class:`~uacpy.data.AlongTrack` (``'wind_speed'``) with the ``'nbs'``
    provenance and the requested date.

    Parameters
    ----------
    start, end : (lat, lon)
        Transect endpoints in decimal degrees.
    date : str or datetime.date
        Calendar date.
    n_points : int, optional
        Waypoints along the great circle. Default 6.
    source : {'nbs', 'local'}, optional
        NBS live (default) or the cached monthly climatology.
    timeout : float, optional
        Per-request network timeout in seconds.
    verbose : bool or str, optional
        Logging gate passed through to ``log_message``.
    max_distance_km : float, optional
        Refuse a waypoint whose cell stands farther than this (km) from it;
        ``None`` (default) sets no limit beyond the per-waypoint warning.
    """
    n_points = checked_n_points(n_points, 'fetch_wind_transect')
    limit = checked_max_distance(max_distance_km, 'fetch_wind_transect')
    _check_source(source)
    lats, lons, ranges_m = geodesic_waypoints(start, end, n_points)
    speeds = np.array([
        wind_at((la, lo), date=date, source=source, timeout=timeout,
                verbose=verbose, max_distance_km=limit,
                who='fetch_wind_transect')[0]
        for la, lo in zip(lats, lons)])
    return AlongTrack(ranges=np.asarray(ranges_m), lats=np.asarray(lats),
                      lons=np.asarray(lons), data=speeds, unit='kn',
                      quantity='wind_speed',
                      provenance=DataProvenance(
                          source=SOURCES['nbs'],
                          requested_date=parse_date(date).isoformat()))
