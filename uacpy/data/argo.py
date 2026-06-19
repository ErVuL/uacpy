"""Argo float profiles → real in-situ ``SoundSpeedProfile``.

The global Argo array of profiling floats measures temperature and salinity vs
pressure throughout the ice-free ocean. Unlike WOA23 (a monthly *climatology*)
or Copernicus (a *model*), this returns the **nearest actual measured profile**
to a point and date — queried live from the Ifremer **ERDDAP** ``ArgoFloats``
table (no auth, CSV). Sound speed is then computed from T, S and pressure with
the same UNESCO/Del Grosso equation used for the other SSP sources.

Coverage is float-dependent: the nearest profile may be tens–hundreds of km and
days away, so a ``max_distance_km`` / ``max_days`` guard raises when nothing is
close enough — it complements WOA23, it does not replace it.

Argo data are **free and unrestricted** (Argo data policy); please acknowledge
the Argo Program when used.
"""

import csv
import io
from typing import Union

import numpy as np

from uacpy._log import log_message
from uacpy.core.acoustics import soundspeed_delgrosso, soundspeed_unesco
from uacpy.core.environment import SoundSpeedProfile
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data._geo import (
    Coordinate, as_coordinate, normalize_lon, great_circle_km,
)
from uacpy.data._http import http_get
from uacpy.data._time import parse_date

__all__ = ['fetch_argo_profile', 'fetch_ssp_argo']

ARGO_ERDDAP_URL = 'https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.csv'
DEFAULT_MAX_DISTANCE_KM = 250.0
DEFAULT_MAX_DAYS = 15
_FORMULAS = {'unesco': soundspeed_unesco, 'delgrosso': soundspeed_delgrosso}
_GOOD_QC = {'1', '2'}                       # good / probably-good Argo QC flags


def _pressure_dbar_to_depth(pres_dbar, lat):
    """Pressure (dbar) → depth (m): Newton inversion of the depth→pressure law."""
    from uacpy.data._geo import depth_to_pressure_dbar
    p = np.asarray(pres_dbar, dtype=float)
    z = p * 0.9905                                   # ~1 m per dbar initial guess
    for _ in range(5):
        f = depth_to_pressure_dbar(z, lat) - p
        df = (depth_to_pressure_dbar(z + 1.0, lat)
              - depth_to_pressure_dbar(z - 1.0, lat)) / 2.0
        z = z - f / df
    return z


def _query_url(point, when, max_distance_km, max_days, base_url):
    lat, lon = point
    dlat = max_distance_km / 111.0
    dlon = dlat / max(np.cos(np.radians(lat)), 1e-3)
    la0, la1 = lat - dlat, lat + dlat
    lo_lo, lo_hi = normalize_lon(lon) - dlon, normalize_lon(lon) + dlon
    t0 = (when - np.timedelta64(max_days, 'D'))
    t1 = (when + np.timedelta64(max_days, 'D'))
    # A box straddling the antimeridian can't be expressed as a single
    # longitude>=A & longitude<=B clause (it would invert to ~the whole globe);
    # drop the longitude clause there and let the haversine distance guard in
    # the caller select the nearest profile. Only triggers within ``dlon`` of ±180°.
    if lo_lo < -180.0 or lo_hi > 180.0:
        lon_clause = ""
    else:
        lon_clause = f"&longitude%3E={lo_lo:.4f}&longitude%3C={lo_hi:.4f}"
    return (
        f"{base_url}?platform_number,cycle_number,time,latitude,longitude,"
        f"pres,temp,psal,temp_qc,psal_qc"
        f"&time%3E={t0}T00:00:00Z&time%3C={t1}T00:00:00Z"
        f"&latitude%3E={la0:.4f}&latitude%3C={la1:.4f}"
        f"{lon_clause}"
    )


def fetch_argo_profile(
    point: Coordinate, date, *,
    max_distance_km: float = DEFAULT_MAX_DISTANCE_KM,
    max_days: int = DEFAULT_MAX_DAYS,
    base_url: str = ARGO_ERDDAP_URL,
    timeout: float = 60.0,
    verbose: Union[bool, str] = False,
) -> dict:
    """Nearest Argo T/S profile to ``(lat, lon)`` and ``date``.

    Returns ``{'platform', 'cycle', 'lat', 'lon', 'distance_km', 'pres', 'temp',
    'psal'}`` (arrays sorted by increasing pressure, good-QC levels only). Raises
    ``DataFetchError`` when no float profile is within ``max_distance_km`` /
    ``max_days``.
    """
    lat, lon = as_coordinate(point)
    when = np.datetime64(parse_date(date), 'D')
    body = http_get(_query_url((lat, lon), when, max_distance_km, max_days,
                               base_url), timeout=timeout, verbose=verbose,
                    source='argo')
    text = body.decode() if isinstance(body, (bytes, bytearray)) else body
    rows = list(csv.reader(io.StringIO(text)))
    # rows[0] = header, rows[1] = units, rows[2:] = data.
    profiles = {}      # (platform, cycle) -> dict(lat, lon, levels=[(p,t,s)])
    for r in rows[2:]:
        if len(r) < 10:
            continue
        plat, cyc, _t, rlat, rlon, pres, temp, psal, tqc, sqc = r[:10]
        if tqc not in _GOOD_QC or sqc not in _GOOD_QC:
            continue
        try:
            vals = (float(rlat), float(rlon), float(pres), float(temp),
                    float(psal))
        except ValueError:
            continue
        prof = profiles.setdefault((plat, cyc),
                                   {'lat': vals[0], 'lon': vals[1], 'lev': []})
        prof['lev'].append(vals[2:])

    if not profiles:
        raise DataFetchError(
            f"No Argo profile within {max_distance_km:.0f} km / {max_days} days "
            f"of {lat:.3f}, {lon:.3f} on {parse_date(date)}.",
            remediation="Widen max_distance_km / max_days, or use "
                        "ssp_sources='woa23' (climatology).",
        )
    (plat, cyc), prof = min(
        profiles.items(),
        key=lambda kv: great_circle_km(lat, lon, kv[1]['lat'], kv[1]['lon']))
    dist = great_circle_km(lat, lon, prof['lat'], prof['lon'])
    if dist > max_distance_km:
        raise DataFetchError(
            f"Nearest Argo profile is {dist:.0f} km away "
            f"(> max_distance_km={max_distance_km:.0f}).",
            remediation="Widen max_distance_km, or use ssp_sources='woa23'.",
        )
    lev = np.array(sorted(prof['lev']), dtype=float)        # sort by pressure
    return {'platform': plat, 'cycle': cyc, 'lat': prof['lat'],
            'lon': prof['lon'], 'distance_km': float(dist),
            'pres': lev[:, 0], 'temp': lev[:, 1], 'psal': lev[:, 2]}


def fetch_ssp_argo(
    point: Coordinate, date, *,
    formula: str = 'unesco',
    max_distance_km: float = DEFAULT_MAX_DISTANCE_KM,
    max_days: int = DEFAULT_MAX_DAYS,
    base_url: str = ARGO_ERDDAP_URL,
    timeout: float = 60.0,
    verbose: Union[bool, str] = False,
) -> SoundSpeedProfile:
    """Real in-situ sound-speed profile from the nearest Argo float.

    Finds the nearest good-QC Argo T/S profile (:func:`fetch_argo_profile`) and
    converts it with ``formula`` (``'unesco'`` / ``'delgrosso'``). Raises
    ``DataFetchError`` when no profile is close enough.
    """
    if formula not in _FORMULAS:
        raise ConfigurationError(
            f"fetch_ssp_argo: unknown formula={formula!r}.",
            remediation=f"Use one of {sorted(_FORMULAS)}.",
        )
    prof = fetch_argo_profile(point, date, max_distance_km=max_distance_km,
                              max_days=max_days, base_url=base_url,
                              timeout=timeout, verbose=verbose)
    speed_fn = _FORMULAS[formula]
    depths = _pressure_dbar_to_depth(prof['pres'], prof['lat'])
    c = np.array([speed_fn(t, s, p)
                  for t, s, p in zip(prof['temp'], prof['psal'], prof['pres'])])
    log_message(
        'sound_speed', f"Argo SSP from float {prof['platform']} "
        f"({prof['distance_km']:.0f} km away): {depths.size} levels, "
        f"c=[{c.min():.1f}, {c.max():.1f}] m/s", verbose=verbose)
    return SoundSpeedProfile(depths=depths, data=c, shape='measured')
