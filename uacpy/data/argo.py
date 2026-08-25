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
from uacpy.core.exceptions import (ConfigurationError, DataFetchError,
                                   FileFormatError)
from uacpy.data._geo import (
    Coordinate, as_coordinate, normalize_lon, great_circle_km,
    pressure_dbar_to_depth,
)
from uacpy.data._http import http_get
from uacpy.data._time import parse_date
from uacpy.data.sources import SOURCES, DataProvenance

__all__ = ['fetch_argo_profile', 'fetch_ssp_argo']

ARGO_ERDDAP_URL = 'https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.csv'
DEFAULT_MAX_DISTANCE_KM = 250.0
# Tight by design: one Argo profile is a single real in-situ snapshot and the
# ocean decorrelates from it within a couple of weeks, so we accept less
# temporal slack here than for the smoother daily-mean Copernicus model
# (DEFAULT_MAX_DAYS=31). The tolerance tracks how slowly the field varies.
DEFAULT_MAX_DAYS = 15
_FORMULAS = {'unesco': soundspeed_unesco, 'delgrosso': soundspeed_delgrosso}
_GOOD_QC = {'1', '2'}                       # good / probably-good Argo QC flags
# ERDDAP returns columns in the order requested, and the rows below are unpacked
# positionally, so the header is checked against this list before it is trusted.
_COLUMNS = ('platform_number', 'cycle_number', 'direction', 'time', 'latitude',
            'longitude', 'pres', 'temp', 'psal', 'temp_qc', 'psal_qc')


def _abs_days(time_str, when):
    """``|days|`` between an ERDDAP ISO time string and ``when`` (a
    ``datetime64[D]``). Returns ``0.0`` (neutral in the cost) when the time is
    missing or unparseable — ERDDAP times are well-formed, this is just a guard.
    """
    if not time_str:
        return 0.0
    try:
        day = np.datetime64(str(time_str)[:10])
    except (ValueError, TypeError):
        return 0.0
    return abs(float((day - when) / np.timedelta64(1, 'D')))


def _query_url(point, when, max_distance_km, max_days, base_url):
    lat, lon = point
    dlat = max_distance_km / 111.0          # ~111 km per degree of latitude
    dlon = dlat / max(np.cos(np.radians(lat)), 1e-3)
    la0, la1 = lat - dlat, lat + dlat
    lo_lo, lo_hi = normalize_lon(lon) - dlon, normalize_lon(lon) + dlon
    t0 = (when - np.timedelta64(max_days, 'D'))
    t1 = (when + np.timedelta64(max_days, 'D'))
    # A box straddling the antimeridian can't be expressed as a single
    # longitude>=A & longitude<=B clause (it would invert to ~the whole globe);
    # drop the longitude clause there and let the haversine distance filter in
    # the caller enforce the bound. Only triggers within ``dlon`` of ±180°.
    if lo_lo < -180.0 or lo_hi > 180.0:
        lon_clause = ""
    else:
        lon_clause = f"&longitude%3E={lo_lo:.4f}&longitude%3C={lo_hi:.4f}"
    return (
        f"{base_url}?{','.join(_COLUMNS)}"
        f"&time%3E={t0}T00:00:00Z&time%3C={t1}T00:00:00Z"
        f"&latitude%3E={la0:.4f}&latitude%3C={la1:.4f}"
        f"{lon_clause}"
    )


def fetch_argo_profile(
    point: Coordinate, *, date,
    max_distance_km: float = DEFAULT_MAX_DISTANCE_KM,
    max_days: int = DEFAULT_MAX_DAYS,
    base_url: str = ARGO_ERDDAP_URL,
    timeout: float = 60.0,
    verbose: Union[bool, str] = False,
) -> dict:
    """Nearest Argo T/S profile to ``(lat, lon)`` and ``date``.

    Among the good-QC profiles within ``max_distance_km`` and ``max_days``, the
    one nearest in **combined space-time** is returned — each axis normalised by
    its own tolerance and added in quadrature, so a slightly farther but fresher
    float is preferred over a marginally closer but staler one.

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
    if not rows or tuple(c.strip() for c in rows[0][:len(_COLUMNS)]) != _COLUMNS:
        raise FileFormatError(
            f"Argo ERDDAP returned columns {rows[0] if rows else []} rather "
            f"than {list(_COLUMNS)}; the rows are unpacked positionally.",
            remediation="Report this: the ArgoFloats table layout has changed.",
        )
    # A cycle can carry two stations — ERDDAP's own ``direction`` conventions are
    # "A: ascending profiles, D: descending profiles" and its
    # ``cdm_profile_variables`` lists ``direction`` — so the cast identity is
    # (platform, cycle, direction), matching the Argo file naming
    # ``<R|D><float>_<cycle>[D].nc``. Keyed on (platform, cycle) alone, the
    # descent and ascent casts of one cycle interleave into a single column:
    # float 3902110 cycle 463 merges casts 4 days and 22 km apart.
    profiles = {}      # (platform, cycle, direction) -> dict(lat, lon, lev)
    for r in rows[2:]:
        if len(r) < len(_COLUMNS):
            continue
        plat, cyc, dirn, _t, rlat, rlon, pres, temp, psal, tqc, sqc = \
            r[:len(_COLUMNS)]
        if tqc not in _GOOD_QC or sqc not in _GOOD_QC:
            continue
        try:
            vals = (float(rlat), float(rlon), float(pres), float(temp),
                    float(psal))
        except ValueError:
            continue
        prof = profiles.setdefault((plat, cyc, dirn),
                                   {'lat': vals[0], 'lon': vals[1],
                                    'time': _t, 'lev': []})
        prof['lev'].append(vals[2:])

    if not profiles:
        raise DataFetchError(
            f"No Argo profile within {max_distance_km:.0f} km / {max_days} days "
            f"of {lat:.3f}, {lon:.3f} on {parse_date(date)}.",
            remediation="Widen max_distance_km / max_days, or use "
                        "ssp_sources='woa23' (climatology).",
        )
    # The query bounds time to ±max_days, so every candidate already satisfies
    # the temporal tolerance; the lat/lon box only approximates the distance
    # circle (and is dropped near the antimeridian), so enforce max_distance_km
    # here as a hard filter. Among the profiles that pass, pick the nearest in
    # combined space-time: each axis normalised by its own tolerance, so a
    # slightly farther but much fresher float can win over a marginally closer,
    # staler one (a profile at the edge of either tolerance costs the same).
    scored = [(key, p, great_circle_km(lat, lon, p['lat'], p['lon']))
              for key, p in profiles.items()]
    within = [t for t in scored if t[2] <= max_distance_km]
    if not within:
        nearest = min(d_km for _, _, d_km in scored)
        raise DataFetchError(
            f"Nearest Argo profile is {nearest:.0f} km away "
            f"(> max_distance_km={max_distance_km:.0f}).",
            remediation="Widen max_distance_km, or use ssp_sources='woa23'.",
        )

    def _spacetime_cost(item):
        _key, p, d_km = item
        dt_days = _abs_days(p.get('time'), when)
        return (d_km / max_distance_km) ** 2 + (dt_days / max_days) ** 2

    (plat, cyc, dirn), prof, dist = min(within, key=_spacetime_cost)
    lev = np.array(sorted(prof['lev']), dtype=float)        # sort by pressure
    return {'platform': plat, 'cycle': cyc, 'direction': dirn,
            'lat': prof['lat'], 'lon': prof['lon'],
            'distance_km': float(dist), 'time': prof.get('time'),
            'pres': lev[:, 0], 'temp': lev[:, 1], 'psal': lev[:, 2]}


def fetch_ssp_argo(
    point: Coordinate, *, date,
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
    prof = fetch_argo_profile(point, date=date, max_distance_km=max_distance_km,
                              max_days=max_days, base_url=base_url,
                              timeout=timeout, verbose=verbose)
    speed_fn = _FORMULAS[formula]
    depths = pressure_dbar_to_depth(prof['pres'], prof['lat'])
    c = np.array([speed_fn(t, s, p)
                  for t, s, p in zip(prof['temp'], prof['psal'], prof['pres'])])
    log_message(
        'sound_speed', f"Argo SSP from float {prof['platform']} cycle "
        f"{prof['cycle']}{prof['direction']} ({prof['distance_km']:.0f} km "
        f"away): {depths.size} levels, c=[{c.min():.1f}, {c.max():.1f}] m/s",
        verbose=verbose)
    lat, lon = as_coordinate(point)
    prov = DataProvenance(
        source=SOURCES['argo'],
        data_date=(prof['time'][:10] if prof.get('time') else None),  # YYYY-MM-DD
        data_point=(float(prof['lat']), float(prof['lon'])),
        requested_point=(lat, lon),
        requested_date=str(parse_date(date)),
    )
    return SoundSpeedProfile(depths=depths, data=c, shape='measured',
                             data_sources=(prov,))
