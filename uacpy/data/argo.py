"""Argo float profiles → real in-situ ``SoundSpeedProfile``.

The global Argo array of profiling floats measures temperature and salinity vs
pressure throughout the ice-free ocean. Unlike WOA23 (a monthly *climatology*)
or Copernicus (a *model*), this returns the **nearest actual measured profile**
to a point and date — queried live from the Ifremer **ERDDAP** ``ArgoFloats``
table (no auth, CSV). Sound speed is then computed from T, S and pressure with
the same ``formula`` as the other SSP sources — TEOS-10 by default; UNESCO,
Del Grosso or Mackenzie on request.

Coverage is float-dependent: the nearest profile may be tens–hundreds of km and
days away, so a ``max_distance_km`` / ``max_days`` guard raises when nothing is
close enough — it complements WOA23, it does not replace it.

Argo data are **free and unrestricted** (Argo data policy); please acknowledge
the Argo Program when used.
"""

import csv
import io
import math
import urllib.error
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from uacpy._log import log_message
from uacpy.core._export import ExportRecord
from uacpy.core.environment import SoundSpeedProfile
from uacpy.core.exceptions import (ConfigurationError, DataFetchError,
                                   FileFormatError)
from uacpy.core.acoustics.seawater import (
    DEFAULT_SOUND_SPEED_FORMULA, SOUND_SPEED_FORMULAS, canonical_formula,
    pressure_dbar_to_depth,
    sound_speed_mackenzie,
)
from uacpy.core.geo import (
    Coordinate, as_coordinate, normalize_lon, great_circle_km,
)
from uacpy.data._geo import checked_max_distance, checked_offset
from uacpy.data._http import http_get
from uacpy.core.geo import parse_date
from uacpy.data.sources import SOURCES, DataProvenance

from uacpy.data._offset_policy import ARGO_OFFSET_WARN_KM
from uacpy.data._offset_policy import checked_date_offset

__all__ = ['ArgoProfile', 'fetch_argo_profile', 'fetch_ssp_argo']

ARGO_ERDDAP_URL = 'https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.csv'
#: How far (km) the Argo search reaches for a cast when the caller sets no
#: ``max_distance_km``: the radius of the query and of the space-time ranking.
ARGO_SEARCH_RADIUS_KM = 250.0
# Tight by design: one Argo profile is a single real in-situ snapshot and the
# ocean decorrelates from it within a couple of weeks, so we accept less
# temporal slack here than for the smoother daily-mean Copernicus model
# (DEFAULT_MAX_DAYS=31). The tolerance tracks how slowly the field varies.
DEFAULT_MAX_DAYS = 15
_GOOD_QC = {'1', '2'}                       # good / probably-good Argo QC flags
# ERDDAP returns columns in the order requested, and the rows below are unpacked
# positionally, so the header is checked against this list before it is trusted.
# ``position_qc`` is the cast's position flag: a cast whose position is not
# good or probably good is dropped, since it describes water somewhere else.
# ``data_mode`` says which triplet a level is read from: the ArgoFloats table's
# convention is "R : real time; D : delayed mode; A : real time with
# adjustment", and in A and D mode the ``*_adjusted`` values carry the sensor
# correction and the ``*_adjusted_qc`` flags carry the adjusted verdict.
_COLUMNS = ('platform_number', 'cycle_number', 'direction', 'time', 'latitude',
            'longitude', 'pres', 'temp', 'psal', 'temp_qc', 'psal_qc',
            'pres_qc', 'position_qc', 'data_mode', 'pres_adjusted',
            'temp_adjusted', 'psal_adjusted', 'pres_adjusted_qc',
            'temp_adjusted_qc', 'psal_adjusted_qc')
# The values each ``data_mode`` reads, as the provenance ``product`` names
# them. A level whose ``data_mode`` is not one of these is dropped.
_DATA_MODE_VALUES = {
    'R': 'real-time values (data_mode R)',
    'A': 'real-time adjusted values (data_mode A)',
    'D': 'delayed-mode adjusted values (data_mode D)',
}


def _abs_days(time_str, when):
    """``|days|`` between an ERDDAP ISO time string and ``when`` (a
    ``datetime64[D]``). Returns ``inf`` (maximally unattractive in the cost,
    so a dated profile always beats an undated one) when the time is missing
    or unparseable — ERDDAP times are well-formed, this is just a guard.
    """
    if not time_str:
        return float('inf')
    try:
        day = np.datetime64(str(time_str)[:10])
    except (ValueError, TypeError):
        return float('inf')
    return abs(float((day - when) / np.timedelta64(1, 'D')))


def _query_url(point, when, max_distance_km, max_days, base_url):
    lat, lon = point
    dlat = max_distance_km / 111.0          # ~111 km per degree of latitude
    dlon = dlat / max(np.cos(np.radians(lat)), 1e-3)
    la0, la1 = lat - dlat, lat + dlat
    lo_lo, lo_hi = normalize_lon(lon) - dlon, normalize_lon(lon) + dlon
    t0 = (when - np.timedelta64(max_days, 'D'))
    # Exclusive upper bound at the first instant AFTER day ``when + max_days``,
    # so the last tolerated day is included whole — symmetric with the lower
    # bound, and consistent with the cost function tolerating dt == max_days.
    t1 = (when + np.timedelta64(max_days + 1, 'D'))
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
        f"&time%3E={t0}T00:00:00Z&time%3C{t1}T00:00:00Z"
        f"&latitude%3E={la0:.4f}&latitude%3C={la1:.4f}"
        f"{lon_clause}"
    )


def _checked_tolerances(max_distance_km, max_days):
    """``(max_distance_km, max_days)`` validated: ``None`` or a positive
    finite distance, and a whole number of days ``>= 0``, else
    ``ConfigurationError``."""
    max_distance_km = checked_max_distance(max_distance_km, 'Argo')
    if isinstance(max_days, bool) or not isinstance(
            max_days, (int, np.integer)) or max_days < 0:
        raise ConfigurationError(
            f"Argo: max_days must be a whole number of days >= 0; got "
            f"{max_days!r}.",
            remediation="Pass e.g. max_days=15 (0 = the requested day only).")
    return max_distance_km, int(max_days)


def _no_profile_error(max_distance_km, max_days, lat, lon, date):
    """The ``DataFetchError`` for an empty neighbourhood."""
    return DataFetchError(
        f"No Argo profile within {max_distance_km:g} km / {max_days} days "
        f"of {lat:.3f}, {lon:.3f} on {parse_date(date)}.",
        remediation="Widen max_distance_km / max_days, or use "
                    "ssp_sources='woa23' (climatology).",
    )


@dataclass(frozen=True, eq=False)
class ArgoProfile(ExportRecord):
    """One Argo cast, as :func:`fetch_argo_profile` returns it: good-QC
    levels only, sorted by strictly increasing pressure.

    Attributes
    ----------
    platform, cycle, direction
        The float's WMO number, its cycle and the cast direction
        (``'A'`` ascending, ``'D'`` descending).
    lat, lon : float
        The cast's position (decimal degrees).
    distance_km : float
        Its distance from the requested point.
    time : str or None
        The cast's ISO time.
    data_mode : str
        ``'R'``, ``'A'`` or ``'D'`` (``'/'``-joined when levels mix modes).
    pressure_dbar, temperature, salinity : ndarray
        The measured columns, read-only; :meth:`to_dataframe` gives one row
        per level.
    provenance : DataProvenance
        The ``'argo'`` record: the cast's date and position, the request,
        and the values read (real-time or adjusted, with the mode) as
        ``product``.
    """

    platform: str
    cycle: int
    direction: str
    lat: float
    lon: float
    distance_km: float
    time: Optional[str]
    data_mode: str
    pressure_dbar: np.ndarray
    temperature: np.ndarray
    salinity: np.ndarray
    provenance: DataProvenance

    _REPR_FIELDS = ('platform', 'cycle', 'lat', 'lon', 'time', 'pressure_dbar',
                    'temperature', 'salinity', 'provenance')
    _REPR_UNITS = {'lat': '°', 'lon': '°'}

    _ARRAY_FIELDS = ('pressure_dbar', 'temperature', 'salinity')
    _TABLE_FIELDS = ('pressure_dbar', 'temperature', 'salinity')


def fetch_argo_profile(
    point: Coordinate, *, date,
    max_distance_km: Optional[float] = None,
    max_days: int = DEFAULT_MAX_DAYS,
    base_url: str = ARGO_ERDDAP_URL,
    timeout: float = 60.0,
    verbose: Union[bool, str] = False,
) -> ArgoProfile:
    """Nearest Argo T/S profile to ``(lat, lon)`` and ``date``.

    Among the good-QC profiles within ``max_distance_km`` (or
    :data:`ARGO_SEARCH_RADIUS_KM` without one) and ``max_days`` (a
    cast is kept only where its ``position_qc`` is good or probably good), the
    one nearest in **combined space-time** is returned — each axis normalised by
    its own tolerance and added in quadrature, so a slightly farther but fresher
    float is preferred over a marginally closer but staler one.

    Each level is read according to the cast's ``data_mode``: in delayed mode
    (``'D'``) and real-time-adjusted mode (``'A'``) the ``*_adjusted`` pressure,
    temperature and salinity with their ``*_adjusted_qc`` flags; in real-time
    mode (``'R'``) the raw values with their raw flags. A level is kept when
    its flags are good or probably good and its values are finite, so a
    salinity the delayed-mode operator rejected is dropped even where its raw
    flag passed.

    Returns an :class:`ArgoProfile` (arrays sorted
    by strictly increasing pressure, good-QC levels only; a negative surface
    pressure is floored at 0 dbar and an exactly repeated level keeps its first
    sample). ``data_mode`` is the cast's mode (``'R'``, ``'A'`` or ``'D'``).
    ``provenance`` is the ``'argo'`` :class:`~uacpy.data.DataProvenance`: the
    cast's date and position as ``data_date``/``data_point``, the request as
    ``requested_date``/``requested_point``, and the values read (real-time or
    adjusted, with the mode) as ``product``. Raises
    ``DataFetchError`` when no float profile is within ``max_distance_km`` /
    ``max_days``. A cast farther than :data:`ARGO_OFFSET_WARN_KM` is returned
    with a ``ProvenanceWarning`` (the offset rule). ``max_distance_km`` must be
    ``None`` or positive; ``max_days`` is a
    whole number of days ``>= 0`` (``0`` = the requested day only, ranked on
    distance alone).

    Parameters
    ----------
    point : (lat, lon)
        Site coordinates in decimal degrees.
    date : str or datetime.date
        Calendar date of interest.
    max_distance_km : float, optional
        Farthest cast accepted (km), > 0; ``None`` (default) searches
        :data:`ARGO_SEARCH_RADIUS_KM` and warns past
        :data:`ARGO_OFFSET_WARN_KM`.
    max_days : int, optional
        Farthest cast accepted in time (whole days, >= 0). Default 15.
    base_url : str, optional
        The ERDDAP ``ArgoFloats`` table address. Default Ifremer's.
    timeout : float, optional
        Per-request network timeout in seconds.
    verbose : bool or str, optional
        Logging gate passed through to ``log_message``.
    """
    lat, lon = as_coordinate(point)
    limit, max_days = _checked_tolerances(max_distance_km, max_days)
    max_distance_km = ARGO_SEARCH_RADIUS_KM if limit is None else limit
    when = np.datetime64(parse_date(date), 'D')
    try:
        body = http_get(_query_url((lat, lon), when, max_distance_km, max_days,
                                   base_url), timeout=timeout, verbose=verbose,
                        source='argo')
    except DataFetchError as exc:
        # tabledap answers a query that matches no rows with HTTP 404
        # ("Your query produced no matching results"), so a 404 is the empty
        # result, not a missing service.
        if not (isinstance(exc.__cause__, urllib.error.HTTPError)
                and exc.__cause__.code == 404):
            raise
        raise _no_profile_error(max_distance_km, max_days, lat, lon,
                                date) from exc
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
    # A level is read from the triplet its ``data_mode`` names, and kept only
    # where that triplet's own QC flags and the cast's position flag are good
    # and all three values are finite. In A/D mode a level whose adjusted value
    # is missing is dropped, never answered with the raw value.
    profiles = {}      # (platform, cycle, direction) -> dict(lat, lon, lev)
    for r in rows[2:]:
        if len(r) < len(_COLUMNS):
            continue
        plat, cyc, dirn, _t, rlat, rlon, pres, temp, psal, tqc, sqc, pqc, \
            posqc, mode, pres_a, temp_a, psal_a, pqc_a, tqc_a, sqc_a = \
            r[:len(_COLUMNS)]
        mode = mode.strip()
        if mode == 'R':
            triplet, flags = (pres, temp, psal), (tqc, sqc, pqc, posqc)
        elif mode in ('A', 'D'):
            triplet = (pres_a, temp_a, psal_a)
            flags = (tqc_a, sqc_a, pqc_a, posqc)
        else:
            continue
        if any(qc not in _GOOD_QC for qc in flags):
            continue
        try:
            vals = (float(rlat), float(rlon)) + tuple(float(v)
                                                      for v in triplet)
        except ValueError:
            continue
        if not all(math.isfinite(v) for v in vals[2:]):
            continue
        prof = profiles.setdefault((plat, cyc, dirn),
                                   {'lat': vals[0], 'lon': vals[1],
                                    'time': _t, 'modes': set(), 'lev': []})
        prof['modes'].add(mode)
        prof['lev'].append(vals[2:])

    if not profiles:
        raise _no_profile_error(max_distance_km, max_days, lat, lon, date)
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
        bound = (f"max_distance_km={limit:g}" if limit is not None
                 else f"the {ARGO_SEARCH_RADIUS_KM:g} km search radius")
        raise DataFetchError(
            f"Nearest Argo profile is {nearest:.0f} km away (> {bound}).",
            remediation="Widen max_distance_km, or use ssp_sources='woa23'.",
        )

    def _spacetime_cost(item):
        _key, p, d_km = item
        dt_days = _abs_days(p.get('time'), when)
        # max_days == 0 admits the requested day only, so every dated
        # candidate has dt == 0 and the ranking falls to distance alone.
        dt_term = (dt_days / max_days if max_days
                   else (0.0 if dt_days == 0.0 else float('inf')))
        return (d_km / max_distance_km) ** 2 + dt_term ** 2

    (plat, cyc, dirn), prof, dist = min(within, key=_spacetime_cost)
    lev = np.array(sorted(prof['lev']), dtype=float)        # sort by pressure
    # Real-time surface PRES may be slightly negative with QC 1 (Argo's
    # global-range test admits -5 dbar): floor it at the surface so the depth
    # conversion cannot place a node above 0 m, then keep the first sample of
    # each exactly repeated level so the depths stay strictly increasing.
    lev[:, 0] = np.maximum(lev[:, 0], 0.0)
    lev = lev[np.unique(lev[:, 0], return_index=True)[1]]
    time = prof.get('time')
    data_mode = '/'.join(sorted(prof['modes']))
    values_read = '; '.join(_DATA_MODE_VALUES[m] for m in sorted(prof['modes']))
    prov = checked_offset(DataProvenance(
        source=SOURCES['argo'],
        data_date=(time[:10] if time else None),     # YYYY-MM-DD
        data_point=(float(prof['lat']), float(prof['lon'])),
        requested_point=(lat, lon),
        requested_date=str(parse_date(date)),
        product=values_read, point_kind='cast'),
        who='fetch_argo_profile', warn_km=ARGO_OFFSET_WARN_KM,
        max_distance_km=limit)
    return ArgoProfile(
        platform=plat, cycle=cyc, direction=dirn,
        lat=prof['lat'], lon=prof['lon'],
        distance_km=float(dist), time=time, data_mode=data_mode,
        pressure_dbar=lev[:, 0], temperature=lev[:, 1],
        salinity=lev[:, 2],
        provenance=checked_date_offset(prov, who='fetch_argo_profile'))


def fetch_ssp_argo(
    point: Coordinate, *, date,
    formula: str = DEFAULT_SOUND_SPEED_FORMULA,
    max_distance_km: Optional[float] = None,
    max_days: int = DEFAULT_MAX_DAYS,
    base_url: str = ARGO_ERDDAP_URL,
    timeout: float = 60.0,
    verbose: Union[bool, str] = False,
) -> SoundSpeedProfile:
    """Real in-situ sound-speed profile from the nearest Argo float.

    Finds the nearest good-QC Argo T/S profile (:func:`fetch_argo_profile`) and
    converts it with ``formula`` (``'teos10'``, the default, or ``'unesco'``
    / ``'delgrosso'`` / ``'mackenzie'`` — the same table
    :func:`uacpy.data.sound_speed.fetch_ssp` reads).
    Raises
    ``DataFetchError`` when no profile is close enough.

    Parameters
    ----------
    point : (lat, lon)
        Site coordinates in decimal degrees.
    date : str or datetime.date
        Calendar date of interest.
    formula : {'teos10', 'unesco', 'delgrosso', 'mackenzie'}, optional
        Sound-speed equation. Default ``'teos10'``.
    max_distance_km : float, optional
        Farthest cast accepted (km), > 0; ``None`` (default) as in
        :func:`fetch_argo_profile`.
    max_days : int, optional
        Farthest cast accepted in time (whole days, >= 0). Default 15.
    base_url : str, optional
        The ERDDAP ``ArgoFloats`` table address. Default Ifremer's.
    timeout : float, optional
        Per-request network timeout in seconds.
    verbose : bool or str, optional
        Logging gate passed through to ``log_message``.
    """
    formula = canonical_formula(formula, 'fetch_ssp_argo')
    prof = fetch_argo_profile(point, date=date, max_distance_km=max_distance_km,
                              max_days=max_days, base_url=base_url,
                              timeout=timeout, verbose=verbose)
    depths = pressure_dbar_to_depth(prof.pressure_dbar, prof.lat)
    # A float measures pressure: the pressure equations take it as measured,
    # and Mackenzie (stated in depth) takes the depths at the float's own
    # latitude.
    if formula == 'mackenzie':
        c = np.asarray(sound_speed_mackenzie(temperature=prof.temperature,
                                             salinity=prof.salinity,
                                             depth=depths), dtype=float)
    else:
        speed_fn = SOUND_SPEED_FORMULAS[formula]
        c = np.array([speed_fn(t, s, p) for t, s, p in
                      zip(prof.temperature, prof.salinity, prof.pressure_dbar)])
    log_message(
        'sound_speed', f"Argo SSP from float {prof.platform} cycle "
        f"{prof.cycle}{prof.direction} ({prof.distance_km:.0f} km "
        f"away): {depths.size} levels, c=[{c.min():.1f}, {c.max():.1f}] m/s",
        verbose=verbose)
    return SoundSpeedProfile(depths=depths, sound_speed=c, kind='measured',
                             data_sources=(prof.provenance,),
                             formula=formula)
