"""Copernicus Marine operational SSP — GPS + date → ``SoundSpeedProfile``.

Phase 3 of the on-demand external-data layer. Where WOA23
(:mod:`uacpy.data.sound_speed`) gives a reproducible *climatology*, the
Copernicus Marine Service gives *date-specific* conditions: reanalysis for
past dates, analysis/forecast for recent/near-future ones. Same output
contract — a :class:`~uacpy.core.environment.SoundSpeedProfile`.

The ``copernicusmarine`` toolbox ships with uacpy (a core dependency); this
source only additionally needs a free Copernicus Marine account:

    copernicusmarine login        # one-time, stores credentials

Temperature comes back as potential temperature (``thetao``); it is used as a
close proxy for in-situ temperature in the sound-speed equation (sub-m/s
effect over typical profiles).
"""

import datetime as _dt
from typing import Optional, Tuple, Union

import numpy as np

from uacpy.core.environment import SoundSpeedProfile
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data._geo import (
    Coordinate, as_coordinate, normalize_lon, depth_to_pressure_dbar,
)
from uacpy.data._time import parse_date
from uacpy.data.sound_speed import (
    _FORMULAS, assemble_range_dependent,
)
from uacpy.data.sources import SOURCES, DataProvenance
from uacpy._log import log_message

__all__ = [
    'fetch_ssp_operational',
    'fetch_ssp_transect_operational',
    'fetch_ts_profile_operational',
    'fetch_waves_operational',
    'fetch_ph_operational',
]

# Global ocean physics, daily means, 1/12° — multi-year reanalysis.
DEFAULT_DATASET_ID = 'cmems_mod_glo_phy_my_0.083deg_P1D-m'
TEMPERATURE_VAR = 'thetao'
SALINITY_VAR = 'so'
# Global wave reanalysis (WAVERYS), 1/5°, 3-hourly, 1980→present.
DEFAULT_WAVE_DATASET_ID = 'cmems_mod_glo_wav_my_0.2deg_PT3H-i'
WAVE_HS_VAR = 'VHM0'                 # spectral significant wave height (m)
WAVE_TP_VAR = 'VTPK'                 # wave peak period (s)
# Global biogeochemistry reanalysis, 0.25°, monthly, 1993→present.
DEFAULT_BGC_DATASET_ID = 'cmems_mod_glo_bgc_my_0.25deg_P1M-m'
BGC_PH_VAR = 'ph'                    # sea-water pH (total scale)
# Max days the nearest available time step may sit from the requested date
# before we treat it as out-of-coverage and raise (shared tolerance contract
# with the other dated SSP sources — cf. argo.DEFAULT_MAX_DAYS=15). Looser than
# Argo's by design: this is daily-mean *model* output (smooth, persistent — and
# in-coverage the nearest step is sub-day, so this is really a coverage-edge
# guard), whereas an Argo profile is one real in-situ snapshot that the ocean
# decorrelates from within a couple of weeks. The default tracks how slowly the
# field varies: climatology (month) > model (31 d) > in-situ obs (15 d).
DEFAULT_MAX_DAYS = 31


def fetch_ssp_operational(
    point: Coordinate,
    *,
    date: Union[str, _dt.date],
    formula: str = 'unesco',
    max_days: int = DEFAULT_MAX_DAYS,
    dataset_id: str = DEFAULT_DATASET_ID,
    timeout: float = 120.0,
    verbose: Union[bool, str] = False,
) -> SoundSpeedProfile:
    """Date-specific sound-speed profile from Copernicus Marine.

    Parameters
    ----------
    point : (lat, lon)
        Latitude/longitude in decimal degrees (WGS84).
    date : str or datetime.date
        Calendar date of interest. The nearest available time step is used.
    formula : {'unesco', 'delgrosso'}, optional
        Sound-speed equation. Default ``'unesco'``.
    max_days : int, optional
        Maximum days the nearest available time step may differ from ``date``
        before raising ``DataFetchError`` (the date is outside the dataset's
        coverage). Default 31. The shared tolerance guard mirrors
        ``argo``'s — the nearest match is never silently substituted.
    dataset_id : str, optional
        Copernicus Marine dataset. Default is the global physics reanalysis;
        for recent dates use an analysis/forecast id, e.g.
        ``'cmems_mod_glo_phy_anfc_0.083deg_P1D-m'``.
    timeout, verbose
        Network timeout (s) and logging gate.

    Returns
    -------
    SoundSpeedProfile

    Raises
    ------
    ConfigurationError
        Unknown ``formula``.
    DataFetchError
        ``copernicusmarine`` is not installed / not authenticated, the
        service fails, the location has no profile, or the nearest available
        time step is more than ``max_days`` from ``date``.
    """
    if formula not in _FORMULAS:
        raise ConfigurationError(
            f"fetch_ssp_operational: unknown formula={formula!r}.",
            remediation=f"Use one of {sorted(_FORMULAS)}.",
        )
    lat, lon = as_coordinate(point)
    depths, temp, sal = fetch_ts_profile_operational(
        point, date=date, max_days=max_days, dataset_id=dataset_id,
        timeout=timeout, verbose=verbose,
    )
    pressure = depth_to_pressure_dbar(depths, lat)
    speed_fn = _FORMULAS[formula]
    c = np.array([speed_fn(t, s, p) for t, s, p in zip(temp, sal, pressure)])
    log_message(
        'copernicus', f"operational SSP at {lat:.3f}, {lon:.3f} ({date}): "
        f"{depths.size} levels, c=[{c.min():.1f}, {c.max():.1f}] m/s",
        verbose=verbose,
    )
    prov = DataProvenance(
        source=SOURCES['copernicus'],
        requested_point=(lat, lon),
        requested_date=str(parse_date(date)),
    )
    return SoundSpeedProfile(depths=depths, data=c, shape='measured',
                             data_sources=(prov,))


def fetch_ssp_transect_operational(
    start: Coordinate,
    end: Coordinate,
    *,
    date: Union[str, _dt.date],
    n_points: int = 6,
    formula: str = 'unesco',
    max_days: int = DEFAULT_MAX_DAYS,
    dataset_id: str = DEFAULT_DATASET_ID,
    timeout: float = 120.0,
    verbose: Union[bool, str] = False,
) -> SoundSpeedProfile:
    """Range-dependent operational SSP along ``start`` → ``end`` (Copernicus).

    The Copernicus counterpart of :func:`uacpy.data.fetch_ssp_transect`: opens
    the dataset once, samples ``n_points`` columns along the great-circle path,
    and assembles a 2-D range-dependent
    :class:`~uacpy.core.environment.SoundSpeedProfile`. See
    :func:`fetch_ssp_operational` for parameters/exceptions.
    """
    from uacpy.data._geo import geodesic_waypoints

    if formula not in _FORMULAS:
        raise ConfigurationError(
            f"fetch_ssp_transect_operational: unknown formula={formula!r}.",
            remediation=f"Use one of {sorted(_FORMULAS)}.",
        )
    when = parse_date(date).isoformat()
    marine = _import_copernicusmarine()
    try:
        ds = marine.open_dataset(dataset_id=dataset_id)
    except Exception as exc:
        raise DataFetchError(
            f"Copernicus Marine open_dataset failed: {exc}",
            remediation="Run `copernicusmarine login` and check the dataset_id.",
        ) from exc

    lats, lons, ranges_m = geodesic_waypoints(start, end, n_points)
    speed_fn = _FORMULAS[formula]
    columns = []
    for la, lo in zip(lats, lons):
        depths, temp, sal = _extract_ts(ds, la, lo, when, max_days=max_days)
        if depths.size == 0:
            raise DataFetchError(
                f"No Copernicus profile at {la:.3f}, {lo:.3f} on {when}.",
                remediation="Keep the transect within the dataset's wet domain.",
            )
        pressure = depth_to_pressure_dbar(depths, la)
        c = np.array([speed_fn(t, s, p) for t, s, p in zip(temp, sal, pressure)])
        prov = DataProvenance(source=SOURCES['copernicus'],
                              requested_point=(float(la), float(lo)),
                              requested_date=when)
        columns.append(SoundSpeedProfile(depths=depths, data=c, shape='measured',
                                         data_sources=(prov,)))

    log_message(
        'copernicus', f"operational range-dependent SSP: {n_points} columns "
        f"over {ranges_m[-1] / 1000:.1f} km", verbose=verbose,
    )
    return assemble_range_dependent(columns, ranges_m)


def fetch_ts_profile_operational(
    point: Coordinate,
    *,
    date: Union[str, _dt.date],
    max_days: int = DEFAULT_MAX_DAYS,
    dataset_id: str = DEFAULT_DATASET_ID,
    timeout: float = 120.0,
    verbose: Union[bool, str] = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw ``(depths_m, temperature_degC, salinity_psu)`` from Copernicus.

    Truncated at the seafloor (first non-finite level). See
    :func:`fetch_ssp_operational` for parameters and exceptions.
    """
    lat, lon = as_coordinate(point)
    when = parse_date(date).isoformat()
    marine = _import_copernicusmarine()

    log_message('copernicus', f"opening {dataset_id} for {lat:.3f}, {lon:.3f}",
                verbose=verbose, level='debug')
    try:
        ds = marine.open_dataset(dataset_id=dataset_id)
    except Exception as exc:  # toolbox raises a variety of auth/network errors
        raise DataFetchError(
            f"Copernicus Marine open_dataset failed: {exc}",
            remediation="Run `copernicusmarine login` (free account) and check "
                        "the dataset_id and network connectivity.",
        ) from exc

    depths, temp, sal = _extract_ts(ds, lat, lon, when, max_days=max_days)
    if depths.size == 0:
        raise DataFetchError(
            f"No Copernicus profile at {lat:.3f}, {lon:.3f} on {when} "
            "(on land or outside the dataset domain).",
            remediation="Pick an ocean location/date within the dataset.",
        )
    return depths, temp, sal


def _extract_ts(
    ds, lat: float, lon: float, when: Optional[str],
    *, temp_var: str = TEMPERATURE_VAR, sal_var: str = SALINITY_VAR,
    max_days: int = DEFAULT_MAX_DAYS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull a single seafloor-truncated water column from an xarray dataset."""
    depth = np.asarray(ds['depth'].values, dtype=float).reshape(-1)
    sel = {'latitude': lat, 'longitude': normalize_lon(lon)}
    if when is not None:
        sel['time'] = when
    t_da = ds[temp_var].sel(method='nearest', **sel)
    if when is not None and 'time' in getattr(t_da, 'coords', {}):
        # ``method='nearest'`` snaps silently to the dataset edge for an
        # out-of-range date; raise rather than substitute an edge value so the
        # tolerance is honoured the same way the other dated sources honour it.
        actual = np.datetime64(np.asarray(t_da['time'].values).reshape(-1)[0], 'D')
        gap = abs((actual - np.datetime64(when, 'D')) / np.timedelta64(1, 'D'))
        if gap > max_days:
            raise DataFetchError(
                f"Copernicus: nearest available time is {actual} "
                f"({gap:.0f} days from requested {when}, > max_days={max_days}) "
                "— the date is outside the dataset's range.",
                remediation="Pass a date within range, a forecast dataset_id, "
                            "raise max_days, or use ssp_sources='woa23'.",
            )
    t = np.asarray(t_da.values, float).reshape(-1)
    s = np.asarray(ds[sal_var].sel(method='nearest', **sel).values, float).reshape(-1)

    n = min(depth.size, t.size, s.size)
    depth, t, s = depth[:n], t[:n], s[:n]
    valid = np.isfinite(t) & np.isfinite(s)
    cut = valid.size if valid.all() else int(np.argmax(~valid))
    return depth[:cut], t[:cut], s[:cut]


def fetch_waves_operational(
    point: Coordinate,
    *,
    date: Union[str, _dt.date],
    max_days: int = DEFAULT_MAX_DAYS,
    dataset_id: str = DEFAULT_WAVE_DATASET_ID,
    timeout: float = 120.0,
    verbose: Union[bool, str] = False,
) -> dict:
    """Significant wave height (m) and peak period (s) from Copernicus WAVERYS.

    Returns ``{'hs', 'tp'}`` at the ``(lat, lon)`` cell nearest ``date`` (full
    reanalysis history, 1980→present). ``tp`` is ``None`` when the dataset does
    not expose a peak-period variable. Raises ``DataFetchError`` on land / out of
    coverage or when the nearest time step is more than ``max_days`` away.
    """
    lat, lon = as_coordinate(point)
    when = parse_date(date).isoformat()
    marine = _import_copernicusmarine()
    log_message('waves', f"opening {dataset_id} for {lat:.3f}, {lon:.3f}",
                verbose=verbose, level='debug')
    try:
        ds = marine.open_dataset(dataset_id=dataset_id)
    except Exception as exc:  # toolbox raises a variety of auth/network errors
        raise DataFetchError(
            f"Copernicus Marine open_dataset failed: {exc}",
            remediation="Run `copernicusmarine login` (free account) and check "
                        "the dataset_id and network connectivity.",
        ) from exc

    sel = {'latitude': lat, 'longitude': normalize_lon(lon), 'time': when}
    hs_da = ds[WAVE_HS_VAR].sel(method='nearest', **sel)
    if 'time' in getattr(hs_da, 'coords', {}):
        actual = np.datetime64(np.asarray(hs_da['time'].values).reshape(-1)[0], 'D')
        gap = abs((actual - np.datetime64(when, 'D')) / np.timedelta64(1, 'D'))
        if gap > max_days:
            raise DataFetchError(
                f"Copernicus waves: nearest time is {actual} ({gap:.0f} days "
                f"from {when}, > max_days={max_days}).",
                remediation="Pass a date within range, raise max_days, or use "
                            "the WaveWatch III source.",
            )
    hs = float(np.asarray(hs_da.values, float).reshape(-1)[0])
    if not np.isfinite(hs):
        raise DataFetchError(
            f"No Copernicus wave height at {lat:.3f}, {lon:.3f} on {when} "
            "(on land or outside the dataset domain).",
            remediation="Pick an ocean location/date within the dataset.",
        )
    tp = None
    if WAVE_TP_VAR in getattr(ds, 'variables', {}) or WAVE_TP_VAR in ds:
        tp_val = float(np.asarray(
            ds[WAVE_TP_VAR].sel(method='nearest', **sel).values, float).reshape(-1)[0])
        tp = tp_val if np.isfinite(tp_val) else None
    return {'hs': hs, 'tp': tp}


def fetch_ph_operational(
    point: Coordinate,
    *,
    date: Union[str, _dt.date],
    reference_depth: Optional[float] = None,
    max_days: int = DEFAULT_MAX_DAYS,
    dataset_id: str = DEFAULT_BGC_DATASET_ID,
    timeout: float = 120.0,
    verbose: Union[bool, str] = False,
) -> float:
    """Date-specific seawater pH from Copernicus Marine biogeochemistry.

    The operational counterpart of the cached GLODAP climatology
    (:func:`uacpy.data.fetch_ph`): returns the pH at ``reference_depth`` (m,
    nearest level), or the shallowest finite level when ``None`` — matching the
    nominal-row convention of :func:`uacpy.data.build_francois_garrison`.

    Raises ``DataFetchError`` when ``copernicusmarine`` is unavailable, the
    service fails, the location has no column, or the nearest time step is more
    than ``max_days`` from ``date``.
    """
    lat, lon = as_coordinate(point)
    when = parse_date(date).isoformat()
    marine = _import_copernicusmarine()
    log_message('copernicus', f"opening {dataset_id} (pH) for {lat:.3f}, "
                f"{lon:.3f}", verbose=verbose, level='debug')
    try:
        ds = marine.open_dataset(dataset_id=dataset_id)
    except Exception as exc:  # toolbox raises a variety of auth/network errors
        raise DataFetchError(
            f"Copernicus Marine open_dataset failed: {exc}",
            remediation="Run `copernicusmarine login` (free account) and check "
                        "the dataset_id and network connectivity.",
        ) from exc

    sel = {'latitude': lat, 'longitude': normalize_lon(lon), 'time': when}
    ph_da = ds[BGC_PH_VAR].sel(method='nearest', **sel)
    if 'time' in getattr(ph_da, 'coords', {}):
        actual = np.datetime64(np.asarray(ph_da['time'].values).reshape(-1)[0], 'D')
        gap = abs((actual - np.datetime64(when, 'D')) / np.timedelta64(1, 'D'))
        if gap > max_days:
            raise DataFetchError(
                f"Copernicus pH: nearest time is {actual} ({gap:.0f} days from "
                f"{when}, > max_days={max_days}) — outside the dataset's range.",
                remediation="Pass a date within range, raise max_days, or rely "
                            "on the GLODAP climatology / model default.",
            )
    depth = np.asarray(ds['depth'].values, float).reshape(-1)
    ph = np.asarray(ph_da.values, float).reshape(-1)
    n = min(depth.size, ph.size)
    depth, ph = depth[:n], ph[:n]
    finite = np.isfinite(ph)
    if not finite.any():
        raise DataFetchError(
            f"No Copernicus pH at {lat:.3f}, {lon:.3f} on {when} "
            "(on land or outside the dataset domain).",
            remediation="Pick an ocean location/date within the dataset.",
        )
    depth, ph = depth[finite], ph[finite]
    if reference_depth is None:
        return float(ph[0])
    i = int(np.argmin(np.abs(depth - float(reference_depth))))
    return float(ph[i])


def _import_copernicusmarine():
    try:
        import copernicusmarine
        return copernicusmarine
    except ImportError as exc:
        raise DataFetchError(
            "The 'copernicusmarine' toolbox is required for operational SSP "
            "but is not installed.",
            remediation="`copernicusmarine` ships with uacpy; reinstall with "
                        "`pip install -e .`, then run `copernicusmarine login` "
                        "(free Copernicus account).",
        ) from exc
