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
import warnings
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
from uacpy._log import log_message

__all__ = [
    'fetch_ssp_operational',
    'fetch_ssp_transect_operational',
    'fetch_ts_profile_operational',
]

# Global ocean physics, daily means, 1/12° — multi-year reanalysis.
DEFAULT_DATASET_ID = 'cmems_mod_glo_phy_my_0.083deg_P1D-m'
TEMPERATURE_VAR = 'thetao'
SALINITY_VAR = 'so'


def fetch_ssp_operational(
    point: Coordinate,
    *,
    date: Union[str, _dt.date],
    formula: str = 'unesco',
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
        service fails, or the location has no profile.
    """
    if formula not in _FORMULAS:
        raise ConfigurationError(
            f"fetch_ssp_operational: unknown formula={formula!r}.",
            remediation=f"Use one of {sorted(_FORMULAS)}.",
        )
    lat, lon = as_coordinate(point)
    depths, temp, sal = fetch_ts_profile_operational(
        point, date=date, dataset_id=dataset_id, timeout=timeout, verbose=verbose,
    )
    pressure = depth_to_pressure_dbar(depths, lat)
    speed_fn = _FORMULAS[formula]
    c = np.array([speed_fn(t, s, p) for t, s, p in zip(temp, sal, pressure)])
    log_message(
        'copernicus', f"operational SSP at {lat:.3f}, {lon:.3f} ({date}): "
        f"{depths.size} levels, c=[{c.min():.1f}, {c.max():.1f}] m/s",
        verbose=verbose,
    )
    return SoundSpeedProfile(depths=depths, data=c, shape='measured')


def fetch_ssp_transect_operational(
    start: Coordinate,
    end: Coordinate,
    *,
    date: Union[str, _dt.date],
    n_points: int = 6,
    formula: str = 'unesco',
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
        depths, temp, sal = _extract_ts(ds, la, lo, when)
        if depths.size == 0:
            raise DataFetchError(
                f"No Copernicus profile at {la:.3f}, {lo:.3f} on {when}.",
                remediation="Keep the transect within the dataset's wet domain.",
            )
        pressure = depth_to_pressure_dbar(depths, la)
        c = np.array([speed_fn(t, s, p) for t, s, p in zip(temp, sal, pressure)])
        columns.append(SoundSpeedProfile(depths=depths, data=c, shape='measured'))

    log_message(
        'copernicus', f"operational range-dependent SSP: {n_points} columns "
        f"over {ranges_m[-1] / 1000:.1f} km", verbose=verbose,
    )
    return assemble_range_dependent(columns, ranges_m)


def fetch_ts_profile_operational(
    point: Coordinate,
    *,
    date: Union[str, _dt.date],
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

    depths, temp, sal = _extract_ts(ds, lat, lon, when)
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull a single seafloor-truncated water column from an xarray dataset."""
    depth = np.asarray(ds['depth'].values, dtype=float).reshape(-1)
    sel = {'latitude': lat, 'longitude': normalize_lon(lon)}
    if when is not None:
        sel['time'] = when
    t_da = ds[temp_var].sel(method='nearest', **sel)
    if when is not None and 'time' in getattr(t_da, 'coords', {}):
        # ``method='nearest'`` snaps silently to the dataset edge for an
        # out-of-range date; warn so the substitution isn't invisible.
        actual = np.datetime64(np.asarray(t_da['time'].values).reshape(-1)[0], 'D')
        gap = abs((actual - np.datetime64(when, 'D')) / np.timedelta64(1, 'D'))
        if gap > 31:
            warnings.warn(
                f"Copernicus: requested {when} but the nearest available time "
                f"is {actual} ({gap:.0f} days away) — the date is outside the "
                "dataset's range; using the edge value. Pass a date within "
                "range or a forecast dataset_id.",
                UserWarning, stacklevel=2)
    t = np.asarray(t_da.values, float).reshape(-1)
    s = np.asarray(ds[sal_var].sel(method='nearest', **sel).values, float).reshape(-1)

    n = min(depth.size, t.size, s.size)
    depth, t, s = depth[:n], t[:n], s[:n]
    valid = np.isfinite(t) & np.isfinite(s)
    cut = valid.size if valid.all() else int(np.argmax(~valid))
    return depth[:cut], t[:cut], s[:cut]


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
