"""WOA23 sound-speed fetch — GPS + date → ``SoundSpeedProfile``.

Phase 2 of the on-demand external-data layer: turn a location (and,
optionally, a calendar date/month) into a depth-vs-sound-speed profile
ready for ``Environment(ssp=...)``.

Temperature and salinity come from the NOAA/NCEI **World Ocean Atlas 2023**
objectively-analyzed climatology (``t_an`` / ``s_an``), read remotely from
the NCEI THREDDS OPeNDAP server. We request a single ``(lat, lon)`` water
column via the DAP ``.ascii`` response — stdlib text, no NetCDF dependency.
Sound speed is then computed from T, S and pressure with the UNESCO
(Chen-Millero) or Del Grosso equation already in :mod:`uacpy.core.acoustics`.

Time handling
-------------
WOA23 is a *climatology*, not a forecast: ``date``/``month`` select a monthly
or seasonal climatological mean, never a specific year's conditions. Monthly
and seasonal fields only resolve the upper 1500 m; below that the annual mean
is spliced on, giving a full-depth, season-aware profile. For true
date-specific conditions use the (planned) Copernicus Marine source.
"""

import datetime as _dt
import re
from typing import List, Optional, Tuple, Union

import numpy as np

from uacpy.core.acoustics import soundspeed_unesco, soundspeed_delgrosso
from uacpy.core.environment import SoundSpeedProfile
from uacpy.data._geo import (
    Coordinate, as_coordinate, normalize_lon, depth_to_pressure_dbar,
    geodesic_waypoints, run_representative_indices, DEFAULT_MAX_TRANSECT_POINTS,
)
from uacpy.data._time import parse_date
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data._http import http_get
from uacpy._log import log_message

__all__ = ['fetch_ssp', 'fetch_ssp_transect', 'fetch_ts_profile']

DEFAULT_BASE_URL = 'https://www.ncei.noaa.gov/thredds-ocean/dodsC/woa23/DATA'
DEFAULT_DECADE = 'decav'              # 1955-2022 average of all decades
WOA_FILL_THRESHOLD = 1e30             # _FillValue is 9.96921e36

# Regular grids: (n_lat, n_lon, file_resolution_code, first_center, step).
_GRIDS = {
    '1.00': (180, 360, '01', -89.5, 1.0),
    '0.25': (720, 1440, '04', -89.875, 0.25),
}

_FORMULAS = {
    'unesco': soundspeed_unesco,
    'delgrosso': soundspeed_delgrosso,
}


def fetch_ssp(
    point: Coordinate,
    *,
    date: Union[str, _dt.date, None] = None,
    month: Optional[int] = None,
    formula: str = 'unesco',
    resolution: str = '1.00',
    source: str = 'opendap',
    decade: str = DEFAULT_DECADE,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 60.0,
    verbose: Union[bool, str] = False,
) -> SoundSpeedProfile:
    """Sound-speed profile at a ``(lat, lon)`` point from World Ocean Atlas 2023.

    Parameters
    ----------
    point : (lat, lon)
        Latitude/longitude in decimal degrees (WGS84). ``lon`` may be given
        in either ``[-180, 180]`` or ``[0, 360]``.
    date : str or datetime.date, optional
        Calendar date; only its month is used to pick the climatological
        month. Mutually exclusive with ``month``.
    month : int, optional
        Climatological month ``1``–``12``. ``None`` (and no ``date``) selects
        the annual mean.
    formula : {'unesco', 'delgrosso'}, optional
        Sound-speed equation. Default ``'unesco'`` (Chen-Millero).
    resolution : {'1.00', '0.25'}, optional
        WOA grid spacing in degrees. Default ``'1.00'``.
    source : {'opendap', 'local'}, optional
        ``'opendap'`` (default) streams from the NCEI THREDDS server; ``'local'``
        reads the install-time WOA23 grids offline (``install.sh --data woa23``).
    decade : str, optional
        WOA averaging period directory (default ``'decav'``).
    base_url, timeout, verbose
        THREDDS root, network timeout, logging gate.

    Returns
    -------
    SoundSpeedProfile
        1-D profile (``depths`` m, ``data`` m/s), ready for
        ``Environment(ssp=...)``.

    Raises
    ------
    ConfigurationError
        Bad ``formula``/``resolution``, or both ``date`` and ``month`` given.
    DataFetchError
        Service failure, or the location is on land / has no profile.
    """
    if formula not in _FORMULAS:
        raise ConfigurationError(
            f"fetch_ssp: unknown formula={formula!r}.",
            remediation=f"Use one of {sorted(_FORMULAS)}.",
        )
    lat, lon = as_coordinate(point)
    depths, temp, sal = fetch_ts_profile(
        point, date=date, month=month, resolution=resolution, source=source,
        decade=decade, base_url=base_url, timeout=timeout, verbose=verbose,
    )
    pressure = depth_to_pressure_dbar(depths, lat)
    speed_fn = _FORMULAS[formula]
    c = np.array([speed_fn(t, s, p) for t, s, p in zip(temp, sal, pressure)])
    log_message(
        'sound_speed', f"WOA23 SSP at {lat:.3f}, {lon:.3f}: {depths.size} "
        f"levels, c=[{c.min():.1f}, {c.max():.1f}] m/s", verbose=verbose,
    )
    return SoundSpeedProfile(depths=depths, data=c, shape='measured')


def ssp_transect_plan(
    start: Coordinate, end: Coordinate, *,
    n_points: Union[int, str] = 'auto',
    max_points: int = DEFAULT_MAX_TRANSECT_POINTS,
    resolution: str = '1.00',
) -> dict:
    """Resolve *where* a WOA23 transect would sample, without fetching.

    Returns ``{'n_points', 'lats', 'lons', 'ranges_m'}`` — the column
    coordinates the transect fetch would use. With ``n_points='auto'`` the
    plan reflects the **distinct WOA cells** the great-circle crosses (the
    grid cell is the sample identity, computed analytically — no network), so
    you can see how many independent columns are actually available before
    paying to fetch them. ``max_points`` caps the probe (and thus the result).
    """
    if resolution not in _GRIDS:
        raise ConfigurationError(
            f"ssp_transect_plan: unknown resolution={resolution!r}.",
            remediation=f"Use one of {sorted(_GRIDS)}.")
    if n_points == 'auto':
        probe_n = int(max_points)
    else:
        if int(n_points) < 2:
            raise ConfigurationError(
                f"ssp_transect_plan: n_points must be >= 2, got {n_points}.",
                remediation="Pass n_points>=2 or 'auto'.")
        probe_n = min(int(n_points), int(max_points))
    lats, lons, ranges_m = geodesic_waypoints(start, end, probe_n)
    if n_points == 'auto':
        # Identity = WOA grid cell (analytic, no fetch). Collapse runs that
        # fall in the same cell so duplicates are never fetched.
        keys = [_grid_index(la, lo, resolution)[:2]
                for la, lo in zip(lats, lons)]
        reps = run_representative_indices(keys)
    else:
        reps = list(range(probe_n))
    idx = np.asarray(reps, dtype=int)
    return {'n_points': int(idx.size), 'lats': lats[idx],
            'lons': lons[idx], 'ranges_m': ranges_m[idx]}


def fetch_ssp_transect(
    start: Coordinate,
    end: Coordinate,
    *,
    n_points: Union[int, str] = 'auto',
    max_points: int = DEFAULT_MAX_TRANSECT_POINTS,
    date: Union[str, _dt.date, None] = None,
    month: Optional[int] = None,
    formula: str = 'unesco',
    resolution: str = '1.00',
    source: str = 'opendap',
    decade: str = DEFAULT_DECADE,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 60.0,
    verbose: Union[bool, str] = False,
) -> SoundSpeedProfile:
    """Range-dependent sound-speed profile along ``start`` → ``end``.

    With ``n_points='auto'`` (default) the transect is sampled at the
    **distinct WOA23 cells** the great-circle crosses: the grid cell is the
    sample identity (computed analytically via ``_grid_index`` — no network),
    consecutive waypoints in the same cell collapse, and one column is fetched
    **per distinct cell** (no duplicate column is ever fetched). This matches
    WOA's native range resolution — neither over- nor under-sampling. Pass an
    integer to sample exactly that many evenly-spaced columns instead.

    ``max_points`` caps the number of waypoints probed *before* the reduction
    (the fetch budget); the result is never larger.

    Columns are placed on a common depth axis (the deepest sampled column);
    shallower columns hold their deepest value below their own seafloor
    (constant extrapolation, the usual SSP convention). Parameters otherwise
    mirror :func:`fetch_ssp`; ``start``/``end`` are ``(lat, lon)``.
    """
    plan = ssp_transect_plan(start, end, n_points=n_points,
                             max_points=max_points, resolution=resolution)
    lats, lons, ranges_m = plan['lats'], plan['lons'], plan['ranges_m']
    columns = [
        fetch_ssp((la, lo), date=date, month=month, formula=formula,
                  resolution=resolution, source=source, decade=decade,
                  base_url=base_url, timeout=timeout, verbose=verbose)
        for la, lo in zip(lats, lons)
    ]
    log_message(
        'sound_speed',
        f"WOA23 range-dependent SSP: {len(columns)} columns "
        f"({'auto' if n_points == 'auto' else n_points}) over "
        f"{ranges_m[-1] / 1000:.1f} km", verbose=verbose,
    )
    return assemble_range_dependent(columns, ranges_m)


def assemble_range_dependent(columns, ranges_m) -> SoundSpeedProfile:
    """Stack 1-D ``SoundSpeedProfile`` columns into a 2-D range-dependent one.

    The common depth axis is the deepest column's; shallower columns hold their
    deepest value below their own seafloor (``np.interp`` constant-edge fill).
    Shared by the WOA23 and Copernicus transect fetchers.
    """
    z = max(columns, key=lambda p: p.depths[-1]).depths
    data = np.column_stack([
        np.interp(z, col.depths, col.data[:, 0]) for col in columns
    ])
    return SoundSpeedProfile(depths=z, data=data, ranges=np.asarray(ranges_m),
                             shape='measured')


def fetch_ts_profile(
    point: Coordinate,
    *,
    date: Union[str, _dt.date, None] = None,
    month: Optional[int] = None,
    resolution: str = '1.00',
    source: str = 'opendap',
    decade: str = DEFAULT_DECADE,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 60.0,
    verbose: Union[bool, str] = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw WOA23 temperature/salinity column at a ``(lat, lon)`` point.

    Returns ``(depths_m, temperature_degC, salinity_psu)`` on the WOA
    standard depth levels, truncated at the seafloor. Useful on its own for
    building absorption models (Francois-Garrison needs T, S — and pH, which
    WOA does not carry).

    See :func:`fetch_ssp` for the parameters; raises identically.
    """
    if resolution not in _GRIDS:
        raise ConfigurationError(
            f"fetch_ts_profile: unknown resolution={resolution!r}.",
            remediation=f"Use one of {sorted(_GRIDS)}.",
        )
    _check_woa_source(source)
    lat, lon = as_coordinate(point)
    period = _resolve_period(date, month)
    lat_idx, lon_idx, lat_c, lon_c = _grid_index(lat, lon, resolution)

    depths, temp, sal = _get_column(
        source, period, lat_idx, lon_idx, resolution=resolution, decade=decade,
        base_url=base_url, timeout=timeout, verbose=verbose,
    )
    if depths.size == 0:
        raise DataFetchError(
            f"WOA23 has no water-column data at {lat_c:.3f}, {lon_c:.3f} "
            "(on land or outside the analyzed domain).",
            remediation="Pick an ocean location, or a coarser resolution.",
        )

    # Monthly/seasonal fields cap at 1500 m. If this column reached that cap
    # (deeper water exists below it), splice the annual mean on underneath; if
    # it stopped shallower it already hit the seafloor, so leave it be.
    if period != 0 and depths[-1] >= _MONTHLY_MAX_DEPTH:
        z_a, t_a, s_a = _get_column(
            source, 0, lat_idx, lon_idx, resolution=resolution, decade=decade,
            base_url=base_url, timeout=timeout, verbose=verbose,
        )
        below = z_a > depths[-1]
        depths = np.concatenate([depths, z_a[below]])
        temp = np.concatenate([temp, t_a[below]])
        sal = np.concatenate([sal, s_a[below]])

    return depths, temp, sal


_MONTHLY_MAX_DEPTH = 1500.0  # deepest level in WOA monthly/seasonal fields


def _resolve_period(date, month) -> int:
    """Map ``date``/``month`` to a WOA period code (0 annual, 1-12 monthly)."""
    if date is not None and month is not None:
        raise ConfigurationError(
            "fetch_ssp: pass either date= or month=, not both.",
        )
    if date is not None:
        month = parse_date(date).month
    if month is None:
        return 0
    if not 1 <= int(month) <= 12:
        raise ConfigurationError(
            f"fetch_ssp: month must be 1-12, got {month}.",
        )
    return int(month)


def _grid_index(lat, lon, resolution) -> Tuple[int, int, float, float]:
    """Nearest WOA grid-cell indices and snapped centre coordinates."""
    if not -90.0 <= lat <= 90.0:
        raise ConfigurationError(f"fetch_ssp: lat must be in [-90, 90], got {lat}.")
    lon = normalize_lon(lon)
    n_lat, n_lon, _code, first, step = _GRIDS[resolution]
    lat_idx = int(np.clip(round((lat - first) / step), 0, n_lat - 1))
    lon_idx = int(np.clip(round((lon - (-180.0 + step / 2)) / step), 0, n_lon - 1))
    lat_c = first + lat_idx * step
    lon_c = (-180.0 + step / 2) + lon_idx * step
    return lat_idx, lon_idx, lat_c, lon_c


def _fetch_column(
    period: int, lat_idx: int, lon_idx: int, *,
    resolution: str, decade: str, base_url: str, timeout: float,
    verbose: Union[bool, str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fetch one ``(depths, temperature, salinity)`` water column.

    The temperature and salinity files share an identical depth axis, so the
    axis length (which varies: ~102 levels annual, ~57 monthly) is read once
    and reused to request the exact hyperslab from each variable.
    """
    code = _GRIDS[resolution][2]
    t_file = _file_url('temperature', 't', period, code, resolution, decade, base_url)
    s_file = _file_url('salinity', 's', period, code, resolution, decade, base_url)

    z = _fetch_axis(t_file, 'depth', timeout=timeout, verbose=verbose)
    last = z.size - 1
    t = _fetch_data(t_file, 't_an', last, lat_idx, lon_idx,
                    timeout=timeout, verbose=verbose)
    s = _fetch_data(s_file, 's_an', last, lat_idx, lon_idx,
                    timeout=timeout, verbose=verbose)

    return _truncate_column(z, t, s)


def _truncate_column(z, t, s) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trim a raw WOA column to its valid (surface-to-seafloor) extent.

    WOA columns are valid from the surface to the seafloor, then fill-valued;
    truncate at the first fill (``cut == 0`` means the cell is on land).
    """
    z, t, s = np.asarray(z, float), np.asarray(t, float), np.asarray(s, float)
    n = min(z.size, t.size, s.size)
    z, t, s = z[:n], t[:n], s[:n]
    valid = (t < WOA_FILL_THRESHOLD) & (s < WOA_FILL_THRESHOLD)
    cut = valid.size if valid.all() else int(np.argmax(~valid))
    return z[:cut], t[:cut], s[:cut]


WOA_SOURCES = ('opendap', 'local')


def _check_woa_source(source):
    if source not in WOA_SOURCES:
        raise ConfigurationError(
            f"WOA23 source must be one of {WOA_SOURCES}; got {source!r}.",
            remediation="Use 'opendap' (online) or 'local' (install.sh --data woa23).",
        )


def _get_column(source, period, lat_idx, lon_idx, *, resolution, decade,
                base_url, timeout, verbose):
    """One ``(z, t, s)`` column from the selected WOA23 backend."""
    if source == 'local':
        from uacpy.data import woa23_local
        return woa23_local.column(period, lat_idx, lon_idx,
                                  resolution=resolution, decade=decade)
    return _fetch_column(
        period, lat_idx, lon_idx, resolution=resolution, decade=decade,
        base_url=base_url, timeout=timeout, verbose=verbose,
    )


def _file_url(folder, var, period, code, resolution, decade, base_url) -> str:
    fname = f"woa23_{decade}_{var}{period:02d}_{code}.nc"
    return (f"{base_url.rstrip('/')}/{folder}/netcdf/{decade}/{resolution}/"
            f"{fname}")


def _fetch_axis(file_url, name, *, timeout, verbose) -> np.ndarray:
    """Read a 1-D coordinate array (e.g. ``depth``) from a WOA file."""
    text = http_get(f"{file_url}.ascii?{name}", timeout=timeout,
                    verbose=verbose, source='sound_speed').decode('utf-8', 'replace')
    axis = _parse_dods_axis(text, name)
    if axis is None:
        raise DataFetchError(
            f"Could not read '{name}' axis from {file_url}.",
            remediation="Check the WOA23 file/resolution exists on the server.",
        )
    return np.asarray(axis, dtype=float)


def _fetch_data(file_url, var, last, lat_idx, lon_idx, *, timeout, verbose) -> np.ndarray:
    """Read a single ``var`` water column ``[0][0:last][lat][lon]``."""
    query = f"{var}[0][0:{last}][{lat_idx}][{lon_idx}]"
    text = http_get(f"{file_url}.ascii?{query}", timeout=timeout,
                    verbose=verbose, source='sound_speed').decode('utf-8', 'replace')
    values, _ = _parse_dods_ascii(text)
    return np.asarray(values, dtype=float)


_DATA_ROW = re.compile(r'^\[[\d\]\[]*\],\s*(\S+)')


def _parse_dods_axis(text: str, name: str) -> Optional[List[float]]:
    """Parse a standalone 1-D coordinate response (``name[N]`` then values)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{name}[") and i + 1 < len(lines):
            return [float(x) for x in lines[i + 1].split(',') if x.strip()]
    return None


def _parse_dods_ascii(text: str) -> Tuple[List[float], Optional[List[float]]]:
    """Parse a DAP ``.ascii`` body into ``(values, depths)``.

    Collects the variable's array rows (``[i][j][k], value``) in order and the
    ``*.depth[...]`` coordinate map. Returns ``depths=None`` if no depth map
    is present.
    """
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('---'):
            start = i + 1
            break

    values: List[float] = []
    depths: Optional[List[float]] = None
    j = start
    while j < len(lines):
        s = lines[j].strip()
        m = _DATA_ROW.match(s)
        if m:
            values.append(float(m.group(1)))
        elif '.depth[' in s and j + 1 < len(lines):
            depths = [float(x) for x in lines[j + 1].split(',') if x.strip()]
            j += 1
        j += 1
    return values, depths
