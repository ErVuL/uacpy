"""WOA23 sound-speed fetch — GPS + date → ``SoundSpeedProfile``.

Phase 2 of the on-demand external-data layer: turn a location (and,
optionally, a calendar date/month) into a depth-vs-sound-speed profile
ready for ``Environment(ssp=...)``.

Temperature and salinity come from the NOAA/NCEI **World Ocean Atlas 2023**
objectively-analyzed climatology (``t_an`` / ``s_an``). ``source='opendap'``
reads a single ``(lat, lon)`` water column from the NCEI THREDDS server via the
DAP ``.ascii`` response — stdlib text, no NetCDF dependency; ``source='local'``
reads the same fields from the install-time NetCDF grids (``install.sh --data
woa23``). Sound speed is then computed from T, S and pressure with the UNESCO
(Chen-Millero) or Del Grosso equation already in :mod:`uacpy.core.acoustics`.

Time handling
-------------
WOA23 is a *climatology*, not a forecast: ``date``/``month`` select a monthly
climatological mean (pass neither for the annual mean), never a specific
year's conditions. WOA's own *seasonal* periods (winter…autumn) are not
reachable from here — a month inside the season is the closest equivalent.
The monthly fields only resolve the upper 1500 m; below that the annual mean
is spliced on, giving a full-depth, season-aware profile. For true
date-specific conditions use the Copernicus Marine source
(:mod:`uacpy.data.copernicus`).
"""

import datetime as _dt
import re
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import brentq

from uacpy.core.acoustics import soundspeed_unesco, soundspeed_delgrosso
from uacpy.core._carrier_validate import _dedupe_provenance
from uacpy.core.environment import SoundSpeedProfile
from uacpy.data import _cache
from uacpy.data._geo import (
    require_month,
    great_circle_km,
    Coordinate, as_coordinate, normalize_lon, depth_to_pressure_dbar,
    geodesic_waypoints, ring_offsets, run_representative_indices,
    DEFAULT_MAX_TRANSECT_POINTS, checked_max_points, checked_n_points,
)
from uacpy.data._time import parse_date
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.data._http import http_get
from uacpy.data.sources import SOURCES, DataProvenance
from uacpy._log import log_message

__all__ = ['fetch_ssp', 'fetch_ssp_transect', 'ssp_transect_plan',
           'fetch_ts_profile', 'assemble_range_dependent',
           'extend_ssp_below_data', 'extend_column_to_seafloor']

DEFAULT_BASE_URL = 'https://www.ncei.noaa.gov/thredds-ocean/dodsC/woa23/DATA'
DEFAULT_DECADE = 'decav'              # 1955-2022 average of all decades
WOA_FILL_THRESHOLD = 1e30             # _FillValue is 9.96921e36

# Regular cell-centred grids: (n_lat, n_lon, file_resolution_code,
# first_lat_center, step_deg). Both axes are cell-centred — the first centre
# sits half a step inside the -90 / -180 edge — so the longitude origin is
# derived as -180 + step/2 rather than tabulated (see _cell_center).
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
    depths, temp, sal, lat_idx, lon_idx = _ts_profile_with_cell(
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
    # Provenance: WOA23 is a climatology snapped to a grid cell — the actual
    # "date" is a month/annual period, and the actual coordinates are the
    # centre of the cell the column was read from: the nearest cell, or the
    # closest wet neighbour when the nearest is dry, so ``offset_km`` measures
    # the real hop.
    period = _resolve_period(date, month)
    lat_c, lon_c = _cell_center(lat_idx, lon_idx, resolution)
    prov = DataProvenance(
        source=SOURCES['woa23'],
        data_date=(f"month {period:02d} (climatology)" if period
                   else "annual mean (climatology)"),
        data_point=(lat_c, lon_c),
        requested_point=(lat, lon),
        requested_date=(str(parse_date(date)) if date is not None else None),
    )
    return SoundSpeedProfile(depths=depths, data=c, shape='measured',
                             data_sources=(prov,), formula=formula)


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
    paying to fetch them. ``max_points`` caps the probe (and thus the result);
    an explicit ``n_points`` above it is capped to it with a ``UserWarning``
    (the same cap warning :func:`fetch_bathy_transect` emits).
    """
    if resolution not in _GRIDS:
        raise ConfigurationError(
            f"ssp_transect_plan: unknown resolution={resolution!r}.",
            remediation=f"Use one of {sorted(_GRIDS)}.")
    max_points = checked_max_points(max_points, 'ssp_transect_plan')
    n_points = checked_n_points(n_points, 'ssp_transect_plan', allow_auto=True)
    if n_points == 'auto':
        probe_n = max_points
    else:
        if n_points > max_points:
            warnings.warn(
                f"ssp_transect_plan: n_points={n_points} exceeds "
                f"max_points={max_points}; sampling {max_points}.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
        probe_n = min(int(n_points), max_points)
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
    seafloor=None,
) -> SoundSpeedProfile:
    """Range-dependent sound-speed profile along ``start`` → ``end``.

    ``seafloor`` (a :class:`~uacpy.core.environment.Bathymetry`, optional)
    supplies the local seafloor along the transect: a column that stops short
    of *its own* seafloor is extended down to it (deep-gradient
    extrapolation, :func:`extend_column_to_seafloor`) before the columns are
    stacked, so a shallower column is never flat-held inside its used water
    column. A column that already reaches past its seafloor is left whole —
    the transect's own reconciliation to the bathymetry happens once, on the
    assembled profile, in :func:`uacpy.data.fetch_environment`.

    With ``n_points='auto'`` (default) the transect is sampled at the
    **distinct WOA23 cells** the great-circle crosses: the grid cell is the
    sample identity (computed analytically via ``_grid_index`` — no network),
    consecutive waypoints in the same cell collapse, and one column is fetched
    **per distinct cell** (no duplicate column is ever fetched). This matches
    WOA's native range resolution — neither over- nor under-sampling. Pass an
    integer to sample exactly that many evenly-spaced columns instead.

    ``max_points`` caps the number of waypoints probed *before* the reduction
    (the fetch budget); the result is never larger.

    Columns are placed on a common depth axis (the union of every sampled
    column's own nodes); shallower columns hold their deepest value below their
    own seafloor (constant extrapolation, the usual SSP convention). Parameters
    otherwise mirror :func:`fetch_ssp`; ``start``/``end`` are ``(lat, lon)``.
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
    # Extend each column to its own local seafloor BEFORE stacking: the
    # common-axis assembly flat-holds a shallower column below its deepest
    # analysed level, and a single post-assembly extension repairs only the
    # segment below the common axis — measured -24 to -68 m/s inside the
    # used water column on a 3-column transect.
    if seafloor is not None:
        columns = [
            extend_column_to_seafloor(col, seafloor, r, latitude=la)
            for col, r, la in zip(columns, ranges_m, lats)
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

    The common depth axis is the union of every column's depth nodes (see the
    comment on the assembly for why an axis taken from one column loses the
    others' nodes); shallower columns hold their deepest value below their own
    seafloor (``np.interp`` constant-edge fill).
    Shared by the WOA23 and Copernicus transect fetchers. Columns are reordered
    to strictly increasing range, so a caller that supplies them out of order
    still gets a correctly-ordered range axis (the carriers assume ascending
    range). The columns' provenance is aggregated onto the assembled profile,
    de-duplicated by source id: one record survives per dataset, carrying the
    **first** column's cell/date specifics — the per-column cells are not
    enumerated on the assembled profile. ``formula`` travels with them when
    every column agrees on it, so a later seafloor extension continues the
    assembled field under the equation that built it.
    """
    ranges = np.asarray(ranges_m, dtype=float)
    order = np.argsort(ranges, kind='stable')
    ranges = ranges[order]
    columns = [columns[i] for i in order]
    # The union of every column's depth nodes: an axis taken from any ONE
    # column drops the others' own nodes (their seafloor-extension samples
    # included), and np.interp across the gaps re-flattens what the
    # extension just fixed — measured -11 to -31 m/s. The union reproduces
    # every column exactly (residual 0.0 across 68 real transects).
    z = np.unique(np.concatenate([np.asarray(c.depths, dtype=float)
                                  for c in columns]))
    data = np.column_stack([
        np.interp(z, col.depths, col.data[:, 0]) for col in columns
    ])
    # Union the columns' provenance through the carriers' own aggregator, so
    # an assembled profile de-duplicates by source id exactly as ``Bottom``,
    # ``Surface`` and ``Environment`` do (first-seen order, one record per
    # dataset, a column without ``data_sources`` contributing nothing).
    sources = _dedupe_provenance(columns)
    # One formula for the assembled field only if every column agrees on it;
    # a mixed stack has none, and the extension then falls back to UNESCO
    # rather than picking an arbitrary column's equation.
    formulas = {getattr(c, 'formula', None) for c in columns}
    formula = formulas.pop() if len(formulas) == 1 else None
    return SoundSpeedProfile(depths=z, data=data, ranges=ranges,
                             shape='measured', data_sources=sources,
                             formula=formula)


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
    depths, temp, sal, _lat_idx, _lon_idx = _ts_profile_with_cell(
        point, date=date, month=month, resolution=resolution, source=source,
        decade=decade, base_url=base_url, timeout=timeout, verbose=verbose,
    )
    return depths, temp, sal


def _ts_profile_with_cell(
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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """:func:`fetch_ts_profile` plus the grid cell the column actually came
    from: ``(depths, temp, sal, lat_idx, lon_idx)``.

    The returned indices are the wet cell the ring search settled on, which
    differs from the nearest cell when a coastal request snapped onto land —
    provenance stamping reads them so ``offset_km`` reports the real cell.
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

    # A coastal request often snaps onto a land cell even though the point is
    # at sea, so fall back to the nearest wet neighbour rather than refusing.
    nearest_idx = (lat_idx, lon_idx)
    depths, temp, sal, lat_idx, lon_idx = _nearest_wet_column(
        lambda i, j: _get_column(
            source, period, i, j, resolution=resolution, decade=decade,
            base_url=base_url, timeout=timeout, verbose=verbose,
        ),
        lat_idx, lon_idx, resolution, lat=lat, lon=lon,
    )
    if depths.size == 0:
        raise DataFetchError(
            f"WOA23 has no water-column data at {lat_c:.3f}, {lon_c:.3f} "
            f"or within {_WET_CELL_SEARCH_RINGS} grid cells "
            "(on land or outside the analyzed domain).",
            remediation="Pick an ocean location, or a coarser resolution.",
        )
    if (lat_idx, lon_idx) != nearest_idx:
        wet_lat, wet_lon = _cell_center(lat_idx, lon_idx, resolution)
        hop_km = float(great_circle_km(lat, lon, wet_lat, wet_lon))
        warnings.warn(
            f"WOA23: nearest cell ({lat_c:.3f}, {lon_c:.3f}) is dry; using the "
            f"closest wet cell ({wet_lat:.3f}, {wet_lon:.3f}), {hop_km:.0f} km "
            f"from the requested point.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
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

    return depths, temp, sal, lat_idx, lon_idx


_MONTHLY_MAX_DEPTH = 1500.0  # deepest level in WOA monthly/seasonal fields


def _resolve_period(date, month) -> int:
    """Map ``date``/``month`` to a WOA period code (0 annual, 1-12 monthly)."""
    if date is not None and month is not None:
        raise ConfigurationError(
            "WOA23: pass either date= or month=, not both.",
        )
    if date is not None:
        month = parse_date(date).month
    if month is None:
        return 0
    return require_month(month, "WOA23")


# How far the wet-cell search may wander from the nearest cell, in grid cells.
# A coastal point can snap onto a land cell whose column is entirely fill; the
# neighbouring cell is usually the same water mass. Kept small so a genuinely
# land-locked request still fails instead of silently sampling a distant sea.
_WET_CELL_SEARCH_RINGS = 2


def _nearest_wet_column(fetch, lat_idx, lon_idx, resolution, lat=None, lon=None):
    """``(depths, temp, sal, lat_idx, lon_idx)`` for the closest wet cell.

    ``fetch(lat_idx, lon_idx)`` returns one column; an empty depth axis means a
    dry (land / unanalysed) cell. The nearest cell is probed first; if it is
    dry, the cells within :data:`_WET_CELL_SEARCH_RINGS` rings are probed in
    order of great-circle distance from the REQUESTED point (``lat``, ``lon``;
    the nearest cell's centre when not given) — not in ring order, whose ties
    break in file order and whose distance ignores where inside the cell the
    request falls (a meridional neighbour is 111 km away, a zonal one 80 km at
    44°N). Longitude wraps; a row past either pole is skipped.
    """
    n_lat, n_lon, _code, _first, _step = _GRIDS[resolution]
    depths, temp, sal = fetch(lat_idx, lon_idx)
    if depths.size:
        return depths, temp, sal, lat_idx, lon_idx
    if lat is None or lon is None:
        lat, lon = _cell_center(lat_idx, lon_idx, resolution)
    candidates = []
    for radius in range(1, _WET_CELL_SEARCH_RINGS + 1):
        for d_lat, d_lon in ring_offsets(radius):
            i = lat_idx + d_lat
            if not 0 <= i < n_lat:
                continue
            j = (lon_idx + d_lon) % n_lon          # longitude wraps
            c_lat, c_lon = _cell_center(i, j, resolution)
            candidates.append((float(great_circle_km(lat, lon, c_lat, c_lon)), i, j))
    for _km, i, j in sorted(candidates):
        depths, temp, sal = fetch(i, j)
        if depths.size:
            return depths, temp, sal, i, j
    return depths, temp, sal, lat_idx, lon_idx


def _cell_center(lat_idx, lon_idx, resolution) -> Tuple[float, float]:
    """Centre ``(lat, lon)`` in degrees of one WOA grid cell."""
    _n_lat, _n_lon, _code, first_lat, step = _GRIDS[resolution]
    return first_lat + lat_idx * step, (-180.0 + step / 2) + lon_idx * step


def _grid_index(lat, lon, resolution) -> Tuple[int, int, float, float]:
    """Nearest WOA grid-cell indices and snapped centre coordinates."""
    if not -90.0 <= lat <= 90.0:
        raise ConfigurationError(f"fetch_ssp: lat must be in [-90, 90], got {lat}.")
    lon = normalize_lon(lon)
    n_lat, n_lon, _code, first_lat, step = _GRIDS[resolution]
    lat_idx = int(np.clip(round((lat - first_lat) / step), 0, n_lat - 1))
    lon_idx = int(np.clip(round((lon - (-180.0 + step / 2)) / step), 0, n_lon - 1))
    return (lat_idx, lon_idx) + _cell_center(lat_idx, lon_idx, resolution)


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
    truncate at the first fill (``cut == 0`` means the cell is on land). A
    level is invalid either as a raw above-threshold fill (the OPeNDAP ASCII
    path, which has no mask to read) or as ``NaN`` (the local reader, which
    resolves ``_FillValue`` through the netCDF mask).
    """
    z, t, s = np.asarray(z, float), np.asarray(t, float), np.asarray(s, float)
    n = min(z.size, t.size, s.size)
    z, t, s = z[:n], t[:n], s[:n]
    valid = (np.isfinite(t) & np.isfinite(s)
             & (t < WOA_FILL_THRESHOLD) & (s < WOA_FILL_THRESHOLD))
    cut = valid.size if valid.all() else int(np.argmax(~valid))
    return z[:cut], t[:cut], s[:cut]


WOA_SOURCES = ('opendap', 'local')


def _check_woa_source(source):
    if source not in WOA_SOURCES:
        raise ConfigurationError(
            f"WOA23 source must be one of {WOA_SOURCES}; got {source!r}.",
            remediation="Use 'opendap' (online) or 'local' (install.sh --data woa23).",
        )


#: Network columns already fetched in this process, keyed on everything
#: that selects one: ``fetch_environment(with_absorption=True)`` reads the
#: same T/S column twice (sound speed, then absorption), and a coastal ring
#: search probes the same dry cells each time. Bounded; a climatology column
#: does not change under a running process.
_COLUMN_MEMO: dict = {}
_COLUMN_MEMO_MAX = 64
_cache.register_cache(_COLUMN_MEMO.clear)


def _get_column(source, period, lat_idx, lon_idx, *, resolution, decade,
                base_url, timeout, verbose):
    """One ``(z, t, s)`` column from the selected WOA23 backend."""
    if source == 'local':
        from uacpy.data import woa23_local
        return woa23_local.column(period, lat_idx, lon_idx,
                                  resolution=resolution, decade=decade)
    # The fetch stack in use is stored beside the column and compared by
    # identity: a caller that swaps ``_fetch_column`` or ``http_get`` for a
    # stub is never answered from a previous one (``id()`` alone is reused
    # once a stub is garbage-collected). In production both are the module's
    # own functions.
    key = (period, int(lat_idx), int(lon_idx), resolution, decade, base_url)
    hit = _COLUMN_MEMO.get(key)
    if hit is None or hit[0] is not _fetch_column or hit[1] is not http_get:
        if len(_COLUMN_MEMO) >= _COLUMN_MEMO_MAX:
            _COLUMN_MEMO.clear()
        hit = (_fetch_column, http_get, _fetch_column(
            period, lat_idx, lon_idx, resolution=resolution, decade=decade,
            base_url=base_url, timeout=timeout, verbose=verbose,
        ))
        _COLUMN_MEMO[key] = hit
    return tuple(np.array(a, copy=True) for a in hit[2])


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
    """Read a single ``var`` water column ``[0][0:last][lat][lon]``.

    WOA fields are stored ``(time, depth, lat, lon)`` with a singleton time
    axis, hence the leading ``[0]``. DAP hyperslab bounds are **inclusive**, so
    the whole depth axis is ``0:n_depth-1`` — what the caller passes as ``last``.
    """
    query = f"{var}[0][0:{last}][{lat_idx}][{lon_idx}]"
    text = http_get(f"{file_url}.ascii?{query}", timeout=timeout,
                    verbose=verbose, source='sound_speed').decode('utf-8', 'replace')
    return np.asarray(_parse_dods_ascii(text), dtype=float)


# A DAP .ascii data row is an index tuple, a comma, then the value:
# ``[0][17][130][188], 3.4512``.
_DATA_ROW = re.compile(r'^\[[\d\]\[]*\],\s*(\S+)')


def _parse_dods_axis(text: str, name: str) -> Optional[List[float]]:
    """Parse a standalone 1-D coordinate response (``name[N]`` then values)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{name}[") and i + 1 < len(lines):
            return [float(x) for x in lines[i + 1].split(',') if x.strip()]
    return None


def _parse_dods_ascii(text: str) -> List[float]:
    """Parse a DAP ``.ascii`` body into the variable's values.

    Collects the variable's array rows (``[i][j][k], value``) in order. The
    depth axis is fetched separately (see :func:`_parse_dods_axis`).
    """
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('---'):
            start = i + 1
            break

    values: List[float] = []
    for line in lines[start:]:
        m = _DATA_ROW.match(line.strip())
        if m:
            values.append(float(m.group(1)))
    return values


# Reference salinity for the extrapolation below. Medwin & Clay (Fundamentals
# of Acoustical Oceanography 3.3.5) and Jensen et al. (Computational Ocean
# Acoustics, prob. 1.1) both take S = 35 for the deep ocean. The extrapolated
# increment moves by under 0.05 m/s over a 3.3 km span across S_ref in
# 33..36, because the inversion absorbs the difference into the temperature.
_DEEP_REFERENCE_SALINITY = 35.0
# Effective-temperature search bracket, wider than any ocean water mass:
# UNESCO spans 1435-1555 m/s across it at the surface. Sub-zero temperatures
# are in range because polar deep water reaches -1.9 C and the profiles being
# extended were themselves built by evaluating UNESCO at those temperatures.
_EFFECTIVE_T_BRACKET_DEGC = (-3.0, 35.0)
# Leroy & Parthiot's own reference latitude, for callers that have none. The
# pressure conversion is the only latitude-dependent step and the increment
# moves 0.33 m/s over a 3.3 km span from equator to pole.
_REFERENCE_LATITUDE_DEG = 45.0
# Below this, holding the last value is close enough to be not worth a warning:
# 50 m at the deep gradient is 0.9 m/s.
_EXTRAPOLATION_WARN_M = 50.0


def _deep_increment(c_deepest: float, z_from: float, z_to: float,
                    latitude: float, speed_fn=soundspeed_unesco) -> float:
    """Sound-speed increment from ``z_from`` down to ``z_to`` under the
    formula ``speed_fn(t, s, p)`` that built the column (UNESCO by default).

    Extrapolation only ever happens below the deepest analysed level, so in the
    deep isothermal layer, where temperature is nearly constant and sound speed
    rises almost linearly under the pressure term alone (Stergiopoulos,
    *Advanced Signal Processing Handbook* 10.2). The increment is therefore
    UNESCO at fixed T/S. The temperature is not assumed: it is inverted from the
    column's own deepest sound speed, which holds the increment to 0.07 m/s over
    a 3.3 km span against UNESCO at the true T/S (worst case over T in -1..6 C,
    S in 33..35.5, z in 1..8 km). Any single gradient is 7.3 m/s out over that
    span, because dc/dz is itself a function of depth: 0.0168 s^-1 at 1 km
    against 0.0189 s^-1 at 8 km.
    """
    p_from = float(depth_to_pressure_dbar(z_from, latitude))
    p_to = float(depth_to_pressure_dbar(z_to, latitude))
    t_lo, t_hi = _EFFECTIVE_T_BRACKET_DEGC
    salinity = _DEEP_REFERENCE_SALINITY
    if c_deepest <= speed_fn(t_lo, salinity, p_from):
        t_eff = t_lo                     # unphysical column: clamp, stay finite
    elif c_deepest >= speed_fn(t_hi, salinity, p_from):
        t_eff = t_hi
    else:
        t_eff = brentq(
            lambda t: speed_fn(t, salinity, p_from) - c_deepest,
            t_lo, t_hi, xtol=1e-8)
    return float(speed_fn(t_eff, salinity, p_to)
                 - speed_fn(t_eff, salinity, p_from))


def extend_ssp_below_data(ssp, depth_max: float,
                          latitude: float = _REFERENCE_LATITUDE_DEG):
    """Extend ``ssp`` down to ``depth_max`` along its own deep gradient.

    Bathymetry (GEBCO, 15 arc-sec) and analysed T/S (WOA23, 1 deg) come from
    independent products, so the seafloor routinely sits below the deepest
    analysed level — by more than 200 m at ~15% of ocean points, and by 3.3 km
    in a trench. The carrier's generic ``extend_to`` holds the last value,
    which drops the entire pressure term: at (29.78, 142.77) WOA ends at
    5500 m / 1551.05 m/s while UNESCO at the 8801 m seafloor gives 1611.93,
    so a held profile is 61 m/s (3.9%) slow over the bottom 3.3 km — enough to
    move ray turning depths and convergence-zone structure.

    The increment is :func:`_deep_increment`, evaluated per column from that
    column's own deepest sound speed, so it carries the depth dependence of the
    pressure term rather than a single gradient. At the trench point above it
    reproduces the 1611.93 m/s reference to 0.01 m/s.
    """
    depths = np.asarray(ssp.depths, dtype=float)
    last = float(depths[-1])
    if depth_max <= last or np.isclose(depth_max, last, rtol=1e-9, atol=1e-9):
        return ssp.extend_to(depth_max)      # trimming is the carrier's job

    data = np.asarray(ssp.data, dtype=float)
    span = depth_max - last
    # The extension continues the column under the formula that built it (a
    # Del Grosso column extended with UNESCO is 0.33 m/s off at 8.8 km); a
    # literal profile carries no formula and takes UNESCO.
    speed_fn = _FORMULAS.get(ssp.formula or 'unesco',
                             soundspeed_unesco)
    new_row = np.empty(data.shape[1], dtype=float)
    for j in range(data.shape[1]):
        new_row[j] = data[-1, j] + _deep_increment(
            float(data[-1, j]), last, depth_max, latitude, speed_fn)

    if span > _EXTRAPOLATION_WARN_M:
        warnings.warn(
            f"sound-speed profile ends at {last:.0f} m but the seafloor is at "
            f"{depth_max:.0f} m; extrapolated the last {span:.0f} m along the "
            f"profile's deep gradient to {new_row[0]:.1f} m/s. Analysed T/S "
            f"products are shallower than bathymetry over much of the deep "
            f"ocean — supply a measured profile if the deep column matters.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )

    return type(ssp)(
        depths=np.append(depths, depth_max),
        data=np.vstack([data, new_row[None, :]]),
        ranges=(ssp.ranges.copy() if ssp.ranges is not None else None),
        shape=ssp.shape,
        data_sources=ssp.data_sources,
        # The rebuild has to restate ``formula``; dropping it made the result
        # look literal, so a *second* extension (transect column, then the
        # assembled profile against the bathymetry) silently reverted to
        # UNESCO. The carrier's own copies keep it (``SoundSpeedProfile``
        # ``extend_to``/``subset``/``copy``), so this is the only gap.
        formula=ssp.formula,
    )


def extend_column_to_seafloor(column, seafloor, range_m: float,
                              latitude: float = _REFERENCE_LATITUDE_DEG):
    """One transect column, extended down to the seafloor under ``range_m``.

    Extension only: a column that already reaches past its local seafloor is
    returned unchanged. :func:`extend_ssp_below_data` delegates the shallower
    case to ``SoundSpeedProfile.extend_to``, which *truncates* — right for a
    single profile being reconciled to one water column, wrong per column
    along a transect. Bathymetry is sampled far more finely (50-60 points)
    than the SSP columns (7-35), so the seafloor *between* two column
    waypoints is routinely deeper than at either; a column cut back to its own
    waypoint's seafloor has lost analysed levels that the assembled field
    still interpolates through at those intermediate ranges, and the cut value
    is flat-held where real data existed. Sampling only inside the genuine
    water column, the cut costs 3.97 / 13.22 / 29.08 m/s on Biscay / North
    Atlantic / Hawaii-ridge transects against 1.64 / 1.96 / 9.63 m/s without
    it. Nothing downstream needs the cut here: ``fetch_environment``
    reconciles the assembled profile to the bathymetry afterwards, and each
    solver masks below its own local seafloor.
    """
    depth = float(np.asarray(seafloor.eval(range=range_m)).flat[0])
    if depth <= float(np.asarray(column.depths)[-1]):
        return column
    return extend_ssp_below_data(column, depth, latitude=latitude)
