"""WOA23 sound-speed fetch — GPS + date → ``SoundSpeedProfile``.

Turns a location (and, optionally, a calendar date/month) into a
depth-vs-sound-speed profile ready for ``Environment(ssp=...)``.

Temperature and salinity come from the NOAA/NCEI **World Ocean Atlas 2023**
objectively-analyzed climatology (``t_an`` / ``s_an``). ``source='woa23'``
(the default) is cache-first, as ``fetch_environment`` is: it reads the fields
from the install-time NetCDF grids (``install.sh --data woa23``) and, when they
are not installed, a single ``(lat, lon)`` water column from the NCEI THREDDS
server via the DAP ``.ascii`` response — stdlib text, no NetCDF dependency.
``source='local'`` reads the installed grids only. Sound speed is then
computed from T, S and pressure with the TEOS-10 equation (default), UNESCO
(Chen-Millero), Del Grosso or Mackenzie, all in :mod:`uacpy.core.acoustics`.

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
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.optimize import brentq

# teos10 is called directly here; every formula is reached by name through
# SOUND_SPEED_FORMULAS, which lives in core beside the equations and is
# imported rather than restated, so every fetcher resolves the same names.
from uacpy.core.acoustics.seawater import (
    DEFAULT_SOUND_SPEED_FORMULA, REFERENCE_LATITUDE_DEG, SOUND_SPEED_FORMULAS,
    canonical_formula,
    depth_to_pressure_dbar, sound_speed_at_depth,
)
from uacpy.core._export import ExportRecord
from uacpy.core._provenance import dedupe_provenance
from uacpy.core.environment import SoundSpeedProfile
from uacpy.data import _cache
from uacpy.core.geo import (
    great_circle_km, Coordinate, as_coordinate, normalize_lon,
    geodesic_waypoints,
)
from uacpy.data._geo import (
    require_month, ring_offsets, run_representative_indices, capped_n_points,
    require_source, DEFAULT_MAX_TRANSECT_POINTS, checked_max_points,
    checked_n_points, checked_max_distance, checked_offset, cell_half_diagonal_km,
)
from uacpy.core.geo import parse_date
from uacpy.core.exceptions import (
    ConfigurationError, DataFetchError, FallbackWarning,
)
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.data._chain import SourceChain, SourceProvider, first_answer
from uacpy.data._http import http_get
from uacpy.data.sources import SOURCES, DataProvenance
from uacpy._log import log_message
from uacpy.data._provenance_notice import one_provenance_notice

__all__ = ['fetch_ssp', 'fetch_ssp_transect', 'ssp_transect_plan',
           'fetch_ts_profile', 'TSProfile', 'assemble_range_dependent',
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




def _fetch_ssp_backend(
    point: Coordinate,
    *,
    date: Union[str, _dt.date, None] = None,
    month: Optional[int] = None,
    formula: str = DEFAULT_SOUND_SPEED_FORMULA,
    resolution: str = '1.00',
    backend: str = 'opendap',
    decade: str = DEFAULT_DECADE,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 60.0,
    verbose: Union[bool, str] = False,
    max_distance_km: Optional[float] = None,
) -> SoundSpeedProfile:
    """Backend of :func:`fetch_ssp`: ``backend`` is a backend token, as
    ``fetch_environment`` resolves them."""
    formula = canonical_formula(formula, 'fetch_ssp')
    lat, lon = as_coordinate(point)
    depths, temp, sal, lat_idx, lon_idx = _ts_profile_with_cell(
        point, date=date, month=month, resolution=resolution, source=backend,
        decade=decade, base_url=base_url, timeout=timeout, verbose=verbose,
    )
    # Mackenzie is stated in depth, so it reads the WOA23 depths directly;
    # the pressure equations convert them at the site latitude.
    c = np.asarray(sound_speed_at_depth(temp, sal, depths, formula=formula,
                                        latitude_deg=lat), dtype=float)
    log_message(
        'sound_speed', f"WOA23 SSP at {lat:.3f}, {lon:.3f}: {depths.size} "
        f"levels, c=[{c.min():.1f}, {c.max():.1f}] m/s", verbose=verbose,
    )
    # Provenance: WOA23 is a climatology snapped to a grid cell — the actual
    # "date" is a month/annual period, and the actual coordinates are the
    # centre of the cell the column was read from: the nearest cell, or the
    # closest wet neighbour when the nearest is dry, so ``offset_km`` measures
    # the real hop.
    prov = _woa_provenance(lat, lon, lat_idx, lon_idx, resolution, date,
                           month, deepest_m=float(np.max(depths)),
                           max_distance_km=max_distance_km)
    return SoundSpeedProfile(depths=depths, sound_speed=c, kind='measured',
                             data_sources=(prov,), formula=formula)


def fetch_ssp(
    point: Coordinate,
    *,
    date: Union[str, _dt.date, None] = None,
    month: Optional[int] = None,
    formula: str = DEFAULT_SOUND_SPEED_FORMULA,
    resolution: str = '1.00',
    source: Union[str, Sequence[str]] = 'woa23',
    decade: str = DEFAULT_DECADE,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 60.0,
    verbose: Union[bool, str] = False,
    max_days: Optional[float] = None,
    max_distance_km: Optional[float] = None,
) -> SoundSpeedProfile:
    """Sound-speed profile at a ``(lat, lon)`` point from World Ocean Atlas
    2023, or from the first source of a chain that answers.

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
    formula : {'teos10', 'unesco', 'delgrosso', 'mackenzie'}, optional
        Sound-speed equation. Default ``'teos10'``; see
        :func:`uacpy.core.acoustics.sound_speed_teos10` for why UNESCO
        (Chen-Millero 1977 as published) sits 0.6 m/s above it in deep water.
        ``'mackenzie'`` is :func:`uacpy.core.acoustics.sound_speed_mackenzie`, which
        takes depth in metres, so the table entry inverts the pressure to
        depth first (Leroy & Parthiot at the 45° reference latitude).
    resolution : {'1.00', '0.25'}, optional
        WOA grid spacing in degrees. Default ``'1.00'``.
    source : str or sequence of str, optional
        ``'woa23'`` (default) is cache-first, as ``fetch_environment`` is: the
        install-time WOA23 grids (``install.sh --data woa23``), else the NCEI
        THREDDS server; ``'local'`` reads the installed grids only.
        ``'copernicus'`` (Copernicus Marine) and ``'argo'`` (the nearest
        Argo cast) need ``date=``; ``'auto'`` is Argo, else Copernicus,
        else WOA23; a sequence of ids is tried in order. A chain answers
        with its first source that returns a profile, each cached first —
        the rule ``fetch_environment(ssp_sources=)`` resolves by.
    decade : str, optional
        WOA averaging period directory (default ``'decav'``).
    base_url, timeout, verbose
        THREDDS root, network timeout, logging gate.
    max_days : float, optional
        The staleness guard (days) of the time-specific sources, as in
        :func:`uacpy.data.fetch_environment`; WOA23 does not read it.
    max_distance_km : float, optional
        Refuse data standing farther than this (km) from ``point``, whichever
        source answers; ``None`` (default) sets no limit beyond the
        ``ProvenanceWarning`` a source gives when its data come from another
        cell than the point's (WOA23's wet-cell hop) or from a far sample.
        ``month``, ``decade`` and ``base_url`` reach WOA23 only.

    Returns
    -------
    SoundSpeedProfile
        1-D profile (``depths`` m, ``sound_speed`` m/s of shape
        ``(n_depth, 1)``), ready for ``Environment(ssp=...)``.

    Raises
    ------
    ConfigurationError
        Bad ``formula``/``resolution``, or both ``date`` and ``month`` given.
    DataFetchError
        Service failure, or the location is on land / has no profile.
    """
    formula = canonical_formula(formula, 'fetch_ssp')
    def woa23_call(backend):
        return _fetch_ssp_backend(
            point, backend=backend, date=date, month=month, formula=formula,
            resolution=resolution, decade=decade, base_url=base_url,
            timeout=timeout, verbose=verbose, max_distance_km=max_distance_km)

    max_distance_km = checked_max_distance(max_distance_km, 'fetch_ssp')
    from uacpy.data.bathymetry import _refuse_a_dry_point
    _refuse_a_dry_point(point, who='fetch_ssp', cache_only=source == 'local',
                        timeout=timeout, verbose=verbose)
    if isinstance(source, str) and source in WOA_SOURCES:
        return _woa_cache_first(source, woa23_call)
    return _ssp_chain_fetch(
        source, (point,), woa23_call,
        dict(date=date, formula=formula, max_days=max_days,
             max_distance_km=max_distance_km, timeout=timeout,
             verbose=verbose, who='fetch_ssp', keyword='source'),
        who='fetch_ssp')


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
    an explicit ``n_points`` above it is capped to it with a ``FallbackWarning``
    (the same cap warning :func:`fetch_bathy_transect` emits).

    Parameters
    ----------
    start, end : (lat, lon)
        Transect endpoints in decimal degrees.
    n_points : int or 'auto', optional
        Columns, or ``'auto'`` for one per distinct WOA cell. Default
        ``'auto'``.
    max_points : int, optional
        Cap on the probe and the result. Default 1000.
    resolution : str, optional
        WOA23 grid resolution in degrees, ``'1.00'`` or ``'0.25'``. Default
        ``'1.00'``.
    """
    if resolution not in _GRIDS:
        raise ConfigurationError(
            f"ssp_transect_plan: unknown resolution={resolution!r}.",
            remediation=f"Use one of {sorted(_GRIDS)}.")
    max_points = checked_max_points(max_points, 'ssp_transect_plan')
    n_points = checked_n_points(n_points, 'ssp_transect_plan', allow_auto=True)
    probe_n = (max_points if n_points == 'auto'
               else capped_n_points(n_points, max_points, 'ssp_transect_plan'))
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


def _fetch_ssp_transect_backend(
    start: Coordinate,
    end: Coordinate,
    *,
    n_points: Union[int, str] = 'auto',
    max_points: int = DEFAULT_MAX_TRANSECT_POINTS,
    date: Union[str, _dt.date, None] = None,
    month: Optional[int] = None,
    formula: str = DEFAULT_SOUND_SPEED_FORMULA,
    resolution: str = '1.00',
    backend: str = 'opendap',
    decade: str = DEFAULT_DECADE,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 60.0,
    verbose: Union[bool, str] = False,
    seafloor=None,
    max_distance_km: Optional[float] = None,
) -> SoundSpeedProfile:
    """Backend of :func:`fetch_ssp_transect`: ``backend`` is a backend token, as
    ``fetch_environment`` resolves them."""
    plan = ssp_transect_plan(start, end, n_points=n_points,
                             max_points=max_points, resolution=resolution)
    lats, lons, ranges_m = plan['lats'], plan['lons'], plan['ranges_m']
    columns = [
        _fetch_ssp_backend((la, lo), date=date, month=month, formula=formula,
                  resolution=resolution, backend=backend, decade=decade,
                  base_url=base_url, timeout=timeout, verbose=verbose,
                  max_distance_km=max_distance_km)
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


@one_provenance_notice(subject="the profile's data",
                       record='uacpy.data.citations(ssp)')
def fetch_ssp_transect(
    start: Coordinate,
    end: Coordinate,
    *,
    n_points: Union[int, str] = 'auto',
    max_points: int = DEFAULT_MAX_TRANSECT_POINTS,
    date: Union[str, _dt.date, None] = None,
    month: Optional[int] = None,
    formula: str = DEFAULT_SOUND_SPEED_FORMULA,
    resolution: str = '1.00',
    source: Union[str, Sequence[str]] = 'woa23',
    decade: str = DEFAULT_DECADE,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 60.0,
    verbose: Union[bool, str] = False,
    seafloor=None,
    max_days: Optional[float] = None,
    max_distance_km: Optional[float] = None,
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
    ``source`` takes the same chain spec; Argo has no transect fetch, so a
    chain passes over it, and ``'argo'`` alone is refused. The WOA23 sampling
    described above is WOA23's; :func:`ssp_transect_plan` resolves it without
    fetching.

    Parameters
    ----------
    start, end : (lat, lon)
        Transect endpoints in decimal degrees.
    n_points : int or 'auto', optional
        Columns, or ``'auto'`` (see above). Default ``'auto'``.
    max_points : int, optional
        Cap on the waypoints probed. Default 1000.
    date : str or datetime.date, optional
        Calendar date; WOA23 uses only its month. Exclusive with ``month``.
    month : int, optional
        Climatological month 1-12; ``None`` with no ``date`` is the annual
        mean.
    formula : {'teos10', 'unesco', 'delgrosso', 'mackenzie'}, optional
        Sound-speed equation. Default ``'teos10'``.
    source : str or sequence of str, optional
        As on :func:`fetch_ssp`, without ``'argo'``. Default ``'woa23'``.
    resolution : {'1.00', '0.25'}, optional
        WOA grid spacing in degrees. Default ``'1.00'``.
    decade : str, optional
        WOA averaging period directory. Default ``'decav'``.
    base_url : str, optional
        The THREDDS root. Default NCEI's WOA23 tree.
    timeout : float, optional
        Per-request network timeout in seconds.
    verbose : bool or str, optional
        Logging gate passed through to ``log_message``.
    seafloor : Bathymetry, optional
        The transect's bathymetry; a column short of its own seafloor is
        extended to it (see above).
    max_days : float, optional
        Staleness guard (days) of the time-specific sources.
    max_distance_km : float, optional
        Refuse a waypoint whose data stand farther than this (km) from it;
        ``None`` (default) sets no limit beyond the per-waypoint
        ``ProvenanceWarning``, as in :func:`fetch_ssp`.
    """
    formula = canonical_formula(formula, 'fetch_ssp_transect')
    max_distance_km = checked_max_distance(max_distance_km, 'fetch_ssp_transect')

    def woa23_call(backend):
        return _fetch_ssp_transect_backend(
            start, end, backend=backend, n_points=n_points,
            max_points=max_points, date=date, month=month, formula=formula,
            resolution=resolution, decade=decade, base_url=base_url,
            timeout=timeout, verbose=verbose, seafloor=seafloor,
            max_distance_km=max_distance_km)

    if isinstance(source, str) and source in WOA_SOURCES:
        return _woa_cache_first(source, woa23_call)
    return _ssp_chain_fetch(
        source, (start, end), woa23_call,
        dict(date=date, n_points=n_points, max_points=max_points,
             formula=formula, max_days=max_days, timeout=timeout,
             verbose=verbose, seafloor=seafloor, who='fetch_ssp_transect',
             keyword='source', max_distance_km=max_distance_km),
        who='fetch_ssp_transect')


def assemble_range_dependent(columns, ranges_m) -> SoundSpeedProfile:
    """Stack 1-D ``SoundSpeedProfile`` columns into a 2-D range-dependent one.

    The common depth axis is the union of every column's depth nodes (see the
    comment on the assembly for why an axis taken from one column loses the
    others' nodes); shallower columns hold their deepest value below their own
    seafloor (``np.interp`` constant-edge fill).
    Shared by the WOA23 and Copernicus transect fetchers. Columns are reordered
    to strictly increasing range, so a caller that supplies them out of order
    still gets a correctly-ordered range axis (the carriers assume ascending
    range). The columns' provenance is aggregated onto the assembled profile:
    one record per column, stamped with the column's range as ``range_m``,
    so every column's cell and offset stay on the profile. ``formula`` travels with them when
    every column agrees on it, so a later seafloor extension continues the
    assembled field under the equation that built it. A column that is itself
    range-dependent (more than one range column) raises
    ``ConfigurationError``.

    Parameters
    ----------
    columns : sequence of SoundSpeedProfile
        Range-independent profiles, one per range.
    ranges_m : array_like
        The range (m) of each column.
    """
    for i, col in enumerate(columns):
        n_ranges = np.asarray(col.sound_speed).shape[1]
        if n_ranges != 1:
            raise ConfigurationError(
                f"assemble_range_dependent: column {i} has {n_ranges} range "
                f"columns; each input must be a 1-D profile.",
                remediation="Pass one single-range SoundSpeedProfile per "
                            "range in ranges_m.")
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
        np.interp(z, col.depths, col.sound_speed[:, 0]) for col in columns
    ])
    # Union the columns' provenance through the carriers' own aggregator, as
    # ``Bottom``, ``Surface`` and ``Environment`` do: one record per column
    # read, each stamped with its column's range, so every column's offset
    # survives the assembly.
    sources = dedupe_provenance(columns, ranges=ranges)
    # One formula for the assembled field only if every column agrees on it;
    # a mixed stack has none, and the extension then falls back to the default
    # rather than picking an arbitrary column's equation.
    formulas = {getattr(c, 'formula', None) for c in columns}
    formula = formulas.pop() if len(formulas) == 1 else None
    return SoundSpeedProfile(depths=z, sound_speed=data, ranges=ranges,
                             kind='measured', data_sources=sources,
                             formula=formula)


def _woa_provenance(lat, lon, lat_idx, lon_idx, resolution, date, month, *,
                    deepest_m=None, who='fetch_ssp', max_distance_km=None):
    """The ``'woa23'`` record of a column read from cell ``(lat_idx,
    lon_idx)``: WOA23 is a climatology snapped to a grid cell — the actual
    "date" is a month/annual period, and the actual coordinates are the
    centre of the cell the column was read from (the nearest cell, or the
    closest wet neighbour when the nearest is dry), so ``offset_km`` measures
    the real hop. The offset rule (:func:`~uacpy.data._geo.checked_offset`)
    then warns when that hop leaves the cell holding the point, and refuses
    past ``max_distance_km``. A monthly column reaching below the 1500 m the
    monthly fields stop at (``deepest_m``, the column's deepest level) was
    continued by the annual mean, and the record says so: ``split_depth_m``
    and ``period_below``."""
    period = _resolve_period(date, month)
    lat_c, lon_c = _cell_center(lat_idx, lon_idx, resolution)
    step = _GRIDS[resolution][4]
    own_cell = _grid_index(lat, lon, resolution)[:2]
    prov = DataProvenance(
        source=SOURCES['woa23'],
        data_date=(f"month {period:02d} (climatology)" if period
                   else "annual mean (climatology)"),
        data_point=(lat_c, lon_c),
        requested_point=(lat, lon),
        requested_date=(str(parse_date(date)) if date is not None else None),
        point_kind='cell', cell_size_deg=float(step),
        from_neighbour_cell=(lat_idx, lon_idx) != own_cell,
        **({'split_depth_m': _MONTHLY_MAX_DEPTH, 'period_below': 'annual mean'}
           if period and deepest_m is not None
           and deepest_m > _MONTHLY_MAX_DEPTH else {}),
    )
    return checked_offset(prov, who=who,
                          warn_km=cell_half_diagonal_km(lat_c, step, step),
                          max_distance_km=max_distance_km)


@dataclass(frozen=True, eq=False)
class TSProfile(ExportRecord):
    """A temperature/salinity column, as :func:`fetch_ts_profile` and
    :func:`~uacpy.data.copernicus.fetch_ts_profile_operational` return it.

    Attributes
    ----------
    depths : ndarray
        The levels (m), truncated at the seafloor.
    temperature : ndarray
        Temperature (°C) at each level, of the kind ``temperature_kind``
        names.
    salinity : ndarray
        Practical salinity at each level.
    temperature_kind : {'in_situ'}
        Every source's temperature is in-situ: WOA23's ``t_an`` is, and the
        Copernicus potential ``thetao`` is converted at each level's
        pressure — what a sound-speed equation and Francois-Garrison take.
    provenance : DataProvenance
        The source, the requested point and date, and the actual cell and
        period (WOA23) or date and point (Copernicus) read.
    """

    depths: np.ndarray
    temperature: np.ndarray
    salinity: np.ndarray
    temperature_kind: str
    provenance: DataProvenance

    _ARRAY_FIELDS = ('depths', 'temperature', 'salinity')
    _TABLE_FIELDS = ('depths', 'temperature', 'salinity')


def _fetch_ts_profile_backend(
    point: Coordinate,
    *,
    date: Union[str, _dt.date, None] = None,
    month: Optional[int] = None,
    resolution: str = '1.00',
    backend: str = 'opendap',
    decade: str = DEFAULT_DECADE,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 60.0,
    verbose: Union[bool, str] = False,
    max_distance_km: Optional[float] = None,
) -> TSProfile:
    """Backend of :func:`fetch_ts_profile`: ``backend`` is a backend token, as
    ``fetch_environment`` resolves them."""
    depths, temp, sal, lat_idx, lon_idx = _ts_profile_with_cell(
        point, date=date, month=month, resolution=resolution, source=backend,
        decade=decade, base_url=base_url, timeout=timeout, verbose=verbose,
    )
    lat, lon = as_coordinate(point)
    return TSProfile(depths=depths, temperature=temp, salinity=sal,
                     temperature_kind='in_situ',
                     provenance=_woa_provenance(
                         lat, lon, lat_idx, lon_idx, resolution, date, month,
                         deepest_m=float(np.max(depths)),
                         who='fetch_ts_profile', max_distance_km=max_distance_km))


def fetch_ts_profile(
    point: Coordinate,
    *,
    date: Union[str, _dt.date, None] = None,
    month: Optional[int] = None,
    resolution: str = '1.00',
    source: str = 'woa23',
    decade: str = DEFAULT_DECADE,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 60.0,
    verbose: Union[bool, str] = False,
    max_distance_km: Optional[float] = None,
) -> TSProfile:
    """The WOA23 temperature/salinity column at a ``(lat, lon)`` point.

    Returns a :class:`TSProfile` on the WOA standard depth levels, truncated
    at the seafloor, its in-situ temperature in °C and its ``'woa23'``
    provenance (the cell and period read). Useful on its own for building
    absorption models (Francois-Garrison needs T, S — and pH, which WOA does
    not carry).

    See :func:`fetch_ssp` for the parameters; raises identically.

    Parameters
    ----------
    point : (lat, lon)
        Site coordinates in decimal degrees.
    date : str or datetime.date, optional
        Calendar date; only its month is used. Exclusive with ``month``.
    month : int, optional
        Climatological month 1-12; ``None`` with no ``date`` is the annual
        mean.
    source : str, optional
        ``'woa23'`` (default, cached first) or ``'local'``.
    resolution : {'1.00', '0.25'}, optional
        WOA grid spacing in degrees. Default ``'1.00'``.
    decade : str, optional
        WOA averaging period directory. Default ``'decav'``.
    base_url : str, optional
        The THREDDS root. Default NCEI's WOA23 tree.
    timeout : float, optional
        Per-request network timeout in seconds.
    verbose : bool or str, optional
        Logging gate passed through to ``log_message``.
    max_distance_km : float, optional
        Refuse a column read farther than this (km) from ``point``; ``None``
        (default) sets no limit beyond the wet-cell hop's ``ProvenanceWarning``.
    """
    max_distance_km = checked_max_distance(max_distance_km, 'fetch_ts_profile')
    from uacpy.data.bathymetry import _refuse_a_dry_point
    _refuse_a_dry_point(point, who='fetch_ts_profile',
                        cache_only=source == 'local', timeout=timeout,
                        verbose=verbose)
    return _woa_cache_first(source, lambda backend: _fetch_ts_profile_backend(
        point, backend=backend, date=date, month=month, resolution=resolution, decade=decade, base_url=base_url, timeout=timeout, verbose=verbose,
        max_distance_km=max_distance_km))


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
    require_source(source, _WOA_BACKENDS, 'WOA23 backend',
                   "Use 'opendap' or 'local'.")
    lat, lon = as_coordinate(point)
    period = _resolve_period(date, month)
    lat_idx, lon_idx, lat_c, lon_c = _grid_index(lat, lon, resolution)

    # A coastal request often snaps onto a land cell even though the point is
    # at sea, so fall back to the nearest wet neighbour rather than refusing.
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
    # A wet-cell hop is reported once, by the offset rule on the provenance
    # (_woa_provenance): its ProvenanceWarning names the cell and the km.

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


# Public source names: ``'woa23'`` is cache-first — the install-time grids,
# else the NCEI THREDDS server — as ``fetch_environment`` resolves it, and
# ``'local'`` the grids only (:data:`_WOA23_SSP`'s cached twin). The private
# ``_*_backend`` functions take the backend tokens.
WOA_SOURCES = ('woa23', 'local')
_WOA_BACKENDS = ('opendap', 'local')


def _woa_cache_first(source, call):
    """``call(backend)`` for each backend ``source`` names, cached first,
    raising the most substantive error if none answers.

    The installed grids fall through to THREDDS only when they are absent or
    unreadable (:func:`uacpy.data._cache.is_cache_miss`); a column the grids
    read as dry ends the chain, since THREDDS serves the same files."""
    require_source(source, WOA_SOURCES, 'WOA23 source',
                   "Use 'woa23' (the installed grids, else NCEI THREDDS "
                   "online) or 'local' (the installed grids only; "
                   "install.sh --data woa23).")
    answer, _attempt = first_answer(
        _WOA23_SSP.attempts(cache_only=(source == 'local')),
        lambda _source, backend: call(backend))
    return answer


def _woa23_point(backend, point, *, date, formula, resolution, timeout,
                 verbose, max_distance_km=None, **_request):
    """A ``fetch_environment`` SSP point fetch from one WOA23 backend."""
    return _fetch_ssp_backend(
        point, date=date, formula=formula, resolution=resolution,
        backend=backend, timeout=timeout, verbose=verbose,
        max_distance_km=max_distance_km)


def _woa23_transect(backend, start, end, *, n_points, max_points, date,
                    formula, resolution, timeout, verbose, seafloor,
                    max_distance_km=None, **_request):
    """A ``fetch_environment`` SSP transect from one WOA23 backend; each
    column extends to its own local ``seafloor`` before stacking."""
    return _fetch_ssp_transect_backend(
        start, end, n_points=n_points, max_points=max_points, date=date,
        formula=formula, resolution=resolution, backend=backend,
        timeout=timeout, verbose=verbose, seafloor=seafloor,
        max_distance_km=max_distance_km)


#: WOA23 as a sound-speed source: the installed grids, then NCEI THREDDS.
_WOA23_SSP = SourceProvider('woa23', ('local', 'opendap'), _woa23_point,
                            _woa23_transect)


def _require_ssp_date(source, date, *, who='fetch_environment',
                      keyword='ssp_sources'):
    """Refuse a time-specific SSP source without ``date=``, naming the
    ``who`` that chained it and the ``keyword`` it was named by."""
    if date is None:
        raise ConfigurationError(
            f"{who}: {keyword}={source!r} requires date=.",
            remediation=f"Pass a date, or use {keyword}='woa23'.",
        )


def _copernicus_ssp_point(backend, point, *, date, formula, max_days,
                          verbose, who='fetch_environment',
                          keyword='ssp_sources', max_distance_km=None,
                          **_request):
    """A Copernicus Marine sound-speed profile at ``point`` for ``date``."""
    _require_ssp_date('copernicus', date, who=who, keyword=keyword)
    from uacpy.data.copernicus import fetch_ssp_operational
    extra = {} if max_days is None else {'max_days': max_days}
    return fetch_ssp_operational(
        point, date=date, formula=formula, verbose=verbose,
        max_distance_km=max_distance_km, **extra,
    )


def _copernicus_ssp_transect(backend, start, end, *, date, n_points,
                             max_points, formula, max_days, verbose, seafloor,
                             who='fetch_environment',
                             keyword='ssp_sources', max_distance_km=None,
                             **_request):
    """A Copernicus Marine sound-speed transect for ``date``; each column
    extends to its own local ``seafloor`` before stacking."""
    _require_ssp_date('copernicus', date, who=who, keyword=keyword)
    from uacpy.data.copernicus import fetch_ssp_transect_operational
    extra = {} if max_days is None else {'max_days': max_days}
    return fetch_ssp_transect_operational(
        start, end, date=date, n_points=n_points, max_points=max_points,
        formula=formula, verbose=verbose, seafloor=seafloor,
        max_distance_km=max_distance_km, **extra)


def _argo_ssp_point(backend, point, *, date, formula, max_distance_km,
                    max_days, timeout, verbose, who='fetch_environment',
                    keyword='ssp_sources', **_request):
    """The sound-speed profile of the Argo cast nearest ``point`` and
    ``date``."""
    _require_ssp_date('argo', date, who=who, keyword=keyword)
    from uacpy.data.argo import fetch_ssp_argo
    extra = {} if max_days is None else {'max_days': max_days}
    return fetch_ssp_argo(
        point, date=date, formula=formula, timeout=timeout, verbose=verbose,
        max_distance_km=max_distance_km, **extra,
    )


#: The SSP sources, each with its backends cached first. ``'auto'`` prefers
#: real → model → climatology (the first two need ``date=`` / a Copernicus
#: login, else they fall through to WOA23). Argo has no transect fetch.
_SSP_CHAIN = SourceChain(
    'ssp',
    providers=(
        _WOA23_SSP,
        SourceProvider('copernicus', ('opendap',), _copernicus_ssp_point,
                       _copernicus_ssp_transect),
        SourceProvider('argo', ('opendap',), _argo_ssp_point),
    ),
    auto=('argo', 'copernicus', 'woa23'),
)


def _ssp_chain_fetch(source, where, woa23_call, request, *, who):
    """The first answer of the SSP chain ``source`` (a chain spec:
    ``'auto'``, ``'local'``, one id or a sequence) at ``where`` — the
    point, or both ends of a transect. WOA23 is fetched by
    ``woa23_call(backend)``, which carries the caller's WOA-only knobs; the
    other sources by their chain fetch with ``request``."""
    sources, cache_only = _SSP_CHAIN.resolve(source)
    attempts = _SSP_CHAIN.attempts(sources, cache_only=cache_only,
                                   who=who, keyword='source=')

    def call(src, backend):
        if src == 'woa23':
            return woa23_call(backend)
        provider = _SSP_CHAIN.provider(src)
        if len(where) == 1:
            return provider.point(backend, *where, **request)
        if provider.transect is None:
            usable = ' or '.join(repr(p.id) for p in _SSP_CHAIN.providers
                                 if p.transect is not None)
            raise ConfigurationError(
                f"{who}: source {src!r} has no transect fetch.",
                remediation=f"Use {usable}.",
            )
        return provider.transect(backend, *where, **request)

    answer, _attempt = first_answer(attempts, call)
    return answer


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
    # Keyed on the request alone; a test that swaps ``http_get`` for a stub
    # clears the memo through ``_cache.invalidate_grids()`` (it is registered
    # above) so it is never answered from a previous stub's column.
    key = (period, int(lat_idx), int(lon_idx), resolution, decade, base_url)
    hit = _COLUMN_MEMO.get(key)
    if hit is None:
        if len(_COLUMN_MEMO) >= _COLUMN_MEMO_MAX:
            _COLUMN_MEMO.clear()
        hit = _fetch_column(
            period, lat_idx, lon_idx, resolution=resolution, decade=decade,
            base_url=base_url, timeout=timeout, verbose=verbose,
        )
        _COLUMN_MEMO[key] = hit
    return tuple(np.array(a, copy=True) for a in hit)


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
# the equations span 1435-1555 m/s across it at the surface. Sub-zero
# temperatures are in range because polar deep water reaches -1.9 C and the
# profiles being extended were themselves built by evaluating the formula at
# those temperatures.
_EFFECTIVE_T_BRACKET_DEGC = (-3.0, 35.0)
# Below this, holding the last value is close enough to be not worth a warning:
# 50 m at the deep gradient is 0.9 m/s.
_EXTRAPOLATION_WARN_M = 50.0


def _deep_increment(c_deepest: float, z_from: float, z_to: float,
                    latitude: float, speed_fn=None) -> float:
    """Sound-speed increment from ``z_from`` down to ``z_to`` under the
    formula ``speed_fn(t, s, p_dbar)`` that built the column, a
    :data:`SOUND_SPEED_FORMULAS` entry (``None``: the default, TEOS-10).

    Extrapolation only ever happens below the deepest analysed level, so in the
    deep isothermal layer, where temperature is nearly constant and sound speed
    rises almost linearly under the pressure term alone (Stergiopoulos,
    *Advanced Signal Processing Handbook* 10.2). The increment is therefore
    the formula at fixed T/S. The temperature is not assumed: it is inverted
    from the column's own deepest sound speed, which holds the increment to
    0.07 m/s over a 3.3 km span against the formula at the true T/S (worst
    case over T in -1..6 C,
    S in 33..35.5, z in 1..8 km). Any single gradient is 7.3 m/s out over that
    span, because dc/dz is itself a function of depth: 0.0168 s^-1 at 1 km
    against 0.0189 s^-1 at 8 km.
    """
    if speed_fn is None:
        speed_fn = SOUND_SPEED_FORMULAS[DEFAULT_SOUND_SPEED_FORMULA]
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
                          latitude: float = REFERENCE_LATITUDE_DEG):
    """Extend ``ssp`` down to ``depth_max`` along its own deep gradient.

    Bathymetry (GEBCO, 15 arc-sec) and analysed T/S (WOA23, 1 deg) come from
    independent products, so the seafloor routinely sits below the deepest
    analysed level — by more than 200 m at ~15% of ocean points, and by 3.3 km
    in a trench. The carrier's generic ``extend_to`` holds the last value,
    which drops the entire pressure term: at (29.78, 142.77) the annual 1°
    WOA23 column ends at 5500 m / 1550.40 m/s (1.56 °C, 34.69), while TEOS-10
    at the 8801 m seafloor holding that deepest T/S gives 1611.07 m/s, so a
    held profile is 61 m/s (3.9%) slow over the bottom 3.3 km — enough to
    move ray turning depths and convergence-zone structure.

    The increment is :func:`_deep_increment`, evaluated per column from that
    column's own deepest sound speed, so it carries the depth dependence of the
    pressure term rather than a single gradient. At the trench point above the
    extension gives 1611.08 m/s, 0.01 m/s from that 1611.07 m/s reference.

    Parameters
    ----------
    ssp : SoundSpeedProfile
        The profile to extend.
    depth_max : float
        Depth (m) to extend to.
    latitude : float, optional
        Latitude (deg) of the depth-to-pressure conversion. Default 45.
    """
    depths = np.asarray(ssp.depths, dtype=float)
    last = float(depths[-1])
    if depth_max <= last or np.isclose(depth_max, last, rtol=1e-9, atol=1e-9):
        return ssp.extend_to(depth_max)      # trimming is the carrier's job

    data = np.asarray(ssp.sound_speed, dtype=float)
    span = depth_max - last
    # The extension continues the column under the formula that built it (a
    # Del Grosso column extended with UNESCO is 0.33 m/s off at 8.8 km); a
    # literal profile carries no formula and takes the package default.
    speed_fn = SOUND_SPEED_FORMULAS.get(
        ssp.formula or DEFAULT_SOUND_SPEED_FORMULA,
        SOUND_SPEED_FORMULAS[DEFAULT_SOUND_SPEED_FORMULA])
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
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )

    return type(ssp)(
        depths=np.append(depths, depth_max),
        sound_speed=np.vstack([data, new_row[None, :]]),
        ranges=(ssp.ranges.copy() if ssp.ranges is not None else None),
        kind=ssp.kind,
        data_sources=ssp.data_sources,
        # The rebuild has to restate ``formula``: without it the result
        # looks literal, and a *second* extension (transect column, then the
        # assembled profile against the bathymetry) reverts to the default. The
        # carrier's own copies (``SoundSpeedProfile._replace``, which every
        # slicer, ``collapse`` and ``extend_to`` go through, and ``copy``)
        # keep it, so this constructor call is the only one that must.
        formula=ssp.formula,
    )


def extend_column_to_seafloor(column, seafloor, range_m: float,
                              latitude: float = REFERENCE_LATITUDE_DEG):
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

    Parameters
    ----------
    column : SoundSpeedProfile
        One range-independent transect column.
    seafloor : Bathymetry
        The transect's bathymetry.
    range_m : float
        The column's range (m).
    latitude : float, optional
        Latitude (deg) of the depth-to-pressure conversion. Default 45.
    """
    depth = float(np.asarray(seafloor.eval(range=range_m)).flat[0])
    if depth <= float(np.asarray(column.depths)[-1]):
        return column
    return extend_ssp_below_data(column, depth, latitude=latitude)
