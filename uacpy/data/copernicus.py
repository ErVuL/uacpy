"""Copernicus Marine operational SSP — GPS + date → ``SoundSpeedProfile``.

Where WOA23
(:mod:`uacpy.data.sound_speed`) gives a reproducible *climatology*, the
Copernicus Marine Service gives *date-specific* conditions: reanalysis for
past dates, analysis/forecast for recent/near-future ones. Same output
contract — a :class:`~uacpy.core.environment.SoundSpeedProfile`.

With ``dataset_id=None`` (the default) the product is picked by date from
:data:`PHYSICS_PRODUCTS` (T/S) or :data:`BGC_PRODUCTS` (pH): the first product
whose time axis reaches the requested date, so a past date reads the
multi-year reanalysis and a date past its end the analysis/forecast. The
Copernicus Marine catalogue (2026-09-26) gives the daily physics reanalysis
1993-01-01 → 2026-06-23 (its interim extension is served as the same
dataset) and the analysis/forecast 2022-06-01 → its forecast horizon; the
reanalysis end advances monthly, which is why the choice reads each dataset's
own time axis rather than a date written here. An explicit ``dataset_id``
reads that one dataset.

The ``copernicusmarine`` toolbox is an optional extra rather than a core
dependency — it is credential-gated, imported at the point of use, and pulls
xarray/dask/zarr/boto3 in behind it. This source therefore needs both the
extra and a free Copernicus Marine account:

    pip install "uacpy[copernicus]"   # or: pip install -e ".[copernicus]"
    copernicusmarine login            # one-time, stores credentials

Temperature comes back as potential temperature (``thetao``), which the
sound-speed equations do not take: they want in-situ temperature. Every
column is therefore converted with
:func:`uacpy.core.acoustics.insitu_from_potential` (UNESCO 44 adiabatic
gradient)
before it leaves this module, the raw T/S accessor
(:func:`fetch_ts_profile_operational`) included. The correction is small in the upper ocean and
not negligible below it — at S=34.7 it is +0.64 m/s at 2000 m, +1.97 m/s at
5000 m and +4.98 m/s at 10000 m. WOA23 and Argo report in-situ temperature
already, so all three SSP routes agree on the deep column.

Unlike every other network source in this layer, these fetchers take no
``timeout``: the transport is the ``copernicusmarine`` session, which exposes
no per-call knob and reads its timeout and retry policy from the environment
(``COPERNICUSMARINE_HTTPS_TIMEOUT``, default 60 s;
``COPERNICUSMARINE_HTTPS_RETRIES``, default 5) when the toolbox is imported.
Set those to bound a call.
"""

import dataclasses
import datetime as _dt
from typing import Optional, Tuple, Union

import numpy as np

from uacpy.core.environment import SoundSpeedProfile
from uacpy.core.exceptions import DataFetchError
from uacpy.core.geo import (
    Coordinate, as_coordinate, normalize_lon, geodesic_waypoints,
)
from uacpy.data._geo import (
    checked_n_points, DEFAULT_MAX_TRANSECT_POINTS, checked_max_points,
    capped_n_points, checked_max_distance, checked_offset, cell_half_diagonal_km,
)
from uacpy.core.geo import parse_date
from uacpy.core.acoustics.seawater import (
    DEFAULT_SOUND_SPEED_FORMULA, canonical_formula,
    depth_to_pressure_dbar,
    insitu_from_potential, sound_speed_at_depth,
)
from uacpy.data.sound_speed import (
    TSProfile,
    assemble_range_dependent,
    extend_column_to_seafloor,
)
from uacpy.data.sources import SOURCES, DataProvenance
from uacpy.data.waves import SeaStateRecord
from uacpy._log import log_message

from uacpy.data._offset_policy import checked_date_offset
from uacpy.data._provenance_notice import one_provenance_notice

__all__ = [
    'fetch_ssp_operational',
    'fetch_ssp_transect_operational',
    'fetch_ts_profile_operational',
    'fetch_waves_operational',
    'fetch_ph_operational',
]

#: Global ocean physics, daily means, 1/12°, in date order: each entry is
#: ``(temperature dataset, salinity dataset)``. The multi-year reanalysis
#: (GLOBAL_MULTIYEAR_PHY_001_030) serves both in one dataset; the
#: analysis/forecast (GLOBAL_ANALYSISFORECAST_PHY_001_024) serves ``thetao``
#: and ``so`` as two datasets.
PHYSICS_PRODUCTS = (
    ('cmems_mod_glo_phy_my_0.083deg_P1D-m',
     'cmems_mod_glo_phy_my_0.083deg_P1D-m'),
    ('cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m',
     'cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m'),
)
TEMPERATURE_VAR = 'thetao'
SALINITY_VAR = 'so'
# Global wave reanalysis (WAVERYS), 1/5°, 3-hourly, 1980→present.
DEFAULT_WAVE_DATASET_ID = 'cmems_mod_glo_wav_my_0.2deg_PT3H-i'
WAVE_HS_VAR = 'VHM0'                 # spectral significant wave height (m)
WAVE_TP_VAR = 'VTPK'                 # wave peak period (s)
#: Global biogeochemistry datasets carrying ``ph``, 0.25°, in date order: the
#: monthly multi-year reanalysis (GLOBAL_MULTIYEAR_BGC_001_029) and the daily
#: analysis/forecast carbonate dataset (GLOBAL_ANALYSISFORECAST_BGC_001_028).
BGC_PRODUCTS = (
    'cmems_mod_glo_bgc_my_0.25deg_P1M-m',
    'cmems_mod_glo_bgc-car_anfc_0.25deg_P1D-m',
)
BGC_PH_VAR = 'ph'                    # sea-water pH (total scale)
# Max days the nearest available time step may sit from the requested date
# before it counts as out-of-coverage and raises (shared tolerance contract
# with the other dated SSP sources — cf. argo.DEFAULT_MAX_DAYS=15). Looser than
# Argo's by design: this is daily-mean *model* output (smooth, persistent — and
# in-coverage the nearest step is sub-day, so this is really a coverage-edge
# guard), whereas an Argo profile is one real in-situ snapshot that the ocean
# decorrelates from within a couple of weeks. The default tracks how slowly the
# field varies: climatology (month) > model (31 d) > in-situ obs (15 d).
DEFAULT_MAX_DAYS = 31
#: Columns an ``n_points='auto'`` operational transect samples: the grid has
#: no cheap cell identity to probe, so 'auto' is a fixed count.
AUTO_TRANSECT_COLUMNS = 6


def _last_time(ds) -> Optional[np.datetime64]:
    """The last step (day) of a dataset's time axis, or ``None`` when the
    dataset exposes none."""
    try:
        times = np.asarray(ds['time'].values).reshape(-1)
    except (KeyError, AttributeError, TypeError):
        return None
    if times.size == 0:
        return None
    return np.datetime64(times.max(), 'D')


def _open_by_date(marine, when: Optional[str], products, *,
                  remediation=None) -> Tuple[tuple, str]:
    """Open the first product whose time axis reaches ``when``.

    ``products`` lists, in date order, tuples of the dataset ids one product
    is read from; the first id's time axis decides. The last product is taken
    when no earlier one reaches the date, and its own date check
    (:func:`_snapped_date`) then refuses a date past its horizon. A dataset
    exposing no time axis cannot be ruled out and is taken. Returns the opened
    datasets and the ``product`` string the provenance records (the distinct
    ids joined with ``' + '``).
    """
    kw = {} if remediation is None else {'remediation': remediation}
    day = None if when is None else np.datetime64(when, 'D')
    for i, ids in enumerate(products):
        first = _open_dataset(marine, ids[0], **kw)
        last = _last_time(first)
        if (i == len(products) - 1 or day is None or last is None
                or day <= last):
            opened = {ids[0]: first}
            for ds_id in ids[1:]:
                if ds_id not in opened:
                    opened[ds_id] = _open_dataset(marine, ds_id, **kw)
            return (tuple(opened[d] for d in ids),
                    ' + '.join(dict.fromkeys(ids)))


class _TwoDatasetColumn:
    """Two physics datasets read as one: ``so`` from the salinity dataset,
    every other name (``thetao``, ``depth``, ``latitude``, ``longitude``,
    ``time``) from the temperature dataset."""

    def __init__(self, t_ds, s_ds):
        self._t, self._s = t_ds, s_ds

    def __getitem__(self, key):
        return (self._s if key == SALINITY_VAR else self._t)[key]


def _open_physics(marine, when: Optional[str], dataset_id: Optional[str], *,
                  remediation=None):
    """``(dataset, product)`` holding ``thetao`` and ``so`` for ``when``:
    ``dataset_id`` when given, else the :data:`PHYSICS_PRODUCTS` entry picked
    by date."""
    products = (PHYSICS_PRODUCTS if dataset_id is None
                else ((dataset_id, dataset_id),))
    (t_ds, s_ds), product = _open_by_date(marine, when, products,
                                          remediation=remediation)
    return (t_ds if s_ds is t_ds else _TwoDatasetColumn(t_ds, s_ds)), product


def fetch_ssp_operational(
    point: Coordinate,
    *,
    date: Union[str, _dt.date],
    formula: str = DEFAULT_SOUND_SPEED_FORMULA,
    max_days: int = DEFAULT_MAX_DAYS,
    dataset_id: Optional[str] = None,
    verbose: Union[bool, str] = False,
    max_distance_km: Optional[float] = None,
) -> SoundSpeedProfile:
    """Date-specific sound-speed profile from Copernicus Marine.

    Parameters
    ----------
    point : (lat, lon)
        Latitude/longitude in decimal degrees (WGS84).
    date : str or datetime.date
        Calendar date of interest. The nearest available time step is used.
    formula : {'teos10', 'unesco', 'delgrosso', 'mackenzie'}, optional
        Sound-speed equation. Default ``'teos10'``.
    max_days : int, optional
        Maximum days the nearest available time step may differ from ``date``
        before raising ``DataFetchError`` (the date is outside the dataset's
        coverage). Default 31. The shared tolerance guard mirrors
        ``argo``'s — the nearest match is never silently substituted.
    max_distance_km : float, optional
        Refuse data whose grid node stands farther than this (km) from the
        requested point; ``None`` (default) sets no limit beyond the
        ``ProvenanceWarning`` given when the node read is not the requested
        point's own cell.
    dataset_id : str, optional
        Copernicus Marine dataset carrying both ``thetao`` and ``so``.
        Default ``None``: the product is picked by date from
        :data:`PHYSICS_PRODUCTS` (reanalysis, else analysis/forecast), and
        the provenance ``product`` names the dataset(s) read.
    verbose : bool or str, optional
        Logging gate.

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
    formula = canonical_formula(formula, 'fetch_ssp_operational')
    lat, lon = as_coordinate(point)
    depths, t_insitu, sal, actual, product = _ts_column(
        point, date=date, max_days=max_days, dataset_id=dataset_id,
        verbose=verbose,
    )
    # Mackenzie is stated in depth and reads the model depths directly.
    c = np.asarray(sound_speed_at_depth(t_insitu, sal, depths, formula=formula,
                                        latitude_deg=lat), dtype=float)
    log_message(
        'copernicus', f"operational SSP at {lat:.3f}, {lon:.3f} ({date}): "
        f"{depths.size} levels, c=[{c.min():.1f}, {c.max():.1f}] m/s",
        verbose=verbose,
    )
    prov = DataProvenance(
        source=SOURCES['copernicus'],
        data_date=actual['date'],
        data_point=actual['point'],
        requested_point=(lat, lon),
        requested_date=str(parse_date(date)),
        product=product,
    )
    prov = _offset_checked(prov, actual['cell'], who='fetch_ssp_operational',
                           max_distance_km=checked_max_distance(
                               max_distance_km, 'fetch_ssp_operational'))
    return SoundSpeedProfile(depths=depths, sound_speed=c, kind='measured',
                             data_sources=(prov,), formula=formula)


@one_provenance_notice(subject="the profile's data",
                       record='uacpy.data.citations(ssp)')
def fetch_ssp_transect_operational(
    start: Coordinate,
    end: Coordinate,
    *,
    date: Union[str, _dt.date],
    n_points: Union[int, str] = 'auto',
    max_points: int = DEFAULT_MAX_TRANSECT_POINTS,
    formula: str = DEFAULT_SOUND_SPEED_FORMULA,
    max_days: int = DEFAULT_MAX_DAYS,
    dataset_id: Optional[str] = None,
    verbose: Union[bool, str] = False,
    seafloor=None,
    max_distance_km: Optional[float] = None,
) -> SoundSpeedProfile:
    """Range-dependent operational SSP along ``start`` → ``end`` (Copernicus).

    ``seafloor`` (a :class:`~uacpy.core.environment.Bathymetry`, optional)
    supplies the local seafloor: a column stopping short of its own local
    depth is extended down to it before stacking, and one already reaching
    past it is left whole (see :func:`uacpy.data.fetch_ssp_transect`).

    The Copernicus counterpart of :func:`uacpy.data.fetch_ssp_transect`: opens
    the dataset once, samples ``n_points`` columns along the great-circle path,
    and assembles a 2-D range-dependent
    :class:`~uacpy.core.environment.SoundSpeedProfile`. See
    :func:`fetch_ssp_operational` for parameters/exceptions.

    ``n_points`` and ``max_points`` take the same values as
    :func:`uacpy.data.fetch_ssp_transect`'s. The Copernicus grid exposes no
    cheap cell identity to probe, so ``'auto'`` samples
    :data:`AUTO_TRANSECT_COLUMNS` columns (capped at ``max_points``); an
    explicit count above ``max_points`` is capped with a ``FallbackWarning``.

    Parameters
    ----------
    start, end : (lat, lon)
        Transect endpoints in decimal degrees.
    date : str or datetime.date
        Calendar date; the nearest available time step is used.
    n_points : int or 'auto', optional
        Columns, or ``'auto'`` for :data:`AUTO_TRANSECT_COLUMNS`. Default
        ``'auto'``.
    max_points : int, optional
        Cap on the columns. Default 1000.
    formula : {'teos10', 'unesco', 'delgrosso', 'mackenzie'}, optional
        Sound-speed equation. Default ``'teos10'``.
    max_days : int, optional
        Days the nearest time step may lie from ``date`` before
        ``DataFetchError``. Default 31.
    max_distance_km : float, optional
        Refuse data whose grid node stands farther than this (km) from the
        requested point; ``None`` (default) sets no limit beyond the
        ``ProvenanceWarning`` given when the node read is not the requested
        point's own cell.
    dataset_id : str, optional
        Copernicus Marine dataset; ``None`` picks it by date (see
        :func:`fetch_ssp_operational`).
    verbose : bool or str, optional
        Logging gate passed through to ``log_message``.
    seafloor : Bathymetry, optional
        The transect's bathymetry; a column short of its own seafloor is
        extended to it (see above).
    """
    formula = canonical_formula(formula, 'fetch_ssp_transect_operational')
    max_points = checked_max_points(max_points,
                                    'fetch_ssp_transect_operational')
    n_points = checked_n_points(n_points, 'fetch_ssp_transect_operational',
                                allow_auto=True)
    n_points = (min(AUTO_TRANSECT_COLUMNS, max_points) if n_points == 'auto'
                else capped_n_points(n_points, max_points,
                                     'fetch_ssp_transect_operational'))
    when = parse_date(date).isoformat()
    marine = _import_copernicusmarine()
    ds, product = _open_physics(
        marine, when, dataset_id,
        remediation="Run `copernicusmarine login` and check the dataset_id.")

    lats, lons, ranges_m = geodesic_waypoints(start, end, n_points)
    limit = checked_max_distance(max_distance_km, 'fetch_ssp_transect_operational')
    columns = []
    for la, lo in zip(lats, lons):
        depths, temp, sal, actual = _extract_ts(ds, la, lo, when,
                                                max_days=max_days)
        if depths.size == 0:
            raise DataFetchError(
                f"No Copernicus profile at {la:.3f}, {lo:.3f} on {when}.",
                remediation="Keep the transect within the dataset's wet domain.",
            )
        pressure = depth_to_pressure_dbar(depths, la)
        t_insitu = insitu_from_potential(salinity=sal, theta=temp,
                                         pressure_dbar=pressure)
        c = np.asarray(sound_speed_at_depth(t_insitu, sal, depths,
                                            formula=formula, latitude_deg=la),
                       dtype=float)
        prov = DataProvenance(source=SOURCES['copernicus'],
                              data_date=actual['date'],
                              data_point=actual['point'],
                              requested_point=(float(la), float(lo)),
                              requested_date=when, product=product)
        prov = _offset_checked(prov, actual['cell'],
                               who='fetch_ssp_transect_operational',
                               max_distance_km=limit)
        columns.append(SoundSpeedProfile(
            depths=depths, sound_speed=c, kind='measured', data_sources=(prov,),
            formula=formula))

    log_message(
        'copernicus', f"operational range-dependent SSP: {n_points} columns "
        f"over {ranges_m[-1] / 1000:.1f} km", verbose=verbose,
    )
    if seafloor is not None:
        # Same per-column extension as fetch_ssp_transect: never flat-hold a
        # shallower column inside its used water column.
        columns = [
            extend_column_to_seafloor(col, seafloor, r, latitude=la)
            for col, r, la in zip(columns, ranges_m, lats)
        ]
    return assemble_range_dependent(columns, ranges_m)


def fetch_ts_profile_operational(
    point: Coordinate,
    *,
    date: Union[str, _dt.date],
    max_days: int = DEFAULT_MAX_DAYS,
    dataset_id: Optional[str] = None,
    verbose: Union[bool, str] = False,
    max_distance_km: Optional[float] = None,
) -> TSProfile:
    """The temperature/salinity column from Copernicus, as a
    :class:`~uacpy.data.TSProfile` with its ``'copernicus'`` provenance (the
    snapped date and point, and the products read).

    Truncated at the seafloor (first non-finite level). See
    :func:`fetch_ssp_operational` for parameters and exceptions.

    The temperature is **in-situ**, like every other source's
    (:func:`uacpy.data.fetch_ts_profile` returns WOA23's in-situ ``t_an``): the
    dataset's ``thetao`` is potential temperature and is converted with
    :func:`uacpy.core.acoustics.insitu_from_potential` at each level's pressure,
    so the column goes straight into a sound-speed equation or
    :meth:`~uacpy.core.absorption.FrancoisGarrison.from_temperature_salinity`.

    Parameters
    ----------
    point : (lat, lon)
        Site coordinates in decimal degrees.
    date : str or datetime.date
        Calendar date; the nearest available time step is used.
    max_days : int, optional
        Days the nearest time step may lie from ``date`` before
        ``DataFetchError``. Default 31.
    max_distance_km : float, optional
        Refuse data whose grid node stands farther than this (km) from the
        requested point; ``None`` (default) sets no limit beyond the
        ``ProvenanceWarning`` given when the node read is not the requested
        point's own cell.
    dataset_id : str, optional
        Copernicus Marine dataset; ``None`` picks it by date (see
        :func:`fetch_ssp_operational`).
    verbose : bool or str, optional
        Logging gate passed through to ``log_message``.

    """
    lat, lon = as_coordinate(point)
    depths, temp, sal, actual, product = _ts_column(
        point, date=date, max_days=max_days, dataset_id=dataset_id,
        verbose=verbose)
    return TSProfile(depths=depths, temperature=temp, salinity=sal,
                     temperature_kind='in_situ',
                     provenance=_offset_checked(
                         DataProvenance(
                             source=SOURCES['copernicus'],
                             data_date=actual['date'], data_point=actual['point'],
                             requested_point=(lat, lon),
                             requested_date=str(parse_date(date)),
                             product=product),
                         actual['cell'], who='fetch_ts_profile_operational',
                         max_distance_km=checked_max_distance(
                             max_distance_km, 'fetch_ts_profile_operational')))


def _ts_column(point, *, date, max_days, dataset_id, verbose):
    """Open the product for ``date`` and pull one T/S column, with the
    temperature converted to in-situ and the snapped date/point.

    Returns ``(depths, temperature, salinity, actual, product)`` — the
    form :func:`fetch_ts_profile_operational` builds its record from.
    """
    lat, lon = as_coordinate(point)
    when = parse_date(date).isoformat()
    marine = _import_copernicusmarine()

    log_message('copernicus', f"opening {dataset_id or 'the physics product'} "
                f"for {lat:.3f}, {lon:.3f} on {when}",
                verbose=verbose, level='debug')
    ds, product = _open_physics(marine, when, dataset_id)

    depths, temp, sal, actual = _extract_ts(ds, lat, lon, when,
                                            max_days=max_days)
    if depths.size == 0:
        raise DataFetchError(
            f"No Copernicus profile at {lat:.3f}, {lon:.3f} on {when} "
            "(on land or outside the dataset domain).",
            remediation="Pick an ocean location/date within the dataset.",
        )
    # thetao is potential temperature; every caller wants in-situ.
    temp = insitu_from_potential(
        salinity=sal, theta=temp,
        pressure_dbar=depth_to_pressure_dbar(depths, lat))
    return depths, temp, sal, actual, product


#: Per fetcher ``kind``, the label its out-of-coverage message carries and the
#: alternative its remediation points at (:func:`_snapped_date`): the three
#: fetchers fall back to different sources.
_DATE_GAP_FALLBACKS = {
    'ssp': ('', "leave dataset_id unset so the product is picked by date, "
                "or use ssp_sources='woa23'"),
    'waves': (' waves', "use the WaveWatch III source"),
    'ph': (' pH', "rely on the GLODAP climatology / model default"),
}


def _snapped_date(da, when: Optional[str], max_days: int,
                  kind: str = 'ssp') -> Optional[str]:
    """The time step ``sel(method='nearest')`` landed on, or ``None``.

    ``method='nearest'`` snaps silently to the dataset edge for an out-of-range
    date; raise rather than substitute an edge value so the tolerance is
    honoured the same way the other dated sources honour it.

    ``kind`` (``'ssp'``, ``'waves'``, ``'ph'``) selects the label the message
    carries and the alternative source the remediation recommends
    (:data:`_DATE_GAP_FALLBACKS`).
    """
    if when is None or 'time' not in getattr(da, 'coords', {}):
        return None
    actual = np.datetime64(np.asarray(da['time'].values).reshape(-1)[0], 'D')
    gap = abs((actual - np.datetime64(when, 'D')) / np.timedelta64(1, 'D'))
    if gap > max_days:
        label, fallback = _DATE_GAP_FALLBACKS[kind]
        raise DataFetchError(
            f"Copernicus{label}: nearest available time is {actual} "
            f"({gap:.0f} days from requested {when}, > max_days={max_days}) "
            "— the date is outside the dataset's range.",
            remediation=f"Pass a date within range, raise max_days, or {fallback}.")
    return str(actual)


def _snapped_point(da) -> Optional[Coordinate]:
    """The grid cell centre ``sel(method='nearest')`` landed on, or ``None``."""
    coords = getattr(da, 'coords', {})
    if 'latitude' not in coords or 'longitude' not in coords:
        return None
    return (float(np.asarray(da['latitude'].values).reshape(-1)[0]),
            normalize_lon(float(np.asarray(da['longitude'].values).reshape(-1)[0])))


#: Spatial snap tolerance, as a multiple of the local grid spacing. A point
#: inside the domain lands at most half a cell from its nearest node, so a
#: hop beyond this many cells means the request fell outside the dataset's
#: lat/lon coverage and snapped to its edge.
_MAX_SNAP_CELLS = 1.5


def _axis_spacing(axis: np.ndarray, value: float) -> Optional[float]:
    """Node spacing (deg) of a 1-D coordinate axis at the node nearest
    ``value`` (the larger of the two adjacent gaps), or ``None`` for an axis
    with fewer than two nodes."""
    axis = np.asarray(axis, dtype=float).reshape(-1)
    if axis.size < 2:
        return None
    i = int(np.argmin(np.abs(axis - value)))
    gaps = np.abs(np.diff(axis))
    return float(max(gaps[max(i - 1, 0)], gaps[min(i, gaps.size - 1)]))


def _check_snapped_point(ds, lat: float, lon: float,
                         snapped: Optional[Coordinate]):
    """Reject a spatial snap larger than ``_MAX_SNAP_CELLS`` × grid spacing,
    and return the ``(lat, lon)`` grid spacing (deg) at the snapped node —
    the cell the offset rule measures against — or ``None`` when the dataset
    exposes no axes to derive it from.

    ``sel(method='nearest')`` carries no spatial tolerance: a point outside
    the dataset's lat/lon domain snaps silently to the edge cell — the hazard
    :func:`_snapped_date` guards on the time axis. ``lon`` must already be
    normalized to the convention handed to ``sel``. A dataset that exposes no
    1-D ``latitude``/``longitude`` axes offers no spacing to derive, so it is
    left unchecked.
    """
    if snapped is None:
        return None
    try:
        lat_axis = np.asarray(ds['latitude'].values, dtype=float).reshape(-1)
        lon_axis = np.asarray(ds['longitude'].values, dtype=float).reshape(-1)
    except (KeyError, AttributeError, TypeError):
        return None
    s_lat, s_lon = snapped
    d_lon = abs((lon - s_lon + 180.0) % 360.0 - 180.0)
    spacings = []
    for name, off, axis, s_val in (
            ('latitude', abs(lat - s_lat), lat_axis, s_lat),
            ('longitude', d_lon, lon_axis, s_lon)):
        spacing = _axis_spacing(axis, s_val)
        spacings.append(spacing)
        if not spacing:
            continue
        if off > _MAX_SNAP_CELLS * spacing:
            raise DataFetchError(
                f"Copernicus: nearest available cell is {s_lat:.4f}, "
                f"{s_lon:.4f} ({off:.3f} deg from the requested {name}, "
                f"> {_MAX_SNAP_CELLS} x the {spacing:.3f} deg grid spacing) "
                "— the point is outside the dataset's spatial domain.",
                remediation="Pick a point inside the dataset's coverage, or "
                            "use a global dataset_id.",
            )
    return tuple(spacings) if all(spacings) else None


def _offset_checked(prov, spacing, *, who, max_distance_km):
    """``prov`` through the offset rule: a warning once the snapped node
    stands outside the cell holding the requested point (``spacing`` is that
    cell, from :func:`_check_snapped_point`), a refusal past
    ``max_distance_km``; and through the date rule
    (:func:`~uacpy.data._offset_policy.checked_date_offset`): a warning once
    the snapped time step is too far from the requested date for its
    product (daily or monthly)."""
    square = bool(spacing) and abs(spacing[0] - spacing[1]) < 1e-12
    prov = dataclasses.replace(
        prov, point_kind='cell',
        cell_size_deg=float(spacing[0]) if square else None)
    warn_km = (cell_half_diagonal_km(prov.data_point[0], *spacing)
               if spacing else float('inf'))
    return checked_date_offset(
        checked_offset(prov, who=who, warn_km=warn_km,
                       max_distance_km=max_distance_km), who=who)


def _extract_ts(
    ds, lat: float, lon: float, when: Optional[str],
    *, temp_var: str = TEMPERATURE_VAR, sal_var: str = SALINITY_VAR,
    max_days: int = DEFAULT_MAX_DAYS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Pull a single seafloor-truncated water column from an xarray dataset.

    Returns ``(depths, temperature, salinity, actual)``; ``actual`` is the
    ``{'date', 'point'}`` the nearest-neighbour selection landed on (either may
    be ``None`` when the dataset does not expose that coordinate), which the
    callers stamp into their :class:`DataProvenance`.
    """
    depth = np.asarray(ds['depth'].values, dtype=float).reshape(-1)
    # sel(method='nearest') carries no tolerance: an out-of-range coordinate
    # snaps silently to the axis edge — _snapped_date guards the time axis and
    # _check_snapped_point the two spatial ones. A caller may give longitude
    # in [0, 360), so it is wrapped into one convention here rather than
    # handed to sel as supplied.
    sel = {'latitude': lat, 'longitude': normalize_lon(lon)}
    if when is not None:
        sel['time'] = when
    t_da = ds[temp_var].sel(method='nearest', **sel)
    actual = {'date': _snapped_date(t_da, when, max_days),
              'point': _snapped_point(t_da)}
    actual['cell'] = _check_snapped_point(ds, lat, sel['longitude'],
                                          actual['point'])
    t = np.asarray(t_da.values, float).reshape(-1)
    s = np.asarray(ds[sal_var].sel(method='nearest', **sel).values, float).reshape(-1)

    n = min(depth.size, t.size, s.size)
    depth, t, s = depth[:n], t[:n], s[:n]
    # Levels below the seafloor come back masked/non-finite, so the first
    # invalid level is the seafloor cut (index 0 = the cell is on land).
    valid = np.isfinite(t) & np.isfinite(s)
    cut = valid.size if valid.all() else int(np.argmax(~valid))
    return depth[:cut], t[:cut], s[:cut], actual


def fetch_waves_operational(
    point: Coordinate,
    *,
    date: Union[str, _dt.date],
    max_days: int = DEFAULT_MAX_DAYS,
    dataset_id: str = DEFAULT_WAVE_DATASET_ID,
    verbose: Union[bool, str] = False,
    max_distance_km: Optional[float] = None,
) -> SeaStateRecord:
    """Significant wave height (m) and peak period (s) from Copernicus WAVERYS.

    Returns a :class:`~uacpy.data.SeaStateRecord` at the ``(lat, lon)`` cell
    nearest
    ``date`` (full reanalysis history, 1980→present). ``tp`` is ``None`` when
    the dataset does not expose a peak-period variable; ``provenance`` is the
    ``'waverys'`` :class:`~uacpy.data.DataProvenance` with the snapped date
    and cell as ``data_date``/``data_point``. Raises ``DataFetchError`` on land / out of
    coverage or when the nearest time step is more than ``max_days`` away.

    Parameters
    ----------
    point : (lat, lon)
        Site coordinates in decimal degrees.
    date : str or datetime.date
        Calendar date; the nearest available time step is used.
    max_days : int, optional
        Days the nearest time step may lie from ``date`` before
        ``DataFetchError``. Default 31.
    max_distance_km : float, optional
        Refuse data whose grid node stands farther than this (km) from the
        requested point; ``None`` (default) sets no limit beyond the
        ``ProvenanceWarning`` given when the node read is not the requested
        point's own cell.
    dataset_id : str, optional
        The WAVERYS dataset. Default the 0.2° 3-hourly reanalysis.
    verbose : bool or str, optional
        Logging gate passed through to ``log_message``.
    """
    lat, lon = as_coordinate(point)
    when = parse_date(date).isoformat()
    marine = _import_copernicusmarine()
    log_message('waves', f"opening {dataset_id} for {lat:.3f}, {lon:.3f}",
                verbose=verbose, level='debug')
    ds = _open_dataset(marine, dataset_id)

    sel = {'latitude': lat, 'longitude': normalize_lon(lon), 'time': when}
    hs_da = ds[WAVE_HS_VAR].sel(method='nearest', **sel)
    snapped_point = _snapped_point(hs_da)
    cell = _check_snapped_point(ds, lat, sel['longitude'], snapped_point)
    snapped_date = _snapped_date(hs_da, when, max_days, 'waves')
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
    prov = DataProvenance(source=SOURCES['waverys'], data_date=snapped_date,
                          data_point=snapped_point, requested_point=(lat, lon),
                          requested_date=when, product=dataset_id)
    return SeaStateRecord(hs=hs, tp=tp, provenance=_offset_checked(
        prov, cell, who='fetch_waves_operational',
        max_distance_km=checked_max_distance(max_distance_km,
                                             'fetch_waves_operational')))


def fetch_ph_operational(
    point: Coordinate,
    *,
    date: Union[str, _dt.date],
    reference_depth: Optional[float] = None,
    max_days: int = DEFAULT_MAX_DAYS,
    dataset_id: Optional[str] = None,
    verbose: Union[bool, str] = False,
    max_distance_km: Optional[float] = None,
) -> float:
    """Date-specific seawater pH from Copernicus Marine biogeochemistry.

    The operational counterpart of the cached GLODAP climatology
    (:func:`uacpy.data.fetch_ph`): returns the pH at ``reference_depth`` (m,
    nearest finite level), or at the **mid-depth** of the finite levels when
    ``None`` — the depth :func:`uacpy.data.fetch_environment` reads its one
    pH at beside a fetched T/S profile. Pass ``reference_depth`` to pin the
    row to a T/S column of a different extent (``fetch_environment`` does).
    ``dataset_id=None`` picks
    the dataset by date from :data:`BGC_PRODUCTS`, as the physics fetchers
    pick theirs.

    Raises ``DataFetchError`` when ``copernicusmarine`` is unavailable, the
    service fails, the location has no column, or the nearest time step is more
    than ``max_days`` from ``date``.

    The value is raw and carries no provenance;
    :func:`uacpy.data.fetch_environment` with ``with_absorption=True``
    returns the carrier that records it in ``.data_sources``.

    Parameters
    ----------
    point : (lat, lon)
        Site coordinates in decimal degrees.
    date : str or datetime.date
        Calendar date; the nearest available time step is used.
    reference_depth : float, optional
        Depth (m) of the row returned; ``None`` is the mid-depth of the finite
        levels.
    max_days : int, optional
        Days the nearest time step may lie from ``date`` before
        ``DataFetchError``. Default 31.
    max_distance_km : float, optional
        Refuse data whose grid node stands farther than this (km) from the
        requested point; ``None`` (default) sets no limit beyond the
        ``ProvenanceWarning`` given when the node read is not the requested
        point's own cell.
    dataset_id : str, optional
        Copernicus Marine dataset; ``None`` picks it by date from
        :data:`BGC_PRODUCTS`.
    verbose : bool or str, optional
        Logging gate passed through to ``log_message``.
    """
    lat, lon = as_coordinate(point)
    when = parse_date(date).isoformat()
    marine = _import_copernicusmarine()
    log_message('copernicus', f"opening {dataset_id or 'the BGC product'} (pH) "
                f"for {lat:.3f}, {lon:.3f} on {when}", verbose=verbose,
                level='debug')
    products = (tuple((d,) for d in BGC_PRODUCTS) if dataset_id is None
                else ((dataset_id,),))
    (ds,), product = _open_by_date(marine, when, products)

    sel = {'latitude': lat, 'longitude': normalize_lon(lon), 'time': when}
    ph_da = ds[BGC_PH_VAR].sel(method='nearest', **sel)
    snapped = _snapped_point(ph_da)
    cell = _check_snapped_point(ds, lat, sel['longitude'], snapped)
    ph_date = _snapped_date(ph_da, when, max_days, 'ph')
    _offset_checked(DataProvenance(source=SOURCES['copernicus_bgc'],
                                   data_point=snapped, data_date=ph_date,
                                   requested_point=(lat, lon),
                                   requested_date=when, product=product),
                    cell, who='fetch_ph_operational',
                    max_distance_km=checked_max_distance(max_distance_km,
                                                         'fetch_ph_operational'))
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
    ref = (0.5 * (float(depth.min()) + float(depth.max()))
           if reference_depth is None else float(reference_depth))
    return float(ph[int(np.argmin(np.abs(depth - ref)))])


_LOGIN_HINT = ("Run `copernicusmarine login` (free account) and check "
               "the dataset_id and network connectivity.")


def _open_dataset(marine, dataset_id, *, remediation=_LOGIN_HINT):
    """Open a Copernicus Marine dataset, wrapping the toolbox's assorted
    auth/network failure modes in one typed :class:`DataFetchError`."""
    try:
        return marine.open_dataset(dataset_id=dataset_id)
    except Exception as exc:
        raise DataFetchError(
            f"Copernicus Marine open_dataset failed: {exc}.",
            remediation=remediation,
        ) from exc


def _import_copernicusmarine():
    try:
        import copernicusmarine
        return copernicusmarine
    except ImportError as exc:
        raise DataFetchError(
            "The 'copernicusmarine' toolbox is required for operational SSP "
            "but is not installed.",
            remediation="`copernicusmarine` is an optional extra: install it "
                        "with `pip install \"uacpy[copernicus]\"` (or "
                        "`pip install -e \".[copernicus]\"` from a checkout), "
                        "then run `copernicusmarine login` (free Copernicus "
                        "account).",
        ) from exc
