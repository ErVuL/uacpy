"""Grid and transect helpers of the data layer.

The coordinate, great-circle and date helpers every layer shares are in
:mod:`uacpy.core.geo`.
"""

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from uacpy.core._export import ExportRecord
from uacpy.data.sources import DataProvenance
from uacpy.core.provenance import point_in_words

from uacpy.core.exceptions import (
    ConfigurationError, DataFetchError, FallbackWarning, ProvenanceWarning,
)
from uacpy.core.geo import great_circle_km
from uacpy.core._warn_frames import USER_FRAME_SKIP

__all__ = [
    'lon_linspace', 'nearest_indices', 'ring_offsets', 'run_representative_indices',
    'run_boundary_indices',
    'DEFAULT_MAX_TRANSECT_POINTS', 'checked_max_points',
    'checked_n_points', 'capped_n_points', 'require_source',
    'depth_from_elevation',
    'checked_max_distance', 'cell_half_diagonal_km', 'checked_offset',
]

#: Default ceiling on the number of points sampled along a transect *before*
#: the ``'auto'`` reduction step — i.e. the fetch budget. The reduced grid is
#: never larger than this (and usually much smaller after collapsing
#: duplicates). At OpenTopoData's ≤100 locations/request this is ≤10 requests
#: for a full-native bathy transect. Override via ``max_points=`` on the
#: fetchers / fetch_environment.
DEFAULT_MAX_TRANSECT_POINTS = 1000


def require_source(source, allowed, what: str, remediation: str) -> None:
    """Refuse a ``source`` keyword outside ``allowed`` (a tuple of ids) with
    a typed error; ``what`` names the keyword in the message."""
    if source not in allowed:
        raise ConfigurationError(
            f"{what} must be one of {allowed}; got {source!r}.",
            remediation=remediation,
        )


def depth_from_elevation(elev, lat, lon, *, dataset: str) -> float:
    """Water depth (m, positive down) from a ``dataset`` elevation (m,
    positive up) at ``(lat, lon)``; a non-negative elevation is land."""
    if elev >= 0.0:
        raise DataFetchError(
            f"{dataset} reports land (elevation {elev:.0f} m) at "
            f"({lat:.4f}, {lon:.4f}); no water column.",
            remediation="Pick a location offshore, or supply a depth directly.",
        )
    return -elev


def checked_max_points(max_points, who: str) -> int:
    """Validate a transect fetch budget: an integer of at least 2.

    ``n_points`` is guarded at >= 2 wherever it is accepted, but the cap it is
    reduced against was not, and ``'auto'`` resolves straight to the cap:
    ``max_points=1`` produced a one-waypoint "transect" with
    ``ranges_m=[0.0]``, ``max_points=0`` an empty one, and a negative value an
    untyped ``ValueError`` out of ``np.linspace``. Two points are the fewest
    that define a path.
    """
    try:
        value = int(max_points)
    except (TypeError, ValueError):
        raise ConfigurationError(
            f"{who}: max_points must be an integer >= 2, got "
            f"{max_points!r}.",
            remediation="Pass max_points>=2, the transect fetch budget.",
        ) from None
    if value < 2:
        raise ConfigurationError(
            f"{who}: max_points must be >= 2, got {max_points}.",
            remediation="Pass max_points>=2; fewer than two waypoints is not "
                        "a transect.",
        )
    return value


def checked_n_points(n_points, label: str, *, allow_auto: bool = False,
                     name: str = 'n_points'):
    """Validate a transect sample count: an integer of at least 2.

    Every transect fetcher validates its count here, so one input means one
    thing everywhere. Two waypoints are the fewest that define a path, and a
    fractional or non-numeric count is a caller mistake rather than something
    to round: ``n_points=1``, ``2.7`` and ``'x'`` all raise
    ``ConfigurationError``.

    Parameters
    ----------
    n_points : int or str
        The caller's value, unvalidated.
    label : str
        Caller name for the message, e.g. ``'fetch_wind_transect'``.
    allow_auto : bool, optional
        Accept the string ``'auto'`` and return it unchanged, for the fetchers
        that resolve a native sample count themselves. Default ``False``.
    name : str, optional
        The argument's name in the message (``'n_lat'`` for a grid axis).
        Default ``'n_points'``.

    Returns
    -------
    int or str
        The count as an ``int``, or ``'auto'`` when ``allow_auto`` admitted it.

    Raises
    ------
    ConfigurationError
        ``n_points`` is below 2, not a whole number, or not a number at all.
    """
    if allow_auto and isinstance(n_points, str) and n_points == 'auto':
        return 'auto'
    forms = ("an integer >= 2, or 'auto'" if allow_auto else "an integer >= 2")
    remediation = (f"Pass {name} as an int (e.g. {name}=50)"
                   + (f" or {name}='auto'." if allow_auto else "."))
    msg = (f"{label}: {name}={n_points!r} is not a sample count. "
           f"Valid forms: {forms}.")
    try:
        value = int(n_points)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(msg, remediation=remediation) from exc
    if value != n_points:            # a fraction (2.7) is refused, not truncated
        raise ConfigurationError(msg, remediation=remediation)
    if value < 2:
        raise ConfigurationError(
            f"{label}: {name} must be >= 2, got {n_points}.",
            remediation=f"Pass {name}>=2"
                        + (" or 'auto'; " if allow_auto else "; ")
                        + "fewer than two samples span nothing.",
        )
    return value


def capped_n_points(n_points: int, max_points, label: str) -> int:
    """``n_points`` clamped to ``max_points`` (``None``: no cap), with a
    ``FallbackWarning`` naming both when the clamp bites.

    The transect fetchers share this one cap rule for an explicit count;
    their ``'auto'`` branches resolve a probe count against ``max_points``
    themselves and never come here.
    """
    if max_points is None or n_points <= max_points:
        return n_points
    warnings.warn(
        f"{label}: n_points={n_points} exceeds max_points={max_points}; "
        f"sampling {max_points}.",
        FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
    return int(max_points)


def lon_linspace(lon0: float, lon1: float, n: int) -> np.ndarray:
    """``n`` longitudes from ``lon0`` to ``lon1`` going **eastward**, wrapped to
    ``[-180, 180)``.

    A range whose end is west of its start (e.g. ``(179, -179)``) is taken to
    cross the antimeridian eastward, so it samples the short strip over 180°
    rather than sweeping the long way through 0°. This mirrors the great-circle
    transect path; a non-crossing range (``lon1 >= lon0``) is a plain linspace.
    """
    lon0, lon1 = float(lon0), float(lon1)
    if lon1 < lon0:
        lon1 += 360.0
    raw = np.linspace(lon0, lon1, int(n))
    wrapped = ((raw + 180.0) % 360.0) - 180.0
    # Keep a node that lies exactly on +180 as +180: wrapping it to -180 made
    # a full-globe axis non-monotonic with a duplicated first column. Exact
    # equality is the right test — only an exact +180 wraps to exactly -180,
    # while other multiples of 360 offset from it (-180, 540) stay -180.
    wrapped[raw == 180.0] = 180.0
    return wrapped


def require_month(month, who: str) -> int:
    """``month`` as an int in 1-12, or ``ConfigurationError``: a bool, a
    fraction (``6.7``) or a non-number is refused rather than truncated to a
    month the caller never asked for."""
    if isinstance(month, (bool, np.bool_)):
        raise ConfigurationError(f"{who}: month={month!r} is a bool, not a month (1-12).")
    try:
        value = float(month)
    except (TypeError, ValueError):
        raise ConfigurationError(
            f"{who}: month={month!r} is not a number; pass an integer 1-12.") from None
    if not value.is_integer() or not 1 <= value <= 12:
        raise ConfigurationError(
            f"{who}: month={month!r} is not an integer month 1-12; a fraction "
            f"would be truncated to a month you did not ask for.")
    return int(value)


def nearest_indices(axis, queries) -> np.ndarray:
    """Nearest-node index into ``axis`` for each query, any axis orientation.

    A COARDS latitude axis is commonly stored **descending** (GMRT, EMODnet
    DTM), so a plain ``searchsorted`` (ascending-only, and an insertion index
    rather than the nearest node) is both biased and wrong. Sort once, bracket
    with ``searchsorted``, then pick the closer of the two neighbours, and map
    back to the original (possibly descending) ordering.
    """
    axis = np.asarray(axis, dtype=float)
    order = np.argsort(axis)
    sorted_axis = axis[order]
    pos = np.searchsorted(sorted_axis, queries)
    lo = np.clip(pos - 1, 0, sorted_axis.size - 1)
    hi = np.clip(pos, 0, sorted_axis.size - 1)
    pick_hi = np.abs(sorted_axis[hi] - queries) < np.abs(queries - sorted_axis[lo])
    nearest_sorted = np.where(pick_hi, hi, lo)
    return order[nearest_sorted]


def ring_offsets(radius: int) -> 'list[tuple[int, int]]':
    """``(d_row, d_col)`` offsets on the Chebyshev ring of the given radius,
    ordered nearest-first by squared Euclidean distance in cells.

    The expanding-ring neighbour search shared by the gridded climatology
    readers (WOA23's nearest-wet-cell fallback, NSIDC sea ice's
    nearest-observed-cell fallback): probe radius 1, then 2, … so the first
    hit is the closest usable cell.
    """
    out = [(d_row, d_col)
           for d_row in range(-radius, radius + 1)
           for d_col in range(-radius, radius + 1)
           if max(abs(d_row), abs(d_col)) == radius]
    return sorted(out, key=lambda o: o[0] ** 2 + o[1] ** 2)


def run_representative_indices(keys) -> 'list[int]':
    """Indices of one representative per maximal run of consecutive-equal keys.

    The reduction behind ``'auto'`` transect sampling for **interpolated**
    columns (the WOA23 SSP): probe a source's sample *identity* (e.g. the grid
    cell, or the nearest sample) at a fine set of waypoints, then keep one
    waypoint per distinct run. ``keys`` must be ``==``-comparable (tuples,
    scalars, or dataclasses); do not pass raw arrays. A carrier reconstructed
    by **nearest-node** lookup (categorical Surface/Bottom) uses
    :func:`run_boundary_indices` instead — a midpoint representative would
    displace each reconstructed transition to midway between run centres.

    Interior runs are represented by their **midpoint** (the centre of the
    range interval that sample covers). The **first and last** runs are
    anchored to the transect **endpoints** (indices ``0`` and ``n-1``) so the
    reduced range axis explicitly spans the full transect ``[0, L]`` rather
    than ``[first-cell-centre, last-cell-centre]``. A single run collapses to
    one representative at the start (range-independent transect).
    """
    runs = []
    i, n = 0, len(keys)
    while i < n:
        j = i
        while j + 1 < n and keys[j + 1] == keys[i]:
            j += 1
        runs.append((i, j))
        i = j + 1
    last = len(runs) - 1
    reps: list = []
    for k, (i, j) in enumerate(runs):
        if k == 0:
            reps.append(0)            # anchor transect start (range 0)
        elif k == last:
            reps.append(n - 1)        # anchor transect end (range L)
        else:
            reps.append((i + j) // 2)  # interior cell centre
    return reps


def run_boundary_indices(keys) -> 'list[int]':
    """Indices keeping both probe samples that bracket every change of key,
    plus the two transect endpoints.

    The reduction behind ``'auto'`` transect sampling for **categorical**
    carriers reconstructed by nearest-node lookup (the ice-canopy/open-water
    ``Surface``, the sediment-identity ``Bottom``): a nearest-node read places
    each transition midway between adjacent kept samples, so keeping the last
    sample of one run and the first of the next pins the reconstructed
    transition to within half a probe step of the boundary the probe observed.
    ``keys`` must be ``==``-comparable, as in
    :func:`run_representative_indices`; a single run collapses to one
    representative at the start (range-independent transect).
    """
    n = len(keys)
    if n == 0:
        return []
    out = [0]
    for i in range(1, n):
        if not keys[i] == keys[i - 1]:
            if out[-1] != i - 1:
                out.append(i - 1)   # last sample of the run ending at i-1
            out.append(i)           # first sample of the run starting at i
    if len(out) == 1:
        return out                  # single run: one representative (start)
    if out[-1] != n - 1:
        out.append(n - 1)           # anchor transect end (range L)
    return out


@dataclass(frozen=True, eq=False)
class AlongTrack(ExportRecord):
    """One quantity sampled at the waypoints of a ``start`` → ``end``
    transect, as the scalar transect fetchers return it.

    Attributes
    ----------
    ranges : ndarray
        The waypoints' great-circle distance from ``start`` (m).
    lats, lons : ndarray
        The waypoints (decimal degrees).
    data : ndarray
        The quantity at each waypoint; ``NaN`` where the source has none.
    unit : str
        The unit of ``data``.
    quantity : str
        What ``data`` is (``'sediment_thickness'``, ``'seabed_density'``,
        ``'sea_ice_concentration'``, ``'wind_speed'``).
    provenance : DataProvenance or None
        The source the values were read from, which the transect fetchers
        always set. ``None`` is user-supplied data: no catalogue source, so
        no credit is drawn.
    """

    ranges: np.ndarray
    lats: np.ndarray
    lons: np.ndarray
    data: np.ndarray
    unit: str
    quantity: str
    provenance: Optional[DataProvenance] = None

    _ARRAY_FIELDS = ('ranges', 'lats', 'lons', 'data')
    _TABLE_FIELDS = ('ranges', 'lats', 'lons', 'data')
    _XARRAY_FIELDS = {'data': 'data', 'range': 'ranges', 'lat': 'lats',
                      'lon': 'lons'}

    def _payload(self):
        return {'data': (self.data, ('range',), self.unit)}

    def _coords(self):
        return {'range': (self.ranges, 'm'),
                'lat': (self.lats, 'degrees_north', 'range'),
                'lon': (self.lons, 'degrees_east', 'range')}


# ── The data offset rule ─────────────────────────────────────────────────────
# Every point fetch records the requested point and the point its data stands
# for (the centre of the grid cell read, or the sample's own position), so
# ``DataProvenance.offset_km`` is always known. One decider then applies the
# same rule to every source: a warning past the source's own threshold, a
# refusal past the caller's ``max_distance_km``.

def checked_max_distance(max_distance_km, who: str) -> Optional[float]:
    """``max_distance_km`` validated: ``None`` (no limit beyond the warning)
    or a positive finite distance in km."""
    if max_distance_km is None:
        return None
    if (isinstance(max_distance_km, bool)
            or not isinstance(max_distance_km, (int, float, np.integer, np.floating))
            or not np.isfinite(max_distance_km) or max_distance_km <= 0):
        raise ConfigurationError(
            f"{who}: max_distance_km must be None or a positive number of km; "
            f"got {max_distance_km!r}.",
            remediation="Pass e.g. max_distance_km=25, or None for no limit.")
    return float(max_distance_km)


def cell_half_diagonal_km(centre_lat: float, dlat_deg: float,
                          dlon_deg: float) -> float:
    """The farthest (km) a point inside the ``dlat_deg`` x ``dlon_deg`` grid
    cell centred at latitude ``centre_lat`` can sit from that centre: the
    great-circle distance to the cell's farthest corner (its equatorward
    ones), measured as the offset is. An offset beyond it means the data did
    not come from the cell holding the point."""
    lat = float(centre_lat)
    half = 0.5 * float(dlat_deg)
    corners_lat = np.array([lat - half, lat - half, lat + half, lat + half])
    corners_lon = 0.5 * float(dlon_deg) * np.array([-1.0, 1.0, -1.0, 1.0])
    # The relative margin absorbs the rounding between this distance and the
    # offset, which reach one corner by different arithmetic.
    return float(np.max(great_circle_km(lat, 0.0, corners_lat,
                                        corners_lon))) * (1.0 + 1e-9)


def checked_offset(prov: DataProvenance, *, who: str, warn_km: float,
                   max_distance_km: Optional[float] = None) -> DataProvenance:
    """``prov`` once the offset rule has passed it.

    Refuses (``DataFetchError``) when the data stand more than
    ``max_distance_km`` from the requested point. Otherwise warns
    (``ProvenanceWarning``) when the data are not the point's own: for a
    record that knows (``prov.from_neighbour_cell`` not ``None``) exactly
    when the cell read is a neighbour, else when they stand more than
    ``warn_km`` away — the source's own threshold, a cell's half diagonal
    for a gridded source, a documented distance for sparse samples. The
    messages name the point in words (:meth:`DataProvenance.describe_point`).
    A record with no offset (either point unknown) passes as it is."""
    offset = prov.offset_km
    if offset is None:
        return prov
    where = (f"{prov.source.name}: the data come from "
             f"{'the ' if prov.point_kind else ''}"
             f"{prov.describe_point()}, {offset:.1f} km from the requested "
             f"point ({point_in_words(*prov.requested_point)})")
    if max_distance_km is not None and offset > max_distance_km:
        raise DataFetchError(
            f"{who}: {where}, past max_distance_km={max_distance_km:g}.",
            remediation=("Widen max_distance_km, move the point, or choose "
                         "another source."))
    if prov.from_neighbour_cell is not None:
        why = ("the requested point's own cell holds no data"
               if prov.from_neighbour_cell else None)
    else:
        why = (f"beyond the {warn_km:.1f} km this source's data stand for"
               if offset > warn_km else None)
    if why is not None:
        warnings.warn(
            f"{who}: {where}: {why}. Pass max_distance_km= to refuse data "
            f"that far.",
            ProvenanceWarning, skip_file_prefixes=USER_FRAME_SKIP)
    return prov
