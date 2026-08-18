"""Offline NSIDC sea-ice concentration monthly climatology.

``install.sh --data seaice`` builds a **monthly climatology** of sea-ice
concentration from the NSIDC **Sea Ice Index** (G02135, NOAA@NSIDC, public
domain): for each calendar month and hemisphere it averages the monthly
concentration grids over a recent reference period, and caches the 12 grids per
pole. This module then returns the climatological ice concentration at a point
and month — the local, offline analogue of WOA23 (also a monthly climatology),
not a per-date observation.

Sea ice matters acoustically at high latitudes: it replaces the wind-roughened
free surface with an ice cover (different scattering, suppressed wind noise).
:func:`sea_ice_surface` turns a fetched concentration into the elastic
``BoundaryProperties`` an ice canopy presents to the water column, and
:func:`fetch_sea_ice_surface` does the fetch-and-convert in one call (used by
``fetch_environment(surface_sources='seaice')``).

The grids are NSIDC polar-stereographic GeoTIFFs (North EPSG:3411, South
EPSG:3412, 25 km); reading them needs ``tifffile`` and the lon/lat → polar
reprojection needs ``pyproj`` (both default uacpy dependencies).
"""

import datetime as _dt
import io
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from uacpy._log import log_message
from uacpy.core.constants import (
    SEA_ICE_COMPRESSIONAL_ATTENUATION, SEA_ICE_COMPRESSIONAL_SPEED,
    SEA_ICE_DENSITY, SEA_ICE_EDGE_CONCENTRATION, SEA_ICE_SHEAR_ATTENUATION,
    SEA_ICE_SHEAR_SPEED,
)
from uacpy.core.environment import BoundaryProperties
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data import _cache
from uacpy.data._geo import (
    Coordinate, as_coordinate, normalize_lon, ring_offsets,
)
from uacpy.data._http import http_get
from uacpy.data._time import parse_date

__all__ = ['download_seaice_db', 'fetch_sea_ice_concentration',
           'fetch_sea_ice_concentration_transect', 'sea_ice_grid',
           'sea_ice_pixel', 'sea_ice_surface', 'fetch_sea_ice_surface',
           'sea_ice_surface_transect']

INDEX_FILE = 'seaice_climatology.pkl'
_BASE_URL = 'https://noaadata.apps.nsidc.org/NOAA/G02135'
_MONTHS = ['01_Jan', '02_Feb', '03_Mar', '04_Apr', '05_May', '06_Jun',
           '07_Jul', '08_Aug', '09_Sep', '10_Oct', '11_Nov', '12_Dec']
# Fixed NSIDC Sea Ice Polar Stereographic grids, 25 km pixels. ``x0``/``y0`` are
# the outer corner of cell (0, 0) in projected metres — x0 the western edge and
# y0 the *northern* (maximum-y) edge, so rows count southward as y decreases.
# The cached grids are 448 x 304 (N) and 332 x 316 (S); under these origins the
# pole projects to (0, 0) and lands mid-grid, at cell (234, 154) and (174, 158).
_GRID = {
    'N': {'epsg': 'EPSG:3411', 'x0': -3850000.0, 'y0': 5850000.0, 'px': 25000.0},
    'S': {'epsg': 'EPSG:3412', 'x0': -3950000.0, 'y0': 4350000.0, 'px': 25000.0},
}
_POLE_HOLE = 2510         # unobserved cap near the pole — perennial ice → 1.0
# Codes <= 1000 are concentration in tenths of a percent (1000 = 100 %, hence
# the /1000 in _to_fraction); higher codes are flags, not data.
_MAX_CONC = 1000
_HEMI_DIR = {'N': 'north', 'S': 'south'}

_MODEL = {}               # cache_root -> dict(N=(12,H,W), S=(12,H,W), tf=...)
_cache.register_cache(_MODEL.clear)


def _monthly_url(hemi, year, month):
    return (f"{_BASE_URL}/{_HEMI_DIR[hemi]}/monthly/geotiff/{_MONTHS[month - 1]}/"
            f"{hemi}_{year}{month:02d}_concentration_v4.0.tif")


def _to_fraction(arr):
    """NSIDC coded concentration → fraction 0-1 (land/coast → NaN)."""
    f = np.full(arr.shape, np.nan, dtype=np.float32)
    valid = arr <= _MAX_CONC
    f[valid] = arr[valid].astype(np.float32) / 1000.0
    f[arr == _POLE_HOLE] = 1.0
    return f


def download_seaice_db(cache_dir=None, *, years=None, timeout=120.0,
                       verbose=False):
    """Build the monthly sea-ice climatology and cache it.

    Averages the NSIDC monthly concentration grids over ``years`` (default: the
    five most recent complete calendar years) per hemisphere and calendar month,
    writing ``<cache>/seaice/seaice_climatology.pkl``. Missing months are skipped.
    """
    import tifffile
    dest = Path(cache_dir) if cache_dir else _cache.dataset_root('seaice')
    dest.mkdir(parents=True, exist_ok=True)
    if years is None:
        end = _dt.date.today().year - 1
        years = range(end - 4, end + 1)
    years = list(years)
    log_message('seaice', f"building NSIDC sea-ice climatology over {years[0]}-"
                f"{years[-1]} (~{len(years) * 24} monthly grids)", verbose=verbose)

    climo = {}
    for hemi in ('N', 'S'):
        stacks = [[] for _ in range(12)]
        for year in years:
            for m in range(1, 13):
                try:
                    blob = http_get(_monthly_url(hemi, year, m), timeout=timeout,
                                    verbose=False, source='seaice')
                    arr = tifffile.imread(io.BytesIO(blob))
                except Exception:               # noqa: BLE001 — skip missing
                    continue
                stacks[m - 1].append(_to_fraction(arr))
        months = []
        for m in range(12):
            if not stacks[m]:
                raise DataFetchError(
                    f"No NSIDC sea-ice grids found for hemisphere {hemi}, "
                    f"month {m + 1} over {years}.",
                    remediation="Retry, or pass a different `years` range.",
                )
            with np.errstate(invalid='ignore'):
                months.append(np.nanmean(np.stack(stacks[m]), axis=0))
        climo[hemi] = np.stack(months).astype(np.float32)    # (12, H, W)
        log_message('seaice', f"hemisphere {hemi}: {climo[hemi].shape}",
                    verbose=verbose)

    out = dest / INDEX_FILE
    with open(out, 'wb') as fh:
        pickle.dump(climo, fh, protocol=pickle.HIGHEST_PROTOCOL)
    _MODEL.clear()
    log_message('seaice', f"sea-ice climatology cached → {out}", verbose=verbose)
    return out


def _pyproj_transformer(epsg):
    try:
        from pyproj import Transformer
    except ImportError as exc:                                  # pragma: no cover
        raise ConfigurationError(
            "The offline sea-ice backend needs 'pyproj'.",
            remediation="pyproj ships with the default uacpy install; reinstall "
                        "with `pip install -e .`, or `pip install pyproj`.",
        ) from exc
    return Transformer.from_crs("EPSG:4326", epsg, always_xy=True)


def _model():
    root = str(_cache.cache_root())
    if root in _MODEL:
        return _MODEL[root]
    path = _cache.require('seaice', INDEX_FILE)
    with open(path, 'rb') as fh:
        climo = pickle.load(fh)
    result = {'tf': {h: _pyproj_transformer(_GRID[h]['epsg']) for h in ('N', 'S')}}
    result.update(climo)
    _MODEL[root] = result
    return result


def _rowcol(model, hemi, lat, lon):
    """``(row, col)`` of a point in hemisphere ``hemi``, or ``None`` if outside.

    ``x0``/``y0`` are the outer corner of the corner cell, so the cell containing
    the point is floor(), not round() (which would bias it half a cell).
    """
    g = _GRID[hemi]
    x, y = model['tf'][hemi].transform(normalize_lon(lon), lat)
    col = int(np.floor((x - g['x0']) / g['px']))
    row = int(np.floor((g['y0'] - y) / g['px']))
    _, height, width = model[hemi].shape
    if not (0 <= row < height and 0 <= col < width):
        return None
    return row, col


# How far the search for an observed cell may wander, in grid cells (25 km each).
# NSIDC withholds a concentration at its coastline class (code 2530) because of
# land spillover in the passive-microwave footprint, not because the cell is dry,
# and `_to_fraction` cannot keep that class apart from true land (2540) once the
# climatology is averaged — both are NaN. So an unobserved cell with an observed
# ocean neighbour is treated as ocean and takes that neighbour's value, while a
# cell with no observed neighbour stays unobserved and raises. Same construct and
# the same reasoning as `sound_speed._WET_CELL_SEARCH_RINGS`; kept small so an
# inland request still fails rather than silently sampling a distant sea.
_OBSERVED_CELL_SEARCH_RINGS = 2


def _observed_at(grid, row, col):
    """Concentration at ``(row, col)``, else the nearest observed neighbour.

    Returns ``NaN`` when no cell within :data:`_OBSERVED_CELL_SEARCH_RINGS`
    carries a value, which is the signature of genuine land.
    """
    value = grid[row, col]
    if np.isfinite(value):
        return float(value)
    height, width = grid.shape
    for radius in range(1, _OBSERVED_CELL_SEARCH_RINGS + 1):
        for dr, dc in ring_offsets(radius):
            r, c = row + dr, col + dc
            if 0 <= r < height and 0 <= c < width:
                candidate = grid[r, c]
                if np.isfinite(candidate):
                    return float(candidate)
    return float('nan')


def _concentration(lat, lon, month):
    m = _model()
    hemi = 'N' if lat >= 0 else 'S'
    rc = _rowcol(m, hemi, lat, lon)
    if rc is None:
        return 0.0                              # outside the polar grid → ice-free
    return _observed_at(m[hemi][month - 1], *rc)


def fetch_sea_ice_concentration(point: Coordinate, *, date=None,
                                month: Optional[int] = None) -> float:
    """Climatological sea-ice concentration (0-1) at ``(lat, lon)`` for a month.

    Pass ``date`` (its month is used) or ``month`` (1-12). Points outside the
    polar grids return 0.0 (ice-free). A cell NSIDC leaves unobserved because of
    coastal land spillover takes its nearest observed ocean neighbour's value; a
    point with no observed cell within :data:`_OBSERVED_CELL_SEARCH_RINGS` is
    inland and raises ``DataFetchError``.
    """
    lat, lon = as_coordinate(point)
    if date is not None and month is not None:
        raise ConfigurationError(
            "fetch_sea_ice_concentration: pass either date= or month=, not both.")
    if date is not None:
        month = parse_date(date).month
    if month is None or not 1 <= int(month) <= 12:
        raise ConfigurationError(
            "fetch_sea_ice_concentration: a date= or month= (1-12) is required.")
    conc = _concentration(lat, lon, int(month))
    if not np.isfinite(conc):
        raise DataFetchError(
            f"NSIDC sea ice has no ocean value at {lat:.3f}, {lon:.3f}, nor "
            f"within {_OBSERVED_CELL_SEARCH_RINGS} grid cells of it — the point "
            f"is inland.",
            remediation="Pick an offshore point.",
        )
    return float(conc)


def sea_ice_surface(
    concentration: float, *,
    threshold: float = SEA_ICE_EDGE_CONCENTRATION,
) -> Optional[BoundaryProperties]:
    """Ice concentration (0-1) → the elastic surface the canopy presents.

    Above ``threshold`` (default the NSIDC 15 % ice-edge), returns a
    half-space :class:`~uacpy.core.environment.BoundaryProperties` for a
    homogeneous Arctic pack-ice canopy — compressional 3500 m/s, shear 1800
    m/s, density 0.9 g/cm³, attenuations 0.4 / 1.0 dB/λ (Jensen, Kuperman,
    Porter & Schmidt, *Computational Ocean Acoustics*). Below the threshold the
    surface is open water, so the function returns ``None`` (leave the
    free-surface default in place). The canopy is treated as a single
    homogeneous elastic boundary regardless of concentration; partial cover is
    reduced to the present/absent ice-edge decision rather than a mixed surface.

    A non-finite concentration (``NaN`` land/coast/out-of-grid cell) is treated
    as open water and returns ``None`` — never silently as ice, since
    ``NaN < threshold`` is False.
    """
    if not np.isfinite(concentration) or concentration < threshold:
        return None
    return BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=SEA_ICE_COMPRESSIONAL_SPEED,
        shear_speed=SEA_ICE_SHEAR_SPEED,
        density=SEA_ICE_DENSITY,
        attenuation=SEA_ICE_COMPRESSIONAL_ATTENUATION,
        shear_attenuation=SEA_ICE_SHEAR_ATTENUATION,
    )


def fetch_sea_ice_surface(
    point: Coordinate, *, date=None, month: Optional[int] = None,
    threshold: float = SEA_ICE_EDGE_CONCENTRATION,
) -> Optional[BoundaryProperties]:
    """Fetch the climatological ice concentration and convert it to a surface.

    Combines :func:`fetch_sea_ice_concentration` and :func:`sea_ice_surface`:
    returns the elastic ice ``BoundaryProperties`` where the point is
    ice-covered (concentration ≥ ``threshold``) for the given month, or ``None``
    for open water. Used by ``fetch_environment(surface_sources='seaice')``.
    """
    conc = fetch_sea_ice_concentration(point, date=date, month=month)
    return sea_ice_surface(conc, threshold=threshold)


def sea_ice_grid(month: int, *, hemi: str = 'N') -> np.ndarray:
    """Monthly climatology concentration grid (0-1, ``NaN`` = land) for mapping.

    ``hemi`` is ``'N'`` / ``'S'``; the array is on the NSIDC polar-stereographic
    grid (North EPSG:3411, South EPSG:3412, 25 km).
    """
    if hemi not in _GRID or not 1 <= int(month) <= 12:
        raise ConfigurationError(
            "sea_ice_grid: hemi must be 'N'/'S' and month 1-12.")
    return _model()[hemi][int(month) - 1]


def sea_ice_pixel(point: Coordinate, *, hemi: str = 'N'):
    """``(row, col)`` of a point in the NSIDC polar grid, or ``None`` if outside.

    Companion to :func:`sea_ice_grid` for overlaying markers on the grid: it
    shares its cell arithmetic with the value lookup, so a marker lands on
    exactly the cell whose concentration
    :func:`fetch_sea_ice_concentration` reads.
    """
    lat, lon = as_coordinate(point)
    return _rowcol(_model(), hemi, lat, lon)


def fetch_sea_ice_concentration_transect(start: Coordinate, end: Coordinate, *,
                                         date=None, month: Optional[int] = None,
                                         n_points: int = 6):
    """``(ranges_m, concentration)`` (0-1) sampled along ``start`` → ``end``."""
    from uacpy.data._geo import geodesic_waypoints
    lats, lons, ranges_m = geodesic_waypoints(start, end, n_points)
    out = []
    for la, lo in zip(lats, lons):
        try:
            out.append(fetch_sea_ice_concentration((la, lo), date=date, month=month))
        except DataFetchError:
            out.append(np.nan)                  # land along the transect
    return np.asarray(ranges_m), np.asarray(out)


def sea_ice_surface_transect(start: Coordinate, end: Coordinate, *,
                             date=None, month: Optional[int] = None,
                             n_points='auto', max_points=None,
                             threshold: float = SEA_ICE_EDGE_CONCENTRATION):
    """Range-dependent ice surface along ``start`` → ``end`` as a ``Surface``.

    Each waypoint becomes the elastic ice canopy where the concentration is
    ≥ ``threshold`` (see :func:`sea_ice_surface`), else open water (a vacuum
    boundary). The resulting :class:`~uacpy.core.surface.Surface` carries the
    marginal ice zone (open water → pack → open water) for inspection and
    plotting. The propagation solvers all carry a single global top boundary,
    so every model collapses a range-dependent surface to one boundary (with a
    ``UserWarning``); use the carrier to study / visualise the zone.

    With ``n_points='auto'`` (default) the transect is probed at ``max_points``
    points (cheap — the NSIDC climatology is a local cached grid) and each run
    of identical zones (ice canopy vs open water) collapses to the probe
    samples bracketing its edges, endpoints anchored — the ``Surface`` reads
    nearest-node, so every reconstructed ice edge lands within one probe step
    of the edge the probe observed, without an oversampled staircase. An
    integer samples exactly that many evenly-spaced waypoints.
    """
    from uacpy.core.surface import Surface
    from uacpy.core.bottom import BoundaryProperties
    from uacpy.data._geo import (
        run_boundary_indices, DEFAULT_MAX_TRANSECT_POINTS,
    )
    if max_points is None:
        max_points = DEFAULT_MAX_TRANSECT_POINTS
    probe_n = (int(max_points) if n_points == 'auto'
               else max(2, min(int(n_points), int(max_points))))
    ranges_m, conc = fetch_sea_ice_concentration_transect(
        start, end, date=date, month=month, n_points=probe_n)
    nodes = []
    for r, c in zip(ranges_m, conc):
        c = 0.0 if not np.isfinite(c) else float(c)
        bp = sea_ice_surface(c, threshold=threshold) \
            or BoundaryProperties(acoustic_type='vacuum')
        nodes.append((float(r), bp))
    if n_points == 'auto':
        # Identity = the boundary kind (homogeneous ice canopy vs open-water
        # vacuum); keep the samples bracketing each zone change plus the
        # endpoints, so the nearest-node Surface reproduces each edge where
        # the probe observed it.
        keys = [(bp.acoustic_type, bp.sound_speed, bp.shear_speed)
                for _, bp in nodes]
        nodes = [nodes[i] for i in run_boundary_indices(keys)]
    return Surface.coerce(nodes)
