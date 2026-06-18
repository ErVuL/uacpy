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
from uacpy.data._geo import Coordinate, as_coordinate, normalize_lon
from uacpy.data._http import http_get
from uacpy.data._time import parse_date

__all__ = ['download_seaice_db', 'fetch_sea_ice_concentration',
           'fetch_sea_ice_concentration_transect', 'sea_ice_grid',
           'sea_ice_pixel', 'sea_ice_surface', 'fetch_sea_ice_surface']

INDEX_FILE = 'seaice_climatology.pkl'
_BASE_URL = 'https://noaadata.apps.nsidc.org/NOAA/G02135'
_MONTHS = ['01_Jan', '02_Feb', '03_Mar', '04_Apr', '05_May', '06_Jun',
           '07_Jul', '08_Aug', '09_Sep', '10_Oct', '11_Nov', '12_Dec']
# Fixed NSIDC Sea Ice Polar Stereographic grids (origin x/y in m, 25 km pixels).
_GRID = {
    'N': {'epsg': 'EPSG:3411', 'x0': -3850000.0, 'y0': 5850000.0, 'px': 25000.0},
    'S': {'epsg': 'EPSG:3412', 'x0': -3950000.0, 'y0': 4350000.0, 'px': 25000.0},
}
_POLE_HOLE = 2510         # unobserved cap near the pole — perennial ice → 1.0
_MAX_CONC = 1000          # values ≤ 1000 are concentration ×10; above are flags
_HEMI_DIR = {'N': 'north', 'S': 'south'}

_MODEL = {}               # cache_root -> dict(N=(12,H,W), S=(12,H,W), tf=...)


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


def _concentration(lat, lon, month):
    m = _model()
    hemi = 'N' if lat >= 0 else 'S'
    g = _GRID[hemi]
    x, y = m['tf'][hemi].transform(normalize_lon(lon), lat)
    # x0/y0 are the outer corner of the corner cell (PixelIsArea), so the cell
    # containing the point is floor(), not round() (which would bias half a cell).
    col = int(np.floor((x - g['x0']) / g['px']))
    row = int(np.floor((g['y0'] - y) / g['px']))
    grid = m[hemi][month - 1]
    if not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]):
        return 0.0                              # outside the polar grid → ice-free
    return grid[row, col]


def fetch_sea_ice_concentration(point: Coordinate, date=None, *,
                                month: Optional[int] = None) -> float:
    """Climatological sea-ice concentration (0-1) at ``(lat, lon)`` for a month.

    Pass ``date`` (its month is used) or ``month`` (1-12). Points outside the
    polar grids return 0.0 (ice-free); land cells raise ``DataFetchError``.
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
            f"NSIDC sea ice has no ocean value at {lat:.3f}, {lon:.3f} "
            "(land / coast).",
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
    m/s, density 0.9 g/cm³, attenuations 0.5 / 1.0 dB/λ (Jensen, Kuperman,
    Porter & Schmidt, *Computational Ocean Acoustics*). Below the threshold the
    surface is open water, so the function returns ``None`` (leave the
    free-surface default in place). The canopy is treated as a single
    homogeneous elastic boundary regardless of concentration; partial cover is
    reduced to the present/absent ice-edge decision rather than a mixed surface.
    """
    if concentration < threshold:
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
    point: Coordinate, date=None, *, month: Optional[int] = None,
    threshold: float = SEA_ICE_EDGE_CONCENTRATION,
) -> Optional[BoundaryProperties]:
    """Fetch the climatological ice concentration and convert it to a surface.

    Combines :func:`fetch_sea_ice_concentration` and :func:`sea_ice_surface`:
    returns the elastic ice ``BoundaryProperties`` where the point is
    ice-covered (concentration ≥ ``threshold``) for the given month, or ``None``
    for open water. Used by ``fetch_environment(surface_sources='seaice')``.
    """
    conc = fetch_sea_ice_concentration(point, date, month=month)
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

    Companion to :func:`sea_ice_grid` for overlaying markers on the grid.
    """
    lat, lon = as_coordinate(point)
    g = _GRID[hemi]
    x, y = _model()['tf'][hemi].transform(normalize_lon(lon), lat)
    # floor (not round) to match _concentration: x0/y0 are the corner of cell
    # (0,0) (PixelIsArea), so a marker lands on the same cell whose value is read.
    col = int(np.floor((x - g['x0']) / g['px']))
    row = int(np.floor((g['y0'] - y) / g['px']))
    grid = _model()[hemi][0]
    if not (0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]):
        return None
    return row, col


def fetch_sea_ice_concentration_transect(start: Coordinate, end: Coordinate, *,
                                         date=None, month: Optional[int] = None,
                                         n_points: int = 6):
    """``(ranges_m, concentration)`` (0-1) sampled along ``start`` → ``end``."""
    from uacpy.data._geo import geodesic_waypoints
    lats, lons, ranges_m = geodesic_waypoints(start, end, n_points)
    out = []
    for la, lo in zip(lats, lons):
        try:
            out.append(fetch_sea_ice_concentration((la, lo), date, month=month))
        except DataFetchError:
            out.append(np.nan)                  # land along the transect
    return np.asarray(ranges_m), np.asarray(out)
