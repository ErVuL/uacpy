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

import dataclasses
import datetime as _dt
import io
import warnings
from typing import Optional

import numpy as np

from uacpy._log import log_message
from uacpy.core.environment import BoundaryProperties
from uacpy.core.exceptions import (
    ConfigurationError, DataFetchError, FallbackWarning, IOWarning,
)
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.data import _cache
from uacpy.core.geo import (
    Coordinate, as_coordinate, normalize_lon, geodesic_waypoints,
    great_circle_km,
)
from uacpy.data._geo import (
    AlongTrack, require_month, ring_offsets, run_boundary_indices,
    DEFAULT_MAX_TRANSECT_POINTS, checked_max_points, checked_n_points,
    capped_n_points, checked_max_distance, checked_offset,
)
from uacpy.data._http import http_get
from uacpy.core.geo import parse_date
from uacpy.data.sources import SOURCES, DataProvenance
from uacpy.data._provenance_notice import one_provenance_notice

__all__ = ['download_seaice_db', 'fetch_sea_ice_concentration',
           'fetch_sea_ice_concentration_transect', 'sea_ice_grid',
           'sea_ice_pixel', 'sea_ice_surface', 'fetch_sea_ice_surface',
           'sea_ice_surface_transect', 'SEA_ICE_TYPICAL_ROUGHNESS_M',
           'climatology_period']

# Sea-ice canopy as a homogeneous elastic surface. Canonical Arctic pack-ice
# values from Jensen, Kuperman, Porter & Schmidt, *Computational Ocean
# Acoustics* (the ice cover modelled as a homogeneous elastic medium): cp 3500
# m/s, cs 1800 m/s, αp 0.4 dB/λ, αs 1.0 dB/λ ("realistic attenuations of
# 0.4 dB/λ for compressional waves and 1.0 dB/λ for shear waves"). Typical
# ranges (Etter, *Underwater Acoustic Modeling*): cp 1300-3900, cs 1400-1900
# m/s.
SEA_ICE_COMPRESSIONAL_SPEED = 3500.0       # m/s
SEA_ICE_SHEAR_SPEED = 1800.0               # m/s
SEA_ICE_DENSITY = 0.9                      # g/cm³
SEA_ICE_COMPRESSIONAL_ATTENUATION = 0.4    # dB/wavelength
SEA_ICE_SHEAR_ATTENUATION = 1.0            # dB/wavelength
# NSIDC standard ice-edge definition: ≥15 % concentration counts as ice-covered.
SEA_ICE_EDGE_CONCENTRATION = 0.15

INDEX_FILE = 'seaice_climatology.npz'
#: The pre-npz pickled climatology, refused by :func:`uacpy.data._cache.require_npz`.
RETIRED_INDEX_FILE = 'seaice_climatology.pkl'
_BASE_URL = 'https://noaadata.apps.nsidc.org/NOAA/G02135'
_MONTHS = ['01_Jan', '02_Feb', '03_Mar', '04_Apr', '05_May', '06_Jun',
           '07_Jul', '08_Aug', '09_Sep', '10_Oct', '11_Nov', '12_Dec']
# Fixed NSIDC Sea Ice Polar Stereographic grids, 25 km pixels. ``x0``/``y0`` are
# the outer corner of cell (0, 0) in projected metres — x0 the western edge and
# y0 the *northern* (maximum-y) edge, so rows count southward as y decreases.
# The cached grids are 448 x 304 (N) and 332 x 316 (S); under these origins the
# pole projects to (0, 0) and lands mid-grid, at cell (234, 154) and (174, 158).
# Both origins are whole multiples of the 25 km pixel, so the pole sits exactly
# on a cell *corner* and _rowcol's floor() picks the cell south-east of it.
_GRID = {
    'N': {'epsg': 'EPSG:3411', 'x0': -3850000.0, 'y0': 5850000.0, 'px': 25000.0},
    'S': {'epsg': 'EPSG:3412', 'x0': -3950000.0, 'y0': 4350000.0, 'px': 25000.0},
}
_POLE_HOLE = 2510         # unobserved cap near the pole — perennial ice → 1.0
# Codes <= 1000 are concentration in tenths of a percent (1000 = 100 %, hence
# the /1000 in _to_fraction); higher codes are flags, not data.
_MAX_CONC = 1000
_HEMI_DIR = {'N': 'north', 'S': 'south'}


def _monthly_url(hemi, year, month, base_url=_BASE_URL):
    return (f"{base_url}/{_HEMI_DIR[hemi]}/monthly/geotiff/{_MONTHS[month - 1]}/"
            f"{hemi}_{year}{month:02d}_concentration_v4.0.tif")


def _to_fraction(arr):
    """NSIDC coded concentration → fraction 0-1 (land/coast → NaN)."""
    f = np.full(arr.shape, np.nan, dtype=np.float32)
    valid = arr <= _MAX_CONC
    f[valid] = arr[valid].astype(np.float32) / 1000.0
    f[arr == _POLE_HOLE] = 1.0
    return f


def download_seaice_db(cache_dir=None, *, years=None, base_url: str = _BASE_URL,
                       timeout=120.0, verbose=False):
    """Build the monthly sea-ice climatology and cache it.

    Averages the NSIDC monthly concentration grids over ``years`` (default: the
    five most recent complete calendar years) per hemisphere and calendar month,
    writing ``<cache>/seaice/seaice_climatology.npz`` — one ``(12, H, W)``
    float32 array per hemisphere, under the keys ``'N'`` and ``'S'``. Missing
    months are skipped.

    ``base_url`` is the address the per-month GeoTIFF paths hang off (default
    the NSIDC G02135 tree), so a mirror that keeps NSIDC's own
    ``<hemisphere>/monthly/geotiff/<MM_Mon>/`` layout builds the same
    climatology.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory to write into; ``None`` is the dataset's own directory under
        the cache root (:func:`dataset_root`).
    years : iterable of int, optional
        Calendar years to average; ``None`` is the five most recent complete
        years.
    base_url : str, optional
        The address the per-month GeoTIFF paths hang off. Default the NSIDC
        G02135 tree.
    timeout : float, optional
        Network timeout in seconds. Default 120.
    verbose : bool or str, optional
        Logging gate passed through to ``log_message``.
    """
    import tifffile
    if years is None:
        end = _dt.date.today().year - 1
        years = range(end - 4, end + 1)
    years = list(years)
    # The banner names the year span, so the reference period is resolved
    # before the directory is announced.
    dest = _cache.prepare_download(
        'seaice', f"building NSIDC sea-ice climatology over {years[0]}-"
        f"{years[-1]} (~{len(years) * 24} monthly grids)",
        cache_dir=cache_dir, verbose=verbose)

    climo = {}
    for hemi in ('N', 'S'):
        stacks = [[] for _ in range(12)]
        for iy, year in enumerate(years):
            for m in range(1, 13):
                try:
                    blob = http_get(_monthly_url(hemi, year, m, base_url),
                                    timeout=timeout, verbose=False,
                                    source='seaice')
                    arr = tifffile.imread(io.BytesIO(blob))
                except Exception:               # noqa: BLE001 — skip missing
                    continue
                stacks[m - 1].append(_to_fraction(arr))
            # An entire year sweep with nothing fetched (12 of 12 failures)
            # is the unreachable-server signature; the remaining years would
            # spend ~4 retries per grid reaching the same place, so the
            # build stops here and says so.
            if iy == 0 and not any(stacks):
                raise DataFetchError(
                    f"No NSIDC sea-ice grids for hemisphere {hemi} in the "
                    f"whole {year} sweep (12 of 12 fetches failed); "
                    f"stopping before the remaining {len(years) - 1} "
                    f"year(s).",
                    remediation="Retry when noaadata.apps.nsidc.org "
                                "answers, or pass a `years` range that "
                                "starts on an available year.",
                )
        # The download loop above skips a grid it cannot fetch or decode, so a
        # month can be averaged over fewer years than were asked for. Only a
        # month with nothing at all raises; a partial month is a valid but
        # thinner climatology, and staying silent about it would present a
        # one-year mean as the multi-year mean the caller requested.
        short = [(m + 1, len(stacks[m])) for m in range(12)
                 if 0 < len(stacks[m]) < len(years)]
        if short:
            detail = ', '.join(f"{m}: {n}/{len(years)}" for m, n in short)
            warnings.warn(
                f"download_seaice_db: hemisphere {hemi} averaged "
                f"{len(short)} of 12 months over fewer years than requested "
                f"({detail}) — grids that could not be fetched or decoded "
                f"were skipped. The climatology is usable but thinner than "
                f"{years[0]}-{years[-1]} implies.",
                IOWarning, skip_file_prefixes=USER_FRAME_SKIP)
        months = []
        for m in range(12):
            if not stacks[m]:
                raise DataFetchError(
                    f"No NSIDC sea-ice grids found for hemisphere {hemi}, "
                    f"month {m + 1} over {years}.",
                    remediation="Retry, or pass a different `years` range.",
                )
            # Land and coast cells are NaN in every monthly grid (the
            # pole hole is not: it is filled at 1.0, see _to_fraction), so
            # their climatology is NaN; numpy reports that case through
            # warnings.warn ("Mean of empty slice"), which np.errstate does
            # not cover, so both channels are silenced here.
            with np.errstate(invalid='ignore'), warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', message='Mean of empty slice',
                    category=RuntimeWarning)
                months.append(np.nanmean(np.stack(stacks[m]), axis=0))
        climo[hemi] = np.stack(months).astype(np.float32)    # (12, H, W)
        log_message('seaice', f"hemisphere {hemi}: {climo[hemi].shape}",
                    verbose=verbose)

    out = dest / INDEX_FILE
    with _cache.atomic_write(out) as part:
        # A file object, not the path: np.savez_compressed appends '.npz' to a
        # name that lacks it, and the staging name does not end in '.npz'.
        with open(part, 'wb') as fh:
            # ``years`` records the reference period. The default is derived
            # from date.today() at BUILD time, so without it the vintage of a
            # cache differs per build and cannot be recovered from the file.
            np.savez_compressed(fh, years=np.asarray(years, dtype=np.int32),
                                **climo)
    _cache.invalidate_grids()
    log_message('seaice', f"sea-ice climatology cached → {out}", verbose=verbose)
    return out


def _pyproj_transformer(epsg, *, backend='sea-ice'):
    """EPSG:4326 → ``epsg`` transformer; ``backend`` names the caller in the
    missing-pyproj error. Also serves :mod:`uacpy.data.diesing_local`'s
    Wagner IV projection."""
    try:
        from pyproj import Transformer
    except ImportError as exc:                                  # pragma: no cover
        raise ConfigurationError(
            f"The offline {backend} backend needs 'pyproj'.",
            remediation="pyproj ships with the default uacpy install; reinstall "
                        "with `pip install -e .`, or `pip install pyproj`.",
        ) from exc
    return Transformer.from_crs("EPSG:4326", epsg, always_xy=True)


def _build_model():
    """Read the cached climatology and pair it with the two projections."""
    path = _cache.require_npz('seaice', INDEX_FILE, RETIRED_INDEX_FILE)
    with _cache.reading('seaice', path):
        # allow_pickle=False is passed rather than left to numpy's default:
        # this file is read straight out of the cache directory, and under
        # allow_pickle an object array in it would execute on load.
        with np.load(path, allow_pickle=False) as z:
            climo = {h: z[h] for h in ('N', 'S')}
            # Optional: caches built before the key existed, and the synthetic
            # ones the tests write, carry no reference period. Absent is not an
            # error — it just leaves the vintage unstated.
            years = ([int(y) for y in z['years']]
                     if 'years' in z.files else None)
    result = {'tf': {h: _pyproj_transformer(_GRID[h]['epsg']) for h in ('N', 'S')}}
    result.update(climo)
    result['years'] = years
    return result


def climatology_period():
    """Reference period of the installed climatology, or ``None``.

    A string such as ``'2019-2023 (climatology)'`` for the provenance
    ``data_date``. ``None`` where the cache predates the ``years`` key or was
    written without one — the grids are still usable, their vintage is simply
    not recorded.
    """
    years = _model().get('years')
    if not years:
        return None
    return f"{min(years)}-{max(years)} (climatology)"


@_cache.per_root_memo
def _model():
    """Load (or reuse) the sea-ice climatology and its projections.

    Built through :func:`uacpy.data._cache.per_root_memo`, so threads racing a cold
    memo load the file once between them rather than once each.
    """
    return _build_model()


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


def _observed_at(grid, row, col, rank=None):
    """``(concentration, cell)`` at ``(row, col)`` or its nearest observed
    neighbour.

    ``cell`` is the ``(row, col)`` the value actually came from, so the caller
    can tell a substitution from a direct hit. ``rank(row, col)`` orders the
    candidate cells — the projected distance from the requested point, from
    :func:`_concentration`. Returns ``(NaN, None)`` when no
    cell within :data:`_OBSERVED_CELL_SEARCH_RINGS` carries a value, which is
    the signature of genuine land.
    """
    value = grid[row, col]
    if np.isfinite(value):
        return float(value), (row, col)
    height, width = grid.shape
    cells = [(row + dr, col + dc)
             for radius in range(1, _OBSERVED_CELL_SEARCH_RINGS + 1)
             for dr, dc in ring_offsets(radius)
             if 0 <= row + dr < height and 0 <= col + dc < width]
    # Nearest to the REQUEST point when the caller says where that is; ring
    # order otherwise (its ties break in file order).
    if rank is not None:
        cells.sort(key=lambda rc: rank(*rc))
    for r, c in cells:
        candidate = grid[r, c]
        if np.isfinite(candidate):
            return float(candidate), (r, c)
    return float('nan'), None


def _cell_center(model, hemi, row, col):
    """``(lat, lon)`` of the centre of grid cell ``(row, col)``.

    ``x0``/``y0`` anchor the outer corner of cell (0, 0), so the centre sits
    half a pixel inside it; rows count southward as y decreases.
    """
    g = _GRID[hemi]
    x = g['x0'] + (col + 0.5) * g['px']
    y = g['y0'] - (row + 0.5) * g['px']
    lon, lat = model['tf'][hemi].transform(x, y, direction='INVERSE')
    return float(lat), float(lon)


def _cell_half_diagonal_km(model, hemi, row, col):
    """Ground distance (km) from the centre of cell ``(row, col)`` to its
    farthest corner: the farthest a point inside the cell stands from the
    centre. The 25 km cell is 25 km on the ground only at the projection's
    true-scale latitude (70°), so the corners are unprojected rather than
    taking half of hypot(25, 25)."""
    g = _GRID[hemi]
    lat_c, lon_c = _cell_center(model, hemi, row, col)
    corners = [model['tf'][hemi].transform(g['x0'] + (col + dc) * g['px'],
                                           g['y0'] - (row + dr) * g['px'],
                                           direction='INVERSE')
               for dr in (0, 1) for dc in (0, 1)]
    return max(float(great_circle_km(lat_c, lon_c, la, lo))
               for lo, la in corners)


def _concentration(lat, lon, month):
    """``(concentration, (lat, lon) of the cell read, half-diagonal km of
    the point's own cell)`` at a point: the cell holding it, or its nearest
    observed neighbour; ``(0.0, None, None)`` outside the polar grids, where
    no cell is read."""
    m = _model()
    hemi = 'N' if lat >= 0 else 'S'
    rc = _rowcol(m, hemi, lat, lon)
    if rc is None:
        return 0.0, None, None                  # outside the polar grid → ice-free
    # Squared projected distance from the requested point to a cell centre,
    # so the substitute is the observed cell nearest the REQUEST, not the
    # first in ring order.
    g = _GRID[hemi]
    x, y = m['tf'][hemi].transform(normalize_lon(lon), lat)

    def rank(r, c):
        return ((g['x0'] + (c + 0.5) * g['px'] - x) ** 2
                + (g['y0'] - (r + 0.5) * g['px'] - y) ** 2)

    value, cell = _observed_at(m[hemi][month - 1], *rc, rank=rank)
    # The cell the value came from is the data point: a substitute for an
    # unobserved cell (up to _OBSERVED_CELL_SEARCH_RINGS cells, 50 km) stands
    # past the own cell's half-diagonal, which the offset rule reports.
    if cell is None:
        return value, None, None
    return (value, _cell_center(m, hemi, *cell),
            _cell_half_diagonal_km(m, hemi, *rc))


def _sea_ice_reading(point, date, month, who):
    """``(concentration, 'seaice' DataProvenance with the cell read, the
    offset (km) past which the cell read is not the point's own)`` at a
    point, before the offset rule; raises DataFetchError at an inland point."""
    lat, lon = as_coordinate(point)
    if date is not None and month is not None:
        raise ConfigurationError(
            f"{who}: pass either date= or month=, not both.")
    if date is not None:
        month = parse_date(date).month
    if month is None:
        raise ConfigurationError(
            f"{who}: a date= or month= (1-12) is required.")
    conc, cell, own_km = _concentration(lat, lon, require_month(month, who))
    if not np.isfinite(conc):
        raise DataFetchError(
            f"NSIDC sea ice has no ocean value at {lat:.3f}, {lon:.3f}, nor "
            f"within {_OBSERVED_CELL_SEARCH_RINGS} grid cells of it — the point "
            f"is inland.",
            remediation="Pick an offshore point.",
        )
    return (float(conc),
            dataclasses.replace(_provenance((lat, lon)), data_point=cell,
                                point_kind='cell'),
            0.0 if own_km is None else own_km)


def _offset_checked(prov, warn_km, *, who, max_distance_km):
    """The offset rule on an NSIDC reading: a ProvenanceWarning past the
    point's own cell's half-diagonal ``warn_km``, a refusal past
    ``max_distance_km``."""
    return checked_offset(prov, who=who, warn_km=warn_km,
                          max_distance_km=max_distance_km)


def sea_ice_at(point: Coordinate, *, date=None, month: Optional[int] = None,
               max_distance_km: Optional[float] = None,
               who: str = 'fetch_sea_ice_concentration'):
    """``(concentration, the 'seaice' DataProvenance of the cell read)``,
    through the offset rule: a warning when an unobserved cell took its
    nearest observed neighbour's value, a refusal past ``max_distance_km``.
    Raises as :func:`fetch_sea_ice_concentration` does."""
    conc, prov, warn_km = _sea_ice_reading(point, date, month, who)
    return conc, _offset_checked(prov, warn_km, who=who,
                                 max_distance_km=max_distance_km)


def fetch_sea_ice_concentration(point: Coordinate, *, date=None,
                                month: Optional[int] = None,
                                max_distance_km: Optional[float] = None) -> float:
    """Climatological sea-ice concentration (0-1) at ``(lat, lon)`` for a month.

    Pass ``date`` (its month is used) or ``month`` (1-12). Points outside the
    polar grids return 0.0 (ice-free). A cell NSIDC leaves unobserved because of
    coastal land spillover takes its nearest observed ocean neighbour's value,
    with the offset rule's ``ProvenanceWarning`` naming that cell and the km
    (``max_distance_km`` refuses it instead); a point with no observed cell within
    :data:`_OBSERVED_CELL_SEARCH_RINGS` is inland and raises ``DataFetchError``.

    The value is raw and carries no provenance;
    :func:`fetch_sea_ice_surface` returns the carrier that records it in
    ``.data_sources``.

    Parameters
    ----------
    point : (lat, lon)
        Site coordinates in decimal degrees.
    date : str or datetime.date, optional
        A date whose month selects the climatology.
    month : int, optional
        The month, 1-12. Pass ``date`` or ``month``, not both.
    max_distance_km : float, optional
        Refuse a cell standing farther than this (km) from ``point``;
        ``None`` (default) sets no limit beyond the warning.
    """
    return sea_ice_at(point, date=date, month=month,
                      max_distance_km=checked_max_distance(
                          max_distance_km, 'fetch_sea_ice_concentration'))[0]


#: Representative RMS roughness (m) of the underside of Arctic pack ice, for
#: callers who want the canopy scattering rather than the smooth plate. The
#: under-ice relief is dominated by deformation features — ridge keels reaching
#: tens of metres over a much flatter undeformed surface — so no single number
#: describes it; a metre is the order the pack-ice literature reports for the
#: RMS of the undeformed-plus-ridged surface, and it is offered as a starting
#: point to vary, not a measurement of any particular ice cover.
SEA_ICE_TYPICAL_ROUGHNESS_M = 1.0


def sea_ice_surface(
    concentration: float, *,
    threshold: float = SEA_ICE_EDGE_CONCENTRATION,
    roughness: float = 0.0,
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

    ``roughness`` is the RMS interface roughness (m), 0 by default — the
    smooth homogeneous plate the source above tabulates these parameters for.
    **That plate is known to under-predict the loss**: Computational Ocean
    Acoustics §4, on this exact 0.4 / 1.0 dB/λ environment, records that "it
    has been demonstrated that the loss computed for such an environment is too
    low ... the ice cover is extremely inhomogeneous ... characterized by
    significant roughness as well as many discrete features such as ridges with
    keels". The default is kept at 0 so the returned properties stay the
    published ones; pass :data:`SEA_ICE_TYPICAL_ROUGHNESS_M` (or a site value)
    to have the solvers scatter off the canopy.

    A non-finite concentration (``NaN`` land/coast/out-of-grid cell) is treated
    as open water and returns ``None`` — never silently as ice, since
    ``NaN < threshold`` is False.

    Parameters
    ----------
    concentration : float
        Ice concentration, 0-1.
    threshold : float, optional
        Lowest concentration taken as ice cover, 0-1. Default 0.15, the NSIDC
        ice edge.
    roughness : float, optional
        RMS roughness (m) of the ice underside. Default 0.
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
        roughness=float(roughness),
    )


def fetch_sea_ice_surface(
    point: Coordinate, *, date=None, month: Optional[int] = None,
    threshold: float = SEA_ICE_EDGE_CONCENTRATION,
    roughness: float = 0.0,
    max_distance_km: Optional[float] = None,
) -> Optional[BoundaryProperties]:
    """Fetch the climatological ice concentration and convert it to a surface.

    Combines :func:`fetch_sea_ice_concentration` and :func:`sea_ice_surface`:
    returns the elastic ice ``BoundaryProperties`` where the point is
    ice-covered (concentration ≥ ``threshold``) for the given month, or ``None``
    for open water. ``roughness`` passes through to :func:`sea_ice_surface`,
    whose docstring records why the 0 default under-predicts the loss. Used by
    ``fetch_environment(surface_sources='seaice')``.

    Parameters
    ----------
    point : (lat, lon)
        Site coordinates in decimal degrees.
    date : str or datetime.date, optional
        A date whose month selects the climatology.
    month : int, optional
        The month, 1-12. Pass ``date`` or ``month``, not both.
    threshold : float, optional
        Lowest concentration taken as ice cover, 0-1. Default 0.15, the NSIDC
        ice edge.
    roughness : float, optional
        RMS roughness (m) of the ice underside. Default 0.
    max_distance_km : float, optional
        As in :func:`fetch_sea_ice_concentration`.
    """
    conc, prov = sea_ice_at(point, date=date, month=month,
                            max_distance_km=checked_max_distance(
                                max_distance_km, 'fetch_sea_ice_surface'),
                            who='fetch_sea_ice_surface')
    surface = sea_ice_surface(conc, threshold=threshold, roughness=roughness)
    if surface is not None:
        surface.data_sources = (prov,)
    return surface


def _provenance(requested_point=None):
    """The ``seaice`` provenance record: the installed climatology's
    reference period and, for a point fetch, the requested point."""
    try:
        period = climatology_period()
    except (ConfigurationError, DataFetchError):
        period = None              # no readable cache: the vintage is unstated
    return DataProvenance(source=SOURCES['seaice'], data_date=period,
                          requested_point=requested_point)


def sea_ice_grid(month: int, *, hemi: str = 'N') -> np.ndarray:
    """Monthly climatology concentration grid (0-1, ``NaN`` = land) for mapping.

    ``hemi`` is ``'N'`` / ``'S'``; the array is on the NSIDC polar-stereographic
    grid (North EPSG:3411, South EPSG:3412, 25 km).

    Parameters
    ----------
    month : int
        The month, 1-12.
    hemi : {'N', 'S'}, optional
        Hemisphere. Default ``'N'``.
    """
    month = require_month(month, 'sea_ice_grid')
    if hemi not in _GRID:
        raise ConfigurationError(
            f"sea_ice_grid: hemi must be 'N'/'S'; got {hemi!r}.")
    return _model()[hemi][month - 1]


def sea_ice_pixel(point: Coordinate, *, hemi: str = 'N'):
    """``(row, col)`` of a point in the NSIDC polar grid, or ``None`` if outside.

    Companion to :func:`sea_ice_grid` for overlaying markers on the grid: it
    shares its cell arithmetic with the value lookup, so a marker lands on
    exactly the cell whose concentration
    :func:`fetch_sea_ice_concentration` reads.

    Parameters
    ----------
    point : (lat, lon)
        Site coordinates in decimal degrees.
    hemi : {'N', 'S'}, optional
        Hemisphere grid. Default ``'N'``.
    """
    lat, lon = as_coordinate(point)
    return _rowcol(_model(), hemi, lat, lon)


@one_provenance_notice(subject='the samples',
                       record="the result's .provenance")
def fetch_sea_ice_concentration_transect(start: Coordinate, end: Coordinate, *,
                                         date=None, month: Optional[int] = None,
                                         n_points: int = 6,
                                         max_distance_km: Optional[float] = None,
                                         ) -> AlongTrack:
    """Sea-ice concentration (0-1) sampled along ``start`` → ``end``, as an
    :class:`~uacpy.data.AlongTrack` (``'sea_ice_concentration'``) with the
    ``'seaice'`` provenance; ``NaN`` at a land waypoint.

    Parameters
    ----------
    start, end : (lat, lon)
        Transect endpoints in decimal degrees.
    date : str or datetime.date, optional
        A date whose month selects the climatology.
    month : int, optional
        The month, 1-12. Pass ``date`` or ``month``, not both.
    n_points : int, optional
        Waypoints along the great circle. Default 6.
    max_distance_km : float, optional
        Refuse a waypoint whose cell stands farther than this (km) from it;
        ``None`` (default) sets no limit beyond the per-waypoint warning.
    """
    who = 'fetch_sea_ice_concentration_transect'
    n_points = checked_n_points(n_points, who)
    limit = checked_max_distance(max_distance_km, who)
    lats, lons, ranges_m = geodesic_waypoints(start, end, n_points)
    out = []
    for la, lo in zip(lats, lons):
        try:
            conc, prov, warn_km = _sea_ice_reading((la, lo), date, month, who)
        except DataFetchError:
            out.append(np.nan)                  # land along the transect
            continue
        # Outside the try: a refusal past max_distance_km is not land.
        _offset_checked(prov, warn_km, who=who, max_distance_km=limit)
        out.append(conc)
    return AlongTrack(ranges=np.asarray(ranges_m), lats=np.asarray(lats),
                      lons=np.asarray(lons), data=np.asarray(out),
                      unit='1', quantity='sea_ice_concentration',
                      provenance=_provenance())


@one_provenance_notice(subject="the ice canopy's data",
                       record='uacpy.data.citations(surface)')
def sea_ice_surface_transect(start: Coordinate, end: Coordinate, *,
                             date=None, month: Optional[int] = None,
                             n_points='auto', max_points=None,
                             threshold: float = SEA_ICE_EDGE_CONCENTRATION,
                             roughness: float = 0.0,
                             max_distance_km: Optional[float] = None):
    """Range-dependent ice surface along ``start`` → ``end`` as a ``Surface``.

    Each waypoint becomes the elastic ice canopy where the concentration is
    ≥ ``threshold`` (see :func:`sea_ice_surface`), else open water (a vacuum
    boundary). The resulting :class:`~uacpy.core.surface.Surface` carries the
    marginal ice zone (open water → pack → open water) for inspection and
    plotting. The propagation solvers all carry a single global top boundary,
    so every model collapses a range-dependent surface to one boundary (with a
    ``FallbackWarning``); use the carrier to study / visualise the zone.

    With ``n_points='auto'`` (default) the transect is probed at ``max_points``
    points (cheap — the NSIDC climatology is a local cached grid) and each run
    of identical zones (ice canopy vs open water) collapses to the probe
    samples bracketing its edges, endpoints anchored — the ``Surface`` reads
    nearest-node, so every reconstructed ice edge lands within one probe step
    of the edge the probe observed, without an oversampled staircase. An
    integer samples exactly that many evenly-spaced waypoints. ``roughness``
    passes through to :func:`sea_ice_surface` for every ice node.

    A waypoint where the climatology has no value (land / unobserved along
    the track) becomes an open-water node, and one ``FallbackWarning`` per call
    reports how many waypoints were classified that way without a
    measurement.

    Parameters
    ----------
    start, end : (lat, lon)
        Transect endpoints in decimal degrees.
    date : str or datetime.date, optional
        A date whose month selects the climatology.
    month : int, optional
        The month, 1-12. Pass ``date`` or ``month``, not both.
    n_points : int or 'auto', optional
        Waypoints, or ``'auto'`` for the zone-edge sampling described below.
        Default ``'auto'``.
    max_points : int, optional
        Probe count of ``'auto'`` and cap on an explicit count; ``None`` is
        :data:`~uacpy.data._geo.DEFAULT_MAX_TRANSECT_POINTS`.
    threshold : float, optional
        Lowest concentration taken as ice cover, 0-1. Default 0.15, the NSIDC
        ice edge.
    roughness : float, optional
        RMS roughness (m) of the ice underside. Default 0.
    max_distance_km : float, optional
        As in :func:`fetch_sea_ice_concentration_transect`.
    """
    from uacpy.core.surface import Surface
    if max_points is None:
        max_points = DEFAULT_MAX_TRANSECT_POINTS
    max_points = checked_max_points(max_points, 'sea_ice_surface_transect')
    n_points = checked_n_points(n_points, 'sea_ice_surface_transect',
                                allow_auto=True)
    probe_n = (max_points if n_points == 'auto'
               else capped_n_points(n_points, max_points,
                                    'sea_ice_surface_transect'))
    track = fetch_sea_ice_concentration_transect(
        start, end, date=date, month=month, n_points=probe_n,
        max_distance_km=max_distance_km)
    ranges_m, conc = track.ranges, track.data
    n_no_data = int(np.count_nonzero(~np.isfinite(np.asarray(conc, float))))
    if n_no_data:
        warnings.warn(
            f"sea_ice_surface_transect: {n_no_data} of {len(conc)} waypoints "
            f"have no NSIDC concentration (land / unobserved cells) and are "
            f"classified as open water — no-data nodes, not measured "
            f"open water.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
    prov = (_provenance(),)
    nodes = []
    for r, c in zip(ranges_m, conc):
        c = 0.0 if not np.isfinite(c) else float(c)
        bp = sea_ice_surface(c, threshold=threshold, roughness=roughness) \
            or BoundaryProperties(acoustic_type='vacuum')
        bp.data_sources = prov
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
