"""Offline Diesing 2020 global deep-sea seafloor lithology → bottom.

``install.sh --data diesing`` downloads the **CC-BY 4.0** global deep-sea
seafloor-lithology map of Diesing (2020) — *Deep-sea sediments of the global
ocean*, Earth Syst. Sci. Data 12, 3367-3381, doi:10.5194/essd-12-3367-2020;
data: PANGAEA doi:10.1594/PANGAEA.911692. A Random-Forest map (5 classes:
calcareous sediment, clay, diatom ooze, lithogenous sediment, radiolarian ooze)
on a 10 km grid. This module reads the predicted-class raster and turns the
lithology at a point into a model-ready bottom (via the grain-size relations).

It is the **measured/modelled global surficial** seabed — the licence-clean
upgrade to the first-principles :mod:`uacpy.data.pelagic` rule. Coverage is the
**deep sea only (water depth > 500 m)**: on the shelf it returns no data, so a
caller's 'auto' bottom falls through (to grain-size / pelagic).

Reading the GeoTIFF (LZW-compressed, Wagner IV equal-area projection) needs
``pyproj`` (a default uacpy dependency) and Pillow (already a Matplotlib one).
"""

import dataclasses
from typing import Optional, Union

import numpy as np

from uacpy._log import log_message
from uacpy.core.environment import BoundaryProperties
from uacpy.core.exceptions import DataFetchError
from uacpy.data import _cache
from uacpy.core.geo import Coordinate, as_coordinate
from uacpy.data._geo import checked_max_distance, checked_offset
from uacpy.data._http import download_member
from uacpy.data.sources import SOURCES, DataProvenance
from uacpy.core.sediment import DEFAULT_GRAIN_SIZE_MODEL
from uacpy.data.sediment import SeabedSample, bottom_from_grain_size

__all__ = ['download_diesing_db', 'fetch_seafloor_lithology',
           'fetch_bottom_diesing']

DIESING_URL = ('https://store.pangaea.de/Publications/DiesingM_2020/'
               'Deep-sea_sediments_5_classes.zip')
RASTER_FILE = 'lithology_classes.tif'
# Wagner IV global equal-area projection the raster is georeferenced in.
WAGNER4_PROJ = ('+proj=wag4 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m '
                '+no_defs')
# Class code → (lithology name, representative mean grain size ϕ). The four
# biogenic classes carry the ϕ the local sediment DB's lithology map gives the
# same term (``sediment_db._DECK41_LITHOLOGY_TO_PHI``). 'Lithogenous' has no
# term there — it is terrigenous gravel/sand/silt rather than one Wentworth
# class — so it takes the Wentworth sand/silt boundary, coarser than that map's
# silt (5.5).
_CLASS = {
    1: ('calcareous sediment', 7.5),
    2: ('clay', 9.0),
    3: ('diatom ooze', 9.0),
    4: ('lithogenous sediment', 4.0),
    5: ('radiolarian ooze', 8.0),
}


def _pyproj_transformer():
    """EPSG:4326 → Wagner IV transformer, through
    :func:`uacpy.data.seaice_local._pyproj_transformer` (which raises
    ``ConfigurationError`` naming this backend when pyproj is missing)."""
    from uacpy.data.seaice_local import _pyproj_transformer as epsg_transformer
    return epsg_transformer(WAGNER4_PROJ, backend='Diesing seafloor-lithology')


def download_diesing_db(cache_dir=None, *, url: Optional[str] = None,
                        timeout=300.0, verbose=False):
    """Download the Diesing 2020 lithology raster into the cache.

    Fetches the CC-BY PANGAEA package and extracts ``lithology_classes.tif`` into
    ``<cache>/diesing/``. Returns the written raster path.

    ``url`` fetches that address instead of :data:`DIESING_URL` — a mirror,
    or a copy staged on an http server of your own. What is written and
    how it is read are the same whatever address served it.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory to write into; ``None`` is the dataset's own directory under
        the cache root (:func:`dataset_root`).
    url : str, optional
        The one address to fetch; ``None`` is :data:`DIESING_URL`.
    timeout : float, optional
        Network timeout in seconds. Default 300.
    verbose : bool or str, optional
        Logging gate passed through to ``log_message``.
    """
    dest = _cache.prepare_download(
        'diesing', "downloading Diesing 2020 seafloor lithology (CC-BY, ~40 MB)",
        cache_dir=cache_dir, verbose=verbose)
    out = download_member('diesing', url or DIESING_URL, RASTER_FILE,
                          dest / RASTER_FILE, timeout=timeout,
                          verbose=verbose)
    _cache.invalidate_grids()
    log_message('diesing', f"Diesing lithology map cached → {out}", verbose=verbose)
    return out


def _build_model():
    """Read the raster and derive its geotransform and projection."""
    from PIL import Image
    path = _cache.require('diesing', RASTER_FILE)
    # Opened under Pillow's decompression-bomb guard as the user has it set:
    # the raster is 3469 x 1739 = 6.0 M pixels, under the 89 M default.
    with _cache.reading('diesing', path):
        im = Image.open(path)
        arr = np.asarray(im, dtype=np.float32)
        tags = im.tag_v2
        sx, sy = float(tags[33550][0]), float(tags[33550][1])  # ModelPixelScale
        # ModelTiepoint is (i, j, k, x, y, z): raster point (i, j) maps to
        # model (x, y). Back out the corner of pixel (0, 0) — x grows with the
        # column, y shrinks with the row.
        tie = tags[33922]
        x0 = float(tie[3]) - float(tie[0]) * sx
        y0 = float(tie[4]) + float(tie[1]) * sy
    return {'arr': arr, 'x0': x0, 'y0': y0, 'sx': sx, 'sy': sy,
            'tf': _pyproj_transformer(), 'H': arr.shape[0], 'W': arr.shape[1]}


@_cache.per_root_memo
def _model():
    """Load (or reuse) the raster, its geotransform and the projection.

    Built through :func:`uacpy.data._cache.per_root_memo`, so threads racing a cold
    memo decode the raster once between them rather than once each.
    """
    return _build_model()


def _sample(m, x, y):
    """``(class code, (x, y) of the cell's centre)`` at projected ``(x, y)``,
    or ``None`` where the raster has none."""
    # The raster declares GTRasterTypeGeoKey = RasterPixelIsArea, so the
    # geotransform anchors the *corner* of pixel (0,0); the cell containing the
    # point is floor(), not round() (which would bias it half a pixel). y0 is
    # the northern edge, hence rows count southward.
    col = int(np.floor((x - m['x0']) / m['sx']))
    row = int(np.floor((m['y0'] - y) / m['sy']))
    if not (0 <= row < m['H'] and 0 <= col < m['W']):
        return None
    v = m['arr'][row, col]
    # Class codes start at 1, so ``v < 1`` rejects both an unclassified cell and
    # the raster's declared GDAL nodata sentinel of -3.4e38.
    if not np.isfinite(v) or v < 1:
        return None
    return int(v), (m['x0'] + (col + 0.5) * m['sx'], m['y0'] - (row + 0.5) * m['sy'])


def _class_code(lat, lon):
    """``(Diesing class code 1-5, (lat, lon) of the cell read)`` at a point,
    or ``None`` outside deep-sea coverage."""
    m = _model()
    # No normalize_lon: PROJ wraps the longitude relative to +lon_0 itself, so a
    # query at 190° and one at −170° project to the same x.
    x, y = m['tf'].transform(lon, lat)
    hit = _sample(m, x, y)
    if hit is not None:
        return _with_cell_point(m, hit)
    # +180 and −180 are the same meridian but sit at opposite ends of the
    # parallel under Wagner IV, and the rasterized nodata margin covers one end
    # without covering the other: (0, 180) reads clay while (0, −180) reads as
    # uncovered. A miss within one pixel of the map edge is retried at the
    # mirrored end of the same parallel, so both spellings answer alike.
    x_edge = abs(m['tf'].transform(180.0, lat)[0])
    if abs(abs(x) - x_edge) <= m['sx']:
        hit = _sample(m, -x, y)
        return None if hit is None else _with_cell_point(m, hit)
    return None


def _with_cell_point(m, hit):
    """``(code, (lat, lon))`` of a raster hit: its cell centre unprojected."""
    code, (xc, yc) = hit
    lon_c, lat_c = m['tf'].transform(xc, yc, direction='INVERSE')
    return code, (float(lat_c), float(lon_c))


def _cell_half_diagonal_km(m):
    """Half the diagonal (km) of one raster cell (an equal-area grid)."""
    return 0.5 * float(np.hypot(m['sx'], m['sy'])) / 1000.0


def fetch_seafloor_lithology(point: Coordinate, *,
                             max_distance_km: Optional[float] = None) -> SeabedSample:
    """Diesing 2020 seafloor lithology at a ``(lat, lon)`` point.

    Returns a :class:`~uacpy.data.SeabedSample`: the lithology as
    ``material``, its ``grain_size_phi``, and the ``'diesing'`` provenance
    with the requested point and the centre of the raster cell read (the
    offset rule: ``max_distance_km`` refuses a cell farther than that, km).
    Raises
    ``DataFetchError`` outside deep-sea coverage (water shallower than 500 m, or
    land).

    Parameters
    ----------
    point : (lat, lon)
        Site coordinates in decimal degrees.
    max_distance_km : float, optional
        Refuse a cell whose centre stands farther than this (km) from
        ``point``; ``None`` (default) sets no limit beyond the warning.
    """
    lat, lon = as_coordinate(point)
    limit = checked_max_distance(max_distance_km, 'fetch_seafloor_lithology')
    hit = _class_code(lat, lon)
    code, cell = (None, None) if hit is None else hit
    if code is None or code not in _CLASS:
        raise DataFetchError(
            f"Diesing has no seafloor lithology at {lat:.3f}, {lon:.3f} "
            "(deep-sea map; coverage is water deeper than 500 m).",
            remediation="On the shelf use bottom_sources='grainsize'/'emodnet', "
                        "or 'pelagic' for a global model.",
        )
    litho, phi = _CLASS[code]
    return SeabedSample(grain_size_phi=phi, material=litho, folk_class=None,
                        folk_class_scheme=None, sample_point=None,
                        distance_km=None,
                        provenance=checked_offset(
                            DataProvenance(source=SOURCES['diesing'],
                                           data_point=cell,
                                           requested_point=(lat, lon),
                                           point_kind='cell'),
                            who='fetch_seafloor_lithology',
                            warn_km=(0.0 if cell is None
                                     else _cell_half_diagonal_km(_model())),
                            max_distance_km=limit))


def fetch_bottom_diesing(point: Coordinate, *, roughness: float = 0.0,
                         water_sound_speed: Optional[float] = None,
                         model: str = DEFAULT_GRAIN_SIZE_MODEL,
                         hamilton_fit: Optional[str] = None,
                         max_distance_km: Optional[float] = None,
                         timeout=None, verbose: Union[bool, str] = False
                         ) -> BoundaryProperties:
    """Model-ready bottom from the Diesing 2020 lithology at ``(lat, lon)``.

    The returned bottom carries a ``diesing`` ``DataProvenance`` with the
    requested point and the raster cell read as ``data_point``;
    ``max_distance_km`` refuses a cell farther than that (km).

    ``timeout`` is accepted (and ignored — this backend is offline) for signature
    parity with the network bottom fetchers. ``water_sound_speed`` (m/s) scales
    the grain-size velocity ratio to the in-situ near-seabed water; ``None``
    uses the Hamilton reference. ``model`` picks the grain-size relations
    (``'hamilton'`` or ``'apl-uw'``).
    """
    from uacpy.core.sediment import canonical_grain_size_selection
    model, hamilton_fit = canonical_grain_size_selection(
        model, hamilton_fit, who='fetch_bottom_diesing')
    lat, lon = as_coordinate(point)
    sub = fetch_seafloor_lithology(point, max_distance_km=max_distance_km)
    bottom = bottom_from_grain_size(
        sub.grain_size_phi, roughness=roughness, model=model,
        hamilton_fit=hamilton_fit,
        water_sound_speed=water_sound_speed)
    log_message(
        'diesing', f"Diesing {sub.material} at {lat:.2f}, {lon:.2f} → "
        f"ϕ={sub.grain_size_phi}", verbose=verbose)
    return dataclasses.replace(bottom, data_sources=(sub.provenance,))
