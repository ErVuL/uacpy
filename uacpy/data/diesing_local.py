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

import io
import zipfile
from pathlib import Path
from typing import Optional, Union

import numpy as np

from uacpy._log import log_message
from uacpy.core.environment import BoundaryProperties, Bottom
from uacpy.core.exceptions import DataFetchError
from uacpy.data import _cache
from uacpy.data._geo import Coordinate, as_coordinate
from uacpy.data._http import http_get, checked_member_size
from uacpy.data.sediment import (
    bottom_from_grain_size, range_dependent_bottom_along, water_sound_speed_at,
)

__all__ = ['download_diesing_db', 'fetch_seafloor_lithology',
           'fetch_bottom_diesing', 'fetch_bottom_diesing_transect']

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

_MODEL = {}   # cache_root -> dict(arr, x0, y0, sx, sy, tf, H, W)
_cache.register_cache(_MODEL.clear)


def _pyproj_transformer():
    """EPSG:4326 → Wagner IV transformer, through
    :func:`uacpy.data.seaice_local._pyproj_transformer` (which raises
    ``ConfigurationError`` naming this backend when pyproj is missing)."""
    from uacpy.data.seaice_local import _pyproj_transformer as epsg_transformer
    return epsg_transformer(WAGNER4_PROJ, backend='Diesing seafloor-lithology')


def download_diesing_db(cache_dir=None, *, timeout=300.0, verbose=False):
    """Download the Diesing 2020 lithology raster into the cache.

    Fetches the CC-BY PANGAEA package and extracts ``lithology_classes.tif`` into
    ``<cache>/diesing/``. Returns the written raster path.
    """
    dest = Path(cache_dir) if cache_dir else _cache.dataset_root('diesing')
    dest.mkdir(parents=True, exist_ok=True)
    log_message('diesing', "downloading Diesing 2020 seafloor lithology "
                "(CC-BY, ~40 MB)", verbose=verbose)
    blob = http_get(DIESING_URL, timeout=timeout, verbose=verbose,
                    source='diesing')
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        try:
            info = zf.getinfo(RASTER_FILE)
        except KeyError as exc:
            raise DataFetchError(
                "Diesing package did not contain the lithology raster.",
                remediation="Retry; the upstream package layout may have changed.",
            ) from exc
        checked_member_size(info.file_size, RASTER_FILE)
        data = zf.read(RASTER_FILE)
    out = dest / RASTER_FILE
    with _cache.atomic_write(out) as part:
        part.write_bytes(data)
    _MODEL.clear()
    log_message('diesing', f"Diesing lithology map cached → {out}", verbose=verbose)
    return out


def _build_model():
    """Read the raster and derive its geotransform and projection."""
    from PIL import Image
    path = _cache.require('diesing', RASTER_FILE)
    Image.MAX_IMAGE_PIXELS = None
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


def _model():
    """Load (or reuse) the raster, its geotransform and the projection.

    Built through :func:`uacpy.data._cache.memoize`, so threads racing a cold
    memo decode the raster once between them rather than once each.
    """
    return _cache.memoize(_MODEL, str(_cache.cache_root()), _build_model)


def _sample(m, x, y):
    """Class code at projected ``(x, y)``, or ``None`` where the raster has none."""
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
    return int(v)


def _class_code(lat, lon):
    """Diesing class code (1-5) at a point, or ``None`` outside deep-sea coverage."""
    m = _model()
    # No normalize_lon: PROJ wraps the longitude relative to +lon_0 itself, so a
    # query at 190° and one at −170° project to the same x.
    x, y = m['tf'].transform(lon, lat)
    code = _sample(m, x, y)
    if code is not None:
        return code
    # +180 and −180 are the same meridian but sit at opposite ends of the
    # parallel under Wagner IV, and the rasterized nodata margin covers one end
    # without covering the other: (0, 180) reads clay while (0, −180) reads as
    # uncovered. A miss within one pixel of the map edge is retried at the
    # mirrored end of the same parallel, so both spellings answer alike.
    x_edge = abs(m['tf'].transform(180.0, lat)[0])
    if abs(abs(x) - x_edge) <= m['sx']:
        return _sample(m, -x, y)
    return None


def fetch_seafloor_lithology(point: Coordinate) -> dict:
    """Diesing 2020 seafloor lithology at a ``(lat, lon)`` point.

    Returns ``{'lithology', 'grain_size_phi', 'source'}``. Raises
    ``DataFetchError`` outside deep-sea coverage (water shallower than 500 m, or
    land).
    """
    lat, lon = as_coordinate(point)
    code = _class_code(lat, lon)
    if code is None or code not in _CLASS:
        raise DataFetchError(
            f"Diesing has no seafloor lithology at {lat:.3f}, {lon:.3f} "
            "(deep-sea map; coverage is water deeper than 500 m).",
            remediation="On the shelf use bottom_sources='grainsize'/'emodnet', "
                        "or 'pelagic' for a global model.",
        )
    litho, phi = _CLASS[code]
    return {'lithology': litho, 'grain_size_phi': phi,
            'source': 'Diesing 2020 deep-sea seafloor lithology (CC-BY)'}


def fetch_bottom_diesing(point: Coordinate, *, roughness: float = 0.0,
                         water_sound_speed: Optional[float] = None,
                         timeout=None, verbose: Union[bool, str] = False
                         ) -> BoundaryProperties:
    """Model-ready bottom from the Diesing 2020 lithology at ``(lat, lon)``.

    Provenance is catalogue-level: the raster cell under the point supplies
    the value, and no per-cell ``data_point``/``offset_km`` is recorded —
    unlike the sample sources (``grainsize``, ``mars``), which record the
    sample the value came from.

    ``timeout`` is accepted (and ignored — this backend is offline) for signature
    parity with the network bottom fetchers. ``water_sound_speed`` (m/s) scales
    the grain-size velocity ratio to the in-situ near-seabed water; ``None``
    uses the Hamilton reference.
    """
    lat, lon = as_coordinate(point)
    sub = fetch_seafloor_lithology(point)
    bottom = bottom_from_grain_size(
        sub['grain_size_phi'], roughness=roughness,
        water_sound_speed=water_sound_speed)
    log_message(
        'diesing', f"Diesing {sub['lithology']} at {lat:.2f}, {lon:.2f} → "
        f"ϕ={sub['grain_size_phi']}", verbose=verbose)
    return bottom


def fetch_bottom_diesing_transect(start: Coordinate, end: Coordinate, *,
                                  n_points=6, max_points=None,
                                  roughness: float = 0.0,
                                  water_sound_speed: Optional[float] = None,
                                  timeout=None, verbose: Union[bool, str] = False
                                  ) -> Bottom:
    """Range-dependent bottom from the Diesing 2020 map along a transect.

    ``water_sound_speed`` also takes a ``(lat, lon) -> m/s`` callable,
    so each column scales to the water over its own seafloor.
    ``timeout``/``verbose`` are accepted (and ignored — this backend is
    offline) for signature parity with the network bottom fetchers.
    """
    return range_dependent_bottom_along(
        lambda la, lo: fetch_bottom_diesing(
            (la, lo), roughness=roughness,
            water_sound_speed=water_sound_speed_at(water_sound_speed, la, lo)),
        start, end, n_points, source_label='Diesing 2020',
        max_points=max_points,
    )
