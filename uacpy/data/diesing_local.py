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
from uacpy.core.exceptions import ConfigurationError, DataFetchError
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
# Class code → (lithology name, representative mean grain size ϕ). ϕ values match
# the local sediment DB's lithology→ϕ map; lithogenous (gravel/sand/silt) ≈ silt.
_CLASS = {
    1: ('calcareous sediment', 7.5),
    2: ('clay', 9.0),
    3: ('diatom ooze', 9.0),
    4: ('lithogenous sediment', 4.0),
    5: ('radiolarian ooze', 8.0),
}

_MODEL = {}   # cache_root -> dict(arr, x0, y0, sx, sy, tf, H, W)


def _pyproj_transformer():
    try:
        from pyproj import Transformer
    except ImportError as exc:                                  # pragma: no cover
        raise ConfigurationError(
            "The offline Diesing seafloor-lithology backend needs 'pyproj'.",
            remediation="pyproj ships with the default uacpy install; reinstall "
                        "with `pip install -e .`, or `pip install pyproj`.",
        ) from exc
    return Transformer.from_crs("EPSG:4326", WAGNER4_PROJ, always_xy=True)


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
    out.write_bytes(data)
    _MODEL.clear()
    log_message('diesing', f"Diesing lithology map cached → {out}", verbose=verbose)
    return out


def _model():
    """Load (or reuse) the raster, its geotransform and the projection."""
    root = str(_cache.cache_root())
    if root in _MODEL:
        return _MODEL[root]
    from PIL import Image
    path = _cache.require('diesing', RASTER_FILE)
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(path)
    arr = np.asarray(im, dtype=np.float32)
    tags = im.tag_v2
    sx, sy = float(tags[33550][0]), float(tags[33550][1])     # ModelPixelScale
    tie = tags[33922]                                         # ModelTiepoint
    x0 = float(tie[3]) - float(tie[0]) * sx
    y0 = float(tie[4]) + float(tie[1]) * sy
    result = {'arr': arr, 'x0': x0, 'y0': y0, 'sx': sx, 'sy': sy,
              'tf': _pyproj_transformer(), 'H': arr.shape[0], 'W': arr.shape[1]}
    _MODEL[root] = result
    return result


def _class_code(lat, lon):
    """Diesing class code (1-5) at a point, or ``None`` outside deep-sea coverage."""
    m = _model()
    x, y = m['tf'].transform(lon, lat)
    # The geotransform anchors the corner of pixel (0,0) (PixelIsArea), so the
    # cell containing the point is floor(), not round() (a half-pixel bias).
    col = int(np.floor((x - m['x0']) / m['sx']))
    row = int(np.floor((m['y0'] - y) / m['sy']))
    if not (0 <= row < m['H'] and 0 <= col < m['W']):
        return None
    v = m['arr'][row, col]
    if not np.isfinite(v) or v < 1:
        return None
    return int(v)


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
    """
    return range_dependent_bottom_along(
        lambda la, lo: fetch_bottom_diesing(
            (la, lo), roughness=roughness,
            water_sound_speed=water_sound_speed_at(water_sound_speed, la, lo)),
        start, end, n_points, source_label='Diesing 2020',
        max_points=max_points,
    )
