"""EMODnet seabed-substrate fetch (European seas) → ``BoundaryProperties``.

Audit follow-up to :mod:`uacpy.data.sediment`. There is no clean *global*
no-auth point service for seabed geoacoustics — the canonical reference
(WOSS) bundles the global **DECK41** sample database and looks it up locally.
EMODnet Geology, however, serves harmonised seabed substrate (Folk
classification) for **European seas** through a public OGC **WFS**, which *is*
lat/lon-queryable. This module turns that into a model-ready bottom.

Coverage is regional: outside the European-seas footprint the fetch raises
``DataFetchError`` and the caller should supply an explicit grain size (ϕ) or
sediment class instead (see :func:`uacpy.data.bottom_from_grain_size`).

The Folk 5-class (EUNIS) categories are mapped to representative grain sizes /
materials, then converted with the calibrated relations in
:mod:`uacpy.data.sediment`.
"""

import json
import math
import urllib.parse
from typing import Dict, Union

from uacpy.core.environment import BoundaryProperties, Bottom
from uacpy.core.exceptions import DataFetchError
from uacpy.data._geo import Coordinate, as_coordinate, normalize_lon
from uacpy.data._http import http_get
from uacpy.data.sediment import (
    bottom_from_class, bottom_from_grain_size, range_dependent_bottom_along,
)
from uacpy._log import log_message

__all__ = ['fetch_seabed_substrate', 'fetch_bottom', 'fetch_bottom_transect']

EMODNET_WFS_URL = 'https://drive.emodnet-geology.eu/geoserver/wfs'
EMODNET_LAYER = 'gtk:seabed_substrate_1m'

# EMODnet Folk 5-class code → representative geoacoustic handle.
# ('phi', ϕ) routes through the grain-size relation; ('class', name) uses a
# material preset (for hard substrata that grain size cannot describe).
_FOLK5_TO_BOTTOM = {
    1: ('phi', 5.0),     # Mud to muddy Sand
    2: ('phi', 2.0),     # Sand
    3: ('phi', -1.0),    # Coarse-grained sediment
    4: ('phi', 3.0),     # Mixed sediment
    5: ('class', 'limestone'),  # Rock or other hard substrata
}


def _to_web_mercator(lat: float, lon: float):
    """(lat, lon) degrees → (x, y) metres in EPSG:3857 (the EMODnet CRS)."""
    r = 6378137.0
    lon = normalize_lon(lon)
    return r * math.radians(lon), r * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def fetch_seabed_substrate(
    point: Coordinate,
    *,
    layer: str = EMODNET_LAYER,
    base_url: str = EMODNET_WFS_URL,
    timeout: float = 60.0,
    verbose: Union[bool, str] = False,
) -> Dict:
    """Raw EMODnet seabed-substrate record at a ``(lat, lon)`` point.

    Returns ``{'folk_5cl', 'folk_5cl_txt', 'original_grain_size', 'source'}``.

    Raises
    ------
    DataFetchError
        Service failure, or no coverage at the location (European seas only).
    """
    lat, lon = as_coordinate(point)
    x, y = _to_web_mercator(lat, lon)
    query = urllib.parse.urlencode({
        'service': 'WFS', 'version': '2.0.0', 'request': 'GetFeature',
        'typeNames': layer, 'outputFormat': 'application/json', 'count': '1',
        'CQL_FILTER': f"INTERSECTS(geom,POINT({x:.1f} {y:.1f}))",
    })
    body = http_get(f"{base_url}?{query}", timeout=timeout, verbose=verbose,
                    source='seabed')
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise DataFetchError(
            f"EMODnet WFS returned a non-JSON body: {exc}.",
        ) from exc

    features = data.get('features') or []
    if not features:
        raise DataFetchError(
            f"EMODnet has no seabed substrate at {lat:.3f}, {lon:.3f} "
            "(coverage is European seas only).",
            remediation="Outside European seas, pass an explicit grain size "
                        "(ϕ) or sediment class as the bottom.",
        )
    p = features[0]['properties']
    return {
        'folk_5cl': int(p['folk_5cl']),
        'folk_5cl_txt': p.get('folk_5cl_txt'),
        'original_grain_size': p.get('original_grain_size'),
        'source': 'EMODnet Geology seabed substrate 1:1M',
    }


def fetch_bottom(
    point: Coordinate,
    *,
    roughness: float = 0.0,
    water_sound_speed: float = None,
    layer: str = EMODNET_LAYER,
    base_url: str = EMODNET_WFS_URL,
    timeout: float = 60.0,
    verbose: Union[bool, str] = False,
) -> BoundaryProperties:
    """Model-ready bottom from EMODnet seabed substrate at a ``(lat, lon)`` point.

    Convenience wrapper: :func:`fetch_seabed_substrate` → Folk-class mapping →
    :func:`uacpy.data.bottom_from_grain_size` / ``bottom_from_class``. Raises
    ``DataFetchError`` outside European-seas coverage. ``water_sound_speed``
    (m/s) scales the grain-size velocity ratio to the in-situ near-seabed
    water; ``None`` uses the Hamilton reference (class bottoms are absolute and
    unaffected).
    """
    lat, lon = as_coordinate(point)
    sub = fetch_seabed_substrate(point, layer=layer, base_url=base_url,
                                 timeout=timeout, verbose=verbose)
    kind, value = _FOLK5_TO_BOTTOM.get(sub['folk_5cl'], ('phi', 3.0))
    if kind == 'phi':
        bottom = bottom_from_grain_size(
            value, roughness=roughness, water_sound_speed=water_sound_speed)
    else:
        bottom = bottom_from_class(value, roughness=roughness)
    log_message(
        'seabed', f"EMODnet '{sub['folk_5cl_txt']}' at {lat:.3f}, {lon:.3f} → "
        f"{bottom.acoustic_type} c_p={bottom.sound_speed:.0f} m/s",
        verbose=verbose,
    )
    return bottom


def fetch_bottom_transect(
    start: Coordinate, end: Coordinate, *,
    n_points: int = 6,
    roughness: float = 0.0,
    water_sound_speed: float = None,
    layer: str = EMODNET_LAYER,
    base_url: str = EMODNET_WFS_URL,
    timeout: float = 60.0,
    verbose: Union[bool, str] = False,
) -> Bottom:
    """Range-dependent bottom from EMODnet sampled along ``start`` → ``end``.

    Queries EMODnet at ``n_points`` evenly-spaced points along the great-circle
    path and assembles a :class:`~uacpy.core.environment.Bottom`
    (explicit ``c_p`` / ρ / α arrays vs range, from the Folk-class mapping), with
    ranges measured from ``start`` — the seafloor analogue of
    :func:`uacpy.data.fetch_ssp_transect`.

    Points outside EMODnet coverage hold the nearest covered value; the call
    raises only if *no* point along the transect is covered.
    """
    return range_dependent_bottom_along(
        lambda la, lo: fetch_bottom((la, lo), roughness=roughness,
                                    water_sound_speed=water_sound_speed,
                                    layer=layer, base_url=base_url,
                                    timeout=timeout, verbose=verbose),
        start, end, n_points, source_label='EMODnet',
    )
