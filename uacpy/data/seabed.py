"""EMODnet seabed-substrate fetch (European seas) → ``BoundaryProperties``.

There is no clean *global* no-auth point service for seabed geoacoustics — the
canonical reference (WOSS) bundles the global **DECK41** sample database and
looks it up locally. EMODnet Geology, however, serves harmonised seabed
substrate (Folk classification) for **European seas** through a public OGC
**WFS**, which *is* lat/lon-queryable. This module turns that into a
model-ready bottom.

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
from typing import Dict, Optional, Union

from uacpy.core.environment import BoundaryProperties, Bottom
from uacpy.core.exceptions import DataFetchError
from uacpy.data._geo import Coordinate, as_coordinate, normalize_lon
from uacpy.data._http import http_get
from uacpy.data.sediment import (
    bottom_from_class, bottom_from_grain_size, range_dependent_bottom_along,
    water_sound_speed_at,
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
#: A sixth code the harmonised layer carries that the Folk 5-class legend does
#: not define. It is not an oddity of one response: the cached 1:1M layer holds
#: 70 such polygons — slivers in the North Sea, the Gulf of Finland and the
#: Caspian totalling 0.8 deg², against class 5's 10 622 polygons and 89 deg² —
#: so they read as gaps in the harmonisation rather than a substrate type.
#: There is nothing to convert, so a point inside one is refused the way a
#: point outside coverage is, and an 'auto' bottom chain falls through to the
#: global grain-size DB. Refused with its own message so a real-data gap is not
#: reported as a schema change.
_FOLK5_UNCLASSIFIED = 6


def _bottom_from_folk5(code, lat, lon, *, roughness, water_sound_speed=None):
    """``BoundaryProperties`` for one Folk 5-class code, or a typed refusal.

    Shared by the live WFS backend and the offline polygon backend
    (:mod:`uacpy.data.emodnet_local`) so both convert a class — and refuse one
    they cannot convert — identically.
    """
    if code not in _FOLK5_TO_BOTTOM:
        message = (
            f"The EMODnet polygon at {lat:.3f}, {lon:.3f} carries "
            f"folk_5cl={_FOLK5_UNCLASSIFIED}, a code the Folk 5-class legend "
            f"(1-5) does not define, so it names no substrate to convert."
            if code == _FOLK5_UNCLASSIFIED else
            f"EMODnet returned an unrecognised Folk-5 class {code!r} at "
            f"{lat:.3f}, {lon:.3f}; refusing to fabricate a default bottom.")
        raise DataFetchError(
            message,
            remediation="Pass an explicit grain size (ϕ) or sediment class, or "
                        "let the 'auto' bottom chain fall through to another "
                        "source.",
        )
    kind, value = _FOLK5_TO_BOTTOM[code]
    if kind == 'phi':
        return bottom_from_grain_size(value, roughness=roughness,
                                      water_sound_speed=water_sound_speed)
    return bottom_from_class(value, roughness=roughness)


#: EPSG:3857's own latitude limit: the Mercator ordinate diverges at the poles,
#: and the projection is defined only where |y| <= pi*R, i.e. within this many
#: degrees of the equator. ``as_coordinate`` admits the full +/-90, so a polar
#: request reached ``log(tan(...))`` with an argument of 0 at -90 (an untyped
#: ``ValueError: math domain error``) and produced y = 2.4e8 m at +90 -- an
#: ordinate twelve times the world extent, which the WFS answers with an empty
#: feature set rather than an error.
WEB_MERCATOR_MAX_LAT_DEG = 85.05112877980659


def _to_web_mercator(lat: float, lon: float):
    """(lat, lon) degrees → (x, y) metres in EPSG:3857 (the EMODnet CRS).

    Latitude is clamped to :data:`WEB_MERCATOR_MAX_LAT_DEG`, the projection's
    own limit. EMODnet's coverage is European seas, so no clamped request was
    ever going to return a substrate: the clamp keeps a polar point on the
    typed "no seabed substrate" path instead of a bare ``ValueError``.
    """
    # EPSG:3857 projects WGS84 coordinates onto a *sphere* of the WGS84
    # semi-major axis, so this single radius is the whole datum.
    radius_m = 6378137.0
    lon = normalize_lon(lon)
    lat = max(-WEB_MERCATOR_MAX_LAT_DEG, min(WEB_MERCATOR_MAX_LAT_DEG, lat))
    return (radius_m * math.radians(lon),
            radius_m * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)))


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
    water_sound_speed: Optional[float] = None,
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
    bottom = _bottom_from_folk5(sub['folk_5cl'], lat, lon, roughness=roughness,
                                water_sound_speed=water_sound_speed)
    log_message(
        'seabed', f"EMODnet '{sub['folk_5cl_txt']}' at {lat:.3f}, {lon:.3f} → "
        f"{bottom.acoustic_type} c_p={bottom.sound_speed:.0f} m/s",
        verbose=verbose,
    )
    return bottom


def fetch_bottom_transect(
    start: Coordinate, end: Coordinate, *,
    n_points=6,
    max_points=None,
    roughness: float = 0.0,
    water_sound_speed: Optional[float] = None,
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
    ``water_sound_speed`` also takes a ``(lat, lon) -> m/s`` callable, so each
    column scales to the water over its own seafloor.
    """
    return range_dependent_bottom_along(
        lambda la, lo: fetch_bottom((la, lo), roughness=roughness,
                                    water_sound_speed=water_sound_speed_at(
                                        water_sound_speed, la, lo),
                                    layer=layer, base_url=base_url,
                                    timeout=timeout, verbose=verbose),
        start, end, n_points, source_label='EMODnet', max_points=max_points,
    )
