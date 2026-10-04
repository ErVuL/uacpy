"""EMODnet seabed-substrate fetch (European seas) → ``BoundaryProperties``.

There is no clean *global* no-auth point service for seabed geoacoustics — the
canonical reference (WOSS) bundles the global **DECK41** sample database and
looks it up locally. EMODnet Geology, however, serves harmonised seabed
substrate (Folk classification) for **European seas** through a public OGC
**WFS**, which *is* lat/lon-queryable. This module turns that into a
model-ready bottom.

Coverage is regional: outside the European-seas footprint the fetch raises
``DataFetchError``; ``fetch_environment(bottom_sources='auto')`` then falls
through to the global sources (grain-size samples, Diesing, MARS, pelagic),
and a direct caller can pick one of those or supply an explicit grain size (ϕ)
or sediment class (see :func:`uacpy.data.bottom_from_grain_size`).

The Folk 5-class (EUNIS) categories are mapped to representative grain sizes /
materials, then converted with the calibrated relations in
:mod:`uacpy.data.sediment`.
"""

import dataclasses
import json
import math
import urllib.parse
from typing import Optional, Union

from uacpy.core.environment import BoundaryProperties
from uacpy.core.exceptions import DataFetchError
from uacpy.core.geo import Coordinate, as_coordinate, normalize_lon
from uacpy.data._geo import checked_max_distance, checked_offset
from uacpy.data._http import http_get
from uacpy.data.sources import SOURCES, DataProvenance
from uacpy.data.sediment import (SeabedSample, bottom_from_class,
                                 bottom_from_grain_size)
from uacpy._log import log_message
from uacpy.core.sediment import DEFAULT_GRAIN_SIZE_MODEL

__all__ = ['fetch_emodnet_substrate', 'fetch_bottom_emodnet']

EMODNET_WFS_URL = 'https://drive.emodnet-geology.eu/geoserver/wfs'
EMODNET_LAYER = 'gtk:seabed_substrate_1m'

# EMODnet Folk 5-class code → representative geoacoustic handle.
# ('phi', ϕ) routes through the grain-size relation; ('class', name) uses a
# material preset (for hard substrata that grain size cannot describe).
# A Folk class is a sediment classification, so the four sediment classes go
# through ϕ, as MARS converts its Folk classes (``uacpy.data.mars``); only a
# DECK41 lithology *word* such as 'gravel' takes a material preset (see
# :func:`uacpy.core.sediment.grain_size_to_geoacoustics`). Class 3 is placed at
# -1 ϕ, the sand/gravel boundary (Medwin & Clay Sect. 14.5: gravel is 2 to 256
# mm, ϕ = -1 to -8) and the coarse end of the grain-size relations, where
# MARS places each Folk class at its own centroid (e.g. 'G' at -3.3 ϕ). At -1 ϕ
# the 'hamilton' relation gives a density of 2.56 g/cm³ (TR 9407 Table 2's
# ratio 2.492 × 1.027).
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


def _bottom_from_folk5(code, lat, lon, *, roughness, water_sound_speed=None,
                       model=DEFAULT_GRAIN_SIZE_MODEL, hamilton_fit=None):
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
        return bottom_from_grain_size(value, roughness=roughness, model=model,
                                      hamilton_fit=hamilton_fit,
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


def fetch_emodnet_substrate(
    point: Coordinate,
    *,
    layer: str = EMODNET_LAYER,
    base_url: str = EMODNET_WFS_URL,
    timeout: float = 60.0,
    verbose: Union[bool, str] = False,
    max_distance_km: Optional[float] = None,
) -> SeabedSample:
    """Raw EMODnet seabed-substrate record at a ``(lat, lon)`` point.

    Returns a :class:`~uacpy.data.SeabedSample`: the ``folk_5cl`` code as
    ``folk_class`` (scheme ``'folk5'``), ``details`` holding
    ``folk_5cl_txt`` and ``original_grain_size``, and the ``'emodnet'``
    provenance with the requested point as its data point (the polygon
    contains it).

    Parameters
    ----------
    point : (lat, lon)
        Site coordinates in decimal degrees.
    layer : str, optional
        The WFS layer queried.
    base_url : str, optional
        The WFS endpoint.
    timeout : float, optional
        Per-request network timeout in seconds.
    verbose : bool or str, optional
        Logging gate passed through to ``log_message``.
    max_distance_km : float, optional
        The offset rule's refusal distance (km). The polygon read contains
        ``point``, so the offset is 0 km and every limit passes.

    Raises
    ------
    DataFetchError
        Service failure, or no coverage at the location (European seas only).
    """
    limit = checked_max_distance(max_distance_km, 'fetch_emodnet_substrate')
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
    p = features[0].get('properties') or {}
    # A polygon without a Folk code is a state of the layer (the offline index
    # builder skips the same features), so it is refused as no coverage.
    try:
        folk = int(p['folk_5cl'])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataFetchError(
            f"EMODnet has no Folk class at {lat:.3f}, {lon:.3f} (the polygon "
            f"there carries folk_5cl={p.get('folk_5cl')!r}).",
            remediation="Pass an explicit grain size (ϕ) or sediment class "
                        "as the bottom, or use another bottom source.",
        ) from exc
    return SeabedSample(
        grain_size_phi=None, material=None, folk_class=folk,
        folk_class_scheme='folk5', sample_point=None, distance_km=None,
        provenance=checked_offset(
            DataProvenance(source=SOURCES['emodnet'], data_point=(lat, lon),
                           requested_point=(lat, lon)),
            who='fetch_emodnet_substrate', warn_km=0.0,
            max_distance_km=limit),
        details={'folk_5cl_txt': p.get('folk_5cl_txt'),
                 'original_grain_size': p.get('original_grain_size')})


def fetch_bottom_emodnet(
    point: Coordinate,
    *,
    roughness: float = 0.0,
    water_sound_speed: Optional[float] = None,
    model: str = DEFAULT_GRAIN_SIZE_MODEL,
    hamilton_fit: Optional[str] = None,
    layer: str = EMODNET_LAYER,
    base_url: str = EMODNET_WFS_URL,
    timeout: float = 60.0,
    verbose: Union[bool, str] = False,
    max_distance_km: Optional[float] = None,
) -> BoundaryProperties:
    """Model-ready bottom from EMODnet seabed substrate at a ``(lat, lon)`` point.

    Convenience wrapper: :func:`fetch_emodnet_substrate` → Folk-class mapping →
    :func:`uacpy.data.bottom_from_grain_size` / ``bottom_from_class``. Raises
    ``DataFetchError`` outside European-seas coverage. ``water_sound_speed``
    (m/s) scales the grain-size velocity ratio to the in-situ near-seabed
    water; ``None`` uses the Hamilton reference (class bottoms are absolute and
    unaffected). ``model`` picks the grain-size relations (``'hamilton'`` or
    ``'apl-uw'``), see :func:`uacpy.core.sediment.grain_size_to_geoacoustics`.
    ``max_distance_km`` is as in :func:`fetch_emodnet_substrate`.
    """
    from uacpy.core.sediment import canonical_grain_size_selection
    model, hamilton_fit = canonical_grain_size_selection(
        model, hamilton_fit, who='fetch_bottom_emodnet')
    lat, lon = as_coordinate(point)
    sub = fetch_emodnet_substrate(point, layer=layer, base_url=base_url,
                                 timeout=timeout, verbose=verbose,
                                 max_distance_km=max_distance_km)
    bottom = _bottom_from_folk5(sub.folk_class, lat, lon, roughness=roughness,
                                water_sound_speed=water_sound_speed,
                                model=model, hamilton_fit=hamilton_fit)
    log_message(
        'seabed', f"EMODnet '{sub.details['folk_5cl_txt']}' at {lat:.3f}, "
        f"{lon:.3f} → {bottom.acoustic_type} "
        f"c_p={bottom.sound_speed:.0f} m/s",
        verbose=verbose,
    )
    return dataclasses.replace(bottom, data_sources=(sub.provenance,))
