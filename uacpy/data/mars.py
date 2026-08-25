"""AusSeabed MARS sediment samples (Australia) → ``BoundaryProperties``.

The Australian counterpart of :mod:`uacpy.data.seabed` (EMODnet, European
seas): Geoscience Australia's Marine Sediments (MARS) database serves ~100k
quality-controlled seabed samples through a public no-auth WFS. Coverage is the
Australian margin; a point outside :data:`_COVERAGE_BOX` raises
``DataFetchError`` without issuing a request, so the ``'auto'`` bottom chain
falls through for free.

Each sample is converted to a mean grain size (ϕ) by the first usable of:

1. ``MEAN_GRAIN_SIZE`` (µm) → ϕ = −log₂(mm),
2. ``MUD/SAND/GRAVEL_PERCENT`` → fraction-weighted representative ϕ,
3. ``FOLK_CLASS`` (Folk code, e.g. ``'mS'``) → representative ϕ,

then through :func:`uacpy.data.bottom_from_grain_size`. The server rejects
CQL ``BBOX`` filters (Oracle backend), so queries use the plain ``bbox=``
parameter over an expanding search-radius ladder and filter client-side.
"""

import dataclasses
import json
import math
import urllib.parse
import warnings
from typing import Dict, Optional, Union

from uacpy.core.environment import BoundaryProperties, Bottom
from uacpy.core.exceptions import DataFetchError
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.data._geo import (
    Coordinate, as_coordinate, great_circle_km, normalize_lon,
)
from uacpy.data._http import http_get
from uacpy.data.sediment import (
    bottom_from_grain_size, range_dependent_bottom_along, water_sound_speed_at,
)
from uacpy.data.sources import SOURCES, DataProvenance
from uacpy._log import log_message

__all__ = ['fetch_mars_sediment', 'fetch_bottom_mars',
           'fetch_bottom_mars_transect']

MARS_WFS_URL = 'https://warehouse.ausseabed.gov.au/geoserver/wfs'
MARS_LAYER = 'ausseabed:Seabed_Sediments_Collection'

DEFAULT_MAX_DISTANCE_KM = 100.0
# Expanding search radii (km): most points on the shelf resolve in the first
# small box; the final rung is max_distance_km itself.
_SEARCH_RADII_KM = (10.0, 30.0)
_MAX_FEATURES = 2000
#: ``(lat_min, lat_max, lon_min, lon_max)`` enclosing Australia's marine
#: jurisdiction (mainland margin, the Indian/Southern Ocean external
#: territories and the Australian Antarctic Territory). Deliberately far wider
#: than the sampled area: it exists only so a point on another ocean's shelf
#: fails without a request, which matters because MARS sits in the ``'auto'``
#: bottom chain and its search is a three-rung radius ladder.
_COVERAGE_BOX = (-90.0, 0.0, 40.0, 180.0)

# Representative ϕ per end-member fraction (matches the DECK41 lithology map).
_GRAVEL_PHI, _SAND_PHI, _MUD_PHI = -2.0, 1.5, 7.5

# Folk code → representative ϕ. Codes are gravel/sand/mud end members with
# s(andy)/m(uddy)/g(ravelly) modifiers; '(g)' = slightly gravelly.
#
# The ϕ scale is Krumbein's, ϕ = -log2(d / 1 mm) — the same conversion
# :func:`_phi_from_properties` applies to MEAN_GRAIN_SIZE just below. Medwin &
# Clay Sect. 14.2 fixes the coarse end against it: "Marine geologists define
# gravel as being the loose material that ranges in size from 2 to 256 mm
# (Gross 1972, Glossary)", i.e. ϕ = -1 down to -8.
#
# These are MIXTURE MEANS, not class ranges, and the distinction matters if
# anyone is tempted to "correct" them: a muddy gravel ('mG') is gravel plus
# mud, so its mean sits at ϕ = 0 — inside the sand range — even though its
# dominant end member is gravel and gravel proper begins at ϕ = -1. Read as
# class bounds the table looks wrong; read as mixture means it is monotone
# from 'G' = -2.0 to 'M' = 7.5, with every gravel-dominant code coarser than
# every sand-dominant one and so on.
#
# The Folk (1954) ternary classification itself is NOT in the local corpus;
# only the ϕ scale and the gravel boundary above are grounded here.
_FOLK_TO_PHI = {
    'G': -2.0, 'sG': -1.0, 'msG': -0.5, 'mG': 0.0, 'gS': 0.0, 'gmS': 1.0,
    '(g)S': 1.0, 'S': 1.5, '(g)mS': 3.0, 'mS': 4.0, 'gM': 5.0, '(g)sM': 5.5,
    'sM': 6.0, '(g)M': 7.0, 'M': 7.5,
}


def _phi_from_properties(p: Dict) -> Optional[Dict]:
    """First usable ϕ from a MARS feature's properties, or ``None``.

    Returns ``{'phi', 'via'}`` — the conversion chain is grain size (µm) →
    mud/sand/gravel percentages → Folk class.
    """
    grain_um = p.get('MEAN_GRAIN_SIZE')
    if grain_um is not None and float(grain_um) > 0.0:
        return {'phi': -math.log2(float(grain_um) / 1000.0),
                'via': 'grain_size'}
    fracs = [p.get('GRAVEL_PERCENT'), p.get('SAND_PERCENT'),
             p.get('MUD_PERCENT')]
    if any(f is not None for f in fracs):
        g, s, m = (0.0 if f is None else max(float(f), 0.0) for f in fracs)
        total = g + s + m
        if total > 0.0:
            phi = (g * _GRAVEL_PHI + s * _SAND_PHI + m * _MUD_PHI) / total
            return {'phi': phi, 'via': 'percentages'}
    folk = p.get('FOLK_CLASS')
    if folk in _FOLK_TO_PHI:
        return {'phi': _FOLK_TO_PHI[folk], 'via': 'folk_class'}
    return None


def _require_coverage(lat, lon, max_distance_km):
    """Raise ``DataFetchError`` for a point outside :data:`_COVERAGE_BOX`.

    The box is padded by ``max_distance_km`` converted at a flat 111 km/degree.
    That is exact in latitude but under-pads in longitude away from the equator
    (at 65°S a degree is only ~47 km, so the pad spans ~42 km of a 100 km
    guard). Harmless only because :data:`_COVERAGE_BOX` is drawn far outside the
    sampled area — the pad is a courtesy margin, not the thing keeping a real
    sample in scope.
    """
    lat_min, lat_max, lon_min, lon_max = _COVERAGE_BOX
    pad = float(max_distance_km) / 111.0    # ~111 km per degree of latitude
    lon = normalize_lon(lon)
    if (lat_min - pad <= lat <= lat_max + pad
            and lon_min - pad <= lon <= lon_max + pad):
        return
    raise DataFetchError(
        f"AusSeabed MARS does not cover {lat:.3f}, {lon:.3f} (coverage is the "
        "Australian margin).",
        remediation="Pick a covered point, use another bottom source, or let "
                    "the 'auto' bottom chain fall through.",
    )


def _query_bbox(lat, lon, radius_km, *, layer, base_url, timeout, verbose):
    """All MARS features inside a ``radius_km`` box around ``(lat, lon)``."""
    dlat = radius_km / 111.0                # ~111 km per degree of latitude
    dlon = radius_km / (111.0 * max(math.cos(math.radians(lat)), 0.1))
    lon = normalize_lon(lon)
    # The server wants lon-lat axis order for EPSG:4326 bbox values; clamp
    # the box into [-180, 180] so a point near the coverage edge (180 E)
    # cannot emit an out-of-range longitude (coverage ends there anyway).
    lon_lo = max(lon - dlon, -180.0)
    lon_hi = min(lon + dlon, 180.0)
    lat_lo = max(lat - dlat, -90.0)
    lat_hi = min(lat + dlat, 90.0)
    bbox = f"{lon_lo:.4f},{lat_lo:.4f},{lon_hi:.4f},{lat_hi:.4f}"
    query = urllib.parse.urlencode({
        'service': 'WFS', 'version': '2.0.0', 'request': 'GetFeature',
        'typeNames': layer, 'outputFormat': 'application/json',
        'count': str(_MAX_FEATURES), 'bbox': f"{bbox},EPSG:4326",
    })
    body = http_get(f"{base_url}?{query}", timeout=timeout, verbose=verbose,
                    source='mars')
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise DataFetchError(
            f"AusSeabed WFS returned a non-JSON body: {exc}.",
        ) from exc
    features = payload.get('features') or []
    matched = payload.get('numberMatched')
    if isinstance(matched, (int, float)) and matched > len(features):
        warnings.warn(
            f"AusSeabed MARS returned {len(features)} of {int(matched)} "
            f"matching samples (server page cap) — the result may not be the "
            f"nearest sample; use a smaller search radius.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
    return features


def fetch_mars_sediment(
    point: Coordinate,
    *,
    max_distance_km: float = DEFAULT_MAX_DISTANCE_KM,
    layer: str = MARS_LAYER,
    base_url: str = MARS_WFS_URL,
    timeout: float = 60.0,
    verbose: Union[bool, str] = False,
) -> Dict:
    """Nearest usable MARS sediment sample to a ``(lat, lon)`` point.

    Returns ``{'phi', 'via', 'distance_km', 'folk_class', 'latitude',
    'longitude'}`` where ``via`` names the conversion that produced ϕ
    (``'grain_size'`` / ``'percentages'`` / ``'folk_class'``) and
    ``latitude``/``longitude`` are the sample's own coordinates, so a caller
    can record where the value actually came from.

    Raises
    ------
    DataFetchError
        Service failure, or no usable sample within ``max_distance_km``
        (coverage is the Australian margin).
    """
    lat, lon = as_coordinate(point)
    _require_coverage(lat, lon, max_distance_km)
    radii = [r for r in _SEARCH_RADII_KM if r < max_distance_km]
    radii.append(max_distance_km)
    best = None
    for radius in radii:
        for f in _query_bbox(lat, lon, radius, layer=layer, base_url=base_url,
                             timeout=timeout, verbose=verbose):
            conv = _phi_from_properties(f.get('properties') or {})
            if conv is None:
                continue
            coords = (f.get('geometry') or {}).get('coordinates') or None
            if not coords or len(coords) < 2:
                continue
            d = great_circle_km(lat, lon, coords[1], coords[0])
            if best is None or d < best['distance_km']:
                best = {**conv, 'distance_km': float(d),
                        'folk_class': (f['properties'] or {}).get('FOLK_CLASS'),
                        'latitude': float(coords[1]),
                        'longitude': float(coords[0])}
        # A bbox-corner hit can lie beyond the rung radius while a closer
        # sample sits just outside the box — only settle once the best find
        # is within the rung actually searched.
        if best is not None and best['distance_km'] <= radius:
            break
    if best is None or best['distance_km'] > max_distance_km:
        raise DataFetchError(
            f"AusSeabed MARS has no usable sediment sample within "
            f"{max_distance_km:.0f} km of {lat:.3f}, {lon:.3f} "
            "(coverage is the Australian margin; max_distance_km guard).",
            remediation="Raise max_distance_km, pick a covered point, or let "
                        "the 'auto' bottom chain fall through.",
        )
    log_message(
        'mars', f"MARS sample {best['distance_km']:.1f} km from "
        f"{lat:.3f}, {lon:.3f}: ϕ={best['phi']:.2f} via {best['via']}",
        verbose=verbose,
    )
    return best


def fetch_bottom_mars(
    point: Coordinate,
    *,
    roughness: float = 0.0,
    water_sound_speed: Optional[float] = None,
    max_distance_km: float = DEFAULT_MAX_DISTANCE_KM,
    layer: str = MARS_LAYER,
    base_url: str = MARS_WFS_URL,
    timeout: float = 60.0,
    verbose: Union[bool, str] = False,
) -> BoundaryProperties:
    """Model-ready bottom from the nearest MARS sample at a ``(lat, lon)`` point.

    Convenience wrapper: :func:`fetch_mars_sediment` →
    :func:`uacpy.data.bottom_from_grain_size`. ``water_sound_speed`` (m/s)
    scales the grain-size velocity ratio to the in-situ near-seabed water;
    ``None`` uses the Hamilton reference.
    """
    lat, lon = as_coordinate(point)
    sample = fetch_mars_sediment(
        point, max_distance_km=max_distance_km, layer=layer,
        base_url=base_url, timeout=timeout, verbose=verbose)
    bottom = bottom_from_grain_size(
        sample['phi'], roughness=roughness,
        water_sound_speed=water_sound_speed)
    # Point samples are sparse, so the nearest one can be up to
    # max_distance_km from the requested position; record where it actually
    # came from so ``citations(env)`` reports the hop and ``prov.offset_km``
    # measures it — the same stamp the local grain-size DB carries.
    prov = DataProvenance(
        source=SOURCES['mars'],
        data_point=(sample['latitude'], sample['longitude']),
        requested_point=(lat, lon),
    )
    return dataclasses.replace(bottom, data_sources=(prov,))


def fetch_bottom_mars_transect(
    start: Coordinate, end: Coordinate, *,
    n_points=6,
    max_points=None,
    roughness: float = 0.0,
    water_sound_speed: Optional[float] = None,
    max_distance_km: float = DEFAULT_MAX_DISTANCE_KM,
    layer: str = MARS_LAYER,
    base_url: str = MARS_WFS_URL,
    timeout: float = 60.0,
    verbose: Union[bool, str] = False,
) -> Bottom:
    """Range-dependent bottom from MARS samples along ``start`` → ``end``.

    ``water_sound_speed`` also takes a ``(lat, lon) -> m/s`` callable,
    so each column scales to the water over its own seafloor.
    """
    return range_dependent_bottom_along(
        lambda la, lo: fetch_bottom_mars(
            (la, lo), roughness=roughness,
            water_sound_speed=water_sound_speed_at(water_sound_speed, la, lo),
            max_distance_km=max_distance_km, layer=layer, base_url=base_url,
            timeout=timeout, verbose=verbose),
        start, end, n_points, source_label='AusSeabed MARS',
        max_points=max_points,
    )
