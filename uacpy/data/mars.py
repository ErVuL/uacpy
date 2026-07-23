"""AusSeabed MARS sediment samples (Australia) → ``BoundaryProperties``.

The Australian counterpart of :mod:`uacpy.data.seabed` (EMODnet, European
seas): Geoscience Australia's Marine Sediments (MARS) database serves ~100k
quality-controlled seabed samples through a public no-auth WFS. Coverage is the
Australian margin; outside it the fetch raises ``DataFetchError`` and the
``'auto'`` bottom chain falls through.

Each sample is converted to a mean grain size (ϕ) by the first usable of:

1. ``MEAN_GRAIN_SIZE`` (µm) → ϕ = −log₂(mm),
2. ``MUD/SAND/GRAVEL_PERCENT`` → fraction-weighted representative ϕ,
3. ``FOLK_CLASS`` (Folk code, e.g. ``'mS'``) → representative ϕ,

then through :func:`uacpy.data.bottom_from_grain_size`. The server rejects
CQL ``BBOX`` filters (Oracle backend), so queries use the plain ``bbox=``
parameter over an expanding search-radius ladder and filter client-side.
"""

import json
import math
import urllib.parse
from typing import Dict, Optional, Union

from uacpy.core.environment import BoundaryProperties, Bottom
from uacpy.core.exceptions import DataFetchError
from uacpy.data._geo import (
    Coordinate, as_coordinate, great_circle_km, normalize_lon,
)
from uacpy.data._http import http_get
from uacpy.data.sediment import (
    bottom_from_grain_size, range_dependent_bottom_along,
)
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

# Representative ϕ per end-member fraction (matches the DECK41 lithology map).
_GRAVEL_PHI, _SAND_PHI, _MUD_PHI = -2.0, 1.5, 7.5

# Folk code → representative ϕ. Codes are gravel/sand/mud end members with
# s(andy)/m(uddy)/g(ravelly) modifiers; '(g)' = slightly gravelly.
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


def _query_bbox(lat, lon, radius_km, *, layer, base_url, timeout, verbose):
    """All MARS features inside a ``radius_km`` box around ``(lat, lon)``."""
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(math.cos(math.radians(lat)), 0.1))
    lon = normalize_lon(lon)
    # The server wants lon-lat axis order for EPSG:4326 bbox values.
    bbox = f"{lon - dlon:.4f},{lat - dlat:.4f},{lon + dlon:.4f},{lat + dlat:.4f}"
    query = urllib.parse.urlencode({
        'service': 'WFS', 'version': '2.0.0', 'request': 'GetFeature',
        'typeNames': layer, 'outputFormat': 'application/json',
        'count': str(_MAX_FEATURES), 'bbox': f"{bbox},EPSG:4326",
    })
    body = http_get(f"{base_url}?{query}", timeout=timeout, verbose=verbose,
                    source='mars')
    try:
        return json.loads(body).get('features') or []
    except json.JSONDecodeError as exc:
        raise DataFetchError(
            f"AusSeabed WFS returned a non-JSON body: {exc}.",
        ) from exc


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

    Returns ``{'phi', 'via', 'distance_km', 'folk_class'}`` where ``via`` names
    the conversion that produced ϕ (``'grain_size'`` / ``'percentages'`` /
    ``'folk_class'``).

    Raises
    ------
    DataFetchError
        Service failure, or no usable sample within ``max_distance_km``
        (coverage is the Australian margin).
    """
    lat, lon = as_coordinate(point)
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
                        'folk_class': (f['properties'] or {}).get('FOLK_CLASS')}
        if best is not None:
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
    sample = fetch_mars_sediment(
        point, max_distance_km=max_distance_km, layer=layer,
        base_url=base_url, timeout=timeout, verbose=verbose)
    return bottom_from_grain_size(
        sample['phi'], roughness=roughness,
        water_sound_speed=water_sound_speed)


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
    """Range-dependent bottom from MARS samples along ``start`` → ``end``."""
    return range_dependent_bottom_along(
        lambda la, lo: fetch_bottom_mars(
            (la, lo), roughness=roughness,
            water_sound_speed=water_sound_speed,
            max_distance_km=max_distance_km, layer=layer, base_url=base_url,
            timeout=timeout, verbose=verbose),
        start, end, n_points, source_label='AusSeabed MARS',
        max_points=max_points,
    )
