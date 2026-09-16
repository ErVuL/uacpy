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
3. ``FOLK_CLASS`` (Folk code, e.g. ``'mS'``) → the ϕ of that class's centroid
   on Folk's ternary diagram, through the same mixture expression as (2),

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
from uacpy.core.sediment import GRAIN_SIZE_MODEL_RANGES
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

# Representative ϕ per end-member fraction, which the percentage route weights.
# ``_SAND_PHI`` and ``_MUD_PHI`` are what the DECK41 lithology map gives those
# same two words (``sediment_db._DECK41_LITHOLOGY_TO_PHI``). ``_GRAVEL_PHI`` has
# no counterpart there: that map carries no ϕ for gravel at all and routes the
# term to the ``'gravel'`` material preset, because both grain-size models stop
# at -1 ϕ. Here a number is unavoidable — a mixture mean needs one per end
# member — and -2.0 ϕ is 4 mm, inside the 2-256 mm the Medwin & Clay definition
# below gives gravel. A mixture that lands outside the model's range is
# announced by ``_warn_outside_the_fit`` rather than converted in silence.
_GRAVEL_PHI, _SAND_PHI, _MUD_PHI = -2.0, 1.5, 7.5


def _phi_of_mixture(gravel: float, sand: float, mud: float) -> float:
    """Mean ϕ of a gravel / sand / mud mixture, weights in any common unit.

    The one expression both conversions use: the percentage route hands it the
    sample's own measured fractions, and every Folk class's representative ϕ is
    this evaluated at that class's centroid, so the two agree wherever they
    describe the same composition.
    """
    total = gravel + sand + mud
    return (gravel * _GRAVEL_PHI + sand * _SAND_PHI + mud * _MUD_PHI) / total


# Folk's 15 textural classes, as boundaries rather than as assigned values:
# ``(gravel_lo, gravel_hi): ((x_lo, x_hi, code), ...)`` where gravel is a
# weight fraction and ``x = s / (s + m)``, so Folk's sand:mud ratio cuts of
# 1:9, 1:1 and 9:1 are x = 0.1, 0.5 and 0.9.
#
# The two families are cut differently and the table keeps that: the gravel
# lines are percentages of gravel (80 / 30 / 5 / 0.01), while the sand-mud
# lines are a ratio, and which ratio lines apply depends on the gravel band —
# the 1:9 cut exists only where there is no gravel modifier to spend, which is
# why the two upper bands hold three classes and the two lower ones hold four.
#
# Read off Folk's diagram as two USGS publications reproduce it, both read
# directly rather than summarised: USGS Open-File Report 2006-1195
# ("Surficial sediment character of the Louisiana offshore continental shelf
# region"), Nomenclature > Folk, figure ``htmldocs/images/folk.gif``; and USGS
# Scientific Investigations Report 2019-5073 ("Sediment Classification and the
# Characterization ... of the Glaciated Gulf of Maine Seabed"), figure 3, whose
# text states the thresholds in prose — "Sediments with only 0.01 to <5 weight
# percent gravel are classified as slightly gravelly, and sediments with >=30
# weight percent are classified as gravel" — and cites "Folk, 1954, fig. 1a,
# table 1; 1980, p. 25-28, table 1". Folk (1954) itself is cited by APL-UW
# TR 9407's reference list (``external:APLTM9407.md:3313``) but is not in the
# local corpus.
_FOLK_CLASS_LIMITS = {
    (0.80, 1.00): ((0.0, 1.0, 'G'),),
    (0.30, 0.80): ((0.0, 0.5, 'mG'), (0.5, 0.9, 'msG'), (0.9, 1.0, 'sG')),
    (0.05, 0.30): ((0.0, 0.5, 'gM'), (0.5, 0.9, 'gmS'), (0.9, 1.0, 'gS')),
    (0.0001, 0.05): ((0.0, 0.1, '(g)M'), (0.1, 0.5, '(g)sM'),
                     (0.5, 0.9, '(g)mS'), (0.9, 1.0, '(g)S')),
    (0.0, 0.0001): ((0.0, 0.1, 'M'), (0.1, 0.5, 'sM'),
                    (0.5, 0.9, 'mS'), (0.9, 1.0, 'S')),
}


def _folk_class_centroid(g_lo, g_hi, x_lo, x_hi):
    """``(gravel, sand, mud)`` at the centroid of one Folk class.

    The centroid is taken over the class's own area on the ternary diagram the
    classification is drawn on, whose measure is ``(1 - g) dg dx``: at a given
    gravel fraction the sand-mud edge is only ``1 - g`` long, so a band's
    coarse end carries less of the diagram than its fine end and a plain
    midpoint would misplace every class. Every class is bounded — ``'G'`` by
    the gravel apex, the ratio classes by x in [0, 1] — so no convention is
    needed for an open end. The zero-gravel band is the diagram's base edge,
    a sliver 0.01 % tall; its centroid sits at 0.005 % gravel, which is the
    base edge to any precision that matters here.
    """
    area = (g_hi - g_lo) - (g_hi ** 2 - g_lo ** 2) / 2.0
    g_moment = (g_hi ** 2 - g_lo ** 2) / 2.0 - (g_hi ** 3 - g_lo ** 3) / 3.0
    gravel = g_moment / area
    sand_and_mud = 1.0 - gravel
    x = 0.5 * (x_lo + x_hi)
    return gravel, sand_and_mud * x, sand_and_mud * (1.0 - x)


# Folk code → representative ϕ, each class evaluated at its own centroid
# through ``_phi_of_mixture`` — the expression the percentage route uses — so a
# sample classified 'sG' and a sample whose measured percentages sit at the
# centroid of 'sG' convert to the same ϕ by construction. The values are
# derived at import rather than written down, so the table cannot drift from
# the boundaries above.
#
# These are MIXTURE MEANS, not class ranges, and the distinction matters if
# anyone is tempted to "correct" them: 'mG' is half gravel and a third mud, so
# its mean sits in the sands at 1.97 ϕ even though its dominant end member is
# gravel and gravel proper begins at ϕ = -1. The ordering by ϕ is therefore
# *not* the ordering by gravel content — 'mG' is finer than 'S' — which is a
# property of a linear mean in ϕ, not a fault in the table.
#
# The ϕ scale is Krumbein's, ϕ = -log2(d / 1 mm) — the same conversion
# :func:`_phi_from_properties` applies to MEAN_GRAIN_SIZE just below. Medwin &
# Clay Sect. 14.2 fixes the coarse end against it: "Marine geologists define
# gravel as being the loose material that ranges in size from 2 to 256 mm
# (Gross 1972, Glossary)", i.e. ϕ = -1 down to -8.
_FOLK_TO_PHI = {
    code: _phi_of_mixture(*_folk_class_centroid(g_lo, g_hi, x_lo, x_hi))
    for (g_lo, g_hi), cells in _FOLK_CLASS_LIMITS.items()
    for x_lo, x_hi, code in cells
}


def _property_as_float(name: str, value) -> float:
    """``float(value)`` for a MARS server property, or ``DataFetchError``.

    A property the server populated with something non-numeric raises the
    typed error the source-fallback chains catch, so the ``'auto'`` bottom
    chain falls through to its next source instead of aborting.
    """
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise DataFetchError(
            f"AusSeabed MARS returned a non-numeric {name}: {value!r}.",
            remediation="Use another bottom source, or let the 'auto' bottom "
                        "chain fall through.",
        ) from exc


def _phi_from_properties(p: Dict) -> Optional[Dict]:
    """First usable ϕ from a MARS feature's properties, or ``None``.

    Returns ``{'phi', 'via'}`` — the conversion chain is grain size (µm) →
    mud/sand/gravel percentages → Folk class. A property populated with a
    non-numeric value raises ``DataFetchError`` (:func:`_property_as_float`).
    """
    grain_um = p.get('MEAN_GRAIN_SIZE')
    if grain_um is not None:
        grain_um = _property_as_float('MEAN_GRAIN_SIZE', grain_um)
        if grain_um > 0.0:
            return {'phi': -math.log2(grain_um / 1000.0),
                    'via': 'grain_size'}
    names = ('GRAVEL_PERCENT', 'SAND_PERCENT', 'MUD_PERCENT')
    fracs = [p.get(n) for n in names]
    if any(f is not None for f in fracs):
        g, s, m = (0.0 if f is None else max(_property_as_float(n, f), 0.0)
                   for n, f in zip(names, fracs))
        if g + s + m > 0.0:
            return {'phi': _phi_of_mixture(g, s, m), 'via': 'percentages'}
    folk = p.get('FOLK_CLASS')
    if folk in _FOLK_TO_PHI:
        return {'phi': _FOLK_TO_PHI[folk], 'via': 'folk_class'}
    return None


def _warn_outside_the_fit(sample: Dict, model: str) -> None:
    """Announce a sample whose ϕ lies outside ``model``'s own fitted range.

    MARS carries seabed coarser than either grain-size relation covers — a
    gravel-dominant Folk class, a gravel-weighted percentage mixture, or a
    measured ``MEAN_GRAIN_SIZE`` above 2 mm — and
    :func:`~uacpy.core.sediment.grain_size_to_geoacoustics` answers those with
    its fit at the nearer end of the range. Under ``'hamilton'`` that
    substitution is invisible downstream (the fit holds its end rows flat, so
    the clamp moves nothing to compare against), which is why it is reported
    here, where the sample and the route that produced ϕ are both known.
    """
    bounds = GRAIN_SIZE_MODEL_RANGES.get(model)
    if bounds is None:
        return                      # unknown model: the conversion raises
    lo, hi = bounds
    phi = float(sample['phi'])
    if lo <= phi <= hi:
        return
    warnings.warn(
        f"AusSeabed MARS sample at {sample['latitude']:.3f}, "
        f"{sample['longitude']:.3f} has ϕ={phi:.2f} (via {sample['via']}), "
        f"outside the [{lo:g}, {hi:g}] ϕ the {model!r} grain-size relations "
        f"are fitted over: the geoacoustics returned are that fit at "
        f"ϕ={min(max(phi, lo), hi):g}, not at the sampled grain size. The "
        f"'apl-uw' relations reach -1 ϕ (2 mm) against 'hamilton''s 0 ϕ; for "
        f"seabed coarser than that, a class from uacpy.core.materials "
        f"describes it and a grain-size relation does not.",
        UserWarning, skip_file_prefixes=USER_FRAME_SKIP)


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
    model: str = 'hamilton',
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
    ``None`` uses the Hamilton reference. ``model`` picks the grain-size
    relations (``'hamilton'`` or ``'apl-uw'``); a sample coarser or finer than
    the chosen relations are fitted over is converted at the nearer end of
    that fit, with a ``UserWarning`` naming the sample and the route its ϕ
    came from.
    """
    lat, lon = as_coordinate(point)
    sample = fetch_mars_sediment(
        point, max_distance_km=max_distance_km, layer=layer,
        base_url=base_url, timeout=timeout, verbose=verbose)
    _warn_outside_the_fit(sample, model)
    bottom = bottom_from_grain_size(
        sample['phi'], roughness=roughness, model=model,
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
    model: str = 'hamilton',
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
            model=model,
            max_distance_km=max_distance_km, layer=layer, base_url=base_url,
            timeout=timeout, verbose=verbose),
        start, end, n_points, source_label='AusSeabed MARS',
        max_points=max_points,
    )
