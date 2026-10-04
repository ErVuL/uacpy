"""First-principles pelagic surficial sediment for the open ocean.

Where *measured* surficial data is unavailable — the grain-size DB is sparse
(hundreds–thousands of km between samples offshore) and EMODnet is European-seas
only — the dominant deep-sea sediment is set from the classic **pelagic
distribution**: water depth relative to the carbonate compensation depth (CCD)
and latitude (the high-productivity siliceous-ooze belts).

It is **approximate** (a climatological rule, not a measurement) but **global**
and **commercial-clean**: it uses only the GEBCO water depth and the latitude —
no licensed data — so it makes ``bottom_sources='auto'`` resolve *anywhere* in
the ocean instead of failing offshore.

The thresholds reproduce the first-order findings of the reference global map of
**Diesing (2020)** — *Deep-sea sediments of the global ocean*, Earth Syst. Sci.
Data 12, 3367–3381, doi:10.5194/essd-12-3367-2020 (CC-BY 4.0) — namely that clay
dominates below the CCD (~4500 m), calcareous sediment above it, and siliceous
(diatom) ooze in the high-latitude productivity belts; after Berger (1974). The
full 10 km Diesing map (CC-BY, PANGAEA doi:10.1594/PANGAEA.911692) is
:mod:`uacpy.data.diesing_local`, which precedes this rule in the ``'auto'``
bottom chain; this rule covers what that map leaves (water shallower than
500 m, and an uninstalled cache).

Provinces (dominant surficial lithology → representative mean grain size ϕ):

* ``lat ≤ −50°``, or ``lat ≥ 50°`` in the subarctic **Pacific** sector
                         → **diatom (siliceous) ooze** — the belts of Berger
                           (1974): Southern Ocean + subarctic Pacific (the
                           subpolar North Atlantic is carbonate-dominated)
* depth ``≥`` CCD        → **pelagic (red/brown) clay** — abyssal, carbonate-free
* otherwise (above CCD)  → **calcareous ooze** — low-to-mid latitude
"""

import dataclasses
import warnings
from typing import Optional, Union

from uacpy._log import log_message
from uacpy.core.environment import BoundaryProperties
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.geo import Coordinate, as_coordinate, normalize_lon
from uacpy.data._geo import checked_max_distance, checked_offset
from uacpy.data.sources import SOURCES, DataProvenance
from uacpy.core.sediment import DEFAULT_GRAIN_SIZE_MODEL
from uacpy.data.sediment import bottom_from_grain_size
from uacpy.core.exceptions import ValidityWarning

__all__ = ['pelagic_lithology', 'pelagic_grain_size', 'fetch_bottom_pelagic']

# Carbonate compensation depth (m): below it calcareous tests dissolve, so
# carbonate ooze gives way to pelagic clay. Global-average value (the CCD is
# ~4–5 km, deeper in the Atlantic, shallower in the Pacific).
CCD_DEPTH = 4500.0
# Latitude (deg) poleward of which siliceous (diatom) ooze dominates the
# high-productivity belts (Southern Ocean, sub-arctic Pacific).
SILICEOUS_LATITUDE = 50.0
# Longitude bounds (deg E, crossing the antimeridian) of the subarctic Pacific
# sector — Sea of Okhotsk, Bering Sea, Gulf of Alaska. Berger (1974; see the
# module docstring) places the *northern* siliceous belt in the Pacific only,
# the subpolar North Atlantic being carbonate-dominated, so latitudes at or
# above SILICEOUS_LATITUDE are diatom ooze only inside this sector.
SILICEOUS_PACIFIC_LON = (140.0, -120.0)

# Representative mean grain size ϕ per lithology (matching the lithology→ϕ values
# used by the local sediment DB), fed to the grain-size → geoacoustics relations.
_LITHOLOGY_PHI = {
    'calcareous ooze': 7.5,
    'diatom ooze': 9.0,
    'pelagic clay': 9.0,
}


#: Conventional shelf break; below it the deep-sea facies model applies.
SHELF_BREAK_DEPTH = 200.0


def _in_siliceous_belt(lat: float, lon: Optional[float]) -> bool:
    """Whether a point lies in a high-latitude siliceous (diatom) belt.

    South of −``SILICEOUS_LATITUDE`` the Southern Ocean belt is circumglobal.
    At or north of +``SILICEOUS_LATITUDE`` the belt is the subarctic Pacific
    sector only (:data:`SILICEOUS_PACIFIC_LON`); a call with no longitude
    keeps the latitude-only classification and counts every such point as
    in the belt.
    """
    if lat <= -SILICEOUS_LATITUDE:
        return True
    if lat >= SILICEOUS_LATITUDE:
        if lon is None:
            return True
        west, east = SILICEOUS_PACIFIC_LON
        lon = normalize_lon(lon)
        return lon >= west or lon <= east
    return False


def pelagic_lithology(depth_m: float, lat: float,
                      lon: Optional[float] = None) -> str:
    """Dominant deep-sea surficial lithology from water depth (m) and position.

    Both facies this returns are **deep-sea**: the siliceous belts and the
    carbonate compensation depth are open-ocean features. Shelf sediment is
    terrigenous and this model does not describe it, so a depth above the
    conventional shelf break warns — the belt test fires first and
    unconditionally, so without it a 80 m shelf point at 60 deg came back as
    diatom ooze (phi 9.0), identical to a 4800 m abyssal one.

    ``lon`` restricts the *northern* siliceous belt to the subarctic Pacific
    sector (see :data:`SILICEOUS_PACIFIC_LON`; the Southern Ocean belt is
    circumglobal). Omitting it classifies on latitude alone, which counts the
    carbonate-dominated subpolar North Atlantic as siliceous — pass the
    longitude whenever the point has one.

    It still returns a value: :func:`fetch_bottom_pelagic` is the last
    fallback in the ``'auto'`` chain and is documented as never failing.

    Parameters
    ----------
    depth_m : float
        Water depth (m).
    lat : float
        Latitude (deg).
    lon : float, optional
        Longitude (deg); see below.
    """
    if depth_m < SHELF_BREAK_DEPTH:
        warnings.warn(
            f"pelagic_lithology: {depth_m:g} m is above the {SHELF_BREAK_DEPTH:g} m "
            f"shelf break, and this model describes deep-sea facies only "
            f"(siliceous belt / carbonate compensation depth). Shelf sediment "
            f"is terrigenous. The value returned is a first-principles "
            f"fallback, not a shelf estimate.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP)
    if _in_siliceous_belt(lat, lon):
        return 'diatom ooze'               # high-latitude siliceous belt
    if depth_m >= CCD_DEPTH:
        return 'pelagic clay'              # below the carbonate compensation depth
    return 'calcareous ooze'               # above the CCD, low-to-mid latitude


def pelagic_grain_size(depth_m: float, lat: float,
                       lon: Optional[float] = None) -> float:
    """Representative mean grain size ϕ for the pelagic lithology at a point.

    ``lon`` is forwarded to :func:`pelagic_lithology` (Pacific-sector test of
    the northern siliceous belt).

    Parameters
    ----------
    depth_m : float
        Water depth (m).
    lat : float
        Latitude (deg).
    lon : float, optional
        Longitude (deg), for the Pacific-sector test.
    """
    return _LITHOLOGY_PHI[pelagic_lithology(depth_m, lat, lon)]


def _water_depth(point, timeout, verbose, cache_only):
    """Water depth (m) from GEBCO — local cache if installed, else the live API
    (skipped when ``cache_only``, so the cache-miss error propagates instead)."""
    from uacpy.data.bathymetry import fetch_bathy
    return fetch_bathy(point, source='local' if cache_only else 'gebco',
                       timeout=timeout, verbose=verbose)


def fetch_bottom_pelagic(point: Coordinate, *, roughness: float = 0.0,
                         water_sound_speed: Optional[float] = None,
                         model: str = DEFAULT_GRAIN_SIZE_MODEL,
                         hamilton_fit: Optional[str] = None,
                         depth: Optional[float] = None, cache_only: bool = False,
                         timeout: float = 30.0,
                         verbose: Union[bool, str] = False,
                         max_distance_km: Optional[float] = None,
                         ) -> BoundaryProperties:
    """Model-ready bottom from the pelagic depth/latitude model at ``(lat, lon)``.

    The returned bottom carries a ``pelagic`` ``DataProvenance``, plus a
    ``gebco`` one when the depth was fetched rather than given. Both record
    the requested point as their data point: the lithology is computed at the
    point from its depth and latitude, and the GEBCO depth is read in the
    point's own 15-arc-second cell, so ``offset_km`` is 0 and every
    ``max_distance_km`` passes.

    The water depth is taken from ``depth`` if given, else fetched from GEBCO
    (local cache, falling back to the live API unless ``cache_only``). The
    resulting lithology maps to a mean grain size ϕ and then a half-space via
    :func:`uacpy.data.bottom_from_grain_size`. ``water_sound_speed`` (m/s)
    scales the grain-size velocity ratio to the in-situ near-seabed water;
    ``None`` uses the Hamilton reference. ``model`` picks the grain-size
    relations (``'hamilton'`` or ``'apl-uw'``). ``timeout`` (s) bounds each
    live GEBCO request; a stalled host raises ``DataFetchError`` instead of
    blocking indefinitely.
    """
    from uacpy.core.sediment import canonical_grain_size_selection
    model, hamilton_fit = canonical_grain_size_selection(
        model, hamilton_fit, who='fetch_bottom_pelagic')
    lat, lon = as_coordinate(point)
    limit = checked_max_distance(max_distance_km, 'fetch_bottom_pelagic')
    d = depth if depth is not None else _water_depth(point, timeout, verbose,
                                                     cache_only)
    litho = pelagic_lithology(d, lat, lon)
    bottom = bottom_from_grain_size(
        _LITHOLOGY_PHI[litho], roughness=roughness, model=model,
        hamilton_fit=hamilton_fit,
        water_sound_speed=water_sound_speed)
    log_message(
        'pelagic', f"pelagic {litho} at {lat:.2f}, {lon:.2f} "
        f"(depth {d:.0f} m) → ϕ={_LITHOLOGY_PHI[litho]}", verbose=verbose)
    ids = ('pelagic',) if depth is not None else ('pelagic', 'gebco')
    return dataclasses.replace(bottom, data_sources=tuple(
        checked_offset(DataProvenance(source=SOURCES[i], data_point=(lat, lon),
                                      requested_point=(lat, lon)),
                       who='fetch_bottom_pelagic', warn_km=0.0,
                       max_distance_km=limit)
        for i in ids))
