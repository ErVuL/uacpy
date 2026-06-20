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
full 10 km Diesing map (CC-BY, PANGAEA doi:10.1594/PANGAEA.911692) can replace
this rule where a reprojection dependency is acceptable.

Provinces (dominant surficial lithology → representative mean grain size ϕ):

* ``|lat| ≥ 50°``        → **diatom (siliceous) ooze** — sub-polar belts
* depth ``≥`` CCD        → **pelagic (red/brown) clay** — abyssal, carbonate-free
* otherwise (above CCD)  → **calcareous ooze** — low-to-mid latitude
"""

from typing import Union

from uacpy._log import log_message
from uacpy.core.environment import BoundaryProperties, Bottom
from uacpy.core.exceptions import ConfigurationError
from uacpy.data._geo import Coordinate, as_coordinate
from uacpy.data.sediment import (
    bottom_from_grain_size, range_dependent_bottom_along,
)

__all__ = ['pelagic_lithology', 'pelagic_grain_size', 'fetch_bottom_pelagic',
           'fetch_bottom_pelagic_transect']

# Carbonate compensation depth (m): below it calcareous tests dissolve, so
# carbonate ooze gives way to pelagic clay. Global-average value (the CCD is
# ~4–5 km, deeper in the Atlantic, shallower in the Pacific).
CCD_DEPTH = 4500.0
# Latitude (deg) poleward of which siliceous (diatom) ooze dominates the
# high-productivity belts (Southern Ocean, sub-arctic Pacific).
SILICEOUS_LATITUDE = 50.0

# Representative mean grain size ϕ per lithology (matching the lithology→ϕ values
# used by the local sediment DB), fed to the grain-size → geoacoustics relations.
_LITHOLOGY_PHI = {
    'calcareous ooze': 7.5,
    'diatom ooze': 9.0,
    'pelagic clay': 9.0,
}


def pelagic_lithology(depth_m: float, lat: float) -> str:
    """Dominant deep-sea surficial lithology from water depth (m) and latitude."""
    if abs(lat) >= SILICEOUS_LATITUDE:
        return 'diatom ooze'               # high-latitude siliceous belt
    if depth_m >= CCD_DEPTH:
        return 'pelagic clay'              # below the carbonate compensation depth
    return 'calcareous ooze'               # above the CCD, low-to-mid latitude


def pelagic_grain_size(depth_m: float, lat: float) -> float:
    """Representative mean grain size ϕ for the pelagic lithology at a point."""
    return _LITHOLOGY_PHI[pelagic_lithology(depth_m, lat)]


def _water_depth(point, timeout, verbose, cache_only):
    """Water depth (m) from GEBCO — local cache if installed, else the live API
    (skipped when ``cache_only``, so the cache-miss error propagates instead)."""
    from uacpy.data.bathymetry import fetch_bathy
    try:
        return fetch_bathy(point, source='local')
    except ConfigurationError:             # local grid not installed
        if cache_only:
            raise
        return fetch_bathy(point, source='api', timeout=timeout,
                                 verbose=verbose)


def fetch_bottom_pelagic(point: Coordinate, *, roughness: float = 0.0,
                         water_sound_speed: float = None,
                         depth: float = None, cache_only: bool = False,
                         timeout=None,
                         verbose: Union[bool, str] = False) -> BoundaryProperties:
    """Model-ready bottom from the pelagic depth/latitude model at ``(lat, lon)``.

    The water depth is taken from ``depth`` if given, else fetched from GEBCO
    (local cache, falling back to the live API unless ``cache_only``). The
    resulting lithology maps to a mean grain size ϕ and then a half-space via
    :func:`uacpy.data.bottom_from_grain_size`. ``water_sound_speed`` (m/s)
    scales the grain-size velocity ratio to the in-situ near-seabed water;
    ``None`` uses the Hamilton reference.
    """
    lat, lon = as_coordinate(point)
    d = depth if depth is not None else _water_depth(point, timeout, verbose,
                                                     cache_only)
    litho = pelagic_lithology(d, lat)
    bottom = bottom_from_grain_size(
        _LITHOLOGY_PHI[litho], roughness=roughness,
        water_sound_speed=water_sound_speed)
    log_message(
        'pelagic', f"pelagic {litho} at {lat:.2f}, {lon:.2f} "
        f"(depth {d:.0f} m) → ϕ={_LITHOLOGY_PHI[litho]}", verbose=verbose)
    return bottom


def fetch_bottom_pelagic_transect(start: Coordinate, end: Coordinate, *,
                                  n_points=6, max_points=None,
                                  roughness: float = 0.0,
                                  water_sound_speed: float = None,
                                  cache_only: bool = False, timeout=None,
                                  verbose: Union[bool, str] = False
                                  ) -> Bottom:
    """Range-dependent bottom from the pelagic model along ``start`` → ``end``."""
    return range_dependent_bottom_along(
        lambda la, lo: fetch_bottom_pelagic(
            (la, lo), roughness=roughness,
            water_sound_speed=water_sound_speed,
            cache_only=cache_only, timeout=timeout),
        start, end, n_points, source_label='pelagic model',
        max_points=max_points,
    )
