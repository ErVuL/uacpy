"""On-demand external-data layer — GPS coordinates → ``Environment`` inputs.

- **Bathymetry** (GEBCO, GMRT, EMODnet DTM): a single water depth, a
  range-dependent transect for ``Environment(bathymetry=...)``, or a lat/lon
  grid for a map.
- **Sound speed** (WOA23 climatology, date/month-aware; Copernicus Marine
  operational; or the nearest Argo float profile): a depth-vs-c profile for
  ``Environment(ssp=...)``, plus the raw T/S column and a Francois-Garrison
  absorption helper; a cast of your own is continued to depth by the WOA23
  column (:func:`deepen_with_climatology`).
- **Bottom**: the grain-size (ϕ) conversion
  (:func:`grain_size_to_geoacoustics`; a seabed from a ϕ or a sediment class
  is ``BoundaryProperties.from_grain_size`` / ``from_preset``), your own
  seabed samples → a range-dependent ``Bottom`` along a transect
  (:func:`bottom_from_samples`), *or* a fetched seafloor — EMODnet
  substrate (European seas), the NCEI grain-size DB, the AusSeabed MARS samples (Australian margin), the
  Diesing 2020 deep-sea map, the Graw 2021 seabed-density grid, a pelagic
  depth/latitude fallback (all surficial), or GlobSed thickness + CRUST1.0 → a
  layered elastic bottom for low-frequency work.
- **Surface**: NSIDC sea-ice concentration → an elastic ice-canopy
  ``BoundaryProperties`` (:func:`fetch_sea_ice_surface`), so an ice-covered
  point replaces the free surface with a pack-ice boundary.
- **Sea state**: significant wave height (Copernicus WAVERYS, WaveWatch III)
  or 10 m wind (NBS live, or its cached climatology) → a Pierson-Moskowitz
  altimetry realization (:func:`fetch_sea_surface`).
- **Absorption inputs**: pH (GLODAP, Copernicus BGC) for the
  Francois-Garrison absorption
  (:meth:`~uacpy.core.absorption.FrancoisGarrison.from_temperature_salinity`).
- **Capstone**: :func:`fetch_environment` assembles them
  (``surface_sources='seaice'`` adds the ice surface). Each axis is a literal
  (``ssp=`` / ``bathymetry=`` / ``bottom=`` / ``surface=`` / ``altimetry=``)
  and/or fetched from ordered-fallback ``*_sources`` (source first, literal as
  fallback; ``'auto'`` = best available, ``'local'`` = local data only, no
  network). Fetching is cache-first *within each source*: a source with a
  locally installed twin samples it before its own live backend. The chain
  order is quality-first, so an ``'auto'`` run can reach the network before
  any cache — use ``'local'`` for an offline or reproducible run.

Examples
--------
>>> import uacpy
>>> from uacpy.data import fetch_environment
>>> env = uacpy.data.fetch_environment((43.2, 7.5), date='2026-06-14',
...                                    bottom='sand')          # doctest: +SKIP
"""

from uacpy.data.bathymetry import (
    BathyGrid, fetch_bathy, fetch_bathy_transect, fetch_bathy_grid,
    transect_length,
    transect_waypoints, bathy_transect_sampling,
)
from uacpy.data.sound_speed import (
    TSProfile, fetch_ssp, fetch_ssp_transect, fetch_ts_profile,
    ssp_transect_sampling,
    extend_ssp_below_data, extend_column_to_seafloor,
    deepen_with_climatology,
)
from uacpy.data.copernicus import (
    fetch_ssp_operational, fetch_ssp_transect_operational,
    fetch_ts_profile_operational, fetch_waves_operational,
    fetch_ph_operational,
)
from uacpy.data._geo import AlongTrack
from uacpy.data.wind_live import fetch_wind, fetch_wind_transect
from uacpy.data.wind_local import download_wind_db
from uacpy.data.waves import SeaStateRecord, fetch_waves
from uacpy.data.sea_surface import fetch_sea_surface, hs_to_pm_wind
from uacpy.data.argo import ArgoProfile, fetch_argo_profile, fetch_ssp_argo
from uacpy.data.glodap_local import (
    PHProfile, download_glodap_db, fetch_ph_profile, fetch_ph,
)
from uacpy.data.sediment import SeabedSample, samples_table
# The ϕ conversion behind every grain-size seabed, and its inverse: plain
# formulas of uacpy.core.sediment, offered here beside the seabed fetchers.
from uacpy.core.sediment import (grain_size_from_density,
                                 grain_size_to_geoacoustics)
from uacpy.data.seabed import fetch_emodnet_substrate
from uacpy.data.mars import fetch_mars_sediment
from uacpy.data.sediment_db import (bottom_from_samples, download_sediment_db,
                                    fetch_sediment_sample)
from uacpy.data.emodnet_local import download_emodnet_db
from uacpy.data.globsed_local import (
    download_globsed_db, fetch_sediment_thickness, fetch_sediment_thickness_transect,
)
from uacpy.data.crust1_local import (Crust1Profile, download_crust1_db,
                                     fetch_crust1_profile)
from uacpy.data.pelagic import pelagic_lithology, pelagic_grain_size
from uacpy.data.diesing_local import (
    download_diesing_db, fetch_seafloor_lithology,
)
from uacpy.data.graw_local import (
    download_graw_db, fetch_seabed_density, fetch_seabed_density_transect,
)
from uacpy.data.seaice_local import (
    download_seaice_db, fetch_sea_ice_concentration,
    fetch_sea_ice_concentration_transect, SeaIceGrid, sea_ice_grid,
    sea_ice_pixel,
    sea_ice_surface, fetch_sea_ice_surface, sea_ice_surface_transect,
    SEA_ICE_TYPICAL_ROUGHNESS,
)
from uacpy.data.environment import (
    fetch_environment, fetch_bottom, fetch_bottom_transect,
)
from uacpy.data.sources import (DataSource, DataProvenance, SOURCES, citations,
                                provenance_table)
# The cache's own introspection: where it lives and what is in it. The rest of
# `_cache` (staging writes, grid memos, the DATASETS registry) is machinery a
# contributor uses, documented in DEV.md; these three answer questions a *user*
# asks — "where is my cache?", "is dataset X installed?" — and are the only
# part of it a guide page or an example should have to reach for.
from uacpy.data._cache import cache_root, dataset_root, is_installed

__all__ = [
    # offline cache
    'cache_root',
    'dataset_root',
    'is_installed',
    # bathymetry
    'fetch_bathy',
    'fetch_bathy_transect',
    'BathyGrid', 'fetch_bathy_grid',
    'transect_length',
    'transect_waypoints', 'AlongTrack',
    'bathy_transect_sampling',
    # sound speed
    'fetch_ssp',
    'fetch_ssp_transect',
    'ssp_transect_sampling',
    'TSProfile', 'fetch_ts_profile',
    'extend_ssp_below_data',
    'extend_column_to_seafloor',
    'deepen_with_climatology',
    'fetch_ssp_operational',
    'fetch_ssp_transect_operational',
    'fetch_ts_profile_operational',
    'fetch_waves_operational',
    'fetch_ph_operational',
    'fetch_wind',
    'fetch_wind_transect',
    'download_wind_db',
    'SeaStateRecord', 'fetch_waves',
    'fetch_sea_surface',
    'hs_to_pm_wind',
    'ArgoProfile', 'fetch_argo_profile',
    'fetch_ssp_argo',
    'download_glodap_db',
    'PHProfile', 'fetch_ph_profile',
    'fetch_ph',
    # bottom
    'SeabedSample', 'samples_table',
    'grain_size_to_geoacoustics', 'grain_size_from_density',
    'bottom_from_samples',
    'fetch_emodnet_substrate',
    'fetch_bottom',
    'fetch_bottom_transect',
    'fetch_mars_sediment',
    'download_sediment_db',
    'download_emodnet_db',
    'download_globsed_db',
    'download_crust1_db',
    'fetch_sediment_sample',
    'fetch_sediment_thickness',
    'fetch_sediment_thickness_transect',
    'Crust1Profile', 'fetch_crust1_profile',
    'pelagic_lithology',
    'pelagic_grain_size',
    'download_diesing_db',
    'fetch_seafloor_lithology',
    'download_graw_db',
    'fetch_seabed_density',
    'fetch_seabed_density_transect',
    'download_seaice_db',
    'fetch_sea_ice_concentration',
    'fetch_sea_ice_concentration_transect',
    'SeaIceGrid',
    'sea_ice_grid',
    'sea_ice_pixel',
    'sea_ice_surface',
    'fetch_sea_ice_surface', 'sea_ice_surface_transect',
    'SEA_ICE_TYPICAL_ROUGHNESS',
    # capstone
    'fetch_environment',
    # provenance / licensing
    'SOURCES',
    'DataSource',
    'DataProvenance',
    'citations', 'provenance_table',
]
