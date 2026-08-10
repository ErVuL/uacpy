"""On-demand external-data layer — GPS coordinates → ``Environment`` inputs.

- **Bathymetry** (GEBCO, static): a single water depth or a range-dependent
  transect for ``Environment(bathymetry=...)``.
- **Sound speed** (WOA23 climatology, date/month-aware; Copernicus Marine
  operational; or the nearest Argo float profile): a depth-vs-c profile for
  ``Environment(ssp=...)``, plus the raw T/S column and a Francois-Garrison
  absorption helper.
- **Bottom**: a grain-size (ϕ) / sediment-class → ``BoundaryProperties``
  conversion, *or* a fetched seafloor — EMODnet substrate (European seas), the
  NCEI grain-size DB, the AusSeabed MARS samples (Australian margin), the
  Diesing 2020 deep-sea map, the Graw 2021 seabed-density grid, a pelagic
  depth/latitude fallback (all surficial), or GlobSed thickness + CRUST1.0 → a
  layered elastic bottom for low-frequency work.
- **Surface**: NSIDC sea-ice concentration → an elastic ice-canopy
  ``BoundaryProperties`` (:func:`fetch_sea_ice_surface`), so an ice-covered
  point replaces the free surface with a pack-ice boundary.
- **Capstone**: :func:`fetch_environment` assembles them
  (``surface_sources='seaice'`` adds the ice surface). Each axis is a literal
  (``ssp=`` / ``bathymetry=`` / ``bottom=`` / ``surface=`` / ``altimetry=``)
  and/or fetched from ordered-fallback ``*_sources`` (source first, literal as
  fallback; ``'auto'`` = best available, ``'local'`` = local data only, no
  network); fetching is cache-first (a locally installed dataset is sampled
  before any network call).

Examples
--------
>>> import uacpy
>>> from uacpy.data import fetch_environment
>>> env = uacpy.data.fetch_environment((43.2, 7.5), date='2026-06-14',
...                                    bottom='sand')          # doctest: +SKIP
"""

from uacpy.data.bathymetry import (
    fetch_bathy, fetch_bathy_transect, fetch_bathy_grid, transect_length,
)
from uacpy.data.sound_speed import (
    fetch_ssp, fetch_ssp_transect, fetch_ts_profile,
)
from uacpy.data.copernicus import (
    fetch_ssp_operational, fetch_ssp_transect_operational,
    fetch_ts_profile_operational, fetch_waves_operational,
    fetch_ph_operational,
)
from uacpy.data.wind_live import fetch_wind, fetch_wind_transect
from uacpy.data.wind_local import download_wind_db
from uacpy.data.waves import fetch_waves
from uacpy.data.sea_surface import fetch_sea_surface
from uacpy.data.argo import fetch_argo_profile, fetch_ssp_argo
from uacpy.data.absorption import build_francois_garrison
from uacpy.data.glodap_local import (
    download_glodap_db, fetch_ph_profile, fetch_ph,
)
from uacpy.data.sediment import (
    grain_size_to_geoacoustics, bottom_from_grain_size, bottom_from_class,
)
from uacpy.data.seabed import (
    fetch_seabed_substrate, fetch_bottom, fetch_bottom_transect,
)
from uacpy.data.mars import (
    fetch_mars_sediment, fetch_bottom_mars, fetch_bottom_mars_transect,
)
from uacpy.data.sediment_db import (
    download_sediment_db, fetch_sediment_sample, fetch_bottom_local,
    fetch_bottom_local_transect,
)
from uacpy.data.emodnet_local import download_emodnet_db
from uacpy.data.globsed_local import (
    download_globsed_db, fetch_sediment_thickness, fetch_sediment_thickness_transect,
)
from uacpy.data.crust1_local import (
    download_crust1_db, fetch_crust1_profile, fetch_bottom_crust1,
    fetch_bottom_crust1_transect,
)
from uacpy.data.pelagic import (
    pelagic_lithology, pelagic_grain_size, fetch_bottom_pelagic,
    fetch_bottom_pelagic_transect,
)
from uacpy.data.diesing_local import (
    download_diesing_db, fetch_seafloor_lithology, fetch_bottom_diesing,
    fetch_bottom_diesing_transect,
)
from uacpy.data.graw_local import (
    download_graw_db, fetch_seabed_density, fetch_seabed_density_transect,
    fetch_bottom_graw, fetch_bottom_graw_transect,
)
from uacpy.data.seaice_local import (
    download_seaice_db, fetch_sea_ice_concentration,
    fetch_sea_ice_concentration_transect, sea_ice_grid, sea_ice_pixel,
    sea_ice_surface, fetch_sea_ice_surface, sea_ice_surface_transect,
)
from uacpy.data.environment import fetch_environment
from uacpy.data.sources import DataSource, DataProvenance, SOURCES, citations

__all__ = [
    # bathymetry
    'fetch_bathy',
    'fetch_bathy_transect',
    'fetch_bathy_grid',
    'transect_length',
    # sound speed
    'fetch_ssp',
    'fetch_ssp_transect',
    'fetch_ts_profile',
    'fetch_ssp_operational',
    'fetch_ssp_transect_operational',
    'fetch_ts_profile_operational',
    'fetch_waves_operational',
    'fetch_ph_operational',
    'fetch_wind',
    'fetch_wind_transect',
    'download_wind_db',
    'fetch_waves',
    'fetch_sea_surface',
    'fetch_argo_profile',
    'fetch_ssp_argo',
    'build_francois_garrison',
    'download_glodap_db',
    'fetch_ph_profile',
    'fetch_ph',
    # bottom
    'grain_size_to_geoacoustics',
    'bottom_from_grain_size',
    'bottom_from_class',
    'fetch_seabed_substrate',
    'fetch_bottom',
    'fetch_bottom_transect',
    'fetch_mars_sediment',
    'fetch_bottom_mars',
    'fetch_bottom_mars_transect',
    'download_sediment_db',
    'download_emodnet_db',
    'download_globsed_db',
    'download_crust1_db',
    'fetch_sediment_sample',
    'fetch_bottom_local',
    'fetch_bottom_local_transect',
    'fetch_sediment_thickness',
    'fetch_sediment_thickness_transect',
    'fetch_crust1_profile',
    'fetch_bottom_crust1',
    'fetch_bottom_crust1_transect',
    'pelagic_lithology',
    'pelagic_grain_size',
    'fetch_bottom_pelagic',
    'fetch_bottom_pelagic_transect',
    'download_diesing_db',
    'fetch_seafloor_lithology',
    'fetch_bottom_diesing',
    'fetch_bottom_diesing_transect',
    'download_graw_db',
    'fetch_seabed_density',
    'fetch_seabed_density_transect',
    'fetch_bottom_graw',
    'fetch_bottom_graw_transect',
    'download_seaice_db',
    'fetch_sea_ice_concentration',
    'fetch_sea_ice_concentration_transect',
    'sea_ice_grid',
    'sea_ice_pixel',
    'sea_ice_surface',
    'fetch_sea_ice_surface', 'sea_ice_surface_transect',
    # capstone
    'fetch_environment',
    # provenance / licensing
    'SOURCES',
    'DataSource',
    'DataProvenance',
    'citations',
]
