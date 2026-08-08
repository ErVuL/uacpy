"""Offline NRL/Graw predicted seabed bulk density (Zenodo, CC-BY 4.0).

``install.sh --data graw`` downloads the Graw, Wood & Phrampus (2021) global
machine-learning prediction of **surficial seabed bulk density (g/cm³)** — a
5-arc-minute ``z(lat, lon)`` grid (Zenodo record 3762390, ``Dataset_S2.nc``) —
into the cache. This module samples it locally.

This is the one *measured/predicted* continuous density product in the layer:
the grain-size backends derive density from ϕ via the Hamilton table, whereas
here the density is the data and the remaining geoacoustics (sound speed,
attenuation) are ϕ-derived by **inverting** that same table — so
:func:`fetch_bottom_graw` emits a half-space whose density is the measured
grid value and whose speed/attenuation are consistent with it.
"""

import os
from pathlib import Path

import numpy as np

from uacpy._log import log_message
from uacpy.core.environment import BoundaryProperties
from uacpy.core.exceptions import DataFetchError
from uacpy.core.sediment import _HB_PHI, _HB_RHO, grain_size_to_geoacoustics
from uacpy.data import _cache
from uacpy.data._geo import as_coordinate
from uacpy.data._netcdf import NetcdfGrid
from uacpy.data.sediment import (
    range_dependent_bottom_along, water_sound_speed_at,
)

__all__ = ['download_graw_db', 'fetch_seabed_density',
           'fetch_seabed_density_transect', 'fetch_bottom_graw',
           'fetch_bottom_graw_transect']

GRAW_FILE = 'Dataset_S2.nc'
GRAW_URL = 'https://zenodo.org/records/3762390/files/Dataset_S2.nc'

# Hamilton & Bachman density column reversed to be strictly increasing for the
# ρ → ϕ inversion (the table runs coarse/dense → fine/light).
_RHO_ASC = _HB_RHO[::-1]
_PHI_DESC = _HB_PHI[::-1]



def download_graw_db(cache_dir=None, *, timeout=300.0, verbose=False):
    """Download the Graw 2021 seabed bulk-density grid into the cache.

    Writes ``<cache>/graw/Dataset_S2.nc`` (~37 MB, Zenodo) and returns the
    path. Uses curl when available, falling back to the urllib fetcher.
    """
    from uacpy.data._http import curl_download, http_get
    dest = Path(cache_dir) if cache_dir else _cache.dataset_root('graw')
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / GRAW_FILE
    log_message('graw', "downloading Graw 2021 seabed density grid (~37 MB)",
                verbose=verbose)
    if not curl_download(GRAW_URL, out, timeout=timeout, verbose=verbose):
        part = Path(str(out) + '.part')
        part.write_bytes(http_get(GRAW_URL, timeout=timeout, verbose=verbose,
                                  source='graw'))
        os.replace(part, out)
    _cache.invalidate_grids()
    log_message('graw', f"Graw density grid cached → {out}", verbose=verbose)
    return out


class _GrawGrid(NetcdfGrid):
    """Nearest-cell accessor over the Graw ``z(lat, lon)`` density grid."""

    def __init__(self, path):
        try:
            super().__init__(path)
            self._z = self.var('z')
        except KeyError as exc:
            raise DataFetchError(
                f"Graw NetCDF {Path(path).name} is missing an expected "
                f"variable ({exc}); its schema may have changed.",
                remediation="Re-run ./install.sh --data graw.",
            ) from exc

    def density(self, lat, lon):
        return self.cell(self._z, self.row(lat), self.col(lon))


def _grid():
    return _cache.cached_grid('graw', GRAW_FILE, _GrawGrid)


def fetch_seabed_density(point):
    """Predicted surficial seabed bulk density (g/cm³) at a ``(lat, lon)`` point.

    The prediction is continuous over the ocean; a non-finite cell raises
    ``DataFetchError``. Note the model is trained on marine sediments — values
    over land are extrapolation artefacts, not soil densities.
    """
    lat, lon = as_coordinate(point)
    rho = _grid().density(lat, lon)
    if not np.isfinite(rho):
        raise DataFetchError(
            f"Graw grid has no seabed density at {lat:.3f}, {lon:.3f}.",
            remediation="Pick another point, or supply a bottom directly.",
        )
    return float(rho)


def fetch_seabed_density_transect(start, end, n_points=6):
    """``(ranges_m, density_gcm3)`` sampled along ``start`` → ``end``.

    ``density_gcm3`` is ``NaN`` at any waypoint without a finite grid value.
    """
    from uacpy.data._geo import geodesic_waypoints
    lats, lons, ranges_m = geodesic_waypoints(start, end, n_points)
    g = _grid()
    rho = np.array([g.density(la, lo) for la, lo in zip(lats, lons)])
    return np.asarray(ranges_m), rho


def _phi_from_density(rho):
    """Invert the Hamilton & Bachman ρ(ϕ) table: bulk density (g/cm³) → mean
    grain size (ϕ), clamped to the table's end members."""
    return float(np.interp(rho, _RHO_ASC, _PHI_DESC))


def fetch_bottom_graw(point, *, roughness=0.0, water_sound_speed=None,
                      timeout=None, verbose=False):
    """Model-ready half-space bottom from the Graw measured-density grid.

    The density is the grid value; the grain size is recovered by inverting
    the Hamilton ρ(ϕ) table and yields the consistent sound speed and
    attenuation via :func:`uacpy.core.sediment.grain_size_to_geoacoustics`.
    ``timeout``/``verbose`` are accepted (and ignored — this backend is
    offline) for signature uniformity with the network bottom fetchers.
    ``water_sound_speed`` (m/s) scales the velocity ratio to the in-situ
    near-seabed water; ``None`` uses the Hamilton reference.
    """
    rho = fetch_seabed_density(point)
    phi = _phi_from_density(rho)
    geo = grain_size_to_geoacoustics(phi, water_sound_speed=water_sound_speed)
    return BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=geo['sound_speed'],
        density=rho,
        attenuation=geo['attenuation'],
        grain_size_phi=phi,
        roughness=roughness,
    )


def fetch_bottom_graw_transect(start, end, *, n_points=6, max_points=None,
                               roughness=0.0, water_sound_speed=None,
                               timeout=None, verbose=False):
    """Range-dependent bottom from the Graw grid along ``start`` → ``end``.

    ``water_sound_speed`` also takes a ``(lat, lon) -> m/s`` callable, so each
    column scales to the water over its own seafloor. ``timeout``/``verbose``
    are accepted (and ignored — this backend is offline) for signature
    uniformity with the network bottom fetchers.
    """
    return range_dependent_bottom_along(
        lambda la, lo: fetch_bottom_graw(
            (la, lo), roughness=roughness,
            water_sound_speed=water_sound_speed_at(water_sound_speed, la, lo)),
        start, end, n_points, source_label='Graw density grid',
        max_points=max_points,
    )
