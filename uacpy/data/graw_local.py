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

from pathlib import Path

import numpy as np

from uacpy.core.environment import BoundaryProperties
from uacpy.core.exceptions import DataFetchError
from uacpy.core.sediment import (DEFAULT_GRAIN_SIZE_MODEL,
                                 DEFAULT_GRAIN_SIZE_ENVIRONMENT,
                                 grain_size_from_density,
                                 grain_size_to_geoacoustics)
from uacpy.data import _cache
from uacpy.data._geo import as_coordinate, checked_n_points, geodesic_waypoints
from uacpy.data._netcdf import NetcdfGrid
from uacpy.data.sediment import (
    range_dependent_bottom_along, water_sound_speed_at,
)

__all__ = ['download_graw_db', 'fetch_seabed_density',
           'fetch_seabed_density_transect', 'fetch_bottom_graw',
           'fetch_bottom_graw_transect']

GRAW_FILE = 'Dataset_S2.nc'
GRAW_URL = 'https://zenodo.org/records/3762390/files/Dataset_S2.nc'

# What the grid holds, measured over its own 6 208 522 finite cells (the other
# third is land): 0.9615 to 2.2107 g/cm³, median 1.4267, and 45.5 % of them
# below the 1.417 g/cm³ where the continental-terrace density relation bottoms
# out. Those are not artefacts — only 0.12 % of cells fall below 1.2, and the
# bulk sits at 1.30-1.42, which is what Hamilton & Bachman's Table IV measures
# for abyssal clay (1.352 and 1.414). They are ordinary deep-ocean mud, which
# is to say **they are not continental terrace**: over this grid the
# abyssal-plain fit represents 65.3 % of cells where the terrace fit reaches
# 40.8 %. So over nearly half the ocean the default conversion returns its
# fine end, and ``grain_size_from_density`` says so and names the environment
# whose range would cover the value.
#
# Where the relation does reach, its sensitivity rises monotonically towards
# the fine end: |dϕ/dρ| is 5.2 ϕ per g/cm³ at -1 ϕ, 6.3 at 1 ϕ, 10.5 at 5 ϕ and
# 32.3 at 9 ϕ (laboratory units), so a density read to ±0.01 g/cm³ fixes the
# grain size to ±0.05 ϕ in sand and ±0.32 ϕ in clay. It diverges only at the
# vertex, which is outside the evaluated range.


def download_graw_db(cache_dir=None, *, timeout=300.0, verbose=False):
    """Download the Graw 2021 seabed bulk-density grid into the cache.

    Writes ``<cache>/graw/Dataset_S2.nc`` (~37 MB, Zenodo) and returns the
    path. Uses curl when available, falling back to the urllib fetcher.
    """
    from uacpy.data._http import download_grid_file
    return download_grid_file(
        'graw', GRAW_URL, GRAW_FILE,
        "downloading Graw 2021 seabed density grid (~37 MB)",
        "Graw density grid cached", cache_dir=cache_dir, timeout=timeout,
        verbose=verbose)


class _GrawGrid(NetcdfGrid):
    """Nearest-cell accessor over the Graw ``z(lat, lon)`` density grid.

    Unlike GlobSed's gridline-registered axes, this 5′ grid is **cell-centre**
    registered: 2160 × 4320 cells whose first centre sits half a cell inside the
    corner, latitude south-up and longitude over ``[-180, 180)``.
    """

    dataset_name = 'graw'

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
    n_points = checked_n_points(n_points, 'fetch_seabed_density_transect')
    lats, lons, ranges_m = geodesic_waypoints(start, end, n_points)
    g = _grid()
    rho = np.array([g.density(la, lo) for la, lo in zip(lats, lons)])
    return np.asarray(ranges_m), rho


def _phi_from_density(rho):
    """Bulk density (g/cm³) → mean grain size (ϕ).

    :func:`uacpy.core.sediment.grain_size_from_density` inverts the very
    relation the forward conversion evaluates, so ϕ → ρ → ϕ returns what it was
    given — and announces the densities it cannot represent, which over this
    grid is nearly half the ocean (see the note at the top of this module).
    """
    return grain_size_from_density(rho)


def fetch_bottom_graw(point, *, roughness=0.0, water_sound_speed=None,
                      model=DEFAULT_GRAIN_SIZE_MODEL, environment=None,
                      timeout=None, verbose=False):
    """Model-ready half-space bottom from the Graw measured-density grid.

    Provenance is catalogue-level: the grid cell under the point supplies the
    value, and no per-cell ``data_point``/``offset_km`` is recorded — unlike
    the sample sources (``grainsize``, ``mars``), which record the sample the
    value came from.

    The density returned is the grid's measured value. The grain size is the
    one *consistent* with it — :func:`uacpy.core.sediment.grain_size_from_density`
    inverts the same Hamilton & Bachman (T) relation that
    :func:`~uacpy.core.sediment.grain_size_to_geoacoustics` evaluates forward,
    so the pair cannot disagree — and the sound speed and attenuation follow
    from that grain size. Below ~1.42 g/cm³ the relation has no solution and
    the grain size is its 9 ϕ end, which is most of the deep ocean; the note at
    the top of this module gives the resolution along the rest of the range.
    ``timeout``/``verbose`` are accepted (and ignored — this backend is
    offline) for signature uniformity with the network bottom fetchers.
    ``water_sound_speed`` (m/s) scales the velocity ratio to the in-situ
    near-seabed water; ``None`` uses the Hamilton reference. ``model`` picks
    the relations that turn that grain size into sound speed and
    attenuation (``'hamilton'`` or ``'apl-uw'``); the density stays the
    grid's measured value, and the grain size is always Hamilton's
    inversion of it.
    """
    rho = fetch_seabed_density(point)
    phi = _phi_from_density(rho)
    geo = grain_size_to_geoacoustics(phi, model=model,
                                     environment=environment or DEFAULT_GRAIN_SIZE_ENVIRONMENT,
                                     water_sound_speed=water_sound_speed)
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
                               model=DEFAULT_GRAIN_SIZE_MODEL, environment=None,
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
            water_sound_speed=water_sound_speed_at(water_sound_speed, la, lo),
            model=model, environment=environment),
        start, end, n_points, source_label='Graw density grid',
        max_points=max_points,
    )
