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
from typing import Optional

import numpy as np

from uacpy.core.environment import BoundaryProperties
from uacpy.core.exceptions import DataFetchError
from uacpy.core.sediment import (DEFAULT_GRAIN_SIZE_MODEL,
                                 DEFAULT_GRAIN_SIZE_ENVIRONMENT,
                                 grain_size_from_density,
                                 grain_size_to_geoacoustics)
from uacpy.data import _cache
from uacpy.core.geo import as_coordinate, geodesic_waypoints
from uacpy.data._geo import AlongTrack, checked_max_distance, checked_n_points
from uacpy.data._netcdf import NetcdfGrid
from uacpy.data.sources import SOURCES, DataProvenance
from uacpy.data._provenance_notice import one_provenance_notice

__all__ = ['download_graw_db', 'fetch_seabed_density',
           'fetch_seabed_density_transect', 'fetch_bottom_graw']

GRAW_FILE = 'Dataset_S2.nc'
GRAW_URL = 'https://zenodo.org/records/3762390/files/Dataset_S2.nc'

# What the grid holds, measured over its own 6 208 522 finite cells (the other
# third is land): 0.9615 to 2.2107 g/cm³, median 1.4267, and 59.9 % of them
# below 1.448 g/cm³, the density the continental-terrace relation reaches at its
# 9 ϕ end against the default water (its vertex, 1.418, lies past 9 ϕ). Those are not artefacts — only 0.12 % of cells fall below 1.2, and the
# bulk sits at 1.30-1.42, which is what Hamilton & Bachman's Table IV measures
# for abyssal clay (1.352 and 1.414). They are ordinary deep-ocean mud, which
# is to say **they are not continental terrace**: over this grid the
# abyssal-plain fit represents 65.7 % of cells where the terrace fit reaches
# 40.1 %. So over about 60 % of the ocean the default conversion returns its
# fine end, and ``grain_size_from_density`` says so and names the environment
# whose range would cover the value.
#
# Where the relation does reach, its sensitivity rises monotonically towards
# the fine end: |dϕ/dρ| is 5.2 ϕ per g/cm³ at -1 ϕ, 6.3 at 1 ϕ, 10.5 at 5 ϕ and
# 32.3 at 9 ϕ (laboratory units), so a density read to ±0.01 g/cm³ fixes the
# grain size to ±0.05 ϕ in sand and ±0.32 ϕ in clay. It diverges only at the
# vertex, which is outside the evaluated range.


def download_graw_db(cache_dir=None, *, url: Optional[str] = None,
                     timeout=300.0, verbose=False):
    """Download the Graw 2021 seabed bulk-density grid into the cache.

    Writes ``<cache>/graw/Dataset_S2.nc`` (~37 MB, Zenodo) and returns the
    path. Uses curl when available, falling back to the urllib fetcher.

    ``url`` fetches that address instead of :data:`GRAW_URL` — a mirror, or
    a copy staged on an http server of your own. What is written and how it
    is read are the same whatever address served it.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory to write into; ``None`` is the dataset's own directory under
        the cache root (:func:`dataset_root`).
    url : str, optional
        The one address to fetch; ``None`` is :data:`GRAW_URL`.
    timeout : float, optional
        Network timeout in seconds. Default 300.
    verbose : bool or str, optional
        Logging gate passed through to ``log_message``.
    """
    from uacpy.data._http import download_grid_file
    return download_grid_file(
        'graw', url or GRAW_URL, GRAW_FILE,
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
            raise _cache.UnreadableCacheError(
                f"Graw NetCDF {Path(path).name} is missing an expected "
                f"variable ({exc}); its schema may have changed.",
                remediation="Re-run ./install.sh --data graw.",
            ) from exc

    def density(self, lat, lon):
        return self.cell(self._z, self.row(lat), self.col(lon))


def _grid():
    return _cache.cached_grid('graw', GRAW_FILE, _GrawGrid)


def graw_node(lat, lon, *, who, max_distance_km=None):
    """The ``'graw'`` provenance of the node a ``(lat, lon)`` read lands on,
    through the offset rule."""
    return _grid().node_provenance('graw', lat, lon, who=who,
                                   max_distance_km=max_distance_km)


def fetch_seabed_density(point, *, max_distance_km=None):
    """Predicted surficial seabed bulk density (g/cm³) at a ``(lat, lon)`` point.

    The prediction is continuous over the ocean; land cells hold no value, and
    a non-finite cell raises ``DataFetchError``.

    The value is raw and carries no provenance;
    :func:`fetch_bottom_graw` returns the carrier that records
    it in ``.data_sources``.

    Parameters
    ----------
    point : (lat, lon)
        Site coordinates in decimal degrees.
    max_distance_km : float, optional
        Refuse a node farther than this (km) from ``point`` (the offset rule);
        ``None`` (default) sets no limit beyond its ``ProvenanceWarning``.
    """
    lat, lon = as_coordinate(point)
    graw_node(lat, lon, who='fetch_seabed_density',
              max_distance_km=checked_max_distance(max_distance_km,
                                                   'fetch_seabed_density'))
    rho = _grid().density(lat, lon)
    if not np.isfinite(rho):
        raise DataFetchError(
            f"Graw grid has no seabed density at {lat:.3f}, {lon:.3f}.",
            remediation="Pick another point, or supply a bottom directly.",
        )
    return float(rho)


@one_provenance_notice(subject='the samples',
                       record="the result's .provenance")
def fetch_seabed_density_transect(start, end, *,
                                  n_points=6, max_distance_km=None) -> AlongTrack:
    """Graw seabed bulk density (g/cm³) sampled along ``start`` → ``end``,
    as an :class:`~uacpy.data.AlongTrack` (``'seabed_density'``) with the
    ``'graw'`` provenance; ``NaN`` at any waypoint without a finite grid
    value.

    Parameters
    ----------
    start, end : (lat, lon)
        Transect endpoints in decimal degrees.
    n_points : int, optional
        Waypoints along the great circle. Default 6.
    max_distance_km : float, optional
        Refuse a waypoint whose node stands farther than this (km) from it;
        ``None`` (default) sets no limit beyond the per-waypoint warning.
    """
    n_points = checked_n_points(n_points, 'fetch_seabed_density_transect')
    limit = checked_max_distance(max_distance_km, 'fetch_seabed_density_transect')
    lats, lons, ranges_m = geodesic_waypoints(start, end, n_points)
    for la, lo in zip(lats, lons):
        graw_node(la, lo, who='fetch_seabed_density_transect', max_distance_km=limit)
    g = _grid()
    rho = np.array([g.density(la, lo) for la, lo in zip(lats, lons)])
    return AlongTrack(ranges=np.asarray(ranges_m), lats=np.asarray(lats),
                      lons=np.asarray(lons), data=rho, unit='g/cm3',
                      quantity='seabed_density',
                      provenance=DataProvenance(source=SOURCES['graw']))


def _phi_from_density(rho, hamilton_fit=None):
    """Bulk density (g/cm³) → mean grain size (ϕ), under ``hamilton_fit``'s
    Hamilton & Bachman fit.

    :func:`uacpy.core.sediment.grain_size_from_density` inverts the very
    relation the forward conversion evaluates, so ϕ → ρ → ϕ returns what it was
    given — and announces the densities it cannot represent, which over this
    grid is about 60 % of the ocean (see the note at the top of this module).
    The fit is the one the forward conversion uses: Hamilton & Bachman (1982,
    Appendix) give separate terrace, abyssal-hill and abyssal-plain
    regressions, so inverting with one and converting forward with another
    returned a grain size, sound speed and attenuation of a different seabed
    (abyssal plain at 1.45 g/cm³: ϕ 9.0 against 7.45, 16 m/s slow).
    """
    return grain_size_from_density(
        rho, hamilton_fit=hamilton_fit or DEFAULT_GRAIN_SIZE_ENVIRONMENT)


def fetch_bottom_graw(point, *, roughness=0.0, water_sound_speed=None,
                      model=DEFAULT_GRAIN_SIZE_MODEL, hamilton_fit=None,
                      max_distance_km=None, timeout=None, verbose=False):
    """Model-ready half-space bottom from the Graw measured-density grid.

    The returned bottom carries a ``graw`` ``DataProvenance`` with the
    requested point and the grid node read as ``data_point`` (the offset
    rule: ``max_distance_km`` refuses a node farther than that, km).

    The density returned is the grid's measured value. The grain size is the
    one *consistent* with it — :func:`uacpy.core.sediment.grain_size_from_density`
    inverts the same Hamilton & Bachman (T) relation that
    :func:`~uacpy.core.sediment.grain_size_to_geoacoustics` evaluates forward,
    so the pair cannot disagree — and the sound speed and attenuation follow
    from that grain size. Below 1.448 g/cm³ (the relation's 9 ϕ density against
    the default water) the grain size is held at its 9 ϕ end, which is most
    of the deep ocean; the note at the top of this module gives the resolution
    along the rest of the range.
    ``timeout``/``verbose`` are accepted (and ignored — this backend is
    offline) for signature uniformity with the network bottom fetchers.
    ``water_sound_speed`` (m/s) scales the velocity ratio to the in-situ
    near-seabed water; ``None`` uses the Hamilton reference. ``model`` picks
    the relations that turn that grain size into sound speed and
    attenuation (``'hamilton'`` or ``'apl-uw'``); the density stays the
    grid's measured value, and the grain size is always Hamilton's
    inversion of it, under the same ``hamilton_fit`` fit the forward step
    uses.
    """
    from uacpy.core.sediment import canonical_grain_size_selection
    model, hamilton_fit = canonical_grain_size_selection(
        model, hamilton_fit, who='fetch_bottom_graw')
    lat, lon = as_coordinate(point)
    prov = graw_node(lat, lon, who='fetch_bottom_graw',
                     max_distance_km=checked_max_distance(max_distance_km,
                                                          'fetch_bottom_graw'))
    rho = fetch_seabed_density(point)
    hamilton_fit = hamilton_fit or DEFAULT_GRAIN_SIZE_ENVIRONMENT
    phi = _phi_from_density(rho, hamilton_fit)
    geo = grain_size_to_geoacoustics(phi, model=model,
                                     hamilton_fit=hamilton_fit,
                                     water_sound_speed=water_sound_speed)
    return BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=geo['sound_speed'],
        density=rho,
        attenuation=geo['attenuation'],
        grain_size_phi=phi,
        roughness=roughness,
        data_sources=(prov,),
    )
