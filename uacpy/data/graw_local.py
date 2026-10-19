"""Offline NRL/Graw predicted seabed bulk density (Zenodo, CC-BY 4.0).

``install.sh --data graw`` downloads the Graw, Wood & Phrampus (2021) global
random-forest prediction of **seafloor bulk density (g/cm³)** — a 5-arc-minute
``z(lat, lon)`` grid (Zenodo record 3762390, ``Dataset_S2.nc``) trained on
densities measured on DSDP, ODP and IODP drill cores (gamma-ray attenuation and
moisture-and-density measurements, the record's ``Dataset_S1``) — into the
cache. This module samples it locally.

This is the one *measured/predicted* continuous density product in the layer:
the grain-size backends derive density from ϕ, whereas here the density is the
data and the grain size is inferred from it. A drill-core density is a *bulk*
value in APL-UW TR 9407's sense, one from below the top few centimetres
(TR 9407 Introduction §C, the ``blos94`` input method 6; §IV.A.2, p. IV-5:
"many geoacoustic measurements consist of averages covering a depth of
several tens of centimeters"), and TR 9407 §IV.A.5 (p. IV-12 to IV-13) gives
the route for it: the mean grain size from the inverted
Hamilton-Bachman regression, its Eq. (12) for the bulk density ratio —
:func:`~uacpy.core.sediment.grain_size_from_density`'s continental-terrace
inversion — and every remaining input from that grain size (§IV.A.4).
:func:`fetch_bottom_graw` takes the sound speed and attenuation from it, and
the density too under ``model='apl-uw'``, whose relations describe the upper
few centimetres. Below 1.448 g/cm³ (the relation's 9 ϕ density against the
default water) the grain size is held at its 9 ϕ end, so there the speed and
attenuation are the 9 ϕ values whatever the measured density.
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
from uacpy.core.geo import as_coordinate
from uacpy.data._geo import (AlongTrack, along_track_waypoints,
                             checked_max_distance)
from uacpy.data._netcdf import NetcdfGrid
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
    :func:`uacpy.data.fetch_bottom` with ``source='graw'`` returns the
    carrier that records it in ``.data_sources``.

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
    return _finite_density(lat, lon)


def _finite_density(lat, lon):
    """The grid's density (g/cm³) at ``(lat, lon)``; ``DataFetchError`` where
    the cell holds no value."""
    rho = _grid().density(lat, lon)
    if not np.isfinite(rho):
        raise DataFetchError(
            f"Graw grid has no seabed density at {lat:.3f}, {lon:.3f}.",
            remediation="Pick another point, or supply a bottom directly.",
        )
    return float(rho)


@one_provenance_notice(subject='the samples',
                       record="the result's .provenance")
def fetch_seabed_density_transect(start, end, *, n_points=None,
                                  max_points=None,
                                  max_distance_km=None) -> AlongTrack:
    """Graw seabed bulk density (g/cm³) sampled along ``start`` → ``end``,
    as an :class:`~uacpy.data.AlongTrack` (``'seabed_density'``) with one
    ``'graw'`` record per waypoint (the node read and its offset); ``NaN``
    at any waypoint without a finite grid value.

    Parameters
    ----------
    start, end : (lat, lon)
        Transect endpoints in decimal degrees.
    n_points : int or 'auto', optional
        Waypoints along the great circle, used as given; ``None`` (default)
        is 6 (uacpy guidance), within ``max_points`` when that is set.
        ``'auto'`` probes ``max_points`` waypoints and keeps one per 5′ grid node
        crossed, endpoints anchored.
    max_points : int, optional
        The waypoints ``'auto'`` probes; ``None`` (default) is 1000 (uacpy
        guidance). An explicit ``n_points`` above an explicit
        ``max_points`` is refused.
    max_distance_km : float, optional
        Refuse a waypoint whose node stands farther than this (km) from it;
        ``None`` (default) sets no limit beyond the per-waypoint warning.
    """
    who = 'fetch_seabed_density_transect'
    limit = checked_max_distance(max_distance_km, who)
    lats, lons, ranges = along_track_waypoints(
        start, end, n_points, max_points, who=who,
        cell_of=lambda la, lo: (_grid().row(la), _grid().col(lo)))
    records = tuple(graw_node(la, lo, who=who, max_distance_km=limit)
                    for la, lo in zip(lats, lons))
    g = _grid()
    rho = np.array([g.density(la, lo) for la, lo in zip(lats, lons)])
    return AlongTrack(ranges=np.asarray(ranges), lats=np.asarray(lats),
                      lons=np.asarray(lons), data=rho, unit='g/cm3',
                      quantity='seabed_density', provenance=records)


def _phi_from_density(rho, hamilton_fit=None, water_density=None):
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
        rho, water_density=water_density,
        hamilton_fit=hamilton_fit or DEFAULT_GRAIN_SIZE_ENVIRONMENT)


def fetch_bottom_graw(point, *, roughness=0.0, water_sound_speed=None,
                      water_density=None,
                      model=DEFAULT_GRAIN_SIZE_MODEL, hamilton_fit=None,
                      max_distance_km=None):
    """Model-ready half-space bottom from the Graw measured-density grid.

    The returned bottom carries a ``graw`` ``DataProvenance`` with the
    requested point and the grid node read as ``data_point`` (the offset
    rule: ``max_distance_km`` refuses a node farther than that, km).

    The grain size is the one the grid's bulk density gives through
    :func:`uacpy.core.sediment.grain_size_from_density` — the inverse of the
    Hamilton & Bachman relation :func:`~uacpy.core.sediment.grain_size_to_geoacoustics`
    evaluates forward, under the same ``hamilton_fit``; for the default
    continental-terrace fit it is APL-UW TR 9407's Eq. (12), the route TR 9407
    §IV.A.5 gives for a bulk density (see the module docstring) — and the
    sound speed and attenuation follow from that grain size. Below
    1.448 g/cm³ (the relation's 9 ϕ density against the default water) the
    grain size is held at its 9 ϕ end, which is most of the deep ocean; the
    note at the top of this module gives the resolution along the rest of
    the range.
    ``water_sound_speed`` (m/s) scales the velocity ratio to the in-situ
    near-seabed water; ``None`` uses the Hamilton reference.
    ``water_density`` (g/cm³) is the near-seabed water the measured density
    is a ratio to in the grain-size inversion; ``None`` uses the package's
    one water (1.027). ``model`` picks
    the relations that turn that grain size into geoacoustics (``'hamilton'``
    or ``'apl-uw'``). Under ``'hamilton'`` the density returned is the
    grid's measured value, the bulk quantity those relations describe.
    Under ``'apl-uw'`` it is APL-UW's density at the grain size, as TR 9407
    §IV.A.5 takes every input from the grain size once a bulk value has
    given it: APL-UW's ratios describe the upper few centimetres and "tend
    to be smaller than the bulk values reported in the literature" (TR 9407
    §IV.A.3, p. IV-5) — 1.176 g/cm³ at 9 ϕ against a measured 1.43, for one.
    """
    from uacpy.core.sediment import _canonical_grain_size_selection
    model, hamilton_fit = _canonical_grain_size_selection(
        model, hamilton_fit, who='fetch_bottom_graw')
    lat, lon = as_coordinate(point)
    prov = graw_node(lat, lon, who='fetch_bottom_graw',
                     max_distance_km=checked_max_distance(max_distance_km,
                                                          'fetch_bottom_graw'))
    rho = _finite_density(lat, lon)
    hamilton_fit = hamilton_fit or DEFAULT_GRAIN_SIZE_ENVIRONMENT
    phi = _phi_from_density(rho, hamilton_fit, water_density=water_density)
    geo = grain_size_to_geoacoustics(phi, model=model,
                                     hamilton_fit=hamilton_fit,
                                     water_sound_speed=water_sound_speed,
                                     water_density=water_density)
    return BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=geo['sound_speed'],
        # TR 9407 §IV.A.5: a bulk density gives the grain size, and APL-UW's
        # surficial inputs, its density included, then follow from it.
        density=geo['density'] if model == 'apl-uw' else rho,
        attenuation=geo['attenuation'],
        grain_size_phi=phi,
        roughness=roughness,
        data_sources=(prov,),
    )
