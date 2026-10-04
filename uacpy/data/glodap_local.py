"""Offline GLODAPv2.2016b seawater pH (GLODAP, CC-BY 4.0).

``install.sh --data glodap`` downloads the GLODAPv2.2016b Mapped Climatology pH
field — a global 1° × 1°, 33-level ``pH(depth, lat, lon)`` grid on the total
scale at in-situ temperature and pressure (Lauvset et al. 2016) — into the
cache. This module samples it locally.

pH is the one Francois-Garrison absorption input WOA23 does not carry, so
without it :func:`uacpy.data.fetch_environment` falls back to a constant
(8.0, ``core.constants.REFERENCE_PH``, on the NBS scale). A cached GLODAP
grid replaces that constant with the real in-situ column, which the
absorption takes as ``(depth, pH)`` pairs on GLODAP's own levels.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from uacpy._log import log_message
from uacpy.core._export import ExportRecord
from uacpy.core.exceptions import DataFetchError
from uacpy.data import _cache
from uacpy.core.geo import as_coordinate
from uacpy.data._geo import checked_max_distance
from uacpy.data._netcdf import NetcdfGrid, netcdf_lock
from uacpy.data.sources import DataProvenance
from uacpy.core.constants import PH_MAX, PH_MIN

__all__ = ['download_glodap_db', 'fetch_ph_profile', 'fetch_ph']

GLODAP_FILE = 'GLODAPv2.2016b.pHtsinsitutp.nc'
GLODAP_TARBALL = 'GLODAPv2.2016b.MappedProduct.tar.gz'
GLODAP_URL = ('https://glodap.info/glodap_files/v2.2023/'
              'GLODAPv2.2016b.MappedProduct.tar.gz')
# The pH variable inside the mapped product; the depth axis is a separate
# coordinate variable (the mapped files name it ``Depth``).
_PH_VARS = ('pHtsinsitutp', 'pHts25p0', 'ph')
_DEPTH_VARS = ('depth', 'depth_surface')



def download_glodap_db(cache_dir=None, *, url: Optional[str] = None,
                       timeout=600.0, verbose=False):
    """Download the GLODAPv2.2016b Mapped pH field into the cache.

    Fetches the mapped-product tarball (~211 MB), extracts only the in-situ pH
    grid to ``<cache>/glodap/GLODAPv2.2016b.pHtsinsitutp.nc`` and discards the
    rest, then returns the path (:func:`uacpy.data._http.download_member`:
    curl first, then urllib).

    ``url`` fetches that address instead of :data:`GLODAP_URL` — a mirror,
    or a copy staged on an http server of your own. What is written and
    how it is read are the same whatever address served it.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory to write into; ``None`` is the dataset's own directory under
        the cache root (:func:`dataset_root`).
    url : str, optional
        The one address to fetch; ``None`` is :data:`GLODAP_URL`.
    timeout : float, optional
        Network timeout in seconds. Default 600.
    verbose : bool or str, optional
        Logging gate passed through to ``log_message``.
    """
    from uacpy.data._http import download_member
    dest = _cache.prepare_download(
        'glodap', "downloading GLODAPv2.2016b mapped product (~211 MB)",
        cache_dir=cache_dir, verbose=verbose)
    # Only the pH member is kept; the 211 MB tarball is staged beside the
    # cache and discarded (download_member).
    out = download_member('glodap', url or GLODAP_URL, GLODAP_FILE,
                          dest / GLODAP_FILE, timeout=timeout,
                          verbose=verbose)
    _cache.invalidate_grids()
    log_message('glodap', f"GLODAP pH grid cached → {out}", verbose=verbose)
    return out


class _GlodapGrid(NetcdfGrid):
    """Column accessor over the GLODAP ``pH(depth, lat, lon)`` grid.

    Reuses :class:`NetcdfGrid` for nearest lat/lon indexing and binds the depth
    axis plus the pH variable for whole-column reads.

    The mapped product's axes are 1° cell centres with latitude running south-up
    (−89.5 → 89.5) and longitude on a **shifted** origin, 20.5 → 379.5 °E — it
    is neither ``[-180, 180)`` nor ``[0, 360)``. ``NetcdfGrid.col`` wraps a query
    modulo 360 onto whatever origin the file declares, so no conversion is needed
    here; the 33 levels of the depth axis run 0 → 5500 m.
    """

    dataset_name = 'glodap'

    def __init__(self, path):
        try:
            super().__init__(path)
            self._ph = self.var(*_PH_VARS)
            with netcdf_lock:
                self._depth = np.asarray(self.var(*_DEPTH_VARS)[:], dtype=float)
        except KeyError as exc:
            raise DataFetchError(
                f"GLODAP NetCDF {Path(path).name} is missing an expected "
                f"variable ({exc}); its schema may have changed.",
                remediation="Re-run ./install.sh --data glodap.",
            ) from exc

    def profile(self, lat, lon):
        """``(depths_m, pH)`` at the nearest cell, trimmed to finite levels.

        The mapped product masks land and sub-seafloor levels as ``_FillValue``;
        drop them so the returned column runs surface → deepest analysed level.
        """
        # np.asarray() on a netCDF4 masked array discards the mask and exposes
        # the raw value, which for this file is the declared _FillValue of -999
        # — a number np.isfinite then accepts as a real pH. Fill *through* the
        # mask instead.
        # Same lock and typing as NetcdfGrid.cell: this reads the netCDF
        # variable directly rather than cell by cell.
        with netcdf_lock, _cache.reading('glodap', self.path):
            col = np.ma.filled(
                np.ma.asarray(self._ph[:, self.row(lat), self.col(lon)],
                              dtype=float),
                np.nan)
        # Backstop for a file whose fill is a bare sentinel with no mask:
        # seawater pH cannot leave the 0-14 scale.
        col[(col < PH_MIN) | (col > PH_MAX)] = np.nan
        valid = np.isfinite(col)
        return self._depth[valid], col[valid]


def _grid():
    return _cache.cached_grid('glodap', GLODAP_FILE, _GlodapGrid)


@dataclass(frozen=True, eq=False)
class PHProfile(ExportRecord):
    """A seawater pH column, as :func:`fetch_ph_profile` returns it.

    Attributes
    ----------
    depths : ndarray
        The GLODAP standard levels (m), trimmed at the seafloor.
    ph : ndarray
        In-situ pH at each level, on the scale ``ph_scale`` names.
    ph_scale : {'total'}
        GLODAP's ``pHtsinsitutp`` is on the total scale;
        :class:`~uacpy.core.absorption.FrancoisGarrison` takes
        ``ph_scale='total'`` and converts it to the NBS scale it was fitted
        on.
    provenance : DataProvenance
        The ``'glodap'`` record with the requested point.
    """

    depths: np.ndarray
    ph: np.ndarray
    ph_scale: str
    provenance: DataProvenance

    _ARRAY_FIELDS = ('depths', 'ph')
    _TABLE_FIELDS = ('depths', 'ph')


def fetch_ph_profile(point, *, max_distance_km=None) -> PHProfile:
    """Seawater pH column (total scale, in-situ) at a ``(lat, lon)`` point.

    Returns a :class:`PHProfile` on the GLODAP standard levels, trimmed at
    the seafloor. Raises ``DataFetchError`` where GLODAP has no column (land
    or unmapped).

    Parameters
    ----------
    point : (lat, lon)
        Site coordinates in decimal degrees.
    max_distance_km : float, optional
        Refuse a node farther than this (km) from ``point`` (the offset rule);
        ``None`` (default) sets no limit beyond its ``ProvenanceWarning``.
    """
    lat, lon = as_coordinate(point)
    prov = _grid().node_provenance(
        'glodap', lat, lon, who='fetch_ph_profile',
        max_distance_km=checked_max_distance(max_distance_km, 'fetch_ph_profile'))
    depths, ph = _grid().profile(lat, lon)
    if depths.size == 0:
        raise DataFetchError(
            f"GLODAP has no pH column at {lat:.3f}, {lon:.3f} "
            "(land or unmapped).",
            remediation="Pick an ocean location, or supply pH directly.",
        )
    return PHProfile(depths=depths, ph=ph, ph_scale='total',
                     provenance=prov)


def fetch_ph(point, *, reference_depth=None, max_distance_km=None):
    """Representative seawater pH at a ``(lat, lon)`` point.

    Samples the GLODAP column and returns the value at ``reference_depth`` (m,
    nearest level), or at the column's **mid-depth** when ``None`` — the
    depth :func:`uacpy.data.fetch_environment` reads its one pH at beside a
    fetched T/S profile. The pH column
    ends at the GLODAP seafloor level, which need not be the T/S column's;
    pass ``reference_depth`` to pin the row (``fetch_environment`` does).

    The value is raw and carries no provenance;
    :func:`uacpy.data.fetch_environment` with ``with_absorption=True``
    returns the carrier that records it in ``.data_sources``.

    Parameters
    ----------
    point : (lat, lon)
        Site coordinates in decimal degrees.
    reference_depth : float, optional
        Depth (m) of the row returned; ``None`` is the column's mid-depth.
    max_distance_km : float, optional
        As in :func:`fetch_ph_profile`.
    """
    return _ph_at_depth(fetch_ph_profile(point, max_distance_km=max_distance_km),
                       reference_depth)


def _ph_at_depth(profile: PHProfile, reference_depth=None) -> float:
    """The pH of ``profile`` at ``reference_depth`` (m, nearest level), or at
    the column's mid-depth when ``None`` — the row :func:`fetch_ph` returns."""
    depths, ph = profile.depths, profile.ph
    ref = (0.5 * (float(depths.min()) + float(depths.max()))
           if reference_depth is None else float(reference_depth))
    return float(ph[int(np.argmin(np.abs(depths - ref)))])
