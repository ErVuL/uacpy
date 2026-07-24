"""Offline GLODAPv2.2016b seawater pH (GLODAP, CC-BY 4.0).

``install.sh --data glodap`` downloads the GLODAPv2.2016b Mapped Climatology pH
field — a global 1° × 1°, 33-level ``pH(depth, lat, lon)`` grid on the total
scale at in-situ temperature and pressure (Lauvset et al. 2016) — into the
cache. This module samples it locally.

pH is the one Francois-Garrison absorption input WOA23 does not carry, so
without it :func:`uacpy.data.build_francois_garrison` falls back to a constant
(8.1). A cached GLODAP grid replaces that constant with the real in-situ column,
letting :func:`uacpy.data.fetch_environment` build absorption from measured pH.
"""

import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from uacpy._log import log_message
from uacpy.core.exceptions import DataFetchError
from uacpy.data import _cache
from uacpy.data._geo import as_coordinate
from uacpy.data._netcdf import NetcdfGrid

__all__ = ['download_glodap_db', 'fetch_ph_profile', 'fetch_ph']

GLODAP_FILE = 'GLODAPv2.2016b.pHtsinsitutp.nc'
GLODAP_TARBALL = 'GLODAPv2.2016b.MappedProduct.tar.gz'
GLODAP_URL = ('https://glodap.info/glodap_files/v2.2023/'
              'GLODAPv2.2016b.MappedProduct.tar.gz')
# The pH variable inside the mapped product; the depth axis is a separate
# coordinate variable (the mapped files name it ``Depth``).
_PH_VARS = ('pHtsinsitutp', 'pHts25p0', 'ph')
_DEPTH_VARS = ('depth', 'depth_surface')

_GRID = {}   # path -> _GlodapGrid


def _curl_download(url, out, *, timeout, verbose):
    """Fetch ``url`` → ``out`` with curl; ``True`` on success, ``False`` if curl
    is absent or fails. The GLODAP host serves curl at full speed; mirrors the
    GlobSed/curl-first pattern for the large static grids. Downloads to
    ``<out>.part`` and moves it into place only on success, so an interrupted
    transfer never leaves a truncated ``out``."""
    curl = shutil.which('curl')
    if not curl:
        return False
    part = Path(str(out) + '.part')
    try:
        subprocess.run(
            [curl, '-fL', '--retry', '3', '--max-time', str(int(timeout)),
             '-o', str(part), url],
            check=True, capture_output=not verbose)
    except (subprocess.SubprocessError, OSError):
        part.unlink(missing_ok=True)
        return False
    if not (part.exists() and part.stat().st_size > 0):
        part.unlink(missing_ok=True)
        return False
    os.replace(part, out)
    return True


def download_glodap_db(cache_dir=None, *, timeout=600.0, verbose=False):
    """Download the GLODAPv2.2016b Mapped pH field into the cache.

    Fetches the mapped-product tarball (~211 MB), extracts only the in-situ pH
    grid to ``<cache>/glodap/GLODAPv2.2016b.pHtsinsitutp.nc`` and discards the
    rest, then returns the path. Uses curl when available, falling back to the
    urllib fetcher.
    """
    from uacpy.data._http import http_get
    dest = Path(cache_dir) if cache_dir else _cache.dataset_root('glodap')
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / GLODAP_FILE
    log_message('glodap', "downloading GLODAPv2.2016b mapped product (~211 MB)",
                verbose=verbose)
    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / GLODAP_TARBALL
        if not _curl_download(GLODAP_URL, tar_path, timeout=timeout,
                              verbose=verbose):
            part = Path(str(tar_path) + '.part')
            part.write_bytes(http_get(GLODAP_URL, timeout=timeout,
                                      verbose=verbose, source='glodap'))
            os.replace(part, tar_path)
        _extract_ph(tar_path, out)
    _GRID.clear()
    log_message('glodap', f"GLODAP pH grid cached → {out}", verbose=verbose)
    return out


def _extract_ph(tar_path, out):
    """Extract the pH member from the mapped-product tarball to ``out``.

    Iterates lazily (member data is never read before its declared size passes
    the decompression-bomb cap) and writes atomically via ``<out>.part``."""
    from uacpy.data._http import checked_member_size
    with tarfile.open(tar_path, 'r:gz') as tar:
        member = next((m for m in tar
                       if Path(m.name).name == GLODAP_FILE), None)
        if member is None:
            raise DataFetchError(
                f"GLODAP tarball has no {GLODAP_FILE} member; its layout may "
                "have changed.",
                remediation="Check the GLODAP mapped-product download.",
            )
        checked_member_size(member.size, member.name)
        src = tar.extractfile(member)
        if src is None:
            raise DataFetchError(
                f"GLODAP tarball member {member.name!r} is not a regular file.",
                remediation="Check the GLODAP mapped-product download.",
            )
        part = Path(str(out) + '.part')
        part.write_bytes(src.read())
        os.replace(part, out)


class _GlodapGrid(NetcdfGrid):
    """Column accessor over the GLODAP ``pH(depth, lat, lon)`` grid.

    Reuses :class:`NetcdfGrid` for nearest lat/lon indexing (the mapped grid's
    longitude axis has a shifted origin, which ``col`` already wraps), and binds
    the depth axis plus the pH variable for whole-column reads.
    """

    def __init__(self, path):
        try:
            super().__init__(path)
            self._ph = self.var(*_PH_VARS)
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
        col = np.ma.filled(
            np.asarray(self._ph[:, self.row(lat), self.col(lon)], dtype=float),
            np.nan)
        valid = np.isfinite(col)
        return self._depth[valid], col[valid]


def _grid():
    path = _cache.require('glodap', GLODAP_FILE)
    key = str(path)
    if key not in _GRID:
        _GRID[key] = _GlodapGrid(path)
    return _GRID[key]


def fetch_ph_profile(point):
    """Seawater pH column (total scale, in-situ) at a ``(lat, lon)`` point.

    Returns ``(depths_m, pH)`` on the GLODAP standard levels, trimmed at the
    seafloor. Raises ``DataFetchError`` where GLODAP has no column (land or
    unmapped).
    """
    lat, lon = as_coordinate(point)
    depths, ph = _grid().profile(lat, lon)
    if depths.size == 0:
        raise DataFetchError(
            f"GLODAP has no pH column at {lat:.3f}, {lon:.3f} "
            "(land or unmapped).",
            remediation="Pick an ocean location, or supply pH directly.",
        )
    return depths, ph


def fetch_ph(point, *, reference_depth=None):
    """Representative seawater pH at a ``(lat, lon)`` point.

    Samples the GLODAP column and returns the value at ``reference_depth`` (m,
    nearest level), or the shallowest level (surface) when ``None`` — matching
    the nominal-row convention of :func:`uacpy.data.build_francois_garrison`.
    """
    depths, ph = fetch_ph_profile(point)
    if reference_depth is None:
        return float(ph[0])
    i = int(np.argmin(np.abs(depths - float(reference_depth))))
    return float(ph[i])
