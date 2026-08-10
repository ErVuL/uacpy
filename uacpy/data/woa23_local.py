"""Read WOA23 temperature/salinity columns from the local cache (offline SSP).

Drop-in for the OPeNDAP column fetch in :mod:`uacpy.data.sound_speed`: returns
the same truncated ``(depths, temperature, salinity)`` from the install-time
WOA23 NetCDF grids (``install.sh --data woa23``), so the seafloor-truncation,
annual splice and sound-speed conversion all stay identical to the online path.

Expected cache layout (canonical WOA filenames under one directory)::

    <cache>/woa23/woa23_<decade>_<var><period>_<code>.nc

e.g. ``woa23_decav_t00_01.nc`` (annual temperature, 1°), ``..._s07_04.nc``
(July salinity, 0.25°).
"""

import numpy as np

from uacpy.core.exceptions import DataFetchError
from uacpy.data import _cache
from uacpy.data._netcdf import open_netcdf

__all__ = ['column', 'close']

# Resolution → WOA file-resolution code (mirror of sound_speed._GRIDS).
_CODE = {'1.00': '01', '0.25': '04'}

_DATASETS = {}   # path -> netCDF Dataset (open once, read many)


def _open(path):
    key = str(path)
    if key not in _DATASETS:
        _DATASETS[key] = open_netcdf(path)
    return _DATASETS[key]


def close():
    """Close the open WOA23 handles and drop them; the next read reopens.

    A monthly SSP touches up to 24 files per resolution and they stay open for
    the process lifetime, so dropping the memo is not enough — the handles have
    to be closed. Registered with :func:`uacpy.data._cache.register_cache`, and
    also public because WOA23 is installed out of band
    (``install.sh --data woa23``) rather than by a ``download_*`` function.
    """
    for ds in _DATASETS.values():
        try:
            ds.close()
        except (RuntimeError, OSError):        # already closed / file vanished
            pass
    _DATASETS.clear()


_cache.register_cache(close)


def _field(period, var, *, resolution, decade):
    code = _CODE[resolution]
    fname = f"woa23_{decade}_{var}{period:02d}_{code}.nc"
    return _cache.require('woa23', fname)


def _read(path, var, lat_idx, lon_idx):
    """One depth axis + variable column, masked/fill cells mapped to ``NaN``.

    netCDF4 returns a masked array wherever ``_FillValue`` is set, so the mask
    is what identifies no-data. Resolving it here keeps the seafloor truncation
    independent of the fill's magnitude: a product filling with -999 rather
    than WOA's 9.96921e36 is still cut, not read as a real temperature.
    """
    ds = _open(path)
    names = {n.lower(): n for n in ds.variables}
    try:
        depth = np.ma.filled(
            np.ma.asarray(ds.variables[names['depth']][:], dtype=float), np.nan)
        an = np.ma.filled(
            np.ma.asarray(ds.variables[names[var]][0, :, lat_idx, lon_idx],
                          dtype=float), np.nan)
    except KeyError as exc:
        raise DataFetchError(
            f"Local WOA23 file {path.name} is missing variable {exc}; "
            "its schema may have changed.",
            remediation="Re-run ./install.sh --data woa23 to refresh it.",
        ) from exc
    return depth, an


def column(period, lat_idx, lon_idx, *, resolution, decade):
    """One ``(z, t, s)`` water column from the local WOA23 grids.

    Mirrors :func:`sound_speed._fetch_column`: the temperature and salinity
    files share a depth axis, and the column is truncated at the seafloor by the
    shared fill-value rule.
    """
    from uacpy.data.sound_speed import _truncate_column

    t_path = _field(period, 't', resolution=resolution, decade=decade)
    s_path = _field(period, 's', resolution=resolution, decade=decade)
    z, t = _read(t_path, 't_an', lat_idx, lon_idx)
    _, s = _read(s_path, 's_an', lat_idx, lon_idx)
    if z.size == 0:
        raise DataFetchError(
            "Local WOA23 file has an empty depth axis.",
            remediation="Re-run ./install.sh --data woa23 to refresh it.",
        )
    return _truncate_column(z, t, s)
