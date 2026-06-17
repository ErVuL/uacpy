"""Shared netCDF4 access for the offline grid readers (GEBCO, WOA23)."""

from uacpy.core.exceptions import ConfigurationError

__all__ = ['open_netcdf']


def open_netcdf(path):
    """Open a local NetCDF file, or raise a typed error if netCDF4 is absent.

    netCDF4 ships with the default install (the offline grid backend); the typed
    error only fires in a stripped-down environment that dropped it.
    """
    try:
        from netCDF4 import Dataset
    except ImportError:
        raise ConfigurationError(
            "Reading local GEBCO/WOA23 grids requires the netCDF4 package.",
            remediation="netCDF4 ships with the default uacpy install; reinstall "
                        "with `pip install -e .`, or `pip install netCDF4`.",
        )
    return Dataset(str(path))
