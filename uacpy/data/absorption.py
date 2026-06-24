"""Build a Francois-Garrison absorption from a fetched T/S column.

Helper shared by the sound-speed sources: both
:func:`uacpy.data.fetch_ts_profile` (WOA23) and the Copernicus operational
fetch return a temperature/salinity column, and Francois-Garrison seawater
absorption is parameterised by temperature, salinity, pH and a reference
depth. This turns one into the other.

pH is **not** carried by WOA23; the default (8.1, typical open-ocean surface)
is used unless the caller supplies a measured value (e.g. from a Copernicus
biogeochemistry product).
"""

from typing import Optional

import numpy as np

from uacpy.core.absorption import FrancoisGarrison
from uacpy.core.exceptions import ConfigurationError

__all__ = ['build_francois_garrison']

DEFAULT_OCEAN_PH = 8.1


def build_francois_garrison(
    depths,
    temperature,
    salinity,
    *,
    pH: float = DEFAULT_OCEAN_PH,
    reference_depth: Optional[float] = None,
) -> FrancoisGarrison:
    """Francois-Garrison absorption from a temperature/salinity column.

    Parameters
    ----------
    depths, temperature, salinity : array-like
        Matching 1-D profiles (m, °C, psu), as returned by the data fetchers.
    pH : float, optional
        Seawater pH (default 8.1). Supply a measured value where available.
    reference_depth : float, optional
        Depth (m) at which the nominal T/S row is taken. Default ``None``
        selects the shallowest sample (surface). The Acoustics-Toolbox
        formula re-evaluates per depth at run time; this only sets the
        single-row nominal.

    Returns
    -------
    FrancoisGarrison
    """
    z = np.asarray(depths, dtype=float).reshape(-1)
    t = np.asarray(temperature, dtype=float).reshape(-1)
    s = np.asarray(salinity, dtype=float).reshape(-1)
    if z.size == 0 or not (z.size == t.size == s.size):
        raise ConfigurationError(
            "build_francois_garrison: depths, temperature and salinity must be "
            f"non-empty and equal length; got {z.size}, {t.size}, {s.size}."
        )
    ref = z[0] if reference_depth is None else float(reference_depth)
    i = int(np.argmin(np.abs(z - ref)))
    return FrancoisGarrison(
        temperature_c=float(t[i]),
        salinity_psu=float(s[i]),
        pH=float(pH),
        z_bar_m=float(z[i]),
    )
