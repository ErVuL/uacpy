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
        GLODAP and Copernicus report pH on the total scale. Francois &
        Garrison (1982, Part II) took their pH from Lovett's (1980) contour
        charts, drawn from the Gorshkov (1978) World Ocean Atlas, whose scale
        is not reported; Brewer & Hester (Oceanography 22(4), 2009, p. 91-92)
        judge it "probably" the NBS scale and state that "the sound
        absorption equations are based on the old NBS scale", with
        interconversion "still difficult". No conversion is applied here.
        With the Takahashi et al. (1982) activity-coefficient fit used by
        CO2SYS, NBS pH exceeds seawater-scale pH by 0.10 at 4 °C and 0.15 at
        25 °C (S = 35); +0.12 raises the boric-acid term, and so the whole
        absorption below 1 kHz, by about 20 %: 1 dB per 100 km at 1 kHz,
        5-6 dB over a 3000 km basin path at 300 Hz, under 2 dB per 100 km
        at any frequency.
    reference_depth : float, optional
        Depth (m) at which the nominal T/S row is taken. Default ``None``
        selects the **mid-depth** of the supplied column.

        The models receive a single T/S row and re-evaluate the formula in
        **depth only**, so this row's temperature governs absorption for the
        whole column. Taking it at the surface therefore carried the warmest
        water down the entire profile: on a mid-latitude column (22 C surface,
        4 C at 2 km) that understated absorption by 34 % at 10 kHz and 20 % at
        1 kHz against a mid-column reference. The mid-depth row is the
        representative choice for a stratified profile; pass an explicit
        value to pin it elsewhere.

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
    # Mid-depth, not z[0]: the single row this picks sets the temperature
    # for the entire column (the models vary only depth).
    ref = (0.5 * (float(z.min()) + float(z.max()))
           if reference_depth is None else float(reference_depth))
    i = int(np.argmin(np.abs(z - ref)))
    return FrancoisGarrison(
        temperature_c=float(t[i]),
        salinity_psu=float(s[i]),
        pH=float(pH),
        z_bar_m=float(z[i]),
    )
