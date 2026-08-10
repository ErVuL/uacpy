"""Boundary and volume scattering-strength laws for reverberation modelling.

Scattering strength ``S`` (dB) is the ratio of the intensity scattered by a
unit area (boundary) or unit volume to the incident plane-wave intensity,
referred to 1 m. It is negative for weak scatterers.

References
----------
Urick, R.J. (1983). *Principles of Underwater Sound*, 3rd ed., Ch. 8.
Chapman, R.P. & Harris, J.H. (1962). JASA 34(10), 1592-1597.
Mackenzie, K.V. (1961). JASA 33(11), 1498-1504 (Lambert bottom backscatter).
Etter, P.C. *Underwater Acoustic Modeling and Simulation*, Ch. 9.
"""

from __future__ import annotations

import numpy as np

from uacpy.core.exceptions import ConfigurationError

# Mackenzie (1961) deep-water bottom backscattering constant 10*log10(mu) [dB].
# He measured it constant at this value for both 530 and 1030 Hz (Etter,
# *Underwater Acoustic Modeling and Simulation*, eq. 9.6). For unconsolidated
# sediments the empirical spread is -25 to -35 dB, with -29 dB a common first
# guess (Jensen et al., *Computational Ocean Acoustics*, §1.7.2) — pass mu_db to
# pick another point in that range.
LAMBERT_MU_DB = -27.0


def lambert_bottom(grazing_deg, mu_db: float = LAMBERT_MU_DB):
    """Bottom backscattering strength from Lambert's law.

    ``S_b(theta) = 10*log10(mu) + 10*log10(sin^2 theta) = mu_db + 20*log10(sin theta)``

    Holds for many deep-water bottoms at grazing angles below ~45 deg
    (Urick 1983, Ch. 8). ``mu_db`` defaults to Mackenzie's -27 dB.

    Parameters
    ----------
    grazing_deg : float or array
        Grazing angle measured from the horizontal (degrees).
    mu_db : float
        Lambert coefficient ``10*log10(mu)`` in dB.

    Returns
    -------
    ndarray
        Backscattering strength (dB). ``-inf`` at zero grazing angle.
    """
    theta = np.deg2rad(np.asarray(grazing_deg, dtype=float))
    with np.errstate(divide="ignore"):
        return mu_db + 20.0 * np.log10(np.sin(theta))


def chapman_harris_surface(grazing_deg, wind_speed_kn: float, frequency: float):
    """Sea-surface backscattering strength, Chapman & Harris (1962).

    ``S_s = 3.3*beta*log10(theta/30) - 42.4*log10(beta) + 2.6``
    with ``beta = 158*(v*f**(1/3))**(-0.58)``.

    Validated 0.4-6.4 kHz, grazing angle below 80 deg (Chapman & Scott 1964).

    Parameters
    ----------
    grazing_deg : float or array
        Grazing angle from the horizontal (degrees).
    wind_speed_kn : float
        Near-surface wind speed (knots), > 0.
    frequency : float
        Acoustic frequency (Hz), > 0.

    Returns
    -------
    ndarray
        Surface backscattering strength (dB).
    """
    v = float(wind_speed_kn)
    f = float(frequency)
    if v <= 0.0 or f <= 0.0:
        raise ConfigurationError(
            "chapman_harris_surface: wind_speed_kn and frequency must be > 0"
        )
    theta = np.asarray(grazing_deg, dtype=float)
    beta = 158.0 * (v * f ** (1.0 / 3.0)) ** (-0.58)
    # -42.4 per Chapman & Harris (1962) JASA 34(10):1592, eq. as reproduced in
    # Urick (1983) Ch. 8 and Jensen et al. "Computational Ocean Acoustics".
    # theta = 0 (horizontal) is -inf, the honest degenerate answer, matching
    # lambert_bottom; silence the spurious divide warning.
    with np.errstate(divide="ignore", invalid="ignore"):
        return 3.3 * beta * np.log10(theta / 30.0) - 42.4 * np.log10(beta) + 2.6


def column_scattering_strength(sv_db, thickness_m: float):
    """Integrate a volume scattering strength to a column (area) strength.

    ``S_col = S_v + 10*log10(thickness)`` — turns ``S_v`` (dB re 1/m, per unit
    volume) into an equivalent per-unit-area strength for a scattering layer of
    the given thickness, so it can be combined with boundary strengths.

    Parameters
    ----------
    sv_db : float or array
        Volume scattering strength (dB re 1/m).
    thickness_m : float
        Layer thickness (m), > 0.
    """
    if thickness_m <= 0.0:
        raise ConfigurationError(
            "column_scattering_strength: thickness_m must be > 0"
        )
    return np.asarray(sv_db, dtype=float) + 10.0 * np.log10(thickness_m)
