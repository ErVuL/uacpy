"""
Attenuation models for underwater acoustics.

Implements:

* :func:`thorp_attenuation` — canonical Thorp 1967 formula.
* :func:`francois_garrison` — Francois & Garrison 1982 multi-relaxation.
* :func:`thorp_with_depth_correction` — Thorp scaled by ``(1 + depth/5000)``.
* :func:`empirical_attenuation` — single-relaxation Thorp-like formula
  with temperature / salinity / depth knobs (handy for sensitivity
  studies; not a published reference formula).
* :func:`convert_attenuation_units` — unit conversion helper.

Notes
-----
This module is a user/example helper. The core uacpy library wires
frequency-dependent volume attenuation inside the individual model writers
(see ``uacpy.io.oalib_writer`` and each model's ``volume_attenuation``
kwarg). Nothing in ``uacpy.core`` or ``uacpy.models`` imports from here;
it is used by ``example_12_attenuation_models.py`` and by users who need
to compute attenuation curves directly.
"""

import numpy as np
from typing import Union


def thorp_attenuation(frequency: Union[float, np.ndarray]) -> np.ndarray:
    """
    Thorp attenuation formula.

    Parameters
    ----------
    frequency : float or array
        Frequency in Hz.

    Returns
    -------
    alpha : float or array
        Attenuation in dB/km.

    References
    ----------
    Thorp, W. H. (1967). "Analytic Description of the Low-Frequency
    Attenuation Coefficient". JASA, 42(1), 270.

    Examples
    --------
    >>> alpha = thorp_attenuation(1000.0)  # 1 kHz
    >>> print(f"Attenuation: {alpha:.4f} dB/km")
    """
    f = np.atleast_1d(frequency) / 1000.0
    f2 = f**2
    alpha = 0.11 * f2 / (1 + f2) + 44 * f2 / (4100 + f2) + 2.75e-4 * f2 + 0.003

    return np.squeeze(alpha)


def francois_garrison(
    frequency: Union[float, np.ndarray],
    temperature: float = 10.0,
    salinity: float = 35.0,
    pH: float = 8.0,
    depth: float = 1000.0
) -> np.ndarray:
    """
    Francois-Garrison attenuation formula (more accurate than Thorp)

    Parameters
    ----------
    frequency : float or array
        Frequency in Hz
    temperature : float, optional
        Temperature in °C. Default is 10.0.
    salinity : float, optional
        Salinity in ppt. Default is 35.0.
    pH : float, optional
        pH value. Default is 8.0.
    depth : float, optional
        Depth in meters. Default is 1000.0.

    Returns
    -------
    alpha : float or array
        Attenuation in dB/km

    References
    ----------
    Francois, R. E. and Garrison, G. R. (1982). "Sound absorption based
    on ocean measurements. Part II: Boric acid contribution and equation
    for total absorption". JASA, 72(6), 1879-1890.

    Notes
    -----
    Implementation based on Acoustics Toolbox AttenMod.f90 by M. Porter.
    Verified using Francois-Garrison Table IV.

    Examples
    --------
    >>> alpha = francois_garrison(1000.0, temperature=15, salinity=35)
    >>> print(f"Attenuation: {alpha:.4f} dB/km")
    """
    f = np.atleast_1d(frequency) / 1000.0  # Convert Hz to kHz

    T = temperature  # °C
    S = salinity  # ppt
    z_bar = depth  # meters

    # Sound speed (Francois-Garrison formula)
    c = 1412.0 + 3.21 * T + 1.19 * S + 0.0167 * z_bar

    # Boric acid contribution
    A1 = 8.86 / c * 10.0 ** (0.78 * pH - 5.0)
    P1 = 1.0
    f1 = 2.8 * np.sqrt(S / 35.0) * 10.0 ** (4.0 - 1245.0 / (T + 273.0))

    # Magnesium sulfate contribution
    A2 = 21.44 * S / c * (1.0 + 0.025 * T)
    P2 = 1.0 - 1.37e-4 * z_bar + 6.2e-9 * z_bar**2
    f2 = 8.17 * 10.0 ** (8.0 - 1990.0 / (T + 273.0)) / (1.0 + 0.0018 * (S - 35.0))

    # Pure water viscosity contribution
    P3 = 1.0 - 3.83e-5 * z_bar + 4.9e-10 * z_bar**2

    # Temperature-dependent coefficients for viscosity
    if np.isscalar(T):
        if T < 20:
            A3 = 4.937e-4 - 2.59e-5 * T + 9.11e-7 * T**2 - 1.5e-8 * T**3
        else:
            A3 = 3.964e-4 - 1.146e-5 * T + 1.45e-7 * T**2 - 6.5e-10 * T**3
    else:
        # Handle array input
        A3 = np.where(T < 20,
                      4.937e-4 - 2.59e-5 * T + 9.11e-7 * T**2 - 1.5e-8 * T**3,
                      3.964e-4 - 1.146e-5 * T + 1.45e-7 * T**2 - 6.5e-10 * T**3)

    # Total attenuation (dB/km)
    # Each term: A * P * (f_relax * f^2) / (f_relax^2 + f^2)
    alpha = (A1 * P1 * (f1 * f**2) / (f1**2 + f**2) +
             A2 * P2 * (f2 * f**2) / (f2**2 + f**2) +
             A3 * P3 * f**2)

    return np.squeeze(alpha)


def thorp_with_depth_correction(
    frequency: Union[float, np.ndarray],
    depth: float = 100.0
) -> np.ndarray:
    """
    Thorp 1967 attenuation with a linear depth-correction factor.

    Returns the canonical Thorp formula multiplied by ``(1 + depth/5000)``
    — a heuristic shallow-vs-deep adjustment for engineering estimates.
    For physics-grade depth/temperature/salinity treatment use
    :func:`francois_garrison`.

    Parameters
    ----------
    frequency : float or array
        Frequency in Hz
    depth : float, optional
        Water depth in meters. Default is 100.0.

    Returns
    -------
    alpha : float or array
        Attenuation in dB/km
    """
    f = np.atleast_1d(frequency) / 1000.0  # kHz
    alpha = 0.11 * f**2 / (1.0 + f**2) + 44.0 * f**2 / (4100.0 + f**2) + 2.75e-4 * f**2 + 0.003
    alpha *= 1.0 + depth / 5000.0
    return np.squeeze(alpha)


def empirical_attenuation(
    frequency: Union[float, np.ndarray],
    temperature: float = 10.0,
    salinity: float = 35.0,
    pH: float = 8.0,
    depth: float = 1000.0
) -> np.ndarray:
    """
    Empirical Thorp-like attenuation with temperature/salinity/depth knobs.

    Single-relaxation form with a temperature-dependent relaxation
    frequency, plus linear salinity and depth scaling — a quick
    sensitivity-study tool. The ``pH`` argument is a placeholder and is
    not consumed by the formula. For physics-grade attenuation use
    :func:`francois_garrison`.

    Parameters
    ----------
    frequency : float or array
        Frequency in Hz
    temperature : float, optional
        Temperature in °C. Default is 10.0.
    salinity : float, optional
        Salinity in ppt. Default is 35.0.
    pH : float, optional
        pH value. Accepted for signature symmetry; not used in the
        empirical formula.
    depth : float, optional
        Depth in meters. Default is 1000.0.

    Returns
    -------
    alpha : float or array
        Attenuation in dB/km
    """
    f = np.atleast_1d(frequency) / 1000.0  # kHz
    T = temperature

    # Single temperature-dependent relaxation frequency.
    f_relax = 21.0 * np.exp(T / 15.0)

    alpha = 0.11 * f**2 / (1.0 + f**2) + 44.0 * f**2 / (f_relax**2 + f**2) + 2.5e-4 * f**2
    alpha *= 1.0 + 0.01 * (salinity - 35.0)
    alpha *= 1.0 + 0.05 * (depth / 1000.0)
    return np.squeeze(alpha)


def convert_attenuation_units(
    alpha: Union[float, np.ndarray],
    frequency: float,
    from_unit: str,
    to_unit: str,
    sound_speed: float = 1500.0
) -> np.ndarray:
    """
    Convert attenuation between different units

    Parameters
    ----------
    alpha : float or array
        Attenuation value
    frequency : float
        Frequency in Hz
    from_unit : str
        Source unit: 'dB/km', 'dB/m', 'dB/wavelength', 'Nepers/m', 'Q', 'L'
    to_unit : str
        Target unit (same options as from_unit)
    sound_speed : float, optional
        Sound speed in m/s. Default is 1500.0.

    Returns
    -------
    alpha_converted : float or array
        Converted attenuation

    Examples
    --------
    >>> # Convert dB/km to dB/m
    >>> alpha_m = convert_attenuation_units(0.1, 1000, 'dB/km', 'dB/m')
    """
    alpha = np.atleast_1d(alpha)

    # First convert to dB/m
    if from_unit == 'dB/km':
        alpha_db_m = alpha / 1000.0
    elif from_unit == 'dB/m':
        alpha_db_m = alpha
    elif from_unit == 'dB/wavelength':
        wavelength = sound_speed / frequency
        alpha_db_m = alpha / wavelength
    elif from_unit == 'Nepers/m':
        alpha_db_m = alpha * 8.686  # 1 Neper = 8.686 dB
    elif from_unit == 'Q':  # Quality factor
        # Q = 2π/α where α is in Nepers/wavelength
        # Convert to dB/m
        alpha_nepers_m = 2 * np.pi * frequency / (alpha * sound_speed)
        alpha_db_m = alpha_nepers_m * 8.686
    elif from_unit == 'L':  # Loss parameter (loss tangent)
        # L = α * λ / π where α is in Nepers/m
        alpha_nepers_m = alpha * np.pi * frequency / sound_speed
        alpha_db_m = alpha_nepers_m * 8.686
    else:
        raise ValueError(f"Unknown unit: {from_unit}")

    # Now convert from dB/m to target unit
    if to_unit == 'dB/km':
        result = alpha_db_m * 1000.0
    elif to_unit == 'dB/m':
        result = alpha_db_m
    elif to_unit == 'dB/wavelength':
        wavelength = sound_speed / frequency
        result = alpha_db_m * wavelength
    elif to_unit == 'Nepers/m':
        result = alpha_db_m / 8.686
    elif to_unit == 'Q':
        alpha_nepers_m = alpha_db_m / 8.686
        result = 2 * np.pi * frequency / (alpha_nepers_m * sound_speed)
    elif to_unit == 'L':
        alpha_nepers_m = alpha_db_m / 8.686
        result = alpha_nepers_m * sound_speed / (np.pi * frequency)
    else:
        raise ValueError(f"Unknown unit: {to_unit}")

    return np.squeeze(result)
