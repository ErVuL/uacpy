"""
Underwater acoustics utilities for UACPY

This module provides various underwater acoustics functions including:
- Sound speed calculations
- Ambient noise modeling (Wenz model)
- Bubble acoustics
- Absorption calculations
- Acoustic pressure utilities

-------------------------------------------------------------------------------
Portions of this file are adapted from arlpy (https://github.com/org-arl/arlpy)
Copyright (c) 2016-2020, Acoustic Research Laboratory
All rights reserved.

Redistributed under the terms of the 3-clause BSD license.  The full
license text, including the required disclaimer and no-endorsement clause,
is reproduced in:

    uacpy/third_party/arlpy/LICENSE

See uacpy/third_party/arlpy/NOTICE for the list of arlpy-adapted functions
in this file.
-------------------------------------------------------------------------------
"""

import numpy as np
import scipy.signal as sp
from typing import Union, Optional, Tuple


def soundspeed(
    temperature: float = 27, salinity: float = 35, depth: float = 10
) -> float:
    """
    Calculate speed of sound in water using Mackenzie (1981) formula.

    Parameters
    ----------
    temperature : float, optional
        Water temperature in degrees Celsius (default: 27)
    salinity : float, optional
        Salinity in parts per thousand (ppt) (default: 35)
    depth : float, optional
        Depth in meters (default: 10)

    Returns
    -------
    float
        Sound speed in m/s

    Examples
    --------
    >>> c = soundspeed()
    >>> print(f"Sound speed: {c:.1f} m/s")

    >>> c = soundspeed(temperature=25, depth=20)
    >>> print(f"Sound speed: {c:.1f} m/s")

    References
    ----------
    Mackenzie, K. V. (1981). "Nine-term equation for sound speed in the oceans".
    The Journal of the Acoustical Society of America, 70(3), 807-812.
    """
    c = (
        1448.96
        + 4.591 * temperature
        - 5.304e-2 * temperature**2
        + 2.374e-4 * temperature**3
    )
    c += 1.340 * (salinity - 35) + 1.630e-2 * depth + 1.675e-7 * depth**2
    c += -1.025e-2 * temperature * (salinity - 35) - 7.139e-13 * temperature * depth**3
    return c


def density(temperature: float = 27, salinity: float = 35) -> float:
    """
    Calculate density of sea water near the surface.

    Uses Fofonoff (1985 - IES 80) formula.

    Parameters
    ----------
    temperature : float, optional
        Water temperature in degrees Celsius (default: 27)
    salinity : float, optional
        Salinity in parts per thousand (ppt) (default: 35)

    Returns
    -------
    float
        Density in kg/m³

    Examples
    --------
    >>> rho = density()
    >>> print(f"Density: {rho:.1f} kg/m³")

    References
    ----------
    Fofonoff, N. P. (1985). "Physical properties of seawater: A new salinity
    scale and equation of state for seawater". Journal of Geophysical Research,
    90(C2), 3332-3342.
    """
    t = temperature
    A = 1.001685e-04 + t * (-1.120083e-06 + t * 6.536332e-09)
    A = 999.842594 + t * (6.793952e-02 + t * (-9.095290e-03 + t * A))
    B = 7.6438e-05 + t * (-8.2467e-07 + t * 5.3875e-09)
    B = 0.824493 + t * (-4.0899e-03 + t * B)
    C = -5.72466e-03 + t * (1.0227e-04 - t * 1.6546e-06)
    D = 4.8314e-04
    return A + salinity * (B + C * np.sqrt(salinity) + D * salinity)


def absorption(
    frequency: Union[float, np.ndarray],
    distance: float = 1000,
    temperature: float = 27,
    salinity: float = 35,
    depth: float = 10,
    pH: float = 8.1,
) -> Union[float, np.ndarray]:
    """
    Calculate acoustic absorption in water using Francois-Garrison model.

    Parameters
    ----------
    frequency : float or array_like
        Frequency in Hz
    distance : float, optional
        Distance in meters (default: 1000)
    temperature : float, optional
        Water temperature in degrees Celsius (default: 27)
    salinity : float, optional
        Salinity in ppt (default: 35)
    depth : float, optional
        Depth in meters (default: 10)
    pH : float, optional
        pH of water (default: 8.1)

    Returns
    -------
    float or ndarray
        Absorption as a linear multiplier (amplitude ratio)

    Examples
    --------
    >>> abs_50khz = absorption(50000)
    >>> abs_db = 20 * np.log10(abs_50khz)  # Convert to dB
    >>> print(f"Absorption at 50 kHz, 1 km: {abs_db:.2f} dB")

    References
    ----------
    Francois, R. E. and Garrison, G. R. (1982). "Sound absorption based on
    ocean measurements". JASA, 72(6), 1879-1890.
    """
    f = frequency / 1000.0  # Convert to kHz
    d = distance / 1000.0  # Convert to km
    c = 1412.0 + 3.21 * temperature + 1.19 * salinity + 0.0167 * depth

    # Boric acid contribution
    A1 = 8.86 / c * 10 ** (0.78 * pH - 5)
    P1 = 1.0
    f1 = 2.8 * np.sqrt(salinity / 35) * 10 ** (4 - 1245 / (temperature + 273))

    # Magnesium sulfate contribution
    A2 = 21.44 * salinity / c * (1 + 0.025 * temperature)
    P2 = 1.0 - 1.37e-4 * depth + 6.2e-9 * depth * depth
    f2 = 8.17 * 10 ** (8 - 1990 / (temperature + 273)) / (1 + 0.0018 * (salinity - 35))

    # Pure water contribution
    P3 = 1.0 - 3.83e-5 * depth + 4.9e-10 * depth * depth
    if temperature < 20:
        A3 = (
            4.937e-4
            - 2.59e-5 * temperature
            + 9.11e-7 * temperature**2
            - 1.5e-8 * temperature**3
        )
    else:
        A3 = (
            3.964e-4
            - 1.146e-5 * temperature
            + 1.45e-7 * temperature**2
            - 6.5e-10 * temperature**3
        )

    # Total absorption coefficient in dB/km
    a = (
        A1 * P1 * f1 * f * f / (f1 * f1 + f * f)
        + A2 * P2 * f2 * f * f / (f2 * f2 + f * f)
        + A3 * P3 * f * f
    )

    # Return as linear multiplier
    return 10 ** (-a * d / 20.0)


def absorption_filter(
    fs: float,
    ntaps: int = 31,
    nfreqs: int = 64,
    distance: float = 1000,
    temperature: float = 27,
    salinity: float = 35,
    depth: float = 10,
) -> np.ndarray:
    """
    Design a FIR filter with response based on acoustic absorption in water.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz
    ntaps : int, optional
        Number of FIR taps (default: 31)
    nfreqs : int, optional
        Number of frequencies for modeling response (default: 64)
    distance : float, optional
        Distance in meters (default: 1000)
    temperature : float, optional
        Water temperature in degrees Celsius (default: 27)
    salinity : float, optional
        Salinity in ppt (default: 35)
    depth : float, optional
        Depth in meters (default: 10)

    Returns
    -------
    ndarray
        FIR filter tap weights

    Examples
    --------
    >>> fs = 250000
    >>> b = absorption_filter(fs, distance=500)
    >>> # Apply filter to signal x
    >>> y_filtered = sp.lfilter(b, 1, x)
    >>> # Also apply spreading loss
    >>> y_filtered /= 500**2
    """
    nyquist = fs / 2.0
    f = np.linspace(0, nyquist, num=nfreqs)
    g = absorption(f, distance, temperature, salinity, depth)
    return sp.firwin2(ntaps, f, g, fs=fs)


def doppler(speed: float, frequency: float, c: Optional[float] = None) -> float:
    """
    Calculate Doppler-shifted frequency.

    The approximation is valid when speed << c (typical for underwater vehicles).

    Parameters
    ----------
    speed : float
        Relative speed between transmitter and receiver in m/s
        (positive = approaching, negative = receding)
    frequency : float
        Transmission frequency in Hz
    c : float, optional
        Sound speed in m/s (default: calculated using soundspeed())

    Returns
    -------
    float
        Doppler shifted frequency in Hz as perceived by the receiver

    Examples
    --------
    >>> f_shifted = doppler(2, 50000)  # 2 m/s approach
    >>> print(f"Shifted frequency: {f_shifted:.2f} Hz")

    >>> f_shifted = doppler(-1, 50000)  # 1 m/s receding
    >>> print(f"Shifted frequency: {f_shifted:.2f} Hz")
    """
    if c is None:
        c = soundspeed()
    return (1 + speed / c) * frequency


def reflection_coeff(
    angle: Union[float, np.ndarray],
    rho1: float,
    c1: float,
    alpha: float = 0,
    rho: Optional[float] = None,
    c: Optional[float] = None,
) -> Union[float, np.ndarray, complex]:
    """
    Calculate Rayleigh reflection coefficient for a given angle.

    Parameters
    ----------
    angle : float or array_like
        Angle of incidence in radians
    rho1 : float
        Density of second medium (e.g., sediment) in kg/m³
    c1 : float
        Sound speed in second medium in m/s
    alpha : float, optional
        Attenuation coefficient (default: 0)
    rho : float, optional
        Density of water in kg/m³ (default: calculated)
    c : float, optional
        Sound speed in water in m/s (default: calculated)

    Returns
    -------
    float, ndarray, or complex
        Reflection coefficient as a linear multiplier

    Examples
    --------
    >>> R = reflection_coeff(np.pi/4, 1200, 1600)
    >>> print(f"Reflection coefficient: {R:.4f}")

    >>> R_db = 20 * np.log10(abs(R))
    >>> print(f"Reflection loss: {R_db:.2f} dB")

    References
    ----------
    Brekhovskikh, L. M. & Lysanov, Y. P. (2003). Fundamentals of Ocean Acoustics.
    """
    if rho is None:
        rho = density()
    if c is None:
        c = soundspeed()

    # Brekhovskikh & Lysanov formulation
    n = float(c) / c1 * (1 + 1j * alpha)
    m = float(rho1) / rho
    t1 = m * np.cos(angle)
    t2 = np.sqrt(n**2 - np.sin(angle) ** 2)
    V = (t1 - t2) / (t1 + t2)

    return V.real if np.all(V.imag == 0) else V


def bubble_resonance(
    radius: Union[float, np.ndarray],
    depth: float = 0.0,
    gamma: float = 1.4,
    p0: float = 1.013e5,
    rho_water: float = 1022.476,
) -> Union[float, np.ndarray]:
    """
    Calculate resonance frequency of freely oscillating gas bubble in water.

    Based on Medwin & Clay (1998). Ignores surface tension, thermal, viscous,
    and acoustic damping effects. Assumes adiabatic pressure-volume relationship.

    Parameters
    ----------
    radius : float or array_like
        Bubble radius in meters
    depth : float, optional
        Depth of bubble in water in meters (default: 0.0)
    gamma : float, optional
        Gas ratio of specific heats (default: 1.4 for air)
    p0 : float, optional
        Atmospheric pressure in Pa (default: 1.013e5)
    rho_water : float, optional
        Density of water in kg/m³ (default: 1022.476)

    Returns
    -------
    float or ndarray
        Resonance frequency in Hz

    Examples
    --------
    >>> f_res = bubble_resonance(100e-6)  # 100 micron radius
    >>> print(f"Resonance frequency: {f_res:.2f} Hz")

    >>> radii = np.logspace(-5, -3, 50)  # 10 to 1000 microns
    >>> f_res = bubble_resonance(radii)

    References
    ----------
    Medwin, H. & Clay, C. S. (1998). Fundamentals of Acoustical Oceanography.
    """
    g = 9.80665  # acceleration due to gravity (m/s²)
    p_air = p0 + rho_water * g * depth
    return 1 / (2 * np.pi * radius) * np.sqrt(3 * gamma * p_air / rho_water)


def bubble_surface_loss(
    windspeed: float,
    frequency: Union[float, np.ndarray],
    angle: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate surface loss due to bubbles using APL model (1994).

    Parameters
    ----------
    windspeed : float
        Windspeed in m/s (measured 10 m above sea surface)
    frequency : float or array_like
        Frequency in Hz
    angle : float or array_like
        Incidence angle in radians

    Returns
    -------
    float or ndarray
        Absorption as a linear multiplier

    Examples
    --------
    >>> loss = bubble_surface_loss(3, 10000, 0)
    >>> loss_db = 20 * np.log10(loss)
    >>> print(f"Surface loss: {loss_db:.2f} dB")

    References
    ----------
    APL-UW (1994). "APL-UW High-Frequency Ocean Environmental Acoustic Models
    Handbook". Technical Report APL-UW TR 9407.
    """
    beta = np.pi / 2 - angle
    f = frequency / 1000.0  # Convert to kHz

    if windspeed >= 6:
        a = 1.26e-3 / np.sin(beta) * windspeed**1.57 * f**0.85
    else:
        a = 1.26e-3 / np.sin(beta) * 6**1.57 * f**0.85 * np.exp(1.2 * (windspeed - 6))

    return 10 ** (-a / 20.0)


def bubble_soundspeed(
    void_fraction: Union[float, np.ndarray],
    c: Optional[float] = None,
    c_gas: float = 340,
    relative_density: float = 1000,
) -> Union[float, np.ndarray]:
    """
    Calculate speed of sound in 2-phase bubbly water.

    Based on Wood (1964) or Buckingham (1997).

    Parameters
    ----------
    void_fraction : float or array_like
        Void fraction (ratio of gas volume to total volume)
    c : float, optional
        Speed of sound in water in m/s (default: calculated)
    c_gas : float, optional
        Speed of sound in gas in m/s (default: 340)
    relative_density : float, optional
        Ratio of density of water to gas (default: 1000)

    Returns
    -------
    float or ndarray
        Sound speed in bubbly water in m/s

    Examples
    --------
    >>> c_bubbly = bubble_soundspeed(1e-5)
    >>> print(f"Sound speed in bubbly water: {c_bubbly:.2f} m/s")

    References
    ----------
    Wood, A. B. (1964). A Textbook of Sound.
    Buckingham, M. J. (1997). "Theory of acoustic attenuation, dispersion,
    and pulse propagation in unconsolidated granular materials". JASA, 102(5).
    """
    if c is None:
        c = soundspeed()

    m = np.sqrt(relative_density)
    numerator = void_fraction * (c / c_gas) ** 2 * m + (1 - void_fraction) / m
    denominator = void_fraction / m + (1 - void_fraction) * m
    return 1 / (1 / c * np.sqrt(numerator * denominator))


def pressure(
    x: np.ndarray,
    sensitivity: float,
    gain: float,
    volt_params: Optional[Tuple[int, float]] = None,
) -> np.ndarray:
    """
    Convert signal to acoustic pressure in micropascals.

    Parameters
    ----------
    x : ndarray
        Signal in voltage or bit depth
    sensitivity : float
        Receiving sensitivity in dB re 1V per micropascal
    gain : float
        Preamplifier gain in dB
    volt_params : tuple of (int, float), optional
        If provided, (nbits, v_ref) where nbits is number of bits per sample
        and v_ref is reference voltage. Used to convert bits to voltage.

    Returns
    -------
    ndarray
        Acoustic pressure signal in micropascals

    Examples
    --------
    >>> # Example with voltage input
    >>> p = pressure(x_volt, sensitivity=0, gain=0)

    >>> # Example with bit depth input
    >>> p = pressure(x_bits, sensitivity=0, gain=0, volt_params=(16, 1.0))
    """
    nu = 10 ** (sensitivity / 20)
    G = 10 ** (gain / 20)

    if volt_params is not None:
        nbits, v_ref = volt_params
        x = x * v_ref / (2 ** (nbits - 1))

    return x / (nu * G)


def spl(x: np.ndarray, ref: float = 1) -> float:
    """
    Calculate Sound Pressure Level (SPL) of acoustic pressure signal.

    Parameters
    ----------
    x : ndarray
        Acoustic pressure signal in micropascals
    ref : float, optional
        Reference acoustic pressure in micropascals (default: 1)
        For water: 1 µPa
        For air: 20 µPa

    Returns
    -------
    float
        Average SPL in dB re reference pressure

    Examples
    --------
    >>> pressure_signal = np.random.randn(1000) * 100  # Example pressure
    >>> spl_db = spl(pressure_signal, ref=1)
    >>> print(f"SPL: {spl_db:.2f} dB re 1 µPa")
    """
    rmsx = np.sqrt(np.mean(np.abs(x) ** 2))
    return 20 * np.log10(rmsx / ref)
