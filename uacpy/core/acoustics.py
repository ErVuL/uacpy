"""
Underwater acoustics utilities for UACPY

This module provides various underwater acoustics functions including:
- Sound speed calculations
- Ambient noise modeling (Wenz model)
- Bubble acoustics
- Absorption calculations
- Acoustic pressure utilities

Note
----
This module is a **user helper**.  The uacpy model wrappers do not
import from ``uacpy.core.acoustics``; it is provided for downstream
notebooks/examples that need direct access to sound-speed / absorption
formulas (e.g. example_12_attenuation_models.py).

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

import warnings as _warnings

import numpy as np
from typing import Union, Optional, Tuple

from uacpy.core.constants import PRESSURE_FLOOR, REFERENCE_PRESSURE_WATER

__all__ = [
    'soundspeed',
    'soundspeed_unesco',
    'soundspeed_delgrosso',
    'density',
    'doppler',
    'reflection_coeff',
    'bottom_loss_curve',
    'bubble_resonance',
    'bubble_surface_loss',
    'bubble_soundspeed',
    'pressure',
    'spl',
    'power_to_db',
    'pekeris_root',
]


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

    Notes
    -----
    Mackenzie's nine-term formula is validated for
    ``temperature ∈ [-2, 30] °C``, ``salinity ∈ [25, 40] PSU``,
    ``depth ∈ [0, 8000] m``. Values outside these ranges trigger a
    :class:`UserWarning` and the formula's output should be treated as
    extrapolation.

    References
    ----------
    Mackenzie, K. V. (1981). "Nine-term equation for sound speed in the oceans".
    The Journal of the Acoustical Society of America, 70(3), 807-812.
    """
    if np.any(np.asarray(temperature) < -2) or np.any(np.asarray(temperature) > 30):
        _warnings.warn(
            "Mackenzie soundspeed: temperature outside validated range "
            "[-2, 30] °C; treating as extrapolation.",
            UserWarning, stacklevel=2,
        )
    if np.any(np.asarray(salinity) < 25) or np.any(np.asarray(salinity) > 40):
        _warnings.warn(
            "Mackenzie soundspeed: salinity outside validated range "
            "[25, 40] PSU; treating as extrapolation.",
            UserWarning, stacklevel=2,
        )
    if np.any(np.asarray(depth) < 0) or np.any(np.asarray(depth) > 8000):
        _warnings.warn(
            "Mackenzie soundspeed: depth outside validated range "
            "[0, 8000] m; treating as extrapolation.",
            UserWarning, stacklevel=2,
        )
    c = (
        1448.96
        + 4.591 * temperature
        - 5.304e-2 * temperature**2
        + 2.374e-4 * temperature**3
    )
    c += 1.340 * (salinity - 35) + 1.630e-2 * depth + 1.675e-7 * depth**2
    c += -1.025e-2 * temperature * (salinity - 35) - 7.139e-13 * temperature * depth**3
    return c


def soundspeed_unesco(temperature=15.0, salinity=35.0, pressure=0.0):
    """Speed of sound in seawater — UNESCO (Chen & Millero 1977 / UNESCO 1983).

    The international standard algorithm. ``pressure`` is in **decibars** (the
    form the equation is defined in; ≈ 1 dbar per metre of depth). Temperature is
    ITS-90 (converted to the IPTS-68 scale the polynomial expects internally).
    Valid for ``T ∈ [0, 40] °C``, ``S ∈ [0, 40] PSU``, ``P ∈ [0, 1000] bar``.

    Parameters
    ----------
    temperature : float
        Temperature [°C, ITS-90].
    salinity : float
        Practical salinity [PSU, PSS-78].
    pressure : float
        Pressure [dbar].

    Returns
    -------
    float
        Sound speed [m/s].

    References
    ----------
    Chen, C.-T. & Millero, F. J. (1977). "Speed of sound in seawater at high
    pressures." JASA 62(5), 1129-1135. UNESCO (1983) Technical Papers in Marine
    Science 44, Eqns 33-37.
    """
    t = np.asarray(temperature, dtype=float)
    s = np.asarray(salinity, dtype=float)
    p = np.asarray(pressure, dtype=float) / 10.0       # dbar -> bar
    t68 = t * 1.00024                                  # ITS-90 -> IPTS-68

    # Eqn 34: pure-water term Cw(T, P)
    c00, c01, c02, c03, c04, c05 = (1402.388, 5.03711, -5.80852e-2,
                                    3.3420e-4, -1.47800e-6, 3.1464e-9)
    c10, c11, c12, c13, c14 = (0.153563, 6.8982e-4, -8.1788e-6,
                               1.3621e-7, -6.1185e-10)
    c20, c21, c22, c23, c24 = (3.1260e-5, -1.7107e-6, 2.5974e-8,
                               -2.5335e-10, 1.0405e-12)
    c30, c31, c32 = (-9.7729e-9, 3.8504e-10, -2.3643e-12)
    cw = ((((c32 * t68 + c31) * t68 + c30) * p
           + ((((c24 * t68 + c23) * t68 + c22) * t68 + c21) * t68 + c20)) * p
          + ((((c14 * t68 + c13) * t68 + c12) * t68 + c11) * t68 + c10)) * p \
        + ((((c05 * t68 + c04) * t68 + c03) * t68 + c02) * t68 + c01) * t68 + c00

    # Eqn 35: A(T, P)
    a00, a01, a02, a03, a04 = (1.389, -1.262e-2, 7.164e-5, 2.006e-6, -3.21e-8)
    a10, a11, a12, a13, a14 = (9.4742e-5, -1.2580e-5, -6.4885e-8,
                               1.0507e-8, -2.0122e-10)
    a20, a21, a22, a23 = (-3.9064e-7, 9.1041e-9, -1.6002e-10, 7.988e-12)
    a30, a31, a32 = (1.100e-10, 6.649e-12, -3.389e-13)
    a = ((((a32 * t68 + a31) * t68 + a30) * p
          + (((a23 * t68 + a22) * t68 + a21) * t68 + a20)) * p
         + ((((a14 * t68 + a13) * t68 + a12) * t68 + a11) * t68 + a10)) * p \
        + (((a04 * t68 + a03) * t68 + a02) * t68 + a01) * t68 + a00

    # Eqn 36/37: B(T, P), D(P)
    b = -1.922e-2 - 4.42e-5 * t68 + (7.3637e-5 + 1.7945e-7 * t68) * p
    d = 1.727e-3 - 7.9836e-6 * p

    c = cw + a * s + b * s ** 1.5 + d * s ** 2
    return float(c) if np.ndim(c) == 0 else c


def soundspeed_delgrosso(temperature=15.0, salinity=35.0, pressure=0.0):
    """Speed of sound in seawater — Del Grosso (1974) "NRL II" equation.

    An alternative to UNESCO, often preferred at high pressure / in deep water.
    ``pressure`` is accepted in **decibars** and converted to the kg/cm² the
    original equation uses (``1 kg/cm² = 9.80665 dbar``). Temperature in °C,
    salinity in PSU. Standard deviation 0.05 m/s; valid for realistic ``(S, T,
    P)`` combinations (``T`` to 35 °C, ``S`` 29-41 PSU).

    References
    ----------
    Del Grosso, V. A. (1974). "New equation for the speed of sound in natural
    waters (with comparisons to other equations)." JASA 56(4), 1084-1091.
    """
    t = np.asarray(temperature, dtype=float)
    s = np.asarray(salinity, dtype=float)
    p = np.asarray(pressure, dtype=float) / 9.80665     # dbar -> kg/cm²

    c000 = 1402.392
    dct = (0.501109398873e1 * t - 0.550946843172e-1 * t ** 2
           + 0.221535969240e-3 * t ** 3)
    dcs = 0.132952290781e1 * s + 0.128955756844e-3 * s ** 2
    dcp = (0.156059257041e0 * p + 0.244998688441e-4 * p ** 2
           - 0.883392332513e-8 * p ** 3)
    dcstp = (-0.127562783426e-1 * t * s
             + 0.635191613389e-2 * t * p
             + 0.265484716608e-7 * t ** 2 * p ** 2
             - 0.159349479045e-5 * t * p ** 2
             + 0.522116437235e-9 * t * p ** 3
             - 0.438031096213e-6 * t ** 3 * p
             - 0.161674495909e-8 * s ** 2 * p ** 2
             + 0.968403156410e-4 * t ** 2 * s
             + 0.485639620015e-5 * t * s ** 2 * p
             - 0.340597039004e-3 * s * t * p)
    c = c000 + dct + dcs + dcp + dcstp
    return float(c) if np.ndim(c) == 0 else c


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

    # Brekhovskikh & Lysanov formulation. ``scimath.sqrt`` returns the
    # complex principal value beyond critical incidence (where
    # ``n**2 - sin**2`` goes negative); a real ``np.sqrt`` would yield NaN
    # there instead of the physical totally-reflecting branch (``|V|=1``).
    n = float(c) / c1 * (1 + 1j * alpha)
    m = float(rho1) / rho
    t1 = m * np.cos(angle)
    t2 = np.lib.scimath.sqrt(n**2 - np.sin(angle) ** 2)
    V = (t1 - t2) / (t1 + t2)

    return V.real if np.all(V.imag == 0) else V


def bottom_loss_curve(
    material: Union[str, dict],
    *,
    grazing_angles_deg: Optional[np.ndarray] = None,
    water_speed: float = 1500.0,
    water_density: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Plane-wave fluid–fluid bottom loss vs grazing angle.

    Wraps :func:`reflection_coeff` with the property dict from
    :mod:`uacpy.core.materials`, returning grazing-angle / loss-in-dB
    arrays ready to plot. Shear is ignored (fluid–fluid only).

    Parameters
    ----------
    material : str or dict
        Preset name (``'sand'``, ``'silt'``, …) or a dict carrying
        ``sound_speed`` (m/s), ``density`` (g/cm³), ``attenuation``
        (dB/λ_p).
    grazing_angles_deg : array_like, optional
        Grazing-angle grid in degrees (``0`` = parallel to interface,
        ``90`` = normal incidence). Default ``np.linspace(0, 90, 181)``.
    water_speed, water_density : float
        Water-column reference values (m/s, g/cm³).

    Returns
    -------
    grazing_angles_deg : ndarray
    loss_db : ndarray
        Bottom loss ``-20·log10|R|`` at each angle.
    """
    if isinstance(material, str):
        from uacpy.core.materials import get_material
        m = get_material(material)
    else:
        m = dict(material)
    if grazing_angles_deg is None:
        grazing_angles_deg = np.linspace(0.0, 90.0, 181)
    grazing = np.deg2rad(np.asarray(grazing_angles_deg, dtype=float))
    angle_from_normal = np.pi / 2.0 - grazing
    alpha = float(m['attenuation']) * np.log(10.0) / (40.0 * np.pi)
    R = reflection_coeff(
        angle=angle_from_normal,
        rho1=float(m['density']) * 1000.0,
        c1=float(m['sound_speed']),
        alpha=alpha,
        rho=float(water_density) * 1000.0,
        c=float(water_speed),
    )
    loss_db = -20.0 * np.log10(np.abs(R) + 1e-300)
    return np.asarray(grazing_angles_deg, dtype=float), np.asarray(loss_db, dtype=float)


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
        Surface reflection as a linear amplitude multiplier in ``(0, 1]``
        (1.0 = no loss). To express the loss as a **positive** dB number,
        consistent with :func:`bottom_loss_curve`, negate the log:
        ``loss_db = -20 * np.log10(multiplier)``.

    Examples
    --------
    >>> mult = bubble_surface_loss(3, 10000, 0)
    >>> loss_db = -20 * np.log10(mult)   # positive dB loss
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


def power_to_db(power, ref: float = REFERENCE_PRESSURE_WATER, *,
                floor: float = PRESSURE_FLOOR):
    """Mean-square / power-like pressure quantity → level in dB re ``ref``.

    For a *squared* quantity (PSD in Pa²/Hz, SEL in Pa²·s, mean-square
    pressure, an f-k spectrum, …) the level is ``10·log10(power / ref²)``. The
    single conversion every spectral estimator should use: ``power`` is floored
    at ``floor`` before the log so a silent (zero) sample yields a finite, very
    negative level instead of ``-inf`` (which would otherwise poison a
    subsequent ``mean`` / ``histogram``).

    Parameters
    ----------
    power : array_like
        Squared-pressure quantity (e.g. PSD, SEL, |p|²); same units as
        ``ref**2``.
    ref : float, optional
        Reference pressure (default: 1 µPa, water). Use
        ``REFERENCE_PRESSURE_AIR`` for air.
    floor : float, optional
        Lower bound applied to ``power`` before the log (default
        :data:`PRESSURE_FLOOR`), guarding ``log10(0)``.

    Returns
    -------
    numpy.ndarray
        Level in dB re ``ref``.
    """
    power = np.asarray(power, dtype=float)
    return 10.0 * np.log10(np.maximum(power, floor) / (ref ** 2))


def pekeris_root(gamma2: np.ndarray) -> np.ndarray:
    """
    Return the Pekeris branch of the complex square root.

    ``sqrt(gamma2)`` for ``Re(gamma2) >= 0``, ``i*sqrt(-gamma2)``
    otherwise — the branch with ``Re(gamma) >= 0`` on the right half
    plane and continuous across the negative real axis, enforcing
    exponential decay of the halfspace solution ``exp(-gamma*(z - D))``
    for trapped modes.

    Parameters
    ----------
    gamma2 : ndarray, complex
        Squared vertical wavenumber, ``gamma^2 = k^2 - k_halfspace^2``.

    Returns
    -------
    gamma : ndarray, complex
        Vertical wavenumber on the Pekeris branch.

    References
    ----------
    Pekeris, C.L., "Theory of propagation of explosive sound in shallow
    water," Geol. Soc. Am. Mem. 27 (1948).

    Adapted from Acoustics-Toolbox ``Matlab/Kraken/PekerisRoot.m``
    (M.B. Porter, 04/2009). Not an arlpy-derived helper — see
    ``third_party/arlpy/NOTICE`` for the arlpy-attributed list.
    """
    gamma2 = np.asarray(gamma2, dtype=complex)
    return np.where(
        np.real(gamma2) >= 0.0,
        np.sqrt(gamma2),
        1j * np.sqrt(-gamma2),
    )
