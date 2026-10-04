"""Bubble acoustics: resonance, bubbly-water speed, surface bubble loss.

A bubble is a mass-spring resonator, and a cloud of them is the strongest
scatterer and the strongest attenuator in the upper ocean.
:func:`bubble_resonance` gives the Minnaert frequency of one bubble,
:func:`bubble_sound_speed` the speed through a void-fraction mixture (Wood's
equation: air's compressibility with water's density, so at intermediate void
fractions the mixture carries sound slower than either pure phase), and
:func:`bubble_surface_loss` the per-bounce loss a wind-driven layer adds to a
surface reflection.

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
from typing import Union, Optional

from uacpy.core.exceptions import ConfigurationError
from uacpy.core.units import knots_to_ms
from uacpy.core.constants import (
    DEFAULT_SOUND_SPEED, DEFAULT_WATER_DENSITY_G_CM3, STANDARD_ATMOSPHERE_PA,
    STANDARD_GRAVITY_M_S2,
)


#: Air's ratio of specific heats, the default bubble gas.
AIR_HEAT_CAPACITY_RATIO = 1.4

__all__ = [
    'AIR_HEAT_CAPACITY_RATIO',
    'bubble_resonance',
    'bubble_surface_loss',
    'bubble_sound_speed',
]


def bubble_resonance(
    radius: Union[float, np.ndarray],
    depth: float = 0.0,
    gamma: float = AIR_HEAT_CAPACITY_RATIO,
    p0: float = STANDARD_ATMOSPHERE_PA,
    water_density_kg_m3: float = DEFAULT_WATER_DENSITY_G_CM3 * 1000.0,
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
        Atmospheric pressure in Pa (default: the standard atmosphere,
        101325)
    water_density_kg_m3 : float, optional
        Density of water in kg/m³ (default: 1027, the package's one water
        density ``DEFAULT_WATER_DENSITY_G_CM3`` in kg/m³)

    Returns
    -------
    float or ndarray
        Resonance frequency in Hz

    Examples
    --------
    >>> f_res = bubble_resonance(100e-6)  # 100 micron radius
    >>> print(f"Resonance frequency: {f_res:.2f} Hz")
    Resonance frequency: 32397.97 Hz

    >>> radii = np.logspace(-5, -3, 50)  # 10 to 1000 microns
    >>> f_res = bubble_resonance(radii)

    References
    ----------
    Medwin, H. & Clay, C. S. (1998). Fundamentals of Acoustical Oceanography,
    eq. (8.2.13): ``f_h = (1/2πa)·√(3γ p_A/ρ_A)`` with ``p_A = p_A0 + ρ_A g z``.
    Valid while ``ka ≲ 1``.
    """
    g = STANDARD_GRAVITY_M_S2
    p_air = p0 + water_density_kg_m3 * g * depth
    return (1 / (2 * np.pi * radius)
            * np.sqrt(3 * gamma * p_air / water_density_kg_m3))


def bubble_surface_loss(
    wind_speed_kn: float,
    frequency: Union[float, np.ndarray],
    grazing_deg: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    Calculate surface loss due to bubbles using APL model (1994).

    Parameters
    ----------
    wind_speed_kn : float
        Wind speed in knots (measured 10 m above sea surface), converted to
        the handbook's m/s at entry. Non-negative.
    frequency : float or array_like
        Frequency in Hz. Positive.
    grazing_deg : float or array_like
        Grazing angle of the surface-bounce path in degrees, measured from
        the surface — ``0`` along it, ``90`` at normal incidence — the angle
        the handbook writes the model against, and the convention of
        :func:`~uacpy.core.acoustics.reflection_coeff` and the
        :mod:`uacpy.sonar` scattering functions. Within ``[0, 90]``: below
        0 ``sin`` turns negative and the formula returns a multiplier
        greater than 1, a surface that amplifies.

    Returns
    -------
    float or ndarray
        Surface reflection as a linear amplitude multiplier in ``[0, 1]``
        (1.0 = no loss). To express the loss as a **positive** dB number,
        consistent with :func:`bottom_loss_curve`, negate the log:
        ``loss_dB = -20 * np.log10(multiplier)``. ``grazing_deg = 0`` is
        the ``1/sin(theta) -> inf`` limit and returns 0.0.

    Raises
    ------
    ConfigurationError
        For a negative wind_speed_kn, a non-positive frequency or a
        grazing angle outside ``[0, 90]`` degrees. Each has a formula branch that answers
        anyway rather than failing: a negative wind_speed_kn takes the ``U < 6``
        branch and reports ~1.0, i.e. no loss; a negative frequency raises a
        negative base to 0.85 and returns a *complex* multiplier; and a zero
        one reports no loss.

    Examples
    --------
    >>> mult = bubble_surface_loss(3, 10000, grazing_deg=90)
    >>> loss_dB = -20 * np.log10(mult)   # positive dB loss
    >>> print(f"Surface loss: {loss_dB:.2f} dB")
    Surface loss: 0.00 dB

    Notes
    -----
    Surface bubble loss ``SBL`` (APL-UW TR 9407 §II.C.4 "Absorption Due to
    Near-Surface Bubbles", eqs. 28a/28b, p. II-21):

    ``SBL = 1.26e-3/sin(theta) * U**1.57 * f**0.85``   for ``U >= 6 m/s``,
    ``SBL = SBL(U=6) * exp(1.2*(U-6))``                for ``U < 6 m/s``,

    with ``U`` the wind speed 10 m above the surface, ``f`` in kHz and
    ``theta`` the nominal grazing angle of the surface-bounce path (θ > 0°).
    The 6 m/s break is the breaking-wave (Beaufort) threshold below which
    bubbles are not produced. Eq. 27 (p. II-20) defines the loss as a power
    ratio, ``RL(dB) = SBL(dB) = -10 log10(a_b)``, which the ``10**(-SBL/20)``
    below turns into the amplitude multiplier ``sqrt(a_b)``. The handbook fits
    20-40 kHz data to within +/-3 dB. It gives 30 dB as a nominal ceiling on
    ``SBL`` (scattering off the underside of a uniform bubble layer), a bound
    it reports as never observed in data, and recommends ``SBLmax = 15 dB``
    in practice (p. II-21); no ceiling is imposed here.

    References
    ----------
    APL-UW (1994). "APL-UW High-Frequency Ocean Environmental Acoustic Models
    Handbook". Technical Report APL-UW TR 9407, sec. II.C.4.
    """
    if not np.isfinite(wind_speed_kn) or wind_speed_kn < 0:
        raise ConfigurationError(
            f"bubble_surface_loss: wind_speed_kn must be a non-negative finite "
            f"speed in knots; got {wind_speed_kn!r}. A negative one takes the "
            f"U < 6 m/s branch and comes back as a multiplier near 1.0 — "
            f"reported as almost no surface loss.")
    f_hz = np.asarray(frequency, dtype=float)
    if not np.all(np.isfinite(f_hz)) or np.any(f_hz <= 0):
        raise ConfigurationError(
            f"bubble_surface_loss: frequency must be positive and finite "
            f"(Hz); got {frequency!r}. f**0.85 takes a negative frequency to "
            f"a complex multiplier and a zero one to 1.0, i.e. no loss. The "
            f"handbook fit is to 20-40 kHz data.")
    grazing = np.asarray(grazing_deg, dtype=float)
    if (not np.all(np.isfinite(grazing)) or np.any(grazing < 0.0)
            or np.any(grazing > 90.0)):
        raise ConfigurationError(
            f"bubble_surface_loss: grazing_deg is the grazing angle in "
            f"degrees from the surface and must lie in [0, 90]; got "
            f"{grazing_deg!r}. Below 0, sin(theta) goes negative and the "
            f"multiplier comes back above 1 — a surface that amplifies the "
            f"ray it reflects.")

    sin_theta = np.sin(np.deg2rad(grazing))
    f = f_hz / 1000.0  # Convert to kHz
    wind_ms = float(knots_to_ms(wind_speed_kn))    # the handbook's U, in m/s

    # Grazing incidence is sin(theta) = 0, where SBL diverges and the
    # multiplier goes to its 0.0 limit; the division states that rather than
    # tripping a RuntimeWarning on the way there.
    with np.errstate(divide='ignore'):
        if wind_ms >= 6:
            a = 1.26e-3 / sin_theta * wind_ms**1.57 * f**0.85
        else:
            a = 1.26e-3 / sin_theta * 6**1.57 * f**0.85 * np.exp(1.2 * (wind_ms - 6))

    return 10 ** (-a / 20.0)


def bubble_sound_speed(
    void_fraction: Union[float, np.ndarray],
    sound_speed: Optional[float] = None,
    c_gas: float = 340,
    relative_density: float = 1000,
) -> Union[float, np.ndarray]:
    """
    Calculate speed of sound in 2-phase bubbly water.

    Wood's equation (Wood 1964; Medwin & Clay eq. 8.3.39).

    Parameters
    ----------
    void_fraction : float or array_like
        Void fraction (ratio of gas volume to total volume)
    sound_speed : float, optional
        Speed of sound in water in m/s (default: ``DEFAULT_SOUND_SPEED``,
        the nominal 1500 m/s)
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
    >>> c_bubbly = bubble_sound_speed(1e-5)
    >>> print(f"Sound speed in bubbly water: {c_bubbly:.2f} m/s")
    Sound speed in bubbly water: 1372.39 m/s

    Notes
    -----
    Wood's equation as given by Medwin & Clay eq. (8.3.39): the mixture takes the
    volume-averaged density ``rho_A = U·rho_b + (1-U)·rho_w`` and the
    volume-averaged compressibility ``1/E_A = U/E_b + (1-U)/E_w``, with
    ``c = sqrt(E_A/rho_A)``. It is the LOW-FREQUENCY limit: Medwin & Clay
    give it for all void fractions and independent of the bubble size
    distribution, but only at frequencies well below the bubbles' resonance
    (:func:`bubble_resonance`). Near and above resonance the mixture is
    dispersive and this speed does not apply; the function takes no
    frequency, so staying below resonance is the caller's to check.

    References
    ----------
    Wood, A. B. (1964). A Textbook of Sound.
    Medwin, H. & Clay, C. S. (1998). Fundamentals of Acoustical
    Oceanography, sect. 8.3.3, eq. (8.3.39).
    """
    if sound_speed is None:
        sound_speed = DEFAULT_SOUND_SPEED

    # Splitting ``relative_density`` as m = sqrt(rho_w/rho_gas) puts the two
    # averages on a common footing: ``numerator`` is (rho_w c_w²)/(rho_A c_A²)/m
    # and ``denominator`` is m·rho_A/rho_w, so their product is exactly
    # (c_w/c_A)² — the density ratio and the factor m both cancel, leaving one
    # square root instead of two.
    m = np.sqrt(relative_density)
    numerator = (void_fraction * (sound_speed / c_gas) ** 2 * m
                 + (1 - void_fraction) / m)
    denominator = void_fraction / m + (1 - void_fraction) * m
    return 1 / (1 / sound_speed * np.sqrt(numerator * denominator))
