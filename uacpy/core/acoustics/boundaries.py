"""What a boundary does to a plane wave.

:func:`reflection_coeff` is the plane-wave coefficient of a fluid or an elastic
half-space — complex, so it carries the phase — and
:func:`uacpy.acoustics.bottom_loss_curve` (in
:mod:`uacpy.core.results.reflection`, as the
:class:`~uacpy.core.results.ReflectionCoefficient` it returns) sweeps it over
grazing angle for a named sediment. :func:`pekeris_root` picks the branch
of the complex square root that keeps a vertical wavenumber physical, which is
what a mode below cutoff needs to decay into the half-space instead of growing.
:func:`scholte_wave_speed` is the speed of the wave a liquid and a solid
half-space carry along their interface with no incident wave (the Scholte
wave); the derived ``c_low`` of the wavenumber engines counts it
(:func:`uacpy.core.engine_defaults.scholte_waves`).

The sediment sits against a water column, and :func:`reflection_coeff`
defaults it to the package's one water: the nominal ``DEFAULT_SOUND_SPEED``
(1500 m/s) and ``DEFAULT_WATER_DENSITY`` (1.027 g/cm³, the value every
deck writes). It speaks the package's units — grazing degrees, m/s, g/cm³,
dB/wavelength.

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


import warnings

import numpy as np
from typing import Union

from uacpy.core.constants import DEFAULT_SOUND_SPEED, DEFAULT_WATER_DENSITY
from uacpy.core.exceptions import ConfigurationError, FallbackWarning
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core._validate import DENSITY_UNITS_SUSPECT

__all__ = [
    'reflection_coeff',
    'critical_angle',
    'scholte_wave_speed',
    'pekeris_root',
]


def reflection_coeff(
    grazing_deg: Union[float, np.ndarray],
    *,
    sound_speed: float,
    density: float,
    attenuation: float = 0.0,
    shear_speed: float = 0.0,
    shear_attenuation: float = 0.0,
    water_sound_speed: float = DEFAULT_SOUND_SPEED,
    water_density: float = DEFAULT_WATER_DENSITY,
) -> Union[float, np.ndarray, complex]:
    """
    Plane-wave reflection coefficient of a fluid or an elastic half-space.

    A fluid half-space (``shear_speed=0``, the default) gives the Rayleigh
    coefficient, Brekhovskikh & Lysanov Eq. (3.1.12). A shear speed makes it
    an elastic (solid) half-space, Eq. (3.4.22): the incident wave also
    excites a transverse wave, which softens the boundary and lowers the
    reflection (B&L §3.4.1), and past the compressional critical angle a
    shear wave slower than the water carries energy away, so the reflection
    is no longer total.

    In the package's units, like :func:`bottom_loss_curve` and
    :class:`~uacpy.core.boundary.BoundaryProperties`: grazing angle in degrees,
    density in g/cm³, attenuation in dB/wavelength.

    Parameters
    ----------
    grazing_deg : float or array_like
        Grazing angle in degrees, ``0`` along the interface to ``90`` at
        normal incidence.
    sound_speed : float
        Compressional sound speed of the half-space (m/s).
    density : float
        Density of the half-space (g/cm³). A value above
        ``20`` g/cm³ is refused as a kg/m³ number in a g/cm³ slot.
    attenuation : float, optional
        Compressional attenuation of the half-space (dB/wavelength); default
        0 (lossless). Brekhovskikh & Lysanov write a lossy medium as
        ``n = n0·(1 + i·α)``; ``α = attenuation · ln(10) / (40·π)`` is that
        loss tangent — the factor Acoustics-Toolbox uses for its ``'L'``
        unit (``Bellhop/ReadEnvironmentBell.f90:527``).
    shear_speed : float, optional
        Shear (transverse) speed of the half-space (m/s); default 0, a
        fluid.
    shear_attenuation : float, optional
        Shear attenuation (dB per shear wavelength), entering the shear speed
        as ``attenuation`` enters the compressional one; default 0. Without
        a shear speed it has no wave to act on and is announced as unused
        (``FallbackWarning``).
    water_sound_speed, water_density : float, optional
        The water above (m/s, g/cm³). Default the package's one water,
        ``DEFAULT_SOUND_SPEED`` (1500 m/s) and
        ``DEFAULT_WATER_DENSITY`` (1.027 g/cm³), the water every deck
        writes. To reproduce a textbook curve drawn against ρ_w = 1 (Jensen
        et al. Table 1.3 quotes densities as ratios to it), pass
        ``water_density=1.0``.

    Returns
    -------
    float, ndarray, or complex
        Reflection coefficient as a linear multiplier, in the package's
        travelling-wave (``exp(+iωt)``) convention — the phase every engine's
        result and :class:`~uacpy.core.results.ReflectionCoefficient` carry,
        so it compares directly with Bounce and OASR. Brekhovskikh &
        Lysanov's formula below is written for ``exp(-iωt)``; the value
        returned is its complex conjugate.

    Examples
    --------
    >>> R = reflection_coeff(45.0, sound_speed=1600.0, density=1.2)
    >>> print(f"Reflection coefficient: {R:.4f}")
    Reflection coefficient: 0.1461

    References
    ----------
    Brekhovskikh, L. M. & Lysanov, Y. P. (2003). Fundamentals of Ocean Acoustics.
    Eq. (3.1.12) / (5.5.1): ``V = (m cos θ − √(n² − sin²θ)) / (m cos θ +
    √(n² − sin²θ))`` with ``m = ρ1/ρ``, ``n = c/c1`` and ``θ`` the angle from
    the normal; §3.1 gives the lossy convention ``n = n0(1 + iα), α > 0``.
    Eq. (3.4.22), the solid: ``V = (Z_in − Z)/(Z_in + Z)``, ``Z_in = Z1·cos²2γ1
    + Zt·sin²2γ1``, ``Z = ρc/cos θ``, ``Z1 = ρ1c1/cos θ1``, ``Zt =
    ρ1b1/cos γ1``, with ``sin θ1 = (c1/c) sin θ`` and ``sin γ1 = (b1/c) sin θ``
    (Eq. 3.4.24); past a critical angle the cosine is ``+i|cos|``, the wave
    bounded away from the boundary (§3.4.1 d).
    """
    grazing = np.asarray(grazing_deg, dtype=float)
    if np.any(grazing < 0.0) or np.any(grazing > 90.0 + 1e-9):
        raise ConfigurationError(
            f"reflection_coeff: grazing_deg is the grazing angle in degrees, "
            f"within [0, 90]; got {grazing_deg!r}.")
    for name, value in (('sound_speed', sound_speed), ('density', density),
                        ('water_sound_speed', water_sound_speed),
                        ('water_density', water_density)):
        if not (np.isfinite(value) and value > 0.0):
            raise ConfigurationError(
                f"reflection_coeff: {name} must be positive and finite; got "
                f"{value!r}.")
    for name, value in (('density', density), ('water_density', water_density)):
        if value > DENSITY_UNITS_SUSPECT:
            raise ConfigurationError(
                f"reflection_coeff: {name}={value:g} reads as kg/m³; the "
                f"argument is g/cm³.",
                remediation=f"Pass {name}={value / 1000.0:g}.")
    for name, value in (('attenuation', attenuation),
                        ('shear_attenuation', shear_attenuation)):
        if not (np.isfinite(value) and value >= 0.0):
            raise ConfigurationError(
                f"reflection_coeff: {name} must be a non-negative "
                f"dB/wavelength; got {value!r}.")
    if not (np.isfinite(shear_speed) and shear_speed >= 0.0):
        raise ConfigurationError(
            f"reflection_coeff: shear_speed must be >= 0 m/s and finite "
            f"(0 is a fluid); got {shear_speed!r}.")
    if shear_speed == 0.0 and shear_attenuation > 0.0:
        warnings.warn(
            f"reflection_coeff: shear_attenuation={shear_attenuation:g} is "
            f"not used: with shear_speed=0 the half-space is a fluid and "
            f"carries no shear wave.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
    # Brekhovskikh & Lysanov formulation, on the angle from the normal.
    # ``scimath.sqrt`` returns the complex principal value beyond critical
    # incidence (where ``n**2 - sin**2`` goes negative), +i|...|, the branch
    # B&L choose for a wave bounded away from the boundary; a real
    # ``np.sqrt`` would yield NaN there instead.
    angle = np.pi / 2.0 - np.deg2rad(grazing)
    sin2 = np.sin(angle) ** 2
    cos = np.cos(angle)
    alpha = float(attenuation) * np.log(10.0) / (40.0 * np.pi)
    n = float(water_sound_speed) / float(sound_speed) * (1 + 1j * alpha)
    m = float(density) / float(water_density)
    t1 = m * cos
    t2 = np.lib.scimath.sqrt(n**2 - sin2)
    if shear_speed > 0.0:
        # Eq. (3.4.22) over rho*c: Z = 1/cos θ, Z1 = m/t2, Zt·sin²2γ1 =
        # 4m·sin²θ·√(ns² − sin²θ)/ns⁴ and cos2γ1 = 1 − 2 sin²θ/ns², with
        # ns = c/b1 (lossy as n). Multiplied through by t2·cos θ, so the
        # grazing limit (cos θ = 0, V = −1) and the compressional critical
        # angle (t2 = 0, V = 1) stay finite.
        alpha_s = float(shear_attenuation) * np.log(10.0) / (40.0 * np.pi)
        ns = float(water_sound_speed) / float(shear_speed) * (1 + 1j * alpha_s)
        cos_2gamma = 1.0 - 2.0 * sin2 / ns**2
        shear = 4.0 * m * sin2 * np.lib.scimath.sqrt(ns**2 - sin2) / ns**4
        t1 = t1 * cos_2gamma**2 + t2 * shear * cos
    # B&L's V is in the exp(-iωt) convention; the package reports exp(+iωt).
    V = np.conj((t1 - t2) / (t1 + t2))

    return V.real if np.all(V.imag == 0) else V


def critical_angle(
    sound_speed: float,
    water_sound_speed: float = DEFAULT_SOUND_SPEED,
) -> float:
    """Grazing angle in degrees below which a faster seabed totally reflects.

    ``theta_c = arccos(c_water / c_bottom)``. Below it the transmitted wave is
    evanescent in the seabed and the plane-wave reflection loss is nominally
    zero; above it energy radiates in and the loss climbs. It is the same
    ratio-of-speeds angle as :meth:`uacpy.core.results.modes.Modes.grazing_angles`,
    seen from the boundary instead of from a mode, which is why the modal
    cutoff *is* the critical angle.

    Returns ``nan`` when the seabed is not faster than the water, because
    there is then no critical angle at all — a slow seabed (uacpy's ``clay``
    preset is 1500 m/s, under sea water) reflects weakly at every angle and
    shows an angle of *intromission* instead, where the loss peaks. Returning
    zero there would read as "totally reflecting everywhere", the opposite of
    the truth.

    :func:`uacpy.acoustics.bottom_loss_curve` carries this angle on the
    table it returns (``critical_angle_deg``), which its loss view rules
    (``mark_critical=True``), so the figure and the number are the same
    arithmetic.

    Parameters
    ----------
    sound_speed : float
        Speed of the seabed wave (m/s), the ``sound_speed`` of
        :func:`reflection_coeff`: the compressional speed for the
        compressional critical angle; a shear speed gives the shear one.
    water_sound_speed : float, optional
        Sound speed of the water at the seabed (m/s). Default
        :data:`~uacpy.core.constants.DEFAULT_SOUND_SPEED`.
    """
    c_b = float(sound_speed)
    c_w = float(water_sound_speed)
    if not (c_b > 0.0 and c_w > 0.0):
        raise ConfigurationError(
            f"critical_angle: both speeds must be > 0 m/s; got "
            f"bottom={c_b}, water={c_w}.")
    if c_b <= c_w:
        return float('nan')
    return float(np.degrees(np.arccos(c_w / c_b)))


#: Bisection steps of :func:`scholte_wave_speed`: each halves the bracket
#: on ``s = (v/c_s)²``, which starts inside ``(0, 1]``, so 100 steps leave it
#: below double-precision spacing.
_SCHOLTE_BISECTIONS = 100

#: The largest shear-to-compressional speed ratio of an isotropic solid,
#: ``c_s/c_p < √3/2``: ``c_p² = (B + 4G/3)/ρ`` and ``c_s² = G/ρ`` (Garrett,
#: *Understanding Acoustics*, §4.2.3 and Table 4.1), and a positive bulk
#: modulus ``B`` (Poisson's ratio above −1, §4.2.3) keeps ``G/(B + 4G/3)``
#: under 3/4.
_MAX_SHEAR_TO_SOUND_SPEED = np.sqrt(3.0) / 2.0


def scholte_wave_speed(
    sound_speed: Union[float, np.ndarray],
    shear_speed: Union[float, np.ndarray],
    density: Union[float, np.ndarray],
    *,
    water_sound_speed: Union[float, np.ndarray] = DEFAULT_SOUND_SPEED,
    water_density: Union[float, np.ndarray] = DEFAULT_WATER_DENSITY,
) -> Union[float, np.ndarray]:
    """Phase speed (m/s) of the Scholte wave: the wave a liquid half-space
    and a solid one carry along their interface, both lossless.

    Jensen et al. (*Computational Ocean Acoustics*, §8.5.4) name the
    interface wave by its media: "a Scholte wave when propagating along a
    fluid-solid interface, and a Stoneley wave when associated with a
    solid-solid boundary"; this is the first. Brekhovskikh & Lysanov,
    *Fundamentals of Ocean Acoustics*, §3.4.2, eq. 3.4.29: ``(ρ/ρ₁) s²
    (1 − sq)^½ (1 − sr)^−½ = 4 (1 − s)^½ (1 − sq)^½ − (2 − s)²`` with
    ``s = (v/c_s)²``, ``q = (c_s/c_p)²``, ``r = (c_s/c)²`` (``c``, ``ρ`` the
    liquid's; ``c_p``, ``c_s``, ``ρ₁`` the solid's), "which always has one
    real root s < 1, sr < 1": the wave is slower than both the solid's
    shear wave and the liquid, its speed does not depend on frequency, and
    its amplitude decays away from the interface in both media (§3.4.2).
    ``water_density=0`` gives the Rayleigh wave of the solid's free surface
    ("in the particular case ϱ/ϱ₁ = 0, (3.4.29) reduces to the equation for
    the velocity of the Rayleigh wave"). Jensen et al. (§4.10.1) place it at
    k_r = 0.24 m⁻¹ at 20 Hz (about 524 m/s) for 1500 m/s water over sand of
    c_p 1800 m/s, c_s 600 m/s and 1800 kg/m³, and at 0.47 m⁻¹ for c_s
    300 m/s.

    The solid is named as on :func:`reflection_coeff` (``sound_speed``,
    ``shear_speed``, ``density``), and so is the liquid above it
    (``water_*``, defaulting to the package's one water): the water, or a
    fluid sediment layer on an elastic one. Below the root the right-hand
    side exceeds the left (``2s(1 − q)`` against ``O(s²)`` near 0) and
    above it the left wins, so bisection on ``(0, min(1, 1/r))`` finds it.
    The inputs broadcast against each other (numpy rules); a scalar call
    returns a float.

    Parameters
    ----------
    sound_speed : float or array_like
        Compressional speed of the solid (m/s), > 0.
    shear_speed : float or array_like
        Shear speed of the solid (m/s), ≥ 0.
    density : float or array_like
        Density of the solid (g/cm³), > 0.
    water_sound_speed : float or array_like, optional
        Sound speed of the liquid (m/s), > 0. Default
        :data:`~uacpy.core.constants.DEFAULT_SOUND_SPEED`.
    water_density : float or array_like, optional
        Density of the liquid (g/cm³), ≥ 0. Default
        :data:`~uacpy.core.constants.DEFAULT_WATER_DENSITY`. Only its
        ratio to ``density`` enters.

    Returns
    -------
    float or ndarray
        The Scholte-wave speed (m/s); ``nan`` where the solid carries no
        shear (``shear_speed = 0``: a liquid-liquid interface carries no
        interface wave, Jensen et al. §8.5.4 "at least one of the media
        must be a solid") or where ``shear_speed ≥ (√3/2)·sound_speed``,
        which no real solid has (elastic stability keeps ``c_s <
        (√3/2)·c_p``, Garrett, *Understanding Acoustics*, §4.2.3), and
        where ``2s(1 − q)`` is not positive near 0, so the root is not
        bracketed.

    Raises
    ------
    ConfigurationError
        A speed or density that is not finite, or outside the ranges above.
    """
    names = ('sound_speed', 'shear_speed', 'density', 'water_sound_speed',
             'water_density')
    cp, cs, rho_solid, c, rho = np.broadcast_arrays(*(
        np.asarray(v, dtype=float) for v in (
            sound_speed, shear_speed, density, water_sound_speed,
            water_density)))
    for name, value, allowed in zip(
            names, (cp, cs, rho_solid, c, rho),
            (cp > 0.0, cs >= 0.0, rho_solid > 0.0, c > 0.0, rho >= 0.0)):
        if not np.all(np.isfinite(value) & allowed):
            bound = '≥ 0' if name in ('shear_speed', 'water_density') else '> 0'
            raise ConfigurationError(
                f"scholte_wave_speed: {name} must be finite and {bound}; "
                f"got {value[~(np.isfinite(value) & allowed)].ravel()[0]:g}.")

    with np.errstate(divide='ignore'):
        q, r, ratio = (cs / cp) ** 2, (cs / c) ** 2, rho / rho_solid
        high = np.minimum(1.0, np.minimum(1.0 / r, 1.0 / q)) * (1.0 - 1e-12)

    def excess(s):
        with np.errstate(invalid='ignore'):
            return (4.0 * np.sqrt((1.0 - s) * (1.0 - s * q)) - (2.0 - s) ** 2
                    - ratio * s * s * np.sqrt(1.0 - s * q)
                    / np.sqrt(1.0 - s * r))

    low = np.full(c.shape, 1e-9)
    bracketed = ((cs > 0.0) & (cs < _MAX_SHEAR_TO_SOUND_SPEED * cp)
                 & (excess(low) > 0.0)
                 & (excess(high) < 0.0))
    for _ in range(_SCHOLTE_BISECTIONS):
        middle = 0.5 * (low + high)
        above = excess(middle) > 0.0
        low = np.where(above, middle, low)
        high = np.where(above, high, middle)
    speed = np.where(bracketed, cs * np.sqrt(0.5 * (low + high)), np.nan)
    return float(speed) if speed.ndim == 0 else speed


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
