"""What a boundary does to a plane wave.

:func:`reflection_coeff` is the Rayleigh coefficient for a fluid half-space —
complex, so it carries the phase — and :func:`bottom_loss_curve` sweeps it
over grazing angle for a named sediment. :func:`pekeris_root` picks the branch
of the complex square root that keeps a vertical wavenumber physical, which is
what a mode below cutoff needs to decay into the half-space instead of growing.

The sediment sits against a water column, and both entry points default it
to the package's one water: the nominal ``DEFAULT_SOUND_SPEED`` (1500 m/s) and
``DEFAULT_WATER_DENSITY_G_CM3`` (1.027 g/cm³, the value every deck writes).
Both speak the package's units — grazing degrees, m/s, g/cm³, dB/wavelength —
and :func:`bottom_loss_curve` is :func:`reflection_coeff` swept over angle for
a named or carried seabed.

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


from collections.abc import Mapping

import numpy as np
from typing import Union, Optional, Tuple

from uacpy.core.constants import DEFAULT_SOUND_SPEED, DEFAULT_WATER_DENSITY_G_CM3
from uacpy.core.exceptions import ConfigurationError

__all__ = [
    'reflection_coeff',
    'bottom_loss_curve',
    'critical_angle',
    'pekeris_root',
]


def reflection_coeff(
    grazing_deg: Union[float, np.ndarray],
    *,
    sound_speed: float,
    density: float,
    attenuation: float = 0.0,
    water_sound_speed: float = DEFAULT_SOUND_SPEED,
    water_density: float = DEFAULT_WATER_DENSITY_G_CM3,
) -> Union[float, np.ndarray, complex]:
    """
    Rayleigh plane-wave reflection coefficient of a fluid half-space.

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
    water_sound_speed, water_density : float, optional
        The water above (m/s, g/cm³). Default the package's one water,
        ``DEFAULT_SOUND_SPEED`` (1500 m/s) and
        ``DEFAULT_WATER_DENSITY_G_CM3`` (1.027 g/cm³), the water every deck
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
        if value > 20.0:
            raise ConfigurationError(
                f"reflection_coeff: {name}={value:g} reads as kg/m³; the "
                f"argument is g/cm³.",
                remediation=f"Pass {name}={value / 1000.0:g}.")
    if not (np.isfinite(attenuation) and attenuation >= 0.0):
        raise ConfigurationError(
            f"reflection_coeff: attenuation must be a non-negative "
            f"dB/wavelength; got {attenuation!r}.")
    # Brekhovskikh & Lysanov formulation, on the angle from the normal.
    # ``scimath.sqrt`` returns the complex principal value beyond critical
    # incidence (where ``n**2 - sin**2`` goes negative); a real ``np.sqrt``
    # would yield NaN there instead of the totally-reflecting branch.
    angle = np.pi / 2.0 - np.deg2rad(grazing)
    alpha = float(attenuation) * np.log(10.0) / (40.0 * np.pi)
    n = float(water_sound_speed) / float(sound_speed) * (1 + 1j * alpha)
    m = float(density) / float(water_density)
    t1 = m * np.cos(angle)
    t2 = np.lib.scimath.sqrt(n**2 - np.sin(angle) ** 2)
    # B&L's V is in the exp(-iωt) convention; the package reports exp(+iωt).
    V = np.conj((t1 - t2) / (t1 + t2))

    return V.real if np.all(V.imag == 0) else V


def critical_angle(
    bottom_sound_speed: float,
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

    :func:`uacpy.plot.plot_bottom_loss` rules this angle on its axes
    by calling here, so the figure and the number are the same arithmetic.

    Parameters
    ----------
    bottom_sound_speed : float
        Compressional speed of the seabed (m/s).
    water_sound_speed : float, optional
        Sound speed of the water at the seabed (m/s). Default
        :data:`~uacpy.core.constants.DEFAULT_SOUND_SPEED`.
    """
    c_b = float(bottom_sound_speed)
    c_w = float(water_sound_speed)
    if not (c_b > 0.0 and c_w > 0.0):
        raise ConfigurationError(
            f"critical_angle: both speeds must be > 0 m/s; got "
            f"bottom={c_b}, water={c_w}.")
    if c_b <= c_w:
        return float('nan')
    return float(np.degrees(np.arccos(c_w / c_b)))


def bottom_loss_curve(
    material,
    *,
    grazing_angles_deg: Optional[np.ndarray] = None,
    water_sound_speed: float = DEFAULT_SOUND_SPEED,
    water_density: float = DEFAULT_WATER_DENSITY_G_CM3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Plane-wave fluid–fluid bottom loss vs grazing angle.

    Wraps :func:`reflection_coeff` with the property dict from
    :mod:`uacpy.core.materials`, returning grazing-angle / loss-in-dB
    arrays ready to plot. Shear is ignored (fluid–fluid only).

    Parameters
    ----------
    material : str, dict, BoundaryProperties or SedimentLayer
        Preset name (``'sand'``, ``'silt'``, …), a dict carrying
        ``sound_speed`` (m/s), ``density`` (g/cm³), ``attenuation``
        (dB/λ_p), or a seabed carrier holding those three attributes — e.g.
        ``env.bottom.halfspace_at(range=0.0)``. A carrier's
        ``acoustic_type`` decides what those attributes mean: a
        ``'half-space'`` reads them; a ``'rigid'`` or ``'vacuum'`` boundary
        reflects totally (``|R| = 1``, 0 dB at every angle) and its three
        numbers are construction placeholders, so they are not read; a
        ``'file'`` or ``'precalc'`` boundary carries its loss in a
        reflection table and is refused.
    grazing_angles_deg : array_like, optional
        Grazing-angle grid in degrees (``0`` = parallel to interface,
        ``90`` = normal incidence). Default ``np.linspace(0, 90, 181)``.
    water_sound_speed, water_density : float
        Water-column reference values (m/s, g/cm³). ``water_density``
        defaults to :data:`~uacpy.core.constants.DEFAULT_WATER_DENSITY_G_CM3`
        (1.027 g/cm³), the value every deck writes for the water column, so
        the curve matches what the propagation models see; pass ``1.0`` to
        reproduce a textbook benchmark that takes ρ_w = 1 (on 'sand' the
        two differ by at most 0.28 dB, at normal incidence).

    Returns
    -------
    grazing_angles_deg : ndarray
    loss_dB : ndarray
        Bottom loss ``-20·log10|R|`` at each angle.
    """
    if grazing_angles_deg is None:
        grazing_angles_deg = np.linspace(0.0, 90.0, 181)
    grazing_angles_deg = np.asarray(grazing_angles_deg, dtype=float)
    if isinstance(material, str):
        from uacpy.core.materials import get_material
        m = get_material(material)
    elif isinstance(material, Mapping):
        m = dict(material)
    elif all(hasattr(material, k)
             for k in ('sound_speed', 'density', 'attenuation')):
        acoustic_type = str(getattr(material, 'acoustic_type',
                                    'half-space')).lower()
        if acoustic_type in ('file', 'precalc'):
            raise ConfigurationError(
                f"bottom_loss_curve: a {acoustic_type!r} seabed carries its "
                f"loss in a reflection-coefficient table; its sound_speed, "
                f"density and attenuation are placeholders, not a half-space.",
                remediation="Read the table itself (uacpy.io."
                            "read_reflection_coefficient), or pass a "
                            "half-space carrier or a preset name.")
        if acoustic_type in ('rigid', 'vacuum'):
            # A rigid (R = +1) and a pressure-release (R = -1) boundary
            # reflect totally at every angle.
            return grazing_angles_deg, np.zeros(grazing_angles_deg.shape)
        m = {k: getattr(material, k)
             for k in ('sound_speed', 'density', 'attenuation')}
    else:
        raise ConfigurationError(
            f"bottom_loss_curve: material must be a preset name, a dict or a "
            f"seabed carrier (BoundaryProperties / SedimentLayer) with "
            f"sound_speed, density and attenuation; got "
            f"{type(material).__name__}.")
    R = reflection_coeff(
        grazing_angles_deg,
        sound_speed=float(m['sound_speed']),
        density=float(m['density']),
        attenuation=float(m['attenuation']),
        water_sound_speed=float(water_sound_speed),
        water_density=float(water_density),
    )
    loss_dB = -20.0 * np.log10(np.abs(R) + 1e-300)
    return grazing_angles_deg, np.asarray(loss_dB, dtype=float)


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
