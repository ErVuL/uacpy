"""What a boundary does to a plane wave.

:func:`reflection_coeff` is the Rayleigh coefficient for a fluid half-space —
complex, so it carries the phase — and :func:`bottom_loss_curve` sweeps it
over grazing angle for a named sediment. :func:`pekeris_root` picks the branch
of the complex square root that keeps a vertical wavenumber physical, which is
what a mode below cutoff needs to decay into the half-space instead of growing.

The sediment sits against a water column, and the two entry points spell it
differently: :func:`reflection_coeff` takes ``c`` / ``rho`` and falls back to
the seawater equations at *their* defaults (:mod:`uacpy.core.acoustics.seawater`,
27 °C surface water) when they are left out, warning once that it did, while
:func:`bottom_loss_curve` pins 1500 m/s and the package water density
(``DEFAULT_WATER_DENSITY_G_CM3``, 1.027 g/cm³ — the value every deck writes)
through ``water_speed`` / ``water_density``, which is where the two routes
differ by roughly 4 dB near the critical angle.

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

from uacpy.core.constants import DEFAULT_SOUND_SPEED, DEFAULT_WATER_DENSITY_G_CM3
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.acoustics.seawater import density, soundspeed

__all__ = [
    'reflection_coeff',
    'bottom_loss_curve',
    'pekeris_root',
]


# Whether the water-column fallback in ``reflection_coeff`` has already been
# announced in this process, so a caller sweeping angle by angle hears it once
# rather than once per call. Same one-shot shape as the endian notice in
# ``io/_fortran_helpers.py``; there is no key because there is nothing to key
# on — ``soundspeed()`` and ``density()`` at their own defaults are one fixed
# water column.
_DEFAULT_WATER_COLUMN_WARN_EMITTED = False


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
        Angle of incidence in radians, measured **from the interface normal**
        (0 = normal incidence, π/2 = grazing). For a grazing-angle grid pass
        ``pi/2 - grazing``.
    rho1 : float
        Density of second medium (e.g., sediment) in kg/m³
    c1 : float
        Sound speed in second medium in m/s
    alpha : float, optional
        Loss tangent of the second medium, dimensionless: B&L write a lossy
        medium as ``n = n0·(1 + i·alpha)`` with ``alpha > 0``. Numerically it is
        nepers per radian of propagation — convert from dB/wavelength with
        ``alpha = alpha_lambda · ln(10) / (40·pi)``, as
        :func:`bottom_loss_curve` does. Default 0 (lossless).
    rho : float, optional
        Density of water in kg/m³. Omitted, it falls back to :func:`density`
        at *its* argument defaults (27 °C, S = 35), 1022.72 kg/m³ — see
        Notes, which is not what :func:`bottom_loss_curve` uses.
    c : float, optional
        Sound speed in water in m/s. Omitted, it falls back to
        :func:`soundspeed` at *its* argument defaults, i.e. Mackenzie at
        27 °C, S = 35, 10 m — a tropical-surface operating point — giving
        1539.087 m/s. See Notes: that is not what :func:`bottom_loss_curve`
        uses, and omitting ``c`` raises a one-shot :class:`UserWarning`
        naming the value taken.

    Returns
    -------
    float, ndarray, or complex
        Reflection coefficient as a linear multiplier

    Notes
    -----
    **The two entry points onto this formula do not share a water column.**
    Called directly with ``c`` and ``rho`` omitted, this function evaluates
    Mackenzie and EOS-80 at their own argument defaults — a tropical surface
    point, 27 °C / S = 35 / 10 m — giving 1539.087 m/s and 1022.72 kg/m³.
    :func:`bottom_loss_curve` below deliberately pins the package reference
    values instead: ``water_speed=DEFAULT_SOUND_SPEED`` (1500.0 m/s) and
    ``water_density=DEFAULT_WATER_DENSITY_G_CM3`` (1.027 g/cm³, 1027 kg/m³ —
    what every deck writes for the water column).

    So the same seabed reflects differently through the two. On a 1700 m/s,
    1800 kg/m³ bottom the critical grazing angle is 28.072° against the
    wrapper's water and 25.130° against the fallback's faster water, and the
    reflection loss between them differs by roughly 4 dB at its worst, near
    that angle (the exact peak depends on how finely the angle grid samples
    the critical region, so treat the magnitude, not the decimals, as the
    result). Nearly all of that is the sound speed; the two water densities
    (1022.72 against 1027 kg/m³) are worth under 0.05 dB, peaking at normal
    incidence rather than at the critical angle. The density difference is a
    units convention, not a disagreement about seawater: this function takes
    SI kg/m³ and falls back to a real seawater value, while
    :func:`bottom_loss_curve` takes g/cm³, the unit every deck writes. Pass
    ``water_density=1.0`` to the wrapper to reproduce a textbook curve that
    takes ρ_w = 1 (COA Table 1.3 quotes sediment densities as ratios to it).

    Pass ``c`` (and ``rho``) explicitly whenever the water column matters.

    Examples
    --------
    >>> R = reflection_coeff(np.pi/4, 1200, 1600)
    >>> print(f"Reflection coefficient: {R:.4f}")
    Reflection coefficient: 0.1198

    >>> R_dB = 20 * np.log10(abs(R))
    >>> print(f"Reflection loss: {R_dB:.2f} dB")
    Reflection loss: -18.43 dB

    References
    ----------
    Brekhovskikh, L. M. & Lysanov, Y. P. (2003). Fundamentals of Ocean Acoustics.
    Eq. (3.1.12) / (5.5.1): ``V = (m cos θ − √(n² − sin²θ)) / (m cos θ +
    √(n² − sin²θ))`` with ``m = ρ1/ρ``, ``n = c/c1``; §3.1 gives the lossy
    convention ``n = n0(1 + iα), α > 0``.
    """
    angle_arr = np.asarray(angle, dtype=float)
    # The convention here is the incidence angle from the NORMAL in radians;
    # the carriers speak grazing degrees, and a grazing angle in degrees is
    # larger than pi/2 for anything but the steepest rays — refused rather
    # than folded into a coefficient of exactly 1.
    if np.any(angle_arr < 0.0) or np.any(angle_arr > np.pi / 2.0 + 1e-9):
        raise ConfigurationError(
            f"reflection_coeff: angle is the incidence angle from the normal, "
            f"in radians within [0, pi/2]; got {angle!r}. For a grazing angle "
            f"in degrees pass np.pi/2 - np.deg2rad(grazing_deg), or use "
            f"bottom_loss_curve, which takes grazing degrees and g/cm^3.")
    global _DEFAULT_WATER_COLUMN_WARN_EMITTED

    if rho is None:
        rho = density()
    if c is None:
        c = soundspeed()
        if not _DEFAULT_WATER_COLUMN_WARN_EMITTED:
            # stacklevel=2 blames whoever called ``reflection_coeff``, which is
            # right because this can only fire on a direct call: the one
            # in-package caller, ``bottom_loss_curve``, always passes ``c``.
            # A future in-package caller that omitted it would add a frame and
            # make this name a uacpy line instead — that site needs
            # ``skip_file_prefixes=USER_FRAME_SKIP``, not a larger count.
            _warnings.warn(
                f"reflection_coeff: no water sound speed given, so c falls "
                f"back to soundspeed() = {c:.3f} m/s — Mackenzie at its own "
                f"argument defaults, 27 °C / S = 35 / 10 m, a "
                f"tropical-surface operating point — with rho = "
                f"{rho:.2f} kg/m³. bottom_loss_curve pins 1500.0 m/s and "
                f"1000 kg/m³ instead, so the same seabed differs by roughly "
                f"4 dB near the critical angle through the two entry points. "
                f"Pass c= (and rho=) to set the water column explicitly. "
                f"Warned once per process.",
                UserWarning, stacklevel=2,
            )
            _DEFAULT_WATER_COLUMN_WARN_EMITTED = True

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
    water_speed: float = DEFAULT_SOUND_SPEED,
    water_density: float = DEFAULT_WATER_DENSITY_G_CM3,
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
        Water-column reference values (m/s, g/cm³). ``water_density``
        defaults to :data:`~uacpy.core.constants.DEFAULT_WATER_DENSITY_G_CM3`
        (1.027 g/cm³), the value every deck writes for the water column, so
        the curve matches what the propagation models see; pass ``1.0`` to
        reproduce a textbook benchmark that takes ρ_w = 1 (on 'sand' the
        two differ by at most 0.29 dB, at normal incidence).

    Returns
    -------
    grazing_angles_deg : ndarray
    loss_dB : ndarray
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
    # The preset carries dB/wavelength; ``reflection_coeff`` wants the loss
    # tangent (the imaginary part it adds to the index of refraction, i.e.
    # nepers per radian). ln(10)/(40π) = (1 / 8.6859) / 2π converts between
    # them — the factor Acoustics-Toolbox uses for its 'L' unit
    # (Bellhop/ReadEnvironmentBell.f90:527).
    alpha = float(m['attenuation']) * np.log(10.0) / (40.0 * np.pi)
    R = reflection_coeff(
        angle=angle_from_normal,
        rho1=float(m['density']) * 1000.0,
        c1=float(m['sound_speed']),
        alpha=alpha,
        rho=float(water_density) * 1000.0,
        c=float(water_speed),
    )
    loss_dB = -20.0 * np.log10(np.abs(R) + 1e-300)
    return np.asarray(grazing_angles_deg, dtype=float), np.asarray(loss_dB, dtype=float)


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
