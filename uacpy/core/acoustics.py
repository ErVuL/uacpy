"""
Underwater acoustics utilities for UACPY

This module provides various underwater acoustics functions including:
- Seawater sound speed (Mackenzie, UNESCO, Del Grosso) and density
- Plane-wave bottom reflection and bottom loss
- Bubble acoustics (resonance, bubbly-water speed, surface bubble loss)
- Acoustic pressure, SPL and power → dB utilities
- The Pekeris branch of the complex square root

Note
----
Physics-only helpers with no uacpy dependencies beyond
:mod:`uacpy.core.constants`, so any layer may import them: the data layer
uses the seawater equations (:mod:`uacpy.data.sound_speed`,
:mod:`uacpy.data.argo`), :mod:`uacpy.core.ssp` builds profiles from
Mackenzie, :mod:`uacpy.io.modes_reader` takes :func:`pekeris_root`, and the
spectral estimators and their plotters share :func:`power_to_db`. They are
also public API for notebooks and examples (e.g.
example_12_attenuation_models.py).

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

from uacpy.core.constants import (
    DEFAULT_SOUND_SPEED, PRESSURE_FLOOR, REFERENCE_PRESSURE_WATER,
)
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.exceptions import ConfigurationError

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
    temperature: Union[float, np.ndarray] = 27,
    salinity: Union[float, np.ndarray] = 35,
    depth: Union[float, np.ndarray] = 10,
) -> Union[float, np.ndarray]:
    """
    Calculate speed of sound in water using Mackenzie (1981) formula.

    Parameters
    ----------
    temperature : float or ndarray, optional
        Water temperature in degrees Celsius (default: 27)
    salinity : float or ndarray, optional
        Salinity in parts per thousand (ppt) (default: 35)
    depth : float or ndarray, optional
        Depth in meters (default: 10)

    Returns
    -------
    float or ndarray
        Sound speed in m/s. The formula is evaluated element-wise, so an
        array argument gives an array of the broadcast shape — which is how
        :meth:`uacpy.SoundSpeedProfile.from_ts` calls it.

    Examples
    --------
    >>> c = soundspeed()
    >>> print(f"Sound speed: {c:.1f} m/s")
    Sound speed: 1539.1 m/s

    >>> c = soundspeed(temperature=25, depth=20)
    >>> print(f"Sound speed: {c:.1f} m/s")
    Sound speed: 1534.6 m/s

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
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    if np.any(np.asarray(salinity) < 25) or np.any(np.asarray(salinity) > 40):
        _warnings.warn(
            "Mackenzie soundspeed: salinity outside validated range "
            "[25, 40] PSU; treating as extrapolation.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    if np.any(np.asarray(depth) < 0) or np.any(np.asarray(depth) > 8000):
        _warnings.warn(
            "Mackenzie soundspeed: depth outside validated range "
            "[0, 8000] m; treating as extrapolation.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
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


# Chen & Millero and Del Grosso both state the cold end of their fit at 0 °C,
# but seawater is liquid well below that — its freezing point falls with
# pressure, reaching about -3 °C in the deepest trenches — and polar deep water
# lives there. Both extrapolations are smooth and monotone across the gap
# (at S = 34.7, surface: UNESCO 1434.45 m/s and Del Grosso 1434.51 m/s at
# -3 °C, agreeing to 0.05 m/s, which is Del Grosso's own standard deviation),
# so each warning starts below the coldest water that exists rather than at the
# fit boundary, where it would fire on every polar profile.
_COLDEST_SEAWATER_C = -3.0


def soundspeed_unesco(temperature=15.0, salinity=35.0, pressure=0.0):
    """Speed of sound in seawater — UNESCO (Chen & Millero 1977 / UNESCO 1983).

    The international standard algorithm. ``pressure`` is in **decibars** (the
    form the equation is defined in; ≈ 1 dbar per metre of depth). Temperature is
    ITS-90 (converted to the IPTS-68 scale the polynomial expects internally).
    Valid for ``T ∈ [0, 40] °C``, ``S ∈ [0, 40] PSU`` and ``P ∈ [0, 1000]``
    **bar** — which in this argument's decibars is ``[0, 10000] dbar``, so
    roughly the full ocean depth. Values outside these ranges trigger a
    :class:`UserWarning` and the output should be treated as extrapolation,
    matching :func:`soundspeed` (Mackenzie). The one relaxation is the cold
    end: the warning starts at −3 °C rather than 0 °C, for the reason given
    at :data:`_COLDEST_SEAWATER_C`.

    Parameters
    ----------
    temperature : float
        Temperature [°C, ITS-90].
    salinity : float
        Practical salinity [PSU, PSS-78].
    pressure : float
        Pressure [dbar] — decibars, *not* bar. The equation is stated in bar
        and is converted internally.

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
    if np.any(t < _COLDEST_SEAWATER_C) or np.any(t > 40):
        _warnings.warn(
            f"UNESCO soundspeed: temperature outside validated range "
            f"[{_COLDEST_SEAWATER_C:g}, 40] °C; treating as "
            f"extrapolation.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    # Split at zero because the two sides are not the same kind of miss: the
    # S^1.5 term of Eqn 36 has no real value for S < 0, so that side returns
    # NaN rather than an extrapolated number, and saying "extrapolation" there
    # would describe a result the function never produces.
    if np.any(s < 0):
        _warnings.warn(
            "UNESCO soundspeed: salinity below the validated range "
            "[0, 40] PSU is undefined, not extrapolated — the S^1.5 term has "
            "no real value there, so the result is NaN.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    elif np.any(s > 40):
        _warnings.warn(
            "UNESCO soundspeed: salinity outside validated range "
            "[0, 40] PSU; treating as extrapolation.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    # Bounds are tested on the converted bar so they read as the equation
    # states them; the argument itself is decibars, ten times the number.
    if np.any(p < 0) or np.any(p > 1000):
        _warnings.warn(
            "UNESCO soundspeed: pressure outside validated range "
            "[0, 1000] bar = [0, 10000] dbar (this argument is in DECIBARS); "
            "treating as extrapolation.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )

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

    # The S < 0 branch above already says the result is NaN; numpy's own
    # "invalid value encountered in power" would only repeat it less clearly.
    with np.errstate(invalid='ignore'):
        c = cw + a * s + b * s ** 1.5 + d * s ** 2
    return float(c) if np.ndim(c) == 0 else c


def soundspeed_delgrosso(temperature=15.0, salinity=35.0, pressure=0.0):
    """Speed of sound in seawater — Del Grosso (1974) "NRL II" equation.

    An alternative to UNESCO, often preferred at high pressure / in deep water.
    ``pressure`` is accepted in **decibars** and converted to the kg/cm² the
    original equation uses (``1 kg/cm² = 9.80665 dbar``). Temperature in °C,
    salinity in PSU. Standard deviation 0.05 m/s.

    Valid over ``T ∈ [0, 35] °C``, ``S ∈ [29, 43] ppt`` and ``P ∈ [0, 1000]``
    **kg/cm² gauge** — which in this argument's decibars is ``[0, 9807] dbar``,
    about the full ocean depth. Outside them the result is an extrapolation and
    a :class:`UserWarning` says so, the same contract :func:`soundspeed` and
    :func:`soundspeed_unesco` keep, with the same relaxed cold end: the warning
    starts at −3 °C, not the fit's 0 °C, so polar deep water does not trip it
    (see :data:`_COLDEST_SEAWATER_C`). The salinity floor is a real floor, not
    a formality: the fit was built on "realistic triads" and 29 ppt is the
    lowest it covers, so brackish and estuarine water is outside this equation
    entirely — use :func:`soundspeed_unesco`, whose fit reaches S = 0.

    References
    ----------
    Del Grosso, V. A. (1974). "New equation for the speed of sound in natural
    waters (with comparisons to other equations)." JASA 56(4), 1084-1091 —
    "The temperatures considered range from 0° to 35 °C … salinity ranges from
    29 to 43 ppt … Pressure ranges from 0 to 1000 kg/cm² gauge". The same
    domain is tabulated in Etter, *Underwater Acoustic Modeling and
    Simulation*, Table 2.1.
    """
    t = np.asarray(temperature, dtype=float)
    s = np.asarray(salinity, dtype=float)
    p = np.asarray(pressure, dtype=float) / 9.80665     # dbar -> kg/cm²
    if np.any(t < _COLDEST_SEAWATER_C) or np.any(t > 35):
        _warnings.warn(
            f"Del Grosso soundspeed: temperature outside validated range "
            f"[{_COLDEST_SEAWATER_C:g}, 35] °C; treating as extrapolation.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    if np.any(s < 29) or np.any(s > 43):
        _warnings.warn(
            "Del Grosso soundspeed: salinity outside validated range "
            "[29, 43] ppt; treating as extrapolation. This equation was fitted "
            "to open-ocean salinities only — for fresher water use "
            "soundspeed_unesco, which is validated to S = 0.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    # Bounds are tested on the converted kg/cm² so they read as the paper
    # states them; the argument itself is decibars, 9.80665 times the number.
    if np.any(p < 0) or np.any(p > 1000):
        _warnings.warn(
            "Del Grosso soundspeed: pressure outside validated range "
            "[0, 1000] kg/cm² = [0, 9807] dbar (this argument is in DECIBARS); "
            "treating as extrapolation.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )

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


def density(
    temperature: Union[float, np.ndarray] = 27,
    salinity: Union[float, np.ndarray] = 35,
) -> Union[float, np.ndarray]:
    """
    Calculate density of sea water near the surface.

    Uses Fofonoff (1985 - IES 80) formula.

    Parameters
    ----------
    temperature : float or ndarray, optional
        Water temperature in degrees Celsius (default: 27)
    salinity : float or ndarray, optional
        Salinity in parts per thousand (ppt) (default: 35)

    Returns
    -------
    float or ndarray
        Density in kg/m³. Evaluated element-wise, so an array argument gives
        an array of the broadcast shape.

    Examples
    --------
    >>> rho = density()
    >>> print(f"Density: {rho:.1f} kg/m³")
    Density: 1022.7 kg/m³

    References
    ----------
    Fofonoff, N. P. (1985). "Physical properties of seawater: A new salinity
    scale and equation of state for seawater". Journal of Geophysical Research,
    90(C2), 3332-3342.
    """
    # EOS-80 one-atmosphere equation of state, in Horner form:
    # rho(S,t,0) = rho_w(t) + A(t)·S + B(t)·S^1.5 + C·S^2. The locals are offset
    # by one from that notation — ``A`` accumulates rho_w, then ``B``/``C``/``D``
    # are the S / S^1.5 / S^2 coefficients. Pressure is not a parameter, so this
    # is surface density only; deeper water needs the full EOS-80 with the
    # secant bulk modulus.
    t = temperature
    A = 1.001685e-04 + t * (-1.120083e-06 + t * 6.536332e-09)
    A = 999.842594 + t * (6.793952e-02 + t * (-9.095290e-03 + t * A))
    B = 7.6438e-05 + t * (-8.2467e-07 + t * 5.3875e-09)
    B = 0.824493 + t * (-4.0899e-03 + t * B)
    C = -5.72466e-03 + t * (1.0227e-04 - t * 1.6546e-06)
    D = 4.8314e-04
    return A + salinity * (B + C * np.sqrt(salinity) + D * salinity)


def doppler(
    speed: Union[float, np.ndarray],
    frequency: Union[float, np.ndarray],
    c: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """
    Calculate Doppler-shifted frequency.

    The approximation is valid when speed << c (typical for underwater vehicles).

    Parameters
    ----------
    speed : float or ndarray
        Relative speed between transmitter and receiver in m/s
        (positive = approaching, negative = receding)
    frequency : float or ndarray
        Transmission frequency in Hz
    c : float, optional
        Sound speed in m/s (default: calculated using soundspeed())

    Returns
    -------
    float or ndarray
        Doppler shifted frequency in Hz as perceived by the receiver.
        Evaluated element-wise, so an array argument gives an array of the
        broadcast shape.

    Examples
    --------
    >>> f_shifted = doppler(2, 50000)  # 2 m/s approach
    >>> print(f"Shifted frequency: {f_shifted:.2f} Hz")
    Shifted frequency: 50064.97 Hz

    >>> f_shifted = doppler(-1, 50000)  # 1 m/s receding
    >>> print(f"Shifted frequency: {f_shifted:.2f} Hz")
    Shifted frequency: 49967.51 Hz
    """
    if c is None:
        c = soundspeed()
    return (1 + speed / c) * frequency


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
    :func:`bottom_loss_curve` below deliberately pins the round reference
    values instead: ``water_speed=DEFAULT_SOUND_SPEED`` (1500.0 m/s) and
    ``water_density=1.0`` g/cm³ (1000 kg/m³).

    So the same seabed reflects differently through the two. On a 1700 m/s,
    1800 kg/m³ bottom the critical grazing angle is 28.072° against the
    wrapper's water and 25.130° against the fallback's faster water, and the
    reflection loss between them differs by roughly 4 dB at its worst, near
    that angle (the exact peak depends on how finely the angle grid samples
    the critical region, so treat the magnitude, not the decimals, as the
    result). Nearly all of that is the sound speed; the density difference is
    worth a few tenths of a dB, peaking at normal incidence rather than at the
    critical angle. The density difference is a units convention, not a
    disagreement about seawater: this function takes SI kg/m³ and falls back
    to a real seawater value, while :func:`bottom_loss_curve` follows the
    Acoustics Toolbox in quoting sediment density relative to water, where
    ``rho = 1.0`` g/cm³ makes ``m = rho1/rho`` the relative density.

    Pass ``c`` (and ``rho``) explicitly whenever the water column matters.

    Examples
    --------
    >>> R = reflection_coeff(np.pi/4, 1200, 1600)
    >>> print(f"Reflection coefficient: {R:.4f}")
    Reflection coefficient: 0.1198

    >>> R_db = 20 * np.log10(abs(R))
    >>> print(f"Reflection loss: {R_db:.2f} dB")
    Reflection loss: -18.43 dB

    References
    ----------
    Brekhovskikh, L. M. & Lysanov, Y. P. (2003). Fundamentals of Ocean Acoustics.
    Eq. (3.1.12) / (5.5.1): ``V = (m cos θ − √(n² − sin²θ)) / (m cos θ +
    √(n² − sin²θ))`` with ``m = ρ1/ρ``, ``n = c/c1``; §3.1 gives the lossy
    convention ``n = n0(1 + iα), α > 0``.
    """
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
    Resonance frequency: 32465.56 Hz

    >>> radii = np.logspace(-5, -3, 50)  # 10 to 1000 microns
    >>> f_res = bubble_resonance(radii)

    References
    ----------
    Medwin, H. & Clay, C. S. (1998). Fundamentals of Acoustical Oceanography,
    eq. (8.2.13): ``f_h = (1/2πa)·√(3γ p_A/ρ_A)`` with ``p_A = p_A0 + ρ_A g z``.
    Valid while ``ka ≲ 1``.
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
        Windspeed in m/s (measured 10 m above sea surface). Non-negative.
    frequency : float or array_like
        Frequency in Hz. Positive.
    angle : float or array_like
        Incidence angle in radians, measured **from the surface normal** —
        0 is a ray striking the surface head-on and ``pi/2`` is grazing. The
        handbook writes the model against the grazing angle ``beta``, which
        is why the body forms ``beta = pi/2 - angle``. ``|angle| <= pi/2``:
        past that, ``sin(beta)`` turns negative and the formula returns a
        multiplier greater than 1, i.e. a surface that amplifies.

    Returns
    -------
    float or ndarray
        Surface reflection as a linear amplitude multiplier in ``[0, 1]``
        (1.0 = no loss). To express the loss as a **positive** dB number,
        consistent with :func:`bottom_loss_curve`, negate the log:
        ``loss_db = -20 * np.log10(multiplier)``. Exact grazing
        (``|angle| = pi/2``) is the ``1/sin(beta) -> inf`` limit and returns
        0.0.

    Raises
    ------
    ConfigurationError
        For a negative windspeed, a non-positive frequency or an angle
        outside ``[-pi/2, pi/2]``. Each has a formula branch that answers
        anyway rather than failing: a negative windspeed takes the ``U < 6``
        branch and reports ~1.0, i.e. no loss; a negative frequency raises a
        negative base to 0.85 and returns a *complex* multiplier; and a zero
        one reports no loss.

    Examples
    --------
    >>> mult = bubble_surface_loss(3, 10000, 0)
    >>> loss_db = -20 * np.log10(mult)   # positive dB loss
    >>> print(f"Surface loss: {loss_db:.2f} dB")
    Surface loss: 0.00 dB

    Notes
    -----
    Surface bubble loss ``SBL`` (APL-UW TR 9407 eqs. 28a/28b, p. II-21):

    ``SBL = 1.26e-3/sin(theta) * U**1.57 * f**0.85``   for ``U >= 6 m/s``,
    ``SBL = SBL(U=6) * exp(1.2*(U-6))``                for ``U < 6 m/s``,

    with ``U`` the wind speed 10 m above the surface, ``f`` in kHz and
    ``theta`` the nominal grazing angle of the surface-bounce path. The 6 m/s
    break is the breaking-wave (Beaufort) threshold below which bubbles are
    not produced. The handbook fits 20-40 kHz data to within +/-3 dB and notes
    a nominal 30 dB ceiling on ``SBL`` from scattering off the underside of the
    bubble layer; that ceiling is not imposed here.

    References
    ----------
    APL-UW (1994). "APL-UW High-Frequency Ocean Environmental Acoustic Models
    Handbook". Technical Report APL-UW TR 9407, sec. II.C.4.
    """
    if not np.isfinite(windspeed) or windspeed < 0:
        raise ConfigurationError(
            f"bubble_surface_loss: windspeed must be a non-negative finite "
            f"speed in m/s; got {windspeed!r}. A negative one takes the "
            f"U < 6 m/s branch and comes back as a multiplier near 1.0 — "
            f"reported as almost no surface loss.")
    f_hz = np.asarray(frequency, dtype=float)
    if not np.all(np.isfinite(f_hz)) or np.any(f_hz <= 0):
        raise ConfigurationError(
            f"bubble_surface_loss: frequency must be positive and finite "
            f"(Hz); got {frequency!r}. f**0.85 takes a negative frequency to "
            f"a complex multiplier and a zero one to 1.0, i.e. no loss. The "
            f"handbook fit is to 20-40 kHz data.")
    theta = np.asarray(angle, dtype=float)
    if not np.all(np.isfinite(theta)) or np.any(np.abs(theta) > np.pi / 2):
        raise ConfigurationError(
            f"bubble_surface_loss: angle is the incidence angle from the "
            f"surface normal in radians and must lie in [-pi/2, pi/2]; got "
            f"{angle!r}. Past pi/2 the grazing angle beta = pi/2 - angle goes "
            f"negative, sin(beta) with it, and the multiplier comes back "
            f"above 1 — a surface that amplifies the ray it reflects.")

    beta = np.pi / 2 - theta
    f = f_hz / 1000.0  # Convert to kHz

    # Exact grazing is sin(beta) = 0, where SBL diverges and the multiplier
    # goes to its 0.0 limit; the division states that rather than tripping a
    # RuntimeWarning on the way there.
    with np.errstate(divide='ignore'):
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
    Sound speed in bubbly water: 1402.13 m/s

    Notes
    -----
    Wood's equation as given by Medwin & Clay eq. (8.3.39): the mixture takes the
    volume-averaged density ``rho_A = U·rho_b + (1-U)·rho_w`` and the
    volume-averaged compressibility ``1/E_A = U/E_b + (1-U)/E_w``, with
    ``c = sqrt(E_A/rho_A)``. Valid for every void fraction, and independent of
    the bubble size distribution.

    References
    ----------
    Wood, A. B. (1964). A Textbook of Sound.
    Buckingham, M. J. (1997). "Theory of acoustic attenuation, dispersion,
    and pulse propagation in unconsolidated granular materials". JASA, 102(5).
    """
    if c is None:
        c = soundspeed()

    # Splitting ``relative_density`` as m = sqrt(rho_w/rho_gas) puts the two
    # averages on a common footing: ``numerator`` is (rho_w c_w²)/(rho_A c_A²)/m
    # and ``denominator`` is m·rho_A/rho_w, so their product is exactly
    # (c_w/c_A)² — the density ratio and the factor m both cancel, leaving one
    # square root instead of two.
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
    With ``sensitivity=0`` and ``gain=0`` both scale factors are unity, so the
    voltage passes through untouched:

    >>> x_volt = np.array([0.0, 0.5, -0.5])
    >>> pressure(x_volt, sensitivity=0, gain=0)
    array([ 0. ,  0.5, -0.5])

    A bit-depth input is divided by the full-scale count first, so half of
    full scale on a signed 16-bit sample (2**15) against a 1 V reference lands
    on the same 0.5 V:

    >>> x_bits = np.array([0, 16384, -16384])
    >>> pressure(x_bits, sensitivity=0, gain=0, volt_params=(16, 1.0))
    array([ 0. ,  0.5, -0.5])
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
    A 100 µPa-rms white signal sits at ``20*log10(100) = 40`` dB re 1 µPa; the
    seed makes the sampling scatter around that reproducible.

    >>> rng = np.random.default_rng(0)
    >>> pressure_signal = rng.standard_normal(1000) * 100
    >>> spl_db = spl(pressure_signal, ref=1)
    >>> print(f"SPL: {spl_db:.2f} dB re 1 µPa")
    SPL: 39.81 dB re 1 µPa

    The rms pressure is floored at ``sqrt(PRESSURE_FLOOR)`` before the log,
    so a silent (all-zero) signal returns a finite
    ``20*log10(sqrt(PRESSURE_FLOOR)/ref)`` — -300 dB re 1 µPa at the default
    ``ref=1`` — instead of ``-inf``. The floor is read in this function's µPa
    working unit, while :func:`power_to_db` applies the same named constant
    to its ``power`` argument in the units of ``ref**2`` (Pa² at its
    default), where a silent signal floors at -180 dB re 1 µPa; the two
    silent-signal levels coincide only when both are called with ``ref=1``.
    """
    rmsx = np.sqrt(np.mean(np.abs(x) ** 2))
    return 20 * np.log10(np.maximum(rmsx, np.sqrt(PRESSURE_FLOOR)) / ref)


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
        :data:`PRESSURE_FLOOR`), guarding ``log10(0)``. Read in ``power``'s
        own units (``ref**2``), so the level a fully-silent input floors at
        depends on ``ref``: ``10*log10(1e-30 / 1e-12) = -180`` dB re 1 µPa
        at the default ``ref``, against the -300 dB re 1 µPa :func:`spl`
        floors a silent signal at under its µPa default.

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
