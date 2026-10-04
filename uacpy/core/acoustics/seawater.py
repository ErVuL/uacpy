"""Seawater sound speed and density, and the Doppler shift they set.

Four sound-speed equations over temperature, salinity and depth — Mackenzie
1981 (:func:`sound_speed_mackenzie`), UNESCO/Chen-Millero
(:func:`sound_speed_unesco`), Del Grosso 1974 (:func:`sound_speed_delgrosso`)
and TEOS-10 (:func:`sound_speed_teos10`) — plus :func:`density` (Fofonoff
IES-80) and :func:`doppler`, which reads a frequency shift off one of them.
:func:`depth_to_pressure_dbar` / :func:`pressure_dbar_to_depth` convert the
depth axis, and :func:`insitu_from_potential` turns an ocean model's potential
temperature into the in-situ temperature the equations take (UNESCO 44).

Each sound-speed equation warns outside the range its own authors fit, so a
profile that strays past a published boundary says so rather than returning a
silently extrapolated number.

Two different defaults live in the package and neither is "the" default.
A helper handed no sound speed takes the nominal ``DEFAULT_SOUND_SPEED``,
1500 m/s (:mod:`~uacpy.core.acoustics.boundaries` and the bubble models).
A sound speed computed from temperature and salinity — the data routes and
:meth:`uacpy.SoundSpeedProfile.from_temperature_salinity` — defaults to
``DEFAULT_SOUND_SPEED_FORMULA`` (TEOS-10). Anything quoting a sound speed
should say which.

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
from typing import Union, Optional

from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.exceptions import ConfigurationError, ValidityWarning
from uacpy.core._validate import water_property
from uacpy.core.constants import (
    DEFAULT_SOUND_SPEED, REFERENCE_DEPTH_M, REFERENCE_SALINITY_PSU,
    REFERENCE_TEMPERATURE_C, STANDARD_GRAVITY_M_S2)

#: dbar per kgf/cm²: 1 kgf/cm² = g·1e4 Pa, which is g decibars.
_DBAR_PER_KGF_CM2 = STANDARD_GRAVITY_M_S2

__all__ = [
    'sound_speed_mackenzie', 'sound_speed_unesco', 'sound_speed_delgrosso',
    'sound_speed_teos10', 'sound_speed_at_depth', 'depth_to_pressure_dbar',
    'pressure_dbar_to_depth', 'insitu_from_potential', 'density', 'doppler',
    'REFERENCE_LATITUDE_DEG', 'DEFAULT_SOUND_SPEED_FORMULA',
    'SOUND_SPEED_FORMULAS', 'canonical_formula',
]


def _sequence_as_array(x):
    """A list or tuple as a float64 array; a scalar or an ndarray is
    returned as it is, so its dtype and the scalar arithmetic stay the
    caller's."""
    if np.isscalar(x) or isinstance(x, np.ndarray):
        return x
    return np.asarray(x, dtype=float)


def sound_speed_mackenzie(
    temperature: Union[float, np.ndarray] = REFERENCE_TEMPERATURE_C,
    salinity: Union[float, np.ndarray] = REFERENCE_SALINITY_PSU,
    depth: Optional[Union[float, np.ndarray]] = None,
) -> Union[float, np.ndarray]:
    """
    Calculate speed of sound in water using Mackenzie (1981) formula.

    Parameters
    ----------
    temperature : float, ndarray or (N, 2) pairs, optional
        Water temperature in degrees Celsius (default: 10, the package's
        reference sea water, ``constants.REFERENCE_TEMPERATURE_C``); see
        the water-property rule below.
    salinity : float, ndarray or (N, 2) pairs, optional
        Salinity in parts per thousand (ppt) (default: 35)
    depth : float or ndarray, optional
        Depth in meters (default: ``None``, the surface, 0 m)

    Returns
    -------
    float or ndarray
        Sound speed in m/s. The formula is evaluated element-wise, so an
        array argument gives an array of the broadcast shape — which is how
        :meth:`uacpy.SoundSpeedProfile.from_temperature_salinity` calls it.

    Examples
    --------
    >>> c = sound_speed_mackenzie()
    >>> print(f"Sound speed: {c:.1f} m/s")
    Sound speed: 1489.8 m/s

    >>> c = sound_speed_mackenzie(temperature=25, depth=20)
    >>> print(f"Sound speed: {c:.1f} m/s")
    Sound speed: 1534.6 m/s

    Notes
    -----
    **Water properties.** ``temperature`` and ``salinity`` each take a
    single value, an array (broadcast with ``depth``), or ``(depth, value)``
    pairs interpolated linearly onto ``depth`` (end values held), which then
    must be given (:func:`uacpy.core._validate.water_property`).

    Mackenzie's nine-term formula is validated for
    ``temperature ∈ [-2, 30] °C``, ``salinity ∈ [25, 40] PSU``,
    ``depth ∈ [0, 8000] m``. Values outside these ranges trigger a
    ``ValidityWarning`` and the formula's output should be treated as
    extrapolation.

    References
    ----------
    Mackenzie, K. V. (1981). "Nine-term equation for sound speed in the oceans".
    The Journal of the Acoustical Society of America, 70(3), 807-812.
    """
    temperature = _sequence_as_array(water_property(
        temperature, depth, name='temperature', who='sound_speed_mackenzie'))
    salinity = _sequence_as_array(water_property(
        salinity, depth, name='salinity', who='sound_speed_mackenzie'))
    depth = _sequence_as_array(REFERENCE_DEPTH_M if depth is None else depth)
    if np.any(np.asarray(temperature) < -2) or np.any(np.asarray(temperature) > 30):
        _warnings.warn(
            "Mackenzie sound speed: temperature outside validated range "
            "[-2, 30] °C; treating as extrapolation.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    if np.any(np.asarray(salinity) < 25) or np.any(np.asarray(salinity) > 40):
        _warnings.warn(
            "Mackenzie sound speed: salinity outside validated range "
            "[25, 40] PSU; treating as extrapolation.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    if np.any(np.asarray(depth) < 0) or np.any(np.asarray(depth) > 8000):
        _warnings.warn(
            "Mackenzie sound speed: depth outside validated range "
            "[0, 8000] m; treating as extrapolation.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
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


def _water_at_pressure(temperature, salinity, depth, pressure_dbar,
                       latitude_deg, who):
    """``(T, S, P)`` as float arrays for an equation stated in pressure
    (dbar), evaluated at ``depth`` (m, converted by
    :func:`depth_to_pressure_dbar` at ``latitude_deg``, default
    ``REFERENCE_LATITUDE_DEG``) or at ``pressure_dbar``; neither is the
    surface. ``temperature`` and ``salinity`` follow the water-property
    rule (:func:`~uacpy.core._validate.water_property`) on ``depth``, whose
    pairs are ``(depth, value)``."""
    if depth is not None and pressure_dbar is not None:
        raise ConfigurationError(
            f"{who}: pass depth= (m) or pressure_dbar= (dbar), not both.")
    if latitude_deg is not None and depth is None:
        raise ConfigurationError(
            f"{who}: latitude_deg= converts depth= to pressure, and no "
            f"depth= was given.")
    t = water_property(temperature, depth, name='temperature', who=who)
    s = water_property(salinity, depth, name='salinity', who=who)
    if depth is not None:
        lat = (REFERENCE_LATITUDE_DEG if latitude_deg is None
               else float(latitude_deg))
        p = depth_to_pressure_dbar(depth, lat)
    else:
        p = 0.0 if pressure_dbar is None else pressure_dbar
    return (np.asarray(t, dtype=float), np.asarray(s, dtype=float),
            np.asarray(p, dtype=float))


# Chen & Millero and Del Grosso both state the cold end of their fit at 0 °C,
# but seawater is liquid well below that — its freezing point falls with
# pressure, reaching about -3 °C in the deepest trenches — and polar deep water
# lives there. Both extrapolations are smooth and monotone across the gap
# (at S = 34.7, surface: UNESCO 1434.45 m/s and Del Grosso 1434.51 m/s at
# -3 °C, agreeing to 0.05 m/s, which is Del Grosso's own standard deviation),
# so each warning starts below the coldest water that exists rather than at the
# fit boundary, where it would fire on every polar profile.
_COLDEST_SEAWATER_C = -3.0


def sound_speed_unesco(temperature=REFERENCE_TEMPERATURE_C,
                       salinity=REFERENCE_SALINITY_PSU, *, depth=None,
                       pressure_dbar=None, latitude_deg=None):
    """Speed of sound in seawater — UNESCO (Chen & Millero 1977 / UNESCO 1983).

    The international standard algorithm. ``pressure`` is in **decibars** (the
    form the equation is defined in; ≈ 1 dbar per metre of depth). Temperature is
    ITS-90 (converted to the IPTS-68 scale the polynomial expects internally).
    Valid for ``T ∈ [0, 40] °C``, ``S ∈ [0, 40] PSU`` and ``P ∈ [0, 1000]``
    **bar** — which in this argument's decibars is ``[0, 10000] dbar``, so
    roughly the full ocean depth. Values outside these ranges trigger a
    ``ValidityWarning`` and the output should be treated as extrapolation,
    matching :func:`sound_speed_mackenzie`. The one relaxation is the cold
    end: the warning starts at −3 °C rather than 0 °C, for the reason given
    at :data:`_COLDEST_SEAWATER_C`.

    Parameters
    ----------
    temperature : float
        Temperature [°C, ITS-90].
    salinity : float
        Practical salinity [PSU, PSS-78].
    depth : float or array_like, optional (keyword)
        Depth [m] the equation is evaluated at, converted to pressure (see
        below).
    pressure_dbar : float or array_like, optional (keyword)
        Pressure [dbar] — decibars, *not* bar — the equation is evaluated at, for pressure-native
        data; ``None`` with no ``depth`` is the surface.
    latitude_deg : float, optional (keyword)
        Latitude [deg] of the depth -> pressure conversion; default 45.

    Returns
    -------
    float
        Sound speed [m/s].

    **Where it is evaluated.** ``depth`` (m, keyword) is converted to the
    pressure the equation is stated in with :func:`depth_to_pressure_dbar`
    (Leroy & Parthiot's standard ocean) at ``latitude_deg``, default
    ``REFERENCE_LATITUDE_DEG`` (45°) — the conversion
    :meth:`~uacpy.core.ssp.SoundSpeedProfile.from_temperature_salinity` and
    :func:`sound_speed_at_depth` use. ``pressure_dbar`` (dbar, keyword)
    gives the pressure itself, for pressure-native data such as an Argo
    cast; pass one of the two, neither being the surface.

    **Water properties.** ``temperature`` and ``salinity`` each take a
    single value, an array broadcast with the coordinate, or ``(depth,
    value)`` pairs (m) interpolated linearly onto ``depth`` (end values
    held), which then must be given
    (:func:`uacpy.core._validate.water_property`).

    References
    ----------
    Chen, C.-T. & Millero, F. J. (1977). "Speed of sound in seawater at high
    pressures." JASA 62(5), 1129-1135. UNESCO (1983) Technical Papers in Marine
    Science 44, Eqns 33-37.
    """
    t, s, p = _water_at_pressure(temperature, salinity, depth,
                                 pressure_dbar, latitude_deg,
                                 'sound_speed_unesco')
    p = p / 10.0                                       # dbar -> bar
    t68 = t * 1.00024                                  # ITS-90 -> IPTS-68
    if np.any(t < _COLDEST_SEAWATER_C) or np.any(t > 40):
        _warnings.warn(
            f"UNESCO sound speed: temperature outside validated range "
            f"[{_COLDEST_SEAWATER_C:g}, 40] °C; treating as "
            f"extrapolation.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    # Split at zero because the two sides are not the same kind of miss: the
    # S^1.5 term of Eqn 36 has no real value for S < 0, so that side returns
    # NaN rather than an extrapolated number, and saying "extrapolation" there
    # would describe a result the function never produces.
    if np.any(s < 0):
        _warnings.warn(
            "UNESCO sound speed: salinity below the validated range "
            "[0, 40] PSU is undefined, not extrapolated — the S^1.5 term has "
            "no real value there, so the result is NaN.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    elif np.any(s > 40):
        _warnings.warn(
            "UNESCO sound speed: salinity outside validated range "
            "[0, 40] PSU; treating as extrapolation.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    # Bounds are tested on the converted bar so they read as the equation
    # states them; the argument itself is decibars, ten times the number.
    if np.any(p < 0) or np.any(p > 1000):
        _warnings.warn(
            "UNESCO sound speed: pressure outside validated range "
            "[0, 1000] bar = [0, 10000] dbar (this argument is in DECIBARS); "
            "treating as extrapolation.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
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


def sound_speed_delgrosso(temperature=REFERENCE_TEMPERATURE_C,
                          salinity=REFERENCE_SALINITY_PSU, *, depth=None,
                          pressure_dbar=None, latitude_deg=None):
    """Speed of sound in seawater — Del Grosso (1974) "NRL II" equation.

    An alternative to UNESCO, often preferred at high pressure / in deep water.
    ``pressure`` is accepted in **decibars** and converted to the kg/cm² the
    original equation uses (``1 kg/cm² = 9.80665 dbar``). Temperature in °C
    on ITS-90, converted internally to the IPTS-68 scale in force when the
    coefficients were fitted (Saunders' ``t68 = 1.00024·t90``, the conversion
    the TEOS-10 manual §2.1 recommends; the same one
    :func:`sound_speed_unesco` applies). Salinity in PSU. Standard deviation
    0.05 m/s.

    Valid over ``T ∈ [0, 35] °C``, ``S ∈ [29, 43] ppt`` and ``P ∈ [0, 1000]``
    **kg/cm² gauge** — which in this argument's decibars is ``[0, 9807] dbar``,
    about the full ocean depth. Outside them the result is an extrapolation and
    a ``ValidityWarning`` says so, the same contract
    :func:`sound_speed_mackenzie` and :func:`sound_speed_unesco` keep,
    with the same relaxed cold end: the warning
    starts at −3 °C, not the fit's 0 °C, so polar deep water does not trip it
    (see :data:`_COLDEST_SEAWATER_C`). The salinity floor is a real floor, not
    a formality: the fit was built on "realistic triads" and 29 ppt is the
    lowest it covers, so brackish and estuarine water is outside this equation
    entirely — use :func:`sound_speed_unesco`, whose fit reaches S = 0.

    Parameters
    ----------
    temperature : float or array_like, optional
        Temperature [°C, ITS-90]. Default 10.
    salinity : float or array_like, optional
        Salinity [PSU]. Default 35.
    depth : float or array_like, optional (keyword)
        Depth [m] the equation is evaluated at, converted to pressure (see
        below).
    pressure_dbar : float or array_like, optional (keyword)
        Pressure [dbar], converted to the equation's kg/cm²,, for pressure-native
        data; ``None`` with no ``depth`` is the surface.
    latitude_deg : float, optional (keyword)
        Latitude [deg] of the depth -> pressure conversion; default 45.

    **Where it is evaluated.** ``depth`` (m, keyword) is converted to the
    pressure the equation is stated in with :func:`depth_to_pressure_dbar`
    (Leroy & Parthiot's standard ocean) at ``latitude_deg``, default
    ``REFERENCE_LATITUDE_DEG`` (45°) — the conversion
    :meth:`~uacpy.core.ssp.SoundSpeedProfile.from_temperature_salinity` and
    :func:`sound_speed_at_depth` use. ``pressure_dbar`` (dbar, keyword)
    gives the pressure itself, for pressure-native data such as an Argo
    cast; pass one of the two, neither being the surface.

    **Water properties.** ``temperature`` and ``salinity`` each take a
    single value, an array broadcast with the coordinate, or ``(depth,
    value)`` pairs (m) interpolated linearly onto ``depth`` (end values
    held), which then must be given
    (:func:`uacpy.core._validate.water_property`).

    References
    ----------
    Del Grosso, V. A. (1974). "New equation for the speed of sound in natural
    waters (with comparisons to other equations)." JASA 56(4), 1084-1091 —
    "The temperatures considered range from 0° to 35 °C … salinity ranges from
    29 to 43 ppt … Pressure ranges from 0 to 1000 kg/cm² gauge". The same
    domain is tabulated in Etter, *Underwater Acoustic Modeling and
    Simulation*, Table 2.1.
    """
    t, s, p = _water_at_pressure(temperature, salinity, depth,
                                 pressure_dbar, latitude_deg,
                                 'sound_speed_delgrosso')
    p = p / _DBAR_PER_KGF_CM2                           # dbar -> kg/cm²
    t68 = t * 1.00024                                   # ITS-90 -> IPTS-68
    if np.any(t < _COLDEST_SEAWATER_C) or np.any(t > 35):
        _warnings.warn(
            f"Del Grosso sound speed: temperature outside validated range "
            f"[{_COLDEST_SEAWATER_C:g}, 35] °C; treating as extrapolation.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    if np.any(s < 29) or np.any(s > 43):
        _warnings.warn(
            "Del Grosso sound speed: salinity outside validated range "
            "[29, 43] ppt; treating as extrapolation. This equation was fitted "
            "to open-ocean salinities only — for fresher water use "
            "sound_speed_unesco, which is validated to S = 0.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    # Bounds are tested on the converted kg/cm² so they read as the paper
    # states them; the argument itself is decibars, 9.80665 times the number.
    if np.any(p < 0) or np.any(p > 1000):
        _warnings.warn(
            "Del Grosso sound speed: pressure outside validated range "
            "[0, 1000] kg/cm² = [0, 9807] dbar (this argument is in DECIBARS); "
            "treating as extrapolation.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )

    t = t68
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


# TEOS-10 Gibbs function of seawater, g(S_A, t, p) = g_W(t, p) + g_S(S_A, t, p),
# as a polynomial in the reduced variables x = sqrt(S_A / S_u), y = t / t_u,
# z = p / p_u with S_u = 40 × 35.16504/35 g/kg, t_u = 40 °C, p_u = 1e8 Pa
# (TEOS-10 manual, IOC Manuals and Guides 56, Table D.4). Pure-water
# coefficients g_jk from appendix G (IAPWS-09, 41 terms); saline coefficients
# g_ijk from appendix H (IAPWS-08, 64 terms), where i = 1 multiplies x²·ln x
# and i ≥ 2 multiplies x^i. Both tables are transcribed from the manual and
# pinned against the GSW toolbox in test_acoustics_check_values.py.
_TEOS10_SU_G_PER_KG = 40.0 * 35.16504 / 35.0


_TEOS10_TU_C = 40.0


_TEOS10_PU_PA = 1.0e8


_TEOS10_PSS78_TO_G_PER_KG = 35.16504 / 35.0    # Practical → Reference Salinity


_TEOS10_GW = (                                   # (j, k, g_jk)
    (0, 0, 0.101342743139674e3), (0, 1, 0.100015695367145e6),
    (0, 2, -0.254457654203630e4), (0, 3, 0.284517778446287e3),
    (0, 4, -0.333146754253611e2), (0, 5, 0.420263108803084e1),
    (0, 6, -0.546428511471039),
    (1, 0, 0.590578347909402e1), (1, 1, -0.270983805184062e3),
    (1, 2, 0.776153611613101e3), (1, 3, -0.196512550881220e3),
    (1, 4, 0.289796526294175e2), (1, 5, -0.213290083518327e1),
    (2, 0, -0.123577859330390e5), (2, 1, 0.145503645404680e4),
    (2, 2, -0.756558385769359e3), (2, 3, 0.273479662323528e3),
    (2, 4, -0.555604063817218e2), (2, 5, 0.434420671917197e1),
    (3, 0, 0.736741204151612e3), (3, 1, -0.672507783145070e3),
    (3, 2, 0.499360390819152e3), (3, 3, -0.239545330654412e3),
    (3, 4, 0.488012518593872e2), (3, 5, -0.166307106208905e1),
    (4, 0, -0.148185936433658e3), (4, 1, 0.397968445406972e3),
    (4, 2, -0.301815380621876e3), (4, 3, 0.152196371733841e3),
    (4, 4, -0.263748377232802e2),
    (5, 0, 0.580259125842571e2), (5, 1, -0.194618310617595e3),
    (5, 2, 0.120520654902025e3), (5, 3, -0.552723052340152e2),
    (5, 4, 0.648190668077221e1),
    (6, 0, -0.189843846514172e2), (6, 1, 0.635113936641785e2),
    (6, 2, -0.222897317140459e2), (6, 3, 0.817060541818112e1),
    (7, 0, 0.305081646487967e1), (7, 1, -0.963108119393062e1),
)


_TEOS10_GS = (                                   # (i, j, k, g_ijk)
    (1, 0, 0, 5812.81456626732), (1, 1, 0, 851.226734946706),
    (2, 0, 0, 1416.27648484197), (3, 0, 0, -2432.14662381794),
    (4, 0, 0, 2025.80115603697), (5, 0, 0, -1091.66841042967),
    (6, 0, 0, 374.601237877840), (7, 0, 0, -48.5891069025409),
    (2, 1, 0, 168.072408311545), (3, 1, 0, -493.407510141682),
    (4, 1, 0, 543.835333000098), (5, 1, 0, -196.028306689776),
    (6, 1, 0, 36.7571622995805), (2, 2, 0, 880.031352997204),
    (3, 2, 0, -43.0664675978042), (4, 2, 0, -68.5572509204491),
    (2, 3, 0, -225.267649263401), (3, 3, 0, -10.0227370861875),
    (4, 3, 0, 49.3667694856254), (2, 4, 0, 91.4260447751259),
    (3, 4, 0, 0.875600661808945), (4, 4, 0, -17.1397577419788),
    (2, 5, 0, -21.6603240875311), (4, 5, 0, 2.49697009569508),
    (2, 6, 0, 2.13016970847183),
    (2, 0, 1, -3310.49154044839), (3, 0, 1, 199.459603073901),
    (4, 0, 1, -54.7919133532887), (5, 0, 1, 36.0284195611086),
    (2, 1, 1, 729.116529735046), (3, 1, 1, -175.292041186547),
    (4, 1, 1, -22.6683558512829), (2, 2, 1, -860.764303783977),
    (3, 2, 1, 383.058066002476), (2, 3, 1, 694.244814133268),
    (3, 3, 1, -460.319931801257), (2, 4, 1, -297.728741987187),
    (3, 4, 1, 234.565187611355),
    (2, 0, 2, 384.794152978599), (3, 0, 2, -52.2940909281335),
    (4, 0, 2, -4.08193978912261), (2, 1, 2, -343.956902961561),
    (3, 1, 2, 83.1923927801819), (2, 2, 2, 337.409530269367),
    (3, 2, 2, -54.1917262517112), (2, 3, 2, -204.889641964903),
    (2, 4, 2, 74.7261411387560),
    (2, 0, 3, -96.5324320107458), (3, 0, 3, 68.0444942726459),
    (4, 0, 3, -30.1755111971161), (2, 1, 3, 124.687671116248),
    (3, 1, 3, -29.4830643494290), (2, 2, 3, -178.314556207638),
    (3, 2, 3, 25.6398487389914), (2, 3, 3, 113.561697840594),
    (2, 4, 3, -36.4872919001588),
    (2, 0, 4, 15.8408172766824), (3, 0, 4, -3.41251932441282),
    (2, 1, 4, -31.6569643860730), (2, 2, 4, 44.2040358308000),
    (2, 3, 4, -11.1282734326413),
    (2, 0, 5, -2.62480156590992), (2, 1, 5, 7.04658803315449),
    (2, 2, 5, -7.92001547211682),
)


def _teos10_gibbs_derivative(n_t, n_p, x, y, z):
    """``∂^(n_t+n_p) g / ∂t^n_t ∂p^n_p`` of the TEOS-10 Gibbs function, in
    SI units (J/kg per K^n_t per Pa^n_p), at reduced ``(x, y, z)``.

    Only temperature and pressure derivatives are needed for sound speed, so
    the salinity factor ``X_i(x)`` (``1`` for the pure-water table, ``x²·ln x``
    for ``i = 1``, ``x^i`` for ``i ≥ 2``) is never differentiated. Terms
    whose power is below the derivative order vanish and are skipped rather
    than evaluated as ``0 × y^(negative)``, which would be ``nan`` at ``y = 0``.
    """
    def falling(power, order):
        out = 1.0
        for r in range(order):
            out *= power - r
        return out

    # x²·ln x → 0 as x → 0⁺: the limit, not 0 × (−inf).
    with np.errstate(divide='ignore', invalid='ignore'):
        x2lnx = np.where(x > 0, x * x * np.log(np.where(x > 0, x, 1.0)), 0.0)

    total = np.zeros(np.broadcast(x, y, z).shape, dtype=float)
    for j, k, g in _TEOS10_GW:
        if j < n_t or k < n_p:
            continue
        total = total + (g * falling(j, n_t) * falling(k, n_p)
                         * y ** (j - n_t) * z ** (k - n_p))
    for i, j, k, g in _TEOS10_GS:
        if j < n_t or k < n_p:
            continue
        xi = x2lnx if i == 1 else x ** i
        total = total + (g * falling(j, n_t) * falling(k, n_p)
                         * xi * y ** (j - n_t) * z ** (k - n_p))
    return total / (_TEOS10_TU_C ** n_t * _TEOS10_PU_PA ** n_p)


def sound_speed_teos10(temperature=REFERENCE_TEMPERATURE_C,
                       salinity=REFERENCE_SALINITY_PSU, *, depth=None,
                       pressure_dbar=None, latitude_deg=None):
    """Speed of sound in seawater — TEOS-10 (IOC, SCOR and IAPSO 2010).

    Evaluates Eqn. (2.17.1) of the TEOS-10 manual,
    ``c = g_P·sqrt(g_TT / (g_TP² − g_TT·g_PP))``, on the full Gibbs function
    of seawater: the IAPWS-09 pure-water part plus the IAPWS-08 saline part
    (Feistel 2008), with the coefficient tables of the manual's appendices G
    and H. This is the ``sound_speed_t_exact`` of the GSW toolbox, written
    out in numpy; it needs no library.

    Same argument triple as :func:`sound_speed_unesco` and
    :func:`sound_speed_delgrosso`: ITS-90 temperature, **Practical Salinity**
    and pressure in **decibars**. TEOS-10 is stated in Absolute Salinity
    (g/kg); the conversion applied here is the Reference-Salinity factor
    ``35.16504/35`` (manual Eqn. 2.4.1), which is exact for seawater of
    Reference Composition. The remaining Absolute Salinity anomaly
    ``δS_A(lon, lat, p)`` of real seawater — at most ≈ 0.025 g/kg in the deep
    North Pacific, ≈ 0.03 m/s of sound speed — needs the global lookup
    atlas and is not applied.

    **Why a third equation.** The Gibbs function was fitted to the laboratory
    sound-speed data (manual appendix O, Table O.1; rms 0.035 m/s) and so
    reproduces Del Grosso (1974) to within a few cm/s over the ocean, while
    the uncorrected Chen–Millero polynomial of :func:`sound_speed_unesco`
    carries a pressure-dependent bias of about +0.6 m/s below 3000 dbar
    (APL-UW TR 9407, "Chen-Millero-Li Equation"; Etter §2, citing Dushaw et
    al. 1993). Choose this equation when the profile must agree with a
    TEOS-10-based oceanographic tool or with travel-time work.

    Valid over the manual's §2.6 range: ``S_A ∈ [0, 42] g/kg`` (``S ∈
    [0, 41.80]`` on the Practical scale), ``t ∈ [−6, 40] °C`` and
    ``p ∈ [0, 10000] dbar``. Outside it the result is an extrapolation and a
    ``ValidityWarning`` says so, the contract the siblings keep; a negative
    salinity is undefined (``x = sqrt(S_A/S_u)``) and returns NaN. The cold
    end needs no relaxation: −6 °C already covers every polar cast.

    Parameters
    ----------
    temperature : float or array
        Temperature [°C, ITS-90].
    salinity : float or array
        Practical salinity [PSU, PSS-78].
    depth : float or array_like, optional (keyword)
        Depth [m] the equation is evaluated at, converted to pressure (see
        below).
    pressure_dbar : float or array_like, optional (keyword)
        Sea pressure [dbar] — decibars, *not* Pa —, for pressure-native
        data; ``None`` with no ``depth`` is the surface.
    latitude_deg : float, optional (keyword)
        Latitude [deg] of the depth -> pressure conversion; default 45.

    Returns
    -------
    float or ndarray
        Sound speed [m/s]; a Python float for scalar input, otherwise the
        broadcast shape of the three arguments.

    **Where it is evaluated.** ``depth`` (m, keyword) is converted to the
    pressure the equation is stated in with :func:`depth_to_pressure_dbar`
    (Leroy & Parthiot's standard ocean) at ``latitude_deg``, default
    ``REFERENCE_LATITUDE_DEG`` (45°) — the conversion
    :meth:`~uacpy.core.ssp.SoundSpeedProfile.from_temperature_salinity` and
    :func:`sound_speed_at_depth` use. ``pressure_dbar`` (dbar, keyword)
    gives the pressure itself, for pressure-native data such as an Argo
    cast; pass one of the two, neither being the surface.

    **Water properties.** ``temperature`` and ``salinity`` each take a
    single value, an array broadcast with the coordinate, or ``(depth,
    value)`` pairs (m) interpolated linearly onto ``depth`` (end values
    held), which then must be given
    (:func:`uacpy.core._validate.water_property`).

    References
    ----------
    IOC, SCOR and IAPSO (2010). *The international thermodynamic equation of
    seawater – 2010: Calculation and use of thermodynamic properties.*
    Intergovernmental Oceanographic Commission, Manuals and Guides No. 56,
    UNESCO. §2.6 (validity), §2.17 Eqn. (2.17.1), appendices G, H, O.
    Feistel, R. (2008). "A Gibbs function for seawater thermodynamics for
    −6 to 80 °C and salinity up to 120 g kg⁻¹." Deep-Sea Res. I 55, 1639-1671.
    """
    t, s, p = _water_at_pressure(temperature, salinity, depth,
                                 pressure_dbar, latitude_deg,
                                 'sound_speed_teos10')
    s_max = 42.0 / _TEOS10_PSS78_TO_G_PER_KG          # 42 g/kg on the PSS-78 scale
    if np.any(t < -6.0) or np.any(t > 40.0):
        _warnings.warn(
            "TEOS-10 sound speed: temperature outside validated range "
            "[-6, 40] °C; treating as extrapolation.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    if np.any(s < 0):
        _warnings.warn(
            "TEOS-10 sound speed: salinity below 0 is undefined, not "
            "extrapolated — x = sqrt(S_A/S_u) has no real value there, so "
            "the result is NaN.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    elif np.any(s > s_max):
        _warnings.warn(
            f"TEOS-10 sound speed: salinity outside validated range "
            f"[0, {s_max:.2f}] PSU (S_A = 42 g/kg); treating as "
            f"extrapolation.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    if np.any(p < 0) or np.any(p > 10000.0):
        _warnings.warn(
            "TEOS-10 sound speed: pressure outside validated range "
            "[0, 10000] dbar (this argument is in DECIBARS); treating as "
            "extrapolation.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )

    with np.errstate(invalid='ignore'):
        x = np.sqrt(s * _TEOS10_PSS78_TO_G_PER_KG / _TEOS10_SU_G_PER_KG)
    y = t / _TEOS10_TU_C
    z = p * 1.0e4 / _TEOS10_PU_PA                     # dbar → Pa → reduced
    g_p = _teos10_gibbs_derivative(0, 1, x, y, z)
    g_tt = _teos10_gibbs_derivative(2, 0, x, y, z)
    g_tp = _teos10_gibbs_derivative(1, 1, x, y, z)
    g_pp = _teos10_gibbs_derivative(0, 2, x, y, z)
    with np.errstate(invalid='ignore'):
        c = g_p * np.sqrt(g_tt / (g_tp * g_tp - g_tt * g_pp))
    return float(c) if np.ndim(c) == 0 else c


def depth_to_pressure_dbar(depth_m, latitude_deg) -> np.ndarray:
    """Depth (m) → pressure (dbar), Leroy & Parthiot (1998) standard ocean.

    JASA 103(3), 1346-1352, eqs. (8)-(11) — ``h(Z,phi) = h(Z,45)·k(Z,phi)``
    with ``Z`` in metres and ``h`` in MPa, hence the ×100 to dbar. The authors
    give the fit as accurate to ±500 Pa over the whole depth/latitude range.

    This is the standard ocean: the per-region geopotential corrective term
    ``delta_h_i`` of their eq. (12) / Table II is not applied, so a basin with
    a strongly non-standard T/S profile (Mediterranean, Baltic, Black Sea)
    carries that residual. ``sound_speed_*`` expect dbar.

    Parameters
    ----------
    depth_m : float or array_like
        Depth (m).
    latitude_deg : float
        Latitude (deg).
    """
    z = np.asarray(depth_m, dtype=float)
    phi = np.radians(latitude_deg)
    g_phi = 9.7803 * (1 + 5.3e-3 * np.sin(phi) ** 2)
    h45 = (1.00818e-2 * z + 2.465e-8 * z ** 2
           - 1.25e-13 * z ** 3 + 2.8e-19 * z ** 4)          # MPa
    k = (g_phi - 2e-5 * z) / (9.80612 - 2e-5 * z)
    return h45 * k * 100.0                                   # MPa → dbar


def pressure_dbar_to_depth(pressure_dbar, latitude_deg) -> np.ndarray:
    """Pressure (dbar) → depth (m): Newton inversion of
    :func:`depth_to_pressure_dbar`.

    Pressure-indexed sources (Argo reports pressure) need depth for a
    ``SoundSpeedProfile``, and only the depth → pressure direction has a
    closed form. The derivative is taken as a central difference on
    :func:`depth_to_pressure_dbar` rather than analytically, so the Leroy &
    Parthiot coefficients live in exactly one place; a 1 m step is safe
    because ``h(z)`` is a smooth quartic whose curvature over a metre is
    negligible against its ~1 dbar/m slope.

    Parameters
    ----------
    pressure_dbar : float or array_like
        Pressure (dbar).
    latitude_deg : float
        Latitude (deg).
    """
    p = np.asarray(pressure_dbar, dtype=float)
    z = p * 0.9905                                   # ~1 m per dbar initial guess
    for _ in range(5):
        f = depth_to_pressure_dbar(z, latitude_deg) - p
        df = (depth_to_pressure_dbar(z + 1.0, latitude_deg)
              - depth_to_pressure_dbar(z - 1.0, latitude_deg)) / 2.0
        z = z - f / df
    return z


def _adiabatic_gradient(salinity, temp, pres):
    """Adiabatic temperature gradient (°C/dbar), Bryden (1973).

    The polynomial as published in UNESCO Technical Papers in Marine Science
    44 (1983), Fofonoff & Millard, routine ``ATG``. Salinity is practical
    salinity (PSS-78), ``temp`` in-situ °C, ``pres`` in decibars. The paper's
    check value ``ATG(S=40, T=40, P=10000) = 3.255976e-4 °C/dbar`` is pinned
    in the tests.
    """
    ds = np.asarray(salinity, dtype=float) - 35.0
    t = np.asarray(temp, dtype=float)
    p = np.asarray(pres, dtype=float)
    return ((((-2.1687e-16 * t + 1.8676e-14) * t - 4.6206e-13) * p
             + ((2.7759e-12 * t - 1.1351e-10) * ds
                + ((-5.4481e-14 * t + 8.733e-12) * t - 6.7795e-10) * t
                + 1.8741e-8)) * p
            + (-4.2393e-8 * t + 1.8932e-6) * ds
            + ((6.6228e-10 * t - 6.836e-8) * t + 8.5258e-6) * t + 3.5803e-5)


def _shift_adiabatically(salinity, temp, pres_from, pres_to):
    """Move a water parcel adiabatically from ``pres_from`` to ``pres_to``.

    Fourth-order Runge-Kutta integration of :func:`_adiabatic_gradient` over
    the pressure interval, in the coefficient form of UNESCO 44's ``THETA``
    (Fofonoff 1977). One RK4 step spans the whole interval, which is what the
    reference routine does and what its check value
    ``THETA(S=40, T=40, P=10000, Pr=0) = 36.89073 °C`` certifies; the gradient
    is a slowly varying polynomial, so the round trip closes to better than
    2e-4 °C over the full oceanic range (both pinned in the tests).
    """
    p = np.asarray(pres_from, dtype=float)
    t = np.asarray(temp, dtype=float)
    h = np.asarray(pres_to, dtype=float) - p
    xk = h * _adiabatic_gradient(salinity, t, p)
    t = t + 0.5 * xk
    q = xk
    p = p + 0.5 * h
    xk = h * _adiabatic_gradient(salinity, t, p)
    t = t + 0.29289322 * (xk - q)
    q = 0.58578644 * xk + 0.121320344 * q
    xk = h * _adiabatic_gradient(salinity, t, p)
    t = t + 1.707106781 * (xk - q)
    q = 3.414213562 * xk - 4.121320344 * q
    p = p + 0.5 * h
    xk = h * _adiabatic_gradient(salinity, t, p)
    return t + (xk - 2.0 * q) / 6.0


def insitu_from_potential(*, salinity, theta, pressure_dbar) -> np.ndarray:
    """Potential temperature (°C, referenced to the surface) → in-situ °C.

    Ocean models report ``thetao``, the temperature a parcel *would* have if
    brought adiabatically to 0 dbar; the sound-speed equations (UNESCO,
    Del Grosso) want the temperature the parcel actually has at depth. The two
    are the same at the surface and diverge with pressure: a parcel is warmed
    by compression, so in-situ is always the warmer of the pair below 0 dbar.

    Ignoring the difference is a deep-water error, not a uniform one. At
    S=34.7 the in-situ excess and the sound-speed error it costs are

    ======  ==========  =======  ==============
    depth   theta (°C)  ΔT (°C)  Δc (m/s)
    ======  ==========  =======  ==============
    2000 m         2.5    0.149  +0.64
    5000 m         1.5    0.462  +1.97
    10000 m        1.2    1.286  +4.98
    ======  ==========  =======  ==============

    (UNESCO; Del Grosso agrees within 0.2 m/s.) Climatology and float sources
    are unaffected — WOA23's ``t_an`` and Argo's ``TEMP`` are already in-situ.

    Keyword-only: every other seawater function here takes temperature first,
    and a positional ``(theta, salinity, …)`` call read as ``(salinity,
    theta, …)`` returned 36.1 °C for a 2 °C parcel with nothing to flag it.

    Parameters
    ----------
    salinity : array_like
        Practical salinity (PSS-78). Conserved by the adiabatic shift.
    theta : array_like
        Potential temperature (°C), referenced to 0 dbar.
    pressure_dbar : array_like
        In-situ pressure (dbar), as :func:`depth_to_pressure_dbar` returns.

    Returns
    -------
    numpy.ndarray
        In-situ temperature (°C).
    """
    return np.asarray(
        _shift_adiabatically(salinity, theta, 0.0, pressure_dbar), dtype=float)


#: The reference latitude the depth<->pressure standard ocean is stated at.
#: Leroy & Parthiot write h(Z, phi) = h(Z, 45) * k(Z, phi), so 45 deg is the
#: equation's own base case rather than an invented value: using it where the
#: caller's latitude is unknown costs at most half the equator-to-pole spread,
#: 0.53 % of pressure, which is 0.24 m/s of sound speed at 5000 m and 0.04 at
#: 1000 m -- under the difference between the formulas themselves.
REFERENCE_LATITUDE_DEG = 45.0


def _mackenzie_at_pressure(temperature, salinity, pressure_dbar):
    """Mackenzie (1981) on the ``(T, S, p_dbar)`` signature the table's other
    entries share. :func:`uacpy.core.acoustics.sound_speed_mackenzie` takes DEPTH in
    metres as its third argument, so the pressure is inverted through the
    same Leroy & Parthiot standard ocean the callers converted with, at the
    45° reference latitude ``extend_ssp_below_data`` defaults to (exact
    there; a column converted at another latitude comes back within the
    ±0.26 % of ``k(Z, φ)``, under 15 m at 5500 m)."""
    depth = pressure_dbar_to_depth(pressure_dbar, REFERENCE_LATITUDE_DEG)
    return sound_speed_mackenzie(temperature=temperature, salinity=salinity, depth=depth)


#: The formula every public ``formula=`` keyword defaults to. Named once so
#: every signature carries the same default rather than its own literal.
DEFAULT_SOUND_SPEED_FORMULA = 'teos10'

def _on_pressure(equation):
    """``equation`` on the positional ``(T, S, p_dbar)`` signature of
    :data:`SOUND_SPEED_FORMULAS`: its ``pressure_dbar=`` keyword."""
    def at_pressure(temperature, salinity, pressure_dbar):
        return equation(temperature, salinity, pressure_dbar=pressure_dbar)
    at_pressure.__name__ = f"{equation.__name__}_at_pressure"
    at_pressure.__doc__ = (f":func:`{equation.__name__}` at a pressure "
                           f"(dbar) given positionally.")
    return at_pressure


#: The three equations stated in pressure, by ``formula`` name.
_PRESSURE_EQUATIONS = {
    'unesco': sound_speed_unesco,
    'delgrosso': sound_speed_delgrosso,
    'teos10': sound_speed_teos10,
}

#: ``formula`` name -> ``(T °C, S PSU, p dbar) -> c m/s``, for pressure-native
#: columns. It lives here, with the four equations it indexes, and every
#: consumer imports it: the fetchers and ``extend_ssp_below_data`` (which
#: continues a column under the formula that built it,
#: ``SoundSpeedProfile.formula``).
SOUND_SPEED_FORMULAS = {
    **{name: _on_pressure(eq) for name, eq in _PRESSURE_EQUATIONS.items()},
    'mackenzie': _mackenzie_at_pressure,
}


def canonical_formula(formula, who: str) -> str:
    """The :data:`SOUND_SPEED_FORMULAS` key ``formula`` names, in any case
    (``'TEOS10'`` → ``'teos10'``); ``None`` is
    :data:`DEFAULT_SOUND_SPEED_FORMULA`. Refuses an unknown name."""
    if formula is None:
        return DEFAULT_SOUND_SPEED_FORMULA
    from uacpy.core._validate import canonical_choice
    return canonical_choice(formula, tuple(SOUND_SPEED_FORMULAS), who,
                            'formula')


def sound_speed_at_depth(
    temperature: Union[float, np.ndarray],
    salinity: Union[float, np.ndarray],
    depth: Union[float, np.ndarray],
    *,
    formula: Optional[str] = None,
    latitude_deg: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """Sound speed (m/s) of a cast held in depth, under any of the four
    equations.

    UNESCO, Del Grosso and TEOS-10 are stated in pressure, so ``depth`` is
    converted with :func:`depth_to_pressure_dbar` at ``latitude_deg``
    (default ``REFERENCE_LATITUDE_DEG``, 45°). Mackenzie is stated in depth
    and is evaluated on ``depth`` directly, so its answer does not depend on
    the latitude. This is the function
    :meth:`~uacpy.core.ssp.SoundSpeedProfile.from_temperature_salinity`
    builds a profile with.

    Parameters
    ----------
    temperature, salinity : float, ndarray or (N, 2) pairs
        In-situ temperature (°C) and practical salinity (psu): a single
        value, an array broadcast with ``depth``, or ``(depth, value)``
        pairs interpolated linearly onto ``depth``
        (:func:`uacpy.core._validate.water_property`).
    depth : float or ndarray
        Depth (m, positive down).
    formula : {'teos10', 'unesco', 'delgrosso', 'mackenzie'}, optional
        Default ``DEFAULT_SOUND_SPEED_FORMULA`` (TEOS-10).
    latitude_deg : float, optional
        Latitude of the cast (degrees) for the depth -> pressure conversion.

    Examples
    --------
    >>> round(float(sound_speed_at_depth(2.0, 34.7, 3000.0,
    ...                                  formula='mackenzie')), 2)
    1507.9
    """
    formula = canonical_formula(formula, 'sound_speed_at_depth')
    temperature = water_property(temperature, depth, name='temperature',
                                 who='sound_speed_at_depth')
    salinity = water_property(salinity, depth, name='salinity',
                              who='sound_speed_at_depth')
    if formula == 'mackenzie':
        return sound_speed_mackenzie(temperature=temperature,
                                     salinity=salinity, depth=depth)
    return _PRESSURE_EQUATIONS[formula](
        temperature, salinity, depth=depth, latitude_deg=latitude_deg)

def density(
    temperature: Union[float, np.ndarray] = REFERENCE_TEMPERATURE_C,
    salinity: Union[float, np.ndarray] = REFERENCE_SALINITY_PSU,
) -> Union[float, np.ndarray]:
    """
    Calculate density of sea water near the surface.

    Uses Fofonoff (1985 - IES 80) formula. EOS-80 is stated on the IPTS-68
    temperature scale; the ITS-90 argument is converted with Saunders'
    ``t68 = 1.00024·t90`` (TEOS-10 manual §2.1), as in
    :func:`sound_speed_unesco`.

    Parameters
    ----------
    temperature : float or ndarray, optional
        Water temperature in degrees Celsius, ITS-90 (default: 10, the
        package's reference sea water)
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
    Density: 1027.0 kg/m³

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
    t = np.asarray(temperature, dtype=float) * 1.00024    # ITS-90 -> IPTS-68
    salinity = _sequence_as_array(salinity)
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
    sound_speed: Optional[float] = None,
) -> Union[float, np.ndarray]:
    """
    Calculate Doppler-shifted frequency.

    The approximation is valid when speed << sound_speed (typical for
    underwater vehicles).

    Parameters
    ----------
    speed : float or ndarray
        Relative speed between transmitter and receiver in m/s
        (positive = approaching, negative = receding)
    frequency : float or ndarray
        Transmission frequency in Hz
    sound_speed : float, optional
        Sound speed in m/s (default: ``DEFAULT_SOUND_SPEED``, the nominal
        1500 m/s every sound-speed default in the package reads)

    Returns
    -------
    float or ndarray
        Doppler shifted frequency in Hz as perceived by the receiver.
        Evaluated element-wise, so an array argument gives an array of the
        broadcast shape.

    See Also
    --------
    uacpy.comms.doppler_from_speed : the dimensionless scale factor
        ``a = v/c`` on the same default ``c``; ``doppler(v, f) == f·(1 + a)``.

    Examples
    --------
    >>> f_shifted = doppler(2, 50000)  # 2 m/s approach
    >>> print(f"Shifted frequency: {f_shifted:.2f} Hz")
    Shifted frequency: 50066.67 Hz

    >>> f_shifted = doppler(-1, 50000)  # 1 m/s receding
    >>> print(f"Shifted frequency: {f_shifted:.2f} Hz")
    Shifted frequency: 49966.67 Hz
    """
    if sound_speed is None:
        sound_speed = DEFAULT_SOUND_SPEED
    speed = _sequence_as_array(speed)
    frequency = _sequence_as_array(frequency)
    return (1 + speed / sound_speed) * frequency
