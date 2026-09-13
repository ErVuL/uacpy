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
Abraham, D.A. *Underwater Acoustic Signal Processing*, eq. (2.69).
Jensen et al. *Computational Ocean Acoustics*, §1.7.1 eq. (1.83).
APL-UW (1994). *High-Frequency Ocean Environmental Acoustic Models Handbook*,
Technical Report APL-UW TR 9407, Section II.B (Surface: Backscatter) — the
10-100 kHz sea-surface model, ``apl_uw_surface_backscatter``. Its seabed
counterpart lives in :mod:`uacpy.sonar.bottom_scattering`.
"""

from __future__ import annotations

import numpy as np

import warnings

from scipy.optimize import brentq

from uacpy.core.acoustics import bubble_surface_loss
from uacpy.core.exceptions import ConfigurationError
from uacpy.core._warn_frames import USER_FRAME_SKIP

# Mackenzie (1961) deep-water bottom backscattering constant 10*log10(mu) [dB].
# He measured it constant at this value for both 530 and 1030 Hz (Etter,
# *Underwater Acoustic Modeling and Simulation*, eq. 9.6). For unconsolidated
# sediments the empirical spread is -25 to -35 dB, with -29 dB a common first
# guess (Jensen et al., *Computational Ocean Acoustics*, §1.7.2) — pass mu_dB to
# pick another point in that range.
LAMBERT_MU_DB = -27.0

#: 1 international knot in m/s, for comparing against JKPS' m/s ceiling.
_KNOT_TO_MS = 1852.0 / 3600.0

#: Grazing angle past which Lambert's law stops matching the data. Etter
#: Sect. 9.2: the relationship "appears to provide a good approximation to the
#: observed data for many deep-water bottoms at grazing angles below about
#: 45 deg". JKPS Sect. 1.7.2 gives the same law without an angular bound, so
#: Etter is the source for this one.
_LAMBERT_GOOD_GRAZING_DEG = 45.0

#: Band Chapman & Harris (1962) FITTED. JKPS Sect. 1.7.1: the curves are
#: "derived from measurements over the frequency range of 400-6400 Hz and wind
#: speed up to 15 m/s".
_CH_FIT_FREQ_HZ = (400.0, 6400.0)

#: Wind-speed ceiling of that fit, from the same sentence. Converted to knots
#: at the call site because this module's argument is in knots.
_CH_MAX_WIND_MS = 15.0

#: Grazing angle past which JKPS Sect. 1.7.1 says the formula stops working:
#: "This simplified formula performs well for grazing angles below 40-50 deg,
#: but fails to account for the high-angle roughness effects." The stricter end
#: of that range is the one worth warning at.
_CH_GOOD_GRAZING_DEG = 50.0

#: Outer grazing angle Chapman & Scott (1964) took DATA to, which is a wider
#: claim than accuracy. Etter Sect. 9.2: "Chapman and Scott (1964) later
#: validated these results over the frequency range 0.1 kHz to 6.4 kHz for
#: grazing angle below 80 deg." Quoted in the message, not used as the
#: threshold — a measurement range is not an accuracy bound.
_CH_SCOTT_GRAZING_DEG = 80.0
_CH_SCOTT_FREQ_HZ = (100.0, 6400.0)


def _warn_outside_chapman_harris_fit(theta, frequency_hz: float,
                                    wind_speed_kn: float) -> None:
    """Warn where the Chapman-Harris fit is being read outside its envelope.

    It is a fit to measurements, not an approximation to a computable
    quantity, so there is no exact reference to bound its error against — the
    fitted envelope is the only information about how far a value can be
    trusted. It does not fail loudly either: the form stays smooth, monotone
    and physically plausible far outside the band, so an extrapolated value is
    indistinguishable from a validated one. Evaluated at 10 kn and 10 deg
    grazing it runs from -76.50 dB at 100 Hz to -28.76 dB at 200 kHz, a 48 dB
    span across which nothing in the return value marks which part was ever
    measured.

    The two corpus sources do not agree on the grazing limit, so both are
    carried. JKPS Sect. 1.7.1 is the accuracy statement and sets the threshold
    (40-50 deg); Etter Sect. 9.2 reports the wider angle Chapman & Scott
    (1964) took DATA over (80 deg), which is a claim about measurement
    coverage rather than about the formula being right there.
    """
    lo, hi = _CH_FIT_FREQ_HZ
    if not lo <= frequency_hz <= hi:
        s_lo, s_hi = _CH_SCOTT_FREQ_HZ
        extra = ("" if s_lo <= frequency_hz <= s_hi else
                 f" It is also outside the {s_lo:g}-{s_hi:g} Hz range Chapman "
                 f"& Scott (1964) validated (Etter Sect. 9.2).")
        warnings.warn(
            f"chapman_harris_surface: frequency {frequency_hz:g} Hz is "
            f"outside the {lo:g}-{hi:g} Hz band Chapman & Harris (1962) "
            f"fitted (JKPS Sect. 1.7.1); the value is an extrapolation of an "
            f"empirical fit and carries no validated error bound.{extra} "
            f"apl_uw_surface_backscatter covers 10-100 kHz.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )

    wind_ms = float(wind_speed_kn) * _KNOT_TO_MS
    if wind_ms > _CH_MAX_WIND_MS:
        warnings.warn(
            f"chapman_harris_surface: wind speed {wind_speed_kn:g} kn "
            f"({wind_ms:.1f} m/s) exceeds the {_CH_MAX_WIND_MS:g} m/s ceiling "
            f"of the measurements Chapman & Harris (1962) fitted "
            f"(JKPS Sect. 1.7.1).",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )

    steep = np.asarray(theta, dtype=float)
    steep = steep[np.isfinite(steep) & (steep > _CH_GOOD_GRAZING_DEG)]
    if steep.size:
        warnings.warn(
            f"chapman_harris_surface: {steep.size} grazing angle(s) exceed "
            f"{_CH_GOOD_GRAZING_DEG:g} deg (steepest "
            f"{float(steep.max()):g} deg). JKPS Sect. 1.7.1: the formula "
            f"\"performs well for grazing angles below 40-50 deg, but fails "
            f"to account for the high-angle roughness effects\". Chapman & "
            f"Scott (1964) took data out to {_CH_SCOTT_GRAZING_DEG:g} deg "
            f"(Etter Sect. 9.2), but that is measurement coverage, not an "
            f"accuracy bound.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )


def lambert_bottom(grazing_deg, mu_dB: float = LAMBERT_MU_DB):
    """Bottom backscattering strength from Lambert's law.

    ``S_b(theta) = 10*log10(mu) + 10*log10(sin^2 theta) = mu_dB + 20*log10(sin theta)``

    Etter Sect. 9.2, citing Urick (1983) Ch. 8: the relationship "appears to
    provide a good approximation to the observed data for many deep-water
    bottoms at grazing angles below about 45 deg". Steeper angles warn.

    ``mu_dB`` defaults to Mackenzie's -27 dB — Etter Eq. 9.6: "The term
    10 log10 mu was found to be constant at -27 dB for both frequencies"
    (530 and 1030 Hz, deep water). JKPS Sect. 1.7.2 puts the unconsolidated
    -sediment spread at -25 to -35 dB with -29 dB "a popular first guess".

    Parameters
    ----------
    grazing_deg : float or array
        Grazing angle measured from the horizontal (degrees).
    mu_dB : float
        Lambert coefficient ``10*log10(mu)`` in dB.

    Returns
    -------
    ndarray
        Backscattering strength (dB). ``-inf`` at zero grazing angle.

    See Also
    --------
    uacpy.sonar.apl_uw_bottom_backscatter : the TR 9407 six-parameter model
        for 10-100 kHz, with the frequency and sediment dependence this
        law has none of.
    """
    theta_deg = np.asarray(grazing_deg, dtype=float)
    # Negative and non-finite angles reach the same warning the surface law
    # gives them, for the same reason: ``sin`` of a negative angle is negative
    # and ``sin(inf)`` is NaN, so both return a non-finite level. Until now the
    # -10 deg case carried only numpy's bare "invalid value encountered in
    # log10", which names neither this function nor the argument, and the NaN
    # case carried nothing at all. theta = 0 stays the documented -inf, so this
    # warns rather than raising.
    if np.any(~np.isfinite(theta_deg) | ~(theta_deg >= 0)):
        warnings.warn(
            "lambert_bottom: grazing angle(s) that are negative or non-finite "
            "return a non-finite level — grazing angles are measured from "
            "horizontal and must be >= 0 and finite.",
            UserWarning, stacklevel=2)
    # Same gap the surface law had: a documented bound that nothing enforced.
    steep = theta_deg[np.isfinite(theta_deg)
                      & (theta_deg > _LAMBERT_GOOD_GRAZING_DEG)]
    if steep.size:
        warnings.warn(
            f"lambert_bottom: {steep.size} grazing angle(s) exceed "
            f"{_LAMBERT_GOOD_GRAZING_DEG:g} deg (steepest "
            f"{float(steep.max()):g} deg). Etter Sect. 9.2: Lambert's law "
            f"\"appears to provide a good approximation to the observed data "
            f"for many deep-water bottoms at grazing angles below about "
            f"45 deg\"; above it the sin^2 form is an extrapolation.",
            UserWarning, stacklevel=2,
        )
    theta = np.deg2rad(theta_deg)
    # ``invalid`` joins ``divide`` because the warning above names the cases
    # that raise it: without that named warning, a negative or non-finite angle
    # escapes only as numpy's anonymous "invalid value encountered", which
    # points at this file rather than at the caller's argument. The surface law
    # silences both the same way.
    with np.errstate(divide="ignore", invalid="ignore"):
        return mu_dB + 20.0 * np.log10(np.sin(theta))


def chapman_harris_surface(grazing_deg, wind_speed_kn: float, frequency: float):
    """Sea-surface backscattering strength, Chapman & Harris (1962).

    ``S_s = 3.3*beta*log10(theta/30) - 42.4*log10(beta) + 2.6``
    with ``beta = 158*(v*f**(1/3))**(-0.58)``.

    Envelope, kept separate by source because the two disagree:

    * **Fitted** over 400-6400 Hz and wind speed up to 15 m/s — the
      measurements Chapman & Harris (1962) derived the curves from
      (JKPS Sect. 1.7.1).
    * **Accuracy**: "performs well for grazing angles below 40-50 deg, but
      fails to account for the high-angle roughness effects" (JKPS
      Sect. 1.7.1). This is what the grazing warning uses.
    * **Data coverage**: Chapman & Scott (1964) "validated these results over
      the frequency range 0.1 kHz to 6.4 kHz for grazing angle below 80 deg"
      (Etter Sect. 9.2). A wider measurement range is not a wider accuracy
      bound, so it is reported rather than used as the threshold.

    Note the earlier docstring attributed "0.4-6.4 kHz" to Chapman & Scott;
    that band is Chapman & Harris's fit, and Chapman & Scott reach down to
    0.1 kHz.

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

    See Also
    --------
    uacpy.sonar.apl_uw_surface_backscatter : the TR 9407 bubble-plus-roughness
        model for 10-100 kHz (wind speed in m/s there).
    """
    v = float(wind_speed_kn)
    f = float(frequency)
    # Negated admissible condition so NaN is refused: ``nan <= 0`` is False and
    # a NaN wind speed or frequency would return a silent NaN scattering
    # strength. ``isfinite`` is the other half of the message's "and finite":
    # ``inf > 0`` is True, and either argument infinite sends ``beta`` to 0, so
    # the ``-42.4*log10(beta)`` term returns +inf — an infinitely loud sea
    # surface — while the extrapolation warning below fires as if the value
    # were merely out of band.
    if (not np.isfinite(v) or not (v > 0.0)
            or not np.isfinite(f) or not (f > 0.0)):
        raise ConfigurationError(
            f"chapman_harris_surface: wind_speed_kn and frequency must be > 0 "
            f"and finite; got wind_speed_kn={v!r}, frequency={f!r}"
        )
    theta = np.asarray(grazing_deg, dtype=float)
    _warn_outside_chapman_harris_fit(theta, f, v)
    beta = 158.0 * (v * f ** (1.0 / 3.0)) ** (-0.58)
    # -42.4 with beta = 158*(v_kn*f^(1/3))^-0.58 is Abraham, *Underwater
    # Acoustic Signal Processing*, eq. (2.69) — the form implemented here, wind
    # speed in knots. Jensen et al., *Computational Ocean Acoustics* eq. (1.83)
    # writes the same -42.4 with beta = 107*(w_ms*f^(1/3))^-0.58 in m/s, which
    # is the same coefficient rescaled (107*1.94384^0.58 = 157.3).
    # The literature is split on the second coefficient: Etter, *Underwater
    # Acoustic Modeling and Simulation* eq. (9.2), citing Urick (1983) Ch. 8 for
    # the same knots-based beta, prints -42.2. The difference is 0.21 dB at
    # 10 kn / 1 kHz; uacpy follows the two sources that agree.
    # theta = 0 (horizontal) is -inf, the honest degenerate answer, matching
    # lambert_bottom; silence only the divide warning that case raises. A
    # NEGATIVE or non-finite angle is bad input, diagnosed like lambert_bottom
    # rather than silenced. Written as ``~(theta >= 0)`` so NaN is caught:
    # ``nan < 0`` is False, and a NaN angle otherwise returned a NaN level with
    # no warning at all. ``isfinite`` is needed alongside it because ``+inf``
    # satisfies ``>= 0`` and so escaped the warning entirely — and it is the one
    # bad angle that does NOT go to NaN: ``log10(inf/30)`` is +inf, so the level
    # comes back +inf, which is why the warning promises a non-finite level
    # rather than NaN specifically.
    if np.any(~np.isfinite(theta) | ~(theta >= 0)):
        warnings.warn(
            "chapman_harris: grazing angle(s) that are negative or non-finite "
            "return a non-finite level — grazing angles are measured from "
            "horizontal and must be >= 0 and finite.",
            UserWarning, stacklevel=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 3.3 * beta * np.log10(theta / 30.0) - 42.4 * np.log10(beta) + 2.6


def column_scattering_strength(sv_dB, thickness_m: float):
    """Integrate a volume scattering strength to a column (area) strength.

    ``S_col = S_v + 10*log10(thickness)`` — turns ``S_v`` (dB re 1/m, per unit
    volume) into an equivalent per-unit-area strength for a scattering layer of
    the given thickness, so it can be combined with boundary strengths.

    Parameters
    ----------
    sv_dB : float or array
        Volume scattering strength (dB re 1/m).
    thickness_m : float
        Layer thickness (m), > 0.
    """
    # Negated admissible condition so a NaN thickness is refused instead of
    # propagating into a silent NaN column strength. ``isfinite`` is the other
    # half of the message's "and finite": ``inf > 0`` is True, so an infinitely
    # thick layer returned an infinite column strength, which then poisons any
    # boundary strength it is combined with.
    if not np.isfinite(thickness_m) or not (thickness_m > 0.0):
        raise ConfigurationError(
            f"column_scattering_strength: thickness_m must be > 0 and finite; "
            f"got {thickness_m!r}"
        )
    return np.asarray(sv_dB, dtype=float) + 10.0 * np.log10(thickness_m)


# ── APL-UW TR 9407 sea-surface backscatter (10-100 kHz) ──────────────────────
# Section II.B of the handbook: the backscattering strength of a wind-driven
# sea surface as the sum of scattering by the near-surface bubble layer
# (Eqs. 2-8) and by the air-sea interface — Bragg scattering from small-scale
# ripples at low and moderate angles (Eqs. 9-10) and specular facets near
# vertical (Eqs. 11-12), joined by a logistic interpolation (Eqs. 13-15) and
# attenuated by the two-way passage through the bubble layer (Eq. 16, with the
# surface bubble loss of II.C.4 that ``bubble_surface_loss`` implements).
# Equation numbers are the report's (Section II). Wind speed is in m/s 10 m
# above the surface, frequency in Hz here (the report's fits take kHz).
# Bases the report names: McDaniel & Gorman (1982), JASA 72; McDaniel (1993),
# JASA 93.

#: Reradiation damping coefficient of a resonant bubble (Eq. 4).
_DELTA_R = 0.0136
#: Wind speed above which the bubble density follows the ``(U/11)^3.5`` law
#: (Eq. 5).
_BUBBLE_WIND_BREAK_MS = 11.0
#: Small-scale (Bragg) roughness cross section is zero above this angle
#: (Eq. 9); the interpolation of Eqs. 13-15 smooths the step.
_BRAGG_MAX_DEG = 85.0
#: Wind speed below which the mean-square facet slope is held at 0.0034
#: (Eq. 12).
_FACET_WIND_FLOOR_MS = 1.0
#: The facet lobe's -15 dB point defines the interpolation angle (p. II-9).
_FACET_LOBE_DB = 15.0
#: Below this grazing angle the strength is extrapolated linearly from its
#: values at 1 and 0.5 degrees, as II.B.5.a recommends.
_EXTRAPOLATE_BELOW_DEG = 0.5
#: Band the handbook covers; its comparisons ran 12-70 kHz (II.B.5.b).
_FREQUENCY_LIMITS_HZ = (10e3, 100e3)


def _bubble_cross_section(theta_rad, f_khz: float, wind_mps: float):
    """Eqs. 2-6: backscattering by the near-surface bubble layer."""
    if wind_mps < _BUBBLE_WIND_BREAK_MS:
        beta_v = 10.0 ** (-5.2577 + 0.4701 * wind_mps) * (f_khz / 25.0) ** 0.85
    else:
        beta_v = (10.0 ** (-5.2577 + 0.4701 * _BUBBLE_WIND_BREAK_MS)
                  * (f_khz / 25.0) ** 0.85
                  * (wind_mps / _BUBBLE_WIND_BREAK_MS) ** 3.5)
    delta = 2.55e-2 * f_khz ** (1.0 / 3.0)                       # Eq. 6
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        beta = beta_v / np.sin(theta_rad)                        # Eq. 3
        bracket = (1.0 + 8.0 * beta * np.exp(-2.0 * beta)
                   - np.exp(-4.0 * beta)) / (2.0 * beta)
        sigma_b = beta_v / (4.0 * np.pi) * _DELTA_R / delta * bracket
    # beta -> inf (grazing 0) is the saturation limit sin(theta) delta_r /
    # (8 pi delta) (Eq. 7), which is 0 there; a NaN from 0/0 means that.
    return np.where(np.isfinite(sigma_b), sigma_b, 0.0)


def _facet_slope_sq(wind_mps: float) -> float:
    """Eq. 12: mean-square surface slope."""
    if wind_mps < _FACET_WIND_FLOOR_MS:
        return 0.0034
    return 4.6e-3 * np.log(2.1 * wind_mps ** 2)


def _facet_cross_section(theta_rad, s2: float):
    """Eq. 11: specular-point facets, gamma = 90 deg - theta."""
    g = np.pi / 2.0 - theta_rad
    with np.errstate(over='ignore'):
        return (1.0 / np.cos(g) ** 4) / (4.0 * np.pi * s2) * np.exp(-np.tan(g) ** 2 / s2)


def _facet_transition_deg(s2: float) -> float:
    """The smallest grazing angle at which the facet lobe has fallen
    ``_FACET_LOBE_DB`` below its vertical peak (p. II-9), found on the
    continuous lobe rather than on the 1-degree grid the report's plots
    used: ``tan^2(g)/s^2 - 4 ln cos(g) = 1.5 ln 10``."""
    target = _FACET_LOBE_DB / 10.0 * np.log(10.0)

    def excess(g):
        return np.tan(g) ** 2 / s2 - 4.0 * np.log(np.cos(g)) - target

    g = brentq(excess, 1e-9, np.pi / 2.0 - 1e-9)
    return 90.0 - np.degrees(g)


def _strength(theta_deg: np.ndarray, f_hz: float, wind_mps: float) -> np.ndarray:
    theta = np.deg2rad(theta_deg)
    f_khz = f_hz / 1000.0
    sigma_b = _bubble_cross_section(theta, f_khz, wind_mps)
    a_u = 1.3e-5 * wind_mps ** 2                                  # Eq. 10
    sigma_sc = np.where(theta_deg <= _BRAGG_MAX_DEG,
                        a_u * np.tan(theta) ** 4, 0.0)           # Eq. 9
    s2 = _facet_slope_sq(wind_mps)
    sigma_f = _facet_cross_section(theta, s2)
    theta_f = _facet_transition_deg(s2)
    fx = 1.0 / (1.0 + np.exp(np.clip(0.524 * (theta_f - theta_deg),
                                     -40.0, 40.0)))              # Eqs. 13, 15
    sigma_r1 = fx * sigma_f + (1.0 - fx) * sigma_sc              # Eq. 14
    # Eq. 16: the two-way passage through the bubble layer. SBL is a power
    # loss; the helper returns the amplitude multiplier 10^(-SBL/20), so its
    # square is 10^(-SBL/10). It takes the angle from the surface normal.
    extinction = bubble_surface_loss(wind_mps, f_hz, np.pi / 2.0 - theta) ** 2
    sigma_r = sigma_r1 * extinction
    with np.errstate(divide='ignore'):
        return 10.0 * np.log10(sigma_r + sigma_b)


def apl_uw_surface_backscatter(grazing_deg, frequency: float,
                               wind_speed_mps: float):
    """Sea-surface backscattering strength, TR 9407 II.B (Eqs. 1-16), dB.

    ``S_s = 10 log10(sigma_r + sigma_b)``: the near-surface bubble layer
    (Eqs. 2-6), which dominates at low and moderate grazing angles, plus the
    interface — Bragg scattering from ripples, ``A(U) tan^4 theta`` (Eqs.
    9-10), and specular facets near vertical (Eqs. 11-12), interpolated at
    the facet lobe's -15 dB angle (Eqs. 13-15) and attenuated by the two-way
    passage through the bubble layer (Eq. 16).

    Parameters
    ----------
    grazing_deg : float or array
        Grazing angle from the horizontal, 0-90 degrees. Below 0.5 deg the
        value is extrapolated linearly from those at 1 and 0.5 deg, as
        II.B.5.a recommends; the model itself tends to -inf at zero grazing.
    frequency : float
        Acoustic frequency (Hz). The handbook's band is 10-100 kHz and its
        comparisons ran 12-70 kHz with no frequency trend in accuracy;
        outside the band the call warns.
    wind_speed_mps : float
        Wind speed 10 m above the surface (m/s), >= 0. Note the unit:
        :func:`chapman_harris_surface` takes knots.

    Returns
    -------
    ndarray
        Backscattering strength (dB), the shape of ``grazing_deg``.

    Notes
    -----
    Accuracy (II.B.5.b): about +-4 dB at all grazing angles for winds above
    about 8 m/s; +-5 dB below, where wave breaking — hence bubble production
    — is uncertain, and where about half the data sets fell faster with
    decreasing angle over 5-20 deg than the model. The report recommends
    running a +-1 m/s spread about the measured wind. Bubble scattering
    saturates at high bubble density (Eq. 7), which is why the curves stop
    moving with wind above about 8 m/s (II.B.3).
    """
    theta_deg = np.asarray(grazing_deg, dtype=float)
    if np.any(~np.isfinite(theta_deg) | (theta_deg < 0.0) | (theta_deg > 90.0)):
        raise ConfigurationError(
            f"apl_uw_surface_backscatter: grazing angles must be finite and "
            f"within 0-90 deg; got {grazing_deg!r}.")
    f = float(frequency)
    u = float(wind_speed_mps)
    if not (np.isfinite(f) and f > 0.0):
        raise ConfigurationError(
            f"apl_uw_surface_backscatter: frequency must be > 0 and finite "
            f"(Hz); got {frequency!r}.")
    if not (np.isfinite(u) and u >= 0.0):
        raise ConfigurationError(
            f"apl_uw_surface_backscatter: wind_speed_mps must be >= 0 and "
            f"finite (m/s); got {wind_speed_mps!r}.")
    lo, hi = _FREQUENCY_LIMITS_HZ
    if not lo <= f <= hi:
        warnings.warn(
            f"apl_uw_surface_backscatter: frequency {f:g} Hz is outside the "
            f"{lo/1e3:g}-{hi/1e3:g} kHz band of TR 9407 (compared with data at "
            f"12-70 kHz, II.B.5.b); the value is an extrapolation.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)

    out = _strength(np.maximum(theta_deg, _EXTRAPOLATE_BELOW_DEG), f, u)
    low = theta_deg < _EXTRAPOLATE_BELOW_DEG
    if np.any(low):
        s_half, s_one = _strength(np.array([_EXTRAPOLATE_BELOW_DEG, 1.0]), f, u)
        slope = (s_one - s_half) / (1.0 - _EXTRAPOLATE_BELOW_DEG)
        out = np.where(low, s_half + slope * (theta_deg - _EXTRAPOLATE_BELOW_DEG), out)
    return out
