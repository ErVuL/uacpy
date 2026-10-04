"""Seawater volume attenuation on plain arrays.

The formulas the absorption models of :mod:`uacpy.core.absorption` are written
on, and the conversion between attenuation units:

* :func:`absorption_thorp` — Thorp (1967), frequency only;
* :func:`absorption_francois_garrison` — Francois & Garrison (1982): frequency,
  temperature, salinity, pH and depth;
* :func:`absorption_biological` — one fish swim-bladder resonance layer;
* :func:`ph_to_nbs` — a pH moved onto the NBS scale Francois–Garrison was
  fitted on;
* :func:`convert_attenuation_units` — dB/km, dB/m, dB/wavelength, Nepers/m,
  Q and L.

The models (:class:`~uacpy.core.absorption.Thorp`,
:class:`~uacpy.core.absorption.FrancoisGarrison`,
:class:`~uacpy.core.absorption.Biological`) and the ``absorption_*`` functions
return an :class:`~uacpy.core.absorption.AbsorptionCoefficient` carrying its
unit; these return numbers in the unit their name states.
"""

from typing import Optional, Union

import numpy as np

from uacpy.core.constants import (
    DEFAULT_SOUND_SPEED, NEPER_TO_DB, REFERENCE_DEPTH_M, REFERENCE_PH,
    REFERENCE_SALINITY_PSU, REFERENCE_TEMPERATURE_C,
)
from uacpy.core.exceptions import ConfigurationError
from uacpy.core._validate import water_property

__all__ = [
    'absorption_thorp', 'PH_SCALES', 'ph_to_nbs',
    'absorption_francois_garrison', 'absorption_biological',
    'convert_attenuation_units',
]


_ArrayLike = Union[float, np.ndarray]


def absorption_thorp(frequency: _ArrayLike,
                     depth: Optional[_ArrayLike] = None) -> np.ndarray:
    """Thorp seawater volume attenuation in dB/km.

    The formula :class:`~uacpy.core.absorption.Thorp` is written on, on plain
    arrays: ``Thorp().table(f)`` returns an :class:`~uacpy.core.absorption.AbsorptionCoefficient`
    that carries its unit as a value and converts and plots itself; this
    returns the dB/km numbers.

    Uses the JKPS Eq. (1.47) coefficients, which match the AT
    ``AttenMod.f90:93`` formula used internally by the Acoustics-Toolbox
    binaries character for character. Note AT's own comment there labels it
    "JKPS Eq. 1.34" — 1st-edition numbering for the same expression, which the
    2nd edition prints as (1.47). The two are not different formulas.

    **Conditions the coefficients were measured under.** JKPS states them
    immediately after the equation: "The above expression applies for a
    temperature of 4 deg C, a salinity of 35 ppt, a pH of 8.0, and a depth of
    about 1000 m, where most of the measurements on which it is based were
    made." Nothing here checks them, and the sensitivity is not small — JKPS
    goes on: "the low-frequency (< 1 kHz) attenuation in the North Pacific
    (pH = 7.7) is only about half that in the North Atlantic (pH = 8.0)", and
    "high-frequency (> 1 kHz) attenuation in, e.g., the Baltic (S = 8 ppt) is
    less than half that in open oceans". For a basin far from those nominal
    values, use Francois-Garrison instead (JKPS cites an overall accuracy of
    5 % for it; AT provides it as ``CASE ( 'F' )`` in the same routine).

    **Frequency band.** The 3.3e-3 dB/km constant term is not an absorption
    mechanism — JKPS attributes that regime to leakage out of the deep sound
    channel — so below ~50 Hz this and Francois-Garrison diverge hard (at 10 Hz
    3.3e-3 dB/km against FG's 1.2e-5 at 4 °C, 35 ppt, pH 8.0, 3000 m, and
    only FG is modelling absorption).
    ``docs/guide/environment.md §6 "Two things the curve does not tell you"``
    works the comparison through.

    Parameters
    ----------
    frequency : float or array
        Frequency in Hz.
    depth : float or array, optional
        Depth (m) the formula is evaluated at, taken for the call shape
        :func:`absorption_francois_garrison` has: Thorp has no depth term,
        so the value is the same at every depth and ``depth`` only broadcasts
        with ``frequency``. Default ``None``: the shape of ``frequency``.

    Returns
    -------
    ndarray, the broadcast shape of ``frequency`` and ``depth``
        0-d for scalar input; an array input keeps its shape (a
        1-element array stays 1-D).

    References
    ----------
    Thorp, W. H. (1967). JASA 42(1), 270 (original).
    Jensen, Kuperman, Porter, Schmidt — *Computational Ocean
    Acoustics*, 2nd ed., Eq. (1.47).
    """
    f = np.asarray(frequency, dtype=float) / 1000.0
    f2 = f * f
    a = (
        3.3e-3
        + 0.11 * f2 / (1.0 + f2)
        + 44.0 * f2 / (4100.0 + f2)
        + 3.0e-4 * f2
    )
    if depth is not None:
        a = a + np.zeros(np.shape(depth))
    return a


PH_SCALES = ('nbs', 'total', 'seawater')

def ph_to_nbs(pH, scale, *, temperature, salinity):
    """Move a seawater pH onto the NBS scale Francois–Garrison was fitted on.

    Francois & Garrison (1982, Part II) took their pH from Lovett's (1980)
    charts of the Gorshkov (1978) atlas, whose scale is not reported; Brewer
    & Hester (Oceanography 22(4), 2009) judge it "probably" NBS and state
    that "the sound absorption equations are based on the old NBS scale",
    and Uzhansky et al. (JGR Oceans, 2025, §2) read the formulas the same
    way. Modern data (GLODAP, the Copernicus BGC field) report pH on the
    **total** hydrogen-ion scale, which sits about 0.1 below NBS for the
    same water (Marion et al. 2011 via Uzhansky et al. 2025: NBS 8.332,
    free 8.195, total 8.087, seawater 8.078 at S = 35, 25 °C). Fed
    unconverted, the boric-acid term below 1 kHz comes out ~20 % low.

    The conversion is the one CO2SYS applies: Takahashi et al. (1982,
    GEOSECS Pacific Expedition vol. 3, p. 80) fitted the activity coefficient
    an NBS-buffer-calibrated glass electrode sees in seawater,
    ``fH(T, S) = 1.2948 − 0.002036·T_K + (0.0004607 − 1.475e-6·T_K)·S²``,
    and ``pH_NBS = pH_SWS − log10(fH)``: +0.100 at 4 °C / 35, +0.147 at
    25 °C / 35; the fit is stated valid for S in 20-40. That is exact for
    the ``'seawater'`` scale. The ``'total'`` scale sits about 0.01 above
    the seawater scale (the fluoride term; Marion et al. 2011: 8.087 vs
    8.078), so a ``'total'`` input converts about 0.01 high — neglected,
    well inside the 5 % the formula claims. It is chosen over
    Marion's Pitzer-model offset (0.245) because the 1970s atlas data were
    electrode readings against NBS buffers, which is what ``fH`` describes,
    not a thermodynamic single-ion activity.

    Parameters
    ----------
    pH : float or array
        The measured pH.
    scale : {'nbs', 'total', 'seawater'}
        The scale ``pH`` is on. ``'nbs'`` returns it unchanged.
    temperature, salinity : float or array
        In-situ temperature (°C) and Practical Salinity of the water the pH
        was measured in; broadcast against ``pH``.

    Returns
    -------
    float or ndarray
        pH on the NBS scale.
    """
    if scale not in PH_SCALES:
        raise ConfigurationError(
            f"ph_to_nbs: unknown pH scale {scale!r}.",
            remediation="Use 'nbs' (Francois-Garrison's own), 'total' "
                        "(GLODAP, Copernicus BGC) or 'seawater'.",
        )
    p = np.asarray(pH, dtype=float)
    if scale == 'nbs':
        return float(p) if np.ndim(p) == 0 else p
    t_k = np.asarray(temperature, dtype=float) + 273.15
    s = np.asarray(salinity, dtype=float)
    f_h = 1.2948 - 0.002036 * t_k + (0.0004607 - 0.000001475 * t_k) * s * s
    out = p - np.log10(f_h)
    return float(out) if np.ndim(out) == 0 else out


def absorption_francois_garrison(
    frequency: _ArrayLike,
    temperature: _ArrayLike = REFERENCE_TEMPERATURE_C,
    salinity: _ArrayLike = REFERENCE_SALINITY_PSU,
    pH: _ArrayLike = REFERENCE_PH,
    depth: Optional[_ArrayLike] = None,
) -> np.ndarray:
    """Francois–Garrison 1982 seawater volume attenuation in dB/km.

    Parameters
    ----------
    frequency : float or array
        Frequency in Hz.
    temperature : float, array or (N, 2) pairs
        Water temperature (°C). Default 10.
    salinity : float, array or (N, 2) pairs
        Salinity (PSU). Default 35.
    pH : float, array or (N, 2) pairs
        Acidity. Default 8.
    depth : float or array, optional
        Depth (m). Default ``None``, the surface (0 m).
        The defaults are the package's reference sea water
        (``core.constants.REFERENCE_*``). Each water property is a single
        value, an array broadcast with the others, or ``(depth, value)``
        pairs interpolated linearly onto ``depth`` (end values held), which
        then must be given (:func:`uacpy.core._validate.water_property`).

    Returns
    -------
    ndarray, the broadcast shape of the inputs
        0-d when every input is scalar; array inputs keep their
        broadcast shape (a 1-element array stays 1-D).

    Notes
    -----
    Implementation follows the Acoustics Toolbox ``AttenMod.f90``.

    Inputs the formula has no value for — a negative salinity under the
    ``sqrt(S/35)`` of the boric-acid relaxation, a temperature at or below
    ``-273`` °C — return NaN here rather than raising; ``AttenMod.f90``
    states units and no validity range, and checks neither.
    :class:`FrancoisGarrison` refuses them at construction instead.

    References
    ----------
    Francois & Garrison (1982). JASA 72(6), 1879–1890.
    """
    who = 'absorption_francois_garrison'
    f = np.asarray(frequency, dtype=float) / 1000.0
    T = np.asarray(water_property(temperature, depth, name='temperature',
                                  who=who), dtype=float)
    S = np.asarray(water_property(salinity, depth, name='salinity', who=who),
                   dtype=float)
    pH = np.asarray(water_property(pH, depth, name='pH', who=who),
                    dtype=float)
    z = np.asarray(REFERENCE_DEPTH_M if depth is None else depth,
                   dtype=float)

    c = 1412.0 + 3.21 * T + 1.19 * S + 0.0167 * z

    # Three additive mechanisms, each ``A * P * f_relax * f^2 / (f_relax^2 +
    # f^2)``: two chemical relaxations plus pure-water viscosity. ``A`` is the
    # strength, ``P`` the pressure (depth) correction, ``f1``/``f2`` the
    # relaxation frequencies in kHz.

    # Boric acid B(OH)3, relaxing near 1 kHz — the only pH-dependent term.
    # 8.86/c is Francois & Garrison's coefficient, the one AttenMod.f90:151
    # evaluates. Medwin & Clay, *Fundamentals of Acoustical Oceanography*,
    # eq. (3.4.33), print 8.68/c: a misprint, which F&G's own Table IV
    # (Part II) settles — its 0.063 dB/km at 1 kHz, 4 degC, 35 psu, pH 8 is
    # what 8.86 gives (0.0631) and 8.68 does not (0.0621).
    A1 = 8.86 / c * 10.0 ** (0.78 * pH - 5.0)
    P1 = 1.0
    # A negative salinity makes the root NaN, which numpy reports as a raw
    # ``RuntimeWarning`` — the one warning category uacpy would emit that is
    # not a ``UACPYWarning``. :class:`FrancoisGarrison` rejects S < 0 at
    # construction; this bare function is documented to answer out-of-domain
    # input with NaN, so the invalid flag is silenced and the NaN carried.
    with np.errstate(invalid='ignore'):
        f1 = 2.8 * np.sqrt(S / 35.0) * 10.0 ** (4.0 - 1245.0 / (T + 273.0))

    # Magnesium sulphate MgSO4, relaxing near 65 kHz.
    A2 = 21.44 * S / c * (1.0 + 0.025 * T)
    P2 = 1.0 - 1.37e-4 * z + 6.2e-9 * z * z
    f2 = 8.17 * 10.0 ** (8.0 - 1990.0 / (T + 273.0)) / (1.0 + 0.0018 * (S - 35.0))

    # Viscosity of pure water: no relaxation frequency, so it enters as plain
    # f^2. Francois & Garrison fit A3 piecewise about 20 degC. P3's
    # 3.83e-5 is the coefficient AttenMod.f90:161 evaluates; Medwin & Clay
    # eq. (3.4.33) print 3.83e-3, a misprint: with it P3, and the
    # pure-water absorption, turn negative below 261 m. F&G Part I
    # (JASA 72, 896), which states the coefficient, is not in the
    # reference corpus.
    P3 = 1.0 - 3.83e-5 * z + 4.9e-10 * z * z
    A3_cold = 4.937e-4 - 2.59e-5 * T + 9.11e-7 * T * T - 1.5e-8 * T * T * T
    A3_warm = 3.964e-4 - 1.146e-5 * T + 1.45e-7 * T * T - 6.5e-10 * T * T * T
    A3 = np.where(T < 20.0, A3_cold, A3_warm)

    a = (
        A1 * P1 * (f1 * f * f) / (f1 * f1 + f * f)
        + A2 * P2 * (f2 * f * f) / (f2 * f2 + f * f)
        + A3 * P3 * f * f
    )
    return a


def absorption_biological(frequency: _ArrayLike, f0_hz: float, Q: float,
                         a0: float) -> np.ndarray:
    """One biological (fish swim-bladder) resonance layer's attenuation, dB/km.

    ``a = a0 / ((1 - f0**2/f**2)**2 + 1/Q**2)``, the form
    ``misc/AttenMod.f90`` evaluates inside a ``'B'`` layer (a formula the
    Acoustics-Toolbox manual attributes to Orest Diachok): ``f0_hz`` is the
    resonance frequency, ``Q`` its quality factor, and ``a0`` the amplitude in
    dB/km, scaled by ``Q**2`` at resonance. :class:`~uacpy.core.absorption.Biological`
    sums it over the layers a depth falls in.

    Parameters
    ----------
    frequency : float or array
        Frequency in Hz.
    f0_hz, Q, a0 : float
        The layer's resonance frequency (Hz), quality factor and amplitude
        (dB/km).

    Returns
    -------
    ndarray, same shape as ``frequency``
        0-d for a scalar input.
    """
    f = np.asarray(frequency, dtype=float)
    return a0 / ((1.0 - f0_hz ** 2 / f ** 2) ** 2 + 1.0 / Q ** 2)


# The units of :func:`convert_attenuation_units` whose definition carries a
# frequency: dB per wavelength lambda = c/f, and Q and L, both written against
# omega = 2*pi*f.
_FREQUENCY_DEPENDENT_UNITS = frozenset({'dB/wavelength', 'Q', 'L'})
# Every unit :func:`convert_attenuation_units` converts between.
_ATTENUATION_UNITS = ('dB/km', 'dB/m', 'dB/wavelength', 'Nepers/m', 'Q', 'L')


def convert_attenuation_units(
    alpha: _ArrayLike,
    frequency: float,
    from_unit: str,
    to_unit: str,
    sound_speed: float = DEFAULT_SOUND_SPEED,
) -> np.ndarray:
    """Convert volume attenuation between unit conventions.

    Every path goes through dB/m, so each unit needs only its own definition
    against the nepers/m attenuation ``a`` of ``exp(-a·x)``, at angular
    frequency ``omega = 2·pi·f`` and sound speed ``c`` (the same definitions
    Acoustics-Toolbox ``AttenMod.f90:57-80`` applies):

    - ``Nepers/m`` — ``a`` itself.
    - ``dB/m`` — ``a · 20/ln(10)``; the pivot every path converts through.
    - ``dB/km`` — dB of amplitude loss per 1000 m.
    - ``dB/wavelength`` — dB per ``lambda = c/f``, hence frequency-independent.
    - ``Q`` — quality factor, ``a = omega/(2·c·Q)``. Q divides, so a
      conversion *from* ``'Q'`` requires ``alpha > 0`` and raises
      :class:`ConfigurationError` otherwise. Going *to* ``'Q'`` from a zero
      attenuation returns ``inf`` — the lossless limit, which converts back
      to zero — rather than raising.
    - ``L`` — loss tangent, ``a = L·omega/c``.

    ``sound_speed`` is therefore required for the wavelength / Q / L paths and
    ignored for the rest, and ``frequency`` the same way: those three paths
    need a positive finite one and raise :class:`ConfigurationError` without
    it, while ``dB/km`` ↔ ``dB/m`` ↔ ``Nepers/m`` convert at any frequency.

    Returns an ndarray shaped like ``alpha``: 0-d for a scalar input; an
    array input keeps its shape (a 1-element array stays 1-D).

    Parameters
    ----------
    alpha : float or ndarray
        Attenuation in ``from_unit``.
    frequency : float
        Frequency (Hz); used by the ``'dB/wavelength'``, ``'Q'`` and ``'L'``
        paths, which need it positive and finite.
    from_unit, to_unit : str
        One of ``'dB/km'``, ``'dB/m'``, ``'dB/wavelength'``, ``'Nepers/m'``,
        ``'Q'``, ``'L'``.
    sound_speed : float, optional
        Sound speed (m/s) of the wavelength, Q and L paths. Default
        :data:`~uacpy.core.constants.DEFAULT_SOUND_SPEED`.

    Notes
    -----
    Acoustics-Toolbox ``AttenMod.f90`` also recognises two units that
    this helper does **not** convert:

    - ``'m'`` (lowercase) — dB/m with a frequency power-law
      ``α(f) = α₀ · (f/f₀)^β`` below a transition frequency ``fT``.
      Round-tripping needs the (``β``, ``f₀``, ``fT``) triple, which is
      outside the scalar-frequency contract here.
    - ``'F'`` — dB/(m·kHz), i.e. ``α(f) = α₀ · f[kHz]``. The single
      ``frequency`` argument would suffice, but the unit is rare enough
      that adding it would broaden the contract for one AT-only use.

    Pass through Acoustics-Toolbox directly (set ``TopOpt`` position 4
    to ``'m'`` or ``'F'``) if you need those formulas.
    """
    alpha = np.asarray(alpha, dtype=float)
    for label, unit in (('from_unit', from_unit), ('to_unit', to_unit)):
        if unit not in _ATTENUATION_UNITS:
            raise ConfigurationError(
                f"convert_attenuation_units: unknown unit {label}={unit!r}.",
                remediation=f"Use one of {list(_ATTENUATION_UNITS)}; the "
                            f"spelling is exact ('dB', not 'db').")

    # lambda = c/f, Q = omega/(2 c a) and L = a c/omega all divide by the
    # frequency, so f = 0 reaches the arithmetic as a bare ZeroDivisionError
    # on the wavelength paths and as a silent 0 or inf on the Q and L ones.
    # The rest of the table is a pure scaling and converts at any frequency.
    needs_frequency = {from_unit, to_unit} & _FREQUENCY_DEPENDENT_UNITS
    if needs_frequency and not (np.isfinite(frequency) and frequency > 0.0):
        raise ConfigurationError(
            f"convert_attenuation_units: {sorted(needs_frequency)} is defined "
            f"per wavelength or per cycle, so it needs a positive finite "
            f"frequency; got frequency={frequency!r}.",
            remediation="Pass the frequency the attenuation was measured at, "
                        "or convert between dB/km, dB/m and Nepers/m, which "
                        "carry no frequency.")
    # The same three units carry a sound speed, and it divides on every one of
    # them, so it needs the same guard as the frequency. Unguarded,
    # ``sound_speed=0`` returned 0.0 dB/wavelength from a real dB/km loss — a
    # lossless medium — and a negative speed returned a negative, i.e.
    # amplifying, attenuation, both silently.
    if needs_frequency and not (np.isfinite(sound_speed) and sound_speed > 0.0):
        raise ConfigurationError(
            f"convert_attenuation_units: {sorted(needs_frequency)} is defined "
            f"per wavelength or per cycle, so it needs a positive finite "
            f"sound speed; got sound_speed={sound_speed!r}.",
            remediation="Pass the sound speed of the medium the attenuation "
                        "was measured in, or convert between dB/km, dB/m and "
                        "Nepers/m, which carry no sound speed.")

    if from_unit == 'dB/km':
        alpha_dB_m = alpha / 1000.0
    elif from_unit == 'dB/m':
        alpha_dB_m = alpha
    elif from_unit == 'dB/wavelength':
        wavelength = sound_speed / frequency
        alpha_dB_m = alpha / wavelength
    elif from_unit == 'Nepers/m':
        alpha_dB_m = alpha * NEPER_TO_DB
    elif from_unit == 'Q':
        # Q sits in the denominator of alphaT = omega / (2 * c * Q), so a
        # non-positive Q has no attenuation to convert (Q -> inf is the
        # lossless limit).
        if np.any(alpha <= 0):
            raise ConfigurationError(
                f"convert_attenuation_units: from_unit='Q' requires a "
                f"positive quality factor (alphaT = omega / (2*c*Q)); "
                f"got {float(np.min(alpha)):g}."
            )
        alpha_nepers_m = np.pi * frequency / (alpha * sound_speed)
        alpha_dB_m = alpha_nepers_m * NEPER_TO_DB
    else:
        # 'L': alphaT = L * omega / c
        alpha_nepers_m = alpha * 2.0 * np.pi * frequency / sound_speed
        alpha_dB_m = alpha_nepers_m * NEPER_TO_DB

    if to_unit == 'dB/km':
        result = alpha_dB_m * 1000.0
    elif to_unit == 'dB/m':
        result = alpha_dB_m
    elif to_unit == 'dB/wavelength':
        wavelength = sound_speed / frequency
        result = alpha_dB_m * wavelength
    elif to_unit == 'Nepers/m':
        result = alpha_dB_m / NEPER_TO_DB
    elif to_unit == 'Q':
        alpha_nepers_m = alpha_dB_m / NEPER_TO_DB
        # A zero attenuation is the lossless limit and ``Q = omega/(2*c*a)``
        # -> inf is its exact value, so the division is answered rather than
        # trapped: ``inf`` converts back through ``from_unit='Q'`` to a = 0.
        # The mirror direction raises because Q = 0 is not the limit of
        # anything representable — it is a -> inf.
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.pi * frequency / (alpha_nepers_m * sound_speed)
    else:
        # 'L'
        alpha_nepers_m = alpha_dB_m / NEPER_TO_DB
        result = alpha_nepers_m * sound_speed / (2.0 * np.pi * frequency)

    return result
