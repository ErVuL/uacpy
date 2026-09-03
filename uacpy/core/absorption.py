"""Volume-attenuation models for the water column.

Volume attenuation is an environmental property — same water column, same
absorption regardless of which propagation solver runs over it. uacpy
stores it on :class:`~uacpy.core.environment.Environment` as
``env.absorption`` and each model writer reads it to emit the right
Acoustics-Toolbox ``TopOpt`` position-4 character and the supporting
per-formula parameters.

Concrete subclasses
-------------------
:class:`Thorp`
    Frequency-only seawater absorption (Thorp 1967). No free parameters.
:class:`FrancoisGarrison`
    Francois–Garrison (1982) frequency / T / S / pH / depth model.
:class:`Biological`
    Layered fish-bladder resonance model (multiple
    ``(Z_top, Z_bottom, f0, Q, a0)`` blocks).
:class:`ConstantAbsorption`
    Frequency-independent baseline written into every SSP-block ``alphaI``
    row (dB/wavelength). Useful for calibrated ad-hoc absorption.

Module-level numerics
---------------------
:func:`thorp_db_per_km`, :func:`francois_garrison_db_per_km`
    Bare formulas returning ``α(f)`` in dB/km. Useful for plotting
    attenuation curves without constructing an :class:`Absorption`.
:func:`convert_attenuation_units`
    Unit conversion helper (dB/km ↔ dB/m ↔ dB/wavelength ↔ Nepers/m
    ↔ Q ↔ L).
"""

from __future__ import annotations

import warnings

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

from uacpy.core.constants import (
    DEFAULT_SOUND_SPEED, MAX_ATTENUATION_DB_PER_WAVELENGTH, NEPER_TO_DB,
)
from uacpy.core.exceptions import ConfigurationError
from uacpy.core._carrier_validate import (
    _require_attenuation_in_range, _require_finite, _require_non_negative,
)
from uacpy.core._warn_frames import USER_FRAME_SKIP


_ArrayLike = Union[float, np.ndarray]


# ─────────────────────────────────────────────────────────────────────────────
# Bare numeric formulas
# ─────────────────────────────────────────────────────────────────────────────


def thorp_db_per_km(frequency: _ArrayLike) -> np.ndarray:
    """Thorp seawater volume attenuation in dB/km.

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
    3.3e-3 dB/km against FG's 1.2e-5, and only FG is modelling absorption).
    ``docs/guide/environment.md §6 "Two things the curve does not tell you"``
    works the comparison through.

    Parameters
    ----------
    frequency : float or array
        Frequency in Hz.

    Returns
    -------
    ndarray, same shape as ``frequency``
        0-d for a scalar input; an array input keeps its shape (a
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
    return a


def francois_garrison_db_per_km(
    frequency: _ArrayLike,
    temperature: _ArrayLike = 10.0,
    salinity: _ArrayLike = 35.0,
    pH: _ArrayLike = 8.0,
    depth: _ArrayLike = 1000.0,
) -> np.ndarray:
    """Francois–Garrison 1982 seawater volume attenuation in dB/km.

    Parameters
    ----------
    frequency : float or array
        Frequency in Hz.
    temperature : float or array
        Water temperature (°C). Default 10.
    salinity : float or array
        Salinity (PSU). Default 35.
    pH : float or array
        Acidity. Default 8.
    depth : float or array
        Depth (m). Default 1000.

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
    f = np.asarray(frequency, dtype=float) / 1000.0
    T = np.asarray(temperature, dtype=float)
    S = np.asarray(salinity, dtype=float)
    z = np.asarray(depth, dtype=float)
    pH = np.asarray(pH, dtype=float)

    c = 1412.0 + 3.21 * T + 1.19 * S + 0.0167 * z

    # Three additive mechanisms, each ``A * P * f_relax * f^2 / (f_relax^2 +
    # f^2)``: two chemical relaxations plus pure-water viscosity. ``A`` is the
    # strength, ``P`` the pressure (depth) correction, ``f1``/``f2`` the
    # relaxation frequencies in kHz.

    # Boric acid B(OH)3, relaxing near 1 kHz — the only pH-dependent term.
    A1 = 8.86 / c * 10.0 ** (0.78 * pH - 5.0)
    P1 = 1.0
    # A negative salinity makes the root NaN, which numpy reports as a raw
    # ``RuntimeWarning`` — the one warning category uacpy would emit that is
    # not a ``UserWarning``. :class:`FrancoisGarrison` rejects S < 0 at
    # construction; this bare function is documented to answer out-of-domain
    # input with NaN, so the invalid flag is silenced and the NaN carried.
    with np.errstate(invalid='ignore'):
        f1 = 2.8 * np.sqrt(S / 35.0) * 10.0 ** (4.0 - 1245.0 / (T + 273.0))

    # Magnesium sulphate MgSO4, relaxing near 65 kHz.
    A2 = 21.44 * S / c * (1.0 + 0.025 * T)
    P2 = 1.0 - 1.37e-4 * z + 6.2e-9 * z * z
    f2 = 8.17 * 10.0 ** (8.0 - 1990.0 / (T + 273.0)) / (1.0 + 0.0018 * (S - 35.0))

    # Viscosity of pure water: no relaxation frequency, so it enters as plain
    # f^2. Francois & Garrison fit A3 piecewise about 20 degC.
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


# The units of :func:`convert_attenuation_units` whose definition carries a
# frequency: dB per wavelength lambda = c/f, and Q and L, both written against
# omega = 2*pi*f.
_FREQUENCY_DEPENDENT_UNITS = frozenset({'dB/wavelength', 'Q', 'L'})


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
        alpha_db_m = alpha / 1000.0
    elif from_unit == 'dB/m':
        alpha_db_m = alpha
    elif from_unit == 'dB/wavelength':
        wavelength = sound_speed / frequency
        alpha_db_m = alpha / wavelength
    elif from_unit == 'Nepers/m':
        alpha_db_m = alpha * NEPER_TO_DB
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
        alpha_db_m = alpha_nepers_m * NEPER_TO_DB
    elif from_unit == 'L':
        # alphaT = L * omega / c
        alpha_nepers_m = alpha * 2.0 * np.pi * frequency / sound_speed
        alpha_db_m = alpha_nepers_m * NEPER_TO_DB
    else:
        raise ConfigurationError(f"Unknown unit: {from_unit}")

    if to_unit == 'dB/km':
        result = alpha_db_m * 1000.0
    elif to_unit == 'dB/m':
        result = alpha_db_m
    elif to_unit == 'dB/wavelength':
        wavelength = sound_speed / frequency
        result = alpha_db_m * wavelength
    elif to_unit == 'Nepers/m':
        result = alpha_db_m / NEPER_TO_DB
    elif to_unit == 'Q':
        alpha_nepers_m = alpha_db_m / NEPER_TO_DB
        # A zero attenuation is the lossless limit and ``Q = omega/(2*c*a)``
        # -> inf is its exact value, so the division is answered rather than
        # trapped: ``inf`` converts back through ``from_unit='Q'`` to a = 0.
        # The mirror direction raises because Q = 0 is not the limit of
        # anything representable — it is a -> inf.
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.pi * frequency / (alpha_nepers_m * sound_speed)
    elif to_unit == 'L':
        alpha_nepers_m = alpha_db_m / NEPER_TO_DB
        result = alpha_nepers_m * sound_speed / (2.0 * np.pi * frequency)
    else:
        raise ConfigurationError(f"Unknown unit: {to_unit}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Env-attachable Absorption hierarchy
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Absorption:
    """Abstract base for water-column absorption models. Do not
    instantiate directly — pick one of :class:`Thorp`,
    :class:`FrancoisGarrison`, :class:`Biological`,
    :class:`ConstantAbsorption`.

    Subclasses implement :meth:`_alpha_db_per_m`, which evaluates
    ``α(f, z)`` at the depths a model needs (used by the Kraken-class
    modal perturbation kernel; the Acoustics-Toolbox writers read
    :meth:`topopt_code` and the per-class fields instead). The public
    :meth:`alpha_db_per_m` checks the frequency and delegates to it.
    """

    def __post_init__(self):
        if type(self) is Absorption:
            raise ConfigurationError(
                "Absorption is abstract; instantiate Thorp / "
                "FrancoisGarrison / Biological / ConstantAbsorption."
            )

    def topopt_code(self) -> str:
        """Single Acoustics-Toolbox character for ``TopOpt`` position 4."""
        raise NotImplementedError

    def alpha_db_per_m(
        self,
        frequency: float,
        depths: _ArrayLike,
    ) -> np.ndarray:
        """Evaluate ``α(f, z)`` in dB/m at one frequency, on a depth grid.

        Parameters
        ----------
        frequency : float
            Frequency in Hz. Must be strictly positive — every model here is
            a formula in ``f`` (or in the wavelength ``c/f``) that has no
            value at or below zero. The check sits on this side of the
            dispatch so all four models share one rule: Thorp and
            Francois-Garrison are polynomials that evaluate happily at
            ``f <= 0`` and hand back a *positive* attenuation for a negative
            frequency. ``NaN`` is rejected here too.
        depths : float or 1-D array
            Depths (m).

        Returns
        -------
        ndarray, the shape of ``depths`` — a scalar depth comes back as a
        1-element array of shape ``(1,)``.
        """
        f = float(frequency)
        if not f > 0.0:
            raise ConfigurationError(
                f"{type(self).__name__}.alpha_db_per_m: frequency must be "
                f"> 0 Hz; got {frequency}"
            )
        return self._alpha_db_per_m(f, depths)

    def _alpha_db_per_m(
        self,
        frequency: float,
        depths: _ArrayLike,
    ) -> np.ndarray:
        """Model-specific ``α(f, z)`` in dB/m, reached through
        :meth:`alpha_db_per_m` with ``frequency`` already checked positive."""
        raise NotImplementedError

    def plot(self, frequencies, *, depth: float = 0.0, ax=None, **kwargs):
        """Plot this model's volume absorption ``α(f)`` (dB/km, log-log).

        Dispatches to :func:`uacpy.visualization.plot_absorption` — the carrier
        counterpart of :meth:`Result.plot`. ``frequencies`` (Hz) is required
        because absorption *is* a function of frequency; ``depth`` (m) is the
        evaluation depth (matters for depth-dependent models such as
        Francois-Garrison; Thorp is depth-invariant). ``ax`` draws into an
        existing Axes, spelled the way every other uacpy plot method spells
        it; the remaining ``kwargs`` are forwarded."""
        # Deferred into the body: ``uacpy.visualization`` imports
        # ``uacpy.core`` at module scope, so this line at file scope makes
        # ``import uacpy`` raise ImportError. docs/DEV.md section 7 records
        # the inversion.
        from uacpy.visualization import plot_absorption
        freqs = np.atleast_1d(np.asarray(frequencies, dtype=float))
        alpha_km = np.array([
            float(np.asarray(self.alpha_db_per_m(f, depth)).reshape(-1)[0])
            * 1000.0 for f in freqs])
        if not np.any(alpha_km > 0):
            warnings.warn(
                f"Absorption.plot: α(f) is entirely non-positive at depth "
                f"{depth:g} m, so the log-log plot will be blank. For layered "
                f"models (e.g. Biological) pick a depth inside a layer.",
                UserWarning, stacklevel=2)
        return plot_absorption(freqs, absorption=alpha_km, ax=ax, **kwargs)


@dataclass
class Thorp(Absorption):
    """Thorp (1967) seawater volume attenuation. No parameters.

    Frequency-only — α(f, z) is constant in depth.
    """

    def topopt_code(self) -> str:
        return 'T'

    def _alpha_db_per_m(
        self,
        frequency: float,
        depths: _ArrayLike,
    ) -> np.ndarray:
        a = float(thorp_db_per_km(float(frequency))) / 1000.0
        z = np.atleast_1d(np.asarray(depths, dtype=float))
        return np.full(z.shape, a)


@dataclass
class FrancoisGarrison(Absorption):
    """Francois–Garrison (1982) seawater absorption.

    The per-instance ``temperature_c``, ``salinity_psu``, ``pH``, and
    ``z_bar_m`` are the Acoustics-Toolbox single-row parameters. When
    :meth:`alpha_db_per_m` is called for a modal perturbation, the
    depth axis the caller provides overrides ``z_bar_m`` so the formula
    is evaluated per depth (pressure-corrected).

    Notes
    -----
    The four fields are checked only where the formula itself has no
    value there (see :func:`francois_garrison_db_per_km`): the
    boric-acid relaxation takes ``sqrt(S/35)``, its temperature factor
    is ``10**(4 - 1245/(T + 273))``, and all three mechanisms divide by
    the sound speed ``c = 1412 + 3.21·T + 1.19·S + 0.0167·z``. Nothing
    here narrows the inputs to a published validity envelope: neither
    ``misc/AttenMod.f90`` (``Franc_Garr``) nor ``Matlab/Misc/franc_garr.m``
    — the two implementations this one follows — states one.

    **The deck does not do what the accessor does.** Evaluating per depth is
    a deliberate refinement over the single-row model AT writes: the solver's
    ``Franc_Garr`` reads a module-level ``z_bar``
    (``misc/AttenMod.f90:148-160``) and applies the one resulting alpha at
    every depth. So an ``alpha_db_per_m`` sampled over a column and a run of
    the same environment absorb different amounts. On
    ``FrancoisGarrison(10, 35, 8, z_bar_m=1000)`` the accessor at the surface
    is +3.0 % over the deck at 1 kHz, +13.9 % at 10 kHz, +15.5 % at 30 kHz
    and +15.0 % at 100 kHz; it agrees exactly at ``z_bar_m`` and runs as far
    below at the bottom of a 2 km column. This matters wherever the two are
    combined rather than compared —
    :meth:`uacpy.core.results.modes.Modes.with_attenuation` documents the
    consequence for a modal perturbation.
    """
    temperature_c: float
    salinity_psu: float
    pH: float
    z_bar_m: float

    def __post_init__(self):
        Absorption.__post_init__(self)
        if not (self.salinity_psu >= 0):
            raise ConfigurationError(
                f"FrancoisGarrison: salinity_psu must be non-negative (PSU); "
                f"got {self.salinity_psu}. The boric-acid relaxation "
                f"frequency carries sqrt(S/35), which has no value below 0."
            )
        if not (self.temperature_c > -273.0):
            raise ConfigurationError(
                f"FrancoisGarrison: temperature_c must be above -273 (°C, "
                f"not kelvin); got {self.temperature_c}. The relaxation "
                f"frequencies carry 10**(4 - 1245/(T + 273)), which is "
                f"singular at -273 °C."
            )
        if not (self.pH >= 0):
            raise ConfigurationError(
                f"FrancoisGarrison: pH must be non-negative; got {self.pH}."
            )
        sound_speed = (1412.0 + 3.21 * self.temperature_c
                       + 1.19 * self.salinity_psu + 0.0167 * self.z_bar_m)
        if not (sound_speed > 0):
            raise ConfigurationError(
                f"FrancoisGarrison: the T/S/z row gives a sound speed of "
                f"{sound_speed:g} m/s (c = 1412 + 3.21·T + 1.19·S + "
                f"0.0167·z, with T={self.temperature_c}, "
                f"S={self.salinity_psu}, z={self.z_bar_m}); every absorption "
                f"mechanism divides by it, so it must be positive."
            )

    def topopt_code(self) -> str:
        return 'F'

    def as_at_tuple(self) -> Tuple[float, float, float, float]:
        """Tuple in the order the AT ``write_fg_params`` writer expects."""
        return (
            float(self.temperature_c), float(self.salinity_psu),
            float(self.pH), float(self.z_bar_m),
        )

    def _alpha_db_per_m(
        self,
        frequency: float,
        depths: _ArrayLike,
    ) -> np.ndarray:
        z = np.atleast_1d(np.asarray(depths, dtype=float))
        a_km = francois_garrison_db_per_km(
            frequency=float(frequency),
            temperature=self.temperature_c,
            salinity=self.salinity_psu,
            pH=self.pH,
            depth=z,
        )
        return a_km / 1000.0


@dataclass(init=False)
class BiologicalLayer:
    """Single fish-bladder resonance layer for :class:`Biological`.

    Parameters
    ----------
    z_top_m, z_bottom_m : float
        Depth bounds (m) of the layer in the water column.
    f0_hz : float
        Resonance frequency (Hz).
    Q : float
        Quality factor (dimensionless).
    a0 : float
        Acoustics-Toolbox resonance coefficient (dB/km): the absorption is
        ``a0 / ((1 - f0²/f²)² + 1/Q²)``, so the peak at ``f = f0`` is
        ``a0·Q²`` (see AttenMod.f90).

    Notes
    -----
    ``a0·Q²`` is checked against the AT solvers' ``CRCI`` ceiling and a
    ``UserWarning`` names the limit when it is over. ``AttenMod.f90``'s
    ``'B'`` branch (:105-106) adds ``a/8685.8896`` Nepers/m to ``alphaT``,
    :113 scales by ``c²/ω`` and :116 aborts the run when the result exceeds
    ``c`` — so a layer aborts once its dB/km absorption passes
    ``8685.8896·2πf/c``, which is 3638 dB/km at 100 Hz in 1500 m/s water
    (``a0 = 1, Q = 60`` gives 3600 and clears it; ``Q = 61`` gives 3721 and
    warns). This warns rather than raises because
    the peak is only reached when the run frequency sits on ``f0``, and the
    ceiling scales with the true ``c(z)`` over the layer, which a layer on
    its own does not carry — the check uses
    :data:`~uacpy.core.constants.DEFAULT_SOUND_SPEED`.

    The ``__init__`` is written out rather than generated (``init=False``)
    so that the ceiling warning can name the line the *user* wrote. A
    generated ``__init__`` lives in the pseudo-file ``<string>``, which the
    attribution walk cannot step over — it matches no package prefix, so the
    walk stops there — and which a hand-counted ``stacklevel`` can only count
    past for one nesting depth. :class:`Biological` builds layers from tuples
    inside its own constructor, so there are two depths and no single count
    covers both. Written out, every frame between the warning and the user is
    an ordinary ``absorption.py`` frame that :data:`USER_FRAME_SKIP` steps
    over, and both entry points land on the caller. ``@dataclass`` still
    supplies ``__repr__`` / ``__eq__`` / ``fields()`` from the annotations
    below; a test pins the signature against them.
    """
    z_top_m: float
    z_bottom_m: float
    f0_hz: float
    Q: float
    a0: float

    def __init__(self, z_top_m: float, z_bottom_m: float, f0_hz: float,
                 Q: float, a0: float):
        self.z_top_m = z_top_m
        self.z_bottom_m = z_bottom_m
        self.f0_hz = f0_hz
        self.Q = Q
        self.a0 = a0
        # Finiteness first, because every sign test below is a bare ``<=``
        # that NaN answers False to and that inf passes for ``a0``/``Q``: a
        # NaN layer reaches the AT deck through ``as_at_tuples`` and turns
        # AttenMod.f90's band test ``z >= Z1 .AND. z <= Z2`` (:104) False at
        # every depth, so it is written to the file and then contributes
        # nothing, while an inf ``a0`` reaches the ceiling arithmetic below
        # and reports its own limit as ``nan dB/km``. A negative depth is
        # inert the same way — the band test compares against a water-column
        # depth measured down from the surface — so the depths are held to
        # ``>= 0`` here rather than only to their ordering. This is the same
        # pair of guards every other core carrier routes through.
        _require_non_negative(self.z_top_m, "BiologicalLayer.z_top_m",
                              hint="metres below the surface")
        _require_non_negative(self.z_bottom_m, "BiologicalLayer.z_bottom_m",
                              hint="metres below the surface")
        _require_finite(self.f0_hz, "BiologicalLayer.f0_hz", hint="Hz")
        _require_finite(self.Q, "BiologicalLayer.Q", hint="dimensionless")
        _require_finite(self.a0, "BiologicalLayer.a0", hint="dB/km")
        if self.z_bottom_m <= self.z_top_m:
            raise ConfigurationError(
                "BiologicalLayer: z_bottom_m must be strictly greater than "
                f"z_top_m (got z_top_m={self.z_top_m}, "
                f"z_bottom_m={self.z_bottom_m})"
            )
        if self.f0_hz <= 0:
            raise ConfigurationError(
                f"BiologicalLayer: f0_hz must be positive (Hz); got {self.f0_hz}"
            )
        if self.Q <= 0:
            raise ConfigurationError(
                f"BiologicalLayer: Q must be positive (dimensionless); got {self.Q}"
            )
        if self.a0 <= 0:
            raise ConfigurationError(
                f"BiologicalLayer: a0 must be positive (dB/km); got {self.a0}"
            )
        # The Lorentzian peaks at f = f0, where the denominator is 1/Q², so
        # a0*Q² is the most absorption this layer can present to CRCI. Taken
        # to dB/wavelength at f0 it meets the same package-wide ceiling the
        # seabed and surface carriers are held to.
        peak_db_km = float(self.a0) * float(self.Q) ** 2
        peak_db_lambda = float(convert_attenuation_units(
            peak_db_km, float(self.f0_hz), 'dB/km', 'dB/wavelength',
            sound_speed=DEFAULT_SOUND_SPEED))
        if peak_db_lambda > MAX_ATTENUATION_DB_PER_WAVELENGTH:
            ceiling_db_km = peak_db_km * (
                MAX_ATTENUATION_DB_PER_WAVELENGTH / peak_db_lambda)
            warnings.warn(
                f"BiologicalLayer: the on-resonance peak a0*Q² = "
                f"{peak_db_km:g} dB/km is {peak_db_lambda:g} dB/wavelength at "
                f"f0={self.f0_hz:g} Hz in {DEFAULT_SOUND_SPEED:g} m/s water, "
                f"over the {MAX_ATTENUATION_DB_PER_WAVELENGTH:.4f} above which "
                f"misc/AttenMod.f90's CRCI (:116) finds an imaginary sound "
                f"speed larger than the real part and aborts. A run at or near "
                f"f0 will fail in every AT solver; the ceiling here is "
                f"{ceiling_db_km:g} dB/km, and scales with the water sound "
                f"speed over the layer.",
                # The walk, not a count: this constructor is reached from a
                # user's ``BiologicalLayer(...)`` and from the normalising
                # loop in ``Biological.__init__`` one frame further down, and
                # a hand count is right for only one of the two. Naming a
                # uacpy line is not merely untidy — ``warnings`` keys its
                # once-per-location registry on the attributed file and line,
                # so every ``Biological(...)`` in a program would collapse
                # onto the loop's line and only the first would be shown.
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP)


@dataclass(init=False)
class Biological(Absorption):
    """Layered biological volume attenuation (fish-bladder resonance).

    Parameters
    ----------
    layers : list of BiologicalLayer or tuples
        Each entry can be a :class:`BiologicalLayer` or a 5-tuple
        ``(z_top, z_bottom, f0, Q, a0)``.

    Notes
    -----
    The ``__init__`` is written out for the reason :class:`BiologicalLayer`
    gives: a tuple entry is turned into a layer *here*, so a generated
    ``__init__`` would put its un-attributable ``<string>`` frame between that
    layer's ceiling warning and the user, and stop the attribution walk on it.
    """
    layers: List[BiologicalLayer] = field(default_factory=list)

    def __init__(self,
                 layers: Optional[List[Union[BiologicalLayer, Tuple]]] = None):
        # Called directly rather than through a generated ``__init__``; it is
        # the abstract-base guard every ``Absorption`` subclass runs.
        Absorption.__post_init__(self)
        normalized: List[BiologicalLayer] = []
        for entry in layers or []:
            if isinstance(entry, BiologicalLayer):
                normalized.append(entry)
            else:
                z_top, z_bottom, f0, Q, a0 = entry
                normalized.append(BiologicalLayer(
                    z_top_m=float(z_top), z_bottom_m=float(z_bottom),
                    f0_hz=float(f0), Q=float(Q), a0=float(a0),
                ))
        if not normalized:
            raise ConfigurationError(
                "Biological absorption requires at least one layer; got 0."
            )
        self.layers = normalized

    def topopt_code(self) -> str:
        return 'B'

    def as_at_tuples(self) -> List[Tuple[float, float, float, float, float]]:
        """List of 5-tuples in the order the AT writer expects."""
        return [
            (layer.z_top_m, layer.z_bottom_m, layer.f0_hz, layer.Q, layer.a0)
            for layer in self.layers
        ]

    def _alpha_db_per_m(
        self,
        frequency: float,
        depths: _ArrayLike,
    ) -> np.ndarray:
        # Sum each layer's Lorentzian resonance over the depths it spans.
        # Matches AttenMod.f90: a = a0 / ((1 - f0²/f²)² + 1/Q²) in dB/km.
        f = float(frequency)
        z = np.atleast_1d(np.asarray(depths, dtype=float))
        a_km = np.zeros(z.shape, dtype=float)
        # Each layer spans the closed interval [z_top, z_bottom] and is
        # tested independently, exactly as the AttenMod.f90:102-109 loop
        # tests ``z >= Z1 .AND. z <= Z2`` per layer and sums — so a depth
        # exactly on a boundary two stacked layers share receives both
        # layers' contributions.
        for layer in self.layers:
            in_layer = (z >= layer.z_top_m) & (z <= layer.z_bottom_m)
            denom = (1.0 - layer.f0_hz ** 2 / f ** 2) ** 2 + 1.0 / layer.Q ** 2
            a_km[in_layer] += layer.a0 / denom
        return a_km / 1000.0


@dataclass
class ConstantAbsorption(Absorption):
    """Frequency-independent baseline absorption written into the SSP
    block's ``alphaI`` column at every depth (dB/wavelength).

    Parameters
    ----------
    value_db_per_wavelength : float
        Absorption coefficient (dB/wavelength). Non-negative.

    Notes
    -----
    dB/wavelength is the unit the deck carries, so the value written to the
    ``alphaI`` column is exact and the divergence is only in
    :meth:`alpha_db_per_m`. That accessor holds no SSP, so it converts to
    dB/m at :data:`~uacpy.core.constants.DEFAULT_SOUND_SPEED`, while the
    solver converts at each SSP row's own ``c``
    (``misc/AttenMod.f90:73``, the ``'W'`` branch: ``alphaT = alpha * freq /
    (8.6858896 * c)``). Over sound speeds of 1450-1550 m/s that is a spread
    of ±3.3 % between the two answers. Pass the SSP's own sound speed to
    :func:`convert_attenuation_units` to reproduce the deck's number
    exactly; see
    :meth:`uacpy.core.results.modes.Modes.with_attenuation` for where the
    difference is felt.
    """
    value_db_per_wavelength: float = 0.0

    def __post_init__(self):
        Absorption.__post_init__(self)
        if not (self.value_db_per_wavelength >= 0):
            raise ConfigurationError(
                f"ConstantAbsorption.value_db_per_wavelength must be "
                f"non-negative; got {self.value_db_per_wavelength}."
            )
        _require_attenuation_in_range(
            self.value_db_per_wavelength,
            "ConstantAbsorption.value_db_per_wavelength")

    def topopt_code(self) -> str:
        return ' '

    def _alpha_db_per_m(
        self,
        frequency: float,
        depths: _ArrayLike,
    ) -> np.ndarray:
        depths = np.atleast_1d(np.asarray(depths, dtype=float))
        # dB/wavelength → dB/m at this frequency (flat in depth). No SSP is
        # carried here, so the conversion uses the reference sound speed.
        alpha = float(convert_attenuation_units(
            self.value_db_per_wavelength, frequency,
            'dB/wavelength', 'dB/m',
        ))
        return np.full(depths.shape, alpha)
