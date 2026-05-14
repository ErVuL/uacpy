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

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Union

from uacpy.core.exceptions import ConfigurationError


_ArrayLike = Union[float, np.ndarray]


# ─────────────────────────────────────────────────────────────────────────────
# Bare numeric formulas
# ─────────────────────────────────────────────────────────────────────────────


def thorp_db_per_km(frequency: _ArrayLike) -> np.ndarray:
    """Thorp seawater volume attenuation in dB/km.

    Uses the JKPS Eq. 1.34 coefficients, which match the AT
    ``AttenMod.f90:94`` formula used internally by the Acoustics-Toolbox
    binaries.

    Parameters
    ----------
    frequency : float or array
        Frequency in Hz.

    References
    ----------
    Thorp, W. H. (1967). JASA 42(1), 270 (original).
    Jensen, Kuperman, Porter, Schmidt — *Computational Ocean
    Acoustics*, 2nd ed., Eq. 1.34.
    """
    f = np.atleast_1d(np.asarray(frequency, dtype=float)) / 1000.0
    f2 = f * f
    a = (
        3.3e-3
        + 0.11 * f2 / (1.0 + f2)
        + 44.0 * f2 / (4100.0 + f2)
        + 3.0e-4 * f2
    )
    return np.squeeze(a)


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

    Notes
    -----
    Implementation follows the Acoustics Toolbox ``AttenMod.f90``.

    References
    ----------
    Francois & Garrison (1982). JASA 72(6), 1879–1890.
    """
    f = np.atleast_1d(np.asarray(frequency, dtype=float)) / 1000.0
    T = np.asarray(temperature, dtype=float)
    S = np.asarray(salinity, dtype=float)
    z = np.asarray(depth, dtype=float)

    c = 1412.0 + 3.21 * T + 1.19 * S + 0.0167 * z

    A1 = 8.86 / c * 10.0 ** (0.78 * pH - 5.0)
    P1 = 1.0
    f1 = 2.8 * np.sqrt(S / 35.0) * 10.0 ** (4.0 - 1245.0 / (T + 273.0))

    A2 = 21.44 * S / c * (1.0 + 0.025 * T)
    P2 = 1.0 - 1.37e-4 * z + 6.2e-9 * z * z
    f2 = 8.17 * 10.0 ** (8.0 - 1990.0 / (T + 273.0)) / (1.0 + 0.0018 * (S - 35.0))

    P3 = 1.0 - 3.83e-5 * z + 4.9e-10 * z * z
    A3_cold = 4.937e-4 - 2.59e-5 * T + 9.11e-7 * T * T - 1.5e-8 * T * T * T
    A3_warm = 3.964e-4 - 1.146e-5 * T + 1.45e-7 * T * T - 6.5e-10 * T * T * T
    A3 = np.where(T < 20.0, A3_cold, A3_warm)

    a = (
        A1 * P1 * (f1 * f * f) / (f1 * f1 + f * f)
        + A2 * P2 * (f2 * f * f) / (f2 * f2 + f * f)
        + A3 * P3 * f * f
    )
    return np.squeeze(a)


def convert_attenuation_units(
    alpha: _ArrayLike,
    frequency: float,
    from_unit: str,
    to_unit: str,
    sound_speed: float = 1500.0,
) -> np.ndarray:
    """Convert volume attenuation between unit conventions.

    Supported units: ``dB/km``, ``dB/m``, ``dB/wavelength``, ``Nepers/m``,
    ``Q`` (quality factor), ``L`` (loss tangent). ``sound_speed`` is
    required for the wavelength / Q / L paths.

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
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))

    if from_unit == 'dB/km':
        alpha_db_m = alpha / 1000.0
    elif from_unit == 'dB/m':
        alpha_db_m = alpha
    elif from_unit == 'dB/wavelength':
        wavelength = sound_speed / frequency
        alpha_db_m = alpha / wavelength
    elif from_unit == 'Nepers/m':
        alpha_db_m = alpha * 8.686
    elif from_unit == 'Q':
        # alphaT = omega / (2 * c * Q)
        alpha_nepers_m = np.pi * frequency / (alpha * sound_speed)
        alpha_db_m = alpha_nepers_m * 8.686
    elif from_unit == 'L':
        # alphaT = L * omega / c
        alpha_nepers_m = alpha * 2.0 * np.pi * frequency / sound_speed
        alpha_db_m = alpha_nepers_m * 8.686
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
        result = alpha_db_m / 8.686
    elif to_unit == 'Q':
        alpha_nepers_m = alpha_db_m / 8.686
        result = np.pi * frequency / (alpha_nepers_m * sound_speed)
    elif to_unit == 'L':
        alpha_nepers_m = alpha_db_m / 8.686
        result = alpha_nepers_m * sound_speed / (2.0 * np.pi * frequency)
    else:
        raise ConfigurationError(f"Unknown unit: {to_unit}")

    return np.squeeze(result)


# ─────────────────────────────────────────────────────────────────────────────
# Env-attachable Absorption hierarchy
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Absorption:
    """Abstract base for water-column absorption models. Do not
    instantiate directly — pick one of :class:`Thorp`,
    :class:`FrancoisGarrison`, :class:`Biological`,
    :class:`ConstantAbsorption`.

    Subclasses implement :meth:`alpha_db_per_m`, which evaluates
    ``α(f, z)`` at the depths a model needs (used by the Kraken-class
    modal perturbation kernel; the Acoustics-Toolbox writers read
    :meth:`topopt_code` and the per-class fields instead).
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
            Frequency in Hz.
        depths : float or 1-D array
            Depths (m).

        Returns
        -------
        ndarray, same shape as ``depths``
        """
        raise NotImplementedError


@dataclass
class Thorp(Absorption):
    """Thorp (1967) seawater volume attenuation. No parameters.

    Frequency-only — α(f, z) is constant in depth.
    """

    def topopt_code(self) -> str:
        return 'T'

    def alpha_db_per_m(
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
    """
    temperature_c: float
    salinity_psu: float
    pH: float
    z_bar_m: float

    def topopt_code(self) -> str:
        return 'F'

    def as_at_tuple(self) -> Tuple[float, float, float, float]:
        """Tuple in the order the AT ``write_fg_params`` writer expects."""
        return (
            float(self.temperature_c), float(self.salinity_psu),
            float(self.pH), float(self.z_bar_m),
        )

    def alpha_db_per_m(
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
        return np.atleast_1d(a_km) / 1000.0


@dataclass
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
        Peak absorption coefficient at resonance (dB/m).
    """
    z_top_m: float
    z_bottom_m: float
    f0_hz: float
    Q: float
    a0: float

    def __post_init__(self):
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
                f"BiologicalLayer: a0 must be positive (dB/m); got {self.a0}"
            )


@dataclass
class Biological(Absorption):
    """Layered biological volume attenuation (fish-bladder resonance).

    Parameters
    ----------
    layers : list of BiologicalLayer or tuples
        Each entry can be a :class:`BiologicalLayer` or a 5-tuple
        ``(z_top, z_bottom, f0, Q, a0)``.
    """
    layers: List[BiologicalLayer] = field(default_factory=list)

    def __post_init__(self):
        Absorption.__post_init__(self)
        normalized: List[BiologicalLayer] = []
        for entry in self.layers:
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


@dataclass
class ConstantAbsorption(Absorption):
    """Frequency-independent baseline absorption written into the SSP
    block's ``alphaI`` column at every depth (dB/wavelength).

    Parameters
    ----------
    value_db_per_wavelength : float
        Absorption coefficient (dB/wavelength). Non-negative.
    """
    value_db_per_wavelength: float = 0.0

    def __post_init__(self):
        Absorption.__post_init__(self)
        if not (self.value_db_per_wavelength >= 0):
            raise ConfigurationError(
                f"ConstantAbsorption.value_db_per_wavelength must be "
                f"non-negative; got {self.value_db_per_wavelength}."
            )

    def topopt_code(self) -> str:
        return ' '
