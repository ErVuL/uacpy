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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Absorption:
    """Abstract base for water-column absorption models. Do not
    instantiate directly — pick one of :class:`Thorp`,
    :class:`FrancoisGarrison`, :class:`Biological`,
    :class:`ConstantAbsorption`."""

    def __post_init__(self):
        if type(self) is Absorption:
            raise TypeError(
                "Absorption is abstract; instantiate Thorp / "
                "FrancoisGarrison / Biological / ConstantAbsorption."
            )

    def topopt_code(self) -> str:
        """Single Acoustics-Toolbox character for ``TopOpt`` position 4."""
        raise NotImplementedError


@dataclass
class Thorp(Absorption):
    """Thorp (1967) seawater volume attenuation. No parameters."""

    def topopt_code(self) -> str:
        return 'T'


@dataclass
class FrancoisGarrison(Absorption):
    """Francois–Garrison (1982) seawater absorption.

    Parameters
    ----------
    temperature_c : float
        Water temperature (°C).
    salinity_psu : float
        Salinity (PSU).
    pH : float
        Acidity (pH).
    z_bar_m : float
        Mean propagation depth (m).
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
        # Mirror SedimentLayer.__post_init__ shape: surface area first,
        # then layer geometry.
        if self.z_bottom_m <= self.z_top_m:
            raise ValueError(
                "BiologicalLayer: z_bottom_m must be strictly greater than "
                f"z_top_m (got z_top_m={self.z_top_m}, "
                f"z_bottom_m={self.z_bottom_m})"
            )
        # Layer thickness positive (a free consequence of the bound check
        # above, but stated for symmetry with SedimentLayer's docstring).
        if (self.z_bottom_m - self.z_top_m) <= 0:
            raise ValueError(
                "BiologicalLayer: layer thickness must be positive (m); "
                f"got {self.z_bottom_m - self.z_top_m}"
            )
        if self.f0_hz <= 0:
            raise ValueError(
                f"BiologicalLayer: f0_hz must be positive (Hz); got {self.f0_hz}"
            )
        if self.Q <= 0:
            raise ValueError(
                f"BiologicalLayer: Q must be positive (dimensionless); got {self.Q}"
            )
        if self.a0 <= 0:
            raise ValueError(
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
            raise ValueError(
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

    Use when you want a calibrated ad-hoc absorption that doesn't match
    any of the standard formulas. AT models still accept a formula on
    top (``Thorp`` etc.) via the env-level choice — pick one.

    Parameters
    ----------
    value_db_per_wavelength : float
        Absorption coefficient (dB/wavelength). Non-negative.
    """
    value_db_per_wavelength: float = 0.0

    def __post_init__(self):
        Absorption.__post_init__(self)
        if not (self.value_db_per_wavelength >= 0):
            raise ValueError(
                f"ConstantAbsorption.value_db_per_wavelength must be "
                f"non-negative; got {self.value_db_per_wavelength}."
            )

    def topopt_code(self) -> str:
        return ' '
