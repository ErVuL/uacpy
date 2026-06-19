"""
Constants and enums shared across UACPY.

Centralizes magic numbers, boundary/SSP codes, and the string/enum
conversions used by writers and model wrappers.
"""

import math
from enum import Enum

from uacpy.core.exceptions import ConfigurationError


DEFAULT_SOUND_SPEED = 1500.0  # m/s — typical ocean value
TL_MAX_DB = 200.0             # dB — deep-shadow-zone TL clamp

# Mean Earth radius (IUGG R1) for spherical great-circle geodesy when
# sampling external geographic datasets (bathymetry transects, …).
EARTH_RADIUS_M = 6_371_008.8  # m

# Seawater density for SI particle-velocity / acoustic-intensity physics
# (Euler relation u = -grad(p)/(i*omega*rho); intensity in W/m²). In kg/m³,
# NOT the g/cm³ used for material/sediment densities elsewhere — convert at
# the boundary. Nominal ocean value (Abraham, UW Acoustic Signal Processing).
DENSITY_SEAWATER = 1027.0  # kg/m³

# Phase-speed search bounds used by AT-family writers when the user
# doesn't pass an explicit (c_low, c_high).
#
# ``C_LOW_FACTOR`` is the wavenumber-integration default: Scooter / SPARC
# pin ``k_max = omega/c_low`` so a positive floor is required (c_low=0
# blows up the integral). 0.95·c_min is the canonical conservative
# choice.
#
# ``C_LOW_FACTOR_KRAKEN`` is the modal-solver default. KRAKEN's c_low is
# the slowest phase speed in the mode search; 0 makes KRAKEN compute it
# automatically (kraken.htm, Phase Speed Limits). A positive c_low excludes
# slow interfacial (Scholte / Stoneley) modes.
C_LOW_FACTOR = 0.95
C_LOW_FACTOR_KRAKEN = 0.0
C_HIGH_FACTOR = 1.05

DEFAULT_C_MIN = 1400.0   # below slowest expected water-column speed
DEFAULT_C_MAX = 10000.0  # above fastest expected compressional speed

# Sea-ice canopy as a homogeneous elastic surface. Canonical Arctic pack-ice
# values from Jensen, Kuperman, Porter & Schmidt, *Computational Ocean
# Acoustics* (the ice cover modelled as a homogeneous elastic medium): cp 3500
# m/s, cs 1800 m/s, ρ 900 kg/m³, αp 0.5 dB/λ, αs 1.0 dB/λ. Typical ranges
# (Etter, *Underwater Acoustic Modeling*): cp 1300-3900, cs 1400-1900 m/s.
SEA_ICE_COMPRESSIONAL_SPEED = 3500.0       # m/s
SEA_ICE_SHEAR_SPEED = 1800.0               # m/s
SEA_ICE_DENSITY = 0.9                      # g/cm³
SEA_ICE_COMPRESSIONAL_ATTENUATION = 0.5    # dB/wavelength
SEA_ICE_SHEAR_ATTENUATION = 1.0            # dB/wavelength
# NSIDC standard ice-edge definition: ≥15 % concentration counts as ice-covered.
SEA_ICE_EDGE_CONCENTRATION = 0.15

# Floor applied whenever we take 20*log10(|p|).
PRESSURE_FLOOR = 1e-30

# SPL reference pressures for dB conversion (levels are dB re ref²).
# Underwater acoustics references 1 µPa; in-air references 20 µPa.
REFERENCE_PRESSURE_WATER = 1e-6  # Pa (1 µPa)
REFERENCE_PRESSURE_AIR = 2e-5    # Pa (20 µPa)

# Broadband-mode auto-generated frequency grid: when the user runs a
# broadband-capable wrapper (Bellhop, Scooter, Kraken, RAM, OASP)
# without an explicit ``frequencies=`` override, the wrapper picks
# ``N`` bins linearly spaced over ``[fc·(1 - BW/2), fc·(1 + BW/2)]``
# (clipped to [1, ∞)) where ``fc = source.frequencies[0]``.
# Default BW=0.5 — Bellhop User Guide §9 recommends sub-banding for
# wide bandwidths because arrivals are computed at a single fc.
DEFAULT_BROADBAND_N_FREQS = 128
DEFAULT_BROADBAND_BANDWIDTH_FACTOR = 0.5

# Exact nepers → dB conversion (20/ln10 ≈ 8.6858896).
NEPER_TO_DB = 20.0 / math.log(10.0)


class BoundaryType(Enum):
    """Acoustic boundary types."""
    VACUUM = 'vacuum'           # pressure-release (free surface)
    RIGID = 'rigid'
    HALF_SPACE = 'half-space'   # acousto-elastic half-space
    GRAIN_SIZE = 'grain-size'   # sediment derived from grain size (phi)
    FILE = 'file'               # reflection coefficients from file
    PRECALC = 'precalc'         # pre-calculated reflection data

    @classmethod
    def from_string(cls, value: str) -> 'BoundaryType':
        """
        Parse a string (or existing ``BoundaryType``) into a ``BoundaryType``.

        Resolves common aliases such as 'halfspace', 'elastic', and 'grainsize'.

        Parameters
        ----------
        value : str or BoundaryType
            Case-insensitive boundary type name, alias, or enum value.

        Returns
        -------
        BoundaryType
            Parsed enum value.
        """
        if isinstance(value, BoundaryType):
            return value

        value_lower = value.lower()
        if value_lower in ['halfspace', 'elastic', 'half-space']:
            return cls.HALF_SPACE
        if value_lower in ['grain-size', 'grainsize', 'grain_size']:
            return cls.GRAIN_SIZE
        # Single-letter Acoustics-Toolbox codes — the inverse of
        # ``to_acoustics_toolbox_code``, symmetric across all six types.
        at_codes = {'v': cls.VACUUM, 'r': cls.RIGID, 'a': cls.HALF_SPACE,
                    'g': cls.GRAIN_SIZE, 'f': cls.FILE, 'p': cls.PRECALC}
        if value_lower in at_codes:
            return at_codes[value_lower]

        try:
            return cls[value.upper().replace('-', '_')]
        except KeyError:
            for bt in cls:
                if bt.value == value_lower:
                    return bt
            raise ConfigurationError(
                f"invalid boundary type: {value!r}",
                remediation=f"Use one of {[bt.value for bt in cls]}.")

    def to_acoustics_toolbox_code(self) -> str:
        """
        Return the single-character Acoustics Toolbox boundary code.

        Returns
        -------
        str
            One of 'V', 'R', 'A', 'G', 'F', or 'P'.
        """
        mapping = {
            BoundaryType.VACUUM: 'V',
            BoundaryType.RIGID: 'R',
            BoundaryType.HALF_SPACE: 'A',
            BoundaryType.GRAIN_SIZE: 'G',
            BoundaryType.FILE: 'F',
            BoundaryType.PRECALC: 'P',
        }
        return mapping[self]


class AttenuationUnits(Enum):
    """Attenuation units understood by the Acoustics Toolbox."""
    DB_PER_WAVELENGTH = 'W'     # dB/wavelength (default)
    NEPERS_PER_M = 'N'          # Nepers/m
    DB_PER_M_KHZ = 'F'          # dB/(m·kHz)
    DB_PER_M = 'M'              # dB/m
    Q_FACTOR = 'Q'              # Q factor
    LOSS_PARAMETER = 'L'        # Loss parameter

    @classmethod
    def from_string(cls, value: str) -> 'AttenuationUnits':
        """
        Parse a string (or existing enum) into an ``AttenuationUnits``.

        Parameters
        ----------
        value : str or AttenuationUnits
            Case-insensitive unit name or single-character code.

        Returns
        -------
        AttenuationUnits
            Parsed enum value.
        """
        if isinstance(value, AttenuationUnits):
            return value
        if value == 'm':
            raise ConfigurationError(
                "attenuation_unit 'm' (dB/m with power-law BETA/fT) is "
                "distinct from 'M' (dB/m). The 'm' variant is rejected by "
                "every uacpy writer and has no enum member; use 'M' for "
                "plain dB/m or one of 'N', 'F', 'W', 'Q', 'L'"
            )
        for au in cls:
            if au.value == value.upper():
                return au
        try:
            return cls[value.upper()]
        except KeyError:
            raise ConfigurationError(
                f"invalid attenuation unit: {value!r}",
                remediation="Use one of 'W', 'N', 'F', 'M', 'Q', 'L'.")

    def to_char(self) -> str:
        """Return the single-character Acoustics Toolbox code."""
        return self.value


def parse_boundary_type(value) -> BoundaryType:
    """
    Parse a boundary type string.

    Parameters
    ----------
    value : str or None
        Boundary type string (e.g., 'vacuum', 'rigid', 'halfspace') or
        ``None`` for the default.

    Returns
    -------
    BoundaryType
        Parsed enum value; ``None`` maps to ``VACUUM``.
    """
    if value is None:
        return BoundaryType.VACUUM
    return BoundaryType.from_string(value)
