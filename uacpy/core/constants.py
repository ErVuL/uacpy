"""
Constants and enums shared across UACPY.

Centralizes magic numbers, boundary/SSP codes, and the string/enum
conversions used by writers and model wrappers.
"""

from enum import Enum


DEFAULT_SOUND_SPEED = 1500.0  # m/s — typical ocean value
TL_MAX_DB = 200.0             # dB — deep-shadow-zone TL clamp

# Phase-speed search bounds used by AT-family writers when the user
# doesn't pass an explicit (c_low, c_high).
#
# ``C_LOW_FACTOR`` is the wavenumber-integration default: Scooter / SPARC
# pin ``k_max = omega/c_low`` so a positive floor is required (c_low=0
# blows up the integral). 0.95·c_min is the canonical conservative
# choice.
#
# ``C_LOW_FACTOR_KRAKEN`` is the modal-solver default. KRAKEN's c_low is
# the slowest phase speed in the mode search; setting it to 0 captures
# Scholte / interfacial modes per the KRAKEN manual.
C_LOW_FACTOR = 0.95
C_LOW_FACTOR_KRAKEN = 0.0
C_HIGH_FACTOR = 1.05

DEFAULT_C_MIN = 1400.0   # below slowest expected water-column speed
DEFAULT_C_MAX = 10000.0  # above fastest expected compressional speed

# Floor applied whenever we take 20*log10(|p|).
PRESSURE_FLOOR = 1e-30

# Broadband-mode auto-generated frequency grid: when the user runs a
# broadband-capable wrapper (Bellhop, Scooter, KrakenField, RAM, OASP)
# without an explicit ``frequencies=`` override, the wrapper picks
# ``N`` bins linearly spaced over ``[fc·(1 - BW/2), fc·(1 + BW/2)]``
# (clipped to [1, ∞)) where ``fc = source.frequencies[0]``.
# Default BW=0.5 — Bellhop User Guide §9 recommends sub-banding for
# wide bandwidths because arrivals are computed at a single fc.
DEFAULT_BROADBAND_N_FREQS = 128
DEFAULT_BROADBAND_BANDWIDTH_FACTOR = 0.5


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
        if value_lower in ['halfspace', 'elastic', 'half-space', 'a']:
            return cls.HALF_SPACE
        if value_lower in ['grain-size', 'grainsize', 'grain_size', 'g']:
            return cls.GRAIN_SIZE

        try:
            return cls[value.upper().replace('-', '_')]
        except KeyError:
            for bt in cls:
                if bt.value == value_lower:
                    return bt
            raise ValueError(f"invalid boundary type: {value!r}")

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
            raise ValueError(
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
            raise ValueError(f"invalid attenuation unit: {value!r}")

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
