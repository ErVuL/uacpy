"""
Constants and enums shared across UACPY.

Centralizes magic numbers, boundary/SSP codes, and the string/enum
conversions used by writers and model wrappers.
"""

from enum import Enum
from typing import Dict


DEFAULT_SOUND_SPEED = 1500.0  # m/s — typical ocean value
TL_MAX_DB = 200.0             # dB — deep-shadow-zone TL clamp

# Phase-speed search bounds (fractions of c_min / c_max for mode finding).
C_LOW_FACTOR = 0.95
C_HIGH_FACTOR = 1.05

DEFAULT_C_MIN = 1400.0   # below slowest expected water-column speed
DEFAULT_C_MAX = 10000.0  # above fastest expected compressional speed

# Floor applied whenever we take 20*log10(|p|).
PRESSURE_FLOOR = 1e-30


class SSPType(Enum):
    """
    Sound speed profile types and interpolation methods.

    Maps user-facing names to internal representations used by environment
    writers.
    """
    ISOVELOCITY = 'isovelocity'
    LINEAR = 'linear'
    BILINEAR = 'bilinear'
    MUNK = 'munk'

    N2LINEAR = 'n2linear'       # N²-linear approximation
    C_LINEAR = 'c-linear'       # C-linear approximation
    CLIN = 'clin'               # C-linear (alias)
    PCHIP = 'pchip'             # PCHIP cubic spline
    CUBIC = 'cubic'             # Cubic spline
    SPLINE = 'spline'           # Spline (alias)
    QUADRATIC = 'quad'          # Quadratic interpolation
    ANALYTIC = 'analytic'       # Analytic profile

    @classmethod
    def from_string(cls, value: str) -> 'SSPType':
        """
        Parse a string (or existing ``SSPType``) into an ``SSPType``.

        Parameters
        ----------
        value : str or SSPType
            Case-insensitive type name or enum value.

        Returns
        -------
        SSPType
            Parsed enum value.
        """
        if isinstance(value, SSPType):
            return value
        try:
            return cls[value.upper().replace('-', '_')]
        except KeyError:
            for ssp in cls:
                if ssp.value == value.lower():
                    return ssp
            raise ValueError(
                f"Invalid SSP type: {value}. "
                f"Valid options: {[s.value for s in cls]}"
            )

    def to_acoustics_toolbox_code(self) -> str:
        """
        Return the single-character Acoustics Toolbox SSP code.

        Returns
        -------
        str
            One of 'N', 'C', 'P', 'S', 'Q', or 'A'.
        """
        return AT_SSP_CODE_MAP[self]


# Profile types default to C-linear; interpolation methods map to their AT code.
AT_SSP_CODE_MAP: Dict[SSPType, str] = {
    SSPType.ISOVELOCITY: 'C',
    SSPType.LINEAR: 'C',
    SSPType.BILINEAR: 'C',
    SSPType.MUNK: 'C',
    SSPType.N2LINEAR: 'N',
    SSPType.C_LINEAR: 'C',
    SSPType.CLIN: 'C',
    SSPType.PCHIP: 'P',
    SSPType.CUBIC: 'S',
    SSPType.SPLINE: 'S',
    SSPType.QUADRATIC: 'Q',
    SSPType.ANALYTIC: 'A',
}


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
            raise ValueError(f"Invalid boundary type: {value}")

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
        for au in cls:
            if au.value == value.upper():
                return au
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(f"Invalid attenuation unit: {value}")

    def to_char(self) -> str:
        """Return the single-character Acoustics Toolbox code."""
        return self.value


class VolumeAttenuation(Enum):
    """Volume attenuation formulas."""
    NONE = ''                   # No volume attenuation
    THORP = 'T'                 # Thorp formula
    FRANCOIS_GARRISON = 'F'     # Francois-Garrison formula
    BIOLOGICAL = 'B'            # Biological attenuation

    @classmethod
    def from_string(cls, value: str) -> 'VolumeAttenuation':
        """
        Parse a string (or existing enum) into a ``VolumeAttenuation``.

        Parameters
        ----------
        value : str, VolumeAttenuation, or None
            Case-insensitive formula name or single-character code.
            ``None`` or empty string maps to ``NONE``.

        Returns
        -------
        VolumeAttenuation
            Parsed enum value.
        """
        if value is None or value == '':
            return cls.NONE
        if isinstance(value, VolumeAttenuation):
            return value
        for va in cls:
            if va.value == value.upper():
                return va
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(f"Invalid volume attenuation: {value}")

    def to_char(self) -> str:
        """Return the single-character Acoustics Toolbox code."""
        return self.value


def parse_ssp_type(value) -> SSPType:
    """
    Parse an SSP interpolation type string.

    Parameters
    ----------
    value : str or None
        SSP type string (e.g., 'c_linear', 'n2_linear') or ``None`` for
        the default.

    Returns
    -------
    SSPType
        Parsed enum value; ``None`` maps to ``ISOVELOCITY``.
    """
    if value is None:
        return SSPType.ISOVELOCITY
    return SSPType.from_string(value)


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
