"""
Constants, enums, and configuration values for UACPY

Centralizes magic strings and numbers used throughout the codebase.
"""

from enum import Enum
from typing import Dict


# ==============================================================================
# MAGIC NUMBERS - Physical and Computational Constants
# ==============================================================================

# Sound speed
DEFAULT_SOUND_SPEED = 1500.0  # m/s - typical ocean sound speed

# Transmission loss
TL_MAX_DB = 200.0  # dB - Maximum TL (deep shadow zone clamp)

# Phase speed search bounds (fraction of c_min / c_max for mode finding)
C_LOW_FACTOR = 0.95   # c_low = c_min * C_LOW_FACTOR
C_HIGH_FACTOR = 1.05  # c_high = c_max * C_HIGH_FACTOR

# Default phase speed bounds for reflection coefficient computations
DEFAULT_C_MIN = 1400.0  # m/s - below slowest expected water column speed
DEFAULT_C_MAX = 10000.0  # m/s - above fastest expected compressional speed


# ==============================================================================
# PRESSURE FLOOR
# ==============================================================================

# Single floor used whenever we take 20*log10(|p|).
PRESSURE_FLOOR = 1e-30


# ==============================================================================
# SOUND SPEED PROFILE ENUMS
# ==============================================================================

class SSPType(Enum):
    """
    Sound speed profile types and interpolation methods

    Maps user-facing names to internal representations.
    """
    # Profile types
    ISOVELOCITY = 'isovelocity'
    LINEAR = 'linear'
    BILINEAR = 'bilinear'
    MUNK = 'munk'

    # Interpolation methods
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
        """Convert string to SSPType enum"""
        if isinstance(value, SSPType):
            return value
        try:
            return cls[value.upper().replace('-', '_')]
        except KeyError:
            # Try value match
            for ssp in cls:
                if ssp.value == value.lower():
                    return ssp
            raise ValueError(
                f"Invalid SSP type: {value}. "
                f"Valid options: {[s.value for s in cls]}"
            )

    def to_acoustics_toolbox_code(self) -> str:
        """
        Convert to Acoustics Toolbox SSP interpolation code

        Returns
        -------
        str
            Single character code: 'N', 'C', 'P', 'S', 'Q', or 'A'
        """
        return AT_SSP_CODE_MAP[self]


# Acoustics Toolbox SSP interpolation codes
AT_SSP_CODE_MAP: Dict[SSPType, str] = {
    # Profile types default to C-Linear
    SSPType.ISOVELOCITY: 'C',
    SSPType.LINEAR: 'C',
    SSPType.BILINEAR: 'C',
    SSPType.MUNK: 'C',
    # Interpolation methods map to AT codes
    SSPType.N2LINEAR: 'N',
    SSPType.C_LINEAR: 'C',
    SSPType.CLIN: 'C',
    SSPType.PCHIP: 'P',
    SSPType.CUBIC: 'S',
    SSPType.SPLINE: 'S',
    SSPType.QUADRATIC: 'Q',
    SSPType.ANALYTIC: 'A',
}


# ==============================================================================
# BOUNDARY CONDITION ENUMS
# ==============================================================================

class BoundaryType(Enum):
    """Acoustic boundary types"""
    VACUUM = 'vacuum'           # Pressure-release (free surface)
    RIGID = 'rigid'             # Rigid boundary
    HALF_SPACE = 'half-space'   # Acousto-elastic half-space
    GRAIN_SIZE = 'grain-size'   # Sediment parameters derived from grain size (phi)
    FILE = 'file'               # Reflection coefficients from file
    PRECALC = 'precalc'         # Pre-calculated reflection data

    @classmethod
    def from_string(cls, value: str) -> 'BoundaryType':
        """Convert string to BoundaryType enum, handling aliases"""
        if isinstance(value, BoundaryType):
            return value

        # Normalize aliases to canonical forms
        value_lower = value.lower()
        if value_lower in ['halfspace', 'elastic', 'half-space', 'a']:
            return cls.HALF_SPACE
        if value_lower in ['grain-size', 'grainsize', 'grain_size', 'g']:
            return cls.GRAIN_SIZE

        # Try standard conversion
        try:
            return cls[value.upper().replace('-', '_')]
        except KeyError:
            for bt in cls:
                if bt.value == value_lower:
                    return bt
            raise ValueError(f"Invalid boundary type: {value}")

    def to_acoustics_toolbox_code(self) -> str:
        """
        Convert to Acoustics Toolbox boundary code

        Returns
        -------
        str
            Single character: 'V', 'R', 'A', 'G', 'F', or 'P'
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


# ==============================================================================
# ATTENUATION UNITS ENUM
# ==============================================================================

class AttenuationUnits(Enum):
    """Attenuation units for Acoustics Toolbox"""
    DB_PER_WAVELENGTH = 'W'     # dB/wavelength (default)
    NEPERS_PER_M = 'N'          # Nepers/m
    DB_PER_M_KHZ = 'F'          # dB/(m*kHz)
    DB_PER_M = 'M'              # dB/m
    Q_FACTOR = 'Q'              # Q factor
    LOSS_PARAMETER = 'L'        # Loss parameter

    @classmethod
    def from_string(cls, value: str) -> 'AttenuationUnits':
        """Convert string to AttenuationUnits enum"""
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
        return self.value


class VolumeAttenuation(Enum):
    """Volume attenuation formulas"""
    NONE = ''                   # No volume attenuation
    THORP = 'T'                 # Thorp formula
    FRANCOIS_GARRISON = 'F'     # Francois-Garrison formula
    BIOLOGICAL = 'B'            # Biological attenuation

    @classmethod
    def from_string(cls, value: str) -> 'VolumeAttenuation':
        """Convert string to VolumeAttenuation enum"""
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
        return self.value


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def parse_ssp_type(value) -> SSPType:
    """Parse SSP interpolation type string, defaulting to ISOVELOCITY if None.

    Parameters
    ----------
    value : str or None
        SSP type string (e.g., 'c_linear', 'n2_linear') or None for default.

    Returns
    -------
    SSPType
        Parsed SSP type enum value.
    """
    if value is None:
        return SSPType.ISOVELOCITY
    return SSPType.from_string(value)


def parse_boundary_type(value) -> BoundaryType:
    """Parse boundary type string, defaulting to VACUUM if None.

    Parameters
    ----------
    value : str or None
        Boundary type string (e.g., 'vacuum', 'rigid', 'halfspace') or None
        for default.

    Returns
    -------
    BoundaryType
        Parsed boundary type enum value.
    """
    if value is None:
        return BoundaryType.VACUUM
    return BoundaryType.from_string(value)
