"""
Constants and enums shared across UACPY.

Centralizes magic numbers, boundary/SSP codes, and the string/enum
conversions used by writers and model wrappers.
"""

import math
from enum import Enum

from uacpy.core.exceptions import ConfigurationError


DEFAULT_SOUND_SPEED = 1500.0  # m/s — typical ocean value

# dB — the level a wrapper reports for a cell carrying no energy: RAM pins both
# a diverged Padé sample and its synthetic pressure-release surface row here,
# and converts back to a magnitude with ``10**(-TL_MAX_DB/20)``.
TL_MAX_DB = 200.0

# Mean Earth radius (IUGG R1) for spherical great-circle geodesy when
# sampling external geographic datasets (bathymetry transects, …).
EARTH_RADIUS_M = 6_371_008.8  # m

# Every AT solver converts attenuation through ``misc/AttenMod.f90``'s ``CRCI``,
# and uacpy writes ``AttenUnit = 'W'`` (dB/wavelength) in every deck. Substituting
# that branch (:73, ``alphaT = alpha*freq/(8.6858896*c)``) into the conversion to an
# imaginary sound speed (:113, ``alphaT = alphaT*c*c/omega``) with
# ``omega = 2*pi*freq`` leaves ``alphaT_imag = alpha*c/(8.6858896*2*pi)``, so the
# fatal test at :116 (``alphaT > c``) reduces to a bound on alpha alone —
# independent of frequency and of sound speed. CRCI is reached for every water
# sample (``misc/sspMod.f90`` UpdateSSPLoss), both half-spaces (UpdateHSLoss) and
# Bellhop's own (``ReadEnvironmentBell.f90``), so the bound is package-wide.
MAX_ATTENUATION_DB_PER_WAVELENGTH = 8.6858896 * 2.0 * math.pi   # 54.575

# ``misc/sspMod.f90:353`` ends a medium's SSP block at the first sample within
# ``100*EPSILON(1.0e0)`` of the declared medium depth — an absolute tolerance in
# metres, single precision, not scaled by depth. A second sample inside that window
# is never read as an SSP row: the next READ consumes it as the bottom-option
# record (``misc/ReadEnvironmentMod.f90``), so the boundary condition is taken from
# a sound speed.
AT_LAST_SSP_POINT_EPS_M = 100.0 * 2.0 ** -23          # 1.1920929e-05 m

# Resolution at which the AT decks print their axes: depths in metres and ranges in
# kilometres, both at ``%.6f``. Two samples closer than this collapse to one token.
# On a range axis the readers then reject it outright — ``misc/sspMod.f90:342``,
# ``Bellhop/bdryMod.f90:132``/``:231``, ``misc/SourceReceiverPositions.f90:163``,
# with ``misc/monotonicMod.f90`` strict (``<=`` fails). On a source/receiver *depth*
# axis the equivalent ERROUTs are commented out
# (``SourceReceiverPositions.f90:142``/``:146``), so the collapse is silent instead:
# the deck simply carries the same depth twice.
#: Decimals each axis is printed at. One number, so the carriers' admission
#: rule below and the writers' column format (``io/oalib_writer.py``, which
#: imports these) are the same statement rather than two that agree today.
DECK_AXIS_DECIMALS = 6
DECK_DEPTH_RESOLUTION_M = 10.0 ** -DECK_AXIS_DECIMALS            # metres at %.6f
DECK_RANGE_RESOLUTION_M = 1.0e3 * 10.0 ** -DECK_AXIS_DECIMALS    # km at %.6f

# The ``.sbp`` source beam pattern writes its angle column at ``%12.6f``
# (``io/refl_io.py``), so two angles closer than this land on the same token.
# Bellhop's load-time guard (``misc/beampattern.f90:56`` via
# ``misc/monotonicMod.f90``) is strict, so the collapsed pair aborts the run
# with ERROUT rather than degrading silently.
SBP_ANGLE_RESOLUTION_DEG = 1.0e-6

# Phase-speed search bounds used by AT-family writers when the user
# doesn't pass an explicit (c_low, c_high).
#
# ``C_LOW_FACTOR`` is the wavenumber-integration default: Scooter / SPARC
# pin ``k_max = omega/c_low`` so a positive floor is required (c_low=0
# blows up the integral). 0.95·c_min is the canonical conservative
# choice.
#
# ``C_HIGH_FACTOR`` pads the upper bound symmetrically: the writers take
# ``c_high = 1.05 · max(c_max, bottom cp)`` so the fastest speed in the problem
# sits strictly inside the interval rather than on its edge.
# The pad is REQUIRED by the integration models and merely TOLERATED by the
# modal one, so it cannot be tuned for either alone: for Scooter
# (``scooter.f90:67,123``) and SPARC (``SPARC._write_sparc_env``'s Nk) c_high is
# the lower limit of a wavenumber INTEGRAL, and padding past the bottom speed
# keeps the branch point k = omega/c_bottom strictly inside the window — which
# is what lets Scooter recover the lateral wave that makes it the right model
# below the modal cutoff. Kraken only searches [c_low, c_high] for roots, so
# there the same pad means a default run returns a few modes with a phase speed
# above the bottom speed (``models/kraken.py`` logs the count and refuses when
# every mode is one of them).
# (KRAKEN's modal-solver c_low default is the literal 0.0 written at its use
# site in ``models/kraken.py`` — 0 makes KRAKEN compute the bound itself, per
# kraken.htm, Phase Speed Limits.)
C_LOW_FACTOR = 0.95
C_HIGH_FACTOR = 1.05

DEFAULT_C_MIN = 1400.0   # below slowest expected water-column speed

# "No upper phase-speed limit" for the AT family. A vacuum / rigid boundary
# traps every mode, so the mode search must not be capped on a half-space
# speed that does not exist. Acoustics-Toolbox/doc/bounce.htm prescribes the
# value: "For a full 90 degree calculation set CMin to the lowest speed in the
# problem (say 1400.0) CMax to 1.0E9." Kraken's ``leaky_modes`` uses the same
# number.
DEFAULT_C_MAX_UNBOUNDED = 1.0e9

# Sea-ice canopy as a homogeneous elastic surface. Canonical Arctic pack-ice
# values from Jensen, Kuperman, Porter & Schmidt, *Computational Ocean
# Acoustics* (the ice cover modelled as a homogeneous elastic medium): cp 3500
# m/s, cs 1800 m/s, αp 0.4 dB/λ, αs 1.0 dB/λ ("realistic attenuations of
# 0.4 dB/λ for compressional waves and 1.0 dB/λ for shear waves"). Typical
# ranges (Etter, *Underwater Acoustic Modeling*): cp 1300-3900, cs 1400-1900
# m/s.
SEA_ICE_COMPRESSIONAL_SPEED = 3500.0       # m/s
SEA_ICE_SHEAR_SPEED = 1800.0               # m/s
SEA_ICE_DENSITY = 0.9                      # g/cm³
SEA_ICE_COMPRESSIONAL_ATTENUATION = 0.4    # dB/wavelength
SEA_ICE_SHEAR_ATTENUATION = 1.0            # dB/wavelength
# NSIDC standard ice-edge definition: ≥15 % concentration counts as ice-covered.
SEA_ICE_EDGE_CONCENTRATION = 0.15

# Floor applied whenever we take 20*log10(|p|). 1e-30 lands at 600 dB of loss,
# three times past the ``TL_MAX_DB`` no-energy sentinel, so a floored cell and a
# clamped one stay distinguishable in the output.
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
    FILE = 'file'               # reflection coefficients from file
    PRECALC = 'precalc'         # pre-calculated reflection data
    # NB: there is no grain-size type — a grain size is converted to an explicit
    # half-space at construction (BoundaryProperties.from_grain_size).

    @classmethod
    def from_string(cls, value: str) -> 'BoundaryType':
        """
        Parse a string (or existing ``BoundaryType``) into a ``BoundaryType``.

        Resolves common aliases such as 'halfspace' and 'elastic'.

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
        if not isinstance(value, str):
            raise ConfigurationError(
                f"invalid boundary type: expected a string or BoundaryType; "
                f"got {type(value).__name__}: {value!r}",
                remediation=f"Use one of {[bt.value for bt in cls]}.")

        value_lower = value.lower()
        if value_lower in ['halfspace', 'elastic', 'half-space']:
            return cls.HALF_SPACE
        # Single-letter Acoustics-Toolbox codes — the inverse of
        # ``to_acoustics_toolbox_code``.
        at_codes = {'v': cls.VACUUM, 'r': cls.RIGID, 'a': cls.HALF_SPACE,
                    'f': cls.FILE, 'p': cls.PRECALC}
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

        The letters are AT's ``TOPOPT(2:2)`` / ``BOTOPT(1:1)`` alphabet
        (``doc/EnvironmentalFile.htm``): 'V' VACUUM, 'A' ACOUSTO-ELASTIC
        half-space, 'R' perfectly RIGID, 'F' reflection coefficient from a
        FILE. ``'P'`` is NOT in that HTML list but is real — the Fortran
        implements it as ``CASE ( 'P' ) ! Precalculated reflection coef``
        (``Kraken/BCImpedanceMod.f90:118``, ``BCImpedancecMod.f90:105``), so
        the source is the authority here rather than the manual.

        AT's remaining bottom letter, ``'G'`` (grain size), is deliberately
        never emitted: uacpy converts a grain size to explicit geoacoustics in
        Python (:func:`uacpy.core.sediment.grain_size_to_geoacoustics`, whose
        APL-UW polynomials reproduce ``Bellhop/ReadEnvironmentBell.f90:497-520``
        to 0.0e+00) and writes the resulting 'A' half-space, so one code path
        serves every engine instead of only the ones that read 'G'.

        Returns
        -------
        str
            One of 'V', 'R', 'A', 'F', or 'P'.
        """
        mapping = {
            BoundaryType.VACUUM: 'V',
            BoundaryType.RIGID: 'R',
            BoundaryType.HALF_SPACE: 'A',
            BoundaryType.FILE: 'F',
            BoundaryType.PRECALC: 'P',
        }
        return mapping[self]


class AttenuationUnits(Enum):
    """Attenuation units understood by the Acoustics Toolbox.

    ``TOPOPT(3:3)``. The names follow the Fortran's own comments in
    ``misc/AttenMod.f90:66-80``; ``doc/EnvironmentalFile.htm`` writes ``'F'``
    as "dB/(kmHz)" where the Fortran writes "dB/(m kHz)" — the SAME unit, not
    two conventions, since ``alpha*f_Hz*r_km`` and ``alpha*f_kHz*r_m`` differ
    by 1000 in both numerator and denominator. The member name follows the
    source.

    The manual's lowercase ``'m'`` (dB/m with power-law frequency scaling, its
    beta and fT given per layer) has no member because no uacpy writer emits
    it: ``write_at_top_block`` hardwires ``TOPOPT(3:3)='W'``, which is the
    package's documented attenuation convention everywhere.

    That hardwiring is why nothing in uacpy consumes an ``AttenuationUnits``:
    the enum names the vocabulary of the ``.env`` format for a caller reading
    or writing decks directly, and :meth:`to_char` is the only member any
    uacpy writer calls.
    """
    DB_PER_WAVELENGTH = 'W'     # dB/wavelength (default; uacpy always writes this)
    NEPERS_PER_M = 'N'          # Nepers/m
    DB_PER_M_KHZ = 'F'          # dB/(m·kHz) == the manual's dB/(km·Hz)
    DB_PER_M = 'M'              # dB/m
    Q_FACTOR = 'Q'              # Q factor
    LOSS_PARAMETER = 'L'        # Loss parameter (a.k.a. loss tangent)

    @classmethod
    def from_string(cls, value: str) -> 'AttenuationUnits':
        """
        Parse a string (or existing enum) into an ``AttenuationUnits``.

        For callers reading a ``TOPOPT`` letter off a third-party deck or
        taking one from configuration. No uacpy API takes an attenuation unit
        as an argument — every writer hardwires ``'W'`` — so nothing in the
        package calls this, and the string vocabulary the *conversion* helper
        :func:`~uacpy.core.absorption.convert_attenuation_units` speaks
        (``'dB/km'``, ``'Nepers/m'``, ``'dB/wavelength'``, ``'Q'``, ``'L'``,
        ``'dB/m'``) is a different one, not these letters.

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
        if not isinstance(value, str):
            raise ConfigurationError(
                f"invalid attenuation unit: expected a string or "
                f"AttenuationUnits; got {type(value).__name__}: {value!r}",
                remediation="Use one of 'W', 'N', 'F', 'M', 'Q', 'L'.")
        # Case is the whole difference between two AT units, and the lookup
        # below upper-cases, so 'm' has to be caught before it silently
        # becomes 'M'.
        if value == 'm':
            raise ConfigurationError(
                "attenuation_unit 'm' (dB/m with power-law BETA/fT) is "
                "distinct from 'M' (dB/m). The 'm' variant has no enum "
                "member, because its beta and fT are per-layer deck fields "
                "this enum cannot carry; use 'M' for plain dB/m or one of "
                "'N', 'F', 'W', 'Q', 'L'"
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
