"""
The Acoustics-Toolbox deck letters uacpy writes and reads.

One table per ``.env`` option field, so a writer and :func:`uacpy.io.read_env`
read the same letters: the SSP interpolation letters of ``TOPOPT(1:1)``
(:data:`SSP_INTERP_CODES`, both directions), the boundary letters of
``TOPOPT(2:2)`` / ``BOTOPT(1:1)`` (:data:`BOUNDARY_CODES`, both directions),
the attenuation units of ``TOPOPT(3:3)`` (:class:`AttenuationUnits`), the
volume attenuation of ``TOPOPT(4:4)`` with the records it reads after the
option line (:data:`VOLUME_ATTENUATION_CODES`,
:func:`francois_garrison_record`, :func:`biological_records`), and the
geometry interpolation letter of a ``.bty`` / ``.ati``
(:data:`GEOMETRY_INTERP_CODES`). The carriers hold physics only; the letters
of one engine family's decks live here.
"""

from enum import Enum
from typing import List, Tuple

from uacpy.core.absorption import (
    Biological, ConstantAbsorption, FrancoisGarrison, Thorp,
)
from uacpy.core.boundary import BoundaryType
from uacpy.core.exceptions import ConfigurationError


#: ``TOPOPT(2:2)`` / ``BOTOPT(1:1)``: the letter each boundary type is written
#: as. AT's alphabet (``doc/EnvironmentalFile.htm``): 'V' VACUUM, 'A'
#: ACOUSTO-ELASTIC half-space, 'R' perfectly RIGID, 'F' reflection coefficient
#: from a FILE. ``'P'`` is NOT in that HTML list but is real — the Fortran
#: implements it as ``CASE ( 'P' ) ! Precalculated reflection coef``
#: (``Kraken/BCImpedanceMod.f90:118``, ``BCImpedancecMod.f90:105``), so the
#: source is the authority here rather than the manual.
#:
#: AT's remaining bottom letter, ``'G'`` (grain size), is deliberately never
#: written: uacpy converts a grain size to explicit geoacoustics in Python
#: (:func:`uacpy.core.sediment.grain_size_to_geoacoustics`, whose APL-UW
#: polynomials reproduce ``Bellhop/ReadEnvironmentBell.f90:497-520`` to
#: 0.0e+00) and writes the resulting 'A' half-space, so one code path serves
#: every engine instead of only the ones that read 'G'.
BOUNDARY_CODES = {
    BoundaryType.VACUUM: 'V',
    BoundaryType.RIGID: 'R',
    BoundaryType.HALF_SPACE: 'A',
    BoundaryType.FILE: 'F',
    BoundaryType.PRECALC: 'P',
}

#: The boundary type each letter a deck may carry reads as: the inverse of
#: :data:`BOUNDARY_CODES`, plus Bellhop's grain-size ``'G'``, which reads as
#: the half-space its formulas give (``Bellhop/ReadEnvironmentBell.f90:488-530``).
BOUNDARY_TYPES_BY_CODE = {
    **{code: kind for kind, code in BOUNDARY_CODES.items()},
    'G': BoundaryType.HALF_SPACE,
}

#: ``TOPOPT(1:1)``: the SSP interpolation letter of each ``interp_ssp`` name.
#: The letters are AT's (``doc/EnvironmentalFile.htm``): 'C' C-linear, 'N'
#: N2-linear (n the index of refraction), 'P' PCHIP, 'S' cubic Spline, 'Q'
#: Quadrilateral 2D SSP read from a file (BELLHOP only). 'H' (Hexahedral 3D,
#: BELLHOP3D) and 'A' (Analytic, needs ANALYT.FOR recompiled) are
#: deliberately absent — no uacpy writer emits either.
SSP_INTERP_CODES = {
    'linear': 'C',
    'n2linear': 'N',
    'pchip': 'P',
    'spline': 'S',
    'quad': 'Q',
}

#: The ``interp_ssp`` name each ``TOPOPT(1:1)`` letter reads as: the inverse
#: of :data:`SSP_INTERP_CODES`.
SSP_INTERP_BY_CODE = {code: name for name, code in SSP_INTERP_CODES.items()}

#: ``TYPE(1:1)`` of a ``.bty`` / ``.ati`` file: the boundary-geometry
#: interpolation letter of each name — 'L' piecewise linear, 'C' curvilinear
#: (``Bellhop/bdryMod.f90``, ``doc/bellhop.htm``).
GEOMETRY_INTERP_CODES = {'linear': 'L', 'curvilinear': 'C'}

#: ``TOPOPT(4:4)``: the volume-attenuation letter of each absorption model
#: (``misc/AttenMod.f90`` ``AttenUnit(2:2)``). A :class:`ConstantAbsorption`
#: is carried in the SSP rows' ``alphaI`` column, so its letter is the blank
#: that no volume model writes, as is no absorption at all. A law with no
#: letter here goes into the ``alphaI`` column too
#: (:func:`writes_alpha_per_ssp_row`). :class:`FrancoisGarrison` is not
#: listed: AT's ``'F'`` evaluates the formula at one ``z_bar`` read from the
#: deck (``misc/AttenMod.f90:148-160``) and applies that alpha at every
#: depth, where the law's depth term takes the depth of each row, so a deck
#: at one frequency carries it in the rows, exact at every node. Only a
#: one-row law on a deck covering several frequencies is written as ``'F'``
#: (``multi_frequency=True``): the rows would freeze it at the deck
#: frequency, the letter is exact in frequency and approximate in depth.
#: Thorp has no depth term, so its letter is exact.
VOLUME_ATTENUATION_CODES = {
    Thorp: 'T',
    Biological: 'B',
    ConstantAbsorption: ' ',
}


def boundary_code(acoustic_type) -> str:
    """The ``TOPOPT(2:2)`` / ``BOTOPT(1:1)`` letter of a boundary type, its
    name in any letter case, or ``None`` (a vacuum)."""
    return BOUNDARY_CODES[parse_boundary_type(acoustic_type)]


def writes_francois_garrison_letter(absorption, *,
                                    multi_frequency: bool = False) -> bool:
    """Whether a deck writes ``absorption`` as AT's ``'F'`` and its ``T S pH
    z_bar`` row: one Francois-Garrison water row on a deck covering several
    frequencies (``multi_frequency``)."""
    return (multi_frequency and isinstance(absorption, FrancoisGarrison)
            and not absorption.is_profile)


def writes_alpha_per_ssp_row(absorption, *,
                             multi_frequency: bool = False) -> bool:
    """Whether a deck carries ``absorption`` in its water SSP rows'
    ``alphaI`` column (dB/wavelength at the deck frequency) under a blank
    ``TOPOPT(4:4)``: a :class:`ConstantAbsorption`, and a law with no letter
    in :data:`VOLUME_ATTENUATION_CODES` (:class:`FrancoisGarrison`, a
    tabulated α(f, z)) — except one Francois-Garrison row on a deck covering
    several frequencies (:func:`writes_francois_garrison_letter`). ``False``
    for ``None``."""
    if absorption is None or writes_francois_garrison_letter(
            absorption, multi_frequency=multi_frequency):
        return False
    for kind in type(absorption).__mro__:
        if kind in VOLUME_ATTENUATION_CODES:
            return VOLUME_ATTENUATION_CODES[kind] == ' '
    return True


def volume_attenuation_code(absorption, *,
                            multi_frequency: bool = False) -> str:
    """The ``TOPOPT(4:4)`` letter of an absorption model, or ``' '`` for
    ``None`` and for a law the SSP rows carry
    (:func:`writes_alpha_per_ssp_row`); ``'F'`` for one Francois-Garrison
    row on a deck covering several frequencies. A subclass writes its
    nearest listed ancestor's letter."""
    if writes_francois_garrison_letter(absorption,
                                       multi_frequency=multi_frequency):
        return 'F'
    if absorption is None or writes_alpha_per_ssp_row(absorption):
        return ' '
    return next(VOLUME_ATTENUATION_CODES[kind]
                for kind in type(absorption).__mro__
                if kind in VOLUME_ATTENUATION_CODES)


def francois_garrison_record(absorption: FrancoisGarrison, z_bar: float
                             ) -> Tuple[float, float, float, float]:
    """The ``T S pH z_bar`` record a ``'F'`` deck reads after the option line
    (``misc/AttenMod.f90``), the formula evaluated at depth ``z_bar`` (m).
    The pH is :attr:`FrancoisGarrison.ph_nbs`, since the solver evaluates the
    same equation on whatever number the deck carries."""
    return (float(absorption.temperature), float(absorption.salinity),
            float(absorption.ph_nbs), float(z_bar))


def biological_records(absorption: Biological
                       ) -> List[Tuple[float, float, float, float, float]]:
    """The ``Z1 Z2 f0 Q a0`` record of each layer a ``'B'`` deck reads after
    the option line (``misc/AttenMod.f90``), in the layers' order."""
    return [
        (layer.z_top_m, layer.z_bottom_m, layer.f0_hz, layer.Q, layer.a0)
        for layer in absorption.layers
    ]


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
    it: ``io.oalib_writer.write_header`` hardwires ``TOPOPT(3:3)='W'``,
    which is the package's documented attenuation convention everywhere.

    The enum is the one table of the ``TOPOPT(3:3)`` letters: the writers
    write :attr:`DB_PER_WAVELENGTH`, and :func:`uacpy.io.read_env` accepts
    exactly the members whose :attr:`converts_to_db_per_wavelength` is true.
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
        :func:`~uacpy.core.acoustics.attenuation.convert_attenuation_units` speaks
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
                f"AttenuationUnits; got {type(value).__name__}: {value!r}.",
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
                f"invalid attenuation unit: {value!r}.",
                remediation="Use one of 'W', 'N', 'F', 'M', 'Q', 'L'.")

    def to_char(self) -> str:
        """Return the single-character Acoustics Toolbox code."""
        return self.value

    @property
    def converts_to_db_per_wavelength(self) -> bool:
        """Whether one dB/wavelength value carries this unit at every frequency.

        ``CRCI`` (``misc/AttenMod.f90:57-80``) turns each unit into Np/m.
        Equated with the ``'W'`` form ``alpha*f/(8.6858896*c)``, the
        frequency cancels for ``'W'``, ``'F'`` (``alpha*f/8685.8896``),
        ``'Q'`` (``omega/(2*c*Q)``) and ``'L'`` (``alpha*omega/c``), so the
        carriers' dB/wavelength holds them exactly. ``'N'`` (Np/m) and
        ``'M'`` (dB/m) are constant in frequency, so no one dB/wavelength
        value reproduces them.
        """
        return self in (AttenuationUnits.DB_PER_WAVELENGTH,
                        AttenuationUnits.DB_PER_M_KHZ,
                        AttenuationUnits.Q_FACTOR,
                        AttenuationUnits.LOSS_PARAMETER)


def parse_boundary_type(value) -> BoundaryType:
    """
    Parse a boundary type string.

    Parameters
    ----------
    value : str or None
        Boundary type string (e.g., 'vacuum', 'rigid', 'half-space') or
        ``None`` for the default.

    Returns
    -------
    BoundaryType
        Parsed enum value; ``None`` maps to ``VACUUM``.
    """
    if value is None:
        return BoundaryType.VACUUM
    return BoundaryType.from_string(value)
