"""Boundary-node carriers: the types a boundary can be (:class:`BoundaryType`),
one boundary's properties (:class:`BoundaryProperties`), one sediment layer
(:class:`SedimentLayer`), the rules a write delegated to a boundary node
obeys, and the reductions that fold many nodes into one.

The seabed (:mod:`uacpy.core.bottom`) and the sea surface
(:mod:`uacpy.core.surface`) both hold their nodes as ``BoundaryProperties``,
so both read this module. Re-exported from :mod:`uacpy.core.environment` for
stable import paths.
"""

import warnings
import copy as _copy
from enum import Enum
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np

from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, ValidityWarning,
)
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.sediment import DEFAULT_GRAIN_SIZE_MODEL
from uacpy.core._validate import (
    require_attenuation_in_range, require_positive, require_non_negative,
    require_finite, warn_speed_typed_in_km_per_s,
)
from uacpy.core._provenance import coerce_data_sources
from uacpy.core._repr import build, qty
from uacpy.core._carrier import (
    DeepCopyMixin, RevalidateOnAssignMixin, carrier,
)
from uacpy.core._export import CarrierExport

__all__ = [
    'BoundaryType', 'SedimentLayer', 'BoundaryProperties',
]


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

        Accepts only the members' values — ``'vacuum'``, ``'rigid'``,
        ``'half-space'``, ``'file'``, ``'precalc'`` — in any letter case.
        Each type has that one name: ``'halfspace'``, ``'elastic'``, the
        enum member names and the Acoustics-Toolbox letters are refused.

        Parameters
        ----------
        value : str or BoundaryType
            Boundary type value (case-insensitive) or enum member.

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
                f"got {type(value).__name__}: {value!r}.",
                remediation=f"Use one of {[bt.value for bt in cls]}.")
        try:
            return cls(value.lower())
        except ValueError:
            raise ConfigurationError(
                f"invalid boundary type: {value!r}.",
                remediation=f"Use one of {[bt.value for bt in cls]}.")

    @property
    def is_geoacoustic(self) -> bool:
        """Whether a boundary of this type carries seabed geoacoustics.

        Only a half-space does. Vacuum and rigid are parameter-free, and a
        reflection table ('file', 'precalc') is read by the solver from its
        file: on all four the cp/ρ/α/cs a :class:`BoundaryProperties` holds
        are construction-time placeholders, which must not feed a numeric
        aggregate or a deck's geoacoustic block."""
        return self is BoundaryType.HALF_SPACE

    @property
    def is_parameter_free(self) -> bool:
        """Whether this type is defined by no parameters at all (vacuum,
        rigid): of the acoustic fields only the interfacial ``roughness`` is
        meaningful on it."""
        return self in (BoundaryType.VACUUM, BoundaryType.RIGID)


def _validate_acoustic_type(value, label: str) -> None:
    """Reject unrecognized ``acoustic_type`` strings up front, so a typo
    like ``'halfspace'`` (vs. ``'half-space'``) fails at construction
    instead of producing a wrong Acoustics-Toolbox bottom-type code
    deep inside a writer.
    """
    try:
        BoundaryType.from_string(value)
    # ConfigurationError alone: ``BoundaryType.from_string`` type-checks its
    # argument before touching it, so a non-string raises no AttributeError off
    # ``.lower()``, and the one internal KeyError from its enum-name lookup is
    # caught and re-raised there as a ConfigurationError too. A broader clause
    # here would swallow an unrelated failure inside the enum and report it as
    # a bad ``acoustic_type``.
    except ConfigurationError as exc:
        valid = sorted({bt.value for bt in BoundaryType})
        raise ConfigurationError(
            f"{label}: acoustic_type={value!r} is not recognized. "
            f"Valid values (any letter case): {valid}."
        ) from exc


# Half-space fields a ``SeabedColumn`` / ``Bottom`` write follows through to
# the stored boundaries. ``Surface`` delegates the same nine names and imports
# this set as ``_SURFACE_DELEGATED`` rather than restating it — both carriers
# hold their nodes in ``BoundaryProperties``, so the field lists cannot
# legitimately differ, and two copies drifted apart in silence.
_HALFSPACE_DELEGATED = frozenset({
    'acoustic_type', 'density', 'sound_speed', 'attenuation', 'roughness',
    'shear_speed', 'shear_attenuation', 'grain_size_phi', 'reflection_file',
})

def _delegate_write(owner: str, nodes, name, value, *, layered=False,
                    noun='columns', hint='.columns[i].halfspace'):
    """Store a write of a delegated boundary field on every node.

    ``owner`` names the carrier for the messages, ``nodes`` are its
    ``BoundaryProperties``, ``layered`` says whether sediment layers sit above
    them, and ``noun``/``hint`` are the per-node spelling the warning offers (a
    ``Surface`` passes ``'nodes'`` / ``'.nodes[i]'``). Every node is
    rebuilt through the ``BoundaryProperties`` constructor with the new value
    before any node changes, so the write gets the constructor's rules, and a
    value one node refuses leaves every node as it was. On more than one node
    the write flattens any range dependence, so it warns — attributed to the
    assigning line by the frame walk, since the write reaches here through
    ``__setattr__``.
    """
    # A layered seabed carries the same field on every layer, so a flat write
    # cannot say which depth the caller means.
    if layered:
        raise ConfigurationError(
            f"{owner}.{name} = {value!r}: this seabed has sediment layers, so "
            f"a flat write cannot say whether you mean a layer or the "
            f"half-space below them. Assign to the layer "
            f"(``.layers[j].{name}``) or to the half-space "
            f"(``.halfspace.{name}``) you mean.")
    try:
        rebuilt = [type(node)(**node._fields_for_assignment(name, value))
                   for node in nodes]
    except ConfigurationError as exc:
        if type(exc) is not ConfigurationError:
            raise
        # The node's refusal, named as the write the caller made.
        raise ConfigurationError(f"{owner}.{name} = {value!r}: {exc.message}",
                                 remediation=exc.remediation) from exc
    value = getattr(rebuilt[0], name)
    if len(nodes) > 1:
        warnings.warn(
            f"{owner}.{name} = {value!r} sets all {len(nodes)} range {noun} "
            f"to the same value, flattening any range dependence. Assign to "
            f"{hint}.{name} to write a single {noun[:-1]}.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
    for node, new in zip(nodes, rebuilt):
        for field in node.__dataclass_fields__:
            object.__setattr__(node, field, new.__dict__[field])


# g/cm³ above which a density reads as kg/m³: no sediment or rock reaches it
# (Hamilton's tables top out below 3), and the smallest kg/m³ value a user
# could plausibly type (fresh water, 1000) is fifty times over it.
_DENSITY_UNITS_SUSPECT_G_CM3 = 20.0


def _warn_implausible_geoacoustics(owner: str, density: float,
                                   sound_speed: float, shear_speed: float):
    """``ValidityWarning`` for a value the constructor accepts but no seabed has.

    Three checks, each a plausibility bound rather than a validity rule, so
    they warn and never raise: a density over
    :data:`_DENSITY_UNITS_SUSPECT_G_CM3` (the number was typed in kg/m³), a
    compressional speed typed in km/s
    (:func:`~uacpy.core._validate.warn_speed_typed_in_km_per_s`), and a shear
    speed above the compressional speed (no real solid: Poisson's ratio
    bounds ``c_s < c_p / sqrt(2)``).

    Reached from both carriers' ``__post_init__`` below the user's constructor
    call or assignment. Both are built by ``@carrier``'s ``__init__``, so every
    frame between here and the caller is an ordinary package frame that
    :data:`USER_FRAME_SKIP` steps over, from a direct ``SedimentLayer(…)``, from
    ``layer.density = …`` and from the in-package factories
    (``Bottom.range_dependent``, the CRUST1 and GRAW readers,
    ``SeabedColumn.collapse_layers``) alike."""
    if density > _DENSITY_UNITS_SUSPECT_G_CM3:
        warnings.warn(
            f"{owner}: density={density:g} looks like kg/m³; uacpy takes "
            f"g/cm³ ({density / 1000.0:g}). Every deck writes the value as "
            f"given, so a seabed this dense reflects like a rigid wall.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP)
    warn_speed_typed_in_km_per_s(sound_speed, f"{owner} sound_speed")
    if shear_speed > sound_speed:
        warnings.warn(
            f"{owner}: shear_speed={shear_speed:g} m/s exceeds "
            f"sound_speed={sound_speed:g} m/s. No real solid has a shear "
            f"speed above its compressional speed (Poisson's ratio bounds "
            f"c_s < c_p/sqrt(2) = {sound_speed / np.sqrt(2.0):g} m/s); check "
            f"whether the two were swapped.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP)


#: The columns of a seabed or surface table: what each row states.
_PROPERTY_COLUMNS = ('sound_speed', 'density', 'attenuation', 'shear_speed',
                     'shear_attenuation', 'roughness')


def _property_row(node) -> dict:
    """One table row's acoustic columns from a layer or a boundary. A
    vacuum or rigid boundary carries no acoustic parameter but its
    roughness, so its others are NaN rather than the placeholders it
    stores."""
    kind = getattr(node, 'acoustic_type', None)
    free = (kind is not None
            and BoundaryType.from_string(kind).is_parameter_free)
    return {name: (np.nan if getattr(node, name, None) is None
                   or (free and name != 'roughness')
                   else float(getattr(node, name)))
            for name in _PROPERTY_COLUMNS}


def _columns(rows) -> dict:
    """Rows of equal keys as one array per column."""
    keys = list(rows[0]) if rows else []
    return {key: np.array([row[key] for row in rows]) for key in keys}


def _material_bits(node) -> List[str]:
    """The acoustic properties of a layer or a half-space in repr words:
    cp, ρ, α, then shear and roughness when present."""
    bits = [f"cp={qty(node.sound_speed, 'm/s')}",
            f"ρ={qty(node.density, 'g/cm³')}",
            f"α={qty(node.attenuation, 'dB/λ')}"]
    if node.shear_speed > 0:
        bits.append(f"cs={qty(node.shear_speed, 'm/s')}")
        if node.shear_attenuation > 0:
            bits.append(f"αs={qty(node.shear_attenuation, 'dB/λ')}")
    if node.roughness > 0:
        bits.append(f"σ={qty(node.roughness, 'm')}")
    return bits


@carrier
class SedimentLayer(RevalidateOnAssignMixin, DeepCopyMixin, CarrierExport):
    """
    Single sediment layer in a layered bottom structure.

    Parameters
    ----------
    thickness : float
        Layer thickness in meters.
    sound_speed : float
        Compressional wave speed (m/s).
    density : float
        Density (g/cm³).
    attenuation : float
        Compressional attenuation (dB/wavelength). Default 0.5.
    shear_speed : float
        Shear wave speed (m/s). Default 0.0 (fluid layer).
    shear_attenuation : float
        Shear attenuation (dB/wavelength). Default 0.0.
    roughness : float
        RMS roughness (m) of the interface at the *top* of this layer, so the
        first layer's value is the seafloor. Default 0.0 (smooth).
    name : str, optional
        Label for the layer, shown in its repr and in the ``name`` column of
        the bottom's layer table; no deck reads it.

    Notes
    -----
    A density above 20 g/cm³ (a kg/m³ value typed into a g/cm³ field) and a
    shear speed above the compressional speed each raise a ``ValidityWarning``
    at construction and on assignment; both are accepted, since neither
    breaks a deck. An assignment rebuilds the layer through the
    constructor, so ``layer.thickness = -1`` is refused as
    ``SedimentLayer(thickness=-1, …)`` is.

    Examples
    --------
    >>> sand = SedimentLayer(thickness=10, sound_speed=1650, density=1.9, attenuation=0.8)
    >>> clay = SedimentLayer(thickness=50, sound_speed=1550, density=1.5, attenuation=0.2)
    """
    thickness: float
    sound_speed: float
    density: float
    attenuation: float = 0.5
    shear_speed: float = 0.0
    shear_attenuation: float = 0.0
    roughness: float = 0.0
    name: Optional[str] = None

    def __post_init__(self):
        # float()-coerce before validating, as BoundaryProperties does: the
        # validators below check a converted copy, so without this a str
        # value would pass them and be stored unconverted.
        for attr in ('thickness', 'sound_speed', 'density', 'attenuation',
                     'shear_speed', 'shear_attenuation', 'roughness'):
            setattr(self, attr, float(getattr(self, attr)))
        require_positive(self.thickness, "SedimentLayer thickness", hint="m")
        require_positive(self.sound_speed, "SedimentLayer sound_speed", hint="m/s")
        require_positive(self.density, "SedimentLayer density", hint="g/cm^3")
        for attr in ('attenuation', 'shear_speed', 'shear_attenuation',
                     'roughness'):
            require_non_negative(getattr(self, attr), f"SedimentLayer {attr}")
        require_attenuation_in_range(
            self.attenuation, "SedimentLayer attenuation")
        require_attenuation_in_range(
            self.shear_attenuation, "SedimentLayer shear_attenuation")
        _warn_implausible_geoacoustics(
            "SedimentLayer", self.density, self.sound_speed, self.shear_speed)

    def __repr__(self) -> str:
        return build('SedimentLayer', [
            repr(self.name) if self.name else None,
            f"thickness={qty(self.thickness, 'm')}",
            *_material_bits(self)])

    @classmethod
    def from_preset(cls, name: str, *, thickness: float, elastic: bool = False,
                    **overrides) -> "SedimentLayer":
        """Build a :class:`SedimentLayer` from a :mod:`uacpy.core.materials`
        preset (``'sand'``, ``'silt'``, ``'clay'``, …).

        ``thickness`` is required (presets only encode acoustic
        properties, not layer geometry). The layer is **fluid by default**;
        pass ``elastic=True`` to keep the preset's shear properties. Any
        additional kwargs override the preset's ``sound_speed`` /
        ``density`` / ``attenuation`` / ``shear_*`` / ``roughness`` for
        site-specific tuning.

        Parameters
        ----------
        name : str
            A preset of :mod:`uacpy.core.materials`.
        thickness : float
            Layer thickness (m).
        elastic : bool, optional
            Keep the preset's shear speed and attenuation. Default False.
        **overrides
            Properties replacing the preset's.
        """
        from uacpy.core.materials import get_material
        m = get_material(name)
        kwargs = dict(
            thickness=thickness,
            sound_speed=m['sound_speed'],
            density=m['density'],
            attenuation=m['attenuation'],
            shear_speed=m['shear_speed'] if elastic else 0.0,
            shear_attenuation=m['shear_attenuation'] if elastic else 0.0,
            roughness=m['roughness'],
            name=name,
        )
        kwargs.update(overrides)
        return cls(**kwargs)


#: Constructor sentinel for ``acoustic_type``: ``None`` means "infer it",
#: which ``__post_init__`` resolves to 'file', 'half-space' or 'vacuum' from
#: the supplied parameters. Typed ``Any`` so the field can declare the ``str``
#: every attribute read sees, without the sentinel widening that declaration
#: back to Optional.
_TYPE_NOT_GIVEN: Any = None


# The constructor takes ``None`` for ``acoustic_type`` ("infer it"); the
# attribute always holds the resolved type.
@carrier(init_annotations=dict(acoustic_type=Optional[str]))
class BoundaryProperties(RevalidateOnAssignMixin, DeepCopyMixin, CarrierExport):
    """
    Properties of ocean boundaries (surface or bottom).

    Carries acoustic properties only — boundary geometry lives on
    ``Environment.bathymetry`` (bottom) or is fixed at z=0 (surface;
    rough surfaces use ``Environment.altimetry``).

    Attributes
    ----------
    acoustic_type : str, optional
        Boundary type: 'vacuum', 'rigid', 'half-space', 'file', 'precalc'.
        Inferred from the supplied parameters when omitted: ``reflection_file``
        → ``'file'``, any **explicitly passed** cp/ρ/α/cs →
        ``'half-space'`` (even a value equal to the documented default —
        passing ``sound_speed=1600`` means a 1600 m/s half-space, never a
        vacuum), nothing → ``'vacuum'``. Pass ``acoustic_type='rigid'``
        explicitly (a parameter-free physical model). To build a bottom from a
        grain size, use :meth:`from_grain_size` — there is no ``'grain-size'``
        type.
    density : float
        Density (g/cm³)
    sound_speed : float
        Compressional wave speed (m/s)
    attenuation : float
        Compressional attenuation (dB/wavelength)
    roughness : float
        RMS interfacial roughness (m). An interface property of *every*
        boundary type (a rough pressure-release sea surface is
        ``BoundaryProperties(roughness=2.0)``), so it never drives the
        ``acoustic_type`` inference.
    shear_speed : float
        Shear wave speed (m/s), 0 = fluid bottom
    shear_attenuation : float
        Shear attenuation (dB/wavelength)
    grain_size_phi : float
        Mean grain size in Wentworth phi units. Informational metadata only
        (e.g. set by :meth:`from_grain_size`); it does not by itself define a
        bottom — see :meth:`from_grain_size`.
    reflection_file : str, optional
        Path to a precomputed reflection-coefficient table, staged beside the
        ``.env`` under the name the solver expects: ``.brc`` for a bottom and
        ``.trc`` for a top with ``acoustic_type='file'``, or ``.irc`` for a
        bottom with ``acoustic_type='precalc'`` (the internal-reflection
        table, a different format). Generated by BOUNCE — a BOUNCE run
        publishes both as ``result.metadata['brc_file']`` and
        ``['irc_file']`` — or written with
        :func:`uacpy.io.write_reflection_coefficient`. OASR's ``.trc`` output
        is a different layout and is not read here.
        Phase-velocity sampling bounds and range stride are carried by the
        consuming model (e.g. ``Kraken(c_low=…, c_high=…)``), not by this
        object.

    Notes
    -----
    On a half-space, a density above 20 g/cm³ (a kg/m³ value typed into a
    g/cm³ field) and a shear speed above the compressional speed each raise
    a ``ValidityWarning`` at construction and on assignment; both are accepted,
    since neither breaks a deck. An assignment rebuilds the boundary through
    the constructor, so it is refused wherever the constructor refuses the
    same value.

    Examples
    --------
    Using pre-computed reflection coefficients from BOUNCE:

    (a sketch — it runs the BOUNCE binary, so it is shown rather than
    executed)::

        import tempfile
        from uacpy.models import Bounce

        # A temporary work dir, so running this leaves nothing behind
        with tempfile.TemporaryDirectory() as work_dir:
            # First, compute reflection coefficients
            bounce = Bounce(work_dir=work_dir)
            result = bounce.run(env, source, receiver)
            brc_file = result.metadata['brc_file']

            # Then use in Bellhop/Kraken/Scooter
            bottom = BoundaryProperties(acoustic_type='file',
                                        reflection_file=brc_file)
            env = Environment(name="test", bathymetry=100, bottom=bottom)
    """

    acoustic_type: str = _TYPE_NOT_GIVEN
    density: Optional[float] = None
    sound_speed: Optional[float] = None
    attenuation: Optional[float] = None
    roughness: Optional[float] = None
    shear_speed: Optional[float] = None
    shear_attenuation: Optional[float] = None
    grain_size_phi: Optional[float] = None
    reflection_file: Optional[str] = None
    name: Optional[str] = None
    data_sources: tuple = ()

    if TYPE_CHECKING:
        # The two roles of a dataclass field annotation, separated for
        # ``acoustic_type``: the attribute holds the resolved boundary type
        # ``__post_init__`` always assigns, while the constructor keeps
        # taking ``None`` to mean "infer it" — declaring both through the
        # field annotation alone gives ``Optional[str]`` to every attribute
        # read, including ``Bottom.acoustic_type``, whose ``-> str`` is then
        # read as a wrong annotation rather than as the total function it
        # is. Never executed; the runtime ``__init__`` is ``@carrier``'s,
        # with the same parameters.
        def __init__(
            self,
            acoustic_type: Optional[str] = None,
            density: Optional[float] = None,
            sound_speed: Optional[float] = None,
            attenuation: Optional[float] = None,
            roughness: Optional[float] = None,
            shear_speed: Optional[float] = None,
            shear_attenuation: Optional[float] = None,
            grain_size_phi: Optional[float] = None,
            reflection_file: Optional[str] = None,
            name: Optional[str] = None,
            data_sources: tuple = (),
        ) -> None: ...

    # Resolved values for acoustic parameters left unset. The dataclass
    # defaults are ``None`` sentinels so "explicitly passed" is detectable:
    # BoundaryProperties(sound_speed=1600) means a 1600 m/s half-space even
    # though 1600 is also the resolved default — value-vs-default comparison
    # cannot tell the two apart. After ``__post_init__`` every attribute
    # carries a concrete float.
    _ACOUSTIC_DEFAULTS = {
        'density': 1.5,
        'sound_speed': 1600.0,
        'attenuation': 0.5,
        'roughness': 0.0,
        'shear_speed': 0.0,
        'shear_attenuation': 0.0,
    }

    def _constructor_fields(self) -> dict:
        """The constructor arguments that rebuild this boundary: a vacuum or
        rigid boundary passes the half-space parameters it ignores as unset,
        since their resolved defaults passed back would read as the conflict
        the constructor refuses."""
        fields = super()._constructor_fields()
        if BoundaryType.from_string(self.acoustic_type).is_parameter_free:
            for key in self._ACOUSTIC_DEFAULTS:
                if key != 'roughness':
                    fields[key] = None
        return fields

    def _fields_for_assignment(self, name, value) -> dict:
        """The constructor arguments an assignment stands for. A vacuum or
        rigid boundary holds the resolved defaults of the half-space
        parameters it ignores; passed back explicitly they would read as
        the conflict the constructor refuses, so they are passed as unset
        — except the one being assigned, which the constructor then refuses
        as it refuses it at construction. An ``acoustic_type`` assignment
        to a parameter-free type leaves them unset the same way."""
        fields = super()._fields_for_assignment(name, value)
        try:
            kind = BoundaryType.from_string(fields['acoustic_type'])
        except ConfigurationError:
            # An unrecognised type: the constructor refuses it by name.
            return fields
        if kind.is_parameter_free:
            for key in self._ACOUSTIC_DEFAULTS:
                if key != name and key != 'roughness':
                    fields[key] = None
        return fields

    def __post_init__(self):
        self.data_sources = coerce_data_sources(
            self.data_sources, "BoundaryProperties")
        if self.grain_size_phi is not None:
            # Signed (gravel is negative), so no sign rule; a finite number
            # all the same — NaN/inf/str stored here reads back as data.
            self.grain_size_phi = float(self.grain_size_phi)
            require_finite(self.grain_size_phi,
                           "BoundaryProperties grain_size_phi", hint="ϕ units")

        explicit = {
            name for name in self._ACOUSTIC_DEFAULTS
            if getattr(self, name) is not None
        }
        for name, default in self._ACOUSTIC_DEFAULTS.items():
            if getattr(self, name) is None:
                setattr(self, name, default)
            else:
                setattr(self, name, float(getattr(self, name)))

        require_positive(self.density, "BoundaryProperties density", hint="g/cm^3")
        # roughness is an RMS magnitude and the OASES writers put it in a
        # column whose sign is an encoding: RG < 0 makes INENVI re-read the
        # record as nine tokens (oases/src/oaseun31.f:72-93), so a negative
        # value shifts every later READ in the deck.
        for name in ('sound_speed', 'attenuation', 'shear_speed',
                     'shear_attenuation', 'roughness'):
            require_non_negative(getattr(self, name), f"BoundaryProperties {name}")
        require_attenuation_in_range(
            self.attenuation, "BoundaryProperties attenuation")
        require_attenuation_in_range(
            self.shear_attenuation, "BoundaryProperties shear_attenuation")

        # Explicitly passed acoustic params drive both the auto-inference
        # (when acoustic_type is None) and the explicit-conflict guard below.
        # ``roughness`` is excluded: it is an interface property every boundary
        # type carries (SSP%sigma), not a half-space acoustic parameter.
        half_space_offenders = [
            f"{name}={getattr(self, name):g}"
            for name in ('sound_speed', 'density', 'attenuation',
                         'shear_speed', 'shear_attenuation')
            if name in explicit
        ]

        if self.acoustic_type is None:
            # Grain size is a construction-time input, not a bottom *type*: a
            # bare ``grain_size_phi`` carries no geoacoustics until converted, so
            # inferring a bottom from it would silently use the default cp/ρ/α.
            # Direct the caller to the explicit factory instead.
            if (self.grain_size_phi is not None and not half_space_offenders
                    and self.reflection_file is None):
                raise ConfigurationError(
                    "BoundaryProperties: grain_size_phi alone does not define a "
                    "bottom (no geoacoustics until converted). Use "
                    "BoundaryProperties.from_grain_size(phi) (or "
                    "Bottom.from_grain_size) to build a half-space from a grain "
                    "size."
                )
            # Auto-infer from the supplied parameters. 'rigid' stays opt-in
            # (a parameter-free physical model).
            if self.reflection_file is not None:
                self.acoustic_type = 'file'
            elif half_space_offenders:
                self.acoustic_type = 'half-space'
            else:
                self.acoustic_type = 'vacuum'

        _validate_acoustic_type(self.acoustic_type, "BoundaryProperties")
        self.acoustic_type = BoundaryType.from_string(self.acoustic_type).value

        # ``misc/ReadEnvironmentMod.f90:292`` aborts a half-space whose
        # compressional speed *or* density vanishes. The density half of that
        # guard is the _require_positive above; this is the other half. Checked
        # after the type is resolved because vacuum/rigid/file boundaries carry
        # placeholder speeds they never use, and reachable only here: an
        # explicitly-passed sound_speed forces 'half-space' or trips the
        # conflict guard below, and the unset default is non-zero.
        if self.acoustic_type == 'half-space':
            require_positive(self.sound_speed,
                             "BoundaryProperties sound_speed on a half-space",
                             hint="m/s")

        # Explicit-conflict guard: vacuum/rigid ignore half-space params,
        # so explicitly setting one alongside non-default cp/ρ/α/cs is a
        # mistake the auto-infer path would never make.
        if BoundaryType.from_string(self.acoustic_type).is_parameter_free:
            offenders = list(half_space_offenders)
            if self.reflection_file is not None:
                offenders.append(f"reflection_file={self.reflection_file!r}")
            if self.grain_size_phi is not None:
                offenders.append(f"grain_size_phi={self.grain_size_phi:g}")
            if offenders:
                raise ConfigurationError(
                    f"BoundaryProperties(acoustic_type={self.acoustic_type!r}) "
                    f"ignores half-space acoustic parameters, but you set "
                    f"{', '.join(offenders)}. Drop ``acoustic_type=`` to let "
                    f"uacpy infer 'half-space', or remove the conflicting "
                    f"parameters."
                )

        # Plausibility, after every rule that raises: a value no seabed has
        # but every deck accepts. Only a half-space carries the numbers it
        # is about; the parameter-free and file types hold the defaults.
        if self.acoustic_type == 'half-space':
            _warn_implausible_geoacoustics(
                "BoundaryProperties", self.density, self.sound_speed,
                self.shear_speed)

    def _repr_bits(self) -> List[str]:
        """The boundary in repr words, which a :class:`SeabedColumn`,
        :class:`Bottom` or :class:`Surface` repr holding it reuses."""
        head = (f"{self.name!r} " if self.name else '') + self.acoustic_type
        if BoundaryType.from_string(self.acoustic_type).is_parameter_free:
            return [head] + ([f"σ={qty(self.roughness, 'm')}"]
                             if self.roughness > 0 else [])
        if self.acoustic_type in ('file', 'precalc'):
            return [f"{head} {self.reflection_file!r}"]
        return [head, *_material_bits(self)]

    def _short(self) -> str:
        """The boundary in a few words, with no comma: ``"'sand' half-space
        cp=1650 m/s"``, ``'rigid'``."""
        bits = [repr(self.name)] if self.name else []
        bits.append(self.acoustic_type)
        if self.acoustic_type in ('file', 'precalc'):
            bits.append(repr(self.reflection_file))
        elif BoundaryType.from_string(self.acoustic_type).is_geoacoustic:
            bits.append(f"cp={qty(self.sound_speed, 'm/s')}")
        return ' '.join(bits)

    def __repr__(self) -> str:
        return build('BoundaryProperties', self._repr_bits())

    @classmethod
    def from_grain_size(
        cls, grain_size_phi: float, *, model: str = DEFAULT_GRAIN_SIZE_MODEL,
        hamilton_fit: Optional[str] = None,
        roughness: float = 0.0,
        water_sound_speed: Optional[float] = None,
        water_density: Optional[float] = None,
    ) -> "BoundaryProperties":
        """Build a half-space bottom from a mean grain size (Wentworth ϕ).

        Converts ϕ to explicit ``sound_speed`` / ``density`` / ``attenuation``
        via :func:`uacpy.core.sediment.grain_size_to_geoacoustics` so the bottom
        works in *every* model. ``grain_size_phi`` is retained as informational
        metadata. This is the only supported way to use a grain size — there is
        no ``'grain-size'`` boundary type.

        Parameters
        ----------
        grain_size_phi : float
            Mean grain size on the Wentworth ϕ scale.
        model : {'hamilton', 'apl-uw'}, optional
            Conversion model (see :func:`grain_size_to_geoacoustics`).
        hamilton_fit : str, optional
            Which of Hamilton & Bachman's three fits ``'hamilton'`` uses —
            ``'continental-terrace'`` (the default when ``None``),
            ``'abyssal-hill'`` or ``'abyssal-plain'``. A deep-ocean seabed
            wants one of the abyssal fits; nothing infers it, because the
            paper states no rule for choosing and only the caller knows the
            site.
        roughness : float, optional
            RMS interface roughness (m).
        water_sound_speed, water_density : float, optional
            In-situ seawater properties the ratios scale by (default: the
            package's one water, the nominal 1500 m/s and 1.027 g/cm³).
        """
        from uacpy.core.sediment import (DEFAULT_GRAIN_SIZE_ENVIRONMENT,
                                         canonical_grain_size_selection,
                                         grain_size_to_geoacoustics)
        if hamilton_fit is None:
            hamilton_fit = DEFAULT_GRAIN_SIZE_ENVIRONMENT
        model, hamilton_fit = canonical_grain_size_selection(
            model, hamilton_fit, who='BoundaryProperties.from_grain_size')
        g = grain_size_to_geoacoustics(
            grain_size_phi, model=model, hamilton_fit=hamilton_fit,
            water_sound_speed=water_sound_speed, water_density=water_density)
        return cls(
            acoustic_type='half-space',
            grain_size_phi=float(grain_size_phi),
            sound_speed=g['sound_speed'],
            density=g['density'],
            attenuation=g['attenuation'],
            roughness=roughness,
        )

    @classmethod
    def from_preset(cls, name: str, *, elastic: bool = False, **overrides) -> "BoundaryProperties":
        """Build a :class:`BoundaryProperties` from a
        :mod:`uacpy.core.materials` preset.

        Picks ``acoustic_type='half-space'`` automatically, copies every
        preset field that maps onto :class:`BoundaryProperties` (sound
        speeds, density, attenuations, ``grain_size_phi`` if defined,
        ``roughness``), and applies any ``**overrides`` last.

        The boundary is **fluid by default** (shear dropped) so it works
        with every model. Pass ``elastic=True`` to keep the preset's shear
        speed / attenuation — needed only for the elastic-capable solvers
        (OASES, Scooter, Kraken). ``shear_*`` in ``**overrides`` wins
        regardless.

        Parameters
        ----------
        name : str
            A preset of :mod:`uacpy.core.materials`.
        elastic : bool, optional
            Keep the preset's shear speed and attenuation. Default False.
        **overrides
            Properties replacing the preset's.
        """
        from uacpy.core.materials import get_material
        m = get_material(name)
        kwargs = dict(
            acoustic_type='half-space',
            sound_speed=m['sound_speed'],
            density=m['density'],
            attenuation=m['attenuation'],
            shear_speed=m['shear_speed'] if elastic else 0.0,
            shear_attenuation=m['shear_attenuation'] if elastic else 0.0,
            roughness=m['roughness'],
            name=name,
        )
        if m['grain_size_phi'] is not None:
            kwargs['grain_size_phi'] = m['grain_size_phi']
        kwargs.update(overrides)
        return cls(**kwargs)


# Numeric acoustic fields a SedimentLayer shares with BoundaryProperties.
# ``roughness`` is absent: it is an interface property, not a bulk one.
_LAYER_ACOUSTIC_FIELDS = ('density', 'sound_speed', 'attenuation',
                          'shear_speed', 'shear_attenuation')


def _boundary_from_values(template: BoundaryProperties, values: dict
                          ) -> BoundaryProperties:
    """Build a :class:`BoundaryProperties` from reduced numeric ``values``.

    ``values`` supplies the numeric acoustic fields (the keys of
    ``BoundaryProperties._ACOUSTIC_DEFAULTS``); the non-blendable fields
    (``acoustic_type``, ``grain_size_phi``, ``reflection_file``, ``name``,
    ``data_sources``) come from ``template``. ``'vacuum'`` / ``'rigid'`` carry
    no acoustic parameters — passing them alongside is a construction-time
    error — so only the interfacial ``roughness`` survives for those types.

    The single home for "reduced numbers → one boundary", shared by
    :func:`_reduce_boundaries` and :meth:`SeabedColumn.collapse_layers`.
    """
    if BoundaryType.from_string(template.acoustic_type).is_parameter_free:
        return BoundaryProperties(
            acoustic_type=template.acoustic_type,
            roughness=values['roughness'],
            name=template.name, data_sources=template.data_sources)
    return BoundaryProperties(
        acoustic_type=template.acoustic_type,
        grain_size_phi=template.grain_size_phi,
        reflection_file=template.reflection_file,
        name=template.name, data_sources=template.data_sources,
        **values)


def _reduce_boundaries(props: List[BoundaryProperties], reducer
                       ) -> BoundaryProperties:
    """Reduce a list of :class:`BoundaryProperties` to a single one.

    ``reducer(values) -> float`` folds the per-node values of each numeric
    acoustic field (the keys of ``BoundaryProperties._ACOUSTIC_DEFAULTS``):
    ``np.mean`` / ``np.median`` for a collapse, ``np.interp`` for a
    range-interpolated blend. The non-blendable fields come from the first
    node (see :func:`_boundary_from_values`).

    The single home for "many boundaries → one", shared by
    :meth:`Bottom.halfspace_at`, :meth:`Bottom.collapse_range` and
    :meth:`uacpy.core.surface.Surface.collapse_range`.
    """
    values = {name: float(reducer([getattr(p, name) for p in props]))
              for name in BoundaryProperties._ACOUSTIC_DEFAULTS}
    return _boundary_from_values(props[0], values)


def _reduce_uniform_nodes(nodes: List[BoundaryProperties], method: str,
                          who: str, noun: str) -> BoundaryProperties:
    """``'mean'`` / ``'median'`` a range axis of boundary nodes down to one
    :class:`BoundaryProperties` — the reduction :meth:`Bottom.collapse_range`
    (``noun='columns'``) and :meth:`uacpy.core.surface.Surface.collapse_range`
    (``noun='nodes'``) share; ``who`` names the caller in the messages.

    Averaging is only meaningful within one boundary type: reducing a vacuum
    node with a sand half-space would fold construction-time placeholders
    into the numbers and stamp one node's type on the result. A uniform
    ``'file'``/``'precalc'`` axis carries no real numbers to reduce — each
    node is its reflection-coefficient table. Nodes sharing one table
    collapse to that shared spec (roughness, the one genuine number they
    carry, is still reduced); distinct tables cannot be averaged into
    anything.
    """
    types = {n.acoustic_type for n in nodes}
    if len(types) > 1:
        raise ConfigurationError(
            f"{who}({method!r}) needs a single boundary type to average; "
            f"got {sorted(types)}. Boundary types cannot be blended — use "
            f"'r0' or 'rmax'.")
    # Refused rather than assumed. ``np.mean if method == 'mean' else
    # np.median`` makes every unrecognised method a median, which is a
    # silently wrong boundary rather than an error. Both callers already
    # reject anything outside ('mean','median') before reaching here, so
    # this is defence in depth and not a path anyone can currently take —
    # it exists so that a caller added later cannot reopen one.
    if method not in ('mean', 'median'):
        raise ConfigurationError(
            f"{who}({method!r}) is not a numeric reduction of a range axis; "
            f"this reducer implements 'mean' and 'median'. Pick one of "
            f"those, or 'r0'/'rmax' to keep one node whole.")
    reduce = np.mean if method == 'mean' else np.median
    (the_type,) = types
    if the_type in ('file', 'precalc'):
        specs = {n.reflection_file for n in nodes}
        if len(specs) > 1:
            raise ConfigurationError(
                f"{who}({method!r}) cannot average '{the_type}' {noun} with "
                f"different reflection files ({sorted(specs, key=str)}). "
                f"Reflection-coefficient tables cannot be blended — use "
                f"'r0' or 'rmax'.")
        shared = _copy.deepcopy(nodes[0])
        shared.roughness = float(reduce([n.roughness for n in nodes]))
        return shared
    return _reduce_boundaries(nodes, reduce)
