"""Seafloor / boundary carriers: sediment layers, half-space and layered
bottom properties, and their range-dependent variants. Re-exported from
:mod:`uacpy.core.environment` for stable import paths.
"""

import warnings
import copy as _copy
import numpy as np
from typing import TYPE_CHECKING, Any, List, Tuple, Optional, Dict
from dataclasses import dataclass

from uacpy.core.exceptions import ConfigurationError
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.constants import DECK_RANGE_RESOLUTION_M
from uacpy.core._grid import (
    _as_finite_scalar_label, _nearest_index_on_axis,
)
from uacpy.core._carrier_validate import (
    _validate_acoustic_type, _require_strictly_increasing,
    _require_attenuation_in_range,
    _require_positive, _require_non_negative, _coerce_data_sources,
    _dedupe_provenance,
)


# Boundary types whose cp/ρ/α/cs are construction-time placeholders, never
# seabed geoacoustics: the solver reads the reflection behaviour from the type
# (or its reflection_file), so these values must not feed numeric aggregates.
_NON_GEOACOUSTIC_TYPES = frozenset({'vacuum', 'rigid', 'file', 'precalc'})

# Boundary types defined by no parameters at all — of the acoustic fields only
# the interfacial ``roughness`` is meaningful on them.
_PARAMETER_FREE_TYPES = frozenset({'vacuum', 'rigid'})

# Half-space fields a ``SeabedColumn`` / ``Bottom`` write follows through to
# the stored boundaries, mirroring ``Surface``'s delegated set.
_HALFSPACE_DELEGATED = frozenset({
    'acoustic_type', 'density', 'sound_speed', 'attenuation', 'roughness',
    'shear_speed', 'shear_attenuation', 'grain_size_phi', 'reflection_file',
})

# Delegated fields that define what kind of boundary a half-space is. A write
# to one cannot be validated field-by-field (the construction rules couple it
# to the other fields), so the write is refused rather than stored unvalidated.
_HALFSPACE_TYPE_FIELDS = frozenset({'acoustic_type', 'reflection_file'})


def _validate_halfspace_write(owner: str, name, value, halfspaces, layered):
    """Apply the ``BoundaryProperties`` construction rules to a write
    delegated to ``halfspaces``, so the carrier cannot store a value its
    constructor would refuse. Returns the value coerced to float."""
    if name in _HALFSPACE_TYPE_FIELDS:
        raise ConfigurationError(
            f"{owner}.{name} cannot be assigned in place: it defines what "
            f"kind of boundary the half-space is, and the construction rules "
            f"couple it to the other fields. Build a new BoundaryProperties "
            f"(and a bottom from it) instead.")
    # A layered seabed carries the same field on every layer, so a flat write
    # cannot say which depth the caller means.
    if layered:
        raise ConfigurationError(
            f"{owner}.{name} = {value!r}: this seabed has sediment layers, so "
            f"a flat write cannot say whether you mean a layer or the "
            f"half-space below them. Assign to the layer "
            f"(``.layers[j].{name}``) or to the half-space "
            f"(``.halfspace.{name}``) you mean.")
    # vacuum/rigid half-spaces carry no acoustic parameters, so a delegated
    # write of one is the conflict the constructor's explicit-conflict guard
    # rejects. ``roughness`` stays writable — every boundary type has an
    # interface.
    if name != 'roughness':
        bare = sorted({h.acoustic_type for h in halfspaces
                       if h.acoustic_type in _PARAMETER_FREE_TYPES})
        if bare:
            raise ConfigurationError(
                f"{owner}.{name} = {value!r}: this seabed has "
                f"{'/'.join(bare)} half-space(s), which ignore half-space "
                f"acoustic parameters. Build a half-space BoundaryProperties "
                f"to give the seabed geoacoustics.")
    if name == 'grain_size_phi':
        # ϕ = −log₂(d/mm) is signed (gravel is negative), so the non-negative
        # rule the other fields take does not apply, and ``None`` is the
        # field's own unset value.
        return None if value is None else float(value)
    value = float(value)
    if name == 'density':
        _require_positive(value, f"{owner} density", hint="g/cm^3")
    elif name == 'sound_speed' and any(h.acoustic_type == 'half-space'
                                       for h in halfspaces):
        _require_positive(value, f"{owner} sound_speed on a half-space",
                          hint="m/s")
    else:
        _require_non_negative(value, f"{owner} {name}")
    if name in ('attenuation', 'shear_attenuation'):
        _require_attenuation_in_range(value, f"{owner} {name}")
    return value


@dataclass
class SedimentLayer:
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
        _require_positive(self.thickness, "SedimentLayer thickness", hint="m")
        _require_positive(self.sound_speed, "SedimentLayer sound_speed", hint="m/s")
        _require_positive(self.density, "SedimentLayer density", hint="g/cm^3")
        for attr in ('attenuation', 'shear_speed', 'shear_attenuation',
                     'roughness'):
            _require_non_negative(getattr(self, attr), f"SedimentLayer {attr}")
        _require_attenuation_in_range(
            self.attenuation, "SedimentLayer attenuation")
        _require_attenuation_in_range(
            self.shear_attenuation, "SedimentLayer shear_attenuation")

    def __repr__(self) -> str:
        tag = f"{self.name!r}, " if self.name else ""
        bits = [
            f"thickness={self.thickness:g} m",
            f"cp={self.sound_speed:g} m/s",
            f"ρ={self.density:g}",
            f"α={self.attenuation:g}",
        ]
        if self.shear_speed > 0:
            bits.append(f"cs={self.shear_speed:g} m/s")
        return f"SedimentLayer({tag}{', '.join(bits)})"

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

    def copy(self) -> 'SedimentLayer':
        """Deep copy (symmetric with the other carriers)."""
        return _copy.deepcopy(self)


#: Constructor sentinel for ``acoustic_type``: ``None`` means "infer it",
#: which ``__post_init__`` resolves to 'file', 'half-space' or 'vacuum' from
#: the supplied parameters. Typed ``Any`` so the field can declare the ``str``
#: every attribute read sees, without the sentinel widening that declaration
#: back to Optional.
_TYPE_NOT_GIVEN: Any = None


@dataclass
class BoundaryProperties:
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
        table, a different format). Generated by BOUNCE or OASR — a BOUNCE
        run publishes both as ``result.metadata['brc_file']`` and
        ``['irc_file']``.
        Phase-velocity sampling bounds and range stride are carried by the
        consuming model (e.g. ``Kraken(c_low=…, c_high=…)``), not by this
        object.

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
        # ``__post_init__`` always assigns, while the constructor keeps taking
        # ``None`` to mean "infer it". Declaring both through the field
        # annotation alone gives ``Optional[str]`` to every attribute read —
        # including ``Bottom.acoustic_type``, whose ``-> str`` is then read as
        # a wrong annotation rather than as the total function it is. Never
        # executed, so the decorator compiles the runtime ``__init__`` from
        # the fields exactly as before.
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

    def __post_init__(self):
        self.data_sources = _coerce_data_sources(
            self.data_sources, "BoundaryProperties")

        explicit = {
            name for name in self._ACOUSTIC_DEFAULTS
            if getattr(self, name) is not None
        }
        for name, default in self._ACOUSTIC_DEFAULTS.items():
            if getattr(self, name) is None:
                setattr(self, name, default)
            else:
                setattr(self, name, float(getattr(self, name)))

        _require_positive(self.density, "BoundaryProperties density", hint="g/cm^3")
        # roughness is an RMS magnitude and the OASES writers put it in a
        # column whose sign is an encoding: RG < 0 makes INENVI re-read the
        # record as nine tokens (oases/src/oaseun31.f:72-93), so a negative
        # value shifts every later READ in the deck.
        for name in ('sound_speed', 'attenuation', 'shear_speed',
                     'shear_attenuation', 'roughness'):
            _require_non_negative(getattr(self, name), f"BoundaryProperties {name}")
        _require_attenuation_in_range(
            self.attenuation, "BoundaryProperties attenuation")
        _require_attenuation_in_range(
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
        from uacpy.core.constants import BoundaryType
        self.acoustic_type = BoundaryType.from_string(self.acoustic_type).value

        # ``misc/ReadEnvironmentMod.f90:292`` aborts a half-space whose
        # compressional speed *or* density vanishes. The density half of that
        # guard is the _require_positive above; this is the other half. Checked
        # after the type is resolved because vacuum/rigid/file boundaries carry
        # placeholder speeds they never use, and reachable only here: an
        # explicitly-passed sound_speed forces 'half-space' or trips the
        # conflict guard below, and the unset default is non-zero.
        if self.acoustic_type == 'half-space':
            _require_positive(self.sound_speed,
                              "BoundaryProperties sound_speed on a half-space",
                              hint="m/s")

        # Explicit-conflict guard: vacuum/rigid ignore half-space params,
        # so explicitly setting one alongside non-default cp/ρ/α/cs is a
        # mistake the auto-infer path would never make.
        if self.acoustic_type in _PARAMETER_FREE_TYPES:
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

    def __repr__(self) -> str:
        if self.acoustic_type in _PARAMETER_FREE_TYPES:
            bits = [self.acoustic_type]
        elif self.acoustic_type in ('file', 'precalc'):
            bits = [f"{self.acoustic_type}={self.reflection_file!r}"]
        else:
            bits = [self.acoustic_type,
                    f"cp={self.sound_speed:g} m/s",
                    f"ρ={self.density:g}",
                    f"α={self.attenuation:g}"]
            if self.shear_speed > 0:
                bits.append(f"cs={self.shear_speed:g} m/s")
        if self.roughness > 0:
            bits.append(f"σ={self.roughness:g} m")
        return f"BoundaryProperties({', '.join(bits)})"

    @classmethod
    def from_grain_size(
        cls, grain_size_phi: float, *, model: str = 'hamilton',
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
        roughness : float, optional
            RMS interface roughness (m).
        water_sound_speed, water_density : float, optional
            In-situ seawater properties the ratios scale by (default: the
            Hamilton reference 1510 m/s, 1.030 g/cm³).
        """
        from uacpy.core.sediment import grain_size_to_geoacoustics
        g = grain_size_to_geoacoustics(
            grain_size_phi, model=model,
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

    def copy(self) -> 'BoundaryProperties':
        """Deep copy (symmetric with the other carriers)."""
        return _copy.deepcopy(self)

# The dataclass compiles ``__init__`` from the *field* annotations, so
# ``inspect.signature`` / ``help()`` would advertise a default the annotation
# refuses (``acoustic_type: str = None``). Restate the input types on the
# generated ``__init__`` so the runtime signature says what the block above
# and the Parameters section say. Annotations only: no default, no field and
# no behaviour changes, and the class annotations — which are what an
# attribute read is checked against — are untouched.
BoundaryProperties.__init__.__annotations__.update(
    acoustic_type=Optional[str],
)


_COLUMN_COLLAPSE_METHODS = ('halfspace', 'top_layer', 'volume_average')

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
    :func:`_reduce_boundaries` and :meth:`SeabedColumn.collapse`.
    """
    if template.acoustic_type in _PARAMETER_FREE_TYPES:
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
    :meth:`Bottom.halfspace_at`, :meth:`Bottom.select_range` and
    :meth:`uacpy.core.surface.Surface.collapse`.
    """
    values = {name: float(reducer([getattr(p, name) for p in props]))
              for name in BoundaryProperties._ACOUSTIC_DEFAULTS}
    return _boundary_from_values(props[0], values)


# ─────────────────────────────────────────────────────────────────────────────
# Unified bottom carrier — one ``Bottom`` (range as an optional axis), mirroring
# ``SoundSpeedProfile``. A ``SeabedColumn`` is "layers over a half-space" at one
# range; an empty layer list means a pure half-space. ``Bottom`` holds one or
# more columns plus an optional ``ranges`` vector.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SeabedColumn:
    """A seabed column at one range: sediment layers over a half-space.

    ``layers`` may be **empty** — that is a pure half-space. A non-empty
    stack is a layered seabed.

    Parameters
    ----------
    layers : list of SedimentLayer
        Sediment layers, shallow → deep. May be empty (pure half-space).
    halfspace : BoundaryProperties
        The deep half-space below all layers.

    Notes
    -----
    The accessors (:meth:`at`, :meth:`isel`) hand back a **copy**, so what a
    caller does to a result never reaches the column. Reach through
    ``.layers`` / ``.halfspace`` to edit in place.
    """
    layers: List[SedimentLayer]
    halfspace: BoundaryProperties

    def __post_init__(self):
        if self.layers is None:
            self.layers = []
        self.layers = list(self.layers)
        for la in self.layers:
            if not isinstance(la, SedimentLayer):
                raise ConfigurationError(
                    "SeabedColumn: every layer must be a SedimentLayer; got "
                    f"{type(la).__name__}")
        if not isinstance(self.halfspace, BoundaryProperties):
            raise ConfigurationError(
                "SeabedColumn: halfspace must be a BoundaryProperties; "
                f"got {type(self.halfspace).__name__}"
            )

    @property
    def is_layered(self) -> bool:
        return len(self.layers) > 0

    @property
    def data_sources(self) -> tuple:
        """Provenance of this column — its half-space's ``data_sources``
        (harmonised with the leaf carriers; a fetched seabed stamps the
        half-space)."""
        return tuple(getattr(self.halfspace, 'data_sources', ()) or ())

    def __repr__(self) -> str:
        bits = [f"n_layers={len(self.layers)}"]
        if self.layers:
            bits.append(f"thickness={self.total_thickness():g} m")
        if any(layer.shear_speed > 0 for layer in self.layers) \
                or self.halfspace.shear_speed > 0:
            bits.append("elastic")
        bits.append(f"halfspace={self.halfspace.acoustic_type}")
        if self.halfspace.acoustic_type not in _NON_GEOACOUSTIC_TYPES:
            bits.append(f"cp={self.halfspace.sound_speed:g} m/s")
        return f"SeabedColumn({', '.join(bits)})"

    def total_thickness(self) -> float:
        """Total thickness of all sediment layers (m); 0 for a half-space."""
        return sum(layer.thickness for layer in self.layers)

    def copy(self) -> 'SeabedColumn':
        """Deep copy (symmetric with the other carriers)."""
        return _copy.deepcopy(self)

    def __setattr__(self, name, value):
        # Writes to a half-space field follow through to ``halfspace``. A plain
        # assignment would create an instance attribute that echoes the new
        # value back while ``at()``, ``sample_at_depths()``, the repr and every
        # writer — all of which read ``halfspace`` — keep the previous one.
        if name in _HALFSPACE_DELEGATED and 'halfspace' in self.__dict__:
            value = _validate_halfspace_write(
                type(self).__name__, name, value, [self.halfspace],
                bool(self.layers))
            setattr(self.halfspace, name, value)
            return
        super().__setattr__(name, value)

    def _layer_at(self, depth: float) -> Optional[SedimentLayer]:
        """Internal: the :class:`SedimentLayer` containing sub-bottom ``depth``
        (m, ``0`` = top of the column), or ``None`` for the deep half-space.

        A private helper — the public accessors are the carrier contract
        :meth:`at` (depth → material) and :meth:`isel` (index → layer); this
        just single-sources the layer-boundary convention they share with
        :meth:`sample_at_depths` (a depth exactly on an internal boundary maps
        to the **upper** layer).
        """
        z = _as_finite_scalar_label(depth, 'depth')
        cumulative = 0.0
        for layer in self.layers:
            cumulative += layer.thickness
            if z <= cumulative:
                return layer
        return None

    def at(self, *, depth: float) -> BoundaryProperties:
        """Material :class:`BoundaryProperties` at sub-bottom ``depth`` (m,
        ``0`` = top of the column).

        A step lookup — returns the containing layer's properties, or the deep
        half-space below the last layer. Distinct materials are never blended,
        so a `SeabedColumn` has no ``eval`` (same as `Bottom`). Positional
        counterpart: :meth:`isel`.

        The layer's ``roughness`` travels with it: both fields describe the
        interface at the top of their material, so the returned boundary
        carries the same number the layer does. (:meth:`collapse` is the one
        place that does not — a reduction over the whole stack has only the
        seabed surface to report, so it takes the half-space's.)

        ``depth`` must be a finite scalar — a NaN/inf or array-valued label
        raises ``ConfigurationError``.
        """
        layer = self._layer_at(depth)
        if layer is None:
            return _copy.deepcopy(self.halfspace)
        return BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=layer.sound_speed, density=layer.density,
            attenuation=layer.attenuation,
            shear_speed=layer.shear_speed,
            shear_attenuation=layer.shear_attenuation,
            roughness=layer.roughness,
            name=layer.name)

    def isel(self, *, layer: int) -> SedimentLayer:
        """A copy of the :class:`SedimentLayer` at integer index ``layer`` —
        the positional counterpart of :meth:`at`. (The deep half-space is
        ``self.halfspace``.)"""
        i = int(layer)
        n = len(self.layers)
        if not -n <= i < n:
            raise IndexError(
                f"SeabedColumn.isel: layer index {i} out of range for "
                f"{n} layer(s)")
        return _copy.deepcopy(self.layers[i])

    def layer_depths(self, seafloor_depth: float) -> List[Tuple[float, float]]:
        """``(top, bottom)`` depth pairs for each layer (empty for a
        half-space). ``seafloor_depth`` is the top of the first layer (m)."""
        depths = []
        current = seafloor_depth
        for layer in self.layers:
            top = current
            bottom = current + layer.thickness
            depths.append((top, bottom))
            current = bottom
        return depths

    def to_piecewise_breakpoints(
        self,
        seafloor_depth: float,
        zmax: Optional[float] = None,
        properties: Tuple[str, ...] = (
            'sound_speed', 'density', 'attenuation',
        ),
    ) -> Dict[str, List[Tuple[float, float]]]:
        """Collins-style ``(depth, value)`` breakpoints per property — each
        layer becomes a (top, bottom) step, then the half-space to ``zmax``.
        With 0 layers the half-space spans from ``seafloor_depth`` down."""
        out = {p: [] for p in properties}
        depths = self.layer_depths(seafloor_depth)
        for (top, bottom), layer in zip(depths, self.layers):
            for prop in properties:
                value = float(getattr(layer, prop, 0.0) or 0.0)
                out[prop].append((float(top), value))
                out[prop].append((float(bottom), value))

        deepest_layer_bottom = depths[-1][1] if depths else seafloor_depth
        final_depth = float(zmax) if zmax is not None else deepest_layer_bottom
        # Give the half-space a non-zero depth extent: a ``zmax`` that does not
        # reach past the layer stack would emit both of its breakpoints at the
        # same depth, i.e. a step of zero thickness.
        if final_depth <= deepest_layer_bottom:
            final_depth = deepest_layer_bottom + 1.0
        for prop in properties:
            hs_value = float(getattr(self.halfspace, prop, 0.0) or 0.0)
            out[prop].append((deepest_layer_bottom, hs_value))
            out[prop].append((final_depth, hs_value))
        return out

    def collapse(self, method: str = 'halfspace') -> BoundaryProperties:
        """Collapse the column to a single ``BoundaryProperties``.

        ``'halfspace'`` → the deep half-space; ``'top_layer'`` → topmost
        layer's properties (half-space when there are no layers);
        ``'volume_average'`` → thickness-weighted mean over the finite layers
        **and the half-space**, the half-space carrying the deepest layer's
        thickness as its weight (it has none of its own). A thin layer over a
        fast basement therefore lands between the two, not on the layer: 1 m of
        1500 m/s mud over a 5250 m/s half-space collapses to 3375 m/s. Reach
        for ``'top_layer'`` when the layer is what matters acoustically.
        Half-space alone when there are no layers.

        The half-space is the template for the non-blendable fields, so a
        ``'vacuum'`` / ``'rigid'`` column collapses back to that parameter-free
        type carrying only its ``roughness``."""
        if method not in _COLUMN_COLLAPSE_METHODS:
            raise ConfigurationError(
                f"SeabedColumn.collapse: unknown method={method!r}; "
                f"valid: {_COLUMN_COLLAPSE_METHODS}"
            )
        if method == 'halfspace' or not self.layers:
            return _copy.deepcopy(self.halfspace)
        # The solver reads no geoacoustic parameters from a non-geoacoustic
        # half-space, so the caller's 'top_layer'/'volume_average' request
        # cannot influence the result. Deliberate, but say so — and say what
        # each type actually does with the reduced values: vacuum/rigid drop
        # them (only roughness survives — see _boundary_from_values), while
        # file/precalc store them on the returned object but the solver reads
        # the reflection-coefficient file instead.
        if self.halfspace.acoustic_type in _NON_GEOACOUSTIC_TYPES:
            if self.halfspace.acoustic_type in _PARAMETER_FREE_TYPES:
                outcome = (f"the {len(self.layers)} sediment layer(s)' "
                           f"properties are dropped (only the interfacial "
                           f"roughness survives)")
            else:
                outcome = (f"the values reduced from the {len(self.layers)} "
                           f"sediment layer(s) are kept on the returned "
                           f"boundary but the solver reads the "
                           f"reflection-coefficient file and ignores them")
            warnings.warn(
                f"SeabedColumn.collapse({method!r}): the half-space is "
                f"'{self.halfspace.acoustic_type}' — the solver reads no "
                f"geoacoustic parameters from it, so {outcome}.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
        if method == 'top_layer':
            top = self.layers[0]
            values = {name: float(getattr(top, name))
                      for name in _LAYER_ACOUSTIC_FIELDS}
        else:
            # Thickness-weighted mean over the finite layers plus the
            # half-space. The semi-infinite half-space has no finite thickness,
            # so it is weighted by the deepest layer's thickness — a pragmatic
            # choice that keeps it comparable to the adjacent layer rather than
            # dominating (∞ weight) or vanishing (0).
            weights = np.array([float(la.thickness) for la in self.layers])
            weights = np.append(weights, float(weights[-1]))

            def _avg(field):
                vals = np.array([getattr(la, field) for la in self.layers]
                                + [getattr(self.halfspace, field)])
                return float(np.average(vals, weights=weights))
            values = {name: _avg(name) for name in _LAYER_ACOUSTIC_FIELDS}
        # Roughness is an interface property of the seabed surface, not of a
        # buried layer, so it always comes from the half-space.
        values['roughness'] = float(self.halfspace.roughness)
        return _boundary_from_values(self.halfspace, values)

    def sample_at_depths(
        self, n_points: int = 4, max_thickness: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample ``(cp, rho, attn)`` — compressional speed (m/s), density
        (g/cm³) and attenuation (dB/λ) — at ``n_points`` evenly-spaced depths in
        ``[0, max_thickness]`` (defaults to this column's own thickness). Used
        by RAM to map arbitrary layers onto its fixed sediment grid."""
        max_thick = (float(max_thickness) if max_thickness is not None
                     else self.total_thickness())
        if max_thick <= 0:
            # A pure half-space has no thickness, and ``linspace(0, 0, n)``
            # would return n identical depths. 1 m keeps the grid non-degenerate
            # without changing the samples: with no layers every depth resolves
            # to the half-space anyway.
            max_thick = 1.0
        sample_depths = np.linspace(0, max_thick, n_points)
        cp = np.empty(n_points)
        rho = np.empty(n_points)
        attn = np.empty(n_points)
        for i, d in enumerate(sample_depths):
            layer = self._layer_at(d)
            src = layer if layer is not None else self.halfspace
            cp[i], rho[i], attn[i] = (src.sound_speed, src.density,
                                      src.attenuation)
        return cp, rho, attn

    @classmethod
    def from_halfspace(
        cls,
        halfspace: BoundaryProperties,
        *,
        water_depth: Optional[float] = None,
        sediment_thickness: Optional[float] = None,
        sediment_fraction: float = 0.10,
        min_thickness: float = 5.0,
        synthesize: bool = False,
    ) -> 'SeabedColumn':
        """Build a column from a half-space.

        ``synthesize=False`` (default) → a pure half-space (0 layers).
        ``synthesize=True`` → a single synthetic sediment layer carrying the
        half-space properties (thickness from ``sediment_thickness`` or
        ``sediment_fraction*water_depth`` clamped to ``min_thickness``), as
        the RAM PE update requires a layer above the half-space."""
        if not synthesize:
            return cls(layers=[], halfspace=_copy.deepcopy(halfspace))
        if sediment_thickness is None:
            base = float(water_depth) if water_depth is not None else 0.0
            sediment_thickness = max(float(sediment_fraction) * base,
                                     float(min_thickness))
        layer = SedimentLayer(
            thickness=float(sediment_thickness),
            sound_speed=float(halfspace.sound_speed),
            density=float(halfspace.density),
            attenuation=float(halfspace.attenuation),
            shear_speed=float(getattr(halfspace, 'shear_speed', 0.0) or 0.0),
            shear_attenuation=float(
                getattr(halfspace, 'shear_attenuation', 0.0) or 0.0),
        )
        return cls(layers=[layer], halfspace=_copy.deepcopy(halfspace))

    @classmethod
    def from_presets(
        cls,
        layers: List[Tuple],
        *,
        halfspace: str,
        halfspace_overrides: Optional[Dict] = None,
        elastic: bool = False,
    ) -> 'SeabedColumn':
        """Build a column from :mod:`uacpy.core.materials` presets. Each
        ``layers`` entry is ``(name, thickness)`` or
        ``(name, thickness, overrides)``; ``halfspace`` is a preset name."""
        sediment_layers = []
        for entry in layers:
            if len(entry) == 2:
                name, thickness = entry
                overrides = {}
            elif len(entry) == 3:
                name, thickness, overrides = entry
            else:
                raise ConfigurationError(
                    "SeabedColumn.from_presets: layer entry must be "
                    f"(name, thickness) or (name, thickness, overrides); "
                    f"got {entry!r}"
                )
            sediment_layers.append(
                SedimentLayer.from_preset(name, thickness=thickness,
                                          elastic=elastic, **overrides)
            )
        hs = BoundaryProperties.from_preset(
            halfspace, elastic=elastic, **(halfspace_overrides or {}))
        return cls(layers=sediment_layers, halfspace=hs)


# eq=False: a dataclass __eq__ over ndarray fields raises; compare by identity.
@dataclass(eq=False)
class Bottom:
    """Unified seabed carrier — one or more :class:`SeabedColumn` columns with
    an optional ``ranges`` axis (metres). ``ranges=None`` ⇒ range-independent
    (exactly one column), mirroring ``SoundSpeedProfile(ranges=None)``.

    Construct via the ``from_*`` factories or let ``Environment(bottom=...)``
    coerce a scalar cp, preset name, ``BoundaryProperties``, ``SeabedColumn``
    or ``Bottom``.

    `Bottom` and :class:`~uacpy.core.surface.Surface` are the two carriers
    without a ``.plot()``: placing the sub-bottom depth axis needs a seafloor
    depth, which lives on ``env.bathymetry``. Plot one with
    ``uacpy.plots.plot_bottom_properties(env)``.

    Every accessor — :meth:`at`, :meth:`isel`, :meth:`halfspace_at` — and
    every reduction — :meth:`select_range`, :meth:`collapse`,
    :meth:`to_halfspace` — returns a **copy**, so a result is always safe to
    mutate and never writes back through to the carrier. Reach through
    ``.columns`` to edit in place, or through
    ``.columns[bottom.column_index_at(range=r)]`` to *read* the nearest
    column without paying for a copy.
    """
    columns: List[SeabedColumn]
    ranges: Optional[np.ndarray] = None

    def __post_init__(self):
        self.columns = list(self.columns)
        if not self.columns:
            raise ConfigurationError("Bottom: requires at least one SeabedColumn")
        for c in self.columns:
            if not isinstance(c, SeabedColumn):
                raise ConfigurationError(
                    "Bottom: columns must be SeabedColumn instances; got "
                    f"{type(c).__name__}")
        if self.ranges is None:
            if len(self.columns) != 1:
                raise ConfigurationError(
                    "Bottom: range-independent bottom (ranges=None) needs "
                    f"exactly one column; got {len(self.columns)}")
        else:
            self.ranges = np.array(self.ranges, dtype=float).ravel()
            _require_non_negative(self.ranges, "Bottom.ranges", hint="metres")
            _require_strictly_increasing(
                self.ranges, "Bottom.ranges", min_step=DECK_RANGE_RESOLUTION_M)
            if len(self.ranges) != len(self.columns):
                raise ConfigurationError(
                    f"Bottom: ranges length ({len(self.ranges)}) must match "
                    f"columns length ({len(self.columns)})")

    # ── queries (replace isinstance dispatch) ──────────────────────────────
    @property
    def data_sources(self) -> tuple:
        """Aggregated provenance across all columns, de-duplicated by source id
        (harmonised with the leaf carriers and ``env.data_sources``)."""
        return _dedupe_provenance(self.columns)

    @property
    def is_range_dependent(self) -> bool:
        """True when the bottom carries more than one range column.

        A structural test (node count on the ranged axis), like
        ``SoundSpeedProfile`` / ``Surface``: columns with identical
        properties still count as range-dependent. Contrast
        ``Bathymetry.is_range_dependent`` / ``Altimetry.is_range_dependent``,
        which test whether the *values* actually vary with range."""
        return self.ranges is not None and len(self.columns) > 1

    @property
    def is_layered(self) -> bool:
        return any(c.is_layered for c in self.columns)

    @property
    def is_elastic(self) -> bool:
        for c in self.columns:
            if c.halfspace.shear_speed > 0 or any(
                    la.shear_speed > 0 for la in c.layers):
                return True
        return False

    @property
    def n_ranges(self) -> int:
        return len(self.columns)

    @property
    def acoustic_type(self) -> str:
        return self.columns[0].halfspace.acoustic_type

    def __repr__(self) -> str:
        if not self.is_range_dependent:
            return f"Bottom({self.columns[0]!r})"
        r_lo = float(self.ranges[0]) / 1000
        r_hi = float(self.ranges[-1]) / 1000
        kind = "layered" if self.is_layered else "half-space"
        return (f"Bottom(range-dependent {kind}, n={len(self.columns)}, "
                f"range=[{r_lo:g}, {r_hi:g}] km)")

    # ── slicing ─────────────────────────────────────────────────────────────
    def _nearest_index(self, range: float) -> int:
        return _nearest_index_on_axis(self.ranges, range)

    def at(self, *, range: float) -> SeabedColumn:
        """Copy of the nearest :class:`SeabedColumn` to ``range`` (m).

        Always nearest — layer stacks cannot be linearly blended, so a `Bottom`
        has no general ``eval`` (unlike ``SoundSpeedProfile``/``Field``). For
        the one blendable quantity use :meth:`halfspace_at` (interpolates the
        half-space when every column is a pure half-space). Positional
        counterpart: :meth:`isel`.

        ``range`` must be a finite scalar — a NaN/inf or array-valued label
        raises ``ConfigurationError``, the same contract ``Field.at`` applies.
        """
        return _copy.deepcopy(self.columns[self._nearest_index(range)])

    def column_index_at(self, *, range: float) -> int:
        """Index into :attr:`columns` of the column nearest ``range`` (m) — the
        read-only counterpart of :meth:`at`.

        :meth:`at` deep-copies the column it answers with, so a caller that
        reads one or two fields per query pays for a whole layer stack it then
        drops. This hands back the index instead, and the caller reads the
        carrier's own column::

            column = bottom.columns[bottom.column_index_at(range=r)]

        That column is **live** — mutating it edits the carrier, exactly as
        ``.columns`` already documents. Use :meth:`at` for anything that will
        be modified or handed on; this is for reading only.

        Same nearest rule and same finite-scalar ``range`` contract as
        :meth:`at`, so which column a query resolves to stays this class's to
        decide rather than the caller's to re-derive.
        """
        return self._nearest_index(range)

    def isel(self, *, range: int) -> SeabedColumn:
        """Copy of the :class:`SeabedColumn` at integer position ``range`` —
        the positional counterpart of :meth:`at`."""
        i = int(range)
        n = len(self.columns)
        if not -n <= i < n:
            raise IndexError(
                f"Bottom.isel: range index {i} out of range for "
                f"{n} column(s)")
        return _copy.deepcopy(self.columns[i])

    def halfspace_at(self, *, range: float,
                     interp: Optional[str] = None) -> BoundaryProperties:
        """Half-space ``BoundaryProperties`` at ``range`` (m). ``interp=None``
        auto-resolves: **linear** when every column is a pure half-space
        (the only case where blending properties is well-defined), else
        **nearest**. A blend takes the non-blendable fields (``acoustic_type``,
        ``reflection_file``, ``grain_size_phi``) from the r = 0 column; a
        nearest lookup returns that column's half-space intact. ``range`` must
        be a finite scalar on both paths."""
        if interp not in (None, 'linear', 'nearest'):
            raise ConfigurationError(
                f"Bottom.halfspace_at: interp must be 'linear', 'nearest' or "
                f"None; got {interp!r}")
        # Checked here as well as in ``_nearest_index`` because the blend below
        # never reaches that helper: ``np.interp`` would carry a NaN range into
        # every blended property, and the error then names the stored density
        # rather than the label the caller passed.
        label = _as_finite_scalar_label(range, 'range')
        # Blending is well-defined only when every column is a genuine
        # 'half-space' carrying real acoustic numbers: a vacuum/rigid/file
        # column holds construction-time placeholders, and interpolating
        # those (with column 0's type stamped on the result) fabricates a
        # boundary that exists nowhere on the axis.
        blendable = (not self.is_layered and all(
            c.halfspace.acoustic_type == 'half-space' for c in self.columns))
        if interp is None:
            interp = 'linear' if blendable else 'nearest'
        if self.ranges is None or interp == 'nearest':
            return _copy.deepcopy(
                self.columns[self._nearest_index(range)].halfspace)
        if not blendable:
            types = sorted({c.halfspace.acoustic_type for c in self.columns})
            raise ConfigurationError(
                f"Bottom.halfspace_at(interp='linear') needs every column to "
                f"be a pure 'half-space' to blend; got {types}"
                f"{' with sediment layers' if self.is_layered else ''}. Use "
                f"interp='nearest' (the default here) to read the nearest "
                f"column intact.")
        return _reduce_boundaries(
            [c.halfspace for c in self.columns],
            lambda values: np.interp(label, self.ranges, values))

    def max_total_thickness(self) -> float:
        """Maximum sediment thickness across all columns (0 if all half-space)."""
        return max(c.total_thickness() for c in self.columns)

    def all_sound_speeds(self) -> List[float]:
        """Every *real* compressional speed in the seabed (all layers, plus the
        half-spaces that carry geoacoustics), for c₀ / grid sizing.

        ``'vacuum'`` / ``'rigid'`` / ``'file'`` / ``'precalc'`` half-spaces are
        skipped: their ``sound_speed`` is the placeholder ``__post_init__``
        resolved, not a seabed speed, and feeding it to a c₀ or dz estimate is
        meaningless."""
        speeds: List[float] = []
        for c in self.columns:
            speeds.extend(float(la.sound_speed) for la in c.layers)
            if c.halfspace.acoustic_type not in _NON_GEOACOUSTIC_TYPES:
                speeds.append(float(c.halfspace.sound_speed))
        return speeds

    # ── SoA half-space views (one value per column) ─────────────────────────
    @property
    def halfspace_sound_speed(self) -> np.ndarray:
        return np.array([c.halfspace.sound_speed for c in self.columns])

    @property
    def halfspace_density(self) -> np.ndarray:
        return np.array([c.halfspace.density for c in self.columns])

    @property
    def halfspace_attenuation(self) -> np.ndarray:
        return np.array([c.halfspace.attenuation for c in self.columns])

    @property
    def halfspace_shear_speed(self) -> np.ndarray:
        return np.array([c.halfspace.shear_speed for c in self.columns])

    @property
    def halfspace_shear_attenuation(self) -> np.ndarray:
        return np.array([c.halfspace.shear_attenuation for c in self.columns])

    @property
    def halfspace_roughness(self) -> np.ndarray:
        return np.array([c.halfspace.roughness for c in self.columns])

    # ── reductions ──────────────────────────────────────────────────────────
    def select_range(self, method: str = 'r0') -> 'Bottom':
        """Reduce the range axis to a single column (range-independent result).

        ``'r0'`` / ``'rmax'`` pick the first / last column (layers kept).
        ``'mean'`` / ``'median'`` numerically average the half-spaces — only
        meaningful when no column is layered. For a layered bottom ``'median'``
        falls back to picking the middle column (layers can't be averaged) and
        ``'mean'`` is rejected. Uniform ``'file'`` / ``'precalc'`` columns
        carry no real numbers to average: they collapse to their shared
        reflection file (reducing only the roughness), and differing files
        are rejected.

        The picking methods return a **copy** of the chosen column, matching
        :meth:`at` / :meth:`isel` / :meth:`halfspace_at`; the averaging ones
        build a new half-space and never held the parent's to begin with."""
        if not self.is_range_dependent:
            # Nothing to reduce: one column in, one column out. A single-node
            # ``ranges`` is a coordinate at that range (``from_halfspaces``
            # keeps it, and ``env.max_range`` reads it), so it travels with
            # the column instead of being dropped here.
            return Bottom(columns=[_copy.deepcopy(self.columns[0])],
                          ranges=self.ranges)
        if method == 'r0':
            return Bottom(columns=[_copy.deepcopy(self.columns[0])],
                          ranges=None)
        if method == 'rmax':
            return Bottom(columns=[_copy.deepcopy(self.columns[-1])],
                          ranges=None)
        if method not in ('mean', 'median'):
            raise ConfigurationError(
                f"Bottom.select_range: unknown method={method!r}; "
                "valid: 'r0', 'rmax', 'mean', 'median'")
        if self.is_layered:
            if method == 'median':
                return Bottom(
                    columns=[_copy.deepcopy(
                        self.columns[len(self.columns) // 2])],
                    ranges=None)
            raise ConfigurationError(
                "Bottom.select_range('mean') is undefined for a layered "
                "bottom (layer stacks can't be averaged); use 'r0', 'rmax' "
                "or 'median'.")
        # Averaging is only meaningful within one boundary type — the same
        # guard Surface.collapse applies: reducing a vacuum node with a sand
        # half-space would fold construction-time placeholders into the
        # numbers and stamp one node's type on the result.
        types = {c.halfspace.acoustic_type for c in self.columns}
        if len(types) > 1:
            raise ConfigurationError(
                f"Bottom.select_range({method!r}) needs a single boundary "
                f"type to average; got {sorted(types)}. Boundary types "
                f"cannot be blended — use 'r0' or 'rmax'.")
        reduce = np.mean if method == 'mean' else np.median
        # A uniform 'file'/'precalc' axis carries no real numbers to reduce —
        # each column is its reflection-coefficient table. Columns sharing one
        # table collapse to that shared spec (roughness, the one genuine
        # number they carry, is still reduced); distinct tables cannot be
        # averaged into anything.
        (the_type,) = types
        if the_type in ('file', 'precalc'):
            specs = {c.halfspace.reflection_file for c in self.columns}
            if len(specs) > 1:
                raise ConfigurationError(
                    f"Bottom.select_range({method!r}) cannot average "
                    f"'{the_type}' columns with different reflection files "
                    f"({sorted(specs, key=str)}). Reflection-coefficient "
                    f"tables cannot be blended — use 'r0' or 'rmax'.")
            shared = _copy.deepcopy(self.columns[0].halfspace)
            shared.roughness = float(
                reduce([c.halfspace.roughness for c in self.columns]))
            return Bottom(columns=[SeabedColumn(layers=[], halfspace=shared)],
                          ranges=None)
        return Bottom(columns=[SeabedColumn(
            layers=[],
            halfspace=_reduce_boundaries(
                [c.halfspace for c in self.columns], reduce))], ranges=None)

    def collapse(self, *, range: Optional[str] = None,
                 layers: Optional[str] = None) -> 'Bottom':
        """Reduce along one or both axes, returning a new ``Bottom``.

        ``range=`` collapses the range axis (see :meth:`select_range`).
        ``layers=`` flattens each column's layers to a half-space (per-column,
        keeping the range axis), via :meth:`SeabedColumn.collapse`."""
        b = self
        if range is not None:
            b = b.select_range(range)
        if layers is not None:
            new_cols = [SeabedColumn(layers=[], halfspace=c.collapse(layers))
                        for c in b.columns]
            # One flattened column per input column, so the range axis is
            # untouched — including a single-node ``ranges``, which is a
            # coordinate ``env.max_range`` reads rather than an empty axis.
            b = Bottom(columns=new_cols, ranges=b.ranges)
        return b

    def to_halfspace(self, range_method: str = 'r0') -> BoundaryProperties:
        """Collapse fully to a single ``BoundaryProperties``."""
        return self.select_range(range_method).columns[0].collapse('halfspace')

    def copy(self) -> 'Bottom':
        """Deep copy (symmetric with the other carriers)."""
        return _copy.deepcopy(self)

    def __setattr__(self, name, value):
        # Writes to a half-space field follow through to every column. A plain
        # assignment would create an instance attribute that echoes the new
        # value back while ``halfspace_at()`` — and every writer and model
        # reading it — kept the stored half-spaces.
        if name in _HALFSPACE_DELEGATED and 'columns' in self.__dict__:
            halfspaces = [c.halfspace for c in self.columns]
            value = _validate_halfspace_write(
                type(self).__name__, name, value, halfspaces,
                any(c.layers for c in self.columns))
            if len(self.columns) > 1:
                warnings.warn(
                    f"Bottom.{name} = {value!r} sets all {len(self.columns)} "
                    f"range columns to the same value, flattening any range "
                    f"dependence. Assign to .columns[i].halfspace.{name} to "
                    f"write a single column.",
                    UserWarning, stacklevel=2)
            for halfspace in halfspaces:
                setattr(halfspace, name, value)
            return
        super().__setattr__(name, value)

    # ── factories (mirror SoundSpeedProfile.from_*) ─────────────────────────
    @classmethod
    def from_halfspace(cls, halfspace: BoundaryProperties) -> 'Bottom':
        """Range-independent pure half-space bottom."""
        return cls(columns=[SeabedColumn(layers=[], halfspace=halfspace)],
                   ranges=None)

    @classmethod
    def from_grain_size(cls, grain_size_phi: float, *, model: str = 'hamilton',
                        roughness: float = 0.0,
                        water_sound_speed: Optional[float] = None,
                        water_density: Optional[float] = None) -> 'Bottom':
        """Range-independent half-space bottom from a mean grain size (ϕ).

        Convenience wrapper over
        :meth:`BoundaryProperties.from_grain_size`."""
        return cls.from_halfspace(BoundaryProperties.from_grain_size(
            grain_size_phi, model=model, roughness=roughness,
            water_sound_speed=water_sound_speed, water_density=water_density))

    @classmethod
    def from_column(cls, column: SeabedColumn) -> 'Bottom':
        """Range-independent bottom from a single column."""
        return cls(columns=[column], ranges=None)

    @classmethod
    def from_columns(cls, columns: List[SeabedColumn],
                     ranges) -> 'Bottom':
        """Range-dependent bottom from one column per range break."""
        return cls(columns=list(columns), ranges=ranges)

    @classmethod
    def from_halfspaces(
        cls,
        ranges,
        *,
        sound_speed,
        density,
        attenuation,
        shear_speed=None,
        shear_attenuation=None,
        roughness=None,
        acoustic_type: Optional[str] = None,
    ) -> 'Bottom':
        """Range-dependent half-space bottom from parallel property arrays.

        Every property is either a scalar applied to every range break or a
        per-range array of the same length as ``ranges``. ``shear_speed`` /
        ``shear_attenuation`` / ``roughness`` default to 0."""
        # RD bottoms always carry user cp/ρ/α, so 'half-space' is the coherent
        # default — never infer vacuum just because a sample happens to equal
        # the BoundaryProperties defaults.
        if acoustic_type is None:
            acoustic_type = 'half-space'
        ranges = np.asarray(ranges, dtype=float).ravel()
        n = len(ranges)

        def _per_range(name, value, default=None):
            if value is None:
                if default is None:
                    raise ConfigurationError(
                        f"Bottom.from_halfspaces: {name} is required")
                return np.full(n, float(default))
            arr = np.asarray(value, dtype=float).ravel()
            if arr.size == 1:
                return np.full(n, arr[0])
            if arr.size != n:
                raise ConfigurationError(
                    f"Bottom.from_halfspaces: {name} length ({arr.size}) must "
                    f"match ranges length ({n})")
            return arr

        cp = _per_range('sound_speed', sound_speed)
        rho = _per_range('density', density)
        alpha = _per_range('attenuation', attenuation)
        cs = _per_range('shear_speed', shear_speed, 0.0)
        a_s = _per_range('shear_attenuation', shear_attenuation, 0.0)
        rough = _per_range('roughness', roughness, 0.0)
        columns = [
            SeabedColumn(layers=[], halfspace=BoundaryProperties(
                acoustic_type=acoustic_type, sound_speed=float(cp[i]),
                density=float(rho[i]), attenuation=float(alpha[i]),
                shear_speed=float(cs[i]), shear_attenuation=float(a_s[i]),
                roughness=float(rough[i])))
            for i in range(n)
        ]
        # The caller's ranges are kept even for a single break: ranges=[r]
        # is a coordinate at range r (it feeds env.max_range), while
        # ``is_range_dependent`` — a >1-node test — stays False for it.
        return cls(columns=columns, ranges=ranges)

    @classmethod
    def from_presets(cls, layers, *, halfspace, halfspace_overrides=None,
                     elastic: bool = False) -> 'Bottom':
        """Range-independent layered bottom from material presets."""
        return cls.from_column(SeabedColumn.from_presets(
            layers, halfspace=halfspace,
            halfspace_overrides=halfspace_overrides, elastic=elastic))
