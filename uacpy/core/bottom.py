"""Seafloor / boundary carriers: sediment layers, half-space and layered
bottom properties, and their range-dependent variants. Split out of
:mod:`uacpy.core.environment`; re-exported from there for stable import paths.
"""

import copy as _copy
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, fields as dataclass_fields

from uacpy.core.exceptions import ConfigurationError
from uacpy.core._carrier_validate import (
    _validate_acoustic_type, _require_strictly_increasing,
)


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

    def __post_init__(self):
        # isfinite first: NaN/inf pass every plain ``<= 0`` / ``< 0`` test.
        if not np.isfinite(self.thickness) or self.thickness <= 0:
            raise ConfigurationError(f"SedimentLayer: thickness must be finite and positive (m); got {self.thickness}")
        if not np.isfinite(self.sound_speed) or self.sound_speed <= 0:
            raise ConfigurationError(f"SedimentLayer: sound_speed must be finite and positive (m/s); got {self.sound_speed}")
        if not np.isfinite(self.density) or self.density <= 0:
            raise ConfigurationError(f"SedimentLayer: density must be finite and positive (g/cm^3); got {self.density}")
        for name in ('attenuation', 'shear_speed', 'shear_attenuation'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ConfigurationError(
                    f"SedimentLayer: {name} must be finite and non-negative; got {value}")

    def __repr__(self) -> str:
        bits = [
            f"thickness={self.thickness:g} m",
            f"cp={self.sound_speed:g} m/s",
            f"ρ={self.density:g}",
            f"α={self.attenuation:g}",
        ]
        if self.shear_speed > 0:
            bits.append(f"cs={self.shear_speed:g} m/s")
        return f"SedimentLayer({', '.join(bits)})"

    @classmethod
    def from_preset(cls, name: str, *, thickness: float, elastic: bool = False,
                    **overrides) -> "SedimentLayer":
        """Build a :class:`SedimentLayer` from a :mod:`uacpy.core.materials`
        preset (``'sand'``, ``'silt'``, ``'clay'``, …).

        ``thickness`` is required (presets only encode acoustic
        properties, not layer geometry). The layer is **fluid by default**;
        pass ``elastic=True`` to keep the preset's shear properties. Any
        additional kwargs override the preset's ``sound_speed`` /
        ``density`` / ``attenuation`` / ``shear_*`` for site-specific tuning.
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
        )
        kwargs.update(overrides)
        return cls(**kwargs)


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
        Boundary type: 'vacuum', 'rigid', 'half-space', 'file'.
        Inferred from the supplied parameters when omitted: ``reflection_file``
        → ``'file'``, any non-default cp/ρ/α/cs/roughness → ``'half-space'``,
        nothing → ``'vacuum'``. Pass ``acoustic_type='rigid'`` explicitly (a
        parameter-free physical model). To build a bottom from a grain size,
        use :meth:`from_grain_size` — there is no ``'grain-size'`` type.
    density : float
        Density (g/cm³)
    sound_speed : float
        Compressional wave speed (m/s)
    attenuation : float
        Compressional attenuation (dB/wavelength)
    roughness : float
        RMS roughness (m)
    shear_speed : float
        Shear wave speed (m/s), 0 = fluid bottom
    shear_attenuation : float
        Shear attenuation (dB/wavelength)
    grain_size_phi : float
        Mean grain size in Wentworth phi units. Informational metadata only
        (e.g. set by :meth:`from_grain_size`); it does not by itself define a
        bottom — see :meth:`from_grain_size`.
    reflection_file : str, optional
        Path to reflection coefficient file (.brc for bottom, .trc for top)
        Used when acoustic_type='file'. Can be generated by BOUNCE or OASR.
        Phase-velocity sampling bounds and range stride are carried by the
        consuming model (e.g. ``Kraken(c_low=…, c_high=…)``), not by this
        object.

    Examples
    --------
    Using pre-computed reflection coefficients from BOUNCE:

    >>> # First, compute reflection coefficients
    >>> from uacpy.models import Bounce
    >>> bounce = Bounce(work_dir='./bounce_out')
    >>> result = bounce.run(env, source, receiver)
    >>> brc_file = result.metadata['brc_file']
    >>>
    >>> # Then use in Bellhop/Kraken/Scooter
    >>> bottom = BoundaryProperties(
    ...     acoustic_type='file',
    ...     reflection_file=brc_file
    ... )
    >>> env = Environment(name="test", bathymetry=100, bottom=bottom)
    """

    acoustic_type: Optional[str] = None
    density: float = 1.5
    sound_speed: float = 1600.0
    attenuation: float = 0.5
    roughness: float = 0.0
    shear_speed: float = 0.0
    shear_attenuation: float = 0.0
    grain_size_phi: Optional[float] = None
    reflection_file: Optional[str] = None

    def __post_init__(self):
        # isfinite first: NaN/inf pass every plain ``<= 0`` / ``< 0`` test.
        if not np.isfinite(self.density) or self.density <= 0:
            raise ConfigurationError(
                f"BoundaryProperties: density must be finite and positive (g/cm^3); got {self.density}"
            )
        for name in ('sound_speed', 'attenuation', 'shear_speed',
                     'shear_attenuation'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ConfigurationError(
                    f"BoundaryProperties: {name} must be finite and "
                    f"non-negative; got {value}"
                )

        # Detect which acoustic params differ from their dataclass defaults
        # (read from the field definitions so a default change cannot
        # silently break the inference). We use this both for
        # auto-inference (when acoustic_type is None) and for the
        # explicit-conflict guard below.
        defaults = {f.name: f.default for f in dataclass_fields(self)}
        half_space_offenders = [
            f"{name}={getattr(self, name):g}"
            for name in ('sound_speed', 'density', 'attenuation',
                         'shear_speed', 'shear_attenuation', 'roughness')
            if getattr(self, name) != defaults[name]
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

        # Explicit-conflict guard: vacuum/rigid ignore half-space params,
        # so explicitly setting one alongside non-default cp/ρ/α/cs is a
        # mistake the auto-infer path would never make.
        if self.acoustic_type in ('vacuum', 'rigid'):
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
        if self.acoustic_type in ('vacuum', 'rigid'):
            return f"BoundaryProperties({self.acoustic_type})"
        if self.acoustic_type == 'file':
            return (
                f"BoundaryProperties(file={self.reflection_file!r})"
            )
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
        )
        if m['grain_size_phi'] is not None:
            kwargs['grain_size_phi'] = m['grain_size_phi']
        kwargs.update(overrides)
        return cls(**kwargs)


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
    """
    layers: List[SedimentLayer]
    halfspace: BoundaryProperties

    def __post_init__(self):
        if self.layers is None:
            self.layers = []
        self.layers = list(self.layers)
        if not isinstance(self.halfspace, BoundaryProperties):
            raise ConfigurationError(
                "SeabedColumn: halfspace must be a BoundaryProperties; "
                f"got {type(self.halfspace).__name__}"
            )

    @property
    def is_layered(self) -> bool:
        return len(self.layers) > 0

    def __repr__(self) -> str:
        bits = [f"n_layers={len(self.layers)}"]
        if self.layers:
            bits.append(f"thickness={self.total_thickness():g} m")
        if any(layer.shear_speed > 0 for layer in self.layers) \
                or self.halfspace.shear_speed > 0:
            bits.append("elastic")
        bits.append(f"halfspace={self.halfspace.acoustic_type}")
        if self.halfspace.acoustic_type not in ('vacuum', 'rigid', 'file'):
            bits.append(f"cp={self.halfspace.sound_speed:g} m/s")
        return f"SeabedColumn({', '.join(bits)})"

    def total_thickness(self) -> float:
        """Total thickness of all sediment layers (m); 0 for a half-space."""
        return sum(layer.thickness for layer in self.layers)

    def copy(self) -> 'SeabedColumn':
        """Deep copy (symmetric with the other carriers)."""
        return _copy.deepcopy(self)

    def _layer_at(self, depth: float) -> Optional[SedimentLayer]:
        """Internal: the :class:`SedimentLayer` containing sub-bottom ``depth``
        (m, ``0`` = top of the column), or ``None`` for the deep half-space.

        A private helper — the public accessors are the carrier contract
        :meth:`at` (depth → material) and :meth:`isel` (index → layer); this
        just single-sources the layer-boundary convention they share with
        :meth:`sample_at_depths` (a depth exactly on an internal boundary maps
        to the **upper** layer).
        """
        z = float(depth)
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
        """
        layer = self._layer_at(depth)
        if layer is None:
            return _copy.deepcopy(self.halfspace)
        return BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=layer.sound_speed, density=layer.density,
            attenuation=layer.attenuation,
            shear_speed=layer.shear_speed,
            shear_attenuation=layer.shear_attenuation)

    def isel(self, *, layer: int) -> SedimentLayer:
        """The :class:`SedimentLayer` at integer index ``layer`` — the
        positional counterpart of :meth:`at`. (The deep half-space is
        ``self.halfspace``.)"""
        i = int(layer)
        n = len(self.layers)
        if not -n <= i < n:
            raise IndexError(
                f"SeabedColumn.isel: layer index {i} out of range for "
                f"{n} layer(s)")
        return self.layers[i]

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
        ``'volume_average'`` → thickness-weighted mean (half-space alone when
        there are no layers)."""
        if method == 'halfspace' or not self.layers:
            if method not in ('halfspace', 'top_layer', 'volume_average'):
                raise ConfigurationError(
                    f"SeabedColumn.collapse: unknown method={method!r}; "
                    "valid: 'halfspace', 'top_layer', 'volume_average'"
                )
            return _copy.deepcopy(self.halfspace)
        if method == 'top_layer':
            top = self.layers[0]
            return BoundaryProperties(
                acoustic_type=self.halfspace.acoustic_type,
                density=top.density, sound_speed=top.sound_speed,
                attenuation=top.attenuation, shear_speed=top.shear_speed,
                shear_attenuation=top.shear_attenuation,
                roughness=self.halfspace.roughness,
            )
        if method == 'volume_average':
            # Thickness-weighted mean over the finite layers plus the
            # half-space. The semi-infinite half-space has no finite thickness,
            # so it is weighted by the deepest layer's thickness — a pragmatic
            # choice that keeps it comparable to the adjacent layer rather than
            # dominating (∞ weight) or vanishing (0). (Layers are non-empty
            # here; the no-layer case returned the half-space above.)
            weights = np.array([float(la.thickness) for la in self.layers])
            weights = np.append(weights, float(weights[-1]))

            def _avg(field):
                vals = np.array([getattr(la, field) for la in self.layers]
                                + [getattr(self.halfspace, field)])
                return float(np.average(vals, weights=weights))
            return BoundaryProperties(
                acoustic_type=self.halfspace.acoustic_type,
                sound_speed=_avg('sound_speed'), density=_avg('density'),
                attenuation=_avg('attenuation'), shear_speed=_avg('shear_speed'),
                shear_attenuation=_avg('shear_attenuation'),
                roughness=self.halfspace.roughness,
            )
        raise ConfigurationError(
            f"SeabedColumn.collapse: unknown method={method!r}; "
            "valid: 'halfspace', 'top_layer', 'volume_average'"
        )

    def sample_at_depths(
        self, n_points: int = 4, max_thickness: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample ``(cs, rho, attn)`` at ``n_points`` evenly-spaced depths in
        ``[0, max_thickness]`` (defaults to this column's own thickness). Used
        by RAM to map arbitrary layers onto its fixed sediment grid."""
        max_thick = (float(max_thickness) if max_thickness is not None
                     else self.total_thickness())
        if max_thick <= 0:
            max_thick = 1.0
        sample_depths = np.linspace(0, max_thick, n_points)
        cs = np.empty(n_points)
        rho = np.empty(n_points)
        attn = np.empty(n_points)
        for i, d in enumerate(sample_depths):
            layer = self._layer_at(d)
            src = layer if layer is not None else self.halfspace
            cs[i], rho[i], attn[i] = (src.sound_speed, src.density,
                                      src.attenuation)
        return cs, rho, attn

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


@dataclass
class Bottom:
    """Unified seabed carrier — one or more :class:`SeabedColumn` columns with
    an optional ``ranges`` axis (metres). ``ranges=None`` ⇒ range-independent
    (exactly one column), mirroring ``SoundSpeedProfile(ranges=None)``.

    Construct via the ``from_*`` factories or let ``Environment(bottom=...)``
    coerce a scalar cp, preset name, ``BoundaryProperties``, ``SeabedColumn``
    or ``Bottom``.
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
            self.ranges = np.asarray(self.ranges, dtype=float).ravel()
            _require_strictly_increasing(self.ranges, "Bottom.ranges")
            if len(self.ranges) != len(self.columns):
                raise ConfigurationError(
                    f"Bottom: ranges length ({len(self.ranges)}) must match "
                    f"columns length ({len(self.columns)})")

    # ── queries (replace isinstance dispatch) ──────────────────────────────
    @property
    def is_range_dependent(self) -> bool:
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
        if self.ranges is None:
            return 0
        return int(np.argmin(np.abs(self.ranges - float(range))))

    def at(self, *, range: float) -> SeabedColumn:
        """Nearest :class:`SeabedColumn` to ``range`` (m).

        Always nearest — layer stacks cannot be linearly blended, so a `Bottom`
        has no general ``eval`` (unlike ``SoundSpeedProfile``/``Field``). For
        the one blendable quantity use :meth:`halfspace_at` (interpolates the
        half-space when every column is a pure half-space). Positional
        counterpart: :meth:`isel`.
        """
        return self.columns[self._nearest_index(range)]

    def isel(self, *, range: int) -> SeabedColumn:
        """:class:`SeabedColumn` at integer position ``range`` — the positional
        counterpart of :meth:`at`."""
        i = int(range)
        n = len(self.columns)
        if not -n <= i < n:
            raise IndexError(
                f"Bottom.isel: range index {i} out of range for "
                f"{n} column(s)")
        return self.columns[i]

    def halfspace_at(self, *, range: float,
                     interp: Optional[str] = None) -> BoundaryProperties:
        """Half-space ``BoundaryProperties`` at ``range`` (m). ``interp=None``
        auto-resolves: **linear** when every column is a pure half-space
        (the only case where blending properties is well-defined), else
        **nearest**."""
        if interp is None:
            interp = 'nearest' if self.is_layered else 'linear'
        if self.ranges is None or interp == 'nearest':
            return _copy.deepcopy(self.at(range=range).halfspace)
        if interp != 'linear':
            raise ConfigurationError(
                f"Bottom.halfspace_at: interp must be 'linear', 'nearest' or "
                f"None; got {interp!r}")
        hs = [c.halfspace for c in self.columns]
        return BoundaryProperties(
            acoustic_type=hs[0].acoustic_type,
            sound_speed=float(np.interp(range, self.ranges,
                                        [h.sound_speed for h in hs])),
            density=float(np.interp(range, self.ranges,
                                    [h.density for h in hs])),
            attenuation=float(np.interp(range, self.ranges,
                                        [h.attenuation for h in hs])),
            shear_speed=float(np.interp(range, self.ranges,
                                        [h.shear_speed for h in hs])),
            shear_attenuation=float(np.interp(range, self.ranges,
                                              [h.shear_attenuation for h in hs])),
        )

    def max_total_thickness(self) -> float:
        """Maximum sediment thickness across all columns (0 if all half-space)."""
        return max(c.total_thickness() for c in self.columns)

    def all_sound_speeds(self) -> List[float]:
        """Every compressional speed in the seabed (all layers + half-spaces),
        for c₀ / grid sizing."""
        speeds: List[float] = []
        for c in self.columns:
            speeds.extend(float(la.sound_speed) for la in c.layers)
            if getattr(c.halfspace, 'sound_speed', None):
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
        ``'mean'`` is rejected."""
        if not self.is_range_dependent:
            return Bottom(columns=[self.columns[0]], ranges=None)
        if method == 'r0':
            return Bottom(columns=[self.columns[0]], ranges=None)
        if method == 'rmax':
            return Bottom(columns=[self.columns[-1]], ranges=None)
        if method not in ('mean', 'median'):
            raise ConfigurationError(
                f"Bottom.select_range: unknown method={method!r}; "
                "valid: 'r0', 'rmax', 'mean', 'median'")
        if self.is_layered:
            if method == 'median':
                return Bottom(columns=[self.columns[len(self.columns) // 2]],
                              ranges=None)
            raise ConfigurationError(
                "Bottom.select_range('mean') is undefined for a layered "
                "bottom (layer stacks can't be averaged); use 'r0', 'rmax' "
                "or 'median'.")
        reduce = np.mean if method == 'mean' else np.median
        hs0 = self.columns[0].halfspace
        return Bottom(columns=[SeabedColumn(layers=[], halfspace=(
            BoundaryProperties(
                acoustic_type=hs0.acoustic_type,
                sound_speed=float(reduce(self.halfspace_sound_speed)),
                density=float(reduce(self.halfspace_density)),
                attenuation=float(reduce(self.halfspace_attenuation)),
                shear_speed=float(reduce(self.halfspace_shear_speed)),
                shear_attenuation=float(
                    reduce(self.halfspace_shear_attenuation)),
                roughness=float(reduce(self.halfspace_roughness)),
            )))], ranges=None)

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
            b = Bottom(columns=new_cols,
                       ranges=None if len(new_cols) == 1 else b.ranges)
        return b

    def to_halfspace(self, range_method: str = 'r0') -> BoundaryProperties:
        """Collapse fully to a single ``BoundaryProperties``."""
        return self.select_range(range_method).columns[0].collapse('halfspace')

    def copy(self) -> 'Bottom':
        """Deep copy (symmetric with the other carriers)."""
        return _copy.deepcopy(self)

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
        acoustic_type: Optional[str] = None,
    ) -> 'Bottom':
        """Range-dependent half-space bottom from parallel property arrays."""
        # RD bottoms always carry user cp/ρ/α, so 'half-space' is the coherent
        # default — never infer vacuum just because a sample happens to equal
        # the BoundaryProperties defaults.
        if acoustic_type is None:
            acoustic_type = 'half-space'
        ranges = np.asarray(ranges, dtype=float).ravel()
        n = len(ranges)
        cp = np.asarray(sound_speed, dtype=float).ravel()
        rho = np.asarray(density, dtype=float).ravel()
        alpha = np.asarray(attenuation, dtype=float).ravel()
        cs = (np.zeros(n) if shear_speed is None
              else np.asarray(shear_speed, dtype=float).ravel())
        a_s = (np.zeros(n) if shear_attenuation is None
               else np.asarray(shear_attenuation, dtype=float).ravel())
        for name, arr in (('sound_speed', cp), ('density', rho),
                          ('attenuation', alpha), ('shear_speed', cs),
                          ('shear_attenuation', a_s)):
            if len(arr) != n:
                raise ConfigurationError(
                    f"Bottom.from_halfspaces: {name} length ({len(arr)}) must "
                    f"match ranges length ({n})")
        columns = [
            SeabedColumn(layers=[], halfspace=BoundaryProperties(
                acoustic_type=acoustic_type, sound_speed=float(cp[i]),
                density=float(rho[i]), attenuation=float(alpha[i]),
                shear_speed=float(cs[i]), shear_attenuation=float(a_s[i])))
            for i in range(n)
        ]
        return cls(columns=columns, ranges=ranges if n > 1 else None)

    @classmethod
    def from_presets(cls, layers, *, halfspace, halfspace_overrides=None,
                     elastic: bool = False) -> 'Bottom':
        """Range-independent layered bottom from material presets."""
        return cls.from_column(SeabedColumn.from_presets(
            layers, halfspace=halfspace,
            halfspace_overrides=halfspace_overrides, elastic=elastic))
