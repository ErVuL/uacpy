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
        if self.thickness <= 0:
            raise ConfigurationError(f"SedimentLayer: thickness must be positive (m); got {self.thickness}")
        if self.sound_speed <= 0:
            raise ConfigurationError(f"SedimentLayer: sound_speed must be positive (m/s); got {self.sound_speed}")
        if self.density <= 0:
            raise ConfigurationError(f"SedimentLayer: density must be positive (g/cm^3); got {self.density}")
        for name in ('attenuation', 'shear_speed', 'shear_attenuation'):
            value = getattr(self, name)
            if value < 0:
                raise ConfigurationError(
                    f"SedimentLayer: {name} must be non-negative; got {value}")

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
        Boundary type: 'vacuum', 'rigid', 'half-space', 'grain-size', 'file'.
        Inferred from the supplied parameters when omitted: ``reflection_file``
        → ``'file'``, any non-default cp/ρ/α/cs/roughness → ``'half-space'``,
        nothing → ``'vacuum'``. Pass ``acoustic_type='rigid'`` or
        ``'grain-size'`` explicitly — they're physically distinct models.
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
        Mean grain size in phi units (for 'grain-size' type)
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
    grain_size_phi: float = 1.0
    reflection_file: Optional[str] = None

    def __post_init__(self):
        if self.density <= 0:
            raise ConfigurationError(
                f"BoundaryProperties: density must be positive (g/cm^3); got {self.density}"
            )
        if self.sound_speed < 0:
            raise ConfigurationError(
                f"BoundaryProperties: sound_speed must be non-negative (m/s); got {self.sound_speed}"
            )
        if self.attenuation < 0:
            raise ConfigurationError(
                f"BoundaryProperties: attenuation must be non-negative; got {self.attenuation}"
            )
        if self.shear_speed < 0:
            raise ConfigurationError(
                f"BoundaryProperties: shear_speed must be non-negative (m/s); got {self.shear_speed}"
            )
        if self.shear_attenuation < 0:
            raise ConfigurationError(
                f"BoundaryProperties: shear_attenuation must be non-negative; "
                f"got {self.shear_attenuation}"
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
            # Auto-infer from the supplied parameters. 'grain-size' and
            # 'rigid' remain opt-in — they're physically distinct models,
            # not just a parameter pattern.
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
        (OASES, Scooter, KrakenC). ``shear_*`` in ``**overrides`` wins
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


@dataclass
class RangeDependentBottom:
    """
    Range-dependent bottom properties for realistic geoacoustic modeling.

    Allows bottom acoustic properties to vary with range, essential for
    continental shelf transitions, sediment type changes, etc.

    Bathymetry is **not** carried here — it lives on
    ``Environment.bathymetry``. Models that need the seafloor depth at
    one of these range points interpolate ``env.bathymetry`` at
    ``ranges[i]``.

    Attributes
    ----------
    ranges : ndarray
        Range points in **meters**, shape (N,).
    sound_speed : ndarray
        Compressional wave speed at each range (m/s), shape (N,)
    density : ndarray
        Density at each range (g/cm³), shape (N,)
    attenuation : ndarray
        Attenuation at each range (dB/wavelength), shape (N,)
    shear_speed : ndarray, optional
        Shear wave speed at each range (m/s), shape (N,). Default is 0 (fluid).
    shear_attenuation : ndarray, optional
        Shear attenuation at each range (dB/wavelength), shape (N,). Default 0.
    acoustic_type : str, optional
        Boundary type (same at all ranges): 'half-space' (default — inferred
        from the required cp/ρ/α arrays), 'grain-size', 'file'. 'vacuum' /
        'rigid' are rejected because they would discard the supplied arrays.

    Examples
    --------
    Continental shelf transition (sediment hardening with range):

    >>> ranges = np.array([0, 10000, 20000, 30000])  # meters
    >>> sound_speed = np.array([1600, 1650, 1700, 1750])
    >>> density = np.array([1.5, 1.7, 1.9, 2.1])
    >>> attenuation = np.array([0.5, 0.4, 0.3, 0.2])
    >>>
    >>> bottom_rd = RangeDependentBottom(
    ...     ranges=ranges,
    ...     sound_speed=sound_speed,
    ...     density=density,
    ...     attenuation=attenuation,
    ...     shear_speed=np.zeros(4),
    ...     acoustic_type='half-space'
    ... )
    """
    ranges: np.ndarray
    sound_speed: np.ndarray
    density: np.ndarray
    attenuation: np.ndarray
    shear_speed: np.ndarray = None
    shear_attenuation: np.ndarray = None
    acoustic_type: Optional[str] = None

    def __post_init__(self):
        """Validate array lengths and set defaults."""
        # Range-dependent bottoms always carry user-supplied cp/ρ/α arrays,
        # so 'half-space' is the only physically coherent default. 'rigid'
        # and 'grain-size' remain opt-in via explicit ``acoustic_type=``.
        if self.acoustic_type is None:
            self.acoustic_type = 'half-space'
        _validate_acoustic_type(self.acoustic_type, "RangeDependentBottom")
        self.ranges = np.asarray(self.ranges, dtype=float).ravel()
        _require_strictly_increasing(self.ranges, "RangeDependentBottom.ranges")
        n = len(self.ranges)

        if self.shear_speed is None:
            self.shear_speed = np.zeros(n)
        if self.shear_attenuation is None:
            self.shear_attenuation = np.zeros(n)

        # Validate after defaulting so the zeros above stay length-n and only
        # explicitly-passed arrays are length-checked (a mismatched shear array
        # would otherwise raise a bare numpy ValueError later, in eval()).
        for attr_name in ['sound_speed', 'density', 'attenuation',
                          'shear_speed', 'shear_attenuation']:
            attr = getattr(self, attr_name)
            if len(attr) != n:
                raise ConfigurationError(
                    f"RangeDependentBottom: {attr_name} length ({len(attr)}) must "
                    f"match ranges length ({n})"
                )

        # Explicit-conflict guard: vacuum/rigid ignore cp/ρ/α, so pairing
        # them with an RD bottom (which requires those arrays) is wrong.
        if self.acoustic_type in ('vacuum', 'rigid'):
            raise ConfigurationError(
                f"RangeDependentBottom(acoustic_type={self.acoustic_type!r}) "
                f"is incoherent — vacuum/rigid boundaries ignore cp/ρ/α, "
                f"but RangeDependentBottom requires those arrays. Drop "
                f"``acoustic_type=`` to let uacpy infer 'half-space'."
            )

    def __repr__(self) -> str:
        n = len(self.ranges)
        r_lo, r_hi = float(self.ranges[0]) / 1000, float(self.ranges[-1]) / 1000
        c_lo, c_hi = float(np.min(self.sound_speed)), float(np.max(self.sound_speed))
        elastic = " elastic" if np.any(np.asarray(self.shear_speed) > 0) else ""
        return (
            f"RangeDependentBottom({self.acoustic_type}{elastic}, "
            f"n={n}, range=[{r_lo:g}, {r_hi:g}] km, "
            f"cp=[{c_lo:g}, {c_hi:g}] m/s)"
        )

    def at(self, *, range: float, interp: str = 'linear') -> BoundaryProperties:
        """``BoundaryProperties`` at the requested range (m).

        ``interp='linear'`` (default) interpolates between stored samples;
        ``interp='nearest'`` returns the nearest stored sample.
        """
        if interp == 'nearest':
            idx = int(np.argmin(np.abs(self.ranges - range)))
            return BoundaryProperties(
                acoustic_type=self.acoustic_type,
                sound_speed=float(self.sound_speed[idx]),
                density=float(self.density[idx]),
                attenuation=float(self.attenuation[idx]),
                shear_speed=float(self.shear_speed[idx]),
                shear_attenuation=float(self.shear_attenuation[idx]),
            )
        if interp != 'linear':
            raise ConfigurationError(
                f"RangeDependentBottom.at: interp must be 'linear' or "
                f"'nearest'; got {interp!r}"
            )
        ranges = self.ranges
        return BoundaryProperties(
            acoustic_type=self.acoustic_type,
            sound_speed=float(np.interp(range, ranges, self.sound_speed)),
            density=float(np.interp(range, ranges, self.density)),
            attenuation=float(np.interp(range, ranges, self.attenuation)),
            shear_speed=float(np.interp(range, ranges, self.shear_speed)),
            shear_attenuation=float(
                np.interp(range, ranges, self.shear_attenuation)
            ),
        )

    def collapse(self, method: str = 'r0') -> BoundaryProperties:
        """Collapse to a single ``BoundaryProperties`` for models that don't
        support range-dependent bottoms.

        Methods
        -------
        ``'r0'``     : range-0 sample.
        ``'rmax'``   : last (deepest range) sample.
        ``'mean'``   : per-property mean across ranges.
        ``'median'`` : per-property median across ranges.
        """
        if method == 'r0':
            return self.at(range=float(self.ranges[0]))
        if method == 'rmax':
            return self.at(range=float(self.ranges[-1]))
        if method == 'mean':
            reduce = np.mean
        elif method == 'median':
            reduce = np.median
        else:
            raise ConfigurationError(
                f"RangeDependentBottom.collapse: unknown method={method!r}; "
                "valid: 'r0', 'rmax', 'mean', 'median'"
            )
        return BoundaryProperties(
            acoustic_type=self.acoustic_type,
            sound_speed=float(reduce(self.sound_speed)),
            density=float(reduce(self.density)),
            attenuation=float(reduce(self.attenuation)),
            shear_speed=float(reduce(self.shear_speed)),
            shear_attenuation=float(reduce(self.shear_attenuation)),
        )


@dataclass
class LayeredBottom:
    """
    Depth-dependent (layered) sediment structure.

    Defines a stack of sediment layers above a deepest half-space.
    Used by models that support multi-layered bottoms (Kraken, Scooter,
    SPARC via NMEDIA > 1; OASES via layered format).

    Parameters
    ----------
    layers : list of SedimentLayer
        Sediment layers from top (shallowest) to bottom (deepest),
        stacked below the water column.
    halfspace : BoundaryProperties
        Properties of the deepest half-space below all layers.

    Examples
    --------
    Continental shelf with sand over clay over rock:

    >>> from uacpy.core.environment import SedimentLayer, LayeredBottom, BoundaryProperties
    >>> bottom = LayeredBottom(
    ...     layers=[
    ...         SedimentLayer(thickness=10, sound_speed=1550, density=1.3, attenuation=0.5),
    ...         SedimentLayer(thickness=50, sound_speed=1650, density=1.7, attenuation=0.3),
    ...     ],
    ...     halfspace=BoundaryProperties(
    ...         acoustic_type='half-space',
    ...         sound_speed=1800, density=2.0, attenuation=0.1
    ...     )
    ... )
    """
    layers: List[SedimentLayer]
    halfspace: BoundaryProperties

    def __post_init__(self):
        if not self.layers:
            raise ConfigurationError("LayeredBottom: requires at least one SedimentLayer; got 0")

    def __repr__(self) -> str:
        n = len(self.layers)
        thick = self.total_thickness()
        bits = [f"n_layers={n}", f"thickness={thick:g} m"]
        if any(layer.shear_speed > 0 for layer in self.layers):
            bits.append("elastic")
        bits.append(f"halfspace={self.halfspace.acoustic_type}")
        if self.halfspace.acoustic_type not in ('vacuum', 'rigid', 'file'):
            bits.append(f"cp={self.halfspace.sound_speed:g} m/s")
        return f"LayeredBottom({', '.join(bits)})"

    def total_thickness(self) -> float:
        """Total thickness of all sediment layers (m)."""
        return sum(layer.thickness for layer in self.layers)

    def layer_depths(self, seafloor_depth: float) -> List[Tuple[float, float]]:
        """
        Compute (top_depth, bottom_depth) for each layer.

        Parameters
        ----------
        seafloor_depth : float
            Depth of the seafloor (top of first layer) in meters.

        Returns
        -------
        list of (float, float)
            (top_depth, bottom_depth) pairs for each layer.
        """
        depths = []
        current_depth = seafloor_depth
        for layer in self.layers:
            top = current_depth
            bottom = current_depth + layer.thickness
            depths.append((top, bottom))
            current_depth = bottom
        return depths

    def to_piecewise_breakpoints(
        self,
        seafloor_depth: float,
        zmax: Optional[float] = None,
        properties: Tuple[str, ...] = (
            'sound_speed', 'density', 'attenuation',
        ),
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Project this layered bottom onto Collins-style ``(depth, value)``
        breakpoint sequences — the format consumed by ``ram.in`` for
        rams0.5 (elastic) and ramsurf1.5 (rough surface).

        Each layer becomes two breakpoints (top depth, bottom depth) with
        the same value, producing a step function under the linear
        interpolation rules of Collins' ``zread`` routine. The half-space
        is appended as one final breakpoint at ``zmax`` (or at the deepest
        layer bottom if ``zmax`` is omitted) carrying the half-space
        value. When ``zmax`` does not exceed the deepest layer bottom the
        final breakpoint is emitted 1 m below it instead — past the
        physical grid but harmless, since Collins clamps to the last
        breakpoint inside the absorbing layer.

        Parameters
        ----------
        seafloor_depth : float
            Depth of the top of the first sediment layer (m).
        zmax : float, optional
            Maximum depth of the PE computational grid. If provided, a
            final breakpoint is emitted at ``zmax`` with the half-space
            value so the absorbing region carries the right properties.
        properties : tuple of str, optional
            Which fields to extract. Pass e.g. ``('sound_speed', 'density',
            'attenuation', 'shear_speed', 'shear_attenuation')`` for RAMS.
            Layers / half-space that don't expose the property contribute
            ``0.0`` (the convention RAM family uses for "no shear").

        Returns
        -------
        dict
            ``{property_name: [(depth, value), ...]}`` — one list per
            requested property, in increasing depth order.
        """
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
        """Collapse layers to a single ``BoundaryProperties`` for models
        that don't support layered bottoms.

        Methods
        -------
        ``'halfspace'``       : return the deep half-space alone.
        ``'top_layer'``       : return the topmost sediment layer's
                                acoustic properties (with the half-space
                                as fallback for missing fields).
        ``'volume_average'``  : thickness-weighted mean of layer
                                properties; the half-space contributes
                                with weight equal to the deepest layer
                                (a stand-in for "infinite extent").
        """
        if method == 'halfspace':
            return _copy.deepcopy(self.halfspace)
        if method == 'top_layer':
            top = self.layers[0]
            return BoundaryProperties(
                acoustic_type=self.halfspace.acoustic_type,
                density=top.density,
                sound_speed=top.sound_speed,
                attenuation=top.attenuation,
                shear_speed=top.shear_speed,
                shear_attenuation=top.shear_attenuation,
            )
        if method == 'volume_average':
            weights = np.array([float(layer.thickness) for layer in self.layers])
            hs_weight = float(weights[-1]) if weights.size else 1.0
            weights = np.append(weights, hs_weight)
            cs = np.array(
                [layer.sound_speed for layer in self.layers]
                + [self.halfspace.sound_speed]
            )
            rho = np.array(
                [layer.density for layer in self.layers]
                + [self.halfspace.density]
            )
            alpha = np.array(
                [layer.attenuation for layer in self.layers]
                + [self.halfspace.attenuation]
            )
            cs_shear = np.array(
                [layer.shear_speed for layer in self.layers]
                + [self.halfspace.shear_speed]
            )
            alpha_shear = np.array(
                [layer.shear_attenuation for layer in self.layers]
                + [self.halfspace.shear_attenuation]
            )
            return BoundaryProperties(
                acoustic_type=self.halfspace.acoustic_type,
                sound_speed=float(np.average(cs, weights=weights)),
                density=float(np.average(rho, weights=weights)),
                attenuation=float(np.average(alpha, weights=weights)),
                shear_speed=float(np.average(cs_shear, weights=weights)),
                shear_attenuation=float(np.average(alpha_shear, weights=weights)),
            )
        raise ConfigurationError(
            f"LayeredBottom.collapse: unknown method={method!r}; "
            "valid: 'halfspace', 'top_layer', 'volume_average'"
        )

    @classmethod
    def from_halfspace(
        cls,
        halfspace: BoundaryProperties,
        water_depth: float,
        sediment_thickness: Optional[float] = None,
        sediment_fraction: float = 0.10,
        min_thickness: float = 5.0,
    ) -> 'LayeredBottom':
        """Wrap a plain half-space as a synthetic single-layer bottom.

        Used by RAM-family backends (which require a sediment layer
        above the half-space for the PE update). The synthetic layer
        carries the same acoustic properties as the half-space, with a
        thickness derived from ``sediment_fraction * water_depth``
        (clamped to ``min_thickness``) unless ``sediment_thickness`` is
        provided explicitly.
        """
        if sediment_thickness is None:
            sediment_thickness = max(
                float(sediment_fraction) * float(water_depth),
                float(min_thickness),
            )
        layer = SedimentLayer(
            thickness=float(sediment_thickness),
            sound_speed=float(halfspace.sound_speed),
            density=float(halfspace.density),
            attenuation=float(halfspace.attenuation),
            shear_speed=float(getattr(halfspace, 'shear_speed', 0.0) or 0.0),
            shear_attenuation=float(
                getattr(halfspace, 'shear_attenuation', 0.0) or 0.0
            ),
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
    ) -> 'LayeredBottom':
        """Build a stratigraphic stack from :mod:`uacpy.core.materials`
        preset names.

        Parameters
        ----------
        layers : list of tuples
            Each entry is ``(name, thickness)`` or
            ``(name, thickness, overrides)`` where ``overrides`` is a
            dict of per-layer field overrides.
        halfspace : str
            Preset name for the substrate half-space.
        halfspace_overrides : dict, optional
            Field overrides applied to the half-space.
        elastic : bool, optional
            Keep the presets' shear properties (default ``False`` ⇒ fluid).
            Per-layer / half-space ``shear_*`` overrides win regardless.

        Examples
        --------
        >>> LayeredBottom.from_presets(
        ...     layers=[('clay', 5), ('silt', 15), ('sand', 30)],
        ...     halfspace='limestone',
        ... )
        """
        sediment_layers = []
        for entry in layers:
            if len(entry) == 2:
                name, thickness = entry
                overrides = {}
            elif len(entry) == 3:
                name, thickness, overrides = entry
            else:
                raise ConfigurationError(
                    f"LayeredBottom.from_presets: layer entry must be "
                    f"(name, thickness) or (name, thickness, overrides); "
                    f"got {entry!r}"
                )
            sediment_layers.append(
                SedimentLayer.from_preset(name, thickness=thickness,
                                          elastic=elastic, **overrides)
            )
        hs = BoundaryProperties.from_preset(
            halfspace, elastic=elastic, **(halfspace_overrides or {}),
        )
        return cls(layers=sediment_layers, halfspace=hs)


@dataclass
class RangeDependentLayeredBottom:
    """
    Range-dependent layered sediment: a LayeredBottom at each range point.

    Combines range variation (different sediment stacks along the
    propagation path) with depth variation (multiple layers at each
    range).  RAM maps each stack to its 4-point sediment profile;
    AT models (Kraken/Scooter/SPARC) warn because NMEDIA is fixed.

    Bathymetry is **not** carried here — it lives on ``Environment.bathymetry``.
    Models that need the seafloor depth at one of these range points
    interpolate ``env.bathymetry`` at ``ranges[i]``.

    Parameters
    ----------
    ranges : ndarray
        Range points in **meters**, shape (N,).
    profiles : list of LayeredBottom
        One LayeredBottom per range point (length N).

    Examples
    --------
    Mud-over-clay near-shore transitioning to sand-over-rock offshore:

    >>> from uacpy.core.environment import (
    ...     SedimentLayer, LayeredBottom, BoundaryProperties,
    ...     RangeDependentLayeredBottom,
    ... )
    >>> near = LayeredBottom(
    ...     layers=[SedimentLayer(5, 1500, 1.2, 1.0),
    ...             SedimentLayer(15, 1550, 1.4, 0.8)],
    ...     halfspace=BoundaryProperties(acoustic_type='half-space',
    ...                                  sound_speed=1800, density=2.0, attenuation=0.1),
    ... )
    >>> far = LayeredBottom(
    ...     layers=[SedimentLayer(3, 1650, 1.8, 0.3),
    ...             SedimentLayer(10, 1750, 2.0, 0.2)],
    ...     halfspace=BoundaryProperties(acoustic_type='half-space',
    ...                                  sound_speed=2200, density=2.5, attenuation=0.05),
    ... )
    >>> rdl = RangeDependentLayeredBottom(
    ...     ranges=np.array([0, 20000]),  # meters
    ...     profiles=[near, far],
    ... )
    """
    ranges: np.ndarray
    profiles: List[LayeredBottom]

    def __post_init__(self):
        self.ranges = np.asarray(self.ranges, dtype=float).ravel()
        n = len(self.ranges)
        if n < 1:
            raise ConfigurationError(
                "RangeDependentLayeredBottom: at least one range point is required"
            )
        _require_strictly_increasing(
            self.ranges, "RangeDependentLayeredBottom.ranges",
        )
        if len(self.profiles) != n:
            raise ConfigurationError(
                f"RangeDependentLayeredBottom: profiles length ({len(self.profiles)}) "
                f"must match ranges length ({n})"
            )

    def __repr__(self) -> str:
        n = len(self.ranges)
        r_lo = float(self.ranges[0]) / 1000
        r_hi = float(self.ranges[-1]) / 1000
        max_layers = max(len(p.layers) for p in self.profiles)
        return (
            f"RangeDependentLayeredBottom(n_profiles={n}, "
            f"range=[{r_lo:g}, {r_hi:g}] km, "
            f"max_layers={max_layers})"
        )

    def max_total_thickness(self) -> float:
        """Maximum total sediment thickness across all range points."""
        return max(p.total_thickness() for p in self.profiles)

    def at(self, *, range: float) -> 'LayeredBottom':
        """Return the nearest LayeredBottom profile for a given range (m)."""
        idx = int(np.argmin(np.abs(self.ranges - range)))
        return self.profiles[idx]

    def sample_at_depths(
        self,
        profile_idx: int,
        n_points: int = 4,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a LayeredBottom profile at evenly-spaced depth points.

        Returns arrays of (sound_speed, density, attenuation) sampled at
        ``n_points`` depths spanning [0, max_total_thickness].  Used by RAM
        to map arbitrary layers to its fixed 4-point sediment grid.

        Parameters
        ----------
        profile_idx : int
            Index into ``self.profiles``.
        n_points : int
            Number of sample points (default 4, matching RAM).

        Returns
        -------
        cs : ndarray, shape (n_points,)
            Sound speed at each depth sample.
        rho : ndarray, shape (n_points,)
            Density at each depth sample.
        attn : ndarray, shape (n_points,)
            Attenuation at each depth sample.
        """
        lb = self.profiles[profile_idx]
        max_thick = self.max_total_thickness()
        if max_thick <= 0:
            max_thick = 1.0
        sample_depths = np.linspace(0, max_thick, n_points)

        cs = np.empty(n_points)
        rho = np.empty(n_points)
        attn = np.empty(n_points)

        for i, d in enumerate(sample_depths):
            # Walk through layers to find which layer this depth falls in
            cumulative = 0.0
            found = False
            for layer in lb.layers:
                if d <= cumulative + layer.thickness:
                    cs[i] = layer.sound_speed
                    rho[i] = layer.density
                    attn[i] = layer.attenuation
                    found = True
                    break
                cumulative += layer.thickness
            if not found:
                # Below all layers → halfspace
                cs[i] = lb.halfspace.sound_speed
                rho[i] = lb.halfspace.density
                attn[i] = lb.halfspace.attenuation

        return cs, rho, attn

    def to_profile(self, method: str = 'r0') -> 'LayeredBottom':
        """Pick one ``LayeredBottom`` profile from the range axis.

        ``method`` ∈ ``'r0'`` | ``'rmax'`` | ``'median'``.
        """
        if method == 'r0':
            idx = 0
        elif method == 'rmax':
            idx = len(self.profiles) - 1
        elif method == 'median':
            idx = len(self.profiles) // 2
        else:
            raise ConfigurationError(
                f"RangeDependentLayeredBottom.to_profile: unknown "
                f"method={method!r}; valid: 'r0', 'rmax', 'median'"
            )
        return self.profiles[idx]

    def collapse(self, method: str = 'halfspace') -> BoundaryProperties:
        """Full collapse to a single ``BoundaryProperties``.

        Selects the median-range profile, then collapses its layers via
        ``method`` (see :meth:`LayeredBottom.collapse`). The median range
        matches what :meth:`PropagationModel._project_environment` uses
        when it auto-collapses an RDLB env. For control over the
        range-axis selection, chain explicitly:
        ``rdl.to_profile('rmax').collapse('top_layer')``.
        """
        return self.to_profile('median').collapse(method)


def _boundary_has_shear(boundary) -> bool:
    """Shared helper: does this boundary carry any non-zero shear speed?

    Handles ``BoundaryProperties``, ``RangeDependentBottom``,
    ``LayeredBottom``, and ``RangeDependentLayeredBottom``. ``None``
    returns ``False`` so callers can pass ``env.surface`` directly.
    """
    if boundary is None:
        return False

    def _scalar(b) -> bool:
        cs = getattr(b, 'shear_speed', None)
        if cs is None:
            return False
        try:
            arr = np.atleast_1d(np.asarray(cs, dtype=float))
        except (TypeError, ValueError):
            return False
        return bool(np.any(arr > 0))

    if isinstance(boundary, RangeDependentLayeredBottom):
        for prof in boundary.profiles:
            for layer in prof.layers:
                if _scalar(layer):
                    return True
            if _scalar(prof.halfspace):
                return True
        return False
    if isinstance(boundary, LayeredBottom):
        for layer in boundary.layers:
            if _scalar(layer):
                return True
        return _scalar(boundary.halfspace)
    return _scalar(boundary)
