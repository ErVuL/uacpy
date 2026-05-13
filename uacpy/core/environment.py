"""
Ocean acoustic environment definition.

Provides the Environment class for specifying water column properties
(sound speed profiles, bathymetry, boundary conditions) used by all
propagation models. Supports both range-independent and range-dependent
configurations.
"""

import copy as _copy
import numpy as np
from typing import TYPE_CHECKING, Union, List, Tuple, Optional, Dict
from dataclasses import dataclass

if TYPE_CHECKING:
    from uacpy.core.absorption import Absorption  # noqa: F401


def _validate_acoustic_type(value, label: str) -> None:
    """Reject unrecognized ``acoustic_type`` strings up front, so a typo
    like ``'halfspace'`` (vs. ``'half-space'``) fails at construction
    instead of producing a wrong Acoustics-Toolbox bottom-type code
    deep inside a writer.
    """
    from uacpy.core.constants import BoundaryType
    try:
        BoundaryType.from_string(value)
    except (ValueError, KeyError, AttributeError) as exc:
        valid = sorted({bt.value for bt in BoundaryType})
        raise ValueError(
            f"{label}: acoustic_type={value!r} is not recognized. "
            f"Valid values (plus the aliases handled by "
            f"BoundaryType.from_string): {valid}"
        ) from exc


def _require_strictly_increasing(values: np.ndarray, label: str) -> None:
    """Raise ``ValueError`` if ``values`` is not strictly monotonically
    increasing. Used to guard every range / depth axis that feeds into
    ``np.interp``, which silently produces garbage on unsorted ``xp``.
    """
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size <= 1:
        return
    diffs = np.diff(arr)
    if not np.all(diffs > 0):
        bad = int(np.argmin(diffs))
        raise ValueError(
            f"{label} must be strictly increasing; "
            f"got {arr[bad]} >= {arr[bad + 1]} at index {bad + 1} "
            f"(full axis: {arr.tolist()})"
        )


def _sanitize_title(name: str) -> str:
    """Strip newlines/control chars and escape single quotes in a Fortran
    title field. Acoustics-Toolbox `.env` titles are quote-delimited and
    column-sensitive; an unsanitized name with a newline silently corrupts
    the file and the binary parses garbage downstream.
    """
    if name is None:
        return 'unnamed'
    s = str(name)
    s = ''.join(ch if (ord(ch) >= 32 and ch != '\x7f') else ' ' for ch in s)
    s = s.replace("'", "''")
    return s.strip() or 'unnamed'


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
            raise ValueError(f"SedimentLayer: thickness must be positive (m); got {self.thickness}")
        if self.sound_speed <= 0:
            raise ValueError(f"SedimentLayer: sound_speed must be positive (m/s); got {self.sound_speed}")
        if self.density <= 0:
            raise ValueError(f"SedimentLayer: density must be positive (g/cm^3); got {self.density}")

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
    def from_preset(cls, name: str, *, thickness: float, **overrides) -> "SedimentLayer":
        """Build a :class:`SedimentLayer` from a :mod:`uacpy.core.materials`
        preset (``'sand'``, ``'silt'``, ``'clay'``, …).

        ``thickness`` is required (presets only encode acoustic
        properties, not layer geometry). Any additional kwargs override
        the preset's ``sound_speed`` / ``density`` / ``attenuation`` /
        ``shear_speed`` / ``shear_attenuation`` for site-specific tuning.
        """
        from uacpy.core.materials import get_material
        m = get_material(name)
        kwargs = dict(
            thickness=thickness,
            sound_speed=m['sound_speed'],
            density=m['density'],
            attenuation=m['attenuation'],
            shear_speed=m['shear_speed'],
            shear_attenuation=m['shear_attenuation'],
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
    acoustic_type : str
        Boundary type: 'vacuum', 'rigid', 'half-space', 'grain-size', 'file'
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

    >>> # First, compute reflection coefficients (output_dir is required)
    >>> from uacpy.models import Bounce
    >>> from pathlib import Path
    >>> bounce = Bounce()
    >>> result = bounce.run(env, source, receiver, output_dir=Path('./bounce_out'))
    >>> brc_file = result.metadata['brc_file']
    >>>
    >>> # Then use in Bellhop/Kraken/Scooter
    >>> bottom = BoundaryProperties(
    ...     acoustic_type='file',
    ...     reflection_file=brc_file
    ... )
    >>> env = Environment(name="test", bathymetry=100, bottom=bottom)
    """

    acoustic_type: str = 'vacuum'
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
            raise ValueError(f"BoundaryProperties: density must be positive (g/cm^3); got {self.density}")
        if self.sound_speed < 0:
            raise ValueError(f"BoundaryProperties: sound_speed must be non-negative (m/s); got {self.sound_speed}")
        if self.attenuation < 0:
            raise ValueError(f"BoundaryProperties: attenuation must be non-negative; got {self.attenuation}")
        if self.shear_speed < 0:
            raise ValueError(f"BoundaryProperties: shear_speed must be non-negative (m/s); got {self.shear_speed}")
        if self.shear_attenuation < 0:
            raise ValueError(
                f"BoundaryProperties: shear_attenuation must be non-negative; "
                f"got {self.shear_attenuation}"
            )
        _validate_acoustic_type(self.acoustic_type, "BoundaryProperties")

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
    def from_preset(cls, name: str, **overrides) -> "BoundaryProperties":
        """Build a :class:`BoundaryProperties` from a
        :mod:`uacpy.core.materials` preset.

        Picks ``acoustic_type='half-space'`` automatically, copies every
        preset field that maps onto :class:`BoundaryProperties` (sound
        speeds, density, attenuations, ``grain_size_phi`` if defined,
        ``roughness``), and applies any ``**overrides`` last.
        """
        from uacpy.core.materials import get_material
        m = get_material(name)
        kwargs = dict(
            acoustic_type='half-space',
            sound_speed=m['sound_speed'],
            density=m['density'],
            attenuation=m['attenuation'],
            shear_speed=m['shear_speed'],
            shear_attenuation=m['shear_attenuation'],
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
    acoustic_type : str
        Boundary type (same at all ranges): 'vacuum', 'rigid', 'half-space', etc.
        Default is 'vacuum' for consistency with BoundaryProperties.

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
    acoustic_type: str = 'vacuum'

    def __post_init__(self):
        """Validate array lengths and set defaults."""
        _validate_acoustic_type(self.acoustic_type, "RangeDependentBottom")
        self.ranges = np.asarray(self.ranges, dtype=float).ravel()
        _require_strictly_increasing(self.ranges, "RangeDependentBottom.ranges")
        n = len(self.ranges)

        for attr_name in ['sound_speed', 'density', 'attenuation']:
            attr = getattr(self, attr_name)
            if len(attr) != n:
                raise ValueError(
                    f"RangeDependentBottom: {attr_name} length ({len(attr)}) must "
                    f"match ranges length ({n})"
                )

        if self.shear_speed is None:
            self.shear_speed = np.zeros(n)
        if self.shear_attenuation is None:
            self.shear_attenuation = np.zeros(n)

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

    def eval(self, *, range: float, interp: str = 'linear') -> BoundaryProperties:
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
            raise ValueError(
                f"RangeDependentBottom.eval: interp must be 'linear' or "
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
            return self.eval(range=float(self.ranges[0]))
        if method == 'rmax':
            return self.eval(range=float(self.ranges[-1]))
        if method == 'mean':
            reduce = np.mean
        elif method == 'median':
            reduce = np.median
        else:
            raise ValueError(
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
            raise ValueError("LayeredBottom: requires at least one SedimentLayer; got 0")

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
        value.

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
        raise ValueError(
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
                raise ValueError(
                    f"LayeredBottom.from_presets: layer entry must be "
                    f"(name, thickness) or (name, thickness, overrides); "
                    f"got {entry!r}"
                )
            sediment_layers.append(
                SedimentLayer.from_preset(name, thickness=thickness, **overrides)
            )
        hs = BoundaryProperties.from_preset(
            halfspace, **(halfspace_overrides or {}),
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
            raise ValueError(
                "RangeDependentLayeredBottom: at least one range point is required"
            )
        _require_strictly_increasing(
            self.ranges, "RangeDependentLayeredBottom.ranges",
        )
        if len(self.profiles) != n:
            raise ValueError(
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
            raise ValueError(
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


_VALID_SSP_SHAPES = (
    'measured', 'isovelocity', 'munk', 'analytic', 'n2linear',
)


@dataclass
class SoundSpeedProfile:
    """
    Unified sound-speed profile (1-D or 2-D).

    Stores the full grid as a 2-D array ``data[n_depth, n_range]``.
    Range-independent profiles use ``n_range = 1`` and ``ranges = None``;
    range-dependent profiles set ``ranges`` to a monotonically-increasing
    metres vector of length ``n_range``.

    Attributes
    ----------
    depths : ndarray, shape (N,)
        Depth axis in metres, monotonically increasing.
    data : ndarray, shape (N, M)
        Sound speed in m/s. ``M = 1`` for 1-D profiles.
    ranges : ndarray, shape (M,), optional
        Range axis in **metres**, monotonically increasing. ``None`` for 1-D.
    shape : str
        Declaration of what the data represents:
        ``'measured'`` (default), ``'isovelocity'``, ``'munk'``,
        ``'analytic'`` or ``'n2linear'``. Only ``'isovelocity'``
        actually overrides ``TopOpt(1)`` (forces ``'C'`` — any connection
        scheme over constant data is constant). The other values are
        informational metadata; the model's ``interp_ssp`` kwarg drives
        the AT character.
    """
    depths: np.ndarray
    data: np.ndarray
    ranges: Optional[np.ndarray] = None
    shape: str = 'measured'

    def __post_init__(self):
        self.depths = np.asarray(self.depths, dtype=float).reshape(-1)
        self.data = np.asarray(self.data, dtype=float)
        if self.data.ndim == 1:
            self.data = self.data.reshape(-1, 1)
        if self.data.ndim != 2:
            raise ValueError(
                f"SoundSpeedProfile: data must be 1-D or 2-D; got {self.data.ndim}-D"
            )
        if self.data.shape[0] != self.depths.size:
            raise ValueError(
                f"SoundSpeedProfile: data rows ({self.data.shape[0]}) must match "
                f"depths length ({self.depths.size})"
            )
        _require_strictly_increasing(self.depths, "SoundSpeedProfile.depths")
        if self.ranges is not None:
            self.ranges = np.asarray(self.ranges, dtype=float).reshape(-1)
            if self.ranges.size != self.data.shape[1]:
                raise ValueError(
                    f"SoundSpeedProfile: ranges length ({self.ranges.size}) must "
                    f"match data columns ({self.data.shape[1]})"
                )
            _require_strictly_increasing(
                self.ranges, "SoundSpeedProfile.ranges",
            )
        elif self.data.shape[1] != 1:
            raise ValueError(
                f"SoundSpeedProfile: ranges=None requires single-column data; "
                f"got shape {self.data.shape}"
            )
        self.shape = str(self.shape).lower()
        if self.shape not in _VALID_SSP_SHAPES:
            raise ValueError(
                f"SoundSpeedProfile: shape={self.shape!r} not in "
                f"{_VALID_SSP_SHAPES}"
            )

    def __repr__(self) -> str:
        c_lo = float(np.min(self.data))
        c_hi = float(np.max(self.data))
        bits = [
            f"shape={self.shape!r}",
            f"n_z={self.depths.size}",
            f"z=[{float(self.depths[0]):g}, {float(self.depths[-1]):g}] m",
        ]
        if self.is_range_dependent:
            r_lo = float(self.ranges[0]) / 1000
            r_hi = float(self.ranges[-1]) / 1000
            bits.append(f"n_r={self.data.shape[1]}")
            bits.append(f"range=[{r_lo:g}, {r_hi:g}] km")
        bits.append(f"c=[{c_lo:g}, {c_hi:g}] m/s")
        return f"SoundSpeedProfile({', '.join(bits)})"

    @property
    def is_range_dependent(self) -> bool:
        return self.ranges is not None and self.data.shape[1] > 1

    @property
    def n_depths(self) -> int:
        return int(self.depths.size)

    @property
    def n_ranges(self) -> int:
        return int(self.data.shape[1])

    def to_pairs(self) -> np.ndarray:
        """Return ``(N, 2)`` ``(depth, c)`` view of the 1-D form.

        For range-dependent profiles, returns the range-0 column. Use
        ``at_range`` for an explicit slice or ``collapse`` for a chosen
        reduction.
        """
        return np.column_stack([self.depths, self.data[:, 0]])

    def eval(
        self,
        *,
        depth: Optional[float] = None,
        range: Optional[float] = None,
        interp: str = 'linear',
    ) -> 'SoundSpeedProfile':
        """Slice the SSP at the requested depth and/or range.

        ``interp='linear'`` (default) interpolates along range and depth,
        with constant extrapolation outside ``[ranges[0], ranges[-1]]``.
        ``interp='nearest'`` returns the closest stored grid sample on
        each axis without interpolation.
        """
        if interp not in ('linear', 'nearest'):
            raise ValueError(
                f"SoundSpeedProfile.eval: interp must be 'linear' or "
                f"'nearest'; got {interp!r}"
            )

        if range is None:
            sliced = self
        elif not self.is_range_dependent:
            sliced = SoundSpeedProfile(
                depths=self.depths.copy(),
                data=self.data[:, :1].copy(),
                ranges=None,
                shape=self.shape,
            )
        elif interp == 'nearest':
            r_idx = int(np.argmin(np.abs(self.ranges - range)))
            sliced = SoundSpeedProfile(
                depths=self.depths.copy(),
                data=self.data[:, r_idx:r_idx + 1].copy(),
                ranges=None,
                shape=self.shape,
            )
        else:
            if range <= self.ranges[0]:
                col = self.data[:, 0].copy()
            elif range >= self.ranges[-1]:
                col = self.data[:, -1].copy()
            else:
                col = np.array([
                    np.interp(range, self.ranges, row)
                    for row in self.data
                ])
            sliced = SoundSpeedProfile(
                depths=self.depths.copy(),
                data=col.reshape(-1, 1),
                ranges=None,
                shape=self.shape,
            )
        if depth is None:
            return sliced
        if interp == 'nearest':
            d_idx = int(np.argmin(np.abs(sliced.depths - depth)))
            return SoundSpeedProfile(
                depths=np.array([float(sliced.depths[d_idx])]),
                data=sliced.data[d_idx:d_idx + 1, :].copy(),
                ranges=None,
                shape=sliced.shape,
            )
        c = float(np.interp(depth, sliced.depths, sliced.data[:, 0]))
        return SoundSpeedProfile(
            depths=np.array([float(depth)]),
            data=np.array([[c]]),
            ranges=None,
            shape=sliced.shape,
        )

    @property
    def value(self) -> float:
        """Scalar sound speed when this profile has been collapsed to a
        single ``(depth, range)`` cell via ``at(depth=, range=)``.
        Raises if the profile carries more than one sample."""
        if self.data.size != 1:
            raise ValueError(
                f"SoundSpeedProfile.value: only valid on a 1×1 slice; "
                f"got shape {self.data.shape}"
            )
        return float(self.data.flat[0])

    def collapse(self, method: str = 'r0') -> 'SoundSpeedProfile':
        """Collapse a 2-D profile to 1-D using ``method``.

        Methods
        -------
        ``'r0'``     : keep the range-0 column.
        ``'mean'``   : depth-wise mean across all ranges.
        ``'median'`` : depth-wise median across all ranges.
        ``'rmax'``   : keep the last (deepest range) column.
        """
        if not self.is_range_dependent:
            return self
        if method == 'r0':
            col = self.data[:, 0]
        elif method == 'rmax':
            col = self.data[:, -1]
        elif method == 'mean':
            col = self.data.mean(axis=1)
        elif method == 'median':
            col = np.median(self.data, axis=1)
        else:
            raise ValueError(
                f"SoundSpeedProfile.collapse: unknown method={method!r}; "
                "valid: 'r0', 'rmax', 'mean', 'median'"
            )
        return SoundSpeedProfile(
            depths=self.depths.copy(),
            data=col.reshape(-1, 1),
            ranges=None,
            shape=self.shape,
        )

    def extend_to(self, depth_max: float) -> 'SoundSpeedProfile':
        """Return a copy with the deepest sample sitting exactly at
        ``depth_max``.

        Three cases:

        * ``depth_max == depths[-1]`` — return ``self`` unchanged.
        * ``depth_max > depths[-1]`` — append a new sample at
          ``depth_max`` carrying the deepest existing sound speed
          (constant extrapolation, the AT writer convention).
        * ``depth_max < depths[-1]`` — truncate samples below
          ``depth_max`` and interpolate a final sample exactly at
          ``depth_max`` so writers that require ``ssp[-1] == env.depth``
          (Bellhop / Kraken) round-trip without manual alignment.
        """
        # Tolerant float compare — caller may pass in e.g. ``env.depth``
        # that's been round-tripped through I/O and differs by a few
        # ulps from ``self.depths[-1]``.
        last = float(self.depths[-1])
        if np.isclose(depth_max, last, rtol=1e-9, atol=1e-9):
            return self
        if depth_max > last:
            new_depths = np.append(self.depths, depth_max)
            new_data = np.vstack([self.data, self.data[-1:, :]])
        else:
            keep = self.depths < depth_max
            kept_depths = self.depths[keep]
            kept_data = self.data[keep]
            interp_row = np.array([
                np.interp(depth_max, self.depths, self.data[:, j])
                for j in range(self.data.shape[1])
            ])
            new_depths = np.append(kept_depths, depth_max)
            new_data = np.vstack([kept_data, interp_row[None, :]])
        return SoundSpeedProfile(
            depths=new_depths,
            data=new_data,
            ranges=(self.ranges.copy() if self.ranges is not None else None),
            shape=self.shape,
        )

    @classmethod
    def from_isovelocity(
        cls, depth_max: float, sound_speed: float = 1500.0
    ) -> 'SoundSpeedProfile':
        return cls(
            depths=np.array([0.0, float(depth_max)]),
            data=np.full((2, 1), float(sound_speed)),
            ranges=None,
            shape='isovelocity',
        )

    @classmethod
    def from_pairs(
        cls,
        pairs: Union[List[Tuple[float, float]], np.ndarray],
        shape: str = 'measured',
    ) -> 'SoundSpeedProfile':
        """Build a 1-D profile from ``[(depth, c), …]`` pairs.

        ``shape`` is informational metadata (``'measured'`` default);
        see :class:`SoundSpeedProfile`. The model's ``interp_ssp`` kwarg
        drives the sample-connection scheme.
        """
        arr = np.asarray(pairs, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"SoundSpeedProfile.from_pairs: pairs must have shape (N, 2) "
                f"as (depth, sound_speed); got shape {arr.shape}"
            )
        return cls(
            depths=arr[:, 0],
            data=arr[:, 1].reshape(-1, 1),
            ranges=None,
            shape=shape,
        )

    @classmethod
    def from_2d(
        cls,
        depths: np.ndarray,
        ranges: np.ndarray,
        matrix: np.ndarray,
        shape: str = 'measured',
    ) -> 'SoundSpeedProfile':
        """Build a 2-D profile from a depth axis, range axis (metres),
        and ``c(depth, range)`` matrix of shape ``(n_depth, n_range)``.

        For Bellhop, pair with ``Bellhop(interp_ssp='quad')`` to enable
        the external ``.ssp`` (quad) file format.
        """
        return cls(
            depths=np.asarray(depths, dtype=float),
            data=np.asarray(matrix, dtype=float),
            ranges=np.asarray(ranges, dtype=float),
            shape=shape,
        )

    @classmethod
    def from_munk(
        cls, depth_max: float, n_points: int = 101
    ) -> 'SoundSpeedProfile':
        """Munk canonical profile with axis at 1300 m, c_min = 1500 m/s."""
        depths = np.linspace(0.0, float(depth_max), int(n_points))
        z_axis = 1300.0
        epsilon = 0.00737
        c_min = 1500.0
        eta = 2.0 * (depths - z_axis) / z_axis
        c = c_min * (1.0 + epsilon * (eta - 1.0 + np.exp(-eta)))
        return cls(
            depths=depths,
            data=c.reshape(-1, 1),
            ranges=None,
            shape='munk',
        )

    @classmethod
    def from_mackenzie(
        cls,
        depths: np.ndarray,
        temperature_c: np.ndarray,
        salinity_psu: np.ndarray,
    ) -> 'SoundSpeedProfile':
        """Build a profile from in-situ ``T(z)`` and ``S(z)`` via Mackenzie's
        nine-term seawater sound-speed equation.

        ``depths``, ``temperature_c``, ``salinity_psu`` must be 1-D arrays
        of equal length sampled at the same depth grid. Use
        ``np.full_like(depths, T_const)`` if the column is isothermal/
        isohaline. Valid range: ``T ∈ [−2, 30] °C``,
        ``S ∈ [25, 40] PSU``, ``z ∈ [0, 8000] m`` (Mackenzie 1981).
        """
        from uacpy.core.acoustics import soundspeed
        z = np.asarray(depths, dtype=float).ravel()
        T = np.asarray(temperature_c, dtype=float).ravel()
        S = np.asarray(salinity_psu, dtype=float).ravel()
        if not (T.shape == S.shape == z.shape):
            raise ValueError(
                "from_mackenzie: depths, temperature_c, salinity_psu must "
                f"share shape; got {z.shape}, {T.shape}, {S.shape}"
            )
        c = soundspeed(temperature=T, salinity=S, depth=z)
        return cls(
            depths=z, data=np.asarray(c).reshape(-1, 1),
            ranges=None,
        )


def generate_sea_surface(
    max_range: float,
    wind_speed_ms: float = 10.0,
    n_points: int = 500,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a random sea surface realization from the Pierson-Moskowitz spectrum.

    Parameters
    ----------
    max_range : float
        Maximum range in meters.
    wind_speed_ms : float
        Wind speed at 10 m height in m/s. Typical values:
        - 5 m/s: calm (sea state 2-3, Hs ~ 0.3 m)
        - 10 m/s: moderate (sea state 4, Hs ~ 1.2 m)
        - 15 m/s: rough (sea state 5, Hs ~ 2.8 m)
        - 20 m/s: very rough (sea state 6, Hs ~ 5.0 m)
    n_points : int
        Number of range points in the output altimetry array.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    altimetry : ndarray, shape (n_points, 2)
        Column 0: range (m), Column 1: surface height (m, positive up).
        Suitable for passing directly to ``Environment(altimetry=...)``.
    """
    g = 9.81
    rng = np.random.default_rng(seed)

    ranges = np.linspace(0, max_range, n_points)
    dx = ranges[1] - ranges[0]

    # Spatial frequency grid (cycles/m)
    n_fft = n_points
    dk = 1.0 / (n_fft * dx)  # spatial freq resolution
    k = np.arange(1, n_fft // 2 + 1) * dk  # positive frequencies
    omega = np.sqrt(g * 2 * np.pi * k)  # deep-water dispersion: omega^2 = g*k_wave

    # Pierson-Moskowitz spectrum S(omega)
    # S(omega) = (alpha * g^2 / omega^5) * exp(-beta * (omega_p / omega)^4)
    alpha_pm = 8.1e-3
    beta_pm = 0.74
    omega_p = g / wind_speed_ms  # peak angular frequency
    S_omega = (alpha_pm * g**2 / omega**5) * np.exp(-beta_pm * (omega_p / omega)**4)

    # Convert to spatial spectrum S(k) via S(k) = S(omega) * domega/dk
    # domega/dk = g / (2*omega) for deep water
    domega_dk = g / (2 * omega)
    S_k = S_omega * domega_dk

    # Generate random amplitudes from spectrum
    amplitude = np.sqrt(2 * S_k * dk)
    phase = rng.uniform(0, 2 * np.pi, len(k))

    surface = (
        amplitude[None, :]
        * np.cos(2 * np.pi * np.outer(ranges, k) + phase[None, :])
    ).sum(axis=1)

    return np.column_stack([ranges, surface])


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


class Environment:
    """
    Ocean environment definition.

    Combines a sound-speed profile, bathymetry, optional surface
    altimetry, and surface/bottom acoustic properties into the input
    object every propagation model consumes.

    Parameters
    ----------
    name : str
        Environment identifier.
    bathymetry : float or array-like
        Either a scalar water depth in metres (flat bottom), or a
        range-dependent bathymetry as ``[(range, depth), …]``.
        The maximum depth in this argument defines the water column
        extent; ``env.depth`` exposes it as a read-only property.
    ssp : scalar (m/s), list of (depth, c_m_s) pairs, or SoundSpeedProfile, optional
        Sound-speed profile.

        * Scalar — isovelocity at the given speed.
        * List/array of ``(depth, sound_speed)`` pairs — linear-interp
          ``SoundSpeedProfile`` built via :meth:`SoundSpeedProfile.from_pairs`.
        * ``SoundSpeedProfile`` instance — used as-is (1-D or 2-D).
        * ``None`` (default) — isovelocity at 1500 m/s.
    altimetry : array-like, optional
        Surface altimetry as ``[(range, height_m), …]`` (height
        positive up). Default ``None`` (flat surface).
    bottom : BoundaryProperties, RangeDependentBottom, LayeredBottom, or RangeDependentLayeredBottom, optional
        Bottom acoustic properties. Default is a fluid sand-like half-space
        (``sound_speed=1600`` m/s, ``density=1.5`` g/cm³,
        ``attenuation=0.5`` dB/wavelength). For a perfectly reflecting bottom,
        pass ``BoundaryProperties(acoustic_type='rigid')``.
    surface : BoundaryProperties, optional
        Surface boundary properties. Default vacuum (pressure release).
    absorption : Absorption, optional
        Water-column volume-absorption model — one of
        :class:`uacpy.core.absorption.Thorp`,
        :class:`uacpy.core.absorption.FrancoisGarrison`,
        :class:`uacpy.core.absorption.Biological`, or
        :class:`uacpy.core.absorption.ConstantAbsorption`. Default ``None``
        (no volume absorption). Models inspect this field to set
        ``TopOpt`` position 4 and write the supporting per-formula lines.

    Examples
    --------
    Isovelocity:

    >>> env = Environment(name='shallow', bathymetry=100, ssp=1500)

    Linear SSP:

    >>> env = Environment(
    ...     name='test', bathymetry=200,
    ...     ssp=SoundSpeedProfile.from_pairs(
    ...         [(0, 1520), (200, 1480)]),
    ... )

    Munk:

    >>> env = Environment(
    ...     name='deep', bathymetry=5000,
    ...     ssp=SoundSpeedProfile.from_munk(5000),
    ... )

    Range-dependent bathymetry:

    >>> env = Environment(
    ...     name='wedge', bathymetry=[(0, 100), (10000, 200)],
    ... )
    """

    def __init__(
        self,
        bathymetry: Union[float, List[Tuple[float, float]], np.ndarray],
        ssp: Optional[Union[
            float, int,
            List[Tuple[float, float]],
            np.ndarray,
            SoundSpeedProfile,
        ]] = None,
        altimetry: Optional[Union[List[Tuple[float, float]], np.ndarray]] = None,
        bottom: Optional[Union[
            BoundaryProperties, RangeDependentBottom,
            'LayeredBottom', 'RangeDependentLayeredBottom',
        ]] = None,
        surface: Optional[BoundaryProperties] = None,
        absorption: Optional['Absorption'] = None,
        *,
        name: str = 'unnamed',
    ):
        from uacpy.core.absorption import Absorption
        if absorption is not None and not isinstance(absorption, Absorption):
            raise TypeError(
                f"Environment: absorption must be an Absorption subclass "
                f"(Thorp / FrancoisGarrison / Biological / ConstantAbsorption); "
                f"got {type(absorption).__name__}"
            )
        self.absorption = absorption
        self.name = _sanitize_title(name)

        if np.isscalar(bathymetry):
            water_depth = float(bathymetry)
            if water_depth <= 0:
                raise ValueError(
                    f"Environment: bathymetry depth must be positive (m); "
                    f"got {water_depth}"
                )
            self.bathymetry = np.array([[0.0, water_depth]], dtype=np.float64)
        else:
            self.bathymetry = np.array(bathymetry, dtype=np.float64)
            if self.bathymetry.ndim != 2 or self.bathymetry.shape[1] != 2:
                raise ValueError(
                    f"Environment: bathymetry must be a positive scalar or shape "
                    f"(N, 2) as [(range, depth), ...]; got shape "
                    f"{self.bathymetry.shape} (example: [(0, 100), (5000, 200)])"
                )
            if np.any(self.bathymetry[:, 0] < 0):
                raise ValueError(
                    f"Environment: bathymetry ranges must be non-negative (m); "
                    f"got {self.bathymetry[:, 0].tolist()}"
                )
            if np.any(self.bathymetry[:, 1] <= 0):
                raise ValueError(
                    f"Environment: bathymetry depths must be positive (m); "
                    f"got {self.bathymetry[:, 1].tolist()}"
                )
            _require_strictly_increasing(
                self.bathymetry[:, 0], "Environment.bathymetry ranges",
            )

        max_bathy_depth = float(np.max(self.bathymetry[:, 1]))

        if ssp is None:
            # default isovelocity at 1500 m/s
            self.ssp = SoundSpeedProfile.from_isovelocity(max_bathy_depth, 1500.0)
        elif isinstance(ssp, SoundSpeedProfile):
            self.ssp = ssp
        elif isinstance(ssp, (int, float, np.integer, np.floating)):
            # scalar → isovelocity at the given speed
            self.ssp = SoundSpeedProfile.from_isovelocity(max_bathy_depth, float(ssp))
        elif isinstance(ssp, (list, tuple, np.ndarray)):
            # list of (z, c) pairs → from_pairs (linear interp)
            self.ssp = SoundSpeedProfile.from_pairs(ssp)
        else:
            raise TypeError(
                f"Environment: ssp must be a scalar (m/s), a list of (depth, "
                f"sound_speed) pairs, or a SoundSpeedProfile; got "
                f"{type(ssp).__name__}"
            )

        if altimetry is not None:
            self.altimetry = np.array(altimetry, dtype=np.float64)
            if self.altimetry.ndim != 2 or self.altimetry.shape[1] != 2:
                raise ValueError(
                    f"Environment: altimetry must have shape (N, 2) as "
                    f"(range, height_m); got shape {self.altimetry.shape}"
                )
            _require_strictly_increasing(
                self.altimetry[:, 0], "Environment.altimetry ranges",
            )
        else:
            self.altimetry = None

        if max_bathy_depth > self.ssp.depths[-1]:
            self.ssp = self.ssp.extend_to(max_bathy_depth)

        if surface is None:
            self.surface = BoundaryProperties(acoustic_type='vacuum')
        else:
            self.surface = surface

        if bottom is None:
            self.bottom = BoundaryProperties(
                acoustic_type='half-space',
                density=1.5,
                sound_speed=1600.0,
                attenuation=0.5,
            )
        elif isinstance(bottom, (BoundaryProperties, RangeDependentBottom,
                                 LayeredBottom, RangeDependentLayeredBottom)):
            self.bottom = bottom
        else:
            raise TypeError(
                f"Environment: bottom must be BoundaryProperties, "
                f"RangeDependentBottom, LayeredBottom, or "
                f"RangeDependentLayeredBottom; got {type(bottom).__name__}"
            )

    @property
    def depth(self) -> float:
        """Maximum water depth in metres (derived from bathymetry)."""
        return float(np.max(self.bathymetry[:, 1]))

    def get_sound_speed(
        self, depth: Union[float, np.ndarray], range: float = 0.0
    ) -> np.ndarray:
        """Sound speed at given depth(s), at ``range`` for 2-D profiles."""
        slice_1d = (self.ssp.eval(range=range)
                    if self.ssp.is_range_dependent else self.ssp)
        return np.interp(np.atleast_1d(depth), slice_1d.depths,
                         slice_1d.data[:, 0])

    def bathymetry_at_range(self, range: Union[float, np.ndarray]) -> np.ndarray:
        """Bathymetry depth at the requested range(s). ``range`` can be
        a scalar or array; ``env.bathymetry`` is a plain ``(N, 2)``
        ndarray, so this helper carries the interpolation logic."""
        range = np.atleast_1d(range)
        if len(self.bathymetry) == 1:
            return np.full_like(range, self.bathymetry[0, 1])
        return np.interp(range, self.bathymetry[:, 0], self.bathymetry[:, 1])

    def halfspace_at_range(self, range: float) -> 'BoundaryProperties':
        """Return the *halfspace* :class:`BoundaryProperties` at ``range`` (m).

        Always returns a flat :class:`BoundaryProperties` regardless of
        the bottom flavour: for :class:`LayeredBottom` and
        :class:`RangeDependentLayeredBottom` this is the underlying
        halfspace beneath all sediment layers; for
        :class:`RangeDependentBottom` it is the linearly-interpolated
        sample; for a plain :class:`BoundaryProperties` it is the
        bottom itself. Used by env-file writers that emit a single
        bottom row (acoustic_type / sound_speed / density / ...).
        """
        b = self.bottom
        if isinstance(b, RangeDependentLayeredBottom):
            return b.at(range=range).halfspace
        if isinstance(b, LayeredBottom):
            return b.halfspace
        if isinstance(b, RangeDependentBottom):
            return b.eval(range=range)
        return b

    def bottom_at_range(self, range: float):
        """Bottom properties at the requested range. Returns
        :class:`LayeredBottom` for layered envs, otherwise
        :class:`BoundaryProperties`."""
        if isinstance(self.bottom, RangeDependentLayeredBottom):
            return self.bottom.at(range=range)
        if isinstance(self.bottom, RangeDependentBottom):
            return self.bottom.eval(range=range)
        return self.bottom

    def has_range_dependent_bathymetry(self) -> bool:
        if len(self.bathymetry) <= 1:
            return False
        depths = self.bathymetry[:, 1]
        return not bool(np.allclose(depths, depths[0]))

    def has_range_dependent_ssp(self) -> bool:
        return self.ssp.is_range_dependent

    def has_range_dependent_bottom(self) -> bool:
        return isinstance(self.bottom, RangeDependentBottom)

    def has_layered_bottom(self) -> bool:
        return isinstance(self.bottom, LayeredBottom)

    def has_range_dependent_layered_bottom(self) -> bool:
        return isinstance(self.bottom, RangeDependentLayeredBottom)

    def has_elastic_bottom(self) -> bool:
        """``True`` iff any sample of ``self.bottom`` carries non-zero shear.

        Walks the layer/profile structure of :class:`LayeredBottom` and
        :class:`RangeDependentLayeredBottom` so a stratified env with a
        single elastic layer reports ``True``. For
        :class:`RangeDependentBottom`, ``True`` iff *any* range sample has
        ``shear_speed > 0``.
        """
        return _boundary_has_shear(self.bottom)

    def has_elastic_surface(self) -> bool:
        """``True`` iff ``self.surface`` carries non-zero shear."""
        return _boundary_has_shear(self.surface)

    @property
    def is_range_dependent(self) -> bool:
        return (
            self.has_range_dependent_bathymetry()
            or self.ssp.is_range_dependent
            or isinstance(self.bottom, (RangeDependentBottom,
                                        RangeDependentLayeredBottom))
        )

    def __repr__(self) -> str:
        range_dep = "range-dep" if self.is_range_dependent else "range-indep"
        return (f"Environment(name='{self.name}', depth={self.depth:.1f}m, "
                f"ssp='{self.ssp.shape}', {range_dep})")

    def get_representative_depth(self, method: str = 'max') -> float:
        """
        Get representative depth from range-dependent bathymetry

        For models that don't support range-dependent environments,
        this provides a single representative depth value.

        Parameters
        ----------
        method : str, optional
            Method for computing representative value:
            - 'max': Maximum depth (deepest, default — matches the
              project-wide ``collapse={'bathymetry': 'max'}``)
            - 'median': Median depth
            - 'mean': Mean depth
            - 'min': Minimum depth (shallowest)
            - 'initial': Initial depth at range=0

        Returns
        -------
        depth : float
            Representative depth in meters

        Examples
        --------
        >>> env = Environment(name='slope',
        ...                   bathymetry=[(0, 100), (5000, 200), (10000, 300)])
        >>> env.get_representative_depth('median')
        200.0
        >>> env.get_representative_depth('mean')
        200.0
        >>> env.get_representative_depth('initial')
        100.0
        """
        depths = self.bathymetry[:, 1]

        if method == 'median':
            return float(np.median(depths))
        elif method == 'mean':
            return float(np.mean(depths))
        elif method == 'min':
            return float(np.min(depths))
        elif method == 'max':
            return float(np.max(depths))
        elif method == 'initial':
            return float(depths[0])
        else:
            raise ValueError(
                f"Environment.get_representative_depth: unknown method={method!r}; "
                "valid: 'max', 'median', 'mean', 'min', 'initial'"
            )

    def copy(self):
        """Deep copy of the environment.

        Uses ``copy.deepcopy`` so every field — including ``ssp``,
        ``altimetry``, and ``bottom`` — is duplicated without aliasing
        back to the original instance.
        """
        return _copy.deepcopy(self)
