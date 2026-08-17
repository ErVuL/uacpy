"""Sea-surface boundary carrier: surface acoustic properties vs range.

The top-surface analogue of :class:`uacpy.core.bottom.Bottom`. A `Surface`
holds one :class:`BoundaryProperties` per range node (vacuum / pressure-release,
a half-space, an elastic ice cover, …). The common uniform case is a single
node (``ranges=None``); a range-dependent surface (e.g. a marginal ice zone,
open water → ice → open water) carries several.

Like `Bottom`, distinct boundary types cannot be blended, so a `Surface` has
``at`` / ``isel`` (select by range / index) but no ``eval`` — only the surface
*shape* (:class:`uacpy.core.altimetry.Altimetry`) interpolates. Reads of
:class:`BoundaryProperties` attributes (``acoustic_type``, ``sound_speed``, …)
delegate to the r = 0 node, so a uniform `Surface` is a drop-in wherever a
single `BoundaryProperties` was used.
"""

import copy as _copy
import numpy as np
from typing import List, Optional
from dataclasses import dataclass

from uacpy.core.exceptions import ConfigurationError
from uacpy.core.constants import DECK_RANGE_RESOLUTION_M
from uacpy.core._carrier_validate import (
    _require_non_negative, _require_positive, _require_strictly_increasing,
    _require_attenuation_in_range, _dedupe_provenance,
)
from uacpy.core.bottom import (
    BoundaryProperties, _reduce_boundaries, _PARAMETER_FREE_TYPES,
)


_SURFACE_DELEGATED = frozenset({
    'acoustic_type', 'density', 'sound_speed', 'attenuation', 'roughness',
    'shear_speed', 'shear_attenuation', 'grain_size_phi', 'reflection_file',
})

# Delegated fields that define what kind of boundary a node is. A write to one
# cannot be validated field-by-field (the construction rules couple it to the
# other fields), so the proxy refuses it instead of storing an unvalidated
# node.
_SURFACE_TYPE_FIELDS = frozenset({'acoustic_type', 'reflection_file'})


# eq=False: a dataclass __eq__ over ndarray fields raises; compare by identity.
@dataclass(eq=False)
class Surface:
    """Surface acoustic properties, optionally range-dependent.

    Attributes
    ----------
    properties : list of BoundaryProperties
        One surface boundary per range node (length ``N >= 1``).
    ranges : ndarray, shape (N,), optional
        Range axis in metres for a range-dependent surface; ``None`` for a
        single uniform surface.

    Notes
    -----
    `Surface` and :class:`~uacpy.core.bottom.Bottom` expose their nodes
    through different mechanisms, by design. A surface node *is* a
    :class:`BoundaryProperties`, so `Surface` delegates attribute access
    (``surface.roughness`` and the other ``BoundaryProperties`` fields) to
    the r = 0 node via ``__getattr__`` / ``__setattr__`` — a uniform
    `Surface` is a drop-in for a single ``BoundaryProperties``. A `Bottom`
    node is a whole :class:`SeabedColumn` (layers over a half-space), for
    which single-attribute delegation is ill-defined, so `Bottom` instead
    exposes explicit aggregate views (``halfspace_sound_speed``,
    ``acoustic_type``, …).
    """

    properties: List[BoundaryProperties]
    ranges: Optional[np.ndarray] = None

    def __post_init__(self):
        self.properties = list(self.properties)
        if not self.properties:
            raise ConfigurationError("Surface: needs at least one boundary.")
        for p in self.properties:
            if not isinstance(p, BoundaryProperties):
                raise ConfigurationError(
                    f"Surface: every node must be a BoundaryProperties; got "
                    f"{type(p).__name__}")
        if self.ranges is not None:
            self.ranges = np.array(self.ranges, dtype=float).reshape(-1)
            if self.ranges.size != len(self.properties):
                raise ConfigurationError(
                    f"Surface: ranges ({self.ranges.size}) and properties "
                    f"({len(self.properties)}) must have the same length.")
            _require_non_negative(self.ranges, "Surface.ranges", hint="metres")
            if self.ranges.size > 1:
                _require_strictly_increasing(
                    self.ranges, "Surface.ranges", min_step=DECK_RANGE_RESOLUTION_M)
        elif len(self.properties) != 1:
            raise ConfigurationError(
                "Surface: multiple boundaries require a matching ranges= axis.")

    def __repr__(self) -> str:
        if not self.is_range_dependent:
            return f"Surface({self.properties[0]!r})"
        r_lo = float(self.ranges[0]) / 1000
        r_hi = float(self.ranges[-1]) / 1000
        types = sorted({p.acoustic_type for p in self.properties})
        return (f"Surface(range-dependent, n={len(self.properties)}, "
                f"range=[{r_lo:g}, {r_hi:g}] km, types={types})")

    @property
    def data_sources(self) -> tuple:
        """Aggregated provenance across surface nodes, de-duplicated by source
        id (harmonised with the leaf carriers and ``env.data_sources``)."""
        return _dedupe_provenance(self.properties)

    # ── constructors ────────────────────────────────────────────────────────
    @classmethod
    def coerce(cls, value) -> 'Surface':
        """Coerce ``None`` (→ vacuum) / ``BoundaryProperties`` / ``Surface`` /
        ``[(range, BoundaryProperties), ...]`` into a :class:`Surface`.

        ``None`` policy: a uniform pressure-release (vacuum) surface — the
        physical default for open water."""
        if isinstance(value, Surface):
            return value
        if value is None:
            return cls(properties=[BoundaryProperties(acoustic_type='vacuum')])
        if isinstance(value, BoundaryProperties):
            return cls(properties=[value])
        try:
            nodes = list(value)
        except TypeError:
            nodes = []
        if nodes and all(isinstance(n, (tuple, list)) and len(n) == 2
                         and isinstance(n[1], BoundaryProperties) for n in nodes):
            ranges = [float(r) for r, _ in nodes]
            props = [p for _, p in nodes]
            return cls(properties=props, ranges=np.asarray(ranges, dtype=float))
        raise ConfigurationError(
            "Surface: expected None, a BoundaryProperties, a Surface, or a list "
            "of (range_m, BoundaryProperties) nodes; got "
            f"{type(value).__name__}.")

    # ── derived ─────────────────────────────────────────────────────────────
    @property
    def n_ranges(self) -> int:
        return len(self.properties)

    @property
    def range_max(self) -> float:
        return 0.0 if self.ranges is None else float(np.max(self.ranges))

    @property
    def is_range_dependent(self) -> bool:
        """True when the surface carries more than one range node.

        A structural test (node count on the ranged axis), like
        ``SoundSpeedProfile`` / ``Bottom``: nodes with identical properties
        still count as range-dependent. Contrast
        ``Bathymetry.is_range_dependent`` / ``Altimetry.is_range_dependent``,
        which test whether the *values* actually vary with range."""
        return self.ranges is not None and len(self.properties) > 1

    @property
    def is_elastic(self) -> bool:
        """True if *any* node carries non-zero shear (mirrors
        :attr:`Bottom.is_elastic`)."""
        return any((getattr(p, 'shear_speed', 0.0) or 0.0) > 0
                   for p in self.properties)

    # ── grid-library selectors ──────────────────────────────────────────────
    def _nearest_index(self, range: float) -> int:
        if self.ranges is None:
            return 0
        return int(np.argmin(np.abs(self.ranges - float(range))))

    def at(self, *, range: float) -> BoundaryProperties:
        """Nearest surface :class:`BoundaryProperties` to ``range`` (m).

        Always nearest — boundary types cannot be blended, so a `Surface` has
        no ``eval`` (the surface *shape* interpolates via ``Altimetry``).
        Positional counterpart: :meth:`isel`.
        """
        return self.properties[self._nearest_index(range)]

    def isel(self, *, range: int) -> BoundaryProperties:
        """Surface :class:`BoundaryProperties` at integer index ``range`` — the
        positional counterpart of :meth:`at`."""
        i = int(range)
        n = len(self.properties)
        if not -n <= i < n:
            raise IndexError(
                f"Surface.isel: range index {i} out of range for {n} node(s)")
        return self.properties[i]

    def collapse(self, method: str = 'r0') -> 'Surface':
        """Collapse a range-dependent surface to a single uniform boundary.

        ``'r0'`` / ``'rmax'`` keep the first / last node. ``'mean'`` /
        ``'median'`` numerically average the boundary properties across nodes
        (keeping the r = 0 ``acoustic_type``) — only physical when the nodes
        share a type, mirroring :meth:`Bottom.select_range` for half-spaces.
        Uniform ``'file'``/``'precalc'`` nodes collapse to their shared
        reflection file with only the roughness reduced, and raise when the
        files differ (tables cannot be blended), again mirroring
        :meth:`Bottom.select_range`."""
        if not self.is_range_dependent:
            return self
        if method == 'r0':
            return Surface(properties=[_copy.deepcopy(self.properties[0])])
        if method == 'rmax':
            return Surface(properties=[_copy.deepcopy(self.properties[-1])])
        if method not in ('mean', 'median'):
            raise ConfigurationError(
                f"Surface.collapse: unknown method={method!r}; valid: 'r0', "
                "'rmax', 'mean', 'median'")
        # Averaging is only meaningful within one boundary type: blending an
        # open-water (vacuum) node with an ice half-space would build an
        # inconsistent boundary (e.g. a 'vacuum' card carrying a shear speed).
        types = {p.acoustic_type for p in self.properties}
        if len(types) > 1:
            raise ConfigurationError(
                f"Surface.collapse({method!r}) needs a single boundary type to "
                f"average; got {sorted(types)}. Boundary types cannot be "
                f"blended — use 'r0' or 'rmax' (e.g. for a marginal ice zone).")
        reduce = np.mean if method == 'mean' else np.median
        # A uniform 'file'/'precalc' surface carries no real numbers to
        # reduce — each node is its reflection-coefficient table. Nodes
        # sharing one table collapse to that shared spec (roughness, the one
        # genuine number they carry, is still reduced); distinct tables
        # cannot be averaged into anything.
        (the_type,) = types
        if the_type in ('file', 'precalc'):
            specs = {p.reflection_file for p in self.properties}
            if len(specs) > 1:
                raise ConfigurationError(
                    f"Surface.collapse({method!r}) cannot average "
                    f"'{the_type}' nodes with different reflection files "
                    f"({sorted(specs, key=str)}). Reflection-coefficient "
                    f"tables cannot be blended — use 'r0' or 'rmax'.")
            shared = _copy.deepcopy(self.properties[0])
            shared.roughness = float(
                reduce([p.roughness for p in self.properties]))
            return Surface(properties=[shared])
        return Surface(properties=[
            _reduce_boundaries(self.properties, reduce)])

    def __getattr__(self, name):
        # Uniform-surface compatibility: forward BoundaryProperties reads to the
        # r = 0 node so a Surface stands in for a single BoundaryProperties.
        if name in _SURFACE_DELEGATED:
            return getattr(self.properties[0], name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}")

    def __setattr__(self, name, value):
        # Writes must follow reads through to the nodes. A plain assignment
        # would create an instance attribute shadowing ``__getattr__``, so
        # ``surface.roughness`` would report the new value while ``at()``,
        # ``collapse()``, the repr and every writer — all of which read
        # ``properties`` — kept the old one.
        if name in _SURFACE_DELEGATED and 'properties' in self.__dict__:
            value = self._validate_delegated_write(name, value)
            for node in self.properties:
                setattr(node, name, value)
            return
        super().__setattr__(name, value)

    def _validate_delegated_write(self, name, value):
        """Apply the ``BoundaryProperties`` construction rules to a delegated
        write, so the proxy cannot store a value on the nodes that their
        constructor would refuse. Returns the value coerced to float."""
        if name in _SURFACE_TYPE_FIELDS:
            raise ConfigurationError(
                f"Surface.{name} cannot be assigned in place: it defines what "
                f"kind of boundary each node is, and the construction rules "
                f"couple it to the other fields. Build new "
                f"BoundaryProperties node(s) (and a Surface from them) "
                f"instead.")
        # vacuum/rigid nodes carry no half-space acoustic parameters, so a
        # delegated write of one is the same conflict the constructor's
        # explicit-conflict guard rejects. ``roughness`` stays writable — an
        # interface property every boundary type carries.
        if name != 'roughness':
            bare = sorted({p.acoustic_type for p in self.properties
                           if p.acoustic_type in _PARAMETER_FREE_TYPES})
            if bare:
                raise ConfigurationError(
                    f"Surface.{name} = {value!r}: this surface has "
                    f"{'/'.join(bare)} node(s), which ignore half-space "
                    f"acoustic parameters. Build half-space "
                    f"BoundaryProperties node(s) to give the surface "
                    f"geoacoustics.")
        value = float(value)
        if name == 'density':
            _require_positive(value, "Surface density", hint="g/cm^3")
        elif name == 'sound_speed' and any(
                p.acoustic_type == 'half-space' for p in self.properties):
            _require_positive(value, "Surface sound_speed on a half-space",
                              hint="m/s")
        elif name != 'grain_size_phi':
            _require_non_negative(value, f"Surface {name}")
        if name in ('attenuation', 'shear_attenuation'):
            _require_attenuation_in_range(value, f"Surface {name}")
        return value

    def copy(self) -> 'Surface':
        """Deep copy (symmetric with ``Source`` / ``Receiver`` / the other
        carriers)."""
        return _copy.deepcopy(self)
