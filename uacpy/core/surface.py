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

from uacpy.core.exceptions import ConfigurationError
from uacpy.core.deck_limits import DECK_RANGE_RESOLUTION_M
from uacpy.core._grid import nearest_index_on_axis
from uacpy.core._repr import axis, build
from uacpy.core._validate import (
    require_non_negative, require_strictly_increasing, reject_complex,
)
from uacpy.core.collapse import RANGE_COLLAPSE_METHODS, _method_list
from uacpy.core._provenance import dedupe_provenance
from uacpy.core._carrier import (
    DeepCopyMixin, RevalidateOnAssignMixin, carrier,
)
from uacpy.core._export import CarrierExport
from uacpy.core.boundary import (
    BoundaryProperties, _columns, _delegate_write, _property_row,
    _reduce_uniform_nodes,
    _HALFSPACE_DELEGATED,
)

__all__ = [
    'Surface',
]


# Half-space fields a ``Surface`` write follows through to the stored
# boundaries. The same nine names ``Bottom`` delegates, and the same set
# object: the two carriers hold their boundaries in the same
# ``BoundaryProperties``, so a field added to one is a field added to both,
# and restating the list here let them drift apart silently.
_SURFACE_DELEGATED = _HALFSPACE_DELEGATED



# eq=False: a dataclass __eq__ over ndarray fields raises; compare by identity.
@carrier(eq=False)
class Surface(RevalidateOnAssignMixin, DeepCopyMixin, CarrierExport):
    """Surface acoustic properties, optionally range-dependent.

    Attributes
    ----------
    nodes : list of BoundaryProperties
        One surface boundary per range node (length ``N >= 1``).
    ranges : ndarray, shape (N,), optional
        Range axis in metres for a range-dependent surface; ``None`` for a
        single uniform surface.

    Notes
    -----
    `Surface` and :class:`~uacpy.core.bottom.Bottom` expose their nodes
    through different mechanisms, by design. A surface node *is* a
    :class:`BoundaryProperties`, so `Surface` delegates attribute access
    (``surface.roughness`` and the other ``BoundaryProperties`` fields)
    through ``__getattr__`` / ``__setattr__`` — a uniform `Surface` is a
    drop-in for a single ``BoundaryProperties``. A `Bottom`
    node is a whole :class:`SeabedColumn` (layers over a half-space), for
    which single-attribute delegation is ill-defined, so `Bottom` instead
    exposes explicit aggregate views (``halfspace_sound_speed``,
    ``acoustic_type``, …).

    :meth:`at` and :meth:`isel` return a **copy**, matching `Bottom`: a
    result is always safe to mutate and never writes back through to the
    carrier. Delegation is asymmetric: a delegated read comes from the
    r = 0 node, while a delegated write propagates to **every** node — a
    uniform broadcast that flattens any range dependence, and warns when
    the surface carries more than one node. ``.nodes[i]`` addresses
    a single node in place.
    """

    nodes: List[BoundaryProperties]
    ranges: Optional[np.ndarray] = None

    def __post_init__(self):
        self.nodes = list(self.nodes)
        if not self.nodes:
            raise ConfigurationError("Surface: needs at least one boundary.")
        for p in self.nodes:
            if not isinstance(p, BoundaryProperties):
                raise ConfigurationError(
                    f"Surface: every node must be a BoundaryProperties; got "
                    f"{type(p).__name__}.")
        # The surface holds copies: a node the caller edits afterwards
        # never reaches it.
        self.nodes = [_copy.deepcopy(p) for p in self.nodes]
        if self.ranges is not None:
            # Ahead of the float64 cast below, which discards an imaginary
            # part — see _reject_complex for the two ways it does it.
            reject_complex(self.ranges, "Surface.ranges")
            self.ranges = np.array(self.ranges, dtype=float).reshape(-1)
            if self.ranges.size != len(self.nodes):
                raise ConfigurationError(
                    f"Surface: ranges ({self.ranges.size}) and nodes "
                    f"({len(self.nodes)}) must have the same length.")
            require_non_negative(self.ranges, "Surface.ranges", hint="metres")
            if self.ranges.size > 1:
                require_strictly_increasing(
                    self.ranges, "Surface.ranges", min_step=DECK_RANGE_RESOLUTION_M)
        elif len(self.nodes) != 1:
            raise ConfigurationError(
                "Surface: multiple boundaries require a matching ranges= axis.")

    def _table(self):
        """One row per range node: ``range`` (m; 0 for a uniform surface),
        ``acoustic_type`` and the acoustic properties, NaN where a boundary
        type carries none."""
        ranges = (np.zeros(len(self.nodes)) if self.ranges is None
                  else np.asarray(self.ranges, dtype=float))
        return _columns([{'range': float(r),
                          'acoustic_type': node.acoustic_type,
                          **_property_row(node)}
                         for r, node in zip(ranges, self.nodes)])

    def __repr__(self) -> str:
        if not self.is_range_dependent:
            return build('Surface', self.nodes[0]._repr_bits())
        types = sorted({p.acoustic_type for p in self.nodes})
        return build('Surface', [axis(self.ranges, 'ranges', 'm'),
                                 f"types [{', '.join(types)}]"])

    @property
    def data_sources(self) -> tuple:
        """Every node's provenance, exact repeats removed (harmonised with the
        leaf carriers and ``env.data_sources``). On a range-dependent surface
        each record carries its node's range as ``range_m``."""
        return dedupe_provenance(
            self.nodes, ranges=self.ranges if len(self.nodes) > 1 else None)

    # ── constructors ────────────────────────────────────────────────────────
    @classmethod
    def coerce(cls, value) -> 'Surface':
        """Coerce ``None`` (→ vacuum) / ``BoundaryProperties`` / ``Surface`` /
        ``[(range, BoundaryProperties), ...]`` into a :class:`Surface`.

        ``None`` policy: a uniform pressure-release (vacuum) surface — the
        physical default for open water.

        Parameters
        ----------
        value : None, BoundaryProperties, Surface or sequence
            The ``surface=`` value (see above).
        """
        if isinstance(value, Surface):
            return value
        if value is None:
            return cls(nodes=[BoundaryProperties(acoustic_type='vacuum')])
        if isinstance(value, BoundaryProperties):
            return cls(nodes=[value])
        try:
            nodes = list(value)
        except TypeError:
            nodes = []
        if nodes and all(isinstance(n, (tuple, list)) and len(n) == 2
                         and isinstance(n[1], BoundaryProperties) for n in nodes):
            ranges = [float(r) for r, _ in nodes]
            props = [p for _, p in nodes]
            return cls(nodes=props, ranges=np.asarray(ranges, dtype=float))
        raise ConfigurationError(
            "Surface: expected None, a BoundaryProperties, a Surface, or a list "
            "of (range_m, BoundaryProperties) nodes; got "
            f"{type(value).__name__}.")

    # ── derived ─────────────────────────────────────────────────────────────
    @property
    def n_ranges(self) -> int:
        return len(self.nodes)

    @property
    def range_max(self) -> float:
        return 0.0 if self.ranges is None else float(np.max(self.ranges))

    @property
    def is_range_dependent(self) -> bool:
        """True when the surface carries more than one range node.

        A structural test (node count on the ranged axis), like
        ``SoundSpeedProfile`` / ``Bottom``: nodes with identical properties
        still count as range-dependent. Contrast
        ``Bathymetry.varies_with_range`` / ``Altimetry.varies_with_range``,
        which test whether the *values* actually vary with range."""
        return self.ranges is not None and len(self.nodes) > 1

    @property
    def is_elastic(self) -> bool:
        """True if *any* node carries non-zero shear (mirrors
        :attr:`Bottom.is_elastic`)."""
        return any((getattr(p, 'shear_speed', 0.0) or 0.0) > 0
                   for p in self.nodes)

    # ── slicing ─────────────────────────────────────────────────────────────
    def _nearest_index(self, range: float) -> int:
        return nearest_index_on_axis(self.ranges, range)

    def at(self, *, range: float) -> BoundaryProperties:
        """Copy of the nearest surface :class:`BoundaryProperties` to ``range``
        (m).

        Always nearest — boundary types cannot be blended, so a `Surface` has
        no ``eval`` (the surface *shape* interpolates via ``Altimetry``).
        Positional counterpart: :meth:`isel`.

        ``range`` must be a finite scalar — a NaN/inf or array-valued label
        raises ``ConfigurationError``, the same contract ``Field.at`` applies.

        Parameters
        ----------
        range : float
            Range (m), a finite scalar.
        """
        return _copy.deepcopy(self.nodes[self._nearest_index(range)])

    def isel(self, *, range: int) -> BoundaryProperties:
        """Copy of the surface :class:`BoundaryProperties` at integer index
        ``range`` — the positional counterpart of :meth:`at`.

        Parameters
        ----------
        range : int
            Node index.
        """
        i = int(range)
        n = len(self.nodes)
        if not -n <= i < n:
            raise IndexError(
                f"Surface.isel: range index {i} out of range for {n} node(s)")
        return _copy.deepcopy(self.nodes[i])

    def collapse_range(self, method: str = 'r0') -> 'Surface':
        """Collapse a range-dependent surface to a single uniform boundary.

        Returns ``self`` (not a copy) when the surface is already uniform.

        ``'r0'`` / ``'rmax'`` keep the first / last node. ``'mean'`` /
        ``'median'`` numerically average the boundary properties across nodes,
        each node weighing the same whatever its range spacing (keeping the r = 0 ``acoustic_type``) — only physical when the nodes
        share a type (a marginal ice zone, open water beside ice, takes
        ``'r0'``/``'rmax'``), mirroring :meth:`Bottom.collapse_range`.
        Uniform ``'file'``/``'precalc'`` nodes collapse to their shared
        reflection file with only the roughness reduced, and raise when the
        files differ (tables cannot be blended), again mirroring
        :meth:`Bottom.collapse_range`.

        Parameters
        ----------
        method : {'r0', 'rmax', 'mean', 'median'}, optional
            The reduction (see above). Default ``'r0'``.
        """
        # Validated before the early return: a range-independent
        # carrier has nothing to reduce, and returning self first
        # made a typo silent until the user switched to a
        # range-dependent environment — the moment they are least
        # looking for one.
        if method not in RANGE_COLLAPSE_METHODS:
            raise ConfigurationError(
                f"Surface.collapse_range: unknown method={method!r}; "
                f"valid: {_method_list(RANGE_COLLAPSE_METHODS)}.")
        if not self.is_range_dependent:
            return self
        if method == 'r0':
            return Surface(nodes=[_copy.deepcopy(self.nodes[0])])
        if method == 'rmax':
            return Surface(nodes=[_copy.deepcopy(self.nodes[-1])])
        return Surface(nodes=[_reduce_uniform_nodes(
            self.nodes, method, 'Surface.collapse_range', 'nodes')])

    def __getattr__(self, name):
        # Uniform-surface compatibility: forward BoundaryProperties reads to the
        # r = 0 node so a Surface stands in for a single BoundaryProperties.
        if name in _SURFACE_DELEGATED:
            return getattr(self.nodes[0], name)
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}.")

    def __setattr__(self, name, value):
        # Writes must follow reads through to the nodes. A plain assignment
        # would create an instance attribute shadowing ``__getattr__``, so
        # ``surface.roughness`` would report the new value while ``at()``,
        # ``collapse_range()``, the repr and every writer — all of which read
        # ``nodes`` — would keep the previous one.
        if name in _SURFACE_DELEGATED and 'nodes' in self.__dict__:
            _delegate_write('Surface', self.nodes, name, value,
                            noun='nodes', hint='.nodes[i]')
            return
        super().__setattr__(name, value)
