"""Seabed carriers: a column of sediment layers over a half-space
(:class:`SeabedColumn`) and the seabed along range (:class:`Bottom`). The
boundary nodes they hold are :mod:`uacpy.core.boundary`'s. Re-exported from
:mod:`uacpy.core.environment` for stable import paths.
"""

import warnings
import copy as _copy
import numpy as np
from typing import List, Tuple, Optional, Dict

from uacpy.core.exceptions import ConfigurationError, FallbackWarning
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.boundary import (
    BoundaryProperties, BoundaryType, SedimentLayer, _columns, _property_row,
    _HALFSPACE_DELEGATED, _LAYER_ACOUSTIC_FIELDS, _boundary_from_values,
    _delegate_write, _reduce_boundaries, _reduce_uniform_nodes,
)
from uacpy.core.deck_limits import DECK_RANGE_RESOLUTION_M
from uacpy.core._repr import axis, build, qty
from uacpy.core.sediment import DEFAULT_GRAIN_SIZE_MODEL
from uacpy.core._grid import (
    _as_finite_scalar_label, nearest_index_on_axis,
)
from uacpy.core._validate import (
    require_strictly_increasing, require_non_negative, reject_complex,
    scalar_or_none,
)
from uacpy.core.collapse import (
    COLUMN_COLLAPSE_METHODS, RANGE_COLLAPSE_METHODS, _method_list,
)
from uacpy.core._provenance import dedupe_provenance
from uacpy.core._carrier import (
    DeepCopyMixin, RevalidateOnAssignMixin, carrier,
)
from uacpy.core._export import CarrierExport

__all__ = [
    'SeabedColumn', 'Bottom', 'medium_density_at',
]


def medium_density_at(depth: float, tops, densities, bottom_depth: float,
                      halfspace_density: float) -> float:
    """``ρ(z)`` (g/cm³) of the medium holding ``depth`` (m) in a stack of
    media.

    Medium ``i`` spans ``tops[i]`` to the next top (to ``bottom_depth`` for
    the last) at ``densities[i]``, and ``halfspace_density`` holds below the
    stack. A depth on an interface belongs to the upper medium (the
    convention of :meth:`SeabedColumn.at`). It is the ``ρ(z_s)`` of the
    point-source modal sum (Jensen et al., *Computational Ocean Acoustics*,
    eq. 5.14): a model run builds the stack from its environment, and a mode
    set from the medium table its ``.mod`` recorded
    (:meth:`~uacpy.core.results.MediaTable.density_at`).
    """
    z = float(depth)
    tops = [float(t) for t in tops]
    for i in range(len(tops)):
        bottom = (tops[i + 1] if i + 1 < len(tops)
                  else float(bottom_depth))
        if z <= bottom:
            return float(densities[i])
    return float(halfspace_density)


# ─────────────────────────────────────────────────────────────────────────────
# Unified bottom carrier — one ``Bottom`` (range as an optional axis), mirroring
# ``SoundSpeedProfile``. A ``SeabedColumn`` is "layers over a half-space" at one
# range; an empty layer list means a pure half-space. ``Bottom`` holds one or
# more columns plus an optional ``ranges`` vector.
# ─────────────────────────────────────────────────────────────────────────────


@carrier
class SeabedColumn(RevalidateOnAssignMixin, DeepCopyMixin, CarrierExport):
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
                    f"{type(la).__name__}.")
        if not isinstance(self.halfspace, BoundaryProperties):
            raise ConfigurationError(
                "SeabedColumn: halfspace must be a BoundaryProperties; "
                f"got {type(self.halfspace).__name__}."
            )
        # The column holds copies: a layer or half-space the caller edits
        # afterwards never reaches it.
        self.layers = [_copy.deepcopy(la) for la in self.layers]
        self.halfspace = _copy.deepcopy(self.halfspace)

    @property
    def is_layered(self) -> bool:
        return len(self.layers) > 0

    @property
    def data_sources(self) -> tuple:
        """Provenance of this column — its half-space's ``data_sources``
        (harmonised with the leaf carriers; a fetched seabed stamps the
        half-space)."""
        return tuple(getattr(self.halfspace, 'data_sources', ()) or ())

    def _rows(self) -> list:
        """One row per layer, shallow first, then the half-space: depth
        below the seafloor of its ``top`` and ``bottom`` (m; ``inf`` for the
        half-space), ``layer`` (index; the half-space's is the layer
        count), ``name``, ``acoustic_type`` (``'layer'`` for a sediment
        layer) and the acoustic properties."""
        rows, top = [], 0.0
        for i, layer in enumerate(self.layers):
            rows.append({'layer': i, 'name': layer.name or '',
                         'acoustic_type': 'layer', 'top': top,
                         'bottom': top + float(layer.thickness),
                         **_property_row(layer)})
            top += float(layer.thickness)
        rows.append({'layer': len(self.layers),
                     'name': self.halfspace.name or '',
                     'acoustic_type': self.halfspace.acoustic_type,
                     'top': top, 'bottom': np.inf,
                     **_property_row(self.halfspace)})
        return rows

    def _table(self):
        """The layers, then the half-space, one row each (see
        :meth:`_rows`)."""
        return _columns(self._rows())

    def _layers_text(self) -> Optional[str]:
        """``'2 layers 30 m'`` (count and total thickness, and ``elastic``
        when a layer carries shear); ``None`` for a bare half-space."""
        if not self.layers:
            return None
        n = len(self.layers)
        text = (f"{n} layer{'s' if n > 1 else ''} "
                f"{qty(self.total_thickness(), 'm')}")
        if any(layer.shear_speed > 0 for layer in self.layers):
            text += ' elastic'
        return text

    def _repr_bits(self) -> List[str]:
        """The column in repr words: its layers, then its half-space."""
        layers = self._layers_text()
        return ([layers] if layers else []) + self.halfspace._repr_bits()

    def _short(self) -> str:
        """The column in a few words, with no comma."""
        layers = self._layers_text()
        return (f"{layers} over " if layers else '') + self.halfspace._short()

    def __repr__(self) -> str:
        return build('SeabedColumn', self._repr_bits())

    def total_thickness(self) -> float:
        """Total thickness of all sediment layers (m); 0 for a half-space."""
        return sum(layer.thickness for layer in self.layers)

    def __setattr__(self, name, value):
        # Writes to a half-space field follow through to ``halfspace``. A plain
        # assignment would create an instance attribute that echoes the new
        # value back while ``at()``, RAM's seabed samples, the repr and every
        # writer — all of which read ``halfspace`` — keep the previous one.
        if name in _HALFSPACE_DELEGATED and 'halfspace' in self.__dict__:
            _delegate_write(type(self).__name__, [self.halfspace], name,
                            value, layered=bool(self.layers))
            return
        super().__setattr__(name, value)

    def layer_at(self, depth: float) -> Optional[SedimentLayer]:
        """The :class:`SedimentLayer` containing sub-bottom ``depth`` (m,
        ``0`` = top of the column), or ``None`` below the stack, in the deep
        half-space.

        A depth exactly on an internal boundary maps to the **upper** layer,
        and the bottom of the stack to the deepest layer. :meth:`at` builds
        on it (depth → material); :meth:`isel` is the positional
        counterpart (index → layer).

        Parameters
        ----------
        depth : float
            Sub-bottom depth (m).
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

        Parameters
        ----------
        depth : float
            Sub-bottom depth (m), a finite scalar.
        """
        layer = self.layer_at(depth)
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
        ``self.halfspace``.)

        Parameters
        ----------
        layer : int
            Layer index.
        """
        i = int(layer)
        n = len(self.layers)
        if not -n <= i < n:
            raise IndexError(
                f"SeabedColumn.isel: layer index {i} out of range for "
                f"{n} layer(s)")
        return _copy.deepcopy(self.layers[i])

    def layer_depths(self, seafloor_depth: float) -> List[Tuple[float, float]]:
        """``(top, bottom)`` depth pairs for each layer (empty for a
        half-space). ``seafloor_depth`` is the top of the first layer (m).

        Parameters
        ----------
        seafloor_depth : float
            Depth (m) of the top of the first layer.
        """
        depths = []
        current = seafloor_depth
        for layer in self.layers:
            top = current
            bottom = current + layer.thickness
            depths.append((top, bottom))
            current = bottom
        return depths

    def collapse_layers(self, method: str = 'halfspace') -> BoundaryProperties:
        """Collapse the column to a single ``BoundaryProperties``.

        ``'halfspace'`` → the deep half-space; ``'top_layer'`` → topmost
        layer's properties (half-space when there are no layers);
        ``'volume_average'`` → thickness-weighted **arithmetic** mean of
        ``sound_speed``, ``density``, ``attenuation``, ``shear_speed`` and
        ``shear_attenuation`` (m/s, g/cm³, dB/λ, averaged as plain numbers)
        over the finite layers **and the half-space**, the half-space carrying
        the deepest layer's thickness as its weight (it has none of its own).
        It is a bookkeeping quantity with no acoustic basis: no effective-
        medium theory averages wave speeds linearly, and the half-space has no
        thickness to weight. Wood's rule — averaging ρ and the compliance
        1/(ρc²) over the same two metres — gives about 1720 m/s for 1 m of
        1500 m/s, 1.4 g/cm³ mud over a 5250 m/s, 2.7 g/cm³ basalt half-space,
        where this method returns 3375 m/s and 2.05 g/cm³: a thin layer over a
        fast basement lands between the two, not on the layer. Reach for
        ``'top_layer'`` when the layer is what the field sees, or a model that
        keeps the stack (``supports_feature('layered_bottom')``). Half-space
        alone when there are no layers.

        The half-space is the template for the non-blendable fields, so a
        ``'vacuum'`` / ``'rigid'`` column collapses back to that parameter-free
        type carrying only its ``roughness``.

        Parameters
        ----------
        method : {'halfspace', 'top_layer', 'volume_average'}, optional
            The reduction (see above). Default ``'halfspace'``.
        """
        if method not in COLUMN_COLLAPSE_METHODS:
            raise ConfigurationError(
                f"SeabedColumn.collapse_layers: unknown method={method!r}; "
                f"valid: {_method_list(COLUMN_COLLAPSE_METHODS)}."
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
        halfspace_type = BoundaryType.from_string(self.halfspace.acoustic_type)
        if not halfspace_type.is_geoacoustic:
            if halfspace_type.is_parameter_free:
                outcome = (f"the {len(self.layers)} sediment layer(s)' "
                           f"properties are dropped (only the interfacial "
                           f"roughness survives)")
            else:
                outcome = (f"the values reduced from the {len(self.layers)} "
                           f"sediment layer(s) are kept on the returned "
                           f"boundary but the solver reads the "
                           f"reflection-coefficient file and ignores them")
            warnings.warn(
                f"SeabedColumn.collapse_layers({method!r}): the half-space is "
                f"'{self.halfspace.acoustic_type}' — the solver reads no "
                f"geoacoustic parameters from it, so {outcome}.",
                FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
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

    @classmethod
    def from_halfspace(cls, halfspace: BoundaryProperties) -> 'SeabedColumn':
        """A pure half-space column (no sediment layers) over a copy of
        ``halfspace``.

        Parameters
        ----------
        halfspace : BoundaryProperties
            The column's deep half-space (copied).
        """
        return cls(layers=[], halfspace=_copy.deepcopy(halfspace))

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
        ``(name, thickness, overrides)``; ``halfspace`` is a preset name.

        Parameters
        ----------
        layers : sequence of tuple
            ``(name, thickness)`` or ``(name, thickness, overrides)`` per layer,
            shallowest first.
        halfspace : str
            The half-space preset.
        halfspace_overrides : dict, optional
            Properties replacing the half-space preset's.
        elastic : bool, optional
            Keep the presets' shear properties. Default False.
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
                    "SeabedColumn.from_presets: layer entry must be "
                    f"(name, thickness) or (name, thickness, overrides); "
                    f"got {entry!r}."
                )
            sediment_layers.append(
                SedimentLayer.from_preset(name, thickness=thickness,
                                          elastic=elastic, **overrides)
            )
        hs = BoundaryProperties.from_preset(
            halfspace, elastic=elastic, **(halfspace_overrides or {}))
        return cls(layers=sediment_layers, halfspace=hs)


# eq=False: a dataclass __eq__ over ndarray fields raises; compare by identity.
@carrier(eq=False)
class Bottom(RevalidateOnAssignMixin, DeepCopyMixin, CarrierExport):
    """Unified seabed carrier — one or more :class:`SeabedColumn` columns with
    an optional ``ranges`` axis (metres). ``ranges=None`` ⇒ range-independent
    (exactly one column), mirroring ``SoundSpeedProfile(ranges=None)``.

    Construct via the ``from_*`` factories or let ``Environment(bottom=...)``
    coerce a scalar cp, preset name, ``BoundaryProperties``, ``SeabedColumn``
    or ``Bottom``.

    `Bottom` and :class:`~uacpy.core.surface.Surface` are the two carriers
    without a ``.plot()``: placing the sub-bottom depth axis needs a seafloor
    depth, which lives on ``env.bathymetry``. Plot one with
    ``uacpy.plot.plot_bottom_properties(env)``.

    Every accessor — :meth:`at`, :meth:`isel`, :meth:`halfspace_at` — and
    every reduction — :meth:`collapse_range`, :meth:`collapse`,
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
            raise ConfigurationError("Bottom: requires at least one SeabedColumn.")
        for c in self.columns:
            if not isinstance(c, SeabedColumn):
                raise ConfigurationError(
                    "Bottom: columns must be SeabedColumn instances; got "
                    f"{type(c).__name__}.")
        # The bottom holds copies: a column the caller edits afterwards
        # never reaches it.
        self.columns = [_copy.deepcopy(c) for c in self.columns]
        if self.ranges is None:
            if len(self.columns) != 1:
                raise ConfigurationError(
                    "Bottom: range-independent bottom (ranges=None) needs "
                    f"exactly one column; got {len(self.columns)}.")
        else:
            # Ahead of the float64 cast below, which discards an imaginary
            # part — see _reject_complex for the two ways it does it.
            reject_complex(self.ranges, "Bottom.ranges")
            self.ranges = np.array(self.ranges, dtype=float).ravel()
            require_non_negative(self.ranges, "Bottom.ranges", hint="metres")
            require_strictly_increasing(
                self.ranges, "Bottom.ranges", min_step=DECK_RANGE_RESOLUTION_M)
            if len(self.ranges) != len(self.columns):
                raise ConfigurationError(
                    f"Bottom: ranges length ({len(self.ranges)}) must match "
                    f"columns length ({len(self.columns)})")

    # ── queries (replace isinstance dispatch) ──────────────────────────────
    @property
    def data_sources(self) -> tuple:
        """Every column's provenance, exact repeats removed (harmonised with
        the leaf carriers and ``env.data_sources``). On a range-dependent
        bottom each record carries its column's range as ``range_m``."""
        return dedupe_provenance(
            self.columns,
            ranges=self.ranges if len(self.columns) > 1 else None)

    @property
    def is_range_dependent(self) -> bool:
        """True when the bottom carries more than one range column.

        A structural test (node count on the ranged axis), like
        ``SoundSpeedProfile`` / ``Surface``: columns with identical
        properties still count as range-dependent. Contrast
        ``Bathymetry.varies_with_range`` / ``Altimetry.varies_with_range``,
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

    def _table(self):
        """Every column's layers and half-space, one row each, with the
        ``range`` (m) of the column they belong to first; see
        :meth:`SeabedColumn._rows` for the other columns."""
        ranges = (np.zeros(len(self.columns)) if self.ranges is None
                  else np.asarray(self.ranges, dtype=float))
        return _columns([{'range': float(r), **row}
                         for r, column in zip(ranges, self.columns)
                         for row in column._rows()])

    def __repr__(self) -> str:
        if not self.is_range_dependent:
            return build('Bottom', self.columns[0]._repr_bits())
        return build('Bottom', [axis(self.ranges, 'ranges', 'm'),
                                'layered' if self.is_layered else 'half-space',
                                'elastic' if self.is_elastic else None])

    def _short(self) -> str:
        """The seabed in a few words, with no comma."""
        if not self.is_range_dependent:
            return self.columns[0]._short()
        return (f"{len(self.columns)} columns "
                f"{'layered' if self.is_layered else 'half-space'}")

    # ── slicing ─────────────────────────────────────────────────────────────
    def _nearest_index(self, range: float) -> int:
        return nearest_index_on_axis(self.ranges, range)

    def at(self, *, range: float) -> SeabedColumn:
        """Copy of the nearest :class:`SeabedColumn` to ``range`` (m).

        Always nearest — layer stacks cannot be linearly blended, so a `Bottom`
        has no general ``eval`` (unlike ``SoundSpeedProfile``/``Field``). For
        the one blendable quantity, the half-space of a pure half-space
        bottom, :meth:`halfspace_at` offers an opt-in ``interp='linear'``.
        Positional counterpart: :meth:`isel`.

        ``range`` must be a finite scalar — a NaN/inf or array-valued label
        raises ``ConfigurationError``, the same contract ``Field.at`` applies.

        Parameters
        ----------
        range : float
            Range (m), a finite scalar.
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

        Parameters
        ----------
        range : float
            Range (m), a finite scalar.
        """
        return self._nearest_index(range)

    def isel(self, *, range: int) -> SeabedColumn:
        """Copy of the :class:`SeabedColumn` at integer position ``range`` —
        the positional counterpart of :meth:`at`.

        Parameters
        ----------
        range : int
            Column index.
        """
        i = int(range)
        n = len(self.columns)
        if not -n <= i < n:
            raise IndexError(
                f"Bottom.isel: range index {i} out of range for "
                f"{n} column(s)")
        return _copy.deepcopy(self.columns[i])

    def halfspace_at(self, *, range: float,
                     interp: Optional[str] = None) -> BoundaryProperties:
        """Half-space ``BoundaryProperties`` at ``range`` (m).

        The default (``interp=None`` or ``'nearest'``) reads the NEAREST
        column's half-space intact — the same step rule as :meth:`at` and
        :meth:`column_index_at`, switching midway between consecutive
        ``ranges`` nodes, which is where every engine deck (Bellhop long
        ``.bty``, Kraken segments, RAM profiles) places the switch.
        ``interp='linear'`` blends the properties between the two bracketing
        columns; it is only defined when every column is a pure
        ``'half-space'``, and it takes the non-blendable fields
        (``acoustic_type``, ``reflection_file``, ``grain_size_phi``) from the
        r = 0 column. ``range`` must be a finite scalar on both paths.

        Parameters
        ----------
        range : float
            Range (m), a finite scalar.
        interp : {None, 'nearest', 'linear'}, optional
            Nearest column (default) or a blend of the two bracketing ones.
        """
        if interp not in (None, 'linear', 'nearest'):
            raise ConfigurationError(
                f"Bottom.halfspace_at: interp must be 'linear', 'nearest' or "
                f"None; got {interp!r}.")
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
        if self.ranges is None or interp in (None, 'nearest'):
            return _copy.deepcopy(
                self.columns[self._nearest_index(range)].halfspace)
        if not blendable:
            types = sorted({c.halfspace.acoustic_type for c in self.columns})
            raise ConfigurationError(
                f"Bottom.halfspace_at(interp='linear') needs every column to "
                f"be a pure 'half-space' to blend; got {types}"
                f"{' with sediment layers' if self.is_layered else ''}. Use "
                f"interp='nearest' (the default) to read the nearest column "
                f"intact.")
        return _reduce_boundaries(
            [c.halfspace for c in self.columns],
            lambda values: np.interp(label, self.ranges, values))

    def total_thickness_max(self) -> float:
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
            if BoundaryType.from_string(
                    c.halfspace.acoustic_type).is_geoacoustic:
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
    def collapse_range(self, method: str = 'r0') -> 'Bottom':
        """Reduce the range axis to a single column (range-independent result).

        ``'r0'`` / ``'rmax'`` pick the first / last column (layers kept).
        ``'mean'`` / ``'median'`` numerically average the half-spaces over
        the columns, each column weighing the same whatever its range
        spacing — only meaningful when no column is layered. For a layered bottom ``'median'``
        falls back to picking the middle column (layers can't be averaged) and
        ``'mean'`` is rejected. Uniform ``'file'`` / ``'precalc'`` columns
        carry no real numbers to average: they collapse to their shared
        reflection file (reducing only the roughness), and differing files
        are rejected.

        The picking methods return a **copy** of the chosen column, matching
        :meth:`at` / :meth:`isel` / :meth:`halfspace_at`; the averaging ones
        build a new half-space and never held the parent's to begin with.

        Parameters
        ----------
        method : {'r0', 'rmax', 'mean', 'median'}, optional
            The reduction (see above). Default ``'r0'``.
        """
        # Validated before the early return: a range-independent bottom has
        # nothing to reduce, and returning first made a typo silent until the
        # user switched to a range-dependent environment.
        if method not in RANGE_COLLAPSE_METHODS:
            raise ConfigurationError(
                f"Bottom.collapse_range: unknown method={method!r}; "
                f"valid: {_method_list(RANGE_COLLAPSE_METHODS)}.")
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
        if self.is_layered:
            if method == 'median':
                return Bottom(
                    columns=[_copy.deepcopy(
                        self.columns[len(self.columns) // 2])],
                    ranges=None)
            raise ConfigurationError(
                "Bottom.collapse_range('mean') is undefined for a layered "
                "bottom (layer stacks can't be averaged); use 'r0', 'rmax' "
                "or 'median'.")
        halfspace = _reduce_uniform_nodes(
            [c.halfspace for c in self.columns], method,
            'Bottom.collapse_range', 'columns')
        return Bottom(columns=[SeabedColumn(layers=[], halfspace=halfspace)],
                      ranges=None)

    def collapse(self, *, range: Optional[str] = None,
                 layers: Optional[str] = None) -> 'Bottom':
        """Reduce along one or both axes, returning a new ``Bottom``.

        ``range=`` collapses the range axis (see :meth:`collapse_range`).
        ``layers=`` flattens each column's layers to a half-space (per-column,
        keeping the range axis), via :meth:`SeabedColumn.collapse_layers`.

        Parameters
        ----------
        range : str, optional
            Method collapsing the range axis (:meth:`collapse_range`).
        layers : str, optional
            Method flattening each column's layers
            (:meth:`SeabedColumn.collapse_layers`).
        """
        b = self
        if range is not None:
            b = b.collapse_range(range)
        if layers is not None:
            new_cols = [SeabedColumn(layers=[], halfspace=c.collapse_layers(layers))
                        for c in b.columns]
            # One flattened column per input column, so the range axis is
            # untouched — including a single-node ``ranges``, which is a
            # coordinate ``env.max_range`` reads rather than an empty axis.
            b = Bottom(columns=new_cols, ranges=b.ranges)
        return b

    def to_halfspace(self, range_method: str = 'r0') -> BoundaryProperties:
        """Collapse fully to a single ``BoundaryProperties``.

        Parameters
        ----------
        range_method : str, optional
            The range reduction (:meth:`collapse_range`). Default ``'r0'``.
        """
        return self.collapse_range(range_method).columns[0].collapse_layers('halfspace')

    def __setattr__(self, name, value):
        # Writes to a half-space field follow through to every column. A plain
        # assignment would create an instance attribute that echoes the new
        # value back while ``halfspace_at()`` — and every writer and model
        # reading it — kept the stored half-spaces.
        if name in _HALFSPACE_DELEGATED and 'columns' in self.__dict__:
            _delegate_write(type(self).__name__,
                            [c.halfspace for c in self.columns], name, value,
                            layered=any(c.layers for c in self.columns))
            return
        super().__setattr__(name, value)

    @classmethod
    def coerce(cls, bottom) -> 'Bottom':
        """Coerce ``bottom=`` into a :class:`Bottom`, mirroring ``ssp=``:
        scalar cp, preset name, ``BoundaryProperties``, ``SeabedColumn`` or
        ``Bottom`` (``None`` → the default half-space).

        Parameters
        ----------
        bottom : float, str, BoundaryProperties, SeabedColumn, Bottom or None
            The value to coerce (see above).
        """
        if bottom is None:
            return cls.from_halfspace(
                BoundaryProperties(acoustic_type='half-space'))
        if isinstance(bottom, cls):
            return bottom
        if isinstance(bottom, SeabedColumn):
            return cls.from_column(bottom)
        if isinstance(bottom, BoundaryProperties):
            return cls.from_halfspace(bottom)
        # A scalar (a 0-d array included) always means "half-space at this
        # cp" — never let inference see it (a bare 1600.0 equals the resolved
        # default and would otherwise be indistinguishable from unset). A bool
        # is refused as one — the shared guard says why.
        sound_speed = scalar_or_none(bottom, lambda v: (
            f"Environment: bottom={v!r} is a bool, not a scalar sound speed "
            f"— as a scalar it would mean a {float(v):g} m/s half-space."))
        if sound_speed is not None:
            return cls.from_halfspace(BoundaryProperties(
                acoustic_type='half-space', sound_speed=sound_speed))
        if isinstance(bottom, str):
            return cls.from_halfspace(BoundaryProperties.from_preset(bottom))
        raise ConfigurationError(
            "Environment: bottom must be a Bottom, SeabedColumn, "
            "BoundaryProperties, a scalar sound speed (m/s), or a material "
            f"preset name; got {type(bottom).__name__}.")

    # ── factories (mirror SoundSpeedProfile.from_*) ─────────────────────────
    @classmethod
    def from_halfspace(cls, halfspace: BoundaryProperties) -> 'Bottom':
        """Range-independent pure half-space bottom.

        Parameters
        ----------
        halfspace : BoundaryProperties
            The half-space, at every range.
        """
        return cls(columns=[SeabedColumn(layers=[], halfspace=halfspace)],
                   ranges=None)

    @classmethod
    def from_grain_size(cls, grain_size_phi: float, *, model: str = DEFAULT_GRAIN_SIZE_MODEL,
                        hamilton_fit: Optional[str] = None,
                        roughness: float = 0.0,
                        water_sound_speed: Optional[float] = None,
                        water_density: Optional[float] = None) -> 'Bottom':
        """Range-independent half-space bottom from a mean grain size (ϕ).

        Convenience wrapper over
        :meth:`BoundaryProperties.from_grain_size`; every argument, including
        ``hamilton_fit=`` (Hamilton's continental-terrace / abyssal-hill /
        abyssal-plain fits), is forwarded unchanged.

        Parameters
        ----------
        grain_size_phi : float
            Mean grain size on the Wentworth ϕ scale.
        model : {'hamilton', 'apl-uw'}, optional
            Conversion model. Default ``'hamilton'``.
        hamilton_fit : str, optional
            Hamilton & Bachman's fit for ``'hamilton'``.
        roughness : float, optional
            RMS interface roughness (m). Default 0.
        water_sound_speed, water_density : float, optional
            In-situ seawater properties the ratios scale by.
        """
        from uacpy.core.sediment import canonical_grain_size_selection
        model, hamilton_fit = canonical_grain_size_selection(
            model, hamilton_fit, who='Bottom.from_grain_size')
        return cls.from_halfspace(BoundaryProperties.from_grain_size(
            grain_size_phi, model=model, hamilton_fit=hamilton_fit,
            roughness=roughness,
            water_sound_speed=water_sound_speed, water_density=water_density))

    @classmethod
    def from_column(cls, column: SeabedColumn) -> 'Bottom':
        """Range-independent bottom from a single column.

        Parameters
        ----------
        column : SeabedColumn
            The column, at every range.
        """
        return cls(columns=[column], ranges=None)

    @classmethod
    def from_columns(cls, columns: List[SeabedColumn],
                     ranges) -> 'Bottom':
        """Range-dependent bottom from one column per range break.

        Parameters
        ----------
        columns : list of SeabedColumn
            One column per range break.
        ranges : array_like
            The range (m) of each column.
        """
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
        ``shear_attenuation`` / ``roughness`` default to 0.

        Parameters
        ----------
        ranges : array_like
            Range breaks (m).
        sound_speed, density, attenuation : float or array_like
            Compressional speed (m/s), density (g/cm³) and attenuation
            (dB/wavelength), scalar or one per range.
        shear_speed, shear_attenuation, roughness : float or array_like, optional
            Shear speed (m/s), shear attenuation (dB/wavelength) and RMS
            roughness (m); ``None`` is 0.
        acoustic_type : str, optional
            The half-spaces' acoustic type.
        """
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
                        f"Bottom.from_halfspaces: {name} is required.")
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
        """Range-independent layered bottom from material presets.

        Parameters
        ----------
        layers : sequence of tuple
            ``(name, thickness)`` or ``(name, thickness, overrides)`` per layer,
            shallowest first.
        halfspace : str
            The half-space preset.
        halfspace_overrides : dict, optional
            Properties replacing the half-space preset's.
        elastic : bool, optional
            Keep the presets' shear properties. Default False.
        """
        return cls.from_column(SeabedColumn.from_presets(
            layers, halfspace=halfspace,
            halfspace_overrides=halfspace_overrides, elastic=elastic))
