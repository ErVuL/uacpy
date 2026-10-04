"""The environment projection: the collapse each feature a model does not
read gets by default, the methods each collapse key takes, the roughness
and shear helpers, and :func:`environment_projection`, which reduces an
environment to what a model reads."""

import copy as _copy
from typing import Dict

import numpy as np

from uacpy.core.bottom import Bottom
from uacpy.core.boundary import BoundaryProperties
from uacpy.core.collapse import (
    COLUMN_COLLAPSE_METHODS, DEPTH_COLLAPSE_METHODS, RANGE_COLLAPSE_METHODS,
)
from uacpy.core.environment import Bathymetry, Environment
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.surface import Surface


DEFAULT_COLLAPSE: Dict[str, str] = {
    'bathymetry': 'max',
    'ssp': 'r0',
    'bottom_range': 'r0',
    'bottom_layers': 'halfspace',
    'altimetry': 'drop',
    'surface': 'r0',
    'elastic': 'fluid',
}

# Allowed method strings per collapse key (validated at construction in
# ``PropagationModel.__init__`` so bad values fail loudly rather than deep
# inside a writer at ``run()``-time).
#
# Read from the carriers that implement each method, never spelled out here:
# this table and the carrier decide the same question from two files, so a
# literal copy lets a method be accepted at construction and refused from
# inside ``run()`` — the exact failure the construction-time check exists to
# prevent. ``'altimetry'`` and ``'elastic'`` are implemented in this module
# (``_project_environment`` drops an altimetry; ``_collapse_elastic`` below
# reads this entry back), so they are the only two written out.
VALID_COLLAPSE_METHODS: Dict[str, frozenset] = {
    'bathymetry':        frozenset(DEPTH_COLLAPSE_METHODS),
    'ssp':               frozenset(RANGE_COLLAPSE_METHODS),
    'bottom_range':      frozenset(RANGE_COLLAPSE_METHODS),
    'bottom_layers':     frozenset(COLUMN_COLLAPSE_METHODS),
    'altimetry':         frozenset({'drop'}),
    'surface':           frozenset(RANGE_COLLAPSE_METHODS),
    'elastic':           frozenset({'fluid', 'vacuum'}),
}
# Dev invariants on the collapse-policy constants (raise, not assert, so they
# survive `python -O`).
if set(VALID_COLLAPSE_METHODS) != set(DEFAULT_COLLAPSE):
    raise RuntimeError("VALID_COLLAPSE_METHODS keys must match DEFAULT_COLLAPSE keys")
if not all(DEFAULT_COLLAPSE[k] in VALID_COLLAPSE_METHODS[k] for k in DEFAULT_COLLAPSE):
    raise RuntimeError("DEFAULT_COLLAPSE values must satisfy VALID_COLLAPSE_METHODS")


def _max_roughness(boundaries) -> float:
    """Largest interfacial sigma over a list of ``BoundaryProperties``."""
    return max(
        (float(b.roughness) for b in boundaries if b is not None),
        default=0.0,
    )


def _smooth_surface(surface):
    """``surface`` with every node's roughness zeroed.

    Writes each node in ``surface.nodes`` directly: the ``Surface``
    delegated write reaches every node too, but warns on a multi-node
    surface that the broadcast flattens range dependence — advice aimed at
    users, not at this deliberate all-nodes write. Assigned rather than
    rebuilt: ``dataclasses.replace`` re-runs ``__post_init__``, which
    rejects explicit acoustic parameters on a vacuum / rigid boundary even
    when they are the values it filled in.
    """
    smoothed = _copy.deepcopy(surface)
    for node in smoothed.nodes:
        node.roughness = 0.0
    return smoothed


def _smooth_bottom(bottom):
    """``bottom`` with every interface roughness zeroed — layers and half-space."""
    smoothed = _copy.deepcopy(bottom)
    for column in smoothed.columns:
        column.halfspace.roughness = 0.0
        for layer in column.layers:
            layer.roughness = 0.0
    return smoothed


def _bottom_roughness(bottom) -> float:
    """Largest interfacial sigma anywhere in a ``Bottom``.

    A sediment layer's roughness is the interface at its top, so the seafloor
    of a layered column lives on ``layers[0]``, not on the half-space.
    """
    if bottom is None:
        return 0.0
    boundaries = []
    for column in bottom.columns:
        boundaries.append(column.halfspace)
        boundaries.extend(column.layers)
    return _max_roughness(boundaries)


def has_shear(boundary) -> bool:
    """True if ``boundary`` carries any non-zero shear speed. Accepts a
    :class:`Bottom` or a surface :class:`BoundaryProperties`."""
    if boundary is None:
        return False
    if isinstance(boundary, (Bottom, Surface)):
        return boundary.is_elastic
    return getattr(boundary, 'shear_speed', 0.0) > 0


def collapse_elastic_boundary(boundary, method: str):
    """Collapse elastic shear on ``boundary`` per ``method``.

    ``'fluid'``  : zero shear_speed and shear_attenuation; keep cp / ρ / α.
    ``'vacuum'`` : replace with a vacuum boundary.

    Accepts a :class:`Bottom` (every column's layers + half-space) or a
    surface :class:`BoundaryProperties`, returning the same kind.
    """
    def _zero_shear(b):
        b.shear_speed = 0.0
        b.shear_attenuation = 0.0

    valid = sorted(VALID_COLLAPSE_METHODS['elastic'])
    if method not in valid:
        raise ConfigurationError(
            f"Unknown elastic collapse method {method!r}. Use "
            f"{' or '.join(repr(m) for m in valid)}."
        )
    if isinstance(boundary, Bottom):
        if method == 'vacuum':
            return Bottom.from_halfspace(
                BoundaryProperties(acoustic_type='vacuum'))
        b = _copy.deepcopy(boundary)
        for col in b.columns:
            for layer in col.layers:
                _zero_shear(layer)
            _zero_shear(col.halfspace)
        return b
    if isinstance(boundary, Surface):
        if method == 'vacuum':
            return Surface(nodes=[
                BoundaryProperties(acoustic_type='vacuum')])
        b = _copy.deepcopy(boundary)
        for p in b.nodes:
            _zero_shear(p)
        return b
    if method == 'vacuum':
        return BoundaryProperties(acoustic_type='vacuum')
    b = _copy.deepcopy(boundary)
    _zero_shear(b)
    return b


def environment_projection(env: 'Environment', *, model_name: str,
                           supported, collapse):
    """``(projected, notices)``: a copy of ``env`` with every unsupported
    feature collapsed, and the text of the notice each dropped feature
    gets, in order. Pure: warns nothing, so a nested model's projection
    can be computed again without repeating what was said.

    Each per-feature axis is checked against ``supported`` (the capability
    names the model carries, :attr:`PropagationModel.supported_features`)
    and reduced via the matching key of ``collapse`` (the model's resolved
    policy); each dropped feature adds one notice.

    Notes
    -----
    The caller's ``env`` is never touched — everything happens on
    ``env.copy()``, so a user can reuse one ``Environment`` across models
    that project it differently. Only ``altimetry``, ``surface``,
    ``bathymetry``, ``ssp`` and ``bottom`` are rewritten;
    ``env.absorption`` and everything else pass through.

    Every branch *narrows* — it removes range dependence, layering or
    shear that the model cannot represent — with one deliberate
    exception: collapsing the bathymetry can deepen the seafloor, and the
    SSP is then extended to match (see below).

    Order matters (the bottom's range axis is collapsed before its layer
    axis).
    """
    e = env.copy()
    notices = []

    if e.altimetry is not None and 'altimetry' not in supported:
        method = collapse["altimetry"]
        # Defensive: unreachable through either entry point, since a
        # user's ``collapse=`` and a subclass's ``ModelSpec.collapse`` are
        # both checked against ``VALID_COLLAPSE_METHODS['altimetry']``,
        # which holds only 'drop'. Only mutating the private
        # ``_collapse`` gets here. Kept because it becomes live the day a
        # second altimetry method is added, and raising (not asserting)
        # keeps it under ``python -O``.
        if method != 'drop':
            raise ConfigurationError(
                f"Unknown collapse['altimetry']={method!r}. "
                "Currently only 'drop' is supported."
            )
        e.altimetry = None
        notices.append(
            f"{model_name} does not support sea-surface altimetry; "
            f"using flat surface (collapse['altimetry']={method!r})."
        )

    # No model consumes a range-dependent surface deck (the AT family
    # takes one TopOpt boundary, RAM one attenuator, OASES one top
    # half-space), so the collapse is unconditional; ``collapse['surface']``
    # picks the reduction method.
    if e.surface.is_range_dependent:
        method = collapse["surface"]
        e.surface = e.surface.collapse_range(method)
        notices.append(
            f"{model_name} does not support range-dependent surface "
            f"properties (e.g. a marginal ice zone); collapsed to a single "
            f"boundary (collapse['surface']={method!r})."
        )

    surf_sigma = _max_roughness(e.surface.nodes)
    if surf_sigma and 'rough_surface' not in supported:
        e.surface = _smooth_surface(e.surface)
        notices.append(
            f"{model_name} does not carry a rough sea surface into "
            f"its deck; env.surface.roughness={surf_sigma:g} m dropped. "
            f"Use Kraken with a vacuum, rigid or half-space surface, an "
            f"OASES model, or Scooter with a vacuum surface, to keep it."
        )

    bot_sigma = _bottom_roughness(e.bottom)
    if bot_sigma and 'rough_bottom' not in supported:
        e.bottom = _smooth_bottom(e.bottom)
        notices.append(
            f"{model_name} does not carry seabed interfacial "
            f"roughness into its deck; env.bottom roughness="
            f"{bot_sigma:g} m dropped. Use Kraken or an OASES model "
            f"to keep it."
        )

    if (e.bathymetry.varies_with_range
            and 'range_dependent_bathymetry' not in supported):
        method = collapse["bathymetry"]
        new_depth = e.bathymetry.collapse_range(method)
        min_d = float(e.bathymetry.depths.min())
        max_d = float(e.bathymetry.depths.max())
        # The assignment extends a profile that ends above the new
        # seafloor down to it (Environment._assignment), so a collapse
        # method that picks a deeper column than the profile was
        # tabulated for (``'max'``, the default, on a sloping bottom)
        # leaves no gap the AT writers would reject.
        e.bathymetry = Bathymetry(
            ranges=np.array([0.0]), depths=np.array([new_depth]))
        notices.append(
            f"{model_name} does not support range-dependent "
            f"bathymetry; collapsed to {new_depth:.1f} m "
            f"(method={method!r}, range {min_d:.1f}–{max_d:.1f} m). "
            f"Override via `collapse={{'bathymetry': "
            f"{'|'.join(repr(m) for m in DEPTH_COLLAPSE_METHODS)}}}`."
        )

    if e.ssp.is_range_dependent and 'range_dependent_ssp' not in supported:
        method = collapse["ssp"]
        e.ssp = e.ssp.collapse_range(method)
        notices.append(
            f"{model_name} does not support range-dependent SSP; "
            f"collapsed to 1-D (collapse['ssp']={method!r})."
        )

    # Bottom: two orthogonal axes. Collapse the range axis first (to a
    # single column, keeping its layers), then flatten the layer axis if
    # the model can't take layers — leaving, for a model that supports RD
    # but not layers (Bellhop), a range-dependent half-space bottom.
    if (e.bottom.is_range_dependent
            and 'range_dependent_bottom' not in supported):
        method = collapse["bottom_range"]
        e.bottom = e.bottom.collapse_range(method)
        notices.append(
            f"{model_name} does not support range-dependent bottoms; "
            f"reduced to a single column "
            f"(collapse['bottom_range']={method!r})."
        )

    if e.bottom.is_layered and 'layered_bottom' not in supported:
        method = collapse["bottom_layers"]
        e.bottom = e.bottom.collapse(layers=method)
        notices.append(
            f"{model_name} does not support layered (depth-dependent) "
            f"bottoms; flattened each column to a half-space "
            f"(collapse['bottom_layers']={method!r})."
        )

    if 'elastic_media' not in supported:
        collapsed_at = []
        if e.surface is not None and has_shear(e.surface):
            e.surface = collapse_elastic_boundary(
                e.surface, collapse["elastic"],
            )
            collapsed_at.append('surface')
        if e.bottom.is_elastic:
            e.bottom = collapse_elastic_boundary(
                e.bottom, collapse["elastic"],
            )
            collapsed_at.append('bottom')
        if collapsed_at:
            method = collapse["elastic"]
            where = '/'.join(collapsed_at)
            notices.append(
                f"{model_name} does not support elastic media; "
                f"collapsed shear properties on {where} "
                f"(collapse['elastic']={method!r})."
            )

    return e, tuple(notices)
