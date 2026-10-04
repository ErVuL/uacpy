"""RAM's views of a :class:`~uacpy.core.bottom.SeabedColumn`: the
Collins-style ``(depth, value)`` breakpoints the Collins decks carry, and the
samples mpiramS's fixed sediment grid takes. They are one engine's adapters,
so they live with the engine; the carrier keeps its own queries
(:meth:`~uacpy.core.bottom.SeabedColumn.layer_at`, ``at``, ``isel``)."""
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from uacpy.core.bottom import SeabedColumn


def piecewise_breakpoints(
    column: 'SeabedColumn',
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
    depths = column.layer_depths(seafloor_depth)
    for (top, bottom), layer in zip(depths, column.layers):
        for prop in properties:
            value = float(getattr(layer, prop))
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
        hs_value = float(getattr(column.halfspace, prop, 0.0) or 0.0)
        out[prop].append((deepest_layer_bottom, hs_value))
        out[prop].append((final_depth, hs_value))
    return out


def sample_at_depths(
    column: 'SeabedColumn', n_points: int = 4,
    max_thickness: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample ``(cp, rho, attn)`` — compressional speed (m/s), density
    (g/cm³) and attenuation (dB/λ) — at ``n_points`` evenly-spaced depths in
    ``[0, max_thickness]`` (defaults to the column's own thickness). Used
    by mpiramS to map arbitrary layers onto its fixed sediment grid."""
    max_thick = (float(max_thickness) if max_thickness is not None
                 else column.total_thickness())
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
        layer = column.layer_at(d)
        src = layer if layer is not None else column.halfspace
        cp[i], rho[i], attn[i] = (src.sound_speed, src.density,
                                  src.attenuation)
    return cp, rho, attn
