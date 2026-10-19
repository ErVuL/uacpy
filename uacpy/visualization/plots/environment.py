"""Ocean-environment cross-section plot (SSP, bathymetry, bottom layers)."""

from __future__ import annotations


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MaxNLocator
from typing import Optional, Tuple

from uacpy.core.constants import DEFAULT_SOUND_SPEED
from uacpy.core.absorption import Absorption, AbsorptionCoefficient
from uacpy.core.environment import Environment
from uacpy.core.ssp import SoundSpeedProfile
from uacpy.visualization.style import (
    BOTTOM_FILL_STYLE, BOTTOM_CMAP, BOTTOM_HALFSPACE_COLOR, BOTTOM_LINE_STYLE,
    BOTTOM_LINE_STYLE_FLAT, SURFACE_LINE_COLOR, hatched_fill,
)
from uacpy.visualization.plots._common import _plot_warn, ZORDER_SEDIMENT, _colorbar_extend, _credit_attributions, _draw_credit, _draw_geometry, _draw_sea_ice, _draw_surface_boundary, _draw_altimetry, _fill_margins, fig_ax, typed_plot_error, invert_yaxis_once, _title_or, _grid_figure
from uacpy.core.exceptions import ConfigurationError, FallbackWarning
from uacpy.core.units import km_to_m, m_to_km


# A single half-space is shaded by sound speed on a fixed (absolute) scale so
# different half-spaces read differently, while the '///' hatch still flags the
# region as the semi-infinite half-space (vs the solid-filled finite layers).
# The scale is a display choice (uacpy guidance) that spreads the
# unconsolidated sediments of Jensen et al., Computational Ocean Acoustics,
# Table 1.3 (clay 1500 → moraine 1950 m/s) across the ramp instead of
# clustering them at the soft end. The right-margin 'Bottom cp' colorbar spans
# exactly this scale, so the fill reads back to a cp value; a speed off the
# scale (the same table's chalk 2400, limestone 3000, basalt 5250 m/s, or a
# seabed slower than 1450 m/s) takes the end colour, and the bar ends in an
# arrow on that side.
_HALFSPACE_CP_LO, _HALFSPACE_CP_HI = 1450.0, 2300.0

# The sound-speed colourbar labels, on two lines: the stacked water/seabed
# bars are each under half the panel tall, and a one-line vertical label
# longer than its bar runs into the bar beside it.
WATER_C_LABEL = 'Water c\n(m/s)'
BOTTOM_CP_LABEL = 'Bottom cp\n(m/s)'

# The panel shows the water column plus a margin below the deepest seafloor, so
# the seabed reads as a band rather than a line. Every bottom branch paints down
# to at least ``_panel_floor`` and the depth limit sits on it, so the margin is
# the same whatever shape the bottom has. 10 % of the water depth: the margin
# is dead space that a thin sediment stack shares the panel with, and at 20 %
# the layers are too few pixels to read (uacpy guidance). The 5 m floor keeps a
# visible band under a shallow seabed, where 10 % is a couple of metres.
_PANEL_MARGIN = 0.10
_PANEL_MARGIN_MIN_M = 5.0


def _panel_floor(seafloor_max) -> float:
    """Depth the panel extends to under a seabed whose deepest point is
    ``seafloor_max``."""
    z = float(seafloor_max)
    return z + max(_PANEL_MARGIN_MIN_M, _PANEL_MARGIN * z)


def _bottom_kind(bottom) -> str:
    """Figure-title description of a ``Bottom``'s shape — the two axes a reader
    needs to know which rendering branch produced the panel. Phrased like
    :meth:`Bottom.__repr__`."""
    kind = "layered" if bottom.is_layered else "half-space"
    if bottom.is_range_dependent:
        return f"range-dependent {kind}, {bottom.n_ranges} ranges"
    return kind


def _halfspace_cp(bottom) -> float | None:
    """The half-space sound speed to shade by, or ``None`` when there is no
    meaningful value (vacuum / rigid / reflection-file boundary)."""
    if bottom is None or bottom.acoustic_type in ('vacuum', 'rigid', 'file'):
        return None
    cp = getattr(bottom, 'sound_speed', None)
    return cp if cp and cp > 0 else None


def _speeds_read_absolute(colorbar):
    """Print a sound-speed colorbar's ticks as absolute m/s: a nearly constant
    profile otherwise draws 0.0/0.4/0.8 over a "+1.5e3" offset."""
    colorbar.ax.yaxis.get_major_formatter().set_useOffset(False)


def _truncated(cmap, lo, hi, n=256):
    """The ``lo``–``hi`` band of a colormap as a colormap of its own."""
    base = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    return LinearSegmentedColormap.from_list(
        f"{getattr(base, 'name', 'cmap')}_clip", base(np.linspace(lo, hi, n)),
    )


def _make_sm(cs_values, cmap):
    """``(cs_min, cs_max, mappable)`` spanning the sound speeds in
    ``cs_values`` — the scale a colorbar for them is drawn on."""
    pool = list(cs_values) if len(cs_values) else [DEFAULT_SOUND_SPEED]
    cs_min = float(min(pool))
    cs_max = float(max(pool))
    if cs_max <= cs_min:
        # A constant profile gets a window centred on its value.
        # Relative, not +1 m/s absolute: ticks over 1500–1501 need six
        # significant digits, which the colorbar prints as 0.0/0.4/0.8
        # over a "+1.5e3" offset.
        pad = 0.05 * abs(cs_min) if cs_min else 0.5
        cs_min, cs_max = cs_min - pad, cs_max + pad
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=cs_min, vmax=cs_max))
    sm.set_array([])
    return cs_min, cs_max, sm


def _voronoi_edges(nodes_km, lo, hi):
    """Cell edges of range nodes: the midpoints between neighbours, with the
    outer ends at ``lo`` / ``hi``."""
    return [lo, *(0.5 * (nodes_km[:-1] + nodes_km[1:])), hi]


def _fill_layer_stack(ax, x, z_top, layers, cmap, cs_min, cs_range, *,
                      linewidth, dashed_edges):
    """Paint ``layers`` downward from the seafloor profile ``z_top`` (one
    depth per ``x``), each shaded by its sound speed on the caller's cp scale,
    with a dashed line under each layer when ``dashed_edges``. Returns the
    base of the stack."""
    z_top = np.asarray(z_top, dtype=float)
    for layer in layers:
        z_bot = z_top + layer.thickness
        colour = cmap(0.25 + 0.6 * (layer.sound_speed - cs_min) / cs_range)
        ax.fill_between(x, z_top, z_bot, color=colour, alpha=1.0,
                        edgecolor='black', linewidth=linewidth,
                        zorder=ZORDER_SEDIMENT + 1)
        if dashed_edges:
            ax.plot(x, z_bot, color='black', linewidth=0.8, linestyle='--',
                    alpha=0.5, zorder=ZORDER_SEDIMENT + 2)
        z_top = z_bot
    return z_top


def _halfspace_fill_style(cp) -> dict:
    """Absolute-scale half-space fill — used for a *single* half-space (no
    per-plot bottom colorbar): flat tan when ``cp`` is ``None`` (vacuum / rigid
    / file), else an earthy BOTTOM_CMAP shade mapped from ``cp`` (m/s) on a fixed scale, so
    different half-spaces read differently across plots. Keeps the hatch."""
    if cp is None:
        return dict(BOTTOM_FILL_STYLE)
    t = float(np.clip((cp - _HALFSPACE_CP_LO)
                      / (_HALFSPACE_CP_HI - _HALFSPACE_CP_LO), 0.0, 1.0))
    return hatched_fill(BOTTOM_CMAP(0.25 + 0.6 * t))


def _layered_halfspace_style(hs, cmap, cs_min, cs_range) -> dict:
    """Basement-half-space fill for a *layered* bottom: shaded on the SAME
    (relative) ``cmap``/range as the sediment layers so it agrees with the
    plot's 'Bottom cp' colorbar; flat tan for vacuum / rigid / file."""
    cp = _halfspace_cp(hs)
    if cp is None:
        return dict(BOTTOM_FILL_STYLE)
    norm = min(1.0, max(0.0, (cp - cs_min) / cs_range))
    return hatched_fill(cmap(0.25 + 0.6 * norm))


def _draw_layered_bottom(ax_bathy, column, r_km, seafloor, z_max_layer,
                        _layer_cmap_and_norm):
    # Per-layer fills (earthy BOTTOM_CMAP by sound speed) + dashed inter-layer
    # edges + hatched half-space, keyed by the caller's right-margin
    # 'Bottom cp' colorbar. Same visual template as the range-dependent
    # layered branch below.
    cmap, cs_min, cs_max = _layer_cmap_and_norm()
    cs_range = max(1e-9, cs_max - cs_min)
    # Every layer rides the seafloor, as in the range-dependent layered branch.
    # A scalar top anchored at the deepest bathymetry point would detach the
    # stack from a sloping seabed and leave the water colormap painted in the
    # gap below the drawn seafloor line.
    z_top = _fill_layer_stack(
        ax_bathy, r_km,
        np.broadcast_to(np.asarray(seafloor, dtype=float), np.shape(r_km)),
        column.layers, cmap, cs_min, cs_range, linewidth=0.4, dashed_edges=True)
    hs = column.halfspace
    # The half-space reaches at least the panel floor, so no axis is left
    # blank below the sediment. Extended by a margin taken from the stack
    # alone, a 2 m seabed under 1500 m of water stopped at 1512 m on a 1650 m
    # panel and the bottom read as absent.
    hs_display = np.maximum(z_top + max(10.0, column.total_thickness() * 0.3),
                            _panel_floor(z_max_layer))
    ax_bathy.fill_between(
        r_km, z_top, hs_display, zorder=ZORDER_SEDIMENT,
        **_layered_halfspace_style(hs, cmap, cs_min, cs_range),
    )

    return float(np.max(hs_display))


def _draw_rdl_bottom(ax_bathy, bottom, r_km, seafloor, z_max_layer,
                     env, _layer_cmap_and_norm):
    # Geological cross-section: one column per profile range, dashed
    # vertical boundaries between columns, P# labels above each
    # column, hatched half-space at the column bottom.
    prof_ranges_km = m_to_km(np.asarray(bottom.ranges, dtype=float))
    # Voronoi cell edges: midpoints between consecutive nodes, with the outer
    # ends clamped to the bathymetry extent — as the range-dependent half-space
    # sibling does. ``Bottom.at()`` holds the first/last column outside the
    # profile nodes, so anchoring the outer edges on those nodes instead would
    # leave the section beyond them bare while the model still uses that column.
    r_lo_panel = float(np.min(r_km)) if np.size(r_km) else prof_ranges_km[0]
    r_hi_panel = float(np.max(r_km)) if np.size(r_km) else prof_ranges_km[-1]
    boundaries = _voronoi_edges(prof_ranges_km,
                                min(r_lo_panel, float(prof_ranges_km[0])),
                                max(r_hi_panel, float(prof_ranges_km[-1])))

    cmap, cs_min, cs_max = _layer_cmap_and_norm()
    cs_range = max(1e-9, cs_max - cs_min)

    max_thickness = max(
        (sum(layer.thickness for layer in prof.layers)
         for prof in bottom.columns), default=0.0,
    )
    # Band below the deepest column, sized like the range-independent layered
    # sibling's, then held to the panel floor so the margin matches every other
    # bottom shape.
    hs_extension = max(10.0, max_thickness * 0.3)
    hs_floor = max(z_max_layer + max_thickness + hs_extension,
                   _panel_floor(z_max_layer))

    total_span = prof_ranges_km[-1] - prof_ranges_km[0]
    for i_r, (r_node, prof) in enumerate(zip(prof_ranges_km,
                                             bottom.columns)):
        r_lo, r_hi = boundaries[i_r], boundaries[i_r + 1]
        n_pts = max(20, int(401 * (r_hi - r_lo) / max(total_span, 1e-9)))
        x_bin = np.linspace(r_lo, r_hi, n_pts)
        # Each layer follows the sloping seafloor across the column.
        z_top_arr = (np.interp(x_bin, r_km, seafloor)
                     if r_km.size > 1
                     else np.full_like(x_bin, env.depth))
        z_top_arr = _fill_layer_stack(
            ax_bathy, x_bin, z_top_arr, prof.layers, cmap, cs_min, cs_range,
            linewidth=0.3, dashed_edges=False)
        # Hatched half-space below this column, shaded by its basement cp.
        ax_bathy.fill_between(
            x_bin, z_top_arr, np.full_like(x_bin, hs_floor),
            zorder=ZORDER_SEDIMENT,
            **_layered_halfspace_style(prof.halfspace, cmap, cs_min, cs_range),
        )
        label_x = 0.5 * (r_lo + r_hi)
        ax_bathy.text(
            label_x, hs_floor * 0.02, f'P{i_r + 1}',
            ha='center', va='top', fontsize='small',
            fontweight='bold', color='dimgray',
            zorder=20,
            bbox=dict(boxstyle='round,pad=0.2',
                      facecolor='white', alpha=0.95,
                      edgecolor='none'),
        )
    # Dashed range-boundary lines between columns.
    for b in boundaries[1:-1]:
        ax_bathy.axvline(b, color='black', linewidth=1.0, alpha=0.6,
                         linestyle='--', zorder=ZORDER_SEDIMENT + 4)

    z_max_layer = hs_floor
    return z_max_layer


def _draw_rd_bottom(ax_bathy, bottom, r_km, seafloor, z_max_layer,
                    _layer_cmap_and_norm):
    # A range-dependent half-space: properties vary with range, uniform with
    # depth. So each node colours its whole column (seafloor → floor) by its
    # sound speed, hatched like any half-space, with Voronoi boundaries: each
    # node owns from the midpoint with its left neighbour to the midpoint with
    # its right; outer nodes reach the bathymetry edges. Tops follow the
    # seafloor so kinks are honoured. The speed scale is the caller's
    # 'Bottom cp' colorbar scale, as for the layered bottoms, so a column's
    # colour reads back to its cp on the bar — nodes sharing one speed sit at
    # the bar's padded mid-scale, not at the light end of a zero-width range.
    bot_r_km = m_to_km(np.asarray(bottom.ranges, dtype=float))
    bathy_r = r_km
    bathy_z = seafloor
    cs = np.asarray(bottom.halfspace_sound_speed, dtype=float)
    cmap, cs_min, cs_max = _layer_cmap_and_norm()
    cs_range = max(1e-9, cs_max - cs_min)
    hs_floor = _panel_floor(z_max_layer)

    # Voronoi cell edges, clamped to the bathymetry extent at the outer ends.
    edges = _voronoi_edges(bot_r_km, float(bathy_r.min()), float(bathy_r.max()))

    for i in range(len(bot_r_km)):
        r_lo = float(edges[i])
        r_hi = float(edges[i + 1])
        if r_hi <= r_lo:
            continue
        inside = (bathy_r > r_lo) & (bathy_r < r_hi)
        poly_r_top = np.concatenate(
            ([r_lo], bathy_r[inside], [r_hi])
        )
        poly_z_top = np.concatenate(
            ([float(np.interp(r_lo, bathy_r, bathy_z))],
             bathy_z[inside],
             [float(np.interp(r_hi, bathy_r, bathy_z))])
        )
        poly_r = np.concatenate([poly_r_top, poly_r_top[::-1]])
        poly_z = np.concatenate(
            [poly_z_top, np.full(poly_r_top.shape, hs_floor)]
        )
        colour = cmap(0.25 + 0.6 * (cs[i] - cs_min) / cs_range)
        ax_bathy.fill(poly_r, poly_z, zorder=ZORDER_SEDIMENT + 1,
                      **hatched_fill(colour, linewidth=0.3))

    layer_top = np.interp(bot_r_km, bathy_r, bathy_z)
    ax_bathy.plot(bot_r_km, layer_top, 'k.',
                  markersize=6, zorder=ZORDER_SEDIMENT + 5)
    for r_node in bot_r_km:
        ax_bathy.axvline(r_node, color='gray', linewidth=0.6,
                         linestyle='--', alpha=0.5,
                         zorder=ZORDER_SEDIMENT + 3)
    z_max_layer = hs_floor
    return z_max_layer


def _draw_halfspace_bottom(ax_bathy, halfspace, r_km, seafloor, z_max_layer):
    zmax_plot = _panel_floor(z_max_layer)
    # Single half-space shaded by its sound speed (the '///' hatch keeps the
    # half-space signature); flat tan for vacuum / rigid / file (no cp value).
    cp = _halfspace_cp(halfspace)
    ax_bathy.fill_between(r_km, seafloor, zmax_plot,
                          zorder=ZORDER_SEDIMENT,
                          **_halfspace_fill_style(cp))
    z_max_layer = zmax_plot
    return z_max_layer


@typed_plot_error
def plot_environment(
    env: Environment,
    ax=None,
    *,
    source=None,
    receiver=None,
    show_bottom_colorbar: bool = True,
    show_data_credit=True,
    sea_ice=None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
    x_max=None,
    source_marker_range: float = 0.0,
):
    """Single-panel water column + bottom structure with two colorbars.

    The water column is colour-mapped by SSP (Blues) and the bottom
    rendering depends on ``env.bottom``:

    * half-space column — half-space fill.
    * layered column — coloured per-layer fills (earthy ground
      colormap) + hatched half-space.
    * range-dependent half-space — Voronoi-tiled solid-colour bands
      under the seafloor, one per range node, each switching midway between
      two nodes; a dot on the seafloor and a grey dashed vertical line mark
      each node itself.
    * range-dependent layered — one column per profile, each column
      drawing the layer stack at that range, a dashed line between columns.

    Everything is coloured by ``sound_speed`` (water ``c``, seabed ``cp``); for
    the seabed's other geoacoustic properties (cs, ρ, αp, αs) use
    :func:`plot_bottom_properties`.

    Two colorbars, so neither is washed out by the other: ``Water c`` (Blues)
    on the profile's own speed range, and ``Bottom cp`` (earthy brown) on the
    seabed's own cp range for a layered or range-dependent bottom, or on a
    fixed 1450–2300 m/s display scale for a single half-space, so different
    half-spaces read differently. A half-space faster or slower than that
    scale takes its end colour and the bar ends in an arrow on that side.

    Pass ``ax=`` to draw into an existing axis (for composite figures); returns
    ``(fig, ax)``. The two colorbars are insets just right of the panel and
    take none of its width, so the panel keeps the full width of its slot and
    lines up with its neighbours; they therefore need the room a layout pass
    gives them — create the figure with ``layout='constrained'`` or call
    ``fig.tight_layout()`` — or they print over the panel to the right.
    ``show_bottom_colorbar=False`` drops the second (bottom cp) colorbar —
    useful in narrow/composite panels.

    ``show_data_credit`` adds a licence-required data-source credit to the
    footnote of the figure the environment is drawn in (its own, or the one
    ``ax`` belongs to): ``True`` (default) uses ``env.data_sources`` (nothing
    shown if the env carries none); ``None`` / ``False`` hides it; or pass an
    ``Environment`` / ``Result`` / list of ``DataSource`` / strings.

    ``sea_ice`` overlays a (symbolic, not-to-scale) ice cover at the surface — a
    concentration 0–1 (uniform), or the ``AlongTrack``
    ``uacpy.data.fetch_sea_ice_concentration_transect`` returns (range-varying,
    ranges in metres).

    ``title`` overrides the default ``"<env.name> — seabed: <shape>"`` panel
    title (``"Environment — …"`` for an unnamed environment).

    ``x_max`` is a further range (m) the panel must reach — the span of a
    TL field it is drawn beside — for an environment that carries no range
    vector of its own.

    Parameters
    ----------
    env : Environment
        The environment to draw.
    source, receiver : Source / Receiver, optional
        Draw the run geometry.
    ax : matplotlib.axes.Axes, optional
        Existing axes (see above); a new figure is made when omitted.
    show_bottom_colorbar : bool, optional
        Draw the seabed ``cp`` colorbar. Default True.
    show_data_credit : bool, Environment, Result or sequence, optional
        The data-source credit of the figure's footnote (see above).
    sea_ice : float or AlongTrack, optional
        Ice cover drawn at the surface (see above).
    title : str, optional
        Axes title; ``None`` draws the default (see above).
    figsize : tuple, optional
        Size (inches) of the new figure. Default ``(10, 5)``.
    x_max : float, optional
        A range (m) the panel must reach (see above).
    source_marker_range : float, optional
        Range (m) the source star is drawn at; a drawing coordinate only.
        Default 0.
    """
    if not isinstance(env, Environment):
        raise ConfigurationError(
            f"plot_environment: expected an Environment, got "
            f"{type(env).__name__}.")
    fig, ax_bathy = fig_ax(ax, figsize)

    ssp = env.ssp

    # ── Bathymetry + bottom structure ────────────────────────────────
    bottom = env.bottom
    # Pull a sensible x-extent from any range-dependent axis available.
    # Falls back to (0, 1) only when nothing carries a range vector.
    candidate_rmaxes_km = []
    if env.bathymetry.varies_with_range:
        candidate_rmaxes_km.append(m_to_km(float(env.bathymetry.ranges[-1])))
    if bottom.is_range_dependent:
        candidate_rmaxes_km.append(
            m_to_km(float(np.max(np.asarray(bottom.ranges, dtype=float)))))
    if (receiver is not None and getattr(receiver, 'ranges', None) is not None
            and len(receiver.ranges) > 0):
        candidate_rmaxes_km.append(m_to_km(float(np.max(receiver.ranges))))
    if (env.ssp.is_range_dependent
            and env.ssp.ranges is not None and len(env.ssp.ranges) > 0):
        candidate_rmaxes_km.append(m_to_km(float(np.max(env.ssp.ranges))))
    if x_max is not None:
        candidate_rmaxes_km.append(m_to_km(float(x_max)))
    x_max_km = max(candidate_rmaxes_km) if candidate_rmaxes_km else 1.0

    if env.bathymetry.varies_with_range:
        r_km = m_to_km(env.bathymetry.ranges)
        seafloor = env.bathymetry.depths
        # Every model holds the last bathymetry value constant out to the
        # furthest receiver, so the profile gets an end anchor at the panel's
        # right edge (as ``_overlay_seafloor`` draws) and the seafloor line and
        # the bottom fills below span the whole xlim.
        if float(r_km[-1]) < x_max_km:
            r_km = np.append(r_km, x_max_km)
            seafloor = np.append(seafloor, seafloor[-1])
        # And the first value constant back to the source at r = 0, where the
        # panel starts whatever range the profile's first node sits at.
        if float(r_km[0]) > 0.0:
            r_km = np.insert(r_km, 0, 0.0)
            seafloor = np.insert(seafloor, 0, seafloor[0])
    else:
        r_km = np.array([0.0, x_max_km])
        seafloor = np.array([env.depth, env.depth])
    x_range = (float(r_km.min()), float(x_max_km))
    # The limits first, so the margins are measured on this panel: the water
    # mesh, seabed fills and seafloor line are painted a marker's width past
    # them (see _fill_margins) for the source star and the furthest receiver
    # dot to widen the axis into.
    ax_bathy.set_xlim(*x_range)
    m_lo, m_hi = _fill_margins(ax_bathy)
    r_km = np.concatenate(([r_km[0] - m_lo], r_km, [r_km[-1] + m_hi]))
    seafloor = np.concatenate(([seafloor[0]], seafloor, [seafloor[-1]]))

    z_max_layer = float(np.max(seafloor))
    seafloor_depth = z_max_layer  # remember the *actual* deepest seafloor
                                  # — branches mutate z_max_layer with a
                                  # hs_floor padding for the half-space
                                  # rendering, so the two are compared at
                                  # the ylim rather than one replacing the
                                  # other.

    # Independent cmaps + colorbars for water vs bottom. Each is
    # normalized to its own cs range so neither is washed out by the
    # other's extent. Convention: blue family for water, the earthy BOTTOM_CMAP for the
    # sediment / bottom.
    water_cmap = _truncated('Blues', 0.25, 0.95)
    bottom_cmap_truncated = _truncated(BOTTOM_CMAP, 0.25, 0.85)

    water_cs_pool = list(np.asarray(ssp.sound_speed, dtype=float).ravel())
    bottom_cs_pool: list = []
    for col in bottom.columns:
        bottom_cs_pool.extend(layer.sound_speed for layer in col.layers)
        if col.halfspace.acoustic_type not in ('vacuum', 'rigid', 'file'):
            bottom_cs_pool.append(col.halfspace.sound_speed)

    water_cs_min, water_cs_max, water_sm = _make_sm(water_cs_pool, water_cmap)
    # Uniform bottom-cp handling: every bottom that carries a cp value gets a
    # 'Bottom cp' colorbar, built the same way regardless of bottom type. A
    # single half-space is shaded on the fixed absolute sediment scale (see
    # _halfspace_fill_style), so its colorbar spans exactly that range; layered
    # / range-dependent bottoms span their own (relative) cp range.
    bottom_has_cp = bool(bottom_cs_pool)
    is_single_halfspace = not bottom.is_range_dependent and not bottom.is_layered
    if not bottom_has_cp:
        # Vacuum / rigid / file half-space, no layers: nothing down there has
        # a cp to normalize against. Neither reader runs — ``has_bottom_cbar``
        # below is False, and ``_layer_cmap_and_norm`` is called only by the
        # layered branches, which cannot be reached with an empty pool because
        # every layer carries a sound speed. A bottom map normalized to the
        # WATER speeds as a stand-in would be a mappable nothing draws.
        bot_cs_min = bot_cs_max = None
        bottom_sm = None
    elif is_single_halfspace:
        bot_cs_min, bot_cs_max = _HALFSPACE_CP_LO, _HALFSPACE_CP_HI
        bottom_sm = ScalarMappable(cmap=bottom_cmap_truncated,
                                   norm=Normalize(bot_cs_min, bot_cs_max))
        bottom_sm.set_array([])
    else:
        bot_cs_min, bot_cs_max, bottom_sm = _make_sm(
            bottom_cs_pool, bottom_cmap_truncated,
        )

    def _layer_cmap_and_norm():
        """Bottom-only normalization shared by the layered / range-dependent
        seabed branches: ``(BOTTOM_CMAP, cs_min, cs_max)``. Branches sample
        the raw map at ``0.25 + 0.6 * norm`` for the truncated band, so the
        colorbar's ``ScalarMappable`` is built on the truncated map to match."""
        return BOTTOM_CMAP, bot_cs_min, bot_cs_max

    # Water column on the bathy panel — water cmap (Blues), normalized
    # to its own cs range. The bottom rendering below covers anything
    # under the seafloor with opaque fills, so we don't need to mask
    # the SSP heatmap.
    if ssp.is_range_dependent:
        ssp_r_km_b = m_to_km(ssp.ranges)
        ssp_grid = np.asarray(ssp.sound_speed, dtype=float)
        # Models hold the end profiles constant past their nodes; anchoring
        # the mesh at the painted span's ends keeps the water colormap under
        # the whole panel rather than stopping at the outer SSP ranges.
        if float(ssp_r_km_b[0]) > float(r_km[0]):
            ssp_r_km_b = np.insert(ssp_r_km_b, 0, r_km[0])
            ssp_grid = np.column_stack([ssp_grid[:, 0], ssp_grid])
        if float(ssp_r_km_b[-1]) < float(r_km[-1]):
            ssp_r_km_b = np.append(ssp_r_km_b, r_km[-1])
            ssp_grid = np.column_stack([ssp_grid, ssp_grid[:, -1]])
        ax_bathy.pcolormesh(
            ssp_r_km_b, ssp.depths, ssp_grid,
            cmap=water_cmap,
            vmin=water_cs_min, vmax=water_cs_max,
            # Linear between the profile's samples: in depth, as every engine
            # interpolates a profile, and in range, as Bellhop's 'Q' option
            # does (Bellhop/sspMod.f90 Quad; the RAM family instead switches
            # to the nearest profile midway between nodes,
            # models/ram/collins.py). A model-agnostic rendering; 'nearest'
            # would paint a 3-sample profile as flat blocks stepping at depths
            # it never has.
            shading='gouraud', zorder=0,
        )
    else:
        ssp_1d = np.asarray(ssp.sound_speed, dtype=float).reshape(-1, 1)
        x_water = np.array([float(r_km.min()),
                            float(r_km.max() if r_km.size > 1 else x_max_km)])
        ax_bathy.pcolormesh(
            x_water, ssp.depths, np.tile(ssp_1d, (1, 2)),
            cmap=water_cmap,
            vmin=water_cs_min, vmax=water_cs_max,
            # Linear in depth between the samples, as in the range-dependent
            # branch above (one profile, so no range interpolation).
            shading='gouraud', zorder=0,
        )

    if bottom.is_range_dependent and bottom.is_layered:
        z_max_layer = _draw_rdl_bottom(
            ax_bathy, bottom, r_km, seafloor, z_max_layer, env,
            _layer_cmap_and_norm)
    elif bottom.is_range_dependent:
        z_max_layer = _draw_rd_bottom(
            ax_bathy, bottom, r_km, seafloor, z_max_layer, _layer_cmap_and_norm)
    elif bottom.is_layered:
        z_max_layer = _draw_layered_bottom(
            ax_bathy, bottom.columns[0], r_km, seafloor, z_max_layer,
            _layer_cmap_and_norm)
    else:  # single half-space
        z_max_layer = _draw_halfspace_bottom(
            ax_bathy, bottom.columns[0].halfspace, r_km, seafloor, z_max_layer)

    # Colorbars on the RIGHT (kept off the left so they never collide with the
    # depth axis in composite/`ax=` layouts). Uniform handling: ANY bottom that
    # carries a cp value (half-space, layered, range-dependent — all the same)
    # gets a 'Bottom cp' bar; only vacuum / rigid / file (no cp) show the water
    # bar alone.
    has_bottom_cbar = show_bottom_colorbar and bottom_has_cp
    if has_bottom_cbar:
        # Stack the two cp colorbars vertically in a single right-margin
        # column so they occupy the width of one bar (keeps narrow composite
        # panels uncrowded). Equal-size, near-full-height halves with a small
        # gap; compact labels + ≤3 ticks so they stay legible even in the
        # small env panel of plot_overview. Water on top, Bottom below.
        water_cax = ax_bathy.inset_axes([1.04, 0.54, 0.03, 0.45])
        bottom_cax = ax_bathy.inset_axes([1.04, 0.01, 0.03, 0.45])
        cbar_water = fig.colorbar(water_sm, cax=water_cax,
                                  label=WATER_C_LABEL)
        # A speed past the bar's scale (a single half-space's fixed scale)
        # takes the end colour, so the bar says it with an arrow.
        cbar_bottom = fig.colorbar(
            bottom_sm, cax=bottom_cax, label=BOTTOM_CP_LABEL,
            extend=_colorbar_extend(bottom_cs_pool, (bot_cs_min, bot_cs_max)))
        for cb in (cbar_water, cbar_bottom):
            cb.ax.tick_params(labelsize='x-small')
            # Relative like the ticks beside it: an absolute 7 pt here left the
            # label SMALLER than its own ticks under any profile that raises
            # font.size (the slide deck's is 20), which reads as a broken axis.
            cb.ax.yaxis.label.set_size('x-small')
            cb.ax.yaxis.set_major_locator(MaxNLocator(3))
            _speeds_read_absolute(cb)
    else:
        # Also an inset, so a composite (``ax=``) layout keeps the caller's
        # full axes width and stays aligned with its neighbours.
        _speeds_read_absolute(fig.colorbar(
            water_sm, cax=ax_bathy.inset_axes([1.04, 0.01, 0.03, 0.98]),
            label=WATER_C_LABEL))

    # Seafloor line on top of the bottom rendering.
    if env.bathymetry.varies_with_range:
        ax_bathy.plot(r_km, seafloor, **BOTTOM_LINE_STYLE, zorder=10)
    else:
        ax_bathy.axhline(env.depth, **BOTTOM_LINE_STYLE_FLAT, zorder=10)

    # Source / receiver markers on the bottom panel, drawn after the x limit
    # is set so the source star at the left limit can widen it. The source
    # sits at r = 0 by the package convention that range is measured from
    # it — but not every scene anchors that end: with a fixed receive array
    # the ARRAY is at the origin and the source is the thing at range.
    # ``source_marker_range`` moves the STAR, and nothing else: it is a
    # drawing coordinate, not a property of the Source and not an input to
    # any model. Named for the marker because "source range" would read as
    # a contradiction in a package where range is measured from the source.
    ax_bathy.set_xlim(*x_range)
    _draw_geometry(ax_bathy, source, receiver, max_markersize=5,
                   source_range=source_marker_range, env=env)
    # Tight ylim — surface to a small margin past the deepest seafloor, but
    # never above what the bottom branch actually painted. Every branch
    # returns ``z_max_layer``, the floor of its own rendering (layer stack +
    # hatched half-space extension); taking the max keeps a thick sediment
    # column on-panel instead of clipping its lower layers away, and leaves
    # the water-column margin in charge whenever the bottom is thin.
    ax_bathy.set_ylim(0, max(_panel_floor(seafloor_depth), z_max_layer))
    if not ax_bathy.get_xlabel():
        ax_bathy.set_xlabel('Range (km)')
    ax_bathy.set_ylabel('Depth (m)')
    invert_yaxis_once(ax_bathy)
    ax_bathy.grid(True, alpha=0.3)
    # Read the surface carriers directly: altimetry = surface shape (rough /
    # sloped surface), surface = top boundary properties (ice cover), both
    # range-dependent-aware. ``sea_ice`` stays an explicit concentration overlay.
    _draw_altimetry(ax_bathy, env)
    _draw_surface_boundary(ax_bathy, env)
    if sea_ice is not None:
        _draw_sea_ice(ax_bathy, sea_ice)
    # The panel is the whole cross-section, water column included, so the
    # default names the environment and then its seabed.
    name = env.name if env.name and env.name != 'unnamed' else 'Environment'
    ax_bathy.set_title(title if title is not None
                       else f"{name} — seabed: {_bottom_kind(bottom)}",
                       fontweight='bold', fontsize='large')

    _draw_credit(fig, _credit_attributions(show_data_credit, carrier=env),
                 reserve=ax is not None)
    if ax is None:
        # The credit is the figure's supxlabel, which tight_layout lays out
        # with the panels.
        fig.tight_layout()
    return fig, ax_bathy


def _ssp_to_draw(env_or_ssp, depths, ranges):
    """The :class:`~uacpy.core.ssp.SoundSpeedProfile` :func:`plot_ssp`
    draws: an environment's, the profile itself, or bare sound speeds on
    ``depths`` (and ``ranges``) with the shapes checked."""
    if isinstance(env_or_ssp, (Environment, SoundSpeedProfile)):
        if depths is not None or ranges is not None:
            raise ConfigurationError(
                "plot_ssp: a SoundSpeedProfile holds its own depths and "
                "ranges; depths= and ranges= are the axes of bare sound "
                "speeds.")
        return (env_or_ssp.ssp if isinstance(env_or_ssp, Environment)
                else env_or_ssp)
    if (isinstance(env_or_ssp, (str, bytes)) or np.ndim(env_or_ssp) == 0
            or not np.size(env_or_ssp)):
        raise ConfigurationError(
            "plot_ssp: pass an Environment or a SoundSpeedProfile, or sound "
            f"speeds on depths=; got {type(env_or_ssp).__name__}.")
    c = np.asarray(env_or_ssp, dtype=float)
    if depths is None:
        raise ConfigurationError(
            "plot_ssp: bare sound speeds need depths= (m), their depth axis.")
    z = np.atleast_1d(np.asarray(depths, dtype=float))
    want = (z.size,) if ranges is None else (z.size, np.size(ranges))
    if c.shape != want:
        raise ConfigurationError(
            f"plot_ssp: sound speeds of shape {c.shape} on {z.size} depths"
            + ("" if ranges is None else f" and {np.size(ranges)} ranges")
            + f" are {want}, depth first"
            + ("; a 2-D array needs ranges=." if ranges is None and c.ndim == 2
               else "."))
    return SoundSpeedProfile(
        depths=z, sound_speed=c.reshape(z.size, -1),
        ranges=None if ranges is None else np.asarray(ranges, dtype=float))


@typed_plot_error
def plot_ssp(env_or_ssp, ax=None, *, depths=None, ranges=None,
             title: Optional[str] = None,
             figsize=(5, 6), label: Optional[str] = None, color=None,
             show_legend: Optional[bool] = None, **line_kwargs):
    """Plot the sound-speed profile ``c(z)`` as a depth-down line.

    Accepts an :class:`~uacpy.core.environment.Environment`, a
    :class:`~uacpy.core.environment.SoundSpeedProfile`, or bare sound speeds
    ``c`` (m/s) on ``depths=``: shaped ``(n_depth,)``, or ``(n_depth,
    n_range)`` with ``ranges=``. A range-independent
    profile draws a single line; a range-dependent profile draws one line per
    range column, coloured by range with a colorbar. Depth increases downward.
    Pass ``ax=`` to draw into an existing axis; returns ``(fig, ax)``.

    **Overlaying several profiles on one axis.** Pass the same ``ax=`` to
    successive calls, each with its own ``label=`` and ``color=``, to compare
    profiles from different sites or different model runs::

        fig, ax = ligurian.ssp.plot(label='Ligurian', color='C0')
        norwegian.ssp.plot(ax=ax, label='Norwegian', color='C1')

    An explicit ``color`` selects overlay mode: every range column is drawn in
    that one colour and the range colorbar is suppressed, because a colorbar
    describes one profile's range axis and means nothing once a second profile
    shares the axis. Without ``color`` a range-dependent profile keeps its
    per-column viridis colouring and its colorbar, unchanged.

    ``label`` names the profile as a whole, so on a range-dependent profile it
    is attached to the first column only — one legend entry per profile rather
    than one per range column. ``show_legend`` forces the legend on or off;
    the default (``None``) draws one exactly when some artist on the axis
    carries a label. Remaining ``line_kwargs`` (``linestyle``, ``alpha``, ``zorder`` ...)
    are forwarded to ``ax.plot``.

    The profile's data sources (``ssp.data_sources``, a fetched profile's
    dataset) are credited in the footnote of the figure it is drawn in, as
    :func:`plot_environment` credits an environment's.

    Parameters
    ----------
    env_or_ssp : Environment, SoundSpeedProfile or array_like
        The profile, the environment holding it, or bare sound speeds (m/s).
    depths : array_like, optional
        The depth axis (m) of bare sound speeds; required for them, refused
        otherwise.
    ranges : array_like, optional
        The range axis (m) of 2-D bare sound speeds.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    title : str, optional
        Axes title; none is drawn when unset.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(5, 6)`` by default; unused when
        ``ax`` is given.
    label : str, optional
        Legend label of the profile (see above).
    color : str, optional
        One colour for every column; selects overlay mode (see above).
    show_legend : bool, optional
        Force the legend on or off; ``None`` draws one when an artist is
        labelled.
    **line_kwargs
        Matplotlib keywords for every line.
    """
    ssp = _ssp_to_draw(env_or_ssp, depths, ranges)

    fig, ax = fig_ax(ax, figsize)

    depths = np.asarray(ssp.depths, dtype=float)
    data = np.asarray(ssp.sound_speed, dtype=float)           # (n_depth, n_range)

    if ssp.is_range_dependent:
        line_kwargs.setdefault('linewidth', 1.2)
        ranges_km = m_to_km(np.asarray(ssp.ranges, dtype=float))
        if color is None:
            cmap = plt.get_cmap('viridis')
            norm = Normalize(vmin=float(ranges_km.min()),
                             vmax=float(ranges_km.max()) or 1.0)
            colours = [cmap(norm(r_km)) for r_km in ranges_km]
        else:
            # Overlay mode: one colour for the whole profile, so a second
            # profile on the same axis stays distinguishable from this one.
            colours = [color] * len(ranges_km)
        for j, _r_km in enumerate(ranges_km):
            # Label the first column only: the label names the profile, and
            # repeating it per column would fill the legend with duplicates.
            ax.plot(data[:, j], depths, color=colours[j],
                    label=label if j == 0 else None, **line_kwargs)
        if color is None:
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label='Range (km)')
    else:
        line_kwargs.setdefault('linewidth', 1.5)
        ax.plot(data[:, 0], depths, color='C0' if color is None else color,
                label=label, **line_kwargs)

    ax.set_xlabel('Sound speed (m/s)')
    # Speeds are 6-character labels ("1482.5"); five ticks is what the
    # default 5-inch panel fits without neighbouring labels touching.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_ylabel('Depth (m)')
    invert_yaxis_once(ax)                               # depth positive down
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    # ``get_legend_handles_labels`` already drops matplotlib's '_'-prefixed
    # placeholder labels, so this is empty exactly when nothing was named.
    if show_legend or (show_legend is None
                        and ax.get_legend_handles_labels()[1]):
        ax.legend()
    _draw_credit(fig, _credit_attributions(True, carrier=ssp))
    return fig, ax


# Seabed geoacoustic properties shown by ``plot_bottom_properties`` —
# (attribute, symbol, unit, colormap). cp / density are always present; the
# rest are skipped when uniformly zero (e.g. shear for a fluid seabed).
# Per-property colormaps, grouped by physical quantity so the panels read as
# a coherent set: cool perceptually-uniform 'viridis' for the wave speeds, a
# neutral 'bone_r' for density (heavier → darker), and a clean warm 'YlOrRd'
# for the attenuations (lossier → redder). Avoids the muddy cividis/YlOrBr mix.
_BOTTOM_PROPERTIES = (
    ('sound_speed',       'cp', 'm/s',   'viridis'),
    ('shear_speed',       'cs', 'm/s',   'viridis'),
    ('density',           'ρ',  'g/cm³', 'bone_r'),
    ('attenuation',       'αp', 'dB/λ',  'YlOrRd'),
    ('shear_attenuation', 'αs', 'dB/λ',  'YlOrRd'),
)


def _layered_property_at_depths(layered, prop, seafloor, z):
    """Step profile of ``prop`` for a ``SeabedColumn``: each layer from
    ``seafloor`` downward, then the half-space. ``z`` is absolute depth (m)."""
    out = np.full(z.shape, float(getattr(layered.halfspace, prop, 0.0) or 0.0))
    top = float(seafloor)
    for layer in layered.layers:
        bot = top + float(layer.thickness)
        out[(z >= top) & (z < bot)] = float(getattr(layer, prop, 0.0) or 0.0)
        top = bot
    return out


def _seabed_depth_samples(bottom, r_km, seafloor_r, n_depths):
    """Absolute depths (m) a seabed cross-section is sampled at.

    ``n_depths`` evenly spaced samples over the seabed — from ``pad`` above
    the shallowest seafloor (never above the surface) to ``pad`` below the
    deepest layer stack, ``pad`` being a quarter of that vertical extent and
    at least 20 m — plus, for every layer thinner than that spacing, three
    samples inside it (near its top, mid-layer, near its bottom) under each
    range sample, so a thin layer is drawn with its own thickness instead of
    falling between two evenly spaced samples."""
    sf_min, sf_max = float(np.min(seafloor_r)), float(np.max(seafloor_r))
    max_thk = bottom.total_thickness_max()
    pad = max(20.0, 0.25 * ((sf_max - sf_min) + max_thk))
    z_top = max(0.0, sf_min - pad)
    z = np.linspace(z_top, sf_max + max_thk + pad, n_depths)
    step = z[1] - z[0] if z.size > 1 else np.inf
    extra = []
    for j, r_node in enumerate(r_km):
        col = bottom.columns[bottom.column_index_at(
            range=float(km_to_m(r_node)))]
        top = float(seafloor_r[j])
        for layer in col.layers:
            thickness = float(layer.thickness)
            if 0.0 < thickness < step:
                extra.extend(top + thickness * np.array([0.01, 0.5, 0.99]))
            top += thickness
    return np.union1d(z, extra) if extra else z


def _seabed_property_grid(bottom, prop, r_km, z, seafloor_r):
    """``[len(z), len(r_km)]`` grid of ``prop`` in the seabed, NaN above the
    range-local seafloor — the water column is left blank because NaN cells
    render as the axes background."""
    grid = np.full((len(z), len(r_km)), np.nan)
    for j, r_node in enumerate(r_km):
        r_m = float(km_to_m(r_node))
        below = z >= seafloor_r[j]
        if not np.any(below):
            continue
        # Read the live column rather than ``at()``'s deep copy: this loop
        # only reads layer scalars, and copying a whole layer stack per
        # range node dominated the panel's cost. Pinned by the test asserting
        # ``Bottom.at`` is never called while this grid is built, and that
        # every cell still carries its layer's property.
        col = bottom.columns[bottom.column_index_at(range=r_m)]
        grid[below, j] = _layered_property_at_depths(
            col, prop, seafloor_r[j], z[below])
    return grid


@typed_plot_error
def plot_bottom_properties(env, *, properties=None, title: Optional[str] = None,
                           figsize=None, n_ranges=240, n_depths=200,
                           show_data_credit=True, fig=None):
    """Small-multiples cross-sections of the **seabed geoacoustic properties**.

    One range × depth heatmap per property present in
    ``env.bottom`` — sound speed ``cp``, shear speed ``cs``, density ``ρ``,
    compressional / shear attenuation ``αp`` / ``αs`` — each with its own
    colorbar. Properties that are uniformly zero or absent (e.g. shear for a
    fluid seabed) are skipped. Complements :func:`plot_environment`, which
    colours the seabed by ``cp`` alone; this is where ``cs`` and friends live.

    Works for every ``Bottom`` shape (half-space, layered, range-dependent
    half-space, range-dependent layered); for layered
    seabeds the layers track the bathymetry. The depth axis is absolute but
    spans the seabed only — from a margin above the shallowest seafloor to
    a margin below the deepest layer stack — and every layer thinner than
    the depth step gets samples of its own, so no layer falls between
    samples however deep the water. Pass ``properties=`` to restrict
    the panels, naming each by its attribute (``'sound_speed'``,
    ``'shear_speed'``, ``'density'``, ``'attenuation'``,
    ``'shear_attenuation'``) or its symbol (``'cp'``, ``'cs'``, ``'ρ'``,
    ``'αp'``, ``'αs'``); any other name is refused with the list. ``title=``
    overrides the figure title. ``fig=`` draws the panels into an existing
    ``Figure`` or ``SubFigure`` (a panel of a larger figure) whose size and
    layout are then the caller's; the data credit joins that figure's
    footnote. Returns ``(fig, axes)``.

    Parameters
    ----------
    env : Environment
        The environment whose seabed is drawn.
    properties : sequence of str, optional
        Panels to draw, by attribute or symbol (see above); ``None`` is every
        property present.
    title : str, optional
        Figure title. ``None`` draws the default caption ("Seabed properties
        — <seabed shape>"); ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure; ``None`` sizes it from the panel
        grid.
    n_ranges, n_depths : int, optional
        Samples of each heatmap along range and depth (plus the samples
        each thin layer adds, see above). Default 240 and 200.
    show_data_credit : bool, Environment, Result or sequence, optional
        The data-source credit of the figure's footnote, as on
        :func:`plot_environment`.
    fig : Figure or SubFigure, optional
        Draw the panels into this (see above).
    """
    # A caller holding the Bottom itself reaches for this first, so the type
    # is checked here and the refusal says why. The seabed has no ``.plot()``
    # of its own because its depth axis is measured DOWN FROM THE SEAFLOOR:
    # placing it needs the water depth, which lives on the Environment
    # (``env.depth`` and ``env.bathymetry.varies_with_range`` below).
    if not hasattr(env, 'bottom'):
        raise ConfigurationError(
            f"plot_bottom_properties: expected an Environment, got "
            f"{type(env).__name__}. The seabed is drawn from the environment "
            f"rather than from the seabed object because its depth axis runs "
            f"down from the seafloor, and only the environment knows where "
            f"that is.",
            remediation="plot_bottom_properties(env), with env.bottom set")
    bottom = env.bottom
    if bottom is None:
        raise ConfigurationError(
            "plot_bottom_properties: env.bottom is None. Attach a seabed to "
            "the environment (env.bottom = Bottom(...)) before plotting its "
            "properties.")
    if properties is not None:
        if isinstance(properties, str):
            properties = [properties]
        known = {name for prop, sym, _u, _c in _BOTTOM_PROPERTIES
                 for name in (prop, sym)}
        unknown = [p for p in properties if p not in known]
        if unknown:
            valid = ', '.join(f"{prop!r}/{sym!r}"
                              for prop, sym, _u, _c in _BOTTOM_PROPERTIES)
            raise ConfigurationError(
                f"plot_bottom_properties: properties= names "
                f"{', '.join(repr(u) for u in unknown)}, which is not a seabed "
                f"property this plot draws. Name each by attribute or "
                f"symbol: {valid}.")

    rmaxes_km = []
    if env.bathymetry.varies_with_range:
        rmaxes_km.append(m_to_km(float(env.bathymetry.ranges[-1])))
    if bottom.is_range_dependent:
        rmaxes_km.append(m_to_km(float(np.max(bottom.ranges))))
    x_max_km = max(rmaxes_km) if rmaxes_km else 1.0
    r_km = np.linspace(0.0, x_max_km, n_ranges)

    if env.bathymetry.varies_with_range:
        b = env.bathymetry
        seafloor_r = np.interp(km_to_m(r_km), b.ranges, b.depths)
    else:
        seafloor_r = np.full(r_km.shape, float(env.depth))

    z = _seabed_depth_samples(bottom, r_km, seafloor_r, n_depths)

    panels = []
    for prop, sym, unit, cmap in _BOTTOM_PROPERTIES:
        if (properties is not None and prop not in properties
                and sym not in properties):
            continue
        grid = _seabed_property_grid(bottom, prop, r_km, z, seafloor_r)
        finite = grid[np.isfinite(grid)]
        if finite.size == 0:
            continue
        if prop not in ('sound_speed', 'density') and np.allclose(finite, 0.0):
            continue
        panels.append((prop, sym, unit, cmap, grid))

    if not panels:
        considered = [prop for prop, sym, _u, _c in _BOTTOM_PROPERTIES
                      if properties is None
                      or prop in properties or sym in properties]
        raise ConfigurationError(
            f"plot_bottom_properties: no plottable seabed properties found. "
            f"None of {considered} carried a finite, non-zero profile on "
            f"this bottom. Set them on env.bottom, or pass properties= "
            f"naming ones it does carry.")

    n = len(panels)
    ncols = min(n, 3)
    nrows = int(np.ceil(n / ncols))
    # Shared range/depth axes (every panel is the same cross-section) so only
    # the edge axes carry labels — far less clutter than per-panel labels.
    fig, axes, _ = _grid_figure(
        fig, nrows, ncols, figsize, (4.4 * ncols, 3.1 * nrows + 0.4),
        who='plot_bottom_properties', sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for idx, (ax_p, (prop, sym, unit, cmap, grid)) in enumerate(
            zip(axes_flat, panels)):
        finite = grid[np.isfinite(grid)]
        vmin, vmax = float(np.min(finite)), float(np.max(finite))
        if vmin == vmax:
            # Relative, not ±0.5 absolute: on a 1650 m/s value an absolute
            # half-unit window makes the colorbar print offset ticks.
            pad = 0.05 * abs(vmin) if vmin else 0.5
            vmin, vmax = vmin - pad, vmax + pad
        pcm = ax_p.pcolormesh(r_km, z, grid, cmap=cmap,
                              vmin=vmin, vmax=vmax, shading='auto')
        ax_p.plot(r_km, seafloor_r, color='black', linewidth=1.2, zorder=5)
        ax_p.set_title(f"{sym}  ({unit})", fontweight='bold', fontsize='large')
        if idx % ncols == 0:                       # left column only
            ax_p.set_ylabel("Depth (m)")
        if idx + ncols >= n:                       # bottom-most visible per col
            ax_p.set_xlabel("Range (km)")
            ax_p.tick_params(labelbottom=True)
        cb = fig.colorbar(pcm, ax=ax_p, pad=0.015, fraction=0.05)
        cb.ax.tick_params(labelsize='small')

    invert_yaxis_once(axes_flat[0])                # shared → inverts all
    for ax_p in axes_flat[n:]:                     # hide unused cells
        ax_p.set_visible(False)

    fig.suptitle(title if title is not None
                 else f"Seabed properties — {_bottom_kind(bottom)}",
                 fontweight='bold', fontsize='large')
    _draw_credit(fig, _credit_attributions(show_data_credit, carrier=env))
    return fig, axes


def _absorption_to_draw(coefficient, frequencies, depths):
    """The :class:`~uacpy.core.absorption.AbsorptionCoefficient`
    :func:`plot_absorption` draws: the carrier itself, or a bare dB/km array
    on its ``frequencies`` (and ``depths``) with the shapes checked. A law
    is refused: the plotter draws data, never evaluates."""
    if isinstance(coefficient, Absorption):
        raise ConfigurationError(
            f"plot_absorption: {type(coefficient).__name__} is a law, and "
            f"a plotter draws data, it does not evaluate a law.",
            remediation="Evaluate it first: law.table(frequencies, "
                        "depths).plot(), or plot_absorption(law.table(f)).")
    if isinstance(coefficient, AbsorptionCoefficient):
        if frequencies is not None or depths is not None:
            raise ConfigurationError(
                "plot_absorption: an AbsorptionCoefficient holds its own "
                "frequencies and depths; frequencies= and depths= are the "
                "axes of a bare array.")
        return coefficient
    alpha = np.asarray(coefficient, dtype=float)
    if frequencies is None:
        raise ConfigurationError(
            "plot_absorption: a bare alpha array needs frequencies= (Hz), "
            "its frequency axis.")
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    if depths is None:
        if alpha.shape != f.shape:
            raise ConfigurationError(
                f"plot_absorption: alpha has shape {alpha.shape} for "
                f"{f.size} frequencies; a curve is (n_freq,), and a 2-D "
                f"alpha needs depths=.")
        return AbsorptionCoefficient(frequencies=f, data=alpha,
                                     units='dB/km')
    z = np.atleast_1d(np.asarray(depths, dtype=float))
    if alpha.shape != (z.size, f.size):
        raise ConfigurationError(
            f"plot_absorption: alpha has shape {alpha.shape}; on {z.size} "
            f"depths and {f.size} frequencies it is ({z.size}, {f.size}), "
            f"depth first.")
    return AbsorptionCoefficient(frequencies=f, data=alpha, units='dB/km',
                                 depths=z)


@typed_plot_error
def plot_absorption(coefficient, ax=None, *, frequencies=None, depths=None,
                    label=None, title=None, figsize=(7.5, 4.5),
                    show_colorbar=True, **mpl_kw):
    """Draw an :class:`~uacpy.core.absorption.AbsorptionCoefficient`, or a
    bare alpha array in dB/km on the ``frequencies`` (and ``depths``) given.

    Log-log against frequency for a 1-D carrier: absorption spans four
    decades across the band this package works in, so a linear axis shows one
    end of it or the other and never both. A carrier that carries a depth
    axis draws as a depth-frequency heatmap instead.

    This function draws data and computes nothing: a law (``Thorp()``,
    ``FrancoisGarrison(...)``, …) is refused — evaluate it first,
    ``law.table(f, depths).plot()`` or ``plot_absorption(law.table(f))``.
    Choosing a formula, and the ocean Francois-Garrison is evaluated in, is
    the caller's job; the carrier records that ocean in ``parameters``, and
    where a fetched ocean came from in ``data_sources``, which are credited
    in the footnote of the figure it is drawn in.

    Call repeatedly with ``ax=`` to overlay several models.

    Parameters
    ----------
    coefficient : AbsorptionCoefficient or array_like
        The carrier to draw, or alpha in dB/km shaped ``(n_freq,)``, or
        ``(n_depth, n_freq)`` with ``depths``.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    frequencies : array_like, optional
        The frequency axis (Hz) of a bare array; required for one, refused
        for a carrier, which holds its own.
    depths : array_like, optional
        The depth axis (m) of a 2-D bare array; refused for a carrier.
    label : str, optional
        Legend label of a 1-D curve; ``None`` is the carrier's model name.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(7.5, 4.5)`` by default; unused when
        ``ax`` is given.
    show_colorbar : bool, optional
        Draw the colorbar of a depth-frequency heatmap. Default True.
    **mpl_kw
        Matplotlib keywords for the curve or the heatmap.
    """
    coefficient = _absorption_to_draw(coefficient, frequencies, depths)
    freqs = np.asarray(coefficient.frequencies, dtype=float)
    values = np.asarray(coefficient.data, dtype=float)
    if not np.any(values > 0.0):
        # The depth is named in both shapes. A curve carries the single depth
        # it was evaluated at, so the advice below ("evaluate at a depth
        # inside one") can be acted on for a curve too, which is the case a
        # Biological user hits first.
        if coefficient.depths is not None:
            where = (f" over depths {np.min(coefficient.depths):g}"
                     f"..{np.max(coefficient.depths):g} m")
        elif coefficient.depth is not None:
            where = f" at {coefficient.depth:g} m"
        else:
            where = ''
        _plot_warn(
            f"plot_absorption: alpha is entirely non-positive{where}, so a "
            f"logarithmic axis draws blank. A layered model such as "
            f"Biological is zero outside its layers — evaluate at a depth "
            f"inside one.", FallbackWarning)
    value_label = f"Absorption ({coefficient.units})"
    fig, ax = fig_ax(ax, figsize)
    if coefficient.is_depth_dependent:
        depths = np.asarray(coefficient.depths, dtype=float)
        mesh = ax.pcolormesh(freqs, depths, values, shading='nearest',
                             **mpl_kw)
        ax.set_xscale('log')
        invert_yaxis_once(ax)
        ax.set_ylabel("Depth (m)")
        if show_colorbar:
            fig.colorbar(mesh, ax=ax, label=value_label)
    else:
        name = label or coefficient.model
        ax.loglog(freqs, values, label=name, **mpl_kw)
        ax.set_ylabel(value_label)
        ax.grid(which="both", alpha=0.3)
        if name or ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize='small')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(_title_or(title, "Volume absorption"))
    _draw_credit(fig, _credit_attributions(True, carrier=coefficient))
    return fig, ax


#: What a range-profile carrier declares for :func:`plot_range_profile`.
_RANGE_PROFILE_DECLARATIONS = ('_values', '_VALUE_LABEL', '_VALUE_UNIT',
                               '_AXIS_DOWN', 'ranges')


@typed_plot_error
def plot_range_profile(profile, ax=None, *, title=None, figsize=(10, 4),
                       **mpl_kw):
    """Render a 1-D ``value(range)`` carrier — :class:`Bathymetry` or
    :class:`Altimetry`; ``profile.plot()`` is the method form.

    The carrier declares what it holds: ``_values``, ``_VALUE_LABEL`` /
    ``_VALUE_UNIT`` for the axis label, and ``_AXIS_DOWN`` for the
    orientation. A new range-profile carrier is drawn correctly without
    touching this function. Its data sources (``profile.data_sources``, a
    fetched profile's dataset) are credited in the footnote of the figure it
    is drawn in.

    Parameters
    ----------
    profile : Bathymetry or Altimetry
        The carrier to draw.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    title : str, optional
        Axes title. ``None`` draws the default caption.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(10, 4)`` by default; unused when
        ``ax`` is given.
    **mpl_kw
        Matplotlib keywords for the line.
    """
    # Read by what the carrier declares, not by its class, so a new
    # range-profile carrier needs no change here.
    if not all(hasattr(profile, name) for name in _RANGE_PROFILE_DECLARATIONS):
        raise ConfigurationError(
            f"plot_range_profile: expected a range-profile carrier "
            f"(Bathymetry, Altimetry), got {type(profile).__name__}.")
    axis_down = profile._AXIS_DOWN
    values = np.asarray(profile._values, dtype=float)
    label = f"{profile._VALUE_LABEL.capitalize()} ({profile._VALUE_UNIT})"
    fig, ax = fig_ax(ax, figsize)
    r_km = m_to_km(np.asarray(profile.ranges, dtype=float))
    style = {'color': BOTTOM_HALFSPACE_COLOR if axis_down
             else SURFACE_LINE_COLOR,
             'linewidth': 1.6}
    style.update(mpl_kw)
    if r_km.size == 1:
        ax.axhline(float(values[0]), **style)
    else:
        ax.plot(r_km, values, **style)
    ax.set_xlabel('Range (km)')
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.3)
    if axis_down:
        invert_yaxis_once(ax)
    ax.set_title(title if title is not None
                 else f"{type(profile).__name__} profile",
                 fontweight='bold', fontsize='large')
    _draw_credit(fig, _credit_attributions(True, carrier=profile))
    return fig, ax
