"""Ocean-environment cross-section plot (SSP, bathymetry, bottom layers)."""

from __future__ import annotations


import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from uacpy.core.environment import Environment
from uacpy.visualization.style import BOTTOM_FILL_STYLE, BOTTOM_LINE_STYLE, BOTTOM_LINE_STYLE_FLAT, RECEIVER_MARKER_STYLE, SOURCE_MARKER_STYLE
from uacpy.visualization.plots._common import ZORDER_SEDIMENT, ZORDER_RECEIVERS, ZORDER_SOURCE, _credit_attributions, _draw_data_credit, _draw_sea_ice
from uacpy.core.environment import BoundaryProperties


def _draw_layered_bottom(ax_bathy, bottom, r_km, z_max_layer,
                        _layer_cmap_and_norm):
    # Per-layer fills (YlOrBr by sound speed) + dashed inter-layer
    # edges + hatched half-space + side legend card. Same visual
    # template as the RangeDependentLayeredBottom branch below.
    cmap, cs_min, cs_max, sm = _layer_cmap_and_norm()
    cs_range = max(1e-9, cs_max - cs_min)
    z_top = z_max_layer
    for layer in bottom.layers:
        z_bot = z_top + layer.thickness
        norm_cs = (layer.sound_speed - cs_min) / cs_range
        colour = cmap(0.25 + 0.6 * norm_cs)
        ax_bathy.fill_between(
            r_km, z_top, z_bot, color=colour, alpha=1.0,
            edgecolor='black', linewidth=0.4,
            zorder=ZORDER_SEDIMENT + 1,
        )
        ax_bathy.axhline(z_bot, color='black', linewidth=0.8,
                         linestyle='--', alpha=0.5,
                         zorder=ZORDER_SEDIMENT + 2)
        z_top = z_bot
    hs = bottom.halfspace
    hs_display = z_top + max(10.0, bottom.total_thickness() * 0.3)
    ax_bathy.fill_between(
        r_km, z_top, hs_display,
        zorder=ZORDER_SEDIMENT, **BOTTOM_FILL_STYLE,
    )

    legend_lines = ['Layered bottom']
    for i, layer in enumerate(bottom.layers):
        line = (f"L{i+1}: thk={layer.thickness:g} m  c={layer.sound_speed:g}  "
                f"ρ={layer.density:g}  α={layer.attenuation:g}")
        if layer.shear_speed > 0:
            line += f"  cs={layer.shear_speed:g}"
            if layer.shear_attenuation > 0:
                line += f"  αs={layer.shear_attenuation:g}"
        legend_lines.append(line)
    if hs.acoustic_type in ('vacuum', 'rigid'):
        hs_line = f"Half-space: {hs.acoustic_type}"
    else:
        hs_line = (f"Half-space ({hs.acoustic_type}): "
                   f"c={hs.sound_speed:g}  ρ={hs.density:g}  α={hs.attenuation:g}")
        if hs.shear_speed > 0:
            hs_line += f"  cs={hs.shear_speed:g}"
            if hs.shear_attenuation > 0:
                hs_line += f"  αs={hs.shear_attenuation:g}"
        if hs.roughness > 0:
            hs_line += f"  σ={hs.roughness:g}"
    legend_lines.append(hs_line)
    ax_bathy.text(
        0.98, 0.03, '\n'.join(legend_lines),
        transform=ax_bathy.transAxes, ha='right', va='bottom',
        fontsize=7, family='monospace',
        zorder=20,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  alpha=0.95),
    )
    z_max_layer = hs_display
    return z_max_layer


def _draw_rdl_bottom(ax_bathy, bottom, r_km, seafloor, z_max_layer,
                     env, _layer_cmap_and_norm):
    # Geological cross-section: one column per profile range, dashed
    # vertical boundaries between columns, P# labels above each
    # column, hatched half-space at the column bottom.
    prof_ranges = np.asarray(bottom.ranges, dtype=float)
    prof_ranges_km = prof_ranges / 1000.0
    boundaries = [prof_ranges_km[0]]
    for i in range(len(prof_ranges_km) - 1):
        boundaries.append(0.5 * (prof_ranges_km[i] + prof_ranges_km[i + 1]))
    boundaries.append(prof_ranges_km[-1])

    cmap, cs_min, cs_max, sm = _layer_cmap_and_norm()
    cs_range = max(1e-9, cs_max - cs_min)

    max_thickness = max(
        (sum(layer.thickness for layer in prof.layers)
         for prof in bottom.profiles), default=0.0,
    )
    hs_extension = max(z_max_layer * 0.25, 20.0)
    hs_floor = z_max_layer + max_thickness + hs_extension

    total_span = prof_ranges_km[-1] - prof_ranges_km[0]
    for i_r, (r_node, prof) in enumerate(zip(prof_ranges_km,
                                             bottom.profiles)):
        r_lo, r_hi = boundaries[i_r], boundaries[i_r + 1]
        n_pts = max(20, int(401 * (r_hi - r_lo) / max(total_span, 1e-9)))
        x_bin = np.linspace(r_lo, r_hi, n_pts)
        # Each layer follows the sloping seafloor across the column.
        z_top_arr = (np.interp(x_bin, r_km, seafloor)
                     if r_km.size > 1
                     else np.full_like(x_bin, env.depth))
        for layer in prof.layers:
            z_bot_arr = z_top_arr + layer.thickness
            norm_cs = (layer.sound_speed - cs_min) / cs_range
            colour = cmap(0.25 + 0.6 * norm_cs)
            ax_bathy.fill_between(
                x_bin, z_top_arr, z_bot_arr,
                color=colour, alpha=1.0,
                edgecolor='black', linewidth=0.3,
                zorder=ZORDER_SEDIMENT + 1,
            )
            z_top_arr = z_bot_arr
        # Hatched half-space below this column.
        ax_bathy.fill_between(
            x_bin, z_top_arr, np.full_like(x_bin, hs_floor),
            zorder=ZORDER_SEDIMENT, **BOTTOM_FILL_STYLE,
        )
        label_x = 0.5 * (r_lo + r_hi)
        ax_bathy.text(
            label_x, hs_floor * 0.02, f'P{i_r + 1}',
            ha='center', va='top', fontsize=9,
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

    legend_lines = ['Profiles']
    for i_p, prof in enumerate(bottom.profiles):
        for j, layer in enumerate(prof.layers):
            line = (f"P{i_p+1} L{j+1}: thk={layer.thickness:g} m  "
                    f"c={layer.sound_speed:g}  ρ={layer.density:g}  "
                    f"α={layer.attenuation:g}")
            if layer.shear_speed > 0:
                line += f"  cs={layer.shear_speed:g}"
                if layer.shear_attenuation > 0:
                    line += f"  αs={layer.shear_attenuation:g}"
            legend_lines.append(line)
        hs_p = prof.halfspace
        if hs_p.acoustic_type in ('vacuum', 'rigid'):
            hs_line = f"P{i_p+1} HS: {hs_p.acoustic_type}"
        else:
            hs_line = (f"P{i_p+1} HS ({hs_p.acoustic_type}): "
                       f"c={hs_p.sound_speed:g}  ρ={hs_p.density:g}  "
                       f"α={hs_p.attenuation:g}")
            if hs_p.shear_speed > 0:
                hs_line += f"  cs={hs_p.shear_speed:g}"
        legend_lines.append(hs_line)
    ax_bathy.text(
        0.98, 0.03, '\n'.join(legend_lines),
        transform=ax_bathy.transAxes, ha='right', va='bottom',
        fontsize=6, family='monospace',
        zorder=20,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  alpha=0.95),
    )
    z_max_layer = hs_floor
    return z_max_layer


def _draw_rd_bottom(ax_bathy, bottom, r_km, seafloor, z_max_layer):
    # Hatched half-space spans the full bathy extent. The cap is
    # piecewise-constant (one solid color per node) with Voronoi
    # boundaries: each node's color extends from the midpoint with
    # its left neighbour to the midpoint with its right neighbour;
    # outer nodes reach the bathymetry edges. Cap edges follow the
    # seafloor so kinks are honoured.
    bot_r_km = np.asarray(bottom.ranges, dtype=float) / 1000.0
    bathy_r = r_km
    bathy_z = seafloor
    sub_thickness = max(z_max_layer * 0.18, 5.0)
    cs = np.asarray(bottom.sound_speed, dtype=float)
    cs_min, cs_max = float(cs.min()), float(cs.max())
    cs_range = max(1e-9, cs_max - cs_min)
    cmap = plt.get_cmap('YlOrBr')
    hs_floor = z_max_layer * 1.3 + sub_thickness

    ax_bathy.fill_between(
        bathy_r, bathy_z, hs_floor,
        zorder=ZORDER_SEDIMENT, **BOTTOM_FILL_STYLE,
    )

    # Voronoi cell edges: midpoints between consecutive nodes,
    # clamped to the bathymetry extent at the outer ends.
    bathy_lo = float(bathy_r.min())
    bathy_hi = float(bathy_r.max())
    edges = [bathy_lo]
    for i in range(len(bot_r_km) - 1):
        edges.append(0.5 * (bot_r_km[i] + bot_r_km[i + 1]))
    edges.append(bathy_hi)

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
            [poly_z_top, (poly_z_top + sub_thickness)[::-1]]
        )
        colour = cmap(0.25 + 0.6 * (cs[i] - cs_min) / cs_range)
        ax_bathy.fill(
            poly_r, poly_z, color=colour, alpha=1.0,
            edgecolor='black', linewidth=0.3,
            zorder=ZORDER_SEDIMENT + 1,
        )

    layer_top = np.interp(bot_r_km, bathy_r, bathy_z)
    ax_bathy.plot(bot_r_km, layer_top, 'k.',
                  markersize=6, zorder=ZORDER_SEDIMENT + 5)
    for r_node in bot_r_km:
        ax_bathy.axvline(r_node, color='gray', linewidth=0.6,
                         linestyle='--', alpha=0.5,
                         zorder=ZORDER_SEDIMENT + 3)
    legend_lines = ['Bottom (per node)']
    ss = getattr(bottom, 'shear_speed', None)
    sa = getattr(bottom, 'shear_attenuation', None)
    ss_arr = np.asarray(ss) if ss is not None else np.zeros(len(bot_r_km))
    sa_arr = np.asarray(sa) if sa is not None else np.zeros(len(bot_r_km))
    for i in range(len(bot_r_km)):
        line = (f"P{i+1}: c={cs[i]:.0f}  ρ={bottom.density[i]:.2f}  "
                f"α={bottom.attenuation[i]:.2f}")
        if ss_arr[i] > 0:
            line += f"  cs={ss_arr[i]:.0f}"
            if sa_arr[i] > 0:
                line += f"  αs={sa_arr[i]:.2f}"
        legend_lines.append(line)
    ax_bathy.text(
        0.98, 0.03, '\n'.join(legend_lines),
        transform=ax_bathy.transAxes, ha='right', va='bottom',
        fontsize=7, family='monospace',
        zorder=20,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  alpha=0.6),
    )
    z_max_layer = hs_floor
    return z_max_layer


def _draw_halfspace_bottom(ax_bathy, bottom, r_km, seafloor, z_max_layer):
    zmax_plot = z_max_layer * 1.2
    # Single half-space — keep the canonical sandy-tan / hatched
    # signature; cs is reported in the property card so a colored
    # cmap fill would add no information.
    ax_bathy.fill_between(r_km, seafloor, zmax_plot,
                          zorder=ZORDER_SEDIMENT,
                          **BOTTOM_FILL_STYLE)
    if isinstance(bottom, BoundaryProperties):
        lines = [bottom.acoustic_type]
        if bottom.acoustic_type == 'file' and bottom.reflection_file:
            lines.append(f"file = {bottom.reflection_file}")
        elif bottom.acoustic_type == 'grain-size':
            lines.append(f"phi  = {bottom.grain_size_phi:g}")
            lines.append(f"ρ    = {bottom.density:.2f} g/cm³")
        elif bottom.acoustic_type not in ('vacuum', 'rigid'):
            lines.append(f"cp = {bottom.sound_speed:.0f} m/s")
            lines.append(f"ρ  = {bottom.density:.2f} g/cm³")
            lines.append(f"α  = {bottom.attenuation:.2f} dB/λ")
            if bottom.shear_speed > 0:
                lines.append(f"cs = {bottom.shear_speed:.0f} m/s")
                if bottom.shear_attenuation > 0:
                    lines.append(f"αs = {bottom.shear_attenuation:.2f} dB/λ")
            if bottom.roughness > 0:
                lines.append(f"σ  = {bottom.roughness:g} m")
        ax_bathy.text(
            0.98, 0.95, '\n'.join(lines),
            transform=ax_bathy.transAxes, ha='right', va='top',
            fontsize=9, family='monospace',
            zorder=20,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      alpha=0.95),
        )
    z_max_layer = zmax_plot
    return z_max_layer


def plot_environment(
    env: Environment,
    *,
    source=None,
    receiver=None,
    ax=None,
    bottom_colorbar: bool = True,
    data_source=True,
    sea_ice=None,
    figsize: Tuple[float, float] = (10, 5),
):
    """Single-panel water column + bottom structure with two colorbars.

    The water column is colour-mapped by SSP (Blues) and the bottom
    rendering depends on ``env.bottom``:

    * :class:`BoundaryProperties` — half-space fill with a property card.
    * :class:`LayeredBottom` — coloured per-layer fills (YlOrBr) +
      hatched half-space + side legend listing ``(thk, c, ρ, α[, cs])``.
    * :class:`RangeDependentBottom` — Voronoi-tiled solid-colour bands
      under the seafloor, one per range node.
    * :class:`RangeDependentLayeredBottom` — one column per profile,
      each column drawing the layer stack at that range; per-profile
      legend below the colorbar.

    Two colorbars: ``Water cp`` (Blues) and ``Bottom cp`` (YlOrBr) — each
    on its own dynamic range so neither is washed out by the other.

    Pass ``ax=`` to draw into an existing axis (for composite figures); returns
    ``(fig, ax)``. ``bottom_colorbar=False`` drops the second (bottom cp)
    colorbar — useful in narrow/composite panels where the per-node property
    card already lists the bottom values.

    ``data_source`` adds a licence-required data-source credit footnote (for a
    standalone figure, ``ax=None``): ``True`` (default) uses ``env.data_sources``
    (nothing shown if the env carries none); ``None`` / ``False`` hides it; or
    pass an ``Environment`` / ``Result`` / list of ``DataSource`` / strings.

    ``sea_ice`` overlays a (symbolic, not-to-scale) ice cover at the surface — a
    concentration 0–1 (uniform) or ``(ranges_km, concentration)`` (range-varying,
    e.g. from ``uacpy.data.fetch_sea_ice_concentration_transect``).
    """
    from uacpy.core.environment import (
        BoundaryProperties, LayeredBottom,
        RangeDependentBottom, RangeDependentLayeredBottom,
    )

    if ax is None:
        fig, ax_bathy = plt.subplots(1, 1, figsize=figsize)
    else:
        ax_bathy = ax
        fig = ax.figure

    ssp = env.ssp

    # ── Bathymetry + bottom structure ────────────────────────────────
    bottom = env.bottom
    # Pull a sensible x-extent from any range-dependent axis available.
    # Falls back to (0, 1) only when nothing carries a range vector.
    candidate_rmaxes = []
    if env.has_range_dependent_bathymetry():
        candidate_rmaxes.append(float(env.bathymetry[-1, 0]) / 1000.0)
    if isinstance(bottom, (RangeDependentBottom, RangeDependentLayeredBottom)):
        candidate_rmaxes.append(float(np.max(bottom.ranges)) / 1000.0)
    if (receiver is not None and getattr(receiver, 'ranges', None) is not None
            and len(receiver.ranges) > 0):
        candidate_rmaxes.append(float(np.max(receiver.ranges)) / 1000.0)
    if (env.ssp.is_range_dependent
            and env.ssp.ranges is not None and len(env.ssp.ranges) > 0):
        candidate_rmaxes.append(float(np.max(env.ssp.ranges)) / 1000.0)
    x_max = max(candidate_rmaxes) if candidate_rmaxes else 1.0

    if env.has_range_dependent_bathymetry():
        r_km = env.bathymetry[:, 0] / 1000.0
        seafloor = env.bathymetry[:, 1]
    else:
        r_km = np.array([0.0, x_max])
        seafloor = np.array([env.depth, env.depth])
    x_range = (float(r_km.min()), float(x_max))

    z_max_layer = float(np.max(seafloor))
    seafloor_depth = z_max_layer  # remember the *actual* deepest seafloor
                                  # — branches mutate z_max_layer with a
                                  # hs_floor padding for the half-space
                                  # rendering, but the final ylim should
                                  # not stretch the panel that far.

    # Independent cmaps + colorbars for water vs bottom. Each is
    # normalized to its own cs range so neither is washed out by the
    # other's extent. Convention: blue family for water, YlOrBr for the
    # sediment / bottom.
    def _truncated(name, lo, hi, n=256):
        from matplotlib.colors import LinearSegmentedColormap
        base = plt.get_cmap(name)
        return LinearSegmentedColormap.from_list(
            f"{name}_clip", base(np.linspace(lo, hi, n)),
        )

    def _make_sm(cs_values, cmap):
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        pool = list(cs_values) if len(cs_values) else [1500.0]
        cs_min = float(min(pool))
        cs_max = float(max(pool))
        sm = ScalarMappable(
            cmap=cmap,
            norm=Normalize(vmin=cs_min,
                           vmax=cs_max if cs_max > cs_min else cs_min + 1.0),
        )
        sm.set_array([])
        return cs_min, cs_max, sm

    water_cmap = _truncated('Blues', 0.25, 0.95)
    bottom_cmap_full = plt.get_cmap('YlOrBr')          # raw, used by 0.25+0.6*x trick
    bottom_cmap_truncated = _truncated('YlOrBr', 0.25, 0.85)

    water_cs_pool = list(np.asarray(ssp.data, dtype=float).ravel())
    bottom_cs_pool: list = []
    if isinstance(bottom, LayeredBottom):
        bottom_cs_pool.extend(layer.sound_speed for layer in bottom.layers)
        bottom_cs_pool.append(bottom.halfspace.sound_speed)
    elif isinstance(bottom, RangeDependentBottom):
        bottom_cs_pool.extend(np.asarray(bottom.sound_speed, dtype=float).ravel())
    elif isinstance(bottom, RangeDependentLayeredBottom):
        for prof in bottom.profiles:
            bottom_cs_pool.extend(layer.sound_speed for layer in prof.layers)
            bottom_cs_pool.append(prof.halfspace.sound_speed)
    elif isinstance(bottom, BoundaryProperties):
        if bottom.acoustic_type not in ('vacuum', 'rigid', 'file'):
            bottom_cs_pool.append(bottom.sound_speed)

    water_cs_min, water_cs_max, water_sm = _make_sm(water_cs_pool, water_cmap)
    bot_cs_min, bot_cs_max, bottom_sm = _make_sm(
        bottom_cs_pool or water_cs_pool, bottom_cmap_truncated,
    )

    def _layer_cmap_and_norm(cs_values=None):
        """Bottom-only normalization (legacy helper used by the LayeredBottom /
        RDLB / RangeDependentBottom branches). Returns ``(base_cmap, cs_min,
        cs_max, sm)`` where ``base_cmap`` is the raw YlOrBr — branches sample
        it at ``0.25 + 0.6 * norm`` for the truncated band, so the
        ``ScalarMappable`` has to match (truncated) for the colorbar to read."""
        if cs_values is None:
            return bottom_cmap_full, bot_cs_min, bot_cs_max, bottom_sm
        cs_min, cs_max, sm = _make_sm(cs_values, bottom_cmap_truncated)
        return bottom_cmap_full, cs_min, cs_max, sm

    # Water column on the bathy panel — water cmap (Blues), normalized
    # to its own cs range. The bottom rendering below covers anything
    # under the seafloor with opaque fills, so we don't need to mask
    # the SSP heatmap.
    if ssp.is_range_dependent:
        ssp_r_km_b = ssp.ranges / 1000.0
        ax_bathy.pcolormesh(
            ssp_r_km_b, ssp.depths, ssp.data,
            cmap=water_cmap,
            vmin=water_cs_min, vmax=water_cs_max,
            shading='nearest', zorder=0,
        )
    else:
        ssp_1d = np.asarray(ssp.data, dtype=float).reshape(-1, 1)
        x_water = np.array([float(r_km.min()),
                            float(r_km.max() if r_km.size > 1 else x_max)])
        ax_bathy.pcolormesh(
            x_water, ssp.depths, np.tile(ssp_1d, (1, 2)),
            cmap=water_cmap,
            vmin=water_cs_min, vmax=water_cs_max,
            shading='nearest', zorder=0,
        )

    if isinstance(bottom, LayeredBottom):
        z_max_layer = _draw_layered_bottom(
            ax_bathy, bottom, r_km, z_max_layer, _layer_cmap_and_norm)
    elif isinstance(bottom, RangeDependentLayeredBottom):
        z_max_layer = _draw_rdl_bottom(
            ax_bathy, bottom, r_km, seafloor, z_max_layer, env,
            _layer_cmap_and_norm)
    elif isinstance(bottom, RangeDependentBottom):
        z_max_layer = _draw_rd_bottom(
            ax_bathy, bottom, r_km, seafloor, z_max_layer)
    else:  # BoundaryProperties or other half-space
        z_max_layer = _draw_halfspace_bottom(
            ax_bathy, bottom, r_km, seafloor, z_max_layer)

    # Two colorbars stacked on the RIGHT (water cp inner, bottom cp outer),
    # each on its own dynamic range — kept off the left so they never collide
    # with the depth axis (important in composite/`ax=` layouts). The bottom
    # colorbar is suppressed for a single half-space (the property card states cs).
    fig.colorbar(water_sm, ax=ax_bathy, label='Water cp (m/s)',
                 location='right', fraction=0.046, pad=0.02)
    # Only add the bottom colorbar when the bottom cp actually varies — a single
    # half-space or a uniform range-dependent bottom carries no gradient (the
    # property card already states its values), and a second bar just crowds.
    if (bottom_colorbar and not isinstance(bottom, BoundaryProperties)
            and bot_cs_max - bot_cs_min > 1e-6):
        fig.colorbar(bottom_sm, ax=ax_bathy, label='Bottom cp (m/s)',
                     location='right', fraction=0.046, pad=0.12)

    # Seafloor line on top of the bottom rendering.
    if env.has_range_dependent_bathymetry():
        ax_bathy.plot(r_km, seafloor, **BOTTOM_LINE_STYLE, zorder=10)
    else:
        ax_bathy.axhline(env.depth, **BOTTOM_LINE_STYLE_FLAT, zorder=10)

    # Source / receiver markers on the bottom panel.
    if source is not None and getattr(source, 'depths', None) is not None:
        for sd in np.atleast_1d(source.depths):
            ax_bathy.plot([x_range[0]], [float(sd)],
                          zorder=ZORDER_SOURCE,
                          **SOURCE_MARKER_STYLE)
    if receiver is not None and getattr(receiver, 'depths', None) is not None:
        rr_full = np.atleast_1d(receiver.ranges) / 1000.0
        rd_full = np.atleast_1d(receiver.depths)
        # Dense grids form solid bars — decimate each axis independently
        # so the spatial structure stays readable. Range typically spans
        # 10× more samples than depth in surveys, so we cap the two axes
        # differently (20 across, 10 down).
        max_range_dots = 20
        max_depth_dots = 10
        step_r = max(1, rr_full.size // max_range_dots)
        step_d = max(1, rd_full.size // max_depth_dots)
        rr = rr_full[::step_r]
        rd = rd_full[::step_d]
        RR, RD = np.meshgrid(rr, rd)
        rcv_style = dict(RECEIVER_MARKER_STYLE)
        rcv_style['markersize'] = min(
            rcv_style.get('markersize', 8), 5,
        )
        ax_bathy.plot(RR.ravel(), RD.ravel(),
                      zorder=ZORDER_RECEIVERS, **rcv_style)

    ax_bathy.set_xlim(*x_range)
    # Tight ylim — surface to a small margin past the deepest seafloor.
    # Bottom rendering may extend below the seafloor visually (hatched
    # half-space / PML-like padding) but the displayed extent stays
    # close to the physical water column.
    ax_bathy.set_ylim(0, seafloor_depth * 1.20)
    if not ax_bathy.get_xlabel():
        ax_bathy.set_xlabel('Range (km)')
    ax_bathy.set_ylabel('Depth (m)')
    ax_bathy.invert_yaxis()
    ax_bathy.grid(True, alpha=0.3)
    if sea_ice is not None:
        _draw_sea_ice(ax_bathy, sea_ice)
    ax_bathy.set_title(f"Bottom — {type(env.bottom).__name__}",
                       fontweight='bold', fontsize=12)

    if ax is None:
        credit = _credit_attributions(data_source, carrier=env)
        fig.tight_layout(rect=(0, 0.05, 1, 1) if credit else (0, 0, 1, 1))
        _draw_data_credit(fig, credit, reserve=False)
    return fig, ax_bathy


# Professional oceanographic depth ramp (shallow → deep): pale aqua through
# teal and ocean blue to deep navy. A dependency-free stand-in for cmocean
# 'deep', and the default for plot_bathymetry_map.
