"""Ocean-environment cross-section plot (SSP, bathymetry, bottom layers)."""

from __future__ import annotations


import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from uacpy.core.environment import Environment
from uacpy.visualization.style import (
    BOTTOM_FILL_STYLE, BOTTOM_FILL_HATCH, BOTTOM_CMAP, BOTTOM_LINE_STYLE,
    BOTTOM_LINE_STYLE_FLAT, _blend,
)
from uacpy.visualization.plots._common import ZORDER_SEDIMENT, _credit_attributions, _draw_credit, _draw_geometry, _draw_sea_ice, _draw_surface_boundary, _draw_altimetry, fig_ax, typed_plot_error, invert_yaxis_once
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.units import km_to_m, m_to_km


# A single half-space is shaded by sound speed on a fixed (absolute) scale so
# different half-spaces read differently, while the '///' hatch still flags the
# region as the semi-infinite half-space (vs the solid-filled finite layers).
# The scale spans realistic *surficial* seabeds (clay ≈1450 → coarse/cemented
# ≈2300 m/s) so ordinary sediments spread across the ramp instead of clustering
# at the soft end; harder bottoms (rock basements) simply saturate at the dark
# end. The right-margin 'Bottom cp' colorbar spans exactly this scale, so the
# fill reads back to a cp value.
_HALFSPACE_CP_LO, _HALFSPACE_CP_HI = 1450.0, 2300.0


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


def _hatched_fill(facecolor) -> dict:
    """``fill_between`` kwargs: a solid ``facecolor`` under the half-space's
    diagonal hatch, so it still reads as the semi-infinite half-space."""
    # facecolor (NOT color): with ``color=`` set, mpl draws the hatch in the
    # fill colour, making the '///' invisible. ``facecolor`` lets the hatch
    # render in ``edgecolor``.
    return {'facecolor': facecolor, 'hatch': BOTTOM_FILL_HATCH,
            'edgecolor': _blend('black', facecolor, 0.4), 'linewidth': 0.4}


def _speeds_read_absolute(colorbar):
    """Print a sound-speed colorbar's ticks as absolute m/s: a nearly constant
    profile otherwise draws 0.0/0.4/0.8 over a "+1.5e3" offset."""
    colorbar.ax.yaxis.get_major_formatter().set_useOffset(False)


def _halfspace_fill_style(cp) -> dict:
    """Absolute-scale half-space fill — used for a *single* half-space (no
    per-plot bottom colorbar): flat tan when ``cp`` is ``None`` (vacuum / rigid
    / file), else an earthy BOTTOM_CMAP shade mapped from ``cp`` (m/s) on a fixed scale, so
    different half-spaces read differently across plots. Keeps the hatch."""
    if cp is None:
        return dict(BOTTOM_FILL_STYLE)
    t = float(np.clip((cp - _HALFSPACE_CP_LO)
                      / (_HALFSPACE_CP_HI - _HALFSPACE_CP_LO), 0.0, 1.0))
    return _hatched_fill(BOTTOM_CMAP(0.25 + 0.6 * t))


def _layered_halfspace_style(hs, cmap, cs_min, cs_range) -> dict:
    """Basement-half-space fill for a *layered* bottom: shaded on the SAME
    (relative) ``cmap``/range as the sediment layers so it agrees with the
    plot's 'Bottom cp' colorbar; flat tan for vacuum / rigid / file."""
    cp = _halfspace_cp(hs)
    if cp is None:
        return dict(BOTTOM_FILL_STYLE)
    norm = min(1.0, max(0.0, (cp - cs_min) / cs_range))
    return _hatched_fill(cmap(0.25 + 0.6 * norm))


def _draw_layered_bottom(ax_bathy, column, r_km, seafloor, z_max_layer,
                        _layer_cmap_and_norm):
    # Per-layer fills (earthy BOTTOM_CMAP by sound speed) + dashed inter-layer
    # edges + hatched half-space, keyed by the caller's right-margin
    # 'Bottom cp' colorbar. Same visual template as the range-dependent
    # layered branch below.
    cmap, cs_min, cs_max, _ = _layer_cmap_and_norm()
    cs_range = max(1e-9, cs_max - cs_min)
    # Every layer rides the seafloor, as in the range-dependent layered branch.
    # A scalar top anchored at the deepest bathymetry point would detach the
    # stack from a sloping seabed and leave the water colormap painted in the
    # gap below the drawn seafloor line.
    z_top = np.broadcast_to(np.asarray(seafloor, dtype=float),
                            np.shape(r_km)).astype(float)
    for layer in column.layers:
        z_bot = z_top + layer.thickness
        norm_cs = (layer.sound_speed - cs_min) / cs_range
        colour = cmap(0.25 + 0.6 * norm_cs)
        ax_bathy.fill_between(
            r_km, z_top, z_bot, color=colour, alpha=1.0,
            edgecolor='black', linewidth=0.4,
            zorder=ZORDER_SEDIMENT + 1,
        )
        ax_bathy.plot(r_km, z_bot, color='black', linewidth=0.8,
                      linestyle='--', alpha=0.5,
                      zorder=ZORDER_SEDIMENT + 2)
        z_top = z_bot
    hs = column.halfspace
    hs_display = z_top + max(10.0, column.total_thickness() * 0.3)
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
    boundaries = [min(r_lo_panel, float(prof_ranges_km[0]))]
    for i in range(len(prof_ranges_km) - 1):
        boundaries.append(0.5 * (prof_ranges_km[i] + prof_ranges_km[i + 1]))
    boundaries.append(max(r_hi_panel, float(prof_ranges_km[-1])))

    cmap, cs_min, cs_max, _ = _layer_cmap_and_norm()
    cs_range = max(1e-9, cs_max - cs_min)

    max_thickness = max(
        (sum(layer.thickness for layer in prof.layers)
         for prof in bottom.columns), default=0.0,
    )
    hs_extension = max(z_max_layer * 0.25, 20.0)
    hs_floor = z_max_layer + max_thickness + hs_extension

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
        # Hatched half-space below this column, shaded by its basement cp.
        ax_bathy.fill_between(
            x_bin, z_top_arr, np.full_like(x_bin, hs_floor),
            zorder=ZORDER_SEDIMENT,
            **_layered_halfspace_style(prof.halfspace, cmap, cs_min, cs_range),
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

    z_max_layer = hs_floor
    return z_max_layer


def _draw_rd_bottom(ax_bathy, bottom, r_km, seafloor, z_max_layer):
    # A range-dependent half-space: properties vary with range, uniform with
    # depth. So each node colours its whole column (seafloor → floor) by its
    # sound speed, hatched like any half-space, with Voronoi boundaries: each
    # node owns from the midpoint with its left neighbour to the midpoint with
    # its right; outer nodes reach the bathymetry edges. Tops follow the
    # seafloor so kinks are honoured.
    bot_r_km = m_to_km(np.asarray(bottom.ranges, dtype=float))
    bathy_r = r_km
    bathy_z = seafloor
    cs = np.asarray(bottom.halfspace_sound_speed, dtype=float)
    cs_min, cs_max = float(cs.min()), float(cs.max())
    cs_range = max(1e-9, cs_max - cs_min)
    cmap = BOTTOM_CMAP
    hs_floor = z_max_layer * 1.3 + max(z_max_layer * 0.18, 5.0)

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
            [poly_z_top, np.full(poly_r_top.shape, hs_floor)]
        )
        colour = cmap(0.25 + 0.6 * (cs[i] - cs_min) / cs_range)
        ax_bathy.fill(
            poly_r, poly_z, facecolor=colour, hatch=BOTTOM_FILL_HATCH,
            edgecolor=_blend('black', colour, 0.4), linewidth=0.3,
            zorder=ZORDER_SEDIMENT + 1,
        )

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
    zmax_plot = z_max_layer * 1.2
    # Single half-space shaded by its sound speed (the '///' hatch keeps the
    # half-space signature); flat tan for vacuum / rigid / file (no cp value).
    cp = _halfspace_cp(halfspace)
    ax_bathy.fill_between(r_km, seafloor, zmax_plot,
                          zorder=ZORDER_SEDIMENT,
                          **_halfspace_fill_style(cp))
    z_max_layer = zmax_plot
    return z_max_layer


@typed_plot_error
def _plot_environment(
    env: Environment,
    *,
    source=None,
    receiver=None,
    ax=None,
    bottom_colorbar: bool = True,
    data_source=True,
    sea_ice=None,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
):
    """Single-panel water column + bottom structure with two colorbars.

    The water column is colour-mapped by SSP (Blues) and the bottom
    rendering depends on ``env.bottom``:

    * half-space column — half-space fill.
    * layered column — coloured per-layer fills (earthy ground
      colormap) + hatched half-space.
    * range-dependent half-space — Voronoi-tiled solid-colour bands
      under the seafloor, one per range node.
    * range-dependent layered — one column per profile, each column
      drawing the layer stack at that range.

    Everything is coloured by ``sound_speed`` (water ``c``, seabed ``cp``); for
    the seabed's other geoacoustic properties (cs, ρ, αp, αs) use
    :func:`plot_bottom_properties`.

    Two colorbars: ``Water c`` (Blues) and ``Bottom cp`` (earthy brown) — each
    on its own dynamic range so neither is washed out by the other.

    Pass ``ax=`` to draw into an existing axis (for composite figures); returns
    ``(fig, ax)``. ``bottom_colorbar=False`` drops the second (bottom cp)
    colorbar — useful in narrow/composite panels.

    ``data_source`` adds a licence-required data-source credit footnote (for a
    standalone figure, ``ax=None``): ``True`` (default) uses ``env.data_sources``
    (nothing shown if the env carries none); ``None`` / ``False`` hides it; or
    pass an ``Environment`` / ``Result`` / list of ``DataSource`` / strings.

    ``sea_ice`` overlays a (symbolic, not-to-scale) ice cover at the surface — a
    concentration 0–1 (uniform) or ``(ranges_km, concentration)`` (range-varying,
    e.g. from ``uacpy.data.fetch_sea_ice_concentration_transect``).

    ``title`` overrides the default ``"Bottom — <shape>"`` panel title.
    """
    if not isinstance(env, Environment):
        raise ConfigurationError(
            f"_plot_environment: expected an Environment, got "
            f"{type(env).__name__}.")
    fig, ax_bathy = fig_ax(ax, figsize)

    ssp = env.ssp

    # ── Bathymetry + bottom structure ────────────────────────────────
    bottom = env.bottom
    # Pull a sensible x-extent from any range-dependent axis available.
    # Falls back to (0, 1) only when nothing carries a range vector.
    candidate_rmaxes_km = []
    if env.has_range_dependent_bathymetry:
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
    x_max_km = max(candidate_rmaxes_km) if candidate_rmaxes_km else 1.0

    if env.has_range_dependent_bathymetry:
        r_km = m_to_km(env.bathymetry.ranges)
        seafloor = env.bathymetry.depths
        # Every model holds the last bathymetry value constant out to the
        # furthest receiver, so the profile gets an end anchor at the panel's
        # right edge (as ``_overlay_seafloor`` draws) and the seafloor line and
        # the bottom fills below span the whole xlim.
        if float(r_km[-1]) < x_max_km:
            r_km = np.append(r_km, x_max_km)
            seafloor = np.append(seafloor, seafloor[-1])
    else:
        r_km = np.array([0.0, x_max_km])
        seafloor = np.array([env.depth, env.depth])
    x_range = (float(r_km.min()), float(x_max_km))

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
    def _truncated(cmap, lo, hi, n=256):
        from matplotlib.colors import LinearSegmentedColormap
        base = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
        return LinearSegmentedColormap.from_list(
            f"{getattr(base, 'name', 'cmap')}_clip", base(np.linspace(lo, hi, n)),
        )

    def _make_sm(cs_values, cmap):
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        pool = list(cs_values) if len(cs_values) else [1500.0]
        cs_min = float(min(pool))
        cs_max = float(max(pool))
        if cs_max <= cs_min:
            # A constant profile gets a window centred on its value.
            # Relative, not +1 m/s absolute: ticks over 1500–1501 need six
            # significant digits, which the colorbar printed as 0.0/0.4/0.8
            # over a "+1.5e3" offset.
            pad = 0.05 * abs(cs_min) if cs_min else 0.5
            cs_min, cs_max = cs_min - pad, cs_max + pad
        sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=cs_min, vmax=cs_max))
        sm.set_array([])
        return cs_min, cs_max, sm

    water_cmap = _truncated('Blues', 0.25, 0.95)
    bottom_cmap_full = BOTTOM_CMAP                      # raw, used by 0.25+0.6*x trick
    bottom_cmap_truncated = _truncated(BOTTOM_CMAP, 0.25, 0.85)

    water_cs_pool = list(np.asarray(ssp.data, dtype=float).ravel())
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
        # every layer carries a sound speed. Normalizing the bottom map to the
        # WATER speeds as a stand-in built a mappable nothing ever drew.
        bot_cs_min = bot_cs_max = None
        bottom_sm = None
    elif is_single_halfspace:
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
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
        seabed branches. Returns ``(base_cmap, cs_min,
        cs_max, sm)`` where ``base_cmap`` is the raw BOTTOM_CMAP — branches sample
        it at ``0.25 + 0.6 * norm`` for the truncated band, so the
        ``ScalarMappable`` has to match (truncated) for the colorbar to read."""
        return bottom_cmap_full, bot_cs_min, bot_cs_max, bottom_sm

    # Water column on the bathy panel — water cmap (Blues), normalized
    # to its own cs range. The bottom rendering below covers anything
    # under the seafloor with opaque fills, so we don't need to mask
    # the SSP heatmap.
    if ssp.is_range_dependent:
        ssp_r_km_b = m_to_km(ssp.ranges)
        ssp_grid = np.asarray(ssp.data, dtype=float)
        # Models hold the last profile constant past its node; anchoring the
        # mesh at the right edge keeps the water colormap under the whole
        # xlim rather than stopping at the last SSP range.
        if float(ssp_r_km_b[-1]) < x_max_km:
            ssp_r_km_b = np.append(ssp_r_km_b, x_max_km)
            ssp_grid = np.column_stack([ssp_grid, ssp_grid[:, -1]])
        ax_bathy.pcolormesh(
            ssp_r_km_b, ssp.depths, ssp_grid,
            cmap=water_cmap,
            vmin=water_cs_min, vmax=water_cs_max,
            # Linear between the profile's samples, in depth and range —
            # what every model does with the SSP. 'nearest' painted a
            # 3-sample profile as flat blocks stepping at depths it never has.
            shading='gouraud', zorder=0,
        )
    else:
        ssp_1d = np.asarray(ssp.data, dtype=float).reshape(-1, 1)
        x_water = np.array([float(r_km.min()),
                            float(r_km.max() if r_km.size > 1 else x_max_km)])
        ax_bathy.pcolormesh(
            x_water, ssp.depths, np.tile(ssp_1d, (1, 2)),
            cmap=water_cmap,
            vmin=water_cs_min, vmax=water_cs_max,
            # Linear between the profile's samples, in depth and range —
            # what every model does with the SSP. 'nearest' painted a
            # 3-sample profile as flat blocks stepping at depths it never has.
            shading='gouraud', zorder=0,
        )

    if bottom.is_range_dependent and bottom.is_layered:
        z_max_layer = _draw_rdl_bottom(
            ax_bathy, bottom, r_km, seafloor, z_max_layer, env,
            _layer_cmap_and_norm)
    elif bottom.is_range_dependent:
        z_max_layer = _draw_rd_bottom(
            ax_bathy, bottom, r_km, seafloor, z_max_layer)
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
    has_bottom_cbar = bottom_colorbar and bottom_has_cp
    if has_bottom_cbar:
        # Stack the two cp colorbars vertically in a single right-margin
        # column so they occupy the width of one bar (keeps narrow composite
        # panels uncrowded). Equal-size, near-full-height halves with a small
        # gap; compact labels + ≤3 ticks so they stay legible even in the
        # small env panel of plot_overview. Water on top, Bottom below.
        from matplotlib.ticker import MaxNLocator
        water_cax = ax_bathy.inset_axes([1.04, 0.54, 0.03, 0.45])
        bottom_cax = ax_bathy.inset_axes([1.04, 0.01, 0.03, 0.45])
        cbar_water = fig.colorbar(water_sm, cax=water_cax,
                                  label='Water c (m/s)')
        cbar_bottom = fig.colorbar(bottom_sm, cax=bottom_cax,
                                   label='Bottom cp (m/s)')
        for cb in (cbar_water, cbar_bottom):
            cb.ax.tick_params(labelsize=6)
            cb.ax.yaxis.label.set_size(7)
            cb.ax.yaxis.set_major_locator(MaxNLocator(3))
            _speeds_read_absolute(cb)
    else:
        # Also an inset, so a composite (``ax=``) layout keeps the caller's
        # full axes width and stays aligned with its neighbours.
        _speeds_read_absolute(fig.colorbar(
            water_sm, cax=ax_bathy.inset_axes([1.04, 0.01, 0.03, 0.98]),
            label='Water c (m/s)'))

    # Seafloor line on top of the bottom rendering.
    if env.has_range_dependent_bathymetry:
        ax_bathy.plot(r_km, seafloor, **BOTTOM_LINE_STYLE, zorder=10)
    else:
        ax_bathy.axhline(env.depth, **BOTTOM_LINE_STYLE_FLAT, zorder=10)

    # Source / receiver markers on the bottom panel.
    _draw_geometry(ax_bathy, source, receiver, max_markersize=5,
                   source_range_m=km_to_m(x_range[0]))

    ax_bathy.set_xlim(*x_range)
    # Tight ylim — surface to a small margin past the deepest seafloor, but
    # never above what the bottom branch actually painted. Every branch
    # returns ``z_max_layer``, the floor of its own rendering (layer stack +
    # hatched half-space extension); taking the max keeps a thick sediment
    # column on-panel instead of clipping its lower layers away, and leaves
    # the 20 % water-column margin in charge whenever the bottom is thin.
    ax_bathy.set_ylim(0, max(seafloor_depth * 1.20, z_max_layer))
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
    ax_bathy.set_title(title if title is not None
                       else f"Bottom — {_bottom_kind(bottom)}",
                       fontweight='bold', fontsize=12)

    if ax is None:
        credit = _credit_attributions(data_source, carrier=env)
        fig.tight_layout(rect=(0, 0.05, 1, 1) if credit else (0, 0, 1, 1))
        _draw_credit(fig, credit, reserve=False)
    return fig, ax_bathy


@typed_plot_error
def _plot_ssp(env_or_ssp, *, ax=None, title: Optional[str] = None,
              figsize=(5, 6), label: Optional[str] = None, color=None,
              legend: Optional[bool] = None, **line_kwargs):
    """Plot the sound-speed profile ``c(z)`` as a depth-down line.

    Accepts an :class:`~uacpy.core.environment.Environment` or a bare
    :class:`~uacpy.core.environment.SoundSpeedProfile`. A range-independent
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
    than one per range column. ``legend`` forces the legend on or off; the
    default (``None``) draws one exactly when some artist on the axis carries a
    label. Remaining ``line_kwargs`` (``linestyle``, ``alpha``, ``zorder`` ...)
    are forwarded to ``ax.plot``.
    """
    from uacpy.core.environment import Environment
    from uacpy.core.ssp import SoundSpeedProfile

    ssp = env_or_ssp.ssp if isinstance(env_or_ssp, Environment) else env_or_ssp
    if not isinstance(ssp, SoundSpeedProfile):
        raise ConfigurationError(
            "_plot_ssp: pass an Environment or a SoundSpeedProfile, got "
            f"{type(env_or_ssp).__name__}."
        )

    fig, ax = fig_ax(ax, figsize)

    depths = np.asarray(ssp.depths, dtype=float)
    data = np.asarray(ssp.data, dtype=float)           # (n_depth, n_range)

    if ssp.is_range_dependent:
        line_kwargs.setdefault('linewidth', 1.2)
        ranges_km = m_to_km(np.asarray(ssp.ranges, dtype=float))
        if color is None:
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
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
    ax.set_ylabel('Depth (m)')
    invert_yaxis_once(ax)                               # depth positive down
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    # ``get_legend_handles_labels`` already drops matplotlib's '_'-prefixed
    # placeholder labels, so this is empty exactly when nothing was named.
    if legend or (legend is None and ax.get_legend_handles_labels()[1]):
        ax.legend()
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
                           figsize=None, n_range=240, n_depth=200,
                           data_source=True):
    """Small-multiples cross-sections of the **seabed geoacoustic properties**.

    One range × sub-bottom-depth heatmap per property present in
    ``env.bottom`` — sound speed ``cp``, shear speed ``cs``, density ``ρ``,
    compressional / shear attenuation ``αp`` / ``αs`` — each with its own
    colorbar. Properties that are uniformly zero or absent (e.g. shear for a
    fluid seabed) are skipped. Complements :func:`_plot_environment`, which
    colours the seabed by ``cp`` alone; this is where ``cs`` and friends live.

    Works for every ``Bottom`` shape (half-space, layered, range-dependent
    half-space, range-dependent layered); for layered
    seabeds the layers track the bathymetry. Pass ``properties=`` (attribute
    names or symbols) to restrict the panels, or ``title=`` to override the
    figure title. Returns ``(fig, axes)``.
    """
    bottom = env.bottom
    if bottom is None:
        raise ConfigurationError(
            "plot_bottom_properties: env.bottom is None. Attach a seabed to "
            "the environment (env.bottom = Bottom(...)) before plotting its "
            "properties.")

    rmaxes_km = []
    if env.has_range_dependent_bathymetry:
        rmaxes_km.append(m_to_km(float(env.bathymetry.ranges[-1])))
    if bottom.is_range_dependent:
        rmaxes_km.append(m_to_km(float(np.max(bottom.ranges))))
    x_max_km = max(rmaxes_km) if rmaxes_km else 1.0
    r_km = np.linspace(0.0, x_max_km, n_range)

    if env.has_range_dependent_bathymetry:
        b = env.bathymetry
        seafloor_r = np.interp(km_to_m(r_km), b.ranges, b.depths)
    else:
        seafloor_r = np.full(r_km.shape, float(env.depth))

    max_thk = bottom.max_total_thickness()
    sf_max = float(np.max(seafloor_r))
    z_floor = sf_max + max_thk + max(20.0, 0.25 * sf_max)
    z = np.linspace(0.0, z_floor, n_depth)

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
    if figsize is None:
        figsize = (4.4 * ncols, 3.1 * nrows + 0.4)
    # Shared range/depth axes (every panel is the same cross-section) so only
    # the edge axes carry labels — far less clutter than per-panel labels.
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False,
                             sharex=True, sharey=True,
                             constrained_layout=True)
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
        ax_p.set_title(f"{sym}  ({unit})", fontweight='bold', fontsize=11)
        if idx % ncols == 0:                       # left column only
            ax_p.set_ylabel("Depth (m)")
        if idx + ncols >= n:                       # bottom-most visible per col
            ax_p.set_xlabel("Range (km)")
            ax_p.tick_params(labelbottom=True)
        cb = fig.colorbar(pcm, ax=ax_p, pad=0.015, fraction=0.05)
        cb.ax.tick_params(labelsize=8)

    invert_yaxis_once(axes_flat[0])                # shared → inverts all
    for ax_p in axes_flat[n:]:                     # hide unused cells
        ax_p.set_visible(False)

    fig.suptitle(title if title is not None
                 else f"Seabed properties — {_bottom_kind(bottom)}",
                 fontweight='bold', fontsize=13)
    _draw_credit(fig, _credit_attributions(data_source, carrier=env),
                      reserve=False)
    return fig, axes


@typed_plot_error
def plot_absorption(frequencies, absorption=None, ax=None, *, model=None,
                    label=None, title=None, figsize=(7.5, 4.5), **mpl_kw):
    """Volume absorption (dB/km) versus frequency, on log-log axes.

    Either pass pre-computed ``absorption`` (dB/km, same length as
    ``frequencies`` in **Hz**), or a ``model`` to compute it from the
    :mod:`uacpy.core.absorption` formulas: ``'thorp'`` or
    ``'francois_garrison'`` (extra model parameters — ``temperature``,
    ``salinity``, ``pH``, ``depth`` — go through ``model_kwargs``). Returns
    ``(fig, ax)`` like the other plotters; call repeatedly with ``ax=`` to
    overlay several models.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    mk = mpl_kw.pop('model_kwargs', None)
    if absorption is not None and model is not None:
        raise ConfigurationError(
            "plot_absorption: model= selects the computed curve and has no "
            "effect on a pre-computed absorption= array. Drop one of the "
            "two.")
    if absorption is not None and mk:
        raise ConfigurationError(
            "plot_absorption: model_kwargs= configures the model= computation "
            "and has no effect on a pre-computed absorption= array. Drop one "
            "of the two.")
    mk = mk or {}
    if absorption is None:
        if model is None:
            raise ConfigurationError(
                "plot_absorption: pass absorption= (dB/km) or model= "
                "('thorp' / 'francois_garrison').")
        m = str(model).lower().replace('-', '_')
        if m == 'thorp':
            from uacpy.core.absorption import thorp_db_per_km
            absorption = thorp_db_per_km(frequencies)
        elif m in ('francois_garrison', 'fg'):
            from uacpy.core.absorption import francois_garrison_db_per_km
            absorption = francois_garrison_db_per_km(frequencies, **mk)
        else:
            raise ConfigurationError(
                f"plot_absorption: unknown model={model!r}; use 'thorp' or "
                "'francois_garrison'.")
        if label is None:
            label = m
    absorption = np.asarray(absorption, dtype=float)
    fig, ax = fig_ax(ax, figsize)
    ax.loglog(frequencies, absorption, label=label, **mpl_kw)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Absorption (dB/km)")
    ax.set_title(title or "Volume absorption", loc="left")
    ax.grid(which="both", alpha=0.3)
    if label is not None:
        ax.legend(fontsize=8)
    return fig, ax


@typed_plot_error
def _plot_range_profile(profile, *, ax=None, title=None, figsize=(10, 4),
                        **mpl_kw):
    """Render a 1-D ``value(range)`` carrier — :class:`Bathymetry` or
    :class:`Altimetry`. Reached via ``profile.plot()``.

    The carrier declares what it holds: ``_values``, ``_VALUE_LABEL`` /
    ``_VALUE_UNIT`` for the axis label, and ``_AXIS_DOWN`` for the
    orientation. A new range-profile carrier is drawn correctly without
    touching this function."""
    axis_down = profile._AXIS_DOWN
    values = np.asarray(profile._values, dtype=float)
    label = f"{profile._VALUE_LABEL.capitalize()} ({profile._VALUE_UNIT})"
    fig, ax = fig_ax(ax, figsize)
    r_km = m_to_km(np.asarray(profile.ranges, dtype=float))
    style = {'color': 'saddlebrown' if axis_down else 'steelblue',
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
                 fontweight='bold', fontsize=12)
    return fig, ax
