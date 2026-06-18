"""Geographic maps: bathymetry, overview and sea-ice, with relief shading and graticules."""

from __future__ import annotations


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as _mcolors
from typing import Optional, Tuple

from uacpy.visualization.style import SOURCE_MARKER_STYLE
from uacpy.visualization.plots._common import ZORDER_SOURCE, _credit_attributions, _draw_data_credit, _draw_sea_ice
# plot_overview composes a map panel with the TL field and the environment
# cross-section, so it reaches across to those plotters.
from uacpy.visualization.plots.fields import plot_field
from uacpy.visualization.plots.environment import plot_environment


BATHYMETRY_CMAP = _mcolors.LinearSegmentedColormap.from_list('uacpy_bathy', [
    '#86c4cf', '#5aa6bd', '#3f86ab', '#2f6c98',
    '#235485', '#193f6e', '#112c54', '#0a1d3d',
])


def plot_bathymetry_map(
    lats,
    lons,
    depth,
    *,
    transect=None,
    basemap: bool = True,
    coastline_resolution: str = '50m',
    graticule: float = 1.0,
    graticule_minor: Optional[float] = 0.5,
    cmap=BATHYMETRY_CMAP,
    relief: bool = True,
    relief_exag: float = 15.0,
    contours=None,
    figsize: Tuple[float, float] = (9, 7.5),
    ax=None,
    title: Optional[str] = None,
    aspect=None,
    source=None,
    data_source=True,
):
    """Plot a fetched bathymetry grid as a geographic map.

    ``lats`` / ``lons`` are the 1-D axes and ``depth`` the ``(n_lat, n_lon)``
    array from :func:`uacpy.data.fetch_bathy_grid` (``NaN`` = land), with a labelled
    lat/lon graticule (``graticule`` major spacing, ``graticule_minor`` fine
    spacing — ``None`` to disable) and an optional A→B ``transect``.

    With ``basemap=True`` (default) the coastline is drawn from **Natural Earth**
    land polygons — *public domain*: no licence, no attribution, no usage limits.
    ``coastline_resolution`` is ``'110m'`` / ``'50m'`` (default) / ``'10m'``.
    ``basemap=False`` draws a plain lon/lat grid.

    ``relief=True`` (default) renders **shaded relief** (hillshade, exaggeration
    ``relief_exag``) so real seafloor texture shows instead of a smooth dome;
    ``relief=False`` is a flat colormap. ``contours`` overlays labelled depth
    contour lines: ``True`` for the default set (30 isobaths, 0–6000 m every
    200 m), an ``int`` for that many auto levels, or an explicit sequence of
    depths (m); ``None`` (default) draws none.

    ``aspect`` overrides the axes aspect ratio: ``None`` (default) uses the
    equirectangular correction (``1/cos(lat)``); pass a number (e.g. ``1`` for
    equal degree-per-unit scaling) or any Matplotlib aspect value.

    ``data_source`` adds a licence-required data-source credit footnote for a
    standalone figure (``ax=None``). The depth grid carries no provenance itself,
    so pass an ``Environment`` / ``Result`` / list of ``DataSource`` / strings
    (``True`` — the default — has nothing to credit here unless one is given).

    Returns ``(fig, ax)``.
    """
    from uacpy.visualization.basemap import land_polygons

    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    dm = np.ma.masked_invalid(depth)
    lon_rng, lat_rng = (lons.min(), lons.max()), (lats.min(), lats.max())
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    cm = plt.get_cmap(cmap).copy()
    proj = (lambda la, lo: (lo, la))

    if basemap:
        rings = land_polygons(resolution=coastline_resolution)
        cm.set_bad(alpha=0.0)
        ax.set_facecolor('#d7ebf7')                        # sea
        pc = _draw_depth(ax, lons, lats, depth, cm, relief, relief_exag, 1)
        if rings is not None:                              # land OVER the depth: covers
            for ring in rings:                             # coastal grid-cell bleed + crisp border
                ax.fill(ring[:, 0], ring[:, 1], facecolor='#e9e3d4',
                        edgecolor='0.3', lw=0.6, zorder=2)
        ax.set_xlim(*lon_rng)
        ax.set_ylim(*lat_rng)
        _draw_graticule(ax, lon_rng, lat_rng, graticule, graticule_minor, proj)
        ax.set_aspect(aspect if aspect is not None
                      else 1.0 / np.cos(np.radians(np.mean(lat_rng))))  # equirectangular
    else:
        cm.set_bad('#d9cdb8')
        pc = _draw_depth(ax, lons, lats, depth, cm, relief, relief_exag, 1)
        ax.set_xlabel("Longitude [°E]")
        ax.set_ylabel("Latitude [°N]")
        ax.set_aspect(aspect if aspect is not None else 'equal')

    if contours is not None and contours is not False and dm.count() > 1:
        levels = list(range(0, 6000, 200)) if contours is True else contours
        cs = ax.contour(lons, lats, dm, levels=levels, colors='#0c1830',
                        linewidths=0.8, alpha=0.85, zorder=1.5)
        ax.clabel(cs, inline=True, fontsize=8)

    if transect is not None:
        (a_lat, a_lon), (b_lat, b_lon) = transect
        pa, pb = proj(a_lat, a_lon), proj(b_lat, b_lon)
        # zorder 5/6 keep the transect line + A/B labels above the graticule
        # (zorder 4, drawn on top of the map), so 'B' isn't crossed out by gridlines.
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], '-', color='crimson', lw=2.5,
                marker='o', mec='k', zorder=5, label='transect')
        for lbl, (la, lo) in (('A', (a_lat, a_lon)), ('B', (b_lat, b_lon))):
            ax.annotate(lbl, proj(la, lo), color='crimson', fontweight='bold',
                        xytext=(6, 6), textcoords='offset points', zorder=6)
        ax.legend(loc='upper left')
    if source is not None:
        s_lat, s_lon = source
        sp = proj(s_lat, s_lon)
        ax.plot(sp[0], sp[1], zorder=7, **SOURCE_MARKER_STYLE)

    fig.colorbar(pc, ax=ax, label="Water depth [m]")
    ax.set_title(title or "Bathymetry", loc='left', fontsize=11, fontweight='bold')
    if own_fig:
        credit = _credit_attributions(data_source)
        fig.tight_layout(rect=(0, 0.05, 1, 1) if credit else (0, 0, 1, 1))
        _draw_data_credit(fig, credit, reserve=False)
    return fig, ax


def plot_overview(
    env,
    map_args,
    *,
    map_fn=None,
    transect=None,
    tl=None,
    source=None,
    receiver=None,
    figsize: Tuple[float, float] = (15.5, 5.2),
    map_title: str = "Map",
    tl_title: str = "Transmission loss",
    env_title: str = "Environment",
    suptitle: Optional[str] = None,
    data_source=True,
    sea_ice=None,
    map_kwargs: Optional[dict] = None,
):
    """One-call composite for a fetched real-world environment.

    Three panels: a regional **map** (left), a **transmission-loss** field (top
    right) and the **range-dependent environment** (bottom right) — each drawn by
    the corresponding uacpy plotter via its ``ax=`` argument.

    The left map is **pluggable** via ``map_fn`` — any plotter with a
    ``(*map_args, ax=, transect=, source=, title=, **map_kwargs)`` interface, such
    as :func:`plot_bathymetry_map` (default) or :func:`plot_sea_ice_map`.

    Parameters
    ----------
    env : Environment
        The (typically range-dependent) environment for the env panel.
    map_args : tuple
        Positional arguments for ``map_fn`` — e.g. ``(lats, lons, depth)`` for
        :func:`plot_bathymetry_map`, or ``(grid,)`` for :func:`plot_sea_ice_map`.
    map_fn : callable, optional
        Left-panel plotter; defaults to :func:`plot_bathymetry_map`. Map-specific
        options (``contours``, ``aspect``, ``hemi``, …) go in ``map_kwargs``.
    transect : ((latA, lonA), (latB, lonB)), optional
        Drawn on the map (the env/TL transect).
    tl : Field, optional
        A transmission-loss result; if ``None`` the TL panel is left blank.
    source, receiver : optional
        Marked on the environment panel (and the source on the TL panel).
    map_title, tl_title, env_title, suptitle : str, optional
        Panel titles and an optional figure suptitle.
    data_source : default ``True``
        Data-source credit shown as a footnote under the map. ``True`` uses
        ``env.data_sources``; ``None`` / ``False`` hides it; or pass an explicit
        ``Environment`` / ``Result`` / list of ``DataSource`` / strings. See
        :func:`plot_environment` for the same argument on a single panel.
    sea_ice : optional
        Sea-ice cover for the environment panel — a concentration 0–1 or
        ``(ranges_km, concentration)`` (forwarded to :func:`plot_environment`).
    map_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`plot_bathymetry_map`
        (e.g. ``coastline_resolution``, ``graticule``).

    Returns ``(fig, (ax_map, ax_tl, ax_env))``.
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], hspace=0.55, wspace=0.1)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_tl = fig.add_subplot(gs[0, 1])
    ax_env = fig.add_subplot(gs[1, 1])

    # Left panel: any map drawer with a (transect, source, title, ax) interface —
    # e.g. plot_bathymetry_map (default) or plot_sea_ice_map. The geographic
    # source sits at the transect start.
    geo_source = transect[0] if transect is not None else None
    (map_fn or plot_bathymetry_map)(
        *map_args, ax=ax_map, transect=transect, source=geo_source,
        title=map_title, **(map_kwargs or {}))

    if tl is not None:
        plot_field(tl, ax=ax_tl, env=env, title=tl_title)
        if sea_ice is not None:
            _draw_sea_ice(ax_tl, sea_ice)
        if source is not None and getattr(source, 'depths', None) is not None:
            ax_tl.plot(0.0, float(np.atleast_1d(source.depths)[0]), marker='*',
                       ms=13, color='red', mec='k', zorder=ZORDER_SOURCE)
    else:
        ax_tl.text(0.5, 0.5, "no TL result", ha='center', va='center',
                   transform=ax_tl.transAxes)
        ax_tl.set_xticks([])
        ax_tl.set_yticks([])

    plot_environment(env, ax=ax_env, source=source, receiver=receiver,
                     bottom_colorbar=False, sea_ice=sea_ice)
    ax_env.set_title(env_title)

    _draw_data_credit(fig, _credit_attributions(data_source, carrier=env),
                      center_ax=ax_map)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight='bold')
    return fig, (ax_map, ax_tl, ax_env)


def plot_sea_ice_map(grid, *, hemi: str = 'N', transect=None, source=None,
                     cmap='Blues_r', graticule: float = 5.0, zoom: bool = True,
                     concentration_label='Sea-ice concentration (fraction)',
                     title=None, ax=None,
                     figsize: Tuple[float, float] = (6.5, 6.5)):
    """Plot a sea-ice concentration grid (0–1, ``NaN`` = land) as a polar map.

    The sibling of :func:`plot_bathymetry_map` for sea ice: ``grid`` is a 2-D
    concentration array on the NSIDC polar-stereographic grid (e.g.
    ``uacpy.data.sea_ice_grid(month, hemi='N')``); ice maps to white, open water
    to blue, land (``NaN``) to a neutral fill. ``transect`` (``((latA, lonA),
    (latB, lonB))``) and ``source`` (``(lat, lon)``) are given in **lon/lat** and
    drawn with the *same* crimson A→B line and source-star notation as the depth
    map (reprojected to the grid via ``hemi``). With ``zoom`` (default) the view
    is framed to a **regional window** around the transect/source — filling the
    panel like the depth map — instead of the whole hemisphere. Pass ``ax=`` to
    compose. Returns ``(fig, ax)``.
    """
    from uacpy.data.seaice_local import sea_ice_pixel
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad('#b8a98f')                          # land / no-data
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cm, vmin=0.0, vmax=1.0,
                   origin='upper')

    if graticule:                                  # labelled lat/lon graticule
        sgn = 1 if hemi == 'N' else -1
        suffix = 'N' if hemi == 'N' else 'S'

        def _line(pts):
            pts = [p for p in pts if p is not None]
            if len(pts) > 1:
                ax.plot([c for _, c in pts], [r for r, _ in pts], color='0.4',
                        lw=0.6, alpha=0.45, zorder=2)
        lon_s = np.arange(-180.0, 181.0, 2.0)
        for lat_c in np.arange(50.0, 90.0, graticule) * sgn:       # parallels
            _line([sea_ice_pixel((lat_c, lo), hemi=hemi) for lo in lon_s])
            p0 = sea_ice_pixel((lat_c, 0.0), hemi=hemi)
            if p0:
                ax.annotate(f"{abs(int(lat_c))}°{suffix}", (p0[1], p0[0]),
                            color='0.25', fontsize=7, ha='center', va='center',
                            zorder=2, bbox=dict(boxstyle='round,pad=0.1',
                                                fc='white', ec='none', alpha=0.6))
        lat_s = np.arange(50.0, 89.0, 1.0) * sgn
        for lon_c in np.arange(-180.0, 180.0, 30.0):               # meridians
            _line([sea_ice_pixel((la, lon_c), hemi=hemi) for la in lat_s])

    pa = pb = ps = None
    if transect is not None:
        (a_lat, a_lon), (b_lat, b_lon) = transect
        pa = sea_ice_pixel((a_lat, a_lon), hemi=hemi)
        pb = sea_ice_pixel((b_lat, b_lon), hemi=hemi)
        if pa and pb:
            # zorder 5/6 keep the transect + A/B labels above the graticule.
            ax.plot([pa[1], pb[1]], [pa[0], pb[0]], '-', color='crimson', lw=2.5,
                    marker='o', mec='k', zorder=5, label='transect')
            for lbl, p in (('A', pa), ('B', pb)):
                ax.annotate(lbl, (p[1], p[0]), color='crimson', fontweight='bold',
                            xytext=(6, 6), textcoords='offset points', zorder=6)
            ax.legend(loc='upper left')
    if source is not None:
        ps = sea_ice_pixel(source, hemi=hemi)
        if ps:
            ax.plot(ps[1], ps[0], zorder=7, **SOURCE_MARKER_STYLE)

    # Frame a regional window around the transect/source (square pixels kept, the
    # window widened to the panel) so the map fills the axis like the depth map,
    # rather than showing the whole hemisphere with the region a speck.
    focus = [p for p in (pa, pb, ps) if p]
    if zoom and focus:
        rr = [p[0] for p in focus]
        cc = [p[1] for p in focus]
        cy, cx = 0.5 * (min(rr) + max(rr)), 0.5 * (min(cc) + max(cc))
        half = max(max(rr) - min(rr), max(cc) - min(cc), 6.0) * 0.7 + 16.0
        ny, nx = np.shape(grid)
        ax.set_xlim(max(cx - 1.6 * half, -0.5), min(cx + 1.6 * half, nx - 0.5))
        ax.set_ylim(min(cy + half, ny - 0.5), max(cy - half, -0.5))  # origin='upper'

    fig.colorbar(im, ax=ax, label=concentration_label)
    ax.set_title(title or "Sea-ice concentration", loc='left', fontsize=11,
                 fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():                   # match the depth-map frame
        spine.set_edgecolor('0.2')
        spine.set_linewidth(1.0)
    if own_fig:
        fig.tight_layout()
    return fig, ax


def _draw_depth(ax, lons, lats, depth, cmap, relief, exag, zorder):
    """Draw the depth field as shaded relief (``relief``) or flat ``pcolormesh``.

    Shaded relief reveals real seafloor texture (slopes, canyons, seamounts) and
    avoids the smooth-"dome" look of a flat sequential colormap on bowl-shaped
    bathymetry. Returns a mappable for the colorbar. Land (NaN) is transparent.
    """
    dm = np.ma.masked_invalid(depth)
    finite = np.asarray(depth, dtype=float)[np.isfinite(depth)]
    if (not relief or finite.size < 4 or min(np.shape(depth)) < 3
            or np.ptp(finite) < 1e-9):
        return ax.pcolormesh(lons, lats, dm, cmap=cmap, shading='auto',
                             zorder=zorder)

    from matplotlib.colors import LightSource, Normalize
    base = plt.get_cmap(cmap)
    norm = Normalize(float(finite.min()), float(finite.max()))
    dy = (lats.max() - lats.min()) / lats.size * 111320.0
    dx = ((lons.max() - lons.min()) / lons.size * 111320.0
          * np.cos(np.radians(float(lats.mean()))))
    filled = np.where(np.isnan(depth), norm.vmax, depth)
    rgb = LightSource(azdeg=315, altdeg=45).shade(
        filled, cmap=base, blend_mode='soft', vert_exag=exag,
        dx=dx, dy=dy, norm=norm)
    rgb[np.isnan(depth), 3] = 0.0                       # land → transparent
    ax.imshow(rgb, extent=[lons.min(), lons.max(), lats.min(), lats.max()],
              origin='lower', zorder=zorder, interpolation='nearest',
              aspect='auto')
    return plt.cm.ScalarMappable(norm=norm, cmap=base)


def _draw_graticule(ax, lon_range, lat_range, major, minor, proj):
    """Labelled lat/lon graticule on a (possibly projected) map axis."""
    def seq(rng, step):
        return np.arange(np.ceil(rng[0] / step) * step, rng[1] + 1e-6, step)

    def fmt(v, pos_hemi, neg_hemi):
        return f"{abs(v):g}°{pos_hemi if v >= 0 else neg_hemi}"

    lon_maj, lat_maj = seq(lon_range, major), seq(lat_range, major)
    ax.set_xticks([proj(0.0, lo)[0] for lo in lon_maj])
    ax.set_xticklabels([fmt(lo, 'E', 'W') for lo in lon_maj])
    ax.set_yticks([proj(la, 0.0)[1] for la in lat_maj])
    ax.set_yticklabels([fmt(la, 'N', 'S') for la in lat_maj])
    ax.set_axisbelow(False)                             # graticule on top of the map
    if minor:
        ax.set_xticks([proj(0.0, lo)[0] for lo in seq(lon_range, minor)], minor=True)
        ax.set_yticks([proj(la, 0.0)[1] for la in seq(lat_range, minor)], minor=True)
        ax.grid(which='minor', color='1.0', lw=0.3, alpha=0.25, zorder=4)
    ax.grid(which='major', color='1.0', lw=0.6, alpha=0.45, zorder=4)
    ax.tick_params(which='major', length=5, direction='out', labelsize=9)
    ax.tick_params(which='minor', length=2.5, direction='out')
    for spine in ax.spines.values():
        spine.set_edgecolor('0.2')
        spine.set_linewidth(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Modes — three distinct views
# ─────────────────────────────────────────────────────────────────────────────
