"""Geographic maps: bathymetry, overview and sea-ice, with relief shading and graticules."""

from __future__ import annotations


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as _mcolors
import matplotlib.collections as _mcoll
import matplotlib.patheffects as _patheffects
import matplotlib.ticker as _mticker
from typing import Optional, Sequence, Tuple

from uacpy.core.exceptions import ConfigurationError
from uacpy.core.units import km_to_m
from uacpy.visualization.style import SOURCE_MARKER_STYLE
from uacpy.visualization.plots._common import _cell_edge_extent, _credit_attributions, _default_value, _draw_credit, _draw_sea_ice, _model_attribution, _refuse_axes_after_carrier, _value_label, typed_plot_error, _title_or
# plot_overview composes a map panel with the TL field and the environment
# cross-section, so it reaches across to those plotters.
from uacpy.visualization.plots.fields import (plot_field, _draw_contours,
                                              _colorbar_extend,
                                              _drawn_levels, _value_mesh)
from uacpy.visualization.plots.environment import plot_environment


#: Hemispheres of the NSIDC polar-stereographic sea-ice grid.
_SEA_ICE_HEMISPHERES = ('N', 'S')

#: Sea-ice map zoom window, in grid pixels (see ``plot_sea_ice_map``).
_ZOOM_MIN_SPAN_PX = 6.0
_ZOOM_SPAN_PAD = 0.7
_ZOOM_MARGIN_PX = 16.0
_ZOOM_WIDTH_RATIO = 1.6

#: The isobaths ``plot_bathymetry_map(contours=True)`` draws: 0-6000 m every
#: 200 m (uacpy guidance).
_DEFAULT_ISOBATHS_M = list(range(0, 6000, 200))

#: How every isobath is drawn: dark navy over the depth colours, under the
#: land fill (zorder 2) so a line never crosses the coast.
_ISOBATH_STYLE = {'colors': '#0c1830', 'linewidths': 0.8, 'alpha': 0.85,
                  'zorder': 1.5}

#: Land on the bathymetry map drawn without a coastline (``basemap=False``).
_PLAIN_MAP_LAND_COLOR = '#d9cdb8'

#: How the sea-ice map's concentration contours are drawn.
_SEA_ICE_CONTOUR_STYLE = {'color': 'black', 'linewidth': 1.0, 'zorder': 3}

BATHYMETRY_CMAP = _mcolors.LinearSegmentedColormap.from_list('uacpy_bathy', [
    '#86c4cf', '#5aa6bd', '#3f86ab', '#2f6c98',
    '#235485', '#193f6e', '#112c54', '#0a1d3d',
])


def _unwrap_lon_axis(lons):
    """``(lons, wrapped)`` — a longitude axis made monotone over the antimeridian.

    :func:`uacpy.data.fetch_bathy_grid` returns an axis that runs eastward but
    is wrapped into [-180, 180), so a range crossing the antimeridian arrives as
    ``[178 … 179.9, -179.9 … -178]``. Read literally that is a full-globe span in
    the wrong order: ``min``/``max`` inflate the window to the whole world, and
    the descending-axis test in :func:`_draw_depth` fires on the wrap and mirrors
    the relief east–west. Adding a turn at each wrap restores the real 4°
    eastward span (``[178 … 182]``), which the extent, the graticule, the
    contours and the hillshade cell size then all read correctly.

    An axis with no wrap comes back untouched, so a plainly ascending or plainly
    descending input keeps its own handling. A step of exactly 360° is two
    samples on the same meridian — the two ends of a full-globe axis — and is a
    real span rather than a wrap, so it is left alone as well.
    """
    lons = np.asarray(lons, dtype=float)
    d = np.diff(lons)
    turn = np.where((-360.0 < d) & (d < -180.0), 360.0,
                    np.where((180.0 < d) & (d < 360.0), -360.0, 0.0))
    if not turn.any():
        return lons, False
    return np.concatenate(([lons[0]], lons[0] + np.cumsum(d + turn))), True


def _lon_label_value(lon):
    """A tick's longitude folded into [-180, 180] for its label, so a tick at
    182° in an unwrapped antimeridian window reads 178°W. A tick already in
    range keeps its own value, sign included at the antimeridian itself."""
    lon = float(lon)
    if -180.0 <= lon <= 180.0:
        return lon
    folded = (lon + 180.0) % 360.0 - 180.0
    return 180.0 if folded == -180.0 else folded


def _lon_tick_label(lon):
    """A longitude tick's label: folded into [-180, 180] and suffixed with its
    hemisphere, so a tick at 182° reads 178°W."""
    value = _lon_label_value(lon)
    return f"{abs(value):g}°{'E' if value >= 0 else 'W'}"


_LABEL_OFFSET_PT = 6.0


def _endpoint_label_offset(ax, here, other):
    """Offset (points) of a transect end label: inside the axes, beside the
    line.

    Of the four diagonal offsets, those that would carry the label out of the
    axes are dropped first — a transect starting at the window's corner
    otherwise pushed its label over the title — and the survivor closest to
    perpendicular to the line wins, so the label sits beside the line rather
    than on it.
    """
    to_px = ax.figure.dpi / 72.0
    p, q = ax.transData.transform([here, other])
    along = (q - p) / (np.hypot(*(q - p)) or 1.0)

    def label_stays_inside(offset):                # 2.5 offsets ~ the label's extent
        x, y = p + np.asarray(offset) * to_px * 2.5
        return ax.bbox.x0 < x < ax.bbox.x1 and ax.bbox.y0 < y < ax.bbox.y1
    candidates = [(sx * _LABEL_OFFSET_PT, sy * _LABEL_OFFSET_PT)
                  for sx in (1, -1) for sy in (1, -1)]
    kept = [c for c in candidates if label_stays_inside(c)] or candidates
    return min(kept, key=lambda c: abs(np.dot(np.asarray(c) / np.hypot(*c),
                                              along)))


def _draw_land(ax, rings, turns, lon_rng, lat_rng):
    """Fill the land rings that reach the ``lon_rng`` × ``lat_rng`` window, each
    shifted by every longitude ``turn``, as one collection.

    A ring is kept when its bounding box (after the shift) overlaps the window,
    which keeps a ring enclosing the whole window too; every other ring would be
    clipped away by the axes, so drawing it only costs time."""
    kept = []
    for ring in rings:
        x, y = ring[:, 0], ring[:, 1]
        if y.max() < lat_rng[0] or y.min() > lat_rng[1]:
            continue
        for turn in turns:
            if x.max() + turn >= lon_rng[0] and x.min() + turn <= lon_rng[1]:
                kept.append(np.column_stack([x + turn, y]))
    land = _mcoll.PolyCollection(kept, facecolors='#e9e3d4', edgecolors='0.3',
                                 linewidths=0.6, zorder=2)
    ax.add_collection(land, autolim=False)
    return land


@typed_plot_error
def plot_bathymetry_map(
    lats,
    lons=None,
    depth=None,
    ax=None,
    *,
    transect=None,
    basemap: bool = True,
    coastline_resolution: str = '50m',
    graticule: Optional[float] = 1.0,
    graticule_minor: Optional[float] = 0.5,
    cmap=BATHYMETRY_CMAP,
    relief: bool = True,
    relief_exag: float = 15.0,
    contours=None,
    figsize: Tuple[float, float] = (9, 7.5),
    title: Optional[str] = None,
    aspect=None,
    source=None,
    show_data_credit=True,
    show_colorbar: bool = True,
):
    """Plot a fetched bathymetry grid as a geographic map.

    Takes the :class:`~uacpy.data.BathyGrid` :func:`uacpy.data.fetch_bathy_grid`
    returns, or its arrays: ``lats`` / ``lons`` the 1-D axes and ``depth`` the
    ``(n_lat, n_lon)`` array (``NaN`` = land). The map has a labelled
    lat/lon graticule (``graticule`` labelled spacing, ``graticule_minor`` fine
    unlabelled spacing — ``None`` on either disables that layer, and ``None`` on
    ``graticule`` leaves the map with no tick labels) and an optional A→B
    ``transect``.

    With ``basemap=True`` (default) the coastline is drawn from **Natural Earth**
    land polygons — *public domain*: no licence, no attribution, no usage limits.
    ``coastline_resolution`` is ``'110m'`` / ``'50m'`` (default) / ``'10m'``.
    ``basemap=False`` draws a plain lon/lat grid.

    ``relief=True`` (default) renders **shaded relief** (hillshade, exaggeration
    ``relief_exag``) so real seafloor texture shows instead of a smooth dome;
    ``relief=False`` is a flat colormap. ``contours`` overlays labelled depth
    contour lines: ``True`` for the default set (30 isobaths, 0–6000 m every
    200 m), or the depths (m) to draw — a number is one isobath, a sequence
    several, as on every map plotter; ``None`` (default) draws none.
    Explicit depths follow the rule of every map plotter: one outside the
    map's depth range is reported in one warning and the others are drawn.

    ``aspect`` overrides the axes aspect ratio: ``None`` (default) uses the
    equirectangular correction (``1/cos(lat)``); pass a number (e.g. ``1`` for
    equal degree-per-unit scaling) or any Matplotlib aspect value.

    ``show_data_credit`` adds a licence-required data-source credit footnote for a
    standalone figure (``ax=None``). ``True`` (the default) credits a
    ``BathyGrid``'s own ``provenance``; bare arrays carry none, so with them
    pass an ``Environment`` / ``Result`` / list of ``DataSource`` / strings.

    Returns ``(fig, ax)``.

    Parameters
    ----------
    lats : BathyGrid or array_like
        The grid itself, or its 1-D latitude axis (deg).
    lons : array_like, optional
        The 1-D longitude axis (deg), with array input.
    depth : ndarray, optional
        ``(n_lat, n_lon)`` depths (m, positive down; ``NaN`` is land), with
        array input.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    transect : ((lat, lon), (lat, lon)), optional
        An A→B line to draw.
    basemap : bool, optional
        Draw the Natural Earth coastline. Default True.
    coastline_resolution : {'110m', '50m', '10m'}, optional
        Coastline scale. Default ``'50m'``.
    graticule, graticule_minor : float, optional
        Labelled and fine graticule spacing (deg); ``None`` drops that layer.
        Default 1 and 0.5.
    cmap : Colormap or str, optional
        Depth colormap. Default the package bathymetry map.
    relief : bool, optional
        Shade the relief. Default True.
    relief_exag : float, optional
        Vertical exaggeration of the shading. Default 15.
    contours : bool, float or sequence of float, optional
        Isobaths to draw (see above).
    figsize : tuple, optional
        Size (inches) of the new figure. Default ``(9, 7.5)``.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    aspect : float or str, optional
        Axes aspect (see above); ``None`` is the equirectangular correction.
    source : (lat, lon), optional
        Mark the source.
    show_data_credit : bool, Environment, Result or sequence, optional
        The data-source credit footnote of a standalone figure (see above).
    show_colorbar : bool, optional
        Draw the water-depth colorbar. Default True.
    """
    from uacpy.visualization.basemap import land_polygons

    provenance = None
    if hasattr(lats, 'depths'):
        _refuse_axes_after_carrier('plot_bathymetry_map', lats,
                                   [('lons', lons), ('depth', depth)])
    if lons is None and depth is None and hasattr(lats, 'depths'):
        # A BathyGrid: its arrays, and the dataset record its credit names.
        provenance = getattr(lats, 'provenance', None)
        lats, lons, depth = lats.lats, lats.lons, lats.depths
    elif lons is None or depth is None:
        raise ConfigurationError(
            "plot_bathymetry_map: pass the BathyGrid fetch_bathy_grid "
            "returns, or all three arrays (lats, lons, depth).")
    lats = np.asarray(lats, dtype=float)
    lons, lon_wrapped = _unwrap_lon_axis(lons)
    dm = np.ma.masked_invalid(depth)
    lon_rng, lat_rng = (lons.min(), lons.max()), (lats.min(), lats.max())
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    cm = plt.get_cmap(cmap)
    # Plate carrée: degrees are plotted raw as (x, y) = (lon, lat), no
    # coordinate transform. The latitude foreshortening is applied to the axes
    # instead, via set_aspect below, so tick values stay in real degrees.
    # Over an unwrapped antimeridian window an overlay handed a raw longitude
    # (a transect end at -178) belongs beside the grid it annotates (182), so it
    # is moved to the turn nearest the window's own centre. A window that does
    # not cross leaves every coordinate exactly as it was given.
    if lon_wrapped:
        lon_mid = 0.5 * (lon_rng[0] + lon_rng[1])
        proj = (lambda la, lo: (lon_mid + (float(lo) - lon_mid + 180.0) % 360.0
                                - 180.0, la))
    else:
        proj = (lambda la, lo: (lo, la))

    if basemap:
        rings = land_polygons(resolution=coastline_resolution)
        cm = cm.with_extremes(bad='none')                  # transparent: sea shows through
        ax.set_facecolor('#d7ebf7')                        # sea
        pc = _draw_depth(ax, lons, lats, depth, cm, relief, relief_exag, 1)
        # Natural Earth cuts its rings at ±180, so an unwrapped window reaching
        # past the antimeridian is covered by a second copy of the land a turn
        # away — without it the half of the window over the dateline draws as
        # open sea. Both turns are laid down and the axes clip to the window.
        turns = (-360.0, 0.0, 360.0) if lon_wrapped else (0.0,)
        if rings is not None:                              # land OVER the depth: covers
            _draw_land(ax, rings, turns, lon_rng, lat_rng) # coastal grid-cell bleed + crisp border
        ax.set_xlim(*lon_rng)
        ax.set_ylim(*lat_rng)
        _draw_graticule(ax, lon_rng, lat_rng, graticule, graticule_minor, proj)
    else:
        # Land is tan with or without relief: the flat mesh paints a NaN cell
        # in the colormap's ``bad`` colour, the relief image leaves it
        # transparent over the axes face.
        cm = cm.with_extremes(bad=_PLAIN_MAP_LAND_COLOR)
        ax.set_facecolor(_PLAIN_MAP_LAND_COLOR)
        pc = _draw_depth(ax, lons, lats, depth, cm, relief, relief_exag, 1)
        if lon_wrapped:
            # This branch draws no graticule, so its ticks are matplotlib's own
            # numbers. An unwrapped window runs past 180, where a plain number
            # reads 182 for a map that is at 178°W — so name the hemisphere on
            # the tick, as the graticule branch does, and drop it from the axis
            # label. A window that does not cross keeps the plain degrees-east
            # numbering it has always had.
            ax.xaxis.set_major_formatter(
                _mticker.FuncFormatter(lambda v, _pos: _lon_tick_label(v)))
            ax.set_xlabel("Longitude")
        else:
            ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
    # Equirectangular: on the ground 1° of longitude is cos(lat) as long as 1°
    # of latitude, so stretching the y axis by 1/cos(lat) makes the panel
    # distance-isotropic at the mid-latitude of the window — with and without
    # the coastline basemap.
    ax.set_aspect(aspect if aspect is not None
                  else 1.0 / np.cos(np.radians(np.mean(lat_rng))))

    if contours is True and dm.count() > 1:
        # The package's own isobath set: none is a level the caller named,
        # so one out of the map's depth range is simply not drawn.
        cs = ax.contour(lons, lats, dm, levels=_DEFAULT_ISOBATHS_M,
                        **_ISOBATH_STYLE)
        ax.clabel(cs, inline=True, fontsize='small')
    elif contours is not None and contours is not False:
        # Depths the caller named follow the rule of every map plotter: one
        # outside the map's range is reported, a degenerate grid warns.
        _draw_contours(ax, lons, lats, np.ma.filled(dm.astype(float), np.nan),
                       contours, who='plot_bathymetry_map', fmt='%g',
                       linewidth=_ISOBATH_STYLE['linewidths'], unit=' m',
                       color=_ISOBATH_STYLE['colors'],
                       alpha=_ISOBATH_STYLE['alpha'],
                       zorder=_ISOBATH_STYLE['zorder'])

    if transect is not None:
        (a_lat, a_lon), (b_lat, b_lon) = transect
        pa, pb = proj(a_lat, a_lon), proj(b_lat, b_lon)
        # zorder 5/6 keep the transect line + A/B labels above the graticule
        # (zorder 4, drawn on top of the map), so 'B' isn't crossed out by gridlines.
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], '-', color='crimson', lw=2.5,
                marker='o', mec='k', zorder=5, label='transect')
        for lbl, here, other in (('A', pa, pb), ('B', pb, pa)):
            dx, dy = _endpoint_label_offset(ax, here, other)
            ax.annotate(lbl, here, color='crimson', fontweight='bold',
                        xytext=(dx, dy), textcoords='offset points',
                        ha='left' if dx > 0 else 'right',
                        va='bottom' if dy > 0 else 'top', zorder=6,
                        path_effects=[_patheffects.withStroke(
                            linewidth=2.5, foreground='white')])
        # 'best' keeps the legend off the transect line and its end markers;
        # 'upper left' sits on the start marker of a transect from that corner.
        ax.legend(loc='best')
    if source is not None:
        s_lat, s_lon = source
        sp = proj(s_lat, s_lon)
        ax.plot(sp[0], sp[1], zorder=7, **SOURCE_MARKER_STYLE)

    if show_colorbar:
        fig.colorbar(pc, ax=ax, label="Water depth (m)")
    ax.set_title(_title_or(title, "Bathymetry"), fontsize='large', fontweight='bold')
    if own_fig:
        credit = _credit_attributions(
            [provenance] if show_data_credit is True and provenance is not None
            else show_data_credit)
        fig.tight_layout(rect=(0, 0.05, 1, 1) if credit else (0, 0, 1, 1))
        _draw_credit(fig, credit, reserve=False)
    return fig, ax


@typed_plot_error
def plot_overview(
    env,
    map_data,
    *,
    map_fn=None,
    transect=None,
    tl=None,
    source=None,
    receiver=None,
    figsize: Optional[Tuple[float, float]] = None,
    map_title: str = "Map",
    tl_title: str = "Transmission loss",
    env_title: str = "Environment",
    title: Optional[str] = None,
    show_data_credit=True,
    sea_ice=None,
    map_kwargs: Optional[dict] = None,
    tl_kwargs: Optional[dict] = None,
    fig=None,
):
    """One-call composite for a fetched real-world environment.

    Three panels: a regional **map** (left), a **transmission-loss** field (top
    right) and the **range-dependent environment** (bottom right) — each drawn by
    the corresponding uacpy plotter via its ``ax=`` argument.

    The left map is **pluggable** via ``map_fn`` — any plotter with a
    ``(data, ax=, transect=, source=, title=, **map_kwargs)`` interface, such
    as :func:`plot_sea_ice_map`; the default draws a
    :class:`~uacpy.data.BathyGrid` with :func:`plot_bathymetry_map`.

    Parameters
    ----------
    env : Environment
        The (typically range-dependent) environment for the env panel.
    map_data : BathyGrid or ndarray
        The map's data, handed to ``map_fn``. The default bathymetry map takes
        the :class:`~uacpy.data.BathyGrid` :func:`uacpy.data.fetch_bathy_grid`
        returns; for arrays of your own, build ``BathyGrid(lats=..., lons=...,
        depths=...)`` first. ``map_fn=plot_sea_ice_map`` takes the sea-ice
        array (:func:`uacpy.data.sea_ice_grid`). A tuple or list is refused.
    map_fn : callable, optional
        Left-panel plotter; defaults to :func:`plot_bathymetry_map`. Map-specific
        options (``contours``, ``aspect``, ``hemi``, …) go in ``map_kwargs``.
    transect : ((latA, lonA), (latB, lonB)), optional
        Drawn on the map (the env/TL transect).
    tl : Field, optional
        A transmission-loss result; if ``None`` the TL panel is left blank.
    source, receiver : optional
        Marked on the environment panel (and the source on the TL panel).
    map_title, tl_title, env_title : str, optional
        Per-panel titles.
    title : str, optional
        Figure-level title over all three panels.
    show_data_credit : default ``True``
        Data-source credit shown as a footnote under the map. ``True`` uses
        ``env.data_sources`` and the provenance of a ``BathyGrid``
        ``map_data``; ``None`` / ``False`` hides it; or pass an explicit
        ``Environment`` / ``Result`` / list of ``DataSource`` / strings. See
        :func:`plot_environment` for the same argument on a single panel.
    sea_ice : optional
        Sea-ice cover for the environment and TL panels — a uniform
        concentration 0–1, or the :class:`~uacpy.data.AlongTrack`
        :func:`uacpy.data.fetch_sea_ice_concentration_transect` returns.
    map_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`plot_bathymetry_map`
        (e.g. ``coastline_resolution``, ``graticule``).
    tl_kwargs : dict, optional
        Extra keyword arguments forwarded to :func:`plot_field` for the TL
        panel — e.g. ``dict(vmin=40, vmax=110, cmap='turbo')`` to override
        the default TL colour scale (20–120 dB). ``show_colorbar=False``
        drops the TL panel's inset colour bar.
    figsize : tuple, optional
        Size of the new figure, ``(15.5, 5.2)`` by default.
    fig : Figure or SubFigure, optional
        Draw the three panels into this (a panel of a larger figure) instead
        of a new figure. Its size, layout and credit line are then the
        caller's, so ``figsize=`` alongside it is refused and no footnote is
        drawn.

    Returns ``(fig, (ax_map, ax_tl, ax_env))``.
    """
    owns_fig = fig is None
    if owns_fig:
        fig = plt.figure(figsize=(15.5, 5.2) if figsize is None else figsize)
    elif figsize is not None:
        raise ConfigurationError(
            "plot_overview: figsize= sizes a new figure, and fig= hands in one "
            "that already has its size — give one of them.")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], hspace=0.55, wspace=0.1)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_tl = fig.add_subplot(gs[0, 1])
    ax_env = fig.add_subplot(gs[1, 1])

    # Left panel: any map drawer with a (transect, source, title, ax) interface —
    # e.g. plot_bathymetry_map (default) or plot_sea_ice_map. The geographic
    # source sits at the transect start.
    geo_source = transect[0] if transect is not None else None
    (map_fn or plot_bathymetry_map)(
        *_map_positional_args(map_data, map_fn), ax=ax_map,
        transect=transect, source=geo_source,
        title=map_title, **(map_kwargs or {}))

    if tl is not None:
        # Suppress plot_field's width-stealing fig.colorbar(fraction=) and draw
        # the TL colorbar as an inset instead, mirroring the env panel's inset
        # cp bars. A stealing colorbar shrinks only ax_tl, leaving the two
        # right-column panels unequal; an inset bar steals no axes width, so
        # both panels keep the full gridspec cell and line up.
        tl_kw = dict(tl_kwargs or {})
        # The TL bar is this composite's inset, so tl_kwargs' show_colorbar
        # decides that inset rather than reaching plot_field.
        show_tl_colorbar = tl_kw.pop('show_colorbar', None)
        # Only the source is marked here: the TL heatmap's cells ARE the
        # receiver grid, so receiver dots would repeat the field's own
        # sampling over it. The environment panel below carries them.
        plot_field(tl, ax=ax_tl, env=env, source=source,
                   title=tl_title, show_colorbar=False, **tl_kw)
        tl_mappable = _value_mesh(ax_tl)
        if tl_mappable is not None and show_tl_colorbar is not False:
            tl_cax = ax_tl.inset_axes((1.04, 0.0, 0.03, 1.0))
            # The bar annotates whatever view tl_kwargs asked plot_field
            # for, so it is labelled from the field and that view — a
            # hardcoded 'TL (dB)' would caption linear |p| as a loss in dB.
            tl_value = tl_kw.get('value') or _default_value(tl)
            fig.colorbar(
                tl_mappable, cax=tl_cax, label=_value_label(tl, tl_value),
                extend=_colorbar_extend(
                    _drawn_levels(tl, tl_value, who='plot_overview'),
                    tl_mappable.get_clim()))
        if sea_ice is not None:
            _draw_sea_ice(ax_tl, sea_ice)
    else:
        ax_tl.text(0.5, 0.5, "no TL result", ha='center', va='center',
                   transform=ax_tl.transAxes)
        ax_tl.set_xticks([])
        ax_tl.set_yticks([])

    # The environment panel paints out to the furthest range it knows; a
    # range-independent environment without ``receiver=`` knows none, so it
    # is handed the TL panel's right limit — the far edge of its last painted
    # cell — the x limit it is synced to below.
    plot_environment(env, ax=ax_env, source=source, receiver=receiver,
                    show_bottom_colorbar=True, sea_ice=sea_ice,
                    x_max=(km_to_m(max(ax_tl.get_xlim()))
                           if tl is not None and 'range' in tl.coords
                           else None))
    ax_env.set_title(env_title)

    # Both right-column panels keep their full gridspec cell — neither
    # colorbar steals axes width (the TL bar is the inset above; the env cp
    # bars are inset by plot_environment) — so they share x0/width by
    # construction. Sync the x-limits so the equal-width panels line up
    # range-for-range.
    if tl is not None:
        ax_env.set_xlim(ax_tl.get_xlim())

    if owns_fig:
        credit = _credit_attributions(show_data_credit, carrier=env)
        map_provenance = getattr(map_data, 'provenance', None)
        if show_data_credit is True and map_provenance is not None:
            credit += [a for a in _credit_attributions([map_provenance])
                       if a not in credit]
        _draw_credit(fig, credit,
                     model=_model_attribution(tl) if tl is not None else None,
                     center_ax=ax_map)

    if title:
        fig.suptitle(title, fontsize='large', fontweight='bold')
    return fig, (ax_map, ax_tl, ax_env)


def _map_positional_args(map_data, map_fn) -> tuple:
    """The positional arguments ``plot_overview`` hands its map plotter: the
    default bathymetry map draws a BathyGrid's ``(lats, lons, depths)``; a
    ``map_fn`` takes ``map_data`` as its one argument."""
    from uacpy.data import BathyGrid
    if isinstance(map_data, (tuple, list)):
        raise ConfigurationError(
            f"plot_overview: map_data is a {type(map_data).__name__}; the map "
            f"takes a record, not loose arrays.",
            remediation="Pass the BathyGrid fetch_bathy_grid returns, or "
                        "build BathyGrid(lats=, lons=, depths=) from your "
                        "arrays; pass the sea-ice array itself with "
                        "map_fn=plot_sea_ice_map.")
    if map_fn is not None:
        return (map_data,)
    if not isinstance(map_data, BathyGrid):
        raise ConfigurationError(
            f"plot_overview: the bathymetry map draws a BathyGrid; map_data "
            f"is a {type(map_data).__name__}.",
            remediation="Pass the BathyGrid fetch_bathy_grid returns, or "
                        "build BathyGrid(lats=, lons=, depths=) from your "
                        "arrays.")
    return (map_data.lats, map_data.lons, map_data.depths)


@typed_plot_error
def plot_sea_ice_map(grid, ax=None, *, hemi: str = 'N', transect=None,
                     source=None, cmap='Blues_r', graticule: float = 5.0,
                     zoom: bool = True,
                     concentration_label='Sea-ice concentration (fraction)',
                     contours: Optional[Sequence[float]] = None,
                     title=None,
                     figsize: Tuple[float, float] = (6.5, 6.5),
                     show_data_credit=True, show_colorbar: bool = True):
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
    compose. ``hemi`` is ``'N'`` or ``'S'``.

    ``show_data_credit`` adds the licence-required data-source credit footnote for
    a standalone figure (``ax=None``), as :func:`plot_bathymetry_map` does:
    ``True`` (default) credits the NSIDC sea-ice product the grid comes from,
    ``None`` / ``False`` hides it, or pass an ``Environment`` / list of
    ``DataSource`` / strings. Returns ``(fig, ax)``.

    Parameters
    ----------
    grid : ndarray
        Concentration on the NSIDC polar grid (0-1, ``NaN`` is land).
    hemi : {'N', 'S'}, optional
        Hemisphere of the grid. Default ``'N'``.
    transect : ((lat, lon), (lat, lon)), optional
        An A→B line to draw.
    source : (lat, lon), optional
        Mark the source.
    cmap : str, optional
        Colormap. Default ``'Blues_r'``.
    graticule : float, optional
        Graticule spacing (deg). Default 5.
    zoom : bool, optional
        Frame a regional window around the transect or source. Default
        True.
    concentration_label : str, optional
        Colorbar label.
    contours : sequence or array of float, optional
        Concentrations (0-1) drawn as labelled contour lines, by the rule of
        every map plotter: a level outside the grid's range is reported and
        not drawn. ``None`` (default) draws none.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    figsize : tuple, optional
        Size (inches) of the new figure. Default ``(6.5, 6.5)``.
    show_data_credit : bool, Environment, Result or sequence, optional
        The data-source credit footnote of a standalone figure (see above).
    show_colorbar : bool, optional
        Draw the concentration colorbar. Default True.
    """
    from uacpy.data.seaice_local import sea_ice_pixel
    if hemi not in _SEA_ICE_HEMISPHERES:
        raise ConfigurationError(
            f"plot_sea_ice_map: hemi={hemi!r} is not a hemisphere of the NSIDC "
            f"grid; pass one of {_SEA_ICE_HEMISPHERES}.")
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    cm = plt.get_cmap(cmap).with_extremes(bad='#b8a98f')   # land / no-data
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cm, vmin=0.0, vmax=1.0,
                   origin='upper')
    # On the image's own pixel coordinates (column, row), which the
    # graticule, the transect and the zoom also use.
    concentration = np.asarray(grid, dtype=float)
    _draw_contours(ax, np.arange(concentration.shape[1]),
                   np.arange(concentration.shape[0]), concentration, contours,
                   who='plot_sea_ice_map', fmt='%g', **_SEA_ICE_CONTOUR_STYLE)

    if graticule:                                  # labelled lat/lon graticule
        sgn = 1 if hemi == 'N' else -1
        suffix = 'N' if hemi == 'N' else 'S'

        def _line(pts):
            pts = [p for p in pts if p is not None]
            if len(pts) > 1:
                ax.plot([c for _, c in pts], [r for r, _ in pts], color='0.4',
                        lw=0.6, alpha=0.45, zorder=2)
        # Builds the graticule grids as plain floats rather than the numpy
        # scalars ``arange`` yields, because they are fed to
        # ``sea_ice_pixel``, whose ``Coordinate`` is a ``tuple[float, float]``.
        # ``np.float64`` subclasses ``float``, so this is what ``py.typed``
        # promises a caller's checker rather than what the values do.
        lon_s = [float(lon) for lon in np.arange(-180.0, 181.0, 2.0)]
        for lat_c in [float(lat)                               # parallels
                      for lat in np.arange(50.0, 90.0, graticule) * sgn]:
            _line([sea_ice_pixel((lat_c, lo), hemi=hemi) for lo in lon_s])
            p0 = sea_ice_pixel((lat_c, 0.0), hemi=hemi)
            if p0:
                ax.annotate(f"{abs(int(lat_c))}°{suffix}", (p0[1], p0[0]),
                            color='0.25', fontsize='x-small', ha='center', va='center',
                            zorder=2, bbox=dict(boxstyle='round,pad=0.1',
                                                fc='white', ec='none', alpha=0.6))
        lat_s = [float(lat) for lat in np.arange(50.0, 89.0, 1.0) * sgn]
        for lon_c in [float(lon)
                      for lon in np.arange(-180.0, 180.0, 30.0)]:  # meridians
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
    if source is not None:
        ps = sea_ice_pixel(source, hemi=hemi)
        if ps:
            ax.plot(ps[1], ps[0], zorder=7, **SOURCE_MARKER_STYLE)

    # Frame a regional window around the transect/source (square pixels kept, the
    # window widened to the panel) so the map fills the axis like the depth map,
    # rather than showing the whole hemisphere with the region a speck. In grid
    # pixels (25 km on the NSIDC grid): the half-height is _ZOOM_SPAN_PAD times
    # the focus span (never under _ZOOM_MIN_SPAN_PX) — the span plus a fifth
    # of it on each side — plus a fixed _ZOOM_MARGIN_PX (400 km) of context,
    # and the half-width is _ZOOM_WIDTH_RATIO times the half-height.
    focus = [p for p in (pa, pb, ps) if p]
    if zoom and focus:
        rr = [p[0] for p in focus]
        cc = [p[1] for p in focus]
        cy, cx = 0.5 * (min(rr) + max(rr)), 0.5 * (min(cc) + max(cc))
        half = (max(max(rr) - min(rr), max(cc) - min(cc), _ZOOM_MIN_SPAN_PX)
                * _ZOOM_SPAN_PAD + _ZOOM_MARGIN_PX)
        ny, nx = np.shape(grid)
        ax.set_xlim(max(cx - _ZOOM_WIDTH_RATIO * half, -0.5),
                    min(cx + _ZOOM_WIDTH_RATIO * half, nx - 0.5))
        ax.set_ylim(min(cy + half, ny - 0.5), max(cy - half, -0.5))  # origin='upper'

    if pa and pb:
        # After the zoom, so the label offsets are measured on the final
        # window: the same inside-the-axes, beside-the-line placement
        # plot_bathymetry_map gives its A/B labels.
        for lbl, here, other in (('A', pa, pb), ('B', pb, pa)):
            dx, dy = _endpoint_label_offset(ax, (here[1], here[0]),
                                            (other[1], other[0]))
            ax.annotate(lbl, (here[1], here[0]), color='crimson',
                        fontweight='bold', xytext=(dx, dy),
                        textcoords='offset points',
                        ha='left' if dx > 0 else 'right',
                        va='bottom' if dy > 0 else 'top', zorder=6,
                        path_effects=[_patheffects.withStroke(
                            linewidth=2.5, foreground='white')])
        # 'best' keeps the legend off the transect line and its end markers,
        # as on the bathymetry map; a fixed corner sits on a transect that
        # starts in it.
        ax.legend(loc='best')
    # A bar the height of the map: the image keeps square pixels, so its
    # drawn box is shorter than the slot whenever the window is wider than
    # the slot, and a bar sized from the slot runs about twice the map's
    # height. The bar takes its width from the slot, as every other
    # colorbar on this surface does, so a composite (plot_overview) keeps its
    # spacing; it is shrunk to the height the equal-aspect map will draw at.
    # ``fraction`` and ``pad`` are matplotlib's own defaults.
    slot = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    fraction, pad = 0.15, 0.05
    drawn_h = min(slot.height, slot.width * (1.0 - fraction - pad)
                  * abs(y1 - y0) / abs(x1 - x0))
    if show_colorbar:
        fig.colorbar(im, ax=ax, label=concentration_label, fraction=fraction,
                     pad=pad, shrink=drawn_h / slot.height)
    ax.set_title(_title_or(title, "Sea-ice concentration"), fontsize='large',
                 fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():                   # match the depth-map frame
        spine.set_edgecolor('0.2')
        spine.set_linewidth(1.0)
    if own_fig:
        from uacpy.data.sources import SOURCES
        credit = _credit_attributions(
            [SOURCES['seaice']] if show_data_credit is True else show_data_credit)
        fig.tight_layout(rect=(0, 0.05, 1, 1) if credit else (0, 0, 1, 1))
        _draw_credit(fig, credit, reserve=False)
    return fig, ax


def _draw_depth(ax, lons, lats, depth, cmap, relief, exag, zorder):
    """Draw the depth field as shaded relief (``relief``) or flat ``pcolormesh``.

    Shaded relief reveals real seafloor texture (slopes, canyons, seamounts) and
    avoids the smooth-"dome" look of a flat sequential colormap on bowl-shaped
    bathymetry. Returns a mappable for the colorbar. Land (NaN) is transparent.
    """
    dm = np.ma.masked_invalid(depth)
    finite = np.asarray(depth, dtype=float)[np.isfinite(depth)]
    # Hillshading takes ``np.gradient`` of the grid (matplotlib
    # ``LightSource.hillshade``), which needs at least two samples along each
    # axis and something to differentiate. Fall back to the flat mesh for a
    # grid too small or too flat to shade, rather than shading a degenerate one.
    if (not relief or finite.size < 4 or min(np.shape(depth)) < 3
            or np.ptp(finite) < 1e-9):
        return ax.pcolormesh(lons, lats, dm, cmap=cmap, shading='auto',
                             zorder=zorder)

    base = plt.get_cmap(cmap)
    norm = _mcolors.Normalize(float(finite.min()), float(finite.max()))
    # Cell size in metres, which the hillshade needs to turn ``vert_exag`` into
    # a real slope: n samples span n-1 intervals. 111320 m is one degree of arc
    # at the equator (40 075 km / 360); longitude degrees shrink by cos(lat).
    dy = (lats.max() - lats.min()) / (lats.size - 1) * 111320.0
    dx = ((lons.max() - lons.min()) / (lons.size - 1) * 111320.0
          * np.cos(np.radians(float(lats.mean()))))
    # Land holes take the shallowest depth, so they are the high ground of the
    # relief and the coastline does not shade as a chasm. Their colour is
    # irrelevant — they end up transparent.
    filled = np.where(np.isnan(depth), norm.vmin, depth)
    nan_mask = np.isnan(depth)
    # imshow's extent below is anchored to (min-lat, min-lon) with
    # origin='lower', so the array must run south→north / west→east. Flip
    # any descending input axis to that canonical order — otherwise a
    # descending lat_range/lon_range renders the relief mirrored relative to
    # the flat pcolormesh path (which follows the real coordinate order).
    if lats.size > 1 and lats[-1] < lats[0]:
        filled = filled[::-1, :]
        nan_mask = nan_mask[::-1, :]
    if lons.size > 1 and lons[-1] < lons[0]:
        filled = filled[:, ::-1]
        nan_mask = nan_mask[:, ::-1]
    # Colour by depth, but light the *relief*: the surface handed to the shader
    # is seafloor height (-depth), or a seamount reads as a pit. And
    # ``LightSource.hillshade`` negates dy ("dy is implicitly negative",
    # matplotlib ``colors.py``) because it assumes row 0 is the top of the
    # image, while ``filled`` runs south→north under origin='lower' — so -dy
    # cancels that and azdeg means what it says. Measured on a Gaussian
    # seamount, azdeg=315 lights the flank bearing 315 deg.
    rgb = base(norm(filled))
    shaded = _mcolors.LightSource(azdeg=315, altdeg=45).shade_rgb(
        rgb, -filled, blend_mode='soft', vert_exag=exag, dx=dx, dy=-dy)
    rgb[..., :3] = shaded[..., :3]
    rgb[nan_mask, 3] = 0.0                              # land → transparent
    # Pad to the cell EDGES: imshow stretches the array onto the outer edges of
    # extent, so passing the first/last centres contracts the relief by
    # (N-1)/N about the grid centre and it stops registering with the
    # graticule, the contours, the coastline and the markers — all of which use
    # the true coordinates, as does the flat pcolormesh branch above.
    ax.imshow(rgb, extent=_cell_edge_extent(lons, lats),
              origin='lower', zorder=zorder, interpolation='nearest',
              aspect='auto')
    return plt.cm.ScalarMappable(norm=norm, cmap=base)


def _draw_graticule(ax, lon_range, lat_range, major, minor, proj):
    """Labelled lat/lon graticule on a (possibly projected) map axis.

    ``major`` is the labelled spacing and ``minor`` the fine unlabelled one.
    ``None`` on either drops that layer; ``None`` on ``major`` also drops the
    tick labels, since they are the major ticks' own.
    """
    def seq(rng, step):
        return np.arange(np.ceil(rng[0] / step) * step, rng[1] + 1e-6, step)

    def fmt(v, pos_hemi, neg_hemi):
        return f"{abs(v):g}°{pos_hemi if v >= 0 else neg_hemi}"

    if major:
        lon_maj, lat_maj = seq(lon_range, major), seq(lat_range, major)
        ax.set_xticks([proj(0.0, lo)[0] for lo in lon_maj])
        ax.set_xticklabels([_lon_tick_label(lo) for lo in lon_maj])
        ax.set_yticks([proj(la, 0.0)[1] for la in lat_maj])
        ax.set_yticklabels([fmt(la, 'N', 'S') for la in lat_maj])
    else:
        # Matplotlib's own auto-ticks would put unlabelled plain degrees where
        # the caller asked for no graticule, so clear them instead.
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_axisbelow(False)                             # graticule on top of the map
    if minor:
        ax.set_xticks([proj(0.0, lo)[0] for lo in seq(lon_range, minor)], minor=True)
        ax.set_yticks([proj(la, 0.0)[1] for la in seq(lat_range, minor)], minor=True)
        ax.grid(which='minor', color='1.0', lw=0.3, alpha=0.25, zorder=4)
    ax.grid(which='major', color='1.0', lw=0.6, alpha=0.45, zorder=4)
    ax.tick_params(which='major', length=5, direction='out', labelsize='small')
    ax.tick_params(which='minor', length=2.5, direction='out')
    for spine in ax.spines.values():
        spine.set_edgecolor('0.2')
        spine.set_linewidth(1.0)
