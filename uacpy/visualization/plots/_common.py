"""Shared plotting primitives: z-orders, axis labels, value extraction, seafloor overlay, data-credit and sea-ice helpers."""

from __future__ import annotations


import numpy as np
from typing import Tuple

from uacpy.core.environment import Environment
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import Field
from uacpy.visualization.style import BOTTOM_FILL_STYLE, BOTTOM_LINE_STYLE, BOTTOM_LINE_STYLE_FLAT


ZORDER_SEDIMENT = 2


ZORDER_RAYS = 2.5


ZORDER_SURFACE = 4
# Above the bottom rendering (fill/markers ≤ 7, seafloor line = 10) so the
# source/receiver geometry is never occluded by the seabed.


ZORDER_RECEIVERS = 11


ZORDER_SOURCE = 12


# ─────────────────────────────────────────────────────────────────────────────
# Result dispatcher (used by Result.plot)
# ─────────────────────────────────────────────────────────────────────────────


_AXIS_LABELS = {
    'depth':        ('Depth', 'm'),
    'range':        ('Range', 'm'),
    'frequency':    ('Frequency', 'Hz'),
    'time':         ('Time', 's'),
    'source_depth': ('Source depth', 'm'),
}


def _value_array(field: Field, value: str) -> Tuple[np.ndarray, str]:
    """Return ``(array, axis_label)`` for ``value`` ∈ ``{'tl', 'mag',
    'phase', 'real', 'imag'}``."""
    if value == 'tl':
        return field.tl, 'TL (dB)'
    if value == 'mag':
        return field.magnitude, '|p|'
    if value == 'phase':
        return field.phase, 'Phase (rad)'
    if value == 'real':
        return field.data.real if field.is_complex else field.data, 'Re(p)'
    if value == 'imag':
        if not field.is_complex:
            raise ConfigurationError("plot_field: value='imag' requires complex data")
        return field.data.imag, 'Im(p)'
    raise ConfigurationError(
        f"plot_field: unknown value={value!r}; "
        "valid: 'tl', 'mag', 'phase', 'real', 'imag'"
    )


def _coord_label(name: str) -> str:
    label, unit = _AXIS_LABELS.get(name, (name, ''))
    return f"{label} ({unit})" if unit else label


def _coord_axis(coord: np.ndarray, name: str) -> Tuple[np.ndarray, str]:
    """``(plot_values, axis_label)`` for a coordinate axis, converting a
    ``range`` axis from metres to km so 1-D cuts, 2-D heatmaps and ``compare``
    all share one x-scale."""
    if name == 'range':
        return np.asarray(coord) / 1000.0, 'Range (km)'
    return np.asarray(coord), _coord_label(name)


# Fixed TL colour scale used everywhere TL is drawn: ``vmin = 20 dB`` →
# ``vmax = 120 dB``. A fixed scale keeps TL panels directly comparable across
# models / frequencies / runs (and the 600 dB no-data sentinel some AT binaries
# emit clips harmlessly to the top).
_TL_LIMITS: Tuple[float, float] = (20.0, 120.0)


def _imshow_extent(ranges_m: np.ndarray, depths: np.ndarray):
    """Edge-aligned ``imshow`` extent for center-sampled (range, depth) data.

    ``imshow`` stretches the array onto the OUTER edges of ``extent``, but
    model ranges/depths are cell CENTERS. Pad by half a cell on each side so
    every pixel centers on its coordinate — i.e. no half-cell shift versus the
    model grid, the ``pcolormesh`` field views, or the seafloor overlay.
    Returns ``(left_km, right_km, bottom, top)`` for ``origin='upper'`` (depth
    increasing downward). Assumes a uniform grid, which is ``imshow``'s own
    assumption anyway.
    """
    r_km = np.asarray(ranges_m, dtype=float) / 1000.0
    z = np.asarray(depths, dtype=float)
    dr = (r_km[1] - r_km[0]) / 2.0 if r_km.size > 1 else 0.5
    dz = (z[1] - z[0]) / 2.0 if z.size > 1 else 0.5
    return (r_km[0] - dr, r_km[-1] + dr, z[-1] + dz, z[0] - dz)


def _overlay_seafloor(ax, env: Environment, ranges_m: np.ndarray) -> None:
    """Draw the seafloor on top of a (depth, range) heatmap.

    Uses high z-orders (sediment + 5, line + 6) so the bathymetry sits
    above contour lines and TL data — matches the original AT-style
    rendering. Bathymetry is clipped to the data x-range, and the y-axis
    is extended downward when the seafloor dips below the data extent so
    the sediment fill stays visible."""
    if env is None:
        return
    data_r_km = np.asarray(ranges_m, dtype=float) / 1000.0
    if data_r_km.size:
        x_lo, x_hi = float(data_r_km.min()), float(data_r_km.max())
    else:
        x_lo, x_hi = ax.get_xlim()
    if x_hi <= x_lo:
        return  # nothing to overlay on a zero-width axis
    ax.set_xlim(x_lo, x_hi)

    if env.has_range_dependent_bathymetry():
        r_km = env.bathymetry.ranges / 1000.0
        z = env.bathymetry.depths
        if r_km.size >= 2 and (r_km.min() < x_lo or r_km.max() > x_hi):
            mask = (r_km >= x_lo) & (r_km <= x_hi)
            r_clip = list(r_km[mask])
            z_clip = list(z[mask])
            if not r_clip or r_clip[0] > x_lo:
                r_clip.insert(0, x_lo)
                z_clip.insert(0, float(np.interp(x_lo, r_km, z)))
            if r_clip[-1] < x_hi:
                r_clip.append(x_hi)
                z_clip.append(float(np.interp(x_hi, r_km, z)))
            r_km = np.array(r_clip)
            z = np.array(z_clip)
        max_seafloor = float(np.max(z))
        depth_max = max(max(ax.get_ylim()), max_seafloor * 1.05)
        if depth_max > max(ax.get_ylim()):
            ax.set_ylim(depth_max, min(ax.get_ylim()))
        ax.fill_between(r_km, z, depth_max,
                        zorder=ZORDER_SEDIMENT + 5, **BOTTOM_FILL_STYLE)
        ax.plot(r_km, z, zorder=ZORDER_SEDIMENT + 6, **BOTTOM_LINE_STYLE)
    else:
        depth_max = max(max(ax.get_ylim()), env.depth * 1.05)
        if depth_max > max(ax.get_ylim()):
            ax.set_ylim(depth_max, min(ax.get_ylim()))
        ax.fill_between(
            data_r_km, env.depth, depth_max,
            zorder=ZORDER_SEDIMENT + 5, **BOTTOM_FILL_STYLE,
        )
        ax.axhline(env.depth, zorder=ZORDER_SEDIMENT + 6,
                   **BOTTOM_LINE_STYLE_FLAT)
    ax.set_xlim(x_lo, x_hi)


def _pinned_subtitle(field: Field) -> str:
    if not field.pinned:
        return ''
    parts = []
    for name, v in field.pinned.items():
        label, unit = _AXIS_LABELS.get(name, (name, ''))
        if unit == 'Hz' and abs(v) >= 1000:
            parts.append(f"{label} = {v / 1000.0:.2f} kHz")
        else:
            parts.append(f"{label} = {v:.3g} {unit}".strip())
    return ", ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# animate_field — time-series Field → matplotlib FuncAnimation
# ─────────────────────────────────────────────────────────────────────────────


def _credit_attributions(data_source, *, carrier=None):
    """Resolve a ``data_source`` plotter argument to attribution strings.

    Accepts ``None`` / ``False`` (no credit), ``True`` (use ``carrier`` — the
    plot's own provenance object), an object with a ``.data_sources`` attribute
    (an ``Environment`` or ``Result``), an iterable of ``DataSource`` / str, or a
    single ``DataSource`` / str. Returns the de-duplicated attribution texts.
    """
    if not data_source:
        return []
    if data_source is True:                       # use the plot's own provenance
        items = list(getattr(carrier, 'data_sources', None) or [])
    else:
        items = getattr(data_source, 'data_sources', None)
        if items is None:                         # an explicit list / DataSource / str
            items = (list(data_source)
                     if isinstance(data_source, (list, tuple, set))
                     else [data_source])
    out, seen = [], set()
    for s in items:
        attr = (getattr(s, 'attribution', None)
                or (s if isinstance(s, str) else getattr(s, 'name', None)))
        if attr and attr not in seen:
            seen.add(attr)
            out.append(attr)
    return out


def _draw_data_credit(fig, attributions, *, center_ax=None, reserve=True):
    """Discreet grey data-source footnote along the figure bottom.

    The licence-**required attribution** for every source used — the way a
    scientific figure or map credits its data. Centred under ``center_ax`` when
    given (e.g. the map panel), else bottom-left. ``reserve`` adds bottom margin
    (set ``False`` when the caller already reserved space via ``tight_layout``).
    """
    if not attributions:
        return
    if reserve:                                        # one line reserved per source
        fig.subplots_adjust(bottom=max(0.06 + 0.025 * len(attributions),
                                       fig.subplotpars.bottom))
    if center_ax is not None:
        pos = center_ax.get_position()
        x, ha = pos.x0 + pos.width / 2.0, 'center'
    else:
        x, ha = 0.012, 'left'
    fig.text(x, 0.012, "\n".join(attributions), ha=ha, va='bottom',
             fontsize=7, color='0.45', linespacing=1.5)


def _draw_sea_ice(ax, sea_ice):
    """Sea-ice cover as a thick surface line coloured by concentration.

    ``sea_ice`` is a concentration 0–1 (uniform) or ``(ranges_km, concentration)``
    (**range-varying** — e.g. an ice edge). Drawn as one bold line riding the water
    surface, coloured **dark → violet** with the local concentration (dark = thin /
    open leads, bright violet = consolidated pack). Nothing is drawn for an
    ice-free section.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    if np.isscalar(sea_ice):
        x0, x1 = ax.get_xlim()
        rngs = np.array([x0, x1], dtype=float)
        conc = np.array([float(sea_ice)] * 2, dtype=float)
    else:
        rngs, conc = (np.asarray(a, dtype=float) for a in sea_ice)
    if rngs.size < 2 or not np.isfinite(np.nanmean(conc)) or np.nanmean(conc) <= 0:
        return
    lo, hi = ax.get_ylim()                             # inverted: surface = hi
    ax.set_ylim(lo, hi - 0.06 * (lo - hi))             # headroom for the line
    pts = np.column_stack([rngs, np.full(rngs.shape, hi)]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    ice_cmap = LinearSegmentedColormap.from_list('ice',
                                                 ['#05051a', '#3a0d63', '#7a2da8'])
    lc = LineCollection(segs, cmap=ice_cmap, norm=Normalize(0.0, 1.0),
                        linewidths=4.5, capstyle='round', zorder=ZORDER_SURFACE)
    lc.set_array(0.5 * (conc[:-1] + conc[1:]))
    ax.add_collection(lc)
    ax.text(0.985, 0.93, "ice", transform=ax.transAxes, ha='right', va='top',
            fontsize=8, style='italic', color='#5b1a8b', zorder=ZORDER_SOURCE)


def _draw_surface_boundary(ax, env):
    """Draw the top boundary from ``env.surface`` (a :class:`Surface` carrier).

    An elastic or non-vacuum half-space surface (e.g. an ice cover) rides the
    water surface as a hatched solid band; a vacuum / pressure-release surface
    leaves the plain free surface untouched. A range-dependent surface (a
    marginal ice zone) is drawn as per-range zones, mirroring the bottom.
    """
    surface = getattr(env, 'surface', None)
    props = getattr(surface, 'properties', None)
    if not props:
        return

    def _is_solid(bp):
        at = getattr(bp, 'acoustic_type', None)
        return (getattr(bp, 'shear_speed', 0.0) or 0.0) > 0 or \
            at == 'half-space'

    x0, x1 = ax.get_xlim()
    ranges = getattr(surface, 'ranges', None)
    if ranges is None or len(props) == 1:
        zones = [(x0, x1, props[0])]
    else:
        rk = np.asarray(ranges, dtype=float) / 1000.0
        bnds = [x0] + [0.5 * (rk[i] + rk[i + 1]) for i in range(len(rk) - 1)] + [x1]
        zones = [(bnds[i], bnds[i + 1], props[i]) for i in range(len(props))]

    lo, hi = ax.get_ylim()                  # inverted: hi = surface (depth 0)
    band = 0.045 * abs(lo - hi)
    drew = False
    for r_lo, r_hi, bp in zones:
        if not _is_solid(bp):
            continue
        ax.fill_between([r_lo, r_hi], hi, hi - band, color='#cfe6f2',
                        hatch='xx', edgecolor='#3a6e8f', linewidth=0.5,
                        zorder=ZORDER_SURFACE)
        drew = True
    if drew:
        ax.set_ylim(lo, hi - 1.25 * band)   # headroom above the surface
        ax.text(0.985, 0.93, "ice", transform=ax.transAxes, ha='right',
                va='top', fontsize=8, style='italic', color='#3a6e8f',
                zorder=ZORDER_SOURCE)


def _draw_altimetry(ax, env):
    """Draw the sea-surface *shape* from ``env.altimetry`` (an
    :class:`Altimetry` carrier): the height profile riding mean sea level
    (heights positive up → above z = 0). A flat / absent altimetry leaves the
    plain free surface.
    """
    alti = getattr(env, 'altimetry', None)
    ranges = getattr(alti, 'ranges', None)
    heights = getattr(alti, 'heights', None)
    if ranges is None or heights is None or np.asarray(ranges).size < 2:
        return
    r_km = np.asarray(ranges, dtype=float) / 1000.0
    z = -np.asarray(heights, dtype=float)          # positive up → negative depth
    if not np.any(np.abs(z) > 0):
        return
    ax.plot(r_km, z, color='#1f6f9f', lw=1.6, zorder=ZORDER_SURFACE)
    ax.fill_between(r_km, 0.0, z, color='#1f6f9f', alpha=0.15,
                    zorder=ZORDER_SURFACE - 1)
    lo, hi = ax.get_ylim()
    if float(np.min(z)) < hi:                       # crest above z=0 → headroom
        ax.set_ylim(lo, float(np.min(z)) * 1.2)
