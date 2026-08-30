"""Shared plotting primitives: z-orders, axis labels, value extraction, seafloor overlay, data-credit and sea-ice helpers."""

from __future__ import annotations


import functools
import warnings
import numpy as np
from typing import Tuple

from uacpy.core.environment import Environment
from uacpy.core.exceptions import ConfigurationError
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.results import Field
from uacpy.core.results.quantities import label as quantity_label
from uacpy.core.units import m_to_km
from uacpy.visualization.style import (
    BOTTOM_FILL_STYLE_SOLID, BOTTOM_LINE_STYLE, BOTTOM_LINE_STYLE_FLAT,
    RECEIVER_MARKER_STYLE,
)


def _close_figures_since(before) -> None:
    """Close every pyplot figure opened after the ``before`` snapshot."""
    import matplotlib.pyplot as plt
    for num in set(plt.get_fignums()) - before:
        plt.close(num)


def typed_plot_error(plotter):
    """Decorator: surface a plotter's raw degenerate-input exceptions as a typed
    :class:`~uacpy.core.exceptions.ConfigurationError`, leaving no figure behind.

    Many plotters pass arrays straight to matplotlib (or index ``[0]``/``[-1]``
    for axis limits, subscript result dicts like an arrival's ``['delay']``,
    or reduce with ``.max()``), so empty / mismatched-length / out-of-range /
    wrong-shape / missing-key input leaks a bare
    ``IndexError``/``KeyError``/``ValueError`` instead of the typed error the
    result-consuming plotters raise. This converts those (and only those) into
    a clear ``ConfigurationError`` while letting an already-typed
    ``ConfigurationError`` pass through unchanged. ``TypeError`` is
    deliberately not caught — a genuine wrong-type call should surface as
    itself, not be relabelled an input error.

    Several plotters build their figure before the operation that can raise, so
    any figure opened during a failed call is closed before the exception
    propagates — a rejected call leaves pyplot's registry exactly as it found
    it."""
    @functools.wraps(plotter)
    def wrapper(*args, **kwargs):
        import matplotlib.pyplot as plt
        before = set(plt.get_fignums())
        try:
            return plotter(*args, **kwargs)
        except (IndexError, KeyError, ValueError) as exc:
            _close_figures_since(before)
            raise ConfigurationError(
                f"{plotter.__name__}: invalid plot input "
                f"({type(exc).__name__}: {exc}). Check the arrays are non-empty "
                f"and their lengths/shapes match the plotter's expected inputs."
            ) from exc
        except Exception:
            _close_figures_since(before)
            raise
    return wrapper


def _plot_warn(message, category=UserWarning) -> None:
    """Warn from inside a plotter, attributed to the **user's** call line.

    Every plotter is decorated, so a raw ``warnings.warn`` lands one frame
    short and blames this module: the user is told to change a knob and handed
    a line in uacpy, and a ``-W`` filter keyed on their own module never
    matches.

    Attribution walks out to the first frame outside the package rather than
    counting frames, so a plotter, a helper the plotter calls, and a public
    function reached without any plotter at all (:func:`land_polygons`) all
    name the user's own line. Counting could not cover the last of those: the
    count that suited the decorated plotter chain overshot the direct call by
    the two frames the chain contributes (measured)."""
    warnings.warn(message, category, skip_file_prefixes=USER_FRAME_SKIP)


def fig_ax(ax, figsize):
    """Return ``(fig, ax)``: a fresh figure of ``figsize`` when ``ax is None``,
    else the supplied axis and its parent figure. Shared by the DSP / comms /
    noise plotters so they all honour the ``ax=None`` convention identically."""
    import matplotlib.pyplot as plt
    if ax is None:
        return plt.subplots(figsize=figsize)
    return ax.figure, ax


def invert_yaxis_once(ax) -> None:
    """Point the y axis downward (depth, or TL in dB), idempotently.

    ``Axes.invert_yaxis`` toggles, so calling a plotter twice into the same
    ``ax=`` would flip the axis back to increasing-upward. Every plotter goes
    through here so overlays compose."""
    if not ax.yaxis_inverted():
        ax.invert_yaxis()


ZORDER_SEDIMENT = 2


ZORDER_RAYS = 2.5


ZORDER_SURFACE = 4
# Above the bottom rendering (fill/markers ≤ 7, seafloor line = 10) so the
# source/receiver geometry is never occluded by the seabed.


ZORDER_RECEIVERS = 11


ZORDER_SOURCE = 12


# Dense receiver grids render as solid bars — decimate each axis independently
# so the lattice stays readable. Range typically spans ~10x more samples than
# depth in a survey, so the two axes are capped differently. These are the caps
# for a full-page panel; a smaller one gets fewer dots (see
# :func:`_receiver_dot_caps`).
_MAX_RECEIVER_RANGE_DOTS = 20
_MAX_RECEIVER_DEPTH_DOTS = 10

# Marker widths kept between neighbouring lattice dots.
_RECEIVER_DOT_PITCH = 3.0


def _receiver_dot_caps(ax, markersize):
    """Per-axis dot caps for the receiver lattice drawn on ``ax``.

    A count-only cap is blind to how much room the panel actually has: 20 x 10
    dots read as a lattice on a full-page figure and as a wall of markers over
    the heatmap of a composite panel a fifth the size. Cap by the panel's own
    width and height instead, keeping neighbouring dots ``_RECEIVER_DOT_PITCH``
    marker widths apart, and never above the full-page counts."""
    bbox = ax.get_window_extent()
    px_per_point = ax.figure.dpi / 72.0
    pitch = max(markersize * _RECEIVER_DOT_PITCH * px_per_point, 1.0)
    return (
        max(2, min(_MAX_RECEIVER_RANGE_DOTS, int(bbox.width / pitch))),
        max(2, min(_MAX_RECEIVER_DEPTH_DOTS, int(bbox.height / pitch))),
    )


def _draw_geometry(ax, source=None, receiver=None, *, source_range_m=0.0,
                   max_markersize=8, source_markersize_bonus=0):
    """Draw the source and receiver markers on a (depth, range) cross-section.

    Shared by the environment, ray and field plotters so the geometry reads
    identically wherever it is shown. ``source`` is a :class:`Source` (or any
    object exposing ``depths``) or a bare array of source depths;
    ``source_range_m`` is where the source sits — 0 by the package convention
    that range is measured from it. ``receiver`` is decimated by
    :func:`_draw_receiver_grid`."""
    from uacpy.visualization.style import SOURCE_MARKER_STYLE
    if receiver is not None and getattr(receiver, 'depths', None) is not None:
        _draw_receiver_grid(ax, receiver.ranges, receiver.depths,
                            max_markersize=max_markersize)
    source_depths = getattr(source, 'depths', source)
    if source_depths is not None and np.size(source_depths):
        style = dict(SOURCE_MARKER_STYLE)
        if source_markersize_bonus:
            style['markersize'] = (style.get('markersize', 15)
                                   + source_markersize_bonus)
        x = m_to_km(np.atleast_1d(source_range_m))[0]
        for sd in np.atleast_1d(source_depths):
            ax.plot([x], [float(sd)], zorder=ZORDER_SOURCE, **style)
        # Models exclude the singular near field, so a TL grid usually starts
        # beyond r = 0 while the source sits at it. Widen the axis to keep the
        # marker on screen rather than clipping it to the spine.
        x_lo, x_hi = ax.get_xlim()
        if not (min(x_lo, x_hi) <= x <= max(x_lo, x_hi)):
            ax.set_xlim(min(x_lo, x_hi, x), max(x_lo, x_hi, x))


def _draw_receiver_grid(ax, ranges_m, depths, *, max_markersize,
                        zorder=ZORDER_RECEIVERS):
    """Draw the decimated receiver lattice; return the full range axis in km.

    Markers keep default clipping so a later user zoom hides out-of-view
    receivers instead of painting them across the figure."""
    rr_km = m_to_km(np.atleast_1d(ranges_m))
    rd = np.atleast_1d(depths)
    style = dict(RECEIVER_MARKER_STYLE)
    style['markersize'] = min(style.get('markersize', 8), max_markersize)
    max_r, max_d = _receiver_dot_caps(ax, style['markersize'])
    step_r = max(1, rr_km.size // max_r)
    step_d = max(1, rd.size // max_d)
    RR, RD = np.meshgrid(rr_km[::step_r], rd[::step_d])
    ax.plot(RR.ravel(), RD.ravel(), zorder=zorder, **style)
    return rr_km


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


def _db_label(field: Field) -> str:
    """Axis label for the ``value='db'`` view of ``field``.

    ``value`` is the caller's choice of *view*; ``field.kind`` is what the data
    *is*. They are independent — one complex field renders every view — but the
    label has to come from the field, or the dB view of a signal-excess grid
    announces itself as transmission loss. Split out from :func:`_value_array`
    so a caller that wants only the label does not build the array to get it."""
    return quantity_label(field.kind, 'dB' if field.is_complex else field.unit)


# Axis / colorbar label per ``value`` view, for the views whose label is fixed.
# ``'db'`` is missing on purpose: its label comes from the field (see
# :func:`_db_label`), because the dB view of a signal-excess grid is not TL.
_VALUE_LABELS = {
    'mag_db': '|H| (dB)',
    'mag': '|p|',
    'phase': 'Phase (rad)',
    'real': 'Re(p)',
    'imag': 'Im(p)',
}


def _default_value(field: Field) -> str:
    """The ``value`` view rendered when the caller names none.

    A time trace is linear pressure, not a level, so it defaults to the raw
    samples; everything else defaults to its dB view. Shared so a panel drawn
    inside a composite figure labels itself with the same view ``plot_field``
    would have picked on its own."""
    return 'real' if 'time' in field.coords else 'db'


def _value_label(field: Field, value: str) -> str:
    """Axis / colorbar label for the ``value`` view of ``field``.

    Split from :func:`_value_array` for callers that label a panel someone else
    drew (a composite figure's shared colorbar) and so must not build the array
    a second time to read its label."""
    if value == 'db':
        return _db_label(field)
    try:
        return _VALUE_LABELS[value]
    except KeyError:
        raise ConfigurationError(
            f"plot_field: unknown value={value!r}; "
            "valid: 'db', 'mag_db', 'mag', 'phase', 'real', 'imag'"
        ) from None


def _value_array(field: Field, value: str) -> Tuple[np.ndarray, str]:
    """Return ``(array, axis_label)`` for ``value`` ∈ ``{'db', 'mag_db',
    'mag', 'phase', 'real', 'imag'}``."""
    label = _value_label(field, value)
    if value == 'db':
        return field.db, label
    if value == 'mag_db':
        # Modulus in dB: 20·log10|H| = −TL (shares the floored dB conversion).
        return -field.db, label
    if value in ('mag', 'phase'):
        if not field.is_complex:
            raise ConfigurationError(
                f"plot_field: value={value!r} requires complex data")
        return (field.magnitude if value == 'mag' else field.phase), label
    if value == 'real':
        return (field.data.real if field.is_complex else field.data), label
    if not field.is_complex:
        raise ConfigurationError("plot_field: value='imag' requires complex data")
    return field.data.imag, label


def _coord_label(name: str) -> str:
    label, unit = _AXIS_LABELS.get(name, (name, ''))
    return f"{label} ({unit})" if unit else label


def _coord_axis(coord: np.ndarray, name: str) -> Tuple[np.ndarray, str]:
    """``(plot_values, axis_label)`` for a coordinate axis, converting a
    ``range`` axis from metres to km so 1-D cuts, 2-D heatmaps and ``compare``
    all share one x-scale."""
    if name == 'range':
        return m_to_km(coord), 'Range (km)'
    return np.asarray(coord), _coord_label(name)


# Fixed TL colour scale used everywhere TL is drawn: ``vmin = 20 dB`` →
# ``vmax = 120 dB``. A fixed scale keeps TL panels directly comparable across
# models / frequencies / runs. No-data cells (e.g. Bellhop cells no ray
# reached) are NaN and render as the axes background.
_TL_LIMITS: Tuple[float, float] = (20.0, 120.0)


def _is_transmission_loss(field: Field, value: str) -> bool:
    """Whether the ``value`` view of ``field`` is transmission loss.

    TL is the one quantity that runs backwards — it is a *loss*, so the least
    of it is the loudest arrival — which is why a 1-D TL cut is drawn with its
    value axis increasing DOWNWARD, putting the loud end at the top. Every
    other dB view is a **level** (signal excess, reverberation, ``mag_db``) and
    more of a level is more, so it reads upward like any other quantity.

    Identifying TL takes both the field and the view, exactly as
    :meth:`Field.max` documents: the dB view of a *pressure* field, whether
    stored as a real dB grid or derived from complex pressure. Every entry
    point that draws a value axis asks here, so one field cuts the same way
    through :func:`plot_field` and :func:`compare` alike."""
    return (value == 'db' and field.kind == 'pressure'
            and (field.is_complex or field.unit == 'dB'))


def _cell_edge_extent(x: np.ndarray, y: np.ndarray):
    """Edge-aligned ``imshow`` extent for centre-sampled ``x``/``y`` axes.

    ``imshow`` stretches the array onto the OUTER edges of ``extent``, so
    passing the first and last *centres* contracts the field by ``(N-1)/N``
    about its middle — zero error at the centre, half a rendered pixel at the
    edges, which is where every overlay (graticule, contours, markers) then
    disagrees with the data. Pad by half a cell on each side instead.

    Returns ``(left, right, bottom, top)`` for ``origin='lower'``, always
    ASCENDING: the axes are canonicalised via min/max, so the caller must
    flip its data rows/columns to ascending order to match — exactly what
    ``maps.py`` does before calling (its ``[::-1]`` flips). Handing a
    descending axis here WITHOUT flipping the data mirrors the image.
    Assumes a uniform grid, which is ``imshow``'s own assumption anyway.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    hx = abs(x[1] - x[0]) / 2.0 if x.size > 1 else 0.5
    hy = abs(y[1] - y[0]) / 2.0 if y.size > 1 else 0.5
    return (x.min() - hx, x.max() + hx, y.min() - hy, y.max() + hy)


def _flip_y(extent):
    """Swap an extent's y bounds, for an ``origin='upper'`` axis whose ordinate
    increases downward (intercept time, depth)."""
    left, right, bottom, top = extent
    return (left, right, top, bottom)


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
    r_km = m_to_km(ranges_m)
    z = np.asarray(depths, dtype=float)
    dr = (r_km[1] - r_km[0]) / 2.0 if r_km.size > 1 else 0.5
    dz = (z[1] - z[0]) / 2.0 if z.size > 1 else 0.5
    return (r_km[0] - dr, r_km[-1] + dr, z[-1] + dz, z[0] - dz)


def _overlay_seafloor(ax, env: Environment, ranges_m: np.ndarray) -> None:
    """Draw the seafloor on top of a (depth, range) heatmap.

    Uses high z-orders (sediment + 5, line + 6) so the bathymetry sits
    above contour lines and TL data — matches the original AT-style
    rendering. Bathymetry is clipped to the data x-range and anchored at both
    ends, and the y-axis is extended downward when the seafloor dips below the
    data extent so the sediment fill stays visible."""
    if env is None:
        return
    data_r_km = m_to_km(ranges_m)
    if data_r_km.size:
        x_lo, x_hi = float(data_r_km.min()), float(data_r_km.max())
    else:
        x_lo, x_hi = ax.get_xlim()
    if x_hi <= x_lo:
        return  # nothing to overlay on a zero-width axis
    ax.set_xlim(x_lo, x_hi)

    if env.has_range_dependent_bathymetry:
        r_km = m_to_km(env.bathymetry.ranges)
        z = env.bathymetry.depths
        # Runs whichever way the two spans differ. A bathymetry NARROWER than
        # the field needs the end anchors just as much as a wider one needs the
        # clip: without them the fill stops at the last bathymetry sample and
        # the panel shows a water column with no seabed under it, while the
        # model held that depth out to the end of the field. ``np.interp``
        # clamps outside the profile, so the anchors continue the end value —
        # the same constant extension the models apply. A no-op when the
        # bathymetry already spans the field exactly.
        if r_km.size >= 2:
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
                        zorder=ZORDER_SEDIMENT + 5, **BOTTOM_FILL_STYLE_SOLID)
        ax.plot(r_km, z, zorder=ZORDER_SEDIMENT + 6, **BOTTOM_LINE_STYLE)
    else:
        depth_max = max(max(ax.get_ylim()), env.depth * 1.05)
        if depth_max > max(ax.get_ylim()):
            ax.set_ylim(depth_max, min(ax.get_ylim()))
        ax.fill_between(
            data_r_km, env.depth, depth_max,
            zorder=ZORDER_SEDIMENT + 5, **BOTTOM_FILL_STYLE_SOLID,
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
        if name == 'range':
            # Range axes are drawn in km everywhere (see _coord_axis), so a
            # range pin reads in km too.
            parts.append(f"{label} = {m_to_km(v):.3g} km")
        elif unit == 'Hz' and abs(v) >= 1000:
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
        # An item is a DataProvenance (→ its .source), a bare DataSource, or a
        # plain string the caller passed.
        src = getattr(s, 'source', s)
        attr = (getattr(src, 'attribution', None)
                or (src if isinstance(src, str) else getattr(src, 'name', None)))
        if attr and attr not in seen:
            seen.add(attr)
            out.append(attr)
    return out


def _model_attribution(result):
    """One-line model credit ``"<model> — <author>, <engine>"`` from a result's
    :class:`~uacpy.models.sources.ModelSource`, or ``None`` when the result
    carries no model provenance. The model-side counterpart of the data-source
    attributions resolved by :func:`_credit_attributions`."""
    src = getattr(result, 'model_source', None)
    if src is None:
        return None
    name = getattr(result, 'model', '') or src.name
    return f"{name} — {src.attribution}"


def _credit_lines(data_attributions, model_attribution):
    """Compose the footnote rows from data + model provenance.

    One citation per line, stacked. Each group (``Data`` / ``Model``) is
    labelled on its first line; further lines align under it. One harmonised
    layout for environment plots (data only), result plots (model, plus data
    when an env is supplied) and maps.
    """
    groups = []
    if data_attributions:
        groups.append(("Data:", list(data_attributions)))
    if model_attribution:
        # A single attribution string, or a list of them (multi-model
        # comparison figures).
        lines = ([model_attribution] if isinstance(model_attribution, str)
                 else list(model_attribution))
        groups.append(("Model:", lines))
    if not groups:
        return []
    width = max(len(label) for label, _ in groups)
    indent = " " * (width + 1)
    lines = []
    for label, items in groups:
        lines.append(f"{label.ljust(width)} {items[0]}")
        lines.extend(f"{indent}{item}" for item in items[1:])
    return lines


def _draw_credit(fig, data_attributions=(), *, model=None,
                 center_ax=None, reserve=True):
    """Discreet grey provenance footnote along the figure bottom.

    Draws the licence-**required attribution** for the data **and** the model
    that produced the figure — the way a scientific figure credits its sources.
    Centred under ``center_ax`` when given (e.g. the map panel), else
    bottom-left. ``reserve`` adds bottom margin (set ``False`` when the caller
    already reserved space via ``tight_layout``).
    """
    lines = _credit_lines(data_attributions, model)
    if not lines:
        return
    if reserve:                                        # one line reserved per row
        fig.subplots_adjust(bottom=max(0.06 + 0.025 * len(lines),
                                       fig.subplotpars.bottom))
    if center_ax is not None:
        pos = center_ax.get_position()
        x, ha = pos.x0 + pos.width / 2.0, 'center'
    else:
        x, ha = 0.012, 'left'
    fig.text(x, 0.012, "\n".join(lines), ha=ha, va='bottom',
             fontsize=7, color='0.45', linespacing=1.5)


def _draw_result_credit(fig, result, *, env=None, data_source=True, **draw_kw):
    """Unified provenance footnote for a *result* figure: the model that
    produced it (always, when known) plus any data sources from ``env``.

    The single call every result plotter makes — keeps data + model credit
    rendering identical across :func:`plot_field`, `_plot_rays`, … ."""
    data = _credit_attributions(data_source, carrier=env)
    _draw_credit(fig, data, model=_model_attribution(result), **draw_kw)


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
    # An all-NaN concentration track means no drawable ice; numpy reports
    # that nanmean through warnings.warn ("Mean of empty slice"), which the
    # isfinite guard already converts into the no-op return below.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice',
                                category=RuntimeWarning)
        mean_conc = np.nanmean(conc)
    if rngs.size < 2 or not np.isfinite(mean_conc) or mean_conc <= 0:
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
        rk = m_to_km(ranges)
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
    r_km = m_to_km(ranges)
    z = -np.asarray(heights, dtype=float)          # positive up → negative depth
    if not np.any(np.abs(z) > 0):
        return
    ax.plot(r_km, z, color='#1f6f9f', lw=1.6, zorder=ZORDER_SURFACE)
    ax.fill_between(r_km, 0.0, z, color='#1f6f9f', alpha=0.15,
                    zorder=ZORDER_SURFACE - 1)
    lo, hi = ax.get_ylim()
    if float(np.min(z)) < hi:                       # crest above z=0 → headroom
        ax.set_ylim(lo, float(np.min(z)) * 1.2)
