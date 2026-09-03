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


ZORDER_SURFACE = 4


# Rays draw over the seafloor fill (ZORDER_SEDIMENT + 5 = 7) and its line
# (+ 6 = 8): the direct eigenray of a bottom-mounted link runs a few metres
# above the seabed, inside the drawn line's own width, and is painted over
# when it sits below that line. They stay under the receiver/source markers.
ZORDER_RAYS = 9


# Above the bottom rendering (fill/markers ≤ 7, seafloor line 8 in the
# ray/field overlay and 10 in the bathymetry panel) and the rays, so the
# source/receiver geometry is never occluded.
ZORDER_RECEIVERS = 11


ZORDER_SOURCE = 12


# Above every drawn artist: a legend explains the picture, so nothing in
# it may cover the key. Matplotlib's own default is 5, which sits under
# the seabed fill, the seafloor line, the receivers and the source.
ZORDER_LEGEND = 13


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


def _marker_half_width_in_data(ax, markersize_pt):
    """Half a marker's width, in the x data units of ``ax``.

    ``markersize`` is in points, so how much room a marker needs at the edge of
    an axes depends on the figure's size and dpi, not on the data span. Padding
    an axis limit by a fixed fraction of the span therefore clips the marker on
    a narrow panel and over-pads a wide one."""
    half_px = 0.5 * float(markersize_pt) * ax.figure.dpi / 72.0
    inv = ax.transData.inverted()
    return abs(inv.transform((half_px, 0.0))[0] - inv.transform((0.0, 0.0))[0])


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
        lo, hi = min(x_lo, x_hi), max(x_lo, x_hi)
        if not (lo <= x <= hi):
            # Widening to EXACTLY the marker's x centres it ON the spine, and
            # markers keep matplotlib's default clipping for the reason
            # _draw_receiver_grid's docstring gives (a later zoom must hide
            # out-of-view markers), so half the marker is cut away — visible
            # on the source star of docs/guide/figures/plot_overlays.png.
            # Pad the side that moved by the marker's own half width, which is
            # what it actually needs: a fixed fraction of the span is a
            # different number of points on every figure size, and 1 % still
            # clipped the star on a 3-inch panel.
            ax.set_xlim(min(lo, x), max(hi, x))
            pad = _marker_half_width_in_data(ax, style.get('markersize', 15))
            new_lo, new_hi = min(lo, x), max(hi, x)
            if x < lo:
                new_lo -= pad
            if x > hi:
                new_hi += pad
            ax.set_xlim(new_lo, new_hi)


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


def _require_nonempty(caller: str, **arrays) -> None:
    """Refuse an empty input array by name: an empty panel with axes and a
    title reads as a result, and the guide promises degenerate input raises."""
    for name, value in arrays.items():
        if np.size(value) == 0:
            raise ConfigurationError(
                f"{caller}: {name} is empty; there is nothing to draw.")


def _default_value(field: Field) -> str:
    """The ``value`` view rendered when the caller names none.

    A time trace is linear pressure, not a level, so it defaults to the raw
    samples; so does a real-valued field that is not a level and has no dB
    view (a detection probability, unit '1'). Everything else defaults to its
    dB view. Shared so a panel drawn inside a composite figure labels itself
    with the same view ``plot_field`` would have picked on its own."""
    if 'time' in field.coords:
        return 'real'
    if not field.is_complex and getattr(field, 'unit', None) == '1':
        return 'real'
    return 'db'


def _value_label(field: Field, value: str) -> str:
    """Axis / colorbar label for the ``value`` view of ``field``.

    Split from :func:`_value_array` for callers that label a panel someone else
    drew (a composite figure's shared colorbar) and so must not build the array
    a second time to read its label."""
    if value == 'db':
        return _db_label(field)
    if value == 'real':
        # A time trace is p(t); a real field that is not pressure (a
        # probability, a signal excess) is named by what it is — 'Re(p)'
        # belongs to the real part of a complex pressure only.
        if 'time' in field.coords:
            return 'p(t)'
        if not field.is_complex and getattr(field, 'kind', 'pressure') != 'pressure':
            # From the registry, the same source :func:`_db_label` reads, not
            # from the tag spelling: mangling ``kind`` produced 'probability
            # of detection' where the dedicated plotter's colorbar and the
            # registry both say 'Probability of detection'.
            return quantity_label(field.kind, field.unit)
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
        # Complex only, for the same reason 'mag' is: ``.db`` negates the
        # modulus of COMPLEX data, but hands back real data untouched
        # because it is already a level — so negating that flips a level
        # rather than converting one. On signal excess, where the sign is
        # the meaning, -20 dB (undetectable) plotted as +20.
        if not field.is_complex:
            raise ConfigurationError(
                f"plot_field: value={value!r} requires complex data; this "
                f"field is real and already a level, so its dB view is "
                f"value='db'.")
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


#: The kinds whose dB view is a LOSS rather than a level. Transmission loss
#: (``pressure`` in dB) is ``-20·log10|p|``; OASS reverberation is
#: ``-10·log10 E[|p_scat|²]`` (``third_party/oases/src/oassun26.f:633-637``
#: and ``:853-857``). Both carry the leading minus, so for both the LEAST of
#: the quantity is the loudest — the same pair :meth:`Field.max` documents.
_LOSS_KINDS = ('pressure', 'reverberation')


def _is_loss_view(field: Field, value: str) -> bool:
    """Whether the ``value`` view of ``field`` is a loss rather than a level.

    A loss runs backwards — the least of it is the loudest — which is why a
    1-D loss cut is drawn with its value axis increasing DOWNWARD, putting the
    loud end at the top. Two quantities read that way: transmission loss (the
    dB view of a ``pressure`` field) and OASS reverberation, whose stored
    numbers are a loss for the reason ``_LOSS_KINDS`` gives. Every other dB
    view is a **level** (signal excess, ``mag_db``) and more of a level is
    more, so it reads upward like any other quantity.

    Identifying a loss takes both the field and the view, exactly as
    :meth:`Field.max` documents: the dB view of a loss kind, whether stored as
    a real dB grid or derived from complex pressure. Every entry point that
    draws a value axis asks here, so one field cuts the same way through
    :func:`plot_field` and :func:`compare` alike."""
    return (value == 'db' and field.kind in _LOSS_KINDS
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
    """``imshow`` extent for a (depth, range) field: cell edges, range in km,
    depth increasing downward — :func:`_cell_edge_extent` flipped."""
    return _flip_y(_cell_edge_extent(m_to_km(np.asarray(ranges_m, dtype=float)),
                                     np.asarray(depths, dtype=float)))


def _sink_line_into_sediment(line) -> None:
    """Shift a seafloor line down by half its own width, in screen points,
    so the whole stroke lies on the sediment side of the boundary and its
    upper edge IS the boundary.

    A stroke centred on the seabed puts half its width into the water: 2 pt
    on a 1 km depth axis covers the bottom 3 m of the water column, which is
    where a bottom-mounted link's direct path runs, whatever z-order the rays
    are drawn at. Depth axes point downward, so "into the sediment" is
    screen-down, a negative display-y offset; fixed in points, it follows the
    line width through a zoom."""
    from matplotlib.transforms import ScaledTranslation
    half_inch = line.get_linewidth() / 2.0 / 72.0
    line.set_transform(line.get_transform() + ScaledTranslation(
        0.0, -half_inch, line.axes.figure.dpi_scale_trans))


def _overlay_seafloor(ax, env: Environment, ranges_m: np.ndarray) -> None:
    """Draw the seafloor on top of a (depth, range) heatmap.

    Uses high z-orders (sediment + 5, line + 6) so the bathymetry sits
    above contour lines and TL data — matches the original AT-style
    rendering. Bathymetry is clipped to the data x-range and anchored at both
    ends, and the y-axis is extended downward when the seafloor dips below the
    data extent so the sediment fill stays visible. The boundary stroke is
    sunk into the sediment (:func:`_sink_line_into_sediment`) so nothing in
    the water column — a ray skimming the bottom, the lowest field row — is
    covered by it."""
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
        (line,) = ax.plot(r_km, z, zorder=ZORDER_SEDIMENT + 6,
                          **BOTTOM_LINE_STYLE)
        _sink_line_into_sediment(line)
    else:
        depth_max = max(max(ax.get_ylim()), env.depth * 1.05)
        if depth_max > max(ax.get_ylim()):
            ax.set_ylim(depth_max, min(ax.get_ylim()))
        ax.fill_between(
            data_r_km, env.depth, depth_max,
            zorder=ZORDER_SEDIMENT + 5, **BOTTOM_FILL_STYLE_SOLID,
        )
        line = ax.axhline(env.depth, zorder=ZORDER_SEDIMENT + 6,
                          **BOTTOM_LINE_STYLE_FLAT)
        _sink_line_into_sediment(line)
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
    bottom-left. ``reserve`` raises the subplot bottom until every axes'
    tick and axis labels clear the footnote (set ``False`` when the caller
    already reserved space via ``tight_layout``).
    """
    lines = _credit_lines(data_attributions, model)
    if not lines:
        return
    if center_ax is not None:
        pos = center_ax.get_position()
        x, ha = pos.x0 + pos.width / 2.0, 'center'
    else:
        x, ha = 0.012, 'left'
    credit = fig.text(x, 0.012, "\n".join(lines), ha=ha, va='bottom',
                      fontsize=7, color='0.45', linespacing=1.5)
    if reserve:
        _reserve_credit_margin(fig, credit)


def _reserve_credit_margin(fig, credit, pad_px=4.0):
    """Raise the subplot bottom by the measured overlap between the credit
    and the lowest axis label, so the footnote never runs into an x-label.

    Measured, not a per-line constant: a fixed ``0.06 + 0.025/line`` margin
    left a one-line credit 7 px inside plot_overview's longitude label and a
    two-line one 22 px inside it.
    """
    renderer = fig.canvas.get_renderer()
    top = credit.get_window_extent(renderer).y1 + pad_px
    boxes = [ax.get_tightbbox(renderer) for ax in fig.axes if ax.get_visible()]
    lowest = min((b.y0 for b in boxes if b is not None), default=top)
    if top > lowest:
        fig.subplots_adjust(
            bottom=fig.subplotpars.bottom + (top - lowest) / fig.bbox.height)


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
                        linewidths=4.5, capstyle='round',
                        zorder=ZORDER_SURFACE + 0.5)   # over the ice band
    lc.set_array(0.5 * (conc[:-1] + conc[1:]))
    ax.add_collection(lc)
    # Its own label, below the ice-band's "ice", so the two never overprint.
    ax.text(0.985, 0.86, "sea-ice concentration", transform=ax.transAxes,
            ha='right', va='top', fontsize=8, style='italic', color='#5b1a8b',
            zorder=ZORDER_SOURCE)


def _draw_surface_boundary(ax, env):
    """Draw the top boundary from ``env.surface`` (a :class:`Surface` carrier).

    An elastic or non-vacuum half-space surface (e.g. an ice cover) rides the
    water surface as a hatched solid band; a vacuum / pressure-release surface
    leaves the plain free surface untouched. A range-dependent surface (a
    marginal ice zone) is drawn as per-range zones, mirroring the bottom.
    """
    surface = env.surface
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
    alti = env.altimetry
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
