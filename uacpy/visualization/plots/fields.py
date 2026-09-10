"""Field plotters: auto-shape heatmaps/line cuts, signal excess, detection probability, and the compare/compare_models overlays."""

from __future__ import annotations

import numpy as np
import matplotlib.collections as _mcoll
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple

from uacpy.core.environment import Environment
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import Field
from uacpy.visualization.style import (cmap_for_field, reversed_cmap,
                                      PROBABILITY_COLORMAP,
                                      PROBABILITY_LIMITS)
from uacpy.visualization.plots._common import _value_array, _value_label, _default_value, _coord_label, _coord_axis, _TL_LIMITS, _is_loss_view, _overlay_seafloor, _pinned_subtitle, _draw_result_credit, _draw_credit, _model_attribution, fig_ax, invert_yaxis_once, _draw_geometry, typed_plot_error, _plot_warn


# Which of ``plot_field``'s knobs each of its three render branches reads.
# Anything a branch does not read is rejected instead of silently dropped.
_HEATMAP_ONLY = ('vmin', 'vmax', 'cmap', 'show_colorbar', 'contours')
# The environment and the run geometry can only be drawn over a physical
# (depth, range) cross-section, so they key on the axes, not just the branch.
_CROSS_SECTION_ONLY = ('env', 'source', 'receiver')
_BRANCH_UNUSED = {
    'heatmap': ('label', 'stack_offset'),
    'line': _HEATMAP_ONLY + _CROSS_SECTION_ONLY + ('stack_offset',),
    'stacked': _HEATMAP_ONLY + _CROSS_SECTION_ONLY + ('label',),
}
_BRANCH_DESCRIPTION = {
    'heatmap': 'a 2-D heatmap',
    'line': 'a 1-D line cut',
    'stacked': 'the stacked-traces view',
    'other_heatmap': 'a heatmap that is not a (depth, range) cross-section',
}

# Contour-label unit, keyed by ``value``. Linear pressure ('mag', 'real',
# 'imag') carries no unit, so its labels are bare numbers.
_CONTOUR_FMT = {'dB': '%g dB', 'mag_dB': '%g dB', 'phase': '%g rad'}

# ``value`` modes that render a dB view of the quantity, and so take its dB
# colormap (``cmap_for_field(kind, dB=True)``). Every other mode is a linear
# view and shares one signed colormap — see ``style.LINEAR_VIEW_COLORMAP``.
# ``plot_field`` and ``compare_models`` both key on this, so one field renders
# the same through either entry point.
from uacpy.core.constants import PRESSURE_FLOOR

_DB_VALUES = ('dB', 'mag_dB')

#: Magnitude of the no-energy marker on a dB axis, 600. ``PRESSURE_FLOOR``
#: is what the package writes where a model reported no energy at all (as
#: against NaN, which is no data), so a sample of this size is a marker
#: rather than a level and takes no part in a colour limit. Its SIGN depends
#: on the view: ``dB`` is a loss and puts it at +600, ``mag_dB`` is
#: ``-field.dB`` and puts the same cell at -600, so the two are read by
#: magnitude and nothing real reaches it from either side.
_NO_ENERGY_DB = abs(20.0 * np.log10(PRESSURE_FLOOR))

# Fraction of its reference span that a length-1 axis's heatmap band occupies.
_SINGLETON_BAND_FRACTION = 0.02


@typed_plot_error
def plot_field(
    field: Field,
    ax=None,
    *,
    env: Optional[Environment] = None,
    source=None,
    receiver=None,
    value: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: Optional[str] = None,
    title: Optional[str] = None,
    label: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
    stacked: bool = False,
    stack_offset: Optional[float] = None,
    show_colorbar: Optional[bool] = None,
    contours: Optional[Sequence[float]] = None,
    **mpl_kw,
):
    """Auto-shape plotter for :class:`Field`.

    The shape is determined by what's in :attr:`Field.coords` after the
    user's :meth:`Field.at` / :meth:`Field.isel` calls:

    * 1 surviving axis → line plot.
    * 2 surviving axes → heatmap (the default), or a stacked-traces view
      when ``stacked=True`` and one axis is ``'time'``.

    Slice ``field`` before calling to control what gets plotted.

    Parameters
    ----------
    field : Field
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    env : Environment, optional
        Overlays the seafloor on a 2-D ``(depth, range)`` heatmap.
    source, receiver : Source / Receiver, optional
        Draw the run geometry over a 2-D ``(depth, range)`` heatmap — same
        markers ``Environment.plot`` and the ray plotter use.

        Any of these three keeps the ``(depth, range)`` plane a heatmap even
        where one axis holds a single sample (a single-receiver-depth run),
        rather than reducing it to the line cut it would otherwise become;
        that row is drawn as a band at its own coordinate.
    value : str
        ``'dB'``, ``'mag_dB'`` (``20·log10|H|``), ``'mag'``, ``'phase'``,
        ``'real'``, ``'imag'``. Defaults to ``'real'`` for a time-series
        field and ``'dB'`` otherwise.
    vmin, vmax : float, optional
        Colour limits (2-D heatmap only). What an unset limit falls back to
        depends on the quantity, since only some of them have a window that
        means something: a **pressure** field's ``value='dB'`` view takes the
        fixed 20–120 dB TL scale (``_TL_LIMITS``), so TL panels stay directly
        comparable across models, frequencies and runs; **signal excess**
        takes a window symmetric about its 0 dB detection boundary; a
        **probability** takes a fixed [0, 1]; and everything else, including
        reverberation and ``value='mag_dB'``, autoscales.
    cmap : str, optional
        Override the default colormap (2-D heatmap only).
    title : str, optional
    label : str, optional
        Legend label for the 1-D line cut.
    figsize : tuple
    stacked : bool
        Only valid on a 2-D field that carries a ``'time'`` axis. Plots
        each row of ``data`` as an offset trace stacked vertically — the
        classic seismic-record waterfall.
    stack_offset : float, optional
        Vertical offset between stacked traces. ``None`` picks
        ``2 × max|data|`` for visual separation.
    show_colorbar : bool, optional
        Draw the value colorbar (2-D heatmap only). Default ``True``.
    contours : sequence of float, optional
        Contour levels drawn over the 2-D heatmap.

    Knobs that the selected branch cannot use are rejected with a
    :class:`~uacpy.core.exceptions.ConfigurationError` rather than silently
    dropped — e.g. ``vmin=`` on a field that reduced to one axis.
    """
    if not isinstance(field, Field):
        raise ConfigurationError(
            f"plot_field: expected Field, got {type(field).__name__}"
        )

    if not stacked:
        # A length-1 axis has no neighbours for ``shading='nearest'`` to build
        # cell edges from, so every heatmap quad collapses to zero extent and the
        # panel renders empty — indistinguishable from an all-NaN field. Models
        # pass the receiver axes through verbatim and ``Receiver`` keeps a scalar
        # as a length-1 array, so a single-receiver-depth run reaches this. Drop
        # singleton axes onto ``.pinned`` (the reduction ``_reduce_to_spectrum``
        # already performs) so a ``(1, n)`` field plots as the 1-D cut it is.
        # At least one axis is always kept.
        #
        # ``env`` / ``source`` / ``receiver`` draw over the physical
        # (depth, range) plane, so a caller who supplies one keeps that plane:
        # a single-receiver-depth run reduced to a line cut leaves the overlay
        # nowhere to go, and the knob check below then rejects the call.
        # ``_plot_field_2d`` gives the surviving length-1 axis an explicit cell
        # band so the row still renders.
        keep_cross_section = (env is not None or source is not None
                              or receiver is not None)
        for axis in ('source_depth', 'frequency', 'depth', 'range', 'time'):
            if len(field.coords) <= 1:
                break
            if keep_cross_section and list(field.coords) == ['depth', 'range']:
                break
            if axis in field.coords and field.coords[axis].size == 1:
                field = field.isel(**{axis: 0})

    if value is None:
        value = _default_value(field)
    arr, value_label = _value_array(field, value)
    axes_present = list(field.coords)
    n_axes = len(axes_present)

    if stacked:
        if n_axes != 2 or 'time' not in axes_present:
            raise ConfigurationError(
                "plot_field(stacked=True): requires a 2-D field with a "
                f"'time' axis; got coords {axes_present}"
            )
        branch = 'stacked'
    elif n_axes == 1:
        branch = 'line'
    elif n_axes == 2:
        branch = 'heatmap'
    else:
        raise ConfigurationError(
            f"plot_field: cannot plot a {n_axes}-axis field (coords "
            f"{axes_present}); slice it first with .at(...) / .isel(...) "
            "so 1 or 2 axes remain."
        )

    supplied = {'vmin': vmin, 'vmax': vmax, 'cmap': cmap, 'label': label,
                'show_colorbar': show_colorbar, 'contours': contours,
                'stack_offset': stack_offset, 'env': env, 'source': source,
                'receiver': receiver}
    reject_branch = branch
    unused = list(_BRANCH_UNUSED[branch])
    if branch == 'heatmap' and axes_present != ['depth', 'range']:
        reject_branch = 'other_heatmap'
        unused += list(_CROSS_SECTION_ONLY)
    unused = [k for k in unused if supplied[k] is not None]
    if unused:
        raise ConfigurationError(
            f"plot_field: {', '.join(f'{k}=' for k in unused)} has no effect on "
            f"{_BRANCH_DESCRIPTION[reject_branch]} (coords {axes_present}). "
            f"{', '.join(f'{k}=' for k in _HEATMAP_ONLY)} apply to the 2-D "
            f"heatmap, {', '.join(f'{k}=' for k in _CROSS_SECTION_ONLY)} to a "
            "(depth, range) cross-section, label= to the 1-D line "
            "cut, stack_offset= to stacked=True."
        )
    if (branch == 'heatmap' and contours
            and min(field.coords[a].size for a in axes_present) < 2):
        # A contour is interpolated between neighbouring samples, so an axis
        # held at one sample (a cross-section kept for its overlay) has nothing
        # to trace a level along.
        sizes = ', '.join(f"{a}={field.coords[a].size}" for a in axes_present)
        raise ConfigurationError(
            "plot_field: contours= needs at least 2 samples on both axes; "
            f"got {sizes}."
        )

    if branch == 'stacked':
        fig, ax_out = _plot_field_stacked(
            field, arr, axes_present, ax=ax, title=title,
            figsize=figsize, offset=stack_offset, **mpl_kw,
        )
    elif branch == 'line':
        fig, ax_out = _plot_field_1d(
            field, arr, value_label, axes_present[0], value,
            ax=ax, title=title, label=label, figsize=figsize, **mpl_kw,
        )
    else:
        fig, ax_out = _plot_field_2d(
            field, arr, value_label, axes_present,
            ax=ax, env=env, source=source, receiver=receiver,
            vmin=vmin, vmax=vmax, cmap=cmap, value=value, title=title,
            figsize=figsize,
            show_colorbar=True if show_colorbar is None else show_colorbar,
            contours=contours, **mpl_kw,
        )
    if ax is None:                       # credit only a figure we own
        _draw_result_credit(fig, field, env=env)
    return fig, ax_out


def _plot_field_stacked(
    field, arr, axes_present, *, ax, title, figsize, offset, **mpl_kw,
):
    """Render a 2-D ``(X, time)`` Field as stacked offset traces."""
    time_pos = axes_present.index('time')
    other_axis = axes_present[1 - time_pos]
    if time_pos == 0:
        traces = arr.T  # (n_other, n_t)
    else:
        traces = arr  # already (n_other, n_t)
    # Through _coord_axis so a range axis reads in km here too — every other
    # view of the same axis is labelled that way.
    other_coord, other_label = _coord_axis(field.coords[other_axis], other_axis)
    time = field.coords['time']

    if offset is None:
        peak = float(np.max(np.abs(traces))) if traces.size else 1.0
        offset = 2.0 * peak if peak > 0 else 1.0

    fig, ax = fig_ax(ax, figsize)
    # setdefault, not a positional style: passing linewidth= through **mpl_kw
    # would otherwise collide with the hardcoded one and raise a raw TypeError.
    mpl_kw.setdefault('linewidth', 0.8)
    for i, c in enumerate(other_coord):
        ax.plot(time, traces[i] + i * offset, **mpl_kw)
    ax.set_xlabel(_coord_label('time'))
    ax.set_ylabel(other_label + ' (stacked)')
    # One tick per trace is unreadable the moment a stack is more than a
    # couple of dozen deep: at 60 ranges the labels overprint into a solid
    # black smear down the axis, and a documented receiver grid runs to
    # hundreds. Label every ``step``-th trace instead, the same ~12-label
    # stride ``plot_decidecade_levels`` uses on its band axis. Every trace is
    # still drawn; only the labelling is thinned.
    step = max(1, len(other_coord) // 12)
    shown = range(0, len(other_coord), step)
    ax.set_yticks([i * offset for i in shown])
    # Significant digits, not fixed decimals: a range axis in km spans values
    # a single decimal place would round to the same label.
    ax.set_yticklabels([f"{float(other_coord[i]):.4g}" for i in shown])
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    return fig, ax


def _plot_field_1d(
    field, arr, value_label, axis_name, value,
    *, ax, title, label, figsize, **mpl_kw,
):
    fig, ax = fig_ax(ax, figsize)
    coord = field.coords[axis_name]
    vals = np.asarray(arr).ravel()
    # A line through one sample has nothing to join, so it draws nothing at all.
    # Give a single-sample cut a marker so the value is visible.
    if vals.size == 1:
        mpl_kw.setdefault('marker', 'o')
    if axis_name == 'depth':
        # Depth cut: depth on the Y axis increasing downward (oceanographic
        # convention, consistent with the 2-D views), value on X.
        line, = ax.plot(vals, coord, label=label, **mpl_kw)
        ax.set_xlabel(value_label)
        ax.set_ylabel(_coord_label('depth'))
        invert_yaxis_once(ax)
    else:
        # Range / frequency cut: coordinate on X, value on Y. For a loss
        # (TL, reverberation), put the louder (smaller-dB) end at the top.
        x_plot, x_label = _coord_axis(coord, axis_name)
        line, = ax.plot(x_plot, vals, label=label, **mpl_kw)
        ax.set_xlabel(x_label)
        ax.set_ylabel(value_label)
        if _is_loss_view(field, value):
            invert_yaxis_once(ax)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    pin_text = _pinned_subtitle(field)
    if pin_text and not title:
        ax.set_title(pin_text)
    if label:
        ax.legend()
    return fig, ax


def _cell_edges(coord, half):
    """Cell edges for a centre-sampled coordinate axis.

    Interior edges are the sample midpoints, the outer two mirror the first and
    last interval. A length-1 axis has no interval to mirror, so it gets a band
    of ``±half`` around its single sample."""
    c = np.asarray(coord, dtype=float)
    if c.size == 1:
        return np.array([c[0] - half, c[0] + half])
    mid = 0.5 * (c[:-1] + c[1:])
    return np.concatenate(([2 * c[0] - mid[0]], mid, [2 * c[-1] - mid[-1]]))


def _band_half(name, coord, env):
    """Half-thickness of the band a length-1 ``name`` axis is drawn as.

    The panel autoscales to whatever is drawn, so the thickness is purely
    presentational — except on ``depth``, where the seafloor overlay stretches
    the axis down to the seabed and a band sized against the sample's own depth
    would vanish in a deep water column. Size that one against the water column
    instead."""
    c = abs(float(np.ravel(coord)[0])) or 1.0
    if name == 'depth' and env is not None:
        if env.has_range_dependent_bathymetry:
            c = max(c, float(np.max(env.bathymetry.depths)))
        else:
            c = max(c, float(env.depth))
    return _SINGLETON_BAND_FRACTION * c


def _mesh_with_singleton_bands(ax, x_plot, x_name, y_plot, y_name, Z, env,
                               **kw):
    """``pcolormesh`` that also renders an axis held at a single sample.

    ``shading='nearest'`` builds each cell's edges from the neighbouring
    samples, so a length-1 axis collapses every quad to zero extent and the
    panel comes out empty — indistinguishable from an all-NaN field. Hand such
    an axis explicit edges instead and draw its samples as a band centred on
    their own coordinate.

    ``'nearest'`` (not ``'auto'``) on the ordinary path: it centres each cell on
    its coordinate and errors loudly if the coords are ever the wrong length,
    where ``'auto'`` would silently switch to edge (``'flat'``) mode and
    half-cell-shift the field."""
    if np.size(x_plot) == 1 or np.size(y_plot) == 1:
        return ax.pcolormesh(
            _cell_edges(x_plot, _band_half(x_name, x_plot, env)),
            _cell_edges(y_plot, _band_half(y_name, y_plot, env)),
            Z, shading='flat', **kw,
        )
    return ax.pcolormesh(x_plot, y_plot, Z, shading='nearest', **kw)


def _signal_excess_title(field) -> str:
    """The auto-title a signal-excess field carries through either plotter:
    the quantity, the budget mode when the field records one, the pinned
    coordinates when it has any."""
    budget = field.metadata.get('sonar_budget') or {}
    mode = budget.get('mode')
    pin = _pinned_subtitle(field)
    auto = 'Signal excess' + (f' ({mode})' if mode else '')
    return f"{auto} — {pin}" if pin else auto


def _symmetric_span(datasets):
    """``(-max|x|, +max|x|)`` pooled over every array in ``datasets``, or
    ``(None, None)`` when none of them holds a finite non-zero sample.

    ``np.abs()`` rather than a float cast: a signal-excess field is normally
    real dB, but the cast warns and discards the imaginary part on the complex
    case, and the magnitude is what the symmetric window needs either way."""
    span = 0.0
    for data in datasets:
        mag = np.abs(np.asarray(data))
        if np.isfinite(mag).any():
            span = max(span, float(np.nanmax(mag)))
    return (-span, span) if span > 0.0 else (None, None)


def _is_time_domain(field) -> bool:
    """Whether ``field`` holds real time-domain pressure — a wavefield to be
    read as a moveout, not a level."""
    return 'time' in field.coords and not field.is_complex


def _value_style(field, value):
    """``(cmap, vmin, vmax)`` — the colormap and any **fixed** colour limits the
    2-D view of ``value`` carries, ``None`` where the mode leaves the choice to
    the data.

    ``plot_field`` and ``compare_models`` both read this, so one field renders
    in the same colours through either entry point. Choosing a figure-level
    colormap independently is a real error and a quiet one: ``'phase'`` is
    cyclic and takes ``twilight``, and on a non-cyclic map -π and +π — the same
    phase — land at opposite ends of the scale, so the wrap where the phase
    rolls over reads as a discontinuity in the field."""
    if _is_time_domain(field):
        # Real time-domain pressure → diverging map centred at 0. Its limits
        # come from the record's own RMS, so they are not fixed here.
        return 'seismic', None, None
    if value == 'dB':
        # The fixed scale is a *transmission-loss* convention, so it applies
        # only to a pressure field. Another dB quantity — signal excess spans
        # roughly -20..+40 dB — renders as one flat block against 20..120.
        if field.kind == 'pressure':
            lo, hi = _TL_LIMITS
        elif field.kind in ('signal_excess', 'difference'):
            # A diverging colormap carries its meaning in the NEUTRAL colour,
            # and for signal excess that colour is the SE = 0 dB detection
            # boundary (style.py says so where 'RdBu_r' is chosen). Leaving the
            # window to matplotlib's asymmetric autoscale put the neutral
            # wherever the data happened to centre: measured on a field
            # spanning -20..+40 dB, white landed at SE = +10 dB, so a 10 dB
            # band of genuinely detectable water was painted the colour the map
            # reserves for "not detectable". Symmetric limits put 0 dB on the
            # neutral, which is what the dedicated plot_signal_excess already
            # does. ``compare_models`` pools the same window over its panels.
            lo, hi = _symmetric_span([field.data])
        else:
            lo, hi = (None, None)
        return cmap_for_field(field.kind, dB=True), lo, hi
    if value == 'phase':
        return 'twilight', -np.pi, np.pi
    if value == 'mag_dB':
        # A dB view of |H|, so larger is LOUDER — the opposite of the loss
        # the dB colormap is built for. Measured under the unreversed map:
        # the loudest water came out dark blue at -20 dB while the same cell
        # reads dark red through ``dB``, against style.py's stated
        # convention that low TL (loud, near) is red.
        return (reversed_cmap(cmap_for_field(field.kind, dB=True)),
                None, None)
    if not field.is_complex and field.unit == '1':
        # A real, dimensionless quantity is a probability: bounded [0, 1] and
        # unsigned. The signed linear map below put the whole field in shades
        # of red on an autoscaled (-1, 1) — the blue half unreachable, its
        # neutral white sitting at P_D = 0 — while the dedicated
        # plot_detection_probability drew the same field green-to-red on a
        # fixed [0, 1]. Same field, two doors, two pictures.
        return PROBABILITY_COLORMAP, *PROBABILITY_LIMITS
    # 'mag' / 'real' / 'imag' are linear views, which share one signed
    # colormap whatever the quantity.
    return cmap_for_field(field.kind, dB=False), None, None


def _plot_field_2d(
    field, arr, value_label, axes_present,
    *, ax, env, vmin, vmax, cmap, value, title, figsize, source=None,
    receiver=None,
    show_colorbar=True, contours=None, **mpl_kw,
):
    # First coord axis on Y, second on X — which for the canonical
    # ['depth', 'range'] field is the usual depth-vs-range cross-section.
    y_name, x_name = axes_present
    Z = arr

    x_coord = field.coords[x_name]
    y_coord = field.coords[y_name]

    # Auto-defaults for value-specific styling.
    is_time_domain = _is_time_domain(field)
    style_cmap, style_vmin, style_vmax = _value_style(field, value)
    if cmap is None:
        cmap = style_cmap
    if vmin is None:
        vmin = style_vmin
    if vmax is None:
        vmax = style_vmax
    if is_time_domain:
        # Clip to ±RMS so silence between arrivals doesn't wash out the
        # wavefront — peaks saturate, which is exactly what we want for a
        # moveout reading.
        if vmin is None or vmax is None:
            finite = np.abs(Z[np.isfinite(Z)])
            if finite.size:
                rms = float(np.sqrt(np.mean(finite ** 2)))
                peak = rms if rms > 0 else float(finite.max())
            else:
                peak = 1.0
            vmin = -peak if vmin is None else vmin
            vmax = peak if vmax is None else vmax
        value_label = 'p(t)'
    elif value in _DB_VALUES and (vmin is None or vmax is None):
        # A no-energy cell is a MARKER, not a level: the package writes
        # ``PRESSURE_FLOOR`` where the model reported no energy, so it lands
        # 600 dB out and drags the bar with it — measured, ``mag_dB`` ran
        # -600..-20 and a loss view 20..600, each packing every real level
        # into a tenth of the scale. Reading it by MAGNITUDE covers both
        # views, which carry the same cell at opposite signs. Nothing the
        # model computed reaches that far, so this needs no threshold and no
        # percentile — and unlike a percentile it keeps a genuine deep null,
        # which is the feature the view exists to show: on a 1/r field with
        # one marked cell and a real -70 dB null, 580 dB of bar becomes 70,
        # where the 1st percentile would have cut the null off at -80 dB.
        finite = Z[np.isfinite(Z)]
        levels = finite[np.abs(finite) < _NO_ENERGY_DB]
        if levels.size:
            vmin = float(levels.min()) if vmin is None else vmin
            vmax = float(levels.max()) if vmax is None else vmax
    elif value in ('mag', 'real', 'imag') and (vmin is None or vmax is None):
        # Zero has to land at a fixed place on the map, so the signed views
        # take symmetric limits (zero on the diverging map's white) and the
        # non-negative modulus starts at 0 (zero at the map's dark end, white
        # at half the maximum). An autoscale puts zero at an arbitrary colour
        # instead.
        # The top is the maximum, deliberately. A linear view compresses
        # everything under its loudest cell — that is what a linear scale
        # is — and no robust statistic helps: on a 1/r field the 99th
        # percentile IS the maximum (measured), and a lower one clips the
        # near field the view exists to show. The no-energy marker cannot
        # reach here: it is ``PRESSURE_FLOOR`` (1e-30), a tiny number, so it
        # never sets ``span``. For dynamic range, read the dB views.
        finite = np.abs(Z[np.isfinite(Z)])
        span = float(finite.max()) if finite.size else 1.0
        if span <= 0:
            span = 1.0
        lo = 0.0 if value == 'mag' else -span
        vmin = lo if vmin is None else vmin
        vmax = span if vmax is None else vmax

    fig, ax = fig_ax(ax, figsize)

    # Both axes go through _coord_axis, so a range axis reads in km whether it
    # lands on x or on y — same scale as every other view of that axis.
    x_plot, x_label = _coord_axis(x_coord, x_name)
    y_plot, y_label = _coord_axis(y_coord, y_name)

    # ``plot_field`` keeps a length-1 axis only for a cross-section overlay;
    # the helper draws it as a band so the row still renders.
    im = _mesh_with_singleton_bands(
        ax, x_plot, x_name, y_plot, y_name, Z, env,
        vmin=vmin, vmax=vmax, cmap=cmap, **mpl_kw,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # Any depth-denoting y axis (depth, source_depth) is positive-down
    # (core/results/field.py documents both in metres below the surface).
    if y_name.endswith('depth'):
        invert_yaxis_once(ax)
    if contours:
        cs = ax.contour(
            x_plot, y_plot, Z, levels=list(contours),
            colors='black', linewidths=1.5, alpha=0.8,
            linestyles='solid',
        )
        ax.clabel(cs, inline=True, fontsize=9,
                  fmt=_CONTOUR_FMT.get(value, '%g'))
    if show_colorbar:
        fig.colorbar(im, ax=ax, label=value_label,
                     fraction=0.046, pad=0.02)
    ax.grid(True, alpha=0.3, zorder=0)
    if title:
        ax.set_title(title)
    elif field.kind == 'signal_excess':
        ax.set_title(_signal_excess_title(field))      # as plot_signal_excess does
    else:
        pin = _pinned_subtitle(field)
        if pin:
            ax.set_title(pin)
    # env / source / receiver are rejected before the figure exists (see
    # _CROSS_SECTION_ONLY), so reaching here with them set means this really
    # is a (depth, range) cross-section.
    if axes_present == ['depth', 'range']:
        if env is not None:
            _overlay_seafloor(ax, env, x_coord)
        if source is not None or receiver is not None:
            # Range is measured from the source, so the source sits at r = 0
            # even when the field's own grid starts further out.
            _draw_geometry(ax, source, receiver, max_markersize=6,
                           source_range_m=0.0)
    return fig, ax


def _begin_sonar_heatmap(field, ax, *, env, figsize, vmin, vmax, cmap,
                         **mpl_kw):
    """Open a ``(depth, range)`` panel for a sonar-equation field and draw its
    mesh. Shared by :func:`plot_signal_excess` and
    :func:`plot_detection_probability`, which differ here only in the colour
    window they hand in.

    Returns ``(fig, ax, im, Z, r_km, x_label, depths, owns_fig)`` — the pieces
    each caller's own overlay (an SE = 0 contour, labelled P_D contours) needs
    before :func:`_finish_sonar_heatmap` closes the panel."""
    Z = np.asarray(field.data, dtype=float)
    owns_fig = ax is None
    fig, ax = fig_ax(ax, figsize)
    r_km, x_label = _coord_axis(field.coords['range'], 'range')
    depths = field.coords['depth']
    # A single-receiver-depth run (or a single range) reaches here as a
    # length-1 axis, which the helper draws as a band rather than the
    # zero-extent — empty — mesh 'nearest' shading would build.
    im = _mesh_with_singleton_bands(
        ax, r_km, 'range', depths, 'depth', Z, env,
        vmin=vmin, vmax=vmax, cmap=cmap, **mpl_kw,
    )
    return fig, ax, im, Z, r_km, x_label, depths, owns_fig


def _finish_sonar_heatmap(fig, ax, im, field, *, env, x_label, colorbar_label,
                          show_colorbar, title, auto_title, owns_fig):
    """Close a ``(depth, range)`` sonar panel: colorbar, axis labels, depth
    downward, title, seafloor overlay and the data credit.

    The colorbar label and the automatic title are the caller's, because each
    plotter is the dedicated view of one quantity and names it itself."""
    if show_colorbar:
        fig.colorbar(im, ax=ax, label=colorbar_label,
                     fraction=0.046, pad=0.02)
    ax.set_xlabel(x_label)
    ax.set_ylabel(_coord_label('depth'))
    invert_yaxis_once(ax)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_title(title if title else auto_title)
    if env is not None:
        _overlay_seafloor(ax, env, field.coords['range'])
    if owns_fig:                         # credit only a figure we own
        _draw_result_credit(fig, field, env=env)
    return fig, ax


@typed_plot_error
def plot_signal_excess(
    field: Field,
    ax=None,
    *,
    env: Optional[Environment] = None,
    vmax: Optional[float] = None,
    cmap: str = 'RdBu_r',
    show_boundary: bool = True,
    show_colorbar: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
    **mpl_kw,
):
    """Heatmap of a signal-excess :class:`Field` over ``(depth, range)``.

    Renders the output of
    :func:`uacpy.sonar.passive_signal_excess_field` /
    :func:`uacpy.sonar.active_signal_excess_field` with a diverging
    colormap centred at SE = 0 dB (warm = detectable, cool = not) and
    draws the SE = 0 contour — the detection boundary.

    Parameters
    ----------
    field : Field
        Real-valued signal excess in dB with canonical
        ``coords == {'depth', 'range'}`` (slice broadband fields first).
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    env : Environment, optional
        Overlays the seafloor, as in :func:`plot_field`.
    vmax : float, optional
        Symmetric colour limit ``[-vmax, +vmax]``. ``None`` uses
        ``max|SE|``, the same symmetric window ``field.plot()`` picks for this
        kind, so the two doors paint one field alike and SE = 0 dB lands on
        the diverging map's neutral colour.
    cmap : str, optional
        Diverging colormap. Default ``'RdBu_r'``.
    show_boundary : bool, optional
        Draw the SE = 0 dB detection-boundary contour. Default True.
    """
    if not isinstance(field, Field):
        raise ConfigurationError(
            f"plot_signal_excess: expected Field, got {type(field).__name__}"
        )
    if field.is_complex:
        raise ConfigurationError(
            "plot_signal_excess: field must carry real signal excess in "
            "dB — build it with passive_signal_excess_field / "
            "active_signal_excess_field."
        )
    if list(field.coords) != ['depth', 'range']:
        raise ConfigurationError(
            "plot_signal_excess: requires canonical ['depth', 'range'] "
            f"coords; got {list(field.coords)} — slice with .at(...) first."
        )

    Z = np.asarray(field.data, dtype=float)
    if vmax is None:
        # The same symmetric window ``field.plot()`` takes for this kind
        # (``_value_style``), so the two doors paint one field alike and 0 dB
        # sits on the diverging map's neutral colour.
        _lo, vmax = _symmetric_span([Z])
        if not vmax or vmax <= 0:
            vmax = 1.0

    fig, ax, im, Z, r_km, x_label, depths, _owns_fig = _begin_sonar_heatmap(
        field, ax, env=env, figsize=figsize, vmin=-vmax, vmax=vmax, cmap=cmap,
        **mpl_kw)
    if show_boundary and np.isfinite(Z).any():
        finite_z = Z[np.isfinite(Z)]
        if finite_z.min() < 0.0 < finite_z.max():
            if min(Z.shape) < 2:
                # A contour is interpolated between neighbouring samples, so a
                # field held at one depth (or one range) has nothing to trace
                # the boundary along. The heatmap itself still renders.
                _plot_warn(
                    "plot_signal_excess: the SE = 0 boundary needs at least 2 "
                    f"samples on both axes; got depth={Z.shape[0]}, "
                    f"range={Z.shape[1]}, so no boundary contour is drawn.")
            else:
                cs = ax.contour(
                    r_km, depths, Z, levels=[0.0],
                    colors='black', linewidths=1.5, linestyles='solid',
                )
                ax.clabel(cs, inline=True, fontsize=9,
                          fmt=lambda _: 'SE = 0 dB')
    return _finish_sonar_heatmap(
        fig, ax, im, field, env=env, x_label=x_label,
        colorbar_label='Signal excess (dB)', show_colorbar=show_colorbar,
        title=title, auto_title=_signal_excess_title(field),
        owns_fig=_owns_fig)


@typed_plot_error
def plot_detection_probability(
    field: Field,
    ax=None,
    *,
    env: Optional[Environment] = None,
    cmap: str = PROBABILITY_COLORMAP,
    contour_levels: Sequence[float] = (0.1, 0.5, 0.9),
    show_colorbar: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
    **mpl_kw,
):
    """Heatmap of a detection-probability :class:`Field` over ``(depth, range)``.

    Renders the output of
    :func:`uacpy.sonar.probability_of_detection_field` on a fixed
    ``[0, 1]`` colour scale (green = detectable) with labelled ``P_D``
    contours.

    Parameters
    ----------
    field : Field
        ``P_D`` values in [0, 1] with canonical
        ``coords == {'depth', 'range'}``.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    env : Environment, optional
        Overlays the seafloor, as in :func:`plot_field`.
    cmap : str, optional
        Colormap. Default ``'RdYlGn'`` (red = lost, green = detected).
    contour_levels : sequence of float, optional
        ``P_D`` contour lines to draw. Default ``(0.1, 0.5, 0.9)``.
    """
    if not isinstance(field, Field):
        raise ConfigurationError(
            f"plot_detection_probability: expected Field, got "
            f"{type(field).__name__}"
        )
    if field.is_complex:
        raise ConfigurationError(
            "plot_detection_probability: field must carry real P_D in "
            "[0, 1] — build it with probability_of_detection_field."
        )
    if list(field.coords) != ['depth', 'range']:
        raise ConfigurationError(
            "plot_detection_probability: requires canonical "
            f"['depth', 'range'] coords; got {list(field.coords)} — "
            "slice with .at(...) first."
        )

    fig, ax, im, Z, r_km, x_label, depths, _owns_fig = _begin_sonar_heatmap(
        field, ax, env=env, figsize=figsize, vmin=PROBABILITY_LIMITS[0],
        vmax=PROBABILITY_LIMITS[1], cmap=cmap, **mpl_kw)
    finite = Z[np.isfinite(Z)]
    if contour_levels and finite.size:
        levels = [
            lv for lv in sorted(contour_levels)
            if finite.min() < lv < finite.max()
        ]
        if levels and min(Z.shape) < 2:
            # Nothing to interpolate a level along on an axis held at one
            # sample; the heatmap itself still renders.
            _plot_warn(
                "plot_detection_probability: contour_levels= needs at least 2 "
                f"samples on both axes; got depth={Z.shape[0]}, "
                f"range={Z.shape[1]}, so no contours are drawn.")
        elif levels:
            cs = ax.contour(
                r_km, depths, Z, levels=levels,
                colors='black', linewidths=1.2, linestyles='solid',
            )
            ax.clabel(cs, inline=True, fontsize=9, fmt='%.1f')
    sigma = field.metadata.get('sigma_dB')
    pin = _pinned_subtitle(field)
    auto = 'Detection probability'
    if sigma is not None:
        auto += f' (σ = {sigma:g} dB)'
    return _finish_sonar_heatmap(
        fig, ax, im, field, env=env, x_label=x_label,
        colorbar_label='Probability of detection',
        show_colorbar=show_colorbar, title=title,
        auto_title=f"{auto} — {pin}" if pin else auto, owns_fig=_owns_fig)


@typed_plot_error
def compare(
    fields: Sequence[Field],
    labels: Optional[Sequence[str]] = None,
    ax=None,
    *,
    value: str = 'dB',
    figsize: Tuple[float, float] = (10, 5),
    title: Optional[str] = None,
    **mpl_kw,
):
    """Overlay multiple 1-D sliced :class:`Field` instances on one axes.

    Every field must reduce to a single surviving coord axis (the same
    axis across all). Caller slices them first::

        compare([f1.at(depth=20), f2.at(depth=20)], labels=['Bellhop', 'RAM'])

    Axes follow :func:`plot_field`, so one field cuts the same way through
    either: depth increases downward, and so does the value axis of a loss
    cut — transmission loss or reverberation, see :func:`_is_loss_view` — but
    not that of any other dB quantity, which is a level and reads upward.
    """
    if not fields:
        raise ConfigurationError(
            "compare: empty fields list — pass the Fields to overlay, each "
            "already cut to one surviving axis (field.at(...)).")
    if labels is not None and len(labels) != len(fields):
        raise ConfigurationError(
            f"compare: labels ({len(labels)}) must match fields "
            f"({len(fields)})"
        )
    _owns_fig = ax is None
    fig, ax = fig_ax(ax, figsize)
    if labels is None:
        labels = [getattr(f, 'model', '') or f"#{i}" for i, f in enumerate(fields)]
    common_axis = None
    x_label = None
    for f, lbl in zip(fields, labels):
        if not isinstance(f, Field):
            raise ConfigurationError(
                f"compare: expected Field, got {type(f).__name__}"
            )
        # Compare the QUANTITY, exactly as compare_models does: overlaying a
        # reverberation loss on a TL cut puts two different physical
        # quantities on one value axis, with one shared label — even though
        # both now run in the same direction.
        if f.kind != fields[0].kind:
            raise ConfigurationError(
                f"compare: {lbl!r} is a {f.kind!r} field but "
                f"{labels[0]!r} is {fields[0].kind!r} — these are different "
                f"physical quantities and share no value axis.",
                remediation="Compare like with like, or plot them separately "
                            "with plot_field.")
        axes = list(f.coords)
        if len(axes) != 1:
            raise ConfigurationError(
                f"compare: each field must have exactly 1 surviving axis; "
                f"{lbl!r} has {axes}"
            )
        if common_axis is None:
            common_axis = axes[0]
        elif axes[0] != common_axis:
            raise ConfigurationError(
                f"compare: axis mismatch — {labels[0]!r} on {common_axis!r}, "
                f"{lbl!r} on {axes[0]!r}"
            )
        arr, _ = _value_array(f, value)
        if common_axis == 'depth':
            # Depth-cut overlays follow plot_field's convention:
            # depth on Y, increasing downward.
            ax.plot(np.asarray(arr).ravel(), f.coords[common_axis],
                    label=lbl, **mpl_kw)
        else:
            x_plot, x_label = _coord_axis(f.coords[common_axis], common_axis)
            ax.plot(x_plot, np.asarray(arr).ravel(), label=lbl, **mpl_kw)
    # The kind check above makes the first field representative of them all,
    # so it settles the shared value-axis label and — for a loss cut — the
    # direction that axis runs.
    vlabel = _value_label(fields[0], value)
    value_is_loss = _is_loss_view(fields[0], value)
    if common_axis == 'depth':
        ax.set_ylabel(_coord_label(common_axis))
        ax.set_xlabel(vlabel)
        invert_yaxis_once(ax)
    else:
        ax.set_xlabel(x_label)
        ax.set_ylabel(vlabel)
        if value_is_loss:
            invert_yaxis_once(ax)
    ax.grid(True, alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title)
    if _owns_fig:
        _draw_multi_model_credit(fig, fields)
    return fig, ax


def _draw_multi_model_credit(fig, fields):
    """One credit footnote listing every distinct contributing model."""
    seen, attrs = set(), []
    for f in fields:
        a = _model_attribution(f)
        if a and a not in seen:
            seen.add(a)
            attrs.append(a)
    if attrs:
        _draw_credit(fig, (), model=attrs)


@typed_plot_error
def compare_models(
    fields,
    labels: Optional[Sequence[str]] = None,
    *,
    env: Optional[Environment] = None,
    value: str = 'dB',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    ncols: Optional[int] = None,
    contours: Optional[Sequence[float]] = None,
):
    """Side-by-side heatmaps of several 2-D :class:`Field` instances.

    ``fields`` is either a list of :class:`Field` (then ``labels`` is
    used as the per-axes title), or a ``{label: Field}`` dict. Shared
    colour scale; one colorbar per axes.

    ``title`` titles the whole figure; the per-panel titles come from
    ``labels``. ``ncols`` controls the grid width — defaults to ``n`` (single
    row). ``contours`` adds dB-level contour lines to every panel.

    Returns
    -------
    fig, axes : Figure, ndarray of Axes
        ``axes`` is the 2-D ``(nrows, ncols)`` array ``plt.subplots``
        produced (unused cells are turned off) — the same shape every
        grid-of-panels helper on this surface returns.
    """
    if isinstance(fields, dict):
        if labels is None:
            labels = list(fields.keys())
        fields = list(fields.values())
    n = len(fields)
    if n == 0:
        raise ConfigurationError(
            "compare_models: empty fields list — pass one Field per model, "
            "as a list or as a {label: field} dict.")
    if labels is None:
        labels = [getattr(f, 'model', '') or f"#{i}" for i, f in enumerate(fields)]
    elif len(labels) != n:
        raise ConfigurationError(
            f"compare_models: got {len(labels)} labels for {n} fields — they "
            f"must match (panels are labelled by zipping the two).")
    if ncols is None:
        ncols = n
    nrows = int(np.ceil(n / ncols))

    ref = fields[0]
    for f, lbl in zip(fields[1:], labels[1:]):
        # Compare the QUANTITY, not the kind: a complex pressure field and a
        # real TL field are the same quantity written two ways, and comparing
        # them is the ordinary case. A reverberation loss shares TL's
        # representation exactly but is a different quantity, and putting the
        # two on one colour scale asserts an equivalence that does not hold.
        if f.kind != ref.kind:
            raise ConfigurationError(
                f"compare_models: {lbl!r} is a {f.kind!r} field but "
                f"{labels[0]!r} is {ref.kind!r} — these are different "
                f"physical quantities and share no colour scale.",
                remediation="Compare like with like, or plot them separately "
                            "with plot_field.")
        for axis in ('depth', 'range'):
            if axis not in ref.coords or axis not in f.coords:
                continue
            ca = ref.coords[axis]
            cb = f.coords[axis]
            if ca.shape != cb.shape or not np.allclose(
                ca, cb, rtol=1e-6, atol=1e-6
            ):
                _plot_warn(
                    f"compare_models: {lbl!r} {axis} axis differs from "
                    f"{labels[0]!r}; the shared colourbar mixes "
                    "different sample grids.",
                )
                break

    # Every panel here shares one kind already, so ``ref`` settles the styling
    # for the whole figure — and it is read from the same table plot_field
    # reads, so a panel is coloured as if it had been plotted on its own.
    style_cmap, style_vmin, style_vmax = _value_style(ref, value)
    if (value == 'dB' and ref.kind == 'signal_excess'
            and not _is_time_domain(ref)):
        # _value_style sizes the symmetric signal-excess window from the one
        # field it is handed, which here is panel 1: measured on a second panel
        # spanning four times as wide, 70% of its samples saturated, and
        # reversing the list changed the shared limits. Pool the magnitude over
        # every panel instead — still SYMMETRIC, so SE = 0 keeps the diverging
        # map's neutral colour. The generic pooled branch below cannot do this
        # job: its asymmetric min/max would put the neutral wherever the data
        # happen to centre.
        style_vmin, style_vmax = _symmetric_span([f.data for f in fields])
    if cmap is None:
        cmap = style_cmap
    if vmin is None:
        vmin = style_vmin
    if vmax is None:
        vmax = style_vmax
    if vmin is None or vmax is None:
        # A view with no fixed scale pools the panels: left to autoscale, each
        # ``plot_field`` would map its own panel's range and the single
        # figure-level colorbar below would then annotate the figure with only
        # the last panel's limits — two fields differing by 100x would render
        # identically. Signed quantities get a symmetric range so zero stays the
        # neutral colour; a *level* (a dB view — signal excess, |H|) is not
        # signed however negative it reads, and forcing -80..-20 dB out to ±80
        # would leave it occupying half the colormap.
        pooled = [np.asarray(_value_array(f, value)[0], dtype=float).ravel()
                  for f in fields]
        finite = np.concatenate([p[np.isfinite(p)] for p in pooled]) \
            if any(np.isfinite(p).any() for p in pooled) else np.array([0.0])
        if _is_time_domain(ref):
            # As in plot_field: a wavefield is clipped to ±RMS so silence
            # between arrivals does not wash out the wavefront. Taken over the
            # pooled samples, so every panel saturates at one shared level.
            mag = np.abs(finite)
            rms = float(np.sqrt(np.mean(mag ** 2)))
            span = rms if rms > 0 else float(mag.max())
            lo, hi = -span, span
        elif value in ('real', 'imag'):
            span = float(np.max(np.abs(finite))) or 1.0
            lo, hi = -span, span
        elif value == 'mag':
            # As in plot_field: the modulus is non-negative and starts at 0, so
            # zero keeps the linear colormap's neutral colour.
            lo, hi = 0.0, float(np.max(finite)) or 1.0
        else:
            lo, hi = float(np.min(finite)), float(np.max(finite))
            if hi <= lo:                       # a constant field has no range
                lo, hi = lo - 0.5, hi + 0.5
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax

    if figsize is None:
        figsize = (6.0 * ncols + 1.6, 5.0 * nrows + 1.2)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()
    im_last = None
    for f, label, ax in zip(fields, labels, axes_flat):
        plot_field(
            f, ax=ax, env=env, value=value,
            vmin=vmin, vmax=vmax, cmap=cmap, title=label,
            contours=contours, show_colorbar=False,
        )
        # Every panel was drawn with the same vmin/vmax/cmap, so any one mesh
        # maps the shared scale — keep the last panel's mesh for the single
        # figure-level colorbar. Picked by TYPE, not by position: the panel also
        # carries the contour set (a Collection since matplotlib 3.8) and the
        # seafloor fills, and the mesh sits first only because it is drawn
        # first. A panel that drew no mesh contributes nothing, so a figure
        # where none did keeps its colorbar suppressed below rather than
        # captioning the shared scale with whatever else the axes holds.
        mesh = next((c for c in ax.collections
                     if isinstance(c, _mcoll.QuadMesh)), None)
        if mesh is not None:
            im_last = mesh
    for ax in axes_flat[n:]:
        ax.axis('off')

    top = 0.90 if title else 0.95
    # One credit line per model sits under the panels; leave it room.
    bottom = 0.08 + 0.025 * max(0, len(fields) - 1)
    fig.subplots_adjust(left=0.05, right=0.88, top=top, bottom=bottom,
                        wspace=0.22, hspace=0.30)
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.97)
    _draw_multi_model_credit(fig, fields)
    if im_last is not None:
        # The label the panels would carry if each had drawn its own colorbar:
        # the raw ``value`` string is the knob's name, not the quantity's.
        cbar_label = 'p(t)' if _is_time_domain(ref) else _value_label(ref, value)
        # Added AFTER the credit: _draw_multi_model_credit reserves its own
        # margin with a second subplots_adjust, and an axes placed at the
        # earlier ``bottom`` does not follow it. Read the panels' final bottom.
        cbar_bottom = fig.subplotpars.bottom
        cbar_ax = fig.add_axes((0.905, cbar_bottom, 0.015, top - cbar_bottom))
        fig.colorbar(im_last, cax=cbar_ax, label=cbar_label)
    # One shape for every grid-of-panels return on this surface: the 2-D
    # axes array, matching _plot_field_stack (documented in the Returns
    # section above).
    return fig, axes


@typed_plot_error
def shared_colorbar(fig, axes, *, label=None, **colorbar_kw):
    """One colorbar for a row or grid of panels, from the panels' own mappable.

    Every plotter returns ``(fig, ax)``, so composing several into one figure
    under a single colorbar meant reaching into ``ax.collections`` and knowing
    to filter out the contour overlays that live there too. This does that, and
    checks the thing a hand-rolled version cannot: **every panel must be on one
    colour scale**, because a single bar drawn over panels with different
    limits or colormaps describes one of them and mislabels the rest.

    Draw the panels with ``show_colorbar=False`` and call this once.

    Parameters
    ----------
    fig : Figure
    axes : Axes, or a sequence / ndarray of Axes
        The panels the bar describes. Space is taken from all of them, so the
        bar lines up with the group rather than needing a hand-placed cax.
    label : str, optional
        Colorbar label. The panels were drawn without their own bars, so there
        is no label to recover from them — pass the quantity's name.
    **colorbar_kw
        Forwarded to ``Figure.colorbar`` (``fraction``, ``pad``, ``shrink``,
        ``location``, …). ``fraction`` and ``pad`` default to a **thin** bar
        (0.02 / 0.02) rather than matplotlib's 0.15, which is sized for a
        single axes and eats a sixth of a multi-panel sheet.

    Returns
    -------
    Colorbar

    Raises
    ------
    ConfigurationError
        No panel drew a heatmap, or the panels disagree on colour limits or
        colormap.

    Notes
    -----
    Call this **after** any ``subplots_adjust``. The bar takes its space from
    the panels as they stand, and a later ``subplots_adjust`` moves the panels
    back over it — the same ordering hazard ``compare_models`` documents for
    its credit footnote.
    """
    panels = [ax for ax in np.atleast_1d(np.asarray(axes, dtype=object)).ravel()
              if ax is not None]
    if not panels:
        raise ConfigurationError(
            "shared_colorbar: no axes given.",
            remediation="Pass the panel (or the array of panels) the bar "
                        "should describe.")

    found = []
    for ax in panels:
        # Contour overlays are Collections too, so the heatmap is identified by
        # type rather than by position in ax.collections.
        meshes = [arr for arr in ax.collections if isinstance(arr, _mcoll.QuadMesh)]
        found.extend(meshes + list(ax.images))
    if not found:
        raise ConfigurationError(
            "shared_colorbar: none of these axes carries a heatmap.",
            remediation="Draw the panels with a heatmap plotter (plot_field on "
                        "a 2-D Field, plot_field_difference, …) before asking "
                        "for a bar over them; a 1-D line cut has no colour "
                        "scale to describe.")

    reference = found[0]
    ref_clim, ref_cmap = reference.get_clim(), reference.get_cmap().name
    for other in found[1:]:
        clim, cmap = other.get_clim(), other.get_cmap().name
        if not np.allclose(clim, ref_clim, equal_nan=True) or cmap != ref_cmap:
            raise ConfigurationError(
                f"shared_colorbar: the panels are not on one colour scale — "
                f"{ref_cmap} over {ref_clim} against {cmap} over {clim}.",
                remediation="Pass the same vmin/vmax (and cmap) to every "
                            "panel, or give each its own colorbar. One bar "
                            "over two scales describes one panel and "
                            "mislabels the other.")
    colorbar_kw.setdefault('fraction', 0.02)
    colorbar_kw.setdefault('pad', 0.02)
    return fig.colorbar(reference, ax=panels, label=label, **colorbar_kw)


def _same_grid_or_raise(caller: str, field: Field, reference: Field) -> None:
    """Refuse two fields that are not on one grid.

    Equal lengths are not equal grids: two 200-point range axes over
    different spans subtract cell by cell into a plausible number for
    positions that never met.
    """
    def axis_lengths(f):
        return {name: len(axis) for name, axis in f.coords.items()}

    if list(field.coords) != list(reference.coords) or \
            field.data.shape != reference.data.shape:
        raise ConfigurationError(
            f"{caller}: the fields are on different grids — "
            f"{axis_lengths(field)} vs {axis_lengths(reference)}.",
            remediation="Run both models on one receiver grid, or resample "
                        "one onto the other's axes before differencing.",
        )
    for name, axis in field.coords.items():
        if not np.allclose(axis, reference.coords[name]):
            raise ConfigurationError(
                f"{caller}: the fields share the shape of their '{name}' "
                f"axis but not its values.",
                remediation="Difference fields sampled at the same "
                            "coordinates; same length is not same axis.",
            )


@typed_plot_error
def plot_field_difference(
    field: Field,
    reference: Field,
    ax=None,
    *,
    env: Optional[Environment] = None,
    vmin: float = -10.0,
    vmax: float = 10.0,
    diff_vmax: Optional[float] = None,
    title: Optional[str] = None,
    **mpl_kw,
):
    """Heatmap of ``field`` minus ``reference``, in dB.

    The panel shows a signed RESIDUAL, not a level: positive means ``field``
    carries the higher loss, so it is the quieter of the two. It takes the
    diverging colormap and a window symmetric about zero, because zero — not
    the smallest value — is the meaningful reading.

    Parameters
    ----------
    field, reference : Field
        The two fields to difference, on the same grid. A mismatch raises.
    ax : Axes, optional
        Draw here instead of a new figure.
    env : Environment, optional
        Overlay the seafloor, as :func:`plot_field` does.
    vmin, vmax : float
        Colour limits, default ±10 dB.
    diff_vmax : float, optional
        Symmetric-window shortcut: sets ``vmin, vmax = -diff_vmax, +diff_vmax``.
    title : str, optional
    **mpl_kw
        Forwarded to :func:`plot_field`.

    Returns
    -------
    fig, ax : Figure, Axes

    See Also
    --------
    compare : the same two fields as 1-D cuts, overlaid.
    plot_field_statistics : the residual reduced to one number per pair.
    """
    _same_grid_or_raise('plot_field_difference', field, reference)
    if diff_vmax is not None:
        vmin, vmax = -abs(diff_vmax), abs(diff_vmax)

    difference = Field(data=field.dB - reference.dB, coords=dict(field.coords))
    # Tag it for what it is. Untagged, a Field inherits ``kind='pressure'``,
    # which would caption the bar 'TL (dB)' over a signed residual and let the
    # loss predicate run a 1-D cut's value axis downward — meaningless for a
    # difference, whose zero is the meaningful value. The 'difference'
    # quantity carries the diverging map and the symmetric window instead.
    difference.metadata['kind'] = 'difference'
    fig, ax = plot_field(difference, ax, env=env, vmin=vmin, vmax=vmax,
                         cmap='RdBu_r', title=title, **mpl_kw)
    # The registry says 'Difference (dB)'; this panel knows which difference it
    # drew and which way its sign runs, so it says that instead. Which way is
    # not a constant: more of a LOSS is quieter, more of a LEVEL is louder, so
    # the sense is read off the view rather than assumed to be transmission
    # loss.
    sense = 'quieter' if _is_loss_view(field, 'dB') else 'louder'
    label = (f'Δ{_value_label(field, "dB")} — positive: field is {sense} '
             f'than reference')
    for artist in ax.collections + ax.images:
        cbar = getattr(artist, 'colorbar', None)
        if cbar is not None:
            cbar.set_label(label)
    return fig, ax


def _rms_between(field, reference, depth):
    """RMS dB difference at ``depth``, over the ranges both fields computed.

    ``NaN`` when they share no range at all — nothing to compare, which is
    not the same as agreeing.
    """
    tl_a = np.asarray(field.at(depth=depth).dB)
    tl_b = np.asarray(reference.at(depth=depth).dB)
    r_a = np.asarray(field.ranges, dtype=float)
    r_b = np.asarray(reference.ranges, dtype=float)
    # The common grid is the coarser axis clipped to the shared span.
    # ``np.interp`` reproduces a node exactly, NaN included, so an already
    # aligned pair is untouched; the clip keeps the flat extrapolation
    # ``np.interp`` does past the ends of its own domain out of the number.
    common = r_a if r_a.size <= r_b.size else r_b
    common = common[(common >= max(r_a[0], r_b[0]))
                    & (common <= min(r_a[-1], r_b[-1]))]
    if common.size == 0:
        return np.nan
    residual = np.interp(common, r_a, tl_a) - np.interp(common, r_b, tl_b)
    finite = np.isfinite(residual)
    return (float(np.sqrt(np.mean(residual[finite] ** 2)))
            if finite.any() else np.nan)


@typed_plot_error
def plot_field_statistics(
    fields,
    labels: Optional[Sequence[str]] = None,
    *,
    depth: float,
    figsize: Tuple[float, float] = (12, 5),
    title: Optional[str] = None,
):
    """Two panels summarising several fields at one depth.

    Left: mean and standard deviation of each field's dB view along the cut.
    Right: the pairwise RMS difference between them — how far apart the
    fields actually are, one number per pair.

    Parameters
    ----------
    fields : sequence of Field, or {label: Field} dict
        The shapes :func:`compare_models` takes. A ``None`` entry is dropped,
        so a comparison whose model did not run still plots the ones that did.
    labels : sequence of str, optional
        Per-field names; taken from a dict's keys when it is one.
    depth : float
        The receiver depth to cut every field at, in metres.
    figsize : tuple, optional
    title : str, optional
        Titles the whole figure.

    Returns
    -------
    fig, axes : Figure, ndarray of Axes
        The two axes, bar chart first.

    Notes
    -----
    Each off-diagonal cell is the RMS over the ranges both fields actually
    computed — the coarser range axis, clipped to the span the two share. A
    pair sharing no range has nothing to compare and is drawn as a neutral
    tile labelled ``n/a``; the diagonal is a true zero and takes the
    colormap's own deep green, so "not comparable" cannot be misread as
    perfect agreement.
    """
    if isinstance(fields, dict):
        if labels is None:
            labels = list(fields.keys())
        fields = list(fields.values())
    fields = list(fields)
    if labels is None:
        labels = [getattr(f, 'model', '') or f"#{i}"
                  for i, f in enumerate(fields)]
    elif len(labels) != len(fields):
        raise ConfigurationError(
            f"plot_field_statistics: got {len(labels)} labels for "
            f"{len(fields)} fields — they must match.")
    kept = [(name, f) for name, f in zip(labels, fields) if f is not None]
    if not kept:
        raise ConfigurationError(
            "plot_field_statistics: no fields to plot.",
            remediation="Pass at least one Field; entries that are None are "
                        "dropped as models that did not run.")
    names = [name for name, _ in kept]
    kept_fields = [f for _, f in kept]

    fig, axes = plt.subplots(1, 2, figsize=figsize, squeeze=False)
    axes = axes[0]

    cuts = [np.asarray(f.at(depth=depth).dB) for f in kept_fields]
    means = [float(np.nanmean(c)) for c in cuts]
    stds = [float(np.nanstd(c)) for c in cuts]
    x = np.arange(len(names))
    width = 0.35
    axes[0].bar(x - width / 2, means, width, label='Mean', alpha=0.8)
    axes[0].bar(x + width / 2, stds, width, label='Std', alpha=0.8)
    axes[0].set_xlabel('Field', fontweight='bold')
    axes[0].set_ylabel(f'{_value_label(kept_fields[0], "dB")}',
                       fontweight='bold')
    axes[0].set_title(f'Level at {depth:g} m', fontweight='bold', fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    n = len(kept_fields)
    if n < 2:
        axes[1].text(0.5, 0.5, 'Need at least 2 fields\nfor an RMS comparison',
                     ha='center', va='center', transform=axes[1].transAxes,
                     fontsize=12)
        axes[1].axis('off')
    else:
        rms = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    rms[i, j] = _rms_between(kept_fields[i], kept_fields[j],
                                             depth)
        comparable = rms[np.isfinite(rms)]
        limit = (max(10.0, float(np.percentile(rms[rms > 0], 95)))
                 if comparable.size and comparable.max() > 0 else 15.0)
        # Only a pair with no shared range is masked. The diagonal stays a
        # real 0.0 and lands on the colormap's own deep green.
        cmap = plt.get_cmap('RdYlGn_r').with_extremes(bad='0.85')
        image = axes[1].imshow(np.ma.masked_invalid(rms), cmap=cmap,
                               vmin=0, vmax=limit, interpolation='none')
        fig.colorbar(image, ax=axes[1], label='RMS difference (dB)')
        axes[1].set_xticks(range(n))
        axes[1].set_yticks(range(n))
        axes[1].set_xticklabels(names, rotation=45, ha='right')
        axes[1].set_yticklabels(names)
        axes[1].set_title('Pairwise agreement', fontweight='bold', fontsize=12)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                value = rms[i, j]
                if not np.isfinite(value):
                    axes[1].text(j, i, 'n/a', ha='center', va='center',
                                 color='0.3', fontweight='bold')
                    continue
                axes[1].text(j, i, f'{value:.1f}', ha='center', va='center',
                             color='white' if value > limit / 2 else 'black',
                             fontweight='bold')
    if title:
        fig.suptitle(title, fontweight='bold')
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95 if title else 1.0))
    _draw_multi_model_credit(fig, kept_fields)
    return fig, axes


@typed_plot_error
def _plot_field_stack(stack, env: Optional[Environment] = None, *,
                      ncols: Optional[int] = None,
                      title: Optional[str] = None,
                      figsize: Optional[Tuple[float, float]] = None, **kwargs):
    """Grid of TL panels, one per slab of a Field :class:`ResultStack`.

    Each panel is a :func:`plot_field` heatmap titled by the slab's stacking
    coordinate (e.g. ``source_depth=20``); ``title`` titles the whole figure.
    Extra kwargs forward to :func:`plot_field`.
    """
    n = len(stack)
    ncols = ncols or min(n, 3)
    nrows = int(np.ceil(n / ncols))
    figsize = figsize or (5.5 * ncols, 4.0 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    flat = axes.ravel()
    for i, (coord, slab) in enumerate(stack):
        plot_field(slab, ax=flat[i], env=env, **kwargs)
        flat[i].set_title(f"{stack.coordinate_name}={coord:g}")
    for j in range(n, len(flat)):
        flat[j].axis('off')
    if title:
        fig.suptitle(title, fontweight='bold')
    # After the suptitle, and leaving it room: tight_layout before it let the
    # figure title overprint the middle panel's title on three or more slabs.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95 if title else 1.0))
    _draw_result_credit(fig, stack.slabs[0], env=env)
    return fig, axes
