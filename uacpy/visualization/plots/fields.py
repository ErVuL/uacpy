"""Field plotters: auto-shape heatmaps/line cuts, signal excess, detection probability, and the compare/compare_models overlays."""

from __future__ import annotations

import numpy as np
import matplotlib.collections as _mcoll
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple

from uacpy.core.environment import Environment
from uacpy.core.exceptions import (
    ConfigurationError, NumericsWarning, ValidityWarning,
)
from uacpy.core.metrics import tl_rmse_on_shared_ranges
from uacpy.core.results import Field
from uacpy.core.units import m_to_km
from uacpy.visualization.style import (cmap_for_field, reversed_cmap,
                                      PROBABILITY_COLORMAP,
                                      PROBABILITY_LIMITS)
from uacpy.visualization.plots._common import _value_array, _value_label, _time_trace_label, _default_value, _coord_label, _coord_axis, _TL_LIMITS, _is_loss_view, _overlay_seafloor, _pinned_subtitle, _draw_result_credit, _draw_credit, _model_attribution, fig_ax, invert_yaxis_once, _draw_geometry, typed_plot_error, _plot_warn, _title_or, _fit_subplot_margins, _fit_colorbar_strip, _grid_figure, _credit_attributions


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

# Contour-label unit, keyed by ``value``. Linear pressure ('magnitude', 'real',
# 'imag') carries no unit, so its labels are bare numbers.
_CONTOUR_FMT = {'dB': '%g dB', 'level': '%g dB', 'phase': '%g rad'}

# ``value`` modes that render a dB view of the quantity, and so take its dB
# colormap (``cmap_for_field(kind, dB=True)``). Every other mode is a linear
# view: 'real'/'imag' share one signed colormap (``style.LINEAR_VIEW_COLORMAP``)
# and 'magnitude' takes the ordered map 'level' takes.
# ``plot_field`` and ``compare_models`` both key on this, so one field renders
# the same through either entry point.
from uacpy.core.acoustics.levels import no_energy_mask


#: Figure fraction tight_layout keeps below a figure title on a
#: multi-panel grid.
_TITLE_ROOM_TOP = 0.95

_DB_VALUES = ('dB', 'level')

#: Views that draw the carrier itself rather than its envelope, so a grid
#: coarser than half a wavelength draws structure the field does not have.
#: ``Field.check_sampling('phase')`` is the check; it is keyed here so
#: ``plot_field`` and ``compare`` cannot drift on which views need it.
_PHASE_VALUES = ('phase', 'real', 'imag')

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
        ``'dB'``, ``'level'`` (``20·log10|H|``), ``'magnitude'``, ``'phase'``,
        ``'real'``, ``'imag'``. Defaults to ``'real'`` for a time-series
        field and for a real dimensionless one (a detection probability,
        ``unit='1'``), ``'dB'`` otherwise.
    vmin, vmax : float, optional
        Colour limits (2-D heatmap only). What an unset limit falls back to
        depends on the quantity, since only some of them have a window that
        means something: a **pressure** field's ``value='dB'`` view takes the
        fixed 20–120 dB TL scale (``_TL_LIMITS``), so TL panels stay directly
        comparable across models, frequencies and runs; **signal excess**
        (and a ``kind='difference'`` residual) takes a window symmetric about
        its 0 dB boundary; a **probability** takes a fixed [0, 1]; and
        everything else, including reverberation and ``value='level'``,
        autoscales.
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
    return _draw_field(
        field, ax, who='plot_field', env=env, source=source,
        receiver=receiver, value=value, vmin=vmin, vmax=vmax, cmap=cmap,
        title=title, label=label, figsize=figsize, stacked=stacked,
        stack_offset=stack_offset, show_colorbar=show_colorbar,
        contours=contours, **mpl_kw)


def _drop_singleton_axes(field, *, keep_cross_section=False):
    """``field`` with its length-1 axes moved onto ``.pinned``.

    A length-1 axis has no neighbours for ``shading='nearest'`` to build cell
    edges from, so every heatmap quad collapses to zero extent and the panel
    renders empty — indistinguishable from an all-NaN field. Models pass the
    receiver axes through verbatim and ``Receiver`` keeps a scalar as a
    length-1 array, so a single-receiver-depth run reaches this. Dropping the
    singleton axes (the reduction ``_reduce_to_spectrum`` performs) makes a
    ``(1, n)`` field the 1-D cut it is. At least one axis is always kept.

    ``keep_cross_section`` keeps a ``(depth, range)`` plane whole: an
    ``env`` / ``source`` / ``receiver`` overlay draws over that plane, and a
    line cut would leave it nowhere to go. ``_plot_field_2d`` gives the
    surviving length-1 axis an explicit cell band so the row still renders.

    Shared by :func:`plot_field`, :func:`compare` and :func:`compare_models`,
    so one field reduces to the same shape through each of them."""
    for axis in ('source_depth', 'frequency', 'depth', 'range', 'time'):
        if len(field.coords) <= 1:
            break
        if keep_cross_section and list(field.coords) == ['depth', 'range']:
            break
        if axis in field.coords and field.coords[axis].size == 1:
            field = field.isel(**{axis: 0})
    return field


def _draw_field(field, ax=None, *, who, env=None, source=None, receiver=None,
                value=None, vmin=None, vmax=None, cmap=None, title=None,
                label=None, figsize=(10, 5), stacked=False, stack_offset=None,
                show_colorbar=None, contours=None, **mpl_kw):
    """The body of :func:`plot_field`, with ``who`` naming the public call a
    refusal is reported against: ``compare_models`` and
    ``plot_field_difference`` draw their panels here, and a refusal from one
    of their panels names them, not a ``plot_field`` the user never called."""
    if not isinstance(field, Field):
        raise ConfigurationError(
            f"{who}: expected Field, got {type(field).__name__}."
        )

    if not stacked:
        field = _drop_singleton_axes(
            field, keep_cross_section=(env is not None or source is not None
                                       or receiver is not None))

    if value is None:
        value = _default_value(field)
    if value in _PHASE_VALUES:
        field.check_sampling('phase', where=f"{who}(value={value!r})")
    arr, value_label = _value_array(field, value, who=who)
    axes_present = list(field.coords)
    n_axes = len(axes_present)

    if stacked:
        if n_axes != 2 or 'time' not in axes_present:
            raise ConfigurationError(
                f"{who}(stacked=True): requires a 2-D field with a "
                f"'time' axis; got coords {axes_present}."
            )
        branch = 'stacked'
    elif n_axes == 1:
        branch = 'line'
    elif n_axes == 2:
        branch = 'heatmap'
    else:
        raise ConfigurationError(
            f"{who}: cannot plot a {n_axes}-axis field (coords "
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
            f"{who}: {', '.join(f'{k}=' for k in unused)} has no effect on "
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
            f"{who}: contours= needs at least 2 samples on both axes; "
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
    # stride ``plot_band_levels`` uses on its band axis. Every trace is
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
    if title is not None:
        ax.set_title(title)
    else:
        pin_text = _pinned_subtitle(field)
        if pin_text:
            ax.set_title(pin_text)
    if label:
        ax.legend()
    return fig, ax


def _seafloor_depth_km(env, r_km):
    """Seafloor depth (m) under ranges ``r_km`` (km), the polyline the overlay
    draws: linear between bathymetry samples, constant beyond them."""
    if env.bathymetry.varies_with_range:
        return np.interp(r_km, m_to_km(env.bathymetry.ranges),
                         env.bathymetry.depths)
    return np.full(np.shape(r_km), float(env.depth))


def _extend_water_cells_to_seafloor(ax, mesh, x_km, depths, Z, env):
    """Carry each range column's deepest water cell down to the seabed line.

    A model masks a receiver whose centre lies below the seafloor, and each
    drawn cell reaches only half a depth step past its centre, so where the
    seabed falls between the last water centre and the masked one the strip
    down to the drawn seabed line is painted blank. Per column, the deepest
    finite cell above a masked one is extended as a polygon from its lower
    edge down to the seabed line, never past the masked cell's centre, in the
    cell's own value and on the mesh's colormap and norm. Nothing is drawn
    for a column with no masked cell under its water (a receiver grid that
    stops above the seabed stays where it stops), and no masked value is
    drawn."""
    x_km = np.asarray(x_km, dtype=float)
    depths = np.asarray(depths, dtype=float)
    Z = np.ma.filled(np.ma.masked_invalid(np.asarray(Z, dtype=float)),
                     np.nan)
    if x_km.size < 2 or depths.size < 2 or np.any(np.diff(depths) <= 0):
        return None
    x_edges = _cell_edges(x_km, 0.0)
    z_edges = _cell_edges(depths, 0.0)
    bathy_km = (m_to_km(np.asarray(env.bathymetry.ranges, dtype=float))
                if env.bathymetry.varies_with_range else np.array([]))
    polygons, values = [], []
    for j in range(x_km.size):
        finite = np.flatnonzero(np.isfinite(Z[:, j]))
        if finite.size == 0 or finite[-1] + 1 >= depths.size:
            continue
        i = finite[-1]
        top, floor = z_edges[i + 1], depths[i + 1]
        xl, xr = x_edges[j], x_edges[j + 1]
        xs = np.concatenate(([xl], bathy_km[(bathy_km > xl) & (bathy_km < xr)],
                             [xr]))
        bottom = np.clip(_seafloor_depth_km(env, xs), top, floor)
        if not np.any(bottom > top):
            continue
        polygons.append(np.column_stack((
            np.concatenate(([xl, xr], xs[::-1])),
            np.concatenate(([top, top], bottom[::-1])))))
        values.append(Z[i, j])
    if not polygons:
        return None
    patch = _mcoll.PolyCollection(polygons, array=np.asarray(values),
                                  cmap=mesh.get_cmap(), norm=mesh.norm,
                                  linewidths=0.0, antialiaseds=False,
                                  zorder=mesh.get_zorder())
    ax.add_collection(patch, autolim=False)
    return patch


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
        if env.bathymetry.varies_with_range:
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
    budget = field.sonar_budget or {}
    mode = budget.get('mode')
    pin = _pinned_subtitle(field)
    auto = 'Signal excess' + (f' ({mode})' if mode else '')
    return f"{auto} — {pin}" if pin else auto


def _symmetric_span(datasets):
    """``(-max|x|, +max|x|)`` pooled over every array in ``datasets``, or
    ``(None, None)`` when none of them holds a finite non-zero sample. A
    no-energy marker cell (:func:`no_energy_mask`) is not a level and takes
    no part in the window.

    ``np.abs()`` rather than a float cast: a signal-excess field is normally
    real dB, but the cast warns and discards the imaginary part on the complex
    case, and the magnitude is what the symmetric window needs either way."""
    span = 0.0
    for data in datasets:
        mag = np.abs(np.asarray(data))
        levels = mag[np.isfinite(mag) & ~no_energy_mask(mag)]
        if levels.size:
            span = max(span, float(levels.max()))
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
    if value in ('level', 'magnitude'):
        # A dB view of |H|, so larger is LOUDER — the opposite of the loss
        # the dB colormap is built for. Measured under the unreversed map:
        # the loudest water came out dark blue at -20 dB while the same cell
        # reads dark red through ``dB``, against style.py's stated
        # convention that low TL (loud, near) is red.
        # The linear modulus is the same unsigned quantity on a linear scale,
        # so it takes the same ordered map. The signed diverging map put its
        # neutral white at half the peak, a level that marks nothing, and
        # spent its blue half on the quietest water.
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
    # 'real' / 'imag' are signed linear views, which share one diverging
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
        # wavefront — peaks saturate, which is the intent for a moveout
        # reading.
        if vmin is None or vmax is None:
            finite = np.abs(Z[np.isfinite(Z)])
            if finite.size:
                rms = float(np.sqrt(np.mean(finite ** 2)))
                peak = rms if rms > 0 else float(finite.max())
            else:
                peak = 1.0
            vmin = -peak if vmin is None else vmin
            vmax = peak if vmax is None else vmax
        value_label = _time_trace_label(field)
    elif value in _DB_VALUES and (vmin is None or vmax is None):
        # A no-energy cell is a MARKER, not a level: the package writes
        # ``PRESSURE_FLOOR`` where the model reported no energy, so it lands
        # 600 dB out and drags the bar with it — measured, ``level`` ran
        # -600..-20 and a loss view 20..600, each packing every real level
        # into a tenth of the scale. Reading it by MAGNITUDE covers both
        # views, which carry the same cell at opposite signs. Nothing the
        # model computed reaches that far, so this needs no threshold and no
        # percentile — and unlike a percentile it keeps a genuine deep null,
        # which is the feature the view exists to show: on a 1/r field with
        # one marked cell and a real -70 dB null, 580 dB of bar becomes 70,
        # where the 1st percentile would have cut the null off at -80 dB.
        finite = Z[np.isfinite(Z)]
        levels = finite[~no_energy_mask(finite)]
        if levels.size:
            vmin = float(levels.min()) if vmin is None else vmin
            vmax = float(levels.max()) if vmax is None else vmax
    elif value in ('magnitude', 'real', 'imag') and (vmin is None or vmax is None):
        # Zero has to land at a fixed place on the map, so the signed views
        # take symmetric limits (zero on the diverging map's white) and the
        # non-negative modulus starts at 0 (zero at the ordered map's quiet
        # end). An autoscale puts zero at an arbitrary colour instead.
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
        lo = 0.0 if value == 'magnitude' else -span
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
        ax.clabel(cs, inline=True, fontsize='small',
                  fmt=_CONTOUR_FMT.get(value, '%g'))
    if show_colorbar:
        fig.colorbar(im, ax=ax, label=value_label,
                     fraction=0.046, pad=0.02)
    ax.grid(True, alpha=0.3, zorder=0)
    if title is not None:
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
            _overlay_seafloor(ax, env, x_coord, painted=im)
            _extend_water_cells_to_seafloor(ax, im, x_plot, y_plot, Z, env)
        if source is not None or receiver is not None:
            # Range is measured from the source, so the source sits at r = 0
            # even when the field's own grid starts further out.
            _draw_geometry(ax, source, receiver, max_markersize=6,
                           source_range_m=0.0, env=env)
    return fig, ax


def _refuse_colour_limits(who, mpl_kw, names, why):
    """Refuse a colour limit this sonar map fixes itself, naming the keyword
    and the door that takes it, instead of letting it reach
    ``_begin_sonar_heatmap`` a second time as a bare ``TypeError``."""
    given = [name for name in names if name in mpl_kw]
    if given:
        raise ConfigurationError(
            f"{who}: {', '.join(f'{n}=' for n in given)} is not a keyword of "
            f"this plotter — {why}.",
            remediation="Call plot_field(field, vmin=..., vmax=...) for "
                        "a free colour window.")


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
                          show_colorbar, title, auto_title, owns_fig,
                          source=None, receiver=None):
    """Close a ``(depth, range)`` sonar panel: colorbar, axis labels, depth
    downward, title, seafloor and geometry overlays, and the data credit.

    The colorbar label and the automatic title are the caller's, because each
    plotter is the dedicated view of one quantity and names it itself.

    ``source`` / ``receiver`` draw the same markers :func:`plot_field` does.
    A sonar panel answers "would this array hear that target", so the two
    things it is about belong on it on the same footing as the seafloor."""
    if show_colorbar:
        fig.colorbar(im, ax=ax, label=colorbar_label,
                     fraction=0.046, pad=0.02)
    ax.set_xlabel(x_label)
    ax.set_ylabel(_coord_label('depth'))
    invert_yaxis_once(ax)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.set_title(_title_or(title, auto_title))
    if env is not None:
        _overlay_seafloor(ax, env, field.coords['range'], painted=im)
        _extend_water_cells_to_seafloor(
            ax, im, m_to_km(field.coords['range']), field.coords['depth'],
            np.real(np.asarray(field.data)), env)
    if source is not None or receiver is not None:
        _draw_geometry(ax, source=source, receiver=receiver, env=env)
    if owns_fig:                         # credit only a figure we own
        _draw_result_credit(fig, field, env=env)
    return fig, ax


@typed_plot_error
def plot_signal_excess(
    field: Field,
    ax=None,
    *,
    env: Optional[Environment] = None,
    source=None,
    receiver=None,
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
    source, receiver : Source / Receiver, optional
        Draw the run geometry over the heatmap, with the markers
        :func:`plot_field` uses.
    show_colorbar : bool, optional
        Draw the colorbar. Default True.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(10, 5)`` by default; unused when
        ``ax`` is given.
    """
    if not isinstance(field, Field):
        raise ConfigurationError(
            f"plot_signal_excess: expected Field, got {type(field).__name__}."
        )
    if field.is_complex:
        raise ConfigurationError(
            "plot_signal_excess: field must carry real signal excess in "
            "dB — build it with passive_signal_excess_field / "
            "active_signal_excess_field."
        )
    if field.kind != 'signal_excess':
        raise ConfigurationError(
            f"plot_signal_excess: this is a {field.kind!r} field, not signal "
            f"excess — it would be drawn and captioned as one.",
            remediation="Build it with passive_signal_excess_field / "
                        "active_signal_excess_field (they tag "
                        "kind='signal_excess'), or draw this field with "
                        "plot_field.")
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
    _refuse_colour_limits(
        'plot_signal_excess', mpl_kw, ('vmin',),
        "the map is symmetric about SE = 0 dB and takes vmax= only")

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
                    f"range={Z.shape[1]}, so no boundary contour is drawn.", NumericsWarning)
            else:
                cs = ax.contour(
                    r_km, depths, Z, levels=[0.0],
                    colors='black', linewidths=1.5, linestyles='solid',
                )
                ax.clabel(cs, inline=True, fontsize='small',
                          fmt=lambda _: 'SE = 0 dB')
    return _finish_sonar_heatmap(
        fig, ax, im, field, env=env, x_label=x_label,
        colorbar_label='Signal excess (dB)', show_colorbar=show_colorbar,
        title=title, auto_title=_signal_excess_title(field),
        owns_fig=_owns_fig, source=source, receiver=receiver)


@typed_plot_error
def plot_detection_probability(
    field: Field,
    ax=None,
    *,
    env: Optional[Environment] = None,
    source=None,
    receiver=None,
    cmap: str = PROBABILITY_COLORMAP,
    contour_levels: Sequence[float] = (0.1, 0.5, 0.9),
    show_colorbar: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
    **mpl_kw,
):
    """Heatmap of a detection-probability :class:`Field` over ``(depth, range)``.

    Renders the output of
    :func:`uacpy.sonar.transition_probability_field` on a fixed
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
    source, receiver : Source / Receiver, optional
        Draw the run geometry over the heatmap, with the markers
        :func:`plot_field` uses.
    show_colorbar : bool, optional
        Draw the colorbar. Default True.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(10, 5)`` by default; unused when
        ``ax`` is given.
    """
    if not isinstance(field, Field):
        raise ConfigurationError(
            f"plot_detection_probability: expected Field, got "
            f"{type(field).__name__}."
        )
    if field.is_complex:
        raise ConfigurationError(
            "plot_detection_probability: field must carry real P_D in "
            "[0, 1] — build it with transition_probability_field."
        )
    if field.kind != 'probability_of_detection':
        raise ConfigurationError(
            f"plot_detection_probability: this is a {field.kind!r} field, "
            f"not a detection probability — on the fixed [0, 1] scale a dB "
            f"grid saturates to a uniform 'P_D = 1'.",
            remediation="Build it with transition_probability_field (it "
                        "tags kind='probability_of_detection'), or draw this "
                        "field with plot_field.")
    if list(field.coords) != ['depth', 'range']:
        raise ConfigurationError(
            "plot_detection_probability: requires canonical "
            f"['depth', 'range'] coords; got {list(field.coords)} — "
            "slice with .at(...) first."
        )
    _refuse_colour_limits(
        'plot_detection_probability', mpl_kw, ('vmin', 'vmax'),
        "a probability is drawn on the fixed [0, 1] scale")

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
                f"range={Z.shape[1]}, so no contours are drawn.", NumericsWarning)
        elif levels:
            cs = ax.contour(
                r_km, depths, Z, levels=levels,
                colors='black', linewidths=1.2, linestyles='solid',
            )
            ax.clabel(cs, inline=True, fontsize='small', fmt='%.1f')
    sigma = field.sigma_dB
    pin = _pinned_subtitle(field)
    auto = 'Detection probability'
    if sigma is not None:
        auto += f' (σ = {sigma:g} dB)'
    return _finish_sonar_heatmap(
        fig, ax, im, field, env=env, x_label=x_label,
        colorbar_label='Probability of detection',
        show_colorbar=show_colorbar, title=title,
        auto_title=f"{auto} — {pin}" if pin else auto, owns_fig=_owns_fig,
        source=source, receiver=receiver)


def _labelled_fields(fields, labels, who, *, each=''):
    """``(fields, labels)`` from a list of Fields or a ``{label: Field}``
    dict, the form every multi-field plotter takes.

    A dict supplies the labels from its keys unless ``labels`` is given.
    ``None`` entries are dropped with their labels, as models that did not
    run, so one comparison can be written for a set whose members may be
    missing. Default labels are each field's ``model``. ``each`` is appended
    to the empty-input message to say what every field must already be."""
    if isinstance(fields, dict):
        if labels is None:
            labels = list(fields.keys())
        fields = list(fields.values())
    fields = list(fields)
    if labels is not None and len(labels) != len(fields):
        raise ConfigurationError(
            f"{who}: got {len(labels)} labels for {len(fields)} fields — "
            f"they must match (panels are labelled by zipping the two).")
    if labels is None:
        labels = [getattr(f, 'model', '') or f"#{i}"
                  for i, f in enumerate(fields)]
    kept = [(lbl, f) for lbl, f in zip(labels, fields) if f is not None]
    if not kept:
        raise ConfigurationError(
            f"{who}: no fields to plot — pass one Field per model, as a list "
            f"or as a {{label: field}} dict{each}.",
            remediation="Pass at least one Field; entries that are None are "
                        "dropped as models that did not run.")
    return [f for _, f in kept], [lbl for lbl, _ in kept]


@typed_plot_error
def compare(
    fields,
    labels: Optional[Sequence[str]] = None,
    ax=None,
    *,
    value: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
    title: Optional[str] = None,
    **mpl_kw,
):
    """Overlay multiple 1-D sliced :class:`Field` instances on one axes.

    Every field must reduce to a single surviving coord axis (the same
    axis across all). Caller slices them first::

        compare([f1.at(depth=20), f2.at(depth=20)], labels=['Bellhop', 'RAM'])
        compare({'Bellhop': f1.at(depth=20), 'RAM': f2.at(depth=20)})

    ``fields`` is a list, or a ``{label: Field}`` dict as
    :func:`compare_models` takes; ``None`` entries are dropped as models that
    did not run.

    Axes follow :func:`plot_field`, so one field cuts the same way through
    either: depth increases downward, and so does the value axis of a loss
    cut — transmission loss or reverberation, see :func:`_is_loss_view` — but
    not that of any other dB quantity, which is a level and reads upward.
    ``value=None`` picks the view :func:`plot_field` would pick for the
    first field on its own (:func:`_default_value`): the dB view wherever
    one exists, the raw samples of a time trace.

    Parameters
    ----------
    fields : list of Field or dict
        1-D cuts, or ``{label: Field}``; ``None`` entries are dropped.
    labels : sequence of str, optional
        Legend label of each field in a list.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    value : str, optional
        The view (see above); ``None`` is :func:`plot_field`'s choice.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(10, 5)`` by default; unused when
        ``ax`` is given.
    title : str, optional
        Axes title; none is drawn when unset.
    **mpl_kw
        Matplotlib keywords for every line.
    """
    fields, labels = _labelled_fields(
        fields, labels, 'compare',
        each=', each already cut to one surviving axis (field.at(...))')
    # Length-1 axes drop onto .pinned exactly as plot_field drops them, so a
    # single-receiver-depth run compares as the range cut plot_field draws.
    fields = [_drop_singleton_axes(f) if isinstance(f, Field) else f
              for f in fields]
    _owns_fig = ax is None
    fig, ax = fig_ax(ax, figsize)
    common_axis = None
    x_label = None
    for f, lbl in zip(fields, labels):
        if not isinstance(f, Field):
            raise ConfigurationError(
                f"compare: expected Field, got {type(f).__name__}."
            )
        # Compare the QUANTITY, exactly as compare_models does: overlaying a
        # reverberation loss on a TL cut puts two different physical
        # quantities on one value axis, with one shared label — even though
        # both run in the same direction.
        if f.kind != fields[0].kind:
            raise ConfigurationError(
                f"compare: {lbl!r} is a {f.kind!r} field but "
                f"{labels[0]!r} is {fields[0].kind!r} — these are different "
                f"physical quantities and share no value axis.",
                remediation="Compare like with like, or plot them separately "
                            "with plot_field.")
        if value is None:
            # Every field shares the first one's kind, so its default view
            # is the shared view.
            value = _default_value(fields[0])
        if value in _PHASE_VALUES:
            f.check_sampling('phase', where=f"compare(value={value!r})")
        axes = list(f.coords)
        if len(axes) != 1:
            raise ConfigurationError(
                f"compare: each field must have exactly 1 surviving axis; "
                f"{lbl!r} has {axes}."
            )
        if common_axis is None:
            common_axis = axes[0]
        elif axes[0] != common_axis:
            raise ConfigurationError(
                f"compare: axis mismatch — {labels[0]!r} on {common_axis!r}, "
                f"{lbl!r} on {axes[0]!r}."
            )
        arr, _ = _value_array(f, value, who='compare')
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
    # direction that axis runs. ``view`` is the resolved name of the view the
    # loop drew (``fields`` is non-empty, so it is never the default here).
    view: str = value if value is not None else _default_value(fields[0])
    vlabel = _value_label(fields[0], view, who='compare')
    value_is_loss = _is_loss_view(fields[0], view)
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


def _model_attributions(fields):
    """Every distinct model credit among ``fields``, in order."""
    seen, attrs = set(), []
    for f in fields:
        a = _model_attribution(f)
        if a and a not in seen:
            seen.add(a)
            attrs.append(a)
    return attrs


def _draw_multi_model_credit(fig, fields):
    """One credit footnote listing every distinct contributing model."""
    attrs = _model_attributions(fields)
    if attrs:
        _draw_credit(fig, (), model=attrs)


@typed_plot_error
def compare_models(
    fields,
    labels: Optional[Sequence[str]] = None,
    *,
    env: Optional[Environment] = None,
    source=None,
    receiver=None,
    value: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    ncols: Optional[int] = None,
    contours: Optional[Sequence[float]] = None,
    fig=None,
    **mpl_kw,
):
    """Side-by-side heatmaps of several 2-D :class:`Field` instances.

    ``fields`` is either a list of :class:`Field` (then ``labels`` is
    used as the per-axes title), or a ``{label: Field}`` dict; ``None``
    entries are dropped as models that did not run. Shared
    colour scale, with one figure-level colorbar for the whole grid on a
    strip outside the panels — no panel draws its own, so calling
    :func:`shared_colorbar` on the result would add a second.

    ``title`` titles the whole figure; the per-panel titles come from
    ``labels``. ``ncols`` controls the grid width — defaults to ``n`` (single
    row). ``contours`` adds dB-level contour lines to every panel.
    ``value=None`` picks the view :func:`plot_field` would pick for the first
    panel on its own (:func:`_default_value`): the dB view wherever one
    exists, the raw samples of a time-domain wavefield. ``source`` and
    ``receiver`` draw the run geometry on **every** panel, exactly as
    :func:`plot_field` draws it — two models of the same scene should carry
    the same markers, and a comparison that hides what is listening to what
    cannot be read as a coverage map.

    Every field must be a 2-D heatmap once its length-1 axes are dropped, as
    :func:`plot_field` drops them; fields that reduce to one axis (a cut, a
    single-receiver-depth run) are refused with a pointer to :func:`compare`,
    which overlays 1-D cuts.

    ``fig`` draws the grid into an existing ``Figure`` or ``SubFigure``
    (a panel of a larger publication figure) instead of a new one. The
    caller then owns its layout — give the parent ``layout='constrained'`` —
    and its credit line, so the grid takes a space-stealing colorbar and
    draws no footnote, as a single-panel plotter handed ``ax=`` draws none.
    ``**mpl_kw`` is forwarded to every panel's ``pcolormesh`` (e.g.
    ``rasterized=True`` for a PDF, ``edgecolors=``).

    Parameters
    ----------
    fields : list of Field or dict
        2-D fields, or ``{label: Field}``; ``None`` entries are dropped.
    labels : sequence of str, optional
        Panel title of each field in a list.
    env : Environment, optional
        Overlays the seafloor on every panel.
    source, receiver : Source / Receiver, optional
        Draw the run geometry on every panel.
    value : str, optional
        The view (see above); ``None`` is :func:`plot_field`'s choice.
    vmin, vmax : float, optional
        The shared colour limits; ``None`` takes the view's own, else the
        fields'.
    cmap : str, optional
        Colormap; ``None`` is the view's.
    figsize : tuple, optional
        Size (inches) of the new figure; ``None`` sizes it from the grid.
    title : str, optional
        Figure title; none is drawn when unset.
    ncols : int, optional
        Panels per row; ``None`` is one row.
    contours : sequence of float, optional
        dB levels contoured on every panel.
    fig : Figure or SubFigure, optional
        Draw the grid into this (see above).
    **mpl_kw
        Matplotlib keywords for every panel's ``pcolormesh``.

    Returns
    -------
    fig, axes : Figure, ndarray of Axes
        ``axes`` is the 2-D ``(nrows, ncols)`` array ``plt.subplots``
        produced (unused cells are turned off) — the same shape every
        grid-of-panels helper on this surface returns.
    """
    fields, labels = _labelled_fields(fields, labels, 'compare_models')
    keep_cross_section = (env is not None or source is not None
                          or receiver is not None)
    for f, lbl in zip(fields, labels):
        if not isinstance(f, Field):
            raise ConfigurationError(
                f"compare_models: expected Field, got {type(f).__name__} "
                f"for {lbl!r}.")
        surviving = list(_drop_singleton_axes(
            f, keep_cross_section=keep_cross_section).coords)
        if len(surviving) != 2:
            raise ConfigurationError(
                f"compare_models: draws one 2-D heatmap per field, but "
                f"{lbl!r} reduces to {surviving} once its length-1 axes are "
                f"dropped.",
                remediation="Overlay 1-D cuts with uacpy.plot.compare(...); "
                            "slice fields with more than two axes with "
                            ".at(...) first.")
    n = len(fields)
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
            if not _axes_agree(ref.coords[axis], f.coords[axis]):
                _plot_warn(
                    f"compare_models: {lbl!r} {axis} axis differs from "
                    f"{labels[0]!r}; the shared colourbar mixes "
                    "different sample grids.", NumericsWarning,
                )
                break

    # Every panel here shares one kind already, so ``ref`` settles the view
    # and the styling for the whole figure — both read from the same tables
    # plot_field reads, so a panel is drawn as if it had been plotted on its
    # own.
    if value is None:
        value = _default_value(ref)
    style_cmap, style_vmin, style_vmax = _value_style(ref, value)
    if (value == 'dB' and ref.kind in ('signal_excess', 'difference')
            and not _is_time_domain(ref)):
        # _value_style sizes the symmetric window of a signed dB kind (signal
        # excess, a difference) from the one field it is handed, which here is
        # panel 1: measured on a second panel spanning four times as wide, 70%
        # of its samples saturated, and reversing the list changed the shared
        # limits. Pool the magnitude over every panel instead — still
        # SYMMETRIC, so 0 dB keeps the diverging map's neutral colour. The generic pooled branch below cannot do this
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
        pooled = [np.asarray(_value_array(f, value, who='compare_models')[0],
                             dtype=float).ravel()
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
        elif value == 'magnitude':
            # As in plot_field: the modulus is non-negative and starts at 0,
            # the quiet end of its ordered colormap.
            lo, hi = 0.0, float(np.max(finite)) or 1.0
        else:
            # A dB view: the no-energy marker is not a level and takes no part
            # in the limits, exactly as in plot_field — left in, one zero cell
            # pulls the shared bar out to -600 dB.
            levels = (finite[~no_energy_mask(finite)]
                      if value in _DB_VALUES else finite)
            if levels.size == 0:
                levels = np.array([0.0])
            lo, hi = float(np.min(levels)), float(np.max(levels))
            if hi <= lo:                       # a constant field has no range
                lo, hi = lo - 0.5, hi + 0.5
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax

    fig, axes, owns_fig = _grid_figure(
        fig, nrows, ncols, figsize, (6.0 * ncols + 1.6, 5.0 * nrows + 1.2),
        who='compare_models')
    axes_flat = axes.ravel()
    im_last = None
    for f, label, ax in zip(fields, labels, axes_flat):
        _draw_field(
            f, ax=ax, who='compare_models', env=env, source=source,
            receiver=receiver, value=value, vmin=vmin, vmax=vmax, cmap=cmap,
            title=label, contours=contours, show_colorbar=False, **mpl_kw,
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

    # Only the lowest live panel in each column keeps its x label. ``plot_field``
    # labels every axes it is given, and the layout below uses a fixed
    # ``hspace``, so in a stacked comparison of short wide panels -- a shallow
    # water TL field is exactly that -- row i's x label printed straight over
    # row i+1's title. Walk each column upward from the bottom so that a short
    # final row leaves the panel above it labelled rather than bare.
    for col in range(ncols):
        live = [r for r in range(nrows) if r * ncols + col < n]
        for r in live[:-1]:
            axes[r][col].set_xlabel('')

    # The raw ``value`` string is the knob's name, not the quantity's: the bar
    # carries the label the panels would carry had each drawn its own.
    cbar_label = (_time_trace_label(ref) if _is_time_domain(ref)
                  else _value_label(ref, value, who='compare_models'))
    if not owns_fig:
        # A caller's Figure or SubFigure: its layout engine places the panels,
        # so the bar steals its space from them rather than sitting on a strip
        # measured against a canvas this plotter does not own, and the credit
        # line is the caller's to draw.
        if title:
            fig.suptitle(title, fontsize='x-large', fontweight='bold')
        if im_last is not None:
            fig.colorbar(im_last, ax=list(axes_flat[:n]), label=cbar_label,
                         fraction=0.03, pad=0.02)
        return fig, axes

    top = 0.90 if title else 0.95
    # One credit line per model sits under the panels; leave it room.
    bottom = 0.08 + 0.025 * max(0, len(fields) - 1)
    #: Panels may run to here; the strip beyond it belongs to the colorbar.
    _PANEL_RIGHT = 0.88
    fig.subplots_adjust(left=0.05, right=_PANEL_RIGHT, top=top, bottom=bottom,
                        wspace=0.22, hspace=0.30)
    # Those fractions are only a starting point. They cannot hold a label
    # whose width is in points, so the margins are then grown from the
    # rendered labels -- see _fit_subplot_margins for the measurements.
    _fit_subplot_margins(fig, axes_flat[:n], right_limit=_PANEL_RIGHT)
    if title:
        suptitle = fig.suptitle(title, fontsize='x-large', fontweight='bold',
                                y=0.97)
        # ``top`` is a fraction and the titles are in points, so on a short
        # figure the panel titles reached into the suptitle (measured 9 px
        # at 13 x 4.6 in). Lower the panels until their titles clear it.
        fig.canvas.draw()
        clear = suptitle.get_window_extent().y0 - 4.0
        reach = max(ax.get_tightbbox().y1 for ax in axes_flat[:n])
        if reach > clear:
            fig.subplots_adjust(
                top=fig.subplotpars.top - (reach - clear) / fig.bbox.height)
    _draw_multi_model_credit(fig, fields)
    if im_last is not None:
        # Added AFTER the credit: _draw_multi_model_credit reserves its own
        # margin with a second subplots_adjust, and an axes placed at the
        # earlier ``bottom`` does not follow it. Read the panels' final bottom.
        cbar_bottom = fig.subplotpars.bottom
        cbar_ax = fig.add_axes((_PANEL_RIGHT + 0.025, cbar_bottom, 0.015,
                                fig.subplotpars.top - cbar_bottom))
        fig.colorbar(im_last, cax=cbar_ax, label=cbar_label)
        # Its ticks and label live outside its axes, so it too is measured.
        _fit_colorbar_strip(fig, cbar_ax)
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


#: Fraction of an axis' own smallest cell within which two samples are one.
_GRID_AGREEMENT_FRACTION = 1e-3


def _axes_agree(a, b) -> bool:
    """Whether two coordinate axes sample the same positions.

    The tolerance is a small fraction of the axis' own smallest cell, so it
    scales with the grid: sub-millimetre offsets between two engines' range
    axes on a 40 m grid agree, while a half-cell shift does not, whatever the
    coordinate's magnitude. A length-1 axis has no cell; it compares with
    numpy's relative tolerance. :func:`compare_models` (which warns) and
    :func:`plot_field_difference` (which refuses) ask this one question, so
    one pair of fields gets one answer."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return False
    if a.size < 2:
        return bool(np.allclose(a, b))
    cell = float(np.min(np.abs(np.diff(a))))
    return bool(np.allclose(a, b, rtol=0.0,
                            atol=_GRID_AGREEMENT_FRACTION * cell))


def _same_grid_or_raise(who: str, field: Field, reference: Field) -> None:
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
            f"{who}: the fields are on different grids — "
            f"{axis_lengths(field)} vs {axis_lengths(reference)}.",
            remediation="Run both models on one receiver grid, or resample "
                        "one onto the other's axes before differencing.",
        )
    for name, axis in field.coords.items():
        if not _axes_agree(axis, reference.coords[name]):
            raise ConfigurationError(
                f"{who}: the fields share the shape of their '{name}' "
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
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    diff_vmax: Optional[float] = None,
    cmap: Optional[str] = None,
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
    vmin, vmax : float, optional
        Colour limits of the heatmap; ``None`` on either takes ±10 dB.
    diff_vmax : float, optional
        Symmetric-window shortcut: sets ``vmin, vmax = -diff_vmax, +diff_vmax``.
        Giving it together with ``vmin`` or ``vmax`` is refused, since the two
        forms would name two windows.
    cmap : str, optional
        Colormap. ``None`` takes the diverging map the ``'difference'``
        quantity carries (``RdBu_r``), whose neutral colour marks zero.
    title : str, optional
    **mpl_kw
        Forwarded to :func:`plot_field`.

    Two fields that reduce to one axis (two cuts, or two single-depth runs)
    draw their residual as a line against that axis, as :func:`plot_field`
    draws any 1-D field; the colour knobs above have no colour axis there and
    are refused if given. The difference keeps ``field``'s pinned
    coordinates, so a sliced pair keeps its subtitle, and a figure the
    plotter owns credits both fields' models.

    Returns
    -------
    fig, ax : Figure, Axes

    See Also
    --------
    compare : the same two fields as 1-D cuts, overlaid.
    plot_field_statistics : the residual reduced to one number per pair.
    """
    who = 'plot_field_difference'
    _same_grid_or_raise(who, field, reference)
    if diff_vmax is not None:
        if vmin is not None or vmax is not None:
            raise ConfigurationError(
                "plot_field_difference: diff_vmax= sets the symmetric window "
                "-diff_vmax..+diff_vmax, and vmin=/vmax= set it as well — "
                "give one form.",
                remediation="diff_vmax=20, or vmin=-20, vmax=20.")
        vmin, vmax = -abs(diff_vmax), abs(diff_vmax)

    # Tag it for what it is. Untagged, a Field inherits ``kind='pressure'``,
    # which would caption the bar 'TL (dB)' over a signed residual and let the
    # loss predicate run a 1-D cut's value axis downward — meaningless for a
    # difference, whose zero is the meaningful value. The 'difference'
    # quantity carries the diverging map and the symmetric window instead.
    difference = Field(
        data=(_value_array(field, 'dB', who=who)[0]
              - _value_array(reference, 'dB', who=who)[0]),
        coords=dict(field.coords), pinned=dict(field.pinned or {}),
        kind='difference', unit='dB')
    is_line = len(_drop_singleton_axes(
        difference, keep_cross_section=env is not None).coords) == 1
    if is_line:
        given = [name for name, v in (('vmin', vmin), ('vmax', vmax),
                                      ('diff_vmax', diff_vmax), ('cmap', cmap))
                 if v is not None]
        if given:
            raise ConfigurationError(
                f"{who}: {', '.join(f'{k}=' for k in given)} sets a colour "
                f"window, but these fields reduce to one axis "
                f"({list(_drop_singleton_axes(difference).coords)}), so the "
                f"residual is drawn as a line with no colour axis.",
                remediation="Drop the colour knobs, or difference 2-D fields "
                            "for the heatmap.")
        colour = {}
    else:
        colour = dict(vmin=-10.0 if vmin is None else vmin,
                      vmax=10.0 if vmax is None else vmax, cmap=cmap)
    owns_fig = ax is None
    fig, ax = fig_ax(ax, mpl_kw.pop('figsize', (10, 5)))
    _draw_field(difference, ax, who=who, env=env, title=title, **colour,
                **mpl_kw)
    # The registry says 'Difference (dB)'; this panel knows which difference it
    # drew and which way its sign runs, so it says that instead. Which way is
    # not a constant: more of a LOSS is quieter, more of a LEVEL is louder, so
    # the sense is read off the view rather than assumed to be transmission
    # loss.
    sense = 'quieter' if _is_loss_view(field, 'dB') else 'louder'
    label = (f'Δ{_value_label(field, "dB", who=who)} — positive: field is '
             f'{sense} than reference')
    if is_line:
        # plot_field puts a depth cut's value on x and every other cut's on y.
        (ax.set_xlabel if 'depth' in _drop_singleton_axes(difference).coords
         else ax.set_ylabel)(label)
    for artist in ax.collections + ax.images:
        cbar = getattr(artist, 'colorbar', None)
        if cbar is not None:
            cbar.set_label(label)
    if owns_fig:
        _draw_credit(fig, _credit_attributions(True, carrier=env),
                     model=_model_attributions([field, reference]) or None)
    return fig, ax


def _rms_between(field, reference, depth):
    """RMS dB difference at ``depth``, over the ranges both fields computed.

    ``NaN`` when there is nothing to compare, which is not the same as
    agreeing: either the two share no range at all, or they are different
    physical quantities. The figure masks NaN grey, so a pair that cannot be
    compared reads as "no answer" rather than as a number.

    The resampling this needs — and that :func:`uacpy.metrics.tl_rmse`
    refuses, because comparing models run on different range axes is the
    whole job of this figure — is
    :func:`uacpy.metrics.tl_rmse_on_shared_ranges`, which is public so a
    reader of the table can obtain the number in it. What stays here is
    the kind check: a figure that prints a green agreement cell for a
    reverberation field against a pressure field is a wrong answer with a
    colour on it, and it belongs blank rather than raised.
    """
    # The kind policy stays here because it is a FIGURE decision — a cell
    # left blank and a warning, rather than an exception that would take
    # the whole table down. The arithmetic below it is
    # metrics.tl_rmse_on_shared_ranges, which a reader of this figure can
    # call to get the number they are looking at.
    if field.kind != reference.kind:
        _plot_warn(
            f"plot_field_statistics: a {field.kind!r} field and a "
            f"{reference.kind!r} field are different physical quantities, so "
            f"their RMS difference is not an agreement metric. That cell is "
            f"left blank rather than scored.", ValidityWarning)
        return np.nan
    return tl_rmse_on_shared_ranges(field, reference, depth=depth)


@typed_plot_error
def plot_field_statistics(
    fields,
    labels: Optional[Sequence[str]] = None,
    *,
    depth: float,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    fig=None,
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
        Size of the new figure, ``(12, 5)`` by default.
    title : str, optional
        Titles the whole figure.
    fig : Figure or SubFigure, optional
        Draw the two panels into this (a panel of a larger figure) instead of
        a new figure; its size, layout and credit line are then the
        caller's, as with :func:`compare_models`.

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
    kept_fields, names = _labelled_fields(fields, labels,
                                          'plot_field_statistics')

    fig, axes, owns_fig = _grid_figure(fig, 1, 2, figsize, (12, 5),
                                       who='plot_field_statistics')
    axes = axes[0]

    # A no-energy cell is a marker, not a 600 dB level: NaN it before the
    # reductions, as plot_field does before autoscaling.
    cuts = []
    for f in kept_fields:
        c = np.asarray(f.at(depth=depth).dB, dtype=float)
        cuts.append(np.where(no_energy_mask(c), np.nan, c))
    means = [float(np.nanmean(c)) for c in cuts]
    stds = [float(np.nanstd(c)) for c in cuts]
    x = np.arange(len(names))
    width = 0.35
    axes[0].bar(x - width / 2, means, width, label='Mean', alpha=0.8)
    axes[0].bar(x + width / 2, stds, width, label='Std', alpha=0.8)
    axes[0].set_xlabel('Field', fontweight='bold')
    axes[0].set_ylabel(f'{_value_label(kept_fields[0], "dB")}',
                       fontweight='bold')
    axes[0].set_title(f'Level at {depth:g} m', fontweight='bold', fontsize='large')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    n = len(kept_fields)
    if n < 2:
        axes[1].text(0.5, 0.5, 'Need at least 2 fields\nfor an RMS comparison',
                     ha='center', va='center', transform=axes[1].transAxes,
                     fontsize='large')
        axes[1].axis('off')
    else:
        rms = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if i > j and kept_fields[i].kind != kept_fields[j].kind:
                    # The (j, i) cell already warned for this pair.
                    rms[i, j] = np.nan
                    continue
                rms[i, j] = _rms_between(kept_fields[i], kept_fields[j],
                                         depth)
        comparable = rms[np.isfinite(rms)]
        limit = (max(10.0, float(np.percentile(rms[rms > 0], 95)))
                 if comparable.size and comparable.max() > 0 else 15.0)
        # Masked: a pair with no shared range, and a pair of different
        # kinds — both are "no answer", not agreement. The diagonal stays a
        # real 0.0 and lands on the colormap's own deep green.
        cmap = plt.get_cmap('RdYlGn_r').with_extremes(bad='0.85')
        image = axes[1].imshow(np.ma.masked_invalid(rms), cmap=cmap,
                               vmin=0, vmax=limit, interpolation='none')
        fig.colorbar(image, ax=axes[1], label='RMS difference (dB)')
        axes[1].set_xticks(range(n))
        axes[1].set_yticks(range(n))
        axes[1].set_xticklabels(names, rotation=45, ha='right')
        axes[1].set_yticklabels(names)
        axes[1].set_title('Pairwise agreement', fontweight='bold', fontsize='large')
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
    if owns_fig:
        fig.tight_layout(rect=(0.0, 0.0, 1.0,
                               _TITLE_ROOM_TOP if title else 1.0))
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
    # A source-depth grid is "one source at a time", so each panel marks the
    # source it belongs to. The whole array on every panel would say the
    # opposite, and nothing at all leaves the reader drawing the marker by
    # hand. ``source=`` from the caller wins, and any other stacking
    # coordinate is left alone.
    per_panel_source = (
        stack.coordinate_name == 'source_depth'
        and 'source' not in kwargs
        and all(getattr(s, 'source_depths', None) is not None
                for s in stack.slabs)
    )
    for i, (coord, slab) in enumerate(stack):
        extra = ({'source': np.atleast_1d(slab.source_depths)}
                 if per_panel_source else {})
        plot_field(slab, ax=flat[i], env=env, **extra, **kwargs)
        flat[i].set_title(f"{stack.coordinate_name}={coord:g}")
    for j in range(n, len(flat)):
        flat[j].axis('off')
    if title:
        fig.suptitle(title, fontweight='bold')
    # After the suptitle, and leaving it room: tight_layout before it let the
    # figure title overprint the middle panel's title on three or more slabs.
    fig.tight_layout(rect=(0.0, 0.0, 1.0, _TITLE_ROOM_TOP if title else 1.0))
    _draw_result_credit(fig, stack.slabs[0], env=env)
    return fig, axes


def _reduce_to_spectrum(field, method: str) -> Field:
    """Reduce a broadband Field to a single ``['frequency']`` spectrum.

    Singleton ``depth`` / ``range`` axes are squeezed automatically (so a
    single-receiver field needs no ``.at()``); any remaining non-frequency
    axis means the caller must pick a cell first. Used by the
    transfer-function / impulse-response plot helpers."""
    if 'frequency' not in field.coords:
        raise ConfigurationError(
            f"Field.{method}: needs a broadband field with a 'frequency' "
            f"axis; got coords {list(field.coords)}."
        )
    f = field
    for axis in ('source_depth', 'depth', 'range'):
        if axis in f.coords and f.coords[axis].size == 1:
            f = f.isel(**{axis: 0})
    if list(f.coords) != ['frequency']:
        raise ConfigurationError(
            f"Field.{method}: reduce to one (depth, range) cell first, "
            f"e.g. H.at(depth=…, range=…) — after squeezing singleton axes "
            f"the remaining coords are {list(f.coords)}."
        )
    return f


@typed_plot_error(who='Field.plot_transfer_function')
def plot_transfer_function(field, *, ax=None, title=None,
                           figsize=(8, 6), **kwargs):
    """Plot the transfer function ``H(f)`` at one receiver cell as two
    stacked panels: modulus in dB (``20·log10|H|``, top) over phase
    (bottom), sharing the frequency axis.

    Reduce-then-plot: pass a field already sliced to one ``(depth,
    range)`` cell (``plot_transfer_function(H.at(depth=…, range=…))``); a
    single-receiver field plots directly (singleton axes are squeezed).
    :meth:`Field.plot_transfer_function` is the method form.
    Pass ``ax=(ax_mag, ax_phase)`` to draw into existing axes. This one
    draws two panels, so ``ax`` takes a **pair**: anything that unpacks
    into two Axes, including the ndarray ``plt.subplots(2, 1)`` returns.
    Returns ``(fig, (ax_mag, ax_phase))``.

    What the panels show depends on the frequency grid. Each pair of
    paths ``dtau`` apart puts a fringe of period ``1/dtau`` on
    ``|H(f)|``, and a grid from ``Arrivals.synthesis_band`` places
    ``margin`` samples on it — 1.2 by default, so the modulus is drawn
    critically sampled: a dense oscillation under a beat envelope, real
    multipath interference but interpolated by the plotter between
    samples. Raise ``margin`` to see the fringes. The phase is drawn
    wrapped, and a bulk delay ``tau`` turns it once every ``1/tau`` Hz,
    so over a band far wider than that it fills the panel; take the
    delay out (``H * exp(2j*pi*f*tau)``) to see what remains.

    Parameters
    ----------
    field : Field
        A spectrum reduced to one receiver cell.
    ax : (Axes, Axes), optional
        The ``(ax_mag, ax_phase)`` pair (see above).
    title : str, optional
        Title of the modulus panel. ``None`` draws the default caption.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(8, 6)`` by default; unused when
        ``ax`` is given.
    **kwargs
        Keywords of :meth:`Field.plot` for both panels.
    """
    axes = ax
    if axes is not None:
        # The acceptance test is the two-target unpack this function
        # performs on ``axes`` further down, so it admits exactly what
        # that admits and cannot narrow it: a tuple, a list, the ndarray
        # ``plt.subplots(2, 1)`` actually returns, ``axs.ravel()``,
        # ``axs.flat``. A type test would have to enumerate those.
        try:
            ax_mag, ax_phase = axes
        except (TypeError, ValueError) as exc:
            try:
                given = len(axes)
            except TypeError:
                given = 1
            raise ConfigurationError(
                f"Field.plot_transfer_function: draws two stacked panels, "
                f"so it needs a pair of Axes; got {given}.",
                remediation="Pass ax=(ax_mag, ax_phase) — the second "
                            "return value of "
                            "plt.subplots(2, 1, sharex=True) is one.",
            ) from exc
        axes = (ax_mag, ax_phase)
    spec = _reduce_to_spectrum(field, 'plot_transfer_function')
    if not spec.is_complex:
        raise ConfigurationError(
            "Field.plot_transfer_function: needs a complex H(f) (a real "
            "dB spectrum has no phase panel) — plot it with "
            ".plot(value='dB') instead."
        )
    owns_fig = axes is None
    if owns_fig:
        fig, (ax_mag, ax_phase) = plt.subplots(
            2, 1, sharex=True, figsize=figsize)
    else:
        ax_mag, ax_phase = axes
        fig = ax_mag.figure
    spec.plot(value='level', ax=ax_mag, title=title, **kwargs)
    spec.plot(value='phase', ax=ax_phase, **kwargs)
    ax_phase.set_title('')       # keep the title/pinned subtitle on top only
    ax_mag.set_xlabel('')        # shared axis: label only the bottom panel
    if owns_fig:
        # plot_field skips its credit when handed an ``ax``; draw the
        # model-source footnote once, from the (attributed) source Field.
        _draw_result_credit(fig, field)
    return fig, (ax_mag, ax_phase)


@typed_plot_error(who='Field.plot_impulse_response')
def plot_impulse_response(field, *, ax=None, title=None, window: str = 'hann',
                          nfft: Optional[int] = None,
                          t_start: Optional[float] = None,
                          figsize=(8, 4), **kwargs):
    """Plot the band-limited impulse response ``p(t)`` at one receiver cell.

    Reduce-then-plot counterpart of :func:`plot_transfer_function`: IFFTs
    the single-cell spectrum (``plot_impulse_response(H.at(depth=…,
    range=…))``; a single-receiver field works directly). For the
    response to a specific source pulse use
    :meth:`Field.synthesize_time_series` instead.
    :meth:`Field.plot_impulse_response` is the method form.
    Returns ``(fig, ax)``.

    Parameters
    ----------
    field : Field
        A spectrum reduced to one receiver cell, with a pinned range.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    title : str, optional
        Axes title. ``None`` draws the default caption.
    window, nfft, t_start : optional
        Passed to :meth:`Field.to_time_trace`. Default ``window='hann'``.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(8, 4)`` by default; unused when
        ``ax`` is given.
    **kwargs
        Keywords of the time trace's :meth:`Field.plot`.
    """
    spec = _reduce_to_spectrum(field, 'plot_impulse_response')
    if 'range' not in spec.pinned:
        raise ConfigurationError(
            "Field.plot_impulse_response: the spectrum carries no pinned "
            "range — the IFFT needs it for t_start and demodulation. "
            "Slice a canonical broadband grid (H.at(depth=…, range=…)), "
            "or use to_time_trace on the grid directly."
        )
    # ``nfft``/``t_start`` belong to the synthesis, not to the line:
    # the sampling warning tells the caller to pass ``t_start=``, so
    # it has to arrive here rather than at matplotlib.
    trace = spec.to_time_trace(window=window, nfft=nfft,
                               t_start=t_start)
    owns_fig = ax is None
    if owns_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    # The trace is tagged kind='impulse_response', unit='1/s' (H integrated
    # over the band with no source spectrum), so its own plot reads
    # 'h(t) (1/s)' from the quantity registry.
    trace.plot(ax=ax, title=title, **kwargs)
    if owns_fig:
        # Draw the model-source footnote from the (attributed) source Field
        # — the IFFT trace does not carry the model provenance.
        _draw_result_credit(fig, field)
    return fig, ax
