"""Field plotters: auto-shape heatmaps/line cuts, signal excess, detection probability, and the compare/compare_models overlays."""

from __future__ import annotations

import warnings

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple

from uacpy.core.environment import Environment
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import Field
from uacpy.visualization.style import get_cmap_for_field
from uacpy.visualization.plots._common import _value_array, _coord_label, _coord_axis, _TL_LIMITS, _overlay_seafloor, _pinned_subtitle, _draw_result_credit, fig_ax, invert_yaxis_once, _draw_geometry, typed_plot_error


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
_CONTOUR_FMT = {'tl': '%g dB', 'mag_db': '%g dB', 'phase': '%g rad'}


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
    value : str
        ``'tl'`` (default, dB), ``'mag_db'`` (``20·log10|H|``), ``'mag'``,
        ``'phase'``, ``'real'``, ``'imag'``.
    vmin, vmax : float, optional
        Colour limits (2-D heatmap only). For ``value='tl'`` an unset limit
        takes the fixed 20–120 dB TL scale (``_TL_LIMITS``), never an
        autoscale: TL panels are meant to stay directly comparable across
        models, frequencies and runs.
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
        for axis in ('source_depth', 'frequency', 'depth', 'range', 'time'):
            if len(field.coords) <= 1:
                break
            if axis in field.coords and field.coords[axis].size == 1:
                field = field.isel(**{axis: 0})

    if value is None:
        value = 'real' if field.kind == 'time_series' else 'tl'
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

    if branch == 'stacked':
        fig, ax_out = _plot_field_stacked(
            field, arr, axes_present, ax=ax, title=title,
            figsize=figsize, offset=stack_offset, **mpl_kw,
        )
    elif branch == 'line':
        fig, ax_out = _plot_field_1d(
            field, arr, value_label, axes_present[0],
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
    other_coord = field.coords[other_axis]
    time = field.coords['time']

    if offset is None:
        peak = float(np.max(np.abs(traces))) if traces.size else 1.0
        offset = 2.0 * peak if peak > 0 else 1.0

    fig, ax = fig_ax(ax, figsize)
    for i, c in enumerate(other_coord):
        ax.plot(time, traces[i] + i * offset, linewidth=0.8, **mpl_kw)
    ax.set_xlabel(_coord_label('time'))
    ax.set_ylabel(_coord_label(other_axis) + ' (stacked)')
    ax.set_yticks([i * offset for i in range(len(other_coord))])
    ax.set_yticklabels([f"{float(c):.1f}" for c in other_coord])
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    return fig, ax


def _plot_field_1d(
    field, arr, value_label, axis_name,
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
        # Range / frequency cut: coordinate on X, value on Y. For TL, put the
        # louder (smaller-dB) end at the top.
        x_plot, x_label = _coord_axis(coord, axis_name)
        line, = ax.plot(x_plot, vals, label=label, **mpl_kw)
        ax.set_xlabel(x_label)
        ax.set_ylabel(value_label)
        if value_label == 'TL (dB)':
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
    is_time_domain = 'time' in axes_present and not field.is_complex
    if is_time_domain:
        # Real time-domain pressure → diverging seismic colormap centred
        # at 0. Clip to ±RMS so silence between arrivals doesn't wash
        # out the wavefront — peaks saturate, which is exactly what we
        # want for a moveout reading.
        finite = np.abs(Z[np.isfinite(Z)])
        if finite.size:
            rms = float(np.sqrt(np.mean(finite ** 2)))
            peak = rms if rms > 0 else float(finite.max())
        else:
            peak = 1.0
        if vmin is None:
            vmin = -peak
        if vmax is None:
            vmax = peak
        if cmap is None:
            cmap = 'seismic'
        value_label = 'p(t)'
    elif value == 'tl':
        if vmin is None or vmax is None:
            v_lo, v_hi = _TL_LIMITS
            vmin = v_lo if vmin is None else vmin
            vmax = v_hi if vmax is None else vmax
        if cmap is None:
            cmap = get_cmap_for_field('tl')
    elif value == 'phase':
        if vmin is None:
            vmin = -np.pi
        if vmax is None:
            vmax = np.pi
        if cmap is None:
            cmap = 'twilight'
    else:
        if cmap is None:
            cmap = get_cmap_for_field('tl')

    fig, ax = fig_ax(ax, figsize)

    x_plot, x_label = _coord_axis(x_coord, x_name)

    im = ax.pcolormesh(
        x_plot, y_coord, Z, vmin=vmin, vmax=vmax, cmap=cmap,
        # 'nearest' (not 'auto') centers each cell on its coordinate and errors
        # loudly if coords are ever the wrong length — 'auto' would silently
        # switch to edge ('flat') mode and half-cell-shift the field.
        shading='nearest', **mpl_kw,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(_coord_label(y_name))
    if y_name == 'depth':
        invert_yaxis_once(ax)
    if contours:
        cs = ax.contour(
            x_plot, y_coord, Z, levels=list(contours),
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
        Symmetric colour limit ``[-vmax, +vmax]``. ``None`` uses the
        99th percentile of ``|SE|`` so outliers don't wash out the
        boundary region.
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
        finite = np.abs(Z[np.isfinite(Z)])
        vmax = float(np.percentile(finite, 99.0)) if finite.size else 1.0
        if vmax <= 0:
            vmax = 1.0

    _owns_fig = ax is None
    fig, ax = fig_ax(ax, figsize)

    r_km, x_label = _coord_axis(field.coords['range'], 'range')
    depths = field.coords['depth']
    im = ax.pcolormesh(
        r_km, depths, Z, vmin=-vmax, vmax=vmax, cmap=cmap,
        shading='nearest', **mpl_kw,
    )
    if show_boundary and np.isfinite(Z).any():
        finite_z = Z[np.isfinite(Z)]
        if finite_z.min() < 0.0 < finite_z.max():
            cs = ax.contour(
                r_km, depths, Z, levels=[0.0],
                colors='black', linewidths=1.5, linestyles='solid',
            )
            ax.clabel(cs, inline=True, fontsize=9,
                      fmt=lambda _: 'SE = 0 dB')
    if show_colorbar:
        fig.colorbar(im, ax=ax, label='Signal excess (dB)',
                     fraction=0.046, pad=0.02)
    ax.set_xlabel(x_label)
    ax.set_ylabel(_coord_label('depth'))
    invert_yaxis_once(ax)
    ax.grid(True, alpha=0.3, zorder=0)
    if title:
        ax.set_title(title)
    else:
        budget = field.metadata.get('sonar_budget') or {}
        mode = budget.get('mode')
        pin = _pinned_subtitle(field)
        auto = 'Signal excess' + (f' ({mode})' if mode else '')
        ax.set_title(f"{auto} — {pin}" if pin else auto)
    if env is not None:
        _overlay_seafloor(ax, env, field.coords['range'])
    if _owns_fig:                        # credit only a figure we own
        _draw_result_credit(fig, field, env=env)
    return fig, ax


@typed_plot_error
def plot_detection_probability(
    field: Field,
    ax=None,
    *,
    env: Optional[Environment] = None,
    cmap: str = 'RdYlGn',
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

    Z = np.asarray(field.data, dtype=float)
    _owns_fig = ax is None
    fig, ax = fig_ax(ax, figsize)

    r_km, x_label = _coord_axis(field.coords['range'], 'range')
    depths = field.coords['depth']
    im = ax.pcolormesh(
        r_km, depths, Z, vmin=0.0, vmax=1.0, cmap=cmap,
        shading='nearest', **mpl_kw,
    )
    finite = Z[np.isfinite(Z)]
    if contour_levels and finite.size:
        levels = [
            lv for lv in sorted(contour_levels)
            if finite.min() < lv < finite.max()
        ]
        if levels:
            cs = ax.contour(
                r_km, depths, Z, levels=levels,
                colors='black', linewidths=1.2, linestyles='solid',
            )
            ax.clabel(cs, inline=True, fontsize=9, fmt='%.1f')
    if show_colorbar:
        fig.colorbar(im, ax=ax, label='Probability of detection',
                     fraction=0.046, pad=0.02)
    ax.set_xlabel(x_label)
    ax.set_ylabel(_coord_label('depth'))
    invert_yaxis_once(ax)
    ax.grid(True, alpha=0.3, zorder=0)
    if title:
        ax.set_title(title)
    else:
        sigma = field.metadata.get('sigma_db')
        pin = _pinned_subtitle(field)
        auto = 'Detection probability'
        if sigma is not None:
            auto += f' (σ = {sigma:g} dB)'
        ax.set_title(f"{auto} — {pin}" if pin else auto)
    if env is not None:
        _overlay_seafloor(ax, env, field.coords['range'])
    if _owns_fig:                        # credit only a figure we own
        _draw_result_credit(fig, field, env=env)
    return fig, ax


@typed_plot_error
def compare(
    fields: Sequence[Field],
    labels: Optional[Sequence[str]] = None,
    ax=None,
    *,
    value: str = 'tl',
    figsize: Tuple[float, float] = (10, 5),
    title: Optional[str] = None,
    **mpl_kw,
):
    """Overlay multiple 1-D sliced :class:`Field` instances on one axes.

    Every field must reduce to a single surviving coord axis (the same
    axis across all). Caller slices them first::

        compare([f1.at(depth=20), f2.at(depth=20)], labels=['Bellhop', 'RAM'])
    """
    if not fields:
        raise ConfigurationError("compare: empty fields list")
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
        arr, vlabel = _value_array(f, value)
        if common_axis == 'depth':
            # Depth-cut overlays follow plot_field's convention:
            # depth on Y, increasing downward.
            ax.plot(np.asarray(arr).ravel(), f.coords[common_axis],
                    label=lbl, **mpl_kw)
        else:
            x_plot, x_label = _coord_axis(f.coords[common_axis], common_axis)
            ax.plot(x_plot, np.asarray(arr).ravel(), label=lbl, **mpl_kw)
    if common_axis == 'depth':
        ax.set_ylabel(_coord_label(common_axis))
        ax.set_xlabel(vlabel)
        invert_yaxis_once(ax)
    else:
        ax.set_xlabel(x_label)
        ax.set_ylabel(vlabel)
        if value == 'tl':
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
    from uacpy.visualization.plots._common import (
        _draw_credit, _model_attribution,
    )
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
    value: str = 'tl',
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
    """
    if isinstance(fields, dict):
        if labels is None:
            labels = list(fields.keys())
        fields = list(fields.values())
    n = len(fields)
    if n == 0:
        raise ConfigurationError("compare_models: empty fields list")
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
        for axis in ('depth', 'range'):
            if axis not in ref.coords or axis not in f.coords:
                continue
            ca = ref.coords[axis]
            cb = f.coords[axis]
            if ca.shape != cb.shape or not np.allclose(
                ca, cb, rtol=1e-6, atol=1e-6
            ):
                warnings.warn(
                    f"compare_models: {lbl!r} {axis} axis differs from "
                    f"{labels[0]!r}; the shared colourbar mixes "
                    "different sample grids.",
                    UserWarning, stacklevel=2,
                )
                break

    if value == 'tl' and (vmin is None or vmax is None):
        v_lo, v_hi = _TL_LIMITS
        vmin = v_lo if vmin is None else vmin
        vmax = v_hi if vmax is None else vmax
    elif vmin is None or vmax is None:
        # Every other ``value`` has no fixed scale, so pool the panels: left to
        # autoscale, each ``plot_field`` would map its own panel's range and the
        # single figure-level colorbar below would then annotate the figure with
        # only the last panel's limits — two fields differing by 100x would
        # render identically. Signed quantities get a symmetric range so zero
        # stays the neutral colour.
        pooled = [np.asarray(_value_array(f, value)[0], dtype=float).ravel()
                  for f in fields]
        finite = np.concatenate([p[np.isfinite(p)] for p in pooled]) \
            if any(np.isfinite(p).any() for p in pooled) else np.array([0.0])
        if value in ('real', 'imag') or np.nanmin(finite) < 0.0:
            span = float(np.max(np.abs(finite))) or 1.0
            lo, hi = -span, span
        else:
            lo, hi = float(np.min(finite)), float(np.max(finite))
            if hi <= lo:                       # a constant field has no range
                lo, hi = lo - 0.5, hi + 0.5
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax
    if cmap is None:
        cmap = get_cmap_for_field('tl' if value == 'tl' else 'pressure')

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
        if ax.collections:
            # Every panel was drawn with the same vmin/vmax/cmap, so any one
            # mesh maps the shared scale — keep the last for the single
            # figure-level colorbar.
            im_last = ax.collections[0]
    for ax in axes_flat[n:]:
        ax.axis('off')

    top = 0.90 if title else 0.95
    fig.subplots_adjust(left=0.05, right=0.88, top=top, bottom=0.08,
                        wspace=0.22, hspace=0.30)
    if im_last is not None:
        cbar_label = 'TL (dB)' if value == 'tl' else value
        cbar_ax = fig.add_axes([0.905, 0.08, 0.015, top - 0.08])
        fig.colorbar(im_last, cax=cbar_ax, label=cbar_label)
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.97)
    _draw_multi_model_credit(fig, fields)
    return fig, axes_flat


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
    fig.tight_layout()
    if title:
        fig.suptitle(title, fontweight='bold')
    _draw_result_credit(fig, stack.slabs[0], env=env)
    return fig, axes
