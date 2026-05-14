"""Plotting for uacpy ``Result`` types.

Canonical surface
-----------------

* :func:`plot_field` — auto-shape plotter for :class:`~uacpy.Field`. Reads
  ``field.coords`` after the user's :meth:`Field.at` / :meth:`Field.isel`
  slicing. 1 surviving axis → line; 2 surviving axes → heatmap. Optional
  ``projection='polar'`` for ``(depth, range)`` fields.
* :func:`compare` — overlay multiple 1-D sliced fields on one axes.
* :func:`compare_models` — side-by-side heatmap grid of 2-D fields.
* :func:`plot_rays`, :func:`plot_arrivals` — ray fans / arrival stems.
* :func:`plot_environment` — SSP + bathymetry + bottom in one figure.
* :func:`plot_mode_functions`, :func:`plot_mode_wavenumbers`,
  :func:`plot_modes_heatmap` — three distinct mode views.
* :func:`plot_reflection_coefficient`, :func:`plot_covariance`,
  :func:`plot_replicas` — niche typed results.
* :func:`plot_result` — type-dispatch to one of the above; used by
  :meth:`Result.plot`.

Cuts and slices are made on the Field, not on the plotter::

    plot_field(tl)                          # full 2-D heatmap
    plot_field(tl.at(depth=20))             # 1-D cut along range
    plot_field(tl.at(range=5000))           # 1-D cut along depth
    plot_field(tf.at(frequency=200))        # narrowband heatmap at 200 Hz
    plot_field(tf.at(depth=z, range=r))     # 1-D spectrum at one point
    plot_field(tl, projection='polar')      # polar TL view
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple

from uacpy.core.environment import Environment
from uacpy.core.results import (
    Field, Arrivals, Rays, Modes,
    Covariance, Replicas, ReflectionCoefficient, ResultStack,
)
from uacpy.visualization.style import (
    get_cmap_for_field,
    BOTTOM_FILL_STYLE,
    BOTTOM_LINE_STYLE,
    BOTTOM_LINE_STYLE_FLAT,
    BOTTOM_HALFSPACE_COLOR,
    RECEIVER_MARKER_STYLE,
    SOURCE_MARKER_STYLE,
)


ZORDER_SEDIMENT = 2
ZORDER_RAYS = 2.5
ZORDER_SURFACE = 4
ZORDER_RECEIVERS = 5
ZORDER_SOURCE = 6


# ─────────────────────────────────────────────────────────────────────────────
# Result dispatcher (used by Result.plot)
# ─────────────────────────────────────────────────────────────────────────────


def plot_result(result, env: Optional[Environment] = None, **kwargs):
    """Type-dispatch to the right plotter. Used by :meth:`Result.plot`."""
    if isinstance(result, ResultStack):
        raise TypeError(
            "plot_result: ResultStack carries multiple slabs — pick one "
            "with stack[i] or stack.at(...) before plotting."
        )
    if isinstance(result, Field):
        return plot_field(result, env=env, **kwargs)
    if isinstance(result, Arrivals):
        return plot_arrivals(result, **kwargs)
    if isinstance(result, Rays):
        return plot_rays(result, env=env, **kwargs)
    if isinstance(result, Modes):
        return plot_mode_functions(result, **kwargs)
    if isinstance(result, Covariance):
        return plot_covariance(result, **kwargs)
    if isinstance(result, Replicas):
        return plot_replicas(result, **kwargs)
    if isinstance(result, ReflectionCoefficient):
        return plot_reflection_coefficient(result, **kwargs)
    raise TypeError(
        f"plot_result: no plotter registered for {type(result).__name__}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Field — auto-shape plotter
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
            raise ValueError("plot_field: value='imag' requires complex data")
        return field.data.imag, 'Im(p)'
    raise ValueError(
        f"plot_field: unknown value={value!r}; "
        "valid: 'tl', 'mag', 'phase', 'real', 'imag'"
    )


def _coord_label(name: str) -> str:
    label, unit = _AXIS_LABELS.get(name, (name, ''))
    return f"{label} ({unit})" if unit else label


def _auto_tl_limits(arr: np.ndarray, span: float = 50.0) -> Tuple[float, float]:
    """Cosmetic auto-clip for TL heatmaps: ``vmax = median + 0.75·std``
    rounded to 10 dB, ``vmin = vmax - span``. Filters out the 600 dB
    no-data sentinel some AT binaries emit."""
    finite = arr[np.isfinite(arr) & (arr < 200.0)]
    if finite.size == 0:
        return (30.0, 80.0)
    vmax = 10.0 * np.round((np.median(finite) + 0.75 * np.std(finite)) / 10.0)
    return (float(vmax - span), float(vmax))


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
        r_km = env.bathymetry[:, 0] / 1000.0
        z = env.bathymetry[:, 1]
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


def plot_field(
    field: Field,
    ax=None,
    *,
    env: Optional[Environment] = None,
    value: Optional[str] = None,
    projection: str = 'cartesian',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: Optional[str] = None,
    title: Optional[str] = None,
    label: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 5),
    stacked: bool = False,
    stack_offset: Optional[float] = None,
    show_colorbar: bool = True,
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
    value : str
        ``'tl'`` (default, dB), ``'mag'``, ``'phase'``, ``'real'``,
        ``'imag'``.
    projection : str
        ``'cartesian'`` (default) or ``'polar'`` (2-D ``(depth, range)``
        only). The polar view shows the range-dependent TL azimuthally
        symmetrically — useful for radial bird's-eye renderings.
    vmin, vmax : float, optional
        Colour limits. ``None`` picks an auto-clip for TL.
    cmap : str, optional
        Override the default colormap.
    title, label : str, optional
    figsize : tuple
    stacked : bool
        Only valid on a 2-D field that carries a ``'time'`` axis. Plots
        each row of ``data`` as an offset trace stacked vertically — the
        classic seismic-record waterfall.
    stack_offset : float, optional
        Vertical offset between stacked traces. ``None`` picks
        ``2 × max|data|`` for visual separation.
    """
    if not isinstance(field, Field):
        raise TypeError(
            f"plot_field: expected Field, got {type(field).__name__}"
        )

    if value is None:
        value = 'real' if field.kind == 'time_series' else 'tl'
    arr, value_label = _value_array(field, value)
    axes_present = list(field.coords)
    n_axes = len(axes_present)

    if projection == 'polar' and axes_present != ['depth', 'range']:
        raise ValueError(
            "plot_field(projection='polar'): requires a 2-D "
            f"['depth', 'range'] field; got coords {axes_present}"
        )

    if stacked:
        if n_axes != 2 or 'time' not in axes_present:
            raise ValueError(
                "plot_field(stacked=True): requires a 2-D field with a "
                f"'time' axis; got coords {axes_present}"
            )
        return _plot_field_stacked(
            field, arr, axes_present, ax=ax, title=title,
            figsize=figsize, offset=stack_offset, **mpl_kw,
        )

    if n_axes == 1:
        return _plot_field_1d(
            field, arr, value_label, axes_present[0],
            ax=ax, title=title, label=label, figsize=figsize, **mpl_kw,
        )
    if n_axes == 2:
        return _plot_field_2d(
            field, arr, value_label, axes_present,
            ax=ax, env=env, projection=projection,
            vmin=vmin, vmax=vmax, cmap=cmap, value=value, title=title,
            figsize=figsize, show_colorbar=show_colorbar,
            contours=contours, **mpl_kw,
        )
    raise ValueError(
        f"plot_field: cannot plot a {n_axes}-axis field (coords "
        f"{axes_present}); slice it first with .at(...) / .isel(...) "
        "so 1 or 2 axes remain."
    )


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

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
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
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    x = field.coords[axis_name]
    y = np.asarray(arr).ravel()
    line, = ax.plot(x, y, label=label, **mpl_kw)
    ax.set_xlabel(_coord_label(axis_name))
    ax.set_ylabel(value_label)
    if value_label == 'TL (dB)' and 'depth' not in axis_name:
        ax.invert_yaxis()
    if axis_name == 'depth':
        ax.invert_yaxis()
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
    *, ax, env, projection, vmin, vmax, cmap, value, title, figsize,
    show_colorbar=True, contours=None, **mpl_kw,
):
    if axes_present == ['depth', 'range']:
        x_name, y_name = 'range', 'depth'
        Z = arr
    else:
        # General two-axis case: first axis on Y, second on X.
        y_name, x_name = axes_present[0], axes_present[1]
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
            v_lo, v_hi = _auto_tl_limits(Z)
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

    if projection == 'polar':
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='polar')
        else:
            fig = ax.figure
        # Use range as radius, depth as angle band — closest to a
        # bird's-eye TL plot. shading='auto' lets matplotlib pick edges.
        n_theta = max(64, len(x_coord))
        theta = np.linspace(0, 2 * np.pi, n_theta)
        # Repeat the (depth, range) field around the full circle so the
        # polar view is meaningful (a single radial line per range).
        radial = x_coord
        Z_polar = np.tile(Z.mean(axis=0)[None, :], (n_theta, 1))
        im = ax.pcolormesh(
            theta, radial, Z_polar.T, vmin=vmin, vmax=vmax, cmap=cmap,
            shading='auto', **mpl_kw,
        )
        ax.set_theta_zero_location('N')
        fig.colorbar(im, ax=ax, label=value_label)
        return fig, ax

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    x_plot = x_coord / 1000.0 if x_name == 'range' else x_coord
    x_label = 'Range (km)' if x_name == 'range' else _coord_label(x_name)

    im = ax.pcolormesh(
        x_plot, y_coord, Z, vmin=vmin, vmax=vmax, cmap=cmap,
        shading='auto', **mpl_kw,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(_coord_label(y_name))
    if y_name == 'depth':
        ax.invert_yaxis()
    if contours:
        cs = ax.contour(
            x_plot, y_coord, Z, levels=list(contours),
            colors='black', linewidths=1.5, alpha=0.8,
            linestyles='solid',
        )
        ax.clabel(cs, inline=True, fontsize=9, fmt='%g dB')
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
    if axes_present == ['depth', 'range'] and env is not None:
        _overlay_seafloor(ax, env, x_coord)
    return fig, ax


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
# compare / compare_models
# ─────────────────────────────────────────────────────────────────────────────


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
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    if labels is None:
        labels = [getattr(f, 'model', '') or f"#{i}" for i, f in enumerate(fields)]
    common_axis = None
    for f, lbl in zip(fields, labels):
        if not isinstance(f, Field):
            raise TypeError(
                f"compare: expected Field, got {type(f).__name__}"
            )
        axes = list(f.coords)
        if len(axes) != 1:
            raise ValueError(
                f"compare: each field must have exactly 1 surviving axis; "
                f"{lbl!r} has {axes}"
            )
        if common_axis is None:
            common_axis = axes[0]
        elif axes[0] != common_axis:
            raise ValueError(
                f"compare: axis mismatch — {labels[0]!r} on {common_axis!r}, "
                f"{lbl!r} on {axes[0]!r}"
            )
        arr, vlabel = _value_array(f, value)
        ax.plot(f.coords[common_axis], np.asarray(arr).ravel(),
                label=lbl, **mpl_kw)
    ax.set_xlabel(_coord_label(common_axis))
    ax.set_ylabel(vlabel)
    if value == 'tl':
        ax.invert_yaxis()
    if common_axis == 'depth':
        ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title)
    return fig, ax


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
    suptitle: Optional[str] = None,
    ncols: Optional[int] = None,
    contours: Optional[Sequence[float]] = None,
):
    """Side-by-side heatmaps of several 2-D :class:`Field` instances.

    ``fields`` is either a list of :class:`Field` (then ``labels`` is
    used as the per-axes title), or a ``{label: Field}`` dict. Shared
    colour scale; one colorbar per axes.

    ``ncols`` controls the grid width — defaults to ``n`` (single row).
    ``contours`` adds dB-level contour lines to every panel.
    """
    if isinstance(fields, dict):
        if labels is None:
            labels = list(fields.keys())
        fields = list(fields.values())
    n = len(fields)
    if n == 0:
        raise ValueError("compare_models: empty fields list")
    if labels is None:
        labels = [getattr(f, 'model', '') or f"#{i}" for i, f in enumerate(fields)]
    if ncols is None:
        ncols = n
    nrows = int(np.ceil(n / ncols))

    if value == 'tl' and (vmin is None or vmax is None):
        cat = np.concatenate(
            [np.asarray(_value_array(f, value)[0]).ravel() for f in fields]
        )
        v_lo, v_hi = _auto_tl_limits(cat)
        vmin = v_lo if vmin is None else vmin
        vmax = v_hi if vmax is None else vmax
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
            im_last = ax.collections[0]
    for ax in axes_flat[n:]:
        ax.axis('off')

    top = 0.90 if suptitle else 0.95
    fig.subplots_adjust(left=0.05, right=0.88, top=top, bottom=0.08,
                        wspace=0.22, hspace=0.30)
    if im_last is not None:
        cbar_label = 'TL (dB)' if value == 'tl' else value
        cbar_ax = fig.add_axes([0.905, 0.08, 0.015, top - 0.08])
        fig.colorbar(im_last, cax=cbar_ax, label=cbar_label)
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.97)
    return fig, axes_flat


# ─────────────────────────────────────────────────────────────────────────────
# Rays / Arrivals
# ─────────────────────────────────────────────────────────────────────────────


def plot_rays(
    rays: Rays,
    ax=None,
    *,
    env: Optional[Environment] = None,
    figsize: Tuple[float, float] = (12, 6),
    color_by: Optional[str] = 'bounces',
    show_receivers: bool = True,
    show_source: bool = True,
    show_legend: bool = True,
    title: Optional[str] = None,
    linewidth: float = 1.0,
    alpha: float = 0.55,
    **mpl_kw,
):
    """Plot a Bellhop ray fan or eigenray set.

    ``color_by='bounces'`` colours rays by direct/surface/bottom/both
    multipath class (red / green / blue / black); ``None`` paints every
    ray in the same colour. The legend reports per-class ray counts.
    """
    if not isinstance(rays, Rays):
        raise TypeError(f"plot_rays: expected Rays, got {type(rays).__name__}")
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    color_map = {
        'direct': '#e53935',
        'surface': '#43a047',
        'bottom': '#1e88e5',
        'both': '#000000',
    }
    bounce_counts = {'direct': 0, 'surface': 0, 'bottom': 0, 'both': 0}
    max_r_km = 0.0
    max_z = 0.0
    for ray in rays.rays:
        r = np.asarray(ray.get('r', []))
        z = np.asarray(ray.get('z', []))
        if r.size == 0:
            continue
        max_r_km = max(max_r_km, float(np.max(r)) / 1000.0)
        max_z = max(max_z, float(np.max(z)))
        n_top = int(ray.get('n_top_bounces', 0) or 0)
        n_bot = int(ray.get('n_bot_bounces', 0) or 0)
        if n_top and n_bot:
            kind = 'both'
        elif n_bot:
            kind = 'bottom'
        elif n_top:
            kind = 'surface'
        else:
            kind = 'direct'
        bounce_counts[kind] += 1
        color = color_map[kind] if color_by == 'bounces' else color_map['bottom']
        ax.plot(r / 1000.0, z, color=color, alpha=alpha,
                linewidth=linewidth, solid_capstyle='round',
                zorder=ZORDER_RAYS, **mpl_kw)

    ax.invert_yaxis()
    depth_for_lim = max_z
    if env is not None:
        depth_for_lim = max(depth_for_lim, float(env.depth))
    if depth_for_lim > 0:
        ax.set_ylim(depth_for_lim * 1.08, -depth_for_lim * 0.04)

    if env is not None:
        # Surface line styled to match the AT convention.
        ax.axhline(0, color='steelblue', linewidth=1.5, alpha=0.55,
                   zorder=ZORDER_SURFACE)
        # Anchor the seafloor overlay from x=0 (source range) to the
        # furthest receiver, so the bathy fill flush-spans the chart and
        # leaves no white sliver under the rays at small ranges.
        if rays.receiver_ranges is not None and len(rays.receiver_ranges):
            r_hi = float(np.max(rays.receiver_ranges))
        else:
            r_hi = max_r_km * 1000.0
        ranges_for_overlay = np.array([0.0, r_hi])
        _overlay_seafloor(ax, env, ranges_for_overlay)

    if show_receivers and rays.receiver_ranges is not None and rays.receiver_depths is not None:
        rr_full = np.atleast_1d(rays.receiver_ranges) / 1000.0
        rd_full = np.atleast_1d(rays.receiver_depths)
        # Dense receiver grids drown out the rays — decimate each axis
        # independently to keep the lattice visible (10 down × 20 across
        # max, matching plot_environment).
        max_range_dots = 20
        max_depth_dots = 10
        step_r = max(1, rr_full.size // max_range_dots)
        step_d = max(1, rd_full.size // max_depth_dots)
        rr = rr_full[::step_r]
        rd = rd_full[::step_d]
        RR, RD = np.meshgrid(rr, rd)
        # Shrink the marker for ray-fan plots — receivers are sampling
        # points, not the visual focus.
        rcv_style = dict(RECEIVER_MARKER_STYLE)
        rcv_style['markersize'] = min(rcv_style.get('markersize', 8), 4)
        ax.plot(RR.ravel(), RD.ravel(),
                zorder=ZORDER_RECEIVERS, **rcv_style)
        # Clip the x-axis to the full receiver extent (not the decimated
        # subset) so rays don't trail off into empty bathy-less range.
        ax.set_xlim(0.0, float(np.max(rr_full)))
    if show_source and rays.source_depths is not None and rays.source_depths.size:
        for sd in rays.source_depths:
            ax.plot([0.0], [float(sd)], zorder=ZORDER_SOURCE,
                    **SOURCE_MARKER_STYLE)

    if show_legend and color_by == 'bounces':
        import matplotlib.lines as mlines
        handles = [
            mlines.Line2D([], [], color=col, linewidth=2,
                          label=f"{kind} ({bounce_counts[kind]})")
            for kind, col in color_map.items()
            if bounce_counts[kind] > 0
        ]
        if handles:
            ax.legend(handles=handles, loc='lower right',
                      fontsize=9, framealpha=0.85)

    ax.set_xlabel('Range (km)')
    ax.set_ylabel('Depth (m)')
    ax.grid(True, alpha=0.3)
    ax.set_title(title or ('Eigenrays' if rays.is_eigen else 'Ray fan'))
    return fig, ax


def plot_arrivals(
    arrivals: Arrivals,
    ax=None,
    *,
    figsize: Tuple[float, float] = (10, 4),
    title: Optional[str] = None,
):
    """Stem plot of arrivals: amplitude vs delay, coloured by multipath class.

    Colour palette matches :func:`plot_rays`: direct = red,
    surface = green, bottom = blue, both = black. Each arrival is drawn
    as a vertical stem plus a head marker."""
    if not isinstance(arrivals, Arrivals):
        raise TypeError(
            f"plot_arrivals: expected Arrivals, got {type(arrivals).__name__}"
        )
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    color_map = {
        'direct': '#e53935',
        'surface': '#43a047',
        'bottom': '#1e88e5',
        'both': '#000000',
    }
    counts = {k: 0 for k in color_map}
    delays_ms = []
    for a in arrivals.arrivals:
        kind = a.get('kind', 'direct')
        col = color_map.get(kind, '#1e88e5')
        d_ms = a['delay'] * 1000.0
        delays_ms.append(d_ms)
        ax.vlines(d_ms, 0, a['amplitude'], colors=col, lw=1.5, alpha=0.85)
        ax.plot(d_ms, a['amplitude'], 'o', color=col, markersize=4,
                markeredgecolor='black', markeredgewidth=0.4)
        counts[kind] += 1
    if delays_ms:
        span = max(delays_ms) - min(delays_ms)
        ax.set_xlim(min(delays_ms) - 0.05 * (span or 1),
                    max(delays_ms) + 0.05 * (span or 1))
    ax.set_xlabel('Delay (ms)')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)
    # Legend with per-class counts (skip empty classes).
    import matplotlib.lines as mlines
    handles = [
        mlines.Line2D([], [], color=col, marker='o', linestyle='-',
                      label=f"{kind} ({counts[kind]})")
        for kind, col in color_map.items() if counts[kind] > 0
    ]
    if handles:
        ax.legend(handles=handles, loc='upper right', fontsize=9,
                  framealpha=0.85)
    if title:
        ax.set_title(title)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────


def plot_environment(
    env: Environment,
    *,
    source=None,
    receiver=None,
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
    """
    from uacpy.core.environment import (
        BoundaryProperties, LayeredBottom,
        RangeDependentBottom, RangeDependentLayeredBottom,
    )

    fig, ax_bathy = plt.subplots(1, 1, figsize=figsize)

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

    _v_lo, _v_hi = water_cs_min, water_cs_max

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
                r_km, z_top, z_bot, color=colour, alpha=0.95,
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

    elif isinstance(bottom, RangeDependentLayeredBottom):
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
                    color=colour, alpha=0.95,
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

    elif isinstance(bottom, RangeDependentBottom):
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
                poly_r, poly_z, color=colour, alpha=0.95,
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
                      alpha=0.95),
        )
        z_max_layer = hs_floor

    else:  # BoundaryProperties or other half-space
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

    # Two colorbars on opposite sides of the bathy panel: water cp on
    # the LEFT, bottom cp on the RIGHT. Each on its own dynamic range.
    # The bottom colorbar is suppressed when the bottom is a single
    # half-space (no cs gradient — the property card already states cs).
    fig.colorbar(water_sm, ax=ax_bathy, label='Water cp (m/s)',
                 location='left', fraction=0.046, pad=0.10)
    if not isinstance(bottom, BoundaryProperties):
        fig.colorbar(bottom_sm, ax=ax_bathy, label='Bottom cp (m/s)',
                     location='right', fraction=0.046, pad=0.02)

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
        # X-axis label gets a "(receivers: N×M, 1/n×1/m shown)" suffix
        # so the decimation is honest without occluding any data.
        if step_r > 1 or step_d > 1:
            decim = (f"  (receivers: {rd_full.size}×{rr_full.size}, "
                     f"1/{step_d}×1/{step_r} shown)")
        else:
            decim = f"  (receivers: {rd_full.size}×{rr_full.size})"
        ax_bathy.set_xlabel(f"Range (km){decim}", fontweight='bold')

    ax_bathy.set_xlim(*x_range)
    # Tight ylim — surface to a small margin past the deepest seafloor.
    # Bottom rendering may extend below the seafloor visually (hatched
    # half-space / PML-like padding) but the displayed extent stays
    # close to the physical water column.
    ax_bathy.set_ylim(0, seafloor_depth * 1.20)
    if not ax_bathy.get_xlabel():
        ax_bathy.set_xlabel('Range (km)', fontweight='bold')
    ax_bathy.set_ylabel('Depth (m)', fontweight='bold')
    ax_bathy.invert_yaxis()
    ax_bathy.grid(True, alpha=0.3)
    ax_bathy.set_title(f"Bottom — {type(env.bottom).__name__}",
                       fontweight='bold', fontsize=12)

    fig.tight_layout()
    return fig, ax_bathy


# ─────────────────────────────────────────────────────────────────────────────
# Modes — three distinct views
# ─────────────────────────────────────────────────────────────────────────────


def plot_mode_functions(
    modes: Modes,
    n_modes: Optional[int] = None,
    ax=None,
    *,
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = None,
):
    """Plot the first ``n_modes`` mode shapes ``ψ_m(z)`` as overlaid 1-D curves."""
    if not isinstance(modes, Modes):
        raise TypeError(
            f"plot_mode_functions: expected Modes, got {type(modes).__name__}"
        )
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    n_modes = modes.n_modes if n_modes is None else min(int(n_modes), modes.n_modes)
    for m in range(n_modes):
        psi = np.asarray(modes.phi[:, m])
        if np.iscomplexobj(psi):
            psi = psi.real
        ax.plot(psi, modes.depths, label=f"m={m+1}", linewidth=1.0)
    ax.set_xlabel(r'$\psi_m(z)$')
    ax.set_ylabel('Depth (m)')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    if n_modes <= 12:
        ax.legend(fontsize=8, loc='best')
    ax.set_title(title or f"Mode functions (n={n_modes})")
    return fig, ax


def plot_mode_wavenumbers(
    modes: Modes,
    ax=None,
    *,
    figsize: Tuple[float, float] = (8, 5),
    title: Optional[str] = None,
):
    """Scatter ``Re(k_m)`` vs mode index; overlay imaginary part if non-zero."""
    if not isinstance(modes, Modes):
        raise TypeError(
            f"plot_mode_wavenumbers: expected Modes, got {type(modes).__name__}"
        )
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    idx = np.arange(1, modes.n_modes + 1)
    k = np.asarray(modes.k)
    ax.plot(idx, k.real, 'o-', label=r'$\mathrm{Re}(k_m)$')
    if np.any(np.abs(k.imag) > 0):
        ax2 = ax.twinx()
        ax2.plot(idx, k.imag, 's--', color='C1', label=r'$\mathrm{Im}(k_m)$')
        ax2.set_ylabel(r'$\mathrm{Im}(k_m)$ (1/m)')
    ax.set_xlabel('Mode index')
    ax.set_ylabel(r'$\mathrm{Re}(k_m)$ (1/m)')
    ax.grid(True, alpha=0.3)
    ax.set_title(title or 'Modal wavenumbers')
    return fig, ax


def plot_modes_heatmap(
    modes: Modes,
    n_modes: Optional[int] = None,
    ax=None,
    *,
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = None,
    mode_range: Optional[Tuple[int, int]] = None,
    normalize: bool = True,
    cmap: str = 'RdBu_r',
):
    """Heatmap of ``ψ_m(z)`` over (depth, mode index).

    ``mode_range=(start, stop)`` selects a half-open mode-index slice.
    ``normalize=True`` (default) rescales each column to peak ``±1`` so
    high-order modes don't disappear next to the dominant low-order ones.
    """
    if not isinstance(modes, Modes):
        raise TypeError(
            f"plot_modes_heatmap: expected Modes, got {type(modes).__name__}"
        )
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    if mode_range is not None:
        start, stop = mode_range
        stop = min(stop, modes.n_modes)
    else:
        start = 0
        stop = (modes.n_modes if n_modes is None
                else min(int(n_modes), modes.n_modes))
    n_plot = stop - start
    phi = np.asarray(modes.phi[:, start:stop])
    if np.iscomplexobj(phi):
        phi = phi.real
    phi = phi.copy()
    if normalize:
        for i in range(n_plot):
            peak = float(np.max(np.abs(phi[:, i])))
            if peak > 0:
                phi[:, i] /= peak
        vmin, vmax = -1.0, 1.0
    else:
        vabs = float(np.max(np.abs(phi))) if phi.size else 1.0
        vmin, vmax = -vabs, vabs
    idx = np.arange(start + 1, stop + 1)
    # ``shading='nearest'`` puts each column at the integer mode index
    # without needing a +1 edges array — matches matlab pcolor.
    im = ax.pcolormesh(idx, modes.depths, phi, cmap=cmap,
                       shading='nearest', vmin=vmin, vmax=vmax)
    fig.colorbar(
        im, ax=ax,
        label='Normalised amplitude' if normalize else r'$\psi_m(z)$',
    )
    ax.set_xlabel('Mode index')
    ax.set_ylabel('Depth (m)')
    ax.invert_yaxis()
    f0 = modes.f0 if modes.f0 is not None else 0.0
    ax.set_title(
        title or f'Mode shapes — {n_plot} modes @ {f0:.1f} Hz',
    )
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Reflection coefficient
# ─────────────────────────────────────────────────────────────────────────────


def plot_reflection_coefficient(
    rc: ReflectionCoefficient,
    ax=None,
    *,
    figsize: Tuple[float, float] = (8, 5),
    title: Optional[str] = None,
    show_phase: bool = False,
):
    """Auto-detect narrowband (line) vs broadband (heatmap) reflection coefficient.

    ``show_phase=True`` overlays the phase ``φ(θ)`` on a twin y-axis
    when the input is narrowband (single frequency)."""
    if not isinstance(rc, ReflectionCoefficient):
        raise TypeError(
            f"plot_reflection_coefficient: expected ReflectionCoefficient, "
            f"got {type(rc).__name__}"
        )
    if rc.is_broadband:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        im = ax.pcolormesh(
            rc.frequencies / 1000.0, rc.theta, rc.R,
            shading='auto', cmap='viridis',
        )
        fig.colorbar(im, ax=ax, label='|R|')
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Grazing angle (°)')
        ax.set_title(title or 'Reflection coefficient |R(θ, f)|')
        return fig, ax

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.plot(rc.theta, rc.R, label='|R|', color='C0')
    ax.set_xlabel('Grazing angle (°)')
    ax.set_ylabel('|R|', color='C0')
    ax.tick_params(axis='y', labelcolor='C0')
    ax.grid(True, alpha=0.3)
    if show_phase:
        ax_phi = ax.twinx()
        ax_phi.plot(rc.theta, np.rad2deg(rc.phi), '--', color='C1',
                    label='φ')
        ax_phi.set_ylabel('Phase (°)', color='C1')
        ax_phi.tick_params(axis='y', labelcolor='C1')
    ax.set_title(title or 'Reflection coefficient')
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Covariance / Replicas
# ─────────────────────────────────────────────────────────────────────────────


def plot_covariance(
    cov: Covariance,
    ax=None,
    *,
    freq_idx: int = 0,
    figsize: Tuple[float, float] = (6, 5),
    title: Optional[str] = None,
):
    """Heatmap of one covariance slice ``|C[freq_idx, :, :]|``."""
    if not isinstance(cov, Covariance):
        raise TypeError(
            f"plot_covariance: expected Covariance, got {type(cov).__name__}"
        )
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    C = np.abs(cov.covariance[freq_idx])
    im = ax.imshow(C, cmap='viridis', aspect='auto', origin='upper')
    fig.colorbar(im, ax=ax, label='|C|')
    ax.set_xlabel('Receiver j')
    ax.set_ylabel('Receiver i')
    f_hz = float(cov.frequencies[freq_idx]) if cov.frequencies is not None else None
    if title is None and f_hz is not None:
        title = f"Covariance at {f_hz:.1f} Hz"
    if title:
        ax.set_title(title)
    return fig, ax


def plot_replicas(
    rep: Replicas,
    ax=None,
    *,
    freq_idx: int = 0,
    sensor_idx: int = 0,
    figsize: Tuple[float, float] = (8, 5),
    title: Optional[str] = None,
):
    """Magnitude of replica response across (z, x) at fixed y=0."""
    if not isinstance(rep, Replicas):
        raise TypeError(
            f"plot_replicas: expected Replicas, got {type(rep).__name__}"
        )
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    R = np.abs(rep.replicas[freq_idx, :, :, 0, sensor_idx])
    im = ax.pcolormesh(
        rep.replica_x, rep.replica_z, R,
        shading='auto', cmap='magma',
    )
    fig.colorbar(im, ax=ax, label='|R|')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.invert_yaxis()
    if title:
        ax.set_title(title)
    return fig, ax


__all__ = [
    'plot_result',
    'plot_field',
    'compare',
    'compare_models',
    'plot_rays',
    'plot_arrivals',
    'plot_environment',
    'plot_mode_functions',
    'plot_mode_wavenumbers',
    'plot_modes_heatmap',
    'plot_reflection_coefficient',
    'plot_covariance',
    'plot_replicas',
]
