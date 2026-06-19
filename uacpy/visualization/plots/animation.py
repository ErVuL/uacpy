"""Time-domain field animation and snapshot montages."""

from __future__ import annotations


import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple

from uacpy.core.environment import Environment
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import Field
from uacpy.visualization.style import SOURCE_MARKER_STYLE
from uacpy.visualization.plots._common import ZORDER_SOURCE, _imshow_extent, _overlay_seafloor


def animate_field(
    field: 'Field',
    *,
    env: Optional[Environment] = None,
    fps: int = 30,
    frame_stride: Optional[int] = None,
    p_max: Optional[float] = None,
    cmap: str = 'RdBu_r',
    ax=None,
    show_source: bool = True,
    show_seafloor: bool = True,
    show_time: bool = True,
    title: Optional[str] = None,
    aspect: str = 'auto',
):
    """Animate a time-series :class:`Field` as a (depth, range) heatmap
    that evolves along the time axis.

    Parameters
    ----------
    field : Field
        Must have ``kind='time_series'`` and ``coords={'depth', 'range',
        'time'}``. Data is real-valued p(d, r, t).
    env : Environment, optional
        When supplied, the seafloor (and surface, if elastic) overlay is
        drawn on top of the field — same convention as :func:`plot_field`.
    fps : int, optional
        Playback frame rate. Default 30. The animation's elapsed wall-
        time is ``n_frames / fps``.
    frame_stride : int, optional
        Sub-sample the time axis. ``None`` (default) caps the animation
        at ~300 frames via ``max(1, n_t // 300)``. Set to 1 to render
        every sample (large outputs).
    p_max : float, optional
        Symmetric colour-scale range ``[-p_max, +p_max]``. ``None``
        (default) uses the 99.5th percentile of ``|data|`` over the
        whole field — keeps the wave visible without the early-time
        near-source spike washing out later frames.
    cmap : str, optional
        Diverging colormap for ±pressure. Default ``'RdBu_r'``.
    ax : matplotlib.axes.Axes, optional
        Target axes. ``None`` creates a fresh figure.
    show_source : bool, optional
        Plot a marker at ``(range=0, depth=field.source_depths[0])`` if
        the field carries a source depth.
    show_seafloor : bool, optional
        Overlay env.bathymetry (or env.depth) on every frame.
    show_time : bool, optional
        Append the current frame time to the title.
    title : str, optional
        Custom title prefix. ``None`` uses ``f"{field.model} — p(d, r, t)"``.
    aspect : str or float, optional
        Passed to :meth:`matplotlib.axes.Axes.imshow`. ``'auto'`` (default)
        stretches the heatmap to fill the axes — fine for wide-aspect
        domains. Use ``'equal'`` when the range and depth extents are
        comparable (small-domain visualisations) so isotropic wavefronts
        stay round instead of being stretched into ellipses.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Caller decides how to render: ``ani.save('out.mp4',
        writer='ffmpeg')`` (requires ffmpeg), ``ani.save('out.gif',
        writer=PillowWriter(fps=fps))`` (no extra deps), or
        ``HTML(ani.to_jshtml())`` for notebook embedding.
    """
    from matplotlib.animation import FuncAnimation
    from uacpy.core.results import Field  # local import to avoid cycle

    if not isinstance(field, Field) or field.kind != 'time_series':
        raise ConfigurationError(
            "animate_field: needs a Field with kind='time_series' "
            "(real-valued, ``coords`` containing a 'time' axis). "
            f"Got kind={getattr(field, 'kind', None)!r}."
        )
    expected_axes = {'depth', 'range', 'time'}
    if expected_axes - set(field.coords):
        missing = expected_axes - set(field.coords)
        raise ConfigurationError(
            f"animate_field: field is missing coord axes {sorted(missing)}. "
            f"Need depth, range, and time — got {list(field.coords)}."
        )

    # Move data into canonical (depth, range, time) layout regardless of
    # the storage order ``field.coords`` declares.
    axis_order = list(field.coords)
    d_axis = axis_order.index('depth')
    r_axis = axis_order.index('range')
    t_axis = axis_order.index('time')
    data = np.moveaxis(np.asarray(field.data), [d_axis, r_axis, t_axis], [0, 1, 2])
    depths = np.asarray(field.coords['depth'], dtype=float)
    ranges = np.asarray(field.coords['range'], dtype=float)
    times = np.asarray(field.coords['time'], dtype=float)
    n_t = times.size

    if frame_stride is None:
        frame_stride = max(1, n_t // 300)
    frame_idx = np.arange(0, n_t, frame_stride)
    n_frames = frame_idx.size

    if p_max is None:
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            p_max = 1.0
        else:
            p_max = float(np.percentile(np.abs(finite), 99.5))
            if p_max <= 0:
                p_max = float(np.max(np.abs(finite))) or 1.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure

    # ``imshow`` is dramatically faster than ``pcolormesh`` for animation —
    # one set_array per frame vs full mesh re-tesselation.
    im = ax.imshow(
        data[:, :, frame_idx[0]],
        extent=_imshow_extent(ranges, depths),
        aspect=aspect,
        cmap=cmap,
        vmin=-p_max, vmax=p_max,
        origin='upper',
        zorder=1,
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Pressure (source-normalised)')

    ax.set_xlabel('Range (km)')
    ax.set_ylabel('Depth (m)')
    base_title = title if title is not None else (
        f"{field.model} — p(d, r, t)" if field.model else "p(d, r, t)"
    )

    if show_seafloor and env is not None:
        _overlay_seafloor(ax, env, ranges)

    if show_source and field.source_depths is not None and len(field.source_depths):
        ax.plot(
            [0.0], [float(field.source_depths[0])],
            zorder=ZORDER_SOURCE, **SOURCE_MARKER_STYLE,
        )

    time_label = ax.text(
        0.98, 0.96, '', transform=ax.transAxes,
        ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85),
        zorder=ZORDER_SOURCE + 1,
    )

    def _update(k: int):
        i = frame_idx[k]
        im.set_array(data[:, :, i])
        if show_time:
            time_label.set_text(f"t = {times[i] * 1000:.1f} ms")
        ax.set_title(base_title)
        return im, time_label

    ani = FuncAnimation(
        fig, _update, frames=n_frames,
        interval=1000.0 / max(fps, 1), blit=False, repeat=True,
    )
    return ani


def save_animation(
    field: 'Field',
    path,
    *,
    fps: int = 20,
    figsize: Tuple[float, float] = (8, 4),
    writer=None,
    **animate_kwargs,
):
    """Render a time-series :class:`Field` to a GIF or MP4 file.

    Wraps :func:`animate_field` with a fresh figure, the right
    matplotlib writer (inferred from ``path`` suffix when ``writer`` is
    ``None``), and closes the figure on the way out. All
    ``animate_field`` knobs are forwarded — e.g. ``aspect``, ``cmap``,
    ``frame_stride``, ``p_max``, ``title``, ``env``.

    Parameters
    ----------
    field : Field
        Time-series field (``kind='time_series'``).
    path : str or Path
        Output file. ``.gif`` → :class:`PillowWriter`; ``.mp4`` →
        ``'ffmpeg'`` (requires ffmpeg installed).
    fps : int, optional
        Playback frame rate. Default 20.
    figsize : tuple, optional
        Figure size (inches). Default ``(8, 4)``.
    writer : matplotlib animation writer, optional
        Override the suffix-inferred writer.

    Returns
    -------
    pathlib.Path
        The path written.
    """
    from pathlib import Path as _Path
    out = _Path(path)
    if writer is None:
        suffix = out.suffix.lower()
        if suffix == '.gif':
            from matplotlib.animation import PillowWriter
            writer = PillowWriter(fps=fps)
        elif suffix in ('.mp4', '.mov', '.mkv'):
            writer = 'ffmpeg'
        else:
            raise ConfigurationError(
                f"save_animation: cannot infer writer for suffix "
                f"{suffix!r} — pass `writer=` explicitly. Known "
                "suffixes: .gif, .mp4, .mov, .mkv."
            )
    fig, ax = plt.subplots(figsize=figsize)
    try:
        ani = animate_field(field, fps=fps, ax=ax, **animate_kwargs)
        ani.save(str(out), writer=writer)
    finally:
        plt.close(fig)
    return out


def plot_time_snapshots(
    fields,
    times_s: Sequence[float],
    *,
    env: Optional[Environment] = None,
    cmap: str = 'RdBu_r',
    aspect=None,
    p_max=None,
    figsize_per_panel: Tuple[float, float] = (3.2, 2.8),
    title: Optional[str] = None,
):
    """Snapshot grid: per-model rows × per-time columns of ``p(d, r, t)``.

    Time-series analogue of :func:`compare_models`. Each row is one
    field, each column is the time slice nearest ``times_s[j]``. Useful
    for multi-solver comparison panels where the same propagation event
    is captured at matched wall-clock times across rows.

    Parameters
    ----------
    fields : Mapping[str, Field] or Sequence[(str, Field)]
        Time-series fields keyed by display name. Insertion order =
        row order.
    times_s : sequence of float
        Wall-clock times to sample (s). Each model's nearest time bin
        is picked independently.
    env : Environment, optional
        When supplied, the seafloor overlay is drawn on every panel.
    cmap : str, optional
        Diverging colormap for ±pressure. Default ``'RdBu_r'``.
    aspect : str or float, optional
        Passed to ``ax.imshow``. ``None`` (default) picks ``1/1000.0``
        when the range axis spans far more than the depth axis (so
        wavefronts stay round when range is in km and depth in m),
        otherwise ``'auto'``.
    p_max : float or sequence, optional
        Symmetric colour-scale ``[-p_max, +p_max]``. ``None`` (default)
        picks a per-row 99.5th-percentile of ``|data|`` so each model's
        absolute amplitude normalisation doesn't wash out the others.
        Pass a scalar for a global scale, or a sequence of length
        ``n_models`` for explicit per-row scales.
    figsize_per_panel : tuple, optional
        ``(width, height)`` inches per snapshot panel. Default ``(3.2,
        2.8)``.
    title : str, optional
        ``fig.suptitle`` text.

    Returns
    -------
    matplotlib.figure.Figure, numpy.ndarray[matplotlib.axes.Axes]
        Figure and 2-D axes array (shape ``(n_models, n_times)``).
    """
    # Accept dict or list-of-(name, field).
    if hasattr(fields, 'items'):
        rows = list(fields.items())
    else:
        rows = list(fields)
    n_models = len(rows)
    n_times = len(times_s)
    if n_models == 0 or n_times == 0:
        raise ConfigurationError("plot_time_snapshots: empty fields or times_s.")

    fig, axes = plt.subplots(
        n_models, n_times,
        figsize=(figsize_per_panel[0] * n_times,
                 figsize_per_panel[1] * n_models),
        squeeze=False,
    )

    # Per-row pmax derivation (default).
    if p_max is None:
        p_max_per_row = []
        for _, f in rows:
            finite = np.asarray(f.data)[np.isfinite(f.data)]
            p_max_per_row.append(
                float(np.percentile(np.abs(finite), 99.5))
                if finite.size else 1.0
            )
    elif np.isscalar(p_max):
        p_max_per_row = [float(p_max)] * n_models
    else:
        p_max_per_row = [float(v) for v in p_max]
        if len(p_max_per_row) != n_models:
            raise ConfigurationError(
                f"plot_time_snapshots: p_max sequence length "
                f"{len(p_max_per_row)} != n_models {n_models}."
            )

    for i, (name, field) in enumerate(rows):
        times = np.asarray(field.coords['time'])
        depths = np.asarray(field.coords['depth'])
        ranges = np.asarray(field.coords['range'])
        ax_order = list(field.coords)
        d_ax = ax_order.index('depth')
        r_ax = ax_order.index('range')
        t_ax = ax_order.index('time')
        data3 = np.moveaxis(
            np.asarray(field.data), [d_ax, r_ax, t_ax], [0, 1, 2],
        )
        # Decide aspect ratio once per row from the data extent.
        if aspect is None:
            range_span = float(ranges[-1] - ranges[0])
            depth_span = float(depths[-1] - depths[0])
            row_aspect = (
                1.0 / 1000.0
                if range_span > 10.0 * depth_span
                else 'auto'
            )
        else:
            row_aspect = aspect

        pm = p_max_per_row[i]
        for j, t_target in enumerate(times_s):
            k = int(np.argmin(np.abs(times - t_target)))
            slab = data3[:, :, k]
            ax = axes[i, j]
            ax.imshow(
                slab,
                extent=_imshow_extent(ranges, depths),
                aspect=row_aspect, cmap=cmap,
                vmin=-pm, vmax=pm, origin='upper',
            )
            if env is not None:
                _overlay_seafloor(ax, env, ranges)
                ax.set_ylim(float(env.depth) * 1.05, 0)
            else:
                ax.set_ylim(depths[-1], depths[0])
            if field.source_depths is not None and len(field.source_depths):
                ax.plot([0.0], [float(field.source_depths[0])],
                        zorder=6, **SOURCE_MARKER_STYLE)
            ax.set_xlim(0, ranges[-1] / 1000)
            if i == 0:
                ax.set_title(f"t = {times[k] * 1000:.0f} ms", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{name}\nDepth (m)", fontsize=10)
            if i == n_models - 1:
                ax.set_xlabel('Range (km)', fontsize=9)
            else:
                ax.set_xticklabels([])

    if title is not None:
        fig.suptitle(title, fontsize=11, fontweight='bold')
    fig.tight_layout()
    return fig, axes
