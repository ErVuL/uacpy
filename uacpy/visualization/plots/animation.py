"""Time-domain field animation and snapshot montages."""

from __future__ import annotations


import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence, Tuple

from uacpy.core.environment import Environment
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import Field, ResultStack
from uacpy.core.units import m_to_km
from uacpy.visualization.plots._common import ZORDER_SOURCE, _draw_geometry, _imshow_extent, _overlay_seafloor, typed_plot_error


_TIME_AXES = ('depth', 'range', 'time')


def _time_layout(field, caller: str, what: str = 'field'):
    """``(data, depths, ranges, times)`` of a time-series field in the
    canonical (depth, range, time) layout, whatever storage order
    ``field.coords`` declares. Refuses a field missing any of the three axes
    by name, so no caller indexes a coordinate it does not have."""
    coords = getattr(field, 'coords', {})
    missing = set(_TIME_AXES) - set(coords)
    if missing:
        raise ConfigurationError(
            f"{caller}: {what} is missing coord axes {sorted(missing)}. "
            f"Need depth, range, and time — got {list(coords)}.")
    order = list(coords)
    data = np.moveaxis(np.asarray(field.data),
                       [order.index(a) for a in _TIME_AXES], [0, 1, 2])
    return (data, *(np.asarray(coords[a], dtype=float) for a in _TIME_AXES))


def _pmax_percentile(data) -> float:
    """Symmetric colour limit for ±pressure: the 99.5th percentile of
    ``|data|`` over the finite samples, so the early-time near-source spike
    does not wash out later frames. Falls back to the maximum when that
    percentile is zero (a field silent almost everywhere) and to 1.0 when
    nothing is finite."""
    finite = np.asarray(data)[np.isfinite(data)]
    if finite.size == 0:
        return 1.0
    p = float(np.percentile(np.abs(finite), 99.5))
    return p if p > 0 else (float(np.max(np.abs(finite))) or 1.0)


@typed_plot_error
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
        Must have ``coords={'depth', 'range',
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
        Mark the source at range 0 and ``field.source_depths`` when the
        field carries them, as :func:`plot_field`'s ``source=`` does; the
        x axis widens so the marker stays on screen when the grid starts
        past r = 0.
    show_seafloor : bool, optional
        Overlay env.bathymetry (or env.depth) on every frame.
    show_time : bool, optional
        Draw the current frame time (ms) in a boxed label in the top-right
        corner of the axes.
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

    if not isinstance(field, Field) or 'time' not in getattr(field, 'coords', {}):
        raise ConfigurationError(
            "animate_field: needs a Field carrying a 'time' axis. "
            f"Got coords={tuple(getattr(field, 'coords', ()))!r}."
        )
    data, depths, ranges, times = _time_layout(field, 'animate_field')
    n_t = times.size

    if frame_stride is None:
        frame_stride = max(1, n_t // 300)
    # A stride of 0 divides by zero inside `np.arange`, and a negative one
    # yields an empty frame index that fails later as an IndexError blaming the
    # caller's arrays. The test is numeric, not `isinstance(int)`, so the
    # numpy integers a `range`/`shape` expression produces keep working.
    if frame_stride < 1:
        raise ConfigurationError(
            f"animate_field: frame_stride must be at least 1, got "
            f"{frame_stride!r}. It sub-samples the time axis — 1 renders every "
            f"sample, and None caps the animation at ~300 frames.")
    frame_idx = np.arange(0, n_t, frame_stride)
    n_frames = frame_idx.size

    if p_max is None:
        p_max = _pmax_percentile(data)

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

    if show_source:
        _draw_geometry(ax, field.source_depths)

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


@typed_plot_error
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
        Time-domain field (carrying a ``'time'`` axis).
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


@typed_plot_error
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
    fields : ResultStack, Mapping[str, Field], Sequence[(str, Field)], Field or Sequence[Field]
        Time-series fields, one row each, in the order given. A
        :class:`ResultStack` names its rows ``"<coordinate>=<value>"``; a
        mapping or a sequence of ``(name, field)`` pairs names them
        explicitly; a bare :class:`Field` or a sequence of fields is named
        by each field's ``model``.
    times_s : sequence of float
        Wall-clock times to sample (s). Each model's nearest time bin
        is picked independently.
    env : Environment, optional
        When supplied, the seafloor overlay is drawn on every panel.
    cmap : str, optional
        Diverging colormap for ±pressure. Default ``'RdBu_r'``.
    aspect : str or float, optional
        Passed to ``ax.imshow``. ``None`` (default) stretches a panel whose
        range axis spans more than ten times its depth axis to a 4:1 floor —
        at least a quarter as tall as it is wide — since drawn isotropically
        (``1/1000.0``, 1 m of depth as long as 1 m of range, wavefronts round
        with range in km and depth in m) such a panel is an unreadable sliver.
        Every other panel gets ``'auto'``. Pass a number to override, e.g.
        ``aspect=1/1000`` for a genuinely isotropic panel.
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
    # Accept the canonical multi-result container (ResultStack), a single
    # Field, a sequence of Fields (auto-named by model), a {name: field} dict,
    # or a sequence of (name, field) pairs.
    if isinstance(fields, ResultStack):
        rows = [(f"{fields.coordinate_name}={c:g}", slab) for c, slab in fields]
    elif hasattr(fields, 'items'):
        rows = list(fields.items())
    elif isinstance(fields, Field):
        rows = [(fields.model or '', fields)]
    else:
        rows = [
            (item.model or '', item) if isinstance(item, Field) else tuple(item)
            for item in fields
        ]
    n_models = len(rows)
    n_times = len(times_s)
    if n_models == 0 or n_times == 0:
        raise ConfigurationError(
            f"plot_time_snapshots: empty fields or times_s. Got "
            f"{n_models} field(s) and {n_times} snapshot time(s); both must "
            f"be non-empty — the grid is one row per field, one column per "
            f"time.")

    layouts = [_time_layout(field, 'plot_time_snapshots',
                            f"field {(name or type(field).__name__)!r}")
               for name, field in rows]

    fig, axes = plt.subplots(
        n_models, n_times,
        figsize=(figsize_per_panel[0] * n_times,
                 figsize_per_panel[1] * n_models),
        squeeze=False,
    )

    # Per-row pmax derivation (default).
    if p_max is None:
        p_max_per_row = [_pmax_percentile(data) for data, *_ in layouts]
    elif np.isscalar(p_max):
        p_max_per_row = [float(np.asarray(p_max).item())] * n_models
    else:
        p_max_per_row = [float(v) for v in p_max]
        if len(p_max_per_row) != n_models:
            raise ConfigurationError(
                f"plot_time_snapshots: p_max sequence length "
                f"{len(p_max_per_row)} != n_models {n_models}."
            )

    for i, ((name, field), (data3, depths, ranges, times)) in enumerate(
            zip(rows, layouts)):
        # Decide aspect ratio once per row from the data extent. The imshow
        # extent is km in x and m in y, so aspect = 1/1000 displays 1 m of
        # depth as long as 1 m of range — isotropic, wavefronts stay round.
        row_aspect: 'float | str'
        if aspect is None:
            range_span = float(ranges[-1] - ranges[0])
            depth_span = float(depths[-1] - depths[0])
            # The gate admits only long, shallow panels — range_span more
            # than ten times depth_span — so ratio < 0.1 throughout it, and an
            # isotropic panel (aspect 1/1000: 1 m of depth as long as 1 m of
            # range, so wavefronts stay round with range in km and depth in m)
            # would be a sliver at most 1:10. Every panel the gate admits is
            # therefore stretched to a 4:1 floor: a quarter as tall as it is
            # wide. Isotropic is not offered inside the gate for exactly that
            # reason — the former ``and ratio >= 0.25`` conjunct contradicted
            # the gate and so selected nothing — but a caller who wants it can
            # still pass ``aspect=1/1000`` explicitly.
            # A single receiver depth has depth_span == 0 and so no ratio to
            # scale by; it falls to 'auto' rather than dividing by zero.
            ratio = depth_span / range_span if range_span > 0 else 1.0
            if depth_span > 0 and range_span > 10.0 * depth_span:
                row_aspect = 0.25 / ratio / 1000.0
            else:
                row_aspect = 'auto'
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
            ax.set_xlim(0, float(m_to_km(ranges[-1])))
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
    # The source star sits on the left limit and widens it by its own half
    # width, measured in the panel's pixels — so it is drawn once the layout
    # has fixed the panel size.
    for (_, field), row in zip(rows, axes):
        for ax in row:
            _draw_geometry(ax, field.source_depths)
    return fig, axes
