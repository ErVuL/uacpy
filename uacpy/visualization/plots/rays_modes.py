"""Ray fans, arrival stems, mode functions/wavenumbers/heatmaps, reflection coefficients, source beam patterns and OASN covariance/replica plots."""

from __future__ import annotations


import numpy as np
from typing import Optional, Tuple

from uacpy.core.environment import Environment
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import Arrivals, Rays, Modes, Covariance, Replicas, ReflectionCoefficient
from uacpy.core.units import m_to_km
from uacpy.visualization.plots._common import ZORDER_LEGEND, ZORDER_RAYS, ZORDER_SURFACE, _overlay_seafloor, _draw_geometry, _draw_receiver_grid, _draw_result_credit, _plot_warn, fig_ax, typed_plot_error, invert_yaxis_once


#: Multipath class -> colour, for the ray fan and the arrival stems alike:
#: direct red, surface-reflected green, bottom-reflected blue, both black.
RAY_CLASS_COLOURS = {
    'direct': '#e53935',
    'surface': '#43a047',
    'bottom': '#1e88e5',
    'both': '#000000',
}


@typed_plot_error
def _plot_rays(
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
        raise ConfigurationError(f"_plot_rays: expected Rays, got {type(rays).__name__}")
    if color_by not in ('bounces', None):
        # A typo'd mode falling through to the monochrome branch would also
        # drop the per-class legend, so the fan would look like a deliberate
        # color_by=None call.
        raise ConfigurationError(
            f"_plot_rays: color_by={color_by!r} is not a colouring mode; pass "
            "'bounces' (colour by multipath class) or None (one colour)."
        )
    _owns_fig = ax is None
    fig, ax = fig_ax(ax, figsize)

    color_map = RAY_CLASS_COLOURS
    bounce_counts = {'direct': 0, 'surface': 0, 'bottom': 0, 'both': 0}
    max_r_km = 0.0
    max_z = 0.0
    for ray in rays.rays:
        r = np.asarray(ray.get('r', []))
        z = np.asarray(ray.get('z', []))
        if r.size == 0:
            continue
        max_r_km = max(max_r_km, m_to_km(float(np.max(r))))
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
        # color_by=None paints the whole fan one colour; the bottom-class blue
        # doubles as that neutral colour.
        color = color_map[kind] if color_by == 'bounces' else color_map['bottom']
        ax.plot(m_to_km(r), z, color=color, alpha=alpha,
                linewidth=linewidth, solid_capstyle='round',
                zorder=ZORDER_RAYS, **mpl_kw)

    invert_yaxis_once(ax)
    depth_for_lim = max_z
    if env is not None:
        depth_for_lim = max(depth_for_lim, float(env.depth))
    if (show_receivers and rays.receiver_ranges is not None
            and rays.receiver_depths is not None and rays.receiver_depths.size):
        # Drawn receiver markers below the deepest ray stay inside the depth
        # axis, matching the x-limit margin that keeps an at-max-range
        # receiver visible.
        depth_for_lim = max(depth_for_lim, float(np.max(rays.receiver_depths)))
    if depth_for_lim > 0:
        # Depth increases downward, so bottom > top. The negative top leaves a
        # sliver of headroom above z = 0 for the surface line and surface-bounce
        # turning points, which otherwise sit exactly on the spine.
        ax.set_ylim(depth_for_lim * 1.08, -depth_for_lim * 0.04)

    if env is not None:
        # Surface line styled to match the AT convention.
        ax.axhline(0, color='steelblue', linewidth=1.5, alpha=0.55,
                   zorder=ZORDER_SURFACE)
        # Anchor the seafloor overlay from x=0 (source range) out to where
        # the x-axis will END, not to the furthest receiver: the limit below
        # adds a margin so an at-max-range receiver is not clipped to the
        # spine, and a fill anchored on the receiver alone stops short of it,
        # leaving bare chart under the rays past the receiver — the sliver
        # this anchoring exists to avoid.
        if rays.receiver_ranges is not None and len(rays.receiver_ranges):
            r_hi = float(np.max(rays.receiver_ranges)) * _RECEIVER_EDGE_MARGIN
        else:
            r_hi = max_r_km * 1000.0
        ranges_for_overlay = np.array([0.0, r_hi])
        _overlay_seafloor(ax, env, ranges_for_overlay)

    if show_receivers and rays.receiver_ranges is not None and rays.receiver_depths is not None:
        # Markers are slightly smaller than on the env cross-section —
        # receivers are sampling points here, not the visual focus.
        rr_km = _draw_receiver_grid(ax, rays.receiver_ranges,
                                    rays.receiver_depths, max_markersize=7)
        # x-axis spans the receiver extent with a small right margin so a
        # receiver sitting at the max range isn't clipped to the spine. The
        # seafloor overlay above is anchored to the same margin.
        r_max = float(np.max(rr_km))
        ax.set_xlim(0.0,
                    r_max * _RECEIVER_EDGE_MARGIN if r_max > 0 else 1.0)
    if show_source and rays.source_depths is not None and rays.source_depths.size:
        # Slightly larger star than the other panels — it has to read against
        # a dense ray fan.
        _draw_geometry(ax, rays.source_depths, source_markersize_bonus=2)

    if show_legend and color_by == 'bounces':
        import matplotlib.lines as mlines
        handles = [
            mlines.Line2D([], [], color=col, linewidth=2,
                          label=f"{kind} ({bounce_counts[kind]})")
            for kind, col in color_map.items()
            if bounce_counts[kind] > 0
        ]
        if handles:
            # 'best', not a fixed corner: the receiver sits at the maximum
            # range, and on a bottom-mounted geometry at the seabed too, so
            # 'lower right' lands exactly on the marker the key explains.
            legend = ax.legend(handles=handles, loc='best',
                               fontsize=9, framealpha=0.85)
            # Matplotlib defaults a legend to zorder 5, under this package's
            # seabed fill, seafloor line, receivers and source: the key ends
            # up drawn beneath the picture it explains.
            legend.set_zorder(ZORDER_LEGEND)

    ax.set_xlabel('Range (km)')
    ax.set_ylabel('Depth (m)')
    ax.grid(True, alpha=0.3)
    ax.set_title(title or ('Eigenrays' if rays.is_eigen else 'Ray fan'))
    if _owns_fig:
        _draw_result_credit(fig, rays, env=env)
    return fig, ax


@typed_plot_error
def _plot_arrivals(
    arrivals: Arrivals,
    ax=None,
    *,
    figsize: Tuple[float, float] = (10, 4),
    title: Optional[str] = None,
):
    """Stem plot of arrivals: received level vs delay, by multipath class.

    Colour palette matches :func:`_plot_rays`: direct = red,
    surface = green, bottom = blue, both = black. Each arrival is drawn
    as a vertical stem plus a head marker.

    Stem height is the level that reaches the receiver, absorption
    included: Bellhop keeps the volume loss in the imaginary travel time,
    so the amplitude column alone stands an absorbed path at its lossless
    height. The delay axis spans the energy
    (:meth:`Arrivals.energy_support`), which one faint straggler cannot
    stretch the way it stretches the first-to-last range, and the legend
    counts the arrivals past the end and names how far they run — off the
    axis they leave no sign of themselves, where an outlier on a colour
    scale at least still paints a pixel."""
    if not isinstance(arrivals, Arrivals):
        raise ConfigurationError(
            f"_plot_arrivals: expected Arrivals, got {type(arrivals).__name__}"
        )
    _owns_fig = ax is None
    fig, ax = fig_ax(ax, figsize)
    color_map = RAY_CLASS_COLOURS
    counts = {k: 0 for k in color_map}
    delays_ms = []
    beyond = []
    hi = 0.0
    # Stem heights are what REACHES the receiver. Bellhop keeps volume
    # absorption in the imaginary travel time, not in the amplitude column
    # (``Arrivals._arrival_power``), so drawing the column alone stands a
    # heavily absorbed late path at its lossless height — on a 1 km 40 kHz
    # link the second bounce cluster draws five times taller than it arrives.
    levels = np.sqrt(arrivals._arrival_power()) if arrivals.arrivals else None
    for index, a in enumerate(arrivals.arrivals):
        kind = a.get('kind', 'direct')
        col = color_map.get(kind, '#1e88e5')
        d_ms = a['delay'] * 1000.0
        delays_ms.append(d_ms)
        level = float(levels[index]) if levels is not None else a['amplitude']
        ax.vlines(d_ms, 0, level, colors=col, lw=1.5, alpha=0.85)
        ax.plot(d_ms, level, 'o', color=col, markersize=4,
                markeredgecolor='black', markeredgewidth=0.4)
        counts[kind] += 1
    if delays_ms:
        # The axis follows the ENERGY, not the last ray. A peak-to-peak span
        # is an extremum: one faint straggler stretches it without bound —
        # 8.06 s against a 0.9 s energy span on that same link — and squeezes
        # every arrival that carries something into a few pixels. What falls
        # outside is named below rather than dropped in silence, because an
        # arrival off the end of the axis leaves no trace of itself the way
        # an outlier on a colour scale still does.
        first = min(delays_ms)
        support_ms = arrivals.energy_support() * 1000.0
        span = support_ms if support_ms > 0 else (max(delays_ms) - first)
        lo = first - 0.05 * (span or 1)
        hi = first + span + 0.05 * (span or 1)
        ax.set_xlim(lo, hi)
        beyond = [d for d in delays_ms if d > hi]
    ax.set_xlabel('Delay (ms)')
    ax.set_ylabel('Received amplitude (re unit source)')
    ax.grid(True, alpha=0.3)
    # Legend with per-class counts (skip empty classes).
    import matplotlib.lines as mlines
    handles = [
        mlines.Line2D([], [], color=col, marker='o', linestyle='-',
                      label=f"{kind} ({counts[kind]})")
        for kind, col in color_map.items() if counts[kind] > 0
    ]
    if beyond:
        handles.append(mlines.Line2D(
            [], [], linestyle='none', marker='',
            label=f"+{len(beyond)} beyond {hi / 1000.0:.2f} s "
                  f"(to {max(beyond) / 1000.0:.2f} s)"))
    if handles:
        ax.legend(handles=handles, loc='upper right', fontsize=9,
                  framealpha=0.85)
    if title:
        ax.set_title(title)
    if _owns_fig:
        _draw_result_credit(fig, arrivals, env=None)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────


@typed_plot_error
def _plot_mode_functions(
    modes: Modes,
    n_modes: Optional[int] = None,
    ax=None,
    *,
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = None,
    show_imaginary: bool = False,
):
    """Plot the first ``n_modes`` mode shapes ``ψ_m(z)`` as overlaid 1-D curves.

    ``show_imaginary`` overlays ``Im(ψ_m)`` as a dashed line in the matching
    colour — meaningful only for a complex-arithmetic solve (``backend=
    'krakenc'``), where leaky modes carry a non-zero imaginary part."""
    if not isinstance(modes, Modes):
        raise ConfigurationError(
            f"_plot_mode_functions: expected Modes, got {type(modes).__name__}"
        )
    if show_imaginary and not np.iscomplexobj(modes.phi):
        raise ConfigurationError(
            "_plot_mode_functions: show_imaginary=True needs complex mode "
            "functions; this Modes result is real (use backend='krakenc')."
        )
    _owns_fig = ax is None
    fig, ax = fig_ax(ax, figsize)
    n_modes = modes.n_modes if n_modes is None else min(int(n_modes), modes.n_modes)
    for m in range(n_modes):
        psi = np.asarray(modes.phi[:, m])
        line, = ax.plot(psi.real if np.iscomplexobj(psi) else psi,
                        modes.depths, label=f"m={m+1}", linewidth=1.0)
        if show_imaginary:
            ax.plot(psi.imag, modes.depths, linestyle='--', linewidth=0.9,
                    color=line.get_color())
    ax.set_xlabel(r'$\psi_m(z)$')
    ax.set_ylabel('Depth (m)')
    invert_yaxis_once(ax)
    ax.grid(True, alpha=0.3)
    if n_modes <= 12:
        ax.legend(fontsize=8, loc='best')
    ax.set_title(title or f"Mode functions (n={n_modes})")
    if _owns_fig:
        _draw_result_credit(fig, modes, env=None)
    return fig, ax


@typed_plot_error
def plot_mode_wavenumbers(
    modes: Modes,
    ax=None,
    *,
    figsize: Tuple[float, float] = (8, 5),
    title: Optional[str] = None,
):
    """Scatter ``Re(k_m)`` vs mode index; overlay imaginary part if non-zero."""
    if not isinstance(modes, Modes):
        raise ConfigurationError(
            f"plot_mode_wavenumbers: expected Modes, got {type(modes).__name__}"
        )
    _owns_fig = ax is None
    fig, ax = fig_ax(ax, figsize)
    idx = np.arange(1, modes.n_modes + 1)
    k = np.asarray(modes.k)
    ax.plot(idx, k.real, 'o-')
    if np.any(np.abs(k.imag) > 0):
        ax2 = ax.twinx()
        ax2.plot(idx, k.imag, 's--', color='C1')
        ax2.set_ylabel(r'$\mathrm{Im}(k_m)$ (1/m)')
    ax.set_xlabel('Mode index')
    ax.set_ylabel(r'$\mathrm{Re}(k_m)$ (1/m)')
    ax.grid(True, alpha=0.3)
    ax.set_title(title or 'Modal wavenumbers')
    if _owns_fig:
        _draw_result_credit(fig, modes, env=None)
    return fig, ax


@typed_plot_error
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

    ``mode_range=(start, stop)`` selects a half-open mode-index slice with
    ``0 <= start < stop`` (``stop`` past the last mode simply clamps), and is
    an alternative to ``n_modes``, not a modifier of it — passing both is a
    :class:`~uacpy.core.exceptions.ConfigurationError`.
    ``normalize=True`` (default) rescales each column to peak ``±1`` so
    high-order modes don't disappear next to the dominant low-order ones.
    """
    if not isinstance(modes, Modes):
        raise ConfigurationError(
            f"plot_modes_heatmap: expected Modes, got {type(modes).__name__}"
        )
    if mode_range is not None:
        if n_modes is not None:
            # mode_range takes the slice wholesale, so a call passing both used
            # to plot the range and drop n_modes without saying so.
            raise ConfigurationError(
                f"plot_modes_heatmap: got both n_modes={n_modes!r} and "
                f"mode_range={mode_range!r}; pass one — n_modes for the first "
                "N modes, mode_range=(start, stop) for a slice."
            )
        start, stop = mode_range
        start, stop = int(start), int(stop)
        if not 0 <= start < stop:
            # A negative start wraps under numpy slicing and start >= stop
            # selects nothing, both of which reach pcolormesh as a shape
            # mismatch rather than as an error about the range.
            raise ConfigurationError(
                f"plot_modes_heatmap: mode_range={mode_range!r} must be a "
                "half-open (start, stop) with 0 <= start < stop."
            )
        if start >= modes.n_modes:
            raise ConfigurationError(
                f"plot_modes_heatmap: mode_range={mode_range!r} starts past "
                f"the {modes.n_modes} mode(s) this result carries."
            )
        stop = min(stop, modes.n_modes)
    _owns_fig = ax is None
    fig, ax = fig_ax(ax, figsize)
    if mode_range is None:
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
    # ``shading='nearest'`` centres each column on its integer mode index,
    # so no (n+1)-long edge array is needed.
    im = ax.pcolormesh(idx, modes.depths, phi, cmap=cmap,
                       shading='nearest', vmin=vmin, vmax=vmax)
    fig.colorbar(
        im, ax=ax,
        label='Normalised amplitude' if normalize else r'$\psi_m(z)$',
    )
    ax.set_xlabel('Mode index')
    # A mode index is an integer — mode 4.5 does not exist. On a short span the
    # default locator subdivides: mode_range=(3, 7) drew 3.5, 4.5, 5.5, 6.5 and
    # 7.5, five of nine ticks naming modes the field does not contain.
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel('Depth (m)')
    invert_yaxis_once(ax)
    # A Modes carrying no f0 gets no frequency in the title rather than a
    # fabricated '@ 0.0 Hz', which a published figure would assert as fact.
    auto = (f'Mode shapes — {n_plot} modes @ {modes.f0:.1f} Hz'
            if modes.f0 is not None else f'Mode shapes — {n_plot} modes')
    ax.set_title(title or auto)
    if _owns_fig:
        _draw_result_credit(fig, modes, env=None)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Reflection coefficient
# ─────────────────────────────────────────────────────────────────────────────


@typed_plot_error
def _plot_reflection_coefficient(
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
        raise ConfigurationError(
            f"_plot_reflection_coefficient: expected ReflectionCoefficient, "
            f"got {type(rc).__name__}"
        )
    if rc.is_broadband:
        _owns_fig = ax is None
        fig, ax = fig_ax(ax, figsize)
        freqs = np.asarray(rc.frequencies, dtype=float)
        im = ax.pcolormesh(
            freqs / 1000.0, rc.theta, rc.R,
            shading='nearest', cmap='viridis',
        )
        fig.colorbar(im, ax=ax, label='|R|')
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel('Grazing angle (°)')
        ax.set_title(title or 'Reflection coefficient |R(θ, f)|')
        if _owns_fig:
            _draw_result_credit(fig, rc, env=None)
        return fig, ax

    _owns_fig = ax is None
    fig, ax = fig_ax(ax, figsize)
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
    if _owns_fig:
        _draw_result_credit(fig, rc, env=None)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Covariance / Replicas
# ─────────────────────────────────────────────────────────────────────────────


@typed_plot_error
def _plot_covariance(
    cov: Covariance,
    ax=None,
    *,
    freq_idx: int = 0,
    figsize: Tuple[float, float] = (6, 5),
    title: Optional[str] = None,
):
    """Heatmap of one covariance slice ``|C[freq_idx, :, :]|``."""
    if not isinstance(cov, Covariance):
        raise ConfigurationError(
            f"_plot_covariance: expected Covariance, got {type(cov).__name__}"
        )
    _owns_fig = ax is None
    fig, ax = fig_ax(ax, figsize)
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
    if _owns_fig:
        _draw_result_credit(fig, cov, env=None)
    return fig, ax


@typed_plot_error
def _plot_replicas(
    rep: Replicas,
    ax=None,
    *,
    freq_idx: int = 0,
    sensor_idx: int = 0,
    figsize: Tuple[float, float] = (8, 5),
    title: Optional[str] = None,
):
    """Magnitude of replica response across (z, x) at the first y node."""
    if not isinstance(rep, Replicas):
        raise ConfigurationError(
            f"_plot_replicas: expected Replicas, got {type(rep).__name__}"
        )
    _owns_fig = ax is None
    fig, ax = fig_ax(ax, figsize)
    # replicas is (n_freq, n_zr, n_xr, n_yr, n_rcv): take one frequency and one
    # array element, and cut the candidate-source grid at its first y node.
    R = np.abs(rep.replicas[freq_idx, :, :, 0, sensor_idx])
    im = ax.pcolormesh(
        rep.replica_x, rep.replica_z, R,
        shading='nearest', cmap='magma',
    )
    fig.colorbar(im, ax=ax, label='|R|')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    invert_yaxis_once(ax)
    if title:
        ax.set_title(title)
    if _owns_fig:
        _draw_result_credit(fig, rep, env=None)
    return fig, ax


def _polar_fig_ax(ax, figsize):
    """``fig_ax`` for a polar plotter: a fresh polar axes when ``ax is None``.

    ``fig_ax`` builds a rectilinear subplot, which silently ignores ``theta``
    as an angle, so a polar plotter needs its own constructor. A supplied
    rectilinear ``ax`` is refused rather than drawn into, because the result
    would be a line of radians against dB that still looks like a plot."""
    import matplotlib.pyplot as plt
    if ax is None:
        fig = plt.figure(figsize=figsize)
        return fig, fig.add_subplot(projection='polar')
    if ax.name != 'polar':
        raise ConfigurationError(
            f"plot_beam_pattern: polar=True needs a polar axes, but ax= is "
            f"a '{ax.name}' axes.",
            remediation="Build it with "
                        "fig.add_subplot(projection='polar'), or pass "
                        "polar=False to draw level against angle instead.",
        )
    return ax.figure, ax


def _resolve_beam_pattern(pattern) -> np.ndarray:
    """Return the ``(N, 2)`` ``[angle_deg, level_dB]`` table ``pattern`` names.

    Accepts what :attr:`uacpy.Source.beam_pattern` accepts — an array, a
    ``.sbp`` path, or ``None`` — so plotting a source and plotting a file on
    disk go through one code path. ``None`` becomes the flat table
    ``ReadPat`` synthesises for an omni source (``misc/beampattern.f90:52-53``
    writes exactly ``[-180, 0], [180, 0]``), which keeps "no pattern" a
    drawable answer rather than an error."""
    from pathlib import Path

    if pattern is None:
        return np.array([[-180.0, 0.0], [180.0, 0.0]])
    if isinstance(pattern, (str, Path)):
        from uacpy.io.refl_io import read_source_beam_pattern
        return np.asarray(read_source_beam_pattern(pattern), dtype=float)
    table = np.asarray(pattern, dtype=float)
    if table.ndim != 2 or table.shape[1] != 2:
        raise ConfigurationError(
            f"plot_beam_pattern: a beam pattern is an (N, 2) "
            f"[angle_deg, level_dB] table; got shape {table.shape}.",
            remediation="Stack the two columns with "
                        "np.column_stack([angles_deg, levels_db]).",
        )
    if len(table) < 2:
        raise ConfigurationError(
            f"plot_beam_pattern: a beam pattern needs at least 2 "
            f"(angle, level) rows to draw; got {len(table)}.",
            remediation="Pass None for an omnidirectional source.",
        )
    return table


def _mirror_about_zero(angles: np.ndarray,
                       levels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Complete a one-sided table by reflecting it through 0°.

    Only a table that stays on one side of 0° is completed; one that already
    straddles it is returned untouched, so ``mirror=True`` is a no-op on a
    full -180…180 pattern rather than a second reflection."""
    if angles.min() < 0.0 < angles.max():
        return angles, levels
    if angles.min() >= 0.0:
        off_axis = angles > 0.0
        other_angles = -angles[off_axis][::-1]
        other_levels = levels[off_axis][::-1]
        return (np.concatenate([other_angles, angles]),
                np.concatenate([other_levels, levels]))
    off_axis = angles < 0.0
    other_angles = -angles[off_axis][::-1]
    other_levels = levels[off_axis][::-1]
    return (np.concatenate([angles, other_angles]),
            np.concatenate([levels, other_levels]))


#: The sector both renderings span, in signed degrees. It is the whole of
#: what a launch fan can reach and nothing else: a launch steeper than +/-90°
#: has ``COS(alpha) < 0`` in ``ray2D(1)%t`` (``Bellhop/bellhop.f90:453``) and
#: traces to NEGATIVE range only, so it never enters the ``r > 0`` the field
#: is evaluated on — measured, ``alpha = +/-127.5°`` gives ``r`` in
#: ``[-6000, 0]`` while ``alpha = +/-42.5°`` reaches ``+6000``. Outside it
#: there is no launch for a response to describe, so there is nothing to
#: choose between: every pattern draws on the same axes and two of them can
#: be compared by eye.
_LAUNCH_FAN_DEG = (-90.0, 90.0)


#: Angle ticks across the fan. 30°, not the 45° a linear axis would pick:
#: 45 divides 180 into four and so labels neither +/-90, and those two ends
#: are what a reader checks first — whether the table reaches the horizontal
#: and whether it reaches straight down.
_BEAM_PATTERN_TICKS = np.arange(_LAUNCH_FAN_DEG[0], _LAUNCH_FAN_DEG[1] + 1e-9,
                                30.0)
_BEAM_PATTERN_TICK_LABELS = [f'{t:g}\u00b0' for t in _BEAM_PATTERN_TICKS]


#: Right-hand margin on the receiver extent, so a receiver at the maximum
#: range is not clipped to the spine. The seafloor overlay and the x-limit
#: both use it — anchoring them differently is what left a bare strip.
_RECEIVER_EDGE_MARGIN = 1.03


def _densify_in_angle(angles, levels, step_deg: float = 1.0):
    """Resample a beam-pattern table onto a ``step_deg`` angle grid.

    Keeps every tabulated angle, and interpolates between them the way the
    engine does: the table is converted to an amplitude factor
    ``10**(dB/20)`` before it is read (``misc/beampattern.f90:59``), so the
    engine is piecewise-linear in AMPLITUDE, not in dB. Interpolating the dB
    column directly would bow each flank the wrong way — 14.7 dB out at worst
    on a 0/-35 dB segment. A table whose own step divides ``step_deg`` comes
    back untouched; one sampled finer but off that grid keeps every row and
    gains collinear points between them, which changes the vertex count and
    not the curve.

    Angles are sorted, so a table listed out of order draws as the curve its
    rows describe. Note that ``ReadPat`` (``misc/beampattern.f90:56``)
    REFUSES such a table — the engine requires strictly increasing angles —
    so a file that draws here may not run.
    """
    angles = np.asarray(angles, dtype=float)
    levels = np.asarray(levels, dtype=float)
    if angles.size < 2:
        return angles, levels
    order = np.argsort(angles)
    angles, levels = angles[order], levels[order]
    if angles[-1] <= angles[0]:
        raise ConfigurationError(
            f"plot_beam_pattern: the table's angles are all "
            f"{angles[0]:g}°, so it spans no angle and there is no pattern to "
            f"draw. Give at least two distinct angles.")
    n = int(np.ceil((angles[-1] - angles[0]) / float(step_deg))) + 1
    grid = np.union1d(angles, np.linspace(angles[0], angles[-1], max(n, 2)))
    amplitude = np.interp(grid, angles, 10.0 ** (levels / 20.0))
    with np.errstate(divide='ignore'):
        return grid, 20.0 * np.log10(amplitude)


@typed_plot_error
def plot_beam_pattern(
    pattern=None,
    ax=None,
    *,
    polar: bool = True,
    mirror: bool = False,
    fill: bool = True,
    figsize: Tuple[float, float] = (6.5, 6.5),
    title: Optional[str] = None,
    rmin: Optional[float] = None,
    **kwargs,
):
    """Plot a source beam pattern — the ``.sbp`` directivity table.

    Parameters
    ----------
    pattern : ndarray or path-like or None
        What :attr:`uacpy.Source.beam_pattern` holds: an ``(N, 2)``
        ``[angle_deg, level_dB]`` table, a ``.sbp`` file, or ``None`` for an
        omnidirectional source (drawn as the flat 0 dB circle Bellhop itself
        substitutes).
    polar : bool, default True
        Draw on polar axes oriented like the field: 0° along increasing
        range, positive angles downward. ``False`` draws level against angle
        on rectilinear axes, where a dB scale is linear and so the deep end
        of the pattern is easier to read off. Both renderings draw the same
        curve — the table interpolated the way the engine reads it.
    mirror : bool, default False
        Reflect a one-sided table through 0° before drawing. Off by default
        because no engine mirrors: ``ReadPat``
        (``misc/beampattern.f90:43-46``) reads the table verbatim, and
        ``bellhop.f90:269-274`` interpolates it with the index clamped but the
        weight unclamped, so the angles a half table omits are extrapolated
        rather than reflected.
    fill : bool, default True
        Shade the area between the curve and the radial floor, so a lobe reads
        as a lobe rather than as the spokes a top-hat pattern degenerates to.
        Polar only.
    rmin : float, optional
        Inner radius of the polar axes in dB. Defaults to just below the
        table's own minimum, so the whole pattern is visible.

    Notes
    -----
    The angle axis is Bellhop's launch declination ``alpha``, in degrees —
    the same convention and the same units the fan is spelled in, which is
    why :meth:`Bellhop._check_beam_pattern_spans_the_fan` can compare the two
    directly. ``ray2D(1)%t = [COS(alpha), SIN(alpha)]/c``
    (``Bellhop/bellhop.f90:453``) over a depth axis that is positive downward
    sends ``alpha > 0`` deeper, so the polar axes run clockwise from due east
    and a lobe drawn below the horizontal is a lobe that ensonifies the
    depths below the source in the field plot.

    Levels are dB re peak. Bellhop applies them as an *amplitude* factor,
    ``10**(dB/20)`` (``misc/beampattern.f90:59``), despite its print header
    calling the column "Power".
    """
    table = _resolve_beam_pattern(pattern)
    angles, levels = table[:, 0], table[:, 1]
    if mirror:
        angles, levels = _mirror_about_zero(angles, levels)

    lo, hi = float(angles.min()), float(angles.max())
    if lo > -90.0 + 1e-9 or hi < 90.0 - 1e-9:
        _plot_warn(
            f"plot_beam_pattern: the pattern spans [{lo:g}, {hi:g}]° and so "
            f"does not cover the [-90, 90]° a launch fan can reach. Bellhop "
            f"neither mirrors nor wraps a partial table — it extrapolates "
            f"past both ends on linear amplitude (bellhop.f90:273) — so the "
            f"uncovered angles are undefined, not symmetric, and "
            f"Bellhop._check_beam_pattern_spans_the_fan rejects any alpha "
            f"reaching into them. Pass mirror=True to reflect the table "
            f"through 0°.")

    level_span = float(levels.max() - levels.min())
    floor = (levels.min() - 0.05 * level_span if level_span > 1e-9
             else levels.max() - 10.0)
    default_title = ('Source beam pattern — omnidirectional'
                     if pattern is None else 'Source beam pattern')

    fan_lo, fan_hi = _LAUNCH_FAN_DEG
    # Reported before the branch, not inside one: both renderings limit the
    # angle axis to the fan, so both hide the same rows.
    drawn = (angles >= fan_lo - 1e-9) & (angles <= fan_hi + 1e-9)
    hidden = ~drawn
    if hidden.any() and drawn.any() and levels[hidden].max() > levels[drawn].max() + 1e-9:
        _plot_warn(
            f"plot_beam_pattern: the strongest level in the table "
            f"({levels[hidden].max():g} dB at "
            f"{angles[hidden][np.argmax(levels[hidden])]:g}°) lies outside "
            f"the [{fan_lo:g}, {fan_hi:g}]° a launch fan can reach, so the "
            f"main lobe is not on this plot — and Bellhop would never launch "
            f"into it either. Aim the table into the fan, or pass "
            f"mirror=True if it was written for the other side of 0°.")

    # Sample the curve ALONG the angle axis before either branch draws it.
    # Both renderings join consecutive samples with a straight line in the
    # coordinates they draw in, and neither is what the engine reads: the
    # table is converted to an amplitude factor first
    # (``misc/beampattern.f90:59``; Bellhop interpolates it inline at
    # ``bellhop.f90:267-274``). On a polar axes a straight screen segment
    # also cuts a chord across the arc — a two-row 0 dB table, omnidirectional
    # to Bellhop, drew as a chord through the origin, i.e. a deep broadside
    # null. Densifying here keeps the two views showing the same curve;
    # doing it in one branch only left them 11.2 dB apart.
    angles, levels = _densify_in_angle(angles, levels)

    if not polar:
        fig, ax = fig_ax(ax, figsize)
        ax.plot(angles, levels, **kwargs)
        ax.set_xlabel('Launch angle (°)')
        ax.set_ylabel('Level (dB re peak)')
        # The same limit the polar axes takes. Which launch angles exist is a
        # fact about the fan, not about how the response is drawn.
        ax.set_xlim(fan_lo, fan_hi)
        ax.grid(True, alpha=0.3)
        ax.set_title(title or default_title)
        return fig, ax

    fig, ax = _polar_fig_ax(ax, figsize)
    theta = np.deg2rad(angles)
    inner = floor if rmin is None else rmin
    line, = ax.plot(theta, levels, **kwargs)
    if fill:
        # A pattern whose sidelobes sit near the radial floor draws as bare
        # spokes: the floor is the origin, so everything but the main lobe has
        # zero radius. Shading the area under the curve gives the lobe back the
        # width the table gave it.
        ax.fill_between(theta, inner, levels,
                        color=line.get_color(), alpha=0.15, linewidth=0)
    # Due east = 0° = increasing range, then clockwise so a positive angle
    # falls below the horizontal — the orientation the field plots use, where
    # the depth axis is inverted and range grows to the right.
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(-1)

    # A limit, not a filter: the whole table stays in the line, so a reader
    # who widens the wedge by hand sees data rather than an empty sector.
    ax.set_thetamin(fan_lo)
    ax.set_thetamax(fan_hi)
    ax.set_thetagrids(_BEAM_PATTERN_TICKS, labels=_BEAM_PATTERN_TICK_LABELS)
    ax.set_rlim(inner, levels.max())
    ax.set_ylabel('Level (dB re peak)', labelpad=22)   # the rectilinear view's label
    # A polar radius is short and every radial label sits on the one spoke, so
    # the ~9 ticks a linear dB axis defaults to overprint one another.
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    # Clear of that thetamin spoke, which runs along the top edge — and through
    # an unpadded title — whenever the wedge opens forward.
    ax.set_title(title or default_title, pad=28.0)
    return fig, ax
