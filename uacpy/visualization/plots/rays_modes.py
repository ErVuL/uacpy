"""Ray fans, arrival stems, mode functions/wavenumbers/heatmaps, reflection coefficients and OASN covariance/replica plots."""

from __future__ import annotations


import numpy as np
from typing import Optional, Tuple

from uacpy.core.environment import Environment
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import Arrivals, Rays, Modes, Covariance, Replicas, ReflectionCoefficient
from uacpy.io.units import m_to_km
from uacpy.visualization.style import SOURCE_MARKER_STYLE
from uacpy.visualization.plots._common import ZORDER_RAYS, ZORDER_SURFACE, ZORDER_SOURCE, _overlay_seafloor, _draw_receiver_grid, _draw_result_credit, fig_ax, typed_plot_error, invert_yaxis_once


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
    _owns_fig = ax is None
    fig, ax = fig_ax(ax, figsize)

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
        color = color_map[kind] if color_by == 'bounces' else color_map['bottom']
        ax.plot(m_to_km(r), z, color=color, alpha=alpha,
                linewidth=linewidth, solid_capstyle='round',
                zorder=ZORDER_RAYS, **mpl_kw)

    invert_yaxis_once(ax)
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
        # Markers are slightly smaller than on the env cross-section —
        # receivers are sampling points here, not the visual focus.
        rr_km = _draw_receiver_grid(ax, rays.receiver_ranges,
                                    rays.receiver_depths, max_markersize=7)
        # x-axis spans the receiver extent with a small right margin so a
        # receiver sitting at the max range isn't clipped to the spine.
        r_max = float(np.max(rr_km))
        ax.set_xlim(0.0, r_max * 1.03 if r_max > 0 else 1.0)
    if show_source and rays.source_depths is not None and rays.source_depths.size:
        src_style = dict(SOURCE_MARKER_STYLE)
        src_style['markersize'] = src_style.get('markersize', 15) + 2
        for sd in rays.source_depths:
            ax.plot([0.0], [float(sd)], zorder=ZORDER_SOURCE, **src_style)

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
    if _owns_fig:
        _draw_result_credit(fig, rays, env=env)
    return fig, ax


def _plot_arrivals(
    arrivals: Arrivals,
    ax=None,
    *,
    figsize: Tuple[float, float] = (10, 4),
    title: Optional[str] = None,
):
    """Stem plot of arrivals: amplitude vs delay, coloured by multipath class.

    Colour palette matches :func:`_plot_rays`: direct = red,
    surface = green, bottom = blue, both = black. Each arrival is drawn
    as a vertical stem plus a head marker."""
    if not isinstance(arrivals, Arrivals):
        raise ConfigurationError(
            f"_plot_arrivals: expected Arrivals, got {type(arrivals).__name__}"
        )
    _owns_fig = ax is None
    fig, ax = fig_ax(ax, figsize)
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
    if _owns_fig:
        _draw_result_credit(fig, arrivals, env=None)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────


def _plot_mode_functions(
    modes: Modes,
    n_modes: Optional[int] = None,
    ax=None,
    *,
    figsize: Tuple[float, float] = (8, 6),
    title: Optional[str] = None,
):
    """Plot the first ``n_modes`` mode shapes ``ψ_m(z)`` as overlaid 1-D curves."""
    if not isinstance(modes, Modes):
        raise ConfigurationError(
            f"_plot_mode_functions: expected Modes, got {type(modes).__name__}"
        )
    _owns_fig = ax is None
    fig, ax = fig_ax(ax, figsize)
    n_modes = modes.n_modes if n_modes is None else min(int(n_modes), modes.n_modes)
    for m in range(n_modes):
        psi = np.asarray(modes.phi[:, m])
        if np.iscomplexobj(psi):
            psi = psi.real
        ax.plot(psi, modes.depths, label=f"m={m+1}", linewidth=1.0)
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
    ax.plot(idx, k.real, 'o-', label=r'$\mathrm{Re}(k_m)$')
    if np.any(np.abs(k.imag) > 0):
        ax2 = ax.twinx()
        ax2.plot(idx, k.imag, 's--', color='C1', label=r'$\mathrm{Im}(k_m)$')
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

    ``mode_range=(start, stop)`` selects a half-open mode-index slice.
    ``normalize=True`` (default) rescales each column to peak ``±1`` so
    high-order modes don't disappear next to the dominant low-order ones.
    """
    if not isinstance(modes, Modes):
        raise ConfigurationError(
            f"plot_modes_heatmap: expected Modes, got {type(modes).__name__}"
        )
    _owns_fig = ax is None
    fig, ax = fig_ax(ax, figsize)
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
    invert_yaxis_once(ax)
    f0 = modes.f0 if modes.f0 is not None else 0.0
    ax.set_title(
        title or f'Mode shapes — {n_plot} modes @ {f0:.1f} Hz',
    )
    if _owns_fig:
        _draw_result_credit(fig, modes, env=None)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Reflection coefficient
# ─────────────────────────────────────────────────────────────────────────────


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
        im = ax.pcolormesh(
            rc.frequencies / 1000.0, rc.theta, rc.R,
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
    """Magnitude of replica response across (z, x) at fixed y=0."""
    if not isinstance(rep, Replicas):
        raise ConfigurationError(
            f"_plot_replicas: expected Replicas, got {type(rep).__name__}"
        )
    _owns_fig = ax is None
    fig, ax = fig_ax(ax, figsize)
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
