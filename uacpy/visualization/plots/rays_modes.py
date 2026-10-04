"""Ray fans, arrival stems, mode functions/wavenumbers/heatmaps, reflection coefficients, source beam patterns and OASN covariance/replica plots."""

from __future__ import annotations


import numpy as np
from typing import Optional, Tuple

from uacpy.core.environment import Environment
from uacpy.core.constants import PRESSURE_FLOOR
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, ValidityWarning,
)
from uacpy.core.acoustics.wavenumber import (alias_period,
                                             ranges_fit_alias_period)
from uacpy.core.results import Arrivals, Rays, Modes, Covariance, Replicas, ReflectionCoefficient, GreensFunction
from uacpy.core.units import m_to_km
from uacpy.visualization.plots._common import ZORDER_LEGEND, ZORDER_RAYS, ZORDER_SURFACE, _overlay_seafloor, _draw_geometry, _draw_receiver_grid, _draw_result_credit, _plot_warn, fig_ax, typed_plot_error, invert_yaxis_once, _title_or, _fit_rotated_axis_label, _checked_dynamic_range_dB


#: Multipath class -> colour, for the ray fan and the arrival stems alike:
#: direct red, surface-reflected green, bottom-reflected blue, both black.
RAY_CLASS_COLOURS = {
    'direct': '#e53935',
    'surface': '#43a047',
    'bottom': '#1e88e5',
    'both': '#000000',
}

#: Right-hand margin on the receiver extent, so a receiver at the maximum
#: range is not clipped to the spine. The seafloor overlay and the x-limit
#: both use it — anchoring them differently is what left a bare strip.
_RECEIVER_EDGE_MARGIN = 1.03

#: Span of the arrival stem plot's dB axis when the caller names none, in dB
#: below the loudest arrival. 60 dB is a 1000:1 amplitude ratio — a path a
#: millionth of the peak power, which moves neither the delay spread nor an
#: equaliser's tap set, while a straggler hundreds of dB down would set the
#: scale for every arrival that does.
_ARRIVALS_DYNAMIC_RANGE_DB = 60.0


@typed_plot_error(who='Rays.plot')
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
        raise ConfigurationError(f"Rays.plot: expected Rays, got {type(rays).__name__}.")
    if color_by not in ('bounces', None):
        # A typo'd mode falling through to the monochrome branch would also
        # drop the per-class legend, so the fan would look like a deliberate
        # color_by=None call.
        raise ConfigurationError(
            f"Rays.plot: color_by={color_by!r} is not a colouring mode; pass "
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
                               fontsize='small', framealpha=0.85)
            # Matplotlib defaults a legend to zorder 5, under this package's
            # seabed fill, seafloor line, receivers and source: the key ends
            # up drawn beneath the picture it explains.
            legend.set_zorder(ZORDER_LEGEND)

    ax.set_xlabel('Range (km)')
    ax.set_ylabel('Depth (m)')
    ax.grid(True, alpha=0.3)
    ax.set_title(_title_or(title, 'Eigenrays' if rays.is_eigen else 'Ray fan'))
    if _owns_fig:
        _draw_result_credit(fig, rays, env=env)
    return fig, ax


@typed_plot_error(who='Arrivals.plot')
def _plot_arrivals(
    arrivals: Arrivals,
    ax=None,
    *,
    receiver: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (10, 4),
    title: Optional[str] = None,
    dB: bool = False,
    dynamic_range_dB: Optional[float] = None,
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
    scale at least still paints a pixel.

    The per-class counts are every arrival the model returned, drawn or
    not, so a qualifier entry reads ``N of M`` and is a **subset** of them:
    with ``bottom (27)`` and ``both (255)``, ``41 of 282 beyond 3583 ms``
    leaves 241 on screen, not 323.

    ``dB=True`` draws ``20·log10`` of that same received level instead —
    dB re unit source, the negative of the transmission loss along the
    path — and bounds the axis at ``dynamic_range_dB`` dB under the loudest
    arrival (:data:`_ARRIVALS_DYNAMIC_RANGE_DB` when unset). The bound is
    what makes the view readable: a level axis has no zero for a stem to
    stand on, and a path hundreds of dB down would otherwise set the scale
    and squeeze every arrival carrying energy onto one pixel. Arrivals
    under the floor are counted in the legend rather than dropped in
    silence, for the reason the ones past the end of the delay axis are.

    ``dynamic_range_dB`` without ``dB=True`` raises: the linear axis is not
    clipped to it, and accepting it would look as though it were.

    One receiver's channel is drawn: ``receiver=(depth_m, range_m)`` picks
    the cell (:meth:`Arrivals.at_receiver`), and a set spanning several
    cells without it is refused, as the channel methods refuse it.

    The default title names the receiver when the set holds one."""
    if not isinstance(arrivals, Arrivals):
        raise ConfigurationError(
            f"Arrivals.plot: expected Arrivals, got {type(arrivals).__name__}."
        )
    # Both checks run before ``fig_ax``, so a rejected call opens no figure.
    if dynamic_range_dB is not None and not dB:
        raise ConfigurationError(
            f"Arrivals.plot: dynamic_range_dB={dynamic_range_dB!r} has "
            "nothing to clip on the linear amplitude axis. Pass dB=True for "
            "the level view it bounds, or drop dynamic_range_dB=."
        )
    range_dB = (_ARRIVALS_DYNAMIC_RANGE_DB if dynamic_range_dB is None
                else _checked_dynamic_range_dB('Arrivals.plot',
                                               dynamic_range_dB))
    if arrivals.arrivals:
        arrivals = arrivals.at_receiver(receiver, who='Arrivals.plot')
    _owns_fig = ax is None
    fig, ax = fig_ax(ax, figsize)
    color_map = RAY_CLASS_COLOURS
    counts = {k: 0 for k in color_map}
    delays_ms = []
    beyond = []
    hi = 0.0
    # Stem heights are what REACHES the receiver. Bellhop keeps volume
    # absorption in the imaginary travel time, not in the amplitude column
    # (``Arrivals.received_amplitudes``), so drawing the column alone stands
    # a heavily absorbed late path at its lossless height — on a 1 km 40 kHz
    # link the second bounce cluster draws five times taller than it arrives.
    levels = (np.abs(arrivals.received_amplitudes) if arrivals.arrivals
              else None)
    # Baseline every stem stands on, and how many the dB floor hides. On the
    # linear axis the baseline is zero and nothing is ever hidden.
    floor = 0.0
    under = 0
    drawn_ms = []
    if dB and levels is not None:
        # Convert to 20·log10 and put the floor ``range_dB`` under the peak.
        # A level is the negative of a transmission loss, so this is the
        # canonical ``transmission_loss_dB`` conversion with its sign flipped,
        # sharing the PRESSURE_FLOOR clamp that holds a silent arrival at
        # -600 dB rather than -inf, which would take the whole axis with it.
        levels = 20.0 * np.log10(np.maximum(levels, PRESSURE_FLOOR))
        floor = float(np.max(levels)) - range_dB
    for index, a in enumerate(arrivals.arrivals):
        kind = a.get('kind', 'direct')
        col = color_map.get(kind, '#1e88e5')
        d_ms = a['delay'] * 1000.0
        # Recorded before the floor test, so the delay axis spans the same
        # arrivals in both views: which stems the level floor hides says
        # nothing about when the energy arrives.
        delays_ms.append(d_ms)
        level = float(levels[index]) if levels is not None else a['amplitude']
        counts[kind] += 1
        if dB and level < floor:
            # Skip it and count it. Clipped to the floor it would read as a
            # level it does not have, on a stem with no length left to read.
            under += 1
            continue
        drawn_ms.append(d_ms)
        ax.vlines(d_ms, floor, level, colors=col, lw=1.5, alpha=0.85)
        ax.plot(d_ms, level, 'o', color=col, markersize=4,
                markeredgecolor='black', markeredgewidth=0.4)
    if delays_ms:
        # The axis follows the ENERGY, not the last ray. A peak-to-peak span
        # is an extremum: one faint straggler stretches it without bound —
        # 8.06 s against a 0.9 s energy span on that same link — and squeezes
        # every arrival that carries something into a few pixels. What falls
        # outside is named below rather than dropped in silence, because an
        # arrival off the end of the axis leaves no trace of itself the way
        # an outlier on a colour scale still does.
        first = min(delays_ms)
        # Every drawn arrival is the one receiver's, so the span is that
        # channel's.
        from uacpy.acoustic_signal.delay_profile import _energy_support
        support_ms = _energy_support(
            arrivals.delays, np.abs(arrivals.received_amplitudes) ** 2,
            who="plot_arrivals") * 1000.0
        span = support_ms if support_ms > 0 else (max(delays_ms) - first)
        if dB and drawn_ms:
            # The energy support collapses when two near-simultaneous
            # arrivals hold all of it: a direct path and its own bottom
            # bounce 0.5 us apart leave a 0.0005 ms axis, and every later
            # arrival falls off the end of a plot that had room for it. The
            # dB view already knows which arrivals cleared the floor, so the
            # axis reaches the last one it actually drew. The linear view
            # keeps the pure energy rule: it has no floor, so it has no such
            # set to span.
            span = max(span, max(drawn_ms) - first)
        lo = first - 0.05 * (span or 1)
        hi = first + span + 0.05 * (span or 1)
        ax.set_xlim(lo, hi)
        # Over the DRAWN stems: in the dB view the axis reaches the last one
        # drawn, so an arrival past the end is one the floor hid, and it is
        # counted with those below rather than a second time here.
        beyond = [d for d in drawn_ms if d > hi]
    ax.set_xlabel('Delay (ms)')
    ax.set_ylabel('Received level (dB re unit source)' if dB
                  else 'Received amplitude (re unit source)')
    if dB and levels is not None:
        # Bottom on the floor exactly, so the dynamic range can be measured
        # off the axis; 5% of it as headroom above the peak, so the loudest
        # head marker is not drawn on the spine.
        ax.set_ylim(floor, floor + range_dB * 1.05)
    ax.grid(True, alpha=0.3)
    # Legend with per-class counts (skip empty classes).
    import matplotlib.lines as mlines
    handles = [
        mlines.Line2D([], [], color=col, marker='o', linestyle='-',
                      label=f"{kind} ({counts[kind]})")
        for kind, col in color_map.items() if counts[kind] > 0
    ]
    counted = sum(counts.values())
    if beyond:
        # "N of M", not "+N": the class counts above are every arrival the
        # model returned, drawn or not, so these are a SUBSET of them. A
        # leading + read as an addition, inviting 282 + 41 where the answer
        # is 282 - 41 = 241 on screen. In ms, the unit of the delay axis the
        # entry refers to.
        handles.append(mlines.Line2D(
            [], [], linestyle='none', marker='',
            label=f"{len(beyond)} of {counted} beyond {hi:.0f} ms "
                  f"(to {max(beyond):.0f} ms)"))
    if under:
        # In dB, the unit of the level axis it refers to, as the delay entry
        # above is in the milliseconds of the delay axis. A hidden arrival
        # past the end of the axis is named here, once.
        hidden_beyond = sum(1 for d in delays_ms if d > hi)
        label = f"{under} of {counted} below {floor:.0f} dB"
        if hidden_beyond:
            label += f", of which {hidden_beyond} beyond the axis"
        handles.append(mlines.Line2D(
            [], [], linestyle='none', marker='', label=label))
    if handles:
        # Placed clear of the stems, as _plot_rays places its own: pinned to
        # a corner, it can cover the head marker of the last stem drawn.
        ax.legend(handles=handles, loc='best', fontsize='small', framealpha=0.85)
    auto = 'Arrivals'
    rd = getattr(arrivals, 'receiver_depths', None)
    rr = getattr(arrivals, 'receiver_ranges', None)
    if (receiver is not None and arrivals.arrivals
            and rd is not None and rr is not None):
        # The cell the arrivals were selected from, read off the result's
        # own axes; the request is named too when it reads differently.
        first = arrivals.arrivals[0]
        depth = float(np.atleast_1d(rd)[int(first.get('depth_idx', 0))])
        rng = float(np.atleast_1d(rr)[int(first.get('range_idx', 0))])
        auto += f" at {depth:g} m depth, {m_to_km(rng):g} km"
        asked_depth, asked_range = (float(v) for v in receiver)
        if (f"{asked_depth:g}", f"{m_to_km(asked_range):g}") != (
                f"{depth:g}", f"{m_to_km(rng):g}"):
            auto += (f" (requested {asked_depth:g} m, "
                     f"{m_to_km(asked_range):g} km)")
    elif (rd is not None and rr is not None
            and np.size(rd) == 1 and np.size(rr) == 1):
        rd, rr = np.atleast_1d(rd), np.atleast_1d(rr)
        auto += f" at {float(rd[0]):g} m depth, {m_to_km(float(rr[0])):g} km"
    ax.set_title(_title_or(title, auto))
    if _owns_fig:
        _draw_result_credit(fig, arrivals, env=None)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Modes
# ─────────────────────────────────────────────────────────────────────────────


@typed_plot_error(who='Modes.plot')
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
            f"Modes.plot: expected Modes, got {type(modes).__name__}."
        )
    if show_imaginary and not np.iscomplexobj(modes.phi):
        raise ConfigurationError(
            "Modes.plot: show_imaginary=True needs complex mode "
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
        ax.legend(fontsize='small', loc='best')
    ax.set_title(_title_or(title, f"Mode functions (n={n_modes})"))
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
    """Scatter ``Re(k_m)`` vs mode index; overlay imaginary part if non-zero.

    Parameters
    ----------
    modes : Modes
        The modes to draw.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(8, 5)`` by default; unused when
        ``ax`` is given.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    """
    if not isinstance(modes, Modes):
        raise ConfigurationError(
            f"plot_mode_wavenumbers: expected Modes, got {type(modes).__name__}."
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
    ax.set_title(_title_or(title, 'Modal wavenumbers'))
    if _owns_fig:
        _draw_result_credit(fig, modes, env=None)
    return fig, ax


@typed_plot_error
def plot_wavenumber_sampling(
    frequency: float,
    c_low: float,
    c_high: float,
    delta_k: float,
    ax=None,
    *,
    rmax_m: Optional[float] = None,
    water_sound_speed: Optional[float] = None,
    c_bottom: Optional[float] = None,
    figsize: Tuple[float, float] = (9, 3.2),
    title: Optional[str] = None,
):
    """Draw the wavenumber axis a Hankel transform is being sampled on.

    The inverse transform of :func:`plot_greens_function`'s ``G(k_r, z)`` back
    to range is a discrete sum, and both of its sampling choices can ruin the
    answer silently:

    * the **window** ``[omega/c_high, omega/c_low]`` decides which physics is
      carried. Closing ``c_high`` below the seabed speed discards trapped
      modes; raising ``c_low`` discards the steep and evanescent components
      that build the near field.
    * the **step** ``Delta k`` sets the alias period ``r_wrap = 2*pi/Delta k``.
      Energy from beyond ``r_wrap`` folds back inside it, and folded energy is
      indistinguishable from real energy once it has landed — it makes the
      field too LOUD, which reads as a physical result rather than an error.

    This draws the window against the wavenumbers that matter — ``omega/c`` in
    the water and in the seabed, whose interval is the trapped band — and, if
    ``rmax_m`` is given, says whether the requested ranges fit inside the alias
    period.

    Parameters
    ----------
    frequency : float
        Hz.
    c_low, c_high : float
        Phase-speed window (m/s). ``k`` runs from ``omega/c_high`` to
        ``omega/c_low``, so ``c_low`` sets the LARGEST wavenumber.
    delta_k : float
        Wavenumber step (rad/m) of the sampled transform.
    rmax_m : float, optional
        Farthest receiver range (m), to test against the alias period.
    water_sound_speed, c_bottom : float, optional
        Water and seabed sound speeds (m/s), marked on the axis; their
        interval is the trapped band.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(9, 3.2)`` by default; unused when
        ``ax`` is given.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    """
    optional = (('rmax_m', rmax_m), ('water_sound_speed', water_sound_speed), ('c_bottom', c_bottom))
    for name, value in (('frequency', frequency), ('c_low', c_low),
                        ('c_high', c_high), ('delta_k', delta_k),
                        *((n, v) for n, v in optional if v is not None)):
        if not np.isfinite(value) or value <= 0:
            raise ConfigurationError(
                f"plot_wavenumber_sampling: {name} must be positive and "
                f"finite, got {value!r}.")
    if c_high <= c_low:
        raise ConfigurationError(
            f"plot_wavenumber_sampling: c_high ({c_high:g}) must exceed "
            f"c_low ({c_low:g}) -- they bracket a phase-speed window.")

    omega = 2.0 * np.pi * float(frequency)
    k_min, k_max = omega / float(c_high), omega / float(c_low)
    # The period and the verdict are acoustics.alias_period /
    # ranges_fit_alias_period, which a reader of this figure can call to
    # get the number the title reports.
    r_wrap = alias_period(delta_k)

    _owns_fig = ax is None
    fig, ax = fig_ax(ax, figsize)
    ax.axvspan(k_min, k_max, color='C0', alpha=0.15,
               label=f'sampled window, $\\Delta k$ = {delta_k:.3g} rad/m')
    # Grown UPWARD from the floor, not down from the top: the legend sits in
    # an upper corner and a speed marked in that half of the window had its
    # label printed underneath the legend box. The text itself is chosen
    # AFTER the layout below, because how much of it fits depends on the axes
    # height, which depends on the font -- see _fit_rotated_axis_label.
    _markers = []
    for c, label, colour in ((c_bottom, 'seabed', 'C3'), (water_sound_speed, 'water', 'C0')):
        if c is not None:
            kc = omega / float(c)
            ax.axvline(kc, color=colour, ls='--', lw=1.2)
            _markers.append((kc, label, colour))
    if water_sound_speed is not None and c_bottom is not None:
        # A seabed SLOWER than the water traps nothing -- there is no angle
        # beyond critical because there is no critical angle. Drawn blind,
        # the span came out reversed and still carried the label, which is a
        # picture of a trapped band over a channel that has none.
        if float(c_bottom) > float(water_sound_speed):
            ax.axvspan(omega / float(water_sound_speed), omega / float(c_bottom),
                       color='C2', alpha=0.12, label='trapped band')
        else:
            ax.axvspan(np.nan, np.nan, color='C2', alpha=0.12,
                       label='no trapped band: seabed is the slower medium')

    ax.set_yticks([])
    ax.set_xlabel('Horizontal wavenumber $k_r$ (rad/m)')
    note = f'alias period $2\\pi/\\Delta k$ = {r_wrap:.0f} m'
    if rmax_m is not None:
        safe = ranges_fit_alias_period(delta_k, rmax_m)
        note += (f'; farthest receiver {float(rmax_m):.0f} m '
                 + ('is inside it — necessary, not sufficient: refine until '
                    'the field stops moving' if safe
                    else 'is BEYOND it — the field folds'))
        ax.set_title(_title_or(title, note),
                     color=('black' if safe else 'C3'), fontsize='small')
    else:
        ax.set_title(_title_or(title, note), fontsize='small')
    # Two full-height spans leave no empty corner for 'best' to find, so the
    # corner is chosen here and the omega/c labels are kept out of it above.
    ax.legend(loc='upper left', fontsize='small', framealpha=0.9)
    if _owns_fig:
        fig.tight_layout()
    for kc, label, colour in _markers:
        _fit_rotated_axis_label(
            ax, kc,
            [f' {label} $\\omega/c$ = {kc:.3f}',
             f' {label} $\\omega/c$',
             f' {label}'],
            colour)
    return fig, ax


@typed_plot_error
def plot_greens_function(
    greens_function: GreensFunction,
    ax=None,
    *,
    frequency_index: int = 0,
    source_index: int = 0,
    depth: Optional[float] = None,
    modes: Optional[Modes] = None,
    dynamic_range_dB: float = 60.0,
    cmap: str = 'viridis',
    figsize: Tuple[float, float] = (9, 6),
    title: Optional[str] = None,
):
    """Image the depth-separated Green's function ``|G(k_r, z)|`` of a ``.grn``.

    ``G(k_r, z)`` is what a wavenumber-integration solver actually computes
    before the Hankel transform to range: the response of the stratified
    column, at one frequency, to a source driven at horizontal wavenumber
    ``k_r`` (JKPS Sect. 4). It is the most instructive object in the method,
    because the sharp ridges in ``k_r`` ARE the normal modes — Kraken finds
    those poles by a root search and sums their residues, while Scooter
    integrates straight through them. Pass ``modes`` to overlay Kraken's
    eigenvalues and see the two methods land on the same poles.

    Feed it the :class:`~uacpy.core.results.GreensFunction`
    :func:`uacpy.io.read_grn_file` returns (``.plot()`` on it draws the
    same). Scooter records the path in ``result.metadata['grn_file']`` when
    ``work_dir`` is pinned, so the file survives the run::

        fld = Scooter(work_dir=tmp).run(env, source, receiver)
        read_grn_file(fld.metadata['grn_file']).plot()

    With ``depth`` the view becomes a cut: ``|G|`` in dB against wavenumber at
    the stored depth nearest the one asked for, which is where the poles are
    easiest to read off individually.

    Parameters
    ----------
    greens_function : GreensFunction
        What :func:`~uacpy.io.read_grn_file` returns.
    frequency_index, source_index : int, optional
        Which frequency and source depth of ``data`` (shape
        ``(nfreq, nsd, nrd, nk)``) to draw. Default the first of each.
    depth : float, optional
        Draw the cut at this receiver depth (m) instead of the 2-D image.
    modes : Modes, optional
        Overlay ``Re(k_m)`` from a Kraken solve of the same environment.
    dynamic_range_dB : float, optional
        How far below the panel maximum the dB scale reaches (default 60).
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    cmap : str, optional
        Colormap of the 2-D panel. Default ``'viridis'``.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(9, 6)`` by default; unused when
        ``ax`` is given.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    """
    floor_dB = -_checked_dynamic_range_dB('plot_greens_function',
                                          dynamic_range_dB)
    if not isinstance(greens_function, GreensFunction):
        raise ConfigurationError(
            f"plot_greens_function: expected a GreensFunction, got "
            f"{type(greens_function).__name__}. read_grn_file returns one; "
            f"Scooter records the file at result.metadata['grn_file'] when "
            f"work_dir is pinned.")

    if greens_function.is_snapshot:
        raise ConfigurationError(
            "plot_greens_function: this is a SPARC snapshot, whose first "
            "axis holds output TIMES, not frequencies. The wavenumber axis "
            "here is omega/c at one frequency per slab, so a time slab "
            "would mislabel every k_r. Use a Scooter .grn for this view.")
    G = greens_function.data
    if not 0 <= frequency_index < G.shape[0]:
        raise ConfigurationError(
            f"plot_greens_function: frequency_index {frequency_index} is "
            f"outside the {G.shape[0]} frequency slab(s) in this file.")
    if not 0 <= source_index < G.shape[1]:
        raise ConfigurationError(
            f"plot_greens_function: source_index {source_index} is outside "
            f"the {G.shape[1]} source depth(s) in this file.")

    freq = float(np.asarray(greens_function.frequencies,
                            dtype=float)[frequency_index])
    with np.errstate(divide='ignore'):
        k_r = greens_function.wavenumbers(freq)
    z = np.asarray(greens_function.receiver_depths, dtype=float)
    panel = np.abs(G[frequency_index, source_index])          # (nrd, nk)

    _owns_fig = ax is None
    fig, ax = fig_ax(ax, figsize)
    peak = float(np.nanmax(panel)) if np.any(np.isfinite(panel)) else 0.0
    ref = peak if peak > PRESSURE_FLOOR else PRESSURE_FLOOR

    if depth is None:
        with np.errstate(divide='ignore'):
            db = 20.0 * np.log10(np.maximum(panel, PRESSURE_FLOOR) / ref)
        im = ax.pcolormesh(k_r, z, db, shading='auto', cmap=cmap,
                           vmin=floor_dB, vmax=0.0)
        fig.colorbar(im, ax=ax, label='$|G|$ (dB re panel max)')
        ax.set_ylabel('Depth (m)')
        invert_yaxis_once(ax)
    else:
        j = int(np.argmin(np.abs(z - float(depth))))
        with np.errstate(divide='ignore'):
            cut = 20.0 * np.log10(
                np.maximum(panel[j], PRESSURE_FLOOR) / ref)
        ax.plot(k_r, cut, lw=1.0, color='C0')
        ax.set_ylabel('$|G|$ (dB re panel max)')
        ax.set_ylim(floor_dB, 5.0)
        ax.set_title(_title_or(
            title, f'Green\'s function at z = {z[j]:.1f} m, {freq:g} Hz'))

    if modes is not None:
        if not isinstance(modes, Modes):
            raise ConfigurationError(
                f"plot_greens_function: modes= expects Modes, got "
                f"{type(modes).__name__}.")
        for i, km in enumerate(np.real(np.asarray(modes.k))):
            ax.axvline(km, color='C3', ls=':', lw=0.9, alpha=0.8,
                       label='Kraken $\\mathrm{Re}\\,k_m$' if i == 0 else None)
        ax.legend(loc='upper right', fontsize='small')

    ax.set_xlabel('Horizontal wavenumber $k_r$ (rad/m)')
    if depth is None:
        ax.set_title(_title_or(
            title, f"Depth-separated Green's function $|G(k_r, z)|$"
                   f" at {freq:g} Hz"))
    if _owns_fig:
        fig.tight_layout()
    return fig, ax


@typed_plot_error
def plot_mode_speeds(
    modes: Modes,
    ax=None,
    *,
    c_bottom: Optional[float] = None,
    figsize: Tuple[float, float] = (8, 5),
    title: Optional[str] = None,
):
    """Phase speed of every mode against mode index, with group speed if known.

    ``plot_mode_wavenumbers`` draws the raw eigenvalue ``Re(k_m)``; this draws
    what the eigenvalue means. The phase speed ``omega / Re(k_m)`` rises with
    mode number, and where it crosses the seabed sound speed the mode stops
    being trapped and starts radiating into the bottom (JKPS Sect. 2.4.5.1) —
    pass ``c_bottom`` to mark that boundary and the trapped count reads off
    the figure.

    The group speed is overlaid when the result carries one. Only
    ``backend='krakenc'`` reports it from a single run
    (:attr:`~uacpy.core.results.Modes.group_velocity`); on the real backend
    it is ``None`` and only the phase speed is drawn. Gaps are left where the
    print file strided over a mode rather than interpolated across them.

    Parameters
    ----------
    modes : Modes
        The mode set to draw.
    c_bottom : float, optional
        Seabed sound speed (m/s). Drawn as the trapped/leaky boundary.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(8, 5)`` by default; unused when
        ``ax`` is given.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    """
    if not isinstance(modes, Modes):
        raise ConfigurationError(
            f"plot_mode_speeds: expected Modes, got {type(modes).__name__}.")
    _owns_fig = ax is None
    fig, ax = fig_ax(ax, figsize)
    idx = np.arange(1, modes.n_modes + 1)
    cp = modes.phase_speeds
    ax.plot(idx, cp, 'o-', ms=4, color='C0', label='phase speed $\\omega/\\mathrm{Re}\\,k_m$')

    vg = getattr(modes, 'group_velocity', None)
    if vg is not None and np.any(np.isfinite(vg)):
        ax.plot(idx, vg, 's--', ms=4, color='C1',
                label='group speed $d\\omega/dk_m$')

    if c_bottom is not None:
        cb = float(c_bottom)
        ax.axhline(cb, color='C3', ls='--', lw=1.0)
        n_trapped = int(np.sum(cp <= cb))
        # The label sits in the band just above the rule, and that is the
        # one band the phase-speed curve is guaranteed to enter: cp rises
        # with mode index and leaves the rule behind at mode ``n_trapped``,
        # so curve and label want the same pixels wherever the crossing is.
        # Put the label at whichever end of the axis is FARTHER from the
        # crossing. A fixed end cannot work: the right edge collides on a
        # mostly-trapped channel (crossing at the right) and the left edge
        # on a mostly-leaky one. The mode index is the abscissa and the
        # modes are evenly spaced along it, so the trapped fraction is
        # where the crossing falls across the panel.
        crossing = n_trapped / max(modes.n_modes, 1)
        at_left = crossing > 0.5
        ax.text(0.01 if at_left else 0.99, cb,
                f' seabed $c_p$ = {cb:g} m/s — {n_trapped} trapped',
                color='C3', fontsize='small', va='bottom',
                ha='left' if at_left else 'right',
                transform=ax.get_yaxis_transform())

    ax.set_xlabel('Mode index $m$')
    ax.set_ylabel('Speed (m/s)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small')
    ax.set_title(_title_or(title, f'Modal speeds at {modes.f0:g} Hz')
                 if modes.f0 else (_title_or(title, 'Modal speeds')))
    if _owns_fig:
        _draw_result_credit(fig, modes, env=None)
    return fig, ax


@typed_plot_error
def plot_dispersion(
    modes_by_frequency,
    ax=None,
    *,
    n_modes: int = 3,
    figsize: Tuple[float, float] = (8, 5),
    title: Optional[str] = None,
):
    """Phase and group speed against frequency — the dispersion diagram.

    Takes a sequence of :class:`~uacpy.core.results.Modes`, one per frequency,
    and draws mode ``m``'s phase speed (solid) and group speed (dashed) as
    curves in frequency. This is the picture of geometric dispersion: both
    speeds start at the seabed speed at that mode's cutoff and tend to the
    water speed at high frequency, the phase speed falling monotonically
    while the group speed passes through a minimum — the frequency of that
    minimum arrives last in a transient, the Airy phase (JKPS Sect. 2.4.5.2).

    The group speed is taken from each result's
    :attr:`~uacpy.core.results.Modes.group_velocity` when the solver supplied
    one (``backend='krakenc'``), and otherwise differenced between
    neighbouring frequencies with
    :meth:`~uacpy.core.results.Modes.group_velocity_between`, which is what
    the real backend needs. Both carry the caveat the KRAKEN source states at
    ``kraken.f90:772``: group speeds are wrong for leaky modes.

    Parameters
    ----------
    modes_by_frequency : sequence of Modes
        At least two mode sets at distinct frequencies. Sorted here, so the
        caller need not.
    n_modes : int, optional
        How many low-order modes to draw (default 3).
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(8, 5)`` by default; unused when
        ``ax`` is given.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    """
    sets = list(modes_by_frequency)
    if len(sets) < 2:
        raise ConfigurationError(
            "plot_dispersion: needs Modes at two or more frequencies — "
            f"got {len(sets)}. Dispersion is a statement about how the modal "
            "speeds vary with frequency, so one solve cannot show it.")
    for m in sets:
        if not isinstance(m, Modes):
            raise ConfigurationError(
                f"plot_dispersion: expected a sequence of Modes, found "
                f"{type(m).__name__}.")
        if m.f0 is None:
            raise ConfigurationError(
                "plot_dispersion: every Modes needs a frequency (f0); one "
                "carries none, so it cannot be placed on the frequency axis.")
    sets.sort(key=lambda m: float(m.f0))
    freqs = np.array([float(m.f0) for m in sets])
    if np.any(np.diff(freqs) <= 0):
        raise ConfigurationError(
            f"plot_dispersion: frequencies must be distinct, got {freqs}.")

    _owns_fig = ax is None
    fig, ax = fig_ax(ax, figsize)
    for mode_i in range(int(n_modes)):
        vp, ff = [], []
        vg, ff_g = [], []
        for j, m in enumerate(sets):
            if m.n_modes <= mode_i:
                continue                      # below this mode's cutoff
            ff.append(freqs[j])
            vp.append(float(m.phase_speeds[mode_i]))
            reported = getattr(m, 'group_velocity', None)
            if reported is not None and np.isfinite(reported[mode_i]):
                # The solver's own value belongs AT this frequency.
                vg.append(float(reported[mode_i]))
                ff_g.append(freqs[j])
            elif j + 1 < len(sets) and sets[j + 1].n_modes > mode_i:
                # A finite difference estimates d(omega)/dk at the MIDPOINT of
                # the pair, which is what group_velocity_between's own
                # docstring says. Plotting it at the left endpoint shifted the
                # whole curve half a step -- and on the last pair it repeated
                # one value at two frequencies, on the very plot whose purpose
                # is reading off the Airy-phase frequency.
                fd = m.group_velocity_between(sets[j + 1])
                if mode_i < len(fd):
                    vg.append(float(fd[mode_i]))
                    ff_g.append(0.5 * (freqs[j] + freqs[j + 1]))
        if not ff:
            continue
        colour = f'C{mode_i}'
        ax.plot(ff, vp, '-', color=colour, lw=1.4, label=f'mode {mode_i + 1}')
        if ff_g:
            ax.plot(ff_g, vg, '--', color=colour, lw=1.2)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Speed (m/s)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize='small', title='solid: phase   dashed: group',
              title_fontsize='x-small')
    ax.set_title(_title_or(title, 'Modal dispersion'))
    if _owns_fig:
        _draw_result_credit(fig, sets[0], env=None)
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

    Parameters
    ----------
    modes : Modes
        The modes to draw.
    n_modes : int, optional
        Draw the first ``n_modes``; exclusive with ``mode_range``.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(8, 6)`` by default; unused when
        ``ax`` is given.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    mode_range : (int, int), optional
        Half-open mode-index slice (see above).
    normalize : bool, optional
        Scale each mode to peak ±1. Default True.
    cmap : str, optional
        Colormap. Default ``'RdBu_r'``.
    """
    if not isinstance(modes, Modes):
        raise ConfigurationError(
            f"plot_modes_heatmap: expected Modes, got {type(modes).__name__}."
        )
    if mode_range is not None:
        if n_modes is not None:
            # mode_range takes the slice wholesale, so a call passing both
            # would plot the range and drop n_modes without saying so.
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
    ax.set_title(_title_or(title, auto))
    if _owns_fig:
        _draw_result_credit(fig, modes, env=None)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Reflection coefficient
# ─────────────────────────────────────────────────────────────────────────────


def _coefficient_symbol(rc: ReflectionCoefficient) -> Tuple[str, str]:
    """``(symbol letter, quantity name)`` for what this result actually holds.

    OASR returns a transmission coefficient under ``reflection_type=
    'transmission'`` (``models/oases/oasr.py``
    ``_resolve_reflection_type``), and
    that column is an amplitude ratio across the interface, not a reflection
    coefficient — it is not bounded by 1 and it is not the same quantity. A
    result carrying no ``reflection_type`` is a reflection coefficient:
    Bounce writes only BRC/TRC tables, and OASR's own default is 'P-P'.
    """
    if rc.reflection_type == 'transmission':
        return 'T', 'Transmission coefficient'
    return 'R', 'Reflection coefficient'


#: The knobs each branch of ``_plot_reflection_coefficient`` cannot use.
_REFLECTION_MAP_ONLY = ('angle_on_x', 'frequency_unit', 'cmap', 'vmin', 'vmax',
                        'show_colorbar')
_REFLECTION_BRANCH_UNUSED = {
    'broadband': ('show_phase',),
    'narrowband': _REFLECTION_MAP_ONLY,
}

#: Where the decibel-loss view stops. The loss drawn is ``rc.dB``, whose null
#: reflection is the 600 dB no-energy marker; a view scaled to that marker
#: flattens every real loss against its floor, so a loss past this level is
#: drawn at its value and cut by the axis (line) or saturates the colour scale
#: (map).
_LOSS_VIEW_MAX_DB = 120.0


def _loss_name(rc: ReflectionCoefficient) -> str:
    """What ``-20 log10 |R|`` is called for what this result holds.

    'Reflection loss', not 'bottom loss': Bounce writes a top-reflection
    table as readily as a bottom one, and the plotter cannot tell which
    interface the caller bounced off.
    """
    if rc.reflection_type == 'transmission':
        return 'Transmission loss'
    return 'Reflection loss'


@typed_plot_error(who='ReflectionCoefficient.plot')
def _plot_reflection_coefficient(
    rc: ReflectionCoefficient,
    ax=None,
    *,
    figsize: Tuple[float, float] = (8, 5),
    title: Optional[str] = None,
    quantity: str = 'magnitude',
    show_phase: Optional[bool] = None,
    angle_on_x: Optional[bool] = None,
    frequency_unit: Optional[str] = None,
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    show_colorbar: Optional[bool] = None,
):
    """Auto-detect narrowband (line) vs broadband (heatmap) reflection coefficient.

    ``quantity='loss'`` draws ``-20 log10 |R|`` in decibels instead of the
    linear magnitude — the form a reflection is read in against grazing
    angle, where the critical angle is the knee and the plateau is the loss
    per bounce. ``quantity='magnitude'`` (the default) keeps ``|R|``.

    ``show_phase=True`` overlays the phase ``φ(θ)`` on a twin y-axis
    when the input is narrowband (single frequency).

    For the broadband map, ``angle_on_x=True`` puts grazing angle on the
    abscissa so the map can sit beside a narrowband ``|R|(θ)`` panel on a
    shared axis, and ``frequency_unit='Hz'`` labels the other axis in hertz --
    a band of tens to thousands of hertz reads as 0.02-2 on a kHz axis.
    ``show_colorbar=False`` leaves the bar to the caller. ``cmap`` (default
    ``'viridis'``), ``vmin`` and ``vmax`` style the map.

    Each of these knobs belongs to one branch, and a knob the drawn branch
    cannot use is refused rather than ignored: ``show_phase`` on a broadband
    result, or any map knob on a narrowband one."""
    if not isinstance(rc, ReflectionCoefficient):
        raise ConfigurationError(
            f"ReflectionCoefficient.plot: expected ReflectionCoefficient, "
            f"got {type(rc).__name__}."
        )
    if quantity not in ('magnitude', 'loss'):
        raise ConfigurationError(
            f"ReflectionCoefficient.plot: quantity must be 'magnitude' or "
            f"'loss', got {quantity!r}.")
    supplied = {'show_phase': show_phase, 'angle_on_x': angle_on_x,
                'frequency_unit': frequency_unit, 'cmap': cmap, 'vmin': vmin,
                'vmax': vmax, 'show_colorbar': show_colorbar}
    branch = 'broadband' if rc.is_broadband else 'narrowband'
    unused = [k for k in _REFLECTION_BRANCH_UNUSED[branch]
              if supplied[k] is not None]
    if unused:
        message = (
            f"ReflectionCoefficient.plot: {', '.join(f'{k}=' for k in unused)} "
            f"has no effect on a {branch} result. show_phase= applies to the "
            f"single-frequency |R|(θ) line; angle_on_x=, frequency_unit=, "
            f"cmap=, vmin=, vmax= and show_colorbar= to the broadband "
            f"|R|(θ, f) map.")
        if branch == 'broadband':
            raise ConfigurationError(
                message,
                remediation="Pick one frequency first, rc.at(frequency=f), "
                            "for the phase line.")
        raise ConfigurationError(message)
    if rc.is_broadband:
        frequency_unit = 'kHz' if frequency_unit is None else frequency_unit
        cmap = 'viridis' if cmap is None else cmap
        _owns_fig = ax is None
        fig, ax = fig_ax(ax, figsize)
        if frequency_unit not in ('kHz', 'Hz'):
            raise ConfigurationError(
                f"ReflectionCoefficient.plot: frequency_unit must be 'kHz' "
                f"or 'Hz', got {frequency_unit!r}.")
        freqs = np.asarray(rc.frequencies, dtype=float)
        scale = 1000.0 if frequency_unit == 'kHz' else 1.0
        f_axis, f_label = freqs / scale, f'Frequency ({frequency_unit})'
        # rc.magnitude is (angle, frequency); pcolormesh wants C indexed (y, x), so the
        # orientation is read off the result's documented layout rather than
        # inferred from which axis length happens to match.
        values = rc.magnitude if quantity == 'magnitude' else rc.dB
        saturates = (quantity == 'loss' and vmax is None
                     and (vmin is None or vmin < _LOSS_VIEW_MAX_DB)
                     and np.nanmax(values) > _LOSS_VIEW_MAX_DB)
        if saturates:
            vmax = _LOSS_VIEW_MAX_DB
        if angle_on_x:
            x, y, C = rc.angles, f_axis, values.T
            xlabel, ylabel = 'Grazing angle (°)', f_label
        else:
            x, y, C = f_axis, rc.angles, values
            xlabel, ylabel = f_label, 'Grazing angle (°)'
        im = ax.pcolormesh(x, y, C, shading='nearest', cmap=cmap,
                           vmin=vmin, vmax=vmax)
        letter, quantity_name = _coefficient_symbol(rc)
        bar_label = (f'|{letter}|' if quantity == 'magnitude'
                     else f'{_loss_name(rc)} (dB)')
        if show_colorbar is None or show_colorbar:
            fig.colorbar(im, ax=ax, label=bar_label,
                         extend='max' if saturates else 'neither')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        auto = (f'{quantity_name} |{letter}(θ, f)|' if quantity == 'magnitude'
                else f'{_loss_name(rc)} (θ, f)')
        ax.set_title(_title_or(title, auto))
        if _owns_fig:
            _draw_result_credit(fig, rc, env=None)
        return fig, ax

    _owns_fig = ax is None
    fig, ax = fig_ax(ax, figsize)
    letter, quantity_name = _coefficient_symbol(rc)
    if quantity == 'magnitude':
        values, ylabel, auto = rc.magnitude, f'|{letter}|', quantity_name
    else:
        values = rc.dB
        ylabel = auto = f'{_loss_name(rc)}'
        ylabel = f'{ylabel} (dB)'
    ax.plot(rc.angles, values, label=ylabel, color='C0')
    if quantity == 'loss' and np.nanmax(values) > _LOSS_VIEW_MAX_DB:
        # The view autoscaling would give a line whose top is the cut level.
        low = min(np.nanmin(values), _LOSS_VIEW_MAX_DB)
        pad = ax.margins()[1] * (_LOSS_VIEW_MAX_DB - low)
        ax.set_ylim(low - pad, _LOSS_VIEW_MAX_DB + pad)
    ax.set_xlabel('Grazing angle (°)')
    ax.set_ylabel(ylabel, color='C0')
    ax.tick_params(axis='y', labelcolor='C0')
    ax.grid(True, alpha=0.3)
    if show_phase:
        ax_phi = ax.twinx()
        ax_phi.plot(rc.angles, np.rad2deg(rc.phase), '--', color='C1',
                    label='φ')
        ax_phi.set_ylabel('Phase (°)', color='C1')
        ax_phi.tick_params(axis='y', labelcolor='C1')
    ax.set_title(_title_or(title, auto))
    if _owns_fig:
        _draw_result_credit(fig, rc, env=None)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# Covariance / Replicas
# ─────────────────────────────────────────────────────────────────────────────


#: Receiver-position columns, and the axis each one is drawn as.
_POSITION_AXES = (('x', 'Receiver x (m)'), ('y', 'Receiver y (m)'),
                  ('z', 'Receiver depth (m)'))


def _array_coordinate(positions):
    """``(coordinate, label, is_depth)`` of the one position column along
    which a line array runs, or ``None`` when there is no such column (no
    positions, co-located receivers, or an array spread over two or more
    columns, which has no single coordinate to draw against)."""
    if positions is None:
        return None
    p = np.asarray(positions, dtype=float)
    varying = [i for i in range(p.shape[1]) if np.ptp(p[:, i]) > 0.0]
    if len(varying) != 1:
        return None
    column = p[:, varying[0]]
    if np.any(np.diff(column) == 0.0) or not (
            np.all(np.diff(column) > 0) or np.all(np.diff(column) < 0)):
        return None
    name, label = _POSITION_AXES[varying[0]]
    return column, label, name == 'z'


@typed_plot_error(who='Covariance.plot')
def _plot_covariance(
    cov: Covariance,
    ax=None,
    *,
    frequency_index: int = 0,
    figsize: Tuple[float, float] = (6, 5),
    title: Optional[str] = None,
):
    """Heatmap of one covariance slice ``|C[frequency_index, :, :]|``.

    A line array whose ``receiver_positions`` run along one coordinate (a
    vertical array in depth, a horizontal one in x or y) is drawn against
    that coordinate, depth increasing downward; any other array is drawn
    against the receiver index."""
    if not isinstance(cov, Covariance):
        raise ConfigurationError(
            f"Covariance.plot: expected Covariance, got {type(cov).__name__}."
        )
    _owns_fig = ax is None
    fig, ax = fig_ax(ax, figsize)
    C = np.abs(cov.covariance[frequency_index])
    along = _array_coordinate(getattr(cov, 'receiver_positions', None))
    if along is None:
        im = ax.imshow(C, cmap='viridis', aspect='auto', origin='upper')
        ax.set_xlabel('Receiver j')
        ax.set_ylabel('Receiver i')
    else:
        coordinate, label, is_depth = along
        im = ax.pcolormesh(coordinate, coordinate, C, cmap='viridis',
                           shading='nearest')
        ax.set_xlabel(f'{label}, j')
        ax.set_ylabel(f'{label}, i')
        if is_depth:
            invert_yaxis_once(ax)
    unit = cov.unit
    fig.colorbar(im, ax=ax, label=f'|C| ({unit})' if unit else '|C|')
    f_hz = (float(cov.frequencies[frequency_index])
            if cov.frequencies is not None else None)
    if title is None and f_hz is not None:
        title = f"Covariance at {f_hz:.1f} Hz"
    if title:
        ax.set_title(title)
    if _owns_fig:
        _draw_result_credit(fig, cov, env=None)
    return fig, ax


@typed_plot_error(who='Replicas.plot')
def _plot_replicas(
    rep: Replicas,
    ax=None,
    *,
    frequency_index: int = 0,
    sensor_index: int = 0,
    y_index: int = 0,
    figsize: Tuple[float, float] = (8, 5),
    title: Optional[str] = None,
):
    """Magnitude of one element's replica response over the candidate
    (depth, range) grid, at one frequency and one ``y`` node.

    Candidate range is drawn in km and depth in m, the axes
    :func:`~uacpy.plot.plot_matched_field` draws the ambiguity
    surface on, so a replica panel and the surface built from it line up.
    ``y_index`` picks the plane of an OASN ``(depth, x, y)`` candidate grid
    (the first by default) and the default title names it, with the
    frequency and the element; a ``(depth, range)`` bank has no ``y``."""
    if not isinstance(rep, Replicas):
        raise ConfigurationError(
            f"Replicas.plot: expected Replicas, got {type(rep).__name__}."
        )
    names = list(rep.candidates)
    horizontal = [name for name in names if name in ('range', 'x')]
    if 'depth' not in names or len(horizontal) != 1 or not set(names) <= {
            'depth', horizontal[0], 'y'}:
        raise ConfigurationError(
            f"Replicas.plot: draws a candidate (depth, range) or (depth, x) "
            f"grid, with an optional y; these candidates are {names}.")
    n_y = rep.candidates['y'].size if 'y' in names else 1
    if not -n_y <= y_index < n_y:
        raise ConfigurationError(
            f"Replicas.plot: y_index={y_index} is outside the {n_y} candidate "
            f"y node(s).")
    _owns_fig = ax is None
    fig, ax = fig_ax(ax, figsize)
    # replicas is (n_freq, *candidates, n_rcv): take one frequency and one
    # array element, and cut the candidate grid at one y node.
    cut = rep.replicas[frequency_index, ..., sensor_index]
    if 'y' in names:
        cut = np.take(cut, y_index, axis=names.index('y'))
        names = [name for name in names if name != 'y']
    R = np.abs(np.transpose(cut, (names.index('depth'),
                                  names.index(horizontal[0]))))
    im = ax.pcolormesh(
        m_to_km(rep.candidates[horizontal[0]]), rep.candidates['depth'], R,
        shading='nearest', cmap='magma',
    )
    fig.colorbar(im, ax=ax, label='|R|')
    ax.set_xlabel('Candidate range (km)')
    ax.set_ylabel('Candidate depth (m)')
    invert_yaxis_once(ax)
    freqs = getattr(rep, 'frequencies', None)
    freqs = np.atleast_1d(freqs if freqs is not None else [])
    at_f = (f", {float(freqs[frequency_index]):g} Hz"
            if freqs.size > abs(frequency_index) else "")
    at_y = (f", y = {float(rep.candidates['y'][y_index]):g} m"
            if 'y' in rep.candidates else "")
    ax.set_title(_title_or(
        title, f"Replica |R|, element {sensor_index}{at_f}{at_y}"))
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
                        "np.column_stack([angles_deg, levels_dB]).",
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
def plot_mode_excitation(
    modes,
    source,
    ax=None,
    *,
    sound_speed: Optional[float] = None,
    env=None,
    show_array_factor: bool = True,
    dynamic_range_dB: float = 40.0,
    figsize: Tuple[float, float] = (8.0, 4.5),
    title: Optional[str] = None,
):
    """What a source array drives, both ways, on one angle axis.

    A stem per trapped mode at that mode's grazing angle
    ``θₘ = arccos(c / vₚ,ₘ)``, whose height is the modal excitation
    ``|Σₙ wₙ·φₘ(zₙ)|`` (:meth:`~uacpy.core.results.Modes.excitation`) — what
    the array actually does in the waveguide, the *mode filter* of Medwin &
    Clay §11.3.1. Over it, optionally, the free-field pattern of the same
    array: its **array beam pattern** ``P(θ) = f(θ)·A(θ)`` when the elements
    are directional, or the bare array factor ``A(θ)`` when they are not
    (Butler & Sherman §7.1.1; :meth:`~uacpy.Source.array_beam_pattern`).
    Both the stems and the curve carry the element pattern the engine
    applies, so the comparison is one array described two ways.

    Both are normalised to their own maximum, so the figure compares
    *shapes*, which is the comparison that means something: the two agree
    while the pattern is symmetric in ±θ and part company once steering
    breaks that symmetry, because a trapped mode is a standing wave with
    equal up- and down-going halves and no single angle to be steered onto.
    A reader who takes the free-field curve for the channel's response is
    the reason this plot draws them together rather than apart.

    Parameters
    ----------
    modes : Modes
        From ``Kraken(...).compute_modes(env, source)``, tabulated over a
        depth range spanning the array.
    source : Source
        The array: its ``depths`` and complex ``weights``.
    sound_speed : float, optional
        Reference speed (m/s) the grazing angles are measured against.
        ``None`` (default) reads it from ``env`` at the source depth — the
        depth at which the array factor's ``θ`` is a launch angle. With
        neither, the call is **refused**: the x axis is an angle measured
        against this speed, so defaulting it silently moved every stem (up to
        5.3° on a 1545 m/s environment). A mode whose phase
        speed is below it has no real grazing angle at that depth and is
        dropped, with a warning naming how many: clipping them onto 0°
        would pile evanescent modes on the axis origin as if they travelled
        horizontally.
    env : Environment, optional
        Supplies ``sound_speed`` at the source depth when it is not given.
    show_array_factor : bool
        Draw the free-field curve. ``False`` leaves the mode filter alone.
    dynamic_range_dB : float
        How far below 0 dB the axis reaches, both curves being dB re their
        own max (default 40).
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(8.0, 4.5)`` by default; unused when
        ``ax`` is given.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    """
    floor_dB = -_checked_dynamic_range_dB('plot_mode_excitation',
                                          dynamic_range_dB)
    if sound_speed is None:
        if env is None:
            raise ConfigurationError(
                "plot_mode_excitation: the x axis is a grazing angle, "
                "arccos(c/v_m), so it needs the speed those angles are "
                "measured against. Pass sound_speed= (the speed at the "
                "source depth), or env= to read it off the profile.",
                remediation="sound_speed=env.ssp.sound_speed_at("
                            "source.depths[0])")
        # env.ssp.sound_speed_at is the accessor for a speed at a depth, and it
        # is called unguarded: a failure here must surface, because a
        # substituted reference speed moves every stem (5.3 degrees on a
        # 1545 m/s environment, one mode from 8.0 to 15.9).
        z_s = float(np.atleast_1d(source.depths)[0])
        sound_speed = float(np.atleast_1d(env.ssp.sound_speed_at(z_s))[0])
    amp = np.abs(modes.excitation(source, sound_speed=sound_speed))
    # One home for arccos(c/v_m): Modes owns it, and owns the knowledge that
    # a mode below the reference speed is evanescent and has no real angle.
    angles = modes.grazing_angles(sound_speed)
    propagating = np.isfinite(angles)
    dropped = int(np.sum(~propagating))
    # Silent when the source carries a pattern: Modes.excitation has already
    # warned about these same modes against this same reference speed, and
    # said they come back as NaN. Two warnings for one condition teach the
    # reader to skip both, and "returned as NaN" already implies "not drawn".
    if dropped and getattr(source, 'beam_pattern', None) is None:
        _plot_warn(
            f"plot_mode_excitation: {dropped} of {angles.size} modes have a "
            f"phase speed below the {float(sound_speed):g} m/s reference, so "
            f"they have no real grazing angle there and are not drawn. Pass "
            f"sound_speed= (or env=) for the speed at the source depth.", ValidityWarning)
    good = (propagating & np.isfinite(angles) & np.isfinite(amp)
            & (amp > 0.0))
    if not np.any(good):
        raise ConfigurationError(
            "plot_mode_excitation: no mode has both a finite grazing angle "
            "and a non-zero excitation — check that the source depths sit "
            "inside the tabulated mode depths.")
    angles, amp = angles[good], amp[good]
    level = 20.0 * np.log10(amp / amp.max())

    _owns_fig = ax is None
    fig, ax = fig_ax(ax, figsize)
    ax.stem(angles, np.maximum(level, floor_dB), bottom=floor_dB,
            basefmt=' ', linefmt='C0-', markerfmt='C0o',
            label='mode excitation (waveguide)')
    if show_array_factor:
        # Folded over +/-theta, and normalised over the whole fan rather than
        # the drawn half. A trapped mode is a standing wave — it responds to
        # e^{+ik_z z} AND e^{-ik_z z} — so the free-field quantity comparable
        # to a mode stem is max(|AF(+θ)|, |AF(−θ)|). Drawing |AF(+θ)| over
        # [0, 90] alone put an upward-steered array's main lobe off the plot
        # and normalised a 12 dB sidelobe to 0 dB.
        span = np.linspace(0.0, max(90.0, float(angles.max())), 721)
        # P = f*A when the elements are directional, A alone when they are
        # not (Butler & Sherman §7.1.1). Drawing the bare factor for a shaded
        # array would show the geometry only and make a directional source
        # look omnidirectional, while the stems beside it carry f.
        shaped = getattr(source, 'beam_pattern', None) is not None
        pattern = (source.array_beam_pattern if shaped
                   else source.array_factor)
        af = np.maximum(
            np.abs(pattern(span, sound_speed=sound_speed)),
            np.abs(pattern(-span, sound_speed=sound_speed)))
        if af.max() > 0.0:
            ax.plot(span, np.maximum(20.0 * np.log10(af / af.max()),
                                      floor_dB),
                    'C3-', lw=1.2, alpha=0.85,
                    label=('array beam pattern P=f·A (free field, ±θ)'
                           if shaped else
                           'array factor (free field, folded ±θ)'))
    ax.set_xlabel('Mode grazing angle (°)')
    ax.set_ylabel('Normalised amplitude (dB re max)')
    ax.set_ylim(floor_dB, 3.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize='small')
    n_src = len(np.atleast_1d(source.depths))
    aperture = float(np.ptp(np.atleast_1d(source.depths)))
    ax.set_title(_title_or(
        title, f"Array response — {n_src} source(s) over {aperture:g} m"))
    if _owns_fig:
        _draw_result_credit(fig, modes)
    return fig, ax


@typed_plot_error
def plot_beam_power(beams, ax=None, *, at=None, normalize: bool = True,
                    title: Optional[str] = None,
                    figsize: Tuple[float, float] = (8, 5), **mpl_kw):
    """Beam power against look angle, from a :class:`BeamformedField`.

    The RECEIVE dual of :func:`plot_beam_pattern`, and deliberately not that
    function: a source beam pattern is a launch fan, so it labels its axis
    'Launch angle' and warns when a table does not span the +/-90 deg
    Bellhop can steer into. A scanned beam is neither — it is what the array
    heard, over whatever sector was scanned.

    Parameters
    ----------
    beams : BeamformedField
        What :func:`uacpy.acoustic_signal.beamform_field` returned.
    at : int, tuple or dict, optional
        Which point of the grid to draw, when the beamformer ran over one.
        A ``(n_elements, n_ranges)`` field gives ``power`` of shape
        ``(n_angles, n_ranges)``, and a single curve needs one range; a
        broadband run needs its frequency bin the same way. An index (int,
        or a tuple over the axes after the angle one), or a dict of
        coordinates, ``{'range': 4000.0}``, read at the nearest sample of
        the ``grid_coords`` the result carries (``'frequency'`` for the
        frequency axis of a band). The title names the point drawn. Not
        needed when the beamformer ran on a single point.
    normalize : bool, default True
        Draw dB re the peak of the curve, which is how a beam is read. False
        keeps the absolute ``10*log10(power)``.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(8, 5)`` by default; unused when
        ``ax`` is given.

    Returns
    -------
    (fig, ax)
    """
    power = np.asarray(getattr(beams, 'power', None))
    angles = np.asarray(getattr(beams, 'angles', None), dtype=float)
    if power.ndim == 0 or angles.ndim != 1:
        raise ConfigurationError(
            "plot_beam_power: expected a BeamformedField from "
            "beamform_field; got something without power/angles.")
    axes = _beam_grid_axes(beams, power.ndim - 1)
    where = ''
    if at is not None:
        index = _beam_grid_index(at, axes)
        power = power[(slice(None),) + index]
        where = ', '.join(
            f"{name}={values[i]:g}" for (name, values), i in zip(axes, index)
            if values is not None)
    if power.ndim != 1:
        raise ConfigurationError(
            f"plot_beam_power: the beamformer ran over a grid, so its power "
            f"is {power.shape} and there is no single curve to draw. Pass "
            f"at= to choose the point — an index into the axes after the "
            f"angle one (a range index, or a frequency bin for a broadband "
            f"run).")
    with np.errstate(divide='ignore'):
        level = 10.0 * np.log10(power)
        if normalize:
            level = level - np.max(level)
    fig, ax = fig_ax(ax, figsize)
    ax.plot(angles, level, **mpl_kw)
    ax.set_xlabel('Look angle (°)')
    ax.set_ylabel('Beam power (dB re max)' if normalize
                  else 'Beam power (dB)')
    ax.grid(True, alpha=0.3)
    ax.set_title(_title_or(title, f'Beam power at {where}' if where
                           else 'Beam power'))
    return fig, ax


def _beam_grid_axes(beams, n_axes):
    """``[(name, coordinates or None), ...]`` for the axes of a beam result
    after its angle axis: its ``grid_coords`` in order, then ``'frequency'``
    for a band."""
    grid = dict(getattr(beams, 'grid_coords', None) or {})
    named = [(name, np.asarray(values, dtype=float))
             for name, values in grid.items()]
    frequencies = getattr(beams, 'frequencies', None)
    if frequencies is not None:
        named.append(('frequency', np.asarray(frequencies, dtype=float)))
    if len(named) != n_axes:
        named = [(f'axis {i + 1}', None) for i in range(n_axes)]
    return named


def _beam_grid_index(at, axes):
    """The index tuple ``at`` selects over ``axes`` (see plot_beam_power)."""
    if isinstance(at, dict):
        names = [name for name, _ in axes]
        unknown = [k for k in at if k not in names
                   or dict(axes)[k] is None]
        if unknown:
            raise ConfigurationError(
                f"plot_beam_power: at= names {unknown}, but the grid's named "
                f"axes are {[n for n, v in axes if v is not None]}.",
                remediation="Pass beamform_field(..., grid_coords={...}) to "
                            "name the grid, or give at= as an index.")
        index = []
        for name, values in axes:
            if name in at:
                index.append(int(np.argmin(np.abs(values - float(at[name])))))
            elif values is not None and values.size == 1:
                index.append(0)
            else:
                raise ConfigurationError(
                    f"plot_beam_power: at= does not name the {name!r} axis "
                    f"({values.size if values is not None else '?'} samples).",
                    remediation=f"Add {name!r} to at=.")
        return tuple(index)
    index = at if isinstance(at, tuple) else (at,)
    if not all(isinstance(i, (int, np.integer)) for i in index):
        raise ConfigurationError(
            f"plot_beam_power: at={at!r} is neither an index nor a dict of "
            f"coordinates.",
            remediation="Pass an int (or tuple of ints) indexing the axes "
                        "after the angle one, or {'range': value}.")
    if len(index) > len(axes):
        raise ConfigurationError(
            f"plot_beam_power: at= gives {len(index)} indices for "
            f"{len(axes)} grid axes.")
    return tuple(int(i) for i in index)


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
    ax : matplotlib.axes.Axes, optional
        Existing axes, a polar one when ``polar`` is True; a new figure is
        made when omitted.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(6.5, 6.5)`` by default; unused when
        ``ax`` is given.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.

    Notes
    -----
    The angle axis is Bellhop's launch declination ``alpha``, in degrees —
    the same convention and the same units the fan is spelled in, which is
    why Bellhop's ``check_beam_pattern_spans_the_fan`` can compare the two
    directly. ``ray2D(1)%t = [COS(alpha), SIN(alpha)]/c``
    (``Bellhop/bellhop.f90:453``) over a depth axis that is positive downward
    sends ``alpha > 0`` deeper, so the polar axes run clockwise from due east
    and a lobe drawn below the horizontal is a lobe that ensonifies the
    depths below the source in the field plot.

    Levels are drawn as the table gives them, in dB, and labelled so: nothing
    here normalises the table to its peak, and neither does the engine —
    Bellhop applies each level as an *amplitude* factor ``10**(dB/20)``
    (``misc/beampattern.f90:59``), despite its print header calling the
    column "Power" — so a table peaking at +6 dB launches 6 dB of gain.
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
            f"Bellhop rejects any launch fan alpha "
            f"reaching into them. Pass mirror=True to reflect the table "
            f"through 0°.", FallbackWarning)

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
            f"mirror=True if it was written for the other side of 0°.", ValidityWarning)

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
        ax.set_ylabel('Level (dB)')
        # The same limit the polar axes takes. Which launch angles exist is a
        # fact about the fan, not about how the response is drawn.
        ax.set_xlim(fan_lo, fan_hi)
        ax.grid(True, alpha=0.3)
        ax.set_title(_title_or(title, default_title))
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
    ax.set_ylabel('Level (dB)', labelpad=22)   # the rectilinear view's label
    # A polar radius is short and every radial label sits on the one spoke, so
    # the ~9 ticks a linear dB axis defaults to overprint one another.
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    # Clear of that thetamin spoke, which runs along the top edge — and through
    # an unpadded title — whenever the wedge opens forward.
    ax.set_title(_title_or(title, default_title), pad=28.0)
    return fig, ax
