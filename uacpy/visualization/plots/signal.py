"""Plots for the signal-processing toolkits (acoustic_signal).

All DSP plotting lives here, not in the computation modules: ``acoustic_signal``
is pure computation and never imports matplotlib. Each function takes the
arrays a transform/estimator returns as its leading positional arguments, or,
for most, the result carrier itself in their place (``plot_psd(welch(x, fs))``).
The target ``ax`` follows the data arrays, so its position depends on how many
there are — pass it by keyword; a new figure is made when it is ``None``. Each
returns ``(fig, ax)``, as :func:`plot_field` does, except :func:`plot_frf`,
which returns ``(fig, (ax_mag, ax_phase))``, and
:func:`plot_lsfir_diagnostics`, which takes no ``ax`` and returns
``(fig, [ax1, ax2, ax3])``.
"""
import numpy as np
import matplotlib.pyplot as plt

from uacpy.core.constants import (REFERENCE_PRESSURE_AIR,
                                  REFERENCE_PRESSURE_WATER)
from uacpy.core.acoustics import power_to_dB
from uacpy.acoustic_signal._results import POWER_UNITS
from uacpy.core.exceptions import ConfigurationError, FallbackWarning
from uacpy.visualization.plots._common import (
    ZORDER_LEGEND, ZORDER_SOURCE, _axes_pair, _carrier_or_arrays,
    _cell_edge_extent, _checked_dynamic_range_dB, _flip_y, _plot_warn,
    _refuse_spread_carrier, _require_nonempty, _title_or, fig_ax,
    typed_plot_error)
from uacpy.visualization.style import SOURCE_MARKER_STYLE, cmap_for_field


def _require_image_grid(arr, n0, n1, who, name0, name1):
    """Guard an ``imshow`` panel against a coordinate/array mismatch.

    ``imshow`` only reads the first/last coordinate for its ``extent`` and
    stretches the whole array onto it, so a length mismatch yields a
    plausible-but-wrong figure with no error (unlike the pcolormesh plotters,
    which raise). Require the data array to be exactly ``(n0, n1)`` and raise a
    typed error naming the mismatch otherwise."""
    a = np.asarray(arr)
    if a.ndim != 2 or a.shape != (n0, n1):
        raise ConfigurationError(
            f"{who}: data array shape {a.shape} does not match the "
            f"coordinate lengths ({name0}={n0}, {name1}={n1}). Pass the "
            f"transform's own output unmodified — a mismatch would render a "
            f"wrong image (imshow) or raise matplotlib's own shading error "
            f"(pcolormesh).")
    return a


def _clamped_freq_limits(frequencies, clamp, hi):
    """``(lo, hi)`` frequency-axis limits that keep the axis ascending.

    A fixed low clamp keeps panels comparable and holds the near-DC bins off
    the axis, but a record whose whole band sits below the clamp (infrasound,
    or a very long window) cannot take it: the low limit would land above the
    high one and the axis would silently reverse, putting the record outside
    its own window. Such a band starts at its first positive bin instead —
    never DC, which a log axis cannot render at all."""
    f = np.asarray(frequencies, dtype=float)
    if hi > clamp:
        return (max(float(f[0]), float(clamp)), hi)
    positive = f[f > 0]
    return ((float(positive[0]) if positive.size else float(f[0])), hi)


def _log_freq_xlim(frequencies):
    """``(lo, hi)`` x-limits for a log frequency axis.

    A log axis cannot render DC and every FFT / Welch grid starts at f = 0, so
    the low end is clamped to 1 Hz; below that the first bin would drag the
    whole decade scale toward -inf."""
    f = np.asarray(frequencies, dtype=float)
    return _clamped_freq_limits(f, 1.0, float(f[-1]))


def _all_outside(values, lo, hi):
    """``(vmin, vmax, lo, hi)`` when every finite sample of ``values`` lies
    outside the window ``[lo, hi]``, else ``None``.

    An empty or all-NaN record is outside nothing and yields ``None``; the
    window is sorted, so a reversed pair is read as the interval it spans."""
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if not v.size or lo is None or hi is None:
        return None
    lo, hi = sorted((float(lo), float(hi)))
    if v.min() > hi or v.max() < lo:
        return (float(v.min()), float(v.max()), lo, hi)
    return None


#: The level axis of a dB plotter reaches at least this far (dB) beyond the
#: finite plotted levels, and its edges sit on multiples of
#: ``_LEVEL_WINDOW_STEP_DB``.
_LEVEL_WINDOW_MARGIN_DB = 5.0
_LEVEL_WINDOW_STEP_DB = 10.0


def _held_window(ax):
    """The y-limits ``ax`` shows when it already holds data drawn before this
    call (an overlay onto a caller's axes), else ``None``. Read before the
    plotter draws, since its own artists would count as data too."""
    return tuple(sorted(ax.get_ylim())) if ax.has_data() else None


def _level_window_dB(levels, ymin=None, ymax=None, held=None):
    """``(lo, hi)`` limits (dB) of a level axis: ``ymin`` / ``ymax`` where
    given, and each edge left ``None`` fitted to the finite plotted
    ``levels`` — the lowest level less ``_LEVEL_WINDOW_MARGIN_DB`` rounded
    down, the highest plus it rounded up, both to a multiple of
    ``_LEVEL_WINDOW_STEP_DB``. A fitted edge then widens to the ``held``
    window (:func:`_held_window`) of the data the axes already showed, so a
    second curve overlaid on a shared axes keeps the first on screen. An
    edge with no finite level to fit stays ``None``, which leaves that end to
    matplotlib; a fitted edge that would meet or cross the given one sits a
    step beyond it instead."""
    v = np.asarray(levels, dtype=float).ravel()
    v = v[np.isfinite(v)]
    step, margin = _LEVEL_WINDOW_STEP_DB, _LEVEL_WINDOW_MARGIN_DB
    lo = None if ymin is None else float(ymin)
    hi = None if ymax is None else float(ymax)
    if v.size:
        if lo is None:
            lo = float(step * np.floor((v.min() - margin) / step))
            if held is not None:
                lo = min(lo, held[0])
            if hi is not None and lo >= hi:
                lo = hi - step
        if hi is None:
            hi = float(step * np.ceil((v.max() + margin) / step))
            if held is not None:
                hi = max(hi, held[1])
            if hi <= lo:
                hi = lo + step
    return lo, hi


def _warn_if_offscreen(ax, values, who, knob):
    """Warn when a fixed y window excludes every finite sample.

    Several of these plotters pin the ordinate to the range their quantity
    normally occupies, which keeps panels comparable but renders a record
    outside that range as an empty panel — a silent, easily-missed failure that
    looks like "no data". Say so, and name the knob that widens the window."""
    outside = _all_outside(values, *ax.get_ylim())
    if outside is not None:
        v_min, v_max, lo, hi = outside
        _plot_warn(
            f"{who}: every sample ({v_min:.4g} … {v_max:.4g}) lies "
            f"outside the plotted y range ({lo:g}, {hi:g}), so the panel is "
            f"empty. Pass {knob}= to widen it.", FallbackWarning)


def _warn_if_colour_saturated(values, vmin, vmax, who, knob):
    """Warn when a fixed colour window excludes every finite sample.

    A pinned dB window keeps panels comparable, but a record entirely above or
    below it maps to one end of the colormap everywhere: a flat single-colour
    image that reads as a valid featureless record rather than as a window
    problem. Say so, and name the knobs that move the window."""
    outside = _all_outside(values, vmin, vmax)
    if outside is not None:
        v_min, v_max, lo, hi = outside
        _plot_warn(
            f"{who}: every sample ({v_min:.4g} … {v_max:.4g} dB) lies "
            f"outside the colour window ({lo:g}, {hi:g}) dB, so the panel is a "
            f"single flat colour. Pass {knob}= to move it.", FallbackWarning)


def _ref_label(ref):
    """The reference pressure as the prefix of a ``Pa`` unit on a level
    axis, spaced as the results' ``.units`` spell it: ``'1 µ'`` in
    ``'dB re 1 µPa²'``."""
    if ref == REFERENCE_PRESSURE_WATER:
        return "1 µ"
    if ref == REFERENCE_PRESSURE_AIR:
        return "20 µ"
    return f"{ref:g} "




# ── f-k / Radon / tau-p gather transforms ───────────────────────────────────

#: How :func:`plot_fk` labels its abscissa per wavenumber unit, and the factor
#: that turns :func:`fk_transform`'s angular ``k`` (rad/m) into that unit.
_FK_WAVENUMBER_AXIS = {
    "rad/m": ("Wavenumber k (rad/m)", 1.0),
    "cycles/m": ("Wavenumber ν (cycles/m)", 1.0 / (2.0 * np.pi)),
}


def draw_sound_cone(ax, freq_max, k_max, sound_speed, *, color="w", ls="--",
                    lw=1.1, alpha=0.85, label=True, wavenumber_unit="rad/m"):
    """Overlay the acoustic cone of speed ``sound_speed`` onto an f-k axis.

    With ``wavenumber_unit='rad/m'`` (the default, matching
    :func:`fk_transform`) the abscissa is the angular wavenumber ``k`` and the
    cone is ``f = c·k/2π``; with ``'cycles/m'`` it is ``ν = k/2π`` and the cone
    is ``f = c·ν``. ``k_max`` is the axis edge in the same unit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The f-k axes to draw on.
    freq_max : float
        Top of the frequency axis (Hz).
    k_max : float
        Edge of the wavenumber axis, in ``wavenumber_unit``.
    sound_speed : float
        Speed (m/s) of the cone.
    color : str, optional
        Line and label colour. Default white.
    ls, lw, alpha : optional
        Line style, width and opacity.
    label : bool, optional
        Write the speed beside the line. Default True.
    wavenumber_unit : {'rad/m', 'cycles/m'}, optional
        Unit of the wavenumber axis. Default ``'rad/m'``.
    """
    c = float(sound_speed)
    if wavenumber_unit not in _FK_WAVENUMBER_AXIS:
        raise ConfigurationError(
            f"draw_sound_cone: wavenumber_unit must be one of "
            f"{tuple(_FK_WAVENUMBER_AXIS)}; got {wavenumber_unit!r}.")
    # The angular wavenumber 2π·freq_max/c, scaled into the axis unit.
    k_at_fmax = 2.0 * np.pi * freq_max / c * _FK_WAVENUMBER_AXIS[wavenumber_unit][1]
    k = min(k_at_fmax, k_max)   # cone reaches freq_max or the axis edge
    f = freq_max * k / k_at_fmax
    ax.plot([0, k], [0, f], color=color, ls=ls, lw=lw, alpha=alpha)
    ax.plot([0, -k], [0, f], color=color, ls=ls, lw=lw, alpha=alpha)
    if label:
        ax.text(k, f, f" {c:.0f} m/s", color=color, fontsize='small',
                va="top", ha="right")


def _fk_scaling(power, scaling):
    """The unit the f-k panel is in: the ``scaling`` an :class:`FKResult`
    carries, or the explicit knob, and both must agree when both are given."""
    carried = getattr(power, "scaling", None)
    if scaling is None:
        if carried is None:
            raise ConfigurationError(
                "plot_fk: scaling= is required with a bare power array, "
                "because a density (fk_transform(..., scaling='density')) "
                "and the raw |FK|² panel (scaling='power') "
                "are labelled differently and nothing in "
                "the array says which it is. Pass scaling='density' or "
                "scaling='power', or hand plot_fk the FKResult itself.")
        return carried
    # Imported here, not at module level: ``arrays`` pulls in scipy.signal,
    # which the plotting surface leaves unloaded until a DSP plot is drawn.
    from uacpy.acoustic_signal.gathers import FK_SCALINGS
    if scaling not in FK_SCALINGS:
        raise ConfigurationError(
            f"plot_fk: scaling must be one of {FK_SCALINGS}; got {scaling!r}.")
    if carried is not None and carried != scaling:
        raise ConfigurationError(
            f"plot_fk: scaling={scaling!r} contradicts the FKResult, whose "
            f"panel is scaling={carried!r}. Drop scaling= to use the "
            f"result's own, or re-run fk_transform with "
            f"scaling={scaling!r}.")
    return scaling


def _fk_colorbar_label(scaling, wavenumber_unit, ref):
    """What the colour axis measures, per panel scaling and wavenumber unit.

    A density is per Hz and per wavenumber unit, so its unit follows the
    abscissa: Pa²·m/(Hz·rad) over rad/m, Pa²·m/Hz over cycles/m (a cycle is
    dimensionless). The raw panel is a sum over the gather with no unit."""
    r = _ref_label(ref)
    if scaling == "power":
        return f"|FK|² (dB re {r}Pa², unnormalised)"
    if wavenumber_unit == "cycles/m":
        return f"PSD (dB re {r}Pa²·m/Hz)"
    return f"PSD (dB re {r}Pa²·m/(Hz·rad))"


# The colour window autoscales, as it does on the other transform panels
# (plot_radon, plot_taup). A fixed -60..+20 dB window suits a PEAK-RELATIVE
# scale — which is what docs/figure_scripts/signal.py hand-rolls, at
# vmin=-40, vmax=0 — but the level here goes through
# power_to_dB(power, ref), an ABSOLUTE dB re 1 µPa². Measured on the
# fk_transform output this function documents itself as consuming, for a 1 Pa
# plane-wave gather at fs = 2 kHz, dx = 2 m: the panel spans 107.3 .. 196.9 dB
# with a median of 122.2, so every pixel would sit above that vmax and the
# figure would come out a uniform block. A fixed absolute window cannot work here anyway: the
# transform sums over the gather, so the level moves with its size.
@typed_plot_error
def plot_fk(frequencies, wavenumbers=None, power=None, ax=None, *,
            scaling=None, wavenumber_unit="rad/m",
            ref=REFERENCE_PRESSURE_WATER, vmin=None, vmax=None, cmap=None,
            sound_speed=None, title=None, figsize=(10, 6), show_colorbar=True,
            **mpl_kw):
    """Image an f-k panel (dB). Consumes :func:`fk_transform` output.

    Handed the result itself — ``plot_fk(fk_transform(gather, fs, dx))`` — the
    colour axis follows the result's ``scaling``: a calibrated density
    (``scaling='density'``) is labelled "PSD" in Pa² per Hz per wavenumber unit,
    the raw ``|FK|²`` panel (``scaling='power'``) is labelled as unnormalised
    power. Handed bare ``(frequencies, wavenumbers, power)`` arrays the
    transform's choice is not recoverable, so ``scaling=`` (``'density'`` or
    ``'power'``) must state it; given alongside a result it must agree.

    Parameters
    ----------
    frequencies : ndarray or FKResult
        The frequency axis (Hz), or the whole :class:`FKResult`, in which case
        ``wavenumbers`` and ``power`` are taken from it.
    wavenumbers : ndarray, optional
        Angular wavenumber ``k`` (rad/m), as :func:`fk_transform` returns it.
    power : ndarray, optional
        The panel ``(len(frequencies), len(wavenumbers))``.
    scaling : {'density', 'power'}, optional
        Which unit ``power`` is in; ``None`` reads it from an ``FKResult``.
    wavenumber_unit : {'rad/m', 'cycles/m'}
        Abscissa unit. ``'cycles/m'`` divides the axis by 2π and, for a
        density, multiplies the panel by 2π so that ``ΣP·Δf·Δν`` still equals
        the gather's mean square; a raw panel is left as it is.
    sound_speed : float, optional
        Draws the acoustic cone ``ω = c·k`` (or ``f = c·ν``) of that speed.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    ref : float, optional
        Reference pressure (Pa) of the dB scale, 1 µPa by default.
    vmin, vmax : float, optional
        Colour limits (dB); ``None`` scales to the data.
    cmap : str, optional
        Colormap; ``None`` takes matplotlib's default.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(10, 6)`` by default; unused when
        ``ax`` is given.
    show_colorbar : bool, optional
        Draw the colorbar, labelled with the scaling and the reference.
        Default True.
    """
    # One FKResult in place of the three arrays: read the axes, the panel and
    # its scaling from it. The result's fourth element is the spectrum, so
    # plot_fk(*result) would land it in ax=; say so instead of failing inside
    # matplotlib.
    result = None
    _unpacked = _carrier_or_arrays(
        frequencies, (wavenumbers, power), count=3, who="plot_fk",
        carrier=(('FKResult',), ('frequencies', 'wavenumbers', 'power')),
        fields=('frequencies', 'wavenumbers', 'power'))
    if _unpacked is not None:
        result = frequencies
        frequencies, wavenumbers, power = _unpacked
    elif isinstance(ax, np.ndarray):
        raise ConfigurationError(
            "plot_fk: ax= received an array — plot_fk(*result) spreads the "
            "spectrum into ax=. Pass the result itself: plot_fk(result).")
    if wavenumbers is None or power is None:
        raise ConfigurationError(
            "plot_fk: pass (frequencies, wavenumbers, power) or one FKResult.")
    scaling = _fk_scaling(result if result is not None else power, scaling)
    if wavenumber_unit not in _FK_WAVENUMBER_AXIS:
        raise ConfigurationError(
            f"plot_fk: wavenumber_unit must be one of "
            f"{tuple(_FK_WAVENUMBER_AXIS)}; got {wavenumber_unit!r}.")
    xlabel, k_scale = _FK_WAVENUMBER_AXIS[wavenumber_unit]
    panel = np.asarray(power, dtype=float)
    _require_image_grid(panel, len(frequencies), len(wavenumbers),
                        "plot_fk", "frequencies", "wavenumbers")
    # A density per rad/m is 2π times larger per cycle/m (dν = dk/2π); the raw
    # panel has no per-wavenumber unit and is not rescaled.
    if scaling == "density":
        panel = panel / k_scale
    k_axis = np.asarray(wavenumbers, dtype=float) * k_scale
    fk_dB = power_to_dB(panel, ref)
    fig, ax = fig_ax(ax, figsize)
    # Edge-aligned: the axes are FFT bin centres, and draw_sound_cone below
    # places the cone at true coordinates, so a half-bin shift would offset
    # the image against the very line used to read it.
    im = ax.imshow(fk_dB, extent=_cell_edge_extent(k_axis, frequencies),
                   origin="lower", aspect="auto",
                   vmin=vmin, vmax=vmax, cmap=cmap, **mpl_kw)
    if sound_speed is not None:
        draw_sound_cone(ax, frequencies[-1], k_axis[-1], sound_speed,
                        wavenumber_unit=wavenumber_unit)
    ax.set_title(_title_or(title, "f–k spectrum"))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency (Hz)")
    ax.grid(alpha=0.3)
    if show_colorbar:
        # With the reference: `power_to_dB(panel, ref)` above makes this an
        # ABSOLUTE level, so the number is meaningless without saying what it
        # is referred to — as every sibling axis in this module does.
        fig.colorbar(im, ax=ax,
                     label=_fk_colorbar_label(scaling, wavenumber_unit, ref))
    return fig, ax


# Axis label and SI → display scale per Radon moveout family.
# ``radon_transform`` scans moveout in SI units of the offset in metres:
# slowness s/m (×1e3 → s/km), curvature s/m² (×1e6 → s/km²), velocity m/s
# (already the display unit, ×1).
_RADON_AXIS = {
    "linear": ("Slowness p (s/km)", 1e3),
    "parabolic": ("Curvature q (s/km²)", 1e6),
    "hyperbolic": ("Velocity v (m/s)", 1.0),
}


def _plot_tau_panel(x, taus, amp, ax, *, vmin, vmax, cmap, figsize,
                    show_colorbar, **mpl_kw):
    """Image a stack ``amp`` (x, tau) with the largest tau at the bottom, so
    intercept time runs downward: the seismic gather convention."""
    fig, ax = fig_ax(ax, figsize)
    im = ax.imshow(amp.T, aspect="auto", origin="upper",
                   extent=_flip_y(_cell_edge_extent(x, taus)),
                   vmin=0.0 if vmin is None else vmin,
                   vmax=amp.max() if vmax is None else vmax,
                   cmap=cmap, **mpl_kw)
    ax.set_ylabel("Intercept time tau (s)")
    if show_colorbar:
        fig.colorbar(im, ax=ax, label="Stack amplitude")
    return fig, ax


@typed_plot_error
def plot_radon(moveout, taus=None, R=None, ax=None, *, kind=None,
               vmin=None,
               vmax=None, cmap="jet", title=None, figsize=(8, 6),
               show_colorbar=True, **mpl_kw):
    """Image ``|R|`` (moveout on x, intercept time on y). Consumes
    :func:`radon_transform` output.

    ``kind`` is the moveout family the panel was scanned in —
    ``'linear'`` (slowness, drawn in s/km), ``'parabolic'`` (curvature,
    s/km²) or ``'hyperbolic'`` (velocity, m/s) — and sets the axis's name and
    scale. A :class:`RadonResult` carries its own; given alongside one,
    ``kind=`` must agree with it. Bare arrays default to ``'linear'``. Any
    other value is refused: drawn under a fallback axis it would put the
    panel on a scale nobody chose.

    Parameters
    ----------
    moveout : RadonResult or array_like
        The result itself, or its moveout axis.
    taus : array_like, optional
        Intercept times (s), with array input.
    R : array_like, optional
        The ``(n_moveout, n_tau)`` panel, with array input.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    kind : {'linear', 'parabolic', 'hyperbolic'}, optional
        The moveout family (see above).
    vmin, vmax : float, optional
        Colour limits of the amplitude; ``None`` is 0 and the panel's peak.
    cmap : str, optional
        Colormap. Default ``'jet'``.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(8, 6)`` by default; unused when
        ``ax`` is given.
    show_colorbar : bool, optional
        Draw the colorbar. Default True.
    **mpl_kw
        Matplotlib keywords for the image.
    """
    # One RadonResult in place of the arrays — the same call
    # ``RadonResult.plot()`` makes, so the two spellings agree.
    carried = getattr(moveout, 'kind', None)
    _unpacked = _carrier_or_arrays(
        moveout, (taus, R,), count=3, who="plot_radon",
        carrier=(('RadonResult',), ('moveout', 'taus', 'panel')),
        fields=('moveout', 'taus', 'R'))
    if _unpacked is not None:
        moveout, taus, R = _unpacked
    else:
        carried = None
    if taus is None or R is None:
        raise ConfigurationError(
            "plot_radon: pass every array, or one RadonResult.")
    if kind is not None and kind not in _RADON_AXIS:
        raise ConfigurationError(
            f"plot_radon: kind={kind!r} is not a Radon moveout family; pass "
            f"one of {tuple(_RADON_AXIS)} — the one radon_transform scanned.")
    if kind is not None and carried is not None and kind != carried:
        raise ConfigurationError(
            f"plot_radon: kind={kind!r} contradicts the RadonResult, which "
            f"was scanned with kind={carried!r}. Drop kind= to use the "
            f"result's own.")
    kind = kind or carried or "linear"
    if kind not in _RADON_AXIS:
        raise ConfigurationError(
            f"plot_radon: the RadonResult carries kind={kind!r}, which is not "
            f"one of {tuple(_RADON_AXIS)}.")
    amp = _require_image_grid(np.abs(np.asarray(R)), len(moveout), len(taus),
                              "plot_radon", "moveout", "taus")
    xlabel, scale = _RADON_AXIS[kind]
    fig, ax = _plot_tau_panel(np.asarray(moveout) * scale, taus, amp, ax,
                              vmin=vmin, vmax=vmax, cmap=cmap, figsize=figsize,
                              show_colorbar=show_colorbar, **mpl_kw)
    ax.set_title(_title_or(title, f"Radon ({kind})"))
    ax.set_xlabel(xlabel)
    return fig, ax


def draw_slowness_line(ax, tau_max, sound_speed, *, color="w", ls="--",
                       lw=1.1, alpha=0.85, label=True):
    """Mark slowness ``p = +/-1/c`` (s/km) of a reference speed on a tau-p axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The tau-p axes to draw on.
    tau_max : float
        Intercept time (s) the label sits at.
    sound_speed : float
        Reference speed (m/s).
    color : str, optional
        Line and label colour. Default white.
    ls, lw, alpha : optional
        Line style, width and opacity.
    label : bool, optional
        Write the speed beside the line. Default True.
    """
    p_skm = 1000.0 / float(sound_speed)          # 1/c in s/m → s/km
    for sgn in (-1.0, 1.0):
        ax.axvline(sgn * p_skm, color=color, ls=ls, lw=lw, alpha=alpha)
    if label:
        ax.text(p_skm, tau_max, f" {sound_speed:.0f} m/s", color=color,
                fontsize='small', va="bottom", ha="left")


@typed_plot_error
def plot_taup(slownesses, taus=None, taup=None, ax=None, *, vmin=None,
              vmax=None,
              cmap="jet", sound_speed=None, title=None, figsize=(8, 6),
              show_colorbar=True, **mpl_kw):
    """Image a tau-p panel (slowness s/km on x, intercept time on y). Consumes
    :func:`taup_transform` output.

    Parameters
    ----------
    slownesses : TauPResult or array_like
        The result itself, or its slowness axis (s/m).
    taus : array_like, optional
        Intercept times (s), with array input.
    taup : array_like, optional
        The ``(n_slowness, n_tau)`` panel, with array input.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    vmin, vmax : float, optional
        Colour limits of the amplitude; ``None`` is 0 and the panel's peak.
    cmap : str, optional
        Colormap. Default ``'jet'``.
    sound_speed : float, optional
        Mark ``p = ±1/c`` of this speed (m/s).
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(8, 6)`` by default; unused when
        ``ax`` is given.
    show_colorbar : bool, optional
        Draw the colorbar. Default True.
    **mpl_kw
        Matplotlib keywords for the image.
    """
    # One TauPResult in place of the arrays — the same call
    # ``TauPResult.plot()`` makes, so the two spellings agree.
    _unpacked = _carrier_or_arrays(
        slownesses, (taus, taup,), count=3, who="plot_taup",
        carrier=(('TauPResult',), ('slownesses', 'taus', 'panel')),
        fields=('slownesses', 'taus', 'taup'))
    if _unpacked is not None:
        slownesses, taus, taup = _unpacked
    if taus is None or taup is None:
        raise ConfigurationError(
            "plot_taup: pass every array, or one TauPResult.")
    p_skm = np.asarray(slownesses) * 1000.0      # taup_transform returns s/m
    amp = _require_image_grid(np.abs(np.asarray(taup)), len(slownesses),
                              len(taus), "plot_taup", "slownesses", "taus")
    fig, ax = _plot_tau_panel(p_skm, taus, amp, ax, vmin=vmin, vmax=vmax,
                              cmap=cmap, figsize=figsize,
                              show_colorbar=show_colorbar, **mpl_kw)
    if sound_speed is not None:
        draw_slowness_line(ax, taus[-1], sound_speed)
    ax.set_title(_title_or(title, "tau-p"))
    ax.set_xlabel("Slowness p (s/km)")
    return fig, ax


# ── Spectral / level estimators (analysis) ──────────────────────────────────

#: What each scaling's panel is called; the unit it puts on a level axis is
#: POWER_UNITS'. An exposure carries the record's duration and a density the
#: band's width, so a plot that names the wrong one is out by that factor and
#: says nothing.
_SCALING_KIND = {"density": "Power spectral density",
                 "spectrum": "Power spectrum",
                 "exposure": "Sound exposure"}
#: What a level HISTOGRAM is called, per scaling. "PPSD" is the name the
#: density one goes by (McNamara & Buland 2004); the other two have no
#: acronym, and calling them PPSD would put "density" over band power.
_HISTOGRAM_KIND = {"density": "PPSD", "spectrum": "Band-power histogram",
                   "exposure": "SEL histogram"}


#: How a standard band ladder is named on a frequency axis.
_BAND_NAMES = {"decidecade": "decidecade", "octave": "octave",
               "linear": "linear"}


def _band_axis_label(band_type):
    """Frequency-axis label for values sitting on ``band_type`` bands."""
    return f"Frequency, {_BAND_NAMES.get(band_type, band_type)} bands (Hz)"


def _scaling_unit(who, scaling):
    """The level unit of ``scaling``, refused by name when it is not one.

    Looked up bare, a typo reached the generic "check the arrays" message
    of :func:`typed_plot_error` as a ``KeyError``."""
    if scaling not in POWER_UNITS:
        raise ConfigurationError(
            f"{who}: scaling={scaling!r} is not a spectral scaling; pass one "
            f"of {tuple(POWER_UNITS)} — the one the estimator used.")
    return POWER_UNITS[scaling]


def _histogram_title(result):
    """The panel's own name: the statistic, and the segment it summarises."""
    kind = _HISTOGRAM_KIND[getattr(result, "scaling", "density")]
    seg = getattr(result, "segment_duration", None)
    return f"{kind} ({seg}s)" if seg is not None else kind


#: Default reach of ``plot_waveform``'s envelope dB axis below its reference.
_WAVEFORM_DYNAMIC_RANGE_DB = 60.0


@typed_plot_error
def plot_waveform(signal, sample_rate, ax=None, *, value='pressure',
                  t0=0.0, reference=None, dynamic_range_dB=None,
                  time_units='s',
                  label=None, title=None, figsize=(10, 4), **mpl_kw):
    """Line plot of a signal against time — pressure, or its envelope.

    The elementary time-domain view, for a bare array and its sample rate.
    A gridded result plots itself (``Field`` over ``{'time'}`` does it through
    :meth:`~uacpy.core.results.Field.plot`); this is the door for a waveform
    that is not one — a generator's output, a synthesised burst, a recording.

    Several signals go on one axis the way the rest of the family does it, by
    handing back the axis::

        _fig, ax = plot_waveform(sent, fs, label='Transmitted')
        plot_waveform(received, fs, ax, label='Received')

    ``value`` picks the view:

    - ``'pressure'`` — the waveform itself. The default.
    - ``'envelope'`` — ``|analytic signal|``
      (:func:`uacpy.acoustic_signal.envelope`), the shape under the carrier.
      At 40 kHz over 60 ms the waveform alone is 2 400 cycles of solid ink
      and the envelope is the only thing a reader can take from it.
    - ``'envelope_dB'`` — the same in decibels, which is where a tail shows.
      On a linear axis a channel that smears 1 % of a pulse into the next
      slot draws the same as one that smears none.

    The envelope is computed by the public transform and not here, so a
    caller can have the numbers without the figure. What IS the plotter's is
    ``dynamic_range_dB``: ``20*log10`` of a silence is minus infinity and takes
    the axis with it, so the dB view is floored, exactly as
    :func:`~uacpy.plot.plot_channel` floors its null.

    Parameters
    ----------
    signal : array_like
        The 1-D waveform. The generators return a ``(time, signal)`` pair, so
        pass the signal alone — ``tone_burst(...)[1]``.
    sample_rate : float
        Hz. Positive and finite.
    t0 : float, default 0.0
        Time of the first sample, in seconds. A bare array carries no time
        origin the way a ``Field`` over ``{'time'}`` does, so an excerpt, a
        trace that starts after a travel time, or two signals being lined up
        on a common instant need one stated. Signed.
    ax : matplotlib.axes.Axes, optional
        Draw here instead of a new figure.
    value : {'pressure', 'envelope', 'envelope_dB'}, default 'pressure'
    reference : float, optional
        ``'envelope_dB'`` only. ``None`` (default) references each trace to
        its own peak, which compares SHAPES; a number references them all to
        it, which compares LEVELS. Two traces normalised separately cannot be
        read against each other for level, so the axis label says which it is.
    dynamic_range_dB : float, optional
        ``'envelope_dB'`` only: how far below the reference the axis reaches,
        a positive number of dB (60 when unset). Refused with any other
        ``value``, which has no dB axis to floor.
    time_units : {'s', 'ms'}, default 's'
        ``'s'`` is what a ``Field`` over ``{'time'}`` draws; ``'ms'`` is
        readable for a burst tens of milliseconds long.
    label, title, figsize, **mpl_kw
        As elsewhere in the family; ``label`` is for the legend the caller
        draws.

    Returns
    -------
    (fig, ax)

    Raises
    ------
    ConfigurationError
        A ``(time, signal)`` pair, an empty or non-1-D signal, a non-positive
        sample rate, an unknown ``value`` or ``time_units``, a non-finite
        ``t0``, a non-positive ``dynamic_range_dB`` or one given without
        ``value='envelope_dB'``, or a non-positive ``reference``.
    """
    who = "plot_waveform"
    if isinstance(signal, tuple) or (
            np.ndim(signal) == 2 and np.shape(signal)[0] == 2):
        raise ConfigurationError(
            f"{who}: signal must be the 1-D waveform, not a (time, signal) "
            f"pair — the generators return the pair, so pass "
            f"tone_burst(...)[1]. Reading the pair would draw the time "
            f"vector as the waveform.")
    raw = np.asarray(signal)
    if np.iscomplexobj(raw):
        # Cast to float would drop the imaginary part behind a bare numpy
        # ComplexWarning — no uacpy guard, and wrong for the one caller who
        # would pass complex deliberately: an analytic signal, whose real
        # part is the waveform and whose modulus is the envelope. Which of
        # those they want is not ours to guess.
        raise ConfigurationError(
            f"{who}: signal is complex. Casting it would silently discard "
            f"the imaginary part. Pass signal.real for the waveform, or "
            f"np.abs(signal) for the envelope of an analytic signal.")
    x = np.asarray(raw, dtype=float)
    if x.ndim > 1:
        # The pair guard above catches (2, n) and (n, 2); any other 2-D or
        # higher shape reached ravel() and was flattened into one long
        # "waveform" — a (3, 100) array plotted as 300 samples, which is the
        # same failure the pair guard exists to prevent, one shape over.
        raise ConfigurationError(
            f"{who}: signal must be 1-D; got shape {x.shape}. Flattening it "
            f"would draw {x.size} samples end to end as one waveform. Pick "
            f"the channel or trace you meant, e.g. signal[0].")
    x = x.ravel()
    _require_nonempty(who, signal=x)
    fs = float(sample_rate)
    if not np.isfinite(fs) or fs <= 0.0:
        raise ConfigurationError(
            f"{who}: sample_rate must be positive and finite (Hz); got "
            f"{sample_rate!r}.")
    if value not in ('pressure', 'envelope', 'envelope_dB'):
        raise ConfigurationError(
            f"{who}: value must be 'pressure' (the waveform), 'envelope' "
            f"(|analytic signal|) or 'envelope_dB' (the same in decibels); "
            f"got {value!r}.")
    if time_units not in ('s', 'ms'):
        raise ConfigurationError(
            f"{who}: time_units must be 's' or 'ms'; got {time_units!r}.")
    if dynamic_range_dB is not None and value != 'envelope_dB':
        raise ConfigurationError(
            f"{who}: dynamic_range_dB= floors the dB axis of "
            f"value='envelope_dB'; value={value!r} draws a linear axis with "
            f"nothing to floor.")

    if value == 'pressure':
        y, ylabel = x, "p(t)"
    else:
        # Deferred like every compute-side import here: at file scope it
        # pulls scipy.signal into every ``import uacpy.visualization``.
        from uacpy.acoustic_signal.timefreq import envelope
        y = np.asarray(envelope(x), dtype=float)
        if value == 'envelope':
            ylabel = "|envelope|"
        else:
            if reference is None:
                ref, ylabel = y.max(), "Envelope (dB re peak)"
            else:
                ref = float(reference)
                if not np.isfinite(ref) or ref <= 0.0:
                    raise ConfigurationError(
                        f"{who}: reference must be a positive, finite "
                        f"amplitude; got {reference!r}.")
                ylabel = f"Envelope (dB re {ref:g})"
            floor_dB = -(_WAVEFORM_DYNAMIC_RANGE_DB if dynamic_range_dB is None
                         else _checked_dynamic_range_dB(who, dynamic_range_dB))
            if ref <= 0.0:
                # An all-zero trace has no peak to refer to, and dividing by
                # it would draw a floor-flat line that looks like silence
                # measured rather than silence handed in.
                raise ConfigurationError(
                    f"{who}: the signal is everywhere zero, so it has no "
                    f"peak to reference; pass reference= to state one.")
            y = 20.0 * np.log10(np.maximum(y / ref, 10.0 ** (floor_dB / 20.0)))

    start = float(t0)
    if not np.isfinite(start):
        raise ConfigurationError(
            f"{who}: t0 must be a finite time in seconds; got {t0!r}.")
    scale = 1.0 if time_units == 's' else 1e3
    t = (start + np.arange(x.size) / fs) * scale
    fig, ax = fig_ax(ax, figsize)
    ax.plot(t, y, label=label, **mpl_kw)
    ax.set_xlabel(f"Time ({time_units})")
    ax.set_ylabel(ylabel)
    ax.set_title(_title_or(title, "Waveform"))
    ax.grid(alpha=0.3)
    if value == 'envelope_dB':
        ax.set_ylim(floor_dB, None)
    return fig, ax


@typed_plot_error
def plot_psd(frequencies, psd_linear=None, ax=None, *,
             ref=REFERENCE_PRESSURE_WATER, scaling=None, label=None, ymin=None,
             ymax=None, title=None, figsize=(10, 6), freq_scale="log",
             **mpl_kw):
    """Line plot of a power spectral density (dB). Consumes
    :func:`uacpy.acoustic_signal.welch` or
    :func:`uacpy.acoustic_signal.constant_q` output.

    Named for the quantity, not the estimator that produced it — it draws
    either, and a third estimator would need no new plotter.

    Handed the result itself — ``plot_psd(welch(x, fs, scaling='spectrum'))`` — the axis
    follows the estimator: Pa² and "Power spectrum" for band power, Pa²/Hz and
    "Power spectral density" for a density. Handed bare arrays the estimator's
    choice is not recoverable, so ``scaling=`` states it and defaults to
    ``'density'``. The two differ by the window's noise-equivalent bandwidth —
    18.5 dB for a 1024-point Hann at 48 kHz — so an axis reading "/Hz" over
    band power is wrong by more than any plot convention.

    ``freq_scale`` is ``'log'`` by default, which is what spaces constant-Q's
    geometric bins evenly and what a decade-spanning soundscape wants;
    ``'linear'`` reads a narrow band the way a spectrum analyser does.

    ``ymin`` / ``ymax`` (dB) pin the level axis; an edge left ``None`` is
    fitted to the finite levels drawn, at least 5 dB beyond them and on a
    multiple of 10 dB — the one rule of this module's level plotters
    (:func:`plot_constant_q_psd`, :func:`plot_ppsd`,
    :func:`plot_constant_q_ppsd`, :func:`plot_sel`).

    Parameters
    ----------
    frequencies : SpectralEstimate or array_like
        The estimate itself, or its frequency axis (Hz).
    psd_linear : array_like, optional
        Power per bin, linear, with array input.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    ref : float, optional
        Reference pressure (Pa) of the dB scale. Default 1 µPa.
    scaling : {'density', 'spectrum'}, optional
        What bare arrays hold (see above); ``None`` is ``'density'``.
    label : str, optional
        Legend label of the line.
    ymin, ymax : float, optional
        Level-axis limits (dB); an edge left ``None`` is fitted to the levels
        drawn (see above).
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(10, 6)`` by default; unused when
        ``ax`` is given.
    freq_scale : {'log', 'linear'}, optional
        Frequency-axis scale. Default ``'log'``.
    **mpl_kw
        Matplotlib keywords for the line.
    """
    method = None
    _unpacked = _carrier_or_arrays(
        frequencies, (psd_linear,), count=2, who="plot_psd",
        carrier=(('SpectralEstimate',), ('frequencies', 'power')),
        fields=('frequencies', 'psd_linear'))
    if _unpacked is not None:
        carried = getattr(frequencies, "scaling", None)
        if (scaling is not None and carried is not None
                and str(scaling) != str(carried)):
            raise ConfigurationError(
                f"plot_psd: scaling={scaling!r} contradicts the "
                f"SpectralEstimate, whose power is scaling={carried!r}. Drop "
                f"scaling= to use the estimate's own, or re-run the "
                f"estimator with scaling={scaling!r}.")
        scaling = carried if carried is not None else scaling
        method = getattr(frequencies, "method", None)
        frequencies, psd_linear = _unpacked
    elif psd_linear is None:
        raise ConfigurationError(
            "plot_psd: pass both arrays, plot_psd(frequencies, psd_linear), "
            "or one SpectralEstimate, plot_psd(welch(x, fs)).")
    scaling = "density" if scaling is None else str(scaling)
    unit = _scaling_unit("plot_psd", scaling)
    # The estimate carries the method too, so the title names the bins the
    # reader is looking at: equal-width from Welch, geometric from constant-Q.
    kind = _SCALING_KIND[scaling]
    if method == "constant_q":
        kind = f"Constant-Q {kind[0].lower()}{kind[1:]}"
    if freq_scale not in ("log", "linear"):
        raise ConfigurationError(
            f"plot_psd: unknown freq_scale {freq_scale!r}.",
            remediation="Use 'log' (the default, and the only one that spaces "
                        "constant-Q's geometric bins evenly) or 'linear', "
                        "which reads a narrow band the way a spectrum "
                        "analyser does.")
    psd_dB = power_to_dB(np.asarray(psd_linear), ref)
    frequencies = np.asarray(frequencies, dtype=float)
    if freq_scale == "log" and np.any(frequencies > 0.0):
        # A log axis has no place for the DC bin; drawn, matplotlib clips it
        # to the axis edge and the line runs flat from there to the first
        # positive bin, a curve over frequencies the estimate has no bin for.
        positive = frequencies > 0.0
        frequencies, psd_dB = frequencies[positive], psd_dB[positive]
    fig, ax = fig_ax(ax, figsize)
    held = _held_window(ax)
    ax.plot(frequencies, psd_dB, label=label, **mpl_kw)
    ax.set_xscale(freq_scale)
    ax.set_title(_title_or(title, kind))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"Level (dB re {_ref_label(ref)}{unit})")
    ax.set_ylim(_level_window_dB(psd_dB, ymin, ymax, held))
    # A log axis cannot show DC, so its lower limit is the first positive
    # bin; a linear one spans the record's own band.
    ax.set_xlim(_log_freq_xlim(frequencies) if freq_scale == "log"
                else (float(frequencies[0]), float(frequencies[-1])))
    _warn_if_offscreen(ax, psd_dB, "plot_psd", "ymin=/ymax")
    ax.grid(which="both", alpha=0.75)
    if label:
        ax.legend()
    return fig, ax


def _plot_level_histogram(result, ax, *, y_label, default_title, who,
                          ymin, ymax, vmin, vmax, cmap, title, figsize,
                          show_colorbar, **mpl_kw):
    """Render one PPSD-style level histogram: the density mesh, the mean and
    ±1 STD lines over it, and a log frequency axis.

    Shared by :func:`plot_ppsd` and :func:`plot_constant_q_ppsd`, which differ
    only in the level axis label, the default title and the name they report
    in an off-screen warning. Any result carrying ``frequencies``,
    ``level_edges``, ``pdf``, ``mean_dB``, ``std_dB`` and ``level_step_dB``
    renders here; a caller's own shape guards run before it, so nothing that
    would raise gets a figure allocated first."""
    if vmax is None:
        # Each frequency column integrates to 1 over the level axis, so the
        # largest attainable density is 1/binwidth (all mass in one bin) —
        # the natural top of the colour scale.
        vmax = 1 / result.level_step_dB
    fig, ax = fig_ax(ax, figsize)
    held = _held_window(ax)
    # ``level_edges`` are bin EDGES and ``pdf`` has one row per bin; shift by
    # half a bin so each row is centred on its own level. Empty bins (0) and
    # columns with no level in the window (NaN) render as the axes
    # background.
    align = result.level_step_dB / 2
    shown = np.asarray(result.pdf, dtype=float)
    shown = np.ma.masked_where(~(shown > 0), shown)
    pcm = ax.pcolormesh(result.frequencies, result.level_edges[:-1] + align,
                        shown, cmap=cmap, shading="auto",
                        vmin=vmin, vmax=vmax, **mpl_kw)
    if show_colorbar:
        fig.colorbar(pcm, ax=ax,
                     label=f"Probability Density ({result.level_step_dB:.1f} dB/bin)")
    ax.plot(result.frequencies, result.mean_dB, "k-", label="Mean level", lw=1.5)
    ax.plot(result.frequencies, result.mean_dB + result.std_dB, "k--",
            label="Mean level ± STD")
    ax.plot(result.frequencies, result.mean_dB - result.std_dB, "k--")
    ax.set_title(_title_or(title, default_title))
    # A banded histogram's values sit on whole bands, so the axis names
    # the ladder, as the bar plot does — the two result types answer
    # "what are these sitting on?" the same way.
    _band_type = getattr(result, "band_type", None)
    ax.set_xlabel(_band_axis_label(_band_type) if _band_type
                  else "Frequency (Hz)")
    ax.set_ylabel(y_label)
    ax.set_xscale("log")
    ax.set_xlim(_log_freq_xlim(result.frequencies))
    # The levels drawn: every row holding any density, and the mean and
    # ±1 STD lines.
    occupied = np.any(np.nan_to_num(np.asarray(result.pdf, dtype=float)) > 0,
                      axis=1)
    rows = (result.level_edges[:-1] + align)[occupied]
    ax.set_ylim(_level_window_dB(
        np.concatenate([np.ravel(rows), np.ravel(result.mean_dB),
                        np.ravel(result.mean_dB + result.std_dB),
                        np.ravel(result.mean_dB - result.std_dB)]),
        ymin, ymax, held))
    # The level axis carries the histogram, not the density: level_edges spans
    # every row the mesh draws.
    _warn_if_offscreen(ax, result.level_edges, who, "ymin=/ymax")
    ax.grid(which="both", alpha=0.5)
    ax.legend(loc="upper right")
    return fig, ax


@typed_plot_error
def plot_ppsd(result, ax=None, *, ymin=None, ymax=None, vmin=0, vmax=None,
              cmap="jet", title=None, figsize=(10, 6), show_colorbar=True,
              **mpl_kw):
    """2-D histogram of Welch spectral levels. Consumes a
    :class:`~uacpy.acoustic_signal.ProbabilisticSpectralEstimate` computed
    with ``method='welch'``.

    The level axis is named from the reference the result carries, so a caller
    who ran the estimator against a Pascal reference is not handed a µPa axis
    120 dB out. An unset ``ymin`` / ``ymax`` is fitted to the levels drawn, as
    :func:`plot_psd` states.

    Parameters
    ----------
    result : ProbabilisticSpectralEstimate
        A Welch or sound-exposure histogram.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    ymin, ymax : float, optional
        Level-axis limits (dB); an edge left ``None`` is fitted to the levels
        drawn (see above).
    vmin, vmax : float, optional
        Colour limits of the probability density; ``vmax=None`` is
        ``1/level_step_dB``, the largest attainable. Default ``vmin`` 0.
    cmap : str, optional
        Colormap. Default ``'jet'``.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(10, 6)`` by default; unused when
        ``ax`` is given.
    show_colorbar : bool, optional
        Draw the colorbar. Default True.
    **mpl_kw
        Matplotlib keywords for the image.
    """
    if not hasattr(result, 'frequencies') or not hasattr(result, 'level_edges'):
        raise ConfigurationError(
            f"plot_ppsd: expected a probabilistic_welch() or "
            f"probabilistic_sound_exposure() result (with .frequencies and "
            f".level_edges); got {type(result).__name__}.")
    # Both histogram estimators return the one type, so the method is what
    # separates them: constant-Q bins are geometric and carry no
    # ``segment_duration``, which this plotter's linear axis and title state.
    if getattr(result, 'method', 'welch') == 'constant_q':
        raise ConfigurationError(
            "plot_ppsd: this estimate was computed with method='constant_q', "
            "whose bins are geometric and carry no segment duration.",
            remediation="Use plot_constant_q_ppsd(result), or result.plot(), "
                        "which picks the plotter from the estimate's method.")
    # The estimator takes ``ref=`` and carries it on the result, so the axis
    # names
    # the caller's reference rather than assuming the package default; a
    # hardcoded "µPa²" would be 120 dB out for anyone working in Pa. A dB axis
    # without a reference is an incomplete unit on a published figure.
    # It also takes ``scaling=``, and the result carries which was used:
    # 'density' levels are per hertz, 'spectrum' levels are per band, so the
    # axis cannot claim /Hz over a spectrum.
    ref = getattr(result, 'ref', REFERENCE_PRESSURE_WATER)
    unit = _scaling_unit("plot_ppsd", getattr(result, 'scaling', 'density'))
    return _plot_level_histogram(
        result, ax, y_label=f"Level (dB re {_ref_label(ref)}{unit})",
        default_title=_histogram_title(result), who="plot_ppsd",
        ymin=ymin, ymax=ymax, vmin=vmin, vmax=vmax, cmap=cmap, title=title,
        figsize=figsize, show_colorbar=show_colorbar, **mpl_kw)


@typed_plot_error
def plot_sel(sel_pa2s, bands=None, ax=None, *, ref=REFERENCE_PRESSURE_WATER,
             duration=None, band_type="decidecade", ymin=None, ymax=None,
             title=None, figsize=(10, 6), **mpl_kw):
    """Bar plot of standard-band levels (dB). Consumes a banded
    :class:`~uacpy.acoustic_signal.SpectralEstimate` — one computed with a
    ``band_type`` — or the same numbers as ``(values, bands)`` arrays.

    The unit comes from the estimate's ``scaling``: an exposure is per band
    and per second of record (Pa²·s), a spectrum per band (Pa²) and a density
    per hertz of the band's own width (Pa²/Hz). A bar chart labelled "·s" over
    band power would misstate the quantity by the record length. An unset ``ymin`` / ``ymax`` is fitted to the levels drawn, as
    :func:`plot_psd` states.

    Parameters
    ----------
    sel_pa2s : SpectralEstimate or array_like
        A banded estimate, or its values in linear units.
    bands : sequence of (low, centre, high), optional
        Band edges (Hz), with array input.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    ref : float, optional
        Reference pressure (Pa) of the dB scale. Default 1 µPa.
    duration : float, optional
        Record duration (s), named in the default title.
    band_type : str, optional
        The band ladder; an estimate passes its own. Default
        ``'decidecade'``.
    ymin, ymax : float, optional
        Level-axis limits (dB); an edge left ``None`` is fitted to the levels
        drawn (see above).
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(10, 6)`` by default; unused when
        ``ax`` is given.
    **mpl_kw
        Matplotlib keywords for the bars.
    """
    scaling = getattr(sel_pa2s, 'scaling', 'exposure')
    if hasattr(sel_pa2s, 'power') and hasattr(sel_pa2s, 'bands'):
        if sel_pa2s.bands is None:
            raise ConfigurationError(
                f"plot_sel: this estimate was asked for no band_type, so its "
                f"values sit on {sel_pa2s.method!r} bins and there are no "
                f"band edges to draw bars between.",
                remediation="Use plot_psd(result), or result.plot(), which "
                            "picks the plotter from the estimate itself; or "
                            "compute it with sound_exposure(), whose values "
                            "sit on bands.")
        # The ladder decides the frequency axis: every standard ladder is
        # geometric and reads on a log axis, while 'linear' does not.
        band_type = getattr(sel_pa2s, 'band_type', None) or band_type
        sel_pa2s, bands = sel_pa2s.power, sel_pa2s.bands
    elif bands is None:
        raise ConfigurationError(
            "plot_sel: pass a banded estimate, or values and the bands "
            "they sit on.",
            remediation="plot_sel(sound_exposure(x, fs)), or "
                        "plot_sel(values, bands).")
    fig, ax = fig_ax(ax, figsize)
    held = _held_window(ax)
    # Bands are (low, centre, high) triples; the contiguous edge vector is
    # every low edge plus the top edge of the last band.
    Fedges = [low for low, _, _ in bands] + [bands[-1][2]]
    width = [Fedges[i + 1] - Fedges[i] for i in range(len(Fedges) - 1)]
    sel_dB = power_to_dB(np.asarray(sel_pa2s), ref)
    ax.bar(Fedges[:-1], sel_dB, width=width,
           align="edge", edgecolor="black", **mpl_kw)
    unit = _scaling_unit("plot_sel", scaling)
    default = {"exposure": "SEL", "spectrum": "Band power",
               "density": "Band density"}[scaling]
    # ``duration`` is the caller's own note about the record; without it the
    # title states the quantity alone rather than "(Nones)".
    ax.set_title(_title_or(title, f"{default} ({duration}s)"
                           if duration is not None else default))
    ax.set_ylabel(f"Level (dB re {_ref_label(ref)}{unit})")
    if band_type != "linear":
        ax.set_xscale("log")
    ax.set_xlabel(_band_axis_label(band_type))
    ax.set_ylim(_level_window_dB(sel_dB, ymin, ymax, held))
    _warn_if_offscreen(ax, sel_dB, "plot_sel", "ymin=/ymax")
    ax.grid(which="both", alpha=0.75)
    ax.set_axisbelow(True)
    return fig, ax


# ── Time-frequency (timefreq) ───────────────────────────────────────────────

#: Depth (dB) below its loudest level that a spectrogram's default colour
#: window spans: a chirp and the noise under it both stay on the scale.
SPECTROGRAM_DYNAMIC_RANGE_DB = 60.0


def _level_window(levels_dB, vmin, vmax, dynamic_range_dB):
    """``(vmin, vmax)``: ``vmax=None`` is the largest finite level,
    ``vmin=None`` is ``vmax - dynamic_range_dB``."""
    if vmax is None:
        finite = np.asarray(levels_dB, dtype=float)
        finite = finite[np.isfinite(finite)]
        vmax = float(finite.max()) if finite.size else 0.0
    if vmin is None:
        vmin = vmax - float(dynamic_range_dB)
    return vmin, vmax


@typed_plot_error
def plot_spectrogram(frequencies, times=None, Sxx=None, ax=None, *,
                     ref=REFERENCE_PRESSURE_WATER, freq_min=1, freq_max=None,
                     vmin=None, vmax=None,
                     dynamic_range_dB=SPECTROGRAM_DYNAMIC_RANGE_DB,
                     cmap="viridis", title=None, figsize=(10, 6),
                     show_colorbar=True, scaling=None, mode=None, **mpl_kw):
    """Spectrogram colormap (dB). Consumes :func:`spectrogram` output.

    ``scaling`` (``'density'`` Pa²/Hz or ``'spectrum'`` Pa²) and ``mode``
    say what the panel holds, and the colour bar is labelled from them; a
    :class:`~uacpy.acoustic_signal.SpectrogramResult` carries both, and bare
    arrays default to the ``spectrogram`` defaults (``'density'``,
    ``'psd'``). A ``'magnitude'`` or ``'complex'`` panel is squared into the
    same power; an ``'angle'`` / ``'phase'`` panel is not a level and is
    refused.

    ``freq_min`` / ``freq_max`` bound the frequency axis (Hz; the level window is
    ``vmin`` / ``vmax``, as on every image plotter) and are symmetric: ``None``
    on either end takes the record's own first / last bin, so ``freq_min=None``
    drops the 1 Hz clamp the default applies. A clamp that sits above the record's
    whole band would reverse the axis, so such a band starts at its own first
    positive bin instead.

    Parameters
    ----------
    frequencies : SpectrogramResult or array_like
        The result itself, or its frequency axis (Hz).
    times : array_like, optional
        Frame times (s), with array input.
    Sxx : array_like, optional
        The ``(n_frequencies, n_times)`` panel, with array input.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    ref : float, optional
        Reference pressure (Pa) of the dB scale. Default 1 µPa.
    freq_min, freq_max : float, optional
        Frequency-axis bounds (Hz; see above). Default 1 Hz and the last
        bin.
    vmin, vmax : float, optional
        Colour limits (dB). ``vmax=None`` (default) is the panel's largest
        finite level; ``vmin=None`` (default) is ``vmax - dynamic_range_dB``,
        so the window follows the record rather than a fixed 0-200 dB.
    dynamic_range_dB : float, optional
        Depth (dB) of the default window below ``vmax``. Default 60.
    cmap : str, optional
        Colormap. Default ``'viridis'`` (perceptually uniform).
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(10, 6)`` by default; unused when
        ``ax`` is given.
    show_colorbar : bool, optional
        Draw the colorbar. Default True.
    scaling : {'density', 'spectrum'}, optional
        What the panel holds (see above).
    mode : str, optional
        The spectrogram mode (see above).
    **mpl_kw
        Matplotlib keywords for the image.
    """
    # One SpectrogramResult in place of the arrays — the same call
    # ``SpectrogramResult.plot()`` makes, so the two spellings agree.
    # Recognised only when the other positional arguments are
    # None, so an explicit array call is never reinterpreted.
    if scaling is None:
        scaling = getattr(frequencies, 'scaling', 'density')
    if mode is None:
        mode = getattr(frequencies, 'mode', 'psd')
    _unpacked = _carrier_or_arrays(
        frequencies, (times, Sxx,), count=3, who="plot_spectrogram",
        carrier=(('SpectrogramResult',), ('frequencies', 'times', 'power')),
        fields=('frequencies', 'times', 'Sxx'))
    if _unpacked is not None:
        frequencies, times, Sxx = _unpacked
    if times is None or Sxx is None:
        raise ConfigurationError(
            "plot_spectrogram: pass every array, or one SpectrogramResult.")
    if mode in ('angle', 'phase'):
        raise ConfigurationError(
            f"plot_spectrogram: a mode={mode!r} panel holds a phase in "
            f"radians, not a level to draw in dB.",
            remediation="Draw it with ax.pcolormesh(times, frequencies, "
                        "power) directly.")
    if scaling not in ('density', 'spectrum'):
        raise ConfigurationError(
            f"plot_spectrogram: scaling must be 'density' or 'spectrum'; "
            f"got {scaling!r}.")
    power = np.asarray(Sxx)
    if mode in ('magnitude', 'complex'):
        power = np.abs(power) ** 2
    unit = "Pa²/Hz" if scaling == 'density' else "Pa²"
    Sxx_dB = _require_image_grid(power_to_dB(power, ref),
                                 len(frequencies), len(times),
                                 'plot_spectrogram', 'frequencies', 'times')
    vmin, vmax = _level_window(Sxx_dB, vmin, vmax, dynamic_range_dB)
    fig, ax = fig_ax(ax, figsize)
    pcm = ax.pcolormesh(times, frequencies, Sxx_dB, cmap=cmap, shading="auto",
                        vmin=vmin, vmax=vmax, **mpl_kw)
    if show_colorbar:
        fig.colorbar(pcm, ax=ax, label=f"Level (dB re {_ref_label(ref)}{unit})")
    ax.set_title(_title_or(title, "Spectrogram"))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    hi = float(frequencies[-1] if freq_max is None else freq_max)
    ax.set_ylim((float(frequencies[0]), hi) if freq_min is None
                else _clamped_freq_limits(frequencies, freq_min, hi))
    _warn_if_offscreen(ax, frequencies, "plot_spectrogram", "freq_min/freq_max")
    _warn_if_colour_saturated(Sxx_dB, vmin, vmax, "plot_spectrogram", "vmin/vmax")
    ax.grid(which="both", alpha=0.25, color="black")
    return fig, ax


# ── Constant-Q (Brown 1991) ─────────────────────────────────────────────────

@typed_plot_error
def plot_constant_q_transform(frequencies, coefficients=None, ax=None, *,
                              label=None, title=None, figsize=(10, 6),
                              **mpl_kw):
    """Line plot of one constant-Q frame's magnitude (log frequency). Consumes
    :func:`constant_q_transform` output ``(frequencies, coefficients)``.

    Linear amplitude, not dB, and deliberately: the coefficients are the
    analytic band amplitude ``A/2``, while the one-sided band power the rest of
    the family reports is ``2*|X_cq|**2``. Read levels off
    :func:`plot_constant_q_psd`, whose estimator applies that conversion; this
    panel shows the raw transform a single frame returns.

    Parameters
    ----------
    frequencies : CQTResult or array_like
        The result itself, or its bin centres (Hz).
    coefficients : array_like, optional
        One complex coefficient per bin, with array input.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    label : str, optional
        Legend label of the line.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(10, 6)`` by default; unused when
        ``ax`` is given.
    **mpl_kw
        Matplotlib keywords for the line.
    """
    # One CQTResult in place of the arrays — the same call
    # ``CQTResult.plot()`` makes, so the two spellings agree.
    _unpacked = _carrier_or_arrays(
        frequencies, (coefficients,), count=2, who="plot_constant_q_transform",
        carrier=(('CQTResult',), ('frequencies', 'coefficients')),
        fields=('frequencies', 'coefficients'))
    if _unpacked is not None:
        frequencies, coefficients = _unpacked
    if coefficients is None:
        raise ConfigurationError(
            "plot_constant_q_transform: pass every array, or one CQTResult.")
    magnitude = np.abs(np.asarray(coefficients))
    if magnitude.ndim != 1 or magnitude.size != len(frequencies):
        raise ConfigurationError(
            f"plot_constant_q_transform: coefficients has shape "
            f"{magnitude.shape}; expected one value per frequency "
            f"(len(frequencies)={len(frequencies)}) — pass "
            f"constant_q_transform()'s own output.")
    fig, ax = fig_ax(ax, figsize)
    ax.semilogx(frequencies, magnitude, label=label, **mpl_kw)
    ax.set_title(_title_or(title, "Constant-Q transform"))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|X_cq|")
    ax.set_xlim(_log_freq_xlim(frequencies))
    ax.grid(which="both", alpha=0.75)
    if label:
        ax.legend()
    return fig, ax


@typed_plot_error
def plot_constant_q_spectrogram(frequencies, times=None, power=None,
                                ax=None, *,
                                ref=REFERENCE_PRESSURE_WATER, scaling=None,
                                vmin=None, vmax=None,
                                dynamic_range_dB=SPECTROGRAM_DYNAMIC_RANGE_DB,
                                cmap="viridis", title=None,
                                figsize=(10, 6), show_colorbar=True, **mpl_kw):
    """Constant-Q spectrogram colormap (dB, log frequency). Consumes
    :func:`constant_q_spectrogram` output ``(frequencies, times, power)``, or
    the :class:`CQSpectrogramResult` itself. ``scaling`` sets the unit,
    ``Pa²/Hz`` (density) or ``Pa²`` (band power); ``None`` reads it off a
    result passed whole and is ``'density'`` for bare arrays — the
    estimator's own default — so pass the value used there with arrays.

    Parameters
    ----------
    frequencies : CQSpectrogramResult or array_like
        The result itself, or its bin centres (Hz).
    times : array_like, optional
        Frame times (s), with array input.
    power : array_like, optional
        The ``(n_frequencies, n_times)`` power, with array input.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    ref : float, optional
        Reference pressure (Pa) of the dB scale. Default 1 µPa.
    scaling : {'density', 'spectrum'}, optional
        The unit (see above).
    vmin, vmax : float, optional
        Colour limits (dB). ``vmax=None`` (default) is the panel's largest
        finite level; ``vmin=None`` (default) is ``vmax - dynamic_range_dB``,
        so the window follows the record rather than a fixed 0-200 dB.
    dynamic_range_dB : float, optional
        Depth (dB) of the default window below ``vmax``. Default 60.
    cmap : str, optional
        Colormap. Default ``'viridis'`` (perceptually uniform).
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(10, 6)`` by default; unused when
        ``ax`` is given.
    show_colorbar : bool, optional
        Draw the colorbar. Default True.
    **mpl_kw
        Matplotlib keywords for the image.
    """
    # One CQSpectrogramResult in place of the arrays — the same call
    # ``CQSpectrogramResult.plot()`` makes, so the two spellings agree.
    given = frequencies
    _unpacked = _carrier_or_arrays(
        frequencies, (times, power,), count=3, who="plot_constant_q_spectrogram",
        carrier=(('CQSpectrogramResult',), ('frequencies', 'times', 'power')),
        fields=('frequencies', 'times', 'power'))
    if _unpacked is not None:
        frequencies, times, power = _unpacked
    if scaling is None:
        # Read off the result when one was passed whole (the carrier guard
        # above has already refused anything else); bare arrays take the
        # estimator's own default.
        scaling = (getattr(given, 'scaling', 'density')
                   if _unpacked is not None else 'density')
    unit = f"{_ref_label(ref)}{_scaling_unit('plot_constant_q_spectrogram', scaling)}"
    if times is None or power is None:
        raise ConfigurationError(
            "plot_constant_q_spectrogram: pass every array, or one CQSpectrogramResult.")
    power_dB = _require_image_grid(power_to_dB(np.asarray(power), ref),
                                   len(frequencies), len(times),
                                   'plot_constant_q_spectrogram',
                                   'frequencies', 'times')
    vmin, vmax = _level_window(power_dB, vmin, vmax, dynamic_range_dB)
    fig, ax = fig_ax(ax, figsize)
    pcm = ax.pcolormesh(times, frequencies, power_dB, cmap=cmap, shading="auto",
                        vmin=vmin, vmax=vmax, **mpl_kw)
    if show_colorbar:
        fig.colorbar(pcm, ax=ax, label=f"Level (dB re {unit})")
    ax.set_title(_title_or(title, "Constant-Q spectrogram"))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_yscale("log")
    ax.set_ylim((frequencies[0], frequencies[-1]))
    _warn_if_colour_saturated(power_dB, vmin, vmax,
                              "plot_constant_q_spectrogram", "vmin/vmax")
    ax.grid(which="both", alpha=0.25, color="black")
    return fig, ax


@typed_plot_error
def plot_constant_q_psd(frequencies, power, ax=None, *,
                        ref=REFERENCE_PRESSURE_WATER, scaling="spectrum",
                        label=None, ymin=None, ymax=None, title=None,
                        figsize=(10, 6), **mpl_kw):
    """Line plot of constant-Q power (dB, log frequency). Consumes
    :func:`uacpy.acoustic_signal.constant_q` output ``(frequencies, power)``. Pass the same
    ``scaling`` used there: ``'spectrum'`` labels band power (``Pa²``),
    ``'density'`` labels PSD (``Pa²/Hz``). An unset ``ymin`` / ``ymax`` is fitted to the levels drawn, as
    :func:`plot_psd` states.

    Parameters
    ----------
    frequencies : array_like
        Bin centres (Hz).
    power : array_like
        Power per bin, linear.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    ref : float, optional
        Reference pressure (Pa) of the dB scale. Default 1 µPa.
    scaling : {'spectrum', 'density'}, optional
        The scaling the estimate used. Default ``'spectrum'``.
    label : str, optional
        Legend label of the line.
    ymin, ymax : float, optional
        Level-axis limits (dB); an edge left ``None`` is fitted to the levels
        drawn (see above).
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(10, 6)`` by default; unused when
        ``ax`` is given.
    **mpl_kw
        Matplotlib keywords for the line.
    """
    unit = f"{_ref_label(ref)}{_scaling_unit('plot_constant_q_psd', scaling)}"
    power_dB = power_to_dB(np.asarray(power), ref)
    fig, ax = fig_ax(ax, figsize)
    held = _held_window(ax)
    ax.semilogx(frequencies, power_dB, label=label, **mpl_kw)
    ax.set_title(_title_or(title, "Constant-Q PSD" if scaling == "density"
                           else "Constant-Q band power"))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"Level (dB re {unit})")
    ax.set_ylim(_level_window_dB(power_dB, ymin, ymax, held))
    ax.set_xlim(_log_freq_xlim(frequencies))
    _warn_if_offscreen(ax, power_dB, "plot_constant_q_psd", "ymin=/ymax")
    ax.grid(which="both", alpha=0.75)
    if label:
        ax.legend()
    return fig, ax


@typed_plot_error
def plot_constant_q_ppsd(result, ax=None, *, scaling="spectrum", ymin=None,
                         ymax=None, vmin=0, vmax=None, cmap="jet", title=None,
                         figsize=(10, 6), show_colorbar=True, **mpl_kw):
    """2-D histogram of constant-Q power levels. Consumes a
    :class:`~uacpy.acoustic_signal.ProbabilisticSpectralEstimate` computed
    with ``method='constant_q'``.

    The dB reference is fixed at compute time by
    :func:`uacpy.acoustic_signal.probabilistic_welch` (default
    1 µPa) and read off the result, so the level axis names whatever reference
    was used; ``scaling`` is read off it too, and the keyword stands in only
    for a hand-built result that carries none. An unset ``ymin`` / ``ymax`` is fitted to the levels drawn, as
    :func:`plot_psd` states.

    Parameters
    ----------
    result : ProbabilisticSpectralEstimate
        A constant-Q histogram.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    scaling : {'spectrum', 'density'}, optional
        The unit, for a result that carries none. Default ``'spectrum'``.
    ymin, ymax : float, optional
        Level-axis limits (dB); an edge left ``None`` is fitted to the levels
        drawn (see above).
    vmin, vmax : float, optional
        Colour limits of the probability density; ``vmax=None`` is
        ``1/level_step_dB``, the largest attainable. Default ``vmin`` 0.
    cmap : str, optional
        Colormap. Default ``'jet'``.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(10, 6)`` by default; unused when
        ``ax`` is given.
    show_colorbar : bool, optional
        Draw the colorbar. Default True.
    **mpl_kw
        Matplotlib keywords for the image.
    """
    # From the result, not assumed: a hardcoded "µPa²" would be 120 dB out for a
    # caller who computed against a Pascal reference.
    ref = getattr(result, 'ref', REFERENCE_PRESSURE_WATER)
    # The result carries the scaling it was computed with; the keyword
    # stays as an override for a hand-built result that has none.
    scaling = getattr(result, 'scaling', None) or scaling
    unit = f"{_ref_label(ref)}{_scaling_unit('plot_constant_q_ppsd', scaling)}"
    return _plot_level_histogram(
        result, ax, y_label=f"Level (dB re {unit})",
        default_title=f"Constant-Q {_histogram_title(result)}",
        who="plot_constant_q_ppsd",
        ymin=ymin, ymax=ymax, vmin=vmin, vmax=vmax, cmap=cmap, title=title,
        figsize=figsize, show_colorbar=show_colorbar, **mpl_kw)


@typed_plot_error
def plot_cwt(frequencies, W=None, sample_rate=None, ax=None, *, cmap="viridis",
             title=None, figsize=(10, 6), show_colorbar=True, **mpl_kw):
    """Scalogram ``|W|`` (time on x, frequency on y). Consumes :func:`cwt`
    output.

    Takes either a whole :class:`~uacpy.acoustic_signal.CWTResult`,
    ``plot_cwt(result)`` — the call ``CWTResult.plot()`` makes, drawn on the
    time axis the result carries — or the arrays plus a rate,
    ``plot_cwt(f, W, fs)``, whose time axis is ``arange(n) / fs``.

    Parameters
    ----------
    frequencies : CWTResult or array_like
        The result itself, or its analysis frequencies (Hz).
    W : array_like, optional
        The ``(n_frequencies, n_samples)`` coefficients, with array input.
    sample_rate : float, optional
        Sample rate (Hz) of the time axis, with array input.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    cmap : str, optional
        Colormap. Default ``'viridis'`` (perceptually uniform).
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(10, 6)`` by default; unused when
        ``ax`` is given.
    show_colorbar : bool, optional
        Draw the colorbar. Default True.
    **mpl_kw
        Matplotlib keywords for the image.
    """
    _unpacked = _carrier_or_arrays(
        frequencies, (W,), count=3, who='plot_cwt',
        carrier=(('CWTResult',), ('frequencies', 'times', 'coefficients')),
        fields=('frequencies', 'times', 'coefficients'))
    if _unpacked is not None:
        if sample_rate is not None:
            raise ConfigurationError(
                "plot_cwt: a CWTResult carries its own time axis (.times); "
                "sample_rate= belongs to the arrays form plot_cwt(f, W, fs).")
        frequencies, t, W = _unpacked
    else:
        if W is None or sample_rate is None:
            raise ConfigurationError(
                "plot_cwt: pass one CWTResult, or (frequencies, W, "
                "sample_rate).")
        t = None
    amp = np.abs(np.asarray(W))
    if amp.ndim != 2 or amp.shape[0] != len(frequencies):
        raise ConfigurationError(
            f"plot_cwt: W has shape {amp.shape}; expected (len(frequencies)="
            f"{len(frequencies)}, n_samples) — pass cwt()'s own output.")
    if t is None:
        t = np.arange(amp.shape[1]) / float(sample_rate)
    fig, ax = fig_ax(ax, figsize)
    pcm = ax.pcolormesh(t, frequencies, amp, cmap=cmap, shading="auto", **mpl_kw)
    if show_colorbar:
        fig.colorbar(pcm, ax=ax, label="|W|")
    ax.set_title(_title_or(title, "CWT scalogram"))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    return fig, ax


@typed_plot_error
def plot_wigner_ville(frequencies, times=None, W=None, ax=None, *,
                      cmap="viridis", title=None,
                      figsize=(10, 6), show_colorbar=True, **mpl_kw):
    """Wigner-Ville distribution image. Consumes :func:`wigner_ville` output
    ``(frequencies, times, W)``.

    Parameters
    ----------
    frequencies : WignerVilleResult or array_like
        The result itself, or its frequency axis (Hz).
    times : array_like, optional
        Times (s), with array input.
    W : array_like, optional
        The ``(n_frequencies, n_times)`` distribution, with array input.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    cmap : str, optional
        Colormap. Default ``'viridis'`` (perceptually uniform).
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(10, 6)`` by default; unused when
        ``ax`` is given.
    show_colorbar : bool, optional
        Draw the colorbar. Default True.
    **mpl_kw
        Matplotlib keywords for the image.
    """
    # One WignerVilleResult in place of the arrays — the same call
    # ``WignerVilleResult.plot()`` makes, so the two spellings agree.
    # Recognised only when the other positional arguments are
    # None, so an explicit array call is never reinterpreted.
    _unpacked = _carrier_or_arrays(
        frequencies, (times, W,), count=3, who="plot_wigner_ville",
        carrier=(('WignerVilleResult',), ('frequencies', 'times', 'distribution')),
        fields=('frequencies', 'times', 'W'))
    if _unpacked is not None:
        frequencies, times, W = _unpacked
    if times is None or W is None:
        raise ConfigurationError(
            "plot_wigner_ville: pass every array, or one WignerVilleResult.")
    W_real = _require_image_grid(np.real(np.asarray(W)), len(frequencies),
                                 len(times), 'plot_wigner_ville',
                                 'frequencies', 'times')
    fig, ax = fig_ax(ax, figsize)
    pcm = ax.pcolormesh(times, frequencies, W_real, cmap=cmap,
                        shading="auto", **mpl_kw)
    if show_colorbar:
        fig.colorbar(pcm, ax=ax, label="WVD")
    ax.set_title(_title_or(title, "Wigner-Ville"))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    return fig, ax


@typed_plot_error
def plot_cepstrum(c, ax=None, *, sample_rate=None, title=None, figsize=(9, 4),
                  **mpl_kw):
    """Line plot of a cepstrum vs quefrency.

    Takes a whole :class:`~uacpy.acoustic_signal.Cepstrum` (what
    :func:`cepstrum` returns, drawn on the quefrency axis it carries: the
    call ``Cepstrum.plot()`` makes), a whole
    :class:`~uacpy.acoustic_signal.ComplexCepstrum` (the call
    ``ComplexCepstrum.plot()`` makes), or a bare cepstrum array, drawn in
    seconds when ``sample_rate`` is given and in samples otherwise. Both
    carriers are two wide, so ``plot_cepstrum(*result)`` would land the
    second field in ``ax=`` and is refused by name rather than failing
    inside matplotlib.

    Parameters
    ----------
    c : Cepstrum, ComplexCepstrum or array_like
        The cepstrum (see above).
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    sample_rate : float, optional
        Sample rate (Hz) for a bare array's quefrency axis in seconds.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(9, 4)`` by default; unused when
        ``ax`` is given.
    **mpl_kw
        Matplotlib keywords for the line.
    """
    _refuse_spread_carrier(
        ax, 'plot_cepstrum', 'second field',
        also='sample_rate=, which is keyword-only: plot_cepstrum(c, sample_rate=...)')
    quefrencies = None
    if (isinstance(c, tuple) and hasattr(c, '_fields')
            and 'Cepstrum' in {k.__name__ for k in type(c).__mro__}):
        if sample_rate is not None:
            raise ConfigurationError(
                "plot_cepstrum: a Cepstrum carries its own quefrency axis "
                "(.quefrencies); sample_rate= belongs to a bare cepstrum "
                "array.")
        seconds = c.sample_rate is not None
        quefrencies, c = c.quefrencies, c.cepstrum
    else:
        _unpacked = _carrier_or_arrays(
            c, (), count=1, who='plot_cepstrum',
            carrier=(('ComplexCepstrum',), ('cepstrum',)),
            fields=('cepstrum',))
        if _unpacked is not None:
            (c,) = _unpacked
    _require_nonempty('plot_cepstrum', c=c)
    c = np.real(np.asarray(c))
    fig, ax = fig_ax(ax, figsize)
    if quefrencies is not None:
        ax.plot(quefrencies, c, **mpl_kw)
        ax.set_xlabel("Quefrency (s)" if seconds
                      else "Quefrency (samples)")
    elif sample_rate is not None:
        q = np.arange(c.size) / float(sample_rate)
        ax.plot(q, c, **mpl_kw)
        ax.set_xlabel("Quefrency (s)")
    else:
        ax.plot(c, **mpl_kw)
        ax.set_xlabel("Quefrency (samples)")
    ax.set_ylabel("Amplitude")
    ax.set_title(_title_or(title, "Cepstrum"))
    ax.grid(alpha=0.3)
    return fig, ax


# ── Decidecade bands / array spectra / ambiguity ────────────────────────────

@typed_plot_error
def plot_band_levels(centers, levels, ax=None, *, title=None, width=0.8,
                     ref=REFERENCE_PRESSURE_WATER, band_type="decidecade",
                     figsize=(9, 4), **mpl_kw):
    """Bar plot of standard band levels vs centre frequency. Consumes a
    :class:`~uacpy.acoustic_signal.BandLevels` (its ``.plot()``) or its
    arrays: levels already in dB re ``ref²``.

    ``band_type`` names the ladder the bands sit on (``'decidecade'`` or
    ``'octave'``) on the axis and in the title; a
    ``BandLevels`` passes its own.

    ``ref`` is the reference pressure in Pa the levels were computed
    against — the ``ref=`` passed to :func:`decidecade_band_levels`, whose
    default this shares — and names the axis, as ``ref=`` does on every
    level plotter in this family.

    For band power still in Pa² (a banded
    :class:`~uacpy.acoustic_signal.SpectralEstimate`), use
    :func:`plot_sel`, which converts it and draws each bar between its band
    edges; this one draws levels that are already decibels.

    Parameters
    ----------
    centers : array_like
        Band centre frequencies (Hz).
    levels : array_like
        Band levels (dB re ``ref``²).
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    width : float, optional
        Bar width as a fraction of the band spacing. Default 0.8.
    ref : float, optional
        Reference pressure (Pa) of the dB scale. Default 1 µPa.
    band_type : {'decidecade', 'octave'}, optional
        The band ladder. Default ``'decidecade'``.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(9, 4)`` by default; unused when
        ``ax`` is given.
    **mpl_kw
        Matplotlib keywords for the bars.
    """
    _require_nonempty('plot_band_levels', centers=centers, levels=levels)
    c = np.asarray(centers, dtype=float)
    lv = np.asarray(levels, dtype=float)
    fig, ax = fig_ax(ax, figsize)
    # Bars are drawn against log10(f) on a LINEAR axis, with the ticks relabelled
    # back to Hz below: decidecade bands are equal-width in log10(f), so every
    # bar comes out the same width and none collapses at the low end (a true log
    # axis would squash them).
    x = np.log10(c)
    bw = width * np.median(np.diff(x)) if c.size > 1 else 0.04
    ax.bar(x, lv, width=bw, **mpl_kw)
    ticks = x[:: max(1, c.size // 12)]          # ~12 labelled bands, else unreadable
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v:.0f}" for v in 10 ** ticks], rotation=45)
    ladder = _BAND_NAMES.get(band_type, band_type)
    ax.set_xlabel(f"{ladder.capitalize()} band centre (Hz)")
    ax.set_ylabel(f"Band level (dB re {_ref_label(ref)}Pa²)")
    ax.set_title(_title_or(title, f"{ladder.capitalize()} band levels"))
    ax.grid(alpha=0.3, axis="y")
    return fig, ax


@typed_plot_error
def plot_angular_spectrum(angles_deg, spectrum, ax=None, *, dB=True, label=None,
                          title=None, figsize=(8, 4), **mpl_kw):
    """Line plot of a beamformer angular spectrum (Bartlett/MVDR/MUSIC).

    Parameters
    ----------
    angles_deg : array_like
        Look angles (deg).
    spectrum : array_like
        Beamformer power per angle.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    dB : bool, optional
        Draw ``10·log10`` relative to the peak. Default True.
    label : str, optional
        Legend label of the line.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(8, 4)`` by default; unused when
        ``ax`` is given.
    **mpl_kw
        Matplotlib keywords for the line.
    """
    P = np.real(np.asarray(spectrum))
    if dB:
        # Beamformer output has no absolute reference (MVDR/MUSIC pseudo-power
        # least of all), so the dB axis is relative to the peak: 0 dB = look
        # direction of maximum response.
        P = 10.0 * np.log10(P / np.max(P))
    fig, ax = fig_ax(ax, figsize)
    ax.plot(angles_deg, P, label=label, **mpl_kw)
    ax.set_xlabel("Look angle (°)")
    ax.set_ylabel("Power (dB re max)" if dB else "Power")
    ax.set_title(_title_or(title, "Angular spectrum"))
    ax.grid(alpha=0.3)
    if label:
        ax.legend()
    return fig, ax


@typed_plot_error
def plot_matched_field(field, ax=None, *, dynamic_range_dB=20.0,
                       cmap=None, true_position=None, mark_peak=True,
                       title=None, figsize=(8, 5), show_colorbar=True,
                       show_legend=True, **mpl_kw):
    """Matched-field ambiguity surface over its candidate grid, in dB re its
    peak.

    Consumes the ambiguity Field a processor returns —
    :func:`uacpy.sonar.bartlett` / :func:`~uacpy.sonar.mvdr` over a replica
    bank, :meth:`~uacpy.core.results.Covariance.bartlett` /
    :meth:`~uacpy.core.results.Covariance.mvdr` over OASN replicas, or
    :func:`~uacpy.core.results.ambiguity_field` of an array-level surface —
    and draws its candidate ``depth`` against its horizontal axis
    (``'range'`` or OASN's ``'x'``). ``true_position`` is an
    ``(horizontal_m, depth_m)`` pair, marked with the package's source star
    for comparison, and ``**mpl_kw`` goes to the pcolormesh.

    Any other axis — a ``'frequency'`` of a multi-frequency covariance, an
    OASN ``'y'`` — must hold one value: choosing a frequency or a ``y``
    plane is the caller's decision (``field.at(frequency=…, y=…)``), and a
    plotter making it silently would draw one slice of a search under a
    title claiming the whole of it.

    A degenerate candidate position -- one the forward model put no energy at
    -- is part of the contract, not a broken surface: ``mvdr`` leaves it NaN
    and ``bartlett`` scores it ``-inf`` dB. Both are drawn, the NaN as the
    colormap's "bad" colour and the ``-inf`` on the floor.
    ``show_legend=False`` drops the marker key, for a grid of panels that
    needs it once.

    Parameters
    ----------
    field : Field
        An ambiguity Field (see above).
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    dynamic_range_dB : float, optional
        Decibels below the peak the colour scale spans. Default 20.
    cmap : str, optional
        Colormap; ``None`` is the registered ambiguity map.
    true_position : (float, float), optional
        ``(horizontal_m, depth_m)`` to mark.
    mark_peak : bool, optional
        Mark the surface's peak. Default True.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(8, 5)`` by default; unused when
        ``ax`` is given.
    show_colorbar : bool, optional
        Draw the colorbar. Default True.
    show_legend : bool, optional
        Draw the marker key. Default True.
    **mpl_kw
        Matplotlib keywords for the pcolormesh.
    """
    span_dB = _checked_dynamic_range_dB('plot_matched_field', dynamic_range_dB)
    if getattr(field, 'kind', None) != 'ambiguity':
        raise ConfigurationError(
            f"plot_matched_field: expected an ambiguity Field (sonar.bartlett "
            f"/ sonar.mvdr, Covariance.bartlett / .mvdr, or ambiguity_field); "
            f"got {type(field).__name__} of kind "
            f"{getattr(field, 'kind', None)!r}.")
    axes = [name for name in field.coords if field.coords[name].size > 1]
    horizontal = [name for name in axes if name in ('range', 'x')]
    if (sorted(axes) != sorted(['depth'] + horizontal)
            or len(horizontal) != 1):
        raise ConfigurationError(
            f"plot_matched_field: the surface spans {axes}, which is not one "
            f"(depth, range) or (depth, x) plane. Select the other axes first "
            f"-- e.g. field.at(frequency=..., y=...) -- before plotting.")
    x = np.asarray(field.coords[horizontal[0]], dtype=float)
    z = np.asarray(field.coords['depth'], dtype=float)
    # The dB values on (depth, horizontal), whatever order the Field holds
    # them in; the length-1 axes drop.
    order = [list(field.coords).index('depth'),
             list(field.coords).index(horizontal[0])]
    rest = [i for i in range(field.data.ndim) if i not in order]
    SdB = np.transpose(np.asarray(field.data, dtype=float),
                       order + rest).reshape(z.size, x.size)
    # A matched-field processor carries no absolute reference -- MVDR's
    # pseudo-power least of all -- so 0 dB is the best-matching candidate
    # position, not a level. Clipped to the floor so bartlett's -inf (a
    # candidate with zero power) draws on it.
    SdB = np.clip(SdB, -span_dB, None)
    # One colormap for one quantity: both ambiguity surfaces ask the
    # style registry rather than naming a literal, so they cannot drift
    # apart from each other or from ``plot_field(kind='ambiguity')``.
    if cmap is None:
        cmap = cmap_for_field('ambiguity', dB=True)
    fig, ax = fig_ax(ax, figsize)
    im = ax.pcolormesh(x / 1000.0, z, SdB, cmap=cmap, vmin=-span_dB,
                       vmax=0.0, shading='auto', **mpl_kw)
    if true_position is not None:
        # A known source position is a source: same red star every other uacpy
        # plot marks one with, so it reads the same across the package.
        tx, tz = true_position
        ax.plot(float(tx) / 1000.0, float(tz), zorder=ZORDER_SOURCE,
                label='true position', **SOURCE_MARKER_STYLE)
    if mark_peak:
        # The estimate, kept visually distinct from the truth marker: black
        # reads on the bright cell a peak sits in, on any of the sequential
        # colormaps this plot is used with. Drawn ABOVE the star and after it,
        # because the case worth reading is the one where they coincide -- a
        # star painted over the cross hid the estimate exactly when the
        # picture's point was that the estimate had landed.
        iz, ix = np.unravel_index(int(np.nanargmax(SdB)), SdB.shape)
        ax.plot(x[ix] / 1000.0, z[iz], '+', color='black', ms=13, mew=2.2,
                zorder=ZORDER_SOURCE + 1, label='peak')
    if show_colorbar:
        fig.colorbar(im, ax=ax, label='dB re peak')
    ax.set_xlabel('Candidate range (km)')
    ax.set_ylabel('Candidate depth (m)')
    ax.set_title(_title_or(title, 'Matched-field ambiguity surface'))
    # Depth downward, set as an explicit descending limit rather than
    # invert_yaxis(): a shared-y pair would call this once per axis and the
    # second call would undo the first.
    ax.set_ylim(float(np.max(z)), float(np.min(z)))
    if show_legend and (mark_peak or true_position is not None):
        # One row, not a block: a heatmap fills its axes, so 'best' has no
        # empty corner to find and any multi-row box lands on the surface --
        # on the deck it covered a third of it, directly over the peak.
        ax.legend(loc='upper center', ncol=2, fontsize='small',
                  framealpha=0.85).set_zorder(ZORDER_LEGEND)
    return fig, ax


#: Default reach of ``plot_ambiguity``'s dB view below its peak.
_AMBIGUITY_DYNAMIC_RANGE_DB = 40.0


@typed_plot_error
def plot_ambiguity(delays_s, doppler_hz=None, chi=None, ax=None, *,
                   dB=False,
                   dynamic_range_dB=None, cmap=None,
                   title=None, figsize=(8, 6), show_colorbar=True, **mpl_kw):
    """Range-Doppler ambiguity surface ``|chi|``. Consumes
    :func:`ambiguity_function` output. ``dB=True`` shows it relative to its
    peak over ``dynamic_range_dB`` decibels (40 when unset); without
    ``dB=True`` there is no dB axis to bound, and ``dynamic_range_dB=`` is
    refused.

    Parameters
    ----------
    delays_s : AmbiguityResult or array_like
        The result itself, or its delay axis (s).
    doppler_hz : array_like, optional
        Doppler shifts (Hz), with array input.
    chi : array_like, optional
        The ``(n_doppler, n_delays)`` surface, with array input.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    dB : bool, optional
        Draw relative to the peak in dB. Default False.
    dynamic_range_dB : float, optional
        Decibels the dB view spans; 40 when unset. Refused without ``dB``.
    cmap : str, optional
        Colormap; ``None`` is the registered ambiguity map.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(8, 6)`` by default; unused when
        ``ax`` is given.
    show_colorbar : bool, optional
        Draw the colorbar. Default True.
    **mpl_kw
        Matplotlib keywords for the image.
    """
    # One AmbiguityResult in place of the arrays — the same call
    # ``AmbiguityResult.plot()`` makes, so the two spellings agree.
    # Recognised only when the other positional arguments are
    # None, so an explicit array call is never reinterpreted.
    _unpacked = _carrier_or_arrays(
        delays_s, (doppler_hz, chi,), count=3, who="plot_ambiguity",
        carrier=(('AmbiguityResult',), ('delays_s', 'doppler_hz', 'amplitude')),
        fields=('delays_s', 'doppler_hz', 'chi'))
    if _unpacked is not None:
        delays_s, doppler_hz, chi = _unpacked
    if doppler_hz is None or chi is None:
        raise ConfigurationError(
            "plot_ambiguity: pass every array, or one AmbiguityResult.")
    if dynamic_range_dB is not None and not dB:
        raise ConfigurationError(
            f"plot_ambiguity: dynamic_range_dB={dynamic_range_dB!r} bounds the "
            f"dB view; the linear |chi| panel has nothing to bound. Pass "
            f"dB=True, or drop dynamic_range_dB=.")
    span_dB = (_AMBIGUITY_DYNAMIC_RANGE_DB if dynamic_range_dB is None
               else _checked_dynamic_range_dB('plot_ambiguity',
                                              dynamic_range_dB))
    amp = _require_image_grid(np.abs(np.asarray(chi)), len(doppler_hz),
                              len(delays_s), "plot_ambiguity",
                              "doppler_hz", "delays_s")
    label = "|χ|"
    if dB:
        # The sidelobe structure IS the reason to draw an ambiguity surface,
        # and it sits tens of dB down, where a linear |chi| is uniformly black.
        peak = float(np.max(amp))
        if not np.isfinite(peak) or peak <= 0.0:
            raise ConfigurationError(
                f"plot_ambiguity: the surface peaks at {peak}, so a dB scale "
                f"relative to it is undefined.")
        floor = 10.0 ** (-span_dB / 20.0)
        amp = 20.0 * np.log10(np.maximum(amp / peak, floor))
        mpl_kw.setdefault("vmin", -span_dB)
        mpl_kw.setdefault("vmax", 0.0)
        label = "|χ| (dB re peak)"
    # One colormap for one quantity: both ambiguity surfaces ask the style
    # registry rather than naming a literal, so they cannot drift from each
    # other or from ``plot_field(kind='ambiguity')``. style.py states why the
    # map is perceptually ordered rather than diverging.
    #
    # ``dB=True`` on both paths, including the linear default, and that is
    # not a description of the view: |chi| is a magnitude, unsigned at either
    # scaling, so it wants the quantity's own ordered map both times.
    # cmap_for_field's dB=False arm returns LINEAR_VIEW_COLORMAP, the signed
    # diverging map every linear *Field* view takes, which would spend half
    # its range on negative values this surface cannot hold. Nothing here
    # varies with ``dB``; the flag selects the map, not the axis.
    if cmap is None:
        cmap = cmap_for_field('ambiguity', dB=True)
    fig, ax = fig_ax(ax, figsize)
    im = ax.imshow(amp, aspect="auto", origin="lower",
                   extent=_cell_edge_extent(np.asarray(delays_s) * 1e3,
                                            doppler_hz),
                   cmap=cmap, **mpl_kw)
    if show_colorbar:
        fig.colorbar(im, ax=ax, label=label)
    ax.set_title(_title_or(title, "Ambiguity surface"))
    ax.set_xlabel("Delay (ms)")
    ax.set_ylabel("Doppler (Hz)")
    return fig, ax


# ── System identification (FRF) ─────────────────────────────────────────────

@typed_plot_error
def plot_frf(frequencies, tf, ax=None, *, tag="", label=None, ymin=-60,
             ymax=60, title=None, figsize=(10, 12), **mpl_kw):
    """Transfer-function magnitude (dB) + phase (deg). Consumes an
    ``FRFResult``'s ``(frequencies, transfer_function)``, which
    ``FRFResult.plot`` hands it. ``ax`` may be a 2-tuple
    ``(ax_mag, ax_phase)``.

    Parameters
    ----------
    frequencies : array_like
        Frequencies (Hz).
    tf : array_like
        The complex transfer function.
    ax : (Axes, Axes), optional
        The ``(ax_mag, ax_phase)`` pair; a new two-panel figure when
        omitted.
    tag : str, optional
        Prefix of the legend label.
    label : str, optional
        Legend label of the line.
    ymin, ymax : float, optional
        Magnitude-axis limits (dB). Default -60 and 60.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(10, 12)`` by default; unused when
        ``ax`` is given.
    **mpl_kw
        Matplotlib keywords for both lines.
    """
    if ax is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    else:
        ax1, ax2 = _axes_pair("plot_frf", ax, "(ax_mag, ax_phase)")
        fig = ax1.figure
    lbl = (f"{tag} {label}").strip() if (tag or label) else None
    mag_dB = 20 * np.log10(np.abs(tf))
    ax1.plot(frequencies, mag_dB, label=lbl, **mpl_kw)
    ax1.set_title(_title_or(title, "Frequency response"))
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_xscale("log")
    ax1.set_ylim((ymin, ymax))
    ax1.set_xlim(_log_freq_xlim(frequencies))
    # Only the magnitude panel is pinned to a fixed window; the phase axis
    # below spans the full ±180° a phase can occupy.
    _warn_if_offscreen(ax1, mag_dB, "plot_frf", "ymin=/ymax")
    ax1.grid(which="both", alpha=0.5)
    ax2.plot(frequencies, np.angle(tf, deg=True), label=lbl, **mpl_kw)
    ax2.set_ylabel("Phase (degrees)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_xscale("log")
    ax2.set_ylim((-180, 180))
    ax2.set_xlim(_log_freq_xlim(frequencies))
    ax2.grid(which="both", alpha=0.5)
    if lbl:
        ax1.legend()
        ax2.legend()
    return fig, (ax1, ax2)


@typed_plot_error
def plot_coherence(frequencies, coh, ax=None, *, label=None, title=None,
                   ymin=0.75, ymax=1.01, figsize=(10, 4), **mpl_kw):
    """Coherence vs frequency. Consumes a ``frf_welch`` result's
    ``frequencies`` and ``coherence``.

    ``ymin`` / ``ymax`` default to the near-unity window a well-conditioned
    FRF lives in; widen it (``ymin=0``) to see a poorly coherent band, which
    would otherwise fall entirely below the default axes.

    Parameters
    ----------
    frequencies : array_like
        Frequencies (Hz).
    coh : array_like
        Coherence per frequency, 0-1.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    label : str, optional
        Legend label of the line.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    ymin, ymax : float, optional
        Axis limits (see above). Default 0.75 and 1.01.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(10, 4)`` by default; unused when
        ``ax`` is given.
    **mpl_kw
        Matplotlib keywords for the line.
    """
    fig, ax = fig_ax(ax, figsize)
    ax.plot(frequencies, coh, label=label, **mpl_kw)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Coherence")
    ax.set_xscale("log")
    ax.set_ylim((ymin, ymax))
    ax.set_xlim(_log_freq_xlim(frequencies))
    _warn_if_offscreen(ax, coh, "plot_coherence", "ymin=/ymax")
    ax.grid(which="both", alpha=0.5)
    ax.set_title(_title_or(title, "Coherence"))
    if label:
        ax.legend()
    return fig, ax


@typed_plot_error
def plot_lsfir_diagnostics(Minfo, Vinfo, g, *, title=None, figsize=(12, 8)):
    """LS-FIR diagnostics: an ``lsfir`` result's ``information_matrix``,
    ``information_vector`` and ``impulse_response``.

    Parameters
    ----------
    Minfo : array_like
        The ``information_matrix``.
    Vinfo : array_like
        The ``information_vector``.
    g : array_like
        The ``impulse_response``.
    title : str, optional
        Title of the matrix panel. ``None`` draws the default caption.
    figsize : tuple, optional
        Figure size (inches). Default ``(12, 8)``.
    """
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(Minfo, cmap="viridis", aspect="equal")
    ax1.set_title(_title_or(title, "Information Matrix"))
    ax1.set_xlabel("Index j")
    ax1.set_ylabel("Index i")
    fig.colorbar(im, ax=ax1, shrink=0.8, label="Correlation Value")
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(np.arange(len(Vinfo)), Vinfo, color="skyblue", edgecolor="navy")
    ax2.set_title("Information Vector")
    ax2.set_xlabel("Index i")
    ax2.set_ylabel("Cross-correlation Value")
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(g, color="red", linestyle="-", marker="o", markersize=4)
    ax3.set_title("Impulse Response")
    ax3.set_xlabel("Time Index")
    ax3.set_ylabel("Amplitude")
    ax3.grid(True)
    return fig, [ax1, ax2, ax3]
