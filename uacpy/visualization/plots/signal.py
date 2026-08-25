"""Plots for the signal-processing toolkits (acoustic_signal).

All DSP plotting lives here, not in the computation modules: ``acoustic_signal``
is pure computation and never imports matplotlib. Each function consumes the
plain arrays a transform/estimator returns (never a compute object), takes the
target ``ax`` as its second positional argument (a new figure is made when it is
``None``), and returns ``(fig, ax)`` — the same convention as :func:`plot_field`.
"""
import numpy as np
import matplotlib.pyplot as plt

from uacpy.core.constants import (REFERENCE_PRESSURE_AIR,
                                  REFERENCE_PRESSURE_WATER)
from uacpy.core.acoustics import power_to_db
from uacpy.core.exceptions import ConfigurationError
from uacpy.visualization.plots._common import (_cell_edge_extent, _flip_y,
                                               fig_ax, typed_plot_error,
                                               _plot_warn)


def _require_image_grid(arr, n0, n1, caller, name0, name1):
    """Guard an ``imshow`` panel against a coordinate/array mismatch.

    ``imshow`` only reads the first/last coordinate for its ``extent`` and
    stretches the whole array onto it, so a length mismatch yields a
    plausible-but-wrong figure with no error (unlike the pcolormesh plotters,
    which raise). Require the data array to be exactly ``(n0, n1)`` and raise a
    typed error naming the mismatch otherwise."""
    a = np.asarray(arr)
    if a.ndim != 2 or a.shape != (n0, n1):
        raise ConfigurationError(
            f"{caller}: data array shape {a.shape} does not match the "
            f"coordinate lengths ({name0}={n0}, {name1}={n1}). Pass the "
            f"transform's own output unmodified — a mismatch would silently "
            f"render a wrong image.")
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


def _warn_if_offscreen(ax, values, caller, knob):
    """Warn when a fixed y window excludes every finite sample.

    Several of these plotters pin the ordinate to the range their quantity
    normally occupies, which keeps panels comparable but renders a record
    outside that range as an empty panel — a silent, easily-missed failure that
    looks like "no data". Say so, and name the knob that widens the window."""
    outside = _all_outside(values, *ax.get_ylim())
    if outside is not None:
        v_min, v_max, lo, hi = outside
        _plot_warn(
            f"{caller}: every sample ({v_min:.4g} … {v_max:.4g}) lies "
            f"outside the plotted y range ({lo:g}, {hi:g}), so the panel is "
            f"empty. Pass {knob}= to widen it.")


def _warn_if_colour_saturated(values, vmin, vmax, caller, knob):
    """Warn when a fixed colour window excludes every finite sample.

    A pinned dB window keeps panels comparable, but a record entirely above or
    below it maps to one end of the colormap everywhere: a flat single-colour
    image that reads as a valid featureless record rather than as a window
    problem. Say so, and name the knobs that move the window."""
    outside = _all_outside(values, vmin, vmax)
    if outside is not None:
        v_min, v_max, lo, hi = outside
        _plot_warn(
            f"{caller}: every sample ({v_min:.4g} … {v_max:.4g} dB) lies "
            f"outside the colour window ({lo:g}, {hi:g}) dB, so the panel is a "
            f"single flat colour. Pass {knob}= to move it.")


def _ref_label(ref):
    if ref == REFERENCE_PRESSURE_WATER:
        return "1µ"
    if ref == REFERENCE_PRESSURE_AIR:
        return "20µ"
    return f"{ref:g}"




# ── f-k / Radon / tau-p gather transforms ───────────────────────────────────

def draw_sound_cone(ax, f_max, k_max, sound_speed, *, color="w", ls="--",
                    lw=1.1, alpha=0.85, label=True):
    """Overlay the acoustic cone ``f = c·k/2π`` onto an f-k axis whose abscissa
    is the angular wavenumber ``k`` (rad/m), matching :func:`fk_transform`."""
    c = float(sound_speed)
    two_pi = 2.0 * np.pi
    k = min(two_pi * f_max / c, k_max)   # cone reaches f_max or the axis edge
    f = c * k / two_pi
    ax.plot([0, k], [0, f], color=color, ls=ls, lw=lw, alpha=alpha)
    ax.plot([0, -k], [0, f], color=color, ls=ls, lw=lw, alpha=alpha)
    if label:
        ax.text(k, f, f" {c:.0f} m/s", color=color, fontsize=8,
                va="top", ha="right")


# The colour window autoscales, as it does on the other transform panels
# (plot_radon, plot_taup). A fixed -60..+20 dB window suits a PEAK-RELATIVE
# scale — which is what docs/figure_scripts/signal.py hand-rolls, at
# vmin=-40, vmax=0 — but the level here goes through
# power_to_db(power, ref), an ABSOLUTE dB re 1 uPa^2. Measured on the
# fk_transform output this function documents itself as consuming, for a 1 Pa
# plane-wave gather at fs = 2 kHz, dx = 2 m: the panel spans 107.3 .. 196.9 dB
# with a median of 122.2, so every pixel would sit above that vmax and the
# figure would come out a uniform block. A fixed absolute window cannot work here anyway: the
# transform sums over the gather, so the level moves with its size.
@typed_plot_error
def plot_fk(frequencies, wavenumbers, power, ax=None, *, ref=REFERENCE_PRESSURE_WATER,
            vmin=None, vmax=None, cmap=None, sound_speed=None, title=None,
            figsize=(10, 6), show_colorbar=True, **mpl_kw):
    """Image an f-k power panel (dB). Consumes :func:`fk_transform` output."""
    _require_image_grid(power, len(frequencies), len(wavenumbers),
                        "plot_fk", "frequencies", "wavenumbers")
    fk_db = power_to_db(np.asarray(power), ref)
    fig, ax = fig_ax(ax, figsize)
    # Edge-aligned: the axes are FFT bin centres, and draw_sound_cone below
    # places f = c*k/(2*pi) at true coordinates, so a half-bin shift would
    # offset the image against the very line used to read it.
    im = ax.imshow(fk_db, extent=_cell_edge_extent(wavenumbers, frequencies),
                   origin="lower", aspect="auto",
                   vmin=vmin, vmax=vmax, cmap=cmap, **mpl_kw)
    if sound_speed is not None:
        draw_sound_cone(ax, frequencies[-1], wavenumbers[-1], sound_speed)
    ax.set_title(title or "f–k spectrum", loc="left")
    ax.set_xlabel("Wavenumber k (rad/m)")
    ax.set_ylabel("Frequency (Hz)")
    ax.grid(alpha=0.3)
    if show_colorbar:
        fig.colorbar(im, ax=ax, label="Power (dB)")
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


@typed_plot_error
def plot_radon(moveout, taus, R, ax=None, *, kind="linear", vmin=None,
               vmax=None, cmap="jet", title=None, figsize=(8, 6),
               show_colorbar=True, **mpl_kw):
    """Image ``|R|`` (moveout on x, intercept time on y). Consumes
    :func:`radon_transform` output."""
    amp = _require_image_grid(np.abs(np.asarray(R)), len(moveout), len(taus),
                              "plot_radon", "moveout", "taus")
    xlabel, scale = _RADON_AXIS.get(kind, ("Moveout", 1.0))
    m = np.asarray(moveout) * scale
    vmax = amp.max() if vmax is None else vmax
    vmin = 0.0 if vmin is None else vmin
    fig, ax = fig_ax(ax, figsize)
    # ``amp`` is (moveout, tau); transposed it is (tau, moveout) = (row, col).
    # The extent puts the largest tau at the bottom, so intercept time runs
    # downward — the seismic gather convention.
    im = ax.imshow(amp.T, aspect="auto", origin="upper",
                   extent=_flip_y(_cell_edge_extent(m, taus)),
                   vmin=vmin, vmax=vmax, cmap=cmap, **mpl_kw)
    ax.set_title(title or f"Radon ({kind})", loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Intercept time tau (s)")
    if show_colorbar:
        fig.colorbar(im, ax=ax, label="Stack amplitude")
    return fig, ax


def draw_slowness_line(ax, tau_max, sound_speed, *, color="w", ls="--",
                       lw=1.1, alpha=0.85, label=True):
    """Mark slowness ``p = +/-1/c`` (s/km) of a reference speed on a tau-p axis."""
    p_skm = 1000.0 / float(sound_speed)          # 1/c in s/m → s/km
    for sgn in (-1.0, 1.0):
        ax.axvline(sgn * p_skm, color=color, ls=ls, lw=lw, alpha=alpha)
    if label:
        ax.text(p_skm, tau_max, f" {sound_speed:.0f} m/s", color=color,
                fontsize=8, va="bottom", ha="left")


@typed_plot_error
def plot_taup(slownesses, taus, taup, ax=None, *, vmin=None, vmax=None,
              cmap="jet", sound_speed=None, title=None, figsize=(8, 6),
              show_colorbar=True, **mpl_kw):
    """Image a tau-p panel (slowness s/km on x, intercept time on y). Consumes
    :func:`taup_transform` output."""
    p_skm = np.asarray(slownesses) * 1000.0      # taup_transform returns s/m
    amp = _require_image_grid(np.abs(np.asarray(taup)), len(slownesses),
                              len(taus), "plot_taup", "slownesses", "taus")
    vmax = amp.max() if vmax is None else vmax
    vmin = 0.0 if vmin is None else vmin
    fig, ax = fig_ax(ax, figsize)
    # Transposed to (tau, slowness) with tau increasing downward, as in
    # plot_radon.
    im = ax.imshow(amp.T, aspect="auto", origin="upper",
                   extent=_flip_y(_cell_edge_extent(p_skm, taus)),
                   vmin=vmin, vmax=vmax, cmap=cmap, **mpl_kw)
    if sound_speed is not None:
        draw_slowness_line(ax, taus[-1], sound_speed)
    ax.set_title(title or "tau-p", loc="left")
    ax.set_xlabel("Slowness p (s/km)")
    ax.set_ylabel("Intercept time tau (s)")
    if show_colorbar:
        fig.colorbar(im, ax=ax, label="Stack amplitude")
    return fig, ax


# ── Spectral / level estimators (analysis) ──────────────────────────────────

@typed_plot_error
def plot_psd(frequencies, psd_linear, ax=None, *, ref=REFERENCE_PRESSURE_WATER,
             label=None, ymin=0, ymax=150, title=None, figsize=(10, 6),
             **mpl_kw):
    """Line plot of a Welch PSD (dB). Consumes :func:`psd` output.

    ``ymin`` / ``ymax`` pin the level axis to the 0–150 dB window an ambient
    record occupies; a quieter one needs them widened or the panel comes out
    empty."""
    psd_db = power_to_db(np.asarray(psd_linear), ref)
    fig, ax = fig_ax(ax, figsize)
    ax.semilogx(frequencies, psd_db, label=label, **mpl_kw)
    ax.set_title(title or "Power spectral density", loc="left")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"Level (dB re {_ref_label(ref)}Pa²/Hz)")
    ax.set_ylim((ymin, ymax))
    ax.set_xlim(_log_freq_xlim(frequencies))
    _warn_if_offscreen(ax, psd_db, "plot_psd", "ymin=/ymax")
    ax.grid(which="both", alpha=0.75)
    if label:
        ax.legend()
    return fig, ax


@typed_plot_error
def plot_ppsd(result, ax=None, *, ymin=0, ymax=200, vmin=0, vmax=None,
              cmap="jet", title=None, figsize=(10, 6), show_colorbar=True,
              **mpl_kw):
    """2-D histogram of PSD levels. Consumes a ``PPSDResult``."""
    if vmax is None:
        # Each frequency column integrates to 1 over the level axis, so the
        # largest attainable density is 1/binwidth (all mass in one bin) —
        # the natural top of the colour scale.
        vmax = 1 / result.binwidth_db
    fig, ax = fig_ax(ax, figsize)
    # ``level_edges`` are bin EDGES and ``pdf`` has one row per bin; shift by
    # half a bin so each row is centred on its own level. Empty bins arrive as
    # NaN and render as the axes background.
    align = result.binwidth_db / 2
    pcm = ax.pcolormesh(result.frequencies, result.level_edges[:-1] + align,
                        result.pdf, cmap=cmap, shading="auto",
                        vmin=vmin, vmax=vmax, **mpl_kw)
    if show_colorbar:
        fig.colorbar(pcm, ax=ax,
                     label=f"Probability Density ({result.binwidth_db:.1f} dB/bin)")
    ax.plot(result.frequencies, result.mean_db, "k-", label="Mean level", lw=1.5)
    ax.plot(result.frequencies, result.mean_db + result.std_db, "k--",
            label="Mean level ± STD")
    ax.plot(result.frequencies, result.mean_db - result.std_db, "k--")
    ax.set_title(title or f"PPSD ({result.seg_duration}s)", loc="left")
    ax.set_xlabel("Frequency (Hz)")
    # ppsd fixes ref=1e-6 and defaults to scaling='density', so the axis
    # is dB re 1 uPa^2/Hz. The constant-Q twin already names its reference;
    # a dB axis without one is an incomplete unit on a published figure.
    ax.set_ylabel("Level (dB re µPa²/Hz)")
    ax.set_xscale("log")
    ax.set_xlim(_log_freq_xlim(result.frequencies))
    ax.set_ylim((ymin, ymax))
    # The level axis carries the histogram, not the density: level_edges spans
    # every row the mesh draws.
    _warn_if_offscreen(ax, result.level_edges, "plot_ppsd", "ymin=/ymax")
    ax.grid(which="both", alpha=0.5)
    ax.legend(loc="upper right")
    return fig, ax


@typed_plot_error
def plot_sel(sel_pa2s, bands, ax=None, *, ref=REFERENCE_PRESSURE_WATER,
             duration=None, band_type="third_octave", ylim=(0, 200),
             title=None, figsize=(10, 6), **mpl_kw):
    """Bar plot of SEL per band (dB). Consumes :func:`sel` output."""
    fig, ax = fig_ax(ax, figsize)
    # ``sel`` returns bands as (low, centre, high) triples; the contiguous edge
    # vector is every low edge plus the top edge of the last band.
    Fedges = [low for low, _, _ in bands] + [bands[-1][2]]
    width = [Fedges[i + 1] - Fedges[i] for i in range(len(Fedges) - 1)]
    sel_db = power_to_db(np.asarray(sel_pa2s), ref)
    ax.bar(Fedges[:-1], sel_db, width=width,
           align="edge", edgecolor="black", **mpl_kw)
    ax.set_title(title or f"SEL ({duration}s)", loc="left")
    ax.set_ylabel(f"Level (dB re {_ref_label(ref)}Pa²·s)")
    if band_type != "linear":
        ax.set_xscale("log")
    ax.set_xlabel(f"Frequency ({band_type}) (Hz)")
    ax.set_ylim(ylim)
    _warn_if_offscreen(ax, sel_db, "plot_sel", "ylim")
    ax.grid(which="both", alpha=0.75)
    ax.set_axisbelow(True)
    return fig, ax


# ── Time-frequency (timefreq) ───────────────────────────────────────────────

@typed_plot_error
def plot_spectrogram(frequencies, times, Sxx, ax=None, *,
                     ref=REFERENCE_PRESSURE_WATER, ymin=1, ymax=None, vmin=0,
                     vmax=200, cmap="jet", title=None, figsize=(10, 6),
                     show_colorbar=True, **mpl_kw):
    """Spectrogram colormap (dB). Consumes :func:`spectrogram` output.

    ``ymin`` / ``ymax`` bound the frequency axis and are symmetric: ``None`` on
    either end takes the record's own first / last bin, so ``ymin=None`` drops
    the 1 Hz clamp the default applies. A clamp that sits above the record's
    whole band would reverse the axis, so such a band starts at its own first
    positive bin instead."""
    Sxx_db = power_to_db(np.asarray(Sxx), ref)
    fig, ax = fig_ax(ax, figsize)
    pcm = ax.pcolormesh(times, frequencies, Sxx_db, cmap=cmap, shading="auto",
                        vmin=vmin, vmax=vmax, **mpl_kw)
    if show_colorbar:
        fig.colorbar(pcm, ax=ax, label=f"Level (dB re {_ref_label(ref)}Pa²/Hz)")
    ax.set_title(title or "Spectrogram", loc="left")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    hi = float(frequencies[-1] if ymax is None else ymax)
    ax.set_ylim((float(frequencies[0]), hi) if ymin is None
                else _clamped_freq_limits(frequencies, ymin, hi))
    _warn_if_offscreen(ax, frequencies, "plot_spectrogram", "ymin/ymax")
    _warn_if_colour_saturated(Sxx_db, vmin, vmax, "plot_spectrogram", "vmin/vmax")
    ax.grid(which="both", alpha=0.25, color="black")
    return fig, ax


# ── Constant-Q (Brown 1991) ─────────────────────────────────────────────────

@typed_plot_error
def plot_constant_q_spectrogram(frequencies, times, power, ax=None, *,
                                ref=REFERENCE_PRESSURE_WATER, scaling="spectrum",
                                vmin=0, vmax=200, cmap="jet", title=None,
                                figsize=(10, 6), show_colorbar=True, **mpl_kw):
    """Constant-Q spectrogram colormap (dB, log frequency). Consumes
    :func:`constant_q_spectrogram` output ``(frequencies, times, power)``. Pass
    the same ``scaling`` used there so the unit reads ``Pa²`` (band power) or
    ``Pa²/Hz`` (density)."""
    unit = f"{_ref_label(ref)}Pa²" + ("/Hz" if scaling == "density" else "")
    power_db = power_to_db(np.asarray(power), ref)
    fig, ax = fig_ax(ax, figsize)
    pcm = ax.pcolormesh(times, frequencies, power_db, cmap=cmap, shading="auto",
                        vmin=vmin, vmax=vmax, **mpl_kw)
    if show_colorbar:
        fig.colorbar(pcm, ax=ax, label=f"Level (dB re {unit})")
    ax.set_title(title or "Constant-Q spectrogram", loc="left")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_yscale("log")
    ax.set_ylim((frequencies[0], frequencies[-1]))
    _warn_if_colour_saturated(power_db, vmin, vmax,
                              "plot_constant_q_spectrogram", "vmin/vmax")
    ax.grid(which="both", alpha=0.25, color="black")
    return fig, ax


@typed_plot_error
def plot_constant_q_psd(frequencies, power, ax=None, *,
                        ref=REFERENCE_PRESSURE_WATER, scaling="spectrum",
                        label=None, ymin=0, ymax=150, title=None,
                        figsize=(10, 6), **mpl_kw):
    """Line plot of constant-Q power (dB, log frequency). Consumes
    :func:`constant_q_psd` output ``(frequencies, power)``. Pass the same
    ``scaling`` used there: ``'spectrum'`` labels band power (``Pa²``),
    ``'density'`` labels PSD (``Pa²/Hz``)."""
    unit = f"{_ref_label(ref)}Pa²" + ("/Hz" if scaling == "density" else "")
    power_db = power_to_db(np.asarray(power), ref)
    fig, ax = fig_ax(ax, figsize)
    ax.semilogx(frequencies, power_db, label=label, **mpl_kw)
    ax.set_title(title or ("Constant-Q PSD" if scaling == "density"
                           else "Constant-Q band power"), loc="left")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"Level (dB re {unit})")
    ax.set_ylim((ymin, ymax))
    ax.set_xlim(_log_freq_xlim(frequencies))
    _warn_if_offscreen(ax, power_db, "plot_constant_q_psd", "ymin=/ymax")
    ax.grid(which="both", alpha=0.75)
    if label:
        ax.legend()
    return fig, ax


@typed_plot_error
def plot_constant_q_ppsd(result, ax=None, *, scaling="spectrum", ymin=0,
                         ymax=200, vmin=0, vmax=None, cmap="jet", title=None,
                         figsize=(10, 6), show_colorbar=True, **mpl_kw):
    """2-D histogram of constant-Q power levels. Consumes a ``CQPPSDResult``.
    The dB reference is fixed at compute time by
    :func:`probabilistic_constant_q` (default 1 µPa); pass the same ``scaling``
    used there so the level axis reads ``dB re µPa²`` (band power) or
    ``dB re µPa²/Hz`` (density)."""
    unit = "µPa²" + ("/Hz" if scaling == "density" else "")
    if vmax is None:
        vmax = 1 / result.binwidth_db          # density ceiling, as in plot_ppsd
    fig, ax = fig_ax(ax, figsize)
    align = result.binwidth_db / 2             # bin edges → bin centres
    pcm = ax.pcolormesh(result.frequencies, result.level_edges[:-1] + align,
                        result.pdf, cmap=cmap, shading="auto",
                        vmin=vmin, vmax=vmax, **mpl_kw)
    if show_colorbar:
        fig.colorbar(pcm, ax=ax,
                     label=f"Probability Density ({result.binwidth_db:.1f} dB/bin)")
    ax.plot(result.frequencies, result.mean_db, "k-", label="Mean level", lw=1.5)
    ax.plot(result.frequencies, result.mean_db + result.std_db, "k--",
            label="Mean level ± STD")
    ax.plot(result.frequencies, result.mean_db - result.std_db, "k--")
    ax.set_title(title or "Constant-Q PPSD", loc="left")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"Level (dB re {unit})")
    ax.set_xscale("log")
    ax.set_xlim(_log_freq_xlim(result.frequencies))
    ax.set_ylim((ymin, ymax))
    # The level axis carries the histogram, not the density: level_edges spans
    # every row the mesh draws.
    _warn_if_offscreen(ax, result.level_edges, "plot_constant_q_ppsd",
                       "ymin=/ymax")
    ax.grid(which="both", alpha=0.5)
    ax.legend(loc="upper right")
    return fig, ax


@typed_plot_error
def plot_cwt(frequencies, W, sample_rate, ax=None, *, cmap="jet", title=None,
             figsize=(10, 6), show_colorbar=True, **mpl_kw):
    """Scalogram ``|W|`` (time on x, frequency on y). Consumes :func:`cwt`
    output ``(frequencies, W)``."""
    amp = np.abs(np.asarray(W))
    t = np.arange(amp.shape[1]) / float(sample_rate)
    fig, ax = fig_ax(ax, figsize)
    pcm = ax.pcolormesh(t, frequencies, amp, cmap=cmap, shading="auto", **mpl_kw)
    if show_colorbar:
        fig.colorbar(pcm, ax=ax, label="|W|")
    ax.set_title(title or "CWT scalogram", loc="left")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    return fig, ax


@typed_plot_error
def plot_wigner_ville(frequencies, times, W, ax=None, *, cmap="jet", title=None,
                      figsize=(10, 6), show_colorbar=True, **mpl_kw):
    """Wigner-Ville distribution image. Consumes :func:`wigner_ville` output
    ``(frequencies, times, W)``."""
    fig, ax = fig_ax(ax, figsize)
    pcm = ax.pcolormesh(times, frequencies, np.real(np.asarray(W)), cmap=cmap,
                        shading="auto", **mpl_kw)
    if show_colorbar:
        fig.colorbar(pcm, ax=ax, label="WVD")
    ax.set_title(title or "Wigner-Ville", loc="left")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    return fig, ax


@typed_plot_error
def plot_cepstrum(c, ax=None, *, sample_rate=None, title=None, figsize=(9, 4),
                  **mpl_kw):
    """Line plot of a cepstrum vs quefrency. Consumes :func:`cepstrum` output."""
    c = np.real(np.asarray(c))
    fig, ax = fig_ax(ax, figsize)
    if sample_rate is not None:
        q = np.arange(c.size) / float(sample_rate)
        ax.plot(q, c, **mpl_kw)
        ax.set_xlabel("Quefrency (s)")
    else:
        ax.plot(c, **mpl_kw)
        ax.set_xlabel("Quefrency (samples)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title or "Cepstrum", loc="left")
    ax.grid(alpha=0.3)
    return fig, ax


# ── Decidecade bands / array spectra / ambiguity ────────────────────────────

@typed_plot_error
def plot_band_levels(centers, levels, ax=None, *, title=None, width=0.8,
                     ref_label="1 µPa²", figsize=(9, 4), **mpl_kw):
    """Bar plot of decidecade band levels vs centre frequency. Consumes
    :func:`decidecade_band_levels` output."""
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
    ax.set_xlabel("Decidecade band centre (Hz)")
    ax.set_ylabel(f"Band level (dB re {ref_label})")
    ax.set_title(title or "Decidecade band levels", loc="left")
    ax.grid(alpha=0.3, axis="y")
    return fig, ax


@typed_plot_error
def plot_angular_spectrum(angles_deg, spectrum, ax=None, *, db=True, label=None,
                          title=None, figsize=(8, 4), **mpl_kw):
    """Line plot of a beamformer angular spectrum (Bartlett/MVDR/MUSIC)."""
    P = np.real(np.asarray(spectrum))
    if db:
        # Beamformer output has no absolute reference (MVDR/MUSIC pseudo-power
        # least of all), so the dB axis is relative to the peak: 0 dB = look
        # direction of maximum response.
        P = 10.0 * np.log10(P / np.max(P))
    fig, ax = fig_ax(ax, figsize)
    ax.plot(angles_deg, P, label=label, **mpl_kw)
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Power (dB)" if db else "Power")
    ax.set_title(title or "Angular spectrum", loc="left")
    ax.grid(alpha=0.3)
    if label:
        ax.legend()
    return fig, ax


@typed_plot_error
def plot_ambiguity(delays_s, doppler_hz, chi, ax=None, *, cmap="jet",
                   title=None, figsize=(8, 6), show_colorbar=True, **mpl_kw):
    """Range-Doppler ambiguity surface ``|chi|``. Consumes
    :func:`ambiguity_function` output."""
    amp = _require_image_grid(np.abs(np.asarray(chi)), len(doppler_hz),
                              len(delays_s), "plot_ambiguity",
                              "doppler_hz", "delays_s")
    fig, ax = fig_ax(ax, figsize)
    im = ax.imshow(amp, aspect="auto", origin="lower",
                   extent=_cell_edge_extent(np.asarray(delays_s) * 1e3,
                                            doppler_hz),
                   cmap=cmap, **mpl_kw)
    if show_colorbar:
        fig.colorbar(im, ax=ax, label="|χ|")
    ax.set_title(title or "Ambiguity surface", loc="left")
    ax.set_xlabel("Delay (ms)")
    ax.set_ylabel("Doppler (Hz)")
    return fig, ax


# ── System identification (FRF) ─────────────────────────────────────────────

@typed_plot_error
def plot_frf(frequencies, tf, ax=None, *, tag="", label=None, ymin=-60,
             ymax=60, title=None, figsize=(10, 12), **mpl_kw):
    """Transfer-function magnitude (dB) + phase (deg). Consumes ``FRF`` output
    ``(frequencies, tf)``. ``ax`` may be a 2-tuple ``(ax_mag, ax_phase)``."""
    if ax is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    else:
        ax1, ax2 = ax
        fig = ax1.figure
    lbl = (f"{tag} {label}").strip() if (tag or label) else None
    mag_db = 20 * np.log10(np.abs(tf))
    ax1.plot(frequencies, mag_db, label=lbl, **mpl_kw)
    ax1.set_title(title or "Frequency response", loc="left")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_xscale("log")
    ax1.set_ylim((ymin, ymax))
    ax1.set_xlim(_log_freq_xlim(frequencies))
    # Only the magnitude panel is pinned to a fixed window; the phase axis
    # below spans the full ±180° a phase can occupy.
    _warn_if_offscreen(ax1, mag_db, "plot_frf", "ymin=/ymax")
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
                   ylim=(0.75, 1.01), figsize=(10, 4), **mpl_kw):
    """Coherence vs frequency. Consumes ``FRF`` ``(frequencies, coh)``.

    ``ylim`` defaults to the near-unity window a well-conditioned FRF lives in;
    widen it (``ylim=(0, 1.01)``) to see a poorly coherent band, which would
    otherwise fall entirely below the default axes."""
    fig, ax = fig_ax(ax, figsize)
    ax.plot(frequencies, coh, label=label, **mpl_kw)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Coherence")
    ax.set_xscale("log")
    ax.set_ylim(ylim)
    ax.set_xlim(_log_freq_xlim(frequencies))
    _warn_if_offscreen(ax, coh, "plot_coherence", "ylim")
    ax.grid(which="both", alpha=0.5)
    ax.set_title(title or "Coherence", loc="left")
    if label:
        ax.legend()
    return fig, ax


@typed_plot_error
def plot_impulse_response_info(Minfo, Vinfo, g, *, title=None, figsize=(12, 8)):
    """LS-FIR diagnostics: information matrix, vector, and impulse response."""
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(Minfo, cmap="viridis", aspect="equal")
    ax1.set_title(title or "Information Matrix", loc="left")
    ax1.set_xlabel("Index j")
    ax1.set_ylabel("Index i")
    fig.colorbar(im, ax=ax1, shrink=0.8, label="Correlation Value")
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(np.arange(len(Vinfo)), Vinfo, color="skyblue", edgecolor="navy")
    ax2.set_title("Information Vector", loc="left")
    ax2.set_xlabel("Index i")
    ax2.set_ylabel("Cross-correlation Value")
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(g, color="red", linestyle="-", marker="o", markersize=4)
    ax3.set_title("Impulse Response", loc="left")
    ax3.set_xlabel("Time Index")
    ax3.set_ylabel("Amplitude")
    ax3.grid(True)
    return fig, [ax1, ax2, ax3]
