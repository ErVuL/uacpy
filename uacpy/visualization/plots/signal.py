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
from uacpy.core.acoustics import power_to_dB
from uacpy.core.exceptions import ConfigurationError
from uacpy.visualization.plots._common import (ZORDER_LEGEND, ZORDER_SOURCE,
                                               _cell_edge_extent, _flip_y,
                                               _require_nonempty,
                                               fig_ax, typed_plot_error,
                                               _plot_warn, _title_or)
from uacpy.visualization.style import SOURCE_MARKER_STYLE


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
        ax.text(k, f, f" {c:.0f} m/s", color=color, fontsize='small',
                va="top", ha="right")


# The colour window autoscales, as it does on the other transform panels
# (plot_radon, plot_taup). A fixed -60..+20 dB window suits a PEAK-RELATIVE
# scale — which is what docs/figure_scripts/signal.py hand-rolls, at
# vmin=-40, vmax=0 — but the level here goes through
# power_to_dB(power, ref), an ABSOLUTE dB re 1 uPa^2. Measured on the
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
    fk_dB = power_to_dB(np.asarray(power), ref)
    fig, ax = fig_ax(ax, figsize)
    # Edge-aligned: the axes are FFT bin centres, and draw_sound_cone below
    # places f = c*k/(2*pi) at true coordinates, so a half-bin shift would
    # offset the image against the very line used to read it.
    im = ax.imshow(fk_dB, extent=_cell_edge_extent(wavenumbers, frequencies),
                   origin="lower", aspect="auto",
                   vmin=vmin, vmax=vmax, cmap=cmap, **mpl_kw)
    if sound_speed is not None:
        draw_sound_cone(ax, frequencies[-1], wavenumbers[-1], sound_speed)
    ax.set_title(_title_or(title, "f–k spectrum"), loc="left")
    ax.set_xlabel("Wavenumber k (rad/m)")
    ax.set_ylabel("Frequency (Hz)")
    ax.grid(alpha=0.3)
    if show_colorbar:
        # With the reference: `power_to_dB(power, ref)` above makes this an
        # ABSOLUTE level, so the number is meaningless without saying what it
        # is referred to — as every sibling axis in this module does.
        fig.colorbar(im, ax=ax, label=f"Power (dB re {_ref_label(ref)}Pa²)")
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
def plot_radon(moveout, taus, R, ax=None, *, kind="linear", vmin=None,
               vmax=None, cmap="jet", title=None, figsize=(8, 6),
               show_colorbar=True, **mpl_kw):
    """Image ``|R|`` (moveout on x, intercept time on y). Consumes
    :func:`radon_transform` output."""
    amp = _require_image_grid(np.abs(np.asarray(R)), len(moveout), len(taus),
                              "plot_radon", "moveout", "taus")
    xlabel, scale = _RADON_AXIS.get(kind, ("Moveout", 1.0))
    fig, ax = _plot_tau_panel(np.asarray(moveout) * scale, taus, amp, ax,
                              vmin=vmin, vmax=vmax, cmap=cmap, figsize=figsize,
                              show_colorbar=show_colorbar, **mpl_kw)
    ax.set_title(_title_or(title, f"Radon ({kind})"), loc="left")
    ax.set_xlabel(xlabel)
    return fig, ax


def draw_slowness_line(ax, tau_max, sound_speed, *, color="w", ls="--",
                       lw=1.1, alpha=0.85, label=True):
    """Mark slowness ``p = +/-1/c`` (s/km) of a reference speed on a tau-p axis."""
    p_skm = 1000.0 / float(sound_speed)          # 1/c in s/m → s/km
    for sgn in (-1.0, 1.0):
        ax.axvline(sgn * p_skm, color=color, ls=ls, lw=lw, alpha=alpha)
    if label:
        ax.text(p_skm, tau_max, f" {sound_speed:.0f} m/s", color=color,
                fontsize='small', va="bottom", ha="left")


@typed_plot_error
def plot_taup(slownesses, taus, taup, ax=None, *, vmin=None, vmax=None,
              cmap="jet", sound_speed=None, title=None, figsize=(8, 6),
              show_colorbar=True, **mpl_kw):
    """Image a tau-p panel (slowness s/km on x, intercept time on y). Consumes
    :func:`taup_transform` output."""
    p_skm = np.asarray(slownesses) * 1000.0      # taup_transform returns s/m
    amp = _require_image_grid(np.abs(np.asarray(taup)), len(slownesses),
                              len(taus), "plot_taup", "slownesses", "taus")
    fig, ax = _plot_tau_panel(p_skm, taus, amp, ax, vmin=vmin, vmax=vmax,
                              cmap=cmap, figsize=figsize,
                              show_colorbar=show_colorbar, **mpl_kw)
    if sound_speed is not None:
        draw_slowness_line(ax, taus[-1], sound_speed)
    ax.set_title(_title_or(title, "tau-p"), loc="left")
    ax.set_xlabel("Slowness p (s/km)")
    return fig, ax


# ── Spectral / level estimators (analysis) ──────────────────────────────────

#: What each scaling puts on a level axis, and what the panel is called.
#: An exposure carries the record's duration and a density the band's width,
#: so a plot that names the wrong one is out by that factor and says nothing.
_SCALING_UNIT = {"density": "Pa²/Hz", "spectrum": "Pa²",
                 "exposure": "Pa²·s"}
_SCALING_KIND = {"density": "Power spectral density",
                 "spectrum": "Power spectrum",
                 "exposure": "Sound exposure"}
#: What a level HISTOGRAM is called, per scaling. "PPSD" is the name the
#: density one goes by (McNamara & Buland 2004); the other two have no
#: acronym, and calling them PPSD would put "density" over band power.
_HISTOGRAM_KIND = {"density": "PPSD", "spectrum": "Band-power histogram",
                   "exposure": "SEL histogram"}


def _histogram_title(result):
    """The panel's own name: the statistic, and the segment it summarises."""
    kind = _HISTOGRAM_KIND[getattr(result, "scaling", "density")]
    seg = getattr(result, "seg_duration", None)
    return f"{kind} ({seg}s)" if seg is not None else kind



@typed_plot_error
def plot_psd(frequencies, psd_linear=None, ax=None, *,
             ref=REFERENCE_PRESSURE_WATER, scaling=None, label=None, ymin=0,
             ymax=150, title=None, figsize=(10, 6), freq_scale="log",
             **mpl_kw):
    """Line plot of a Welch estimate (dB). Consumes
    :func:`uacpy.acoustic_signal.welch` or
    :func:`uacpy.acoustic_signal.constant_q` output.

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

    ``ymin`` / ``ymax`` pin the level axis to the 0–150 dB window an ambient
    record occupies; a quieter one needs them widened or the panel comes out
    empty."""
    method = None
    if psd_linear is None:
        frequencies, psd_linear, scaling, method = (
            frequencies.frequencies, frequencies.power,
            getattr(frequencies, "scaling", scaling),
            getattr(frequencies, "method", None))
    scaling = "density" if scaling is None else str(scaling)
    unit = _SCALING_UNIT[scaling]
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
    fig, ax = fig_ax(ax, figsize)
    ax.plot(frequencies, psd_dB, label=label, **mpl_kw)
    ax.set_xscale(freq_scale)
    ax.set_title(_title_or(title, kind), loc="left")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"Level (dB re {_ref_label(ref)}{unit})")
    ax.set_ylim((ymin, ymax))
    # A log axis cannot show DC, so its lower limit is the first positive
    # bin; a linear one spans the record's own band.
    ax.set_xlim(_log_freq_xlim(frequencies) if freq_scale == "log"
                else (float(frequencies[0]), float(frequencies[-1])))
    _warn_if_offscreen(ax, psd_dB, "plot_psd", "ymin=/ymax")
    ax.grid(which="both", alpha=0.75)
    if label:
        ax.legend()
    return fig, ax


def _plot_level_histogram(result, ax, *, y_label, default_title, caller,
                          ymin, ymax, vmin, vmax, cmap, title, figsize,
                          show_colorbar, **mpl_kw):
    """Render one PPSD-style level histogram: the density mesh, the mean and
    ±1 STD lines over it, and a log frequency axis.

    Shared by :func:`plot_ppsd` and :func:`plot_constant_q_ppsd`, which differ
    only in the level axis label, the default title and the name they report
    in an off-screen warning. Any result carrying ``frequencies``,
    ``level_edges``, ``pdf``, ``mean_dB``, ``std_dB`` and ``binwidth_dB``
    renders here; a caller's own shape guards run before it, so nothing that
    would raise gets a figure allocated first."""
    if vmax is None:
        # Each frequency column integrates to 1 over the level axis, so the
        # largest attainable density is 1/binwidth (all mass in one bin) —
        # the natural top of the colour scale.
        vmax = 1 / result.binwidth_dB
    fig, ax = fig_ax(ax, figsize)
    # ``level_edges`` are bin EDGES and ``pdf`` has one row per bin; shift by
    # half a bin so each row is centred on its own level. Empty bins arrive as
    # NaN and render as the axes background.
    align = result.binwidth_dB / 2
    pcm = ax.pcolormesh(result.frequencies, result.level_edges[:-1] + align,
                        result.pdf, cmap=cmap, shading="auto",
                        vmin=vmin, vmax=vmax, **mpl_kw)
    if show_colorbar:
        fig.colorbar(pcm, ax=ax,
                     label=f"Probability Density ({result.binwidth_dB:.1f} dB/bin)")
    ax.plot(result.frequencies, result.mean_dB, "k-", label="Mean level", lw=1.5)
    ax.plot(result.frequencies, result.mean_dB + result.std_dB, "k--",
            label="Mean level ± STD")
    ax.plot(result.frequencies, result.mean_dB - result.std_dB, "k--")
    ax.set_title(_title_or(title, default_title), loc="left")
    # A banded histogram's values sit on whole bands, so the axis names
    # the ladder, as the bar plot does — the two result types answer
    # "what are these sitting on?" the same way.
    _band_type = getattr(result, "band_type", None)
    ax.set_xlabel(f"Frequency ({_band_type}) (Hz)" if _band_type
                  else "Frequency (Hz)")
    ax.set_ylabel(y_label)
    ax.set_xscale("log")
    ax.set_xlim(_log_freq_xlim(result.frequencies))
    ax.set_ylim((ymin, ymax))
    # The level axis carries the histogram, not the density: level_edges spans
    # every row the mesh draws.
    _warn_if_offscreen(ax, result.level_edges, caller, "ymin=/ymax")
    ax.grid(which="both", alpha=0.5)
    ax.legend(loc="upper right")
    return fig, ax


@typed_plot_error
def plot_ppsd(result, ax=None, *, ymin=0, ymax=200, vmin=0, vmax=None,
              cmap="jet", title=None, figsize=(10, 6), show_colorbar=True,
              **mpl_kw):
    """2-D histogram of Welch spectral levels. Consumes a
    :class:`~uacpy.acoustic_signal.ProbabilisticSpectralEstimate` computed
    with ``method='welch'``.

    The level axis is named from the reference the result carries, so a caller
    who ran the estimator against a Pascal reference is not handed a µPa axis
    120 dB out."""
    if not hasattr(result, 'frequencies') or not hasattr(result, 'level_edges'):
        raise ConfigurationError(
            f"plot_ppsd: expected a probabilistic_welch() or "
            f"probabilistic_sound_exposure() result (with .frequencies and "
            f".level_edges); got {type(result).__name__}.")
    # Both histogram estimators return the one type, so the method is what
    # separates them: constant-Q bins are geometric and carry no
    # ``seg_duration``, which this plotter's linear axis and title state.
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
    unit = _SCALING_UNIT[getattr(result, 'scaling', 'density')]
    return _plot_level_histogram(
        result, ax, y_label=f"Level (dB re {_ref_label(ref)}{unit})",
        default_title=_histogram_title(result), caller="plot_ppsd",
        ymin=ymin, ymax=ymax, vmin=vmin, vmax=vmax, cmap=cmap, title=title,
        figsize=figsize, show_colorbar=show_colorbar, **mpl_kw)


@typed_plot_error
def plot_sel(sel_pa2s, bands=None, ax=None, *, ref=REFERENCE_PRESSURE_WATER,
             duration=None, band_type="third_octave", ylim=(0, 200),
             title=None, figsize=(10, 6), **mpl_kw):
    """Bar plot of standard-band levels (dB). Consumes a banded
    :class:`~uacpy.acoustic_signal.SpectralEstimate` — one computed with a
    ``band_type`` — or the same numbers as ``(values, bands)`` arrays.

    The unit comes from the estimate's ``scaling``: an exposure is per band
    and per second of record (Pa²·s), a spectrum per band (Pa²) and a density
    per hertz of the band's own width (Pa²/Hz). A bar chart labelled "·s" over
    band power would misstate the quantity by the record length."""
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
    # Bands are (low, centre, high) triples; the contiguous edge vector is
    # every low edge plus the top edge of the last band.
    Fedges = [low for low, _, _ in bands] + [bands[-1][2]]
    width = [Fedges[i + 1] - Fedges[i] for i in range(len(Fedges) - 1)]
    sel_dB = power_to_dB(np.asarray(sel_pa2s), ref)
    ax.bar(Fedges[:-1], sel_dB, width=width,
           align="edge", edgecolor="black", **mpl_kw)
    unit = _SCALING_UNIT[scaling]
    default = {"exposure": "SEL", "spectrum": "Band power",
               "density": "Band density"}[scaling]
    # ``duration`` is the caller's own note about the record; without it the
    # title states the quantity alone rather than "(Nones)".
    ax.set_title(_title_or(title, f"{default} ({duration}s)"
                           if duration is not None else default), loc="left")
    ax.set_ylabel(f"Level (dB re {_ref_label(ref)}{unit})")
    if band_type != "linear":
        ax.set_xscale("log")
    ax.set_xlabel(f"Frequency ({band_type}) (Hz)")
    ax.set_ylim(ylim)
    _warn_if_offscreen(ax, sel_dB, "plot_sel", "ylim")
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
    Sxx_dB = _require_image_grid(power_to_dB(np.asarray(Sxx), ref),
                                 len(frequencies), len(times),
                                 'plot_spectrogram', 'frequencies', 'times')
    fig, ax = fig_ax(ax, figsize)
    pcm = ax.pcolormesh(times, frequencies, Sxx_dB, cmap=cmap, shading="auto",
                        vmin=vmin, vmax=vmax, **mpl_kw)
    if show_colorbar:
        fig.colorbar(pcm, ax=ax, label=f"Level (dB re {_ref_label(ref)}Pa²/Hz)")
    ax.set_title(_title_or(title, "Spectrogram"), loc="left")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    hi = float(frequencies[-1] if ymax is None else ymax)
    ax.set_ylim((float(frequencies[0]), hi) if ymin is None
                else _clamped_freq_limits(frequencies, ymin, hi))
    _warn_if_offscreen(ax, frequencies, "plot_spectrogram", "ymin/ymax")
    _warn_if_colour_saturated(Sxx_dB, vmin, vmax, "plot_spectrogram", "vmin/vmax")
    ax.grid(which="both", alpha=0.25, color="black")
    return fig, ax


# ── Constant-Q (Brown 1991) ─────────────────────────────────────────────────

@typed_plot_error
def plot_constant_q_transform(frequencies, coefficients, ax=None, *,
                              label=None, title=None, figsize=(10, 6),
                              **mpl_kw):
    """Line plot of one constant-Q frame's magnitude (log frequency). Consumes
    :func:`constant_q_transform` output ``(frequencies, coefficients)``.

    Linear amplitude, not dB, and deliberately: the coefficients are the
    analytic band amplitude ``A/2``, while the one-sided band power the rest of
    the family reports is ``2*|X_cq|**2``. Read levels off
    :func:`plot_constant_q_psd`, whose estimator applies that conversion; this
    panel shows the raw transform a single frame returns.
    """
    magnitude = np.abs(np.asarray(coefficients))
    if magnitude.ndim != 1 or magnitude.size != len(frequencies):
        raise ConfigurationError(
            f"plot_constant_q_transform: coefficients has shape "
            f"{magnitude.shape}; expected one value per frequency "
            f"(len(frequencies)={len(frequencies)}) — pass "
            f"constant_q_transform()'s own output.")
    fig, ax = fig_ax(ax, figsize)
    ax.semilogx(frequencies, magnitude, label=label, **mpl_kw)
    ax.set_title(_title_or(title, "Constant-Q transform"), loc="left")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|X_cq|")
    ax.set_xlim(_log_freq_xlim(frequencies))
    ax.grid(which="both", alpha=0.75)
    if label:
        ax.legend()
    return fig, ax


@typed_plot_error
def plot_constant_q_spectrogram(frequencies, times, power, ax=None, *,
                                ref=REFERENCE_PRESSURE_WATER, scaling="spectrum",
                                vmin=0, vmax=200, cmap="jet", title=None,
                                figsize=(10, 6), show_colorbar=True, **mpl_kw):
    """Constant-Q spectrogram colormap (dB, log frequency). Consumes
    :func:`constant_q_spectrogram` output ``(frequencies, times, power)``. Pass
    the same ``scaling`` used there so the unit reads ``Pa²`` (band power) or
    ``Pa²/Hz`` (density)."""
    unit = f"{_ref_label(ref)}{_SCALING_UNIT[scaling]}"
    power_dB = _require_image_grid(power_to_dB(np.asarray(power), ref),
                                   len(frequencies), len(times),
                                   'plot_constant_q_spectrogram',
                                   'frequencies', 'times')
    fig, ax = fig_ax(ax, figsize)
    pcm = ax.pcolormesh(times, frequencies, power_dB, cmap=cmap, shading="auto",
                        vmin=vmin, vmax=vmax, **mpl_kw)
    if show_colorbar:
        fig.colorbar(pcm, ax=ax, label=f"Level (dB re {unit})")
    ax.set_title(_title_or(title, "Constant-Q spectrogram"), loc="left")
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
                        label=None, ymin=0, ymax=150, title=None,
                        figsize=(10, 6), **mpl_kw):
    """Line plot of constant-Q power (dB, log frequency). Consumes
    :func:`uacpy.acoustic_signal.constant_q` output ``(frequencies, power)``. Pass the same
    ``scaling`` used there: ``'spectrum'`` labels band power (``Pa²``),
    ``'density'`` labels PSD (``Pa²/Hz``)."""
    unit = f"{_ref_label(ref)}{_SCALING_UNIT[scaling]}"
    power_dB = power_to_dB(np.asarray(power), ref)
    fig, ax = fig_ax(ax, figsize)
    ax.semilogx(frequencies, power_dB, label=label, **mpl_kw)
    ax.set_title(_title_or(title, "Constant-Q PSD" if scaling == "density"
                           else "Constant-Q band power"), loc="left")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(f"Level (dB re {unit})")
    ax.set_ylim((ymin, ymax))
    ax.set_xlim(_log_freq_xlim(frequencies))
    _warn_if_offscreen(ax, power_dB, "plot_constant_q_psd", "ymin=/ymax")
    ax.grid(which="both", alpha=0.75)
    if label:
        ax.legend()
    return fig, ax


@typed_plot_error
def plot_constant_q_ppsd(result, ax=None, *, scaling="spectrum", ymin=0,
                         ymax=200, vmin=0, vmax=None, cmap="jet", title=None,
                         figsize=(10, 6), show_colorbar=True, **mpl_kw):
    """2-D histogram of constant-Q power levels. Consumes a
    :class:`~uacpy.acoustic_signal.ProbabilisticSpectralEstimate` computed
    with ``method='constant_q'``.

    The dB reference is fixed at compute time by
    :func:`uacpy.acoustic_signal.probabilistic_welch` (default
    1 µPa) and read off the result, so the level axis names whatever reference
    was used; ``scaling`` is read off it too, and the keyword stands in only
    for a hand-built result that carries none."""
    # From the result, not assumed: a hardcoded "µPa²" would be 120 dB out for a
    # caller who computed against a Pascal reference.
    ref = getattr(result, 'ref', REFERENCE_PRESSURE_WATER)
    # The result carries the scaling it was computed with; the keyword
    # stays as an override for a hand-built result that has none.
    scaling = getattr(result, 'scaling', None) or scaling
    unit = f"{_ref_label(ref)}{_SCALING_UNIT[scaling]}"
    return _plot_level_histogram(
        result, ax, y_label=f"Level (dB re {unit})",
        default_title=f"Constant-Q {_histogram_title(result)}",
        caller="plot_constant_q_ppsd",
        ymin=ymin, ymax=ymax, vmin=vmin, vmax=vmax, cmap=cmap, title=title,
        figsize=figsize, show_colorbar=show_colorbar, **mpl_kw)


@typed_plot_error
def plot_cwt(frequencies, W, sample_rate, ax=None, *, cmap="jet", title=None,
             figsize=(10, 6), show_colorbar=True, **mpl_kw):
    """Scalogram ``|W|`` (time on x, frequency on y). Consumes :func:`cwt`
    output ``(frequencies, W)``."""
    amp = np.abs(np.asarray(W))
    if amp.ndim != 2 or amp.shape[0] != len(frequencies):
        raise ConfigurationError(
            f"plot_cwt: W has shape {amp.shape}; expected (len(frequencies)="
            f"{len(frequencies)}, n_samples) — pass cwt()'s own output.")
    t = np.arange(amp.shape[1]) / float(sample_rate)
    fig, ax = fig_ax(ax, figsize)
    pcm = ax.pcolormesh(t, frequencies, amp, cmap=cmap, shading="auto", **mpl_kw)
    if show_colorbar:
        fig.colorbar(pcm, ax=ax, label="|W|")
    ax.set_title(_title_or(title, "CWT scalogram"), loc="left")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    return fig, ax


@typed_plot_error
def plot_wigner_ville(frequencies, times, W, ax=None, *, cmap="jet", title=None,
                      figsize=(10, 6), show_colorbar=True, **mpl_kw):
    """Wigner-Ville distribution image. Consumes :func:`wigner_ville` output
    ``(frequencies, times, W)``."""
    W_real = _require_image_grid(np.real(np.asarray(W)), len(frequencies),
                                 len(times), 'plot_wigner_ville',
                                 'frequencies', 'times')
    fig, ax = fig_ax(ax, figsize)
    pcm = ax.pcolormesh(times, frequencies, W_real, cmap=cmap,
                        shading="auto", **mpl_kw)
    if show_colorbar:
        fig.colorbar(pcm, ax=ax, label="WVD")
    ax.set_title(_title_or(title, "Wigner-Ville"), loc="left")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    return fig, ax


@typed_plot_error
def plot_cepstrum(c, ax=None, *, sample_rate=None, title=None, figsize=(9, 4),
                  **mpl_kw):
    """Line plot of a cepstrum vs quefrency. Consumes :func:`cepstrum` output."""
    _require_nonempty('plot_cepstrum', c=c)
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
    ax.set_title(_title_or(title, "Cepstrum"), loc="left")
    ax.grid(alpha=0.3)
    return fig, ax


# ── Decidecade bands / array spectra / ambiguity ────────────────────────────

@typed_plot_error
def plot_band_levels(centers, levels, ax=None, *, title=None, width=0.8,
                     ref_label="1 µPa²", figsize=(9, 4), **mpl_kw):
    """Bar plot of decidecade band levels vs centre frequency. Consumes
    :func:`decidecade_band_levels` output."""
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
    ax.set_xlabel("Decidecade band centre (Hz)")
    ax.set_ylabel(f"Band level (dB re {ref_label})")
    ax.set_title(_title_or(title, "Decidecade band levels"), loc="left")
    ax.grid(alpha=0.3, axis="y")
    return fig, ax


@typed_plot_error
def plot_angular_spectrum(angles_deg, spectrum, ax=None, *, dB=True, label=None,
                          title=None, figsize=(8, 4), **mpl_kw):
    """Line plot of a beamformer angular spectrum (Bartlett/MVDR/MUSIC)."""
    P = np.real(np.asarray(spectrum))
    if dB:
        # Beamformer output has no absolute reference (MVDR/MUSIC pseudo-power
        # least of all), so the dB axis is relative to the peak: 0 dB = look
        # direction of maximum response.
        P = 10.0 * np.log10(P / np.max(P))
    fig, ax = fig_ax(ax, figsize)
    ax.plot(angles_deg, P, label=label, **mpl_kw)
    ax.set_xlabel("Angle (deg)")
    ax.set_ylabel("Power (dB)" if dB else "Power")
    ax.set_title(_title_or(title, "Angular spectrum"), loc="left")
    ax.grid(alpha=0.3)
    if label:
        ax.legend()
    return fig, ax


@typed_plot_error
def plot_matched_field(x_m, z_m, surface, ax=None, *, dynamic_range=20.0,
                       cmap="viridis", true_position=None, mark_peak=True,
                       title=None, figsize=(8, 5), show_colorbar=True,
                       show_legend=True, **mpl_kw):
    """Matched-field ambiguity surface over a replica grid, in dB re its peak.

    Consumes the candidate-position axes a replica set carries
    (``replicas.replica_x`` / ``replica_z``, in metres) and the processor
    output of ``Covariance.bartlett`` / ``Covariance.mvdr``.
    ``true_position`` is an ``(x_m, z_m)`` pair, marked with the package's
    source star for comparison, and ``**mpl_kw`` goes to the pcolormesh.

    Both processors return ``(n_frequencies, n_zr, n_xr, n_yr)`` -- frequency
    FIRST, ``y`` last -- so a single-frequency run over a range/depth grid
    squeezes straight to the ``(z, x)`` plane this draws. Anything left over
    is refused rather than sliced: choosing a frequency or a ``y`` plane is
    the caller's decision, and a plotter making it silently would draw one
    slice of a search under a title claiming the whole of it.

    A degenerate candidate position -- one the forward model put no energy at
    -- is part of the contract, not a broken surface: ``mvdr`` writes NaN
    there and ``bartlett`` an exact zero. Both are drawn, the NaN as the
    colormap's "bad" colour and the zero on the floor; only a surface with no
    positive value anywhere is refused. ``show_legend=False`` drops the marker
    key, for a grid of panels that needs it once.
    """
    x = np.asarray(x_m, dtype=float)
    z = np.asarray(z_m, dtype=float)
    _require_nonempty('plot_matched_field', x_m=x, z_m=z)
    raw = np.asarray(surface)
    # Squeeze drops the length-1 frequency and y axes a single-frequency run
    # over a range/depth grid carries, whichever end they sit at.
    S = np.real(np.squeeze(raw))
    if S.ndim != 2:
        raise ConfigurationError(
            f"plot_matched_field: surface is {raw.shape}, which is not one "
            f"(z, x) plane once its length-1 axes are dropped. "
            f"Covariance.bartlett / .mvdr return "
            f"(n_frequencies, n_zr, n_xr, n_yr), so index the frequency and "
            f"y axes you want -- e.g. surface[0, :, :, 0] -- before plotting.")
    if S.shape != (z.size, x.size):
        raise ConfigurationError(
            f"plot_matched_field: surface is {S.shape}, but the replica grid "
            f"is (z, x) = ({z.size}, {x.size}). Pass the same replica set the "
            f"processor was run against.")
    # nanmax, not max: mvdr writes NaN at a candidate position the forward
    # model put no energy at, so one degenerate cell out of thousands used to
    # refuse the whole surface -- and blame the array geometry for it.
    finite = np.isfinite(S)
    peak = float(np.max(S[finite])) if finite.any() else float('nan')
    if not np.isfinite(peak) or peak <= 0.0:
        raise ConfigurationError(
            f"plot_matched_field: the surface peaks at {peak}, so a dB scale "
            f"relative to it is undefined. Check the covariance and replicas "
            f"come from the same array geometry.")
    # A matched-field processor carries no absolute reference -- MVDR's
    # pseudo-power least of all -- so 0 dB is the best-matching candidate
    # position, not a level. Clipped to the floor before the log because
    # bartlett scores that same degenerate cell as an exact zero, and
    # log10(0) is -inf plus a RuntimeWarning.
    floor = 10.0 ** (-abs(dynamic_range) / 10.0)
    SdB = 10.0 * np.log10(np.clip(S / peak, floor, None))
    fig, ax = fig_ax(ax, figsize)
    im = ax.pcolormesh(x / 1000.0, z, SdB, cmap=cmap, vmin=-abs(dynamic_range),
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
        iz, ix = np.unravel_index(int(np.nanargmax(S)), S.shape)
        ax.plot(x[ix] / 1000.0, z[iz], '+', color='black', ms=13, mew=2.2,
                zorder=ZORDER_SOURCE + 1, label='peak')
    if show_colorbar:
        fig.colorbar(im, ax=ax, label='dB re peak')
    ax.set_xlabel('Candidate range (km)')
    ax.set_ylabel('Candidate depth (m)')
    ax.set_title(_title_or(title, 'Matched-field ambiguity surface'),
                 loc='left')
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


@typed_plot_error
def plot_ambiguity(delays_s, doppler_hz, chi, ax=None, *, dB=False,
                   dynamic_range=40.0, cmap="jet",
                   title=None, figsize=(8, 6), show_colorbar=True, **mpl_kw):
    """Range-Doppler ambiguity surface ``|chi|``. Consumes
    :func:`ambiguity_function` output. ``dB=True`` shows it relative to its
    peak over ``dynamic_range`` decibels."""
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
        floor = 10.0 ** (-abs(dynamic_range) / 20.0)
        amp = 20.0 * np.log10(np.maximum(amp / peak, floor))
        mpl_kw.setdefault("vmin", -abs(dynamic_range))
        mpl_kw.setdefault("vmax", 0.0)
        label = "|χ| (dB re peak)"
    fig, ax = fig_ax(ax, figsize)
    im = ax.imshow(amp, aspect="auto", origin="lower",
                   extent=_cell_edge_extent(np.asarray(delays_s) * 1e3,
                                            doppler_hz),
                   cmap=cmap, **mpl_kw)
    if show_colorbar:
        fig.colorbar(im, ax=ax, label=label)
    ax.set_title(_title_or(title, "Ambiguity surface"), loc="left")
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
    mag_dB = 20 * np.log10(np.abs(tf))
    ax1.plot(frequencies, mag_dB, label=lbl, **mpl_kw)
    ax1.set_title(_title_or(title, "Frequency response"), loc="left")
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
    ax.set_title(_title_or(title, "Coherence"), loc="left")
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
    ax1.set_title(_title_or(title, "Information Matrix"), loc="left")
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
