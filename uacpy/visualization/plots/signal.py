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
from uacpy.visualization.plots._common import fig_ax


def _ref_label(ref):
    if ref == REFERENCE_PRESSURE_WATER:
        return "1µ"
    if ref == REFERENCE_PRESSURE_AIR:
        return "20µ"
    return f"{ref:02e}"




# ── f-k / Radon / tau-p gather transforms ───────────────────────────────────

def draw_sound_cone(ax, f_max, k_max, sound_speed, *, color="w", ls="--",
                    lw=1.1, alpha=0.85, label=True):
    """Overlay the acoustic cone ``f = c * nu`` onto an f-k axis."""
    c = float(sound_speed)
    nu = min(f_max / c, k_max)
    f = nu * c
    ax.plot([0, nu], [0, f], color=color, ls=ls, lw=lw, alpha=alpha)
    ax.plot([0, -nu], [0, f], color=color, ls=ls, lw=lw, alpha=alpha)
    if label:
        ax.text(nu, f, f" {c:.0f} m/s", color=color, fontsize=8,
                va="top", ha="right")


def plot_fk(frequencies, wavenumbers, power, ax=None, *, ref=REFERENCE_PRESSURE_WATER,
            vmin=-60, vmax=20, cmap=None, sound_speed=None, title=None,
            figsize=(10, 6), show_colorbar=True, **mpl_kw):
    """Image an f-k power panel (dB). Consumes :func:`fk_transform` output."""
    fk_db = power_to_db(np.asarray(power), ref)
    fig, ax = fig_ax(ax, figsize)
    im = ax.imshow(fk_db, extent=[wavenumbers[0], wavenumbers[-1],
                   frequencies[0], frequencies[-1]], origin="lower", aspect="auto",
                   vmin=vmin, vmax=vmax, cmap=cmap, **mpl_kw)
    if sound_speed is not None:
        draw_sound_cone(ax, frequencies[-1], wavenumbers[-1], sound_speed)
    ax.set_title(title or "f–k spectrum", loc="left")
    ax.set_xlabel("Spatial frequency [cycles/m]")
    ax.set_ylabel("Frequency [Hz]")
    ax.grid(alpha=0.3)
    if show_colorbar:
        fig.colorbar(im, ax=ax, label="Power [dB]")
    return fig, ax


_RADON_AXIS = {
    "linear": ("Slowness p [s/km]", 1e3),
    "parabolic": ("Curvature q [s/km^2]", 1e6),
    "hyperbolic": ("Velocity v [m/s]", 1.0),
}


def plot_radon(moveout, taus, R, ax=None, *, kind="linear", vmin=None,
               vmax=None, cmap="jet", title=None, figsize=(8, 6),
               show_colorbar=True, **mpl_kw):
    """Image ``|R|`` (moveout on x, intercept time on y). Consumes
    :func:`radon_transform` output."""
    amp = np.abs(np.asarray(R))
    xlabel, scale = _RADON_AXIS.get(kind, ("Moveout", 1.0))
    m = np.asarray(moveout) * scale
    vmax = amp.max() if vmax is None else vmax
    vmin = 0.0 if vmin is None else vmin
    fig, ax = fig_ax(ax, figsize)
    im = ax.imshow(amp.T, aspect="auto", origin="upper",
                   extent=[m[0], m[-1], taus[-1], taus[0]],
                   vmin=vmin, vmax=vmax, cmap=cmap, **mpl_kw)
    ax.set_title(title or f"Radon ({kind})", loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Intercept time tau [s]")
    if show_colorbar:
        fig.colorbar(im, ax=ax, label="Stack amplitude")
    return fig, ax


def draw_slowness_line(ax, tau_max, sound_speed, *, color="w", ls="--",
                       lw=1.1, alpha=0.85, label=True):
    """Mark slowness ``p = +/-1/c`` (s/km) of a reference speed on a tau-p axis."""
    p_skm = 1000.0 / float(sound_speed)
    for sgn in (-1.0, 1.0):
        ax.axvline(sgn * p_skm, color=color, ls=ls, lw=lw, alpha=alpha)
    if label:
        ax.text(p_skm, tau_max, f" {sound_speed:.0f} m/s", color=color,
                fontsize=8, va="bottom", ha="left")


def plot_taup(slownesses, taus, taup, ax=None, *, vmin=None, vmax=None,
              cmap="jet", sound_speed=None, title=None, figsize=(8, 6),
              show_colorbar=True, **mpl_kw):
    """Image a tau-p panel (slowness s/km on x, intercept time on y). Consumes
    :func:`taup_transform` output."""
    p_skm = np.asarray(slownesses) * 1000.0
    amp = np.abs(np.asarray(taup))
    vmax = amp.max() if vmax is None else vmax
    vmin = 0.0 if vmin is None else vmin
    fig, ax = fig_ax(ax, figsize)
    im = ax.imshow(amp.T, aspect="auto", origin="upper",
                   extent=[p_skm[0], p_skm[-1], taus[-1], taus[0]],
                   vmin=vmin, vmax=vmax, cmap=cmap, **mpl_kw)
    if sound_speed is not None:
        draw_slowness_line(ax, taus[-1], sound_speed)
    ax.set_title(title or "tau-p", loc="left")
    ax.set_xlabel("Slowness p [s/km]")
    ax.set_ylabel("Intercept time tau [s]")
    if show_colorbar:
        fig.colorbar(im, ax=ax, label="Stack amplitude")
    return fig, ax


# ── Spectral / level estimators (analysis) ──────────────────────────────────

def plot_psd(frequencies, psd_linear, ax=None, *, ref=REFERENCE_PRESSURE_WATER,
             label=None, ymin=0, ymax=150, title=None, figsize=(10, 6),
             **mpl_kw):
    """Line plot of a Welch PSD (dB). Consumes :func:`psd` output."""
    psd_db = power_to_db(np.asarray(psd_linear), ref)
    fig, ax = fig_ax(ax, figsize)
    ax.semilogx(frequencies, psd_db, label=label, **mpl_kw)
    ax.set_title(title or "Power spectral density", loc="left")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(f"Level [dB re {_ref_label(ref)}Pa²/Hz]")
    ax.set_ylim((ymin, ymax))
    ax.set_xlim((np.max((frequencies[0], 1)), frequencies[-1]))
    ax.grid(which="both", alpha=0.75)
    if label:
        ax.legend()
    return fig, ax


def plot_ppsd(result, ax=None, *, ymin=0, ymax=200, vmin=0, vmax=None,
              cmap="jet", title=None, figsize=(10, 6), show_colorbar=True,
              **mpl_kw):
    """2-D histogram of PSD levels. Consumes a ``PPSDResult``."""
    if vmax is None:
        vmax = 1 / result.binwidth_db
    fig, ax = fig_ax(ax, figsize)
    align = result.binwidth_db / 2
    pcm = ax.pcolormesh(result.frequencies, result.level_edges[:-1] + align,
                        result.pdf, cmap=cmap, shading="auto",
                        vmin=vmin, vmax=vmax, **mpl_kw)
    if show_colorbar:
        fig.colorbar(pcm, ax=ax,
                     label=f"Probability Density [{result.binwidth_db:.1f} dB/bin]")
    ax.plot(result.frequencies, result.mean_db, "k-", label="Mean level", lw=1.5)
    ax.plot(result.frequencies, result.mean_db + result.std_db, "k--",
            label="Mean level ± STD")
    ax.plot(result.frequencies, result.mean_db - result.std_db, "k--")
    ax.set_title(title or f"PPSD ({result.seg_duration}s)", loc="left")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Level [dB]")
    ax.set_xscale("log")
    ax.set_xlim((np.max((result.frequencies[0], 1)), result.frequencies[-1]))
    ax.set_ylim((ymin, ymax))
    ax.grid(which="both", alpha=0.5)
    ax.legend(loc="upper right")
    return fig, ax


def plot_sel(sel_pa2s, bands, ax=None, *, ref=REFERENCE_PRESSURE_WATER,
             duration=None, band_type="third_octave", ylim=(0, 200),
             title=None, figsize=(10, 6), **mpl_kw):
    """Bar plot of SEL per band (dB). Consumes :func:`sel` output."""
    fig, ax = fig_ax(ax, figsize)
    Fedges = [low for low, _, _ in bands] + [bands[-1][2]]
    width = [Fedges[i + 1] - Fedges[i] for i in range(len(Fedges) - 1)]
    ax.bar(Fedges[:-1], power_to_db(np.asarray(sel_pa2s), ref), width=width,
           align="edge", edgecolor="black", **mpl_kw)
    ax.set_title(title or f"SEL ({duration}s)", loc="left")
    ax.set_ylabel(f"Level [dB re {_ref_label(ref)}Pa²·s]")
    if band_type != "linear":
        ax.set_xscale("log")
    ax.set_xlabel(f"Frequency ({band_type}) [Hz]")
    ax.set_ylim(ylim)
    ax.grid(which="both", alpha=0.75)
    ax.set_axisbelow(True)
    return fig, ax


# ── Time-frequency (timefreq) ───────────────────────────────────────────────

def plot_spectrogram(frequencies, times, Sxx, ax=None, *,
                     ref=REFERENCE_PRESSURE_WATER, ymin=1, ymax=None, vmin=0,
                     vmax=200, cmap="jet", title=None, figsize=(10, 6),
                     show_colorbar=True, **mpl_kw):
    """Spectrogram colormap (dB). Consumes :func:`spectrogram` output."""
    Sxx_db = power_to_db(np.asarray(Sxx), ref)
    fig, ax = fig_ax(ax, figsize)
    pcm = ax.pcolormesh(times, frequencies, Sxx_db, cmap=cmap, shading="auto",
                        vmin=vmin, vmax=vmax, **mpl_kw)
    if show_colorbar:
        fig.colorbar(pcm, ax=ax, label=f"Level [dB re {_ref_label(ref)}Pa²/Hz]")
    ax.set_title(title or "Spectrogram", loc="left")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_ylim((ymin, frequencies[-1] if ymax is None else ymax))
    ax.grid(which="both", alpha=0.25, color="black")
    return fig, ax


# ── Constant-Q (Brown 1991) ─────────────────────────────────────────────────

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
        fig.colorbar(pcm, ax=ax, label=f"Level [dB re {unit}]")
    ax.set_title(title or "Constant-Q spectrogram", loc="left")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    ax.set_yscale("log")
    ax.set_ylim((frequencies[0], frequencies[-1]))
    ax.grid(which="both", alpha=0.25, color="black")
    return fig, ax


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
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(f"Level [dB re {unit}]")
    ax.set_ylim((ymin, ymax))
    ax.set_xlim((np.max((frequencies[0], 1)), frequencies[-1]))
    ax.grid(which="both", alpha=0.75)
    if label:
        ax.legend()
    return fig, ax


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
        vmax = 1 / result.binwidth_db
    fig, ax = fig_ax(ax, figsize)
    align = result.binwidth_db / 2
    pcm = ax.pcolormesh(result.frequencies, result.level_edges[:-1] + align,
                        result.pdf, cmap=cmap, shading="auto",
                        vmin=vmin, vmax=vmax, **mpl_kw)
    if show_colorbar:
        fig.colorbar(pcm, ax=ax,
                     label=f"Probability Density [{result.binwidth_db:.1f} dB/bin]")
    ax.plot(result.frequencies, result.mean_db, "k-", label="Mean level", lw=1.5)
    ax.plot(result.frequencies, result.mean_db + result.std_db, "k--",
            label="Mean level ± STD")
    ax.plot(result.frequencies, result.mean_db - result.std_db, "k--")
    ax.set_title(title or "Constant-Q PPSD", loc="left")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(f"Level [dB re {unit}]")
    ax.set_xscale("log")
    ax.set_xlim((np.max((result.frequencies[0], 1)), result.frequencies[-1]))
    ax.set_ylim((ymin, ymax))
    ax.grid(which="both", alpha=0.5)
    ax.legend(loc="upper right")
    return fig, ax


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
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    return fig, ax


def plot_wigner_ville(times, frequencies, W, ax=None, *, cmap="jet", title=None,
                      figsize=(10, 6), show_colorbar=True, **mpl_kw):
    """Wigner-Ville distribution image. Consumes :func:`wigner_ville` output
    ``(t, f, W)``."""
    fig, ax = fig_ax(ax, figsize)
    pcm = ax.pcolormesh(times, frequencies, np.real(np.asarray(W)), cmap=cmap,
                        shading="auto", **mpl_kw)
    if show_colorbar:
        fig.colorbar(pcm, ax=ax, label="WVD")
    ax.set_title(title or "Wigner-Ville", loc="left")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    return fig, ax


def plot_cepstrum(c, ax=None, *, sample_rate=None, title=None, figsize=(9, 4),
                  **mpl_kw):
    """Line plot of a cepstrum vs quefrency. Consumes :func:`cepstrum` output."""
    c = np.real(np.asarray(c))
    fig, ax = fig_ax(ax, figsize)
    if sample_rate is not None:
        q = np.arange(c.size) / float(sample_rate)
        ax.plot(q, c, **mpl_kw)
        ax.set_xlabel("Quefrency [s]")
    else:
        ax.plot(c, **mpl_kw)
        ax.set_xlabel("Quefrency [samples]")
    ax.set_ylabel("Amplitude")
    ax.set_title(title or "Cepstrum", loc="left")
    ax.grid(alpha=0.3)
    return fig, ax


# ── Decidecade bands / array spectra / ambiguity ────────────────────────────

def plot_band_levels(centers, levels, ax=None, *, title=None, width=0.8,
                     ref_label="1 µPa²", figsize=(9, 4), **mpl_kw):
    """Bar plot of decidecade band levels vs centre frequency. Consumes
    :func:`decidecade_band_levels` output."""
    c = np.asarray(centers, dtype=float)
    lv = np.asarray(levels, dtype=float)
    fig, ax = fig_ax(ax, figsize)
    x = np.log10(c)
    bw = width * np.median(np.diff(x)) if c.size > 1 else 0.04
    ax.bar(x, lv, width=bw, **mpl_kw)
    ticks = x[:: max(1, c.size // 12)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{v:.0f}" for v in 10 ** ticks], rotation=45)
    ax.set_xlabel("Decidecade band centre [Hz]")
    ax.set_ylabel(f"Band level [dB re {ref_label}]")
    ax.set_title(title or "Decidecade band levels", loc="left")
    ax.grid(alpha=0.3, axis="y")
    return fig, ax


def plot_angular_spectrum(angles_deg, spectrum, ax=None, *, db=True, label=None,
                          title=None, figsize=(8, 4), **mpl_kw):
    """Line plot of a beamformer angular spectrum (Bartlett/MVDR/MUSIC)."""
    P = np.real(np.asarray(spectrum))
    if db:
        P = 10.0 * np.log10(P / np.max(P))
    fig, ax = fig_ax(ax, figsize)
    ax.plot(angles_deg, P, label=label, **mpl_kw)
    ax.set_xlabel("Angle [deg]")
    ax.set_ylabel("Power [dB]" if db else "Power")
    ax.set_title(title or "Angular spectrum", loc="left")
    ax.grid(alpha=0.3)
    if label:
        ax.legend()
    return fig, ax


def plot_ambiguity(delays_s, doppler_hz, chi, ax=None, *, cmap="jet",
                   title=None, figsize=(8, 6), show_colorbar=True, **mpl_kw):
    """Range-Doppler ambiguity surface ``|chi|``. Consumes
    :func:`ambiguity_function` output."""
    amp = np.abs(np.asarray(chi))
    fig, ax = fig_ax(ax, figsize)
    im = ax.imshow(amp, aspect="auto", origin="lower",
                   extent=[delays_s[0] * 1e3, delays_s[-1] * 1e3,
                           doppler_hz[0], doppler_hz[-1]], cmap=cmap, **mpl_kw)
    if show_colorbar:
        fig.colorbar(im, ax=ax, label="|χ|")
    ax.set_title(title or "Ambiguity surface", loc="left")
    ax.set_xlabel("Delay [ms]")
    ax.set_ylabel("Doppler [Hz]")
    return fig, ax


# ── System identification (FRF) ─────────────────────────────────────────────

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
    ax1.plot(frequencies, 20 * np.log10(np.abs(tf)), label=lbl, **mpl_kw)
    ax1.set_title(title or "Frequency response", loc="left")
    ax1.set_ylabel("Magnitude [dB]")
    ax1.set_xscale("log")
    ax1.set_ylim((ymin, ymax))
    ax1.set_xlim((np.max((frequencies[0], 1)), frequencies[-1]))
    ax1.grid(which="both", alpha=0.5)
    ax2.plot(frequencies, np.angle(tf, deg=True), label=lbl, **mpl_kw)
    ax2.set_ylabel("Phase [degrees]")
    ax2.set_xlabel("Frequency [Hz]")
    ax2.set_xscale("log")
    ax2.set_ylim((-180, 180))
    ax2.set_xlim((np.max((frequencies[0], 1)), frequencies[-1]))
    ax2.grid(which="both", alpha=0.5)
    if lbl:
        ax1.legend()
        ax2.legend()
    return fig, (ax1, ax2)


def plot_coherence(frequencies, coh, ax=None, *, label=None, title=None,
                   figsize=(10, 4), **mpl_kw):
    """Coherence vs frequency. Consumes ``FRF`` ``(frequencies, coh)``."""
    fig, ax = fig_ax(ax, figsize)
    ax.plot(frequencies, coh, label=label, **mpl_kw)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Coherence")
    ax.set_xscale("log")
    ax.set_ylim((0.75, 1.01))
    ax.set_xlim((np.max((frequencies[0], 1)), frequencies[-1]))
    ax.grid(which="both", alpha=0.5)
    ax.set_title(title or "Coherence", loc="left")
    if label:
        ax.legend()
    return fig, ax


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
