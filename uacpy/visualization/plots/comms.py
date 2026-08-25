"""Plots for the digital-communications toolkit (uacpy.comms).

All comms plotting lives here; ``uacpy.comms`` modules are pure computation and
do not import matplotlib. Each function consumes plain arrays, takes the target
``ax`` as its second positional argument (a new figure is made when it is
``None``), and returns ``(fig, ax)`` — the same convention as :func:`plot_field`.
"""
import numpy as np
import matplotlib.pyplot as plt

from uacpy.visualization.plots._common import fig_ax, typed_plot_error




@typed_plot_error
def plot_channel(h, sample_rate, ax=None, *, title=None, figsize=(12, 4),
                 **mpl_kw):
    """Two-panel channel view: |h[n]| (delay) and |H(f)| (frequency response).
    ``ax`` may be a 2-tuple ``(ax_delay, ax_freq)``."""
    h = np.asarray(h, dtype=complex)
    fs = float(sample_rate)
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=figsize)
    else:
        fig = ax[0].figure
    t = np.arange(h.size) / fs * 1e3
    ax[0].stem(t, np.abs(h))
    ax[0].set_xlabel("Delay (ms)")
    ax[0].set_ylabel("|h|")
    ax[0].set_title(title or "Channel impulse response", loc="left")
    ax[0].grid(alpha=0.3)
    # h is complex (baseband IR): use the full FFT, not rfft (which rejects
    # complex input), and fftshift so |H(f)| is centred on 0 Hz.
    nfft = max(1024, 2 * h.size)
    f = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))
    H = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(h, nfft))) + 1e-12)
    ax[1].plot(f, H, **mpl_kw)
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("|H(f)| (dB)")
    ax[1].set_title("Frequency response", loc="left")
    ax[1].grid(alpha=0.3)
    return fig, ax


@typed_plot_error
def plot_doppler_ambiguity(scales, peak_metric, ax=None, *, title=None,
                           figsize=(7, 4), **mpl_kw):
    """Doppler-scale ambiguity curve (peak correlation vs scale)."""
    s = np.asarray(scales, dtype=float)
    p = np.asarray(peak_metric, dtype=float)
    fig, ax = fig_ax(ax, figsize)
    ax.plot(s * 1e3, p / (p.max() + 1e-12), **mpl_kw)
    best = s[int(np.argmax(p))] * 1e3
    ax.axvline(best, color="r", ls="--", lw=1, label=f"a = {best:.2f} e-3")
    ax.set_xlabel("Doppler scale a (×10⁻³)")
    ax.set_ylabel("Norm. peak correlation")
    ax.set_title(title or "Doppler ambiguity", loc="left")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig, ax


@typed_plot_error
def plot_convergence(mse, ax=None, *, label=None, title=None, figsize=(7, 4),
                     **mpl_kw):
    """Equalizer learning curve (MSE vs symbol index, dB)."""
    m = np.asarray(mse, dtype=float)
    fig, ax = fig_ax(ax, figsize)
    ax.plot(10 * np.log10(np.maximum(m, 1e-12)), label=label, **mpl_kw)
    ax.set_xlabel("Symbol index")
    ax.set_ylabel("MSE (dB)")
    ax.set_title(title or "Equalizer convergence", loc="left")
    ax.grid(alpha=0.3)
    if label:
        ax.legend()
    return fig, ax


@typed_plot_error
def plot_sync_metric(metric, ax=None, *, threshold=None, title=None,
                     figsize=(8, 3.5), **mpl_kw):
    """Synchronization metric vs sample index."""
    m = np.asarray(metric, dtype=float)
    fig, ax = fig_ax(ax, figsize)
    ax.plot(m, **mpl_kw)
    if threshold is not None:
        ax.axhline(threshold, color="r", ls="--", lw=1,
                   label=f"threshold {threshold:g}")
        ax.legend()
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Norm. correlation")
    ax.set_title(title or "Sync metric", loc="left")
    ax.grid(alpha=0.3)
    return fig, ax


@typed_plot_error
def plot_subcarriers(channel, n_subcarriers, ax=None, *, title=None,
                     figsize=(8, 3.5), **mpl_kw):
    """Channel magnitude across the OFDM subcarriers."""
    nsc = int(n_subcarriers)
    H = np.fft.fft(np.asarray(channel, dtype=complex), nsc)
    fig, ax = fig_ax(ax, figsize)
    # Unshifted, so index k is the subcarrier ``ofdm_modulate`` /
    # ``ofdm_demodulate`` address as k.
    ax.plot(np.arange(nsc),
            20 * np.log10(np.abs(H) + 1e-12), **mpl_kw)
    ax.set_xlabel("Subcarrier index")
    ax.set_ylabel("|H| (dB)")
    ax.set_title(title or "OFDM subcarrier response", loc="left")
    ax.grid(alpha=0.3)
    return fig, ax


@typed_plot_error
def plot_scatter(symbols, ax=None, *, ideal=None, title=None, figsize=(5, 5),
                 **mpl_kw):
    """Constellation/scatter plot of received ``symbols``; optional ``ideal``
    constellation overlay."""
    s = np.asarray(symbols, dtype=complex).ravel()
    fig, ax = fig_ax(ax, figsize)
    mpl_kw.setdefault("s", 6)
    mpl_kw.setdefault("alpha", 0.4)
    ax.scatter(s.real, s.imag, **mpl_kw)
    if ideal is not None:
        ideal = np.asarray(ideal, dtype=complex).ravel()
        ax.scatter(ideal.real, ideal.imag, marker="x", s=80, color="k",
                   zorder=5, label="ideal")
        ax.legend()
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("In-phase")
    ax.set_ylabel("Quadrature")
    ax.set_title(title or "Constellation", loc="left")
    return fig, ax


@typed_plot_error
def plot_constellation(constellation, ax=None, *, scheme="", annotate=True,
                       title=None, figsize=(5, 5), **mpl_kw):
    """Plot an ideal Gray-labeled constellation."""
    c = np.asarray(constellation, dtype=complex)
    fig, ax = fig_ax(ax, figsize)
    ax.scatter(c.real, c.imag, marker="o", s=60, **mpl_kw)
    if annotate and len(c) >= 2:
        bps = int(np.ceil(np.log2(len(c))))
        for label, pt in enumerate(c):
            ax.annotate(format(label, f"0{bps}b"), (pt.real, pt.imag),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("In-phase")
    ax.set_ylabel("Quadrature")
    ax.set_title(title or f"{scheme} constellation".strip(), loc="left")
    return fig, ax


@typed_plot_error
def plot_eye_diagram(signal, samples_per_symbol, ax=None, *, n_symbols=2,
                     title=None, figsize=(7, 4), **mpl_kw):
    """Eye diagram: overlay ``n_symbols``-wide windows of the real signal."""
    x = np.real(np.asarray(signal)).ravel()
    sps = int(samples_per_symbol)
    span = sps * int(n_symbols)
    fig, ax = fig_ax(ax, figsize)
    mpl_kw.setdefault("color", "#1f77b4")
    mpl_kw.setdefault("alpha", 0.15)
    mpl_kw.setdefault("lw", 0.7)
    t = np.arange(span) / sps
    n = (x.size - span) // sps
    for k in range(max(n, 0)):
        ax.plot(t, x[k * sps: k * sps + span], **mpl_kw)
    ax.set_xlabel("Symbol intervals")
    ax.set_ylabel("Amplitude")
    ax.set_title(title or "Eye diagram", loc="left")
    ax.grid(alpha=0.3)
    return fig, ax


@typed_plot_error
def plot_ber_curve(ebn0_db, ber_measured, ax=None, *, scheme=None,
                   label="measured", title=None, figsize=(7, 5), **mpl_kw):
    """Measured BER vs Eb/N0 (semilog-y) with optional theory overlay."""
    ebn0 = np.atleast_1d(np.asarray(ebn0_db, dtype=float))
    ber = np.atleast_1d(np.asarray(ber_measured, dtype=float))
    fig, ax = fig_ax(ax, figsize)
    mpl_kw.setdefault("marker", "o")
    # A run with zero observed errors gives BER = 0, which a log axis drops
    # silently; floor it so the point stays visible at the bottom of the plot.
    ax.semilogy(ebn0, np.maximum(ber, 1e-12), label=label, **mpl_kw)
    if scheme is not None:
        # In-function so importing the plotting surface does not drag the whole
        # comms toolkit (and scipy.signal) in behind it — the same rule
        # signal.py and noise.py follow for their compute imports.
        from uacpy.comms.metrics import ber_theory
        fine = np.linspace(ebn0.min(), ebn0.max(), 100)
        ax.semilogy(fine, ber_theory(scheme, fine), "k--",
                    label=f"{scheme} theory")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("BER")
    ax.set_title(title or "Bit error rate", loc="left")
    ax.grid(which="both", alpha=0.3)
    ax.legend()
    return fig, ax
