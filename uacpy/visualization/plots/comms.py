"""Plots for the digital-communications toolkit (uacpy.comms).

All comms plotting lives here; ``uacpy.comms`` modules are pure computation and
do not import matplotlib. Each function consumes plain arrays, takes the target
``ax`` as its second positional argument (a new figure is made when it is
``None``), and returns ``(fig, ax)`` — the same convention as :func:`plot_field`.
"""
import numpy as np
import matplotlib.pyplot as plt

from uacpy.core.exceptions import ConfigurationError, FallbackWarning
from uacpy.visualization.plots._common import (
    fig_ax, typed_plot_error, _axes_pair, _carrier_or_arrays, _plot_warn,
    _refuse_spread_carrier, _require_nonempty, _title_or)




@typed_plot_error
def plot_channel(h, sample_rate=None, ax=None, *, delays_s=None, title=None,
                 freq_title=None, figsize=(12, 4), **mpl_kw):
    """Two-panel channel view: |h[n]| (delay) and |H(f)| (frequency response).

    Takes a :class:`~uacpy.comms.channel.ChannelTaps` — ``plot_channel(taps)``,
    which is the call ``taps.plot()`` makes — or the taps and their sample
    rate as arrays. From a carrier the rate is ``symbol_rate * sps``, which is
    not a field of it, and the delay axis is its ``delays_s``.

    **The delay axis is not ``arange(n) / fs``.** ``Arrivals.channel_taps``
    starts its grid ``span/2`` symbols before the first arrival, on the
    leading skirt of the transmit pulse, so an axis rebuilt from the index
    alone puts the first arrival at +80 ms on a span-8, 50 Bd channel instead
    of at 0 — the whole tap structure drawn one pulse-length late, with
    nothing on the figure to say so. ``delays_s`` places them; an array call
    that omits it gets the index axis, which is what taps with no delay
    reference have.

    ``ax`` may be a 2-tuple ``(ax_delay, ax_freq)``. ``title`` names the
    delay panel and ``freq_title`` the frequency one, one keyword per panel
    as :func:`uacpy.plot.plot_overview` spells it (``map_title`` /
    ``tl_title`` / ``env_title``); each falls back to its own default. A
    single ``title`` that is sometimes a string and sometimes a pair has no
    way to reject a 3-tuple except by letting the unpacking raise, which
    names neither the argument nor the fix.

    Parameters
    ----------
    h : ChannelTaps or array_like
        The channel, or its taps (then ``sample_rate`` is required).
    sample_rate : float, optional
        Tap rate (Hz) of array taps; a ChannelTaps carries its own.
    ax : Axes or (Axes, Axes), optional
        The ``(ax_delay, ax_freq)`` pair; a new two-panel figure when omitted.
    delays_s : array_like, optional
        Delay (s) of each tap; ``None`` is the carrier's, else the index axis.
    title : str, optional
        Delay-panel title. ``None`` draws the default caption.
    freq_title : str, optional
        Frequency-panel title. ``None`` draws the default caption.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(12, 4)`` by default; unused when
        ``ax`` is given.
    **mpl_kw
        Matplotlib keywords for the frequency-response line.
    """
    _unpacked = _carrier_or_arrays(
        h, (sample_rate,), count=1, who="plot_channel",
        carrier=(('ChannelTaps',), ('taps',)), fields=('taps',))
    if _unpacked is not None:
        carrier, (h,) = h, _unpacked
        sample_rate = float(carrier.symbol_rate) * int(carrier.sps)
        if delays_s is None:
            delays_s = carrier.delays_s
    elif sample_rate is None:
        raise ConfigurationError(
            "plot_channel: pass the ChannelTaps itself, plot_channel(taps), "
            "or the taps and their sample rate, plot_channel(h, fs).")
    for name, value in (('title', title), ('freq_title', freq_title)):
        if value is not None and not isinstance(value, str):
            # A pair would be str()'d into the delay panel — "('a', 'b')"
            # drawn as a title, which is worse than refusing it.
            raise ConfigurationError(
                f"plot_channel: {name} must be a string; got "
                f"{type(value).__name__}. One keyword per panel — pass "
                f"title= for the delay panel and freq_title= for the "
                f"frequency one."
            )
    # Outside the carrier branch, because the spread that lands here is
    # ``plot_channel(taps.taps, taps.delays_s, taps.symbol_rate)``: three
    # positionals, so the signature accepts it, ``sample_rate`` is not None
    # so the carrier branch is not taken, and the rate ends up in ``ax``,
    # where ``ax[0].figure`` raises about a float not being subscriptable.
    _refuse_spread_carrier(ax, "plot_channel", 'symbol_rate',
                           also="plot_channel(taps, sample_rate)")
    title_delay, title_freq = title, freq_title
    h = np.asarray(h, dtype=complex)
    fs = float(sample_rate)
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=figsize)
    else:
        fig = _axes_pair("plot_channel", ax, "(ax_delay, ax_freq)")[0].figure
    if delays_s is None:
        t = np.arange(h.size) / fs * 1e3
    else:
        t = np.asarray(delays_s, dtype=float) * 1e3
        if t.size != h.size:
            raise ConfigurationError(
                f"plot_channel: delays_s has {t.size} entries for "
                f"{h.size} taps; one delay per tap.")
    ax[0].stem(t, np.abs(h))
    ax[0].set_xlabel("Delay (ms)")
    ax[0].set_ylabel("|h|")
    ax[0].set_title(_title_or(title_delay, "Channel impulse response"))
    ax[0].grid(alpha=0.3)
    # Deferred into the body, like every compute-side import in this
    # module: at file scope it pulls uacpy.acoustic_signal — and scipy.signal
    # behind it — into every ``import uacpy.visualization``, which
    # test_importing_the_plotting_surface_leaves_the_comms_toolkit_unloaded
    # refuses.
    from uacpy.acoustic_signal.channel import channel_response
    # The transform is public, so a reader can get these numbers without
    # drawing them: it picks the two-sided FFT and the zero-padding, and
    # returns complex H. The dB floor is the plotter's, because it is a
    # drawing decision — it sets how deep a null is painted, and 20*log10 of
    # a perfect null is -inf, which no axis can show.
    f, H = channel_response(h, fs)
    ax[1].plot(f, 20 * np.log10(np.abs(H) + 1e-12), **mpl_kw)
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("|H(f)| (dB)")
    ax[1].set_title(_title_or(title_freq, "Frequency response"))
    ax[1].grid(alpha=0.3)
    return fig, ax


@typed_plot_error
def plot_doppler_ambiguity(scales, peak_metric, ax=None, *, title=None,
                           figsize=(7, 4), **mpl_kw):
    """Doppler-scale ambiguity curve (peak correlation vs scale).

    Parameters
    ----------
    scales : array_like
        The Doppler scales scanned (:func:`~uacpy.comms.estimate_doppler_scale`).
    peak_metric : array_like
        The peak matched-filter metric at each scale.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(7, 4)`` by default; unused when
        ``ax`` is given.
    **mpl_kw
        Matplotlib keywords for the curve.
    """
    s = np.asarray(scales, dtype=float)
    p = np.asarray(peak_metric, dtype=float)
    fig, ax = fig_ax(ax, figsize)
    ax.plot(s * 1e3, p / (p.max() + 1e-12), **mpl_kw)
    best = s[int(np.argmax(p))] * 1e3
    ax.axvline(best, color="r", ls="--", lw=1, label=f"a = {best:.2f} e-3")
    ax.set_xlabel("Doppler scale a (×10⁻³)")
    ax.set_ylabel("Norm. peak correlation")
    ax.set_title(_title_or(title, "Doppler ambiguity"))
    ax.grid(alpha=0.3)
    ax.legend()
    return fig, ax


@typed_plot_error
def plot_convergence(mse, ax=None, *, label=None, title=None, figsize=(7, 4),
                     **mpl_kw):
    """Equalizer learning curve (MSE vs symbol index, dB).

    Parameters
    ----------
    mse : array_like
        Squared error per symbol, linear (an equalizer's ``mse``).
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    label : str, optional
        Legend label of the line.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(7, 4)`` by default; unused when
        ``ax`` is given.
    **mpl_kw
        Matplotlib keywords for the curve.
    """
    _require_nonempty('plot_convergence', mse=mse)
    m = np.asarray(mse, dtype=float)
    fig, ax = fig_ax(ax, figsize)
    ax.plot(10 * np.log10(np.maximum(m, 1e-12)), label=label, **mpl_kw)
    ax.set_xlabel("Symbol index")
    ax.set_ylabel("MSE (dB)")
    ax.set_title(_title_or(title, "Equalizer convergence"))
    ax.grid(alpha=0.3)
    if label:
        ax.legend()
    return fig, ax


@typed_plot_error
def plot_sync_metric(metric, ax=None, *, threshold=None, title=None,
                     figsize=(8, 3.5), **mpl_kw):
    """Synchronization metric vs sample index.

    Parameters
    ----------
    metric : array_like
        The synchronization metric per sample.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    threshold : float, optional
        Draw the detection threshold at this level.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(8, 3.5)`` by default; unused when
        ``ax`` is given.
    **mpl_kw
        Matplotlib keywords for the metric line.
    """
    _require_nonempty('plot_sync_metric', metric=metric)
    m = np.asarray(metric, dtype=float)
    fig, ax = fig_ax(ax, figsize)
    ax.plot(m, **mpl_kw)
    if threshold is not None:
        ax.axhline(threshold, color="r", ls="--", lw=1,
                   label=f"threshold {threshold:g}")
        ax.legend()
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Norm. correlation")
    ax.set_title(_title_or(title, "Sync metric"))
    ax.grid(alpha=0.3)
    return fig, ax


@typed_plot_error
def plot_subcarriers(channel, n_subcarriers, ax=None, *, title=None,
                     figsize=(8, 3.5), **mpl_kw):
    """Channel magnitude across the OFDM subcarriers.

    Parameters
    ----------
    channel : array_like or ChannelTaps
        The channel impulse response.
    n_subcarriers : int
        Subcarriers of the OFDM grid.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(8, 3.5)`` by default; unused when
        ``ax`` is given.
    **mpl_kw
        Matplotlib keywords for the response line.
    """
    # Deferred into the body: a module-scope import here pulls the whole
    # comms toolkit into every ``import uacpy.visualization``, which is the
    # defect test_importing_the_plotting_surface_leaves_the_comms_toolkit_
    # unloaded exists to catch.
    from uacpy.comms.ofdm import subcarrier_response
    # comms owns the subcarrier grid, including the unshifted indexing
    # ofdm_modulate/ofdm_demodulate address k by, so the plotter asks for it
    # rather than repeating the DFT, so the plot and ofdm_demodulate's
    # equalizer read one expression, which has a public door.
    H = subcarrier_response(channel, n_subcarriers)
    nsc = H.size
    fig, ax = fig_ax(ax, figsize)
    # Unshifted, so index k is the subcarrier ``ofdm_modulate`` /
    # ``ofdm_demodulate`` address as k.
    ax.plot(np.arange(nsc),
            20 * np.log10(np.abs(H)), **mpl_kw)      # a zero bin is -inf, not -240 dB
    ax.set_xlabel("Subcarrier index")
    ax.set_ylabel("|H| (dB)")
    ax.set_title(_title_or(title, "OFDM subcarrier response"))
    ax.grid(alpha=0.3)
    return fig, ax


def _iq_axes(ax) -> None:
    """Furnish an I/Q plane: zero lines, equal aspect, grid, axis names."""
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("In-phase")
    ax.set_ylabel("Quadrature")


@typed_plot_error
def plot_scatter(symbols, ax=None, *, ideal=None, title=None, figsize=(5, 5),
                 **mpl_kw):
    """Constellation/scatter plot of received ``symbols``; optional ``ideal``
    constellation overlay.

    Parameters
    ----------
    symbols : array_like
        Received complex symbols.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    ideal : array_like, optional
        The constellation, overlaid as crosses.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(5, 5)`` by default; unused when
        ``ax`` is given.
    **mpl_kw
        Matplotlib keywords for the scatter (``s`` 6 and ``alpha`` 0.4 by default).
    """
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
    _iq_axes(ax)
    ax.set_title(_title_or(title, "Constellation"))
    return fig, ax


@typed_plot_error
def plot_constellation(constellation, ax=None, *, scheme="", annotate=True,
                       title=None, figsize=(5, 5), **mpl_kw):
    """Plot an ideal Gray-labeled constellation.

    Parameters
    ----------
    constellation : array_like
        The constellation points, indexed by bit label.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    scheme : str, optional
        Scheme name for the default title.
    annotate : bool, optional
        Label each point with its bits. Default True.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(5, 5)`` by default; unused when
        ``ax`` is given.
    **mpl_kw
        Matplotlib keywords for the points.
    """
    _require_nonempty('plot_constellation', constellation=constellation)
    c = np.asarray(constellation, dtype=complex)
    own_fig = ax is None
    fig, ax = fig_ax(ax, figsize)
    ax.scatter(c.real, c.imag, marker="o", s=60, **mpl_kw)
    if annotate and len(c) >= 2:
        bps = int(np.ceil(np.log2(len(c))))
        for label, pt in enumerate(c):
            ax.annotate(format(label, f"0{bps}b"), (pt.real, pt.imag),
                        textcoords="offset points", xytext=(6, 4), fontsize='small')
    _iq_axes(ax)
    ax.set_title(_title_or(title, f"{scheme} constellation".strip()))
    if own_fig:
        # Lay the plotter's own figure out around its labels: four-digit tick
        # labels need more left margin than the square panel's default.
        fig.tight_layout()
    return fig, ax


@typed_plot_error
def plot_eye_diagram(signal, samples_per_symbol, ax=None, *, n_symbols=2,
                     title=None, figsize=(7, 4), **mpl_kw):
    """Eye diagram: overlay ``n_symbols``-wide windows of the real signal.

    Parameters
    ----------
    signal : array_like
        The sampled signal; its real part is drawn.
    samples_per_symbol : int
        Samples per symbol.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    n_symbols : int, optional
        Symbols per overlaid window. Default 2.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(7, 4)`` by default; unused when
        ``ax`` is given.
    **mpl_kw
        Matplotlib keywords for every trace.
    """
    _require_nonempty('plot_eye_diagram', signal=signal)
    x = np.real(np.asarray(signal)).ravel()
    sps = int(samples_per_symbol)
    span = sps * int(n_symbols)
    fig, ax = fig_ax(ax, figsize)
    mpl_kw.setdefault("color", "#1f77b4")
    mpl_kw.setdefault("alpha", 0.15)
    mpl_kw.setdefault("lw", 0.7)
    t = np.arange(span) / sps
    # Floor division keeps n <= 0 for signals shorter than one window.
    n = (x.size - span) // sps + 1
    for k in range(max(n, 0)):
        ax.plot(t, x[k * sps: k * sps + span], **mpl_kw)
    ax.set_xlabel("Symbol intervals")
    ax.set_ylabel("Amplitude")
    ax.set_title(_title_or(title, "Eye diagram"))
    ax.grid(alpha=0.3)
    return fig, ax


@typed_plot_error
def plot_ber_curve(ebn0_dB, ber_measured, ax=None, *, scheme=None,
                   label="measured", n_bits=None, title=None, figsize=(7, 5),
                   **mpl_kw):
    """Measured BER vs Eb/N0 (semilog-y) with optional theory overlay.

    A point with zero observed errors measured no BER: the run only shows the
    rate is too low to resolve with the bits it sent. It is therefore not
    drawn on the measured line. Given ``n_bits`` (the bits per point, a
    scalar or one per point), each such point is marked with a downward
    caret at ``1/n_bits``, the smallest rate that run could have resolved;
    without it, the points are left out with a warning naming ``n_bits=``.

    Parameters
    ----------
    ebn0_dB : array_like
        Eb/N0 (dB) of each measured point.
    ber_measured : array_like
        Measured bit error rate at each point.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    scheme : str, optional
        Overlay :func:`~uacpy.comms.ber_theory` for this scheme.
    label : str, optional
        Legend label of the measured line. Default ``'measured'``.
    n_bits : int or array_like, optional
        Bits sent per point, to mark error-free points (see above).
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(7, 5)`` by default; unused when
        ``ax`` is given.
    **mpl_kw
        Matplotlib keywords for the measured line.
    """
    _require_nonempty('plot_ber_curve', ebn0_dB=ebn0_dB, ber_measured=ber_measured)
    ebn0 = np.atleast_1d(np.asarray(ebn0_dB, dtype=float))
    ber = np.atleast_1d(np.asarray(ber_measured, dtype=float))
    fig, ax = fig_ax(ax, figsize)
    mpl_kw.setdefault("marker", "o")
    error_free = ber == 0.0
    line, = ax.semilogy(ebn0[~error_free], ber[~error_free], label=label,
                        **mpl_kw)
    if np.any(error_free):
        if n_bits is None:
            _plot_warn(
                f"plot_ber_curve: {int(error_free.sum())} point(s) have zero "
                f"errors, which measure no BER, and are not drawn. Pass "
                f"n_bits= (bits sent per point) to mark each at the 1/n_bits "
                f"the run could resolve.", FallbackWarning)
        else:
            floor = 1.0 / np.broadcast_to(
                np.asarray(n_bits, dtype=float), ber.shape)[error_free]
            ax.semilogy(ebn0[error_free], floor, linestyle="none",
                        marker="v", color=line.get_color(),
                        markerfacecolor="none",
                        label=f"{label}: no errors (1/n_bits)")
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
    ax.set_title(_title_or(title, "Bit error rate"))
    ax.grid(which="both", alpha=0.3)
    ax.legend()
    return fig, ax
