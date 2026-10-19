"""Plots for the digital-communications toolkit (uacpy.comms).

All comms plotting lives here; ``uacpy.comms`` modules are pure computation and
do not import matplotlib. Each function takes the arrays a comms routine
returns as its leading positional arguments, or the result record itself in
their place (a ``ChannelTaps``, ``ChannelResponse``, ``BerCurve``,
``BerTheory`` or ``DopplerEstimate``). The
target ``ax`` follows the data, so its position depends on how many arguments
there are — pass it by keyword; a new figure is made when it is ``None``. Each
returns ``(fig, ax)``, as :func:`plot_field` does.
"""
import numpy as np

from uacpy.core.exceptions import ConfigurationError, FallbackWarning
from uacpy.visualization.plots._common import (
    _VALUE_LABELS, fig_ax, typed_plot_error, _carrier_or_arrays, _floored_dB,
    _plot_warn, _refuse_spread_carrier, _require_nonempty, _title_or)


#: Decades the BER axis reaches below the lowest rate a run on the axes
#: resolved (its lowest measured BER, or 1/n_bits) when a theory curve is
#: overlaid (uacpy guidance): the closed-form curve keeps falling, tens of
#: decades past anything a run could measure.
_BER_AXIS_DECADES_BELOW = 1.0


def _ber_curve_label(curve) -> str:
    """The legend label a :class:`~uacpy.comms.BerCurve` names itself by:
    its scheme and what the link carried (a channel, the equalizer's and the
    code's types), so two curves on one axes read apart."""
    parts = [str(getattr(curve, 'scheme', None) or 'measured')]
    if getattr(curve, 'channel', None) is not None:
        parts.append('channel')
    for attribute in ('equalizer', 'code'):
        value = getattr(curve, attribute, None)
        if value is not None:
            parts.append(type(value).__name__)
    return ', '.join(parts)


@typed_plot_error
def plot_channel(h, sample_rate=None, ax=None, *, delays=None, title=None,
                 figsize=(7, 4)):
    """The channel taps ``|h|`` against delay, one stem per tap.

    Takes a :class:`~uacpy.comms.channel.ChannelTaps` — ``plot_channel(taps)``,
    which is the call ``taps.plot()`` makes — or the taps and their sample
    rate as arrays. From a carrier the delay axis is its ``delays``. The
    frequency response is its own result:
    ``uacpy.acoustic_signal.channel_response(taps).plot()``
    (:func:`plot_channel_response`).

    **The delay axis is not ``arange(n) / fs``.** ``Arrivals.channel_taps``
    starts its grid ``span/2`` symbols before the first arrival, on the
    leading skirt of the transmit pulse, so an axis rebuilt from the index
    alone puts the first arrival at +80 ms on a span-8, 50 Bd channel instead
    of at 0 — the whole tap structure drawn one pulse-length late, with
    nothing on the figure to say so. ``delays`` places them; an array call
    that omits it gets the index axis, which is what taps with no delay
    reference have.

    Parameters
    ----------
    h : ChannelTaps or array_like
        The channel, or its taps (then ``sample_rate`` is required).
    sample_rate : float, optional
        Tap rate (Hz) of array taps; a ChannelTaps carries its own
        (``symbol_rate * sps``).
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    delays : array_like, optional
        Delay (s) of each tap; ``None`` is the carrier's, else the index axis.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(7, 4)`` by default; unused when
        ``ax`` is given.
    """
    _unpacked = _carrier_or_arrays(
        h, (sample_rate,), count=1, who="plot_channel",
        carrier=(('ChannelTaps',), ('taps',)), fields=('taps',),
        slots=('sample_rate',))
    if _unpacked is not None:
        carrier, (h,) = h, _unpacked
        sample_rate = float(carrier.symbol_rate) * int(carrier.sps)
        if delays is None:
            delays = carrier.delays
    elif sample_rate is None:
        raise ConfigurationError(
            "plot_channel: pass the ChannelTaps itself, plot_channel(taps), "
            "or the taps and their sample rate, plot_channel(h, fs).")
    # Outside the carrier branch, because the spread that lands here is
    # ``plot_channel(taps.taps, taps.delays, taps.symbol_rate)``: three
    # positionals, so the signature accepts it, ``sample_rate`` is not None
    # so the carrier branch is not taken, and the rate ends up in ``ax``.
    _refuse_spread_carrier(ax, "plot_channel", 'symbol_rate',
                           also="plot_channel(taps, sample_rate)")
    h = np.asarray(h, dtype=complex)
    fs = float(sample_rate)
    fig, ax = fig_ax(ax, figsize)
    if delays is None:
        t = np.arange(h.size) / fs * 1e3
    else:
        t = np.asarray(delays, dtype=float) * 1e3
        if t.size != h.size:
            raise ConfigurationError(
                f"plot_channel: delays has {t.size} entries for "
                f"{h.size} taps; one delay per tap.")
    ax.stem(t, np.abs(h))
    ax.set_xlabel("Delay (ms)")
    ax.set_ylabel("|h|")
    ax.set_title(_title_or(title, "Channel impulse response"))
    ax.grid(alpha=0.3)
    return fig, ax


@typed_plot_error
def plot_channel_response(frequencies, response=None, ax=None, *,
                          dynamic_range_dB=None, title=None, figsize=(7, 4),
                          **mpl_kw):
    """``|H(f)|`` in dB of a baseband channel, two-sided about 0 Hz.

    Draws the :class:`~uacpy.acoustic_signal.ChannelResponse`
    :func:`uacpy.acoustic_signal.channel_response` returns —
    ``plot_channel_response(r)``, the call ``r.plot()`` makes — or its two
    arrays. The dB floor is this plotter's, a drawing decision: a null is
    drawn ``dynamic_range_dB`` below the peak.

    Parameters
    ----------
    frequencies : ChannelResponse or array_like
        The response, or its frequencies (Hz).
    response : array_like, optional
        The complex response at each frequency, with array input.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    dynamic_range_dB : float, optional
        Depth (dB) below the peak of ``|H(f)|`` at which the response is
        floored, a null drawn on the floor; ``None`` is 120 dB, as on
        :func:`plot_subcarriers`.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(7, 4)`` by default; unused when
        ``ax`` is given.
    **mpl_kw
        Matplotlib keywords for the line.
    """
    _unpacked = _carrier_or_arrays(
        frequencies, (response,), count=2, who="plot_channel_response",
        carrier=(('ChannelResponse',), ('frequencies', 'response')),
        fields=('frequencies', 'response'))
    if _unpacked is not None:
        frequencies, response = _unpacked
    elif response is None:
        raise ConfigurationError(
            "plot_channel_response: pass one ChannelResponse, "
            "plot_channel_response(channel_response(h, fs)), or both arrays, "
            "plot_channel_response(frequencies, response).")
    _require_nonempty('plot_channel_response', frequencies=frequencies,
                      response=response)
    fig, ax = fig_ax(ax, figsize)
    ax.plot(np.asarray(frequencies, dtype=float),
            _floored_dB(response, dynamic_range_dB=dynamic_range_dB,
                        who='plot_channel_response'), **mpl_kw)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(_VALUE_LABELS['level'])
    ax.set_title(_title_or(title, "Frequency response"))
    ax.grid(alpha=0.3)
    return fig, ax


@typed_plot_error
def plot_doppler_ambiguity(estimate, ax=None, *, title=None, figsize=(7, 4),
                           **mpl_kw):
    """Doppler-scale ambiguity curve (peak correlation vs scale), with the
    estimate marked.

    Parameters
    ----------
    estimate : DopplerEstimate
        What :func:`~uacpy.comms.estimate_doppler_scale` returns: the curve
        is its ``peak_metric`` over its ``scales``, and the dashed line marks
        its ``best_scale``, the estimator's own answer.
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
    if not (hasattr(estimate, '_fields') and tuple(estimate._fields) == (
            'best_scale', 'scales', 'peak_metric')):
        raise ConfigurationError(
            f"plot_doppler_ambiguity: needs the DopplerEstimate "
            f"estimate_doppler_scale returns; got "
            f"{type(estimate).__name__}.",
            remediation="Pass the estimate whole: "
                        "plot_doppler_ambiguity(estimate_doppler_scale(rx, "
                        "probe)), or call .plot() on it.")
    s = np.asarray(estimate.scales, dtype=float)
    p = np.asarray(estimate.peak_metric, dtype=float)
    best = float(estimate.best_scale) * 1e3
    fig, ax = fig_ax(ax, figsize)
    # Normalised to its own peak; an all-zero metric has none and is drawn
    # as it is.
    peak = float(np.max(p)) if p.size else 0.0
    ax.plot(s * 1e3, p / peak if peak > 0.0 else p, **mpl_kw)
    ax.axvline(best, color="r", ls="--", lw=1, label=f"a = {best:.2f} e-3")
    ax.set_xlabel("Doppler scale a (×10⁻³)")
    ax.set_ylabel("Norm. peak correlation")
    ax.set_title(_title_or(title, "Doppler ambiguity"))
    ax.grid(alpha=0.3)
    ax.legend()
    return fig, ax


@typed_plot_error
def plot_convergence(mse, ax=None, *, label=None, title=None, figsize=(7, 4),
                     dynamic_range_dB=None, **mpl_kw):
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
    dynamic_range_dB : float, optional
        Depth (dB) below the largest MSE at which the curve is floored, a
        zero error drawn on the floor; ``None`` is 120 dB.
    **mpl_kw
        Matplotlib keywords for the curve.
    """
    _require_nonempty('plot_convergence', mse=mse)
    m = np.asarray(mse, dtype=float)
    fig, ax = fig_ax(ax, figsize)
    ax.plot(_floored_dB(m, power=True, dynamic_range_dB=dynamic_range_dB,
                        who='plot_convergence'), label=label, **mpl_kw)
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
    own_fig = ax is None
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
    if own_fig:
        # Lay the plotter's own figure out around its labels, as
        # plot_constellation does: on the default figsize the x label
        # otherwise ends below the canvas.
        fig.tight_layout()
    return fig, ax


@typed_plot_error
def plot_subcarriers(response, ax=None, *, title=None, figsize=(8, 3.5),
                     dynamic_range_dB=None, **mpl_kw):
    """Channel magnitude ``|H[k]|`` (dB) across the OFDM subcarriers.

    Draws the per-subcarrier response it is handed — what
    :func:`uacpy.comms.subcarrier_response` returns for a channel, or a
    channel estimate — against the subcarrier index ``k``, unshifted, so
    ``k`` is the subcarrier ``ofdm_modulate`` / ``ofdm_demodulate`` address
    as ``k``::

        plot_subcarriers(comms.subcarrier_response(channel, 64))

    Parameters
    ----------
    response : array_like
        Complex (or magnitude) response, one value per subcarrier.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(8, 3.5)`` by default; unused when
        ``ax`` is given.
    dynamic_range_dB : float, optional
        Depth (dB) below the peak ``|H|`` at which the response is floored,
        a null subcarrier drawn on the floor; ``None`` is 120 dB, as on
        :func:`plot_channel_response`.
    **mpl_kw
        Matplotlib keywords for the response line.
    """
    if isinstance(ax, (int, np.integer)) and not isinstance(ax, bool):
        raise ConfigurationError(
            "plot_subcarriers: draws a per-subcarrier response it is handed, "
            f"not a channel and a subcarrier count (got {int(ax)} where the "
            "axes go).",
            remediation="Compute it first: plot_subcarriers("
                        "uacpy.comms.subcarrier_response(channel, "
                        "n_subcarriers)).")
    _require_nonempty('plot_subcarriers', response=response)
    H = np.asarray(response).ravel()
    fig, ax = fig_ax(ax, figsize)
    ax.plot(np.arange(H.size),
            _floored_dB(H, dynamic_range_dB=dynamic_range_dB,
                        who='plot_subcarriers'), **mpl_kw)
    ax.set_xlabel("Subcarrier index")
    ax.set_ylabel(_VALUE_LABELS['level'])
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
def plot_received_symbols(symbols, ax=None, *, ideal=None, title=None,
                          figsize=(5, 5), **mpl_kw):
    """The received ``symbols`` scattered on the I/Q plane; optional
    ``ideal`` constellation overlay.

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


#: The ``gid`` of a measured BER line (its points and its 1/n_bits carets)
#: and of a theory line, so the axis rule reads every curve already on the
#: axes, not only the one being drawn.
_MEASURED_BER_GID = 'uacpy-ber-measured'
_THEORY_BER_GID = 'uacpy-ber-theory'


def _bound_ber_axis(ax) -> None:
    """Rescale the BER axis to everything on ``ax``, its top at most 1 (a
    rate above 1 is not a value; the log axis' own margin over a theory
    curve many decades deep reaches 10¹); with a theory line on it, stop
    the axis one decade below the lowest rate any measured line on ``ax``
    resolved (its lowest BER, or a 1/n_bits caret). Reading every measured
    line, not the last one drawn, keeps a second curve from clipping the
    first."""
    ax.relim()
    ax.autoscale(enable=True, axis='y')
    ax.autoscale_view(scalex=False)
    if ax.get_ylim()[1] > 1.0:
        ax.set_ylim(top=1.0)
    if not any(line.get_gid() == _THEORY_BER_GID for line in ax.lines):
        return
    measured = [np.asarray(line.get_ydata(), dtype=float)
                for line in ax.lines if line.get_gid() == _MEASURED_BER_GID]
    rates = np.concatenate(measured) if measured else np.array([])
    rates = rates[np.isfinite(rates) & (rates > 0.0)]
    if rates.size:
        bottom = float(rates.min()) * 10.0 ** -_BER_AXIS_DECADES_BELOW
        ax.set_ylim(bottom=max(ax.get_ylim()[0], bottom))


@typed_plot_error
def plot_ber_curve(ebn0_dB, ber_measured=None, ax=None, *, label=None,
                   n_bits=None, title=None, figsize=(7, 5), **mpl_kw):
    """BER vs Eb/N0 (semilog-y): a measured curve, or a theory curve.

    Draws a :class:`~uacpy.comms.BerCurve` (``ber_sweep(...)``), its arrays,
    or a :class:`~uacpy.comms.BerTheory` (``comms.ber_theory_curve(scheme,
    ebn0_dB)``), which is drawn as a dashed black line. Each one's
    ``.plot()`` is the same call; ``ax=`` overlays them::

        fig, ax = comms.ber_sweep('qpsk', ebn0, 20000).plot()
        comms.ber_theory_curve('qpsk', np.linspace(0, 10, 100)).plot(ax=ax)

    A ``BerCurve`` supplies ``n_bits`` when ``n_bits=`` is unset. A point
    with zero observed errors measured no BER: the run only shows the rate is
    too low to resolve with the bits it sent. It is therefore not drawn on
    the measured line. Given ``n_bits`` (the bits per point, a scalar or one
    per point), each such point is marked with a downward caret at
    ``1/n_bits``, the smallest rate that run could have resolved; without it,
    the points are left out with a warning naming ``n_bits=``.

    A theory curve keeps falling past anything a run could measure, so with
    one on the axes the BER axis stops one decade below the lowest rate any
    measured curve on the axes resolved: its lowest measured BER, or
    ``1/n_bits`` when that is lower.

    Parameters
    ----------
    ebn0_dB : BerCurve, BerTheory or array_like
        The curve itself, or the Eb/N0 (dB) of each measured point.
    ber_measured : array_like, optional
        Measured bit error rate at each point, with array input.
    ax : matplotlib.axes.Axes, optional
        Existing axes; a new figure is made when omitted.
    label : str, optional
        Legend label of the line. ``None`` names a ``BerCurve`` by its
        scheme and what its link carried (``'qpsk, channel, DFE'``), a
        ``BerTheory`` ``'qpsk theory'``, and bare arrays ``'measured'``.
    n_bits : int or array_like, optional
        Bits sent per point, to mark error-free points (see above).
    title : str, optional
        Axes title. ``None`` draws the default caption; ``''`` draws none.
    figsize : tuple, optional
        Size (inches) of the new figure, ``(7, 5)`` by default; unused when
        ``ax`` is given.
    **mpl_kw
        Matplotlib keywords for the line.
    """
    given = ebn0_dB
    _unpacked = _carrier_or_arrays(
        ebn0_dB, (ber_measured,), count=2, who="plot_ber_curve",
        carrier=(('BerCurve', 'BerTheory'), ('ebn0_dB', 'ber')),
        fields=('ebn0_dB', 'ber_measured'))
    is_theory = _unpacked is not None and 'BerTheory' in {
        klass.__name__ for klass in type(given).__mro__}
    if is_theory:
        theory = _unpacked
        if n_bits is not None:
            raise ConfigurationError(
                "plot_ber_curve: n_bits= marks the error-free points of a "
                "measured curve; a BerTheory is computed, not counted.",
                remediation="Drop n_bits=, or pass it with the measured "
                            "curve.")
        _require_nonempty('plot_ber_curve', ebn0_dB=theory[0], ber=theory[1])
        fig, ax = fig_ax(ax, figsize)
        mpl_kw.setdefault('color', 'k')
        mpl_kw.setdefault('linestyle', '--')
        ax.semilogy(np.asarray(theory[0], dtype=float),
                    np.asarray(theory[1], dtype=float),
                    label=f"{given.scheme} theory" if label is None else label,
                    gid=_THEORY_BER_GID, **mpl_kw)
    else:
        if _unpacked is not None:
            ebn0_dB, ber_measured = _unpacked
            if n_bits is None:
                n_bits = getattr(given, 'n_bits', None)
            if label is None:
                label = _ber_curve_label(given)
        elif ber_measured is None:
            raise ConfigurationError(
                "plot_ber_curve: pass both arrays, plot_ber_curve(ebn0_dB, "
                "ber_measured), or one curve, plot_ber_curve(ber_sweep(...)) "
                "or plot_ber_curve(comms.ber_theory_curve(scheme, ebn0_dB)).")
        _require_nonempty('plot_ber_curve', ebn0_dB=ebn0_dB,
                          ber_measured=ber_measured)
        ebn0 = np.atleast_1d(np.asarray(ebn0_dB, dtype=float))
        ber = np.atleast_1d(np.asarray(ber_measured, dtype=float))
        label = 'measured' if label is None else label
        fig, ax = fig_ax(ax, figsize)
        mpl_kw.setdefault("marker", "o")
        error_free = ber == 0.0
        line, = ax.semilogy(ebn0[~error_free], ber[~error_free], label=label,
                            gid=_MEASURED_BER_GID, **mpl_kw)
        if np.any(error_free):
            if n_bits is None:
                _plot_warn(
                    f"plot_ber_curve: {int(error_free.sum())} point(s) have "
                    f"zero errors, which measure no BER, and are not drawn. "
                    f"Pass n_bits= (bits sent per point) to mark each at the "
                    f"1/n_bits the run could resolve.", FallbackWarning)
            else:
                floor = 1.0 / np.broadcast_to(
                    np.asarray(n_bits, dtype=float), ber.shape)[error_free]
                ax.semilogy(ebn0[error_free], floor, linestyle="none",
                            marker="v", color=line.get_color(),
                            markerfacecolor="none", gid=_MEASURED_BER_GID,
                            label=f"{label}: no errors (1/n_bits)")
    _bound_ber_axis(ax)
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("BER")
    ax.set_title(_title_or(title, "Bit error rate"))
    ax.grid(which="both", alpha=0.3)
    ax.legend()
    return fig, ax
