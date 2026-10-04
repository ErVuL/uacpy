"""OFDM, both halves: the subcarrier modulator and demodulator, the
Schmidl-Cox preamble and timing metric, carrier-frequency-offset removal, and
the pilot channel estimate and one-tap subcarrier equalisation."""

from __future__ import annotations

import warnings

import numpy as np

from uacpy.comms.channel import _channel_array
from uacpy.comms.equalize import regularizer
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.exceptions import ConfigurationError, NumericsWarning


# Smallest fraction of the peak pilot magnitude a subcarrier can carry and
# still define a channel estimate. Relative because |pilot| carries whatever
# amplitude scale the caller's constellation is in: an absolute floor silently
# makes the result a function of those units. `receive._ZF_REL_FLOOR`
# is the same rule for |H|^2 and `acoustic_signal.frf._etfe_divide` for a
# transfer function.
_PILOT_REL_FLOOR = 1e-12


# Fraction of the channel's total energy beyond the cyclic prefix above which
# ofdm_demodulate warns of inter-block interference (ISI-to-signal ~ -20 dB).
# Relative to the channel's own energy, so the verdict is independent of its
# amplitude units and of its representation length: a full-length
# np.fft.ifft(H_est) of a channel whose support fits the prefix carries only
# numerical noise past cp+1 taps and passes silently.
_ISI_TAIL_REL_ENERGY = 0.01


def _require_subcarrier_count(n_subcarriers, who: str) -> int:
    """Validate ``n_subcarriers >= 1`` and return it as ``int``.

    Every entry point here divides or reshapes by this count: zero reaches
    ``%`` as ``integer modulo by zero``, a negative one reaches ``reshape``
    or ``np.fft`` as ``negative dimensions are not allowed`` — untyped errors
    naming no argument the caller passed.
    """
    nsc = int(n_subcarriers)
    if nsc < 1:
        raise ConfigurationError(
            f"{who}: n_subcarriers must be >= 1; got {n_subcarriers!r}. "
            f"It is the FFT length of one OFDM block, so every block, cyclic "
            f"prefix and subcarrier index is measured in it.")
    return nsc


def _require_cp_len(cp_len, nsc: int, who: str) -> int:
    """Validate ``0 <= cp_len <= n_subcarriers`` and return it as ``int``.

    A negative cp_len is as wrong as an over-long one and quieter: the
    modulator's ``time[:, nsc - cp:]`` slice returns fewer samples than a
    block, so the emitted signal carries no cyclic prefix and every later
    block is mis-framed by the demodulator's nsc + cp_len stride; the
    demodulator's ``[:, cp:]`` slice keeps the last ``|cp|`` columns of every
    mis-framed block and returns the wrong number of garbage symbols; and
    ``estimate_channel``'s ``rx[cp:cp + nsc]`` slice goes empty and reaches
    ``np.fft`` as a zero-point-FFT ValueError naming nothing the caller
    passed.
    """
    cp = int(cp_len)
    if not 0 <= cp <= nsc:
        raise ConfigurationError(
            f"{who}: cp_len ({cp_len}) must satisfy "
            f"0 <= cp_len <= n_subcarriers ({nsc})"
        )
    return cp


def ofdm_modulate(symbols, n_subcarriers, cp_len):
    """Map symbols onto ``n_subcarriers`` and prepend a cyclic prefix.

    Symbols are zero-padded to a whole number of OFDM blocks. Returns the
    complex time-domain signal (blocks concatenated).

    Parameters
    ----------
    symbols : array_like
        Subcarrier symbols, filled block by block.
    n_subcarriers : int
        Subcarriers per OFDM block (the FFT length).
    cp_len : int
        Cyclic-prefix length in samples.
    """
    s = np.asarray(symbols, dtype=complex).ravel()
    nsc = _require_subcarrier_count(n_subcarriers, "ofdm_modulate")
    cp_len = _require_cp_len(cp_len, nsc, "ofdm_modulate")
    if s.size % nsc:
        s = np.concatenate([s, np.zeros(nsc - s.size % nsc, dtype=complex)])
    blocks = s.reshape(-1, nsc)
    time = np.fft.ifft(blocks, axis=1) * np.sqrt(nsc)   # unit-energy per subcarrier
    cp = time[:, nsc - cp_len:]
    return np.concatenate([cp, time], axis=1).ravel()


def ofdm_demodulate(received, n_subcarriers, cp_len, channel=None, snr_linear=None,
                    *, channel_response=None):
    """Recover symbols: strip CP, FFT, and optionally equalize per subcarrier.

    The channel is given one of two ways, never both:

    ``channel``
        The channel **impulse response** — its taps, as
        :func:`multipath_channel` or :meth:`Arrivals.channel_taps` give them.
        It is transformed to each subcarrier's response
        (:func:`subcarrier_response`).
    ``channel_response``
        The channel **frequency response** on the subcarrier grid, one value
        per subcarrier — what :func:`estimate_channel` returns from pilots.
        Used as it is.

    Each subcarrier is then divided by its response (zero-forcing), or by
    the MMSE weight when ``snr_linear`` is also set. Handing a frequency
    response to ``channel`` transforms it a second time: measured on a
    4-tap 16-QAM link, symbol error rate 0.66 against 0 as
    ``channel_response``. Note that with the hard-decision slicer this package decodes with,
    the per-subcarrier MMSE output is the zero-forcing output times a positive
    real factor ``|H|^2 / (|H|^2 + N/S)``: every PSK decision is identical and
    QAM decisions are slightly WORSE (the estimate is biased toward zero, so
    outer points fall inward). It earns its keep only with soft decisions or
    bias removal, which this receiver does not do. Returns the flat complex
    symbol array.

    A channel longer than ``n_subcarriers`` raises. One carrying more than
    1 % of its energy in the taps beyond the cyclic prefix (ISI-to-signal
    above about -20 dB) is accepted with a ``NumericsWarning`` — an under-CP
    study is legitimate but the result carries inter-block interference no
    equalizer or SNR removes. The criterion is the tail's *energy*, not the
    tap count, so a full-length representation of a short channel (e.g.
    ``np.fft.ifft(H_est)``) passes silently.

    A multipath channel can put a spectral null on a subcarrier. Both equalizers
    are written as ``conj(H)/(|H|^2 + eps)`` so such a subcarrier comes back as
    zero rather than inf/NaN; for MMSE ``eps`` is the physical noise-to-signal
    ratio ``mean(|H|^2)/snr_linear``, for zero-forcing it is only a floor at
    ``1e-12`` of the peak ``|H|^2``, and the subcarrier is unrecoverable either
    way. Both scale with the channel, so equalizing the same link with the
    channel expressed in any amplitude unit gives the same symbols; a channel
    normalized to unit peak (ZF) or unit mean (MMSE) power reproduces the
    plain ``1e-12`` / ``1/snr_linear`` offsets exactly.

    Parameters
    ----------
    received : array_like
        The received CP-prefixed blocks, starting on a block boundary.
    n_subcarriers : int
        Subcarriers per OFDM block (the FFT length).
    cp_len : int
        Cyclic-prefix length in samples.
    channel : array_like or ChannelTaps, optional
        The channel impulse response (see above).
    snr_linear : float, optional
        SNR at the equalizer input, a linear power ratio; set, it selects the
        MMSE weight.
    channel_response : array_like, optional
        The channel frequency response on the subcarrier grid (see above).
    """
    nsc = _require_subcarrier_count(n_subcarriers, "ofdm_demodulate")
    cp = _require_cp_len(cp_len, nsc, "ofdm_demodulate")
    r = np.asarray(received, dtype=complex).ravel()
    blk = nsc + cp
    nblocks = r.size // blk
    if nblocks == 0:
        raise ConfigurationError(
            f"ofdm_demodulate: signal shorter than one block — received holds "
            f"{r.size} samples, and one block is n_subcarriers + cp_len = "
            f"{nsc} + {cp} = {blk}.")
    grid = r[: nblocks * blk].reshape(nblocks, blk)[:, cp:]
    freq = np.fft.fft(grid, axis=1) / np.sqrt(nsc)
    if channel is not None and channel_response is not None:
        raise ConfigurationError(
            "ofdm_demodulate: pass channel= (impulse-response taps) or "
            "channel_response= (per-subcarrier H), not both.")
    if channel_response is not None:
        Hk = np.asarray(channel_response, dtype=complex).ravel()
        if Hk.size != nsc:
            raise ConfigurationError(
                f"ofdm_demodulate: channel_response has {Hk.size} values; "
                f"it is one response per subcarrier, {nsc} here.",
                remediation="Pass estimate_channel()'s output, or the taps "
                            "as channel=.")
        return equalize_subcarriers(freq, Hk, snr_linear).ravel()
    if channel is not None:
        hc = np.asarray(_channel_array(channel), dtype=complex).ravel()
        if hc.size > nsc:
            # The length-nsc transform would truncate the channel, equalizing
            # a shorter one than the caller described; and a channel longer
            # than a block breaks the cyclic-prefix assumption outright.
            raise ConfigurationError(
                f"ofdm_demodulate: channel has {hc.size} taps, longer than the "
                f"{nsc} subcarriers, so the length-{nsc} transform would drop "
                f"its tail. One-tap-per-subcarrier equalization needs a channel "
                f"no longer than the cyclic prefix ({cp} samples)."
            )
        total = float(np.sum(np.abs(hc) ** 2))
        tail = float(np.sum(np.abs(hc[cp + 1:]) ** 2))
        if tail > _ISI_TAIL_REL_ENERGY * total:
            # The equalization itself still runs — an under-CP study is a
            # legitimate experiment — but the result carries inter-block
            # interference, an error floor no SNR removes.
            warnings.warn(
                f"ofdm_demodulate: {tail / total:.2%} of the channel energy "
                f"lies in the taps beyond the {cp}-sample cyclic prefix, so "
                f"each block's convolution tail outlives the prefix and "
                f"leaks inter-block interference into the next block. If "
                f"this channel is a frequency response (estimate_channel's "
                f"output, one value per subcarrier), pass it as "
                f"channel_response= instead: channel= takes the taps.",
                NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
        freq = equalize_subcarriers(
            freq, subcarrier_response(hc, nsc), snr_linear)
    return freq.ravel()


def subcarrier_response(channel, n_subcarriers):
    """Channel response ``H`` sampled on the OFDM subcarrier grid.

    ``H[k]`` is the gain the subcarrier :func:`ofdm_modulate` and
    :func:`ofdm_demodulate` address as ``k``: an ``n_subcarriers``-point DFT
    of the impulse response, **unshifted**, so the index is the subcarrier
    number and not a position on a two-sided frequency axis. That is what
    separates this from
    :func:`uacpy.acoustic_signal.channel_response`, which centres its grid on
    0 Hz and returns a frequency axis with it — the right answer for a
    spectrum, the wrong indexing for a subcarrier.

    Public because it is the equalizer's input: ``ofdm_demodulate(...,
    channel=h)`` computes it internally, and a caller who wants to see which
    subcarriers the channel has nulled — or to supply an estimate rather than
    an impulse response — needs the same grid, by the same convention.

    Parameters
    ----------
    channel : array_like or ChannelTaps
        The channel impulse response, no longer than ``n_subcarriers``.
    n_subcarriers : int
        Subcarriers per OFDM block (the FFT length).
    """
    nsc = _require_subcarrier_count(n_subcarriers, "subcarrier_response")
    h = np.asarray(_channel_array(channel), dtype=complex).ravel()
    if h.size == 0:
        raise ConfigurationError(
            "subcarrier_response: channel is empty, so there is no response "
            "to sample.")
    if h.size > nsc:
        raise ConfigurationError(
            f"subcarrier_response: the channel is {h.size} taps and the grid "
            f"is {nsc} subcarriers, so the DFT would alias the tail back over "
            f"the head rather than truncate it. Use at least {h.size} "
            f"subcarriers, or shorten the channel.")
    return np.fft.fft(h, nsc)


def equalize_subcarriers(spectra, H, snr_linear=None):
    """One-tap-per-subcarrier equalisation of block spectra ``spectra``
    (``(..., n_subcarriers)``) by the channel response ``H``: the
    ``conj(H)/(|H|^2 + eps)`` form with ``eps`` from :func:`regularizer` —
    zero-forcing with a floor, or the MMSE weight when ``snr_linear`` is
    given. A channel with no power anywhere returns zeros: every subcarrier
    is unrecoverable, which is what the epsilon form tends to. ``eps`` scales
    with ``|H|^2`` so a pilot estimate at any receive amplitude is used: with
    a fixed offset, 16-QAM over a 4-tap channel measured BER 0.24 at an
    amplitude of 1e-12 (MMSE at 1e-9).

    Parameters
    ----------
    spectra : ndarray
        Block spectra, ``(..., n_subcarriers)``.
    H : ndarray
        The channel response per subcarrier.
    snr_linear : float, optional
        SNR at the equalizer input, a linear power ratio; ``None`` is
        zero-forcing.
    """
    h2 = np.abs(H) ** 2
    eps = regularizer(h2, snr_linear)
    if eps <= 0.0:
        return np.zeros_like(spectra)
    return spectra * (np.conj(H) / (h2 + eps))


def ofdm_symbol(subcarrier_values, n_subcarriers, cp_len):
    """One CP-prefixed OFDM time-domain symbol from a length-``n_subcarriers``
    spectrum ``subcarrier_values`` (the subcarrier values).

    Parameters
    ----------
    subcarrier_values : array_like
        The ``n_subcarriers`` subcarrier values.
    n_subcarriers : int
        Subcarriers per OFDM block (the FFT length).
    cp_len : int
        Cyclic-prefix length in samples.
    """
    nsc = _require_subcarrier_count(n_subcarriers, "ofdm_symbol")
    cp = _require_cp_len(cp_len, nsc, "ofdm_symbol")
    t = np.fft.ifft(subcarrier_values) * np.sqrt(nsc)
    return np.concatenate([t[nsc - cp:], t])


def schmidl_cox_preamble(n_subcarriers, cp_len, seed=0x5C0FFEE):
    """Schmidl & Cox training symbol — two identical time-domain halves.

    Even subcarriers carry a PN-QPSK sequence, odd subcarriers are null, so the
    IFFT produces two identical halves of length ``n_subcarriers/2``. Returns the
    CP-prefixed complex time-domain preamble.

    Parameters
    ----------
    n_subcarriers : int
        Subcarriers per OFDM block (the FFT length), even.
    cp_len : int
        Cyclic-prefix length in samples.
    seed : int, optional
        Seed of the PN-QPSK sequence; the receiver needs the same one.
    """
    nsc = _require_subcarrier_count(n_subcarriers, "schmidl_cox_preamble")
    cp_len = _require_cp_len(cp_len, nsc, "schmidl_cox_preamble")
    if nsc % 2:
        # Loading only the even subcarriers of an odd-length FFT does not
        # produce two identical time-domain halves, so schmidl_cox_sync's
        # metric never reaches its plateau and the preamble is undetectable.
        raise ConfigurationError(
            f"schmidl_cox_preamble: n_subcarriers must be even; got {nsc}.")
    rng = np.random.default_rng(seed)
    freq = np.zeros(nsc, dtype=complex)
    even = np.arange(0, nsc, 2)
    # Schmidl & Cox 1997 sec. III-A: "the frequency components of this training
    # symbol are multiplied by sqrt(2) at the transmitter" — it compensates for
    # loading only the even subcarriers, so the block carries the same total
    # energy (nsc) as a fully loaded one.
    freq[even] = np.exp(1j * np.pi / 2 * rng.integers(0, 4, even.size)) * np.sqrt(2)
    return ofdm_symbol(freq, nsc, cp_len)


#: Fraction of the peak sliding-window energy R(d) below which the
#: Schmidl-Cox timing metric is not evaluated (set to 0): a normalised ratio
#: over a near-silent window takes arbitrary values that would dwarf the
#: preamble plateau. Relative to the record's own peak, so the metric does
#: not depend on the units the record is held in.
_SC_ENERGY_GATE = 0.25


def _schmidl_cox_terms(r, nsc):
    """``(metric, P)`` of Schmidl & Cox 1997 for ``d = 0 .. len(r) - nsc - 1``;
    both empty when the record is shorter than one symbol."""
    L = nsc // 2
    n = r.size - 2 * L
    if n <= 0:
        return np.zeros(0), np.zeros(0, dtype=complex)
    # Schmidl & Cox 1997 eqs. (5) and (7): P(d) = sum conj(r[d+m]) r[d+m+L]
    # and R(d) = sum |r[d+m+L]|^2 — both length-L sliding sums, O(n) here via
    # cumulative sums rather than the paper's iterative form (6).
    a = np.conj(r[:-L]) * r[L:]
    ca = np.concatenate(([0.0 + 0.0j], np.cumsum(a)))
    p = ca[L:L + n] - ca[:n]
    energy = np.abs(r[L:]) ** 2
    ce = np.concatenate(([0.0], np.cumsum(energy)))
    rr = ce[L:L + n] - ce[:n]
    metric = np.zeros(n)
    rr_max = float(rr.max())
    if rr_max <= 0.0:
        return metric, p
    # The energy gate runs first and the division is done only where it passes,
    # so the metric needs no epsilon. An absolute one is wrong here anyway:
    # rr**2 scales as amplitude**4, so `rr**2 + 1e-12` made M(d) a function of
    # the units the caller held the record in — a noise-free frame synced in
    # µPa and returned (None, 0.0) in Pa, below ~5e-4 amplitude. The same
    # reasoning is written out at `acoustic_signal.frf._etfe_divide`.
    loud = rr >= _SC_ENERGY_GATE * rr_max
    # Timing metric M(d) = |P(d)|^2 / R(d)^2, eq. (8).
    metric[loud] = np.abs(p[loud]) ** 2 / rr[loud] ** 2
    return metric, p


def schmidl_cox_metric(received, n_subcarriers):
    """Schmidl & Cox timing metric ``M(d) = |P(d)|^2 / R(d)^2`` over a record.

    ``P`` and ``R`` are the length-``n_subcarriers/2`` sliding correlation and
    energy sums of Schmidl & Cox (1997) eqs. (5), (7) and (8); two exactly
    identical halves give ``M = 1``, and the preamble shows as a plateau as
    long as its cyclic prefix. Entry ``d`` scores the window starting at
    sample ``d``, for ``d = 0 .. len(received) - n_subcarriers - 1`` (empty when the
    record is shorter than one symbol). Windows whose energy ``R(d)`` is below
    a quarter of the record's peak ``R`` read 0: the ratio there is not
    meaningful, and the gate is relative, so the metric is the same whatever
    units the record is in. This is the curve :func:`schmidl_cox_sync`
    searches.

    Parameters
    ----------
    received : array_like
        The record to score.
    n_subcarriers : int
        Subcarriers per OFDM block (the FFT length).
    """
    r = np.asarray(received, dtype=complex).ravel()
    nsc = _require_subcarrier_count(n_subcarriers, "schmidl_cox_metric")
    return _schmidl_cox_terms(r, nsc)[0]


#: Fraction of the timing-metric peak :func:`schmidl_cox_sync` walks left
#: to from its argmax.
_SC_EDGE_FRACTION = 0.9


def _schmidl_cox_backoff(n_subcarriers):
    """Samples :func:`schmidl_cox_sync` returns ahead of the frame boundary
    on a clean frame: the metric ramps as ``((L-k)/L)**2`` into its plateau,
    ``L = n_subcarriers/2``, and crosses :data:`_SC_EDGE_FRACTION` of the
    peak at ``k = (1 - sqrt(0.9))·L``, about ``L/20 = n_subcarriers/40``."""
    return int(np.ceil((1.0 - np.sqrt(_SC_EDGE_FRACTION))
                       * n_subcarriers / 2.0))


def schmidl_cox_sync(received, n_subcarriers, threshold=0.5):
    """Locate the Schmidl & Cox preamble and estimate the fractional CFO.

    Returns ``(start, cfo)`` — ``start`` is the index of the preamble's cyclic
    prefix; ``cfo`` the normalized carrier frequency offset (cycles/sample) from
    the half-symbol phase. ``start`` is ``None`` when the peak of the timing
    metric (:func:`schmidl_cox_metric`) is below ``threshold``. The metric
    depends only on the two identical ``n_subcarriers/2`` halves, so the
    cyclic-prefix length plays no part in the search.

    Noise lowers the plateau: with per-sample SNR ``g`` each half carries
    signal power ``g/(1+g)`` of its total, so the plateau sits near
    ``(g/(1+g))**2``, 0.25 at 0 dB. The default ``threshold=0.5`` therefore
    loses frames below a few dB; measured over 20 frames at 256
    subcarriers it found 0/20 at 0 dB, 11/20 at 2 dB and 20/20 at 4 dB. Lower
    it to sync a coded link that decodes below that, keeping it above the
    peak a noise-only record of the same length reaches: 0.07 at most over
    20 white-noise records of 3000 samples at 256 subcarriers (measured).

    Parameters
    ----------
    received : array_like
        The record to search.
    n_subcarriers : int
        Subcarriers per OFDM block (the FFT length).
    threshold : float, optional
        Lowest timing-metric peak taken as a preamble, in ``(0, 1]``.
        Default 0.5.
    """
    r = np.asarray(received, dtype=complex).ravel()
    nsc = _require_subcarrier_count(n_subcarriers, "schmidl_cox_sync")
    thr = float(threshold)
    if not 0.0 < thr <= 1.0:
        raise ConfigurationError(
            f"schmidl_cox_sync: threshold must be in (0, 1], the range of "
            f"the timing metric; got {threshold!r}.")
    metric, p = _schmidl_cox_terms(r, nsc)
    if metric.size == 0 or not np.any(metric):
        return None, 0.0
    peak = int(np.argmax(metric))
    # Two exactly identical halves give |P| = R and hence M = 1; a peak below
    # `thr` is not taken as a preamble.
    if metric[peak] < thr:
        return None, 0.0
    # The two halves are L = nsc/2 samples apart, so P accumulates a phase of
    # 2*pi*cfo*L = pi*cfo*nsc; unambiguous only for |cfo| < 1/nsc.
    cfo = np.angle(p[peak]) / (np.pi * nsc)        # cycles/sample
    # The CP is a copy of the symbol tail, so M(d) is already maximal from
    # the frame boundary through the CP — the plateau STARTS at the frame
    # start and argmax wanders it under noise (S&C Fig. 3). Walk left from
    # the argmax to where M(d) crosses 90 % of the peak. M(d) ramps as
    # ((L-k)/L)^2 into the plateau, so the crossing sits (1 - sqrt(0.9))·L
    # ~ L/20 = nsc/40 samples BEFORE the frame boundary
    # (_schmidl_cox_backoff) — a deliberate margin that keeps the FFT
    # window inside the CP under multipath (measured: clean-frame yield at
    # 20 dB, D=6, is 100 % here against 50 % for a fixed -cp step). Lowering
    # the 0.9 constant widens that margin; raising it removes it. The margin
    # is spent from the CP, which this function does not know:
    # OFDMReceiver.receive caps it at half the CP.
    threshold = _SC_EDGE_FRACTION * metric[peak]
    start = peak
    while start > 0 and metric[start - 1] >= threshold:
        start -= 1
    return start, float(cfo)


def remove_cfo(signal, cfo):
    """Remove a carrier frequency offset ``cfo`` (cycles/sample) from a
    baseband signal: ``x * e^{-j2pi cfo n}``, the de-rotation
    :func:`schmidl_cox_sync`'s estimate is fed back into. A signal carrying
    an offset ``cfo`` is ``remove_cfo(x, -cfo)``.

    Parameters
    ----------
    signal : array_like
        Complex baseband samples.
    cfo : float
        The offset to remove (cycles/sample).
    """
    x = np.asarray(signal, dtype=complex)
    n = np.arange(x.size)
    return x * np.exp(-2j * np.pi * float(cfo) * n)


def estimate_channel(rx_pilot_symbol, pilot_values, n_subcarriers, cp_len):
    """Per-subcarrier LS channel estimate ``H`` from one known pilot OFDM symbol.

    ``rx_pilot_symbol`` is the received CP-prefixed pilot block; ``pilot_values``
    the transmitted subcarrier values. Returns ``H`` of length ``n_subcarriers``.

    A pilot symbol need not load every subcarrier — a Schmidl & Cox training
    block nulls the odd ones — and an unloaded subcarrier excites nothing, so
    its channel is undefined and comes back as **zero**, the same value the
    equalizers give an unrecoverable subcarrier. The threshold is
    ``1e-12`` of the peak pilot magnitude rather than an absolute level, so it
    means the same thing whatever amplitude the pilot constellation is in.

    Parameters
    ----------
    rx_pilot_symbol : array_like
        The received CP-prefixed pilot block.
    pilot_values : array_like
        The transmitted subcarrier values of the pilot.
    n_subcarriers : int
        Subcarriers per OFDM block (the FFT length).
    cp_len : int
        Cyclic-prefix length in samples.
    """
    nsc = _require_subcarrier_count(n_subcarriers, "estimate_channel")
    cp = _require_cp_len(cp_len, nsc, "estimate_channel")
    rx = np.asarray(rx_pilot_symbol, dtype=complex)
    # A short block slices to fewer than nsc samples and reaches the division
    # by ``pilot`` as a broadcast ValueError naming only the two lengths. The
    # frame layout is what the caller got wrong, so name it: zero-padding the
    # FFT instead would fabricate an estimate for subcarriers nothing excited.
    if rx.size < cp + nsc:
        raise ConfigurationError(
            f"estimate_channel: rx_pilot_symbol holds {rx.size} samples but a "
            f"CP-prefixed pilot block is cp_len + n_subcarriers = {cp} + "
            f"{nsc} = {cp + nsc} samples. Pass the whole block, starting at "
            f"the cyclic prefix.")
    rx = rx[cp:cp + nsc]
    rxf = np.fft.fft(rx) / np.sqrt(nsc)
    pilot = np.asarray(pilot_values, dtype=complex)
    mag = np.abs(pilot)
    peak = float(mag.max()) if mag.size else 0.0
    loaded = (mag > _PILOT_REL_FLOOR * peak if peak > 0.0
              else np.zeros(mag.shape, dtype=bool))
    return np.where(loaded, rxf / np.where(loaded, pilot, 1.0), 0.0)
