"""OFDM modulation/demodulation with a cyclic prefix and per-subcarrier equalization.

OFDM turns a frequency-selective channel into many flat subchannels, so a long
UW delay spread is handled by a one-tap-per-subcarrier equalizer instead of a
long time-domain filter — the basis of most modern UW multicarrier modems
(Stojanovic; Li, Zhou et al.).

References
----------
Proakis & Salehi. *Digital Communications* (multicarrier/OFDM, cyclic prefix).
Schmidl & Cox (1997), *Robust frequency and timing synchronization for OFDM*,
    IEEE Trans. Comms — the two-identical-halves preamble used for timing + CFO.
Li, Stojanovic et al. (2007), *Multicarrier communication over underwater
    acoustic channels with nonuniform Doppler shifts*, IEEE JOE — the practical
    resample-then-residual-CFO receiver.
"""

from __future__ import annotations

import warnings

import numpy as np

from uacpy.core.exceptions import ConfigurationError
from uacpy.comms._equalizer_core import regularizer

# Smallest fraction of the peak pilot magnitude a subcarrier can carry and
# still define a channel estimate. Relative because |pilot| carries whatever
# amplitude scale the caller's constellation is in: an absolute floor silently
# makes the result a function of those units. `_equalizer_core._ZF_REL_FLOOR`
# is the same rule for |H|^2 and `system_id._etfe_divide` for a transfer
# function.
_PILOT_REL_FLOOR = 1e-12

# Fraction of the channel's total energy beyond the cyclic prefix above which
# ofdm_demodulate warns of inter-block interference (ISI-to-signal ~ -20 dB).
# Relative to the channel's own energy, so the verdict is independent of its
# amplitude units and of its representation length: a full-length
# np.fft.ifft(H_est) of a channel whose support fits the prefix carries only
# numerical noise past cp+1 taps and passes silently.
_ISI_TAIL_REL_ENERGY = 0.01


def _require_subcarrier_count(n_subcarriers, caller: str) -> int:
    """Validate ``n_subcarriers >= 1`` and return it as ``int``.

    Every entry point here divides or reshapes by this count: zero reaches
    ``%`` as ``integer modulo by zero``, a negative one reaches ``reshape``
    or ``np.fft`` as ``negative dimensions are not allowed`` — untyped errors
    naming no argument the caller passed.
    """
    nsc = int(n_subcarriers)
    if nsc < 1:
        raise ConfigurationError(
            f"{caller}: n_subcarriers must be >= 1; got {n_subcarriers!r}. "
            f"It is the FFT length of one OFDM block, so every block, cyclic "
            f"prefix and subcarrier index is measured in it.")
    return nsc


def _require_cp_len(cp_len, nsc: int, caller: str) -> int:
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
            f"{caller}: cp_len ({cp_len}) must satisfy "
            f"0 <= cp_len <= n_subcarriers ({nsc})"
        )
    return cp


def ofdm_modulate(symbols, n_subcarriers, cp_len):
    """Map symbols onto ``n_subcarriers`` and prepend a cyclic prefix.

    Symbols are zero-padded to a whole number of OFDM blocks. Returns the
    complex time-domain signal (blocks concatenated).
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


def ofdm_demodulate(rx, n_subcarriers, cp_len, channel=None, snr_linear=None):
    """Recover symbols: strip CP, FFT, and optionally equalize per subcarrier.

    With ``channel`` given, divides each subcarrier by the channel frequency
    response ``H(f)`` (zero-forcing), or the MMSE weight when ``snr_linear`` is
    also set. Note that with the hard-decision slicer this package decodes with,
    the per-subcarrier MMSE output is the zero-forcing output times a positive
    real factor ``|H|^2 / (|H|^2 + N/S)``: every PSK decision is identical and
    QAM decisions are slightly WORSE (the estimate is biased toward zero, so
    outer points fall inward). It earns its keep only with soft decisions or
    bias removal, which this receiver does not do. Returns the flat complex
    symbol array.

    A channel longer than ``n_subcarriers`` raises. One carrying more than
    1 % of its energy in the taps beyond the cyclic prefix (ISI-to-signal
    above about -20 dB) is accepted with a ``UserWarning`` — an under-CP
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
    """
    nsc = _require_subcarrier_count(n_subcarriers, "ofdm_demodulate")
    cp = _require_cp_len(cp_len, nsc, "ofdm_demodulate")
    r = np.asarray(rx, dtype=complex).ravel()
    blk = nsc + cp
    nblocks = r.size // blk
    if nblocks == 0:
        raise ConfigurationError(
            f"ofdm_demodulate: signal shorter than one block — rx holds "
            f"{r.size} samples, and one block is n_subcarriers + cp_len = "
            f"{nsc} + {cp} = {blk}.")
    grid = r[: nblocks * blk].reshape(nblocks, blk)[:, cp:]
    freq = np.fft.fft(grid, axis=1) / np.sqrt(nsc)
    if channel is not None:
        hc = np.asarray(channel, dtype=complex).ravel()
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
                f"leaks inter-block interference into the next block.",
                UserWarning, stacklevel=2)
        freq = equalize_subcarriers(freq, np.fft.fft(hc, nsc), snr_linear)
    return freq.ravel()


def equalize_subcarriers(freq, H, snr_linear=None):
    """One-tap-per-subcarrier equalisation of block spectra ``freq``
    (``(..., n_subcarriers)``) by the channel response ``H``: the
    ``conj(H)/(|H|^2 + eps)`` form with ``eps`` from :func:`regularizer` —
    zero-forcing with a floor, or the MMSE weight when ``snr_linear`` is
    given. A channel with no power anywhere returns zeros: every subcarrier
    is unrecoverable, which is what the epsilon form tends to."""
    h2 = np.abs(H) ** 2
    eps = regularizer(h2, snr_linear)
    if eps <= 0.0:
        return np.zeros_like(freq)
    return freq * (np.conj(H) / (h2 + eps))


def ofdm_symbol(freq, n_sc, cp):
    """One CP-prefixed OFDM time-domain symbol from a length-``n_sc`` spectrum."""
    n_sc = _require_subcarrier_count(n_sc, "ofdm_symbol")
    cp = _require_cp_len(cp, n_sc, "ofdm_symbol")
    t = np.fft.ifft(freq) * np.sqrt(n_sc)
    return np.concatenate([t[n_sc - cp:], t])


def schmidl_cox_preamble(n_subcarriers, cp_len, seed=0x5C0FFEE):
    """Schmidl & Cox training symbol — two identical time-domain halves.

    Even subcarriers carry a PN-QPSK sequence, odd subcarriers are null, so the
    IFFT produces two identical halves of length ``n_subcarriers/2``. Returns the
    CP-prefixed complex time-domain preamble.
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


def schmidl_cox_sync(rx, n_subcarriers):
    """Locate the Schmidl & Cox preamble and estimate the fractional CFO.

    Returns ``(start, cfo)`` — ``start`` is the index of the preamble's cyclic
    prefix; ``cfo`` the normalized carrier frequency offset (cycles/sample) from
    the half-symbol phase. ``start`` is ``None`` if no clear plateau is found.
    The metric depends only on the two identical ``n_subcarriers/2`` halves,
    so the cyclic-prefix length plays no part in the search.
    """
    r = np.asarray(rx, dtype=complex).ravel()
    nsc = _require_subcarrier_count(n_subcarriers, "schmidl_cox_sync")
    L = nsc // 2
    n = r.size - 2 * L
    if n <= 0:
        return None, 0.0
    # Schmidl & Cox 1997 eqs. (5) and (7): P(d) = sum conj(r[d+m]) r[d+m+L]
    # and R(d) = sum |r[d+m+L]|^2 — both length-L sliding sums, O(n) here via
    # cumulative sums rather than the paper's iterative form (6).
    a = np.conj(r[:-L]) * r[L:]
    ca = np.concatenate(([0.0 + 0.0j], np.cumsum(a)))
    p = ca[L:L + n] - ca[:n]
    energy = np.abs(r[L:]) ** 2
    ce = np.concatenate(([0.0], np.cumsum(energy)))
    rr = ce[L:L + n] - ce[:n]
    rr_max = float(rr.max())
    if rr_max <= 0.0:
        return None, 0.0
    # The energy gate runs first and the division is done only where it passes,
    # so the metric needs no epsilon. An absolute one is wrong here anyway:
    # rr**2 scales as amplitude**4, so `rr**2 + 1e-12` made M(d) a function of
    # the units the caller held the record in — a noise-free frame synced in
    # µPa and returned (None, 0.0) in Pa, below ~5e-4 amplitude. The same
    # reasoning is written out at `system_id._etfe_divide`.
    loud = rr >= 0.25 * rr_max              # energy gate: ignore silent regions
    metric = np.zeros(n)
    metric[loud] = np.abs(p[loud]) ** 2 / rr[loud] ** 2
    peak = int(np.argmax(metric))
    # Timing metric M(d) = |P(d)|^2 / R(d)^2, eq. (8). Two exactly identical
    # halves give |P| = R and hence M = 1, so half the ideal plateau height
    # separates a preamble from a noise peak (the paper's Fig. 3 plateau sits
    # near 0.8 at 10 dB SNR).
    if metric[peak] < 0.5:
        return None, 0.0
    # The two halves are L = nsc/2 samples apart, so P accumulates a phase of
    # 2*pi*cfo*L = pi*cfo*nsc; unambiguous only for |cfo| < 1/nsc.
    cfo = np.angle(p[peak]) / (np.pi * nsc)        # cycles/sample
    # The CP is a copy of the symbol tail, so M(d) is already maximal from
    # the frame boundary through the CP — the plateau STARTS at the frame
    # start and argmax wanders it under noise (S&C Fig. 3). Walk left from
    # the argmax to where M(d) crosses 90 % of the peak. M(d) ramps as
    # ((L-k)/L)^2 into the plateau, so the crossing sits ~L/10 samples
    # BEFORE the frame boundary — a deliberate margin that keeps the FFT
    # window inside the CP under multipath (measured: clean-frame yield at
    # 20 dB, D=6, is 100 % here against 50 % for a fixed -cp step). Lowering
    # the 0.9 constant widens that margin; raising it removes it.
    threshold = 0.9 * metric[peak]
    start = peak
    while start > 0 and metric[start - 1] >= threshold:
        start -= 1
    return start, float(cfo)


def apply_cfo(signal, cfo):
    """De-rotate a baseband signal by ``cfo`` (cycles/sample): ``x * e^{-j2pi cfo n}``."""
    x = np.asarray(signal, dtype=complex)
    n = np.arange(x.size)
    return x * np.exp(-2j * np.pi * float(cfo) * n)


def estimate_channel(rx_pilot_symbol, pilot_freq, n_subcarriers, cp_len):
    """Per-subcarrier LS channel estimate ``H`` from one known pilot OFDM symbol.

    ``rx_pilot_symbol`` is the received CP-prefixed pilot block; ``pilot_freq``
    the transmitted subcarrier values. Returns ``H`` of length ``n_subcarriers``.

    A pilot symbol need not load every subcarrier — a Schmidl & Cox training
    block nulls the odd ones — and an unloaded subcarrier excites nothing, so
    its channel is undefined and comes back as **zero**, the same value the
    equalizers give an unrecoverable subcarrier. The threshold is
    ``1e-12`` of the peak pilot magnitude rather than an absolute level, so it
    means the same thing whatever amplitude the pilot constellation is in.
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
    pilot = np.asarray(pilot_freq, dtype=complex)
    mag = np.abs(pilot)
    peak = float(mag.max()) if mag.size else 0.0
    loaded = (mag > _PILOT_REL_FLOOR * peak if peak > 0.0
              else np.zeros(mag.shape, dtype=bool))
    return np.where(loaded, rxf / np.where(loaded, pilot, 1.0), 0.0)
