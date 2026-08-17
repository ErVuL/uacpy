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

import numpy as np

from uacpy.core.exceptions import ConfigurationError


def ofdm_modulate(symbols, n_subcarriers, cp_len):
    """Map symbols onto ``n_subcarriers`` and prepend a cyclic prefix.

    Symbols are zero-padded to a whole number of OFDM blocks. Returns the
    complex time-domain signal (blocks concatenated).
    """
    s = np.asarray(symbols, dtype=complex).ravel()
    nsc = int(n_subcarriers)
    if int(cp_len) > nsc:
        raise ConfigurationError(
            f"ofdm_modulate: cp_len ({cp_len}) must be <= n_subcarriers ({nsc})"
        )
    if s.size % nsc:
        s = np.concatenate([s, np.zeros(nsc - s.size % nsc, dtype=complex)])
    blocks = s.reshape(-1, nsc)
    time = np.fft.ifft(blocks, axis=1) * np.sqrt(nsc)   # unit-energy per subcarrier
    cp = time[:, nsc - int(cp_len):]
    return np.concatenate([cp, time], axis=1).ravel()


def ofdm_demodulate(rx, n_subcarriers, cp_len, channel=None, snr_linear=None):
    """Recover symbols: strip CP, FFT, and optionally equalize per subcarrier.

    With ``channel`` given, divides each subcarrier by the channel frequency
    response ``H(f)`` (zero-forcing), or applies MMSE when ``snr_linear`` is also
    set. Returns the flat complex symbol array.

    A multipath channel can put a spectral null on a subcarrier. Both equalizers
    are written as ``conj(H)/(|H|^2 + eps)`` so such a subcarrier comes back as
    zero rather than inf/NaN; for MMSE ``eps`` is the physical ``1/SNR``, for
    zero-forcing it is only a floor, and the subcarrier is unrecoverable either
    way.
    """
    nsc = int(n_subcarriers)
    cp = int(cp_len)
    r = np.asarray(rx, dtype=complex).ravel()
    blk = nsc + cp
    nblocks = r.size // blk
    if nblocks == 0:
        raise ConfigurationError("ofdm_demodulate: signal shorter than one block")
    grid = r[: nblocks * blk].reshape(nblocks, blk)[:, cp:]
    freq = np.fft.fft(grid, axis=1) / np.sqrt(nsc)
    if channel is not None:
        H = np.fft.fft(np.asarray(channel, dtype=complex), nsc)
        eps = 1e-12 if snr_linear is None else 1.0 / float(snr_linear)
        freq = freq * (np.conj(H) / (np.abs(H) ** 2 + eps))[None, :]
    return freq.ravel()


def ofdm_symbol(freq, n_sc, cp):
    """One CP-prefixed OFDM time-domain symbol from a length-``n_sc`` spectrum."""
    t = np.fft.ifft(freq) * np.sqrt(n_sc)
    return np.concatenate([t[n_sc - cp:], t])


def schmidl_cox_preamble(n_subcarriers, cp_len, seed=0x5C0FFEE):
    """Schmidl & Cox training symbol — two identical time-domain halves.

    Even subcarriers carry a PN-QPSK sequence, odd subcarriers are null, so the
    IFFT produces two identical halves of length ``n_subcarriers/2``. Returns the
    CP-prefixed complex time-domain preamble.
    """
    nsc = int(n_subcarriers)
    rng = np.random.default_rng(seed)
    freq = np.zeros(nsc, dtype=complex)
    even = np.arange(0, nsc, 2)
    # Schmidl & Cox 1997 sec. III-A: "the frequency components of this training
    # symbol are multiplied by sqrt(2) at the transmitter" — it compensates for
    # loading only the even subcarriers, so the block carries the same total
    # energy (nsc) as a fully loaded one.
    freq[even] = np.exp(1j * np.pi / 2 * rng.integers(0, 4, even.size)) * np.sqrt(2)
    return ofdm_symbol(freq, nsc, int(cp_len))


def schmidl_cox_sync(rx, n_subcarriers, cp_len):
    """Locate the Schmidl & Cox preamble and estimate the fractional CFO.

    Returns ``(start, cfo)`` — ``start`` is the index of the preamble's cyclic
    prefix; ``cfo`` the normalized carrier frequency offset (cycles/sample) from
    the half-symbol phase. ``start`` is ``None`` if no clear plateau is found.
    """
    r = np.asarray(rx, dtype=complex).ravel()
    nsc = int(n_subcarriers)
    cp = int(cp_len)
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
    metric = np.abs(p) ** 2 / (rr ** 2 + 1e-12)
    metric[rr < 0.25 * rr.max()] = 0.0      # energy gate: ignore silent regions
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
    # 20 dB rose 50%->100 % at D=6 vs the old fixed -cp step). Lowering
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
    """
    nsc = int(n_subcarriers)
    cp = int(cp_len)
    rx = np.asarray(rx_pilot_symbol, dtype=complex)[cp:cp + nsc]
    rxf = np.fft.fft(rx) / np.sqrt(nsc)
    pilot = np.asarray(pilot_freq, dtype=complex)
    return rxf / (pilot + 1e-12)
