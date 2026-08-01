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
        if snr_linear is None:
            freq = freq / H[None, :]
        else:
            freq = freq * (np.conj(H) / (np.abs(H) ** 2 + 1.0 / float(snr_linear)))[None, :]
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
    # P(d) = sum conj(r[d+m]) r[d+m+L];  R(d) = sum |r[d+m+L]|^2 —
    # both are length-L sliding sums, O(n) via cumulative sums.
    a = np.conj(r[:-L]) * r[L:]
    ca = np.concatenate(([0.0 + 0.0j], np.cumsum(a)))
    p = ca[L:L + n] - ca[:n]
    energy = np.abs(r[L:]) ** 2
    ce = np.concatenate(([0.0], np.cumsum(energy)))
    rr = ce[L:L + n] - ce[:n]
    metric = np.abs(p) ** 2 / (rr ** 2 + 1e-12)
    metric[rr < 0.25 * rr.max()] = 0.0      # energy gate: ignore silent regions
    peak = int(np.argmax(metric))
    if metric[peak] < 0.5:
        return None, 0.0
    # argmax sits at the useful-symbol start; step back by the CP to the block
    # boundary (a residual offset within the CP is absorbed by the pilot estimate)
    cfo = np.angle(p[peak]) / (np.pi * nsc)        # cycles/sample
    return max(peak - cp, 0), float(cfo)


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
