"""Digital modulation / demodulation for underwater acoustic links.

Gray-mapped, unit-average-energy constellations: M-PSK (incl. BPSK/QPSK) and
square M-QAM, plus differential DPSK (non-coherent-friendly) and binary/M-FSK
(non-coherent, waveform domain). Coherent schemes work in the symbol domain so
they compose with equalizers, OFDM and channel estimation.

References
----------
Proakis & Salehi. *Digital Communications* (constellations, Gray mapping).
Stojanovic, in Istepanian & Stojanovic. *Underwater Acoustic Digital Signal
    Processing and Communication Systems* (coherent vs differential UW links).
"""

from __future__ import annotations

import numpy as np

from uacpy.core.exceptions import ConfigurationError

_PSK = {"bpsk": 2, "qpsk": 4, "8psk": 8, "16psk": 16}
_QAM = {"16qam": 16, "64qam": 64, "256qam": 256}


def _gray(n):
    return n ^ (n >> 1)


def _psk_lut(M):
    """Gray-mapped unit-energy M-PSK LUT: ``lut[label]`` -> constellation point."""
    pts = np.exp(1j * 2 * np.pi * np.arange(M) / M)
    lut = np.empty(M, dtype=complex)
    for m in range(M):
        lut[_gray(m)] = pts[m]
    return lut


def _qam_lut(M):
    """Gray-mapped unit-average-energy square M-QAM LUT (M a perfect square)."""
    L = int(round(np.sqrt(M)))
    if L * L != M:
        raise ConfigurationError(f"square QAM needs a perfect-square order, got {M}")
    levels = np.arange(-(L - 1), L, 2)  # PAM levels: -(L-1)..(L-1) step 2
    gray = [_gray(i) for i in range(L)]
    pos = {g: levels[i] for i, g in enumerate(gray)}  # gray-label -> PAM level
    bps_axis = int(np.log2(L))
    lut = np.empty(M, dtype=complex)
    for label in range(M):
        i_lbl = label >> bps_axis
        q_lbl = label & (L - 1)
        lut[label] = pos[i_lbl] + 1j * pos[q_lbl]
    lut /= np.sqrt(np.mean(np.abs(lut) ** 2))  # unit average energy
    return lut


def constellation(scheme):
    """Unit-average-energy Gray-mapped constellation for ``scheme``.

    ``scheme`` in {bpsk, qpsk, 8psk, 16psk, 16qam, 64qam, 256qam}. Returns a
    complex array where index = the integer bit-label of that symbol.
    """
    s = scheme.lower()
    if s in _PSK:
        return _psk_lut(_PSK[s])
    if s in _QAM:
        return _qam_lut(_QAM[s])
    raise ConfigurationError(
        f"modulation: unknown scheme {scheme!r}; choose from "
        f"{sorted(_PSK) + sorted(_QAM)}"
    )


class Modulator:
    """Bit<->symbol mapper for a Gray-coded constellation.

    Parameters
    ----------
    scheme : str
        Constellation name (see :func:`constellation`).
    """

    def __init__(self, scheme: str = "qpsk"):
        self.scheme = scheme
        self.constellation = constellation(scheme)
        self.M = self.constellation.size
        self.bits_per_symbol = int(np.log2(self.M))

    def modulate(self, bits):
        """Map a 1-D 0/1 bit array to complex symbols (zero-padded to a whole symbol)."""
        b = np.asarray(bits, dtype=int).ravel()
        bps = self.bits_per_symbol
        if b.size % bps:
            b = np.concatenate([b, np.zeros(bps - b.size % bps, dtype=int)])
        groups = b.reshape(-1, bps)
        weights = 1 << np.arange(bps - 1, -1, -1)
        labels = groups @ weights
        return self.constellation[labels]

    def demodulate(self, symbols):
        """Hard minimum-distance decision: complex symbols -> 1-D bit array."""
        s = np.asarray(symbols, dtype=complex).ravel()
        d = np.abs(s[:, None] - self.constellation[None, :])
        labels = np.argmin(d, axis=1)
        bps = self.bits_per_symbol
        bits = ((labels[:, None] >> np.arange(bps - 1, -1, -1)) & 1)
        return bits.ravel()

def dpsk_modulate(bits, M: int = 2):
    """Differential M-PSK: encode phase *differences* (no carrier-phase reference).

    Robust to slow carrier-phase drift — common in non-coherent UW links.
    """
    bits = np.asarray(bits, dtype=int).ravel()
    bps = int(np.log2(M))
    if bits.size % bps:
        bits = np.concatenate([bits, np.zeros(bps - bits.size % bps, dtype=int)])
    groups = bits.reshape(-1, bps)
    sym = (groups @ (1 << np.arange(bps - 1, -1, -1)))
    dphi = 2 * np.pi * np.array([_gray(int(s)) for s in sym]) / M
    phase = np.cumsum(np.concatenate([[0.0], dphi]))
    return np.exp(1j * phase)  # length len(sym)+1 (incl. reference symbol)


def dpsk_demodulate(symbols, M: int = 2):
    """Inverse of :func:`dpsk_modulate` (differential phase detection)."""
    s = np.asarray(symbols, dtype=complex).ravel()
    if s.size < 2:
        return np.zeros(0, dtype=int)
    dphi = np.angle(s[1:] * np.conj(s[:-1])) % (2 * np.pi)
    sym = np.round(dphi / (2 * np.pi / M)).astype(int) % M
    bps = int(np.log2(M))
    inv = np.array([_gray(i) for i in range(M)])
    label = np.array([np.where(inv == v)[0][0] for v in sym])
    bits = ((label[:, None] >> np.arange(bps - 1, -1, -1)) & 1)
    return bits.ravel()


def fsk_modulate(bits, frequencies, symbol_dur_s: float, sample_rate: float):
    """Binary/M-FSK waveform: each symbol is a tone from ``frequencies``.

    ``len(frequencies)`` must be a power of two (M-ary). Returns the real passband
    waveform (continuous-phase not enforced).
    """
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    M = f.size
    if M < 2 or (M & (M - 1)):
        raise ConfigurationError(
            "fsk_modulate: number of freqs must be a power of two >= 2"
        )
    bps = int(np.log2(M))
    b = np.asarray(bits, dtype=int).ravel()
    if b.size % bps:
        b = np.concatenate([b, np.zeros(bps - b.size % bps, dtype=int)])
    sym = b.reshape(-1, bps) @ (1 << np.arange(bps - 1, -1, -1))
    if sym.size == 0:
        return np.zeros(0, dtype=float)
    n = int(round(symbol_dur_s * sample_rate))
    t = np.arange(n) / float(sample_rate)
    return np.concatenate([np.cos(2 * np.pi * f[int(s)] * t) for s in sym])


def fsk_demodulate(signal, frequencies, symbol_dur_s: float, sample_rate: float):
    """Non-coherent M-FSK detection (per-symbol max tone energy) -> bit array."""
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    M = f.size
    if M < 2 or (M & (M - 1)):
        raise ConfigurationError(
            "fsk_demodulate: number of freqs must be a power of two >= 2"
        )
    bps = int(np.log2(M))
    n = int(round(symbol_dur_s * sample_rate))
    x = np.asarray(signal, dtype=float)
    nsym = x.size // n
    t = np.arange(n) / float(sample_rate)
    bank = np.exp(-2j * np.pi * np.outer(f, t))  # (M, n)
    bits = []
    for k in range(nsym):
        seg = x[k * n:(k + 1) * n]
        energy = np.abs(bank @ seg)
        s = int(np.argmax(energy))
        bits.extend((s >> np.arange(bps - 1, -1, -1)) & 1)
    return np.array(bits, dtype=int)
