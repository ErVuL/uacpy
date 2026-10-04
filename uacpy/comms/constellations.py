"""Bits to symbols and back: the scheme table (:data:`SCHEMES`), the
Gray-mapped constellations (:func:`constellation`, :class:`Modulator`), the
hard slicer (:func:`slicer`), and the non-coherent DPSK and FSK modems."""

from __future__ import annotations

import warnings
from types import MappingProxyType

import numpy as np

from uacpy.core._validate import (
    require_below_nyquist, require_positive_finite_scalar,
)
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core._repr import SettingsRepr
from uacpy.core.exceptions import ConfigurationError, ValidityWarning


#: Every modulation scheme the package maps, by name: ``(family, order)``,
#: the family ``'psk'`` (phase shift keying) or ``'qam'`` (square quadrature
#: amplitude modulation) and the order ``M``. :func:`constellation`,
#: :class:`Modulator` and :func:`~uacpy.comms.ber_theory` read this one table.
SCHEMES = MappingProxyType({
    'bpsk': ('psk', 2), 'qpsk': ('psk', 4), '8psk': ('psk', 8),
    '16psk': ('psk', 16),
    '16qam': ('qam', 16), '64qam': ('qam', 64), '256qam': ('qam', 256),
})


def _gray(n):
    return n ^ (n >> 1)


def _psk_lut(order):
    """Gray-mapped unit-energy M-PSK LUT: ``lut[label]`` -> constellation point."""
    pts = np.exp(1j * 2 * np.pi * np.arange(order) / order)
    lut = np.empty(order, dtype=complex)
    for m in range(order):
        lut[_gray(m)] = pts[m]
    return lut


def _qam_lut(order):
    """Gray-mapped unit-average-energy square M-QAM LUT (M a perfect square)."""
    L = int(round(np.sqrt(order)))
    if L * L != order:
        raise ConfigurationError(f"square QAM needs a perfect-square order, got {order}.")
    levels = np.arange(-(L - 1), L, 2)  # PAM levels: -(L-1)..(L-1) step 2
    gray = [_gray(i) for i in range(L)]
    pos = {g: levels[i] for i, g in enumerate(gray)}  # gray-label -> PAM level
    bps_axis = int(np.log2(L))
    lut = np.empty(order, dtype=complex)
    for label in range(order):
        i_lbl = label >> bps_axis
        q_lbl = label & (L - 1)
        lut[label] = pos[i_lbl] + 1j * pos[q_lbl]
    lut /= np.sqrt(np.mean(np.abs(lut) ** 2))  # unit average energy
    return lut


def constellation(scheme):
    """Unit-average-energy Gray-mapped constellation for ``scheme``.

    ``scheme`` in {bpsk, qpsk, 8psk, 16psk, 16qam, 64qam, 256qam}. Returns a
    complex array where index = the integer bit-label of that symbol.

    Parameters
    ----------
    scheme : str
        A scheme of :data:`SCHEMES`, in any case.
    """
    s = scheme.lower()
    if s in SCHEMES:
        family, order = SCHEMES[s]
        return _psk_lut(order) if family == 'psk' else _qam_lut(order)
    by_family = [sorted(name for name, (fam, _) in SCHEMES.items()
                        if fam == family) for family in ('psk', 'qam')]
    raise ConfigurationError(
        f"constellation: unknown scheme {scheme!r}; choose from "
        f"{by_family[0] + by_family[1]}."
    )


def _require_binary_bits(who: str, bits):
    """Return ``bits`` as a validated 1-D array of 0/1 integers.

    The bit-label arithmetic downstream (``groups @ weights``, trellis column
    lookup) indexes tables by value, so a -1 bit selects an entry from the
    END of its table — a silently wrong symbol, not an error. The package's
    own :func:`~uacpy.acoustic_signal.m_sequence` chips
    are bipolar ±1 exactly like that; map them back with
    ``bits = (1 - chips) // 2``. Strings are rejected rather than parsed as
    numbers.
    """
    if isinstance(bits, (str, bytes, bytearray)):
        raise ConfigurationError(
            f"{who}: bits must be a sequence of 0/1 integers, got "
            f"{type(bits).__name__} {bits!r:.60}; convert text with "
            f"uacpy.comms.bytes_to_bits(text.encode()) or "
            f"[int(ch) for ch in bit_string].")
    try:
        b = np.asarray(bits, dtype=int).ravel()
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"{who}: bits must be a sequence of 0/1 integers "
            f"({exc}).") from exc
    bad = b[(b != 0) & (b != 1)]
    if bad.size:
        raise ConfigurationError(
            f"{who}: bits must be 0/1, got value(s) "
            f"{np.unique(bad)[:8].tolist()}. A bipolar ±1 sequence (the "
            f"m_sequence() chip convention) must be mapped back "
            f"first — bits = (1 - chips) // 2 — a -1 label would select "
            f"a symbol from the wrong end of the table.")
    return b


def _bits_to_labels(bits, bits_per_symbol):
    """Integer symbol labels from 0/1 ``bits``, ``bits_per_symbol`` at a
    time, most significant bit first; the last group is zero-padded to a
    whole symbol."""
    bps = int(bits_per_symbol)
    if bits.size % bps:
        bits = np.concatenate([bits, np.zeros(bps - bits.size % bps,
                                              dtype=int)])
    return bits.reshape(-1, bps) @ (1 << np.arange(bps - 1, -1, -1))


def _labels_to_bits(labels, bits_per_symbol):
    """The inverse of :func:`_bits_to_labels`: each integer label as
    ``bits_per_symbol`` bits, most significant first, in one 1-D array."""
    bps = int(bits_per_symbol)
    labels = np.asarray(labels, dtype=int)
    return ((labels[:, None] >> np.arange(bps - 1, -1, -1)) & 1).ravel()


class Modulator(SettingsRepr):
    """Bit<->symbol mapper for a Gray-coded constellation.

    Parameters
    ----------
    scheme : str
        Constellation name (see :func:`constellation`).
    """

    def __init__(self, scheme: str = "qpsk"):
        self.scheme = scheme
        self.constellation = constellation(scheme)
        self.order = self.constellation.size
        self.bits_per_symbol = int(np.log2(self.order))

    def modulate(self, bits):
        """Map a 1-D 0/1 bit array to complex symbols (zero-padded to a whole symbol)."""
        b = _require_binary_bits("Modulator.modulate", bits)
        return self.constellation[_bits_to_labels(b, self.bits_per_symbol)]

    def demodulate(self, symbols):
        """Hard minimum-distance decision: complex symbols -> 1-D bit array.

        Slicing is against the constellation at its native unit average
        energy: an amplitude-scaled QAM input decodes to the wrong rings
        silently, so the caller must restore scale first — the package's
        equalizer and OFDM chains already do; constant-modulus PSK is
        unaffected by a common gain.

        The nearest-point search is :func:`uacpy.comms.slicer`'s, which
        works in blocks so its memory is bounded whatever the record length.
        """
        s = np.asarray(symbols, dtype=complex).ravel()
        labels = _nearest_labels(s, self.constellation)
        return _labels_to_bits(labels, self.bits_per_symbol)


def _require_power_of_two_m(who: str, order):
    """Reject an M-ary order that is not a power of two >= 2.

    ``bits_per_symbol`` is ``floor(log2(M))``, so a non-power-of-two M silently
    modulates onto only ``2**floor(log2(M))`` of the M phase points — order = 12
    uses 8 phases spaced 2*pi/12, covering two thirds of the circle at a
    smaller minimum distance than 8-DPSK, with no indication. M <= 1 divides by
    a zero bits-per-symbol. :func:`fsk_modulate` makes the same check.
    """
    m = int(order)
    if m < 2 or (m & (m - 1)):
        raise ConfigurationError(
            f"{who}: M must be a power of two >= 2 (got {order!r})")


def dpsk_modulate(bits, order: int = 2):
    """Differential M-PSK: encode phase *differences* (no carrier-phase reference).

    Robust to slow carrier-phase drift — common in non-coherent UW links.
    ``order`` must be a power of two >= 2, as in :func:`fsk_modulate`.

    Parameters
    ----------
    bits : array_like of int
        Information bits, 0/1.
    order : int, optional
        The M of M-DPSK, a power of two >= 2. Default 2.
    """
    bits = _require_binary_bits("dpsk_modulate", bits)
    _require_power_of_two_m("dpsk_modulate", order)
    sym = _bits_to_labels(bits, int(np.log2(order)))
    # Gray constellation: bit-label `l` sits on phase point `p` with gray(p) = l,
    # the same labelling as the coherent `_psk_lut`.
    point = {_gray(p): p for p in range(order)}
    dphi = 2 * np.pi * np.array([point[int(s)] for s in sym]) / order
    phase = np.cumsum(np.concatenate([[0.0], dphi]))
    return np.exp(1j * phase)  # length len(sym)+1 (incl. reference symbol)


def dpsk_demodulate(symbols, order: int = 2):
    """Inverse of :func:`dpsk_modulate` (differential phase detection).

    Parameters
    ----------
    symbols : array_like
        Received symbols, the reference symbol first.
    order : int, optional
        The M :func:`dpsk_modulate` used. Default 2.
    """
    s = np.asarray(symbols, dtype=complex).ravel()
    _require_power_of_two_m("dpsk_demodulate", order)
    if s.size < 2:
        return np.zeros(0, dtype=int)
    dphi = np.angle(s[1:] * np.conj(s[:-1])) % (2 * np.pi)
    sym = np.round(dphi / (2 * np.pi / order)).astype(int) % order
    label = np.array([_gray(int(p)) for p in sym])
    return _labels_to_bits(label, int(np.log2(order)))


def _whole_samples_per_symbol(who, duration_s, sample_rate) -> int:
    """``round(duration_s * sample_rate)``, the samples one FSK symbol
    spans, refused when it rounds to zero."""
    n = int(round(duration_s * float(sample_rate)))
    if n < 1:
        raise ConfigurationError(
            f"{who}: symbol_duration_s ({duration_s:g} s) x sample_rate "
            f"({float(sample_rate):g} Hz) is "
            f"{duration_s * float(sample_rate):g} samples per symbol, which "
            f"rounds to zero samples; require symbol_duration_s >= "
            f"0.5/sample_rate.")
    return n


def fsk_modulate(bits, frequencies, symbol_duration_s: float, sample_rate: float):
    """Binary/M-FSK waveform: each symbol is a tone from ``frequencies``.

    ``len(frequencies)`` must be a power of two (M-ary). Returns the real passband
    waveform (continuous-phase not enforced).

    Parameters
    ----------
    bits : array_like of int
        Information bits, 0/1.
    frequencies : array_like
        The M tone frequencies (Hz), M a power of two, below Nyquist.
    symbol_duration_s : float
        Symbol duration (s).
    sample_rate : float
        Sample rate (Hz).
    """
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    M = f.size
    _require_power_of_two_m("fsk_modulate", M)
    require_below_nyquist(f, sample_rate, "fsk_modulate", "tone(s)",
                          "the sampled tones alias")
    dur = require_positive_finite_scalar(
        symbol_duration_s, "fsk_modulate", "symbol_duration_s", " s")
    bps = int(np.log2(M))
    b = _require_binary_bits("fsk_modulate", bits)
    sym = _bits_to_labels(b, bps)
    if sym.size == 0:
        return np.zeros(0, dtype=float)
    n = _whole_samples_per_symbol("fsk_modulate", dur, sample_rate)
    t = np.arange(n) / float(sample_rate)
    return np.concatenate([np.cos(2 * np.pi * f[int(s)] * t) for s in sym])


# Largest distance, in cycles over one symbol, between a tone spacing times
# the symbol duration and the nearest whole number for the tones to count
# as orthogonal.
_FSK_ORTHOGONALITY_TOL = 1e-6


def _warn_unless_orthogonal_tones(f, duration_s):
    """Warn when two tones are not a whole number of cycles apart over one
    symbol: their correlators then leak into each other and the
    non-coherent detector falls short of ``ber_theory('bfsk')``."""
    cycles = np.abs(f[:, None] - f[None, :])[np.triu_indices(f.size, 1)] \
        * duration_s
    off = np.abs(cycles - np.round(cycles))
    bad = (off > _FSK_ORTHOGONALITY_TOL) | (np.round(cycles) < 1)
    if np.any(bad):
        worst = float(cycles[bad][0])
        warnings.warn(
            f"fsk_demodulate: tones {worst:g} cycles apart over one symbol "
            f"({duration_s:g} s as sampled); the correlators are orthogonal "
            f"only for a whole, non-zero number of cycles, i.e. a spacing "
            f"that is a multiple of 1/symbol_duration_s. Detection runs, but "
            f"each tone leaks into the others and the error rate exceeds the "
            f"orthogonal-FSK curve.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP)


def fsk_demodulate(signal, frequencies, symbol_duration_s: float, sample_rate: float):
    """Non-coherent M-FSK detection (per-symbol max tone energy) -> bit array.

    A trailing partial symbol (fewer than one symbol period of samples) is
    dropped, so a signal shorter than one symbol demodulates to an empty
    bit array.

    The tones must be spaced by a multiple of ``1/symbol_duration_s`` (whole
    cycles over the symbol as sampled, ``round(symbol_duration_s *
    sample_rate)`` samples) for the correlators to be orthogonal, which is
    what :func:`~uacpy.comms.ber_theory` ``'bfsk'`` assumes; any other
    spacing warns.

    Parameters
    ----------
    signal : array_like
        The real received waveform, starting on a symbol boundary.
    frequencies : array_like
        The tone frequencies :func:`fsk_modulate` used (Hz).
    symbol_duration_s : float
        Symbol duration (s).
    sample_rate : float
        Sample rate (Hz).
    """
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    M = f.size
    _require_power_of_two_m("fsk_demodulate", M)
    # The detector builds its correlation bank from the same two quantities the
    # modulator validates, so it carries the same two guards. Without them an
    # above-Nyquist tone decodes its aliased image to plausible-looking bits,
    # and a non-finite, negative or subsample symbol duration reaches the
    # ``x.size // n`` below as NaN, a negative count (empty bit array) or zero
    # (ZeroDivisionError).
    require_below_nyquist(
        f, sample_rate, "fsk_demodulate", "tone(s)",
        "the correlation bank matches their aliased images rather than the "
        "tones themselves")
    dur = require_positive_finite_scalar(
        symbol_duration_s, "fsk_demodulate", "symbol_duration_s", " s")
    bps = int(np.log2(M))
    n = _whole_samples_per_symbol("fsk_demodulate", dur, sample_rate)
    _warn_unless_orthogonal_tones(f, n / float(sample_rate))
    x = np.asarray(signal, dtype=float)
    nsym = x.size // n
    t = np.arange(n) / float(sample_rate)
    bank = np.exp(-2j * np.pi * np.outer(f, t))  # (M, n)
    labels = [int(np.argmax(np.abs(bank @ x[k * n:(k + 1) * n])))
              for k in range(nsym)]
    return _labels_to_bits(np.array(labels, dtype=int), bps)


# Symbols per nearest-point block: the pairwise distance matrix is
# (block, M) instead of (N, M), so its size is bounded whatever the record
# length. The peak is 385 MiB for 256-QAM at this block size (measured with
# tracemalloc), not the 128 MiB of the distance matrix alone: the complex128
# `block[:, None] - constellation[None, :]` difference is 256 MiB and is live
# at the same time as the 128 MiB float64 `np.abs` result. 16-QAM peaks at
# 25 MiB.
_DEMOD_CHUNK = 1 << 16


def _nearest_labels(x, constellation):
    """Index of the nearest constellation point to each of the 1-D complex
    ``x``, searched in blocks of ``_DEMOD_CHUNK`` symbols. Each decision is
    independent, so the result is identical to the one-shot search."""
    labels = np.empty(x.size, dtype=np.intp)
    for i in range(0, x.size, _DEMOD_CHUNK):
        block = x[i:i + _DEMOD_CHUNK]
        d = np.abs(block[:, None] - constellation[None, :])
        labels[i:i + _DEMOD_CHUNK] = np.argmin(d, axis=1)
    return labels


def slicer(x, constellation):
    """Nearest constellation point(s) to ``x``.

    The same search :meth:`uacpy.comms.Modulator.demodulate` makes, in
    blocks, so a long record of a large constellation stays in bounded
    memory.

    Parameters
    ----------
    x : complex or array_like
        Symbols to decide.
    constellation : array_like
        The symbol set.
    """
    c = np.asarray(constellation, dtype=complex)
    x = np.atleast_1d(np.asarray(x, dtype=complex))
    return c[_nearest_labels(x.ravel(), c).reshape(x.shape)]

