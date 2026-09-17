"""Building a transmission: symbols, subcarriers, spreading, coding, frames.

One question — *turn these bits into a waveform* — from the constellation
(:class:`Modulator`, :func:`constellation`) up through OFDM subcarriers,
direct-sequence spreading, forward error correction and the framing that packs
a payload for the air. Its inverse half lives in
:mod:`uacpy.comms.receive`.
"""

from __future__ import annotations
import numpy as np
from uacpy.core.exceptions import ConfigurationError
from uacpy.acoustic_signal._signal_validate import require_below_nyquist
import warnings
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.comms.receive import regularizer
import zlib


# ──────────────────────────────────────────────────────────────────────
# Constellations
#
# Symbol mapping and demapping.
# ──────────────────────────────────────────────────────────────────────

_PSK = {"bpsk": 2, "qpsk": 4, "8psk": 8, "16psk": 16}
_QAM = {"16qam": 16, "64qam": 64, "256qam": 256}

# Symbols per Modulator.demodulate block: the pairwise distance matrix is
# (block, M) instead of (N, M), so its size is bounded whatever the record
# length. The peak is 385 MiB for 256-QAM at this block size (measured with
# tracemalloc), not the 128 MiB of the distance matrix alone: the complex128
# `block[:, None] - constellation[None, :]` difference is 256 MiB and is live
# at the same time as the 128 MiB float64 `np.abs` result. 16-QAM peaks at
# 25 MiB.
_DEMOD_CHUNK = 1 << 16


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


def _require_binary_bits(caller: str, bits):
    """Return ``bits`` as a validated 1-D array of 0/1 integers.

    The bit-label arithmetic downstream (``groups @ weights``, trellis column
    lookup) indexes tables by value, so a -1 bit selects an entry from the
    END of its table — a silently wrong symbol, not an error. The package's
    own :func:`~uacpy.acoustic_signal.generate.mseq` / ``m_sequence`` chips
    are bipolar ±1 exactly like that; map them back with
    ``bits = (1 - chips) // 2``. Strings are rejected rather than parsed as
    numbers.
    """
    if isinstance(bits, (str, bytes, bytearray)):
        raise ConfigurationError(
            f"{caller}: bits must be a sequence of 0/1 integers, got "
            f"{type(bits).__name__} {bits!r:.60}; convert text with "
            f"uacpy.comms.bytes_to_bits(text.encode()) or "
            f"[int(ch) for ch in bit_string].")
    try:
        b = np.asarray(bits, dtype=int).ravel()
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"{caller}: bits must be a sequence of 0/1 integers "
            f"({exc}).") from exc
    bad = b[(b != 0) & (b != 1)]
    if bad.size:
        raise ConfigurationError(
            f"{caller}: bits must be 0/1, got value(s) "
            f"{np.unique(bad)[:8].tolist()}. A bipolar ±1 sequence (the "
            f"mseq()/m_sequence() chip convention) must be mapped back "
            f"first — bits = (1 - chips) // 2 — a -1 label would select "
            f"a symbol from the wrong end of the table.")
    return b


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
        b = _require_binary_bits("Modulator.modulate", bits)
        bps = self.bits_per_symbol
        if b.size % bps:
            b = np.concatenate([b, np.zeros(bps - b.size % bps, dtype=int)])
        groups = b.reshape(-1, bps)
        weights = 1 << np.arange(bps - 1, -1, -1)
        labels = groups @ weights
        return self.constellation[labels]

    def demodulate(self, symbols):
        """Hard minimum-distance decision: complex symbols -> 1-D bit array.

        Slicing is against the constellation at its native unit average
        energy: an amplitude-scaled QAM input decodes to the wrong rings
        silently, so the caller must restore scale first — the package's
        equalizer and OFDM chains already do; constant-modulus PSK is
        unaffected by a common gain.

        Symbols are processed in blocks of ``_DEMOD_CHUNK``, so the pairwise
        distance matrix stays a bounded ``(block, M)`` whatever the record
        length. Each symbol's decision is independent, so the result is
        identical to the one-shot computation.
        """
        s = np.asarray(symbols, dtype=complex).ravel()
        labels = np.empty(s.size, dtype=np.intp)
        for i in range(0, s.size, _DEMOD_CHUNK):
            block = s[i:i + _DEMOD_CHUNK]
            d = np.abs(block[:, None] - self.constellation[None, :])
            labels[i:i + _DEMOD_CHUNK] = np.argmin(d, axis=1)
        bps = self.bits_per_symbol
        bits = ((labels[:, None] >> np.arange(bps - 1, -1, -1)) & 1)
        return bits.ravel()

def _require_power_of_two_m(caller: str, M):
    """Reject an M-ary order that is not a power of two >= 2.

    ``bits_per_symbol`` is ``floor(log2(M))``, so a non-power-of-two M silently
    modulates onto only ``2**floor(log2(M))`` of the M phase points — M = 12
    uses 8 phases spaced 2*pi/12, covering two thirds of the circle at a
    smaller minimum distance than 8-DPSK, with no indication. M <= 1 divides by
    a zero bits-per-symbol. :func:`fsk_modulate` makes the same check.
    """
    m = int(M)
    if m < 2 or (m & (m - 1)):
        raise ConfigurationError(
            f"{caller}: M must be a power of two >= 2 (got {M!r})")


def dpsk_modulate(bits, M: int = 2):
    """Differential M-PSK: encode phase *differences* (no carrier-phase reference).

    Robust to slow carrier-phase drift — common in non-coherent UW links.
    ``M`` must be a power of two >= 2, as in :func:`fsk_modulate`.
    """
    bits = _require_binary_bits("dpsk_modulate", bits)
    _require_power_of_two_m("dpsk_modulate", M)
    bps = int(np.log2(M))
    if bits.size % bps:
        bits = np.concatenate([bits, np.zeros(bps - bits.size % bps, dtype=int)])
    groups = bits.reshape(-1, bps)
    sym = (groups @ (1 << np.arange(bps - 1, -1, -1)))
    # Gray constellation: bit-label `l` sits on phase point `p` with gray(p) = l,
    # the same labelling as the coherent `_psk_lut`.
    point = {_gray(p): p for p in range(M)}
    dphi = 2 * np.pi * np.array([point[int(s)] for s in sym]) / M
    phase = np.cumsum(np.concatenate([[0.0], dphi]))
    return np.exp(1j * phase)  # length len(sym)+1 (incl. reference symbol)


def dpsk_demodulate(symbols, M: int = 2):
    """Inverse of :func:`dpsk_modulate` (differential phase detection)."""
    s = np.asarray(symbols, dtype=complex).ravel()
    _require_power_of_two_m("dpsk_demodulate", M)
    if s.size < 2:
        return np.zeros(0, dtype=int)
    dphi = np.angle(s[1:] * np.conj(s[:-1])) % (2 * np.pi)
    sym = np.round(dphi / (2 * np.pi / M)).astype(int) % M
    bps = int(np.log2(M))
    label = np.array([_gray(int(p)) for p in sym])
    bits = ((label[:, None] >> np.arange(bps - 1, -1, -1)) & 1)
    return bits.ravel()


def fsk_modulate(bits, frequencies, symbol_dur_s: float, sample_rate: float):
    """Binary/M-FSK waveform: each symbol is a tone from ``frequencies``.

    ``len(frequencies)`` must be a power of two (M-ary). Returns the real passband
    waveform (continuous-phase not enforced).
    """
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    M = f.size
    _require_power_of_two_m("fsk_modulate", M)
    require_below_nyquist(f, sample_rate, "fsk_modulate", "tone(s)",
                          "the sampled tones alias")
    dur = float(symbol_dur_s)
    if not (np.isfinite(dur) and dur > 0):
        raise ConfigurationError(
            f"fsk_modulate: symbol_dur_s must be > 0 s and finite "
            f"(got {symbol_dur_s!r}).")
    bps = int(np.log2(M))
    b = _require_binary_bits("fsk_modulate", bits)
    if b.size % bps:
        b = np.concatenate([b, np.zeros(bps - b.size % bps, dtype=int)])
    sym = b.reshape(-1, bps) @ (1 << np.arange(bps - 1, -1, -1))
    if sym.size == 0:
        return np.zeros(0, dtype=float)
    n = int(round(dur * float(sample_rate)))
    if n < 1:
        raise ConfigurationError(
            f"fsk_modulate: symbol_dur_s ({dur:g} s) x sample_rate "
            f"({float(sample_rate):g} Hz) is {dur * float(sample_rate):g} "
            f"samples per symbol, which rounds to zero samples; require "
            f"symbol_dur_s >= 0.5/sample_rate.")
    t = np.arange(n) / float(sample_rate)
    return np.concatenate([np.cos(2 * np.pi * f[int(s)] * t) for s in sym])


def fsk_demodulate(signal, frequencies, symbol_dur_s: float, sample_rate: float):
    """Non-coherent M-FSK detection (per-symbol max tone energy) -> bit array.

    A trailing partial symbol (fewer than one symbol period of samples) is
    dropped, so a signal shorter than one symbol demodulates to an empty
    bit array.
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
    dur = float(symbol_dur_s)
    if not (np.isfinite(dur) and dur > 0):
        raise ConfigurationError(
            f"fsk_demodulate: symbol_dur_s must be > 0 s and finite "
            f"(got {symbol_dur_s!r}).")
    bps = int(np.log2(M))
    n = int(round(dur * float(sample_rate)))
    if n < 1:
        raise ConfigurationError(
            f"fsk_demodulate: symbol_dur_s ({dur:g} s) x sample_rate "
            f"({float(sample_rate):g} Hz) is {dur * float(sample_rate):g} "
            f"samples per symbol, which rounds to zero samples; require "
            f"symbol_dur_s >= 0.5/sample_rate.")
    x = np.asarray(signal, dtype=float)
    nsym = x.size // n
    t = np.arange(n) / float(sample_rate)
    bank = np.exp(-2j * np.pi * np.outer(f, t))  # (M, n)
    bits: list[int] = []
    for k in range(nsym):
        seg = x[k * n:(k + 1) * n]
        energy = np.abs(bank @ seg)
        s = int(np.argmax(energy))
        bits.extend((s >> np.arange(bps - 1, -1, -1)) & 1)
    return np.array(bits, dtype=int)


# ──────────────────────────────────────────────────────────────────────
# OFDM
#
# Subcarriers, the cyclic prefix, and per-subcarrier equalisation.
# ──────────────────────────────────────────────────────────────────────

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
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
        freq = equalize_subcarriers(freq, np.fft.fft(hc, nsc), snr_linear)
    return freq.ravel()


def equalize_subcarriers(freq, H, snr_linear=None):
    """One-tap-per-subcarrier equalisation of block spectra ``freq``
    (``(..., n_subcarriers)``) by the channel response ``H``: the
    ``conj(H)/(|H|^2 + eps)`` form with ``eps`` from :func:`regularizer` —
    zero-forcing with a floor, or the MMSE weight when ``snr_linear`` is
    given. A channel with no power anywhere returns zeros: every subcarrier
    is unrecoverable, which is what the epsilon form tends to. ``eps`` scales
    with ``|H|^2`` so a pilot estimate at any receive amplitude is used: with
    a fixed offset, 16-QAM over a 4-tap channel measured BER 0.24 at an
    amplitude of 1e-12 (MMSE at 1e-9)."""
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


# ──────────────────────────────────────────────────────────────────────
# Direct-sequence spreading
#
# Spreading a symbol over a code.
# ──────────────────────────────────────────────────────────────────────

def m_sequence(n_register, taps):
    """Maximal-length PN sequence (``+/-1``) of length ``2**n_register - 1``.

    ``taps`` are the 1-based feedback-tap positions of the LFSR (e.g. ``[5, 2]``
    for a length-31 sequence). Chips use the standard BPSK mapping
    ``s = 1 - 2*bit`` (bit 0 → +1, bit 1 → -1). Sibling of
    :func:`uacpy.acoustic_signal.generate.mseq`, which generates m-sequences
    from preset polynomials keyed by register length with the same chip
    polarity — this variant takes the taps explicitly for spreading-code
    experiments.
    """
    n = int(n_register)
    tap_list = [int(t) for t in taps]
    if n < 2:
        raise ConfigurationError(
            f"m_sequence: n_register must be >= 2; got {n_register!r}")
    if not tap_list or any(t < 1 or t > n for t in tap_list):
        raise ConfigurationError(
            f"m_sequence: taps must be 1-based positions in 1..{n}; "
            f"got {taps!r}")
    if len(set(tap_list)) != len(tap_list):
        raise ConfigurationError(
            f"m_sequence: repeated tap position in {taps!r}; a tap XORed with "
            "itself cancels, leaving the register with no feedback")
    seed = (1,) * n
    reg = list(seed)
    length = (1 << n) - 1
    seq = np.empty(length, dtype=int)
    period = None
    for i in range(length):
        seq[i] = reg[-1]
        fb = 0
        for t in tap_list:
            fb ^= reg[t - 1]
        reg = [fb] + reg[:-1]
        if period is None and tuple(reg) == seed:
            period = i + 1
    # A non-primitive polynomial returns early to the seed, so the register
    # cycles through a subset of its states. The output still has the right
    # length, dtype and +/-1 alphabet, and processing_gain_dB still reports
    # 10log10(N) — nothing looks wrong until a link budget is far out.
    if period != length:
        raise ConfigurationError(
            f"m_sequence: taps {taps!r} on a {n}-stage register give an LFSR "
            f"period of {period if period is not None else '>' + str(length)}, "
            f"not the maximal {length} — the feedback polynomial is not "
            f"primitive, so the output is not an m-sequence and its off-peak "
            f"autocorrelation is far above 1.",
            remediation=f"Use uacpy.acoustic_signal.generate.mseq({n}) for a "
                        f"preset primitive polynomial, or a known primitive "
                        f"tap set (e.g. [5,2], [5,3], [6,1], [6,5], [7,6]).")
    return 1 - 2 * seq          # {0,1} -> {+1,-1}


def spread(symbols, code):
    """Spread each symbol by the chip ``code`` (Kronecker product). Returns chips."""
    s = np.asarray(symbols, dtype=complex).ravel()
    c = np.asarray(code, dtype=complex).ravel()
    if c.size < 1:
        raise ConfigurationError(
            "spread: empty code (0 chips); pass the spreading sequence the "
            "receiver will despread with, e.g. m_sequence(n_stages, taps).")
    return np.kron(s, c)


def despread(chips, code):
    """Correlate chips against ``code`` per symbol period -> symbol estimates.

    The correlation is normalised by the code energy ``sum(|c|^2)``, so
    ``despread(spread(s, c), c)`` returns ``s`` itself for any code amplitude
    (dividing by the chip count is equivalent only for unit-modulus codes).
    A trailing partial symbol (fewer than ``len(code)`` chips) is dropped.
    """
    c = np.asarray(code, dtype=complex).ravel()
    x = np.asarray(chips, dtype=complex).ravel()
    if c.size < 1:
        raise ConfigurationError(
            "despread: empty code (0 chips); pass the same spreading "
            "sequence spread() was called with.")
    energy = float(np.sum(np.abs(c) ** 2))
    if energy == 0.0:
        raise ConfigurationError(
            "despread: code energy sum(|c|^2) is zero (every chip is 0), so "
            "the per-symbol correlation carries no signal and the "
            "normalisation is undefined")
    n = c.size
    nsym = x.size // n
    blocks = x[: nsym * n].reshape(nsym, n)
    return (blocks @ np.conj(c)) / energy


def processing_gain_dB(code):
    """Processing gain ``10*log10(N)`` in dB for an ``N``-chip code."""
    return float(10.0 * np.log10(np.asarray(code).size))


# ──────────────────────────────────────────────────────────────────────
# Forward error correction
#
# Convolutional coding and Viterbi decoding.
# ──────────────────────────────────────────────────────────────────────

# Rate-1/2 maximum-free-distance generator pair for K=7 (Proakis & Salehi
# Table 8.3-1, after Odenwalder 1970 / Larsen 1973): d_free = 10, which meets
# that table's upper bound, so no other K=7 rate-1/2 code does better.
DEFAULT_POLYS = (0o171, 0o133)
DEFAULT_K = 7


def _poly_bits(poly, K):
    return [(poly >> (K - 1 - i)) & 1 for i in range(K)]


def conv_encode(bits, polys=DEFAULT_POLYS, K=DEFAULT_K):
    """Rate ``1/len(polys)`` convolutional encoding with zero tail-flush.

    Returns the coded bit stream (``len(polys)`` output bits per input bit,
    including the ``K-1`` flush bits). ``polys`` are octal generator taps.
    """
    b = list(np.asarray(bits, dtype=int).ravel()) + [0] * (K - 1)
    taps = [_poly_bits(p, K) for p in polys]
    reg = [0] * K
    out = []
    for bit in b:
        reg = [bit] + reg[:-1]
        for t in taps:
            out.append(sum(r & g for r, g in zip(reg, t)) & 1)
    return np.array(out, dtype=int)


def viterbi_decode(coded, polys=DEFAULT_POLYS, K=DEFAULT_K):
    """Hard-decision Viterbi decoding (inverse of :func:`conv_encode`).

    Returns the decoded information bits (tail removed).
    """
    # The package's only Viterbi decoder. uacpy.comms.janus.janus_decode used
    # to carry a specialised K=9 twin with precomputed trellis words and a
    # comment obliging each side to mirror the other's add-compare-select and
    # traceback changes; it now calls this one with the reversed CMRE
    # generators, which was verified bit-exact against that twin on 300 random
    # packets clean and with 1-12 chip flips. There is nothing left to mirror.
    n = len(polys)
    c = np.asarray(coded, dtype=int).ravel()
    c = c[: (c.size // n) * n]            # drop a trailing partial symbol (garbage)
    taps = [_poly_bits(p, K) for p in polys]
    n_states = 1 << (K - 1)

    def outputs(state, bit):
        """Encoder output word for a state transition on ``bit``."""
        reg = [bit] + [(state >> (K - 2 - i)) & 1 for i in range(K - 1)]
        return [sum(r & g for r, g in zip(reg, t)) & 1 for t in taps]

    # Butterfly structure: the transition (st, bit) -> ((bit << (K-2)) |
    # (st >> 1)) means state s is reached from exactly two predecessors,
    # 2s and 2s+1 (mod n_states), both on input bit = the top bit of s.
    states = np.arange(n_states)
    prev0 = (states << 1) & (n_states - 1)
    prev1 = prev0 | 1
    bit_in = states >> (K - 2)
    # branch[st, b] = encoder output word (n bits) leaving state st on bit b
    branch = np.array([[outputs(st, b) for b in (0, 1)]
                       for st in range(n_states)], dtype=int)

    nsteps = c.size // n
    rx = c.reshape(nsteps, n)
    # Hamming branch metrics per step for the two transitions into each state:
    # bm0[k, s] = distance of rx[k] from the word on prev0[s] -> s.
    bm_all = np.count_nonzero(branch[None, :, :, :] ^ rx[:, None, None, :],
                              axis=3)
    bm0 = bm_all[:, prev0, bit_in]
    bm1 = bm_all[:, prev1, bit_in]

    bits = viterbi_hard(bm0, bm1, prev0, prev1, n_states,
                        lambda state: (state >> (K - 2)) & 1)
    return bits[: nsteps - (K - 1)]


def viterbi_hard(bm0, bm1, prev0, prev1, n_states, bit_of_state):
    """Hard-decision Viterbi over a rate-1/2 butterfly trellis.

    ``bm0[k]``/``bm1[k]`` are the branch metrics into every state at step
    ``k`` from its two predecessors ``prev0``/``prev1``; ``bit_of_state``
    reads the input bit off the arriving state. Starts and ends in state 0
    (the tail flush), so the traceback begins there rather than at the
    survivor with the smallest metric, which noisy input could move.
    Factored out of :func:`viterbi_decode` so a caller that already holds its
    branch metrics can reuse the survivor selection without rebuilding them.
    """
    nsteps = len(bm0)
    pm = np.full(n_states, np.inf)
    pm[0] = 0.0
    back = np.zeros((nsteps, n_states), dtype=np.int32)   # prev-state index
    for k in range(nsteps):
        cand0 = pm[prev0] + bm0[k]
        cand1 = pm[prev1] + bm1[k]
        # Strict < keeps the even predecessor on a tie — the same survivor a
        # state-ascending scan that replaces only on improvement selects.
        take1 = cand1 < cand0
        pm = np.where(take1, cand1, cand0)
        back[k] = np.where(take1, prev1, prev0)
    state = 0
    bits = np.zeros(nsteps, dtype=int)
    for k in range(nsteps - 1, -1, -1):
        bits[k] = bit_of_state(state)
        state = back[k, state]
    return bits


class ConvCode:
    """Configurable convolutional codec bundling encode/decode + interleaving.

    Holds the generator polynomials, constraint length and (optional) block
    interleaver depth in one place, so :meth:`encode` and :meth:`decode` always
    use matching settings.

    Parameters
    ----------
    polys : tuple of int
        Octal generator polynomials (rate ``1/len(polys)``).
    K : int
        Constraint length.
    interleave_depth : int, optional
        Block interleaver depth; ``None`` disables interleaving.

    Examples
    --------
    >>> import numpy as np
    >>> bits = np.random.default_rng(0).integers(0, 2, 64)
    >>> code = ConvCode(interleave_depth=16)
    >>> rx = code.decode(code.encode(bits))
    >>> bool(np.array_equal(np.asarray(rx)[:bits.size], bits))
    True
    """

    def __init__(self, polys=DEFAULT_POLYS, K: int = DEFAULT_K,
                 interleave_depth=None):
        self.polys = tuple(polys)
        self.K = int(K)
        self.interleave_depth = interleave_depth
        self._info_len = None   # last encoded payload length, for exact decode

    @property
    def rate(self):
        """Code rate ``1 / len(polys)``."""
        return 1.0 / len(self.polys)

    def encode(self, bits):
        """Encode information bits (convolutional + optional interleave)."""
        b = np.asarray(bits, dtype=int).ravel()
        self._info_len = b.size
        coded = conv_encode(b, self.polys, self.K)
        if self.interleave_depth:
            coded = interleave(coded, self.interleave_depth)
        return coded

    def decode(self, coded, info_len=None):
        """Decode (optional deinterleave + Viterbi) back to information bits.

        Parameters
        ----------
        info_len : int, optional
            Number of information bits to return — strips the
            interleaver's block padding (and any whole-block zeros an
            outer framing layer appended) so the payload comes back
            exactly. ``None`` falls back to the length recorded by this
            codec's most recent :meth:`encode` call (loopback use); a
            codec that decodes messages it did not just encode must pass
            ``info_len`` explicitly, otherwise the full Viterbi output
            is returned (or, worse, the *previous* message's length is
            applied). Across separate transmit/receive codecs no length
            is recorded and the caller slices.
        """
        if self.interleave_depth:
            coded = deinterleave(coded, self.interleave_depth)
        bits = viterbi_decode(coded, self.polys, self.K)
        n_keep = info_len if info_len is not None else self._info_len
        if n_keep is not None:
            bits = bits[: int(n_keep)]
        return bits


def _require_depth(caller: str, depth) -> int:
    """Reject an interleaver depth below 1.

    The block size is ``depth*depth``, so depth 0 divides by zero and a
    negative depth reaches the reshape as an unknown dimension — both bare
    numpy/Python errors with nothing naming the argument.
    """
    d = int(depth)
    if d < 1:
        raise ConfigurationError(
            f"{caller}: depth must be >= 1 (it is the side of the "
            f"depth x depth interleaver block); got {depth!r}")
    return d


def interleave(bits, depth):
    """Block-local ``depth x depth`` interleaver (transpose per square block).

    Operating in independent ``depth*depth`` blocks (zero-padded to a whole
    number of blocks) keeps the block boundaries aligned regardless of payload
    length, so an outer framing layer that pads to whole symbols (e.g. OFDM) only
    ever appends complete zero blocks. The interleaver therefore grows the stream
    to a multiple of ``depth*depth``; :class:`ConvCode` records the information
    length so :meth:`ConvCode.decode` can strip the resulting tail exactly.
    """
    b = np.asarray(bits, dtype=int).ravel()
    d = _require_depth("interleave", depth)
    block = d * d
    nblk = int(np.ceil(b.size / block))
    pad = np.zeros(nblk * block, dtype=int)
    pad[: b.size] = b
    return pad.reshape(nblk, d, d).transpose(0, 2, 1).reshape(-1)


def deinterleave(bits, depth):
    """Inverse of :func:`interleave`; drops a trailing partial (garbage) block."""
    b = np.asarray(bits, dtype=int).ravel()
    d = _require_depth("deinterleave", depth)
    block = d * d
    nblk = b.size // block
    b = b[: nblk * block]
    return b.reshape(nblk, d, d).transpose(0, 2, 1).reshape(-1)


# ──────────────────────────────────────────────────────────────────────
# Framing
#
# Packing a payload into bytes and back.
# ──────────────────────────────────────────────────────────────────────

_HEADER_BYTES = 4   # uint32 payload length
_CRC_BYTES = 4      # uint32 CRC-32


def bytes_to_bits(data):
    """Unpack bytes to an MSB-first 0/1 bit array."""
    if isinstance(data, str):
        raise ConfigurationError(
            "bytes_to_bits: data is a str; the framing layer carries bytes "
            "— encode it first, e.g. data.encode('utf-8').")
    return np.unpackbits(np.frombuffer(bytes(data), dtype=np.uint8))


def bits_to_bytes(bits):
    """Pack an MSB-first 0/1 bit array back to bytes (truncates a partial byte)."""
    b = np.asarray(bits, dtype=np.uint8).ravel()
    n = (b.size // 8) * 8
    return np.packbits(b[:n]).tobytes()


def pack_frame(payload):
    """Frame a byte payload as ``[len:4][payload][crc32:4]`` -> bit array."""
    payload = bytes(payload)
    header = len(payload).to_bytes(_HEADER_BYTES, "big")
    # The CRC covers the header as well as the payload: a corrupted length
    # field otherwise steers the slice that the CRC is then read from, so the
    # check can never see the error that caused it.
    crc = zlib.crc32(header + payload).to_bytes(_CRC_BYTES, "big")
    return bytes_to_bits(header + payload + crc)


def unpack_frame(bits):
    """Recover ``(payload_bytes, crc_ok)`` from a framed bit stream.

    Reads the length header, slices the payload, and checks the trailing
    CRC-32 over ``header + payload``. A stream that is too short, or whose
    length header does not fit the data supplied, is a **failed frame**, not a
    caller error: it returns ``(b"", False)``. Only ``crc_ok=True`` means the
    frame arrived intact — a receiver cannot distinguish "corrupt" from
    "malformed" and should not have to.
    """
    data = bits_to_bytes(bits)
    if len(data) < _HEADER_BYTES + _CRC_BYTES:
        return b"", False
    length = int.from_bytes(data[:_HEADER_BYTES], "big")
    end = _HEADER_BYTES + length
    if len(data) < end + _CRC_BYTES:
        return b"", False
    payload = data[_HEADER_BYTES:end]
    crc_rx = int.from_bytes(data[end:end + _CRC_BYTES], "big")
    return payload, zlib.crc32(data[:end]) == crc_rx
