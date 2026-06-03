"""JANUS — the NATO STANAG 4748 baseline underwater communications standard.

JANUS is the first internationally standardised digital underwater acoustic
protocol (NATO STANAG 4748, 2017): an open, deliberately simple FH-BFSK scheme
meant as an interoperability beacon between otherwise incompatible modems. This
module implements the **baseline 64-bit packet** — its field layout, CRC-8,
rate-1/2 K=9 convolutional coding, depth-13 interleaving — and the **FH-BFSK
physical layer** (frequency-hopped binary FSK with the standard tone table, the
32-chip detection preamble, optional Tukey chip windowing and wake-up tones).

Conformance
-----------
Bit-exact to STANAG 4748 / the CMRE reference implementation for: the packet
bit-allocation, the CCITT CRC-8 ``x^8+x^2+x+1``, the convolutional generators
``g1=0o657, g2=0o435`` (k=9) with 8-bit zero flush -> 144 symbols, the depth-13
interleaver, the initial band (Fc=11520 Hz, Bw=4160 Hz, FSw=160 Hz, Cd=6.25 ms),
the Table III tone frequencies, the **frequency-hop sequence** (the CMRE
``janus_hop_index`` Galois-field generator with ``alpha=2, q=13`` — universal
across bands since ``nblock = Bw/(FSw*2) = 13`` always) and the **32-chip
preamble** (``JANUS_32_CHIP_SEQUENCE = 0xAEC7CD20``). Pass ``fh_seq=`` only to
experiment with non-standard hop orders. **Verified interoperable** with the CMRE
janus-c 3.0.5 reference: uacpy's encoder is bit-exact to ``janus-tx`` coded-symbol
vectors, and uacpy decodes the reference implementation's emitted ``.wav`` back to
the original packet (the encoder convention — reversed-polynomial trellis,
``out[i]=conv[(i*13)%144]`` interleaver — was reverse-engineered from and checked
against the reference trellis tables).

References
----------
Potter, Alves, Green, Zappa, Nissen & McCoy (2014), *The JANUS Underwater
    Communications Standard*, IEEE UComms. NATO STANAG 4748. CMRE reference
    implementation (GPLv3, janus-c 3.0.5): hop_index.c / defaults.h / primitive.c.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from uacpy.core.exceptions import ConfigurationError

JANUS_VERSION = 3

# Initial JANUS acoustic band (STANAG 4748 sec. K)
FC_INITIAL = 11520.0        # centre frequency [Hz]
BW_INITIAL = 4160.0         # bandwidth [Hz]

# Convolutional code: rate 1/2, constraint length 9. The CMRE reference applies
# the generators (g1=0o657, g2=0o435) in reversed register bit-order, i.e. the
# output masks below (verified bit-exact against the reference trellis tables).
_CONV_K = 9
_G_HI = 0o753               # = bit-reversed g1 (0o657)
_G_LO = 0o561               # = bit-reversed g2 (0o435)
_N_STATES = 1 << (_CONV_K - 1)     # 256
_INTERLEAVE_DEPTH = 13
_N_INFO = 64                        # baseline packet bits
_N_CODED = 2 * (_N_INFO + (_CONV_K - 1))   # = 144 coded symbols
_PREAMBLE_CHIPS = 32
_N_SLOTS = 13                       # frequency-hop slot pairs


def _parity(x):
    return bin(x).count("1") & 1


def _build_trellis():
    """JANUS rate-1/2 K=9 trellis: ``(out_hi, out_lo)`` and next state per (state, bit)."""
    out = np.zeros((_N_STATES, 2, 2), dtype=int)
    nxt = np.zeros((_N_STATES, 2), dtype=int)
    for s in range(_N_STATES):
        for b in (0, 1):
            r = (b << 8) | s
            out[s, b] = (_parity(r & _G_HI), _parity(r & _G_LO))
            nxt[s, b] = ((b << 7) | (s >> 1)) & 0xFF
    return out, nxt


_TRELLIS_OUT, _TRELLIS_NXT = _build_trellis()

# Frequency-hop generator (CMRE reference: primitives table entry for 13 slots).
# nblock = Bw / (FSw * 2) = 13 for every JANUS band, so alpha/q are universal.
_FH_ALPHA = 2
_FH_Q = _N_SLOTS                    # = 13

# 32-chip detection/synchronisation preamble (CMRE JANUS_32_CHIP_SEQUENCE).
_PREAMBLE_WORD = 0xAEC7CD20
FH_PREAMBLE_BITS = np.array(
    [(_PREAMBLE_WORD >> (31 - i)) & 1 for i in range(32)], dtype=int)


def _hop_index(idx, alpha=_FH_ALPHA, q=_FH_Q):
    """CMRE ``janus_hop_index``: Galois-field FH slot for chip ``idx`` (0..q-1)."""
    u1 = -(-(idx + 1) // ((q - 1) * q))          # ceil((idx+1) / ((q-1)*q))
    u2 = idx // (q - 1)
    gp = (idx % (q - 1)) + 1
    b = pow(alpha, gp, q)
    return (b * (u1 + u2 * b)) % q


# Standard JANUS frequency-hop sequence (bit-exact to the CMRE reference).
FH_SEQUENCE = np.array([_hop_index(i) for i in range(256)], dtype=int)


def _crc8(bits):
    """JANUS CRC-8 (CCITT ``x^8 + x^2 + x + 1``, init 0) over a bit array -> 8 bits."""
    reg = 0
    for b in np.asarray(bits, dtype=int).ravel():
        reg ^= (int(b) & 1) << 7          # XOR data bit into the MSB, then shift
        if reg & 0x80:
            reg = ((reg << 1) ^ 0x07) & 0xFF
        else:
            reg = (reg << 1) & 0xFF
    out = [(reg >> (7 - i)) & 1 for i in range(8)]
    return np.array(out, dtype=int)


@dataclass
class JanusPacket:
    """A baseline 64-bit JANUS packet (STANAG 4748 Table I).

    The 34-bit Application Data Block (``app_data``) is user-defined per
    ``class_id`` / ``app_type``. ``mobility``, ``schedule``, ``tx_rx`` and
    ``forward`` are single-bit flags.
    """

    class_id: int = 16                 # 16 = NATO JANUS reference implementation
    app_type: int = 0                  # 0 = Emergency (per class 16)
    app_data: np.ndarray = field(default_factory=lambda: np.zeros(34, dtype=int))
    mobility: int = 0
    schedule: int = 0
    tx_rx: int = 1
    forward: int = 0

    def to_bits(self):
        """Encode to the 64-bit packet (56 payload bits + 8-bit CRC)."""
        if not 0 <= self.class_id < 256:
            raise ConfigurationError("JanusPacket: class_id must be 0..255")
        if not 0 <= self.app_type < 64:
            raise ConfigurationError("JanusPacket: app_type must be 0..63")
        adb = np.asarray(self.app_data, dtype=int).ravel()
        if adb.size != 34:
            raise ConfigurationError("JanusPacket: app_data must be 34 bits")
        bits = np.concatenate([
            _int_bits(JANUS_VERSION, 4),
            [self.mobility & 1, self.schedule & 1, self.tx_rx & 1, self.forward & 1],
            _int_bits(self.class_id, 8),
            _int_bits(self.app_type, 6),
            adb,
        ]).astype(int)                                 # 56 bits
        return np.concatenate([bits, _crc8(bits)])     # 64 bits

    @classmethod
    def from_bits(cls, bits64):
        """Decode a 64-bit packet. Returns ``(packet, crc_ok)``."""
        b = np.asarray(bits64, dtype=int).ravel()
        if b.size != 64:
            raise ConfigurationError("JanusPacket.from_bits: need exactly 64 bits")
        crc_ok = np.array_equal(_crc8(b[:56]), b[56:64])
        pkt = cls(
            class_id=_bits_int(b[8:16]),
            app_type=_bits_int(b[16:22]),
            app_data=b[22:56].copy(),
            mobility=int(b[4]), schedule=int(b[5]),
            tx_rx=int(b[6]), forward=int(b[7]),
        )
        return pkt, crc_ok


def _int_bits(value, n):
    return np.array([(int(value) >> (n - 1 - i)) & 1 for i in range(n)], dtype=int)


def _bits_int(bits):
    v = 0
    for b in np.asarray(bits, dtype=int).ravel():
        v = (v << 1) | int(b)
    return v


def _interleave_perm():
    """Depth-13 interleaver permutation over the 144 coded symbols (``out[i]=conv[perm[i]]``)."""
    return (np.arange(_N_CODED) * _INTERLEAVE_DEPTH) % _N_CODED


def janus_encode(bits64):
    """Baseline packet bits (64) -> 144 coded+interleaved channel symbols."""
    b = np.asarray(bits64, dtype=int).ravel()
    if b.size != 64:
        raise ConfigurationError("janus_encode: need exactly 64 packet bits")
    inp = np.concatenate([b, np.zeros(_CONV_K - 1, dtype=int)])   # 64 + 8 flush = 72
    conv = np.empty(_N_CODED, dtype=int)
    state = 0
    for n, bit in enumerate(inp):
        hi, lo = _TRELLIS_OUT[state, bit]
        conv[2 * n] = hi
        conv[2 * n + 1] = lo
        state = _TRELLIS_NXT[state, bit]
    return conv[_interleave_perm()]


def janus_decode(symbols144):
    """Inverse of :func:`janus_encode`: 144 symbols -> 64 packet bits (Viterbi)."""
    y = np.asarray(symbols144, dtype=int).ravel()
    if y.size != _N_CODED:
        raise ConfigurationError(f"janus_decode: need exactly {_N_CODED} symbols")
    conv = np.empty(_N_CODED, dtype=int)
    conv[_interleave_perm()] = y                       # de-interleave
    nsteps = _N_CODED // 2
    inf = float("inf")
    pm = [inf] * _N_STATES
    pm[0] = 0.0
    prev = np.zeros((nsteps, _N_STATES), dtype=np.int32)
    pbit = np.zeros((nsteps, _N_STATES), dtype=np.int8)
    for k in range(nsteps):
        r0, r1 = conv[2 * k], conv[2 * k + 1]
        npm = [inf] * _N_STATES
        for s in range(_N_STATES):
            if pm[s] == inf:
                continue
            for bit in (0, 1):
                hi, lo = _TRELLIS_OUT[s, bit]
                ns = _TRELLIS_NXT[s, bit]
                m = pm[s] + (r0 ^ hi) + (r1 ^ lo)
                if m < npm[ns]:
                    npm[ns] = m
                    prev[k, ns] = s
                    pbit[k, ns] = bit
        pm = npm
    state = 0                                          # tail-flushed to state 0
    bits = np.zeros(nsteps, dtype=int)
    for k in range(nsteps - 1, -1, -1):
        bits[k] = pbit[k, state]
        state = prev[k, state]
    return bits[:_N_INFO]


def _band_params(fc, bw):
    """Return ``(f_low, fsw, n_tones)`` for a JANUS band (FSw = Bw/26)."""
    fsw = bw / (2 * _N_SLOTS)            # 26 slots
    f_low = fc - bw / 2
    return f_low, fsw


def _tone_freq(fh_index, bit, f_low, fsw):
    """Tone frequency for a hop index + data bit (Table III, evenly spaced)."""
    return f_low + (2 * int(fh_index) + int(bit)) * fsw


def janus_modulate(bits64, sample_rate=48000.0, fc=FC_INITIAL, bw=BW_INITIAL,
                   cd=None, fh_seq=None, tukey=True, wakeup=False):
    """Generate the real FH-BFSK JANUS waveform for a 64-bit packet.

    The waveform is ``[optional wake-up tones][32-chip preamble][144 data chips]``;
    each chip is a CW tone (duration ``cd``, default ``1/FSw`` = 6.25 ms in the
    initial band) at the frequency selected by the hop index and the bit value.

    Returns the real-valued passband waveform (the tones already sit in the
    acoustic band, so no separate up-conversion is needed).
    """
    sym = janus_encode(bits64)
    f_low, fsw = _band_params(fc, bw)
    cd = 1.0 / fsw if cd is None else float(cd)
    fh = FH_SEQUENCE if fh_seq is None else np.asarray(fh_seq, dtype=int)
    fs = float(sample_rate)
    n_chip = int(round(cd * fs))
    t = np.arange(n_chip) / fs
    win = _tukey(n_chip, 0.05) if tukey else np.ones(n_chip)

    chips = []
    if wakeup:
        for wf in (fc - bw / 2, fc, fc + bw / 2):
            chips.append(np.cos(2 * np.pi * wf * np.arange(4 * n_chip) / fs))
        chips.append(np.zeros(int(round(0.4 * fs))))        # 0.4 s reverberation gap

    # 32-chip preamble: hop indices fh[0:32], bits = FH_PREAMBLE_BITS
    pre_bits = FH_PREAMBLE_BITS
    for k in range(_PREAMBLE_CHIPS):
        f = _tone_freq(fh[k], pre_bits[k], f_low, fsw)
        chips.append(np.cos(2 * np.pi * f * t) * win)
    # 144 data chips: hop indices continue after the preamble
    for i, bit in enumerate(sym):
        f = _tone_freq(fh[_PREAMBLE_CHIPS + i], bit, f_low, fsw)
        chips.append(np.cos(2 * np.pi * f * t) * win)
    return np.concatenate(chips)


def _tukey(n, alpha):
    """Tukey (tapered-cosine) window, ``alpha`` fraction tapered each end."""
    if alpha <= 0:
        return np.ones(n)
    w = np.ones(n)
    edge = int(np.floor(alpha * (n - 1) / 2.0))
    if edge < 1:
        return w
    k = np.arange(edge)
    taper = 0.5 * (1 + np.cos(np.pi * (k / edge - 1)))
    w[:edge] = taper
    w[-edge:] = taper[::-1]
    return w


def _chip_energy(seg, freq, fs):
    """Non-coherent energy at ``freq`` over a chip segment (single-bin DFT)."""
    n = np.arange(seg.size)
    return np.abs(np.sum(seg * np.exp(-2j * np.pi * freq * n / fs)))


def janus_detect(waveform, sample_rate=48000.0, fc=FC_INITIAL, bw=BW_INITIAL,
                 cd=None, fh_seq=None):
    """Locate the 32-chip preamble by matched filtering. Returns ``(start, metric)``.

    ``start`` is the sample index of the first preamble chip (``None`` if no peak
    clears the detection threshold); ``metric`` the normalized correlation.
    """
    f_low, fsw = _band_params(fc, bw)
    cd = 1.0 / fsw if cd is None else float(cd)
    fh = FH_SEQUENCE if fh_seq is None else np.asarray(fh_seq, dtype=int)
    fs = float(sample_rate)
    n_chip = int(round(cd * fs))
    t = np.arange(n_chip) / fs
    ref = np.concatenate([
        np.cos(2 * np.pi * _tone_freq(fh[k], FH_PREAMBLE_BITS[k], f_low, fsw) * t)
        for k in range(_PREAMBLE_CHIPS)])
    x = np.asarray(waveform, dtype=float)
    if x.size < ref.size:
        return None, np.zeros(0)
    corr = np.correlate(x, ref, mode="valid")
    pe = np.sum(ref ** 2)
    win = np.convolve(x ** 2, np.ones(ref.size), mode="valid")
    metric = np.abs(corr) / np.sqrt(pe * win + 1e-12)
    k = int(np.argmax(metric))
    return (k if metric[k] >= 0.5 else None), metric


def janus_demodulate(waveform, sample_rate=48000.0, fc=FC_INITIAL, bw=BW_INITIAL,
                     cd=None, fh_seq=None, start=None):
    """Demodulate a JANUS waveform -> ``(JanusPacket, crc_ok)``.

    Detects the preamble (unless ``start`` is given), then non-coherently detects
    each of the 144 data chips (energy at the two candidate tones) and decodes.
    """
    f_low, fsw = _band_params(fc, bw)
    cd = 1.0 / fsw if cd is None else float(cd)
    fh = FH_SEQUENCE if fh_seq is None else np.asarray(fh_seq, dtype=int)
    fs = float(sample_rate)
    n_chip = int(round(cd * fs))
    x = np.asarray(waveform, dtype=float)
    if start is None:
        start, _ = janus_detect(x, fs, fc, bw, cd, fh)
        if start is None:
            raise ConfigurationError("janus_demodulate: preamble not found")
    data0 = start + _PREAMBLE_CHIPS * n_chip
    sym = np.empty(_N_CODED, dtype=int)
    for i in range(_N_CODED):
        a = data0 + i * n_chip
        seg = x[a:a + n_chip]
        if seg.size < n_chip:
            seg = np.concatenate([seg, np.zeros(n_chip - seg.size)])
        f0 = _tone_freq(fh[_PREAMBLE_CHIPS + i], 0, f_low, fsw)
        f1 = _tone_freq(fh[_PREAMBLE_CHIPS + i], 1, f_low, fsw)
        sym[i] = 1 if _chip_energy(seg, f1, fs) > _chip_energy(seg, f0, fs) else 0
    return JanusPacket.from_bits(janus_decode(sym))


def transmit(packet: JanusPacket, sample_rate=48000.0, **kwargs):
    """Convenience: a :class:`JanusPacket` -> real JANUS waveform."""
    return janus_modulate(packet.to_bits(), sample_rate, **kwargs)


def receive(waveform, sample_rate=48000.0, **kwargs):
    """Convenience: a JANUS waveform -> ``(JanusPacket, crc_ok)``."""
    return janus_demodulate(waveform, sample_rate, **kwargs)
