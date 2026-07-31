"""JANUS — the NATO STANAG 4748 baseline underwater communications standard.

JANUS is the first internationally standardised digital underwater acoustic
protocol (NATO STANAG 4748, 2017): an open, deliberately simple FH-BFSK scheme
meant as an interoperability beacon between otherwise incompatible modems. This
module implements the **baseline 64-bit packet** — its field layout, CRC-8,
rate-1/2 K=9 convolutional coding, depth-13 interleaving — and the **FH-BFSK
physical layer** (frequency-hopped binary FSK with the standard tone table, the
32-chip detection preamble, optional Tukey chip windowing and wake-up tones).

The receiver mirrors the CMRE reference as a batch (post-processing) pipeline:
the recording is resampled once to a canonical rate (integer samples/chip) so any
``sample_rate`` works; the preamble is found by the Goertzel-bank chips-alignment
statistic + Greatest-Of CFAR detector (``chips_alignment`` + ``go_cfar``); and
wideband Doppler is removed by the CMRE preamble-tone estimator (windowed-FFT peak
+ 9-point quadratic interpolation per chip, median scale, single resample by it).

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
    implementation (GPLv3, janus-c 3.0.5): packet.c / crc.c / trellis.c / convolve.c /
    interleave.c / hop_index.c / primitive.c / modulator.c (encoder); chips_alignment.c /
    go_cfar.c / doppler.c (receiver detection, CFAR and Doppler).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from uacpy.core.constants import DEFAULT_SOUND_SPEED
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
_N_TOTAL_CHIPS = _PREAMBLE_CHIPS + _N_CODED   # 176: 32 preamble + 144 data

WAKEUP_GAP = 0.4                   # s silence after wake-up tones (CMRE JANUS_WUT_GAP)
DEFAULT_DOPPLER_MAX_SPEED = 5.0    # m/s search half-range (CMRE reference default)
_DOPPLER_FIT_POINTS = 9            # CMRE DOPPLER_INT_NPOINT: quadratic peak interpolation
_DOPPLER_C0 = 1540.0               # m/s reference speed in CMRE chips_alignment/doppler

# GO-CFAR preamble detector (CMRE chips_alignment + go_cfar, batch form).
_CHIP_OVERSAMPLING = 4             # CMRE JANUS_PREAMBLE_CHIP_OVERSAMPLING
_CFAR_THRESHOLD = 2.5             # CMRE default detection_threshold
_CFAR_WINDOW_CORRECTION = 0.1538   # CMRE window_correction factor
_CFAR_CHANNEL_SPREAD = 64          # chips searched for the peak past first detection
_CFAR_MOV_AVG_TIME = 0.150         # s training-window length floor


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
        version = _bits_int(b[0:4])
        if version != 3:
            import warnings
            warnings.warn(
                f"JanusPacket.from_bits: packet version {version} != 3 "
                f"(the only version this codec implements); field layout "
                f"may not match.",
                UserWarning, stacklevel=2,
            )
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
    """Return ``(f_low, fsw)`` for a JANUS band (FSw = Bw/26)."""
    fsw = bw / (2 * _N_SLOTS)            # 26 slots
    f_low = fc - bw / 2
    return f_low, fsw


def _chip_bounds(n_chips, cd, fs):
    """Per-chip sample edges with the chip duration tracked at sub-sample precision.

    The CMRE reference places chip ``k`` at ``[round(k*cd*fs), round((k+1)*cd*fs))``
    rather than a fixed integer length, so chip boundaries never drift when
    ``cd*fs`` is non-integer (e.g. 6.25 ms at 44.1 kHz = 275.625 samples).
    """
    return np.round(np.arange(n_chips + 1) * cd * fs).astype(int)


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
    bounds = _chip_bounds(_N_TOTAL_CHIPS, cd, fs)

    chips = []
    if wakeup:
        n_chip = bounds[1] - bounds[0]
        for wf in (fc - bw / 2, fc, fc + bw / 2):
            chips.append(np.cos(2 * np.pi * wf * np.arange(4 * n_chip) / fs))
        chips.append(np.zeros(int(round(WAKEUP_GAP * fs))))

    # 32 preamble chips (hops fh[0:32], bits FH_PREAMBLE_BITS) + 144 data chips.
    chip_bits = np.concatenate([FH_PREAMBLE_BITS, sym])
    for k in range(_N_TOTAL_CHIPS):
        f = _tone_freq(fh[k], chip_bits[k], f_low, fsw)
        n = bounds[k + 1] - bounds[k]
        t = np.arange(n) / fs
        win = _tukey(n, 0.05) if tukey else 1.0
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


def _canonical_fs(cd, fs):
    """Sample rate giving an integer, oversampling-aligned chip length.

    Returns ``(samples_per_chip, fs_c)`` with ``samples_per_chip`` a multiple of
    the chip oversampling (so column hops are integers). A recording is resampled
    to ``fs_c`` once up front; everything downstream then assumes integer chips.
    """
    m = _CHIP_OVERSAMPLING * int(round(cd * fs / _CHIP_OVERSAMPLING))
    return m, m / cd


def _resample(x, scale):
    """Time-scale ``x`` by ``scale`` (>1 lengthens) via linear interpolation."""
    if scale == 1.0:
        return x
    n_new = max(1, int(round(x.size * scale)))
    return np.interp(np.linspace(0.0, x.size - 1, n_new), np.arange(x.size), x)


def _chips_alignment(x, fs, f_low, fsw, fh, cd, max_speed):
    """CMRE chips-alignment detection statistic (Goertzel bank, batch form).

    For every quarter-chip column, a Hamming-windowed single-bin DFT (Goertzel) is
    taken at each preamble tone over a Doppler-shrunk window; a 2-point max filter
    across columns absorbs sub-chip timing, and the magnitudes one chip apart are
    summed over the 32-chip preamble pattern. Returns ``(statistic, hop_samples)``.
    """
    chip = cd * fs
    hstep = int(round(chip / _CHIP_OVERSAMPLING))
    tones = np.array([_tone_freq(fh[k], FH_PREAMBLE_BITS[k], f_low, fsw)
                      for k in range(_PREAMBLE_CHIPS)])
    gf = int(np.floor(fs * (_DOPPLER_C0 * cd)
                      / ((cd * float(tones.max()) + 1) * max_speed + _DOPPLER_C0)))
    gf = max(gf, int(0.75 * chip))
    if x.size < gf + hstep:
        return None, hstep
    ref = np.hamming(gf)[:, None] * np.exp(
        -2j * np.pi * np.outer(np.arange(gf), tones) / fs)
    mag = np.abs(sliding_window_view(x, gf)[::hstep] @ ref) / _PREAMBLE_CHIPS
    mag = np.maximum(mag[:-1], mag[1:])                       # 2-point max filter
    nstat = mag.shape[0] - _CHIP_OVERSAMPLING * (_PREAMBLE_CHIPS - 1)
    if nstat <= 0:
        return None, hstep
    cols = (np.arange(nstat)[:, None]
            + _CHIP_OVERSAMPLING * np.arange(_PREAMBLE_CHIPS)[None, :])
    stat = mag[cols, np.arange(_PREAMBLE_CHIPS)[None, :]].sum(axis=1)
    return stat, hstep


def _go_cfar(stat, cd):
    """Greatest-Of CFAR detection on the chips-alignment statistic (CMRE go_cfar).

    Adaptive threshold from guard-separated left/right training windows; returns
    the column of the peak within one channel-spread past the first threshold
    crossing (argmax fallback if nothing crosses).
    """
    if stat.size == 0:
        return None
    if stat.size < 2:
        return int(np.argmax(stat))
    mov_avg = max(_CFAR_MOV_AVG_TIME, 2 * _N_SLOTS * cd)
    m = int(np.floor(_CHIP_OVERSAMPLING * mov_avg / cd))      # training half (cols)
    hg = _CHIP_OVERSAMPLING                                   # half guard
    hn = m + hg
    n = stat.size
    # Reflect-pad the statistic so edge cells get a representative (not zero-biased)
    # training background, preserving CFAR's constant-false-alarm behaviour near the
    # clip boundaries; the streaming reference always has real stream context here.
    csum = np.concatenate([[0.0], np.cumsum(np.pad(stat, hn, mode="reflect"))])
    spread = _CFAR_CHANNEL_SPREAD * _CHIP_OVERSAMPLING
    first = None
    for i in range(n):
        z = stat[i]
        left = csum[(i - hg) + hn] - csum[(i - hn) + hn]
        right = csum[(i + hn) + hn] - csum[(i + hg) + hn]
        z_go = max(left, right - z * _CFAR_WINDOW_CORRECTION) * _CFAR_THRESHOLD / m
        if z > z_go and z > 1e-9:
            first = i
            break
    if first is None:
        return int(np.argmax(stat))
    return first + int(np.argmax(stat[first:min(first + spread, n)]))


def _detect(x, fs, f_low, fsw, fh, cd, max_speed):
    """Preamble start sample in ``x`` (assumed already at the canonical rate)."""
    stat, hstep = _chips_alignment(x, fs, f_low, fsw, fh, cd, max_speed)
    if stat is None:
        return None, stat
    m0 = _go_cfar(stat, cd)
    if m0 is None:
        return None, stat
    coarse = m0 * hstep
    tones = np.array([_tone_freq(fh[k], FH_PREAMBLE_BITS[k], f_low, fsw)
                      for k in range(_PREAMBLE_CHIPS)])
    rel = _chip_bounds(_PREAMBLE_CHIPS, cd, fs)
    best, start = -1.0, coarse
    for s in range(max(coarse - 2 * hstep, 0), coarse + 2 * hstep):
        if s + rel[-1] > x.size:
            break
        tot = sum(_chip_energy(x[s + rel[k]:s + rel[k + 1]], tones[k], fs)
                  for k in range(_PREAMBLE_CHIPS))
        if tot > best:
            best, start = tot, s
    return start, stat


def janus_detect(waveform, sample_rate=48000.0, fc=FC_INITIAL, bw=BW_INITIAL,
                 cd=None, fh_seq=None, doppler_max_speed=DEFAULT_DOPPLER_MAX_SPEED):
    """Locate the 32-chip preamble (CMRE GO-CFAR). Returns ``(start, statistic)``.

    The recording is resampled once to a canonical rate (integer samples/chip)
    before the Goertzel-bank + GO-CFAR detector runs; ``start`` is mapped back to a
    sample index in the original ``waveform`` (``None`` if no packet is found).
    ``statistic`` is the chips-alignment statistic over quarter-chip columns.
    """
    f_low, fsw = _band_params(fc, bw)
    cd = 1.0 / fsw if cd is None else float(cd)
    fh = FH_SEQUENCE if fh_seq is None else np.asarray(fh_seq, dtype=int)
    fs = float(sample_rate)
    _, fs_c = _canonical_fs(cd, fs)
    xc = _resample(np.asarray(waveform, dtype=float), fs_c / fs)
    start, stat = _detect(xc, fs_c, f_low, fsw, fh, cd, doppler_max_speed)
    if start is None:
        return None, stat
    return int(round(start * fs / fs_c)), stat


def _estimate_doppler(x, start, f_low, fsw, fh, cd, fs, max_speed, sound_speed):
    """Estimate the Doppler scale ``gamma`` from the preamble (CMRE method).

    Replicates ``janus_doppler_execute``: for each known preamble chip, take a
    Hamming-windowed FFT, locate the tone's spectral peak, refine it by a
    9-point least-squares quadratic fit, and form a per-chip factor
    ``gamma = 1 + df / f_tone``. The median over the chips whose implied speed is
    within ``max_speed`` is returned; the signal is then resampled by ``gamma``.
    """
    bounds = start + _chip_bounds(_PREAMBLE_CHIPS, cd, fs)
    gamma_tol = max_speed / sound_speed
    half = _DOPPLER_FIT_POINTS // 2
    gammas = []
    for k in range(_PREAMBLE_CHIPS):
        seg = x[bounds[k]:bounds[k + 1]]
        if seg.size < 8:
            continue
        f_k = _tone_freq(fh[k], FH_PREAMBLE_BITS[k], f_low, fsw)
        nfft = 1 << int(np.ceil(np.log2(seg.size * 16)))
        spec = np.abs(np.fft.rfft(seg * np.hamming(seg.size), nfft)) ** 2
        freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
        band = f_k * gamma_tol + 0.4 * fsw          # stay clear of neighbouring tones
        lo = int(np.searchsorted(freqs, f_k - band))
        hi = int(np.searchsorted(freqs, f_k + band))
        if hi - lo < 3:
            continue
        p = lo + int(np.argmax(spec[lo:hi]))
        a0, b0 = max(p - half, 1), min(p + half + 1, freqs.size)
        if b0 - a0 < 3:
            continue
        c2, c1, _ = np.polyfit(freqs[a0:b0] - f_k, spec[a0:b0], 2)
        if c2 >= 0:                                 # require a concave peak
            continue
        gamma = 1.0 + (-c1 / (2.0 * c2)) / f_k
        if abs(gamma - 1.0) < gamma_tol:
            gammas.append(gamma)
    return float(np.median(gammas)) if gammas else 1.0


def janus_demodulate(waveform, sample_rate=48000.0, fc=FC_INITIAL, bw=BW_INITIAL,
                     cd=None, fh_seq=None, start=None,
                     doppler_max_speed=DEFAULT_DOPPLER_MAX_SPEED,
                     sound_speed=DEFAULT_SOUND_SPEED):
    """Demodulate a JANUS waveform -> ``(bits64, crc_ok)``.

    The inverse of :func:`janus_modulate` (bits in, bits out). The recording is
    first resampled to a canonical rate (integer samples/chip), so any
    ``sample_rate`` works. The preamble is then located by the CMRE Goertzel-bank
    + GO-CFAR detector (unless ``start`` is given), wideband Doppler is removed by
    resampling to the scale estimated from the preamble tones (disable with
    ``doppler_max_speed=0``). Passing ``start=`` skips the detector *and*
    the Doppler compensation — use it only for clean, Doppler-free
    recordings. The 144 data chips are detected non-coherently
    and decoded. Parse the 64 bits with :meth:`JanusPacket.from_bits`, or use
    :func:`janus_receive`.
    """
    f_low, fsw = _band_params(fc, bw)
    cd = 1.0 / fsw if cd is None else float(cd)
    fh = FH_SEQUENCE if fh_seq is None else np.asarray(fh_seq, dtype=int)
    fs = float(sample_rate)
    _, fs_c = _canonical_fs(cd, fs)
    x = _resample(np.asarray(waveform, dtype=float), fs_c / fs)
    if start is not None:
        start = int(round(start * fs_c / fs))
    fs = fs_c

    if start is None:
        start, _ = _detect(x, fs, f_low, fsw, fh, cd, doppler_max_speed)
        if start is None:
            raise ConfigurationError("janus_demodulate: preamble not found")
        if doppler_max_speed > 0:
            gamma = _estimate_doppler(x, start, f_low, fsw, fh, cd, fs,
                                      doppler_max_speed, sound_speed)
            x = _resample(x, gamma)
            start, _ = _detect(x, fs, f_low, fsw, fh, cd, doppler_max_speed)
            if start is None:
                raise ConfigurationError("janus_demodulate: preamble not found")

    bounds = start + _chip_bounds(_N_TOTAL_CHIPS, cd, fs)
    sym = np.empty(_N_CODED, dtype=int)
    for i in range(_N_CODED):
        seg = x[bounds[_PREAMBLE_CHIPS + i]:bounds[_PREAMBLE_CHIPS + i + 1]]
        if seg.size < 1:
            seg = np.zeros(1)
        f0 = _tone_freq(fh[_PREAMBLE_CHIPS + i], 0, f_low, fsw)
        f1 = _tone_freq(fh[_PREAMBLE_CHIPS + i], 1, f_low, fsw)
        sym[i] = 1 if _chip_energy(seg, f1, fs) > _chip_energy(seg, f0, fs) else 0
    bits64 = janus_decode(sym)
    _, crc_ok = JanusPacket.from_bits(bits64)
    return bits64, crc_ok


def janus_transmit(packet: JanusPacket, sample_rate=48000.0, **kwargs):
    """Convenience: a :class:`JanusPacket` -> real JANUS waveform."""
    return janus_modulate(packet.to_bits(), sample_rate, **kwargs)


def janus_receive(waveform, sample_rate=48000.0, **kwargs):
    """Convenience: a JANUS waveform -> ``(JanusPacket, crc_ok)``."""
    bits64, crc_ok = janus_demodulate(waveform, sample_rate, **kwargs)
    pkt, _ = JanusPacket.from_bits(bits64)
    return pkt, crc_ok
