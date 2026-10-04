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
wideband Doppler is removed by one resample at the scale a maximum-likelihood
Doppler filter bank estimates from the known preamble, jointly with the start
of the preamble (``_estimate_doppler``).

Conformance
-----------
Bit-exact to STANAG 4748 / the CMRE reference implementation for: the packet
bit-allocation, the CCITT CRC-8 ``x^8+x^2+x+1``, the convolutional generators
``g1=0o657, g2=0o435`` (k=9) with 8-bit zero flush -> 144 symbols, the depth-13
interleaver, the initial band (Fc=11520 Hz, Bw=4160 Hz, FSw=160 Hz, Cd=6.25 ms),
the STANAG 4748 Table III tone frequencies, the **frequency-hop sequence** (the CMRE
``janus_hop_index`` Galois-field generator with ``alpha=2, q=13`` — universal
across bands since ``nblock = Bw/(FSw*2) = 13`` always) and the **32-chip
preamble** (``JANUS_32_CHIP_SEQUENCE = 0xAEC7CD20``, janus-c
``external:defaults.h:61``). Pass ``hop_sequence=`` only to
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

    That tree is not vendored here, so line citations into it carry the
    ``external:`` prefix DEV.md §10 defines and no gate can check them.
"""

from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass, field

import numpy as np

from uacpy.acoustic_signal._results import ResultTuple

from uacpy.comms.coding import conv_encode, viterbi_decode
from numpy.lib.stride_tricks import sliding_window_view

from uacpy.core._export import CarrierExport
from uacpy.core._repr import FieldsRepr
from uacpy.core.constants import DEFAULT_SOUND_SPEED
from uacpy.core.exceptions import ConfigurationError, IOWarning
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.comms.constellations import _require_binary_bits

JANUS_VERSION = 3

# Initial JANUS acoustic band (STANAG 4748 sec. K)
FC_INITIAL = 11520.0        # centre frequency [Hz]
BW_INITIAL = 4160.0         # bandwidth [Hz]

# Convolutional code: rate 1/2, constraint length 9. The CMRE reference applies
# the generators (g1=0o657, g2=0o435) in reversed register bit-order, i.e. the
# reversed pair below (verified bit-exact against the reference trellis
# tables). The reversed pair is also the K=9 rate-1/2 maximum-free-distance
# code of Proakis & Salehi Table 8.3-1 (d_free = 12), an independent check on
# the bit-order convention. Fed in that order to the package's own
# ``coding.conv_encode`` / ``coding.viterbi_decode``, which take the generators
# per call, they reproduce the CMRE code exactly — so JANUS carries only its
# own interleaver, not its own codec.
_CONV_K = 9
_G_HI = 0o753               # = bit-reversed g1 (0o657)
_G_LO = 0o561               # = bit-reversed g2 (0o435)
_INTERLEAVE_DEPTH = 13
_N_INFO = 64                        # baseline packet bits
_N_CODED = 2 * (_N_INFO + (_CONV_K - 1))   # = 144 coded symbols
_PREAMBLE_CHIPS = 32
_N_SLOTS = 13                       # frequency-hop slot pairs
_N_TOTAL_CHIPS = _PREAMBLE_CHIPS + _N_CODED   # 176: 32 preamble + 144 data

WAKEUP_GAP = 0.4                   # s silence after wake-up tones (CMRE JANUS_WUT_GAP)
DEFAULT_DOPPLER_MAX_SPEED = 5.0    # m/s search half-range (CMRE reference default)
# Maximum-likelihood Doppler bank (_estimate_doppler): scales per grid, the
# coarse start-offset step and the fine offset half-width (samples).
_ML_SCALE_STEPS = 21
_ML_OFFSET_STEP = 3
_ML_FINE_OFFSETS = 3
# m/s: the reference sound speed both external:doppler.c:108 and
# external:chips_alignment.c:143 hard-code.
_DOPPLER_C0 = 1540.0

# GO-CFAR preamble detector (CMRE chips_alignment + go_cfar, batch form). Values
# read from the janus-c reference: external:defaults.h:63,
# external:parameters.c:65, external:rx.c:301, external:rx.c:407,
# external:rx.c:413.
_CHIP_OVERSAMPLING = 4             # JANUS_PREAMBLE_CHIP_OVERSAMPLING
_CFAR_THRESHOLD = 2.5              # params->detection_threshold
_CFAR_WINDOW_CORRECTION = 0.1538   # window_correction leading factor
# 116 chips, not the C implementation's 64. Two terms set the minimum:
# janus_modulate(wakeup=True) emits 12 wake-up chips + a 0.4 s gap (64
# chips at the initial band) = 76 chips BEFORE the preamble, and the
# 32-chip alignment sum reaches forward, so the statistic ramps up — and
# the CFAR threshold can trip — as much as 31 chips BEFORE the first
# wake-up sample. The argmax window past the first crossing must therefore
# cover 76 + 31 = 107 chips (+9 margin), or the preamble peak sits beyond
# the window and the round trip fails with crc_ok=False.
_CFAR_CHANNEL_SPREAD = 116         # chips past first detection (x oversampling in C)
_CFAR_MOV_AVG_TIME = 0.150         # s training-window length floor


# Frequency-hop generator (CMRE reference: primitives table entry for 13 slots).
# nblock = Bw / (FSw * 2) = 13 for every JANUS band, so alpha/q are universal.
_FH_ALPHA = 2
_FH_Q = _N_SLOTS                    # = 13

# 32-chip detection/synchronisation preamble (CMRE JANUS_32_CHIP_SEQUENCE,
# external:defaults.h:61): a 31-chip m-sequence plus one trailing 0. Potter
# et al. 2014 sec. H prints a different 31-bit pattern; the janus-c value is
# the one this module follows, because it is what the reference modem emits
# and decodes.
_PREAMBLE_WORD = 0xAEC7CD20
FH_PREAMBLE_BITS = np.array(
    [(_PREAMBLE_WORD >> (31 - i)) & 1 for i in range(32)], dtype=int)


def _hop_index(idx, alpha=_FH_ALPHA, q=_FH_Q):
    """CMRE ``janus_hop_index`` (janus-c ``hop_index.c``): Galois-field FH slot for chip
    ``idx`` (0..q-1); ``q = 13, alpha = 2`` is ``primitive.c``'s table entry for 13 blocks."""
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


@dataclass(eq=False)
class JanusPacket(FieldsRepr, CarrierExport):
    """A baseline 64-bit JANUS packet (STANAG 4748 Table I).

    The 34-bit Application Data Block (``app_data``) is user-defined per
    ``class_id`` / ``app_type``. ``mobility``, ``schedule``, ``tx_rx`` and
    ``forward`` are single-bit flags. On the export protocol as a carrier:
    ``to_dict`` / ``from_dict`` save and rebuild every field through the
    constructor, and ``to_xarray`` / ``to_netcdf`` carry them as JSON.
    """

    class_id: int = 16                 # 16 = NATO JANUS reference implementation
    app_type: int = 0                  # 0 = Emergency (per class 16)
    app_data: np.ndarray = field(default_factory=lambda: np.zeros(34, dtype=int))
    mobility: int = 0
    schedule: int = 0
    tx_rx: int = 1
    forward: int = 0

    def __eq__(self, other):
        """Field-wise equality, with ``app_data`` compared element-wise.

        Hand-written because ``app_data`` is an ndarray: the generated
        ``__eq__`` compares the field tuples, which puts an array inside a
        ``bool()`` and raises "The truth value of an array with more than one
        element is ambiguous" — so the first assertion a codec user writes,
        ``JanusPacket.from_bits(p.to_bits())[0] == p``, could not be written
        at all. Field-wise rather than ``to_bits()``-based so two packets
        differing only outside the 64 encoded bits still compare unequal.

        Defining ``__eq__`` in a class body is also what makes the class
        unhashable — Python sets ``__hash__`` to ``None`` for it — which is
        the contract wanted here, since the packet is mutable and
        ``app_data`` is an ndarray. No explicit ``__hash__ = None`` is needed;
        adding a real one would make a mutable object usable as a dict key.
        """
        if not isinstance(other, JanusPacket):
            return NotImplemented
        return (
            self.class_id == other.class_id
            and self.app_type == other.app_type
            and self.mobility == other.mobility
            and self.schedule == other.schedule
            and self.tx_rx == other.tx_rx
            and self.forward == other.forward
            and np.array_equal(np.asarray(self.app_data),
                               np.asarray(other.app_data))
        )


    def to_bits(self):
        """Encode to the 64-bit packet (56 payload bits + 8-bit CRC)."""
        if not 0 <= self.class_id < 256:
            raise ConfigurationError(
                f"JanusPacket: class_id must be 0..255; got {self.class_id!r}.")
        if not 0 <= self.app_type < 64:
            raise ConfigurationError(
                f"JanusPacket: app_type must be 0..63; got {self.app_type!r}.")
        adb = np.asarray(self.app_data, dtype=int).ravel()
        if adb.size != 34:
            raise ConfigurationError(
                f"JanusPacket: app_data must be 34 bits; got {adb.size}.")
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
            raise ConfigurationError(
                f"JanusPacket.from_bits: need exactly 64 bits; got {b.size}.")
        version = _bits_int(b[0:4])
        crc_ok = np.array_equal(_crc8(b[:56]), b[56:64])
        if version != 3 and crc_ok:
            # The warning fires only when the CRC vouches for the bits: on a
            # failed decode the version field is as garbled as the rest, and
            # a version-mismatch claim would point away from the real failure
            # (crc_ok=False already reports it).
            import warnings
            warnings.warn(
                f"JanusPacket.from_bits: packet version {version} != 3 "
                f"(the only version this codec implements); field layout "
                f"may not match.",
                IOWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
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
    """Depth-13 interleaver permutation over the 144 coded symbols (``out[i]=conv[perm[i]]``),
    janus-c ``interleave.c``: ``perm[i] = (perm[i-1] + q) % n`` with
    ``q = janus_interleave_q(144) = 13``."""
    return (np.arange(_N_CODED) * _INTERLEAVE_DEPTH) % _N_CODED


def janus_encode(bits64):
    """Baseline packet bits (64) -> 144 coded+interleaved channel symbols.

    Parameters
    ----------
    bits64 : array_like of int
        The 64 baseline-packet bits, 0/1.
    """
    b = _require_binary_bits("janus_encode", bits64)
    if b.size != 64:
        raise ConfigurationError(
            f"janus_encode: need exactly 64 packet bits; got {b.size}.")
    # The package's own rate-1/2 encoder with the reversed CMRE generators is
    # the JANUS code: ``conv_encode`` zero-flushes K-1 bits, so 64 + 8 input
    # bits give 2*72 = 144 coded symbols. Only the depth-13 interleaver below
    # is JANUS's own.
    conv = conv_encode(b, polys=(_G_HI, _G_LO), constraint_length=_CONV_K)
    return conv[_interleave_perm()]


def janus_decode(symbols144):
    """Inverse of :func:`janus_encode`: 144 symbols -> 64 packet bits (Viterbi).

    Parameters
    ----------
    symbols144 : array_like of int
        The 144 detected channel symbols, 0/1.
    """
    y = np.asarray(symbols144, dtype=int).ravel()
    if y.size != _N_CODED:
        raise ConfigurationError(
            f"janus_decode: need exactly {_N_CODED} symbols; got {y.size}.")
    conv = np.empty(_N_CODED, dtype=int)
    conv[_interleave_perm()] = y                       # de-interleave
    # The package's one Viterbi decoder, given the same reversed generators the
    # encoder used. It rebuilds its branch table per call (about 3.8 ms per
    # packet), which is affordable because janus_decode runs once per packet,
    # not per candidate alignment.
    return viterbi_decode(conv, polys=(_G_HI, _G_LO), constraint_length=_CONV_K)[:_N_INFO]


def _band_params(fc, bandwidth):
    """Return ``(freq_min, fsw)`` for a JANUS band (FSw = Bw/26)."""
    fsw = bandwidth / (2 * _N_SLOTS)            # 26 slots
    freq_min = fc - bandwidth / 2
    return freq_min, fsw


def _chip_bounds(n_chips, chip_duration, sample_rate):
    """Per-chip sample edges with the chip duration tracked at sub-sample precision.

    The CMRE reference places chip ``k`` at ``[round(k*cd*sample_rate), round((k+1)*cd*sample_rate))``
    rather than a fixed integer length, so chip boundaries never drift when
    ``cd*sample_rate`` is non-integer (e.g. 6.25 ms at 44.1 kHz = 275.625 samples).
    """
    return np.round(np.arange(n_chips + 1) * chip_duration * sample_rate).astype(int)


def _tone_freq(fh_index, bit, freq_min, fsw):
    """Tone frequency for a hop index + data bit (STANAG 4748 Table III, evenly spaced)."""
    return freq_min + (2 * int(fh_index) + int(bit)) * fsw


def _require_hop_sequence(hop_sequence, who):
    """Validate a hop sequence and return it as an int array.

    A JANUS band holds ``_N_SLOTS`` = 13 slot pairs, so a hop index outside
    ``0..12`` places its chip at ``freq_min + (2*fh + bit)*FSw``, outside the band
    the receiver searches — the waveform comes back the documented length and
    the tones are simply somewhere else. The modulator reads
    ``_N_TOTAL_CHIPS`` = 176 entries, so a shorter sequence indexes past its
    own end.
    """
    fh = np.asarray(hop_sequence, dtype=int)
    if fh.ndim != 1 or fh.size < _N_TOTAL_CHIPS:
        raise ConfigurationError(
            f"{who}: hop_sequence must be a 1-D sequence of at least "
            f"{_N_TOTAL_CHIPS} hop indices ({_PREAMBLE_CHIPS} preamble + "
            f"{_N_CODED} data chips); got shape {fh.shape}.")
    bad = (fh[:_N_TOTAL_CHIPS] < 0) | (fh[:_N_TOTAL_CHIPS] >= _N_SLOTS)
    if np.any(bad):
        raise ConfigurationError(
            f"{who}: hop_sequence hop indices must be in 0..{_N_SLOTS - 1} — a "
            f"JANUS band holds {_N_SLOTS} slot pairs and index k transmits at "
            f"freq_min + (2k + bit)*FSw, so a larger index puts the chip outside "
            f"the band. Got {int(np.count_nonzero(bad))} out-of-range "
            f"value(s) of {_N_TOTAL_CHIPS}, first "
            f"{int(fh[:_N_TOTAL_CHIPS][bad][0])} at index "
            f"{int(np.flatnonzero(bad)[0])}.")
    return fh


def _preamble_tones(fh, freq_min, fsw):
    """The 32 tone frequencies the preamble actually transmits, in chip order."""
    return np.array([_tone_freq(fh[k], FH_PREAMBLE_BITS[k], freq_min, fsw)
                     for k in range(_PREAMBLE_CHIPS)])


def janus_modulate(bits64, *, sample_rate=48000.0, fc=FC_INITIAL, bandwidth=BW_INITIAL,
                   chip_duration=None, hop_sequence=None, window='tukey', wakeup=False):
    """Generate the real FH-BFSK JANUS waveform for a 64-bit packet.

    The waveform is ``[optional wake-up tones][32-chip preamble][144 data chips]``;
    each chip is a CW tone (duration ``chip_duration``, default ``1/FSw`` = 6.25 ms in the
    initial band) at the frequency selected by the hop index and the bit value.

    ``window='tukey'`` tapers each chip with a tapered cosine over 5 % of its
    length (2.5 % at each end), softening the edge transient a rectangular
    chip would radiate; ``window=None`` emits the plain rectangular CW chip. The CMRE
    reference tapers too, but differently — external:modulator.c:78-90 puts a
    Hamming half-taper over ``lt/16`` samples (6.25 %) at each end, from 0.08
    rather than from 0 — so neither setting is sample-identical to it, and the
    two differ only over the outer few percent of each chip. ``wakeup``
    prepends the three wake-up tones, ``chip_duration`` overrides the chip duration and
    ``hop_sequence`` the hop sequence.

    Returns the real-valued passband waveform (the tones already sit in the
    acoustic band, so no separate up-conversion is needed).

    ``sample_rate`` must be at least ``2*(fc + bw/2)`` — twice the upper band
    edge — or a :class:`~uacpy.core.exceptions.ConfigurationError` is raised:
    an aliased waveform is undetectable downstream, because the loopback
    demodulator correlates with the same aliased tone kernels and decodes it
    cleanly.

    Parameters
    ----------
    bits64 : array_like of int
        The 64 baseline-packet bits, 0/1.
    sample_rate : float, optional
        Sample rate (Hz), at least ``2*(fc + bandwidth/2)``. Default 48 kHz.
    fc : float, optional
        Band centre frequency (Hz). Default 11520, the JANUS initial band.
    bandwidth : float, optional
        Bandwidth (Hz); the tone spacing is ``bandwidth/26``. Default 4160.
    chip_duration : float, optional
        Chip duration (s); ``None`` is one over the tone spacing, 6.25 ms in
        the initial band.
    hop_sequence : array_like of int, optional
        Frequency-hop indices, each in 0..12; ``None`` is the JANUS sequence.
    window : {'tukey', None}, optional
        Chip taper (see above). Default ``'tukey'``.
    wakeup : bool, optional
        Prepend the three wake-up tones. Default False.
    """
    if window not in ('tukey', None):
        raise ConfigurationError(
            f"janus_modulate: window must be 'tukey' (5 % tapered-cosine "
            f"chips) or None (rectangular chips); got {window!r}.")
    if sample_rate < 2 * (fc + bandwidth / 2):
        raise ConfigurationError(
            f"janus_modulate: the band edge fc + bw/2 = {fc + bandwidth / 2:g} Hz "
            f"is above the Nyquist frequency sample_rate/2 = "
            f"{sample_rate / 2:g} Hz, so the tones alias; require "
            f"sample_rate >= {2 * (fc + bandwidth / 2):g} Hz.")
    sym = janus_encode(bits64)
    freq_min, fsw = _band_params(fc, bandwidth)
    chip_duration = 1.0 / fsw if chip_duration is None else float(chip_duration)
    fh = (FH_SEQUENCE if hop_sequence is None else
          _require_hop_sequence(hop_sequence, "janus_modulate"))
    fs = float(sample_rate)
    bounds = _chip_bounds(_N_TOTAL_CHIPS, chip_duration, fs)

    chips = []
    if wakeup:
        # Each wake-up tone lasts four BASELINE chips, 4/FSw: a dyadic chip
        # duration set through `chip_duration` stretches the preamble and the data but
        # "does not affect wake-up tones" (Potter et al. 2014, sec. G).
        n_wakeup = int(round(4 * fs / fsw))
        for wf in (fc - bandwidth / 2, fc, fc + bandwidth / 2):
            chips.append(np.cos(2 * np.pi * wf * np.arange(n_wakeup) / fs))
        chips.append(np.zeros(int(round(WAKEUP_GAP * fs))))

    # 32 preamble chips (hops fh[0:32], bits FH_PREAMBLE_BITS) + 144 data chips.
    chip_bits = np.concatenate([FH_PREAMBLE_BITS, sym])
    for k in range(_N_TOTAL_CHIPS):
        f = _tone_freq(fh[k], chip_bits[k], freq_min, fsw)
        n = bounds[k + 1] - bounds[k]
        t = np.arange(n) / fs
        win = _tukey(n, 0.05) if window == 'tukey' else 1.0
        chips.append(np.cos(2 * np.pi * f * t) * win)
    return np.concatenate(chips)


def _tukey(n, alpha):
    """Tukey (tapered-cosine) window; ``alpha`` is the **total** tapered
    fraction, so each end gets ``alpha/2`` of the length."""
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


def _chip_energy(seg, freq, sample_rate):
    """Non-coherent energy at ``freq`` over a chip segment (single-bin DFT)."""
    n = np.arange(seg.size)
    return np.abs(np.sum(seg * np.exp(-2j * np.pi * freq * n / sample_rate)))


def _canonical_fs(chip_duration, sample_rate):
    """Sample rate giving an integer, oversampling-aligned chip length.

    Returns ``(samples_per_chip, fs_c)`` with ``samples_per_chip`` a multiple of
    the chip oversampling (so column hops are integers). A recording is resampled
    to ``fs_c`` once up front; everything downstream then assumes integer chips.
    """
    m = _CHIP_OVERSAMPLING * int(round(chip_duration * sample_rate / _CHIP_OVERSAMPLING))
    return m, m / chip_duration


def _resample(x, scale):
    """Time-scale ``x`` by ``scale`` (>1 lengthens) via linear interpolation.

    Linear interpolation, not :func:`~uacpy.comms.compensate_doppler`'s FFT
    resample, for the Doppler step: the two decode as well as each other.
    Measured with :func:`_estimate_doppler` supplying ``scale``, on exactly
    Doppler-shifted packets at 48 kHz, -12, -9 and -6 dB in-band SNR, -5 to
    +5 m/s, 16 seeds (528 records, each decoded by both): 263 CRC-valid and
    correct decodes here against 268 for the FFT resample, 12 records decoded
    by this one only and 17 by the FFT only (two-sided sign test p = 0.46).
    """
    if scale == 1.0:
        return x
    n_new = max(1, int(round(x.size * scale)))
    return np.interp(np.linspace(0.0, x.size - 1, n_new), np.arange(x.size), x)


def _chips_alignment(x, sample_rate, freq_min, fsw, fh, chip_duration, max_speed):
    """CMRE chips-alignment detection statistic (Goertzel bank, batch form).

    For every quarter-chip column, a Hamming-windowed single-bin DFT (Goertzel) is
    taken at each preamble tone over a Doppler-shrunk window; a 2-point max filter
    across columns absorbs sub-chip timing, and the magnitudes one chip apart are
    summed over the 32-chip preamble pattern. Returns ``(statistic, hop_samples)``.
    """
    chip = chip_duration * sample_rate
    hstep = int(round(chip / _CHIP_OVERSAMPLING))
    tones = _preamble_tones(fh, freq_min, fsw)
    # Goertzel window length, from CMRE external:chips_alignment.c:143-144
    # (``_DOPPLER_C0`` is that file's hard-coded 1540 m/s). At
    # ``max_speed = 0`` it reduces to exactly one chip, ``cd*sample_rate``; widening the
    # Doppler search, or moving to a band with higher tones, shortens it below
    # a chip so the window still fits within one Doppler-scaled chip. The floor
    # holds it at 3/4 chip, the reference's lower bound on the DFT length.
    # The operand is the highest PASSBAND preamble tone (13 440 Hz in the
    # initial band: freq_min + 25 * fsw), because this detector runs on the
    # passband record; the reference detects on the down-converted stream
    # and feeds the same formula the highest BASEBAND slot offset (+1920 Hz,
    # external:rx.c:329-332). At 5 m/s that is 0.78 chip here against 0.96 chip
    # there (~44 Hz of tone drift over a chip against a ~800 Hz Hamming
    # main lobe), a marginal difference that every detection statistic
    # nevertheless depends on.
    gf = int(np.floor(sample_rate * (_DOPPLER_C0 * chip_duration)
                      / ((chip_duration * float(tones.max()) + 1) * max_speed + _DOPPLER_C0)))
    gf = max(gf, int(0.75 * chip))
    if x.size < gf + hstep:
        return None, hstep
    ref = np.hamming(gf)[:, None] * np.exp(
        -2j * np.pi * np.outer(np.arange(gf), tones) / sample_rate)
    # Per-column normalisation applied before the max filter and the sum, as in
    # external:chips_alignment.c:276, so ``stat`` is the mean per-chip tone
    # magnitude.
    mag = np.abs(sliding_window_view(x, gf)[::hstep] @ ref) / _PREAMBLE_CHIPS
    mag = np.maximum(mag[:-1], mag[1:])                       # 2-point max filter
    nstat = mag.shape[0] - _CHIP_OVERSAMPLING * (_PREAMBLE_CHIPS - 1)
    if nstat <= 0:
        return None, hstep
    cols = (np.arange(nstat)[:, None]
            + _CHIP_OVERSAMPLING * np.arange(_PREAMBLE_CHIPS)[None, :])
    stat = mag[cols, np.arange(_PREAMBLE_CHIPS)[None, :]].sum(axis=1)
    return stat, hstep


def _cfar_window_correction(m):
    """Right-window self-contamination weight ``w`` for a training window of ``m`` cells.

    ``external:rx.c:413`` passes
    ``0.1538 * fmin((n_chips + dmod_chip_count) / (step_length // 4), 1)`` to
    ``janus_go_cfar_new``, whose ``external:go_cfar.c:327`` scales it by the
    training-window length ``hn - hg == step_length``. The weight is therefore
    proportional to ``m``: 16.0 at the initial band, not 0.1538.
    """
    return _CFAR_WINDOW_CORRECTION * m * min(_N_TOTAL_CHIPS / (m // 4), 1.0)


def _go_cfar(stat, chip_duration):
    """Greatest-Of CFAR detection on the chips-alignment statistic (CMRE go_cfar).

    Adaptive threshold from guard-separated left/right training windows.
    Returns ``(column, crossed)``: the column of the peak within one
    channel-spread past the first threshold crossing and ``True``, or, when
    nothing crosses, the argmax column and ``False`` (the fallback the
    reference does not have: it decodes only after a crossing).
    """
    if stat.size == 0:
        return None, False
    if stat.size < 2:
        return int(np.argmax(stat)), False
    mov_avg = max(_CFAR_MOV_AVG_TIME, 2 * _N_SLOTS * chip_duration)
    m = int(np.floor(_CHIP_OVERSAMPLING * mov_avg / chip_duration))      # training half (cols)
    hg = _CHIP_OVERSAMPLING                                   # half guard
    hn = m + hg
    w = _cfar_window_correction(m)
    n = stat.size
    # Reflect-pad the statistic so edge cells get a representative (not zero-biased)
    # training background, preserving CFAR's constant-false-alarm behaviour near the
    # clip boundaries; the streaming reference always has real stream context here.
    csum = np.concatenate([[0.0], np.cumsum(np.pad(stat, hn, mode="reflect"))])
    spread = _CFAR_CHANNEL_SPREAD * _CHIP_OVERSAMPLING
    # Detection floor relative to the statistic's own peak (an absolute floor
    # made crossing depend on the recording's amplitude scale: a quiet but
    # clean packet fell to the argmax fallback while a loud noise floor
    # crossed). All-zero statistic -> floor 0, nothing crosses, argmax below.
    z_floor = 1e-9 * float(np.max(stat))
    first = None
    for i in range(n):
        # The +hn offsets map stat index i onto the reflect-padded csum. In stat
        # coordinates the two training windows are [i-hn, i-hg) and [i+hg, i+hn):
        # m cells each, held off the cell under test by the hg-wide guard.
        z = stat[i]
        left = csum[(i - hg) + hn] - csum[(i - hn) + hn]
        right = csum[(i + hn) + hn] - csum[(i + hg) + hn]
        z_go = max(left, right - z * w) * _CFAR_THRESHOLD / m
        if z > z_go and z > z_floor:
            first = i
            break
    if first is None:
        return int(np.argmax(stat)), False
    return first + int(np.argmax(stat[first:min(first + spread, n)])), True


def _detect(x, sample_rate, freq_min, fsw, fh, chip_duration, max_speed):
    """``(start, statistic, crossed)``: the preamble start sample in ``x``
    (assumed already at the canonical rate), the chips-alignment statistic,
    and whether GO-CFAR crossed its threshold (``False``: the argmax
    fallback placed the start)."""
    stat, hstep = _chips_alignment(x, sample_rate, freq_min, fsw, fh, chip_duration, max_speed)
    if stat is None:
        return None, stat, False
    m0, crossed = _go_cfar(stat, chip_duration)
    if m0 is None:
        return None, stat, False
    coarse = m0 * hstep
    tones = _preamble_tones(fh, freq_min, fsw)
    rel = _chip_bounds(_PREAMBLE_CHIPS, chip_duration, sample_rate)
    best, start = -1.0, coarse
    for s in range(max(coarse - 2 * hstep, 0), coarse + 2 * hstep):
        if s + rel[-1] > x.size:
            break
        tot = sum(_chip_energy(x[s + rel[k]:s + rel[k + 1]], tones[k], sample_rate)
                  for k in range(_PREAMBLE_CHIPS))
        if tot > best:
            best, start = tot, s
    return start, stat, crossed


def janus_detect(waveform, sample_rate=48000.0, fc=FC_INITIAL, bandwidth=BW_INITIAL,
                 chip_duration=None, hop_sequence=None, doppler_max_speed=DEFAULT_DOPPLER_MAX_SPEED):
    """Locate the 32-chip preamble (CMRE GO-CFAR). Returns ``(start, statistic)``.

    The recording is resampled once to a canonical rate (integer samples/chip)
    before the Goertzel-bank + GO-CFAR detector runs; ``start`` is mapped back to a
    sample index in the original ``waveform``.
    ``statistic`` is the chips-alignment statistic over quarter-chip columns.

    ``start`` is ``None`` only when the recording is too short for the
    detector to run at all (shorter than one Goertzel frame, or than the
    32-chip alignment span). On any longer recording a start is **always**
    returned — when nothing crosses the GO-CFAR threshold the detector falls
    back to the argmax of the alignment statistic, so a noise-only recording
    yields the best-looking (spurious) candidate, not ``None``. Deciding
    whether a packet is really present is the caller's job (e.g. decode it:
    :func:`janus_demodulate` CRC-checks the baseline packet).

    Parameters
    ----------
    waveform : array_like
        The real passband recording.
    sample_rate : float, optional
        Sample rate (Hz), at least ``2*(fc + bandwidth/2)``. Default 48 kHz.
    fc : float, optional
        Band centre frequency (Hz). Default 11520, the JANUS initial band.
    bandwidth : float, optional
        Bandwidth (Hz); the tone spacing is ``bandwidth/26``. Default 4160.
    chip_duration : float, optional
        Chip duration (s); ``None`` is one over the tone spacing, 6.25 ms in
        the initial band.
    hop_sequence : array_like of int, optional
        Frequency-hop indices, each in 0..12; ``None`` is the JANUS sequence.
    doppler_max_speed : float, optional
        Half-range (m/s) of the radial speeds searched. It shortens the
        detector's Goertzel window as well as bounding the Doppler estimate;
        0 disables the Doppler compensation. Default 5.
    """
    freq_min, fsw = _band_params(fc, bandwidth)
    chip_duration = 1.0 / fsw if chip_duration is None else float(chip_duration)
    fh = (FH_SEQUENCE if hop_sequence is None else
          _require_hop_sequence(hop_sequence, "janus_detect"))
    fs = float(sample_rate)
    _, fs_c = _canonical_fs(chip_duration, fs)
    xc = _resample(np.asarray(waveform, dtype=float), fs_c / fs)
    start, stat, _ = _detect(xc, fs_c, freq_min, fsw, fh, chip_duration, doppler_max_speed)
    if start is None:
        return None, stat
    return int(round(start * fs / fs_c)), stat


def _estimate_doppler(x, start, freq_min, fsw, fh, chip_duration, sample_rate, max_speed, sound_speed):
    """Maximum-likelihood Doppler scale ``a = v/c`` over the known preamble.

    The Doppler filter bank of Abraham, *Underwater Acoustic Signal
    Processing* (2019), Sect. 8.7: a bank of replicas of the known signal over
    the scales of interest, whose peak response estimates the arrival time and
    the Doppler scale together. The replica for a scale ``a`` and a start
    offset ``d`` is the 32-chip preamble compressed by ``1 + a``: chip ``k``
    spans ``start + d + round(k cd sample_rate / (1 + a))`` and carries tone
    ``f_k (1 + a)``. Each chip is integrated coherently and the chips
    incoherently (unknown phase per chip), so the statistic is
    ``sum_k |sum_n x[n] exp(-2 pi i f_k (1 + a) n / sample_rate)|^2``. It is maximised
    on a coarse grid (``_ML_SCALE_STEPS`` scales over
    ``|a| <= max_speed / sound_speed``, offsets over a quarter chip either
    side in ``_ML_OFFSET_STEP`` samples), then on a fine grid around the
    coarse peak, and the scale is refined by a parabola. ``a`` is the
    package's scale (:func:`~uacpy.comms.doppler_from_speed`), positive for a
    closing range.

    The start offset is searched because the detector's start carries a bias
    of about -3.4 samples per m/s at 48 kHz (300 samples per chip), which a
    scale-only search trades against the scale. The CMRE reference instead
    takes each chip's spectral peak and the median of the per-chip scales
    (``janus_doppler_execute``, ``external:doppler.c:296``, where
    ``gamma = 1 + a``; its chip acceptance hard-codes c = 1540 m/s,
    ``external:doppler.c:108, 297-298``). Measured on exactly Doppler-shifted
    packets at 48 kHz with the default ``max_speed`` of 5 m/s:

    - at -9 and -6 dB in-band SNR the per-chip median comes out shrunk toward
      zero, at 14-51 % of the true scale; the slope of this estimate's error
      against the true speed is -0.002 at -6 dB and +0.001 at +20 dB (the
      median's: -0.50 at -6 dB);
    - over -12 to -6 dB and -5 to +5 m/s, 16 seeds, correct decodes are 260
      here against 197 for the median, out of 528; at v = 0 the two tie (29
      and 31 of 64 at -9 dB), because shrinking toward zero is shrinking
      toward the truth there;
    - over -5 to +5 m/s, 8 seeds, at +20 dB the median / p95 speed error is
      0.042 / 0.14 m/s here, against 0.040 / 0.20 m/s for the median;
    - it costs 57 ms against 5 ms, about 15 % of a whole decode.
    """
    a_max = max_speed / sound_speed
    tones = _preamble_tones(fh, freq_min, fsw)
    chip = chip_duration * sample_rate
    k = np.arange(_PREAMBLE_CHIPS)
    n = int(np.floor(chip / (1.0 + a_max)))     # fits every candidate's chip
    reach = int(round(chip / 4))
    margin = _ML_FINE_OFFSETS
    lo = start - reach - margin
    hi = (start + reach + margin + n
          + int(np.ceil(_PREAMBLE_CHIPS * chip / (1.0 - a_max))))
    seg = np.zeros(hi - lo)
    a0, b0 = max(lo, 0), min(hi, x.size)
    if b0 > a0:
        seg[a0 - lo:b0 - lo] = x[a0:b0]
    t = np.arange(n)

    def bank(scales, offsets):
        out = np.empty((scales.size, offsets.size))
        for i, a in enumerate(scales):
            edges = np.round(k * chip / (1.0 + a)).astype(int)
            replica = np.exp(-2j * np.pi * (tones * (1.0 + a))[:, None]
                             * t[None, :] / sample_rate)
            idx = ((start - lo) + offsets[:, None, None]
                   + edges[None, :, None] + t[None, None, :])
            out[i] = np.sum(np.abs(np.einsum('dkn,kn->dk', seg[idx],
                                             replica)) ** 2, axis=1)
        return out

    coarse_a = np.linspace(-a_max, a_max, _ML_SCALE_STEPS)
    coarse_d = np.arange(-reach, reach + 1, _ML_OFFSET_STEP)
    ia, id_ = np.unravel_index(int(np.argmax(bank(coarse_a, coarse_d))),
                               (coarse_a.size, coarse_d.size))
    step = coarse_a[1] - coarse_a[0] if coarse_a.size > 1 else 0.0
    fine_a = np.linspace(coarse_a[ia] - step, coarse_a[ia] + step,
                         _ML_SCALE_STEPS)
    fine_d = np.arange(coarse_d[id_] - margin, coarse_d[id_] + margin + 1)
    fine = bank(fine_a, fine_d)
    ja, jd = np.unravel_index(int(np.argmax(fine)), fine.shape)
    a = float(fine_a[ja])
    if 0 < ja < fine_a.size - 1:
        y0, y1, y2 = fine[ja - 1, jd], fine[ja, jd], fine[ja + 1, jd]
        curvature = y0 - 2.0 * y1 + y2
        if curvature < 0:
            a += 0.5 * (y0 - y2) / curvature * (fine_a[1] - fine_a[0])
    return float(np.clip(a, -a_max, a_max))


class JanusReception(ResultTuple,
                     namedtuple("JanusReception", "bits crc_ok")):
    """What :func:`janus_demodulate` returns: the 64 decoded ``bits`` and
    ``crc_ok``, whether the CRC-8 over the first 56 bits matches the last 8.

    The tuple is the decode, so ``bits, crc_ok = janus_demodulate(x, fs)``
    unpacks it. How the packet was found rides on attributes, which survive
    pickling and copying and take no part in equality:

    ``detected``
        Whether the GO-CFAR threshold was crossed on the recording
        (``False``: the start is the detection statistic's argmax). ``None``
        when ``start=`` was given and no detection ran.
    ``start``
        Sample index in the input ``waveform`` of the preamble start: the
        detector's (before Doppler compensation), or the ``start=`` given.
    ``doppler_scale``
        The Doppler scale ``a = v/c`` the record was compensated by
        (positive closing, :func:`~uacpy.comms.doppler_from_speed`); 0.0 when
        the compensation did not run (``doppler_max_speed=0`` or ``start=``).
    ``statistic``
        The chips-alignment statistic over quarter-chip columns that the
        detector searched; ``None`` when ``start=`` was given.
    """

    _attrs = ("detected", "start", "doppler_scale", "statistic")

    def __new__(cls, bits, crc_ok, *, detected=None, start=None,
                doppler_scale=0.0, statistic=None):
        self = super().__new__(cls, bits, crc_ok)
        self.detected = detected
        self.start = start
        self.doppler_scale = doppler_scale
        self.statistic = statistic
        return self

    def _field_units(self):
        return {"bits": "", "crc_ok": ""}


def janus_demodulate(waveform, sample_rate=48000.0, fc=FC_INITIAL, bandwidth=BW_INITIAL,
                     chip_duration=None, hop_sequence=None, start=None,
                     doppler_max_speed=DEFAULT_DOPPLER_MAX_SPEED,
                     sound_speed=DEFAULT_SOUND_SPEED):
    """Demodulate a JANUS waveform -> :class:`JanusReception`, which unpacks
    as ``(bits64, crc_ok)``.

    The inverse of :func:`janus_modulate` (bits in, bits out). The recording is
    first resampled to a canonical rate (integer samples/chip), so any
    ``sample_rate`` works. The preamble is then located by the CMRE Goertzel-bank
    + GO-CFAR detector (unless ``start`` is given), wideband Doppler is removed by
    resampling to the scale estimated from the preamble tones (disable with
    ``doppler_max_speed=0``). Passing ``start=`` skips the detector *and*
    the Doppler compensation — use it only for clean, Doppler-free
    recordings. The 144 data chips are detected non-coherently
    and decoded. Parse the 64 bits with :meth:`JanusPacket.from_bits`, or use
    :func:`janus_receive`. A frame whose data chips run past the end of the
    recording is refused with :class:`ConfigurationError` rather than
    decoded.

    The :class:`JanusReception` carries, beside the bits and ``crc_ok``,
    whether the detector's GO-CFAR threshold was crossed, the preamble
    start, the Doppler scale and the detection statistic. A packet is
    decoded without a crossing too (the detector then takes the statistic's
    argmax): near the decoding threshold that is how every correct decode is
    found, so ``detected`` says which way this one came, and ``crc_ok``
    remains the test of the bits.

    ``sample_rate`` must be at least ``2*(fc + bw/2)``, as on
    :func:`janus_modulate`: a recording sampled below that holds the upper
    tones aliased, and the detector would search it for tones it cannot
    hold.

    Parameters
    ----------
    waveform : array_like
        The real passband recording.
    sample_rate : float, optional
        Sample rate (Hz), at least ``2*(fc + bandwidth/2)``. Default 48 kHz.
    fc : float, optional
        Band centre frequency (Hz). Default 11520, the JANUS initial band.
    bandwidth : float, optional
        Bandwidth (Hz); the tone spacing is ``bandwidth/26``. Default 4160.
    chip_duration : float, optional
        Chip duration (s); ``None`` is one over the tone spacing, 6.25 ms in
        the initial band.
    hop_sequence : array_like of int, optional
        Frequency-hop indices, each in 0..12; ``None`` is the JANUS sequence.
    start : int, optional
        Sample index of the preamble; given, it skips the detector and the
        Doppler compensation.
    doppler_max_speed : float, optional
        Half-range (m/s) of the radial speeds searched. It shortens the
        detector's Goertzel window as well as bounding the Doppler estimate;
        0 disables the Doppler compensation. Default 5.
    sound_speed : float, optional
        Sound speed (m/s) converting the speed range to a Doppler scale.
        Default :data:`~uacpy.core.constants.DEFAULT_SOUND_SPEED`.
    """
    if sample_rate < 2 * (fc + bandwidth / 2):
        raise ConfigurationError(
            f"janus_demodulate: the band edge fc + bw/2 = {fc + bandwidth / 2:g} Hz "
            f"is above the Nyquist frequency sample_rate/2 = "
            f"{sample_rate / 2:g} Hz, so the recording holds those tones "
            f"aliased; require sample_rate >= {2 * (fc + bandwidth / 2):g} Hz.")
    freq_min, fsw = _band_params(fc, bandwidth)
    chip_duration = 1.0 / fsw if chip_duration is None else float(chip_duration)
    fh = (FH_SEQUENCE if hop_sequence is None else
          _require_hop_sequence(hop_sequence, "janus_demodulate"))
    fs = float(sample_rate)
    _, fs_c = _canonical_fs(chip_duration, fs)
    x = _resample(np.asarray(waveform, dtype=float), fs_c / fs)
    fs_in = fs
    given_start = start
    if start is not None:
        start = int(round(start * fs_c / fs))
    fs = fs_c

    detected, statistic, a, found_at = None, None, 0.0, None
    if start is None:
        start, statistic, detected = _detect(x, fs, freq_min, fsw, fh, chip_duration,
                                             doppler_max_speed)
        if start is not None:
            found_at = int(round(start * fs_in / fs_c))
        if start is None:
            raise ConfigurationError(
                f"janus_demodulate: preamble not found — no window of the "
                f"{x.size / fs:.3f} s waveform cleared the detector at "
                f"freq_min={freq_min} Hz. Check the recording carries a JANUS "
                f"packet on that band, or pass start= to skip detection.")
        if doppler_max_speed > 0:
            a = _estimate_doppler(x, start, freq_min, fsw, fh, chip_duration, fs,
                                  doppler_max_speed, sound_speed)
            x = _resample(x, 1.0 + a)
            start = _detect(x, fs, freq_min, fsw, fh, chip_duration, doppler_max_speed)[0]
            if start is None:
                # The preamble cleared the detector before resampling and not
                # after, so the Doppler stage is what lost it -- say which
                # stage failed, or the two raises read as the same failure.
                raise ConfigurationError(
                    f"janus_demodulate: preamble not found after Doppler "
                    f"correction (Doppler scale a = v/c = {a:+.6f}, "
                    f"estimated under "
                    f"doppler_max_speed={doppler_max_speed} m/s), though it "
                    f"was found before. Lower doppler_max_speed, set it to 0 "
                    f"to skip the correction, or pass start= directly.")

    bounds = start + _chip_bounds(_N_TOTAL_CHIPS, chip_duration, fs)
    # A data chip cut off by the end of the recording has nothing to decide;
    # scored as silence it reads as a 0, and a frame of such chips decodes to
    # the all-zero packet, whose CRC-8 is valid. The reference demodulates a
    # chip only once all its samples have arrived
    # (external:demodulator.c:118-119). Half a chip is the tolerance: _detect
    # refines the start over two quarter-chip hops either side, so a frame
    # that ends with the recording can be placed up to half a chip late.
    data = bounds[_PREAMBLE_CHIPS:]
    held = np.clip(x.size - data[:-1], 0, np.diff(data))
    past = int(np.count_nonzero(held < np.diff(data) / 2.0))
    if past:
        raise ConfigurationError(
            f"janus_demodulate: the frame found at {start / fs:.3f} s runs "
            f"past the end of the {x.size / fs:.3f} s recording: {past} of "
            f"its {_N_CODED} data chips hold less than half their samples, "
            f"so it is not decoded. Record the whole packet "
            f"({_N_TOTAL_CHIPS} chips of "
            f"{chip_duration * 1e3:g} ms after the preamble start); on a recording that "
            f"holds no packet this is the detector's best candidate in noise.")
    sym = np.empty(_N_CODED, dtype=int)
    for i in range(_N_CODED):
        seg = x[bounds[_PREAMBLE_CHIPS + i]:bounds[_PREAMBLE_CHIPS + i + 1]]
        f0 = _tone_freq(fh[_PREAMBLE_CHIPS + i], 0, freq_min, fsw)
        f1 = _tone_freq(fh[_PREAMBLE_CHIPS + i], 1, freq_min, fsw)
        sym[i] = 1 if _chip_energy(seg, f1, fs) > _chip_energy(seg, f0, fs) else 0
    bits64 = janus_decode(sym)
    # CRC checked directly rather than via JanusPacket.from_bits: demodulate
    # returns bits + crc_ok only, and parsing the fields here would emit any
    # packet warning a second time when janus_receive parses the same bits.
    crc_ok = bool(np.array_equal(_crc8(bits64[:56]), bits64[56:64]))
    return JanusReception(
        bits64, crc_ok, detected=detected,
        start=given_start if given_start is not None else found_at,
        doppler_scale=a, statistic=statistic)


def janus_transmit(packet: JanusPacket, *, sample_rate=48000.0, **kwargs):
    """Convenience: a :class:`JanusPacket` -> real JANUS waveform.

    Parameters
    ----------
    packet : JanusPacket
        The packet to send.
    sample_rate : float, optional
        Sample rate (Hz). Default 48 kHz.
    **kwargs
        :func:`janus_modulate`'s keywords.
    """
    return janus_modulate(packet.to_bits(), sample_rate=sample_rate, **kwargs)


def janus_receive(waveform, sample_rate=48000.0, **kwargs):
    """Convenience: a JANUS waveform -> ``(JanusPacket, crc_ok)``.

    ``kwargs`` are :func:`janus_demodulate`'s, whose
    :class:`JanusReception` also says how the packet was found.

    Parameters
    ----------
    waveform : array_like
        The real passband recording.
    sample_rate : float, optional
        Sample rate (Hz). Default 48 kHz.
    **kwargs
        :func:`janus_demodulate`'s keywords.
    """
    bits64, crc_ok = janus_demodulate(waveform, sample_rate, **kwargs)
    pkt, _ = JanusPacket.from_bits(bits64)
    return pkt, crc_ok
