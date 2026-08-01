"""End-to-end Transmitter / CommsReceiver for the single-carrier coherent link.

Wraps the symbol-domain modem (FEC + modulation + preamble) and the passband
physical layer (pulse shaping, up/down-conversion, timing recovery) into two
objects that turn a real payload into real samples and back. The preamble plays
the dual role the underwater-channel literature gives it — a sync probe *and* the
equalizer training sequence (Istepanian & Stojanovic: a frame is "the channel
probe, a pause, and a data block which starts with a training sequence").

Two link architectures are provided as separate, parallel pairs:

* :class:`Transmitter` / :class:`CommsReceiver` — single-carrier coherent (RRC +
  Gardner timing recovery + adaptive DFE/PLL).
* :class:`OFDMTransmitter` / :class:`OFDMReceiver` — multicarrier (cyclic prefix
  + Schmidl-Cox sync + the practical resample-then-residual-CFO Doppler handling
  + one-tap-per-subcarrier equalization from a pilot symbol).

The non-coherent schemes (FSK, DPSK) keep their standalone functional API.

References
----------
Istepanian & Stojanovic. *Underwater Acoustic DSP & Comms* (baseband front end,
    frame structure, raised-cosine shaping, resampling-based Doppler handling).
Schmidl & Cox (1997); Li, Stojanovic et al. (2007). Proakis & Salehi.
"""

from __future__ import annotations

import numpy as np

from uacpy.comms.coding import ConvCode
from uacpy.comms.doppler import compensate_doppler, estimate_doppler_scale
from uacpy.comms.equalization import DFE, slicer
from uacpy.comms.modulation import Modulator
from uacpy.comms.ofdm import (
    ofdm_symbol,
    apply_cfo,
    estimate_channel,
    schmidl_cox_preamble,
    schmidl_cox_sync,
)
from uacpy.comms.phy import (
    downconvert,
    matched_filter,
    pulse_shape,
    symbol_sync,
    upconvert,
)
from uacpy.comms.sync import detect_preamble
from uacpy.core.exceptions import ConfigurationError


def _default_preamble(n_symbols, scheme, seed=0xC0FFEE):
    """A fixed pseudo-random preamble with good autocorrelation."""
    rng = np.random.default_rng(seed)
    mod = Modulator(scheme)
    return mod.modulate(rng.integers(0, 2, n_symbols * mod.bits_per_symbol))


class Transmitter:
    """Map an information payload to symbols, and optionally to real passband.

    Parameters
    ----------
    modulation : str
        Constellation name (see :class:`~uacpy.comms.modulation.Modulator`).
    code : ConvCode, optional
        FEC codec applied before modulation.
    preamble : array_like or int, optional
        Known leading symbols for the receiver's sync + training. An int means
        "generate this many" (matched by :class:`CommsReceiver` with the same count).
    """

    def __init__(self, modulation: str, code: ConvCode = None, preamble=None):
        self.modulation = modulation
        self.modulator = Modulator(modulation)
        self.code = code
        if preamble is None or np.isscalar(preamble):
            n = 64 if preamble is None else int(preamble)
            self.preamble = _default_preamble(n, modulation)
        else:
            self.preamble = np.asarray(preamble, dtype=complex)

    def transmit(self, bits):
        """Information bits -> complex symbols ``[preamble | payload]``."""
        b = self.code.encode(bits) if self.code is not None else np.asarray(bits, int)
        sym = self.modulator.modulate(b)
        return np.concatenate([self.preamble, sym])

    def to_passband(self, symbols, sample_rate, fc, sps=8, rolloff=0.25, span=8):
        """Pulse-shape and up-convert symbols to a real passband signal at ``fc``."""
        if fc >= sample_rate / 2:
            raise ConfigurationError("to_passband: need fc < sample_rate/2")
        return upconvert(pulse_shape(symbols, sps, rolloff, span), sample_rate, fc)

    def transmit_passband(self, bits, sample_rate, fc, sps=8, rolloff=0.25, span=8):
        """Information bits straight to real passband samples (one call)."""
        return self.to_passband(self.transmit(bits), sample_rate, fc, sps, rolloff, span)


class CommsReceiver:
    """Recover information bits from symbols or real passband samples.

    Parameters mirror :class:`Transmitter`; ``equalizer`` is an optional
    :class:`~uacpy.comms.equalization.DFE` (trained on the preamble, with its PLL
    tracking residual carrier offset). ``preamble`` must match the transmitter's.
    """

    def __init__(self, modulation: str, code: ConvCode = None,
                 equalizer: DFE = None, preamble=None):
        self.modulation = modulation
        self.modulator = Modulator(modulation)
        self.code = code
        self.equalizer = equalizer
        if preamble is None or np.isscalar(preamble):
            n = 64 if preamble is None else int(preamble)
            self.preamble = _default_preamble(n, modulation)
        else:
            self.preamble = np.asarray(preamble, dtype=complex)

    def from_passband(self, samples, sample_rate, fc, sps=8, rolloff=0.25, span=8,
                      loop_bw=0.005):
        """Down-convert, matched-filter, and timing-recover to symbol-rate samples."""
        bb = downconvert(np.asarray(samples, dtype=float), sample_rate, fc)
        mf = matched_filter(bb, sps, rolloff, span)
        return symbol_sync(mf, sps, loop_bw=loop_bw, start=span * sps)

    def receive(self, symbols, threshold=0.4):
        """Symbols ``[preamble | payload]`` -> information bits.

        Detects the preamble (frame sync), trains the equalizer on it, then
        equalizes/demodulates/decodes the payload. Without an equalizer the
        payload is assumed to start exactly ``len(preamble)`` symbols after
        the detected start — residual channel delay spread leaks preamble
        ISI into the first payload symbols.
        """
        sym = np.asarray(symbols, dtype=complex).ravel()
        start = 0
        k, _ = detect_preamble(sym, self.preamble, threshold=threshold)
        if k is not None:
            start = k
        sym = sym[start:]
        pre = self.preamble
        if self.equalizer is not None:
            delay = self.equalizer.n_ff // 2
            ref = np.concatenate([np.zeros(delay, dtype=complex), pre])
            eq, _ = self.equalizer.equalize(sym, self.modulator.constellation, train=ref)
            payload = eq[delay + pre.size:]
        else:
            payload = sym[pre.size:]
        bits = self.modulator.demodulate(payload)
        if self.code is not None:
            bits = self.code.decode(bits)
        return bits

    def receive_passband(self, samples, sample_rate, fc, sps=8, rolloff=0.25, span=8,
                         loop_bw=0.005, threshold=0.4):
        """Real passband samples straight to information bits (one call)."""
        syms = self.from_passband(samples, sample_rate, fc, sps, rolloff, span, loop_bw)
        return self.receive(syms, threshold=threshold)


def _pilot_spectrum(n_subcarriers, scheme, seed=0xACE0FDA):
    """Known QAM/PSK pilot loaded on every subcarrier (channel-estimation symbol)."""
    rng = np.random.default_rng(seed)
    mod = Modulator(scheme)
    return mod.modulate(rng.integers(0, 2, n_subcarriers * mod.bits_per_symbol))


class OFDMTransmitter:
    """OFDM passband transmitter (FEC + QAM + Schmidl-Cox preamble + pilot + CP).

    Frame layout: ``[SC preamble | pilot symbol | data symbols...]``, each block a
    cyclic-prefixed OFDM symbol.

    Parameters
    ----------
    modulation : str
        Subcarrier constellation (see :class:`~uacpy.comms.modulation.Modulator`).
    n_subcarriers, cp_len : int
        FFT size and cyclic-prefix length.
    code : ConvCode, optional
        FEC codec applied before mapping.
    """

    def __init__(self, modulation: str, n_subcarriers: int = 256,
                 cp_len: int = 32, code: ConvCode = None):
        self.modulation = modulation
        self.modulator = Modulator(modulation)
        self.n_subcarriers = int(n_subcarriers)
        self.cp_len = int(cp_len)
        self.code = code
        self.preamble = schmidl_cox_preamble(self.n_subcarriers, self.cp_len)
        self.pilot_freq = _pilot_spectrum(self.n_subcarriers, modulation)

    def transmit(self, bits):
        """Information bits -> baseband OFDM frame (complex time samples)."""
        b = self.code.encode(bits) if self.code is not None else np.asarray(bits, int)
        sym = self.modulator.modulate(b)
        nsc = self.n_subcarriers
        if sym.size % nsc:
            sym = np.concatenate([sym, np.zeros(nsc - sym.size % nsc, dtype=complex)])
        data = [ofdm_symbol(sym[i:i + nsc], nsc, self.cp_len)
                for i in range(0, sym.size, nsc)]
        pilot = ofdm_symbol(self.pilot_freq, nsc, self.cp_len)
        guard = np.zeros(nsc + self.cp_len, dtype=complex)   # protects the last block
        return np.concatenate([self.preamble, pilot] + data + [guard])

    def to_passband(self, baseband, sample_rate, fc, oversample=4):
        """Up-convert a baseband OFDM frame to real passband at carrier ``fc``.

        The baseband is interpolated by ``oversample`` so the OFDM band occupies
        ``sample_rate/oversample`` Hz around ``fc`` (leaving room in the passband and an
        image gap the receiver's decimation filter rejects).
        """
        from scipy.signal import resample_poly
        os = int(oversample)
        if fc - sample_rate / (2 * os) <= 0 or fc + sample_rate / (2 * os) >= sample_rate / 2:
            raise ConfigurationError(
                "to_passband: OFDM band fc +/- sample_rate/(2*oversample) must lie in (0, sample_rate/2)")
        up = resample_poly(baseband, os, 1)
        return upconvert(up, sample_rate, fc)

    def transmit_passband(self, bits, sample_rate, fc, oversample=4):
        """Information bits straight to real passband samples (one call)."""
        return self.to_passband(self.transmit(bits), sample_rate, fc, oversample)


class OFDMReceiver:
    """OFDM passband receiver: resample (Doppler) + Schmidl-Cox + residual CFO.

    Implements the practical underwater multicarrier receiver — estimate and
    remove the common Doppler scale by resampling, then correct the residual
    carrier frequency offset, FFT each block, estimate the channel from the pilot
    symbol, and equalize each subcarrier (ZF, or MMSE with ``snr_linear``).

    Parameters mirror :class:`OFDMTransmitter`.
    """

    def __init__(self, modulation: str, n_subcarriers: int = 256,
                 cp_len: int = 32, code: ConvCode = None, snr_linear=None):
        self.modulation = modulation
        self.modulator = Modulator(modulation)
        self.n_subcarriers = int(n_subcarriers)
        self.cp_len = int(cp_len)
        self.code = code
        self.snr_linear = snr_linear
        self.preamble = schmidl_cox_preamble(self.n_subcarriers, self.cp_len)
        self.pilot_freq = _pilot_spectrum(self.n_subcarriers, modulation)

    def _equalize(self, freq, h):
        if self.snr_linear is None:
            return freq / (h + 1e-12)
        return freq * np.conj(h) / (np.abs(h) ** 2 + 1.0 / float(self.snr_linear))

    def receive(self, baseband):
        """Baseband OFDM frame -> information bits (sync, channel est, equalize)."""
        nsc, cp = self.n_subcarriers, self.cp_len
        blk = nsc + cp
        x = np.asarray(baseband, dtype=complex).ravel()
        start, cfo = schmidl_cox_sync(x, nsc, cp)
        if start is None:
            start = 0
        x = apply_cfo(x[start:], cfo)
        nblocks = x.size // blk
        if nblocks < 3:
            raise ConfigurationError("OFDMReceiver: frame too short (need preamble+pilot+data)")

        def block_spectrum(b):
            """FFT of OFDM block ``b`` (cyclic prefix removed)."""
            seg = x[b * blk + cp: b * blk + cp + nsc]
            return np.fft.fft(seg) / np.sqrt(nsc)

        h = estimate_channel(x[blk:2 * blk], self.pilot_freq, nsc, cp)
        c = self.modulator.constellation
        data = []
        for b in range(2, nblocks):
            d = self._equalize(block_spectrum(b), h)
            # decision-directed common-phase-error correction (residual CFO drift)
            dec = slicer(d, c)
            d *= np.exp(-1j * np.angle(np.vdot(dec, d)))
            data.append(d)
        syms = np.concatenate(data) if data else np.array([], dtype=complex)
        bits = self.modulator.demodulate(syms)
        if self.code is not None:
            bits = self.code.decode(bits)
        return bits

    def from_passband(self, samples, sample_rate, fc, oversample=4, doppler_scale=None,
                      scales=None):
        """Resample for Doppler, down-convert, and decimate to a baseband frame.

        Estimates and removes the common Doppler scale by resampling the passband
        (``doppler_scale=None`` estimates it from the known passband preamble),
        then down-converts and decimates by ``oversample`` (the polyphase filter
        rejects the image), leaving residual CFO for :meth:`receive`.
        """
        from scipy.signal import resample_poly
        os = int(oversample)
        pb = np.asarray(samples, dtype=float)
        if doppler_scale is None:
            probe = upconvert(resample_poly(self.preamble, os, 1), sample_rate, fc)
            doppler_scale, _, _ = estimate_doppler_scale(pb, probe, scales)
        if abs(doppler_scale) > 1e-9:
            # doppler_scale is a = v/c; compensate_doppler(pb, a) removes it.
            pb = np.real(compensate_doppler(pb, doppler_scale))
        bb = downconvert(pb, sample_rate, fc)
        return resample_poly(bb, 1, os)          # LPF + decimate removes 2*fc image

    def receive_passband(self, samples, sample_rate, fc, oversample=4, doppler_scale=None,
                         scales=None):
        """Real passband samples straight to information bits (one call)."""
        return self.receive(self.from_passband(samples, sample_rate, fc, oversample,
                                               doppler_scale, scales))
