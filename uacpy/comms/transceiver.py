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

import warnings
from typing import Optional

import numpy as np

from uacpy.comms.coding import ConvCode
from uacpy.comms.doppler import compensate_doppler, estimate_doppler_scale
from uacpy.comms.equalization import DFE, slicer
from uacpy.comms.modulation import Modulator
from uacpy.comms.ofdm import (equalize_subcarriers,
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
from uacpy.core._warn_frames import USER_FRAME_SKIP


def _require_passband_fits(sample_rate, fc, sps, rolloff, where):
    """Raise unless the RRC band sits clear of DC and of Nyquist at ``fc``.

    The occupied bandwidth is ``Rs·(1+rolloff)`` for symbol rate
    ``Rs = sample_rate/sps``. Synchronous demodulation needs the whole band
    above DC and its image at ``-2·fc`` clear of it, so ``fc`` has to keep
    half a bandwidth from both edges; ``fc < sample_rate/2`` alone lets a
    sideband fold back onto the signal.
    """
    fs = float(sample_rate)
    bw = fs * (1.0 + float(rolloff)) / float(sps)
    lo, hi = bw / 2.0, fs / 2.0 - bw / 2.0
    if lo >= hi:
        raise ConfigurationError(
            f"{where}: the RRC band (Rs*(1+rolloff) = {bw:g} Hz) does not fit "
            f"below Nyquist ({fs / 2:g} Hz) at any carrier.",
            remediation="Raise sample_rate, raise sps, or lower rolloff.")
    if not lo < float(fc) < hi:
        raise ConfigurationError(
            f"{where}: fc ({float(fc):g} Hz) must lie in ({lo:g}, {hi:g}) Hz so "
            f"the RRC band (Rs*(1+rolloff) = {bw:g} Hz) and its image at -2*fc "
            f"do not fold onto the signal.",
            remediation=f"Use a carrier inside ({lo:g}, {hi:g}) Hz, or change "
                        f"sample_rate/sps/rolloff to narrow the band.")


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

    def __init__(self, modulation: str, code: Optional[ConvCode] = None,
                 preamble=None):
        self.modulation = modulation
        self.modulator = Modulator(modulation)
        self.code = code
        if preamble is None or np.isscalar(preamble):
            n = 64 if preamble is None else int(np.asarray(preamble).item())
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
        _require_passband_fits(sample_rate, fc, sps, rolloff, 'to_passband')
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

    def __init__(self, modulation: str, code: Optional[ConvCode] = None,
                 equalizer: Optional[DFE] = None, preamble=None):
        self.modulation = modulation
        self.modulator = Modulator(modulation)
        self.code = code
        self.equalizer = equalizer
        if preamble is None or np.isscalar(preamble):
            n = 64 if preamble is None else int(np.asarray(preamble).item())
            self.preamble = _default_preamble(n, modulation)
        else:
            self.preamble = np.asarray(preamble, dtype=complex)
        if equalizer is not None and getattr(equalizer, 'forget', None) is None:
            # LMS converges in ~20 N symbols against RLS's ~2 N (Istepanian &
            # Stojanovic); a preamble shorter than that leaves the taps
            # half-trained when the payload starts — measured, 16-QAM at
            # step 0.01 with 64 training symbols decoded at BER 0.22 from a
            # 0.8 rad carrier offset, while 256 symbols or RLS gave 0.
            n_taps = int(getattr(equalizer, 'n_ff', 0)) + int(getattr(equalizer, 'n_fb', 0))
            needed = 20 * n_taps
            if self.preamble.size < needed:
                warnings.warn(
                    f"CommsReceiver: the equalizer adapts by LMS, which needs "
                    f"about 20 x {n_taps} = {needed} training symbols to "
                    f"converge, and the preamble has {self.preamble.size}. "
                    f"Higher-order constellations may decode wrongly after a "
                    f"carrier-phase offset. Pass preamble={needed} or more, or "
                    f"DFE(forget=0.99...) for RLS, which converges in ~2 N.",
                    UserWarning, skip_file_prefixes=USER_FRAME_SKIP)

    def from_passband(self, samples, sample_rate, fc, sps=8, rolloff=0.25, span=8,
                      loop_bw=0.005):
        """Down-convert, matched-filter, and timing-recover to symbol-rate samples."""
        _require_passband_fits(sample_rate, fc, sps, rolloff, 'from_passband')
        bb = downconvert(np.asarray(samples, dtype=float), sample_rate, fc)
        mf = matched_filter(bb, sps, rolloff, span)
        return symbol_sync(mf, sps, loop_bw=loop_bw, start=span * sps)

    def receive(self, symbols, threshold=0.4):
        """Symbols ``[preamble | payload]`` -> information bits.

        Detects the preamble (frame sync), trains the equalizer on it, then
        equalizes/demodulates/decodes the payload. Without an equalizer the
        known preamble still sets the carrier phase and gain — one complex
        least-squares scalar ``<pre, rx_pre> / <pre, pre>`` divides the
        payload — and the payload is assumed to start exactly
        ``len(preamble)`` symbols after the detected start: residual channel
        delay spread leaks preamble ISI into the first payload symbols, and
        nothing here tracks a phase that drifts through the frame (an
        equalizer with ``pll_bandwidth`` does).
        """
        sym = np.asarray(symbols, dtype=complex).ravel()
        start = 0
        k, metric = detect_preamble(sym, self.preamble, threshold=threshold)
        if k is not None:
            start = k
        else:
            warnings.warn(
                f"CommsReceiver.receive: preamble not detected (best metric "
                f"{float(np.max(metric)):.3f} < threshold {float(threshold):.3f}); "
                f"decoding from sample 0. The returned bits are not frame-aligned "
                f"and carry no indication of that.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
        sym = sym[start:]
        pre = self.preamble
        if self.equalizer is not None:
            delay = self.equalizer.n_ff // 2
            ref = np.concatenate([np.zeros(delay, dtype=complex), pre])
            eq, _ = self.equalizer.equalize(sym, self.modulator.constellation, train=ref)
            payload = eq[delay + pre.size:]
        else:
            # A passband record delayed by a fraction of a sample arrives
            # with tens of degrees of carrier phase, and any gain moves the
            # QAM decision rings: measured, QPSK decoded at BER 0.50 and
            # 16-QAM at 0.44 with the preamble detected and no warning. The
            # preamble is known, so its least-squares complex gain is one
            # vdot away.
            rx_pre = sym[:pre.size]
            denom = np.vdot(pre, pre)
            gain = np.vdot(pre, rx_pre) / denom if denom else 1.0
            payload = sym[pre.size:]
            if gain and np.isfinite(gain):
                payload = payload / gain
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
                 cp_len: int = 32, code: Optional[ConvCode] = None):
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
                "to_passband: OFDM band fc +/- sample_rate/(2*oversample) "
                "must lie in (0, sample_rate/2); got fc="
                f"{float(fc):g} Hz, sample_rate={float(sample_rate):g} Hz, "
                f"oversample={os} — band "
                f"{fc - sample_rate / (2 * os):g}-"
                f"{fc + sample_rate / (2 * os):g} Hz against Nyquist "
                f"{sample_rate / 2:g} Hz")
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
                 cp_len: int = 32, code: Optional[ConvCode] = None,
                 snr_linear=None):
        self.modulation = modulation
        self.modulator = Modulator(modulation)
        self.n_subcarriers = int(n_subcarriers)
        self.cp_len = int(cp_len)
        self.code = code
        self.snr_linear = snr_linear
        self.preamble = schmidl_cox_preamble(self.n_subcarriers, self.cp_len)
        self.pilot_freq = _pilot_spectrum(self.n_subcarriers, modulation)

    def _equalize(self, freq, h):
        """One-tap-per-subcarrier ZF (or MMSE) division by the channel estimate.

        Both branches take the ``conj(h)/(|h|^2 + eps)`` form of
        :func:`~uacpy.comms.ofdm.ofdm_demodulate`, with ``eps`` scaled to the
        estimate's own power. ``h`` here comes from the received pilot, so its
        magnitude is the receive amplitude: against a fixed offset the estimate
        stopped being used at all once the record fell near it — measured,
        16-QAM over a 4-tap channel ran at BER 0.24 at a receive amplitude of
        1e-12, and MMSE at 1e-9. A subcarrier the estimate calls silent comes
        back as zero.
        """
        return equalize_subcarriers(freq, h, self.snr_linear)

    def receive(self, baseband):
        """Baseband OFDM frame -> information bits (sync, channel est, equalize).

        Every whole block after the pilot is decoded as data — the
        transmitter's trailing zero guard block and any extra captured
        samples included — so the returned stream runs past the payload (the
        guard alone contributes ``n_subcarriers * bits_per_symbol`` coded
        bits of noise). Slice the result to the known payload length.
        """
        nsc, cp = self.n_subcarriers, self.cp_len
        blk = nsc + cp
        x = np.asarray(baseband, dtype=complex).ravel()
        start, cfo = schmidl_cox_sync(x, nsc)
        if start is None:
            warnings.warn(
                "OFDMReceiver.receive: Schmidl-Cox timing metric never reached "
                "the 0.5 plateau threshold, so no preamble was found; decoding "
                "from sample 0 with cfo=0. The returned bits are not frame-"
                "aligned and carry no indication of that.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP)
            start = 0
        x = apply_cfo(x[start:], cfo)
        nblocks = x.size // blk
        if nblocks < 3:
            raise ConfigurationError(
                "OFDMReceiver: frame too short (need preamble+pilot+data); "
                f"got {nblocks} block(s) of {blk} samples from {x.size} "
                f"samples, need >= 3")

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
