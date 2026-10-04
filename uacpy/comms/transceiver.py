"""The objects that drive a whole chain: the single-carrier
:class:`Transmitter` / :class:`CommsReceiver` pair and the OFDM
:class:`OFDMTransmitter` / :class:`OFDMReceiver` pair, with the receiver's
diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import warnings

import numpy as np

from uacpy.comms.coding import ConvCode, _encoded
from uacpy.comms.constellations import Modulator, slicer
from uacpy.comms.equalize import DFE, complex_gain
from uacpy.comms.ofdm import (
    _schmidl_cox_backoff, equalize_subcarriers, estimate_channel,
    ofdm_demodulate, ofdm_symbol, remove_cfo, schmidl_cox_metric,
    schmidl_cox_preamble, schmidl_cox_sync,
)
from uacpy.comms.phy import (
    downconvert, pulse_shape, rrc_matched_filter, upconvert,
)
from uacpy.comms.sync import (
    compensate_doppler, detect_preamble, estimate_doppler_scale,
    symbol_sync,
)
from uacpy.core._export import ExportRecord
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core._repr import SettingsRepr
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, NumericsWarning,
)


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


# The seeds are the contract between the two ends of a link: a receiver
# regenerates the preamble / pilot from the same seed and correlates against
# exactly these symbols.
_PREAMBLE_SEED = 0xC0FFEE

#: Symbols in the seeded preamble a Transmitter / CommsReceiver pair
#: shares when ``preamble`` is not given.
DEFAULT_PREAMBLE_SYMBOLS = 64


_PILOT_SEED = 0xACE0FDA


# The pilot is constant-modulus whatever the data constellation: the LS
# estimate on subcarrier k divides by pilot[k], so its noise grows as
# 1/|pilot[k]|^2, and a pilot drawn from 16/64/256-QAM would put
# 2.66/3.98/5.99 dB more noise into the estimate on average.
_PILOT_SCHEME = "qpsk"


def _preamble_symbols(preamble, modulation):
    """The preamble a :class:`Transmitter` / :class:`CommsReceiver` pair
    shares: ``preamble`` as given when it is a symbol array, else that many
    (default 64) seeded symbols of ``modulation``."""
    if preamble is None or np.isscalar(preamble):
        n = (DEFAULT_PREAMBLE_SYMBOLS if preamble is None
             else int(np.asarray(preamble).item()))
        return _seeded_symbols(n, modulation, _PREAMBLE_SEED)
    return np.asarray(preamble, dtype=complex)


def _seeded_symbols(n_symbols, scheme, seed):
    """``n_symbols`` pseudo-random ``scheme`` symbols from a fixed seed: the
    default preamble (good autocorrelation) or the known pilot loaded on every
    OFDM subcarrier for channel estimation."""
    rng = np.random.default_rng(seed)
    mod = Modulator(scheme)
    return mod.modulate(rng.integers(0, 2, n_symbols * mod.bits_per_symbol))


class Transmitter(SettingsRepr):
    """Map an information payload to symbols, and optionally to real passband.

    Parameters
    ----------
    modulation : str
        Constellation name (see
        :class:`~uacpy.comms.constellations.Modulator`).
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
        self.preamble = _preamble_symbols(preamble, modulation)

    def transmit(self, bits):
        """Information bits -> complex symbols ``[preamble | payload]``."""
        b = _encoded(self.code, bits)
        sym = self.modulator.modulate(b)
        return np.concatenate([self.preamble, sym])

    def to_passband(self, symbols, sample_rate, fc, sps=8, rolloff=0.25, span=8):
        """Pulse-shape and up-convert symbols to a real passband signal at ``fc``."""
        _require_passband_fits(sample_rate, fc, sps, rolloff, 'to_passband')
        return upconvert(pulse_shape(symbols, sps, rolloff, span), sample_rate, fc)

    def transmit_passband(self, bits, sample_rate, fc, sps=8, rolloff=0.25, span=8):
        """Information bits straight to real passband samples (one call)."""
        return self.to_passband(self.transmit(bits), sample_rate, fc, sps, rolloff, span)


@dataclass(frozen=True, eq=False)
class ReceiverDiagnostics(ExportRecord):
    """What a receiver computed on the way to its bits, for inspection and
    plotting; returned by the ``receive`` methods of :class:`CommsReceiver`
    and :class:`OFDMReceiver` with ``return_diagnostics=True``.

    A record on the export protocol: ``to_dict`` / ``from_dict``,
    ``to_xarray`` / ``from_xarray`` and ``to_netcdf``; its arrays are
    read-only views (``.copy()`` gives a writeable one).

    Attributes
    ----------
    bits : ndarray
        The decoded information bits (what ``receive`` returns by default).
    symbols : ndarray
        The equalised payload symbols handed to the demodulator. For
        :class:`OFDMReceiver` these are every decoded block's symbols unless
        ``receive(..., n_symbols=)`` limits them to the data symbols the
        transmitter filled (:meth:`OFDMReceiver.payload_symbol_count`).
    mse : ndarray or None
        Squared error per symbol. :class:`CommsReceiver` with an equalizer:
        the :class:`~uacpy.comms.equalize.DFE`'s ``|e|**2`` over the whole
        frame (training, then decision-directed) — its convergence curve;
        ``None`` without an equalizer. :class:`OFDMReceiver`: ``|y - d|**2``
        of each payload symbol ``y`` against its nearest constellation point
        ``d``, after the common-phase correction.
    sync_metric : ndarray
        The frame-synchronisation metric searched:
        :func:`~uacpy.comms.sync.matched_filter_metric` of the symbols
        against the preamble (:class:`CommsReceiver`), or
        :func:`~uacpy.comms.ofdm.schmidl_cox_metric` of the baseband
        record (:class:`OFDMReceiver`).
    start : int or None
        Index of the detected frame start in the record ``sync_metric``
        scores; ``None`` when no preamble was detected (decoding then ran
        from index 0, with a warning).
    """
    bits: np.ndarray
    symbols: np.ndarray
    mse: Optional[np.ndarray]
    sync_metric: np.ndarray
    start: Optional[int]

    _ARRAY_FIELDS = ('bits', 'symbols', 'mse', 'sync_metric')


class CommsReceiver(SettingsRepr):
    """Recover information bits from symbols or real passband samples.

    Parameters mirror :class:`Transmitter`; ``equalizer`` is an optional
    :class:`~uacpy.comms.equalize.DFE` (trained on the preamble, with its PLL
    tracking residual carrier offset). ``preamble`` must match the transmitter's.
    """

    def __init__(self, modulation: str, code: Optional[ConvCode] = None,
                 equalizer: Optional[DFE] = None, preamble=None):
        self.modulation = modulation
        self.modulator = Modulator(modulation)
        self.code = code
        self.equalizer = equalizer
        self.preamble = _preamble_symbols(preamble, modulation)
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
                    NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)

    def from_passband(self, samples, sample_rate, fc, sps=8, rolloff=0.25, span=8,
                      loop_bw=0.005):
        """Down-convert, matched-filter, and timing-recover to symbol-rate samples."""
        _require_passband_fits(sample_rate, fc, sps, rolloff, 'from_passband')
        bb = downconvert(np.asarray(samples, dtype=float), sample_rate, fc)
        mf = rrc_matched_filter(bb, sps, rolloff, span)
        return symbol_sync(mf, sps, loop_bw=loop_bw, start=span * sps)

    def receive(self, symbols, threshold=0.4, *, return_diagnostics=False):
        """Symbols ``[preamble | payload]`` -> information bits.

        With ``return_diagnostics=True`` the return is a
        :class:`ReceiverDiagnostics` carrying the bits together with the
        equalised payload symbols, the equalizer's squared-error curve, the
        preamble-detection metric and the detected start.

        Detects the preamble (frame sync), trains the equalizer on it, then
        equalizes/demodulates/decodes the payload. Without an equalizer the
        known preamble still sets the carrier phase and gain — one complex
        least-squares scalar ``<pre, rx_pre> / <pre, pre>`` divides the
        payload — and the payload is assumed to start exactly
        ``len(preamble)`` symbols after the detected start: residual channel
        delay spread leaks preamble ISI into the first payload symbols, and
        nothing here tracks a phase that drifts through the frame (an
        equalizer with ``pll_gain`` does).
        """
        sym = np.asarray(symbols, dtype=complex).ravel()
        start = 0
        mse = None
        k, metric = detect_preamble(sym, self.preamble, threshold=threshold)
        if k is not None:
            start = k
        else:
            warnings.warn(
                f"CommsReceiver.receive: preamble not detected (best metric "
                f"{float(np.max(metric)):.3f} < threshold {float(threshold):.3f}); "
                f"decoding from sample 0. The returned bits are not frame-aligned "
                f"and carry no indication of that.",
                FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
        sym = sym[start:]
        pre = self.preamble
        if self.equalizer is not None:
            delay = self.equalizer.output_delay
            ref = np.concatenate([np.zeros(delay, dtype=complex), pre])
            # The equalizer's output lags its input by `delay` symbols, so the
            # last `delay` payload symbols come out only once `delay` more
            # inputs have gone in: the zeros appended here are those inputs.
            padded = np.concatenate([sym, np.zeros(delay, dtype=complex)])
            eq, mse = self.equalizer.equalize(padded, self.modulator.constellation,
                                              train=ref)
            payload = eq[delay + pre.size: delay + sym.size]
        else:
            # A passband record delayed by a fraction of a sample arrives
            # with tens of degrees of carrier phase, and any gain moves the
            # QAM decision rings: measured, QPSK decoded at BER 0.50 and
            # 16-QAM at 0.44 with the preamble detected and no warning. The
            # preamble is known, so its least-squares complex gain is one
            # vdot away.
            gain = complex_gain(pre, sym[:pre.size])
            payload = sym[pre.size:]
            if gain and np.isfinite(gain):
                payload = payload / gain
        bits = self.modulator.demodulate(payload)
        if self.code is not None:
            bits = self.code.decode(bits)
        if return_diagnostics:
            return ReceiverDiagnostics(bits=bits, symbols=payload, mse=mse,
                                       sync_metric=metric, start=k)
        return bits

    def receive_passband(self, samples, sample_rate, fc, sps=8, rolloff=0.25, span=8,
                         loop_bw=0.005, threshold=0.4, *,
                         return_diagnostics=False):
        """Real passband samples straight to information bits (one call).

        ``return_diagnostics`` is :meth:`receive`'s; the symbols it reports
        are the timing-recovered symbols :meth:`from_passband` produced.
        """
        syms = self.from_passband(samples, sample_rate, fc, sps, rolloff, span, loop_bw)
        return self.receive(syms, threshold=threshold,
                            return_diagnostics=return_diagnostics)


class OFDMTransmitter(SettingsRepr):
    """OFDM passband transmitter (FEC + QAM + Schmidl-Cox preamble + pilot + CP).

    Frame layout: ``[SC preamble | pilot symbol | data symbols...]``, each block a
    cyclic-prefixed OFDM symbol.

    Parameters
    ----------
    modulation : str
        Subcarrier constellation (see
        :class:`~uacpy.comms.constellations.Modulator`).
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
        self.pilot_values = _seeded_symbols(self.n_subcarriers, _PILOT_SCHEME,
                                          _PILOT_SEED)

    def transmit(self, bits):
        """Information bits -> baseband OFDM frame (complex time samples)."""
        b = _encoded(self.code, bits)
        sym = self.modulator.modulate(b)
        nsc = self.n_subcarriers
        if sym.size % nsc:
            sym = np.concatenate([sym, np.zeros(nsc - sym.size % nsc, dtype=complex)])
        data = [ofdm_symbol(sym[i:i + nsc], nsc, self.cp_len)
                for i in range(0, sym.size, nsc)]
        pilot = ofdm_symbol(self.pilot_values, nsc, self.cp_len)
        guard = np.zeros(nsc + self.cp_len, dtype=complex)   # protects the last block
        return np.concatenate([self.preamble, pilot] + data + [guard])

    def to_passband(self, baseband, sample_rate, fc, oversample=4):
        """Up-convert a baseband OFDM frame to real passband at carrier ``fc``.

        The baseband is interpolated by ``oversample`` so the OFDM band occupies
        ``sample_rate/oversample`` Hz around ``fc`` (leaving room in the passband and an
        image gap the receiver's decimation filter rejects).
        """
        from scipy.signal import resample_poly
        factor = int(oversample)
        if fc - sample_rate / (2 * factor) <= 0 or fc + sample_rate / (2 * factor) >= sample_rate / 2:
            raise ConfigurationError(
                "to_passband: OFDM band fc +/- sample_rate/(2*oversample) "
                "must lie in (0, sample_rate/2); got fc="
                f"{float(fc):g} Hz, sample_rate={float(sample_rate):g} Hz, "
                f"oversample={factor} — band "
                f"{fc - sample_rate / (2 * factor):g}-"
                f"{fc + sample_rate / (2 * factor):g} Hz against Nyquist "
                f"{sample_rate / 2:g} Hz.")
        up = resample_poly(baseband, factor, 1)
        return upconvert(up, sample_rate, fc)

    def transmit_passband(self, bits, sample_rate, fc, oversample=4):
        """Information bits straight to real passband samples (one call)."""
        return self.to_passband(self.transmit(bits), sample_rate, fc, oversample)


class OFDMReceiver(SettingsRepr):
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
        self.pilot_values = _seeded_symbols(self.n_subcarriers, _PILOT_SCHEME,
                                          _PILOT_SEED)

    def payload_symbol_count(self, n_bits):
        """Data symbols :class:`OFDMTransmitter` fills for ``n_bits``
        information bits under this receiver's code and modulation.

        The FEC adds ``K - 1`` flush bits and multiplies by
        ``len(polys)``, the interleaver pads to whole ``depth**2`` blocks (with
        coded zeros, which are real symbols), and the result is mapped
        ``bits_per_symbol`` at a time. The transmitter's zero padding to a
        whole OFDM block and its trailing guard block come after these.
        """
        n = int(n_bits)
        if n < 0:
            raise ConfigurationError(
                f"OFDMReceiver.payload_symbol_count: n_bits must be >= 0; "
                f"got {n_bits!r}.")
        if self.code is not None:
            n = (n + self.code.constraint_length - 1) * len(self.code.polys)
            d = self.code.interleave_depth
            if d:
                n = -(-n // (d * d)) * d * d
        return -(-n // self.modulator.bits_per_symbol)

    def receive(self, baseband, threshold=0.5, *, return_diagnostics=False,
                n_symbols=None):
        """Baseband OFDM frame -> information bits (sync, channel est, equalize).

        ``threshold`` is :func:`~uacpy.comms.schmidl_cox_sync`'s: the timing
        metric the preamble plateau must reach, whose height falls with the
        SNR (see there).

        With ``return_diagnostics=True`` the return is a
        :class:`ReceiverDiagnostics` carrying the bits together with the
        equalised, phase-corrected data symbols, their squared decision
        error, the Schmidl-Cox timing metric of ``baseband`` and the detected
        start. The receiver does not know where the payload ends, so pass
        ``n_symbols`` — :meth:`payload_symbol_count` of the information bit
        count — to limit the diagnostics' ``symbols`` and ``mse`` to the data
        symbols the transmitter filled; left ``None`` they run over every
        decoded block, and the transmitter's zero padding and guard block
        appear as symbols at the origin. ``bits`` are not affected by
        ``n_symbols``.

        Every whole block after the pilot is decoded as data — the
        transmitter's trailing zero guard block and any extra captured
        samples included — so the returned stream runs past the payload (the
        guard alone contributes ``n_subcarriers * bits_per_symbol`` coded
        bits of noise). Slice the result to the known payload length.
        """
        nsc, cp = self.n_subcarriers, self.cp_len
        blk = nsc + cp
        x = np.asarray(baseband, dtype=complex).ravel()
        start, cfo = schmidl_cox_sync(x, nsc, threshold=threshold)
        detected = start
        if start is None:
            warnings.warn(
                f"OFDMReceiver.receive: Schmidl-Cox timing metric never reached "
                f"the {float(threshold):g} plateau threshold, so no preamble was "
                f"found; decoding from sample 0 with cfo=0. The returned bits "
                f"are not frame-aligned and carry no indication of that.",
                FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
            start = 0
        else:
            # The sync returns its start _schmidl_cox_backoff samples ahead
            # of the frame boundary, a multipath margin every FFT window
            # spends from the cyclic prefix; past half the prefix the margin
            # would reach into the previous block (nsc 256 with cp 4, or
            # nsc 1024 with cp 16, lost bits on a noise-free channel), so it
            # is held to half.
            start += max(0, _schmidl_cox_backoff(nsc) - cp // 2)
        x = remove_cfo(x[start:], cfo)
        nblocks = x.size // blk
        if nblocks < 3:
            raise ConfigurationError(
                "OFDMReceiver: frame too short (need preamble+pilot+data); "
                f"got {nblocks} block(s) of {blk} samples from {x.size} "
                f"samples, need >= 3.")

        h = estimate_channel(x[blk:2 * blk], self.pilot_values, nsc, cp)
        c = self.modulator.constellation
        data = []
        # Every whole block after the pilot, CP stripped and transformed;
        # equalised here (channel=None) by the pilot estimate h rather than
        # by a channel the caller describes.
        spectra = ofdm_demodulate(x[2 * blk:], nsc, cp, channel=None)
        for d in spectra.reshape(-1, nsc):
            d = equalize_subcarriers(d, h, self.snr_linear)
            # decision-directed common-phase-error correction (residual CFO drift)
            dec = slicer(d, c)
            d *= np.exp(-1j * np.angle(np.vdot(dec, d)))
            data.append(d)
        syms = np.concatenate(data) if data else np.array([], dtype=complex)
        bits = self.modulator.demodulate(syms)
        if self.code is not None:
            bits = self.code.decode(bits)
        if return_diagnostics:
            if n_symbols is not None:
                n_keep = int(n_symbols)
                if not 0 <= n_keep <= syms.size:
                    raise ConfigurationError(
                        f"OFDMReceiver.receive: n_symbols={n_symbols!r} must "
                        f"lie in 0..{syms.size}, the data symbols decoded "
                        f"from this frame.")
                syms = syms[:n_keep]
            return ReceiverDiagnostics(
                bits=bits, symbols=syms, mse=np.abs(syms - slicer(syms, c)) ** 2,
                sync_metric=schmidl_cox_metric(np.asarray(baseband, dtype=complex)
                                               .ravel(), nsc),
                start=detected)
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
        factor = int(oversample)
        pb = np.asarray(samples, dtype=float)
        if doppler_scale is None:
            probe = upconvert(resample_poly(self.preamble, factor, 1), sample_rate, fc)
            doppler_scale, _, _ = estimate_doppler_scale(pb, probe, scales)
        if abs(doppler_scale) > 1e-9:
            # doppler_scale is a = v/c; compensate_doppler(pb, a) removes it.
            pb = np.real(compensate_doppler(pb, doppler_scale))
        bb = downconvert(pb, sample_rate, fc)
        return resample_poly(bb, 1, factor)          # LPF + decimate removes 2*fc image

    def receive_passband(self, samples, sample_rate, fc, oversample=4, doppler_scale=None,
                         scales=None, threshold=0.5, *,
                         return_diagnostics=False, n_symbols=None):
        """Real passband samples straight to information bits (one call).

        ``threshold``, ``return_diagnostics`` and ``n_symbols`` are
        :meth:`receive`'s; the diagnostics' ``sync_metric`` and ``start``
        refer to the baseband frame :meth:`from_passband` produced.
        """
        return self.receive(self.from_passband(samples, sample_rate, fc, oversample,
                                               doppler_scale, scales),
                            threshold=threshold,
                            return_diagnostics=return_diagnostics,
                            n_symbols=n_symbols)
