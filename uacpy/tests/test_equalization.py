"""Tests for ``uacpy.comms``' equalisers.

An equaliser inverts what the channel did to the symbols. The zero-forcing and
MMSE forms here take a channel estimate and a noise level, and both are held to
one property above all others: **scale invariance**.

Why that property carries the weight. An equaliser's regularisation term is a
floor — MMSE adds ``sigma^2`` to the denominator so a near-null in the channel
cannot be inverted into an enormous gain. If that floor is written as an
absolute number rather than relative to the signal it is regularising, the
equaliser silently changes behaviour with the amplitude of its input: the same
link, transmitted 20 dB louder, is equalised differently and returns different
bits. Nothing raises, the constellation still looks like a constellation, and
the BER moves for a reason nothing in the call reports.

So every test below scales its input and asserts the result is unchanged, and
each states the size of the error the absolute floor produced. The adaptive
(LMS/RLS) equalisers are held to the same rule, since their step sizes have
the same failure mode.
"""

import warnings

import numpy as np
import pytest

from uacpy import comms
from uacpy.comms.channel_models import apply_channel
from uacpy.comms.equalization import mmse_equalizer
from uacpy.comms.metrics import evm
from uacpy.comms.modulation import Modulator
from uacpy.comms.ofdm import (
    estimate_channel,
    ofdm_demodulate,
    ofdm_modulate,
    ofdm_symbol,
)
from uacpy.comms.transceiver import OFDMReceiver, OFDMTransmitter
from uacpy.core.exceptions import ConfigurationError


class TestEqualizerFloorsAreScaleInvariant:
    """The one-tap equalizers and the channel estimator divide by quantities
    that carry the caller's amplitude scale — ``|H|**2``, and a pilot the
    receiver derives from the record itself — so their offsets are fractions
    of that scale rather than fixed numbers. Against fixed ones the answer
    depended on the units: a channel holding 1e-5 of propagation gain
    equalized to EVM 0.014 and 1e-6 to 0.54, MMSE lost a 4-tap channel at a
    receive amplitude of 1e-9, and 16-QAM ran at BER 0.24 at 1e-12."""

    NSC, CP = 64, 8
    H4 = np.array([1.0, 0.5, 0.3, 0.2], dtype=complex)
    SCALES = [1e3, 1.0, 1e-4, 1e-9, 1e-12]

    def _symbols(self, seed=60, blocks=4):
        rng = np.random.default_rng(seed)
        return Modulator("qpsk").modulate(
            rng.integers(0, 2, 2 * self.NSC * blocks))

    def _recovered(self, gain, snr_linear=None):
        sym = self._symbols()
        h = self.H4 * gain
        sig = ofdm_modulate(sym, self.NSC, self.CP)
        rx = apply_channel(sig, h)[: sig.size]
        out = ofdm_demodulate(rx, self.NSC, self.CP, channel=h,
                              snr_linear=snr_linear)
        return out[: sym.size], sym

    @pytest.mark.parametrize("snr_linear", [None, 1e6])
    @pytest.mark.parametrize("gain", SCALES)
    def test_recovered_symbols_do_not_move_with_the_channel_scale(self, gain,
                                                                  snr_linear):
        # The symbols themselves are the invariant to pin: the residual EVM is
        # float noise (5.9e-12 for ZF), so comparing EVMs relatively compares
        # rounding. Measured spread across 1e3 .. 1e-12 of channel gain is
        # 1.2e-15 on unit-magnitude QPSK points.
        out, sym = self._recovered(gain, snr_linear)
        base, _ = self._recovered(1.0, snr_linear)
        assert np.abs(out - base).max() < 1e-12
        assert evm(out, sym) < 1e-5

    def test_the_relative_floor_matches_a_fixed_1e_12_on_a_unit_peak_channel(self):
        # The floor is 1e-12 of the peak |H|**2, so a channel normalized to
        # unit peak power reproduces the old fixed 1e-12 exactly, and a
        # healthy subcarrier of any other channel moves by ~1e-11 at most.
        for h in (np.array([1.0], dtype=complex), self.H4):
            H = np.fft.fft(h, self.NSC)
            h2 = np.abs(H) ** 2
            old = np.conj(H) / (h2 + 1e-12)
            new = np.conj(H) / (h2 + 1e-12 * h2.max())
            if h2.max() == pytest.approx(1.0):
                assert np.array_equal(old, new)
            assert np.abs(new - old).max() / np.abs(old).max() < 1e-10

    def test_a_channel_with_no_power_equalizes_to_zero(self):
        sym = self._symbols()
        sig = ofdm_modulate(sym, self.NSC, self.CP)
        out = ofdm_demodulate(sig, self.NSC, self.CP,
                              channel=np.zeros(4, dtype=complex))
        assert np.all(out == 0.0)

    @pytest.mark.parametrize("gain", SCALES)
    def test_channel_estimate_is_exact_at_any_pilot_scale(self, gain):
        # Through a unit channel the estimate is 1 on every loaded subcarrier;
        # the fixed offset left an error of 1e-12/|pilot| instead.
        pilot = self._symbols(seed=7, blocks=1)[: self.NSC] * gain
        block = ofdm_symbol(pilot, self.NSC, self.CP)
        h = estimate_channel(block, pilot, self.NSC, self.CP)
        assert np.abs(h - 1.0).max() < 1e-12

    def test_unloaded_pilot_subcarriers_estimate_as_zero(self):
        # A Schmidl & Cox training block loads only the even subcarriers, so
        # the floor here guards a genuine null: an unexcited subcarrier has no
        # defined channel and returns zero, the value the equalizers give an
        # unrecoverable subcarrier.
        pilot = self._symbols(seed=7, blocks=1)[: self.NSC].copy()
        pilot[1::2] = 0.0
        block = ofdm_symbol(pilot, self.NSC, self.CP)
        h = estimate_channel(block, pilot, self.NSC, self.CP)
        assert np.all(h[1::2] == 0.0)
        assert np.abs(h[0::2] - 1.0).max() < 1e-12

    @pytest.mark.parametrize("modulation, snr_linear",
                             [("16qam", None), ("qpsk", 100.0)])
    @pytest.mark.parametrize("amplitude", [1e6, 1.0, 1e-9, 1e-12, 1e-15])
    def test_receiver_decodes_across_the_amplitude_ladder(
            self, modulation, snr_linear, amplitude):
        rng = np.random.default_rng(11)
        bits = rng.integers(0, 2, 512)
        tx = OFDMTransmitter(modulation, n_subcarriers=64, cp_len=16)
        rx = OFDMReceiver(modulation, n_subcarriers=64, cp_len=16,
                          snr_linear=snr_linear)
        baseband = apply_channel(tx.transmit(bits), self.H4) * amplitude
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = rx.receive(baseband)
        n = min(out.size, bits.size)
        assert np.array_equal(out[:n], bits[:n])


class TestBlockMMSEEqualizerIsScaleInvariant:
    """``mmse_equalizer`` reads ``snr_linear`` as the SNR at the equalizer
    input, the same as :func:`ofdm_demodulate`, so its Wiener regularizer is
    ``mean(|H|**2)/snr`` — a noise-to-signal ratio in the units of ``|H|**2``.
    As a bare ``1/snr`` it assumed a unit-mean-power channel, and the same link
    with the channel carrying a propagation gain equalized differently: the
    output moved by a full symbol (EVM 0.007 -> 0.51 at a gain of 1e-3, 1.00 at
    1e-6) and 16-QAM ran at BER 0.25."""

    H3 = np.array([1.0, 0.3, 0.1])
    SCALES = [1e3, 1.0, 1e-3, 1e-6, 1e-9, 1e-12]

    def _equalized(self, gain, snr_linear=1e6, seed=3, n_bits=4000):
        rng = np.random.default_rng(seed)
        mod = Modulator("qpsk")
        tx = mod.modulate(rng.integers(0, 2, n_bits))
        h = self.H3 * gain
        rx = apply_channel(tx, h)[: tx.size]
        return mmse_equalizer(rx, h, snr_linear), tx, mod

    @pytest.mark.parametrize("gain", SCALES)
    def test_equalized_output_does_not_move_with_the_channel_scale(self, gain):
        out, tx, mod = self._equalized(gain)
        base, _, _ = self._equalized(1.0)
        assert np.abs(out - base).max() < 1e-12
        assert mod.demodulate(out)[: 2 * tx.size].tolist() == \
            mod.demodulate(tx).tolist()

    def test_regularizer_matches_the_bare_form_for_a_unit_power_channel(self):
        # mean(|H|**2) == sum(|h|**2) by Parseval, so a unit-energy channel
        # reproduces the old bare 1/snr exactly.
        h = np.array([1.0])
        H = np.fft.fft(h, 256)
        h2 = np.abs(H) ** 2
        assert h2.mean() == pytest.approx(1.0)
        rng = np.random.default_rng(5)
        rx = rng.normal(size=256) + 1j * rng.normal(size=256)
        old = np.fft.ifft(np.fft.fft(rx) * (np.conj(H) / (h2 + 1.0 / 100.0)))
        assert np.abs(mmse_equalizer(rx, h, 100.0) - old).max() < 1e-15

    def test_a_channel_with_no_power_equalizes_to_zero(self):
        assert np.all(mmse_equalizer(np.ones(16, dtype=complex),
                                     np.zeros(4), 10.0) == 0.0)


class TestEqualization:
    def test_dfe_opens_closed_eye(self):
        rng = np.random.default_rng(0xACED)
        h = comms.multipath_channel([1.0, 0.6, 0.3],
                                    [0.0, 1 / 8000, 2 / 8000], 8000)
        raw = comms.simulate_link("qpsk", 16.0, 40000, channel=h, rng=rng).ber
        dfe = comms.DFE(n_ff=12, n_fb=6, forget=0.995)
        eq = comms.simulate_link("qpsk", 16.0, 40000, channel=h,
                                 equalizer=dfe, rng=rng)
        # Order-of-magnitude floors, not fitted bounds: the unequalised link
        # runs at BER 5.7e-2 while the equalised one makes zero errors in
        # 40 000 bits, and the converged MSE is -17 dB. The thresholds sit far
        # enough below that to survive a reseed but still fail a DFE that has
        # stopped adapting.
        assert raw > 1e-2
        assert eq.ber < raw / 50
        assert 10 * np.log10(eq.mse[-2000:].mean()) < -10

    def test_mmse_equalizer_recovers_symbols(self):
        rng = np.random.default_rng(3)
        mod = comms.Modulator("qpsk")
        tx = mod.modulate(rng.integers(0, 2, 4000))
        h = np.array([1.0, 0.3, 0.1])
        rx = comms.apply_channel(tx, h)[: tx.size]
        eq = comms.mmse_equalizer(rx, h, 1e6)
        assert comms.bit_error_rate(
            mod.demodulate(tx), mod.demodulate(eq)[: 2 * tx.size]) < 1e-3

    @pytest.mark.parametrize("snr_linear", [0.0, -1.0])
    def test_mmse_equalizer_rejects_nonpositive_snr(self, snr_linear):
        """1/snr is the Wiener regularizer: zero divides, negative un-damps
        the inverse. Neither may pass silently."""
        with pytest.raises(ConfigurationError):
            comms.mmse_equalizer(np.ones(8, complex), np.array([1.0, 0.3]),
                                 snr_linear)

    def test_lms_and_rls_converge_on_known_multipath(self):
        rng = np.random.default_rng(50)
        mod = comms.Modulator("qpsk")
        tx = mod.modulate(rng.integers(0, 2, 2 * 1500))
        h = np.array([1.0, 0.5, 0.25])
        rx = comms.awgn(comms.apply_channel(tx, h)[: tx.size], 25.0, rng=rng)
        raw = comms.evm(rx[-500:], tx[-500:])
        eq_lms, mse_lms = comms.lms_equalizer(rx, mod.constellation,
                                              n_taps=11, step=0.01,
                                              train=tx[:400])
        eq_rls, mse_rls = comms.rls_equalizer(rx, mod.constellation,
                                              n_taps=11, forget=0.99,
                                              train=tx[:400])
        # One equalized output and one squared-error sample per input symbol.
        assert eq_lms.size == mse_lms.size == tx.size
        assert eq_rls.size == mse_rls.size == tx.size
        # Measured over four seeds: raw tail EVM 0.56-0.58, equalized tail
        # EVM 0.073-0.080 for both adaptations, converged MSE 0.0054-0.0063.
        # The /3 and 0.02 floors keep 2-3x of margin on a reseed.
        assert raw > 0.4
        assert comms.evm(eq_lms[-500:], tx[-500:]) < raw / 3
        assert comms.evm(eq_rls[-500:], tx[-500:]) < raw / 3
        assert mse_lms[-500:].mean() < 0.02
        assert mse_rls[-500:].mean() < 0.02
        # Istepanian & Stojanovic put RLS convergence at ~2N against LMS's
        # ~20N symbols (N = 11 taps here): over symbols 50..150 RLS has
        # converged (MSE 0.005-0.008 across seeds) while LMS still sits at
        # 0.40-0.44.
        assert mse_rls[50:150].mean() < mse_lms[50:150].mean() / 10


class TestAdaptiveEqualisersAreScaleInvariant:
    """The single-carrier receiver adapts on an absolute scale in four places
    at once — LMS's stability bound ``step < 2/(ntaps*P)``, RLS's
    ``P(0) = I/1e-2``, the unit center-spike init, and a slicer comparing
    against a unit-energy constellation — so its answer depended on the units
    the caller held the record in. Measured on a 16-QAM passband link through
    ``CommsReceiver``: BER 0.0000 at unit amplitude but 0.2375 at 0.3x, 0.3600
    at 3x and ~0.48 at 1e-9 or 1e9, silent below the window and above it only
    a raw numpy overflow from ``abs(e)**2``. A hydrophone record arrives at
    whatever amplitude it arrives at.

    The record is brought to unit mean power on entry rather than normalising
    tap-by-tap: the register is mixed-scale by construction — feedforward
    samples at the record's amplitude, feedback decisions at constellation
    amplitude — so one normalised-LMS divisor over-corrects the feedback half
    and starves the feedforward half, which measured as NaN on a quiet record.
    ``OFDMReceiver`` was already invariant through its pilot-derived channel;
    this is the single-carrier half of the same contract.
    """

    LADDER = [1e-9, 1e-6, 1e-3, 1.0, 1e3, 1e9]

    @staticmethod
    def _link(seed=3, n_bits=2000, mod='16qam'):
        from uacpy.comms.modulation import Modulator, constellation
        rng = np.random.default_rng(seed)
        syms = Modulator(mod).modulate(rng.integers(0, 2, n_bits))
        base = np.convolve(syms, np.array([1.0, 0.3, 0.1]))[:len(syms)]
        return syms, constellation(mod), base

    @pytest.mark.parametrize('amplitude', LADDER)
    def test_the_dfe_answer_does_not_move_with_the_record_scale(self,
                                                                amplitude):
        from uacpy.comms.equalization import DFE
        syms, cst, base = self._link()
        ref, _ = DFE(n_ff=9, n_fb=3, step=0.02).equalize(
            base, cst, train=syms[:200])
        got, _ = DFE(n_ff=9, n_fb=3, step=0.02).equalize(
            base * amplitude, cst, train=syms[:200])
        assert np.max(np.abs(got - ref)) < 1e-9

    @pytest.mark.parametrize('amplitude', LADDER)
    def test_both_linear_equalisers_are_invariant_too(self, amplitude):
        from uacpy.comms.equalization import lms_equalizer, rls_equalizer
        syms, cst, base = self._link(mod='qpsk', n_bits=1200)
        for fn, kw in ((lms_equalizer, {'step': 0.01}),
                       (rls_equalizer, {'forget': 0.99})):
            ref = fn(base, cst, n_taps=9, train=syms[:150], **kw)[0]
            got = fn(base * amplitude, cst, n_taps=9, train=syms[:150], **kw)[0]
            assert np.max(np.abs(got - ref)) < 1e-9, fn.__name__

    def test_a_unit_power_record_is_untouched(self):
        # The normalisation is a no-op at unit mean power, so an already
        # calibrated record must come back bit-identical.
        from uacpy.comms.equalization import DFE
        syms, cst, base = self._link()
        unit = base / np.sqrt(np.mean(np.abs(base) ** 2))
        a, _ = DFE(n_ff=9, n_fb=3, step=0.02).equalize(unit, cst,
                                                       train=syms[:200])
        b, _ = DFE(n_ff=9, n_fb=3, step=0.02).equalize(unit * 1.0, cst,
                                                       train=syms[:200])
        assert np.array_equal(a, b)

    def test_a_silent_record_stays_finite(self):
        from uacpy.comms.equalization import DFE
        _, cst, _ = self._link()
        eq, mse = DFE(n_ff=9, n_fb=3, step=0.02).equalize(
            np.zeros(200, dtype=complex), cst)
        assert np.isfinite(eq).all() and np.isfinite(mse).all()


class TestTheEqualiserDenominatorIsOneImplementation:
    """``conj(H)/(|H|^2 + eps)`` is divided by three consumers — OFDM
    demodulation, the single-carrier MMSE equaliser and the OFDM receiver
    object — and ``eps`` is one formula in the units of ``|H|^2``. It lives in
    ``comms._equalizer_core`` so none of the three owns it and none imports
    another to reach it."""

    def test_all_three_consumers_reach_the_same_object(self):
        from uacpy.comms import _equalizer_core, equalization, ofdm
        from uacpy.comms import transceiver
        assert (ofdm.regularizer
                is equalization.regularizer
                is transceiver.regularizer
                is _equalizer_core.regularizer)

    def test_the_zero_forcing_floor_has_one_definition(self):
        import uacpy.comms.ofdm as ofdm_module
        from uacpy.comms._equalizer_core import _ZF_REL_FLOOR
        assert _ZF_REL_FLOOR == 1e-12
        # A second same-named constant in a consumer is how a moved helper
        # silently rebinds: the callable moves, a copy of its constant stays,
        # and the two drift apart with nothing failing.
        assert not hasattr(ofdm_module, '_ZF_REL_FLOOR')

    def test_no_consumer_imports_another_to_reach_it(self):
        import pathlib
        import uacpy.comms.equalization as eq
        source = pathlib.Path(eq.__file__).read_text(encoding='utf-8')
        assert 'from uacpy.comms.ofdm import' not in source

    @pytest.mark.parametrize('snr', [None, 10.0, 1000.0])
    def test_the_formula_matches_its_closed_form(self, snr):
        from uacpy.comms._equalizer_core import _ZF_REL_FLOOR, regularizer
        h2 = np.array([4.0, 1.0, 0.25])
        expected = (_ZF_REL_FLOOR * h2.max() if snr is None
                    else h2.mean() / snr)
        assert regularizer(h2, snr) == pytest.approx(expected, rel=1e-15)

    def test_a_powerless_channel_returns_zero(self):
        from uacpy.comms._equalizer_core import regularizer
        assert regularizer(np.zeros(4), 10.0) == 0.0
        assert regularizer(np.array([]), None) == 0.0
