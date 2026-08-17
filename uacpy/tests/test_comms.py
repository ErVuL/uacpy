"""Tests for the ``uacpy.comms`` package — modulation, channels, receivers, coding.

Behavioral: BER vs AWGN theory, equalizer opens a closed eye, OFDM/FEC/DSSS
round-trip, sparse channel estimation. Pure-Python; no model binary.
"""

import warnings

import numpy as np
import pytest

from uacpy import comms
from uacpy.comms import janus
from uacpy.core.exceptions import ConfigurationError


class TestModulation:
    @pytest.mark.parametrize("scheme", ["bpsk", "qpsk", "8psk", "16qam", "64qam"])
    def test_bit_symbol_round_trip(self, scheme):
        rng = np.random.default_rng(0)
        mod = comms.Modulator(scheme)
        bits = rng.integers(0, 2, mod.bits_per_symbol * 500)
        assert np.array_equal(bits, mod.demodulate(mod.modulate(bits))[: bits.size])

    def test_constellation_unit_average_energy(self):
        for scheme in ["bpsk", "qpsk", "16qam", "256qam"]:
            c = comms.constellation(scheme)
            assert np.mean(np.abs(c) ** 2) == pytest.approx(1.0)

    def test_unknown_scheme_raises(self):
        with pytest.raises(ConfigurationError):
            comms.constellation("32qam")

    def test_plot_helpers_return_fig_ax(self):
        import matplotlib
        matplotlib.use("Agg")
        from uacpy.visualization import plot_constellation, plot_scatter
        mod = comms.Modulator("qpsk")
        fig, ax = plot_constellation(mod.constellation, scheme=mod.scheme)
        assert ax.has_data()
        fig2, ax2 = plot_scatter(mod.constellation, ideal=mod.constellation)
        assert ax2.has_data()

    @pytest.mark.parametrize("M", [2, 4])
    def test_dpsk_round_trip(self, M):
        rng = np.random.default_rng(1)
        bits = rng.integers(0, 2, int(np.log2(M)) * 400)
        out = comms.dpsk_demodulate(comms.dpsk_modulate(bits, M), M)
        assert np.array_equal(bits, out[: bits.size])

    def test_fsk_round_trip(self):
        rng = np.random.default_rng(2)
        bits = rng.integers(0, 2, 200)
        freqs = [1000, 2000, 3000, 4000]
        wav = comms.fsk_modulate(bits, freqs, 0.01, 48000)
        out = comms.fsk_demodulate(wav, freqs, 0.01, 48000)
        assert np.array_equal(bits, out[: bits.size])


class TestMetrics:
    @pytest.mark.parametrize("scheme", ["bpsk", "qpsk", "16qam"])
    def test_awgn_ber_matches_theory(self, scheme):
        # Monte-Carlo sampling error sets the tolerance. At Eb/N0 = 7 dB over
        # 300 000 bits the expected error COUNT is 232 for BPSK/QPSK
        # (BER 7.7e-4) and 5090 for 16QAM (BER 1.7e-2), so the relative
        # standard error is 1/sqrt(count) = 6.6 % and 1.4 %; rel=0.2 is ~3
        # sigma on the worst case. Shrinking n_bits without widening rel makes
        # this flaky, not stricter.
        rng = np.random.default_rng(0xACED)
        ebn0 = 7.0
        meas = comms.simulate_link(scheme, ebn0, 300000, rng=rng).ber
        theory = float(comms.ber_theory(scheme, ebn0))
        assert meas == pytest.approx(theory, rel=0.2)

    def test_evm_zero_for_identical(self):
        s = comms.constellation("qpsk")
        assert comms.evm(s, s) == pytest.approx(0.0)

    def test_bit_error_rate_half_for_inverted(self):
        b = np.array([0, 1, 0, 1])
        assert comms.bit_error_rate(b, 1 - b) == pytest.approx(1.0)

    @pytest.mark.parametrize("scheme", ["4psk", "32qam", "2qam", "fsk", "psk"])
    def test_ber_theory_unsupported_order_is_typed(self, scheme):
        """'4psk'/'32qam' are the near-misses a user types; the PSK/QAM
        branches must not leak a raw KeyError past the typed guard."""
        with pytest.raises(ConfigurationError):
            comms.ber_theory(scheme, 8.0)

    def test_evm_rejects_empty_input_like_its_siblings(self):
        """bit_error_rate/symbol_error_rate raise on empty input; evm must not
        silently return nan."""
        with pytest.raises(ConfigurationError):
            comms.evm([], [])


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


class TestDoppler:
    def test_estimate_and_compensate(self):
        fs = 48000.0
        t = np.arange(4000) / fs
        # wideband chirp probe — what a real Doppler estimator operates on
        template = np.exp(1j * 2 * np.pi * (6000 * t + 0.5 * 4e6 * t ** 2))
        a_true = 2e-3
        # simulate a closing geometry (v/c = a_true): rx is compressed
        received = comms.compensate_doppler(template, -a_true)
        best, scales, peak = comms.estimate_doppler_scale(
            received, template, np.linspace(0, 4e-3, 41))
        # estimate returns +v/c; compensate_doppler(received, best) recovers template
        # estimate_doppler_scale returns a scan node verbatim (no sub-grid
        # interpolation), so the resolution is the 1e-4 spacing of this
        # 41-point 0..4e-3 scan; a_true sits exactly on a node and abs=2e-4
        # allows a two-node miss.
        assert best == pytest.approx(a_true, abs=2e-4)

    def test_doppler_from_speed_is_the_v_over_c_scale(self):
        """The estimator's output is the same quantity a platform speed maps
        to, so the two are directly comparable."""
        c = 1500.0
        assert comms.doppler_from_speed(3.0, c) == pytest.approx(3.0 / c)
        assert comms.doppler_from_speed(-3.0, c) == pytest.approx(-3.0 / c)


class TestSync:
    def test_preamble_detected_at_offset(self):
        rng = np.random.default_rng(5)
        pre = comms.Modulator("qpsk").modulate(rng.integers(0, 2, 128))
        offset = 200
        rx = np.concatenate([np.zeros(offset, complex), pre,
                             0.1 * (rng.standard_normal(300) + 1j * rng.standard_normal(300))])
        k, metric = comms.detect_preamble(rx, pre, threshold=0.5)
        assert abs(k - offset) <= 1


class TestMultipathChannel:
    def test_shares_the_acoustic_signal_primitive(self):
        """``comms.multipath_channel`` and
        ``acoustic_signal.impulse_response`` are one primitive with one
        argument order — a reversed ``(delays, gains)`` call is silent, since
        both arguments are same-length float arrays."""
        from uacpy.acoustic_signal import impulse_response
        fs = 8000.0
        gains, delays = [1.0, 0.6, 0.3], [0.0, 1 / fs, 2 / fs]
        h = comms.multipath_channel(gains, delays, fs)
        _, h_ref = impulse_response(gains, delays, fs, fractional=False)
        assert np.array_equal(h, h_ref.astype(complex))
        assert h.dtype == complex

    def test_complex_gains_carry_path_phase(self):
        fs = 8000.0
        g = np.array([1.0, 0.6 * np.exp(1j * 1.1)])
        h = comms.multipath_channel(g, [0.0, 3 / fs], fs)
        assert h[0] == g[0] and h[3] == g[1]

    def test_nearest_sample_placement_rounds(self):
        """``fractional=False`` promises *nearest*-sample placement; floor
        placement puts a 0.7-sample arrival one tap early."""
        from uacpy.acoustic_signal import impulse_response
        fs = 8000.0
        _, h = impulse_response([1.0], [0.7 / fs], fs, fractional=False)
        assert np.argmax(np.abs(h)) == 1
        # fractional=True uses a windowed-sinc kernel, not a two-tap linear
        # split: the latter is a frac-dependent lowpass (-3.0 dB at
        # f/fs = 0.25 for frac = 0.5), so equal arrivals came back unequal.
        # Placed clear of the array ends, where the kernel is not truncated.
        _, hf = impulse_response([1.0], [100.7 / fs], fs, fractional=True,
                                 n_samples=256)
        assert hf.sum() == pytest.approx(1.0)
        centroid = float(np.sum(np.arange(hf.size) * hf) / hf.sum())
        assert centroid == pytest.approx(100.7, abs=0.01)

    @pytest.mark.parametrize("gains,delays", [([1.0, 0.5], [0.0]),
                                              ([1.0], [-1e-3])])
    def test_bad_arrivals_raise_configurationerror(self, gains, delays):
        with pytest.raises(ConfigurationError):
            comms.multipath_channel(gains, delays, 8000.0)


class TestFadingChannelConvention:
    def test_gain_sampled_at_input_time(self):
        # Pins the documented convention y[n] = Σ taps[i, n-d_i]·x[n-d_i]:
        # tap gain multiplies the input sample it delays, so taps needs only
        # len(signal) columns.
        x = np.array([1.0, 2.0, 3.0], dtype=complex)
        taps = np.array([[1.0, 1.0, 1.0],
                         [10.0, 20.0, 30.0]], dtype=complex)
        y = comms.apply_fading_channel(x, taps, [0, 2])
        # direct path: x itself; delayed path: taps[1, m]*x[m] shifted by 2
        expected = np.array([1.0, 2.0, 3.0 + 10.0, 40.0, 90.0], dtype=complex)
        assert np.allclose(y, expected)


class TestFadingStatistics:
    """``fading_taps`` must stay Rayleigh however few DFT bins the Doppler
    band spans. Normalising each realisation by its own RMS pinned |H| to a
    constant whenever the band collapsed to the DC bin, turning the channel
    into a unit-modulus phase — a *perfect* channel returned where the
    Rayleigh one was asked for, and the AWGN BER curve reported in its place."""

    @staticmethod
    def _stats(doppler_hz, seed, n=4096, fs=8000.0, n_taps=2000):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            H = comms.fading_taps(n_taps, n, doppler_hz, fs,
                                  rng=np.random.default_rng(seed))
        return float(np.mean(np.abs(H) ** 2)), float(np.abs(H).std())

    @pytest.mark.parametrize('doppler_hz', [0.5, 1.0, 1.9, 2.5, 10.0, 50.0])
    @pytest.mark.parametrize('seed', [3, 7])
    def test_envelope_is_rayleigh_at_every_doppler(self, doppler_hz, seed):
        # fs/n = 1.953 Hz, so the first three cases retain the DC bin alone —
        # exactly where the per-realisation normalisation used to give
        # std|H| == 0.0000 (constant modulus). Rayleigh with unit mean power
        # has std|H| = sqrt(1 - pi/4) = 0.4633.
        #
        # The bound is sized from the scatter across SEEDS, not across
        # dopplers: below fs/n each tap is a single complex draw, so the
        # estimator has one degree of freedom and ~5x the scatter of the
        # many-bin case. Measured over 12 seeds, worst deviation is 0.021 in
        # std and 0.037 in power at 0.5 Hz, against 0.0013 / 0.005 at 50 Hz.
        # Sizing on the 50 Hz end (as a doppler-only sweep would) gives a
        # bound the 0.5 Hz rows exceed on most seeds. These admit the measured
        # spread with ~40 % margin and still exclude the pathology (std = 0)
        # by 15x.
        power, std = self._stats(doppler_hz, seed)
        assert power == pytest.approx(1.0, abs=0.06)
        assert std == pytest.approx(np.sqrt(1.0 - np.pi / 4.0), abs=0.03)

    def test_unresolvable_doppler_warns(self):
        # 0.5 Hz over a 4096-sample block at 8 kHz spans one bin: the process
        # has too few degrees of freedom to realise the requested spread.
        with pytest.warns(UserWarning, match='DFT bin'):
            comms.fading_taps(4, 4096, 0.5, 8000.0, rng=np.random.default_rng(1))

    def test_well_resolved_doppler_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            comms.fading_taps(4, 4096, 50.0, 8000.0, rng=np.random.default_rng(1))


class TestChannelEstimation:
    def test_ls_and_omp_recover_sparse_channel(self):
        rng = np.random.default_rng(6)
        h = np.zeros(20, complex)
        h[[0, 5, 11]] = [1.0, 0.5j, -0.3]
        pilots = comms.Modulator("qpsk").modulate(rng.integers(0, 2, 2000))
        rx = comms.apply_channel(pilots, h)[: pilots.size]
        # No noise is added, so 1e-6 is a float-conditioning bound on the
        # least-squares solve, not a statistical one. A regression that biases
        # either estimator shows up orders of magnitude above it.
        assert np.linalg.norm(comms.ls_estimate(rx, pilots, 20) - h) < 1e-6
        h_omp = comms.omp_estimate(rx, pilots, 20, sparsity=3)
        assert np.linalg.norm(h_omp - h) < 1e-6
        assert set(np.flatnonzero(np.abs(h_omp) > 1e-6)) == {0, 5, 11}


class TestOFDM:
    def test_round_trip_over_multipath(self):
        rng = np.random.default_rng(7)
        mod = comms.Modulator("qpsk")
        bits = rng.integers(0, 2, 4096)
        sig = comms.ofdm_modulate(mod.modulate(bits), 256, 32)
        h = np.zeros(8, complex)
        h[[0, 3, 6]] = [1.0, 0.4, 0.2]
        rx = comms.awgn(comms.apply_channel(sig, h), 25.0, rng=rng)
        out = comms.ofdm_demodulate(rx, 256, 32, channel=h)
        assert comms.bit_error_rate(bits, mod.demodulate(out)[: bits.size]) < 1e-3

    def test_zero_forcing_survives_a_spectral_null(self):
        # A channel whose response is exactly zero on one subcarrier must not
        # turn that subcarrier into inf/NaN and poison the whole symbol.
        nsc, cp = 64, 16
        rng = np.random.default_rng(11)
        sym = np.exp(1j * np.pi / 2 * rng.integers(0, 4, nsc))
        tx = comms.ofdm_modulate(sym, nsc, cp)
        h = np.array([1.0, -1.0])                      # H[0] = 1 - 1 = 0 exactly
        out = comms.ofdm_demodulate(comms.apply_channel(tx, h)[: tx.size],
                                    nsc, cp, channel=h)
        assert np.all(np.isfinite(out))
        assert abs(out[0]) < 1e-6                      # null carries nothing
        assert np.allclose(out[1:], sym[1:], atol=1e-9)   # the rest is untouched


class TestCoding:
    def test_viterbi_corrects_random_errors(self):
        rng = np.random.default_rng(8)
        info = rng.integers(0, 2, 1000)
        coded = comms.interleave(comms.conv_encode(info), 16)
        flip = rng.choice(coded.size, int(0.03 * coded.size), replace=False)
        coded[flip] ^= 1
        dec = comms.viterbi_decode(comms.deinterleave(coded, 16))
        assert np.array_equal(dec[: info.size], info)

    def test_interleave_round_trip(self):
        rng = np.random.default_rng(9)
        b = rng.integers(0, 2, 320)
        assert np.array_equal(b, comms.deinterleave(comms.interleave(b, 16), 16)[: b.size])

    def test_convcode_codec_round_trip(self):
        rng = np.random.default_rng(11)
        code = comms.ConvCode(interleave_depth=16)
        assert code.rate == pytest.approx(0.5)
        info = rng.integers(0, 2, 800)
        coded = code.encode(info)
        flip = rng.choice(coded.size, int(0.03 * coded.size), replace=False)
        coded[flip] ^= 1
        assert np.array_equal(code.decode(coded)[: info.size], info)

    def test_coded_link_beats_uncoded(self):
        rng = np.random.default_rng(12)
        code = comms.ConvCode(interleave_depth=16)
        raw = comms.simulate_link("bpsk", 4.0, 40000, rng=rng).ber
        cod = comms.simulate_link("bpsk", 4.0, 40000, code=code, rng=rng).ber
        assert cod < raw


class TestFraming:
    def test_byte_bit_round_trip(self):
        data = b"hello underwater \x00\xff\x10"
        assert comms.bits_to_bytes(comms.bytes_to_bits(data)) == data

    def test_frame_round_trip_and_crc(self):
        payload = b"the quick brown fox"
        bits = comms.pack_frame(payload)
        out, ok = comms.unpack_frame(bits)
        assert ok and out == payload

    def test_crc_catches_corruption(self):
        bits = comms.pack_frame(b"important data")
        bits[40] ^= 1
        _, ok = comms.unpack_frame(bits)
        assert not ok


class TestPHY:
    def test_rrc_pulse_then_matched_filter_recovers_symbols(self):
        rng = np.random.default_rng(20)
        sps = 8
        sym = comms.Modulator("qpsk").modulate(rng.integers(0, 2, 2000))
        bb = comms.pulse_shape(sym, sps)
        mf = comms.rrc_matched_filter(bb, sps)
        gd = 8 * sps                                  # two RRC of span=8
        peaks = mf[gd:gd + sym.size * sps:sps]
        # The link is noiseless: the 5 % residual is the ISI left by truncating
        # the RRC pair to span 8, which a longer span would shrink. It is not a
        # sampling-error bar, so it does not move with the seed.
        assert np.linalg.norm(peaks - sym) / np.linalg.norm(sym) < 0.05

    def test_upconvert_real_and_downconvert_inverts(self):
        rng = np.random.default_rng(21)
        sps, fs, fc = 8, 96000.0, 24000.0
        sym = comms.Modulator("qpsk").modulate(rng.integers(0, 2, 1000))
        bb = comms.pulse_shape(sym, sps)
        pb = comms.upconvert(bb, fs, fc)
        assert np.isrealobj(pb)
        mf = comms.rrc_matched_filter(comms.downconvert(pb, fs, fc), sps)
        gd = 8 * sps
        peaks = mf[gd:gd + sym.size * sps:sps]
        assert comms.bit_error_rate(
            comms.Modulator("qpsk").demodulate(sym),
            comms.Modulator("qpsk").demodulate(peaks)) < 1e-3

    def test_symbol_sync_recovers_under_timing_offset(self):
        rng = np.random.default_rng(22)
        sps = 8
        mod = comms.Modulator("qpsk")
        sym = mod.modulate(rng.integers(0, 2, 3000))
        mf = comms.rrc_matched_filter(comms.pulse_shape(sym, sps), sps)
        # fractional-delay the (complex) matched-filter output, then recover timing
        f = np.fft.fftfreq(mf.size)
        delayed = np.fft.ifft(np.fft.fft(mf) * np.exp(-2j * np.pi * f * 2.7))
        out = comms.symbol_sync(delayed, sps, start=8 * sps)
        ph = np.angle(np.vdot(sym[:1000], out[:1000]))
        ber = comms.bit_error_rate(mod.demodulate(sym[:1000]),
                                   mod.demodulate(out[:1000] * np.exp(-1j * ph)))
        assert ber < 1e-2


class TestDFEPLL:
    def test_pll_harmless_at_zero_offset(self):
        rng = np.random.default_rng(23)
        mod = comms.Modulator("qpsk")
        pre = mod.modulate(rng.integers(0, 2, 200))
        bits = rng.integers(0, 2, 6000); payload = mod.modulate(bits)
        sym = np.concatenate([pre, payload])
        rx = sym + np.sqrt(0.01) * (rng.standard_normal(sym.size)
                                    + 1j * rng.standard_normal(sym.size))
        dfe = comms.DFE(n_ff=8, n_fb=2, forget=0.998, pll_bandwidth=0.05)
        delay = dfe.n_ff // 2
        ref = np.concatenate([np.zeros(delay, complex), pre])
        eq, _ = dfe.equalize(rx, mod.constellation, train=ref)
        rxp = eq[delay + pre.size: delay + pre.size + payload.size]
        assert comms.bit_error_rate(bits, mod.demodulate(rxp)[:bits.size]) < 1e-3

    def test_pll_tracks_carrier_offset(self):
        rng = np.random.default_rng(24)
        mod = comms.Modulator("qpsk")
        pre = mod.modulate(rng.integers(0, 2, 200))
        bits = rng.integers(0, 2, 6000); payload = mod.modulate(bits)
        sym = np.concatenate([pre, payload])
        n = np.arange(sym.size)
        rx = sym * np.exp(2j * np.pi * 0.003 * n)       # 0.003 cycle/symbol offset
        rx = rx + np.sqrt(0.01) * (rng.standard_normal(rx.size)
                                   + 1j * rng.standard_normal(rx.size))
        without = comms.DFE(n_ff=8, n_fb=2, forget=0.998, pll_bandwidth=0.0)
        withpll = comms.DFE(n_ff=8, n_fb=2, forget=0.998, pll_bandwidth=0.06)
        delay = 4
        ref = np.concatenate([np.zeros(delay, complex), pre])
        eq0, _ = without.equalize(rx, mod.constellation, train=ref)
        eq1, _ = withpll.equalize(rx, mod.constellation, train=ref)
        b0 = comms.bit_error_rate(bits, mod.demodulate(
            eq0[delay + pre.size: delay + pre.size + payload.size])[:bits.size])
        b1 = comms.bit_error_rate(bits, mod.demodulate(
            eq1[delay + pre.size: delay + pre.size + payload.size])[:bits.size])
        assert b0 > 0.1 and b1 < 1e-3            # PLL essential to track the offset


class TestTransceiver:
    def test_symbol_domain_round_trip(self):
        rng = np.random.default_rng(25)
        bits = rng.integers(0, 2, 2000)
        tx = comms.Transmitter("qpsk", code=comms.ConvCode(interleave_depth=16))
        rx = comms.CommsReceiver("qpsk", code=comms.ConvCode(interleave_depth=16))
        out = rx.receive(tx.transmit(bits))
        assert np.array_equal(out[: bits.size], bits)

    def test_real_payload_through_passband_channel(self):
        rng = np.random.default_rng(26)
        from scipy.signal import resample_poly
        fs, fc, sps = 96000.0, 24000.0, 8
        message = b"uacpy comms passband round-trip test 0123456789"
        code = comms.ConvCode(interleave_depth=16)
        tx = comms.Transmitter("qpsk", code=code, preamble=256)
        wav = tx.transmit_passband(comms.pack_frame(message), fs, fc, sps=sps)
        assert np.isrealobj(wav)
        rxsig = np.convolve(wav, np.array([1.0, 0, 0.4, 0, 0.2]))   # multipath
        rxsig = resample_poly(rxsig, 100020, 100000)               # clock skew
        rxsig = np.concatenate([np.zeros(11), rxsig])
        rxsig = rxsig + np.sqrt(np.mean(rxsig ** 2) / 10 ** (22 / 10)) \
            * rng.standard_normal(rxsig.size)
        dfe = comms.DFE(n_ff=16, n_fb=6, forget=0.997, pll_bandwidth=0.04)
        rx = comms.CommsReceiver("qpsk", code=code, equalizer=dfe, preamble=256)
        payload, ok = comms.unpack_frame(rx.receive_passband(rxsig, fs, fc, sps=sps))
        assert ok and payload == message


class TestPassbandCarrierPlacement:
    """``fc < sample_rate/2`` is not the physical constraint. The RRC band is
    ``Rs*(1+rolloff)`` wide and its image sits at ``-2*fc``, so a carrier too
    near either edge folds a sideband back onto the signal — silently, on a
    noiseless loopback. Both ends of the link enforce it (Rule 6: the guard
    belongs on the funnel, not at each call site)."""

    FS, SPS, ROLLOFF = 48000.0, 8, 0.25       # Rs = 6 kHz, band = 7.5 kHz
    BW = FS * (1.0 + ROLLOFF) / SPS
    LO, HI = BW / 2.0, FS / 2.0 - BW / 2.0

    def test_the_admissible_band_is_the_formula_not_four_numbers(self):
        # Ties the cases below to Rs*(1+rolloff) rather than to hand-picked
        # carriers, so changing sps/rolloff moves the test with the physics.
        assert (self.LO, self.HI) == (3750.0, 20250.0)

    @pytest.mark.parametrize('fc', [960.0, 3000.0, 22006.0, 23760.0])
    def test_folding_carrier_is_refused_by_both_ends(self, fc):
        # 3000 Hz is inside the guard's brickwall bound but was measured to
        # round-trip at BER 0.0000, because an RRC skirt is not a brickwall.
        # The guard is deliberately conservative; this pins that choice, so
        # relaxing it later is a contract change, not a bug fix.
        assert not self.LO < fc < self.HI
        tx = comms.Transmitter("qpsk", preamble=64)
        rx = comms.CommsReceiver("qpsk", preamble=64)
        bits = np.random.default_rng(3).integers(0, 2, 400)
        with pytest.raises(ConfigurationError, match='must lie in'):
            tx.transmit_passband(bits, self.FS, fc, sps=self.SPS,
                                 rolloff=self.ROLLOFF)
        with pytest.raises(ConfigurationError, match='must lie in'):
            rx.from_passband(np.zeros(4096), self.FS, fc, sps=self.SPS,
                             rolloff=self.ROLLOFF)

    @pytest.mark.parametrize('fc', [6000.0, 12000.0, 18000.0])
    def test_interior_carrier_round_trips_exactly(self, fc):
        # The discriminating half: an in-band carrier must still reach BER 0
        # on a noiseless loopback, so the guard cannot be satisfied by simply
        # refusing everything.
        tx = comms.Transmitter("qpsk", preamble=64)
        rx = comms.CommsReceiver("qpsk", preamble=64)
        bits = np.random.default_rng(3).integers(0, 2, 4000)
        pb = tx.transmit_passband(bits, self.FS, fc, sps=self.SPS,
                                  rolloff=self.ROLLOFF)
        out = rx.receive_passband(pb, self.FS, fc, sps=self.SPS,
                                  rolloff=self.ROLLOFF)
        n = min(out.size, bits.size)
        assert np.mean(out[:n] != bits[:n]) == 0.0


class TestUnsyncedReceiveIsAnnounced:
    """Both receivers compute an explicit "no frame here" verdict and then
    fall back to sample 0. The fallback is legitimate — a detector finding
    nothing is a real outcome — but returning a full, same-dtype bit array
    with no signal at all made a 43 %-BER decode indistinguishable from a
    clean one. The verdict has to reach the caller."""

    def test_undetected_preamble_warns_and_clean_decode_does_not(self):
        tx = comms.Transmitter("qpsk", preamble=64)
        rx = comms.CommsReceiver("qpsk", preamble=64)
        bits = np.random.default_rng(4).integers(0, 2, 1200)
        sym = tx.transmit(bits)

        with warnings.catch_warnings():       # acquired: must stay silent
            warnings.simplefilter('error')
            clean = rx.receive(comms.awgn(sym, 20.0,
                                          rng=np.random.default_rng(101)))
        assert np.array_equal(clean[:bits.size], bits)

        buried = comms.awgn(sym, -15.0, rng=np.random.default_rng(101))
        assert comms.detect_preamble(buried, tx.preamble, threshold=0.4)[0] is None
        with pytest.warns(UserWarning, match='preamble not detected'):
            rx.receive(buried)

    def test_unsynced_ofdm_frame_warns(self):
        rx = comms.OFDMReceiver("qpsk", n_subcarriers=64, cp_len=8)
        rng = np.random.default_rng(9)
        noise = (rng.standard_normal(64 * 12) + 1j * rng.standard_normal(64 * 12))
        assert comms.schmidl_cox_sync(noise, 64, 8)[0] is None
        with pytest.warns(UserWarning, match='no preamble was found'):
            rx.receive(noise)


class TestOFDMModem:
    def test_schmidl_cox_preamble_has_two_identical_halves(self):
        nsc, cp = 256, 32
        pre = comms.schmidl_cox_preamble(nsc, cp)
        assert np.allclose(pre[cp:cp + nsc // 2], pre[cp + nsc // 2:cp + nsc], atol=1e-9)

    def test_schmidl_cox_estimates_timing_and_cfo(self):
        rng = np.random.default_rng(30)
        nsc, cp = 256, 32
        pre = comms.schmidl_cox_preamble(nsc, cp)
        tail = rng.standard_normal(3 * (nsc + cp)) + 1j * rng.standard_normal(3 * (nsc + cp))
        frame = np.concatenate([pre, 0.5 * tail])
        off, cfo_true = 137, 0.0007
        rx = comms.apply_cfo(np.concatenate([np.zeros(off, complex), frame]), -cfo_true)
        rx = rx + 0.02 * (rng.standard_normal(rx.size) + 1j * rng.standard_normal(rx.size))
        start, cfo = comms.schmidl_cox_sync(rx, nsc, cp)
        assert abs(start - off) <= cp           # within the cyclic prefix
        assert abs(cfo - cfo_true) < 1e-4

    def test_ofdm_baseband_round_trip_multipath(self):
        rng = np.random.default_rng(31)
        nsc, cp = 256, 32
        code = comms.ConvCode(interleave_depth=16)
        bits = rng.integers(0, 2, 4000)
        tx = comms.OFDMTransmitter("qpsk", nsc, cp, code=code)
        rx = comms.OFDMReceiver("qpsk", nsc, cp, code=code)
        bb = tx.transmit(bits)
        y = np.convolve(bb, np.array([1.0, 0, 0, 0.4, 0, 0, 0.25j]))
        y = comms.apply_cfo(y, -6e-4)
        y = np.concatenate([np.zeros(50, complex), y])
        y = y + 0.02 * (rng.standard_normal(y.size) + 1j * rng.standard_normal(y.size))
        assert np.array_equal(rx.receive(y)[: bits.size], bits)

    def test_ofdm_real_payload_through_passband_with_doppler(self):
        from scipy.signal import resample_poly
        rng = np.random.default_rng(32)
        nsc, cp = 256, 32
        fs, fc, os = 96000.0, 24000.0, 4
        code = comms.ConvCode(interleave_depth=16)
        message = b"OFDM passband round-trip 0123456789 abcdefghij"
        tx = comms.OFDMTransmitter("qpsk", nsc, cp, code=code)
        rx = comms.OFDMReceiver("qpsk", nsc, cp, code=code)
        pb = tx.transmit_passband(comms.pack_frame(message), fs, fc, oversample=os)
        assert np.isrealobj(pb)
        sig = np.convolve(pb, [1, 0, 0, 0.3, 0, 0, 0.2])
        sig = resample_poly(sig, 100015, 100000)          # 150 ppm Doppler
        sig = np.concatenate([np.zeros(40), sig])
        sig = sig + np.sqrt(np.mean(sig ** 2) / 10 ** (25 / 10)) * rng.standard_normal(sig.size)
        out = rx.receive_passband(sig, fs, fc, oversample=os,
                                  scales=np.linspace(-3e-4, 3e-4, 31))
        payload, ok = comms.unpack_frame(out)
        assert ok and payload == message


class TestJanus:
    def _packet(self):
        adb = np.zeros(34, dtype=int)
        adb[:8] = [1, 0, 1, 0, 1, 1, 0, 0]
        return comms.JanusPacket(class_id=16, app_type=0, app_data=adb, mobility=1)

    def test_packet_bits_layout_and_crc(self):
        pkt = self._packet()
        bits = pkt.to_bits()
        assert bits.size == 64
        assert list(bits[:4]) == [0, 0, 1, 1]            # version 3
        out, ok = comms.JanusPacket.from_bits(bits)
        assert ok
        assert out.class_id == 16 and out.app_type == 0 and out.mobility == 1
        assert np.array_equal(out.app_data, pkt.app_data)

    def test_crc_detects_corruption(self):
        bits = self._packet().to_bits()
        bits[30] ^= 1
        _, ok = comms.JanusPacket.from_bits(bits)
        assert not ok

    def test_encode_matches_cmre_reference_vector(self):
        # Golden vector captured from the CMRE janus-c reference (janus-tx) for
        # this exact packet -> locks the encoder convention to the standard.
        pkt = comms.JanusPacket(class_id=16, app_type=0,
                                app_data=np.zeros(34, dtype=int), mobility=1, tx_rx=1)
        bits = pkt.to_bits()
        assert int("".join(map(str, bits[56:64])), 2) == 0xBF      # reference CRC
        ref_hex = 0x101e0200e8200580900581f0280600c00303
        ref = np.array([(ref_hex >> (143 - i)) & 1 for i in range(144)])
        assert np.array_equal(comms.janus_encode(bits), ref)

    def test_fec_is_144_symbols_and_corrects_errors(self):
        bits = self._packet().to_bits()
        sym = comms.janus_encode(bits)
        assert sym.size == 144
        assert np.array_equal(comms.janus_decode(sym), bits)
        noisy = sym.copy()
        noisy[np.random.default_rng(0).choice(144, 8, replace=False)] ^= 1
        assert np.array_equal(comms.janus_decode(noisy), bits)

    def test_fh_bfsk_waveform_round_trip_clean(self):
        bits = self._packet().to_bits()
        fs = 48000.0
        wav = comms.janus_modulate(bits, fs)
        assert np.isrealobj(wav)
        start, metric = comms.janus_detect(wav, fs)
        assert start is not None and metric.size > 0
        bits_out, ok = comms.janus_demodulate(wav, fs)
        assert ok and np.array_equal(bits_out, bits)

    def test_fh_bfsk_through_noisy_delayed_channel(self):
        rng = np.random.default_rng(1)
        bits = self._packet().to_bits()
        fs = 48000.0
        wav = comms.janus_modulate(bits, fs)
        rx = np.concatenate([np.zeros(523), wav])
        rx = rx + np.sqrt(np.mean(wav ** 2) / 10 ** (10 / 10)) * rng.standard_normal(rx.size)
        bits_out, ok = comms.janus_demodulate(rx, fs)
        pkt, _ = comms.JanusPacket.from_bits(bits_out)
        assert ok and pkt.class_id == 16

    def test_decode_at_non_integer_samples_per_chip(self):
        # 44.1 kHz -> 6.25 ms chip = 275.625 samples; sub-sample chip tracking
        # must still align and decode.
        bits = self._packet().to_bits()
        fs = 44100.0
        wav = comms.janus_modulate(bits, fs)
        out, ok = comms.janus_demodulate(wav, fs)
        assert ok and np.array_equal(out, bits)

    def test_decode_with_doppler_resampling(self):
        bits = self._packet().to_bits()
        fs = 48000.0
        wav = comms.janus_modulate(bits, fs)
        c = 1500.0
        for v in (-4.0, 4.0):
            scale = 1.0 + v / c          # time-scale a moving-platform Doppler
            n = int(round(wav.size / scale))
            shifted = np.interp(np.linspace(0, wav.size - 1, n),
                                np.arange(wav.size), wav)
            out, ok = comms.janus_demodulate(shifted, fs)
            assert ok and np.array_equal(out, bits)

    def test_detect_locates_buried_packet(self):
        # GO-CFAR detector must find a packet buried in silence + noise and
        # report its sample offset in the original recording.
        bits = self._packet().to_bits()
        fs = 48000.0
        wav = comms.janus_modulate(bits, fs)
        lead = 4321
        rng = np.random.default_rng(2)
        rx = np.concatenate([np.zeros(lead), wav, np.zeros(2000)])
        rx = rx + 0.05 * np.max(np.abs(wav)) * rng.standard_normal(rx.size)
        start, stat = comms.janus_detect(rx, fs)
        assert start is not None and stat.size > 0
        # janus_detect reports a column of the quarter-chip alignment grid
        # mapped back to samples, so its resolution is a fraction of the
        # 6.25 ms chip; 10 ms is under two chips and pins the packet to the
        # right hop, which is what the demodulator needs.
        assert abs(start - lead) <= int(0.01 * fs)

    @pytest.mark.parametrize("m, expected", [(104, 0.1538 * 104),      # initial band
                                             (300, 0.1538 * 300),
                                             (1200, 0.1538 * 1200 * (176 / 300))])
    def test_cfar_window_correction_scales_with_training_window(self, m, expected):
        # CMRE janus-c: rx.c:413 passes 0.1538 * fmin(176 / (step_length // 4), 1)
        # and go_cfar.c:327 multiplies it by the training-window length hn - hg == m.
        # The clamp only bites above m = 704, hence the third case.
        assert janus._cfar_window_correction(m) == pytest.approx(expected)
        assert janus._cfar_window_correction(104) == pytest.approx(16.0, abs=0.01)

    def test_cfar_threshold_discounts_contaminated_right_window(self):
        # The correction exists so the preamble's own energy, which fills the right
        # training window for the 128 quarter-chip columns that follow the leading
        # edge, does not inflate the threshold above the edge itself. Levels are
        # chosen to sit between the two multipliers: with w = 0.1538 * m the leading
        # edge crosses, with the bare 0.1538 it does not and the detector falls
        # through to the global argmax of a stronger later arrival.
        cd = 1.0 / (janus.BW_INITIAL / (2 * janus._N_SLOTS))
        m = int(np.floor(janus._CHIP_OVERSAMPLING
                         * max(janus._CFAR_MOV_AVG_TIME, 2 * janus._N_SLOTS * cd) / cd))
        assert m == 104
        edge = 300
        stat = np.full(1200, 3.0)
        stat[edge] = 10.0                                    # leading edge
        stat[edge + 1:edge + 129] = 4.8                      # preamble in the right window
        stat[edge + 400] = 12.0                              # beyond the channel spread
        assert janus._go_cfar(stat, cd) == edge

    def test_doppler_search_can_be_disabled(self):
        bits = self._packet().to_bits()
        fs = 48000.0
        wav = comms.janus_modulate(bits, fs)
        out, ok = comms.janus_demodulate(wav, fs, doppler_max_speed=0)
        assert ok and np.array_equal(out, bits)

    def test_receive_returns_packet_at_non_default_rate(self):
        # End-to-end convenience path at a non-48 kHz rate (resample-first).
        pkt = self._packet()
        fs = 96000.0
        wav = comms.janus_transmit(pkt, fs)
        out, ok = comms.janus_receive(wav, fs)
        assert ok and out.class_id == 16 and out.mobility == 1


class TestSpread:
    def test_despread_recovers_and_gain(self):
        rng = np.random.default_rng(10)
        code = comms.m_sequence(5, [5, 2])
        assert code.size == 31
        syms = comms.Modulator("qpsk").modulate(rng.integers(0, 2, 200))
        rec = comms.despread(comms.spread(syms, code), code)
        assert np.linalg.norm(rec - syms) < 1e-9
        assert comms.processing_gain_db(code) == pytest.approx(10 * np.log10(31))


class TestAgainstPublishedExpressions:
    """Closed forms checked against Proakis & Salehi *Digital Communications*
    5th ed, and the convolutional generators against the code's own trellis.

    Read what the error-probability tests do and do not establish. The expected
    side re-states each expression with ``M`` and ``k`` as literals instead of
    calling :func:`comms.ber_theory`, which pins the scheme-name → ``(M, k)``
    tables and freezes the expressions against later edits. It is not an
    independent reading of the printed equations: the expressions themselves
    are the same ones the implementation carries, so a misreading shared by
    both survives. The equation numbers below are what to re-read for that.

    :meth:`test_constellations_are_gray_mapped` is the one genuinely
    independent check here — it derives the Gray property from the
    constellation rather than assuming it.
    """

    def test_ber_theory_matches_proakis_closed_forms(self):
        """eq. (4.3-29)/(4.3-27): square M-QAM symbol error
        ``4(1-1/sqrt(M))Q(sqrt(3k/(M-1) Eb/N0))``; eq. (4.3-17), M-PSK
        ``2Q(sqrt(2k Eb/N0) sin(pi/M))``; and eq. (4.3-20), Gray mapping makes
        ``P_b = P_M / k``. BPSK/QPSK are exact."""
        from scipy.special import erfc

        def q(x):
            return 0.5 * erfc(np.asarray(x, float) / np.sqrt(2.0))

        ebn0_db = np.array([4.0, 8.0, 12.0, 16.0])
        e = 10 ** (ebn0_db / 10)
        assert np.allclose(comms.ber_theory('bpsk', ebn0_db), q(np.sqrt(2 * e)))
        assert np.allclose(comms.ber_theory('qpsk', ebn0_db), q(np.sqrt(2 * e)))
        for scheme, M in (('8psk', 8), ('16psk', 16)):
            k = np.log2(M)
            assert np.allclose(comms.ber_theory(scheme, ebn0_db),
                               (2 / k) * q(np.sqrt(2 * k * e) * np.sin(np.pi / M)))
        for scheme, M in (('16qam', 16), ('64qam', 64), ('256qam', 256)):
            k = np.log2(M)
            assert np.allclose(
                comms.ber_theory(scheme, ebn0_db),
                (4 / k) * (1 - 1 / np.sqrt(M)) * q(np.sqrt(3 * k / (M - 1) * e)))

    @pytest.mark.parametrize("scheme", ["qpsk", "8psk", "16psk", "16qam", "64qam"])
    def test_constellations_are_gray_mapped(self, scheme):
        """``P_b = P_M/k`` only holds because nearest neighbours differ in one
        bit — the assumption eq. (4.3-20) is built on."""
        mod = comms.Modulator(scheme)
        c = mod.constellation
        k = mod.bits_per_symbol
        bits = ((np.arange(c.size)[:, None] >> np.arange(k)[::-1]) & 1)
        for i in range(c.size):
            d = np.abs(c - c[i])
            d[i] = np.inf
            for j in np.where(np.isclose(d, d.min()))[0]:
                assert np.sum(bits[i] != bits[j]) == 1

    def test_convolutional_generators_give_the_published_free_distance(self):
        """(171, 133) octal at K=7 is the standard rate-1/2 code, free distance
        10. Computing it from uacpy's own trellis checks the generators and the
        state/output mapping together."""
        import heapq
        from uacpy.comms.coding import DEFAULT_K, DEFAULT_POLYS

        K = DEFAULT_K

        def out_bit(state, inbit, poly):
            return bin(((inbit << (K - 1)) | state) & poly).count('1') & 1

        best, pq, dfree = {}, [(0, 0, 0)], None
        while pq:
            w, s, moved = heapq.heappop(pq)
            if moved and s == 0:
                dfree = w
                break
            if best.get((s, moved), 1 << 30) < w:
                continue
            for b in (0, 1):
                if not moved and b == 0:
                    continue                      # must leave the all-zero path
                ns = (s >> 1) | (b << (K - 2))
                nw = w + sum(out_bit(s, b, p) for p in DEFAULT_POLYS)
                if best.get((ns, 1), 1 << 30) > nw:
                    best[(ns, 1)] = nw
                    heapq.heappush(pq, (nw, ns, 1))
        assert dfree == 10

    def test_dsss_processing_gain_is_ten_log_n(self):
        from uacpy.comms.dsss import m_sequence, processing_gain_db
        for n, taps in ((3, [3, 2]), (5, [5, 3]), (7, [7, 6])):
            code = m_sequence(n, taps)
            assert code.size == 2 ** n - 1
            assert processing_gain_db(code) == pytest.approx(
                10 * np.log10(code.size))


class TestMSequencePrimitivity:
    """A non-primitive tap set returns to the seed early, so the register
    cycles a subset of its states. The output still has the right length,
    dtype and +/-1 alphabet, and ``processing_gain_db`` still reports the full
    ``10log10(N)`` — nothing looks wrong until a link budget is far out.
    Measured off-peak autocorrelation: 1 for a real m-sequence, 11 for
    ``[5,1]``, 31 for ``[5]``."""

    @staticmethod
    def _offpeak(s):
        return int(max(abs(np.dot(s, np.roll(s, k))) for k in range(1, s.size)))

    @pytest.mark.parametrize('n,taps', [(5, [5, 2]), (5, [5, 3]),
                                        (6, [6, 1]), (7, [7, 6])])
    def test_primitive_taps_give_a_real_m_sequence(self, n, taps):
        s = comms.m_sequence(n, taps)
        assert s.size == 2 ** n - 1
        assert set(np.unique(s)) <= {-1, 1}
        assert self._offpeak(s) == 1        # the defining property

    @pytest.mark.parametrize('taps', [[5, 1], [5], [5, 4]])
    def test_non_primitive_taps_raise(self, taps):
        with pytest.raises(ConfigurationError, match='not primitive'):
            comms.m_sequence(5, taps)

    @pytest.mark.parametrize('taps', [[0, 2], [9], [2, 2]])
    def test_malformed_tap_sets_raise(self, taps):
        # tap 0 silently aliased to reg[-1] before; tap 9 raised an untyped
        # IndexError; a repeated tap cancels itself leaving no feedback.
        with pytest.raises(ConfigurationError):
            comms.m_sequence(5, taps)


class TestFramingReportsInsteadOfRaising:
    """A corrupted length header is a *failed frame*, not a caller error — a
    receiver cannot tell "corrupt" from "malformed" and should not have to.
    It used to raise ``ConfigurationError: frame truncated (need 4294902280
    bytes, have 520)``. The CRC now also covers the header: with the header
    outside it, a corrupted length steers the slice the CRC is read from, so
    the check could never see the error that caused it."""

    MSG = b"uacpy framing round trip"

    def test_clean_round_trip(self):
        assert comms.unpack_frame(comms.pack_frame(self.MSG)) == (self.MSG, True)

    def test_corrupt_length_header_is_reported_not_raised(self):
        bits = comms.pack_frame(self.MSG).copy()
        bits[3] ^= 1
        assert comms.unpack_frame(bits) == (b"", False)

    def test_corrupt_payload_fails_the_crc(self):
        bits = comms.pack_frame(self.MSG).copy()
        bits[100] ^= 1
        _, ok = comms.unpack_frame(bits)
        assert ok is False

    def test_short_stream_is_reported_not_raised(self):
        assert comms.unpack_frame(np.zeros(8, dtype=np.uint8)) == (b"", False)


def test_janus_wakeup_round_trip():
    """janus_modulate(wakeup=True) and janus_demodulate are inverses: the
    CFAR's forward search must reach past the wake-up prefix the modulator
    itself emits (12 tone chips + the 0.4 s gap = 76 chips; the C
    implementation's 64-chip spread left the preamble outside the window
    and the round trip failed with crc_ok=False)."""
    adb = np.zeros(34, dtype=int)
    adb[:8] = [1, 0, 1, 0, 1, 1, 0, 0]
    bits = comms.JanusPacket(class_id=16, app_type=0, app_data=adb,
                             mobility=1).to_bits()
    fs = 48000.0
    for wakeup in (False, True):
        wav = comms.janus_modulate(bits, fs, wakeup=wakeup)
        out_bits, ok = comms.janus_demodulate(wav, fs)
        assert ok, f"crc failed (wakeup={wakeup})"
        assert np.array_equal(out_bits, bits), f"bits differ (wakeup={wakeup})"


class TestViterbiTieDeterminism:
    """The trellis update resolves equal path metrics toward the even
    predecessor (a state-ascending scan replacing only on improvement), so
    degenerate tie-heavy inputs decode deterministically."""

    def test_all_zero_symbols_decode_to_zeros(self):
        dec = comms.viterbi_decode(np.zeros(200, dtype=int))
        assert np.array_equal(dec, np.zeros(dec.size, dtype=int))
        jdec = comms.janus_decode(np.zeros(144, dtype=int))
        assert np.array_equal(jdec, np.zeros(64, dtype=int))

    def test_roundtrip_survives_heavy_ties(self):
        rng = np.random.default_rng(7)
        info = rng.integers(0, 2, 120)
        coded = comms.conv_encode(info)
        assert np.array_equal(comms.viterbi_decode(coded), info)
        noisy = coded.copy()
        idx = rng.choice(coded.size, size=8, replace=False)
        noisy[idx] ^= 1
        assert np.array_equal(comms.viterbi_decode(noisy), info)
        bits64 = rng.integers(0, 2, 64)
        assert np.array_equal(comms.janus_decode(comms.janus_encode(bits64)),
                              bits64)


class TestDopplerTwoStageMatchesFullScan:
    """The default-grid search (coarse every 15th candidate + fine window)
    returns the same grid candidate as a full 601-point scan, and the
    all-zero-metric guard still raises."""

    def _rx(self, a_true, seed, noise=0.3):
        fs = 12000.0
        t = np.arange(int(0.2 * fs)) / fs
        template = np.sin(2 * np.pi * (2000 * t + 3000 / 0.2 * t ** 2 / 2))
        rng = np.random.default_rng(seed)
        n_rx = int(round(template.size / (1.0 + a_true)))
        sig = np.interp(np.linspace(0, template.size - 1, n_rx),
                        np.arange(template.size), template)
        rx = np.concatenate([np.zeros(200), sig, np.zeros(200)])
        return rx + rng.standard_normal(rx.size) * noise, template

    @pytest.mark.parametrize('a_true', [-5e-3, -1.3e-3, 0.0, 2.1e-3, 5e-3])
    def test_same_candidate_as_full_scan(self, a_true):
        rx, template = self._rx(a_true, seed=int(abs(a_true) * 1e6) + 1)
        best2, _, _ = comms.estimate_doppler_scale(rx, template)
        full = np.linspace(-5e-3, 5e-3, 601)
        bestf, _, _ = comms.estimate_doppler_scale(rx, template, scales=full)
        assert best2 == bestf

    def test_all_zero_metric_still_raises(self):
        with pytest.raises(ConfigurationError, match='no scale candidate'):
            comms.estimate_doppler_scale(np.zeros(100), np.ones(500))
