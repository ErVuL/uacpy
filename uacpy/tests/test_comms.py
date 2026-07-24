"""Tests for the ``uacpy.comms`` package — modulation, channels, receivers, coding.

Behavioral: BER vs AWGN theory, equalizer opens a closed eye, OFDM/FEC/DSSS
round-trip, sparse channel estimation. Pure-Python; no model binary.
"""

import numpy as np
import pytest

from uacpy import comms
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


class TestEqualization:
    def test_dfe_opens_closed_eye(self):
        rng = np.random.default_rng(0xACED)
        h = comms.multipath_channel([0.0, 1 / 8000, 2 / 8000],
                                    [1.0, 0.6, 0.3], 8000)
        raw = comms.simulate_link("qpsk", 16.0, 40000, channel=h, rng=rng).ber
        dfe = comms.DFE(n_ff=12, n_fb=6, forget=0.995)
        eq = comms.simulate_link("qpsk", 16.0, 40000, channel=h,
                                 equalizer=dfe, rng=rng)
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
        assert best == pytest.approx(a_true, abs=2e-4)


class TestSync:
    def test_preamble_detected_at_offset(self):
        rng = np.random.default_rng(5)
        pre = comms.Modulator("qpsk").modulate(rng.integers(0, 2, 128))
        offset = 200
        rx = np.concatenate([np.zeros(offset, complex), pre,
                             0.1 * (rng.standard_normal(300) + 1j * rng.standard_normal(300))])
        k, metric = comms.detect_preamble(rx, pre, threshold=0.5)
        assert abs(k - offset) <= 1


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


class TestChannelEstimation:
    def test_ls_and_omp_recover_sparse_channel(self):
        rng = np.random.default_rng(6)
        h = np.zeros(20, complex)
        h[[0, 5, 11]] = [1.0, 0.5j, -0.3]
        pilots = comms.Modulator("qpsk").modulate(rng.integers(0, 2, 2000))
        rx = comms.apply_channel(pilots, h)[: pilots.size]
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
        assert abs(start - lead) <= int(0.01 * fs)        # within ~10 ms

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
        wav = comms.janus.transmit(pkt, fs)
        out, ok = comms.janus.receive(wav, fs)
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
