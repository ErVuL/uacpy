"""Tests for ``uacpy.acoustic_signal`` channel / modal / time-frequency tools."""

import numpy as np
import pytest

from uacpy.acoustic_signal import (
    analytic_signal,
    cepstrum,
    complex_cepstrum,
    cwt,
    envelope,
    inverse_cwt,
    inverse_complex_cepstrum,
    impulse_response,
    impulse_response_from_transfer_function,
    instantaneous_frequency,
    modal_group_velocity,
    simulate_reception,
    unwarp_signal,
    warp_signal,
    wigner_ville,
)
from uacpy.core.exceptions import ConfigurationError


FS = 10000.0


class TestChannel:
    def test_integer_delays_place_taps(self):
        t, h = impulse_response([1.0, 0.5], [0.01, 0.02], FS, fractional=False)
        assert h[100] == pytest.approx(1.0)
        assert h[200] == pytest.approx(0.5)

    def test_fractional_splits_energy(self):
        # Delay 0.0105 s = sample 105.0 exactly -> single tap even when fractional.
        _, h = impulse_response([1.0], [0.0105], FS, fractional=True)
        assert h[105] == pytest.approx(1.0)
        # A genuinely fractional delay splits between two taps but conserves sum.
        _, h2 = impulse_response([1.0], [0.01005], FS, fractional=True)
        assert h2.sum() == pytest.approx(1.0)
        assert np.count_nonzero(h2) == 2

    def test_simulate_reception_shifts_transmit(self):
        tx = np.array([1.0, -1.0, 0.5])
        t, rx = simulate_reception(tx, [1.0], [0.01], FS)
        assert np.allclose(rx[100:103], tx)

    def test_ir_from_flat_transfer_function_is_delta(self):
        f = np.linspace(0, FS / 2, 65)
        H = np.ones_like(f, dtype=complex)
        _, h = impulse_response_from_transfer_function(H, f, FS, n_samples=128)
        assert np.argmax(np.abs(h)) == 0

    def test_negative_delay_raises(self):
        with pytest.raises(ConfigurationError):
            impulse_response([1.0], [-0.01], FS)


class TestModal:
    def test_group_velocity_of_nondispersive_guide(self):
        f = np.linspace(50, 500, 50)
        c = 1500.0
        kr = 2 * np.pi * f / c  # omega = c*kr -> vg = c
        vg = modal_group_velocity(f, kr)
        assert np.allclose(vg, c, rtol=1e-6)

    def test_warp_unwarp_roundtrip(self):
        # Band-limited modal-like transient (warping targets dispersive arrivals).
        n = 1024
        t = np.arange(n) / FS
        x = (np.sin(2 * np.pi * 40 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)) \
            * np.hanning(n)
        w, tw = warp_signal(x, FS, 500.0)
        x2, _ = unwarp_signal(w, tw, FS, 500.0)
        m = min(x.size, x2.size)
        corr = np.corrcoef(x[10:m - 10], x2[10:m - 10])[0, 1]
        assert corr > 0.9


class TestTimeFrequency:
    def test_analytic_signal_real_part(self):
        x = np.cos(2 * np.pi * 50 * np.arange(1000) / FS)
        z = analytic_signal(x)
        assert np.allclose(z.real, x, atol=1e-9)

    def test_envelope_of_tone_is_flat(self):
        x = np.cos(2 * np.pi * 200 * np.arange(2000) / FS)
        env = envelope(x)
        assert np.std(env[100:-100]) < 0.05

    def test_instantaneous_frequency_of_tone(self):
        f0 = 300.0
        x = np.cos(2 * np.pi * f0 * np.arange(2000) / FS)
        inst = instantaneous_frequency(x, FS)
        assert np.median(inst[50:-50]) == pytest.approx(f0, abs=2.0)

    def test_wigner_ville_localises_tone(self):
        f0 = 250.0
        x = np.cos(2 * np.pi * f0 * np.arange(256) / FS)
        t, f, W = wigner_ville(x, FS)
        peak_f = f[np.argmax(W.mean(axis=1))]
        assert peak_f == pytest.approx(f0, abs=FS / 256)

    def test_cepstrum_finite(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(512)
        assert np.all(np.isfinite(cepstrum(x)))
        assert np.all(np.isfinite(complex_cepstrum(x)))

    def test_complex_cepstrum_round_trip(self):
        # Smooth signal: phase unwraps cleanly so the homomorphic round trip
        # is near-exact (white noise is the pathological case for unwrap).
        t = np.arange(256) / FS
        x = np.sin(2 * np.pi * 60 * t) * np.exp(-0.5 * ((t - 0.12) / 0.04) ** 2)
        rec = inverse_complex_cepstrum(complex_cepstrum(x))
        assert np.max(np.abs(x - rec)) < 1e-8

    def test_cepstrum_echo_peak(self):
        # An echo at delay D shows a cepstral peak at quefrency D.
        rng = np.random.default_rng(0)
        n = 2048
        x = rng.standard_normal(n)
        d = 120
        x[d:] += 0.8 * x[:-d]
        c = cepstrum(x)
        lo, hi = 20, n // 2
        assert lo + np.argmax(c[lo:hi]) == pytest.approx(d, abs=1)


class TestCWT:
    @pytest.mark.parametrize("wavelet", ["morlet", "paul", "dog"])
    def test_localizes_tone(self, wavelet):
        x = np.cos(2 * np.pi * 200 * np.arange(2048) / FS)
        freqs, W = cwt(x, FS, wavelet=wavelet)
        ridge = freqs[np.argmax(np.abs(W).mean(axis=1))]
        assert ridge == pytest.approx(200.0, rel=0.08)

    def test_shape_and_explicit_freqs(self):
        x = np.cos(2 * np.pi * 100 * np.arange(1024) / FS)
        freqs = np.array([50.0, 100.0, 200.0])
        f, W = cwt(x, FS, freqs=freqs)
        assert W.shape == (3, 1024)
        assert np.iscomplexobj(W)

    def test_bad_wavelet_raises(self):
        with pytest.raises(ConfigurationError):
            cwt(np.zeros(128), FS, wavelet="haar")

    @pytest.mark.parametrize("wavelet", ["morlet", "paul", "dog"])
    def test_icwt_round_trip_shape(self, wavelet):
        t = np.arange(512) / FS
        x = (np.sin(2 * np.pi * 60 * t) * np.exp(-0.5 * ((t - 0.25) / 0.05) ** 2)
             + 0.4 * np.sin(2 * np.pi * 150 * t))
        f, W = cwt(x, FS, wavelet=wavelet, n_freqs=96)
        xr = inverse_cwt(W, f, FS, wavelet=wavelet)
        assert np.corrcoef(x, xr)[0, 1] > 0.95

    def test_icwt_bad_shape_raises(self):
        with pytest.raises(ConfigurationError):
            inverse_cwt(np.zeros((3, 10)), np.array([1.0, 2.0]), FS)
