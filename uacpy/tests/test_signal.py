"""
Smoke tests for uacpy.acoustic_signal (uacpy.acoustic_signal).

Bare minimum that the public API is reachable and behaves on simple inputs.
"""

import numpy as np
import pytest

from uacpy.acoustic_signal.generation import (
    gaussian_pulse, hfm_chirp, lfm_chirp, ricker_wavelet, tone_burst,
)
from uacpy.acoustic_signal.generation import (
    add_noise, make_bandlimited_noise,
)


class TestGenerators:
    """Pulse / chirp generators return finite samples on plausible inputs."""

    def test_gaussian_pulse_shape(self):
        time = np.linspace(0, 0.1, 1024)
        s = gaussian_pulse(time, delay=0.05, duration=0.01)
        assert len(s) == len(time)
        assert np.all(np.isfinite(s))

    def test_lfm_chirp_runs(self):
        fs = 10_000.0
        t, s = lfm_chirp(fmin=100, fmax=2000, T=0.1, sample_rate=fs)
        assert len(t) == len(s)
        assert np.all(np.isfinite(s))

    def test_hfm_chirp_runs(self):
        fs = 10_000.0
        t, s = hfm_chirp(fmin=100, fmax=2000, T=0.1, sample_rate=fs)
        assert len(t) == len(s)
        assert np.all(np.isfinite(s))

    def test_ricker_wavelet_runs(self):
        time = np.linspace(0, 0.1, 1024)
        s = ricker_wavelet(time, F=200.0)
        assert len(s) == len(time)
        assert np.all(np.isfinite(s))

    def test_tone_burst_peaks_at_requested_frequency(self):
        f = 1000.0
        fs = 48_000.0
        s, t = tone_burst(frequency=f, n_cycles=20, sample_rate=fs)
        # FFT peak should sit at f within the resolution.
        S = np.fft.rfft(s)
        freqs = np.fft.rfftfreq(len(s), 1.0 / fs)
        peak = freqs[np.argmax(np.abs(S))]
        assert abs(peak - f) < (fs / len(s)) * 2

    def test_tone_burst_dt_equals_inverse_sample_rate(self):
        """``tone_burst`` builds ``time`` so ``dt == 1 / sample_rate``
        exactly, which keeps round-trip Fourier identities
        (``np.fft.rfftfreq(N, dt)``) honest."""
        fs = 48_000.0
        s, t = tone_burst(frequency=1000.0, n_cycles=5, sample_rate=fs)
        # Identical length.
        assert len(s) == len(t)
        # First sample sits at t=0 (no spurious offset).
        assert t[0] == 0.0
        # ``dt`` exact to float precision — no rescaling.
        dt = t[1] - t[0]
        assert dt == 1.0 / fs
        # Uniform spacing across the whole vector (tolerant of the
        # 1-ulp roundoff that ``np.diff`` introduces on a stride-built
        # array).
        np.testing.assert_allclose(np.diff(t), 1.0 / fs, rtol=1e-12, atol=0)


class TestProcessing:
    """Processing helpers don't blow up on synthetic signals."""

    def test_add_noise_increases_variance(self):
        fs = 10_000.0
        x = np.zeros(1024)
        y = add_noise(
            x, sample_rate=fs,
            source_level_db=120.0, noise_level_db=80.0,
            fc=1000.0, bandwidth=200.0,
        )
        assert np.var(y) > 0

    def test_make_bandlimited_noise_runs(self):
        n = make_bandlimited_noise(
            fc=1000.0, bandwidth=500.0,
            duration=0.1, sample_rate=10_000.0,
        )
        assert len(n) > 0
        assert np.all(np.isfinite(n))


class TestDecidecadeBands:
    def test_standard_iso_centre_frequencies_and_ratio(self):
        from uacpy.acoustic_signal.bands import decidecade_bands
        lo, c, hi = decidecade_bands(100, 10000)
        # base-10 ratio 10^(1/10)
        assert c[1] / c[0] == pytest.approx(10 ** 0.1, rel=1e-6)
        # the 1 kHz band has the ISO nominal edges 891-1122 Hz
        i = int(np.argmin(np.abs(c - 1000)))
        assert lo[i] == pytest.approx(891.25, rel=1e-3)
        assert hi[i] == pytest.approx(1122.0, rel=1e-3)

    def test_white_noise_band_levels_rise_1db_per_band(self):
        from uacpy.acoustic_signal.bands import decidecade_band_levels
        f = np.linspace(1, 20000, 40000)
        psd = np.ones_like(f) * 1e-12               # flat Pa^2/Hz
        c, lv = decidecade_band_levels(psd, f)
        step = np.diff(lv[(c > 200) & (c < 5000)])
        assert np.allclose(step, 1.0, atol=0.05)    # each band 10^0.1 wider -> +1 dB

    def test_bands_validate_input(self):
        from uacpy.acoustic_signal.bands import decidecade_bands
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError):
            decidecade_bands(1000, 100)


def test_acoustic_signal_is_importable():
    import uacpy.acoustic_signal as sig
    import uacpy
    assert sig is uacpy.acoustic_signal


def test_signal_symbols_resolve():
    import uacpy
    for name in ('lfm_chirp', 'hfm_chirp', 'tone_burst', 'gaussian_pulse',
                 'ricker_wavelet', 'add_noise', 'make_bandlimited_noise',
                 'PPSD', 'Spectrogram'):
        assert hasattr(uacpy.acoustic_signal, name), \
            f"uacpy.acoustic_signal.{name} missing"


class TestSEL:
    """SEL must integrate power (Parseval-exact), not over-read the way a
    coherent-normalization Hann taper would (+1.76 dB on stationary signals,
    and an impulse at a segment boundary annihilated)."""

    def test_tone_exposure_is_parseval_exact(self):
        from uacpy.acoustic_signal.analysis import SEL
        fs = 48000
        t = np.arange(fs) / fs
        x = 2.0 * np.sin(2 * np.pi * 1000.0 * t)   # exposure = A^2/2 * T = 2.0
        sel, _ = SEL(band_type='third_octave', fmin=10, fmax=20000).compute(
            x, fs, nfft=fs)
        assert sel.sum() == pytest.approx(np.sum(x ** 2) / fs, rel=1e-6)

    def test_impulse_not_annihilated(self):
        from uacpy.acoustic_signal.analysis import SEL
        fs = 48000
        imp = np.zeros(fs)
        imp[0] = 10.0   # a Hann-windowed single segment would zero this out
        sel, _ = SEL(band_type='linear', fmin=1.0, fmax=fs / 2,
                     num_bands=240).compute(imp, fs, nfft=fs)
        # full-band exposure ≈ Σx²/fs (only the excluded DC bin is dropped)
        assert sel.sum() == pytest.approx(np.sum(imp ** 2) / fs, rel=1e-3)


class TestFRF:
    """FRF automatic FIR-order selection (m='AIC'|'BIC'|'FPE'|'CP') must run,
    not crash with 'count >= None' from an un-defaulted stop_count."""

    @pytest.mark.parametrize("criterion", ['AIC', 'BIC', 'FPE', 'CP'])
    def test_auto_order_runs_and_recovers_order(self, criterion):
        from uacpy.acoustic_signal.system_id import FRF
        rng = np.random.default_rng(1)
        u = rng.standard_normal(2000)
        g = np.array([1.0, -0.5, 0.25])                  # order-3 FIR
        y = np.convolve(u, g)[:u.size] + 0.01 * rng.standard_normal(2000)
        frf = FRF()
        _, tf = frf.compute(u, y, 1000.0, method='ls_fir', m=criterion)
        assert np.isfinite(tf).all()
        # every criterion recovers the true order-3 FIR at this SNR
        assert frf.m == 3
