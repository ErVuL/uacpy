"""
Smoke tests for uacpy.acoustic_signal (uacpy.acoustic_signal).

Bare minimum that the public API is reachable and behaves on simple inputs.
"""

import numpy as np

from uacpy.acoustic_signal.generation import (
    gaussian_pulse, hfm_chirp, lfm_chirp, ricker_wavelet, tone_burst,
)
from uacpy.acoustic_signal.processing import (
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
