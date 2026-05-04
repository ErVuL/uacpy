"""
Smoke tests for uacpy.signal (uacpy.acoustic_signal).

This is the bare minimum coverage of the signal-generation and processing
helpers — shape, dtype, and a few easily-checked numerical invariants:
  - LFM/HFM chirps span the requested duration
  - tone burst peaks at the requested frequency in the FFT
  - Ricker / Gaussian pulses are real and finite
  - bandlimited noise has the requested bandwidth in the spectrum
  - add_noise scales the signal by the source level
"""

import numpy as np
import pytest

from uacpy.acoustic_signal.generation import (
    gaussian_pulse,
    hfm_chirp,
    lfm_chirp,
    ricker_wavelet,
    tone_burst,
)
from uacpy.acoustic_signal.processing import (
    add_noise,
    make_bandlimited_noise,
)


class TestGenerators:
    def test_lfm_chirp_shape_and_finite(self):
        s, t = lfm_chirp(fmin=100.0, fmax=1000.0, T=0.5, sample_rate=10_000.0)
        assert s.shape == t.shape
        assert s.shape == (5000,)
        assert np.all(np.isfinite(s))
        # Sweeps from low to high — instantaneous frequency monotone in mean.
        # Cheap sanity: zero-crossings must occur (signal is oscillatory).
        zc = np.sum(np.diff(np.sign(s)) != 0)
        assert zc > 50, "chirp should have many zero crossings"

    def test_hfm_chirp_shape_and_finite(self):
        s, t = hfm_chirp(fmin=1000.0, fmax=5000.0, T=0.1, sample_rate=48_000.0)
        assert s.shape == t.shape
        assert np.all(np.isfinite(s))
        assert np.max(np.abs(s)) <= 1.0 + 1e-9, "HFM uses sine, magnitude ≤1"

    def test_tone_burst_peaks_at_requested_frequency(self):
        f = 1000.0
        fs = 48_000.0
        s, t = tone_burst(frequency=f, n_cycles=20, sample_rate=fs)
        spec = np.abs(np.fft.rfft(s))
        freqs = np.fft.rfftfreq(len(s), d=1.0 / fs)
        peak_freq = freqs[np.argmax(spec)]
        # Hanning-windowed burst: peak should be within one bin of nominal f
        bin_width = freqs[1] - freqs[0]
        assert abs(peak_freq - f) <= 2 * bin_width, (
            f"tone burst peak at {peak_freq:.1f} Hz, expected ~{f:.1f} Hz"
        )

    def test_ricker_wavelet_real_and_finite(self):
        time = np.linspace(0, 0.1, 1000)
        s = ricker_wavelet(time, F=50.0)
        assert s.shape == time.shape
        assert np.all(np.isfinite(s))
        assert np.isrealobj(s)

    def test_gaussian_pulse_peaks_at_delay(self):
        time = np.linspace(0, 1.0, 1001)
        pulse = gaussian_pulse(time, delay=0.5, duration=0.05)
        peak_idx = int(np.argmax(pulse))
        assert abs(time[peak_idx] - 0.5) < 1e-3
        # Pulse must decay below 1% by ~3*duration away from peak
        far_idx = int(np.argmin(np.abs(time - (0.5 + 0.3))))
        assert pulse[far_idx] < 0.01


class TestProcessing:
    def test_make_bandlimited_noise_has_correct_band(self):
        fc, bw = 5_000.0, 1_000.0
        fs, T = 50_000.0, 0.5
        noise = make_bandlimited_noise(fc=fc, bandwidth=bw, duration=T, sample_rate=fs)
        assert np.all(np.isfinite(noise))
        spec = np.abs(np.fft.rfft(noise))
        freqs = np.fft.rfftfreq(len(noise), d=1.0 / fs)

        in_band = (freqs >= fc - bw / 2) & (freqs <= fc + bw / 2)
        # Out-of-band should be much smaller than in-band — at least 10× drop
        # (the brick-wall isn't perfect, but order-of-magnitude is unmistakable).
        in_power = np.mean(spec[in_band] ** 2)
        out_power = np.mean(spec[~in_band] ** 2)
        assert in_power > 10 * out_power, (
            f"bandlimited noise leaks: in-band power {in_power:.2g} vs "
            f"out-of-band {out_power:.2g}"
        )

    def test_add_noise_scales_clean_signal(self):
        """Source level must scale a 0 dB clean signal accordingly."""
        fs = 10_000.0
        # Tiny clean signal at amplitude 1
        clean = np.sin(2 * np.pi * 1000 * np.arange(int(fs)) / fs)
        # Source level 100 dB ≡ amplitude factor 10**5 = 100000
        noisy = add_noise(
            clean, sample_rate=fs,
            source_level_db=100.0, noise_level_db=-100.0,
            fc=1000.0, bandwidth=200.0,
        )
        # With noise effectively zero, RMS(noisy) ≈ RMS(clean)*10^(SL/20)
        rms_clean = np.sqrt(np.mean(clean**2))
        rms_noisy = np.sqrt(np.mean(noisy**2))
        expected_factor = 10.0 ** (100.0 / 20.0)
        ratio = rms_noisy / (rms_clean * expected_factor)
        assert 0.5 < ratio < 2.0, (
            f"add_noise scaling off: ratio {ratio:.3f}, expected ≈1 "
            f"(noisy_rms={rms_noisy:.3g}, clean_rms*SL={rms_clean*expected_factor:.3g})"
        )


def test_signal_alias_resolves_to_acoustic_signal_package():
    """`uacpy.signal` is exposed as an alias for the acoustic_signal package."""
    import uacpy

    assert hasattr(uacpy, "signal")
    # Submodules must be reachable via the alias
    for sub in ("generation", "processing", "advanced", "analysis"):
        assert hasattr(uacpy.signal, sub), f"uacpy.signal.{sub} missing"
    # Curated top-level generators must resolve directly off the alias —
    # used to silently disappear because the package __init__ wrapped its
    # imports in try/except over names that didn't exist in generation.py.
    assert uacpy.signal.lfm_chirp is lfm_chirp
    assert uacpy.signal.tone_burst is tone_burst
    assert uacpy.signal.ricker_wavelet is ricker_wavelet


def test_signal_all_symbols_resolve():
    """Every name in __all__ must actually be importable. Guards against the
    drifted-import-block bug where __init__ listed names that did not exist."""
    from uacpy import signal

    missing = [name for name in signal.__all__ if not hasattr(signal, name)]
    assert not missing, f"uacpy.signal.__all__ lists missing names: {missing}"
