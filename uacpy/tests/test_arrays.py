"""Tests for ``uacpy.acoustic_signal.arrays`` — steering vectors and the
Bartlett / MVDR / MUSIC beamformers on a synthetic line array.
"""

import numpy as np
import pytest

from uacpy.acoustic_signal import (
    bartlett_spectrum,
    music_spectrum,
    mvdr_spectrum,
    sample_covariance,
    steering_vectors,
    shading_taper,
)
from uacpy.core.exceptions import ConfigurationError


FREQ = 1000.0
C = 1500.0


def _array():
    # Half-wavelength spacing at FREQ.
    spacing = C / FREQ / 2.0
    return np.arange(16) * spacing


def _snapshots(positions, true_angle_deg, n_snaps=400, snr=20.0):
    a = steering_vectors(positions, [true_angle_deg], FREQ, C)[0]
    rng = np.random.default_rng(0)
    src = rng.standard_normal(n_snaps) + 1j * rng.standard_normal(n_snaps)
    noise_amp = 10 ** (-snr / 20.0)
    noise = noise_amp * (rng.standard_normal((positions.size, n_snaps))
                         + 1j * rng.standard_normal((positions.size, n_snaps)))
    return np.outer(a, src) + noise


class TestSteering:
    def test_unit_norm_rows(self):
        e = steering_vectors(_array(), [-30, 0, 30], FREQ, C)
        assert np.allclose(np.linalg.norm(e, axis=1), 1.0)
        assert e.shape == (3, 16)


class TestCovariance:
    def test_hermitian_and_loading(self):
        x = _snapshots(_array(), 10.0)
        R = sample_covariance(x)
        assert np.allclose(R, R.conj().T)
        R_loaded = sample_covariance(x, diagonal_loading=0.1)
        assert np.trace(R_loaded).real > np.trace(R).real

    def test_requires_2d(self):
        with pytest.raises(ConfigurationError):
            sample_covariance(np.zeros(8))


class TestBeamformers:
    angles = np.linspace(-60, 60, 241)

    def test_bartlett_recovers_doa(self):
        R = sample_covariance(_snapshots(_array(), 15.0))
        p = bartlett_spectrum(R, steering_vectors(_array(), self.angles, FREQ, C))
        assert self.angles[np.argmax(p)] == pytest.approx(15.0, abs=1.0)

    def test_mvdr_recovers_doa(self):
        R = sample_covariance(_snapshots(_array(), -20.0))
        p = mvdr_spectrum(R, steering_vectors(_array(), self.angles, FREQ, C))
        assert self.angles[np.argmax(p)] == pytest.approx(-20.0, abs=1.0)

    def test_music_recovers_doa(self):
        R = sample_covariance(_snapshots(_array(), 5.0))
        p = music_spectrum(R, steering_vectors(_array(), self.angles, FREQ, C), 1)
        assert self.angles[np.argmax(p)] == pytest.approx(5.0, abs=1.0)

    def test_music_rejects_bad_source_count(self):
        R = sample_covariance(_snapshots(_array(), 5.0))
        with pytest.raises(ConfigurationError):
            music_spectrum(R, steering_vectors(_array(), self.angles, FREQ, C), 16)


class TestTaper:
    def test_mean_normalised(self):
        w = shading_taper(16, "hann")
        assert w.size == 16
        assert np.mean(w) == pytest.approx(1.0)
