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
    # 241 nodes over 120 deg = 0.5 deg spacing, and the spectra are read with
    # a bare argmax, so the abs=1.0 below is a two-node allowance on the scan
    # grid — not a bearing-accuracy claim.
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


@pytest.mark.parametrize("true_deg", [15.0, -30.0, 45.0])
def test_beamform_resolves_true_angle_not_mirror(true_deg):
    """beamform must resolve a source at +theta (not the mirror -theta) —
    consistent with bartlett/mvdr/music (the steering vector is conjugated)."""
    from uacpy.acoustic_signal import beamform, steering_vectors
    c, f = 1500.0, 1500.0
    pos = np.arange(16) * (c / f / 2.0)
    ang = np.linspace(-60, 60, 241)
    a = steering_vectors(pos, [true_deg], f, c)[0]
    snr, angles, _ = beamform(a[:, None], pos, f, angles=ang, SL=0, NL=0)
    assert abs(angles[np.argmax(snr[:, 0])] - true_deg) < 1.0


def test_mvdr_music_no_divide_warning_and_music_peaks_at_source():
    import warnings
    pos = np.arange(6) * 0.75
    angles = np.linspace(-60, 60, 121)
    e = steering_vectors(pos, angles, 1000.0)
    src = steering_vectors(pos, [10.0], 1000.0)[0]
    R = np.eye(6, dtype=complex) + 8 * np.outer(src, src.conj())
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)   # no spurious 1/denom warning
        m = mvdr_spectrum(R, e)
        mu = music_spectrum(R, e, 1)
    assert np.all(np.isfinite(m))
    # the sharp MUSIC peak at the source direction is the intended behaviour,
    # preserved (not clamped) — it localises the source at 10 deg
    assert abs(angles[np.argmax(mu)] - 10.0) < 2.0


class TestPowerlessCovariance:
    """``diagonal_loading`` is a *fraction of* ``trace(R)/N``, so it vanishes
    with the trace: it stabilises a rank-deficient covariance that still
    carries power but cannot rescue an all-zero one. That is ordinary data —
    ``sample_covariance`` of a silent segment (dead element, digital silence)
    returns exactly it. Without the guard MVDR's inverse is singular and
    MUSIC's noise subspace is arbitrary, so both must decline rather than
    return a finite *uniform* pseudospectrum that looks like an answer."""

    @staticmethod
    def _rig(n=8):
        from uacpy.acoustic_signal.arrays import steering_vectors
        return steering_vectors(np.arange(n) * 0.75,
                                np.linspace(-90.0, 90.0, 37), 1000.0)

    def test_silence_gives_nan_and_warns_not_a_raw_linalg_error(self):
        from uacpy.acoustic_signal.arrays import (
            sample_covariance, mvdr_spectrum, music_spectrum)
        E = self._rig()
        R = sample_covariance(np.zeros((8, 200), dtype=complex))
        for fn in (lambda: mvdr_spectrum(R, E),
                   lambda: music_spectrum(R, E, 1)):
            with pytest.warns(UserWarning, match='no power'):
                out = fn()
            assert np.all(np.isnan(out))

    def test_bartlett_reports_zero_power_which_is_the_true_answer(self):
        from uacpy.acoustic_signal.arrays import (
            sample_covariance, bartlett_spectrum)
        R = sample_covariance(np.zeros((8, 200), dtype=complex))
        assert np.all(bartlett_spectrum(R, self._rig()) == 0.0)

    def test_a_powered_rank_deficient_covariance_still_resolves(self):
        """The case the loading exists for must keep working: fewer snapshots
        than elements, but non-zero trace."""
        from uacpy.acoustic_signal.arrays import (
            steering_vectors, sample_covariance, mvdr_spectrum)
        rng = np.random.default_rng(0)
        a = steering_vectors(np.arange(8) * 0.75, [15.0], 1000.0).T * np.sqrt(8)
        x = a @ (rng.normal(size=(1, 4)) + 1j * rng.normal(size=(1, 4)))
        angles = np.linspace(-90.0, 90.0, 721)
        P = mvdr_spectrum(sample_covariance(x),
                          steering_vectors(np.arange(8) * 0.75, angles, 1000.0))
        assert np.all(np.isfinite(P))
        assert angles[np.argmax(P)] == pytest.approx(15.0, abs=0.5)


@pytest.mark.requires_binary
def test_look_direction_agrees_with_an_independently_propagated_field():
    """A downgoing arrival is reported at *positive* declination.

    The field comes from Bellhop, so its depth-phase sign is the one the solver
    produces under the Acoustics-Toolbox ``exp(+i*omega*t)`` convention
    (``KrakenField/EvaluateMod.f90:41``) with depth positive down
    (``Bellhop/bellhop.f90:453``), rather than one this module assumes. A test
    that builds its snapshots from :func:`steering_vectors` cannot detect a
    mirrored look direction, because the same convention appears on both sides
    and the error cancels.
    """
    from uacpy import (Environment, Source, Receiver, Bellhop,
                       SoundSpeedProfile, BoundaryProperties)
    from uacpy.acoustic_signal import beamform

    c, freq, z_src, range_m = 1500.0, 200.0, 200.0, 10_000.0
    z_rcv = np.arange(1400.0, 1601.0, 3.0)
    env = Environment(
        bathymetry=5000.0,
        ssp=SoundSpeedProfile.from_pairs([(0.0, c), (5000.0, c)]),
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=1800.0, density=1.8,
                                  attenuation=0.5),
    )
    field = Bellhop(beam_type='G', n_beams=4001, alpha=(6.0, 9.0)).compute_tl(
        env, Source(depths=z_src, frequencies=freq),
        Receiver(depths=z_rcv, ranges=[range_m]))
    p = np.asarray(field.data).ravel()

    # The narrow fan traces the direct eigenray only, so this is one plane wave
    # and the array sees a single unambiguous arrival angle.
    amp = np.abs(p)
    assert amp.max() / amp.min() < 1.05

    expected_deg = np.degrees(np.arctan2(z_rcv.mean() - z_src, range_m))
    assert expected_deg > 0.0                      # receivers below the source

    angles = np.arange(-20.0, 20.01, 0.05)
    out = beamform(p, z_rcv, freq, angles=angles, c=c)
    peak_deg = out.angles[np.argmax(out.snr)]
    assert peak_deg == pytest.approx(expected_deg, abs=0.5)
