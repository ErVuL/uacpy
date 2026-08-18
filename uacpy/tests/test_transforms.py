"""Tests for ``uacpy.acoustic_signal.transforms`` — tau-p / Radon / f-k
transforms and their inverses (functional API)."""

import numpy as np
import pytest

from uacpy.acoustic_signal import (
    fk_transform,
    inverse_fk,
    inverse_radon,
    inverse_taup,
    radon_transform,
    taup_transform,
)
from uacpy.core.exceptions import ConfigurationError

FS = 1000.0
NT, NX, DX = 512, 48, 10.0
_t = np.arange(NT) / FS
_x = np.arange(NX) * DX


def _ricker(tc, F=40.0):
    u = 2 * np.pi * F * (_t - tc)
    return (1 - 0.5 * u ** 2) * np.exp(-0.25 * u ** 2)


def _linear_gather(p0, tau0):
    g = np.zeros((NT, NX))
    for ix in range(NX):
        g[:, ix] += _ricker(tau0 + p0 * _x[ix])
    return g


def _hyperbolic_gather(v0, tau0):
    g = np.zeros((NT, NX))
    for ix in range(NX):
        g[:, ix] += _ricker(np.sqrt(tau0 ** 2 + (_x[ix] / v0) ** 2))
    return g


class TestTauP:
    def test_focuses_linear_event(self):
        p0, tau0 = 1 / 1500.0, 0.05
        p, tau, U = taup_transform(_linear_gather(p0, tau0), FS, DX,
                                   p_max=1 / 1000.0, n_slowness=241)
        i, j = np.unravel_index(np.argmax(np.abs(U)), U.shape)
        assert p[i] == pytest.approx(p0, abs=p[1] - p[0])
        assert tau[j] == pytest.approx(tau0, abs=2 / FS)

    def test_compute_then_standalone_inverse(self):
        g = _linear_gather(1 / 1500.0, 0.05)
        slow, tau, U = taup_transform(g, FS, DX, p_max=1 / 1000.0, n_slowness=241)
        rec = inverse_taup(U, slow, FS, DX, NX)
        assert rec.shape == g.shape
        # inverse_taup is the adjoint slant stack, not an exact inverse: a
        # finite slowness fan and 48 traces leave amplitude taper and aliasing
        # fringes. 0.85 asks that the event be recovered in shape and position,
        # which is what the adjoint promises; it is not a round-trip bound.
        # The same figure governs the two other reconstruction checks below.
        assert np.corrcoef(g.ravel(), rec.ravel())[0, 1] > 0.85

    def test_requires_2d(self):
        with pytest.raises(ConfigurationError):
            taup_transform(np.zeros(10), FS, DX)

    def test_zero_pad_and_window_still_focus(self):
        p0, tau0 = 1 / 1500.0, 0.05
        p, tau, U = taup_transform(_linear_gather(p0, tau0), FS, DX,
                                   p_max=1 / 1000.0, n_slowness=241,
                                   window="hann", nfft=1024)
        assert U.shape[1] == 1024 and tau.size == 1024
        i, j = np.unravel_index(np.argmax(np.abs(U)), U.shape)
        assert p[i] == pytest.approx(p0, abs=p[1] - p[0])
        assert tau[j] == pytest.approx(tau0, abs=2 / FS)

    def test_nfft_truncation_raises(self):
        with pytest.raises(ConfigurationError):
            taup_transform(np.zeros((512, 8)), FS, DX, nfft=256)


class TestRadon:
    def test_linear_peak_slowness(self):
        p0 = 1 / 1500.0
        mv = np.linspace(-1e-3, 1e-3, 241)
        m, tau, R = radon_transform(_linear_gather(p0, 0.05), FS, DX, mv,
                                    kind='linear')
        assert m[np.argmax(np.abs(R)) // R.shape[1]] == pytest.approx(
            p0, abs=mv[1] - mv[0])

    def test_hyperbolic_peak_velocity(self):
        v0, tau0 = 1500.0, 0.08
        vels = np.linspace(1200, 2000, 81)
        m, tau, R = radon_transform(_hyperbolic_gather(v0, tau0), FS, DX, vels,
                                    kind='hyperbolic')
        i, j = np.unravel_index(np.argmax(np.abs(R)), R.shape)
        # Two velocity nodes and three time samples of slack: the hyperbolic
        # apex is flat in v near the true value, so the focus straddles nodes
        # more than the linear case above (one node, two samples) does.
        assert m[i] == pytest.approx(v0, abs=2 * (vels[1] - vels[0]))
        assert tau[j] == pytest.approx(tau0, abs=3 / FS)

    def test_parabolic_runs(self):
        g = _linear_gather(1 / 2000.0, 0.05)
        q = np.linspace(-1e-6, 1e-6, 41)
        m, tau, R = radon_transform(g, FS, DX, q, kind='parabolic')
        assert R.shape == (41, NT)

    def test_inverse_reconstructs(self):
        g = _linear_gather(1 / 1500.0, 0.05)
        mv = np.linspace(-1e-3, 1e-3, 241)
        _, _, R = radon_transform(g, FS, DX, mv, kind='linear')
        back = inverse_radon(R, FS, DX, mv, NX, kind='linear')
        assert back.shape == g.shape
        assert np.corrcoef(g.ravel(), back.ravel())[0, 1] > 0.85

    def test_bad_kind_raises(self):
        with pytest.raises(ConfigurationError):
            radon_transform(np.zeros((NT, NX)), FS, DX, [1.0], kind='cubic')


class TestFKInverse:
    def test_exact_round_trip(self):
        g = _linear_gather(1 / 1500.0, 0.05)
        _, _, _, spectrum = fk_transform(g, FS, DX)
        assert np.max(np.abs(inverse_fk(spectrum) - g)) < 1e-9

    def test_calibrated_psd_parseval(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((256, 64))
        f, k, psd, _ = fk_transform(data, FS, DX, normalize=True)
        lhs = psd.sum() * (f[1] - f[0]) * (k[1] - k[0])
        assert lhs == pytest.approx(np.mean(data ** 2), rel=1e-9)


class TestFKWindowing:
    def test_windowed_psd_parseval(self):
        from scipy.signal import get_window
        rng = np.random.default_rng(0)
        data = rng.standard_normal((256, 64))
        wt = get_window("hann", 256, fftbins=True)
        wx = get_window("hann", 64, fftbins=True)
        f, k, psd, _ = fk_transform(data, FS, DX, normalize=True, window="hann")
        lhs = psd.sum() * (f[1] - f[0]) * (k[1] - k[0])
        expected = (np.sum(data ** 2 * wt[:, None] ** 2 * wx[None, :] ** 2)
                    / (np.sum(wt ** 2) * np.sum(wx ** 2)))
        assert lhs == pytest.approx(expected, rel=1e-9)

    def test_per_axis_window_list(self):
        data = _linear_gather(1 / 1500.0, 0.05)
        f, k, panel, _ = fk_transform(data, FS, DX,
                                      window=["hann", ("kaiser", 8)])
        assert panel.shape == data.shape

    def test_bad_window_list_raises(self):
        with pytest.raises(ConfigurationError):
            fk_transform(np.zeros((32, 8)), FS, DX, window=["hann"])

    def test_zero_pad_grows_axes_pad_independent(self):
        rng = np.random.default_rng(1)
        data = rng.standard_normal((256, 64))
        f, k, psd, _ = fk_transform(data, FS, DX, normalize=True, nfft=(512, 128))
        assert psd.shape == (512, 128)
        assert f.size == 512 and k.size == 128
        lhs = psd.sum() * (f[1] - f[0]) * (k[1] - k[0])
        assert lhs == pytest.approx(np.mean(data ** 2), rel=1e-9)

    def test_nfft_truncation_raises(self):
        with pytest.raises(ConfigurationError):
            fk_transform(np.zeros((256, 64)), FS, DX, nfft=(128, 32))

    def test_compute_requires_2d(self):
        with pytest.raises(ConfigurationError):
            fk_transform(np.zeros(10), FS, DX)


class TestBringYourOwn:
    """Inverse a transform panel obtained elsewhere (no prior forward call)."""

    def test_taup_functional_round_trip(self):
        g = _linear_gather(1 / 1500.0, 0.05)
        slow, tau, U = taup_transform(g, FS, DX, p_max=1 / 1000.0, n_slowness=241)
        rec = inverse_taup(U, slow, FS, DX, NX)
        assert rec.shape == g.shape
        assert np.corrcoef(g.ravel(), rec.ravel())[0, 1] > 0.85

    def test_fk_inverse_standalone(self):
        g = _linear_gather(1 / 1500.0, 0.05)
        my_FK = np.fft.fftshift(np.fft.fft2(g))
        assert np.max(np.abs(inverse_fk(my_FK) - g)) < 1e-9

    def test_inverse_taup_bad_shape_raises(self):
        with pytest.raises(ConfigurationError):
            inverse_taup(np.zeros((3, 10)), np.array([1.0, 2.0]), FS, DX, NX)


class TestRadonHyperbolicMoveoutValidation:
    def test_forward_rejects_nonpositive_velocity(self):
        with pytest.raises(ConfigurationError, match="> 0"):
            radon_transform(np.zeros((NT, NX)), FS, DX, [0.0, 1500.0],
                            kind='hyperbolic')

    def test_inverse_rejects_nonpositive_velocity(self):
        with pytest.raises(ConfigurationError, match="> 0"):
            inverse_radon(np.zeros((2, NT)), FS, DX, [-1.0, 1500.0], NX,
                          kind='hyperbolic')

    def test_positive_velocities_yield_finite_panel(self):
        m, tau, R = radon_transform(_hyperbolic_gather(1500.0, 0.08), FS, DX,
                                    np.linspace(1200, 2000, 5),
                                    kind='hyperbolic')
        assert np.all(np.isfinite(R))
