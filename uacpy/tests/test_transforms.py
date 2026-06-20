"""Tests for ``uacpy.acoustic_signal.transforms`` — tau-p / Radon transforms and
their inverses, plus the FK inverse round-trip.
"""

import numpy as np
import pytest

from uacpy.acoustic_signal import (
    FK,
    Radon,
    TauP,
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
        p, tau, U = TauP().compute(_linear_gather(p0, tau0), FS, DX,
                                   p_max=1 / 1000.0, n_slowness=241)
        i, j = np.unravel_index(np.argmax(np.abs(U)), U.shape)
        assert p[i] == pytest.approx(p0, abs=p[1] - p[0])
        assert tau[j] == pytest.approx(tau0, abs=2 / FS)

    def test_compute_then_standalone_inverse(self):
        g = _linear_gather(1 / 1500.0, 0.05)
        tp = TauP()
        slow, tau, U = tp.compute(g, FS, DX, p_max=1 / 1000.0, n_slowness=241)
        rec = inverse_taup(U, slow, FS, DX, NX)   # inverse the (returned) panel
        assert rec.shape == g.shape
        assert np.corrcoef(g.ravel(), rec.ravel())[0, 1] > 0.85

    def test_requires_2d(self):
        with pytest.raises(ConfigurationError):
            TauP().compute(np.zeros(10), FS, DX)

    def test_zero_pad_and_window_still_focus(self):
        p0, tau0 = 1 / 1500.0, 0.05
        p, tau, U = TauP().compute(_linear_gather(p0, tau0), FS, DX,
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
        assert m[np.argmax(np.abs(R)) // R.shape[1]] == pytest.approx(p0, abs=mv[1] - mv[0])

    def test_hyperbolic_peak_velocity(self):
        v0, tau0 = 1500.0, 0.08
        vels = np.linspace(1200, 2000, 81)
        m, tau, R = radon_transform(_hyperbolic_gather(v0, tau0), FS, DX, vels,
                                    kind='hyperbolic')
        i, j = np.unravel_index(np.argmax(np.abs(R)), R.shape)
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
        fk = FK()
        fk.compute(g, FS, DX)
        assert np.max(np.abs(inverse_fk(fk.FK) - g)) < 1e-9

    def test_calibrated_psd_parseval(self):
        # normalize=True -> 2-D PSD; Parseval: sum(PSD)*df*dk == mean(data^2)
        rng = np.random.default_rng(0)
        data = rng.standard_normal((256, 64))
        fk = FK()
        f, k, psd = fk.compute(data, FS, DX, normalize=True)
        lhs = psd.sum() * (f[1] - f[0]) * (k[1] - k[0])
        assert lhs == pytest.approx(np.mean(data ** 2), rel=1e-9)
        assert fk.normalized is True

    def test_normalize_default_off(self):
        data = _linear_gather(1 / 1500.0, 0.05)
        fk = FK()
        fk.compute(data, FS, DX)
        assert fk.normalized is False


class TestFKWindowing:
    def test_windowed_psd_parseval(self):
        # Calibrated PSD with a Hann taper stays Parseval-consistent against the
        # window-weighted mean power: sum(PSD)*df*dk == sum(x^2 w^2)/sum(w^2).
        from scipy.signal import get_window
        rng = np.random.default_rng(0)
        data = rng.standard_normal((256, 64))
        wt = get_window("hann", 256, fftbins=True)
        wx = get_window("hann", 64, fftbins=True)
        fk = FK()
        f, k, psd = fk.compute(data, FS, DX, normalize=True, window="hann")
        lhs = psd.sum() * (f[1] - f[0]) * (k[1] - k[0])
        expected = (np.sum(data ** 2 * wt[:, None] ** 2 * wx[None, :] ** 2)
                    / (np.sum(wt ** 2) * np.sum(wx ** 2)))
        assert lhs == pytest.approx(expected, rel=1e-9)

    def test_per_axis_window_list(self):
        data = _linear_gather(1 / 1500.0, 0.05)
        f, k, panel = FK().compute(data, FS, DX, window=["hann", ("kaiser", 8)])
        assert panel.shape == data.shape

    def test_bad_window_list_raises(self):
        with pytest.raises(ConfigurationError):
            FK().compute(np.zeros((32, 8)), FS, DX, window=["hann"])

    def test_zero_pad_grows_axes_pad_independent(self):
        rng = np.random.default_rng(1)
        data = rng.standard_normal((256, 64))
        fk = FK()
        f, k, psd = fk.compute(data, FS, DX, normalize=True, nfft=(512, 128))
        assert psd.shape == (512, 128)
        assert f.size == 512 and k.size == 128
        # Zeros add no power: Parseval still equals the true mean power.
        lhs = psd.sum() * (f[1] - f[0]) * (k[1] - k[0])
        assert lhs == pytest.approx(np.mean(data ** 2), rel=1e-9)

    def test_nfft_truncation_raises(self):
        with pytest.raises(ConfigurationError):
            FK().compute(np.zeros((256, 64)), FS, DX, nfft=(128, 32))

    def test_compute_requires_2d(self):
        with pytest.raises(ConfigurationError):
            FK().compute(np.zeros(10), FS, DX)


class TestRadonClass:
    def test_plot_does_not_mutate_panel(self):
        g = _linear_gather(1 / 1500.0, 0.05)
        r = Radon()
        mv = np.linspace(-1e-3, 1e-3, 201)
        r.compute(g, FS, DX, mv, kind='linear')
        panel = r.R.copy()
        fig, ax = r.plot()
        import matplotlib.pyplot as plt
        plt.close(fig)
        assert np.allclose(panel, r.R)  # plot took |R| on a copy; panel intact
        rec = inverse_radon(r.R, FS, DX, r.moveout, NX, kind='linear')
        assert np.corrcoef(g.ravel(), rec.ravel())[0, 1] > 0.85

    def test_plot_before_compute_raises(self):
        with pytest.raises(ConfigurationError):
            Radon().plot()


class TestBringYourOwn:
    """Inverse a transform panel obtained elsewhere (no prior compute())."""

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
