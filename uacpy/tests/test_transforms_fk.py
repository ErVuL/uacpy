import numpy as np
import pytest
import uacpy
from uacpy.acoustic_signal.transforms import fk_transform, inverse_fk
from uacpy.core.exceptions import ConfigurationError


def _gather(seed=0):
    rng = np.random.default_rng(seed)
    nt, nx = 128, 32
    t = np.arange(nt)
    x = np.arange(nx)
    return (np.cos(2 * np.pi * (0.1 * t[:, None] - 0.05 * x[None, :]))
            + rng.standard_normal((nt, nx)))


def test_fk_single_segment_matches_direct_fft():
    d = _gather()
    fs, dx = 1000.0, 5.0
    nt, nx = d.shape
    FKc = np.fft.fftshift(np.fft.fft2(d), axes=(0, 1))
    p0 = np.abs(FKc) ** 2
    f0 = np.fft.fftshift(np.fft.fftfreq(nt, d=1.0 / fs))
    k0 = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    f, k, power, spectrum = fk_transform(d, fs=fs, dx=dx)
    assert np.allclose(f, f0) and np.allclose(k, k0)
    assert np.allclose(power, p0)
    assert spectrum is not None


def test_fk_averaging_reduces_variance():
    cv = []
    for kw in ({}, dict(nperseg=32, noverlap=0)):
        floors = []
        for s in range(8):
            _, _, p, _ = fk_transform(_gather(s), 1000.0, 5.0, **kw)
            floors.append(p[:6, :6])
        floors = np.array(floors)
        cv.append(floors.std() / floors.mean())
    assert cv[1] < cv[0] * 0.6


def test_fk_average_panel_not_invertible():
    _, _, _, spec = fk_transform(_gather(), 1000.0, 5.0, nperseg=32)
    assert spec is None


def test_inverse_fk_roundtrip_and_none_guard():
    d = _gather()
    _, _, _, spec = fk_transform(d, 1000.0, 5.0)
    rec = inverse_fk(spec)
    assert np.linalg.norm(rec - d) / np.linalg.norm(d) < 1e-10
    with pytest.raises(ConfigurationError):
        inverse_fk(None)


@pytest.mark.parametrize("kw", [dict(nperseg=999),
                                dict(nperseg=32, noverlap=32),
                                dict(nperseg=32, nfft=8)])
def test_fk_validation(kw):
    with pytest.raises(ConfigurationError):
        fk_transform(_gather(), 1000.0, 5.0, **kw)
