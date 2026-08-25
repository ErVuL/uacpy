import numpy as np
import pytest
from uacpy.acoustic_signal.transforms import fk_transform, inverse_fk
from uacpy.core.exceptions import ConfigurationError


def _gather(seed=0):
    rng = np.random.default_rng(seed)
    nt, nx = 128, 32
    t = np.arange(nt)
    x = np.arange(nx)
    return (np.cos(2 * np.pi * (0.1 * t[:, None] - 0.05 * x[None, :]))
            + rng.standard_normal((nt, nx)))


def _fft2_on_fk_convention(d, shape=None):
    """``np.fft.fft2`` re-indexed onto ``fk_transform``'s wavenumber sign.

    ``fft2``'s ``exp(-i2πνx)`` kernel puts a +x-travelling wave on ``ω = -c·k``;
    ``fk_transform`` reports the package-wide ``k = ω/c``, so its spatial axis
    is the negative of ``fft2``'s — column ``ν`` holds ``-ν``.
    """
    F = np.fft.fft2(d, s=shape)
    return np.fft.fftshift(np.roll(F[:, ::-1], 1, axis=1), axes=(0, 1))


def test_fk_single_segment_matches_direct_fft():
    d = _gather()
    fs, dx = 1000.0, 5.0
    nt, nx = d.shape
    FKc = _fft2_on_fk_convention(d)
    p0 = np.abs(FKc) ** 2
    f0 = np.fft.fftshift(np.fft.fftfreq(nt, d=1.0 / fs))
    # Angular wavenumber k = 2π·ν rad/m (fk_transform's convention; ν = fftfreq).
    k0 = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    f, k, power, spectrum = fk_transform(d, sample_rate=fs, dx=dx)
    assert np.allclose(f, f0) and np.allclose(k, k0)
    assert np.allclose(power, p0)
    assert np.allclose(spectrum, FKc)
    assert spectrum is not None


def test_fk_averaging_reduces_variance():
    # nperseg=32 with no overlap splits the 128-sample record into 4
    # independent segments, so Welch averaging should cut the coefficient of
    # variation of the noise floor by 1/sqrt(4) = 0.5. The 0.6 bound is that
    # theoretical factor with slack for the finite 8-seed sample.
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
    # A single-segment forward/inverse FFT pair is algebraically exact, so
    # 1e-10 is float round-off headroom, not a physical tolerance.
    assert np.linalg.norm(rec - d) / np.linalg.norm(d) < 1e-10
    with pytest.raises(ConfigurationError):
        inverse_fk(None)


@pytest.mark.parametrize("kw", [dict(nperseg=999),
                                dict(nperseg=32, noverlap=32),
                                dict(nperseg=32, nfft=8)])
def test_fk_validation(kw):
    with pytest.raises(ConfigurationError):
        fk_transform(_gather(), 1000.0, 5.0, **kw)


@pytest.mark.parametrize("direction", [+1, -1])
def test_fk_wavenumber_sign_is_omega_over_c(direction):
    """A plane wave of speed ``c`` towards ``+x`` peaks at ``k = +ω/c``.

    This is the ``k = ω/c`` convention the docstring states and the models use,
    and the sign directional f-k muting depends on; raw ``np.fft.fft2`` would
    put the same wave on ``ω = -c·k``.
    """
    fs, dx, nt, nx, c, f0 = 200.0, 2.0, 256, 64, 1000.0, 25.0
    t = np.arange(nt) / fs
    x = np.arange(nx) * dx
    d = np.sin(2 * np.pi * f0 * (t[:, None] - direction * x[None, :] / c))
    f, k, power, _ = fk_transform(d, fs, dx)
    pos = f > 0
    i, j = np.unravel_index(np.argmax(power[pos]), power[pos].shape)
    assert f[pos][i] == pytest.approx(f0, abs=f[1] - f[0])
    # Half a wavenumber bin of slack: the k grid steps by 2π/(nx·dx) =
    # 0.049 rad/m and the event sits at |k| = 2π·25/1000 = 0.157 rad/m.
    assert k[j] == pytest.approx(direction * 2 * np.pi * f0 / c,
                                 abs=0.5 * (k[1] - k[0]))
    assert np.sign(k[j]) == direction


def test_fk_wavenumber_sign_agrees_with_taup_and_radon():
    """The three gather transforms report the same sign for one event.

    ``taup_transform``/``radon_transform`` give a +x-travelling wave the
    slowness ``p = +1/c``; ``fk_transform`` must therefore place it at
    ``k = +ω/c``, not at ``-ω/c``.
    """
    from uacpy.acoustic_signal.transforms import (radon_transform,
                                                  taup_transform)
    fs, dx, nt, nx, c, f0 = 200.0, 2.0, 256, 64, 1000.0, 25.0
    t = np.arange(nt) / fs
    x = np.arange(nx) * dx
    d = np.sin(2 * np.pi * f0 * (t[:, None] - x[None, :] / c))

    f, k, power, _ = fk_transform(d, fs, dx)
    pos = f > 0
    _, j = np.unravel_index(np.argmax(power[pos]), power[pos].shape)
    assert k[j] > 0

    tp = taup_transform(d, fs, dx)
    assert tp.slownesses[np.argmax(np.abs(tp.panel).max(axis=1))] > 0
    ps = np.linspace(-2e-3, 2e-3, 201)
    rd = radon_transform(d, fs, dx, ps, kind="linear")
    assert rd.moveout[np.argmax(np.abs(rd.panel).max(axis=1))] > 0


@pytest.mark.parametrize("nx", [31, 32])
@pytest.mark.parametrize("pad", [False, True])
def test_inverse_fk_undoes_the_wavenumber_flip(nx, pad):
    """The spatial re-indexing is its own inverse, for odd and even ``nx``
    and with zero padding, so the round trip stays exact."""
    rng = np.random.default_rng(4)
    nt = 33
    d = rng.standard_normal((nt, nx))
    nfft = (nt + 7, nx + 5) if pad else None
    _, _, _, spec = fk_transform(d, 500.0, 2.0, nfft=nfft)
    rec = inverse_fk(spec)
    assert np.max(np.abs(rec[:nt, :nx] - d)) < 1e-10
