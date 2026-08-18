import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy.signal import get_window as _get_window  # noqa: E402
from scipy.signal import spectrogram as _scipy_spec  # noqa: E402
from uacpy.core.exceptions import ConfigurationError  # noqa: E402
from uacpy.acoustic_signal.timefreq import (  # noqa: E402
    spectrogram, cwt, inverse_cwt, wigner_ville, cepstrum, analytic_signal,
    envelope, instantaneous_frequency, _smoothing_window)
from uacpy.visualization.plots.signal import (  # noqa: E402
    plot_spectrogram, plot_cwt, plot_wigner_ville, plot_cepstrum)


def test_spectrogram_matches_scipy():
    x = np.random.default_rng(0).standard_normal(48000)
    # spectrogram passes noverlap=None straight through, letting scipy derive
    # nperseg//8; the reference must use the same default or it compares two
    # different overlaps.
    f0, t0, S0 = _scipy_spec(x, 48000.0, window="hann", nperseg=8192,
                             noverlap=None, scaling="density", mode="psd")
    f, t, S = spectrogram(x, 48000.0)
    assert np.allclose(f, f0) and np.allclose(S, S0)
    fig, ax = plot_spectrogram(f, t, S)
    assert ax.collections
    plt.close(fig)


def test_timefreq_plots():
    fs = 1000.0
    x = np.sin(2 * np.pi * 50 * np.arange(1024) / fs)
    fr, W = cwt(x, fs, np.linspace(20, 200, 40))
    fig, ax = plot_cwt(fr, W, fs)
    assert ax.collections
    plt.close(fig)
    f, t, wv = wigner_ville(x[:256], fs)
    fig, ax = plot_wigner_ville(f, t, wv)
    plt.close(fig)
    c = cepstrum(x)
    fig, ax = plot_cepstrum(c, sample_rate=fs)
    assert ax.lines
    plt.close(fig)

def test_even_array_freq_window_keeps_transform_real():
    """An even-length array `freq_window` keeps its centre sample on-centre,
    so acc(-tau) = conj(acc(tau)) holds and the imaginary part the transform
    discards is pure rounding error."""
    fs = 1000.0
    x = np.sin(2 * np.pi * 100 * np.arange(256) / fs)
    even = _get_window("hann", 4, fftbins=False)
    hv, Lh = _smoothing_window(even, "freq_window")
    assert hv.size % 2 == 1
    np.testing.assert_allclose(hv, hv[::-1])
    # Measure the imaginary part wigner_ville's `.real` discards by rebuilding
    # its lag kernel with the same window.
    z = analytic_signal(x)
    n = z.size
    num = den = 0.0
    for ti in range(n):
        taumax = min(ti, n - 1 - ti, Lh)
        taus = np.arange(-taumax, taumax + 1)
        kernel = np.zeros(n, dtype=complex)
        kernel[(taus + n) % n] = (z[ti + taus] * np.conj(z[ti - taus])
                                  * hv[Lh + taus])
        F = np.fft.fft(kernel)
        num += np.sum(np.imag(F) ** 2)
        den += np.sum(np.real(F) ** 2)
    assert np.sqrt(num / den) < 1e-12
    # And the public API accepts the even array, matching the explicit
    # centre-deleted odd window.
    Wa = wigner_ville(x, fs, freq_window=even)
    Wb = wigner_ville(x, fs, freq_window=np.delete(even, even.size // 2))
    np.testing.assert_allclose(Wa.distribution, Wb.distribution)


def test_smoothed_pseudo_wvd_matches_per_lag_reference():
    """The time-smoothed lag kernel equals a per-lag scalar-loop reference."""
    fs = 1000.0
    n = 96
    tt = np.arange(n) / fs
    x = np.sin(2 * np.pi * 80 * tt) + 0.5 * np.sin(2 * np.pi * 200 * tt)
    z = analytic_signal(x)
    hv, Lh = _smoothing_window(15, "freq_window")
    gv, Lg = _smoothing_window(7, "time_window")
    W_ref = np.zeros((n, n))
    for ti in range(n):
        taumax = min(ti, n - 1 - ti, Lh)
        taus = np.arange(-taumax, taumax + 1)
        acc = np.empty(taus.size, dtype=complex)
        for a, tau in enumerate(taus):
            mmax = min(Lg, ti - abs(tau), n - 1 - ti - abs(tau))
            ms = np.arange(-mmax, mmax + 1)
            gw = gv[Lg + ms]
            acc[a] = (np.sum(gw * z[ti + tau + ms]
                             * np.conj(z[ti - tau + ms])) / np.sum(gw))
        acc = acc * hv[Lh + taus]
        kernel = np.zeros(n, dtype=complex)
        kernel[(taus + n) % n] = acc
        W_ref[:, ti] = np.real(np.fft.fft(kernel))
    W = wigner_ville(x, fs, freq_window=15, time_window=7)
    np.testing.assert_allclose(W.distribution, W_ref, atol=1e-12)


def test_cwt_rejects_complex_input():
    xc = np.exp(2j * np.pi * 0.1 * np.arange(512))
    with pytest.raises(ConfigurationError, match="complex"):
        cwt(xc, 1000.0)


def test_cwt_explicit_frequencies_are_not_nyquist_checked():
    """An explicit ``frequencies=`` array is analysed as given — only the
    default grid is capped at fs/2 — so requesting 700 Hz at fs=1000 returns
    coefficients, not an error, and what comes back is numerical residue far
    below any in-band coefficient."""
    fs = 1000.0
    x = np.sin(2 * np.pi * 50 * np.arange(128) / fs)
    r = cwt(x, fs, frequencies=[50.0, 700.0])
    assert r.coefficients.shape == (2, 128)
    np.testing.assert_allclose(r.frequencies, [50.0, 700.0])
    assert np.isfinite(r.coefficients).all()
    assert (np.abs(r.coefficients[1]).max()
            < 0.02 * np.abs(r.coefficients[0]).max())


def test_analytic_signal_and_cepstrum_reject_complex_input():
    xc = np.exp(2j * np.pi * 0.05 * np.arange(256))
    with pytest.raises(ConfigurationError, match="complex"):
        analytic_signal(xc)
    with pytest.raises(ConfigurationError, match="complex"):
        cepstrum(xc)


def test_spectrogram_passes_scipy_nperseg_clamp_warning_through():
    """A 512-sample signal against the default nperseg=8192: scipy clamps
    nperseg to the input length, its UserWarning reaches the caller, and the
    result is a single frame."""
    x = np.sin(2 * np.pi * 50 * np.arange(512) / 1000.0)
    with pytest.warns(UserWarning, match="nperseg"):
        r = spectrogram(x, 1000.0)
    assert r.times.size == 1


def test_two_tone_instantaneous_frequency_is_the_mean():
    """Two equal-amplitude tones at f1 and f2: the analytic-signal
    instantaneous frequency reads (f1+f2)/2 — 115 Hz for 100 and 130 Hz, a
    frequency not present in the signal. The trace holds that value wherever
    the beat envelope is away from its nulls; at each null the sampled phase
    steps by ~ -pi, so the naive full-record mean lands near f1 instead."""
    fs, f1, f2 = 2000.0, 100.0, 130.0
    t = np.arange(int(fs)) / fs
    x = np.cos(2 * np.pi * f1 * t) + np.cos(2 * np.pi * f2 * t)
    fi = instantaneous_frequency(x, fs)
    env = envelope(x)
    core = slice(100, -100)
    assert np.median(fi[core]) == pytest.approx((f1 + f2) / 2, abs=1e-6)
    kept = fi[core][env[core] > 0.25 * env[core].max()]
    assert kept.mean() == pytest.approx((f1 + f2) / 2, abs=1e-3)
    assert np.abs(kept - (f1 + f2) / 2).max() < 1e-6
    assert fi[core].mean() == pytest.approx(f1, abs=2.0)


def test_cwt_and_spectrogram_reject_nonpositive_sample_rate():
    x = np.ones(512)
    with pytest.raises(ConfigurationError, match="sample_rate"):
        cwt(x, 0.0)
    with pytest.raises(ConfigurationError, match="sample_rate"):
        spectrogram(x, 0.0)


def test_inverse_cwt_warns_off_log2_uniform_scale_grid():
    fs = 1000.0
    x = np.sin(2 * np.pi * 50 * np.arange(2048) / fs)
    lin = cwt(x, fs, np.linspace(10, 400, 64))
    with pytest.warns(UserWarning, match="log2"):
        inverse_cwt(lin.coefficients, lin.frequencies, fs)
    single = cwt(x, fs, np.array([50.0]))
    with pytest.warns(UserWarning, match="single scale"):
        inverse_cwt(single.coefficients, single.frequencies, fs)
    # cwt's default log-spaced grid reconstructs silently at the right level.
    log = cwt(x, fs)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        xr = inverse_cwt(log.coefficients, log.frequencies, fs)
    mid = slice(256, 1792)
    assert abs(np.std(xr[mid]) / np.std(x[mid]) - 1.0) < 0.05
