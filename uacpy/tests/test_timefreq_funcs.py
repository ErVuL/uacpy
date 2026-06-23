import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import spectrogram as _scipy_spec  # noqa: E402
from uacpy.acoustic_signal.timefreq import (  # noqa: E402
    spectrogram, cwt, wigner_ville, cepstrum)
from uacpy.visualization.plots.signal import (  # noqa: E402
    plot_spectrogram, plot_cwt, plot_wigner_ville, plot_cepstrum)


def test_spectrogram_matches_scipy():
    x = np.random.default_rng(0).standard_normal(48000)
    # spectrogram now defaults noverlap=None (B2), so scipy derives nperseg//8;
    # the reference must use the same default to validate the pass-through.
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
    t, f, wv = wigner_ville(x[:256], fs)
    fig, ax = plot_wigner_ville(t, f, wv)
    plt.close(fig)
    c = cepstrum(x)
    fig, ax = plot_cepstrum(c, sample_rate=fs)
    assert ax.lines
    plt.close(fig)
