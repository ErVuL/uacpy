import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import welch  # noqa: E402
from uacpy.acoustic_signal.analysis import psd, ppsd, sel, PPSDResult  # noqa: E402
from uacpy.visualization.plots.signal import (  # noqa: E402
    plot_psd, plot_ppsd, plot_sel)


def test_psd_matches_welch():
    x = np.random.default_rng(0).standard_normal(48000) * 3.0
    f0, p0 = welch(x, 48000.0, window="hann", nperseg=8192, noverlap=4096,
                   scaling="density")
    f, p = psd(x, 48000.0)
    assert np.allclose(f, f0) and np.allclose(p, p0)
    fig, ax = plot_psd(f, p, label="x")
    assert ax.lines
    plt.close(fig)


def test_ppsd_function():
    x = np.random.default_rng(0).standard_normal(48000 * 4)
    r = ppsd(x, 48000.0, seg_duration=1.0)
    assert isinstance(r, PPSDResult)
    assert r.pdf.shape[1] == r.frequencies.size
    fig, ax = plot_ppsd(r)
    assert ax.collections
    plt.close(fig)


def test_sel_parseval():
    fs, T = 48000.0, 5.0
    t = np.arange(int(T * fs)) / fs
    x = 2.5 * np.sin(2 * np.pi * 1000.0 * t)
    s, bands = sel(x, fs)
    assert abs(np.sum(s) / (np.sum(x ** 2) / fs) - 1.0) < 0.01
    fig, ax = plot_sel(s, bands, duration=T)
    assert ax.patches
    plt.close(fig)
