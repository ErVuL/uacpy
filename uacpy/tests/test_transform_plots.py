import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from uacpy.acoustic_signal.transforms import (  # noqa: E402
    fk_transform, radon_transform, taup_transform)
from uacpy.visualization.plots.signal import (  # noqa: E402
    plot_fk, plot_radon, plot_taup)


def test_plot_fk_returns_fig_ax():
    f, k, p, _ = fk_transform(
        np.random.default_rng(0).standard_normal((128, 32)), 1000.0, 5.0)
    fig, ax = plot_fk(f, k, p, sound_speed=1500.0, title="t")
    assert fig is not None and ax.images
    plt.close(fig)
    fig2, ax2 = plt.subplots()
    _, ax3 = plot_fk(f, k, p, ax=ax2)
    assert ax3 is ax2
    plt.close(fig2)


def test_plot_radon():
    d = np.zeros((256, 24))
    d[60, :] = 1.0
    mo = np.linspace(-1.5e-3, 1.5e-3, 41)
    m, taus, R = radon_transform(d, 1000.0, 10.0, mo, kind="linear")
    fig, ax = plot_radon(m, taus, R, kind="linear")
    assert ax.images
    plt.close(fig)


def test_plot_taup():
    d = np.zeros((256, 24))
    d[60, :] = 1.0
    p, taus, tp = taup_transform(d, 1000.0, 10.0, n_slowness=41, p_max=1.5e-3)
    fig, ax = plot_taup(p, taus, tp, sound_speed=1500.0)
    assert ax.images
    plt.close(fig)
