"""Smoke tests for the f-k / Radon / tau-p plotters, plus a documented-
signature + minimal-call sweep of the other ``plots.signal`` free plotters.

Each transform is fed a synthetic array-record and its plotter checked for an
image artist and for honouring ``ax=``. The transforms' own numerics live in
``test_transforms.py`` / ``test_transforms_fk.py``; nothing here asserts on
the values drawn.
"""

import inspect

import matplotlib.pyplot as plt
import numpy as np
import pytest
from uacpy.acoustic_signal.transforms import (
    fk_transform, radon_transform, taup_transform)
from uacpy.visualization.plots.signal import (
    plot_fk, plot_radon, plot_taup,
    plot_angular_spectrum, plot_band_levels, plot_coherence, plot_frf,
    plot_impulse_response_info)


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


_FREQS3 = np.array([10.0, 100.0, 1000.0])


@pytest.mark.parametrize('fn, params, args', [
    (plot_band_levels, ('centers', 'levels', 'ax'),
     (np.array([63.0, 80.0, 100.0]), np.array([90.0, 95.0, 92.0]))),
    (plot_angular_spectrum, ('angles_deg', 'spectrum', 'ax', 'db'),
     (np.linspace(-90.0, 90.0, 7), np.linspace(1.0, 2.0, 7))),
    (plot_frf, ('frequencies', 'tf', 'ax', 'tag'),
     (_FREQS3, np.array([1.0 + 1.0j, 2.0 + 0.0j, 0.5 - 0.5j]))),
    (plot_coherence, ('frequencies', 'coh', 'ax'),
     (_FREQS3, np.array([0.9, 0.95, 0.99]))),
    (plot_impulse_response_info, ('Minfo', 'Vinfo', 'g'),
     (np.eye(3), np.arange(3.0), np.linspace(0.0, 1.0, 4))),
])
def test_signal_plotter_signature_and_smoke(fn, params, args):
    # Each plotter exposes its documented parameters and draws from a
    # minimal call; plot_frf returns a 2-tuple of axes and
    # plot_impulse_response_info a 3-panel list, so axes are flattened.
    sig = inspect.signature(fn)
    assert all(p in sig.parameters for p in params)
    fig, axes = fn(*args)
    for ax in (axes if isinstance(axes, (list, tuple)) else [axes]):
        assert ax.has_data()
    plt.close(fig)


def test_plot_impulse_response_info_is_figure_level():
    # It builds its own three-panel figure and takes no ax=.
    assert 'ax' not in inspect.signature(plot_impulse_response_info).parameters
