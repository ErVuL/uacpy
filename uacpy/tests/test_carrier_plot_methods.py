"""Tests for the carrier ``.plot()`` convenience methods.

Harmonises the plotting API: every uacpy object that renders on its own has a
``.plot()`` that dispatches to its free ``plot_*`` function (mirroring
``Result.plot()``). ``Environment`` / ``SoundSpeedProfile`` plot with no extra
context; ``Absorption`` needs a frequency axis (it *is* a function of
frequency). ``Bottom`` is intentionally excluded — its geoacoustic section is an
environment-level view.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

import uacpy
from uacpy.core.absorption import FrancoisGarrison, Thorp


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close('all')


def _env():
    return uacpy.Environment(bathymetry=100.0, ssp=1500.0)


# ── Environment.plot ─────────────────────────────────────────────────────────

def test_environment_plot_returns_fig_ax():
    fig, ax = _env().plot()
    assert isinstance(fig, plt.Figure)


def test_environment_plot_forwards_kwargs():
    fig, ax = _env().plot(title='My env')
    assert ax.get_title() == 'My env'


# ── SoundSpeedProfile.plot ───────────────────────────────────────────────────

def test_ssp_plot_returns_fig_ax():
    ssp = _env().ssp
    fig, ax = ssp.plot()
    assert isinstance(fig, plt.Figure)
    # plot_ssp draws depth increasing downward → y-axis inverted.
    assert ax.yaxis_inverted()


# ── Absorption.plot ──────────────────────────────────────────────────────────

_FREQS = np.logspace(2, 4, 20)          # 100 Hz – 10 kHz


def test_thorp_plot_frequency_curve():
    fig, ax = Thorp().plot(_FREQS)
    assert ax.get_xlabel() == 'Frequency (Hz)'
    assert ax.get_ylabel() == 'Absorption (dB/km)'
    line = ax.lines[0]
    assert np.allclose(line.get_xdata(), _FREQS)
    assert np.all(line.get_ydata() > 0)


def test_absorption_plot_requires_frequencies():
    with pytest.raises(TypeError):
        Thorp().plot()


def test_francois_garrison_plot_depth_dependent():
    fg = FrancoisGarrison(temperature_c=10, salinity_psu=35, pH=8.0, z_bar_m=0)
    fig, ax = fg.plot(_FREQS, depth=1000.0)
    assert ax.get_xlabel() == 'Frequency (Hz)'
    assert np.all(ax.lines[0].get_ydata() > 0)


def test_absorption_plot_forwards_kwargs():
    fig, ax = Thorp().plot(_FREQS, title='α(f)')
    assert ax.get_title(loc='left') == 'α(f)'   # plot_absorption titles left


def test_biological_plot_outside_layer_warns():
    from uacpy.core.absorption import Biological
    bio = Biological(layers=[(40.0, 60.0, 1000.0, 5.0, 10.0)])
    # depth 0 m is outside the 40-60 m layer → α ≡ 0 → blank log-log axes.
    with pytest.warns(UserWarning, match='depth 0'):
        fig, ax = bio.plot(_FREQS, depth=0.0)


def test_biological_plot_inside_layer_no_warning():
    import warnings
    from uacpy.core.absorption import Biological
    bio = Biological(layers=[(40.0, 60.0, 1000.0, 5.0, 10.0)])
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        fig, ax = bio.plot(_FREQS, depth=50.0)
    assert np.all(ax.lines[0].get_ydata() > 0)


# ── Bathymetry / Altimetry .plot() ───────────────────────────────────────────

def test_bathymetry_plot_points_depth_downward():
    env = uacpy.Environment(bathymetry=[(0.0, 100.0), (5000.0, 150.0)],
                            ssp=1500.0)
    fig, ax = env.bathymetry.plot()
    assert ax.get_ylabel() == 'Depth (m)'
    assert ax.get_xlabel() == 'Range (km)'
    assert ax.yaxis_inverted()


def test_altimetry_plot_keeps_height_upward():
    env = uacpy.Environment(bathymetry=100.0, ssp=1500.0,
                            altimetry=[(0.0, 0.0), (5000.0, 1.5)])
    fig, ax = env.altimetry.plot()
    # The exact wording comes from the carrier's own ``_VALUE_LABEL``; what
    # matters here is that it is a height in metres and that the axis is NOT
    # inverted (altimetry is positive up, unlike bathymetry).
    assert ax.get_ylabel().lower().endswith('height (m)')
    assert not ax.yaxis_inverted()


def test_range_profile_plot_accepts_title_and_ax():
    env = uacpy.Environment(bathymetry=[(0.0, 100.0), (5000.0, 150.0)],
                            ssp=1500.0)
    fig, ax = plt.subplots()
    _, out = env.bathymetry.plot(ax=ax, title='Seafloor')
    assert out is ax and ax.get_title() == 'Seafloor'


def test_flat_bathymetry_plots_as_a_single_level():
    env = uacpy.Environment(bathymetry=100.0, ssp=1500.0)
    fig, ax = env.bathymetry.plot()
    assert ax.get_ylabel() == 'Depth (m)'
