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
    assert ax.get_xlabel() == 'Frequency [Hz]'
    assert ax.get_ylabel() == 'Absorption [dB/km]'
    line = ax.lines[0]
    assert np.allclose(line.get_xdata(), _FREQS)
    assert np.all(line.get_ydata() > 0)


def test_absorption_plot_requires_frequencies():
    with pytest.raises(TypeError):
        Thorp().plot()


def test_francois_garrison_plot_depth_dependent():
    fg = FrancoisGarrison(temperature_c=10, salinity_psu=35, pH=8.0, z_bar_m=0)
    fig, ax = fg.plot(_FREQS, depth=1000.0)
    assert ax.get_xlabel() == 'Frequency [Hz]'
    assert np.all(ax.lines[0].get_ydata() > 0)


def test_absorption_plot_forwards_kwargs():
    fig, ax = Thorp().plot(_FREQS, title='α(f)')
    assert ax.get_title(loc='left') == 'α(f)'   # plot_absorption titles left
