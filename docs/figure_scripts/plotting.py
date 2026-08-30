"""Figures for ``docs/guide/plotting.md``.

The code here is the code shown on the page: each builder is the worked
example, so a figure cannot drift from the snippet that claims to produce it.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np

from figure_scripts._common import deep_water, shallow_water

import uacpy
from uacpy.acoustic_signal import lfm_chirp, psd, spectrogram
from uacpy.comms import Modulator, awgn, constellation
from uacpy.models import Bellhop, Kraken, RunMode

GUIDE = True

CUT_DEPTH = 60.0        # m — the depth the range cut is taken at
CUT_RANGE = 3000.0      # m — the range the depth cut is taken at


def _time_series():
    """``p(depth, range, time)`` — the field every render branch is drawn from."""
    env, _, _ = shallow_water()
    source = uacpy.Source(depths=25.0,
                          frequencies=np.arange(150.0, 450.1, 0.5))
    receiver = uacpy.Receiver(depths=CUT_DEPTH,
                              ranges=np.linspace(1000.0, 3000.0, 9))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        H = Bellhop(n_beams=3000).run(env, source, receiver,
                                      run_mode=RunMode.BROADBAND)
    _, waveform = lfm_chirp(150.0, 450.0, 0.04, 4000.0)
    return H.synthesize_time_series(waveform, 4000.0)


def dispatch():
    """Carriers and results, one convention: every object has ``.plot()``."""
    env, source, receiver = shallow_water()
    tl = Bellhop(n_beams=3000).run(env, source, receiver).to_db()
    rays = Bellhop(n_beams=25, alpha=(-12.0, 12.0)).run(
        env, source, receiver, run_mode=RunMode.RAYS)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
    env.plot(ax=axes[0][0], title='env.plot()  —  Environment')
    env.ssp.plot(ax=axes[0][1], title='env.ssp.plot()  —  SoundSpeedProfile')
    tl.plot(env=env, ax=axes[1][0], title='tl.plot(env=env)  —  Field')
    rays.plot(env=env, ax=axes[1][1], show_receivers=False, show_legend=False,
              title='rays.plot(env=env)  —  Rays')
    for ax in axes.ravel():
        ax.title.set_fontsize(10)
    fig.suptitle('Carriers and results plot themselves — .plot() everywhere',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


def render_branches():
    """One Field, three render branches — chosen from ``coords`` alone.

    ``series`` is ``p(depth, range, time)``. Reducing it to two axes gives a
    heatmap, to one axis a line cut; ``stacked=True`` asks the same two-axis
    field for the waterfall instead.
    """
    series = _time_series()
    panel = series.isel(depth=0)                 # coords {range, time}

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    panel.plot(ax=axes[0],
               title="2 axes  →  heatmap")
    panel.at(range=CUT_RANGE).plot(ax=axes[1],
                                   title="1 axis  →  line cut")
    panel.plot(stacked=True, ax=axes[2],
               title="stacked=True  →  offset traces")
    for ax in axes:
        ax.title.set_fontsize(10)
    fig.suptitle('plot_field picks its branch from Field.coords',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


def heatmap_knobs():
    """The knobs only the 2-D heatmap branch reads."""
    env, source, receiver = deep_water()
    model = Bellhop(n_beams=3000)
    p = model.run(env, source, receiver)
    incoherent = model.run(env, source, receiver,
                           run_mode=RunMode.INCOHERENT_TL)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4))
    p.plot(env=env, ax=axes[0][0],
           title='default — fixed 20-120 dB TL scale')
    p.plot(env=env, ax=axes[0][1], vmin=70.0, vmax=110.0, cmap='viridis',
           title="vmin=70, vmax=110, cmap='viridis'")
    p.plot(env=env, ax=axes[1][0], value='phase',
           title="value='phase' — twilight, ±π")
    incoherent.plot(env=env, ax=axes[1][1], contours=(80.0, 90.0, 100.0),
                    title='contours=(80, 90, 100) — on the incoherent run')
    for ax in axes.ravel():
        ax.title.set_fontsize(10)
    fig.suptitle('vmin / vmax / cmap / contours / show_colorbar — '
                 'heatmap only', fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


def overlays():
    """What ``env=``, ``source=`` and ``receiver=`` add to a cross-section.

    The receiver grid stops at 60 m in a 100 m channel, so the bare plot spans
    only the data. ``env=`` extends the depth axis past the seafloor and draws
    it; ``source=`` puts the source back at ``r = 0``, off the data grid.
    """
    env, source, _ = shallow_water()
    receiver = uacpy.Receiver(depths=np.linspace(1.0, 60.0, 80),
                              ranges=np.linspace(50.0, 5000.0, 250))
    tl = Bellhop(n_beams=3000).run(env, source, receiver).to_db()

    fig, axes = plt.subplots(3, 1, figsize=(8.6, 9.0))
    tl.plot(ax=axes[0], show_colorbar=False,
            title='tl.plot()  —  depth axis spans the receiver grid')
    tl.plot(env=env, ax=axes[1], show_colorbar=False,
            title='tl.plot(env=env)  —  seafloor drawn, axis extended')
    tl.plot(env=env, source=source, receiver=receiver, ax=axes[2],
            show_colorbar=False,
            title='+ source=, receiver=  —  the run geometry')
    for ax in axes:
        ax.title.set_fontsize(10)
    fig.suptitle('A result carries no Environment — overlays are explicit',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    return fig


def tl_scale():
    """Why the TL scale is fixed: the same two fields, two scalings."""
    env_s, src_s, rcv_s = shallow_water()
    env_d, src_d, rcv_d = deep_water()
    shallow = Bellhop(n_beams=3000).run(env_s, src_s, rcv_s).to_db()
    deep = Bellhop(n_beams=3000).run(env_d, src_d, rcv_d).to_db()

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.6))
    for col, (field, env, name) in enumerate(
            [(shallow, env_s, 'Shallow, 200 Hz, 5 km'),
             (deep, env_d, 'Deep (Munk), 50 Hz, 100 km')]):
        field.plot(env=env, ax=axes[0][col],
                   title=f'{name} — default scale')
        lo, hi = np.nanpercentile(field.data, [2.0, 98.0])
        field.plot(env=env, ax=axes[1][col], vmin=lo, vmax=hi,
                   title=f'{name} — vmin/vmax per panel')
    for ax in axes.ravel():
        ax.title.set_fontsize(10)
    fig.suptitle('Top: one fixed 20-120 dB scale, the panels are comparable.  '
                 'Bottom: per-panel limits hide a ~28 dB level difference.',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    return fig


def composition():
    """A figure built by hand: carrier, result and overlay share one canvas."""
    env, source, receiver = shallow_water()
    bellhop = Bellhop(n_beams=3000).run(env, source, receiver).to_db()
    kraken = Kraken().run(env, source, receiver).to_db()

    fig = plt.figure(figsize=(11.0, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 2.2],
                          hspace=0.38, wspace=0.24)

    env.ssp.plot(ax=fig.add_subplot(gs[:, 0]), title='env.ssp.plot()')
    ax_tl = fig.add_subplot(gs[0, 1])
    bellhop.plot(env=env, source=source, ax=ax_tl, title='Bellhop TL')
    ax_tl.axhline(CUT_DEPTH, color='white', lw=1.2, ls='--')

    uacpy.plot.compare(
        [bellhop.at(depth=CUT_DEPTH), kraken.at(depth=CUT_DEPTH)],
        labels=['Bellhop', 'Kraken'], ax=fig.add_subplot(gs[1, 1]),
        title=f'compare() — TL at {CUT_DEPTH:g} m')
    for ax in fig.axes:
        ax.title.set_fontsize(10)
    fig.suptitle('ax= composes: every plotter draws into axes you own',
                 fontweight='bold', fontsize=13)
    return fig


def dsp_plotters():
    """The free plotters: arrays in, no uacpy object required."""
    rng = np.random.default_rng(0)
    sample_rate = 4000.0
    trace = _time_series().isel(depth=0).at(range=1000.0)
    # Model pressure is referenced to a unit source at 1 m; scale it to a
    # 170 dB re 1 µPa @ 1 m projector so the dB axes read physically.
    p_t = 316.0 * np.asarray(trace.data, dtype=float)

    f_s, t_s, S = spectrogram(p_t, sample_rate, nperseg=256)
    f_p, P = psd(p_t, sample_rate, nperseg=1024)

    mod = Modulator('16qam')
    bits = rng.integers(0, 2, size=4 * 600)
    symbols = awgn(mod.modulate(bits), 18.0, rng=rng)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    uacpy.plot.plot_spectrogram(f_s, t_s, S, ax=axes[0][0], vmin=30, vmax=85,
                                ymax=800.0,
                                title='plot_spectrogram — chirp at 1 km')
    uacpy.plot.plot_psd(f_p, P, ax=axes[0][1], ymin=0, ymax=80,
                        title='plot_psd — same trace')
    uacpy.plot.plot_constellation(constellation('16qam'), ax=axes[1][0],
                                  scheme='16qam',
                                  title='plot_constellation — 16-QAM map')
    uacpy.plot.plot_scatter(symbols, ax=axes[1][1],
                            ideal=constellation('16qam'),
                            title='plot_scatter — received, 18 dB SNR')
    for ax in axes.ravel():
        ax.title.set_fontsize(10)
    fig.suptitle('Free plotters take arrays — there is no object to hang '
                 '.plot() on', fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


FIGURES = {
    'plot_dispatch': dispatch,
    'plot_branches': render_branches,
    'plot_heatmap_knobs': heatmap_knobs,
    'plot_overlays': overlays,
    'plot_tl_scale': tl_scale,
    'plot_composition': composition,
    'plot_dsp': dsp_plotters,
}
