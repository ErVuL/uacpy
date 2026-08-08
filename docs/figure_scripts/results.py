"""Figures for ``docs/guide/results.md``.

The code here is the code shown on the page: each builder is the worked
example, so a figure cannot drift from the snippet that claims to produce it.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np

from figure_scripts._common import WIDE, layered_elastic, shallow_water

import uacpy
from uacpy.acoustic_signal import lfm_chirp
from uacpy.models import Bellhop, Bounce, Kraken, RunMode

GUIDE = True

CUT_DEPTH = 60.0        # m — the depth the range cut is taken at
CUT_RANGE = 3000.0      # m — the range the depth cut is taken at


def _broadband_point():
    """``H(f)`` at one receiver cell — the source of every broadband figure."""
    env, _, _ = shallow_water()
    source = uacpy.Source(depths=25.0,
                          frequencies=np.linspace(150.0, 450.0, 192))
    point = uacpy.Receiver(depths=CUT_DEPTH, ranges=CUT_RANGE)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Bellhop(n_beams=3000).run(env, source, point,
                                         run_mode=RunMode.BROADBAND)


def field_kinds():
    """One container, four meanings — ``.kind`` derived from (dtype, coords).

    Every panel is a ``Field``. Nothing declares what it holds: the dtype of
    ``.data`` and the keys of ``.coords`` decide.
    """
    env, source, receiver = shallow_water()
    pressure = Bellhop(n_beams=3000).run(env, source, receiver)
    H = _broadband_point()
    spectrum = H.isel(depth=0, range=0)
    trace = H.to_time_trace()

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4))
    pressure.to_tl().plot(
        env=env, ax=axes[0][0], show_colorbar=False,
        title="real + {depth, range}  →  .kind = 'tl'")
    pressure.plot(
        env=env, ax=axes[0][1], value='phase', show_colorbar=False,
        title="complex + {depth, range}  →  .kind = 'pressure'")
    spectrum.plot(
        ax=axes[1][0], value='mag_db',
        title="complex + {frequency}  →  .kind = 'transfer_function'")
    trace.plot(
        ax=axes[1][1],
        title="real + {time}  →  .kind = 'time_series'")
    for ax in axes.ravel():
        ax.title.set_fontsize(10)
    fig.suptitle('One Field class — the kind is derived, never declared',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


def field_slicing():
    """The same Field three ways: heatmap, range cut, depth cut.

    ``at()`` picks the nearest stored sample, drops that axis from ``coords``
    and records its value in ``pinned`` — which is what the cut panels are
    titled with (no title was passed).
    """
    env, source, receiver = shallow_water()
    tl = Bellhop(n_beams=3000).run(env, source, receiver).to_tl()
    loudest = tl.max()

    fig = plt.figure(figsize=(10.0, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0],
                          hspace=0.34, wspace=0.26)

    ax_map = fig.add_subplot(gs[0, :])
    tl.plot(env=env, source=source, ax=ax_map,
            title='tl  —  coords = {depth, range}')
    ax_map.axhline(CUT_DEPTH, color='white', lw=1.2, ls='--')
    ax_map.axvline(CUT_RANGE / 1000.0, color='white', lw=1.2, ls='--')
    ax_map.plot(loudest.pinned['range'] / 1000.0, loudest.pinned['depth'],
                marker='*', markersize=13, color='white',
                markeredgecolor='black', linestyle='none',
                label='tl.max()')
    ax_map.legend(loc='lower right', fontsize=8)

    ax_r = fig.add_subplot(gs[1, 0])
    tl.at(depth=CUT_DEPTH).plot(ax=ax_r)
    ax_d = fig.add_subplot(gs[1, 1])
    tl.at(range=CUT_RANGE).plot(ax=ax_d)
    for ax in (ax_map, ax_r, ax_d):
        ax.title.set_fontsize(10)

    fig.suptitle('at() collapses an axis into pinned — the cut titles are '
                 'field.pinned', fontweight='bold', fontsize=12)
    return fig


def stack_panels():
    """A multi-source run returns a ResultStack, one Field slab per depth."""
    env, _, receiver = shallow_water()
    source = uacpy.Source(depths=[15.0, 50.0, 85.0], frequencies=200.0)
    stack = Bellhop(n_beams=3000).run(env, source, receiver)
    fig, _ = stack.plot(
        env=env, figsize=(12.5, 3.8),
        title='ResultStack[Field] — one slab per source depth')
    fig.tight_layout()   # re-fit: the panel plotter sets its suptitle last
    return fig


def time_synthesis():
    """``synthesize_time_series`` — H(f) convolved with a source chirp.

    The frequency grid sets the record length (1/Δf), so it is chosen fine
    enough to hold the travel-time spread across the receiver ranges.
    """
    env, _, _ = shallow_water()
    source = uacpy.Source(depths=25.0,
                          frequencies=np.arange(150.0, 450.1, 0.5))
    receiver = uacpy.Receiver(depths=CUT_DEPTH,
                              ranges=np.linspace(1000.0, 3000.0, 9))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        H = Bellhop(n_beams=3000).run(env, source, receiver,
                                      run_mode=RunMode.BROADBAND)

    sample_rate = 4000.0
    t_src, waveform = lfm_chirp(150.0, 450.0, 0.04, sample_rate)
    series = H.synthesize_time_series(waveform, sample_rate)

    fig, (ax_src, ax_rx) = plt.subplots(
        1, 2, figsize=(11.0, 4.2), gridspec_kw={'width_ratios': [1.0, 2.0]})
    ax_src.plot(t_src * 1000.0, waveform, lw=0.8, color='C1')
    ax_src.set_xlabel('Time (ms)')
    ax_src.set_ylabel('Amplitude')
    ax_src.set_title('source: 150→450 Hz LFM chirp, 40 ms', fontsize=10)
    ax_src.grid(True, alpha=0.3)

    series.isel(depth=0).plot(ax=ax_rx, stacked=True)
    ax_rx.set_ylabel('Range (m), stacked')
    ax_rx.set_title('received p(t) — one trace per range', fontsize=10)

    fig.suptitle('H(f).synthesize_time_series(waveform, sample_rate)',
                 fontweight='bold', fontsize=12)
    fig.tight_layout()
    return fig


def rays_and_arrivals():
    """The two sparse result types: the geometry, and what it delivers."""
    env, source, _ = shallow_water()
    point = uacpy.Receiver(depths=CUT_DEPTH, ranges=CUT_RANGE)
    model = Bellhop(n_beams=4000, alpha=(-45.0, 45.0))
    eig = model.run(env, source, point, run_mode=RunMode.EIGENRAYS)
    arr = model.run(env, source, point, run_mode=RunMode.ARRIVALS)

    fig, (ax_rays, ax_arr) = plt.subplots(2, 1, figsize=(9.0, 7.0))
    eig.top_n_by_miss(12).plot(
        env=env, ax=ax_rays,
        title='Rays — 12 closest eigenrays to (3 km, 60 m)')
    arr.plot(ax=ax_arr,
             title='Arrivals — amplitude and delay of every path')
    for ax in (ax_rays, ax_arr):
        ax.title.set_fontsize(10)
    fig.tight_layout()
    return fig


def modes_and_reflection():
    """Two more result types, from two other models."""
    env, source, receiver = shallow_water()
    modes = Kraken().run(env, source, receiver, run_mode=RunMode.MODES)

    env_el, source_el, receiver_el = layered_elastic()
    refl = Bounce().run(env_el, source_el, receiver_el,
                        run_mode=RunMode.REFLECTION)

    fig, (ax_m, ax_r) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    modes.plot(n_modes=6, ax=ax_m,
               title='Modes — Kraken depth eigenfunctions ψ(z)')
    refl.plot(ax=ax_r, show_phase=True,
              title='ReflectionCoefficient — Bounce |R(θ)|')
    for ax in (ax_m, ax_r):
        ax.title.set_fontsize(10)
    fig.tight_layout()
    return fig


FIGURES = {
    'results_field_kinds': field_kinds,
    'results_slicing': field_slicing,
    'results_stack': stack_panels,
    'results_time_synthesis': time_synthesis,
    'results_rays_arrivals': rays_and_arrivals,
    'results_modes_reflection': modes_and_reflection,
}
