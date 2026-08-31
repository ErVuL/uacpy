"""Figures for ``docs/guide/source-receiver.md``.

The code here is the code shown on the page: each builder is the worked
example, so a figure cannot drift from the snippet that claims to produce it.

This page is about the two geometry carriers, so every builder varies exactly
one thing about the geometry — the source depth, the source geometry, the
shape of the receiver grid — over water borrowed from
:mod:`figure_scripts._common`, so what changes in the picture is what changed
in the carrier.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np

from figure_scripts._common import (deep_water, shallow_water,
                                    source_beam_pattern)

import uacpy
from uacpy.models import Bellhop, Kraken, RAM, RunMode

# Write into docs/guide/figures/ rather than docs/models/figures/.
GUIDE = True


def geometry_to_grid():
    """The receiver lattice you draw is the grid the model fills in."""
    env, source, receiver = shallow_water()
    tl = Bellhop(n_beams=3000).run(env, source, receiver).to_db()

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0))
    env.plot(source=source, receiver=receiver, ax=axes[0],
             title='Geometry — source at 25 m, 100 × 250 receiver grid')
    tl.plot(env=env, source=source, ax=axes[1],
            title=f'Result — Field{tl.data.shape}, '
                  f'coords={tuple(tl.coords)}')
    fig.tight_layout()
    return fig


def source_depth():
    """One `Source`, three depths: a ResultStack and three different oceans."""
    env, _, _ = deep_water()
    source = uacpy.Source(depths=[100.0, 1300.0, 3000.0], frequencies=50.0)
    receiver = uacpy.Receiver(depths=np.linspace(0.0, 5000.0, 140),
                              ranges=np.linspace(2000.0, 100_000.0, 320))
    stack = Bellhop(n_beams=3000).run(env, source, receiver,
                                      run_mode=RunMode.INCOHERENT_TL)

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 9.5), sharex=True, sharey=True)
    for ax, (depth, slab) in zip(axes, stack):
        slab.plot(env=env, source=uacpy.Source(depths=depth, frequencies=50.0),
                  ax=ax, show_colorbar=(ax is axes[0]))
        ax.set_title(f'Source depth {depth:.0f} m',
                     fontweight='bold', fontsize=11)
        if ax is not axes[-1]:
            ax.set_xlabel('')
    fig.suptitle('Deep water, 50 Hz — only the source depth changed',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


def spreading():
    """point vs line vs scaled: what the source geometry costs you in dB."""
    env, _, _ = shallow_water()
    ranges = np.linspace(100.0, 5000.0, 400)
    receiver = uacpy.Receiver(depths=60.0, ranges=ranges)

    curves = {}
    for kind in ('point', 'line', 'scaled'):
        source = uacpy.Source(depths=25.0, frequencies=200.0, source_type=kind)
        curves[kind] = Kraken().run(env, source, receiver).to_db().data.ravel()

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.4), sharex=True)
    for kind in ('point', 'line', 'scaled'):
        axes[0].plot(ranges, curves[kind], lw=1.2,
                     label=f"source_type='{kind}'")
    axes[0].set_ylabel('Transmission loss (dB)')
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=9, loc='lower left')
    axes[0].set_title('Source geometry — Kraken, 200 Hz, receiver at 60 m',
                      fontweight='bold', fontsize=12)

    axes[1].plot(ranges, 10.0 * np.log10(ranges), color='0.4', lw=5.0,
                 alpha=0.35, solid_capstyle='butt', label=r'$10\log_{10} r$')
    axes[1].plot(ranges, curves['point'] - curves['scaled'], lw=1.4,
                 label='point − scaled')
    axes[1].plot(ranges, curves['point'] - curves['line'], lw=1.2,
                 label='point − line')
    axes[1].set_xlabel('Range (m)')
    axes[1].set_ylabel('Difference (dB)')
    axes[1].legend(fontsize=9, loc='lower right')
    axes[1].set_title('The whole difference is the cylindrical-spreading term',
                      fontweight='bold', fontsize=11)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def array_shapes():
    """A vertical array and a horizontal one, and the Fields they return."""
    env, source, _ = shallow_water()
    vla = uacpy.Receiver(depths=np.linspace(5.0, 95.0, 46), ranges=3000.0)
    hla = uacpy.Receiver(depths=60.0, ranges=np.linspace(200.0, 5000.0, 240))

    model = Bellhop(n_beams=3000)
    tl_v = model.run(env, source, vla).to_db()
    tl_h = model.run(env, source, hla).to_db()

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    tl_v.isel(range=0).plot(ax=axes[0])
    axes[0].set_title(f'46 depths × 1 range at 3 km — Field{tl_v.data.shape}\n'
                      '.isel(range=0) → coords=(depth,)',
                      fontweight='bold', fontsize=10)
    tl_h.isel(depth=0).plot(ax=axes[1])
    axes[1].set_title(f'1 depth at 60 m × 240 ranges — Field{tl_h.data.shape}\n'
                      '.isel(depth=0) → coords=(range,)',
                      fontweight='bold', fontsize=10)
    fig.tight_layout()
    return fig


def below_the_domain():
    """A receiver under the seabed is accepted; the answer is model-specific."""
    env, source, _ = shallow_water()
    receiver = uacpy.Receiver(depths=np.linspace(1.0, 160.0, 160),
                              ranges=3000.0)

    cuts = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for model in (Bellhop(n_beams=4000), Kraken(), RAM()):
            field = model.run(env, source, receiver).to_db()
            cuts[model.model_name] = field.isel(range=0).data

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    for name, tl in cuts.items():
        ax.plot(tl, receiver.depths, lw=1.3, label=name)
    ax.axhline(env.depth, color='saddlebrown', lw=2.0)
    ax.text(0.985, env.depth - 2.0, 'seabed, 100 m',
            transform=ax.get_yaxis_transform(), color='saddlebrown',
            fontsize=9, va='bottom', ha='right')
    ax.set_xlabel('Transmission loss (dB)')
    ax.set_ylabel('Receiver depth (m)')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title('Receivers below the seabed, 200 Hz at 3 km\n'
                 '(a source down there raises InvalidDepthError instead)',
                 fontweight='bold', fontsize=11)
    fig.tight_layout()
    return fig


def beam_pattern():
    """A directional source, staged as an .sbp and read by Bellhop."""
    env, _, receiver = shallow_water()
    # 30 deg wide between its -3 dB points, so the beam covers about what a
    # hand-drawn "+/-15 deg" table would, but with the skirt and the -13.3 dB
    # sidelobes a smooth main lobe implies rather than a cliff.
    pattern = source_beam_pattern(np.linspace(-90.0, 90.0, 721),
                                  beamwidth_deg=30.0)

    omni = uacpy.Source(depths=25.0, frequencies=200.0)
    beamed = uacpy.Source(depths=25.0, frequencies=200.0, beam_pattern=pattern)

    model = Bellhop(n_beams=3000)
    fig = plt.figure(figsize=(12.5, 6.6))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.0, 2.3))
    ax_pattern = fig.add_subplot(grid[:, 0], projection='polar')
    beamed.plot_beam_pattern(ax=ax_pattern, title='30° beam, aimed level')
    ax_pattern.title.set_fontsize(11)

    ax_omni = fig.add_subplot(grid[0, 1])
    ax_beam = fig.add_subplot(grid[1, 1], sharex=ax_omni, sharey=ax_omni)
    for ax, src, label in ((ax_omni, omni, 'beam_pattern=None (omni)'),
                           (ax_beam, beamed, 'beam_pattern=30° beam')):
        model.run(env, src, receiver).plot(
            env=env, source=src, ax=ax, show_colorbar=(ax is ax_omni))
        ax.set_title(label, fontweight='bold', fontsize=11)
    ax_omni.set_xlabel('')
    fig.suptitle('Source directivity — Bellhop, 200 Hz',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


FIGURES = {
    'srcrcv_geometry': geometry_to_grid,
    'srcrcv_source_depth': source_depth,
    'srcrcv_spreading': spreading,
    'srcrcv_arrays': array_shapes,
    'srcrcv_below_domain': below_the_domain,
    'srcrcv_beam_pattern': beam_pattern,
}
