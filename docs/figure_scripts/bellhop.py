"""Figures for ``docs/models/bellhop.md``.

The code here is the code shown on the page: each builder is the worked
example, so a figure cannot drift from the snippet that claims to produce it.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np

from figure_scripts._common import SQUARE, WIDE, shallow_water

import uacpy
from uacpy.models import Bellhop, RunMode


def tl_field():
    """Coherent TL over the shallow-water channel, with the seabed drawn."""
    env, source, receiver = shallow_water()
    tl = Bellhop(n_beams=3000).run(
        env, source, receiver, run_mode=RunMode.COHERENT_TL)
    fig, ax = tl.plot(env=env, source=source, figsize=WIDE,
                      title='Bellhop — coherent transmission loss, 200 Hz')
    return fig


def ray_fan():
    """A ray fan coloured by boundary interaction."""
    env, source, receiver = shallow_water()
    rays = Bellhop(n_beams=41, alpha=(-20.0, 20.0)).run(
        env, source, receiver, run_mode=RunMode.RAYS)
    fig, ax = rays.plot(env=env, show_receivers=False, figsize=WIDE,
                        title='Bellhop — ray fan, alpha=(-20°, +20°)')
    # The seafloor fill (zorder 7) outranks the default legend zorder (5) and
    # paints over the lower legend rows where the two overlap.
    ax.get_legend().set_zorder(10)
    return fig


def eigenrays():
    """Only the rays that actually connect source to one receiver."""
    env, source, _ = shallow_water()
    point = uacpy.Receiver(depths=60.0, ranges=3000.0)
    eig = Bellhop(n_beams=4000, alpha=(-45.0, 45.0)).run(
        env, source, point, run_mode=RunMode.EIGENRAYS)
    fig, ax = eig.top_n_by_miss(12).plot(
        env=env, figsize=WIDE,
        title='Bellhop — 12 closest eigenrays to (3 km, 60 m)')
    return fig


def arrivals():
    """The impulse structure at that receiver: amplitude vs delay."""
    env, source, _ = shallow_water()
    point = uacpy.Receiver(depths=60.0, ranges=3000.0)
    arr = Bellhop(n_beams=4000, alpha=(-45.0, 45.0)).run(
        env, source, point, run_mode=RunMode.ARRIVALS)
    fig, ax = arr.plot(figsize=WIDE,
                       title='Bellhop — arrivals at (3 km, 60 m)')
    return fig


def coherence_modes():
    """Coherent vs incoherent vs semi-coherent, same run otherwise.

    Coherent keeps the interference nulls; incoherent sums intensities and
    smooths them away; semi-coherent is the compromise Bellhop offers.
    """
    env, source, receiver = shallow_water()
    modes = [(RunMode.COHERENT_TL, 'Coherent'),
             (RunMode.SEMICOHERENT_TL, 'Semi-coherent'),
             (RunMode.INCOHERENT_TL, 'Incoherent')]
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 9.0), sharex=True, sharey=True)
    for ax, (mode, label) in zip(axes, modes):
        field = Bellhop(n_beams=3000).run(env, source, receiver, run_mode=mode)
        field.plot(env=env, ax=ax, show_colorbar=(ax is axes[0]))
        ax.set_title(label, fontweight='bold', fontsize=11)
        if ax is not axes[-1]:
            ax.set_xlabel('')
    fig.suptitle('Bellhop — the three TL summation modes',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


def transfer_function():
    """Broadband H(f) and the impulse response it inverts to."""
    env, _, _ = shallow_water()
    source = uacpy.Source(depths=25.0,
                          frequencies=np.linspace(150.0, 450.0, 192))
    point = uacpy.Receiver(depths=60.0, ranges=3000.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        H = Bellhop(n_beams=3000).run(
            env, source, point, run_mode=RunMode.BROADBAND)
    fig, _ = H.plot_transfer_function(
        title='Bellhop — transfer function at (3 km, 60 m)')
    return fig


FIGURES = {
    'bellhop_tl': tl_field,
    'bellhop_rays': ray_fan,
    'bellhop_eigenrays': eigenrays,
    'bellhop_arrivals': arrivals,
    'bellhop_coherence': coherence_modes,
    'bellhop_transfer_function': transfer_function,
}
