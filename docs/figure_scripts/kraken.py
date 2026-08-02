"""Figures for ``docs/models/kraken.md``.

The code here is the code shown on the page: each builder is the worked
example, so a figure cannot drift from the snippet that claims to produce it.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np

from figure_scripts._common import (
    SQUARE, TALL, WIDE, deep_water, layered_elastic, shallow_water,
    sloping_shelf,
)

import uacpy
from uacpy.models import Kraken, RunMode
from uacpy.visualization.plots import plot_modes_heatmap


def mode_functions():
    """The first five depth eigenfunctions of the shallow-water channel."""
    env, source, _ = shallow_water()
    modes = Kraken().compute_modes(env, source)
    fig, ax = modes.plot(
        n_modes=5, figsize=TALL,
        title=f'Kraken — mode functions ψ$_m$(z), 200 Hz '
              f'({modes.n_modes} modes)')
    return fig


def mode_heatmap():
    """Every mode of the layered seabed, as a (depth, mode index) image."""
    env, source, _ = layered_elastic()
    modes = Kraken().compute_modes(env, source)
    fig, ax = plot_modes_heatmap(
        modes, figsize=TALL,
        title=f'Kraken — all {modes.n_modes} modes over an 8 m sand layer')
    ax.axhline(env.depth, color='k', linestyle='--', linewidth=1.2)
    ax.text(0.02, env.depth - 1.5, 'seafloor', transform=ax.get_yaxis_transform(),
            ha='left', va='bottom', fontsize=9)
    return fig


def tl_field():
    """Coherent TL over the shallow-water channel, with the seabed drawn."""
    env, source, receiver = shallow_water()
    tl = Kraken().run(env, source, receiver, run_mode=RunMode.COHERENT_TL)
    fig, ax = tl.plot(env=env, source=source, figsize=WIDE,
                      title='Kraken — coherent transmission loss, 200 Hz')
    return fig


def phase_speed_window():
    """What ``c_low`` and ``c_high`` cut out of the modal sum.

    Left: the phase-speed spectrum. Kraken keeps every mode whose phase
    speed falls in ``[c_low, c_high]``; raising the floor drops the
    best-trapped modes, lowering the ceiling drops the near-cutoff ones.
    Right: the TL those truncated mode sets produce along a 50 m track.
    """
    env, source, _ = shallow_water()
    line = uacpy.Receiver(depths=50.0, ranges=np.linspace(50.0, 5000.0, 400))
    cases = [
        ('default (0 – 1732 m/s)', {}, 'C0'),
        ('c_high=1600', {'c_high': 1600.0}, 'C1'),
        ('c_low=1520', {'c_low': 1520.0}, 'C2'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    c_water = float(np.min(env.ssp.to_pairs()[:, 1]))
    c_bottom = float(env.bottom.halfspace_at(range=0.0).sound_speed)
    for label, kwargs, colour in cases:
        model = Kraken(**kwargs)
        modes = model.compute_modes(env, source)
        axes[0].plot(np.arange(1, modes.n_modes + 1),
                     modes.compute_phase_speeds(), 'o-', color=colour,
                     markersize=4, label=f'{label} — {modes.n_modes} modes')
        tl = model.run(env, source, line)
        axes[1].plot(np.asarray(line.ranges) / 1000.0,
                     np.asarray(tl.tl, dtype=float).ravel(),
                     color=colour, linewidth=1.0, label=label)

    for c, name in ((c_water, f'min water speed ({c_water:.0f} m/s)'),
                    (c_bottom, f'bottom speed ({c_bottom:.0f} m/s)')):
        axes[0].axhline(c, color='0.4', linestyle='--', linewidth=1.0)
        axes[0].text(0.5, c + 6, name, fontsize=8, color='0.3')
    axes[0].set_xlabel('Mode index')
    axes[0].set_ylabel('Phase speed $c_p = \\omega/\\mathrm{Re}(k_m)$ (m/s)')
    axes[0].set_title('Phase-speed spectrum', fontweight='bold', fontsize=11)
    axes[0].legend(fontsize=8, loc='lower right')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Range (km)')
    axes[1].set_ylabel('TL (dB)')
    axes[1].set_title('Resulting TL at 50 m depth', fontweight='bold',
                      fontsize=11)
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Kraken — the phase-speed window bounds the mode set',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


def leaky_modes():
    """Trapped modes versus the leaky continuum ``leaky_modes=True`` admits."""
    env, source, _ = shallow_water()
    trapped = Kraken().compute_modes(env, source)
    leaky = Kraken(leaky_modes=True).compute_modes(env, source)
    c_bottom = float(env.bottom.halfspace_at(range=0.0).sound_speed)

    fig, ax = plt.subplots(figsize=SQUARE)
    for modes, label, marker, colour in (
            (leaky, f'leaky_modes=True — {leaky.n_modes} modes', 's', 'C1'),
            (trapped, f'default — {trapped.n_modes} modes', 'o', 'C0')):
        ax.plot(modes.compute_phase_speeds(), np.abs(np.imag(modes.k)),
                marker, color=colour, markersize=5, label=label)
    ax.axvline(c_bottom, color='0.4', linestyle='--', linewidth=1.0)
    ax.text(c_bottom * 1.02, 2e-5, f'bottom speed ({c_bottom:.0f} m/s)',
            fontsize=8, color='0.3', rotation=90)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Phase speed $c_p$ (m/s)')
    ax.set_ylabel('Modal attenuation $|\\mathrm{Im}(k_m)|$ (1/m)')
    ax.set_title('Kraken — trapped modes and the leaky tail, 200 Hz',
                 fontweight='bold', fontsize=12)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    return fig


def range_dependent():
    """Adiabatic versus coupled modes across the shelf break.

    The shared receiver grid reaches 395 m, which is below the seafloor for
    the inshore half of the track; Kraken warns per sample and returns the
    below-seafloor field there, so the warning is silenced here. The seabed
    overlay covers those samples in the plot.
    """
    env, source, receiver = sloping_shelf()
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.6), sharex=True, sharey=True)
    for ax, coupling in zip(axes, ('adiabatic', 'coupled')):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tl = Kraken(mode_coupling=coupling, n_segments=12).run(
                env, source, receiver)
        tl.plot(env=env, source=source, ax=ax,
                show_colorbar=(ax is axes[0]))
        ax.set_title(f"mode_coupling='{coupling}' — "
                     f"{tl.metadata['n_profiles']} range segments",
                     fontweight='bold', fontsize=11)
        if ax is not axes[-1]:
            ax.set_xlabel('')
    fig.suptitle('Kraken — range dependence by segmentation, 100 Hz',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


def deep_water_modes():
    """The Munk profile and the modes it traps, side by side."""
    env, source, _ = deep_water()
    modes = Kraken().compute_modes(env, source)
    pairs = env.ssp.to_pairs()
    axis_depth = float(pairs[np.argmin(pairs[:, 1]), 0])

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.6), sharey=True,
                             gridspec_kw={'width_ratios': [1.0, 2.2]})
    axes[0].plot(pairs[:, 1], pairs[:, 0], color='C0', linewidth=1.4)
    axes[0].axhline(axis_depth, color='C3', linestyle='--', linewidth=1.0)
    axes[0].set_xlabel('Sound speed (m/s)')
    axes[0].set_ylabel('Depth (m)')
    axes[0].set_title('Munk profile', fontweight='bold', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].invert_yaxis()

    plot_modes_heatmap(modes, n_modes=60, ax=axes[1],
                       title=f'First 60 of {modes.n_modes} modes @ 50 Hz')
    axes[1].axhline(axis_depth, color='C3', linestyle='--', linewidth=1.0)
    axes[1].set_title(axes[1].get_title(), fontweight='bold', fontsize=11)
    axes[1].set_ylabel('')
    fig.suptitle('Kraken — the deep sound channel confines the low-order modes',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


FIGURES = {
    'kraken_modes': mode_functions,
    'kraken_mode_heatmap': mode_heatmap,
    'kraken_tl': tl_field,
    'kraken_phase_speed_window': phase_speed_window,
    'kraken_leaky_modes': leaky_modes,
    'kraken_range_dependent': range_dependent,
    'kraken_deep_modes': deep_water_modes,
}
