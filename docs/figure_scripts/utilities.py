"""Figures for ``docs/guide/utilities.md``.

The code here is the code shown on the page: each builder is the worked
example, so a figure cannot drift from the snippet that claims to produce it.

The cross-model metric figure runs on :func:`figure_scripts._common.shallow_water`
— the same 100 m channel every model page uses — so the numbers it prints are
comparable with the TL images on those pages.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from figure_scripts._common import WIDE, shallow_water

import uacpy
from uacpy.core import acoustics
from uacpy.metrics import tl_bias, tl_max_error, tl_rmse
from uacpy.models import Bellhop, Kraken, RunMode

# Write into docs/guide/figures/ rather than docs/models/figures/.
GUIDE = True


def material_presets():
    """The nine catalogue entries, one panel per geoacoustic property."""
    names = sorted(uacpy.list_materials(),
                   key=lambda n: uacpy.get_material(n)['sound_speed'])
    props = [('sound_speed', 'Compressional speed $c_p$ (m/s)', 'C0'),
             ('shear_speed', 'Shear speed $c_s$ (m/s)', 'C3'),
             ('density', r'Density $\rho$ (g/cm$^3$)', 'C2'),
             ('attenuation', r'Attenuation $\alpha_p$ (dB/$\lambda$)', 'C1')]

    fig, axes = plt.subplots(1, 4, figsize=(11.0, 4.0), sharey=True)
    y = np.arange(len(names))
    for ax, (key, label, colour) in zip(axes, props):
        values = [uacpy.get_material(n)[key] for n in names]
        ax.barh(y, values, color=colour, height=0.68)
        ax.set_xlabel(label, fontsize=9)
        ax.grid(axis='x', alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    axes[0].axvline(1500.0, color='gray', linestyle='--', linewidth=0.9)
    axes[0].text(1500.0, len(names) - 0.4, ' water', fontsize=8, color='gray',
                 va='top')
    fig.suptitle('uacpy.materials — the nine presets, ordered by $c_p$',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


def bottom_loss():
    """What the preset numbers buy you: reflection loss vs grazing angle."""
    fig, ax = plt.subplots(figsize=WIDE)
    for name in ['clay', 'silt', 'sand', 'gravel', 'limestone']:
        angles, loss_db = acoustics.bottom_loss_curve(name)
        ax.plot(angles, loss_db, linewidth=1.7, label=name)
        c_p = uacpy.get_material(name)['sound_speed']
        if c_p > 1500.0:
            ax.axvline(np.degrees(np.arccos(1500.0 / c_p)),
                       color='gray', linestyle=':', linewidth=0.8)
    ax.set_xlim(0.0, 90.0)
    ax.set_xlabel('Grazing angle (deg)')
    ax.set_ylabel('Bottom loss $-20\\log_{10}|R|$ (dB)')
    ax.set_title('Plane-wave bottom loss from the preset geoacoustics',
                 fontweight='bold', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(title='preset', fontsize=9)
    fig.tight_layout()
    return fig


def sound_speed_equations():
    """UNESCO vs Del Grosso vs Mackenzie, and where they disagree."""
    temperatures = np.linspace(0.0, 30.0, 121)
    pressures = np.linspace(0.0, 6000.0, 121)          # dbar ≈ metres

    unesco = acoustics.soundspeed_unesco(temperatures, 35.0, 0.0)
    delgrosso = acoustics.soundspeed_delgrosso(temperatures, 35.0, 0.0)
    mackenzie = acoustics.soundspeed(temperatures, 35.0, 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))

    ax = axes[0]
    ax.plot(temperatures, unesco, linewidth=1.8, label='UNESCO (Chen–Millero)')
    ax.plot(temperatures, delgrosso, linewidth=1.8, linestyle='--',
            label='Del Grosso (NRL II)')
    ax.plot(temperatures, mackenzie, linewidth=1.4, linestyle=':',
            label='Mackenzie')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Sound speed (m/s)')
    ax.set_title('S = 35 PSU, surface', fontsize=11)

    ax = axes[1]
    ax.axhline(0.0, color='C0', linewidth=1.8, label='UNESCO (reference)')
    ax.plot(temperatures, delgrosso - unesco, linewidth=1.8, linestyle='--',
            color='C1', label='Del Grosso − UNESCO')
    ax.plot(temperatures, mackenzie - unesco, linewidth=1.4, linestyle=':',
            color='C2', label='Mackenzie − UNESCO')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Difference (m/s)')
    ax.set_title('Residual at the surface', fontsize=11)

    ax = axes[2]
    T, P = np.meshgrid(temperatures, pressures)
    delta = (acoustics.soundspeed_delgrosso(T, 35.0, P)
             - acoustics.soundspeed_unesco(T, 35.0, P))
    limit = float(np.max(np.abs(delta)))
    mesh = ax.pcolormesh(temperatures, pressures, delta, cmap='RdBu_r',
                         vmin=-limit, vmax=limit, shading='auto')
    ax.invert_yaxis()
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Pressure (dbar)')
    ax.set_title('Del Grosso − UNESCO (m/s)', fontsize=11)
    fig.colorbar(mesh, ax=ax)

    for ax in axes[:2]:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle('uacpy.acoustics — the sound-speed equations compared',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


def cross_model_metrics():
    """Two models on one environment, scored with uacpy.metrics."""
    env, source, receiver = shallow_water()
    window = (1000.0, 5000.0)

    bellhop = Bellhop(n_beams=3000).run(env, source, receiver,
                                        run_mode=RunMode.COHERENT_TL)
    kraken = Kraken().run(env, source, receiver, run_mode=RunMode.COHERENT_TL)

    rmse = tl_rmse(bellhop, kraken, range_window=window)
    bias = tl_bias(bellhop, kraken, range_window=window)
    peak = tl_max_error(bellhop, kraken, range_window=window)

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 9.0), sharex=True, sharey=True)
    bellhop.plot(env=env, ax=axes[0], show_colorbar=True)
    axes[0].set_title('Bellhop — Gaussian beams', fontweight='bold', fontsize=11)
    kraken.plot(env=env, ax=axes[1], show_colorbar=True)
    axes[1].set_title('Kraken — normal modes', fontweight='bold', fontsize=11)

    difference = np.asarray(bellhop.db) - np.asarray(kraken.db)
    mesh = axes[2].pcolormesh(receiver.ranges / 1000.0, receiver.depths,
                              difference, cmap='RdBu_r', vmin=-20.0, vmax=20.0,
                              shading='auto')
    axes[2].set_xlabel('Range (km)')
    axes[2].set_ylabel('Depth (m)')
    axes[2].set_title('Bellhop − Kraken (dB)', fontweight='bold', fontsize=11)
    fig.colorbar(mesh, ax=axes[2], label='ΔTL (dB)')
    for ax in axes:
        ax.axvline(window[0] / 1000.0, color='k', linestyle='--', linewidth=1.0)
        if ax is not axes[-1]:
            ax.set_xlabel('')
    axes[2].text(
        0.99, 0.06,
        f'1–5 km:  RMSE {rmse:.1f} dB   bias {bias:+.1f} dB   max {peak:.1f} dB',
        transform=axes[2].transAxes, ha='right', va='bottom', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    fig.suptitle('uacpy.metrics — scoring one model against another',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


FIGURES = {
    'util_materials': material_presets,
    'util_bottom_loss': bottom_loss,
    'util_soundspeed': sound_speed_equations,
    'util_metrics': cross_model_metrics,
}
