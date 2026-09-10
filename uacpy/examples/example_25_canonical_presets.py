"""Canonical SSP shapes and bottom-loss curves.

Two catalogues side by side: the sound-speed profiles uacpy can build from a
factory — isovelocity, Munk, and one derived from T(z), S(z) through Mackenzie
— and the plane-wave bottom loss of the sediment and rock presets.

Granite is left out of the loss panel: its 5500 m/s would sit on top of
basalt's 5250 m/s. bottom_loss_curve ignores each preset's shear speed by
construction.

Uses: SoundSpeedProfile.from_isovelocity / from_munk / from_mackenzie ·
ssp.plot(ax=, label=, color=) · core.acoustics.bottom_loss_curve
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.core.acoustics import bottom_loss_curve

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

depths = np.linspace(0.0, 4000.0, 161)
temperature = 4.0 + 14.0 * np.exp(-depths / 400.0)
salinity = 35.0 - 0.5 * np.exp(-depths / 300.0)

fig, axes = plt.subplots(1, 2, figsize=(15, 8))

# Each profile draws itself; an explicit colour overlays them on one axes.
uacpy.SoundSpeedProfile.from_isovelocity(
    depth_max=4000.0, sound_speed=1500.0).plot(
        ax=axes[0], label='isovelocity', color='C0')
uacpy.SoundSpeedProfile.from_munk(depth_max=4000.0, n_points=81).plot(
    ax=axes[0], label='Munk', color='C1')
uacpy.SoundSpeedProfile.from_mackenzie(depths, temperature, salinity).plot(
    ax=axes[0], label='Mackenzie T,S', color='C2')
axes[0].set_title('Canonical SSP shapes')

for preset in ('clay', 'silt', 'sand', 'gravel', 'moraine',
               'chalk', 'limestone', 'basalt'):
    angles, loss_dB = bottom_loss_curve(preset)
    axes[1].plot(angles, loss_dB, label=preset, lw=1.5)
axes[1].set_xlabel('Grazing angle (°)')
axes[1].set_ylabel('Bottom loss (dB)')
axes[1].set_title('Plane-wave bottom loss')
axes[1].legend(loc='upper right', fontsize=9)
axes[1].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / 'example_25_canonical_presets.png', dpi=120)
plt.close(fig)
