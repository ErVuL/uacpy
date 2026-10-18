"""Canonical SSP shapes and bottom-loss curves.

Two catalogues side by side: the sound-speed profiles uacpy can build from a
factory — isovelocity, Munk, and one derived from T(z), S(z) through TEOS-10
(the default formula) — and the plane-wave bottom loss of the sediment and rock presets.

Granite is left out of the loss panel: its 5750 m/s would sit on top of
basalt's 5250 m/s. bottom_loss_curve ignores each preset's shear speed by
construction, and its plot rules each critical angle against the 1500 m/s
water. Clay
gets none: its 1500 m/s equals the water's, so no angle is critical and its
loss is set by the density contrast alone, (ρ − 1)/(ρ + 1) = 0.2, a flat
14 dB above a few degrees grazing.

Uses: SoundSpeedProfile.from_isovelocity / from_munk / from_temperature_salinity ·
ssp.plot(ax=, label=, color=) · acoustics.bottom_loss_curve
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import uacpy

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

depths = np.linspace(0.0, 4000.0, 161)
temperature = 4.0 + 14.0 * np.exp(-depths / 400.0)
salinity = 35.0 - 0.5 * np.exp(-depths / 300.0)

fig, axes = plt.subplots(1, 2, figsize=(15, 8))

# Each profile draws itself; an explicit colour overlays them on one axes.
uacpy.SoundSpeedProfile.from_isovelocity(depth_max=4000.0).plot(
    ax=axes[0], label='isovelocity (1500 m/s)', color='C0')
uacpy.SoundSpeedProfile.from_munk(depth_max=4000.0, n_points=81).plot(
    ax=axes[0], label='Munk', color='C1')
# formula=None is TEOS-10; 'mackenzie', 'unesco' and 'delgrosso' are the others.
uacpy.SoundSpeedProfile.from_temperature_salinity(
    depths, temperature, salinity).plot(
        ax=axes[0], label='TEOS-10 T,S', color='C2')
axes[0].set_title('Canonical SSP shapes')

for name in ['clay', 'silt', 'sand', 'gravel', 'moraine', 'chalk',
             'limestone', 'basalt']:
    uacpy.acoustics.bottom_loss_curve(name).plot(
        value='loss', ax=axes[1], label=name, mark_critical=True, lw=1.5,
        title='Plane-wave bottom loss (fluid seabed)')

fig.tight_layout()
fig.savefig(OUT / 'example_25_canonical_presets.png', dpi=120)
plt.close(fig)
