"""RAM Padé-error grid optimizer (Lytaev 2023).

The two grid-selection modes of the RAM wrapper on a Pekeris waveguide: c₀
pinned to the water speed, against c₀ resolved from Lytaev Eq. (15), the value
that centres [ξ_min, ξ_max] and minimises the Padé error. Same accuracy budget
in both, so the only difference is c₀ — and it typically buys 2-3× coarser dr
for TL that agrees to a few dB.

Uses: uacpy.RAM(c0=) · RunMode.COHERENT_TL · Field.metadata (the chosen grid) ·
plot_field · plot.plot_field_difference
"""

import os
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

env = uacpy.Environment(
    name='pekeris-100hz',
    bathymetry=100.0,
    ssp=1500.0,
    bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                    sound_speed=1700.0, density=1.7,
                                    attenuation=0.5),
)
source = uacpy.Source(depths=25.0, frequencies=100.0)
# Ranges start at 300 m: closer in, the deepest receivers see direct and
# surface-reflected paths steeper than the band the PE propagates (28°).
receiver = uacpy.Receiver(depths=np.linspace(2.0, 98.0, 30),
                          ranges=np.linspace(300.0, 10000.0, 39))

# One accuracy budget, two reference speeds. The dispatcher picks mpiramS here
# (fluid seabed, flat surface). The budget sets dr, which is what this example
# compares; dz is then raised to RAM's depth-grid cost floor (about λ/20 here),
# where the predicted Padé error is about 0.99, far above the 1e-2 budget. A
# pinned budget the grid does not meet warns, so both runs record that warning
# and print it as a note: it is the expected outcome, not a fault.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    pinned = uacpy.RAM(accuracy=1e-2, angle_max=20.0, c0=1500.0).run(
        env, source, receiver, run_mode=uacpy.RunMode.COHERENT_TL)
    optimized = uacpy.RAM(accuracy=1e-2, angle_max=20.0).run(
        env, source, receiver, run_mode=uacpy.RunMode.COHERENT_TL)
for warning in caught:
    text = ' '.join(str(warning.message).split())
    if 'is not met on this grid' in text:
        print(f"  noted: {text.split(' Set dr/dz')[0]}")
    else:
        warnings.showwarning(warning.message, warning.category,
                             warning.filename, warning.lineno)

# The grid the Padé optimizer settled on comes back in the run settings.
for label, field in (('c₀=1500 pinned', pinned), ('c₀=Eq.(15)', optimized)):
    engine = field.run_settings.engine
    grid = engine.grids[0]
    print(f"  {label:16s} c₀={engine.c0:6.1f} m/s  "
          f"dr={grid.dr:7.2f} m  dz={grid.dz:6.3f} m")

rms = float(np.sqrt(np.nanmean((pinned.dB - optimized.dB) ** 2)))
print(f"  RMS |ΔTL| between the two grids: {rms:.2f} dB")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)
for ax, field, title in ((axes[0], pinned, 'c₀=1500 (pinned)\nasymmetric ξ-range'),
                         (axes[1], optimized, 'c₀=Eq.(15) (default)\ncentred ξ-range')):
    engine = field.run_settings.engine
    grid = engine.grids[0]
    uacpy.plot.plot_field(field, ax=ax, env=env, vmin=40, vmax=100,
                     title=f"{title}\nc₀={engine.c0:.1f} m/s, "
                           f"dr={grid.dr:.2f} m, dz={grid.dz:.3f} m")
# The signed residual, on a ±5 dB diverging window.
uacpy.plot.plot_field_difference(pinned, optimized, axes[2], diff_vmax=5,
                                 title=f'pinned − optimized, RMS {rms:.2f} dB')
fig.suptitle('RAM Padé-error grid optimizer (Lytaev) — Pekeris 100 Hz, '
             'src 25 m, 10 km', y=1.02)
fig.tight_layout()
fig.savefig(OUT / 'example_22_ram_lytaev_grid.png', dpi=120, bbox_inches='tight')
