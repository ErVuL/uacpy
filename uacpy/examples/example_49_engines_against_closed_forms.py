"""Three engines against closed-form fields — the agreement, measured.

uacpy.analytic returns textbook solutions as the same complex-pressure Field
an engine returns (Jensen et al., Computational Ocean Acoustics, 2nd ed.):
the Lloyd mirror (a point source under a pressure-release surface, §1.4.2.1)
and the ideal waveguide (pressure-release surface and bottom, §2.4.4). Bellhop
runs against the first, Kraken and Scooter against the second, and
uacpy.metrics states the agreement on the shared grid.

Uses: analytic.lloyd_mirror · analytic.ideal_waveguide · Bellhop(beam_type=) ·
Kraken · Scooter · metrics.tl_rmse · metrics.tl_max_error · plot.compare ·
plot.plot_field_difference
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy import analytic, metrics

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)
c = 1500.0                                      # isovelocity water, m/s

# ── Lloyd mirror: a deep bottom matched to the water (no reflection), so the
# field is the direct path and its surface image alone.
deep = uacpy.Environment(
    name='Lloyd mirror', water_density=1.0, bathymetry=3000.0, ssp=c,
    bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                    sound_speed=c, density=1.0,
                                    attenuation=0.0))
source = uacpy.Source(depths=18.0, frequencies=200.0)
line = uacpy.Receiver(depths=[18.0], ranges=np.linspace(200.0, 2000.0, 120))
mirror = analytic.lloyd_mirror(source, line, sound_speed=c)
# Hat beams (beam_type='G', the default) carry the geometric two-path sum.
# Gaussian beams (beam_type='B') floor each beam's width at pi*lambda, 23.6 m
# at 200 Hz, wider than the 18 m between the source and its surface image,
# and read about 2 dB too much loss here (docs/models/bellhop.md, the beam
# types): the floor that keeps a caustic finite costs accuracy elsewhere.
beams = {
    'Bellhop, hat beams': uacpy.Bellhop(
        backend='fortran', beam_type='G', n_beams=5001,
        launch_angles=(-20.0, 20.0)).run(deep, source, line),
    'Bellhop, Gaussian beams': uacpy.Bellhop(
        backend='fortran', beam_type='B', n_beams=5001,
        launch_angles=(-20.0, 20.0)).run(deep, source, line),
}

# ── Ideal waveguide, 200 m, pressure-release at both ends: an exact modal sum.
guide = uacpy.Environment(
    name='Ideal waveguide', water_density=1.0, bathymetry=200.0, ssp=c,
    bottom=uacpy.BoundaryProperties(acoustic_type='vacuum'))
source_25 = uacpy.Source(depths=100.0, frequencies=25.0)
grid = uacpy.Receiver(depths=np.linspace(5.0, 195.0, 39),
                      ranges=np.arange(300.0, 3301.0, 25.0))
modal = analytic.ideal_waveguide(guide, source_25, grid)
modes = {'Kraken': uacpy.Kraken().run(guide, source_25, grid),
         'Scooter': uacpy.Scooter().run(guide, source_25, grid)}

for name, field, reference in ([(n, f, mirror) for n, f in beams.items()]
                               + [(n, f, modal) for n, f in modes.items()]):
    diff = np.abs(field.dB - reference.dB)
    worst = np.unravel_index(np.nanargmax(diff), diff.shape)
    # The reference TL at the worst cell, against its median: far above it
    # is an interference null, where a null moved by a fraction of a cell
    # reads as many dB.
    print(f"  {name:24s} RMS {metrics.tl_rmse(field, reference):5.3f} dB  "
          f"median {np.nanmedian(diff):5.3f} dB  max "
          f"{metrics.tl_max_error(field, reference):5.2f} dB, where the "
          f"exact TL is {reference.dB[worst]:.1f} dB (median "
          f"{np.nanmedian(reference.dB):.1f} dB)")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
uacpy.plot.compare([mirror.at(depth=18.0)]
                   + [f.at(depth=18.0) for f in beams.values()],
                   labels=['Lloyd mirror (exact)', *beams], ax=axes[0],
                   title='Lloyd mirror, 200 Hz, source and receiver 18 m')
uacpy.plot.compare([modal.at(depth=30.0)]
                   + [f.at(depth=30.0) for f in modes.values()],
                   labels=['ideal waveguide (exact)', *modes], ax=axes[1],
                   title='Ideal waveguide, 25 Hz, receiver 30 m')
uacpy.plot.plot_field_difference(modes['Scooter'], modal, axes[2],
                                 diff_vmax=1.0,
                                 title='Scooter − ideal waveguide (dB)')
fig.tight_layout()
fig.savefig(OUT / 'example_49_engines_against_closed_forms.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)
print(f"wrote {OUT / 'example_49_engines_against_closed_forms.png'}")
