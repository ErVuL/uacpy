"""Basic shallow-water propagation — the Pekeris waveguide.

The shortest complete uacpy run: build an Environment, place a Source and a
Receiver grid, run Bellhop for coherent transmission loss, and plot the field
with two cuts through it. Start here.

Uses: uacpy.Environment · Source · Receiver · Bellhop.run · plot_field ·
Field.at().plot() · env.plot()
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

# A Pekeris waveguide: isovelocity water over a faster fluid half-space. The
# classic shallow-water benchmark, and the simplest environment uacpy takes.
env = uacpy.Environment(
    name="Pekeris waveguide",
    bathymetry=100.0,                   # flat seafloor, m
    ssp=1500.0,                         # isovelocity water column, m/s
    bottom=uacpy.BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1600.0,             # sediment, m/s — faster than the water
        density=1.5,                    # g/cm³
        attenuation=0.5,                # dB/wavelength
    ),
)

source = uacpy.Source(depths=50.0, frequencies=100.0)
receiver = uacpy.Receiver(depths=np.linspace(5, 95, 50),
                          ranges=np.linspace(100, 10000, 100))

# 300 Gaussian beams over ±80°, enough to fill a 100 m duct out to 10 km.
tl = uacpy.Bellhop(beam_type='B', n_beams=300, alpha=(-80, 80)).run(
    env, source, receiver, run_mode=uacpy.RunMode.COHERENT_TL)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# The TL field on the fixed 20–120 dB scale every uacpy figure shares, with
# the jet_r colormap of the Acoustics Toolbox: red is loud, blue is quiet.
uacpy.plot_field(tl, ax=axes[0, 0], env=env)
# A Field sliced to one axis plots itself as a line cut.
tl.at(depth=50.0).plot(ax=axes[0, 1], color='b',
                       title='TL vs range at source depth')
tl.at(range=5000.0).plot(ax=axes[1, 0], color='r',
                         title='TL vs depth at 5 km')
env.plot(ax=axes[1, 1], source=source, receiver=receiver,
         title='Environment')
fig.tight_layout()
fig.savefig(OUT / 'example_01_basic_shallow_water.png', dpi=150,
            bbox_inches='tight')

print(f"TL spans {np.nanmin(tl.dB):.1f} to {np.nanmax(tl.dB):.1f} dB "
      f"over {tl.ranges[-1] / 1000:.0f} km")
print(f"wrote {OUT / 'example_01_basic_shallow_water.png'}")
