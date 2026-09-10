"""Deep-water long range — the SOFAR channel out to 150 km.

A 20 Hz source at 1000 m in a 5500 m Munk profile, whose channel axis sits at
1300 m, run to 150 km by four models. Low frequency and a deep sound channel
are what buy the range: energy is trapped by refraction, meets the abyssal
bottom rarely, and re-focuses into convergence zones along the way.

Uses: SoundSpeedProfile.from_munk · four models on one .run signature ·
Field.at(depth=, range=) · env.plot · compare_models · plot.compare ·
plot.plot_field_statistics
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

env = uacpy.Environment(
    name="Deep ocean — long-range SOFAR",
    bathymetry=5500.0,
    ssp=uacpy.SoundSpeedProfile.from_munk(5500.0),
    bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                    sound_speed=3000.0,   # hard abyssal plain
                                    density=2.2,
                                    attenuation=0.05),    # and low-loss
)
source = uacpy.Source(depths=1000.0,      # in the channel, 300 m above its axis
                      frequencies=20.0)   # low frequency buys the range
receiver = uacpy.Receiver(depths=np.linspace(100, 5400, 30),
                          ranges=np.linspace(1000, 150000, 150))

models = {'Bellhop': uacpy.Bellhop(), 'Kraken': uacpy.Kraken(),
          'Scooter': uacpy.Scooter(), 'OAST': uacpy.OAST()}
fields = {name: model.run(env, source, receiver)
          for name, model in models.items()}

# .at() takes labels, not indices, and snaps to the nearest sample on each
# axis — so the on-axis level at 100 km is one call per model.
print("On-axis TL at 100 km (channel axis 1300 m):")
for name, field in fields.items():
    print(f"  {name:8s} {float(field.at(depth=1300.0, range=100000.0).dB):5.1f} dB")

fig, _ = env.plot(source=source, receiver=receiver)
fig.savefig(OUT / 'example_08_environment.png', dpi=150, bbox_inches='tight')
plt.close(fig)

fig, _ = uacpy.compare_models(fields, env=env)
fig.savefig(OUT / 'example_08_fields.png', dpi=150, bbox_inches='tight')
plt.close(fig)

mid_range = float(np.median(receiver.ranges))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
uacpy.plot.compare([f.at(depth=1000.0) for f in fields.values()],
                   labels=list(fields), ax=axes[0],
                   title='TL vs range at source depth (1000 m)')
uacpy.plot.compare([f.at(range=mid_range) for f in fields.values()],
                   labels=list(fields), ax=axes[1],
                   title=f'TL vs depth at {mid_range / 1000:.0f} km')
fig.tight_layout()
fig.savefig(OUT / 'example_08_curves.png', dpi=150, bbox_inches='tight')
plt.close(fig)

fig, _ = uacpy.plot.plot_field_statistics(fields, depth=1000.0)
fig.savefig(OUT / 'example_08_stats.png', dpi=150, bbox_inches='tight')
plt.close(fig)
