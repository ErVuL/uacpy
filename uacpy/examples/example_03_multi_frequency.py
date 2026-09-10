"""Five models on one thermocline at a single reference frequency.

Bellhop, RAM, Kraken, Scooter and OAST run on one summer-thermocline
environment at 100 Hz, compared as fields, as cuts, and as agreement numbers.
The 0-25 m mixed layer has zero gradient and so ducts nothing (a surface duct
needs a positive gradient — Etter §3.7), and at 100 Hz it is far below the
~1434 Hz cutoff a 25 m duct would need anyway: energy refracts downward through
the -1.0 (m/s)/m thermocline instead.

Uses: SoundSpeedProfile.from_pairs · five model classes with one .run signature ·
env.plot · compare_models · plot.compare · plot.plot_field_statistics
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
    name="Summer thermocline",
    bathymetry=200.0,
    ssp=uacpy.SoundSpeedProfile.from_pairs(
        [(0, 1525), (25, 1525), (60, 1490), (200, 1490)]),
    bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                    sound_speed=1650.0, shear_speed=250.0,
                                    density=1.7, attenuation=0.4),
)
source = uacpy.Source(depths=15.0,        # in the mixed layer, above the duct
                      frequencies=100.0)
receiver = uacpy.Receiver(depths=np.linspace(3, 197, 70),
                          ranges=np.linspace(200, 25000, 120))

# Every model takes the same (env, source, receiver): swapping the solver is
# swapping one word.
models = {'Bellhop': uacpy.Bellhop(), 'RAM': uacpy.RAM(),
          'Kraken': uacpy.Kraken(), 'Scooter': uacpy.Scooter(),
          'OAST': uacpy.OAST()}
fields = {name: model.run(env, source, receiver)
          for name, model in models.items()}
for name, field in fields.items():
    print(f"  {name:8s} TL {np.nanmin(field.dB):.1f} to "
          f"{np.nanmax(field.dB):.1f} dB")

fig, _ = env.plot(source=source, receiver=receiver)
fig.savefig(OUT / 'example_03_environment.png', dpi=150, bbox_inches='tight')
plt.close(fig)

fig, _ = uacpy.compare_models(fields, env=env)
fig.savefig(OUT / 'example_03_fields.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# The same five fields as 1-D cuts: slice each one first, then overlay.
mid_range = float(np.median(receiver.ranges))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
uacpy.plot.compare([f.at(depth=15.0) for f in fields.values()],
                   labels=list(fields), ax=axes[0],
                   title='TL vs range at 15 m depth')
uacpy.plot.compare([f.at(range=mid_range) for f in fields.values()],
                   labels=list(fields), ax=axes[1],
                   title=f'TL vs depth at {mid_range / 1000:.1f} km')
fig.tight_layout()
fig.savefig(OUT / 'example_03_curves.png', dpi=150, bbox_inches='tight')
plt.close(fig)

fig, _ = uacpy.plot.plot_field_statistics(fields, depth=15.0)
fig.savefig(OUT / 'example_03_stats.png', dpi=150, bbox_inches='tight')
plt.close(fig)
