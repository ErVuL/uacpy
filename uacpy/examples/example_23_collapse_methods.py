"""Per-feature collapse methods.

When an Environment carries a feature a model cannot represent, uacpy reduces
it rather than refusing — and the `collapse={…}` constructor parameter is where
you choose HOW, one key per feature, with a UserWarning naming each thing it
dropped.

The same range-dependent environment goes to Scooter (range-independent
wavenumber integration, so it drops both the RD bathymetry and the RD SSP) four
times, varying 'bathymetry' and 'ssp'. A range-independent model is the right
vehicle: RAM and Kraken honour range dependence natively, so their collapse
kwargs would be no-ops here.

Uses: Scooter(collapse={'bathymetry': …, 'ssp': …}) ·
SoundSpeedProfile.from_2d · compare_models(ncols=, contours=)
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

# A slope from 80 m to 200 m under a strongly contrasted range-dependent SSP:
# downward-refracting summer thermocline inshore, nearly isothermal offshore.
# That contrast is what makes 'r0', 'rmax' and 'mean' land on visibly different
# profiles. 'median' is not exercised — the middle profile is the half-sum of
# the outer two, so over three ranges it coincides with 'mean' exactly.
ssp_depths = np.linspace(0.0, 200.0, 21)
inshore = 1525.0 + (1480.0 - 1525.0) * np.tanh(ssp_depths / 30.0)
offshore = 1480.0 + (ssp_depths / 200.0) * 35.0

env = uacpy.Environment(
    name='Continental shelf — range-dependent demo',
    ssp=uacpy.SoundSpeedProfile.from_2d(
        depths=ssp_depths,
        ranges=np.array([0.0, 10000.0, 20000.0]),
        matrix=np.column_stack([inshore, 0.5 * (inshore + offshore),
                                offshore])),
    bathymetry=np.column_stack([np.linspace(0.0, 20000.0, 11),
                                np.linspace(80.0, 200.0, 11)]),
    bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                    sound_speed=1700.0, density=1.7,
                                    attenuation=0.5),
)
source = uacpy.Source(depths=20.0, frequencies=200.0)
receiver = uacpy.Receiver(depths=np.linspace(5.0, 195.0, 39),
                          ranges=np.linspace(500.0, 15000.0, 60))

fig, _ = env.plot()
fig.savefig(OUT / 'example_23_environment.png', dpi=150, bbox_inches='tight')
plt.close(fig)

fields = {
    f"bathy={bathymetry!r}, ssp={ssp!r}":
        uacpy.Scooter(collapse={'bathymetry': bathymetry, 'ssp': ssp}).run(
            env, source, receiver)
    for bathymetry, ssp in [('max', 'r0'), ('max', 'rmax'),
                            ('median', 'mean'), ('min', 'rmax')]
}
for label, field in fields.items():
    print(f"  {label:34s} TL {np.nanmin(field.dB):.1f}-"
          f"{np.nanmax(field.dB):.1f} dB")

fig, _ = uacpy.compare_models(
    fields, env=env, ncols=2, vmin=40, vmax=110, contours=[60, 80],
    title='One range-dependent environment, collapsed four ways')
fig.savefig(OUT / 'example_23_collapse_methods.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)
