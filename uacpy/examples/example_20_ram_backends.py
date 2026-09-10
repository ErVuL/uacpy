"""RAM multi-backend dispatch — mpiramS, RAMS (elastic), RAMSurf (rough).

One RAM class, three vendored Collins-family PE binaries, chosen from what the
environment contains:

* mpiramS (default) — fluid bottom, flat surface; native broadband Q/T loop.
* rams0.5 — an ELASTIC bottom (any shear_speed > 0); single frequency,
  fluid-elastic coupling through Lamé parameters in the sediment.
* ramsurf1.5 — fluid bottom with a variable surface (env.altimetry); single
  frequency, rough-surface propagation.

Elastic AND rough together raises UnsupportedFeatureError: no published
Collins PE covers that combination. Use OASES for range-independent elastic
problems, or fluidise / flatten one side as an approximation.

Uses: RAM.select_backend (the dispatch, before any run) · Field.backend (what
actually ran) · env altimetry · SeabedColumn with shear
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.core.exceptions import UnsupportedFeatureError

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

depth, r_max = 100.0, 5000.0
source = uacpy.Source(depths=50.0, frequencies=100.0)
receiver = uacpy.Receiver(depths=np.linspace(2.0, depth - 2.0, 30),
                          ranges=np.linspace(200.0, r_max, 40))

fluid = uacpy.BoundaryProperties(acoustic_type='half-space',
                                 sound_speed=1700.0, density=1.7,
                                 attenuation=0.5)
elastic = uacpy.SeabedColumn(
    layers=[uacpy.SedimentLayer(thickness=15.0, sound_speed=1700.0,
                                density=1.7, attenuation=0.5,
                                shear_speed=400.0, shear_attenuation=1.5)],
    halfspace=uacpy.BoundaryProperties(acoustic_type='half-space',
                                       sound_speed=1900.0, density=2.0,
                                       attenuation=0.2, shear_speed=600.0,
                                       shear_attenuation=0.5))

# ramsurf1.5 can only place the pressure-release surface at or below mean sea
# level; any node above 0 is clamped. A zero-mean sinusoid would arrive
# half-wave rectified — 29 of these 50 nodes flattened, with the panel still
# labelled "rough" — so the corrugation is biased down until the whole 3 m
# peak-to-peak profile is representable.
surface = [(r, -1.6 + 1.5 * np.sin(2 * np.pi * r / 2000.0))
           for r in np.linspace(0.0, r_max, 50)]
heights = np.array([z for _, z in surface])
print(f"  surface corrugation {heights.min():.2f} to {heights.max():.2f} m, "
      f"{int((heights > 0).sum())}/{heights.size} nodes above 0 "
      f"(any such node would be clamped)")

cases = {
    'mpiramS (fluid + flat)': uacpy.Environment(
        name='fluid-flat', bathymetry=depth, ssp=1500.0, bottom=fluid),
    'rams0.5 (elastic + flat)': uacpy.Environment(
        name='elastic-flat', bathymetry=depth, ssp=1500.0, bottom=elastic),
    'ramsurf1.5 (fluid + rough)': uacpy.Environment(
        name='fluid-rough', bathymetry=depth, ssp=1500.0, bottom=fluid,
        altimetry=surface),
}

ram = uacpy.RAM(accuracy=1e-1)
for label, env in cases.items():
    print(f"  {label:28s} → {ram.select_backend(env)}")

# The elastic + rough combination is the documented gap, demonstrated rather
# than described.
try:
    ram.select_backend(uacpy.Environment(name='elastic-rough',
                                         bathymetry=depth, ssp=1500.0,
                                         bottom=elastic, altimetry=surface))
except UnsupportedFeatureError as exc:
    # First line only: the remediation block below it is several lines long.
    print(f"  elastic + altimetry → UnsupportedFeatureError: "
          f"{str(exc).splitlines()[0]}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, (label, env) in zip(axes, cases.items()):
    field = ram.run(env, source, receiver,
                    run_mode=uacpy.RunMode.COHERENT_TL)
    uacpy.plot_field(field, ax=ax, env=env, vmin=30, vmax=110,
                     title=f"{label}\nbackend={field.backend}")
fig.tight_layout()
fig.savefig(OUT / 'example_20_ram_backends.png', dpi=120)
plt.close(fig)
