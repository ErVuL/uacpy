"""RAM (mpiramS) — range-dependent bottom and bathymetry.

A sediment transition over a sloping shelf: soft mud → hard sand between 0 and
3 km, while the seafloor falls from 100 m to 150 m. The water column is one
stratified profile (range-independent), so the range dependence is entirely in
the seabed and the bathymetry — which is RAM's case.

The receiver grid spans 5-145 m while the seafloor runs 100-150 m, so the
shallow-range columns place receivers inside the sub-bottom and come back NaN.
Every number below is NaN-aware and says how much of the grid is water.

Uses: Bottom.from_halfspaces (range-dependent seabed) · a bathymetry array ·
env.is_range_dependent flags · RAM(accuracy=) · plot_field(contours=) ·
compare_models · plot.plot_field_difference
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

# Warm stratified water: 20 °C at the surface, ~8 °C deep, through Medwin's
# simplified c(T, z) — lead constant rounded from 1449.2 and the salinity term
# dropped at its S = 35 PSU reference value (Stergiopoulos, Advanced Signal
# Processing Handbook, Table 10.1).
depths = np.array([0., 25, 50, 75, 100, 120, 150])
temperature = 8 + 12 * np.exp(-depths / 50)
sound_speed = (1449 + 4.6 * temperature - 0.055 * temperature ** 2
               + 0.00029 * temperature ** 3 + 0.016 * depths)

# Range-dependent seabed: one half-space per range node, mud → sand.
bottom = uacpy.Bottom.from_halfspaces(
    np.array([0, 1000, 2000, 3000]),
    sound_speed=np.array([1500, 1580, 1640, 1700]),
    density=np.array([1.2, 1.5, 1.8, 2.0]),
    attenuation=np.array([1.0, 0.7, 0.5, 0.3]),
    shear_speed=np.zeros(4),
    acoustic_type='half-space',
)

env = uacpy.Environment(
    name="Sediment transition with sloping shelf",
    ssp=uacpy.SoundSpeedProfile.from_pairs(
        np.column_stack([depths, sound_speed])),
    bathymetry=np.array([[0, 100], [1000, 110], [2000, 130], [3000, 150]]),
    bottom=bottom,
)
print(f"  range-dependent: {env.is_range_dependent} "
      f"(ssp {env.has_range_dependent_ssp}, "
      f"bottom {env.has_range_dependent_bottom})")

source = uacpy.Source(depths=50.0, frequencies=100.0)
receiver = uacpy.Receiver(depths=np.linspace(5, 145, 30),
                          ranges=np.linspace(100, 3000, 30))

# accuracy is the Lytaev optimiser's per-run Padé error budget; 1e-1 is 100×
# looser than the default, so it picks a coarser dr/dz and the example runs
# quickly. Leave it at the default for production work.
fields = {'RAM': uacpy.RAM(accuracy=1e-1).run(env, source, receiver),
          'Bellhop': uacpy.Bellhop().run(env, source, receiver),
          'Kraken': uacpy.Kraken().run(env, source, receiver)}

for name, field in fields.items():
    tl = np.asarray(field.dB)
    in_water = int(np.isfinite(tl).sum())
    print(f"  {name:8s} TL {np.nanmin(tl):.1f} to {np.nanmax(tl):.1f} dB "
          f"[{in_water}/{tl.size} cells in water]")

# Read the model spread with care: at 100 Hz this guide is only
# D/λ = 7-10 wavelengths deep. Stergiopoulos §10.2.2 puts ray methods at
# D/λ ≳ 100 and mode/PE methods below ~30, so Bellhop is outside its regime
# here and RAM is the reference, not the other way round.
wavelength = 1500.0 / source.frequencies[0]
print(f"  guide is D/λ = {env.bathymetry.depths.min() / wavelength:.0f}-"
      f"{env.bathymetry.depths.max() / wavelength:.0f} wavelengths deep — "
      f"a PE regime, not a ray one")
print(f"  mean |RAM − Bellhop| = "
      f"{np.nanmean(np.abs(fields['RAM'].dB - fields['Bellhop'].dB)):.1f} dB, "
      f"|RAM − Kraken| = "
      f"{np.nanmean(np.abs(fields['RAM'].dB - fields['Kraken'].dB)):.1f} dB")

fig, _ = env.plot()
fig.savefig(OUT / 'example_05_environment.png', dpi=150, bbox_inches='tight')
plt.close(fig)

fig, ax = uacpy.plot_field(fields['RAM'], env=env, contours=[70, 85, 100],
                           vmin=40, vmax=100,
                           title='RAM — sediment transition, sloping shelf')
fig.savefig(OUT / 'example_05_result.png', dpi=150, bbox_inches='tight')
plt.close(fig)

fig, _ = uacpy.compare_models(
    fields, env=env, vmin=40, vmax=100,
    title='Three models — sediment transition + sloping shelf')
fig.savefig(OUT / 'example_05_comparison.png', dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(20, 5))
for ax, (a, b) in zip(axes, [('RAM', 'Bellhop'), ('RAM', 'Kraken'),
                             ('Bellhop', 'Kraken')]):
    uacpy.plot.plot_field_difference(fields[a], fields[b], ax, env=env,
                                     title=f'{a} − {b}')
fig.suptitle('Pairwise differences (signed, dB)', fontsize=13,
             fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'example_05_differences.png', dpi=150)
plt.close(fig)
