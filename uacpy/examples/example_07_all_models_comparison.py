"""All five models on one hard environment.

A continental margin with everything switched on at once: a 2-D range-dependent
sound speed across a thermal front, a seabed whose properties change along
range, a shelf-break slope, and Thorp volume attenuation. Bellhop, RAM, Kraken,
Scooter and OAST each take the same Environment and reduce whatever they cannot
represent — warning about it — rather than refusing.

Volume attenuation is honoured by Bellhop, Kraken, Scooter and RAM (every RAM
backend takes it as a dB-per-wavelength profile on the water wavenumber). OASES
substitutes its own internal Skretting-Leroy attenuation for AC=0 water layers
and says so at runtime.

Uses: SoundSpeedProfile.from_2d · Bottom.from_halfspaces · uacpy.Thorp ·
Kraken(mode_coupling=, n_segments=) · env.plot · compare_models(ncols=,
contours=)
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

# A thermal front: warm shelf water inshore, cooler water offshore, stratified
# exponentially with depth. c(T, z) is Medwin's equation kept to its linear
# terms — a smooth stand-in for a frontal profile, not a calibrated seawater
# equation.
depths = np.linspace(0, 200, 21)
ranges_m = np.array([0.0, 2000.0, 4000.0, 6000.0, 8000.0])
t_surface = 18 - (ranges_m / 1000.0) * 0.3          # 18 °C → 15.6 °C at 8 km
t_deep = 8 - (ranges_m / 1000.0) * 0.1
temperature = t_deep + (t_surface - t_deep) * np.exp(-depths[:, None] / 40)
ssp_matrix = 1449 + 4.6 * temperature + 0.016 * depths[:, None]

bathymetry = np.array([[0, 100], [2000, 110], [4000, 130],
                       [6000, 150], [8000, 170]])
env = uacpy.Environment(
    name="Continental margin — frontal zone",
    ssp=uacpy.SoundSpeedProfile.from_2d(depths=depths, ranges=ranges_m,
                                        matrix=ssp_matrix),
    bathymetry=bathymetry,
    bottom=uacpy.Bottom.from_halfspaces(
        bathymetry[:, 0],
        sound_speed=np.array([1550, 1600, 1640, 1680, 1720]),
        density=np.array([1.4, 1.55, 1.7, 1.85, 2.0]),
        attenuation=np.array([1.0, 0.8, 0.6, 0.5, 0.4]),
        shear_speed=np.zeros(5),
        acoustic_type='half-space'),
    absorption=uacpy.Thorp(),
)
print(f"  SSP {ssp_matrix.min():.1f}-{ssp_matrix.max():.1f} m/s, "
      f"range-dependent ssp={env.has_range_dependent_ssp} "
      f"bottom={env.has_range_dependent_bottom}")

source = uacpy.Source(depths=50.0, frequencies=100.0)
receiver = uacpy.Receiver(depths=np.linspace(5, 165, 30),
                          ranges=np.linspace(100, 8000, 40))

# Kraken runs adiabatic here: each mode propagates independently, so the
# range-dependent guide costs one mode solve per segment and no more.
models = {'Bellhop': uacpy.Bellhop(),
          'RAM': uacpy.RAM(accuracy=1e-1),
          'Kraken': uacpy.Kraken(mode_coupling='adiabatic', n_segments=4),
          'Scooter': uacpy.Scooter(),
          'OAST': uacpy.OAST()}
fields = {name: model.run(env, source, receiver)
          for name, model in models.items()}
for name, field in fields.items():
    # NaN-aware: RAM masks sub-seafloor cells, so a plain mean would be nan.
    print(f"  {name:8s} TL [{np.nanmin(field.dB):5.1f}, "
          f"{np.nanmax(field.dB):5.1f}] dB, mean {np.nanmean(field.dB):5.1f}")

fig, ax = env.plot()
ax.set_title('Thermal front: 2-D range-dependent SSP + bottom')
fig.savefig(OUT / 'example_07_environment.png', dpi=150, bbox_inches='tight')
plt.close(fig)

fig, _ = uacpy.compare_models(fields, env=env)
fig.savefig(OUT / 'example_07_comparison.png', dpi=150, bbox_inches='tight')
plt.close(fig)

fig, _ = uacpy.compare_models(fields, env=env, ncols=3, vmin=50, vmax=110,
                              contours=[70, 90],
                              title='All models — TL with 70/90 dB contours')
fig.savefig(OUT / 'example_07_models.png', dpi=150)
plt.close(fig)
