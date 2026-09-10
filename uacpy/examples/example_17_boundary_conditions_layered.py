"""Boundary conditions — top surfaces and layered bottoms.

Seven boundaries on one 100 m waveguide, at 200 Hz, so the panels are directly
comparable: a flat pressure-release surface, a Pierson-Moskowitz rough sea, an
elastic ice cover; then a single sediment layer, three layers, the same three
built from material presets, and finally a layered seabed that changes along
range. Bellhop takes the surface cases, RAM the bottom ones.

Uses: env.altimetry from generate_sea_surface · an elastic `surface=` ·
SedimentLayer / SeabedColumn · SeabedColumn.from_presets ·
Bottom.from_columns · env.has_range_dependent_layered_bottom
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.core.environment import generate_sea_surface

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

source = uacpy.Source(frequencies=200, depths=25)
receiver = uacpy.Receiver(depths=np.linspace(1, 95, 30),
                          ranges=np.linspace(500, 5000, 40))
isovelocity = uacpy.SoundSpeedProfile.from_pairs([(0, 1500), (100, 1500)])
sand = uacpy.BoundaryProperties(acoustic_type='half-space', sound_speed=1600,
                                density=1.5, attenuation=0.5)
rock = uacpy.BoundaryProperties(acoustic_type='half-space', sound_speed=2500,
                                density=2.5, attenuation=0.1)

# Ice: angles here follow the ocean-acoustics convention, θ from the
# HORIZONTAL (grazing), so a critical angle is arccos(c1/c2), not arcsin
# (Jensen, Kuperman, Porter & Schmidt, 2nd ed., §1.4). Ice cp = 3500 m/s gives
# a compressional critical grazing angle of arccos(1480/3500) = 65.0°, but ice
# also has shear, and cs = 1800 m/s > c_water makes the SHEAR angle
# arccos(1480/1800) = 34.7° the binding one. Shallow-water modes sit at small
# grazing angles, well below both, so TL stays close to the vacuum case and the
# difference shows up in the interference phase rather than the loss level.
ice = uacpy.BoundaryProperties(acoustic_type='half-space', sound_speed=3500.0,
                               shear_speed=1800.0, density=0.9,
                               attenuation=1.0, shear_attenuation=2.0)

# Presets are fluid by default (elastic=False), so RAM dispatches to the fluid
# mpiramS backend; the rams0.5 elastic PE is conservative on its dz cap and
# would degrade accuracy at 200 Hz over 100 m of water.
preset_column = uacpy.SeabedColumn.from_presets(
    layers=[('clay', 5.0), ('silt', 15.0), ('sand', 30.0)],
    halfspace='limestone')

scenarios = {
    'Flat surface (Bellhop)': (
        uacpy.Environment(name='vacuum_surface', bathymetry=100,
                          ssp=isovelocity, bottom=sand),
        uacpy.Bellhop()),
    'Rough sea, 15 m/s (Bellhop)': (
        uacpy.Environment(
            name='rough_surface', bathymetry=100, ssp=isovelocity, bottom=sand,
            altimetry=generate_sea_surface(max_range=10000,
                                           wind_speed_mps=15,
                                           n_points=300, seed=42)),
        uacpy.Bellhop()),
    'Ice surface (Bellhop)': (
        uacpy.Environment(
            name='ice_surface', bathymetry=100,
            ssp=uacpy.SoundSpeedProfile.from_pairs([(0, 1480), (100, 1480)]),
            surface=ice, bottom=sand),
        uacpy.Bellhop()),
    'Single-layer bottom (RAM)': (
        uacpy.Environment(
            name='single_layer_bottom', bathymetry=100, ssp=isovelocity,
            bottom=uacpy.SeabedColumn(
                layers=[uacpy.SedimentLayer(thickness=10.0, sound_speed=1550,
                                            density=1.3, attenuation=0.8)],
                halfspace=rock)),
        uacpy.RAM(accuracy=1e-1)),
    'Multi-layer bottom (RAM)': (
        uacpy.Environment(
            name='multi_layer_bottom', bathymetry=100, ssp=isovelocity,
            bottom=uacpy.SeabedColumn(
                layers=[uacpy.SedimentLayer(thickness=5.0, sound_speed=1550,
                                            density=1.3, attenuation=0.8),
                        uacpy.SedimentLayer(thickness=15.0, sound_speed=1650,
                                            density=1.7, attenuation=0.4),
                        uacpy.SedimentLayer(thickness=30.0, sound_speed=1800,
                                            density=2.0, attenuation=0.2)],
                halfspace=rock)),
        uacpy.RAM(accuracy=1e-1)),
    'Preset layered bottom (RAM)': (
        uacpy.Environment(name='preset_layered_bottom', bathymetry=100,
                          ssp=isovelocity, bottom=preset_column),
        uacpy.RAM(accuracy=1e-1)),
    'Range-dep layered (RAM)': (
        uacpy.Environment(
            name='rd_layered_bottom', bathymetry=100, ssp=isovelocity,
            bottom=uacpy.Bottom.from_columns(
                [uacpy.SeabedColumn(                     # mud over clay
                    layers=[uacpy.SedimentLayer(thickness=8.0,
                                                sound_speed=1500, density=1.2,
                                                attenuation=1.0),
                            uacpy.SedimentLayer(thickness=20.0,
                                                sound_speed=1580, density=1.5,
                                                attenuation=0.6)],
                    halfspace=uacpy.BoundaryProperties(
                        acoustic_type='half-space', sound_speed=1800,
                        density=2.0, attenuation=0.2)),
                 uacpy.SeabedColumn(                     # sand over rock
                    layers=[uacpy.SedimentLayer(thickness=3.0,
                                                sound_speed=1650, density=1.8,
                                                attenuation=0.3),
                            uacpy.SedimentLayer(thickness=10.0,
                                                sound_speed=1750, density=2.0,
                                                attenuation=0.2)],
                    halfspace=uacpy.BoundaryProperties(
                        acoustic_type='half-space', sound_speed=2500,
                        density=2.5, attenuation=0.05))],
                ranges=np.array([0, 10000]))),
        uacpy.RAM(accuracy=1e-1)),
}

fields = {label: model.run(env, source, receiver)
          for label, (env, model) in scenarios.items()}
for label, field in fields.items():
    print(f"  {label:30s} TL [{np.nanmin(field.dB):5.1f}, "
          f"{np.nanmax(field.dB):5.1f}] dB")

# One colour window across all seven, so the panels can be read against each
# other rather than each against itself.
every_tl = np.concatenate([f.dB.ravel() for f in fields.values()])
vmin = 5 * round(max(30, np.nanpercentile(every_tl, 5)) / 5)
vmax = 5 * round(min(140, np.nanpercentile(every_tl, 95)) / 5)

fig, axes = plt.subplots(2, 4, figsize=(22, 11))
for index, (label, field) in enumerate(fields.items()):
    ax = axes.flat[index]
    uacpy.plot_field(field, ax, env=scenarios[label][0], show_colorbar=False,
                     vmin=vmin, vmax=vmax, title=label)
    if index % 4:                     # depth label on the left column only
        ax.set_ylabel('')
axes.flat[-1].axis('off')
fig.suptitle('Boundary conditions — surface and bottom', fontsize=14,
             fontweight='bold', y=0.995)
fig.subplots_adjust(left=0.06, right=0.93, top=0.90, bottom=0.07,
                    wspace=0.20, hspace=0.30)
fig.colorbar(axes.flat[0].collections[0],
             cax=fig.add_axes([0.945, 0.07, 0.012, 0.83]), label='TL (dB)')
fig.savefig(OUT / 'example_17_boundary_conditions.png', dpi=150)
plt.close(fig)

rd_env = scenarios['Range-dep layered (RAM)'][0]
print(f"  range-dependent layered bottom: "
      f"{rd_env.has_range_dependent_layered_bottom}")
fig, _ = rd_env.plot()
fig.savefig(OUT / 'example_17_rd_layered_structure.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)
