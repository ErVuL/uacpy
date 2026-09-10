"""Bellhop + BOUNCE, layered bottoms, range-dependent bottoms.

Four seabeds of rising complexity, each with the model that handles it:

1. an elastic half-space through Bellhop.run_with_bounce(), which is how you
   pin BOUNCE's own c_low / c_high / rmax (plain run() auto-routes layered
   bottoms through BOUNCE already, with a warning);
2. a three-layer sediment column with Scooter, which takes NMEDIA > 1;
3. a scalar seabed that varies along range, with RAM;
4. a LAYERED seabed that varies along range — depth and range together.

Uses: Bellhop.run_with_bounce · SedimentLayer / SeabedColumn ·
Bottom.from_halfspaces / from_columns · SoundSpeedProfile.from_2d ·
env.has_layered_bottom flags · plot_field(contours=) · plot_bottom_properties ·
env.ssp.plot · plot.plot_field_difference
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

# ── 1. Elastic half-space: native Bellhop against explicit BOUNCE ───────────
elastic_env = uacpy.Environment(
    name='Elastic sandy bottom',
    bathymetry=100,
    ssp=1500.0,
    bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                    sound_speed=1700.0, shear_speed=400.0,
                                    density=1.9, attenuation=0.5,
                                    shear_attenuation=1.0),
)
source = uacpy.Source(frequencies=500.0, depths=25.0)
receiver = uacpy.Receiver(depths=np.linspace(1, 99, 30),
                          ranges=np.linspace(100, 3000, 40))

# Only *layered* bottoms auto-route through BOUNCE, so this elastic half-space
# runs natively either way — bellhop.f90 applies its exact acousto-elastic
# reflection coefficient, shear included. auto_bounce=False makes that explicit.
native = uacpy.Bellhop(auto_bounce=False).run(
    elastic_env, source, receiver, run_mode=uacpy.RunMode.COHERENT_TL)
bounced = uacpy.Bellhop().run_with_bounce(
    elastic_env, source, receiver, run_mode=uacpy.RunMode.COHERENT_TL,
    c_low=1400.0, c_high=10000.0, rmax=10000.0)
print(f"  half-space TL {np.nanmin(native.dB):.1f}-{np.nanmax(native.dB):.1f} dB, "
      f"BOUNCE TL {np.nanmin(bounced.dB):.1f}-{np.nanmax(bounced.dB):.1f} dB, "
      f"max |Δ| {np.nanmax(np.abs(bounced.dB - native.dB)):.1f} dB")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
uacpy.plot_field(native, env=elastic_env, ax=axes[0], vmin=40, vmax=90,
                 title='Native elastic half-space')
uacpy.plot_field(bounced, env=elastic_env, ax=axes[1], vmin=40, vmax=90,
                 title='BOUNCE (with shear)')
uacpy.plot.plot_field_difference(bounced, native, axes[2], env=elastic_env,
                                 diff_vmax=10, title='BOUNCE − half-space')
fig.suptitle(f'Bellhop + BOUNCE at {source.frequencies[0]:.0f} Hz, elastic '
             f'sandy bottom (cp=1700, cs=400 m/s)', fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'example_16_bounce_comparison.png', dpi=150)
plt.close(fig)

fig, _ = uacpy.plot_field(bounced, env=elastic_env, contours=[60, 70, 80],
                          title='Bellhop + BOUNCE TL with contours')
fig.savefig(OUT / 'example_16_bounce_tl.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ── 2. Three sediment layers over a half-space, with Scooter ────────────────
layered = uacpy.SeabedColumn(
    layers=[
        uacpy.SedimentLayer(thickness=5.0, sound_speed=1550.0, density=1.5,
                            attenuation=0.3, shear_speed=100.0),
        uacpy.SedimentLayer(thickness=10.0, sound_speed=1650.0, density=1.7,
                            attenuation=0.5),
        uacpy.SedimentLayer(thickness=20.0, sound_speed=1800.0, density=2.0,
                            attenuation=0.8, shear_speed=200.0,
                            shear_attenuation=0.5),
    ],
    halfspace=uacpy.BoundaryProperties(acoustic_type='half-space',
                                       sound_speed=2000.0, density=2.2,
                                       attenuation=0.1),
)
layered_env = uacpy.Environment(name='Continental shelf — layered sediment',
                                bathymetry=200.0, ssp=1500.0, bottom=layered)
layered_tl = uacpy.Scooter().run(
    layered_env,
    uacpy.Source(frequencies=100.0, depths=50.0),
    uacpy.Receiver(depths=np.linspace(1, 199, 30),
                   ranges=np.linspace(100, 5000, 40)))
print(f"  {layered.total_thickness():.1f} m of sediment in "
      f"{len(layered.layers)} layers, has_layered_bottom="
      f"{layered_env.has_layered_bottom}; Scooter TL "
      f"{np.nanmin(layered_tl.dB):.1f}-{np.nanmax(layered_tl.dB):.1f} dB")

fig, _ = uacpy.plot_field(layered_tl, env=layered_env, contours=[70, 80, 90],
                          title='Scooter TL — 3 layers + half-space')
fig.savefig(OUT / 'example_16_layered_tl.png', dpi=150, bbox_inches='tight')
plt.close(fig)
fig, _ = uacpy.plot.plot_bottom_properties(layered_env)
fig.savefig(OUT / 'example_16_layered_structure.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)
fig, _ = layered_env.plot()
fig.savefig(OUT / 'example_16_layered_env.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ── 3. A scalar seabed that varies along range, with RAM ────────────────────
bathymetry = np.array([[0, 150], [3000, 160], [6000, 180],
                       [9000, 200], [12000, 220], [15000, 250]])
rd_bottom = uacpy.Bottom.from_halfspaces(
    bathymetry[:, 0].astype(float),
    sound_speed=np.array([1500, 1550, 1600, 1650, 1700, 1750]),
    density=np.array([1.2, 1.4, 1.5, 1.7, 1.8, 2.0]),
    attenuation=np.array([1.0, 0.8, 0.6, 0.5, 0.4, 0.3]),
    shear_speed=np.zeros(6),
    acoustic_type='half-space',
)

# A 2-D water column too: warm inshore, cooler offshore. c(T, z) is Medwin's
# equation truncated after the T² term, salinity fixed at S = 35 PSU.
ssp_depths = np.array([0, 25, 50, 100, 150, 200, 250])
ssp_ranges_km = np.array([0, 5, 10, 15])
t_surface = 18 - 0.4 * ssp_ranges_km
t_deep = 8 - 0.1 * ssp_ranges_km
temperature = t_deep + (t_surface - t_deep) * np.exp(-ssp_depths[:, None] / 60)
ssp_matrix = (1449 + 4.6 * temperature - 0.055 * temperature ** 2
              + 0.016 * ssp_depths[:, None])

rd_env = uacpy.Environment(
    name='Shelf break: mud to sand',
    ssp=uacpy.SoundSpeedProfile.from_2d(depths=ssp_depths,
                                        ranges=ssp_ranges_km * 1000.0,
                                        matrix=ssp_matrix),
    bathymetry=bathymetry,
    bottom=rd_bottom,
)
rd_tl = uacpy.RAM(accuracy=1e-1).run(
    rd_env,
    uacpy.Source(frequencies=100.0, depths=30.0),
    uacpy.Receiver(depths=np.linspace(5, 240, 30),
                   ranges=np.linspace(100, 5000, 40)))
print(f"  range-dependent seabed: RAM TL {np.nanmin(rd_tl.dB):.1f}-"
      f"{np.nanmax(rd_tl.dB):.1f} dB")

fig, _ = uacpy.plot_field(rd_tl, env=rd_env, contours=[70, 85, 100],
                          title='RAM TL — range-dependent bottom (mud to sand)')
fig.savefig(OUT / 'example_16_rd_bottom_tl.png', dpi=150, bbox_inches='tight')
plt.close(fig)
fig, _ = uacpy.plot.plot_bottom_properties(rd_env)
fig.savefig(OUT / 'example_16_rd_bottom_props.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)
fig, _ = rd_env.ssp.plot()      # one c(z) line per range column
fig.savefig(OUT / 'example_16_rd_ssp.png', dpi=150, bbox_inches='tight')
plt.close(fig)
fig, _ = rd_env.plot()
fig.savefig(OUT / 'example_16_rd_env.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ── 4. A LAYERED seabed that varies along range — depth and range together ──
inshore = uacpy.SeabedColumn(          # soft mud over clay
    layers=[uacpy.SedimentLayer(thickness=5.0, sound_speed=1500.0,
                                density=1.2, attenuation=1.0),
            uacpy.SedimentLayer(thickness=15.0, sound_speed=1550.0,
                                density=1.4, attenuation=0.8)],
    halfspace=uacpy.BoundaryProperties(acoustic_type='half-space',
                                       sound_speed=1700.0, density=1.8,
                                       attenuation=0.2))
midshelf = uacpy.SeabedColumn(         # mixed sediment
    layers=[uacpy.SedimentLayer(thickness=3.0, sound_speed=1580.0,
                                density=1.5, attenuation=0.6),
            uacpy.SedimentLayer(thickness=12.0, sound_speed=1650.0,
                                density=1.7, attenuation=0.4)],
    halfspace=uacpy.BoundaryProperties(acoustic_type='half-space',
                                       sound_speed=1900.0, density=2.0,
                                       attenuation=0.1))
offshore = uacpy.SeabedColumn(         # hard sand over rock
    layers=[uacpy.SedimentLayer(thickness=2.0, sound_speed=1700.0,
                                density=1.9, attenuation=0.3),
            uacpy.SedimentLayer(thickness=8.0, sound_speed=1850.0,
                                density=2.1, attenuation=0.2)],
    halfspace=uacpy.BoundaryProperties(acoustic_type='half-space',
                                       sound_speed=2200.0, density=2.5,
                                       attenuation=0.05))

rd_layered = uacpy.Bottom.from_columns([inshore, midshelf, offshore],
                                       ranges=np.array([0, 7500, 15000]))
rdl_env = uacpy.Environment(
    name='Shelf: mud/clay to sand/rock',
    ssp=1500.0,
    bathymetry=np.column_stack([[0.0, 7500.0, 15000.0],
                                [120.0, 180.0, 280.0]]),
    bottom=rd_layered,
)
rdl_tl = uacpy.RAM(accuracy=1e-1).run(
    rdl_env,
    uacpy.Source(frequencies=100.0, depths=30.0),
    uacpy.Receiver(depths=np.linspace(5, 270, 30),
                   ranges=np.linspace(100, 8000, 40)))
print(f"  range-dependent LAYERED seabed: "
      f"has_range_dependent_layered_bottom="
      f"{rdl_env.has_range_dependent_layered_bottom}, up to "
      f"{rd_layered.max_total_thickness():.1f} m of sediment; RAM TL "
      f"{np.nanmin(rdl_tl.dB):.1f}-{np.nanmax(rdl_tl.dB):.1f} dB")

fig, _ = uacpy.plot_field(rdl_tl, env=rdl_env, contours=[70, 85, 100],
                          title='RAM TL — range-dependent layered bottom')
fig.savefig(OUT / 'example_16_rdl_tl.png', dpi=150, bbox_inches='tight')
plt.close(fig)
fig, _ = rdl_env.plot()
fig.savefig(OUT / 'example_16_rdl_structure.png', dpi=150, bbox_inches='tight')
plt.close(fig)
