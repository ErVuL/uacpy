"""Sound-speed profiles — Munk, Pekeris, thermocline.

Three water columns, the same comparison run on each: a deep-ocean Munk profile
with its SOFAR channel, a shallow isovelocity Pekeris waveguide over an elastic
seabed, and a coastal mixed layer over a thermocline. The SSP shape is what
decides the physics — trapping, convergence zones, downward refraction.

A zero-gradient mixed layer ducts nothing: trapping needs dc/dz > 0, and a duct
of thickness H cuts off below c / (8.51e-3·H^1.5) anyway (Etter §3.7.3) — 505 Hz
for the 50 m layer in scenario C, five times its 100 Hz.

Uses: SoundSpeedProfile.from_munk / from_pairs · a float ssp for isovelocity ·
elastic bottom via shear_speed · Kraken.compute_modes().n_modes / .k ·
env.plot · compare_models · plot.compare · plot.plot_field_statistics
"""

import os
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

# A. Munk canonical profile: c_min = 1500 m/s on the axis at 1300 m, source
# 300 m above it and still well inside the channel.
munk_env = uacpy.Environment(
    name="Deep ocean — Munk profile",
    bathymetry=5000.0,
    ssp=uacpy.SoundSpeedProfile.from_munk(5000.0),
    bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                    sound_speed=1600, density=1.5,
                                    attenuation=0.2),
)

# B. Pekeris waveguide: isovelocity water (a bare float is a whole SSP) over an
# ELASTIC seabed — shear_speed > 0 is what makes it elastic.
pekeris_env = uacpy.Environment(
    name="Pekeris waveguide — elastic bottom",
    bathymetry=100.0,
    ssp=1500.0,
    bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                    sound_speed=1700.0, shear_speed=400.0,
                                    density=1.8, attenuation=0.5,
                                    shear_attenuation=0.8),
)

# C. Coastal: a bilinear profile, zero-gradient mixed layer over a
# -0.27 (m/s)/m thermocline.
coastal_env = uacpy.Environment(
    name="Coastal — mixed layer over thermocline",
    bathymetry=200.0,
    ssp=uacpy.SoundSpeedProfile.from_pairs([(0, 1520), (50, 1520),
                                            (200, 1480)]),
    bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                    sound_speed=1600, density=1.7,
                                    attenuation=0.5),
)

# What the elastic seabed changes, measured on its modes. The trapped count is
# set by the compressional critical angle, so the fluid half-space with the
# same c_p traps as many modes as the elastic one (both 4 at 50 Hz; a rigid
# bottom traps 7). Shear shows up in the loss instead: the seabed carries
# energy away as shear waves, so every trapped mode decays faster
# (Im k about 1.2-2.3x the fluid's here).
pekeris_source = uacpy.Source(depths=50.0, frequencies=50.0)
fluid_env = uacpy.Environment(
    name="Pekeris waveguide — fluid bottom, same c_p", bathymetry=100.0,
    ssp=1500.0,
    bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                    sound_speed=1700.0, density=1.8,
                                    attenuation=0.5))
rigid_env = uacpy.Environment(
    name="Pekeris waveguide — rigid bottom", bathymetry=100.0, ssp=1500.0,
    bottom=uacpy.BoundaryProperties(acoustic_type='rigid'))
elastic_modes = uacpy.Kraken(backend='krakenc').compute_modes(
    pekeris_env, pekeris_source)
fluid_modes = uacpy.Kraken(backend='krakenc').compute_modes(
    fluid_env, pekeris_source)
n_rigid = uacpy.Kraken(backend='kraken').compute_modes(
    rigid_env, pekeris_source).n_modes
print(f"  Trapped modes at 50 Hz: elastic {elastic_modes.n_modes}, fluid "
      f"same c_p {fluid_modes.n_modes}, rigid {n_rigid}")
# Modal attenuation in dB/km: 20·log10(e)·|Im k| per metre, times 1000.
n_shared = min(elastic_modes.n_modes, fluid_modes.n_modes)
for m in range(n_shared):
    loss_elastic = 8.686e3 * abs(elastic_modes.k[m].imag)
    loss_fluid = 8.686e3 * abs(fluid_modes.k[m].imag)
    print(f"    mode {m + 1}: {loss_elastic:6.3f} dB/km elastic vs "
          f"{loss_fluid:6.3f} dB/km fluid")

scenarios = [
    ('example_02a_munk', munk_env,
     uacpy.Source(depths=1000.0, frequencies=25.0),
     uacpy.Receiver(depths=np.linspace(100, 4900, 25),
                    ranges=np.linspace(1000, 20000, 25)),
     {'Bellhop': uacpy.Bellhop(backend='fortran'), 'Kraken': uacpy.Kraken()}),
    ('example_02b_pekeris', pekeris_env, pekeris_source,
     uacpy.Receiver(depths=np.linspace(3, 97, 30),
                    ranges=np.linspace(300, 5000, 40)),
     {'Bellhop': uacpy.Bellhop(backend='fortran'), 'RAM': uacpy.RAM(),
      'Kraken': uacpy.Kraken(), 'Scooter': uacpy.Scooter()}),
    ('example_02c_thermocline', coastal_env,
     uacpy.Source(depths=30.0, frequencies=100.0),
     uacpy.Receiver(depths=np.linspace(3, 197, 30),
                    ranges=np.linspace(500, 5000, 40)),
     {'Bellhop': uacpy.Bellhop(backend='fortran'), 'RAM': uacpy.RAM(),
      'Kraken': uacpy.Kraken(), 'Scooter': uacpy.Scooter()}),
]

# Two model limits stay in the comparison on purpose, and their notices are
# printed rather than hidden: in the Munk column the 1 km receivers see paths
# steeper than Kraken's mode window keeps (none are past about 14 km), and at
# 50 Hz the 100 m Pekeris guide is 3.3 wavelengths deep, under the D/λ >= 5
# floor of ray theory. The panels show where each departs from the others.
for prefix, env, source, receiver, models in scenarios:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fields = {name: model.run(env, source, receiver)
                  for name, model in models.items()}
    for warning in caught:
        print(f"  noted: {str(warning.message).splitlines()[0]}")
    depth = float(source.depths[0])
    mid_range = float(np.median(receiver.ranges))
    print(f"  {env.name}: " + ", ".join(
        f"{name} {np.nanmin(f.dB):.0f}-{np.nanmax(f.dB):.0f} dB"
        for name, f in fields.items()))

    fig, _ = env.plot(source=source, receiver=receiver)
    fig.savefig(OUT / f'{prefix}_environment.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    fig, _ = uacpy.plot.compare_models(fields, env=env)
    fig.savefig(OUT / f'{prefix}_fields.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    uacpy.plot.compare([f.at(depth=depth) for f in fields.values()],
                       labels=list(fields), ax=axes[0],
                       title=f'TL vs range at {depth:.0f} m')
    uacpy.plot.compare([f.at(range=mid_range) for f in fields.values()],
                       labels=list(fields), ax=axes[1],
                       title=f'TL vs depth at {mid_range / 1000:.1f} km')
    fig.tight_layout()
    fig.savefig(OUT / f'{prefix}_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, _ = uacpy.plot.plot_field_statistics(fields, depth=depth)
    fig.savefig(OUT / f'{prefix}_stats.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
