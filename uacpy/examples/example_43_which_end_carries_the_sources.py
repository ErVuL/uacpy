"""Which end of the path carries the sources, and what getting it wrong costs.

A receive array sits at r = 0 and a target out at range, over the
range-dependent section of example_42. Three runs per separation:

    (a) sources at the array's element depths, receiver at the target
    (b) the mirror: source at the target depth, receivers on the array
    (c) the true reciprocal of (a): (b) on the section reversed about that range

(b) vs (c) is what the mirror arrangement costs, (a) vs (c) the solver's own
departure from reciprocity, and on a flat bottom (a) and (b) agree. With the
sources on the array, a ``Source(beam_pattern=)`` is the elements' receive
directivity. docs/guide/source-receiver.md §5 and §7 give the reasoning.

Uses: Environment.plot(source_marker_range=) · Kraken(mode_coupling=) ·
Bathymetry.eval · Field.p over source_depth ·
Source(beam_pattern=) as element directivity
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import uacpy

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

FREQ, C_REF = 200.0, 1500.0          # Hz, m/s
N_EL, ARRAY_DEPTH = 24, 120.0        # elements at lambda/2 about this depth (m)
TARGET_DEPTH = 60.0                  # m
SEPARATIONS = np.linspace(4000.0, 20000.0, 6)

# Env definition: a shelf, a sill at 130 m, a basin, a shoaling run; a flat control
bathy_ranges = np.linspace(0.0, 20000.0, 41)
bathy_depths = np.interp(bathy_ranges,
                         [0.0, 3000.0, 6000.0, 9000.0,
                          11000.0, 13500.0, 16500.0, 20000.0],
                         [200.0, 200.0, 160.0, 130.0,
                          150.0, 175.0, 145.0, 120.0])
env = uacpy.Environment(
    name='sill-and-basin',
    bathymetry=list(zip(bathy_ranges, bathy_depths)),
    ssp=[(0.0, 1520.0), (30.0, 1518.0), (80.0, 1502.0), (200.0, 1498.0)],
    bottom=uacpy.Bottom.from_halfspace(uacpy.BoundaryProperties(
        acoustic_type='half-space', sound_speed=1700.0, density=1.9,
        attenuation=0.5)),
    absorption=uacpy.Thorp(),
)
flat = uacpy.Environment(name='flat-control', bathymetry=200.0, ssp=env.ssp,
                         bottom=env.bottom, absorption=uacpy.Thorp())

elements = ARRAY_DEPTH + 0.5 * C_REF / FREQ * (np.arange(N_EL) - (N_EL - 1) / 2.0)
target = uacpy.Source(depths=TARGET_DEPTH, frequencies=FREQ)
array_as_sources = uacpy.Source(depths=elements, frequencies=FREQ)
kraken = uacpy.Kraken(mode_coupling='coupled')

fig, ax = env.plot(source=target,
                   receiver=uacpy.Receiver(depths=elements, ranges=[0.0]),
                   source_marker_range=9000.0,
                   title='The array is on the shelf, the target beyond the sill')
ax.plot([9.0] * N_EL, elements, 'x', color='C1', ms=4,
        label='where the mirror puts the array')
ax.legend(fontsize=8, loc='lower left')
fig.savefig(OUT / 'example_43_scene.png', dpi=140, bbox_inches='tight')
plt.close(fig)

# The three arrangements at each separation, and (a) vs (b) on the flat control
step = float(bathy_ranges[1] - bathy_ranges[0])
physics, solver, naive, flat_gap = [], [], [], []
for R in SEPARATIONS:
    rx_array = uacpy.Receiver(depths=elements, ranges=[R])
    rx_target = uacpy.Receiver(depths=[TARGET_DEPTH], ranges=[R])
    walk = np.arange(0.0, R + step, step)
    reversed_env = uacpy.Environment(
        name='reversed-about-R',
        bathymetry=list(zip(walk, env.bathymetry.eval(range=R - walk))),
        ssp=env.ssp, bottom=env.bottom, absorption=uacpy.Thorp())
    tl_a, tl_b, tl_c, flat_a, flat_b = (
        -10.0 * np.log10(np.mean(np.abs(p) ** 2)) for p in (
            kraken.run(env, array_as_sources, rx_target).p,
            kraken.run(env, target, rx_array).p,
            kraken.run(reversed_env, target, rx_array).p,
            uacpy.Kraken().run(flat, array_as_sources, rx_target).p,
            uacpy.Kraken().run(flat, target, rx_array).p))
    physics.append(abs(tl_b - tl_c))
    solver.append(abs(tl_a - tl_c))
    naive.append(abs(tl_a - tl_b))
    flat_gap.append(abs(flat_a - flat_b))
print(f"element-averaged TL over {SEPARATIONS[0] / 1e3:.0f}-"
      f"{SEPARATIONS[-1] / 1e3:.0f} km (median / worst):")
print(f"  (b) vs (c), the mirror's cost      {np.median(physics):5.2f} / "
      f"{np.max(physics):5.2f} dB")
print(f"  (a) vs (c), the solver's residual  {np.median(solver):5.2f} / "
      f"{np.max(solver):5.2f} dB")
print(f"  (a) vs (b), the naive difference   {np.median(naive):5.2f} / "
      f"{np.max(naive):5.2f} dB")
print(f"  (a) vs (b) on the flat bottom      {np.max(flat_gap):.4f} dB at worst")

# Element directivity by reciprocity: a beam pattern on the sources at the array
pattern_angles = np.linspace(-90.0, 90.0, 361)
rx_probe = uacpy.Receiver(depths=[TARGET_DEPTH], ranges=[5000.0])
print("\nreceive directivity of each element (5 km, 60 m), element-mean TL:")
for label, response_dB in (
        ('omnidirectional', None),
        ('steep angles rejected 20 dB',
         np.where(np.abs(pattern_angles) > 15.0, -20.0, 0.0)),
        ('only steep angles kept',
         np.where(np.abs(pattern_angles) < 15.0, -40.0, 0.0))):
    shaded = uacpy.Source(
        depths=elements, frequencies=FREQ,
        beam_pattern=None if response_dB is None
        else np.column_stack([pattern_angles, response_dB]))
    p = kraken.run(env, shaded, rx_probe).p
    print(f"  {label:28s} {-10.0 * np.log10(np.mean(np.abs(p) ** 2)):6.2f} dB")

# Plots
fig, ax = plt.subplots(figsize=(7, 4.2))
km = SEPARATIONS / 1e3
ax.plot(km, physics, 'C3o-', lw=1.4, label="(b) vs (c): the mirror's cost")
ax.plot(km, naive, 'C0s--', lw=1.2, label='(a) vs (b): the naive difference')
ax.plot(km, solver, 'C7^:', lw=1.2, label="(a) vs (c): the solver's residual")
ax.plot(km, flat_gap, 'k.-', lw=1.0, label='(a) vs (b), flat bottom')
ax.set(xlabel='Separation (km)', ylabel='|TL difference| (dB)',
       title='Which end carries the sources')
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'example_43_reciprocity.png', dpi=140, bbox_inches='tight')
plt.close(fig)
print(f"\nFigures written to {OUT}")
