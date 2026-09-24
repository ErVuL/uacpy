"""A vertical source array: the stack, the two totals, the level, the response.

Several sources driven together raise questions one source never does. How do
their fields combine — coherently, as the elements of one array, or in
intensity, as unrelated platforms? How loud is the result, rather than how
much quieter than a unit source? And what does the array do to the channel?

Transmission loss is referenced to ONE unit source at 1 m (JKPS 1.3.4), so a
map of a multi-source total is a level with the array's gain inside it, not a
loss — which is why the last panels name a source level and read out dB re
1 µPa instead.

The array response has two right answers and this draws both. Free field, the
product theorem gives the array beam pattern P(theta) = f(theta)*A(theta)
(Butler & Sherman 7.1.1). In a waveguide the array is a mode filter (Medwin &
Clay 11.3.1): it sets each mode's amplitude to sum_n w_n phi_m(z_n). They
agree while the pattern is symmetric in +/-theta, and part company once
steering breaks that symmetry.

Uses: Source(depths=[...], weights=, source_level_dB=, beam_pattern=) ·
ResultStack.plot / .superpose(coherent=True|False) · Field.at_source_level ·
Source.array_factor / .array_beam_pattern · Modes.excitation ·
plot_mode_excitation · Kraken (one launch for every depth) vs RAM (one march
per depth, same contract)
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

FREQ = 150.0
C_REF = 1500.0

# A 200 m isovelocity duct over a sand half-space.
env = uacpy.Environment(
    name='array-duct', bathymetry=200.0, ssp=C_REF,
    bottom=uacpy.Bottom.from_halfspace(uacpy.BoundaryProperties(
        acoustic_type='half-space', sound_speed=1700.0, density=1.9,
        attenuation=0.5)),
)
receiver = uacpy.Receiver(depths=np.linspace(5.0, 195.0, 60),
                          ranges=np.linspace(200.0, 12000.0, 150))

# Five elements on a half-wavelength ladder, driven at 180 dB each. The
# weights are the relative complex drives; source_level_dB is how hard one
# unit weight is driven, and rides with every result the run produces.
spacing = 0.5 * C_REF / FREQ
depths = 100.0 + spacing * np.arange(-2, 3)
k = 2.0 * np.pi * FREQ / C_REF
steer_deg = 12.0
weights = np.exp(-1j * k * (depths - depths.mean())
                 * np.sin(np.deg2rad(steer_deg)))

array = uacpy.Source(depths=depths, frequencies=FREQ, weights=weights,
                     source_level_dB=180.0)
# The same five positions, but independent sources: no steering phase, equal
# drives. An intensity sum reads only |w|, so giving it a steered weight set
# would be a modelling error the incoherent path warns about.
fleet = uacpy.Source(depths=depths, frequencies=FREQ,
                     source_level_dB=180.0)
kraken = uacpy.Kraken(verbose=False)

# ── 1. one slab per source ──────────────────────────────────────────────
# Kraken writes every depth into ONE deck and the .shd reader splits it, so
# this is a single binary launch; RAM marches once per depth. Either way the
# contract is the same: a ResultStack over source_depth whose slabs are the
# unit-amplitude field of one source.
stack = kraken.run(env, array, receiver)
fleet_stack = kraken.run(env, fleet, receiver)
print(f"Kraken: {stack.n_slabs} slabs over source_depth "
      f"{np.round(stack.coordinate, 1).tolist()}")

# env= puts each panel in the waveguide it was computed in, and the grid
# marks each panel's OWN source — the stacking coordinate is source_depth, so
# 'one source at a time' is exactly what a panel shows.
fig, _ = fleet_stack.plot(
    env=env, title='One source at a time — the slabs a total is built from')
fig.savefig(OUT / 'example_40_slabs.png', dpi=140, bbox_inches='tight')
plt.close(fig)

# The same Source on a PE gives the same shape of answer through the looped
# path, which is what makes the two engines comparable cell by cell.
ram_stack = uacpy.RAM(verbose=False).run(env, array, receiver)
print(f"RAM   : {ram_stack.n_slabs} slabs, looped one march per depth")

# ── 2. the two totals ───────────────────────────────────────────────────
# coherent: the elements are one array, driven with a fixed relative phase.
# incoherent: the same five sources, mutually incoherent - unrelated tones,
# separate platforms. Same slabs, different physics, different answer.
coherent = stack.superpose()
incoherent = fleet_stack.superpose(coherent=False)

one = kraken.run(env, uacpy.Source(depths=float(depths[2]), frequencies=FREQ,
                                   source_level_dB=180.0), receiver)
med = lambda f: float(np.nanmedian(np.asarray(f.dB)))
print(f"median level, one source        : {med(one):6.2f} dB")
print(f"median level, coherent total    : {med(coherent):6.2f} dB "
      f"({med(one) - med(coherent):+.2f} dB of array gain)")
print(f"median level, incoherent total  : {med(incoherent):6.2f} dB "
      f"({med(one) - med(incoherent):+.2f} dB)")

# ── 3. the level, not the loss ──────────────────────────────────────────
# at_source_level() needs no argument: the Source carried 180 dB, so every
# result of this run knows it. The colourbar says so too - a level is its own
# kind, captioned 'Level (dB re 1 µPa)' and read upward.
fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
# source=array draws a marker at every element depth, so the aperture the
# gain comes from is visible against the field it produced.
uacpy.plot_field(coherent.at_source_level(), ax=axes[0], env=env, source=array,
                 title=f'Coherent array, steered {steer_deg:g} deg')
uacpy.plot_field(incoherent.at_source_level(), ax=axes[1], env=env, source=fleet,
                 title='The same five sources, mutually incoherent')
fig.tight_layout()
fig.savefig(OUT / 'example_40_level.png', dpi=140, bbox_inches='tight')
plt.close(fig)

# ── 4. what the array does to the channel ───────────────────────────────
# The modal excitation is exact here; the free-field pattern is the design
# intuition. A directional element shades both - the engine multiplies the
# modal excitation by f(theta_m) before summing, and f factors out of the
# array sum, which is the product theorem in the modal domain.
element = np.array([[-90.0, -18.0], [-40.0, -8.0], [0.0, 0.0],
                    [40.0, -8.0], [90.0, -18.0]])
shaded = uacpy.Source(depths=depths, frequencies=FREQ, weights=weights,
                      beam_pattern=element, source_level_dB=180.0)
modes = kraken.compute_modes(env, uacpy.Source(depths=depths,
                                               frequencies=FREQ))

fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
uacpy.plot_mode_excitation(modes, array, ax=axes[0], sound_speed=C_REF,
                           title='Omnidirectional elements')
uacpy.plot_mode_excitation(modes, shaded, ax=axes[1], sound_speed=C_REF,
                           title='Directional elements — P = f·A')
fig.tight_layout()
fig.savefig(OUT / 'example_40_array_response.png', dpi=140,
            bbox_inches='tight')
plt.close(fig)

angles = np.linspace(-90.0, 90.0, 721)
A = np.abs(array.array_factor(angles, sound_speed=C_REF))
P = np.abs(shaded.array_beam_pattern(angles, sound_speed=C_REF))
print(f"array factor A peaks at {angles[int(np.argmax(A))]:+.1f} deg "
      f"(steered to {steer_deg:+.1f})")
print(f"array beam pattern P = f*A peaks at "
      f"{angles[int(np.argmax(P))]:+.1f} deg — the element pattern pulls the "
      f"lobe back toward broadside")
print(f"\nFigures written to {OUT}")
