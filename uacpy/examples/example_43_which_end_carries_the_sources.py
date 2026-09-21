"""Which end of the path carries the sources, and what getting it wrong costs.

A fixed receive array listening for a target somewhere in range is two
points and a path between them. The model wants a source at r = 0 and
receivers out at range, so there are two ways to set the problem up, and on
a range-independent bottom they give the same answer to 0.0 dB. That
agreement is a trap: it holds because such a guide depends on the
separation alone, so the wrong arrangement passes every test until the day
someone runs it over a slope.

Call this example's arrangement (a): the SOURCES sit at the array's own
element depths, at r = 0 where the array physically is, and the receivers
cover the candidate-target plane. Reciprocity — "if source and receiver are
interchanged, the field remains the same" (Jensen, Kuperman, Porter &
Schmidt, *Computational Ocean Acoustics*; the interchange theorem is
Pierce's 4.9.3, and it holds in a lossy medium too, his 4.9.1) — makes slab
i the field element i would measure from a target at each point.

Call (b) the mirror: source at the target depth, receiver on the array.
That is a different problem. It stands the array on the far side of the
sill instead of on the shelf where it sits.

Measuring the cost of (b) needs a THIRD field, because the obvious
difference (a) - (b) confounds two effects. Let (c) be the true reciprocal
of (a): the same physics read from the other end, which means the source at
the target depth on a section reversed **about that range**. Reversing the
whole 20 km polyline instead would put a different stretch of bottom under
any path shorter than 20 km — the reversal has to be about each range
separately. Then

    (b) vs (c)   the PHYSICS: the array on the wrong side of the sill
    (a) vs (c)   the SOLVER's own failure to be reciprocal
    (a) vs (b)   the difference of the two, which is what a naive
                 comparison prints

The third is the one to distrust. It understates the physics and hides the
solver residual inside itself, which is the kind of number that looks
conservative and is merely confounded.

Kraken with coupled modes throughout: the adiabatic approximation "assumes
that all energy in a given mode transfers to the corresponding mode in the
new environment, provided that environmental variations in range are
gradual" (Etter, *Underwater Acoustic Modeling and Simulation*, 4.4.5), and
a sill that cuts the steeper modes off leaves no corresponding mode to
transfer to. uacpy's own KRAKEN notes put the coupled-mode reciprocity
residual at "about 1 dB", which is where the measurement below lands.

Reciprocity also exchanges the ENDS' properties, not just their places, so
once the sources sit on the array a ``Source.beam_pattern`` is the
elements' receive directivity — the only route to it, since ``Receiver``
carries no pattern and a modelled pressure has already summed the arrivals.

Uses: Environment.plot(source_marker_range_m=) · Kraken(mode_coupling=) ·
Bathymetry.eval · Source(beam_pattern=) as element directivity · a
range-dependent section reversed about each range
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

FREQ, C_REF = 200.0, 1500.0
HALF = 0.5 * C_REF / FREQ

# ── the section ─────────────────────────────────────────────────────────
# A shelf under the array, a sill rising to 130 m, a basin behind it, then
# a shoaling run. A monotonic slope would not do: it never cuts a mode off
# and then gives it back, and it cannot put a feature between the two ends.
BATHY_RANGES = np.linspace(0.0, 20000.0, 41)
BATHY_DEPTHS = np.interp(BATHY_RANGES,
                         [0.0, 3000.0, 6000.0, 9000.0,
                          11000.0, 13500.0, 16500.0, 20000.0],
                         [200.0, 200.0, 160.0, 130.0,
                          150.0, 175.0, 145.0, 120.0])
env = uacpy.Environment(
    name='sill-and-basin',
    bathymetry=list(zip(BATHY_RANGES, BATHY_DEPTHS)),
    ssp=[(0.0, 1520.0), (30.0, 1518.0), (80.0, 1502.0), (200.0, 1498.0)],
    bottom=uacpy.Bottom.from_halfspace(uacpy.BoundaryProperties(
        acoustic_type='half-space', sound_speed=1700.0, density=1.9,
        attenuation=0.5)),
    absorption=uacpy.Thorp(),
)
flat = uacpy.Environment(name='flat-control', bathymetry=200.0, ssp=env.ssp,
                         bottom=env.bottom, absorption=uacpy.Thorp())

n_el = 24
elements = 120.0 + HALF * (np.arange(n_el) - (n_el - 1) / 2.0)
target = uacpy.Source(depths=60.0, frequencies=FREQ)
array_as_sources = uacpy.Source(depths=elements, frequencies=FREQ)
kraken = uacpy.Kraken(verbose=False, mode_coupling='coupled')

fig, ax = env.plot(source=target,
                   receiver=uacpy.Receiver(depths=elements, ranges=[0.0]),
                   source_marker_range_m=9000.0,
                   title='The array is on the shelf; the target is beyond '
                         'the sill. Swapping the two ends is a different '
                         'problem.')
fig.savefig(OUT / 'example_43_scene.png', dpi=140, bbox_inches='tight')
plt.close(fig)

# ── the three fields, at a handful of separations ───────────────────────
probe_ranges = np.linspace(4000.0, 20000.0, 6)
step = float(BATHY_RANGES[1] - BATHY_RANGES[0])
solver, physical, naive, flat_gap = [], [], [], []
for R in probe_ranges:
    rx_array = uacpy.Receiver(depths=elements, ranges=[float(R)])
    rx_target = uacpy.Receiver(depths=[60.0], ranges=[float(R)])
    # (a) sources on the array, receiver at the target: this is the one to use
    # a multi-depth Source returns one slab per depth, so stack them:
    p_a = np.array([np.asarray(f.data).ravel()[0]
                    for _, f in kraken.run(env, array_as_sources,
                                           rx_target)])
    # (b) the mirror: source at the target depth, receivers on the array
    p_b = np.asarray(kraken.run(env, target, rx_array).data)[:, 0]
    # (c) the true reciprocal of (a): the section reversed ABOUT THIS RANGE,
    # sampled at the same node spacing so the comparison measures
    # reciprocity and not a difference in how finely each side is drawn.
    walk = np.arange(0.0, R + 0.5 * step, step)
    reversed_env = uacpy.Environment(
        name='reversed-about-R',
        bathymetry=list(zip(walk, env.bathymetry.eval(range=R - walk))),
        ssp=env.ssp, bottom=env.bottom, absorption=uacpy.Thorp())
    p_c = np.asarray(kraken.run(reversed_env, target, rx_array).data)[:, 0]
    tl_a, tl_b, tl_c = (-10.0 * np.log10(np.mean(np.abs(q) ** 2))
                        for q in (p_a, p_b, p_c))
    solver.append(abs(tl_a - tl_c))
    physical.append(abs(tl_b - tl_c))
    naive.append(abs(tl_a - tl_b))
    # the control: on a flat bottom (a) and (b) must agree exactly
    f_a = np.array([np.asarray(f.data).ravel()[0]
                    for _, f in kraken.run(flat, array_as_sources,
                                           rx_target)])
    f_b = np.asarray(kraken.run(flat, target, rx_array).data)[:, 0]
    flat_gap.append(abs(-10.0 * np.log10(np.mean(np.abs(f_a) ** 2))
                        + 10.0 * np.log10(np.mean(np.abs(f_b) ** 2))))
solver, physical, naive, flat_gap = map(
    np.asarray, (solver, physical, naive, flat_gap))

print(f"over {probe_ranges.size} separations from "
      f"{probe_ranges[0] / 1e3:.0f} to {probe_ranges[-1] / 1e3:.0f} km, "
      f"element-averaged TL:")
print(f"  the array on the wrong side of the sill costs "
      f"{np.median(physical):.1f} dB median, {np.max(physical):.1f} dB worst")
print(f"  the solver's own non-reciprocity is "
      f"{np.median(solver):.2f} dB median, {np.max(solver):.2f} dB worst "
      f"— a factor {np.median(physical) / np.median(solver):.0f}")
print(f"  the naive (a)-(b) difference, {np.median(naive):.1f} dB, is those "
      f"two partly cancelling")
print(f"  on a FLAT bottom the same pair agrees to "
      f"{np.max(flat_gap):.4f} dB at every range — which is why a "
      f"range-independent\n  example cannot tell the two constructions apart")

# ── and what else rides along with the sources: DIRECTIVITY ─────────────
# Reciprocity exchanges the two ends completely, not just their positions.
# A directional source at A heard by an omni receiver at B is the same
# field as an omni source at B heard by a DIRECTIONAL receiver at A. So
# putting the sources on the array buys per-element receive directivity for
# free: give that Source a beam_pattern and it is the element's response.
#
# There is no other way to get it. Receiver carries depths and ranges and
# no pattern, and a modelled pressure has already summed the arrivals, so
# an element response cannot be applied after the fact — each mode leaves
# at its own angle and must be weighted before the sum, which is what the
# source beam pattern does.
pattern_angles = np.linspace(-90.0, 90.0, 361)
rx_probe = uacpy.Receiver(depths=[60.0], ranges=[5000.0])
print("\nper-element receive directivity, by reciprocity (5 km, 60 m):")
for label, response_dB in (
        ('omnidirectional elements', None),
        ('cardioid', 20.0 * np.log10(np.maximum(
            0.5 * (1.0 + np.cos(np.deg2rad(pattern_angles))), 1e-1))),
        ('steep angles rejected 20 dB',
         np.where(np.abs(pattern_angles) > 15.0, -20.0, 0.0)),
        ('ONLY steep angles kept',
         np.where(np.abs(pattern_angles) < 15.0, -40.0, 0.0))):
    shaded = uacpy.Source(
        depths=elements, frequencies=FREQ,
        beam_pattern=None if response_dB is None
        else np.column_stack([pattern_angles, response_dB]))
    q = np.array([np.asarray(f.data).ravel()[0]
                  for _, f in kraken.run(env, shaded, rx_probe)])
    print(f"  {label:28s} element-mean TL "
          f"{-10.0 * np.log10(np.mean(np.abs(q) ** 2)):6.2f} dB")
print("  the shallow modes carry the energy, so rejecting the steep ones "
      "costs little\n  and keeping only them costs a great deal — the "
      "pattern weights each mode by\n  the angle it leaves at, which is "
      "why it has to be applied before the modal sum")

# ── the figure ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

ax = axes[0]
ax.plot(BATHY_RANGES / 1e3, BATHY_DEPTHS, 'k-', lw=1.6)
ax.fill_between(BATHY_RANGES / 1e3, BATHY_DEPTHS, 210.0, color='#d9c6a5')
ax.plot(np.zeros_like(elements), elements, 'o', color='C2', ms=4,
        label='array, where it is (r = 0, 200 m of water)')
ax.plot([9.0], [60.0], '*', color='C3', ms=16, label='target, beyond the sill')
ax.plot([9.0] * n_el, elements, 'x', color='C1', ms=4,
        label='array, where the MIRROR puts it')
ax.set(xlabel='Range (km)', ylabel='Depth (m)', ylim=(210, 0),
       title='Two arrangements of the same two points')
ax.legend(fontsize=7, loc='lower left')
ax.grid(alpha=0.3)

ax = axes[1]
km = probe_ranges / 1e3
ax.plot(km, physical, 'C3o-', lw=1.4, label='(b) vs (c): the physics')
ax.plot(km, naive, 'C0s--', lw=1.2, label='(a) vs (b): the naive difference')
ax.plot(km, solver, 'C7^:', lw=1.2, label="(a) vs (c): the solver's residual")
ax.plot(km, flat_gap, 'k.-', lw=1.0, label='the same pair, flat bottom')
ax.set(xlabel='Separation (km)', ylabel='|TL difference| (dB)',
       title='The naive difference is the physics minus the residual')
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / 'example_43_reciprocity.png', dpi=140, bbox_inches='tight')
plt.close(fig)
print(f"\nFigures written to {OUT}")
