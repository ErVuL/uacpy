"""Bellhop run modes — TL, rays, eigenrays, arrivals.

One model, six run modes, four scenarios.

The three TL modes share one ray trace and differ only in the accumulator:

* COHERENT_TL keeps phase, so the modal interference is there — use it for CW.
* INCOHERENT_TL sums power, giving the smooth long-term average a broadband
  signal sees.
* SEMICOHERENT_TL is NOT a hybrid of the two. The vendored solver accumulates
  'S' exactly as it accumulates 'I' — influence.f90 branches on
  ``CASE ( 'I', 'S' )`` and squares the magnitude for both. The one 'S'-specific
  line is a Lloyd-mirror SOURCE AMPLITUDE pattern applied at launch
  (bellhop.f90:276-278, ``Amp0 * SQRT(2) * ABS(SIN(omega/c * zs * SIN(alpha)))``).
  So 'S' is an incoherent sum with the source's surface-image directivity baked
  in, and its smoothness matches 'I'. The statistics below show that directly,
  and the run times are measured rather than asserted.

Then: a ray fan through the Munk channel, eigenrays found two ways (the
EIGENRAYS run mode and compute_eigenrays, whose Fortran miss tolerance is loose
enough to need filtering), and the arrival structure behind them.

Uses: RunMode.COHERENT_TL / INCOHERENT_TL / SEMICOHERENT_TL / RAYS / EIGENRAYS
/ ARRIVALS · Bellhop.compute_eigenrays · Rays.filter_by_miss_distance /
top_n_by_miss / truncate_at_receiver · Rays.plot · Arrivals.plot · plot.compare
"""

import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

# ── A. The three TL modes on a Munk profile ─────────────────────────────────
munk = uacpy.Environment(name="Munk profile", bathymetry=5000.0,
                         ssp=uacpy.SoundSpeedProfile.from_munk(5000.0))
source = uacpy.Source(depths=1000.0,     # in the channel, above the 1300 m axis
                      frequencies=50.0)
receiver = uacpy.Receiver(depths=np.linspace(100, 4900, 40),
                          ranges=np.linspace(1000, 50000, 80))

# The first Bellhop call in a process pays binary and library load — about
# 1.8 s here — which would be charged entirely to whichever mode ran first.
uacpy.Bellhop().run(munk, source, receiver,
                    run_mode=uacpy.RunMode.COHERENT_TL)

fields, elapsed = {}, {}
for mode in (uacpy.RunMode.COHERENT_TL, uacpy.RunMode.INCOHERENT_TL,
             uacpy.RunMode.SEMICOHERENT_TL):
    label = mode.name.split('_')[0].capitalize()
    started = time.perf_counter()
    fields[label] = uacpy.Bellhop().run(munk, source, receiver, run_mode=mode)
    elapsed[label] = time.perf_counter() - started

# Whole-grid spread is the fair smoothness measure — one depth slice is too
# short a sample to separate 'I' from 'S'.
for label, field in fields.items():
    print(f"  {label:12s} TL std {np.nanstd(field.dB):5.2f} dB, "
          f"{elapsed[label]:.2f} s")
print(f"  max |Incoherent − Semicoherent| = "
      f"{np.nanmax(np.abs(fields['Incoherent'].dB - fields['Semicoherent'].dB)):.2f} dB"
      f" — the Lloyd-mirror launch pattern, and nothing else")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for ax, (label, field) in zip(axes.flat, fields.items()):
    # The default 20-120 dB TL scale, narrowed to where this deep field lives.
    uacpy.plot_field(field, ax, env=munk, source=source, vmin=60, vmax=120,
                     title=f'{label} TL')
cuts = {label: field.at(depth=1000) for label, field in fields.items()}
uacpy.plot.compare(list(cuts.values()), list(cuts), ax=axes[1, 1],
                   linewidth=2.5, alpha=0.8,
                   title='TL at 1000 m (300 m above the channel axis)')
fig.tight_layout()
fig.savefig(OUT / 'example_11a_tl_modes.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ── B. Ray paths through the channel ────────────────────────────────────────
ray_model = uacpy.Bellhop(alpha=(-15.0, 15.0), n_beams=31)
rays = ray_model.run(munk, source,
                     uacpy.Receiver(depths=np.array([1000]),
                                    ranges=np.linspace(0, 100000, 100)),
                     run_mode=uacpy.RunMode.RAYS)
print(f"  {ray_model.n_beams} rays over "
      f"{ray_model.alpha[0]:.0f}° to {ray_model.alpha[1]:.0f}°, out to 100 km")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
rays.plot(env=munk, ax=axes[0],
          title=f'Ray paths ({ray_model.n_beams} rays, ±15° launch)')
axes[0].set_xlim(0, 100)
# 20-40 km is the axial crossing region, not a convergence zone: a CZ is the
# near-surface refocusing of bottom-limited rays from a near-surface source.
# This source sits inside the channel with a ±15° fan, so every ray here is
# channel-trapped and never reaches the bottom.
rays.plot(env=munk, ax=axes[1],
          title='Ray paths — zoom on the axial crossings (20-40 km)')
axes[1].set_xlim(20, 40)
axes[1].set_ylim(2000, 0)
fig.tight_layout()
fig.savefig(OUT / 'example_11b_ray_tracing.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ── C. Eigenrays and arrivals at one receiver ───────────────────────────────
shelf = uacpy.Environment(
    name="Shallow water", bathymetry=100.0,
    ssp=uacpy.SoundSpeedProfile.from_pairs([(0, 1500), (100, 1520)]))
shelf_source = uacpy.Source(depths=50.0, frequencies=100.0)
target = uacpy.Receiver(depths=[30.0], ranges=[2000.0])

# compute_eigenrays returns every ray Bellhop wrote — the Fortran tolerance is
# loose — so the Rays methods filter to the ones that actually land on the
# receiver. A dense launch fan is what makes them converge sharply: with a
# coarse fan the per-angle vertical spacing at 2 km already exceeds the miss
# tolerance.
wavelength = 1500.0 / float(shelf_source.frequencies[0])
eigenrays = uacpy.Bellhop(alpha=(-20.0, 20.0), n_beams=2001).compute_eigenrays(
    shelf, shelf_source, target).filter_by_miss_distance(
    wavelength / 4).top_n_by_miss(12).truncate_at_receiver()
context = uacpy.Bellhop(alpha=(-20.0, 20.0), n_beams=21).run(
    shelf, shelf_source,
    uacpy.Receiver(depths=np.array([30.0]),
                   ranges=np.linspace(0, 2200.0, 50)),
    run_mode=uacpy.RunMode.RAYS)
arrivals = uacpy.Bellhop(alpha=(-20.0, 20.0), n_beams=201).run(
    shelf, shelf_source, uacpy.Receiver(depths=np.array([30.0]),
                                        ranges=np.array([2000.0])),
    run_mode=uacpy.RunMode.ARRIVALS)
delays = [record['delay'] for record in arrivals.arrivals]
print(f"  at 2 km / 30 m: {len(eigenrays.rays)} eigenrays within λ/4 "
      f"({wavelength / 4:.1f} m), {len(delays)} arrivals spread over "
      f"{max(delays) - min(delays):.4f} s")

fig = plt.figure(figsize=(14, 13))
grid = fig.add_gridspec(3, 1, hspace=0.45)
context.plot(env=shelf, ax=fig.add_subplot(grid[0]), linewidth=1.1, alpha=0.75,
             title=f'Context ray fan ({len(context.rays)} rays, ±20°)')
eigenrays.plot(env=shelf, ax=fig.add_subplot(grid[1]), linewidth=1.5,
               alpha=0.9,
               title=f'{len(eigenrays.rays)} eigenrays at the receiver '
                     f'(miss < λ/4 ≈ {wavelength / 4:.1f} m)')
arrivals.plot(ax=fig.add_subplot(grid[2]))
fig.savefig(OUT / 'example_11c_eigenrays_arrivals.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)

# ── D. The same eigenray API on a clean Pekeris guide ───────────────────────
pekeris = uacpy.Environment(
    name='Pekeris', bathymetry=100.0, ssp=1500.0,
    bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                    sound_speed=1600.0, density=1.5,
                                    attenuation=0.5))
pekeris_source = uacpy.Source(depths=20.0, frequencies=200.0)
target_range, target_depth = 3000.0, 80.0
paths = uacpy.Bellhop(alpha=(-30, 30), n_beams=2001).compute_eigenrays(
    pekeris, pekeris_source,
    uacpy.Receiver(depths=[target_depth], ranges=[target_range])
).top_n_by_miss(8).truncate_at_receiver()

# The multipath structure reads straight off the table: launch angle, how close
# it came, and how many times it touched each boundary.
print(f"  Pekeris guide, {len(paths.rays)} eigenrays at 3 km / 80 m:")
print(f"    {'α (deg)':>10s} {'miss (m)':>9s} {'top':>4s} {'bot':>4s}")
for ray in paths.rays:
    print(f"    {ray['alpha']:>10.3f} {ray['miss_distance_m']:>9.3f} "
          f"{ray['n_top_bounces']:>4d} {ray['n_bot_bounces']:>4d}")

fig, ax = paths.plot(env=pekeris)
ax.plot(target_range / 1000.0, target_depth, 'ro', markersize=10,
        markeredgecolor='black', label='Receiver')
ax.legend(loc='lower right')
fig.savefig(OUT / 'example_11d_compute_eigenrays_pekeris.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)
