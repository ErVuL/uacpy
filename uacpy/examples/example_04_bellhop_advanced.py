"""Bellhop — the full feature set on one continental shelf.

Six runs over the same range-dependent environment, each turning on one thing:
Gaussian beams with Thorp volume attenuation, Cerveny beams with their own
width/curvature controls and beam shift, a line source instead of a point
source, three source depths in a single binary call, a ray trace, and a
directional source driven by a beam-pattern table.

The water column is the top 500 m of the Munk profile. Munk's sound-channel
axis is at 1300 m, below this domain, so over 0-500 m the profile only
decreases (1548.5 → 1513.2 m/s): downward refraction, no channel.

Uses: Bellhop(beam_type=, grid_type=, n_beams=, alpha=, beam_shift=,
beam_width_type=, beam_curvature=, eps_multiplier=, r_loop=, n_image=, ib_win=)
· Source(source_type='line') · Source(beam_pattern=) ·
Source.plot_beam_pattern · multi-depth Source → ResultStack ·
RunMode.RAYS · Rays.plot(color_by=) · uacpy.Thorp · Bottom.from_halfspaces ·
plot.shared_colorbar
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


bathymetry = np.array([[0, 100], [10000, 150], [20000, 300], [30000, 500]])
env = uacpy.Environment(
    name="Continental shelf — Munk profile (upper 500 m)",
    ssp=uacpy.SoundSpeedProfile.from_munk(500.0),
    bathymetry=bathymetry,
    bottom=uacpy.Bottom.from_halfspaces(
        bathymetry[:, 0].astype(float),
        sound_speed=np.array([1600, 1650, 1700, 1750]),   # hardening
        density=np.array([1.5, 1.7, 1.9, 2.1]),
        attenuation=np.array([0.8, 0.6, 0.4, 0.3]),       # less lossy
        shear_speed=np.zeros(4),
        acoustic_type='half-space'),
    absorption=uacpy.Thorp(),
)
source = uacpy.Source(depths=75.0, frequencies=100.0)
receiver = uacpy.Receiver(depths=np.linspace(10, 450, 50),
                          ranges=np.linspace(100, 30000, 150))

gaussian = uacpy.Bellhop(beam_type='B', grid_type='R', n_beams=500,
                         alpha=(-85, 85)).run(
    env, source, receiver, run_mode=uacpy.RunMode.COHERENT_TL)

cerveny = uacpy.Bellhop(
    beam_type='C', grid_type='R', n_beams=500, alpha=(-85, 85),
    beam_width_type='M', beam_curvature='Z', eps_multiplier=0.7,
    r_loop=10000.0, n_image=2, ib_win=4, beam_shift=True).run(
    env, source, receiver, run_mode=uacpy.RunMode.COHERENT_TL)

# Geometry lives on the Source, not the model: same Bellhop, different source.
# 'line' is an infinite coherent line source (Cartesian spreading) rather than
# a point source (cylindrical).
line = uacpy.Bellhop(beam_type='B', grid_type='R', n_beams=500).run(
    env, uacpy.Source(depths=source.depths, frequencies=source.frequencies,
                      source_type='line'),
    receiver, run_mode=uacpy.RunMode.COHERENT_TL)

# Three source depths in ONE binary call — the Bellhop binary loops the source
# axis natively. The shelf is 100 m deep at r=0, so every source must sit in
# the water column there.
stack = uacpy.Bellhop(n_beams=500, alpha=(-85, 85)).run(
    env, uacpy.Source(depths=[20.0, 50.0, 80.0], frequencies=300.0),
    receiver, run_mode=uacpy.RunMode.COHERENT_TL)
# A ResultStack of Field slabs: iterate for (source_depth, slab) pairs, or
# stack.at(source_depth=z) for one 2-D Field. The slab accessors (.dB, .p,
# .at(depth=, range=)) live on the Field, not on the stack.
print(f"  ResultStack of {stack.n_slabs} {stack.slab_type.__name__} slabs")
for depth, slab in stack:
    finite = np.asarray(slab.dB)[np.isfinite(slab.dB)]   # NaN = no ray reached
    print(f"    source depth {depth:5.1f} m → median TL "
          f"{np.median(finite):.1f} dB")

rays = uacpy.Bellhop(beam_type='g', grid_type='R', n_beams=50,
                     alpha=(-80, 80), beam_shift=True).run(
    env, source, receiver, run_mode=uacpy.RunMode.RAYS)

# A beam pattern is an (angle_deg, level_dB re peak) table. The angle axis is
# Bellhop's launch declination α, so POSITIVE IS DOWNWARD (ray2D(1)%t =
# [COS(α), SIN(α)]/c, bellhop.f90:453, over a depth axis that increases
# downward): this beam is tilted below the horizontal, aimed down the shelf.
#
# A projector is specified by its beamwidth, and beam_pattern is an angular
# weighting on the launch amplitude, so the table is a main lobe BEAMWIDTH wide
# between its -3 dB points with the nulls and sidelobes a smooth lobe implies.
# sinc is -3 dB at 0.442946, so that constant sets the width; the nulls are
# true zeros, so the levels are floored to keep the table finite.
TILT_DEG, BEAMWIDTH_DEG, FLOOR_DB = 20.0, 24.0, -40.0
pattern_angles = np.linspace(-90.0, 90.0, 721)
lobe = 0.442946 * (pattern_angles - TILT_DEG) / (0.5 * BEAMWIDTH_DEG)
pattern_levels = 20.0 * np.log10(
    np.maximum(np.abs(np.sinc(lobe)), 10.0 ** (FLOOR_DB / 20.0)))
# The table has to cover every launch angle α spans. Bellhop clamps the table
# index but not the interpolation weight (bellhop.f90:269-274), so a table that
# stops short is EXTRAPOLATED on linear amplitude — the outer beams come back
# louder than declared and phase-inverted — and uacpy refuses the run rather
# than let that reach the field.
directional_source = uacpy.Source(
    depths=source.depths, frequencies=source.frequencies,
    beam_pattern=np.column_stack([pattern_angles, pattern_levels]))
directional = uacpy.Bellhop(beam_type='B', grid_type='R', n_beams=500,
                            alpha=(-85, 85)).run(
    env, directional_source, receiver, run_mode=uacpy.RunMode.COHERENT_TL)

fig, _ = env.plot()
fig.savefig(OUT / 'example_04_environment.png', dpi=150, bbox_inches='tight')
plt.close(fig)

fig, (left, right) = plt.subplots(1, 2, figsize=(16, 6))
uacpy.plot_field(gaussian, left, env=env, show_colorbar=False,
                 contours=[70, 85, 100],
                 title='Standard Gaussian beams\n(with Thorp attenuation)')
uacpy.plot_field(cerveny, right, env=env, show_colorbar=False,
                 contours=[70, 85, 100],
                 title='Cerveny beams, minimum width\n(with beam shift)')
uacpy.plot.shared_colorbar(fig, (left, right), label='TL (dB)')
fig.suptitle('Gaussian vs Cerveny beams', fontsize=16, fontweight='bold')
fig.savefig(OUT / 'example_04_beam_comparison.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)

fig, (left, right) = plt.subplots(1, 2, figsize=(16, 6))
uacpy.plot_field(gaussian, left, env=env, show_colorbar=False,
                 title="Point source (cylindrical)\nRunType: 'CB RR  '")
uacpy.plot_field(line, right, env=env, show_colorbar=False,
                 title="Line source (Cartesian)\nRunType: 'CB XR  '")
uacpy.plot.shared_colorbar(fig, (left, right), label='TL (dB)')
fig.suptitle('Point vs line source', fontsize=16, fontweight='bold')
fig.savefig(OUT / 'example_04_source_comparison.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)

fig, ax = rays.plot(env=env, color_by="bounces")
ax.set_title("Ray trace with beam shift\nRunType: 'Rg RR S' — "
             "rays coloured by bounce type")
fig.savefig(OUT / 'example_04_rays.png', dpi=150, bbox_inches='tight')
plt.close(fig)

fig, axes = plt.subplots(1, stack.n_slabs, figsize=(6 * stack.n_slabs, 5))
for ax, (depth, slab) in zip(np.atleast_1d(axes), stack):
    uacpy.plot_field(slab.to_dB(), ax, env=env, show_colorbar=False,
                     title=f'Source depth = {depth:.0f} m')
    # The source, at r = 0 km and its own depth; TL panels use km on x, m on y.
    ax.plot(0.0, depth, marker='*', markersize=18, color='white',
            markeredgecolor='black', markeredgewidth=1.2, zorder=10,
            clip_on=False)
uacpy.plot.shared_colorbar(fig, axes, label='TL (dB)')
fig.suptitle('Multi-source-depth: one binary call, one ResultStack',
             fontsize=15, fontweight='bold')
fig.savefig(OUT / 'example_04_multi_source.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# The directivity beside the field it produces. The polar axes are oriented
# like the TL panels next to them — 0° along increasing range, positive angles
# downward — so the lobe points at the water it ensonifies.
fig = plt.figure(figsize=(18, 5.5))
directional_source.plot_beam_pattern(
    ax=fig.add_subplot(1, 3, 1, projection='polar'),
    title=f'Source directivity\n{BEAMWIDTH_DEG:.0f}° beam aimed at '
          f'{TILT_DEG:.0f}°')
omni_ax = fig.add_subplot(1, 3, 2)
uacpy.plot_field(gaussian, omni_ax, env=env, show_colorbar=False,
                 title='Omnidirectional source\n(beam_pattern=None)')
dir_ax = fig.add_subplot(1, 3, 3)
uacpy.plot_field(directional, dir_ax, env=env, show_colorbar=False,
                 title="Directional source\n(.sbp, RunType(3:3) = '*')")
# Room for the two-line panel titles: add_subplot fills more of the figure than
# plt.subplots leaves, so the default top margin puts the suptitle through them.
# This has to come BEFORE the colorbar: the bar takes its space from the panels
# as they stand, and a later subplots_adjust moves the panels back over it.
fig.subplots_adjust(top=0.74)
uacpy.plot.shared_colorbar(fig, (omni_ax, dir_ax), label='TL (dB)')
fig.suptitle('Source directivity shapes the field', fontsize=15,
             fontweight='bold')
fig.savefig(OUT / 'example_04_beam_pattern.png', dpi=150, bbox_inches='tight')
plt.close(fig)
