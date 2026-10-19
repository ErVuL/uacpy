"""Propagation to detection, end to end, on one receive array.

A 24-element vertical array at 120 m listens for a target at 60 m on a
range-dependent shelf, at 200 Hz. One coupled-mode Kraken run with the sources
on the array's element depths gives, by reciprocity, what every element hears
from a target anywhere on the plane. The beamformer's realised array gain then
enters the passive sonar equation as a grid, next to the constant plane-wave
value:

    SE = SL - TL - (NL - AG) - DT

docs/guide/sonar.md §2 explains the array-gain grid, docs/guide/arrays.md the
count of independent looks, and docs/guide/source-receiver.md §7 why the
sources go on the array.

Uses: Environment.plot(source=, receiver=, source_marker_range=) ·
Kraken(mode_coupling='coupled') · Field.p over source_depth · beamform_field ·
plane_wave_array_gain · matched_replica_gain · independent_beams(weights=) ·
shading_taper · sonar.detection_threshold_energy · per_look_false_alarm ·
passive_signal_excess_field(array_gain_dB=<grid>) ·
transition_probability_field · detection_ranges_field ·
plot_field / plot_detection_probability with env/receiver overlays ·
draw_contours (a second field's contour over a map)
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.plot import draw_contours, plot_detection_probability, plot_field
from uacpy.acoustic_signal import (beamform_field, independent_beams,
                                   matched_replica_gain,
                                   plane_wave_array_gain, shading_taper)
from uacpy.sonar import (detection_ranges_field, detection_threshold_energy,
                         passive_signal_excess_field, per_look_false_alarm,
                         transition_probability_field)

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

FREQ, C_REF = 200.0, 1500.0          # Hz, m/s
N_EL, ARRAY_DEPTH = 24, 120.0        # elements at lambda/2 about this depth (m)
TARGET_DEPTH = 60.0                  # m
SL, NL = 120.0, 75.0                 # band-integrated, dB re 1 µPa²
PD, PF = 0.5, 1e-4                   # Pf of the whole scan
BW_HZ, INT_S = 10.0, 10.0
SIGMA_DB = 8.0

# Env definition: a shelf, a sill at 130 m, a basin at 175 m, a shoaling run
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

# Array at r = 0, target, and the plane of candidate target positions
elements = ARRAY_DEPTH + 0.5 * C_REF / FREQ * (np.arange(N_EL) - (N_EL - 1) / 2.0)
target = uacpy.Source(depths=TARGET_DEPTH, frequencies=FREQ, source_level_dB=SL)
array_at_origin = uacpy.Receiver(depths=elements, ranges=[0.0])
ranges = np.linspace(600.0, 20000.0, 196)   # past 0.6 km the modes hold every path
plane_depths = np.union1d(np.linspace(5.0, 195.0, 60), [TARGET_DEPTH])
plane = uacpy.Receiver(depths=plane_depths, ranges=ranges)
in_water = plane_depths[:, None] <= np.asarray(env.bathymetry.eval(range=ranges))
i_target = int(np.argmin(np.abs(plane_depths - TARGET_DEPTH)))

fig, ax = env.plot(source=target, receiver=array_at_origin,
                   source_marker_range=5000.0,
                   title=f'The scene: the {N_EL}-element array at '
                         f'{ARRAY_DEPTH:.0f} m, a {SL:g} dB target 5 km up '
                         f'the slope at {TARGET_DEPTH:.0f} m')
fig.savefig(OUT / 'example_42_scene.png', dpi=140, bbox_inches='tight')
plt.close(fig)

# Propagation: sources at the element depths, receivers over the target plane
p_map = uacpy.Kraken(mode_coupling='coupled').run(
    env, uacpy.Source(depths=elements, frequencies=FREQ), plane).p

# Array gain: the plane-wave value, the matched-replica ceiling, the realised one
taper = shading_taper(N_EL, 'hann')
angles = np.linspace(-45.0, 45.0, 361)
ag_plane = plane_wave_array_gain(taper)
ag_matched = matched_replica_gain(p_map[:, i_target, :])
row = beamform_field(p_map[:, i_target, :], elements, angles, FREQ,
                     sound_speed=C_REF, weights=taper)
ag_row = row.array_gain()
print(f"array gain: plane wave {ag_plane:.2f} dB, matched replica "
      f"{np.median(ag_matched):.2f} dB, realised at {TARGET_DEPTH:.0f} m "
      f"{np.median(ag_row):.1f} dB (median over range)")

# Detection threshold holding the whole scan at PF
n_looks = independent_beams(elements, angles, FREQ, sound_speed=C_REF,
                            weights=taper)
dt = detection_threshold_energy(pd=PD, pf=per_look_false_alarm(PF, n_looks),
                                bandwidth=BW_HZ, integration_time=INT_S)
print(f"{n_looks:.1f} independent looks -> DT = {dt:.2f} dB")

# Sonar equation over the plane, with the constant and the realised array gain
scan = beamform_field(p_map, elements, angles, FREQ, sound_speed=C_REF,
                      weights=taper)
with np.errstate(divide='ignore', invalid='ignore'):
    ag_map = np.where(in_water, scan.array_gain(), np.nan)
    tl_map = np.where(in_water, -10.0 * np.log10(scan.element_power), np.nan)
print(f"realised AG over the water: median {np.nanmedian(ag_map):.1f} dB")
tl = uacpy.Field(data=tl_map, coords={'depth': plane_depths, 'range': ranges},
                 model='Kraken')
se, pd_map = {}, {}
for label, gain in (('constant', ag_plane), ('realised', ag_map)):
    se[label] = passive_signal_excess_field(
        tl, source_level_dB=SL, noise_level_dB=NL, array_gain_dB=gain,
        detection_threshold_dB=dt)
    pd_map[label] = transition_probability_field(se[label], sigma_dB=SIGMA_DB)
    r_by_z = detection_ranges_field(se[label])
    covered = 100.0 * np.mean(np.asarray(se[label].data)[in_water] > 0.0)
    print(f"{label:>8} AG: detection range at {TARGET_DEPTH:.0f} m "
          f"{r_by_z[i_target] / 1e3:.1f} km, {covered:.0f} % of the water "
          f"above threshold")

# Plots
km = ranges / 1e3
fig, axes = plt.subplots(2, 2, figsize=(14, 8.4))
lo, hi = np.nanpercentile(tl_map, [1, 99])
plot_field(tl, ax=axes[0, 0], env=env, receiver=array_at_origin,
           vmin=float(np.floor(lo / 5) * 5), vmax=float(np.ceil(hi / 5) * 5),
           title='1. Target-to-array loss, mean of 24 elements')
draw_contours(axes[0, 0], se['realised'], [0.0], label='SE = 0, realised AG')
draw_contours(axes[0, 0], se['constant'], [0.0], color='C3', ls='--',
              label=f'SE = 0, AG = {ag_plane:.1f} dB')
axes[0, 0].legend(fontsize=7, loc='upper right', framealpha=0.85)

row.plot(ax=axes[0, 1], at=int(np.argmin(np.abs(ranges - 5000.0))),
         color='C0', lw=1.3)
axes[0, 1].set(ylim=(-30, 2), title='2. The arrival fan at 5 km')

ax = axes[1, 0]
ax.plot(km, ag_row, 'C0-', lw=1.3, label='realised by the best beam')
ax.axhline(float(np.median(ag_matched)), color='C4', ls='-.', lw=1.3,
           label=f'matched replica ({np.median(ag_matched):.1f} dB)')
ax.axhline(ag_plane, color='C3', ls='--', lw=1.2,
           label=f'matched plane wave ({ag_plane:.1f} dB)')
ax.set(xlabel='Range (km)', ylabel='Array gain (dB)',
       title=f'3. Array gain on the {TARGET_DEPTH:.0f} m row')
ax.legend(fontsize=8, loc='lower left')
ax.grid(alpha=0.3)

plot_detection_probability(pd_map['realised'], ax=axes[1, 1], env=env,
                           receiver=array_at_origin, contours=(0.5,),
                           title=f'4. Pd vs target position, realised AG '
                                 f'(scan Pf = {PF:g})')
draw_contours(axes[1, 1], pd_map['constant'], [0.5], color='C3', ls='--',
              label='Pd = 0.5 if AG were constant')
axes[1, 1].legend(fontsize=7, loc='upper right', framealpha=0.85)
fig.tight_layout()
fig.savefig(OUT / 'example_42_chain.png', dpi=140, bbox_inches='tight')
plt.close(fig)
print(f"\nFigures written to {OUT}")
