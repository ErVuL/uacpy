"""Kraken — adiabatic modes over a continental shelf.

A guide that changes along range: 100 m of shelf falling to 400 m of slope,
with the seabed hardening from sand to rock and Francois-Garrison volume
attenuation throughout. Kraken solves it by splitting the range into segments,
solving the modes in each, and propagating them adiabatically — each mode
travelling independently, with no energy transferred between them. (The
counterpart, mode_coupling='coupled', does solve for that transfer, at much
higher cost.)

The modes are also computed at each end on their own, so the shelf set and the
slope set can be compared directly. The slope bottom has shear, so it needs
krakenc, and it admits Scholte/Stoneley interface waves whose phase speed sits
below the water sound speed — those are filtered out of the shape panel so the
two sides are comparable.

A last run drives two sources in antiphase through the same segmented guide.
The mode sum is linear in the source amplitude, so the stack the run returns
adds as a vertical dipole: a mode whose shape takes the same value at both
depths is not excited at all.

Uses: uacpy.FrancoisGarrison on the Environment · Bottom.from_halfspaces ·
Bottom.halfspace_at · Kraken.compute_modes · Kraken(backend='krakenc') ·
Kraken(mode_coupling='adiabatic', n_segments=) ·
multi-depth Source(weights=) → ResultStack · ResultStack.superpose ·
Modes.excitation · Modes.phase_speeds · Modes.plot(show_imaginary=) ·
plot_mode_wavenumbers · plot_modes_heatmap · plot_field(contours=, source=)
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

N_SEGMENTS = 5          # read by the run, the summary and the plot markers

bathymetry = np.array([[0, 100], [8000, 120], [10000, 150],
                       [15000, 250], [20000, 400]])
# The seabed's property nodes deliberately sit on a different range vector than
# the bathymetry: they are independent fields in uacpy, and geologically the
# seafloor shape and the sediment-to-rock transition are set by different
# processes. The rock of the 18 km node continues to the 20 km end of the
# section, so the last node repeats it there.
bottom = uacpy.Bottom.from_halfspaces(
    np.array([0.0, 6000.0, 12000.0, 18000.0, 20000.0]),
    sound_speed=np.array([1600, 1650, 1750, 1800, 1800]),   # hardening
    density=np.array([1.5, 1.7, 2.0, 2.2, 2.2]),            # compacting
    attenuation=np.array([0.8, 0.5, 0.3, 0.2, 0.2]),        # less lossy
    shear_speed=np.array([0, 0, 400, 600, 600]),            # rock on the slope
    acoustic_type='half-space')
ssp_pairs = np.array([[0, 1520], [50, 1505], [100, 1495],
                      [200, 1490], [400, 1485]])
# Francois-Garrison lives on the Environment, so the same volume attenuation
# acts in the mode computations and in the segmented TL run.
absorption = uacpy.FrancoisGarrison(temperature=10.0, salinity=35.0,
                                    pH=8.0)

env = uacpy.Environment(name="Continental shelf", bathymetry=bathymetry,
                        ssp=uacpy.SoundSpeedProfile.from_pairs(ssp_pairs),
                        bottom=bottom, absorption=absorption)
source = uacpy.Source(depths=50.0, frequencies=50.0)
# Ranges start at 1 km: closer in, the deepest receivers see direct and
# surface-reflected paths steeper than the modes' 25° window carries.
receiver = uacpy.Receiver(depths=np.linspace(5, 380, 60),
                          ranges=np.linspace(1000, 20000, 96))

# The two ends of the guide, each as a range-independent environment.
shelf_env = uacpy.Environment(
    name="Shelf (100 m)", bathymetry=100.0,
    ssp=uacpy.SoundSpeedProfile.from_pairs(ssp_pairs[ssp_pairs[:, 0] <= 100]),
    bottom=bottom.halfspace_at(range=0, interp='nearest'),
    absorption=absorption)
slope_env = uacpy.Environment(
    name="Slope (400 m)", bathymetry=400.0,
    ssp=uacpy.SoundSpeedProfile.from_pairs(ssp_pairs),
    bottom=bottom.halfspace_at(range=20000, interp='nearest'),
    absorption=absorption)

# Francois & Garrison fitted their equation from 200 Hz up, so at this 50 Hz
# every run evaluates it below its fitted band, and says so. The notice is
# printed once here and filtered for the runs that follow.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    shelf_modes = uacpy.Kraken().compute_modes(shelf_env, source)
for warning in caught:
    if 'below 200 Hz' in str(warning.message):
        print(f"  noted: {str(warning.message).split(' — ')[0]}")
    else:
        warnings.showwarning(warning.message, warning.category,
                             warning.filename, warning.lineno)
warnings.filterwarnings('ignore', message='FrancoisGarrison: .* below 200 Hz',
                        category=uacpy.ValidityWarning)
# The slope bottom has shear, which is what krakenc adds over kraken.
slope_modes = uacpy.Kraken(backend='krakenc').compute_modes(slope_env, source)
print(f"  shelf (cs={shelf_env.bottom.halfspace_at(range=0).shear_speed:.0f} "
      f"m/s): {len(shelf_modes.k)} modes; "
      f"slope (cs={slope_env.bottom.halfspace_at(range=0).shear_speed:.0f} "
      f"m/s): {len(slope_modes.k)} modes")

tl = uacpy.Kraken(mode_coupling='adiabatic', n_segments=N_SEGMENTS).run(
    env, source, receiver)
# result.data is complex pressure; min/max on complex is numpy's lexicographic
# order, not a TL bound, so TL is read off .dB.
print(f"  adiabatic run over {N_SEGMENTS} segments: TL "
      f"{np.nanmin(tl.dB):.1f} to {np.nanmax(tl.dB):.1f} dB")

# Two sources 20 m apart with weights [1, -1]: the run returns a ResultStack
# holding the unit-amplitude field of each depth, and superpose() adds them
# with the Source's weights — Σ wᵢ·pᵢ as one complex Field. Each shelf mode
# is excited in proportion to φ_m(40) − φ_m(60), so the modes whose shapes
# match at the two depths drop out of the dipole's field.
dipole_source = uacpy.Source(depths=[40.0, 60.0], frequencies=50.0,
                             weights=[1, -1])
dipole_stack = uacpy.Kraken(mode_coupling='adiabatic',
                            n_segments=N_SEGMENTS).run(
    env, dipole_source, receiver, run_mode=uacpy.RunMode.COHERENT_TL)
dipole = dipole_stack.superpose()                 # weights = [1, -1]
in_phase = dipole_stack.superpose(weights=[1, 1])
# Modes.excitation is Σ wᵢ·φₘ(zᵢ): shelf mode 1 is driven by the difference of
# its shape at the two depths under the dipole, by the sum in phase.
in_phase_source = uacpy.Source(depths=[40.0, 60.0], frequencies=50.0,
                               weights=[1, 1])
excitation_ratio = (abs(shelf_modes.excitation(dipole_source)[0])
                    / abs(shelf_modes.excitation(in_phase_source)[0]))
print(f"  antiphase pair at 40/60 m: {dipole_stack.n_slabs} slabs superposed; "
      f"shelf mode 1 excited {20 * np.log10(excitation_ratio):.1f} dB "
      f"relative to the in-phase pair; median TL "
      f"{np.nanmedian(dipole.dB):.1f} dB vs {np.nanmedian(in_phase.dB):.1f} "
      f"dB in phase")

fig, _ = env.plot()
fig.savefig(OUT / 'example_06_bottom.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# Mode shapes at both ends. On the slope, krakenc numbers the interface waves
# first (highest Re(k)); their phase speed is below the water sound speed, so
# filtering on that leaves the trapped water-column modes the shelf panel shows.
phase_speed = slope_modes.phase_speeds      # ω / Re(k), m/s
c_water_min = float(np.min(ssp_pairs[:, 1]))
trapped = np.where(phase_speed >= c_water_min)[0]
n_interface = int(np.sum(phase_speed < c_water_min))
print(f"  slope: {n_interface} interface mode(s) below {c_water_min:.0f} m/s "
      f"(Scholte/Stoneley type), {trapped.size} trapped")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for index in range(min(5, shelf_modes.phi.shape[1])):
    axes[0].plot(shelf_modes.phi[:, index].real, shelf_modes.depths,
                 label=f'Mode {index + 1}')
axes[0].set_title(f'Shelf modes (100 m)\n{shelf_modes.phi.shape[1]} total')
for index in trapped[:5]:
    axes[1].plot(slope_modes.phi[:, index].real, slope_modes.depths,
                 label=f'Mode {index + 1}')
axes[1].set_title(f'Slope modes (400 m)\n{slope_modes.phi.shape[1]} total '
                  f'({n_interface} interface, {trapped.size} trapped)')
for ax in axes:
    ax.invert_yaxis()
    ax.set_xlabel('Mode amplitude')
    ax.set_ylabel('Depth (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
fig.suptitle('Mode evolution: shelf to slope', fontsize='xx-large', fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'example_06_modes.png', dpi=150, bbox_inches='tight')
plt.close(fig)

fig, ax = uacpy.plot.plot_mode_wavenumbers(shelf_modes)
ax.set_title(f'Mode wavenumbers in the complex k-plane\n'
             f'shelf — {shelf_modes.phi.shape[1]} modes', fontweight='bold')
fig.savefig(OUT / 'example_06_wavenumbers.png', dpi=150, bbox_inches='tight')
plt.close(fig)

fig, ax = shelf_modes.plot(show_imaginary=True)
ax.set_title('Mode shapes with imaginary parts\n'
             'shelf (solid = real, dashed = imaginary)', fontweight='bold')
fig.savefig(OUT / 'example_06_mode_shapes.png', dpi=150, bbox_inches='tight')
plt.close(fig)

fig, _ = uacpy.plot.plot_modes_heatmap(shelf_modes, mode_range=None,
                                       normalize=True, figsize=(14, 8))
fig.suptitle('All shelf mode shapes', fontsize='x-large', fontweight='bold')
fig.savefig(OUT / 'example_06_modes_heatmap.png', dpi=150, bbox_inches='tight')
plt.close(fig)

fig, ax = uacpy.plot.plot_field(
    tl, env=env, contours=[70, 85, 100],
    title=f'Kraken adiabatic modes over the shelf transition\n'
          f'({N_SEGMENTS} segments, contours at 70/85/100 dB)')
# The run holds one profile at each of np.linspace(0, 20, N_SEGMENTS) km and
# the adiabatic evaluator interpolates between neighbours, so an interior node
# marks where one profile's weight peaks — not a hard switch between two.
for node in np.linspace(0, 20, N_SEGMENTS)[1:-1]:
    ax.axvline(node, color='white', ls='--', alpha=0.3, lw=0.5, zorder=8)
fig.savefig(OUT / 'example_06_result.png', dpi=150, bbox_inches='tight')
plt.close(fig)

fig, (left, right) = plt.subplots(1, 2, figsize=(16, 6))
uacpy.plot.plot_field(in_phase.to_dB(), left, env=env, show_colorbar=False,
                 source=in_phase_source,
                 title='Pair in phase\nsuperpose(weights=[1, 1])')
uacpy.plot.plot_field(dipole.to_dB(), right, env=env, show_colorbar=False,
                 source=dipole_source,
                 title='Pair in antiphase\n'
                       'Source(weights=[1, -1]) → superpose()')
uacpy.plot.shared_colorbar(fig, (left, right), label='TL (dB)')
fig.suptitle('Two sources, one ResultStack, two weightings',
             fontsize='x-large', fontweight='bold')
fig.savefig(OUT / 'example_06_superposed_pair.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)
