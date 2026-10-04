"""Sonar equation, reverberation and detection range.

Turning transmission loss into sonar performance. Part 1 uses a spherical-
spreading + Thorp TL so the sonar-equation mechanics are visible on their own:
passive and active signal excess, the noise-versus-reverberation crossover, and
the range where signal excess reaches zero. Part 2 replaces that curve with a
Bellhop TL field and maps the same quantities over the whole (depth, range)
grid.

Part 2's environment has a mild surface duct (sound-speed maximum near 30 m),
so the map shows real propagation structure — a low-loss surface channel over a
weaker sub-duct shadow — rather than plain spreading. Incoherent TL is the
standard basis for such maps: it keeps that geometric ray-density structure
while dropping the fine multipath fringes a coherent field would imprint.

Uses: sonar.detection_threshold_energy · passive/active_signal_excess ·
lambert_bottom · boundary_reverberation · detection_range · ts_cylinder ·
passive/active_signal_excess_field · transition_probability_field ·
detection_ranges_by_depth · plot_signal_excess · plot_detection_probability ·
plot_roc · Thorp.table
"""

import os
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy import sonar

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

frequency = 2000.0
# The range vector has to extend past the SE = 0 crossing or detection_range
# has nothing to bracket and returns inf. With SL=140, NL-DI=45 and DT=-3.8 the
# passive curve crosses near 44 km, so 30 km would stop short of the very
# quantity this example is about.
ranges = np.linspace(100.0, 60000.0, 600)
# units='dB/m' rather than dB/km divided by 1000: the conversion is a value
# the library applies, not arithmetic to get right here.
alpha_dB_per_m = float(np.ravel(
    uacpy.Thorp().table(frequency, units='dB/m').data)[0])
tl = (20.0 * np.log10(ranges)
      + alpha_dB_per_m * ranges)

# Detection threshold for Pd=0.5, Pf=1e-4 over a 100 Hz / 1 s integration.
threshold = sonar.detection_threshold_energy(0.5, 1e-4, bandwidth_hz=100.0,
                                             integration_time_s=1.0)

passive_se = sonar.passive_signal_excess(
    source_level_dB=140.0, tl_dB=tl, noise_level_dB=60.0, directivity_index_dB=15.0,
    detection_threshold_dB=threshold)
passive_range = sonar.detection_range(ranges, signal_excess_dB=passive_se)

# Active: the echo competes with bottom reverberation as well as noise. The
# grazing angle is set by a sonar 100 m above the seafloor.
grazing = np.rad2deg(np.arctan2(100.0, ranges))
reverberation = sonar.boundary_reverberation(
    ranges, 220.0, sonar.lambert_bottom(grazing), pulse_length_s=0.05,
    horizontal_beamwidth_rad=0.1, tl_dB=tl)
active_se = sonar.active_signal_excess(
    220.0, tl, target_strength_dB=10.0, noise_level_dB=60.0,
    directivity_index_dB=15.0, reverberation_level_dB=reverberation,
    detection_threshold_dB=threshold)
active_range = sonar.detection_range(ranges, signal_excess_dB=active_se)
print(f"  DT = {threshold:.1f} dB → passive detection range "
      f"{passive_range / 1000:.2f} km, active "
      f"{active_range / 1000:.2f} km")

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes[0, 0].plot(ranges / 1000, tl, 'b-')
axes[0, 0].set_title('Transmission loss (spherical + Thorp)',
                     fontweight='bold')
axes[0, 0].set_xlabel('Range (km)')
axes[0, 0].set_ylabel('TL (dB)')
axes[0, 0].invert_yaxis()
axes[0, 0].grid(True, alpha=0.3)

for ax, se, detection, colour, name in (
        (axes[0, 1], passive_se, passive_range, 'g', 'Passive'),
        (axes[1, 1], active_se, active_range, 'r', 'Active')):
    ax.plot(ranges / 1000, se, f'{colour}-', label=f'{name} SE')
    ax.axhline(0, color='k', lw=0.8)
    ax.axvline(detection / 1000, color='k', ls='--',
               label=f'detection range {detection / 1000:.1f} km')
    ax.set_title(f'{name} signal excess', fontweight='bold')
    ax.set_xlabel('Range (km)')
    ax.set_ylabel('SE (dB)')
    ax.legend()
    ax.grid(True, alpha=0.3)

axes[1, 0].plot(ranges / 1000, reverberation, 'm-',
                label='Reverberation level')
axes[1, 0].axhline(60.0 - 15.0, color='c', ls='--',
                   label='Noise background (NL−DI)')
axes[1, 0].set_title('Active background: reverberation vs noise',
                     fontweight='bold')
axes[1, 0].set_xlabel('Range (km)')
axes[1, 0].set_ylabel('Level (dB)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / 'example_27_sonar_equation.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)

# ── Part 2: the same budget over a modelled TL field ────────────────────────
# Ray theory is valid here: D/λ ≈ 267, well above the D/λ ≳ 100 rule of thumb
# (modes take over below ≈ 30 — Stergiopoulos §10.2.2).
env = uacpy.Environment(
    name="SE grid demo", bathymetry=200.0,
    ssp=[(0.0, 1500.0), (30.0, 1512.0), (120.0, 1496.0), (200.0, 1500.0)],
    bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                    sound_speed=1600.0, density=1.5,
                                    attenuation=0.5),
    absorption=uacpy.Thorp())       # the volume loss Part 1's curve carries
source = uacpy.Source(depths=18.0, frequencies=frequency)   # inside the duct
# Ranges start at 200 m, where the seafloor patch is seen at 42° grazing,
# inside the 45° Lambert's law holds to.
receiver = uacpy.Receiver(depths=np.linspace(0.0, 200.0, 201),
                          ranges=np.linspace(200.0, 20000.0, 350))
# Geometric-hat beams with automatic beam count and step (n_beams=0), the
# BELLHOP User Guide's recommendation for TL runs.
tl_field = uacpy.Bellhop(backend='fortran', beam_type='G', n_beams=0, launch_angles=(-80, 80)).run(
    env, source, receiver, run_mode=uacpy.RunMode.INCOHERENT_TL)

se_passive = sonar.passive_signal_excess_field(
    tl_field, source_level_dB=125.0, noise_level_dB=75.0, directivity_index_dB=15.0,
    detection_threshold_dB=threshold)
# A geometric-regime target strength (Urick Table 9.1) instead of an assumed
# number: a 1.5 m × 5 m rigid cylinder at broadside.
target_strength = sonar.ts_cylinder(1.5, 5.0, frequency=frequency)
# Reverberation uses the same modelled TL as the echo, evaluated at the
# seafloor where the scattering patch sits. The 18 m source is 182 m above it.
se_active = sonar.active_signal_excess_field(
    tl_field, source_level_dB=190.0, target_strength_dB=target_strength,
    noise_level_dB=75.0, directivity_index_dB=15.0,
    reverberation_level_dB=sonar.boundary_reverberation(
        np.hypot(receiver.ranges, 182.0), 190.0,        # slant range to the seabed
        sonar.lambert_bottom(np.rad2deg(np.arctan2(182.0, receiver.ranges))),
        pulse_length_s=0.05, horizontal_beamwidth_rad=0.1,
        tl_dB=tl_field.at(depth=float(env.depth)).dB),
    detection_threshold_dB=threshold)

cut = se_passive.at(depth=100.0)
print(f"  target strength {target_strength:.1f} dB; passive detection range "
      f"at 100 m over the Bellhop field = "
      f"{sonar.detection_range_from_field(cut) / 1000:.2f} km")

# Mean SE → detection probability through Urick's transition curve
# (σ = 5.6 dB, Dyer's saturated-multipath fluctuation).
probability = sonar.transition_probability_field(se_passive, sigma_dB=5.6)
# The surface duct carries the target past the 20 km grid on its top rows, so
# the outermost crossing there lies beyond it and inf comes back; the notice
# says so and is printed, and those rows are left off the profile below.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    profile_depths, profile_ranges = sonar.detection_ranges_by_depth(
        se_passive)
for warning in caught:
    print(f"  noted: {str(warning.message).split(' — ')[0]}")

fig, axes = plt.subplots(2, 2, figsize=(15, 9))
uacpy.plot.plot_signal_excess(se_passive, ax=axes[0, 0], env=env,
                              title='Passive signal excess')
uacpy.plot.plot_signal_excess(
    se_active, ax=axes[0, 1], env=env,
    title='Active signal excess (noise + bottom reverb)')
uacpy.plot.plot_detection_probability(
    probability, ax=axes[1, 0], env=env,
    title='Passive detection probability (σ = 5.6 dB)')
finite = np.isfinite(profile_ranges)
axes[1, 1].plot(profile_ranges[finite] / 1000.0, profile_depths[finite], 'b-')
axes[1, 1].set_xlabel('Detection range (km)')
axes[1, 1].set_ylabel('Receiver depth (m)')
axes[1, 1].set_xlim(0, receiver.ranges.max() / 1000.0)
axes[1, 1].invert_yaxis()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_title('Passive detection range vs depth (SE = 0)')
fig.suptitle('Sonar performance over a Bellhop TL grid', fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'example_27_signal_excess_grid.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)

# Receiver operating characteristic for a family of detector deflections d'
# (detection index d = d'²). The dashed line marks the Pfa = 1e-4 operating
# point used above.
fig, ax = uacpy.plot.plot_roc([1.0, 2.0, 3.0, 4.0, 5.0],
                              title='ROC — Gaussian detector')
ax.axvline(1e-4, color='k', ls='--', lw=1, alpha=0.6)
ax.text(1.1e-4, 0.05, 'Pfa = 1e-4', fontsize='small')
fig.savefig(OUT / 'example_27_roc.png', dpi=150, bbox_inches='tight')
plt.close(fig)
