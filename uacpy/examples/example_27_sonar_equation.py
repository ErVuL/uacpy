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
passive/active_signal_excess_field · probability_of_detection_field ·
detection_range_by_depth · plot_signal_excess · plot_detection_probability ·
plot_roc · core.absorption.thorp_dB_per_km
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy import sonar
from uacpy.core.absorption import thorp_dB_per_km

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

frequency = 2000.0
# The range vector has to extend past the SE = 0 crossing or detection_range
# has nothing to bracket and returns inf. With SL=140, NL-DI=45 and DT=-4.3 the
# passive curve crosses near 45 km, so 30 km would stop short of the very
# quantity this example is about.
ranges = np.linspace(100.0, 60000.0, 600)
tl = (20.0 * np.log10(ranges)
      + thorp_dB_per_km(frequency) * ranges / 1000.0)

# Detection threshold for Pd=0.5, Pf=1e-4 over a 100 Hz / 1 s integration.
threshold = sonar.detection_threshold_energy(0.5, 1e-4, bandwidth_hz=100.0,
                                             integration_time_s=1.0)

passive_se = sonar.passive_signal_excess(
    source_level=140.0, tl=tl, noise_level=60.0, directivity_index=15.0,
    detection_threshold=threshold)
passive_range = sonar.detection_range(ranges, passive_se)

# Active: the echo competes with bottom reverberation as well as noise. The
# grazing angle is set by a sonar 100 m above the seafloor.
grazing = np.rad2deg(np.arctan2(100.0, ranges))
reverberation = sonar.boundary_reverberation(
    ranges, 220.0, sonar.lambert_bottom(grazing), pulse_length_s=0.05,
    horizontal_beamwidth_rad=0.1, tl_dB=tl)
active_se = sonar.active_signal_excess(
    220.0, tl, target_strength=10.0, noise_level=60.0,
    directivity_index=15.0, reverberation_level=reverberation,
    detection_threshold=threshold)
active_range = sonar.detection_range(ranges, active_se)
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
                                    attenuation=0.5))
source = uacpy.Source(depths=18.0, frequencies=frequency)   # inside the duct
receiver = uacpy.Receiver(depths=np.linspace(0.0, 200.0, 201),
                          ranges=np.linspace(100.0, 20000.0, 350))
# Geometric-hat beams with automatic beam count and step (n_beams=0), the
# BELLHOP User Guide's recommendation for TL runs.
tl_field = uacpy.Bellhop(beam_type='G', n_beams=0, alpha=(-80, 80)).run(
    env, source, receiver, run_mode=uacpy.RunMode.INCOHERENT_TL)

se_passive = sonar.passive_signal_excess_field(
    tl_field, source_level=125.0, noise_level=75.0, directivity_index=15.0,
    detection_threshold=threshold)
# A geometric-regime target strength (Urick Table 9.1) instead of an assumed
# number: a 1.5 m × 5 m rigid cylinder at broadside.
target_strength = sonar.ts_cylinder(1.5, 5.0, frequency)
# Reverberation uses the same modelled TL as the echo, evaluated at the
# seafloor where the scattering patch sits. The 18 m source is 182 m above it.
se_active = sonar.active_signal_excess_field(
    tl_field, source_level=190.0, target_strength=target_strength,
    noise_level=75.0, directivity_index=15.0,
    reverberation_level=sonar.boundary_reverberation(
        receiver.ranges, 190.0,
        sonar.lambert_bottom(np.rad2deg(np.arctan2(182.0, receiver.ranges))),
        pulse_length_s=0.05, horizontal_beamwidth_rad=0.1,
        tl_dB=tl_field.at(depth=float(env.depth)).dB),
    detection_threshold=threshold)

cut = se_passive.at(depth=100.0)
print(f"  target strength {target_strength:.1f} dB; passive detection range "
      f"at 100 m over the Bellhop field = "
      f"{sonar.detection_range(cut.coords['range'], cut.data) / 1000:.2f} km")

# Mean SE → detection probability through Urick's transition curve
# (σ = 5.6 dB, Dyer's saturated-multipath fluctuation).
probability = sonar.probability_of_detection_field(se_passive, sigma_dB=5.6)
profile_depths, profile_ranges = sonar.detection_range_by_depth(se_passive)

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
ax.text(1.1e-4, 0.05, 'Pfa = 1e-4', fontsize=8)
fig.savefig(OUT / 'example_27_roc.png', dpi=150, bbox_inches='tight')
plt.close(fig)
