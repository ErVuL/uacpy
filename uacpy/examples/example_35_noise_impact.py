"""Underwater noise impact assessment — the standards chain.

One workflow from a ship measurement to a marine-mammal auditory-impact
estimate, each step an international standard: UNESCO c(T, S, depth) for the
site, ISO 18405 / IEC 61260-1 decidecade bands, ISO 17208 for measured radiated
noise → monopole source level, spreading + Thorp absorption to the receiver,
and Southall et al. 2019 auditory weighting, summed over the bands to the
weighted received level and, over an exposure time, the weighted SEL their
non-impulsive criteria are stated on.

Weighting is the step that matters: a porpoise and a baleen whale hear the same
spectrum very differently, so the unweighted received level is not the number
an assessment turns on.

Uses: sound_speed_unesco · decidecade_bands · nominal_source_depth ·
radiated_noise_level · monopole_source_level · apply_weighting ·
sum_levels_dB ·
Thorp.table · plot_source_level · plot_weighting
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.acoustics import sound_speed_unesco, sum_levels_dB
from uacpy.acoustic_signal import decidecade_bands
from uacpy.noise import (apply_weighting, auditory_weighting,
                         monopole_source_level,
                         nominal_source_depth, radiated_noise_level)

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

temperature, salinity, depth = 12.0, 35.0, 50.0
sound_speed = sound_speed_unesco(temperature, salinity, depth=depth)
_, band_centers, _ = decidecade_bands(10, 25000)
print(f"  site c (UNESCO, {temperature}°C, S={salinity}, "
      f"{depth:.0f} m): {sound_speed:.2f} m/s")

# A merchant ship: a typical measured received spectrum at the ISO 17208
# geometry, back to radiated noise, then to an equivalent monopole.
draught = 8.0
source_depth = nominal_source_depth(draught)
received_spl = 130.0 - 18.0 * np.log10(np.maximum(band_centers / 60.0, 1.0))
radiated = radiated_noise_level(received_spl, 150.0)     # 150 m slant range
monopole = monopole_source_level(radiated, band_centers, source_depth,
                                 sound_speed=sound_speed)
print(f"  draught {draught} m → source depth {source_depth} m; peak MSL "
      f"{monopole.max():.1f} dB re 1 µPa²·m² at "
      f"{band_centers[np.argmax(monopole)]:.0f} Hz")

# Out to 2 km: spherical spreading plus Thorp volume absorption.
range_m = 2000.0
# units='dB/m' rather than dB/km divided by 1000: the conversion is a
# value the library applies, not arithmetic to get right here.
alpha = uacpy.Thorp().table(band_centers, units='dB/m').data
tl = 20 * np.log10(range_m) + np.ravel(alpha) * range_m
received = monopole - tl
print(f"  received level at {range_m / 1000:.0f} km: "
      f"{received.max():.1f} dB re 1 µPa (band peak)")

groups = {"LF": "baleen whale", "VHF": "harbour porpoise"}
weighted = {group: apply_weighting(received, frequency=band_centers, weighting=group)
            for group in groups}
# Southall et al. (2019) state non-impulsive criteria as weighted SEL; a
# level held for T seconds has SEL = level + 10·log10(T / 1 s) (ISO 18405).
exposure_s = 3600.0
for group, animal in groups.items():
    level = sum_levels_dB(weighted[group], axis=-1)
    print(f"  {group} ({animal}) weighted received level: {level:.1f} dB re "
          f"1 µPa; held {exposure_s / 3600:.0f} h, weighted SEL "
          f"{level + 10 * np.log10(exposure_s):.1f} dB re 1 µPa²·s")

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
uacpy.plot.plot_source_level(band_centers, monopole, ax=axes[0, 0],
                             title="(ISO 17208 monopole)")
axes[0, 0].semilogx(band_centers, radiated, "--", color="gray",
                    label="RNL (measured)")
axes[0, 0].legend()

f_weighting = np.logspace(1.0, 5.5, 400)
for group in groups:
    uacpy.plot.plot_weighting(f_weighting,
                              auditory_weighting(f_weighting, group),
                              ax=axes[0, 1], label=group)

axes[1, 0].semilogx(band_centers, received, "k-", label="unweighted RL @ 2 km")
for group, animal in groups.items():
    axes[1, 0].semilogx(band_centers, weighted[group],
                        label=f"{group}-weighted ({animal})")
axes[1, 0].set_xlabel("Frequency [Hz]")
axes[1, 0].set_ylabel("Level [dB re 1 µPa]")
axes[1, 0].set_title("received vs auditory-weighted")
axes[1, 0].grid(which="both", alpha=0.3)
axes[1, 0].legend()

axes[1, 1].semilogx(band_centers, tl, color="C3")
axes[1, 1].set_xlabel("Frequency [Hz]")
axes[1, 1].set_ylabel("Transmission loss [dB]")
axes[1, 1].set_title("spreading + Thorp @ 2 km")
axes[1, 1].grid(which="both", alpha=0.3)

fig.savefig(OUT / "example_35_noise_impact.png", dpi=120)
plt.close(fig)
