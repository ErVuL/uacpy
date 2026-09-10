"""Underwater noise impact assessment — the standards chain.

One workflow from a ship measurement to a marine-mammal auditory-impact
estimate, each step an international standard: UNESCO c(T, S, depth) for the
site, ISO 18405 / IEC 61260-1 decidecade bands, ISO 17208 for measured radiated
noise → monopole source level, spreading + Thorp absorption to the receiver,
and Southall et al. 2019 auditory weighting for the impact-relevant level.

Weighting is the step that matters: a porpoise and a baleen whale hear the same
spectrum very differently, so the unweighted received level is not the number
an assessment turns on.

Uses: soundspeed_unesco · decidecade_bands · nominal_source_depth ·
radiated_noise_level · monopole_source_level · apply_weighting ·
thorp_dB_per_km · plot_source_level · plot_weighting
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.core.absorption import thorp_dB_per_km
from uacpy.core.acoustics import soundspeed_unesco
from uacpy.acoustic_signal.bands import decidecade_bands
from uacpy.noise import (apply_weighting, monopole_source_level,
                         nominal_source_depth, radiated_noise_level)

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

temperature, salinity, depth_dbar = 12.0, 35.0, 50.0
sound_speed = soundspeed_unesco(temperature, salinity, depth_dbar)
_, band_centres, _ = decidecade_bands(10, 25000)
print(f"  site c (UNESCO, {temperature}°C, S={salinity}, "
      f"{depth_dbar:.0f} m): {sound_speed:.2f} m/s")

# A merchant ship: a typical measured received spectrum at the ISO 17208
# geometry, back to radiated noise, then to an equivalent monopole.
draught = 8.0
source_depth = nominal_source_depth(draught)
received_spl = 130.0 - 18.0 * np.log10(np.maximum(band_centres / 60.0, 1.0))
radiated = radiated_noise_level(received_spl, 150.0)     # 150 m slant range
monopole = monopole_source_level(radiated, band_centres, source_depth,
                                 sound_speed=sound_speed)
print(f"  draught {draught} m → source depth {source_depth} m; peak MSL "
      f"{monopole.max():.1f} dB re 1 µPa·m at "
      f"{band_centres[np.argmax(monopole)]:.0f} Hz")

# Out to 2 km: spherical spreading plus Thorp volume absorption.
range_m = 2000.0
tl = 20 * np.log10(range_m) + thorp_dB_per_km(band_centres) * (range_m / 1000)
received = monopole - tl
print(f"  received level at {range_m / 1000:.0f} km: "
      f"{received.max():.1f} dB re 1 µPa (band peak)")

groups = {"LF": "baleen whale", "VHF": "harbour porpoise"}
weighted = {group: apply_weighting(received, band_centres, group)
            for group in groups}
for group, animal in groups.items():
    print(f"  {group} ({animal}) weighted band peak: "
          f"{np.nanmax(weighted[group]):.1f} dB")

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
uacpy.plot.plot_source_level(band_centres, monopole, ax=axes[0, 0],
                             title="(ISO 17208 monopole)")
axes[0, 0].semilogx(band_centres, radiated, "--", color="gray",
                    label="RNL (measured)")
axes[0, 0].legend()

uacpy.plot.plot_weighting(list(groups), ax=axes[0, 1])

axes[1, 0].semilogx(band_centres, received, "k-", label="unweighted RL @ 2 km")
for group, animal in groups.items():
    axes[1, 0].semilogx(band_centres, weighted[group],
                        label=f"{group}-weighted ({animal})")
axes[1, 0].set_xlabel("Frequency [Hz]")
axes[1, 0].set_ylabel("Level [dB re 1 µPa]")
axes[1, 0].set_title("received vs auditory-weighted", loc="left")
axes[1, 0].grid(which="both", alpha=0.3)
axes[1, 0].legend()

axes[1, 1].semilogx(band_centres, tl, color="C3")
axes[1, 1].set_xlabel("Frequency [Hz]")
axes[1, 1].set_ylabel("Transmission loss [dB]")
axes[1, 1].set_title("spreading + Thorp @ 2 km", loc="left")
axes[1, 1].grid(which="both", alpha=0.3)

fig.savefig(OUT / "example_35_noise_impact.png", dpi=120)
plt.close(fig)
