"""Modeled noise impact — a ship's source level through a real TL field.

The physically-modeled version of example 35. Instead of spherical spreading,
a ship's ISO 17208 monopole source level is propagated to a marine mammal
through a Bellhop transmission-loss field over a UNESCO sound-speed profile,
band by band, and then auditory-weighted (Southall 2019).

Running Bellhop per decidecade band is the point: transmission loss is
frequency-dependent in a way a spreading law cannot capture, and the weighted
level is what an assessment turns on.

Uses: soundspeed_unesco → Environment SSP · decidecade_bands ·
radiated_noise_level / monopole_source_level (ISO 17208) · Bellhop per band ·
apply_weighting · env.ssp.plot
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.core.acoustics import soundspeed_unesco
from uacpy.acoustic_signal.bands import decidecade_bands
from uacpy.noise import (apply_weighting, monopole_source_level,
                         nominal_source_depth, radiated_noise_level)

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

# A summer thermocline, turned into c(z) by UNESCO. soundspeed_unesco takes
# pressure in dbar, which is ~1 dbar per metre at these depths.
depths = np.array([0.0, 25.0, 50.0, 100.0, 200.0])
temperatures = np.array([18.0, 16.0, 12.0, 8.0, 6.0])
sound_speeds = np.array([soundspeed_unesco(t, 35.0, z)
                         for t, z in zip(temperatures, depths)])
env = uacpy.Environment(
    name="UNESCO thermocline", bathymetry=200.0,
    ssp=uacpy.SoundSpeedProfile.from_pairs(list(zip(depths, sound_speeds))))
print("  UNESCO c(z): " + " ".join(f"{c:.0f}" for c in sound_speeds) + " m/s")

_, bands, _ = decidecade_bands(50, 1000)
source_depth = nominal_source_depth(8.0)      # from an 8 m draught
# A stand-in for a measured received SPL at the 150 m slant range below.
received_spl = 128.0 - 16.0 * np.log10(bands / 50.0)
monopole = monopole_source_level(radiated_noise_level(received_spl, 150.0),
                                 bands, source_depth,
                                 sound_speed=float(sound_speeds[0]))
print(f"  ship source depth {source_depth} m, {bands.size} decidecade bands "
      f"{bands[0]:.0f}-{bands[-1]:.0f} Hz")

# One Bellhop run per band, at the animal's position.
animal_range, animal_depth = 5000.0, 30.0
receiver = uacpy.Receiver(depths=animal_depth, ranges=animal_range)
tl = np.array([
    float(np.asarray(uacpy.Bellhop().run(
        env, uacpy.Source(depths=source_depth, frequencies=float(f)),
        receiver).dB).squeeze())
    for f in bands])

received = monopole - tl
groups = {"LF": "baleen whale", "VHF": "harbour porpoise"}
weighted = {group: apply_weighting(received, bands, group) for group in groups}
print(f"  received level at {animal_range / 1000:.0f} km, "
      f"{animal_depth:.0f} m: peak {received.max():.1f} dB re 1 µPa")
for group, animal in groups.items():
    print(f"  {group} ({animal}) weighted peak: "
          f"{np.nanmax(weighted[group]):.1f} dB")

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
env.ssp.plot(ax=axes[0, 0], color='C0')
axes[0, 0].set_title("UNESCO sound-speed profile", loc="left")

axes[0, 1].semilogx(bands, tl, "C3o-")
axes[0, 1].set_xlabel("Frequency [Hz]")
axes[0, 1].set_ylabel("Transmission loss [dB]")
axes[0, 1].set_title(f"Bellhop TL @ {animal_range / 1000:.0f} km", loc="left")
axes[0, 1].grid(which="both", alpha=0.3)

axes[1, 0].semilogx(bands, monopole, "C0o-", label="ship MSL (source)")
axes[1, 0].semilogx(bands, received, color="k", marker="s",
                    label="received @ animal")
axes[1, 0].set_xlabel("Frequency [Hz]")
axes[1, 0].set_ylabel("Level [dB re 1 µPa(·m)]")
axes[1, 0].set_title("source vs received", loc="left")
axes[1, 0].grid(which="both", alpha=0.3)
axes[1, 0].legend()

axes[1, 1].semilogx(bands, received, "k-", label="unweighted")
for group, animal in groups.items():
    axes[1, 1].semilogx(bands, weighted[group], "o-",
                        label=f"{group} ({animal})")
axes[1, 1].set_xlabel("Frequency [Hz]")
axes[1, 1].set_ylabel("Weighted level [dB]")
axes[1, 1].set_title("auditory-weighted (Southall 2019)", loc="left")
axes[1, 1].grid(which="both", alpha=0.3)
axes[1, 1].legend()

fig.savefig(OUT / "example_36_noise_impact_modeled.png", dpi=120)
plt.close(fig)
