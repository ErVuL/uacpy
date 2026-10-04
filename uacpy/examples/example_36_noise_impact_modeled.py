"""Modeled noise impact — a ship's source level through a real TL field.

The physically-modeled version of example 35. Instead of spherical spreading,
a ship's ISO 17208 monopole source level is propagated to a marine mammal
through a Bellhop transfer function over a UNESCO sound-speed profile, band by
band, and then auditory-weighted (Southall 2019).

The loss in each decidecade band is averaged OVER the band: a band level is
the energy of every frequency in it, and one tone at the band centre samples a
single point of the multipath interference pattern instead — 12.2 dB off the
band average in the 250 Hz band here, and 4.8 dB at 50 Hz. One BROADBAND run gives H(f) across the whole
range; ``Field.window(...).broadband_loss()`` averages it per band.

Uses: sound_speed_unesco → Environment SSP · decidecade_bands ·
radiated_noise_level / monopole_source_level (ISO 17208) · Bellhop BROADBAND ·
Field.window · Field.broadband_loss · apply_weighting · env.ssp.plot
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.acoustics import sound_speed_unesco
from uacpy.acoustic_signal import decidecade_bands
from uacpy.noise import (apply_weighting, monopole_source_level,
                         nominal_source_depth, radiated_noise_level)

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

# A summer thermocline, turned into c(z) by UNESCO: depth= is converted to
# the pressure the equation is stated in (45° standard ocean).
depths = np.array([0.0, 25.0, 50.0, 100.0, 200.0])
temperatures = np.array([18.0, 16.0, 12.0, 8.0, 6.0])
sound_speeds = sound_speed_unesco(temperatures, 35.0, depth=depths)
env = uacpy.Environment(
    name="UNESCO thermocline", bathymetry=200.0,
    ssp=uacpy.SoundSpeedProfile.from_pairs(list(zip(depths, sound_speeds))))
print("  UNESCO c(z): " + " ".join(f"{c:.0f}" for c in sound_speeds) + " m/s")

band_low, bands, band_high = decidecade_bands(50, 1000)
source_depth = nominal_source_depth(8.0)      # from an 8 m draught
# A stand-in for a measured received SPL at the 150 m slant range below.
received_spl = 128.0 - 16.0 * np.log10(bands / 50.0)
monopole = monopole_source_level(radiated_noise_level(received_spl, 150.0),
                                 bands, source_depth,
                                 sound_speed=float(sound_speeds[0]))
print(f"  ship source depth {source_depth} m, {bands.size} decidecade bands "
      f"{bands[0]:.0f}-{bands[-1]:.0f} Hz")

# One BROADBAND run across every band at the animal's position, on a 0.5 Hz
# grid (the narrowest band, at 50 Hz, is 11.5 Hz wide), then the loss averaged
# over each band.
animal_range, animal_depth = 5000.0, 30.0
receiver = uacpy.Receiver(depths=animal_depth, ranges=animal_range)
grid = np.arange(np.floor(band_low[0]), np.ceil(band_high[-1]) + 0.5, 0.5)
H = uacpy.Bellhop(backend='fortran').run(
    env, uacpy.Source(depths=source_depth, frequencies=float(bands[0])),
    receiver, run_mode=uacpy.RunMode.BROADBAND, frequencies=grid)
tl = np.array([
    float(np.asarray(H.window(frequency=(lo, hi)).broadband_loss().data)
          .squeeze())
    for lo, hi in zip(band_low, band_high)])

received = monopole - tl
groups = {"LF": "baleen whale", "VHF": "harbour porpoise"}
weighted = {group: apply_weighting(received, frequency=bands, group=group) for group in groups}
print(f"  received level at {animal_range / 1000:.0f} km, "
      f"{animal_depth:.0f} m: peak {received.max():.1f} dB re 1 µPa")
for group, animal in groups.items():
    print(f"  {group} ({animal}) weighted peak: "
          f"{np.nanmax(weighted[group]):.1f} dB")

fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
env.ssp.plot(ax=axes[0, 0], color='C0')
axes[0, 0].set_title("UNESCO sound-speed profile")

axes[0, 1].semilogx(bands, tl, "C3o-")
axes[0, 1].set_xlabel("Frequency [Hz]")
axes[0, 1].set_ylabel("Transmission loss [dB]")
axes[0, 1].set_title(f"Bellhop band-averaged TL @ {animal_range / 1000:.0f} km")
axes[0, 1].grid(which="both", alpha=0.3)

axes[1, 0].semilogx(bands, monopole, "C0o-", label="ship MSL (source)")
axes[1, 0].semilogx(bands, received, color="k", marker="s",
                    label="received @ animal")
axes[1, 0].set_xlabel("Frequency [Hz]")
axes[1, 0].set_ylabel("Level [dB re 1 µPa(·m)]")
axes[1, 0].set_title("source vs received")
axes[1, 0].grid(which="both", alpha=0.3)
axes[1, 0].legend()

axes[1, 1].semilogx(bands, received, "k-", label="unweighted")
for group, animal in groups.items():
    axes[1, 1].semilogx(bands, weighted[group], "o-",
                        label=f"{group} ({animal})")
axes[1, 1].set_xlabel("Frequency [Hz]")
axes[1, 1].set_ylabel("Weighted level [dB]")
axes[1, 1].set_title("auditory-weighted (Southall 2019)")
axes[1, 1].grid(which="both", alpha=0.3)
axes[1, 1].legend()

fig.savefig(OUT / "example_36_noise_impact_modeled.png", dpi=120)
plt.close(fig)
