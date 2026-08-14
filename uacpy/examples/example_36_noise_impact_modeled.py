"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 36: Modeled Noise Impact — ship source level through a real TL field
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    The physically-modeled version of example 35: instead of simple spreading,
    propagate a ship's ISO 17208 monopole source level through a *real* Bellhop
    transmission-loss field (with a UNESCO sound-speed profile) to a marine
    mammal, then apply Southall (2019) auditory weighting.
      • UNESCO sound speed builds the environment's SSP, c(z)
      • ISO 17208 ship monopole source level (decidecade bands)
      • Bellhop COHERENT_TL at each band -> TL at the animal's location
      • received level = MSL - TL, then LF/VHF auditory weighting

FEATURES DEMONSTRATED:
    ✓ soundspeed_unesco -> Environment SSP · monopole_source_level (ISO 17208)
    ✓ Bellhop per decidecade band · auditory_weighting (Southall 2019)

NOTE: requires the Bellhop binary (install.sh).
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import uacpy  # noqa: E402
from uacpy.core.environment import SoundSpeedProfile  # noqa: E402
from uacpy.models import Bellhop  # noqa: E402
from uacpy.core.acoustics import soundspeed_unesco  # noqa: E402
from uacpy.acoustic_signal.bands import decidecade_bands  # noqa: E402
from uacpy.noise import (  # noqa: E402
    nominal_source_depth, monopole_source_level, radiated_noise_level,
    apply_weighting,
)


def main():
    print("═" * 80)
    print("EXAMPLE 36: Modeled Noise Impact (ship MSL through a Bellhop TL field)")
    print("═" * 80)

    # --- (1) environment with a UNESCO sound-speed profile ---
    depths = np.array([0.0, 25.0, 50.0, 100.0, 200.0])
    temps = np.array([18.0, 16.0, 12.0, 8.0, 6.0])       # summer thermocline
    sal = 35.0
    cz = np.array([soundspeed_unesco(t, sal, z) for t, z in zip(temps, depths)])
    env = uacpy.Environment(
        name="UNESCO thermocline",
        bathymetry=200.0,
        ssp=SoundSpeedProfile.from_pairs(list(zip(depths, cz))),
    )
    print("\n  UNESCO SSP c(z):", " ".join(f"{c:.0f}" for c in cz), "m/s")

    # --- (2) ship monopole source level on decidecade bands (50 Hz - 1 kHz) ---
    _, fbands, _ = decidecade_bands(50, 1000)
    draught = 8.0
    d_s = nominal_source_depth(draught)
    # Stand-in for a measured received SPL at the 150 m slant range below;
    # radiated_noise_level turns it into an RNL, monopole_source_level into MSL.
    rx_spl = 128.0 - 16.0 * np.log10(fbands / 50.0)
    msl = monopole_source_level(radiated_noise_level(rx_spl, 150.0), fbands, d_s,
                                sound_speed=float(cz[0]))
    print(f"  ship d_s={d_s} m, {fbands.size} decidecade bands {fbands[0]:.0f}-{fbands[-1]:.0f} Hz")

    # --- (3) Bellhop TL at the marine mammal (5 km, 30 m) per band ---
    mammal_range, mammal_depth = 5000.0, 30.0
    receiver = uacpy.Receiver(depths=mammal_depth, ranges=mammal_range)
    bellhop = Bellhop(verbose=False)
    tl = np.empty(fbands.size)
    print("  running Bellhop per band ...", end=" ", flush=True)
    for i, f in enumerate(fbands):
        src = uacpy.Source(depths=d_s, frequencies=float(f))
        field = bellhop.run(env, src, receiver)
        tl[i] = float(np.asarray(field.db).squeeze())
    print("done")

    # --- (4) received + auditory-weighted levels ---
    rl = msl - tl
    groups = {"LF": "baleen whale", "VHF": "harbour porpoise"}
    weighted = {g: apply_weighting(rl, fbands, g) for g in groups}
    print(f"  received level @ {mammal_range/1000:.0f} km, {mammal_depth:.0f} m : "
          f"peak {rl.max():.1f} dB re 1 µPa")
    for g, name in groups.items():
        print(f"  {g} ({name}) weighted peak : {np.nanmax(weighted[g]):.1f} dB")

    # ----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(cz, depths, "o-"); ax.invert_yaxis()
    ax.set_xlabel("Sound speed [m/s]"); ax.set_ylabel("Depth [m]")
    ax.set_title("[environment] UNESCO sound-speed profile", loc="left")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.semilogx(fbands, tl, "C3o-")
    ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("Transmission loss [dB]")
    ax.set_title(f"[propagation] Bellhop TL @ {mammal_range/1000:.0f} km", loc="left")
    ax.grid(which="both", alpha=0.3)

    ax = axes[1, 0]
    ax.semilogx(fbands, msl, "C0o-", label="ship MSL (source)")
    ax.semilogx(fbands, rl, color="k", marker="s", label="received @ animal")
    ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("Level [dB re 1 µPa(·m)]")
    ax.set_title("[impact] source vs received", loc="left")
    ax.grid(which="both", alpha=0.3); ax.legend()

    ax = axes[1, 1]
    ax.semilogx(fbands, rl, "k-", label="unweighted")
    for g, name in groups.items():
        ax.semilogx(fbands, weighted[g], "o-", label=f"{g} ({name})")
    ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("Weighted level [dB]")
    ax.set_title("[impact] auditory-weighted (Southall 2019)", loc="left")
    ax.grid(which="both", alpha=0.3); ax.legend()

    out = OUTPUT_DIR / "example_36_noise_impact_modeled.png"
    fig.savefig(out, dpi=120)
    print(f"\n  saved : {out.name}")
    plt.close(fig)


if __name__ == "__main__":
    main()
