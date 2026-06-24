"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 35: Underwater Noise Impact Assessment (standards chain)
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    Chain the new international-standard tools into one workflow — from a ship
    measurement to a marine-mammal auditory-impact estimate:
      • site sound speed         — UNESCO (Chen-Millero) c(T, S, depth)
      • decidecade bands         — ISO 18405 / IEC 61260-1 (10 Hz - 25 kHz)
      • ship source level        — ISO 17208: measured RNL -> monopole MSL
      • propagation              — spherical spreading + Thorp absorption
      • marine-mammal weighting  — Southall et al. 2019 (LF whale, VHF porpoise)
      • weighted received level  — the impact-relevant quantity

FEATURES DEMONSTRATED:
    ✓ soundspeed_unesco · decidecade_bands · radiated_noise_level /
      monopole_source_level (ISO 17208) · auditory_weighting (Southall 2019)
    ✓ consistent plot helpers: plot_source_level · plot_weighting
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from uacpy.core.acoustics import soundspeed_unesco  # noqa: E402
from uacpy.acoustic_signal.bands import decidecade_bands  # noqa: E402
from uacpy.noise import (  # noqa: E402
    radiated_noise_level, nominal_source_depth, monopole_source_level,
    apply_weighting,
)
from uacpy.visualization import plot_source_level, plot_weighting  # noqa: E402


def _thorp_db_per_km(f_hz):
    """Thorp (1967) absorption [dB/km], f in kHz."""
    f = np.asarray(f_hz, float) / 1000.0
    return (3.3e-3 + 0.11 * f**2 / (1 + f**2)
            + 44.0 * f**2 / (4100 + f**2) + 3.0e-4 * f**2)


def main():
    print("═" * 80)
    print("EXAMPLE 35: Underwater Noise Impact Assessment (standards chain)")
    print("═" * 80)

    # --- (1) site sound speed (UNESCO) ---
    T, S, depth_dbar = 12.0, 35.0, 50.0
    c = soundspeed_unesco(T, S, depth_dbar)
    print(f"\n  site c (UNESCO, T={T}°C S={S} ~{depth_dbar:.0f} m) : {c:.2f} m/s")

    # --- (2) decidecade bands 10 Hz - 25 kHz ---
    _, fc, _ = decidecade_bands(10, 25000)

    # --- (3) ship: a measured decidecade RNL spectrum -> monopole source level ---
    draught = 8.0
    d_s = nominal_source_depth(draught)                      # 5.6 m
    slant = 150.0
    # a typical merchant-ship received SPL spectrum at the standard geometry:
    rx_spl = 130.0 - 18.0 * np.log10(np.maximum(fc / 60.0, 1.0))
    rnl = radiated_noise_level(rx_spl, slant)
    msl = monopole_source_level(rnl, fc, d_s, sound_speed=c)
    print(f"  ship draught {draught} m -> source depth d_s = {d_s} m")
    print(f"  peak MSL : {msl.max():.1f} dB re 1 µPa·m at {fc[np.argmax(msl)]:.0f} Hz")

    # --- (4) propagate to a receiver 2 km away (spreading + Thorp) ---
    R = 2000.0
    tl = 20 * np.log10(R) + _thorp_db_per_km(fc) * (R / 1000.0)
    rl = msl - tl
    print(f"  received level @ {R/1000:.0f} km : {rl.max():.1f} dB re 1 µPa "
          f"(band peak)")

    # --- (5) marine-mammal weighting (Southall 2019) ---
    groups = {"LF": "baleen whale", "VHF": "harbour porpoise"}
    weighted = {g: apply_weighting(rl, fc, g) for g in groups}
    for g, name in groups.items():
        print(f"  {g} ({name}) weighted band-peak : {np.nanmax(weighted[g]):.1f} dB")

    # ----------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    plot_source_level(fc, msl, ax=axes[0, 0], title="(ISO 17208 monopole)")
    axes[0, 0].semilogx(fc, rnl, "--", color="gray", label="RNL (measured)")
    axes[0, 0].legend()

    plot_weighting(list(groups), ax=axes[0, 1])

    ax = axes[1, 0]
    ax.semilogx(fc, rl, "k-", label="unweighted RL @ 2 km")
    for g, name in groups.items():
        ax.semilogx(fc, weighted[g], label=f"{g}-weighted ({name})")
    ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("Level [dB re 1 µPa]")
    ax.set_title("[impact] received vs auditory-weighted", loc="left")
    ax.grid(which="both", alpha=0.3); ax.legend()

    ax = axes[1, 1]
    ax.semilogx(fc, tl, color="C3")
    ax.set_xlabel("Frequency [Hz]"); ax.set_ylabel("Transmission loss [dB]")
    ax.set_title("[propagation] spreading + Thorp @ 2 km", loc="left")
    ax.grid(which="both", alpha=0.3)

    out = OUTPUT_DIR / "example_35_noise_impact.png"
    fig.savefig(out, dpi=120)
    print(f"\n  saved : {out.name}")
    plt.close(fig)


if __name__ == "__main__":
    main()
