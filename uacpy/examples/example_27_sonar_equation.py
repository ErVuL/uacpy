"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 27: Sonar Equation, Reverberation & Detection Range
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    Turn a transmission-loss curve into sonar performance using ``uacpy.sonar``:
    passive/active signal excess, the noise-vs-reverberation crossover, and the
    detection range where the signal excess reaches zero.

FEATURES DEMONSTRATED:
    ✓ uacpy.sonar.passive_signal_excess / active_signal_excess
    ✓ Lambert bottom backscatter + boundary_reverberation
    ✓ detection_range from a signal-excess curve
    ✓ detection_threshold_energy from (Pd, Pf)

NOTE:
    A spherical-spreading + Thorp-absorption TL is used so the example runs with
    no model binary. In practice pass ``tl_db`` from a uacpy Field (Bellhop /
    Kraken / RAM) instead.
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from uacpy import sonar  # noqa: E402


def thorp_db_per_km(freq_hz):
    f = freq_hz / 1000.0
    return (0.11 * f**2 / (1 + f**2) + 44 * f**2 / (4100 + f**2)
            + 2.75e-4 * f**2 + 0.003)


def main():
    print("═" * 80)
    print("EXAMPLE 27: Sonar Equation, Reverberation & Detection Range")
    print("═" * 80)

    freq = 2000.0
    ranges = np.linspace(100.0, 30000.0, 600)
    tl = 20.0 * np.log10(ranges) + thorp_db_per_km(freq) * ranges / 1000.0

    # Detection threshold for Pd=0.5, Pf=1e-4 over a 100 Hz / 1 s integration.
    dt = sonar.detection_threshold_energy(0.5, 1e-4, bandwidth_hz=100.0,
                                          integration_time_s=1.0)

    # --- Passive: detect a 140 dB target against ambient noise ---
    passive_se = sonar.passive_signal_excess(
        source_level=140.0, tl=tl, noise_level=60.0,
        directivity_index=15.0, detection_threshold=dt,
    )
    passive_range = sonar.detection_range(ranges, passive_se)

    # --- Active: echo from a TS=10 dB target, noise vs bottom reverberation ---
    grazing = np.rad2deg(np.arctan2(100.0, ranges))  # 100 m sonar/seafloor height
    sb = sonar.lambert_bottom(grazing)
    rl = sonar.boundary_reverberation(
        ranges, 220.0, sb,
        pulse_length_s=0.05, horizontal_beamwidth_rad=0.1, tl_db=tl,
    )
    active_se = sonar.active_signal_excess(
        220.0, tl, target_strength=10.0, noise_level=60.0,
        directivity_index=15.0, reverberation_level=rl, detection_threshold=dt,
    )
    active_range = sonar.detection_range(ranges, active_se)

    print(f"\n  Detection threshold DT = {dt:.1f} dB")
    print(f"  Passive detection range = {passive_range/1000:.2f} km")
    print(f"  Active  detection range = {active_range/1000:.2f} km")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    ax.plot(ranges / 1000, tl, 'b-')
    ax.set_title('Transmission Loss (spherical + Thorp)', fontweight='bold')
    ax.set_xlabel('Range (km)'); ax.set_ylabel('TL (dB)')
    ax.invert_yaxis(); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(ranges / 1000, passive_se, 'g-')
    ax.axhline(0, color='k', lw=0.8)
    if np.isfinite(passive_range):
        ax.axvline(passive_range / 1000, color='r', ls='--',
                   label=f'{passive_range/1000:.1f} km')
    ax.set_title('Passive Signal Excess', fontweight='bold')
    ax.set_xlabel('Range (km)'); ax.set_ylabel('SE (dB)')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(ranges / 1000, rl, 'm-', label='Reverberation level')
    ax.axhline(60.0 - 15.0, color='c', ls='--', label='Noise background (NL-DI)')
    ax.set_title('Active Background: Reverb vs Noise', fontweight='bold')
    ax.set_xlabel('Range (km)'); ax.set_ylabel('Level (dB)')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(ranges / 1000, active_se, 'r-')
    ax.axhline(0, color='k', lw=0.8)
    if np.isfinite(active_range):
        ax.axvline(active_range / 1000, color='b', ls='--',
                   label=f'{active_range/1000:.1f} km')
    ax.set_title('Active Signal Excess', fontweight='bold')
    ax.set_xlabel('Range (km)'); ax.set_ylabel('SE (dB)')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / 'example_27_sonar_equation.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out}")
    print("\n✓ Example 27 complete\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
