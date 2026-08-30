"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 03: Five Models on One Thermocline at a Single Reference Frequency
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    Run five propagation models on one thermocline environment at a single
    100 Hz reference frequency and compare their TL fields side by side.
    The closing notes explain how to extend the script into a true
    multi-frequency sweep (25 / 50 / 100 / 200 Hz).

ENVIRONMENT:
    - Medium depth water (200m)
    - Summer thermocline profile:
        * Warm mixed layer (0-25m): 1525 m/s, dc/dz = 0
        * Thermocline (25-60m): (1490 - 1525) / (60 - 25) = -1.0 (m/s)/m
        * Isovelocity deep layer (60-200m): 1490 m/s
    - Elastic sediment bottom
    - Flat bathymetry

SOURCE:
    - Shallow depth (15m, inside the mixed layer, above the thermocline)
    - 100 Hz reference frequency

    The mixed layer has dc/dz = 0, so it traps nothing: a surface duct needs a
    positive gradient (Etter, *Underwater Acoustic Modeling and Simulation*,
    §3.7). Nor would this frequency reach a 25 m duct if one existed - Etter
    §3.7.3 gives lambda_max = 8.51e-3 * H**1.5 = 1.06 m, i.e. a cutoff near
    1434 Hz, fourteen times the 100 Hz run here. Energy leaving the source
    refracts downward through the thermocline instead.

MODELS TESTED:
    ✓ Bellhop      (ray tracing)
    ✓ RAM          (parabolic equation)
    ✓ Kraken  (normal modes)
    ✓ Scooter      (wavenumber integration)
    ✓ OAST         (wavenumber integration)

WHAT TO EXPECT IF YOU RERUN AT OTHER FREQUENCIES:
    - Low freq (25 Hz): Deep penetration, few modes, coarse interference
    - Mid freq (50-100 Hz): More modes, clear interference structure
    - High freq (200 Hz): Fine interference; still far below the 1434 Hz a
      25 m duct would need, so the mixed layer stays acoustically transparent
    - Bottom loss increases with frequency; more modes at higher frequency

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path
# Repo root, so ``import uacpy`` resolves from a source checkout.
sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np  # noqa: E402
import uacpy  # noqa: E402
from uacpy.core.environment import SoundSpeedProfile  # noqa: E402
from uacpy.models import Bellhop, RAM, Kraken, Scooter, OAST  # noqa: E402
from plotting_utils import create_example_report  # noqa: E402


def main():
    print("\n" + "═" * 80)
    print("EXAMPLE 03: Five Models on One Thermocline at a Single Reference Frequency")
    print("═" * 80)

    # ═══════════════════════════════════════════════════════════════════════
    # ENVIRONMENT SETUP - Summer thermocline
    # ═══════════════════════════════════════════════════════════════════════
    bottom_props = uacpy.BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1650.0,
        shear_speed=250.0,
        density=1.7,
        attenuation=0.4
    )

    env = uacpy.Environment(
        name="Summer Thermocline - Multi-Frequency",
        bathymetry=200.0,
        ssp=SoundSpeedProfile.from_pairs(
            [(0, 1525), (25, 1525), (60, 1490), (200, 1490)]
        ),
        bottom=bottom_props
    )

    # ═══════════════════════════════════════════════════════════════════════
    # SOURCE CONFIGURATION - Reference frequency
    # ═══════════════════════════════════════════════════════════════════════
    source = uacpy.Source(
        depths=15.0,       # In the warm mixed layer, above the thermocline
        frequencies=100.0   # Mid-band reference
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RECEIVER GRID
    # ═══════════════════════════════════════════════════════════════════════
    receiver = uacpy.Receiver(
        depths=np.linspace(3, 197, 70),        # High resolution depth
        ranges=np.linspace(200, 25000, 120)    # 0.2-25 km
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RUN ALL MODELS
    # ═══════════════════════════════════════════════════════════════════════
    results = {}

    models = [
        ('Bellhop', Bellhop(verbose=False)),
        ('RAM', RAM(verbose=False)),
        ('Kraken', Kraken(verbose=False)),
        ('Scooter', Scooter(verbose=False)),
        ('OAST', OAST(verbose=False)),
    ]

    print("\nRunning models at 100 Hz (reference frequency):")
    print("For true multi-frequency analysis, run at: 25, 50, 100, 200 Hz\n")

    for name, model in models:
        print(f"  {name:15s}...", end=" ", flush=True)
        try:
            results[name] = model.run(env, source, receiver)
            print("✓")
        except Exception as e:
            print(f"✗ {e}")
            results[name] = None

    # ═══════════════════════════════════════════════════════════════════════
    # GENERATE REPORT
    # ═══════════════════════════════════════════════════════════════════════
    if any(r is not None for r in results.values()):
        create_example_report(
            example_num=3,
            title="Five Models on One Thermocline at a Single Reference Frequency",
            description="Five propagation models compared side by side on one summer-thermocline "
            "environment at 100 Hz. The 0-25m mixed layer has zero "
            "gradient and does not duct; energy refracts downward through the "
            "-1.0 (m/s)/m thermocline.",
            env=env,
            source=source, receiver=receiver,
            results=results,
            output_prefix="example_03"
        )

        print("\n" + "="*80)
        print("FREQUENCY-DEPENDENT ANALYSIS NOTES:")
        print("="*80)
        print("To perform full multi-frequency analysis:")
        print("  1. Run this example with frequencies=25, 50, 100, 200 Hz")
        print("  2. Compare TL patterns and duct trapping")
        print("  3. Analyze modal content and cutoff frequencies")
        print("  4. Evaluate model performance vs frequency")
        print("="*80 + "\n")
    else:
        print("\n⚠ No models ran successfully!")
        return 1

    print("\n✓ Example 03 complete\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
