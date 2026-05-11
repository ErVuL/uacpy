"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 03: Multi-Frequency Analysis - Broadband Propagation
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    Comprehensive multi-frequency propagation analysis across octave bands.
    Test models across wide frequency range to understand spectral behavior.

COMPLEXITY LEVEL: ⭐⭐⭐⭐ (4/5) - Advanced (Comprehensive)

ENVIRONMENT:
    - Medium depth water (200m)
    - Summer thermocline profile:
        * Warm mixed layer (0-25m): 1525 m/s
        * Strong thermocline (25-60m): -3.0 (m/s)/m
        * Isothermal deep layer (60-200m): 1490 m/s
    - Elastic sediment bottom
    - Flat bathymetry

SOURCE:
    - Shallow depth (15m, above thermocline)
    - Multiple frequencies: 25, 50, 100, 200 Hz
    - Point source

MODELS TESTED:
    ✓ Bellhop      (ray tracing)
    ✓ RAM          (parabolic equation)
    ✓ KrakenField  (normal modes)
    ✓ Scooter      (wavenumber integration)    ✓ OAST         (wavenumber integration)

FEATURES DEMONSTRATED:
    - Broadband propagation characteristics
    - Frequency-dependent thermocline interaction
    - Modal cutoff frequencies
    - Frequency-dependent bottom loss
    - Wavelength effects on interference
    - Model performance across frequency bands
    - Surface duct trapping vs frequency

EXPECTED BEHAVIOR:
    - Low freq (25 Hz): Deep penetration, minimal duct trapping
    - Mid freq (50-100 Hz): Moderate ducting, clear interference
    - High freq (200 Hz): Strong surface duct, fine interference
    - Bottom loss increases with frequency
    - More modes at higher frequencies

NOTE:
    This example runs at a single reference frequency. For true multi-frequency
    analysis, run separately at each frequency and compare results.

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import uacpy  # noqa: E402
from uacpy.core.environment import SoundSpeedProfile  # noqa: E402
from uacpy.models import Bellhop, RAM, KrakenField, Scooter, OAST  # noqa: E402
from example_helpers import create_example_report  # noqa: E402


def main():
    print("\n" + "═" * 80)
    print("EXAMPLE 03: Multi-Frequency Analysis - Broadband Propagation")
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
        depths=15.0,       # In warm mixed layer (surface duct)
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
        ('KrakenField', KrakenField(verbose=False)),
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
            title="Multi-Frequency Analysis - Broadband Propagation",
            description="Thermocline environment demonstrating frequency-dependent propagation, "
            "surface duct trapping, and modal characteristics across frequencies.",
            env=env,
            source=source,
            receiver=receiver,
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
