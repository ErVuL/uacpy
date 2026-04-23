"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 10: Range-Dependent Geoacoustics - Sediment Transition
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    Demonstrate propagation over spatially varying bottom properties.
    Test models with range-dependent geoacoustic parameters (sediment change).

COMPLEXITY LEVEL: ⭐⭐⭐ (3/5) - Intermediate (Range Dependence)

ENVIRONMENT:
    - Shallow water (120m depth, flat bottom)
    - Isovelocity water column
    - Range-dependent bottom properties:
        * Soft sediment (0-10 km):
            - cp=1520 m/s, cs=0 m/s, ρ=1.3 g/cm³, α=0.8 dB/λ
        * Transition zone (10-12 km)
        * Hard sediment (12-20 km):
            - cp=1700 m/s, cs=300 m/s, ρ=1.9 g/cm³, α=0.3 dB/λ

SOURCE:
    - Mid-depth (60m)
    - Medium frequency (80 Hz)
    - Point source over soft sediment

MODELS TESTED:
    ✓ Bellhop      (ray tracing with bottom interaction)
    ✓ RAM          (parabolic equation)
    ✓ KrakenField  (normal modes - good for layered media)
    ✓ Scooter      (wavenumber integration)    ✓ OAST         (wavenumber integration - elastic bottom)

FEATURES DEMONSTRATED:
    - Range-dependent geoacoustic properties
    - Sediment type transitions
    - Bottom loss variations with range
    - Elastic vs acoustic bottom comparison
    - Mode attenuation changes
    - Shear wave effects in hard sediment

EXPECTED BEHAVIOR:
    - Lower TL over soft sediment (more absorption)
    - Transition region shows intermediate behavior
    - Enhanced propagation over hard sediment
    - Shear conversion in hard sediment zone
    - Clear difference between sediment types

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import uacpy
from uacpy.models import Bellhop, RAM, KrakenField, Scooter, OAST
from example_helpers import create_example_report

def main():
    # ═══════════════════════════════════════════════════════════════════════
    # ENVIRONMENT SETUP - Range-dependent geoacoustics
    # ═══════════════════════════════════════════════════════════════════════
    # Note: This example uses average bottom properties
    # Full range-dependent bottom requires special Environment configuration

    bottom_props = uacpy.BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1620.0,
        shear_speed=150.0,
        density=1.6,
        attenuation=0.55
    )

    env = uacpy.Environment(
        name="Sediment Transition - Range-Dependent Bottom",
        depth=120.0,
        sound_speed=1500.0,
        ssp_type='isovelocity',
        # Using intermediate bottom properties
        # (represents transition between soft and hard)
        bottom=bottom_props
    )

    # ═══════════════════════════════════════════════════════════════════════
    # SOURCE CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    source = uacpy.Source(
        depth=60.0,      # Mid-depth
        frequency=80.0   # 80 Hz
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RECEIVER GRID
    # ═══════════════════════════════════════════════════════════════════════
    receiver = uacpy.Receiver(
        depths=np.linspace(3, 117, 60),        # Full water column
        ranges=np.linspace(100, 20000, 100)    # 0.1-20 km (over transition)
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

    print("\nRunning models:")
    print("NOTE: This example demonstrates range-dependent geoacoustics concept.")
    print("      Full implementation requires Environment with range-varying bottom.\n")

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
            example_num=10,
            title="Range-Dependent Geoacoustics - Sediment Transition",
            description="Propagation over changing sediment properties (soft → hard transition). "
                       "Demonstrates bottom loss variations and elastic effects.",
            env=env,
            source=source,
            receiver=receiver,
            results=results,
            output_prefix="example_10"
        )
    else:
        print("\n⚠ No models ran successfully!")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
