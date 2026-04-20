"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 06: Sloping Bottom - Simple Range-Dependent Bathymetry
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    Introduce range-dependent bathymetry with a simple upslope scenario.
    Test models' ability to handle smoothly varying seafloor depth.

COMPLEXITY LEVEL: ⭐⭐⭐ (3/5) - Intermediate (Range Dependence)

ENVIRONMENT:
    - Range-dependent bathymetry: 200m → 50m (upslope)
    - Slope: 1:100 (1% grade, ~0.57 degrees)
    - Linear SSP: 1500 m/s (surface) → 1520 m/s (bottom)
    - Positive gradient environment
    - Smooth depth transition over 15 km

SOURCE:
    - Deep water start (100m depth)
    - Medium frequency (75 Hz)
    - Point source

MODELS TESTED:
    ✓ Bellhop      (ray tracing - excellent for range dependence)
    ✓ RAM          (parabolic equation - designed for range dependence)
    ✓ KrakenField  (normal modes - adiabatic approximation)
    ✓ Scooter      (wavenumber integration)    ✓ OAST         (wavenumber integration)

FEATURES DEMONSTRATED:
    - Range-dependent bathymetry
    - Upslope propagation
    - Adiabatic mode coupling
    - Slope-induced mode stripping
    - Bottom interaction with varying depth
    - Model comparison for range dependence

EXPECTED BEHAVIOR:
    - Energy concentration near shoaling bottom
    - Mode stripping as water depth decreases
    - Increased TL on slope due to bottom interaction
    - PE models (RAM) should handle well
    - Ray models (Bellhop) excellent for slopes

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import uacpy
from uacpy.models import Bellhop, RAM, KrakenField, Scooter, SPARC, OAST
from example_helpers import create_example_report

def main():
    # ═══════════════════════════════════════════════════════════════════════
    # ENVIRONMENT SETUP - Sloping bottom
    # ═══════════════════════════════════════════════════════════════════════
    # Create upslope bathymetry: 200m → 50m over 15 km
    ranges_bathy = np.linspace(0, 15000, 50)
    depths_bathy = 200 - (150 * ranges_bathy / 15000)  # Linear slope
    bathymetry = np.column_stack([ranges_bathy, depths_bathy])

    env = uacpy.Environment(
        name="Upslope - Range-Dependent Bathymetry",
        depth=200.0,  # Initial depth
        ssp_type='linear',
        ssp_data=[(0, 1500), (200, 1520)],  # Mild positive gradient
        bathymetry=bathymetry,
        bottom=uacpy.BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600.0,
            density=1.5,
            attenuation=0.3
        )
    )

    # ═══════════════════════════════════════════════════════════════════════
    # SOURCE CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    source = uacpy.Source(
        depth=100.0,     # Mid-depth in deep water
        frequency=75.0   # 75 Hz
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RECEIVER GRID
    # ═══════════════════════════════════════════════════════════════════════
    receiver = uacpy.Receiver(
        depths=np.linspace(3, 197, 60),        # Cover initial water column
        ranges=np.linspace(100, 15000, 100)    # 0.1-15 km (full slope)
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
        ('SPARC', SPARC(verbose=False)),
        ('OAST', OAST(verbose=False)),
    ]

    print("\nRunning models:")
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
            example_num=6,
            title="Sloping Bottom - Range-Dependent Bathymetry",
            description="Simple upslope bathymetry (200m → 50m) demonstrating range-dependent "
                       "propagation, mode stripping, and slope-induced effects.",
            env=env,
            source=source,
            receiver=receiver,
            results=results,
            output_prefix="example_06"
        )
    else:
        print("\n⚠ No models ran successfully!")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
