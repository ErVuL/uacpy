"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 04: Seamount - Complex Bathymetry Feature
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    Demonstrate propagation over a seamount with complex 3D-like bathymetry.
    Test models with challenging range-dependent bottom topography.

COMPLEXITY LEVEL: ⭐⭐⭐ (3/5) - Intermediate (Range Dependence)

ENVIRONMENT:
    - Range-dependent bathymetry with seamount:
        * Deep water baseline: 1000m
        * Seamount peak: rises to 200m depth
        * Seamount centered at 15 km range
        * Gaussian-shaped feature (5 km width)
    - Munk-like SSP (modified for intermediate depth)
    - Hard rock seamount (high reflection)

SOURCE:
    - Mid-water depth (500m)
    - Medium-low frequency (35 Hz)
    - Point source

MODELS TESTED:
    ✓ Bellhop      (ray tracing - excellent for complex bathymetry)
    ✓ RAM          (parabolic equation - handles topography)
    ✓ KrakenField  (normal modes - challenging for large depth changes)
    ✓ Scooter      (wavenumber integration)    ✓ OAST         (wavenumber integration)

FEATURES DEMONSTRATED:
    - Complex range-dependent bathymetry
    - Seamount scattering and diffraction
    - Shadow zone formation behind obstacle
    - Multipath around/over bathymetric feature
    - Bottom bounce focusing effects
    - Model performance with large depth variations

EXPECTED BEHAVIOR:
    - Strong bottom interaction over seamount
    - Shadow zone and focusing beyond seamount
    - Multipath enhancement near peak
    - Significant scattering from steep slopes
    - Ray methods may outperform in this scenario

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
    # ENVIRONMENT SETUP - Seamount bathymetry
    # ═══════════════════════════════════════════════════════════════════════
    # Create Gaussian seamount bathymetry
    ranges_bathy = np.linspace(0, 30000, 100)
    seamount_center = 15000  # 15 km
    seamount_width = 5000    # 5 km std dev
    seamount_height = 800    # Rise 800m from 1000m baseline

    # Gaussian seamount shape
    depths_bathy = 1000 - seamount_height * np.exp(
        -((ranges_bathy - seamount_center) ** 2) / (2 * seamount_width ** 2)
    )
    bathymetry = np.column_stack([ranges_bathy, depths_bathy])

    # Modified Munk-like profile for intermediate depth
    bottom_props = uacpy.BoundaryProperties(
        acoustic_type='half-space',  # Acousto-elastic half-space (hard rock seamount)
        sound_speed=3500.0,
        density=2.5,
        attenuation=0.1
    )

    env = uacpy.Environment(
        name="Seamount - Complex Bathymetry",
        depth=1000.0,
        ssp_type='munk',
        bathymetry=bathymetry,
        bottom=bottom_props
    )

    # ═══════════════════════════════════════════════════════════════════════
    # SOURCE CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    source = uacpy.Source(
        depth=500.0,     # Mid-water
        frequency=35.0   # Medium-low frequency
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RECEIVER GRID
    # ═══════════════════════════════════════════════════════════════════════
    receiver = uacpy.Receiver(
        depths=np.linspace(50, 950, 200),       # Fine depth grid for modal resolution
        ranges=np.linspace(500, 30000, 200)     # Fine range grid
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
            example_num=4,
            title="Seamount - Complex Bathymetry Feature",
            description="Propagation over Gaussian seamount (1000m → 200m → 1000m) demonstrating "
                       "scattering, diffraction, and shadow zone formation.",
            env=env,
            source=source,
            receiver=receiver,
            results=results,
            output_prefix="example_04"
        )
    else:
        print("\n⚠ No models ran successfully!")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
