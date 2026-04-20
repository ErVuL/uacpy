"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 02: Sound Speed Profiles - Munk, Pekeris, Thermocline
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    Demonstrate different SSP types and their acoustic effects.
    Compare model performance with complex sound speed structures.

COMPLEXITY LEVEL: ⭐⭐ (2/5) - Foundational

FEATURES DEMONSTRATED:
    ✓ Munk canonical profile (SOFAR channel, deep ocean)
    ✓ Pekeris waveguide (isovelocity with elastic bottom)
    ✓ Bilinear thermocline profile (surface duct)
    ✓ SSP interpolation types (munk, linear, bilinear)
    ✓ Elastic bottom properties (compressional + shear waves)
    ✓ Deep water convergence zones
    ✓ Modal propagation in waveguides
    ✓ Six propagation models comparison

SCENARIOS:

    Scenario A: Munk Profile - Deep Ocean Sound Channel
    ────────────────────────────────────────────────────
    - Water depth: 5000m (deep ocean)
    - SSP type: Munk canonical profile
    - Sound channel axis: ~1000m depth
    - Physics: SOFAR channel trapping, convergence zones, long-range propagation
    - Expected: Strong channel trapping, very low propagation loss

    Scenario B: Pekeris Waveguide - Elastic Bottom
    ───────────────────────────────────────────────
    - Water depth: 100m (shallow water)
    - SSP type: Isovelocity (1500 m/s)
    - Bottom: Elastic half-space (cp=1700 m/s, cs=400 m/s, density=1.8 g/cm³)
    - Physics: Modal propagation with bottom loss, shear wave conversion
    - Expected: Fewer modes than rigid bottom, increased attenuation

    Scenario C: Thermocline - Surface Duct
    ───────────────────────────────────────
    - Water depth: 200m (coastal/shelf)
    - SSP type: Bilinear (surface duct + thermocline)
    - Surface duct: 0-50m (isothermal layer)
    - Thermocline: 50-200m (negative gradient)
    - Physics: Surface duct trapping, thermocline refraction
    - Expected: Strong surface duct at source depth, downward refraction below

SOURCE:
    - Scenario A: 1000m (channel axis)
    - Scenario B: 50m (mid-depth)
    - Scenario C: 30m (in surface duct)
    - Frequencies: 25 Hz (deep water), 50 Hz (shallow), 100 Hz (coastal)

MODELS COMPARED:
    1. Bellhop      - Ray tracing
    2. RAM          - Parabolic equation
    3. KrakenField  - Normal modes
    4. Scooter      - Wavenumber integration
    5. SPARC        - Seismo-acoustic PE
    6. OAST         - OASES wavenumber integration

LEARNING OUTCOMES:
    - How SSP shape affects propagation
    - Sound channel trapping mechanisms
    - Elastic bottom effects on modes
    - Thermocline and surface duct physics
    - When to use each SSP type

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import uacpy
from uacpy.models import Bellhop, RAM, KrakenField, Scooter, OAST
from example_helpers import create_example_report


def scenario_a_munk_profile():
    """
    Scenario A: Munk canonical profile - Deep ocean SOFAR channel.

    Classic deep water sound channel with Munk profile demonstrating
    long-range propagation and convergence zone formation.
    """
    print("\n" + "="*80)
    print("SCENARIO A: Munk Profile - Deep Ocean Sound Channel")
    print("="*80)

    # ═══════════════════════════════════════════════════════════════════════
    # ENVIRONMENT SETUP - Munk canonical profile
    # ═══════════════════════════════════════════════════════════════════════
    env = uacpy.Environment(
        name="Deep Ocean - Munk Profile",
        depth=5000.0,            # Deep ocean
        ssp_type='munk',         # Canonical Munk profile (built-in)
        surface=uacpy.BoundaryProperties(
            acoustic_type='vacuum'
        ),
        bottom=uacpy.BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600,
            density=1.5,
            attenuation=0.2
        )
    )

    # ═══════════════════════════════════════════════════════════════════════
    # SOURCE CONFIGURATION - At channel axis
    # ═══════════════════════════════════════════════════════════════════════
    source = uacpy.Source(
        depth=1000.0,       # Near channel axis (~1000m for Munk profile)
        frequency=25.0,     # Low frequency for long range
        source_type='point'
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RECEIVER GRID - Deep water column, long range
    # ═══════════════════════════════════════════════════════════════════════
    receiver = uacpy.Receiver(
        depths=np.linspace(100, 4900, 50),      # Deep water (avoid boundaries)
        ranges=np.linspace(1000, 100000, 100),  # 1-100 km
        receiver_type='grid'
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

    print("\nRunning propagation models:")
    for name, model in models:
        print(f"  {name:15s}...", end=" ", flush=True)
        try:
            results[name] = model.run(env, source, receiver)
            print("✓")
        except Exception as e:
            print(f"✗ {e}")
            results[name] = None

    # ═══════════════════════════════════════════════════════════════════════
    # GENERATE COMPREHENSIVE REPORT
    # ═══════════════════════════════════════════════════════════════════════
    if any(r is not None for r in results.values()):
        create_example_report(
            example_num="2A",
            title="Sound Speed Profiles - Munk Deep Ocean",
            description="Deep water sound channel with Munk canonical profile. Demonstrates "
                       "SOFAR channel trapping, convergence zones, and ultra-long-range propagation. "
                       "Channel axis at ~1000m depth provides optimal trapping.",
            env=env,
            source=source,
            receiver=receiver,
            results=results,
            output_prefix="example_02a_munk"
        )
    else:
        print("\n⚠ No models ran successfully!")
        return None

    return results


def scenario_b_pekeris_waveguide():
    """
    Scenario B: Pekeris waveguide with elastic bottom.

    Classic shallow water waveguide demonstrating elastic bottom
    effects, modal propagation, and shear wave conversion.
    """
    print("\n" + "="*80)
    print("SCENARIO B: Pekeris Waveguide - Elastic Bottom")
    print("="*80)

    # ═══════════════════════════════════════════════════════════════════════
    # ENVIRONMENT SETUP - Pekeris waveguide with elastic bottom
    # ═══════════════════════════════════════════════════════════════════════
    # Define elastic bottom properties
    bottom_props = uacpy.BoundaryProperties(
        acoustic_type='elastic',        # Elastic half-space
        sound_speed=1700.0,             # Compressional wave speed (m/s)
        shear_speed=400.0,              # Shear wave speed (m/s) - KEY for elastic
        density=1.8,                    # Density (g/cm³)
        attenuation=0.5,                # Compressional attenuation (dB/wavelength)
        shear_attenuation=0.8           # Shear attenuation (dB/wavelength)
    )

    env = uacpy.Environment(
        name="Pekeris Waveguide - Elastic Bottom",
        depth=100.0,             # Shallow water
        sound_speed=1500.0,      # Isovelocity water column
        ssp_type='isovelocity',
        surface=uacpy.BoundaryProperties(
            acoustic_type='vacuum'
        ),
        bottom=bottom_props
    )

    # ═══════════════════════════════════════════════════════════════════════
    # SOURCE CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    source = uacpy.Source(
        depth=50.0,         # Mid-depth
        frequency=50.0,     # Medium frequency
        source_type='point'
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RECEIVER GRID
    # ═══════════════════════════════════════════════════════════════════════
    receiver = uacpy.Receiver(
        depths=np.linspace(3, 97, 60),          # Full water column (margins)
        ranges=np.linspace(100, 15000, 100),    # 0.1-15 km
        receiver_type='grid'
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

    print("\nRunning propagation models:")
    for name, model in models:
        print(f"  {name:15s}...", end=" ", flush=True)
        try:
            results[name] = model.run(env, source, receiver)
            print("✓")
        except Exception as e:
            print(f"✗ {e}")
            results[name] = None

    # ═══════════════════════════════════════════════════════════════════════
    # GENERATE COMPREHENSIVE REPORT
    # ═══════════════════════════════════════════════════════════════════════
    if any(r is not None for r in results.values()):
        create_example_report(
            example_num="2B",
            title="Sound Speed Profiles - Pekeris Elastic Bottom",
            description="Classic Pekeris waveguide with elastic seafloor. Demonstrates modal "
                       "propagation with bottom loss, shear wave conversion at interface, and "
                       "reduced mode count compared to rigid bottom.",
            env=env,
            source=source,
            receiver=receiver,
            results=results,
            output_prefix="example_02b_pekeris"
        )
    else:
        print("\n⚠ No models ran successfully!")
        return None

    return results


def scenario_c_thermocline():
    """
    Scenario C: Thermocline with surface duct.

    Coastal/shelf environment with bilinear SSP demonstrating
    surface duct trapping and thermocline refraction effects.
    """
    print("\n" + "="*80)
    print("SCENARIO C: Thermocline - Surface Duct")
    print("="*80)

    # ═══════════════════════════════════════════════════════════════════════
    # ENVIRONMENT SETUP - Bilinear thermocline profile
    # ═══════════════════════════════════════════════════════════════════════
    # Create bilinear SSP: surface duct + thermocline
    ssp_data = [
        (0,   1520),      # Surface (warm, fast sound speed)
        (50,  1520),      # Bottom of surface duct (isothermal layer)
        (200, 1480)       # Bottom (cooler water, slower sound speed)
    ]

    env = uacpy.Environment(
        name="Coastal - Thermocline with Surface Duct",
        depth=200.0,
        ssp_type='bilinear',    # Bilinear interpolation
        ssp_data=ssp_data,
        surface=uacpy.BoundaryProperties(
            acoustic_type='vacuum'
        ),
        bottom=uacpy.BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600,
            density=1.7,
            attenuation=0.5
        )
    )

    # ═══════════════════════════════════════════════════════════════════════
    # SOURCE CONFIGURATION - In surface duct
    # ═══════════════════════════════════════════════════════════════════════
    source = uacpy.Source(
        depth=30.0,         # Within surface duct (0-50m)
        frequency=100.0,    # Higher frequency for coastal environment
        source_type='point'
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RECEIVER GRID
    # ═══════════════════════════════════════════════════════════════════════
    receiver = uacpy.Receiver(
        depths=np.linspace(3, 197, 60),         # Full water column
        ranges=np.linspace(500, 20000, 100),    # 0.5-20 km
        receiver_type='grid'
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

    print("\nRunning propagation models:")
    for name, model in models:
        print(f"  {name:15s}...", end=" ", flush=True)
        try:
            results[name] = model.run(env, source, receiver)
            print("✓")
        except Exception as e:
            print(f"✗ {e}")
            results[name] = None

    # ═══════════════════════════════════════════════════════════════════════
    # GENERATE COMPREHENSIVE REPORT
    # ═══════════════════════════════════════════════════════════════════════
    if any(r is not None for r in results.values()):
        create_example_report(
            example_num="2C",
            title="Sound Speed Profiles - Thermocline Surface Duct",
            description="Coastal environment with bilinear SSP creating surface duct. Source "
                       "within duct demonstrates strong trapping in isothermal layer (0-50m) with "
                       "downward refraction below thermocline.",
            env=env,
            source=source,
            receiver=receiver,
            results=results,
            output_prefix="example_02c_thermocline"
        )
    else:
        print("\n⚠ No models ran successfully!")
        return None

    return results


def main():
    """
    Run all three SSP scenarios.
    """
    print("\n" + "═"*80)
    print("EXAMPLE 02: Sound Speed Profiles - Munk, Pekeris, Thermocline")
    print("═"*80)
    print("\nThis example demonstrates:")
    print("  • Munk canonical profile (SOFAR channel)")
    print("  • Pekeris waveguide (elastic bottom)")
    print("  • Bilinear thermocline (surface duct)")
    print("  • SSP interpolation types (munk, isovelocity, bilinear)")
    print("  • Elastic bottom properties")
    print("  • Deep vs shallow vs coastal acoustics")

    # Run all three scenarios
    results_a = scenario_a_munk_profile()
    results_b = scenario_b_pekeris_waveguide()
    results_c = scenario_c_thermocline()

    # Summary
    print("\n" + "═"*80)
    print("EXAMPLE 02 COMPLETE")
    print("═"*80)
    print("\nKey Takeaways:")
    print("  ✓ Munk profile: Deep water sound channel, convergence zones, ultra-long range")
    print("  ✓ Pekeris: Elastic bottom reduces mode count, increases bottom loss")
    print("  ✓ Thermocline: Surface duct traps energy when source is in duct")
    print("  ✓ SSP shape profoundly affects propagation physics")
    print("\nNext Steps:")
    print("  → Example 03: Range-dependent bathymetry (sloping, shelf break)")
    print("  → Example 06: Multi-frequency analysis")
    print("  → Example 09: Kraken advanced (mode analysis)")
    print("\n" + "═"*80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
