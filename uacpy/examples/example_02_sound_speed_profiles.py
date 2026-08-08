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
    ✓ Bilinear mixed-layer / thermocline profile
    ✓ SSP interpolation types (munk, linear, bilinear)
    ✓ Elastic bottom properties (compressional + shear waves)
    ✓ Deep water convergence zones
    ✓ Modal propagation in waveguides
    ✓ Four propagation models compared

SCENARIOS:

    Scenario A: Munk Profile - Deep Ocean Sound Channel
    ────────────────────────────────────────────────────
    - Water depth: 5000m (deep ocean)
    - SSP type: Munk canonical profile
    - Sound channel axis: 1300m depth (``SoundSpeedProfile.from_munk`` puts the
      c_min = 1500 m/s axis at 1300 m; c(1000 m) = 1501.4 m/s)
    - Physics: SOFAR channel trapping, convergence zones, long-range propagation
    - Expected: Channel trapping, low propagation loss

    Scenario B: Pekeris Waveguide - Elastic Bottom
    ───────────────────────────────────────────────
    - Water depth: 100m (shallow water)
    - SSP type: Isovelocity (1500 m/s)
    - Bottom: Elastic half-space (cp=1700 m/s, cs=400 m/s, density=1.8 g/cm³)
    - Physics: Modal propagation with bottom loss, shear wave conversion
    - Expected: Fewer modes than rigid bottom, increased attenuation

    Scenario C: Mixed Layer over a Thermocline
    ───────────────────────────────────────────
    - Water depth: 200m (coastal/shelf)
    - SSP type: Bilinear (zero-gradient mixed layer + thermocline)
    - Mixed layer: 0-50m, dc/dz = 0 exactly
    - Thermocline: 50-200m, dc/dz = -0.27 (m/s)/m
    - Physics: downward refraction below the layer base; no surface duct
    - This layer does NOT duct, on two independent counts, and the scenario is
      here to make both explicit:
        1. Trapping needs dc/dz > 0. In a genuinely isothermal layer the
           pressure term alone gives about +0.017 (m/s)/m (Etter, *Underwater
           Acoustic Modeling and Simulation*, §3.7); a flat 1520 m/s layer has
           zero gradient, so the trapped launch-angle half-width
           sqrt(2*g*H/c) is exactly 0.
        2. Even a real 50 m duct has a low-frequency cutoff. Etter §3.7.3 gives
           lambda_max = 8.51e-3 * H**1.5 = 3.01 m, i.e. f_cutoff ~ 505 Hz; this
           scenario runs at 100 Hz, five times below it. (Jensen, Kuperman,
           Porter & Schmidt, *Computational Ocean Acoustics* 2nd ed. §1.4: "a
           50-m mixed layer acts as a sound channel only for frequencies above
           530 Hz.")
      For a duct that does trap, see example_27 Part 2: a sound-speed maximum
      at 30 m run at 2 kHz, above its own ~1073 Hz cutoff.

SOURCE:
    - Scenario A: 1000m (inside the channel, 300m above the 1300m axis)
    - Scenario B: 50m (mid-depth)
    - Scenario C: 30m (in the mixed layer)
    - Frequencies: 25 Hz (deep water), 50 Hz (shallow), 100 Hz (coastal)

MODELS COMPARED:
    1. Bellhop      - Ray tracing            (all three scenarios)
    2. RAM          - Parabolic equation     (scenarios B and C)
    3. Kraken       - Normal modes           (all three scenarios)
    4. Scooter      - Wavenumber integration (scenarios B and C)

LEARNING OUTCOMES:
    - How SSP shape affects propagation
    - Sound channel trapping mechanisms
    - Elastic bottom effects on modes
    - What a mixed layer does, and what it takes for one to duct
    - When to use each SSP type

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import uacpy  # noqa: E402
from uacpy.core.environment import SoundSpeedProfile  # noqa: E402
from uacpy.models import Bellhop, RAM, Kraken, Scooter  # noqa: E402
from plotting_utils import create_example_report  # noqa: E402


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
        bathymetry=5000.0,            # Deep ocean
        ssp=SoundSpeedProfile.from_munk(5000.0),
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
    # SOURCE CONFIGURATION - Inside the sound channel
    # ═══════════════════════════════════════════════════════════════════════
    source = uacpy.Source(
        depths=1000.0,       # Inside the channel, 300 m above the 1300 m Munk axis
        frequencies=25.0,     # Low frequency for long range
        source_type='point'
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RECEIVER GRID - Deep water column, long range
    # ═══════════════════════════════════════════════════════════════════════
    receiver = uacpy.Receiver(
        depths=np.linspace(100, 4900, 25),
        ranges=np.linspace(1000, 20000, 25),
        receiver_type='grid'
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RUN MODELS
    # ═══════════════════════════════════════════════════════════════════════
    results = {}
    models = [
        ('Bellhop', Bellhop(verbose=False)),
        ('Kraken', Kraken(verbose=False)),
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
            "The Munk axis (c_min = 1500 m/s) sits at 1300 m; the source at 1000 m is "
            "300 m above it, still well inside the channel.",
            env=env,
            source=source, receiver=receiver,
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
        acoustic_type='half-space',     # Elastic is signalled via shear_speed > 0
        sound_speed=1700.0,             # Compressional wave speed (m/s)
        shear_speed=400.0,              # Shear wave speed (m/s) - KEY for elastic
        density=1.8,                    # Density (g/cm³)
        attenuation=0.5,                # Compressional attenuation (dB/wavelength)
        shear_attenuation=0.8           # Shear attenuation (dB/wavelength)
    )

    env = uacpy.Environment(
        name="Pekeris Waveguide - Elastic Bottom",
        bathymetry=100.0,             # Shallow water
        ssp=1500.0,      # Isovelocity water column
        surface=uacpy.BoundaryProperties(
            acoustic_type='vacuum'
        ),
        bottom=bottom_props
    )

    # ═══════════════════════════════════════════════════════════════════════
    # SOURCE CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    source = uacpy.Source(
        depths=50.0,         # Mid-depth
        frequencies=50.0,     # Medium frequency
        source_type='point'
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RECEIVER GRID
    # ═══════════════════════════════════════════════════════════════════════
    receiver = uacpy.Receiver(
        depths=np.linspace(3, 97, 30),
        ranges=np.linspace(100, 5000, 40),
        receiver_type='grid'
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RUN MODELS
    # ═══════════════════════════════════════════════════════════════════════
    results = {}
    models = [
        ('Bellhop', Bellhop(verbose=False)),
        ('RAM', RAM(verbose=False)),
        ('Kraken', Kraken(verbose=False)),
        ('Scooter', Scooter(verbose=False)),
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
    # MODE COUNT: elastic seabed vs a rigid reference
    # ═══════════════════════════════════════════════════════════════════════
    env_rigid = uacpy.Environment(
        name="Pekeris Waveguide - Rigid Bottom",
        bathymetry=100.0,
        ssp=1500.0,
        surface=uacpy.BoundaryProperties(acoustic_type='vacuum'),
        bottom=uacpy.BoundaryProperties(acoustic_type='rigid')
    )
    n_elastic = Kraken(verbose=False, backend='krakenc').compute_modes(env, source).n_modes
    n_rigid = Kraken(verbose=False, backend='kraken').compute_modes(env_rigid, source).n_modes
    print(f"\n  Trapped modes at {source.frequencies[0]:.0f} Hz:"
          f" elastic seabed {n_elastic}, rigid reference {n_rigid}")

    # ═══════════════════════════════════════════════════════════════════════
    # GENERATE COMPREHENSIVE REPORT
    # ═══════════════════════════════════════════════════════════════════════
    if any(r is not None for r in results.values()):
        create_example_report(
            example_num="2B",
            title="Sound Speed Profiles - Pekeris Elastic Bottom",
            description="Classic Pekeris waveguide with elastic seafloor. Demonstrates modal "
            "propagation with bottom loss and shear wave conversion at the interface. "
            f"Kraken finds {n_elastic} modes here against {n_rigid} for the same guide "
            "closed by a rigid bottom.",
            env=env,
            source=source, receiver=receiver,
            results=results,
            output_prefix="example_02b_pekeris"
        )
    else:
        print("\n⚠ No models ran successfully!")
        return None

    return results


def scenario_c_thermocline():
    """
    Scenario C: Mixed layer over a thermocline.

    Coastal/shelf environment with a bilinear SSP. The upper layer has zero
    gradient, so it does not duct; the lesson is the thermocline's downward
    refraction plus the two conditions a real duct would have to satisfy.
    """
    print("\n" + "="*80)
    print("SCENARIO C: Mixed Layer over a Thermocline")
    print("="*80)

    # ═══════════════════════════════════════════════════════════════════════
    # ENVIRONMENT SETUP - Bilinear mixed-layer / thermocline profile
    # ═══════════════════════════════════════════════════════════════════════
    ssp_data = [
        (0,   1520),      # Surface (warm, fast sound speed)
        (50,  1520),      # Base of the mixed layer: dc/dz = 0 through the layer
        (200, 1480)       # Bottom (cooler water, slower sound speed)
    ]

    env = uacpy.Environment(
        name="Coastal - Mixed Layer over Thermocline",
        bathymetry=200.0,
        ssp=SoundSpeedProfile.from_pairs(ssp_data),
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
        depths=30.0,         # Within the mixed layer (0-50m)
        frequencies=100.0,    # Higher frequency for coastal environment
        source_type='point'
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RECEIVER GRID
    # ═══════════════════════════════════════════════════════════════════════
    receiver = uacpy.Receiver(
        depths=np.linspace(3, 197, 30),
        ranges=np.linspace(500, 5000, 40),
        receiver_type='grid'
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RUN MODELS
    # ═══════════════════════════════════════════════════════════════════════
    results = {}
    models = [
        ('Bellhop', Bellhop(verbose=False)),
        ('RAM', RAM(verbose=False)),
        ('Kraken', Kraken(verbose=False)),
        ('Scooter', Scooter(verbose=False)),
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
            title="Sound Speed Profiles - Mixed Layer over Thermocline",
            description="Coastal environment with a bilinear SSP: a zero-gradient mixed layer "
            "(0-50m) over a -0.27 (m/s)/m thermocline. The layer does not duct - trapping "
            "needs dc/dz > 0, and a 50m duct would in any case cut off near 505 Hz, five "
            "times this scenario's 100 Hz. What the fields show is the thermocline "
            "refracting energy downward below 50m.",
            env=env,
            source=source, receiver=receiver,
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
    print("\n" + "═" * 80)
    print("EXAMPLE 02: Sound Speed Profiles - Munk, Pekeris, Thermocline")
    print("═" * 80)
    print("\nThis example demonstrates:")
    print("  • Munk canonical profile (SOFAR channel, axis at 1300 m)")
    print("  • Pekeris waveguide (elastic bottom)")
    print("  • Bilinear mixed layer over a thermocline")
    print("  • SSP interpolation types (munk, isovelocity, bilinear)")
    print("  • Elastic bottom properties")
    print("  • Deep vs shallow vs coastal acoustics")

    # Run all three scenarios
    scenario_a_munk_profile()
    scenario_b_pekeris_waveguide()
    scenario_c_thermocline()

    # Summary
    print("\nKey Takeaways:")
    print("  ✓ Munk profile: Deep water sound channel, convergence zones, ultra-long range")
    print("  ✓ Pekeris: Elastic bottom reduces mode count, increases bottom loss")
    print("  ✓ Mixed layer: a zero-gradient layer does not duct - trapping needs dc/dz > 0,")
    print("    and a duct of thickness H cuts off below c / (8.51e-3 * H**1.5)")
    print("  ✓ SSP shape profoundly affects propagation physics")
    print("\nNext Steps:")
    print("  → Example 03: Multi-frequency / broadband")
    print("  → Example 06: Kraken advanced (mode analysis)")
    print("  → Example 23: Range-dependent collapse methods")

    print("\n✓ Example 02 complete\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
