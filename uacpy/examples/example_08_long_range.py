"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 08: Deep Water Long Range - 100+ km Propagation
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    Demonstrate very long-range propagation in deep ocean with SOFAR channel.
    Test model accuracy and efficiency for extended propagation distances.

COMPLEXITY LEVEL: ⭐⭐⭐⭐ (4/5) - Advanced (Comprehensive)

ENVIRONMENT:
    - Very deep water (5500m depth)
    - Munk canonical sound speed profile
    - Sound channel axis at 1300m (``from_munk`` puts c_min = 1500 m/s there;
      c(1000 m) = 1501.4 m/s)
    - Low-loss hard bottom (abyssal plain)
    - Range: 1-150 km

SOURCE:
    - 1000m: inside the channel, 300m above the 1300m axis
    - Very low frequency (20 Hz) for maximum range
    - Point source

MODELS TESTED:
    ✓ Bellhop      (ray tracing - efficient for long range)
    ✓ Kraken  (normal modes - very efficient for ducted long range)
    ✓ Scooter      (wavenumber integration)
    ✓ OAST         (wavenumber integration)

FEATURES DEMONSTRATED:
    - Very long-range propagation (>100 km)
    - Deep ocean SOFAR channel
    - Convergence zone formation and spacing
    - Channel trapping efficiency
    - Computational efficiency at long range
    - Model accuracy over extended distances
    - Bottom interaction at extreme range

EXPECTED BEHAVIOR:
    - Multiple convergence zones across the 150 km span
    - Strong energy trapping with minimal bottom interaction
    - Clear modal structure in channel
    - Kraken likely most efficient

    The script measures on-axis TL at 100 km for every model that runs and
    prints it, rather than asserting a threshold it never checks.

COMPUTATIONAL NOTES:
    - Long range requires fine grid spacing
    - Memory usage scales with range
    - Consider reduced resolution for very long range

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path
# Repo root, so ``import uacpy`` resolves from a source checkout.
sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np  # noqa: E402
import uacpy  # noqa: E402
from uacpy.core.environment import SoundSpeedProfile  # noqa: E402
from uacpy.models import Bellhop, Kraken, Scooter, OAST  # noqa: E402
from plotting_utils import create_example_report  # noqa: E402


def main():
    print("\n" + "═" * 80)
    print("EXAMPLE 08: Deep Water Long Range - 100+ km Propagation")
    print("═" * 80)

    # ═══════════════════════════════════════════════════════════════════════
    # ENVIRONMENT SETUP - Deep ocean Munk profile
    # ═══════════════════════════════════════════════════════════════════════
    bottom_props = uacpy.BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=3000.0,
        density=2.2,
        attenuation=0.05
    )

    env = uacpy.Environment(
        name="Deep Ocean - Long Range SOFAR",
        bathymetry=5500.0,
        ssp=SoundSpeedProfile.from_munk(5500.0),
        bottom=bottom_props
    )

    # ═══════════════════════════════════════════════════════════════════════
    # SOURCE CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    source = uacpy.Source(
        depths=1000.0,    # Inside the channel, 300 m above the 1300 m axis
        frequencies=20.0   # Very low frequency for long range
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RECEIVER GRID - Long range, coarser spacing
    # ═══════════════════════════════════════════════════════════════════════
    receiver = uacpy.Receiver(
        depths=np.linspace(100, 5400, 30),
        ranges=np.linspace(1000, 150000, 150)   # out to the advertised 150 km
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RUN MODELS
    # ═══════════════════════════════════════════════════════════════════════
    results = {}
    models = [
        ('Bellhop', Bellhop(verbose=False)),
        ('Kraken', Kraken(verbose=False)),
        ('Scooter', Scooter(verbose=False)),
        ('OAST', OAST(verbose=False)),
    ]

    print("\nRunning long-range propagation models:")
    print("WARNING: Long-range calculations may take significant time.\n")

    for name, model in models:
        print(f"  {name:15s}...", end=" ", flush=True)
        try:
            results[name] = model.run(env, source, receiver)
            print("✓")
        except Exception as e:
            print(f"✗ {e}")
            results[name] = None

    # ═══════════════════════════════════════════════════════════════════════
    # MEASURE ON-AXIS TL AT 100 km
    # ═══════════════════════════════════════════════════════════════════════
    print("\nOn-axis TL at 100 km (channel axis = 1300 m):")
    for name, result in results.items():
        if result is None:
            continue
        tl = np.asarray(result.db)
        depths_m = np.asarray(result.depths)
        ranges_m = np.asarray(result.ranges)
        i_axis = int(np.argmin(np.abs(depths_m - 1300.0)))
        i_100km = int(np.argmin(np.abs(ranges_m - 100000.0)))
        print(f"  {name:15s} {tl[i_axis, i_100km]:5.1f} dB"
              f"  (z = {depths_m[i_axis]:.0f} m, r = {ranges_m[i_100km] / 1000:.1f} km)")

    # ═══════════════════════════════════════════════════════════════════════
    # GENERATE REPORT
    # ═══════════════════════════════════════════════════════════════════════
    if any(r is not None for r in results.values()):
        create_example_report(
            example_num=8,
            title="Deep Water Long Range - 100+ km Propagation",
            description="Very long-range propagation (1-150 km) in a deep SOFAR channel, "
            "Munk profile with its axis at 1300 m. Demonstrates convergence zones, "
            "channel trapping, and model efficiency.",
            env=env,
            source=source, receiver=receiver,
            results=results,
            output_prefix="example_08"
        )
    else:
        print("\n⚠ No models ran successfully!")
        return 1

    print("\n✓ Example 08 complete\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
