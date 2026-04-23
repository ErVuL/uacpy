"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 13: Deep Water Long Range - 100+ km Propagation
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    Demonstrate very long-range propagation in deep ocean with SOFAR channel.
    Test model accuracy and efficiency for extended propagation distances.

COMPLEXITY LEVEL: ⭐⭐⭐⭐ (4/5) - Advanced (Comprehensive)

ENVIRONMENT:
    - Very deep water (5500m depth)
    - Munk canonical sound speed profile
    - Sound channel axis at ~1000m
    - Low-loss hard bottom (abyssal plain)
    - Range: 0-150 km

SOURCE:
    - Channel axis depth (1000m)
    - Very low frequency (20 Hz) for maximum range
    - Point source

MODELS TESTED:
    ✓ Bellhop      (ray tracing - efficient for long range)
    ✓ RAM          (parabolic equation - may be memory intensive)
    ✓ KrakenField  (normal modes - very efficient for ducted long range)
    ✓ Scooter      (wavenumber integration)
    ✓ SPARC        (spectral PE)
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
    - Multiple convergence zones (30-60 km spacing)
    - Very low TL in sound channel (<80 dB at 100 km)
    - Strong energy trapping with minimal bottom interaction
    - Clear modal structure in channel
    - KrakenField likely most efficient
    - RAM may require significant memory/time

COMPUTATIONAL NOTES:
    - Long range requires fine grid spacing
    - Memory usage scales with range
    - Consider reduced resolution for very long range

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import uacpy
from uacpy.models import Bellhop, RAM, KrakenField, Scooter, SPARC, OAST
from plotting_utils import create_example_report

def main():
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
        depth=5500.0,
        ssp_type='munk',
        bottom=bottom_props
    )

    # ═══════════════════════════════════════════════════════════════════════
    # SOURCE CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    source = uacpy.Source(
        depth=1000.0,    # Channel axis
        frequency=20.0   # Very low frequency for long range
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RECEIVER GRID - Long range, coarser spacing
    # ═══════════════════════════════════════════════════════════════════════
    receiver = uacpy.Receiver(
        depths=np.linspace(100, 5400, 60),      # Full deep water column
        ranges=np.linspace(1000, 150000, 150)   # 1-150 km (coarse for speed)
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
    # GENERATE REPORT
    # ═══════════════════════════════════════════════════════════════════════
    if any(r is not None for r in results.values()):
        create_example_report(
            example_num=12,
            title="Deep Water Long Range - 100+ km Propagation",
            description="Very long-range propagation (150 km) in deep SOFAR channel. "
                       "Demonstrates convergence zones, channel trapping, and model efficiency.",
            env=env,
            source=source,
            receiver=receiver,
            results=results,
            output_prefix="example_12"
        )
    else:
        print("\n⚠ No models ran successfully!")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
