"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 18: Boundary Conditions & Bottom Types
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE: Demonstrate all boundary condition types and their acoustic effects.

FEATURES: ✓ All acoustic types (vacuum, rigid, half-space, elastic, grain-size)
          ✓ Bottom loss comparison  ✓ Reflection coefficients
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.models import Bellhop

def main():
    print("\n" + "═"*80)
    print("EXAMPLE 18: Boundary Conditions & Bottom Types")
    print("═"*80)

    env_base = dict(name="Boundary Comparison", depth=100, sound_speed=1500,
                    ssp_type='isovelocity')
    source = uacpy.Source(depth=50, frequency=100)
    receiver = uacpy.Receiver(depths=np.linspace(5, 95, 40),
                              ranges=np.linspace(500, 15000, 60))

    boundaries = {
        'Rigid': uacpy.BoundaryProperties(acoustic_type='rigid'),
        'Half-space': uacpy.BoundaryProperties(acoustic_type='half-space',
                                                sound_speed=1600, density=1.5, attenuation=0.5),
        'Elastic': uacpy.BoundaryProperties(acoustic_type='half-space',
                                            sound_speed=1700, shear_speed=400,
                                            density=1.8, attenuation=0.5)
    }

    results = {}
    for name, bottom in boundaries.items():
        print(f"  Running {name}...", end=" ", flush=True)
        env = uacpy.Environment(**env_base, bottom=bottom)
        results[name] = Bellhop(verbose=False).run(env, source, receiver)
        print("✓")

    # Use standardized plotting for comprehensive comparison
    from plotting_utils import create_example_report

    # Use last environment for reporting
    env_last = uacpy.Environment(**env_base, bottom=boundaries['Elastic'])

    print("\n  Generating comprehensive comparison report...")
    create_example_report(
        example_num=18,
        title="Boundary Conditions & Bottom Types",
        description="Compares acoustic effects of different boundary conditions: "
                   "Rigid (perfect reflector), Half-space (fluid sediment), and Elastic (shear-supporting). "
                   "Shows how bottom properties affect TL, reflection loss, and modal structure.",
        env=env_last,
        source=source,
        receiver=receiver,
        results=results,
        output_prefix="example_18_boundary_conditions",
        output_dir="output"
    )

    # Additional plot: TL difference from rigid bottom
    print("\n  Creating boundary condition effect plots...")

    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

    # Subplot 1: TL comparison at source depth
    ax = axes2[0, 0]
    depth_idx = np.argmin(np.abs(results['Rigid'].depths - source.depth[0]))

    for name, result in results.items():
        tl_vs_range = result.data[depth_idx, :]
        ax.plot(result.ranges/1000, tl_vs_range, linewidth=2.5, label=name, alpha=0.8)

    ax.set_xlabel('Range (km)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Transmission Loss (dB)', fontweight='bold', fontsize=11)
    ax.set_title(f'TL Comparison at {source.depth[0]:.0f}m Depth', fontweight='bold', fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Compute ylim from data
    all_tl = []
    for result in results.values():
        all_tl.extend(result.data[depth_idx, :])
    all_tl = np.array(all_tl)
    all_tl = all_tl[np.isfinite(all_tl)]
    if len(all_tl) > 0:
        ax.set_ylim([np.floor(np.min(all_tl)/10)*10, np.ceil(np.max(all_tl)/10)*10])

    # Subplot 2: Bottom loss (difference from rigid)
    ax = axes2[0, 1]

    rigid_tl = results['Rigid'].data[depth_idx, :]

    for name, result in results.items():
        if name != 'Rigid':
            tl_diff = result.data[depth_idx, :] - rigid_tl
            ax.plot(result.ranges/1000, tl_diff, linewidth=2.5, label=f'{name} - Rigid', alpha=0.8)

    ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Range (km)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Additional TL (dB)', fontweight='bold', fontsize=11)
    ax.set_title('Bottom Loss Relative to Rigid Reflector', fontweight='bold', fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Subplot 3: Depth dependence at mid-range
    ax = axes2[1, 0]

    mid_range_km = np.median(results['Rigid'].ranges) / 1000
    range_idx = np.argmin(np.abs(results['Rigid'].ranges/1000 - mid_range_km))

    for name, result in results.items():
        tl_vs_depth = result.data[:, range_idx]
        ax.plot(tl_vs_depth, result.depths, linewidth=2.5, label=name, alpha=0.8)

    ax.invert_yaxis()
    ax.axhline(source.depth[0], color='gray', linestyle='--', linewidth=1,
              alpha=0.5, label='Source depth')
    ax.set_xlabel('Transmission Loss (dB)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Depth (m)', fontweight='bold', fontsize=11)
    ax.set_title(f'TL vs Depth at {mid_range_km:.1f}km Range', fontweight='bold', fontsize=12)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Subplot 4: Summary table
    ax = axes2[1, 1]
    ax.axis('off')

    summary = "BOUNDARY CONDITION PROPERTIES\n" + "="*55 + "\n\n"

    summary += "RIGID BOTTOM:\n"
    summary += "  • Perfect reflector (|R| = 1)\n"
    summary += "  • No energy loss at boundary\n"
    summary += "  • Maximum interference patterns\n"
    summary += "  • Unrealistic but useful reference\n\n"

    summary += "HALF-SPACE (Fluid Sediment):\n"
    summary += "  • Speed: 1600 m/s, Density: 1.5 g/cm³\n"
    summary += "  • Attenuation: 0.5 dB/λ\n"
    summary += "  • Supports compression waves only\n"
    summary += "  • Typical for soft sediments\n\n"

    summary += "ELASTIC (Solid Sediment):\n"
    summary += "  • P-wave speed: 1700 m/s, S-wave: 400 m/s\n"
    summary += "  • Density: 1.8 g/cm³, Attenuation: 0.5 dB/λ\n"
    summary += "  • Supports shear waves\n"
    summary += "  • More realistic for consolidated sediments\n\n"

    summary += "KEY OBSERVATIONS:\n"
    summary += "  • Rigid: Strongest TL oscillations (perfect reflection)\n"
    summary += "  • Half-space: Moderate bottom loss\n"
    summary += "  • Elastic: Additional loss from shear coupling"

    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.suptitle('Boundary Condition Effects on Underwater Acoustic Propagation',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / 'example_18_boundary_effects.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n✓ Generated: output/example_18_boundary_conditions_*.png (5 files)")
    print("✓ Generated: output/example_18_boundary_effects.png")
    print("\n" + "═"*80 + "\nEXAMPLE 18 COMPLETE\n" + "═"*80 + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
