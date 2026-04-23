"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 01: Basic Shallow Water Propagation - Pekeris Waveguide
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    The simplest possible UACPY example - a "Hello World" for underwater acoustics.
    Demonstrates basic propagation modeling with minimal complexity.

ENVIRONMENT:
    - Pekeris waveguide (classic benchmark)
    - Flat bottom at 100m depth
    - Isovelocity water (1500 m/s)
    - Fluid bottom (no shear)

FEATURES DEMONSTRATED:
    ✓ Basic Environment setup
    ✓ Source and Receiver configuration
    ✓ Bellhop propagation model
    ✓ Simple TL visualization
    ✓ Quick start for new users

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.models import Bellhop
from uacpy.core.environment import BoundaryProperties
from uacpy.visualization.plots import plot_transmission_loss
from uacpy.models import RunMode
import os

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    print("=" * 80)
    print("EXAMPLE 01: Basic Shallow Water Propagation - Pekeris Waveguide")
    print("=" * 80)
    print("\nThis is the simplest UACPY example - a 'Hello World' for underwater acoustics!")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Create Environment
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[Step 1/4] Creating environment...")

    # Create bottom boundary properties
    bottom = BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1600.0,      # Slightly faster sediment
        density=1.5,              # Typical sediment density (g/cm³)
        attenuation=0.5           # Moderate attenuation (dB/wavelength)
    )

    env = uacpy.Environment(
        name="Pekeris Waveguide",
        depth=100.0,              # Flat bottom at 100m
        sound_speed=1500.0,       # Isovelocity water column
        ssp_type='isovelocity',
        bottom=bottom             # Use BoundaryProperties object
    )

    print(f"  ✓ Created Pekeris waveguide:")
    print(f"    - Water depth: {env.depth}m")
    print(f"    - Sound speed: {env.sound_speed} m/s")
    print(f"    - Bottom type: Fluid half-space")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Define Source
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[Step 2/4] Defining acoustic source...")

    source = uacpy.Source(
        depth=50.0,       # Mid-water column
        frequency=100.0   # 100 Hz (typical low-frequency sonar)
    )

    print(f"  ✓ Source configured:")
    print(f"    - Depth: {source.depth[0]}m")
    print(f"    - Frequency: {source.frequency[0]} Hz")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: Define Receiver Grid
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[Step 3/4] Defining receiver grid...")

    receiver = uacpy.Receiver(
        depths=np.linspace(5, 95, 50),      # 50 depths from 5m to 95m
        ranges=np.linspace(100, 10000, 100) # 100 ranges from 0.1km to 10km
    )

    print(f"  ✓ Receiver grid:")
    print(f"    - Depths: {len(receiver.depths)} points ({receiver.depths[0]}m to {receiver.depths[-1]}m)")
    print(f"    - Ranges: {len(receiver.ranges)} points ({receiver.ranges[0]/1000:.1f}km to {receiver.ranges[-1]/1000:.1f}km)")
    print(f"    - Total receivers: {len(receiver.depths) * len(receiver.ranges)}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: Run Propagation Model
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[Step 4/4] Running Bellhop propagation model...")

    bellhop = Bellhop(verbose=False, beam_type='B', n_beams=300, alpha=(-80, 80))

    try:
        result = bellhop.run(
            env, source, receiver,
            run_mode=RunMode.COHERENT_TL,        # Coherent TL
        )
        print("  ✓ Propagation complete!")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return 1

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: Visualize Results
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[Step 5/5] Creating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: TL field using NEW uacpy plotting module
    ax = axes[0, 0]
    #Use plot_transmission_loss with auto TL limits and show_colorbar=False for subplots
    _, _, cbar = plot_transmission_loss(result, env, ax=ax, show_colorbar=True)
    ax.set_title('Transmission Loss Field (auto TL limits, jet_r colormap)',
                fontweight='bold', fontsize=12)

    # Plot 2: TL vs Range (at source depth)
    ax = axes[0, 1]
    depth_idx = np.argmin(np.abs(result.depths - source.depth[0]))
    tl_vs_range = result.data[depth_idx, :]
    ax.plot(result.ranges/1000, tl_vs_range, 'b-', linewidth=2)
    ax.set_xlabel('Range (km)', fontweight='bold')
    ax.set_ylabel('Transmission Loss (dB)', fontweight='bold')
    ax.set_title(f'TL vs Range (at {source.depth[0]:.0f}m depth)', fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Compute ylim from data
    tl_finite = tl_vs_range[np.isfinite(tl_vs_range)]
    if len(tl_finite) > 0:
        ax.set_ylim([np.floor(np.min(tl_finite)/10)*10, np.ceil(np.max(tl_finite)/10)*10])

    # Plot 3: TL vs Depth (at mid-range)
    ax = axes[1, 0]
    mid_range_km = np.median(result.ranges) / 1000
    range_idx = np.argmin(np.abs(result.ranges/1000 - mid_range_km))
    tl_vs_depth = result.data[:, range_idx]
    ax.plot(tl_vs_depth, result.depths, 'r-', linewidth=2)
    ax.invert_yaxis()
    ax.axhline(source.depth[0], color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Source depth')
    ax.axhline(env.depth, color='k', linewidth=2, label='Bottom')
    ax.set_xlabel('Transmission Loss (dB)', fontweight='bold')
    ax.set_ylabel('Depth (m)', fontweight='bold')
    ax.set_title(f'TL vs Depth (at {mid_range_km:.1f}km range)', fontweight='bold', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Compute xlim from data
    tl_finite = tl_vs_depth[np.isfinite(tl_vs_depth)]
    if len(tl_finite) > 0:
        ax.set_xlim([np.floor(np.min(tl_finite)/10)*10, np.ceil(np.max(tl_finite)/10)*10])

    # Plot 4: Environment schematic
    ax = axes[1, 1]

    # Water column
    ax.fill_between([0, 10], [0, 0], [env.depth, env.depth],
                    color='lightblue', alpha=0.3, label='Water')

    # Bottom
    ax.fill_between([0, 10], [env.depth, env.depth], [env.depth*1.2, env.depth*1.2],
                    color='#8B4513', alpha=0.5, label='Bottom')

    # Seafloor line
    ax.plot([0, 10], [env.depth, env.depth], 'k-', linewidth=3, label='Seafloor')

    # Source
    ax.plot(0.5, source.depth[0], 'r*', markersize=20, label='Source', zorder=10)

    # Receiver positions (sample)
    R, Z = np.meshgrid(result.ranges/1000, result.depths)
    ax.scatter(R.flatten()[::100], Z.flatten()[::100], c='green', s=10,
               alpha=0.3, label='Receivers', zorder=5)

    ax.set_xlim([0, receiver.ranges[-1]/1000])
    ax.set_ylim([env.depth*1.2, 0])
    ax.set_xlabel('Range (km)', fontweight='bold')
    ax.set_ylabel('Depth (m)', fontweight='bold')
    ax.set_title('Environment Setup', fontweight='bold', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add text box with simulation parameters
    textstr = (f'Simulation Parameters:\n'
               f'  Water depth: {env.depth}m\n'
               f'  Sound speed: {env.sound_speed} m/s\n'
               f'  Frequency: {source.frequency[0]} Hz\n'
               f'  Source depth: {source.depth[0]}m\n'
               f'  Model: Bellhop (Gaussian beams)\n'
               f'  Grid: {len(receiver.depths)}×{len(receiver.ranges)} receivers')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props, family='monospace')

    plt.tight_layout()

    # Save figure
    plt.savefig(OUTPUT_DIR / 'example_01_basic_shallow_water.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {OUTPUT_DIR / 'example_01_basic_shallow_water.png'}")

    # ═══════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("EXAMPLE 01 COMPLETE!")
    print("=" * 80)

    print("\nResults:")
    print(f"  • TL range: {np.nanmin(result.data):.1f} to {np.nanmax(result.data):.1f} dB")
    print(f"  • Max range: {result.ranges[-1]/1000:.1f} km")
    print(f"  • Computation time: < 1 second")

    print("\nWhat you learned:")
    print("  ✓ How to create a basic Environment")
    print("  ✓ How to define Source and Receiver")
    print("  ✓ How to run Bellhop propagation model")
    print("  ✓ How to visualize transmission loss")

    print("\nNEW Plotting features used:")
    print("  ✓ plot_transmission_loss() with auto TL limits")
    print("  ✓ jet_r colormap (blue=good, red=poor) - Acoustic Toolbox standard")
    print("  ✓ Auto TL limits (median + 0.75σ, rounded to 10 dB)")

    print("\nNext steps:")
    print("  • Try example_02 for different sound speed profiles")
    print("  • Try example_03 for range-dependent bathymetry")
    print("  • Try example_11 to compare multiple models")
    print("  • Try example_21 for all new plotting features")

    print("\n" + "=" * 80 + "\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
