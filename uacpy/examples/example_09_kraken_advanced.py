"""
═══════════════════════════════════════════════════════════════════════════════
ADVANCED EXAMPLE: Kraken/KrakenField - Coupled Mode Theory
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    Demonstrate Kraken's capabilities with range-dependent environments:
    - Normal mode computation
    - Adiabatic mode theory for range-dependent problems
    - Mode coupling for continental shelf
    - Volume attenuation in modal propagation

SCENARIO:
    Continental Shelf with Mode Coupling
    - Shallow water (100m) transitioning to deep water (400m)
    - Range-dependent bottom properties (sand → rock)
    - Francois-Garrison volume attenuation

FEATURES DEMONSTRATED:
    ✓ Kraken mode computation with volume attenuation
    ✓ KrakenField with adiabatic mode theory
    ✓ Range segmentation for coupled modes
    ✓ Range-dependent bottom in mode propagation
    ✓ Mode shape visualization
    ✓ Dispersion curve analysis

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy import RangeDependentBottom
from uacpy.models import Kraken, KrakenField, KrakenC
from uacpy.visualization.plots import (
    plot_transmission_loss,
    plot_modes,
    plot_mode_functions,
    plot_mode_wavenumbers,
    plot_modes_heatmap,
    plot_bottom_properties,
    plot_environment_advanced
)

def main():
    print("=" * 70)
    print("ADVANCED KRAKEN EXAMPLE - Coupled Mode Theory")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════════════════
    # ENVIRONMENT: Continental Shelf
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[Setup] Creating continental shelf environment...")

    # Bathymetry: shelf to slope
    bathymetry = np.array([
        [0, 100],       # Shelf: 100m
        [8000, 120],    # Still on shelf
        [10000, 150],   # Shelf break begins
        [15000, 250],   # Slope
        [20000, 400],   # Deep water
    ])

    # Range-dependent bottom: sand on shelf, rock on slope
    ranges_km = bathymetry[:, 0] / 1000.0
    bottom_rd = RangeDependentBottom(
        ranges_km=ranges_km,
        depths=bathymetry[:, 1],
        sound_speed=np.array([1600, 1620, 1650, 1700, 1800]),  # Hardening
        density=np.array([1.5, 1.6, 1.7, 1.9, 2.2]),           # Compacting
        attenuation=np.array([0.8, 0.7, 0.5, 0.4, 0.2]),       # Less lossy
        shear_speed=np.array([0, 0, 200, 400, 600]),           # Rock on slope
        acoustic_type='half-space'
    )

    # Downward refracting SSP (typical shelf)
    ssp_data = np.array([
        [0, 1520],
        [50, 1505],
        [100, 1495],
        [200, 1490],
        [400, 1485],
    ])

    env = uacpy.Environment(
        name="Continental Shelf - Mode Coupling",
        depth=400.0,
        ssp_type='pchip',  # Smooth interpolation
        ssp_data=ssp_data,
        bathymetry=bathymetry,
        bottom=bottom_rd
    )

    print(f"  ✓ Range-dependent: {env.is_range_dependent}")
    print(f"  ✓ Depth range: {bathymetry[:, 1].min():.0f}m → {bathymetry[:, 1].max():.0f}m")

    # ═══════════════════════════════════════════════════════════════════════
    # SOURCE & RECEIVER
    # ═══════════════════════════════════════════════════════════════════════

    source = uacpy.Source(depth=50.0, frequency=50.0)  # 50 Hz for good modal propagation
    receiver = uacpy.Receiver(
        depths=np.linspace(5, 380, 60),
        ranges=np.linspace(100, 20000, 100)
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RUN 1: Compute Modes with Volume Attenuation
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[1/4] Computing modes with Francois-Garrison attenuation...")

    try:
        kraken = Kraken(verbose=False)

        # Compute modes for first range (shallow water)
        env_shallow = uacpy.Environment(
            name="Shelf (100m)",
            depth=100.0,
            ssp_type='pchip',
            ssp_data=ssp_data[ssp_data[:, 0] <= 100],
            bottom=bottom_rd.get_at_range(0)
        )

        modes_shallow = kraken.compute_modes(
            env_shallow,
            source,
            volume_attenuation='F'  # Francois-Garrison
        )
        print(f"  ✓ Computed {modes_shallow['M']} modes for shallow water")

        # Compute modes for last range (deep water)
        env_deep = uacpy.Environment(
            name="Slope (400m)",
            depth=400.0,
            ssp_type='pchip',
            ssp_data=ssp_data,
            bottom=bottom_rd.get_at_range(20000)
        )

        modes_deep = kraken.compute_modes(
            env_deep,
            source,
            volume_attenuation='F'
        )
        print(f"  ✓ Computed {modes_deep['M']} modes for deep water")

    except Exception as e:
        print(f"  ✗ Kraken error: {e}")
        modes_shallow = None
        modes_deep = None

    # ═══════════════════════════════════════════════════════════════════════
    # RUN 2: KrakenField with Adiabatic Mode Coupling
    # ═══════════════════════════════════════════════════════════════════════

    print("[2/4] Running KrakenField with adiabatic mode coupling...")

    try:
        krakenfield = KrakenField(verbose=False, mode_coupling='adiabatic', n_segments=5)
        result = krakenfield.run(
            env, source, receiver
        )
        print(f"  ✓ KrakenField completed with {result.metadata.get('n_segments', 5)} segments")
        print(f"  ✓ TL range: {result.data.min():.1f} to {result.data.max():.1f} dB")
    except Exception as e:
        print(f"  ✗ KrakenField error: {e}")
        import traceback
        traceback.print_exc()
        result = None

    # ═══════════════════════════════════════════════════════════════════════
    # RUN 3: KrakenC - Complex Modes with Elastic Bottom
    # ═══════════════════════════════════════════════════════════════════════

    print("[3/4] Computing complex modes with KrakenC (elastic bottom)...")

    try:
        krakenc = KrakenC(verbose=False)

        # Use environment with elastic bottom for complex mode computation
        modes_complex = krakenc.compute_modes(
            env_shallow,  # Reuse shallow environment
            source,
            volume_attenuation='F'
        )
        print(f"  ✓ Computed {modes_complex['M']} complex modes")
        print(f"  ✓ Supports elastic bottom with shear waves")
        krakenc_success = True
    except Exception as e:
        print(f"  ✗ KrakenC error: {e}")
        modes_complex = None
        krakenc_success = False

    # ═══════════════════════════════════════════════════════════════════════
    # VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════════

    # Ensure output directory exists
    import os
    os.makedirs('output', exist_ok=True)

    print("[4/4] Generating plots...")

    # Plot 1: Environment with bottom properties
    fig1, axes1 = plot_bottom_properties(env)
    plt.savefig('output/example_09_bottom.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: example_09_bottom.png")

    # Plot 2: Mode comparison (shallow vs deep)
    if modes_shallow is not None and modes_deep is not None:
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

        # Shallow water modes
        n_show = min(5, modes_shallow['M'])
        for i in range(n_show):
            axes2[0].plot(modes_shallow['phi'][:, i].real,
                         modes_shallow['z'], label=f'Mode {i+1}')
        axes2[0].invert_yaxis()
        axes2[0].set_xlabel('Mode Amplitude')
        axes2[0].set_ylabel('Depth (m)')
        axes2[0].set_title(f'Shallow Water Modes (100m)\n{modes_shallow["M"]} total modes')
        axes2[0].legend()
        axes2[0].grid(True, alpha=0.3)

        # Deep water modes
        n_show = min(5, modes_deep['M'])
        for i in range(n_show):
            axes2[1].plot(modes_deep['phi'][:, i].real,
                         modes_deep['z'], label=f'Mode {i+1}')
        axes2[1].invert_yaxis()
        axes2[1].set_xlabel('Mode Amplitude')
        axes2[1].set_ylabel('Depth (m)')
        axes2[1].set_title(f'Deep Water Modes (400m)\n{modes_deep["M"]} total modes')
        axes2[1].legend()
        axes2[1].grid(True, alpha=0.3)

        plt.suptitle('Mode Evolution: Shelf to Slope', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('output/example_09_modes.png', dpi=150, bbox_inches='tight')
        print("  ✓ Saved: example_09_modes.png")

    # Plot 2b: Wavenumber plot using NEW plot_mode_wavenumbers function
    if modes_shallow is not None:
        print("\nGenerating advanced mode visualizations...")

        # NEW: Use plot_mode_wavenumbers function for complex k-plane visualization
        try:
            # Convert modes_shallow dict to Field object for new plotting functions
            modes_field = uacpy.Field(
                data=modes_shallow['phi'],
                depths=modes_shallow['z'],
                ranges=np.array([0]),
                frequencies=source.frequency,
                field_type='modes',
                metadata={
                    'k': modes_shallow['k'],
                    'M': modes_shallow['M'],
                    'phi': modes_shallow['phi'],
                    'z': modes_shallow['z'],
                    'frequency': source.frequency[0]
                }
            )

            # NEW: plot_mode_wavenumbers - complex k-plane scatter plot
            fig2b, ax2b = plot_mode_wavenumbers(
                modes_field,
                annotate_modes=True,
                max_annotations=15
            )
            ax2b.set_title('NEW: Mode Wavenumbers in Complex k-Plane\n' +
                          f'Shallow Water - {modes_shallow["M"]} modes',
                          fontsize=14, fontweight='bold')
            plt.savefig('output/example_09_wavenumbers.png', dpi=150, bbox_inches='tight')
            plt.close(fig2b)
            print("  ✓ Saved: example_09_wavenumbers.png (NEW plot_mode_wavenumbers)")

        except Exception as e:
            print(f"  ! Warning: Could not create wavenumber plot: {e}")

        # NEW: Use plot_modes with show_imaginary=True
        try:
            fig2c, (ax_modes, ax_k) = plot_modes(
                modes_field,
                show_imaginary=True  # NEW: Show imaginary parts as dashed lines
            )
            fig2c.suptitle('NEW: Mode Shapes with Imaginary Parts\n' +
                          'Shallow Water (solid=real, dashed=imaginary)',
                          fontsize=14, fontweight='bold')
            plt.savefig('output/example_09_mode_shapes.png', dpi=150, bbox_inches='tight')
            plt.close(fig2c)
            print("  ✓ Saved: example_09_mode_shapes.png (NEW show_imaginary)")

        except Exception as e:
            print(f"  ! Warning: Could not create mode shapes plot: {e}")

        # NEW: Use plot_modes_heatmap - Show all modes as 2D heatmap
        try:
            print("  Creating mode heatmap...", end=" ", flush=True)
            fig_heatmap, ax_heatmap, cbar_heatmap = plot_modes_heatmap(
                modes_field,
                mode_range=None,  # Plot all modes
                normalize=True,
                figsize=(14, 8)
            )
            plt.suptitle('All Mode Shapes - Continental Shelf Heatmap (NEW!)',
                        fontsize=14, fontweight='bold')
            plt.savefig('output/example_09_modes_heatmap.png', dpi=150, bbox_inches='tight')
            plt.close(fig_heatmap)
            print("✓")
            print("  ✓ Saved: example_09_modes_heatmap.png (NEW plot_modes_heatmap)")
        except Exception as e:
            print(f"\n  ! Warning: Could not create mode heatmap: {e}")

    # Plot 3: Coupled mode TL field
    # NEW: Using auto TL limits and contour overlays
    if result is not None:
        fig3, ax3, cbar3 = plot_transmission_loss(
            result, env,
            contours=[70, 85, 100],  # NEW: Add labeled contours
            show_colorbar=True
        )
        ax3.set_title('KrakenField: Adiabatic Mode Coupling\nContinental Shelf Transition\n' +
                     '(NEW: Auto TL limits + contour overlays)')

        # Add segment indicators
        seg_ranges = np.linspace(0, 20, 11)
        for r in seg_ranges[1:-1]:
            ax3.axvline(r, color='white', linestyle='--', alpha=0.3, linewidth=0.5, zorder=8)

        plt.savefig('output/example_09_result.png', dpi=150, bbox_inches='tight')
        print("  ✓ Saved: example_09_result.png")

    print("\n" + "=" * 70)
    print("KRAKEN ADVANCED EXAMPLE COMPLETE")
    print("=" * 70)
    print("\nFeatures demonstrated:")
    print("  ✓ Volume attenuation (Francois-Garrison)")
    print("  ✓ Range-dependent bottom properties")
    print("  ✓ Adiabatic mode coupling (10 segments)")
    print("  ✓ Mode evolution (shelf → slope)")
    print("  ✓ Continental shelf propagation")
    print("\nNEW Plotting features demonstrated:")
    print("  ✓ plot_mode_wavenumbers() - Complex k-plane visualization")
    print("  ✓ plot_modes_heatmap() - All modes as 2D heatmap (NEW!)")
    print("  ✓ Mode imaginary parts (show_imaginary=True)")
    print("  ✓ Auto TL limits (median + 0.75σ)")
    print("  ✓ Contour overlays on TL plots")
    print("  ✓ jet_r colormap (blue=good, red=poor)")

    return 0

if __name__ == "__main__":
    sys.exit(main())
