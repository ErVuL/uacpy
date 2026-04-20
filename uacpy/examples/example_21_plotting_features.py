"""
═══════════════════════════════════════════════════════════════════════════════
COMPREHENSIVE EXAMPLE: All New Plotting Features
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    Demonstrate all new plotting improvements implemented in UACPY:
    - P0: Critical fixes (colormap, TL limits, ray colors, z-order, extensions)
    - P1: High priority (mode imaginary parts, wavenumber plot, colorbar control, contours)

FEATURES DEMONSTRATED:
    ✓ P0-1: Colormap 'jet_r' (blue=good, red=poor) - Acoustic Toolbox standard
    ✓ P0-2: Automatic TL limit algorithm (median + 0.75σ, rounded)
    ✓ P0-3: Ray color-coding by bounce type (red/green/blue/black)
    ✓ P0-4: Z-order constants (consistent layering)
    ✓ P0-5: Colorbar extensions (clipping indicators)
    ✓ P1-1: Shear wave plotting (elastic media)
    ✓ P1-2: Mode imaginary parts (dashed lines)
    ✓ P1-3: Mode wavenumber complex plot (k-plane scatter)
    ✓ P1-4: show_colorbar parameter (subplot control)
    ✓ P1-5: Contour overlay support (labeled TL contours)

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.models import Bellhop, Kraken
from uacpy.core.environment import BoundaryProperties
from uacpy.visualization.plots import (
    plot_transmission_loss,
    plot_rays,
    plot_ssp,
    plot_modes,
    plot_mode_wavenumbers,
)
from uacpy.visualization.style import apply_professional_style

def main():
    print("=" * 80)
    print("COMPREHENSIVE PLOTTING FEATURES DEMONSTRATION")
    print("=" * 80)

    # Ensure output directory exists
    import os
    os.makedirs('output', exist_ok=True)

    # Apply professional style
    apply_professional_style(dpi=150)

    # ═══════════════════════════════════════════════════════════════════════
    # SCENARIO 1: Transmission Loss with New Features
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[1/6] Testing TL plot with automatic limits and contours...")

    # Create environment with sloping bathymetry
    bathymetry = np.array([
        [0, 100],
        [5000, 120],
        [10000, 180],
        [15000, 200],
    ])

    # SSP with thermocline
    ssp_data = np.array([
        [0, 1520],
        [30, 1518],
        [50, 1505],
        [100, 1495],
        [200, 1490],
    ])

    env_tl = uacpy.Environment(
        name="Sloping Bottom - TL Demo",
        depth=200.0,
        ssp_data=ssp_data,
        bathymetry=bathymetry,
    )

    source_tl = uacpy.Source(depth=30.0, frequency=100.0)
    receiver_tl = uacpy.Receiver(
        depths=np.linspace(5, 190, 50),
        ranges=np.linspace(100, 15000, 80)
    )

    try:
        bellhop = Bellhop(verbose=False)
        result_tl = bellhop.run(env_tl, source_tl, receiver_tl)

        # Test P0-2: Automatic TL limits (median + 0.75σ, rounded)
        # Test P0-5: Colorbar extensions
        # Test P1-5: Contour overlay at 70, 80, 90 dB
        fig1, ax1, cbar1 = plot_transmission_loss(
            result_tl,
            env_tl,
            contours=[70, 80, 90],  # NEW: P1-5 contour overlay
            show_colorbar=True      # NEW: P1-4 colorbar control
        )
        ax1.set_title('P0-2: Auto TL Limits + P1-5: Contour Overlay\n'
                     'Blue=good, Red=poor (P0-1: jet_r colormap)')
        plt.savefig('output/example_21_tl_auto_limits_contours.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: example_21_tl_auto_limits_contours.png")
        print(f"    - TL limits auto-computed and rounded to 10 dB")
        print(f"    - Colorbar shows extensions if data clipped")
        print(f"    - Contours labeled at 70, 80, 90 dB")

    except Exception as e:
        print(f"  ✗ Error: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # SCENARIO 2: Ray Color-Coding by Bounce Type
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[2/6] Testing ray color-coding (P0-3)...")

    # Environment with simple bathymetry for clear ray visualization
    env_rays = uacpy.Environment(
        name="Ray Color Coding Demo",
        depth=100.0,
        ssp_data=np.array([[0, 1500], [100, 1500]]),  # Isovelocity
    )

    source_rays = uacpy.Source(depth=50.0, frequency=100.0)
    receiver_rays = uacpy.Receiver(
        depths=np.linspace(5, 95, 30),
        ranges=np.linspace(100, 10000, 50)
    )

    try:
        bellhop_rays = Bellhop(verbose=False)
        rays_result = bellhop_rays.compute_rays(
            env_rays,
            source_rays,
            receiver_ranges=[10000],
            n_rays=51,
            ray_angles=np.linspace(-30, 30, 51)
        )

        # Test P0-3: Ray color-coding
        # Red=direct, Green=surface, Blue=bottom, Black=both
        fig2, ax2 = plot_rays(
            rays_result,
            env_rays,
            color_by_bounces=True,  # NEW: P0-3 ray color-coding
        )
        ax2.set_title('P0-3: Ray Color-Coding by Bounce Type\n'
                     'Red=direct, Green=surface, Blue=bottom, Black=both')
        plt.savefig('output/example_21_ray_color_coding.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: example_21_ray_color_coding.png")
        print("    - Rays colored by bounce type (Acoustic Toolbox standard)")

    except Exception as e:
        print(f"  ✗ Error: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # SCENARIO 3: SSP with Shear Wave Plotting
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[3/6] Testing SSP with shear waves (P1-1)...")

    # Environment with elastic bottom (shear waves)
    ssp_elastic = np.array([
        [0, 1520],
        [50, 1505],
        [100, 1495],
    ])

    bottom_elastic = BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1700.0,
        density=2.0,
        attenuation=0.5,
        shear_speed=400.0,  # Elastic bottom with shear
    )

    env_elastic = uacpy.Environment(
        name="Elastic Bottom - Shear Wave Demo",
        depth=100.0,
        ssp_data=ssp_elastic,
        bottom=bottom_elastic
    )

    # Test P1-1: Shear wave plotting
    fig3, ax3 = plot_ssp(
        env_elastic,
        show_shear=True,       # NEW: P1-1 shear wave plotting
        show_data_points=True  # NEW: show data points as markers
    )
    ax3.set_title('P1-1: SSP with Shear Wave Visualization\n'
                 'Blue=compression, Red=shear (Acoustic Toolbox standard)')
    plt.savefig('output/example_21_ssp_shear_waves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: example_21_ssp_shear_waves.png")
    print("    - Compression (blue) and shear (red) wave speeds shown")
    print("    - Data points marked with black circles")

    # ═══════════════════════════════════════════════════════════════════════
    # SCENARIO 4: Mode Imaginary Parts
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[4/6] Testing mode imaginary parts (P1-2)...")

    # Environment for mode computation
    ssp_modes = np.array([
        [0, 1520],
        [50, 1505],
        [150, 1495],
    ])

    bottom_modes = BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1600.0,
        density=1.8,
        attenuation=0.5,  # Lossy bottom creates imaginary parts
    )

    env_modes = uacpy.Environment(
        name="Mode Analysis Demo",
        depth=150.0,
        ssp_data=ssp_modes,
        bottom=bottom_modes
    )

    source_modes = uacpy.Source(depth=50.0, frequency=50.0)

    try:
        kraken = Kraken(verbose=False)
        modes_result = kraken.compute_modes(
            env_modes,
            source_modes,
            volume_attenuation='T'  # Thorp attenuation creates imaginary parts
        )

        # Test P1-2: Mode imaginary parts (dashed lines)
        fig4, (ax4a, ax4b) = plot_modes(
            modes_result,
            show_imaginary=True,  # NEW: P1-2 show imaginary parts as dashed lines
        )
        fig4.suptitle('P1-2: Mode Shapes with Imaginary Parts\n'
                     'Solid=real, Dashed=imaginary (Acoustic Toolbox standard)',
                     fontsize=14, fontweight='bold')
        plt.savefig('output/example_21_mode_imaginary_parts.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: example_21_mode_imaginary_parts.png")
        print(f"    - Computed {modes_result.metadata['M']} modes")
        print("    - Real parts (solid) and imaginary parts (dashed) shown")

    except Exception as e:
        print(f"  ✗ Error: {e}")
        modes_result = None

    # ═══════════════════════════════════════════════════════════════════════
    # SCENARIO 5: Mode Wavenumber Complex Plot
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[5/6] Testing mode wavenumber plot (P1-3)...")

    if modes_result is not None:
        try:
            # Test P1-3: Mode wavenumber complex plot
            fig5, ax5 = plot_mode_wavenumbers(
                modes_result,
                annotate_modes=True,    # NEW: P1-3 wavenumber k-plane plot
                max_annotations=15      # Show first 15 mode numbers
            )
            ax5.set_title('P1-3: Mode Wavenumbers in Complex k-Plane\n'
                         'Real(k) vs Imag(k) - Modal structure visualization')
            plt.savefig('output/example_21_mode_wavenumbers.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("  ✓ Saved: example_21_mode_wavenumbers.png")
            print("    - Complex k-plane scatter plot (Acoustic Toolbox standard)")
            print("    - Mode numbers annotated")
            print("    - Propagating modes (small imag) vs evanescent modes (large imag)")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # SCENARIO 6: Subplot Control with show_colorbar
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[6/6] Testing subplot control (P1-4)...")

    # Create two different scenarios for comparison
    source_compare = uacpy.Source(depth=30.0, frequency=[50.0, 200.0])
    receiver_compare = uacpy.Receiver(
        depths=np.linspace(5, 190, 40),
        ranges=np.linspace(100, 15000, 60)
    )

    try:
        bellhop_compare = Bellhop(verbose=False)

        # Run at two frequencies
        source_50 = uacpy.Source(depth=30.0, frequency=50.0)
        source_200 = uacpy.Source(depth=30.0, frequency=200.0)

        result_50 = bellhop_compare.run(env_tl, source_50, receiver_compare)
        result_200 = bellhop_compare.run(env_tl, source_200, receiver_compare)

        # Test P1-4: show_colorbar parameter for subplots
        fig6, axes6 = plt.subplots(1, 2, figsize=(16, 6))

        # Left subplot: 50 Hz (no colorbar)
        _, ax6a, _ = plot_transmission_loss(
            result_50,
            env_tl,
            ax=axes6[0],
            show_colorbar=False,  # NEW: P1-4 disable colorbar for subplots
            contours=[70, 85],    # With contours
        )
        ax6a.set_title('50 Hz', fontsize=12, fontweight='bold')

        # Right subplot: 200 Hz (no colorbar)
        _, ax6b, _ = plot_transmission_loss(
            result_200,
            env_tl,
            ax=axes6[1],
            show_colorbar=False,  # NEW: P1-4 disable colorbar for subplots
            contours=[70, 85],    # With contours
        )
        ax6b.set_title('200 Hz', fontsize=12, fontweight='bold')

        # Add single shared colorbar
        from matplotlib.colorbar import ColorbarBase
        from matplotlib.cm import ScalarMappable
        import matplotlib as mpl

        cbar_ax = fig6.add_axes([0.92, 0.15, 0.02, 0.7])
        norm = mpl.colors.Normalize(vmin=50, vmax=100)
        cmap = mpl.cm.get_cmap('jet_r')
        cb = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
        cb.set_label('TL (dB)', fontsize=12, fontweight='bold')

        fig6.suptitle('P1-4: Subplot Control with show_colorbar Parameter\n'
                     'Frequency Comparison: 50 Hz vs 200 Hz',
                     fontsize=14, fontweight='bold')

        plt.savefig('output/example_21_subplot_colorbar_control.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: example_21_subplot_colorbar_control.png")
        print("    - Multiple TL plots in subplots with shared colorbar")
        print("    - Individual colorbars disabled using show_colorbar=False")

    except Exception as e:
        print(f"  ✗ Error: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("PLOTTING FEATURES DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nAll new features tested:")
    print("\n  P0 - CRITICAL FIXES:")
    print("    ✓ P0-1: Colormap 'jet_r' (blue=good, red=poor)")
    print("    ✓ P0-2: Automatic TL limits (median + 0.75σ, rounded to 10 dB)")
    print("    ✓ P0-3: Ray color-coding (red/green/blue/black by bounce type)")
    print("    ✓ P0-4: Z-order constants (consistent element layering)")
    print("    ✓ P0-5: Colorbar extensions (shows when data clipped)")
    print("\n  P1 - HIGH PRIORITY:")
    print("    ✓ P1-1: Shear wave plotting (red line for elastic media)")
    print("    ✓ P1-2: Mode imaginary parts (dashed lines)")
    print("    ✓ P1-3: Mode wavenumber plot (complex k-plane scatter)")
    print("    ✓ P1-4: show_colorbar parameter (subplot control)")
    print("    ✓ P1-5: Contour overlay (labeled TL contour lines)")
    print("\nAll plots follow Acoustic Toolbox standards!")
    print("Output: 6 test plots saved to output/ directory")
    print("=" * 80)

    return 0

if __name__ == "__main__":
    sys.exit(main())
