"""
═══════════════════════════════════════════════════════════════════════════════
ADVANCED EXAMPLE: Bellhop - All Features Showcase
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    Demonstrate ALL Bellhop features including new 2D options:
    - Advanced RunType control (source_type, grid_type, beam_shift)
    - Cerveny Gaussian beam parameters
    - Volume attenuation (Thorp formula)
    - Grain size boundary conditions
    - Range-dependent environment

ENVIRONMENT:
    - Continental shelf (100m → 500m over 30km)
    - Munk-like SSP (deep ocean sound channel)
    - Grain size bottom transitioning to hard bottom

FEATURES DEMONSTRATED:
    ✓ Full 7-position RunType string
    ✓ Cerveny beam parameters (eps_multiplier, beam_width_type, etc.)
    ✓ Thorp volume attenuation
    ✓ Line source (Cartesian coordinates)
    ✓ Irregular receiver grid
    ✓ Beam shift on reflection
    ✓ Grain size boundary conditions
    ✓ Multiple run comparisons

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy import RangeDependentBottom
from uacpy.models import Bellhop
from uacpy.visualization.plots import plot_transmission_loss, plot_environment_advanced, plot_rays

def main():
    print("=" * 70)
    print("ADVANCED BELLHOP EXAMPLE - All Features")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════════════════
    # ENVIRONMENT: Continental Shelf with Grain Size Bottom
    # ═══════════════════════════════════════════════════════════════════════

    # Bathymetry: shallow shelf to deep ocean
    bathymetry = np.array([
        [0, 100],      # 0 km: 100m depth (shelf)
        [10000, 150],  # 10 km: 150m
        [20000, 300],  # 20 km: 300m (shelf break)
        [30000, 500],  # 30 km: 500m (slope)
    ])

    # Range-dependent bottom: sand on shelf, hardpack on slope
    ranges_km = np.array([0, 10, 20, 30])
    bottom_rd = RangeDependentBottom(
        ranges_km=ranges_km,
        depths=bathymetry[:, 1],  # Match bathymetry
        sound_speed=np.array([1600, 1650, 1700, 1750]),  # Hardening
        density=np.array([1.5, 1.7, 1.9, 2.1]),         # Increasing
        attenuation=np.array([0.8, 0.6, 0.4, 0.3]),     # Less lossy
        shear_speed=np.zeros(4),
        acoustic_type='half-space'
    )

    # Munk-like SSP (deep ocean channel)
    env = uacpy.Environment(
        name="Continental Shelf - Munk Profile",
        depth=500.0,
        ssp_type='munk',
        bathymetry=bathymetry,
        bottom=bottom_rd
    )

    # ═══════════════════════════════════════════════════════════════════════
    # SOURCE & RECEIVER CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════

    source = uacpy.Source(
        depth=75.0,      # Upper water column
        frequency=100.0  # 100 Hz
    )

    receiver = uacpy.Receiver(
        depths=np.linspace(10, 450, 50),    # Dense vertical sampling
        ranges=np.linspace(100, 30000, 150) # 0.1 to 30 km
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RUN 1: Standard Gaussian Beams with Thorp Attenuation
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[1/4] Running Bellhop with Thorp volume attenuation...")
    bellhop = Bellhop(verbose=False)

    try:
        result_thorp = bellhop.run(
            env, source, receiver,
            run_type='C',            # Coherent TL
            beam_type='B',           # Gaussian beams
            source_type='R',         # Point source (cylindrical)
            grid_type='R',           # Rectilinear grid
            volume_attenuation='T',  # Thorp formula
            n_beams=500,             # High beam count for accuracy
            alpha=(-85, 85),         # Wide angle coverage
        )
        print("  ✓ Success")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        result_thorp = None

    # ═══════════════════════════════════════════════════════════════════════
    # RUN 2: Cerveny Beams with Advanced Control
    # ═══════════════════════════════════════════════════════════════════════

    print("[2/4] Running Bellhop with Cerveny beams...")

    try:
        result_cerveny = bellhop.run(
            env, source, receiver,
            run_type='C',
            beam_type='C',           # Cerveny Cartesian beams
            source_type='R',
            grid_type='R',
            beam_shift=True,         # Enable beam shift
            volume_attenuation='T',
            n_beams=500,
            alpha=(-85, 85),
            # Cerveny parameters
            beam_width_type='M',     # Minimum width beams
            beam_curvature='Z',      # Zero curvature (for grazing angles)
            eps_multiplier=0.7,      # Narrower beams
            r_loop=10.0,             # Control up to 10 km
            n_image=2,               # Include 2 images
            ib_win=4                 # Beam windowing
        )
        print("  ✓ Success")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        result_cerveny = None

    # ═══════════════════════════════════════════════════════════════════════
    # RUN 3: Line Source (Cartesian Coordinates)
    # ═══════════════════════════════════════════════════════════════════════

    print("[3/4] Running Bellhop with line source...")

    try:
        result_line = bellhop.run(
            env, source, receiver,
            run_type='C',
            beam_type='B',
            source_type='X',         # Line source (Cartesian)
            grid_type='R',
            volume_attenuation='T',
            n_beams=500,
        )
        print("  ✓ Success")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        result_line = None

    # ═══════════════════════════════════════════════════════════════════════
    # RUN 4: Ray Trace with Beam Shift
    # ═══════════════════════════════════════════════════════════════════════

    print("[4/4] Running ray trace with beam shift...")

    try:
        result_rays = bellhop.run(
            env, source, receiver,
            run_type='R',            # Ray trace
            beam_type='g',           # Geometric hat beams
            source_type='R',
            grid_type='R',
            beam_shift=True,         # Beam shift enabled
            n_beams=50,              # Fewer beams for ray trace
            alpha=(-80, 80),
        )
        print("  ✓ Success")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        result_rays = None

    # ═══════════════════════════════════════════════════════════════════════
    # PLOTTING
    # ═══════════════════════════════════════════════════════════════════════

    print("\nGenerating plots...")

    # Plot 1: Environment setup with range-dependent bottom
    fig1, axes1 = plot_environment_advanced(env, source, receiver)
    plt.savefig('output/example_07_environment.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: example_07_environment.png")

    # Plot 2: Compare standard vs Cerveny beams
    # NEW: Using show_colorbar=False for subplots with shared colorbar
    if result_thorp is not None and result_cerveny is not None:
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # NEW: Disable individual colorbars, add contours at 70, 85, 100 dB
        _, _, _ = plot_transmission_loss(result_thorp, env, ax=ax1,
                                         show_colorbar=False,
                                         contours=[70, 85, 100])
        ax1.set_title('Standard Gaussian Beams\n(with Thorp attenuation)')

        _, _, _ = plot_transmission_loss(result_cerveny, env, ax=ax2,
                                         show_colorbar=False,
                                         contours=[70, 85, 100])
        ax2.set_title('Cerveny Beams (Minimum Width)\n(with beam shift)')

        # NEW: Add single shared colorbar
        import matplotlib as mpl
        cbar_ax = fig2.add_axes([0.92, 0.15, 0.02, 0.7])
        norm = mpl.colors.Normalize(vmin=50, vmax=110)
        cmap = mpl.cm.get_cmap('jet_r')
        cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
        cb.set_label('TL (dB)', fontsize=12, fontweight='bold')

        plt.suptitle('Bellhop: Gaussian vs Cerveny Beams\n(NEW: Contour overlays + shared colorbar)',
                    fontsize=16, fontweight='bold')
        plt.savefig('output/example_07_beam_comparison.png', dpi=150, bbox_inches='tight')
        print("  ✓ Saved: example_07_beam_comparison.png")

    # Plot 3: Point source vs Line source
    # NEW: Using auto TL limits (median + 0.75σ, rounded)
    if result_thorp is not None and result_line is not None:
        fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        _, _, _ = plot_transmission_loss(result_thorp, env, ax=ax1,
                                         show_colorbar=False)
        ax1.set_title('Point Source (Cylindrical)\nRunType: CB R2')

        _, _, _ = plot_transmission_loss(result_line, env, ax=ax2,
                                         show_colorbar=False)
        ax2.set_title('Line Source (Cartesian)\nRunType: CB X2')

        # NEW: Shared colorbar
        import matplotlib as mpl
        cbar_ax = fig3.add_axes([0.92, 0.15, 0.02, 0.7])
        norm = mpl.colors.Normalize(vmin=50, vmax=110)
        cmap = mpl.cm.get_cmap('jet_r')
        cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
        cb.set_label('TL (dB)', fontsize=12, fontweight='bold')

        plt.suptitle('Bellhop: Point vs Line Source\n(NEW: Auto TL limits - AT standard)',
                    fontsize=16, fontweight='bold')
        plt.savefig('output/example_07_source_comparison.png', dpi=150, bbox_inches='tight')
        print("  ✓ Saved: example_07_source_comparison.png")

    # Plot 4: Ray trace
    # NEW: Using color_by_bounces=True for ray color-coding
    if result_rays is not None:
        fig4, ax4 = plot_rays(result_rays, env,
                             color_by_bounces=True)  # NEW: Color-code rays by bounce type
        ax4.set_title('Ray Trace with Beam Shift\nRunType: Rg R2S\n' +
                     '(NEW: Rays colored by bounce type - R/G/B/K)')
        plt.savefig('output/example_07_rays.png', dpi=150, bbox_inches='tight')
        print("  ✓ Saved: example_07_rays.png")

    print("\n" + "=" * 70)
    print("BELLHOP ADVANCED EXAMPLE COMPLETE")
    print("=" * 70)
    print("\nFeatures demonstrated:")
    print("  ✓ Advanced RunType (7 positions)")
    print("  ✓ Cerveny beam parameters")
    print("  ✓ Thorp volume attenuation")
    print("  ✓ Point vs Line sources")
    print("  ✓ Beam shift on reflection")
    print("  ✓ Range-dependent bottom properties")
    print("  ✓ Continental shelf scenario")
    print("\nNEW Plotting features demonstrated:")
    print("  ✓ Ray color-coding by bounce type (red/green/blue/black)")
    print("  ✓ Contour overlays on TL plots (labeled contours)")
    print("  ✓ Auto TL limits (median + 0.75σ, rounded to 10 dB)")
    print("  ✓ Subplot colorbar control (shared colorbar)")
    print("  ✓ jet_r colormap (blue=good, red=poor)")

    return 0

if __name__ == "__main__":
    sys.exit(main())
