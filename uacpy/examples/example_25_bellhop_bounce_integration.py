"""
===============================================================================
EXAMPLE 25: Bellhop + BOUNCE Integration, Layered Bottom, Range-Dependent Bottom
===============================================================================

OBJECTIVE:
    Demonstrate three key features added for range/depth-dependent modeling:
    1. Bellhop.run_with_bounce() — one-call elastic bottom workflow
    2. LayeredBottom — multi-layer sediment with Kraken
    3. Range-dependent bottom properties — with RAM and visualization

FEATURES DEMONSTRATED:
    - Bellhop.run_with_bounce() convenience method
    - LayeredBottom + SedimentLayer for depth-dependent sediment
    - RangeDependentBottom with RAM (true Fortran RD support)
    - plot_transmission_loss(), plot_layered_bottom(), plot_bottom_properties()
    - plot_ssp_2d(), plot_environment_advanced()

===============================================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import uacpy
from uacpy import (
    RangeDependentBottom, SedimentLayer, LayeredBottom,
    RangeDependentLayeredBottom, BoundaryProperties,
)
from uacpy.models import Bellhop, RAM
from uacpy.visualization.plots import (
    plot_transmission_loss,
    plot_layered_bottom,
    plot_rd_layered_bottom,
    plot_bottom_properties,
    plot_ssp_2d,
    plot_environment_advanced,
)

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)


def demo_bellhop_bounce():
    """Part 1: Bellhop + BOUNCE for elastic bottom modeling."""
    print("\n" + "=" * 70)
    print("PART 1: Bellhop + BOUNCE Integration")
    print("=" * 70)

    # Sandy bottom with significant shear properties
    bottom = BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1700.0,
        shear_speed=400.0,
        density=1.9,
        attenuation=0.5,
        shear_attenuation=1.0,
    )

    env = uacpy.Environment(
        name='Elastic Sandy Bottom',
        depth=100,
        ssp_type='isovelocity',
        sound_speed=1500.0,
        bottom=bottom,
    )

    source = uacpy.Source(frequency=500.0, depth=25.0)
    receiver = uacpy.Receiver(
        depths=np.linspace(1, 99, 50),
        ranges=np.linspace(100, 5000, 100),
    )

    bellhop = Bellhop(verbose=True)

    # Method 1: Standard half-space (ignores shear)
    print("\n--- Standard Bellhop (half-space, no shear) ---")
    result_hs = bellhop.run(env, source, receiver, run_type='C')

    # Method 2: With BOUNCE (accounts for shear)
    print("\n--- Bellhop with BOUNCE reflection coefficients ---")
    result_bounce = bellhop.run_with_bounce(
        env, source, receiver,
        run_type='C',
        cmin=1400.0, cmax=10000.0, rmax_km=10.0,
    )

    # Compare
    tl_hs = result_hs.data
    tl_bn = result_bounce.data
    diff = tl_bn - tl_hs
    print(f"\nHalf-space TL: {np.nanmin(tl_hs):.1f} to {np.nanmax(tl_hs):.1f} dB")
    print(f"BOUNCE TL:     {np.nanmin(tl_bn):.1f} to {np.nanmax(tl_bn):.1f} dB")
    print(f"Max |diff|:    {np.nanmax(np.abs(diff)):.1f} dB")

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ranges_km = receiver.ranges / 1000.0
    depths = receiver.depths
    vmin, vmax = 40, 90

    for ax, data, title in [
        (axes[0], tl_hs, 'Half-space (no shear)'),
        (axes[1], tl_bn, 'BOUNCE (with shear)'),
    ]:
        im = ax.pcolormesh(ranges_km, depths, data,
                           cmap='jet_r', vmin=vmin, vmax=vmax, shading='auto')
        ax.set_title(title)
        ax.set_ylabel('Depth (m)')
        ax.invert_yaxis()
        plt.colorbar(im, ax=ax, label='TL (dB)')

    im2 = axes[2].pcolormesh(ranges_km, depths, diff,
                              cmap='RdBu_r', vmin=-10, vmax=10, shading='auto')
    axes[2].set_title('Difference (BOUNCE - half-space)')
    axes[2].invert_yaxis()
    plt.colorbar(im2, ax=axes[2], label='dTL (dB)')

    for ax in axes:
        ax.set_xlabel('Range (km)')

    plt.suptitle(f'Bellhop + BOUNCE: f={source.frequency[0]:.0f} Hz, '
                 f'elastic sandy bottom (cp={bottom.sound_speed}, cs={bottom.shear_speed} m/s)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'example_25_bounce_comparison.png', dpi=150)
    print(f"  Saved: example_25_bounce_comparison.png")

    # Also plot TL with contours using the plot function
    fig2, ax2, _ = plot_transmission_loss(result_bounce, env, contours=[60, 70, 80])
    ax2.set_title('Bellhop + BOUNCE TL with Contours')
    plt.savefig(OUTPUT_DIR / 'example_25_bounce_tl.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: example_25_bounce_tl.png")

    plt.close('all')


def demo_layered_bottom():
    """Part 2: Layered bottom with Kraken."""
    print("\n" + "=" * 70)
    print("PART 2: Layered Bottom (Multi-Layer Sediment) with Kraken")
    print("=" * 70)

    # Define sediment layers
    layers = [
        SedimentLayer(thickness=5.0, sound_speed=1550.0, density=1.5,
                      attenuation=0.3, shear_speed=100.0),
        SedimentLayer(thickness=10.0, sound_speed=1650.0, density=1.7,
                      attenuation=0.5),
        SedimentLayer(thickness=20.0, sound_speed=1800.0, density=2.0,
                      attenuation=0.8, shear_speed=200.0, shear_attenuation=0.5),
    ]

    halfspace = BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=2000.0, density=2.2, attenuation=0.1,
    )

    layered = LayeredBottom(layers=layers, halfspace=halfspace)
    print(f"Total sediment thickness: {layered.total_thickness():.1f} m")

    env = uacpy.Environment(
        name='Continental Shelf - Layered Sediment',
        depth=200.0,
        ssp_type='isovelocity',
        sound_speed=1500.0,
        bottom=layered,
    )
    print(f"has_layered_bottom: {env.has_layered_bottom()}")

    source = uacpy.Source(frequency=100.0, depth=50.0)
    receiver = uacpy.Receiver(
        depths=np.linspace(1, 199, 50),
        ranges=np.linspace(100, 10000, 100),
    )

    # Run Scooter (supports layered via NMEDIA > 1, gives TL directly)
    print("\n--- Running Scooter with layered bottom ---")
    try:
        from uacpy.models import Scooter
        scooter = Scooter(verbose=True)
        result = scooter.compute_tl(env, source, receiver)
        print(f"Scooter TL: {np.nanmin(result.data):.1f} to {np.nanmax(result.data):.1f} dB")

        # Plot TL
        fig1, ax1, _ = plot_transmission_loss(result, env, contours=[70, 80, 90])
        ax1.set_title('Scooter TL — Layered Sediment (3 layers + halfspace)')
        plt.savefig(OUTPUT_DIR / 'example_25_layered_tl.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: example_25_layered_tl.png")
    except Exception as e:
        print(f"  Scooter error: {e}")
        import traceback
        traceback.print_exc()

    # Plot layered bottom structure
    fig2, ax2 = plot_layered_bottom(env)
    plt.savefig(OUTPUT_DIR / 'example_25_layered_structure.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: example_25_layered_structure.png")

    # Plot environment overview
    fig3, axes3 = plot_environment_advanced(env, source, receiver)
    plt.savefig(OUTPUT_DIR / 'example_25_layered_env.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: example_25_layered_env.png")

    plt.close('all')


def demo_range_dependent_bottom():
    """Part 3: Range-dependent bottom with RAM."""
    print("\n" + "=" * 70)
    print("PART 3: Range-Dependent Bottom Properties with RAM")
    print("=" * 70)

    # Create RD bottom: mud transitioning to sand over 15 km
    bottom_rd = RangeDependentBottom(
        ranges_km=np.array([0, 3, 6, 9, 12, 15]),
        depths=np.array([150, 160, 180, 200, 220, 250]),
        sound_speed=np.array([1500, 1550, 1600, 1650, 1700, 1750]),
        density=np.array([1.2, 1.4, 1.5, 1.7, 1.8, 2.0]),
        attenuation=np.array([1.0, 0.8, 0.6, 0.5, 0.4, 0.3]),
        shear_speed=np.zeros(6),
        acoustic_type='half-space',
    )

    # 2D SSP (warm to cold transition)
    depths_ssp = np.array([0, 25, 50, 100, 150, 200, 250])
    ranges_ssp = np.array([0, 5, 10, 15])
    ssp_2d = np.zeros((len(depths_ssp), len(ranges_ssp)))
    for i, r in enumerate(ranges_ssp):
        t_surf = 18 - r * 0.4
        t_bot = 8 - r * 0.1
        temp = t_bot + (t_surf - t_bot) * np.exp(-depths_ssp / 60)
        ssp_2d[:, i] = 1449 + 4.6 * temp - 0.055 * temp**2 + 0.016 * depths_ssp

    ssp_1d = np.column_stack([depths_ssp, ssp_2d[:, 0]])

    env = uacpy.Environment(
        name='Shelf Break: Mud to Sand',
        depth=250.0,
        ssp_type='pchip',
        ssp_data=ssp_1d,
        ssp_2d_ranges=ranges_ssp,
        ssp_2d_matrix=ssp_2d,
        bathymetry=np.column_stack([bottom_rd.ranges_km * 1000, bottom_rd.depths]),
        bottom=bottom_rd,
    )

    source = uacpy.Source(frequency=100.0, depth=30.0)
    receiver = uacpy.Receiver(
        depths=np.linspace(5, 240, 50),
        ranges=np.linspace(100, 15000, 100),
    )

    # Run RAM with true RD bottom (via modified Fortran)
    print("\n--- Running RAM with range-dependent bottom ---")
    try:
        ram = RAM(verbose=True)
        result = ram.run(env, source, receiver)
        print(f"RAM TL: {np.nanmin(result.data):.1f} to {np.nanmax(result.data):.1f} dB")

        fig1, ax1, _ = plot_transmission_loss(result, env, contours=[70, 85, 100])
        ax1.set_title('RAM TL — Range-Dependent Bottom (Mud to Sand)')
        plt.savefig(OUTPUT_DIR / 'example_25_rd_bottom_tl.png', dpi=150, bbox_inches='tight')
        print(f"  Saved: example_25_rd_bottom_tl.png")
    except Exception as e:
        print(f"  RAM error: {e}")
        import traceback
        traceback.print_exc()

    # Plot RD bottom properties
    fig2, axes2 = plot_bottom_properties(env)
    plt.savefig(OUTPUT_DIR / 'example_25_rd_bottom_props.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: example_25_rd_bottom_props.png")

    # Plot 2D SSP
    fig3, ax3 = plot_ssp_2d(env)
    plt.savefig(OUTPUT_DIR / 'example_25_rd_ssp.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: example_25_rd_ssp.png")

    # Plot full environment
    fig4, axes4 = plot_environment_advanced(env, source, receiver)
    plt.savefig(OUTPUT_DIR / 'example_25_rd_env.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: example_25_rd_env.png")

    plt.close('all')


def demo_rd_layered_bottom():
    """Part 4: Range-dependent layered bottom with RAM."""
    print("\n" + "=" * 70)
    print("PART 4: Range-Dependent Layered Bottom (depth+range) with RAM")
    print("=" * 70)

    # Near-shore: soft mud over clay
    near = LayeredBottom(
        layers=[
            SedimentLayer(thickness=5.0, sound_speed=1500.0, density=1.2,
                          attenuation=1.0),
            SedimentLayer(thickness=15.0, sound_speed=1550.0, density=1.4,
                          attenuation=0.8),
        ],
        halfspace=BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1700.0, density=1.8, attenuation=0.2,
        ),
    )

    # Mid-range: mixed sediment
    mid = LayeredBottom(
        layers=[
            SedimentLayer(thickness=3.0, sound_speed=1580.0, density=1.5,
                          attenuation=0.6),
            SedimentLayer(thickness=12.0, sound_speed=1650.0, density=1.7,
                          attenuation=0.4),
        ],
        halfspace=BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1900.0, density=2.0, attenuation=0.1,
        ),
    )

    # Offshore: hard sand over rock
    far = LayeredBottom(
        layers=[
            SedimentLayer(thickness=2.0, sound_speed=1700.0, density=1.9,
                          attenuation=0.3),
            SedimentLayer(thickness=8.0, sound_speed=1850.0, density=2.1,
                          attenuation=0.2),
        ],
        halfspace=BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=2200.0, density=2.5, attenuation=0.05,
        ),
    )

    rdl = RangeDependentLayeredBottom(
        ranges_km=np.array([0, 7.5, 15]),
        depths=np.array([120, 180, 280]),
        profiles=[near, mid, far],
    )

    print(f"Max sediment thickness: {rdl.max_total_thickness():.1f} m")
    print(f"Ranges: {rdl.ranges_km} km")
    for i, lb in enumerate(rdl.profiles):
        print(f"  r={rdl.ranges_km[i]:.1f} km: {len(lb.layers)} layers, "
              f"total {lb.total_thickness():.0f} m")

    env = uacpy.Environment(
        name='Shelf: Mud/Clay to Sand/Rock',
        depth=280.0,
        ssp_type='isovelocity',
        sound_speed=1500.0,
        bottom=rdl,
    )

    print(f"has_range_dependent_layered_bottom: "
          f"{env.has_range_dependent_layered_bottom()}")

    source = uacpy.Source(frequency=100.0, depth=30.0)
    receiver = uacpy.Receiver(
        depths=np.linspace(5, 270, 50),
        ranges=np.linspace(100, 15000, 100),
    )

    # Run RAM (supports RD layered via Fortran sediment file)
    print("\n--- Running RAM with range-dependent layered bottom ---")
    try:
        ram = RAM(verbose=True)
        result = ram.run(env, source, receiver)
        print(f"RAM TL: {np.nanmin(result.data):.1f} to "
              f"{np.nanmax(result.data):.1f} dB")

        fig1, ax1, _ = plot_transmission_loss(result, env, contours=[70, 85, 100])
        ax1.set_title('RAM TL — Range-Dependent Layered Bottom')
        plt.savefig(OUTPUT_DIR / 'example_25_rdl_tl.png', dpi=150,
                    bbox_inches='tight')
        print(f"  Saved: example_25_rdl_tl.png")
    except Exception as e:
        print(f"  RAM error: {e}")
        import traceback
        traceback.print_exc()

    # Plot the RD layered structure
    fig2, axes2 = plot_rd_layered_bottom(env)
    plt.savefig(OUTPUT_DIR / 'example_25_rdl_structure.png', dpi=150,
                bbox_inches='tight')
    print(f"  Saved: example_25_rdl_structure.png")

    plt.close('all')


def main():
    print("\n" + "=" * 80)
    print("EXAMPLE 25: Range/Depth-Dependent Features + Bellhop+BOUNCE")
    print("=" * 80)

    demo_bellhop_bounce()
    demo_layered_bottom()
    demo_range_dependent_bottom()
    demo_rd_layered_bottom()

    print("\n" + "=" * 80)
    print("EXAMPLE 25 COMPLETE")
    print("=" * 80)
    print("\nFeatures demonstrated:")
    print("  - Bellhop.run_with_bounce() for elastic bottom")
    print("  - LayeredBottom with Scooter (NMEDIA > 1)")
    print("  - Range-dependent bottom (scalar) with RAM")
    print("  - Range-dependent LAYERED bottom (depth+range) with RAM")
    print("  - plot_transmission_loss() with contours")
    print("  - plot_layered_bottom() for sediment structure")
    print("  - plot_rd_layered_bottom() for RD layered structure")
    print("  - plot_bottom_properties() for RD bottom")
    print("  - plot_ssp_2d() for range-dependent SSP")
    print("  - plot_environment_advanced() for full overview")
    return 0


if __name__ == '__main__':
    sys.exit(main())
