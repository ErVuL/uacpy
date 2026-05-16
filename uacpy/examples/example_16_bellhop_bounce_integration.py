"""
===============================================================================
EXAMPLE 16: Bellhop + BOUNCE Integration, Layered Bottom, Range-Dependent Bottom
===============================================================================

OBJECTIVE:
    Demonstrate three key features for range/depth-dependent modeling:
    1. Bellhop.run_with_bounce() — explicit BOUNCE control for elastic
       bottoms. (Bellhop.run() already auto-routes through BOUNCE for
       elastic / layered bottoms with a UserWarning; use
       run_with_bounce when you need to pin c_low / c_high / rmax.)
    2. LayeredBottom — multi-layer sediment with Kraken
    3. Range-dependent bottom properties — with RAM and visualization

FEATURES DEMONSTRATED:
    - Bellhop.run_with_bounce() for explicit BOUNCE parameters
    - LayeredBottom + SedimentLayer for depth-dependent sediment
    - RangeDependentBottom with RAM (true Fortran RD support)
    - plot_field(), plot_environment(), plot_environment()
    - plot_environment(), plot_environment()

===============================================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import uacpy  # noqa: E402
from uacpy.core.environment import SoundSpeedProfile  # noqa: E402
from uacpy import (  # noqa: E402
    RangeDependentBottom, SedimentLayer, LayeredBottom,
    RangeDependentLayeredBottom, BoundaryProperties,
)
from uacpy.models import Bellhop, RAM, RunMode  # noqa: E402
from uacpy.visualization.plots import (  # noqa: E402
    plot_field,
    plot_environment,
)

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)


def _plot_tl_difference(a, b, env=None, *, ax=None, title=None,
                         vmin=-10.0, vmax=10.0, diff_vmax=None, **kw):
    """Plot TL(a) - TL(b) as a diverging-colourmap heatmap.

    ``diff_vmax`` is a symmetric range shortcut: ``vmin = -diff_vmax``,
    ``vmax = +diff_vmax``.
    """
    from uacpy import Field
    from uacpy.visualization import plot_field
    if diff_vmax is not None:
        vmin, vmax = -abs(diff_vmax), abs(diff_vmax)
    diff = Field(data=a.tl - b.tl, coords=dict(a.coords))
    return plot_field(
        diff, env=env, ax=ax, vmin=vmin, vmax=vmax,
        cmap='RdBu_r', title=title, **kw,
    )


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
        bathymetry=100,
        ssp=1500.0,
        bottom=bottom,
    )

    source = uacpy.Source(frequencies=500.0, depths=25.0)
    receiver = uacpy.Receiver(
        depths=np.linspace(1, 99, 30),
        ranges=np.linspace(100, 3000, 40),
    )

    bellhop = Bellhop(verbose=True)

    # Method 1: Standard half-space (ignores shear)
    print("\n--- Standard Bellhop (half-space, no shear) ---")
    result_hs = bellhop.run(env, source, receiver, run_mode=RunMode.COHERENT_TL)

    # Method 2: With BOUNCE (accounts for shear)
    print("\n--- Bellhop with BOUNCE reflection coefficients ---")
    result_bounce = bellhop.run_with_bounce(
        env, source, receiver,
        run_mode=RunMode.COHERENT_TL,
        c_low=1400.0, c_high=10000.0, rmax=10000.0,
    )

    # Compare
    tl_hs = result_hs.tl
    tl_bn = result_bounce.tl
    diff = tl_bn - tl_hs
    print(f"\nHalf-space TL: {np.nanmin(tl_hs):.1f} to {np.nanmax(tl_hs):.1f} dB")
    print(f"BOUNCE TL:     {np.nanmin(tl_bn):.1f} to {np.nanmax(tl_bn):.1f} dB")
    print(f"Max |diff|:    {np.nanmax(np.abs(diff)):.1f} dB")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmin, vmax = 40, 90
    plot_field(result_hs, env=env, ax=axes[0],
                           show_colorbar=False, vmin=vmin, vmax=vmax)
    axes[0].set_title('Half-space (no shear)', fontsize=11, fontweight='bold')
    plot_field(result_bounce, env=env, ax=axes[1],
                           show_colorbar=False, vmin=vmin, vmax=vmax)
    axes[1].set_title('BOUNCE (with shear)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('')
    _plot_tl_difference(result_bounce, result_hs, env, ax=axes[2],
                       label='BOUNCE − half-space',
                       diff_vmax=10, show_colorbar=False)
    axes[2].set_title('BOUNCE − half-space', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('')

    fig.suptitle(f'Bellhop + BOUNCE: f={source.frequencies[0]:.0f} Hz, '
                 f'elastic sandy bottom (cp={bottom.sound_speed}, '
                 f'cs={bottom.shear_speed} m/s)',
                 fontsize=12, fontweight='bold', y=0.995)
    fig.subplots_adjust(left=0.05, right=0.91, top=0.86, bottom=0.13,
                        wspace=0.18)
    tl_im = axes[1].collections[0] if axes[1].collections else None
    diff_im = axes[2].collections[0] if axes[2].collections else None
    if tl_im is not None:
        cax_tl = fig.add_axes([0.918, 0.13, 0.010, 0.36])
        fig.colorbar(tl_im, cax=cax_tl, label='TL (dB)')
    if diff_im is not None:
        cax_dif = fig.add_axes([0.918, 0.51, 0.010, 0.35])
        fig.colorbar(diff_im, cax=cax_dif, label='Δ TL (dB)')
    fig.savefig(OUTPUT_DIR / 'example_16_bounce_comparison.png', dpi=150)
    print("  ✓ Saved: output/example_16_bounce_comparison.png")

    # Also plot TL with contours using the plot function
    fig2, ax2 = plot_field(result_bounce, env=env, contours=[60, 70, 80])
    ax2.set_title('Bellhop + BOUNCE TL with Contours')
    plt.savefig(OUTPUT_DIR / 'example_16_bounce_tl.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: output/example_16_bounce_tl.png")

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
        bathymetry=200.0,
        ssp=1500.0,
        bottom=layered,
    )
    print(f"has_layered_bottom: {env.has_layered_bottom()}")

    source = uacpy.Source(frequencies=100.0, depths=50.0)
    receiver = uacpy.Receiver(
        depths=np.linspace(1, 199, 30),
        ranges=np.linspace(100, 5000, 40),
    )

    # Run Scooter (supports layered via NMEDIA > 1, gives TL directly)
    print("\n--- Running Scooter with layered bottom ---")
    try:
        from uacpy.models import Scooter
        scooter = Scooter(verbose=True)
        result = scooter.compute_tl(env, source, receiver)
        print(f"Scooter TL: {np.nanmin(result.tl):.1f} to {np.nanmax(result.tl):.1f} dB")

        # Plot TL
        fig1, ax1 = plot_field(result, env=env, contours=[70, 80, 90])
        ax1.set_title('Scooter TL — Layered Sediment (3 layers + halfspace)')
        plt.savefig(OUTPUT_DIR / 'example_16_layered_tl.png', dpi=150, bbox_inches='tight')
        print("  ✓ Saved: output/example_16_layered_tl.png")
    except Exception as e:
        print(f"  Scooter error: {e}")
        import traceback
        traceback.print_exc()

    # Plot layered bottom structure
    fig2, ax2 = plot_environment(env)
    plt.savefig(OUTPUT_DIR / 'example_16_layered_structure.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: output/example_16_layered_structure.png")

    # Plot environment overview
    fig3, axes3 = plot_environment(env)
    plt.savefig(OUTPUT_DIR / 'example_16_layered_env.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: output/example_16_layered_env.png")

    plt.close('all')


def demo_range_dependent_bottom():
    """Part 3: Range-dependent bottom with RAM."""
    print("\n" + "=" * 70)
    print("PART 3: Range-Dependent Bottom Properties with RAM")
    print("=" * 70)

    # Create RD bottom: mud transitioning to sand over 15 km
    bathymetry_rd = np.array([
        [0, 150], [3000, 160], [6000, 180],
        [9000, 200], [12000, 220], [15000, 250],
    ])

    bottom_rd = RangeDependentBottom(
        ranges=bathymetry_rd[:, 0].astype(float),
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
        ssp=SoundSpeedProfile.from_2d(depths=ssp_1d[:, 0], ranges=ranges_ssp * 1000.0, matrix=ssp_2d,
                                      ),
        bathymetry=bathymetry_rd,
        bottom=bottom_rd,
    )

    source = uacpy.Source(frequencies=100.0, depths=30.0)
    receiver = uacpy.Receiver(
        depths=np.linspace(5, 240, 30),
        ranges=np.linspace(100, 5000, 40),
    )

    # Run RAM with true RD bottom (via modified Fortran)
    print("\n--- Running RAM with range-dependent bottom ---")
    try:
        ram = RAM(verbose=True, accuracy=1e-1)
        result = ram.run(env, source, receiver)
        print(f"RAM TL: {np.nanmin(result.tl):.1f} to {np.nanmax(result.tl):.1f} dB")

        fig1, ax1 = plot_field(result, env=env, contours=[70, 85, 100])
        ax1.set_title('RAM TL — Range-Dependent Bottom (Mud to Sand)')
        plt.savefig(OUTPUT_DIR / 'example_16_rd_bottom_tl.png', dpi=150, bbox_inches='tight')
        print("  ✓ Saved: output/example_16_rd_bottom_tl.png")
    except Exception as e:
        print(f"  RAM error: {e}")
        import traceback
        traceback.print_exc()

    # Plot RD bottom properties
    fig2, _ = plot_environment(env)
    plt.savefig(OUTPUT_DIR / 'example_16_rd_bottom_props.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: output/example_16_rd_bottom_props.png")

    # Plot 2D SSP
    fig3, ax3 = plot_environment(env)
    plt.savefig(OUTPUT_DIR / 'example_16_rd_ssp.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: output/example_16_rd_ssp.png")

    # Plot full environment
    fig4, axes4 = plot_environment(env)
    plt.savefig(OUTPUT_DIR / 'example_16_rd_env.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: output/example_16_rd_env.png")

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
        ranges=np.array([0, 7500, 15000]),
        profiles=[near, mid, far],
    )
    rdl_bathymetry = np.column_stack([
        np.array([0.0, 7500.0, 15000.0]),
        np.array([120.0, 180.0, 280.0]),
    ])

    print(f"Max sediment thickness: {rdl.max_total_thickness():.1f} m")
    print(f"Ranges: {rdl.ranges / 1000.0} km")
    for i, lb in enumerate(rdl.profiles):
        print(f"  r={(rdl.ranges / 1000.0)[i]:.1f} km: {len(lb.layers)} layers, "
              f"total {lb.total_thickness():.0f} m")

    env = uacpy.Environment(
        name='Shelf: Mud/Clay to Sand/Rock',
        ssp=1500.0,
        bathymetry=rdl_bathymetry,
        bottom=rdl,
    )

    print(f"has_range_dependent_layered_bottom: "
          f"{env.has_range_dependent_layered_bottom()}")

    source = uacpy.Source(frequencies=100.0, depths=30.0)
    receiver = uacpy.Receiver(
        depths=np.linspace(5, 270, 30),
        ranges=np.linspace(100, 8000, 40),
    )

    # Run RAM (supports RD layered via Fortran sediment file)
    print("\n--- Running RAM with range-dependent layered bottom ---")
    try:
        ram = RAM(verbose=True, accuracy=1e-1)
        result = ram.run(env, source, receiver)
        print(f"RAM TL: {np.nanmin(result.tl):.1f} to "
              f"{np.nanmax(result.tl):.1f} dB")

        fig1, ax1 = plot_field(result, env=env, contours=[70, 85, 100])
        ax1.set_title('RAM TL — Range-Dependent Layered Bottom')
        plt.savefig(OUTPUT_DIR / 'example_16_rdl_tl.png', dpi=150,
                    bbox_inches='tight')
        print("  ✓ Saved: output/example_16_rdl_tl.png")
    except Exception as e:
        print(f"  RAM error: {e}")
        import traceback
        traceback.print_exc()

    # Plot the RD layered structure
    fig2, axes2 = plot_environment(env)
    plt.savefig(OUTPUT_DIR / 'example_16_rdl_structure.png', dpi=150,
                bbox_inches='tight')
    print("  ✓ Saved: output/example_16_rdl_structure.png")

    plt.close('all')


def main():
    print("\n" + "═" * 80)
    print("EXAMPLE 16: Range/Depth-Dependent Features + Bellhop+BOUNCE")
    print("═" * 80)

    demo_bellhop_bounce()
    demo_layered_bottom()
    demo_range_dependent_bottom()
    demo_rd_layered_bottom()

    print("\nFeatures demonstrated:")
    print("  - Bellhop.run_with_bounce() for elastic bottom")
    print("  - LayeredBottom with Scooter (NMEDIA > 1)")
    print("  - Range-dependent bottom (scalar) with RAM")
    print("  - Range-dependent LAYERED bottom (depth+range) with RAM")
    print("  - plot_field() with contours")
    print("  - plot_environment() for sediment structure")
    print("  - plot_environment() for RD layered structure")
    print("  - plot_environment() for RD bottom")
    print("  - plot_environment() for range-dependent SSP")
    print("  - plot_environment() for full overview")

    print("\n✓ Example 16 complete\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
