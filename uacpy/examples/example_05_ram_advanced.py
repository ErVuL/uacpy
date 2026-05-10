"""
===============================================================================
ADVANCED EXAMPLE: RAM (mpiramS) - Range-Dependent SSP and Bottom
===============================================================================

OBJECTIVE:
    Demonstrate RAM's full range-dependent capabilities using the mpiramS
    Fortran PE backend:
    - 2D range-dependent SSP (thermal front/eddy)
    - Range-dependent bottom properties (sediment transition)
    - Parabolic equation accuracy for complex environments
    - Broadband (time-series) mode for time-domain analysis

SCENARIO:
    Thermal Front - Warm water meets cold water with sediment change

    Range progression (0 -> 20 km):
    - SSP: Warm stratified -> Cold well-mixed
    - Bottom: Soft mud -> Hard sand
    - Depth: Shallow shelf -> Deeper slope

FEATURES DEMONSTRATED:
    - 2D SSP matrix (sound speed varies with depth AND range)
    - Range-dependent bottom properties
    - Thermal front modeling
    - Continental shelf transition
    - COHERENT_TL mode (narrowband TL over range-depth grid)
    - TIME_SERIES mode (broadband complex field for IFFT)

===============================================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import uacpy  # noqa: E402
from uacpy.core.environment import SoundSpeedProfile  # noqa: E402
from uacpy import RangeDependentBottom  # noqa: E402
from uacpy.models import RAM  # noqa: E402
from uacpy.visualization.plots import (  # noqa: E402
    plot_transmission_loss,
    plot_rd_bottom,
    plot_environment_advanced,
)


def main():
    print("\n" + "═" * 80)
    print("EXAMPLE 05: RAM advanced features - Range-Dependent SSP & Bottom")
    print("═" * 80)

    # ═══════════════════════════════════════════════════════════════════════
    # CREATE 2D RANGE-DEPENDENT SSP (Thermal Front)
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[Setup] Creating warm stratified SSP profile...")

    depths = np.array([0., 25, 50, 75, 100, 120, 150])
    T = 8 + 12 * np.exp(-depths / 50)  # 20 °C surface, ~8 °C deep
    ssp_c = 1449 + 4.6*T - 0.055*T**2 + 0.00029*T**3 + 0.016*depths
    ssp_1d = np.column_stack([depths, ssp_c])
    print(f"  ✓ SSP range: {ssp_c.min():.1f} to {ssp_c.max():.1f} m/s")

    # ═══════════════════════════════════════════════════════════════════════
    # CREATE RANGE-DEPENDENT BOTTOM (Sediment Transition)
    # ═══════════════════════════════════════════════════════════════════════

    print("[Setup] Creating range-dependent bottom (mud → sand)...")

    # Bathymetry: shelf with mild slope over 3 km
    bathymetry = np.array([
        [0, 100],
        [1000, 110],
        [2000, 130],
        [3000, 150],
    ])

    # Bottom: soft mud → hard sand
    bottom_rd = RangeDependentBottom(
        ranges=np.array([0, 1000, 2000, 3000]),
        sound_speed=np.array([1500, 1580, 1640, 1700]),
        density=np.array([1.2, 1.5, 1.8, 2.0]),
        attenuation=np.array([1.0, 0.7, 0.5, 0.3]),
        shear_speed=np.zeros(4),
        acoustic_type='half-space'
    )

    print(f"  ✓ Bottom sound speed: {bottom_rd.sound_speed.min():.0f} → {bottom_rd.sound_speed.max():.0f} m/s")
    print(f"  ✓ Bottom density: {bottom_rd.density.min():.1f} → {bottom_rd.density.max():.1f} g/cm³")

    # ═══════════════════════════════════════════════════════════════════════
    # CREATE ENVIRONMENT WITH ALL RANGE-DEPENDENT FEATURES
    # ═══════════════════════════════════════════════════════════════════════

    env = uacpy.Environment(
        name="Sediment Transition with Sloping Shelf",
        ssp=SoundSpeedProfile.from_pairs(ssp_1d, interp='pchip'),
        bathymetry=bathymetry,
        bottom=bottom_rd
    )

    print("\n✓ Environment created:")
    print(f"    - is_range_dependent: {env.is_range_dependent}")
    print(f"    - has_range_dependent_ssp: {env.has_range_dependent_ssp()}")
    print(f"    - has_range_dependent_bottom: {env.has_range_dependent_bottom()}")

    # ═══════════════════════════════════════════════════════════════════════
    # SOURCE & RECEIVER
    # ═══════════════════════════════════════════════════════════════════════

    source = uacpy.Source(depths=50.0, frequencies=100.0)
    receiver = uacpy.Receiver(
        depths=np.linspace(5, 145, 30),
        ranges=np.linspace(100, 3000, 30)
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RUN RAM with Range-Dependent Features
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[Run] RAM (mpiramS) with range-dependent SSP and bottom...")
    print("  Mode: COHERENT_TL (narrowband, range-depth TL grid)")

    try:
        ram = RAM(verbose=True, accuracy=1e-1)
        result = ram.run(env, source, receiver)
        print("  RAM TL completed successfully")
        print(f"  TL range: {np.nanmin(result.tl):.1f} to {np.nanmax(result.tl):.1f} dB")
    except Exception as e:
        print(f"  RAM error: {e}")
        import traceback
        traceback.print_exc()
        result = None

    # ═══════════════════════════════════════════════════════════════════════
    # COMPARISON: KrakenField with Range-Independent Approximation
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[Comparison] Running KrakenField for comparison...")
    print("  Note: KrakenField will use range-independent approximation for the bottom")
    print("  → Range-dependent bottom effects will not be fully captured")
    print("  → For full accuracy, use RAM or Bellhop\n")

    from uacpy.models import KrakenField, Bellhop

    # Run KrakenField
    try:
        krakenfield = KrakenField(verbose=False)
        result_krakenfield = krakenfield.compute_tl(env, source, receiver)
        print("  ✓ KrakenField completed (using range-independent approximation)")
        print(f"  ✓ TL range: {result_krakenfield.tl.min():.1f} to {result_krakenfield.tl.max():.1f} dB")
    except Exception as e:
        print(f"  ✗ KrakenField: {e}")
        result_krakenfield = None

    # Run Bellhop (supports range-dependent natively)
    print("\n  Running Bellhop (native range-dependent support)...")
    try:
        bellhop = Bellhop(verbose=False)
        result_bellhop = bellhop.compute_tl(env, source, receiver)
        print("  ✓ Bellhop completed (full range-dependent capability)")
        print(f"  ✓ TL range: {result_bellhop.tl.min():.1f} to {result_bellhop.tl.max():.1f} dB")
    except Exception as e:
        print(f"  ✗ Bellhop: {e}")
        result_bellhop = None

    # Comparisons (use nanmean because RAM masks sub-bottom cells as NaN)
    if result is not None and result_krakenfield is not None:
        diff_kf = np.abs(result.tl - result_krakenfield.tl)
        print(f"\n  RAM vs KrakenField: Mean diff = {np.nanmean(diff_kf):.1f} dB (range-dependent effects)")

    if result is not None and result_bellhop is not None:
        diff_bh = np.abs(result.tl - result_bellhop.tl)
        print(f"  RAM vs Bellhop: Mean diff = {np.nanmean(diff_bh):.1f} dB (PE vs ray methods)")

    if result_bellhop is not None and result_krakenfield is not None:
        diff_bk = np.abs(result_bellhop.tl - result_krakenfield.tl)
        print(f"  Bellhop vs KrakenField: Mean diff = {np.nanmean(diff_bk):.1f} dB")

    # ═══════════════════════════════════════════════════════════════════════
    # VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════════

    print("\nGenerating plots...")

    # Plot 1: Advanced environment overview
    fig1, axes1 = plot_environment_advanced(env, source, receiver)
    plt.savefig(OUTPUT_DIR / 'example_05_environment.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: example_05_environment.png")

    # Plot 2: SSP profile
    from uacpy.visualization.plots import plot_ssp
    fig2, ax2 = plot_ssp(env)
    plt.savefig(OUTPUT_DIR / 'example_05_ssp.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: example_05_ssp.png")

    # Plot 3: Bottom properties detail
    fig3, _ = plot_rd_bottom(env)
    plt.savefig(OUTPUT_DIR / 'example_05_bottom.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: example_05_bottom.png")

    # Plot 4: TL field result
    if result is not None:
        fig4, ax4 = plot_transmission_loss(
            result, env,
            contours=[70, 85, 100],
            show_colorbar=True,
            vmin=40, vmax=100,
        )
        ax4.set_title('RAM: Sediment Transition with Sloping Shelf')

        plt.savefig(OUTPUT_DIR / 'example_05_result.png', dpi=150, bbox_inches='tight')
        print("  ✓ Saved: example_05_result.png")

    # Plot 5: Three-Model Comparison (RAM, Bellhop, KrakenField)
    if result is not None and result_bellhop is not None and result_krakenfield is not None:
        from uacpy.visualization.plots import compare_models, plot_tl_difference
        fig5, _ = compare_models(
            {'RAM': result, 'Bellhop': result_bellhop, 'KrakenField': result_krakenfield},
            env, vmin=40, vmax=100,
            suptitle='Three-Model Comparison — Sediment Transition + Sloping Shelf',
        )
        fig5.savefig(OUTPUT_DIR / 'example_05_comparison.png', dpi=150)
        plt.close(fig5)
        print("  ✓ Saved: example_05_comparison.png")

        fig6, axes6 = plt.subplots(1, 3, figsize=(20, 5))
        plot_tl_difference(result, result_bellhop, env, ax=axes6[0],
                           label='RAM − Bellhop', show_colorbar=True)
        plot_tl_difference(result, result_krakenfield, env, ax=axes6[1],
                           label='RAM − KrakenField', show_colorbar=True)
        plot_tl_difference(result_bellhop, result_krakenfield, env,
                           ax=axes6[2], label='Bellhop − KrakenField',
                           show_colorbar=True)
        fig6.suptitle('Pairwise Differences (signed, dB)',
                      fontsize=13, fontweight='bold')
        fig6.tight_layout()
        fig6.savefig(OUTPUT_DIR / 'example_05_differences.png', dpi=150)
        plt.close(fig6)
        print("  ✓ Saved: example_05_differences.png")

    print("\nFeatures demonstrated:")
    print("  ✓ Range-dependent bottom properties (mud → sand)")
    print("  ✓ Sloping shelf bathymetry")
    print("  ✓ RAM parabolic equation with PE accuracy control")
    print("  ✓ Three-model comparison (RAM / Bellhop / KrakenField)")

    print("\n✓ Example 05 complete\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
