"""
═══════════════════════════════════════════════════════════════════════════════
ADVANCED EXAMPLE: All Models - Comprehensive Comparison
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    Compare ALL propagation models with advanced features:
    - Bellhop (with Cerveny beams)
    - RAM (with range-dependent SSP and bottom)
    - KrakenField (with mode coupling)
    - Scooter (with volume attenuation)
    - OAST (wavenumber integration)
SCENARIO:
    Realistic Ocean: Continental Margin with Thermal Structure
    - 2D range-dependent SSP (frontal zone)
    - Range-dependent bottom (sediment transition)
    - Complex bathymetry (shelf break)
    - All models with volume attenuation

FEATURES DEMONSTRATED:
    ✓ All 6 propagation models (including OAST)
    ✓ 2D SSP (range-dependent sound speed)
    ✓ Range-dependent bottom properties
    ✓ Volume attenuation (Thorp) in all models
    ✓ Modal model comparison (Kraken, Scooter, OAST)
    ✓ Advanced plotting (2D SSP heatmap, bottom properties)
    ✓ Statistical model comparison
    ✓ Comprehensive visualization

═══════════════════════════════════════════════════════════════════════════════
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
from uacpy.models import Bellhop, RAM, KrakenField, Scooter, OAST  # noqa: E402
from uacpy.visualization.plots import (  # noqa: E402
    plot_ssp_2d,
    plot_rd_bottom,
    plot_environment_advanced,
    compare_models,
)


def main():
    print("\n" + "═" * 80)
    print("EXAMPLE 07: All-models comparison")
    print("═" * 80)

    # ═══════════════════════════════════════════════════════════════════════
    # CREATE 2D RANGE-DEPENDENT SSP
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[Setup 1/3] Creating 2D range-dependent SSP...")

    depths = np.linspace(0, 200, 21)
    ranges_m = np.array([0.0, 2000.0, 4000.0, 6000.0, 8000.0])

    # Thermal front: warm shallow water on shelf, cold deep water offshore
    ssp_2d_matrix = np.zeros((len(depths), len(ranges_m)))

    for i_range, r_m in enumerate(ranges_m):
        r_km = r_m / 1000.0
        # Temperature decreases with range (frontal zone)
        T_surface = 18 - r_km * 0.3  # 18°C → 10.5°C
        T_bottom = 8 - r_km * 0.1    # 8°C → 5.5°C

        # Exponential stratification
        T_profile = T_bottom + (T_surface - T_bottom) * np.exp(-depths / 40)

        # Mackenzie sound speed formula (simplified)
        c = 1449 + 4.6 * T_profile + 0.016 * depths

        ssp_2d_matrix[:, i_range] = c

    ssp_1d = np.column_stack([depths, ssp_2d_matrix[:, 0]])

    print(f"  ✓ 2D SSP created: {ssp_2d_matrix.shape}")
    print(f"  ✓ Sound speed range: {ssp_2d_matrix.min():.1f} - {ssp_2d_matrix.max():.1f} m/s")

    # ═══════════════════════════════════════════════════════════════════════
    # CREATE RANGE-DEPENDENT BOTTOM
    # ═══════════════════════════════════════════════════════════════════════

    print("[Setup 2/3] Creating range-dependent bottom...")

    bathymetry = np.array([
        [0, 100],
        [2000, 110],
        [4000, 130],
        [6000, 150],
        [8000, 170],
    ])

    bottom_rd = RangeDependentBottom(
        ranges=bathymetry[:, 0],
        sound_speed=np.array([1550, 1600, 1640, 1680, 1720]),
        density=np.array([1.4, 1.55, 1.7, 1.85, 2.0]),
        attenuation=np.array([1.0, 0.8, 0.6, 0.5, 0.4]),
        shear_speed=np.zeros(5),
        acoustic_type='half-space'
    )

    print(f"  ✓ Bottom properties vary over {bathymetry[-1, 0]/1000:.0f} km")

    # ═══════════════════════════════════════════════════════════════════════
    # CREATE ENVIRONMENT WITH ALL FEATURES
    # ═══════════════════════════════════════════════════════════════════════

    print("[Setup 3/3] Creating full environment...")

    env = uacpy.Environment(
        name="Continental Margin - Frontal Zone",
        ssp=SoundSpeedProfile.from_2d(depths=ssp_1d[:, 0], ranges=ranges_m, matrix=ssp_2d_matrix,
                                      interp='pchip',
                                      ),
        bathymetry=bathymetry,
        bottom=bottom_rd,
        absorption=uacpy.Thorp(),
    )

    print(f"  ✓ is_range_dependent: {env.is_range_dependent}")
    print(f"  ✓ has_range_dependent_ssp: {env.has_range_dependent_ssp()}")
    print(f"  ✓ has_range_dependent_bottom: {env.has_range_dependent_bottom()}")

    source = uacpy.Source(depths=50.0, frequencies=100.0)
    receiver = uacpy.Receiver(
        depths=np.linspace(5, 165, 30),
        ranges=np.linspace(100, 8000, 40)
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RUN ALL MODELS
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("RUNNING ALL MODELS")
    print("=" * 70)

    results = {}

    print("\n[1/5] Bellhop (Gaussian beams + Thorp attenuation)...")
    try:
        bellhop = Bellhop(verbose=False)
        results['Bellhop'] = bellhop.compute_tl(env, source, receiver)
        print("  ✓ Success — uacpy auto-collapses unsupported axes")
    except Exception as e:
        print(f"  ✗ {e}")

    # RAM with range-dependent features
    print("[2/5] RAM (range-dependent SSP + bottom)...")
    try:
        ram = RAM(verbose=False, accuracy=1e-1)
        results['RAM'] = ram.run(env, source, receiver)
        print("  ✓ Success - using 2D SSP and range-dependent bottom!")
    except Exception as e:
        print(f"  ✗ {e}")

    # KrakenField with mode coupling
    print("[3/5] KrakenField (adiabatic mode coupling)...")
    try:
        krakenfield = KrakenField(verbose=False, mode_coupling='adiabatic', n_segments=4)
        results['KrakenField'] = krakenfield.run(
            env, source, receiver
        )
        print("  ✓ Success - coupled modes with range-dependent bottom!")
    except Exception as e:
        print(f"  ✗ {e}")

    # Scooter with volume attenuation
    print("[4/5] Scooter (volume attenuation)...")
    try:
        scooter = Scooter(verbose=False)
        results['Scooter'] = scooter.run(env, source, receiver)
        print("  ✓ Success — uacpy auto-collapses unsupported axes")
    except Exception as e:
        print(f"  ✗ {e}")

    # OAST with volume attenuation
    print("[5/5] OAST (wavenumber integration)...")
    try:
        oast = OAST(verbose=False)
        results['OAST'] = oast.run(env, source, receiver)
        print("  ✓ Success — uacpy auto-collapses unsupported axes")
    except Exception as e:
        print(f"  ✗ {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    # Plot 1: Advanced environment overview
    print("\n[1/5] Environment overview...")
    fig1, axes1 = plot_environment_advanced(env, source, receiver)
    plt.savefig(OUTPUT_DIR / 'example_07_environment.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / 'example_07_environment.png'}")

    # Plot 2: 2D SSP heatmap
    print("[2/5] 2D SSP heatmap...")
    fig2, ax2 = plot_ssp_2d(env, cmap='RdYlBu_r')
    ax2.set_title('Thermal Front: 2D Range-Dependent SSP')
    plt.savefig(OUTPUT_DIR / 'example_07_ssp_2d.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / 'example_07_ssp_2d.png'}")

    # Plot 3: Range-dependent bottom (geological cross-section)
    print("[3/5] Range-dependent bottom...")
    fig3, _ = plot_rd_bottom(env)
    fig3.savefig(OUTPUT_DIR / 'example_07_bottom.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"  ✓ Saved: {OUTPUT_DIR / 'example_07_bottom.png'}")

    # Plot 4: Model comparison
    if len(results) >= 2:
        print("[4/5] Model comparison...")
        fig4, axes4 = compare_models(results, env)
        plt.savefig(OUTPUT_DIR / 'example_07_comparison.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {OUTPUT_DIR / 'example_07_comparison.png'}")

    # Plot 5: TL fields with contour overlay and shared colorbar
    if len(results) > 0:
        print("[5/5] TL fields with contour overlay...")
        fig5, _ = compare_models(
            results, env, ncols=3, vmin=50, vmax=110, contours=[70, 90],
            suptitle='All Models — TL with 70/90 dB contours',
        )
        fig5.savefig(OUTPUT_DIR / 'example_07_models.png', dpi=150)
        plt.close(fig5)
        print(f"  ✓ Saved: {OUTPUT_DIR / 'example_07_models.png'}")

    # ═══════════════════════════════════════════════════════════════════════
    # STATISTICS
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)

    if len(results) > 0:
        print(f"\nModels run: {len(results)}")
        for model_name, result in results.items():
            if result is not None:
                tl_min = result.tl.min()
                tl_max = result.tl.max()
                tl_mean = result.tl.mean()
                print(f"  {model_name:12s}: TL range [{tl_min:5.1f}, {tl_max:5.1f}] dB, mean = {tl_mean:5.1f} dB")
            else:
                print(f"  {model_name:12s}: Failed")

    print("\nFeatures demonstrated across all models:")
    print("  ✓ 2D range-dependent SSP (thermal front)")
    print("  ✓ Range-dependent bottom properties")
    print("  ✓ Volume attenuation (Thorp)")
    print("  ✓ Continental margin scenario")
    print("  ✓ Model comparison and statistics")
    print("  ✓ Advanced visualization suite")
    print("\nPlotting features demonstrated:")
    print("  ✓ Shared colorbar for multi-panel comparisons (show_colorbar=False)")
    print("  ✓ Auto TL limits (median + 0.75σ, rounded to 10 dB)")
    print("  ✓ Contour overlays at 70, 90 dB")
    print("  ✓ jet_r colormap (blue=good, red=poor) - AT standard")
    print("\n  All models tested with realistic, complex environment!")

    print("\n✓ Example 07 complete\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
