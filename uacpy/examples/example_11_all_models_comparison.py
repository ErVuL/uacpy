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

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy import RangeDependentBottom
from uacpy.models import Bellhop, RAM, KrakenField, Scooter, OAST
from uacpy.visualization.plots import (
    plot_transmission_loss,
    plot_ssp_2d,
    plot_bottom_properties,
    plot_environment_advanced,
    compare_models
)

def main():
    print("=" * 70)
    print("ADVANCED ALL-MODELS COMPARISON")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════════════════
    # CREATE 2D RANGE-DEPENDENT SSP
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[Setup 1/3] Creating 2D range-dependent SSP...")

    depths = np.linspace(0, 250, 26)
    ranges_km = np.array([0, 5, 10, 15, 20, 25])

    # Thermal front: warm shallow water on shelf, cold deep water offshore
    ssp_2d_matrix = np.zeros((len(depths), len(ranges_km)))

    for i_range, r_km in enumerate(ranges_km):
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
        [5000, 120],
        [10000, 150],
        [15000, 200],
        [20000, 250],
        [25000, 300],
    ])

    bottom_rd = RangeDependentBottom(
        ranges_km=bathymetry[:, 0] / 1000.0,
        depths=bathymetry[:, 1],
        sound_speed=np.array([1550, 1580, 1620, 1670, 1720, 1780]),
        density=np.array([1.4, 1.5, 1.65, 1.8, 1.95, 2.1]),
        attenuation=np.array([1.0, 0.8, 0.6, 0.5, 0.4, 0.3]),
        shear_speed=np.zeros(6),
        acoustic_type='half-space'
    )

    print(f"  ✓ Bottom properties vary over {bathymetry[-1, 0]/1000:.0f} km")

    # ═══════════════════════════════════════════════════════════════════════
    # CREATE ENVIRONMENT WITH ALL FEATURES
    # ═══════════════════════════════════════════════════════════════════════

    print("[Setup 3/3] Creating full environment...")

    env = uacpy.Environment(
        name="Continental Margin - Frontal Zone",
        depth=300.0,
        ssp_type='pchip',
        ssp_data=ssp_1d,
        ssp_2d_ranges=ranges_km,
        ssp_2d_matrix=ssp_2d_matrix,
        bathymetry=bathymetry,
        bottom=bottom_rd
    )

    print(f"  ✓ is_range_dependent: {env.is_range_dependent}")
    print(f"  ✓ has_range_dependent_ssp: {env.has_range_dependent_ssp()}")
    print(f"  ✓ has_range_dependent_bottom: {env.has_range_dependent_bottom()}")

    source = uacpy.Source(depth=50.0, frequency=100.0)
    receiver = uacpy.Receiver(
        depths=np.linspace(5, 280, 50),
        ranges=np.linspace(100, 25000, 100)
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RUN ALL MODELS
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("RUNNING ALL MODELS")
    print("=" * 70)

    results = {}

    # Bellhop with Cerveny beams
    print("\n[1/5] Bellhop (Cerveny beams + Thorp attenuation)...")
    try:
        bellhop = Bellhop(verbose=False)
        results['Bellhop'] = bellhop.run(
            env, source, receiver,
            beam_type='C',
            volume_attenuation='T',
            beam_width_type='F',
            eps_multiplier=1.0,
            n_beams=500
        )
        print("  ✓ Success")
    except Exception as e:
        print(f"  ✗ {e}")

    # RAM with range-dependent features
    print("[2/5] RAM (range-dependent SSP + bottom)...")
    try:
        ram = RAM(verbose=False)
        results['RAM'] = ram.run(env, source, receiver)
        print("  ✓ Success - using 2D SSP and range-dependent bottom!")
    except Exception as e:
        print(f"  ✗ {e}")

    # KrakenField with mode coupling
    print("[3/5] KrakenField (adiabatic mode coupling)...")
    try:
        krakenfield = KrakenField(verbose=False, mode_coupling='adiabatic', n_segments=8)
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
        # Scooter is range-independent - use approximation
        env_approx = env.get_range_independent_approximation(method='median')
        results['Scooter'] = scooter.run(env_approx, source, receiver,
                                        volume_attenuation='T')
        print("  ✓ Success (range-independent approximation)")
    except Exception as e:
        print(f"  ✗ {e}")

    # OAST with volume attenuation
    print("[5/5] OAST (wavenumber integration)...")
    try:
        oast = OAST(verbose=False)
        # OAST is range-independent - use approximation
        env_approx = env.get_range_independent_approximation(method='median')
        results['OAST'] = oast.run(env_approx, source, receiver,
                                   volume_attenuation='T')
        print("  ✓ Success (range-independent approximation)")
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
    plt.savefig(OUTPUT_DIR / 'example_11_environment.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / 'example_11_environment.png'}")

    # Plot 2: 2D SSP heatmap
    print("[2/5] 2D SSP heatmap...")
    fig2, ax2 = plot_ssp_2d(env, cmap='RdYlBu_r')
    ax2.set_title('Thermal Front: 2D Range-Dependent SSP')
    plt.savefig(OUTPUT_DIR / 'example_11_ssp_2d.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / 'example_11_ssp_2d.png'}")

    # Plot 3: Bottom properties
    print("[3/5] Bottom properties...")
    fig3, axes3 = plot_bottom_properties(env)
    plt.savefig(OUTPUT_DIR / 'example_11_bottom.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {OUTPUT_DIR / 'example_11_bottom.png'}")

    # Plot 4: Model comparison
    if len(results) >= 2:
        print("[4/5] Model comparison...")
        fig4, axes4 = compare_models(results, env)
        plt.savefig(OUTPUT_DIR / 'example_11_comparison.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {OUTPUT_DIR / 'example_11_comparison.png'}")

    # Plot 5: Individual TL fields with shared colorbar
    if len(results) > 0:
        print("[5/5] Individual TL fields with shared colorbar...")
        n_models = len(results)
        fig5, axes5 = plt.subplots(2, 3, figsize=(18, 10))
        axes5 = axes5.flatten()

        # Use show_colorbar=False for cleaner subplots; add a single shared colorbar below
        for idx, (model_name, result) in enumerate(results.items()):
            if idx < 6:
                _, _, _ = plot_transmission_loss(result, env, ax=axes5[idx],
                                                 show_colorbar=False,  # Disable individual colorbars
                                                 contours=[70, 90])    # Overlay TL contours
                if result is not None:
                    axes5[idx].set_title(f'{model_name}\n({result.metadata.get("model", "")})')
                else:
                    axes5[idx].set_title(f'{model_name}\n(Failed)')

        # Hide unused subplots
        for idx in range(len(results), 6):
            axes5[idx].axis('off')

        # Add single shared colorbar
        import matplotlib as mpl
        cbar_ax = fig5.add_axes([0.92, 0.15, 0.015, 0.7])
        norm = mpl.colors.Normalize(vmin=50, vmax=110)
        cmap = mpl.cm.get_cmap('jet_r')
        cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
        cb.set_label('TL (dB)', fontsize=12, fontweight='bold')

        plt.suptitle('All Models Comparison (shared colorbar + auto TL limits + contours)',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'example_11_models.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {OUTPUT_DIR / 'example_11_models.png'} (shared colorbar)")

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
                tl_min = result.data.min()
                tl_max = result.data.max()
                tl_mean = result.data.mean()
                print(f"  {model_name:12s}: TL range [{tl_min:5.1f}, {tl_max:5.1f}] dB, mean = {tl_mean:5.1f} dB")
            else:
                print(f"  {model_name:12s}: Failed")

    print("\n" + "=" * 70)
    print("ALL-MODELS ADVANCED EXAMPLE COMPLETE")
    print("=" * 70)
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

    return 0

if __name__ == "__main__":
    sys.exit(main())
