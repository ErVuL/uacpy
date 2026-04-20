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
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy import RangeDependentBottom
from uacpy.models import RAM
from uacpy.visualization.plots import (
    plot_transmission_loss,
    plot_ssp_2d,
    plot_bottom_properties,
    plot_environment_advanced
)

def main():
    print("=" * 70)
    print("ADVANCED RAM EXAMPLE - Range-Dependent SSP & Bottom")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════════════════
    # CREATE 2D RANGE-DEPENDENT SSP (Thermal Front)
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[Setup] Creating 2D SSP for thermal front...")

    # Depth grid for SSP
    depths = np.array([0, 25, 50, 75, 100, 150, 200, 250, 300])

    # Range points (0, 5, 10, 15, 20 km)
    ranges_km = np.array([0, 5, 10, 15, 20])

    # Create SSP matrix: warm water (left) transitions to cold (right)
    # Shape: (n_depth, n_range)
    ssp_2d_matrix = np.zeros((len(depths), len(ranges_km)))

    for i_range, r_km in enumerate(ranges_km):
        # Temperature transition: warm → cold
        # 0 km: 20°C surface, 10°C bottom
        # 20 km: 10°C surface, 8°C bottom
        temp_surface = 20 - (r_km / 20) * 10  # 20°C → 10°C
        temp_bottom = 10 - (r_km / 20) * 2    # 10°C → 8°C

        # Exponential temperature profile
        temp_profile = temp_bottom + (temp_surface - temp_bottom) * np.exp(-depths / 50)

        # Convert temperature to sound speed (simplified Mackenzie)
        # c ≈ 1449 + 4.6*T - 0.055*T² + 0.00029*T³ + 0.016*z
        T = temp_profile
        z = depths
        c = 1449 + 4.6*T - 0.055*T**2 + 0.00029*T**3 + 0.016*z

        ssp_2d_matrix[:, i_range] = c

    # Reference 1D SSP (first range profile)
    ssp_1d = np.column_stack([depths, ssp_2d_matrix[:, 0]])

    print(f"  ✓ Created 2D SSP: {ssp_2d_matrix.shape} (depth × range)")
    print(f"  ✓ SSP range: {ssp_2d_matrix.min():.1f} to {ssp_2d_matrix.max():.1f} m/s")

    # ═══════════════════════════════════════════════════════════════════════
    # CREATE RANGE-DEPENDENT BOTTOM (Sediment Transition)
    # ═══════════════════════════════════════════════════════════════════════

    print("[Setup] Creating range-dependent bottom (mud → sand)...")

    # Bathymetry
    bathymetry = np.array([
        [0, 100],       # 0 km: 100m (shelf)
        [5000, 150],    # 5 km
        [10000, 200],   # 10 km
        [15000, 250],   # 15 km
        [20000, 300],   # 20 km (slope)
    ])

    # Bottom: soft mud → hard sand
    bottom_rd = RangeDependentBottom(
        ranges_km=np.array([0, 5, 10, 15, 20]),
        depths=bathymetry[:, 1],
        sound_speed=np.array([1500, 1550, 1600, 1650, 1700]),  # Hardening
        density=np.array([1.2, 1.4, 1.6, 1.8, 2.0]),           # Compacting
        attenuation=np.array([1.0, 0.8, 0.6, 0.4, 0.3]),       # Less lossy
        shear_speed=np.zeros(5),
        acoustic_type='half-space'
    )

    print(f"  ✓ Bottom sound speed: {bottom_rd.sound_speed.min():.0f} → {bottom_rd.sound_speed.max():.0f} m/s")
    print(f"  ✓ Bottom density: {bottom_rd.density.min():.1f} → {bottom_rd.density.max():.1f} g/cm³")

    # ═══════════════════════════════════════════════════════════════════════
    # CREATE ENVIRONMENT WITH ALL RANGE-DEPENDENT FEATURES
    # ═══════════════════════════════════════════════════════════════════════

    env = uacpy.Environment(
        name="Thermal Front with Sediment Transition",
        depth=300.0,
        ssp_type='pchip',        # PCHIP interpolation for smooth profiles
        ssp_data=ssp_1d,         # Reference 1D profile
        ssp_2d_ranges=ranges_km, # Range points for 2D SSP
        ssp_2d_matrix=ssp_2d_matrix,  # Full 2D SSP matrix
        bathymetry=bathymetry,
        bottom=bottom_rd         # Range-dependent bottom
    )

    print(f"\n✓ Environment created:")
    print(f"    - is_range_dependent: {env.is_range_dependent}")
    print(f"    - has_range_dependent_ssp: {env.has_range_dependent_ssp()}")
    print(f"    - has_range_dependent_bottom: {env.has_range_dependent_bottom()}")

    # ═══════════════════════════════════════════════════════════════════════
    # SOURCE & RECEIVER
    # ═══════════════════════════════════════════════════════════════════════

    source = uacpy.Source(depth=50.0, frequency=100.0)
    receiver = uacpy.Receiver(
        depths=np.linspace(5, 280, 50),
        ranges=np.linspace(100, 20000, 100)
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RUN RAM with Range-Dependent Features
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[Run] RAM (mpiramS) with range-dependent SSP and bottom...")
    print("  Mode: COHERENT_TL (narrowband, range-depth TL grid)")

    try:
        ram = RAM(verbose=True)
        result = ram.run(env, source, receiver)
        print("  RAM TL completed successfully")
        print(f"  TL range: {np.nanmin(result.data):.1f} to {np.nanmax(result.data):.1f} dB")
    except Exception as e:
        print(f"  RAM error: {e}")
        import traceback
        traceback.print_exc()
        result = None

    # =====================================================================
    # BROADBAND MODE (TIME_SERIES)
    # =====================================================================

    print("\n[Run] RAM broadband mode (TIME_SERIES)...")
    print("  Single range, multiple frequencies for time-domain analysis")

    from uacpy.models.base import RunMode

    try:
        ram_bb = RAM(Q=2.0, T=5.0, verbose=True)
        receiver_bb = uacpy.Receiver(
            depths=np.linspace(5, 280, 50),
            ranges=np.array([10000.0])  # Single range at 10 km
        )
        result_bb = ram_bb.run(env, source, receiver_bb, run_mode=RunMode.TIME_SERIES)
        print("  Broadband completed successfully")
        print(f"  Output shape: {result_bb.data.shape} (depth x freq x range)")
        print(f"  Frequencies: {result_bb.frequencies[0]:.1f} to {result_bb.frequencies[-1]:.1f} Hz")
    except Exception as e:
        print(f"  Broadband error: {e}")
        import traceback
        traceback.print_exc()

    # ═══════════════════════════════════════════════════════════════════════
    # COMPARISON: KrakenField with Range-Independent Approximation
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[Comparison] Running KrakenField for comparison...")
    print("  Note: This environment has 2D range-dependent SSP (thermal front)")
    print("  → KrakenField will use range-independent approximation (median SSP)")
    print("  → Range-dependent effects will NOT be captured")
    print("  → For full accuracy, use RAM or Bellhop\n")

    from uacpy.models import KrakenField, Bellhop

    # Run KrakenField
    try:
        krakenfield = KrakenField(verbose=False)
        result_krakenfield = krakenfield.compute_tl(env, source, receiver)
        print("  ✅ KrakenField completed (using range-independent approximation)")
        print(f"  ✓ TL range: {result_krakenfield.data.min():.1f} to {result_krakenfield.data.max():.1f} dB")
    except Exception as e:
        print(f"  ✗ KrakenField: {e}")
        result_krakenfield = None

    # Run Bellhop (supports range-dependent natively)
    print("\n  Running Bellhop (native range-dependent support)...")
    try:
        bellhop = Bellhop(verbose=False)
        result_bellhop = bellhop.compute_tl(env, source, receiver)
        print("  ✅ Bellhop completed (full range-dependent capability)")
        print(f"  ✓ TL range: {result_bellhop.data.min():.1f} to {result_bellhop.data.max():.1f} dB")
    except Exception as e:
        print(f"  ✗ Bellhop: {e}")
        result_bellhop = None

    # Comparisons (use nanmean because RAM masks sub-bottom cells as NaN)
    if result is not None and result_krakenfield is not None:
        diff_kf = np.abs(result.data - result_krakenfield.data)
        print(f"\n  RAM vs KrakenField: Mean diff = {np.nanmean(diff_kf):.1f} dB (range-dependent effects)")

    if result is not None and result_bellhop is not None:
        diff_bh = np.abs(result.data - result_bellhop.data)
        print(f"  RAM vs Bellhop: Mean diff = {np.nanmean(diff_bh):.1f} dB (PE vs ray methods)")

    if result_bellhop is not None and result_krakenfield is not None:
        diff_bk = np.abs(result_bellhop.data - result_krakenfield.data)
        print(f"  Bellhop vs KrakenField: Mean diff = {np.nanmean(diff_bk):.1f} dB")

    # ═══════════════════════════════════════════════════════════════════════
    # VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════════

    print("\nGenerating plots...")

    # Plot 1: Advanced environment overview
    fig1, axes1 = plot_environment_advanced(env, source, receiver)
    plt.savefig('output/example_08_environment.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: example_08_environment.png")

    # Plot 2: 2D SSP detail
    fig2, ax2 = plot_ssp_2d(env)
    plt.savefig('output/example_08_ssp_2d.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: example_08_ssp_2d.png")

    # Plot 3: Bottom properties detail
    fig3, axes3 = plot_bottom_properties(env)
    plt.savefig('output/example_08_bottom.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: example_08_bottom.png")

    # Plot 4: TL field result with NEW features
    if result is not None:
        # NEW: Use default jet_r colormap, auto TL limits, and contours
        fig4, ax4, cbar4 = plot_transmission_loss(
            result, env,
            contours=[70, 85, 100],  # NEW: Add labeled contours
            show_colorbar=True
        )
        ax4.set_title('RAM: Thermal Front with Range-Dependent Bottom\n' +
                     '(NEW: Auto TL limits + contours + jet_r colormap)')

        # Add annotations
        ax4.annotate('Warm Water', xy=(2, 50), fontsize=12, color='white',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
        ax4.annotate('Cold Water', xy=(16, 50), fontsize=12, color='white',
                    bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7))
        ax4.annotate('Soft Mud', xy=(2, 280), fontsize=10, color='white',
                    bbox=dict(boxstyle='round', facecolor='brown', alpha=0.7))
        ax4.annotate('Hard Sand', xy=(16, 280), fontsize=10, color='white',
                    bbox=dict(boxstyle='round', facecolor='gray', alpha=0.7))

        plt.savefig('output/example_08_result.png', dpi=150, bbox_inches='tight')
        print("  ✓ Saved: example_08_result.png")

    # Plot 5: Three-Model Comparison (RAM, Bellhop, KrakenField)
    if result is not None and result_bellhop is not None and result_krakenfield is not None:
        fig5, axes5 = plt.subplots(2, 3, figsize=(18, 12))

        # Row 1: TL Fields
        # RAM
        im1 = axes5[0, 0].pcolormesh(result.ranges/1000, result.depths, result.data,
                                     cmap='jet_r', shading='auto', vmin=60, vmax=120)
        axes5[0, 0].invert_yaxis()
        axes5[0, 0].set_xlabel('Range (km)')
        axes5[0, 0].set_ylabel('Depth (m)')
        axes5[0, 0].set_title('RAM\n(PE, Range-Dependent)', fontweight='bold')
        plt.colorbar(im1, ax=axes5[0, 0], label='TL (dB)')

        # Bellhop
        im2 = axes5[0, 1].pcolormesh(result_bellhop.ranges/1000, result_bellhop.depths,
                                     result_bellhop.data, cmap='jet_r', shading='auto',
                                     vmin=60, vmax=120)
        axes5[0, 1].invert_yaxis()
        axes5[0, 1].set_xlabel('Range (km)')
        axes5[0, 1].set_ylabel('Depth (m)')
        axes5[0, 1].set_title('Bellhop\n(Ray, Range-Dependent)', fontweight='bold')
        plt.colorbar(im2, ax=axes5[0, 1], label='TL (dB)')

        # KrakenField
        im3 = axes5[0, 2].pcolormesh(result_krakenfield.ranges/1000, result_krakenfield.depths,
                                     result_krakenfield.data, cmap='jet_r', shading='auto',
                                     vmin=60, vmax=120)
        axes5[0, 2].invert_yaxis()
        axes5[0, 2].set_xlabel('Range (km)')
        axes5[0, 2].set_ylabel('Depth (m)')
        axes5[0, 2].set_title('KrakenField\n(Modal, Range-Independent)', fontweight='bold')
        plt.colorbar(im3, ax=axes5[0, 2], label='TL (dB)')

        # Row 2: Differences
        # RAM vs Bellhop
        diff_rb = np.abs(result.data - result_bellhop.data)
        im4 = axes5[1, 0].pcolormesh(result.ranges/1000, result.depths, diff_rb,
                                     cmap='hot', shading='auto', vmin=0, vmax=30)
        axes5[1, 0].invert_yaxis()
        axes5[1, 0].set_xlabel('Range (km)')
        axes5[1, 0].set_ylabel('Depth (m)')
        axes5[1, 0].set_title(f'|RAM - Bellhop|\nMean: {np.mean(diff_rb):.1f} dB', fontweight='bold')
        plt.colorbar(im4, ax=axes5[1, 0], label='|ΔTL| (dB)')

        # RAM vs KrakenField
        diff_rk = np.abs(result.data - result_krakenfield.data)
        im5 = axes5[1, 1].pcolormesh(result.ranges/1000, result.depths, diff_rk,
                                     cmap='hot', shading='auto', vmin=0, vmax=30)
        axes5[1, 1].invert_yaxis()
        axes5[1, 1].set_xlabel('Range (km)')
        axes5[1, 1].set_ylabel('Depth (m)')
        axes5[1, 1].set_title(f'|RAM - KrakenField|\nMean: {np.mean(diff_rk):.1f} dB', fontweight='bold')
        plt.colorbar(im5, ax=axes5[1, 1], label='|ΔTL| (dB)')

        # Bellhop vs KrakenField
        diff_bk = np.abs(result_bellhop.data - result_krakenfield.data)
        im6 = axes5[1, 2].pcolormesh(result.ranges/1000, result.depths, diff_bk,
                                     cmap='hot', shading='auto', vmin=0, vmax=30)
        axes5[1, 2].invert_yaxis()
        axes5[1, 2].set_xlabel('Range (km)')
        axes5[1, 2].set_ylabel('Depth (m)')
        axes5[1, 2].set_title(f'|Bellhop - KrakenField|\nMean: {np.mean(diff_bk):.1f} dB', fontweight='bold')
        plt.colorbar(im6, ax=axes5[1, 2], label='|ΔTL| (dB)')

        plt.suptitle('Three-Model Comparison: Range-Dependent Thermal Front\n' +
                    'RAM & Bellhop capture range dependence | KrakenField uses median SSP',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('output/example_08_comparison.png', dpi=150, bbox_inches='tight')
        print("  ✓ Saved: example_08_comparison.png")

    print("\n" + "=" * 70)
    print("RAM ADVANCED EXAMPLE COMPLETE")
    print("=" * 70)
    print("\nFeatures demonstrated:")
    print("  ✓ 2D range-dependent SSP (thermal front)")
    print("  ✓ Range-dependent bottom properties")
    print("  ✓ Sediment transition (mud → sand)")
    print("  ✓ RAM parabolic equation with full range dependency")
    print("\nNEW Plotting features demonstrated:")
    print("  ✓ Auto TL limits (median + 0.75σ, rounded to 10 dB)")
    print("  ✓ Contour overlays at 70, 85, 100 dB")
    print("  ✓ jet_r colormap (blue=good, red=poor) - AT standard")
    print("  ✓ Advanced visualization (SSP heatmap, bottom properties)")

    return 0

if __name__ == "__main__":
    sys.exit(main())
