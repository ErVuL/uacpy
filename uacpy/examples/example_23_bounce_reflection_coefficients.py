"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 23: BOUNCE Reflection Coefficients & Integration
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    Demonstrate BOUNCE reflection coefficient computation and integration with
    SCOOTER for accurate elastic bottom modeling. This is the PROFESSIONAL
    workflow for complex layered elastic media and reusable reflection data.

FEATURES DEMONSTRATED:
    ✓ BOUNCE reflection coefficient computation
    ✓ Elastic bottom properties (compressional + shear)
    ✓ BOUNCE → SCOOTER workflow (using .brc files)
    ✓ Comparison: Direct elastic vs file-based reflection
    ✓ Reflection coefficient visualization
    ✓ KRAKENC with .brc files (experimental)

WORKFLOW:
    1. Define elastic bottom with shear properties
    2. Run BOUNCE to compute reflection coefficients (.brc file)
    3. Use .brc file in SCOOTER (RECOMMENDED - fully supported)
    4. Use .brc file in KRAKENC (experimental - may have issues)
    5. Compare results

IMPORTANT NOTES - Model Support for BOUNCE Files:
    • BELLHOP: ✅ Fully supports .brc files
    • SCOOTER: ✅ Fully supports .brc files (RECOMMENDED for modes)
    • KRAKENC: ⚠️ Experimental support for .brc files
    • KRAKEN: ✅ Supports .irc files (NOT .brc, uses internal reflection coefficient)
    • SPARC: ❌ Does NOT support reflection coefficient files

WHEN TO USE THIS APPROACH:
    ✓ Complex layered elastic bottoms
    ✓ Multiple simulations with same bottom
    ✓ Reusable reflection coefficient data
    ✓ Professional workflows
    ✓ Sharing reflection data between projects

    For simple elastic boundaries, see Example 24 for the simpler
    KrakenField auto-detection approach.

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
from uacpy.models import Bounce, Scooter
from uacpy.core import BoundaryProperties


def main():
    print("\n" + "=" * 80)
    print("EXAMPLE 23: BOUNCE Reflection Coefficients & Integration")
    print("=" * 80)
    print("\nDemonstrates computing reflection coefficients and using them in other models")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: Define Environment with Elastic Bottom
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[Step 1/6] Defining elastic bottom environment...")

    # Elastic bottom with shear wave properties
    bottom_elastic = BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1600.0,         # Compressional wave speed (m/s)
        shear_speed=400.0,          # Shear wave speed (m/s)
        density=1.8,                # Density (g/cm³)
        attenuation=0.2,            # P-wave attenuation (dB/wavelength)
        shear_attenuation=0.5,      # S-wave attenuation (dB/wavelength)
        depth=100
    )

    env = uacpy.Environment(
        name="Elastic Bottom Environment",
        depth=100.0,
        sound_speed=1500.0,
        ssp_type='isovelocity',
        bottom=bottom_elastic
    )

    print(f"  ✓ Environment created:")
    print(f"    - Water depth: {env.depth}m")
    print(f"    - Water sound speed: {env.sound_speed} m/s")
    print(f"    - Bottom type: Elastic half-space with shear")
    print(f"    - Compressional speed: {bottom_elastic.sound_speed} m/s")
    print(f"    - Shear speed: {bottom_elastic.shear_speed} m/s")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: Run BOUNCE to Compute Reflection Coefficients
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[Step 2/6] Running BOUNCE reflection coefficient computation...")

    source = uacpy.Source(depth=50.0, frequency=100.0)
    receiver_bounce = uacpy.Receiver(
        depths=np.array([50.0]),
        ranges=np.array([1000.0])
    )

    bounce = Bounce(verbose=False)

    print("  • Computing reflection coefficients...")
    print("    - Phase velocity range: 1400 - 10000 m/s")
    print("    - Maximum range: 10 km (for angular sampling)")

    bounce_result = bounce.run(
        env=env,
        source=source,
        receiver=receiver_bounce,
        cmin=1400.0,    # Minimum phase velocity (m/s)
        cmax=10000.0,   # Maximum phase velocity (m/s)
        rmax_km=10.0    # Max range for angular sampling (km)
    )

    print(f"  ✓ BOUNCE completed successfully")
    print(f"    - Output: {Path(bounce_result.metadata['brc_file']).name}")

    if 'irc_file' in bounce_result.metadata:
        print(f"    - IRC file: {Path(bounce_result.metadata['irc_file']).name}")

    # Extract reflection coefficient data
    has_rc_data = 'theta' in bounce_result.metadata and 'R' in bounce_result.metadata

    if has_rc_data:
        angles = bounce_result.metadata['theta']
        R_mag = bounce_result.metadata['R']
        print(f"    - Computed at {len(angles)} angles ({angles.min():.1f}° to {angles.max():.1f}°)")
        print(f"    - |R| range: {R_mag.min():.3f} to {R_mag.max():.3f}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: Use Reflection Coefficients in SCOOTER
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[Step 3/6] Running SCOOTER with BOUNCE reflection coefficients...")

    # Create environment using reflection coefficient file
    # Extract parameters from BOUNCE result
    bottom_with_file = BoundaryProperties(
        acoustic_type='file',
        reflection_file=bounce_result.metadata['brc_file'],
        depth=100,
        sound_speed=1600.0,
        density=1.8,
        attenuation=0.2,
        reflection_cmin=bounce_result.metadata['cmin'],
        reflection_cmax=bounce_result.metadata['cmax'],
        reflection_rmax_km=bounce_result.metadata['rmax_km']
    )

    env_with_rc = uacpy.Environment(
        name="SCOOTER with BOUNCE RC",
        depth=100.0,
        sound_speed=1500.0,
        ssp_type='isovelocity',
        bottom=bottom_with_file
    )

    # Receiver grid for TL computation
    receiver = uacpy.Receiver(
        depths=np.linspace(5, 95, 50),
        ranges=np.linspace(100, 10000, 100)
    )

    scooter = Scooter(verbose=False)

    print("  • Running SCOOTER with .brc file...")
    tl_with_file = scooter.compute_tl(env_with_rc, source, receiver)

    print(f"  ✓ SCOOTER completed")
    print(f"    - TL field: {tl_with_file.data.shape}")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: Also test with KrakenC (Complex Modes) - EXPERIMENTAL
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[Step 4/6] Computing modes with KRAKENC using reflection coefficients...")
    print("  NOTE: .brc file support in KRAKENC is EXPERIMENTAL and may fail")

    from uacpy.models import KrakenC

    krakenc = KrakenC(verbose=False)

    try:
        print("  • Running KRAKENC with .brc file...")
        modes_result = krakenc.run(env_with_rc, source, receiver)

        print(f"  ✓ KRAKENC completed")
        print(f"    - Computed {len(modes_result.metadata['k'])} modes")
        print(f"    - Mode shapes: {modes_result.metadata['phi'].shape}")
        krakenc_worked = True
    except Exception as e:
        print(f"  ✗ KRAKENC failed (expected - .brc support is experimental)")
        print(f"    Error: {str(e)[:100]}...")
        print(f"    → This is why SCOOTER is RECOMMENDED for .brc files")
        krakenc_worked = False

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: Compare with Direct Elastic Bottom
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[Step 5/6] Running SCOOTER with direct elastic bottom (for comparison)...")

    env_direct = uacpy.Environment(
        name="SCOOTER Direct Elastic",
        depth=100.0,
        sound_speed=1500.0,
        ssp_type='isovelocity',
        bottom=bottom_elastic
    )

    print("  • Running SCOOTER with direct elastic properties...")
    tl_direct = scooter.compute_tl(env_direct, source, receiver)

    print(f"  ✓ SCOOTER completed")

    # Compute difference
    tl_diff = tl_with_file.data - tl_direct.data
    max_diff = np.nanmax(np.abs(tl_diff))
    mean_diff = np.nanmean(np.abs(tl_diff))

    print(f"\n  TL Comparison (SCOOTER with .brc vs direct elastic):")
    print(f"    - Maximum difference: {max_diff:.2f} dB")
    print(f"    - Mean absolute difference: {mean_diff:.2f} dB")
    print(f"    - Note: Should be very similar (same bottom properties)")

    # ═══════════════════════════════════════════════════════════════════════
    # STEP 6: Visualization
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[Step 6/6] Creating visualizations...")

    # Create figure with 4 subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # ─────────────────────────────────────────────────────────────────────
    # Plot 1: Reflection Coefficient
    # ─────────────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])

    if has_rc_data:
        angles_plot = bounce_result.metadata['theta']
        R_mag_plot = bounce_result.metadata['R']

        ax1.plot(angles_plot, R_mag_plot, 'b-', linewidth=2.5)
        ax1.set_xlabel('Grazing Angle (degrees)', fontweight='bold', fontsize=11)
        ax1.set_ylabel('|R| - Magnitude', fontweight='bold', fontsize=11)
        ax1.set_title('BOUNCE: Bottom Reflection Coefficient', fontweight='bold', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([angles_plot.min(), angles_plot.max()])
        ax1.set_ylim([0, 1.1])

        # Add annotation for critical angle (if visible)
        critical_idx = np.where(np.diff(R_mag_plot) > 0.05)[0]
        if len(critical_idx) > 0:
            crit_angle = angles_plot[critical_idx[0]]
            ax1.axvline(crit_angle, color='r', linestyle='--', linewidth=1.5, alpha=0.7)
            ax1.text(crit_angle + 2, 0.5, f'Critical angle\n≈{crit_angle:.1f}°',
                    fontsize=9, color='red')
    else:
        ax1.text(0.5, 0.5, 'Reflection coefficient\ndata not available',
                ha='center', va='center', transform=ax1.transAxes, fontsize=11)
        ax1.set_title('BOUNCE: Bottom Reflection Coefficient', fontweight='bold', fontsize=12)

    # ─────────────────────────────────────────────────────────────────────
    # Plot 2: TL with BOUNCE file
    # ─────────────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])

    im2 = ax2.pcolormesh(
        tl_with_file.ranges / 1000,
        tl_with_file.depths,
        tl_with_file.data,
        shading='auto',
        cmap='jet_r',
        vmin=50,
        vmax=100
    )
    ax2.set_xlabel('Range (km)', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Depth (m)', fontweight='bold', fontsize=11)
    ax2.set_title('SCOOTER TL (with BOUNCE .brc file)', fontweight='bold', fontsize=12)
    ax2.invert_yaxis()
    plt.colorbar(im2, ax=ax2, label='TL (dB)')

    # ─────────────────────────────────────────────────────────────────────
    # Plot 3: TL with direct elastic
    # ─────────────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])

    im3 = ax3.pcolormesh(
        tl_direct.ranges / 1000,
        tl_direct.depths,
        tl_direct.data,
        shading='auto',
        cmap='jet_r',
        vmin=50,
        vmax=100
    )
    ax3.set_xlabel('Range (km)', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Depth (m)', fontweight='bold', fontsize=11)
    ax3.set_title('SCOOTER TL (direct elastic bottom)', fontweight='bold', fontsize=12)
    ax3.invert_yaxis()
    plt.colorbar(im3, ax=ax3, label='TL (dB)')

    # ─────────────────────────────────────────────────────────────────────
    # Plot 4: TL difference
    # ─────────────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])

    diff_max = max(abs(np.nanmin(tl_diff)), abs(np.nanmax(tl_diff)))
    im4 = ax4.pcolormesh(
        tl_with_file.ranges / 1000,
        tl_with_file.depths,
        tl_diff,
        shading='auto',
        cmap='RdBu_r',
        vmin=-diff_max if diff_max > 0 else -1,
        vmax=diff_max if diff_max > 0 else 1
    )
    ax4.set_xlabel('Range (km)', fontweight='bold', fontsize=11)
    ax4.set_ylabel('Depth (m)', fontweight='bold', fontsize=11)
    ax4.set_title('TL Difference (file - direct)', fontweight='bold', fontsize=12)
    ax4.invert_yaxis()
    plt.colorbar(im4, ax=ax4, label='ΔTL (dB)')

    # ─────────────────────────────────────────────────────────────────────
    # Plot 5: TL comparison at source depth
    # ─────────────────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])

    depth_idx = np.argmin(np.abs(tl_with_file.depths - source.depth[0]))

    ax5.plot(tl_with_file.ranges/1000, tl_with_file.data[depth_idx, :],
            'b-', linewidth=2.5, label='With BOUNCE file', alpha=0.8)
    ax5.plot(tl_direct.ranges/1000, tl_direct.data[depth_idx, :],
            'r--', linewidth=2.5, label='Direct elastic', alpha=0.8)

    ax5.set_xlabel('Range (km)', fontweight='bold', fontsize=11)
    ax5.set_ylabel('Transmission Loss (dB)', fontweight='bold', fontsize=11)
    ax5.set_title(f'TL Comparison at {source.depth[0]:.0f}m Depth', fontweight='bold', fontsize=12)
    ax5.legend(fontsize=10, framealpha=0.9)
    ax5.grid(True, alpha=0.3)
    ax5.invert_yaxis()

    # ─────────────────────────────────────────────────────────────────────
    # Plot 6: Summary information
    # ─────────────────────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    summary = "BOUNCE WORKFLOW SUMMARY\n" + "="*50 + "\n\n"

    summary += "STEP 1: Compute Reflection Coefficients\n"
    summary += f"  • Model: BOUNCE\n"
    summary += f"  • Bottom: Elastic (Cp={bottom_elastic.sound_speed} m/s,\n"
    summary += f"           Cs={bottom_elastic.shear_speed} m/s)\n"
    summary += f"  • Output: .brc file (bottom reflection)\n\n"

    summary += "STEP 2: Use in Other Models\n"
    summary += f"  • acoustic_type='file'\n"
    summary += f"  • .brc file: {Path(bounce_result.metadata['brc_file']).name}\n"
    summary += f"  • .irc file: {Path(bounce_result.metadata.get('irc_file', 'N/A')).name if 'irc_file' in bounce_result.metadata else 'N/A'}\n"
    summary += f"  • BELLHOP: ✅ Uses .brc\n"
    summary += f"  • SCOOTER: ✅ Uses .brc\n"
    summary += f"  • KRAKENC: ⚠️ Uses .brc (experimental)\n"
    summary += f"  • KRAKEN: ✅ Uses .irc (NOT .brc)\n"
    summary += f"  • SPARC: ❌ Not supported\n\n"

    summary += "RESULTS:\n"
    summary += f"  • Max TL difference: {max_diff:.2f} dB\n"
    summary += f"  • Mean abs difference: {mean_diff:.2f} dB\n"
    summary += f"  • Small differences are expected due to\n"
    summary += f"    numerical implementation variations\n\n"

    summary += "WHEN TO USE BOUNCE:\n"
    summary += "  ✓ Complex layered bottoms\n"
    summary += "  ✓ Frequency-dependent reflection\n"
    summary += "  ✓ Pre-computing for multiple simulations\n"
    summary += "  ✓ Sharing reflection data between models\n"

    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Add main title
    fig.suptitle('EXAMPLE 23: BOUNCE Reflection Coefficients & Integration with SCOOTER/KRAKENC',
                fontsize=14, fontweight='bold', y=0.995)

    # Save
    output_file = OUTPUT_DIR / 'example_23_bounce_reflection_coefficients.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Figure saved: {output_file}")

    plt.show()

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
    print("\nKEY TAKEAWAYS:")
    print("  • BOUNCE computes accurate reflection coefficients for layered media")
    print("  • BOUNCE outputs: .brc (bottom reflection) + .irc (internal reflection)")
    print("  • BELLHOP: ✅ Uses .brc files")
    print("  • SCOOTER: ✅ Uses .brc files")
    print("  • KRAKENC: ⚠️ Uses .brc files (experimental)")
    print("  • KRAKEN: ✅ Uses .irc files (NOT .brc)")
    print("  • SPARC: ❌ No reflection file support")
    print("  • Essential for complex bottom modeling and professional workflows")
    print("\nRECOMMENDATIONS:")
    print("  • For simple elastic bottoms → Use KrakenField (Example 24)")
    print("  • For complex layered media → Use BOUNCE→SCOOTER (this example)")
    print("  • For reusable reflection data → Use BOUNCE→SCOOTER")
    print("=" * 80)


if __name__ == '__main__':
    main()
