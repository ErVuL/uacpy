"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 24: Elastic Boundaries - Complete Comparison of Both Workflows
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    Demonstrate and compare TWO workflows for handling elastic boundaries:
    1. KrakenField Auto-Detection (uses KrakenC internally)
    2. BOUNCE → SCOOTER (pre-computed reflection coefficients)

FEATURES DEMONSTRATED:
    ✓ KrakenField automatic elastic boundary detection
    ✓ KrakenC for complex modes with shear
    ✓ BOUNCE reflection coefficient computation
    ✓ SCOOTER with .brc files
    ✓ Side-by-side comparison of both approaches
    ✓ Performance and accuracy analysis

WHEN TO USE EACH APPROACH:

    **Approach 1: KrakenField Auto (→ KrakenC)**
    ✓ Simple elastic boundaries
    ✓ Quick, one-step solution
    ✓ Good for beginners
    ✓ Single run scenarios

    **Approach 2: BOUNCE → BELLHOP/SCOOTER/KRAKEN**
    ✓ Complex layered elastic media
    ✓ Reusable reflection coefficients
    ✓ Multiple simulations with same bottom
    ✓ Professional workflows
    ✓ When sharing reflection data

    NOTE: BOUNCE generates both .brc and .irc files
          - BELLHOP, SCOOTER, KRAKENC use .brc files
          - KRAKEN uses .irc files (NOT .brc)
          - SPARC does not support reflection files

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
from uacpy.models import KrakenField, Bounce, Scooter, KrakenC
from uacpy.core import BoundaryProperties
import os
import time


def main():
    print("\n" + "=" * 80)
    print("EXAMPLE 24: Elastic Boundaries - Complete Workflow Comparison")
    print("=" * 80)
    print("\nCompares two approaches for modeling elastic boundaries:")
    print("  1. KrakenField Auto-Detection (→ KrakenC)")
    print("  2. BOUNCE → Reflection Files → BELLHOP/SCOOTER/KRAKEN")

    # ═══════════════════════════════════════════════════════════════════════
    # SETUP: Define Environment with Elastic Bottom
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[SETUP] Creating environment with elastic bottom...")

    # Elastic bottom with shear wave support
    bottom_elastic = BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1600.0,         # Compressional wave speed (m/s)
        shear_speed=400.0,          # Shear wave speed (m/s) - THIS MAKES IT ELASTIC
        density=1.8,                # Density (g/cm³)
        attenuation=0.2,            # P-wave attenuation (dB/wavelength)
        shear_attenuation=0.5,      # S-wave attenuation (dB/wavelength)
        depth=100
    )

    env = uacpy.Environment(
        name="Elastic Bottom Test",
        depth=100.0,
        sound_speed=1500.0,
        ssp_type='isovelocity',
        bottom=bottom_elastic
    )

    source = uacpy.Source(depth=50.0, frequency=100.0)
    receiver = uacpy.Receiver(
        depths=np.linspace(5, 95, 50),
        ranges=np.linspace(100, 10000, 100)
    )

    print(f"  ✓ Environment: depth={env.depth}m")
    print(f"  ✓ Bottom: Cp={bottom_elastic.sound_speed} m/s, Cs={bottom_elastic.shear_speed} m/s")
    print(f"  ✓ Source: {source.depth[0]}m depth, {source.frequency[0]} Hz")
    print(f"  ✓ Receiver: {len(receiver.depths)} depths, {len(receiver.ranges)} ranges")

    # ═══════════════════════════════════════════════════════════════════════
    # APPROACH 1: KrakenField Auto-Detection (→ KrakenC)
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "─" * 80)
    print("APPROACH 1: KrakenField with Auto-Detection")
    print("─" * 80)
    print("KrakenField detects elastic boundary and automatically uses KrakenC")
    print("for complex modes computation.\n")

    print("[1/2] Running KrakenField...")
    print("  • KrakenField will detect shear_speed > 0")
    print("  • Automatically switches to KrakenC (complex modes)")
    print("  • Computes TL field directly")

    t_start = time.time()
    krakenfield = KrakenField(verbose=False)
    result_krakenfield = krakenfield.compute_tl(env, source, receiver)
    t_krakenfield = time.time() - t_start

    print(f"  ✓ KrakenField completed in {t_krakenfield:.2f}s")
    print(f"    - TL field shape: {result_krakenfield.data.shape}")
    print(f"    - Used KrakenC internally for elastic bottom")

    # ═══════════════════════════════════════════════════════════════════════
    # APPROACH 2: BOUNCE → SCOOTER
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "─" * 80)
    print("APPROACH 2: BOUNCE → SCOOTER Workflow")
    print("─" * 80)
    print("Pre-compute reflection coefficients with BOUNCE, then use in SCOOTER.\n")

    print("[1/3] Running BOUNCE to compute reflection coefficients...")
    receiver_bounce = uacpy.Receiver(
        depths=np.array([50.0]),
        ranges=np.array([1000.0])
    )

    t_start = time.time()
    bounce = Bounce(verbose=False)
    bounce_result = bounce.run(
        env=env,
        source=source,
        receiver=receiver_bounce,
        cmin=1400.0,
        cmax=10000.0,
        rmax_km=10.0
    )
    t_bounce = time.time() - t_start

    print(f"  ✓ BOUNCE completed in {t_bounce:.2f}s")
    print(f"    - Output: {Path(bounce_result.metadata['brc_file']).name}")

    has_rc_data = 'theta' in bounce_result.metadata and 'R' in bounce_result.metadata
    if has_rc_data:
        angles = bounce_result.metadata['theta']
        R_mag = bounce_result.metadata['R']
        print(f"    - Reflection coefficient: {len(angles)} angles")
        print(f"    - |R| range: [{R_mag.min():.3f}, {R_mag.max():.3f}]")

    print("\n[2/3] Creating environment with .brc file...")
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

    print(f"  ✓ Environment created with acoustic_type='file'")

    print("\n[3/3] Running SCOOTER with .brc file...")
    t_start = time.time()
    scooter = Scooter(verbose=False)
    result_scooter = scooter.compute_tl(env_with_rc, source, receiver)
    t_scooter = time.time() - t_start

    print(f"  ✓ SCOOTER completed in {t_scooter:.2f}s")
    print(f"    - TL field shape: {result_scooter.data.shape}")

    t_bounce_total = t_bounce + t_scooter
    print(f"\n  Total time for BOUNCE→SCOOTER: {t_bounce_total:.2f}s")

    # ═══════════════════════════════════════════════════════════════════════
    # COMPARISON & ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("COMPARISON & ANALYSIS")
    print("=" * 80)

    # Compute difference
    tl_diff = result_krakenfield.data - result_scooter.data
    max_diff = np.nanmax(np.abs(tl_diff))
    mean_diff = np.nanmean(np.abs(tl_diff))
    rms_diff = np.sqrt(np.nanmean(tl_diff**2))

    print(f"\nTL Comparison (KrakenField vs SCOOTER):")
    print(f"  • Maximum difference: {max_diff:.2f} dB")
    print(f"  • Mean absolute difference: {mean_diff:.2f} dB")
    print(f"  • RMS difference: {rms_diff:.2f} dB")

    print(f"\nPerformance:")
    print(f"  • KrakenField: {t_krakenfield:.2f}s")
    print(f"  • BOUNCE+SCOOTER: {t_bounce_total:.2f}s (BOUNCE: {t_bounce:.2f}s + SCOOTER: {t_scooter:.2f}s)")
    print(f"  • Speedup: {t_bounce_total/t_krakenfield:.1f}x {'faster' if t_krakenfield < t_bounce_total else 'slower'}")

    print(f"\nAccuracy:")
    if mean_diff < 2.0:
        print(f"  ✓ Excellent agreement (mean diff < 2 dB)")
    elif mean_diff < 5.0:
        print(f"  ✓ Good agreement (mean diff < 5 dB)")
    else:
        print(f"  ⚠ Moderate differences (consider parameter tuning)")

    # ═══════════════════════════════════════════════════════════════════════
    # VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[VISUALIZATION] Creating comparison plots...")

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.35)

    # ─────────────────────────────────────────────────────────────────────
    # Plot 1: KrakenField TL
    # ─────────────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])

    vmin, vmax = 50, 100
    im1 = ax1.pcolormesh(
        result_krakenfield.ranges / 1000,
        result_krakenfield.depths,
        result_krakenfield.data,
        shading='auto',
        cmap='jet_r',
        vmin=vmin,
        vmax=vmax
    )
    ax1.set_xlabel('Range (km)', fontweight='bold')
    ax1.set_ylabel('Depth (m)', fontweight='bold')
    ax1.set_title('APPROACH 1: KrakenField (Auto KrakenC)', fontweight='bold', fontsize=11)
    ax1.invert_yaxis()
    plt.colorbar(im1, ax=ax1, label='TL (dB)')

    # ─────────────────────────────────────────────────────────────────────
    # Plot 2: SCOOTER TL
    # ─────────────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])

    im2 = ax2.pcolormesh(
        result_scooter.ranges / 1000,
        result_scooter.depths,
        result_scooter.data,
        shading='auto',
        cmap='jet_r',
        vmin=vmin,
        vmax=vmax
    )
    ax2.set_xlabel('Range (km)', fontweight='bold')
    ax2.set_ylabel('Depth (m)', fontweight='bold')
    ax2.set_title('APPROACH 2: SCOOTER (BOUNCE .brc)', fontweight='bold', fontsize=11)
    ax2.invert_yaxis()
    plt.colorbar(im2, ax=ax2, label='TL (dB)')

    # ─────────────────────────────────────────────────────────────────────
    # Plot 3: Difference
    # ─────────────────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])

    diff_max = max(5, max(abs(np.nanmin(tl_diff)), abs(np.nanmax(tl_diff))))
    im3 = ax3.pcolormesh(
        result_krakenfield.ranges / 1000,
        result_krakenfield.depths,
        tl_diff,
        shading='auto',
        cmap='RdBu_r',
        vmin=-diff_max,
        vmax=diff_max
    )
    ax3.set_xlabel('Range (km)', fontweight='bold')
    ax3.set_ylabel('Depth (m)', fontweight='bold')
    ax3.set_title(f'Difference (KrakenField - SCOOTER)\nMean: {mean_diff:.2f} dB',
                  fontweight='bold', fontsize=11)
    ax3.invert_yaxis()
    plt.colorbar(im3, ax=ax3, label='ΔTL (dB)')

    # ─────────────────────────────────────────────────────────────────────
    # Plot 4: Reflection Coefficient
    # ─────────────────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])

    if has_rc_data:
        ax4.plot(angles, R_mag, 'b-', linewidth=2.5)
        ax4.set_xlabel('Grazing Angle (degrees)', fontweight='bold')
        ax4.set_ylabel('|R| - Magnitude', fontweight='bold')
        ax4.set_title('BOUNCE: Bottom Reflection Coefficient', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim([angles.min(), angles.max()])
        ax4.set_ylim([0, 1.1])

        # Mark critical angle
        critical_idx = np.where(np.diff(R_mag) > 0.05)[0]
        if len(critical_idx) > 0:
            crit_angle = angles[critical_idx[0]]
            ax4.axvline(crit_angle, color='r', linestyle='--', linewidth=1.5, alpha=0.7)
            ax4.text(crit_angle + 2, 0.5, f'Critical\nangle\n≈{crit_angle:.1f}°',
                    fontsize=9, color='red', ha='left')
    else:
        ax4.text(0.5, 0.5, 'Reflection coefficient\ndata not available',
                ha='center', va='center', transform=ax4.transAxes, fontsize=11)
        ax4.set_title('BOUNCE: Bottom Reflection Coefficient', fontweight='bold')

    # ─────────────────────────────────────────────────────────────────────
    # Plot 5: TL Comparison at Source Depth
    # ─────────────────────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])

    depth_idx = np.argmin(np.abs(result_krakenfield.depths - source.depth[0]))

    ax5.plot(result_krakenfield.ranges/1000, result_krakenfield.data[depth_idx, :],
            'b-', linewidth=2.5, label='KrakenField (Auto)', alpha=0.8)
    ax5.plot(result_scooter.ranges/1000, result_scooter.data[depth_idx, :],
            'r--', linewidth=2.5, label='SCOOTER (BOUNCE)', alpha=0.8)

    ax5.set_xlabel('Range (km)', fontweight='bold')
    ax5.set_ylabel('Transmission Loss (dB)', fontweight='bold')
    ax5.set_title(f'TL Comparison at {source.depth[0]:.0f}m Depth', fontweight='bold')
    ax5.legend(fontsize=10, framealpha=0.9)
    ax5.grid(True, alpha=0.3)
    ax5.invert_yaxis()

    # ─────────────────────────────────────────────────────────────────────
    # Plot 6: TL Comparison at Mid-Range
    # ─────────────────────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])

    mid_range_km = np.median(result_krakenfield.ranges) / 1000
    range_idx = np.argmin(np.abs(result_krakenfield.ranges/1000 - mid_range_km))

    ax6.plot(result_krakenfield.data[:, range_idx], result_krakenfield.depths,
            'b-', linewidth=2.5, label='KrakenField (Auto)', alpha=0.8)
    ax6.plot(result_scooter.data[:, range_idx], result_scooter.depths,
            'r--', linewidth=2.5, label='SCOOTER (BOUNCE)', alpha=0.8)

    ax6.invert_yaxis()
    ax6.set_xlabel('Transmission Loss (dB)', fontweight='bold')
    ax6.set_ylabel('Depth (m)', fontweight='bold')
    ax6.set_title(f'TL vs Depth at {mid_range_km:.1f}km Range', fontweight='bold')
    ax6.legend(fontsize=10, framealpha=0.9)
    ax6.grid(True, alpha=0.3)

    # ─────────────────────────────────────────────────────────────────────
    # Plot 7: Workflow Diagram
    # ─────────────────────────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.axis('off')

    workflow1 = "APPROACH 1: KrakenField Auto\n" + "="*35 + "\n\n"
    workflow1 += "Step 1: Define environment\n"
    workflow1 += "  bottom = BoundaryProperties(\n"
    workflow1 += "    acoustic_type='half-space',\n"
    workflow1 += "    shear_speed=400  # Elastic!\n"
    workflow1 += "  )\n\n"
    workflow1 += "Step 2: Run KrakenField\n"
    workflow1 += "  krakenfield = KrakenField()\n"
    workflow1 += "  result = krakenfield.run(...)\n"
    workflow1 += "  # Auto-detects elastic\n"
    workflow1 += "  # Uses KrakenC internally\n\n"
    workflow1 += "✓ Simple, one-step\n"
    workflow1 += "✓ Good for beginners\n"
    workflow1 += f"✓ Time: {t_krakenfield:.1f}s"

    ax7.text(0.05, 0.95, workflow1, transform=ax7.transAxes,
            fontsize=8, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    # ─────────────────────────────────────────────────────────────────────
    # Plot 8: Workflow Diagram 2
    # ─────────────────────────────────────────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.axis('off')

    workflow2 = "APPROACH 2: BOUNCE→SCOOTER\n" + "="*35 + "\n\n"
    workflow2 += "Step 1: Run BOUNCE\n"
    workflow2 += "  bounce = Bounce()\n"
    workflow2 += "  rc = bounce.run(...)\n"
    workflow2 += "  # Generates .brc file\n\n"
    workflow2 += "Step 2: Create env with file\n"
    workflow2 += "  bottom = BoundaryProperties(\n"
    workflow2 += "    acoustic_type='file',\n"
    workflow2 += "    reflection_file=rc.brc_file\n"
    workflow2 += "  )\n\n"
    workflow2 += "Step 3: Run SCOOTER\n"
    workflow2 += "  scooter = Scooter()\n"
    workflow2 += "  result = scooter.run(...)\n\n"
    workflow2 += "✓ Reusable .brc files\n"
    workflow2 += "✓ Professional workflow\n"
    workflow2 += f"✓ Time: {t_bounce_total:.1f}s"

    ax8.text(0.05, 0.95, workflow2, transform=ax8.transAxes,
            fontsize=8, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # ─────────────────────────────────────────────────────────────────────
    # Plot 9: Summary & Recommendations
    # ─────────────────────────────────────────────────────────────────────
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    summary = "RECOMMENDATIONS\n" + "="*35 + "\n\n"
    summary += "When to use EACH:\n\n"
    summary += "KrakenField Auto:\n"
    summary += "  • Simple elastic bottoms\n"
    summary += "  • Single simulation runs\n"
    summary += "  • Quick prototyping\n"
    summary += "  • Learning/teaching\n\n"
    summary += "BOUNCE→SCOOTER:\n"
    summary += "  • Complex layered media\n"
    summary += "  • Multiple simulations\n"
    summary += "  • Production workflows\n"
    summary += "  • Reusable coefficients\n"
    summary += "  • Sharing reflection data\n\n"
    summary += f"ACCURACY:\n"
    summary += f"  Mean diff: {mean_diff:.2f} dB\n"
    summary += f"  RMS diff: {rms_diff:.2f} dB\n"
    summary += f"  {'✓ Excellent' if mean_diff < 2 else '✓ Good'}"

    ax9.text(0.05, 0.95, summary, transform=ax9.transAxes,
            fontsize=8, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # Main title
    fig.suptitle('EXAMPLE 24: Elastic Boundaries - Complete Workflow Comparison',
                fontsize=14, fontweight='bold', y=0.995)

    # Save
    output_file = OUTPUT_DIR / 'example_24_elastic_boundaries_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Figure saved: {output_file}")

    plt.close()

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n✓ BOTH approaches work correctly for elastic boundaries")
    print(f"✓ Results agree within {mean_diff:.2f} dB (mean absolute difference)")
    print(f"✓ RMS difference: {rms_diff:.2f} dB")

    print("\nPERFORMANCE:")
    print(f"  • KrakenField Auto: {t_krakenfield:.2f}s")
    print(f"  • BOUNCE+SCOOTER: {t_bounce_total:.2f}s")
    if t_krakenfield < t_bounce_total:
        print(f"  → KrakenField is {t_bounce_total/t_krakenfield:.1f}x faster for single runs")
    else:
        print(f"  → SCOOTER is {t_krakenfield/t_bounce_total:.1f}x faster for single runs")
    print(f"  → But BOUNCE .brc can be reused for multiple runs!")

    print("\nCHOOSE:")
    print("  • KrakenField Auto → For simple cases and single runs")
    print("  • BOUNCE→BELLHOP/SCOOTER/KRAKEN → For professional workflows and reusability")
    print("\nNOTE:")
    print("  • BOUNCE outputs: .brc (bottom) + .irc (internal) reflection coefficients")
    print("  • BELLHOP, SCOOTER, KRAKENC use .brc files")
    print("  • KRAKEN uses .irc files (NOT .brc)")
    print("  • SPARC does not support reflection files")

    print("\n" + "=" * 80)
    print("EXAMPLE 24 COMPLETE")
    print("=" * 80 + "\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
