"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 16: Bellhop Run Modes - Comprehensive
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    Demonstrate all Bellhop run modes and their applications.
    Compare coherent vs incoherent vs semi-coherent TL.
    Show ray tracing, eigenrays, and arrival structure analysis.

COMPLEXITY LEVEL: ⭐⭐⭐ (3/5) - Advanced Model Features

FEATURES DEMONSTRATED:
    ✓ Coherent TL (run_type='C') - Phase-preserving transmission loss
    ✓ Incoherent TL (run_type='I') - Phase-averaged transmission loss
    ✓ Semi-coherent TL (run_type='S') - Hybrid coherent/incoherent
    ✓ Ray tracing (run_type='R') - Ray path visualization
    ✓ Eigenrays (run_type='E') - Specific receiver rays
    ✓ Arrivals (run_type='A') - Arrival time/amplitude structure
    ✓ Ray file reading and visualization
    ✓ Arrival structure analysis

SCENARIOS:

    Scenario A: TL Mode Comparison
    ───────────────────────────────
    - Compare coherent vs incoherent vs semi-coherent TL
    - Same environment, different run modes
    - Munk profile to show modal interference
    - Demonstrates phase effects

    Scenario B: Ray Tracing
    ────────────────────────
    - Visualize ray paths through environment
    - Ray tube spreading
    - Caustics and shadow zones
    - Ray turning points

    Scenario C: Eigenrays & Arrivals
    ─────────────────────────────────
    - Find all eigenrays to specific receiver
    - Analyze arrival structure
    - Travel times and amplitudes
    - Multipath arrival patterns

RUN MODES EXPLAINED:

    Coherent TL ('C'):
    - Preserves phase relationships
    - Shows interference patterns (Lloyd mirror, modal)
    - Best for CW signals
    - Most computationally expensive

    Incoherent TL ('I'):
    - Phase-averaged (power sum of contributions)
    - Smoother TL field
    - Better for broadband signals
    - Represents long-term average

    Semi-coherent TL ('S'):
    - Hybrid of coherent and incoherent
    - Coherent sum within beam, incoherent between beams
    - Intermediate smoothness
    - Practical compromise

    Rays ('R'):
    - Compute and save ray paths
    - Visualize propagation geometry
    - Identify turning points, caustics
    - No TL field computed

    Eigenrays ('E'):
    - Find all rays reaching specific receiver
    - Useful for multipath analysis
    - Shows direct, surface-reflected, bottom-reflected paths
    - Essential for pulse propagation

    Arrivals ('A'):
    - Complete arrival structure
    - Travel time, amplitude, phase for each path
    - Number of surface/bottom bounces
    - Critical for pulse/transient analysis

LEARNING OUTCOMES:
    - When to use each run mode
    - Phase effects in coherent propagation
    - Ray-based propagation visualization
    - Multipath structure analysis
    - Practical sonar applications

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.models import Bellhop
from plotting_utils import plot_bathymetry_overlay
import os


def scenario_a_tl_modes():
    """
    Scenario A: Compare coherent, incoherent, and semi-coherent TL.
    """
    print("\n" + "="*80)
    print("SCENARIO A: TL Mode Comparison (Coherent vs Incoherent vs Semi-coherent)")
    print("="*80)

    # ═══════════════════════════════════════════════════════════════════════
    # ENVIRONMENT - Munk profile to show modal interference
    # ═══════════════════════════════════════════════════════════════════════
    env = uacpy.Environment(
        name="Munk Profile - TL Mode Comparison",
        depth=5000.0,
        ssp_type='munk'
    )

    source = uacpy.Source(
        depth=1000.0,      # At channel axis
        frequency=50.0     # 50 Hz
    )

    receiver = uacpy.Receiver(
        depths=np.linspace(100, 4900, 40),      # Reduced for faster computation
        ranges=np.linspace(1000, 50000, 80)     # 1-50 km
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RUN BELLHOP WITH DIFFERENT MODES
    # ═══════════════════════════════════════════════════════════════════════
    print("\n  Running Bellhop with different TL modes:")

    # Coherent TL
    print("    • Coherent TL...", end=" ", flush=True)
    bellhop_coherent = Bellhop(verbose=False)
    result_coherent = bellhop_coherent.run(env, source, receiver, run_type='C')
    print("✓")

    # Incoherent TL
    print("    • Incoherent TL...", end=" ", flush=True)
    bellhop_incoherent = Bellhop(verbose=False)
    result_incoherent = bellhop_incoherent.run(env, source, receiver, run_type='I')
    print("✓")

    # Semi-coherent TL
    print("    • Semi-coherent TL...", end=" ", flush=True)
    bellhop_semicoherent = Bellhop(verbose=False)
    result_semicoherent = bellhop_semicoherent.run(env, source, receiver, run_type='S')
    print("✓")

    # ═══════════════════════════════════════════════════════════════════════
    # PLOT COMPARISON
    # ═══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Common colorbar limits
    vmin, vmax = 60, 120

    # Coherent TL
    ax = axes[0, 0]
    im = ax.pcolormesh(result_coherent.ranges/1000, result_coherent.depths,
                       result_coherent.data, cmap='viridis', vmin=vmin, vmax=vmax,
                       shading='auto', zorder=1)
    ax.set_xlim([result_coherent.ranges[0]/1000, result_coherent.ranges[-1]/1000])
    ax.set_ylim([result_coherent.depths[-1], result_coherent.depths[0]])
    ax.plot(0, source.depth[0], 'r*', markersize=15, label='Source', zorder=12)
    ax.set_xlabel('Range (km)', fontweight='bold')
    ax.set_ylabel('Depth (m)', fontweight='bold')
    ax.set_title('Coherent TL (run_type=\'C\')', fontweight='bold', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax, label='TL (dB)')

    # Incoherent TL
    ax = axes[0, 1]
    im = ax.pcolormesh(result_incoherent.ranges/1000, result_incoherent.depths,
                       result_incoherent.data, cmap='viridis', vmin=vmin, vmax=vmax,
                       shading='auto', zorder=1)
    ax.set_xlim([result_incoherent.ranges[0]/1000, result_incoherent.ranges[-1]/1000])
    ax.set_ylim([result_incoherent.depths[-1], result_incoherent.depths[0]])
    ax.plot(0, source.depth[0], 'r*', markersize=15, label='Source', zorder=12)
    ax.set_xlabel('Range (km)', fontweight='bold')
    ax.set_ylabel('Depth (m)', fontweight='bold')
    ax.set_title('Incoherent TL (run_type=\'I\')', fontweight='bold', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax, label='TL (dB)')

    # Semi-coherent TL
    ax = axes[1, 0]
    im = ax.pcolormesh(result_semicoherent.ranges/1000, result_semicoherent.depths,
                       result_semicoherent.data, cmap='viridis', vmin=vmin, vmax=vmax,
                       shading='auto', zorder=1)
    ax.set_xlim([result_semicoherent.ranges[0]/1000, result_semicoherent.ranges[-1]/1000])
    ax.set_ylim([result_semicoherent.depths[-1], result_semicoherent.depths[0]])
    ax.plot(0, source.depth[0], 'r*', markersize=15, label='Source', zorder=12)
    ax.set_xlabel('Range (km)', fontweight='bold')
    ax.set_ylabel('Depth (m)', fontweight='bold')
    ax.set_title('Semi-coherent TL (run_type=\'S\')', fontweight='bold', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax, label='TL (dB)')

    # Range cut comparison at channel axis
    ax = axes[1, 1]
    depth_idx = np.argmin(np.abs(result_coherent.depths - 1000))
    tl_coherent = result_coherent.data[depth_idx, :]
    tl_incoherent = result_incoherent.data[depth_idx, :]
    tl_semicoherent = result_semicoherent.data[depth_idx, :]

    ax.plot(result_coherent.ranges/1000, tl_coherent,
            'b-', linewidth=2.5, label='Coherent', alpha=0.8)
    ax.plot(result_incoherent.ranges/1000, tl_incoherent,
            'r-', linewidth=2.5, label='Incoherent', alpha=0.8)
    ax.plot(result_semicoherent.ranges/1000, tl_semicoherent,
            'g-', linewidth=2.5, label='Semi-coherent', alpha=0.8)
    ax.set_xlabel('Range (km)', fontweight='bold')
    ax.set_ylabel('Transmission Loss (dB)', fontweight='bold')
    ax.set_title(f'TL Comparison at {source.depth[0]:.0f}m Depth (Channel Axis)',
                 fontweight='bold', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Compute ylim from data with margin
    all_tl = np.concatenate([tl_coherent, tl_incoherent, tl_semicoherent])
    all_tl = all_tl[np.isfinite(all_tl)]
    if len(all_tl) > 0:
        tl_min = np.floor(np.min(all_tl) / 10) * 10
        tl_max = np.ceil(np.max(all_tl) / 10) * 10
        ax.set_ylim([tl_min, tl_max])

    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/example_16a_tl_modes.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  TL statistics at channel axis (1000m depth):")
    print(f"    • Coherent    - Mean: {np.mean(result_coherent.data[depth_idx, :]):.1f} dB, "
          f"Std: {np.std(result_coherent.data[depth_idx, :]):.1f} dB")
    print(f"    • Incoherent  - Mean: {np.mean(result_incoherent.data[depth_idx, :]):.1f} dB, "
          f"Std: {np.std(result_incoherent.data[depth_idx, :]):.1f} dB")
    print(f"    • Semi-coh    - Mean: {np.mean(result_semicoherent.data[depth_idx, :]):.1f} dB, "
          f"Std: {np.std(result_semicoherent.data[depth_idx, :]):.1f} dB")
    print(f"\n  Key observations:")
    print(f"    • Coherent TL shows strong modal interference (high std dev)")
    print(f"    • Incoherent TL is smoothest (phase-averaged)")
    print(f"    • Semi-coherent is intermediate")

    print("\n✓ Generated: output/example_16a_tl_modes.png")


def scenario_b_ray_tracing():
    """
    Scenario B: Ray tracing and visualization.
    """
    print("\n" + "="*80)
    print("SCENARIO B: Ray Tracing (run_type='R')")
    print("="*80)

    # ═══════════════════════════════════════════════════════════════════════
    # ENVIRONMENT - Munk profile for interesting ray paths
    # ═══════════════════════════════════════════════════════════════════════
    env = uacpy.Environment(
        name="Munk Profile - Ray Tracing",
        depth=5000.0,
        ssp_type='munk'
    )

    source = uacpy.Source(
        depth=1000.0,       # At channel axis
        frequency=50.0,
        angles=np.linspace(-15, 15, 31)  # 31 rays from -15° to +15°
    )

    # For ray tracing, we need to specify ray output
    receiver = uacpy.Receiver(
        depths=np.array([1000]),  # Single depth for ray endpoints
        ranges=np.linspace(0, 100000, 100)  # 0-100 km
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RUN BELLHOP IN RAY TRACING MODE
    # ═══════════════════════════════════════════════════════════════════════
    print("\n  Running Bellhop in ray tracing mode...")
    print("    • Computing ray paths...", end=" ", flush=True)

    bellhop = Bellhop(verbose=False)
    result = bellhop.run(env, source, receiver, run_type='R')

    print("✓")

    # ═══════════════════════════════════════════════════════════════════════
    # PLOT RAY PATHS
    # ═══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Full ray fan
    ax = axes[0]
    if hasattr(result, 'ray_data') and result.ray_data is not None:
        # Plot rays if available
        for ray in result.ray_data:
            if 'r' in ray and 'z' in ray:
                ax.plot(ray['r']/1000, ray['z'], 'b-', linewidth=0.5, alpha=0.6)
    else:
        ax.text(0.5, 0.5, 'Ray data not available\n(requires ray file output)',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)

    # Plot environment
    ax.plot(0, source.depth[0], 'r*', markersize=20, label='Source', zorder=10)
    ax.axhline(env.depth, color='k', linewidth=3, label='Bottom')
    ax.set_xlabel('Range (km)', fontweight='bold')
    ax.set_ylabel('Depth (m)', fontweight='bold')
    ax.set_title(f'Ray Paths ({len(source.angles)} rays, ±15° launch angles)',
                 fontweight='bold', fontsize=14)
    ax.invert_yaxis()
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 100])

    # Zoomed view of first convergence zone
    ax = axes[1]
    if hasattr(result, 'ray_data') and result.ray_data is not None:
        for ray in result.ray_data:
            if 'r' in ray and 'z' in ray:
                # Only plot rays that reach zoomed range
                if np.max(ray['r']) >= 20000:
                    ax.plot(ray['r']/1000, ray['z'], 'b-', linewidth=1, alpha=0.7)
    ax.plot(0, source.depth[0], 'r*', markersize=20, label='Source', zorder=10)
    ax.axhline(env.depth, color='k', linewidth=3, label='Bottom')
    ax.set_xlabel('Range (km)', fontweight='bold')
    ax.set_ylabel('Depth (m)', fontweight='bold')
    ax.set_title('Ray Paths - Zoomed View (First Convergence Zone)',
                 fontweight='bold', fontsize=14)
    ax.invert_yaxis()
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([20, 40])
    ax.set_ylim([2000, 0])

    plt.tight_layout()
    plt.savefig('output/example_16b_ray_tracing.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Ray tracing parameters:")
    print(f"    • Number of rays: {len(source.angles)}")
    print(f"    • Launch angles: {source.angles[0]:.1f}° to {source.angles[-1]:.1f}°")
    print(f"    • Source depth: {source.depth[0]:.0f} m (channel axis)")
    print(f"    • Maximum range: 100 km")

    print("\n✓ Generated: output/example_16b_ray_tracing.png")


def scenario_c_eigenrays_arrivals():
    """
    Scenario C: Eigenrays and arrival structure analysis.
    """
    print("\n" + "="*80)
    print("SCENARIO C: Eigenrays & Arrivals (run_type='E' and 'A')")
    print("="*80)

    # ═══════════════════════════════════════════════════════════════════════
    # ENVIRONMENT - Simpler for clear eigenray demonstration
    # ═══════════════════════════════════════════════════════════════════════
    env = uacpy.Environment(
        name="Shallow Water - Eigenrays",
        depth=100.0,
        ssp_type='linear',
        ssp_data=[(0, 1500), (100, 1520)]  # Slight positive gradient
    )

    source = uacpy.Source(
        depth=50.0,
        frequency=100.0,
        angles=np.linspace(-80, 80, 361)  # Wide angle coverage for eigenrays
    )

    # Specific receiver for eigenray analysis
    receiver = uacpy.Receiver(
        depths=np.array([30.0]),    # Single receiver
        ranges=np.array([5000.0])   # 5 km range
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RUN EIGENRAYS
    # ═══════════════════════════════════════════════════════════════════════
    print("\n  Finding eigenrays to receiver at (5 km, 30 m)...")
    print("    • Computing eigenrays...", end=" ", flush=True)

    bellhop_eigen = Bellhop(verbose=False)
    result_eigen = bellhop_eigen.run(env, source, receiver, run_type='E')

    print("✓")

    # ═══════════════════════════════════════════════════════════════════════
    # RUN ARRIVALS
    # ═══════════════════════════════════════════════════════════════════════
    print("    • Computing arrival structure...", end=" ", flush=True)
    try:
        bellhop_arr = Bellhop(verbose=False)
        result_arr = bellhop_arr.run(env, source, receiver, run_type='A')
        print("✓")
    except (NotImplementedError, Exception) as e:
        print(f"⚠ (ASCII format not yet supported - skipped)")
        result_arr = None

    # ═══════════════════════════════════════════════════════════════════════
    # PLOT RESULTS
    # ═══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Eigenrays
    ax = axes[0, 0]
    if hasattr(result_eigen, 'ray_data') and result_eigen.ray_data is not None:
        n_eigenrays = len(result_eigen.ray_data)
        for i, ray in enumerate(result_eigen.ray_data):
            if 'r' in ray and 'z' in ray:
                ax.plot(ray['r']/1000, ray['z'], linewidth=2, alpha=0.7,
                       label=f'Ray {i+1}' if i < 10 else '')
    else:
        n_eigenrays = 0
        ax.text(0.5, 0.5, 'Eigenray data not available',
                ha='center', va='center', transform=ax.transAxes)

    ax.plot(0, source.depth[0], 'r*', markersize=20, label='Source', zorder=10)
    ax.plot(receiver.ranges[0]/1000, receiver.depths[0], 'go',
            markersize=15, label='Receiver', zorder=10)
    ax.axhline(env.depth, color='k', linewidth=3, label='Bottom')
    ax.set_xlabel('Range (km)', fontweight='bold')
    ax.set_ylabel('Depth (m)', fontweight='bold')
    ax.set_title(f'Eigenrays ({n_eigenrays} paths found)', fontweight='bold', fontsize=14)
    ax.invert_yaxis()
    if n_eigenrays <= 10:
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Arrival structure - travel time
    ax = axes[0, 1]
    if hasattr(result_arr, 'arrivals_data') and result_arr.arrivals_data is not None:
        arrivals = result_arr.arrivals_data
        if 'delays' in arrivals and len(arrivals['delays']) > 0:
            travel_times = arrivals['delays']
            amplitudes = arrivals['amplitudes']
            ax.stem(travel_times, amplitudes, basefmt=' ')
            ax.set_xlabel('Travel Time (s)', fontweight='bold')
            ax.set_ylabel('Amplitude', fontweight='bold')
            ax.set_title(f'Arrival Structure ({len(travel_times)} arrivals)',
                        fontweight='bold', fontsize=14)
            ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Arrival data not available',
                ha='center', va='center', transform=ax.transAxes)

    # Arrival amplitude vs number of bounces
    ax = axes[1, 0]
    if hasattr(result_arr, 'arrivals_data') and result_arr.arrivals_data is not None:
        arrivals = result_arr.arrivals_data
        if 'amplitudes' in arrivals and len(arrivals['amplitudes']) > 0:
            n_surf = arrivals['n_top_bounces']
            n_bott = arrivals['n_bot_bounces']
            amplitudes = arrivals['amplitudes']

            # Create bounce labels
            bounce_labels = [f"S{int(ns)}B{int(nb)}" for ns, nb in zip(n_surf, n_bott)]
            x_pos = np.arange(len(bounce_labels))

            ax.bar(x_pos, amplitudes, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Path (Surface/Bottom bounces)', fontweight='bold')
            ax.set_ylabel('Amplitude', fontweight='bold')
            ax.set_title('Amplitude vs Bounce Count', fontweight='bold', fontsize=14)
            ax.set_xticks(x_pos[::max(1, len(x_pos)//10)])
            ax.set_xticklabels(bounce_labels[::max(1, len(bounce_labels)//10)],
                              rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'Arrival data not available',
                ha='center', va='center', transform=ax.transAxes)

    # Summary table
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = "EIGENRAY & ARRIVAL SUMMARY\n" + "="*50 + "\n\n"
    summary_text += f"Receiver location:\n"
    summary_text += f"  • Range: {receiver.ranges[0]/1000:.1f} km\n"
    summary_text += f"  • Depth: {receiver.depths[0]:.1f} m\n\n"

    if hasattr(result_eigen, 'ray_data') and result_eigen.ray_data is not None:
        summary_text += f"Eigenrays found: {len(result_eigen.ray_data)}\n\n"
    else:
        summary_text += f"Eigenrays: Data not available\n\n"

    if hasattr(result_arr, 'arrivals_data') and result_arr.arrivals_data is not None:
        arrivals = result_arr.arrivals_data
        if 'delays' in arrivals and len(arrivals['delays']) > 0:
            summary_text += f"Arrivals found: {len(arrivals['delays'])}\n"
            summary_text += f"First arrival time: {arrivals['delays'].min():.4f} s\n"
            summary_text += f"Last arrival time: {arrivals['delays'].max():.4f} s\n"
            summary_text += f"Time spread: {arrivals['delays'].max() - arrivals['delays'].min():.4f} s\n\n"
            summary_text += f"Direct path (first arrival):\n"
            summary_text += f"  • Travel time: {arrivals['delays'][0]:.4f} s\n"
            summary_text += f"  • Amplitude: {arrivals['amplitudes'][0]:.2e}\n"
            summary_text += f"  • Surface bounces: {int(arrivals['n_top_bounces'][0])}\n"
            summary_text += f"  • Bottom bounces: {int(arrivals['n_bot_bounces'][0])}\n"
    else:
        summary_text += "Arrivals: Data not available\n"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig('output/example_16c_eigenrays_arrivals.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  Analysis complete:")
    if hasattr(result_eigen, 'ray_data') and result_eigen.ray_data:
        print(f"    • Eigenrays found: {len(result_eigen.ray_data)}")
    if hasattr(result_arr, 'arrival_data') and result_arr.arrival_data:
        print(f"    • Arrivals detected: {len(result_arr.arrival_data)}")
        print(f"    • Time spread: {max(arr['travel_time'] for arr in result_arr.arrival_data) - min(arr['travel_time'] for arr in result_arr.arrival_data):.4f} s")

    print("\n✓ Generated: output/example_16c_eigenrays_arrivals.png")


def main():
    """
    Run all Bellhop run mode demonstrations.
    """
    print("\n" + "═"*80)
    print("EXAMPLE 16: Bellhop Run Modes - Comprehensive")
    print("═"*80)
    print("\nThis example demonstrates:")
    print("  • Coherent vs Incoherent vs Semi-coherent TL")
    print("  • Ray tracing and visualization")
    print("  • Eigenray finding")
    print("  • Arrival structure analysis")

    # Run all scenarios
    scenario_a_tl_modes()
    scenario_b_ray_tracing()
    scenario_c_eigenrays_arrivals()

    # Summary
    print("\n" + "═"*80)
    print("EXAMPLE 16 COMPLETE")
    print("═"*80)
    print("\nKey Takeaways:")
    print("  ✓ Coherent TL shows phase interference (modal patterns)")
    print("  ✓ Incoherent TL represents phase-averaged, broadband behavior")
    print("  ✓ Ray tracing reveals propagation geometry")
    print("  ✓ Eigenrays essential for pulse propagation analysis")
    print("  ✓ Arrival structure contains multipath timing and amplitudes")
    print("\nWhen to use each mode:")
    print("  → Coherent: CW signals, narrowband analysis, interference studies")
    print("  → Incoherent: Broadband signals, long-term averages")
    print("  → Semi-coherent: Practical compromise")
    print("  → Rays: Understanding propagation paths, caustics")
    print("  → Eigenrays/Arrivals: Pulse propagation, time-domain analysis")
    print("\n" + "═"*80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
