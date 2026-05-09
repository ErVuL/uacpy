"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 11: Bellhop Run Modes - Comprehensive
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    Demonstrate all Bellhop run modes and their applications.
    Compare coherent vs incoherent vs semi-coherent TL.
    Show ray tracing, eigenrays, and arrival structure analysis.

COMPLEXITY LEVEL: ⭐⭐⭐ (3/5) - Advanced Model Features

FEATURES DEMONSTRATED:
    ✓ Coherent TL (run_mode=RunMode.COHERENT_TL) - Phase-preserving transmission loss
    ✓ Incoherent TL (run_mode=RunMode.INCOHERENT_TL) - Phase-averaged transmission loss
    ✓ Semi-coherent TL (run_mode=RunMode.SEMICOHERENT_TL) - Hybrid coherent/incoherent
    ✓ Ray tracing (run_mode=RunMode.RAYS) - Ray path visualization
    ✓ Eigenrays (run_mode=RunMode.EIGENRAYS) - Specific receiver rays
    ✓ Arrivals (run_mode=RunMode.ARRIVALS) - Arrival time/amplitude structure
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

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.core.environment import SoundSpeedProfile
from uacpy.models import Bellhop
from uacpy.models import RunMode
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
        bathymetry=5000.0,
        ssp=SoundSpeedProfile.from_munk(5000.0),
    )

    source = uacpy.Source(
        depths=1000.0,      # At channel axis
        frequencies=50.0     # 50 Hz
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
    result_coherent = bellhop_coherent.run(env, source, receiver, run_mode=RunMode.COHERENT_TL)
    print("✓")

    # Incoherent TL
    print("    • Incoherent TL...", end=" ", flush=True)
    bellhop_incoherent = Bellhop(verbose=False)
    result_incoherent = bellhop_incoherent.run(env, source, receiver, run_mode=RunMode.INCOHERENT_TL)
    print("✓")

    # Semi-coherent TL
    print("    • Semi-coherent TL...", end=" ", flush=True)
    bellhop_semicoherent = Bellhop(verbose=False)
    result_semicoherent = bellhop_semicoherent.run(env, source, receiver, run_mode=RunMode.SEMICOHERENT_TL)
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
                       result_coherent.tl, cmap='viridis', vmin=vmin, vmax=vmax,
                       shading='auto', zorder=1)
    ax.set_xlim([result_coherent.ranges[0]/1000, result_coherent.ranges[-1]/1000])
    ax.set_ylim([result_coherent.depths[-1], result_coherent.depths[0]])
    ax.plot(0, source.depths[0], 'r*', markersize=15, label='Source', zorder=12)
    ax.set_xlabel('Range (km)', fontweight='bold')
    ax.set_ylabel('Depth (m)', fontweight='bold')
    ax.set_title('Coherent TL (run_mode=RunMode.COHERENT_TL)', fontweight='bold', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax, label='TL (dB)')

    # Incoherent TL
    ax = axes[0, 1]
    im = ax.pcolormesh(result_incoherent.ranges/1000, result_incoherent.depths,
                       result_incoherent.tl, cmap='viridis', vmin=vmin, vmax=vmax,
                       shading='auto', zorder=1)
    ax.set_xlim([result_incoherent.ranges[0]/1000, result_incoherent.ranges[-1]/1000])
    ax.set_ylim([result_incoherent.depths[-1], result_incoherent.depths[0]])
    ax.plot(0, source.depths[0], 'r*', markersize=15, label='Source', zorder=12)
    ax.set_xlabel('Range (km)', fontweight='bold')
    ax.set_ylabel('Depth (m)', fontweight='bold')
    ax.set_title('Incoherent TL (run_mode=RunMode.INCOHERENT_TL)', fontweight='bold', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax, label='TL (dB)')

    # Semi-coherent TL
    ax = axes[1, 0]
    im = ax.pcolormesh(result_semicoherent.ranges/1000, result_semicoherent.depths,
                       result_semicoherent.tl, cmap='viridis', vmin=vmin, vmax=vmax,
                       shading='auto', zorder=1)
    ax.set_xlim([result_semicoherent.ranges[0]/1000, result_semicoherent.ranges[-1]/1000])
    ax.set_ylim([result_semicoherent.depths[-1], result_semicoherent.depths[0]])
    ax.plot(0, source.depths[0], 'r*', markersize=15, label='Source', zorder=12)
    ax.set_xlabel('Range (km)', fontweight='bold')
    ax.set_ylabel('Depth (m)', fontweight='bold')
    ax.set_title('Semi-coherent TL (run_mode=RunMode.SEMICOHERENT_TL)', fontweight='bold', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax, label='TL (dB)')

    # Range cut comparison at channel axis
    ax = axes[1, 1]
    tl_coherent = result_coherent.at(depth=1000).tl
    tl_incoherent = result_incoherent.at(depth=1000).tl
    tl_semicoherent = result_semicoherent.at(depth=1000).tl

    ax.plot(result_coherent.ranges/1000, tl_coherent,
            'b-', linewidth=2.5, label='Coherent', alpha=0.8)
    ax.plot(result_incoherent.ranges/1000, tl_incoherent,
            'r-', linewidth=2.5, label='Incoherent', alpha=0.8)
    ax.plot(result_semicoherent.ranges/1000, tl_semicoherent,
            'g-', linewidth=2.5, label='Semi-coherent', alpha=0.8)
    ax.set_xlabel('Range (km)', fontweight='bold')
    ax.set_ylabel('Transmission Loss (dB)', fontweight='bold')
    ax.set_title(f'TL Comparison at {source.depths[0]:.0f}m Depth (Channel Axis)',
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
    plt.savefig(OUTPUT_DIR / 'example_11a_tl_modes.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  TL statistics at channel axis (1000m depth):")
    print(f"    • Coherent    - Mean: {np.mean(tl_coherent):.1f} dB, "
          f"Std: {np.std(tl_coherent):.1f} dB")
    print(f"    • Incoherent  - Mean: {np.mean(tl_incoherent):.1f} dB, "
          f"Std: {np.std(tl_incoherent):.1f} dB")
    print(f"    • Semi-coh    - Mean: {np.mean(tl_semicoherent):.1f} dB, "
          f"Std: {np.std(tl_semicoherent):.1f} dB")
    print(f"\n  Key observations:")
    print(f"    • Coherent TL shows strong modal interference (high std dev)")
    print(f"    • Incoherent TL is smoothest (phase-averaged)")
    print(f"    • Semi-coherent is intermediate")

    print("\n  ✓ Saved: output/example_11a_tl_modes.png")


def scenario_b_ray_tracing():
    """
    Scenario B: Ray tracing and visualization.
    """
    print("\n" + "="*80)
    print("SCENARIO B: Ray Tracing (run_mode=RunMode.RAYS)")
    print("="*80)

    # ═══════════════════════════════════════════════════════════════════════
    # ENVIRONMENT - Munk profile for interesting ray paths
    # ═══════════════════════════════════════════════════════════════════════
    env = uacpy.Environment(
        name="Munk Profile - Ray Tracing",
        bathymetry=5000.0,
        ssp=SoundSpeedProfile.from_munk(5000.0),
    )

    source = uacpy.Source(
        depths=1000.0,       # At channel axis
        frequencies=50.0,
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

    bellhop = Bellhop(verbose=False, alpha=(-15.0, 15.0), n_beams=31)
    result = bellhop.run(env, source, receiver, run_mode=RunMode.RAYS)

    print("✓")

    from uacpy.visualization.plots import plot_rays

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    plot_rays(result, env=env, source=source, ax=axes[0],
              title=f'Ray Paths ({len(source.angles)} rays, ±15° launch angles)',
              xlim=(0, 100))
    plot_rays(result, env=env, source=source, ax=axes[1],
              title='Ray Paths — Zoomed View (First Convergence Zone)',
              xlim=(20, 40), ylim=(2000, 0))
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'example_11b_ray_tracing.png', dpi=150,
                bbox_inches='tight')
    plt.close()

    print(f"\n  Ray tracing parameters:")
    print(f"    • Number of rays: {len(source.angles)}")
    print(f"    • Launch angles: {source.angles[0]:.1f}° to {source.angles[-1]:.1f}°")
    print(f"    • Source depth: {source.depths[0]:.0f} m (channel axis)")
    print(f"    • Maximum range: 100 km")

    print("\n  ✓ Saved: output/example_11b_ray_tracing.png")


def scenario_c_eigenrays_arrivals():
    """
    Scenario C: Eigenrays and arrival structure analysis.

    Uses a dense launch fan (n_beams=2001 over ±20°) so Bellhop's native
    EIGENRAYS run-mode converges sharply on the receiver — coarse fans
    return rays that visibly miss because the per-angle vertical spacing
    at the receiver range exceeds the eigenray miss-distance tolerance.
    """
    print("\n" + "="*80)
    print("SCENARIO C: Eigenrays & Arrivals (run_mode=RunMode.EIGENRAYS and ARRIVALS)")
    print("="*80)

    env = uacpy.Environment(
        name="Shallow Water - Eigenrays",
        bathymetry=100.0,
        ssp=SoundSpeedProfile.from_pairs(
            [(0, 1500), (100, 1520)], interp='linear',
        ),
    )

    source = uacpy.Source(depths=50.0, frequencies=100.0)
    receiver = uacpy.Receiver(
        depths=np.array([30.0]),
        ranges=np.array([2000.0]),
    )

    print(f"\n  Receiver at (r={receiver.ranges[0]/1000:.1f} km, "
          f"z={receiver.depths[0]:.0f} m)")

    # RunMode.EIGENRAYS returns every ray Bellhop wrote — the Fortran
    # eigenray tolerance is loose. Filter via Rays methods so only rays
    # that actually land within λ/4 of the receiver survive.
    bellhop_eigen = Bellhop(verbose=False, alpha=(-20.0, 20.0), n_beams=2001)
    wavelength_m = 1500.0 / float(np.atleast_1d(source.frequencies)[0])
    print(f"    • Filtering eigenrays "
          f"(miss < λ/4 ≈ {wavelength_m / 4:.1f} m)...",
          end=" ", flush=True)
    result_eigen = bellhop_eigen.compute_eigenrays(
        env, source,
        range_m=float(receiver.ranges[0]),
        depth_m=float(receiver.depths[0]),
    ).filter_by_miss_distance(wavelength_m / 4).top_n_by_miss(12).truncate_at_receiver()
    print("✓")

    print("    • Computing context ray fan (RunMode.RAYS)...",
          end=" ", flush=True)
    bellhop_full = Bellhop(verbose=False, alpha=(-20.0, 20.0), n_beams=21)
    receiver_fan = uacpy.Receiver(
        depths=np.array([receiver.depths[0]]),
        ranges=np.linspace(0, receiver.ranges[0] * 1.1, 50),
    )
    result_rays = bellhop_full.run(env, source, receiver_fan,
                                   run_mode=RunMode.RAYS)
    print("✓")

    print("    • Computing arrival structure (RunMode.ARRIVALS)...",
          end=" ", flush=True)
    try:
        bellhop_arr = Bellhop(verbose=False, alpha=(-20.0, 20.0), n_beams=201)
        result_arr = bellhop_arr.run(env, source, receiver,
                                     run_mode=RunMode.ARRIVALS)
        print("✓")
    except Exception as e:
        print(f"⚠ ({e})")
        result_arr = None

    from uacpy.visualization.plots import plot_rays

    arrivals_ok = result_arr is not None and len(result_arr) > 0

    n_panels = 3 if arrivals_ok else 2
    fig = plt.figure(figsize=(14, 4 * n_panels + 1))
    gs = fig.add_gridspec(n_panels, 1, hspace=0.45)
    ax_full = fig.add_subplot(gs[0])
    ax_eigen = fig.add_subplot(gs[1])
    ax_arr = fig.add_subplot(gs[2]) if arrivals_ok else None

    plot_rays(result_rays, env=env, source=source, receiver=receiver,
              ax=ax_full, linewidth=1.1, alpha=0.75,
              title=f'Context ray fan ({len(result_rays.rays)} rays, ±20°)')

    n_eigenrays = len(result_eigen.rays)
    if n_eigenrays > 0:
        plot_rays(result_eigen, env=env, source=source, receiver=receiver,
                  ax=ax_eigen, linewidth=1.5, alpha=0.9,
                  title=f'{n_eigenrays} eigenrays at receiver '
                        f'(miss < λ/4 ≈ {wavelength_m / 4:.1f} m)')
    else:
        ax_eigen.text(0.5, 0.5, 'No eigenrays returned',
                      ha='center', va='center', transform=ax_eigen.transAxes)
        ax_eigen.axis('off')

    if arrivals_ok:
        from uacpy.visualization.plots import plot_arrivals
        plot_arrivals(result_arr, ax=ax_arr)

    fig.savefig(OUTPUT_DIR / 'example_11c_eigenrays_arrivals.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    print(f"\n  Analysis complete:")
    print(f"    • Eigenrays found: {n_eigenrays}")
    if arrivals_ok:
        rec = result_arr.to_table()
        if rec:
            delays = [r['delay'] for r in rec]
            print(f"    • Arrivals detected: {len(rec)}")
            print(f"    • Time spread: {max(delays) - min(delays):.4f} s")

    print("\n  ✓ Saved: output/example_11c_eigenrays_arrivals.png")


def scenario_d_compute_eigenrays_pekeris():
    """
    Scenario D: focused ``Bellhop.compute_eigenrays`` demo on a Pekeris guide.

    Same API as scenario C uses internally, but on a clean Pekeris waveguide
    so the multipath structure (direct + surface- and bottom-bounces) is
    easy to read off the ray plot and the printed (alpha, miss, top, bot)
    table.
    """
    print("\n" + "="*80)
    print("SCENARIO D: compute_eigenrays() on a Pekeris waveguide")
    print("="*80)

    from uacpy.core.environment import BoundaryProperties

    bottom = BoundaryProperties(
        acoustic_type='half-space', sound_speed=1600.0,
        density=1.5, attenuation=0.5,
    )
    env = uacpy.Environment(
        name='Pekeris', bathymetry=100.0, ssp=1500.0, bottom=bottom,
    )
    source = uacpy.Source(depths=20.0, frequencies=200.0)

    target_range_m = 3000.0
    target_depth_m = 80.0

    print(f"\n  Finding eigenrays at "
          f"(r={target_range_m/1000:.1f} km, z={target_depth_m:.0f} m)...",
          end=" ", flush=True)
    bellhop = Bellhop(verbose=False, alpha=(-30, 30), n_beams=2001)
    rays = bellhop.compute_eigenrays(
        env, source,
        range_m=target_range_m, depth_m=target_depth_m,
    ).top_n_by_miss(8).truncate_at_receiver()
    print("✓")

    print(f"\n  Found {len(rays.rays)} eigenrays:")
    print(f"    {'alpha (deg)':>12s} {'miss (m)':>10s} {'top':>4s} {'bot':>4s}")
    for r in rays.rays:
        print(f"    {r['alpha']:>12.3f} {r['miss_distance_m']:>10.3f} "
              f"{r['n_top_bounces']:>4d} {r['n_bot_bounces']:>4d}")

    fig, ax = rays.plot(env=env)
    ax.plot(target_range_m / 1000.0, target_depth_m, 'ro', markersize=10,
            markeredgecolor='black', label='Receiver')
    ax.legend(loc='lower right')
    fig.savefig(OUTPUT_DIR / 'example_11d_compute_eigenrays_pekeris.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("\n  ✓ Saved: output/example_11d_compute_eigenrays_pekeris.png")


def main():
    """
    Run all Bellhop run mode demonstrations.
    """
    print("\n" + "═" * 80)
    print("EXAMPLE 11: Bellhop Run Modes - Comprehensive")
    print("═" * 80)
    print("\nThis example demonstrates:")
    print("  • Coherent vs Incoherent vs Semi-coherent TL")
    print("  • Ray tracing and visualization")
    print("  • Eigenray finding (run_mode + compute_eigenrays)")
    print("  • Arrival structure analysis")

    # Run all scenarios
    scenario_a_tl_modes()
    scenario_b_ray_tracing()
    scenario_c_eigenrays_arrivals()
    scenario_d_compute_eigenrays_pekeris()

    # Summary
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

    print("\n✓ Example 11 complete\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
