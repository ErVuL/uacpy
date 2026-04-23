"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 22: Plotting Features — Time Series and Mode Heatmaps
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE: Demonstrate three plotting functions in uacpy:
    1. plot_time_series() - Time-domain waveforms
    2. plot_modes_heatmap() - Mode shape heatmaps

FEATURES: ✓ Time-series analysis ✓ Mode heatmap overview  
          ✓ MATLAB equivalents demonstrated
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.models import SPARC, Kraken
from uacpy.models.base import RunMode
from uacpy.visualization import (
    plot_time_series,
    plot_modes_heatmap
)
import os


def demo_time_series():
    """Demo 1: Time Series Plotting (SPARC model)"""
    print("\n" + "─"*70)
    print("DEMO 1: Time Series Plotting")
    print("─"*70)
    print("Equivalent to MATLAB's plotts.m")

    # Simple isovelocity environment
    env = uacpy.Environment(
        name="Time Series Demo",
        depth=100,
        ssp_type='isovelocity',
        sound_speed=1500,
        bottom=uacpy.BoundaryProperties(acoustic_type='rigid')
    )

    source = uacpy.Source(depth=50, frequency=100)

    # Single depth for time series data
    receiver = uacpy.Receiver(
        depths=np.array([30]),  # Single depth preserves time series
        ranges=np.linspace(1000, 10000, 20)
    )

    print("  Running SPARC (time-domain)...", end=" ", flush=True)
    sparc = SPARC(verbose=False)
    result = sparc.run(env, source, receiver, max_depths=1)
    print("✓")

    # Check if time series data exists
    has_ts = 'time_series' in result.metadata
    if not has_ts:
        print("  ! Warning: Time series data not available")
        return

    print(f"  Time series: {result.metadata['time_series']['nt']} samples, " +
          f"{result.metadata['time_series']['time'][-1]*1000:.1f} ms duration")

    #plot_time_series() - Stacked version
    print("\n  Creating stacked time series plot...", end=" ", flush=True)
    fig1, ax1 = plot_time_series(result, stacked=True)
    fig1.savefig(OUTPUT_DIR / 'example_22_time_series_stacked.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("✓")

    #plot_time_series() - Overlaid version
    print("  Creating overlaid time series plot...", end=" ", flush=True)
    fig2, ax2 = plot_time_series(result, stacked=False)
    fig2.savefig(OUTPUT_DIR / 'example_22_time_series_overlaid.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print("✓")

    print("\n  ✓ Demo 1 complete!")
    print("    - plot_time_series() creates professional waveform plots")
    print("    - Stacked mode: offset by depth with automatic scaling")
    print("    - Overlaid mode: all traces on same axis")
    print("    - Equivalent to MATLAB plotts.m")


def demo_modes_heatmap():
    """Demo 2: Mode Shape Heatmap (Kraken model)"""
    print("\n" + "─"*70)
    print("DEMO 2: Mode Shape Heatmap")
    print("─"*70)
    print("Equivalent to MATLAB's plotmode.m (pcolor panel)")

    # Pekeris waveguide (classic for modes)
    env = uacpy.Environment(
        name="Pekeris Waveguide",
        depth=100,
        ssp_type='isovelocity',
        sound_speed=1500,
        bottom=uacpy.BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1700,
            density=1.8,
            attenuation=0.5
        )
    )

    source = uacpy.Source(depth=50, frequency=100)

    print("  Running Kraken (normal modes)...", end=" ", flush=True)
    kraken = Kraken(verbose=False)

    # For mode heatmap visualization, we need mode shapes at many depths
    # Create a fine depth grid and use it as a dummy receiver
    mode_depths = np.linspace(0, env.depth, 200)
    dummy_receiver = uacpy.Receiver(depths=mode_depths, ranges=np.array([1000.0]))

    # Run Kraken directly with the dummy receiver to get mode shapes at all depths
    modes = kraken.run(env, source, dummy_receiver, run_mode=RunMode.MODES, n_modes=50)

    M = len(modes.metadata.get('k', []))
    print(f"✓ ({M} modes)")

    #plot_modes_heatmap() - All modes
    print("\n  Creating mode heatmap (all modes)...", end=" ", flush=True)
    fig1, ax1, cbar1 = plot_modes_heatmap(
        modes,
        mode_range=None,  # Plot all modes
        normalize=True,
        figsize=(14, 8)
    )
    fig1.suptitle('Mode Shape Heatmap - All Modes (Normalized)',
                  fontsize=14, fontweight='bold')
    fig1.savefig(OUTPUT_DIR / 'example_22_modes_heatmap_all.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("✓")

    #plot_modes_heatmap() - Subset of modes
    print("  Creating mode heatmap (first 20 modes)...", end=" ", flush=True)
    fig2, ax2, cbar2 = plot_modes_heatmap(
        modes,
        mode_range=(0, 20),  # First 20 modes
        normalize=True,
        figsize=(12, 8)
    )
    fig2.suptitle('Mode Shape Heatmap - First 20 Modes',
                  fontsize=14, fontweight='bold')
    fig2.savefig(OUTPUT_DIR / 'example_22_modes_heatmap_subset.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print("✓")

    #plot_modes_heatmap() - Without normalization
    print("  Creating mode heatmap (unnormalized)...", end=" ", flush=True)
    fig3, ax3, cbar3 = plot_modes_heatmap(
        modes,
        mode_range=(0, 20),
        normalize=False,  # Show relative amplitudes
        figsize=(12, 8)
    )
    fig3.suptitle('Mode Shape Heatmap - Relative Amplitudes (Unnormalized)',
                  fontsize=14, fontweight='bold')
    fig3.savefig(OUTPUT_DIR / 'example_22_modes_heatmap_unnorm.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print("✓")

    print("\n  ✓ Demo 2 complete!")
    print("    - plot_modes_heatmap() shows all modes at once")
    print("    - Useful for identifying mode structure and cutoff depths")
    print("    - Normalized vs unnormalized options")
    print("    - Equivalent to MATLAB plotmode.m pcolor panel")

def main():
    print("\n" + "═"*70)
    print("EXAMPLE 22: Plotting Features — Time Series and Mode Heatmaps")
    print("═"*70)
    print("\nTwo plotting functions demonstrated:")
    print("  1. plot_time_series() - Time-domain waveforms")
    print("  2. plot_modes_heatmap() - Mode shape heatmaps")
    print("\nThese functions match Acoustics Toolbox MATLAB capabilities:")
    print("  • plotts.m → plot_time_series()")
    print("  • plotmode.m (pcolor) → plot_modes_heatmap()")

    # Run demonstrations
    demo_time_series()
    demo_modes_heatmap()

    print("\n" + "═"*70)
    print("EXAMPLE 22 COMPLETE")
    print("═"*70)
    print("\nGenerated files:")
    print("  ✓ output/example_22_time_series_stacked.png")
    print("  ✓ output/example_22_time_series_overlaid.png")
    print("  ✓ output/example_22_modes_heatmap_all.png")
    print("  ✓ output/example_22_modes_heatmap_subset.png")
    print("  ✓ output/example_22_modes_heatmap_unnorm.png")



    return 0


if __name__ == "__main__":
    sys.exit(main())
