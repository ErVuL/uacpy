"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 20: SPARC Time-Domain Analysis
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE: Demonstrate SPARC seismo-acoustic time-domain capabilities.

FEATURES: ✓ Time-domain pressure response  ✓ SPARC parabolic equation
          ✓ Elastic bottom support  ✓ Broadband analysis
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.models import SPARC
import os

def main():
    print("\n" + "═"*80)
    print("EXAMPLE 20: SPARC Time-Domain Analysis")
    print("═"*80)

    # Environment with rigid bottom (SPARC supports vacuum/rigid only)
    env = uacpy.Environment(
        name="SPARC Time-Domain",
        depth=100,
        ssp_type='isovelocity',
        sound_speed=1500,
        bottom=uacpy.BoundaryProperties(
            acoustic_type='rigid'
        )
    )

    source = uacpy.Source(depth=50, frequency=100)

    # SPARC requires single receiver depth per run
    # Use single depth to get time-series data
    receiver = uacpy.Receiver(
        depths=np.array([30]),  # Single depth - enables time-series extraction
        ranges=np.linspace(500, 15000, 30)  # Reduced for faster execution
    )

    print("  Running SPARC (time-domain computation)...", end=" ", flush=True)
    sparc = SPARC(verbose=False)
    result = sparc.run(env, source, receiver, max_depths=1)
    print("✓")

    # Extract time-series data from metadata
    time_series = result.metadata.get('time_series', None) if hasattr(result, 'metadata') else None
    has_time_series = time_series is not None

    # NEW: Plot time series using new plotting function
    if has_time_series:
        from uacpy.visualization import plot_time_series
        print("  Plotting time series...", end=" ", flush=True)
        fig_ts, ax_ts = plot_time_series(result, stacked=True)
        fig_ts.savefig('output/example_20_time_series_stacked.png', dpi=150, bbox_inches='tight')
        plt.close(fig_ts)

        # Also plot overlaid version
        fig_ts_overlay, ax_ts_overlay = plot_time_series(result, stacked=False)
        fig_ts_overlay.savefig('output/example_20_time_series_overlaid.png', dpi=150, bbox_inches='tight')
        plt.close(fig_ts_overlay)
        print("✓")

    # Plot - Enhanced visualization with time-series
    if has_time_series:
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    else:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Plot 1: TL vs range (main result)
    ax1 = fig.add_subplot(gs[0, :])
    tl_data = result.data[0, :]
    ax1.plot(result.ranges/1000, tl_data, 'b-', linewidth=2.5, label='SPARC TL')
    ax1.set_xlabel('Range (km)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Transmission Loss (dB)', fontweight='bold', fontsize=12)
    ax1.set_title('SPARC Time-Domain Transmission Loss at 30m Depth',
                 fontweight='bold', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Compute and set ylim from data
    tl_finite = tl_data[np.isfinite(tl_data)]
    if len(tl_finite) > 0:
        ax1.set_ylim([np.floor(np.min(tl_finite)/10)*10, np.ceil(np.max(tl_finite)/10)*10])

    # Plot 1b & 1c: Time-series data (if available)
    if has_time_series:
        time = time_series['time']
        pressure = time_series['pressure']  # shape: (nt, nr)

        # Plot 1b: Pressure time-series at near, mid, and far ranges
        ax_ts = fig.add_subplot(gs[1, 0])

        # Select 3 representative ranges
        nr = pressure.shape[1]
        idx_near = nr // 6
        idx_mid = nr // 2
        idx_far = nr * 5 // 6

        ax_ts.plot(time * 1000, pressure[:, idx_near], 'b-', linewidth=1.5,
                  label=f'Range: {result.ranges[idx_near]/1000:.1f} km', alpha=0.8)
        ax_ts.plot(time * 1000, pressure[:, idx_mid], 'g-', linewidth=1.5,
                  label=f'Range: {result.ranges[idx_mid]/1000:.1f} km', alpha=0.8)
        ax_ts.plot(time * 1000, pressure[:, idx_far], 'r-', linewidth=1.5,
                  label=f'Range: {result.ranges[idx_far]/1000:.1f} km', alpha=0.8)

        ax_ts.set_xlabel('Time (ms)', fontweight='bold')
        ax_ts.set_ylabel('Pressure (arbitrary units)', fontweight='bold')
        ax_ts.set_title('Time-Domain Pressure Waveforms', fontweight='bold', fontsize=12)
        ax_ts.grid(True, alpha=0.3)
        ax_ts.legend(fontsize=9, framealpha=0.9)
        ax_ts.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)

        # Plot 1c: Spectrogram at mid-range
        ax_spec = fig.add_subplot(gs[1, 1])

        from scipy import signal as scipy_signal
        f_spec, t_spec, Sxx = scipy_signal.spectrogram(
            pressure[:, idx_mid],
            fs=1.0 / time_series['dt'],
            window='hann',
            nperseg=min(256, len(time) // 4)
        )

        im_spec = ax_spec.pcolormesh(t_spec * 1000, f_spec, 10 * np.log10(Sxx + 1e-10),
                                     cmap='hot', shading='gouraud', zorder=1)
        ax_spec.set_xlabel('Time (ms)', fontweight='bold')
        ax_spec.set_ylabel('Frequency (Hz)', fontweight='bold')
        ax_spec.set_title(f'Spectrogram at {result.ranges[idx_mid]/1000:.1f} km',
                         fontweight='bold', fontsize=12)
        ax_spec.set_ylim([0, min(500, f_spec[-1])])  # Show up to 500 Hz
        plt.colorbar(im_spec, ax=ax_spec, label='Power (dB)')

        row_offset = 2
    else:
        row_offset = 1

    # Plot 2: Environment schematic
    ax2 = fig.add_subplot(gs[row_offset, 0])

    # Water column
    ax2.fill_between([0, 15], [0, 0], [env.depth, env.depth],
                    color='lightblue', alpha=0.3, label='Water', zorder=1)

    # Bottom (rigid)
    ax2.fill_between([0, 15], [env.depth, env.depth], [env.depth*1.3, env.depth*1.3],
                    color='gray', alpha=0.5, label='Rigid Bottom', zorder=2)

    # Seafloor line
    ax2.plot([0, 15], [env.depth, env.depth], 'k-', linewidth=3, label='Seafloor', zorder=10)

    # Source
    ax2.plot(0.5, source.depth[0], 'r*', markersize=20, label='Source', zorder=12)

    # Receiver depth line
    ax2.axhline(receiver.depths[0], color='green', linestyle='--', linewidth=2,
               label=f'Receiver ({receiver.depths[0]}m)', alpha=0.7, zorder=11)

    ax2.set_xlim([0, receiver.ranges[-1]/1000 * 1.05])  # 0 to max + 5% margin
    ax2.set_ylim([env.depth*1.3, 0])
    ax2.set_xlabel('Range (km)', fontweight='bold')
    ax2.set_ylabel('Depth (m)', fontweight='bold')
    ax2.set_title('Environment Setup', fontweight='bold', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: TL Gradient (range derivative)
    ax3 = fig.add_subplot(gs[row_offset, 1])

    # Compute gradient
    dtl_dr = np.gradient(tl_data, result.ranges/1000)

    ax3.plot(result.ranges/1000, dtl_dr, 'r-', linewidth=2)
    ax3.set_xlabel('Range (km)', fontweight='bold')
    ax3.set_ylabel('dTL/dR (dB/km)', fontweight='bold')
    ax3.set_title('TL Gradient (Spreading Loss Rate)', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)

    # Compute and set ylim from data
    grad_finite = dtl_dr[np.isfinite(dtl_dr)]
    if len(grad_finite) > 0:
        ylim_val = max(abs(np.percentile(grad_finite, 5)), abs(np.percentile(grad_finite, 95)))
        ax3.set_ylim([-ylim_val*1.1, ylim_val*1.1])

    # Plot 4: Statistics and Summary
    ax4 = fig.add_subplot(gs[row_offset + 1, :])
    ax4.axis('off')

    # Compute statistics
    tl_mean = np.nanmean(tl_data)
    tl_std = np.nanstd(tl_data)
    tl_min = np.nanmin(tl_data)
    tl_max = np.nanmax(tl_data)

    summary = "SPARC MODEL OVERVIEW\n" + "="*70 + "\n\n"

    summary += "MODEL FEATURES:\n"
    summary += "  • Seismo-acoustic Parabolic Equation (PE) in time domain\n"
    summary += "  • Handles elastic bottom with shear waves\n"
    summary += "  • Broadband time-domain computation\n"
    summary += "  • Supports rigid and vacuum bottom conditions\n"
    summary += "  • Range-dependent environments\n\n"

    summary += "SIMULATION PARAMETERS:\n"
    summary += f"  • Environment: {env.name}\n"
    summary += f"  • Water depth: {env.depth}m\n"
    summary += f"  • Sound speed: {env.sound_speed} m/s\n"
    summary += f"  • Bottom type: {env.bottom.acoustic_type if env.bottom else 'Unknown'}\n"
    summary += f"  • Source depth: {source.depth[0]}m\n"
    summary += f"  • Frequency: {source.frequency[0]} Hz\n"
    summary += f"  • Receiver depth: {receiver.depths[0]}m\n"
    summary += f"  • Range: {result.ranges[0]/1000:.2f} - {result.ranges[-1]/1000:.2f} km\n\n"

    summary += "TRANSMISSION LOSS STATISTICS:\n"
    summary += f"  • Mean TL: {tl_mean:.2f} dB\n"
    summary += f"  • Std Dev: {tl_std:.2f} dB\n"
    summary += f"  • Range: {tl_min:.2f} - {tl_max:.2f} dB\n\n"

    if has_time_series:
        summary += f"TIME-SERIES DATA:\n"
        summary += f"  • Time samples: {time_series['nt']}\n"
        summary += f"  • Time step: {time_series['dt']*1000:.3f} ms\n"
        summary += f"  • Duration: {time_series['time'][-1]*1000:.1f} ms\n"
        summary += f"  • Sampling rate: {1.0/time_series['dt']:.1f} Hz\n\n"
        summary += "NOTE: Time-series data preserved in result.metadata['time_series']\n"
        summary += "Access via: time = result.metadata['time_series']['time']\n"
        summary += "           pressure = result.metadata['time_series']['pressure']"
    else:
        summary += "NOTE: Time-series data only preserved for single-depth runs.\n"
        summary += "Use receiver with single depth to access raw time-series data."

    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.suptitle('SPARC Time-Domain Seismo-Acoustic Propagation Model',
                fontsize=16, fontweight='bold', y=0.995)
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/example_20_sparc_time_domain.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n✓ Generated: output/example_20_sparc_time_domain.png")
    if has_time_series:
        print("✓ Generated: output/example_20_time_series_stacked.png (NEW)")
        print("✓ Generated: output/example_20_time_series_overlaid.png (NEW)")
    print("\n" + "═"*80 + "\nEXAMPLE 20 COMPLETE\n" + "═"*80 + "\n")
    print("ALL 20 EXAMPLES COMPLETE!")
    print("="*80 + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
