"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 14: Plotting Features — Stacked Time Series and Mode Heatmaps
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE: Showcase two visualization helpers driven by real model output:
    1. plot_time_series(stacked=True) — stacked impulse responses across ranges
    2. plot_modes_heatmap()           — Kraken mode shapes as a 2-D pcolor panel

The stacked time series uses a Bellhop ``BROADBAND`` transfer function
synthesised against a Ricker source pulse — fast, clean, and the propagation
delay is visible as a linear slope across stacked traces.

Equivalents in the Acoustics-Toolbox MATLAB suite:
    plotts.m     →  plot_time_series()
    plotmode.m   →  plot_modes_heatmap()
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
from uacpy.models import Bellhop, Kraken
from uacpy.models.base import RunMode
from uacpy.visualization import plot_time_series, plot_modes_heatmap


def _ricker(fc: float, fs: float, duration: float = 0.04) -> np.ndarray:
    """Ricker (Mexican-hat) pulse centered at ``fc`` Hz."""
    t = np.arange(int(duration * fs)) / fs
    t0 = duration / 3.0
    arg = (np.pi * fc * (t - t0)) ** 2
    return (1.0 - 2.0 * arg) * np.exp(-arg)


def demo_stacked_time_series():
    """plot_time_series(stacked=True) on Bellhop broadband impulse responses."""
    print("\n" + "─" * 70)
    print("DEMO 1: Stacked time series (Bellhop BROADBAND → Ricker → IFFT)")
    print("─" * 70)

    env = uacpy.Environment(
        name='Pekeris waveguide',
        bathymetry=100,
        sound_speed=1500,
        bottom=uacpy.BoundaryProperties(acoustic_type='rigid'),
    )
    source = uacpy.Source(depths=50, frequencies=100)

    receiver = uacpy.Receiver(
        depths=np.array([50.0]),
        ranges=np.linspace(500, 5000, 12),
    )

    frequencies = np.linspace(50.0, 200.0, 600)

    print("  Running Bellhop BROADBAND...", end=" ", flush=True)
    bellhop = Bellhop(verbose=False)
    tf = bellhop.run(env, source, receiver,
                    run_mode=RunMode.BROADBAND, frequencies=frequencies)
    print(f"✓  ({tf.data.shape[-1]} frequencies, "
          f"df={frequencies[1]-frequencies[0]:.3f} Hz)")

    fs = 1000.0
    waveform = _ricker(fc=source.frequencies[0], fs=fs)

    print("  Synthesising time series...", end=" ", flush=True)
    ts = tf.synthesize_time_series(source_waveform=waveform, sample_rate=fs)
    print(f"✓  shape={ts.data.shape}, "
          f"duration={ts.metadata['time'][-1]:.3f} s")

    fig, _ = plot_time_series(ts, stacked=True)
    out = OUTPUT_DIR / 'example_14_time_series_stacked.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: output/{out.name}")

    fig, _ = plot_time_series(ts, stacked=False)
    out = OUTPUT_DIR / 'example_14_time_series_overlaid.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: output/{out.name}")


def demo_modes_heatmap():
    """plot_modes_heatmap() on Kraken mode shapes — full set + first 20 only."""
    print("\n" + "─" * 70)
    print("DEMO 2: Mode-shape heatmap (Kraken)")
    print("─" * 70)

    env = uacpy.Environment(
        name='Pekeris waveguide',
        bathymetry=100,
        sound_speed=1500,
        bottom=uacpy.BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1700,
            density=1.8,
            attenuation=0.5,
        ),
    )
    source = uacpy.Source(depths=50, frequencies=100)

    receiver = uacpy.Receiver(
        depths=np.linspace(0, env.depth, 200),
        ranges=np.array([1000.0]),
    )

    print("  Running Kraken modes...", end=" ", flush=True)
    kraken = Kraken(verbose=False)
    modes = kraken.run(env, source, receiver,
                       run_mode=RunMode.MODES, n_modes=50)
    n_modes = len(modes.metadata.get('k', []))
    print(f"✓  ({n_modes} modes)")

    fig, _ = plot_modes_heatmap(modes, mode_range=None,
                                   normalize=True, figsize=(14, 8))
    out = OUTPUT_DIR / 'example_14_modes_heatmap_all.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: output/{out.name}")

    fig, _ = plot_modes_heatmap(modes, mode_range=(0, 20),
                                   normalize=True, figsize=(12, 8))
    out = OUTPUT_DIR / 'example_14_modes_heatmap_subset.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: output/{out.name}")


def main():
    print("\n" + "═" * 80)
    print("EXAMPLE 14: plot_time_series & plot_modes_heatmap")
    print("═" * 80)

    demo_stacked_time_series()
    demo_modes_heatmap()

    print("\n✓ Example 14 complete\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
