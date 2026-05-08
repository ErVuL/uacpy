"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 09: Ambient noise (Wenz) + ssrp synthesis + PPSD verification
═══════════════════════════════════════════════════════════════════════════════

Pipeline:
  1. Build a Wenz total ambient-noise PSD via :class:`uacpy.noise.WenzNoise`
     (deep water, Beaufort-6 wind, heavy shipping + rain).
  2. Synthesise a time-domain realisation with :func:`uacpy.signal.ssrp`
     (spectral synthesis of random processes).
  3. Visualise the time–frequency content with
     :class:`uacpy.signal.Spectrogram` — a stationary process should show
     a uniform spectral pattern across time.
  4. Round-trip the realisation through :class:`uacpy.signal.PPSD` to
     verify the synthesis recovers the input spectrum and to visualise
     the level distribution across time segments.

Outputs
-------
output/example_09_wenz_components.png  — Wenz components (per-source).
output/example_09_ssrp_timeseries.png  — synthesised noise waveform snapshot.
output/example_09_ssrp_spectrogram.png — time–frequency content of the noise.
output/example_09_ppsd.png             — PPSD with analytic Wenz overlay.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import uacpy
from uacpy.noise import WenzNoise

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

# 1 µPa = 10⁻⁶ Pa (water-acoustics dB reference).
UPA = 1e-6


def main():
    print("\n" + "═" * 80)
    print("EXAMPLE 09: Ambient noise + ssrp synthesis + PPSD")
    print("═" * 80)

    # ── 1. Wenz spectrum ────────────────────────────────────────────────
    f_plot = np.linspace(1.0, 1e5, int(1e5 - 1))
    wenz_plot = WenzNoise(
        f_plot,
        wind_speed=24,                  # knots (Beaufort 6)
        water_depth='deep',
        shipping_level='high',
        rain_rate='heavy',
    )
    fig, _ = wenz_plot.plot(
        title=(f'{wenz_plot.wind_speed:g} kn / '
               f'{wenz_plot.shipping_level} shipping / '
               f'{wenz_plot.rain_rate} rain'),
    )
    fig.savefig(OUTPUT_DIR / 'example_09_wenz_components.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Saved: output/example_09_wenz_components.png")

    # ── 2. ssrp time-domain realisation of the Wenz total ───────────────
    # ssrp prefers a power-of-two PSD length and samples at fs = f_max·2.
    # Use 2¹⁵ + 1 frequency points up to 50 kHz → fs = 100 kHz.
    n_freq = 2 ** 15 + 1
    f_ssrp = np.linspace(1.0, 5e4, n_freq)
    wenz_ssrp = WenzNoise(
        f_ssrp,
        wind_speed=wenz_plot.wind_speed,
        water_depth=wenz_plot.water_depth,
        shipping_level=wenz_plot.shipping_level,
        rain_rate=wenz_plot.rain_rate,
    )
    Pxx = wenz_ssrp.as_psd()                           # Pa² / Hz (linear)

    duration = 30.0                                    # seconds
    t, x, fs = uacpy.signal.ssrp(Pxx, f_ssrp,
                                 duration=duration, scale=1.0)
    print(f"  ssrp: synthesised {duration:.1f} s @ fs = {fs/1e3:.1f} kHz "
          f"({len(x):,} samples)")

    # Snapshot of the waveform (first 0.2 s).
    n_show = int(0.2 * fs)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t[:n_show] * 1e3, x[:n_show] / UPA, linewidth=0.5, color='C0')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel(r'Pressure [µPa]')
    ax.set_title('Synthesised Wenz-noise time series '
                 f'(first 0.2 s of {duration:.0f} s)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'example_09_ssrp_timeseries.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Saved: output/example_09_ssrp_timeseries.png")

    # ── 3. Spectrogram of the synthesised noise ─────────────────────────
    spec = uacpy.signal.Spectrogram(ref=UPA, nperseg=4096, noverlap=2048)
    spec.compute(x, fs)
    fig, ax = spec.plot(
        title=(f'Wenz @ {wenz_ssrp.wind_speed:g} kn / '
               f'{wenz_ssrp.shipping_level} shipping / '
               f'{wenz_ssrp.rain_rate} rain'),
        ymin=10, ymax=fs / 2,
        vmin=20, vmax=120,
    )
    fig.savefig(OUTPUT_DIR / 'example_09_ssrp_spectrogram.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Saved: output/example_09_ssrp_spectrogram.png")

    # ── 4. PPSD of the synthesised noise ────────────────────────────────
    ppsd = uacpy.signal.PPSD(
        ref=UPA,
        seg_duration=1.0,
        overlap_pct=50,
        ddB=1.0,
        lvlmin=20,
        lvlmax=140,
    )
    ppsd.compute(x, fs)
    fig, ax = ppsd.plot(
        title=(f'Wenz @ {wenz_ssrp.wind_speed:g} kn / '
               f'{wenz_ssrp.shipping_level} shipping / '
               f'{wenz_ssrp.rain_rate} rain'),
        ymin=20, ymax=120,
    )
    # Overlay the analytic Wenz total for direct comparison.
    ax.semilogx(wenz_ssrp.frequencies, wenz_ssrp.total,
                color='magenta', linewidth=2.0,
                label='Wenz total (analytic)')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.85)
    fig.savefig(OUTPUT_DIR / 'example_09_ppsd.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Saved: output/example_09_ppsd.png")

    print("\n✓ Example 09 complete\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
