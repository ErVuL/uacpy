"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 09: Ambient noise (Wenz) + PSD→time-series synthesis + PPSD verification
═══════════════════════════════════════════════════════════════════════════════

Pipeline:
  1. Build a Wenz total ambient-noise PSD via :class:`uacpy.noise.WenzNoise`
     (deep water, Beaufort-6 wind, heavy shipping + rain).
  2. Synthesise a time-domain realisation with
     :func:`uacpy.acoustic_signal.synthesize_noise_from_psd`
     (spectral synthesis of random processes).
  3. Visualise the time–frequency content with
     :class:`uacpy.acoustic_signal.Spectrogram` — a stationary process should show
     a uniform spectral pattern across time.
  4. Round-trip the realisation through :class:`uacpy.acoustic_signal.PPSD` to
     verify the synthesis recovers the input spectrum and to visualise
     the level distribution across time segments.
  5. Integrate the realisation into a per-band Sound Exposure Level (SEL)
     with :class:`uacpy.acoustic_signal.SEL` (ISO 18405) — the cumulative
     energy dose of the synthesised dataset, third-octave band by band.

Outputs
-------
output/example_09_wenz_components.png  — Wenz components (per-source).
output/example_09_ssrp_timeseries.png  — synthesised noise waveform snapshot.
output/example_09_ssrp_spectrogram.png — time–frequency content of the noise.
output/example_09_ppsd.png             — PPSD with analytic Wenz overlay.
output/example_09_sel.png              — per-band SEL of the realisation.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import uacpy  # noqa: E402
from uacpy.noise import WenzNoise  # noqa: E402

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
    fig, _ = uacpy.visualization.plot_wenz(
        wenz_plot,
        title=(f'{wenz_plot.wind_speed:g} kn / '
               f'{wenz_plot.shipping_level} shipping / '
               f'{wenz_plot.rain_rate} rain'),
    )
    fig.savefig(OUTPUT_DIR / 'example_09_wenz_components.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Saved: output/example_09_wenz_components.png")

    # ── 2. ssrp time-domain realisation of the Wenz total ───────────────
    n_fft = 1
    f_ssrp = np.linspace(1.0, 5e4, 10000)
    wenz_ssrp = WenzNoise(
        f_ssrp,
        wind_speed=wenz_plot.wind_speed,
        water_depth=wenz_plot.water_depth,
        shipping_level=wenz_plot.shipping_level,
        rain_rate=wenz_plot.rain_rate,
    )
    Pxx = wenz_ssrp.as_psd(ref=UPA)                    # SI Pa²/Hz (linear)

    duration = 30.0                                    # seconds
    t, x, fs = uacpy.acoustic_signal.synthesize_noise_from_psd(
        Pxx, f_ssrp, sample_rate=96000,
        duration=duration, scale=1.0, n_fft=n_fft)
    print(f"  noise synthesis: {duration:.1f} s @ fs = {fs/1e3:.1f} kHz "
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
    sf, st, sSxx = uacpy.acoustic_signal.spectrogram(
        x, fs, nperseg=4096, noverlap=2048)
    fig, ax = uacpy.visualization.plot_spectrogram(
        sf, st, sSxx, ref=UPA,
        title=(f'Wenz @ {wenz_ssrp.wind_speed:g} kn / '
               f'{wenz_ssrp.shipping_level} shipping / '
               f'{wenz_ssrp.rain_rate} rain'),
        ymin=10, ymax=fs / 2,
        vmin=20, vmax=120,
    )
    fig.savefig(OUTPUT_DIR / 'example_09_ssrp_spectrogram.png',
                dpi=150, bbox_inches='tight')
    # plt.close(fig)
    print("  ✓ Saved: output/example_09_ssrp_spectrogram.png")

    # ── 4. PPSD of the synthesised noise ────────────────────────────────
    ppsd_result = uacpy.acoustic_signal.ppsd(
        x, fs, ref=UPA, seg_duration=1.0, overlap_pct=50, ddB=1.0,
        lvlmin=20, lvlmax=140,
    )
    fig, ax = uacpy.visualization.plot_ppsd(
        ppsd_result,
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
    # plt.close(fig)
    print("  ✓ Saved: output/example_09_ppsd.png")

    # ── 5. Sound Exposure Level (SEL) of the synthesised realisation ─────
    # SEL is the time-integral of p²(t) (ISO 18405) — the cumulative energy
    # dose of the record. Computed here per third-octave band over the whole
    # 30 s realisation; each bar is dB re 1 µPa²·s. The broadband total is the
    # incoherent (energy) sum across bands.
    sel_dur = len(x) / fs
    sel_vals, sel_bands = uacpy.acoustic_signal.sel(
        x, fs, fmin=10.0, fmax=fs / 2.0, band_type='third_octave',
    )
    total_sel_db = 10.0 * np.log10(sel_vals.sum() / UPA ** 2)
    print(f"  SEL: broadband {total_sel_db:.1f} dB re 1 µPa²·s over "
          f"{sel_dur:.0f} s across {len(sel_bands)} third-octave bands")
    fig, ax = uacpy.visualization.plot_sel(
        sel_vals, sel_bands, ref=UPA, duration=sel_dur,
        band_type='third_octave',
        title=(f'Wenz @ {wenz_ssrp.wind_speed:g} kn / '
               f'{wenz_ssrp.shipping_level} shipping / '
               f'{wenz_ssrp.rain_rate} rain'),
    )
    fig.savefig(OUTPUT_DIR / 'example_09_sel.png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✓ Saved: output/example_09_sel.png")

    print("\n✓ Example 09 complete\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
