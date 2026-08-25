"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 10: Signal Processing
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE: Demonstrate signal processing capabilities in UACPY

FEATURES: ✓ Signal generation  ✓ Chirps and wavelets
          ✓ M-sequences  ✓ Basic waveforms
          ✓ Constant-Q spectrogram & band power (Brown 1991)
"""

import sys
import os
from pathlib import Path
# Repo root, so ``import uacpy`` resolves from a source checkout.
sys.path.insert(0, str(Path(__file__).parents[2]))

OUTPUT_DIR = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
                  or Path(__file__).parent / 'output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from uacpy.acoustic_signal.waveforms import (  # noqa: E402
    lfm_chirp, hfm_chirp, ricker_wavelet, gaussian_pulse,
)
from uacpy.acoustic_signal.sequences import mseq  # noqa: E402
from uacpy.acoustic_signal import (  # noqa: E402
    constant_q_spectrogram, constant_q_psd,
)
from uacpy.visualization import (  # noqa: E402
    plot_constant_q_spectrogram, plot_constant_q_psd,
)


def main():
    print("\n" + "═" * 80)
    print("EXAMPLE 10: Signal Processing")
    print("═" * 80)

    # Generate various signals
    print("\n  Generating signals...")

    # LFM chirp
    fs = 10000
    duration = 0.5
    t_lfm, lfm_sig = lfm_chirp(fmin=100, fmax=1000, duration=duration, sample_rate=fs)

    # HFM chirp
    t_hfm, hfm_sig = hfm_chirp(fmin=100, fmax=1000, duration=duration, sample_rate=fs)

    # Ricker wavelet
    t_ricker = np.linspace(0, duration, int(fs * duration))
    ricker_sig = ricker_wavelet(t_ricker, frequency=500)

    # Gaussian pulse
    t_gauss = np.linspace(0, duration, int(fs * duration))
    gauss_sig = gaussian_pulse(t_gauss, delay=duration/2, duration=0.1)

    # M-sequence
    mseq_bits = mseq(m=7)  # 127-bit sequence

    print("  ✓ Generated 5 signal types")

    # Plot results
    fig, axes = plt.subplots(4, 2, figsize=(14, 13))

    # LFM chirp
    ax = axes[0, 0]
    ax.plot(t_lfm[:500], lfm_sig[:500], 'b-', linewidth=1)
    ax.set_title('LFM Chirp (100-1000 Hz)', fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)

    # HFM chirp
    ax = axes[0, 1]
    ax.plot(t_hfm[:500], hfm_sig[:500], 'r-', linewidth=1)
    ax.set_title('HFM Chirp (100-1000 Hz)', fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)

    # Ricker wavelet
    ax = axes[1, 0]
    ax.plot(t_ricker[:1000], ricker_sig[:1000], 'g-', linewidth=1)
    ax.set_title('Ricker Wavelet (500 Hz)', fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)

    # Gaussian pulse
    ax = axes[1, 1]
    ax.plot(t_gauss, gauss_sig, 'm-', linewidth=1)
    ax.set_title('Gaussian Pulse', fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)

    # M-sequence
    ax = axes[2, 0]
    ax.stem(mseq_bits[:50], basefmt=' ', linefmt='b-', markerfmt='bo')
    ax.set_title('M-Sequence (first 50 bits)', fontweight='bold')
    ax.set_xlabel('Bit index')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)

    # Frequency spectrum of LFM
    ax = axes[2, 1]
    freqs = np.fft.rfftfreq(len(lfm_sig), 1/fs)
    spectrum = np.abs(np.fft.rfft(lfm_sig))
    ax.plot(freqs, 20*np.log10(spectrum + 1e-10), 'c-', linewidth=1)
    ax.set_title('LFM Spectrum', fontweight='bold')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.grid(True, alpha=0.3)
    # Twice the sweep's top frequency: shows the 100-1000 Hz band the chirp
    # occupies plus the roll-off on either side of it.
    ax.set_xlim([0, min(2000, freqs[-1])])

    # Constant-Q spectrogram of the LFM chirp (log-frequency, Brown 1991): the
    # sweep traces a curve whose resolution scales with frequency.
    cqt = constant_q_spectrogram(lfm_sig, fs, fmin=80, fmax=2000,
                                 bins_per_octave=24)
    # dB re 1 µPa: the chirp peaks near 116 dB, so a 60 dB window (vmin=60)
    # shows the sweep hot while pushing the weak low-frequency constant-Q
    # leakage (long windows at low fmin) down to the floor instead of
    # saturating everything above 60 dB to one colour.
    plot_constant_q_spectrogram(cqt.frequencies, cqt.times, cqt.power,
                                ax=axes[3, 0], show_colorbar=False,
                                vmin=60, vmax=120)
    axes[3, 0].set_title('', loc='left')   # drop the plotter's own left title
    axes[3, 0].set_title('LFM Constant-Q Spectrogram', fontweight='bold')

    # Time-averaged constant-Q band power per geometric bin.
    cqp = constant_q_psd(lfm_sig, fs, fmin=80, fmax=2000, bins_per_octave=24)
    plot_constant_q_psd(cqp.frequencies, cqp.power, ax=axes[3, 1], ymin=-20,
                        ymax=80)
    axes[3, 1].set_title('', loc='left')   # drop the plotter's own left title
    axes[3, 1].set_title('LFM Constant-Q Band Power', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'example_10_signal_processing.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {OUTPUT_DIR / 'example_10_signal_processing.png'}")
    print("\n✓ Example 10 complete\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
