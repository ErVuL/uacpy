"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 15: Signal Processing
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE: Demonstrate signal processing capabilities in UACPY

FEATURES: ✓ Signal generation  ✓ Chirps and wavelets
          ✓ M-sequences  ✓ Basic waveforms
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from uacpy.acoustic_signal.generation import (
    lfm_chirp, hfm_chirp, ricker_wavelet,
    gaussian_pulse, mseq
)
import os

def main():
    print("\n" + "═"*80)
    print("EXAMPLE 15: Signal Processing")
    print("═"*80)

    # Generate various signals
    print("\n  Generating signals...")

    # LFM chirp
    fs = 10000
    duration = 0.5
    t_lfm, lfm_sig = lfm_chirp(fmin=100, fmax=1000, T=duration, sample_rate=fs)

    # HFM chirp
    t_hfm, hfm_sig = hfm_chirp(fmin=100, fmax=1000, T=duration, sample_rate=fs)

    # Ricker wavelet
    t_ricker = np.linspace(0, duration, int(fs * duration))
    ricker_sig = ricker_wavelet(t_ricker, F=500)

    # Gaussian pulse
    t_gauss = np.linspace(0, duration, int(fs * duration))
    gauss_sig = gaussian_pulse(t_gauss, delay=duration/2, duration=0.1)

    # M-sequence
    mseq_bits = mseq(m=7)  # 127-bit sequence

    print("  ✓ Generated 5 signal types")

    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

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
    ax.set_title('HFM Chirp (1000-100 Hz)', fontweight='bold')
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
    # Set xlim to show relevant frequency range (2x chirp bandwidth)
    ax.set_xlim([0, min(2000, freqs[-1])])

    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/example_15_signal_processing.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n✓ Generated: output/example_15_signal_processing.png")
    print("\n" + "═"*80 + "\nEXAMPLE 15 COMPLETE\n" + "═"*80 + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
