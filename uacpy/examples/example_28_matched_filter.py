"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 28: Matched Filtering, Pulse Compression & the Ambiguity Function
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    Transmit an LFM chirp through a simple two-path channel, then recover the
    echo delays by replica correlation (pulse compression) and inspect the
    waveform's range-Doppler ambiguity function.

FEATURES DEMONSTRATED:
    ✓ uacpy.acoustic_signal.lfm_chirp + channel.simulate_reception
    ✓ matched_filter / pulse_compression / processing_gain
    ✓ ambiguity_function (range-Doppler resolution)
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from uacpy.acoustic_signal import (  # noqa: E402
    ambiguity_function,
    lfm_chirp,
    processing_gain,
    pulse_compression,
    simulate_reception,
)


def main():
    print("═" * 80)
    print("EXAMPLE 28: Matched Filtering & Ambiguity Function")
    print("═" * 80)

    fs = 20000.0
    fmin, fmax, T = 1000.0, 5000.0, 0.02
    tx, t_tx = lfm_chirp(fmin, fmax, T, fs)
    bandwidth = fmax - fmin

    # Two echoes at 0.05 s and 0.075 s, second weaker.
    amplitudes = [1.0, 0.5]
    delays = [0.05, 0.075]
    t_rx, rx = simulate_reception(tx, amplitudes, delays, fs)
    rx = rx + 0.1 * np.random.randn(rx.size)

    lags, comp = pulse_compression(rx, tx, fs)
    pg = processing_gain(bandwidth, T)
    print(f"\n  Processing gain = {pg:.1f} dB  (B*T = {bandwidth*T:.0f})")
    print(f"  Echo delays (truth): {delays}")

    lag_axis, dop_axis, amb = ambiguity_function(tx.astype(complex), fs,
                                                 n_doppler=121)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    ax.plot(t_tx * 1e3, tx, 'b-', lw=0.7)
    ax.set_title('Transmitted LFM chirp', fontweight='bold')
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t_rx * 1e3, rx, 'g-', lw=0.6)
    ax.set_title('Received (2 echoes + noise)', fontweight='bold')
    ax.set_xlabel('Time (ms)'); ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(lags * 1e3, 20 * np.log10(np.abs(comp) + 1e-6), 'r-', lw=0.8)
    for d in delays:
        ax.axvline(d * 1e3, color='k', ls='--', alpha=0.5)
    ax.set_title('Matched-filter output (pulse compression)', fontweight='bold')
    ax.set_xlabel('Delay (ms)'); ax.set_ylabel('Level (dB)')
    ax.set_xlim(0, 100); ax.set_ylim(-60, 5); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    extent = [lag_axis[0] * 1e3, lag_axis[-1] * 1e3, dop_axis[0], dop_axis[-1]]
    ax.imshow(20 * np.log10(amb + 1e-3), aspect='auto', origin='lower',
              extent=extent, vmin=-40, vmax=0, cmap='viridis')
    ax.set_title('Ambiguity function |chi(tau, nu)| (dB)', fontweight='bold')
    ax.set_xlabel('Delay (ms)'); ax.set_ylabel('Doppler (Hz)')
    ax.set_xlim(-5, 5)

    plt.tight_layout()
    out = OUTPUT_DIR / 'example_28_matched_filter.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {out}")
    print("\n✓ Example 28 complete\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
