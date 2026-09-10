"""Matched filtering, pulse compression and the ambiguity function.

An LFM chirp through a two-path channel, with the echo delays recovered by
correlating against a replica of the transmission. Pulse compression trades the
long pulse needed for energy against the short one needed for resolution: the
processing gain is 10·log10(B·T). The ambiguity function then shows what the
waveform can and cannot separate in delay and Doppler together.

Uses: acoustic_signal.lfm_chirp · simulate_reception · pulse_compression ·
processing_gain · ambiguity_function · plot_ambiguity
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import uacpy
from uacpy.acoustic_signal import (ambiguity_function, lfm_chirp,
                                   processing_gain, pulse_compression,
                                   simulate_reception)

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0xACED)
fs = 20000.0
f_min, f_max, pulse_length = 1000.0, 5000.0, 0.02
t_tx, transmitted = lfm_chirp(f_min, f_max, pulse_length, fs)
bandwidth = f_max - f_min

# Two echoes, the second weaker and 25 ms later. The noise is drawn from the
# seeded generator above, so the figure is identical run to run.
delays = [0.05, 0.075]
t_rx, received = simulate_reception(transmitted, [1.0, 0.5], delays, fs)
received = received + 0.1 * rng.standard_normal(received.size)

lags, compressed = pulse_compression(received, transmitted, fs)
gain = processing_gain(bandwidth, pulse_length)
peaks = lags[np.argsort(np.abs(compressed))[-1]]
print(f"  B·T = {bandwidth * pulse_length:.0f} → processing gain "
      f"{gain:.1f} dB")
print(f"  echoes at {delays[0] * 1e3:.0f} and {delays[1] * 1e3:.0f} ms; "
      f"strongest compressed peak at {peaks * 1e3:.1f} ms")

lag_axis, doppler_axis, ambiguity = ambiguity_function(
    transmitted.astype(complex), fs, n_doppler=121)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes[0, 0].plot(t_tx * 1e3, transmitted, 'b-', lw=0.7)
axes[0, 0].set_title('Transmitted LFM chirp', fontweight='bold')
axes[0, 0].set_xlabel('Time (ms)')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t_rx * 1e3, received, 'g-', lw=0.6)
axes[0, 1].set_title('Received (2 echoes + noise)', fontweight='bold')
axes[0, 1].set_xlabel('Time (ms)')
axes[0, 1].set_ylabel('Amplitude')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(lags * 1e3, 20 * np.log10(np.abs(compressed) + 1e-6), 'r-',
                lw=0.8)
for delay in delays:
    axes[1, 0].axvline(delay * 1e3, color='k', ls='--', alpha=0.5)
axes[1, 0].set_title('Matched-filter output (pulse compression)',
                     fontweight='bold')
axes[1, 0].set_xlabel('Delay (ms)')
axes[1, 0].set_ylabel('Level (dB)')
axes[1, 0].set_xlim(0, 100)
axes[1, 0].set_ylim(-60, 5)
axes[1, 0].grid(True, alpha=0.3)

# Log colour scale over |χ| from 1e-2 to 1 — the -40..0 dB window that shows
# the sidelobe structure, not only the mainlobe.
uacpy.plot.plot_ambiguity(lag_axis, doppler_axis, ambiguity, ax=axes[1, 1],
                          cmap='viridis',
                          norm=LogNorm(vmin=1e-2, vmax=1.0),
                          title='Ambiguity function |χ(τ, ν)|')
axes[1, 1].set_xlim(-5, 5)

fig.tight_layout()
fig.savefig(OUT / 'example_28_matched_filter.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)
