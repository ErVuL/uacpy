"""Matched filtering, pulse compression and the ambiguity function.

An LFM chirp through a two-path channel, with the echo delays recovered by
correlating against a replica of the transmission. Pulse compression trades the
long pulse needed for energy against the short one needed for resolution: the
processing gain is 10·log10(B·T). The ambiguity function then shows what the
waveform can and cannot separate in delay and Doppler together.

Uses: acoustic_signal.lfm_chirp · simulate_reception · pulse_compression ·
processing_gain_dB · analytic_signal · ambiguity_function · plot_ambiguity
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.acoustic_signal import (ambiguity_function, analytic_signal,
                                   lfm_chirp, processing_gain_dB,
                                   pulse_compression, simulate_reception)

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0xACED)
fs = 20000.0
freq_start, freq_end, pulse_length = 1000.0, 5000.0, 0.02
_, transmitted = lfm_chirp(freq_start, freq_end, pulse_length, sample_rate=fs)
bandwidth = freq_end - freq_start

# Two echoes, the second weaker and 25 ms later. The noise is drawn from the
# seeded generator above, so the figure is identical run to run.
delays = [0.05, 0.075]
t_rx, received = simulate_reception([1.0, 0.5], delays,
                                    source_waveform=transmitted,
                                    sample_rate=fs)
received = received + 0.1 * rng.standard_normal(received.size)

lags, compressed = pulse_compression(received, transmitted, fs)
gain = processing_gain_dB(bandwidth, pulse_length)
strongest_lag = lags[np.argmax(np.abs(compressed))]
print(f"  B·T = {bandwidth * pulse_length:.0f} → processing gain "
      f"{gain:.1f} dB")
print(f"  echoes at {delays[0] * 1e3:.0f} and {delays[1] * 1e3:.0f} ms; "
      f"strongest compressed peak at {strongest_lag * 1e3:.1f} ms")

# The ambiguity function takes a complex waveform. The real chirp holds its
# sweep at both +f and -f, which would draw a second, mirrored ridge (an X);
# its analytic signal keeps the positive-frequency sweep alone, so the surface
# shows the up-chirp's one tilted delay-Doppler ridge.
lag_axis, doppler_axis, ambiguity = ambiguity_function(
    analytic_signal(transmitted), fs, n_doppler_shifts=121)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
uacpy.plot.plot_waveform(transmitted, fs, axes[0, 0], time_unit='ms',
                         color='b', lw=0.7, title='Transmitted LFM chirp')
uacpy.plot.plot_waveform(received, fs, axes[0, 1], t0=float(t_rx[0]),
                         time_unit='ms', color='g', lw=0.6,
                         title='Received (2 echoes + noise)')
# The compressed output's envelope in dB re its peak; dashed lines at the
# true echo delays.
uacpy.plot.plot_waveform(compressed, fs, axes[1, 0], t0=float(lags[0]),
                         value='envelope_dB', time_unit='ms', color='r',
                         lw=0.8,
                         title='Matched-filter output (pulse compression)')
for delay in delays:
    axes[1, 0].axvline(delay * 1e3, color='k', ls='--', alpha=0.5)
axes[1, 0].set_xlim(0, 100)

# In dB re the peak, whose 40 dB window shows the sidelobe structure, not
# only the mainlobe.
uacpy.plot.plot_ambiguity(lag_axis, doppler_axis, ambiguity, ax=axes[1, 1],
                          value='dB', cmap='viridis',
                          title='Ambiguity function |χ(τ, ν)|')
axes[1, 1].set_xlim(-5, 5)

fig.tight_layout()
fig.savefig(OUT / 'example_28_matched_filter.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)
