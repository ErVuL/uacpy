"""Signal-processing tour — waveforms, sequences and constant-Q analysis.

The transmit waveforms uacpy can build (LFM and HFM chirps, a Ricker wavelet, a
Gaussian pulse, an m-sequence), the LFM's ordinary spectrum, and the same chirp
seen through a constant-Q transform — log-frequency, resolution scaling with
frequency, the way hearing does (Brown 1991).

Uses: acoustic_signal.lfm_chirp / hfm_chirp / ricker_wavelet / gaussian_pulse ·
sequences.mseq · constant_q_spectrogram · constant_q_psd ·
plot_constant_q_spectrogram · plot_constant_q_psd
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.acoustic_signal import constant_q_psd, constant_q_spectrogram
from uacpy.acoustic_signal.sequences import mseq
from uacpy.acoustic_signal.waveforms import (gaussian_pulse, hfm_chirp,
                                             lfm_chirp, ricker_wavelet)

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

fs, duration = 10000, 0.5
t_lfm, lfm = lfm_chirp(fmin=100, fmax=1000, duration=duration, sample_rate=fs)
t_hfm, hfm = hfm_chirp(fmin=100, fmax=1000, duration=duration, sample_rate=fs)
t = np.linspace(0, duration, int(fs * duration))
ricker = ricker_wavelet(t, frequency=500)
gaussian = gaussian_pulse(t, delay=duration / 2, duration=0.1)
sequence = mseq(m=7)                                   # 127 bits
print(f"  {t_lfm.size} samples per chirp @ {fs} Hz, "
      f"m-sequence length {sequence.size}")

fig, axes = plt.subplots(4, 2, figsize=(14, 13))
for ax, time, signal, title, colour in (
        (axes[0, 0], t_lfm[:500], lfm[:500], 'LFM chirp (100-1000 Hz)', 'b'),
        (axes[0, 1], t_hfm[:500], hfm[:500], 'HFM chirp (100-1000 Hz)', 'r'),
        (axes[1, 0], t[:1000], ricker[:1000], 'Ricker wavelet (500 Hz)', 'g'),
        (axes[1, 1], t, gaussian, 'Gaussian pulse', 'm')):
    ax.plot(time, signal, colour, linewidth=1)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.grid(True, alpha=0.3)

axes[2, 0].stem(sequence[:50], basefmt=' ', linefmt='b-', markerfmt='bo')
axes[2, 0].set_title('M-sequence (first 50 bits)', fontweight='bold')
axes[2, 0].set_xlabel('Bit index')
axes[2, 0].set_ylabel('Value')
axes[2, 0].grid(True, alpha=0.3)

frequencies = np.fft.rfftfreq(lfm.size, 1 / fs)
axes[2, 1].plot(frequencies,
                20 * np.log10(np.abs(np.fft.rfft(lfm)) + 1e-10), 'c-', lw=1)
axes[2, 1].set_title('LFM spectrum', fontweight='bold')
axes[2, 1].set_xlabel('Frequency (Hz)')
axes[2, 1].set_ylabel('Magnitude (dB)')
axes[2, 1].grid(True, alpha=0.3)
# Twice the sweep's top frequency: the 100-1000 Hz band plus the roll-off.
axes[2, 1].set_xlim([0, min(2000, frequencies[-1])])

# Constant-Q: the sweep traces a curve whose resolution scales with frequency.
cqt = constant_q_spectrogram(lfm, fs, fmin=80, fmax=2000, bins_per_octave=24)
# dB re 1 µPa: the chirp peaks near 116 dB, so a 60 dB window shows the sweep
# hot while pushing the weak low-frequency constant-Q leakage (long windows at
# low fmin) to the floor instead of saturating everything above 60 dB.
uacpy.plot.plot_constant_q_spectrogram(cqt.frequencies, cqt.times, cqt.power,
                                       ax=axes[3, 0], show_colorbar=False,
                                       vmin=60, vmax=120)
axes[3, 0].set_title('', loc='left')       # drop the plotter's own left title
axes[3, 0].set_title('LFM constant-Q spectrogram', fontweight='bold')

cq_psd = constant_q_psd(lfm, fs, fmin=80, fmax=2000, bins_per_octave=24)
# The same 60 dB window as the spectrogram: the in-band bins peak near 107 dB.
uacpy.plot.plot_constant_q_psd(cq_psd.frequencies, cq_psd.power,
                               ax=axes[3, 1], ymin=60, ymax=120)
axes[3, 1].set_title('', loc='left')
axes[3, 1].set_title('LFM constant-Q band power', fontweight='bold')

fig.tight_layout()
fig.savefig(OUT / 'example_10_signal_processing.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)
