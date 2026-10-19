"""Signal-processing tour — waveforms, sequences and constant-Q analysis.

The transmit waveforms uacpy can build (LFM and HFM chirps, a Ricker wavelet, a
Gaussian pulse, an m-sequence), the LFM's ordinary spectrum, and the same chirp
seen through a constant-Q transform — log-frequency, resolution scaling with
frequency, the way hearing does (Brown 1991).

Uses: acoustic_signal.lfm_chirp / hfm_chirp / ricker_wavelet / gaussian_pulse ·
m_sequence · welch · plot_waveform · plot_psd · constant_q_transform · constant_q_spectrogram ·
constant_q · plot_constant_q_transform ·
plot_constant_q_spectrogram
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.acoustic_signal import (constant_q, constant_q_spectrogram,
                                   constant_q_transform, gaussian_pulse,
                                   hfm_chirp, lfm_chirp, m_sequence,
                                   ricker_wavelet, welch)

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

fs, duration = 10000, 0.5
t_lfm, lfm = lfm_chirp(freq_start=100, freq_end=1000, duration=duration, sample_rate=fs)
_, hfm = hfm_chirp(freq_start=100, freq_end=1000, duration=duration, sample_rate=fs)
t = np.arange(int(round(fs * duration))) / fs      # the (t, x) builders' step
ricker = ricker_wavelet(t, frequency=500)
gaussian = gaussian_pulse(t, delay=duration / 2, duration=0.1)
sequence = m_sequence(7)                               # 127 bits
print(f"  {t_lfm.size} samples per chirp @ {fs} Hz, "
      f"m-sequence length {sequence.size}")

fig, axes = plt.subplots(3, 3, figsize=(18, 11))
uacpy.plot.plot_waveform(lfm[:500], fs, axes[0, 0], color='b',
                         title='LFM chirp (100-1000 Hz)')
uacpy.plot.plot_waveform(hfm[:500], fs, axes[0, 1], color='r',
                         title='HFM chirp (100-1000 Hz)')
uacpy.plot.plot_waveform(ricker[:100], fs, axes[0, 2], color='g',
                         title='Ricker wavelet (500 Hz)')
uacpy.plot.plot_waveform(gaussian, fs, axes[1, 0], color='m',
                         title='Gaussian pulse')

axes[1, 1].stem(sequence[:50], basefmt=' ', linefmt='b-', markerfmt='bo')
axes[1, 1].set_title('M-sequence (first 50 bits)', fontweight='bold')
axes[1, 1].set_xlabel('Bit index')
axes[1, 1].set_ylabel('Value')
axes[1, 1].grid(True, alpha=0.3)

# The chirp's spectrum: one Welch segment over the whole record
spectrum = welch(lfm, fs, nperseg=lfm.size, scaling='spectrum')
uacpy.plot.plot_psd(spectrum, ax=axes[1, 2], freq_scale='linear', color='c',
                    title='LFM spectrum')
# Twice the sweep's top frequency: the 100-1000 Hz band plus the roll-off.
axes[1, 2].set_xlim([0, 2000])

# Constant-Q: the sweep traces a curve whose resolution scales with frequency.
cqt = constant_q_spectrogram(lfm, fs, freq_min=80, freq_max=2000, bins_per_octave=24,
                             scaling='spectrum')
# dB re 1 µPa: the chirp peaks near 114 dB, so a 60 dB window shows the sweep
# hot while pushing the weak low-frequency constant-Q leakage (long windows at
# low freq_min) to the floor instead of saturating everything above 60 dB.
uacpy.plot.plot_constant_q_spectrogram(cqt.frequencies, cqt.times, cqt.power,
                                       ax=axes[2, 0], scaling='spectrum',
                                       show_colorbar=False,
                                       vmin=60, vmax=120)
axes[2, 0].set_title('LFM constant-Q spectrogram', fontweight='bold')

cq_psd = constant_q(lfm, fs, scaling='spectrum', freq_min=80, freq_max=2000,
                             bins_per_octave=24)
# The same 60 dB window as the spectrogram: the in-band bins peak near 104 dB.
# Passed whole, so the axis reads the estimate's own scaling (band power).
uacpy.plot.plot_psd(cq_psd, ax=axes[2, 1], ymin=60, ymax=120,
                    title='LFM constant-Q band power')

# The raw transform the two panels above are built from: one frame, centred on
# the record, complex coefficients on the same geometric bin grid. The frame
# centre is mid-sweep, where a linear chirp is at the mean of its limits, and
# that is the bin the magnitude peaks in.
f_mid = 0.5 * (100 + 1000)
cqt = constant_q_transform(lfm, fs, freq_min=80, freq_max=2000, bins_per_octave=24)
magnitude = np.abs(cqt.coefficients)
uacpy.plot.plot_constant_q_transform(cqt.frequencies, cqt.coefficients,
                                     ax=axes[2, 2], lw=1.3)
axes[2, 2].axvline(f_mid, color='crimson', ls='--', lw=1.1,
                   label=f'{f_mid:.0f} Hz at the frame centre')
axes[2, 2].set_title('LFM constant-Q frame (raw |X|)', fontweight='bold')
axes[2, 2].legend(loc='upper left', fontsize='small')
print(f"  raw constant-Q frame peaks at "
      f"{cqt.frequencies[magnitude.argmax()]:.0f} Hz in {cqt.frequencies.size} "
      f"bins, against {f_mid:.0f} Hz swept at the frame centre")

fig.tight_layout()
fig.savefig(OUT / 'example_10_signal_processing.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)
