"""Synthesize a time series from H(f).

The frequency-to-time-domain workflow end to end: run Bellhop in BROADBAND mode
for the transfer function H(d, r, f), build a source waveform, and let the field
convolve them into p(t) at the receiver — p(t) = IFFT(H·S), done for you.

Uses: RunMode.BROADBAND with frequencies= · acoustic_signal.tone_burst ·
Field.synthesize_time_series · Field.at().plot()
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.acoustic_signal import tone_burst

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

env = uacpy.Environment(
    name='Pekeris', bathymetry=100.0, ssp=1500.0,
    bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                    sound_speed=1700.0, density=1.5,
                                    attenuation=0.5))
f_center = 200.0
source = uacpy.Source(depths=20.0, frequencies=f_center)
receiver = uacpy.Receiver(depths=np.array([50.0]),      # one cell, so the
                          ranges=np.array([5000.0]))    # time series is one trace

# BROADBAND returns H(f) at every receiver, one slab per frequency. The grid's
# spacing Δf sets the synthesised record's length, 1/Δf: 512 bins over 50-400 Hz
# give 1.46 s, which holds the 5 km multipath (3.34-3.79 s above -40 dB); 256
# bins give 0.73 s, which, opened just before the first arrival (3.26 s),
# still holds it, with 0.2 s to spare and no wrap notice.
H = uacpy.Bellhop(backend='fortran').run(env, source, receiver,
                        run_mode=uacpy.RunMode.BROADBAND,
                        frequencies=np.linspace(50.0, 400.0, 512))
print(f"  H{H.data.shape} over "
      f"{H.frequencies[0]:.0f}-{H.frequencies[-1]:.0f} Hz")

# A 5-cycle Hann-windowed tone burst at the centre frequency.
fs = 4000.0
_, pulse = tone_burst(f_center, 5, sample_rate=fs)

waveform = H.synthesize_time_series(pulse, sample_rate=fs)
print(f"  p(t){waveform.data.shape}, dt={waveform.dt * 1e3:.3f} ms, "
      f"{waveform.n_times} samples")

# Each cut plots itself: TL(f) with the loss axis downward, p(t) linear.
fig, axes = plt.subplots(2, 1, figsize=(10, 7))
H.at(depth=50.0, range=5000.0).plot(
    ax=axes[0], color='C0', lw=1.2,
    title='Transmission loss at r=5 km, z=50 m')
waveform.at(depth=50.0, range=5000.0).plot(
    ax=axes[1], color='C1', lw=1.0, title='Synthesized time series')
fig.tight_layout()
fig.savefig(OUT / 'example_24_synthesize_time_series.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)
