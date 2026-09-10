"""Synthesize a time series from H(f).

The frequency-to-time-domain workflow end to end: run Bellhop in BROADBAND mode
for the transfer function H(d, r, f), build a source waveform, and let the field
convolve them into p(t) at the receiver — p(t) = IFFT(H·S), done for you.

Uses: RunMode.BROADBAND with frequencies= · acoustic_signal.gaussian_pulse ·
Field.synthesize_time_series · Field.at().plot()
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.acoustic_signal.waveforms import gaussian_pulse

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

# BROADBAND returns H(f) at every receiver, one slab per frequency.
H = uacpy.Bellhop().run(env, source, receiver,
                        run_mode=uacpy.RunMode.BROADBAND,
                        frequencies=np.linspace(50.0, 400.0, 256))
print(f"  H{H.data.shape} over "
      f"{H.frequencies[0]:.0f}-{H.frequencies[-1]:.0f} Hz")

# A 5-cycle Gaussian-windowed sinusoid at the centre frequency.
fs = 4000.0
duration = 5 / f_center
t = np.arange(0, duration, 1.0 / fs)
sigma = duration / 6
# exp(-(t - T/2)² / 2σ²): gaussian_pulse's width argument is σ·√2.
pulse = gaussian_pulse(t, duration / 2, sigma * np.sqrt(2)) * np.sin(
    2 * np.pi * f_center * t)

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
