"""Adaptive and high-resolution array processing.

Two plane waves 14° apart on a 16-element half-wavelength line array, resolved
three ways: the conventional Bartlett beamformer, the adaptive MVDR (Capon)
estimator, and the subspace MUSIC estimator. Bartlett's beamwidth sets the
classical resolution limit; the other two are built to beat it.

Uses: acoustic_signal.steering_vectors · sonar.csdm ·
bartlett · mvdr · music_spectrum · plot.plot_beam_power
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.acoustic_signal import (bartlett, music_spectrum, mvdr,
                                   steering_vectors)
from uacpy.sonar import csdm

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

frequency, c = 1500.0, 1500.0
positions = np.arange(16) * (c / frequency / 2.0)   # half-wavelength spacing
true_angles = [-8.0, 6.0]

# Two uncorrelated sources on the array, plus a little sensor noise.
rng = np.random.default_rng(0)
n_snapshots = 500
data = np.zeros((positions.size, n_snapshots), dtype=complex)
for angle in true_angles:
    steering = steering_vectors(positions, [angle], frequency, c)[0]
    amplitude = (rng.standard_normal(n_snapshots)
                 + 1j * rng.standard_normal(n_snapshots))
    data += np.outer(steering, amplitude)
data += 0.05 * (rng.standard_normal(data.shape)
                + 1j * rng.standard_normal(data.shape))

covariance = csdm(data)
angles = np.linspace(-40, 40, 801)
replicas = steering_vectors(positions, angles, frequency, c)
conventional = bartlett(covariance, replicas)
capon = mvdr(covariance, replicas)
music = music_spectrum(covariance, replicas, n_sources=2)

# A bearing is a local maximum of the spectrum; the two highest are the
# estimates (the two highest samples would both sit on the strongest lobe).
for name, spectrum in (('Bartlett', conventional), ('MVDR', capon),
                       ('MUSIC', music)):
    maxima = np.flatnonzero((spectrum[1:-1] > spectrum[:-2])
                            & (spectrum[1:-1] >= spectrum[2:])) + 1
    peaks = angles[maxima[np.argsort(spectrum[maxima])[-2:]]]
    print(f"  {name:9s} two strongest bearings: "
          f"{np.sort(peaks)[0]:+.1f}°, {np.sort(peaks)[1]:+.1f}° "
          f"(true {true_angles[0]:+.0f}°, {true_angles[1]:+.0f}°)")

fig, ax = plt.subplots(figsize=(11, 6))
for label, spectrum in (('Bartlett (conventional)', conventional),
                        ('MVDR / Capon', capon), ('MUSIC', music)):
    uacpy.plot.plot_beam_power(
        angles, spectrum, ax=ax, label=label, dynamic_range_dB=50,
        title='Direction-of-arrival spectra (16-element line array)')
for angle in true_angles:
    ax.axvline(angle, color='k', ls='--', alpha=0.4)
ax.legend()

fig.tight_layout()
fig.savefig(OUT / 'example_29_array_processing.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)
