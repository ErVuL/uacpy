"""Adaptive and high-resolution array processing.

Two plane waves 14° apart on a 16-element half-wavelength line array, resolved
three ways: the conventional Bartlett beamformer, the adaptive MVDR (Capon)
estimator, and the subspace MUSIC estimator. Bartlett's beamwidth sets the
classical resolution limit; the other two are built to beat it.

Uses: acoustic_signal.steering_vectors · sample_covariance ·
bartlett_spectrum · mvdr_spectrum · music_spectrum
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
from uacpy.acoustic_signal import (bartlett_spectrum, music_spectrum,
                                   mvdr_spectrum, sample_covariance,
                                   steering_vectors)

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

covariance = sample_covariance(data)
angles = np.linspace(-40, 40, 801)
replicas = steering_vectors(positions, angles, frequency, c)
bartlett = bartlett_spectrum(covariance, replicas)
capon = mvdr_spectrum(covariance, replicas)
music = music_spectrum(covariance, replicas, n_sources=2)

for name, spectrum in (('Bartlett', bartlett), ('MVDR', capon),
                       ('MUSIC', music)):
    peaks = angles[np.argsort(spectrum)[-2:]]
    print(f"  {name:9s} two strongest bearings: "
          f"{np.sort(peaks)[0]:+.1f}°, {np.sort(peaks)[1]:+.1f}° "
          f"(true {true_angles[0]:+.0f}°, {true_angles[1]:+.0f}°)")

fig, ax = plt.subplots(figsize=(11, 6))
for label, spectrum in (('Bartlett (conventional)', bartlett),
                        ('MVDR / Capon', capon), ('MUSIC', music)):
    ax.plot(angles, 10 * np.log10(spectrum / spectrum.max()), label=label)
for angle in true_angles:
    ax.axvline(angle, color='k', ls='--', alpha=0.4)
ax.set_title('Direction-of-arrival spectra (16-element line array)',
             fontweight='bold')
ax.set_xlabel('Angle from broadside (deg)')
ax.set_ylabel('Normalised power (dB)')
ax.set_ylim(-50, 2)
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / 'example_29_array_processing.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)
