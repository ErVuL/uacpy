"""Matched-field source localization with KRAKEN replicas.

Find an unknown source in range AND depth by matched-field processing: scan
candidate positions, correlate the modeled pressure at the array (the
*replica*) against the measured cross-spectral density matrix, and take the
peak.

The replicas come straight out of a KRAKEN mode set. The eigenpairs (k_m, φ_m)
depend only on the environment, so the modes are computed ONCE and every
candidate is a cheap analytic re-sum of the far-field modal series

    p(r, z) = r^-1/2 · Σ_m φ_m(z_s) φ_m(z) k_m^-1/2 exp(-i k_m r)

Scenario: isovelocity 100 m Pekeris waveguide at 150 Hz, a 16-element vertical
array, a source hidden at (62 m, 3.2 km) seen through 50 snapshots at 10 dB
SNR, localized by the Bartlett (linear) and MVDR (Capon) processors.

Uses: Kraken.compute_modes · sonar.synthesize_replica / replica_bank · csdm ·
bartlett · mvdr · a Field of kind 'ambiguity' through plot_field ·
plot.shared_colorbar
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.core.results import Field
from uacpy.sonar import bartlett, csdm, mvdr, replica_bank, synthesize_replica

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

env = uacpy.Environment(
    name="Pekeris 100 m",
    bathymetry=100.0,
    ssp=uacpy.SoundSpeedProfile.from_pairs(np.array([[0, 1500.0],
                                                     [100, 1500.0]])),
    bottom=uacpy.Bottom.from_halfspaces(
        np.array([0.0]), sound_speed=np.array([1800.0]),
        density=np.array([1.8]), attenuation=np.array([0.2]),
        acoustic_type='half-space'),
)
# The Source depth here only sets the frequency for the mode solve; MFP scans
# source depth itself.
source = uacpy.Source(depths=25.0, frequencies=150.0)
array_depths = np.linspace(5, 95, 16)               # 16-element vertical array

modes = uacpy.Kraken().compute_modes(env, source)
print(f"  {modes.n_modes} modes at {source.frequencies[0]:.0f} Hz, "
      f"computed once and reused for every replica")

# Array data for a hidden source: one replica, random phase per snapshot, noise.
true_depth, true_range = 62.0, 3200.0
truth = synthesize_replica(modes, true_depth, true_range, array_depths)
rng = np.random.default_rng(1)
n_snapshots, snr_dB = 50, 10.0
signal = truth[:, None] * np.exp(1j * rng.uniform(0, 2 * np.pi, n_snapshots))
noise_power = np.mean(np.abs(truth) ** 2) / 10 ** (snr_dB / 10)
noise = np.sqrt(noise_power / 2) * (
    rng.standard_normal((array_depths.size, n_snapshots))
    + 1j * rng.standard_normal((array_depths.size, n_snapshots)))
covariance = csdm(signal + noise)

candidate_depths = np.linspace(5, 95, 91)
candidate_ranges = np.linspace(500, 5000, 121)
bank = replica_bank(modes, array_depths, candidate_depths, candidate_ranges)
# Diagonal loading as a fraction of the average eigenvalue (the mvdr default,
# named here because it sets the trade-off): smaller sharpens the Capon peak
# but makes it brittle to environmental mismatch, larger relaxes the surface
# back toward Bartlett.
surfaces = {'Bartlett': bartlett(covariance, bank),
            'MVDR': mvdr(covariance, bank, diagonal_loading=1e-2)}

print(f"  hidden source at {true_depth:.0f} m, {true_range / 1e3:.1f} km")
for name, surface in surfaces.items():
    depth_index, range_index = np.unravel_index(np.argmax(surface),
                                                surface.shape)
    print(f"  {name:8s} estimate: {candidate_depths[depth_index]:5.1f} m, "
          f"{candidate_ranges[range_index] / 1e3:.2f} km")

# Each surface becomes a Field of kind 'ambiguity' — normalised power in dB re
# its own maximum — so the library owns the rendering: the turbo map, the
# "Normalised power (dB re max)" label, depth downward, range in km, and cell
# edges computed from the candidate positions rather than an extent built by
# hand (which is how a surface ends up drawn half a cell off the grid it was
# computed on).
ambiguity = {
    name: Field(data=10 * np.log10(np.clip(surface / surface.max(), 1e-3,
                                           None)),
                coords={'depth': candidate_depths, 'range': candidate_ranges},
                model='MFP', frequencies=source.frequencies,
                metadata={'kind': 'ambiguity'})
    for name, surface in surfaces.items()}

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, (name, field) in zip(axes, ambiguity.items()):
    uacpy.plot_field(field, ax, vmin=-15, vmax=0, show_colorbar=False,
                     title=f"{name} ambiguity surface")
    ax.plot(true_range / 1e3, true_depth, 'w*', ms=16, mec='k', label='truth')
    depth_index, range_index = np.unravel_index(
        np.argmax(surfaces[name]), surfaces[name].shape)
    ax.plot(candidate_ranges[range_index] / 1e3,
            candidate_depths[depth_index], 'o', mfc='none', mec='w', ms=12,
            mew=2, label='estimate')
    ax.legend(loc='upper right', fontsize=8)
# Both panels are on the same -15..0 dB window, so one bar describes both.
uacpy.plot.shared_colorbar(fig, axes, label='Normalised power [dB re max]')
fig.suptitle("Matched-field localization — KRAKEN replicas "
             "(150 Hz, 16-element VLA)")
fig.savefig(OUT / 'example_38_matched_field.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)
