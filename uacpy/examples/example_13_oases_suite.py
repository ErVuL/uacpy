"""The OASES suite — OAST, OASN, OASR, OASP on one environment.

Four wavenumber-integration modules, each answering a different question about
the same elastic seabed: transmission loss (OAST), the array cross-spectral
covariance of a surface-generated noise field (OASN), plane-wave reflection
coefficients (OASR), and a wideband transfer function that can be turned into a
time trace (OASP).

OASES is a global-matrix suite, so all four solve a horizontally stratified
medium; uacpy collapses any range dependence and warns.

Uses: OAST · OASN(surface_noise_level=).compute_covariance · OASR(angles=) ·
OASP(n_time_samples=, freq_max=) with RunMode.BROADBAND · every result's own
.plot() · Field.at(frequency=) · Field.synthesize_time_series
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
    name="OASES demonstration",
    bathymetry=100,
    ssp=uacpy.SoundSpeedProfile.from_pairs([(0, 1500), (100, 1520)]),
    bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                    sound_speed=1700, shear_speed=400,
                                    density=1.8, attenuation=0.5),
)
source = uacpy.Source(depths=50, frequencies=100)
receiver = uacpy.Receiver(depths=np.linspace(5, 95, 40),
                          ranges=np.linspace(500, 15000, 60))

# OAST — transmission loss.
tl = uacpy.OAST().run(env, source, receiver)
print(f"  OAST TL {np.nanmin(tl.dB):.1f}-{np.nanmax(tl.dB):.1f} dB")
fig, _ = uacpy.plot_field(tl, env=env)
fig.savefig(OUT / 'example_13_oast_tl.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# OASN — spatial covariance. Without surface_noise_level the covariance
# collapses to the 0 dB white-noise floor (the identity); 70 dB is about the
# Wenz amplitude at 100 Hz.
covariance = uacpy.OASN(surface_noise_level=70.0).compute_covariance(
    env, source, receiver)
print(f"  OASN covariance: {covariance.n_receivers} receivers × "
      f"{covariance.n_frequencies} frequency")
fig, _ = covariance.plot()
fig.savefig(OUT / 'example_13_oasn_covariance.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)

# OASR — plane-wave reflection coefficients, which know how to plot themselves.
reflection = uacpy.OASR(angles=np.linspace(0, 90, 91)).run(env, source,
                                                           receiver)
one_frequency = (reflection.at(
    frequency=reflection.frequencies[len(reflection.frequencies) // 2])
    if reflection.is_broadband else reflection)
print(f"  OASR |R| in [{one_frequency.R.min():.3f}, "
      f"{one_frequency.R.max():.3f}] at {one_frequency.f0:.1f} Hz")
fig, _ = one_frequency.plot(
    show_phase=True,
    title=f"OASR {reflection.metadata.get('reflection_type', 'P-P')} "
          f"@ {one_frequency.f0:.1f} Hz")
fig.savefig(OUT / 'example_13_oasr_reflection.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)

# OASP — a wideband transfer function. The pulse synthesis below is a DFT over
# the frequency grid, so its time window is 1/df = (n_time_samples/2)/freq_max.
# Every receiver's travel time must fit inside that window or far arrivals wrap
# into early bins: 1024/2 / 120 Hz = 4.27 s of window against 5000 m ≈ 3.3 s of
# travel, so it fits.
transfer = uacpy.OASP(n_time_samples=1024, freq_max=120).run(
    env, source,
    uacpy.Receiver(depths=np.linspace(5, 95, 20),
                   ranges=np.linspace(500, 5000, 30)),
    run_mode=uacpy.RunMode.BROADBAND)
frequencies = transfer.coords['frequency']
centre = float(frequencies[len(frequencies) // 2])
print(f"  OASP H(f) {transfer.data.shape} over "
      f"{frequencies[0]:.0f}-{frequencies[-1]:.0f} Hz")

fig, _ = uacpy.plot_field(transfer.at(frequency=centre), env=env)
fig.savefig(OUT / 'example_13_oasp_tl.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# One synthesized trace: a Gaussian-windowed sinusoid at the centre frequency.
# gaussian_pulse's width parameter is σ·√2, so the envelope is
# exp(-(t - t0)² / 2σ²).
fs = 4.0 * float(frequencies[-1])
t = np.arange(64) / fs
sigma = 64 / (8.0 * fs)
pulse = (np.sin(2 * np.pi * centre * (t - t[-1] / 2))
         * gaussian_pulse(t, t[-1] / 2, sigma * np.sqrt(2)))
trace = transfer.synthesize_time_series(source_waveform=pulse,
                                        sample_rate=fs).at(
    depth=float(source.depths[0]), range=5000.0)
fig, _ = uacpy.plot_field(trace)
fig.savefig(OUT / 'example_13_oasp_trace.png', dpi=150, bbox_inches='tight')
plt.close(fig)
