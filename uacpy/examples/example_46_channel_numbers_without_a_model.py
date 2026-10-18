"""The same acoustics, on data that never came from a model.

Every quantity here is also a method on a uacpy carrier (``Field``,
``Arrivals``, ``Modes``); each method wraps a function over plain arrays,
and those functions take data from anywhere: a chirp sounding, a ``.mat``
file, a hydrophone record, another solver's modes. The inputs below are
arrays typed in as a measurement would arrive.

Uses: rms_delay_spread · energy_support · coherence_bandwidth ·
COHERENCE_BANDWIDTH_FACTORS · channel_regime · arrival_transfer_function ·
truncate_response · broadband_loss · uniform_frequency_step ·
tone_phasor · spl · peak_level · sound_exposure_level ·
impulse_response_from_transfer_function ·
transfer_function_from_impulse_response · pulse_shaped_taps ·
modal_attenuation · modal_field · hankel_transform · wavenumber_taper ·
tl_rmse_on_shared_ranges
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.acoustic_signal import (COHERENCE_BANDWIDTH_FACTORS,
                                   arrival_transfer_function,
                                   broadband_loss,
                                   channel_regime, coherence_bandwidth,
                                   energy_support, truncate_response,
                                   impulse_response_from_transfer_function,
                                   rms_delay_spread, tone_phasor,
                                   transfer_function_from_impulse_response,
                                   uniform_frequency_step)
from uacpy.comms import pulse_shaped_taps
from uacpy.acoustics import (hankel_transform, modal_attenuation,
                             modal_field, peak_level, sound_exposure_level,
                             spl, wavenumber_taper)
from uacpy.metrics import tl_rmse_on_shared_ranges

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

# A. A measured power delay profile: four delays and their powers
delays = np.array([0.0, 3.5e-3, 11.0e-3, 30.0e-3])
powers = np.array([1.0, 0.30, 0.096, 0.005])      # |a|**2, any consistent scale
spread = rms_delay_spread(delays, powers)
span99 = energy_support(delays, powers, 0.99)
print(f"A. rms delay spread {spread * 1e3:.2f} ms, energy span "
      f"{span99 * 1e3:.2f} ms (99 %) / "
      f"{energy_support(delays, powers, 0.999) * 1e3:.2f} ms (99.9 %)")
for name, factor in COHERENCE_BANDWIDTH_FACTORS.items():
    print(f"   coherence bandwidth, {name:14s} (k = {factor:4.0f}): "
          f"{coherence_bandwidth(delays, powers, convention=name):7.1f} Hz")
for rate in (200.0, 2000.0):
    print(f"   at {rate:6.0f} Bd: {channel_regime(delays, powers, rate)}")

# B. Its transfer function, all paths and gated to the first 2 ms
freqs = np.arange(2000.0, 8000.01, 20.0)
amplitudes = np.sqrt(powers)
H = arrival_transfer_function(freqs, amplitudes, delays,
                              phases=[0.0, np.pi, 0.0, np.pi])
gated = truncate_response(H, frequencies=freqs, duration=2.0e-3, origin=0.0)
band_loss = broadband_loss(H)
gated_loss = broadband_loss(gated)
print(f"\nB. df = {uniform_frequency_step(freqs):g} Hz; band average "
      f"{band_loss:.2f} dB, gated {gated_loss:.2f} dB (re one unit path)")

# C. A recorded waveform: levels, and one tone evaluated at its own frequency
fs = 20000.0
t = np.arange(int(0.25 * fs)) / fs
f_tone = 3137.0                                   # not on a DFT bin
record = 0.8 * np.exp(-((t - 0.10) / 0.03) ** 2) * np.sin(2 * np.pi * f_tone * t + 0.7)
phasor = tone_phasor(record, t, f_tone)
print(f"\nC. rms {spl(record):.2f} dB re 1 µPa, peak {peak_level(record):.2f} "
      f"dB re 1 µPa, SEL {sound_exposure_level(record, fs):.2f} dB re 1 µPa²·s")
print(f"   tone at {f_tone} Hz: |A| = {abs(phasor):.5f}, "
      f"phase {np.degrees(np.angle(phasor)):+.2f} deg")

# D. Transfer function to impulse response and back, and the taps a modem sees
t_ir, h = impulse_response_from_transfer_function(H, frequencies=freqs,
                                                  sample_rate=fs)
f_back, H_back = transfer_function_from_impulse_response(
    h, fs, band=(freqs[0], freqs[-1]))
reference = np.interp(f_back, freqs, H.real) + 1j * np.interp(f_back, freqs, H.imag)
shaped = pulse_shaped_taps(amplitudes, delays, 2000.0,
                           pulse='rrc', rolloff=0.25, span=8)
print(f"\nD. H -> h -> H: max|error| {np.abs(H_back - reference).max():.2e}; "
      f"{shaped.taps.size} root-raised-cosine taps at 2000 Bd")

# E. A mode set written down by hand: an ideal guide over a rigid seabed
depth, c0, freq, rho = 100.0, 1500.0, 100.0, 1.0
z = np.linspace(0.0, depth, 201)
kz = (np.arange(4) + 0.5) * np.pi / depth
k = np.sqrt((2 * np.pi * freq / c0) ** 2 - kz ** 2).astype(complex)
psi = np.sqrt(2.0 * rho / depth) * np.sin(np.outer(z, kz))   # ∫ψ²/ρ dz = 1, JKPS Eq. 5.169
alpha_m = modal_attenuation(k, psi, z, 1e-4, frequency=freq, sound_speed_z=c0,
                            density_z=rho,
                            bottom=uacpy.BoundaryProperties(acoustic_type='rigid'))
print(f"\nE. modal attenuation for 1e-4 dB/m: "
      f"{np.array2string(alpha_m, precision=8)} Np/m")
ranges = np.linspace(100.0, 20000.0, 400)
psi_zs = np.array([np.interp(25.0, z, psi[:, m]) for m in range(4)])
psi_zr = np.column_stack([np.interp([50.0], z, psi[:, m]) for m in range(4)])
lossless = uacpy.Field(
    data=modal_field(k, psi_zs, psi_zr, ranges, source_density=rho),
    coords={'depth': np.array([50.0]), 'range': ranges}, frequencies=freq)
lossy = uacpy.Field(
    data=modal_field(k + 1j * alpha_m, psi_zs, psi_zr, ranges,
                     source_density=rho),
    coords={'depth': np.array([50.0]), 'range': ranges}, frequencies=freq)

# F. A wavenumber-domain kernel, tapered and transformed to range
k_grid = np.linspace(0.05, 0.50, 1024)
G_k = np.zeros((1, k_grid.size), dtype=complex)
for k_pole, strength in ((0.38, 1.0), (0.30, 0.6)):
    G_k[0] += strength / ((k_grid - k_pole) + 0.0015j)
taper = wavenumber_taper(k_grid, freq, 1400.0, 1900.0)
r_out = np.linspace(200.0, 6000.0, 300)
p_full = hankel_transform(G_k, k_grid, r_out, attenuation=0.0)
p_cut = hankel_transform(G_k * taper, k_grid, r_out, attenuation=0.0)
for pole in (0.38, 0.30):
    print(f"F. pole at c = {2 * np.pi * freq / pole:.0f} m/s keeps "
          f"{taper[int(np.argmin(np.abs(k_grid - pole)))]:.4f} of the taper")

# G. Two fields on different range axes, compared on the coarser one
coarse_r, fine_r = np.linspace(500.0, 6000.0, 12), np.linspace(500.0, 6000.0, 300)
model_a = uacpy.Field(data=np.interp(fine_r, r_out, np.abs(p_full[0]))
                      .astype(complex).reshape(1, -1),
                      coords={'depth': np.array([50.0]), 'range': fine_r},
                      frequencies=freq)
model_b = uacpy.Field(data=np.interp(coarse_r, r_out, np.abs(p_cut[0]))
                      .astype(complex).reshape(1, -1),
                      coords={'depth': np.array([50.0]), 'range': coarse_r},
                      frequencies=freq)
print(f"G. untapered vs tapered: "
      f"{tl_rmse_on_shared_ranges(model_a, model_b, depth=50.0):.2f} dB rms")

# Plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].stem(delays * 1e3, powers, basefmt=' ')
axes[0, 0].axvspan(0.0, span99 * 1e3, color='tab:orange', alpha=0.15,
                   label=f'99 % energy span {span99 * 1e3:.1f} ms')
axes[0, 0].set(xlabel='Delay (ms)', ylabel='Power (linear)',
               title=f'A. A measured power delay profile '
                     f'(rms spread {spread * 1e3:.1f} ms)')
axes[0, 0].legend(fontsize=8)

axes[0, 1].plot(freqs * 1e-3, 20 * np.log10(np.abs(H)), lw=0.8,
                label='all four paths')
axes[0, 1].plot(freqs * 1e-3, 20 * np.log10(np.abs(gated)), lw=1.2,
                label='gated to ±2 ms')
axes[0, 1].set(xlabel='Frequency (kHz)', ylabel='|H| (dB)',
               title=f'B. Its transfer function — band average '
                     f'{band_loss:.1f} dB, gated {gated_loss:.1f} dB')
axes[0, 1].legend(fontsize=8)

uacpy.plot.plot_waveform(record, fs, axes[1, 0], time_units='ms', lw=0.6,
                         title=f'C. A record: rms {spl(record):.1f}, peak '
                               f'{peak_level(record):.1f}, SEL '
                               f'{sound_exposure_level(record, fs):.1f} dB')

lossless.at(depth=50.0).plot(ax=axes[1, 1], label='lossless')
lossy.at(depth=50.0).plot(ax=axes[1, 1], label='with modal attenuation',
                          title='E. A hand-written mode set, propagated')
axes[1, 1].legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT / 'example_46_channel_numbers_without_a_model.png', dpi=200,
            bbox_inches='tight')
plt.close(fig)
print(f"\nFigures written to {OUT}")
