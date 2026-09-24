"""The same acoustics, on data that never came from a model.

Every quantity here is also reachable as a method on a uacpy carrier — a
``Field``, an ``Arrivals``, a ``Modes``. That is the convenient route when a
uacpy model produced the data. It is the wrong route when the data came from
somewhere else: a chirp sounding, a ``.mat`` file, a hydrophone record, a
mode set from another solver, or an arrival list you wrote down.

So each computation lives in a **function over plain arrays**, and the
carrier's method is a wrapper around it. Nothing below constructs a model or
a carrier: the inputs are arrays, typed in as a measurement would arrive.

Uses: rms_delay_spread · energy_support · coherence_bandwidth ·
channel_regime · arrival_transfer_function · gate_transfer_function ·
broadband_propagation_loss · uniform_frequency_step · tone_phasor · spl ·
peak_level · sound_exposure_level · impulse_response_from_transfer_function ·
transfer_function_from_impulse_response · pulse_shaped_taps ·
modal_attenuation · modal_field · hankel_transform · wavenumber_taper ·
coherence_factor · tl_rmse_on_shared_ranges
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.acoustic_signal import (arrival_transfer_function,
                                   broadband_propagation_loss,
                                   channel_regime, coherence_bandwidth,
                                   coherence_factor,
                                   energy_support, gate_transfer_function,
                                   impulse_response_from_transfer_function,
                                   rms_delay_spread, tone_phasor,
                                   transfer_function_from_impulse_response,
                                   uniform_frequency_step)
from uacpy.comms import pulse_shaped_taps
from uacpy.core.bottom import BoundaryProperties
from uacpy.core.acoustics import (hankel_transform, modal_attenuation,
                                  modal_field, peak_level,
                                  sound_exposure_level, spl,
                                  wavenumber_taper)
from uacpy.metrics import tl_rmse_on_shared_ranges

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

# ── A. A measured power delay profile ─────────────────────────────────────
# Four arrivals as a chirp sounding would report them: a delay and a power.
# No model, no Arrivals object — two arrays.
delays_s = np.array([0.0, 3.5e-3, 11.0e-3, 30.0e-3])
powers = np.array([1.0, 0.30, 0.096, 0.005])      # |a|**2, any consistent scale

spread = rms_delay_spread(delays_s, powers)
span99 = energy_support(delays_s, powers, 0.99)
span = energy_support(delays_s, powers, 0.999)
bc = coherence_bandwidth(delays_s, powers)

print(f"A. rms delay spread      {spread * 1e3:6.2f} ms")
print(f"   99%  energy span      {span99 * 1e3:6.2f} ms")
print(f"   99.9% energy span     {span * 1e3:6.2f} ms   "
      f"(peak-to-peak {np.ptp(delays_s) * 1e3:.1f} ms)")
# The last arrival carries 0.4% of the energy, so it is inside the 99.9%
# span and outside the 99% one. Which fraction you ask for is the question
# "how much of the tail does my processing have to cover", and the answer
# moves by 19 ms between two reasonable choices.
print(f"   coherence bandwidth   {bc:6.1f} Hz")

# The span is what a synthesis window has to cover; the spread is a width.
# They differ because one faint straggler sets the peak-to-peak and neither
# of the other two.
for rate in (200.0, 2000.0):
    verdict = channel_regime(delays_s, powers, rate)
    print(f"   at {rate:6.0f} Bd: {verdict}")

# ── B. The transfer function of that arrival list ─────────────────────────
# Amplitudes are MAGNITUDES here; a sign belongs in phases_rad as pi.
freqs = np.arange(2000.0, 8000.01, 20.0)
amplitudes = np.sqrt(powers)
H = arrival_transfer_function(freqs, amplitudes, delays_s,
                              phases_rad=[0.0, np.pi, 0.0, np.pi])

df = uniform_frequency_step(freqs)
print(f"\nB. df = {df:g} Hz -> the response repeats every "
      f"{1.0 / df * 1e3:.0f} ms; the profile ends at "
      f"{delays_s[-1] * 1e3:.0f} ms, so it fits")

# Keep only the paths a 4 ms pulse could overlap, and see what that costs.
gated = gate_transfer_function(H, freqs, 2.0e-3, origin=0.0)
band_loss = broadband_propagation_loss(H)
gated_loss = broadband_propagation_loss(gated)
# The amplitudes above are RELATIVE (the first path is 1), so these are
# band averages relative to one unit path, not absolute losses: a negative
# number means the four paths add constructively across the band on
# average. Feed it a model's H(f) and the number is an absolute loss.
print(f"   band average, all paths                 {band_loss:6.2f} dB")
print(f"   the same, gated to +-2 ms of the first  {gated_loss:6.2f} dB")
print("   (relative to one unit-amplitude path)")

# ── C. A recorded waveform ────────────────────────────────────────────────
# As a hydrophone would deliver it: pressure in Pa and a sample rate.
fs = 20000.0
t = np.arange(int(0.25 * fs)) / fs
f_tone = 3137.0                      # deliberately NOT on a DFT bin
envelope = np.exp(-((t - 0.10) / 0.03) ** 2)
record = 0.8 * envelope * np.sin(2 * np.pi * f_tone * t + 0.7)

print(f"\nC. rms level   {spl(record):7.2f} dB re 1 uPa")
print(f"   peak level  {peak_level(record):7.2f} dB re 1 uPa")
print(f"   exposure    {sound_exposure_level(record, 1 / fs):7.2f} "
      f"dB re 1 uPa^2 s")

# One tone out of that record, evaluated AT the frequency. Sampling the
# nearest DFT bin instead is wrong by a growing amount across the bin, and
# the phase goes first: the level is still within 1.5 dB when the phase is
# already 90 degrees out.
phasor = tone_phasor(record, t, f_tone)
bin_index = int(np.argmin(np.abs(np.fft.rfftfreq(t.size, 1 / fs) - f_tone)))
window = np.hanning(t.size)
nearest = 2.0 * np.fft.rfft(record * window)[bin_index] / np.sum(window)
print(f"   tone at {f_tone} Hz: |A| = {abs(phasor):.5f}, "
      f"phase = {np.degrees(np.angle(phasor)):+.2f} deg")
print(f"   nearest-bin instead:  {20 * np.log10(abs(nearest / phasor)):+.3f}"
      f" dB and {np.degrees(np.angle(nearest / phasor)):+.1f} deg away")

# ── D. There and back ─────────────────────────────────────────────────────
# The two array-level transforms are exact inverses of each other.
t_ir, h = impulse_response_from_transfer_function(H, freqs, 20000.0)
f_back, H_back = transfer_function_from_impulse_response(
    h, 20000.0, band=(freqs[0], freqs[-1]))
reference = (np.interp(f_back, freqs, H.real)
             + 1j * np.interp(f_back, freqs, H.imag))
print(f"\nD. H -> h -> H   |ratio| {np.abs(H_back).mean() / np.abs(reference).mean():.6f}"
      f"   max|err| {np.abs(H_back - reference).max():.2e}")

# The same arrivals as a modem would see them, through its own pulse.
tap_times, taps = pulse_shaped_taps(amplitudes, delays_s, 2000.0,
                                    pulse='rrc', rolloff=0.25, span=8)
print(f"   as {taps.size} root-raised-cosine taps at 2000 Bd, "
      f"first at {tap_times[0] * 1e3:.1f} ms")

# ── E. A mode set written down by hand ────────────────────────────────────
# An ideal waveguide: sin(m pi z / D) shapes and the dispersion that goes
# with them. No solver ran.
depth_m, c0, freq = 100.0, 1500.0, 100.0
z = np.linspace(0.0, depth_m, 201)
kz = (np.arange(4) + 0.5) * np.pi / depth_m
k = np.sqrt((2 * np.pi * freq / c0) ** 2 - kz ** 2).astype(complex)
psi = np.sin(np.outer(z, kz))

# kz = (m + 1/2) pi / D puts a pressure release at the surface and a
# RIGID seabed at D, so these shapes have no evanescent tail and the
# perturbation is exact. Saying so is also what silences the notice that
# otherwise warns the returned Im(k) is an upper bound.
# 1e-4 dB/m is an EXAGGERATED volume attenuation, chosen so the effect is
# visible over 20 km: real seawater at 100 Hz is near 1e-6 dB/m, which would
# cost 0.02 dB across the whole panel. The perturbation is linear in alpha,
# so the shape of the answer does not depend on the value.
alpha_m = modal_attenuation(k, psi, z, 1e-4, frequency=freq,
                            bottom=BoundaryProperties(acoustic_type='rigid'))
print(f"\nE. modal attenuation (Np/m): "
      f"{np.array2string(alpha_m, precision=8)}")
# The four are within 3.5% of each other: these shapes all fill the same
# water column, so they see the same alpha(z). A guide whose modes reach
# different depths — a duct, a sediment-loaded bottom — is where the spread
# becomes mode stripping.
print(f"   spread across the four modes: "
      f"{100 * (alpha_m.max() / alpha_m.min() - 1):.1f}%")
print(f"   excess loss at 20 km: "
      f"{8.686 * alpha_m.min() * 20e3:.1f} to "
      f"{8.686 * alpha_m.max() * 20e3:.1f} dB")

ranges_m = np.linspace(100.0, 20000.0, 400)
psi_zs = np.array([np.interp(25.0, z, psi[:, m]) for m in range(4)])
psi_zr = np.column_stack([np.interp([50.0], z, psi[:, m]) for m in range(4)])
p_lossless = modal_field(k, psi_zs, psi_zr, ranges_m)
p_lossy = modal_field(k + 1j * alpha_m, psi_zs, psi_zr, ranges_m)

# ── F. A wavenumber-domain kernel ─────────────────────────────────────────
# What a spectral code hands back before the range transform: G(k) on a
# uniform wavenumber grid. Two poles standing in for two trapped modes.
k_grid = np.linspace(0.05, 0.50, 1024)
G_k = np.zeros((1, k_grid.size), dtype=complex)
for k_pole, strength in ((0.38, 1.0), (0.30, 0.6)):
    G_k[0] += strength / ((k_grid - k_pole) + 0.0015j)

# The c_min/c_max window is ones across [w/cmax, w/cmin] and rolls off with
# a Hanning skirt to ZERO AT THE ENDS OF THE GRID — it is not a band-pass
# that cuts whatever falls outside. What it suppresses is the truncation
# discontinuity where the stored spectrum stops, which is what rings in the
# range transform. A pole just outside the band is barely touched: measured
# below, the in-band pole keeps 1.0000 and the out-of-band one still keeps
# 0.9698.
taper = wavenumber_taper(k_grid, freq, 1400.0, 1900.0)
r_out = np.linspace(200.0, 6000.0, 300)
p_full = hankel_transform(G_k, k_grid, r_out, atten=0.0)
p_cut = hankel_transform(G_k * taper, k_grid, r_out, atten=0.0)
print(f"\nF. dk = {k_grid[1] - k_grid[0]:.2e} 1/m -> alias period "
      f"{2 * np.pi / (k_grid[1] - k_grid[0]) / 1e3:.1f} km; "
      f"panel ends at {r_out[-1] / 1e3:.1f} km")
print(f"   taper keeps {int(np.count_nonzero(taper > 0.99))} of "
      f"{k_grid.size} wavenumbers above 0.99, and reaches "
      f"{taper[0]:.3f}/{taper[-1]:.3f} at the grid ends")
for pole in (0.38, 0.30):
    i = int(np.argmin(np.abs(k_grid - pole)))
    print(f"   pole at k = {pole:.2f} (c = {2 * np.pi * freq / pole:.0f} m/s)"
          f" keeps {taper[i]:.4f}")

# ── G. Two models on different range axes ─────────────────────────────────
# tl_rmse refuses a grid mismatch, which is right when the question is "do
# these two runs agree". Comparing MODELS means different axes by nature.
coarse_r = np.linspace(500.0, 6000.0, 12)
fine_r = np.linspace(500.0, 6000.0, 300)
model_a = uacpy.core.results.Field(
    data=(np.interp(fine_r, r_out, np.abs(p_full[0]))
          ).astype(complex).reshape(1, -1),
    coords={'depth': np.array([50.0]), 'range': fine_r}, frequencies=freq)
model_b = uacpy.core.results.Field(
    data=(np.interp(coarse_r, r_out, np.abs(p_cut[0]))
          ).astype(complex).reshape(1, -1),
    coords={'depth': np.array([50.0]), 'range': coarse_r}, frequencies=freq)
print(f"\nG. untapered vs tapered, resampled onto the coarser axis: "
      f"{tl_rmse_on_shared_ranges(model_a, model_b, depth=50.0):.2f} dB rms")

# The k of 1/(k tau_rms) is a convention, so it is named rather than assumed.
for name in ('inverse_spread', 'rappaport_0.9'):
    kf = coherence_factor(name)
    print(f"   coherence convention {name:15s} k = {kf:4.0f} -> "
          f"{coherence_bandwidth(delays_s, powers, convention=name):7.1f} Hz")

# ── the figure ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].stem(delays_s * 1e3, powers, basefmt=' ')
axes[0, 0].axvspan(0.0, span99 * 1e3, color='tab:orange', alpha=0.15,
                   label=f'99% energy span {span99 * 1e3:.1f} ms')
axes[0, 0].axvline(spread * 1e3, color='tab:red', ls='--',
                   label=f'rms spread {spread * 1e3:.1f} ms')
axes[0, 0].set(xlabel='Delay (ms)', ylabel='Power (linear)',
               title='A. A measured power delay profile')
axes[0, 0].legend(fontsize=8)

axes[0, 1].plot(freqs * 1e-3, 20 * np.log10(np.abs(H)), lw=0.8,
                label='all four paths')
axes[0, 1].plot(freqs * 1e-3, 20 * np.log10(np.abs(gated)), lw=1.2,
                label='gated to ±2 ms')
axes[0, 1].set(xlabel='Frequency (kHz)', ylabel='|H| (dB)',
               title=f'B. Its transfer function — band average '
                     f'{band_loss:.1f} dB, gated {gated_loss:.1f} dB '
                     f'(re one unit path)')
axes[0, 1].legend(fontsize=8)

axes[1, 0].plot(t * 1e3, record, lw=0.6)
axes[1, 0].set(xlabel='Time (ms)', ylabel='Pressure (Pa)',
               title=f'C. A record: rms {spl(record):.1f}, peak '
                     f'{peak_level(record):.1f}, SEL '
                     f'{sound_exposure_level(record, 1 / fs):.1f} dB')

axes[1, 1].plot(ranges_m * 1e-3, -20 * np.log10(np.abs(p_lossless[0])),
                lw=0.9, label='lossless')
axes[1, 1].plot(ranges_m * 1e-3, -20 * np.log10(np.abs(p_lossy[0])),
                lw=0.9, label='with modal attenuation')
axes[1, 1].invert_yaxis()
axes[1, 1].set(xlabel='Range (km)', ylabel='Transmission loss (dB)',
               title='E. A hand-written mode set, propagated')
axes[1, 1].legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT / 'example_46_channel_numbers_without_a_model.png', dpi=200,
            bbox_inches='tight')
plt.close(fig)
print(f"\nwrote {OUT / 'example_46_channel_numbers_without_a_model.png'}")
