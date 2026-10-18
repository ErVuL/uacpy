"""The level a signal with bandwidth or duration reaches, as a map.

A model's transmission loss is the continuous-wave answer. One broadband run
of a Lloyd mirror (a source 20 m below the surface in deep water) gives the
maps a real signal meets: the broadband propagation loss over a band (Ainslie
Eq. 11.46) against the incoherent run mode (Eq. 11.47), the sound exposure
level and peak sound pressure level of a burst at a stated source level, and
the received waveform at one receiver. docs/guide/results.md and
docs/theory/broadband_products.md give the reasoning.

Uses: RunMode.BROADBAND / INCOHERENT_TL / ARRIVALS · Field.at ·
Field.window · Field.broadband_loss(source_waveform=) ·
Field.sound_exposure_level · Field.peak_sound_pressure_level ·
metrics.tl_difference · Field.to_time_trace(source_waveform=) ·
Arrivals.synthesis_band / energy_support · acoustic_signal.tone_burst /
lfm_chirp · plot_field · plot_waveform
"""

import os
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.acoustic_signal import lfm_chirp, tone_burst
from uacpy.plot import plot_field, plot_waveform

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

SOURCE_DEPTH, TONE = 20.0, 1000.0    # m, Hz
SOURCE_LEVEL = 190.0                 # dB re 1 µPa @ 1 m
RATE = 8000.0                        # waveform sample rate (Hz)
LOOK_DEPTH, LOOK_RANGE = 150.0, 1500.0

# Env definition: deep isovelocity water; +-60 deg keeps the bottom return off the panel
env = uacpy.Environment(name='Deep water', bathymetry=5000.0, ssp=1500.0)
model = uacpy.Bellhop(backend='fortran', n_beams=12001,
                      launch_angles_deg=(-60.0, 60.0))
depths = np.linspace(2.0, 300.0, 150)
ranges = np.linspace(300.0, 2500.0, 350)
receiver = uacpy.Receiver(depths=depths, ranges=ranges)

# One broadband run, read as a CW map and as two band averages
band = np.arange(800.0, 1201.0, 20.0)
H = model.run(env, uacpy.Source(depths=SOURCE_DEPTH, frequencies=band),
              receiver, run_mode=uacpy.RunMode.BROADBAND, frequencies=band)
cw = H.at(frequency=TONE)
narrow = H.window(frequency=(TONE - 25.0, TONE + 25.0)).broadband_loss()
wide = H.window(frequency=(TONE - 200.0, TONE + 200.0)).broadband_loss()
for label, field in (('continuous wave', cw), ('B =  50 Hz', narrow),
                     ('B = 400 Hz', wide)):
    print(f"{label:>15}: TL spread over the panel {np.nanstd(field.tl):5.2f} dB")

fig, axes = plt.subplots(3, 1, figsize=(13, 14), sharex=True, sharey=True)
for ax, (title, field) in zip(axes, (
        (f'Continuous wave, {TONE:.0f} Hz', cw),
        ('B = 50 Hz — Ainslie Eq. 11.46', narrow),
        ('B = 400 Hz — Ainslie Eq. 11.46', wide))):
    plot_field(field, ax, env=env, vmin=45, vmax=100, title=title)
    ax.set_ylim(300.0, 0.0)
fig.tight_layout()
fig.savefig(OUT / 'example_45a_bandwidth_washes_interference.png', dpi=200,
            bbox_inches='tight')
plt.close(fig)

# The coherent band average against the incoherent run mode
incoherent = model.run(env, uacpy.Source(depths=SOURCE_DEPTH, frequencies=TONE),
                       receiver, run_mode=uacpy.RunMode.INCOHERENT_TL)
difference = uacpy.metrics.tl_difference(wide, incoherent)
gap = np.abs(np.asarray(difference.data))
print(f"Eq. 11.46 vs RunMode.INCOHERENT_TL: median |difference| "
      f"{np.nanmedian(gap):.2f} dB, largest {np.nanmax(gap):.1f} dB")

fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True, sharey=True)
plot_field(incoherent, axes[0], env=env, vmin=45, vmax=100,
           title='RunMode.INCOHERENT_TL — Ainslie Eq. 11.47')
difference.plot(ax=axes[1], env=env, title='Eq. 11.46 minus Eq. 11.47')
for ax in axes:
    ax.set_ylim(300.0, 0.0)
fig.tight_layout()
fig.savefig(OUT / 'example_45b_coherent_against_incoherent.png', dpi=200,
            bbox_inches='tight')
plt.close(fig)

# A 20 ms burst at SOURCE_LEVEL: its own loss, its SEL and its peak SPL
_, pulse = tone_burst(TONE, 20, sample_rate=RATE)
unit = pulse / np.sqrt(np.mean(pulse ** 2))
waveform = unit * 1e-6 * 10 ** (SOURCE_LEVEL / 20.0)    # rms pressure at 1 m (Pa)
exposure = H.sound_exposure_level(source_waveform=waveform, sample_rate=RATE)
peak = H.peak_sound_pressure_level(source_waveform=waveform, sample_rate=RATE)
burst_loss = H.broadband_loss(source_waveform=unit, sample_rate=RATE)
esl = SOURCE_LEVEL + 10.0 * np.log10(pulse.size / RATE)   # Ainslie Eq. 3.155
print(f"burst: SEL {np.nanmedian(exposure.dB):.2f} dB re 1 µPa²·s, "
      f"ESL - TPL {np.nanmedian(esl - burst_loss.tl):.2f} dB, peak SPL "
      f"{np.nanmedian(peak.dB):.2f} dB re 1 µPa (medians over the panel)")

fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True, sharey=True)
plot_field(exposure, axes[0], env=env,
           title=f'Sound exposure level of one {pulse.size / RATE * 1e3:.0f} ms '
                 f'burst at SL = {SOURCE_LEVEL:.0f} dB re 1 µPa @ 1 m')
plot_field(peak, axes[1], env=env,
           title='Peak sound pressure level of the same burst')
for ax in axes:
    ax.set_ylim(300.0, 0.0)
fig.tight_layout()
fig.savefig(OUT / 'example_45c_exposure_and_peak.png', dpi=200,
            bbox_inches='tight')
plt.close(fig)

# A chirp at one receiver, on a frequency grid sized from the arrivals
_, chirp = lfm_chirp(TONE - 100.0, TONE + 100.0, 0.020, sample_rate=RATE)
look = uacpy.Receiver(depths=[LOOK_DEPTH], ranges=[LOOK_RANGE])
paths = model.run(env, uacpy.Source(depths=SOURCE_DEPTH, frequencies=TONE),
                  look, run_mode=uacpy.RunMode.ARRIVALS)
span = paths.energy_support(0.999) + chirp.size / RATE
trace_band = paths.synthesis_band(bandwidth=400.0, fc=TONE,
                                  output_duration=6.0 * span)
trace_field = model.run(
    env, uacpy.Source(depths=SOURCE_DEPTH, frequencies=trace_band),
    look, run_mode=uacpy.RunMode.BROADBAND, frequencies=trace_band)
with warnings.catch_warnings():
    # the rectangular band cuts the chirp's spectral skirts, as intended here
    warnings.simplefilter('ignore', uacpy.NumericsWarning)
    received = trace_field.to_time_trace(
        depth=LOOK_DEPTH, range=LOOK_RANGE, source_waveform=chirp,
        sample_rate=RATE)
clock = np.asarray(received.coords['time'])
print(f"chirp at {LOOK_DEPTH:.0f} m, {LOOK_RANGE / 1e3:.1f} km: "
      f"{len(paths.delays)} arrivals, peak at "
      f"{clock[np.argmax(np.abs(received.data))] * 1e3:.2f} ms")

fig, axes = plt.subplots(2, 1, figsize=(11, 7))
plot_waveform(chirp, RATE, axes[0], time_units='ms',
              title=f'Transmitted: 200 Hz chirp, '
                    f'{chirp.size / RATE * 1e3:.0f} ms')
plot_waveform(np.asarray(received.data).real.ravel(),
              1.0 / float(np.diff(clock)[0]), axes[1], t0=float(clock[0]),
              time_units='ms',
              title=f'Received at {LOOK_DEPTH:.0f} m, '
                    f'{LOOK_RANGE / 1e3:.1f} km — the direct path and its '
                    f'surface image')
fig.tight_layout()
fig.savefig(OUT / 'example_45d_received_chirp.png', dpi=200,
            bbox_inches='tight')
plt.close(fig)
