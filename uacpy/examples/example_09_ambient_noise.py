"""Ambient noise (Wenz) → time series → PPSD → sound exposure level.

A full noise pipeline: build a Wenz total ambient-noise spectrum, synthesise a
time-domain realisation of it, look at that realisation two ways, and integrate
it into a dose.

The round trip is the point. A stationary process should show a uniform pattern
across the spectrogram, and its PPSD should land back on the analytic Wenz curve
it was synthesised from — which is what the magenta overlay checks.

Uses: noise.WenzNoise(.as_psd) · plot_wenz ·
acoustic_signal.synthesize_noise_from_psd · spectrogram ·
probabilistic_welch · constant_q · probabilistic_constant_q ·
sound_exposure (ISO 18405, decidecade bands) ·
plot_spectrogram · plot_ppsd · plot_band_levels
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.acoustic_signal import (constant_q,
                                   probabilistic_constant_q,
                                   probabilistic_welch,
                                   sound_exposure, spectrogram,
                                   synthesize_noise_from_psd)
from uacpy.noise import WenzNoise

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

# Deep water, Beaufort-6 wind, heavy shipping and rain.
conditions = dict(wind_speed_kn=24, water_depth='deep',
                  shipping_level='high', rain_rate='heavy')
label = "Wenz @ 24 kn / high shipping / heavy rain"

fig, _ = uacpy.plot.plot_wenz(
    WenzNoise(np.linspace(1.0, 1e5, int(1e5 - 1)), **conditions), title=label)
fig.savefig(OUT / 'example_09_wenz_components.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)

# nfft is the IFFT chunk size and sets the synthesis bin width
# df = sample_rate / nfft. The Wenz target starts at 1 Hz, so the chunk has to
# be long enough for the low-frequency shape to survive resampling onto the
# FFT-native grid: 96 kHz / 65536 gives df = 1.46 Hz.
sample_rate, nfft, duration = 96000, 65536, 30.0
frequencies = np.linspace(1.0, sample_rate / 2, 10000)   # up to Nyquist
wenz = WenzNoise(frequencies, **conditions)
# A seeded generator: every run draws the same realisation, so the printed
# levels and the saved figures reproduce.
rng = np.random.default_rng(0)
fs = float(sample_rate)
t, pressure = synthesize_noise_from_psd(
    wenz.as_psd(), frequencies, sample_rate=fs,
    duration=duration, nfft=nfft, rng=rng)
print(f"  synthesised {duration:.0f} s @ {fs / 1e3:.1f} kHz "
      f"({pressure.size:,} samples), df = {fs / nfft:.2f} Hz")

fig, _ = uacpy.plot.plot_waveform(
    pressure[:int(0.2 * fs)], fs, time_unit='ms', lw=0.5,
    title=f'Synthesised Wenz noise (first 0.2 s of {duration:.0f} s)')
fig.savefig(OUT / 'example_09_ssrp_timeseries.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)

# Stationary noise: the spectrogram should look the same at every time.
f_spec, t_spec, power = spectrogram(pressure, fs, nperseg=4096, noverlap=2048)
fig, _ = uacpy.plot.plot_spectrogram(f_spec, t_spec, power, title=label, freq_min=10, freq_max=fs / 2,
                                     vmin=20, vmax=120)
fig.savefig(OUT / 'example_09_ssrp_spectrogram.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)

# Welch's default 8192-sample segment makes 11.7 Hz bins at 96 kHz, too coarse
# for the curve's steep fall below 20 Hz, so the PPSD starts at 20 Hz as the
# constant-Q one below does. The check: the analytic curve the realisation came
# from, drawn first, under its own PPSD, which then sets the axes' window.
fig, ax = uacpy.plot.plot_wenz(wenz, show_components=False, color='magenta',
                               label='Wenz total (analytic)', figsize=(10, 6))
uacpy.plot.plot_ppsd(
    probabilistic_welch(pressure, fs, level_min_dB=20, level_max_dB=140,
                        freq_min=20.0),
    ax=ax, title=label, ymin=20, ymax=120)
fig.savefig(OUT / 'example_09_ppsd.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# The constant-Q twin of that PPSD: geometric bins, so the resolution follows
# the decades a soundscape spans rather than a fixed Hz spacing.
# The density doors are what make the levels comparable to a Wenz curve, which
# is a density; freq_max below Nyquist keeps the near-Nyquist bins (which read a
# coherent tone high) out of the picture.
cq_kw = dict(freq_min=20.0, freq_max=20000.0, bins_per_octave=24)
cq_hist = probabilistic_constant_q(
    pressure, fs, level_min_dB=20, level_max_dB=140, **cq_kw)
cq_mean = constant_q(pressure, fs, **cq_kw)

# vmax: the plotter defaults to 1/level_step_dB — the largest density a 1 dB bin
# could hold — which suits a Welch-averaged PPSD. A single-look constant-Q
# histogram is far broader than the linear PPSD above, so its peak density sits
# far below that default; vmax=0.18 sits just above it.
fig, ax = uacpy.plot.plot_wenz(wenz, show_components=False, color='magenta',
                               label='Wenz total (analytic)', figsize=(10, 6))
uacpy.plot.plot_ppsd(cq_hist, ax=ax, title=label, ymin=20, ymax=120,
                     vmax=0.18)
# The power mean of the same bins lands on that curve, while the histogram's
# own mean line sits below it — the single-look bias the print below measures,
# not an error in the synthesis. Dotted because the plotter already draws that
# mean solid black and its ±STD dashed black.
cq_mean.plot(ax=ax, color='k', lw=2.2, ls=':', label='constant_q',
             title=label, ymin=20, ymax=120)
fig.savefig(OUT / 'example_09_cq_ppsd.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# Each constant-Q frame is one look, so the mean of its dB levels sits
# 10·γ/ln10 = 2.51 dB under the power mean — read mean_dB as the centre of the
# histogram, and compare a target curve against the power mean instead.
analytic = np.interp(cq_hist.frequencies, wenz.frequencies, wenz.total)
print(f"  constant-Q: power mean "
      f"{np.nanmedian(cq_mean.level_dB() - analytic):+.2f}"
      f" dB against the analytic curve, per-bin dB mean "
      f"{np.nanmedian(cq_hist.mean_dB - analytic):+.2f} dB "
      f"(single-look log mean, 2.51 dB low by construction)")

# SEL is the time-integral of p²(t) (ISO 18405) — the cumulative energy dose of
# the record, per decidecade band, in dB re 1 µPa²·s. The broadband total is
# the incoherent (energy) sum across bands.
exposure = sound_exposure(pressure, fs, freq_min=10.0, freq_max=fs / 2.0)
print(f"  SEL: broadband "
      f"{uacpy.acoustics.power_to_dB(exposure.power.sum()):.1f} dB "
      f"re 1 µPa²·s over {pressure.size / fs:.0f} s across "
      f"{len(exposure.bands)} decidecade bands")
# The estimate carries its bands and its scaling, so the bars and the unit on
# the level axis come from the estimate rather than from arguments repeated
# here: exposure.plot() is the same figure.
fig, _ = uacpy.plot.plot_band_levels(exposure, title=label)
fig.savefig(OUT / 'example_09_sel.png', dpi=150, bbox_inches='tight')
plt.close(fig)
