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
plot_spectrogram · plot_ppsd · plot_constant_q_ppsd · plot_sel
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

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

UPA = 1e-6                     # 1 µPa, the water-acoustics dB reference

# Deep water, Beaufort-6 wind, heavy shipping and rain.
conditions = dict(wind_speed_kn=24, water_depth='deep',
                  shipping_level='high', rain_rate='heavy')
label = "Wenz @ 24 kn / high shipping / heavy rain"

fig, _ = uacpy.plot.plot_wenz(
    WenzNoise(np.linspace(1.0, 1e5, int(1e5 - 1)), **conditions), title=label)
fig.savefig(OUT / 'example_09_wenz_components.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)

# n_fft is the IFFT chunk size and sets the synthesis bin width
# df = sample_rate / n_fft. The Wenz target starts at 1 Hz, so the chunk has to
# be long enough for the low-frequency shape to survive resampling onto the
# FFT-native grid: 96 kHz / 65536 gives df = 1.46 Hz. (The library clamps n_fft
# to [16, 262144], so a token value like 1 would silently become the default.)
sample_rate, n_fft, duration = 96000, 65536, 30.0
frequencies = np.linspace(1.0, 5e4, 10000)
wenz = WenzNoise(frequencies, **conditions)
t, pressure, fs = synthesize_noise_from_psd(
    wenz.as_psd(ref=UPA), frequencies, sample_rate=sample_rate,
    duration=duration, scale=1.0, n_fft=n_fft)
print(f"  synthesised {duration:.0f} s @ {fs / 1e3:.1f} kHz "
      f"({pressure.size:,} samples), df = {fs / n_fft:.2f} Hz")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t[:int(0.2 * fs)] * 1e3, pressure[:int(0.2 * fs)] / UPA, lw=0.5,
        color='C0')
ax.set_xlabel('Time [ms]')
ax.set_ylabel('Pressure [µPa]')
ax.set_title(f'Synthesised Wenz noise (first 0.2 s of {duration:.0f} s)')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'example_09_ssrp_timeseries.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)

# Stationary noise: the spectrogram should look the same at every time.
f_spec, t_spec, power = spectrogram(pressure, fs, nperseg=4096, noverlap=2048)
fig, _ = uacpy.plot.plot_spectrogram(f_spec, t_spec, power, ref=UPA,
                                     title=label, ymin=10, ymax=fs / 2,
                                     vmin=20, vmax=120)
fig.savefig(OUT / 'example_09_ssrp_spectrogram.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)

fig, ax = uacpy.plot.plot_ppsd(
    probabilistic_welch(pressure, fs, ref=UPA, seg_duration=1.0, overlap_pct=50, ddB=1.0,
         lvlmin=20, lvlmax=140),
    title=label, ymin=20, ymax=120)
# The check: the analytic curve the realisation came from, over its own PPSD.
ax.semilogx(wenz.frequencies, wenz.total, color='magenta', linewidth=2.0,
            label='Wenz total (analytic)')
ax.legend(loc='upper right', fontsize='small', framealpha=0.85)
fig.savefig(OUT / 'example_09_ppsd.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# The constant-Q twin of that PPSD: geometric bins, so the resolution follows
# the decades a soundscape spans rather than a fixed Hz spacing.
# The density doors are what make the levels comparable to a Wenz curve, which
# is a density; fmax below Nyquist keeps the near-Nyquist bins (which read a
# coherent tone high) out of the picture.
cq_kw = dict(fmin=20.0, fmax=20000.0, bins_per_octave=24)
cq_hist = probabilistic_constant_q(
    pressure, fs, ref=UPA, lvlmin=20, lvlmax=140, **cq_kw)
cq_mean = constant_q(pressure, fs, **cq_kw)

# vmax: the plotter defaults to 1/binwidth_dB — the largest density a 1 dB bin
# could hold — which suits a Welch-averaged PPSD. A single-look constant-Q
# histogram is far broader (median per-bin std 5.5 dB against 0.9 dB for the
# linear PPSD above), so its peak density is 0.164 and the default would render
# the whole panel in the bottom sixth of the colormap.
fig, ax = uacpy.plot.plot_constant_q_ppsd(cq_hist, title=label, ymin=20,
                                          ymax=120, vmax=0.18)
ax.semilogx(wenz.frequencies, wenz.total, color='magenta', linewidth=2.0,
            label='Wenz total (analytic)')
# The power mean of the same bins lands on that curve, while the histogram's
# own mean line sits below it — the single-look bias the print below measures,
# not an error in the synthesis. Dotted because the plotter already draws that
# mean solid black and its ±STD dashed black.
ax.semilogx(cq_mean.frequencies, 10 * np.log10(cq_mean.power / UPA ** 2),
            color='k', lw=2.2, ls=':',
            label='constant_q')
ax.legend(loc='upper right', fontsize='small', framealpha=0.85)
fig.savefig(OUT / 'example_09_cq_ppsd.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# Each constant-Q frame is one look, so the mean of its dB levels sits
# 10·γ/ln10 = 2.51 dB under the power mean — read mean_dB as the centre of the
# histogram, and compare a target curve against the power mean instead.
analytic = np.interp(cq_hist.frequencies, wenz.frequencies, wenz.total)
print(f"  constant-Q: power mean "
      f"{np.nanmedian(10 * np.log10(cq_mean.power / UPA ** 2) - analytic):+.2f}"
      f" dB against the analytic curve, per-bin dB mean "
      f"{np.nanmedian(cq_hist.mean_dB - analytic):+.2f} dB "
      f"(single-look log mean, 2.51 dB low by construction)")

# SEL is the time-integral of p²(t) (ISO 18405) — the cumulative energy dose of
# the record, per decidecade band, in dB re 1 µPa²·s. The broadband total is
# the incoherent (energy) sum across bands.
exposure = sound_exposure(pressure, fs, fmin=10.0, fmax=fs / 2.0)
print(f"  SEL: broadband "
      f"{10 * np.log10(exposure.power.sum() / UPA ** 2):.1f} dB "
      f"re 1 µPa²·s over {pressure.size / fs:.0f} s across "
      f"{len(exposure.bands)} decidecade bands")
# The estimate carries its bands and its scaling, so the bars and the unit on
# the level axis come from the estimate rather than from arguments repeated
# here: exposure.plot() is the same figure.
fig, _ = uacpy.plot.plot_sel(exposure, ref=UPA,
                             duration=pressure.size / fs,
                             band_type='decidecade', title=label)
fig.savefig(OUT / 'example_09_sel.png', dpi=150, bbox_inches='tight')
plt.close(fig)
