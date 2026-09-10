"""Plotting features — stacked time series and mode heatmaps.

Two visualization helpers driven by real model output:

* plot_field(stacked=True) — one offset trace per range, the seismic-waterfall
  view of impulse responses. The propagation delay shows up as a linear slope
  across the traces. (The Acoustics Toolbox's plotts.m.)
* plot_modes_heatmap — Kraken mode shapes as one 2-D panel, whole set or a
  slice of it. (plotmode.m.)

Uses: RunMode.BROADBAND · acoustic_signal.ricker_wavelet ·
Field.synthesize_time_series · Field.at() · plot_field(stacked=) ·
RunMode.MODES · plot_modes_heatmap(mode_range=, normalize=)
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.acoustic_signal.waveforms import ricker_wavelet

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

# ── Stacked time series: Bellhop BROADBAND → Ricker source → p(t) ───────────
env = uacpy.Environment(name='Pekeris waveguide', bathymetry=100, ssp=1500,
                        bottom=uacpy.BoundaryProperties(acoustic_type='rigid'))
source = uacpy.Source(depths=50, frequencies=100)
receiver = uacpy.Receiver(depths=np.array([50.0]),
                          ranges=np.linspace(500, 5000, 12))

frequencies = np.linspace(50.0, 200.0, 600)
transfer = uacpy.Bellhop().run(env, source, receiver,
                               run_mode=uacpy.RunMode.BROADBAND,
                               frequencies=frequencies)
fs = 1000.0
pulse = ricker_wavelet(np.arange(int(0.04 * fs)) / fs,
                       float(source.frequencies[0]))
waveform = transfer.synthesize_time_series(source_waveform=pulse,
                                           sample_rate=fs)
print(f"  {transfer.data.shape[-1]} frequencies at "
      f"df={frequencies[1] - frequencies[0]:.3f} Hz → p(t) "
      f"{waveform.data.shape}, {waveform.times[-1]:.3f} s")

# The synthesized field has {depth, range, time}; slice the single receiver
# depth so plot_field sees a 2-D (range, time) field.
traces = waveform.at(depth=50.0)
fig, _ = uacpy.plot_field(traces, stacked=True,
                          title='Stacked impulse responses per range')
fig.savefig(OUT / 'example_14_time_series_stacked.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)

# stacked=False falls back to the (range × time) heatmap.
fig, _ = uacpy.plot_field(traces, title='Time-series heatmap (range × time)')
fig.savefig(OUT / 'example_14_time_series_overlaid.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)

# ── Mode shapes: the whole set, then the first four ─────────────────────────
mode_env = uacpy.Environment(
    name='Pekeris waveguide', bathymetry=100, ssp=1500,
    bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                    sound_speed=1700, density=1.8,
                                    attenuation=0.5))
modes = uacpy.Kraken(n_modes=50).run(
    mode_env, source,
    uacpy.Receiver(depths=np.linspace(0, mode_env.depth, 200),
                   ranges=np.array([1000.0])),
    run_mode=uacpy.RunMode.MODES)
print(f"  Kraken found {len(modes.k)} modes")

fig, _ = uacpy.plot.plot_modes_heatmap(modes, mode_range=None, normalize=True,
                                       figsize=(14, 8))
fig.savefig(OUT / 'example_14_modes_heatmap_all.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)

fig, _ = uacpy.plot.plot_modes_heatmap(modes, mode_range=(0, 4),
                                       normalize=True, figsize=(12, 8))
fig.savefig(OUT / 'example_14_modes_heatmap_subset.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)
