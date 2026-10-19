"""The exposure a moving source delivers: one shot, a weighting, a track.

A survey vessel tows a sparker across a 50 m shelf, firing once a second.
One broadband run gives each shot's sound exposure level (SEL) map,
unweighted and weighted for two hearing groups (Southall et al. 2019); the
track sum accumulates the shots a receiver hears at each distance from the
line, against the impulsive TTS-onset thresholds of Southall Table 7. The
last panel follows one receiver's record during a passage. The source
signature is illustrative: a Ricker wavelet at 2.8 kHz with an energy source
level of 180 dB re 1 µPa²·s·m². docs/guide/noise.md gives the reasoning.

Uses: RunMode.BROADBAND · Field.sound_exposure_level(weighting=) ·
Field.track_exposure(along=) · plot_track_exposure ·
acoustics.track_exposure · acoustics.cumulative_sound_exposure_level(...).plot ·
acoustics.sound_exposure_level(weighting=) · acoustics.sum_levels_dB ·
sonar.detection_ranges · acoustic_signal.ricker_wavelet · plot_field
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.acoustic_signal import ricker_wavelet
from uacpy.acoustics import (cumulative_sound_exposure_level,
                             sound_exposure_level, sum_levels_dB,
                             track_exposure)
from uacpy.sonar import detection_ranges

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

SOURCE_DEPTH, RECEIVER_DEPTH = 3.0, 25.0         # m
ESL = 180.0                                      # dB re 1 µPa²·s·m²
SPEED, INTERVAL, LENGTH = 2.0, 1.0, 4000.0       # m/s, s between shots, m
CPA, WINDOW = 200.0, 120.0                       # passage: m from the line, s

# Env definition and one broadband run over the (depth, range) grid
env = uacpy.Environment(name='Shelf', bathymetry=50.0,
                        ssp=[(0.0, 1510.0), (20.0, 1508.0), (50.0, 1500.0)],
                        bottom='sand', absorption=uacpy.Thorp())
ranges = np.linspace(10.0, 6000.0, 160)
receiver = uacpy.Receiver(depths=np.linspace(5.0, 45.0, 9), ranges=ranges)
model = uacpy.Bellhop()
# the 1/df = 0.5 s record must hold the pulse and every path to 6 km
arrivals = model.run(env, uacpy.Source(depths=SOURCE_DEPTH, frequencies=1500.0),
                     uacpy.Receiver(depths=[RECEIVER_DEPTH], ranges=[ranges[-1]]),
                     run_mode=uacpy.RunMode.ARRIVALS)
frequencies = np.arange(155.0, 8000.0, 2.0)
print(f"arrivals at {ranges[-1] / 1e3:.0f} km span "
      f"{arrivals.energy_support(0.999) * 1e3:.0f} ms of the 500 ms record")
H = model.run(env, uacpy.Source(depths=SOURCE_DEPTH, frequencies=frequencies),
              receiver, run_mode=uacpy.RunMode.BROADBAND,
              frequencies=frequencies)

# One shot, unweighted and weighted for the LF and VHF groups
fs = 16000.0
pulse = ricker_wavelet(np.arange(-0.004, 0.004, 1.0 / fs), 2800.0)
waveform = pulse / np.sqrt(np.sum(pulse ** 2) / fs) * 1e-6 * 10 ** (ESL / 20)
single = {name: H.sound_exposure_level(source_waveform=waveform,
                                       sample_rate=fs, weighting=w)
          for name, w in (('unweighted', None), ('LF', 'LF'), ('VHF', 'VHF'))}
for name in ('LF', 'VHF'):
    kept = (sound_exposure_level(waveform, fs, weighting=name)
            - sound_exposure_level(waveform, fs))
    print(f"{name} weighting keeps {kept:6.1f} dB of the pulse's energy")

fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
for ax, (name, sel) in zip(axes, single.items()):
    uacpy.plot.plot_field(sel, ax, env=env, vmin=100, vmax=160,
                          title=f'One shot, {name} SEL')
fig.tight_layout()
fig.savefig(OUT / 'example_47a_single_shot_sel.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)

# The track sum against distance from the line, and where TTS onset is reached
positions = np.arange(-LENGTH / 2, LENGTH / 2 + 1e-9, SPEED * INTERVAL)
closest = np.geomspace(20.0, 3000.0, 80)
thresholds = {g: uacpy.noise.EXPOSURE_THRESHOLDS[g]['impulsive']['TTS']['SEL']
              for g in ('LF', 'VHF')}
fig, ax = plt.subplots(figsize=(9, 5))
for name, sel in single.items():
    cumulative = sel.track_exposure(positions, closest_approach=closest)
    cumulative.at(depth=RECEIVER_DEPTH).plot(ax=ax, label=f'{name}, all shots')
    if name in thresholds:
        curve = np.asarray(cumulative.at(depth=RECEIVER_DEPTH).data, dtype=float)
        reach = float(detection_ranges(
            closest, signal_excess_dB=curve - thresholds[name]))
        ax.axhline(thresholds[name], linestyle='--', linewidth=0.8,
                   color=ax.lines[-1].get_color())
        print(f"{name}: TTS onset ({thresholds[name]:.0f} dB) reached within "
              f"{reach:.0f} m of the line")
ax.set_xscale('log')
ax.set_ylabel('Cumulative SEL (dB re 1 µPa²·s)')
ax.set_title(f'{positions.size} shots along {LENGTH / 1e3:.0f} km, '
             f'receiver at {RECEIVER_DEPTH:.0f} m')
ax.legend()
fig.tight_layout()
fig.savefig(OUT / 'example_47b_cumulative_vs_distance.png', dpi=150)
plt.close(fig)

# The plan view: the track sum on both sides of the line, VHF-weighted
plan = single['VHF'].at(depth=RECEIVER_DEPTH).track_exposure(
    positions, closest_approach=np.linspace(20.0, 2500.0, 63),
    along=np.linspace(-3000.0, 3000.0, 121))
fig, ax = uacpy.plot.plot_track_exposure(plan, positions=positions,
                                         contours=[thresholds['VHF']])
fig.savefig(OUT / 'example_47c_plan_view.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# One receiver's record during a passage, and its running SEL
stretch = positions[np.abs(positions) <= SPEED * WINDOW / 2]
distances = np.hypot(CPA, stretch)
heard = np.unique(np.round(distances, 3))
H_heard = model.run(env, uacpy.Source(depths=SOURCE_DEPTH,
                                      frequencies=frequencies),
                    uacpy.Receiver(depths=[RECEIVER_DEPTH], ranges=heard),
                    run_mode=uacpy.RunMode.BROADBAND, frequencies=frequencies)
traces = H_heard.synthesize_time_series(source_waveform=waveform,
                                        sample_rate=fs)
rate = traces.sample_rate
times = np.asarray(traces.coords['time'])
shot = np.searchsorted(heard, np.round(distances, 3))
record = np.zeros(int(round((WINDOW + times[-1] + 1.0) * rate)))
for k, t_i in zip(shot, (stretch - stretch[0]) / SPEED):
    trace = np.asarray(traces.data)[0, k]
    start = int(round((t_i + times[0]) * rate))
    record[start:start + trace.size] += trace
running = cumulative_sound_exposure_level(record, rate, step=1.0,
                                          weighting='VHF')
per_shot = np.asarray(H_heard.sound_exposure_level(
    source_waveform=waveform, sample_rate=fs, weighting='VHF').data)[0, shot]
coarse = track_exposure(np.asarray(single['VHF'].at(depth=RECEIVER_DEPTH).data),
                        ranges, stretch, closest_approach=CPA)
print(f"{stretch.size} shots at {CPA:.0f} m, VHF-weighted: record "
      f"{running.levels_dB[-1]:.2f} dB, energy sum of the shots "
      f"{sum_levels_dB(per_shot, axis=0):.2f} dB, track sum on the map "
      f"{coarse:.2f} dB")

fig, (ax_p, ax_s) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
uacpy.plot.plot_waveform(
    record, rate, ax_p, linewidth=0.4,
    title=f'Received at {CPA:.0f} m from the line, {RECEIVER_DEPTH:.0f} m deep')
running.plot(ax=ax_s, label='VHF-weighted, accumulated',
             criterion_dB=thresholds['VHF'],
             criterion_label='TTS onset, Southall Table 7', title='')
fig.tight_layout()
fig.savefig(OUT / 'example_47d_record_and_running_sel.png', dpi=150)
plt.close(fig)
