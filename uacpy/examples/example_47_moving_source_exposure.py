"""The exposure a moving source delivers: one shot, a weighting, a track.

A survey vessel tows a sparker across a 50 m shelf, firing once a second.
What reaches a harbour porpoise or a baleen whale nearby is not one shot's
level but the energy of every shot it hears. Southall et al. (2019) define
the exposure that way: the sound exposure level (SEL), "the time integral of
the square of the instantaneous frequency-weighted sound pressure", is "a
measure of sound energy of exposure accumulated over time and over multiple
exposures". The weighting is the hearing group's; their Table 7 gives the
impulsive TTS-onset thresholds used here: 140 dB re 1 µPa²·s for
very-high-frequency cetaceans (porpoises) and 168 dB for low-frequency ones.

* **A** — one broadband run gives H(f) on a (depth, range) grid.
* **B** — one shot's SEL map, unweighted and weighted for the two groups. The
  sparker's band sits where baleen whales hear best: the LF weighting keeps
  nearly all of a shot's energy, the VHF weighting about 19 dB less, and it
  is the porpoise's 28 dB lower threshold that puts its onset farther out.
* **C** — the track sum: each shot adds its energy at its own distance, so a
  receiver ``d`` from the line accumulates ``Σ 10^(SEL(r_i)/10)``,
  ``r_i = √(d² + x_i²)``. The distance inside which a threshold is reached is
  read off the curve, as a detection range is.
* **D** — the plan view: a receiver abeam of the line's end hears half the
  shots a receiver abeam of its middle hears.
* **E** — one receiver's own record during a passage, and the SEL it
  accumulates second by second. Its end value must equal the energy sum of
  the same shots' single-shot SELs; the track sum on the coarser map adds
  only its range interpolation.

The source signature is illustrative, not a measured sparker's: a Ricker
wavelet at 2.8 kHz, scaled to an energy source level of 180 dB re
1 µPa²·s·m².

Uses: RunMode.BROADBAND · Field.sound_exposure_level(weighting=) ·
Field.track_exposure(along=) · plot_track_exposure ·
acoustics.track_exposure · acoustics.cumulative_sound_exposure_level ·
acoustics.frequency_weighting · acoustics.sum_levels_dB ·
sonar.detection_ranges ·
acoustic_signal.ricker_wavelet · plot_field
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.acoustic_signal import ricker_wavelet
from uacpy.acoustics import (cumulative_sound_exposure_level,
                             frequency_weighting, sum_levels_dB,
                             track_exposure)
from uacpy.sonar import detection_ranges

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

# ── A. One broadband run ────────────────────────────────────────────────────
env = uacpy.Environment(name='Shelf', bathymetry=50.0,
                        ssp=[(0.0, 1510.0), (20.0, 1508.0), (50.0, 1500.0)],
                        bottom='sand', absorption=uacpy.Thorp())
source_depth = 3.0
ranges = np.linspace(10.0, 6000.0, 160)
receiver = uacpy.Receiver(depths=np.linspace(5.0, 45.0, 9), ranges=ranges)
model = uacpy.Bellhop(verbose=False)

# The record a broadband run can hold is 1/df long. It must hold the pulse
# and every path the farthest receiver hears, or late paths fold onto early
# ones and bias the energy. The arrivals say how long that is.
arrivals = model.run(env, uacpy.Source(depths=source_depth, frequencies=1500.0),
                     uacpy.Receiver(depths=[25.0], ranges=[ranges[-1]]),
                     run_mode=uacpy.RunMode.ARRIVALS)
spread = arrivals.energy_support(0.999)
# 155 Hz keeps ray theory valid here (50 m is 5.1 wavelengths); the pulse's
# spectrum is 42 dB down there and 44 dB down at 8 kHz.
frequencies = np.arange(155.0, 8000.0, 2.0)            # record 1/df = 0.5 s
print(f"Arrivals at {ranges[-1] / 1e3:.0f} km span {spread * 1e3:.0f} ms of "
      f"the {1e3 / 2.0:.0f} ms record")

H = model.run(env, uacpy.Source(depths=source_depth, frequencies=frequencies),
              receiver, run_mode=uacpy.RunMode.BROADBAND,
              frequencies=frequencies)

# ── B. One shot, three ways of counting it ──────────────────────────────────
fs = 16000.0
t = np.arange(-0.004, 0.004, 1.0 / fs)
pulse = ricker_wavelet(t, 2800.0)
# A waveform of energy (1 µPa)²·s, times 10^(ESL/20): the map is then the
# SEL itself, in dB re 1 µPa²·s (see Field.sound_exposure_level).
unit = pulse / np.sqrt(np.sum(pulse ** 2) / fs) * 1e-6
ESL = 180.0
waveform = unit * 10 ** (ESL / 20)

groups = {'unweighted': None, 'LF': 'LF', 'VHF': 'VHF'}
single = {name: H.sound_exposure_level(source_waveform=waveform,
                                       sample_rate=fs, weighting=w)
          for name, w in groups.items()}

# How much of the pulse's energy each group counts: the weighting over the
# pulse's own spectrum, the same number at every range.
spectrum = np.abs(np.fft.rfft(waveform)) ** 2
f_bins = np.fft.rfftfreq(waveform.size, 1.0 / fs)
for name in ('LF', 'VHF'):
    kept = (np.sum(spectrum * 10 ** (frequency_weighting(f_bins, name) / 10))
            / np.sum(spectrum))
    print(f"  {name} weighting keeps {10 * np.log10(kept):6.1f} dB of the "
          f"pulse's energy")

fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
for ax, (name, sel) in zip(axes, single.items()):
    uacpy.plot.plot_field(sel, ax, env=env, vmin=100, vmax=160,
                          title=f'One shot, {name} SEL')
fig.tight_layout()
fig.savefig(OUT / 'example_47a_single_shot_sel.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)

# ── C. The track sum, against distance from the line ────────────────────────
speed, interval, length = 2.0, 1.0, 4000.0      # m/s, s between shots, m
positions = np.arange(-length / 2, length / 2 + 1e-9, speed * interval)
closest = np.geomspace(20.0, 3000.0, 80)
depth = 25.0
thresholds = {g: uacpy.noise.EXPOSURE_THRESHOLDS[g]['impulsive']['TTS']['SEL']
              for g in ('LF', 'VHF')}      # Southall Table 7: 168, 140 dB
print(f"\n{positions.size} shots, {speed * interval:.0f} m apart, over "
      f"{length / 1e3:.0f} km; receivers at {depth:.0f} m depth")
fig, ax = plt.subplots(figsize=(9, 5))
for name, sel in single.items():
    cumulative = sel.track_exposure(positions, closest_approach=closest)
    curve = np.asarray(cumulative.at(depth=depth).data, dtype=float)
    one = np.asarray(sel.at(depth=depth).data, dtype=float)
    ax.semilogx(closest, curve, label=f'{name}, all shots')
    line = f"  {name:>10}: one shot at {closest[0]:.0f} m " \
           f"{np.interp(closest[0], ranges, one):5.1f} dB, " \
           f"the whole line {curve[0]:5.1f} dB"
    if name in thresholds:
        # The outermost distance at which the curve still reaches the
        # threshold: a detection range with the threshold as the boundary.
        reach = float(detection_ranges(
            closest, signal_excess_dB=curve - thresholds[name]))
        ax.axhline(thresholds[name], linestyle='--', linewidth=0.8,
                   color=ax.lines[-1].get_color())
        line += (f"; TTS onset ({thresholds[name]:.0f} dB) "
                 + (f"reached within {reach:.0f} m of the line"
                    if np.isfinite(reach) else
                    f"not reached {closest[0]:.0f} m or more from it"))
    print(line)
ax.set_xlabel('Closest approach to the line (m)')
ax.set_ylabel('Cumulative SEL (dB re 1 µPa²·s)')
ax.set_title(f'{positions.size} shots along {length / 1e3:.0f} km, '
             f'receiver at {depth:.0f} m')
ax.grid(True, which='both', alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / 'example_47b_cumulative_vs_distance.png', dpi=150)
plt.close(fig)

# ── D. The plan view ────────────────────────────────────────────────────────
# Every receiver must stay within the map's 6 km of every shot, so the view
# spans 3 km either side of the line's middle and 2.5 km out from it.
plan = single['VHF'].at(depth=depth).track_exposure(
    positions, closest_approach=np.linspace(20.0, 2500.0, 63),
    along=np.linspace(-3000.0, 3000.0, 121))
middle, end = (float(plan.at(along=a, range=200.0).data)
               for a in (0.0, length / 2))
print(f"\n  VHF, 200 m from the line: abeam of its middle {middle:5.1f} dB, "
      f"abeam of its end {end:5.1f} dB ({middle - end:.1f} dB apart, "
      f"just under the 3.0 dB of half the shots at the same distances)")
fig, ax = uacpy.plot.plot_track_exposure(plan, positions=positions,
                                         contours=[thresholds['VHF']])
fig.savefig(OUT / 'example_47c_plan_view.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ── E. One receiver's record during a passage ───────────────────────────────
# A receiver 200 m from the line, abeam of its middle, hears the shots of a
# two-minute stretch. The model is run at exactly the distances those shots
# are from it, and each shot's received pulse is placed at its firing time
# plus the travel time its trace carries.
cpa, window = 200.0, 120.0
stretch = positions[np.abs(positions) <= speed * window / 2]
distances = np.hypot(cpa, stretch)
heard = np.unique(np.round(distances, 3))
H_heard = model.run(env, uacpy.Source(depths=source_depth,
                                      frequencies=frequencies),
                    uacpy.Receiver(depths=[depth], ranges=heard),
                    run_mode=uacpy.RunMode.BROADBAND, frequencies=frequencies)
traces = H_heard.synthesize_time_series(source_waveform=waveform,
                                        sample_rate=fs)
rate = traces.sample_rate
times = np.asarray(traces.coords['time'])
shot = np.searchsorted(heard, np.round(distances, 3))
record = np.zeros(int(round((window + times[-1] + 1.0) * rate)))
for k, t_i in zip(shot, (stretch - stretch[0]) / speed):
    trace = np.asarray(traces.data)[0, k]
    start = int(round((t_i + times[0]) * rate))
    record[start:start + trace.size] += trace
step_times, running = cumulative_sound_exposure_level(
    record, rate, step=1.0, weighting='VHF')
per_shot = np.asarray(H_heard.sound_exposure_level(source_waveform=waveform,
                                                   sample_rate=fs,
                                                   weighting='VHF').data)[0, shot]
coarse = track_exposure(np.asarray(single['VHF'].at(depth=depth).data),
                        ranges, stretch, closest_approach=cpa)
print(f"\n  {stretch.size} shots heard over {window:.0f} s at {cpa:.0f} m, "
      f"VHF-weighted:")
print(f"    accumulated over the record      {running[-1]:6.2f} dB")
print(f"    energy sum of the shots' SELs    "
      f"{sum_levels_dB(per_shot, axis=0):6.2f} dB")
print(f"    track sum on the {np.diff(ranges)[0]:.0f} m range map    "
      f"{coarse:6.2f} dB")
if running[-1] >= thresholds['VHF']:
    crossing = step_times[np.argmax(running >= thresholds['VHF'])]
    print(f"    crosses {thresholds['VHF']:.0f} dB after {crossing:.0f} s")

fig, (ax_p, ax_s) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
t_record = np.arange(record.size) / rate
ax_p.plot(t_record, record, linewidth=0.4)
ax_p.set_ylabel('Pressure (Pa)')
ax_p.set_title(f'Received at {cpa:.0f} m from the line, {depth:.0f} m deep')
ax_s.plot(step_times, running, label='VHF-weighted, accumulated')
ax_s.axhline(thresholds['VHF'], linestyle='--', color='k', linewidth=0.8,
             label='TTS onset, Southall Table 7')
ax_s.set_xlabel('Time (s)')
ax_s.set_ylabel('SEL (dB re 1 µPa²·s)')
ax_s.grid(True, alpha=0.3)
ax_s.legend()
fig.tight_layout()
fig.savefig(OUT / 'example_47d_record_and_running_sel.png', dpi=150)
plt.close(fig)
