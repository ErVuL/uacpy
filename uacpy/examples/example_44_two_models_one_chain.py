"""Two propagation models through one detection chain, and what the beam hears.

The detection chain of example_42 run twice, on a Kraken (coupled modes) and a
Bellhop (beam tracing) field of the same section, with the array gain each
beamformed field realises. Then the same array over a band: a broadband run,
each bin steered at its own frequency, and the pulse one beam receives.
docs/guide/sonar.md §3 reads the two detection maps, docs/guide/arrays.md
§"Broadband" the per-bin steering.

Uses: Kraken(mode_coupling='coupled') · Bellhop · Field.p over source_depth ·
beamform_field (narrowband and broadband) · BeamformedField.to_time_trace ·
Field.to_time_trace(source_waveform=) · RunMode.BROADBAND · compare_models ·
passive_signal_excess_field(array_gain_dB=<grid>) · Field.plot(value='envelope')
"""

import os
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.acoustic_signal import (beamform_field, independent_beams,
                                   shading_taper, tone_burst)
from uacpy.models import RunMode
from uacpy.plot import compare_models
from uacpy.sonar import (detection_ranges_field, detection_threshold_energy,
                         passive_signal_excess_field, per_look_false_alarm,
                         transition_probability_field)

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

FREQ, C_REF = 200.0, 1500.0          # Hz, m/s
N_EL, ARRAY_DEPTH = 24, 120.0        # elements at lambda/2 about this depth (m)
TARGET_DEPTH = 60.0                  # m
SL, NL = 120.0, 75.0                 # band-integrated, dB re 1 µPa²
PD, PF = 0.5, 1e-4                   # Pf of the whole scan
BW_HZ, INT_S = 10.0, 10.0
SIGMA_DB = 8.0
R_SHOT, FS = 5000.0, 2000.0          # broadband reception: range (m), sample rate (Hz)

# Env definition: the sill-and-basin section of example_42
bathy_ranges = np.linspace(0.0, 20000.0, 41)
bathy_depths = np.interp(bathy_ranges,
                         [0.0, 3000.0, 6000.0, 9000.0,
                          11000.0, 13500.0, 16500.0, 20000.0],
                         [200.0, 200.0, 160.0, 130.0,
                          150.0, 175.0, 145.0, 120.0])
env = uacpy.Environment(
    name='sill-and-basin',
    bathymetry=list(zip(bathy_ranges, bathy_depths)),
    ssp=[(0.0, 1520.0), (30.0, 1518.0), (80.0, 1502.0), (200.0, 1498.0)],
    bottom=uacpy.Bottom.from_halfspace(uacpy.BoundaryProperties(
        acoustic_type='half-space', sound_speed=1700.0, density=1.9,
        attenuation=0.5)),
    absorption=uacpy.Thorp(),
)

# Array at r = 0 and the plane of candidate target positions
elements = ARRAY_DEPTH + 0.5 * C_REF / FREQ * (np.arange(N_EL) - (N_EL - 1) / 2.0)
array_as_sources = uacpy.Source(depths=elements, frequencies=FREQ)
array_at_origin = uacpy.Receiver(depths=elements, ranges=[0.0])
ranges = np.linspace(600.0, 20000.0, 196)   # past 0.6 km the modes hold every path
plane_depths = np.union1d(np.linspace(5.0, 195.0, 60), [TARGET_DEPTH])
plane = uacpy.Receiver(depths=plane_depths, ranges=ranges)
in_water = plane_depths[:, None] <= np.asarray(env.bathymetry.eval(range=ranges))
angles = np.linspace(-45.0, 45.0, 361)
taper = shading_taper(N_EL, 'hann')

# The two fields: sources at the element depths, receivers over the plane
kraken = uacpy.Kraken(mode_coupling='coupled')
bellhop = uacpy.Bellhop(backend='fortran')
pressure = {'Kraken': kraken.run(env, array_as_sources, plane).p,
            'Bellhop': bellhop.run(env, array_as_sources, plane).p}

# The same detection chain on each
n_looks = independent_beams(elements, angles, FREQ, sound_speed=C_REF,
                            weights=taper)
dt = detection_threshold_energy(pd=PD, pf=per_look_false_alarm(PF, n_looks),
                                bandwidth=BW_HZ, integration_time=INT_S)
se, pd_map, r_by_z = {}, {}, {}
for label, p_map in pressure.items():
    with warnings.catch_warnings(record=True) as caught:   # Bellhop's blank cells
        warnings.simplefilter('always')
        scan = beamform_field(p_map, elements, angles, FREQ, sound_speed=C_REF,
                              weights=taper)
    for warning in caught:
        print(f"noted ({label}): {str(warning.message).splitlines()[0]}")
    with np.errstate(divide='ignore', invalid='ignore'):
        gain = np.where(in_water, scan.array_gain(), np.nan)
        tl_map = np.where(in_water, -10.0 * np.log10(scan.element_power),
                          np.nan)
    tl = uacpy.Field(data=tl_map,
                     coords={'depth': plane_depths, 'range': ranges},
                     model=label)
    se[label] = passive_signal_excess_field(
        tl, source_level_dB=SL, noise_level_dB=NL, array_gain_dB=gain,
        detection_threshold_dB=dt)
    pd_map[label] = transition_probability_field(se[label], sigma_dB=SIGMA_DB)
    r_by_z[label] = detection_ranges_field(se[label])
    covered = 100.0 * np.mean(np.asarray(se[label].data)[in_water] > 0.0)
    print(f"{label:>8}: realised AG median {np.nanmedian(gain):.1f} dB, "
          f"{covered:.0f} % of the water above threshold, median detection "
          f"range {np.nanmedian(r_by_z[label]) / 1e3:.1f} km")
pd_gap = np.nanmean(np.abs(np.asarray(pd_map['Kraken'].data)
                           - np.asarray(pd_map['Bellhop'].data))[in_water])
print(f"mean |Pd difference| over the water: {pd_gap:.2f}")

fig, _ = compare_models(
    [pd_map['Kraken'], pd_map['Bellhop']],
    labels=['Kraken — coupled modes', 'Bellhop — hat beams'],
    env=env, receiver=array_at_origin, figsize=(13, 4.6), contours=(0.5,),
    title=f'One detection chain under two models — mean |ΔPd| {pd_gap:.2f}')
fig.savefig(OUT / 'example_44_two_models.png', dpi=140, bbox_inches='tight')
plt.close(fig)

# Broadband: one Bellhop run over a band, each bin steered at its own frequency
band = np.linspace(FREQ - 60.0, FREQ + 60.0, 97)
H = bellhop.run(env, uacpy.Source(depths=TARGET_DEPTH, frequencies=band),
                uacpy.Receiver(depths=elements, ranges=[R_SHOT]),
                run_mode=RunMode.BROADBAND)
beams = beamform_field(np.asarray(H.data)[:, 0, :], elements, angles, band,
                       sound_speed=C_REF, weights=taper)
_, pulse = tone_burst(FREQ, 10, sample_rate=FS)
t_start = R_SHOT / float(np.max(env.ssp.sound_speed))   # fastest water path
look = float(angles[int(np.argmax(beams.power.mean(axis=1)))])

# What one element and two beams receive
fig, ax = plt.subplots(figsize=(11, 3.4))
traces = {'one element': H.to_time_trace(
    depth=float(elements[N_EL // 2]), range=R_SHOT, source_waveform=pulse,
    sample_rate=FS, t_start=t_start)}
for label, angle in ((f'beam at {look:+.1f}°', look),
                     (f'beam at {look + 30.0:+.1f}°', look + 30.0)):
    traces[label] = beams.to_time_trace(angle, range=R_SHOT,
                                        source_waveform=pulse, sample_rate=FS,
                                        t_start=t_start)
peak_one = np.max(np.abs(np.asarray(traces['one element'].data)))
for label, trace in traces.items():
    peak = np.max(np.abs(np.asarray(trace.data)))
    print(f"{label:>15}: peak {20.0 * np.log10(peak / peak_one):+5.1f} "
          f"dB re one element")
    trace.plot(ax=ax, value='envelope', label=label,
               title=f'A {FREQ:.0f} Hz tone burst from {TARGET_DEPTH:.0f} m '
                     f'at {R_SHOT / 1e3:.0f} km')
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / 'example_44_reception.png', dpi=140, bbox_inches='tight')
plt.close(fig)
print(f"\nFigures written to {OUT}")
