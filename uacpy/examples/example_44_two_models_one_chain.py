"""Two propagation models through one detection chain, and what the beam hears.

One model agreeing with itself proves nothing. Kraken is a full-wave modal
code and Bellhop a ray/beam one: they approximate different things and fail
in different places, so where they agree the propagation underneath a sonar
prediction is worth trusting.

Comparing them fairly is its own small problem. "Both incoherent" is not on
offer — coupled modes cannot be added incoherently at all — so some
asymmetry is forced, and the tempting protocol (Kraken coherent and
smoothed against Bellhop INCOHERENT) is the wrong way to handle it: its
answer tracks the smoothing kernel, so there is no plateau to quote, and
smoothing one side cancels part of the real difference. Both fields are
therefore taken coherent and put through the same boxcar, and the kernel is
swept to measure what the protocol itself is worth. The spread across
kernels IS the noise floor, and a comparison without one should not be
quoted.

Agreeing on transmission loss is not the same as agreeing on the answer, so
the second model is carried all the way to a detection map of its own. Two
results come out, pointing opposite ways:

* the realised array gain agrees to well under a decibel. Two methods with
  nothing in common see the same shortfall against the plane-wave value, so
  that shortfall is a property of the problem and not of normal-mode
  theory;
* the detection MAPS agree while the detection RANGES do not. A model
  difference of order a decibel barely moves the map and moves the
  outermost zero crossing by a kilometre, because that crossing is a tail
  statistic: one constructive fringe past the main boundary owns it. Quote
  the detectable AREA, not the detection RANGE.

Finally, the same array run over a BAND instead of one line gives the
reception itself — this array, steered this way, listening to that target
radiate that pulse. Each bin is steered at its own frequency, because a
beam delay is a phase that scales with it: one steering vector at the band
centre mis-steers both edges.

Uses: Kraken(mode_coupling='coupled') · Bellhop · RunMode.BROADBAND ·
beamform_field (narrowband and broadband) · BeamformedField.to_time_trace ·
compare_models · passive_signal_excess_field(array_gain=<grid>)
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.acoustic_signal import (beamform_field, independent_beams,
                                   shading_taper)
from uacpy.core.results import Field
from uacpy.models import RunMode
from uacpy.sonar import (detection_threshold_energy,
                         passive_signal_excess_field, per_look_false_alarm,
                         probability_of_detection_field)
from uacpy.sonar.sonar_equation import detection_range_by_depth
from uacpy.visualization import compare_models

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

FREQ, C_REF = 200.0, 1500.0
HALF = 0.5 * C_REF / FREQ
SL, NL = 120.0, 75.0          # band-integrated, dB re 1 µPa²

# ── the scene ───────────────────────────────────────────────────────────
# The same shoaling section example 42 uses: a shelf under the array, a
# sill at 130 m, a basin behind it, then a shoaling run.
BATHY_RANGES = np.linspace(0.0, 20000.0, 41)
BATHY_DEPTHS = np.interp(BATHY_RANGES,
                         [0.0, 3000.0, 6000.0, 9000.0,
                          11000.0, 13500.0, 16500.0, 20000.0],
                         [200.0, 200.0, 160.0, 130.0,
                          150.0, 175.0, 145.0, 120.0])
env = uacpy.Environment(
    name='sill-and-basin',
    bathymetry=list(zip(BATHY_RANGES, BATHY_DEPTHS)),
    ssp=[(0.0, 1520.0), (30.0, 1518.0), (80.0, 1502.0), (200.0, 1498.0)],
    bottom=uacpy.Bottom.from_halfspace(uacpy.BoundaryProperties(
        acoustic_type='half-space', sound_speed=1700.0, density=1.9,
        attenuation=0.5)),
    absorption=uacpy.Thorp(),
)
n_el = 24
elements = 120.0 + HALF * (np.arange(n_el) - (n_el - 1) / 2.0)
target = uacpy.Source(depths=60.0, frequencies=FREQ, source_level_dB=SL)
array = uacpy.Receiver(depths=elements, ranges=np.linspace(200.0, 20000.0, 200))
array_at_origin = uacpy.Receiver(depths=elements, ranges=[0.0])
plane_depths = np.union1d(np.linspace(5.0, 195.0, 60), [60.0])
plane = uacpy.Receiver(depths=plane_depths, ranges=array.ranges)
angles = np.linspace(-45.0, 45.0, 361)
taper = shading_taper(n_el, 'hann')
mid = n_el // 2

# A target cannot be inside the sediment; env.bathymetry.eval is the same
# interpolation the solver uses, so the mask cannot drift from its seafloor.
seabed = np.asarray(env.bathymetry.eval(range=array.ranges), dtype=float)
in_water = plane_depths[:, None] <= seabed[None, :]
i60 = int(np.argmin(np.abs(plane_depths - 60.0)))

# ── the two fields ──────────────────────────────────────────────────────
# Sources at the array's own element depths, at r = 0 where the array is —
# reciprocity then makes slab i the field element i would measure from a
# target at each point, and it holds whatever the bottom does in range.
# See example 43 for what the mirror arrangement costs.
array_as_sources = uacpy.Source(depths=elements, frequencies=FREQ)
kraken = uacpy.Kraken(verbose=False, mode_coupling='coupled')
bellhop = uacpy.Bellhop(verbose=False, n_beams=0, beam_type='G')
p_map = np.stack([np.asarray(f.data)
                  for _, f in kraken.run(env, array_as_sources, plane)])
bh_map = np.stack([np.asarray(f.data) for _, f in bellhop.run(
    env, array_as_sources, plane, run_mode=RunMode.COHERENT_TL)])

# ── propagation, compared symmetrically ─────────────────────────────────
edge = 5
for width in (5, 9, 21):
    k = np.ones(width) / width
    kr_s = np.apply_along_axis(lambda r: np.convolve(r, k, mode='same'), 1,
                               np.abs(p_map[mid]) ** 2)
    bh_s = np.apply_along_axis(lambda r: np.convolve(r, k, mode='same'), 1,
                               np.abs(bh_map[mid]) ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        diff = -10.0 * np.log10(kr_s) + 10.0 * np.log10(bh_s)
    ok = in_water & np.isfinite(diff)
    ok[:, :edge] = ok[:, -edge:] = False
    d = diff[ok]
    print(f"Kraken vs Bellhop, both coherent, {width:2d}-bin boxcar: bias "
          f"{np.mean(d):+.2f} dB, median |diff| {np.median(np.abs(d)):.2f} dB,"
          f" p90 {np.percentile(np.abs(d), 90):.2f} dB")
print("  the answer barely moves across those kernels, and that spread is "
      "the protocol's\n  own noise floor — a comparison without such a "
      "plateau should not be quoted")

# ── the same chain, twice ───────────────────────────────────────────────
PD, PF = 0.5, 1e-4
BW_HZ, INT_S = 10.0, 10.0
# weights= matters: two beams' noise correlates as the array factor of
# |w|^2, and hann^2 is three DFT bins wide against a rectangular window's
# one, so this scan holds about a third of the looks the unshaded geometry
# would give.
n_looks = independent_beams(elements, angles, FREQ, c=C_REF, weights=taper)
dt = detection_threshold_energy(pd=PD, pf=per_look_false_alarm(PF, n_looks),
                                bandwidth_hz=BW_HZ, integration_time_s=INT_S)
print(f"\nthe {angles.size}-point scan holds {n_looks:.0f} independent "
      f"looks, so DT = {dt:.2f} dB at a scan Pf of {PF:g}")

se, pd_map, r_by_z = {}, {}, {}
for label, pressure in (('Kraken', p_map), ('Bellhop', bh_map)):
    scan = beamform_field(pressure, elements, angles, FREQ, c=C_REF,
                          weights=taper)
    with np.errstate(divide='ignore', invalid='ignore'):
        gain = np.where(in_water, scan.array_gain(), np.nan)
        loss = np.where(in_water, -10.0 * np.log10(
            np.maximum(scan.element_power, 1e-300)), np.nan)
    tl = Field(data=loss, coords={'depth': plane_depths,
                                  'range': np.asarray(array.ranges,
                                                      dtype=float)},
               model=label)
    se[label] = passive_signal_excess_field(tl, source_level=SL,
                                            noise_level=NL, array_gain=gain,
                                            detection_threshold=dt)
    pd_map[label] = probability_of_detection_field(se[label], sigma_dB=8.0)
    _, r_by_z[label] = detection_range_by_depth(se[label])
    print(f"  {label:8s} realised AG median {np.nanmedian(gain):.1f} dB, "
          f"{100.0 * np.mean(np.asarray(se[label].data)[in_water] > 0.0):.0f} "
          f"% of the water above threshold")

k_se, b_se = (np.asarray(se[m].data) for m in ('Kraken', 'Bellhop'))
verdict = 100.0 * np.mean((k_se[in_water] > 0.0) != (b_se[in_water] > 0.0))
pd_gap = np.nanmean(np.abs(np.asarray(pd_map['Kraken'].data)
                           - np.asarray(pd_map['Bellhop'].data))[in_water])
print(f"the two MAPS agree: mean |Pd difference| {pd_gap:.02f}, and they "
      f"disagree on detect /\n  do-not-detect at {verdict:.1f} % of the water")
print(f"the two RANGES do not: median across target depth "
      f"{np.nanmedian(r_by_z['Kraken']) / 1e3:.1f} km against "
      f"{np.nanmedian(r_by_z['Bellhop']) / 1e3:.1f} km")
print("  -> a 1 dB model difference barely moves the map and moves the "
      "outermost crossing\n  by a kilometre; the AREA is the statistic to "
      "quote, not the RANGE")

fig, _ = compare_models(
    [pd_map['Kraken'], pd_map['Bellhop']],
    labels=['Kraken — coupled modes', 'Bellhop — Gaussian beams'],
    env=env, receiver=array_at_origin, figsize=(13, 4.6), contours=(0.5,),
    title=f'The same detection chain under two models — maps agree to '
          f'{pd_gap:.02f} in P_D, headline ranges differ by '
          f'{abs(np.nanmedian(r_by_z["Kraken"]) - np.nanmedian(r_by_z["Bellhop"])) / 1e3:.1f} km')
# compare_models sets its margins as fixed fractions, so the suptitle lands
# on the panel titles at this aspect; give it back the strip it needs.
fig.subplots_adjust(top=0.80)
fig.savefig(OUT / 'example_44_two_models.png', dpi=140, bbox_inches='tight')
plt.close(fig)

# ── what the beam actually hears ────────────────────────────────────────
# Everything above is a power budget. The same machinery over a BAND gives
# the reception itself. Bellhop for this one, for speed rather than
# capability: Kraken does run a range-dependent band (uacpy loops the
# multi-profile deck, one mode solve per bin) but that is ~0.3 s a bin.
BAND = np.linspace(FREQ - 50.0, FREQ + 50.0, 81)
R_SHOT = float(array.ranges[np.argmin(np.abs(np.asarray(array.ranges) - 5000.0))])
H = bellhop.run(env, uacpy.Source(depths=60.0, frequencies=BAND),
                uacpy.Receiver(depths=elements, ranges=[R_SHOT]),
                run_mode=RunMode.BROADBAND)
beams = beamform_field(np.asarray(H.data)[:, 0, :], elements, angles, BAND,
                       c=C_REF, weights=taper)
pulse = np.exp(-((BAND - FREQ) / 25.0) ** 2)     # a Gaussian tone burst
# The window opens at the earliest arrival this MODEL can produce, r over
# the fastest WATER speed. The seabed's 1700 m/s must not enter it: a ray
# code has no head wave, and anchoring above the true fastest speed opens
# the window early, so late multipath wraps into the record.
t_start = R_SHOT / float(np.max(np.asarray(env.ssp.data)))
# power is (n_angles, n_frequencies) here, so the band-averaged winner is
# the look to use.
look_best = float(angles[int(np.argmax(beams.power.mean(axis=1)))])
trace_one = H.to_time_trace(depth=float(elements[mid]), range=R_SHOT,
                            source_spectrum=pulse, t_start=t_start)
t_one = np.asarray(trace_one.coords['time'])
y_one = np.asarray(trace_one.data).ravel()

print(f"\nreception at {R_SHOT / 1e3:.1f} km of a {BAND[0]:.0f}-"
      f"{BAND[-1]:.0f} Hz pulse, peak level re one element:")
fig, ax = plt.subplots(figsize=(11, 3.4))
ax.plot(t_one, np.abs(y_one) / np.abs(y_one).max(), color='0.6', lw=1.0,
        label='one element')
for (label, look), style in ((('steered to the arrival', look_best), 'C0-'),
                             (('steered 30 deg off', look_best + 30.0),
                              'C3--')):
    tr = beams.to_time_trace(look, range_m=R_SHOT, source_spectrum=pulse,
                             t_start=t_start)
    y = np.asarray(tr.data).ravel()
    print(f"  {label:24s} ({look:+5.1f} deg): "
          f"{20.0 * np.log10(np.max(np.abs(y)) / np.max(np.abs(y_one))):+5.1f} dB")
    ax.plot(np.asarray(tr.coords['time']), np.abs(y) / np.abs(y_one).max(),
            style, lw=1.2, label=f'{label} ({look:+.1f} deg)')
ax.set(xlabel='Time (s)', ylabel='|p| (re one element peak)',
       title=f'What the beam hears: a {BAND[0]:.0f}-{BAND[-1]:.0f} Hz pulse '
             f'from 60 m at {R_SHOT / 1e3:.1f} km')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'example_44_reception.png', dpi=140, bbox_inches='tight')
plt.close(fig)
print(f"\nFigures written to {OUT}")
