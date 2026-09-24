"""The level a signal with bandwidth or duration reaches, as a map.

A model's transmission loss is the **continuous-wave** answer: one frequency,
every path interfering at once. A real signal — a chirp, a burst, a comms
packet — never meets that field, and the literature names two quantities for
what it does meet:

* **Broadband propagation loss**, the frequency average of the COHERENT
  ``|H|^2`` over the signal's band (Ainslie, *Sonar Performance Modeling*,
  sect. 11.3.3 Eq. 11.46; the same ratio Abraham writes as a pulse's
  propagation loss in sect. 3.2.4.2). It says how much of the continuous-wave
  interference a bandwidth survives.
* **Sound exposure level**, the time integral of ``p(t)^2`` (Abraham
  sect. 3.2.1.5, acoustic energy flux density; ISO 18405). It says how much
  energy one transmission delivers.

The geometry is a Lloyd mirror — a source 20 m below the surface in deep
water, so the panel is one direct path against its surface image. Two paths,
not a modal sum, and that matters: a coherent map is only worth drawing where
its fringes are RESOLVED, and part A measures that rather than assuming it.

Part B is why ``broadband_loss`` exists when every ray model already ships an
incoherent run mode. Ainslie's Eq. 11.47 — the incoherent step, which is what
``RunMode.INCOHERENT_TL`` computes — "neglects coherent interference effects
such as cancellation between direct and surface-reflected paths. This
coherent effect is not negligible, even for incoherent broadband processing,
if the distance between the sonar (or target) and the sea surface is a few
wavelengths or less at the center frequency, in which case use of Equation
(11.46) is required." This panel is that case, and the numbers say where.

Uses: RunMode.BROADBAND / COHERENT_TL / INCOHERENT_TL · Field.at ·
Field.window · Field.broadband_loss(waveform=) · Field.sound_exposure_level ·
Field.to_time_trace(waveform=) · Field.synthesize_time_series ·
Arrivals.synthesis_band / energy_support ·
acoustic_signal.tone_burst / lfm_chirp · plot_field · plot_field_difference ·
plot_waveform
"""

import os
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.acoustic_signal import lfm_chirp, tone_burst
from uacpy.visualization import (plot_field_difference,
                                 plot_waveform)

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

env = uacpy.Environment(name='Deep water', bathymetry=5000.0, ssp=1500.0)
# +-60 deg covers the steepest direct path (300 m down at 300 m of range is
# 43 deg); the 5000 m bottom returns at 2*5000/tan(60) = 5.8 km, well outside
# the panel, so what is drawn is the direct path and its surface image alone.
model = uacpy.Bellhop(verbose=False, n_beams=12001, alpha=(-60.0, 60.0))
source_depth, tone = 20.0, 1000.0
depths = np.linspace(2.0, 300.0, 150)
ranges = np.linspace(300.0, 2500.0, 350)
receiver = uacpy.Receiver(depths=depths, ranges=ranges)

# ── A. Is the coherent panel resolved at all? ───────────────────────────────
# A fringe pattern finer than the pixel does not disappear, it ALIASES, and
# the alias looks like structure. The test is scaling: a resolved field's
# pixel-to-pixel step is proportional to the pixel, so halving the pixel
# halves the step. An aliased one barely moves.
coarse = model.run(env, uacpy.Source(depths=source_depth, frequencies=tone),
                   receiver, run_mode=uacpy.RunMode.COHERENT_TL)
fine = model.run(env, uacpy.Source(depths=source_depth, frequencies=tone),
                 uacpy.Receiver(depths=np.linspace(2.0, 300.0, 300),
                                ranges=ranges),
                 run_mode=uacpy.RunMode.COHERENT_TL)
coarse_step = np.nanmedian(np.abs(np.diff(np.asarray(coarse.tl), axis=0)))
fine_step = np.nanmedian(np.abs(np.diff(np.asarray(fine.tl), axis=0)))
print(f"Lloyd mirror, source {source_depth:.0f} m, {tone:.0f} Hz, "
      f"{depths[0]:.0f}-{depths[-1]:.0f} m over "
      f"{ranges[0] / 1e3:.1f}-{ranges[-1] / 1e3:.1f} km")
print(f"  depth pixel {np.diff(depths)[0]:4.2f} m -> median step "
      f"{coarse_step:5.3f} dB")
print(f"  depth pixel {298 / 299:4.2f} m -> median step {fine_step:5.3f} dB"
      f"   (ratio {fine_step / coarse_step:.2f}; 0.50 is resolved, "
      f"1.00 is aliased)")

# ── B. One run, three bandwidths ────────────────────────────────────────────
# The whole band is traced once — Bellhop finds the arrivals and evaluates
# the phase term per frequency — so each band below is a window onto one
# result rather than another run.
band = np.arange(800.0, 1201.0, 20.0)
H = model.run(env, uacpy.Source(depths=source_depth, frequencies=band),
              receiver, run_mode=uacpy.RunMode.BROADBAND, frequencies=band)
cw = H.at(frequency=tone)
narrow = H.window(frequency=(tone - 25.0, tone + 25.0)).broadband_loss()
wide = H.window(frequency=(tone - 200.0, tone + 200.0)).broadband_loss()

print(f"\n  {'view':<38}{'TL spread':>11}{'vs CW':>9}")
print(f"  {'continuous wave':<38}{np.nanstd(cw.tl):9.2f} dB{'—':>9}")
for label, field in (('B =  50 Hz — a 20 ms burst', narrow),
                     ('B = 400 Hz — a wideband chirp', wide)):
    moved = np.abs(field.tl - cw.tl)
    print(f"  {label:<38}{np.nanstd(field.tl):9.2f} dB"
          f"{np.nanmedian(moved):7.2f} dB")
print("  a 20 ms burst carries 50 Hz, which is far too narrow to reach "
      "these\n  fringes — the duration has to beat the paths' delay gap, "
      "not the carrier")

fig, axes = plt.subplots(3, 1, figsize=(13, 14), sharex=True, sharey=True)
for ax, (title, field) in zip(axes, (
        (f'Continuous wave, {tone:.0f} Hz — every path interfering', cw),
        ('B = 50 Hz — Ainslie Eq. 11.46 over a 20 ms burst', narrow),
        ('B = 400 Hz — the same, over a wideband chirp', wide))):
    uacpy.plot_field(field, ax, env=env, vmin=45, vmax=100, title=title)
    ax.set_ylim(300.0, 0.0)
fig.tight_layout()
fig.savefig(OUT / 'example_45a_bandwidth_washes_interference.png', dpi=200,
            bbox_inches='tight')
plt.close(fig)

# ── C. The coherent band average against the incoherent stand-in ────────────
incoherent = model.run(env,
                       uacpy.Source(depths=source_depth, frequencies=tone),
                       receiver, run_mode=uacpy.RunMode.INCOHERENT_TL)
gap = np.abs(np.asarray(incoherent.tl) - wide.tl)
finite = np.isfinite(gap)
print("\n  Eq. 11.46 (coherent, band-averaged) vs Eq. 11.47 "
      "(RunMode.INCOHERENT_TL):")
print(f"    over the panel   median {np.median(gap[finite]):5.2f} dB, "
      f"largest {gap[finite].max():5.1f} dB")
for lo, hi in ((2.0, 25.0), (25.0, 100.0), (100.0, 300.0)):
    rows = (depths >= lo) & (depths < hi)
    slab = gap[rows]
    ok = np.isfinite(slab)
    print(f"    {lo:5.0f}-{hi:3.0f} m     median {np.median(slab[ok]):5.2f} "
          f"dB, largest {slab[ok].max():5.1f} dB")

fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True, sharey=True)
uacpy.plot_field(incoherent, axes[0], env=env, vmin=45, vmax=100,
                 title='RunMode.INCOHERENT_TL — Ainslie Eq. 11.47')
plot_field_difference(wide, incoherent, axes[1], env=env,
                      title='Eq. 11.46 minus Eq. 11.47 — the coherent '
                            'interference the incoherent step drops')
for ax in axes:
    ax.set_ylim(300.0, 0.0)
fig.tight_layout()
fig.savefig(OUT / 'example_45b_coherent_against_incoherent.png', dpi=200,
            bbox_inches='tight')
plt.close(fig)

# ── D. What one 20 ms burst at a stated source level delivers ───────────────
# There is no source-level argument: the level rides on the waveform's
# amplitude, because a source level of SL dB re 1 uPa at 1 m IS an rms
# pressure of 1e-6 * 10**(SL/20) Pa there.
rate, source_level = 8000.0, 190.0
_, pulse = tone_burst(tone, 20, rate)                # 20 cycles = 20 ms
duration = pulse.size / rate
unit = pulse / np.sqrt(np.mean(pulse ** 2))
with warnings.catch_warnings():        # the 1/df record holds the pulse
    warnings.simplefilter('ignore')
    exposure = H.sound_exposure_level(
        unit * 1e-6 * 10 ** (source_level / 20.0), rate)

# The transmission loss OF THIS SIGNAL — pass the waveform and it is
# weighted by its own spectrum, evaluated on the field's axis by the same
# DTFT the synthesis uses. This is the transient's TL in the ordinary sense:
# received level is SL - TPL, and the energy form below is SEL = ESL - TPL,
# with ESL = SL + 10log10(T) for constant power (Ainslie Eq. 3.155).
burst_loss = H.broadband_loss(waveform=unit, sample_rate=rate)
energy_source_level = source_level + 10.0 * np.log10(duration)
by_equation = energy_source_level - burst_loss.tl
residual = np.abs(exposure.dB - by_equation)
ok = np.isfinite(residual)

print(f"\n  one {duration * 1e3:.0f} ms burst at SL = {source_level:.0f} dB "
      f"re 1 uPa @ 1 m")
print(f"    its own TPL                 {np.nanmedian(burst_loss.tl):7.2f} dB"
      f"   (median over the panel)")
print(f"    ESL = SL + 10log10(T)       {energy_source_level:7.1f} dB "
      f"re 1 uPa^2 s")
print(f"    SEL, integrated from p(t)   "
      f"{np.nanmedian(exposure.dB):7.2f} dB")
print(f"    SEL, as ESL - TPL           {np.nanmedian(by_equation):7.2f} dB")
print(f"    the energy sonar equation closes to "
      f"{np.median(residual[ok]):.6f} dB median, "
      f"{residual[ok].max():.6f} dB largest")

# The SEL map is THIS loss map sign-flipped and shifted by ESL — which is
# what SEL = ESL - TPL says, so the correlation below is an identity check
# and not a discovery. It is specifically NOT panel A's middle map: that one
# averages a flat 50 Hz window, while the burst's own spectrum puts only
# part of its energy there and decays through the rest of the band.
same = np.corrcoef(burst_loss.tl[ok].ravel(), -exposure.dB[ok].ravel())[0, 1]
apart = np.nanmax(np.abs(burst_loss.tl - narrow.tl))
print(f"    SEL vs this loss map: correlation {same:+.4f} — the identity, "
      f"not a finding")
print(f"    it is NOT panel A's flat 50 Hz window: those two part by up to "
      f"{apart:.2f} dB")

# The metric the band average cannot give. Exposure criteria are stated as a
# PAIR — "frequency-weighted SEL and unweighted peak sound pressure level",
# either of which triggers (Southall et al. 2019) — and a peak is a property
# of the waveform in time, not of the energy in the band. Same synthesis,
# different reduction.
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    traces = H.synthesize_time_series(
        unit * 1e-6 * 10 ** (source_level / 20.0), rate, window='none')
peak = 20.0 * np.log10(np.max(np.abs(np.asarray(traces.data).real), axis=-1)
                       / 1e-6)
# A crest factor is deliberately NOT reported here. It needs the received
# pulse's own duration, and multipath stretches that well past the 20 ms
# transmitted — the path gap reaches 18.6 ms across this panel — so dividing
# by the transmitted length OVERstates the rms and understates the crest
# (6.62 dB against 7.18 dB measured on the received pulse's own support).
# The point of the panel does not need it: peak is a property of the
# waveform in time, so it is not recoverable from any band average, which
# is exactly why the criteria name both metrics.
good = np.isfinite(peak)
print(f"\n    peak SPL   {np.nanmedian(peak):7.2f} dB re 1 uPa (median), "
      f"{peak[good].max() - peak[good].min():.1f} dB across the panel")
print(f"    SEL        {np.nanmedian(exposure.dB):7.2f} dB re 1 uPa^2 s "
      f"(median) — the other half of the pair")
# Spearman from THIS run's arrays rather than a remembered figure — a
# caption and the figure it describes have to come from one run. Rank
# correlation is Pearson on the ranks, and a double argsort ranks without
# scipy or a helper.
rho = float(np.corrcoef(
    np.argsort(np.argsort(peak[good].ravel())),
    np.argsort(np.argsort(exposure.dB[good].ravel())))[0, 1])
print(f"    the two rank cells almost alike here (Spearman {rho:.2f}) — what "
      f"separates them\n    is that PEAK is a property of the waveform in "
      f"time, so no band average\n    yields it, which is why the criteria "
      f"name both metrics rather than one")

# ── E. The waveform behind the level, at one receiver ───────────────────────
# Every panel above is a reduction of the same H. Here is what it is a
# reduction OF: the signal itself at one position, transmitted and received.
# ``Field.to_time_trace`` takes the waveform and the receiver, and warns if
# that receiver is off the grid rather than silently using the nearest edge.
# (Example 19 does this across eight solvers; this is the one-receiver form
# on an H already in hand, with no second model run.)
sweep = 200.0
_, chirp = lfm_chirp(tone - sweep / 2.0, tone + sweep / 2.0, 0.020, rate)
look_depth, look_range = 150.0, 1500.0
# The map's frequency grid will NOT do for a waveform. Its record is 1/df
# long and circular, and the maps only need it to hold the 20 ms burst; this
# chirp plus its image tail overruns it, and the overrun does not vanish —
# it folds onto the record's start, where it reads as energy arriving before
# any path could. At the map's 20 Hz that is 1.8 % of the trace's energy.
# A fold is NOT harmless to the level maps either — sampling H every df
# periodises the response, so a late arrival adds COHERENTLY to the early
# part and the maps inherit the error (measured: -9.27 to +2.75 dB over 400
# wrapped delays). The SEL check above still closes because broadband_loss
# reads the same undersampled H, so both sides move together. What a fold
# ruins VISIBLY is the waveform. So the trace gets its own grid.
# Do not guess df. ``Arrivals.synthesis_band`` sizes it from the arrivals
# themselves — it measures where 99.9 % of their energy sits
# (``energy_support``) rather than assuming which paths exist, which is the
# difference between a record that holds this channel and one that holds the
# two paths I happened to think of. ``record=`` is stated because the helper
# sizes for the ARRIVALS and the transmitted pulse's own length has to be
# added to them.
look = uacpy.Receiver(depths=[look_depth], ranges=[look_range])
paths = model.run(env, uacpy.Source(depths=source_depth, frequencies=tone),
                  look, run_mode=uacpy.RunMode.ARRIVALS)
spread = paths.energy_support(0.999)
span = spread + chirp.size / rate
# ``record=`` is taken as given, bypassing synthesis_band's own ``margin``,
# so the headroom is supplied here. It is not a token factor: the pulse has
# to fit AND the rectangular band's precursor has to have somewhere to sit.
# Measured on this geometry, energy landing before the direct path could
# arrive ran 21.2 % at 1.5x the span, 0.185 % at 4x, and 0.189 % at 60x —
# so it converges at about 4x, and the 0.19 % floor is the band edge's own
# precursor rather than anything folded.
trace_band = paths.synthesis_band(bandwidth=2.0 * sweep, centre=tone,
                                  record=6.0 * span)
trace_df = float(np.diff(trace_band)[0])
trace_field = model.run(
    env, uacpy.Source(depths=source_depth, frequencies=trace_band),
    look, run_mode=uacpy.RunMode.BROADBAND, frequencies=trace_band)
direct = np.hypot(look_range, look_depth - source_depth)
image = np.hypot(look_range, look_depth + source_depth)
gap = (image - direct) / 1500.0
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    # nfft sets the output RATE (nfft*df), not the record length, which
    # stays 1/df. The automatic size only guarantees Nyquist; a display of
    # the waveform wants several samples per cycle, not two.
    sent = trace_field.to_time_trace(
        depth=look_depth, range=look_range, waveform=chirp,
        sample_rate=rate, window='none', nfft=8192)
received = np.asarray(sent.data).real
clock = np.asarray(sent.coords['time'])
# The record's own audit. Nothing can arrive before the direct path, so
# energy sitting there came from somewhere it should not. HOW MUCH is not
# enough to say what it is — a band edge's precursor and a folded tail both
# show up as a percentage — so WHERE it sits decides: a precursor hugs the
# arrival, a fold is parked back at the record's start. Measured on this
# geometry, a 35 ms record put 21.2 % of the energy there with its median
# 18.7 ms before the arrival and only 0.9 % of it within 5 ms; a 137 ms one
# puts 0.18 % there with the median 0.3 ms out and 94.8 % within 5 ms.
onset = direct / 1500.0
early = clock < onset
power = received ** 2
leak = power[early].sum() / power.sum()
ahead = onset - clock[early]
# The energy-weighted MEDIAN, not the mean: the mean is pulled toward the
# record's far end by the little energy out there and separates the two
# cases by about 10x, where the median separates them by 64x (18.67 ms
# folded against 0.29 ms for a precursor). The 5 ms mark is 1/B, the width
# a precursor has; it discriminates over any threshold from 1 to 10 ms, but
# it must stay small against the RECORD — at 20 ms against a 34 ms record
# the folded case reads 73 % and the verdict flips.
order = np.argsort(ahead)
share = np.cumsum(power[early][order]) / power[early].sum()
middle = float(ahead[order][int(np.searchsorted(share, 0.5))])
hugging = float(power[early][ahead < 0.005].sum() / power[early].sum())
print(f"\n  a {sweep:.0f} Hz chirp received at {look_depth:.0f} m, "
      f"{look_range / 1e3:.1f} km")
print(f"    transmitted {chirp.size / rate * 1e3:5.1f} ms, "
      f"received span {(clock[-1] - clock[0]) * 1e3:5.1f} ms at "
      f"{1e-3 / float(np.diff(clock)[0]):.1f} kHz")
print(f"    the low ripple before the arrival is the rectangular band edge "
      f"at\n    {band[0]:.0f}/{band[-1]:.0f} Hz — window='none' keeps the "
      f"energy exact and pays a sinc skirt")
# The image path is the one reflected off the surface: it leaves the source
# as if from source_depth ABOVE it, so its extra length is the difference of
# the two slant ranges, and the gap is that over the sound speed.
print(f"    {len(paths.delays)} arrivals, 99.9 % of their energy within "
      f"{spread * 1e3:.2f} ms")
print(f"    synthesis_band chose {trace_df:.2f} Hz -> a "
      f"{1 / trace_df * 1e3:.0f} ms record for {span * 1e3:.1f} ms of "
      f"pulse + multipath")
print(f"    energy before the direct path: {100 * leak:.3f} %, median "
      f"{middle * 1e3:.2f} ms ahead,")
verdict = ('a precursor, not a fold' if hugging > 0.5
           else 'PARKED AT THE RECORD START — FOLDED')
print(f"    {100 * hugging:.1f} % of it within 5 ms of the arrival -> "
      f"{verdict}")
print(f"    peak at {clock[int(np.argmax(np.abs(received)))] * 1e3:8.2f} ms"
      f"   direct path r/c = {direct / 1500.0 * 1e3:.2f} ms")
# Whether the two arrive as one echo or two is set by the gap against the
# COMPRESSED length 1/B, not against the transmitted 20 ms: a receiver that
# matched-filters the chirp resolves 1/B. Below 1, they overlap and
# interfere — which is the same criterion Field.truncate_response applies.
print(f"    the surface image trails it by {gap * 1e3:.3f} ms, "
      f"{gap * sweep:.2f} of the chirp's")
print(f"    {1 / sweep * 1e3:.2f} ms compressed length — under 1, so the two "
      f"overlap and interfere\n    rather than landing as separate echoes")

fig, axes = plt.subplots(2, 1, figsize=(11, 7))
plot_waveform(
    chirp, rate, axes[0], time_units='ms',
    title=f'Transmitted: {sweep:.0f} Hz chirp, '
          f'{chirp.size / rate * 1e3:.0f} ms')
plot_waveform(
    received, 1.0 / float(np.diff(clock)[0]), axes[1],
    t0=float(clock[0]), time_units='ms',
    title=f'Received at {look_depth:.0f} m, {look_range / 1e3:.1f} km — '
          f'the direct path and its surface image, {gap * 1e3:.2f} ms apart')
fig.tight_layout()
fig.savefig(OUT / 'example_45d_received_chirp.png', dpi=200,
            bbox_inches='tight')
plt.close(fig)

fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True, sharey=True)
uacpy.plot_field(exposure, axes[0], env=env,
                 title=f'Sound exposure level of one '
                       f'{duration * 1e3:.0f} ms burst at '
                       f'SL = {source_level:.0f} dB re 1 uPa @ 1 m')
uacpy.plot_field(
    uacpy.core.results.Field(
        data=peak, coords={'depth': depths, 'range': ranges},
        metadata={'kind': 'level', 'unit': 'dB'}),
    axes[1], env=env,
    title='Peak sound pressure level of the same burst — the other half of '
          'the dual metric, and not a band average')
for ax in axes:
    ax.set_ylim(300.0, 0.0)
fig.tight_layout()
fig.savefig(OUT / 'example_45c_exposure_and_peak.png', dpi=200,
            bbox_inches='tight')
plt.close(fig)
