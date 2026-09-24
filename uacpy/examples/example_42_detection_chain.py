"""Propagation to detection, end to end, on one receive array.

The whole chain a passive sonar problem asks for, in the order it is asked:
where the geometry sits, how much the channel takes away, how much the array
gives back, and what is left to detect with.

    SE = SL - TL - (NL - AG) - DT          (the passive sonar equation)

Every term is computed rather than assumed, and the array term is the one
worth care. Start with the ceiling: at lambda/2 spacing in isotropic noise
every inter-element noise correlation vanishes, sinc(k*d) = sinc(pi) = 0, so
"the array gain is 10 log N as it is for completely incoherent noise"
(Butler & Sherman, *Transducers and Arrays for Underwater Sound*, 8.3.1).
For 24 elements that is 13.80 dB, and the Hann taper spends 1.95 dB of it
on sidelobes, leaving 11.86 dB for a signal arriving as one plane wave on
the steered axis. One caveat travels with that and is not this example's to
fix: the same section notes that "sea noise is probably never isotropic".

It is not the whole story on the SIGNAL side. AG "depends on the beam
pattern ... and on the directional properties of the signal and noise
fields", and is "arguably the most difficult term of the passive sonar
equation to calculate precisely" (Ainslie, *Sonar Performance Modelling*,
6.1); it measures "the coherence of the signal of interest with respect to
the coherence of the noise across the line array" (Stergiopoulos, *Advanced
Signal Processing Handbook*, 11.1.2). Replacing AG by the directivity index
is exact only "in isotropic noise and plane-wave propagation across the
array" (Abraham, 2.3.2, which defers the detail to his 8.4) — and a
waveguide delivers no single plane wave: every trapped mode arrives at its
own +/- grazing angle, so one beam holds the modes inside its main lobe and
loses the rest.

Be precise about whose fault that is. "The narrowband conventional
beamformer is itself a matched filter where the replica is obtained by
exploiting the form of the signal enforced by propagation ... using plane
waves as a solution when the array is in the far field of the source"
(Abraham, 8.4.2). The plane wave is the replica, not the physics. In a
waveguide the right replica is the channel's Green's function, which is
matched-field processing (Etter, *Underwater Acoustic Modeling and
Simulation*, 11.5.7.1, after Baggeroer et al. 1993). Feeding the modelled
field back as its own replica returns 13.80 dB at every range — exactly
10 log N, an identity rather than a measurement, but it shows the bound is
reachable with no plane wave in sight. The 3.2 dB the scan gives up is the
price of the plane-wave replica, not something the channel takes away.

Ainslie's Equation (6.70) says what to do instead: "the signal-to-noise
ratio must be calculated not just once, but twice, with and without the
effects of the beamformer". That is the realised gain below — the best
beam's output against the per-element mean, on the modelled field. It comes
out near 8.7 dB and does not decay with range: the trend over 20 km is
+0.07 dB/km against 1.8 dB of scatter, so the gain fluctuates rather more
than it drifts. Taking the plane-wave value anyway stretches the predicted
detection range at 60 m from 2.7 to 4.5 km, a 40 % overestimate, and
doubles the share of the water column it calls detectable, 13 % to 26 %.

Keeping the largest of many beams is many chances to false-alarm, so the
threshold pays for it. The independent looks are not the 361 grid points,
and not even the orthogonal count: two beams' NOISE outputs correlate as
the array factor of |w|^2, and hann^2 reaches its first null at three DFT
bins against a rectangular window's one, so this sector holds about 5
independent looks. Holding the whole scan at Pf = 1e-4 then needs
Pf = 1.8e-5 in each: +0.45 dB of DT, and it is what the maps use.

One anchor for the whole figure. The array is the fixed object, at r = 0,
drawn as the receiver it is; every point of both maps is a candidate TARGET
position, and the range axis is the separation. Panel 4 is panel 1 pushed
through the sonar equation, and both contour the SIGNAL EXCESS at 0 rather
than standing an iso-TL level in for it — AG is a grid spanning 6 dB here,
so the single TL level that looks like the boundary sits at a median SE of
-1.1 dB and claims break-even where the target is a decibel short. Each
panel also carries the dashed line a constant AG would have promised, and
the gap between the pair is what this example is about.

Getting the 24-element field for a target at every point is one run, by
reciprocity: the SOURCES go at the array's own element depths, at r = 0
where the array is. Which end carries them is not cosmetic on a bottom that
varies in range — see `example_43`, which measures what the mirror
arrangement costs. Nothing here is smoothed: the fringes are real, and
sigma_dB in the P_D map is the honest place to put how far a single
prediction can be trusted.

The bottom is four regimes in one section — a flat shelf under the array, a
sill at 130 m, a basin behind it at 175 m, then a shoaling run to 120 m —
because a monotonic slope never cuts a mode off and then gives it back.
Coupled modes throughout: adiabatic "assumes that all energy in a given
mode transfers to the corresponding mode in the new environment, provided
that environmental variations in range are gradual" (Etter 4.4.5), and a
sill leaves no corresponding mode to transfer to. Measured over the water
column the two disagree by 2.0 dB in the median. `example_44` runs the same
chain through a second propagation model and synthesises what the beam
hears.

Uses: Environment.plot(source=, receiver=, source_marker_range_m=) ·
Kraken(mode_coupling='coupled') · beamform_field · plane_wave_array_gain ·
matched_replica_gain · independent_beams(weights=) · shading_taper ·
sonar.detection_threshold_energy · per_look_false_alarm ·
passive_signal_excess_field(array_gain=<grid>) ·
probability_of_detection_field · detection_range_by_depth ·
plot_field / plot_detection_probability with env/receiver overlays
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.visualization import plot_detection_probability, plot_field
from uacpy.acoustic_signal import (beamform_field, independent_beams,
                                   matched_replica_gain,
                                   plane_wave_array_gain, shading_taper)
from uacpy.core.results import Field
from uacpy.sonar import (detection_threshold_energy,
                         passive_signal_excess_field, per_look_false_alarm,
                         probability_of_detection_field)
from uacpy.sonar.sonar_equation import detection_range_by_depth

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

FREQ, C_REF = 200.0, 1500.0
HALF = 0.5 * C_REF / FREQ

# ── the scene ───────────────────────────────────────────────────────────
# A downward-refracting summer profile over a sand seabed: the surface duct
# is gone, so energy leaves the surface and the shadow it leaves is what the
# detection map has to live with.
#
# The bottom is the interesting part. Four regimes in one section — a flat
# shelf under the array, a sill rising to 130 m, a basin behind it at 175 m,
# then a shoaling run to 120 m — because a monotonic slope lets too much
# slide: it never cuts a mode off and then gives it back, and it cannot
# show a feature casting a shadow behind itself.
#
# The relief is sized by the solver, not by taste. uacpy segments a
# range-dependent section on a quarter-wavelength-of-depth-change criterion
# and warns when that asks for more profiles than its 200 ceiling; a
# steeper version of this bottom (a 95 m sill, 240 m of total relief) trips
# that warning at 200 Hz, and its field is then not converged. 170 m of
# total relief stays inside it.
BATHY_NODES_M = np.array([0.0, 3000.0, 6000.0, 9000.0,
                          11000.0, 13500.0, 16500.0, 20000.0])
BATHY_NODE_DEPTHS = np.array([200.0, 200.0, 160.0, 130.0,
                              150.0, 175.0, 145.0, 120.0])
BATHY_RANGES = np.linspace(0.0, 20000.0, 41)
BATHY_DEPTHS = np.interp(BATHY_RANGES, BATHY_NODES_M, BATHY_NODE_DEPTHS)
env = uacpy.Environment(
    name='sill-and-basin',
    bathymetry=list(zip(BATHY_RANGES, BATHY_DEPTHS)),
    ssp=[(0.0, 1520.0), (30.0, 1518.0), (80.0, 1502.0), (200.0, 1498.0)],
    bottom=uacpy.Bottom.from_halfspace(uacpy.BoundaryProperties(
        acoustic_type='half-space', sound_speed=1700.0, density=1.9,
        attenuation=0.5)),
    absorption=uacpy.Thorp(),
)


SL, NL = 120.0, 75.0          # band-integrated, dB re 1 µPa²
target = uacpy.Source(depths=60.0, frequencies=FREQ, source_level_dB=SL)

n_el = 24
elements = 120.0 + HALF * (np.arange(n_el) - (n_el - 1) / 2.0)
# The range axis is the target-to-array separation, and the ARRAY is the end
# anchored at r = 0 — see the run below, which is what makes this correct on
# a bottom that changes with range. The scene draws one of those geometries.
array = uacpy.Receiver(depths=elements, ranges=np.linspace(200.0, 20000.0, 200))
# The array is the anchored end, so the scene draws it at r = 0 and puts
# the target out at range — the same way round as every panel and as the
# run itself. Drawing the target at the origin instead would show the
# mirror-image geometry, which on this slope stands the array in 170 m of
# water when it is really in 200 m.
SCENE_R = 5000.0
# The array as a receiver AT THE ORIGIN: the marker set every map in this
# example is drawn with, since the array is the end that does not move.
array_at_origin = uacpy.Receiver(depths=elements, ranges=[0.0])
scene_array = array_at_origin

fig, ax = env.plot(source=target, receiver=scene_array,
                   source_marker_range_m=SCENE_R,
                   title=f'The scene: the {n_el}-element array at 120 m in '
                         f'{BATHY_DEPTHS[0]:.0f} m of water, a {SL:g} dB '
                         f'target {SCENE_R / 1e3:.0f} km up the slope at 60 m')
fig.savefig(OUT / 'example_42_scene.png', dpi=140, bbox_inches='tight')
plt.close(fig)

# ── 1. propagation ──────────────────────────────────────────────────────
# One run gives the whole problem. The SOURCES go at the array's own element
# depths, at r = 0 where the array is; the receivers cover the candidate
# target plane. Reciprocity then makes slab i the field element i would
# measure from a target at each point — and it holds whatever the bottom
# does in range, which the mirror-image alternative does not.
# Coupled modes, not the adiabatic default. The adiabatic approximation
# "assumes that all energy in a given mode transfers to the corresponding
# mode in the new environment, provided that environmental variations in
# range are gradual" (Etter, *Underwater Acoustic Modeling and
# Simulation*, 4.4.5) — and a sill that shoals to 130 m cuts the steeper
# modes off, leaving no corresponding mode to transfer to. The
# example measures the difference below rather than asserting it.
kraken = uacpy.Kraken(verbose=False, mode_coupling='coupled')
plane_depths = np.union1d(np.linspace(5.0, 195.0, 60),
                          np.atleast_1d(target.depths))
plane = uacpy.Receiver(depths=plane_depths, ranges=array.ranges)
array_as_sources = uacpy.Source(depths=elements, frequencies=FREQ)
stack = kraken.run(env, array_as_sources, plane)   # n_el slabs of (n_z, n_r)

# Each element's depth travels with its own field; 1 mm rather than exact
# because the depth round-trips through the model deck's text format.
for i, (z_el, _) in enumerate(stack):
    assert abs(z_el - elements[i]) < 1e-3, 'stack order'
p_map = np.stack([np.asarray(f.data) for _, f in stack])   # (n_el, n_z, n_r)

# A target cannot be inside the sediment, so the shoaling bottom takes a
# bite out of the plane. Everything below is masked, not merely ignored:
# the model returns the sub-bottom evanescent tail there, about 30 dB
# quieter, which would otherwise draw as ordinary shadow.
# env.bathymetry.eval, not a hand-rolled interp: the mask then cannot drift
# from the seafloor the solver itself sees.
seabed = np.asarray(env.bathymetry.eval(range=array.ranges), dtype=float)
in_water = plane_depths[:, None] <= seabed[None, :]
print(f"the shoaling bottom puts {100 * (~in_water).mean():.0f} % of the "
      f"({plane_depths.size} x {array.n_ranges}) plane inside the seabed")

i60 = int(np.argmin(np.abs(plane_depths - np.atleast_1d(target.depths)[0])))
p = p_map[:, i60, :]                      # the 60 m target, on all elements

# What the adiabatic default would have said, for the mid-array element.
mid = n_el // 2
adiabatic = np.asarray(uacpy.Kraken(verbose=False).run(
    env, uacpy.Source(depths=float(elements[mid]), frequencies=FREQ),
    plane).data)
d_mode = np.where(in_water,
                  np.abs(-20.0 * np.log10(np.abs(adiabatic))
                         + 20.0 * np.log10(np.abs(p_map[mid]))), np.nan)
print(f"adiabatic vs coupled on this bathymetry: median "
      f"{np.nanmedian(d_mode):.1f} dB, 90th pct "
      f"{np.nanpercentile(d_mode, 90):.1f} dB — the sill cuts modes off, "
      f"which is the\n  one thing the adiabatic assumption cannot follow")

# ── 2. array processing: the gain the beamformer actually realises ──────
# AG is a ratio of SNRs, so it has a signal half and a noise half.
#
#   noise:  at lambda/2 spacing the coherence of 3-D isotropic noise is
#           sinc(k*d) = sinc(pi) = 0, so the elements are uncorrelated and a
#           unit-norm weight vector leaves the noise power unchanged.
#   signal: a matched plane wave on the steered axis gives |sum w_n|^2.
#
# so AG = |sum w_n|^2 / ||w||^2 for a unit-norm w. NOT -10log10(sum|w|^4),
# which agrees only for an unshaded array and is 1.1 dB low for a Hann one.
taper = shading_taper(n_el, 'hann')
ag_plane = plane_wave_array_gain(taper)
print(f"AG for a matched plane wave, white noise: {ag_plane:.2f} dB "
      f"({10 * np.log10(n_el):.2f} dB unshaded — the Hann taper costs "
      f"{10 * np.log10(n_el) - ag_plane:.2f} dB)")

# But the channel does not deliver ONE plane wave. Each trapped mode arrives
# at its own +/- grazing angle, so a single beam holds only the modes inside
# its main lobe and the rest is lost. Measure what the beam actually gets:
# the best beam's power against a single element at the array centre.
angles = np.linspace(-45.0, 45.0, 361)
row = beamform_field(p, elements, angles, FREQ, c=C_REF, weights=taper)
ag_realised = row.array_gain()                             # per range

# And the ceiling, which is worth computing rather than asserting. At
# lambda/2 spacing in isotropic noise every inter-element noise correlation
# vanishes, so the noise sum is N; a replica matched to the field makes the
# signal sum N^2, and AG = 10 log N exactly, whatever shape the field has
# (Butler & Sherman, *Transducers and Arrays*, 8.3.1). That replica is the
# channel's Green's function — which is matched-field processing, and which
# a propagation model is precisely what hands you.
ag_matched = matched_replica_gain(p)
print(f"AG a MATCHED replica realises: {np.median(ag_matched):.2f} dB at "
      f"every range (spread {np.ptp(ag_matched):.1e} dB) "
      f"= 10log10({n_el}) = {10 * np.log10(n_el):.2f} dB")

# The plane-wave replica is the right one only in the array's far field.
aperture = float(elements.max() - elements.min())
r_far = 2.0 * aperture ** 2 / (C_REF / FREQ)
inside = np.asarray(array.ranges) < r_far
print(f"  the plane-wave replica assumes the far field, 2L^2/lambda = "
      f"{r_far:.0f} m for this {aperture:.0f} m aperture; inside that the "
      f"scan still gets {np.median(ag_realised[inside]):.1f} dB against "
      f"{np.median(ag_realised):.1f} dB overall, so wavefront curvature is "
      f"not what costs the gain — the mode spread is")
best_angle = row.best_angle                                # and where it looks
km_axis = np.asarray(array.ranges) / 1e3
fit = np.polyfit(km_axis, ag_realised, 1)
trend = float(fit[0])
scatter = float(np.std(ag_realised - np.polyval(fit, km_axis)))
print(f"AG the beam REALISES: median {np.median(ag_realised):.1f} dB, "
      f"spread {ag_realised.min():.1f}-{ag_realised.max():.1f} dB")
print(f"  -> {ag_plane - np.median(ag_realised):.1f} dB below the plane-wave "
      f"value. Over {km_axis[-1]:.0f} km the trend is {trend:+.3f} dB/km "
      f"= {trend * km_axis[-1]:+.1f} dB,")
print(f"     against {scatter:.1f} dB of scatter about it — so the gain "
      f"fluctuates more than it drifts,\n     and a range-varying AG is "
      f"still not a decaying one")
# What the re-steering is worth: the best single fixed beam, for comparison.
fixed_i = int(np.argmax(row.power.mean(axis=1)))
ag_fixed = 10.0 * np.log10(row.power[fixed_i] / row.element_power)
print(f"  re-steering per range is worth "
      f"{np.median(ag_realised) - np.median(ag_fixed):+.1f} dB over the best "
      f"fixed beam (at {angles[fixed_i]:+.1f} deg)")

# ── 3. detection, both ways ─────────────────────────────────────────────
PD, PF = 0.5, 1e-4
BW_HZ, INT_S = 10.0, 10.0
dt_one_beam = detection_threshold_energy(pd=PD, pf=PF, bandwidth_hz=BW_HZ,
                                         integration_time_s=INT_S)
print(f"detection threshold for Pd={PD}, Pf={PF:g}, "
      f"{BW_HZ:g} Hz x {INT_S:g} s (M={BW_HZ * INT_S:g}): "
      f"{dt_one_beam:.2f} dB on ONE beam")

# But the gain above is the best of the whole scan, and keeping the largest
# of many beams is many chances to false-alarm. The independent looks are
# not the 361 grid points: orthogonal beams sit lambda/(N*d) apart in
# sin(theta), so the scanned sector holds only that many resolution cells.
# Hold the Pf of a whole scan at PF by tightening each beam's own.
# weights= matters: two beams' noise correlates as the array factor of
# |w|^2, and hann^2 is three DFT bins wide against a rectangular window's
# one, so this scan holds about a third of the looks the unshaded geometry
# would give. Leaving the taper out over-counts and sets a needlessly
# strict threshold.
n_beams_independent = independent_beams(elements, angles, FREQ, c=C_REF,
                                        weights=taper)
pf_per_beam = per_look_false_alarm(PF, n_beams_independent)
dt = detection_threshold_energy(pd=PD, pf=pf_per_beam, bandwidth_hz=BW_HZ,
                                integration_time_s=INT_S)
print(f"  the {angles.size}-point scan is {n_beams_independent:.0f} "
      f"INDEPENDENT beams, so each needs Pf={pf_per_beam:.1e} to hold the "
      f"scan at {PF:g}: DT = {dt:.2f} dB, {dt - dt_one_beam:+.2f} dB")

# ── the coverage map: hold the ARRAY still and move the TARGET ──────────
# The colour at (r, z) is a target AT that point, heard by the one array at
# r = 0 — not a receiver at that point. The field for it was computed in
# section 1; all that is left is to run the beamformer over the plane and
# mask the part of it that is now inside the seabed.
# Nothing is smoothed: the fringes are real, and the sigma_dB below is the
# right place to say how far a single prediction can be trusted.


# The beamformer over the whole plane. Seabed cells go NaN, which the
# sonar-equation grid carries through: _budget_array_gain summarises an AG
# grid with nan-aware statistics for exactly this case.
scan = beamform_field(p_map, elements, angles, FREQ, c=C_REF, weights=taper)
with np.errstate(divide='ignore', invalid='ignore'):
    ag_map = np.where(in_water, scan.array_gain(), np.nan)
    tl_map = np.where(in_water, -10.0 * np.log10(
        np.maximum(scan.element_power, 1e-300)), np.nan)
# kind defaults to 'pressure', whose dB unit is spelled "TL (dB)" — which
# is exactly what this holds.
tl_cov = Field(data=tl_map,
               coords={'depth': plane_depths,
                       'range': np.asarray(array.ranges, dtype=float)},
               model='Kraken')
print(f"realised AG over the water column: median "
      f"{np.nanmedian(ag_map):.1f} dB, {np.nanpercentile(ag_map, 5):.1f}-"
      f"{np.nanpercentile(ag_map, 95):.1f} dB over 5-95 %")

se_map, pd_map = {}, {}
for label, gain in (('constant', ag_plane), ('realised', ag_map)):
    se_map[label] = passive_signal_excess_field(
        tl_cov, source_level=SL, noise_level=NL, array_gain=gain,
        detection_threshold=dt)
    pd_map[label] = probability_of_detection_field(se_map[label], sigma_dB=8.0)
    # Over the WATER, not the whole rectangle: a masked cell is seabed, not
    # an undetected target, and counting it as one would dilute the number
    # with geology.
    se_water = np.asarray(se_map[label].data)[in_water]
    covered = 100.0 * np.mean(se_water > 0.0)
    print(f"map, {label:8s} AG: {covered:4.0f} % of the plane above threshold")

_, r_by_z_const = detection_range_by_depth(se_map['constant'])
_, r_by_z_real = detection_range_by_depth(se_map['realised'])
# np.inf is a legitimate return from detection_range_by_depth — SE >= 0 at
# every sampled range — and nanmedian does NOT drop it. The median survives
# a few, but silently: with a third of rows saturated the headline would
# read inf. Count them and take the median over the finite rows.
both_finite = np.isfinite(r_by_z_const) & np.isfinite(r_by_z_real)
saturated = int(np.sum(np.isinf(r_by_z_const) | np.isinf(r_by_z_real)))
gap = np.nanmedian((r_by_z_const - r_by_z_real)[both_finite])
print(f"  -> the constant AG adds {gap / 1e3:+.1f} km to the median "
      f"detection range across target depth "
      f"({int(both_finite.sum())} rows; {saturated} saturated the range "
      f"axis and are excluded)")

# One definition of detection range for the whole example: read off the
# same field the map draws, so this number and that contour cannot drift
# apart. It is the OUTERMOST zero crossing, which on a fringing curve is a
# tail statistic — a single constructive fringe moves it by a kilometre, so
# it is a fair headline only alongside the map that shows the fringes.
r_const, r_real = r_by_z_const[i60], r_by_z_real[i60]
print(f"detection range at 60 m: {r_const / 1e3:.1f} km with a constant AG, "
      f"{r_real / 1e3:.1f} km with the realised one "
      f"({100 * (r_const - r_real) / r_const:+.0f} %)")

# ── 4. the figure ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8.4))
# The array is the fixed object on every map, drawn as the receiver it is
# at r = 0; every point of either map is a candidate TARGET position. Panel
# 4 is panel 1 pushed through the sonar equation, on the same axes.
z_row = float(np.atleast_1d(target.depths)[0])
# The colour window is set from the data: the default 20-120 dB spends two
# thirds of the map on levels this channel never reaches, which flattens
# every feature into one green.
lo, hi = np.nanpercentile(np.asarray(tl_cov.data), [1, 99])
plot_field(tl_cov, ax=axes[0, 0], env=env, receiver=array_at_origin,
           vmin=float(np.floor(lo / 5) * 5), vmax=float(np.ceil(hi / 5) * 5),
           title='1. Propagation — target-to-array loss, mean of 24 elements')
# The break-even boundary, contoured on the SIGNAL EXCESS itself — not on
# an iso-TL level standing in for it. SE = 0 is TL = SL - NL + AG - DT, and
# the AG in that expression is a GRID spanning 6 dB over this plane, so a
# single iso-TL line is not that boundary: drawn at the median AG it sits
# at a median SE of -1.1 dB, claiming break-even where the target is a
# decibel short. Contouring SE puts panel 4's own line on panel 1, which is
# what this panel claims to show. It crosses 5 dB of TL along its length,
# so it cannot be misread as a level of the plotted field — provided the
# label does not assert a dB value, which is why it does not.
km_panel = np.asarray(array.ranges) / 1e3
cs = axes[0, 0].contour(km_panel, plane_depths,
                        np.asarray(se_map['realised'].data), levels=[0.0],
                        colors='k', linewidths=1.3, zorder=4)
axes[0, 0].clabel(cs, fmt='SE = 0', fontsize=7, inline_spacing=2)
# And the same pair panel 4 draws, for the same reason: what a constant AG
# would have promised. The gap between the two lines IS the example.
axes[0, 0].contour(km_panel, plane_depths,
                   np.asarray(se_map['constant'].data), levels=[0.0],
                   colors='C3', linestyles='--', linewidths=1.4, zorder=4)
axes[0, 0].plot([], [], 'k-', lw=1.3, label='break even, SE = 0')
axes[0, 0].plot([], [], 'C3--', lw=1.4,
                label=f'SE = 0 if AG were constant ({ag_plane:.1f} dB)')
axes[0, 0].plot([], [], 'k:', lw=0.8, label=f'the {z_row:.0f} m row')
# Upper right: the lower edge belongs to the seabed band, which swallowed
# the second entry there.
axes[0, 0].legend(fontsize=7, loc='upper right', framealpha=0.85)

ax = axes[0, 1]
i5 = int(np.argmin(np.abs(np.asarray(array.ranges) - 5000.0)))
# at= picks the range out of the scanned plane; the plotter normalises to
# the peak, which is how a beam is read.
row.plot(ax=ax, at=i5, color='C0', lw=1.3)
ax.set(xlabel='Look angle (deg)', ylabel='Beam power (dB re max)',
       ylim=(-30, 2), title='2. The arrival fan at 5 km — wider than one beam')
ax.grid(alpha=0.3)

ax = axes[1, 0]
km = np.asarray(array.ranges) / 1e3
ax.plot(km, ag_realised, 'C0-', lw=1.3, label='realised by the best beam')
ax.axhline(float(np.median(ag_matched)), color='C4', ls='-.', lw=1.3,
           label=f'matched replica = 10log10(N) ({np.median(ag_matched):.1f} dB)')
ax.axhline(ag_plane, color='C3', ls='--', lw=1.2,
           label=f'matched plane wave ({ag_plane:.1f} dB)')
ax.axhline(float(np.median(ag_realised)), color='C0', ls=':', lw=1.0,
           label=f'median realised ({np.median(ag_realised):.1f} dB)')
ax.set(xlabel='Range (km)', ylabel='Array gain (dB)',
       title='3. What the scan realises, against what is on the table')
ax.legend(fontsize=8, loc='lower left')
ax.grid(alpha=0.3)
twin = ax.twinx()                      # where the winning beam is looking
twin.plot(km, best_angle, '.', ms=2.0, color='C2', alpha=0.6)
twin.set_ylabel('Best look angle (deg)', color='C2')
twin.tick_params(axis='y', labelcolor='C2')

axd = axes[1, 1]
plot_detection_probability(pd_map['realised'], ax=axd, env=env,
                           receiver=array_at_origin, contour_levels=(0.5,),
                           title=f'4. Pd vs TARGET position, realised AG '
                                 f'(array at r=0, scan Pf={PF:g})')
# What assuming the plane-wave AG would have promised, on the same axes.
axd.contour(km, plane_depths, np.asarray(pd_map['constant'].data), levels=[0.5],
            colors='C3', linestyles='--', linewidths=1.4, zorder=4)
axd.plot([], [], 'C3--', lw=1.4, label='P_D = 0.5 if AG were constant')
# The 60 m row is the one panels 1-3 work on, so it is marked: the signal
# excess they compute is this line through this map.
for a in (axes[0, 0], axd):
    a.axhline(z_row, color='k', lw=0.8, ls=':', alpha=0.55, zorder=5)
axd.plot([], [], 'k:', lw=0.8, label=f'the {z_row:.0f} m row panels 2-3 use')
# Upper right: the lower left holds the contours, and the seabed band below
# the axis swallowed the text there.
axd.legend(fontsize=7, loc='upper right', framealpha=0.85)
fig.tight_layout()
fig.savefig(OUT / 'example_42_chain.png', dpi=140, bbox_inches='tight')
plt.close(fig)
print(f"\nFigures written to {OUT}")
