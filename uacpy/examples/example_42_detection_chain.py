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
For 24 elements that is 13.80 dB, and the Hann taper this example uses
spends 1.95 dB of it on sidelobes, leaving 11.86 dB for a signal that
arrives as one plane wave on the steered axis.

One caveat travels with that number and is not this example's to fix: the
same section notes that "sea noise is probably never isotropic". A vertical
array under surface-generated noise, whose intensity goes as cos^2(theta)
about the vertical, sees correlations that do not vanish at lambda/2, so the
noise half of AG would move too. Everything below holds the noise isotropic
and varies only the signal.

It is not the whole story on the SIGNAL side, and the literature is blunt
about why. AG "depends on the beam pattern ... and on the directional
properties of the signal and noise fields", and is "arguably the most
difficult term of the passive sonar equation to calculate precisely"
(Ainslie, *Sonar Performance Modelling*, 6.1); it measures "the coherence
of the signal of interest with respect to the coherence of the noise across
the line array" (Stergiopoulos, *Advanced Signal Processing Handbook*,
11.1.2). Replacing AG by the directivity index is exact only "in isotropic
noise and plane-wave propagation across the array" (Abraham, 2.3.2, which
defers the detail to his 8.4) — and a
waveguide delivers no single plane wave: every trapped mode arrives at its
own +/- grazing angle, so one beam holds the modes inside its main lobe and
loses the rest.

Be precise about whose fault that is. "The narrowband conventional
beamformer is itself a matched filter where the replica is obtained by
exploiting the form of the signal enforced by propagation ... using plane
waves as a solution when the array is in the far field of the source"
(Abraham, 8.4.2). The plane wave is the replica, not the physics. In a
waveguide the right replica is the channel's Green's function, which is
matched-field processing — algorithms that "exploit the full-field
structure of the signals propagating in an ocean waveguide", their replica
"derived from the Green's function" (Etter, *Underwater Acoustic Modeling
and Simulation*, 11.5.7.1, after Baggeroer et al. 1993).

So the ceiling is reachable, and the example evaluates it: feeding the
modelled field back as its own replica returns 13.80 dB at every range, to
1e-14 dB. That agreement is an identity, not a measurement — with a replica
proportional to the field, the ratio is N before any physics enters, and
1e-14 is double-precision round-off. What the evaluation shows is that the
bound is 10 log N with no plane wave anywhere in sight. The 3.2 dB the scan
gives up is the price of the plane-wave replica, not something the channel
takes away. A propagation model is what hands you the better replica, which
is the argument for computing this term rather than tabulating it.

Ainslie's Equation (6.70) says what to do instead: "the signal-to-noise
ratio must be calculated not just once, but twice, with and without the
effects of the beamformer". That is what the realised gain below is — the
best beam's output against the per-element mean, on the modelled field.

It comes out near 8.7 dB against the 11.9 dB a matched plane wave would
give, and it does not decay with range. Fitted over 20 km the trend is
+0.07 dB/km — upward, as the shoaling sections strip the steepest modes and
narrow the arrival fan — but that is +1.5 dB against 1.8 dB of scatter
about the fit, so the gain fluctuates rather more than it drifts. The
deficit is not something that accumulates with distance: a multi-mode
arrival never matches one plane wave, and which modes fall inside the beam
changes range to range with the interference. Taking the plane-wave value
anyway stretches the predicted detection range at 60 m from 2.7 to 4.5 km,
a 40 % overestimate, and doubles the share of the water column it calls
detectable, from 13 % to 26 %.

Two limits on the plane-wave scan are worth naming, because only one of
them binds. The replica is a far-field one, and this 86 m aperture has a
far-field distance 2L^2/lambda = 1984 m, so the inner 2 km of the map sits
inside it. That is not where the gain is lost, though: the ~3 dB shortfall
is present at all 182 range samples OUTSIDE 1984 m, where the plane-wave
replica is the correct one, so whatever causes it cannot be wavefront
curvature. (The scan does read 7.9 dB inside against 8.7 dB overall, which
is the direction curvature would push — but on 18 of 200 samples, too thin
to carry the argument either way.) The mode spread is what costs the gain,
at every range.

The look angle is re-chosen at every range here (the best of 361 beams),
which is why the third panel draws it: it hops between +/- theta rather than
migrating, because the modes arrive in up- and down-going pairs and the
interference decides which one wins. That re-steering is worth about 3.4 dB
over the best single fixed beam.

Keeping the largest of many beams is many chances to false-alarm, so the
threshold has to pay for it. The independent looks are not the 361 grid
points, and it is not even the orthogonal count. Two beams' NOISE outputs
correlate as the array factor of |w|^2, so the Hann taper widens the cell by
a factor 3 (hann^2 reaches its first null at three DFT bins against a
rectangular window's one) and the +/-45 deg sector holds about 5 independent
looks, not the 17 the unshaded geometry would give. Holding the whole scan
at Pf = 1e-4 then needs Pf = 1.8e-5 in each look: +0.45 dB of DT, and it is
what the maps use — a scanning detector judged against a scan's false-alarm
rate rather than one beam's. Ignoring the taper here would have cost 0.26 dB
of threshold, every decibel of it in the pessimistic direction.

One anchor for the whole figure, which is what makes the panels
comparable. The array is the fixed object, at r = 0, drawn as the receiver
it is; every point of both maps is a candidate TARGET position, and the
range axis is the separation between the two. Nothing is ever coloured "at"
the array. Panel 4 is then literally panel 1 pushed through the sonar
equation, and the two carry the same boundary — literally the same line,
because panel 1 contours the signal-excess field at 0 rather than standing
an iso-TL level in for it. That distinction is not pedantry: SE = 0 is
TL = SL - NL + AG - DT, and AG here is a grid spanning 6 dB, so the single
TL level that looks like the boundary sits at a median SE of -1.1 dB and
claims break-even where the target is a decibel short. Each panel also
carries the dashed line a constant AG would have promised, and the gap
between the pair is what the example is about. Both stop at the seabed: a
target cannot be inside the sediment, so the shoaling bottom takes a bite
out of the plane and the masked cells are excluded from every statistic
rather than counted as undetected.

Getting the 24-element field for a target at every point is one run, by
reciprocity — "if source and receiver are interchanged, the field remains
the same" (Jensen, Kuperman, Porter & Schmidt, *Computational Ocean
Acoustics*, whose source/listener interchange is Pierce's 4.9.3), and it
holds in a lossy medium too — "reciprocity does not depend on the system's
being nondissipative" (Pierce, *Acoustics*, 4.9.1). So the SOURCES go at
the array's own element depths, at r = 0 where
the array is, and the receivers cover the candidate-target plane.

Which end carries the sources is not cosmetic, and the shoaling bottom is
here to prove it. Putting the sources on the target depths instead —
receiver on the array — is the mirror-image geometry: it stands the array
on the far side of the sill instead of on the shelf where it sits.

Measuring that cost takes THREE fields, not two, because the obvious
difference confounds two effects. Call this example's construction (a),
the mirror (b), and (c) the true reciprocal of (a): the same physics read
from the other end, which means the source at the target depth on a section
reversed **about that range** — reversing the whole 20 km polyline instead
would put a different stretch of bottom under a path shorter than 20 km.
Then (b) vs (c) is the physics, and (a) vs (c) is the solver's own failure
to be reciprocal. Measured: the geometry costs 1.9 dB median and 8.8 dB at
worst, against a solver residual of 0.75 dB median, 1.17 dB worst — a
factor of about three, and both are printed.

The difference the example used to quote on its own, (a) vs (b) at 1.5 dB,
is those two partly cancelling. It understates the physics and hides the
solver residual inside itself, which is the kind of number that looks
conservative and is merely confounded.

On a flat bottom all three agree to 0.0 dB, which is the trap: a
range-independent example cannot tell any of these constructions apart, so
the wrong one passes every test until the day someone runs it over a slope.

That is also what gives the exchange check teeth. In a range-independent
*modal* solution "the symmetry between source and receiver is evident" in
the formula itself (JKPS again), so an exchange test there is structural
and can only confirm the plumbing. Once the profile varies in range the
mode solution is adiabatic or coupled rather than exact (Etter,
*Underwater Acoustic Modeling and Simulation*, 4.4.5) and reciprocity has
to be earned — uacpy's own KRAKEN notes put the coupled-mode residual at
"about 1 dB", which is where the 0.75 dB measured here sits.

Two model choices carry the whole chain, so neither is left as a default.
The first is coupled modes over the adiabatic approximation: adiabatic
assumes each mode's energy passes to the corresponding mode in the next
section, which needs the variation to be gradual (Etter 4.4.5), and a sill
that cuts the steeper modes off leaves no corresponding mode to pass it to.
Measured here over the water column, the two disagree by 2.0 dB in the
median and 7.7 dB at the 90th percentile — not a rounding difference.

The second is that one model agreeing with itself proves nothing, so the
section is run again through Bellhop, a ray/beam code that approximates
different things and fails in different places.

Comparing them fairly is its own small problem. "Both incoherent" is not
available — coupled modes cannot be added incoherently at all — so the
asymmetry is forced, and the tempting protocol (Kraken coherent, smoothed,
against Bellhop INCOHERENT) is the wrong way to handle it: its answer
tracks the smoothing kernel, +0.9 / +0.7 / +0.4 dB at 5 / 9 / 21 bins, so
there is no plateau to quote and half the model difference is cancelled by
the asymmetry. Both fields are therefore taken coherent and put through the
same boxcar, and the kernel is swept to measure what the protocol is worth:
**bias +1.5 dB, median |diff| 1.6 dB**, the bias moving 0.03 dB and the
median 0.17 dB across those kernels. That spread is the protocol's noise
floor, and a comparison without such a plateau should not be quoted at all.

Agreeing on TL is not the same as agreeing on the answer, so Bellhop is
carried all the way to a detection map of its own (a coherent run gives the
complex field on every element, which is what the beamformer needs). Two
things come out of that, and they point opposite ways.

The reassuring one: the realised array gain is 9.0 dB by the modal model
and 8.3 dB by the ray model — a 0.7 dB spread, against the 1.5 dB the two
models differ by on TL alone. Two methods with nothing in common agree that
a scanning plane-wave beamformer gives up about 3 dB against its plane-wave
value in this channel, so that shortfall is a property of the problem, not
of normal-mode theory.

The cautionary one: the two detection MAPS agree — mean |P_D difference|
0.05, and they disagree on detect / do-not-detect at 3.0 % of the water —
while the two detection RANGES do not. At 60 m they give 2.7 and 2.4 km,
but the median across target depth is 2.7 km against 4.0 km. A 1 dB model
difference barely moves the map and moves the outermost zero crossing by a
kilometre, because that crossing is a tail statistic: one constructive
fringe past the main boundary owns it. Quote the detectable AREA, not the
detection RANGE. `example_42_models.png` puts the two maps side by side.

Pushing that comparison up in frequency, where ray theory ought to win, is
worth knowing about and does NOT clean up: from 200 Hz to 3200 Hz the
median difference stays near 1.3-2.1 dB. The reason is on the modal side —
uacpy segments a range-dependent section on a quarter-wavelength criterion
and caps the decomposition at 200 profiles, so above a few hundred hertz it
warns that the field may not be converged, and Kraken's error grows as fast
as Bellhop's shrinks. The bathymetry here is sized to stay under that cap
at 200 Hz.

Nothing here is smoothed. A boxcar wide enough to flatten the interference
spans about 900 m on this grid: it buys a tidier contour and pays for it by
moving the headline detection range, which is the outermost crossing of a
fringing curve. The fringes are real, and sigma_dB in the P_D map is the
honest place to put how far a single prediction can be trusted.

Uses: Environment.plot(source=, receiver=, source_marker_range_m=) ·
Kraken(mode_coupling='coupled') · Bellhop · compare_models ·
steering_vectors ·
shading_taper · sonar.detection_threshold_energy · passive_signal_excess_field
· probability_of_detection_field · detection_range_by_depth ·
plot_field / plot_detection_probability with env/source/receiver overlays
· passive_signal_excess_field(array_gain=<grid>) — a per-sample AG
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.visualization import (compare_models, plot_detection_probability,
                                 plot_field)
from uacpy.acoustic_signal import (beamform_field, independent_beams,
                                   matched_replica_gain,
                                   plane_wave_array_gain, shading_taper)
from uacpy.core.results import Field
from uacpy.models import RunMode
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


def seabed_at(ranges_m):
    """Local water depth (m), for masking targets inside the sediment."""
    return np.interp(np.asarray(ranges_m, dtype=float),
                     BATHY_RANGES, BATHY_DEPTHS)

SL, NL = 120.0, 75.0          # band-integrated, dB re 1 uPa^2
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
in_water = plane_depths[:, None] <= seabed_at(array.ranges)[None, :]
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

# ── a second opinion: a different model on the same section ─────────────
# One model agreeing with itself proves nothing. Bellhop is a ray/beam code
# and Kraken a full-wave modal one; they approximate different things and
# fail in different places, so where they agree the propagation under this
# whole chain is worth trusting.
#
# Compare them SYMMETRICALLY. The obvious protocol — Kraken's coherent
# intensity smoothed in range against Bellhop's INCOHERENT TL — smooths one
# side and not the other, and that asymmetry is worth as much as the answer:
# it cancels about half the real model bias. Both fields are therefore taken
# coherent and put through the same boxcar, and the protocol is given a
# noise floor by running it on Bellhop against ITSELF.
bellhop = uacpy.Bellhop(verbose=False, n_beams=0, beam_type='G')
kern = np.ones(9) / 9


def range_smoothed_tl(pressure):
    """TL after a 9-point boxcar on intensity along range, water only."""
    inten = np.apply_along_axis(lambda r: np.convolve(r, kern, mode='same'),
                                1, np.abs(pressure) ** 2)
    return np.where(in_water, -10.0 * np.log10(np.maximum(inten, 1e-300)),
                    np.nan)


print(f"TL on {n_el} elements x {array.n_ranges} ranges, target at "
      f"{plane_depths[i60]:.0f} m: median "
      f"{np.nanmedian(-20 * np.log10(np.abs(p))):.1f} dB")

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


def beamform_plane(pressure, label):
    """Per-target TL and realised array gain, from one model's element fields.

    ``pressure`` is (n_elements, n_depths, n_ranges). Seabed cells come back
    NaN, which the sonar-equation grid carries through — the budget metadata
    summarises an AG grid with nan-aware statistics for exactly this case.
    """
    scan = beamform_field(pressure, elements, angles, FREQ, c=C_REF,
                          weights=taper)
    with np.errstate(divide='ignore', invalid='ignore'):
        gain = np.where(in_water, scan.array_gain(), np.nan)
        loss = np.where(in_water, -10.0 * np.log10(
            np.maximum(scan.element_power, 1e-300)), np.nan)
    field = Field(data=loss,
                  coords={'depth': plane_depths,
                          'range': np.asarray(array.ranges, dtype=float)},
                  model=label)          # kind defaults to 'pressure', whose
    # dB unit is spelled "TL (dB)" — which is exactly what this holds.
    return field, gain


tl_cov, ag_map = beamform_plane(p_map, 'Kraken')
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

# What the mirror-image construction would have cost. Running the target
# depth as the SOURCE and the array as receiver puts the array at the far,
# shallow end of the slope instead of the near, deep end where it sits. In
# a range-independent guide the two are the same number; here they are not,
# and the difference is the whole reason the sources go on the array.
# Three fields, not two, because the naive difference confounds two things.
#   (a) this example: sources at the element depths, forward section
#   (b) the mirror:   source at the target depth, forward section
#   (c) the true reciprocal of (a): source at the target depth on the
#       REVERSED section, which is the same physics as (a) read from the
#       other end.
# (a) vs (c) is therefore the SOLVER's own non-reciprocity, and (b) vs (c)
# is the physical cost of standing the array on the wrong side of the sill.
# What (a) vs (b) prints is the difference of the two, so quoting it alone
# understates the physics and hides the solver.
# The reversal has to be about EACH range, not once about the whole
# section: in a section reversed about 20 km, the stretch from 0 to R is the
# forward section's 20-R..20 km, which is different bottom. Reversing about
# R is what makes (c) the same physics as (a).
def element_mean_tl(pressure):
    """TL averaged over the array elements, in intensity."""
    return -10.0 * np.log10(np.mean(np.abs(pressure) ** 2, axis=0))


probe_idx = np.linspace(40, array.n_ranges - 1, 6).astype(int)
solver, physical, naive = [], [], []
for j in probe_idx:
    R = float(array.ranges[j])
    # Same node spacing as the forward section, so the comparison measures
    # reciprocity and not a difference in how finely each side is sampled.
    step = float(BATHY_RANGES[1] - BATHY_RANGES[0])
    walk = np.arange(0.0, R + 0.5 * step, step)
    reversed_env = uacpy.Environment(
        name='reversed-about-R',
        bathymetry=list(zip(walk, seabed_at(R - walk))),
        ssp=env.ssp, bottom=env.bottom, absorption=uacpy.Thorp())
    rx = uacpy.Receiver(depths=elements, ranges=[R])
    tl_a = element_mean_tl(p[:, j])                       # this example
    tl_b = element_mean_tl(np.asarray(kraken.run(env, target, rx).data))
    tl_c = element_mean_tl(np.asarray(
        kraken.run(reversed_env, target, rx).data))       # reciprocal of (a)
    solver.append(abs(tl_a - tl_c))
    physical.append(abs(tl_b - tl_c))
    naive.append(abs(tl_a - tl_b))
solver, physical, naive = map(np.asarray, (solver, physical, naive))
print(f"putting the array on the wrong side of the sill costs "
      f"{np.median(physical):.1f} dB median, {np.max(physical):.1f} dB worst")
print(f"  against the solver's own non-reciprocity of "
      f"{np.median(solver):.2f} dB median, {np.max(solver):.2f} dB worst — "
      f"a factor {np.median(physical) / np.median(solver):.0f}")
print(f"  (the naive a-vs-b difference, {np.median(naive):.1f} dB, is those "
      f"two partly cancelling:\n  it understates the geometry and buries "
      f"the solver residual inside itself)")
print(f"  on a flat bottom all three would agree to 0.0 dB — which is why a "
      f"range-independent\n  example cannot tell any of these constructions "
      f"apart")

# One definition of detection range for the whole example: read off the
# same field the map draws, so this number and that contour cannot drift
# apart. It is the OUTERMOST zero crossing, which on a fringing curve is a
# tail statistic — a single constructive fringe moves it by a kilometre, so
# it is a fair headline only alongside the map that shows the fringes.
r_const, r_real = r_by_z_const[i60], r_by_z_real[i60]
print(f"detection range at 60 m: {r_const / 1e3:.1f} km with a constant AG, "
      f"{r_real / 1e3:.1f} km with the realised one "
      f"({100 * (r_const - r_real) / r_const:+.0f} %)")

# ── the same chain, the other model ─────────────────────────────────────
# Agreeing on TL is not the same as agreeing on the answer, so the second
# model is carried all the way to a detection map rather than stopped at
# propagation. A COHERENT Bellhop run gives the complex field on every
# element, which is exactly what the beamformer needs.
bh_map = np.stack([np.asarray(f.data) for _, f in bellhop.run(
    env, array_as_sources, plane, run_mode=RunMode.COHERENT_TL)])

# Propagation first, and symmetrically. Comparing Kraken's COHERENT field
# against Bellhop's INCOHERENT TL is the obvious protocol and the wrong one:
# smoothing one side and not the other cancels part of the model difference,
# and it has no plateau to call a noise floor — its bias tracks the kernel
# (+0.9 / +0.7 / +0.4 dB at 5 / 9 / 21 bins). "Both incoherent" is not on
# offer either, because coupled modes cannot be added incoherently at all.
# So: both fields coherent, through the same boxcar, and the kernel swept to
# measure what the protocol itself is worth.
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
print("  the answer barely moves across those kernels, which is what lets "
      "it be quoted:\n  that spread — 0.03 dB of bias, 0.17 dB of median "
      "— IS the protocol's noise floor")
tl_bh, ag_bh = beamform_plane(bh_map, 'Bellhop')
se_bh = passive_signal_excess_field(tl_bh, source_level=SL, noise_level=NL,
                                    array_gain=ag_bh, detection_threshold=dt)
pd_bh = probability_of_detection_field(se_bh, sigma_dB=8.0)
_, r_by_z_bh = detection_range_by_depth(se_bh)

print(f"the array term itself, two ways: realised AG median "
      f"{np.nanmedian(ag_map):.1f} dB (Kraken) vs "
      f"{np.nanmedian(ag_bh):.1f} dB (Bellhop) — a ray code and a modal one "
      f"see the\n  same ~3 dB shortfall against the plane-wave value, so it "
      f"is not a modal artefact")

se_k = np.asarray(se_map['realised'].data)
se_b = np.asarray(se_bh.data)
verdict = 100.0 * np.mean((se_k[in_water] > 0.0) != (se_b[in_water] > 0.0))
pd_gap = np.nanmean(np.abs(np.asarray(pd_map['realised'].data) - np.asarray(
    pd_bh.data))[in_water])
print(f"the two detection MAPS agree: mean |Pd difference| {pd_gap:.02f}, "
      f"and they disagree on detect / do-not-detect at {verdict:.1f} % "
      f"of the water")
print(f"the two detection RANGES do not: at 60 m "
      f"{r_real / 1e3:.1f} vs {r_by_z_bh[i60] / 1e3:.1f} km, but the median "
      f"across depth is {np.nanmedian(r_by_z_real) / 1e3:.1f} vs "
      f"{np.nanmedian(r_by_z_bh) / 1e3:.1f} km")
print("  -> a 1 dB model difference barely moves the map and moves the "
      "outermost\n  crossing by a kilometre; the AREA is the statistic to "
      "quote, not the RANGE")

fig, _ = compare_models(
    [pd_map['realised'], pd_bh],
    labels=['Kraken — coupled modes', 'Bellhop — Gaussian beams'],
    env=env, receiver=array_at_origin, figsize=(13, 4.6), contours=(0.5,),
    title=f'The same detection chain under two models — maps agree to '
          f'{pd_gap:.02f} in P_D, headline ranges differ by '
          f'{abs(np.nanmedian(r_by_z_real) - np.nanmedian(r_by_z_bh)) / 1e3:.1f} km')
# compare_models sets its margins as fixed fractions, so the suptitle lands
# on the panel titles at this aspect; give it back the strip it needs.
fig.subplots_adjust(top=0.80)
fig.savefig(OUT / 'example_42_models.png', dpi=140, bbox_inches='tight')
plt.close(fig)

# ── what the beam actually hears ────────────────────────────────────────
# Everything above is a power budget. The same machinery run over a BAND
# instead of one line gives the reception itself: this array, steered this
# way, listening to that target radiate that pulse. Each bin is steered at
# its own frequency, because a beam delay is a phase that scales with it —
# one steering vector at the band centre mis-steers both edges.
BAND = np.linspace(FREQ - 50.0, FREQ + 50.0, 81)
R_SHOT = float(array.ranges[np.argmin(np.abs(np.asarray(array.ranges)
                                             - SCENE_R))])
#
# Bellhop, not Kraken, for this one — for speed, not capability. Kraken
# does run a range-dependent band (uacpy loops the multi-profile deck, one
# mode solve per bin), but that is ~0.3 s a bin against Bellhop's whole
# run, and the ray model is already this example's second opinion: the two
# agreed on the array term to 0.7 dB above.
H = bellhop.run(env, uacpy.Source(depths=float(plane_depths[i60]),
                                  frequencies=BAND),
                uacpy.Receiver(depths=elements, ranges=[R_SHOT]),
                run_mode=RunMode.BROADBAND)
beams = beamform_field(np.asarray(H.data)[:, 0, :], elements, angles, BAND,
                       c=C_REF, weights=taper)
# A Gaussian-shaded tone burst, given as its spectrum on the same bins.
pulse = np.exp(-((BAND - FREQ) / 25.0) ** 2)
# The window opens at the earliest arrival this MODEL can produce, which
# is r / (fastest WATER speed). The seabed's 1700 m/s must not enter it: a
# ray code has no head wave, so nothing travels through the sediment, and
# anchoring above the true fastest speed opens the window early — the span
# is 1/df whatever happens, so every millisecond of lead is a millisecond
# the late multipath wraps into. Counted on this geometry's own arrival
# set: anchoring at r/1520 puts every arrival inside the record (0 wrapped,
# 0.04 s of dead lead), while r/1700 — folding in the seabed — pushes 250
# of them past the end and wastes 0.38 s of a 0.80 s window. Bellhop stamps
# c0 = 1520 and deliberately no c_max for exactly this reason: a ray code
# has no head wave (field.py's _window_start, "no algorithmic speed may
# enter this max").
t_start = R_SHOT / float(np.max(np.asarray(env.ssp.data)))
# power is (n_angles, n_frequencies) here, so the band-averaged winner is
# the look to use — best_angle would give the winner per BIN, which is a
# different (and noisier) question.
look_best = float(angles[int(np.argmax(beams.power.mean(axis=1)))])
shots = {}
for label, look in (('steered to the arrival', look_best),
                    ('steered 30 deg off', look_best + 30.0)):
    tr = beams.to_time_trace(look, range_m=R_SHOT, source_spectrum=pulse,
                             t_start=t_start)
    shots[label] = (np.asarray(tr.coords['time']),
                    np.asarray(tr.data).ravel(), look)
# And one element, for the same pulse, as the thing the beam is beating.
one = Field(data=np.asarray(H.data)[n_el // 2, 0, :][None, None, :],
            coords={'depth': np.array([0.0]), 'range': np.array([R_SHOT]),
                    'frequency': BAND}, model='Bellhop')
trace_one = one.to_time_trace(depth=0.0, range=R_SHOT,
                              source_spectrum=pulse, t_start=t_start)
t_one = np.asarray(trace_one.coords['time'])
y_one = np.asarray(trace_one.data).ravel()
peak = lambda y: 20.0 * np.log10(np.max(np.abs(y)))
print(f"reception at {R_SHOT / 1e3:.1f} km of a {BAND[0]:.0f}-{BAND[-1]:.0f} "
      f"Hz pulse, peak level re one element:")
for label, (_, y, look) in shots.items():
    print(f"  {label:24s} ({look:+5.1f} deg): "
          f"{peak(y) - peak(y_one):+5.1f} dB")

fig, ax = plt.subplots(figsize=(11, 3.4))
ax.plot(t_one, np.abs(y_one) / np.abs(y_one).max(), color='0.6', lw=1.0,
        label='one element')
for (label, (t, y, look)), style in zip(shots.items(), ('C0-', 'C3--')):
    ax.plot(t, np.abs(y) / np.abs(y_one).max(), style, lw=1.2,
            label=f'{label} ({look:+.1f} deg)')
ax.set(xlabel='Time (s)', ylabel='|p| (re one element peak)',
       title=f'What the beam hears: a {BAND[0]:.0f}-{BAND[-1]:.0f} Hz pulse '
             f'from {plane_depths[i60]:.0f} m at {R_SHOT / 1e3:.1f} km')
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'example_42_reception.png', dpi=140, bbox_inches='tight')
plt.close(fig)

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
fan = row.power[:, i5]
ax.plot(angles, 10 * np.log10(fan / fan.max()), 'C0-', lw=1.3)
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
