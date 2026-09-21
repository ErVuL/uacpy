"""Beamforming a modelled field on a vertical receive array.

Example 29 resolves two synthetic plane waves. This one puts a real array in a
real waveguide and asks what it hears — which is not one plane wave but a
handful, because a trapped mode IS a plane-wave pair travelling at
+/- its grazing angle. Beamform the modelled field and the trapped modes come
out as discrete peaks, at the angles arccos(c/v_p,m) their phase speeds
predict. That is the receive half of what plot_mode_excitation shows on the
transmit side: the same waveguide, described by angle or by mode.

Three practical levers follow, and all three are array design rather than
acoustics: shading trades beamwidth for sidelobes, element spacing above
lambda/2 folds a grating lobe back into the fan, and aperture sets how finely
two arrivals can be separated.

Uses: Receiver as a vertical array · acoustic_signal.steering_vectors ·
shading_taper · beamform · Modes.compute_phase_speeds (the angles the peaks
should land on) · Kraken
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.acoustic_signal import beamform, beamform_field, shading_taper

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

FREQ, C_REF = 200.0, 1500.0
HALF = 0.5 * C_REF / FREQ                     # the Nyquist element spacing

env = uacpy.Environment(
    name='vla-duct', bathymetry=100.0, ssp=C_REF,
    bottom=uacpy.Bottom.from_halfspace(uacpy.BoundaryProperties(
        acoustic_type='half-space', sound_speed=1700.0, density=1.9,
        attenuation=0.5)),
)
source = uacpy.Source(depths=25.0, frequencies=FREQ, source_level_dB=180.0)
kraken = uacpy.Kraken(verbose=False)

# 24 elements at lambda/2, centred in the duct: an 86 m aperture in 100 m.
n_el = 24
elements = 50.0 + HALF * (np.arange(n_el) - (n_el - 1) / 2.0)
array = uacpy.Receiver(depths=elements, ranges=[5000.0])
field = kraken.run(env, source, array)
p = np.asarray(field.data)                     # (n_elements, 1)
print(f"array: {n_el} elements, {HALF:.2f} m apart, "
      f"{float(np.ptp(elements)):.1f} m aperture at {FREQ:g} Hz")

# The angles the waveguide can deliver: each trapped mode is a plane-wave pair
# at +/- arccos(c / v_p). Anything the beamformer finds should sit on one.
modes = kraken.compute_modes(env, source)
v_p = np.asarray(modes.compute_phase_speeds(), dtype=float)
mode_angles = np.degrees(np.arccos(np.clip(C_REF / v_p[v_p >= C_REF], -1, 1)))
print(f"{mode_angles.size} trapped modes span "
      f"{mode_angles.min():.1f}-{mode_angles.max():.1f} deg grazing")

angles = np.linspace(-60.0, 60.0, 1201)


fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

# ── 1. what the array hears, against what the waveguide can send ────────
ax = axes[0]
# beamform_field does |w^H p|^2 over the scan; normalise to dB re max.
pw = beamform_field(p[:, 0], elements, angles, FREQ, c=C_REF,
                    weights=shading_taper(n_el, 'boxcar')).power
beam = 10.0 * np.log10(pw / pw.max())
theta_max = float(mode_angles.max())
# The honest claim. This array's beamwidth (~2.5 deg over an 86 m aperture)
# is wider than the 2 deg between neighbouring modes, so it CANNOT resolve
# them one by one - a peak landing near some mode angle would prove nothing
# when 14 of them span 28 deg. What it does show is the fan: the waveguide
# traps nothing steeper than arccos(c/v_p) of the last mode, so outside
# +/-theta_max there is no arrival to find, and the response there is the
# array's own sidelobe floor.
ax.axvspan(-theta_max, theta_max, color='C3', alpha=0.10,
           label=f'trapped-mode fan (±{theta_max:.1f}°)')
for s_ in (+1, -1):
    ax.axvline(s_ * theta_max, color='C3', alpha=0.7, lw=1.1)
ax.plot(angles, beam, 'C0-', lw=1.3, label='conventional (unshaded)')
ax.set(xlabel='Angle from horizontal (deg)', ylabel='Beam power (dB re max)',
       ylim=(-35, 2), title='Arrivals live inside the trapped-mode fan')
ax.legend(fontsize=8, loc='lower right')
ax.grid(alpha=0.3)

# ── 2. shading: beamwidth against sidelobes ─────────────────────────────
ax = axes[1]
for name, style in (('boxcar', 'C0-'), ('hann', 'C1-'), ('hamming', 'C2--')):
    bp = beamform_field(p[:, 0], elements, angles, FREQ, c=C_REF,
                        weights=shading_taper(n_el, name)).power
    b = 10.0 * np.log10(bp / bp.max())
    # The highest sidelobe is the largest local maximum OUTSIDE the fan the
    # arrivals occupy - taking the maximum of the whole curve just returns
    # the main lobe at 0 dB.
    out = np.abs(angles) > theta_max + 2.0
    ax.plot(angles, b, style, lw=1.2,
            label=f"{name} (worst sidelobe {b[out].max():.0f} dB)")
ax.set(xlabel='Angle from horizontal (deg)', ylabel='Beam power (dB re max)',
       ylim=(-45, 2), title='Shading buys sidelobes with beamwidth')
ax.legend(fontsize=8, loc='lower right')
ax.grid(alpha=0.3)

# ── 3. element spacing: the grating lobe ────────────────────────────────
# Same aperture, half the elements -> spacing lambda, and a second copy of
# the response folds back into the fan. Nothing warns: the beamformer cannot
# know the spacing was chosen rather than sampled.
ax = axes[2]
coarse = elements[::2]
p_coarse = p[::2]
pc = beamform_field(p_coarse[:, 0], coarse, angles, FREQ, c=C_REF,
                    weights=shading_taper(len(coarse), 'boxcar')).power
ax.plot(angles, beam, 'C0-', lw=1.3, label=f'{HALF:.2f} m spacing (λ/2)')
ax.plot(angles, 10.0 * np.log10(pc / pc.max()), 'C3-', lw=1.3,
        alpha=0.85, label=f'{2 * HALF:.2f} m spacing (λ) — aliased')
ax.set(xlabel='Angle from horizontal (deg)', ylabel='Beam power (dB re max)',
       ylim=(-35, 2), title='Above λ/2 a grating lobe folds in')
ax.legend(fontsize=8, loc='lower right')
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(OUT / 'example_41_beam_response.png', dpi=140,
            bbox_inches='tight')
plt.close(fig)

# ── 4. the levelled answer: SNR per look angle ──────────────────────────
# beamform() takes the same (n_phones, n_ranges) a run on this Receiver
# returns, and folds in a source level and a per-element noise level.
res = beamform(p, elements, FREQ, angles=angles, SL=180.0, NL=60.0,
               c=C_REF)
snr = np.asarray(res.snr).ravel()               # (n_angles,) - one range here
look = np.asarray(res.angles).ravel()
print(f"peak beam at {look[int(np.nanargmax(snr))]:+.1f} deg, "
      f"SNR {np.nanmax(snr):.1f} dB (SL 180, NL 60 per element)")

# Every LOCAL maximum of the beam response, against the angles the waveguide
# can actually deliver. This is the claim the first panel makes, so it is
# checked here in numbers rather than left to the eye.
interior = np.flatnonzero((snr[1:-1] > snr[:-2]) & (snr[1:-1] >= snr[2:])) + 1
peaks = look[interior[np.argsort(snr[interior])[-10:]]]
allowed = np.concatenate([mode_angles, -mode_angles])
# Only the peaks inside the modal fan are modes. Outside it the waveguide
# sends nothing, so whatever the beamformer shows there is its own sidelobe
# structure — which is the distinction the first panel's red lines draw.
inside = peaks[np.abs(peaks) <= mode_angles.max() + 1.0]
miss = [float(np.min(np.abs(allowed - a))) for a in inside]
print(f"beam peaks inside the modal fan (|θ| <= {mode_angles.max():.1f} deg): "
      f"{np.round(np.sort(inside), 1).tolist()}")
print(f"  distance to the nearest mode angle: max {max(miss):.2f} deg, "
      f"median {float(np.median(miss)):.2f} deg")
print(f"  ({peaks.size - inside.size} further peaks lie outside the fan — "
      f"sidelobes, not arrivals)")
# The claim panel 1 actually makes, in numbers: how much of the beam power
# sits inside the fan the waveguide can fill.
lin = 10.0 ** (beam / 10.0)
inside_fan = np.abs(angles) <= mode_angles.max()
print(f"beam power inside the ±{mode_angles.max():.1f} deg fan: "
      f"{100.0 * lin[inside_fan].sum() / lin.sum():.1f} % "
      f"(the fan is {100.0 * inside_fan.mean():.0f} % of the scanned angles)")
print(f"\nFigures written to {OUT}")
