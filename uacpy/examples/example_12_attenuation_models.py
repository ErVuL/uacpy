"""Attenuation models — Thorp against Francois-Garrison.

Volume absorption from 10 Hz to 1 MHz, two formulas, what the environment does
to it, and the unit conversions in between.

Francois-Garrison has three terms, and which one dominates depends on the
frequency:

1. boric acid relaxation, below ~1 kHz — temperature, salinity and pH
   dependent (there is no low-frequency viscous mechanism; viscosity is term 3);
2. magnesium sulfate relaxation, ~10-500 kHz — relaxation at
   f₂ = 8.17·10^(8 − 1990/T_K) kHz, i.e. 76 kHz at 10 °C and rising steeply
   with temperature. Its A₂P₂ coefficient FALLS as temperature rises, so at
   10 kHz warming *reduces* attenuation — the sign people get wrong;
3. pure-water viscous absorption, above ~500 kHz — proportional to f².

Uses: Thorp.table · FrancoisGarrison.table ·
convert_attenuation_units · core.acoustics.sound_speed_mackenzie ·
AbsorptionCoefficient.plot (curve and overlay) · the α(z, f) values of
a depth-resolved AbsorptionCoefficient
"""

import os
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.acoustics import convert_attenuation_units
from uacpy.acoustics import sound_speed_mackenzie

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

# ── The two models over the full band, at standard conditions ───────────────
frequencies = np.logspace(1, 6, 500)                 # 10 Hz - 1 MHz
TEMPERATURE, SALINITY, PH, DEPTH = 10.0, 35.0, 8.0, 100.0
a_thorp = uacpy.Thorp().table(frequencies)
# The band runs below the 200 Hz Francois & Garrison fitted down to, on
# purpose: the comparison is over the whole band. The notice says how much of
# it is outside, and is printed here.
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter('always')
    a_francois = uacpy.FrancoisGarrison(
        temperature=TEMPERATURE, salinity=SALINITY,
        pH=PH).table(frequencies, depths=DEPTH)
for warning in caught:
    print(f"  noted: {str(warning.message).split(' — ')[0]}")
thorp, francois = a_thorp.data, a_francois.data

print(f"  at {TEMPERATURE:.0f} °C, S={SALINITY:.0f}, pH {PH}, {DEPTH:.0f} m:")
for probe in (100, 1000, 10000, 100000):
    index = int(np.argmin(np.abs(frequencies - probe)))
    print(f"    {probe:>7d} Hz  Thorp {thorp[index]:9.4f}  "
          f"F-G {francois[index]:9.4f}  Δ {francois[index] - thorp[index]:+8.4f}"
          f" dB/km")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# A carrier draws itself on log-log axes; a second .plot() with the same ax=
# overlays the other model.
a_thorp.plot(ax=axes[0], label='Thorp (1967)',
             title='Attenuation vs frequency (full range)',
             color='b', linewidth=2.5, alpha=0.8)
a_francois.plot(ax=axes[0], label='Francois-Garrison (1982)', color='r',
                linewidth=2.5, alpha=0.8)
axes[0].set_xlim([frequencies[0], frequencies[-1]])

low_band = frequencies[frequencies <= 10000]
uacpy.Thorp().table(low_band).plot(
    ax=axes[1], label='Thorp', title='Low frequency (10 Hz - 10 kHz)',
    color='b', linewidth=2.5, alpha=0.8)
with warnings.catch_warnings():          # the same sub-200 Hz notice
    warnings.simplefilter('ignore', uacpy.ValidityWarning)
    low_francois = uacpy.FrancoisGarrison(
        temperature=TEMPERATURE, salinity=SALINITY,
        pH=PH).table(low_band, depths=DEPTH)
low_francois.plot(ax=axes[1], label='Francois-Garrison', color='r',
                  linewidth=2.5, alpha=0.8)
axes[1].set_yscale('linear')

axes[2].semilogx(frequencies / 1000, francois - thorp, 'g-', linewidth=2.5)
axes[2].axhline(0, color='k', ls='--', lw=1, alpha=0.5)
axes[2].set_xlabel('Frequency (kHz)', fontweight='bold')
axes[2].set_ylabel('Difference (dB/km)', fontweight='bold')
axes[2].set_title('Francois-Garrison − Thorp', fontweight='bold', fontsize='x-large')
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim([frequencies[0] / 1000, frequencies[-1] / 1000])

fig.tight_layout()
fig.savefig(OUT / 'example_12a_model_comparison.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)

# ── One parameter at a time, at 10 kHz ──────────────────────────────────────
PROBE_HZ = 10000


def alpha_fg(depth=DEPTH, **varied):
    """Francois-Garrison at PROBE_HZ and ``depth`` in dB/km, varying one
    parameter."""
    water = dict(temperature=TEMPERATURE, salinity=SALINITY, pH=PH)
    water.update(varied)
    return float(np.ravel(uacpy.FrancoisGarrison(**water).table(
        PROBE_HZ, depths=depth).data)[0])


sweeps = [
    ('Temperature (°C)', np.linspace(0, 30, 31), TEMPERATURE, 'r',
     lambda v: alpha_fg(temperature=v)),
    ('Salinity (ppt)', np.linspace(0, 40, 41), SALINITY, 'b',
     lambda v: alpha_fg(salinity=v)),
    ('pH', np.linspace(7.5, 8.5, 21), PH, 'g',
     lambda v: alpha_fg(pH=v)),
    ('Depth (m)', np.linspace(0, 6000, 61), DEPTH, 'm',
     lambda v: alpha_fg(depth=v)),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
print(f"  sensitivity at {PROBE_HZ / 1000:.0f} kHz:")
for ax, (label, values, baseline, colour, evaluate) in zip(axes.flat, sweeps):
    # Each sweep crosses the fitted envelope on purpose (salinity down to
    # fresh water, pH past 7.7-8.3), and every value outside it warns; the
    # count is printed instead of one notice per value.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', uacpy.ValidityWarning)
        alpha = np.array([float(evaluate(v)) for v in values])
    if caught:
        print(f"    {label}: {len(caught)} of {values.size} values outside "
              f"the fitted envelope, evaluated as given")
    ax.plot(values, alpha, f'{colour}-', linewidth=2.5)
    ax.axvline(baseline, color='k', ls='--', alpha=0.5,
               label=f'baseline ({baseline:g})')
    ax.set_xlabel(label, fontweight='bold')
    ax.set_ylabel('Attenuation (dB/km)', fontweight='bold')
    ax.set_title(f'{label.split(" (")[0]} effect', fontweight='bold',
                 fontsize='large')
    ax.legend()
    ax.grid(True, alpha=0.3)
    print(f"    {label:<18} {alpha.min():.3f} to {alpha.max():.3f} dB/km "
          f"(spread {np.ptp(alpha):.3f})")
fig.tight_layout()
fig.savefig(OUT / 'example_12b_environmental_sensitivity.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)

# The sign is the thing people get wrong, so it is measured rather than
# asserted: at 10 kHz the MgSO4 term dominates and warming REDUCES attenuation.
warm = alpha_fg(temperature=20.0)
cool = alpha_fg(temperature=10.0)
print(f"  warming 10 → 20 °C at 10 kHz: {cool:.3f} → {warm:.3f} dB/km "
      f"({100 * (warm / cool - 1):+.0f}%, a decrease)")

# ── The same number in every unit ───────────────────────────────────────────
sound_speed = sound_speed_mackenzie()
per_km = float(np.ravel(uacpy.Thorp().table(PROBE_HZ).data)[0])
print(f"  {PROBE_HZ / 1000:.0f} kHz, c={sound_speed:.1f} m/s, "
      f"λ={sound_speed / PROBE_HZ:.4f} m:")
for unit, digits in (('dB/m', 7), ('dB/wavelength', 7), ('Nepers/m', 10)):
    value = convert_attenuation_units(per_km, PROBE_HZ, 'dB/km', unit,
                                      sound_speed)
    print(f"    {per_km:.4f} dB/km = {value:.{digits}f} {unit}")
back = convert_attenuation_units(
    convert_attenuation_units(per_km, PROBE_HZ, 'dB/km', 'dB/m', sound_speed),
    PROBE_HZ, 'dB/m', 'dB/km', sound_speed)
print(f"    round trip through dB/m: {back:.4f} dB/km, "
      f"matches {np.isclose(back, per_km)}")

ranges_km = np.linspace(0, 100, 101)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ranges_km, per_km * ranges_km, 'b-', linewidth=2.5)
ax.set_xlabel('Range (km)', fontweight='bold')
ax.set_ylabel('Total attenuation loss (dB)', fontweight='bold')
ax.set_title(f'Cumulative attenuation vs range ({PROBE_HZ / 1000:.0f} kHz)',
             fontweight='bold', fontsize='large')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / 'example_12c_unit_conversions.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)

# α over depth as well as frequency: pass a depth axis and the carrier holds
# α(z, f). Thorp has no depth term, so this is Francois-Garrison alone. T, S
# and pH are held constant with depth here, so pressure (through depth) is the
# only thing that varies. In F-G it enters as pressure-correction factors on
# the MgSO4 and pure-water terms, P2 = 1 − 1.37e-4 z + 6.2e-9 z² and
# P3 = 1 − 3.83e-5 z + 4.9e-10 z²; the boric-acid factor is P1 = 1 (F&G 1982
# Part II, Eq. (1) and Fig. 7; these values as AttenMod.f90:152-161 codes
# them). The boric and MgSO4 strengths also scale as
# 1/c, and F-G's own c = 1412 + 3.21 T + 1.19 S + 0.0167 z rises 67 m/s over
# 4 km, which takes about 4 % off the boric term too.
# On a colour map spanning ~8 decades of α, a factor of two is invisible, so
# the left panel maps the ratio α(z)/α(0) and the right one overlays α(f)
# at four depths.
fg_frequencies = np.logspace(np.log10(200.0), 6, 300)    # F-G's fitted band
fg_depths = np.linspace(0.0, 4000.0, 81)
over_depth = uacpy.FrancoisGarrison(
    temperature=TEMPERATURE, salinity=SALINITY,
    pH=PH).table(fg_frequencies, depths=fg_depths)
alpha_zf = np.asarray(over_depth.data)                 # (n_depths, n_freq)
ratio = alpha_zf / alpha_zf[0]
for probe in (1e3, 1e4, 1e5, 1e6):
    index = int(np.argmin(np.abs(fg_frequencies - probe)))
    print(f"    α(4 km)/α(0) at {probe / 1e3:6.0f} kHz: {ratio[-1, index]:.2f}")

fig, (left, right) = plt.subplots(1, 2, figsize=(15, 5.5))
mesh = left.pcolormesh(fg_frequencies / 1e3, fg_depths / 1e3, ratio,
                       cmap='viridis', vmin=0.5, vmax=1.0, shading='auto')
left.set_xscale('log')
left.invert_yaxis()
left.set_xlabel('Frequency (kHz)')
left.set_ylabel('Depth (km)')
left.set_title('α(z) / α(z = 0) — Francois-Garrison', fontweight='bold')
fig.colorbar(mesh, ax=left, label='α(z) / α(0)')
khz_band = fg_frequencies >= 1e3                         # 1 kHz - 1 MHz
for depth_km, colour in ((0, 'C0'), (1, 'C1'), (2, 'C2'), (4, 'C3')):
    row = int(np.argmin(np.abs(fg_depths - 1000.0 * depth_km)))
    right.loglog(fg_frequencies[khz_band] / 1e3, alpha_zf[row, khz_band],
                 color=colour, label=f'z = {depth_km} km')
right.set_xlabel('Frequency (kHz)')
right.set_ylabel('α (dB/km)')
right.set_title('α(f) at four depths', fontweight='bold')
right.grid(True, which='both', alpha=0.3)
right.legend()
fig.suptitle(f'Volume absorption over depth — {TEMPERATURE:.0f} °C, '
             f'S = {SALINITY:.0f}, pH {PH:g}, held constant with depth',
             fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'example_12d_absorption_over_depth.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)
