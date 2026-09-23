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

Uses: absorption_thorp · absorption_francois_garrison ·
core.absorption.thorp_dB_per_km · francois_garrison_dB_per_km ·
convert_attenuation_units · core.acoustics.sound_speed_mackenzie ·
AbsorptionCoefficient.plot (curve, overlay, and the α(f, z) heatmap)
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.core.absorption import (convert_attenuation_units,
                                   francois_garrison_dB_per_km,
                                   thorp_dB_per_km)
from uacpy.core.acoustics import sound_speed_mackenzie

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

# ── The two models over the full band, at standard conditions ───────────────
frequencies = np.logspace(1, 6, 500)                 # 10 Hz - 1 MHz
TEMPERATURE, SALINITY, PH, DEPTH = 10.0, 35.0, 8.0, 100.0
a_thorp = uacpy.absorption_thorp(frequencies)
a_francois = uacpy.absorption_francois_garrison(
    frequencies, temperature_c=TEMPERATURE, salinity_psu=SALINITY, pH=PH,
    z_bar_m=DEPTH)
thorp, francois = a_thorp.values, a_francois.values

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
uacpy.absorption_thorp(low_band).plot(
    ax=axes[1], label='Thorp', title='Low frequency (10 Hz - 10 kHz)',
    color='b', linewidth=2.5, alpha=0.8)
uacpy.absorption_francois_garrison(
    low_band, temperature_c=TEMPERATURE, salinity_psu=SALINITY, pH=PH,
    z_bar_m=DEPTH).plot(ax=axes[1], label='Francois-Garrison', color='r',
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
sweeps = [
    ('Temperature (°C)', np.linspace(0, 30, 31), TEMPERATURE, 'r',
     lambda v: francois_garrison_dB_per_km(PROBE_HZ, v, SALINITY, PH, DEPTH)),
    ('Salinity (ppt)', np.linspace(0, 40, 41), SALINITY, 'b',
     lambda v: francois_garrison_dB_per_km(PROBE_HZ, TEMPERATURE, v, PH,
                                           DEPTH)),
    ('pH', np.linspace(7.5, 8.5, 21), PH, 'g',
     lambda v: francois_garrison_dB_per_km(PROBE_HZ, TEMPERATURE, SALINITY, v,
                                           DEPTH)),
    ('Depth (m)', np.linspace(0, 6000, 61), DEPTH, 'm',
     lambda v: francois_garrison_dB_per_km(PROBE_HZ, TEMPERATURE, SALINITY, PH,
                                           v)),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
print(f"  sensitivity at {PROBE_HZ / 1000:.0f} kHz:")
for ax, (label, values, baseline, colour, evaluate) in zip(axes.flat, sweeps):
    alpha = np.array([float(evaluate(v)) for v in values])
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
warm = float(francois_garrison_dB_per_km(1e4, 20.0, SALINITY, PH, DEPTH))
cool = float(francois_garrison_dB_per_km(1e4, 10.0, SALINITY, PH, DEPTH))
print(f"  warming 10 → 20 °C at 10 kHz: {cool:.3f} → {warm:.3f} dB/km "
      f"({100 * (warm / cool - 1):+.0f}%, a decrease)")

# ── The same number in every unit ───────────────────────────────────────────
sound_speed = sound_speed_mackenzie()
per_km = float(thorp_dB_per_km(PROBE_HZ))
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

# α over depth as well as frequency: pass a depth axis and the same carrier
# draws a heatmap instead. Thorp is depth independent, so this is F-G's
# pressure dependence on its own.
fig, ax = plt.subplots(figsize=(8, 5))
uacpy.absorption_francois_garrison(
    frequencies, temperature_c=TEMPERATURE, salinity_psu=SALINITY, pH=PH,
    z_bar_m=DEPTH, depths=np.linspace(0.0, 4000.0, 60)).plot(
        ax=ax, title='Volume absorption α(f, z) — Francois-Garrison')
fig.savefig(OUT / 'example_12d_absorption_over_depth.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)
