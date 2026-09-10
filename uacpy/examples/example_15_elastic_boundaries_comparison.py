"""Elastic boundaries — the two workflows, side by side.

An elastic seabed (shear_speed > 0) can be handled two ways, and this runs both
on one environment:

1. Kraken alone. It detects the shear speed and switches to krakenc for complex
   modes — one call, nothing to manage.
2. BOUNCE first, writing reflection coefficients to a .brc file, then Scooter
   reading that file through acoustic_type='file'. More steps, but the .brc is
   reusable across runs and shareable.

BOUNCE writes both .brc and .irc: Bellhop, Scooter and krakenc read .brc, while
plain Kraken needs .irc (kraken.f90:47-48 aborts on a tabulated-'F' bottom), and
SPARC reads neither.

Uses: Kraken auto-detection · Bounce(work_dir=) · BoundaryProperties(
acoustic_type='file', reflection_file=) · ReflectionCoefficient.plot ·
plot_field · plot.compare · plot.plot_field_difference
"""

import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

N_REPEATS = 3


def median_time(call, repeats=N_REPEATS):
    """(result, median seconds) for ``call``, after a discarded warm-up.

    One cold call mostly measures process and binary load, which would land on
    whichever model happened to run first.
    """
    call()
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = call()
        times.append(time.perf_counter() - started)
    return result, float(np.median(times))


bottom = uacpy.BoundaryProperties(
    acoustic_type='half-space',
    sound_speed=1600.0,        # compressional, m/s
    shear_speed=400.0,         # shear > 0 is what makes the seabed ELASTIC
    density=1.8,
    attenuation=0.2,
    shear_attenuation=0.5,
)
env = uacpy.Environment(name="Elastic bottom test", bathymetry=100.0,
                        ssp=1500.0, bottom=bottom)
source = uacpy.Source(depths=50.0, frequencies=100.0)
receiver = uacpy.Receiver(depths=np.linspace(5, 95, 50),
                          ranges=np.linspace(100, 10000, 100))

# Approach 1: Kraken sees shear_speed > 0 and routes itself to krakenc.
kraken_tl, t_kraken = median_time(
    lambda: uacpy.Kraken().run(env, source, receiver))

# Approach 2: BOUNCE writes the reflection coefficients, Scooter reads them.
# A pinned work_dir keeps the .brc/.irc files after the run.
bounce, t_bounce = median_time(
    lambda: uacpy.Bounce(c_low=1400.0, c_high=10000.0, rmax=10000.0,
                         work_dir=OUT / 'bounce_brc').run(
        env=env, source=source,
        receiver=uacpy.Receiver(depths=np.array([50.0]),
                                ranges=np.array([1000.0]))))
env_from_file = uacpy.Environment(
    name="Scooter with BOUNCE reflection coefficients",
    bathymetry=100.0, ssp=1500.0,
    bottom=uacpy.BoundaryProperties(
        acoustic_type='file',
        reflection_file=bounce.metadata['brc_file'],
        sound_speed=1600.0, density=1.8, attenuation=0.2),
)
scooter_tl, t_scooter = median_time(
    lambda: uacpy.Scooter(c_low=bounce.metadata['c_low'],
                          c_high=bounce.metadata['c_high']).run(
        env_from_file, source, receiver))

residual = kraken_tl.dB - scooter_tl.dB
mean_diff = float(np.nanmean(np.abs(residual)))
p95_diff = float(np.nanpercentile(np.abs(residual), 95))
max_diff = float(np.nanmax(np.abs(residual)))
print(f"  |Kraken − Scooter|: mean {mean_diff:.2f} dB, "
      f"95th pct {p95_diff:.2f} dB, worst cell {max_diff:.2f} dB")
print("  Judge the two on the mean and the percentile: a large single-cell gap "
      "is where the\n  methods put an interference null a little differently, "
      "and near a null a small\n  shift in position is a big shift in dB.")
print(f"  {len(bounce.theta)} reflection angles, "
      f"|R| in [{bounce.R.min():.3f}, {bounce.R.max():.3f}]")
print(f"  timing: Kraken {t_kraken:.3f} s against BOUNCE {t_bounce:.3f} s + "
      f"Scooter {t_scooter:.3f} s (medians of {N_REPEATS})")
print("  That ratio is a property of THIS grid and frequency — a krakenc mode "
      "sum against\n  a Scooter FFP integration — and the .brc is reusable, "
      "which changes the sum\n  entirely once you run more than once.")

fig, axes = plt.subplots(2, 3, figsize=(18, 9))
uacpy.plot_field(kraken_tl, axes[0, 0], vmin=50, vmax=100,
                 title='1: Kraken (auto krakenc)')
uacpy.plot_field(scooter_tl, axes[0, 1], vmin=50, vmax=100,
                 title='2: Scooter (BOUNCE .brc)')
uacpy.plot.plot_field_difference(
    kraken_tl, scooter_tl, axes[0, 2],
    diff_vmax=max(5, max_diff),
    title=f'Kraken − Scooter (mean |Δ| {mean_diff:.2f} dB)')

# The BOUNCE result plots itself: |R| against grazing angle.
bounce.plot(ax=axes[1, 0])
# Mark the compressional critical angle, arccos(c_water / c_p): |R| is 1 below
# it and falls past it. The shear speed (400 m/s) is below the water speed, so
# there is no shear critical angle.
critical = np.degrees(np.arccos(env.ssp.data.min() / bottom.sound_speed))
axes[1, 0].axvline(critical, color='r', ls='--', lw=1.5, alpha=0.7)
axes[1, 0].text(critical + 2, 0.5, f'critical\n≈{critical:.1f}°', color='red',
                fontsize=9)

labels = ['Kraken (auto)', 'Scooter (BOUNCE)']
mid_range = float(np.median(kraken_tl.ranges))
uacpy.plot.compare([kraken_tl.at(depth=50.0), scooter_tl.at(depth=50.0)],
                   labels, ax=axes[1, 1], linewidth=2.5, alpha=0.8,
                   title='TL vs range at 50 m')
uacpy.plot.compare([kraken_tl.at(range=mid_range),
                    scooter_tl.at(range=mid_range)],
                   labels, ax=axes[1, 2], linewidth=2.5, alpha=0.8,
                   title=f'TL vs depth at {mid_range / 1000:.1f} km')

fig.suptitle('Elastic boundaries — Kraken auto-detection against '
             'BOUNCE → Scooter', fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT / 'example_15_elastic_boundaries_comparison.png', dpi=150,
            bbox_inches='tight')
plt.close(fig)
