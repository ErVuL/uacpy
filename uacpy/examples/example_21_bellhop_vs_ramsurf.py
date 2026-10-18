"""Bellhop against RAM(ramsurf) on one rough-surface environment.

Same environment, same source, same receiver — different physics: ray tracing
against a parabolic equation, over a Pekeris waveguide with ice-keel altimetry.
The point is that uacpy's surface convention (env.altimetry, positive up from
sea level) hands both models the same physical scenario, and that the two then
agree to a few dB.

A flipped altimetry sign would hand ramsurf a surface above sea level, which
it clamps back to flat; the agreement printed below is what rules that out.

Uses: env altimetry through two different solvers · Field.backend ·
metrics.tl_rmse / tl_max_error / tl_bias(range_window=) · Field.at() ·
plot.compare · metrics.tl_difference
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.metrics import tl_bias, tl_max_error, tl_rmse

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

# Two 1.5 m ice keels. uacpy keeps altimetry positive-up; the RAM dispatcher
# converts to ramsurf's "depth below z=0" internally.
env = uacpy.Environment(
    name='altimetry-rough', bathymetry=100.0, ssp=1500.0,
    bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                    sound_speed=1700.0, density=1.7,
                                    attenuation=0.5),
    altimetry=[(0.0, 0.0), (1500.0, -1.5), (3000.0, 0.0),
               (4500.0, -1.5), (6000.0, 0.0)],
)
source = uacpy.Source(depths=50.0, frequencies=200.0)
receiver = uacpy.Receiver(depths=np.array([50.0]),
                          ranges=np.linspace(500.0, 6000.0, 200))

bellhop = uacpy.Bellhop(backend='fortran').run(env, source, receiver)
ram = uacpy.RAM().run(env, source, receiver)
print(f"  RAM dispatched to {ram.backend}")

window = (1000.0, 5000.0)
print(f"  agreement over {window[0] / 1000:.0f}-{window[1] / 1000:.0f} km: "
      f"RMSE {tl_rmse(ram, bellhop, range_window=window):.2f} dB, "
      f"max |Δ| {tl_max_error(ram, bellhop, range_window=window):.2f} dB, "
      f"bias {tl_bias(ram, bellhop, range_window=window):+.2f} dB "
      f"(RAM − Bellhop)")
print("  range   Bellhop   RAM(ramsurf)      Δ")
for range_km in (1.0, 2.0, 3.0, 4.0, 5.0):
    tl_b = float(bellhop.at(depth=50.0, range=range_km * 1000).dB)
    tl_r = float(ram.at(depth=50.0, range=range_km * 1000).dB)
    print(f"  {range_km:>4.1f} km  {tl_b:>6.1f}   {tl_r:>10.1f}   "
          f"{tl_r - tl_b:>+6.2f} dB")

fig, (ax_tl, ax_diff) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                     gridspec_kw={'height_ratios': [2, 1.4]})
uacpy.plot.compare([bellhop.at(depth=50.0), ram.at(depth=50.0)],
                   ['Bellhop (rays + .ati)', 'RAM → ramsurf1.5 (PE + zsrf)'],
                   ax=ax_tl,
                   title=f'Identical Pekeris + altimetry environment at '
                         f'{source.frequencies[0]:.0f} Hz — RMSE '
                         f'{tl_rmse(ram, bellhop, range_window=window):.2f} dB')

# Ray and PE diverge past ~3 km as surface multipaths accumulate; the single
# large spikes are interference nulls the two place a little apart.
uacpy.metrics.tl_difference(ram, bellhop).plot(ax=ax_diff)
ax_diff.set_ylim(-25, 25)

fig.tight_layout()
fig.savefig(OUT / 'example_21_bellhop_vs_ramsurf.png', dpi=120)
plt.close(fig)
