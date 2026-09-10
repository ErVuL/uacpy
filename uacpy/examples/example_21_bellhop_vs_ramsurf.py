"""Bellhop against RAM(ramsurf) on one rough-surface environment.

Same environment, same source, same receiver — different physics: ray tracing
against a parabolic equation, over a Pekeris waveguide with ice-keel altimetry.
The point is that uacpy's surface convention (env.altimetry, positive up from
sea level) hands both models the same physical scenario, and that the two then
agree to a few dB.

A flipped altimetry sign would push the RMSE past 25 dB, which is what the
regression test `altimetry-consistency-bellhop-vs-ramsurf` guards on every run.

Uses: env altimetry through two different solvers · Field.backend ·
core.metrics.tl_rmse / tl_max_error / tl_bias(range_window=) · Field.at() ·
plot.compare
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.core.metrics import tl_bias, tl_max_error, tl_rmse

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

bellhop = uacpy.Bellhop().run(env, source, receiver,
                              run_mode=uacpy.RunMode.COHERENT_TL)
ram = uacpy.RAM().run(env, source, receiver,
                      run_mode=uacpy.RunMode.COHERENT_TL)
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

fig, (ax_tl, ax_diff) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                     gridspec_kw={'height_ratios': [3, 1]})
uacpy.plot.compare([bellhop.at(depth=50.0), ram.at(depth=50.0)],
                   ['Bellhop (rays + .ati)', 'RAM → ramsurf1.5 (PE + zsrf)'],
                   ax=ax_tl,
                   title=f'Identical Pekeris + altimetry environment at '
                         f'{source.frequencies[0]:.0f} Hz — RMSE '
                         f'{tl_rmse(ram, bellhop, range_window=window):.2f} dB')

ax_diff.axhline(0, color='k', lw=0.5)
ax_diff.plot(receiver.ranges / 1000, ram.dB[0] - bellhop.dB[0], 'b-', lw=1.0,
             label='RAM − Bellhop')
# 8 dB is the empirical bar the cross-model test holds this pair to over 1-5 km
# (tolerance_dB=8.0): ray and PE diverge past ~3 km as surface multipaths
# accumulate, while a flipped altimetry sign would push the RMSE past 25 dB.
ax_diff.fill_between(receiver.ranges / 1000, -8, 8, color='green', alpha=0.15,
                     label='±8 dB regression band')
ax_diff.set_xlabel('Range (km)')
ax_diff.set_ylabel('Δ TL (dB)')
ax_diff.set_ylim(-25, 25)
ax_diff.grid(alpha=0.3)
ax_diff.legend(loc='upper right')

fig.tight_layout()
fig.savefig(OUT / 'example_21_bellhop_vs_ramsurf.png', dpi=120)
plt.close(fig)
