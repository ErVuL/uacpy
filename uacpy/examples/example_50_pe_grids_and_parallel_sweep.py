"""A frequency sweep in parallel, with the PE grid each frequency needs.

optimize_grid is the Padé-error search (Lytaev 2023) the RAM wrapper runs: the
coarsest (dr, dz) whose accumulated error stays under a budget over the
march. Run on its own it says how the grid scales with frequency before
anything is launched. RAM runs the same search, then applies what the error
model leaves out — a λ/15 cost floor that starts the dz search (refined
where the seabed's trapped modes need it) and the seafloor placed in its
cell — and grid_error scores the grid it marched. The four runs go to
run_parallel as one Job each.

The `if __name__ == '__main__':` guard is required: the workers start with
'spawn' and re-import this script, and an unguarded run_parallel call would
start again in every one of them.

Uses: models.optimize_grid · models.optimal_c0 · models.grid_error · RAM ·
Field.run_settings · run_parallel · Job · plot.compare
"""

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.models import grid_error, optimal_c0, optimize_grid

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

env = uacpy.Environment(
    name='Pekeris waveguide', bathymetry=100.0, ssp=1500.0,
    bottom=uacpy.BoundaryProperties(acoustic_type='half-space',
                                    sound_speed=1700.0, density=1.7,
                                    attenuation=0.5))
# From 250 m on, no direct or surface-reflected path is steeper than the
# 30° the PE propagates.
receiver = uacpy.Receiver(depths=np.linspace(2.0, 98.0, 25),
                          ranges=np.linspace(250.0, 5000.0, 191))
frequencies = [25.0, 50.0, 100.0, 200.0]

if __name__ == '__main__':
    # The medium as RAM scores it: c0 from Lytaev's Eq. (15), the accuracy
    # band the water's speeds widened to contain c0, the stability band
    # reaching the seabed's 1700 m/s (the Padé order is RAM's by default).
    c0 = optimal_c0(1500.0, 1500.0, 30.0, c_max_all=1700.0)
    medium = dict(c_min=1500.0, c_max=max(1500.0, c0), c_max_all=1700.0,
                  x_max=5000.0, angle_max_deg=30.0)
    plans = {f: optimize_grid(frequency=f, c0=c0, eps=1e-3, **medium)
             for f in frequencies}

    jobs = [uacpy.Job(uacpy.RAM(), env,
                      uacpy.Source(depths=25.0, frequencies=f), receiver,
                      label=f)
            for f in frequencies]
    batch = uacpy.run_parallel(jobs, n_workers=2)

    # The marched dz sits on RAM's cost floor, far coarser than the search's,
    # so the marched grid's predicted error is far above the 1e-3 budget:
    # the default accuracy gives way to the floor, while a pinned accuracy=
    # is met as given and dz= pins a finer grid.
    print(f"  c0 = {c0:.1f} m/s, error budget 1e-3 over 5 km at 30°")
    print("   f (Hz)   search dr / dz (m)    marched dr / dz (m)   "
          "error of the marched grid")
    marched = {}
    for f, field in zip(frequencies, batch):
        grid = field.run_settings.engine.grids[0]
        marched[f] = grid
        error = grid_error(dr=grid.dr, dz=grid.dz, frequency=f, c0=c0,
                           **medium)
        print(f"  {f:6.0f}   {plans[f].dr:7.2f} / {plans[f].dz:6.4f}"
              f"      {grid.dr:7.2f} / {grid.dz:6.4f}       {error:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    uacpy.plot.compare([field.at(depth=50.0) for field in batch],
                       labels=[f'{f:g} Hz' for f in frequencies], ax=axes[0],
                       title='RAM, one Job per frequency — TL at 50 m')
    for key, style in (('dr', '-'), ('dz', ':')):
        axes[1].plot(frequencies, [getattr(plans[f], key) for f in frequencies],
                     'o' + style, color='C0',
                     label=f'{key}, Padé-error search')
        axes[1].plot(frequencies,
                     [getattr(marched[f], key) for f in frequencies],
                     's' + style, color='C1', label=f'{key}, marched')
    axes[1].set_xscale('log', base=2)
    axes[1].set_yscale('log')
    axes[1].set_xticks(frequencies, [f'{f:g}' for f in frequencies])
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Step (m)')
    axes[1].set_title('The grid each frequency needs')
    axes[1].grid(True, which='both', alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / 'example_50_pe_grids_and_parallel_sweep.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {OUT / 'example_50_pe_grids_and_parallel_sweep.png'}")
