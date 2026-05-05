"""
Example 29: RAM multi-backend dispatch — mpiramS, RAMS (elastic), RAMSurf (rough)
=================================================================================

The unified ``RAM`` class auto-selects one of three vendored Collins-family PE
binaries based on the environment:

- **mpiramS** (default) — fluid bottom + flat surface; native broadband Q/T loop.
- **rams0.5** — *elastic* bottom (any ``shear_speed > 0`` anywhere); single
  frequency, fluid-vs-elastic coupling via Lamé parameters in the sediment.
- **ramsurf1.5** — fluid bottom + variable surface (``env.altimetry``); single
  frequency, rough-surface / beach-style propagation.

Combining elastic + rough surface raises ``NotImplementedError`` — there is no
published Collins PE for that combination. Use OASES for range-independent
elastic problems, or fluidise / flatten one side as an approximation.

This example runs the same Pekeris-like waveguide through all three backends
and overlays the TL maps so you can see the dispatch in action.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from uacpy.core.environment import (
    BoundaryProperties, Environment, LayeredBottom, SedimentLayer,
)
from uacpy.core.receiver import Receiver
from uacpy.core.source import Source
from uacpy.models import RAM, RunMode


def main():
    fc = 100.0  # Hz
    waveguide_depth = 100.0
    rmax = 5000.0

    src = Source(depth=50.0, frequency=fc)
    rcv = Receiver(
        depths=np.linspace(2.0, waveguide_depth - 2.0, 30),
        ranges=np.linspace(200.0, rmax, 40),
    )

    fluid_halfspace = BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1700.0, density=1.7, attenuation=0.5,
    )

    elastic_layered = LayeredBottom(
        layers=[
            SedimentLayer(
                thickness=15.0, sound_speed=1700.0, density=1.7,
                attenuation=0.5, shear_speed=400.0, shear_attenuation=1.5,
            ),
        ],
        halfspace=BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1900.0, density=2.0, attenuation=0.2,
            shear_speed=600.0, shear_attenuation=0.5,
        ),
    )

    surface = [(r, 1.5 * np.sin(2 * np.pi * r / 2000.0))
               for r in np.linspace(0.0, rmax, 50)]

    cases = [
        ('mpiramS (fluid + flat)', Environment(
            name='fluid-flat', depth=waveguide_depth, sound_speed=1500.0, bottom=fluid_halfspace,
        )),
        ('rams0.5 (elastic + flat)', Environment(
            name='elastic-flat', depth=waveguide_depth, sound_speed=1500.0, bottom=elastic_layered,
        )),
        ('ramsurf1.5 (fluid + rough)', Environment(
            name='fluid-rough', depth=waveguide_depth, sound_speed=1500.0, bottom=fluid_halfspace, altimetry=surface,
        )),
    ]

    ram = RAM(verbose=False, accuracy=1e-1)

    print("Dispatch:")
    for label, env in cases:
        backend = ram.select_backend(env)
        print(f"  {label:34s} → {backend}")

    # The elastic + altimetry combination is the documented gap.
    print("\nGap demonstration:")
    bad_env = Environment(
        name='elastic-rough', depth=waveguide_depth, sound_speed=1500.0, bottom=elastic_layered, altimetry=surface,
    )
    try:
        ram.select_backend(bad_env)
    except NotImplementedError as e:
        print(f"  elastic + altimetry → NotImplementedError: {str(e)[:80]}…")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (label, env) in zip(axes, cases):
        try:
            field = ram.run(env, src, rcv, run_mode=RunMode.COHERENT_TL)
            im = ax.pcolormesh(
                field.ranges / 1000.0, field.depths, field.data,
                shading='auto', cmap='jet_r', vmin=30, vmax=110,
            )
            ax.invert_yaxis()
            ax.set_title(f"{label}\nbackend={field.metadata['backend']}")
            ax.set_xlabel("Range (km)")
            if ax is axes[0]:
                ax.set_ylabel("Depth (m)")
            fig.colorbar(im, ax=ax, label='TL (dB)')
        except FileNotFoundError as exc:
            ax.set_title(f"{label}\n(skipped: {exc})")

    fig.tight_layout()
    out = Path(__file__).parent / 'output' / 'example_20_ram_backends.png'
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\nSaved: {out}")


if __name__ == '__main__':
    main()
