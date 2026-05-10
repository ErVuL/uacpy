"""
Example 22: RAM Padé-error grid optimizer (Lytaev 2023)
========================================================

Compares the two grid-selection modes of the ``RAM`` wrapper on a
Pekeris waveguide:

1. ``RAM(c0=1500.0)`` — c₀ pinned to the physical water speed. The
   Padé optimizer still picks (dr, dz) but the ξ-spectrum is asymmetric
   around 0, so the achievable grid is bounded by the wider tail of
   the Padé error.
2. ``RAM()`` (default) — c₀ resolved via Lytaev Eq. (15), the value
   that centres ``[ξ_min, ξ_max]`` and minimises the Padé error.
   Typically buys 2-3× coarser ``dr`` for the same accuracy.

Both runs use the same accuracy budget; the only difference is c₀.
The TL panels should agree to within a few dB.

Output: ``output/example_22_ram_lytaev_grid.png``.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from uacpy.core.environment import BoundaryProperties, Environment  # noqa: E402
from uacpy.core.receiver import Receiver  # noqa: E402
from uacpy.core.source import Source  # noqa: E402
from uacpy.models import RAM, RunMode  # noqa: E402


def main():
    print("\n" + "═" * 80)
    print("EXAMPLE 22: RAM Padé-error grid optimizer (Lytaev 2023)")
    print("═" * 80)

    fc = 100.0
    waveguide_depth = 100.0
    rmax = 10000.0

    src = Source(depths=25.0, frequencies=fc)
    rcv = Receiver(
        depths=np.linspace(2.0, waveguide_depth - 2.0, 30),
        ranges=np.linspace(200.0, rmax, 40),
    )

    env = Environment(
        name='pekeris-100hz',
        bathymetry=waveguide_depth,
        ssp=1500.0,
        bottom=BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1700.0, density=1.7, attenuation=0.5,
        ),
    )

    cases = [
        ('c1500',
         RAM(verbose=False, accuracy=1e-2, theta_max=20.0, c0=1500.0),
         "c₀=1500 (pinned)\nasymmetric ξ-range"),
        ('c_eq15',
         RAM(verbose=False, accuracy=1e-2, theta_max=20.0),
         "c₀=Eq.(15) (default)\ncentred ξ-range"),
    ]

    fields = {}
    print(f"\nRAM Lytaev grid on Pekeris {fc:.0f} Hz / {rmax/1000:.0f} km")
    print("  backend: mpiramS (fluid + flat surface)")
    print()
    for label, ram, _ in cases:
        field = ram.run(env, src, rcv, run_mode=RunMode.COHERENT_TL)
        fields[label] = field
        meta = field.metadata
        print(
            f"  {label:8s}  c₀={meta.get('c0'):6.1f} m/s  "
            f"dr={meta.get('dr'):7.2f} m  "
            f"dz={meta.get('dz'):6.3f} m"
        )

    diff = np.abs(fields['c1500'].tl - fields['c_eq15'].tl)
    rms = float(np.sqrt(np.nanmean(diff ** 2)))
    print(f"\n  RMS |TL_c1500 - TL_c_eq15| = {rms:.2f} dB")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), sharey=True)

    extent = [
        fields['c_eq15'].ranges[0] / 1000.0,
        fields['c_eq15'].ranges[-1] / 1000.0,
        fields['c_eq15'].depths[-1],
        fields['c_eq15'].depths[0],
    ]

    for ax, (label, _, title) in zip(axes[:2], cases):
        f = fields[label]
        meta = f.metadata
        im = ax.imshow(
            f.tl, aspect='auto', origin='upper', extent=extent,
            cmap='jet_r', vmin=40, vmax=100,
        )
        ax.set_title(
            f"{title}\n"
            f"c₀={meta.get('c0'):.1f} m/s, "
            f"dr={meta.get('dr'):.2f} m, dz={meta.get('dz'):.3f} m"
        )
        ax.set_xlabel('Range (km)')
        if ax is axes[0]:
            ax.set_ylabel('Depth (m)')
        fig.colorbar(im, ax=ax, label='TL (dB)')

    im = axes[2].imshow(
        diff, aspect='auto', origin='upper', extent=extent,
        cmap='magma', vmin=0, vmax=5,
    )
    axes[2].set_title(f'|TL_c1500 - TL_c_eq15|   RMS = {rms:.2f} dB')
    axes[2].set_xlabel('Range (km)')
    fig.colorbar(im, ax=axes[2], label='|ΔTL| (dB)')

    fig.suptitle(
        f'RAM Padé-error grid optimizer (Lytaev) — Pekeris {fc:.0f} Hz, '
        f'src 25 m, {rmax / 1000:.0f} km, mpiramS backend',
        y=1.02,
    )
    fig.tight_layout()

    out = Path(__file__).parent / 'output' / 'example_22_ram_lytaev_grid.png'
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches='tight')
    print(f"\n  ✓ Saved: {out}")

    print("\n✓ Example 22 complete\n")


if __name__ == '__main__':
    main()
