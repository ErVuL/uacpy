"""
Example 23 — Per-feature collapse-method API
============================================

When an `Environment` carries a feature a model does not natively
support, `PropagationModel._project_environment` reduces it via the
matching key in the model's `collapse={…}` constructor parameter and
emits one `UserWarning` per dropped feature.

This example feeds **the same range-dependent environment** to
`Scooter` (range-independent wavenumber-integration TL solver — drops
both RD bathymetry and RD SSP). Four runs vary the `'bathymetry'` and
`'ssp'` keys of `collapse={…}`. The four TL fields are laid out
side-by-side via `compare_models` so the effect of each collapse choice
is visible.

A range-independent model is the right vehicle here: range-aware models
like RAM or KrakenField *honour* RD bathymetry and SSP natively, so
their collapse kwargs would be no-ops on this env.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import uacpy
from uacpy import SoundSpeedProfile
from uacpy.models import Scooter
from uacpy.visualization.plots import compare_models, plot_environment_advanced


OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)


def build_rd_environment() -> uacpy.Environment:
    """Slope from 80 m → 200 m with a strongly contrasted RD SSP.

    The SSP at the near range is a downward-refracting summer thermocline
    (warm surface, cool below). The far range is upward-refracting / nearly
    isothermal. This contrast makes the four 1-D collapses (r0, rmax, mean,
    median) produce visibly different profiles.
    """
    bathy_ranges_m = np.linspace(0.0, 20_000.0, 11)
    bathy_depths_m = np.linspace(80.0, 200.0, 11)
    bathymetry = np.column_stack([bathy_ranges_m, bathy_depths_m])

    ssp_depths = np.linspace(0.0, 200.0, 21)
    ssp_ranges_m = np.array([0.0, 10000.0, 20000.0])

    near = 1525.0 + (1480.0 - 1525.0) * np.tanh(ssp_depths / 30.0)
    far = 1480.0 + (ssp_depths / 200.0) * 35.0
    middle = 0.5 * (near + far)
    ssp_2d = np.column_stack([near, middle, far])

    return uacpy.Environment(
        name='Continental shelf — RD demo',
        ssp=SoundSpeedProfile.from_2d(depths=ssp_depths, ranges=ssp_ranges_m, matrix=ssp_2d,
                                      ),
        bathymetry=bathymetry,
        bottom=uacpy.BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1700.0, density=1.7, attenuation=0.5,
        ),
    )


def main() -> None:
    print("\n" + "═" * 80)
    print("EXAMPLE 23: Per-feature collapse-method API")
    print("═" * 80)

    env = build_rd_environment()
    source = uacpy.Source(depths=20.0, frequencies=200.0)
    receiver = uacpy.Receiver(
        depths=np.linspace(5.0, 195.0, 39),
        ranges=np.linspace(500.0, 15_000.0, 60),
    )

    print("[1/2] Environment overview…")
    fig_env, _ = plot_environment_advanced(env, source, receiver)
    out_env = OUTPUT_DIR / 'example_23_environment.png'
    fig_env.savefig(out_env, dpi=150, bbox_inches='tight')
    plt.close(fig_env)
    print(f"  ✓ Saved: {out_env}")

    combos = [
        ('max', 'r0'),
        ('max', 'rmax'),
        ('median', 'mean'),
        ('min', 'rmax'),
    ]

    print("[2/2] Running Scooter with four collapse policies…")
    results = {}
    for bathy_m, ssp_m in combos:
        sc = Scooter(
            collapse={'bathymetry': bathy_m, 'ssp': ssp_m},
            verbose=False,
        )
        label = f"bathy={bathy_m!r}, ssp={ssp_m!r}"
        try:
            results[label] = sc.compute_tl(env, source, receiver)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"  · {label} skipped: {exc.__class__.__name__}")

    if not results:
        print("  · no runs completed (Scooter binary missing?)")
        return

    fig_cmp, _ = compare_models(
        results, env, ncols=2, vmin=40, vmax=110, contours=[60, 80],
        suptitle='Same RD env collapsed four ways via collapse={…}',
    )
    out_cmp = OUTPUT_DIR / 'example_23_collapse_methods.png'
    fig_cmp.savefig(out_cmp, dpi=150, bbox_inches='tight')
    plt.close(fig_cmp)
    print(f"  ✓ Saved: {out_cmp}")

    print("\n✓ Example 23 complete\n")


if __name__ == '__main__':
    main()
