"""
Example 21: Bellhop vs RAM(ramsurf) on identical altimetry env
================================================================

Same env, same source, same receiver — different physics (ray tracing vs
parabolic equation). The point is to show that uacpy's surface convention
(``env.altimetry`` with positive heights up from sea level) gives both
models the same physical scenario, and to put numbers on how close the
two methods stay on a rough-surface Pekeris waveguide.

The comparison is intentionally pretty close to the regression test
``altimetry-consistency-bellhop-vs-ramsurf`` in
``tests/test_cross_model_agreement.py``, which guards the sign convention
on every test run.
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
from uacpy.models import Bellhop, RAM, RunMode  # noqa: E402


def main():
    print("\n" + "═" * 80)
    print("EXAMPLE 21: Bellhop vs RAM(ramsurf) on identical altimetry env")
    print("═" * 80)

    # Identical inputs to both models — Pekeris fluid waveguide with ice-keel
    # altimetry. ``env.altimetry`` follows uacpy's "positive up" convention;
    # the RAM dispatcher converts to ramsurf's "depth below z=0" internally.
    fluid_bottom = BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1700.0, density=1.7, attenuation=0.5,
    )
    surface = [
        (0.0, 0.0),
        (1500.0, -1.5),  # 1.5 m depression (ice keel)
        (3000.0, 0.0),
        (4500.0, -1.5),
        (6000.0, 0.0),
    ]
    env = Environment(
        name='altimetry-rough',
        bathymetry=100.0, ssp=1500.0,
        bottom=fluid_bottom, altimetry=surface,
    )
    src = Source(depths=50.0, frequencies=200.0)
    rcv = Receiver(
        depths=np.array([50.0]),
        ranges=np.linspace(500.0, 6000.0, 200),
    )

    print(f"Env: {env.depth:.0f} m Pekeris, isovelocity {float(env.ssp.data[0, 0]):.0f}, "
          f"fluid bottom (c=1700, ρ=1.7, α=0.5)")
    print(f"Surface: {len(surface)} altimetry breakpoints, two -1.5 m depressions")
    print(f"Source: depth={src.depths[0]:.0f} m, frequencies={src.frequencies[0]:.0f} Hz")
    print(f"Receiver: depth={rcv.depths[0]:.0f} m, range = "
          f"{rcv.ranges[0]:.0f}-{rcv.ranges[-1]:.0f} m ({len(rcv.ranges)} pts)")
    print()

    print("Running Bellhop ...")
    res_bh = Bellhop(verbose=False).run(
        env, src, rcv, run_mode=RunMode.COHERENT_TL,
    )
    tl_bh = res_bh.tl[0]

    print("Running RAM (dispatches to ramsurf1.5) ...")
    res_ram = RAM(verbose=False).run(env, src, rcv, run_mode=RunMode.COHERENT_TL)
    tl_ram = res_ram.tl[0]
    print(f"  → backend: {res_ram.backend}")
    print()

    from uacpy.core.metrics import tl_rmse, tl_max_error, tl_bias
    ranges = rcv.ranges
    rmin, rmax = 1000.0, 5000.0
    rmse = tl_rmse(res_ram, res_bh, range_window=(rmin, rmax))
    mxe = tl_max_error(res_ram, res_bh, range_window=(rmin, rmax))
    bias = tl_bias(res_ram, res_bh, range_window=(rmin, rmax))
    mask = (
        (ranges >= rmin) & (ranges <= rmax)
        & np.isfinite(tl_bh) & np.isfinite(tl_ram)
    )
    diff = tl_ram[mask] - tl_bh[mask]
    median_abs = float(np.median(np.abs(diff)))

    print(f"Agreement in {rmin/1000:.1f}-{rmax/1000:.1f} km window "
          f"({mask.sum()} samples):")
    print(f"  RMSE       = {rmse:.2f} dB")
    print(f"  median |Δ| = {median_abs:.2f} dB")
    print(f"  max  |Δ|   = {mxe:.2f} dB")
    print(f"  bias       = {bias:+.2f} dB (RAM minus Bellhop)")
    print()

    print("TL at representative ranges:")
    print("  range (km)  Bellhop   RAM(ramsurf)   Δ (dB)")
    print("  ----------  -------   ------------   ------")
    for r_target in [1.0, 2.0, 3.0, 4.0, 5.0]:
        i = int(np.argmin(np.abs(ranges - r_target * 1000)))
        print(f"  {r_target:>9.1f}    {tl_bh[i]:>5.1f}     {tl_ram[i]:>5.1f}      "
              f"{tl_ram[i] - tl_bh[i]:>+5.2f}")

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]},
    )
    ax_top.plot(ranges / 1000, tl_bh, 'k-', lw=1.5, label='Bellhop (rays + .ati)')
    ax_top.plot(ranges / 1000, tl_ram, 'r--', lw=1.2,
                label='RAM → ramsurf1.5 (PE + zsrf)')
    ax_top.invert_yaxis()
    ax_top.set_ylabel('TL (dB)')
    ax_top.set_title(
        f"Bellhop vs RAM(ramsurf) — identical Pekeris+altimetry env, "
        f"{src.frequencies[0]:.0f} Hz\n"
        f"RMSE = {rmse:.2f} dB, bias = {bias:+.2f} dB "
        f"(window {rmin/1000:.0f}-{rmax/1000:.0f} km)"
    )
    ax_top.grid(alpha=0.3)
    ax_top.legend()

    ax_bot.axhline(0, color='k', lw=0.5)
    ax_bot.plot(ranges / 1000, tl_ram - tl_bh, 'b-', lw=1.0,
                label='RAM − Bellhop')
    ax_bot.fill_between(ranges / 1000, -8, 8, color='green', alpha=0.15,
                        label='±8 dB regression band')
    ax_bot.set_xlabel('Range (km)')
    ax_bot.set_ylabel('Δ TL (dB)')
    ax_bot.set_ylim(-25, 25)
    ax_bot.grid(alpha=0.3)
    ax_bot.legend(loc='upper right')

    fig.tight_layout()
    out = Path(__file__).parent / 'output' / 'example_21_bellhop_vs_ramsurf.png'
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"\n  ✓ Saved: {out}")

    print("\n✓ Example 21 complete\n")


if __name__ == '__main__':
    main()
