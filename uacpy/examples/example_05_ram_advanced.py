"""
===============================================================================
ADVANCED EXAMPLE: RAM (mpiramS) - Range-Dependent SSP and Bottom
===============================================================================

OBJECTIVE:
    Demonstrate RAM's range-dependent capabilities using the mpiramS
    Fortran PE backend:
    - Range-dependent bottom properties (sediment transition)
    - Sloping-shelf bathymetry
    - Parabolic equation accuracy control

SCENARIO:
    Sediment transition over a sloping shelf.

    Range progression (0 -> 3 km):
    - SSP: single warm stratified profile, range-independent
      (``env.has_range_dependent_ssp`` prints False below)
    - Bottom: Soft mud -> Hard sand
    - Depth: 100 m shelf -> 150 m slope

FEATURES DEMONSTRATED:
    - Range-dependent bottom properties
    - Continental shelf transition
    - COHERENT_TL mode (narrowband TL over range-depth grid)
    - Reading a TL field that is partly below the seafloor

NOTE ON COVERAGE:
    The receiver grid spans 5-145 m while the seafloor runs 100 m (r=0) to
    150 m (r=3 km), so the shallow-range columns place receivers inside the
    sub-bottom. Those cells come back NaN. Every summary below is therefore
    NaN-aware and reports how much of the grid is actually water.

===============================================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import uacpy  # noqa: E402
from uacpy.core.environment import SoundSpeedProfile  # noqa: E402
from uacpy import Bottom  # noqa: E402
from uacpy.models import RAM  # noqa: E402
from uacpy.visualization.plots import (  # noqa: E402
    plot_field,
)


def _plot_tl_difference(a, b, env=None, *, ax=None, title=None,
                         vmin=-10.0, vmax=10.0, diff_vmax=None, **kw):
    """Plot TL(a) - TL(b) as a diverging-colourmap heatmap.

    ``diff_vmax`` is a symmetric range shortcut: ``vmin = -diff_vmax``,
    ``vmax = +diff_vmax``.
    """
    from uacpy import Field
    from uacpy.visualization import plot_field
    if diff_vmax is not None:
        vmin, vmax = -abs(diff_vmax), abs(diff_vmax)
    diff = Field(data=a.tl - b.tl, coords=dict(a.coords))
    return plot_field(
        diff, env=env, ax=ax, vmin=vmin, vmax=vmax,
        cmap='RdBu_r', title=title, **kw,
    )


def _print_tl_summary(label, field):
    """NaN-aware TL summary that also states how much of the grid is water.

    Receivers below the local seafloor come back NaN; a bare ``.min()``/
    ``.max()`` on such a grid returns ``nan`` and hides the gap.
    """
    tl = np.asarray(field.tl)
    n_water = int(np.isfinite(tl).sum())
    print(f"  ✓ {label}: TL range {np.nanmin(tl):.1f} to {np.nanmax(tl):.1f} dB"
          f"  [{n_water}/{tl.size} cells in water,"
          f" {tl.size - n_water} below seafloor]")


def main():
    print("\n" + "═" * 80)
    print("EXAMPLE 05: RAM advanced features - Range-Dependent Bottom")
    print("═" * 80)

    # ═══════════════════════════════════════════════════════════════════════
    # CREATE 2D RANGE-DEPENDENT SSP (Thermal Front)
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[Setup] Creating warm stratified SSP profile...")

    depths = np.array([0., 25, 50, 75, 100, 120, 150])
    T = 8 + 12 * np.exp(-depths / 50)  # 20 °C surface, ~8 °C deep
    ssp_c = 1449 + 4.6*T - 0.055*T**2 + 0.00029*T**3 + 0.016*depths
    ssp_1d = np.column_stack([depths, ssp_c])
    print(f"  ✓ SSP range: {ssp_c.min():.1f} to {ssp_c.max():.1f} m/s")

    # ═══════════════════════════════════════════════════════════════════════
    # CREATE RANGE-DEPENDENT BOTTOM (Sediment Transition)
    # ═══════════════════════════════════════════════════════════════════════

    print("[Setup] Creating range-dependent bottom (mud → sand)...")

    # Bathymetry: shelf with mild slope over 3 km
    bathymetry = np.array([
        [0, 100],
        [1000, 110],
        [2000, 130],
        [3000, 150],
    ])

    # Bottom: soft mud → hard sand
    bottom_rd = Bottom.from_halfspaces(np.array([0, 1000, 2000, 3000]),
        sound_speed=np.array([1500, 1580, 1640, 1700]),
        density=np.array([1.2, 1.5, 1.8, 2.0]),
        attenuation=np.array([1.0, 0.7, 0.5, 0.3]),
        shear_speed=np.zeros(4),
        acoustic_type='half-space'
    )

    print(f"  ✓ Bottom sound speed: {bottom_rd.halfspace_sound_speed.min():.0f} → {bottom_rd.halfspace_sound_speed.max():.0f} m/s")
    print(f"  ✓ Bottom density: {bottom_rd.halfspace_density.min():.1f} → {bottom_rd.halfspace_density.max():.1f} g/cm³")

    # ═══════════════════════════════════════════════════════════════════════
    # CREATE ENVIRONMENT WITH ALL RANGE-DEPENDENT FEATURES
    # ═══════════════════════════════════════════════════════════════════════

    env = uacpy.Environment(
        name="Sediment Transition with Sloping Shelf",
        ssp=SoundSpeedProfile.from_pairs(ssp_1d),
        bathymetry=bathymetry,
        bottom=bottom_rd
    )

    print("\n✓ Environment created:")
    print(f"    - is_range_dependent: {env.is_range_dependent}")
    print(f"    - has_range_dependent_ssp: {env.has_range_dependent_ssp}")
    print(f"    - has_range_dependent_bottom: {env.has_range_dependent_bottom}")

    # ═══════════════════════════════════════════════════════════════════════
    # SOURCE & RECEIVER
    # ═══════════════════════════════════════════════════════════════════════

    source = uacpy.Source(depths=50.0, frequencies=100.0)
    receiver = uacpy.Receiver(
        depths=np.linspace(5, 145, 30),
        ranges=np.linspace(100, 3000, 30)
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RUN RAM with Range-Dependent Features
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[Run] RAM (mpiramS) with range-dependent SSP and bottom...")
    print("  Mode: COHERENT_TL (narrowband, range-depth TL grid)")

    try:
        ram = RAM(verbose=True, accuracy=1e-1)
        result = ram.run(env, source, receiver)
        print("  RAM TL completed successfully")
        _print_tl_summary("RAM", result)
    except Exception as e:
        print(f"  RAM error: {e}")
        import traceback
        traceback.print_exc()
        result = None

    # ═══════════════════════════════════════════════════════════════════════
    # COMPARISON: Kraken with Range-Independent Approximation
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[Comparison] Running Kraken for comparison...")
    print("  Note: Kraken will use range-independent approximation for the bottom")
    print("  → Range-dependent bottom effects will not be fully captured")
    print("  → For full accuracy, use RAM or Bellhop\n")

    from uacpy.models import Kraken, Bellhop

    # Run Kraken
    try:
        krakenfield = Kraken(verbose=False)
        result_krakenfield = krakenfield.compute_tl(env, source, receiver)
        print("  ✓ Kraken completed (using range-independent approximation)")
        _print_tl_summary("Kraken", result_krakenfield)
    except Exception as e:
        print(f"  ✗ Kraken: {e}")
        result_krakenfield = None

    # Run Bellhop (supports range-dependent natively)
    print("\n  Running Bellhop (native range-dependent support)...")
    try:
        bellhop = Bellhop(verbose=False)
        result_bellhop = bellhop.compute_tl(env, source, receiver)
        print("  ✓ Bellhop completed (full range-dependent capability)")
        _print_tl_summary("Bellhop", result_bellhop)
    except Exception as e:
        print(f"  ✗ Bellhop: {e}")
        result_bellhop = None

    # Comparisons (use nanmean because RAM masks sub-bottom cells as NaN)
    if result is not None and result_krakenfield is not None:
        diff_kf = np.abs(result.tl - result_krakenfield.tl)
        print(f"\n  RAM vs Kraken: Mean diff = {np.nanmean(diff_kf):.1f} dB (range-dependent effects)")

    if result is not None and result_bellhop is not None:
        diff_bh = np.abs(result.tl - result_bellhop.tl)
        print(f"  RAM vs Bellhop: Mean diff = {np.nanmean(diff_bh):.1f} dB (PE vs ray methods)")
        lam = 1500.0 / source.frequencies[0]
        d_lo, d_hi = env.bathymetry.depths.min(), env.bathymetry.depths.max()
        print(f"    Read that difference with care: at {source.frequencies[0]:.0f} Hz the"
              f" guide is only D/lambda = {d_lo / lam:.0f}-{d_hi / lam:.0f}")
        print("    wavelengths deep. DOCUMENTATION.md section 5 puts ray methods at"
              " D/lambda >~ 100, with")
        print("    modes/PE taking over below D/lambda ~ 30, so Bellhop is outside its"
              " regime here and")
        print("    RAM is the reference, not the other way round.")

    if result_bellhop is not None and result_krakenfield is not None:
        diff_bk = np.abs(result_bellhop.tl - result_krakenfield.tl)
        print(f"  Bellhop vs Kraken: Mean diff = {np.nanmean(diff_bk):.1f} dB")

    # ═══════════════════════════════════════════════════════════════════════
    # VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════════

    print("\nGenerating plots...")

    # Plot 1: Advanced environment overview
    fig1, axes1 = env.plot()
    plt.savefig(OUTPUT_DIR / 'example_05_environment.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: example_05_environment.png")

    # ``env.plot()`` already renders SSP and bottom in two panels;
    # the previous SSP-only and bottom-only plots are subsumed.

    # Plot 4: TL field result
    if result is not None:
        fig4, ax4 = plot_field(
            result, env=env, contours=[70, 85, 100],
            show_colorbar=True,
            vmin=40, vmax=100,
        )
        ax4.set_title('RAM: Sediment Transition with Sloping Shelf')

        plt.savefig(OUTPUT_DIR / 'example_05_result.png', dpi=150, bbox_inches='tight')
        print("  ✓ Saved: example_05_result.png")

    # Plot 5: Three-Model Comparison (RAM, Bellhop, Kraken)
    if result is not None and result_bellhop is not None and result_krakenfield is not None:
        from uacpy.visualization.plots import compare_models
        fig5, _ = compare_models(
            {'RAM': result, 'Bellhop': result_bellhop, 'Kraken': result_krakenfield},
            env=env, vmin=40, vmax=100,
            title='Three-Model Comparison — Sediment Transition + Sloping Shelf',
        )
        fig5.savefig(OUTPUT_DIR / 'example_05_comparison.png', dpi=150)
        plt.close(fig5)
        print("  ✓ Saved: example_05_comparison.png")

        fig6, axes6 = plt.subplots(1, 3, figsize=(20, 5))
        _plot_tl_difference(result, result_bellhop, env, ax=axes6[0],
                           title='RAM − Bellhop', show_colorbar=True)
        _plot_tl_difference(result, result_krakenfield, env, ax=axes6[1],
                           title='RAM − Kraken', show_colorbar=True)
        _plot_tl_difference(result_bellhop, result_krakenfield, env,
                           ax=axes6[2], title='Bellhop − Kraken',
                           show_colorbar=True)
        fig6.suptitle('Pairwise Differences (signed, dB)',
                      fontsize=13, fontweight='bold')
        fig6.tight_layout()
        fig6.savefig(OUTPUT_DIR / 'example_05_differences.png', dpi=150)
        plt.close(fig6)
        print("  ✓ Saved: example_05_differences.png")

    print("\nFeatures demonstrated:")
    print("  ✓ Range-dependent bottom properties (mud → sand)")
    print("  ✓ Sloping shelf bathymetry")
    print("  ✓ RAM parabolic equation with PE accuracy control")
    print("  ✓ Three-model comparison (RAM / Bellhop / Kraken)")

    print("\n✓ Example 05 complete\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
