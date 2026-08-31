"""
═══════════════════════════════════════════════════════════════════════════════
ADVANCED EXAMPLE: Bellhop - All Features Showcase
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    Demonstrate ALL Bellhop features including new 2D options:
    - Advanced RunType control (grid_type, beam_shift) + Source geometry
    - Source directivity from a .sbp beam pattern
    - Cerveny Gaussian beam parameters
    - Volume attenuation (Thorp formula)
    - Grain size boundary conditions
    - Range-dependent environment

ENVIRONMENT:
    - Continental shelf (100m → 500m over 30km)
    - The top 500 m of the Munk profile. Munk's c_min axis is at 1300 m, below
      this domain, so over 0-500 m the profile is monotonically decreasing
      (1548.5 → 1513.2 m/s): downward refraction, no sound channel.
    - Grain size bottom transitioning to hard bottom

FEATURES DEMONSTRATED:
    ✓ Full 7-position RunType string
    ✓ Cerveny beam parameters (eps_multiplier, beam_width_type, etc.)
    ✓ Thorp volume attenuation
    ✓ Line source (Cartesian coordinates)
    ✓ Source directivity (.sbp beam pattern) + its polar plot
    ✓ Rectilinear receiver grid (RunType position 5 = 'R')
    ✓ Beam shift on reflection
    ✓ Grain size boundary conditions
    ✓ Multiple run comparisons

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import os
from pathlib import Path
# Repo root, so ``import uacpy`` resolves from a source checkout.
sys.path.insert(0, str(Path(__file__).parents[2]))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import QuadMesh  # noqa: E402
import uacpy  # noqa: E402
from uacpy.core.environment import SoundSpeedProfile  # noqa: E402
from uacpy import Bottom  # noqa: E402
from uacpy.models import Bellhop  # noqa: E402
from uacpy.visualization.plots import plot_field  # noqa: E402
from uacpy.models import RunMode  # noqa: E402

OUTPUT_DIR = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
                  or Path(__file__).parent / 'output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _tl_mappable(ax):
    """Return the TL QuadMesh ``plot_field`` drew on ``ax``.

    Feeding this to ``fig.colorbar`` ties a shared colorbar to the panels'
    own colour scale (``_TL_LIMITS``, 20-120 dB) instead of restating it.
    Contour overlays also live in ``ax.collections``, hence the type filter.
    """
    return next(c for c in ax.collections if isinstance(c, QuadMesh))


def main():
    print("\n" + "═" * 80)
    print("EXAMPLE 04: Bellhop advanced features")
    print("═" * 80)

    # ═══════════════════════════════════════════════════════════════════════
    # ENVIRONMENT: Continental Shelf with Grain Size Bottom
    # ═══════════════════════════════════════════════════════════════════════

    # Bathymetry: shallow shelf to deep ocean
    bathymetry = np.array([
        [0, 100],      # 0 km: 100m depth (shelf)
        [10000, 150],  # 10 km: 150m
        [20000, 300],  # 20 km: 300m (shelf break)
        [30000, 500],  # 30 km: 500m (slope)
    ])

    # Range-dependent bottom: sand on shelf, hardpack on slope
    ranges = np.array([0.0, 10000.0, 20000.0, 30000.0])
    bottom_rd = Bottom.from_halfspaces(ranges,
        sound_speed=np.array([1600, 1650, 1700, 1750]),  # Hardening
        density=np.array([1.5, 1.7, 1.9, 2.1]),         # Increasing
        attenuation=np.array([0.8, 0.6, 0.4, 0.3]),     # Less lossy
        shear_speed=np.zeros(4),
        acoustic_type='half-space'
    )

    # Top 500 m of the Munk profile: monotonically decreasing over this
    # domain, so the water column refracts downward throughout.
    env = uacpy.Environment(
        name="Continental Shelf - Munk Profile (upper 500 m)",
        ssp=SoundSpeedProfile.from_munk(500.0),
        bathymetry=bathymetry,
        bottom=bottom_rd,
        absorption=uacpy.Thorp(),
    )

    # ═══════════════════════════════════════════════════════════════════════
    # SOURCE & RECEIVER CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════

    source = uacpy.Source(
        depths=75.0,      # Upper water column
        frequencies=100.0  # 100 Hz
    )

    receiver = uacpy.Receiver(
        depths=np.linspace(10, 450, 50),    # Dense vertical sampling
        ranges=np.linspace(100, 30000, 150)  # 0.1 to 30 km
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RUN 1: Standard Gaussian Beams with Thorp Attenuation
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[1/6] Running Bellhop with Thorp volume attenuation...")
    bellhop_thorp = Bellhop(
        verbose=False,
        beam_type='B', grid_type='R',
        n_beams=500, alpha=(-85, 85),
    )

    try:
        result_thorp = bellhop_thorp.run(
            env, source, receiver, run_mode=RunMode.COHERENT_TL,
        )
        print("  ✓ Success")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        result_thorp = None

    # ═══════════════════════════════════════════════════════════════════════
    # RUN 2: Cerveny Beams with Advanced Control
    # ═══════════════════════════════════════════════════════════════════════

    print("[2/6] Running Bellhop with Cerveny beams...")

    bellhop_cerveny = Bellhop(
        verbose=False,
        beam_type='C', grid_type='R',
        n_beams=500, alpha=(-85, 85),
        beam_width_type='M', beam_curvature='Z',
        eps_multiplier=0.7, r_loop=10000.0, n_image=2, ib_win=4,
        beam_shift=True,
    )

    try:
        result_cerveny = bellhop_cerveny.run(
            env, source, receiver, run_mode=RunMode.COHERENT_TL,
        )
        print("  ✓ Success")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        result_cerveny = None

    # ═══════════════════════════════════════════════════════════════════════
    # RUN 3: Line Source (Cartesian Coordinates)
    # ═══════════════════════════════════════════════════════════════════════

    print("[3/6] Running Bellhop with line source...")

    bellhop_line = Bellhop(
        verbose=False,
        beam_type='B', grid_type='R',
        n_beams=500,
    )

    # Geometry lives on the Source, not the model: same Bellhop, different
    # source. 'line' is an infinite coherent line source (Cartesian
    # spreading) rather than a point source (cylindrical).
    source_line = uacpy.Source(
        depths=source.depths, frequencies=source.frequencies,
        source_type='line',
    )

    try:
        result_line = bellhop_line.run(
            env, source_line, receiver, run_mode=RunMode.COHERENT_TL,
        )
        print("  ✓ Success")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        result_line = None

    # ═══════════════════════════════════════════════════════════════════════
    # RUN 4: Multi-source-depth (Bellhop binary loops source axis natively)
    # ═══════════════════════════════════════════════════════════════════════

    print("[4/6] Running Bellhop with three source depths in one binary call...")
    # The shelf depth at r=0 is 100 m, so every source must sit in
    # the water column at the launch point (z < 100 m).
    source_multi = uacpy.Source(depths=[20.0, 50.0, 80.0],
                                frequencies=300.0)
    bellhop_multi = Bellhop(verbose=False, n_beams=500, alpha=(-85, 85))
    try:
        stack = bellhop_multi.run(
            env, source_multi, receiver, run_mode=RunMode.COHERENT_TL,
        )
        # ``stack`` is a ResultStack of Field slabs. Iterate to walk
        # (source_depth, slab) pairs, or stack.at(source_depth=z) to
        # pick a single 2-D Field by label. Slab accessors (.db, .p,
        # .at(depth=, range=)) live on the Field, not on the stack.
        print(f"  ✓ Success — ResultStack of {stack.slab_type.__name__} "
              f"with {stack.n_slabs} source-depth slabs")
        for sd_value, slab in stack:
            tl = np.asarray(slab.db)
            real = tl[np.isfinite(tl)]        # NaN = no-data (no ray reached)
            if real.size:
                print(f"      sd={sd_value:6.1f} m  →  median TL "
                      f"{np.median(real):.1f} dB")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        stack = None

    # ═══════════════════════════════════════════════════════════════════════
    # RUN 5: Ray Trace with Beam Shift
    # ═══════════════════════════════════════════════════════════════════════

    print("[5/6] Running ray trace with beam shift...")

    bellhop_rays = Bellhop(
        verbose=False,
        beam_type='g', grid_type='R',
        n_beams=50, alpha=(-80, 80),
        beam_shift=True,
    )

    try:
        result_rays = bellhop_rays.run(
            env, source, receiver, run_mode=RunMode.RAYS,
        )
        print("  ✓ Success")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        result_rays = None

    # ═══════════════════════════════════════════════════════════════════════
    # RUN 6: Directional Source (.sbp Beam Pattern)
    # ═══════════════════════════════════════════════════════════════════════

    print("[6/6] Running Bellhop with a directional source...")

    # A beam pattern is an (angle_deg, level_dB re peak) table. The angle axis
    # is Bellhop's launch declination alpha, so POSITIVE IS DOWNWARD
    # (ray2D(1)%t = [COS(alpha), SIN(alpha)]/c, bellhop.f90:453, over a depth
    # axis that increases downward): this is a beam tilted below the
    # horizontal, aimed down the shelf.
    # A projector is specified by its beamwidth, and beam_pattern is just an
    # angular weighting on this point source's launch amplitude — so the table
    # is a main lobe BEAMWIDTH_DEG wide between its -3 dB points, aimed at
    # TILT_DEG, with the nulls and sidelobes a smooth lobe implies rather than
    # the rectangular on/off a hand-drawn table suggests. (No aperture or
    # element spacing enters: that is receiving-array geometry, not this.)
    # sinc is -3 dB at 0.442946, so that constant sets the width; the nulls
    # are true zeros, so the levels are floored to keep the table finite.
    TILT_DEG, BEAMWIDTH_DEG, FLOOR_DB = 20.0, 24.0, -40.0
    pattern_angles = np.linspace(-90.0, 90.0, 721)
    lobe = 0.442946 * (pattern_angles - TILT_DEG) / (0.5 * BEAMWIDTH_DEG)
    pattern_levels = 20.0 * np.log10(
        np.maximum(np.abs(np.sinc(lobe)), 10.0 ** (FLOOR_DB / 20.0)))

    # The table has to cover every launch angle alpha spans. Bellhop clamps the
    # table index but not the interpolation weight (bellhop.f90:269-274), so a
    # table that stops short is EXTRAPOLATED on linear amplitude — the outer
    # beams come back louder than declared and phase-inverted — and uacpy
    # refuses the run rather than let that reach the field.
    source_directional = uacpy.Source(
        depths=source.depths, frequencies=source.frequencies,
        beam_pattern=np.column_stack([pattern_angles, pattern_levels]),
    )

    bellhop_directional = Bellhop(
        verbose=False,
        beam_type='B', grid_type='R',
        n_beams=500, alpha=(-85, 85),
    )

    try:
        result_directional = bellhop_directional.run(
            env, source_directional, receiver, run_mode=RunMode.COHERENT_TL,
        )
        print("  ✓ Success")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        result_directional = None

    # ═══════════════════════════════════════════════════════════════════════
    # PLOTTING
    # ═══════════════════════════════════════════════════════════════════════

    print("\nGenerating plots...")

    # Plot 1: Environment setup with range-dependent bottom
    fig1, axes1 = env.plot()
    fig1.savefig(OUTPUT_DIR / 'example_04_environment.png', dpi=150,
                 bbox_inches='tight')
    plt.close(fig1)
    print("  ✓ Saved: example_04_environment.png")

    # Plot 2: Compare standard vs Cerveny beams
    # Using show_colorbar=False for subplots with shared colorbar
    if result_thorp is not None and result_cerveny is not None:
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Disable individual colorbars, add contours at 70, 85, 100 dB
        _, _ = plot_field(result_thorp, env=env, ax=ax1,
                                      show_colorbar=False,
                                      contours=[70, 85, 100])
        ax1.set_title('Standard Gaussian Beams\n(with Thorp attenuation)')

        _, _ = plot_field(result_cerveny, env=env, ax=ax2,
                                      show_colorbar=False,
                                      contours=[70, 85, 100])
        ax2.set_title('Cerveny Beams (Minimum Width)\n(with beam shift)')

        # Single shared colorbar, taken from a panel's own mappable so its
        # scale cannot disagree with the images it labels.
        cbar_ax = fig2.add_axes([0.92, 0.15, 0.02, 0.7])
        cb = fig2.colorbar(_tl_mappable(ax1), cax=cbar_ax, orientation='vertical')
        cb.set_label('TL (dB)', fontsize=12, fontweight='bold')

        fig2.suptitle('Bellhop: Gaussian vs Cerveny Beams (contour overlays '
                      '+ shared colorbar)', fontsize=16, fontweight='bold')
        fig2.savefig(OUTPUT_DIR / 'example_04_beam_comparison.png', dpi=150,
                     bbox_inches='tight')
        plt.close(fig2)
        print("  ✓ Saved: example_04_beam_comparison.png")

    # Plot 3: Point source vs Line source
    if result_thorp is not None and result_line is not None:
        fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        _, _ = plot_field(result_thorp, env=env, ax=ax1,
                                      show_colorbar=False)
        ax1.set_title("Point Source (Cylindrical)\nRunType: 'CB RR2 '")

        _, _ = plot_field(result_line, env=env, ax=ax2,
                                      show_colorbar=False)
        ax2.set_title("Line Source (Cartesian)\nRunType: 'CB XR2 '")

        # Shared colorbar, taken from a panel's own mappable.
        cbar_ax = fig3.add_axes([0.92, 0.15, 0.02, 0.7])
        cb = fig3.colorbar(_tl_mappable(ax1), cax=cbar_ax, orientation='vertical')
        cb.set_label('TL (dB)', fontsize=12, fontweight='bold')

        fig3.suptitle('Bellhop: Point vs Line Source (fixed 20-120 dB TL scale)',
                      fontsize=16, fontweight='bold')
        fig3.savefig(OUTPUT_DIR / 'example_04_source_comparison.png', dpi=150,
                     bbox_inches='tight')
        plt.close(fig3)
        print("  ✓ Saved: example_04_source_comparison.png")

    # Plot 4: Ray trace
    # Using color_by="bounces" for ray color-coding
    if result_rays is not None:
        fig4, ax4 = result_rays.plot(env=env,
                                     color_by="bounces")  # Color-code rays by bounce type
        ax4.set_title("Ray Trace with Beam Shift\nRunType: 'Rg RR2S'\n" +
                      '(rays colored by bounce type - R/G/B/K)')
        fig4.savefig(OUTPUT_DIR / 'example_04_rays.png', dpi=150,
                     bbox_inches='tight')
        plt.close(fig4)
        print("  ✓ Saved: example_04_rays.png")

    # Plot 5: Multi-source-depth — one TL panel per source slab.
    if stack is not None:
        n_sd = stack.n_slabs
        fig5, axes5 = plt.subplots(1, n_sd, figsize=(6 * n_sd, 5))
        if n_sd == 1:
            axes5 = [axes5]
        for ax, (sd_value, slab) in zip(axes5, stack):
            plot_field(slab.to_db(), env=env, ax=ax,
                                   show_colorbar=False)
            # Mark the source location (r = 0 km, z = source depth)
            # — TL plots use km on x and m on y.
            ax.plot(0.0, sd_value, marker='*', markersize=18,
                    color='white', markeredgecolor='black',
                    markeredgewidth=1.2, zorder=10, clip_on=False)
            ax.set_title(f'Source depth = {sd_value:.0f} m')
        cbar_ax = fig5.add_axes([0.92, 0.15, 0.015, 0.7])
        cb = fig5.colorbar(_tl_mappable(axes5[0]), cax=cbar_ax,
                           orientation='vertical')
        cb.set_label('TL (dB)', fontsize=12, fontweight='bold')
        fig5.suptitle(
            'Bellhop multi-source-depth: one binary call, '
            'ResultStack[Field] slabs',
            fontsize=15, fontweight='bold',
        )
        fig5.savefig(OUTPUT_DIR / 'example_04_multi_source.png',
                     dpi=150, bbox_inches='tight')
        plt.close(fig5)
        print("  ✓ Saved: example_04_multi_source.png")

    # Plot 6: the directivity beside the field it produces. The polar axes are
    # oriented like the TL panels next to them — 0 deg along increasing range,
    # positive angles downward — so the lobe points at the water it ensonifies.
    if result_directional is not None and result_thorp is not None:
        fig6 = plt.figure(figsize=(18, 5.5))

        ax_pattern = fig6.add_subplot(1, 3, 1, projection='polar')
        source_directional.plot_beam_pattern(
            ax=ax_pattern,
            title=f'Source directivity\n{BEAMWIDTH_DEG:.0f}° beam aimed at {TILT_DEG:.0f}°')

        ax_omni = fig6.add_subplot(1, 3, 2)
        plot_field(result_thorp, env=env, ax=ax_omni, show_colorbar=False)
        ax_omni.set_title('Omnidirectional source\n(beam_pattern=None)')

        ax_dir = fig6.add_subplot(1, 3, 3)
        plot_field(result_directional, env=env, ax=ax_dir, show_colorbar=False)
        ax_dir.set_title('Directional source\n(.sbp, RunType(3:3) = \'*\')')

        cbar_ax = fig6.add_axes([0.92, 0.15, 0.015, 0.7])
        cb = fig6.colorbar(_tl_mappable(ax_omni), cax=cbar_ax,
                           orientation='vertical')
        cb.set_label('TL (dB)', fontsize=12, fontweight='bold')

        # Room for a two-line panel title under the suptitle: add_subplot fills
        # more of the figure than plt.subplots leaves, so the default top
        # margin puts the suptitle through the titles.
        fig6.subplots_adjust(top=0.74)
        fig6.suptitle('Bellhop: source directivity shapes the field',
                      fontsize=15, fontweight='bold')
        fig6.savefig(OUTPUT_DIR / 'example_04_beam_pattern.png', dpi=150,
                     bbox_inches='tight')
        plt.close(fig6)
        print("  ✓ Saved: example_04_beam_pattern.png")

    print("\nFeatures demonstrated:")
    print("  ✓ Advanced RunType (7 positions)")
    print("  ✓ Cerveny beam parameters")
    print("  ✓ Thorp volume attenuation")
    print("  ✓ Point vs Line sources")
    print("  ✓ Directional source from a .sbp beam pattern")
    print("  ✓ Multi-source-depth → ResultStack[Field] (.at(source_depth=z))")
    print("  ✓ Beam shift on reflection")
    print("  ✓ Range-dependent bottom properties")
    print("  ✓ Continental shelf scenario")
    print("\nPlotting features demonstrated:")
    print("  ✓ Polar beam pattern, oriented like the field it produced")
    print("  ✓ Ray color-coding by bounce type (red/green/blue/black)")
    print("  ✓ Contour overlays on TL plots (labeled contours)")
    print("  ✓ Fixed TL limits, 20 to 120 dB, shared by every panel")
    print("  ✓ Subplot colorbar control (shared colorbar off a panel's mappable)")
    print("  ✓ jet_r colormap (red=low TL/loud, blue=high TL/quiet)")

    print("\n✓ Example 04 complete\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
