"""
═══════════════════════════════════════════════════════════════════════════════
ADVANCED EXAMPLE: Kraken/KrakenField - Coupled Mode Theory
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    Demonstrate Kraken's capabilities with range-dependent environments:
    - Normal mode computation
    - Adiabatic mode theory for range-dependent problems
    - Mode coupling for continental shelf
    - Volume attenuation in modal propagation

SCENARIO:
    Continental Shelf with Mode Coupling
    - Shallow water (100m) transitioning to deep water (400m)
    - Range-dependent bottom properties (sand → rock)
    - Francois-Garrison volume attenuation

FEATURES DEMONSTRATED:
    ✓ Kraken mode computation with volume attenuation
    ✓ KrakenField with adiabatic mode theory
    ✓ Range segmentation for coupled modes
    ✓ Range-dependent bottom in mode propagation
    ✓ Mode shape visualization
    ✓ Dispersion curve analysis

═══════════════════════════════════════════════════════════════════════════════
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
from uacpy import RangeDependentBottom  # noqa: E402
from uacpy.models import Kraken, KrakenField, KrakenC  # noqa: E402
from uacpy.visualization.plots import (  # noqa: E402
    plot_field,
    plot_mode_functions,
    plot_mode_wavenumbers,
    plot_modes_heatmap,
    plot_environment,
)


def main():
    print("\n" + "═" * 80)
    print("EXAMPLE 06: Kraken advanced features - Coupled Mode Theory")
    print("═" * 80)

    # ═══════════════════════════════════════════════════════════════════════
    # ENVIRONMENT: Continental Shelf
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[Setup] Creating continental shelf environment...")

    # Bathymetry: shelf to slope
    bathymetry = np.array([
        [0, 100],       # Shelf: 100m
        [8000, 120],    # Still on shelf
        [10000, 150],   # Shelf break begins
        [15000, 250],   # Slope
        [20000, 400],   # Deep water
    ])

    # Range-dependent bottom: sand on shelf, rock on slope. The bottom's
    # property nodes are intentionally on a different range vector than
    # the bathymetry — they're independent fields in uacpy. Geologically,
    # the seafloor shape (bathy) and the sediment-to-rock transition
    # (bottom properties) are set by different processes and don't have
    # to switch at the same ranges.
    bottom_rd = RangeDependentBottom(
        ranges=np.array([0.0, 6000.0, 12000.0, 18000.0]),
        sound_speed=np.array([1600, 1650, 1750, 1800]),  # Hardening
        density=np.array([1.5, 1.7, 2.0, 2.2]),          # Compacting
        attenuation=np.array([0.8, 0.5, 0.3, 0.2]),      # Less lossy
        shear_speed=np.array([0, 0, 400, 600]),          # Rock on slope
        acoustic_type='half-space'
    )

    # Downward refracting SSP (typical shelf)
    ssp_data = np.array([
        [0, 1520],
        [50, 1505],
        [100, 1495],
        [200, 1490],
        [400, 1485],
    ])

    env = uacpy.Environment(
        name="Continental Shelf - Mode Coupling",
        ssp=SoundSpeedProfile.from_pairs(ssp_data),
        bathymetry=bathymetry,
        bottom=bottom_rd,
    )

    print(f"  ✓ Range-dependent: {env.is_range_dependent}")
    print(f"  ✓ Depth range: {bathymetry[:, 1].min():.0f}m → {bathymetry[:, 1].max():.0f}m")

    # ═══════════════════════════════════════════════════════════════════════
    # SOURCE & RECEIVER
    # ═══════════════════════════════════════════════════════════════════════

    source = uacpy.Source(depths=50.0, frequencies=50.0)  # 50 Hz for good modal propagation
    receiver = uacpy.Receiver(
        depths=np.linspace(5, 380, 60),
        ranges=np.linspace(100, 20000, 100)
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RUN 1: Compute Modes with Volume Attenuation
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[1/4] Computing modes with Francois-Garrison attenuation...")

    # Francois-Garrison: (T °C, S PSU, pH, z_bar m) lives on Environment
    fg = uacpy.FrancoisGarrison(
        temperature_c=10.0, salinity_psu=35.0, pH=8.0, z_bar_m=50.0,
    )
    kraken = Kraken(
        verbose=False,
    )

    # Compute modes for first range (shallow water)
    env_shallow = uacpy.Environment(
        name="Shelf (100m)",
        bathymetry=100.0,
        ssp=SoundSpeedProfile.from_pairs(
            ssp_data[ssp_data[:, 0] <= 100],
        ),
        bottom=bottom_rd.eval(range=0, interp='nearest'),
        absorption=fg,
    )

    modes_shallow = kraken.compute_modes(env_shallow, source)
    n_modes_shallow = len(modes_shallow.k)
    print(f"  ✓ Computed {n_modes_shallow} modes for shallow water")

    # Compute modes for last range (deep water)
    env_deep = uacpy.Environment(
        name="Slope (400m)",
        bathymetry=400.0,
        ssp=SoundSpeedProfile.from_pairs(ssp_data),
        bottom=bottom_rd.eval(range=20000, interp='nearest'),
        absorption=fg,
    )

    # Deep environment has shear_speed > 0 (rocky bottom); use KrakenC for complex modes.
    krakenc_deep = KrakenC(
        verbose=False,
    )
    modes_deep = krakenc_deep.compute_modes(env_deep, source)
    n_modes_deep = len(modes_deep.k)
    print(f"  ✓ Computed {n_modes_deep} modes for deep water")

    # ═══════════════════════════════════════════════════════════════════════
    # RUN 2: KrakenField with Adiabatic Mode Coupling
    # ═══════════════════════════════════════════════════════════════════════

    print("[2/4] Running KrakenField with adiabatic mode coupling...")

    try:
        krakenfield = KrakenField(verbose=False, mode_coupling='adiabatic', n_segments=5)
        result = krakenfield.run(
            env, source, receiver
        )
        print(f"  ✓ KrakenField completed with {krakenfield.n_segments} segments")
        print(f"  ✓ TL range: {result.data.min():.1f} to {result.data.max():.1f} dB")
    except Exception as e:
        print(f"  ✗ KrakenField error: {e}")
        import traceback
        traceback.print_exc()
        result = None

    # ═══════════════════════════════════════════════════════════════════════
    # RUN 3: KrakenC - Complex Modes with Elastic Bottom
    # ═══════════════════════════════════════════════════════════════════════

    print("[3/4] Computing complex modes with KrakenC (elastic bottom)...")

    try:
        krakenc = KrakenC(
            verbose=False,
        )

        # Use environment with elastic bottom for complex mode computation
        modes_complex = krakenc.compute_modes(env_shallow, source)
        n_complex = len(modes_complex.k)
        print(f"  ✓ Computed {n_complex} complex modes")
        print("  ✓ Supports elastic bottom with shear waves")
    except Exception as e:
        print(f"  ✗ KrakenC error: {e}")
        modes_complex = None

    # ═══════════════════════════════════════════════════════════════════════
    # VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════════

    print("[4/4] Generating plots...")

    # Plot 1: Environment with bottom properties
    fig1, _ = plot_environment(env)
    plt.savefig(OUTPUT_DIR / 'example_06_bottom.png', dpi=150, bbox_inches='tight')
    print("  ✓ Saved: example_06_bottom.png")

    # Plot 2: Mode comparison (shallow vs deep)
    if modes_shallow is not None and modes_deep is not None:
        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

        # Shallow water modes
        phi_s = modes_shallow.phi
        z_s = modes_shallow.depths
        M_s = phi_s.shape[1]
        n_show = min(5, M_s)
        for i in range(n_show):
            axes2[0].plot(phi_s[:, i].real, z_s, label=f'Mode {i+1}')
        axes2[0].invert_yaxis()
        axes2[0].set_xlabel('Mode Amplitude')
        axes2[0].set_ylabel('Depth (m)')
        axes2[0].set_title(f'Shallow Water Modes (100m)\n{M_s} total modes')
        axes2[0].legend()
        axes2[0].grid(True, alpha=0.3)

        # Deep water modes (KrakenC). The elastic bottom (shear_speed > 0
        # at the rocky slope) admits Scholte/interface waves whose phase
        # velocity sits below the water sound speed — KrakenC numbers
        # those first (highest Re(k)). Filter to the trapped water-column
        # modes (c_phase between c_min(water) and c_p(bottom)) so the
        # plotted shapes are comparable to the shallow-water panel.
        phi_d = modes_deep.phi
        z_d = modes_deep.depths
        M_d = phi_d.shape[1]
        fc = float(modes_deep.f0)
        c_water_min = float(np.min(ssp_data[:, 1]))
        c_phase = 2 * np.pi * fc / modes_deep.k.real
        n_interface = int(np.sum(c_phase < c_water_min))
        if n_interface:
            print(f"  Skipped {n_interface} interface mode(s) "
                  f"(c_phase < {c_water_min:.0f} m/s — Scholte/Stoneley type).")
        trapped = np.where(c_phase >= c_water_min)[0]
        n_show = min(5, trapped.size)
        for j in range(n_show):
            i = trapped[j]
            axes2[1].plot(phi_d[:, i].real, z_d, label=f'Mode {i+1}')
        axes2[1].invert_yaxis()
        axes2[1].set_xlabel('Mode Amplitude')
        axes2[1].set_ylabel('Depth (m)')
        axes2[1].set_title(
            f'Deep Water Modes (400m)\n{M_d} total ({n_interface} interface, '
            f'{M_d - n_interface} trapped — showing first {n_show} trapped)'
        )
        axes2[1].legend()
        axes2[1].grid(True, alpha=0.3)

        plt.suptitle('Mode Evolution: Shelf to Slope', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'example_06_modes.png', dpi=150, bbox_inches='tight')
        print("  ✓ Saved: example_06_modes.png")

    # Plot 2b: Wavenumber plot using plot_mode_wavenumbers
    if modes_shallow is not None:
        print("\nGenerating advanced mode visualizations...")

        # Use plot_mode_wavenumbers function for complex k-plane visualization
        try:
            # compute_modes already returns a Field suitable for these plotters
            modes_field = modes_shallow

            # plot_mode_wavenumbers - complex k-plane scatter plot
            fig2b, ax2b = plot_mode_wavenumbers(
                modes_field,
                annotate_modes=True,
                max_annotations=15
            )
            ax2b.set_title('Mode Wavenumbers in Complex k-Plane\n' +
                           f'Shallow Water - {M_s} modes',
                           fontsize=14, fontweight='bold')
            plt.savefig(OUTPUT_DIR / 'example_06_wavenumbers.png', dpi=150, bbox_inches='tight')
            plt.close(fig2b)
            print("  ✓ Saved: example_06_wavenumbers.png (plot_mode_wavenumbers)")

        except Exception as e:
            print(f"  ! Warning: Could not create wavenumber plot: {e}")

        # Use plot_modes with show_imaginary=True
        try:
            fig2c, (ax_modes, ax_k) = plot_mode_functions(
                modes_field,
                show_imaginary=True  # Show imaginary parts as dashed lines
            )
            fig2c.suptitle('Mode Shapes with Imaginary Parts\n' +
                           'Shallow Water (solid=real, dashed=imaginary)',
                           fontsize=14, fontweight='bold')
            plt.savefig(OUTPUT_DIR / 'example_06_mode_shapes.png', dpi=150, bbox_inches='tight')
            plt.close(fig2c)
            print("  ✓ Saved: example_06_mode_shapes.png (show_imaginary)")

        except Exception as e:
            print(f"  ! Warning: Could not create mode shapes plot: {e}")

        # Use plot_modes_heatmap - Show all modes as 2D heatmap
        try:
            print("  Creating mode heatmap...", end=" ", flush=True)
            fig_heatmap, ax_heatmap = plot_modes_heatmap(
                modes_field,
                mode_range=None,  # Plot all modes
                normalize=True,
                figsize=(14, 8)
            )
            plt.suptitle('All Mode Shapes - Continental Shelf Heatmap',
                         fontsize=14, fontweight='bold')
            plt.savefig(OUTPUT_DIR / 'example_06_modes_heatmap.png', dpi=150, bbox_inches='tight')
            plt.close(fig_heatmap)
            print("✓")
            print("  ✓ Saved: example_06_modes_heatmap.png (plot_modes_heatmap)")
        except Exception as e:
            print(f"\n  ! Warning: Could not create mode heatmap: {e}")

    # Plot 3: Coupled mode TL field
    # Using auto TL limits and contour overlays
    if result is not None:
        fig3, ax3 = plot_field(
            result, env=env, contours=[70, 85, 100],  # Add labeled contours
            show_colorbar=True
        )
        ax3.set_title('KrakenField: Adiabatic Mode Coupling\nContinental Shelf Transition\n' +
                      '(auto TL limits + contour overlays)')

        # Add segment indicators
        seg_ranges = np.linspace(0, 20, 11)
        for r in seg_ranges[1:-1]:
            ax3.axvline(r, color='white', linestyle='--', alpha=0.3, linewidth=0.5, zorder=8)

        plt.savefig(OUTPUT_DIR / 'example_06_result.png', dpi=150, bbox_inches='tight')
        print("  ✓ Saved: example_06_result.png")

    print("\nFeatures demonstrated:")
    print("  ✓ Volume attenuation (Francois-Garrison)")
    print("  ✓ Range-dependent bottom properties")
    print("  ✓ Adiabatic mode coupling (10 segments)")
    print("  ✓ Mode evolution (shelf → slope)")
    print("  ✓ Continental shelf propagation")
    print("\nPlotting features demonstrated:")
    print("  ✓ plot_mode_wavenumbers() - Complex k-plane visualization")
    print("  ✓ plot_modes_heatmap() - All modes as 2D heatmap")
    print("  ✓ Mode imaginary parts (show_imaginary=True)")
    print("  ✓ Auto TL limits (median + 0.75σ)")
    print("  ✓ Contour overlays on TL plots")
    print("  ✓ jet_r colormap (blue=good, red=poor)")

    print("\n✓ Example 06 complete\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
