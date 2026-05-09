"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 13: OASES Suite Comprehensive
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE: Demonstrate OASES suite models (OAST, OASN, OASR, OASP).

FEATURES: ✓ OAST transmission loss
          ✓ OASN spatial covariance C(f, i, j)
          ✓ OASR reflection coefficients
          ✓ OASP pulse/wideband TRF
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.core.environment import SoundSpeedProfile
from uacpy.models import OAST, OASN, OASR, OASP

def main():
    print("\n" + "═" * 80)
    print("EXAMPLE 13: OASES Suite Comprehensive")
    print("═" * 80)

    # Simple environment for OASES demonstration
    env = uacpy.Environment(
        name="OASES Demonstration",
        bathymetry=100,
        ssp=SoundSpeedProfile.from_pairs(
            [(0, 1500), (100, 1520)], interp='linear',
        ),
        bottom=uacpy.BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1700,
            shear_speed=400,
            density=1.8,
            attenuation=0.5
        )
    )

    source = uacpy.Source(depths=50, frequencies=100)
    receiver = uacpy.Receiver(
        depths=np.linspace(5, 95, 40),
        ranges=np.linspace(500, 15000, 60)
    )

    # ═══════════════════════════════════════════════════════════════════════
    # RUN 1: OAST - Transmission Loss
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[1/4] Running OAST (Transmission Loss)...", end=" ", flush=True)
    oast = OAST(verbose=False)
    result_oast = oast.run(env, source, receiver)
    print("✓")

    # ═══════════════════════════════════════════════════════════════════════
    # RUN 2: OASN - Spatial covariance matrix
    # ═══════════════════════════════════════════════════════════════════════

    print("[2/4] Running OASN (spatial covariance)...", end=" ", flush=True)
    try:
        oasn = OASN(verbose=False)
        # Without surface_noise_level the cov collapses to the 0 dB
        # white-noise floor (identity); 70 dB ≈ Wenz amplitude at 100 Hz.
        result_oasn = oasn.compute_covariance(
            env, source, receiver, surface_noise_level=70.0,
        )
        print("✓")
        oasn_success = True
    except Exception as e:
        print(f"✗ (Error: {e})")
        result_oasn = None
        oasn_success = False

    # ═══════════════════════════════════════════════════════════════════════
    # RUN 3: OASR - Reflection Coefficients
    # ═══════════════════════════════════════════════════════════════════════

    print("[3/4] Running OASR (Reflection Coefficients)...", end=" ", flush=True)
    try:
        oasr = OASR(verbose=False)
        # OASR computes reflection coefficients as function of angle
        angles = np.linspace(0, 90, 91)
        result_oasr = oasr.run(env, source, receiver, angles=angles)
        print("✓")
        oasr_success = True
    except Exception as e:
        print(f"✗ (Error: {e})")
        result_oasr = None
        oasr_success = False

    # ═══════════════════════════════════════════════════════════════════════
    # RUN 4: OASP - Pulse / Wideband Transfer Function
    # ═══════════════════════════════════════════════════════════════════════

    print("[4/4] Running OASP (Pulse / Wideband TRF)...", end=" ", flush=True)
    try:
        oasp = OASP(verbose=False)
        receiver_small = uacpy.Receiver(
            depths=np.linspace(5, 95, 20),
            ranges=np.linspace(500, 15000, 30),
        )
        # BROADBAND returns a TransferFunction H(d, r, f) so the example can
        # render TL at center frequency and synthesize a time trace.
        result_oasp = oasp.run(
            env, source, receiver_small,
            run_mode=uacpy.RunMode.BROADBAND,
            n_time_samples=256, freq_max=120,
        )
        print("✓")
        oasp_success = True
    except Exception as e:
        print(f"✗ (Error: {e})")
        result_oasp = None
        oasp_success = False

    # ═══════════════════════════════════════════════════════════════════════
    # PLOTTING — driven by uacpy.plot helpers. Each result type knows how
    # to render itself via ``.plot(env=...)``.
    # ═══════════════════════════════════════════════════════════════════════

    print("\nGenerating visualizations...")

    # Plot 1: OAST transmission loss (PressureField (units="dB") → plot_transmission_loss).
    fig1, _ = uacpy.plot.plot_transmission_loss(result_oast, env=env)
    fig1.savefig(OUTPUT_DIR / 'example_13_oast_tl.png',
                 dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print("  ✓ Saved: output/example_13_oast_tl.png")

    # Plot 2: OASR reflection coefficient (1-D R(θ) and phase).
    if oasr_success and result_oasr is not None:
        rc_for_plot = (
            result_oasr.at_frequency(result_oasr.frequencies[len(result_oasr.frequencies) // 2])
            if result_oasr.is_broadband else result_oasr
        )
        fig2, _ = uacpy.plot.plot_reflection_coefficient(
            rc_for_plot, show_phase=True,
            title=(
                f"OASR {result_oasr.metadata.get('reflection_type', 'P-P')} "
                f"@ {rc_for_plot.f0:.1f} Hz"
            ),
        )
        fig2.savefig(OUTPUT_DIR / 'example_13_oasr_reflection.png',
                     dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print("  ✓ Saved: output/example_13_oasr_reflection.png")

        # Broadband: also save a |R(θ, f)| heatmap (no helper for this yet).
        if result_oasr.is_broadband:
            fig2b, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(
                result_oasr.R, origin='lower', aspect='auto', cmap='viridis',
                extent=[
                    result_oasr.frequencies[0], result_oasr.frequencies[-1],
                    result_oasr.theta[0], result_oasr.theta[-1],
                ],
                vmin=0, vmax=1,
            )
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Grazing angle (deg)')
            ax.set_title('|R(θ, f)|')
            fig2b.colorbar(im, ax=ax, label='|R|')
            fig2b.savefig(OUTPUT_DIR / 'example_13_oasr_broadband.png',
                          dpi=150, bbox_inches='tight')
            plt.close(fig2b)
            print("  ✓ Saved: output/example_13_oasr_broadband.png")

    # Plot 3: OASN spatial covariance heatmap (Covariance → plot_covariance).
    if oasn_success and result_oasn is not None:
        fig3, _ = uacpy.plot.plot_covariance(result_oasn)
        fig3.savefig(OUTPUT_DIR / 'example_13_oasn_covariance.png',
                     dpi=150, bbox_inches='tight')
        plt.close(fig3)
        print("  ✓ Saved: output/example_13_oasn_covariance.png")

    # Plot 4: OASP — broadband transfer function. Slice |H| → dB
    # PressureField at the center frequency, then use the helper. Also
    # synthesize a time trace with a Gaussian pulse and let
    # TimeTrace.plot() render it.
    if oasp_success and result_oasp is not None:
        from uacpy.core.constants import PRESSURE_FLOOR
        from uacpy import PressureField

        H = result_oasp.data
        freqs_h = result_oasp.frequencies
        depths_h = result_oasp.depths
        ranges_h = result_oasp.ranges
        f_center = float(freqs_h[len(freqs_h) // 2])
        k_c = int(np.argmin(np.abs(freqs_h - f_center)))

        TL_centre = -20.0 * np.log10(np.maximum(np.abs(H[..., k_c]), PRESSURE_FLOOR))
        tl_field = PressureField(
            data=TL_centre, depths=depths_h, ranges=ranges_h,
            units='dB',
            model='OASP', backend='oasp',
            source_depths=result_oasp.source_depths,
            frequencies=f_center,
        )
        fig4, _ = uacpy.plot.plot_transmission_loss(tl_field, env=env)
        fig4.savefig(OUTPUT_DIR / 'example_13_oasp_tl.png',
                     dpi=150, bbox_inches='tight')
        plt.close(fig4)
        print("  ✓ Saved: output/example_13_oasp_tl.png")

        # Synthesized time trace at one (depth, range) using a Gaussian pulse.
        d_pick = float(depths_h[int(np.argmin(np.abs(depths_h - source.depths[0])))])
        r_pick = float(ranges_h[int(np.argmin(np.abs(ranges_h - 5000.0)))])
        fs = 4.0 * float(freqs_h[-1])
        nt_pulse = 64
        t_pulse = np.arange(nt_pulse) / fs
        sigma = nt_pulse / (8.0 * fs)
        pulse = (
            np.sin(2 * np.pi * f_center * (t_pulse - t_pulse[-1] / 2))
            * np.exp(-((t_pulse - t_pulse[-1] / 2) ** 2) / (2 * sigma ** 2))
        )
        try:
            ts = result_oasp.synthesize_time_series(source_waveform=pulse, sample_rate=fs)
            trace = ts.get_trace(depth=d_pick, range_m=r_pick)
            fig5, _ = uacpy.plot.plot_time_trace(trace)
            fig5.savefig(OUTPUT_DIR / 'example_13_oasp_trace.png',
                         dpi=150, bbox_inches='tight')
            plt.close(fig5)
            print("  ✓ Saved: output/example_13_oasp_trace.png")
        except Exception as e:
            print(f"  ! Skipped time-series synthesis: {e}")

    print("\n✓ OASES suite examples complete")
    print("\nAll 4 OASES Models Demonstrated:")
    print("  • OAST: Fast wavenumber integration for TL")
    print("  • OASN: Normal mode analysis with eigenfunctions")
    print("  • OASR: Reflection coefficients (P-P, P-SV)")
    print("  • OASP: Pulse / wideband transfer-function propagation")
    print("\nKey Capabilities:")
    print("  • Complete elastic modeling (compression + shear)")
    print("  • Range-independent and range-dependent scenarios")
    print("  • Broadband and narrowband analysis")
    print("\n✓ Example 13 complete\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
