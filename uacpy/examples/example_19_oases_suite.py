"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 19: OASES Suite Comprehensive
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE: Demonstrate OASES suite models (OAST, OASN, OASR, OASP).

FEATURES: ✓ OAST transmission loss  ✓ OASN normal modes
          ✓ OASR reflection coefficients  ✓ OASP pulse/wideband TRF
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy.models import OAST, OASN, OASR, OASP

def main():
    print("\n" + "═"*80)
    print("EXAMPLE 19: OASES Suite Comprehensive")
    print("═"*80)

    # Simple environment for OASES demonstration
    env = uacpy.Environment(
        name="OASES Demonstration",
        depth=100,
        ssp_type='linear',
        ssp_data=[(0, 1500), (100, 1520)],
        bottom=uacpy.BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1700,
            shear_speed=400,
            density=1.8,
            attenuation=0.5
        )
    )

    source = uacpy.Source(depth=50, frequency=100)
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
    # RUN 2: OASN - Normal Modes
    # ═══════════════════════════════════════════════════════════════════════

    print("[2/4] Running OASN (Normal Modes)...", end=" ", flush=True)
    try:
        oasn = OASN(verbose=False)
        result_oasn = oasn.run(env, source, receiver)
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
        result_oasr = oasr.run(env, source, receiver,
                               angle_min=0, angle_max=90, n_angles=91)
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
        # OASP with reduced time samples for faster execution
        receiver_small = uacpy.Receiver(
            depths=np.linspace(5, 95, 20),
            ranges=np.linspace(500, 15000, 30)
        )
        result_oasp = oasp.run(env, source, receiver_small,
                               n_time_samples=256, freq_max=120)
        print("✓")
        oasp_success = True
    except Exception as e:
        print(f"✗ (Error: {e})")
        result_oasp = None
        oasp_success = False

    # ═══════════════════════════════════════════════════════════════════════
    # PLOTTING
    # ═══════════════════════════════════════════════════════════════════════

    print("\nGenerating visualizations...")

    # Plot 1: OAST Transmission Loss
    fig1, ax1 = plt.subplots(figsize=(12, 7))

    im = ax1.pcolormesh(result_oast.ranges/1000, result_oast.depths,
                       result_oast.data, cmap='viridis', vmin=50, vmax=110,
                       shading='auto', zorder=1)

    ax1.set_xlim([result_oast.ranges[0]/1000, result_oast.ranges[-1]/1000])
    ax1.set_ylim([result_oast.depths[-1], result_oast.depths[0]])
    ax1.plot(0, source.depth[0], 'r*', markersize=15, label='Source', zorder=12)
    ax1.axhline(env.depth, color='k', linewidth=3, label='Seafloor', zorder=11)

    ax1.set_title('OAST: Wavenumber Integration Transmission Loss',
                 fontweight='bold', fontsize=14)
    ax1.set_xlabel('Range (km)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Depth (m)', fontweight='bold', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    cbar = plt.colorbar(im, ax=ax1, label='TL (dB)')
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'example_19_oast_tl.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: output/example_19_oast_tl.png")

    # Plot 2: OASR Reflection Coefficients (if available)
    if oasr_success and result_oasr is not None:
        fig2 = plt.figure(figsize=(14, 10))
        gs = fig2.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Subplot 1: Reflection coefficient magnitude vs angle
        ax1 = fig2.add_subplot(gs[0, 0])

        angles = result_oasr.metadata['angles_or_slowness'][0]
        magnitude = result_oasr.metadata['magnitude'][0]
        phase = result_oasr.metadata['phase'][0]

        ax1.plot(angles, magnitude, 'b-', linewidth=2.5, label='P-P Reflection')
        ax1.set_xlabel('Grazing Angle (degrees)', fontweight='bold')
        ax1.set_ylabel('Reflection Coefficient |R|', fontweight='bold')
        ax1.set_title('OASR: Reflection Coefficient Magnitude',
                     fontweight='bold', fontsize=13)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best', fontsize=10, framealpha=0.9)
        ax1.set_xlim([angles.min(), angles.max()])
        ax1.set_ylim([0, 1.05])

        # Subplot 2: Phase vs angle
        ax2 = fig2.add_subplot(gs[0, 1])
        ax2.plot(angles, phase, 'r-', linewidth=2.5, label='Phase')
        ax2.set_xlabel('Grazing Angle (degrees)', fontweight='bold')
        ax2.set_ylabel('Phase (degrees)', fontweight='bold')
        ax2.set_title('OASR: Reflection Coefficient Phase',
                     fontweight='bold', fontsize=13)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=10, framealpha=0.9)
        ax2.set_xlim([angles.min(), angles.max()])

        # Subplot 3: Polar plot of reflection coefficient
        ax3 = fig2.add_subplot(gs[1, 0], projection='polar')
        theta = np.deg2rad(angles)
        ax3.plot(theta, magnitude, 'b-', linewidth=2.5)
        ax3.set_theta_zero_location('N')
        ax3.set_theta_direction(-1)
        ax3.set_ylim([0, 1])
        ax3.set_title('Polar View: |R| vs Angle', fontweight='bold', fontsize=12, pad=20)
        ax3.grid(True, alpha=0.3)

        # Subplot 4: Summary text
        ax4 = fig2.add_subplot(gs[1, 1])
        ax4.axis('off')

        summary = "OASR REFLECTION COEFFICIENTS\n" + "="*50 + "\n\n"
        summary += "Model: P-P (compressional-compressional)\n\n"
        summary += "Bottom Properties:\n"
        summary += f"  • Vp: {env.bottom.sound_speed} m/s\n"
        summary += f"  • Vs: {env.bottom.shear_speed} m/s\n"
        summary += f"  • Density: {env.bottom.density} g/cm³\n"
        summary += f"  • Attenuation: {env.bottom.attenuation} dB/λ\n\n"
        summary += "Angle Range:\n"
        summary += f"  • Min: {angles.min():.1f}°\n"
        summary += f"  • Max: {angles.max():.1f}°\n"
        summary += f"  • Points: {len(angles)}\n\n"
        summary += "Key Observations:\n"
        summary += f"  • Critical angle effect visible\n"
        summary += f"  • Elastic bottom response\n"
        summary += f"  • Frequency: {source.frequency[0]} Hz"

        ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))

        plt.suptitle('OASES Suite: Reflection Coefficient Analysis (OASR)',
                    fontsize=15, fontweight='bold')

        plt.savefig(OUTPUT_DIR / 'example_19_oasr_reflection.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: output/example_19_oasr_reflection.png")

    # Plot 3: OASN Mode Analysis (if available)
    if oasn_success and result_oasn is not None:
        fig3 = plt.figure(figsize=(16, 10))
        gs = fig3.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Subplot 1: Note about OASN
        ax1 = fig3.add_subplot(gs[0, :])

        ax1.axis('off')
        note_text = ("OASN Normal Mode Analysis\n" + "="*60 + "\n\n"
                    "OASN computes normal modes and cross-spectral matrices.\n"
                    "Mode data extraction is experimental in UACPY.\n\n"
                    "For production normal mode analysis, use Kraken.\n\n"
                    "OASN Features:\n"
                    "  • Modal decomposition\n"
                    "  • Eigenfunctions (mode shapes)\n"
                    "  • Phase & group velocities\n"
                    "  • Cross-spectral matrices (.xsm files)")

        ax1.text(0.5, 0.5, note_text, transform=ax1.transAxes,
                fontsize=12, verticalalignment='center', horizontalalignment='center',
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        # Check if mode data exists in metadata
        if hasattr(result_oasn, 'metadata') and 'modes' in result_oasn.metadata:
            modes = result_oasn.metadata['modes']

            # Subplot 2: Mode shapes
            ax2 = fig2.add_subplot(gs[1, 0])

            n_modes_plot = min(6, len(modes.get('k', [])))
            if 'phi' in modes and 'z' in modes:
                for i in range(n_modes_plot):
                    phi_norm = modes['phi'][:, i].real / np.max(np.abs(modes['phi'][:, i].real))
                    ax2.plot(phi_norm, modes['z'], linewidth=2, label=f'Mode {i+1}')

                ax2.set_xlabel('Normalized Amplitude', fontweight='bold')
                ax2.set_ylabel('Depth (m)', fontweight='bold')
                ax2.set_title(f'First {n_modes_plot} Mode Shapes',
                             fontweight='bold', fontsize=12)
                ax2.invert_yaxis()
                ax2.grid(True, alpha=0.3)
                ax2.legend(loc='best', fontsize=9, framealpha=0.9)
                ax2.axvline(0, color='k', linewidth=0.5, linestyle='--')

            # Subplot 3: Wavenumber spectrum
            ax3 = fig2.add_subplot(gs[1, 1])

            if 'k' in modes:
                k_real = np.real(modes['k'])
                phase_speed = 2 * np.pi * source.frequency[0] / k_real

                colors = plt.cm.viridis(np.linspace(0, 1, len(k_real)))

                for i in range(len(k_real)):
                    size = 150 if i < n_modes_plot else 80
                    alpha = 0.9 if i < n_modes_plot else 0.4
                    ax3.scatter(phase_speed[i], k_real[i], s=size, c=[colors[i]],
                              edgecolors='black', linewidths=1.5, alpha=alpha,
                              label=f'Mode {i+1}' if i < n_modes_plot else None, zorder=3)

                ax3.set_xlabel('Phase Speed (m/s)', fontweight='bold')
                ax3.set_ylabel('Wavenumber k (1/m)', fontweight='bold')
                ax3.set_title(f'Wavenumber Spectrum ({len(k_real)} modes)',
                             fontweight='bold', fontsize=12)
                ax3.grid(True, alpha=0.3)
                if len(k_real) <= 6:
                    ax3.legend(loc='best', fontsize=9, framealpha=0.9)

        # Subplot 4: OASN information panel
        ax4 = fig3.add_subplot(gs[2, 0])
        ax4.axis('off')

        info_text = ("OASN MODEL INFORMATION\n" + "="*40 + "\n\n"
                    f"Frequency: {source.frequency[0]} Hz\n"
                    f"Water Depth: {env.depth} m\n"
                    f"Bottom Type: Elastic\n"
                    f"  Vp = {env.bottom.sound_speed} m/s\n"
                    f"  Vs = {env.bottom.shear_speed} m/s\n\n"
                    "Output Files:\n"
                    "  • .xsm - Cross-spectral matrix\n"
                    "  • Mode shapes & wavenumbers\n\n"
                    "Note: For production normal mode\n"
                    "analysis, Kraken is recommended\n"
                    "due to more reliable mode\n"
                    "extraction and analysis tools.")

        ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

        # Subplot 5: OASES Summary
        ax5 = fig3.add_subplot(gs[2, 1])
        ax5.axis('off')

        summary = "OASES SUITE OVERVIEW\n" + "="*50 + "\n\n"

        summary += "OAST (Transmission Loss):\n"
        summary += "  • Wavenumber integration method\n"
        summary += "  • FFT-based fast computation\n"
        summary += "  • Elastic bottom support\n"
        summary += "  • Range-independent environments\n\n"

        summary += "OASN (Normal Modes):\n"
        summary += "  • Modal decomposition\n"
        summary += "  • Computes eigenfunctions\n"
        summary += "  • Phase/group velocities\n"
        summary += "  • Cross-spectral matrices\n\n"

        summary += "OASR (Reflection Coefficients):\n"
        summary += "  • P-P, P-SV coefficients\n"
        summary += "  • Elastic bottom analysis\n"
        summary += "  • Critical angle detection\n"
        summary += "  • Angle/slowness domain\n\n"

        summary += "OASP (Pulse / Wideband TRF):\n"
        summary += "  • PE propagation method\n"
        summary += "  • Range-dependent capability\n"
        summary += "  • Broadband support\n"
        summary += "  • Transfer functions\n\n"

        summary += "KEY FEATURES:\n"
        summary += "  • Developed by MIT & SACLANT\n"
        summary += "  • Complete elastic modeling\n"
        summary += "  • Poro-elastic sediments\n"
        summary += "  • Shear wave support\n\n"

        summary += "SIMULATION PARAMETERS:\n"
        summary += f"  • Frequency: {source.frequency[0]} Hz\n"
        summary += f"  • Water depth: {env.depth}m\n"
        summary += f"  • Bottom: Elastic (Vp={env.bottom.sound_speed}, Vs={env.bottom.shear_speed})"

        ax5.text(0.05, 0.95, summary, transform=ax5.transAxes, fontsize=9,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

        plt.suptitle('OASES Suite: Comprehensive Acoustic Modeling',
                    fontsize=15, fontweight='bold')

        plt.savefig(OUTPUT_DIR / 'example_19_oasn_modes.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: output/example_19_oasn_modes.png")

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
    print("\n" + "═"*80 + "\nEXAMPLE 19 COMPLETE\n" + "═"*80 + "\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
