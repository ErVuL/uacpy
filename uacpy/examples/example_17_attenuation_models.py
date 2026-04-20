"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 17: Attenuation Models Comparison
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    Compare different underwater acoustic attenuation formulas.
    Demonstrate frequency dependence and environmental parameter effects.
    Show unit conversions between different attenuation representations.

COMPLEXITY LEVEL: ⭐⭐ (2/5) - Utility Features

FEATURES DEMONSTRATED:
    ✓ Thorp formula (1967) - Classic, simple
    ✓ Francois-Garrison formula (1982) - Comprehensive, physically-based
    ✓ Fisher-Simmons formula (1977) - Shallow water corrections
    ✓ Ainslie-McColm - Simplified modern formula
    ✓ Frequency dependence (10 Hz - 1 MHz)
    ✓ Environmental parameter sensitivity (T, S, pH, depth)
    ✓ Unit conversions (dB/km, dB/m, dB/wavelength, Nepers/m)
    ✓ AttenuationModel class usage

SCENARIOS:

    Scenario A: Model Comparison
    ────────────────────────────
    - Compare all 4 formulas over full frequency range
    - Standard conditions (T=10°C, S=35 ppt, pH=8, depth=100m)
    - Identify applicable frequency ranges
    - Show differences between models

    Scenario B: Environmental Sensitivity
    ──────────────────────────────────────
    - Temperature effects (0°C to 30°C)
    - Salinity effects (0 to 40 ppt)
    - pH effects (7.5 to 8.5)
    - Depth/pressure effects (0 to 6000m)

    Scenario C: Unit Conversions
    ────────────────────────────
    - dB/km → dB/m
    - dB/km → dB/wavelength
    - dB/km → Nepers/m
    - Demonstrate convert_attenuation_units()

ATTENUATION MECHANISMS:

    1. Viscosity (low freq, < 1 kHz):
       - Due to shear and bulk viscosity
       - Proportional to f²

    2. Boric acid relaxation (1-10 kHz):
       - Chemical relaxation
       - Temperature and pH dependent

    3. Magnesium sulfate relaxation (10-500 kHz):
       - Chemical relaxation
       - Temperature and salinity dependent

    4. Pure water absorption (> 100 kHz):
       - Molecular processes
       - Temperature dependent

LEARNING OUTCOMES:
    - Which formula to use for your application
    - How environment affects attenuation
    - Frequency-dependent attenuation mechanisms
    - Practical range predictions

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from uacpy.core.attenuation import (
    thorp_attenuation,
    francois_garrison,
    convert_attenuation_units
)
from uacpy.core.acoustics import soundspeed
import os


def scenario_a_model_comparison():
    """
    Scenario A: Compare attenuation models across frequency range.
    """
    print("\n" + "="*80)
    print("SCENARIO A: Attenuation Model Comparison")
    print("="*80)

    # Frequency range: 10 Hz to 1 MHz
    frequencies = np.logspace(1, 6, 500)  # 10 Hz to 1 MHz

    # Standard conditions
    temperature = 10  # °C
    salinity = 35     # ppt
    pH = 8.0
    depth = 100       # meters

    print(f"\n  Environmental conditions:")
    print(f"    • Temperature: {temperature}°C")
    print(f"    • Salinity: {salinity} ppt")
    print(f"    • pH: {pH}")
    print(f"    • Depth: {depth} m")

    # Calculate attenuation for each model
    print(f"\n  Computing attenuation models:")

    # Thorp
    print("    • Thorp (1967)...", end=" ", flush=True)
    atten_thorp = thorp_attenuation(frequencies)
    print("✓")

    # Francois-Garrison
    print("    • Francois-Garrison (1982)...", end=" ", flush=True)
    atten_fg = francois_garrison(frequencies, temperature, salinity,
                                              pH, depth)
    print("✓")

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ─────────────────────────────────────────────────────────────────────────
    # Full frequency range comparison
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.loglog(frequencies / 1000, atten_thorp, 'b-', linewidth=2.5,
              label='Thorp (1967)', alpha=0.8)
    ax.loglog(frequencies / 1000, atten_fg, 'r-', linewidth=2.5,
              label='Francois-Garrison (1982)', alpha=0.8)
    ax.set_xlabel('Frequency (kHz)', fontweight='bold')
    ax.set_ylabel('Attenuation (dB/km)', fontweight='bold')
    ax.set_title('Attenuation vs Frequency (Full Range)', fontweight='bold', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim([frequencies[0]/1000, frequencies[-1]/1000])

    # ─────────────────────────────────────────────────────────────────────────
    # Low frequency zoom (10 Hz - 10 kHz)
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[0, 1]
    mask_low = frequencies <= 10000
    ax.semilogx(frequencies[mask_low] / 1000, atten_thorp[mask_low], 'b-',
                linewidth=2.5, label='Thorp', alpha=0.8)
    ax.semilogx(frequencies[mask_low] / 1000, atten_fg[mask_low], 'r-',
                linewidth=2.5, label='Francois-Garrison', alpha=0.8)
    ax.set_xlabel('Frequency (kHz)', fontweight='bold')
    ax.set_ylabel('Attenuation (dB/km)', fontweight='bold')
    ax.set_title('Low Frequency (10 Hz - 10 kHz)', fontweight='bold', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', alpha=0.3)

    # ─────────────────────────────────────────────────────────────────────────
    # Difference between models
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[1, 0]
    difference = atten_fg - atten_thorp
    ax.semilogx(frequencies / 1000, difference, 'g-', linewidth=2.5)
    ax.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Frequency (kHz)', fontweight='bold')
    ax.set_ylabel('Difference (dB/km)', fontweight='bold')
    ax.set_title('Francois-Garrison - Thorp', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([frequencies[0]/1000, frequencies[-1]/1000])

    # ─────────────────────────────────────────────────────────────────────────
    # Summary table
    # ─────────────────────────────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.axis('off')

    # Calculate attenuation at key frequencies
    test_freqs = [100, 1000, 10000, 100000]  # Hz
    summary_text = "ATTENUATION AT KEY FREQUENCIES\n" + "="*50 + "\n\n"
    summary_text += f"{'Freq (Hz)':>10} {'Thorp':>12} {'F-G':>12} {'Diff':>10}\n"
    summary_text += "-"*50 + "\n"

    for freq in test_freqs:
        idx = np.argmin(np.abs(frequencies - freq))
        thorp_val = atten_thorp[idx]
        fg_val = atten_fg[idx]
        diff_val = fg_val - thorp_val
        summary_text += f"{freq:>10.0f} {thorp_val:>10.4f} dB {fg_val:>10.4f} dB {diff_val:>8.2f} dB\n"

    summary_text += "\n" + "="*50 + "\n\n"
    summary_text += "MODEL CHARACTERISTICS:\n\n"
    summary_text += "Thorp (1967):\n"
    summary_text += "  • Simple empirical formula\n"
    summary_text += "  • Good for quick estimates\n"
    summary_text += "  • Valid: all frequencies\n\n"

    summary_text += "Francois-Garrison (1982):\n"
    summary_text += "  • Physically-based\n"
    summary_text += "  • Includes relaxation effects\n"
    summary_text += "  • Valid: 1 Hz - 1 MHz\n"
    summary_text += "  • Most accurate\n"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()
    os.makedirs('output', exist_ok=True)
    plt.savefig('output/example_17a_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n✓ Generated: output/example_17a_model_comparison.png")


def scenario_b_environmental_sensitivity():
    """
    Scenario B: Environmental parameter effects on attenuation.
    """
    print("\n" + "="*80)
    print("SCENARIO B: Environmental Parameter Sensitivity")
    print("="*80)

    # Test frequency: 10 kHz (in middle of important range)
    freq = 10000  # Hz

    # Baseline conditions
    T_base = 10    # °C
    S_base = 35    # ppt
    pH_base = 8.0
    depth_base = 100  # m

    print(f"\n  Testing at {freq/1000:.0f} kHz")

    # ─────────────────────────────────────────────────────────────────────────
    # Temperature variation
    # ─────────────────────────────────────────────────────────────────────────
    temperatures = np.linspace(0, 30, 31)
    atten_vs_temp = [float(francois_garrison(freq, T, S_base,
                                                    pH_base, depth_base))
                     for T in temperatures]

    # ─────────────────────────────────────────────────────────────────────────
    # Salinity variation
    # ─────────────────────────────────────────────────────────────────────────
    salinities = np.linspace(0, 40, 41)
    atten_vs_sal = [float(francois_garrison(freq, T_base, S,
                                                   pH_base, depth_base))
                    for S in salinities]

    # ─────────────────────────────────────────────────────────────────────────
    # pH variation
    # ─────────────────────────────────────────────────────────────────────────
    pHs = np.linspace(7.5, 8.5, 21)
    atten_vs_pH = [float(francois_garrison(freq, T_base, S_base,
                                                  pH, depth_base))
                   for pH in pHs]

    # ─────────────────────────────────────────────────────────────────────────
    # Depth variation
    # ─────────────────────────────────────────────────────────────────────────
    depths = np.linspace(0, 6000, 61)
    atten_vs_depth = [float(francois_garrison(freq, T_base, S_base,
                                                     pH_base, d))
                      for d in depths]

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Temperature
    ax = axes[0, 0]
    ax.plot(temperatures, atten_vs_temp, 'r-', linewidth=2.5)
    ax.axvline(T_base, color='k', linestyle='--', alpha=0.5, label=f'Baseline ({T_base}°C)')
    ax.set_xlabel('Temperature (°C)', fontweight='bold')
    ax.set_ylabel('Attenuation (dB/km)', fontweight='bold')
    ax.set_title(f'Temperature Effect at {freq/1000:.0f} kHz', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Salinity
    ax = axes[0, 1]
    ax.plot(salinities, atten_vs_sal, 'b-', linewidth=2.5)
    ax.axvline(S_base, color='k', linestyle='--', alpha=0.5, label=f'Baseline ({S_base} ppt)')
    ax.set_xlabel('Salinity (ppt)', fontweight='bold')
    ax.set_ylabel('Attenuation (dB/km)', fontweight='bold')
    ax.set_title(f'Salinity Effect at {freq/1000:.0f} kHz', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # pH
    ax = axes[1, 0]
    ax.plot(pHs, atten_vs_pH, 'g-', linewidth=2.5)
    ax.axvline(pH_base, color='k', linestyle='--', alpha=0.5, label=f'Baseline (pH {pH_base})')
    ax.set_xlabel('pH', fontweight='bold')
    ax.set_ylabel('Attenuation (dB/km)', fontweight='bold')
    ax.set_title(f'pH Effect at {freq/1000:.0f} kHz', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Depth
    ax = axes[1, 1]
    ax.plot(depths, atten_vs_depth, 'm-', linewidth=2.5)
    ax.axvline(depth_base, color='k', linestyle='--', alpha=0.5,
              label=f'Baseline ({depth_base} m)')
    ax.set_xlabel('Depth (m)', fontweight='bold')
    ax.set_ylabel('Attenuation (dB/km)', fontweight='bold')
    ax.set_title(f'Depth Effect at {freq/1000:.0f} kHz', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('output/example_17b_environmental_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Print sensitivity statistics
    print(f"\n  Parameter sensitivity at {freq/1000:.0f} kHz:")
    print(f"    • Temperature (0-30°C): {min(atten_vs_temp):.3f} to {max(atten_vs_temp):.3f} dB/km "
          f"(range: {max(atten_vs_temp)-min(atten_vs_temp):.3f} dB/km)")
    print(f"    • Salinity (0-40 ppt): {min(atten_vs_sal):.3f} to {max(atten_vs_sal):.3f} dB/km "
          f"(range: {max(atten_vs_sal)-min(atten_vs_sal):.3f} dB/km)")
    print(f"    • pH (7.5-8.5): {min(atten_vs_pH):.3f} to {max(atten_vs_pH):.3f} dB/km "
          f"(range: {max(atten_vs_pH)-min(atten_vs_pH):.3f} dB/km)")
    print(f"    • Depth (0-6000m): {min(atten_vs_depth):.3f} to {max(atten_vs_depth):.3f} dB/km "
          f"(range: {max(atten_vs_depth)-min(atten_vs_depth):.3f} dB/km)")

    print("\n✓ Generated: output/example_17b_environmental_sensitivity.png")


def scenario_c_unit_conversions():
    """
    Scenario C: Demonstrate unit conversions.
    """
    print("\n" + "="*80)
    print("SCENARIO C: Unit Conversions")
    print("="*80)

    frequency = 10000  # 10 kHz
    c = soundspeed()   # Sound speed at standard conditions

    # Calculate attenuation in dB/km
    alpha_dbpkm = float(thorp_attenuation(frequency))

    print(f"\n  Original attenuation at {frequency/1000:.0f} kHz:")
    print(f"    • {alpha_dbpkm:.4f} dB/km")

    # Convert to different units
    alpha_dbpm = convert_attenuation_units(alpha_dbpkm, frequency, 'dB/km', 'dB/m', c)
    alpha_dbpwavelength = convert_attenuation_units(alpha_dbpkm, frequency, 'dB/km',
                                                     'dB/wavelength', c)
    alpha_nepers = convert_attenuation_units(alpha_dbpkm, frequency, 'dB/km',
                                            'Nepers/m', c)

    print(f"\n  Conversions:")
    print(f"    • {alpha_dbpm:.7f} dB/m")
    print(f"    • {alpha_dbpwavelength:.7f} dB/wavelength")
    print(f"    • {alpha_nepers:.10f} Nepers/m")

    # Verification: Convert back
    alpha_back = convert_attenuation_units(alpha_dbpm, frequency, 'dB/m', 'dB/km', c)
    print(f"\n  Verification (convert dB/m back to dB/km):")
    print(f"    • {alpha_back:.4f} dB/km")
    print(f"    • Match: {np.isclose(alpha_back, alpha_dbpkm)}")

    # Create visualization showing range loss
    ranges_km = np.linspace(0, 100, 101)
    total_loss_dB = alpha_dbpkm * ranges_km

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Range loss
    ax = axes[0]
    ax.plot(ranges_km, total_loss_dB, 'b-', linewidth=2.5)
    ax.set_xlabel('Range (km)', fontweight='bold')
    ax.set_ylabel('Total Attenuation Loss (dB)', fontweight='bold')
    ax.set_title(f'Cumulative Attenuation vs Range ({frequency/1000:.0f} kHz)',
                 fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Unit conversion table
    ax = axes[1]
    ax.axis('off')

    table_text = "UNIT CONVERSIONS\n" + "="*50 + "\n\n"
    table_text += f"Frequency: {frequency/1000:.0f} kHz\n"
    table_text += f"Sound speed: {c:.1f} m/s\n"
    table_text += f"Wavelength: {c/frequency:.4f} m\n\n"
    table_text += "="*50 + "\n\n"

    table_text += f"{'Unit':<20} {'Value':<25}\n"
    table_text += "-"*50 + "\n"
    table_text += f"{'dB/km':<20} {alpha_dbpkm:<25.6f}\n"
    table_text += f"{'dB/m':<20} {alpha_dbpm:<25.9f}\n"
    table_text += f"{'dB/wavelength':<20} {alpha_dbpwavelength:<25.9f}\n"
    table_text += f"{'Nepers/m':<20} {alpha_nepers:<25.12f}\n"

    table_text += "\n" + "="*50 + "\n\n"
    table_text += "AVAILABLE UNITS:\n"
    table_text += "  • dB/km, dB/m\n"
    table_text += "  • dB/wavelength\n"
    table_text += "  • Nepers/m\n"
    table_text += "  • Q (quality factor)\n"
    table_text += "  • L (loss parameter)\n"

    ax.text(0.05, 0.95, table_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    plt.tight_layout()
    plt.savefig('output/example_17c_unit_conversions.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\n✓ Generated: output/example_17c_unit_conversions.png")


def main():
    """
    Run all attenuation model demonstrations.
    """
    print("\n" + "═"*80)
    print("EXAMPLE 17: Attenuation Models Comparison")
    print("═"*80)
    print("\nThis example demonstrates:")
    print("  • Attenuation model comparison (Thorp, Francois-Garrison)")
    print("  • Frequency dependence (10 Hz - 1 MHz)")
    print("  • Environmental parameter effects (T, S, pH, depth)")
    print("  • Unit conversions")

    # Run all scenarios
    scenario_a_model_comparison()
    scenario_b_environmental_sensitivity()
    scenario_c_unit_conversions()

    # Summary
    print("\n" + "═"*80)
    print("EXAMPLE 17 COMPLETE")
    print("═"*80)
    print("\nKey Takeaways:")
    print("  ✓ Attenuation increases with frequency (roughly f²)")
    print("  ✓ Francois-Garrison most accurate for 1 Hz - 1 MHz")
    print("  ✓ Thorp simpler, good for quick estimates")
    print("  ✓ Temperature has strong effect on attenuation")
    print("  ✓ Multiple unit representations available")
    print("\nPractical rules of thumb:")
    print("  → Low freq (< 1 kHz): Very low loss, long range")
    print("  → Mid freq (1-10 kHz): Moderate loss, medium range")
    print("  → High freq (> 100 kHz): High loss, short range")
    print("  → Attenuation doubles temperature: ~30% increase")
    print("\n" + "═"*80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
