"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 14: Ambient Noise & Bubble Acoustics
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE: Demonstrate ambient noise modeling using AmbientNoiseSimulator

FEATURES: ✓ Wind noise models  ✓ Shipping noise  ✓ Rain noise
          ✓ Biological sources  ✓ Built-in plotting with text boxes
          ✓ Multiple noise source composition
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from uacpy.noise import AmbientNoiseSimulator
import os

def scenario_a_wind_shipping():
    """Basic wind and shipping noise with built-in plotting."""
    print("\n" + "="*80)
    print("SCENARIO A: Wind and Shipping Noise")
    print("="*80)

    # Create simulator
    simulator = AmbientNoiseSimulator()

    # Add wind noise (10 m/s) using Piggot-Merklinger model
    simulator.add_wind("piggot_merklinger", wind_speed=10, water_depth="deep",
                      label="Wind (10 m/s)")

    # Add shipping noise (medium level) using Wenz model
    simulator.add_shipping("wenz", shipping_level="medium", water_depth="deep",
                          label="Shipping")

    # Plot with built-in function (includes text box with model details)
    fig, ax = simulator.plot(
        title='Wind (10 m/s) + Medium Shipping',
        show_total=True
    )

    os.makedirs('output', exist_ok=True)
    plt.savefig('output/example_14a_wind_shipping.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  ✓ Generated: output/example_14a_wind_shipping.png")
    print("  ✓ Plot includes text box with model details")


def scenario_b_multiple_sources():
    """Multiple noise sources - coastal scenario."""
    print("\n" + "="*80)
    print("SCENARIO B: Multiple Noise Sources (Coastal)")
    print("="*80)

    # Create simulator
    simulator = AmbientNoiseSimulator()

    # Add multiple noise sources
    simulator.add_wind("piggot_merklinger", wind_speed=8, water_depth="deep",
                      label="Wind (8 m/s)")
    simulator.add_shipping("wenz", shipping_level="medium", water_depth="deep",
                          label="Shipping")
    simulator.add_rain("ma_nystuen", rain_rate=5, label="Rain (5 mm/hr)")
    simulator.add_biological("snapping_shrimp_cato", shrimp_activity="moderate",
                            label="Snapping Shrimp")

    # Plot with built-in function (includes text box with all source details)
    fig, ax = simulator.plot(
        title='Coastal Scenario: Multiple Noise Sources',
        show_total=True
    )

    plt.savefig('output/example_14b_multiple_sources.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  ✓ Generated: output/example_14b_multiple_sources.png")
    print("  ✓ Plot includes text box with all source parameters")


def scenario_c_wind_model_comparison():
    """Compare different wind noise models."""
    print("\n" + "="*80)
    print("SCENARIO C: Wind Noise Model Comparison")
    print("="*80)

    wind_speed = 12  # m/s

    # Test different wind models
    models = [
        ("knudsen", "Knudsen (1948)"),
        ("piggot_merklinger", "Piggot-Merklinger"),
        ("wilson", "Wilson"),
        ("kewley", "Kewley et al.")
    ]

    fig, ax = plt.subplots(figsize=(12, 8))

    for model_name, label in models:
        simulator = AmbientNoiseSimulator()
        simulator.add_wind(model_name, wind_speed=wind_speed, water_depth="deep")

        # Compute and get the noise data
        components, total_noise = simulator.compute()
        freq = simulator.freq

        ax.semilogx(freq, total_noise, linewidth=2.5, label=label, alpha=0.8)

    ax.set_xlabel('Frequency (Hz)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Noise Level (dB re 1 µPa²/Hz)', fontweight='bold', fontsize=12)
    ax.set_title(f'Wind Noise Model Comparison - {wind_speed} m/s',
                fontweight='bold', fontsize=14)
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='upper right')
    ax.set_xlim([freq.min(), freq.max()])

    # Add text box with parameters
    textstr = f'Parameters:\n  Wind speed: {wind_speed} m/s\n  Water: Deep\n  Models: 4'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')

    plt.tight_layout()
    plt.savefig('output/example_14c_wind_models.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Generated: output/example_14c_wind_models.png")
    print(f"  ✓ Compared {len(models)} wind noise models")


def scenario_d_deep_vs_shallow():
    """Compare deep vs shallow water noise."""
    print("\n" + "="*80)
    print("SCENARIO D: Deep vs Shallow Water Comparison")
    print("="*80)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Collect all data first to determine consistent axis limits
    all_noise_values = []
    plot_data = []

    for idx, water_depth in enumerate(["deep", "shallow"]):
        simulator = AmbientNoiseSimulator()

        # Add same sources for both environments
        simulator.add_wind("piggot_merklinger", wind_speed=10,
                          water_depth=water_depth, label="Wind")
        simulator.add_shipping("wenz", shipping_level="medium",
                              water_depth=water_depth, label="Shipping")

        # Compute noise
        components, total_noise = simulator.compute()
        freq = simulator.freq

        # Store data for plotting
        plot_data.append((idx, water_depth, components, total_noise, freq))

        # Collect all noise values for ylim computation
        for noise_values in components.values():
            all_noise_values.extend(noise_values)
        all_noise_values.extend(total_noise)

    # Compute consistent axis limits from all data
    all_noise_values = np.array(all_noise_values)
    all_noise_values = all_noise_values[np.isfinite(all_noise_values)]
    ylim_min = np.floor(np.min(all_noise_values) / 10) * 10  # Round down to nearest 10
    ylim_max = np.ceil(np.max(all_noise_values) / 10) * 10   # Round up to nearest 10

    # Now plot with consistent limits
    for idx, water_depth, components, total_noise, freq in plot_data:
        ax = axes[idx]

        # Plot individual sources
        for name, noise_values in components.items():
            ax.semilogx(freq, noise_values, linewidth=2,
                       label=name, alpha=0.7)

        # Plot total
        ax.semilogx(freq, total_noise, 'k-', linewidth=3,
                   label='Total', alpha=0.9)

        ax.set_xlabel('Frequency (Hz)', fontweight='bold')
        ax.set_ylabel('Noise Level (dB re 1 µPa²/Hz)', fontweight='bold')
        ax.set_title(f'{water_depth.capitalize()} Water', fontweight='bold', fontsize=14)
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_xlim([freq.min(), freq.max()])
        ax.set_ylim([ylim_min, ylim_max])

    plt.tight_layout()
    plt.savefig('output/example_14d_deep_shallow.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("  ✓ Generated: output/example_14d_deep_shallow.png")
    print("  ✓ Compared deep vs shallow water environments")


def main():
    print("\n" + "═"*80)
    print("EXAMPLE 14: Ambient Noise & Bubble Acoustics")
    print("═"*80)
    print("\nUsing AmbientNoiseSimulator with built-in plotting (includes text boxes)")

    # Run all scenarios
    scenario_a_wind_shipping()
    scenario_b_multiple_sources()
    scenario_c_wind_model_comparison()
    scenario_d_deep_vs_shallow()

    # Summary
    print("\n" + "═"*80)
    print("EXAMPLE 14 COMPLETE")
    print("═"*80)
    print("\nKey features demonstrated:")
    print("  • AmbientNoiseSimulator class with built-in plotting")
    print("  • Text boxes showing model details and parameters")
    print("  • Wind noise models (Knudsen, Piggot-Merklinger, Wilson, Kewley)")
    print("  • Shipping noise (Wenz model)")
    print("  • Rain noise (Ma-Nystuen model)")
    print("  • Biological sources (snapping shrimp)")
    print("  • Deep vs shallow water comparison")
    print("\nGenerated 4 plots in output/ directory")
    print("═"*80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
