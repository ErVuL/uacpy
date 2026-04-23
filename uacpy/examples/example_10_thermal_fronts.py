"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 22: Thermal Front - 2D SSP (Range AND Depth Dependent)
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    Demonstrate full 2D sound speed field where SSP varies with BOTH range
    AND depth simultaneously. This represents a thermal front/frontal zone
    where warm stratified water meets cold well-mixed water.

COMPLEXITY LEVEL: ⭐⭐⭐⭐ (4/5) - Advanced

ENVIRONMENT:
    - 2D SSP MATRIX: Sound speed = f(range, depth)
    - Flat bottom (300m, no bathymetry variation)
    - Constant bottom properties (no range dependence)
    - Warm water (left) → Cold water (right)
    - Stratified (left) → Well-mixed (right)

THERMAL FRONT PHYSICS:
    Range 0-5 km:   Warm (20°C surface, 10°C bottom) - Strong stratification
    Range 5-10 km:  Transitional mixing
    Range 10-20 km: Cold (12°C surface, 8°C bottom) - Weak stratification

    Sound speed follows Mackenzie formula:
    c ≈ 1449 + 4.6*T - 0.055*T² + 0.00029*T³ + 0.016*z

    Results in:
    - Horizontal refraction (sound bends toward cold water)
    - Vertical stratification changes with range
    - Mode structure variation across front
    - Complex TL field with 2D focusing/defocusing

MODELS TESTED:
    ✓ RAM           (excellent 2D SSP support)
    ✓ KrakenField   (adiabatic mode coupling with range-dependent SSP)
    ✓ Bellhop       (uses median SSP, for comparison)

FEATURES DEMONSTRATED:
    - 2D SSP creation (temperature-based)
    - Range AND depth dependence simultaneously
    - Thermal front propagation physics
    - Horizontal refraction effects
    - Mode coupling across frontal zone
    - 2D SSP visualization (heatmap)
    - Model comparison for range-dependent SSP

EXPECTED BEHAVIOR:
    - RAM: Smooth propagation through 2D SSP, horizontal refraction
    - KrakenField: Mode coupling, adiabatic approximation
    - Bellhop: Uses median SSP (less accurate for strong fronts)
    - TL differences of 5-15 dB across models due to 2D effects

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path
import os

# Import matplotlib BEFORE uacpy to avoid signal module conflict
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for uacpy imports
example_dir = Path(__file__).parent
sys.path.insert(0, str(example_dir.parent))

OUTPUT_DIR = example_dir / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

import uacpy
from uacpy.models import Bellhop, RAM, KrakenField

# Import from examples directory
sys.path.insert(0, str(example_dir))
from example_helpers import create_example_report
from uacpy.visualization.plots import plot_ssp_2d

def create_thermal_front_ssp(depths, ranges_km):
    """
    Create 2D SSP for thermal front scenario.

    Parameters
    ----------
    depths : array_like
        Depth array (m)
    ranges_km : array_like
        Range array (km)

    Returns
    -------
    ssp_2d_matrix : ndarray (n_depths × n_ranges)
        Sound speed matrix [m/s]
    ssp_1d : ndarray (n_depths × 2)
        Reference 1D SSP [depth, c] for first range
    """
    n_depth = len(depths)
    n_range = len(ranges_km)
    ssp_2d_matrix = np.zeros((n_depth, n_range))

    for i_range, r_km in enumerate(ranges_km):
        # Temperature profile varies with range
        # Warm water (0-5 km) → Cold water (10-20 km)

        # Surface temperature: 20°C → 12°C
        T_surface = 20.0 - (r_km / 20.0) * 8.0

        # Bottom temperature: 10°C → 8°C
        T_bottom = 10.0 - (r_km / 20.0) * 2.0

        # Stratification strength varies with range
        # Strong stratification (left) → Weak stratification (right)
        strat_scale = 50.0 - (r_km / 20.0) * 30.0  # 50m → 20m

        # Exponential temperature profile
        T_profile = T_bottom + (T_surface - T_bottom) * np.exp(-depths / strat_scale)

        # Mackenzie sound speed formula (simplified, accurate to ~1 m/s)
        # c = 1449.2 + 4.6*T - 0.055*T² + 0.00029*T³ + (1.34 - 0.010*T)*(S - 35) + 0.016*z
        # Assuming salinity S = 35 psu (standard), simplifies to:
        T = T_profile
        z = depths
        c = 1449.2 + 4.6*T - 0.055*T**2 + 0.00029*T**3 + 0.016*z

        ssp_2d_matrix[:, i_range] = c

    # Reference 1D SSP (first range column)
    ssp_1d = np.column_stack([depths, ssp_2d_matrix[:, 0]])

    return ssp_2d_matrix, ssp_1d


def main():
    # ═══════════════════════════════════════════════════════════════════════
    # CREATE 2D SSP FOR THERMAL FRONT
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "="*80)
    print("EXAMPLE 22: Thermal Front - 2D SSP (Range AND Depth Dependent)")
    print("="*80)

    print("\n[Step 1/5] Creating 2D SSP matrix...")

    # Depth grid for SSP (fine resolution for accuracy)
    depths = np.linspace(0, 300, 31)  # 10m spacing

    # Range points for SSP definition
    ranges_km = np.array([0, 5, 10, 15, 20])

    # Create 2D SSP
    ssp_2d_matrix, ssp_1d = create_thermal_front_ssp(depths, ranges_km)

    print(f"  ✓ 2D SSP created: shape = {ssp_2d_matrix.shape} (depth × range)")
    print(f"  ✓ Sound speed range: {ssp_2d_matrix.min():.1f} - {ssp_2d_matrix.max():.1f} m/s")
    print(f"  ✓ SSP at 0 km (warm): surface = {ssp_2d_matrix[0, 0]:.1f} m/s, bottom = {ssp_2d_matrix[-1, 0]:.1f} m/s")
    print(f"  ✓ SSP at 20 km (cold): surface = {ssp_2d_matrix[0, -1]:.1f} m/s, bottom = {ssp_2d_matrix[-1, -1]:.1f} m/s")

    # ═══════════════════════════════════════════════════════════════════════
    # CREATE ENVIRONMENT WITH 2D SSP
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[Step 2/5] Creating environment with 2D SSP...")

    env = uacpy.Environment(
        name="Thermal Front - 2D SSP",
        depth=300.0,              # Flat bottom (no bathymetry variation)
        ssp_type='pchip',         # PCHIP interpolation for smooth profiles
        ssp_data=ssp_1d,          # Reference 1D profile (first range)
        ssp_2d_ranges=ranges_km,  # Range points for 2D SSP
        ssp_2d_matrix=ssp_2d_matrix,  # Full 2D SSP matrix
    )

    print(f"  ✓ Environment created: {env.name}")
    print(f"  ✓ is_range_dependent: {env.is_range_dependent}")
    print(f"  ✓ has_range_dependent_ssp: {env.has_range_dependent_ssp()}")
    print(f"  ✓ Bathymetry: Flat (constant {env.depth}m)")

    # ═══════════════════════════════════════════════════════════════════════
    # SOURCE & RECEIVER CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[Step 3/5] Configuring source and receivers...")

    source = uacpy.Source(
        depth=50.0,        # Shallow source (in thermocline)
        frequency=100.0    # 100 Hz
    )

    receiver = uacpy.Receiver(
        depths=np.linspace(5, 280, 50),      # 50 depth points
        ranges=np.linspace(100, 20000, 100)  # 0.1 to 20 km
    )

    print(f"  ✓ Source: {source.depth[0]}m depth, {source.frequency[0]} Hz")
    print(f"  ✓ Receivers: {len(receiver.depths)} depths × {len(receiver.ranges)} ranges")

    # ═══════════════════════════════════════════════════════════════════════
    # RUN ALL MODELS
    # ═══════════════════════════════════════════════════════════════════════

    print("\n[Step 4/5] Running propagation models...")
    print("="*80)

    results = {}

    # RAM: Excellent 2D SSP support (native)
    print("\n[Model 1/3] RAM - Parabolic Equation with 2D SSP")
    print("  • Native 2D SSP support (no approximations)")
    print("  • Handles horizontal refraction naturally")
    try:
        ram = RAM(verbose=False)
        results['RAM'] = ram.run(env, source, receiver)
        print(f"  ✓ Success - TL range: {results['RAM'].data.min():.1f} to {results['RAM'].data.max():.1f} dB")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results['RAM'] = None

    # KrakenField: Adiabatic mode coupling (approximation for 2D SSP)
    print("\n[Model 2/3] KrakenField - Adiabatic Mode Coupling")
    print("  • Segments environment by range")
    print("  • Computes local modes at each segment")
    print("  • Applies adiabatic mode theory (gradual SSP variation)")
    try:
        krakenfield = KrakenField(verbose=False, mode_coupling='adiabatic', n_segments=8)
        results['KrakenField'] = krakenfield.run(
            env, source, receiver
        )
        print(f"  ✓ Success - TL range: {results['KrakenField'].data.min():.1f} to {results['KrakenField'].data.max():.1f} dB")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results['KrakenField'] = None

    # Bellhop: Uses median SSP (for comparison)
    print("\n[Model 3/3] Bellhop - Ray Tracing (Median SSP Approximation)")
    print("  • Range-independent: uses median SSP across all ranges")
    print("  • Provides baseline for comparison")
    print("  • Less accurate for strong frontal variations")
    try:
        bellhop = Bellhop(verbose=False, n_beams=500)
        # Bellhop will use range-independent approximation (median SSP)
        results['Bellhop'] = bellhop.run(env, source, receiver)
        print(f"  ✓ Success - TL range: {results['Bellhop'].data.min():.1f} to {results['Bellhop'].data.max():.1f} dB")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results['Bellhop'] = None

    # ═══════════════════════════════════════════════════════════════════════
    # VISUALIZATION AND ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "="*80)
    print("[Step 5/5] Generating visualizations...")
    print("="*80)

    # Additional plot: 2D SSP heatmap
    print("\n[Plot 1/2] 2D SSP heatmap visualization...")
    fig_ssp, ax_ssp = plot_ssp_2d(env, cmap='RdYlBu_r')
    ax_ssp.set_title('Thermal Front: 2D Sound Speed Field\n(Warm Stratified → Cold Well-Mixed)',
                     fontweight='bold', fontsize=13)

    # Add annotations
    ax_ssp.annotate('Warm Water\n(Strong Stratification)',
                   xy=(2, 50), fontsize=11, color='white', ha='center',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.7), zorder=15)
    ax_ssp.annotate('Cold Water\n(Weak Stratification)',
                   xy=(16, 50), fontsize=11, color='white', ha='center',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='blue', alpha=0.7), zorder=15)
    ax_ssp.annotate('Frontal\nZone',
                   xy=(9, 150), fontsize=10, color='black', ha='center',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7), zorder=15)

    plt.savefig(OUTPUT_DIR / 'example_10_thermal_fronts_ssp.png', dpi=150, bbox_inches='tight')
    plt.close(fig_ssp)
    print("  ✓ Saved: output/example_10_thermal_fronts_ssp.png")

    # Standard report (uses matrix comparison)
    print("\n[Plot 2/2] Generating standard example report...")
    if any(r is not None for r in results.values()):
        create_example_report(
            example_num=11,
            title="Thermal Front - 2D SSP (Range AND Depth Dependent)",
            description="Demonstrates full 2D sound speed field where SSP varies with both range AND depth. "
                       "Thermal front scenario: warm stratified water transitions to cold well-mixed water. "
                       "Shows horizontal refraction, mode coupling, and range-dependent propagation physics.",
            env=env,
            source=source,
            receiver=receiver,
            results=results,
            output_prefix="example_10"
        )
    else:
        print("  ⚠ No models ran successfully!")
        return 1

    # ═══════════════════════════════════════════════════════════════════════
    # ANALYSIS AND CONCLUSIONS
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)

    print("\n2D SSP Characteristics:")
    print(f"  • Sound speed variation: {ssp_2d_matrix.max() - ssp_2d_matrix.min():.1f} m/s total")
    print(f"  • Horizontal gradient: {(ssp_2d_matrix[0, -1] - ssp_2d_matrix[0, 0]) / 20:.2f} m/s per km (surface)")
    print(f"  • Vertical gradient (0 km): {(ssp_2d_matrix[-1, 0] - ssp_2d_matrix[0, 0]) / 300:.3f} m/s per m")
    print(f"  • Vertical gradient (20 km): {(ssp_2d_matrix[-1, -1] - ssp_2d_matrix[0, -1]) / 300:.3f} m/s per m")

    if results.get('RAM') and results.get('Bellhop'):
        # Compare RAM (2D SSP) vs Bellhop (median SSP)
        tl_ram = results['RAM'].data
        tl_bellhop = results['Bellhop'].data

        # Find common grid
        if tl_ram.shape == tl_bellhop.shape:
            diff = tl_ram - tl_bellhop
            print(f"\nModel Differences (RAM vs Bellhop):")
            print(f"  • Mean difference: {np.nanmean(diff):.1f} dB")
            print(f"  • Max difference: {np.nanmax(np.abs(diff)):.1f} dB")
            print(f"  • RMS difference: {np.sqrt(np.nanmean(diff**2)):.1f} dB")
            print(f"  → Significant differences indicate importance of 2D SSP modeling")

    print("\n" + "="*80)
    print("EXAMPLE 11 COMPLETE")
    print("="*80)
    print("\nKey Demonstrations:")
    print("  ✓ 2D SSP creation (range AND depth dependent)")
    print("  ✓ Thermal front physics (warm → cold transition)")
    print("  ✓ Horizontal refraction effects")
    print("  ✓ RAM: Native 2D SSP support")
    print("  ✓ KrakenField: Adiabatic mode coupling")
    print("  ✓ Model comparison for range-dependent SSP")
    print("  ✓ 2D SSP visualization (heatmap)")
    print("\nPhysics Insight:")
    print("  → Sound refracts horizontally toward colder (slower) water")
    print("  → Mode coupling occurs across frontal zone")
    print("  → Stratification changes affect vertical propagation")
    print("  → 2D effects produce 5-15 dB differences vs range-independent")

    return 0

if __name__ == "__main__":
    sys.exit(main())
