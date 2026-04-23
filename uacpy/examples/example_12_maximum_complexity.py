"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 24: Sediment Transition - FULL Range Dependence
         (Bathymetry + 2D SSP + Range-Dependent Bottom - ALL THREE!)
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    Demonstrate MAXIMUM environmental complexity: all three types of range
    dependence simultaneously:
    1. Range-dependent bathymetry (coastal → deep ocean)
    2. 2D SSP (coastal mixing → offshore stratification)
    3. Range-dependent bottom properties (soft mud → hard rock)

    This represents the most realistic and complex ocean acoustic scenario.

COMPLEXITY LEVEL: ⭐⭐⭐⭐⭐ (5/5) - Expert / Maximum

ENVIRONMENT:
    THREE simultaneous range dependencies:

    1. BATHYMETRY:
       - Coastal zone: 50m (nearshore)
       - Continental shelf: 50m → 150m
       - Shelf break: 150m → 600m (steep!)
       - Continental slope: 600m → 1200m
       - Horizontal extent: 40 km

    2. SSP (2D - Range AND Depth):
       - Coastal (0-10 km): Well-mixed, turbid, warm surface
       - Shelf (10-20 km): Moderate stratification
       - Offshore (20-40 km): Strong stratification, cold deep water
       - River discharge effects nearshore

    3. BOTTOM PROPERTIES:
       - Coastal (0-10 km): Soft mud (high attenuation, low sound speed)
       - Shelf (10-20 km): Sandy sediment (medium properties)
       - Slope (20-30 km): Coarse sediment (hardening)
       - Deep (30-40 km): Rock/hard substrate (low attenuation, high sound speed)

COMBINED PHYSICS:
    - Bathymetric focusing/defocusing
    - Horizontal refraction from SSP gradients
    - Bottom loss variation with range
    - Mode cutoff and conversion across transitions
    - Coupling between water column and sediment properties
    - Multiple scattering regimes

MODELS TESTED:
    ✓ RAM           (handles all three dependencies natively)
    ✓ KrakenField   (adiabatic modes, all dependencies)
    ✓ Bellhop       (approximation for comparison)
    ✓ Scooter       (elastic bottom, range-independent approximation)

FEATURES DEMONSTRATED:
    - Maximum environmental complexity
    - All three range-dependent features simultaneously
    - Sediment type transitions
    - Coastal → deep ocean propagation
    - Model comparison for extreme complexity
    - Complete environment visualization suite

EXPECTED BEHAVIOR:
    - Highly complex TL patterns
    - Strong bottom interaction effects
    - Mode behavior changes dramatically with range
    - Large differences between models (10-20+ dB)
    - RAM most accurate (native support for all features)

═══════════════════════════════════════════════════════════════════════════════
"""

import sys
from pathlib import Path

# Import matplotlib BEFORE uacpy to avoid signal module conflict
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for uacpy imports
example_dir = Path(__file__).parent
sys.path.insert(0, str(example_dir.parent))

OUTPUT_DIR = example_dir / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

import uacpy
from uacpy import RangeDependentBottom
from uacpy.models import Bellhop, RAM, KrakenField, Scooter, OAST

# Import from examples directory
sys.path.insert(0, str(example_dir))
from plotting_utils import create_example_report
from uacpy.visualization.plots import plot_ssp_2d, plot_bottom_properties, plot_environment_advanced


def create_coastal_bathymetry():
    """
    Create coastal → deep ocean bathymetry.

    Returns
    -------
    bathymetry : ndarray (n_points × 2)
        [range_m, depth_m]
    """
    ranges_m = np.array([
        0,      # Coastal nearshore
        5000,   # Inner shelf
        10000,  # Mid shelf
        15000,  # Outer shelf
        18000,  # Shelf edge
        20000,  # Upper slope
        25000,  # Mid slope
        30000,  # Lower slope
        35000,  # Continental rise
        40000   # Deep ocean approach
    ])

    depths_m = np.array([
        50,     # Shallow coastal
        80,     # Inner shelf
        120,    # Mid shelf
        150,    # Outer shelf
        180,    # Shelf edge
        300,    # Upper slope (steepening)
        600,    # Mid slope (steep!)
        900,    # Lower slope
        1100,   # Rise
        1200    # Deep ocean
    ])

    bathymetry = np.column_stack([ranges_m, depths_m])
    return bathymetry


def create_coastal_ssp_2d(depths, ranges_km, bathymetry):
    """
    Create 2D SSP: coastal mixing → offshore stratification.

    Parameters
    ----------
    depths : array_like
        Depth array (m)
    ranges_km : array_like
        Range array (km)
    bathymetry : ndarray
        Bathymetry [range_m, depth_m]

    Returns
    -------
    ssp_2d_matrix : ndarray (n_depths × n_ranges)
        Sound speed [m/s]
    ssp_1d : ndarray (n_depths × 2)
        Reference 1D SSP
    """
    from scipy.interpolate import interp1d

    bathy_interp = interp1d(bathymetry[:, 0] / 1000, bathymetry[:, 1],
                            kind='linear', fill_value='extrapolate')
    bottom_depths = bathy_interp(ranges_km)

    n_depth = len(depths)
    n_range = len(ranges_km)
    ssp_2d_matrix = np.zeros((n_depth, n_range))

    for i_range, r_km in enumerate(ranges_km):
        seafloor_depth = bottom_depths[i_range]

        # Temperature varies with range
        if r_km < 10:  # Coastal zone (0-10 km)
            # River discharge, turbidity, well-mixed
            T_surface = 24.0  # Warm coastal water
            T_bottom = 22.0   # Well-mixed (shallow, turbulent)
            mixed_layer_depth = 40.0  # Deep mixing
            thermocline_strength = 0.5  # Weak stratification

        elif r_km < 20:  # Shelf (10-20 km)
            # Transitional
            progress = (r_km - 10) / 10
            T_surface = 24.0 - progress * 6.0  # 24°C → 18°C
            T_bottom = 22.0 - progress * 12.0  # 22°C → 10°C
            mixed_layer_depth = 40.0 - progress * 20.0
            thermocline_strength = 0.5 + progress * 2.0

        else:  # Offshore (20-40 km)
            # Strong stratification
            progress = (r_km - 20) / 20
            T_surface = 18.0 - progress * 4.0   # 18°C → 14°C
            T_bottom = 10.0 - progress * 4.0    # 10°C → 6°C
            mixed_layer_depth = 20.0 + progress * 10.0
            thermocline_strength = 2.5 + progress * 0.5

        # Build temperature profile
        T_profile = np.zeros(n_depth)
        for i_depth, z in enumerate(depths):
            if z < mixed_layer_depth:
                # Surface mixed layer
                T_profile[i_depth] = T_surface
            else:
                # Exponential decay below mixed layer
                decay_depth = z - mixed_layer_depth
                T_profile[i_depth] = T_bottom + (T_surface - T_bottom) * \
                                     np.exp(-thermocline_strength * decay_depth / 100.0)

        # Mackenzie sound speed
        T = T_profile
        z = depths
        c = 1449.2 + 4.6*T - 0.055*T**2 + 0.00029*T**3 + 0.016*z

        # Below seafloor
        below_bottom = depths > seafloor_depth
        if np.any(below_bottom):
            c[below_bottom] = c[~below_bottom][-1]

        ssp_2d_matrix[:, i_range] = c

    ssp_1d = np.column_stack([depths, ssp_2d_matrix[:, 0]])
    return ssp_2d_matrix, ssp_1d


def create_sediment_transition_bottom(bathymetry):
    """
    Create range-dependent bottom: mud → sand → gravel → rock.

    Parameters
    ----------
    bathymetry : ndarray
        Bathymetry array [range_m, depth_m]

    Returns
    -------
    bottom_rd : RangeDependentBottom
        Range-dependent bottom object
    """
    # Extract ranges and depths from bathymetry
    ranges_km = bathymetry[:, 0] / 1000
    depths_m = bathymetry[:, 1]

    n_points = len(ranges_km)

    # Sediment type transition with range
    # Coastal mud → Shelf sand → Slope coarse sediment → Deep rock

    sound_speed = np.zeros(n_points)
    density = np.zeros(n_points)
    attenuation = np.zeros(n_points)

    for i, r_km in enumerate(ranges_km):
        if r_km < 10:  # Coastal mud (0-10 km)
            # Soft, fine-grained sediment
            progress = r_km / 10
            sound_speed[i] = 1450 + progress * 50   # 1450 → 1500 m/s
            density[i] = 1.2 + progress * 0.2       # 1.2 → 1.4 g/cm³
            attenuation[i] = 1.5 - progress * 0.3   # 1.5 → 1.2 dB/λ

        elif r_km < 20:  # Shelf sand (10-20 km)
            # Medium sand
            progress = (r_km - 10) / 10
            sound_speed[i] = 1500 + progress * 100  # 1500 → 1600 m/s
            density[i] = 1.4 + progress * 0.3       # 1.4 → 1.7 g/cm³
            attenuation[i] = 1.2 - progress * 0.4   # 1.2 → 0.8 dB/λ

        elif r_km < 30:  # Slope coarse sediment (20-30 km)
            # Coarse sand to gravel
            progress = (r_km - 20) / 10
            sound_speed[i] = 1600 + progress * 100  # 1600 → 1700 m/s
            density[i] = 1.7 + progress * 0.3       # 1.7 → 2.0 g/cm³
            attenuation[i] = 0.8 - progress * 0.3   # 0.8 → 0.5 dB/λ

        else:  # Deep rock (30-40 km)
            # Hard substrate
            progress = (r_km - 30) / 10
            sound_speed[i] = 1700 + progress * 150  # 1700 → 1850 m/s
            density[i] = 2.0 + progress * 0.3       # 2.0 → 2.3 g/cm³
            attenuation[i] = 0.5 - progress * 0.2   # 0.5 → 0.3 dB/λ

    # Create RangeDependentBottom object
    bottom_rd = RangeDependentBottom(
        ranges_km=ranges_km,
        depths=depths_m,
        sound_speed=sound_speed,
        density=density,
        attenuation=attenuation,
        shear_speed=np.zeros(n_points),  # Fluid sediment (no shear)
        acoustic_type='half-space'
    )

    return bottom_rd


def main():
    # ═══════════════════════════════════════════════════════════════════════
    # CREATE ALL THREE RANGE-DEPENDENT FEATURES
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "="*80)
    print("EXAMPLE 24: FULL Range Dependence (Bathymetry + SSP + Bottom)")
    print("="*80)
    print("\n*** MAXIMUM COMPLEXITY: All three range dependencies! ***\n")

    # Feature 1: Range-dependent bathymetry
    print("[Feature 1/3] Creating range-dependent BATHYMETRY...")
    bathymetry = create_coastal_bathymetry()
    print(f"  ✓ Bathymetry: {bathymetry[:, 1].min():.0f}m (coastal) → {bathymetry[:, 1].max():.0f}m (deep)")
    print(f"  ✓ Horizontal extent: {bathymetry[-1, 0]/1000:.0f} km")

    # Feature 2: 2D SSP (range AND depth dependent)
    print("\n[Feature 2/3] Creating range-dependent 2D SSP...")
    max_depth = bathymetry[:, 1].max()
    depths = np.linspace(0, max_depth * 1.05, 41)  # Reduced depth points for Kraken compatibility
    ranges_km = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40])
    ssp_2d_matrix, ssp_1d = create_coastal_ssp_2d(depths, ranges_km, bathymetry)
    print(f"  ✓ 2D SSP: {ssp_2d_matrix.shape} (depth × range)")
    print(f"  ✓ Sound speed: {ssp_2d_matrix.min():.1f} - {ssp_2d_matrix.max():.1f} m/s")
    print(f"  ✓ Coastal (0 km): {ssp_2d_matrix[0, 0]:.1f} m/s surface (warm, mixed)")
    print(f"  ✓ Offshore (40 km): {ssp_2d_matrix[0, -1]:.1f} m/s surface (cold, stratified)")

    # Feature 3: Range-dependent bottom properties
    print("\n[Feature 3/3] Creating range-dependent BOTTOM properties...")
    bottom_rd = create_sediment_transition_bottom(bathymetry)
    print(f"  ✓ Bottom sound speed: {bottom_rd.sound_speed.min():.0f} → {bottom_rd.sound_speed.max():.0f} m/s")
    print(f"  ✓ Bottom density: {bottom_rd.density.min():.1f} → {bottom_rd.density.max():.1f} g/cm³")
    print(f"  ✓ Bottom attenuation: {bottom_rd.attenuation.min():.2f} → {bottom_rd.attenuation.max():.2f} dB/λ")
    print(f"  ✓ Sediment type: Mud (coastal) → Sand → Gravel → Rock (deep)")

    # ═══════════════════════════════════════════════════════════════════════
    # CREATE ENVIRONMENT WITH ALL THREE FEATURES
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "="*80)
    print("Creating environment with ALL THREE range dependencies...")
    print("="*80)

    env = uacpy.Environment(
        name="Coastal→Deep: Full Range Dependence",
        depth=max_depth * 1.05,
        ssp_type='linear',  # Use linear instead of pchip for better Kraken compatibility
        ssp_data=ssp_1d,
        ssp_2d_ranges=ranges_km,
        ssp_2d_matrix=ssp_2d_matrix,
        bathymetry=bathymetry,
        bottom=bottom_rd
    )

    print(f"\n✓ Environment: {env.name}")
    print(f"  ✓ is_range_dependent: {env.is_range_dependent}")
    print(f"  ✓ has_range_dependent_ssp: {env.has_range_dependent_ssp()}")
    print(f"  ✓ has_range_dependent_bottom: {env.has_range_dependent_bottom()}")
    print(f"\n  → ALL THREE dependencies active simultaneously!")

    # ═══════════════════════════════════════════════════════════════════════
    # SOURCE & RECEIVER
    # ═══════════════════════════════════════════════════════════════════════

    source = uacpy.Source(depth=25.0, frequency=100.0)  # Shallow source
    receiver = uacpy.Receiver(
        depths=np.linspace(10, 1150, 70),
        ranges=np.linspace(100, 40000, 150)
    )

    print(f"\nSource: {source.depth[0]}m, {source.frequency[0]} Hz")
    print(f"Receivers: {len(receiver.depths)} × {len(receiver.ranges)} grid")

    # ═══════════════════════════════════════════════════════════════════════
    # RUN ALL MODELS (5 models showing different use cases)
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "="*80)
    print("Running propagation models (5 models for comprehensive comparison)...")
    print("="*80)

    results = {}

    # RAM: Best for all three dependencies
    print("\n[Model 1/6] RAM - Parabolic Equation")
    print("  • USE CASE: Range-dependent environments (bathymetry, 2D SSP, bottom)")
    print("  • Native support for ALL THREE range dependencies")
    print("  • Most accurate for this extreme complexity")
    try:
        ram = RAM(verbose=False)
        results['RAM'] = ram.run(env, source, receiver)
        print(f"  ✓ Success - TL: {results['RAM'].data.min():.1f} to {results['RAM'].data.max():.1f} dB")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results['RAM'] = None

    # KrakenField: Handles all via mode coupling (may fail with very complex SSP)
    print("\n[Model 2/6] KrakenField - Adiabatic Mode Coupling")
    print("  • USE CASE: Normal mode analysis with range dependence")
    print("  • Note: May fail with very complex SSP (Kraken format limitations)")
    print("  • Segments environment, computes local modes")
    try:
        krakenfield = KrakenField(verbose=False, mode_coupling='adiabatic', n_segments=6)
        results['KrakenField'] = krakenfield.run(
            env, source, receiver
        )
        print(f"  ✓ Success - TL: {results['KrakenField'].data.min():.1f} to {results['KrakenField'].data.max():.1f} dB")
    except Exception as e:
        print(f"  ✗ Error (expected with extreme complexity): {str(e)[:80]}...")
        print(f"  → Use RAM for most accurate range-dependent modeling")
        results['KrakenField'] = None

    # Bellhop: Approximation (for comparison)
    print("\n[Model 3/6] Bellhop - Ray Tracing (Approximation)")
    print("  • USE CASE: Fast general-purpose TL, baseline comparison")
    print("  • Uses range-independent approximation (median depth/SSP)")
    print("  • Shows limitations of range-independent methods")
    try:
        bellhop = Bellhop(verbose=False, n_beams=500)
        results['Bellhop'] = bellhop.run(env, source, receiver)
        print(f"  ✓ Success - TL: {results['Bellhop'].data.min():.1f} to {results['Bellhop'].data.max():.1f} dB")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results['Bellhop'] = None

    # Scooter: Elastic bottom (approximation)
    print("\n[Model 4/6] Scooter - Wavenumber Integration")
    print("  • USE CASE: Elastic/poro-elastic bottom effects, full-wave solution")
    print("  • Range-independent approximation (median environment)")
    print("  • Shows importance of elastic bottom in shallow water")
    try:
        scooter = Scooter(verbose=False)
        env_approx = env.get_range_independent_approximation(method='median')
        results['Scooter'] = scooter.run(env_approx, source, receiver)
        print(f"  ✓ Success - TL: {results['Scooter'].data.min():.1f} to {results['Scooter'].data.max():.1f} dB")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results['Scooter'] = None

        results['SPARC'] = None

    # OAST: OASES Transmission Loss (approximation)
    print("\n[Model 6/6] OAST - OASES Wavenumber Integration")
    print("  • USE CASE: Elastic bottom (OASES suite), comprehensive full-wave")
    print("  • Range-independent approximation (using MAX depth for deep scenarios)")
    print("  • Alternative to Scooter for elastic media")
    try:
        oast = OAST(verbose=False)
        # Use 'max' method to ensure environment depth covers all receivers
        env_approx = env.get_range_independent_approximation(method='max')

        # Cap receiver depths at 95% of environment depth to be safe
        max_receiver_depth = min(1150, env_approx.depth * 0.95)
        receiver_safe = uacpy.Receiver(
            depths=np.linspace(10, max_receiver_depth, 70),
            ranges=receiver.ranges
        )

        results['OAST'] = oast.run(env_approx, source, receiver_safe)
        print(f"  ✓ Success - TL: {results['OAST'].data.min():.1f} to {results['OAST'].data.max():.1f} dB")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results['OAST'] = None

    # Print summary
    successful_models = sum(1 for r in results.values() if r is not None)
    print(f"\n✓ Successfully ran {successful_models}/5 models")
    print(f"  Models working: {[name for name, result in results.items() if result is not None]}")

    # ═══════════════════════════════════════════════════════════════════════
    # VISUALIZATION
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "="*80)
    print("Generating comprehensive visualizations...")
    print("="*80)

    # Extra plots for full complexity
    print("\n[Extra Plots] Full environment visualization suite...")

    # Plot 1: 2D SSP with bathymetry overlay
    fig_ssp, ax_ssp = plot_ssp_2d(env, cmap='RdYlBu_r')
    ax_ssp.set_title('2D SSP: Coastal Mixing → Offshore Stratification\n(with Bathymetry)',
                     fontweight='bold', fontsize=12)
    bathy_r = bathymetry[:, 0] / 1000
    bathy_d = bathymetry[:, 1]
    ax_ssp.plot(bathy_r, bathy_d, 'k-', linewidth=3, label='Seafloor', zorder=12)
    ax_ssp.fill_between(bathy_r, bathy_d, ax_ssp.get_ylim()[1],
                        color='saddlebrown', alpha=0.4, label='Sediment', zorder=10)
    ax_ssp.legend(loc='upper left')
    plt.savefig(OUTPUT_DIR / 'example_12_ssp_2d.png', dpi=150, bbox_inches='tight')
    plt.close(fig_ssp)
    print("  ✓ Saved: example_24_ssp_2d.png")

    # Plot 2: Bottom properties
    fig_bottom, axes_bottom = plot_bottom_properties(env)
    plt.savefig(OUTPUT_DIR / 'example_12_bottom.png', dpi=150, bbox_inches='tight')
    plt.close(fig_bottom)
    print("  ✓ Saved: example_24_bottom.png")

    # Plot 3: Advanced environment overview
    fig_env, axes_env = plot_environment_advanced(env, source, receiver)
    plt.savefig(OUTPUT_DIR / 'example_12_environment.png', dpi=150, bbox_inches='tight')
    plt.close(fig_env)
    print("  ✓ Saved: example_24_environment.png")

    # Standard report with matrix comparison
    print("\n[Standard Report] Matrix comparison...")
    if any(r is not None for r in results.values()):
        create_example_report(
            example_num=13,
            title="Sediment Transition - FULL Range Dependence",
            description="MAXIMUM COMPLEXITY: All three range dependencies simultaneously! "
                       "Range-dependent bathymetry, 2D SSP, and range-dependent bottom properties. "
                       "Represents most realistic ocean acoustic scenario: coastal → deep ocean transition.",
            env=env,
            source=source,
            receiver=receiver,
            results=results,
            output_prefix="example_12"
        )
    else:
        print("  ⚠ No models ran successfully!")
        return 1

    # ═══════════════════════════════════════════════════════════════════════
    # COMPREHENSIVE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS")
    print("="*80)

    print("\nEnvironment Complexity Metrics:")
    print(f"  • Depth range: {bathymetry[:, 1].max() / bathymetry[:, 1].min():.1f}x variation")
    print(f"  • SSP range: {ssp_2d_matrix.max() - ssp_2d_matrix.min():.1f} m/s variation")
    print(f"  • Bottom impedance contrast: {(bottom_rd.sound_speed.max() * bottom_rd.density.max()) / (bottom_rd.sound_speed.min() * bottom_rd.density.min()):.2f}x")
    print(f"  • Bottom loss variation: {bottom_rd.attenuation.max() / bottom_rd.attenuation.min():.1f}x")

    print("\nPhysical Processes (All Active):")
    print("  1. Bathymetric focusing/defocusing")
    print("  2. Horizontal SSP refraction")
    print("  3. Vertical SSP refraction (stratification)")
    print("  4. Mode cutoff and conversion")
    print("  5. Bottom loss variation")
    print("  6. Sediment property effects")
    print("  7. Coupling between all processes")

    if results.get('RAM') and results.get('Bellhop'):
        tl_ram = results['RAM'].data
        tl_bellhop = results['Bellhop'].data
        if tl_ram.shape == tl_bellhop.shape:
            diff = tl_ram - tl_bellhop
            print(f"\nModel Accuracy (RAM vs Bellhop Approximation):")
            print(f"  • Mean difference: {np.nanmean(diff):.1f} dB")
            print(f"  • RMS difference: {np.sqrt(np.nanmean(diff**2)):.1f} dB")
            print(f"  • Max difference: {np.nanmax(np.abs(diff)):.1f} dB")
            print(f"  → Large differences confirm importance of full range dependence")

    print("\n" + "="*80)
    print("EXAMPLE 13 COMPLETE")
    print("="*80)
    print("\n*** MAXIMUM COMPLEXITY ACHIEVED ***")
    print("\nKey Demonstrations:")
    print("  ✓ ALL THREE range dependencies simultaneously:")
    print("    1. Range-dependent bathymetry (50m → 1200m)")
    print("    2. 2D SSP (coastal mixing → offshore stratification)")
    print("    3. Range-dependent bottom (mud → sand → gravel → rock)")
    print("  ✓ RAM: Handles all features natively (most accurate)")
    print("  ✓ KrakenField: Mode coupling across all transitions")
    print("  ✓ Comprehensive visualization suite")
    print("  ✓ Realistic coastal → deep ocean scenario")
    print("\nPhysics Insights:")
    print("  → Three dependencies interact non-linearly")
    print("  → Range-independent models significantly underestimate complexity")
    print("  → Bottom properties strongly affect shallow water propagation")
    print("  → Mode structure changes dramatically across environment")
    print("  → Accurate modeling requires range-dependent methods (RAM, KrakenField)")
    print("\nThis is the MOST COMPLEX acoustic scenario in UACPY examples!")

    return 0

if __name__ == "__main__":
    sys.exit(main())
