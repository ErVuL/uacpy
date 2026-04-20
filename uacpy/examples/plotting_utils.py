"""
Harmonized plotting utilities for UACPY examples.
Provides consistent, professional visualization across all examples.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple
import uacpy

# Professional color scheme for all models
COLORS = {
    'Bellhop': '#1f77b4',       # Blue
    'BellhopCUDA': '#5DA5DA',   # Light Blue
    'RAM': '#ff7f0e',           # Orange
    'Kraken': '#2ca02c',        # Green
    'KrakenField': '#2ca02c',   # Green
    'Scooter': '#d62728',       # Red
    'SPARC': '#9467bd',         # Purple
    'OAST': '#8c564b',          # Brown
    'OASN': '#e377c2',          # Pink
    'OASR': '#7f7f7f',          # Gray
    'OASP': '#bcbd22',          # Olive
    'OASES': '#17becf'          # Cyan
}

# Plot style defaults
PLOT_STYLE = {
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'lines.linewidth': 2,
}

# Z-order hierarchy for consistent layering
# Lower values appear below higher values
ZORDER = {
    'TL_FIELD': 1,           # Transmission loss heatmap (bottom layer)
    'GRID': 5,               # Grid lines
    'BATHYMETRY_FILL': 10,   # Sediment/bathymetry fill
    'BATHYMETRY_LINE': 11,   # Seafloor boundary line
    'SOURCE': 12,            # Source marker
    'ANNOTATIONS': 15,       # Text annotations (top layer)
}

def apply_plot_style():
    """Apply consistent plot style"""
    plt.rcParams.update(PLOT_STYLE)

def plot_bathymetry_overlay(ax, env, result, range_min_km=None, range_max_km=None, zorder_fill=10, zorder_line=11):
    """
    Overlay bathymetry on TL plot with proper masking.

    Parameters
    ----------
    ax : matplotlib axis
    env : Environment object
    result : Field result object
    range_min_km : float, optional
        Minimum range in km for x-axis
    range_max_km : float, optional
        Maximum range in km for x-axis
    """
    if range_min_km is None:
        range_min_km = np.min(result.ranges) / 1000
    if range_max_km is None:
        range_max_km = np.max(result.ranges) / 1000

    # Determine plot depth range
    plot_max_depth = np.max(result.depths) * 1.05

    if env.bathymetry is not None and len(env.bathymetry) > 0:
        bathy_ranges_km = env.bathymetry[:, 0] / 1000
        bathy_depths = env.bathymetry[:, 1]

        # Extend bathymetry to plot edges
        bathy_ranges_extended = np.concatenate([[range_min_km], bathy_ranges_km, [range_max_km]])
        bathy_depths_extended = np.concatenate([
            [bathy_depths[0]],
            bathy_depths,
            [bathy_depths[-1]]
        ])

        # Fill below seafloor to bottom of plot
        ax.fill_between(bathy_ranges_extended, bathy_depths_extended, plot_max_depth,
                        color='#8B4513', alpha=1.0, zorder=zorder_fill, linewidth=0)

        # Seafloor line on top
        ax.plot(bathy_ranges_extended, bathy_depths_extended, 'k-', linewidth=2.5,
               label='Seafloor', zorder=zorder_line)
    else:
        # Flat bottom case
        seafloor_depth = env.depth

        # Fill below seafloor to bottom of plot
        ax.fill_between([range_min_km, range_max_km], seafloor_depth, plot_max_depth,
                        color='#8B4513', alpha=1.0, zorder=zorder_fill, linewidth=0)

        # Seafloor line
        ax.plot([range_min_km, range_max_km], [seafloor_depth, seafloor_depth],
               'k-', linewidth=2.5, label='Seafloor', zorder=zorder_line)

def plot_source_receiver_config(env, source, receiver):
    """
    Plot showing source and receiver configuration.

    Returns figure with environment, source, and receiver specs.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Sound Speed Profile
    ax = axes[0]
    ax.plot(env.ssp_data[:,1], env.ssp_data[:,0], 'b-', linewidth=2)
    ax.axhline(source.depth[0], color='r', linestyle='--', linewidth=2, label='Source', alpha=0.7)
    for rd in receiver.depths[::len(receiver.depths)//5]:  # Show 5 receiver depths
        ax.axhline(rd, color='g', linestyle=':', linewidth=1, alpha=0.5)
    ax.axhline(receiver.depths[0], color='g', linestyle=':', linewidth=1, label='Receivers')
    ax.invert_yaxis()
    ax.set_xlabel('Sound Speed (m/s)', fontweight='bold')
    ax.set_ylabel('Depth (m)', fontweight='bold')
    ax.set_title('Sound Speed Profile', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: Bathymetry Profile
    ax = axes[1]
    if env.bathymetry is not None and len(env.bathymetry) > 1:
        ax.plot(env.bathymetry[:,0]/1000, env.bathymetry[:,1], 'k-', linewidth=2)
        ax.fill_between(env.bathymetry[:,0]/1000, env.bathymetry[:,1],
                        np.max(env.bathymetry[:,1])*1.2, color='#8B4513', alpha=0.5)
    else:
        ax.axhline(env.depth, color='k', linewidth=2, label='Flat bottom')
        ax.fill_between([0, np.max(receiver.ranges)/1000], env.depth, env.depth*1.2,
                        color='#8B4513', alpha=0.5)

    ax.plot(0, source.depth[0], 'r*', markersize=20, label='Source', zorder=10)
    ax.invert_yaxis()
    ax.set_xlabel('Range (km)', fontweight='bold')
    ax.set_ylabel('Depth (m)', fontweight='bold')
    ax.set_title('Bathymetry Profile', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, np.max(receiver.ranges)/1000)

    # Plot 3: Receiver Grid
    ax = axes[2]
    # Create 2D receiver grid visualization
    R, Z = np.meshgrid(receiver.ranges/1000, receiver.depths)
    ax.scatter(R.flatten()[::10], Z.flatten()[::10], c='green', s=5, alpha=0.3, label='Receiver grid')
    ax.plot(0, source.depth[0], 'r*', markersize=20, label='Source')

    if env.bathymetry is not None and len(env.bathymetry) > 1:
        ax.plot(env.bathymetry[:,0]/1000, env.bathymetry[:,1], 'k-', linewidth=2, label='Seafloor')
    else:
        ax.axhline(env.depth, color='k', linewidth=2, label='Flat bottom')

    ax.invert_yaxis()
    ax.set_xlabel('Range (km)', fontweight='bold')
    ax.set_ylabel('Depth (m)', fontweight='bold')
    ax.set_title(f'Receiver Grid ({len(receiver.depths)}×{len(receiver.ranges)})', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig

def plot_tl_field_comparison(results: Dict, env, source, vmin=None, vmax=None):
    """
    Plot TL fields from multiple models in harmonized format.

    Parameters
    ----------
    results : dict
        Dictionary of {model_name: Field} results
    env : Environment
    source : Source
    vmin, vmax : float, optional
        TL colorbar limits. If None, computed from data (10th and 90th percentiles)
    """
    # Filter out None results (models that were skipped)
    results = {k: v for k, v in results.items() if v is not None}

    if len(results) == 0:
        return None

    # Compute dynamic colorbar limits if not provided
    if vmin is None or vmax is None:
        all_tl_values = []
        for result in results.values():
            all_tl_values.extend(result.data.flatten())
        all_tl_values = np.array(all_tl_values)
        # Filter out NaN and Inf values before computing percentiles
        all_tl_values = all_tl_values[np.isfinite(all_tl_values)]
        # Filter out extreme outliers (Bellhop sentinel value is 740 dB)
        # Cap at 200 dB which is already extremely high for TL
        all_tl_values = all_tl_values[all_tl_values < 200]

        if len(all_tl_values) == 0:
            # Fallback if all values are NaN/Inf or extreme
            vmin, vmax = 40, 100
        else:
            # Use percentiles to avoid outliers
            if vmin is None:
                vmin = np.percentile(all_tl_values, 2)  # More aggressive to avoid outliers
            if vmax is None:
                vmax = np.percentile(all_tl_values, 98)  # More aggressive to avoid outliers

    n_models = len(results)
    ncols = min(3, n_models)
    nrows = int(np.ceil(n_models / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx]

        # Mask TL data below seafloor (if applicable)
        tl_data = result.data.copy()
        if env.bathymetry is not None and len(env.bathymetry) > 0:
            # Interpolate bathymetry depth for each range
            from scipy.interpolate import interp1d
            bathy_interp = interp1d(env.bathymetry[:, 0], env.bathymetry[:, 1],
                                   kind='linear', bounds_error=False,
                                   fill_value=(env.bathymetry[0, 1], env.bathymetry[-1, 1]))
            seafloor_depths = bathy_interp(result.ranges)

            # Mask data below seafloor
            for i_range in range(len(result.ranges)):
                depth_mask = result.depths > seafloor_depths[i_range]
                tl_data[depth_mask, i_range] = np.nan
        else:
            # Flat bottom - mask below env.depth
            depth_mask = result.depths > env.depth
            tl_data[depth_mask, :] = np.nan

        # Plot TL field (with lower zorder so bathymetry appears on top)
        im = ax.pcolormesh(result.ranges/1000, result.depths, tl_data,
                          cmap='viridis', vmin=vmin, vmax=vmax, shading='auto', zorder=1)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, label='Transmission Loss (dB)')
        cbar.ax.tick_params(labelsize=9)

        # Set proper axis limits BEFORE overlay
        range_min_km = np.min(result.ranges) / 1000
        range_max_km = np.max(result.ranges) / 1000
        depth_min = np.min(result.depths)
        depth_max = np.max(result.depths)

        ax.set_xlim(range_min_km, range_max_km)
        ax.set_ylim(depth_max, depth_min)  # Inverted for depth

        # Overlay bathymetry (AFTER setting limits)
        plot_bathymetry_overlay(ax, env, result, range_min_km, range_max_km)

        # Source marker - at minimum range
        ax.plot(range_min_km, source.depth[0], 'r*', markersize=15, label='Source', zorder=12)

        # Formatting
        ax.set_xlabel('Range (km)', fontweight='bold')
        ax.set_ylabel('Depth (m)', fontweight='bold')
        ax.set_title(f'{name} - Transmission Loss', fontweight='bold', fontsize=12)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    return fig

def plot_tl_comparison_curves(results: Dict, source_depth: float):
    """
    Plot TL vs range and TL vs depth comparison curves.

    Returns figure with two subplots.
    """
    # Filter out None results (models that were skipped)
    results = {k: v for k, v in results.items() if v is not None}

    if len(results) == 0:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # TL vs Range at source depth
    ax = axes[0]
    for name, result in results.items():
        depth_idx = np.argmin(np.abs(result.depths - source_depth))
        tl_vs_range = result.data[depth_idx, :]
        ax.plot(result.ranges/1000, tl_vs_range, linewidth=2.5,
               label=name, color=COLORS.get(name, None), alpha=0.8)

    ax.set_xlabel('Range (km)', fontweight='bold')
    ax.set_ylabel('Transmission Loss (dB)', fontweight='bold')
    ax.set_title(f'TL vs Range at {source_depth:.0f}m Depth', fontweight='bold', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # TL vs Depth at mid-range
    ax = axes[1]
    for name, result in results.items():
        mid_range_km = np.median(result.ranges) / 1000
        range_idx = np.argmin(np.abs(result.ranges/1000 - mid_range_km))
        tl_vs_depth = result.data[:, range_idx]
        ax.plot(tl_vs_depth, result.depths, linewidth=2.5,
               label=name, color=COLORS.get(name, None), alpha=0.8)

    ax.invert_yaxis()
    ax.set_xlabel('Transmission Loss (dB)', fontweight='bold')
    ax.set_ylabel('Depth (m)', fontweight='bold')
    ax.set_title(f'TL vs Depth at {mid_range_km:.1f}km Range', fontweight='bold', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def plot_model_statistics(results: Dict, source_depth: float):
    """
    Plot statistical comparison and RMS error matrix.
    """
    # Filter out None results (models that were skipped)
    results = {k: v for k, v in results.items() if v is not None}

    if len(results) == 0:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart of mean/std TL
    ax = axes[0]
    model_names = list(results.keys())
    stats = []
    for name, result in results.items():
        depth_idx = np.argmin(np.abs(result.depths - source_depth))
        tl_vs_range = result.data[depth_idx, :]
        stats.append([np.mean(tl_vs_range), np.std(tl_vs_range)])

    stats = np.array(stats)
    x = np.arange(len(model_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, stats[:,0], width, label='Mean TL', alpha=0.8)
    bars2 = ax.bar(x + width/2, stats[:,1], width, label='Std TL', alpha=0.8)

    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Transmission Loss (dB)', fontweight='bold')
    ax.set_title('Statistical Comparison', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # RMS error matrix
    if len(results) >= 2:
        ax = axes[1]
        n_models = len(model_names)
        rms_matrix = np.zeros((n_models, n_models))

        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    result_i = results[model_names[i]]
                    result_j = results[model_names[j]]

                    depth_idx_i = np.argmin(np.abs(result_i.depths - source_depth))
                    depth_idx_j = np.argmin(np.abs(result_j.depths - source_depth))

                    tl_i = result_i.data[depth_idx_i, :]
                    tl_j = result_j.data[depth_idx_j, :]

                    # Interpolate to common grid
                    if len(tl_i) != len(tl_j):
                        ranges_i = result_i.ranges
                        ranges_j = result_j.ranges
                        if len(ranges_i) < len(ranges_j):
                            tl_j = np.interp(ranges_i, ranges_j, tl_j)
                        else:
                            tl_i = np.interp(ranges_j, ranges_i, tl_i)

                    rms_matrix[i,j] = np.sqrt(np.mean((tl_i - tl_j)**2))

        # Compute colormap normalization from data
        rms_max = np.max(rms_matrix)
        vmax_rms = max(10, np.percentile(rms_matrix[rms_matrix > 0], 95)) if rms_max > 0 else 15
        im = ax.imshow(rms_matrix, cmap='RdYlGn_r', vmin=0, vmax=vmax_rms, interpolation='none')
        cbar = plt.colorbar(im, ax=ax, label='RMS Error (dB)')

        ax.set_xticks(range(n_models))
        ax.set_yticks(range(n_models))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_yticklabels(model_names)
        ax.set_title('Model Agreement (RMS Error)', fontweight='bold', fontsize=12)

        # Add text annotations
        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    text_color = 'white' if rms_matrix[i,j] > (vmax_rms/2) else 'black'
                    ax.text(j, i, f'{rms_matrix[i,j]:.1f}',
                           ha='center', va='center', color=text_color, fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'Need at least 2 models\nfor RMS comparison',
                    ha='center', va='center', transform=axes[1].transAxes, fontsize=12)
        axes[1].axis('off')

    plt.tight_layout()
    return fig

def create_example_report(example_num: int, title: str, description: str,
                         env, source, receiver, results: Dict,
                         output_prefix: str, output_dir: str = "output"):
    """
    Create complete report for an example with all plots.

    Generates:
    - Configuration plot (SSP, bathymetry, receiver grid)
    - TL field comparison
    - TL curve comparison
    - Statistics and RMS errors

    All plots saved to output_dir (default: "output/")
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    apply_plot_style()

    # Print header
    print("=" * 80)
    print(f"Example {example_num}: {title}")
    print("=" * 80)
    print(f"\n{description}\n")

    # Environment info
    print(f"Environment: {env.name}")
    print(f"  Depth: {env.depth}m")
    print(f"  SSP type: {env.ssp_type}")
    if env.is_range_dependent:
        print(f"  Range-dependent: YES (bathymetry points: {len(env.bathymetry)})")
    else:
        print(f"  Range-dependent: NO")

    # Source info
    print(f"\nSource:")
    print(f"  Depth: {source.depth[0]}m")
    print(f"  Frequency: {source.frequency[0]}Hz")

    # Receiver info
    print(f"\nReceivers:")
    print(f"  Depths: {len(receiver.depths)} points ({receiver.depths[0]:.1f}m to {receiver.depths[-1]:.1f}m)")
    print(f"  Ranges: {len(receiver.ranges)} points ({receiver.ranges[0]/1000:.1f}km to {receiver.ranges[-1]/1000:.1f}km)")

    # Model results
    print(f"\nModels tested: {len(results)}")
    for name, result in results.items():
        if result is not None:
            print(f"  {name:15s}: TL range {result.data.min():.1f} to {result.data.max():.1f} dB")
        else:
            print(f"  {name:15s}: Skipped")

    # Generate plots
    print(f"\nGenerating plots...")

    # Plot 1: Configuration
    fig = plot_source_receiver_config(env, source, receiver)
    fig.savefig(f'{output_dir}/{output_prefix}_config.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {output_dir}/{output_prefix}_config.png")

    # Plot 2: TL fields
    fig = plot_tl_field_comparison(results, env, source)
    fig.savefig(f'{output_dir}/{output_prefix}_fields.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {output_dir}/{output_prefix}_fields.png")

    # Plot 3: TL curves
    fig = plot_tl_comparison_curves(results, source.depth[0])
    fig.savefig(f'{output_dir}/{output_prefix}_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {output_dir}/{output_prefix}_curves.png")

    # Plot 4: Statistics
    fig = plot_model_statistics(results, source.depth[0])
    fig.savefig(f'{output_dir}/{output_prefix}_stats.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {output_dir}/{output_prefix}_stats.png")

    print(f"\nExample {example_num} complete!")
    print("=" * 80 + "\n")
