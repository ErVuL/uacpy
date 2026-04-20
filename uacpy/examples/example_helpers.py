"""
Example helper functions for UACPY examples

⚠️ IMPORTANT: These are examples-only utilities for UACPY demonstration purposes.
For production code, please use the official API from uacpy.visualization, particularly
the quickplot() function and plot_* functions which provide the full feature set.

Provides common utilities for examples 01-10 to ensure consistency
and reduce code duplication.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional

# Import from main visualization module
from uacpy.visualization import (
    plot_transmission_loss,
    plot_environment_advanced,
    plot_comparison_curves,
    plot_model_comparison_matrix
)


def create_example_report(
    example_num: int,
    title: str,
    description: str,
    env,
    source,
    receiver,
    results: Dict,
    output_prefix: str
):
    """
    Create complete report for an example with all plots

    Parameters
    ----------
    example_num : int
        Example number (1-10)
    title : str
        Example title
    description : str
        Description of what the example demonstrates
    env : Environment
        Environment object
    source : Source
        Source object
    receiver : Receiver
        Receiver object
    results : dict
        Dictionary of {model_name: Field} results
    output_prefix : str
        Prefix for output files (e.g., 'example_01')

    Generates
    ----------
    - Configuration plot (SSP, bathymetry, receiver grid)
    - TL field comparison
    - TL curve comparison
    - Model comparison matrix
    """
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
        print(f"  Range-dependent: YES")
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
            print(f"  {name:15s}: SKIPPED (returned None)")

    # Generate plots
    print(f"\nGenerating plots...")

    # Ensure output directory exists
    import os
    os.makedirs('output', exist_ok=True)

    # Plot 1: Environment overview
    result = plot_environment_advanced(env, source, receiver)
    if isinstance(result, tuple):
        fig, _ = result  # Unpack if it returns (fig, axes)
    else:
        fig = result
    fig.savefig(f'output/{output_prefix}_environment.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ output/{output_prefix}_environment.png")

    # Plot 2: TL fields
    if len(results) > 0:
        n_models = len(results)
        ncols = min(2, n_models)
        nrows = int(np.ceil(n_models / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(8*ncols, 6*nrows))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, (name, field) in enumerate(results.items()):
            if idx < len(axes):
                plot_transmission_loss(field, env, ax=axes[idx], vmin=40, vmax=100)
                axes[idx].set_title(f'{name} - Transmission Loss', fontweight='bold')

        # Hide unused subplots
        for idx in range(len(results), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        fig.savefig(f'output/{output_prefix}_fields.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ output/{output_prefix}_fields.png")

    # Plot 3: Comparison curves (if multiple models)
    if len(results) >= 2:
        fig, axes = plot_comparison_curves(results, source_depth=source.depth[0])
        fig.savefig(f'output/{output_prefix}_curves.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ output/{output_prefix}_curves.png")

    # Plot 4: Comparison matrix (if multiple models)
    if len(results) >= 2:
        fig, ax = plot_model_comparison_matrix(results, comparison_metric='rms', source_depth=source.depth[0])
        fig.savefig(f'output/{output_prefix}_matrix.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ output/{output_prefix}_matrix.png")

    print(f"\nExample {example_num} complete!")
    print("=" * 80 + "\n")


def print_model_statistics(results: Dict, source_depth: float):
    """
    Print statistical comparison of models

    Parameters
    ----------
    results : dict
        Dictionary of {model_name: Field} results
    source_depth : float
        Depth for comparison (meters)
    """
    print("\n" + "=" * 80)
    print("MODEL STATISTICS")
    print("=" * 80)

    for name, field in results.items():
        # Find closest depth
        depth_idx = np.argmin(np.abs(field.depths - source_depth))
        tl_at_depth = field.data[depth_idx, :]

        print(f"\n{name}:")
        print(f"  Mean TL: {np.mean(tl_at_depth):.2f} dB")
        print(f"  Std TL:  {np.std(tl_at_depth):.2f} dB")
        print(f"  Min TL:  {np.min(tl_at_depth):.2f} dB")
        print(f"  Max TL:  {np.max(tl_at_depth):.2f} dB")

    # Pairwise RMS errors
    if len(results) >= 2:
        print("\nPairwise RMS Errors (dB):")
        model_names = list(results.keys())
        for i, name_i in enumerate(model_names):
            for j, name_j in enumerate(model_names[i+1:], start=i+1):
                field_i = results[name_i]
                field_j = results[name_j]

                depth_idx_i = np.argmin(np.abs(field_i.depths - source_depth))
                depth_idx_j = np.argmin(np.abs(field_j.depths - source_depth))

                tl_i = field_i.data[depth_idx_i, :]
                tl_j = field_j.data[depth_idx_j, :]

                # Interpolate if needed
                if not np.array_equal(field_i.ranges, field_j.ranges):
                    tl_j = np.interp(field_i.ranges, field_j.ranges, tl_j)

                rms = np.sqrt(np.mean((tl_i - tl_j)**2))
                print(f"  {name_i} vs {name_j}: {rms:.2f} dB")


__all__ = [
    'create_example_report',
    'print_model_statistics',
]
