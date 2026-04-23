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
from pathlib import Path
from typing import Dict, Optional

from uacpy.visualization import (
    plot_transmission_loss,
    plot_environment_advanced,
    plot_comparison_curves,
    plot_model_comparison_matrix
)

# Resolve relative to this file so outputs always land next to the examples,
# regardless of the caller's current working directory.
OUTPUT_DIR = Path(__file__).parent / 'output'


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

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Plot 1: Environment overview
    result = plot_environment_advanced(env, source, receiver)
    if isinstance(result, tuple):
        fig, _ = result
    else:
        fig = result
    env_path = OUTPUT_DIR / f'{output_prefix}_environment.png'
    fig.savefig(env_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {env_path}")

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

        for idx in range(len(results), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        fields_path = OUTPUT_DIR / f'{output_prefix}_fields.png'
        fig.savefig(fields_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ {fields_path}")

    # Plot 3: Comparison curves (if multiple models)
    if len(results) >= 2:
        fig, axes = plot_comparison_curves(results, source_depth=source.depth[0])
        curves_path = OUTPUT_DIR / f'{output_prefix}_curves.png'
        fig.savefig(curves_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ {curves_path}")

    # Plot 4: Comparison matrix (if multiple models)
    if len(results) >= 2:
        fig, ax = plot_model_comparison_matrix(results, comparison_metric='rms', source_depth=source.depth[0])
        matrix_path = OUTPUT_DIR / f'{output_prefix}_matrix.png'
        fig.savefig(matrix_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ {matrix_path}")

    print(f"\nExample {example_num} complete!")
    print("=" * 80 + "\n")


__all__ = [
    'create_example_report',
]
