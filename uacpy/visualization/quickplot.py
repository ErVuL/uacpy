"""
Quick-plot convenience functions for rapid visualization

These functions provide the absolute simplest way to visualize acoustic results.
They handle all defaults automatically and require minimal parameters.

For more control, use the full plotting API in uacpy.visualization.plots
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union, Dict

from uacpy.core.field import Field
from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.visualization import plots


def quick_tl(field: Field, env: Optional[Environment] = None, save: Optional[str] = None):
    """
    Quickest way to plot transmission loss

    Parameters
    ----------
    field : Field
        TL field from any model
    env : Environment, optional
        Environment (for bathymetry overlay)
    save : str, optional
        Filename to save figure (e.g., 'result.png')

    Examples
    --------
    >>> result = bellhop.compute_tl(env, source, max_range=10000)
    >>> quick_tl(result, env)

    >>> # Save directly
    >>> quick_tl(result, env, save='tl_field.png')
    """
    fig, ax = field.plot(env=env, vmin=30, vmax=100)

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {save}")
    else:
        plt.show()

    return fig, ax


def quick_compare(results: Dict[str, Field], env: Optional[Environment] = None,
                  save: Optional[str] = None):
    """
    Quickest way to compare multiple models

    Parameters
    ----------
    results : dict
        Dictionary with model names as keys, Field objects as values
    env : Environment, optional
        Environment reference
    save : str, optional
        Filename to save figure

    Examples
    --------
    >>> results = {
    ...     'Bellhop': bellhop.compute_tl(env, source, max_range=10000),
    ...     'RAM': ram.compute_tl(env, source, max_range=10000),
    ... }
    >>> quick_compare(results, env)
    """
    fig, axes = Field.plot_comparison(results, env=env)

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {save}")
    else:
        plt.show()

    return fig, axes


def quick_rays(field: Field, env: Optional[Environment] = None,
               n_rays: int = 50, save: Optional[str] = None):
    """
    Quickest way to plot ray paths

    Parameters
    ----------
    field : Field
        Ray field from Bellhop
    env : Environment, optional
        Environment (for bathymetry)
    n_rays : int, optional
        Maximum number of rays to plot (default: 50)
    save : str, optional
        Filename to save figure

    Examples
    --------
    >>> rays = bellhop.compute_rays(env, source, max_range=5000)
    >>> quick_rays(rays, env, n_rays=100)
    """
    fig, ax = field.plot(env=env, max_rays=n_rays)

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {save}")
    else:
        plt.show()

    return fig, ax


def quick_env(env: Environment, source: Optional[Source] = None,
              save: Optional[str] = None):
    """
    Quickest way to visualize environment

    Parameters
    ----------
    env : Environment
        Environment to plot
    source : Source, optional
        Source to show position
    save : str, optional
        Filename to save figure

    Examples
    --------
    >>> quick_env(env, source)
    """
    fig, axes = plots.plot_environment(env, source)

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {save}")
    else:
        plt.show()

    return fig, axes


def quick_modes(field: Field, n_modes: int = 6, save: Optional[str] = None):
    """
    Quickest way to plot normal modes

    Parameters
    ----------
    field : Field
        Mode field from Kraken
    n_modes : int, optional
        Number of modes to show (default: 6)
    save : str, optional
        Filename to save figure

    Examples
    --------
    >>> modes = kraken.compute_modes(env, source, n_modes=20)
    >>> quick_modes(modes, n_modes=10)
    """
    fig, axes = field.plot(n_modes=n_modes)

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {save}")
    else:
        plt.show()

    return fig, axes


def quick_cut(field: Field, depth: Optional[float] = None,
              range_m: Optional[float] = None, save: Optional[str] = None):
    """
    Quickest way to plot range or depth cut

    Parameters
    ----------
    field : Field
        TL field
    depth : float, optional
        Depth for range cut (TL vs range at this depth)
    range_m : float, optional
        Range for depth cut (TL vs depth at this range)
    save : str, optional
        Filename to save figure

    Examples
    --------
    >>> # Range cut at 50m depth
    >>> quick_cut(result, depth=50)

    >>> # Depth cut at 5km range
    >>> quick_cut(result, range_m=5000)

    Notes
    -----
    Provide either depth or range_m, not both
    """
    if depth is not None and range_m is None:
        fig, ax = plots.plot_range_cut(field, depth)
    elif range_m is not None and depth is None:
        fig, ax = plots.plot_depth_cut(field, range_m)
    elif depth is not None and range_m is not None:
        raise ValueError("Provide either depth or range_m, not both")
    else:
        # Auto-select middle depth
        depth = field.depths[len(field.depths) // 2]
        fig, ax = plots.plot_range_cut(field, depth)
        print(f"Auto-selected depth: {depth}m")

    if save:
        plt.savefig(save, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {save}")
    else:
        plt.show()

    return fig, ax


def quick_analysis(field: Field, env: Optional[Environment] = None,
                   save_prefix: Optional[str] = None):
    """
    Comprehensive quick analysis - generates multiple plots

    Creates a comprehensive figure with:
    - TL field
    - Range cut at mid-depth
    - Depth cut at mid-range
    - TL statistics

    Parameters
    ----------
    field : Field
        TL field to analyze
    env : Environment, optional
        Environment reference
    save_prefix : str, optional
        Prefix for saved files (e.g., 'analysis' -> 'analysis_summary.png')

    Examples
    --------
    >>> result = bellhop.compute_tl(env, source, max_range=10000)
    >>> quick_analysis(result, env)

    >>> # Save all plots
    >>> quick_analysis(result, env, save_prefix='shallow_water')
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 1. TL field (top, spanning both columns)
    ax1 = fig.add_subplot(gs[0, :])
    plots.plot_transmission_loss(field, env, ax=ax1, vmin=30, vmax=100)
    ax1.set_title('Transmission Loss Field', fontsize=14, fontweight='bold')

    # 2. Range cut at mid-depth
    ax2 = fig.add_subplot(gs[1, 0])
    mid_depth = field.depths[len(field.depths) // 2]
    plots.plot_range_cut(field, mid_depth, ax=ax2)
    ax2.set_title(f'Range Cut @ {mid_depth:.1f}m Depth', fontsize=12)

    # 3. Depth cut at mid-range
    ax3 = fig.add_subplot(gs[1, 1])
    mid_range = field.ranges[len(field.ranges) // 2]
    plots.plot_depth_cut(field, mid_range, ax=ax3)
    ax3.set_title(f'Depth Cut @ {mid_range/1000:.1f}km Range', fontsize=12)

    fig.suptitle('Acoustic Field Analysis', fontsize=16, fontweight='bold')

    if save_prefix:
        filename = f'{save_prefix}_analysis.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Saved comprehensive analysis to {filename}")
    else:
        plt.show()

    return fig


def quick_report(results: Dict[str, Field], env: Environment,
                 save: str = 'acoustic_report.png'):
    """
    Generate comprehensive comparison report for multiple models

    Creates a publication-ready comparison figure with:
    - Side-by-side TL fields
    - Range cuts at multiple depths
    - Statistical comparison

    Parameters
    ----------
    results : dict
        Dictionary with model names as keys, Field objects as values
    env : Environment
        Environment reference
    save : str, optional
        Filename for report (default: 'acoustic_report.png')

    Examples
    --------
    >>> results = {
    ...     'Bellhop': bellhop.compute_tl(env, source, max_range=10000),
    ...     'RAM': ram.compute_tl(env, source, max_range=10000),
    ...     'KrakenField': krakenfield.compute_tl(env, source, max_range=10000),
    ... }
    >>> quick_report(results, env, save='model_comparison_report.png')
    """
    n_models = len(results)

    # Create comprehensive figure
    fig = plt.figure(figsize=(6 * n_models, 12))
    gs = fig.add_gridspec(3, n_models, hspace=0.3, wspace=0.3)

    # Row 1: TL fields
    for idx, (model_name, field) in enumerate(results.items()):
        ax = fig.add_subplot(gs[0, idx])
        plots.plot_transmission_loss(field, env, ax=ax, vmin=30, vmax=100, show_bathymetry=False)
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')

    # Row 2: Range cuts at 1/3 depth
    ax_row2 = fig.add_subplot(gs[1, :])
    depth = list(results.values())[0].depths[len(list(results.values())[0].depths) // 3]
    for model_name, field in results.items():
        r_idx = np.argmin(np.abs(field.depths - depth))
        ranges_km = field.ranges / 1000.0
        tl_slice = field.data[r_idx, :]
        ax_row2.plot(ranges_km, tl_slice, label=model_name, linewidth=2)
    ax_row2.set_xlabel('Range (km)', fontsize=11)
    ax_row2.set_ylabel('TL (dB)', fontsize=11)
    ax_row2.set_title(f'Range Cut @ {depth:.1f}m Depth', fontsize=12)
    ax_row2.legend()
    ax_row2.grid(True, alpha=0.3)
    ax_row2.invert_yaxis()

    # Row 3: Statistics
    ax_row3 = fig.add_subplot(gs[2, :])
    model_names = list(results.keys())
    stats = []
    for field in results.values():
        data = field.data[np.isfinite(field.data)]
        stats.append([np.mean(data), np.std(data), np.min(data), np.max(data)])
    stats = np.array(stats)

    x = np.arange(len(model_names))
    width = 0.2
    ax_row3.bar(x - width, stats[:, 0], width, label='Mean TL')
    ax_row3.bar(x, stats[:, 2], width, label='Min TL')
    ax_row3.bar(x + width, stats[:, 3], width, label='Max TL')
    ax_row3.set_ylabel('TL (dB)', fontsize=11)
    ax_row3.set_title('Statistical Comparison', fontsize=12)
    ax_row3.set_xticks(x)
    ax_row3.set_xticklabels(model_names)
    ax_row3.legend()
    ax_row3.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Model Comparison Report', fontsize=16, fontweight='bold')

    plt.savefig(save, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comprehensive report to {save}")

    return fig


__all__ = [
    'quick_tl',
    'quick_compare',
    'quick_rays',
    'quick_env',
    'quick_modes',
    'quick_cut',
    'quick_analysis',
    'quick_report',
]
