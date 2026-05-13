"""Example-side report scaffolding.

Two helpers:

* :func:`plot_model_statistics` — bar chart (mean / std TL per model) +
  RMS-error matrix across models. Unique to the report flow; not part
  of the canonical :mod:`uacpy.visualization` surface because it consumes
  a *dict of models* and operates on already-sliced TL profiles.
* :func:`create_example_report` — one-shot orchestrator that runs the
  four canonical plotters (:func:`plot_environment`,
  :func:`compare_models`, :func:`compare`, plus
  :func:`plot_model_statistics` above) and saves them under a common
  prefix.

Everything else lives in :mod:`uacpy.visualization` — TL heatmaps,
bathymetry overlays, range / depth cuts, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict

from uacpy.visualization import (
    plot_environment, compare_models, compare,
)

# Default output directory: next to this file, so examples drop plots under
# uacpy/uacpy/examples/output/ regardless of the caller's cwd.
DEFAULT_OUTPUT_DIR = Path(__file__).parent / 'output'


def plot_model_statistics(results: Dict, source_depth: float):
    """Bar chart (mean ± std TL per model) + pairwise RMS-error matrix.

    ``results`` is a ``{model_name: Field}`` dict; ``None`` entries
    (skipped models) are filtered out. The TL is slice-extracted at
    ``source_depth`` then reduced with NaN-aware stats (RAM masks
    below-seafloor cells with NaN).

    The RMS matrix masks its diagonal so it gets the cmap's ``set_bad``
    deep-green ``cmap(0.0)`` colour — i.e. a clean "zero error" tile,
    not a white off-scale square.
    """
    results = {k: v for k, v in results.items() if v is not None}
    if not results:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Bar chart of mean / std TL ────────────────────────────────────
    ax = axes[0]
    model_names = list(results.keys())
    stats = []
    for name, result in results.items():
        tl = np.asarray(result.at(depth=source_depth).tl)
        stats.append([np.nanmean(tl), np.nanstd(tl)])
    stats = np.array(stats)
    x = np.arange(len(model_names))
    width = 0.35
    ax.bar(x - width / 2, stats[:, 0], width, label='Mean TL', alpha=0.8)
    ax.bar(x + width / 2, stats[:, 1], width, label='Std TL', alpha=0.8)
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Transmission Loss (dB)', fontweight='bold')
    ax.set_title('Statistical Comparison', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # ── Pairwise RMS-error matrix ─────────────────────────────────────
    if len(results) >= 2:
        ax = axes[1]
        n = len(model_names)
        rms_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                ri = results[model_names[i]]
                rj = results[model_names[j]]
                tl_i = np.asarray(ri.at(depth=source_depth).tl)
                tl_j = np.asarray(rj.at(depth=source_depth).tl)
                if len(tl_i) != len(tl_j):
                    if len(ri.ranges) < len(rj.ranges):
                        tl_j = np.interp(ri.ranges, rj.ranges, tl_j)
                    else:
                        tl_i = np.interp(rj.ranges, ri.ranges, tl_i)
                diff = tl_i - tl_j
                finite = np.isfinite(diff)
                rms_matrix[i, j] = (
                    np.sqrt(np.mean(diff[finite] ** 2))
                    if finite.any() else np.nan
                )
        rms_max = np.max(rms_matrix)
        vmax = (max(10, np.percentile(rms_matrix[rms_matrix > 0], 95))
                if rms_max > 0 else 15)
        display = np.ma.array(rms_matrix, mask=np.eye(n, dtype=bool))
        cmap = plt.get_cmap('RdYlGn_r').copy()
        cmap.set_bad(color=cmap(0.0))
        im = ax.imshow(display, cmap=cmap, vmin=0, vmax=vmax,
                       interpolation='none')
        plt.colorbar(im, ax=ax, label='RMS Error (dB)')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_yticklabels(model_names)
        ax.set_title('Model Agreement (RMS Error)',
                     fontweight='bold', fontsize=12)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                colour = 'white' if rms_matrix[i, j] > vmax / 2 else 'black'
                ax.text(j, i, f'{rms_matrix[i, j]:.1f}',
                        ha='center', va='center',
                        color=colour, fontweight='bold')
    else:
        axes[1].text(
            0.5, 0.5, 'Need at least 2 models\nfor RMS comparison',
            ha='center', va='center',
            transform=axes[1].transAxes, fontsize=12,
        )
        axes[1].axis('off')

    plt.tight_layout()
    return fig


def create_example_report(example_num: int, title: str, description: str,
                          env, source, receiver, results: Dict,
                          output_prefix: str, output_dir=None):
    """Run the four canonical plots for an example and save them.

    Generates and persists, under ``output_dir`` (default
    ``<examples>/output/``):

    * ``<prefix>_environment.png`` — :func:`plot_environment` with source
      and receiver markers.
    * ``<prefix>_fields.png`` — :func:`compare_models` heatmap grid.
    * ``<prefix>_curves.png`` — TL-vs-range and TL-vs-depth overlays via
      two :func:`compare` calls.
    * ``<prefix>_stats.png`` — :func:`plot_model_statistics` bar +
      RMS-matrix.
    """
    output_dir = (Path(output_dir) if output_dir is not None
                  else DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Example {example_num}: {title}")
    print("=" * 80)
    print(f"\n{description}\n")
    print(f"Environment: {env.name}")
    print(f"  Depth: {env.depth}m")
    print(f"  SSP shape: {env.ssp.shape}")
    if env.is_range_dependent:
        print(f"  Range-dependent: YES (bathymetry points: {len(env.bathymetry)})")
    else:
        print("  Range-dependent: NO")
    print("\nSource:")
    print(f"  Depth: {source.depths[0]}m")
    print(f"  Frequency: {source.frequencies[0]}Hz")
    print("\nReceivers:")
    print(f"  Depths: {len(receiver.depths)} points "
          f"({receiver.depths[0]:.1f}m to {receiver.depths[-1]:.1f}m)")
    r0_km = receiver.ranges[0] / 1000
    r1_km = receiver.ranges[-1] / 1000
    print(f"  Ranges: {len(receiver.ranges)} points "
          f"({r0_km:.1f}km to {r1_km:.1f}km)")
    print(f"\nModels tested: {len(results)}")
    for name, result in results.items():
        if result is not None:
            print(f"  {name:15s}: TL range "
                  f"{np.nanmin(result.tl):.1f} to {np.nanmax(result.tl):.1f} dB")
        else:
            print(f"  {name:15s}: Skipped")

    print("\nGenerating plots...")

    # Plot 1: Environment overview
    fig, _ = plot_environment(env, source=source, receiver=receiver)
    env_path = output_dir / f'{output_prefix}_environment.png'
    fig.savefig(env_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {env_path}")

    # Plot 2: TL field comparison (drop None entries first)
    live = {k: v for k, v in results.items() if v is not None}
    if live:
        fig, _ = compare_models(live, env=env)
        fields_path = output_dir / f'{output_prefix}_fields.png'
        fig.savefig(fields_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ {fields_path}")

        # Plot 3: TL vs range and TL vs depth (canonical compare overlays)
        z = float(source.depths[0])
        r_mid = float(np.median(receiver.ranges))
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        compare(
            [r.at(depth=z) for r in live.values()],
            labels=list(live), ax=axes[0],
            title=f'TL vs Range at {z:.0f} m depth',
        )
        compare(
            [r.at(range=r_mid) for r in live.values()],
            labels=list(live), ax=axes[1],
            title=f'TL vs Depth at {r_mid / 1000:.1f} km range',
        )
        fig.tight_layout()
        curves_path = output_dir / f'{output_prefix}_curves.png'
        fig.savefig(curves_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ {curves_path}")

    # Plot 4: Statistics
    fig = plot_model_statistics(results, source.depths[0])
    if fig is not None:
        stats_path = output_dir / f'{output_prefix}_stats.png'
        fig.savefig(stats_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ {stats_path}")

    print(f"\nExample {example_num} complete!")
    print("=" * 80 + "\n")
