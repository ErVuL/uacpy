"""
Example 25: Canonical SSP shapes and bottom-loss curves
=========================================================

Showcases the SSP factories on :class:`SoundSpeedProfile` and the
plane-wave bottom-loss helper:

1. Side-by-side plot of three canonical SSPs — isothermal, Munk, and a
   Mackenzie-derived T(z), S(z) profile.
2. Overlay of the fluid–fluid bottom-loss curves for the standard
   sediment + rock presets across grazing angle.

Output: ``output/example_25_canonical_presets.png``.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from uacpy.core.environment import SoundSpeedProfile  # noqa: E402
from uacpy.visualization.plots import plot_bottom_loss  # noqa: E402


def main():
    out_dir = Path(__file__).parent / 'output'
    out_dir.mkdir(exist_ok=True)

    print("\n" + "═" * 80)
    print("EXAMPLE 25: Canonical SSP shapes + bottom-loss curves")
    print("═" * 80)

    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    iso = SoundSpeedProfile.from_isothermal(c=1500.0, depth_max=4000.0)
    munk = SoundSpeedProfile.from_munk(depth_max=4000.0, n_points=81)
    z = np.linspace(0.0, 4000.0, 161)
    T = 4.0 + 14.0 * np.exp(-z / 400.0)
    S = 35.0 - 0.5 * np.exp(-z / 300.0)
    mackenzie = SoundSpeedProfile.from_temperature_salinity(z, T, S)

    for label, ssp, style in [
        ('isothermal',     iso,        '-'),
        ('Munk',           munk,       '--'),
        ('Mackenzie T,S',  mackenzie,  '-.'),
    ]:
        axes[0].plot(ssp.data[:, 0], ssp.depths, style, lw=2, label=label)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Sound speed (m/s)')
    axes[0].set_ylabel('Depth (m)')
    axes[0].set_title('Canonical SSP shapes')
    axes[0].legend(loc='lower left', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    plot_bottom_loss(
        ['clay', 'silt', 'sand', 'gravel', 'moraine',
         'chalk', 'limestone', 'basalt'],
        ax=axes[1],
    )

    fig.tight_layout()
    out = out_dir / 'example_25_canonical_presets.png'
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  ✓ Saved {out.name}")


if __name__ == '__main__':
    main()
