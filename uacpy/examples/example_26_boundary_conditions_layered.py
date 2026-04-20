"""
Example 26: Boundary Conditions - Top BC and Layered Bottoms
=============================================================

Demonstrates:
  1. Top boundary conditions: vacuum, rough sea surface, elastic (ice)
  2. Layered bottoms: single layer, multi-layer sediment
  3. Range-dependent layered bottoms
  4. Sea surface wave scattering (Pierson-Moskowitz spectrum)

Models exercised: RAM (bottom scenarios), Bellhop (surface scenarios)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import uacpy
from uacpy.core.environment import (
    BoundaryProperties, SedimentLayer, LayeredBottom,
    RangeDependentLayeredBottom, generate_sea_surface,
)


def make_source_receiver():
    """Standard source/receiver for all tests."""
    source = uacpy.Source(frequency=200, depth=25)
    receiver = uacpy.Receiver(
        depths=np.linspace(1, 95, 200),
        ranges=np.linspace(500, 10000, 500),
    )
    return source, receiver


# ── 1. Top Boundary Conditions ──────────────────────────────────────────────

def example_vacuum_surface():
    """Default vacuum (pressure-release) surface — Bellhop for comparison."""
    source, receiver = make_source_receiver()
    env = uacpy.Environment(
        name='vacuum_surface',
        depth=100,
        ssp_data=[(0, 1500), (100, 1500)],
        bottom=BoundaryProperties(
            acoustic_type='half-space', sound_speed=1600,
            density=1.5, attenuation=0.5,
        ),
    )
    return env, source, receiver


def example_rough_surface():
    """Rough sea surface from Pierson-Moskowitz spectrum (15 m/s wind)."""
    source, receiver = make_source_receiver()
    surface = generate_sea_surface(
        max_range_m=10000, wind_speed_ms=15, n_points=300, seed=42,
    )
    env = uacpy.Environment(
        name='rough_surface',
        depth=100,
        ssp_data=[(0, 1500), (100, 1500)],
        altimetry=surface,
        bottom=BoundaryProperties(
            acoustic_type='half-space', sound_speed=1600,
            density=1.5, attenuation=0.5,
        ),
    )
    return env, source, receiver


def example_ice_surface():
    """Elastic (ice) surface — half-space upper boundary.

    Note: ice cp (3500 m/s) >> water c (1480 m/s), so the critical angle
    is only ~25 deg. Most shallow-water modes propagate below critical and
    reflect perfectly, producing TL similar to vacuum. The main difference
    is in the interference pattern phase, not overall loss level.
    """
    source, receiver = make_source_receiver()
    ice = BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=3500.0,     # compressional speed in ice (m/s)
        shear_speed=1800.0,     # shear speed in ice (m/s)
        density=0.9,            # ice density (g/cm^3)
        attenuation=1.0,        # compressional attenuation (dB/wavelength)
        shear_attenuation=2.0,  # shear attenuation (dB/wavelength)
    )
    env = uacpy.Environment(
        name='ice_surface',
        depth=100,
        ssp_data=[(0, 1480), (100, 1480)],
        surface=ice,
        bottom=BoundaryProperties(
            acoustic_type='half-space', sound_speed=1600,
            density=1.5, attenuation=0.5,
        ),
    )
    return env, source, receiver


# ── 2. Layered Bottoms ──────────────────────────────────────────────────────

def example_single_layer_bottom():
    """Single sediment layer over a rock halfspace."""
    source, receiver = make_source_receiver()
    lb = LayeredBottom(
        layers=[
            SedimentLayer(thickness=10.0, sound_speed=1550, density=1.3,
                          attenuation=0.8),
        ],
        halfspace=BoundaryProperties(
            acoustic_type='half-space', sound_speed=2500,
            density=2.5, attenuation=0.1,
        ),
    )
    env = uacpy.Environment(
        name='single_layer_bottom', depth=100,
        ssp_data=[(0, 1500), (100, 1500)],
        bottom=lb,
    )
    return env, source, receiver


def example_multi_layer_bottom():
    """Sand over clay over rock — 3 sediment layers."""
    source, receiver = make_source_receiver()
    lb = LayeredBottom(
        layers=[
            SedimentLayer(thickness=5.0, sound_speed=1550, density=1.3,
                          attenuation=0.8),
            SedimentLayer(thickness=15.0, sound_speed=1650, density=1.7,
                          attenuation=0.4),
            SedimentLayer(thickness=30.0, sound_speed=1800, density=2.0,
                          attenuation=0.2),
        ],
        halfspace=BoundaryProperties(
            acoustic_type='half-space', sound_speed=2500,
            density=2.5, attenuation=0.1,
        ),
    )
    env = uacpy.Environment(
        name='multi_layer_bottom', depth=100,
        ssp_data=[(0, 1500), (100, 1500)],
        bottom=lb,
    )
    return env, source, receiver


# ── 3. Range-Dependent Layered Bottoms ───────────────────────────────────────

def example_range_dependent_layered():
    """Mud-over-clay nearshore, sand-over-rock offshore."""
    source, receiver = make_source_receiver()
    near = LayeredBottom(
        layers=[
            SedimentLayer(thickness=8.0, sound_speed=1500, density=1.2,
                          attenuation=1.0),
            SedimentLayer(thickness=20.0, sound_speed=1580, density=1.5,
                          attenuation=0.6),
        ],
        halfspace=BoundaryProperties(
            acoustic_type='half-space', sound_speed=1800,
            density=2.0, attenuation=0.2,
        ),
    )
    far = LayeredBottom(
        layers=[
            SedimentLayer(thickness=3.0, sound_speed=1650, density=1.8,
                          attenuation=0.3),
            SedimentLayer(thickness=10.0, sound_speed=1750, density=2.0,
                          attenuation=0.2),
        ],
        halfspace=BoundaryProperties(
            acoustic_type='half-space', sound_speed=2500,
            density=2.5, attenuation=0.05,
        ),
    )
    rdl = RangeDependentLayeredBottom(
        ranges_km=np.array([0.0, 10.0]),
        depths=np.array([100.0, 100.0]),
        profiles=[near, far],
    )
    env = uacpy.Environment(
        name='rd_layered_bottom', depth=100,
        ssp_data=[(0, 1500), (100, 1500)],
        bottom=rdl,
    )
    return env, source, receiver


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from uacpy.models.ram import RAM
    from uacpy.models.bellhop import Bellhop
    from uacpy.visualization.plots import plot_transmission_loss

    # (label, setup_fn, model_class)
    scenarios = [
        ('Flat surface (Bellhop)',         example_vacuum_surface,          Bellhop),
        ('Rough sea, 15 m/s (Bellhop)',    example_rough_surface,           Bellhop),
        ('Ice surface (Bellhop)',          example_ice_surface,             Bellhop),
        ('Single-layer bottom (RAM)',      example_single_layer_bottom,     RAM),
        ('Multi-layer bottom (RAM)',       example_multi_layer_bottom,      RAM),
        ('Range-dep layered (RAM)',        example_range_dependent_layered, RAM),
    ]

    print("Example 26: Boundary Conditions - Top BC and Layered Bottoms")
    print("=" * 65)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes_flat = axes.flatten()

    # Run all scenarios first, collect fields
    fields = []
    envs_out = []
    for idx, (label, setup_fn, model_cls) in enumerate(scenarios):
        env, source, receiver = setup_fn()
        model = model_cls(verbose=False)
        try:
            field = model.run(env, source, receiver)
            tl = field.data
            print(f"  {label:40s}  TL: [{np.nanmin(tl):5.1f}, {np.nanmax(tl):5.1f}] dB")
            fields.append(field)
            envs_out.append(env)
        except Exception as e:
            print(f"  {label:40s}  ERROR: {e}")
            fields.append(None)
            envs_out.append(env)

    # Compute shared color limits across all fields
    all_tl = [f.data for f in fields if f is not None]
    if all_tl:
        vmin = max(30, np.nanpercentile(np.concatenate([a.ravel() for a in all_tl]), 5))
        vmax = min(140, np.nanpercentile(np.concatenate([a.ravel() for a in all_tl]), 95))
        vmin = 5 * round(vmin / 5)
        vmax = 5 * round(vmax / 5)
    else:
        vmin, vmax = 40, 100

    # Plot with shared color scale
    for idx, (label, _, _) in enumerate(scenarios):
        ax = axes_flat[idx]
        field = fields[idx]
        env = envs_out[idx]
        if field is not None:
            plot_transmission_loss(field, env, ax=ax, show_colorbar=True,
                                  vmin=vmin, vmax=vmax)
        else:
            ax.text(0.5, 0.5, 'ERROR', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, color='red')
        ax.set_title(label, fontsize=11)

    fig.suptitle('Example 26: Boundary Conditions — Surface and Bottom',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir = Path(__file__).parent / 'output'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'example_26_boundary_conditions.png'
    fig.savefig(out_path, dpi=150)
    print(f"\n  Figure saved to {out_path}")
    print("Done.")


if __name__ == '__main__':
    main()
