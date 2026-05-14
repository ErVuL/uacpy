"""
Example 18: Range-Dependent Bottom — Adiabatic vs Coupled Modes vs RAM
========================================================================

Compares transmission loss from KrakenField in adiabatic and coupled mode
theory, and RAM (parabolic equation) for a range-dependent scenario with:
  - Sloping bathymetry (100 m to 200 m over 20 km)
  - Range-dependent layered sediment
  - Range-dependent SSP (warmer nearshore, cooler offshore)

Two bottom cases are shown:
  1. Hard layered bottom (high impedance, low loss) — all models agree well
  2. Soft lossy layered bottom (low impedance, high loss) — adiabatic
     modes over-attenuate; coupled modes recover energy transfer and
     match the RAM reference more closely

This demonstrates the value of coupled mode theory (via AT's field.exe)
over the adiabatic approximation for range-dependent propagation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import uacpy  # noqa: E402
from uacpy.core.environment import (  # noqa: E402
    BoundaryProperties, SedimentLayer, LayeredBottom,
    RangeDependentLayeredBottom, SoundSpeedProfile,
)
from uacpy.models.ram import RAM  # noqa: E402
from uacpy.models.kraken import KrakenField  # noqa: E402


def _plot_tl_difference(a, b, env=None, *, ax=None, title=None,
                         vmin=-10.0, vmax=10.0, diff_vmax=None, **kw):
    """Plot TL(a) - TL(b) as a diverging-colourmap heatmap.

    ``diff_vmax`` is a symmetric range shortcut: ``vmin = -diff_vmax``,
    ``vmax = +diff_vmax``.
    """
    from uacpy import Field
    from uacpy.visualization import plot_field
    if diff_vmax is not None:
        vmin, vmax = -abs(diff_vmax), abs(diff_vmax)
    diff = Field(data=a.tl - b.tl, coords=dict(a.coords))
    return plot_field(
        diff, env=env, ax=ax, vmin=vmin, vmax=vmax,
        cmap='RdBu_r', title=title, **kw,
    )


def make_base_env(bottom):
    """Build the shared environment with given bottom type."""
    bathy_ranges_m = np.array([0, 5000, 10000, 15000, 20000.0])
    bathy_depths_m = np.array([100, 120, 150, 180, 200.0])
    bathymetry = np.column_stack([bathy_ranges_m, bathy_depths_m])

    ssp_depths = np.array([0, 50, 100, 150, 200.0])
    ssp_near = np.array([1510, 1505, 1500, 1500, 1500.0])
    ssp_far = np.array([1500, 1495, 1490, 1492, 1495.0])
    ssp_ranges_m = np.array([0.0, 20000.0])
    ssp_2d = np.column_stack([ssp_near, ssp_far])

    env = uacpy.Environment(
        name='rd_comparison',
        ssp=SoundSpeedProfile.from_2d(depths=ssp_depths, ranges=ssp_ranges_m, matrix=ssp_2d
                                      ),
        bathymetry=bathymetry,
        bottom=bottom,
    )
    return env, bathymetry


def make_hard_bottom():
    """Hard layered bottom: high impedance contrast, low attenuation."""
    near = LayeredBottom(
        layers=[
            SedimentLayer(thickness=8.0, sound_speed=1600, density=1.8,
                          attenuation=0.2),
            SedimentLayer(thickness=20.0, sound_speed=1700, density=2.0,
                          attenuation=0.1),
        ],
        halfspace=BoundaryProperties(
            acoustic_type='half-space', sound_speed=2500,
            density=2.5, attenuation=0.05,
        ),
    )
    far = LayeredBottom(
        layers=[
            SedimentLayer(thickness=3.0, sound_speed=1800, density=2.0,
                          attenuation=0.1),
        ],
        halfspace=BoundaryProperties(
            acoustic_type='half-space', sound_speed=3000,
            density=2.8, attenuation=0.02,
        ),
    )
    return RangeDependentLayeredBottom(
        ranges=np.array([0, 20000]),
        profiles=[near, far],
    )


def make_soft_bottom():
    """Soft lossy layered bottom: low impedance contrast, high attenuation."""
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
        ],
        halfspace=BoundaryProperties(
            acoustic_type='half-space', sound_speed=2500,
            density=2.5, attenuation=0.05,
        ),
    )
    return RangeDependentLayeredBottom(
        ranges=np.array([0, 20000]),
        profiles=[near, far],
    )


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from uacpy.visualization.plots import (
    plot_field,
    plot_environment,
)

    print("\n" + "═" * 80)
    print("EXAMPLE 18: Range-Dependent Bottom — Adiabatic vs Coupled vs RAM")
    print("═" * 80)

    source = uacpy.Source(frequencies=100, depths=30)
    receiver = uacpy.Receiver(
        depths=np.linspace(5, 195, 30),
        ranges=np.linspace(1000, 6000, 300),
    )

    np.array([0, 5000, 10000, 15000, 20000.0])
    bathy_depths_m = np.array([100, 120, 150, 180, 200.0])

    # ── Run both bottom cases with all three models ─────────────
    cases = [
        ('Hard layered', make_hard_bottom()),
        ('Soft layered', make_soft_bottom()),
    ]

    models = [
        ('RAM', RAM(verbose=False, accuracy=1e-1)),
        ('KF adiabatic', KrakenField(verbose=False, n_segments=8, mode_coupling='adiabatic')),
        ('KF coupled', KrakenField(verbose=False, n_segments=8, mode_coupling='coupled')),
    ]

    # results[case_label][model_label] = field
    results = {}
    envs = {}

    for case_label, bottom in cases:
        env, _ = make_base_env(bottom)
        envs[case_label] = env
        results[case_label] = {}

        print(f"\n  {case_label} bottom:")
        for model_label, model in models:
            try:
                field = model.run(env, source, receiver)
                results[case_label][model_label] = field
                print(f"    {model_label:15s} TL: [{np.nanmin(field.tl):.1f}, {np.nanmax(field.tl):.1f}] dB")
            except Exception as e:
                print(f"    {model_label:15s} ERROR: {e}")
                results[case_label][model_label] = None

    # ── Statistics ───────────────────────────────────────────────
    mid_idx = receiver.depths.shape[0] // 2
    mid_depth = receiver.depths[mid_idx]
    ranges_km = receiver.ranges / 1000

    print(f"\n  Comparison vs RAM at {mid_depth:.0f} m depth:")
    for case_label in results:
        f_ram = results[case_label].get('RAM')
        for kf_label in ['KF adiabatic', 'KF coupled']:
            f_kf = results[case_label].get(kf_label)
            if f_ram is not None and f_kf is not None:
                diff = f_ram.tl[mid_idx, :] - f_kf.tl[mid_idx, :]
                print(f"    {case_label:15s} {kf_label:15s}  mean diff: {np.nanmean(diff):+.1f} dB,  "
                      f"RMS: {np.sqrt(np.nanmean(diff**2)):.1f} dB")

    # ── Plot: 3 rows x 4 cols, shared colorbars per row ──────────
    fig, axes = plt.subplots(3, 4, figsize=(22, 14),
                             gridspec_kw={'width_ratios': [1, 1, 1, 1]})

    # Shared TL color limits across all fields
    all_tl = []
    for case_label in results:
        for field in results[case_label].values():
            if field is not None:
                all_tl.append(field.tl)
    if all_tl:
        vmin_shared = max(30, np.nanpercentile(np.concatenate([a.ravel() for a in all_tl]), 5))
        vmax_shared = min(140, np.nanpercentile(np.concatenate([a.ravel() for a in all_tl]), 95))
        vmin_shared = 5 * round(vmin_shared / 5)
        vmax_shared = 5 * round(vmax_shared / 5)
    else:
        vmin_shared, vmax_shared = 40, 100

    model_panels = [
        ('RAM', 'RAM (PE)'),
        ('KF adiabatic', 'KrakenField (adiabatic)'),
        ('KF coupled', 'KrakenField (coupled)'),
    ]

    tl_im = None
    for row_idx, case_label in enumerate(['Hard layered', 'Soft layered']):
        env_plot = envs[case_label]

        for col_idx, (key, title_suffix) in enumerate(model_panels):
            ax = axes[row_idx, col_idx]
            f = results[case_label].get(key)
            if f is not None:
                plot_field(f, env=env_plot, ax=ax, show_colorbar=False,
                                       vmin=vmin_shared, vmax=vmax_shared)
                tl_im = ax.collections[0] if ax.collections else tl_im
            ax.set_title(f'{case_label} — {title_suffix}', fontsize=10,
                         fontweight='bold')
            if col_idx > 0:
                ax.set_ylabel('')

        ax = axes[row_idx, 3]
        colors = {'RAM': 'C0', 'KF adiabatic': 'C1', 'KF coupled': 'C2'}
        for key in ['RAM', 'KF adiabatic', 'KF coupled']:
            f = results[case_label].get(key)
            if f is not None:
                ax.plot(ranges_km, f.tl[mid_idx, :], color=colors[key],
                        label=key)
        ax.set_xlabel('Range (km)')
        ax.set_ylabel('TL (dB)')
        ax.invert_yaxis()
        ax.legend(fontsize=8)
        ax.set_title(f'{case_label} — TL at {mid_depth:.0f} m', fontsize=10,
                     fontweight='bold')
        ax.grid(True, alpha=0.3)

    diff_panels = [
        ('KF adiabatic', 'RAM - Adiabatic'),
        ('KF coupled', 'RAM - Coupled'),
    ]

    diff_im = None
    diff_vmax_shared = None
    for case_idx, case_label in enumerate(['Hard layered', 'Soft layered']):
        f_ram = results[case_label].get('RAM')
        for diff_idx, (kf_key, _) in enumerate(diff_panels):
            f_kf = results[case_label].get(kf_key)
            if f_ram is not None and f_kf is not None:
                d = np.asarray(f_ram.tl) - np.asarray(f_kf.tl)
                finite = d[np.isfinite(d)]
                if finite.size:
                    v = max(5.0, float(np.nanpercentile(np.abs(finite), 95)))
                    v = 5.0 * np.ceil(v / 5.0)
                    diff_vmax_shared = max(diff_vmax_shared or 0.0, v)
    if diff_vmax_shared is None:
        diff_vmax_shared = 10.0

    for case_idx, case_label in enumerate(['Hard layered', 'Soft layered']):
        f_ram = results[case_label].get('RAM')
        env_plot = envs[case_label]
        for diff_idx, (kf_key, diff_title) in enumerate(diff_panels):
            col = case_idx * 2 + diff_idx
            ax = axes[2, col]
            f_kf = results[case_label].get(kf_key)
            if f_ram is not None and f_kf is not None:
                _plot_tl_difference(f_ram, f_kf, env_plot, ax=ax,
                                   label=diff_title, show_colorbar=False,
                                   diff_vmax=diff_vmax_shared)
                diff_im = ax.collections[0] if ax.collections else diff_im
            ax.set_title(f'{case_label} — {diff_title}', fontsize=10,
                         fontweight='bold')
            if col > 0:
                ax.set_ylabel('')

    fig.suptitle(
        'Example 18: Range-Dependent Bottom — Adiabatic vs Coupled Modes vs RAM\n'
        f'f={source.frequencies[0]:.0f} Hz, z_s={source.depths[0]:.0f} m, '
        f'bathy {bathy_depths_m[0]:.0f}-{bathy_depths_m[-1]:.0f} m',
        fontsize=13, fontweight='bold', y=0.995)
    fig.subplots_adjust(left=0.05, right=0.93, top=0.92, bottom=0.06,
                        wspace=0.18, hspace=0.30)
    if tl_im is not None:
        cbar_top = fig.add_axes([0.945, 0.36, 0.010, 0.56])
        fig.colorbar(tl_im, cax=cbar_top, label='TL (dB)')
    if diff_im is not None:
        cbar_bot = fig.add_axes([0.945, 0.06, 0.010, 0.24])
        fig.colorbar(diff_im, cax=cbar_bot, label='Δ TL (dB)')

    out_dir = Path(__file__).parent / 'output'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'example_18_rd_krakenfield_vs_ram.png'
    fig.savefig(out_path, dpi=150)
    print(f"\n  ✓ Saved: {out_path}")

    for case_label in ('Hard layered', 'Soft layered'):
        fig_b, _ = plot_environment(envs[case_label])
        slug = case_label.lower().replace(' ', '_')
        path = out_dir / f'example_18_rd_layered_{slug}.png'
        fig_b.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig_b)
        print(f"  ✓ Saved: {path}")

    print("\n✓ Example 18 complete\n")


if __name__ == '__main__':
    main()
