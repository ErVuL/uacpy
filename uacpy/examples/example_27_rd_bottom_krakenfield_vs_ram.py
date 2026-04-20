"""
Example 27: Range-Dependent Bottom — Adiabatic vs Coupled Modes vs RAM
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

import numpy as np
import uacpy
from uacpy.core.environment import (
    BoundaryProperties, SedimentLayer, LayeredBottom,
    RangeDependentLayeredBottom,
)
from uacpy.models.ram import RAM
from uacpy.models.kraken import KrakenField


def make_base_env(bottom):
    """Build the shared environment with given bottom type."""
    bathy_ranges_m = np.array([0, 5000, 10000, 15000, 20000.0])
    bathy_depths_m = np.array([100, 120, 150, 180, 200.0])
    bathymetry = np.column_stack([bathy_ranges_m, bathy_depths_m])

    ssp_depths = np.array([0, 50, 100, 150, 200.0])
    ssp_near = np.array([1510, 1505, 1500, 1500, 1500.0])
    ssp_far = np.array([1500, 1495, 1490, 1492, 1495.0])
    ssp_ranges_km = np.array([0.0, 20.0])
    ssp_2d = np.column_stack([ssp_near, ssp_far])

    env = uacpy.Environment(
        name='rd_comparison',
        depth=200,
        ssp_type='linear',
        ssp_data=list(zip(ssp_depths, ssp_near)),
        ssp_2d_ranges=ssp_ranges_km,
        ssp_2d_matrix=ssp_2d,
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
        ranges_km=np.array([0.0, 20.0]),
        depths=np.array([100.0, 200.0]),
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
        ranges_km=np.array([0.0, 20.0]),
        depths=np.array([100.0, 200.0]),
        profiles=[near, far],
    )


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from uacpy.visualization.plots import plot_transmission_loss

    print("Example 27: Range-Dependent Bottom — Adiabatic vs Coupled vs RAM")
    print("=" * 65)

    source = uacpy.Source(frequency=100, depth=30)
    receiver = uacpy.Receiver(
        depths=np.linspace(5, 195, 100),
        ranges=np.linspace(1000, 20000, 80),
    )

    bathy_ranges_m = np.array([0, 5000, 10000, 15000, 20000.0])
    bathy_depths_m = np.array([100, 120, 150, 180, 200.0])

    # ── Run both bottom cases with all three models ─────────────
    cases = [
        ('Hard layered', make_hard_bottom()),
        ('Soft layered', make_soft_bottom()),
    ]

    models = [
        ('RAM', RAM(verbose=False)),
        ('KF adiabatic', KrakenField(verbose=False, n_segments=20, mode_coupling='adiabatic')),
        ('KF coupled', KrakenField(verbose=False, n_segments=20, mode_coupling='coupled')),
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
                print(f"    {model_label:15s} TL: [{np.nanmin(field.data):.1f}, {np.nanmax(field.data):.1f}] dB")
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
                diff = f_ram.data[mid_idx, :] - f_kf.data[mid_idx, :]
                print(f"    {case_label:15s} {kf_label:15s}  mean diff: {np.nanmean(diff):+.1f} dB,  "
                      f"RMS: {np.sqrt(np.nanmean(diff**2)):.1f} dB")

    # ── Plot: 3 rows x 4 cols ──────────────────────────────────
    # Row 1: Hard bottom — RAM, KF adiabatic, KF coupled, TL curves
    # Row 2: Soft bottom — RAM, KF adiabatic, KF coupled, TL curves
    # Row 3: Difference panels (RAM-adiabatic, RAM-coupled) for each case
    fig, axes = plt.subplots(3, 4, figsize=(26, 16))

    # Compute shared TL color limits across all fields
    all_tl = []
    for case_label in results:
        for field in results[case_label].values():
            if field is not None:
                all_tl.append(field.data)
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

    for row_idx, case_label in enumerate(['Hard layered', 'Soft layered']):
        env_plot = envs[case_label]

        # TL panels: RAM, adiabatic, coupled
        for col_idx, (key, title_suffix) in enumerate(model_panels):
            ax = axes[row_idx, col_idx]
            f = results[case_label].get(key)
            if f is not None:
                plot_transmission_loss(f, env_plot, ax=ax, show_colorbar=True,
                                      vmin=vmin_shared, vmax=vmax_shared)
            ax.set_title(f'{case_label} — {title_suffix}', fontsize=10,
                         fontweight='bold')

        # TL curves at mid-depth
        ax = axes[row_idx, 3]
        colors = {'RAM': 'C0', 'KF adiabatic': 'C1', 'KF coupled': 'C2'}
        for key in ['RAM', 'KF adiabatic', 'KF coupled']:
            f = results[case_label].get(key)
            if f is not None:
                ax.plot(ranges_km, f.data[mid_idx, :], color=colors[key],
                        label=key)
        ax.set_xlabel('Range (km)')
        ax.set_ylabel('TL (dB)')
        ax.invert_yaxis()
        ax.legend(fontsize=8)
        ax.set_title(f'{case_label} — TL at {mid_depth:.0f} m', fontsize=10,
                     fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Row 3: Difference panels
    diff_panels = [
        ('KF adiabatic', 'RAM - Adiabatic'),
        ('KF coupled', 'RAM - Coupled'),
    ]

    for case_idx, case_label in enumerate(['Hard layered', 'Soft layered']):
        f_ram = results[case_label].get('RAM')
        for diff_idx, (kf_key, diff_title) in enumerate(diff_panels):
            col = case_idx * 2 + diff_idx
            ax = axes[2, col]
            f_kf = results[case_label].get(kf_key)
            if f_ram is not None and f_kf is not None:
                diff_data = f_ram.data - f_kf.data
                R, Z = np.meshgrid(ranges_km, receiver.depths)
                dmax = max(5, np.nanpercentile(np.abs(diff_data), 95))
                dmax = 5 * round(dmax / 5 + 0.5)
                im = ax.pcolormesh(R, Z, diff_data, cmap='RdBu_r',
                                   vmin=-dmax, vmax=dmax, shading='auto')
                ax.invert_yaxis()
                ax.set_xlabel('Range (km)')
                ax.set_ylabel('Depth (m)')
                fig.colorbar(im, ax=ax, label='dB')
                ax.fill_between(bathy_ranges_m / 1000, bathy_depths_m,
                                np.max(bathy_depths_m) * 1.1,
                                color='saddlebrown', alpha=0.7)
                ax.set_ylim([np.max(bathy_depths_m) * 1.05, 0])
            ax.set_title(f'{case_label} — {diff_title}', fontsize=10,
                         fontweight='bold')

    fig.suptitle(
        'Example 27: Range-Dependent Bottom — Adiabatic vs Coupled Modes vs RAM\n'
        f'f={source.frequency[0]:.0f} Hz, z_s={source.depth[0]:.0f} m, '
        f'bathy {bathy_depths_m[0]:.0f}-{bathy_depths_m[-1]:.0f} m\n'
        'Coupled modes recover mode-coupling energy lost by adiabatic approximation',
        fontsize=13, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_dir = Path(__file__).parent / 'output'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / 'example_27_rd_krakenfield_vs_ram.png'
    fig.savefig(out_path, dpi=150)
    print(f"\n  Figure saved to {out_path}")
    print("Done.")


if __name__ == '__main__':
    main()
