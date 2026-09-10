"""Range-dependent bottom — adiabatic modes, coupled modes and RAM.

One sloping, range-varying scenario (100 m → 200 m over 20 km, layered sediment
that changes along range, warmer nearshore water) run three ways: Kraken in
adiabatic mode theory, Kraken in coupled mode theory, and RAM's parabolic
equation. Two seabeds, hard and soft, so the comparison is not read off a single
bottom.

Measured here, coupling does NOT track RAM more closely than the adiabatic
treatment — it comes out slightly further away in both cases. Coupling changes
the answer, which is the point worth seeing; it does not automatically improve
it. The printed table is the result, not this paragraph.

Two constraints, both real limits rather than incidental settings:

* n_segments is pinned to 2 for BOTH Kraken runs. The coupled path through AT's
  field.exe writes a .shd whose header disagrees with its payload at 3 or more
  segments, and the reader rejects it. Adiabatic is held at the same 2 so the
  mode treatments differ only in coupling.
* Kraken has no range-dependent bottom: both runs collapse the seabed to one
  column (and say so). The hard/soft contrast survives, the sediment's range
  variation does not — so part of every difference below is that missing degree
  of freedom, not the mode theory.

Uses: Bottom.from_columns · SoundSpeedProfile.from_2d ·
Kraken(n_segments=, mode_coupling=) · RAM(accuracy=) · plot_field ·
plot.plot_field_difference · plot.shared_colorbar
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

BATHYMETRY = np.column_stack([[0, 5000, 10000, 15000, 20000.0],
                              [100, 120, 150, 180, 200.0]])


def two_column_bottom(near_layers, near_halfspace, far_layer, far_halfspace):
    """A layered seabed with one column at 0 km and another at 20 km.

    Each layer is ``(thickness_m, cp, rho, alpha)`` and each half-space
    ``(cp, rho, alpha)``; the near column carries two layers, the far one.
    """
    def column(layers, halfspace):
        return uacpy.SeabedColumn(
            layers=[uacpy.SedimentLayer(thickness=thickness, sound_speed=cp,
                                        density=rho, attenuation=alpha)
                    for thickness, cp, rho, alpha in layers],
            halfspace=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=halfspace[0],
                density=halfspace[1], attenuation=halfspace[2]))

    return uacpy.Bottom.from_columns(
        [column(near_layers, near_halfspace), column([far_layer],
                                                     far_halfspace)],
        ranges=np.array([0, 20000]))


def build_env(bottom):
    """The shared range-dependent environment, over the given seabed."""
    return uacpy.Environment(
        name='rd_comparison',
        ssp=uacpy.SoundSpeedProfile.from_2d(
            depths=np.array([0, 50, 100, 150, 200.0]),
            ranges=np.array([0.0, 20000.0]),
            matrix=np.column_stack([[1510, 1505, 1500, 1500, 1500.0],
                                    [1500, 1495, 1490, 1492, 1495.0]])),
        bathymetry=BATHYMETRY,
        bottom=bottom,
    )


source = uacpy.Source(frequencies=100, depths=30)
receiver = uacpy.Receiver(depths=np.linspace(5, 195, 30),
                          ranges=np.linspace(1000, 6000, 300))

cases = {
    # Hard: high impedance contrast, low attenuation.
    'Hard layered': build_env(two_column_bottom(
        near_layers=[(8.0, 1600, 1.8, 0.2), (20.0, 1700, 2.0, 0.1)],
        near_halfspace=(2500, 2.5, 0.05),
        far_layer=(3.0, 1800, 2.0, 0.1), far_halfspace=(3000, 2.8, 0.02))),
    # Soft: low impedance contrast, high attenuation.
    'Soft layered': build_env(two_column_bottom(
        near_layers=[(8.0, 1500, 1.2, 1.0), (20.0, 1580, 1.5, 0.6)],
        near_halfspace=(1800, 2.0, 0.2),
        far_layer=(3.0, 1650, 1.8, 0.3), far_halfspace=(2500, 2.5, 0.05))),
}
models = {
    'RAM': uacpy.RAM(accuracy=1e-1),
    'Kraken adiabatic': uacpy.Kraken(n_segments=2, mode_coupling='adiabatic'),
    'Kraken coupled': uacpy.Kraken(n_segments=2, mode_coupling='coupled'),
}

fields = {case: {name: model.run(env, source, receiver)
                 for name, model in models.items()}
          for case, env in cases.items()}

mid_depth = float(receiver.depths[receiver.depths.size // 2])
print(f"Against the RAM reference at {mid_depth:.0f} m "
      f"(Kraken's seabed is collapsed to one column — part of every "
      f"difference is that):")
rms = {}
for case, per_model in fields.items():
    ram_cut = per_model['RAM'].at(depth=mid_depth).dB
    for name in ('Kraken adiabatic', 'Kraken coupled'):
        residual = ram_cut - per_model[name].at(depth=mid_depth).dB
        rms[(case, name)] = float(np.sqrt(np.nanmean(residual ** 2)))
        print(f"  {case:13s} {name:17s} mean {np.nanmean(residual):+.1f} dB, "
              f"RMS {rms[(case, name)]:.1f} dB")
for case in fields:
    adiabatic, coupled = rms[(case, 'Kraken adiabatic')], rms[(case, 'Kraken coupled')]
    print(f"  {case:13s} coupling moves RMS {adiabatic:.1f} → {coupled:.1f} dB "
          f"({'closer to' if coupled < adiabatic else 'further from'} RAM)")

# One TL colour window across every panel, so the six are comparable, and one
# difference window across the four residuals.
every_tl = np.concatenate([f.dB.ravel() for per_model in fields.values()
                           for f in per_model.values()])
vmin = 5 * round(max(30, np.nanpercentile(every_tl, 5)) / 5)
vmax = 5 * round(min(140, np.nanpercentile(every_tl, 95)) / 5)
residuals = [np.asarray(per_model['RAM'].dB) - np.asarray(per_model[name].dB)
             for per_model in fields.values()
             for name in ('Kraken adiabatic', 'Kraken coupled')]
diff_vmax = max(5.0, 5.0 * np.ceil(max(
    float(np.nanpercentile(np.abs(r), 95)) for r in residuals) / 5.0))

fig, axes = plt.subplots(3, 4, figsize=(22, 14))
for row, (case, per_model) in enumerate(fields.items()):
    for col, name in enumerate(models):
        uacpy.plot_field(per_model[name], axes[row, col], env=cases[case],
                         show_colorbar=False, vmin=vmin, vmax=vmax,
                         title=f'{case} — {name}')
    for name in models:                       # the same three as one cut
        axes[row, 3].plot(receiver.ranges / 1000,
                          per_model[name].at(depth=mid_depth).dB, label=name)
    axes[row, 3].set_xlabel('Range (km)')
    axes[row, 3].set_ylabel('TL (dB)')
    axes[row, 3].invert_yaxis()
    axes[row, 3].legend(fontsize=8)
    axes[row, 3].grid(True, alpha=0.3)
    axes[row, 3].set_title(f'{case} — TL at {mid_depth:.0f} m')

for col, (case, name) in enumerate([(c, n) for c in fields
                                    for n in ('Kraken adiabatic',
                                              'Kraken coupled')]):
    uacpy.plot.plot_field_difference(
        fields[case]['RAM'], fields[case][name], axes[2, col], env=cases[case],
        show_colorbar=False, diff_vmax=diff_vmax,
        title=f'{case} — RAM − {name.split()[1]}')

fig.suptitle('Range-dependent bottom — adiabatic vs coupled modes vs RAM\n'
             f'f={source.frequencies[0]:.0f} Hz, z_s={source.depths[0]:.0f} m, '
             f'n_segments=2, window {receiver.ranges[0] / 1000:.0f}-'
             f'{receiver.ranges[-1] / 1000:.0f} km of a '
             f'{BATHYMETRY[0, 1]:.0f}-{BATHYMETRY[-1, 1]:.0f} m slope',
             fontsize=13, fontweight='bold', y=0.995)
# Margins first: each bar takes its space from the panels as they stand, so a
# subplots_adjust after them would move the panels back over the bars.
fig.subplots_adjust(left=0.05, top=0.90, bottom=0.06, wspace=0.18, hspace=0.30)
# Two bars, because there are two quantities: one over the six TL panels,
# one over the four residuals. Asking for a single bar across both would
# be refused — they are not on one scale, and could not honestly share one.
uacpy.plot.shared_colorbar(fig, axes[:2, :3], label='TL (dB)')
uacpy.plot.shared_colorbar(fig, axes[2, :], label='Δ TL (dB)')
fig.savefig(OUT / 'example_18_rd_krakenfield_vs_ram.png', dpi=150)
plt.close(fig)

for case, env in cases.items():
    fig, _ = env.plot()
    fig.savefig(OUT / f"example_18_rd_layered_{case.lower().replace(' ', '_')}.png",
                dpi=150, bbox_inches='tight')
    plt.close(fig)
