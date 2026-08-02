"""Figures for ``docs/models/ram.md``.

The code here is the code shown on the page: each builder is the worked
example, so a figure cannot drift from the snippet that claims to produce it.
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np

from figure_scripts._common import WIDE, shallow_water, sloping_shelf

import uacpy
from uacpy.models import RAM, RunMode


def tl_field():
    """Coherent TL over the shallow-water channel, with the seabed drawn."""
    env, source, receiver = shallow_water()
    tl = RAM().run(env, source, receiver, run_mode=RunMode.COHERENT_TL)
    fig, ax = tl.plot(env=env, source=source, figsize=WIDE,
                      title='RAM — coherent transmission loss, 200 Hz')
    return fig


def sloping_shelf_tl():
    """The shelf break: bathymetry the PE marches straight through.

    The shared receiver grid reaches 395 m, which is below the seafloor over
    the inshore half of the track; those samples come back as no-data and RAM
    warns about them, so the warning is silenced here.
    """
    env, source, receiver = sloping_shelf()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tl = RAM().run(env, source, receiver, run_mode=RunMode.COHERENT_TL)
    fig, ax = tl.plot(env=env, source=source, figsize=WIDE,
                      title='RAM — 100 m shelf falling to 400 m over 20 km, '
                            '100 Hz')
    return fig


def backend_dispatch():
    """One API, four binaries — the same water routed four ways.

    Each panel changes exactly one feature of the seabed or the surface, and
    that feature alone decides which Collins-family binary runs. The last two
    differ only by ``elastic=``: identical geoacoustics, but declaring shear
    moves the run from the layered fluid PE onto the elastic one.
    """
    env, source, receiver = shallow_water()

    def sand_over_moraine(*, elastic):
        return uacpy.Environment(
            bathymetry=100.0, ssp=env.ssp,
            bottom=uacpy.SeabedColumn.from_presets(
                layers=[('sand', 8.0)], halfspace='moraine', elastic=elastic),
        )

    swell = uacpy.Environment(
        bathymetry=100.0, ssp=env.ssp, bottom=env.bottom,
        altimetry=[(r, -1.0 - np.cos(2.0 * np.pi * r / 750.0))
                   for r in np.linspace(0.0, 5000.0, 60)],
    )
    cases = [('half-space seabed, flat surface', env),
             ('swell at the surface', swell),
             ('layered fluid seabed', sand_over_moraine(elastic=False)),
             ('shear in the seabed', sand_over_moraine(elastic=True))]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.6), sharex=True,
                             sharey=True, layout='constrained')
    for ax, (label, case) in zip(axes.ravel(), cases):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tl = RAM().run(case, source, receiver, run_mode=RunMode.COHERENT_TL)
        tl.plot(env=case, source=source, ax=ax, show_colorbar=False)
        ax.set_title(f'{label} → backend={tl.backend!r}',
                     fontweight='bold', fontsize=10)
        if case.altimetry is not None:
            pairs = case.altimetry.to_pairs()
            ax.plot(pairs[:, 0] / 1000.0, -pairs[:, 1], color='C3',
                    linewidth=1.4, zorder=6)
    for ax in axes[0]:
        ax.set_xlabel('')
    for ax in axes[:, 1]:
        ax.set_ylabel('')
    fig.colorbar(axes[0, 0].collections[0], ax=axes, label='TL (dB)',
                 shrink=0.8)
    fig.suptitle('RAM — the dispatcher picks the binary from the environment',
                 fontweight='bold', fontsize=13)
    return fig


def pade_order():
    """What ``np_pade`` buys: the same answer for far fewer range steps.

    Every order resolves this waveguide, so the TL curves lie on top of one
    another. What changes is the range step the grid optimiser can afford at
    a fixed accuracy budget — two Padé terms need a 1.6 m step, eight need
    almost 60 m.
    """
    env, source, _ = shallow_water()
    line = uacpy.Receiver(depths=50.0, ranges=np.linspace(50.0, 5000.0, 400))
    orders = (2, 4, 6, 8)

    dr = []
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for order, colour in zip(orders, ('C0', 'C1', 'C2', 'C3')):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tl = RAM(np_pade=order).run(env, source, line)
        dr.append(tl.metadata['dr'])
        axes[0].plot(np.asarray(line.ranges) / 1000.0,
                     np.asarray(tl.tl, dtype=float).ravel(),
                     color=colour, linewidth=1.0,
                     label=f'np_pade={order} — dr={dr[-1]:.1f} m')

    axes[0].set_xlabel('Range (km)')
    axes[0].set_ylabel('TL (dB)')
    axes[0].set_title('TL at 50 m depth', fontweight='bold', fontsize=11)
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].bar([str(o) for o in orders], dr, color='C0', width=0.6)
    for x, value in enumerate(dr):
        axes[1].text(x, value, f'{value:.1f} m', ha='center', va='bottom',
                     fontsize=9)
    axes[1].set_xlabel('np_pade')
    axes[1].set_ylabel('Range step dr the optimiser affords (m)')
    axes[1].set_title('Cost of one accuracy budget', fontweight='bold',
                      fontsize=11)
    axes[1].set_ylim(0.0, max(dr) * 1.25)
    axes[1].grid(True, axis='y', alpha=0.3)

    fig.suptitle('RAM — higher Padé order, coarser march, same field',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


def lytaev_grid():
    """The two knobs the grid optimiser actually responds to.

    Left: widening ``theta_max`` forces a finer march — the PE spectrum has to
    stay accurate over a wider angular window. Right: leaving ``c0=None`` lets
    Lytaev Eq. (15) centre that spectrum, which buys a coarser step than
    pinning ``c0`` to the water speed.
    """
    env, source, _ = shallow_water()
    point = uacpy.Receiver(depths=50.0, ranges=np.linspace(50.0, 5000.0, 50))
    angles = (10.0, 20.0, 30.0, 45.0)

    def grid_of(model):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tl = model.run(env, source, point)
        return tl.metadata['dr'], tl.metadata['c0']

    swept = [grid_of(RAM(theta_max=angle)) for angle in angles]
    dr_auto, c0_auto = swept[2]
    dr_pinned, c0_pinned = grid_of(RAM(c0=1500.0))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    axes[0].plot(angles, [d for d, _ in swept], 'o-', color='C0', markersize=6)
    for angle, (d, c0) in zip(angles, swept):
        axes[0].annotate(f'c₀={c0:.0f} m/s', (angle, d),
                         textcoords='offset points', xytext=(0, -14),
                         ha='center', va='top', fontsize=8, color='0.3')
    axes[0].set_ylim(0.0, max(d for d, _ in swept) * 1.15)
    axes[0].set_xlabel('theta_max (degrees)')
    axes[0].set_ylabel('Range step dr (m)')
    axes[0].set_title('Angular window vs march cost', fontweight='bold',
                      fontsize=11)
    axes[0].grid(True, alpha=0.3)

    labels = [f'c0=None\nEq. (15) → {c0_auto:.0f} m/s',
              f'c0={c0_pinned:.0f}.0\npinned to the water']
    axes[1].bar(labels, [dr_auto, dr_pinned], color=['C0', 'C1'], width=0.55)
    for x, value in enumerate((dr_auto, dr_pinned)):
        axes[1].text(x, value, f'{value:.1f} m', ha='center', va='bottom',
                     fontsize=9)
    axes[1].set_ylabel('Range step dr (m)')
    axes[1].set_title('Reference speed vs march cost (theta_max=30°)',
                      fontweight='bold', fontsize=11)
    axes[1].set_ylim(0.0, max(dr_auto, dr_pinned) * 1.25)
    axes[1].grid(True, axis='y', alpha=0.3)

    fig.suptitle('RAM — what the Lytaev grid optimiser is trading off',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


def transfer_function():
    """Broadband H(f) at one point, from mpiramS's native frequency sweep."""
    env, _, _ = shallow_water()
    source = uacpy.Source(depths=25.0,
                          frequencies=np.linspace(150.0, 450.0, 192))
    point = uacpy.Receiver(depths=60.0, ranges=3000.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        H = RAM().run(env, source, point, run_mode=RunMode.BROADBAND)
    fig, _ = H.plot_transfer_function(
        title='RAM — transfer function at (3 km, 60 m)')
    return fig


FIGURES = {
    'ram_tl': tl_field,
    'ram_shelf': sloping_shelf_tl,
    'ram_backends': backend_dispatch,
    'ram_pade': pade_order,
    'ram_grid': lytaev_grid,
    'ram_transfer_function': transfer_function,
}
