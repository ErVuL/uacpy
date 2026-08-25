"""Figures for ``docs/guide/environment.md``.

The code here is the code shown on the page: each builder is the worked
example, so a figure cannot drift from the snippet that claims to produce it.

Unlike the model pages, this one is *about* the carriers, so it builds a few
bespoke environments alongside the shared scenarios in
:mod:`figure_scripts._common`. Each is physically ordinary — a shelf-break
transect, a summer thermocline, a sand-over-limestone seabed — so the plots
show real structure rather than artefacts of a contrived setup.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from figure_scripts._common import WIDE, sloping_shelf

import uacpy

# Write into docs/guide/figures/ rather than docs/models/figures/.
GUIDE = True


# ── bespoke scenarios ────────────────────────────────────────────────────────


def shelf_break():
    """A shelf-break transect exercising every carrier at once.

    Seafloor falling from 90 m to 380 m over 20 km, a summer profile with a
    thermocline at 60 m, a seabed that turns from sand-over-limestone to
    silt-over-chalk past the break, and a wind-driven sea surface. The
    receiver is a vertical array at 18 km.
    """
    env = uacpy.Environment(
        name='Shelf break',
        bathymetry=[(0.0, 90.0), (6000.0, 110.0),
                    (12000.0, 210.0), (20000.0, 380.0)],
        ssp=[(0.0, 1522.0), (20.0, 1521.0), (60.0, 1502.0),
             (150.0, 1492.0), (380.0, 1488.0)],
        bottom=uacpy.Bottom.from_columns(
            [uacpy.SeabedColumn.from_presets(
                layers=[('sand', 12.0)], halfspace='limestone'),
             uacpy.SeabedColumn.from_presets(
                layers=[('silt', 30.0)], halfspace='chalk')],
            ranges=[0.0, 20_000.0],
        ),
        altimetry=uacpy.generate_sea_surface(
            20_000.0, wind_speed_mps=15.0, n_points=1200, seed=0),
    )
    source = uacpy.Source(depths=40.0, frequencies=150.0)
    receiver = uacpy.Receiver(depths=np.linspace(40.0, 320.0, 8),
                              ranges=18_000.0)
    return env, source, receiver


# A range-independent environment carries no range axis, so ``env.plot()``
# falls back to a nominal 1 km. Handing it the receiver pins the panel to the
# geometry that is actually being modelled — and keeps the four bottom-shape
# panels on one x-axis.
_ARRAY = uacpy.Receiver(depths=70.0, ranges=8000.0)


def _flat_shelf(bottom, name):
    """A flat 120 m shelf carrying ``bottom``.

    The water column is identical in all four bottom-shape panels, so the only
    thing that changes between them is the seabed carrier itself.
    """
    return uacpy.Environment(
        name=name,
        bathymetry=120.0,
        ssp=[(0.0, 1510.0), (30.0, 1500.0), (120.0, 1492.0)],
        bottom=bottom,
    )


def _rd_layered_bottom():
    """Sand over limestone inshore, silt over chalk offshore."""
    return uacpy.Bottom.from_columns(
        [uacpy.SeabedColumn.from_presets(
            layers=[('sand', 8.0)], halfspace='limestone'),
         uacpy.SeabedColumn.from_presets(
            layers=[('silt', 20.0)], halfspace='chalk')],
        ranges=[0.0, 8000.0],
    )


# ── figures ──────────────────────────────────────────────────────────────────


def overview():
    """Everything at once: the six carriers of one Environment."""
    env, source, receiver = shelf_break()
    fig, _ = env.plot(source=source, receiver=receiver, figsize=(9.5, 4.6),
                      title='Environment — shelf-break transect')
    return fig


def ssp_shapes():
    """Canonical sound-speed profiles, shallow and deep."""
    shallow = {
        'Isovelocity': uacpy.SoundSpeedProfile.from_isovelocity(200.0, 1500.0),
        'Summer thermocline': uacpy.SoundSpeedProfile.from_pairs(
            [(0.0, 1523.0), (15.0, 1522.0), (30.0, 1512.0),
             (60.0, 1497.0), (120.0, 1492.0), (200.0, 1493.0)]),
        'Winter (isothermal)': uacpy.SoundSpeedProfile.from_pairs(
            [(0.0, 1487.0), (50.0, 1487.8), (120.0, 1489.0),
             (200.0, 1490.3)]),
        'Surface duct': uacpy.SoundSpeedProfile.from_pairs(
            [(0.0, 1496.0), (40.0, 1499.0), (55.0, 1494.0),
             (120.0, 1490.0), (200.0, 1489.0)]),
    }
    munk = uacpy.SoundSpeedProfile.from_munk(5000.0)

    fig, (ax_s, ax_d) = plt.subplots(1, 2, figsize=(9.0, 5.0))
    for label, ssp in shallow.items():
        ax_s.plot(ssp.data[:, 0], ssp.depths, linewidth=1.6, label=label)
    ax_s.set_title('Shallow water (0–200 m)', fontweight='bold', fontsize=11)
    ax_s.legend(fontsize=8, loc='lower right')

    ax_d.plot(munk.data[:, 0], munk.depths, color='C4', linewidth=1.6,
              label='Munk (axis 1300 m)')
    ax_d.axhline(1300.0, color='gray', linestyle='--', linewidth=0.9)
    ax_d.set_title('Deep water (0–5000 m)', fontweight='bold', fontsize=11)
    ax_d.legend(fontsize=8, loc='lower left')

    for ax in (ax_s, ax_d):
        ax.set_xlabel('Sound speed (m/s)')
        ax.set_ylabel('Depth (m)')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
    fig.suptitle('SoundSpeedProfile — canonical shapes',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


def ssp_range_dependent():
    """A 2-D profile: the same carrier as a fan of casts and as a field."""
    depths = np.linspace(0.0, 300.0, 61)
    ranges = np.linspace(0.0, 30_000.0, 13)
    # A front: the thermocline deepens from 40 m to 110 m across the transect,
    # which is what a warm eddy or a shelf-slope front looks like in c(z, r).
    z_therm = np.linspace(40.0, 110.0, ranges.size)
    matrix = np.empty((depths.size, ranges.size))
    for j, zt in enumerate(z_therm):
        matrix[:, j] = 1492.0 + 30.0 / (1.0 + np.exp((depths - zt) / 12.0))
    ssp = uacpy.SoundSpeedProfile.from_2d(depths, ranges, matrix)

    env = uacpy.Environment(name='Frontal transect', bathymetry=300.0, ssp=ssp,
                            bottom='sand')

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ssp.plot(ax=ax_l, title='One cast per range node')
    ax_l.title.set(fontweight='bold', fontsize=12)
    env.plot(ax=ax_r, bottom_colorbar=False,
             title='The same profile in the environment')
    fig.suptitle('SoundSpeedProfile — range-dependent (2-D)',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


def range_profiles():
    """Bathymetry and Altimetry: two 1-D value(range) carriers, opposite signs."""
    env, _, _ = sloping_shelf()
    altimetry = uacpy.Altimetry.coerce(uacpy.generate_sea_surface(
        2000.0, wind_speed_mps=12.0, n_points=800, seed=7))

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(9.0, 5.2))
    altimetry.plot(
        ax=ax_a,
        title='Altimetry — sea-surface height, positive up (2 km of a 12 m/s sea)')
    env.bathymetry.plot(
        ax=ax_b,
        title='Bathymetry — seafloor depth, positive down (20 km of shelf)')
    fig.tight_layout()
    return fig


def bottom_shapes():
    """The four shapes a Bottom can take, over one identical seafloor."""
    cases = [
        ('Half-space',
         uacpy.BoundaryProperties.from_preset('sand')),
        ('Layered',
         uacpy.SeabedColumn.from_presets(
             layers=[('sand', 6.0), ('silt', 14.0)], halfspace='limestone')),
        ('Range-dependent half-space',
         uacpy.Bottom.from_halfspaces(
             [0.0, 4000.0, 8000.0], sound_speed=[1550.0, 1650.0, 1800.0],
             density=[1.6, 1.9, 2.0], attenuation=[0.3, 0.8, 0.6])),
        ('Range-dependent layered', _rd_layered_bottom()),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.4))
    for ax, (label, bottom) in zip(axes.ravel(), cases):
        _flat_shelf(bottom, label).plot(ax=ax, receiver=_ARRAY,
                                        bottom_colorbar=False, title=label)
    fig.suptitle('Bottom — the four shapes', fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


def bottom_properties():
    """Every geoacoustic property of an elastic layered seabed."""
    env = uacpy.Environment(
        name='Sand over limestone',
        bathymetry=[(0.0, 100.0), (10_000.0, 160.0)],
        ssp=[(0.0, 1505.0), (160.0, 1492.0)],
        bottom=uacpy.SeabedColumn.from_presets(
            layers=[('sand', 10.0), ('silt', 25.0)],
            halfspace='limestone', elastic=True),
    )
    fig, _ = uacpy.plot.plot_bottom_properties(
        env, figsize=(11.0, 6.0),
        title='Seabed properties — sand / silt over limestone (elastic)')
    return fig


def absorption_models():
    """The four Absorption models across five decades of frequency."""
    freqs = np.logspace(1, 5.7, 400)
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    uacpy.Thorp().plot(freqs, ax=ax, label='Thorp')
    uacpy.FrancoisGarrison(
        temperature_c=20.0, salinity_psu=35.0, pH=8.0, z_bar_m=50.0,
    ).plot(freqs, depth=50.0, ax=ax,
           label='Francois–Garrison, 20 °C / 50 m')
    uacpy.FrancoisGarrison(
        temperature_c=4.0, salinity_psu=35.0, pH=8.0, z_bar_m=3000.0,
    ).plot(freqs, depth=3000.0, ax=ax,
           label='Francois–Garrison, 4 °C / 3000 m')
    uacpy.ConstantAbsorption(value_db_per_wavelength=1.0e-4).plot(
        freqs, ax=ax, label='ConstantAbsorption, 1e-4 dB/λ')
    uacpy.Biological(layers=[(20.0, 80.0, 1500.0, 4.0, 0.02)]).plot(
        freqs, depth=50.0, ax=ax, label='Biological, f0 = 1.5 kHz')

    ax.set_title('Absorption — volume attenuation models', loc='left',
                 fontweight='bold')
    ax.set_ylim(1e-4, 1e3)
    return fig


def collapse():
    """One environment, three views: as written, as Bellhop and Scooter take it.

    The two reduced panels are built with the public carrier reductions the
    models apply themselves — ``Bottom.collapse(layers=…)``,
    ``Bottom.select_range(…)`` and ``get_representative_depth(…)`` — at each
    model's documented default method, so each panel reproduces exactly what
    that model's ``_project_environment`` hands its writer.
    """
    env, source, _ = shelf_break()
    # The collapsed panel is range-independent, so its own carriers span no
    # range; the shared receiver pins all three panels to one x-axis.
    receiver = uacpy.Receiver(depths=np.linspace(40.0, 320.0, 8),
                              ranges=20_000.0)

    # Bellhop takes range dependence and the rough surface natively. It has no
    # layered bottom, so every column is flattened to its half-space.
    bellhop_view = env.copy()
    bellhop_view.bottom = env.bottom.collapse(layers='halfspace')

    # Scooter is a stratified solver: the bathymetry, the bottom's range axis
    # and the sea surface all go, but the layer stack survives intact.
    scooter_view = env.copy()
    scooter_view.bathymetry = uacpy.Bathymetry.coerce(
        env.get_representative_depth('max'))
    scooter_view.bottom = env.bottom.select_range('median')
    scooter_view.altimetry = None

    panels = [
        (env, 'As written — range-dependent layered bottom, rough surface'),
        (bellhop_view, "Bellhop's view — layers flattened, range kept"),
        (scooter_view, "Scooter's view — range and surface gone, layers kept"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.0))
    for ax, (e, label) in zip(axes, panels):
        e.plot(ax=ax, source=source, receiver=receiver,
               bottom_colorbar=False, title=label)
        if ax is not axes[-1]:
            ax.set_xlabel('')
    fig.suptitle('Collapse — what a model does with a feature it cannot take',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


FIGURES = {
    'env_overview': overview,
    'env_ssp_shapes': ssp_shapes,
    'env_ssp_range_dependent': ssp_range_dependent,
    'env_range_profiles': range_profiles,
    'env_bottom_shapes': bottom_shapes,
    'env_bottom_properties': bottom_properties,
    'env_absorption': absorption_models,
    'env_collapse': collapse,
}
