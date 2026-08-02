"""Figures for ``docs/guide/data.md``.

The code here is the code shown on the page: each builder is the worked
example, so a figure cannot drift from the snippet that claims to produce it.

Every figure resolves from the **offline cache** (``./install.sh --data all``)
and never touches the network: bathymetry and sound speed are pinned with
``*_sources='local'``, and the seabed / sea-ice / wind figures call the cached
readers directly. That keeps the page reproducible and keeps figure generation
independent of a live service.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import uacpy
from uacpy.data import SOURCES

# Write into docs/guide/figures/ rather than docs/models/figures/.
GUIDE = True


# ── the site every figure uses ───────────────────────────────────────────────
# A Celtic-shelf-break transect into the Biscay abyssal plain: 320 km, 150 m of
# shelf falling to 4.8 km of abyss, inside EMODnet's European-seas seabed
# coverage, so every axis has real data along the whole track.

A, B = (48.2, -8.0), (45.6, -6.2)
REGION_LAT, REGION_LON = (44.5, 49.5), (-11.0, -3.5)
DATE = '2026-07-15'


def fetch_env():
    """The range-dependent environment along A → B, from local data only."""
    return uacpy.data.fetch_environment(
        A, transect_to=B, date=DATE,
        name='Celtic shelf break → Biscay abyssal plain',
        bathymetry_sources='local',      # cached GEBCO grid
        ssp_sources='local',             # cached WOA23 climatology
        bottom_sources='local',          # cached EMODnet substrate
        n_points=140, ssp_n_points='auto', bottom_n_points=10,
    )


# ── figures ──────────────────────────────────────────────────────────────────


def bathymetry_map():
    """A regional GEBCO grid with the transect drawn on it."""
    lats, lons, depth = uacpy.data.fetch_bathy_grid(
        REGION_LAT, REGION_LON, n_lat=420, n_lon=420, source='local')

    fig, _ = uacpy.plot.plot_bathymetry_map(
        lats, lons, depth, transect=(A, B), source=A,
        contours=[200.0, 1000.0, 2000.0, 3000.0, 4000.0],
        figsize=(8.0, 7.0), data_source=[SOURCES['gebco']],
        title='GEBCO 2025 — Celtic shelf break to the Biscay abyssal plain')
    return fig


def transect_environment():
    """The Environment `fetch_environment` returns for that transect."""
    env = fetch_env()
    fig, _ = env.plot(
        figsize=(9.5, 4.6),
        title=f'fetch_environment({A}, transect_to={B}, date={DATE!r})')
    return fig


def seasonal_ssp():
    """One WOA23 site, four climatological months — full depth and the surface layer."""
    months = [(2, 'February'), (5, 'May'), (8, 'August'), (11, 'November')]
    profiles = [(label, uacpy.data.fetch_ssp(B, month=month, source='local'))
                for month, label in months]

    fig, (ax_full, ax_top) = plt.subplots(1, 2, figsize=(9.0, 5.0))
    styles = (('C0', '-'), ('C2', '--'), ('C3', '-.'), ('C1', ':'))
    for (label, ssp), (colour, dash) in zip(profiles, styles):
        for ax in (ax_full, ax_top):
            ax.plot(ssp.data[:, 0], ssp.depths, color=colour, linestyle=dash,
                    linewidth=1.6, label=label)
    # WOA23's monthly fields stop at 1500 m; the annual mean is spliced on below,
    # so all four profiles are identical there (the dashes overlap).
    ax_full.axhline(1500.0, color='0.5', linewidth=0.9)
    ax_full.annotate('monthly → annual splice', (1540.0, 1500.0), color='0.4',
                     fontsize=8, ha='right', va='bottom')
    ax_full.set_title('Full column', fontweight='bold', fontsize=11)
    ax_full.legend(fontsize=8, loc='lower left')
    ax_top.set_title('Upper 300 m — the seasonal layer', fontweight='bold',
                     fontsize=11)
    ax_top.set_ylim(300.0, 0.0)

    for ax in (ax_full, ax_top):
        ax.set_xlabel('Sound speed (m/s)')
        ax.set_ylabel('Depth (m)')
        ax.grid(True, alpha=0.3)
    ax_full.invert_yaxis()
    fig.suptitle(f'WOA23 climatology at {B[0]}°N, {abs(B[1])}°W',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


def grain_size_conversion():
    """Wentworth ϕ → c_p, ρ, α, under both conversion models."""
    phi = np.linspace(-0.5, 9.0, 381)
    water_c, water_rho = 1490.0, 1.03           # near-seabed seawater

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0))
    for model, colour in (('hamilton', 'C0'), ('apl-uw', 'C3')):
        rows = [uacpy.data.grain_size_to_geoacoustics(
            p, model=model, water_sound_speed=water_c, water_density=water_rho)
            for p in phi]
        for ax, key in zip(axes, ('sound_speed', 'density', 'attenuation')):
            ax.plot(phi, [r[key] for r in rows], color=colour, linewidth=1.7,
                    label=model)

    axes[0].axhline(water_c, color='0.4', linestyle='--', linewidth=0.9)
    axes[0].annotate('seawater', (9.0, water_c), color='0.4', fontsize=8,
                     ha='right', va='bottom', xytext=(0, 3),
                     textcoords='offset points')
    for ax, label in zip(axes, ('Sound speed c_p (m/s)', 'Density ρ (g/cm³)',
                                'Attenuation α (dB/λ)')):
        for edge in (4.0, 8.0):                 # Wentworth sand | silt | clay
            ax.axvline(edge, color='0.75', linewidth=0.8, zorder=0)
        ax.set_xlabel('Mean grain size ϕ  (coarse → fine)')
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    for centre, name in ((1.7, 'sand'), (6.0, 'silt'), (8.6, 'clay')):
        axes[0].annotate(name, (centre, 0.96), xycoords=('data', 'axes fraction'),
                         color='0.45', fontsize=8, ha='center', va='top')
    axes[0].legend(fontsize=8, loc='center right')
    fig.suptitle('grain_size_to_geoacoustics — ϕ to a model-ready half-space',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


def seabed_sources():
    """Five surficial seabed sources, sampled along the same transect."""
    from uacpy.data import (
        diesing_local, emodnet_local, graw_local, pelagic, sediment_db,
    )

    sources = [
        ('emodnet', emodnet_local.fetch_bottom_local_transect(A, B, n_points=14)),
        ('grainsize', sediment_db.fetch_bottom_local_transect(A, B, n_points=14)),
        ('diesing', diesing_local.fetch_bottom_diesing_transect(A, B, n_points=14)),
        ('graw', graw_local.fetch_bottom_graw_transect(A, B, n_points=14)),
        ('pelagic', pelagic.fetch_bottom_pelagic_transect(
            A, B, n_points=14, cache_only=True)),
    ]

    fig, (ax_c, ax_r) = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for label, bottom in sources:
        km = np.asarray(bottom.ranges) / 1000.0
        ax_c.step(km, bottom.halfspace_sound_speed, where='post',
                  linewidth=1.6, label=label)
        ax_r.step(km, bottom.halfspace_density, where='post', linewidth=1.6,
                  label=label)
    ax_c.set_ylabel('Half-space c_p (m/s)')
    ax_r.set_ylabel('Half-space ρ (g/cm³)')
    for ax in (ax_c, ax_r):
        ax.set_xlabel('Range along A → B (km)')
        ax.grid(True, alpha=0.3)
    ax_c.set_ylim(top=1880.0)
    ax_c.legend(fontsize=8, loc='upper right', ncol=2)
    fig.suptitle('bottom_sources — five surficial seabeds, one transect',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


def crust1_column():
    """The CRUST1.0 + GlobSed layered elastic column under the deep end."""
    from uacpy.data import crust1_local

    column = crust1_local.fetch_bottom_crust1(B)
    z = np.linspace(0.0, column.total_thickness() * 1.4, 800)
    edges = np.cumsum([layer.thickness for layer in column.layers])

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 4.6), sharey=True)
    for ax, key, label in zip(
            axes, ('sound_speed', 'shear_speed', 'density'),
            ('c_p (m/s)', 'c_s (m/s)', 'ρ (g/cm³)')):
        ax.plot([getattr(column.at(depth=zz), key) for zz in z], z,
                color='saddlebrown', linewidth=1.8)
        for edge in edges:
            ax.axhline(edge, color='0.6', linestyle='--', linewidth=0.9)
        ax.set_xlabel(label)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('Depth below the seafloor (m)')
    axes[0].invert_yaxis()
    bands = np.concatenate([[0.0], edges, [z[-1]]])
    names = ([f'sediment {i}' for i in range(1, len(column.layers) + 1)]
             + ['crystalline basement'])
    for name, top, base in zip(names, bands[:-1], bands[1:]):
        axes[0].annotate(name, (0.96, 0.5 * (top + base)),
                         xycoords=('axes fraction', 'data'), fontsize=8,
                         color='0.35', ha='right', va='center',
                         bbox=dict(boxstyle='round,pad=0.15', fc='white',
                                   ec='none', alpha=0.75))
    fig.suptitle(f'CRUST1.0 + GlobSed column at {B[0]}°N, {abs(B[1])}°W',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


def sea_ice():
    """The NSIDC ice climatology at its seasonal extremes, over one transect."""
    from uacpy.data import seaice_local

    fram = ((81.0, -2.0), (76.5, 2.0))          # Fram Strait, marginal ice zone

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for ax, month, label in ((axes[0], 3, 'March — winter maximum'),
                             (axes[1], 9, 'September — summer minimum')):
        uacpy.plot.plot_sea_ice_map(
            seaice_local.sea_ice_grid(month, hemi='N'), hemi='N',
            transect=fram, ax=ax, title=label)
    fig.suptitle('NSIDC sea-ice concentration climatology — Fram Strait',
                 fontweight='bold', fontsize=13)
    fig.tight_layout()
    return fig


FIGURES = {
    'data_bathymetry_map': bathymetry_map,
    'data_transect_environment': transect_environment,
    'data_seasonal_ssp': seasonal_ssp,
    'data_grain_size': grain_size_conversion,
    'data_seabed_sources': seabed_sources,
    'data_crust1_column': crust1_column,
    'data_sea_ice': sea_ice,
}
