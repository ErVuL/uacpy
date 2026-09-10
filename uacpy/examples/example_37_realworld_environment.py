"""Real-world environment — map · transmission loss · section.

The whole pipeline in one figure, from GPS coordinates to a modelled field:
fetch a regional GEBCO grid and draw it on a coastline map, build a fully
range-dependent Environment along a transect (GEBCO seafloor, WOA23 sound
speed, fetched seabed — all varying with range), run Bellhop for coherent TL,
and compose map, TL and environment with one call to plot_overview.

Region: the North Sea, shelf into the Norwegian Trench, chosen because the
cached EMODnet seabed actually varies along range (sand ↔ mud ↔ coarse), so the
range-dependent bottom is visible.

Then three further composites: a modelled range-dependent LAYERED seabed run
through RAM, a comparison of two seabed descriptions (grain-size half-space
against CRUST1.0 layered elastic), and an under-ice Arctic transect.

The `try` blocks here are the feature, not scaffolding: this is the one example
that reaches outside the process. It runs from the offline cache when installed
(./install.sh --data all), otherwise hits the live APIs, and falls back to
representative synthetic data — labelled `[offline fallback]` — if a service is
unreachable, so it always plots.

Uses: data.fetch_bathy_grid · data.fetch_environment (range-dependent
bathymetry + SSP + bottom + ice) · data.is_installed · emodnet_local ·
seaice_local · sediment_db · crust1_local · data.citations ·
plot.plot_overview(map_fn=) · plot.plot_sea_ice_map · plot.plot_bottom_properties
"""

import datetime as _dt
import os
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))   # uacpy from a checkout

import numpy as np
import matplotlib.pyplot as plt
import uacpy
from uacpy import data
from uacpy.core.exceptions import UACPYError

OUT = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
           or Path(__file__).parent / 'output')
OUT.mkdir(parents=True, exist_ok=True)

# ── Settings — everything tunable is here ───────────────────────────────────
REGION_LAT, REGION_LON = (56.5, 62.0), (-2.0, 9.0)
TRANSECT_START, TRANSECT_END = (61.0, 2.0), (58.0, 5.0)   # shelf → trench
TRANSECT_POINTS, SSP_POINTS, BOTTOM_POINTS = 40, 5, 5
COASTLINE_RES = '50m'                    # Natural Earth: 110m | 50m | 10m
GRATICULE_DEG, GRATICULE_MINOR_DEG = 2.0, 1.0
SOURCE_DEPTH, FREQ_HZ, RDLB_FREQ_HZ = 100.0, 800.0, 50.0
DATE = '2026-01-15'                      # picks the WOA climatology month
FORCE_ONLINE = False                     # True ⇒ ignore ./data_cache
BATHY_SOURCE = 'gebco'                   # 'gebco' (global) or 'gmrt'

# fetch_environment is cache-first WITHIN EACH SOURCE: an installed dataset is
# sampled before that source's own network call. It is NOT cache-first across
# the chain — 'auto' is ordered by data quality, so it can reach the network
# for a better source before falling through to an installed global grid.
# (Passing *_sources='local' keeps only the cached backends.)
GRID_SOURCE = 'local' if (not FORCE_ONLINE
                          and data.is_installed('gebco')) else 'api'
# The local GEBCO grid has no rate limit, so go fine; the online OpenTopoData
# API is fair-use-capped, so stay coarse.
N_LAT = N_LON = 400 if GRID_SOURCE == 'local' else 10

cached = [name for name in ('gebco', 'woa23', 'sediment', 'emodnet',
                            'coastline', 'globsed', 'crust1', 'diesing',
                            'seaice') if data.is_installed(name)]
print(f"  cache-first ({BATHY_SOURCE}), map grid {N_LAT}×{N_LON}")
print(f"  offline cache installed: {', '.join(cached) if cached else 'none'}")


def run_tl(env, frequency=FREQ_HZ):
    """Coherent TL over the whole section, with the geometry it used."""
    source = uacpy.Source(depths=SOURCE_DEPTH, frequencies=frequency)
    receiver = uacpy.Receiver(
        depths=np.linspace(1, env.bathymetry.depth, 150),
        ranges=np.linspace(100.0, env.bathymetry.range_max, 350))
    try:
        tl = uacpy.Bellhop().run(env, source, receiver,
                                 run_mode=uacpy.RunMode.COHERENT_TL)
        print(f"  TL: ok ({tl.dB.shape} grid, {frequency:g} Hz)")
    except Exception as exc:                    # model not built / run failed
        print(f"  TL: [skipped] {str(exc).splitlines()[0]}")
        tl = None
    return tl, source, receiver


# ── The regional bathymetry grid the map is drawn from ──────────────────────
try:
    lats, lons, depth = data.fetch_bathy_grid(REGION_LAT, REGION_LON,
                                              n_lat=N_LAT, n_lon=N_LON,
                                              source=GRID_SOURCE)
    print(f"  grid: ok ({N_LAT}×{N_LON} via {GRID_SOURCE}, "
          f"{np.isnan(depth).sum()} land cells)")
except UACPYError as exc:
    print(f"  grid: [offline fallback] {exc.message.splitlines()[0]}")
    lats = np.linspace(*REGION_LAT, N_LAT)
    lons = np.linspace(*REGION_LON, N_LON)
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    # A basin centred on the region, so the stand-in grid has sea cells.
    depth = 2800.0 * np.exp(-(((lat_mesh - np.mean(REGION_LAT)) / 3) ** 2
                              + ((lon_mesh - np.mean(REGION_LON)) / 3) ** 2))
    depth[depth < 200] = np.nan
grid = (lats, lons, depth)

# ── The transect environment: seafloor, water column and seabed all vary ────
try:
    env = data.fetch_environment(
        TRANSECT_START, date=DATE, transect_to=TRANSECT_END,
        name="North Sea — Norwegian Trench transect",
        n_points=TRANSECT_POINTS, bathymetry_sources=BATHY_SOURCE,
        range_dependent_ssp=True, ssp_n_points=SSP_POINTS,
        range_dependent_bottom=True, bottom_sources='local',
        bottom_n_points=BOTTOM_POINTS)
    print(f"  environment: ok ({env!r})")
except UACPYError as exc:
    print(f"  environment: [offline fallback] {exc.message.splitlines()[0]}")
    ranges = np.linspace(0.0, 370000.0, 5)
    depths = np.array([0, 50, 100, 200, 300.0])
    base = np.array([1505, 1500, 1496, 1494, 1493.0])
    section = np.linspace(0, 370000, 40)
    env = uacpy.Environment(
        name='North Sea transect (fallback)',
        # Shelf (~110 m) dipping into the Norwegian Trench (~300 m) and back.
        bathymetry=np.column_stack(
            [section, 110 + 190 * np.sin(np.pi * section / 370000) ** 2]),
        ssp=uacpy.SoundSpeedProfile(
            depths=depths, ranges=ranges,
            data=np.column_stack([base + 1.0 * np.sin(0.7 * k)
                                  for k in range(5)])),
        bottom=data.bottom_from_class('silt'))

tl, source, receiver = run_tl(env)

# The North Sea transect lies inside EMODnet coverage, so its cached 'local'
# bottom is EMODnet's Folk seabed; this is the substrate at one waypoint,
# through the offline polygons when installed and the live WFS otherwise —
# exactly what fetch_environment does cache-first.
point = (56.0, 3.0)
try:
    if not FORCE_ONLINE and data.is_installed('emodnet'):
        from uacpy.data import emodnet_local
        seabed, via = emodnet_local.fetch_bottom_local(point), 'offline polygons'
    else:
        seabed, via = data.fetch_bottom(point), 'live WFS'
    print(f"  EMODnet seabed @ {point}: {seabed.acoustic_type}, "
          f"c_p={seabed.sound_speed:.0f} m/s, ϕ={seabed.grain_size_phi} ({via})")
except UACPYError as exc:
    print(f"  EMODnet seabed: [skipped] {exc.message.splitlines()[0]}")

# ── The composite: map · TL · section, from one call ────────────────────────
fig, _ = uacpy.plot.plot_overview(
    env, grid, transect=(TRANSECT_START, TRANSECT_END),
    tl=tl, source=source, receiver=receiver,
    map_title="North Sea — Norwegian Trench (GEBCO)",
    tl_title=f"Transmission loss (Bellhop, {FREQ_HZ:g} Hz)",
    env_title="Range-dependent environment A→B",
    title="uacpy — real-world environment from GPS, modelled & plotted",
    map_kwargs=dict(contours=True, aspect=1,
                    coastline_resolution=COASTLINE_RES,
                    graticule=GRATICULE_DEG,
                    graticule_minor=GRATICULE_MINOR_DEG))
fig.savefig(OUT / 'example_37_realworld.png', dpi=130, bbox_inches='tight')
plt.close(fig)

# ── A modelled range-dependent LAYERED seabed, run through RAM ──────────────
if data.is_installed('sediment'):
    from uacpy.data import sediment_db
    grain = sediment_db.fetch_bottom_local_transect(
        TRANSECT_START, TRANSECT_END, n_points=BOTTOM_POINTS)

    # A fluid (no-shear) layered column per range node: the real surficial
    # sediment over a faster consolidated half-space. Fluid + layered +
    # range-varying is ramgeo's case, and its bathymetry-parallel layering
    # models it robustly. The elastic backend is deliberately not used:
    # Collins' rotated elastic PE is accurate only for WEAK range variation
    # (Collins 1991), and a strongly range-dependent elastic seabed is outside
    # every RAM backend uacpy vendors — which is why CRUST1.0 below is plotted
    # rather than propagated.
    columns = []
    for node in grain.ranges:
        half = grain.halfspace_at(range=float(node))
        columns.append(uacpy.SeabedColumn(
            layers=[uacpy.SedimentLayer(thickness=15.0,
                                        sound_speed=float(half.sound_speed),
                                        density=float(half.density),
                                        attenuation=float(half.attenuation))],
            halfspace=uacpy.BoundaryProperties(
                acoustic_type='half-space',
                sound_speed=float(half.sound_speed) + 350.0,
                density=float(half.density) + 0.4, attenuation=0.2)))
    layered_env = uacpy.Environment(
        name="Range-dependent layered seabed", bathymetry=env.bathymetry,
        ssp=env.ssp, bottom=uacpy.Bottom.from_columns(
            columns, ranges=np.asarray(grain.ranges, dtype=float)))

    model = uacpy.RAM(accuracy=1e-1)
    backend = model.select_backend(layered_env)
    layered_source = uacpy.Source(depths=SOURCE_DEPTH,
                                  frequencies=RDLB_FREQ_HZ)
    layered_receiver = uacpy.Receiver(
        depths=np.linspace(1, env.bathymetry.depth, 100),
        ranges=np.linspace(100.0, env.bathymetry.range_max, 200))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        layered_tl = model.run(layered_env, layered_source, layered_receiver,
                               run_mode=uacpy.RunMode.COHERENT_TL)
    print(f"  layered seabed: RAM→{backend} at {RDLB_FREQ_HZ:g} Hz, "
          f"{layered_tl.dB.shape} grid")
    # Surface any grid compromise rather than presenting the field as if it met
    # the requested accuracy budget.
    for warning in caught:
        text = ' '.join(str(warning.message).split())
        if 'accuracy budget' in text or 'not met' in text:
            print(f"  layered seabed: ACCURACY CAVEAT — {text}")

    fig, _ = uacpy.plot.plot_overview(
        layered_env, grid, transect=(TRANSECT_START, TRANSECT_END),
        tl=layered_tl, source=layered_source, receiver=layered_receiver,
        map_title="North Sea — Norwegian Trench (GEBCO)",
        tl_title=f"TL · range-dependent layered seabed (RAM→{backend}, "
                 f"{RDLB_FREQ_HZ:g} Hz)",
        env_title="Range-dependent LAYERED bottom A→B",
        title="uacpy — modelled range-dependent layered bottom",
        map_kwargs=dict(contours=True, aspect=1,
                        coastline_resolution=COASTLINE_RES,
                        graticule=GRATICULE_DEG,
                        graticule_minor=GRATICULE_MINOR_DEG))
    fig.savefig(OUT / 'example_37_rdlb.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
else:
    print("  layered seabed: [skipped] needs ./install.sh --data sediment")

# ── Two seabed descriptions of the same transect ────────────────────────────
if data.is_installed('sediment') and data.is_installed('crust1'):
    from uacpy.data import crust1_local, sediment_db
    grain = sediment_db.fetch_bottom_local_transect(
        TRANSECT_START, TRANSECT_END, n_points=BOTTOM_POINTS)
    crust = crust1_local.fetch_bottom_crust1_transect(
        TRANSECT_START, TRANSECT_END, n_points=BOTTOM_POINTS)
    grain_env = uacpy.Environment(name='grain-size',
                                  bathymetry=env.bathymetry, ssp=env.ssp,
                                  bottom=grain)
    crust_env = uacpy.Environment(name='crust1', bathymetry=env.bathymetry,
                                  ssp=env.ssp, bottom=crust)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    grain_env.plot(ax=axes[0], bottom_colorbar=False, data_source=None)
    axes[0].set_title("Grain-size half-space (surficial · high-freq)")
    crust_env.plot(ax=axes[1], bottom_colorbar=False, data_source=None)
    axes[1].set_title("CRUST1.0 layered elastic (deep · low-freq)")
    fig.suptitle("Seabed model comparison — grain size vs sediment layers",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT / 'example_37_bottom_comparison.png', dpi=130,
                bbox_inches='tight')
    plt.close(fig)

    # CRUST1.0 is elastic. plot_bottom_properties lays out every geoacoustic
    # property (cp, cs, ρ, αp, αs) as its own cross-section — the visual home
    # for shear, which env.plot() (cp only) omits.
    fig, _ = uacpy.plot.plot_bottom_properties(crust_env, data_source=None)
    fig.suptitle("CRUST1.0 seabed properties along A→B (cp · cs · ρ · αp · αs)",
                 fontsize=13, fontweight='bold')
    fig.savefig(OUT / 'example_37_bottom_properties.png', dpi=130,
                bbox_inches='tight')
    plt.close(fig)

    surficial = grain.halfspace_at(range=float(grain.ranges[0]))
    column = crust.columns[0]
    print(f"  seabed at A: grain-size half-space c_p="
          f"{surficial.sound_speed:.0f} m/s, ρ={surficial.density:.2f}, "
          f"α={surficial.attenuation:.2f} dB/λ, no shear")
    print(f"               CRUST1.0 {len(column.layers)} layer(s) over "
          f"basement c_p={column.halfspace.sound_speed:.0f}, "
          f"c_s={column.halfspace.shear_speed:.0f} m/s")
else:
    print("  seabed comparison: [skipped] needs "
          "./install.sh --data sediment,crust1")

# ── An under-ice Arctic transect, the same composite with a different map ───
if data.is_installed('seaice'):
    from uacpy.data import seaice_local
    from uacpy.data.sources import SOURCES
    ice_a, ice_b = (84.0, 0.0), (87.0, 40.0)      # central Arctic pack
    month = _dt.date.fromisoformat(DATE).month
    try:
        # The ice canopy is fetched along the transect too, so the marginal ice
        # zone rides on env.surface (a range-dependent Surface) and is drawn on
        # the environment panel. The solvers carry a single global top
        # boundary, so the TL run collapses it (with a warning) — the carrier
        # is for the visualisation.
        ice_env = data.fetch_environment(
            ice_a, date=DATE, transect_to=ice_b, n_points=TRANSECT_POINTS,
            bathymetry_sources=BATHY_SOURCE,
            range_dependent_ssp=True, ssp_n_points=SSP_POINTS,
            range_dependent_bottom=True, bottom_sources='local',
            bottom_n_points=BOTTOM_POINTS,
            range_dependent_surface=True, surface_n_points=BOTTOM_POINTS)
        ice_tl, ice_source, ice_receiver = run_tl(ice_env)
        ice_ranges, concentration = (
            seaice_local.fetch_sea_ice_concentration_transect(
                ice_a, ice_b, month=month, n_points=BOTTOM_POINTS))
        ice_grid = seaice_local.sea_ice_grid(month, hemi='N')

        for name, spot in [("Central Arctic (88 N, perennial)", (88.0, 0.0)),
                           ("Fram Strait (79 N, seasonal)", (79.0, -3.0))]:
            march = seaice_local.fetch_sea_ice_concentration(spot, month=3)
            september = seaice_local.fetch_sea_ice_concentration(spot, month=9)
            print(f"  {name}: March {march:.0%} ice, "
                  f"September {september:.0%} ice")

        # The same composite, with the pluggable left panel swapped for the
        # NSIDC sea-ice map.
        fig, _ = uacpy.plot.plot_overview(
            ice_env, (ice_grid,), map_fn=uacpy.plot.plot_sea_ice_map,
            map_kwargs=dict(hemi='N'), transect=(ice_a, ice_b), tl=ice_tl,
            source=ice_source, receiver=ice_receiver,
            sea_ice=(ice_ranges / 1000.0, concentration),
            data_source=list(ice_env.data_sources) + [SOURCES['seaice']],
            map_title="Arctic sea ice (NSIDC climatology)",
            tl_title=f"Transmission loss (Bellhop, {FREQ_HZ:g} Hz)",
            env_title=f"Under-ice environment A→B "
                      f"({int(np.nanmean(concentration) * 100)}% ice)",
            title="uacpy — under-ice environment, central Arctic")
        fig.savefig(OUT / 'example_37_sea_ice.png', dpi=130,
                    bbox_inches='tight')
        plt.close(fig)
    except Exception as exc:              # noqa: BLE001 — outside-world demo
        print(f"  sea-ice overview: [skipped] {str(exc).splitlines()[0]}")
else:
    print("  sea ice: [skipped] needs ./install.sh --data seaice")

if getattr(env, 'data_sources', None):
    print("\nData sources (attribution required):")
    print(data.citations(env))
