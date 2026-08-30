"""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLE 37: Real-world environment — map · transmission loss · section
═══════════════════════════════════════════════════════════════════════════════

OBJECTIVE:
    The full pipeline in one figure: fetch a regional GEBCO bathymetry grid and
    draw it on a coastline map (`plot_bathymetry_map`); build a fully
    **range-dependent** `Environment` along a transect (GEBCO seafloor, WOA23
    sound speed, fetched seafloor — all varying with range); run **Bellhop** for
    coherent transmission loss; and compose all three — map (left), TL (top
    right), environment (bottom right) — into a single 3-panel figure with one
    call to `uacpy.plot.plot_overview`.

    Region: the North Sea — a continental-shelf → Norwegian-Trench transect,
    chosen because the cached EMODnet seabed actually varies along range
    (sand ↔ mud ↔ coarse), so the **range-dependent bottom** is visible.

FEATURES DEMONSTRATED:
    ✓ uacpy.data.fetch_bathy_grid (regional bathymetry, land → NaN)
    ✓ uacpy.data.fetch_environment (range-dependent bathymetry + SSP + bottom + ice surface)
    ✓ uacpy.plot.plot_overview (map · transmission loss · environment, one call)
    ✓ Offline cache (./install.sh --data all): GEBCO, WOA23, grain-size,
      EMODnet, GlobSed, CRUST1.0 and the Natural Earth coastline backdrop
    ✓ uacpy.data.emodnet_local (offline EMODnet seabed, where covered)
    ✓ uacpy.data.seaice_local (NSIDC sea-ice concentration, high-lat int'l waters)
    ✓ Range-dependent LAYERED bottom modelled end-to-end: a fluid sediment
      column (grain-size surficial over a consolidated half-space) that varies
      along range, run through RAM. The dispatcher picks **ramgeo** for this
      env — a layered, range-varying fluid seabed is ramgeo's case, not
      mpiramS's — and the script prints the backend it actually selected
      rather than naming one here.
    ✓ Seabed model comparison — grain-size half-space vs CRUST1.0 layered
      elastic bottom (uacpy.data.crust1_local), the low-frequency description
    ✓ uacpy.data.citations (attribution for every source used)

NOTE: runs from the offline cache when installed (no network); otherwise hits
      the live APIs, falling back to representative synthetic data
      ("[offline fallback]") if a service is unreachable, so it always plots.
      The map coastline is drawn from the cached Natural Earth polygons when
      present, else fetched live.
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import os
from pathlib import Path

OUTPUT_DIR = Path(os.environ.get('UACPY_EXAMPLE_OUTPUT')
                  or Path(__file__).parent / 'output')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Repo root, so ``import uacpy`` resolves from a source checkout.
sys.path.insert(0, str(Path(__file__).parents[2]))

import datetime as _dt  # noqa: E402
import warnings  # noqa: E402
import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import uacpy  # noqa: E402
from uacpy import data  # noqa: E402
from uacpy.models import Bellhop, RAM, RunMode  # noqa: E402
from uacpy.core.environment import (  # noqa: E402
    SoundSpeedProfile, Bottom, SeabedColumn,
    SedimentLayer, BoundaryProperties,
)
from uacpy.core.exceptions import UACPYError  # noqa: E402


def _have(dataset):
    """True if a local dataset (./data_cache/<dataset>) is installed."""
    return data.is_installed(dataset)

# ── USER SETTINGS — tune everything here ─────────────────────────────────────
REGION_LAT, REGION_LON = (56.5, 62.0), (-2.0, 9.0)   # North Sea / Norwegian Trench
COASTLINE_RES = '50m'                                # Natural Earth coastline resolution: 110m|50m|10m
GRATICULE_DEG = 2.0                                  # labelled lat/lon grid spacing
GRATICULE_MINOR_DEG = 1.0                            # fine (unlabelled) grid spacing
# North Sea: continental shelf (~100 m) descending into the Norwegian Trench
# (~300 m), across EMODnet seabed that changes along range (sand/mud/coarse).
TRANSECT_START, TRANSECT_END = (61.0, 2.0), (58.0, 5.0)   # shelf → Norwegian Trench
TRANSECT_POINTS = 40                                 # bathymetry samples along A→B
SSP_POINTS = 5                                       # WOA sound-speed columns along A→B
BOTTOM_POINTS = 5                                    # seafloor samples along A→B
SOURCE_DEPTH = 100.0                                 # m
FREQ_HZ = 800.0                                      # Bellhop TL frequency
RDLB_FREQ_HZ = 50.0                                  # low-freq RAM TL through the layered seabed
DATE = '2026-01-15'                                  # for the WOA climatology month
FORCE_ONLINE = False                                 # True ⇒ ignore ./data_cache
# fetch_environment is cache-first by default: it uses the install-time cache
# (./install.sh --data all) when present — no network, no rate limits — and
# falls back to the live APIs for whatever isn't installed. (Passing
# *_sources='local' pins an axis to local data only, forbidding the network.)
BATHY_SOURCE = 'gebco'      # bathymetry source: 'gebco' (global) or 'gmrt'
GRID_SOURCE = 'local' if (not FORCE_ONLINE and _have('gebco')) else 'api'
# Map resolution: the local GEBCO grid has no rate limit, so go fine; the online
# OpenTopoData API is fair-use-capped, so stay coarse.
N_LAT = N_LON = 400 if GRID_SOURCE == 'local' else 10
_CACHED = [d for d in ('gebco', 'woa23', 'sediment', 'emodnet', 'coastline',
                       'globsed', 'crust1', 'diesing', 'seaice') if _have(d)]
print(f"  data source: cache-first ({BATHY_SOURCE})  ·  map grid {N_LAT}×{N_LON}")
print(f"  offline cache installed: {', '.join(_CACHED) if _CACHED else 'none'}")
# ─────────────────────────────────────────────────────────────────────────────


def _fetch_grid():
    try:
        lats, lons, depth = data.fetch_bathy_grid(REGION_LAT, REGION_LON,
                                            n_lat=N_LAT, n_lon=N_LON,
                                            source=GRID_SOURCE)
        print(f"  grid: ok  ({N_LAT}×{N_LON} via {GRID_SOURCE}, "
              f"{np.isnan(depth).sum()} land cells)")
        return lats, lons, depth
    except UACPYError as exc:
        print(f"  grid: [offline fallback] {exc.message.splitlines()[0]}")
        lats = np.linspace(*REGION_LAT, N_LAT)
        lons = np.linspace(*REGION_LON, N_LON)
        lon_m, lat_m = np.meshgrid(lons, lats)
        depth = 2800.0 * np.exp(-(((lat_m - 40) / 3) ** 2 + ((lon_m - 5) / 3) ** 2))
        depth[depth < 200] = np.nan
        return lats, lons, depth


def _fetch_env():
    """Fully range-dependent environment along the transect."""
    try:
        env = data.fetch_environment(
            TRANSECT_START, date=DATE, transect_to=TRANSECT_END,
            name="North Sea — Norwegian Trench transect", n_points=TRANSECT_POINTS,
            bathymetry_sources=BATHY_SOURCE,
            range_dependent_ssp=True, ssp_n_points=SSP_POINTS,
            range_dependent_bottom=True, bottom_sources='local',
            bottom_n_points=BOTTOM_POINTS)
        print(f"  environment: ok  ({env!r})")
        return env
    except UACPYError as exc:
        print(f"  environment: [offline fallback] {exc.message.splitlines()[0]}")
        rng = np.linspace(0.0, 370000.0, 5)
        z = np.array([0, 50, 100, 200, 300.0])
        base = np.array([1505, 1500, 1496, 1494, 1493.0])
        ssp = SoundSpeedProfile(depths=z, ranges=rng,
                                data=np.column_stack([base + 1.0 * np.sin(0.7 * k)
                                                      for k in range(5)]))
        # Shelf (~100 m) dipping into the Norwegian Trench (~300 m) and back.
        r = np.linspace(0, 370000, 40)
        bathy = np.column_stack([r, 110 + 190 * np.sin(np.pi * r / 370000) ** 2])
        return uacpy.Environment(name='North Sea transect (fallback)',
                                 bathymetry=bathy, ssp=ssp,
                                 bottom=data.bottom_from_class('silt'))


def _run_tl(env):
    """Coherent TL plus the source/receiver geometry it was computed on."""
    zmax = env.bathymetry.depth
    rmax = env.bathymetry.range_max
    source = uacpy.Source(depths=SOURCE_DEPTH, frequencies=FREQ_HZ)
    receiver = uacpy.Receiver(depths=np.linspace(1, zmax, 150),
                              ranges=np.linspace(100.0, rmax, 350))
    try:
        tl = Bellhop(verbose=False).run(env, source, receiver,
                                        run_mode=RunMode.COHERENT_TL)
        print(f"  TL: ok  ({tl.db.shape} grid, {FREQ_HZ:g} Hz, "
              f"source {SOURCE_DEPTH:g} m)")
    except Exception as exc:                       # model not built / run failed
        print(f"  TL: [skipped] {str(exc).splitlines()[0]}")
        tl = None
    return tl, source, receiver


def _report_emodnet_seabed():
    """EMODnet seabed at a European-seas point. The North Sea transect lies
    inside EMODnet coverage, so its cached 'local' bottom is EMODnet's Folk
    seabed (varying along range); this prints the substrate at one waypoint.

    Uses the offline EMODnet polygons (``--data emodnet``) when installed, else
    the live WFS — exactly what ``fetch_environment`` does cache-first."""
    point = (56.0, 3.0)                                  # central North Sea
    try:
        if not FORCE_ONLINE and _have('emodnet'):
            from uacpy.data import emodnet_local
            bp, via = emodnet_local.fetch_bottom_local(point), 'offline polygons'
        else:
            bp, via = data.fetch_bottom(point), 'live WFS'
        print(f"  EMODnet seabed @ North Sea {point}: {bp.acoustic_type}, "
              f"c_p={bp.sound_speed:.0f} m/s, ϕ={bp.grain_size_phi}  ({via})")
    except UACPYError as exc:
        print(f"  EMODnet seabed: [skipped] {exc.message.splitlines()[0]}")


def _report_sea_ice():
    """Sea-ice concentration at high-latitude **international** points (NSIDC
    monthly climatology). The North Sea (mid-latitude) is ice-free; the central Arctic and
    Fram Strait — both beyond any EEZ — show perennial vs seasonal ice cover,
    which would switch the surface boundary from free-surface to under-ice."""
    if not _have('seaice'):
        print("  sea ice: [skipped] needs ./install.sh --data seaice")
        return
    from uacpy.data import seaice_local
    print("  Sea-ice concentration (NSIDC climatology, international waters):")
    for name, pt in [("Central Arctic (88 N, perennial)", (88.0, 0.0)),
                     ("Fram Strait  (79 N, seasonal) ", (79.0, -3.0))]:
        try:
            mar = seaice_local.fetch_sea_ice_concentration(pt, month=3)
            sep = seaice_local.fetch_sea_ice_concentration(pt, month=9)
            print(f"     {name}: March {mar:.0%} ice · September {sep:.0%} ice")
        except UACPYError as exc:
            print(f"     {name}: [skipped] {exc.message.splitlines()[0]}")


def _sea_ice_overview(plt):
    """A second `plot_overview` composite — same layout as the main figure — for
    a **central-Arctic** under-ice transect (international waters, deep Nansen /
    Amundsen basins, ~95–100% winter ice): map · TL · environment, with the
    sea-ice cover drawn at the environment's surface."""
    if not _have('seaice'):
        print("  sea-ice overview: [skipped] needs ./install.sh --data seaice")
        return
    from uacpy.data import seaice_local
    from uacpy.data.sources import SOURCES
    A, B = (84.0, 0.0), (87.0, 40.0)                 # central Arctic pack, int'l
    month = _dt.date.fromisoformat(DATE).month
    try:
        env = data.fetch_environment(
            A, date=DATE, transect_to=B, n_points=TRANSECT_POINTS,
            bathymetry_sources=BATHY_SOURCE,
            range_dependent_ssp=True, ssp_n_points=SSP_POINTS,
            range_dependent_bottom=True, bottom_sources='local',
            bottom_n_points=BOTTOM_POINTS,
            # The sea-ice canopy is fetched along the transect too, so the
            # marginal ice zone rides on env.surface (a range-dependent
            # Surface) and is drawn on the environment panel. The solvers carry
            # a single global top boundary, so the TL run collapses it to one
            # boundary (with a warning) — the carrier is for the visualisation.
            range_dependent_surface=True, surface_n_points=BOTTOM_POINTS)
        tl, source, receiver = _run_tl(env)
        rng_m, conc = seaice_local.fetch_sea_ice_concentration_transect(
            A, B, month=month, n_points=BOTTOM_POINTS)
        ice_grid = seaice_local.sea_ice_grid(month, hemi='N')
    except Exception as exc:                         # noqa: BLE001 — robust demo
        print(f"  sea-ice overview: [skipped] {str(exc).splitlines()[0]}")
        return

    # Same composite as the main figure, but the pluggable left panel is the
    # NSIDC **sea-ice map** (map_fn=plot_sea_ice_map) instead of the depth map.
    fig, _ = uacpy.plot.plot_overview(
        env, (ice_grid,), map_fn=uacpy.plot.plot_sea_ice_map,
        map_kwargs=dict(hemi='N'), transect=(A, B), tl=tl, source=source,
        receiver=receiver, sea_ice=(rng_m / 1000.0, conc),
        data_source=list(env.data_sources) + [SOURCES['seaice']],
        map_title="Arctic sea ice (NSIDC climatology)",
        tl_title=f"Transmission loss (Bellhop, {FREQ_HZ:g} Hz)",
        env_title=f"Under-ice environment A→B ({int(np.nanmean(conc) * 100)}% ice)",
        title="uacpy — under-ice environment, central Arctic (international waters)")
    out = OUTPUT_DIR / 'example_37_sea_ice.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Sea-ice overview saved → {out}")


def _fluid_rdlb_from_grain(grain):
    """A fluid (no-shear) range-dependent LAYERED bottom from the real
    surficial sediment: a sediment layer (grain-size c_p/ρ/α) over a faster
    consolidated half-space, varying along range.

    Fluid (no shear) + layered + range-varying → the RAM dispatcher selects
    ramgeo, whose bathymetry-parallel layering models this seabed robustly.
    The elastic backend rams0.5 is deliberately not used here: Collins'
    rotated elastic PE is only accurate for *weak* range variation of the
    seismoacoustic parameters (Collins 1991), and a strongly range-dependent
    *elastic* seabed (e.g. CRUST1.0) is outside every RAM backend uacpy
    vendors — so the elastic CRUST1.0 column is shown as a plot in
    ``_compare_bottoms`` rather than propagated."""
    profiles = []
    for r in grain.ranges:
        bp = grain.halfspace_at(range=float(r))
        cp = float(bp.sound_speed)
        profiles.append(SeabedColumn(
            layers=[SedimentLayer(thickness=15.0, sound_speed=cp,
                                  density=float(bp.density),
                                  attenuation=float(bp.attenuation))],
            halfspace=BoundaryProperties(acoustic_type='half-space',
                                         sound_speed=cp + 350.0,
                                         density=float(bp.density) + 0.4,
                                         attenuation=0.2)))
    return Bottom.from_columns(
        profiles, ranges=np.asarray(grain.ranges, dtype=float))


def _rdlb_overview(env, grid, plt):
    """Composite for a MODELLED range-dependent layered bottom: build a fluid
    range-dependent layered seabed from the real surficial sediment and run
    RAM (the dispatcher selects ramgeo) for a low-frequency TL through the
    layered, range-varying bottom — map · TL · environment, same layout as
    the main figure.

    The grid RAM ends up on does not meet the requested Lytaev accuracy
    budget at this frequency; the run captures that warning and reports it
    rather than presenting the TL as converged."""
    if not _have('sediment'):
        print("  RDLB overview: [skipped] needs ./install.sh --data sediment")
        return
    from uacpy.data import sediment_db
    A, B = TRANSECT_START, TRANSECT_END
    try:
        grain = sediment_db.fetch_bottom_local_transect(A, B, n_points=BOTTOM_POINTS)
    except UACPYError as exc:
        print(f"  RDLB overview: [skipped] {exc.message.splitlines()[0]}")
        return

    env_rdlb = uacpy.Environment(name="Range-dependent layered seabed",
                                 bathymetry=env.bathymetry, ssp=env.ssp,
                                 bottom=_fluid_rdlb_from_grain(grain))
    zmax = env.bathymetry.depth
    rmax = env.bathymetry.range_max
    source = uacpy.Source(depths=SOURCE_DEPTH, frequencies=RDLB_FREQ_HZ)
    receiver = uacpy.Receiver(depths=np.linspace(1, zmax, 100),
                              ranges=np.linspace(100.0, rmax, 200))
    backend = 'unknown'
    try:
        model = RAM(verbose=False, accuracy=1e-1)
        backend = model.select_backend(env_rdlb)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            tl = model.run(env_rdlb, source, receiver,
                           run_mode=RunMode.COHERENT_TL)
        print(f"  RDLB TL: ok  ({tl.db.shape} grid, {RDLB_FREQ_HZ:g} Hz, "
              f"RAM→{backend}, range-dependent fluid layered bottom)")
        # Surface any grid/accuracy compromise rather than presenting the
        # field as if it met the requested budget.
        for w in caught:
            text = ' '.join(str(w.message).split())
            if 'accuracy budget' in text or 'not met' in text:
                print(f"  RDLB TL: ACCURACY CAVEAT — {text}")
    except Exception as exc:                       # model not built / run failed
        print(f"  RDLB TL: [skipped] {str(exc).splitlines()[0]}")
        tl = None

    fig, _ = uacpy.plot.plot_overview(
        env_rdlb, grid, transect=(A, B), tl=tl, source=source, receiver=receiver,
        map_title="North Sea — Norwegian Trench (GEBCO)",
        tl_title=f"TL · range-dependent layered seabed (RAM→{backend}, {RDLB_FREQ_HZ:g} Hz)",
        env_title="Range-dependent LAYERED bottom A→B",
        title="uacpy — modelled range-dependent layered bottom",
        map_kwargs=dict(contours=True, aspect=1, coastline_resolution=COASTLINE_RES,
                        graticule=GRATICULE_DEG, graticule_minor=GRATICULE_MINOR_DEG))
    out = OUTPUT_DIR / 'example_37_rdlb.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  RDLB overview saved → {out}")


def _compare_bottoms(env, plt):
    """Side-by-side seabed comparison: grain-size half-space vs CRUST1.0 layers.

    The same transect (shared bathymetry + SSP), two seabed descriptions:
    the surficial **grain-size half-space** (good at high frequency, a single
    fluid layer) and the **CRUST1.0 layered elastic** sediment column over
    crystalline basement (with shear — what a tens-of-Hz field actually needs).
    Both come from the offline cache (``--data sediment`` / ``--data crust1``).
    """
    from uacpy.data import crust1_local, sediment_db
    A, B = TRANSECT_START, TRANSECT_END
    if not (_have('sediment') and _have('crust1')):
        print("  bottom comparison: [skipped] needs "
              "./install.sh --data sediment,crust1")
        return
    try:
        grain = sediment_db.fetch_bottom_local_transect(A, B, n_points=BOTTOM_POINTS)
        crust = crust1_local.fetch_bottom_crust1_transect(A, B, n_points=BOTTOM_POINTS)
    except UACPYError as exc:
        print(f"  bottom comparison: [skipped] {exc.message.splitlines()[0]}")
        return

    env_grain = uacpy.Environment(name='grain-size', bathymetry=env.bathymetry,
                                  ssp=env.ssp, bottom=grain)
    env_crust = uacpy.Environment(name='crust1', bathymetry=env.bathymetry,
                                  ssp=env.ssp, bottom=crust)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    env_grain.plot(ax=axes[0], bottom_colorbar=False,
                   data_source=None)
    axes[0].set_title("Grain-size half-space  (surficial · high-freq)")
    env_crust.plot(ax=axes[1], bottom_colorbar=False,
                   data_source=None)
    axes[1].set_title("CRUST1.0 layered elastic  (deep · low-freq)")
    fig.suptitle("Seabed model comparison — grain size vs sediment layers",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    out = OUTPUT_DIR / 'example_37_bottom_comparison.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Bottom comparison saved → {out}")

    # The CRUST1.0 seabed is elastic (shear); plot_bottom_properties lays out
    # every geoacoustic property (cp, cs, ρ, αp, αs) as its own cross-section —
    # the visual home for shear & friends that env.plot() (cp-only) omits.
    figp, _ = uacpy.plot.plot_bottom_properties(env_crust, data_source=None)
    figp.suptitle("CRUST1.0 seabed properties along A→B "
                  "(cp · cs · ρ · αp · αs)", fontsize=13, fontweight='bold')
    out_p = OUTPUT_DIR / 'example_37_bottom_properties.png'
    figp.savefig(out_p, dpi=130, bbox_inches='tight')
    plt.close(figp)
    print(f"  Seabed properties saved → {out_p}")

    print("  Note: CRUST1.0 is a strongly range-dependent *elastic* seabed. No "
          "published\n        PE handles strong range-dependent *elastic* layering "
          "robustly (Collins 1991\n        flags it as future research); RAMGEO "
          "(now vendored) covers the range-\n        dependent *fluid* layered "
          "case, which is what the modelled TL\n        (example_37_rdlb.png) "
          "uses. So CRUST1.0 is shown here as a plot.")
    print(f"  Seabed @ A {A}:")
    grain_hs = grain.halfspace_at(range=float(grain.ranges[0]))
    print(f"     grain-size : half-space  c_p={grain_hs.sound_speed:.0f} m/s, "
          f"ρ={grain_hs.density:.2f}, α={grain_hs.attenuation:.2f} dB/λ  (no shear)")
    p0 = crust.columns[0]
    print(f"     CRUST1.0   : {len(p0.layers)} sediment layer(s) over basement "
          f"c_p={p0.halfspace.sound_speed:.0f} m/s, "
          f"c_s={p0.halfspace.shear_speed:.0f} m/s")
    for i, layer in enumerate(p0.layers, 1):
        print(f"        layer {i}: {layer.thickness:.0f} m  "
              f"c_p={layer.sound_speed:.0f}  c_s={layer.shear_speed:.0f}  "
              f"ρ={layer.density:.2f}")


def main():
    print("═" * 80)
    print(f"EXAMPLE 37: real-world environment → map · TL · section  ({DATE})")
    print("═" * 80)
    print("\nFetching:")
    lats, lons, depth = _fetch_grid()
    env = _fetch_env()
    tl, source, receiver = _run_tl(env)
    _report_emodnet_seabed()
    _report_sea_ice()

    # ── One composite figure via the library helper ─────────────────────────
    fig, _axes = uacpy.plot.plot_overview(
        env, (lats, lons, depth), transect=(TRANSECT_START, TRANSECT_END),
        tl=tl, source=source, receiver=receiver,
        map_title="North Sea — Norwegian Trench (GEBCO)",
        tl_title=f"Transmission loss (Bellhop, {FREQ_HZ:g} Hz)",
        env_title="Range-dependent environment A→B",
        title="uacpy — real-world environment from GPS, modelled & plotted",
        map_kwargs=dict(contours=True, aspect=1, coastline_resolution=COASTLINE_RES,
                        graticule=GRATICULE_DEG, graticule_minor=GRATICULE_MINOR_DEG))
    # plot_overview annotates the data provenance (Bathy/SSP/Seabed) by default.

    out = OUTPUT_DIR / 'example_37_realworld.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Composite figure saved → {out}")

    # ── Modelled range-dependent layered bottom (fluid column → ramgeo) ───────
    _rdlb_overview(env, (lats, lons, depth), plt)

    # ── Seabed model comparison: grain-size half-space vs CRUST1.0 layers ─────
    _compare_bottoms(env, plt)

    # ── Second composite: an under-ice central-Arctic scenario (international) ─
    _sea_ice_overview(plt)

    # ── Attribution for every source used ────────────────────────────────────
    if getattr(env, 'data_sources', None):
        print("\nData sources (attribution required):")
        print(data.citations(env))


if __name__ == '__main__':
    main()
