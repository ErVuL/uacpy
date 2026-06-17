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

    Region: the NE Atlantic / Reykjanes mid-ocean ridge (international waters).

FEATURES DEMONSTRATED:
    ✓ uacpy.data.fetch_grid (regional bathymetry, land → NaN)
    ✓ uacpy.data.fetch_environment (range-dependent bathymetry + SSP + bottom)
    ✓ uacpy.plot.plot_overview (map · transmission loss · environment, one call)
    ✓ Offline cache (./install.sh --data all): GEBCO, WOA23, grain-size,
      EMODnet, GlobSed, CRUST1.0 and the Natural Earth coastline backdrop
    ✓ uacpy.data.emodnet_local (offline EMODnet seabed, where covered)
    ✓ uacpy.data.seaice_local (NSIDC sea-ice concentration, high-lat int'l waters)
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
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402

import uacpy  # noqa: E402
from uacpy import data  # noqa: E402
from uacpy.data import _cache  # noqa: E402
from uacpy.models import Bellhop, RunMode  # noqa: E402
from uacpy.core.environment import SoundSpeedProfile  # noqa: E402
from uacpy.core.exceptions import UACPYError  # noqa: E402


def _have(dataset):
    """True if a local dataset (./data_cache/<dataset>) is installed."""
    try:
        _cache.require(dataset)
        return True
    except UACPYError:
        return False

# ── USER SETTINGS — tune everything here ─────────────────────────────────────
REGION_LAT, REGION_LON = (53.0, 61.0), (-33.0, -20.0)  # NE Atlantic / Reykjanes Ridge
COASTLINE_RES = '50m'                                # Natural Earth coastline resolution: 110m|50m|10m
GRATICULE_DEG = 2.0                                  # labelled lat/lon grid spacing
GRATICULE_MINOR_DEG = 1.0                            # fine (unlabelled) grid spacing
# International waters (mid-Atlantic, >500 km from any coast); the transect
# crosses the Reykjanes mid-ocean ridge.
TRANSECT_START, TRANSECT_END = (56.94, -28.50), (59.16, -27.40)   # ~255 km across the ridge
TRANSECT_POINTS = 40                                 # bathymetry samples along A→B
SSP_POINTS = 5                                       # WOA sound-speed columns along A→B
BOTTOM_POINTS = 5                                    # seafloor samples along A→B
SOURCE_DEPTH = 100.0                                 # m
FREQ_HZ = 800.0                                      # Bellhop TL frequency
DATE = '2026-01-15'                                  # for the WOA climatology month
FORCE_ONLINE = False                                 # True ⇒ ignore ./data_cache
# Auto-use the install-time offline cache (./install.sh --data all) when present:
# no network, no rate limits. Falls back to the live APIs otherwise.
OFFLINE = (not FORCE_ONLINE) and _have('gebco') and _have('woa23')
BATHY_SOURCE = 'local' if (not FORCE_ONLINE and _have('gebco')) else 'api'
#   'api' (GEBCO/OpenTopoData) · 'gmrt' (multibeam, higher-res) · 'local' (cache)
# Map resolution: the local GEBCO grid has no rate limit, so go fine; the online
# OpenTopoData API is fair-use-capped, so stay coarse.
N_LAT = N_LON = 400 if BATHY_SOURCE == 'local' else 10
_CACHED = [d for d in ('gebco', 'woa23', 'sediment', 'emodnet', 'coastline',
                       'globsed', 'crust1', 'diesing', 'seaice') if _have(d)]
print(f"  data source: {'OFFLINE cache' if OFFLINE else BATHY_SOURCE + ' (live)'}"
      f"  ·  map grid {N_LAT}×{N_LON}")
print(f"  offline cache installed: {', '.join(_CACHED) if _CACHED else 'none'}")
# ─────────────────────────────────────────────────────────────────────────────


def _fetch_grid():
    try:
        lats, lons, depth = data.fetch_grid(REGION_LAT, REGION_LON,
                                            n_lat=N_LAT, n_lon=N_LON,
                                            source=BATHY_SOURCE)
        print(f"  grid: ok  ({N_LAT}×{N_LON} via {BATHY_SOURCE}, "
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
            name="Reykjanes Ridge transect", n_points=TRANSECT_POINTS,
            bathymetry_source=BATHY_SOURCE, offline=OFFLINE,
            range_dependent_ssp=True, ssp_n_points=SSP_POINTS,
            range_dependent_bottom=True, bottom='auto', bottom_n_points=BOTTOM_POINTS)
        print(f"  environment: ok  ({env!r})")
        return env
    except UACPYError as exc:
        print(f"  environment: [offline fallback] {exc.message.splitlines()[0]}")
        rng = np.linspace(0.0, 445000.0, 5)
        z = np.array([0, 100, 500, 1500, 2500, 3100.0])
        base = np.array([1509, 1505, 1500, 1502, 1512, 1525.0])
        ssp = SoundSpeedProfile(depths=z, ranges=rng,
                                data=np.column_stack([base + 1.5 * np.sin(0.7 * k)
                                                      for k in range(5)]))
        bathy = np.column_stack([np.linspace(0, 445000, 40),
                                 np.linspace(2160, 3100, 40)])
        return uacpy.Environment(name='Reykjanes transect (fallback)',
                                 bathymetry=bathy, ssp=ssp,
                                 bottom=data.bottom_from_class('silt'))


def _run_tl(env):
    """Coherent TL plus the source/receiver geometry it was computed on."""
    zmax = float(np.max(np.asarray(env.bathymetry)[:, 1]))
    rmax = float(np.max(np.asarray(env.bathymetry)[:, 0]))
    source = uacpy.Source(depths=SOURCE_DEPTH, frequencies=FREQ_HZ)
    receiver = uacpy.Receiver(depths=np.linspace(1, zmax, 150),
                              ranges=np.linspace(100.0, rmax, 350))
    try:
        tl = Bellhop(verbose=False).run(env, source, receiver,
                                        run_mode=RunMode.COHERENT_TL)
        print(f"  TL: ok  ({tl.tl.shape} grid, {FREQ_HZ:g} Hz, "
              f"source {SOURCE_DEPTH:g} m)")
    except Exception as exc:                       # model not built / run failed
        print(f"  TL: [skipped] {str(exc).splitlines()[0]}")
        tl = None
    return tl, source, receiver


def _report_emodnet_seabed():
    """EMODnet seabed at a European-seas point (Reykjanes is outside EMODnet
    coverage, so the transect's 'auto' bottom uses the global grain-size DB).

    Uses the offline EMODnet polygons (``--data emodnet``) when installed, else
    the live WFS — exactly what ``fetch_environment(offline=...)`` selects."""
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
    monthly climatology). Reykjanes (57 N) is ice-free; the central Arctic and
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
    month = int(DATE.split('-')[1])
    try:
        env = data.fetch_environment(
            A, date=DATE, transect_to=B, n_points=TRANSECT_POINTS,
            bathymetry_source=BATHY_SOURCE, offline=OFFLINE,
            range_dependent_ssp=True, ssp_n_points=SSP_POINTS,
            range_dependent_bottom=True, bottom='auto', bottom_n_points=BOTTOM_POINTS)
        tl, source, receiver = _run_tl(env)
        rng_m, conc = seaice_local.fetch_sea_ice_concentration_transect(
            A, B, month=month, n_points=BOTTOM_POINTS)
        ice_grid = seaice_local.sea_ice_grid(month, hemi='N')
    except (UACPYError, Exception) as exc:           # noqa: BLE001 — robust demo
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
        suptitle="uacpy — under-ice environment, central Arctic (international waters)")
    out = OUTPUT_DIR / 'example_37_sea_ice.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Sea-ice overview saved → {out}")


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
    uacpy.plot.plot_environment(env_grain, ax=axes[0], bottom_colorbar=False,
                                data_source=None)
    axes[0].set_title("Grain-size half-space  (surficial · high-freq)")
    uacpy.plot.plot_environment(env_crust, ax=axes[1], bottom_colorbar=False,
                                data_source=None)
    axes[1].set_title("CRUST1.0 layered elastic  (deep · low-freq)")
    fig.suptitle("Seabed model comparison — grain size vs sediment layers",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    out = OUTPUT_DIR / 'example_37_bottom_comparison.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"  Bottom comparison saved → {out}")

    print(f"  Seabed @ A {A}:")
    print(f"     grain-size : half-space  c_p={grain.sound_speed[0]:.0f} m/s, "
          f"ρ={grain.density[0]:.2f}, α={grain.attenuation[0]:.2f} dB/λ  (no shear)")
    p0 = crust.profiles[0]
    print(f"     CRUST1.0   : {len(p0.layers)} sediment layer(s) over basement "
          f"c_p={p0.halfspace.sound_speed:.0f} m/s, "
          f"c_s={p0.halfspace.shear_speed:.0f} m/s")
    for i, layer in enumerate(p0.layers, 1):
        print(f"        layer {i}: {layer.thickness:.0f} m  "
              f"c_p={layer.sound_speed:.0f}  c_s={layer.shear_speed:.0f}  "
              f"ρ={layer.density:.2f}")


def main():
    plt = uacpy.plot.plt
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
        map_title="NE Atlantic — Reykjanes Ridge (GEBCO)",
        tl_title=f"Transmission loss (Bellhop, {FREQ_HZ:g} Hz)",
        env_title="Range-dependent environment A→B",
        suptitle="uacpy — real-world environment from GPS, modelled & plotted",
        map_kwargs=dict(contours=True, aspect=1, coastline_resolution=COASTLINE_RES,
                        graticule=GRATICULE_DEG, graticule_minor=GRATICULE_MINOR_DEG))
    # plot_overview annotates the data provenance (Bathy/SSP/Seabed) by default.

    out = OUTPUT_DIR / 'example_37_realworld.png'
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Composite figure saved → {out}")

    # ── Seabed model comparison: grain-size half-space vs CRUST1.0 layers ─────
    _compare_bottoms(env, plt)

    # ── Second composite: an under-ice Fram Strait scenario (international) ────
    _sea_ice_overview(plt)

    # ── Attribution for every source used ────────────────────────────────────
    if getattr(env, 'data_sources', None):
        print("\nData sources (attribution required):")
        print(data.citations(env))


if __name__ == '__main__':
    main()
