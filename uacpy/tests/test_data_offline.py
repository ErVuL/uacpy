import warnings
import datetime as _dt
"""Tests for the offline local-cache backend (uacpy.data._cache + readers).

Builds a synthetic ``$UACPY_DATA_CACHE`` (tiny GEBCO/WOA23 NetCDF + sediment
CSVs) so the GEBCO/WOA23/DECK41 readers (``source='local'``) and a cache-first
``fetch_environment`` (gebco/woa23 + ``bottom_sources='grainsize'``) run from the
cache with no network. Skipped where netCDF4 is unavailable (the grids need it).
"""

import re

import numpy as np
import pytest

from uacpy.data import _cache

netCDF4 = pytest.importorskip('netCDF4')

import uacpy.data as data
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data import (
    crust1_local, emodnet_local, gebco_local, pelagic,
    sediment_db, sound_speed, woa23_local,
)
from uacpy.tests._cache_builders import (
    _FILL, _write_crust1, _write_gebco, _write_globsed, _write_sediment,
    _write_woa,
)


def _write_emodnet(cache):
    """One synthetic Folk-5 Sand polygon over the North Sea; skipped sans shapely.

    Deliberately *not* over the grain-size test point (30.5, -40.5) so 'auto'
    there exercises the EMODnet-miss → grain-size fallthrough.
    """
    try:
        import pickle
        import shapely
    except ImportError:                              # pragma: no cover
        return
    edir = cache / 'emodnet'; edir.mkdir(parents=True)
    poly = shapely.geometry.box(2.0, 54.0, 3.0, 56.0)       # lon0,lat0,lon1,lat1
    with open(edir / 'seabed_substrate.pkl', 'wb') as fh:
        pickle.dump({'codes': [2], 'wkb': [shapely.to_wkb(poly)]}, fh)


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """A synthetic, fully-populated offline cache wired via $UACPY_DATA_CACHE."""
    root = tmp_path / 'data_cache'
    monkeypatch.setenv('UACPY_DATA_CACHE', str(root))
    # Module-level open-once caches are keyed by path; clear for isolation.
    _cache.invalidate_grids()
    woa23_local._DATASETS.clear()
    sediment_db._SAMPLES.clear()
    emodnet_local._INDEX.clear()
    _write_gebco(root, land_at=(0.0, 0.0))
    _write_woa(root, periods=('00', '03'))
    _write_sediment(root)
    _write_emodnet(root)
    return root


@pytest.fixture
def seismic_cache(tmp_path, monkeypatch):
    """Cache with the GlobSed + CRUST1.0 layered-bottom stack (and GEBCO/WOA).

    Kept separate from ``cache`` so the bulky GlobSed grid / CRUST1.0 columns are
    built only for the one capstone test that needs the layered seabed, not on
    every offline test.
    """
    root = tmp_path / 'seismic_cache'
    monkeypatch.setenv('UACPY_DATA_CACHE', str(root))
    _cache.invalidate_grids()
    woa23_local._DATASETS.clear()
    crust1_local._MODEL.clear()
    _cache.invalidate_grids()
    _write_gebco(root)
    _write_woa(root, periods=('00', '03'))
    _write_globsed(root)
    _write_crust1(root)
    return root


# ── cache resolver ──────────────────────────────────────────────────────────

def test_cache_root_honors_env(tmp_path, monkeypatch):
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'x'))
    assert _cache.cache_root() == tmp_path / 'x'


def test_require_missing_names_install_flag(tmp_path, monkeypatch):
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    with pytest.raises(ConfigurationError, match='install.sh --data gebco'):
        _cache.require('gebco')


# ── GEBCO local ─────────────────────────────────────────────────────────────

def test_gebco_point_depth_and_land(cache):
    assert gebco_local.point_depth((10.0, 20.0)) == 1500.0
    with pytest.raises(DataFetchError, match='land'):
        gebco_local.point_depth((0.0, 0.0))


def test_gebco_region_grid_latitudes_ascend_for_descending_range(cache):
    ref_lats, _ref_lons, ref_depth = gebco_local.region_grid((-3, 3), (-3, 3), 7, 7)
    lats, lons, depth = gebco_local.region_grid((3, -3), (-3, 3), 7, 7)
    assert np.all(np.diff(lats) > 0)
    np.testing.assert_array_equal(lats, ref_lats)
    np.testing.assert_array_equal(depth, ref_depth)


def test_gebco_region_grid_marks_land_nan(cache):
    lats, lons, depth = gebco_local.region_grid((-3, 3), (-3, 3), 7, 7)
    assert depth.shape == (7, 7)
    assert np.isnan(depth).sum() == 1            # the single land cell
    assert np.nanmin(depth) == 1500.0


def test_gebco_masked_cell_raises(tmp_path, monkeypatch):
    # A _FillValue/masked cell must raise DataFetchError, not coerce a fill
    # sentinel into a depth. GEBCO_2025 itself declares no _FillValue (int16
    # elevation, every cell valued), so this guards the shared NetcdfGrid.cell
    # path that the fill-bearing sibling grids (WOA23, GLODAP) depend on, and a
    # future GEBCO release that starts masking. The sentinel value is arbitrary.
    root = tmp_path / 'masked_cache'
    monkeypatch.setenv('UACPY_DATA_CACHE', str(root))
    _cache.invalidate_grids()
    gdir = root / 'gebco'; gdir.mkdir(parents=True)
    lat = np.arange(-90, 91, 1.0)
    lon = np.arange(-180, 180, 1.0)
    elev = np.full((lat.size, lon.size), -1500.0)
    ds = netCDF4.Dataset(gdir / 'GEBCO_2025.nc', 'w')
    ds.createDimension('lat', lat.size); ds.createDimension('lon', lon.size)
    ds.createVariable('lat', 'f8', ('lat',))[:] = lat
    ds.createVariable('lon', 'f8', ('lon',))[:] = lon
    var = ds.createVariable('elevation', 'f4', ('lat', 'lon'),
                            fill_value=_FILL)
    var[:] = elev
    var[120, 220] = _FILL                         # masked cell at (30, 40)
    ds.close()
    with pytest.raises(DataFetchError, match='fill / masked'):
        gebco_local.point_depth((30.0, 40.0))


def test_gebco_missing_variable_raises_typed(tmp_path, monkeypatch):
    # A GEBCO file whose schema changed must surface as DataFetchError naming
    # the install flag, like every sibling grid — not a bare KeyError.
    root = tmp_path / 'schema_cache'
    monkeypatch.setenv('UACPY_DATA_CACHE', str(root))
    _cache.invalidate_grids()
    gdir = root / 'gebco'; gdir.mkdir(parents=True)
    ds = netCDF4.Dataset(gdir / 'GEBCO_2025.nc', 'w')
    ds.createDimension('lat', 3); ds.createDimension('lon', 3)
    ds.createVariable('lat', 'f8', ('lat',))[:] = [-1.0, 0.0, 1.0]
    ds.createVariable('lon', 'f8', ('lon',))[:] = [-1.0, 0.0, 1.0]
    ds.createVariable('bathymetry', 'f4', ('lat', 'lon'))[:] = -1500.0
    ds.close()
    with pytest.raises(DataFetchError, match='install.sh --data gebco'):
        gebco_local.point_depth((0.0, 0.0))


def test_gebco_region_across_the_antimeridian(cache):
    # An eastward range crossing 180° splits into a high-index and a low-index
    # column run. Reading min..max would pull the whole globe; the result must
    # equal the point-wise depths and each slab must stay narrow.
    grid = gebco_local._grid()
    widths = []
    raw = grid._elev

    class _Spy:
        def __getitem__(self, key):
            block = raw[key]
            widths.append(np.shape(block)[1])
            return block

    grid._elev = _Spy()
    try:
        lats, lons, depth = gebco_local.region_grid((-3, 3), (179, -179), 7, 7)
    finally:
        grid._elev = raw
    assert lons[0] == 179.0 and lons[-1] == pytest.approx(-179.0)
    expected = np.array([[gebco_local.point_depth((la, lo)) for lo in lons]
                         for la in lats])
    assert np.allclose(depth, expected)
    # Two runs, each a handful of columns — never the 360-column full width.
    assert len(widths) == 2 and max(widths) <= 10


def test_gebco_region_reads_only_requested_rows_and_cols(cache):
    # Regression: a coarse request over a wide box must read only the
    # requested rows/cols (orthogonal indexing), never the min..max bounding
    # slab — on the real 43 200 × 86 400 grid that slab is tens of GB for a
    # 50×50 global request. Values must still match the point-wise depths.
    grid = gebco_local._grid()
    shapes = []
    raw = grid._elev

    class _Spy:
        def __getitem__(self, key):
            block = raw[key]
            shapes.append(np.shape(block))
            return block

    grid._elev = _Spy()
    try:
        # Latitudes chosen so no waypoint hits the fixture's land cell (0, 0).
        lats, lons, depth = gebco_local.region_grid((-58, 62), (-150, 150), 7, 9)
    finally:
        grid._elev = raw
    # One contiguous column run; the block is exactly the 7 unique rows by the
    # 9 unique cols — not the 121 × 301 bounding slab of the 1° fixture.
    assert shapes == [(7, 9)]
    expected = np.array([[gebco_local.point_depth((la, lo)) for lo in lons]
                         for la in lats])
    assert np.allclose(depth, expected)


def test_bathymetry_sources_local(cache):
    assert data.fetch_bathy((12.0, 34.0), source='local') == 1500.0
    t = data.fetch_bathy_transect((1.0, 1.0), (2.0, 2.0), n_points=4, source='local')
    assert t.shape == (4, 2) and np.all(t[:, 1] == 1500.0)


def test_bathymetry_bad_source_raises():
    with pytest.raises(ConfigurationError, match='source'):
        data.fetch_bathy((1.0, 2.0), source='nope')


# ── WOA23 local ─────────────────────────────────────────────────────────────

def test_woa_local_ssp(cache):
    prof = sound_speed.fetch_ssp((30.5, -40.5), source='local')
    assert prof.depths.tolist() == [0.0, 50.0, 100.0, 500.0, 1000.0]
    assert np.all((1450 < prof.data[:, 0]) & (prof.data[:, 0] < 1560))


def test_woa_local_ts_column(cache):
    # The local column feeds the *same* UNESCO conversion as OPeNDAP; check the
    # raw T/S surface values come through untouched.
    z, t, s = sound_speed.fetch_ts_profile((30.5, -40.5), source='local')
    assert t[0] == 18.0 and s[0] == 36.0 and z[0] == 0.0


def test_woa_local_reads_fill_through_the_mask(tmp_path, monkeypatch):
    # The reader must resolve _FillValue via the netCDF mask, not rely on the
    # fill being numerically huge: a product filling with -999 would otherwise
    # enter the column as a real temperature.
    root = tmp_path / 'fill_cache'
    monkeypatch.setenv('UACPY_DATA_CACHE', str(root))
    woa23_local.close()
    wdir = root / 'woa23'; wdir.mkdir(parents=True)
    depth = np.array([0, 50, 100, 500, 1000.0])
    for var, vals in (('t00', [18, 16, 13, -999.0, -999.0]),
                      ('s00', [36, 36.1, 36.2, -999.0, -999.0])):
        ds = netCDF4.Dataset(wdir / f'woa23_decav_{var}_01.nc', 'w')
        for d, n in [('time', 1), ('depth', depth.size), ('lat', 180),
                     ('lon', 360)]:
            ds.createDimension(d, n)
        ds.createVariable('depth', 'f4', ('depth',))[:] = depth
        name = 't_an' if var[0] == 't' else 's_an'
        v = ds.createVariable(name, 'f4', ('time', 'depth', 'lat', 'lon'),
                              fill_value=-999.0)
        v[:] = np.ma.masked_equal(np.full((1, depth.size, 180, 360), -999.0),
                                  -999.0)
        v[0, :, 120, 139] = np.ma.masked_equal(np.asarray(vals), -999.0)
        ds.close()
    z, t, s = sound_speed.fetch_ts_profile((30.5, -40.5), source='local')
    assert z.tolist() == [0.0, 50.0, 100.0]        # truncated at the fill
    assert t.min() > 0.0 and s.min() > 0.0         # no -999 admitted as data


def test_woa_close_releases_the_handles(cache):
    sound_speed.fetch_ssp((30.5, -40.5), source='local')
    assert woa23_local._DATASETS                   # opened once, kept open
    woa23_local.close()
    assert woa23_local._DATASETS == {}
    # The next read reopens transparently.
    assert sound_speed.fetch_ssp((30.5, -40.5), source='local').depths[0] == 0.0


# ── sediment local ──────────────────────────────────────────────────────────

def test_sediment_sample_and_bottom(cache):
    s = sediment_db.fetch_sediment_sample((30.51, -40.51))
    assert s['phi'] == 3.0 and s['distance_km'] < 5.0
    bp = sediment_db.fetch_bottom_local((30.51, -40.51))
    assert bp.acoustic_type == 'half-space' and bp.grain_size_phi == 3.0


def test_sediment_deck41_lithology_path(cache):
    bp = sediment_db.fetch_bottom_local((44.01, 8.01))
    assert bp.grain_size_phi == 1.5            # 'Sand' → ϕ 1.5


def test_download_sediment_db_normalizes(tmp_path, monkeypatch):
    # Build a minimal G00127-style tarball (sample.txt + phi.txt) and stub the
    # download; the normalizer must join lat/lon with a weighted-mean ϕ.
    import io
    import tarfile

    sample = ("mggid\tsample\tlat\tlon\n"
              "A\t1\t40.0\t-60.0\n"
              "B\t1\t10.0\t20.0\n")
    # A: 50% in ϕ[1,2] + 50% in ϕ[3,4] → mean 2.5 ; B: all in ϕ[7,8] → 7.5
    phi = ("mggid\tsample\tinterval\tlower_phi_limit\tupper_phi_limit\tweight_percent\n"
           "A\t1\t0\t1\t2\t50\n"
           "A\t1\t0\t3\t4\t50\n"
           "B\t1\t0\t7\t8\t100\n")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tf:
        for name, text in [('g00127sample.txt', sample), ('g00127phi.txt', phi)]:
            data = text.encode()
            info = tarfile.TarInfo(name); info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    monkeypatch.setattr(sediment_db, 'http_get', lambda url, **kw: buf.getvalue())

    out = sediment_db.download_sediment_db(cache_dir=str(tmp_path))
    import csv
    rows = {r['latitude']: r for r in csv.DictReader(open(out))}
    assert float(rows['40.0']['mean_phi']) == pytest.approx(2.5)
    assert float(rows['10.0']['mean_phi']) == pytest.approx(7.5)


def test_sediment_too_far_raises(cache):
    with pytest.raises(DataFetchError, match='km away'):
        sediment_db.fetch_sediment_sample((-80.0, 150.0), max_distance_km=100)


def test_sediment_transect(cache):
    rdb = sediment_db.fetch_bottom_local_transect(
        (30.5, -40.5), (43.0, 7.0), n_points=4)
    assert rdb.halfspace_sound_speed.shape == (4,)


# ── capstone from the cache ───────────────────────────────────────────────────
# These pin cache-only sources (gebco/woa23 cache-first, bottom_sources='grainsize'
# /'crust1'), so every fetch resolves locally with no network call.

def test_fetch_environment_from_cache(cache):
    # Cache-first: gebco/woa23 resolve to their local twin, grainsize to the
    # installed sediment DB — no network, fully reproducible.
    env = data.fetch_environment((30.5, -40.5), bottom_sources='grainsize')
    assert env.depth == 1500.0
    assert env.ssp.n_depths >= 5
    assert env.bottom.columns[0].halfspace.grain_size_phi == 3.0
    assert [s.source.id for s in env.data_sources] == ['gebco', 'woa23', 'grainsize']


def test_fetch_environment_cache_preset(cache):
    # *_sources='local' resolves every axis from local data only (no network):
    # bathy→GEBCO-local, ssp→WOA23-local, bottom→cache chain (EMODnet local miss
    # here → grain-size DB).
    env = data.fetch_environment((30.5, -40.5), bathymetry_sources='local',
                                 ssp_sources='local', bottom_sources='local')
    assert env.depth == 1500.0
    assert env.bottom.columns[0].halfspace.grain_size_phi == 3.0
    assert [s.source.id for s in env.data_sources] == ['gebco', 'woa23', 'grainsize']


def test_cache_preset_never_hits_network(tmp_path, monkeypatch):
    # 'local' must never reach the network. With an empty cache and literal
    # bathy/ssp, the bottom 'local' chain falls through every cached source to
    # the pelagic last resort, which classifies off the *supplied* bathymetry
    # instead of looking up its own — so the environment builds with no fetch.
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    _cache.invalidate_grids()
    from uacpy.data import bathymetry
    monkeypatch.setattr(bathymetry, '_fetch_depths', lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("ssp_sources/bottom_sources='local' hit the live API")))
    env = data.fetch_environment((30.5, -40.5), bathymetry=3000.0, ssp=1500.0,
                                 bottom_sources='local')
    assert [s.source.id for s in env.data_sources] == ['pelagic']
    # 3000 m is above the CCD at this latitude → calcareous ooze (ϕ 7.5), and
    # ϕ 7.5 interpolates the Hamilton & Bachman density table between clayey
    # silt (7.13, 1.484) and silty clay (8.80, 1.480) → 1.4831.
    assert env.bottom.columns[0].halfspace.density == pytest.approx(1.483, abs=1e-3)


def test_pelagic_without_a_supplied_depth_still_needs_the_cache(tmp_path,
                                                                monkeypatch):
    # The depth lookup is only skipped when the caller supplies one. Called
    # directly with cache_only and no depth=, pelagic must still fail fast on
    # the missing GEBCO cache rather than falling back to the live API.
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    _cache.invalidate_grids()
    from uacpy.data import bathymetry
    monkeypatch.setattr(bathymetry, '_fetch_depths', lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("cache_only pelagic hit the live API")))
    with pytest.raises(ConfigurationError, match='install.sh --data'):
        pelagic.fetch_bottom_pelagic((30.5, -40.5), cache_only=True)


def test_cache_preset_ssp_no_cache_raises(tmp_path, monkeypatch):
    # ssp_sources='local' with no WOA23 cache fails fast (no OpenDAP fallback).
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    woa23_local._DATASETS.clear()
    with pytest.raises(ConfigurationError, match='install.sh --data woa23'):
        data.fetch_environment((30.5, -40.5), bathymetry=3000.0,
                               ssp_sources='local')


def test_fetch_environment_crust1_pulls_globsed(seismic_cache):
    # bottom_sources='crust1' rescales its column with GlobSed by default, so
    # both CRUST1.0 and GlobSed appear in the environment's provenance.
    env = data.fetch_environment((30.5, -40.5), bottom_sources='crust1')
    ids = [s.source.id for s in env.data_sources]
    assert ids == ['gebco', 'woa23', 'crust1', 'globsed']
    assert env.bottom.columns[0].total_thickness() == pytest.approx(500.0)


def test_fetch_environment_sea_ice(cache, monkeypatch):
    # surface_sources='seaice' sets the surface from the climatological
    # concentration; an ice-covered point gets an elastic canopy + provenance.
    from uacpy.data import seaice_local
    monkeypatch.setattr(seaice_local, 'fetch_sea_ice_concentration',
                        lambda pt, date=None, *, month=None: 0.85)
    env = data.fetch_environment((30.5, -40.5), date='2026-03-15',
                                 bottom_sources='grainsize',
                                 surface_sources='seaice')
    assert env.surface.acoustic_type == 'half-space'
    assert env.surface.shear_speed == 1800.0 and env.has_elastic_surface
    assert 'seaice' in [s.source.id for s in env.data_sources]


def test_fetch_environment_sea_ice_open_water(cache, monkeypatch):
    # Below the ice-edge → free surface untouched, no 'seaice' provenance.
    from uacpy.data import seaice_local
    monkeypatch.setattr(seaice_local, 'fetch_sea_ice_concentration',
                        lambda pt, date=None, *, month=None: 0.0)
    env = data.fetch_environment((30.5, -40.5), date='2026-03-15',
                                 bottom_sources='grainsize',
                                 surface_sources='auto')
    assert env.surface.acoustic_type == 'vacuum'
    assert 'seaice' not in [s.source.id for s in env.data_sources]


def test_fetch_environment_sea_ice_requires_date(cache):
    with pytest.raises(ConfigurationError, match="surface_sources='seaice' needs date"):
        data.fetch_environment((30.5, -40.5), surface_sources='seaice')


def test_fetch_environment_sea_ice_transect_classifies_each_zone(monkeypatch):
    # Concentration keyed on latitude — 0.9 pack ice at/above 83°N, 0.0 open
    # water below — with literal bathymetry/SSP so only the surface fetches.
    # The waypoints at 85, 83.33, 81.67, 80°N each classify from their own
    # zone: canopy, canopy, open water, open water.
    from uacpy.data import seaice_local
    monkeypatch.setattr(seaice_local, 'fetch_sea_ice_concentration',
                        lambda pt, date=None, *, month=None:
                        0.9 if pt[0] >= 83.0 else 0.0)
    env = data.fetch_environment((85.0, 0.0), transect_to=(80.0, 0.0),
                                 date='2026-03-15', bathymetry=3000.0,
                                 ssp=1500.0, surface_sources='seaice',
                                 surface_n_points=4)
    assert [bp.acoustic_type for bp in env.surface.properties] == \
        ['half-space', 'half-space', 'vacuum', 'vacuum']
    assert env.surface.at(range=0.0).shear_speed == 1800.0
    assert [s.source.id for s in env.data_sources] == ['seaice']


def test_fetch_environment_sea_ice_straddles_the_ice_edge_threshold(monkeypatch):
    # 15.1 % — just above the NSIDC 15 % ice-edge → elastic canopy with
    # 'seaice' provenance.
    from uacpy.data import seaice_local
    monkeypatch.setattr(seaice_local, 'fetch_sea_ice_concentration',
                        lambda pt, date=None, *, month=None: 0.151)
    env = data.fetch_environment((30.5, -40.5), date='2026-03-15',
                                 bathymetry=3000.0, ssp=1500.0,
                                 surface_sources='seaice')
    assert env.surface.acoustic_type == 'half-space'
    assert 'seaice' in [s.source.id for s in env.data_sources]
    # 14.9 % — just below → the free surface stands, no 'seaice' provenance.
    monkeypatch.setattr(seaice_local, 'fetch_sea_ice_concentration',
                        lambda pt, date=None, *, month=None: 0.149)
    env = data.fetch_environment((30.5, -40.5), date='2026-03-15',
                                 bathymetry=3000.0, ssp=1500.0,
                                 surface_sources='seaice')
    assert env.surface.acoustic_type == 'vacuum'
    assert 'seaice' not in [s.source.id for s in env.data_sources]


def test_fetch_environment_sea_ice_canopy_full_property_set(monkeypatch):
    # A canopy point carries the complete ice canopy of core.constants —
    # elastic half-space, c_p 3500 m/s, c_s 1800 m/s, ρ 0.9 g/cm³,
    # α_p 0.4 dB/λ, α_s 1.0 dB/λ, roughness 0, no grain size.
    from uacpy.data import seaice_local
    monkeypatch.setattr(seaice_local, 'fetch_sea_ice_concentration',
                        lambda pt, date=None, *, month=None: 0.85)
    env = data.fetch_environment((30.5, -40.5), date='2026-03-15',
                                 bathymetry=3000.0, ssp=1500.0,
                                 surface_sources='seaice')
    s = env.surface
    assert s.acoustic_type == 'half-space'
    assert (s.sound_speed, s.shear_speed, s.density) == (3500.0, 1800.0, 0.9)
    assert (s.attenuation, s.shear_attenuation) == (0.4, 1.0)
    assert s.roughness == 0.0 and s.grain_size_phi is None


def test_offline_emodnet_local_bottom(cache):
    pytest.importorskip('shapely')
    bp = emodnet_local.fetch_bottom_local((55.0, 2.5))     # inside the polygon
    assert bp.acoustic_type == 'half-space' and bp.grain_size_phi == 2.0
    with pytest.raises(DataFetchError, match='European seas only'):
        emodnet_local.fetch_seabed_local((30.5, -40.5))    # outside it


def test_offline_emodnet_missing_names_flag(tmp_path, monkeypatch):
    pytest.importorskip('shapely')
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    emodnet_local._INDEX.clear()
    with pytest.raises(ConfigurationError, match='install.sh --data emodnet'):
        emodnet_local.fetch_bottom_local((56.0, 3.0))


def test_every_backend_memo_is_registered_for_invalidation():
    # invalidate_grids() is the single "drop everything" entry point, so a
    # backend that memoises on its own must register that memo. Scan the layer
    # for module-level cache dicts and require a register_cache call alongside.
    import pathlib
    import uacpy.data
    root = pathlib.Path(uacpy.data.__file__).parent
    # Matches the shape every backend memo currently uses: a module-level
    # `_UPPER_NAME = {}` (optionally annotated `: dict`). A digit in the name or
    # a parameterised annotation would slip past — widen this if one appears.
    declares = re.compile(r'^_[A-Z_]+ *(?:: *dict *)?= *\{\}', re.M)
    missing = [p.name for p in sorted(root.glob('*.py'))
               if p.name != '_cache.py'
               and declares.search(p.read_text())
               and 'register_cache' not in p.read_text()]
    assert not missing, f"memo declared but never registered: {missing}"


def test_invalidate_grids_empties_every_registered_memo():
    # Hand-listed, so it pins the *behaviour* of the registered clears rather
    # than the roster; the scanner above is what fails when a new backend adds
    # a memo without registering it.
    from uacpy.data import (crust1_local, diesing_local, emodnet_local,
                            sediment_db, seaice_local, wind_local, woa23_local)
    memos = [_cache._GRIDS, crust1_local._MODEL, diesing_local._MODEL,
             emodnet_local._INDEX, sediment_db._SAMPLES, seaice_local._MODEL,
             wind_local._CLIM, woa23_local._DATASETS]

    class _Handle:                     # WOA23 closes its handles before dropping
        closed = False

        def close(self):
            self.closed = True

    sentinel = _Handle()
    for m in memos:
        m['sentinel'] = sentinel
    _cache.invalidate_grids()
    assert all(not m for m in memos)
    assert sentinel.closed


class TestParseDateUsesUTC:
    """Every dataset the date feeds — WOA23, NSIDC, Copernicus — is indexed in
    UTC, and the **month** selects the climatology slice. Taking the local
    calendar date picked the wrong month either side of midnight UTC:
    ``2024-01-01T01:00+05:00`` is 2023-12-31 UTC (December, not January) and
    ``2024-06-30T22:00-06:00`` is 2024-07-01 UTC (July, not June)."""

    @pytest.mark.parametrize('value,expected', [
        (_dt.datetime(2024, 1, 1, 1, 0,
                      tzinfo=_dt.timezone(_dt.timedelta(hours=5))),
         _dt.date(2023, 12, 31)),
        (_dt.datetime(2024, 6, 30, 22, 0,
                      tzinfo=_dt.timezone(_dt.timedelta(hours=-6))),
         _dt.date(2024, 7, 1)),
    ])
    def test_tz_aware_datetime_resolves_in_utc(self, value, expected):
        from uacpy.data._time import parse_date
        assert parse_date(value) == expected

    def test_tz_aware_iso_string_resolves_in_utc(self):
        from uacpy.data._time import parse_date
        assert parse_date('2024-01-01T01:00:00+05:00') == _dt.date(2023, 12, 31)

    @pytest.mark.parametrize('value,expected', [
        ('2024-01-01', _dt.date(2024, 1, 1)),
        (_dt.date(2024, 1, 1), _dt.date(2024, 1, 1)),
        (_dt.datetime(2024, 1, 1, 1, 0), _dt.date(2024, 1, 1)),
    ])
    def test_naive_and_plain_dates_are_unchanged(self, value, expected):
        # The discriminating counterpart: a naive value has no offset to
        # apply and must not move.
        from uacpy.data._time import parse_date
        assert parse_date(value) == expected


class TestPelagicIsDeepSeaOnly:
    """Both facies this model returns are open-ocean: the siliceous belt and
    the carbonate compensation depth. The latitude test fires first and
    returns unconditionally, so an 80 m shelf point at 60 deg came back as
    diatom ooze (phi 9.0, c 1494.9, rho 1.48) — identical to a 4800 m abyssal
    point — from a function whose own docstring says "deep-sea". It is the
    last fallback in the 'auto' chain and documented as never failing, so it
    still returns a value; it just no longer does so silently."""

    @pytest.mark.parametrize('lat,depth', [(60.0, 80.0), (30.0, 80.0),
                                           (-65.0, 120.0)])
    def test_shelf_depth_warns(self, lat, depth):
        from uacpy.data.pelagic import pelagic_lithology
        with pytest.warns(UserWarning, match='shelf break'):
            pelagic_lithology(depth, lat)

    @pytest.mark.parametrize('lat,depth,expected', [
        (60.0, 4800.0, 'diatom ooze'),
        (30.0, 5000.0, 'pelagic clay'),
        (30.0, 1000.0, 'calcareous ooze'),
    ])
    def test_deep_sea_is_unchanged_and_silent(self, lat, depth, expected):
        # The discriminating counterpart: the model's own domain must keep
        # working, unchanged and without noise.
        from uacpy.data.pelagic import pelagic_lithology
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert pelagic_lithology(depth, lat) == expected
