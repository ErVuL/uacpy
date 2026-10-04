"""Tests for the fetch_environment capstone (uacpy.data.environment)."""

import numpy as np
import pytest

from uacpy.core.environment import BoundaryProperties, Environment, SoundSpeedProfile
from uacpy.core.exceptions import ConfigurationError
from uacpy.data import argo, copernicus
from uacpy.data import bathymetry as bathy_mod
from uacpy.data import environment as env_mod
from uacpy.data import seaice_local
from uacpy.data import sound_speed as ssp_mod
from uacpy.tests.conftest import recorded_warnings


@pytest.fixture(autouse=True)
def _no_live_dry_point_check(monkeypatch):
    """fetch_ssp / fetch_ts_profile / fetch_bottom first ask whether the
    point is dry, from the GEBCO grid or else OpenTopoData; these tests are
    about the fetchers behind that check, so it never reaches the network."""
    from uacpy.data import bathymetry, environment
    monkeypatch.setattr(bathymetry, '_refuse_a_dry_point',
                        lambda point, **kw: None)
    monkeypatch.setattr(environment, '_refuse_a_dry_point',
                        lambda point, **kw: None)


def _ts_record():
    """A two-level WOA23 column, as the T/S backends return it."""
    from uacpy.data import SOURCES, DataProvenance
    return ssp_mod.TSProfile(
        depths=np.array([0.0, 50.0]), temperature=np.array([18.0, 16.0]),
        salinity=np.array([36.0, 36.1]), temperature_kind='in_situ',
        provenance=DataProvenance(source=SOURCES['woa23']))


@pytest.fixture
def stub_fetchers(monkeypatch):
    """Replace the network fetchers with deterministic stand-ins.

    Stubs both the point and the transect SSP fetchers — a transect request
    routes through ``fetch_ssp_transect`` (range-dependent SSP), so without that
    stub a transect test would fall through to a live WOA23 OPeNDAP fetch and
    block offline.
    """
    ssp = SoundSpeedProfile(depths=[0.0, 100.0, 2000.0],
                            sound_speed=[1500.0, 1490.0, 1510.0])
    rd_ssp = SoundSpeedProfile(depths=[0.0, 100.0, 2000.0],
                               sound_speed=[[1500.0, 1502.0], [1490.0, 1492.0],
                                     [1510.0, 1512.0]],
                               ranges=[0.0, 5000.0])
    monkeypatch.setattr(bathy_mod, '_fetch_bathy_backend',
                        lambda point, **kw: 2000.0)
    monkeypatch.setattr(bathy_mod, '_fetch_bathy_transect_backend',
                        lambda a, b, **kw: np.array([[0.0, 2000.0], [5000.0, 2200.0]]))
    monkeypatch.setattr(ssp_mod, '_fetch_ssp_backend',
                        lambda point, **kw: ssp)
    monkeypatch.setattr(ssp_mod, '_fetch_ssp_transect_backend',
                        lambda start, end, **kw: rd_ssp)
    return ssp


def test_point_environment(stub_fetchers):
    env = env_mod.fetch_environment((43.2, 7.5), date='2026-06-14')
    assert isinstance(env, Environment)
    assert env.depth == 2000.0
    assert env.name == '43.200, 7.500'
    assert not env.bathymetry.varies_with_range


def test_transect_environment(stub_fetchers):
    env = env_mod.fetch_environment((43.2, 7.5), transect_to=(42.8, 8.1), name='slope')
    assert env.bathymetry.varies_with_range
    assert env.name == 'slope'


@pytest.mark.parametrize('bottom_kw, warns', [
    ({}, True),
    ({'bottom': 'sand'}, False),
    ({'bottom': 2.0}, False),
    ({'bottom_sources': 'pelagic'}, False),
], ids=['no-bottom', 'material', 'phi', 'fetched'])
def test_the_generic_default_seabed_is_announced(stub_fetchers, bottom_kw,
                                                 warns):
    """Neither ``bottom=`` nor ``bottom_sources=`` leaves the Environment's
    generic half-space, which is no data about the site: that one path warns,
    and a literal or fetched seabed does not."""
    with recorded_warnings() as caught:
        env_mod.fetch_environment((43.2, 7.5), **bottom_kw)
    announced = [w for w in caught if 'generic default half-space' in str(w.message)]
    assert bool(announced) is warns


def test_bottom_from_phi(stub_fetchers):
    env = env_mod.fetch_environment((43.2, 7.5), bottom=2.0)  # ϕ → sand-ish
    assert env.bottom is not None
    assert env.bottom.acoustic_type == 'half-space'   # universal across models
    assert env.bottom.columns[0].halfspace.grain_size_phi == 2.0


def test_bottom_from_class_name(stub_fetchers):
    env = env_mod.fetch_environment((43.2, 7.5), bottom='clay')
    assert env.bottom.acoustic_type == 'half-space'


def test_bottom_passthrough(stub_fetchers):
    bp = BoundaryProperties(acoustic_type='half-space', sound_speed=1700, density=1.8)
    env = env_mod.fetch_environment((43.2, 7.5), bottom=bp)
    # A bare BoundaryProperties is coerced into a Bottom holding a copy of it.
    assert env.bottom.columns[0].halfspace == bp
    assert env.bottom.columns[0].halfspace is not bp


def test_a_fetched_boundary_carries_its_roughness_through_the_literal(
        stub_fetchers, monkeypatch):
    """The route ``bottom=``'s docstring names for getting fetched geoacoustics
    *and* a caller-chosen seabed roughness at a point. ``fetch_environment``
    has no roughness knob of its own, so this is the only way to have both."""
    fetched = BoundaryProperties(acoustic_type='half-space', sound_speed=1792.5,
                                 density=2.014, roughness=0.5)
    env = env_mod.fetch_environment((43.2, 7.5), bottom=fetched)
    assert env.bottom.columns[0].halfspace.roughness == pytest.approx(0.5)
    assert env.bottom.columns[0].halfspace.sound_speed == pytest.approx(1792.5)


def test_a_resolving_bottom_source_discards_the_literals_roughness(
        stub_fetchers, monkeypatch):
    """The other half of the documented behaviour: the literal is a *fallback*,
    so a resolving ``bottom_sources`` replaces it whole — roughness included.
    That is why the route above names no ``bottom_sources``."""
    monkeypatch.setattr(
        env_mod, '_fetch_bottom',
        lambda *a, **kw: (BoundaryProperties(acoustic_type='half-space',
                                             sound_speed=1785.36, density=1.9),
                          None))
    fetched = BoundaryProperties(acoustic_type='half-space', sound_speed=1792.5,
                                 density=2.014, roughness=0.5)
    env = env_mod.fetch_environment((43.2, 7.5), bottom_sources='local',
                                    bottom=fetched)
    halfspace = env.bottom.columns[0].halfspace
    assert halfspace.sound_speed == pytest.approx(1785.36)
    assert not halfspace.roughness


def test_a_range_dependent_bottom_literal_passes_with_its_provenance(
        stub_fetchers):
    """The ``Bottom`` a ``fetch_bottom_transect`` returns is the transect's
    roughness route: it reaches the Environment whole, columns, ranges,
    roughness and provenance."""
    from uacpy.core.environment import Bottom, SeabedColumn
    from uacpy.data.sources import SOURCES, DataProvenance

    prov = (DataProvenance(source=SOURCES['grainsize']),)
    rd = Bottom.from_columns(
        [SeabedColumn(layers=[], halfspace=BoundaryProperties(
            acoustic_type='half-space', sound_speed=c, density=1.8,
            roughness=0.5, data_sources=prov)) for c in (1700.0, 1650.0)],
        ranges=np.array([0.0, 5000.0]))
    env = env_mod.fetch_environment((43.2, 7.5), transect_to=(42.8, 8.1),
                                    bottom=rd)
    assert env.bottom is not rd                 # the environment's own copy
    assert env.bottom.ranges.tolist() == [0.0, 5000.0]
    assert [c.halfspace.roughness for c in env.bottom.columns] == [0.5, 0.5]
    assert 'grainsize' in [p.source.id for p in env.data_sources]


def test_a_layered_seabed_column_literal_passes_at_a_point(stub_fetchers):
    """``fetch_bottom(pt, source='crust1')`` returns a ``SeabedColumn``;
    ``bottom=`` takes it as ``Environment(bottom=)`` does."""
    from uacpy.core.environment import SeabedColumn, SedimentLayer
    from uacpy.data.sources import SOURCES, DataProvenance

    col = SeabedColumn(
        layers=[SedimentLayer(thickness=200.0, sound_speed=1700.0,
                              density=1.9)],
        halfspace=BoundaryProperties(
            acoustic_type='half-space', sound_speed=4500.0, density=2.6,
            data_sources=(DataProvenance(source=SOURCES['crust1']),)))
    env = env_mod.fetch_environment((43.2, 7.5), bottom=col)
    assert env.bottom.columns[0].layers[0].sound_speed == 1700.0
    assert env.bottom.columns[0].halfspace.sound_speed == 4500.0
    assert 'crust1' in [p.source.id for p in env.data_sources]


def test_literal_ssp_skips_fetch(monkeypatch):
    # A literal ssp= is used verbatim; the SSP fetcher must NOT be called.
    monkeypatch.setattr(bathy_mod, '_fetch_bathy_backend', lambda point, **kw: 2000.0)
    monkeypatch.setattr(ssp_mod, '_fetch_ssp_backend',
                        lambda *a, **k: pytest.fail("SSP should not be fetched"))
    prof = SoundSpeedProfile(depths=[0.0, 2000.0], sound_speed=[1500.0, 1520.0])
    env = env_mod.fetch_environment((43.2, 7.5), ssp=prof)
    assert env.ssp.sound_speed[0] == 1500.0
    assert 'woa23' not in [s.source.id for s in env.data_sources]


def test_literal_bathymetry_skips_fetch(monkeypatch):
    # A literal bathymetry= is used verbatim; the bathy fetcher must NOT run.
    monkeypatch.setattr(bathy_mod, '_fetch_bathy_backend',
                        lambda *a, **k: pytest.fail("bathy should not be fetched"))
    monkeypatch.setattr(ssp_mod, '_fetch_ssp_backend',
                        lambda point, **kw: SoundSpeedProfile(depths=[0.0, 50.0],
                                                              sound_speed=[1500.0, 1490.0]))
    env = env_mod.fetch_environment((43.2, 7.5), bathymetry=120.0)
    assert env.depth == 120.0
    assert 'gebco' not in [s.source.id for s in env.data_sources]


def test_source_fetch_wins_over_literal(stub_fetchers):
    # Both given + fetch succeeds → the fetched value is used (literal ignored).
    env = env_mod.fetch_environment((43.2, 7.5), bathymetry=999.0,
                                    bathymetry_sources='gebco')
    assert env.depth == 2000.0                       # stub-fetched, not 999.0
    assert 'gebco' in [s.source.id for s in env.data_sources]


def test_bathymetry_sources_auto_prefers_emodnet_dtm(stub_fetchers):
    # 'auto' bathymetry = ('emodnet_dtm', 'gmrt', 'gebco'); the stub fetcher
    # succeeds, so the first (highest-res regional DTM) wins in provenance.
    env = env_mod.fetch_environment((43.2, 7.5), bathymetry_sources='auto')
    assert 'emodnet_dtm' in [s.source.id for s in env.data_sources]


def test_ssp_sources_auto_falls_to_woa23_without_date(stub_fetchers):
    # 'auto' ssp = ('argo','copernicus','woa23'); argo/copernicus need date=, so
    # with none they fall through to WOA23 (the stub).
    env = env_mod.fetch_environment((43.2, 7.5), ssp_sources='auto')
    assert 'woa23' in [s.source.id for s in env.data_sources]
    assert 'argo' not in [s.source.id for s in env.data_sources]


def test_literal_fallback_when_fetch_fails(monkeypatch):
    # Both given + fetch fails → fall back to the user literal (no raise).
    from uacpy.core.exceptions import DataFetchError
    monkeypatch.setattr(bathy_mod, '_fetch_bathy_backend',
                        lambda *a, **k: (_ for _ in ()).throw(
                            DataFetchError("service down")))
    monkeypatch.setattr(ssp_mod, '_fetch_ssp_backend',
                        lambda point, **kw: SoundSpeedProfile(depths=[0.0, 50.0],
                                                              sound_speed=[1500.0, 1490.0]))
    env = env_mod.fetch_environment((43.2, 7.5), bathymetry=120.0,
                                    bathymetry_sources='gebco')
    assert env.depth == 120.0                        # literal fallback
    assert 'gebco' not in [s.source.id for s in env.data_sources]


def test_surface_source_without_date_falls_back_to_literal():
    # surface= is the documented fallback when the seaice fetch cannot run;
    # a missing date= must reach that fallback, not raise past it.
    ice = BoundaryProperties(acoustic_type='half-space', sound_speed=3500.0,
                             density=0.9, attenuation=0.4, shear_speed=1800.0,
                             shear_attenuation=1.0)
    env = env_mod.fetch_environment((75.0, -10.0), bathymetry=2000.0,
                                    ssp=1500.0, surface_sources='seaice',
                                    surface=ice)
    assert env.surface is not None
    assert env.surface.nodes[0].sound_speed == 3500.0


def test_surface_source_without_date_or_literal_raises():
    with pytest.raises(ConfigurationError, match='needs date'):
        env_mod.fetch_environment((75.0, -10.0), bathymetry=2000.0,
                                  ssp=1500.0, surface_sources='seaice')


def test_range_dependent_ssp(monkeypatch, stub_fetchers):
    rd_ssp = SoundSpeedProfile(depths=[0.0, 100.0, 2000.0],
                               sound_speed=[[1500, 1502], [1490, 1492], [1510, 1512]],
                               ranges=[0.0, 5000.0])
    monkeypatch.setattr(ssp_mod, '_fetch_ssp_transect_backend',
                        lambda start, end, **kw: rd_ssp)
    env = env_mod.fetch_environment((43.2, 7.5), transect_to=(42.8, 8.1),
                                    range_dependent_ssp=True)
    assert env.ssp.is_range_dependent
    assert env.bathymetry.varies_with_range


def test_range_dependent_ssp_requires_transect(stub_fetchers):
    with pytest.raises(ConfigurationError, match='requires transect_to'):
        env_mod.fetch_environment((43.2, 7.5), range_dependent_ssp=True)


def test_range_dependent_ssp_copernicus(monkeypatch, stub_fetchers):
    import uacpy.data.copernicus as cop_mod
    rd = SoundSpeedProfile(depths=[0.0, 100.0], sound_speed=[[1500, 1502], [1490, 1492]],
                           ranges=[0.0, 5000.0])
    monkeypatch.setattr(cop_mod, 'fetch_ssp_transect_operational',
                        lambda start, end, date, **kw: rd)
    env = env_mod.fetch_environment((43.2, 7.5), transect_to=(42.8, 8.1),
                                    range_dependent_ssp=True, ssp_sources='copernicus',
                                    date='2026-01-01')
    assert env.ssp.is_range_dependent


def _spy_copernicus_transect(monkeypatch):
    import uacpy.data.copernicus as cop_mod
    seen = {}
    rd = SoundSpeedProfile(depths=[0.0, 100.0], sound_speed=[[1500, 1502], [1490, 1492]],
                           ranges=[0.0, 5000.0])

    def fake(start, end, date, **kw):
        seen.update(kw)
        return rd
    monkeypatch.setattr(cop_mod, 'fetch_ssp_transect_operational', fake)
    return seen


def test_range_dependent_ssp_copernicus_honours_a_numpy_integer_count(
        monkeypatch, stub_fetchers):
    seen = _spy_copernicus_transect(monkeypatch)
    env_mod.fetch_environment((43.2, 7.5), transect_to=(42.8, 8.1),
                              range_dependent_ssp=True, ssp_sources='copernicus',
                              date='2026-01-01', ssp_n_points=np.int64(4))
    assert seen['n_points'] == 4


def test_range_dependent_ssp_copernicus_forwards_the_count_and_the_cap(
        monkeypatch, stub_fetchers):
    # The operational fetcher resolves 'auto' and applies the cap itself
    # (test_data_copernicus pins that); fetch_environment passes both on.
    seen = _spy_copernicus_transect(monkeypatch)
    env_mod.fetch_environment((43.2, 7.5), transect_to=(42.8, 8.1),
                              range_dependent_ssp=True,
                              ssp_sources='copernicus', date='2026-01-01',
                              ssp_n_points=30, max_points=10)
    assert (seen['n_points'], seen['max_points']) == (30, 10)


def test_range_dependent_ssp_copernicus_requires_date(stub_fetchers):
    with pytest.raises(ConfigurationError, match='requires date'):
        env_mod.fetch_environment((43.2, 7.5), transect_to=(42.8, 8.1),
                                  range_dependent_ssp=True, ssp_sources='copernicus')


def test_removed_global_keyword_rejected(stub_fetchers):
    # 'global' names no fetch source, so ``bottom='global'`` falls through to
    # the sediment-class lookup and is refused there. uacpy carries no global
    # seabed grid to bind it to: the CC-BY-NC candidate for that role is not
    # redistributable, which is what ``pelagic`` covers instead.
    with pytest.raises(ConfigurationError, match='unknown sediment class'):
        env_mod.fetch_environment((43.2, 7.5), bottom='global')


def test_bottom_auto_falls_back_to_pelagic(tmp_path, monkeypatch, stub_fetchers):
    # 'auto' always resolves: outside measured coverage (and with no DBs
    # installed) it uses the global, commercial-clean pelagic model rather than
    # raising. An empty cache forces the fall-through past EMODnet/Diesing.
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    import uacpy.data.seabed as seabed_mod
    import uacpy.data.pelagic as pelagic_mod
    from uacpy.core.exceptions import DataFetchError

    def no_emodnet(point, **kw):
        raise DataFetchError("European seas only")
    monkeypatch.setattr(seabed_mod, 'fetch_bottom_emodnet', no_emodnet)
    monkeypatch.setattr(pelagic_mod, '_water_depth', lambda *a, **k: 5000.0)
    env = env_mod.fetch_environment((43.2, 7.5), bottom_sources='auto')
    assert env.bottom is not None
    assert env.data_sources[-1].source.id == 'pelagic'
    # The classification uses the environment's own 2000 m bathymetry (above
    # the CCD → calcareous ooze), not the 5000 m its own lookup would return.
    assert env.bottom.columns[0].halfspace.grain_size_phi == pytest.approx(7.5)


def test_pelagic_never_refetches_the_bathymetry(tmp_path, monkeypatch,
                                                stub_fetchers):
    # fetch_environment already holds the depth, so the pelagic model must be
    # handed it rather than issuing its own GEBCO lookup per waypoint.
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    import uacpy.data.pelagic as pelagic_mod
    monkeypatch.setattr(pelagic_mod, '_water_depth', lambda *a, **k: (
        pytest.fail("pelagic re-fetched the bathymetry")))
    point = env_mod.fetch_environment((43.2, 7.5), bottom_sources='pelagic')
    assert point.bottom.columns[0].halfspace.grain_size_phi == pytest.approx(7.5)
    # A transect crossing the CCD (4500 m) must classify each column at *its
    # own* range: calcareous ooze inshore, pelagic clay beyond the crossing.
    monkeypatch.setattr(
        bathy_mod, '_fetch_bathy_transect_backend',
        lambda a, b, **kw: np.column_stack([np.linspace(0.0, 50e3, 21),
                                            np.linspace(3000.0, 5500.0, 21)]))
    transect = env_mod.fetch_environment(
        (43.2, 7.5), transect_to=(42.8, 8.1), bottom_sources='pelagic',
        range_dependent_bottom=True, bottom_n_points='auto', max_points=40)
    # Compare densities, not speeds: a column's sound speed is its Hamilton
    # ratio times the *local* water speed, which itself rises with depth, so it
    # does not isolate the lithology. Density is monotone in grain size. The
    # 'auto' collapse keeps the probe columns bracketing the CCD crossing plus
    # the endpoints: an ooze pair, then a clay pair.
    rho = [c.halfspace.density for c in transect.bottom.columns]
    assert len(rho) == 4
    assert rho[0] == rho[1] > rho[2] == rho[3]          # ooze pair, clay pair


def test_range_dependent_bottom(monkeypatch, stub_fetchers):
    # range_dependent_bottom=True alone resolves the 'auto' chain at every
    # waypoint, and 'auto' is cache-first: EMODnet's local polygons answer
    # each waypoint they cover.
    import uacpy.data.emodnet_local as emodnet_mod
    seabeds = iter([1650.0, 1640.0, 1630.0, 1620.0, 1610.0, 1600.0])
    monkeypatch.setattr(
        emodnet_mod, 'fetch_bottom_emodnet_local',
        lambda point, **kw: BoundaryProperties(
            acoustic_type='half-space', sound_speed=next(seabeds),
            density=1.8))
    env = env_mod.fetch_environment((43.2, 7.5), transect_to=(42.8, 8.1),
                                    range_dependent_bottom=True)
    assert (env.bottom.is_range_dependent and not env.bottom.is_layered)
    assert [c.halfspace.sound_speed for c in env.bottom.columns] == [
        1650.0, 1640.0, 1630.0, 1620.0, 1610.0, 1600.0]


def test_range_dependent_bottom_requires_transect(stub_fetchers):
    with pytest.raises(ConfigurationError, match='requires transect_to'):
        env_mod.fetch_environment((43.2, 7.5), range_dependent_bottom=True)


def test_with_absorption(monkeypatch, stub_fetchers):
    from uacpy.core.absorption import FrancoisGarrison
    monkeypatch.setattr(env_mod, '_fetch_ts_profile_backend',
                        lambda point, **kw: _ts_record())
    env = env_mod.fetch_environment((43.2, 7.5), with_absorption=True)
    assert isinstance(env.absorption, FrancoisGarrison)
    # The whole fetched column, each depth with its own water.
    np.testing.assert_array_equal(env.absorption.temperature,
                                  [[0.0, 18.0], [50.0, 16.0]])
    np.testing.assert_array_equal(env.absorption.salinity,
                                  [[0.0, 36.0], [50.0, 36.1]])


@pytest.mark.parametrize('resolution', ['1.00', '0.25'])
def test_with_absorption_uses_the_requested_woa_grid(monkeypatch, stub_fetchers,
                                                     resolution):
    """The T/S column must come from the same WOA23 grid as the SSP.

    A 0.25° SSP against a 1.00° absorption column is a different cell — up to
    ~50 km away at mid latitudes.
    """
    seen = {}

    def fake_ts(point, **kw):
        seen.update(kw)
        return _ts_record()

    monkeypatch.setattr(env_mod, '_fetch_ts_profile_backend', fake_ts)
    env_mod.fetch_environment((43.2, 7.5), with_absorption=True,
                              resolution=resolution)
    assert seen['resolution'] == resolution


@pytest.mark.parametrize('local_installed, expected', [
    (True, ['local']),
    (False, ['local', 'opendap']),
])
def test_with_absorption_under_a_literal_ssp_reads_woa23_cache_first(
        monkeypatch, local_installed, expected):
    """A literal ``ssp=`` leaves no SSP backend to reuse, so the absorption
    T/S column goes through the public cache-first WOA23 fetcher: the
    installed grid answers when present, NCEI THREDDS only when it is absent."""
    from uacpy.core.exceptions import ConfigurationError as CfgError
    tried = []

    def fake_ts(point, **kw):
        tried.append(kw.get('backend'))
        if kw.get('backend') == 'local' and not local_installed:
            raise CfgError('Offline woa23 data not found')
        return _ts_record()

    monkeypatch.setattr(ssp_mod, '_fetch_ts_profile_backend', fake_ts)
    monkeypatch.setattr(env_mod, '_fetch_ts_profile_backend', fake_ts)
    env = env_mod.fetch_environment((43.2, 7.5), bathymetry=2000.0,
                                    ssp=1500.0, with_absorption=True)
    assert tried == expected
    np.testing.assert_array_equal(env.absorption.temperature[:, 1],
                                  [18.0, 16.0])


@pytest.mark.parametrize('ssp_sources', [None, 'local'])
def test_the_absorption_ts_dataset_is_cited_beside_a_literal_ssp(
        monkeypatch, ssp_sources):
    """The WOA23 row sets the absorption and the water density, so WOA23 is
    in the provenance even when the SSP itself is a literal (``'local'``: a
    cache-pinned SSP that fell back to the literal)."""
    from uacpy.core.exceptions import DataFetchError as FetchError

    def fake_ts(point, **kw):
        return _ts_record()

    def no_ssp(point, **kw):
        raise FetchError('no WOA23 column in the cache')

    monkeypatch.setattr(ssp_mod, '_fetch_ts_profile_backend', fake_ts)
    monkeypatch.setattr(env_mod, '_fetch_ts_profile_backend', fake_ts)
    monkeypatch.setattr(ssp_mod, '_fetch_ssp_backend', no_ssp)
    env = env_mod.fetch_environment((43.2, 7.5), bathymetry=2000.0,
                                    ssp=1500.0, ssp_sources=ssp_sources,
                                    with_absorption=True, bottom='sand')
    assert env.ssp.data_sources == ()
    assert 'woa23' in [p.source.id for p in env.data_sources]


def _two_seabed_providers(monkeypatch):
    """A regional 'emodnet' covering only lon > -8 (cp 1595) and a global
    'pelagic' (cp 1510), each stamping its own provenance; point and transect
    fetchers alike, the transect being the real per-source gap filler."""
    import dataclasses
    from uacpy.core.exceptions import DataFetchError
    from uacpy.data.sediment import range_dependent_bottom_along
    from uacpy.data.sources import SOURCES, DataProvenance

    def provider(sid, cp, covers):
        def point(pt, **kw):
            if not covers(pt[1]):
                raise DataFetchError(f'{sid} has no seabed here')
            return BoundaryProperties(
                acoustic_type='half-space', sound_speed=cp, density=1.5,
                data_sources=(DataProvenance(source=SOURCES[sid]),))

        def transect(start, end, *, n_points=6, max_points=None, **kw):
            return range_dependent_bottom_along(
                lambda la, lo: point((la, lo)), start, end, n_points,
                source_label=sid, max_points=max_points)
        return lambda *cached: (point, transect)

    for sid, cp, covers in (('emodnet', 1595.0, lambda lon: lon > -8.0),
                            ('pelagic', 1510.0, lambda lon: True)):
        monkeypatch.setitem(env_mod._BOTTOM_BY_ID, sid, dataclasses.replace(
            env_mod._BOTTOM_BY_ID[sid], resolve=provider(sid, cp, covers)))


def test_a_bottom_chain_on_a_transect_resolves_each_waypoint(monkeypatch):
    """Waypoints the regional source covers take its seabed; the others take
    the next source's seabed there, and the environment cites both."""
    _two_seabed_providers(monkeypatch)
    with recorded_warnings() as caught:
        env = env_mod.fetch_environment(
            (45.0, -5.0), transect_to=(45.0, -11.0), bathymetry=4000.0,
            ssp=1500.0, bottom_sources=('emodnet', 'pelagic'),
            bottom_n_points=6)
    lons = np.linspace(-5.0, -11.0, 6)       # the waypoints, to 0.02 deg
    cp = [c.halfspace.sound_speed for c in env.bottom.columns]
    assert cp == [1595.0 if lon > -8.0 else 1510.0 for lon in lons]
    assert [c.halfspace.data_sources[0].source.id
            for c in env.bottom.columns] == [
        'emodnet' if lon > -8.0 else 'pelagic' for lon in lons]
    assert {'emodnet', 'pelagic'} <= {p.source.id for p in env.data_sources}
    assert not [w for w in caught if 'filled from the nearest' in str(w.message)]


def test_a_single_bottom_source_on_a_transect_fills_its_gaps(monkeypatch):
    """One named source has no next source to ask: its uncovered waypoints
    take the nearest covered one, with the warning that says so."""
    _two_seabed_providers(monkeypatch)
    with recorded_warnings() as caught:
        env = env_mod.fetch_environment(
            (45.0, -5.0), transect_to=(45.0, -11.0), bathymetry=4000.0,
            ssp=1500.0, bottom_sources='emodnet', bottom_n_points=6)
    assert {c.halfspace.sound_speed for c in env.bottom.columns} == {1595.0}
    assert [w for w in caught if 'filled from the nearest' in str(w.message)]


@pytest.mark.parametrize('lon, expected', [(-5.0, 1595.0), (-10.0, 1510.0)])
def test_fetch_bottom_takes_the_first_source_in_the_chain_that_covers_the_point(
        monkeypatch, lon, expected):
    from uacpy import data
    _two_seabed_providers(monkeypatch)
    bottom = data.fetch_bottom((45.0, lon), source=('emodnet', 'pelagic'),
                               depth=4000.0)
    assert bottom.sound_speed == expected


def test_fetch_bottom_transect_resolves_a_chain_at_each_waypoint(monkeypatch):
    from uacpy import data
    _two_seabed_providers(monkeypatch)
    bottom = data.fetch_bottom_transect((45.0, -5.0), (45.0, -11.0),
                                        source=('emodnet', 'pelagic'),
                                        n_points=6, depth=4000.0)
    lons = np.linspace(-5.0, -11.0, 6)
    assert [c.halfspace.sound_speed for c in bottom.columns] == [
        1595.0 if lon > -8.0 else 1510.0 for lon in lons]


def _recording_crust1(monkeypatch):
    """CRUST1.0's point and transect fetchers replaced by recorders that keep
    the real signatures (the option check reads them); returns the calls."""
    import functools
    from uacpy.data import crust1_local
    calls = []

    def recorder(real):
        @functools.wraps(real)
        def fetch(*args, **kw):
            calls.append(kw)
            return BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.8)
        return fetch

    monkeypatch.setattr(crust1_local, 'fetch_bottom_crust1',
                        recorder(crust1_local.fetch_bottom_crust1))
    monkeypatch.setattr(crust1_local, 'fetch_bottom_crust1_transect',
                        recorder(crust1_local.fetch_bottom_crust1_transect))
    return calls


def test_fetch_bottom_forwards_the_sources_own_options(monkeypatch):
    from uacpy import data
    calls = _recording_crust1(monkeypatch)
    data.fetch_bottom((45.0, -20.0), source='crust1',
                      sediment_attenuation=0.5, use_globsed=False)
    assert calls[0]['sediment_attenuation'] == 0.5
    assert calls[0]['use_globsed'] is False
    data.fetch_bottom_transect((45.0, -20.0), (45.0, -22.0), source='crust1',
                               basement_attenuation=0.05)
    assert calls[1]['basement_attenuation'] == 0.05


@pytest.mark.parametrize('source, options, match', [
    ('crust1', {'sediment_attenuaton': 0.5}, "takes no option 'sediment_att"),
    (('crust1', 'pelagic'), {'use_globsed': False},
     "'pelagic' takes no option 'use_globsed'"),
    ('crust1', {'cache_only': True}, "is set by fetch_bottom itself"),
])
def test_a_source_option_no_fetcher_takes_is_refused_before_any_fetch(
        monkeypatch, source, options, match):
    from uacpy import data
    calls = _recording_crust1(monkeypatch)
    with pytest.raises(ConfigurationError, match=match):
        data.fetch_bottom((45.0, -20.0), source=source, depth=4000.0,
                          **options)
    assert calls == []


def _argo_cast_and_woa23(monkeypatch, *, woa_available=True):
    """An Argo cast to 2000 m at 1490 m/s, and a WOA23 column to 5000 m at
    1500 m/s, each stamped with its own provenance; returns the WOA calls."""
    from uacpy.data.sources import SOURCES, DataProvenance
    from uacpy.core.exceptions import DataFetchError
    cast = SoundSpeedProfile(
        depths=[5.0, 500.0, 1000.0, 1500.0, 1750.0, 2000.0],
        sound_speed=[1490.0] * 6, kind='measured', formula='teos10',
        data_sources=(DataProvenance(source=SOURCES['argo']),))
    woa = SoundSpeedProfile(
        depths=[0.0, 1000.0, 1600.0, 1900.0, 2500.0, 5000.0],
        sound_speed=[1500.0] * 6, kind='measured', formula='teos10',
        data_sources=(DataProvenance(source=SOURCES['woa23']),))
    calls = []

    def fake_woa(point, **kw):
        calls.append(kw)
        if not woa_available:
            raise DataFetchError('WOA23 has no column here')
        return woa

    monkeypatch.setattr(argo, 'fetch_ssp_argo', lambda point, **kw: cast)
    monkeypatch.setattr(env_mod, 'fetch_ssp', fake_woa)
    return calls


def test_an_argo_cast_is_continued_by_woa23_through_a_blend(monkeypatch):
    """Above the blend the profile is the cast, below the cast it is WOA23,
    and in between the raised-cosine weight moves between them without a
    step; both records are cited."""
    _argo_cast_and_woa23(monkeypatch)
    env = env_mod.fetch_environment((30.0, -40.0), date='2024-06-04',
                                    bathymetry=5000.0, ssp_sources='argo',
                                    bottom='sand')
    z = np.asarray(env.ssp.depths)
    c = np.asarray(env.ssp.sound_speed)[:, 0]
    top = 2000.0 - env_mod.CAST_BLEND_M
    assert np.all(c[z <= top] == 1490.0)
    assert np.all(c[z >= 2000.0] == 1500.0)
    blend = (z > top) & (z < 2000.0)
    assert np.all((c[blend] > 1490.0) & (c[blend] < 1500.0))
    assert np.all(np.diff(c[(z >= top) & (z <= 2000.0)]) > 0.0)
    assert z[-1] == 5000.0
    assert [p.source.id for p in env.ssp.data_sources] == ['argo', 'woa23']


@pytest.mark.parametrize('bathymetry, woa_available, asks_woa, cited', [
    (2000.0, True, False, ['argo']),           # the cast reaches the seafloor
    (2050.0, True, False, ['argo']),           # within the 50 m tolerance
    (2051.0, True, True, ['argo', 'woa23']),   # just past it: WOA23 joins
    (5000.0, False, True, ['argo']),           # WOA23 unreadable: extended
])
def test_woa23_continues_an_argo_cast_only_past_the_extension_tolerance(
        monkeypatch, bathymetry, woa_available, asks_woa, cited):
    calls = _argo_cast_and_woa23(monkeypatch, woa_available=woa_available)
    env = env_mod.fetch_environment((30.0, -40.0), date='2024-06-04',
                                    bathymetry=bathymetry, ssp_sources='argo',
                                    bottom='sand')
    assert bool(calls) is asks_woa
    assert [p.source.id for p in env.ssp.data_sources] == cited
    assert np.asarray(env.ssp.depths)[-1] == pytest.approx(bathymetry)


def test_copernicus_requires_date(stub_fetchers):
    with pytest.raises(ConfigurationError, match='requires date'):
        env_mod.fetch_environment((43.2, 7.5), ssp_sources='copernicus')


def test_unknown_ssp_sources(stub_fetchers):
    with pytest.raises(ConfigurationError, match='unknown ssp source'):
        env_mod.fetch_environment((43.2, 7.5), ssp_sources='nope')


class TestWoaWetCellSearch:
    """A coastal point must not be refused because it snaps onto a land cell.

    The documented quick-start ``fetch_environment((43.2, 7.5))`` is in the
    Ligurian Sea, close enough to shore that the nearest WOA cell can be dry,
    which must not be reported as "on land or outside the analyzed domain".
    """

    def test_ring_offsets_are_nearest_first_and_complete(self):
        from uacpy.data._geo import ring_offsets
        r1 = ring_offsets(1)
        assert len(r1) == 8, r1
        assert r1[0] in {(0, -1), (0, 1), (-1, 0), (1, 0)}
        assert len(ring_offsets(2)) == 16

    def test_dry_nearest_cell_falls_back_to_a_wet_neighbour(self):
        import numpy as np
        from uacpy.data.sound_speed import _nearest_wet_column
        wet = (60, 100)

        def fetch(i, j):
            if (i, j) == wet:
                return (np.array([0.0, 10.0]), np.array([15.0, 14.0]),
                        np.array([38.0, 38.1]))
            return np.array([]), np.array([]), np.array([])

        z, t, s, i, j = _nearest_wet_column(fetch, 60, 99, '1.00')
        assert (i, j) == wet
        assert z.size == 2

    def test_longitude_wraps_during_the_search(self):
        import numpy as np
        from uacpy.data.sound_speed import _nearest_wet_column
        n_lon = 360
        wet = (10, 0)

        def fetch(i, j):
            if (i, j) == wet:
                return np.array([0.0]), np.array([15.0]), np.array([38.0])
            return np.array([]), np.array([]), np.array([])

        z, t, s, i, j = _nearest_wet_column(fetch, 10, n_lon - 1, '1.00')
        assert (i, j) == wet, "search must wrap across the antimeridian"

    def test_land_locked_request_fails(self):
        import numpy as np
        from uacpy.data.sound_speed import _nearest_wet_column
        empty = (np.array([]), np.array([]), np.array([]))
        z, t, s, i, j = _nearest_wet_column(lambda i, j: empty, 60, 100, '1.00')
        assert z.size == 0, "a genuinely dry region must not be papered over"


class TestDeepSSPExtension:
    """Analysed T/S products (WOA23, 1 deg) are routinely shallower than
    bathymetry (GEBCO, 15 arc-sec). Holding the last sound speed drops the
    whole pressure term: 61 m/s over the bottom 3.3 km in the Izu-Bonin
    trench."""

    @staticmethod
    def _profile():
        from uacpy.core.environment import SoundSpeedProfile
        # A deep tail at the adiabatic gradient: 1.85 m/s per 100 m.
        z = np.array([0.0, 1000.0, 5300.0, 5400.0, 5500.0])
        c = np.array([1540.0, 1484.0, 1547.34, 1549.20, 1551.05])
        return SoundSpeedProfile.from_pairs(np.column_stack([z, c]))

    def test_extends_along_the_profiles_own_gradient(self):
        import warnings as _w
        from uacpy.data.sound_speed import extend_ssp_below_data
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            out = extend_ssp_below_data(self._profile(), 8801.0, latitude=29.78)
        c_end = float(np.asarray(out.sound_speed)[-1, 0])
        # TEOS-10 at 8801 m holding the deepest T/S gives 1611.68 m/s. The
        # tolerance is tight on purpose: at 2 m/s a fixed 0.0165 s^-1 gradient
        # (6.4 m/s slow here) is only just excluded, and a fixed 0.017 is not.
        assert c_end == pytest.approx(1611.68, abs=0.05), (
            f"deep sound speed {c_end:.2f} m/s — holding the last value would "
            f"give 1551.05, which is 61 m/s slow")
        assert float(np.asarray(out.depths)[-1]) == pytest.approx(8801.0)

    def test_no_single_gradient_can_reproduce_the_extension(self):
        """dc/dz is a function of depth (0.0168 s^-1 at 1 km against 0.0189 at
        8 km, UNESCO), so the same sound speed extended over the same span must
        gain *more* starting deeper. Any fixed-gradient implementation returns
        exactly equal increments here."""
        from uacpy.data.sound_speed import _deep_increment
        shallow = _deep_increment(1500.0, 1500.0, 2500.0, 45.0)
        deep = _deep_increment(1500.0, 5500.0, 6500.0, 45.0)
        assert deep > shallow + 0.5, (
            f"increment over 1000 m is {shallow:.3f} m/s at 1500 m and "
            f"{deep:.3f} m/s at 5500 m — a constant gradient gives both alike")

    def test_a_fixed_canonical_gradient_is_excluded(self):
        """The shipped 0.0165 s^-1 sat below the UNESCO envelope (0.0168-0.0189)
        at every deep condition, so it was slow by 6.4 m/s in the trench."""
        from uacpy.data.sound_speed import _deep_increment
        span = 8801.0 - 5500.0
        got = _deep_increment(1551.05, 5500.0, 8801.0, 29.78)
        assert got - 0.0165 * span > 5.0, (
            f"increment {got:.2f} m/s against {0.0165 * span:.2f} m/s for the "
            f"old constant — the bias has not been removed")
        assert got / span > 0.0168, "below the UNESCO deep-gradient envelope"

    def test_a_noisy_last_segment_cannot_contaminate_the_extension(self):
        """The old gate admitted any measured gradient in (0.005, 0.030), so a
        noisy WOA bottom pair was extrapolated for kilometres: 0.028 s^-1 over
        3.3 km is +92 m/s against a true +61. The increment must depend only on
        the deepest sound speed, never on the segment above it."""
        import warnings as _w
        from uacpy.core.environment import SoundSpeedProfile
        from uacpy.data.sound_speed import extend_ssp_below_data

        def extend(c_second_last):
            p = SoundSpeedProfile.from_pairs(np.array(
                [[0.0, 1540.0], [1000.0, 1484.0],
                 [5400.0, c_second_last], [5500.0, 1551.05]]))
            with _w.catch_warnings():
                _w.simplefilter('ignore')
                out = extend_ssp_below_data(p, 8801.0, latitude=29.78)
            return float(np.asarray(out.sound_speed)[-1, 0])

        clean = extend(1549.20)                       # 0.0185 s^-1
        noisy = extend(1548.25)                       # 0.028  s^-1, was accepted
        inverted = extend(1560.00)                    # negative, was rejected
        assert clean == pytest.approx(noisy, abs=1e-9)
        assert clean == pytest.approx(inverted, abs=1e-9)

    def test_an_unphysical_column_clamps_rather_than_diverging(self):
        """A sound speed outside UNESCO's range over the whole -3..35 C bracket
        cannot be inverted; the extension must stay finite and monotone."""
        from uacpy.data.sound_speed import _deep_increment
        for c_absurd in (900.0, 2200.0):
            got = _deep_increment(c_absurd, 1010.0, 3000.0, 45.0)
            assert np.isfinite(got) and got > 0.0, (
                f"c={c_absurd} m/s gave increment {got}")

    def test_warm_deep_basins_are_handled(self):
        """Mediterranean (~13 C) and Red Sea (~21 C) deep water are far off the
        canonical polar values, and dc/dz falls with temperature. The inversion
        recovers it from the column itself. The truth is the increment's
        own (default) equation at the true T/S, so what is measured is the
        inversion, not the 0.16 m/s by which UNESCO's and TEOS-10's pressure
        terms differ over this span."""
        from uacpy.core.acoustics import sound_speed_teos10
        from uacpy.core.acoustics.seawater import depth_to_pressure_dbar
        from uacpy.data.sound_speed import _deep_increment
        for t_true, s_true in ((13.0, 38.5), (21.0, 40.5)):
            p0 = float(depth_to_pressure_dbar(1500.0, 45.0))
            p1 = float(depth_to_pressure_dbar(3000.0, 45.0))
            c0 = sound_speed_teos10(t_true, s_true, pressure_dbar=p0)
            truth = sound_speed_teos10(t_true, s_true, pressure_dbar=p1) - c0
            # The inversion holds S at the 35 reference; the residual is the
            # equation's salinity-pressure cross term over 1500 m: 0.04 m/s
            # for the Mediterranean, 0.16 for the Red Sea at 40.5 PSU (UNESCO
            # gave 0.14 there). A fixed gradient is 7 m/s out on the same span.
            assert _deep_increment(c0, 1500.0, 3000.0, 45.0) == pytest.approx(
                truth, abs=0.2)

    def test_long_extrapolation_warns(self):
        from uacpy.data.sound_speed import extend_ssp_below_data
        with pytest.warns(UserWarning, match="extrapolated"):
            extend_ssp_below_data(self._profile(), 8801.0)

    def test_short_extension_is_quiet(self):
        import warnings as _w
        from uacpy.data.sound_speed import extend_ssp_below_data
        with _w.catch_warnings():
            _w.simplefilter('error')
            extend_ssp_below_data(self._profile(), 5510.0)

    def test_shallower_target_trims(self):
        import warnings as _w
        from uacpy.data.sound_speed import extend_ssp_below_data
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            out = extend_ssp_below_data(self._profile(), 2000.0)
        assert float(np.asarray(out.depths)[-1]) == pytest.approx(2000.0)

    def test_a_single_level_profile_extends(self):
        """The increment needs only the deepest sound speed, so a profile with
        no segment to measure is no longer a special case."""
        import warnings as _w
        from uacpy.core.environment import SoundSpeedProfile
        from uacpy.data.sound_speed import extend_ssp_below_data, _deep_increment
        p = SoundSpeedProfile.from_pairs(np.array([[1500.0, 1500.0]]))
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            out = extend_ssp_below_data(p, 3000.0, latitude=45.0)
        expected = 1500.0 + _deep_increment(1500.0, 1500.0, 3000.0, 45.0)
        assert float(np.asarray(out.sound_speed)[-1, 0]) == pytest.approx(expected)

    def test_each_column_is_extended_from_its_own_deep_value(self):
        """A range-dependent profile carries one column per range; a cold column
        and a warm one must not be given the same increment."""
        import warnings as _w
        from uacpy.core.environment import SoundSpeedProfile
        from uacpy.data.sound_speed import extend_ssp_below_data
        p = SoundSpeedProfile(
            depths=np.array([1000.0, 5500.0]),
            sound_speed=np.array([[1484.0, 1500.0], [1551.05, 1575.0]]),
            ranges=np.array([0.0, 50.0]), kind='measured')
        with _w.catch_warnings():
            _w.simplefilter('ignore')
            out = extend_ssp_below_data(p, 8801.0, latitude=29.78)
        cold, warm = np.asarray(out.sound_speed)[-1, :]
        assert (cold - 1551.05) - (warm - 1575.0) > 0.5, (
            f"increments {cold - 1551.05:.2f} and {warm - 1575.0:.2f} m/s — "
            f"dc/dz falls with temperature, so the colder column gains more")


def test_empty_bottom_sources_raises_a_typed_error():
    with pytest.raises(ConfigurationError, match='No data source was tried'):
        env_mod._fetch_bottom((), transect=False)


def _block_all_fetchers(monkeypatch):
    """Replace every fetcher ``fetch_environment`` can reach with a recorder,
    so a test can assert that a rejected call fetched nothing."""
    calls = []

    def _recorder(name):
        def fn(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"{name} was called")
        return fn

    for mod, name in [(bathy_mod, '_fetch_bathy_backend'),
                      (bathy_mod, '_fetch_bathy_transect_backend'),
                      (ssp_mod, '_fetch_ssp_backend'),
                      (ssp_mod, '_fetch_ssp_transect_backend'),
                      (copernicus, 'fetch_ssp_operational'),
                      (copernicus, 'fetch_ssp_transect_operational'),
                      (argo, 'fetch_ssp_argo'),
                      (env_mod, '_fetch_bottom'),
                      (seaice_local, 'fetch_sea_ice_surface'),
                      (seaice_local, 'sea_ice_surface_transect')]:
        monkeypatch.setattr(mod, name, _recorder(f'{mod.__name__}.{name}'))
    return calls


@pytest.mark.parametrize('axis', ['ssp', 'bathymetry', 'bottom', 'surface'])
def test_an_empty_sources_sequence_is_rejected_before_any_fetch(
        axis, monkeypatch):
    calls = _block_all_fetchers(monkeypatch)
    kwargs = {f'{axis}_sources': ()}
    with pytest.raises(ConfigurationError, match=f'{axis}_sources'):
        env_mod.fetch_environment((43.0, 7.5), date='2024-06-01',
                                      ssp=1500.0, bathymetry=200.0, **kwargs)
    assert calls == []


def test_an_empty_sources_list_is_rejected_like_an_empty_tuple(monkeypatch):
    calls = _block_all_fetchers(monkeypatch)
    with pytest.raises(ConfigurationError, match='selects no source'):
        env_mod.fetch_environment((43.0, 7.5), ssp=1500.0,
                                      bathymetry=200.0, bottom_sources=[])
    assert calls == []


def test_the_empty_sources_error_blames_the_sequence_not_the_local_preset(
        monkeypatch):
    _block_all_fetchers(monkeypatch)
    with pytest.raises(ConfigurationError,
                       match=r'ssp_sources=\(\) selects no source') as err:
        env_mod.fetch_environment((43.0, 7.5), ssp_sources=())
    assert 'selects no source' in err.value.message
    assert 'at least one source' in err.value.remediation
    # The diagnosis half must not claim a 'local' preset was passed; only
    # the remediation offers 'local' as an alternative.
    assert "'local'" not in err.value.message


@pytest.mark.parametrize('bad, says', [
    ('nope', "unknown bottom source 'nope'"),
    ((), 'selects no source'),
], ids=['unknown', 'empty'])
@pytest.mark.parametrize('caller', ['fetch_bottom', 'fetch_bottom_transect',
                                    'fetch_environment'])
def test_a_bad_bottom_source_is_refused_by_the_function_called(
        caller, bad, says, monkeypatch):
    """An unknown source id or an empty sequence is refused before any
    fetch, in the name of the public function the user called, with the
    presets listed — whichever of the three bottom entry points it came
    through."""
    calls = _block_all_fetchers(monkeypatch)
    point, end = (43.0, 7.5), (43.1, 7.6)
    call = {
        'fetch_bottom': lambda: env_mod.fetch_bottom(point, source=bad),
        'fetch_bottom_transect': lambda: env_mod.fetch_bottom_transect(
            point, end, source=bad),
        'fetch_environment': lambda: env_mod.fetch_environment(
            point, ssp=1500.0, bathymetry=200.0, bottom_sources=bad),
    }[caller]
    with pytest.raises(ConfigurationError,
                       match='selects no source|unknown bottom source') as err:
        call()
    assert err.value.message.startswith(f"{caller}: ")
    assert says in err.value.message
    assert "'auto', 'local'" in err.value.remediation
    assert calls == []


def test_empty_surface_sources_raise_instead_of_fetching_sea_ice(monkeypatch):
    calls = _block_all_fetchers(monkeypatch)
    with pytest.raises(ConfigurationError, match='surface_sources'):
        env_mod.fetch_environment((85.0, 0.0), date='2024-03-01',
                                      ssp=1450.0, bathymetry=500.0,
                                      surface_sources=())
    assert calls == []


def test_a_preset_inside_a_source_sequence_says_presets_go_alone():
    with pytest.raises(ConfigurationError, match='pass it alone'):
        env_mod._SSP_CHAIN.attempts(('local', 'woa23'), cache_only=False)


def test_auto_bottom_chain_prefers_a_measured_grain_size_sample():
    """'auto' must consult the cached NCEI grain-size database, and must do so
    before the modelled/interpolated maps behind it."""
    order, cache_only = env_mod._bottom_order('auto')
    assert cache_only is False
    assert 'grainsize' in order
    assert (order.index('emodnet') < order.index('grainsize')
            < order.index('diesing') < order.index('mars')
            < order.index('pelagic'))


def test_every_grain_size_provider_takes_the_environment_with_the_model():
    """Structural, because the failure mode is silence: ``_fetch_bottom``
    forwards ``environment`` to any provider it forwards ``model`` to, so a
    provider that took one and not the other would have a non-default
    environment dropped on the way to the fetcher and hand back a
    continental-terrace seabed without saying so. Asserted by introspection
    rather than by listing the providers, so a new one is covered the day it
    is registered."""
    import inspect
    for provider in env_mod._BOTTOM_PROVIDERS:
        if not provider.accepts_grain_size_model:
            continue
        for resolve_args in (((True,), (False,)) if provider.has_cached_variant
                             else ((),)):
            for fn in provider.resolve(*resolve_args):
                # A sampled transect passes its options to its point fetcher.
                fn = getattr(fn, 'point_fetcher', fn)
                params = inspect.signature(fn).parameters
                assert 'model' in params, f"{provider.id}: {fn.__name__}"
                assert 'hamilton_fit' in params, (
                    f"{provider.id}: {fn.__name__} takes model= but not "
                    f"hamilton_fit=, so the environment would be dropped")


def test_an_abyssal_environment_reaches_a_phi_literal_and_is_refused_with_apl_uw():
    """The knob has to work at the entry point people use, and refuse there
    too: a caller who names an abyssal fit beside ``bottom_model='apl-uw'``
    several layers up would otherwise have it silently dropped by a path that
    never reaches the conversion — a class-name seabed, say."""
    import warnings
    from uacpy.core.exceptions import ConfigurationError
    kwargs = dict(bathymetry=4000.0, ssp=1500.0, bottom=8.5)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        terrace = env_mod.fetch_environment((30.0, -40.0), **kwargs)
        abyssal = env_mod.fetch_environment((30.0, -40.0),
                                         bottom_environment='abyssal-plain',
                                         **kwargs)
    rho = [e.bottom.columns[0].halfspace.density for e in (terrace, abyssal)]
    assert rho[1] < rho[0]          # the abyssal fit is lighter at 8.5 ϕ
    assert abs(rho[1] - 1.39) < 0.02
    with pytest.raises(ConfigurationError, match='has no'):
        env_mod.fetch_environment((30.0, -40.0), bottom='sand',
                               bathymetry=4000.0, ssp=1500.0,
                               bottom_model='apl-uw',
                               bottom_environment='abyssal-hill')


@pytest.mark.parametrize('preset', ['auto', 'local'])
def test_grain_size_sample_beats_pelagic_ooze_off_cape_hatteras(preset):
    """At 36 N 75 W the grain-size database holds a sand sample 124 km away.
    Leaving it out of 'auto' handed back the pelagic model's ooze instead --
    rho*c 2245 against the sample's 3608, a 61 % error in the impedance that
    sets the bottom reflection coefficient."""
    import warnings
    from uacpy.data import sediment_db
    point = (36.0, -75.0)
    try:                                   # skip only when the cache is absent
        sample = sediment_db.fetch_bottom_grainsize(point)
    except Exception as exc:
        pytest.skip(f'grain-size cache not installed: {exc}')
    order, _ = env_mod._bottom_order(preset)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bottom, source = env_mod._fetch_bottom(
            order, point, transect=False, cache_only=True, depth=40.0)
    assert source == 'grainsize'
    assert bottom.sound_speed == pytest.approx(sample.sound_speed, rel=1e-9)
    # The sample is coarse sand, and coarse of ~1 ϕ the grain-size conversion
    # evaluates the Hamilton & Bachman (T) regression where it used to hold the
    # table's 0.92 ϕ end row, so this baseline moved up with it. The ratio is
    # carried back to the package's one water, the nominal 1500 m/s.
    assert bottom.sound_speed == pytest.approx(1805.5, abs=0.5)
    assert bottom.density == pytest.approx(2.152, abs=0.01)


def test_with_absorption_declares_a_fetched_ph_on_the_total_scale(
        monkeypatch, stub_fetchers):
    """GLODAP and the Copernicus BGC field report pH on the total scale;
    Francois–Garrison was fitted on NBS. The environment builder has to say
    which one it hands over, or the ~0.1 offset is silently lost."""
    monkeypatch.setattr(env_mod, '_fetch_ts_profile_backend',
                        lambda point, **kw: _ts_record())
    monkeypatch.setattr(env_mod, '_fetch_ph', lambda point, **kw: (7.9, 'glodap'))
    env = env_mod.fetch_environment((43.2, 7.5), with_absorption=True)
    assert env.absorption.pH == 7.9
    assert env.absorption.ph_scale == 'total'
    assert np.all(env.absorption.ph_nbs > 7.9)


def test_with_absorption_records_the_ph_node_it_read(monkeypatch,
                                                   stub_fetchers):
    """The GLODAP row in env.data_sources carries the node's point, so its
    offset is known like every other axis's: the bare catalogue id it had
    made offset_km None and round(p.offset_km) raise."""
    import uacpy.data.glodap_local as glodap_local
    from uacpy.data.sources import SOURCES, DataProvenance
    node = DataProvenance(source=SOURCES['glodap'], data_point=(43.5, 7.5),
                          requested_point=(43.2, 7.5))
    monkeypatch.setattr(env_mod, '_fetch_ts_profile_backend',
                        lambda point, **kw: _ts_record())
    monkeypatch.setattr(glodap_local, 'fetch_ph_profile',
                        lambda point, **kw: glodap_local.PHProfile(
                            depths=np.array([0.0, 1000.0, 2000.0]),
                            ph=np.array([8.1, 7.8, 7.7]), ph_scale='total',
                            provenance=node))
    env = env_mod.fetch_environment((43.2, 7.5), with_absorption=True)
    (record,) = [p for p in env.data_sources if p.source.id == 'glodap']
    assert record.data_point == (43.5, 7.5)
    assert record.offset_km == pytest.approx(33.4, abs=0.1)


def test_with_absorption_without_a_ph_source_keeps_the_model_default_as_nbs(
        monkeypatch, stub_fetchers):
    """The 8.1 fallback is a model default, not a measurement on any scale;
    it stays on the formula's own scale so a source-less run is unchanged."""
    from uacpy.core.constants import REFERENCE_PH
    monkeypatch.setattr(env_mod, '_fetch_ts_profile_backend',
                        lambda point, **kw: _ts_record())
    monkeypatch.setattr(env_mod, '_fetch_ph',
                        lambda point, **kw: (REFERENCE_PH, None))
    env = env_mod.fetch_environment((43.2, 7.5), with_absorption=True)
    assert env.absorption.ph_scale == 'nbs'
    assert np.all(env.absorption.ph_nbs == REFERENCE_PH)


@pytest.mark.parametrize('fetcher, args', [
    ('fetch_bottom', ((45.6, -6.2),)),
    ('fetch_bottom_transect', ((45.6, -6.2), (45.0, -6.0)))])
@pytest.mark.parametrize('model, refused', [
    ('aplu', True), ('hamiltonn', True), ('apl-uw', False), ('APL-UW', False),
    ('hamilton', False)])
def test_a_bottom_model_is_checked_before_any_source_is_tried(
        monkeypatch, fetcher, args, model, refused):
    """A misspelt model= surfaced as the last source's coverage failure ('raise
    max_distance_km'): the source that covered the point had swallowed the
    refusal and the chain moved on."""
    tried = []
    monkeypatch.setattr(env_mod, '_fetch_bottom',
                        lambda *a, **kw: tried.append(kw) or ('bottom', 'x'))
    call = getattr(env_mod, fetcher)
    if refused:
        with pytest.raises(ConfigurationError,
                           match=f"{fetcher}: unknown model {model!r}"):
            call(*args, source='local', model=model)
        assert tried == []
    else:
        try:
            call(*args, source='local', model=model)
        except ConfigurationError as exc:
            assert 'unknown model' not in str(exc)
        except Exception:
            pass


@pytest.mark.parametrize('flag', [True, False])
def test_a_bool_bottom_is_refused_rather_than_read_as_phi(flag):
    with pytest.raises(ConfigurationError, match='bottom must be'):
        env_mod._resolve_bottom(flag, water_sound_speed=1500.0)


def test_an_integer_bottom_is_a_grain_size():
    bp = env_mod._resolve_bottom(1, water_sound_speed=1500.0)
    assert isinstance(bp, BoundaryProperties)
    assert bp.grain_size_phi == 1.0


def test_bottom_model_selects_the_grain_size_relations_for_a_phi_literal():
    """The literal route honours the model: an APL-UW seabed is the
    apl-uw conversion at the same water speed, and a class name (its own
    numbers) ignores it."""
    from uacpy.core.sediment import grain_size_to_geoacoustics
    from uacpy.data import bottom_from_class
    apl = env_mod._resolve_bottom(2.0, water_sound_speed=1500.0, model='apl-uw')
    ham = env_mod._resolve_bottom(2.0, water_sound_speed=1500.0)
    expected = grain_size_to_geoacoustics(2.0, model='apl-uw',
                                          water_sound_speed=1500.0)
    assert apl.sound_speed == pytest.approx(expected['sound_speed'])
    assert apl.attenuation == pytest.approx(expected['attenuation'])
    assert apl.sound_speed != pytest.approx(ham.sound_speed)
    assert env_mod._resolve_bottom('sand', model='apl-uw') == \
        bottom_from_class('sand')


def test_an_unknown_bottom_model_is_refused_before_any_fetch(monkeypatch):
    calls = _block_all_fetchers(monkeypatch)
    with pytest.raises(ConfigurationError, match="bottom_model 'bachman'"):
        env_mod.fetch_environment((43.0, 7.5), bottom_model='bachman')
    assert calls == []


def test_bottom_model_reaches_every_grain_size_source(stub_fetchers):
    """End to end through a source: the pelagic seabed fetched under
    bottom_model='apl-uw' is the apl-uw conversion of its grain size at the
    seafloor water speed the environment holds, and differs from the
    default's."""
    from uacpy.core.sediment import grain_size_to_geoacoustics
    ham = env_mod.fetch_environment((43.2, 7.5), bottom_sources='pelagic')
    apl = env_mod.fetch_environment((43.2, 7.5), bottom_sources='pelagic',
                                    bottom_model='apl-uw')
    hs_h, hs_a = ham.bottom.columns[0].halfspace, apl.bottom.columns[0].halfspace
    assert hs_h.grain_size_phi == hs_a.grain_size_phi == pytest.approx(7.5)
    water_c = apl.ssp.sound_speed_at(apl.bathymetry.depth).item()
    for model, hs in (('hamilton', hs_h), ('apl-uw', hs_a)):
        expected = grain_size_to_geoacoustics(7.5, model=model,
                                              water_sound_speed=water_c)
        assert hs.sound_speed == pytest.approx(expected['sound_speed'])
        assert hs.attenuation == pytest.approx(expected['attenuation'])
    assert hs_a.sound_speed != pytest.approx(hs_h.sound_speed)


def test_every_grain_size_provider_takes_the_model_keyword():
    """The registry flag and the fetcher signatures agree: each provider
    that converts a grain size (point and transect fetcher, cached and live
    backend) accepts ``model``, and CRUST1.0 — measured layer properties, no
    grain size — is the one that does not, so ``_fetch_bottom`` never hands
    it a keyword it would refuse. A transect sampled from a point fetcher
    passes its options to that point fetcher, whose signature is checked."""
    import inspect
    for provider in env_mod._BOTTOM_PROVIDERS:
        arg_sets = ((True,), (False,)) if provider.has_cached_variant else ((),)
        for resolve_args in arg_sets:
            for fn in provider.resolve(*resolve_args):
                fn = getattr(fn, 'point_fetcher', fn)
                takes_model = 'model' in inspect.signature(fn).parameters
                assert takes_model == provider.accepts_grain_size_model, \
                    (provider.id, getattr(fn, '__name__', fn))
    flags = {p.id: p.accepts_grain_size_model for p in env_mod._BOTTOM_PROVIDERS}
    assert flags == {'emodnet': True, 'grainsize': True, 'crust1': False,
                     'graw': True, 'diesing': True, 'mars': True,
                     'pelagic': True}


@pytest.mark.parametrize('axis, spelling', [
    ('bathymetry_sources', 'GEBCO'),
    ('bathymetry_sources', 'Gebco'),
    ('bathymetry_sources', 'AUTO'),
    ('ssp_sources', 'WOA23'),
    ('ssp_sources', 'Woa23'),
    ('bottom_sources', 'PELAGIC'),
    ('bottom_sources', 'Pelagic'),
    ('bottom_sources', 'EMODNET'),
])
def test_source_names_resolve_in_any_case_on_every_route(
        monkeypatch, stub_fetchers, axis, spelling):
    """``'GEBCO'`` selects the same backend as ``'gebco'`` on the bathymetry,
    SSP and bottom routes, presets included: every route lowers its names
    the same way, so a spelling accepted on one route is accepted on all."""
    if axis == 'bottom_sources':
        # EMODnet is a network provider; the pelagic one reads the stubbed
        # bathymetry. Either way the name must pass the unknown-source check
        # before any fetch is tried, which the stub below observes.
        seen = []

        def _spy(order, *args, **kwargs):
            seen.append(order)
            raise env_mod.DataFetchError('stubbed bottom fetch')
        monkeypatch.setattr(env_mod, '_fetch_bottom', _spy)
        env_mod.fetch_environment((43.2, 7.5), bottom=1650.0,
                                  **{axis: spelling})
        assert seen and seen[0] == (spelling.lower(),)
        return
    env = env_mod.fetch_environment((43.2, 7.5), **{axis: spelling})
    assert env.depth == 2000.0


def test_a_mixed_case_surface_source_passes_the_name_check(monkeypatch):
    """``'SeaIce'`` is the sea-ice source, so with no ``date=`` the run
    reaches the documented literal fallback instead of an unknown-source
    refusal."""
    ice = BoundaryProperties(acoustic_type='half-space', sound_speed=3500.0,
                             density=0.9, attenuation=0.4, shear_speed=1800.0,
                             shear_attenuation=1.0)
    env = env_mod.fetch_environment((75.0, -10.0), bathymetry=2000.0,
                                    ssp=1500.0, surface_sources='SeaIce',
                                    surface=ice)
    assert env.surface.nodes[0].sound_speed == 3500.0


def test_an_unknown_source_is_refused_whatever_its_case(stub_fetchers):
    with pytest.raises(ConfigurationError, match="unknown bathymetry source"):
        env_mod.fetch_environment((43.2, 7.5), bathymetry_sources='NOPE')


@pytest.mark.parametrize('spec, presets_named', [('gebc0', True),
                                                 (['gebc0'], True),
                                                 (['gebco', 'auto'], False)])
def test_an_unknown_source_name_lists_the_presets_too(stub_fetchers, spec,
                                                      presets_named):
    """Every axis's remedy names 'auto' and 'local' beside the source ids, as
    the bottom axis's did; a preset put inside a sequence gets its own
    remedy instead."""
    with pytest.raises(ConfigurationError,
                       match='unknown bathymetry source') as err:
        env_mod.fetch_environment((43.2, 7.5), bathymetry_sources=spec)
    assert ("Use 'auto', 'local' or one of" in str(err.value)) == presets_named


def test_the_fetch_path_scales_sound_speed_in_situ_and_density_by_the_default():
    """``fetch_environment`` hands a grain-size seabed the in-situ seafloor
    sound speed but no water density, so its density ratio is carried
    against DEFAULT_WATER_DENSITY_G_CM3 whatever the deck's water density.
    The ``bottom_model`` doc says exactly that; this pins both halves."""
    import warnings
    from uacpy.core.sediment import grain_size_to_geoacoustics
    from uacpy.data import environment as data_env
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bottom = data_env._resolve_bottom(4.0, water_sound_speed=1530.0)
        geo = grain_size_to_geoacoustics(4.0, water_sound_speed=1530.0)
    assert bottom.sound_speed == pytest.approx(geo['sound_speed'])
    assert bottom.density == pytest.approx(geo['density'])
    doc = ' '.join(data_env.fetch_environment.__doc__.split())
    assert 'DENSITY ratio is carried against ``DEFAULT_WATER_DENSITY_G_CM3``' in doc


class TestFetchEnvironmentGivesOneProvenanceNotice:
    """Every provenance notice the fetchers give inside one
    ``fetch_environment`` call is folded into ONE ``ProvenanceWarning``
    naming each; any other warning passes on as it came."""

    @staticmethod
    def _notices(*emitted):
        import warnings
        from uacpy.core.exceptions import FallbackWarning, ProvenanceWarning
        from uacpy.data._provenance_notice import one_provenance_notice

        @one_provenance_notice(subject="this environment's data",
                               record='uacpy.data.citations(env)')
        def fetch_environment():
            for category, text in emitted:
                warnings.warn(text, {'P': ProvenanceWarning,
                                     'F': FallbackWarning}[category])
            return 'env'

        with recorded_warnings() as rec:
            assert fetch_environment() == 'env'
        return ([str(w.message) for w in rec
                 if issubclass(w.category, ProvenanceWarning)],
                [str(w.message) for w in rec
                 if issubclass(w.category, FallbackWarning)])

    def test_every_kind_of_item_lands_in_the_one_notice(self):
        (notice,), others = self._notices(
            ('P', 'fetch_sediment_sample: the data come from the sample at '
                  '62.08 N, 2.02 E, 206.5 km away: beyond 25 km. Pass '
                  'max_distance_km= to refuse data that far.'),
            ('P', 'CRUST1.0 has no formal licence — cite Laske et al. 2013 '
                  'and verify the terms. See uacpy.data.citations().'),
            ('F', 'local sediment DB: 3 of 5 transect waypoints have no '
                  'seabed data and were filled from the nearest covered '
                  'waypoint, up to 281 km away. Narrow the transect.'),
            ('F', 'sound-speed profile ends at 2900 m but the seafloor is at '
                  '3344 m; extrapolated the last 444 m along the profile\'s '
                  'deep gradient to 1502.1 m/s. Supply a measured profile.'))
        assert notice.startswith('fetch_environment: 4 note(s)')
        assert '206.5 km away: beyond 25 km;' in notice
        assert ('(2) CRUST1.0 has no formal licence — cite Laske et al. 2013 '
                'and verify the terms;') in notice
        assert 'up to 281 km away;' in notice
        assert 'to 1502.1 m/s.' in notice
        assert 'Pass max_distance_km=' not in notice
        assert others == []

    def test_a_fallback_that_is_not_provenance_passes_on_alone(self):
        notices, others = self._notices(
            ('F', 'fetch_environment: range-dependent SSP fetch failed (x); '
                  'falling back to the supplied ssp= literal.'))
        assert notices == []
        assert len(others) == 1 and 'falling back' in others[0]

    def test_no_item_no_notice_and_one_item_one_notice(self):
        assert self._notices() == ([], [])
        (notice,), _ = self._notices(
            ('P', 'CRUST1.0 has no formal licence. Cite it.'))
        assert notice.startswith('fetch_environment: 1 note(s)')

    def test_the_signature_is_the_fetchers_own(self):
        import inspect
        assert 'transect_to' in inspect.signature(
            env_mod.fetch_environment).parameters


class TestATransectGivesOneProvenanceNotice:
    """A transect fetcher samples a point fetcher at every waypoint; one call
    gives ONE ``ProvenanceWarning`` listing the samples' notices, and a
    decorated fetch run inside another passes its notices up to the outer
    one."""

    TRANSECTS = ['fetch_bottom_transect', 'fetch_ssp_transect',
                 'fetch_ssp_transect_operational',
                 'fetch_sea_ice_concentration_transect',
                 'sea_ice_surface_transect', 'fetch_seabed_density_transect',
                 'fetch_sediment_thickness_transect', 'fetch_wind_transect']

    @pytest.mark.parametrize('name', TRANSECTS)
    def test_every_sampling_transect_is_summarised(self, name):
        import uacpy.data as data
        fetch = getattr(data, name)
        assert fetch.__wrapped__ is not None
        assert fetch.__code__.co_name == 'fetch_with_one_provenance_notice'

    def test_a_transect_of_point_notices_gives_one_notice(self, monkeypatch):
        import warnings
        from uacpy.core.exceptions import ProvenanceWarning
        from uacpy.data import graw_local

        def node(lat, lon, **kw):
            warnings.warn(f"fetch_seabed_density: the data come from the "
                          f"node {lat:.2f} N, 30 km away. Pass "
                          f"max_distance_km=.", ProvenanceWarning)
            return 1.8
        monkeypatch.setattr(graw_local, 'graw_node', node)
        with recorded_warnings() as rec:
            graw_local.fetch_seabed_density_transect(
                (61.0, 2.0), (58.0, 5.0), n_points=3)
        (notice,) = [str(w.message) for w in rec
                     if issubclass(w.category, ProvenanceWarning)]
        assert notice.startswith('fetch_seabed_density_transect: 3 note(s)')
        assert '(3) fetch_seabed_density: the data come from the node 58.00 N' \
            in notice

    def test_a_summarised_fetch_inside_another_passes_its_notices_up(self):
        import warnings
        from uacpy.core.exceptions import ProvenanceWarning
        from uacpy.data._provenance_notice import one_provenance_notice

        @one_provenance_notice(subject='the samples', record='x')
        def inner():
            warnings.warn('inner one. More.', ProvenanceWarning)
            warnings.warn('inner two. More.', ProvenanceWarning)

        @one_provenance_notice(subject='the data', record='y')
        def outer():
            inner()
            warnings.warn('outer. More.', ProvenanceWarning)

        with recorded_warnings() as rec:
            outer()
        (notice,) = [str(w.message) for w in rec]
        assert notice == ('outer: 3 note(s) on where the data come from: '
                          '(1) inner one; (2) inner two; (3) outer. Each '
                          "source's full record is in y.")
