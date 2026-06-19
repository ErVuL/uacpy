"""Tests for the fetch_environment capstone (uacpy.data.environment)."""

import numpy as np
import pytest

from uacpy.core.environment import BoundaryProperties, Environment, SoundSpeedProfile
from uacpy.core.exceptions import ConfigurationError
from uacpy.data import environment as env_mod


@pytest.fixture
def stub_fetchers(monkeypatch):
    """Replace the network fetchers with deterministic stand-ins."""
    ssp = SoundSpeedProfile(depths=[0.0, 100.0, 2000.0],
                            data=[1500.0, 1490.0, 1510.0])
    monkeypatch.setattr(env_mod, 'fetch_bathy',
                        lambda point, **kw: 2000.0)
    monkeypatch.setattr(env_mod, 'fetch_bathy_transect',
                        lambda a, b, **kw: np.array([[0.0, 2000.0], [5000.0, 2200.0]]))
    monkeypatch.setattr(env_mod, 'fetch_ssp',
                        lambda point, **kw: ssp)
    return ssp


def test_point_environment(stub_fetchers):
    env = env_mod.fetch_environment((43.2, 7.5), date='2026-06-14')
    assert isinstance(env, Environment)
    assert env.depth == 2000.0
    assert env.name == '43.200, 7.500'
    assert not env.has_range_dependent_bathymetry()


def test_transect_environment(stub_fetchers):
    env = env_mod.fetch_environment((43.2, 7.5), transect_to=(42.8, 8.1), name='slope')
    assert env.has_range_dependent_bathymetry()
    assert env.name == 'slope'


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
    # A bare BoundaryProperties is coerced into a Bottom, preserving the halfspace.
    assert env.bottom.columns[0].halfspace is bp


def test_literal_ssp_skips_fetch(monkeypatch):
    # A literal ssp= is used verbatim; the SSP fetcher must NOT be called.
    monkeypatch.setattr(env_mod, 'fetch_bathy', lambda point, **kw: 2000.0)
    monkeypatch.setattr(env_mod, 'fetch_ssp',
                        lambda *a, **k: pytest.fail("SSP should not be fetched"))
    prof = SoundSpeedProfile(depths=[0.0, 2000.0], data=[1500.0, 1520.0])
    env = env_mod.fetch_environment((43.2, 7.5), ssp=prof)
    assert env.ssp.data[0] == 1500.0
    assert 'woa23' not in [s.id for s in env.data_sources]


def test_literal_bathymetry_skips_fetch(monkeypatch):
    # A literal bathymetry= is used verbatim; the bathy fetcher must NOT run.
    monkeypatch.setattr(env_mod, 'fetch_bathy',
                        lambda *a, **k: pytest.fail("bathy should not be fetched"))
    monkeypatch.setattr(env_mod, 'fetch_ssp',
                        lambda point, **kw: SoundSpeedProfile(depths=[0.0, 50.0],
                                                              data=[1500.0, 1490.0]))
    env = env_mod.fetch_environment((43.2, 7.5), bathymetry=120.0)
    assert env.depth == 120.0
    assert 'gebco' not in [s.id for s in env.data_sources]


def test_source_fetch_wins_over_literal(stub_fetchers):
    # Both given + fetch succeeds → the fetched value is used (literal ignored).
    env = env_mod.fetch_environment((43.2, 7.5), bathymetry=999.0,
                                    bathymetry_sources='gebco')
    assert env.depth == 2000.0                       # stub-fetched, not 999.0
    assert 'gebco' in [s.id for s in env.data_sources]


def test_bathymetry_sources_auto_prefers_gmrt(stub_fetchers):
    # 'auto' bathymetry = ('gmrt', 'gebco'); the stub fetcher succeeds, so the
    # first source (gmrt) wins and is recorded in provenance.
    env = env_mod.fetch_environment((43.2, 7.5), bathymetry_sources='auto')
    assert 'gmrt' in [s.id for s in env.data_sources]


def test_ssp_sources_auto_falls_to_woa23_without_date(stub_fetchers):
    # 'auto' ssp = ('argo','copernicus','woa23'); argo/copernicus need date=, so
    # with none they fall through to WOA23 (the stub).
    env = env_mod.fetch_environment((43.2, 7.5), ssp_sources='auto')
    assert 'woa23' in [s.id for s in env.data_sources]
    assert 'argo' not in [s.id for s in env.data_sources]


def test_literal_fallback_when_fetch_fails(monkeypatch):
    # Both given + fetch fails → fall back to the user literal (no raise).
    from uacpy.core.exceptions import DataFetchError
    monkeypatch.setattr(env_mod, 'fetch_bathy',
                        lambda *a, **k: (_ for _ in ()).throw(
                            DataFetchError("service down")))
    monkeypatch.setattr(env_mod, 'fetch_ssp',
                        lambda point, **kw: SoundSpeedProfile(depths=[0.0, 50.0],
                                                              data=[1500.0, 1490.0]))
    env = env_mod.fetch_environment((43.2, 7.5), bathymetry=120.0,
                                    bathymetry_sources='gebco')
    assert env.depth == 120.0                        # literal fallback
    assert 'gebco' not in [s.id for s in env.data_sources]


def test_range_dependent_ssp(monkeypatch, stub_fetchers):
    rd_ssp = SoundSpeedProfile(depths=[0.0, 100.0, 2000.0],
                               data=[[1500, 1502], [1490, 1492], [1510, 1512]],
                               ranges=[0.0, 5000.0])
    monkeypatch.setattr(env_mod, 'fetch_ssp_transect',
                        lambda start, end, **kw: rd_ssp)
    env = env_mod.fetch_environment((43.2, 7.5), transect_to=(42.8, 8.1),
                                    range_dependent_ssp=True)
    assert env.ssp.is_range_dependent
    assert env.has_range_dependent_bathymetry()


def test_range_dependent_ssp_requires_transect(stub_fetchers):
    with pytest.raises(ConfigurationError, match='requires transect_to'):
        env_mod.fetch_environment((43.2, 7.5), range_dependent_ssp=True)


def test_range_dependent_ssp_copernicus(monkeypatch, stub_fetchers):
    import uacpy.data.copernicus as cop_mod
    rd = SoundSpeedProfile(depths=[0.0, 100.0], data=[[1500, 1502], [1490, 1492]],
                           ranges=[0.0, 5000.0])
    monkeypatch.setattr(cop_mod, 'fetch_ssp_transect_operational',
                        lambda start, end, date, **kw: rd)
    env = env_mod.fetch_environment((43.2, 7.5), transect_to=(42.8, 8.1),
                                    range_dependent_ssp=True, ssp_sources='copernicus',
                                    date='2026-01-01')
    assert env.ssp.is_range_dependent


def test_range_dependent_ssp_copernicus_requires_date(stub_fetchers):
    with pytest.raises(ConfigurationError, match='requires date'):
        env_mod.fetch_environment((43.2, 7.5), transect_to=(42.8, 8.1),
                                  range_dependent_ssp=True, ssp_sources='copernicus')


def test_removed_global_keyword_rejected(stub_fetchers):
    # 'global' (the CC-BY-NC Dutkiewicz source) was removed; not a fetch source.
    with pytest.raises(ConfigurationError, match='unknown sediment class'):
        env_mod.fetch_environment((43.2, 7.5), bottom='global')


def test_bottom_auto_falls_back_to_pelagic(tmp_path, monkeypatch, stub_fetchers):
    # 'auto' now always resolves: outside measured coverage (and with no DBs
    # installed) it uses the global, commercial-clean pelagic model rather than
    # raising. An empty cache forces the fall-through past EMODnet/Diesing.
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    import uacpy.data.seabed as seabed_mod
    import uacpy.data.pelagic as pelagic_mod
    from uacpy.core.exceptions import DataFetchError

    def no_emodnet(point, **kw):
        raise DataFetchError("European seas only")
    monkeypatch.setattr(seabed_mod, 'fetch_bottom', no_emodnet)
    monkeypatch.setattr(pelagic_mod, '_water_depth', lambda *a, **k: 5000.0)
    env = env_mod.fetch_environment((43.2, 7.5), bottom_sources='auto')
    assert env.bottom is not None
    assert env.data_sources[-1].id == 'pelagic'


def test_range_dependent_bottom(monkeypatch, stub_fetchers):
    # 'auto' is cache-first, so EMODnet's local transect fetcher is tried first.
    import uacpy.data.emodnet_local as emodnet_mod
    from uacpy.core.environment import Bottom
    rdb = Bottom.from_halfspaces([0.0, 5000.0], sound_speed=[1650.0, 1600.0],
                               density=[1.9, 1.7], attenuation=[0.8, 1.0])
    monkeypatch.setattr(emodnet_mod, 'fetch_bottom_local_transect',
                        lambda start, end, **kw: rdb)
    env = env_mod.fetch_environment((43.2, 7.5), transect_to=(42.8, 8.1),
                                    range_dependent_bottom=True)
    assert env.bottom is rdb


def test_range_dependent_bottom_requires_transect(stub_fetchers):
    with pytest.raises(ConfigurationError, match='requires transect_to'):
        env_mod.fetch_environment((43.2, 7.5), range_dependent_bottom=True)


def test_with_absorption(monkeypatch, stub_fetchers):
    import uacpy.data.sound_speed as ss_mod
    from uacpy.core.absorption import FrancoisGarrison
    monkeypatch.setattr(ss_mod, 'fetch_ts_profile',
                        lambda point, **kw: (np.array([0.0, 50.0]),
                                                np.array([18.0, 16.0]),
                                                np.array([36.0, 36.1])))
    env = env_mod.fetch_environment((43.2, 7.5), with_absorption=True)
    assert isinstance(env.absorption, FrancoisGarrison)
    assert env.absorption.temperature_c == 18.0


def test_copernicus_requires_date(stub_fetchers):
    with pytest.raises(ConfigurationError, match='requires date'):
        env_mod.fetch_environment((43.2, 7.5), ssp_sources='copernicus')


def test_unknown_ssp_sources(stub_fetchers):
    with pytest.raises(ConfigurationError, match='unknown ssp source'):
        env_mod.fetch_environment((43.2, 7.5), ssp_sources='nope')
