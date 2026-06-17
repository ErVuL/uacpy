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
    assert env.bottom.grain_size_phi == 2.0


def test_bottom_from_class_name(stub_fetchers):
    env = env_mod.fetch_environment((43.2, 7.5), bottom='clay')
    assert env.bottom.acoustic_type == 'half-space'


def test_bottom_passthrough(stub_fetchers):
    bp = BoundaryProperties(acoustic_type='half-space', sound_speed=1700, density=1.8)
    env = env_mod.fetch_environment((43.2, 7.5), bottom=bp)
    assert env.bottom is bp


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
                                    range_dependent_ssp=True, ssp_source='copernicus',
                                    date='2026-01-01')
    assert env.ssp.is_range_dependent


def test_range_dependent_ssp_copernicus_requires_date(stub_fetchers):
    with pytest.raises(ConfigurationError, match='requires date'):
        env_mod.fetch_environment((43.2, 7.5), transect_to=(42.8, 8.1),
                                  range_dependent_ssp=True, ssp_source='copernicus')


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
    env = env_mod.fetch_environment((43.2, 7.5), bottom='auto')
    assert env.bottom is not None
    assert env.data_sources[-1].id == 'pelagic'


def test_range_dependent_bottom(monkeypatch, stub_fetchers):
    import uacpy.data.seabed as seabed_mod
    from uacpy.core.environment import RangeDependentBottom
    rdb = RangeDependentBottom(ranges=[0.0, 5000.0], sound_speed=[1650.0, 1600.0],
                               density=[1.9, 1.7], attenuation=[0.8, 1.0])
    monkeypatch.setattr(seabed_mod, 'fetch_bottom_transect',
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
        env_mod.fetch_environment((43.2, 7.5), ssp_source='copernicus')


def test_unknown_ssp_source(stub_fetchers):
    with pytest.raises(ConfigurationError, match='unknown ssp_source'):
        env_mod.fetch_environment((43.2, 7.5), ssp_source='nope')
