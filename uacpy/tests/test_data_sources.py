"""Tests for the data-source catalogue + provenance (uacpy.data.sources)."""

import inspect

import pytest

import uacpy
from uacpy.core.environment import BoundaryProperties, SoundSpeedProfile
from uacpy.data import sources
from uacpy.data import environment as env_mod


def test_catalogue_pins_the_registry():
    # The documented catalogue: 21 entries, keyed by the *_sources ids.
    # 'deck41' is its own entry rather than being reported as 'grainsize':
    # the two local sediment indices are different datasets with different
    # licences and DOIs, and `fetch_sediment_sample` answers from whichever
    # is nearer, so citing both under one name credited a dataset the value
    # never touched.
    assert len(sources.SOURCES) == 21
    assert set(sources.SOURCES) == {
        'gebco', 'gmrt', 'emodnet_dtm', 'woa23', 'argo', 'copernicus',
        'glodap', 'copernicus_bgc', 'nbs', 'ww3', 'waverys', 'seaice',
        'emodnet', 'globsed', 'crust1', 'diesing', 'pelagic', 'mars',
        'graw', 'grainsize', 'deck41'}


def test_catalogue_complete_and_consistent():
    for key, src in sources.SOURCES.items():
        assert src.id == key
        assert src.name and src.license and src.attribution and src.citation
        assert isinstance(src.commercial_use, bool)
    # All sources permit commercial use except CRUST1.0, which ships with no
    # formal licence (flagged by citations() as "commercial use not confirmed").
    non_commercial = {s.id for s in sources.SOURCES.values()
                      if not s.commercial_use}
    assert non_commercial == {'crust1'}


def test_citations_whole_catalogue():
    text = sources.citations()
    assert 'GEBCO' in text and 'World Ocean Atlas' in text


def test_citations_from_ids():
    text = sources.citations(['woa23', 'emodnet'])
    assert 'World Ocean Atlas' in text and 'EMODnet' in text
    assert 'GEBCO' not in text


@pytest.fixture
def stub_fetchers(monkeypatch):
    ssp = SoundSpeedProfile(depths=[0.0, 100.0], data=[1500.0, 1490.0])
    monkeypatch.setattr(env_mod, 'fetch_bathy', lambda point, **kw: 2000.0)
    monkeypatch.setattr(env_mod, 'fetch_ssp', lambda point, **kw: ssp)


def test_environment_records_data_sources(stub_fetchers):
    env = env_mod.fetch_environment((43.2, 7.5), bottom=2.0)   # ϕ bottom, no fetch
    ids = [s.source.id for s in env.data_sources]
    assert ids == ['gebco', 'woa23']        # bathy + ssp, no fetched bottom
    # citations(env) renders just those.
    text = sources.citations(env)
    assert 'GEBCO' in text and 'EMODnet' not in text


def test_environment_records_fetched_bottom_sources(monkeypatch, stub_fetchers):
    import uacpy.data.seabed as seabed_mod
    from uacpy.core.environment import BoundaryProperties as BP
    monkeypatch.setattr(seabed_mod, 'fetch_bottom',
                        lambda point, **kw: BP(acoustic_type='half-space',
                                                  grain_size_phi=2.0, sound_speed=1650,
                                                  density=1.9))
    env = env_mod.fetch_environment((43.2, 7.5), bottom_sources='emodnet')
    ids = [s.source.id for s in env.data_sources]
    assert ids == ['gebco', 'woa23', 'emodnet']


def test_provenance_survives_environment_copy(stub_fetchers):
    env = env_mod.fetch_environment((43.2, 7.5), bottom=2.0)
    dup = env.copy()
    assert [s.source.id for s in dup.data_sources] == ['gebco', 'woa23']
    assert dup.data_sources[0].source == sources.SOURCES['gebco']


def test_union_dedupes_by_source_id_in_first_seen_order():
    # Carriers aggregate in axis order bathymetry → ssp → bottom; the ssp's
    # dated gebco record is first-seen, so the bottom's bare gebco duplicate
    # is dropped and the dated record survives.
    dated = sources.DataProvenance(source=sources.SOURCES['gebco'],
                                   data_date='2026-03-15')
    ssp = SoundSpeedProfile(
        depths=[0.0, 100.0], data=[1500.0, 1490.0],
        data_sources=(sources.DataProvenance(source=sources.SOURCES['woa23']),
                      dated))
    bottom = BoundaryProperties(
        acoustic_type='half-space', sound_speed=1650.0, density=1.9,
        data_sources=(sources.DataProvenance(source=sources.SOURCES['gebco']),
                      sources.DataProvenance(source=sources.SOURCES['emodnet'])))
    env = uacpy.Environment(bathymetry=100.0, ssp=ssp, bottom=bottom)
    assert [s.source.id for s in env.data_sources] == \
        ['woa23', 'gebco', 'emodnet']
    assert env.data_sources[1].data_date == '2026-03-15'


def test_argo_defaults_pin_search_radius_and_time_window():
    # The documented Argo nearest-cast guards: 250 km search radius, ±15 day
    # time window, on both the profile and SSP fetchers.
    from uacpy.data import argo
    assert argo.DEFAULT_MAX_DISTANCE_KM == 250.0
    assert argo.DEFAULT_MAX_DAYS == 15
    for fn in (argo.fetch_argo_profile, argo.fetch_ssp_argo):
        params = inspect.signature(fn).parameters
        assert params['max_distance_km'].default == 250.0
        assert params['max_days'].default == 15
