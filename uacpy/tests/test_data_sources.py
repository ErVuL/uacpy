"""Tests for the data-source catalogue + provenance (uacpy.data.sources)."""

import pytest

from uacpy.core.environment import SoundSpeedProfile
from uacpy.data import sources
from uacpy.data import environment as env_mod


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
