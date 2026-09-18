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


class TestBathymetryProvenanceNamesBackendAndVintage:
    def test_the_gebco_record_is_not_the_service_that_may_serve_it(self):
        from uacpy.data.sources import SOURCES
        assert SOURCES['gebco'].name == 'GEBCO grid'
        assert 'OpenTopoData' not in SOURCES['gebco'].name

    def test_the_vintage_says_which_backend_answered(self):
        from uacpy.data.environment import _bathymetry_vintage
        assert _bathymetry_vintage('gebco', 'api') == 'gebco2020 via OpenTopoData'
        assert _bathymetry_vintage('gmrt', 'gmrt') == 'gmrt (live)'
        assert _bathymetry_vintage('gebco', None) is None

    def test_the_local_grid_is_cited_by_its_release(self):
        from uacpy.data import gebco_local
        from uacpy.core.exceptions import ConfigurationError, DataFetchError
        try:
            name = gebco_local.grid_name()
        except (ConfigurationError, DataFetchError, FileNotFoundError):
            pytest.skip("no local GEBCO grid cached")
        assert name.startswith('GEBCO_')


class TestEveryPublicHelperOfTheDataLayerIsReachableFromThePackage:
    """What ``uacpy.data`` does itself, a caller can do by hand.

    Five helpers were declared public by their own modules, used by
    ``fetch_environment`` on the way to an answer, and re-exported nowhere —
    so reproducing one step of what the capstone did meant importing a
    sub-module and reading the source to find the name. They are exported now,
    and this drives each one on a worked input rather than only asserting the
    attribute exists, because an export that raises on its first argument is
    not a reachable feature either.
    """

    def test_a_profile_extends_below_the_deepest_level_its_source_carries(self):
        import numpy as np
        import uacpy

        # WOA23 stops at 5500 m; a 6000 m basin needs the rest of the column.
        ssp = uacpy.SoundSpeedProfile(depths=np.array([0.0, 1000.0, 5500.0]),
                                      data=np.array([1500.0, 1485.0, 1540.0]))
        deeper = uacpy.data.extend_ssp_below_data(ssp, 6000.0)
        assert deeper.depths[-1] >= 6000.0
        assert deeper.depths[-1] > ssp.depths[-1]
        # the gradient continues rather than the last value being held flat
        assert deeper.data[-1] > ssp.data[-1]

    def test_a_wave_height_inverts_to_the_wind_that_would_raise_it(self):
        import uacpy

        # Pierson-Moskowitz: Hs = 0.0214 U², so the inverse returns that U.
        assert uacpy.data.hs_to_pm_wind(0.0214 * 10.0 ** 2) == pytest.approx(10.0)
        assert uacpy.data.hs_to_pm_wind(0.0) == pytest.approx(0.0)

    def test_a_measured_density_inverts_to_the_grain_size_that_explains_it(self):
        import uacpy

        # The round trip through the pair: phi -> geoacoustics -> phi.
        phi = 5.0
        rows = uacpy.data.grain_size_to_geoacoustics(phi)
        back = uacpy.data.grain_size_from_density(rows['density'])
        assert back == pytest.approx(phi, abs=0.3)

    def test_the_names_the_guide_teaches_are_the_names_the_package_exports(self):
        import uacpy.data as data

        for name in ('extend_ssp_below_data', 'extend_column_to_seafloor',
                     'hs_to_pm_wind', 'grain_size_from_density',
                     'fetch_seabed_local'):
            assert name in data.__all__, name
            assert callable(getattr(data, name)), name
