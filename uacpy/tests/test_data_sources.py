"""Tests for the data-source catalogue + provenance (uacpy.data.sources)."""

import inspect

import pytest

import uacpy
from uacpy.core.environment import BoundaryProperties, SoundSpeedProfile
from uacpy.data import sources
from uacpy.data import bathymetry, sound_speed
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


@pytest.mark.parametrize('key, doi', [
    ('deck41', '10.7289/V5VD6WCZ'),       # NCEI G02094 landing record
    ('seaice', '10.7265/a98x-0f50'),      # G02135 Version 4, the files read
    ('nbs', '10.3389/fmars.2022.935549'),  # the NBS v2 paper
])
def test_a_catalogue_doi_names_the_dataset_version_read(key, doi):
    """Each DOI was resolved at doi.org (2026-09-26) to the record of the
    product the fetcher reads; the README licensing table mirrors it."""
    from pathlib import Path
    assert f"doi:{doi}" in sources.SOURCES[key].citation
    readme = Path(uacpy.__file__).resolve().parents[1] / 'README.md'
    assert f"doi.org/{doi}" in readme.read_text(encoding='utf-8')


def test_the_sea_ice_index_citation_names_its_six_authors():
    """NSIDC's recommended citation of G02135 v4 (DataCite record, fetched
    2026-09-26), in the catalogue and in the README licensing table."""
    from pathlib import Path
    authors = ['Fetterer, F.', 'Knowles, K.', 'Meier, W.', 'Savoie, M.',
               'Windnagel, A.', 'Stafford, T.']
    citation = sources.SOURCES['seaice'].citation
    assert citation.startswith('Fetterer, F.')
    readme = Path(uacpy.__file__).resolve().parents[1] / 'README.md'
    row = next(line for line in readme.read_text(encoding='utf-8').splitlines()
               if 'a98x-0f50' in line)
    for text in (citation, row):
        positions = [text.index(a) for a in authors]
        assert positions == sorted(positions)


def test_citations_whole_catalogue():
    text = sources.citations()
    assert 'GEBCO' in text and 'World Ocean Atlas' in text


def test_citations_from_ids():
    text = sources.citations(['woa23', 'emodnet'])
    assert 'World Ocean Atlas' in text and 'EMODnet' in text
    assert 'GEBCO' not in text


@pytest.fixture
def stub_fetchers(monkeypatch):
    ssp = SoundSpeedProfile(depths=[0.0, 100.0], sound_speed=[1500.0, 1490.0])
    monkeypatch.setattr(bathymetry, '_fetch_bathy_backend', lambda point, **kw: 2000.0)
    monkeypatch.setattr(sound_speed, '_fetch_ssp_backend', lambda point, **kw: ssp)


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
    monkeypatch.setattr(seabed_mod, 'fetch_bottom_emodnet',
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


def test_union_drops_a_bare_record_its_source_already_describes():
    # Carriers aggregate in axis order bathymetry → ssp → bottom; the bottom's
    # bare gebco record says nothing the ssp's dated one does not, so it is
    # dropped and the dated record survives.
    dated = sources.DataProvenance(source=sources.SOURCES['gebco'],
                                   data_date='2026-03-15')
    ssp = SoundSpeedProfile(
        depths=[0.0, 100.0], sound_speed=[1500.0, 1490.0],
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
    # The documented Argo nearest-cast guards: 250 km search radius, a
    # 50 km offset warning, ±15 day time window; max_distance_km is the
    # user's refusal limit, unset by default, on both fetchers.
    from uacpy.data import argo
    assert argo.ARGO_SEARCH_RADIUS_KM == 250.0
    assert argo.ARGO_OFFSET_WARN_KM == 50.0
    assert argo.DEFAULT_MAX_DAYS == 15
    for fn in (argo.fetch_argo_profile, argo.fetch_ssp_argo):
        params = inspect.signature(fn).parameters
        assert params['max_distance_km'].default is None
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
                                      sound_speed=np.array([1500.0, 1485.0, 1540.0]))
        deeper = uacpy.data.extend_ssp_below_data(ssp, 6000.0)
        assert deeper.depths[-1] >= 6000.0
        assert deeper.depths[-1] > ssp.depths[-1]
        # the gradient continues rather than the last value being held flat
        assert deeper.sound_speed[-1] > ssp.sound_speed[-1]

    def test_a_wave_height_inverts_to_the_wind_that_would_raise_it(self):
        import uacpy

        # Pierson-Moskowitz: Hs = 0.0214 U², so the inverse returns that U.
        assert uacpy.data.hs_to_pm_wind(0.0214 * 10.0 ** 2) == pytest.approx(10.0)
        assert uacpy.data.hs_to_pm_wind(0.0) == pytest.approx(0.0)

    def test_a_measured_density_inverts_to_the_grain_size_that_explains_it(self):
        import uacpy

        # The round trip through the pair: phi -> geoacoustics -> phi.
        phi = 5.0
        rows = uacpy.core.sediment.grain_size_to_geoacoustics(phi)
        back = uacpy.core.sediment.grain_size_from_density(rows['density'])
        assert back == pytest.approx(phi, abs=0.3)

    def test_the_names_the_guide_teaches_are_the_names_the_package_exports(self):
        import uacpy.data as data

        for name in ('extend_ssp_below_data', 'extend_column_to_seafloor',
                     'hs_to_pm_wind', 'fetch_emodnet_substrate_local'):
            assert name in data.__all__, name
            assert callable(getattr(data, name)), name


class TestACatalogueRecordIsSharedAndImmutable:
    """A copied or loaded environment points at the catalogue's own
    DataSource (``__copy__`` / ``__deepcopy__`` return it), which is safe
    only because the record cannot change: frozen, and every field an
    immutable value."""

    _IMMUTABLE = (str, bool, int, float, type(None))

    @classmethod
    def _immutable(cls, value) -> bool:
        if isinstance(value, (tuple, frozenset)):
            return all(cls._immutable(v) for v in value)
        return isinstance(value, cls._IMMUTABLE)

    def test_every_field_of_every_record_is_immutable(self):
        import dataclasses
        from uacpy.data.sources import SOURCES
        for source in SOURCES.values():
            for f in dataclasses.fields(source):
                assert self._immutable(getattr(source, f.name)), \
                    (source.id, f.name)

    def test_a_field_cannot_be_set(self):
        import dataclasses
        from uacpy.data.sources import SOURCES
        with pytest.raises(dataclasses.FrozenInstanceError, match='name'):
            SOURCES['gebco'].name = 'x'

    def test_a_copy_is_the_record_itself(self):
        import copy
        from uacpy.data.sources import SOURCES
        assert copy.deepcopy(SOURCES['gebco']) is SOURCES['gebco']
        assert copy.copy(SOURCES['gebco']) is SOURCES['gebco']


class TestProvenanceIsData:
    """A provenance record saves as plain types and reads back, and
    ``provenance_table`` puts the datasets and the engine in one table."""

    @staticmethod
    def _prov():
        from uacpy.data import SOURCES, DataProvenance
        return DataProvenance(source=SOURCES['woa23'],
                              data_date='month 07 (climatology)',
                              data_point=(45.5, -6.5),
                              requested_point=(45.6, -6.2),
                              requested_date='2026-07-15')

    def test_a_record_round_trips_through_plain_types(self):
        import json
        from uacpy.data import DataProvenance
        prov = self._prov()
        d = prov.to_dict()
        assert d['source'] == 'woa23' and d['data_point'] == [45.5, -6.5]
        assert DataProvenance.from_dict(json.loads(json.dumps(d))) == prov

    def test_the_table_holds_the_data_and_the_engine(self):
        pytest.importorskip('pandas')
        import numpy as np
        from uacpy.core.results import Field
        from uacpy.data import provenance_table
        from uacpy.models.provenance import model_provenance

        class _Env:
            data_sources = (self._prov(),)

        result = Field(data=np.zeros((1, 2)),
                       coords={'depth': np.array([10.0]),
                               'range': np.array([100.0, 200.0])},
                       model='Bellhop', model_source=model_provenance('acoustics_toolbox'))
        table = provenance_table(_Env(), result)
        assert table['kind'].tolist() == ['data', 'engine']
        assert table['id'].tolist() == ['woa23', 'acoustics_toolbox']
        assert table['offset_km'].iloc[0] == pytest.approx(
            self._prov().offset_km)
        assert table['citation'].iloc[1] == model_provenance(
            'acoustics_toolbox').citation_for('Bellhop')



# ── a transect's provenance: one record per column, said in words ───────────

def _woa_column_record(requested, centre, hopped):
    return sources.DataProvenance(
        source=sources.SOURCES['woa23'], data_date='annual mean (climatology)',
        data_point=centre, requested_point=requested, point_kind='cell',
        cell_size_deg=1.0, from_neighbour_cell=hopped)


def _transect_environment():
    """A four-column SSP over 245 km, its columns read from WOA23 cells at
    78.3, 25.9, 56.9 and 0.0 km (the first a wet-cell hop), over a two-column
    seabed from two grain-size samples."""
    from uacpy.core.bottom import Bottom, SeabedColumn
    from uacpy.data.sound_speed import assemble_range_dependent
    cells = [((43.2, 4.6), (42.5, 4.5), True),
             ((42.6383, 5.2448), (42.5, 5.5), False),
             ((42.0730, 5.8808), (42.5, 5.5), False),
             ((41.5, 6.5), (41.5, 6.5), False)]
    columns = [SoundSpeedProfile(depths=[0.0, 1000.0],
                                 sound_speed=[1500.0, 1490.0 + i],
                                 data_sources=(_woa_column_record(*c),))
               for i, c in enumerate(cells)]
    ranges = [0.0, 81720.96, 163441.92, 245162.88]
    ssp = assemble_range_dependent(columns, ranges)
    samples = [sources.DataProvenance(
        source=sources.SOURCES['grainsize'], data_point=sample,
        requested_point=req, point_kind='sample')
        for req, sample in (((43.2, 4.6), (43.25, 4.65)),
                            ((41.5, 6.5), (41.45, 6.40)))]
    bottom = Bottom(columns=[SeabedColumn([], BoundaryProperties(
        acoustic_type='half-space', sound_speed=1650.0, density=1.9,
        data_sources=(s,))) for s in samples], ranges=[0.0, 245162.88])
    return uacpy.Environment(bathymetry=2000.0, ssp=ssp, bottom=bottom)


def test_every_column_keeps_its_record_at_its_range():
    """The union by source id kept the first column's woa23 record alone,
    so citations claimed its 78.3 km for the whole transect."""
    env = _transect_environment()
    woa = [p for p in env.data_sources if p.source.id == 'woa23']
    assert [round(p.offset_km, 1) for p in woa] == [78.3, 25.9, 56.9, 0.0]
    assert [p.range_m for p in woa] == list(env.ssp.ranges)
    seabed = [p for p in env.data_sources if p.source.id == 'grainsize']
    assert [p.range_m for p in seabed] == [0.0, 245162.88]


def test_a_source_read_at_several_points_is_cited_once_with_a_summary():
    text = sources.citations(_transect_environment())
    assert text.count('World Ocean Atlas 2023 (NOAA NCEI)  [') == 1
    assert text.count('Cite:') == 2
    assert ('  Fetched:     date annual mean (climatology); 4 points along '
            'the transect, 0.0–78.3 km away (largest at range 0 km: nearest '
            'wet 1° cell, centred 42.50 N, 4.50 E)') in text
    assert ('2 points along the transect, 6.9–10.0 km away (largest at range '
            '245.163 km: sample at 41.45 N, 6.40 E)') in text


def test_every_record_survives_the_dict_and_netcdf_round_trips(tmp_path):
    env = _transect_environment()
    back = uacpy.Environment.from_dict(env.to_dict())
    assert back.data_sources == env.data_sources
    pytest.importorskip('xarray')
    pytest.importorskip('h5netcdf')
    env.to_netcdf(tmp_path / 'env.nc', engine='h5netcdf')
    loaded = uacpy.Environment.from_netcdf(tmp_path / 'env.nc',
                                           engine='h5netcdf')
    assert loaded.data_sources == env.data_sources
    assert loaded.ssp.data_sources[0].from_neighbour_cell is True


@pytest.mark.parametrize('fields, words, line', [
    ({'point_kind': 'cell', 'cell_size_deg': 1.0,
      'from_neighbour_cell': True},
     'nearest wet 1° cell, centred 42.50 N, 4.50 E',
     'at the nearest wet 1° cell, centred 42.50 N, 4.50 E, 78.3 km from '
     'requested'),
    ({'point_kind': 'cell', 'cell_size_deg': 1.0,
      'from_neighbour_cell': False},
     '1° cell centred 42.50 N, 4.50 E',
     'at the 1° cell centred 42.50 N, 4.50 E, 78.3 km from requested'),
    ({'point_kind': 'cell', 'cell_size_deg': 1 / 12},
     '1/12° cell centred 42.50 N, 4.50 E', None),
    ({'point_kind': 'cell', 'cell_size_deg': 0.25},
     '0.25° cell centred 42.50 N, 4.50 E', None),
    ({'point_kind': 'cast'}, 'cast at 42.50 N, 4.50 E',
     'cast at 42.50 N, 4.50 E, 78.3 km from requested'),
    ({'point_kind': 'sample'}, 'sample at 42.50 N, 4.50 E',
     'sample at 42.50 N, 4.50 E, 78.3 km from requested'),
    ({}, '42.50 N, 4.50 E', 'at 42.50 N, 4.50 E, 78.3 km from requested'),
])
def test_the_data_point_is_said_in_words(fields, words, line):
    prov = sources.DataProvenance(source=sources.SOURCES['woa23'],
                                  data_point=(42.5, 4.5),
                                  requested_point=(43.2, 4.6), **fields)
    assert prov.describe_point() == words
    if line is not None:
        assert prov._fetch_line() == '  Fetched:     ' + line
    assert sources.DataProvenance.from_dict(prov.to_dict()) == prov


def test_the_southern_and_western_hemispheres_are_lettered():
    from uacpy.core.provenance import point_in_words
    assert point_in_words(-33.25, -70.5) == '33.25 S, 70.50 W'
    assert point_in_words(0.0, 0.0) == '0.00 N, 0.00 E'


def test_a_point_kind_outside_the_list_is_refused():
    from uacpy.core.exceptions import ConfigurationError
    with pytest.raises(ConfigurationError, match="point_kind must be one of"):
        sources.DataProvenance(source=sources.SOURCES['woa23'],
                               point_kind='node')


@pytest.mark.parametrize('fetcher, kind', [
    ('argo', 'cast'), ('grainsize', 'sample'), ('mars', 'sample')])
def test_a_sparse_sources_warning_names_its_kind_of_point(fetcher, kind):
    from uacpy.core.exceptions import ProvenanceWarning
    from uacpy.data import _geo
    prov = sources.DataProvenance(source=sources.SOURCES[fetcher],
                                  data_point=(42.5, 4.5),
                                  requested_point=(43.2, 4.6),
                                  point_kind=kind)
    with pytest.warns(ProvenanceWarning,
                      match=rf'the data come from the {kind} at 42\.50 N, '
                            rf'4\.50 E, 78\.3 km from the requested point '
                            rf'\(43\.20 N, 4\.60 E\)'):
        _geo.checked_offset(prov, who='probe', warn_km=10.0)
