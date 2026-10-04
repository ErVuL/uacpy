"""The data offset rule: every point fetcher records the point its value came
from, warns (``ProvenanceWarning``) once that point stands farther from the
request than the source's own threshold, and refuses (``DataFetchError``)
past a caller's ``max_distance_km``.

One decider (:func:`uacpy.data._geo.checked_offset`) applies the rule; the
pins below hold it on both sides of each threshold, and hold the per-source
thresholds themselves.
"""

import warnings

import numpy as np
import pytest

from uacpy.core.exceptions import (
    ConfigurationError, DataFetchError, ProvenanceWarning,
)
from uacpy.core.geo import EARTH_RADIUS_KM
from uacpy.data import _geo, argo, mars, sediment_db, seaice_local, wind_live
from uacpy.data.sources import SOURCES, DataProvenance


def _km_east(km):
    """Longitude (deg) of the point ``km`` east of (0, 0) along the equator."""
    return float(np.degrees(km / EARTH_RADIUS_KM))


def _record(data_point, requested=(0.0, 0.0)):
    return DataProvenance(source=SOURCES['woa23'], data_point=data_point,
                          requested_point=requested)


def _silent(call):
    """``call()`` with every warning raised as an error."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        return call()


# ── the decider ───────────────────────────────────────────────────────────────

class TestTheDecider:

    prov = _record((0.0, _km_east(30.0)))       # 30 km from the request

    def test_the_offset_is_the_great_circle_km(self):
        assert self.prov.offset_km == pytest.approx(30.0, rel=1e-9)

    def test_an_offset_just_inside_the_threshold_is_silent(self):
        got = _silent(lambda: _geo.checked_offset(
            self.prov, who='probe', warn_km=30.0 + 1e-6))
        assert got is self.prov

    def test_an_offset_just_past_the_threshold_warns_naming_it(self):
        with pytest.warns(ProvenanceWarning) as rec:
            got = _geo.checked_offset(self.prov, who='probe',
                                      warn_km=30.0 - 1e-6)
        assert got is self.prov
        (w,) = rec
        msg = str(w.message)
        assert msg.startswith('probe: World Ocean Atlas 2023')
        assert ('the data come from 0.00 N, 0.27 E, 30.0 km from the '
                'requested point (0.00 N, 0.00 E)') in msg
        assert 'max_distance_km=' in msg

    def test_a_limit_just_past_the_offset_keeps_the_data(self):
        got = _silent(lambda: _geo.checked_offset(
            self.prov, who='probe', warn_km=50.0,
            max_distance_km=30.0 + 1e-6))
        assert got is self.prov

    def test_a_limit_just_inside_the_offset_refuses_the_data(self):
        with pytest.raises(DataFetchError,
                           match=r'probe: .* 30\.0 km .* past max_distance_km=29\.9'):
            _geo.checked_offset(self.prov, who='probe', warn_km=50.0,
                                max_distance_km=29.9)

    def test_a_refusal_wins_over_the_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            with pytest.raises(DataFetchError, match='past max_distance_km=2'):
                _geo.checked_offset(self.prov, who='probe', warn_km=1.0,
                                    max_distance_km=2.0)

    @pytest.mark.parametrize('known', ['data_point', 'requested_point'])
    def test_an_unknown_point_passes_as_it_is(self, known):
        fields = {'data_point': None, 'requested_point': None,
                  known: (0.0, 1.0)}
        prov = DataProvenance(source=SOURCES['woa23'], **fields)
        assert prov.offset_km is None
        got = _silent(lambda: _geo.checked_offset(
            prov, who='probe', warn_km=0.0, max_distance_km=1e-9))
        assert got is prov


class TestTheLimitIsValidated:

    @pytest.mark.parametrize('value', [None, 5, 5.0, np.float64(0.5)])
    def test_none_or_a_positive_finite_km_passes(self, value):
        got = _geo.checked_max_distance(value, 'probe')
        assert got is None if value is None else got == float(value)

    @pytest.mark.parametrize('value', [0, 0.0, -1.0, float('nan'),
                                       float('inf'), True, '5'])
    def test_anything_else_is_a_configuration_error(self, value):
        with pytest.raises(ConfigurationError, match='probe: max_distance_km'):
            _geo.checked_max_distance(value, 'probe')


def test_a_cells_half_diagonal_is_the_farthest_point_inside_it():
    # 1° x 1° at the equator: half of hypot(111.19, 111.19) km.
    side = np.pi * EARTH_RADIUS_KM / 180.0
    assert _geo.cell_half_diagonal_km(0.0, 1.0, 1.0) == pytest.approx(
        0.5 * np.hypot(side, side), rel=1e-4)


@pytest.mark.parametrize('centre_lat', [0.5, 30.5, 45.5, 60.5, 80.5, -60.5])
def test_every_corner_of_a_cell_is_within_its_own_threshold(centre_lat):
    """The threshold is the cell's own geometry, at its centre. Sized at the
    requested latitude with a flat formula it fell short at the poleward
    corners: 67.795 km against 67.696 km for a 1° cell at 45.5° (the
    README's point A, (61.0, 2.0), was one such corner). A cell centre one
    more cell away along the diagonal stays outside."""
    from uacpy.core.geo import great_circle_km
    limit = _geo.cell_half_diagonal_km(centre_lat, 1.0, 1.0)
    for dlat in (-0.5, 0.5):
        for dlon in (-0.5, 0.5):
            corner = (centre_lat + dlat, 2.5 + dlon)
            own = float(great_circle_km(centre_lat, 2.5, *corner))
            beyond = float(great_circle_km(centre_lat + 4 * dlat,
                                           2.5 + 4 * dlon, *corner))
            assert own <= limit < beyond


def _cell_record(requested, centre=(45.5, 2.5), **fields):
    return DataProvenance(source=SOURCES['glodap'], data_point=centre,
                          requested_point=requested, point_kind='cell',
                          cell_size_deg=1.0, **fields)


@pytest.mark.parametrize('requested, warns', [
    ((46.0 - 1e-6, 2.0 + 1e-6), False),      # just inside a poleward corner
    ((45.0 + 1e-6, 3.0 - 1e-6), False),      # just inside an equatorward one
    ((46.5, 2.5), True),                     # the neighbouring cell's centre
    ((44.5, 1.5), True)])
def test_the_distance_rule_reads_the_cell_it_was_given(requested, warns):
    """Without a hop flag, a point inside the cell read is silent at every
    corner, and data from a neighbouring cell warn."""
    prov = _cell_record(requested)
    limit = _geo.cell_half_diagonal_km(prov.data_point[0], 1.0, 1.0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        _geo.checked_offset(prov, who='probe', warn_km=limit)
    assert len([w for w in rec
                if issubclass(w.category, ProvenanceWarning)]) == int(warns)


@pytest.mark.parametrize('hopped', [False, True])
def test_a_record_that_knows_its_cell_warns_exactly_on_a_hop(hopped):
    """A point near the edge of its dry cell, read from the wet neighbour
    across that edge, is closer to the neighbour's centre than any corner
    distance: only the flag can tell. A point near a far corner of its own
    cell is silent although its offset is larger."""
    requested = (45.0 + 1e-3, 2.5) if hopped else (46.0 - 1e-6, 2.0 + 1e-6)
    centre = (44.5, 2.5) if hopped else (45.5, 2.5)
    prov = _cell_record(requested, centre=centre,
                        from_neighbour_cell=hopped)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        # The distance rule alone would answer the other way on both
        # sides: silent past an infinite threshold, warning past a zero one.
        _geo.checked_offset(prov, who='probe', warn_km=float('inf') if hopped
                            else 0.0)
    msgs = [str(w.message) for w in rec
            if issubclass(w.category, ProvenanceWarning)]
    if hopped:
        assert prov.offset_km < _geo.cell_half_diagonal_km(44.5, 1.0, 1.0)
        assert len(msgs) == 1
        assert ('the data come from the nearest wet 1° cell, centred '
                '44.50 N, 2.50 E') in msgs[0]
        assert "own cell holds no data" in msgs[0]
    else:
        assert msgs == []


# ── per-source thresholds ─────────────────────────────────────────────────────

def test_the_sparse_source_thresholds():
    """Policy distances for sparse samples, not cell geometry: 50 km for an
    Argo cast (the open-ocean mesoscale a single cast stands for), 10 km for
    a grain-size sample or a MARS sample (seabed patchiness)."""
    assert argo.ARGO_OFFSET_WARN_KM == 50.0
    assert argo.ARGO_SEARCH_RADIUS_KM == 250.0
    assert sediment_db.SAMPLE_OFFSET_WARN_KM == 10.0
    assert sediment_db.SAMPLE_SEARCH_RADIUS_KM == 250.0
    assert mars.MARS_OFFSET_WARN_KM == 10.0
    assert mars.MARS_SEARCH_RADIUS_KM == 100.0


def test_the_gridded_source_thresholds():
    assert wind_live.NBS_GRID_DEG == 0.25


def test_a_sea_ice_cell_is_measured_on_the_ground():
    """The NSIDC 25 km cell is 25 km on the ground only at the projection's
    70° true-scale latitude: its half-diagonal there is half of
    hypot(25, 25) km, and poleward of it the cell is larger, so a fixed
    17.7 km threshold would warn inside a polar point's own cell."""
    pytest.importorskip('pyproj')
    # The NSIDC North grid: 448 rows x 304 columns.
    m = {'tf': {h: seaice_local._pyproj_transformer(
        seaice_local._GRID[h]['epsg']) for h in ('N', 'S')},
        'N': np.zeros((12, 448, 304)), 'S': np.zeros((12, 332, 316))}

    def own_cell_km(lat):
        rc = seaice_local._rowcol(m, 'N', lat, -45.0)
        return seaice_local._cell_half_diagonal_km(m, 'N', *rc)

    nominal = 0.5 * np.hypot(25.0, 25.0)
    assert own_cell_km(70.0) == pytest.approx(nominal, rel=0.01)
    assert own_cell_km(89.0) > nominal * 1.02


# ── a grain-size sample, both sides of 10 km ──────────────────────────────────

@pytest.fixture
def one_sample(monkeypatch):
    """A sample index holding one grain-size sample ``km`` east of (0, 0)."""
    def place(km):
        lon = _km_east(km)
        monkeypatch.setattr(sediment_db, '_samples', lambda: ('index', None))
        monkeypatch.setattr(sediment_db, '_nearest',
                            lambda index, lat, lon_: (km, 2.0, 0.0, lon))
    return place


def test_a_sample_inside_10_km_is_silent(one_sample):
    one_sample(9.9)
    s = _silent(lambda: sediment_db.fetch_sediment_sample((0.0, 0.0)))
    assert s.provenance.offset_km == pytest.approx(9.9, rel=1e-9)


def test_a_sample_past_10_km_warns(one_sample):
    one_sample(10.1)
    with pytest.warns(ProvenanceWarning,
                      match=r'the sample at 0\.00 N, 0\.09 E, 10\.1 km'):
        sediment_db.fetch_sediment_sample((0.0, 0.0))


def test_a_sample_past_the_limit_is_refused(one_sample):
    one_sample(10.1)
    with pytest.raises(DataFetchError, match='max_distance_km'):
        sediment_db.fetch_sediment_sample((0.0, 0.0), max_distance_km=10.0)


def test_without_a_limit_the_search_stops_at_250_km(one_sample):
    """The search radius keeps 'auto' falling through to the Diesing map
    and the pelagic model where no sample lies in the same sea."""
    one_sample(249.9)
    with pytest.warns(ProvenanceWarning):
        sediment_db.fetch_sediment_sample((0.0, 0.0))
    one_sample(250.1)
    with pytest.raises(DataFetchError, match='250 km, the search radius'):
        sediment_db.fetch_sediment_sample((0.0, 0.0))
    with pytest.warns(ProvenanceWarning):
        sediment_db.fetch_sediment_sample((0.0, 0.0), max_distance_km=300.0)


# ── an Argo cast, both sides of 50 km ─────────────────────────────────────────

_ARGO_HEADER = (
    "platform_number,cycle_number,direction,time,latitude,longitude,"
    "pres,temp,psal,temp_qc,psal_qc,pres_qc,position_qc,data_mode,"
    "pres_adjusted,temp_adjusted,psal_adjusted,pres_adjusted_qc,"
    "temp_adjusted_qc,psal_adjusted_qc\n"
    ",,,UTC,degrees_north,degrees_east,decibar,degree_Celsius,PSU,,,,,,"
    "decibar,degree_Celsius,PSU,,,\n")


def _argo_cast_at(monkeypatch, lat, lon):
    rows = "".join(
        f"4900001,1,A,2024-06-04T00:00:00Z,{lat},{lon},{p},{t},36,1,1,1,1"
        ",R,NaN,NaN,NaN,,,\n"
        for p, t in ((5, 20), (100, 15), (1000, 5)))
    monkeypatch.setattr(argo, 'http_get',
                        lambda url, **kw: _ARGO_HEADER + rows)


def test_an_argo_cast_inside_50_km_is_silent(monkeypatch):
    _argo_cast_at(monkeypatch, 0.0, _km_east(49.0))
    prof = _silent(lambda: argo.fetch_argo_profile((0.0, 0.0),
                                                   date='2024-06-04'))
    assert prof.provenance.offset_km == pytest.approx(49.0, abs=0.01)


def test_an_argo_cast_past_50_km_warns_and_a_limit_refuses_it(monkeypatch):
    _argo_cast_at(monkeypatch, 0.0, _km_east(51.0))
    with pytest.warns(ProvenanceWarning, match=r'the cast at .* 51\.0 km'):
        prof = argo.fetch_argo_profile((0.0, 0.0), date='2024-06-04')
    assert prof.provenance.offset_km == pytest.approx(51.0, abs=0.01)
    with pytest.raises(DataFetchError, match='50.9'):
        argo.fetch_argo_profile((0.0, 0.0), date='2024-06-04',
                                max_distance_km=50.9)
    with pytest.warns(ProvenanceWarning):
        argo.fetch_argo_profile((0.0, 0.0), date='2024-06-04',
                                max_distance_km=51.1)


# ── the date rule: a time offset is treated like a distance ──────────────────

from uacpy.data import _offset_policy    # noqa: E402


def _dated(source, data_date, requested_date, product=None):
    return DataProvenance(source=SOURCES[source], data_date=data_date,
                          requested_date=requested_date, product=product)


def test_offset_days_is_the_requested_date_minus_the_data_date():
    assert _dated('argo', '2025-02-09', '2025-02-14').offset_days == 5
    assert _dated('argo', '2025-02-19T06:00:00', '2025-02-14').offset_days == -5
    # A climatology period is no day: no offset.
    assert _dated('woa23', 'month 02 (climatology)',
                  '2025-02-14').offset_days is None
    assert _dated('argo', '2025-02-09', None).offset_days is None


def test_the_fetched_line_states_the_days_from_requested():
    line = _dated('argo', '2025-02-09', '2025-02-14')._fetch_line()
    assert 'date 2025-02-09, 5 days from requested 2025-02-14' in line


@pytest.mark.parametrize('source, product, data_date, requested, warns', [
    # Argo: 5 days stand, 6 do not.
    ('argo', None, '2025-02-09', '2025-02-14', False),
    ('argo', None, '2025-02-08', '2025-02-14', True),
    ('argo', None, '2025-02-20', '2025-02-14', True),       # either side
    # Copernicus daily physics: 3 days stand, 4 do not.
    ('copernicus', 'cmems_mod_glo_phy_my_0.083deg_P1D-m',
     '2025-02-11', '2025-02-14', False),
    ('copernicus', 'cmems_mod_glo_phy_my_0.083deg_P1D-m',
     '2025-02-10', '2025-02-14', True),
    # Waves (3-hourly WAVERYS) follow their product's daily rule.
    ('waverys', 'cmems_mod_glo_wav_my_0.2deg_PT3H-i',
     '2025-02-11', '2025-02-14', False),
    ('waverys', 'cmems_mod_glo_wav_my_0.2deg_PT3H-i',
     '2025-02-10', '2025-02-14', True),
    # pH from the monthly BGC reanalysis: the same calendar month stands
    # however many days apart; another month does not, even one day apart.
    ('copernicus_bgc', 'cmems_mod_glo_bgc_my_0.25deg_P1M-m',
     '2025-02-01', '2025-02-28', False),
    ('copernicus_bgc', 'cmems_mod_glo_bgc_my_0.25deg_P1M-m',
     '2025-02-28', '2025-03-01', True),
    # pH from the daily analysis/forecast follows the daily rule.
    ('copernicus_bgc', 'cmems_mod_glo_bgc-car_anfc_0.25deg_P1D-m',
     '2025-02-10', '2025-02-14', True),
    # A climatology carries no day, so no date rule.
    ('woa23', None, 'month 01 (climatology)', '2025-02-14', False),
])
def test_the_date_rule_warns_past_each_sources_threshold(
        source, product, data_date, requested, warns):
    prov = _dated(source, data_date, requested, product)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        got = _offset_policy.checked_date_offset(prov, who='probe')
    assert got is prov
    msgs = [str(w.message) for w in rec
            if issubclass(w.category, ProvenanceWarning)]
    assert len(msgs) == int(warns)
    if warns:
        assert f'dated {data_date}' in msgs[0] and 'max_days=' in msgs[0]


def test_the_thresholds_live_beside_the_distance_ones():
    assert _offset_policy.ARGO_OFFSET_WARN_DAYS == 5
    assert _offset_policy.COPERNICUS_DAILY_OFFSET_WARN_DAYS == 3
    assert argo.ARGO_OFFSET_WARN_KM is _offset_policy.ARGO_OFFSET_WARN_KM
    assert mars.MARS_OFFSET_WARN_KM is _offset_policy.MARS_OFFSET_WARN_KM
    assert (sediment_db.SAMPLE_OFFSET_WARN_KM
            is _offset_policy.SAMPLE_OFFSET_WARN_KM)


def _argo_cast_on(monkeypatch, day):
    rows = "".join(
        f"4900001,1,A,{day}T00:00:00Z,0.0,0.0,{p},{t},36,1,1,1,1"
        ",R,NaN,NaN,NaN,,,\n"
        for p, t in ((5, 20), (100, 15), (1000, 5)))
    monkeypatch.setattr(argo, 'http_get',
                        lambda url, **kw: _ARGO_HEADER + rows)


@pytest.mark.parametrize('day, warns', [('2024-05-30', False),
                                        ('2024-05-29', True)])
def test_an_argo_cast_warns_past_5_days(monkeypatch, day, warns):
    _argo_cast_on(monkeypatch, day)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        prof = argo.fetch_argo_profile((0.0, 0.0), date='2024-06-04')
    dated = [w for w in rec if 'the data are dated' in str(w.message)]
    assert len(dated) == int(warns)
    assert prof.provenance.offset_days == (5 if not warns else 6)


def test_a_transect_keeps_each_columns_date_offset():
    from uacpy.data.sound_speed import assemble_range_dependent
    import uacpy
    columns = [uacpy.SoundSpeedProfile(
        depths=[0.0, 100.0], sound_speed=[1500.0, 1490.0],
        data_sources=(_dated('argo', d, '2025-02-14'),))
        for d in ('2025-02-14', '2025-02-12', '2025-02-09')]
    ssp = assemble_range_dependent(columns, [0.0, 1000.0, 2000.0])
    assert [p.offset_days for p in ssp.data_sources] == [0, 2, 5]
    from uacpy.data.sources import citations
    assert '0–5 days from requested' in citations(ssp)


def test_the_date_fields_survive_the_dict_and_netcdf_round_trips(tmp_path):
    import uacpy
    rec = DataProvenance(source=SOURCES['woa23'],
                         data_date='month 02 (climatology)',
                         requested_date='2025-02-14', split_depth_m=1500.0,
                         period_below='annual mean')
    assert DataProvenance.from_dict(rec.to_dict()) == rec
    env = uacpy.Environment(bathymetry=3000.0, ssp=uacpy.SoundSpeedProfile(
        depths=[0.0, 3000.0], sound_speed=[1500.0, 1510.0],
        data_sources=(rec, _dated('argo', '2025-02-09', '2025-02-14'))))
    assert uacpy.Environment.from_dict(env.to_dict()).data_sources == \
        env.data_sources
    pytest.importorskip('xarray')
    pytest.importorskip('h5netcdf')
    env.to_netcdf(tmp_path / 'env.nc', engine='h5netcdf')
    back = uacpy.Environment.from_netcdf(tmp_path / 'env.nc',
                                         engine='h5netcdf')
    assert back.data_sources == env.data_sources
    assert back.data_sources[1].offset_days == 5
