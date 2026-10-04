"""Tests for the Copernicus operational SSP fetch (uacpy.data.copernicus).

The ``copernicusmarine`` toolbox is an optional dependency and is not
installed in CI, so these tests stub it (and a minimal xarray-like dataset)
to exercise the extraction and error paths offline.
"""

import sys
import types

import numpy as np
import pytest

from uacpy.core.environment import SoundSpeedProfile
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data.sources import DataProvenance
from uacpy.data import copernicus


class _DAStub:
    """Minimal xarray.DataArray stand-in: ``.sel(...).values``."""
    def __init__(self, values):
        self._v = np.asarray(values, dtype=float)

    def sel(self, **kwargs):       # ignore selectors; single column fixture
        return self

    @property
    def values(self):
        return self._v


class _Coord:
    def __init__(self, value):
        self.values = np.datetime64(value)


class _TimeDAStub(_DAStub):
    """DataArray stub that also exposes a fixed ``time`` coordinate."""
    def __init__(self, values, time):
        super().__init__(values)
        self.coords = {'time': _Coord(time)}

    def __getitem__(self, key):        # t_da['time'] → the time coord
        return self.coords[key]


class _DSStub:
    def __init__(self, depth, thetao, so, time=None):
        mk = (lambda v: _TimeDAStub(v, time)) if time else _DAStub
        self._vars = {'depth': _DAStub(depth), 'thetao': mk(thetao), 'so': mk(so)}

    def __getitem__(self, key):
        return self._vars[key]


_DEPTH = [0.0, 100.0, 1000.0, 3000.0]
_T = [22.0, 15.0, 5.0, 3.0]
_S = [36.0, 36.2, 35.0, 34.9]


def _install_fake_toolbox(monkeypatch, dataset):
    fake = types.ModuleType('copernicusmarine')
    fake.open_dataset = lambda **kwargs: dataset
    monkeypatch.setitem(sys.modules, 'copernicusmarine', fake)


def test_extract_ts_truncates_at_seafloor():
    ds = _DSStub(_DEPTH, [22.0, 15.0, np.nan, np.nan], [36.0, 36.2, np.nan, np.nan])
    z, t, s, _ = copernicus._extract_ts(ds, 30.0, -40.0, '2020-06-01')
    assert z.tolist() == [0.0, 100.0]
    assert t.tolist() == [22.0, 15.0]


def test_fetch_ssp_operational_end_to_end(monkeypatch):
    _install_fake_toolbox(monkeypatch, _DSStub(_DEPTH, _T, _S))
    ssp = copernicus.fetch_ssp_operational((30.0, -40.0), date='2020-06-15')
    assert isinstance(ssp, SoundSpeedProfile)
    assert ssp.depths.tolist() == _DEPTH
    assert np.all((1440 < ssp.sound_speed) & (ssp.sound_speed < 1560))


def test_fetch_ssp_transect_operational(monkeypatch):
    _install_fake_toolbox(monkeypatch, _DSStub(_DEPTH, _T, _S))
    ssp = copernicus.fetch_ssp_transect_operational(
        (30.0, -40.0), (31.0, -40.0), date='2020-06-15', n_points=4)
    assert ssp.is_range_dependent
    assert ssp.sound_speed.shape == (len(_DEPTH), 4)
    assert ssp.ranges[0] == 0.0


def test_the_operational_transect_takes_the_woa23_sampling_keywords(
        monkeypatch):
    _install_fake_toolbox(monkeypatch, _DSStub(_DEPTH, _T, _S))
    auto = copernicus.fetch_ssp_transect_operational(
        (30.0, -40.0), (31.0, -40.0), date='2020-06-15')
    assert auto.sound_speed.shape[1] == copernicus.AUTO_TRANSECT_COLUMNS
    capped = copernicus.fetch_ssp_transect_operational(
        (30.0, -40.0), (31.0, -40.0), date='2020-06-15', max_points=3)
    assert capped.sound_speed.shape[1] == 3
    with pytest.warns(UserWarning, match='max_points'):
        over = copernicus.fetch_ssp_transect_operational(
            (30.0, -40.0), (31.0, -40.0), date='2020-06-15', n_points=4,
            max_points=3)
    assert over.sound_speed.shape[1] == 3
    at_cap = copernicus.fetch_ssp_transect_operational(
        (30.0, -40.0), (31.0, -40.0), date='2020-06-15', n_points=3,
        max_points=3)
    assert at_cap.sound_speed.shape[1] == 3
    for bad in (2.5, 1, 'x'):
        with pytest.raises(ConfigurationError, match='n_points'):
            copernicus.fetch_ssp_transect_operational(
                (30.0, -40.0), (31.0, -40.0), date='2020-06-15', n_points=bad)


def test_mackenzie_reads_the_model_depths_directly(monkeypatch):
    # Mackenzie is stated in depth: evaluated on the model depths with the
    # in-situ temperature, no pressure round trip at the reference latitude.
    from uacpy.core.acoustics import sound_speed_mackenzie
    from uacpy.core.acoustics.seawater import depth_to_pressure_dbar, insitu_from_potential
    _install_fake_toolbox(monkeypatch, _DSStub(_DEPTH, _T, _S))
    z, s = np.array(_DEPTH), np.array(_S)
    for lat in (0.0, 60.0):
        t = insitu_from_potential(salinity=s, theta=np.array(_T),
                                  pressure_dbar=depth_to_pressure_dbar(z, lat))
        expected = sound_speed_mackenzie(temperature=t, salinity=s, depth=z)
        point = copernicus.fetch_ssp_operational(
            (lat, -40.0), date='2020-06-15', formula='mackenzie')
        np.testing.assert_allclose(point.sound_speed[:, 0], expected, rtol=0, atol=1e-9)
        transect = copernicus.fetch_ssp_transect_operational(
            (lat, -40.0), (lat, -39.5), date='2020-06-15',
            n_points=2, formula='mackenzie')
        np.testing.assert_allclose(transect.sound_speed[:, 0], expected, rtol=0,
                                   atol=1e-6)


def test_missing_toolbox_raises_helpful_error(monkeypatch):
    # Force the import to fail even if the package were present.
    monkeypatch.setitem(sys.modules, 'copernicusmarine', None)
    with pytest.raises(DataFetchError, match='copernicusmarine'):
        copernicus.fetch_ts_profile_operational((0.0, 0.0), date='2020-01-01')


def test_open_dataset_failure_wrapped(monkeypatch):
    fake = types.ModuleType('copernicusmarine')

    def boom(**kwargs):
        raise RuntimeError("auth failed")
    fake.open_dataset = boom
    monkeypatch.setitem(sys.modules, 'copernicusmarine', fake)
    with pytest.raises(DataFetchError, match='open_dataset failed'):
        copernicus.fetch_ssp_operational((0.0, 0.0), date='2020-01-01')


def test_bad_formula_raises():
    with pytest.raises(ConfigurationError, match='formula'):
        copernicus.fetch_ssp_operational((0.0, 0.0), date='2020-01-01', formula='x')


def test_bad_date_raises(monkeypatch):
    _install_fake_toolbox(monkeypatch, _DSStub(_DEPTH, _T, _S))
    with pytest.raises(ConfigurationError, match='parse date'):
        copernicus.fetch_ssp_operational((0.0, 0.0), date='not-a-date')


def test_out_of_range_date_raises(monkeypatch):
    # Dataset's only time step is 2021; asking for 2030 is far beyond max_days,
    # so the nearest-edge value is rejected rather than silently substituted.
    _install_fake_toolbox(monkeypatch, _DSStub(_DEPTH, _T, _S, time='2021-01-15'))
    with pytest.raises(DataFetchError, match='outside the'):
        copernicus.fetch_ssp_operational((30.0, -40.0), date='2030-06-15')


def test_out_of_range_within_widened_max_days_ok(monkeypatch):
    # The same edge case succeeds once max_days is widened past the gap.
    _install_fake_toolbox(monkeypatch, _DSStub(_DEPTH, _T, _S, time='2021-01-15'))
    ssp = copernicus.fetch_ssp_operational(
        (30.0, -40.0), date='2021-02-01', max_days=60)
    assert isinstance(ssp, SoundSpeedProfile)


def test_in_range_date_ok(monkeypatch):
    _install_fake_toolbox(monkeypatch, _DSStub(_DEPTH, _T, _S, time='2020-06-15'))
    ssp = copernicus.fetch_ssp_operational((30.0, -40.0), date='2020-06-15')
    assert isinstance(ssp, SoundSpeedProfile)


# ── BGC pH ──────────────────────────────────────────────────────────────────

class _BGCStub:
    """BGC dataset stub: a ``ph`` depth column (with optional time coord)."""
    def __init__(self, ph, time=None, depth=None):
        mk = (lambda v: _TimeDAStub(v, time)) if time else _DAStub
        if depth is None:
            depth = [0.0, 500.0, 2000.0][:len(ph)]
        self._vars = {'ph': mk(ph), 'depth': _DAStub(depth)}

    def __getitem__(self, key):
        return self._vars[key]


def test_fetch_ph_operational_defaults_to_the_level_nearest_mid_depth(
        monkeypatch):
    # Levels 0 / 500 / 2000 m: the mid-depth 1000 m is nearest the 500 m level.
    _install_fake_toolbox(monkeypatch, _BGCStub([8.05, 8.00, 7.90]))
    ph = copernicus.fetch_ph_operational((30.0, -40.0), date='2020-06-15')
    assert ph == pytest.approx(8.00)
    assert ph == copernicus.fetch_ph_operational(
        (30.0, -40.0), date='2020-06-15', reference_depth=1000.0)


def test_fetch_ph_operational_reference_depth_picks_the_nearest_level(
        monkeypatch):
    _install_fake_toolbox(monkeypatch, _BGCStub([8.05, 8.00, 7.90]))
    assert copernicus.fetch_ph_operational(
        (30.0, -40.0), date='2020-06-15',
        reference_depth=0.0) == pytest.approx(8.05)


def test_fetch_ph_operational_drops_masked_levels_before_the_mid_depth(
        monkeypatch):
    # Only the finite levels (500, 2000 m) define the column: mid-depth 1250 m
    # is nearest 500 m.
    _install_fake_toolbox(monkeypatch, _BGCStub([np.nan, 8.00, 7.90]))
    ph = copernicus.fetch_ph_operational((30.0, -40.0), date='2020-06-15')
    assert ph == pytest.approx(8.00)


def test_fetch_ph_operational_land_raises(monkeypatch):
    _install_fake_toolbox(monkeypatch, _BGCStub([np.nan, np.nan, np.nan]))
    with pytest.raises(DataFetchError, match='No Copernicus pH'):
        copernicus.fetch_ph_operational((0.0, 0.0), date='2020-06-15')


def test_fetch_ph_operational_out_of_range_date(monkeypatch):
    _install_fake_toolbox(monkeypatch,
                          _BGCStub([8.05, 8.00, 7.90], time='2021-01-15'))
    with pytest.raises(DataFetchError, match='outside the'):
        copernicus.fetch_ph_operational((30.0, -40.0), date='2030-06-15')


# ── fetch_environment wiring: live BGC pH on the Copernicus SSP branch ──────

def _install_routing_toolbox(monkeypatch, *, bgc):
    """Physics datasets → the T/S stub; BGC dataset ids → ``bgc``."""
    fake = types.ModuleType('copernicusmarine')
    physics = _DSStub(_DEPTH, _T, _S)

    def open_dataset(*, dataset_id):
        if 'bgc' in dataset_id:
            if isinstance(bgc, Exception):
                raise bgc
            return bgc
        return physics
    fake.open_dataset = open_dataset
    monkeypatch.setitem(sys.modules, 'copernicusmarine', fake)


def test_environment_copernicus_ssp_prefers_bgc_ph(monkeypatch, tmp_path):
    import uacpy.data as data
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    _install_routing_toolbox(monkeypatch, bgc=_BGCStub([8.02, 7.95, 7.90]))
    env = data.fetch_environment((30.0, -40.0), bathymetry=1000.0,
                                 ssp_sources='copernicus', date='2020-06-15',
                                 with_absorption=True)
    # pH is read at the Francois-Garrison nominal-row depth (the T/S column
    # mid-depth), so the stub level nearest that depth wins — not the 8.02
    # surface value the old surface-pH pairing returned.
    assert env.absorption.pH == pytest.approx(7.90)
    assert 'copernicus_bgc' in [s.source.id for s in env.data_sources]


def test_environment_copernicus_absorption_row_is_in_situ_temperature(
        monkeypatch, tmp_path):
    import uacpy.data as data
    from uacpy.core.acoustics.seawater import depth_to_pressure_dbar, insitu_from_potential
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    _install_routing_toolbox(monkeypatch, bgc=RuntimeError("no bgc"))
    env = data.fetch_environment((30.0, -40.0), bathymetry=1000.0,
                                 ssp_sources='copernicus', date='2020-06-15',
                                 with_absorption=True)
    # The column is kept whole; at its 1000 m level the dataset's thetao is
    # 5.0 °C potential, and Francois-Garrison takes the in-situ value at
    # that pressure, as the sound-speed route already does.
    assert env.absorption.is_profile
    expected = float(insitu_from_potential(
        salinity=35.0, theta=5.0,
        pressure_dbar=depth_to_pressure_dbar(1000.0, 30.0)))
    assert expected != 5.0
    t_pairs = env.absorption.temperature
    at_1000 = int(np.flatnonzero(t_pairs[:, 0] == 1000.0)[0])
    assert t_pairs[at_1000, 1] == pytest.approx(expected)


def test_environment_bgc_failure_falls_back(monkeypatch, tmp_path):
    import uacpy.data as data
    from uacpy.core.constants import REFERENCE_PH
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    _install_routing_toolbox(monkeypatch, bgc=RuntimeError("auth failed"))
    env = data.fetch_environment((30.0, -40.0), bathymetry=1000.0,
                                 ssp_sources='copernicus', date='2020-06-15',
                                 with_absorption=True)
    # No BGC, no GLODAP cache → the model-default constant, silently.
    assert env.absorption.pH == pytest.approx(REFERENCE_PH)
    assert 'copernicus_bgc' not in [s.source.id for s in env.data_sources]


def test_fetch_ph_woa_source_does_not_hit_bgc(monkeypatch, tmp_path):
    # A non-Copernicus SSP source must not silently open a Copernicus dataset
    # for pH — the BGC preference rides the existing Copernicus session only.
    from uacpy.data import environment
    from uacpy.core.constants import REFERENCE_PH
    calls = []
    fake = types.ModuleType('copernicusmarine')

    def open_dataset(*, dataset_id):
        calls.append(dataset_id)
        raise RuntimeError("should not be called")
    fake.open_dataset = open_dataset
    monkeypatch.setitem(sys.modules, 'copernicusmarine', fake)
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    pH, src = environment._fetch_ph((30.0, -40.0), date='2020-06-15',
                                    ssp_source='woa23', cache_only=False,
                                    timeout=5.0, verbose=False)
    assert pH == pytest.approx(REFERENCE_PH)
    assert src is None
    assert calls == []


# ── provenance stamping ─────────────────────────────────────────────────────

def test_fetch_ssp_operational_stamps_provenance(monkeypatch):
    _install_fake_toolbox(monkeypatch, _DSStub(_DEPTH, _T, _S))
    ssp = copernicus.fetch_ssp_operational((30.0, -40.0), date='2020-06-15')
    assert len(ssp.data_sources) == 1
    prov = ssp.data_sources[0]
    assert prov.source.id == 'copernicus'
    assert prov.requested_point == (30.0, -40.0)
    assert prov.requested_date == '2020-06-15'


def test_fetch_ssp_transect_operational_stamps_provenance(monkeypatch):
    _install_fake_toolbox(monkeypatch, _DSStub(_DEPTH, _T, _S))
    ssp = copernicus.fetch_ssp_transect_operational(
        (30.0, -40.0), (31.0, -40.0), date='2020-06-15', n_points=4)
    assert [p.source.id for p in ssp.data_sources] == ['copernicus'] * 4
    assert [p.range_m for p in ssp.data_sources] == list(ssp.ranges)


@pytest.mark.parametrize('dataset_id', [
    copernicus.PHYSICS_PRODUCTS[0][0],
    'cmems_mod_glo_phy_anfc_0.083deg_PT1H-m'])
def test_the_operational_record_names_the_product_it_read(monkeypatch,
                                                          dataset_id):
    """The Copernicus licence asks for the product's DOI, so the record names
    the dataset read, and the rendered attribution carries it."""
    from uacpy.data.sources import citations
    _install_fake_toolbox(monkeypatch, _DSStub(_DEPTH, _T, _S))
    point = copernicus.fetch_ssp_operational((30.0, -40.0), date='2020-06-15',
                                             dataset_id=dataset_id)
    line = copernicus.fetch_ssp_transect_operational(
        (30.0, -40.0), (31.0, -40.0), date='2020-06-15', n_points=4,
        dataset_id=dataset_id)
    assert point.data_sources[0].product == dataset_id
    assert line.data_sources[0].product == dataset_id
    text = citations(point)
    assert f'product {dataset_id}' in text
    assert '<product DOI>' not in text


# ── the product is picked by date ───────────────────────────────────────────

class _Times:
    def __init__(self, first, last):
        self.values = np.array([first, last], dtype='datetime64[ns]')


class _Dataset:
    """A dataset stub holding the named variables and a time axis."""
    def __init__(self, first, last, **variables):
        self._vars = {name: _DAStub(v) for name, v in variables.items()}
        self._vars['time'] = _Times(first, last)

    def __getitem__(self, key):
        return self._vars[key]


# Coverage from the Copernicus Marine catalogue (2026-09-26).
_REANALYSIS_END = '2026-06-23'


def _install_dated_products(monkeypatch):
    """The two physics products and two BGC datasets, each with its own time
    axis and a salinity that tells them apart; returns the opened ids."""
    (my_t, my_s), (fc_t, fc_s) = copernicus.PHYSICS_PRODUCTS
    bgc_my, bgc_fc = copernicus.BGC_PRODUCTS
    datasets = {
        my_t: _Dataset('1993-01-01', _REANALYSIS_END, depth=_DEPTH,
                       thetao=_T, so=_S),
        fc_t: _Dataset('2022-06-01', '2026-10-03', depth=_DEPTH, thetao=_T),
        fc_s: _Dataset('2022-06-01', '2026-10-03', so=[35.0] * len(_DEPTH)),
        bgc_my: _Dataset('1993-01-01', '2026-05-01', depth=[0.0, 500.0],
                         ph=[8.10, 8.00]),
        bgc_fc: _Dataset('2021-11-01', '2026-10-03', depth=[0.0, 500.0],
                         ph=[8.05, 7.95]),
    }
    opened = []
    fake = types.ModuleType('copernicusmarine')

    def open_dataset(*, dataset_id):
        opened.append(dataset_id)
        return datasets[dataset_id]
    fake.open_dataset = open_dataset
    monkeypatch.setitem(sys.modules, 'copernicusmarine', fake)
    return opened


def test_a_past_date_reads_the_reanalysis(monkeypatch):
    opened = _install_dated_products(monkeypatch)
    ts = copernicus.fetch_ts_profile_operational(
        (30.0, -40.0), date='2020-06-15')
    assert opened == [copernicus.PHYSICS_PRODUCTS[0][0]]
    assert ts.salinity.tolist() == _S
    assert ts.provenance.source.id == 'copernicus'
    assert ts.provenance.requested_point == (30.0, -40.0)


def test_a_date_past_the_reanalysis_reads_the_analysis_forecast(monkeypatch):
    """Salinity comes from the forecast's own salinity dataset, and the record
    names both datasets read."""
    opened = _install_dated_products(monkeypatch)
    ssp = copernicus.fetch_ssp_operational((30.0, -40.0), date='2026-09-26')
    fc_t, fc_s = copernicus.PHYSICS_PRODUCTS[1]
    assert opened[1:] == [fc_t, fc_s]
    assert ssp.data_sources[0].product == f"{fc_t} + {fc_s}"
    ts = copernicus.fetch_ts_profile_operational(
        (30.0, -40.0), date='2026-09-26')
    assert ts.salinity.tolist() == [35.0] * len(_DEPTH)
    assert ts.provenance.product == f"{fc_t} + {fc_s}"


def test_the_last_reanalysis_day_reads_the_reanalysis(monkeypatch):
    opened = _install_dated_products(monkeypatch)
    copernicus.fetch_ssp_operational((30.0, -40.0), date=_REANALYSIS_END)
    assert opened == [copernicus.PHYSICS_PRODUCTS[0][0]]
    copernicus.fetch_ssp_operational((30.0, -40.0), date='2026-06-24')
    assert opened[-1] == copernicus.PHYSICS_PRODUCTS[1][1]


def test_an_explicit_dataset_id_is_the_only_one_opened(monkeypatch):
    opened = _install_dated_products(monkeypatch)
    my_t = copernicus.PHYSICS_PRODUCTS[0][0]
    copernicus.fetch_ssp_operational((30.0, -40.0), date='2026-06-01',
                                     dataset_id=my_t)
    assert opened == [my_t]


def test_the_ph_dataset_is_picked_by_date(monkeypatch):
    opened = _install_dated_products(monkeypatch)
    bgc_my, bgc_fc = copernicus.BGC_PRODUCTS
    assert copernicus.fetch_ph_operational(
        (30.0, -40.0), date='2020-06-15', reference_depth=0.0) == 8.10
    assert opened == [bgc_my]
    assert copernicus.fetch_ph_operational(
        (30.0, -40.0), date='2026-09-26', reference_depth=0.0) == 8.05
    assert opened[-1] == bgc_fc


def test_fetch_environment_reaches_the_forecast_for_a_recent_date(
        monkeypatch, tmp_path):
    """The one-call entry point has no dataset_id to pass, so a current date
    works only when the product is picked by date."""
    import uacpy.data as data
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    _install_dated_products(monkeypatch)
    env = data.fetch_environment((30.0, -40.0), bathymetry=3000.0,
                                 ssp_sources='copernicus', date='2026-09-26',
                                 bottom='sand')
    fc_t, fc_s = copernicus.PHYSICS_PRODUCTS[1]
    assert env.ssp.data_sources[0].product == f"{fc_t} + {fc_s}"


class _SnappedDAStub(_DAStub):
    """DataArray stub exposing the time/lat/lon coords ``sel`` snapped to."""
    def __init__(self, values, time, lat, lon):
        super().__init__(values)
        self.coords = {'time': _Coord(time), 'latitude': _Scalar(lat),
                       'longitude': _Scalar(lon)}

    def __getitem__(self, key):
        return self.coords[key]


class _Scalar:
    def __init__(self, value):
        self.values = np.asarray(value, dtype=float)


class _SnappedDSStub:
    """Dataset stub whose nearest-neighbour selection lands on a fixed cell."""
    def __init__(self, time, lat, lon):
        self._vars = {
            'depth': _DAStub(_DEPTH),
            'thetao': _SnappedDAStub(_T, time, lat, lon),
            'so': _SnappedDAStub(_S, time, lat, lon),
        }

    def __getitem__(self, key):
        return self._vars[key]


def test_provenance_records_the_snapped_date_and_cell(monkeypatch):
    """The daily mean snaps to a day and a 1/12° cell; both must be recorded."""
    _install_fake_toolbox(
        monkeypatch, _SnappedDSStub('2020-06-14', 30.0417, -40.0417))
    ssp = copernicus.fetch_ssp_operational((30.0, -40.0), date='2020-06-15')
    prov = ssp.data_sources[0]
    assert prov.data_date == '2020-06-14'
    assert prov.data_point == pytest.approx((30.0417, -40.0417))
    assert prov.requested_date == '2020-06-15'
    assert prov.offset_km == pytest.approx(6.1, abs=0.5)


def test_transect_provenance_records_the_snapped_date_and_cell(monkeypatch):
    _install_fake_toolbox(
        monkeypatch, _SnappedDSStub('2020-06-14', 30.0417, -40.0417))
    ssp = copernicus.fetch_ssp_transect_operational(
        (30.0, -40.0), (31.0, -40.0), date='2020-06-15', n_points=3)
    prov = ssp.data_sources[0]
    assert prov.data_date == '2020-06-14'
    assert prov.data_point == pytest.approx((30.0417, -40.0417))


def test_citations_reports_the_fetched_line(monkeypatch):
    """A Copernicus-sourced profile must render a ``Fetched:`` line."""
    from uacpy.data.sources import citations
    _install_fake_toolbox(
        monkeypatch, _SnappedDSStub('2020-06-14', 30.0417, -40.0417))
    ssp = copernicus.fetch_ssp_operational((30.0, -40.0), date='2020-06-15')
    text = citations(ssp)
    assert 'Fetched:' in text
    assert '2020-06-14' in text


def test_no_snapping_coords_leaves_provenance_unstamped(monkeypatch):
    """A dataset exposing no time/lat/lon coords records nothing it did not get."""
    _install_fake_toolbox(monkeypatch, _DSStub(_DEPTH, _T, _S))
    ssp = copernicus.fetch_ssp_operational((30.0, -40.0), date='2020-06-15')
    prov = ssp.data_sources[0]
    assert prov.data_date is None
    assert prov.data_point is None


# ── timeout ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize('name', [
    'fetch_ssp_operational', 'fetch_ssp_transect_operational',
    'fetch_ts_profile_operational', 'fetch_waves_operational',
    'fetch_ph_operational',
])
def test_fetchers_expose_no_timeout(name):
    """The copernicusmarine session owns the timeout; no fetcher may pretend to.

    Accepting a ``timeout=`` it cannot honour is worse than not offering one —
    the toolbox reads ``COPERNICUSMARINE_HTTPS_TIMEOUT`` at import and exposes
    no per-call knob.
    """
    import inspect
    sig = inspect.signature(getattr(copernicus, name))
    assert 'timeout' not in sig.parameters


def test_assemble_range_dependent_aggregates_provenance():
    from uacpy.data.sound_speed import assemble_range_dependent
    from uacpy.data.sources import SOURCES, DataProvenance
    provs = (DataProvenance(source=SOURCES['copernicus'],
                            requested_point=(30.0, -40.0)),
             DataProvenance(source=SOURCES['copernicus'],
                            requested_point=(31.0, -40.0)),
             DataProvenance(source=SOURCES['woa23']))
    cols = [SoundSpeedProfile(depths=[0.0, 100.0], sound_speed=[1500.0, 1490.0],
                              data_sources=(p,)) for p in provs]
    out = assemble_range_dependent(cols, [0.0, 1000.0, 2000.0])
    assert [(p.source.id, p.range_m) for p in out.data_sources] == [
        ('copernicus', 0.0), ('copernicus', 1000.0), ('woa23', 2000.0)]


def _install_operational_dataset(monkeypatch, ds):
    """Route the copernicusmarine open through a synthetic xarray dataset."""
    monkeypatch.setattr(copernicus, '_import_copernicusmarine', lambda: None)
    monkeypatch.setattr(copernicus, '_open_dataset',
                        lambda marine, dataset_id, **kw: ds)


def _metocean_coords():
    """0.25-degree axes over 30-31N, 41-40W with a single 2020-06-15 step."""
    return (np.array(['2020-06-15'], dtype='datetime64[ns]'),
            np.arange(30.0, 31.001, 0.25),
            np.arange(-41.0, -39.999, 0.25))


def _wave_xr_dataset():
    xr = pytest.importorskip('xarray')
    time, lat, lon = _metocean_coords()
    dims = ('time', 'latitude', 'longitude')
    coords = {'time': time, 'latitude': lat, 'longitude': lon}
    shape = (time.size, lat.size, lon.size)
    return xr.Dataset({
        copernicus.WAVE_HS_VAR: xr.DataArray(np.full(shape, 2.5),
                                             dims=dims, coords=coords),
        copernicus.WAVE_TP_VAR: xr.DataArray(np.full(shape, 9.0),
                                             dims=dims, coords=coords),
    })


def _bgc_xr_dataset():
    xr = pytest.importorskip('xarray')
    time, lat, lon = _metocean_coords()
    depth = np.array([0.0, 100.0])
    dims = ('time', 'depth', 'latitude', 'longitude')
    coords = {'time': time, 'depth': depth, 'latitude': lat, 'longitude': lon}
    shape = (time.size, depth.size, lat.size, lon.size)
    return xr.Dataset({
        copernicus.BGC_PH_VAR: xr.DataArray(np.full(shape, 8.05),
                                            dims=dims, coords=coords),
    })


def test_fetch_waves_operational_accepts_a_point_inside_the_domain(
        monkeypatch):
    _install_operational_dataset(monkeypatch, _wave_xr_dataset())
    out = copernicus.fetch_waves_operational((30.4, -40.4), date='2020-06-15')
    assert out.hs == pytest.approx(2.5)
    assert out.tp == pytest.approx(9.0)
    prov = out.provenance
    assert isinstance(prov, DataProvenance) and prov.source.id == 'waverys'
    assert prov.requested_point == (30.4, -40.4)
    assert prov.requested_date == '2020-06-15'
    assert prov.data_date is not None and prov.data_point is not None


def test_fetch_waves_operational_rejects_a_point_outside_the_domain(
        monkeypatch):
    _install_operational_dataset(monkeypatch, _wave_xr_dataset())
    with pytest.raises(DataFetchError, match='spatial domain'):
        copernicus.fetch_waves_operational((45.0, -40.4), date='2020-06-15')


def test_fetch_ph_operational_accepts_a_point_inside_the_domain(monkeypatch):
    _install_operational_dataset(monkeypatch, _bgc_xr_dataset())
    ph = copernicus.fetch_ph_operational((30.4, -40.4), date='2020-06-15')
    assert ph == pytest.approx(8.05)


def test_fetch_ph_operational_rejects_a_point_outside_the_domain(monkeypatch):
    _install_operational_dataset(monkeypatch, _bgc_xr_dataset())
    with pytest.raises(DataFetchError, match='spatial domain'):
        copernicus.fetch_ph_operational((30.4, -10.0), date='2020-06-15')


def _regional_xr_dataset():
    """A 0.25-degree regional T/S dataset over 30-31N, 41-40W."""
    xr = pytest.importorskip('xarray')
    lat = np.arange(30.0, 31.001, 0.25)
    lon = np.arange(-41.0, -39.999, 0.25)
    depth = np.array([0.0, 100.0])
    dims = ('depth', 'latitude', 'longitude')
    coords = {'depth': depth, 'latitude': lat, 'longitude': lon}
    shape = (depth.size, lat.size, lon.size)
    return xr.Dataset({
        'thetao': xr.DataArray(np.full(shape, 15.0), dims=dims, coords=coords),
        'so': xr.DataArray(np.full(shape, 35.0), dims=dims, coords=coords),
    })


def test_extract_ts_accepts_a_point_inside_the_domain():
    ds = _regional_xr_dataset()
    z, t, s, actual = copernicus._extract_ts(ds, 30.4, -40.4, None)
    assert z.tolist() == [0.0, 100.0]
    assert t.tolist() == [15.0, 15.0]
    assert actual['point'] == pytest.approx((30.5, -40.5))


def test_extract_ts_rejects_a_point_outside_the_domain():
    ds = _regional_xr_dataset()
    with pytest.raises(DataFetchError, match='outside the dataset'):
        copernicus._extract_ts(ds, 45.0, -40.4, None)


def test_extract_ts_rejects_a_longitude_outside_the_domain():
    ds = _regional_xr_dataset()
    with pytest.raises(DataFetchError, match='longitude'):
        copernicus._extract_ts(ds, 30.4, -10.0, None)


_DEEP_DEPTH = [0.0, 1000.0, 5000.0]
_DEEP_THETA = [18.0, 4.0, 1.5]
_DEEP_S = [36.0, 34.9, 34.7]


def test_deep_ssp_converts_potential_temperature_to_in_situ(monkeypatch):
    """``thetao`` is potential temperature; the sound-speed equations want
    in-situ, which is warmer under pressure. Feeding theta straight in cost
    about 2 m/s at 5000 m — this pins the conversion at the deep level and
    pins that it leaves the surface (0 dbar) alone."""
    from uacpy.core.acoustics.seawater import depth_to_pressure_dbar
    from uacpy.core.acoustics.seawater import SOUND_SPEED_FORMULAS

    _install_fake_toolbox(monkeypatch, _DSStub(_DEEP_DEPTH, _DEEP_THETA, _DEEP_S))
    ssp = copernicus.fetch_ssp_operational((30.0, -40.0), date='2020-06-15')

    pressure = depth_to_pressure_dbar(np.array(_DEEP_DEPTH), 30.0)
    raw = np.array([SOUND_SPEED_FORMULAS['teos10'](t, s, p)
                    for t, s, p in zip(_DEEP_THETA, _DEEP_S, pressure)])
    excess = ssp.sound_speed.ravel() - raw
    assert excess[0] == pytest.approx(0.0, abs=1e-9)     # surface: no shift
    assert 0.2 < excess[1] < 0.4                          # 1000 m
    assert 1.8 < excess[2] < 2.1                          # 5000 m


def test_deep_transect_ssp_converts_potential_temperature(monkeypatch):
    """The transect fetcher carries the same conversion as the single point."""
    _install_fake_toolbox(monkeypatch, _DSStub(_DEEP_DEPTH, _DEEP_THETA, _DEEP_S))
    point = copernicus.fetch_ssp_operational((30.0, -40.0), date='2020-06-15')
    transect = copernicus.fetch_ssp_transect_operational(
        (30.0, -40.0), (30.0, -40.5), date='2020-06-15', n_points=2)
    assert transect.sound_speed[-1, 0] == pytest.approx(point.sound_speed[-1], abs=0.05)


def test_the_ts_profile_returns_in_situ_temperature(monkeypatch):
    """Like WOA23's ``t_an``: the dataset's potential temperature is
    converted once, at each level's pressure at the point's latitude."""
    from uacpy.core.acoustics.seawater import depth_to_pressure_dbar, insitu_from_potential
    _install_fake_toolbox(monkeypatch, _DSStub(_DEEP_DEPTH, _DEEP_THETA, _DEEP_S))
    t = copernicus.fetch_ts_profile_operational(
        (30.0, -40.0), date='2020-06-15').temperature
    z, s = np.array(_DEEP_DEPTH), np.array(_DEEP_S)
    expected = insitu_from_potential(
        salinity=s, theta=np.array(_DEEP_THETA),
        pressure_dbar=depth_to_pressure_dbar(z, 30.0))
    np.testing.assert_array_equal(t, expected)
    assert t[-1] > _DEEP_THETA[-1]


@pytest.mark.parametrize('formula', ['unesco', 'delgrosso', 'teos10'])
def test_copernicus_profiles_record_the_formula_that_built_them(monkeypatch,
                                                                formula):
    """Both fetchers stamp ``formula``, so a later seafloor extension continues
    the column under its own equation rather than defaulting to UNESCO."""
    _install_fake_toolbox(monkeypatch, _DSStub(_DEPTH, _T, _S))
    point = copernicus.fetch_ssp_operational((30.0, -40.0), date='2020-06-15',
                                             formula=formula)
    assert point.formula == formula
    transect = copernicus.fetch_ssp_transect_operational(
        (30.0, -40.0), (31.0, -40.0), date='2020-06-15', n_points=3,
        formula=formula)
    assert transect.formula == formula


@pytest.mark.parametrize('kind,fetcher,var,needle,remediation', [
    ('waves', 'fetch_waves_operational', copernicus.WAVE_HS_VAR,
     'Copernicus waves: nearest available time is', 'WaveWatch III'),
    ('ph', 'fetch_ph_operational', copernicus.BGC_PH_VAR,
     'Copernicus pH: nearest available time is', 'GLODAP'),
])
def test_the_waves_and_ph_fetchers_keep_their_own_date_gap_wording(
        monkeypatch, kind, fetcher, var, needle, remediation):
    """Both call ``_snapped_date``'s guard; each names itself in the message
    and keeps the alternative source its remediation recommends."""
    class _VarDSStub(_DSStub):
        def __init__(self):
            super().__init__([0.0, 10.0], [20.0, 19.0], [35.0, 35.0],
                             time='1990-01-01')
            self._vars[var] = _TimeDAStub([1.5, 1.5], '1990-01-01')

    _install_fake_toolbox(monkeypatch, _VarDSStub())
    with pytest.raises(
            DataFetchError,
            match="the date is outside the dataset's range") as excinfo:
        getattr(copernicus, fetcher)((30.0, -40.0), date='2020-06-15')
    assert needle in str(excinfo.value)
    assert remediation in (excinfo.value.remediation or '')


def test_the_ssp_date_gap_keeps_its_own_wording(monkeypatch):
    _install_fake_toolbox(monkeypatch,
                          _DSStub(_DEPTH, _T, _S, time='1990-01-01'))
    with pytest.raises(
            DataFetchError,
            match="the date is outside the dataset's range") as excinfo:
        copernicus.fetch_ssp_operational((30.0, -40.0), date='2020-06-15')
    assert 'nearest available time is' in str(excinfo.value)
    assert "ssp_sources='woa23'" in (excinfo.value.remediation or '')


# ── the date rule on the Copernicus fetchers ────────────────────────────────

def _date_warnings(fetch):
    import warnings
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        fetch()
    return [str(w.message) for w in rec if 'the data are dated' in str(w.message)]


@pytest.mark.parametrize('requested, warns', [('2021-01-18', False),
                                              ('2021-01-19', True)])
def test_a_daily_field_warns_past_3_days(monkeypatch, requested, warns):
    _install_fake_toolbox(monkeypatch, _DSStub(_DEPTH, _T, _S, time='2021-01-15'))
    msgs = _date_warnings(lambda: copernicus.fetch_ssp_operational(
        (30.0, -40.0), date=requested, max_days=30))
    assert len(msgs) == int(warns)


@pytest.mark.parametrize('requested, warns', [('2021-01-31', False),
                                              ('2021-02-01', True)])
def test_a_monthly_ph_field_warns_from_another_month(monkeypatch, requested,
                                                     warns):
    _install_fake_toolbox(monkeypatch,
                          _BGCStub([8.05, 8.00, 7.90], time='2021-01-15'))
    msgs = _date_warnings(lambda: copernicus.fetch_ph_operational(
        (30.0, -40.0), date=requested, max_days=30))
    assert len(msgs) == int(warns)
