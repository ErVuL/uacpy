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
    assert np.all((1440 < ssp.data) & (ssp.data < 1560))


def test_fetch_ssp_transect_operational(monkeypatch):
    _install_fake_toolbox(monkeypatch, _DSStub(_DEPTH, _T, _S))
    ssp = copernicus.fetch_ssp_transect_operational(
        (30.0, -40.0), (31.0, -40.0), date='2020-06-15', n_points=4)
    assert ssp.is_range_dependent
    assert ssp.data.shape == (len(_DEPTH), 4)
    assert ssp.ranges[0] == 0.0


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


def test_fetch_ph_operational_surface_value(monkeypatch):
    _install_fake_toolbox(monkeypatch, _BGCStub([8.05, 8.00, 7.90]))
    ph = copernicus.fetch_ph_operational((30.0, -40.0), date='2020-06-15')
    assert ph == pytest.approx(8.05)


def test_fetch_ph_operational_skips_nan_surface(monkeypatch):
    # A masked surface level falls through to the shallowest finite one.
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


def test_environment_bgc_failure_falls_back(monkeypatch, tmp_path):
    import uacpy.data as data
    from uacpy.data.absorption import DEFAULT_OCEAN_PH
    monkeypatch.setenv('UACPY_DATA_CACHE', str(tmp_path / 'empty'))
    _install_routing_toolbox(monkeypatch, bgc=RuntimeError("auth failed"))
    env = data.fetch_environment((30.0, -40.0), bathymetry=1000.0,
                                 ssp_sources='copernicus', date='2020-06-15',
                                 with_absorption=True)
    # No BGC, no GLODAP cache → the model-default constant, silently.
    assert env.absorption.pH == pytest.approx(DEFAULT_OCEAN_PH)
    assert 'copernicus_bgc' not in [s.source.id for s in env.data_sources]


def test_fetch_ph_woa_source_does_not_hit_bgc(monkeypatch, tmp_path):
    # A non-Copernicus SSP source must not silently open a Copernicus dataset
    # for pH — the BGC preference rides the existing Copernicus session only.
    from uacpy.data import environment
    from uacpy.data.absorption import DEFAULT_OCEAN_PH
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
    assert pH == pytest.approx(DEFAULT_OCEAN_PH)
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
    assert [p.source.id for p in ssp.data_sources] == ['copernicus']


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
    cols = [SoundSpeedProfile(depths=[0.0, 100.0], data=[1500.0, 1490.0],
                              data_sources=(p,)) for p in provs]
    out = assemble_range_dependent(cols, [0.0, 1000.0, 2000.0])
    assert [p.source.id for p in out.data_sources] == ['copernicus', 'woa23']


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
    assert out['hs'] == pytest.approx(2.5)
    assert out['tp'] == pytest.approx(9.0)


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
