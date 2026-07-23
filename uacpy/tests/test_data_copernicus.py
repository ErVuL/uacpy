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
    z, t, s = copernicus._extract_ts(ds, 30.0, -40.0, '2020-06-01')
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
    assert env.absorption.pH == pytest.approx(8.02)
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
