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
    ssp = copernicus.fetch_ssp_operational((30.0, -40.0), '2020-06-15')
    assert isinstance(ssp, SoundSpeedProfile)
    assert ssp.depths.tolist() == _DEPTH
    assert np.all((1440 < ssp.data) & (ssp.data < 1560))


def test_fetch_ssp_transect_operational(monkeypatch):
    _install_fake_toolbox(monkeypatch, _DSStub(_DEPTH, _T, _S))
    ssp = copernicus.fetch_ssp_transect_operational(
        (30.0, -40.0), (31.0, -40.0), '2020-06-15', n_points=4)
    assert ssp.is_range_dependent
    assert ssp.data.shape == (len(_DEPTH), 4)
    assert ssp.ranges[0] == 0.0


def test_missing_toolbox_raises_helpful_error(monkeypatch):
    # Force the import to fail even if the package were present.
    monkeypatch.setitem(sys.modules, 'copernicusmarine', None)
    with pytest.raises(DataFetchError, match='copernicusmarine'):
        copernicus.fetch_ts_profile_operational((0.0, 0.0), '2020-01-01')


def test_open_dataset_failure_wrapped(monkeypatch):
    fake = types.ModuleType('copernicusmarine')

    def boom(**kwargs):
        raise RuntimeError("auth failed")
    fake.open_dataset = boom
    monkeypatch.setitem(sys.modules, 'copernicusmarine', fake)
    with pytest.raises(DataFetchError, match='open_dataset failed'):
        copernicus.fetch_ssp_operational((0.0, 0.0), '2020-01-01')


def test_bad_formula_raises():
    with pytest.raises(ConfigurationError, match='formula'):
        copernicus.fetch_ssp_operational((0.0, 0.0), '2020-01-01', formula='x')


def test_bad_date_raises(monkeypatch):
    _install_fake_toolbox(monkeypatch, _DSStub(_DEPTH, _T, _S))
    with pytest.raises(ConfigurationError, match='parse date'):
        copernicus.fetch_ssp_operational((0.0, 0.0), 'not-a-date')


def test_out_of_range_date_warns(monkeypatch):
    # Dataset's only time step is 2021; asking for 2030 snaps to the edge.
    _install_fake_toolbox(monkeypatch, _DSStub(_DEPTH, _T, _S, time='2021-01-15'))
    with pytest.warns(UserWarning, match='outside the'):
        copernicus.fetch_ssp_operational((30.0, -40.0), '2030-06-15')


def test_in_range_date_no_warn(monkeypatch, recwarn):
    _install_fake_toolbox(monkeypatch, _DSStub(_DEPTH, _T, _S, time='2020-06-15'))
    copernicus.fetch_ssp_operational((30.0, -40.0), '2020-06-15')
    assert not [w for w in recwarn if 'outside' in str(w.message)]
