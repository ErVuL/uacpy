"""Tests for the Argo real-profile SSP source (uacpy.data.argo)."""

import numpy as np
import pytest

from uacpy.core.environment import SoundSpeedProfile
from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data import argo

_HEADER = ("platform_number,cycle_number,time,latitude,longitude,"
           "pres,temp,psal,temp_qc,psal_qc\n"
           ",,UTC,degrees_north,degrees_east,decibar,degree_Celsius,PSU,,\n")


def _csv(rows):
    return _HEADER + "".join(rows)


# Float A is ~15 km from (30,-40); float B is far. A has one bad-QC level.
_ROWS = [
    "4900001,1,2024-06-04T00:00:00Z,30.1,-40.1,5,20,36,1,1\n",
    "4900001,1,2024-06-04T00:00:00Z,30.1,-40.1,100,15,36.2,1,1\n",
    "4900001,1,2024-06-04T00:00:00Z,30.1,-40.1,1000,5,35,1,1\n",
    "4900001,1,2024-06-04T00:00:00Z,30.1,-40.1,1500,4,35,4,1\n",   # bad temp_qc
    "4900002,5,2024-06-04T00:00:00Z,33.0,-43.0,5,19,36,1,1\n",     # far float
]


def test_fetch_profile_picks_nearest_and_filters_qc(monkeypatch):
    monkeypatch.setattr(argo, 'http_get', lambda url, **kw: _csv(_ROWS))
    prof = argo.fetch_argo_profile((30.0, -40.0), date='2024-06-04')
    assert prof['platform'] == '4900001'           # nearest, not the far float
    assert prof['distance_km'] < 20
    assert prof['pres'].tolist() == [5.0, 100.0, 1000.0]   # bad-QC level dropped
    assert np.all(np.diff(prof['pres']) > 0)       # sorted by pressure


def test_fetch_ssp_argo_builds_profile(monkeypatch):
    monkeypatch.setattr(argo, 'http_get', lambda url, **kw: _csv(_ROWS))
    ssp = argo.fetch_ssp_argo((30.0, -40.0), date='2024-06-04')
    assert isinstance(ssp, SoundSpeedProfile)
    assert ssp.n_depths == 3
    assert np.all((1450 < ssp.data) & (ssp.data < 1560))
    assert ssp.depths[0] < ssp.depths[-1]          # increasing depth


def test_no_profile_raises(monkeypatch):
    monkeypatch.setattr(argo, 'http_get', lambda url, **kw: _csv([]))
    with pytest.raises(DataFetchError, match='No Argo profile'):
        argo.fetch_argo_profile((0.0, -150.0), date='2024-06-04')


def test_too_far_raises(monkeypatch):
    monkeypatch.setattr(argo, 'http_get', lambda url, **kw: _csv(_ROWS))
    with pytest.raises(DataFetchError, match='km away'):
        argo.fetch_argo_profile((30.0, -40.0), date='2024-06-04', max_distance_km=5)


def test_bad_formula_raises(monkeypatch):
    monkeypatch.setattr(argo, 'http_get', lambda url, **kw: _csv(_ROWS))
    with pytest.raises(ConfigurationError, match='formula'):
        argo.fetch_ssp_argo((30.0, -40.0), date='2024-06-04', formula='nope')


def test_pressure_to_depth_inverts():
    # ``_pressure_dbar_to_depth`` is a 5-step Newton inversion of
    # ``depth_to_pressure_dbar``, so the round trip is exact up to Newton
    # convergence — the residual here is ~5e-13 m. ``atol=0.1`` is an
    # acceptability bound on depth (a decimetre is far below any Argo level
    # spacing), not a measure of the method's accuracy.
    from uacpy.data._geo import depth_to_pressure_dbar
    z = np.array([0.0, 100.0, 1000.0, 4000.0])
    p = depth_to_pressure_dbar(z, 30.0)
    z_back = argo._pressure_dbar_to_depth(p, 30.0)
    assert np.allclose(z, z_back, atol=0.1)
