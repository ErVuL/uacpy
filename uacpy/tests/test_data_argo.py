"""Tests for the Argo real-profile SSP source (uacpy.data.argo)."""

import numpy as np
import pytest

from uacpy.core.environment import SoundSpeedProfile
from uacpy.core.exceptions import (ConfigurationError, DataFetchError,
                                   FileFormatError)
from uacpy.data import argo

_HEADER = ("platform_number,cycle_number,direction,time,latitude,longitude,"
           "pres,temp,psal,temp_qc,psal_qc\n"
           ",,,UTC,degrees_north,degrees_east,decibar,degree_Celsius,PSU,,\n")


def _csv(rows):
    return _HEADER + "".join(rows)


# Float A is ~15 km from (30,-40); float B is far. A has one bad-QC level.
_ROWS = [
    "4900001,1,A,2024-06-04T00:00:00Z,30.1,-40.1,5,20,36,1,1\n",
    "4900001,1,A,2024-06-04T00:00:00Z,30.1,-40.1,100,15,36.2,1,1\n",
    "4900001,1,A,2024-06-04T00:00:00Z,30.1,-40.1,1000,5,35,1,1\n",
    "4900001,1,A,2024-06-04T00:00:00Z,30.1,-40.1,1500,4,35,4,1\n",  # bad temp_qc
    "4900002,5,A,2024-06-04T00:00:00Z,33.0,-43.0,5,19,36,1,1\n",    # far float
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


# One Argo cycle carries up to two stations. ERDDAP's own ``direction``
# conventions are "A: ascending profiles, D: descending profiles" and its
# ``cdm_profile_variables`` lists ``direction``, so the cast identity is
# (platform, cycle, direction). Modelled on the measured case: float 3902110
# cycle 463 in the Baltic, whose D and A casts are 4 days and 22 km apart.
_TWO_CAST_ROWS = [
    "3902110,463,D,2023-03-08T09:34:00Z,58.9670,20.1725,10,2.66,7.04,1,1\n",
    "3902110,463,D,2023-03-08T09:34:00Z,58.9670,20.1725,20,2.66,7.04,1,1\n",
    "3902110,463,D,2023-03-08T09:34:00Z,58.9670,20.1725,30,2.70,7.10,1,1\n",
    "3902110,463,A,2023-03-12T09:08:30Z,58.7737,19.9920,10,2.49,7.08,1,1\n",
    "3902110,463,A,2023-03-12T09:08:30Z,58.7737,19.9920,20,2.50,7.08,1,1\n",
    "3902110,463,A,2023-03-12T09:08:30Z,58.7737,19.9920,30,2.52,7.12,1,1\n",
]


class TestCastsOfOneCycleAreDistinctStations:

    def test_the_two_casts_are_not_merged(self, monkeypatch):
        monkeypatch.setattr(argo, 'http_get',
                            lambda url, **kw: _csv(_TWO_CAST_ROWS))
        prof = argo.fetch_argo_profile((58.967, 20.1725), date='2023-03-08',
                                       max_distance_km=25.0, max_days=6)
        assert prof['direction'] == 'D'                # at the requested point
        assert prof['pres'].tolist() == [10.0, 20.0, 30.0], (
            f"got {prof['pres'].tolist()} — the descent and ascent casts have "
            f"been interleaved into one column")
        assert np.unique(prof['pres']).size == prof['pres'].size
        assert np.all(np.diff(prof['pres']) > 0)

    def test_each_cast_stays_selectable(self, monkeypatch):
        """Both casts must remain separate candidates — the fix must not drop
        one, only stop them merging."""
        monkeypatch.setattr(argo, 'http_get',
                            lambda url, **kw: _csv(_TWO_CAST_ROWS))
        prof = argo.fetch_argo_profile((58.7737, 19.9920), date='2023-03-12',
                                       max_distance_km=25.0, max_days=6)
        assert prof['direction'] == 'A'
        assert prof['pres'].size == 3

    def test_a_merged_cycle_cannot_build_an_ssp(self, monkeypatch):
        """Merging duplicated every pressure, so the carrier rejected the column
        with a ConfigurationError that blamed the caller's configuration."""
        monkeypatch.setattr(argo, 'http_get',
                            lambda url, **kw: _csv(_TWO_CAST_ROWS))
        ssp = argo.fetch_ssp_argo((58.967, 20.1725), date='2023-03-08',
                                  max_distance_km=25.0, max_days=6)
        assert isinstance(ssp, SoundSpeedProfile)
        assert ssp.n_depths == 3
        assert np.all(np.diff(np.asarray(ssp.depths)) > 0)

    def test_an_unexpected_column_layout_is_rejected(self, monkeypatch):
        """The rows are unpacked positionally, so a changed table layout has to
        raise rather than silently assign temperature to salinity."""
        swapped = _HEADER.replace('temp,psal', 'psal,temp')
        monkeypatch.setattr(argo, 'http_get',
                            lambda url, **kw: swapped + "".join(_ROWS))
        with pytest.raises(FileFormatError, match='columns'):
            argo.fetch_argo_profile((30.0, -40.0), date='2024-06-04')


def test_pressure_to_depth_inverts():
    # ``pressure_dbar_to_depth`` is a 5-step Newton inversion of
    # ``depth_to_pressure_dbar``, so the round trip is exact up to Newton
    # convergence — the residual here is ~5e-13 m. ``atol=0.1`` is an
    # acceptability bound on depth (a decimetre is far below any Argo level
    # spacing), not a measure of the method's accuracy.
    from uacpy.data._geo import depth_to_pressure_dbar, pressure_dbar_to_depth
    z = np.array([0.0, 100.0, 1000.0, 4000.0])
    p = depth_to_pressure_dbar(z, 30.0)
    z_back = pressure_dbar_to_depth(p, 30.0)
    assert np.allclose(z, z_back, atol=0.1)


def _argo_csv(rows):
    from uacpy.data import argo
    header = ','.join(argo._COLUMNS)
    units = ','.join([''] * len(argo._COLUMNS))
    return '\n'.join([header, units] + rows)


def test_a_dated_argo_profile_beats_an_undated_one_at_equal_distance(
        monkeypatch):
    from uacpy.data import argo

    def _row(platform, time_str, pres):
        return f"{platform},1,A,{time_str},45.0,-30.0,{pres},10.0,35.0,1,1"

    body = _argo_csv([_row('1', '', 10.0), _row('1', '', 20.0),
                      _row('2', '2026-08-14T00:00:00Z', 10.0),
                      _row('2', '2026-08-14T00:00:00Z', 20.0)])
    monkeypatch.setattr(
        argo, 'http_get',
        lambda url, timeout=None, verbose=False, source=None: body)
    profile = argo.fetch_argo_profile((45.0, -30.0), date='2026-08-15')
    assert profile['platform'] == '2'


def test_an_unparseable_argo_time_costs_more_than_any_real_one():
    from uacpy.data import argo
    when = np.datetime64('2026-08-15', 'D')
    assert argo._abs_days(None, when) == float('inf')
    assert argo._abs_days('garbage', when) == float('inf')


def test_the_argo_query_window_includes_the_whole_last_tolerated_day():
    from uacpy.data import argo
    when = np.datetime64('2026-08-15', 'D')
    url = argo._query_url((45.0, -30.0), when, 300.0, 10, 'http://x')
    assert 'time%3E=2026-08-05T00:00:00Z' in url
    assert 'time%3C2026-08-26T00:00:00Z' in url
