"""Tests for the Argo real-profile SSP source (uacpy.data.argo)."""

import numpy as np
import pytest

from uacpy.core.environment import Environment, SoundSpeedProfile
from uacpy.core.exceptions import (ConfigurationError, DataFetchError,
                                   FileFormatError)
from uacpy.data import argo

_HEADER = ("platform_number,cycle_number,direction,time,latitude,longitude,"
           "pres,temp,psal,temp_qc,psal_qc,pres_qc,position_qc,data_mode,"
           "pres_adjusted,temp_adjusted,psal_adjusted,pres_adjusted_qc,"
           "temp_adjusted_qc,psal_adjusted_qc\n"
           ",,,UTC,degrees_north,degrees_east,decibar,degree_Celsius,PSU,,,,,,"
           "decibar,degree_Celsius,PSU,,,\n")
# A real-time cast's adjusted fields as the ArgoFloats table serves them.
_REALTIME_TAIL = ",R,NaN,NaN,NaN,,,"


def _realtime(row):
    """A row given in the 13 raw columns, completed as a real-time cast."""
    return row.rstrip('\n') + _REALTIME_TAIL + '\n'


def _csv(rows):
    """The table body; a row in the 13 raw columns is a real-time cast."""
    return _HEADER + "".join(
        _realtime(r) if r.count(',') == 12 else r for r in rows)


# Float A is ~15 km from (30,-40); float B is far. A has one bad-QC level.
_ROWS = [
    "4900001,1,A,2024-06-04T00:00:00Z,30.1,-40.1,5,20,36,1,1,1,1\n",
    "4900001,1,A,2024-06-04T00:00:00Z,30.1,-40.1,100,15,36.2,1,1,1,1\n",
    "4900001,1,A,2024-06-04T00:00:00Z,30.1,-40.1,1000,5,35,1,1,1,1\n",
    "4900001,1,A,2024-06-04T00:00:00Z,30.1,-40.1,1500,4,35,4,1,1,1\n",  # bad temp_qc
    "4900002,5,A,2024-06-04T00:00:00Z,33.0,-43.0,5,19,36,1,1,1,1\n",    # far float
]


def test_fetch_profile_picks_nearest_and_filters_qc(monkeypatch):
    monkeypatch.setattr(argo, 'http_get', lambda url, **kw: _csv(_ROWS))
    prof = argo.fetch_argo_profile((30.0, -40.0), date='2024-06-04')
    assert prof.platform == '4900001'           # nearest, not the far float
    assert prof.distance_km < 20
    assert prof.pressure_dbar.tolist() == [5.0, 100.0, 1000.0]   # bad-QC level dropped
    assert np.all(np.diff(prof.pressure_dbar) > 0)       # sorted by pressure


class TestTheCastIsARecord:
    """``fetch_argo_profile`` returns an :class:`~uacpy.data.ArgoProfile`:
    the measured columns read-only, one table row per level, and a dict
    form that rebuilds it, provenance included."""

    @staticmethod
    def _cast(monkeypatch):
        monkeypatch.setattr(argo, 'http_get', lambda url, **kw: _csv(_ROWS))
        return argo.fetch_argo_profile((30.0, -40.0), date='2024-06-04')

    def test_the_columns_refuse_edits(self, monkeypatch):
        prof = self._cast(monkeypatch)
        assert type(prof) is argo.ArgoProfile
        with pytest.raises(ValueError, match='read-only'):
            prof.temperature[0] = 0.0

    def test_the_table_has_one_row_per_level(self, monkeypatch):
        pytest.importorskip('pandas')
        frame = self._cast(monkeypatch).to_dataframe()
        assert list(frame.columns) == ['pressure_dbar', 'temperature',
                                       'salinity']
        assert frame['pressure_dbar'].tolist() == [5.0, 100.0, 1000.0]
        assert frame['temperature'].tolist() == [20.0, 15.0, 5.0]
        assert frame['salinity'].tolist() == [36.0, 36.2, 35.0]

    def test_the_dict_form_rebuilds_the_cast(self, monkeypatch, tmp_path):
        prof = self._cast(monkeypatch)
        path = tmp_path / 'cast.npz'
        np.savez(path, **prof.to_dict())
        back = argo.ArgoProfile.from_dict(dict(np.load(path,
                                                       allow_pickle=True)))
        assert back.provenance.source is prof.provenance.source
        assert back.provenance.data_point == prof.provenance.data_point
        assert back.provenance.product == prof.provenance.product
        for name in ('platform', 'cycle', 'direction', 'lat', 'lon',
                     'distance_km', 'time', 'data_mode'):
            assert getattr(back, name) == getattr(prof, name), name
        for name in argo.ArgoProfile._ARRAY_FIELDS:
            assert np.array_equal(getattr(back, name), getattr(prof, name))


def test_the_absorption_column_reads_the_casts_columns(monkeypatch):
    """With ``ssp_sources='argo'`` the Francois-Garrison T/S column is the
    cast's own: depths from its pressures at its latitude, its temperature
    and its salinity."""
    import uacpy.data.environment as environment
    from uacpy.core.acoustics.seawater import pressure_dbar_to_depth
    monkeypatch.setattr(argo, 'http_get', lambda url, **kw: _csv(_ROWS))
    monkeypatch.setattr(environment, '_fetch_ph', lambda point, **kw:
                        (8.0, None))
    seen = {}

    def capture(depths, temp, sal, **kw):
        seen.update(depths=depths, temp=temp, sal=sal)
        return 'absorption'

    monkeypatch.setattr(environment.FrancoisGarrison,
                        'from_temperature_salinity', capture)
    out = environment._fetch_absorption(
        (30.0, -40.0), date='2024-06-04', ssp_source='argo',
        ssp_backend=None, cache_only=False, resolution=None, timeout=1.0,
        verbose=False)
    assert out == ('absorption', 'argo', None)
    np.testing.assert_array_equal(
        seen['depths'], pressure_dbar_to_depth(np.array([5.0, 100.0, 1000.0]),
                                               30.1))
    assert list(seen['temp']) == [20.0, 15.0, 5.0]
    assert list(seen['sal']) == [36.0, 36.2, 35.0]


def test_a_level_with_a_bad_pressure_flag_is_dropped(monkeypatch):
    rows = _ROWS[:3] + [
        "4900001,1,A,2024-06-04T00:00:00Z,30.1,-40.1,700,8,35.2,1,1,4,1\n",
    ]
    monkeypatch.setattr(argo, 'http_get', lambda url, **kw: _csv(rows))
    prof = argo.fetch_argo_profile((30.0, -40.0), date='2024-06-04')
    assert prof.pressure_dbar.tolist() == [5.0, 100.0, 1000.0]


@pytest.mark.parametrize('position_qc, platform', [
    ('2', '4900001'),     # probably-good position: the nearer cast stands
    ('3', '4900002'),     # probably-bad position: the cast is dropped
    ('4', '4900002'),
])
def test_a_cast_is_kept_only_with_a_good_position_flag(monkeypatch,
                                                       position_qc, platform):
    """A cast whose position is not good describes water somewhere else, so
    the next cast is the nearest usable one."""
    near = [r.replace(',1\n', f',{position_qc}\n') for r in _ROWS[:3]]
    monkeypatch.setattr(argo, 'http_get',
                        lambda url, **kw: _csv(near + _ROWS[4:]))
    prof = argo.fetch_argo_profile((30.0, -40.0), date='2024-06-04',
                                   max_distance_km=500.0)
    assert prof.platform == platform


def test_fetch_ssp_argo_builds_profile(monkeypatch):
    monkeypatch.setattr(argo, 'http_get', lambda url, **kw: _csv(_ROWS))
    ssp = argo.fetch_ssp_argo((30.0, -40.0), date='2024-06-04')
    assert isinstance(ssp, SoundSpeedProfile)
    assert ssp.n_depths == 3
    assert np.all((1450 < ssp.sound_speed) & (ssp.sound_speed < 1560))
    assert ssp.depths[0] < ssp.depths[-1]          # increasing depth


def test_no_profile_raises(monkeypatch):
    monkeypatch.setattr(argo, 'http_get', lambda url, **kw: _csv([]))
    with pytest.raises(DataFetchError, match='No Argo profile'):
        argo.fetch_argo_profile((0.0, -150.0), date='2024-06-04')


def _http_error(code):
    import urllib.error

    def fail(url, **kw):
        cause = urllib.error.HTTPError(url, code, 'x', {}, None)
        raise DataFetchError(f"Request to {url} failed: HTTP {code}.") from cause
    return fail


def test_an_empty_tabledap_answer_is_no_profile(monkeypatch):
    monkeypatch.setattr(argo, 'http_get', _http_error(404))
    with pytest.raises(DataFetchError, match='No Argo profile') as info:
        argo.fetch_argo_profile((46.5, 2.5), date='2024-06-04')
    assert 'max_distance_km' in info.value.remediation


def test_another_http_failure_keeps_its_own_message(monkeypatch):
    monkeypatch.setattr(argo, 'http_get', _http_error(500))
    with pytest.raises(DataFetchError, match='HTTP 500'):
        argo.fetch_argo_profile((46.5, 2.5), date='2024-06-04')


def test_max_days_zero_ranks_same_day_casts_on_distance(monkeypatch):
    monkeypatch.setattr(argo, 'http_get', lambda url, **kw: _csv(_ROWS))
    prof = argo.fetch_argo_profile((30.0, -40.0), date='2024-06-04',
                                   max_days=0)
    assert prof.platform == '4900001'


@pytest.mark.parametrize('kw', [
    {'max_days': -1}, {'max_days': 1.5}, {'max_days': True},
    {'max_distance_km': 0.0}, {'max_distance_km': -5.0},
    {'max_distance_km': float('nan')},
])
def test_an_invalid_tolerance_is_a_configuration_error(monkeypatch, kw):
    monkeypatch.setattr(argo, 'http_get', lambda url, **k: pytest.fail(
        "an invalid tolerance reached the network"))
    with pytest.raises(ConfigurationError,
                       match='Argo: max_(days|distance_km) must be'):
        argo.fetch_argo_profile((30.0, -40.0), date='2024-06-04', **kw)


def test_argo_mackenzie_reads_the_depths_at_the_floats_latitude(monkeypatch):
    from uacpy.core.acoustics import sound_speed_mackenzie
    from uacpy.core.acoustics.seawater import pressure_dbar_to_depth
    monkeypatch.setattr(argo, 'http_get', lambda url, **kw: _csv(_ROWS))
    ssp = argo.fetch_ssp_argo((30.0, -40.0), date='2024-06-04',
                              formula='mackenzie')
    z = pressure_dbar_to_depth(np.array([5.0, 100.0, 1000.0]), 30.1)
    expected = sound_speed_mackenzie(temperature=np.array([20.0, 15.0, 5.0]),
                                     salinity=np.array([36.0, 36.2, 35.0]),
                                     depth=z)
    np.testing.assert_allclose(ssp.sound_speed[:, 0], expected, rtol=0, atol=1e-9)


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
    "3902110,463,D,2023-03-08T09:34:00Z,58.9670,20.1725,10,2.66,7.04,1,1,1,1\n",
    "3902110,463,D,2023-03-08T09:34:00Z,58.9670,20.1725,20,2.66,7.04,1,1,1,1\n",
    "3902110,463,D,2023-03-08T09:34:00Z,58.9670,20.1725,30,2.70,7.10,1,1,1,1\n",
    "3902110,463,A,2023-03-12T09:08:30Z,58.7737,19.9920,10,2.49,7.08,1,1,1,1\n",
    "3902110,463,A,2023-03-12T09:08:30Z,58.7737,19.9920,20,2.50,7.08,1,1,1,1\n",
    "3902110,463,A,2023-03-12T09:08:30Z,58.7737,19.9920,30,2.52,7.12,1,1,1,1\n",
]


class TestCastsOfOneCycleAreDistinctStations:

    def test_the_two_casts_are_not_merged(self, monkeypatch):
        monkeypatch.setattr(argo, 'http_get',
                            lambda url, **kw: _csv(_TWO_CAST_ROWS))
        prof = argo.fetch_argo_profile((58.967, 20.1725), date='2023-03-08',
                                       max_distance_km=25.0, max_days=6)
        assert prof.direction == 'D'                # at the requested point
        assert prof.pressure_dbar.tolist() == [10.0, 20.0, 30.0], (
            f"got {prof.pressure_dbar.tolist()} — the descent and ascent casts have "
            f"been interleaved into one column")
        assert np.unique(prof.pressure_dbar).size == prof.pressure_dbar.size
        assert np.all(np.diff(prof.pressure_dbar) > 0)

    def test_each_cast_stays_selectable(self, monkeypatch):
        """Both casts must remain separate candidates — the fix must not drop
        one, only stop them merging."""
        monkeypatch.setattr(argo, 'http_get',
                            lambda url, **kw: _csv(_TWO_CAST_ROWS))
        prof = argo.fetch_argo_profile((58.7737, 19.9920), date='2023-03-12',
                                       max_distance_km=25.0, max_days=6)
        assert prof.direction == 'A'
        assert prof.pressure_dbar.size == 3

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
    from uacpy.core.acoustics.seawater import depth_to_pressure_dbar, pressure_dbar_to_depth
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
        return (f"{platform},1,A,{time_str},45.0,-30.0,{pres},10.0,35.0,"
                f"1,1,1,1{_REALTIME_TAIL}")

    body = _argo_csv([_row('1', '', 10.0), _row('1', '', 20.0),
                      _row('2', '2026-08-14T00:00:00Z', 10.0),
                      _row('2', '2026-08-14T00:00:00Z', 20.0)])
    monkeypatch.setattr(
        argo, 'http_get',
        lambda url, timeout=None, verbose=False, source=None: body)
    profile = argo.fetch_argo_profile((45.0, -30.0), date='2026-08-15')
    assert profile.platform == '2'


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


@pytest.mark.parametrize('formula', ['unesco', 'delgrosso', 'teos10'])
def test_argo_profile_records_the_formula_that_built_it(monkeypatch, formula):
    """The profile has to carry its own equation: a float ending at 1000 m over
    a deep seafloor is extended, and the extension continues under whatever
    ``formula`` says (TEOS-10 when it says nothing)."""
    monkeypatch.setattr(argo, 'http_get', lambda url, **kw: _csv(_ROWS))
    ssp = argo.fetch_ssp_argo((30.0, -40.0), date='2024-06-04', formula=formula)
    assert ssp.formula == formula


def test_an_argo_profile_extends_under_its_own_equation(monkeypatch):
    """A float ending at 1000 m over a deep seafloor is extended. Without the
    stamp the extension always ran the package default. Both branches below
    extend the very same numbers, so the stamp is the only variable; UNESCO
    is the stamp because its deep pressure term is the one that differs from
    the TEOS-10 default (Del Grosso's agrees with it to a few cm/s)."""
    import warnings
    from uacpy.core.environment import SoundSpeedProfile
    from uacpy.data.sound_speed import extend_ssp_below_data
    monkeypatch.setattr(argo, 'http_get', lambda url, **kw: _csv(_ROWS))
    ssp = argo.fetch_ssp_argo((30.0, -40.0), date='2024-06-04',
                              formula='unesco')
    stripped = SoundSpeedProfile(depths=ssp.depths, sound_speed=ssp.sound_speed,
                                 kind='measured', formula=None)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        stamped = float(np.asarray(
            extend_ssp_below_data(ssp, 5000.0).sound_speed)[-1, 0])
        default = float(np.asarray(
            extend_ssp_below_data(stripped, 5000.0).sound_speed)[-1, 0])
    assert abs(stamped - default) > 0.05


def test_a_repeated_pressure_level_keeps_its_first_sample(monkeypatch):
    rows = [
        "4900001,1,A,2024-06-04T00:00:00Z,30.1,-40.1,5,20,36,1,1,1,1\n",
        "4900001,1,A,2024-06-04T00:00:00Z,30.1,-40.1,5,21,36.5,1,1,1,1\n",
        "4900001,1,A,2024-06-04T00:00:00Z,30.1,-40.1,10,18,36,1,1,1,1\n",
        "4900001,1,A,2024-06-04T00:00:00Z,30.1,-40.1,20,15,36,1,1,1,1\n",
    ]
    monkeypatch.setattr(argo, 'http_get', lambda url, **kw: _csv(rows))
    prof = argo.fetch_argo_profile((30.0, -40.0), date='2024-06-04')
    assert prof.pressure_dbar.tolist() == [5.0, 10.0, 20.0]
    assert prof.temperature[0] == 20.0                  # first sample of the pair
    assert prof.salinity[0] == 36.0
    ssp = argo.fetch_ssp_argo((30.0, -40.0), date='2024-06-04')
    assert ssp.n_depths == 3                        # strictly increasing depths


def test_a_negative_surface_pressure_becomes_the_0_m_node(monkeypatch):
    rows = [
        "4900001,1,A,2024-06-04T00:00:00Z,30.1,-40.1,-0.2,20,36,1,1,1,1\n",
        "4900001,1,A,2024-06-04T00:00:00Z,30.1,-40.1,0.0,20,36,1,1,1,1\n",
        "4900001,1,A,2024-06-04T00:00:00Z,30.1,-40.1,10,18,36,1,1,1,1\n",
        "4900001,1,A,2024-06-04T00:00:00Z,30.1,-40.1,20,15,36,1,1,1,1\n",
    ]
    monkeypatch.setattr(argo, 'http_get', lambda url, **kw: _csv(rows))
    prof = argo.fetch_argo_profile((30.0, -40.0), date='2024-06-04')
    assert prof.pressure_dbar.tolist() == [0.0, 10.0, 20.0]   # clamped, then merged
    ssp = argo.fetch_ssp_argo((30.0, -40.0), date='2024-06-04')
    assert ssp.depths[0] == 0.0
    assert np.all(np.diff(ssp.depths) > 0)
    env = Environment(ssp=ssp, bathymetry=50.0)
    assert env.ssp.depths[0] == 0.0                 # the deck's first SSP row


# The levels of a cast are read from the triplet its ``data_mode`` names. The
# delayed-mode rows below are modelled on the measured case: float 6901192
# cycle 251 (2023-05-17, D mode), whose 3.5 dbar level carries raw psal 28.869
# with raw psal_qc 1 while its psal_adjusted is NaN with psal_adjusted_qc 4.
def _row(platform, mode, pres, temp, psal, raw_qc, adjusted, adjusted_qc,
         lat=30.1, lon=-40.1):
    pa, ta, sa = adjusted
    return (f"{platform},1,A,2024-06-04T00:00:00Z,{lat},{lon},{pres},{temp},"
            f"{psal},{raw_qc},{raw_qc},{raw_qc},1,{mode},{pa},{ta},{sa},"
            f"{adjusted_qc},{adjusted_qc},{adjusted_qc}\n")


class TestTheDataModeChoosesTheValuesRead:

    def test_a_delayed_mode_cast_reads_the_adjusted_values(self, monkeypatch):
        rows = [_row('4900001', 'D', 5.0, 20.0, 36.0, 1, (5.4, 20.01, 36.05), 1),
                _row('4900001', 'D', 100.0, 15.0, 36.2, 1,
                     (100.4, 15.01, 36.24), 1)]
        monkeypatch.setattr(argo, 'http_get', lambda url, **kw: _csv(rows))
        prof = argo.fetch_argo_profile((30.0, -40.0), date='2024-06-04')
        assert prof.data_mode == 'D'
        assert prof.pressure_dbar.tolist() == [5.4, 100.4]
        assert prof.temperature.tolist() == [20.01, 15.01]
        assert prof.salinity.tolist() == [36.05, 36.24]
        assert prof.provenance.product == (
            'delayed-mode adjusted values (data_mode D)')

    def test_a_real_time_adjusted_cast_reads_the_adjusted_values(
            self, monkeypatch):
        rows = [_row('4900001', 'A', 5.0, 20.0, 36.0, 1, (5.0, 20.0, 35.9), 1),
                _row('4900001', 'A', 100.0, 15.0, 36.2, 1,
                     (100.0, 15.0, 36.1), 1)]
        monkeypatch.setattr(argo, 'http_get', lambda url, **kw: _csv(rows))
        prof = argo.fetch_argo_profile((30.0, -40.0), date='2024-06-04')
        assert prof.data_mode == 'A'
        assert prof.salinity.tolist() == [35.9, 36.1]
        assert prof.provenance.product == (
            'real-time adjusted values (data_mode A)')

    def test_a_real_time_cast_reads_the_raw_values(self, monkeypatch):
        monkeypatch.setattr(argo, 'http_get', lambda url, **kw: _csv(_ROWS))
        prof = argo.fetch_argo_profile((30.0, -40.0), date='2024-06-04')
        assert prof.data_mode == 'R'
        assert prof.salinity.tolist() == [36.0, 36.2, 35.0]
        assert prof.provenance.product == 'real-time values (data_mode R)'

    def test_a_level_rejected_in_delayed_mode_is_dropped(self, monkeypatch):
        """Raw psal_qc 1 passes, but the adjusted verdict is the one read."""
        rows = [_row('4900001', 'D', 3.5, 12.658, 28.869, 1,
                     (3.5, 12.658, 'NaN'), 4),
                _row('4900001', 'D', 100.0, 12.4, 35.4, 1,
                     (100.0, 12.4, 35.4), 1)]
        monkeypatch.setattr(argo, 'http_get', lambda url, **kw: _csv(rows))
        prof = argo.fetch_argo_profile((30.0, -40.0), date='2024-06-04')
        assert prof.pressure_dbar.tolist() == [100.0]
        assert 28.869 not in prof.salinity.tolist()

    def test_a_delayed_mode_level_is_judged_by_its_adjusted_flags(
            self, monkeypatch):
        rows = [_row('4900001', 'D', 5.0, 20.0, 36.0, 1, (5.0, 20.0, 36.0), 4),
                _row('4900001', 'D', 100.0, 15.0, 36.2, 4,
                     (100.0, 15.0, 36.2), 1)]
        monkeypatch.setattr(argo, 'http_get', lambda url, **kw: _csv(rows))
        prof = argo.fetch_argo_profile((30.0, -40.0), date='2024-06-04')
        assert prof.pressure_dbar.tolist() == [100.0]

    def test_a_missing_adjusted_value_is_dropped_not_read_raw(
            self, monkeypatch):
        rows = [_row('4900001', 'D', 5.0, 20.0, 36.0, 1, (5.0, 20.0, 'NaN'), 1),
                _row('4900001', 'D', 100.0, 15.0, 36.2, 1,
                     (100.0, 15.0, 36.2), 1)]
        monkeypatch.setattr(argo, 'http_get', lambda url, **kw: _csv(rows))
        prof = argo.fetch_argo_profile((30.0, -40.0), date='2024-06-04')
        assert prof.pressure_dbar.tolist() == [100.0]

    def test_a_cast_rejected_in_delayed_mode_yields_to_the_next_cast(
            self, monkeypatch):
        rows = [_row('4900001', 'D', 3.5, 12.658, 28.869, 1,
                     (3.5, 12.658, 'NaN'), 4),
                _row('4900002', 'R', 5.0, 19.0, 36.0, 1,
                     ('NaN', 'NaN', 'NaN'), '', lat=31.0, lon=-41.0)]
        monkeypatch.setattr(argo, 'http_get', lambda url, **kw: _csv(rows))
        prof = argo.fetch_argo_profile((30.0, -40.0), date='2024-06-04')
        assert prof.platform == '4900002'

    def test_a_level_without_a_data_mode_is_dropped(self, monkeypatch):
        rows = [_row('4900001', '', 5.0, 20.0, 36.0, 1, (5.0, 20.0, 36.0), 1),
                _row('4900001', 'R', 100.0, 15.0, 36.2, 1,
                     ('NaN', 'NaN', 'NaN'), '')]
        monkeypatch.setattr(argo, 'http_get', lambda url, **kw: _csv(rows))
        prof = argo.fetch_argo_profile((30.0, -40.0), date='2024-06-04')
        assert prof.pressure_dbar.tolist() == [100.0]

    def test_the_query_requests_the_mode_and_the_adjusted_triplet(self):
        when = np.datetime64('2024-06-04', 'D')
        url = argo._query_url((30.0, -40.0), when, 250.0, 15, 'http://x')
        columns = url.split('?', 1)[1].split('&', 1)[0].split(',')
        for name in ('data_mode', 'pres_adjusted', 'temp_adjusted',
                     'psal_adjusted', 'pres_adjusted_qc', 'temp_adjusted_qc',
                     'psal_adjusted_qc'):
            assert name in columns
