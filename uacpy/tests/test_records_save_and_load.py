"""Every public record type saves and loads: the dataclass carriers through
CarrierExport, ChannelTaps as a ResultTuple, WenzNoise by its constructor
arguments, DataSource and ParallelResult through to_dict / from_dict. A pickle
written before a type joined the protocol still loads."""
import pickle
from pathlib import Path

import numpy as np
import pytest

from uacpy.acoustic_signal import ChannelRegime, channel_regime
from uacpy.comms import ChannelTaps, LinkResult
from uacpy.core._export import CarrierExport
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.provenance import SOURCES, DataSource
from uacpy.io import OasnNoise, OasnReplicaGrid
from uacpy.noise import WenzNoise
from uacpy.parallel import ParallelResult
from uacpy.sonar import BottomParameters

_PICKLED = Path(__file__).parent / 'data' / 'pickled'


def _regime():
    return channel_regime([0.0, 1e-3, 3e-3], [1.0, 0.5, 0.25],
                          symbol_rate=2000.0)


def _link():
    return LinkResult(ber=0.01, evm=0.1, scheme='qpsk', ebn0_dB=8.0,
                      tx_symbols=np.array([1 + 1j, -1 - 1j]),
                      rx_symbols=np.array([0.9 + 1j, -1 - 0.9j]),
                      mse=np.array([0.1, 0.2]))


CARRIERS = {
    'ChannelRegime': _regime,
    'LinkResult': _link,
    'OasnNoise': lambda: OasnNoise(surface_level=50.0, white_level=10.0,
                                   discrete_sources=[{'depth': 10.0, 'x': 100.0,
                                                      'y': 0.0, 'level': 120.0}]),
    'OasnReplicaGrid': lambda: OasnReplicaGrid(z=(10.0, 90.0, 9), c_low=1400.0),
    'BottomParameters': lambda: BottomParameters(
        density_ratio=1.8, speed_ratio=1.1, loss_parameter=0.01,
        volume_parameter=0.002, spectral_strength=0.001),
}


def _same(a, b):
    assert type(a) is type(b)
    for name, value in vars(a).items():
        other = getattr(b, name)
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(other, value)
        else:
            assert other == value, name


@pytest.mark.parametrize('name', sorted(CARRIERS))
class TestTheCarriersRoundTrip:

    def test_it_is_on_the_protocol(self, name):
        assert isinstance(CARRIERS[name](), CarrierExport)

    def test_from_dict_of_to_dict_is_the_record(self, name):
        record = CARRIERS[name]()
        _same(record, type(record).from_dict(record.to_dict()))

    def test_from_xarray_of_to_xarray_is_the_record(self, name):
        pytest.importorskip('xarray')
        record = CARRIERS[name]()
        _same(record, type(record).from_xarray(record.to_xarray()))


def test_a_regime_equals_the_record_it_was_saved_from():
    assert ChannelRegime.from_dict(_regime().to_dict()) == _regime()


def _taps():
    return ChannelTaps(taps=np.array([1 + 0j, 0.5j]),
                       delays_s=np.array([0.0, 1e-3]), symbol_rate=1000.0,
                       fc=12000.0, sps=1, first_arrival_s=0.5)


def _same_taps(a, b):
    assert type(b) is ChannelTaps
    np.testing.assert_array_equal(b.taps, a.taps)
    np.testing.assert_array_equal(b.delays_s, a.delays_s)
    assert (b.symbol_rate, b.fc, b.sps, b.first_arrival_s) == \
        (a.symbol_rate, a.fc, a.sps, a.first_arrival_s)
    assert type(b.sps) is int and type(b.fc) is float


class TestChannelTaps:

    def test_it_unpacks_as_a_named_tuple(self):
        taps, delays, rate, fc, sps, first = _taps()
        assert _taps()._fields[0] == 'taps' and sps == 1

    def test_from_dict_of_to_dict_is_the_record(self):
        _same_taps(_taps(), ChannelTaps.from_dict(_taps().to_dict()))

    def test_the_taps_sit_on_the_delay_axis(self):
        pytest.importorskip('xarray')
        ds = _taps().to_xarray()
        assert ds['taps'].dims == ('delays_s',)
        assert ds['delays_s'].attrs['units'] == 's'
        _same_taps(_taps(), ChannelTaps.from_xarray(ds))

    def test_a_pickle_written_before_it_joined_the_protocol_loads(self):
        """``channel_taps.pkl`` was pickled from the NamedTuple this record
        was before it joined the protocol."""
        back = pickle.loads((_PICKLED / 'channel_taps.pkl').read_bytes())
        _same_taps(_taps(), back)


def test_a_link_result_pickled_before_the_protocol_loads():
    back = pickle.loads((_PICKLED / 'link_result.pkl').read_bytes())
    assert back == _link()


class TestWenzNoise:

    @staticmethod
    def _noise(**kw):
        return WenzNoise(np.geomspace(10.0, 1e4, 8), wind_speed_kn=10.0,
                         shipping_level='high', **kw)

    def test_from_dict_of_to_dict_rebuilds_the_spectrum(self):
        back = WenzNoise.from_dict(self._noise().to_dict())
        np.testing.assert_array_equal(back.total, self._noise().total)
        assert back.models == self._noise().models

    def test_the_npz_round_trip_rebuilds_the_spectrum(self, tmp_path):
        np.savez(tmp_path / 'w.npz', **self._noise().to_dict())
        with np.load(tmp_path / 'w.npz', allow_pickle=True) as f:
            back = WenzNoise.from_dict(dict(f))
        np.testing.assert_array_equal(back.wind, self._noise().wind)

    def test_one_without_a_grid_round_trips(self):
        back = WenzNoise.from_dict(WenzNoise(wind_speed_kn=10.0).to_dict())
        assert back.frequencies is None

    def test_the_xarray_form_holds_the_six_spectra(self):
        pytest.importorskip('xarray')
        ds = self._noise().to_xarray()
        assert set(ds.data_vars) == {'total', 'wind', 'shipping', 'rain',
                                     'thermal', 'turbulence'}
        assert ds['total'].dims == ('frequency',)
        back = WenzNoise.from_xarray(ds)
        np.testing.assert_array_equal(back.total, self._noise().total)

    def test_a_callable_submodel_is_refused(self):
        noise = self._noise(wind_model=lambda f, **kw: 0.0 * f)
        with pytest.raises(ConfigurationError, match='register_noise_model'):
            noise.to_dict()


class TestDataSource:

    def test_a_catalogue_entry_loads_as_the_catalogue_s_own(self):
        entry = SOURCES['gebco']
        assert DataSource.from_dict(entry.to_dict()) is entry

    def test_an_edited_record_loads_as_a_new_one(self):
        d = SOURCES['gebco'].to_dict()
        d['name'] = 'edited'
        back = DataSource.from_dict(d)
        assert back is not SOURCES['gebco'] and back.name == 'edited'


def _field():
    from uacpy.core.results import Field
    return Field(data=np.full((1, 2), 60.0),
                 coords={'depth': np.array([10.0]),
                         'range': np.array([100.0, 200.0])},
                 kind='pressure', unit='dB', model='Synthetic')


class TestParallelResult:

    @staticmethod
    def _outcome():
        return ParallelResult(results=[_field(), None],
                              errors={1: ValueError('job failed')},
                              labels=['a', 'b'], coordinate_name='case',
                              warnings={0: [(UserWarning, 'careful')]})

    def test_to_dict_nests_each_result_by_its_class(self):
        saved = self._outcome().to_dict()['results']
        assert saved[0]['__class__'] == 'uacpy.core.results.Field'
        assert saved[1] is None

    def test_from_dict_of_to_dict_is_the_outcome(self):
        back = ParallelResult.from_dict(self._outcome().to_dict())
        np.testing.assert_array_equal(back.results[0].data,
                                      self._outcome().results[0].data)
        assert back.results[1] is None
        assert type(back.errors[1]) is ValueError
        assert str(back.errors[1]) == 'job failed'
        assert back.warnings == {0: [(UserWarning, 'careful')]}
        assert (back.labels, back.coordinate_name) == (['a', 'b'], 'case')

    def test_the_npz_round_trip_is_the_outcome(self, tmp_path):
        np.savez(tmp_path / 'p.npz', **self._outcome().to_dict())
        with np.load(tmp_path / 'p.npz', allow_pickle=True) as f:
            back = ParallelResult.from_dict(dict(f))
        assert back.results[0].model == 'Synthetic'

    def test_a_pickle_written_before_the_protocol_loads(self):
        back = pickle.loads((_PICKLED / 'parallel_result.pkl').read_bytes())
        assert back.results == [None, None]
        assert str(back.errors[1]) == 'job failed'
        assert back.warnings == {0: [(UserWarning, 'careful')]}
