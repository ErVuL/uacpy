"""The export protocol on ResultStack, comms.ReceiverDiagnostics and
comms.JanusPacket: every save form reads back the object it saved, and a
pickle written before these types joined the protocol still loads."""
import pickle
from pathlib import Path

import numpy as np
import pytest

from uacpy.comms import JanusPacket, ReceiverDiagnostics
from uacpy.core._export import CarrierExport, ExportRecord
from uacpy.core.results import Arrivals, Field, ResultStack

_PICKLED = Path(__file__).parent / 'data' / 'pickled'


# ── ResultStack ───────────────────────────────────────────────────────────

def _field(scale, *, complex_data=True):
    data = np.arange(6, dtype=float).reshape(2, 3) * scale
    if complex_data:
        data = data * (1 - 0.5j)
    return Field(data=data,
                 coords={'depth': np.array([10.0, 20.0]),
                         'range': np.array([100.0, 200.0, 300.0])},
                 kind='pressure', unit='Pa', model='Kraken',
                 frequencies=[50.0], source_depths=[scale],
                 phase_reference='travelling_wave')


def _field_stack(**kw):
    return ResultStack([_field(5.0, **kw), _field(15.0, **kw)], [5.0, 15.0],
                       coordinate_name='source_depth')


def _arrivals(depth):
    cell = {'amplitudes': np.array([1.0, 0.5]), 'phases': np.array([0.1, 0.2]),
            'delays': np.array([0.10, 0.12]),
            'delays_imag': np.array([0.0, -1e-6]),
            'source_angles': np.array([5.0, -7.0]),
            'receiver_angles': np.array([-5.0, 7.0]),
            'n_top_bounces': np.array([0, 1], dtype='int32'),
            'n_bot_bounces': np.array([1, 1], dtype='int32'), 'n_arrivals': 2}
    return Arrivals(by_receiver=[[[cell]]], receiver_depths=[20.0],
                    receiver_ranges=[1000.0], frequencies=1000.0,
                    source_depths=[depth], model='Bellhop')


def _same_stacks(a, b):
    assert type(a) is type(b)
    assert a.coordinate_name == b.coordinate_name
    np.testing.assert_array_equal(a.coordinate, b.coordinate)
    assert len(a.slabs) == len(b.slabs)
    for x, y in zip(a.slabs, b.slabs):
        assert type(x) is type(y)
        dx, dy = x.to_dict(), y.to_dict()
        assert dx.keys() == dy.keys()
        for key in dx:
            if isinstance(dx[key], np.ndarray):
                np.testing.assert_array_equal(dx[key], dy[key], err_msg=key)
                assert dx[key].dtype == np.asarray(dy[key]).dtype, key


class TestAResultStackRoundTrips:

    def test_from_dict_of_to_dict_is_the_stack(self):
        stack = _field_stack()
        _same_stacks(stack, ResultStack.from_dict(stack.to_dict()))

    def test_each_slab_is_saved_under_its_class_path(self):
        d = _field_stack().to_dict()
        assert d['__class__'] == 'uacpy.core.results.ResultStack'
        assert [s['__class__'] for s in d['slabs']] == \
            ['uacpy.core.results.Field'] * 2

    def test_a_stack_of_any_slab_type_round_trips(self):
        """Not only Field: a stack of Arrivals (which has no xarray form as
        one array) saves and loads through to_dict."""
        stack = ResultStack([_arrivals(5.0), _arrivals(5.0)], [1.0, 2.0],
                            coordinate_name='case')
        back = ResultStack.from_dict(stack.to_dict())
        assert all(isinstance(s, Arrivals) for s in back.slabs)
        assert back.coordinate_name == 'case'
        _same_stacks(stack, back)

    def test_the_npz_round_trip_is_the_stack(self, tmp_path):
        stack = _field_stack()
        np.savez(tmp_path / 's.npz', **stack.to_dict())
        with np.load(tmp_path / 's.npz', allow_pickle=True) as f:
            _same_stacks(stack, ResultStack.from_dict(dict(f)))

    def test_from_xarray_of_to_xarray_is_the_stack(self):
        pytest.importorskip('xarray')
        stack = _field_stack()
        _same_stacks(stack, ResultStack.from_xarray(stack.to_xarray()))

    @pytest.mark.parametrize('complex_data', [True, False])
    def test_the_netcdf_round_trip_is_the_stack(self, tmp_path, complex_data):
        xr = pytest.importorskip('xarray')
        pytest.importorskip('h5netcdf')
        stack = _field_stack(complex_data=complex_data)
        stack.to_netcdf(tmp_path / 's.nc')
        with xr.open_dataarray(tmp_path / 's.nc', engine='h5netcdf') as da:
            back = ResultStack.from_xarray(da.load())
        _same_stacks(stack, back)

    @pytest.mark.parametrize('complex_data', [True, False])
    def test_from_netcdf_reads_back_what_to_netcdf_wrote(self, tmp_path,
                                                         complex_data):
        pytest.importorskip('xarray')
        stack = _field_stack(complex_data=complex_data)
        stack.to_netcdf(tmp_path / 's.nc')
        _same_stacks(stack, ResultStack.from_netcdf(tmp_path / 's.nc'))

    def test_a_foreign_class_path_is_refused(self):
        from uacpy.core.exceptions import ConfigurationError
        d = _field_stack().to_dict()
        d['slabs'][0]['__class__'] = 'os.path.join'
        with pytest.raises(ConfigurationError, match='not a uacpy class'):
            ResultStack.from_dict(d)


def test_a_real_field_writes_to_netcdf_and_reads_back(tmp_path):
    """A real-valued Field carries the boolean attribute ``coherent``, which
    the netCDF4 backend refuses; the writer picks a backend that stores it,
    as it does for complex values."""
    xr = pytest.importorskip('xarray')
    pytest.importorskip('h5netcdf')
    field = _field(5.0, complex_data=False)
    field.to_netcdf(tmp_path / 'f.nc')
    with xr.open_dataarray(tmp_path / 'f.nc', engine='h5netcdf') as da:
        back = Field.from_xarray(da.load())
    np.testing.assert_array_equal(back.data, field.data)
    assert back.coherent == field.coherent


# ── ReceiverDiagnostics ───────────────────────────────────────────────────

def _diagnostics(mse=True):
    return ReceiverDiagnostics(
        bits=np.array([1, 0, 1, 1]), symbols=np.array([1 + 1j, -1 - 1j]),
        mse=np.array([0.1, 0.05]) if mse else None,
        sync_metric=np.array([0.0, 0.4, 1.0]), start=1)


def _same_records(a, b, fields):
    assert type(a) is type(b)
    for name in fields:
        x, y = getattr(a, name), getattr(b, name)
        if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
            np.testing.assert_array_equal(x, y, err_msg=name)
        else:
            assert x == y, name


_DIAGNOSTIC_FIELDS = ('bits', 'symbols', 'mse', 'sync_metric', 'start')


class TestReceiverDiagnosticsIsAnExportRecord:

    def test_it_is_on_the_protocol(self):
        assert issubclass(ReceiverDiagnostics, ExportRecord)

    @pytest.mark.parametrize('mse', [True, False])
    def test_from_dict_of_to_dict_is_the_record(self, mse):
        d = _diagnostics(mse)
        _same_records(d, ReceiverDiagnostics.from_dict(d.to_dict()),
                      _DIAGNOSTIC_FIELDS)

    def test_the_npz_round_trip_is_the_record(self, tmp_path):
        d = _diagnostics()
        np.savez(tmp_path / 'd.npz', **d.to_dict())
        with np.load(tmp_path / 'd.npz', allow_pickle=True) as f:
            back = ReceiverDiagnostics.from_dict(dict(f))
        _same_records(d, back, _DIAGNOSTIC_FIELDS)

    def test_from_xarray_of_to_xarray_is_the_record(self):
        pytest.importorskip('xarray')
        d = _diagnostics()
        _same_records(d, ReceiverDiagnostics.from_xarray(d.to_xarray()),
                      _DIAGNOSTIC_FIELDS)

    def test_the_netcdf_round_trip_is_the_record(self, tmp_path):
        xr = pytest.importorskip('xarray')
        pytest.importorskip('h5netcdf')
        d = _diagnostics()
        d.to_netcdf(tmp_path / 'd.nc')
        with xr.open_dataset(tmp_path / 'd.nc', engine='h5netcdf') as ds:
            back = ReceiverDiagnostics.from_xarray(ds.load())
        _same_records(d, back, _DIAGNOSTIC_FIELDS)

    def test_its_arrays_are_read_only(self):
        d = _diagnostics()
        with pytest.raises(ValueError, match='read-only'):
            d.bits[0] = 0

    def test_a_pickle_written_before_it_joined_the_protocol_loads(self):
        """``receiver_diagnostics.pkl`` was pickled from the plain frozen
        dataclass this record was before it joined the protocol."""
        back = pickle.loads((_PICKLED / 'receiver_diagnostics.pkl')
                            .read_bytes())
        assert isinstance(back, ReceiverDiagnostics)
        np.testing.assert_array_equal(back.bits, [1, 0, 1])
        np.testing.assert_array_equal(back.symbols, [1 + 1j, -1 - 1j])
        np.testing.assert_array_equal(back.mse, [0.1, 0.2])
        np.testing.assert_array_equal(back.sync_metric, [0.0, 0.5, 1.0])
        assert back.start == 1
        assert not back.bits.flags.writeable


# ── JanusPacket ───────────────────────────────────────────────────────────

def _packet():
    return JanusPacket(class_id=16, app_type=3,
                       app_data=np.arange(34) % 2, mobility=1, forward=1)


class TestJanusPacketIsACarrier:

    def test_it_is_on_the_protocol(self):
        assert issubclass(JanusPacket, CarrierExport)

    def test_from_dict_of_to_dict_is_the_packet(self):
        p = _packet()
        assert JanusPacket.from_dict(p.to_dict()) == p

    def test_the_npz_round_trip_is_the_packet(self, tmp_path):
        p = _packet()
        np.savez(tmp_path / 'p.npz', **p.to_dict())
        with np.load(tmp_path / 'p.npz', allow_pickle=True) as f:
            back = JanusPacket.from_dict(dict(f))
        assert back == p
        assert back.to_bits().tolist() == p.to_bits().tolist()

    def test_from_xarray_of_to_xarray_is_the_packet(self):
        pytest.importorskip('xarray')
        p = _packet()
        assert JanusPacket.from_xarray(p.to_xarray()) == p

    def test_the_netcdf_round_trip_is_the_packet(self, tmp_path):
        xr = pytest.importorskip('xarray')
        pytest.importorskip('h5netcdf')
        p = _packet()
        p.to_netcdf(tmp_path / 'p.nc')
        with xr.open_dataset(tmp_path / 'p.nc', engine='h5netcdf') as ds:
            assert JanusPacket.from_xarray(ds.load()) == p

    def test_a_pickle_written_before_it_joined_the_protocol_loads(self):
        """``janus_packet.pkl`` was pickled from the plain dataclass this
        carrier was before it joined the protocol."""
        back = pickle.loads((_PICKLED / 'janus_packet.pkl').read_bytes())
        assert back == JanusPacket(class_id=16, app_type=3,
                                   app_data=np.arange(34) % 2, mobility=1)
