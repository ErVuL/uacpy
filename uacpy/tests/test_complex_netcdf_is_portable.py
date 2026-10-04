"""A complex result writes a portable netCDF file by default: no warning, a
file plain netCDF4 opens, and a bit-exact round trip through from_netcdf and
through a plain xarray open handed to from_xarray, under each backend."""
import importlib.util
import warnings

import numpy as np
import pytest

from uacpy.core._export import COMPLEX_DIM, join_complex, split_complex
from uacpy.core.results import Field, ResultStack

xr = pytest.importorskip('xarray')

ENGINES = [e for e, module in (('netcdf4', 'netCDF4'), ('h5netcdf', 'h5netcdf'))
           if importlib.util.find_spec(module) is not None]


def _field(dtype=np.complex128):
    data = (np.arange(6.0).reshape(2, 3) * (1 - 0.5j)).astype(dtype)
    data[0, 1] = complex(np.nan, 2.0)
    data[1, 2] = complex(-0.0, -0.0)
    return Field(data=data,
                 coords={'depth': np.array([10.0, 20.0]),
                         'range': np.array([100.0, 200.0, 300.0])},
                 kind='pressure', unit='Pa', model='Kraken',
                 frequencies=[50.0], source_depths=[5.0])


def _warnings_of(call):
    """The warnings ``call`` raises, less those about the installed NumPy
    (netCDF4 built against another NumPy, NumPy's own deprecations inside
    netCDF4): the install's, not the save's."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        call()
    return [f"{w.category.__name__}: {w.message}" for w in record
            if 'numpy' not in str(w.message).lower()]


def _same_bits(a, b):
    assert b.dtype == a.dtype
    assert a.real.tobytes() == b.real.tobytes()
    assert a.imag.tobytes() == b.imag.tobytes()


@pytest.mark.parametrize('dtype', [np.complex128, np.complex64])
def test_a_default_complex_save_warns_nothing_and_round_trips_bit_exactly(
        tmp_path, dtype):
    field = _field(dtype)
    assert _warnings_of(lambda: field.to_netcdf(tmp_path / 'f.nc')) == []
    _same_bits(field.data, Field.from_netcdf(tmp_path / 'f.nc').data)


def test_the_file_opens_with_plain_netcdf4(tmp_path):
    netCDF4 = pytest.importorskip('netCDF4')
    _field().to_netcdf(tmp_path / 'f.nc')
    with netCDF4.Dataset(tmp_path / 'f.nc') as nc:
        variable = nc.variables['pressure']
        assert variable.dimensions[-1] == COMPLEX_DIM
        assert variable.dtype == np.float64


@pytest.mark.parametrize('engine', ENGINES)
def test_each_backend_writes_it_and_a_plain_open_reads_it_back(tmp_path,
                                                              engine):
    field = _field()
    assert _warnings_of(
        lambda: field.to_netcdf(tmp_path / 'f.nc', engine=engine)) == []
    for reader in ENGINES:
        with xr.open_dataarray(tmp_path / 'f.nc', engine=reader) as da:
            _same_bits(field.data, Field.from_xarray(da.load()).data)


def test_a_complex_stack_round_trips_through_a_plain_open(tmp_path):
    slabs = [_field(), _field()]
    for slab, depth in zip(slabs, (5.0, 15.0)):
        slab.source_depths = np.array([depth])
    stack = ResultStack(slabs, [5.0, 15.0], coordinate_name='source_depth')
    stack.to_netcdf(tmp_path / 's.nc')
    with xr.open_dataarray(tmp_path / 's.nc') as da:
        back = ResultStack.from_xarray(da.load())
    _same_bits(stack.slabs[1].data, back.slabs[1].data)


@pytest.mark.parametrize('form', ['dataarray', 'dataset'])
def test_split_then_join_is_the_identity_bit_for_bit(form):
    da = _field().to_xarray()
    obj = da if form == 'dataarray' else da.to_dataset()
    split = split_complex(obj)
    variable = split if form == 'dataarray' else split['pressure']
    assert not np.iscomplexobj(variable.values)
    assert variable.dims[-1] == COMPLEX_DIM
    back = join_complex(split)
    joined = back if form == 'dataarray' else back['pressure']
    _same_bits(da.values, joined.values)
    assert joined.dims == da.dims


def test_a_real_object_passes_through_unchanged():
    da = _field().to_xarray().real
    assert split_complex(da) is da
    assert join_complex(da) is da
