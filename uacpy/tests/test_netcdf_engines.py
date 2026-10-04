"""Every result writes and reads under each netCDF backend installed:
netCDF4 and h5netcdf, each way and across. A real Field, a complex one, a
Field with boolean attributes, and results with components (a Bellhop
broadband field carrying its arrivals and a BOUNCE table, an OASS covariance
carrying its mean field) come back as they were written."""
import importlib.util
import warnings

import numpy as np
import pytest

import uacpy
from uacpy.core.exceptions import FallbackWarning
from uacpy.core.results import Covariance, Field
from uacpy.tests._synthetic_fields import _arrivals, _bounce, _broadband

xr = pytest.importorskip('xarray')

#: The netCDF backends installed here.
ENGINES = [e for e, module in (('netcdf4', 'netCDF4'), ('h5netcdf', 'h5netcdf'))
           if importlib.util.find_spec(module) is not None]
if not ENGINES:
    pytest.skip('no netCDF backend installed', allow_module_level=True)

#: Every (writing, reading) pair of backends.
PAIRS = [(w, r) for w in ENGINES for r in ENGINES]


def _field(kind):
    """A Field of each kind the writer has to carry: real, complex, and
    with boolean attributes (``coherent`` and a ``True`` metadata entry)."""
    data = np.arange(6.0).reshape(2, 3)
    if kind == 'complex':
        data = data * (1 - 0.5j)
    extra = {}
    if kind == 'bool attrs':
        extra = dict(coherent=True, metadata={'flag': True, 'n': 3})
    return Field(data=data,
                 coords={'depth': np.array([10.0, 20.0]),
                         'range': np.array([100.0, 200.0, 300.0])},
                 kind='pressure', unit='Pa', model='Kraken',
                 frequencies=[50.0], source_depths=[5.0], **extra)


def _same(a, b):
    """The same type, arrays, identity and metadata (with its value types),
    components included."""
    assert type(a) is type(b)
    da, db = a.to_dict(), b.to_dict()
    da.pop('components', None)
    db.pop('components', None)
    assert da.keys() == db.keys()
    for key in da:
        if isinstance(da[key], np.ndarray):
            np.testing.assert_array_equal(da[key], db[key], err_msg=key)
            assert np.asarray(da[key]).dtype == np.asarray(db[key]).dtype, key
    for key, value in (a.metadata or {}).items():
        # a boolean comes back a boolean; numbers by value (an attribute
        # reads back as its numpy scalar)
        if isinstance(value, bool):
            assert b.metadata[key] is value, key
        elif isinstance(value, (str, int, float, np.ndarray, list, tuple)):
            np.testing.assert_array_equal(b.metadata[key], value, err_msg=key)
    assert sorted(a.components) == sorted(b.components)
    for name in a.components:
        _same(a.components[name], b.components[name])


@pytest.mark.parametrize('write,read', PAIRS)
@pytest.mark.parametrize('kind', ['real', 'complex', 'bool attrs'])
def test_every_field_kind_round_trips_under_each_backend(tmp_path, kind,
                                                         write, read):
    field = _field(kind)
    field.to_netcdf(tmp_path / 'f.nc', engine=write)
    back = Field.from_netcdf(tmp_path / 'f.nc', engine=read)
    _same(field, back)
    assert back.coherent is field.coherent


@pytest.mark.parametrize('write,read', PAIRS)
def test_a_stack_of_fields_with_boolean_attrs_round_trips(tmp_path, write,
                                                          read):
    """The single-object writer (ResultStack.to_netcdf, every carrier's)
    stores the booleans as Result.to_netcdf does."""
    from uacpy.core.results import ResultStack
    slabs = [_field('bool attrs'), _field('bool attrs')]
    for slab, depth in zip(slabs, (5.0, 15.0)):
        slab.source_depths = np.array([depth])
    stack = ResultStack(slabs, [5.0, 15.0], coordinate_name='source_depth')
    stack.to_netcdf(tmp_path / 's.nc', engine=write)
    with xr.open_dataarray(tmp_path / 's.nc', engine=read) as da:
        back = ResultStack.from_xarray(da.load())
    assert all(s.coherent is True for s in back.slabs)
    np.testing.assert_array_equal(back.slabs[1].data, stack.slabs[1].data)


@pytest.mark.parametrize('engine', ENGINES)
def test_a_boolean_attribute_is_stored_as_int8_and_named(tmp_path, engine):
    _field('bool attrs').to_netcdf(tmp_path / 'f.nc', engine=engine)
    with xr.open_dataset(tmp_path / 'f.nc', engine=engine) as ds:
        (variable,) = ds.data_vars.values()
        names = variable.attrs['uacpy_bool_attrs'].split(',')
        assert {'coherent', 'flag'} <= set(names)
        assert variable.attrs['coherent'].dtype == np.int8


@pytest.mark.parametrize('engine', ENGINES)
def test_a_plain_open_and_from_xarray_restores_the_booleans(tmp_path, engine):
    _field('bool attrs').to_netcdf(tmp_path / 'f.nc', engine=engine)
    with xr.open_dataarray(tmp_path / 'f.nc', engine=engine) as da:
        back = Field.from_xarray(da.load())
    assert back.coherent is True
    assert back.metadata['flag'] is True
    assert 'uacpy_bool_attrs' not in back.metadata


# ── components, under each backend ────────────────────────────────────────

@pytest.mark.parametrize('write,read', PAIRS)
def test_components_round_trip_under_each_backend_without_a_warning(
        tmp_path, write, read):
    field = _broadband(components={
        'arrivals': _arrivals(components={'bounce': _bounce()}),
        'bounce': _bounce()})
    with warnings.catch_warnings():
        warnings.simplefilter('error', FallbackWarning)
        field.to_netcdf(tmp_path / 'f.nc', engine=write)
        back = Field.from_netcdf(tmp_path / 'f.nc', engine=read)
    _same(field, back)


# ── real runs ─────────────────────────────────────────────────────────────

@pytest.mark.requires_binary
@pytest.mark.parametrize('write,read', PAIRS)
def test_a_bellhop_broadband_run_over_a_layered_bottom_round_trips(
        tmp_path, write, read):
    """Bellhop's BROADBAND field over a layered bottom carries the Arrivals
    it was synthesised from and the BOUNCE table the route used."""
    from uacpy.core.bottom import Bottom
    from uacpy.core.run_settings import RunMode
    env = uacpy.Environment(
        name='layered', bathymetry=100.0, ssp=1500.0,
        bottom=Bottom.from_presets([('silt', 5.0)], halfspace='sand'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        field = uacpy.Bellhop(verbose=False).run(
            env, uacpy.Source(depths=25.0, frequencies=200.0),
            uacpy.Receiver(depths=[50.0], ranges=[1000.0]),
            run_mode=RunMode.BROADBAND,
            frequencies=np.array([180.0, 200.0, 220.0]))
    assert {'arrivals', 'bounce'} <= set(field.components)
    with warnings.catch_warnings():
        warnings.simplefilter('error', FallbackWarning)
        field.to_netcdf(tmp_path / 'f.nc', engine=write)
    _same(field, Field.from_netcdf(tmp_path / 'f.nc', engine=read))


@pytest.mark.requires_oases
@pytest.mark.parametrize('write,read', PAIRS)
def test_an_oass_covariance_round_trips_with_its_mean_field(tmp_path, write,
                                                            read):
    from uacpy.core.run_settings import RunMode
    from uacpy.tests.conftest import make_pekeris
    env = make_pekeris(
        ssp=uacpy.SoundSpeedProfile(depths=[0.0, 100.0],
                                    sound_speed=[1500.0, 1500.0]),
        roughness=0.5)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cov = uacpy.OASS(correlation_length=10.0, rms_roughness=0.5).run(
            env, uacpy.Source(depths=50.0, frequencies=100.0),
            uacpy.Receiver(depths=[30.0, 70.0],
                           ranges=np.linspace(200.0, 2000.0, 8)),
            run_mode=RunMode.COVARIANCE)
    assert isinstance(cov.components['mean_field'], Field)
    cov.to_netcdf(tmp_path / 'c.nc', engine=write)
    _same(cov, Covariance.from_netcdf(tmp_path / 'c.nc', engine=read))


# ── without h5netcdf ──────────────────────────────────────────────────────

def _backends(monkeypatch, *, h5netcdf, netcdf4_version):
    """The NetCDF backends as if installed or not: h5netcdf's lookup and
    import fail when ``h5netcdf`` is False, and netCDF4 reports
    ``netcdf4_version`` (or is not found, for ``None``)."""
    import builtins
    import importlib.metadata
    import importlib.util
    real_find_spec = importlib.util.find_spec
    real_version = importlib.metadata.version
    real_import = builtins.__import__

    def hidden(name):
        top = name.split('.')[0]
        return (top == 'h5netcdf' and not h5netcdf) or \
            (top == 'netCDF4' and netcdf4_version is None)

    def find_spec(name, *args, **kwargs):
        return None if hidden(name) else real_find_spec(name, *args, **kwargs)

    def version(name):
        if name == 'netCDF4' and netcdf4_version is not None:
            return netcdf4_version
        return real_version(name)

    def blocked_import(name, *args, **kwargs):
        if hidden(name):
            raise ImportError(f'{name} is blocked for this test')
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(importlib.util, 'find_spec', find_spec)
    monkeypatch.setattr(importlib.metadata, 'version', version)
    monkeypatch.setattr(builtins, '__import__', blocked_import)


@pytest.mark.skipif('netcdf4' not in ENGINES, reason='netCDF4 not installed')
@pytest.mark.parametrize('make', ['field', 'stack'])
def test_a_complex_write_needs_no_backend_that_stores_complex_values(
        tmp_path, monkeypatch, make):
    """Neither h5netcdf nor netCDF4 1.7.1+: a complex result still writes
    (as real and imaginary parts) and reads back as written."""
    from uacpy.core.results import ResultStack
    field = _field('complex')
    obj = field if make == 'field' else ResultStack(
        [field], [5.0], coordinate_name='source_depth')
    _backends(monkeypatch, h5netcdf=False, netcdf4_version='1.6.5')
    obj.to_netcdf(tmp_path / 'f.nc')
    with xr.open_dataarray(tmp_path / 'f.nc') as da:
        back = (Field.from_xarray(da.load()) if make == 'field'
                else ResultStack.from_xarray(da.load()).slabs[0])
    np.testing.assert_array_equal(back.data, field.data)


@pytest.mark.skipif('netcdf4' not in ENGINES, reason='netCDF4 not installed')
def test_without_h5netcdf_netcdf4_writes_and_reads_complex_values(
        tmp_path, monkeypatch):
    """netCDF4 1.7.1 or later is the second backend for complex values:
    with h5netcdf absent and no engine named, it writes them and from_netcdf
    reads them back."""
    import importlib.metadata
    if tuple(int(p) for p in importlib.metadata.version('netCDF4')
             .split('.')[:3] if p.isdigit()) < (1, 7, 1):
        pytest.skip('this netCDF4 predates its complex type')
    field = _field('complex')
    _backends(monkeypatch, h5netcdf=False,
              netcdf4_version=importlib.metadata.version('netCDF4'))
    field.to_netcdf(tmp_path / 'f.nc')
    _same(field, Field.from_netcdf(tmp_path / 'f.nc'))


@pytest.mark.skipif('netcdf4' not in ENGINES or 'h5netcdf' not in ENGINES,
                    reason='needs both backends to write the file first')
def test_reading_complex_values_with_no_backend_for_them_is_refused(
        tmp_path, monkeypatch):
    """A natively complex file read where only an older netCDF4 is
    installed would
    come back as (r, i) records; it is refused, naming the variable and both
    ways to get a backend."""
    from uacpy.core.exceptions import ConfigurationError
    # A file holding the complex values natively, as h5netcdf writes them.
    _field('complex').to_xarray().to_netcdf(
        tmp_path / 'f.nc', engine='h5netcdf', invalid_netcdf=True)
    _backends(monkeypatch, h5netcdf=False, netcdf4_version='1.6.5')
    with pytest.raises(ConfigurationError,
                       match=r"Field\.from_netcdf: .* holds complex values"
                       ) as e:
        Field.from_netcdf(tmp_path / 'f.nc')
    assert 'pip install h5netcdf' in str(e.value)


@pytest.mark.skipif('netcdf4' not in ENGINES, reason='netCDF4 not installed')
def test_without_h5netcdf_a_real_field_needs_nothing_more(tmp_path,
                                                         monkeypatch):
    """A real Field with boolean attributes writes and reads with neither
    h5netcdf nor a complex-capable netCDF4: the booleans are stored as
    int8."""
    field = _field('bool attrs')
    _backends(monkeypatch, h5netcdf=False, netcdf4_version='1.6.5')
    field.to_netcdf(tmp_path / 'f.nc')
    _same(field, Field.from_netcdf(tmp_path / 'f.nc'))
