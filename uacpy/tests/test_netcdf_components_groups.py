"""A result's components travel through NetCDF as groups: ``to_netcdf``
writes each one (and its own) below the result, and ``from_netcdf`` reads the
whole result back."""
import warnings

import numpy as np
import pytest

from uacpy.core.exceptions import FallbackWarning
from uacpy.core.results import (Arrivals, Covariance, Field,
                                ReflectionCoefficient, Result)
from uacpy.tests._synthetic_fields import _arrivals, _bounce, _broadband

xr = pytest.importorskip('xarray')
pytest.importorskip('h5netcdf')


def _same(a, b):
    """Two results hold the same arrays and identity (their ``to_dict``s
    agree entry for entry), components included."""
    assert type(a) is type(b)
    da, db = a.to_dict(), b.to_dict()
    da.pop('components', None)
    db.pop('components', None)
    assert da.keys() == db.keys()
    for key in da:
        if isinstance(da[key], np.ndarray):
            np.testing.assert_array_equal(da[key], db[key], err_msg=key)
    assert sorted(a.components) == sorted(b.components)
    for name in a.components:
        _same(a.components[name], b.components[name])


class TestComponentsTravelAsGroups:

    def test_a_field_and_its_nested_components_round_trip(self, tmp_path):
        """Bellhop's broadband field: 'arrivals', whose own 'bounce' sits
        one group deeper, and 'bounce' itself."""
        arrivals = _arrivals(components={'bounce': _bounce()})
        field = _broadband(components={'arrivals': arrivals,
                                       'bounce': _bounce()})
        with warnings.catch_warnings():
            warnings.simplefilter('error', FallbackWarning)
            field.to_netcdf(tmp_path / 'f.nc')
        back = Field.from_netcdf(tmp_path / 'f.nc')
        _same(field, back)
        assert isinstance(back.components['arrivals'], Arrivals)
        assert isinstance(back.components['arrivals'].components['bounce'],
                          ReflectionCoefficient)

    def test_the_groups_are_where_the_file_names_them(self, tmp_path):
        field = _broadband(components={
            'arrivals': _arrivals(components={'bounce': _bounce()})})
        field.to_netcdf(tmp_path / 'f.nc')
        with xr.open_dataset(tmp_path / 'f.nc', engine='h5netcdf',
                             group='components/arrivals/components/bounce') as ds:
            assert ds.attrs['uacpy_class'] == \
                'uacpy.core.results.ReflectionCoefficient'

    def test_a_dataset_root_carries_a_field_component(self, tmp_path):
        """OASS's covariance keeps the mean field it scattered."""
        cov = Covariance(covariance=np.eye(2, dtype=complex)[None] * 2.0,
                         frequencies=[50.0], model='OASS',
                         components={'mean_field': _broadband()})
        cov.to_netcdf(tmp_path / 'c.nc')
        _same(cov, Covariance.from_netcdf(tmp_path / 'c.nc'))

    def test_a_result_without_components_round_trips(self, tmp_path):
        field = _broadband()
        field.to_netcdf(tmp_path / 'f.nc')
        back = Field.from_netcdf(tmp_path / 'f.nc')
        _same(field, back)
        assert not back.components

    def test_the_file_attrs_do_not_land_in_metadata(self, tmp_path):
        field = _broadband(components={'bounce': _bounce()})
        field.to_netcdf(tmp_path / 'f.nc')
        with xr.open_dataarray(tmp_path / 'f.nc', engine='h5netcdf') as da:
            plain = Field.from_xarray(da.load())
        assert not {'uacpy_class', 'uacpy_components'} & set(plain.metadata)

    def test_to_xarray_alone_says_it_leaves_them_out(self):
        field = _broadband(components={'bounce': _bounce()})
        with pytest.warns(FallbackWarning, match='to_netcdf'):
            field.to_xarray()

    def test_a_root_of_another_class_is_refused(self, tmp_path):
        from uacpy.core.exceptions import ConfigurationError
        _broadband().to_netcdf(tmp_path / 'f.nc')
        with pytest.raises(ConfigurationError, match='is not a Covariance'):
            Covariance.from_netcdf(tmp_path / 'f.nc')

    def test_a_plain_xarray_file_reads_through_from_xarray(self, tmp_path):
        """A file written by to_xarray and xarray alone names no class; the
        caller's class reads it."""
        field = _broadband()
        field.to_xarray().to_netcdf(tmp_path / 'f.nc', engine='h5netcdf',
                                    invalid_netcdf=True)
        _same(field, Field.from_netcdf(tmp_path / 'f.nc', engine='h5netcdf'))

    def test_from_netcdf_is_on_every_result(self):
        assert Result.from_netcdf.__func__ is Field.from_netcdf.__func__
