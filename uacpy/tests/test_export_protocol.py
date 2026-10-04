"""The export protocol (``uacpy.core._export``): every result type declares its
arrays once and gets ``values``, ``to_dict``/``from_dict``,
``to_xarray``/``from_xarray``, ``to_netcdf`` and, for a table,
``to_dataframe`` from them."""

import numpy as np
import pytest

from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import (Arrivals, Covariance, Field, GreensFunction,
                                Modes, Rays, ReflectionCoefficient, Replicas)


def _arrivals():
    cell = {'amplitudes': np.array([1.0, 0.5]), 'phases': np.array([0.1, 0.2]),
            'delays': np.array([0.10, 0.12]),
            'delays_imag': np.array([0.0, -1e-6]),
            'source_angles': np.array([5.0, -7.0]),
            'receiver_angles': np.array([-5.0, 7.0]),
            'n_top_bounces': np.array([0, 1], dtype='int32'),
            'n_bot_bounces': np.array([1, 1], dtype='int32'), 'n_arrivals': 2}
    empty = {k: (v[:0] if isinstance(v, np.ndarray) else 0)
             for k, v in cell.items()}
    return Arrivals(by_receiver=[[[cell, empty], [empty, cell]]],
                    receiver_depths=[10.0, 20.0],
                    receiver_ranges=[100.0, 200.0], frequencies=1000.0,
                    source_depths=[5.0], model='Bellhop')


def _rays():
    return Rays(rays=[{'r': np.array([0.0, 3.0, 6.0]),
                       'z': np.array([5.0, 9.0, 5.0]), 'launch_angle': -10.0,
                       'n_top_bounces': 0, 'n_bot_bounces': 1},
                      {'r': np.array([0.0, 4.0]), 'z': np.array([5.0, 2.0]),
                       'launch_angle': 12.0, 'n_top_bounces': 1, 'n_bot_bounces': 0}],
                receiver_depths=[5.0], receiver_ranges=[6.0],
                frequencies=1000.0)


def _field():
    return Field(data=np.arange(6.0).reshape(2, 3) * (1 + 1j),
                 coords={'depth': np.array([10.0, 20.0]),
                         'range': np.array([100.0, 200.0, 300.0])},
                 frequencies=100.0, model='Kraken', metadata={'note': 'x'})


def _greens_function(snapshot):
    if snapshot:
        return GreensFunction(
            data=np.ones((3, 1, 2, 4), np.complex64) * (1 + 2j),
            phase_speeds=np.linspace(1700.0, 1400.0, 4),
            receiver_depths=np.array([10.0, 20.0], np.float32),
            source_depths=[5.0], frequencies=50.0, times=[0.0, 0.1, 0.2],
            title='SPARC- test', model='SPARC')
    return GreensFunction(
        data=np.ones((2, 1, 2, 4), np.complex64) * (1 + 2j),
        phase_speeds=np.linspace(1700.0, 1400.0, 4),
        receiver_depths=np.array([10.0, 20.0], np.float32),
        source_depths=[5.0], frequencies=[50.0, 60.0],
        stabilizing_attenuation=0.25, title='SCOOTER- test', model='Scooter')


MAKERS = {
    'Field': _field,
    'Arrivals': _arrivals,
    'Rays': _rays,
    'Modes': lambda: Modes(k=np.array([0.40 + 1e-5j, 0.39 + 2e-5j]),
                           phi=np.arange(6.0).reshape(3, 2),
                           depths=np.array([1.0, 2.0, 3.0]),
                           group_velocity=np.array([1490.0, 1480.0]),
                           frequencies=100.0, model='Kraken'),
    'ReflectionCoefficient': lambda: ReflectionCoefficient(
        angles=[10.0, 20.0, 30.0], magnitude=[[0.9, 0.8], [0.7, 0.6], [0.5, 0.4]],
        phase=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], frequencies=[50.0, 60.0],
        model='Bounce'),
    'Covariance': lambda: Covariance(
        covariance=np.eye(2, dtype=complex)[None] * 2.0,
        receiver_positions=[[0.0, 0.0, 10.0], [0.0, 0.0, 20.0]],
        frequencies=[50.0], model='OASN', unit='Pa^2/Hz'),
    'Replicas': lambda: Replicas(
        replicas=np.ones((1, 2, 3, 2), complex),
        candidates={'depth': np.array([10.0, 20.0]),
                    'range': np.array([1.0, 2.0, 3.0])},
        receiver_positions=[[0.0, 0.0, 10.0], [0.0, 0.0, 20.0]],
        frequencies=[50.0], model='OASN'),
    'GreensFunction': lambda: _greens_function(False),
    'GreensFunction snapshot': lambda: _greens_function(True),
}


def _same(a, b):
    """Two results hold the same arrays and identity: their ``to_dict``s
    agree entry for entry, arrays by value and dtype."""
    da, db = a.to_dict(), b.to_dict()
    assert da.keys() == db.keys()
    for key in da:
        x, y = da[key], db[key]
        if isinstance(x, np.ndarray):
            np.testing.assert_array_equal(x, y, err_msg=key)
            assert np.asarray(x).dtype == np.asarray(y).dtype, key
        elif isinstance(x, dict):
            assert x.keys() == y.keys(), key


@pytest.mark.parametrize('name', MAKERS)
class TestEveryResultRoundTrips:

    def test_from_dict_of_to_dict_is_the_result(self, name):
        x = MAKERS[name]()
        _same(x, type(x).from_dict(x.to_dict()))

    def test_the_npz_round_trip_is_the_result(self, name, tmp_path):
        x = MAKERS[name]()
        np.savez(tmp_path / 'x.npz', **x.to_dict())
        with np.load(tmp_path / 'x.npz', allow_pickle=True) as f:
            _same(x, type(x).from_dict(dict(f)))

    def test_from_xarray_of_to_xarray_is_the_result(self, name):
        pytest.importorskip('xarray')
        x = MAKERS[name]()
        _same(x, type(x).from_xarray(x.to_xarray()))

    def test_the_netcdf_round_trip_is_the_result(self, name, tmp_path):
        xr = pytest.importorskip('xarray')
        pytest.importorskip('h5netcdf')
        x = MAKERS[name]()
        x.to_netcdf(tmp_path / 'x.nc')
        opener = (xr.open_dataarray if isinstance(x, Field)
                  else xr.open_dataset)
        with opener(tmp_path / 'x.nc', engine='h5netcdf') as obj:
            _same(x, type(x).from_xarray(obj.load()))

    def test_values_is_a_read_only_view_of_the_payload(self, name):
        x = MAKERS[name]()
        view = x.values()
        assert not view.flags.writeable
        with pytest.raises(ValueError, match='read-only'):
            view.flat[0] = 0


class TestUnitsAreWrittenAsCfAttributes:

    def test_every_payload_and_coordinate_states_its_unit(self):
        pytest.importorskip('xarray')
        ds = MAKERS['Arrivals']().to_xarray()
        assert ds['delays'].attrs['units'] == 's'
        assert ds['phases'].attrs['units'] == 'rad'
        assert ds['source_angles'].attrs['units'] == 'deg'
        assert ds['depth'].attrs['units'] == 'm'

    def test_a_unitless_column_carries_no_units_attribute(self):
        pytest.importorskip('xarray')
        ds = MAKERS['Arrivals']().to_xarray()
        assert 'units' not in ds['amplitudes'].attrs

    def test_modes_state_the_wavenumber_and_group_speed_units(self):
        pytest.importorskip('xarray')
        ds = MAKERS['Modes']().to_xarray()
        assert (ds['k'].attrs['units'], ds['group_velocity'].attrs['units']) \
            == ('rad/m', 'm/s')


class TestTablesAreDataFrames:

    @pytest.mark.parametrize('name, rows', [('Arrivals', 4), ('Rays', 2),
                                            ('Modes', 2),
                                            ('ReflectionCoefficient', 6)])
    def test_one_row_per_record(self, name, rows):
        pytest.importorskip('pandas')
        assert len(MAKERS[name]().to_dataframe()) == rows

    @pytest.mark.parametrize('name', ['Field', 'Covariance', 'Replicas',
                                      'GreensFunction'])
    def test_a_gridded_type_names_its_labelled_form(self, name):
        with pytest.raises(ConfigurationError, match='gridded'):
            MAKERS[name]().to_dataframe()

    def test_the_arrival_table_places_each_arrival_at_its_receiver(self):
        pytest.importorskip('pandas')
        df = MAKERS['Arrivals']().to_dataframe()
        assert list(df['receiver_depth']) == [10.0, 10.0, 20.0, 20.0]
        assert list(df['receiver_range']) == [100.0, 100.0, 200.0, 200.0]
        assert list(df['kinds']) == ['bottom', 'both', 'bottom', 'both']

    def test_the_mode_table_turns_im_k_into_a_decay_rate(self):
        pytest.importorskip('pandas')
        df = MAKERS['Modes']().to_dataframe()
        # 20 log10(e) dB per neper, 1000 m per km.
        np.testing.assert_allclose(df['attenuation_dB_per_km'],
                                   [8.685889638e-2, 1.737177928e-1])
        np.testing.assert_allclose(df['phase_speed'],
                                   2 * np.pi * 100.0 / np.array([0.40, 0.39]))

    def test_the_ray_table_measures_each_path(self):
        pytest.importorskip('pandas')
        df = MAKERS['Rays']().to_dataframe()
        np.testing.assert_allclose(df['length'], [10.0, 5.0])
        assert list(df['n_vertices']) == [3, 2]
        assert np.isnan(df['miss_distance']).all()


class TestArrivalsKeepOneTable:

    def test_by_receiver_is_regrouped_from_the_table_cell_for_cell(self):
        a = MAKERS['Arrivals']()
        cell = a.by_receiver[0][1][1]
        assert cell['n_arrivals'] == 2
        np.testing.assert_array_equal(cell['delays'], [0.10, 0.12])
        assert cell['n_top_bounces'].dtype == np.int32
        empty = a.by_receiver[0][0][1]
        assert empty['n_arrivals'] == 0 and empty['delays'].size == 0

    def test_a_filter_shrinks_the_records_and_the_cells_together(self):
        a = MAKERS['Arrivals']().filter_by_bounces('both')
        assert len(a) == 2
        assert sum(c['n_arrivals'] for row in a.by_receiver[0]
                   for c in row) == 2

    @pytest.mark.parametrize('edit', [
        lambda a: a.arrivals.__setitem__(0, {}),
        lambda a: a.arrivals.append({}),
        lambda a: a.arrivals[0].__setitem__('delay', 99.0),
        lambda a: a.by_receiver.__setitem__(0, []),
        lambda a: a.by_receiver[0][1].append({}),
        lambda a: a.by_receiver[0][1][1].__setitem__('n_arrivals', 0),
    ])
    def test_an_edit_of_a_derived_view_is_refused(self, edit):
        a = MAKERS['Arrivals']()
        with pytest.raises(TypeError, match='read-only view'):
            edit(a)
        assert a.delays[0] == 0.10 and len(a) == 4

    def test_a_cell_array_is_read_only(self):
        cell = MAKERS['Arrivals']().by_receiver[0][1][1]
        with pytest.raises(ValueError, match='read-only'):
            cell['delays'][0] = 0.0

    def test_a_copy_of_a_view_is_editable(self):
        import copy
        records = copy.copy(MAKERS['Arrivals']().arrivals)
        records.append({})
        assert type(records) is list

    def test_the_bulk_columns_are_read_only(self):
        a = MAKERS['Arrivals']()
        for name in ('delays', 'amplitudes', 'n_top_bounces', 'delays_imag',
                     'receiver_depth', 'receiver_range'):
            assert not getattr(a, name).flags.writeable, name

    def test_trimming_ranges_renumbers_the_kept_cells(self):
        # Bellhop trims the range columns it padded: what is kept must be
        # the cells lo..hi-1 of the grid, under indices that start at 0.
        a = MAKERS['Arrivals']()
        before = a.by_receiver
        a._trim_ranges(1, 2)
        assert list(a.receiver_ranges) == [200.0]
        after = a.by_receiver
        assert len(after[0][0]) == 1
        for d in range(2):
            np.testing.assert_array_equal(after[0][d][0]['delays'],
                                          before[0][d][1]['delays'])
        assert set(np.asarray(a.to_dict()['range_idx'])) == {0}

    def test_an_irregular_grid_pairs_the_depth_with_the_range(self):
        cell = MAKERS['Arrivals']().by_receiver[0][1][1]
        a = Arrivals(by_receiver=[[[cell, cell]]],
                     receiver_depths=[10.0, 20.0],
                     receiver_ranges=[100.0, 200.0])
        assert list(a.receiver_depth) == [10.0, 10.0, 20.0, 20.0]


class TestFieldAxesAndViews:

    @pytest.mark.parametrize('accessor, axis, first', [
        ('depths', 'depth', 10.0), ('ranges', 'range', 100.0)])
    def test_an_axis_accessor_cannot_change_the_field(self, accessor, axis,
                                                      first):
        f = _field()
        with pytest.raises(ValueError, match='read-only'):
            getattr(f, accessor)[0] = -1.0
        assert f.coords[axis][0] == first

    def test_the_level_view_is_minus_the_db_view(self):
        f = _field()
        np.testing.assert_array_equal(f.view('level'), -f.view('dB'))

    @pytest.mark.parametrize('value, expected', [
        ('magnitude', np.abs), ('phase', np.angle),
        ('real', np.real), ('imag', np.imag)])
    def test_each_view_is_its_function_of_the_data(self, value, expected):
        f = _field()
        np.testing.assert_array_equal(f.view(value), expected(f.data))

    def test_an_unknown_view_is_refused(self):
        with pytest.raises(ConfigurationError, match='not a view'):
            _field().view('mag_dB')

    def test_a_complex_view_of_real_data_is_refused(self):
        f = Field(data=np.ones((2, 2)), coords={'depth': [1.0, 2.0],
                                               'range': [1.0, 2.0]},
                  unit='dB')
        with pytest.raises(ConfigurationError, match='requires complex'):
            f.view('level')


def test_a_greens_function_npz_holding_values_loads_its_payload(tmp_path):
    gf = _greens_function(False)
    d = gf.to_dict()
    d['values'] = d.pop('data')
    np.savez(tmp_path / 'g.npz', **d)
    with np.load(tmp_path / 'g.npz', allow_pickle=True) as f:
        back = GreensFunction.from_dict(dict(f))
    np.testing.assert_array_equal(back.data, gf.data)


# ── carriers ───────────────────────────────────────────────────────────────


def _environment():
    import uacpy
    from uacpy.data.sources import SOURCES, DataProvenance
    gebco = DataProvenance(source=SOURCES['gebco'], data_date='2024',
                           data_point=(43.01, 7.02),
                           requested_point=(43.0, 7.0))
    return uacpy.Environment(
        name='saved', date='2024-05-01', location=(43.0, 7.0),
        bathymetry=uacpy.Bathymetry(ranges=[0.0, 1000.0, 2000.0],
                                    depths=[100.0, 120.0, 90.0],
                                    data_sources=[gebco]),
        ssp=uacpy.SoundSpeedProfile(
            depths=[0.0, 50.0, 120.0],
            sound_speed=[[1500.0, 1501.0], [1490.0, 1491.0],
                         [1480.0, 1481.0]], ranges=[0.0, 2000.0]),
        altimetry=uacpy.Altimetry(ranges=[0.0, 2000.0], heights=[0.0, 1.0]),
        bottom=uacpy.Bottom(columns=[uacpy.SeabedColumn(
            layers=[uacpy.SedimentLayer(thickness=10.0, sound_speed=1600.0,
                                        density=1.6, attenuation=0.3)],
            halfspace=uacpy.BoundaryProperties(
                sound_speed=1800.0, density=2.0, attenuation=0.2,
                shear_speed=400.0))]),
        absorption=uacpy.FrancoisGarrison(temperature=10.0,
                                          salinity=35.0, pH=8.0))


def _plain(d):
    """A to_dict as comparable plain values: arrays as (dtype, list)."""
    if isinstance(d, dict):
        return {k: _plain(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_plain(v) for v in d]
    if isinstance(d, np.ndarray):
        return (d.dtype.str, d.tolist())
    return d


class TestAnEnvironmentSavesAndLoads:

    def test_the_npz_round_trip_is_the_environment(self, tmp_path):
        import uacpy
        env = _environment()
        np.savez(tmp_path / 'env.npz', **env.to_dict())
        with np.load(tmp_path / 'env.npz', allow_pickle=True) as f:
            back = uacpy.Environment.from_dict(dict(f))
        assert _plain(back.to_dict()) == _plain(env.to_dict())

    def test_the_netcdf_holds_one_group_per_carrier(self, tmp_path):
        xr = pytest.importorskip('xarray')
        pytest.importorskip('h5netcdf')
        import uacpy
        env = _environment()
        env.to_netcdf(tmp_path / 'env.nc', engine='h5netcdf')
        with xr.open_dataset(tmp_path / 'env.nc', group='ssp',
                             engine='h5netcdf') as ssp:
            assert ssp['sound_speed'].dims == ('depth', 'range')
            assert ssp['sound_speed'].attrs['units'] == 'm/s'
        back = uacpy.Environment.from_netcdf(tmp_path / 'env.nc',
                                             engine='h5netcdf')
        assert _plain(back.to_dict()) == _plain(env.to_dict())

    def test_the_provenance_is_saved_by_source_id_and_restored(self):
        import uacpy
        env = _environment()
        saved = env.to_dict()['bathymetry']['data_sources'][0]
        assert saved['__provenance__'] == 'gebco'
        back = uacpy.Environment.from_dict(env.to_dict())
        record = back.bathymetry.data_sources[0]
        assert record.source.id == 'gebco'
        assert record.data_point == (43.01, 7.02)
        # The catalogue record itself, the object a fetch attaches.
        from uacpy.data.sources import SOURCES
        assert record.source is SOURCES['gebco']

    def test_a_class_path_outside_the_package_is_refused(self):
        import uacpy
        d = _environment().to_dict()
        d['ssp'] = dict(d['ssp'], __class__='os.system')
        with pytest.raises(ConfigurationError, match='not a uacpy class'):
            uacpy.Environment.from_dict(d)

    def test_a_vacuum_surface_round_trips(self):
        import uacpy
        env = _environment()
        assert env.surface.acoustic_type == 'vacuum'
        back = uacpy.Environment.from_dict(env.to_dict())
        assert back.surface.acoustic_type == 'vacuum'


class TestGriddedCarriersHaveLabelledForms:

    @pytest.mark.parametrize('name, dims, unit', [
        ('ssp', {'sound_speed': ('depth', 'range')}, 'm/s'),
        ('bathymetry', {'depth': ('range',)}, 'm'),
        ('altimetry', {'height': ('range',)}, 'm'),
    ])
    def test_the_values_sit_on_their_axes(self, name, dims, unit):
        pytest.importorskip('xarray')
        carrier = getattr(_environment(), name)
        ds = carrier.to_xarray()
        ((var, want),) = dims.items()
        assert ds[var].dims == want
        assert ds[var].attrs['units'] == unit
        back = type(carrier).from_xarray(ds)
        assert _plain(back.to_dict()) == _plain(carrier.to_dict())

    def test_an_absorption_coefficient_round_trips(self):
        pytest.importorskip('xarray')
        import uacpy
        a = uacpy.Thorp().table([1000.0, 2000.0])
        ds = a.to_xarray()
        assert (ds['alpha'].attrs['units'], ds['frequency'].attrs['units']) \
            == ('dB/km', 'Hz')
        back = type(a).from_xarray(ds)
        np.testing.assert_array_equal(back.data, a.data)
        assert not a.values().flags.writeable


class TestSeabedAndSurfaceTables:

    def test_a_layered_column_lists_its_layers_then_the_halfspace(self):
        pytest.importorskip('pandas')
        df = _environment().bottom.to_dataframe()
        assert list(df['acoustic_type']) == ['layer', 'half-space']
        assert list(df['top']) == [0.0, 10.0]
        assert list(df['bottom']) == [10.0, np.inf]
        assert list(df['sound_speed']) == [1600.0, 1800.0]

    def test_a_vacuum_surface_carries_no_acoustic_parameter(self):
        pytest.importorskip('pandas')
        df = _environment().surface.to_dataframe()
        assert df['acoustic_type'][0] == 'vacuum'
        assert np.isnan(df['sound_speed'][0])
        assert df['roughness'][0] == 0.0


def test_a_receiver_grid_is_depth_first():
    import uacpy
    Z, R = uacpy.Receiver(depths=[1.0, 2.0], ranges=[10.0, 20.0, 30.0]).grid()
    assert Z.shape == R.shape == (2, 3)
    assert Z[1, 0] == 2.0 and R[0, 2] == 30.0


def test_a_parallel_result_summarises_its_jobs():
    pytest.importorskip('pandas')
    from uacpy.parallel import ParallelResult
    pr = ParallelResult(results=[object(), None], errors={1: ValueError('x')},
                        labels=['a', 'b'], warnings={0: [(UserWarning, 'w')]})
    df = pr.summary()
    assert list(df['ok']) == [True, False]
    assert list(df['error_type']) == ['', 'ValueError']
    assert list(df['n_warnings']) == [1, 0]


# ── signal results ─────────────────────────────────────────────────────────


class TestSignalResultsExport:

    @staticmethod
    def _x():
        return np.random.default_rng(0).standard_normal(4096)

    def test_a_spectral_estimate_round_trips_with_its_scaling(self):
        pytest.importorskip('xarray')
        from uacpy.acoustic_signal import welch
        w = welch(self._x(), 1000.0, nperseg=256)
        ds = w.to_xarray()
        assert ds['power'].dims == ('frequencies',)
        assert ds['frequencies'].attrs['units'] == 'Hz'
        back = type(w).from_xarray(ds)
        np.testing.assert_array_equal(back.power, w.power)
        assert back.scaling == w.scaling

    def test_a_spectrogram_panel_is_labelled_by_its_two_axes(self):
        pytest.importorskip('xarray')
        from uacpy.acoustic_signal import spectrogram
        sg = spectrogram(self._x(), 1000.0, nperseg=256)
        assert sg.to_xarray()['power'].dims == ('frequencies', 'times')

    @pytest.mark.parametrize('estimator', ['welch', 'spectrogram',
                                           'probabilistic_welch'])
    def test_an_axis_field_is_the_coordinate_of_its_dimension(self,
                                                              estimator):
        pytest.importorskip('xarray')
        from uacpy import acoustic_signal
        x = np.tile(self._x(), 4)
        kw = ({'segment_duration': 0.5}
              if estimator == 'probabilistic_welch' else {})
        r = getattr(acoustic_signal, estimator)(x, 1000.0, nperseg=256, **kw)
        ds = r.to_xarray()
        assert ds['frequencies'].dims == ('frequencies',)
        assert 'frequencies_dim0' not in ds.dims
        value = 'pdf' if estimator == 'probabilistic_welch' else 'power'
        near = ds[value].sel(frequencies=251.0, method='nearest')
        assert float(near['frequencies']) == pytest.approx(250.0)

    def test_a_one_dimensional_field_labelling_no_axis_stays_a_variable(self):
        pytest.importorskip('xarray')
        from uacpy.acoustic_signal import probabilistic_welch
        pw = probabilistic_welch(np.tile(self._x(), 4), 1000.0, nperseg=256,
                                 segment_duration=0.5)
        ds = pw.to_xarray()
        assert 'level_edges' in ds.data_vars
        assert ds['level_edges'].dims == ('level_edges_dim0',)

    def test_the_dict_round_trip_keeps_the_attributes(self, tmp_path):
        # Attributes other than the defaults, so a round trip that dropped
        # them would not come back equal by accident.
        from uacpy.acoustic_signal import spectrogram
        sg = spectrogram(self._x(), 1000.0, nperseg=256, scaling='spectrum',
                         mode='magnitude')
        np.savez(tmp_path / 's.npz', **sg.to_dict())
        with np.load(tmp_path / 's.npz', allow_pickle=True) as f:
            back = type(sg).from_dict(dict(f))
        np.testing.assert_array_equal(back.power, sg.power)
        assert (back.scaling, back.mode) == (sg.scaling, sg.mode)

    def test_equal_length_fields_are_a_table(self):
        pytest.importorskip('pandas')
        from uacpy.acoustic_signal import welch
        w = welch(self._x(), 1000.0, nperseg=256)
        assert list(w.to_dataframe().columns) == ['frequencies', 'power']


def test_a_source_with_complex_weights_round_trips_through_xarray():
    pytest.importorskip('xarray')
    import uacpy
    s = uacpy.Source(depths=[10.0, 20.0], frequencies=[100.0, 200.0],
                     weights=[1.0, 0.5j])
    back = uacpy.Source.from_xarray(s.to_xarray())
    np.testing.assert_array_equal(back.weights, s.weights)
    assert back.weights.dtype == s.weights.dtype


# ── components and the provenance catalogue ────────────────────────────────


class TestComponentsTravelWithTheirResult:

    @staticmethod
    def _with_components():
        rc = MAKERS['ReflectionCoefficient']()
        arrivals = MAKERS['Arrivals']()
        f = _field()
        return Field(data=f.data, coords=f.coords, frequencies=100.0,
                     components={'arrivals': arrivals, 'bounce': rc})

    def test_to_dict_nests_each_component_and_from_dict_rebuilds_it(
            self, tmp_path):
        f = self._with_components()
        np.savez(tmp_path / 'f.npz', **f.to_dict())
        with np.load(tmp_path / 'f.npz', allow_pickle=True) as saved:
            back = Field.from_dict(dict(saved))
        assert set(back.components) == {'arrivals', 'bounce'}
        assert type(back.components['bounce']) is ReflectionCoefficient
        # Equal, not the same object: a file holds values, not identity.
        _same(back.components['bounce'], f.components['bounce'])
        _same(back.components['arrivals'], f.components['arrivals'])

    def test_a_result_without_components_writes_no_key(self):
        assert 'components' not in _field().to_dict()

    def test_xarray_says_it_leaves_the_components_out(self):
        pytest.importorskip('xarray')
        from uacpy.core.exceptions import FallbackWarning
        with pytest.warns(FallbackWarning, match=r"\['arrivals', 'bounce'\]"):
            self._with_components().to_xarray()


def test_the_data_layer_hands_out_the_core_catalogue():
    from uacpy.core import provenance
    from uacpy.data import sources
    assert sources.SOURCES is provenance.SOURCES
    assert sources.DataProvenance is provenance.DataProvenance
    assert sources.DataSource is provenance.DataSource


class TestTheShadeFileRecordExports:
    """``ShdFile`` (what ``read_shd_bin`` / ``read_shd_asc`` return) carries
    the export protocol of the records: read-only arrays, a dict and an
    xarray form that rebuild it."""

    @staticmethod
    def _record():
        from uacpy.io import ShdFile
        pressure = np.array([[[[1 + 2j, np.nan], [3 - 1j, 0.5j]]]])
        return ShdFile(
            title='run', plot_type='rectilin  ',
            frequencies=np.array([100.0, 200.0]), source_frequency=100.0,
            stabilizing_attenuation=0.25, bearings=np.array([0.0]),
            source_x=np.array([0.0]), source_y=np.array([0.0]),
            source_depths=np.array([50.0]),
            receiver_depths=np.array([10.0, 20.0]),
            receiver_ranges=np.array([100.0, 200.0]), pressure=pressure,
            pressure_frequency=200.0)

    @staticmethod
    def _assert_same(a, b):
        import dataclasses
        for field in dataclasses.fields(a):
            x, y = getattr(a, field.name), getattr(b, field.name)
            if isinstance(x, np.ndarray):
                np.testing.assert_array_equal(x, y)
            else:
                assert x == y, field.name

    def test_the_arrays_refuse_edits(self):
        shd = self._record()
        with pytest.raises(ValueError, match='read-only'):
            shd.pressure[0, 0, 0, 0] = 0.0
        with pytest.raises(ValueError, match='read-only'):
            shd.frequencies[0] = 0.0

    def test_a_pickled_record_keeps_its_arrays_read_only(self):
        import pickle
        back = pickle.loads(pickle.dumps(self._record()))
        assert not back.pressure.flags.writeable
        self._assert_same(back, self._record())

    def test_the_dict_round_trips_through_npz(self, tmp_path):
        from uacpy.io import ShdFile
        path = tmp_path / 'shd.npz'
        np.savez(path, **self._record().to_dict())
        back = ShdFile.from_dict(dict(np.load(path, allow_pickle=True)))
        self._assert_same(back, self._record())

    def test_the_xarray_form_labels_the_pressure_and_round_trips(self):
        pytest.importorskip('xarray')
        from uacpy.io import ShdFile
        ds = self._record().to_xarray()
        assert ds['pressure'].dims == ('bearing', 'source_depth', 'row',
                                       'range')
        assert ds['range'].attrs['units'] == 'm'
        assert ds['bearing'].attrs['units'] == 'deg'
        self._assert_same(ShdFile.from_xarray(ds), self._record())

    def test_a_shade_file_is_not_a_table(self):
        with pytest.raises(ConfigurationError, match='gridded'):
            self._record().to_dataframe()



class TestTheTimeSeriesFileRecordExports:
    """``RtsFile`` (what ``read_rts_file`` / ``read_ts`` return): the
    pressure on its time and position axes, read-only, with a dict and an
    xarray form that rebuild it."""

    @staticmethod
    def _record():
        from uacpy.io import RtsFile
        return RtsFile(title='run', positions=np.array([1000.0, 2000.0]),
                       times=np.array([0.0, 0.01, 0.02]),
                       pressure=np.arange(6.0).reshape(3, 2))

    def test_the_arrays_refuse_edits(self):
        with pytest.raises(ValueError, match='read-only'):
            self._record().pressure[0, 0] = 1.0

    def test_the_dict_round_trips_through_npz(self, tmp_path):
        from uacpy.io import RtsFile
        path = tmp_path / 'rts.npz'
        np.savez(path, **self._record().to_dict())
        back = RtsFile.from_dict(dict(np.load(path, allow_pickle=True)))
        assert back.title == 'run'
        np.testing.assert_array_equal(back.pressure, self._record().pressure)
        np.testing.assert_array_equal(back.times, self._record().times)

    def test_the_xarray_form_puts_the_pressure_on_time_and_position(self):
        pytest.importorskip('xarray')
        from uacpy.io import RtsFile
        ds = self._record().to_xarray()
        assert ds['pressure'].dims == ('time', 'position')
        assert ds['time'].attrs['units'] == 's'
        assert ds['position'].attrs['units'] == 'm'
        back = RtsFile.from_xarray(ds)
        np.testing.assert_array_equal(back.pressure, self._record().pressure)
        np.testing.assert_array_equal(back.positions,
                                      self._record().positions)
        assert back.title == 'run'


def _record_classes():
    import uacpy.data  # noqa: F401  (registers the data records)
    import uacpy.io  # noqa: F401  (registers the io records)
    from uacpy.core._export import ExportRecord
    found, todo = [], list(ExportRecord.__subclasses__())
    while todo:
        cls = todo.pop()
        found.append(cls)
        todo.extend(cls.__subclasses__())
    return sorted(found, key=lambda c: c.__name__)


class TestEveryRecordFreezesEveryArrayField:
    """A record's array fields are read-only because it names them in
    ``_ARRAY_FIELDS``; a field typed as an array and left off that list
    would hand out a writeable array."""

    def test_the_sweep_finds_the_records(self):
        names = {cls.__name__ for cls in _record_classes()}
        assert {'ArgoProfile', 'RtsFile', 'ShdFile', 'Ssp3dFile'} <= names

    @pytest.mark.parametrize('cls', _record_classes(),
                             ids=lambda c: c.__name__)
    def test_no_field_shadows_a_protocol_name(self, cls):
        """A field named after an inherited method (``values``) takes the
        method as its dataclass default and hides it on every instance."""
        import dataclasses
        from uacpy.core._export import CarrierExport
        protocol = {name for name in dir(CarrierExport)
                    if not name.startswith('__')}
        assert not {f.name for f in dataclasses.fields(cls)} & protocol

    @pytest.mark.parametrize('cls', _record_classes(),
                             ids=lambda c: c.__name__)
    def test_the_module_holds_the_class_itself(self, cls):
        """A decorator left above a reader whose record class was
        inserted between the two would wrap the class instead."""
        import sys
        assert getattr(sys.modules[cls.__module__], cls.__name__) is cls

    @pytest.mark.parametrize('cls', _record_classes(),
                             ids=lambda c: c.__name__)
    def test_every_array_typed_field_is_declared(self, cls):
        import dataclasses
        import typing
        hints = typing.get_type_hints(cls)
        typed = {f.name for f in dataclasses.fields(cls)
                 if np.ndarray in (hints[f.name],
                                   *typing.get_args(hints[f.name]))}
        assert typed == set(cls._ARRAY_FIELDS)


class TestPerLaunchSettingsAreTables:
    """``run_settings(...).engine.to_dataframe()`` gives one row per binary
    launch, one column per field of the launch record; an engine with no
    per-launch records refuses."""

    @staticmethod
    def _rig():
        import uacpy
        env = uacpy.Environment(bathymetry=100.0, ssp=1500.0, bottom='sand')
        src = uacpy.Source(depths=30.0, frequencies=[100.0, 200.0])
        rcv = uacpy.Receiver(depths=[20.0, 50.0],
                             ranges=np.linspace(500.0, 3000.0, 6))
        return env, src, rcv

    def test_ram_grids_are_one_row_per_launch(self):
        pytest.importorskip('pandas')
        import dataclasses
        import uacpy
        engine = uacpy.RAM().run_settings(
            *self._rig(), run_mode=uacpy.RunMode.BROADBAND).engine
        frame = engine.to_dataframe()
        assert list(frame.columns) == [f.name for f in
                                       dataclasses.fields(engine.grids[0])]
        assert len(frame) == len(engine.grids)
        assert frame['frequency'].tolist() == [g.frequency
                                               for g in engine.grids]
        assert frame['dz'].tolist() == [g.dz for g in engine.grids]

    def test_kraken_launches_keep_a_tuple_in_one_cell(self):
        pytest.importorskip('pandas')
        import uacpy
        engine = uacpy.Kraken().run_settings(
            *self._rig(), run_mode=uacpy.RunMode.BROADBAND).engine
        frame = engine.to_dataframe()
        assert len(frame) == len(engine.launches)
        assert frame['c_high'].tolist() == [l.c_high
                                            for l in engine.launches]
        assert frame['deck_frequency'].tolist() == [
            l.deck_frequency for l in engine.launches]

    def test_an_engine_without_launch_records_refuses(self):
        import uacpy
        env, _, rcv = self._rig()
        engine = uacpy.Bellhop().run_settings(
            env, uacpy.Source(depths=30.0, frequencies=100.0), rcv).engine
        with pytest.raises(ConfigurationError, match='no per-launch'):
            engine.to_dataframe()

