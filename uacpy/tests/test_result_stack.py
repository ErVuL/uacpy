"""``ResultStack``: the sparse results of one run (rays, arrivals), one per
value of a coordinate.

Its invariants (a Field is never a slab), the lookups, finite labels, the
identity it forwards, and the result export protocol it shares with every
other result.
"""

import numpy as np
import pytest
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import Arrivals, Field, Rays, Result
from uacpy.core.results.stack import ResultStack


def _rays(*, source_depth=50.0, frequencies=100.0, model='Test',
          model_source=None, n=2):
    return Rays(rays=[{'r': np.array([0.0, 3.0, 6.0]),
                       'z': np.array([source_depth, 9.0, 5.0]),
                       'launch_angle': -10.0 + i,
                       'n_top_bounces': 0, 'n_bot_bounces': 1}
                      for i in range(n)],
                receiver_depths=[5.0], receiver_ranges=[6.0],
                frequencies=frequencies, model=model,
                source_depths=np.array([float(source_depth)]),
                model_source=model_source)


def _arrivals(source_depth=5.0, amplitude=1.0):
    cell = {'amplitudes': np.array([amplitude, 0.5]),
            'phases': np.array([0.1, 0.2]),
            'delays': np.array([0.10, 0.12]),
            'delays_imag': np.array([0.0, -1e-6]),
            'source_angles': np.array([5.0, -7.0]),
            'receiver_angles': np.array([-5.0, 7.0]),
            'n_top_bounces': np.array([0, 1], dtype='int32'),
            'n_bot_bounces': np.array([1, 1], dtype='int32'), 'n_arrivals': 2}
    return Arrivals(by_receiver=[[[cell]]], receiver_depths=[10.0],
                    receiver_ranges=[100.0], frequencies=1000.0,
                    source_depths=[source_depth], model='Bellhop')


def _field(source_depth):
    return Field(data=np.ones((2, 3), dtype=complex),
                 coords={'depth': np.arange(2.0),
                         'range': np.arange(3.0) * 100.0},
                 model='Test', frequencies=100.0,
                 source_depths=np.array([source_depth]))


class TestResultStackInvariants:
    """The constructor enforces uniform slab type, uniform model / backend /
    frequencies, and ``len(slabs) == len(coordinate)``, so the stack's
    read-through properties never silently disagree with a slab."""

    def test_requires_at_least_one_slab(self):
        with pytest.raises(ConfigurationError, match="at least one slab"):
            ResultStack(slabs=[], coordinate=[])

    def test_rejects_length_mismatch(self):
        with pytest.raises(ConfigurationError, match="coordinate length"):
            ResultStack(slabs=[_rays(source_depth=10.0)],
                        coordinate=[10.0, 20.0])

    def test_rejects_mixed_slab_types(self):
        with pytest.raises(ConfigurationError, match="same concrete type"):
            ResultStack(slabs=[_rays(source_depth=10.0), _arrivals(20.0)],
                        coordinate=[10.0, 20.0])

    def test_a_field_is_never_a_slab(self):
        """The fields of several sources are one Field over a leading axis
        (``Field.superpose`` sums it); the refusal names the join that
        builds it, stack_fields."""
        with pytest.raises(ConfigurationError,
                           match='a Field is not a slab') as ei:
            ResultStack([_field(10.0), _field(20.0)], [10.0, 20.0])
        assert 'stack_fields(fields, depths)' in str(ei.value)
        assert 'Field.superpose' in str(ei.value)

    def test_a_slab_must_be_a_result(self):
        with pytest.raises(ConfigurationError, match='must be a Result'):
            ResultStack([object(), object()], [1.0, 2.0])

    def test_rejects_disagreeing_frequencies(self):
        a = _rays(source_depth=10.0, frequencies=100.0)
        b = _rays(source_depth=20.0, frequencies=200.0)
        with pytest.raises(ConfigurationError, match="frequencies"):
            ResultStack(slabs=[a, b], coordinate=[10.0, 20.0])

    def test_a_source_depth_sweep_on_another_axis_is_told_the_axis_that_admits_it(
            self):
        a = _rays(source_depth=10.0)
        b = _rays(source_depth=20.0)
        with pytest.raises(
                ConfigurationError,
                match='every slab must share the same source_depths') as ei:
            ResultStack(slabs=[a, b], coordinate=[0, 1], coordinate_name='case')
        assert "coordinate_name='source_depth'" in str(ei.value)
        assert 'Source(depths=[...])' in str(ei.value)

    def test_a_disagreeing_model_names_no_stacking_axis(self):
        a = _rays(source_depth=10.0, model='Bellhop')
        b = _rays(source_depth=10.0, model='Kraken')
        with pytest.raises(ConfigurationError,
                           match='every slab must share the same model') as ei:
            ResultStack(slabs=[a, b], coordinate=[0, 1], coordinate_name='case')
        assert 'coordinate_name=' not in str(ei.value)

    def test_accepts_uniform_slabs(self):
        a = _rays(source_depth=10.0)
        b = _rays(source_depth=20.0)
        stack = ResultStack(slabs=[a, b], coordinate=[10.0, 20.0])
        assert stack.slab_type is Rays
        assert stack.coordinate_name == 'source_depth'
        assert stack.n_slabs == 2
        assert len(stack) == 2
        assert stack.model == 'Test'
        np.testing.assert_array_equal(
            stack.coordinate, np.array([10.0, 20.0]))

    def test_iteration_and_label_select_share_slab_identity(self):
        a = _rays(source_depth=10.0)
        b = _rays(source_depth=20.0)
        stack = ResultStack(slabs=[a, b], coordinate=[10.0, 20.0])
        assert stack[0] is a
        assert stack[1] is b
        assert stack.at(source_depth=20.0) is b
        assert stack.isel(source_depth=1) is b
        assert list(stack) == [(10.0, a), (20.0, b)]

    def test_frequency_axis_stack(self):
        """Stacking along ``frequency`` is a coordinate-name swap: the slabs
        differ on ``frequencies`` and share ``source_depths``."""
        a = _rays(source_depth=50.0, frequencies=100.0)
        b = _rays(source_depth=50.0, frequencies=200.0)
        stack = ResultStack(slabs=[a, b], coordinate=[100.0, 200.0],
                            coordinate_name='frequency')
        assert stack.at(frequency=200.0) is b
        np.testing.assert_array_equal(stack.frequencies, [100.0, 200.0])
        np.testing.assert_array_equal(stack.source_depths, [50.0])
        with pytest.raises(ConfigurationError, match="frequency"):
            stack.at(source_depth=200.0)

    def test_frequency_axis_rejects_disagreeing_source_depths(self):
        a = _rays(source_depth=10.0, frequencies=100.0)
        b = _rays(source_depth=99.0, frequencies=200.0)
        with pytest.raises(ConfigurationError, match="source_depths"):
            ResultStack(slabs=[a, b], coordinate=[100.0, 200.0],
                        coordinate_name='frequency')

    def test_external_coordinate_axis(self):
        a = _rays(source_depth=50.0)
        b = _rays(source_depth=50.0)
        stack = ResultStack(slabs=[a, b], coordinate=[5.0, 15.0],
                            coordinate_name='wind_speed')
        assert stack.at(wind_speed=15.0) is b
        c = _rays(source_depth=99.0)
        with pytest.raises(ConfigurationError, match="source_depths"):
            ResultStack(slabs=[a, c], coordinate=[5.0, 15.0],
                        coordinate_name='wind_speed')

    def test_forwards_the_whole_identity_surface(self):
        """A stack forwards every identity field a slab carries — the
        plotters read ``model_source`` for the model-credit footnote."""
        a = _rays(source_depth=10.0, model_source='engine-provenance')
        b = _rays(source_depth=20.0, model_source='engine-provenance')
        stack = ResultStack(slabs=[a, b], coordinate=[10.0, 20.0])
        missing = [k for k in a.id_kwargs() if not hasattr(stack, k)]
        assert not missing, f"ResultStack does not forward {missing}"
        assert stack.model_source == 'engine-provenance'
        np.testing.assert_array_equal(stack.source_depths, [10.0, 20.0])
        np.testing.assert_array_equal(stack.frequencies, [100.0])

    def test_id_kwargs_carries_the_slabs_run_settings(self):
        """``id_kwargs`` reads the stack's identity, ``run_settings``
        included, which the stack takes from its first slab."""
        a = _rays(source_depth=10.0)
        b = _rays(source_depth=20.0)
        stack = ResultStack(slabs=[a, b], coordinate=[10.0, 20.0])
        kwargs = stack.id_kwargs()
        assert kwargs['run_settings'] is a.run_settings
        assert kwargs['model'] == a.model
        np.testing.assert_array_equal(kwargs['source_depths'], [10.0, 20.0])

    def test_from_slabs_returns_the_lone_slab_itself(self):
        a = _rays(source_depth=10.0)
        assert ResultStack.from_slabs([a], [10.0]) is a

    def test_from_slabs_stacks_two_slabs_along_the_coordinate(self):
        a = _rays(source_depth=10.0)
        b = _rays(source_depth=20.0)
        stack = ResultStack.from_slabs((s for s in (a, b)), [10.0, 20.0],
                                       coordinate_name='source_depth')
        assert isinstance(stack, ResultStack)
        assert stack.slabs == [a, b]

    def test_from_slabs_refuses_no_slab(self):
        with pytest.raises(ConfigurationError, match='at least one slab'):
            ResultStack.from_slabs([], [])

    def test_a_stack_is_drawn_one_slab_at_a_time(self):
        stack = ResultStack([_rays(source_depth=10.0),
                             _rays(source_depth=20.0)], [10.0, 20.0])
        with pytest.raises(ConfigurationError,
                           match=r'holds Rays slabs — pick one with stack\[i\]'):
            stack.plot()


class TestResultStackLabelsMustBeFinite:
    """``ResultStack.at`` picks its slab by ``argmin`` and refuses a label
    ``Field.at`` would refuse."""

    @staticmethod
    def _stack():
        return ResultStack(slabs=[_rays(source_depth=5.0),
                                  _rays(source_depth=50.0)],
                           coordinate=np.array([5.0, 50.0]),
                           coordinate_name='source_depth')

    def test_stack_at_nan_coordinate_raises_naming_the_stacking_axis(self):
        with pytest.raises(ConfigurationError,
                           match="source_depth=nan is not a finite label"):
            self._stack().at(source_depth=np.nan)

    def test_stack_at_inf_coordinate_raises(self):
        with pytest.raises(ConfigurationError, match="not a finite label"):
            self._stack().at(source_depth=np.inf)

    def test_stack_at_a_finite_coordinate_returns_the_nearest_slab(self):
        stack = self._stack()
        assert stack.at(source_depth=40.0) is stack.slabs[1]

    @pytest.mark.parametrize('label', [1e300, -1e300])
    def test_stack_at_a_label_that_absorbs_the_axis_is_refused(self, label):
        with pytest.raises(ConfigurationError, match='same distance'):
            self._stack().at(source_depth=label)


class TestAResultStackIsAnExportableResult:
    """The stack is a :class:`Result` and saves and loads as every result
    does; the slabs share no grid, so it is no single xarray object."""

    @staticmethod
    def _stack():
        return ResultStack([_arrivals(5.0, 1.0), _arrivals(15.0, 2.0)],
                           [5.0, 15.0])

    @staticmethod
    def _same(a, b):
        assert type(a) is type(b)
        assert a.coordinate_name == b.coordinate_name
        np.testing.assert_array_equal(a.coordinate, b.coordinate)
        for x, y in zip(a.slabs, b.slabs):
            assert type(x) is type(y)
            np.testing.assert_array_equal(x.amplitudes, y.amplitudes)
            np.testing.assert_array_equal(x.source_depths, y.source_depths)

    def test_it_is_a_result(self):
        assert isinstance(self._stack(), Result)

    def test_the_npz_round_trip_is_the_stack(self, tmp_path):
        stack = self._stack()
        np.savez(tmp_path / 's.npz', **stack.to_dict())
        with np.load(tmp_path / 's.npz', allow_pickle=True) as f:
            self._same(stack, ResultStack.from_dict(dict(f)))

    def test_the_netcdf_round_trip_is_the_stack(self, tmp_path):
        pytest.importorskip('xarray')
        stack = self._stack()
        stack.to_netcdf(tmp_path / 's.nc')
        self._same(stack, ResultStack.from_netcdf(tmp_path / 's.nc'))
        # The base class reads the file back to the class it names.
        self._same(stack, Result.from_netcdf(tmp_path / 's.nc'))

    def test_each_slab_keeps_its_components_through_netcdf(self, tmp_path):
        pytest.importorskip('xarray')
        from uacpy.tests._synthetic_fields import _bounce
        slabs = [_arrivals(5.0), _arrivals(15.0)]
        for slab in slabs:
            slab._components = {'bounce': _bounce()}
        ResultStack(slabs, [5.0, 15.0]).to_netcdf(tmp_path / 's.nc')
        back = Result.from_netcdf(tmp_path / 's.nc')
        assert sorted(back[1].components) == ['bounce']

    def test_the_table_lists_every_slab_under_its_coordinate(self):
        pytest.importorskip('pandas')
        frame = self._stack().to_dataframe()
        assert list(frame.columns)[0] == 'source_depth'
        assert frame['source_depth'].tolist() == [5.0, 5.0, 15.0, 15.0]
        np.testing.assert_array_equal(frame['amplitudes'],
                                      [1.0, 0.5, 2.0, 0.5])

    @pytest.mark.parametrize('call', ['values', 'to_xarray'])
    def test_it_has_no_array_of_its_own(self, call):
        with pytest.raises(ConfigurationError,
                           match=rf'ResultStack\.{call}: .*share no grid') as ei:
            getattr(self._stack(), call)()
        assert f'stack[i].{call}()' in str(ei.value)

    def test_from_xarray_names_the_file_route(self):
        with pytest.raises(ConfigurationError,
                           match=r'ResultStack\.from_netcdf\(path\)'):
            ResultStack.from_xarray(None)

    def test_a_foreign_class_path_is_refused(self):
        d = self._stack().to_dict()
        d['slabs'][0]['__class__'] = 'os.path.join'
        with pytest.raises(ConfigurationError, match='not a uacpy class'):
            ResultStack.from_dict(d)

    @pytest.mark.parametrize('key', ['coordinate', 'coordinate_name',
                                     'slabs'])
    def test_a_mapping_missing_a_written_key_is_refused_naming_it(self, key):
        d = self._stack().to_dict()
        del d[key]
        with pytest.raises(ConfigurationError,
                           match=rf"^ResultStack\.from_dict: .*{key}"):
            ResultStack.from_dict(d)
