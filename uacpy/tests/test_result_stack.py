"""``ResultStack``, one result per source.

Its invariants, the dB and complex-pressure views over the slabs, finite
labels, and the export as one xarray ``DataArray``.
"""

import numpy as np
import pytest
import uacpy
import warnings
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import Field
from uacpy.core.results.stack import ResultStack
from uacpy.tests._synthetic_fields import _field
from uacpy.tests._synthetic_fields import _two_path_grid


class TestResultStackInvariants:
    """:class:`ResultStack` is a thin composition wrapper. The
    constructor enforces uniform slab type, uniform model / backend /
    frequencies, and matching ``len(slabs) == len(source_depths)`` so
    the stack's read-through properties (``stack.model``,
    ``stack.frequencies``) never silently disagree with a slab."""

    @staticmethod
    def _slab(*, depths=2, ranges=3, frequencies=100.0, model='Test',
              source_depth=50.0, model_source=None, phase_reference=None):
        from uacpy.core.results import Field
        return Field(
            data=np.ones((depths, ranges), dtype=complex),
            coords={
                'depth': np.arange(depths, dtype=float),
                'range': np.arange(ranges, dtype=float) * 100.0,
            },
            model=model,
            frequencies=frequencies,
            source_depths=np.array([float(source_depth)]),
            model_source=model_source,
            phase_reference=phase_reference,
        )

    def test_requires_at_least_one_slab(self):
        from uacpy.core.results import ResultStack
        with pytest.raises(ConfigurationError, match="at least one slab"):
            ResultStack(slabs=[], coordinate=[])

    def test_rejects_length_mismatch(self):
        from uacpy.core.results import ResultStack
        with pytest.raises(ConfigurationError, match="coordinate length"):
            ResultStack(slabs=[self._slab(source_depth=10.0)],
                        coordinate=[10.0, 20.0])

    def test_rejects_mixed_slab_types(self):
        from uacpy.core.results import Rays, ResultStack
        pf = self._slab(source_depth=10.0)
        ry = Rays(rays=[], model='Test', backend='')
        with pytest.raises(ConfigurationError, match="same concrete type"):
            ResultStack(slabs=[pf, ry], coordinate=[10.0, 20.0])

    def test_rejects_disagreeing_frequencies(self):
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=10.0, frequencies=100.0)
        b = self._slab(source_depth=20.0, frequencies=200.0)
        with pytest.raises(ConfigurationError, match="frequencies"):
            ResultStack(slabs=[a, b], coordinate=[10.0, 20.0])

    def test_a_source_depth_sweep_on_another_axis_is_told_the_axis_that_admits_it(
            self):
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=10.0)
        b = self._slab(source_depth=20.0)
        with pytest.raises(
                ConfigurationError,
                match='every slab must share the same source_depths') as ei:
            ResultStack(slabs=[a, b], coordinate=[0, 1], coordinate_name='case')
        assert "coordinate_name='source_depth'" in str(ei.value)
        assert 'Source(depths=[...])' in str(ei.value)

    def test_a_disagreeing_model_names_no_stacking_axis(self):
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=10.0, model='Bellhop')
        b = self._slab(source_depth=10.0, model='Kraken')
        with pytest.raises(ConfigurationError,
                           match='every slab must share the same model') as ei:
            ResultStack(slabs=[a, b], coordinate=[0, 1], coordinate_name='case')
        assert 'coordinate_name=' not in str(ei.value)

    def test_rejects_disagreeing_model(self):
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=10.0, model='Bellhop')
        b = self._slab(source_depth=20.0, model='Kraken')
        with pytest.raises(ConfigurationError, match="model"):
            ResultStack(slabs=[a, b], coordinate=[10.0, 20.0])

    def test_accepts_uniform_slabs(self):
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=10.0)
        b = self._slab(source_depth=20.0)
        stack = ResultStack(slabs=[a, b], coordinate=[10.0, 20.0])
        assert stack.slab_type is Field
        assert stack.coordinate_name == 'source_depth'
        assert stack.n_slabs == 2
        assert len(stack) == 2
        # Universally-shared metadata reads through from slab[0].
        assert stack.model == 'Test'
        np.testing.assert_array_equal(
            stack.coordinate, np.array([10.0, 20.0]))

    def test_dB_stacks_slab_views_into_a_dense_array(self):
        """``stack.dB`` is one dense ``(n_slabs, *slab.shape)`` ndarray, so
        generic code can read ``result.dB`` whether one or many source
        depths were requested."""
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=10.0)               # |p| = 1 → 0 dB
        b = self._slab(source_depth=20.0)
        b.data[...] = 10.0 + 0j                          # |p| = 10 → -20 dB
        stack = ResultStack(slabs=[a, b], coordinate=[10.0, 20.0])
        dB = stack.dB
        assert isinstance(dB, np.ndarray)
        assert dB.shape == (2, 2, 3)                     # (n_slabs, z, r)
        np.testing.assert_allclose(dB[0], 0.0)
        np.testing.assert_allclose(dB[1], -20.0)

    def test_iteration_and_label_select_share_slab_identity(self):
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=10.0)
        b = self._slab(source_depth=20.0)
        stack = ResultStack(slabs=[a, b], coordinate=[10.0, 20.0])
        # __getitem__ returns the same object stored in slabs[i].
        assert stack[0] is a
        assert stack[1] is b
        # at(source_depth=z) routes to the nearest slab by label.
        assert stack.at(source_depth=20.0) is b
        # Iteration yields (source_depth, slab) pairs.
        pairs = list(stack)
        assert pairs == [(10.0, a), (20.0, b)]

    def test_frequency_axis_stack(self):
        """Stacking along ``frequency`` is just a coordinate-name swap.
        Slabs legitimately differ on ``frequencies`` (the stacking axis)
        while sharing ``source_depths`` and ``model``."""
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=50.0, frequencies=100.0)
        b = self._slab(source_depth=50.0, frequencies=200.0)
        stack = ResultStack(slabs=[a, b], coordinate=[100.0, 200.0],
                            coordinate_name='frequency')
        assert stack.coordinate_name == 'frequency'
        assert stack.at(frequency=200.0) is b
        # A kwarg that is not the stacking axis names an axis the stack
        # does not have.
        with pytest.raises(ConfigurationError, match="frequency"):
            stack.at(source_depth=200.0)

    def test_frequency_axis_rejects_disagreeing_source_depths(self):
        """When stacking by ``frequency`` the slabs must still agree on
        ``source_depths`` — ``frequency`` is the varying axis, not depth."""
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=10.0, frequencies=100.0)
        b = self._slab(source_depth=99.0, frequencies=200.0)
        with pytest.raises(ConfigurationError, match="source_depths"):
            ResultStack(slabs=[a, b], coordinate=[100.0, 200.0],
                        coordinate_name='frequency')

    def test_external_coordinate_axis(self):
        """An external coordinate (e.g. wind speed) requires both
        ``frequencies`` and ``source_depths`` to agree across slabs;
        ``at(<coordinate_name>=…)`` keys off the custom name."""
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=50.0, frequencies=100.0)
        b = self._slab(source_depth=50.0, frequencies=100.0)
        stack = ResultStack(slabs=[a, b], coordinate=[5.0, 15.0],
                            coordinate_name='wind_speed')
        assert stack.coordinate_name == 'wind_speed'
        assert stack.at(wind_speed=15.0) is b
        # An external coordinate requires both internal axes to agree, so a
        # disagreeing source_depth is rejected.
        c = self._slab(source_depth=99.0, frequencies=100.0)
        with pytest.raises(ConfigurationError, match="source_depths"):
            ResultStack(slabs=[a, c], coordinate=[5.0, 15.0],
                        coordinate_name='wind_speed')

    def test_forwards_the_whole_identity_surface(self):
        """A stack forwards every identity field a slab carries, not just
        model/backend — the plotters read ``model_source`` for the
        model-credit footnote."""
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=10.0, model_source='engine-provenance',
                       phase_reference='travelling_wave')
        b = self._slab(source_depth=20.0, model_source='engine-provenance',
                       phase_reference='travelling_wave')
        stack = ResultStack(slabs=[a, b], coordinate=[10.0, 20.0])
        # Every field of the shared identity surface, so a field added to
        # ``Result.__init__`` cannot silently stop at the stack boundary.
        missing = [k for k in a.id_kwargs() if not hasattr(stack, k)]
        assert not missing, f"ResultStack does not forward {missing}"
        assert stack.model_source == 'engine-provenance'
        assert stack.phase_reference == 'travelling_wave'
        # The stacking axis reads back as the stack coordinate; the other
        # axis reads through from the (verified identical) slabs.
        np.testing.assert_array_equal(stack.source_depths, [10.0, 20.0])
        np.testing.assert_array_equal(stack.frequencies, [100.0])

    def test_frequency_stack_forwards_the_frequency_axis(self):
        from uacpy.core.results import ResultStack
        a = self._slab(source_depth=50.0, frequencies=100.0)
        b = self._slab(source_depth=50.0, frequencies=200.0)
        stack = ResultStack(slabs=[a, b], coordinate=[100.0, 200.0],
                            coordinate_name='frequency')
        np.testing.assert_array_equal(stack.frequencies, [100.0, 200.0])
        np.testing.assert_array_equal(stack.source_depths, [50.0])

    def test_from_slabs_returns_the_lone_slab_itself(self):
        a = self._slab(source_depth=10.0)
        assert ResultStack.from_slabs([a], [10.0]) is a

    def test_from_slabs_stacks_two_slabs_along_the_coordinate(self):
        a = self._slab(source_depth=10.0)
        b = self._slab(source_depth=20.0)
        stack = ResultStack.from_slabs((s for s in (a, b)), [10.0, 20.0],
                                       coordinate_name='source_depth')
        assert isinstance(stack, ResultStack)
        assert stack.slabs == [a, b]
        np.testing.assert_array_equal(stack.coordinate, [10.0, 20.0])
        assert stack.coordinate_name == 'source_depth'

    def test_from_slabs_refuses_no_slab(self):
        with pytest.raises(ConfigurationError, match='at least one slab'):
            ResultStack.from_slabs([], [])


class TestResultStackDbRefusesNonDbSlabs:
    """``ResultStack.dB`` raises the stack's typed error for slabs whose
    real data is not a level (unit other than ``'dB'``), while complex
    slabs — whose dB view is derived — still stack."""

    def _slab(self, data, **quantity):
        return Field(data=data,
                     coords={'depth': np.array([1.0, 2.0]),
                             'range': np.array([10.0, 20.0, 30.0])},
                     **quantity)

    def _stack(self, slabs):
        return ResultStack(slabs, np.array([5.0, 10.0]),
                           coordinate_name='source_depth')

    def test_dimensionless_slabs_raise_the_stack_typed_error(self):
        pd = self._slab(np.full((2, 3), 0.5),
                        kind='probability_of_detection', unit='1')
        with pytest.raises(ConfigurationError,
                           match=r"ResultStack\.dB: slabs are in '1', not dB"):
            self._stack([pd, pd]).dB

    def test_complex_pressure_slabs_stack_to_a_dB_view(self):
        stack = self._stack([self._slab(np.full((2, 3), 1j)),
                             self._slab(np.full((2, 3), 1j))])
        assert stack.dB.shape == (2, 2, 3)
        assert np.allclose(stack.dB, 0.0)

    def test_time_domain_slabs_raise_the_stack_typed_error(self):
        trace = Field(data=np.zeros(4),
                      coords={'time': np.arange(4.0)})
        with pytest.raises(ConfigurationError, match="time-domain slabs"):
            ResultStack([trace, trace], np.array([5.0, 10.0]),
                        coordinate_name='source_depth').dB


def _tl_field(value):
    return Field(data=np.full((2, 3), value),
                 coords={'depth': np.array([10.0, 20.0]),
                         'range': np.array([100.0, 200.0, 300.0])})


class TestResultStackAndTraceLabelsMustBeFinite:
    """``ResultStack.at`` and the single-cell IFFT pick their slab / cell by
    ``argmin``, where ``Field.at`` routes through ``collapse_axis`` and
    already raised — the same query answered two different ways."""

    def _stack(self):
        return ResultStack(slabs=[_tl_field(10.0), _tl_field(20.0)],
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
        """The rule ``Field.at`` applies: every ``|c - 1e300|`` rounds to
        1e300, argmin ties on index 0, and the first slab would be handed
        back as if it were the nearest."""
        with pytest.raises(ConfigurationError, match='same distance'):
            self._stack().at(source_depth=label)

    def _broadband(self):
        n_freq = 64
        data = np.zeros((3, 2, n_freq), dtype=complex)
        data[1, 0, :] = 1.0
        return Field(data=data,
                     coords={'depth': np.array([10.0, 20.0, 30.0]),
                             'range': np.array([1000.0, 2000.0]),
                             'frequency': np.linspace(100.0, 500.0, n_freq)})

    def test_to_time_trace_nan_depth_raises_a_typed_label_error(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(ConfigurationError,
                               match="depth=nan is not a finite label"):
                self._broadband().to_time_trace(depth=np.nan)

    def test_to_time_trace_inf_range_raises(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with pytest.raises(ConfigurationError, match="not a finite label"):
                self._broadband().to_time_trace(range=np.inf)

    def test_to_time_trace_pins_the_nearest_cell_for_a_finite_label(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            trace = self._broadband().to_time_trace(depth=19.0, range=1900.0)
        assert trace.pinned == {'depth': 20.0, 'range': 2000.0}

    def test_to_time_trace_without_labels_takes_the_middle_depth(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            trace = self._broadband().to_time_trace()
        assert trace.pinned == {'depth': 20.0, 'range': 1000.0}


class TestAStackHasAComplexPressureView:
    """``ResultStack.p`` stacks every slab's ``Field.p``, shape
    ``(n_slabs, *slab.shape)``, and refuses a real (dB) stack as
    ``Field.p`` does."""

    @staticmethod
    def _slab(data):
        return Field(data=np.asarray(data),
                     coords={'depth': np.array([10.0]),
                             'range': np.array([1.0, 2.0])},
                     model='Test')

    def test_the_complex_stack_returns_each_slabs_pressure(self):
        from uacpy.core.results import ResultStack
        a, b = np.array([[1 + 1j, 2j]]), np.array([[3.0 + 0j, -1j]])
        stack = ResultStack([self._slab(a), self._slab(b)],
                            np.array([10.0, 20.0]),
                            coordinate_name='source_depth')
        assert stack.p.shape == (2, 1, 2)
        np.testing.assert_array_equal(stack.p, np.stack([a, b]))

    def test_a_db_stack_is_refused(self):
        from uacpy.core.results import ResultStack
        slab = Field(data=np.array([[60.0, 70.0]]),
                     coords={'depth': np.array([10.0]),
                             'range': np.array([1.0, 2.0])},
                     model='Test', unit='dB')
        stack = ResultStack([slab, slab.copy()], np.array([10.0, 20.0]),
                            coordinate_name='source_depth')
        with pytest.raises(ConfigurationError, match='phase'):
            stack.p


class TestAResultStackExportsAsOneDataArray:
    """``ResultStack.to_xarray`` joins the slabs along a leading dimension
    named by the stack's coordinate; each slab reads back with
    ``Field.from_xarray``, and the attribute the stacking axis varies is not
    reported from slab 0 alone."""

    @staticmethod
    def _stack():
        from uacpy.core.results import ResultStack

        def slab(z, scale):
            return Field(data=scale * (np.arange(6.0).reshape(2, 3) + 1j),
                         coords={'depth': np.array([10.0, 20.0]),
                                 'range': np.array([100.0, 200.0, 300.0])},
                         model='Bellhop', frequencies=np.array([250.0]),
                         source_depths=np.array([z]),
                         phase_reference=uacpy.PhaseReference.TRAVELLING_WAVE)
        return ResultStack([slab(5.0, 1.0), slab(15.0, 2.0)], [5.0, 15.0])

    def test_the_slabs_stack_along_the_coordinate(self):
        pytest.importorskip('xarray')
        stack = self._stack()
        da = stack.to_xarray()
        assert list(da.dims) == ['source_depth', 'depth', 'range']
        np.testing.assert_array_equal(da.coords['source_depth'], [5.0, 15.0])
        np.testing.assert_array_equal(da.values, stack.p)
        assert 'source_depths' not in da.attrs
        assert da.attrs['phase_reference'] == 'travelling_wave'

    def test_each_slab_reads_back_as_its_field(self):
        pytest.importorskip('xarray')
        stack = self._stack()
        da = stack.to_xarray()
        back = Field.from_xarray(da.isel(source_depth=1))
        np.testing.assert_array_equal(back.data, stack[1].data)
        assert back.pinned == {'source_depth': 15.0}

    def test_a_non_field_stack_is_refused(self):
        from uacpy.core.results import Rays, ResultStack
        stack = ResultStack([Rays(rays=[], model='T'), Rays(rays=[], model='T')],
                            [1.0, 2.0], coordinate_name='case')
        with pytest.raises(ConfigurationError, match='not Field'):
            stack.to_xarray()


class TestAStackIsOneQuantityOverSources:
    def test_slabs_holding_different_quantities_are_refused(self):
        p = _two_path_grid().at(frequency=200.0)
        with pytest.raises(ConfigurationError,
                           match="storage 'complex' vs 'real'"):
            ResultStack([p, p.to_dB()], [1.0, 2.0])
        tl = _field(data=np.full((4, 3), 60.0))
        level = _field(data=np.full((4, 3), 120.0), kind='level', unit='dB')
        with pytest.raises(ConfigurationError, match="kind 'pressure' vs"):
            ResultStack([tl, level], [1.0, 2.0])

    def test_a_stack_over_anything_but_sources_does_not_superpose(self):
        p = _two_path_grid().at(frequency=200.0)
        stack = ResultStack([p, p], [1600.0, 1700.0],
                            coordinate_name='bottom_speed')
        with pytest.raises(ConfigurationError, match="'bottom_speed'"):
            stack.superpose()


class TestOneSourceWeightIsTheStackSum:
    """M-33: the n = 1 weight a run applies to a one-depth ``Source`` and
    ``ResultStack.superpose`` form one sum, ``_weighted_slab_sum``."""

    @staticmethod
    def _slab():
        return Field(data=np.array([[1.0 + 2.0j, -3.0 + 0.5j]]),
                     coords={'depth': np.array([10.0]),
                             'range': np.array([100.0, 200.0])},
                     model='Test', frequencies=100.0,
                     source_depths=np.array([50.0]),
                     phase_reference='travelling_wave')

    def test_both_paths_call_the_one_sum(self, monkeypatch):
        from uacpy.core.results import stack as stack_module
        from uacpy.models import _stacking
        calls = []
        real = stack_module._weighted_slab_sum

        def spy(slabs, weights, *, where):
            calls.append(where)
            return real(slabs, weights, where=where)
        monkeypatch.setattr(stack_module, '_weighted_slab_sum', spy)
        monkeypatch.setattr(_stacking, '_weighted_slab_sum', spy)
        ResultStack(slabs=[self._slab()], coordinate=[50.0]).superpose(
            [2.0 - 1.0j])
        source = uacpy.Source(depths=50.0, frequencies=100.0,
                              weights=[2.0 - 1.0j])
        _stacking.apply_single_source_weight(self._slab(), source,
                                             model_name='Test')
        assert len(calls) == 2

    def test_the_one_source_weight_equals_the_one_slab_superpose(self):
        from uacpy.models import _stacking
        w = 2.0 - 1.0j
        summed = ResultStack(slabs=[self._slab()],
                             coordinate=[50.0]).superpose([w])
        source = uacpy.Source(depths=50.0, frequencies=100.0, weights=[w])
        weighted = _stacking.apply_single_source_weight(
            self._slab(), source, model_name='Test')
        np.testing.assert_array_equal(weighted.data, summed.data)
