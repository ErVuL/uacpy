"""
Tests for core UACPY classes: Environment, Source, Receiver, Result
"""

import pytest
import numpy as np
import uacpy


class TestEnvironment:
    """Tests for Environment class."""

    def test_create_simple_environment(self, simple_env):
        """Test creating a simple isovelocity environment."""
        assert simple_env.name == "Test Environment"
        assert simple_env.depth == 100.0
        assert float(simple_env.ssp.data[0, 0]) == 1500.0
        assert simple_env.ssp.interp == 'isovelocity'
        assert not simple_env.is_range_dependent

    def test_create_munk_environment(self, munk_env):
        """Test creating environment with Munk profile."""
        assert munk_env.name == "Munk Profile"
        assert munk_env.depth == 100.0
        assert munk_env.ssp.n_depths == 21
        assert munk_env.ssp.interp == 'pchip'

    def test_range_dependent_environment(self, range_dependent_env):
        """Test range-dependent environment."""
        assert range_dependent_env.is_range_dependent
        assert len(range_dependent_env.bathymetry) == 11
        assert range_dependent_env.bathymetry[0, 1] == 80.0
        assert range_dependent_env.bathymetry[-1, 1] == 120.0

    def test_ssp_pairs_shape(self, simple_env, munk_env):
        """SSP pairs view always has shape (N, 2)."""
        assert simple_env.ssp.to_pairs().shape[1] == 2
        assert munk_env.ssp.to_pairs().shape[1] == 2

    def test_get_representative_depth(self, range_dependent_env):
        """Test getting representative depth for range-dependent environment."""
        median_depth = range_dependent_env.get_representative_depth('median')
        assert 80 <= median_depth <= 120

    def test_invalid_depth(self):
        """Test that negative depth raises error."""
        with pytest.raises(ValueError):
            uacpy.Environment(name="Test", bathymetry=-10, ssp=1500)


class TestSource:
    """Tests for Source class."""

    def test_create_source(self, source):
        """Test creating a source."""
        assert source.depths[0] == 50.0
        assert source.frequencies[0] == 100.0

    def test_source_array_conversion(self):
        """Test that single values are converted to arrays."""
        source = uacpy.Source(depths=30.0, frequencies=200.0)
        assert isinstance(source.depths, np.ndarray)
        assert isinstance(source.frequencies, np.ndarray)
        assert len(source.depths) == 1
        assert len(source.frequencies) == 1

    def test_multiple_sources(self):
        """Test multiple source depths."""
        source = uacpy.Source(depths=[10.0, 20.0, 30.0], frequencies=100.0)
        assert len(source.depths) == 3
        assert np.allclose(source.depths, [10, 20, 30])

    def test_multiple_frequencies(self):
        """Test multiple frequencies."""
        source = uacpy.Source(depths=50.0, frequencies=[50.0, 100.0, 200.0])
        assert len(source.frequencies) == 3


class TestReceiver:
    """Tests for Receiver class."""

    def test_create_receiver_grid(self, receiver_grid):
        """Test creating receiver grid."""
        assert len(receiver_grid.depths) == 9
        assert len(receiver_grid.ranges) == 11
        assert receiver_grid.receiver_type == 'grid'
        assert receiver_grid.depth_min == 10.0
        assert receiver_grid.depth_max == 90.0
        assert receiver_grid.range_min == 100.0
        assert receiver_grid.range_max == 5000.0

    def test_small_receiver_grid(self, receiver_small):
        """Test small receiver grid."""
        assert len(receiver_small.depths) == 3
        assert len(receiver_small.ranges) == 3

    def test_receiver_line_array(self):
        """Test line array receiver."""
        receiver = uacpy.Receiver(
            depths=[50.0],
            ranges=np.linspace(1000, 10000, 100)
        )
        assert len(receiver.depths) == 1
        assert len(receiver.ranges) == 100


class TestField:
    """Tests for the typed Result hierarchy (PressureField etc.)."""

    def test_create_tl_field(self):
        from uacpy.core.results import PressureField
        data = np.random.rand(10, 20) * 50 + 40  # Random TL between 40-90 dB
        ranges = np.linspace(100, 5000, 20)
        depths = np.linspace(10, 90, 10)

        field = PressureField(units="dB", data=data, ranges=ranges, depths=depths,
                        model='Test', frequencies=100.0)

        assert field.field_type == 'tl'
        assert field.shape == (10, 20)
        assert field.n_ranges == 20
        assert field.n_depths == 10

    def test_field_at_point(self):
        from uacpy.core.results import PressureField
        data = np.arange(100).reshape(10, 10).astype(float)
        ranges = np.linspace(0, 9000, 10)
        depths = np.linspace(0, 90, 10)

        field = PressureField(units="dB", data=data, ranges=ranges, depths=depths,
                        model='Test', frequencies=100.0)
        value = float(field.at(range_m=4500, depth=45).tl)
        assert 44 <= value <= 55

    def test_field_at_range(self):
        from uacpy.core.results import PressureField
        data = np.arange(100).reshape(10, 10).astype(float)
        ranges = np.linspace(0, 9000, 10)
        depths = np.linspace(0, 90, 10)

        field = PressureField(units="dB", data=data, ranges=ranges, depths=depths,
                        model='Test', frequencies=100.0)
        values = field.at(range_m=4500).tl
        assert len(values) == 10
        assert 50 <= values[5] <= 59

    def test_field_at_depth(self):
        from uacpy.core.results import PressureField
        data = np.arange(100).reshape(10, 10).astype(float)
        ranges = np.linspace(0, 9000, 10)
        depths = np.linspace(0, 90, 10)

        field = PressureField(units="dB", data=data, ranges=ranges, depths=depths,
                        model='Test', frequencies=100.0)
        values = field.at(depth=45).tl
        assert len(values) == 10
        assert 40 <= values[5] <= 49

    def test_field_copy(self):
        from uacpy.core.results import PressureField
        data = np.random.rand(10, 20)
        ranges = np.linspace(100, 5000, 20)
        depths = np.linspace(10, 90, 10)

        field = PressureField(units="dB", data=data, ranges=ranges, depths=depths,
                        model='Test', frequencies=100.0)
        field_copy = field.copy()

        assert type(field_copy) is type(field)
        assert np.array_equal(field_copy.data, field.data)
        assert field_copy is not field
        assert field_copy.data is not field.data

    def test_field_repr(self):
        from uacpy.core.results import PressureField
        data = np.random.rand(10, 20)
        ranges = np.linspace(100, 5000, 20)
        depths = np.linspace(10, 90, 10)

        field = PressureField(units="dB", data=data, ranges=ranges, depths=depths,
                        model='Test', frequencies=100.0)
        repr_str = repr(field)
        assert 'PressureField' in repr_str
        assert field.shape == (10, 20)
        assert field.n_ranges == 20
        assert field.n_depths == 10


class TestPublicReexports:
    """Top-level re-exports added for the API audit."""

    def test_environment_helpers_at_top_level(self):
        from uacpy import SoundSpeedProfile, generate_sea_surface
        assert SoundSpeedProfile is uacpy.core.environment.SoundSpeedProfile
        assert generate_sea_surface is uacpy.core.environment.generate_sea_surface

    def test_environment_helpers_at_core(self):
        from uacpy.core import SoundSpeedProfile, generate_sea_surface
        assert SoundSpeedProfile is uacpy.core.environment.SoundSpeedProfile
        assert generate_sea_surface is uacpy.core.environment.generate_sea_surface

    def test_signal_analysis_classes_at_top_level(self):
        sig = uacpy.signal
        for name in ('PPSD', 'PSD', 'FRF', 'SEL', 'FKTransform', 'Spectrogram'):
            assert hasattr(sig, name), f"uacpy.signal.{name} not reachable"
        assert 'PSD' in sig.__all__
        assert 'FRF' in sig.__all__
        assert 'SEL' in sig.__all__
        assert 'FKTransform' in sig.__all__


class TestModesFieldType:
    """Single canonical ``field_type`` declaration on Modes."""

    def test_modes_field_type_value(self):
        from uacpy.core.results import Modes
        assert Modes.field_type == "modes"


class TestArrivalsToTableKeys:
    """``Arrivals.to_table()`` emits writer-aligned bounce keys."""

    def test_to_table_uses_n_bounce_keys(self):
        from uacpy.core.results import Arrivals
        # Build a minimal payload with the canonical IO key naming.
        payload = [[[{
            "delays": np.array([0.1, 0.2]),
            "amplitudes": np.array([1.0, 0.5]),
            "phases": np.array([0.0, 0.1]),
            "n_top_bounces": np.array([0, 1], dtype=int),
            "n_bot_bounces": np.array([1, 2], dtype=int),
            "src_angles": np.array([0.0, 5.0]),
            "rcv_angles": np.array([0.0, -5.0]),
            "n_arrivals": 2,
        }]]]
        arr = Arrivals(
            by_receiver=payload,
            receiver_depths=np.array([50.0]),
            receiver_ranges=np.array([1000.0]),
            model='Test',
            frequencies=100.0,
        )
        table = arr.to_table()
        assert len(table) == 2
        for row in table:
            assert 'n_top_bounces' in row
            assert 'n_bot_bounces' in row
        assert table[0]['kind'] == 'bottom'
        assert table[1]['kind'] == 'both'
        assert table[0]['n_bot_bounces'] == 1
        assert table[1]['n_top_bounces'] == 1


class TestArrivalsFilterChain:
    """Arrivals exposes a Rays-style filter chain over a flat list of
    arrival events; no continuous-axis ``at(...)`` slicer."""

    def _arrivals(self):
        from uacpy.core.results import Arrivals
        # Two cells (1 src × 1 depth × 2 ranges) with a mix of bounce kinds.
        cell0 = {
            "delays": np.array([0.1, 0.2, 0.3]),
            "amplitudes": np.array([1.0, 0.5, 0.2]),
            "phases": np.array([0.0, 0.0, 0.0]),
            "n_top_bounces": np.array([0, 1, 1], dtype=int),
            "n_bot_bounces": np.array([0, 0, 2], dtype=int),
            "src_angles": np.array([0.0, 5.0, 10.0]),
            "rcv_angles": np.array([0.0, -5.0, -10.0]),
        }
        cell1 = {
            "delays": np.array([0.4]),
            "amplitudes": np.array([0.8]),
            "phases": np.array([0.0]),
            "n_top_bounces": np.array([0], dtype=int),
            "n_bot_bounces": np.array([1], dtype=int),
            "src_angles": np.array([2.0]),
            "rcv_angles": np.array([-2.0]),
        }
        return Arrivals(
            by_receiver=[[[cell0, cell1]]],
            receiver_depths=np.array([50.0]),
            receiver_ranges=np.array([1000.0, 2000.0]),
            model='Test', frequencies=100.0,
        )

    def test_flat_list_length(self):
        a = self._arrivals()
        assert len(a) == 4    # 3 from cell0 + 1 from cell1

    def test_filter_by_bounces_kind(self):
        a = self._arrivals()
        direct = a.filter_by_bounces(kind='direct')
        assert len(direct) == 1
        assert direct.arrivals[0]['delay'] == pytest.approx(0.1)
        bottom = a.filter_by_bounces(kind='bottom')
        assert len(bottom) == 1
        assert bottom.arrivals[0]['delay'] == pytest.approx(0.4)

    def test_filter_by_bounces_top_low_high(self):
        a = self._arrivals()
        # 0-1 surface bounces — keeps everything but the 'both' arrival
        # has top=1 too, so all four pass; instead use bot=(1, None).
        few_bot = a.filter_by_bounces(bot=(1, None))
        assert len(few_bot) == 2     # cell0 last + cell1 last
        exact_top = a.filter_by_bounces(top=1)
        assert len(exact_top) == 2   # cell0 idx 1 and 2

    def test_in_delay_window(self):
        a = self._arrivals()
        mid = a.in_delay_window(0.15, 0.35)
        assert len(mid) == 2
        assert all(0.15 <= x['delay'] <= 0.35 for x in mid.arrivals)

    def test_top_n_by_amplitude(self):
        a = self._arrivals()
        top2 = a.top_n_by_amplitude(2)
        assert len(top2) == 2
        amps = [x['amplitude'] for x in top2.arrivals]
        assert amps == sorted(amps, reverse=True)
        assert amps[0] == 1.0   # cell0 first

    def test_filter_chain_returns_arrivals(self):
        from uacpy.core.results import Arrivals
        a = self._arrivals()
        chained = a.filter(lambda x: x['n_bot_bounces'] >= 1).top_n_by_amplitude(1)
        assert isinstance(chained, Arrivals)
        assert len(chained) == 1


class TestSoundSpeedProfileExtendTo:
    """``SoundSpeedProfile.extend_to(z_max)`` is the canonical alignment
    hook used by every env writer. Must extend OR truncate so that
    ``ssp.depths[-1] == z_max`` exactly."""

    def _profile(self, depths, speeds):
        from uacpy.core.environment import SoundSpeedProfile
        return SoundSpeedProfile(
            depths=np.asarray(depths, dtype=float),
            data=np.asarray(speeds, dtype=float).reshape(-1, 1),
            interp='linear',
        )

    def test_noop_when_depth_max_equals_deepest(self):
        ssp = self._profile([0, 100, 200], [1500, 1490, 1485])
        assert ssp.extend_to(200.0) is ssp

    def test_extend_with_constant_extrapolation(self):
        out = self._profile([0, 100], [1500, 1490]).extend_to(300.0)
        assert out.depths[-1] == 300.0
        assert out.data[-1, 0] == 1490.0

    def test_truncate_with_linear_interpolation(self):
        out = self._profile([0, 100, 200], [1500, 1490, 1480]).extend_to(150.0)
        assert out.depths[-1] == 150.0
        assert out.data[-1, 0] == pytest.approx(1485.0)
        assert (out.depths <= 150.0).all()

    def test_truncate_then_extend_round_trip(self):
        ssp = self._profile([0, 100, 200], [1500, 1490, 1480])
        out = ssp.extend_to(150.0).extend_to(150.0)
        assert out.depths[-1] == 150.0
        assert out.data[-1, 0] == pytest.approx(1485.0)


class TestPressureFieldChainAccessors:
    """``PressureField`` slicing returns ``_SlicedPressureField`` whose
    ``.tl`` / ``.p`` auto-squeeze singleton axes; full grids preserve
    ``.data.shape``."""

    def _full_grid(self, units='complex'):
        from uacpy.core.results import PressureField
        if units == 'complex':
            data = (np.arange(20).reshape(4, 5) + 1j).astype(complex)
        else:
            data = np.arange(20, dtype=float).reshape(4, 5) + 30.0
        return PressureField(
            data=data,
            depths=np.linspace(10, 90, 4),
            ranges=np.linspace(100, 1000, 5),
            units=units,
            model='Test', frequencies=100.0,
        )

    def test_full_grid_tl_preserves_data_shape(self):
        f = self._full_grid('complex')
        assert f.tl.shape == f.data.shape == (4, 5)

    def test_full_grid_p_preserves_data_shape(self):
        f = self._full_grid('complex')
        assert f.p.shape == f.data.shape == (4, 5)

    def test_p_raises_on_db_units(self):
        f = self._full_grid('dB')
        with pytest.raises(AttributeError):
            _ = f.p

    def test_at_depth_returns_sliced_subtype_with_squeezed_tl(self):
        from uacpy.core.results import _SlicedPressureField
        f = self._full_grid('complex')
        sliced = f.at(depth=50.0)
        assert isinstance(sliced, _SlicedPressureField)
        assert sliced.tl.shape == (5,)
        assert sliced.data.shape == (1, 5)

    def test_at_range_returns_sliced_subtype_with_squeezed_tl(self):
        from uacpy.core.results import _SlicedPressureField
        f = self._full_grid('complex')
        sliced = f.at(range_m=500.0)
        assert isinstance(sliced, _SlicedPressureField)
        assert sliced.tl.shape == (4,)
        assert sliced.data.shape == (4, 1)

    def test_at_point_returns_zero_d_scalar_tl(self):
        from uacpy.core.results import _SlicedPressureField
        f = self._full_grid('complex')
        point = f.at(range_m=500.0, depth=50.0)
        assert isinstance(point, _SlicedPressureField)
        assert point.tl.shape == ()
        assert isinstance(float(point.tl), float)

    def test_max_returns_zero_d_scalar_tl(self):
        from uacpy.core.results import _SlicedPressureField
        f = self._full_grid('complex')
        m = f.max()
        assert isinstance(m, _SlicedPressureField)
        assert m.tl.shape == ()
        flat = int(np.argmax(np.abs(f.data)))
        d_idx, r_idx = np.unravel_index(flat, f.data.shape)
        assert m.depths[0] == f.depths[d_idx]
        assert m.ranges[0] == f.ranges[r_idx]

    def test_max_on_3d_collapses_frequency_axis(self):
        """3-D broadband fields: get_max picks the global argmax across
        all three axes — depth, range AND frequency — and the picked
        frequency lands in .frequencies[0]."""
        from uacpy.core.results import PressureField, _SlicedPressureField
        rng = np.random.default_rng(0)
        data = (rng.standard_normal((3, 4, 5))
                + 1j * rng.standard_normal((3, 4, 5)))
        # Spike one specific (d, r, f) cell.
        data[2, 1, 3] = 100.0 + 0j
        pf = PressureField(
            data=data,
            depths=np.linspace(10, 90, 3),
            ranges=np.linspace(100, 1000, 4),
            units='complex',
            frequencies=np.linspace(50, 250, 5),
            model='Test',
        )
        m = pf.max()
        assert isinstance(m, _SlicedPressureField)
        assert m.tl.shape == ()
        assert m.depths[0] == pf.depths[2]
        assert m.ranges[0] == pf.ranges[1]
        assert m.frequencies[0] == pf.frequencies[3]


class TestTransferFunction:
    """``TransferFunction`` is a sibling of ``PressureField`` (both are
    ``_GridResult`` subclasses) — the synthesis methods belong on TF
    only, and the two ``isinstance`` checks are mutually exclusive."""

    def _tf(self):
        from uacpy.core.results import TransferFunction
        data = (np.arange(24).reshape(2, 3, 4) + 1j).astype(complex)
        return TransferFunction(
            data=data,
            depths=np.array([10., 20.]),
            ranges=np.array([100., 200., 300.]),
            frequencies=np.array([100., 200., 300., 400.]),
            phase_reference='travelling_wave',
            model='Test',
        )

    def test_tf_is_not_a_pressurefield(self):
        from uacpy.core.results import PressureField
        assert not isinstance(self._tf(), PressureField)

    def test_tf_phase_reference_coerced_to_enum(self):
        from uacpy.core.results import PhaseReference
        tf = self._tf()
        assert isinstance(tf.phase_reference, PhaseReference)
        assert tf.phase_reference == PhaseReference.TRAVELLING_WAVE

    def test_tf_tl_and_p_preserve_data_shape(self):
        tf = self._tf()
        assert tf.tl.shape == tf.data.shape == (2, 3, 4)
        assert tf.p.shape == tf.data.shape

    def test_tf_synthesis_methods_present(self):
        tf = self._tf()
        assert hasattr(tf, 'synthesize_time_series')
        assert hasattr(tf, 'to_time_trace')
        assert hasattr(tf, 'at')
        assert hasattr(tf, 'to_tl')
        assert hasattr(tf, 'max')


class TestTransferFunctionSlicing:
    """Spatial slicing of a TransferFunction degrades to
    ``_SlicedPressureField`` — the slice is a pressure field, no longer
    a transfer function (synthesis machinery only on the full TF)."""

    def _tf(self):
        from uacpy.core.results import TransferFunction
        data = (np.arange(24).reshape(2, 3, 4) + 1j).astype(complex)
        return TransferFunction(
            data=data,
            depths=np.array([10., 20.]),
            ranges=np.array([100., 200., 300.]),
            frequencies=np.array([100., 200., 300., 400.]),
            phase_reference='travelling_wave',
            model='Test',
        )

    def test_at_depth_returns_sliced_pressurefield(self):
        from uacpy.core.results import (
            TransferFunction, _SlicedPressureField,
        )
        sliced = self._tf().at(depth=15.0)
        assert isinstance(sliced, _SlicedPressureField)
        assert not isinstance(sliced, TransferFunction)
        # synthesis is unreachable from a slice — that's the point.
        assert not hasattr(sliced, 'synthesize_time_series')
        # squeeze drops the 1-element depth axis.
        assert sliced.tl.shape == (3, 4)

    def test_at_range_returns_sliced_pressurefield(self):
        from uacpy.core.results import _SlicedPressureField
        sliced = self._tf().at(range_m=200.0)
        assert isinstance(sliced, _SlicedPressureField)
        assert sliced.tl.shape == (2, 4)

    def test_at_point_to_tl_returns_sliced_pressurefield(self):
        from uacpy.core.results import _SlicedPressureField
        # tl_at preserves the frequency axis at one (depth, range) cell.
        view = self._tf().at(depth=15.0, range_m=200.0).to_tl()
        assert isinstance(view, _SlicedPressureField)
        assert view.tl.shape == (4,)        # squeezed (1, 1, 4) → (4,)
        assert view.units == 'dB'


class TestTransferFunctionFromPressureField:
    """``TransferFunction.from_pressure_field(pf, phase_reference=...)``
    promotes a 3-D complex :class:`PressureField` to a transfer function."""

    def _broadband_pf(self):
        from uacpy.core.results import PressureField
        data = (np.arange(24).reshape(2, 3, 4) + 1j).astype(complex)
        return PressureField(
            data=data,
            depths=np.array([10., 20.]),
            ranges=np.array([100., 200., 300.]),
            units='complex',
            frequencies=np.array([100., 200., 300., 400.]),
            model='Pekeris', backend='bellhop',
            metadata={'extra': 'roundtrip'},
        )

    def test_promotes_broadband_complex_field(self):
        from uacpy.core.results import (
            PhaseReference, PressureField, TransferFunction,
        )
        pf = self._broadband_pf()
        tf = TransferFunction.from_pressure_field(
            pf, phase_reference='travelling_wave',
        )
        assert isinstance(tf, TransferFunction)
        assert not isinstance(tf, PressureField)   # sibling, not subclass
        assert tf.phase_reference == PhaseReference.TRAVELLING_WAVE
        # axes / data / metadata round-trip
        np.testing.assert_array_equal(tf.data, pf.data)
        np.testing.assert_array_equal(tf.depths, pf.depths)
        np.testing.assert_array_equal(tf.ranges, pf.ranges)
        np.testing.assert_array_equal(tf.frequencies, pf.frequencies)
        assert tf.model == pf.model
        assert tf.metadata['extra'] == 'roundtrip'

    def test_rejects_db_units(self):
        from uacpy.core.results import PressureField, TransferFunction
        pf = PressureField(
            data=np.zeros((2, 3, 4)),
            depths=np.array([10., 20.]),
            ranges=np.array([100., 200., 300.]),
            units='dB',
            frequencies=np.array([100., 200., 300., 400.]),
        )
        with pytest.raises(ValueError, match="units='complex'"):
            TransferFunction.from_pressure_field(
                pf, phase_reference='travelling_wave',
            )

    def test_rejects_narrowband_field(self):
        from uacpy.core.results import PressureField, TransferFunction
        pf = PressureField(
            data=np.zeros((2, 3), dtype=complex),
            depths=np.array([10., 20.]),
            ranges=np.array([100., 200., 300.]),
            units='complex',
            frequencies=100.0,
        )
        with pytest.raises(ValueError, match="3-D"):
            TransferFunction.from_pressure_field(
                pf, phase_reference='travelling_wave',
            )

    def test_rejects_non_pressurefield(self):
        from uacpy.core.results import TransferFunction
        with pytest.raises(TypeError):
            TransferFunction.from_pressure_field(
                object(), phase_reference='travelling_wave',
            )
