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

    def test_field_get_value(self):
        from uacpy.core.results import PressureField
        data = np.arange(100).reshape(10, 10).astype(float)
        ranges = np.linspace(0, 9000, 10)
        depths = np.linspace(0, 90, 10)

        field = PressureField(units="dB", data=data, ranges=ranges, depths=depths,
                        model='Test', frequencies=100.0)
        value = field.get_value(range_m=4500, depth=45)
        assert 44 <= value <= 55

    def test_field_get_at_range(self):
        from uacpy.core.results import PressureField
        data = np.arange(100).reshape(10, 10).astype(float)
        ranges = np.linspace(0, 9000, 10)
        depths = np.linspace(0, 90, 10)

        field = PressureField(units="dB", data=data, ranges=ranges, depths=depths,
                        model='Test', frequencies=100.0)
        values = field.get_at_range(4500)
        assert len(values) == 10
        assert 50 <= values[5] <= 59

    def test_field_get_at_depth(self):
        from uacpy.core.results import PressureField
        data = np.arange(100).reshape(10, 10).astype(float)
        ranges = np.linspace(0, 9000, 10)
        depths = np.linspace(0, 90, 10)

        field = PressureField(units="dB", data=data, ranges=ranges, depths=depths,
                        model='Test', frequencies=100.0)
        values = field.get_at_depth(45)
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
