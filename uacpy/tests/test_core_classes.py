"""
Tests for core UACPY classes: Environment, Source, Receiver, Field
"""

import pytest
import numpy as np
import uacpy
from uacpy.core.results import Result as Field  # legacy alias


class TestEnvironment:
    """Tests for Environment class."""

    def test_create_simple_environment(self, simple_env):
        """Test creating a simple isovelocity environment."""
        assert simple_env.name == "Test Environment"
        assert simple_env.depth == 100.0
        assert simple_env.sound_speed == 1500.0
        assert simple_env.ssp_type == 'isovelocity'
        assert not simple_env.is_range_dependent

    def test_create_munk_environment(self, munk_env):
        """Test creating environment with Munk profile."""
        assert munk_env.name == "Munk Profile"
        assert munk_env.depth == 100.0
        assert len(munk_env.ssp_data) == 21
        assert munk_env.ssp_type == 'pchip'

    def test_range_dependent_environment(self, range_dependent_env):
        """Test range-dependent environment."""
        assert range_dependent_env.is_range_dependent
        assert len(range_dependent_env.bathymetry) == 11
        assert range_dependent_env.bathymetry[0, 1] == 80.0
        assert range_dependent_env.bathymetry[-1, 1] == 120.0

    def test_ssp_data_shape(self, simple_env, munk_env):
        """Test SSP data has correct shape."""
        assert simple_env.ssp_data.shape[1] == 2  # [depth, sound_speed]
        assert munk_env.ssp_data.shape[1] == 2

    def test_get_representative_depth(self, range_dependent_env):
        """Test getting representative depth for range-dependent environment."""
        median_depth = range_dependent_env.get_representative_depth('median')
        assert 80 <= median_depth <= 120

    def test_invalid_depth(self):
        """Test that negative depth raises error."""
        with pytest.raises(ValueError):
            uacpy.Environment(name="Test", depth=-10, sound_speed=1500, ssp_type='isovelocity')


class TestSource:
    """Tests for Source class."""

    def test_create_source(self, source):
        """Test creating a source."""
        assert source.depth[0] == 50.0
        assert source.frequency[0] == 100.0

    def test_source_array_conversion(self):
        """Test that single values are converted to arrays."""
        source = uacpy.Source(depth=30.0, frequency=200.0)
        assert isinstance(source.depth, np.ndarray)
        assert isinstance(source.frequency, np.ndarray)
        assert len(source.depth) == 1
        assert len(source.frequency) == 1

    def test_multiple_sources(self):
        """Test multiple source depths."""
        source = uacpy.Source(depth=[10.0, 20.0, 30.0], frequency=100.0)
        assert len(source.depth) == 3
        assert np.allclose(source.depth, [10, 20, 30])

    def test_multiple_frequencies(self):
        """Test multiple frequencies."""
        source = uacpy.Source(depth=50.0, frequency=[50.0, 100.0, 200.0])
        assert len(source.frequency) == 3


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
    """Tests for the typed Result hierarchy (TLField etc.)."""

    def test_create_tl_field(self):
        from uacpy.core.results import TLField
        data = np.random.rand(10, 20) * 50 + 40  # Random TL between 40-90 dB
        ranges = np.linspace(100, 5000, 20)
        depths = np.linspace(10, 90, 10)

        field = TLField(data=data, ranges=ranges, depths=depths,
                        model='Test', frequency=100.0)

        assert field.field_type == 'tl'
        assert field.shape == (10, 20)
        assert field.n_ranges == 20
        assert field.n_depths == 10

    def test_field_get_value(self):
        from uacpy.core.results import TLField
        data = np.arange(100).reshape(10, 10).astype(float)
        ranges = np.linspace(0, 9000, 10)
        depths = np.linspace(0, 90, 10)

        field = TLField(data=data, ranges=ranges, depths=depths,
                        model='Test', frequency=100.0)
        value = field.get_value(range_m=4500, depth=45)
        assert 44 <= value <= 55

    def test_field_get_at_range(self):
        from uacpy.core.results import TLField
        data = np.arange(100).reshape(10, 10).astype(float)
        ranges = np.linspace(0, 9000, 10)
        depths = np.linspace(0, 90, 10)

        field = TLField(data=data, ranges=ranges, depths=depths,
                        model='Test', frequency=100.0)
        values = field.get_at_range(4500)
        assert len(values) == 10
        assert 50 <= values[5] <= 59

    def test_field_get_at_depth(self):
        from uacpy.core.results import TLField
        data = np.arange(100).reshape(10, 10).astype(float)
        ranges = np.linspace(0, 9000, 10)
        depths = np.linspace(0, 90, 10)

        field = TLField(data=data, ranges=ranges, depths=depths,
                        model='Test', frequency=100.0)
        values = field.get_at_depth(45)
        assert len(values) == 10
        assert 40 <= values[5] <= 49

    def test_field_copy(self):
        from uacpy.core.results import TLField
        data = np.random.rand(10, 20)
        ranges = np.linspace(100, 5000, 20)
        depths = np.linspace(10, 90, 10)

        field = TLField(data=data, ranges=ranges, depths=depths,
                        model='Test', frequency=100.0)
        field_copy = field.copy()

        assert type(field_copy) is type(field)
        assert np.array_equal(field_copy.data, field.data)
        assert field_copy is not field
        assert field_copy.data is not field.data

    def test_invalid_field_type(self):
        """Constructing a TLField with mismatched shapes raises ValueError."""
        from uacpy.core.results import TLField
        with pytest.raises(ValueError):
            TLField(data=np.array([1.0, 2.0, 3.0]),
                    depths=np.array([1.0]), ranges=np.array([1.0]),
                    model='Test')

    def test_field_repr(self):
        from uacpy.core.results import TLField
        data = np.random.rand(10, 20)
        ranges = np.linspace(100, 5000, 20)
        depths = np.linspace(10, 90, 10)

        field = TLField(data=data, ranges=ranges, depths=depths,
                        model='Test', frequency=100.0)
        repr_str = repr(field)
        assert 'TLField' in repr_str
        assert field.shape == (10, 20)
        assert field.n_ranges == 20
        assert field.n_depths == 10
