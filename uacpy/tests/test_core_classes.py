"""
Tests for core UACPY classes: Environment, Source, Receiver, Field
"""

import pytest
import numpy as np
import uacpy
from uacpy.core.field import Field


class TestEnvironment:
    """Tests for Environment class"""

    def test_create_simple_environment(self, simple_env):
        """Test creating a simple isovelocity environment"""
        assert simple_env.name == "Test Environment"
        assert simple_env.depth == 100.0
        assert simple_env.sound_speed == 1500.0
        assert simple_env.ssp_type == 'isovelocity'
        assert not simple_env.is_range_dependent

    def test_create_munk_environment(self, munk_env):
        """Test creating environment with Munk profile"""
        assert munk_env.name == "Munk Profile"
        assert munk_env.depth == 100.0
        assert len(munk_env.ssp_data) == 21
        assert munk_env.ssp_type == 'pchip'

    def test_range_dependent_environment(self, range_dependent_env):
        """Test range-dependent environment"""
        assert range_dependent_env.is_range_dependent
        assert len(range_dependent_env.bathymetry) == 11
        assert range_dependent_env.bathymetry[0, 1] == 80.0
        assert range_dependent_env.bathymetry[-1, 1] == 120.0

    def test_ssp_data_shape(self, simple_env, munk_env):
        """Test SSP data has correct shape"""
        assert simple_env.ssp_data.shape[1] == 2  # [depth, sound_speed]
        assert munk_env.ssp_data.shape[1] == 2

    def test_get_representative_depth(self, range_dependent_env):
        """Test getting representative depth for range-dependent environment"""
        median_depth = range_dependent_env.get_representative_depth('median')
        assert 80 <= median_depth <= 120

    def test_invalid_depth(self):
        """Test that negative depth raises error"""
        with pytest.raises(ValueError):
            uacpy.Environment(name="Test", depth=-10, sound_speed=1500, ssp_type='isovelocity')


class TestSource:
    """Tests for Source class"""

    def test_create_source(self, source):
        """Test creating a source"""
        assert source.depth[0] == 50.0
        assert source.frequency[0] == 100.0

    def test_source_array_conversion(self):
        """Test that single values are converted to arrays"""
        source = uacpy.Source(depth=30.0, frequency=200.0)
        assert isinstance(source.depth, np.ndarray)
        assert isinstance(source.frequency, np.ndarray)
        assert len(source.depth) == 1
        assert len(source.frequency) == 1

    def test_multiple_sources(self):
        """Test multiple source depths"""
        source = uacpy.Source(depth=[10.0, 20.0, 30.0], frequency=100.0)
        assert len(source.depth) == 3
        assert np.allclose(source.depth, [10, 20, 30])

    def test_multiple_frequencies(self):
        """Test multiple frequencies"""
        source = uacpy.Source(depth=50.0, frequency=[50.0, 100.0, 200.0])
        assert len(source.frequency) == 3


class TestReceiver:
    """Tests for Receiver class"""

    def test_create_receiver_grid(self, receiver_grid):
        """Test creating receiver grid"""
        assert len(receiver_grid.depths) == 9
        assert len(receiver_grid.ranges) == 11
        assert receiver_grid.receiver_type == 'grid'
        assert receiver_grid.depth_min == 10.0
        assert receiver_grid.depth_max == 90.0
        assert receiver_grid.range_min == 100.0
        assert receiver_grid.range_max == 5000.0

    def test_small_receiver_grid(self, receiver_small):
        """Test small receiver grid"""
        assert len(receiver_small.depths) == 3
        assert len(receiver_small.ranges) == 3

    def test_receiver_line_array(self):
        """Test line array receiver"""
        receiver = uacpy.Receiver(
            depths=[50.0],
            ranges=np.linspace(1000, 10000, 100)
        )
        assert len(receiver.depths) == 1
        assert len(receiver.ranges) == 100


class TestField:
    """Tests for Field class"""

    def test_create_tl_field(self):
        """Test creating transmission loss field"""
        data = np.random.rand(10, 20) * 50 + 40  # Random TL between 40-90 dB
        ranges = np.linspace(100, 5000, 20)
        depths = np.linspace(10, 90, 10)

        field = Field(
            field_type='tl',
            data=data,
            ranges=ranges,
            depths=depths
        )

        assert field.field_type == 'tl'
        assert field.shape == (10, 20)
        assert field.n_ranges == 20
        assert field.n_depths == 10

    def test_field_get_value(self):
        """Test getting value at specific location"""
        data = np.arange(100).reshape(10, 10)
        ranges = np.linspace(0, 9000, 10)
        depths = np.linspace(0, 90, 10)

        field = Field('tl', data, ranges, depths)

        # Get value at closest point (data is row-major: depth x range)
        value = field.get_value(range_m=4500, depth=45)
        # depth=45 is index 5, range=4500 is index 5, so data[5,5] = 55
        # But actual closest indices may vary slightly
        assert 44 <= value <= 55  # Should be near middle of grid

    def test_field_get_at_range(self):
        """Test getting field values at specific range"""
        data = np.arange(100).reshape(10, 10)
        ranges = np.linspace(0, 9000, 10)
        depths = np.linspace(0, 90, 10)

        field = Field('tl', data, ranges, depths)

        values = field.get_at_range(4500)
        assert len(values) == 10
        # Values should be from the closest range column
        assert 50 <= values[5] <= 59  # Middle depth, near middle range

    def test_field_get_at_depth(self):
        """Test getting field values at specific depth"""
        data = np.arange(100).reshape(10, 10)
        ranges = np.linspace(0, 9000, 10)
        depths = np.linspace(0, 90, 10)

        field = Field('tl', data, ranges, depths)

        values = field.get_at_depth(45)
        assert len(values) == 10
        # Values should be from the closest depth row
        assert 40 <= values[5] <= 49  # Near middle depth, middle range

    def test_field_copy(self):
        """Test field copy"""
        data = np.random.rand(10, 20)
        ranges = np.linspace(100, 5000, 20)
        depths = np.linspace(10, 90, 10)

        field = Field('tl', data, ranges, depths)
        field_copy = field.copy()

        assert field_copy.field_type == field.field_type
        assert np.array_equal(field_copy.data, field.data)
        assert field_copy is not field
        assert field_copy.data is not field.data

    def test_invalid_field_type(self):
        """Test that invalid field type raises error"""
        with pytest.raises(ValueError):
            Field('invalid_type', np.array([1, 2, 3]))

    def test_field_repr(self):
        """Test field string representation"""
        data = np.random.rand(10, 20)
        ranges = np.linspace(100, 5000, 20)
        depths = np.linspace(10, 90, 10)

        field = Field('tl', data, ranges, depths)
        repr_str = repr(field)

        assert 'tl' in repr_str
        assert '10x20' in repr_str
        assert '20 ranges' in repr_str
        assert '10 depths' in repr_str
