"""
Tests for I/O functions
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

import uacpy
from uacpy.io.file_manager import FileManager


class TestFileManager:
    """Tests for FileManager class."""

    def test_file_manager_creation(self):
        """Test creating FileManager."""
        fm = FileManager(use_tmpfs=False, base_dir=None, cleanup=True)
        assert fm is not None

    def test_file_manager_work_dir_creation(self):
        """Test creating work directory."""
        fm = FileManager(use_tmpfs=False, base_dir=None, cleanup=True)
        fm.create_work_dir()

        assert fm.work_dir is not None
        assert fm.work_dir.exists()
        assert fm.work_dir.is_dir()

        # Cleanup
        fm.cleanup_work_dir()

    def test_file_manager_get_path(self):
        """Test getting file path."""
        fm = FileManager(use_tmpfs=False, base_dir=None, cleanup=True)
        fm.create_work_dir()

        file_path = fm.get_path('test.txt')
        assert file_path.parent == fm.work_dir
        assert file_path.name == 'test.txt'

        # Cleanup
        fm.cleanup_work_dir()

    def test_file_manager_custom_base_dir(self):
        """Test FileManager with custom base directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fm = FileManager(use_tmpfs=False, base_dir=Path(tmpdir), cleanup=False)
            fm.create_work_dir()

            assert fm.work_dir.parent == Path(tmpdir)
            assert fm.work_dir.exists()

    def test_file_manager_cleanup(self):
        """Test FileManager cleanup."""
        fm = FileManager(use_tmpfs=False, base_dir=None, cleanup=True)
        fm.create_work_dir()

        work_dir = fm.work_dir
        assert work_dir.exists()

        fm.cleanup_work_dir()
        assert not work_dir.exists()


class TestEnvironmentIO:
    """Tests for Environment I/O."""

    def test_environment_with_bathymetry_file(self):
        """Test loading environment with bathymetry from file."""
        # Create temporary bathymetry file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("# Range(m) Depth(m)\n")
            f.write("0 80\n")
            f.write("5000 100\n")
            f.write("10000 120\n")
            bathy_file = f.name

        try:
            # Load bathymetry
            bathymetry = np.loadtxt(bathy_file)

            env = uacpy.Environment(
                name="Test",
                depth=100.0,
                sound_speed=1500.0,
                ssp_type='isovelocity',
                bathymetry=bathymetry
            )

            assert env.is_range_dependent
            assert len(env.bathymetry) == 3
            assert env.bathymetry[0, 1] == 80
            assert env.bathymetry[-1, 1] == 120

        finally:
            Path(bathy_file).unlink()

    def test_environment_ssp_from_array(self):
        """Test creating environment with SSP from array."""
        depths = np.linspace(0, 100, 11)
        sound_speeds = 1500 + depths * 0.1  # Linear gradient

        ssp_data = np.column_stack([depths, sound_speeds])

        env = uacpy.Environment(
            name="Test",
            depth=100.0,
            ssp_data=ssp_data,
            ssp_type='pchip'
        )

        assert len(env.ssp_data) == 11
        assert np.allclose(env.ssp_data[:, 0], depths)


class TestFieldIO:
    """Tests for Field I/O operations."""

    def test_field_metadata_preservation(self):
        """Test that Field preserves metadata."""
        from uacpy.core.field import Field

        metadata = {
            'model': 'Bellhop',
            'frequency': 100.0,
            'source_depth': 50.0,
            'custom_param': 'test_value'
        }

        field = Field(
            field_type='tl',
            data=np.random.rand(10, 20),
            ranges=np.linspace(100, 5000, 20),
            depths=np.linspace(10, 90, 10),
            metadata=metadata
        )

        assert field.metadata['model'] == 'Bellhop'
        assert field.metadata['frequency'] == 100.0
        assert field.metadata['custom_param'] == 'test_value'

    def test_field_copy_preserves_metadata(self):
        """Test that Field.copy() preserves metadata."""
        from uacpy.core.field import Field

        metadata = {'test_key': 'test_value'}

        field = Field(
            field_type='tl',
            data=np.random.rand(10, 20),
            ranges=np.linspace(100, 5000, 20),
            depths=np.linspace(10, 90, 10),
            metadata=metadata
        )

        field_copy = field.copy()

        assert field_copy.metadata['test_key'] == 'test_value'
        assert field_copy.metadata is not field.metadata


class TestDataValidation:
    """Tests for data validation."""

    def test_environment_depth_validation(self):
        """Test environment depth validation."""
        with pytest.raises(ValueError):
            uacpy.Environment(name="Test", depth=-10, sound_speed=1500, ssp_type='isovelocity')

    def test_source_receiver_depth_validation(self, simple_env):
        """Test source/receiver depth validation."""
        from uacpy.models.base import PropagationModel

        model = type('TestModel', (PropagationModel,), {
            'run': lambda self, *args, **kwargs: None
        })()

        # Source deeper than environment
        source_deep = uacpy.Source(depth=150, frequency=100)
        receiver = uacpy.Receiver(depths=[50], ranges=[1000])

        with pytest.raises(ValueError, match="Source depth.*exceeds"):
            model.validate_inputs(simple_env, source_deep, receiver)

        # Receiver deeper than environment
        source = uacpy.Source(depth=50, frequency=100)
        receiver_deep = uacpy.Receiver(depths=[150], ranges=[1000])

        with pytest.raises(ValueError, match="Receiver depth.*exceeds"):
            model.validate_inputs(simple_env, source, receiver_deep)

    def test_negative_depth_validation(self, simple_env):
        """Test negative depth validation."""
        # Negative source depth - should raise during construction
        with pytest.raises(ValueError, match="Source depths must be positive"):
            source_neg = uacpy.Source(depth=-10, frequency=100)

        # Negative receiver depth - should raise during construction
        with pytest.raises(ValueError, match="Receiver depths must be positive"):
            receiver_neg = uacpy.Receiver(depths=[-10], ranges=[1000])
