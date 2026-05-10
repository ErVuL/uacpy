"""
Tests for I/O functions
"""

import numpy as np
from pathlib import Path
import tempfile

import uacpy
from uacpy.core.environment import SoundSpeedProfile
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
                ssp=1500.0,
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
            bathymetry=100.0,
            ssp=SoundSpeedProfile.from_pairs(ssp_data, interp='pchip')
        )

        assert len(env.ssp.to_pairs()) == 11
        assert np.allclose(env.ssp.to_pairs()[:, 0], depths)


class TestFieldIO:
    """Tests for Field I/O operations."""

    def test_field_metadata_preservation(self):
        """Result subclasses preserve metadata + identification kwargs."""
        from uacpy.core.results import PressureField
        field = PressureField(units="dB",
                              data=np.random.rand(10, 20),
                              ranges=np.linspace(100, 5000, 20),
                              depths=np.linspace(10, 90, 10),
                              model='Bellhop',
                              frequencies=100.0,
                              metadata={'source_depth': 50.0, 'custom_param': 'test_value'},
                              )
        # ``model`` and ``frequencies`` are typed attributes; everything else
        # lives in metadata. Both are also mirrored into metadata.
        assert field.model == 'Bellhop'
        assert field.f0 == 100.0
        assert list(field.frequencies) == [100.0]
        assert field.metadata['model'] == 'Bellhop'
        assert list(field.metadata['frequencies']) == [100.0]
        assert field.metadata['custom_param'] == 'test_value'

    def test_field_copy_preserves_metadata(self):
        """Result.copy() returns a deep copy with metadata preserved."""
        from uacpy.core.results import PressureField
        field = PressureField(units="dB",
                              data=np.random.rand(10, 20),
                              ranges=np.linspace(100, 5000, 20),
                              depths=np.linspace(10, 90, 10),
                              model='Bellhop',
                              frequencies=100.0,
                              metadata={'test_key': 'test_value'},
                              )
        field_copy = field.copy()
        assert field_copy.metadata['test_key'] == 'test_value'
        assert field_copy.metadata is not field.metadata
