"""``FileManager``: the work directory, its ownership, and its cleanup."""

import os
import pytest
import tempfile
from pathlib import Path
from uacpy.core.exceptions import ConfigurationError
from uacpy.models._workspace import FileManager


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

    def test_a_base_dir_that_does_not_exist_is_refused(self, tmp_path):
        with pytest.raises(ConfigurationError,
                           match='Base directory does not exist'):
            FileManager(use_tmpfs=False, base_dir=tmp_path / 'absent')

    @pytest.mark.skipif(os.geteuid() == 0,
                        reason='root writes to a read-only directory anyway')
    def test_a_base_dir_that_is_not_writable_is_refused(self, tmp_path):
        base = tmp_path / 'ro'
        base.mkdir()
        base.chmod(0o500)
        try:
            with pytest.raises(ConfigurationError,
                               match='Base directory not writable'):
                FileManager(use_tmpfs=False, base_dir=base)
        finally:
            base.chmod(0o755)


class TestUserWorkDirIsNeverDestroyed:
    """``cleanup`` removes uacpy's scratch. A work_dir the *caller* supplied is
    not uacpy's to delete: neither the directory itself nor anything that was
    already in it."""

    @staticmethod
    def _seed(tmp_path):
        d = tmp_path / 'mine'
        d.mkdir()
        (d / 'PRECIOUS.txt').write_text('do not delete')
        return d

    @pytest.mark.requires_binary
    def test_pinned_work_dir_survives_cleanup_true(self, tmp_path):
        import uacpy
        d = self._seed(tmp_path)
        m = uacpy.Bellhop(work_dir=str(d), cleanup=True, verbose=False)
        fm = m._setup_file_manager()
        fm.get_path('scratch.env').write_text('x')
        fm.cleanup_work_dir()
        assert d.exists(), "uacpy deleted the caller's directory"
        assert (d / 'PRECIOUS.txt').exists(), "uacpy deleted a pre-existing file"
        assert not (d / 'scratch.env').exists(), "uacpy left its own scratch behind"

    @pytest.mark.requires_binary
    def test_copy_onto_a_pinned_work_dir_survives(self, tmp_path):
        """Model(cleanup=True).copy(work_dir=d) — the _cleanup_explicit path,
        where the caller's True rides onto a directory they only named later."""
        import uacpy
        d = self._seed(tmp_path)
        m = uacpy.Bellhop(cleanup=True, verbose=False).copy(work_dir=str(d))
        fm = m._setup_file_manager()
        fm.get_path('scratch.env').write_text('x')
        fm.cleanup_work_dir()
        assert d.exists() and (d / 'PRECIOUS.txt').exists()

    def test_uacpy_owned_temp_dir_is_fully_removed(self):
        from uacpy.models._workspace import FileManager
        fm = FileManager(use_tmpfs=False, base_dir=None, cleanup=True)
        wd = fm.create_work_dir()
        fm.get_path('scratch.env').write_text('x')
        assert wd.exists()
        fm.cleanup_work_dir()
        assert not wd.exists(), "a uacpy-created temp dir must be removed whole"


class TestFileManagerStateMatchesTheDirectoryUsed:

    def test_base_dir_overrides_a_tmpfs_request(self, tmp_path):
        from uacpy.models._workspace import FileManager
        fm = FileManager(use_tmpfs=True, base_dir=tmp_path)
        assert fm.use_tmpfs is False
        assert 'disk' in repr(fm)

    def test_tmpfs_fallback_reports_disk(self, monkeypatch):
        from uacpy.models._workspace import FileManager
        monkeypatch.setattr(FileManager, 'dev_shm_usable',
                            classmethod(lambda cls: False))
        fm = FileManager(use_tmpfs=True)
        assert fm.use_tmpfs is False
        assert 'disk' in repr(fm)

    def test_available_tmpfs_reports_tmpfs(self):
        from uacpy.models._workspace import FileManager
        if not FileManager.dev_shm_usable():
            pytest.skip('/dev/shm unavailable')
        fm = FileManager(use_tmpfs=True)
        assert fm.use_tmpfs is True
        assert fm.base_dir == Path('/dev/shm')


class TestCleanupWorkDirForgetsOwnershipState:

    def test_cleanup_of_an_adopted_dir_drops_the_snapshot(self, tmp_path):
        from uacpy.models._workspace import FileManager
        fm = FileManager(base_dir=tmp_path)
        adopted = tmp_path / 'caller_dir'
        adopted.mkdir()
        (adopted / 'keep.txt').write_text('x')
        fm.adopt_work_dir(adopted)
        assert fm._preexisting == {'keep.txt'}
        fm.cleanup_work_dir()
        assert fm.work_dir is None
        assert fm._owns_work_dir is False
        assert fm._preexisting is None
        assert (adopted / 'keep.txt').exists()

    def test_cleanup_of_an_owned_dir_drops_ownership(self, tmp_path):
        from uacpy.models._workspace import FileManager
        fm = FileManager(base_dir=tmp_path)
        fm.create_work_dir()
        assert fm._owns_work_dir is True
        fm.cleanup_work_dir()
        assert fm.work_dir is None
        assert fm._owns_work_dir is False
        assert fm._preexisting is None


class TestFileManagerReportsWorkDirMisuseTypedly:
    """``models/base.py`` hands the user's ``work_dir=`` straight to
    ``adopt_work_dir``, so ``Bellhop(work_dir='some_file.txt')`` reaches
    ``mkdir`` on a regular file. And the manager tracks exactly one directory:
    a second ``create_work_dir`` would drop the only reference to the first
    and leave it on disk with nothing able to remove it."""

    def test_adopting_a_path_that_is_a_file_raises_configurationerror(
            self, tmp_path):
        from uacpy.models._workspace import FileManager
        target = tmp_path / 'notadir.txt'
        target.write_text('x')
        with pytest.raises(ConfigurationError, match='not a directory'):
            FileManager(base_dir=tmp_path).adopt_work_dir(target)

    def test_adopting_a_missing_directory_creates_and_returns_it(
            self, tmp_path):
        from uacpy.models._workspace import FileManager
        target = tmp_path / 'fresh'
        adopted = FileManager(base_dir=tmp_path).adopt_work_dir(target)
        assert adopted == target and target.is_dir()

    def test_a_second_create_work_dir_raises_rather_than_stranding_the_first(
            self, tmp_path):
        from uacpy.models._workspace import FileManager
        fm = FileManager(base_dir=tmp_path)
        first = fm.create_work_dir()
        with pytest.raises(ConfigurationError, match='already holds work_dir'):
            fm.create_work_dir()
        assert fm.work_dir == first
        fm.cleanup_work_dir()
        assert not first.exists()

    def test_entering_the_context_reuses_a_directory_already_created(
            self, tmp_path):
        from uacpy.models._workspace import FileManager
        fm = FileManager(base_dir=tmp_path, cleanup=True)
        first = fm.create_work_dir()
        with fm as entered:
            assert entered.work_dir == first
        assert not first.exists()
        assert [p for p in tmp_path.iterdir()] == []
