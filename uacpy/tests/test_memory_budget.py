"""models/_budget.py: one memory policy for every engine's estimate
(M-14, decision 40) — over half the free memory announced, over all of it
refused, an unreadable host announced above 2 GiB and never refused — and
the memory-backed test a work directory's files go through."""
import tempfile
from pathlib import Path

import pytest

from uacpy.core.exceptions import ConfigurationError
from uacpy.models import _budget
from uacpy.models._budget import (
    HEADROOM_FRACTION, UNREADABLE_HOST_BYTES, filesystem_type, memory_budget,
    work_dir_is_memory_backed,
)
from uacpy.models._workspace import FileManager, ScratchPolicy


def _weigh(monkeypatch, free, n_bytes):
    monkeypatch.setattr(_budget, 'available_memory_bytes', lambda: free)
    return memory_budget(n_bytes, model_name='Engine', what='table',
                         detail='The table is big.', remediation='Shrink it.')


class TestTheBudgetWeighsAgainstTheFreeMemory:

    FREE = 8 * 1024 ** 3

    def test_the_thresholds_are_half_the_free_memory_and_2_gib(
            self, monkeypatch):
        """Decision 40's values, in literal bytes: 4 GiB of 8 GiB free is
        silent and one byte more is announced; 2 GiB on an unreadable host
        is silent and one byte more is announced."""
        assert _weigh(monkeypatch, self.FREE, 4 * 1024 ** 3) is None
        assert _weigh(monkeypatch, self.FREE, 4 * 1024 ** 3 + 1) is not None
        assert _weigh(monkeypatch, None, 2 * 1024 ** 3) is None
        assert _weigh(monkeypatch, None, 2 * 1024 ** 3 + 1) is not None

    def test_half_the_free_memory_is_silent(self, monkeypatch):
        assert _weigh(monkeypatch, self.FREE,
                      int(HEADROOM_FRACTION * self.FREE)) is None

    def test_one_byte_over_half_is_announced(self, monkeypatch):
        notice = _weigh(monkeypatch, self.FREE,
                        int(HEADROOM_FRACTION * self.FREE) + 1)
        assert 'over half' in notice.note
        assert notice.message.startswith('Engine: The table is big.')
        assert 'headroom' in notice.message
        assert notice.message.endswith('Shrink it.')

    def test_all_of_the_free_memory_is_announced_not_refused(
            self, monkeypatch):
        assert _weigh(monkeypatch, self.FREE, self.FREE) is not None

    def test_one_byte_over_the_free_memory_is_refused(self, monkeypatch):
        with pytest.raises(ConfigurationError,
                           match='more than the 8.0 GiB') as exc:
            _weigh(monkeypatch, self.FREE, self.FREE + 1)
        assert 'The table is big.' in str(exc.value)
        assert exc.value.remediation == 'Shrink it.'


    def test_an_upper_bound_over_the_free_memory_is_announced(
            self, monkeypatch):
        monkeypatch.setattr(_budget, 'available_memory_bytes',
                            lambda: self.FREE)
        notice = memory_budget(self.FREE + 1, model_name='Engine',
                               what='file', detail='The file is big.',
                               remediation='Shrink it.', upper_bound=True)
        assert 'up to' in notice.note and 'not refused' in notice.message
        assert memory_budget(
            int(HEADROOM_FRACTION * self.FREE), model_name='Engine',
            what='file', detail='d', remediation='r', upper_bound=True) is None


class TestAnUnreadableHostIsAnnouncedNeverRefused:

    def test_the_cap_itself_is_silent(self, monkeypatch):
        assert _weigh(monkeypatch, None, UNREADABLE_HOST_BYTES) is None

    def test_one_byte_over_the_cap_is_announced(self, monkeypatch):
        notice = _weigh(monkeypatch, None, UNREADABLE_HOST_BYTES + 1)
        assert 'cannot be read' in notice.message
        assert 'not refused' in notice.message
        assert notice.message.endswith('Shrink it.')

    def test_any_size_is_announced_not_refused(self, monkeypatch):
        assert _weigh(monkeypatch, None,
                      1000 * UNREADABLE_HOST_BYTES) is not None


class TestTheFilesystemIsReadFromTheMountTable:

    @staticmethod
    def _table(tmp_path, rows):
        path = tmp_path / 'mounts'
        path.write_text(''.join(f'{dev} {mnt} {kind} rw 0 0\n'
                                for dev, mnt, kind in rows))
        return path

    def test_the_longest_mount_point_decides(self, tmp_path):
        table = self._table(tmp_path, [('/dev/sda1', '/', 'ext4'),
                                       ('tmpfs', '/uacpy_no_dir', 'tmpfs')])
        assert filesystem_type('/uacpy_no_dir/x/a.mod',
                               mount_table=table) == 'tmpfs'
        assert filesystem_type('/uacpy_no_dirx/a.mod',
                               mount_table=table) == 'ext4'

    def test_of_two_mounts_on_one_point_the_later_is_in_effect(
            self, tmp_path):
        table = self._table(tmp_path, [('/dev/sda1', '/', 'ext4'),
                                       ('/dev/sdb1', '/uacpy_no_dir', 'ext4'),
                                       ('tmpfs', '/uacpy_no_dir', 'tmpfs')])
        assert filesystem_type('/uacpy_no_dir/run',
                               mount_table=table) == 'tmpfs'

    def test_an_escaped_space_in_a_mount_point_is_decoded(self, tmp_path):
        table = self._table(tmp_path, [('/dev/sda1', '/', 'ext4'),
                                       ('tmpfs', r'/uacpy_no\040dir',
                                        'tmpfs')])
        assert filesystem_type('/uacpy_no dir/run',
                               mount_table=table) == 'tmpfs'

    def test_an_unreadable_table_answers_none(self, tmp_path):
        assert filesystem_type('/', mount_table=tmp_path / 'none') is None


class TestTheWorkDirectoryIsTheOneTheRunWritesInto:

    @staticmethod
    def _seen(monkeypatch, kind='ext4'):
        seen = []

        def fake(path, **kwargs):
            seen.append(Path(path))
            return kind
        monkeypatch.setattr(_budget, 'filesystem_type', fake)
        return seen

    def test_a_pinned_work_dir_is_weighed(self, monkeypatch, tmp_path):
        # Not tmp_path itself: the suite points the system temp directory
        # there, which would hide which of the two was weighed.
        seen = self._seen(monkeypatch, 'tmpfs')
        assert work_dir_is_memory_backed(
            ScratchPolicy(pinned_dir=tmp_path / 'wd', keeps_files=True,
                          on_tmpfs=False))
        assert seen == [tmp_path / 'wd']
        assert Path(tempfile.gettempdir()) != tmp_path / 'wd'

    def test_use_tmpfs_weighs_the_tmpfs_root(self, monkeypatch):
        seen = self._seen(monkeypatch)
        monkeypatch.setattr(FileManager, 'dev_shm_usable',
                            classmethod(lambda cls: True))
        assert not work_dir_is_memory_backed(
            ScratchPolicy(pinned_dir=None, keeps_files=False, on_tmpfs=True))
        assert seen == [FileManager.DEV_SHM_ROOT]

    def test_otherwise_the_system_temp_directory_is_weighed(
            self, monkeypatch):
        seen = self._seen(monkeypatch)
        work_dir_is_memory_backed(
            ScratchPolicy(pinned_dir=None, keeps_files=False,
                          on_tmpfs=False))
        assert seen == [Path(tempfile.gettempdir())]

    @pytest.mark.parametrize('kind, backed', [('tmpfs', True),
                                              ('ramfs', True),
                                              ('ext4', False),
                                              (None, False)])
    def test_only_tmpfs_and_ramfs_are_memory(self, monkeypatch, tmp_path,
                                              kind, backed):
        self._seen(monkeypatch, kind)
        assert work_dir_is_memory_backed(
            ScratchPolicy(pinned_dir=tmp_path, keeps_files=True,
                          on_tmpfs=False)) is backed
