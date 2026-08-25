"""What a model run does when the install or the work directory is broken.

Each class below is a reproduced failure: a real thing that goes wrong on a
real install, reaching the caller as a raw OS exception, as a message pointing
at the wrong file, or as another error entirely because cleanup threw on the
way out.

**F24-5** a binary that is present but will not exec — the execute bit lost in
transit, an unexpanded LFS pointer, a wrong-architecture build, a directory
where the file should be — escaped as ``PermissionError`` or ``OSError
[Errno 8]``, so ``except UACPYError`` missed it entirely.

**F24-6** a work directory deleted mid-run was announced as "Executable not
found: <the work dir>", because CPython sets ``filename`` to the cwd when the
child's chdir fails; the user goes hunting for a binary that is fine.

**F24-3** cleanup failures replaced the model's own error. ``Path.exists()``
re-raises EACCES on 3.13, and both ``read_prt`` and ``cleanup_work_dir`` call
it outside the ``try`` their docstrings promise covers it — so a
``ModelExecutionError`` in flight was demoted to ``__context__``, or (via
``_attach_prt_tail``, which runs before ``raise exc``) lost outright.

**F24-7** a read-only work directory died with a raw ``PermissionError`` at
deck-write time, because the typed writability check validated the work
directory's PARENT rather than the directory actually written.
"""

import errno
import os
import stat
import threading

import pytest

from uacpy.core import Environment, Receiver, Source
from uacpy.core.exceptions import (
    ConfigurationError,
    ExecutableNotFoundError,
    ModelExecutionError,
)
from uacpy.io import file_manager as file_manager_module
from uacpy.io.oalib_reader import read_prt
from uacpy.models import RAM
from uacpy.models import base as base_module


LFS_POINTER = ('version https://git-lfs.github.com/spec/v1\n'
               'oid sha256:0000000000000000000000000000000000000000000000000'
               '000000000000000\nsize 1234567\n')


@pytest.mark.requires_binary  # constructs RAM (resolves its binary)
class TestUnrunnableBinaryIsTyped:
    """Every shape of broken install lands as ``ExecutableNotFoundError``."""

    @staticmethod
    def _model():
        return RAM(verbose=False, timeout=30.0)

    def test_a_file_without_the_execute_bit_is_typed(self, tmp_path):
        dud = tmp_path / 'ramgeo'
        dud.write_bytes(b'\x7fELF' + b'\x00' * 64)
        dud.chmod(0o644)
        with pytest.raises(ExecutableNotFoundError) as excinfo:
            self._model()._run_subprocess([str(dud)], cwd=tmp_path)
        assert str(dud) in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, PermissionError)

    def test_an_lfs_pointer_with_the_execute_bit_is_typed(self, tmp_path):
        dud = tmp_path / 'ramgeo'
        dud.write_text(LFS_POINTER)
        dud.chmod(0o755)
        with pytest.raises(ExecutableNotFoundError) as excinfo:
            self._model()._run_subprocess([str(dud)], cwd=tmp_path)
        assert str(dud) in str(excinfo.value)
        assert excinfo.value.__cause__.errno == errno.ENOEXEC

    def test_a_directory_where_the_binary_belongs_is_typed(self, tmp_path):
        dud = tmp_path / 'ramgeo'
        dud.mkdir()
        with pytest.raises(ExecutableNotFoundError) as excinfo:
            self._model()._run_subprocess([str(dud)], cwd=tmp_path)
        assert str(dud) in str(excinfo.value)

    def test_the_resolver_refuses_a_non_executable_pinned_path(self, tmp_path):
        dud = tmp_path / 'ramgeo'
        dud.write_bytes(b'\x7fELF' + b'\x00' * 64)
        dud.chmod(0o644)
        with pytest.raises(ExecutableNotFoundError, match='cannot be run'):
            RAM(backend='ramgeo', executable=str(dud), verbose=False)

    def test_the_resolver_refuses_a_directory(self, tmp_path):
        dud = tmp_path / 'ramgeo'
        dud.mkdir()
        with pytest.raises(ExecutableNotFoundError, match='cannot be run'):
            RAM(backend='ramgeo', executable=str(dud), verbose=False)

    def test_a_missing_path_raises_executable_not_found(self, tmp_path):
        with pytest.raises(ExecutableNotFoundError, match='not found'):
            RAM(backend='ramgeo', executable=str(tmp_path / 'absent'),
                verbose=False)

    def test_a_dud_candidate_does_not_shadow_a_working_one(self, tmp_path,
                                                           monkeypatch):
        """The search walks several install locations; a leftover text file in
        an early one must not be selected over the real build behind it."""
        model = self._model()      # built before _PACKAGE_DIR is redirected
        for sub in ('first', 'second'):
            (tmp_path / 'bin' / sub).mkdir(parents=True)
        dud = tmp_path / 'bin' / 'first' / 'probe'
        dud.write_text(LFS_POINTER)
        dud.chmod(0o644)
        good = tmp_path / 'bin' / 'second' / 'probe'
        good.write_text('#!/bin/sh\nexit 0\n')
        good.chmod(0o755)

        monkeypatch.setattr(base_module, '_PACKAGE_DIR', tmp_path)
        found = model._find_executable_in_paths(
            'probe', bin_subdirs=['first', 'second'])
        assert found == good


@pytest.mark.requires_binary  # constructs RAM (resolves its binary)
class TestVanishedWorkDirIsNamedAsSuch:
    """A chdir failure names the work directory, not a missing binary."""

    @staticmethod
    def _exe():
        return str(RAM(backend='ramgeo', verbose=False)._collins_binary('ramgeo'))

    def test_a_deleted_work_dir_is_reported_as_the_work_dir(self, tmp_path):
        gone = tmp_path / 'work'
        gone.mkdir()
        gone.rmdir()
        with pytest.raises(ModelExecutionError) as excinfo:
            RAM(verbose=False, timeout=30.0)._run_subprocess(
                [self._exe()], cwd=gone)
        message = str(excinfo.value)
        assert 'Work directory' in message, message
        assert 'Executable not found' not in message, message


class TestCleanupFailuresDoNotMaskTheModelError:
    """``read_prt`` and ``cleanup_work_dir`` swallow their own OS failures."""

    @staticmethod
    def _locked_dir(tmp_path):
        """A directory whose search bit is cleared, so ``Path.exists()`` on
        anything inside and ``iterdir()`` on the directory itself raise
        EACCES — what a run leaves behind when the enclosing mount goes
        read-protected, or a stale NFS handle, mid-run."""
        locked = tmp_path / 'locked'
        locked.mkdir()
        (locked / 'run.prt').write_text('tail\n')
        locked.chmod(0o000)
        return locked

    def _adopted(self, tmp_path):
        """A FileManager that adopted a directory which then became
        unreadable — the ordering a real run sees, since ``adopt_work_dir``
        snapshots the directory while it is still readable."""
        locked = tmp_path / 'locked'
        locked.mkdir()
        (locked / 'run.prt').write_text('tail\n')
        fm = file_manager_module.FileManager(
            base_dir=str(tmp_path), cleanup=True, prefix='r24_')
        fm.adopt_work_dir(locked)
        locked.chmod(0o000)
        return fm, locked

    @pytest.mark.skipif(os.geteuid() == 0,
                        reason='root ignores the directory search bit')
    def test_read_prt_returns_none_when_the_file_is_unreadable(self, tmp_path):
        locked = self._locked_dir(tmp_path)
        try:
            assert read_prt(locked / 'run.prt') is None
        finally:
            locked.chmod(0o755)

    @pytest.mark.skipif(os.geteuid() == 0,
                        reason='root ignores the directory search bit')
    def test_cleanup_work_dir_swallows_an_unreadable_work_dir(self, tmp_path):
        fm, locked = self._adopted(tmp_path)
        try:
            fm.cleanup_work_dir()
        finally:
            locked.chmod(0o755)

    @pytest.mark.skipif(os.geteuid() == 0,
                        reason='root ignores the directory search bit')
    def test_the_model_error_survives_cleanup(self, tmp_path):
        """The whole point: an exception raised while the run is in flight
        reaches the caller intact, not replaced by the cleanup's EACCES."""
        fm, locked = self._adopted(tmp_path)
        try:
            with pytest.raises(ModelExecutionError):
                try:
                    raise ModelExecutionError('RAM', return_code=1,
                                              stderr='boom')
                finally:
                    fm.cleanup_work_dir()
        finally:
            locked.chmod(0o755)

    @pytest.mark.skipif(os.geteuid() == 0,
                        reason='root ignores the directory search bit')
    def test_cleanup_hands_back_the_claim_when_it_cannot_read(
            self, tmp_path):
        """Swallowing the OSError must not cost the release hooks their run —
        those are what free a pinned work_dir for the next caller."""
        fm, locked = self._adopted(tmp_path)
        released = []
        fm.on_release(lambda: released.append(True))
        try:
            fm.cleanup_work_dir()
        finally:
            locked.chmod(0o755)
        assert released == [True]


@pytest.mark.requires_binary  # constructs RAM (resolves its binary)
class TestReadOnlyWorkDirIsTyped:
    """A work directory the user pinned but cannot be written to.

    The FileManager's typed writability check validated ``base_dir``, which
    for an adopted work directory is its PARENT — writable in every realistic
    case — so the refusal fell through to a raw ``PermissionError`` from the
    first deck write.
    """

    @pytest.mark.skipif(os.geteuid() == 0,
                        reason='root writes to a read-only directory anyway')
    def test_a_read_only_work_dir_raises_configuration_error(self, tmp_path):
        work = tmp_path / 'ro'
        work.mkdir()
        work.chmod(stat.S_IRUSR | stat.S_IXUSR)
        try:
            model = RAM(verbose=False, backend='ramgeo',
                        work_dir=str(work), cleanup=False)
            with pytest.raises(ConfigurationError) as excinfo:
                model.run(
                    Environment(name='ro', bathymetry=100.0, ssp=1500.0),
                    Source(depths=50.0, frequencies=50.0),
                    Receiver(depths=[50.0], ranges=[1000.0]),
                )
            assert str(work) in str(excinfo.value)
        finally:
            work.chmod(0o755)


class TestFinishHandsBackThePinnedClaim:
    """**F24-2** — a hole in the round-23 claim machinery.

    A pinned ``work_dir`` makes ``cleanup`` default to False (``base.py``), so
    the old ``finally: if fm.cleanup: fm.cleanup_work_dir()`` skipped the only
    deterministic release and left it to the weakref finalizer. But the
    FileManager is a local of ``run()``, kept alive by the raised exception's
    traceback — so any caller that retains the exception (``errors.append(e)``,
    ``pytest.raises``) pins the claim, and the next *thread* to use that
    directory is refused with "still running".

    Narrow but real: it needs a pinned work_dir, cleanup False, a raised run,
    a retained exception, and a different live thread. A same-thread retry is
    re-entrant and allowed; ``parallel.py`` pre-checks the multithreaded route.
    ``FileManager.finish()`` closes it by deciding internally: remove the files
    only when asked, hand back the claim always.
    """

    @staticmethod
    def _failing_model(work_dir, monkeypatch):
        def boom(model_self, cmd, cwd, **kwargs):
            raise ModelExecutionError('RAM', return_code=1, stderr='simulated')

        monkeypatch.setattr(RAM, '_run_subprocess', boom)
        return RAM(verbose=False, backend='ramgeo', work_dir=str(work_dir),
                   cleanup=False)

    def test_finish_runs_the_release_hooks_when_cleanup_is_off(self, tmp_path):
        released = []
        fm = file_manager_module.FileManager(
            base_dir=str(tmp_path), cleanup=False, prefix='r24_')
        work = fm.create_work_dir()
        (work / 'kept.txt').write_text('x')
        fm.on_release(lambda: released.append(True))
        fm.finish()
        assert released == [True]
        assert (work / 'kept.txt').exists(), "cleanup=False must keep the files"

    def test_finish_removes_the_files_when_cleanup_is_on(self, tmp_path):
        released = []
        fm = file_manager_module.FileManager(
            base_dir=str(tmp_path), cleanup=True, prefix='r24_')
        work = fm.create_work_dir()
        (work / 'scratch.txt').write_text('x')
        fm.on_release(lambda: released.append(True))
        fm.finish()
        assert released == [True]
        assert not work.exists()

    @pytest.mark.requires_binary  # constructs RAM (resolves its binary)
    def test_a_retained_failure_does_not_hold_the_claim(self, tmp_path,
                                                        monkeypatch):
        work = tmp_path / 'pinned'
        work.mkdir()
        model = self._failing_model(work, monkeypatch)
        retained = []
        try:
            model.run(Environment(name='leak', bathymetry=100.0, ssp=1500.0),
                      Source(depths=50.0, frequencies=50.0),
                      Receiver(depths=[50.0], ranges=[1000.0]))
        except ModelExecutionError as exc:
            retained.append(exc)
        assert retained, "the stubbed run was supposed to raise"
        assert str(work.resolve()) not in base_module._PINNED_WORK_DIRS

    @pytest.mark.requires_binary  # constructs RAM (resolves its binary)
    def test_another_thread_may_use_the_dir_after_a_retained_failure(
            self, tmp_path, monkeypatch):
        """The user-visible half: the refusal message asserted the first
        thread was 'still running', which it no longer was."""
        work = tmp_path / 'pinned'
        work.mkdir()
        model = self._failing_model(work, monkeypatch)
        env = Environment(name='leak', bathymetry=100.0, ssp=1500.0)
        src = Source(depths=50.0, frequencies=50.0)
        rcv = Receiver(depths=[50.0], ranges=[1000.0])

        retained = []
        try:
            model.run(env, src, rcv)
        except ModelExecutionError as exc:
            retained.append(exc)

        second = {}

        def worker():
            try:
                self._failing_model(work, monkeypatch).run(env, src, rcv)
            except BaseException as exc:  # noqa: BLE001 - classifying it
                second['error'] = exc

        thread = threading.Thread(target=worker, name='R24-Worker')
        thread.start()
        thread.join(timeout=30)

        assert retained and 'error' in second
        assert not isinstance(second['error'], ConfigurationError), (
            f"second thread was refused the released directory: "
            f"{second['error']}")
        assert isinstance(second['error'], ModelExecutionError)


@pytest.mark.requires_binary  # constructs RAM (resolves its binary)
class TestABorrowedFileManagerOutlivesOneFrequency:
    """``_run_collins_one_freq`` finishes the manager only when it made it.

    The broadband sweep hands one manager down for the whole band, so a
    per-frequency ``finish()`` would free a directory the sweep is still
    marching through — the trap in rewriting ``if owns_fm and fm.cleanup:``.
    """

    def test_a_caller_supplied_manager_is_left_alone(self, tmp_path,
                                                     monkeypatch):
        def boom(model_self, cmd, cwd, **kwargs):
            raise ModelExecutionError('RAM', return_code=1, stderr='simulated')

        monkeypatch.setattr(RAM, '_run_subprocess', boom)
        released = []
        fm = file_manager_module.FileManager(
            base_dir=str(tmp_path), cleanup=True, prefix='r24_')
        work = fm.create_work_dir()
        fm.on_release(lambda: released.append(True))

        model = RAM(verbose=False, backend='ramgeo', dr=20.0, dz=2.0)
        with pytest.raises(ModelExecutionError):
            model._run_collins_one_freq(
                Environment(name='band', bathymetry=100.0, ssp=1500.0),
                Source(depths=50.0, frequencies=50.0),
                Receiver(depths=[50.0], ranges=[1000.0]),
                kind='ramgeo', freq=50.0, theta=0.0, fm=fm)

        assert work.exists(), "the sweep's own work dir was removed under it"
        assert released == []


class TestSignalDeathIsNamed:
    """**F24-10** — a signalled binary reported only "exit code: -9".

    ``subprocess`` encodes a signalled child as ``-signum``, and -9 is the
    shape an out-of-memory kill takes: the most likely real cause of a big
    grid failing, reading as an ordinary crash inside the binary.
    """

    def test_sigkill_is_named_and_blames_memory(self):
        error = ModelExecutionError('Kraken', return_code=-9)
        assert 'SIGKILL' in str(error)
        assert 'out-of-memory' in str(error)

    def test_other_signals_are_named_without_the_memory_note(self):
        error = ModelExecutionError('Kraken', return_code=-11)
        assert 'SIGSEGV' in str(error)
        assert 'out-of-memory' not in str(error)

    def test_an_ordinary_exit_code_is_reported_as_its_number(self):
        assert 'exit code: 2' in str(ModelExecutionError('Kraken',
                                                         return_code=2))

    def test_the_never_launched_sentinel_is_not_read_as_a_signal(self):
        """uacpy passes -1 when the binary never ran; ``-1`` is SIGHUP's code,
        and naming it would invent a kill that never happened."""
        error = ModelExecutionError('Kraken', return_code=-1,
                                    stderr='Executable not found: …')
        assert 'SIGHUP' not in str(error)
        assert 'exit code: -1' in str(error)

    @pytest.mark.requires_binary  # constructs RAM (resolves its binary)
    def test_a_really_killed_child_is_named(self, tmp_path):
        """End to end through ``_run_subprocess``, not just the constructor."""
        with pytest.raises(ModelExecutionError) as excinfo:
            RAM(verbose=False, timeout=30.0)._run_subprocess(
                ['/bin/sh', '-c', 'kill -9 $$'], cwd=tmp_path)
        assert excinfo.value.return_code == -9
        assert 'SIGKILL' in str(excinfo.value)
