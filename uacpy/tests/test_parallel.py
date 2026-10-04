"""Tests for parallel runs (``uacpy.run_parallel`` / ``Job``). The
``model.copy()`` round trip that knob sweeps rely on, and the check that a
parallel run returns what a direct run returns, are held for every registered
engine by ``test_engine_conformance.py``."""

import numpy as np
import pytest

import uacpy
from uacpy.parallel import Job, ParallelResult, run_parallel
from uacpy.tests.conftest import build_engine, engine_params


@pytest.mark.requires_binary
def test_copy_override_changes_only_target():
    model = uacpy.models.Bellhop(n_beams=100, beam_type='G')
    clone = model.copy(n_beams=777)
    assert clone.n_beams == 777
    assert clone.beam_type == model.beam_type
    assert model.n_beams == 100  # original untouched


# ── ParallelResult container semantics (no subprocess needed) ────────────────

def _tiny_field(value):
    return uacpy.Field(
        data=np.full((2, 3), value, dtype=complex),
        coords={'depth': np.array([10.0, 20.0]), 'range': np.array([1.0, 2.0, 3.0])},
        model='', backend='',
        source_depths=np.array([5.0]),
        frequencies=100.0,
        phase_reference='travelling_wave',
    )


def test_parallelresult_collect_and_stack():
    f0, f2 = _tiny_field(1.0), _tiny_field(2.0)
    sr = ParallelResult(
        results=[f0, None, f2],
        errors={1: ValueError('boom')},
        labels=[10.0, 20.0, 30.0],
        coordinate_name='depth',
    )
    assert not sr.ok
    assert len(sr) == 3
    assert sr[1] is None and isinstance(sr.errors[1], ValueError)
    assert [r for r in sr].count(None) == 1

    stack = sr.stack()                       # skips the failed case
    assert len(stack) == 2
    assert np.array_equal(stack.coordinate, np.array([10.0, 30.0]))  # labels

    # isel: positional slab selection, parity with stack[i] and at().
    assert stack.isel(depth=0) is stack[0]
    assert stack.isel(depth=1) is stack.at(depth=30.0)   # label 30 → index 1
    with pytest.raises(uacpy.ConfigurationError,
                       match='pass exactly the stacking-axis keyword'):
        stack.isel(range=0)                              # wrong (non-stacking) axis


def test_parallelresult_stack_all_failed_raises():
    sr = ParallelResult(
        results=[None], errors={0: ValueError()},
        labels=[0], coordinate_name='case',
    )
    assert sr.ok is False
    with pytest.raises(uacpy.ConfigurationError,
                       match=r"no successful results to stack \(1 failed\)"):
        sr.stack()


# ── end-to-end parallel runs (need the Bellhop / Kraken binaries) ─────────

@pytest.fixture
def pekeris_env():
    return uacpy.Environment(
        name='Pekeris', bathymetry=100.0, ssp=1500.0,
        bottom=uacpy.BoundaryProperties(
            acoustic_type='half-space', sound_speed=1600.0,
            density=1.8, attenuation=0.3,
        ),
    )


def test_run_parallel_empty_raises():
    with pytest.raises(uacpy.ConfigurationError, match='jobs is empty'):
        run_parallel([])


def test_run_parallel_shared_pinned_work_dir_raises(tmp_path):
    """One work_dir pinned on two jobs is rejected up front, before any pool
    or worker exists. The guard resolves paths, so a str spelling and a Path
    spelling of the same directory collide."""
    class _PinnedModel:
        def __init__(self, work_dir):
            self.work_dir = work_dir

        def scratch_policy(self):
            from pathlib import Path
            from uacpy.models._workspace import ScratchPolicy
            return ScratchPolicy(pinned_dir=Path(self.work_dir),
                                 keeps_files=True, on_tmpfs=False)

    shared = tmp_path / 'shared_scratch'
    jobs = [
        Job(model=_PinnedModel(str(shared)), env='e', source='s', receiver='r'),
        Job(model=_PinnedModel(shared), env='e', source='s', receiver='r'),
    ]
    with pytest.raises(uacpy.ConfigurationError,
                       match="pinned on more than one job"):
        run_parallel(jobs)


@pytest.mark.parametrize('name', engine_params('name'))
def test_every_engine_states_its_scratch_policy_from_its_knobs(
        name, tmp_path):
    """``run_parallel`` decides which work dirs a batch leaves behind, and
    which jobs collide, from ``scratch_policy()``: the pinned ``work_dir``,
    whether the files are kept (``cleanup=False``, the default for a pinned
    dir) and ``use_tmpfs``, each read from the knob it names."""
    from uacpy.models._workspace import ScratchPolicy
    assert build_engine(name).scratch_policy() == ScratchPolicy(
        pinned_dir=None, keeps_files=False, on_tmpfs=False)
    assert build_engine(name, cleanup=False,
                        use_tmpfs=True).scratch_policy() == ScratchPolicy(
        pinned_dir=None, keeps_files=True, on_tmpfs=True)
    pinned = build_engine(name, work_dir=tmp_path)
    assert pinned.scratch_policy() == ScratchPolicy(
        pinned_dir=tmp_path, keeps_files=True, on_tmpfs=False)
    assert build_engine(name, work_dir=str(tmp_path),
                        cleanup=True).scratch_policy() == ScratchPolicy(
        pinned_dir=tmp_path, keeps_files=False, on_tmpfs=False)


def test_only_a_main_named_by_an_unreadable_file_blocks_the_workers(
        monkeypatch, tmp_path):
    """A spawned worker re-runs ``__main__`` only when it names a file, so a
    REPL / Jupyter / ``python -c`` main (no ``__file__``) and a real ``.py``
    script both boot; piped stdin (``__file__ == '<stdin>'``) does not."""
    import sys, types
    from uacpy.parallel import _main_cannot_be_reimported
    fake = types.ModuleType('__main__')
    monkeypatch.setitem(sys.modules, '__main__', fake)
    assert _main_cannot_be_reimported() is False     # REPL / Jupyter / -c
    fake.__file__ = '<stdin>'
    assert _main_cannot_be_reimported() is True      # piped stdin
    f = tmp_path / "m.py"; f.write_text("")
    fake.__file__ = str(f)
    assert _main_cannot_be_reimported() is False     # a .py script


def test_run_parallel_broken_pool_interactive_message(monkeypatch):
    """A BrokenProcessPool is always translated into a clear, typed
    ConfigurationError: the interactive-session variant (no importable __main__)
    points at the __main__ footgun, a genuine worker crash (importable __main__)
    gets the 'died mid-run' message — and in both the original BrokenProcessPool
    is preserved as ``__cause__`` so nothing is lost. Driven by a fake pool so
    it's deterministic and needs no real subprocess/binary."""
    import uacpy.parallel as P
    from concurrent.futures.process import BrokenProcessPool

    class _DeadPool:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def submit(self, *a, **k): raise BrokenProcessPool("pool died on bootstrap")
    monkeypatch.setattr(P, 'ProcessPoolExecutor', _DeadPool)
    job = Job(model=object(), env='e', source='s', receiver='r')

    monkeypatch.setattr(P, '_main_cannot_be_reimported', lambda: True)  # stdin
    with pytest.raises(uacpy.ConfigurationError, match="piped stdin"):
        P.run_parallel([job], start_method='spawn')

    # A re-runnable __main__ (a .py script) and the pool dies before any job
    # completes: that is a *startup* death, and for a script the usual cause is
    # a module-level run_parallel with no `if __name__ == "__main__":` guard,
    # so the message must name the guard rather than blame a segfault.
    monkeypatch.setattr(P, '_main_cannot_be_reimported', lambda: False)
    with pytest.raises(uacpy.ConfigurationError, match="__main__") as ei:
        P.run_parallel([job], start_method='spawn')
    assert isinstance(ei.value.__cause__, BrokenProcessPool)       # original kept


@pytest.mark.parametrize('main_blocks', [False, True],
                         ids=['script-or-interactive-main', 'stdin-main'])
def test_run_parallel_broken_pool_after_a_job_completes_says_mid_run(
        monkeypatch, main_blocks):
    """Once a worker has booted, a dead pool is a mid-run crash whatever
    ``__main__`` is, and ``raise_on_error=False`` keeps the finished result."""
    import uacpy.parallel as P
    from concurrent.futures.process import BrokenProcessPool
    from uacpy.models import _launch
    # The emulated initializer below also makes this process's launches
    # single-threaded; monkeypatch puts the flag back at teardown, or every
    # later test in this pytest process would build pool-worker commands.
    monkeypatch.setattr(_launch, '_single_threaded_launches',
                        _launch._single_threaded_launches)

    class _OneThenDead:
        """First future succeeds; the pool then breaks."""
        def __init__(self, *a, **k):
            self._n = 0
            # Emulate one worker booting: the real pool runs the initializer
            # in each worker, and run_parallel's bootstrap-vs-mid-run
            # discriminator counts those runs.
            if k.get('initializer') is not None:
                # _worker_init sets the process-global tempfile.tempdir (in
                # a REAL pool that happens inside the worker process);
                # emulating it in-process must restore the global, or every
                # later temp-file user in this pytest process points at a
                # directory run_parallel deletes.
                import tempfile as _tf
                _saved = _tf.tempdir
                try:
                    k['initializer'](*k.get('initargs', ()))
                finally:
                    _tf.tempdir = _saved

        def __enter__(self): return self
        def __exit__(self, *a): return False

        def submit(self, fn, *a, **k):
            self._n += 1
            f = _Fut(self._n == 1)
            return f

    class _Fut:
        def __init__(self, ok): self._ok = ok
        def cancel(self): return True
        def result(self, *a, **k):
            if self._ok:
                # What _job_worker hands back: (result, warnings).
                return 'result-object', []
            raise BrokenProcessPool("pool died")

    monkeypatch.setattr(P, 'ProcessPoolExecutor', _OneThenDead)
    monkeypatch.setattr(P, 'as_completed', lambda m: list(m))
    # Every __main__ shape: a worker that booted cannot have died of one.
    monkeypatch.setattr(P, '_main_cannot_be_reimported', lambda: main_blocks)
    jobs = [Job(model=object(), env='e', source='s', receiver='r'),
            Job(model=object(), env='e', source='s', receiver='r')]
    # A solver that died is a ModelExecutionError, as DOCUMENTATION §4 types
    # it, so ``except ModelExecutionError`` around a batch catches it.
    with pytest.raises(uacpy.ModelExecutionError, match="died mid-run") as ei:
        P.run_parallel(jobs, start_method='spawn')
    assert isinstance(ei.value.__cause__, BrokenProcessPool)
    assert not isinstance(ei.value, uacpy.ConfigurationError)
    # The worker ran, so the never-launched sentinel would be false; the
    # pool reports no exit code, so it is unknown.
    assert ei.value.return_code is None
    assert ei.value.return_code != uacpy.ModelExecutionError.NEVER_LAUNCHED

    # raise_on_error=False keeps what finished: the first job's result comes
    # back and only the job the crash cut off carries the error.
    out = P.run_parallel(jobs, start_method='spawn', raise_on_error=False)
    assert out.results == ['result-object', None]
    assert list(out.errors) == [1]
    assert isinstance(out.errors[1], uacpy.ModelExecutionError)
    assert 'died mid-run' in str(out.errors[1])


class _KillsItsWorker:
    """A picklable stand-in model: ``run`` returns its label, or SIGKILLs the
    worker process running it when the label is ``'die'`` — the shape of a
    native binary OOM-killed mid-batch."""

    def run(self, env, source, receiver, run_mode=None, **kw):
        import os
        import signal
        if env == 'die':
            os.kill(os.getpid(), signal.SIGKILL)
        return env


def test_a_worker_killed_mid_batch_under_a_fileless_main_keeps_the_finished_job(
        monkeypatch):
    """A REPL / Jupyter / ``python -c`` session has a ``__main__`` with no
    file. A real spawned pool boots there; when one worker is then killed, the
    crash is a ``ModelExecutionError`` and ``raise_on_error=False`` returns the
    job that finished before it."""
    import sys
    import types
    monkeypatch.setitem(sys.modules, '__main__', types.ModuleType('__main__'))
    jobs = [Job(model=_KillsItsWorker(), env='done', source=None, receiver=None),
            Job(model=_KillsItsWorker(), env='die', source=None, receiver=None)]
    out = run_parallel(jobs, n_workers=1, start_method='spawn',
                       raise_on_error=False)
    assert out.results[0] == 'done'
    assert out.results[1] is None
    assert isinstance(out.errors[1], uacpy.ModelExecutionError)
    assert not isinstance(out.errors[1], uacpy.ConfigurationError)


@pytest.mark.parametrize('module, refused', [('__main__', True),
                                               ('a_user_module', False)])
def test_a_model_class_defined_in_an_interactive_main_is_refused_up_front(
        monkeypatch, module, refused):
    """Under a REPL / Jupyter / ``python -c`` main, a class defined there is
    unknown to a spawned worker; ``run_parallel`` names it before any pool
    starts instead of reporting the worker's death as a crashed binary."""
    import sys
    import types
    import uacpy.parallel as P
    monkeypatch.setitem(sys.modules, '__main__', types.ModuleType('__main__'))

    class _NoPool:
        def __init__(self, *a, **k):
            raise AssertionError('reached the pool')
    monkeypatch.setattr(P, 'ProcessPoolExecutor', _NoPool)
    model_cls = type('SessionModel', (), {})
    model_cls.__module__ = module
    job = Job(model=model_cls(), env='e', source='s', receiver='r')
    if refused:
        with pytest.raises(uacpy.ConfigurationError, match='SessionModel'):
            P.run_parallel([job], start_method='spawn')
    else:
        with pytest.raises(AssertionError, match='reached the pool'):
            P.run_parallel([job], start_method='spawn')


@pytest.mark.requires_binary
def test_job_defaults():
    j = Job(model=uacpy.models.Bellhop(), env='e', source='s', receiver='r')
    assert j.run_mode is None and j.run_kwargs == {} and j.label is None


@pytest.mark.requires_binary
def test_run_parallel_knob_sweep(pekeris_env):
    """Sweep one model's knob by building jobs with ``model.copy()``.

    Each parallel result must equal the same job run serially in-process:
    the worker executes the identical model/scenario/run_mode, so the data,
    grid coordinates and shape all agree element-wise."""
    src = uacpy.Source(depths=25.0, frequencies=200.0)
    rcv = uacpy.Receiver(depths=np.linspace(10, 90, 9), ranges=np.linspace(100, 5000, 21))
    # bellhopcuda's GPU reductions are nondeterministic at ~1e-7 relative;
    # the fortran backend reruns bit-identically, which the equality needs.
    base = uacpy.models.Bellhop(backend='fortran')
    jobs = [
        Job(base.copy(n_beams=n), pekeris_env, src, rcv,
            run_mode=uacpy.RunMode.COHERENT_TL, label=n)
        for n in (200, 400, 800)
    ]
    batch = run_parallel(jobs, n_workers=3, coordinate_name='n_beams')
    assert batch.ok and len(batch) == 3
    assert all(np.isfinite(np.nanmax(r.dB)) for r in batch)
    stack = batch.stack()
    assert len(stack) == 3
    assert np.array_equal(stack.coordinate, np.array([200.0, 400.0, 800.0]))

    for job, par in zip(jobs, batch):
        ser = job.model.run(job.env, job.source, job.receiver,
                            run_mode=job.run_mode, **job.run_kwargs)
        assert type(par) is type(ser)
        assert par.shape == ser.shape
        assert np.array_equal(par.data, ser.data, equal_nan=True)
        for name in ('depth', 'range'):
            assert np.array_equal(par.coords[name], ser.coords[name])


@pytest.mark.requires_binary
def test_run_parallel_scenario_sweep(pekeris_env):
    """Same model, a different source per job."""
    rcv = uacpy.Receiver(depths=np.linspace(10, 90, 9), ranges=np.linspace(100, 5000, 21))
    depths = [10.0, 50.0, 90.0]
    # bellhopcuda's GPU reductions are nondeterministic at ~1e-7 relative;
    # the fortran backend reruns bit-identically, which the equality needs.
    base = uacpy.models.Bellhop(backend='fortran')
    jobs = [
        Job(base.copy(), pekeris_env, uacpy.Source(depths=d, frequencies=200.0), rcv,
            run_mode=uacpy.RunMode.COHERENT_TL, label=d)
        for d in depths
    ]
    batch = run_parallel(jobs, n_workers=3, coordinate_name='source_depth')
    assert batch.ok and len(batch) == 3
    maxes = [float(np.nanmax(r.dB)) for r in batch]
    assert len(set(np.round(maxes, 3))) > 1


@pytest.mark.requires_binary
def test_run_parallel_cross_model(pekeris_env):
    """Different models on the same scenario — the cross-model batch case.

    Each Job carries its own model (with its own native options), so the batch
    is heterogeneous with no special handling. All three produce a TL Field.
    """
    src = uacpy.Source(depths=25.0, frequencies=200.0)
    rcv = uacpy.Receiver(depths=np.linspace(10, 90, 9), ranges=np.linspace(100, 5000, 21))
    jobs = [
        Job(uacpy.models.Bellhop(n_beams=800), pekeris_env, src, rcv,
            run_mode=uacpy.RunMode.COHERENT_TL, label='bellhop'),
        Job(uacpy.models.Kraken(), pekeris_env, src, rcv,
            run_mode=uacpy.RunMode.COHERENT_TL, label='kraken'),
        Job(uacpy.models.RAM(), pekeris_env, src, rcv,
            run_mode=uacpy.RunMode.COHERENT_TL, label='ram'),
    ]
    batch = run_parallel(jobs, n_workers=3)
    assert batch.ok and len(batch) == 3
    assert batch.labels == ['bellhop', 'kraken', 'ram']
    for res in batch:
        assert isinstance(res, uacpy.Field)
        assert np.isfinite(np.nanmax(res.dB))


@pytest.mark.requires_binary
def test_run_parallel_preserves_rays_and_eigenrays(pekeris_env):
    """Ray geometry must survive the pickle round-trip even though the worker
    wipes its scratch .ray file."""
    src = uacpy.Source(depths=25.0, frequencies=200.0)
    rcv = uacpy.Receiver(depths=np.array([50.0]), ranges=np.array([2000.0]))
    # bellhopcuda's GPU reductions are nondeterministic at ~1e-7 relative;
    # the fortran backend reruns bit-identically, which the equality needs.
    base = uacpy.models.Bellhop(backend='fortran')
    jobs = [
        Job(base.copy(n_beams=n), pekeris_env, src, rcv, run_mode=uacpy.RunMode.RAYS)
        for n in (21, 41)
    ]
    batch = run_parallel(jobs, n_workers=2)
    assert batch.ok
    for res in batch:
        assert len(res.rays) > 0
        ray = res.rays[0]
        assert ray['r'].size > 1 and ray['z'].size == ray['r'].size


@pytest.mark.requires_binary
def test_run_parallel_preserves_modes(pekeris_env):
    src = uacpy.Source(depths=25.0, frequencies=200.0)
    rcv = uacpy.Receiver(depths=np.linspace(0, 100, 51), ranges=np.array([1000.0]))
    jobs = [
        Job(uacpy.models.Kraken(), pekeris_env, src, rcv, run_mode=uacpy.RunMode.MODES)
        for _ in range(2)
    ]
    batch = run_parallel(jobs, n_workers=2)
    assert batch.ok
    for res in batch:
        assert res.n_modes > 0
        assert res.k.size == res.n_modes
        assert res.phi.shape == (res.depths.size, res.k.size)


@pytest.mark.requires_binary
def test_run_parallel_workdir_keeps_artifacts(pekeris_env, tmp_path):
    """Models built with a pinned ``work_dir`` (cleanup=False) keep their
    on-disk files and valid metadata paths after the run."""
    from pathlib import Path
    src = uacpy.Source(depths=25.0, frequencies=200.0)
    rcv = uacpy.Receiver(depths=np.linspace(10, 90, 9), ranges=np.linspace(100, 5000, 11))
    jobs = []
    for i, n in enumerate((200, 400)):
        wd = tmp_path / f"case_{i}"
        jobs.append(Job(
            uacpy.models.Bellhop(n_beams=n, work_dir=str(wd), cleanup=False),
            pekeris_env, src, rcv, run_mode=uacpy.RunMode.COHERENT_TL,
        ))
    batch = run_parallel(jobs, n_workers=2)
    assert batch.ok
    for res in batch:
        shd = res.metadata.get('shd_file')
        assert shd is not None and Path(shd).exists()


@pytest.mark.requires_binary
def test_run_parallel_collects_errors(pekeris_env):
    """With raise_on_error=False, a failing job is collected while others
    still return."""
    rcv = uacpy.Receiver(depths=np.linspace(10, 90, 9), ranges=np.linspace(100, 5000, 11))
    good = uacpy.Source(depths=25.0, frequencies=200.0)
    # RAYS with a multi-frequency source is rejected in run() before the binary
    # launches — a deterministic per-job failure.
    bad = uacpy.Source(depths=25.0, frequencies=np.array([150.0, 200.0, 250.0]))
    # bellhopcuda's GPU reductions are nondeterministic at ~1e-7 relative;
    # the fortran backend reruns bit-identically, which the equality needs.
    base = uacpy.models.Bellhop(backend='fortran')
    jobs = [
        Job(base.copy(), pekeris_env, good, rcv, run_mode=uacpy.RunMode.RAYS),
        Job(base.copy(), pekeris_env, bad, rcv, run_mode=uacpy.RunMode.RAYS),
    ]
    batch = run_parallel(jobs, n_workers=2, raise_on_error=False)
    assert not batch.ok
    assert batch[0] is not None
    assert 1 in batch.errors
    assert len(batch.stack()) == 1


@pytest.mark.requires_binary
def test_copy_onto_a_user_work_dir_does_not_inherit_cleanup(tmp_path):
    """``copy(work_dir=...)`` must not wipe the caller's directory.

    ``cleanup`` resolves to ``work_dir is None`` at construction, so a plain
    ``Bellhop()`` carries ``cleanup=True``. ``copy()`` rebuilds from the
    *resolved* attributes, so re-pointing the clone at a user directory has to
    re-resolve ``cleanup`` too — carrying the parent's ``True`` across would
    rmtree that directory after ``run()``.
    """
    d = tmp_path / 'user_outputs'
    d.mkdir()
    keep = d / 'precious.txt'
    keep.write_text('do not delete')

    # bellhopcuda's GPU reductions are nondeterministic at ~1e-7 relative;
    # the fortran backend reruns bit-identically, which the equality needs.
    base = uacpy.models.Bellhop(backend='fortran')
    assert base.cleanup is True, "unpinned model should own its temp dir"

    clone = base.copy(work_dir=d)
    assert clone.cleanup is False, (
        "copy() inherited cleanup=True onto a caller-supplied work_dir; "
        "run() would rmtree it")

    # cleanup_work_dir() wipes unconditionally; the flag gates whether it is
    # reached (FileManager.__exit__), so exercise that path.
    fm = clone._setup_file_manager()
    assert fm.cleanup is False
    with fm:
        pass
    assert keep.exists(), "caller's work_dir was wiped by an inherited cleanup"


@pytest.mark.requires_binary
def test_copy_preserves_an_explicit_cleanup_choice(tmp_path):
    """An explicitly requested cleanup=True still survives copy()."""
    d = tmp_path / 'scratch'
    base = uacpy.models.Bellhop(cleanup=True)
    assert base.copy(work_dir=d).cleanup is True
    base2 = uacpy.models.Bellhop(work_dir=tmp_path / 'a', cleanup=False)
    assert base2.copy(work_dir=tmp_path / 'b').cleanup is False


@pytest.mark.requires_binary
def test_pool_death_before_any_job_names_the_main_guard(monkeypatch):
    """A pool that dies before any job completes must blame the __main__ guard.

    An unguarded module-level ``run_parallel`` in a .py script leaves
    ``__main__`` re-runnable, so the piped-stdin message does not fire.
    Dying before any job completes is a *startup* death, and for a script the
    usual cause is the missing guard — not the segfault/OOM that a mid-run
    death would indicate.
    """
    from concurrent.futures.process import BrokenProcessPool
    import uacpy.parallel as par
    from uacpy.core.exceptions import ConfigurationError

    class _DeadPool:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def submit(self, *a, **k): raise BrokenProcessPool("pool died")

    monkeypatch.setattr(par, 'ProcessPoolExecutor', _DeadPool)
    env = uacpy.Environment(bathymetry=100.0, ssp=1500.0)
    job = par.Job(uacpy.models.Bellhop(), env,
                  uacpy.Source(depths=50.0, frequencies=100.0),
                  uacpy.Receiver(depths=50.0, ranges=[1000.0]))
    with pytest.raises(ConfigurationError, match="__main__"):
        par.run_parallel([job], n_workers=1)


class TestScratchRootIsReaped:
    """``run_parallel`` points every worker's ``tempfile`` at one parent-owned
    scratch root so a SIGKILLed worker's model tempdirs are reaped instead of
    leaking. What survives there is only the caller's when a job asked to keep
    it — ``cleanup=False`` with an unpinned ``work_dir``. Debris from a failed
    job (which ``raise_on_error=False`` collects and then returns normally from)
    reads identically on disk, so a leftover-count-only test kept it forever."""

    class _Model:
        def __init__(self, cleanup=True, work_dir=None):
            self.cleanup = cleanup
            self.work_dir = work_dir

        def scratch_policy(self):
            from pathlib import Path
            from uacpy.models._workspace import ScratchPolicy
            return ScratchPolicy(
                pinned_dir=None if self.work_dir is None else Path(self.work_dir),
                keeps_files=self.cleanup is False, on_tmpfs=False)

    def _job(self, **kw):
        return Job(self._Model(**kw), None, None, None)

    @staticmethod
    def _root_with_debris(tmp_path, name):
        root = tmp_path / name
        (root / 'uacpy_bellhop_abc').mkdir(parents=True)
        return root

    def test_debris_from_a_cleanup_true_job_is_removed(self, tmp_path):
        from uacpy.parallel import _reap_scratch_root
        root = self._root_with_debris(tmp_path, 'failed')
        assert _reap_scratch_root(str(root), [self._job()]) is True
        assert not root.exists()

    def test_a_kept_work_dir_survives_with_a_warning(self, tmp_path):
        from uacpy.parallel import _reap_scratch_root
        root = self._root_with_debris(tmp_path, 'kept')
        jobs = [self._job(cleanup=False)]
        with pytest.warns(UserWarning, match=r"work dir\(s\) were kept"):
            assert _reap_scratch_root(str(root), jobs) is False
        assert root.exists()

    def test_a_pinned_work_dir_does_not_keep_the_root(self, tmp_path):
        """A pinned ``work_dir`` lives outside the scratch root, so a job using
        one cannot be what left anything inside it."""
        from uacpy.parallel import _reap_scratch_root
        root = self._root_with_debris(tmp_path, 'pinned')
        jobs = [self._job(cleanup=False, work_dir=str(tmp_path / 'mine'))]
        assert _reap_scratch_root(str(root), jobs) is True
        assert not root.exists()

    def test_a_kept_work_dir_survives_a_broken_pool(self, tmp_path,
                                                    monkeypatch):
        """The reap hangs off ``run_parallel``'s ``finally``, so it runs on the
        path where a leak actually happens — the pool dying and the batch
        aborting. The keep decision there is still the jobs' configuration
        alone: a job that asked to keep its work dir keeps the root, because
        deleting would drop files the caller asked for."""
        from concurrent.futures.process import BrokenProcessPool
        import uacpy.parallel as par
        from uacpy.core.exceptions import ConfigurationError

        root = tmp_path / 'scratch'
        root.mkdir()
        monkeypatch.setattr(par.tempfile, 'mkdtemp', lambda **kw: str(root))

        class _DeadPool:
            def __init__(self, *a, **k): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False

            def submit(self, *a, **k):
                # The model tempdir a worker allocated under the root before
                # it was killed — indistinguishable on disk from a work dir
                # a job asked to keep.
                (root / 'uacpy_bellhop_abc').mkdir(exist_ok=True)
                raise BrokenProcessPool("pool died")

        monkeypatch.setattr(par, 'ProcessPoolExecutor', _DeadPool)
        job = par.Job(self._Model(cleanup=False), None, None, None)
        with pytest.warns(UserWarning, match=r"work dir\(s\) were kept"):
            with pytest.raises(ConfigurationError,
                               match='workers died on bootstrap'):
                par.run_parallel([job], n_workers=1)
        assert root.exists()

    def test_a_broken_pool_reaps_debris_from_cleanup_true_jobs(
            self, tmp_path, monkeypatch):
        """The other half of the same ``finally``: with no job asking to keep
        anything, the dead pool's leftovers are debris by elimination and the
        root goes, rather than leaking for the life of the machine."""
        from concurrent.futures.process import BrokenProcessPool
        import uacpy.parallel as par
        from uacpy.core.exceptions import ConfigurationError

        root = tmp_path / 'scratch'
        root.mkdir()
        monkeypatch.setattr(par.tempfile, 'mkdtemp', lambda **kw: str(root))

        class _DeadPool:
            def __init__(self, *a, **k): pass
            def __enter__(self): return self
            def __exit__(self, *a): return False

            def submit(self, *a, **k):
                (root / 'uacpy_bellhop_abc').mkdir(exist_ok=True)
                raise BrokenProcessPool("pool died")

        monkeypatch.setattr(par, 'ProcessPoolExecutor', _DeadPool)
        job = par.Job(self._Model(), None, None, None)
        with pytest.raises(ConfigurationError,
                           match='workers died on bootstrap'):
            par.run_parallel([job], n_workers=1)
        assert not root.exists()

    def test_an_empty_root_is_removed(self, tmp_path):
        from uacpy.parallel import _reap_scratch_root
        root = tmp_path / 'empty'
        root.mkdir()
        assert _reap_scratch_root(str(root), [self._job()]) is True
        assert not root.exists()


class _WarningModel:
    """A picklable stand-in model whose run warns once and returns its
    label."""

    def run(self, env, source, receiver, run_mode=None, **kw):
        import warnings
        warnings.warn(f"cells of {env} are NaN", UserWarning)
        return env


def test_a_workers_warnings_reach_the_caller_and_the_result():
    """RA-CONTRACT-11: a warning raised in a worker never reached the
    caller's filters; every one is recorded per job and re-emitted in the
    calling process under its own category, prefixed with the job label."""
    jobs = [Job(model=_WarningModel(), env='a', source=None, receiver=None,
                label='first'),
            Job(model=_WarningModel(), env='b', source=None, receiver=None)]
    with pytest.warns(UserWarning) as record:
        out = run_parallel(jobs, n_workers=1)
    texts = sorted(str(w.message) for w in record
                   if 'are NaN' in str(w.message))
    assert texts == ["run_parallel job 'first': cells of a are NaN",
                     "run_parallel job 1: cells of b are NaN"]
    assert out.results == ['a', 'b']
    assert out.warnings == {0: [(UserWarning, 'cells of a are NaN')],
                            1: [(UserWarning, 'cells of b are NaN')]}


def test_a_worker_launches_its_engines_single_threaded(monkeypatch):
    """RA-CONTRACT-6: the pool supplies the parallelism, so the worker
    initializer makes every engine launch single-threaded (bellhopcxx
    ``-1``) instead of one thread per core per worker."""
    import multiprocessing as mp
    import tempfile
    from uacpy.models import _launch
    from uacpy.models.bellhop import _backend
    from uacpy.parallel import _worker_init
    monkeypatch.setattr(_launch, '_single_threaded_launches', False)
    # Read through the binding an engine calls, so the reset and the
    # initializer's write are both seen where the launch decision is made.
    assert _backend._launch_single_threaded() is False
    saved = tempfile.tempdir
    try:
        _worker_init(mp.get_context('fork').Value('i', 0), saved or '.')
    finally:
        tempfile.tempdir = saved
    assert _backend._launch_single_threaded() is True


@pytest.mark.requires_binary
def test_bellhopcxx_in_a_worker_runs_one_thread_and_reads_merged_arrivals(
        monkeypatch):
    """The two halves of one decision: launched with ``-1``, bellhopcxx
    merges its arrivals itself (``arr.hpp:79``), so the reader must not
    merge them again."""
    from uacpy.core.exceptions import ExecutableNotFoundError
    from uacpy.models import _launch
    from uacpy.models.bellhop._backend import (
        arrivals_need_merge, build_command)
    try:
        model = uacpy.Bellhop(backend='cxx')
    except ExecutableNotFoundError:
        pytest.skip('bellhopcxx not installed')
    monkeypatch.setattr(_launch, '_single_threaded_launches', False)
    def command():
        return build_command(model._exe, 'model', backend=model._resolved_backend,
                             dimensionality=model.dimensionality)

    assert '-1' not in command()
    monkeypatch.setattr(_launch, '_single_threaded_launches', True)
    assert '-1' in command()
    assert arrivals_need_merge(backend=model._resolved_backend, exe=model._exe,
                               model_name=model.model_name) is False


def test_stacking_jobs_that_returned_stacks_is_refused():
    """RA-CONTRACT-17: a multi-depth job returns a ResultStack, and a stack
    of stacks neither superposes nor plots; ``stack()`` says so."""
    from uacpy.core.results import Field, ResultStack
    f = Field(data=np.ones((1, 1)), coords={'depth': [1.0], 'range': [1.0]})
    inner = ResultStack([f, f], [10.0, 20.0], coordinate_name='source_depth')
    with pytest.raises(uacpy.ConfigurationError, match='ResultStack'):
        ParallelResult([inner, inner], {}, [1, 2]).stack()
