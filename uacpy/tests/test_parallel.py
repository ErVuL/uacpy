"""Tests for parallel runs (``uacpy.run_parallel`` / ``Job``) and the
``model.copy()`` round-trip that knob sweeps rely on."""

import numpy as np
import pytest

import uacpy
from uacpy.models.base import _collect_init_params
from uacpy.parallel import Job, ParallelResult, run_parallel

CONCRETE_MODELS = [
    'Bellhop', 'Bounce', 'Kraken',
    'OASN', 'OASP', 'OASR', 'OAST', 'RAM', 'SPARC', 'Scooter',
]


def _values_equal(a, b):
    if a is b:
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return np.array_equal(np.asarray(a), np.asarray(b))
        except Exception:
            return False
    try:
        return bool(a == b)
    except Exception:
        return repr(a) == repr(b)


# ── model.copy() round-trip (the invariant knob sweeps depend on) ─────────

@pytest.mark.parametrize('model_name', CONCRETE_MODELS)
def test_copy_roundtrip_preserves_all_constructor_args(model_name):
    """``copy()`` must reproduce every stored constructor argument — otherwise
    a parallel batch would silently run with the wrong configuration."""
    cls = getattr(uacpy.models, model_name)
    model = cls()
    clone = model.copy()

    assert type(clone) is type(model)
    for name, _default in _collect_init_params(cls):
        if hasattr(model, name):
            assert hasattr(clone, name), f"{model_name}.copy() dropped '{name}'"
            assert _values_equal(getattr(model, name), getattr(clone, name)), (
                f"{model_name}.copy() changed '{name}': "
                f"{getattr(model, name)!r} != {getattr(clone, name)!r}"
            )


def test_copy_override_changes_only_target():
    model = uacpy.models.Bellhop(n_beams=100, beam_type='G')
    clone = model.copy(n_beams=777)
    assert clone.n_beams == 777
    assert clone.beam_type == model.beam_type
    assert model.n_beams == 100  # original untouched


def test_copy_rejects_unknown_override():
    with pytest.raises(uacpy.ConfigurationError):
        uacpy.models.Bellhop().copy(definitely_not_a_param=1)


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
    with pytest.raises(uacpy.ConfigurationError):
        stack.isel(range=0)                              # wrong (non-stacking) axis


def test_parallelresult_stack_all_failed_raises():
    sr = ParallelResult(
        results=[None], errors={0: ValueError()},
        labels=[0], coordinate_name='case',
    )
    assert sr.ok is False
    with pytest.raises(uacpy.ConfigurationError):
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
    with pytest.raises(uacpy.ConfigurationError):
        run_parallel([])


def test_run_parallel_shared_pinned_work_dir_raises(tmp_path):
    """One work_dir pinned on two jobs is rejected up front, before any pool
    or worker exists. The guard resolves paths, so a str spelling and a Path
    spelling of the same directory collide."""
    class _PinnedModel:
        def __init__(self, work_dir):
            self.work_dir = work_dir

    shared = tmp_path / 'shared_scratch'
    jobs = [
        Job(model=_PinnedModel(str(shared)), env='e', source='s', receiver='r'),
        Job(model=_PinnedModel(shared), env='e', source='s', receiver='r'),
    ]
    with pytest.raises(uacpy.ConfigurationError,
                       match="pinned on more than one job"):
        run_parallel(jobs)


def test_main_is_importable_helper(monkeypatch, tmp_path):
    """The spawn-safety probe: True only when __main__ is a real file."""
    import sys, types
    from uacpy.parallel import _main_is_importable
    fake = types.ModuleType('__main__')
    monkeypatch.setitem(sys.modules, '__main__', fake)
    assert _main_is_importable() is False           # no __file__ (REPL/stdin)
    f = tmp_path / "m.py"; f.write_text("")
    fake.__file__ = str(f)
    assert _main_is_importable() is True             # importable .py script


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

    monkeypatch.setattr(P, '_main_is_importable', lambda: False)   # interactive
    with pytest.raises(uacpy.ConfigurationError, match="interactive session"):
        P.run_parallel([job], start_method='spawn')

    # Importable __main__ (a real .py script) and the pool dies before any job
    # completes: that is a *startup* death, and for a script the usual cause is
    # a module-level run_parallel with no `if __name__ == "__main__":` guard,
    # so the message must name the guard rather than blame a segfault.
    monkeypatch.setattr(P, '_main_is_importable', lambda: True)    # real script
    with pytest.raises(uacpy.ConfigurationError, match="__main__") as ei:
        P.run_parallel([job], start_method='spawn')
    assert isinstance(ei.value.__cause__, BrokenProcessPool)       # original kept


def test_run_parallel_broken_pool_after_a_job_completes_says_mid_run(monkeypatch):
    """Once a job has completed, a dead pool really is a mid-run crash."""
    import uacpy.parallel as P
    from concurrent.futures.process import BrokenProcessPool

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
                return 'result-object'
            raise BrokenProcessPool("pool died")

    monkeypatch.setattr(P, 'ProcessPoolExecutor', _OneThenDead)
    monkeypatch.setattr(P, 'as_completed', lambda m: list(m))
    monkeypatch.setattr(P, '_main_is_importable', lambda: True)
    jobs = [Job(model=object(), env='e', source='s', receiver='r'),
            Job(model=object(), env='e', source='s', receiver='r')]
    with pytest.raises(uacpy.ConfigurationError, match="died mid-run"):
        P.run_parallel(jobs, start_method='spawn')


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
    assert all(np.isfinite(np.nanmax(r.db)) for r in batch)
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
    maxes = [float(np.nanmax(r.db)) for r in batch]
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
        assert np.isfinite(np.nanmax(res.db))


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


def test_copy_preserves_an_explicit_cleanup_choice(tmp_path):
    """An explicitly requested cleanup=True still survives copy()."""
    d = tmp_path / 'scratch'
    base = uacpy.models.Bellhop(cleanup=True)
    assert base.copy(work_dir=d).cleanup is True
    base2 = uacpy.models.Bellhop(work_dir=tmp_path / 'a', cleanup=False)
    assert base2.copy(work_dir=tmp_path / 'b').cleanup is False


def test_pool_death_before_any_job_names_the_main_guard(monkeypatch):
    """A pool that dies before any job completes must blame the __main__ guard.

    An unguarded module-level ``run_parallel`` in a .py script leaves
    ``__main__`` importable, so the interactive-session check does not fire.
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
