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
    """A BrokenProcessPool from an interactive session (no importable __main__)
    must be translated into a clear, actionable ConfigurationError; a genuine
    crash (importable __main__) keeps the original BrokenProcessPool. Driven by
    a fake pool so it's deterministic and needs no real subprocess/binary."""
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

    monkeypatch.setattr(P, '_main_is_importable', lambda: True)    # real script
    with pytest.raises(BrokenProcessPool):
        P.run_parallel([job], start_method='spawn')


def test_job_defaults():
    j = Job(model=uacpy.models.Bellhop(), env='e', source='s', receiver='r')
    assert j.run_mode is None and j.run_kwargs == {} and j.label is None


@pytest.mark.requires_binary
def test_run_parallel_knob_sweep(pekeris_env):
    """Sweep one model's knob by building jobs with ``model.copy()``."""
    src = uacpy.Source(depths=25.0, frequencies=200.0)
    rcv = uacpy.Receiver(depths=np.linspace(10, 90, 9), ranges=np.linspace(100, 5000, 21))
    base = uacpy.models.Bellhop()
    jobs = [
        Job(base.copy(n_beams=n), pekeris_env, src, rcv,
            run_mode=uacpy.RunMode.COHERENT_TL, label=n)
        for n in (200, 400, 800)
    ]
    batch = run_parallel(jobs, n_workers=3, coordinate_name='n_beams')
    assert batch.ok and len(batch) == 3
    assert all(np.isfinite(np.nanmax(r.tl)) for r in batch)
    stack = batch.stack()
    assert len(stack) == 3
    assert np.array_equal(stack.coordinate, np.array([200.0, 400.0, 800.0]))


@pytest.mark.requires_binary
def test_run_parallel_scenario_sweep(pekeris_env):
    """Same model, a different source per job."""
    rcv = uacpy.Receiver(depths=np.linspace(10, 90, 9), ranges=np.linspace(100, 5000, 21))
    depths = [10.0, 50.0, 90.0]
    base = uacpy.models.Bellhop()
    jobs = [
        Job(base.copy(), pekeris_env, uacpy.Source(depths=d, frequencies=200.0), rcv,
            run_mode=uacpy.RunMode.COHERENT_TL, label=d)
        for d in depths
    ]
    batch = run_parallel(jobs, n_workers=3, coordinate_name='source_depth')
    assert batch.ok and len(batch) == 3
    maxes = [float(np.nanmax(r.tl)) for r in batch]
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
        assert np.isfinite(np.nanmax(res.tl))


@pytest.mark.requires_binary
def test_run_parallel_preserves_rays_and_eigenrays(pekeris_env):
    """Ray geometry must survive the pickle round-trip even though the worker
    wipes its scratch .ray file."""
    src = uacpy.Source(depths=25.0, frequencies=200.0)
    rcv = uacpy.Receiver(depths=np.array([50.0]), ranges=np.array([2000.0]))
    base = uacpy.models.Bellhop()
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
    base = uacpy.models.Bellhop()
    jobs = [
        Job(base.copy(), pekeris_env, good, rcv, run_mode=uacpy.RunMode.RAYS),
        Job(base.copy(), pekeris_env, bad, rcv, run_mode=uacpy.RunMode.RAYS),
    ]
    batch = run_parallel(jobs, n_workers=2, raise_on_error=False)
    assert not batch.ok
    assert batch[0] is not None
    assert 1 in batch.errors
    assert len(batch.stack()) == 1
