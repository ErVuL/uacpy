"""Thread safety: shared state that two threads used to corrupt.

Nothing in ``docs/``, ``README.md`` or ``DOCUMENTATION.md`` says uacpy is
single-threaded, and the pure-Python layers (carriers, dsp, plotting) and the
``run_parallel`` process path are in fact thread-safe. The defects were four
localised pieces of shared state plus one blind spot in the parallel reaper.

**F1/F4** every memo in :mod:`uacpy.data` was a bare
``if key not in store: store[key] = factory()``. N threads racing a cold entry
all ran the factory: the array-backed readers built N copies and kept the last
(8 threads took EMODnet from 620 MB to 4158 MB of peak RSS, with the 7 losers
unreachable from ``invalidate_grids``), and the netCDF-backed ones opened one
HDF5 file N times at once, which the non-thread-safe HDF5 build under netCDF4
answered with a SIGSEGV — measured 3/3 runs, exit 139.

**F2** two threads pointed at one pinned ``work_dir`` returned each other's
answers: 3 of 6 results wrong across 3 trials, the 1600 m/s job handing back
the 1800 m/s job's ``|R|`` bit-identically, with nothing raised.
``run_parallel`` refuses that configuration across its own jobs; nothing
refused it anywhere else.

**F3** 8 threads x 480 reads of one already-open netCDF handle raised
``RuntimeError: NetCDF: HDF error`` or segfaulted (3 of 7 runs). The untyped
``RuntimeError`` also escaped the ``data.environment`` source-fallback chains,
which catch ``DataFetchError`` / ``ConfigurationError`` only — so an ``'auto'``
chain that is documented never to fail aborted.

**F5** ``run_parallel``'s reaper decided from ``os.listdir(scratch_root)``,
but ``use_tmpfs=True`` work dirs live in ``/dev/shm``, outside that root. Three
kept RAM-backed dirs produced zero warnings, against a correctly-warning
non-tmpfs control.

The threading tests force their race with barriers and events rather than
waiting for one, so a pass means the invariant holds and not that the timing
happened to be kind.
"""

import contextlib
import gc
import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import pytest

from uacpy.core.exceptions import ConfigurationError, DataFetchError
from uacpy.data import _cache
from uacpy.data._netcdf import NetcdfGrid, netcdf_lock
from uacpy.io.file_manager import FileManager
from uacpy.models.base import PropagationModel, _PINNED_WORK_DIRS
from uacpy.parallel import Job, _reap_scratch_root

# Threads per race. Small enough to stay quick, large enough that an unguarded
# check-then-set loses every time.
N_THREADS = 4

# How long the first thread through a cold memo waits for the others to join
# it. Pre-fix they arrive and the barrier passes at once; post-fix they cannot
# get there, so the wait is the (once-per-test) cost of proving they did not.
JOIN_TIMEOUT_S = 0.3


class _CountingBuilder:
    """Factory that counts its calls and pauses inside the check-then-set window.

    The pause is a barrier the whole thread group has to reach. Against an
    unguarded memo every thread enters the factory, the barrier fills and
    releases immediately, and the call count comes out at ``N_THREADS``.
    Against a guarded one only the first thread gets in, the barrier times out,
    and the count is 1 — so the assertion reads the same either way and the
    only thing that decides it is whether the memo serialises.
    """

    def __init__(self, parties=N_THREADS):
        self.calls = 0
        self._lock = threading.Lock()
        self._barrier = threading.Barrier(parties)

    def __call__(self, *args, **kwargs):
        with self._lock:
            self.calls += 1
        try:
            self._barrier.wait(timeout=JOIN_TIMEOUT_S)
        except threading.BrokenBarrierError:
            pass
        return object()


def _race(fn, n=N_THREADS):
    """Run ``fn()`` on ``n`` threads released together; return their results."""
    return _race_calls([fn] * n)


def _race_calls(calls):
    """Run each callable in ``calls`` on its own thread, all released together.

    Takes a list rather than one callable repeated, so a read and a close can
    be raced against each other and not only against their own kind.
    """
    start = threading.Barrier(len(calls))

    def call(fn):
        start.wait(timeout=5.0)
        return fn()

    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        return list(pool.map(call, calls))


# ── F1 / F4: one factory call per cold memo entry ────────────────────────────

def test_threads_racing_a_cold_grid_memo_open_the_file_once(tmp_path):
    path = tmp_path / 'grid.nc'
    builder = _CountingBuilder()
    try:
        grids = _race(lambda: _cache.cached_grid_at(path, builder, 'globsed'))
    finally:
        _cache._GRIDS.pop(str(path), None)
    assert builder.calls == 1
    assert len({id(g) for g in grids}) == 1


def test_threads_racing_a_cold_woa23_handle_open_the_file_once(tmp_path,
                                                               monkeypatch):
    from uacpy.data import woa23_local

    path = tmp_path / 'woa23_decav_t00_01.nc'
    builder = _CountingBuilder()
    monkeypatch.setattr(woa23_local, 'open_netcdf', builder)
    try:
        handles = _race(lambda: woa23_local._open(path))
    finally:
        woa23_local._DATASETS.pop(str(path), None)
    assert builder.calls == 1
    assert len({id(h) for h in handles}) == 1


# One row per data memo that used to double-load: the module, its memo dict,
# the entry point threads call, and the builder behind it.
_MEMOS = [
    ('uacpy.data.emodnet_local', '_INDEX', '_index', '_build_index'),
    ('uacpy.data.seaice_local', '_MODEL', '_model', '_build_model'),
    ('uacpy.data.crust1_local', '_MODEL', '_model', '_build_model'),
    ('uacpy.data.diesing_local', '_MODEL', '_model', '_build_model'),
    ('uacpy.data.sediment_db', '_SAMPLES', '_samples', '_build_samples'),
    ('uacpy.data.wind_local', '_CLIM', '_clim', '_Climatology'),
]


@pytest.mark.parametrize('module_name, memo_attr, entry_attr, builder_attr',
                         _MEMOS,
                         ids=[m[0].rsplit('.', 1)[1] for m in _MEMOS])
def test_threads_racing_a_cold_data_memo_build_it_once(
        module_name, memo_attr, entry_attr, builder_attr, tmp_path,
        monkeypatch):
    import importlib

    module = importlib.import_module(module_name)
    memo = getattr(module, memo_attr)
    builder = _CountingBuilder()
    monkeypatch.setattr(module, builder_attr, builder)
    # wind_local resolves the cached file before it memoises, so the dataset
    # would have to be installed for the entry point to be reachable at all.
    monkeypatch.setattr(_cache, 'require', lambda *parts: tmp_path / 'stub')
    monkeypatch.setattr(_cache, 'cache_root', lambda: tmp_path)
    entry = getattr(module, entry_attr)
    try:
        built = _race(entry)
    finally:
        memo.clear()
    assert builder.calls == 1
    assert len({id(b) for b in built}) == 1


def test_a_warm_memo_entry_is_returned_without_rebuilding(tmp_path):
    path = tmp_path / 'grid.nc'
    calls = []

    def factory(p):
        calls.append(p)
        return object()

    try:
        first = _cache.cached_grid_at(path, factory, 'globsed')
        again = _race(lambda: _cache.cached_grid_at(path, factory, 'globsed'))
    finally:
        _cache._GRIDS.pop(str(path), None)
    assert len(calls) == 1
    assert all(g is first for g in again)


def test_a_failed_memo_build_is_not_cached(tmp_path):
    path = tmp_path / 'grid.nc'

    def factory(p):
        raise OSError('truncated')

    with pytest.raises(DataFetchError):
        _cache.cached_grid_at(path, factory, 'globsed')
    assert str(path) not in _cache._GRIDS


# ── F3: netCDF reads are serialised and typed ────────────────────────────────

class _NetcdfEntryCounter:
    """Records the most threads ever inside the netCDF4 stand-ins at once.

    One counter is shared by the variable and the handle below, so it counts
    entries into the *library* rather than into one object: a close landing
    inside another thread's read reads as two simultaneous entrants, which is
    exactly what the non-thread-safe HDF5 build answers with a SIGSEGV.
    """

    def __init__(self):
        self.max_concurrent = 0
        self._live = 0
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def entered(self):
        """Count one thread as inside for the length of the block."""
        with self._lock:
            self._live += 1
            self.max_concurrent = max(self.max_concurrent, self._live)
        try:
            # Long enough that the GIL is handed on: unserialised, every
            # thread piles into this window.
            threading.Event().wait(0.002)
            yield
        finally:
            with self._lock:
                self._live -= 1


class _OverlapCountingVariable:
    """netCDF variable stand-in that reports every read to a counter.

    Given no counter it keeps one of its own, so ``max_concurrent`` reads back
    off the variable for the tests that race reads against reads alone.
    """

    def __init__(self, counter=None, value=1.0):
        self.counter = _NetcdfEntryCounter() if counter is None else counter
        self.value = value

    @property
    def max_concurrent(self):
        return self.counter.max_concurrent

    def __getitem__(self, index):
        with self.counter.entered():
            return self.value


class _OverlapCountingDataset:
    """netCDF handle stand-in that reports every close to a counter.

    ``variables`` maps each name to what slicing it yields, wrapped in an
    ``_OverlapCountingVariable`` on the same counter — so one handle stands in
    for a whole file and its slices and its close all count as entries into
    the library.
    """

    def __init__(self, counter, variables=()):
        self.counter = counter
        self.variables = {name: _OverlapCountingVariable(counter, value)
                          for name, value in dict(variables).items()}
        self.closes = 0
        self._lock = threading.Lock()

    def close(self):
        with self.counter.entered(), self._lock:
            self.closes += 1


def _detached_grid(path):
    """A NetcdfGrid with no file behind it — only ``cell`` is exercised."""
    grid = object.__new__(NetcdfGrid)
    grid.path = path
    return grid


def test_concurrent_cell_reads_never_enter_netcdf_together(tmp_path):
    grid = _detached_grid(tmp_path / 'grid.nc')
    variable = _OverlapCountingVariable()
    values = _race(lambda: grid.cell(variable, 0, 0), n=8)
    assert variable.max_concurrent == 1
    assert values == [1.0] * 8


def test_a_cell_read_that_throws_raises_the_typed_cache_error(tmp_path):
    class Exploding:
        def __getitem__(self, index):
            raise RuntimeError('NetCDF: HDF error')

    grid = _detached_grid(tmp_path / 'grid.nc')
    grid.dataset_name = 'globsed'
    with pytest.raises(DataFetchError, match='present but unreadable'):
        grid.cell(Exploding(), 0, 0)


def test_a_close_waits_for_a_read_on_another_thread_to_leave_netcdf(tmp_path):
    """``invalidate_grids`` closes handles from whatever thread called it, so
    a close races the reads of every other thread. Unlocked, closing a handle
    mid-read exited 139/135 on 3 of 3 runs; the counter stands in for HDF5 so
    the invariant is pinned without taking the interpreter down.
    """
    counter = _NetcdfEntryCounter()
    grid = _detached_grid(tmp_path / 'grid.nc')
    grid.ds = _OverlapCountingDataset(counter)
    variable = _OverlapCountingVariable(counter)
    # Interleaved so the pool starts a closer between two readers.
    calls = []
    for _ in range(4):
        calls += [lambda: grid.cell(variable, 0, 0), grid.close]
    _race_calls(calls)
    # close() swallows every exception, so the count is what proves the closes
    # ran at all rather than failing silently and leaving reads to be counted.
    assert grid.ds.closes == 4
    assert counter.max_concurrent == 1


def test_the_netcdf_lock_is_reentrant():
    with netcdf_lock:
        with netcdf_lock:
            assert True


# ── R1: the live downloaders read and close inside the lock too ──────────────
#
# Both reach netCDF4 through a downloaded scratch file, so they cannot be run
# offline against a real grid. The handle is the stand-in instead: the module's
# own ``open_netcdf`` is patched out, and everything from the open to the close
# is the real code path.

def test_concurrent_gmrt_region_reads_never_enter_netcdf_together(monkeypatch):
    from uacpy.data import _netcdf, gmrt_live

    counter = _NetcdfEntryCounter()
    handles = []

    def open_stub(path):
        ds = _OverlapCountingDataset(counter, {
            'lon': np.array([0.0, 1.0]),
            'lat': np.array([0.0, 1.0]),
            'altitude': np.array([[-10.0, -20.0], [-30.0, -40.0]]),
        })
        handles.append(ds)
        return ds

    monkeypatch.setattr(gmrt_live, 'http_get',
                        lambda *a, **k: b'not a real grid')
    monkeypatch.setattr(_netcdf, 'open_netcdf', open_stub)
    _race(lambda: gmrt_live.region_grid((0.0, 1.0), (0.0, 1.0), 2, 2), n=8)
    assert len(handles) == 8
    assert all(ds.closes == 1 for ds in handles)
    assert counter.max_concurrent == 1


def test_concurrent_wind_grid_fetches_never_enter_netcdf_together(monkeypatch):
    from uacpy.data import _netcdf, wind_local

    counter = _NetcdfEntryCounter()
    handles = []
    speed_var = wind_local._SPEED_VARS[0]

    def open_stub(path):
        ds = _OverlapCountingDataset(counter, {
            'latitude': np.array([0.0, 1.0]),
            'longitude': np.array([0.0, 1.0]),
            speed_var: np.array([[5.0, 6.0], [7.0, 8.0]]),
        })
        handles.append(ds)
        return ds

    monkeypatch.setattr(wind_local, 'http_get',
                        lambda *a, **k: b'not a real grid')
    monkeypatch.setattr(_netcdf, 'open_netcdf', open_stub)
    grids = _race(lambda: wind_local._fetch_monthly_grid(
        2020, 1, timeout=1.0, verbose=False), n=8)
    assert len(handles) == 8
    assert all(ds.closes == 1 for ds in handles)
    assert all(g is not None for g in grids)
    assert counter.max_concurrent == 1


# ── F2: one pinned work_dir, one owning thread ───────────────────────────────

class _StubModel:
    """Minimal stand-in driving the real ``_setup_file_manager`` recipe."""

    model_name = 'Stub'
    _setup_file_manager = PropagationModel._setup_file_manager

    def __init__(self, work_dir, cleanup=False):
        self.work_dir = work_dir
        self.cleanup = cleanup
        self.use_tmpfs = False

    def run(self, tag):
        """Write a marker and read it back, the way a model writes a deck and
        reads the binary's output back out of the same directory."""
        fm = self._setup_file_manager()
        try:
            marker = fm.get_path('marker.txt')
            marker.write_text(tag)
            return marker.read_text()
        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()


def test_a_second_thread_on_one_pinned_work_dir_is_refused(tmp_path):
    shared = tmp_path / 'shared'
    claimed = threading.Event()
    may_finish = threading.Event()

    def hold():
        fm = _StubModel(shared)._setup_file_manager()
        claimed.set()
        may_finish.wait(timeout=5.0)
        fm.cleanup_work_dir()

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert claimed.wait(timeout=5.0)
        with pytest.raises(ConfigurationError,
                           match='already in use by thread'):
            _StubModel(shared)._setup_file_manager()
    finally:
        may_finish.set()
        holder.join(timeout=5.0)
    assert str(shared.resolve()) not in _PINNED_WORK_DIRS


def test_a_refusal_names_the_live_thread_that_holds_the_work_dir(tmp_path):
    """The refusal survives only where it is true, so it may say so outright:
    every claim that still stands after the reclaim attempts below is held by a
    thread that is still running."""
    shared = tmp_path / 'shared'
    claimed = threading.Event()
    may_finish = threading.Event()

    def hold():
        fm = _StubModel(shared)._setup_file_manager()
        claimed.set()
        may_finish.wait(timeout=5.0)
        fm.cleanup_work_dir()

    holder = threading.Thread(target=hold, name='owning-worker')
    holder.start()
    try:
        assert claimed.wait(timeout=5.0)
        with pytest.raises(ConfigurationError) as refusal:
            _StubModel(shared)._setup_file_manager()
        message = str(refusal.value)
        assert "thread 'owning-worker'" in message
        assert 'still running' in message
        # The refused attempt must not have disturbed the standing claim.
        assert _PINNED_WORK_DIRS[str(shared.resolve())][0] is holder
    finally:
        may_finish.set()
        holder.join(timeout=5.0)


def test_a_work_dir_reachable_only_through_a_reference_cycle_is_reclaimed(
        tmp_path):
    """A manager stranded in a reference cycle outlives the run that made it.

    Its owner thread is still alive — a pool worker back to waiting for its
    next job — so the claim is not stale by liveness, and only the cyclic
    collector can free the manager that holds it. Ordinary allocation churn
    does not: the claim was still held after 200 000 allocations.
    """
    shared = tmp_path / 'shared'
    parked = threading.Event()
    may_finish = threading.Event()

    def worker():
        cycle = {'fm': _StubModel(shared)._setup_file_manager()}
        cycle['self'] = cycle       # reachable only from itself once dropped
        del cycle
        parked.set()
        may_finish.wait(timeout=5.0)

    owner = threading.Thread(target=worker)
    owner.start()
    try:
        assert parked.wait(timeout=5.0)
        assert owner.is_alive()
        assert _StubModel(shared).run('later') == 'later'
    finally:
        may_finish.set()
        owner.join(timeout=5.0)


def test_a_work_dir_left_by_an_exited_thread_is_reclaimed(tmp_path):
    """A manager the collector cannot free — still strongly referenced — whose
    owner thread has exited. Nothing is running it, so the claim is stale."""
    shared = tmp_path / 'shared'
    stranded = []

    def worker():
        stranded.append(_StubModel(shared)._setup_file_manager())

    owner = threading.Thread(target=worker)
    owner.start()
    owner.join(timeout=5.0)
    assert not owner.is_alive()
    assert stranded             # still referenced, so still uncollectable

    assert _StubModel(shared).run('later') == 'later'
    # Dropping the stranded manager now must not free the claim taken since:
    # its release names the thread that took it, which no longer owns the key.
    key = str(shared.resolve())
    holder = _StubModel(shared)._setup_file_manager()
    try:
        stranded.clear()
        gc.collect()
        assert _PINNED_WORK_DIRS[key][0] is threading.current_thread()
    finally:
        holder.cleanup_work_dir()


def test_a_claim_is_released_by_a_collection_on_another_thread(tmp_path):
    """The finalizer that hands a claim back runs wherever the collector does,
    which is not the thread that took it."""
    shared = tmp_path / 'shared'
    key = str(shared.resolve())

    def worker():
        cycle = {'fm': _StubModel(shared)._setup_file_manager()}
        cycle['self'] = cycle

    gc.disable()
    try:
        owner = threading.Thread(target=worker)
        owner.start()
        owner.join(timeout=5.0)
        assert key in _PINNED_WORK_DIRS
    finally:
        gc.enable()
    gc.collect()                # on this thread, not the one that claimed
    assert key not in _PINNED_WORK_DIRS


def test_a_recycled_thread_ident_does_not_inherit_a_work_dir_claim(tmp_path):
    """CPython hands a dead thread's ident to the next thread started — both
    threads after one exited were measured getting its ident. Keyed on the
    ident, an unrelated later thread reads as the owner and is counted in as a
    re-entrant *second* claim on a directory it has never touched."""
    shared = tmp_path / 'shared'
    key = str(shared.resolve())
    stranded = []
    entries = []

    def leaver():
        stranded.append(_StubModel(shared)._setup_file_manager())

    def newcomer():
        # Bound, so the manager outlives the read below: an unreferenced one
        # is collected the moment the call returns and hands its claim back.
        fm = _StubModel(shared)._setup_file_manager()
        entries.append(list(_PINNED_WORK_DIRS[key]))
        fm.cleanup_work_dir()

    first = threading.Thread(target=leaver)
    first.start()
    first.join(timeout=5.0)
    second = threading.Thread(target=newcomer)
    second.start()
    second.join(timeout=5.0)

    owner, depth = entries[0]
    assert owner is second, "the newcomer took the claim in its own name"
    assert depth == 1, "a fresh claim, not a re-entrant second on a stale one"


def test_two_threads_on_one_pinned_work_dir_never_trade_results(tmp_path):
    """The measured failure: both threads wrote their own marker into one
    shared directory and both read back whichever landed last."""
    shared = tmp_path / 'shared'
    outcomes = {}
    both_wrote = threading.Barrier(2, timeout=2.0)

    def job(tag):
        try:
            fm = _StubModel(shared)._setup_file_manager()
        except ConfigurationError:
            outcomes[tag] = 'refused'
            both_wrote.abort()
            return
        marker = fm.get_path('marker.txt')
        marker.write_text(tag)
        # Hold the directory until the other thread has written too, so an
        # unguarded pair really does read each other's marker.
        try:
            both_wrote.wait()
        except threading.BrokenBarrierError:
            pass
        outcomes[tag] = marker.read_text()

    threads = [threading.Thread(target=job, args=(tag,)) for tag in 'AB']
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)

    assert 'refused' in outcomes.values()
    for tag, got in outcomes.items():
        assert got in (tag, 'refused')


def test_one_thread_may_run_twice_into_one_pinned_work_dir(tmp_path):
    shared = tmp_path / 'shared'
    assert _StubModel(shared).run('first') == 'first'
    assert _StubModel(shared).run('second') == 'second'
    assert str(shared.resolve()) not in _PINNED_WORK_DIRS


def test_one_thread_may_nest_two_runs_on_one_pinned_work_dir(tmp_path):
    shared = tmp_path / 'shared'
    outer = _StubModel(shared)._setup_file_manager()
    try:
        assert _StubModel(shared).run('inner') == 'inner'
    finally:
        outer.cleanup_work_dir()
    assert str(shared.resolve()) not in _PINNED_WORK_DIRS


def test_a_pinned_work_dir_kept_by_one_thread_is_free_for_the_next(tmp_path):
    """``cleanup=False`` means the model recipe never calls
    ``cleanup_work_dir``, so the claim has to come back some other way."""
    shared = tmp_path / 'shared'
    results = []
    for tag in ('first', 'second'):
        thread = threading.Thread(
            target=lambda t=tag: results.append(_StubModel(shared).run(t)))
        thread.start()
        thread.join(timeout=5.0)
    assert results == ['first', 'second']
    assert str(shared.resolve()) not in _PINNED_WORK_DIRS


def test_a_pinned_work_dir_is_released_when_setup_itself_raises(tmp_path):
    not_a_dir = tmp_path / 'file.txt'
    not_a_dir.write_text('x')
    with pytest.raises(ConfigurationError, match='is not a directory'):
        _StubModel(not_a_dir)._setup_file_manager()
    assert str(not_a_dir.resolve()) not in _PINNED_WORK_DIRS


def test_two_names_for_one_pinned_work_dir_collide(tmp_path):
    shared = tmp_path / 'shared'
    shared.mkdir()
    alias = tmp_path / 'alias'
    alias.symlink_to(shared, target_is_directory=True)
    claimed = threading.Event()
    may_finish = threading.Event()

    def hold():
        fm = _StubModel(shared)._setup_file_manager()
        claimed.set()
        may_finish.wait(timeout=5.0)
        fm.cleanup_work_dir()

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert claimed.wait(timeout=5.0)
        with pytest.raises(ConfigurationError,
                           match='already in use by thread'):
            _StubModel(alias)._setup_file_manager()
    finally:
        may_finish.set()
        holder.join(timeout=5.0)


def test_an_unpinned_work_dir_is_never_claimed(tmp_path):
    """Two threads on ``work_dir=None`` each get their own tempdir, so there is
    nothing to collide over and nothing to refuse."""
    class Unpinned(_StubModel):
        def __init__(self):
            super().__init__(work_dir=None, cleanup=True)

    tags = _race(lambda: Unpinned().run('x'), n=2)
    assert tags == ['x', 'x']
    assert _PINNED_WORK_DIRS == {}


# ── F5: kept RAM-backed work dirs are named ──────────────────────────────────

class _TmpfsJobModel:
    """Job model stand-in carrying only the flags the reaper reads."""

    def __init__(self, use_tmpfs):
        self.cleanup = False
        self.work_dir = None
        self.use_tmpfs = use_tmpfs


class _ResultWithFiles:
    """Result stand-in carrying only the ``*_file`` metadata the reaper reads."""

    def __init__(self, **metadata):
        self.metadata = metadata


def _reap(use_tmpfs, scratch_root, results=()):
    jobs = [Job(model=_TmpfsJobModel(use_tmpfs), env=None, source=None,
                receiver=None)]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _reap_scratch_root(str(scratch_root), jobs, results)
    return [str(w.message) for w in caught]


@pytest.mark.skipif(not FileManager._tmpfs_available(),
                    reason='no writable /dev/shm')
def test_kept_tmpfs_work_dirs_are_reported_even_though_the_root_is_empty(
        tmp_path):
    scratch_root = tmp_path / 'scratch'
    scratch_root.mkdir()
    messages = _reap(True, scratch_root)
    assert any('/dev/shm' in m for m in messages), messages
    assert any('use_tmpfs=True' in m for m in messages), messages


@pytest.mark.skipif(not FileManager._tmpfs_available(),
                    reason='no writable /dev/shm')
def test_a_kept_tmpfs_work_dir_is_named_in_the_warning(tmp_path):
    """/dev/shm is shared with every other process, so naming only the root
    leaves the caller to work out which of its entries are this batch's — the
    leftovers warning names its directory outright."""
    scratch_root = tmp_path / 'scratch'
    scratch_root.mkdir()
    results = [_ResultWithFiles(shd_file='/dev/shm/bellhop_ab12/run.shd',
                                prt_file='/dev/shm/bellhop_ab12/run.prt'),
               _ResultWithFiles(mod_file='/dev/shm/kraken_cd34/run.mod')]
    messages = _reap(True, scratch_root, results)
    assert any('/dev/shm/bellhop_ab12' in m for m in messages), messages
    assert any('/dev/shm/kraken_cd34' in m for m in messages), messages


def test_a_kept_work_dir_outside_tmpfs_is_left_out_of_the_ram_warning(tmp_path):
    """Only the dirs actually holding RAM belong in the list: a tmpfs request
    falls back to disk where /dev/shm is unwritable, and those land under the
    scratch root the other warning already covers."""
    from uacpy.parallel import _kept_tmpfs_dirs

    results = [_ResultWithFiles(shd_file=str(tmp_path / 'bellhop_ab12/run.shd')),
               _ResultWithFiles(shd_file='/dev/shm/bellhop_ef56/run.shd')]
    assert _kept_tmpfs_dirs(results) == ['/dev/shm/bellhop_ef56']


def test_a_result_with_no_output_paths_leaves_the_warning_unnamed(tmp_path):
    """A job that raised before writing an output has no ``*_file`` metadata,
    so its dir cannot be named; the warning still reports the RAM as held."""
    scratch_root = tmp_path / 'scratch'
    scratch_root.mkdir()
    messages = _reap(True, scratch_root, [_ResultWithFiles(), None])
    assert any('/dev/shm' in m for m in messages), messages
    assert any('Remove them' in m for m in messages), messages


def test_a_kept_disk_work_dir_is_not_reported_as_ram_backed(tmp_path):
    scratch_root = tmp_path / 'scratch'
    (scratch_root / 'job_abc').mkdir(parents=True)
    messages = _reap(False, scratch_root)
    assert any(str(scratch_root) in m for m in messages), messages
    assert not any('/dev/shm' in m for m in messages), messages
    assert scratch_root.exists()


def test_an_empty_scratch_root_with_no_keepers_is_removed(tmp_path):
    scratch_root = tmp_path / 'scratch'
    scratch_root.mkdir()
    jobs = [Job(model=_TmpfsJobModel(False), env=None, source=None,
                receiver=None)]
    jobs[0].model.cleanup = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        removed = _reap_scratch_root(str(scratch_root), jobs)
    assert removed is True
    assert not scratch_root.exists()
    assert [str(w.message) for w in caught] == []


def test_the_fork_reset_drops_inherited_work_dir_claims(tmp_path):
    """A forked child inherits the parent's claims and its lock; both are
    reset, or ``run_parallel(start_method='fork')`` would refuse a directory
    no live thread is using."""
    from uacpy.models.base import _reset_work_dir_claims

    if not hasattr(os, 'register_at_fork'):
        pytest.skip('no os.register_at_fork on this platform')
    shared = tmp_path / 'shared'
    fm = _StubModel(shared)._setup_file_manager()
    try:
        assert str(shared.resolve()) in _PINNED_WORK_DIRS
        _reset_work_dir_claims()
        assert _PINNED_WORK_DIRS == {}
        assert _StubModel(shared).run('after fork') == 'after fork'
    finally:
        fm.cleanup_work_dir()
