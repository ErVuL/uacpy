"""Auto transect sampling: distinct-cell reduction, endpoint anchoring,
monotonic ranges, and the max_points cap. All offline — the SSP/bathy *plans*
are analytic, and the fetch paths are stubbed."""

import numpy as np
import pytest

from uacpy.data._geo import run_representative_indices
from uacpy.data import sound_speed as ss
from uacpy.data import bathymetry as bath
from uacpy.core.environment import SoundSpeedProfile
from uacpy.core.exceptions import ConfigurationError


# ── the reduction primitive ─────────────────────────────────────────────────

def test_run_representatives_anchor_and_monotonic():
    # interior runs -> midpoints; first/last -> endpoints (anchored).
    assert run_representative_indices([0, 0, 0, 1, 1, 2, 2, 2, 2]) == [0, 3, 8]
    assert run_representative_indices([0, 1]) == [0, 1]
    assert run_representative_indices([5, 5, 5]) == [0]          # one run


@pytest.mark.parametrize("seed", range(50))
def test_run_representatives_strictly_increasing(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 80))
    keys = list(rng.integers(0, 4, size=n))
    reps = run_representative_indices(keys)
    assert reps == sorted(reps) and len(set(reps)) == len(reps)
    assert reps[0] == 0 and reps[-1] <= n - 1
    n_runs = 1 + sum(1 for a, b in zip(keys, keys[1:]) if a != b)
    if n_runs > 1:
        assert reps[-1] == n - 1            # last run anchored to the end


# ── SSP 'auto' plan (analytic — no network) ─────────────────────────────────

def test_ssp_plan_auto_collapses_to_distinct_cells():
    # ~330 km across several 1deg WOA cells -> a handful of columns, not 1000.
    p = ss.ssp_transect_plan((45.0, -2.5), (45.5, -6.5), n_points='auto')
    assert 2 <= p['n_points'] <= 8
    assert np.all(np.diff(p['ranges_m']) > 0)   # strictly increasing
    assert p['ranges_m'][0] == pytest.approx(0.0)   # anchored start


def test_ssp_plan_auto_anchors_full_span():
    from uacpy.data._geo import geodesic_waypoints
    a, b = (10.0, 0.0), (10.0, 5.0)
    _, _, dense = geodesic_waypoints(a, b, 200)
    p = ss.ssp_transect_plan(a, b, n_points='auto')
    assert p['ranges_m'][0] == pytest.approx(0.0)
    assert p['ranges_m'][-1] == pytest.approx(dense[-1])


def test_ssp_plan_explicit_and_cap():
    p = ss.ssp_transect_plan((0.0, 0.0), (0.0, 2.0), n_points=4)
    assert p['n_points'] == 4
    # auto over a huge transect is capped by max_points
    p2 = ss.ssp_transect_plan((0.0, -60.0), (0.0, 60.0),
                              n_points='auto', max_points=12)
    assert p2['n_points'] <= 12
    with pytest.raises(ConfigurationError):
        ss.ssp_transect_plan((0.0, 0.0), (0.0, 1.0), n_points=1)


def test_ssp_fetch_auto_fetches_once_per_cell(monkeypatch):
    # Stub fetch_ssp: a distinct column per 1deg lon cell. The auto fetch must
    # call it exactly n_points (plan) times — one per distinct cell, no refetch.
    calls = []

    def fake_fetch_ssp(point, **kw):
        lat, lon = point
        cell = int(np.floor(lon))
        calls.append(cell)
        c0 = 1500.0 + 10.0 * cell
        return SoundSpeedProfile(depths=np.array([0.0, 100.0]),
                                 data=np.array([[c0], [c0 + 20.0]]))

    monkeypatch.setattr(ss, 'fetch_ssp', fake_fetch_ssp)
    a, b = (10.0, -2.5), (10.0, 3.5)
    plan = ss.ssp_transect_plan(a, b, n_points='auto')
    prof = ss.fetch_ssp_transect(a, b, n_points='auto')
    assert prof.n_ranges == plan['n_points']        # one column per distinct cell
    assert len(calls) == plan['n_points']           # fetched once each, no refetch
    assert np.all(np.diff(prof.ranges) > 0)         # monotonic range axis


# ── bathy 'auto' plan (continuous → native count, capped) ───────────────────

def test_bathy_plan_auto_native_and_cap():
    short = bath.bathy_transect_plan((45.0, -2.5), (45.1, -2.5), n_points='auto')
    # short transect: native (length / ~0.45 km) below the default cap
    assert short['n_points'] < 200
    long = bath.bathy_transect_plan((0.0, -40.0), (0.0, 40.0), n_points='auto')
    assert long['n_points'] == 1000                 # capped at default max_points
    assert np.all(np.diff(long['ranges_m']) > 0)
