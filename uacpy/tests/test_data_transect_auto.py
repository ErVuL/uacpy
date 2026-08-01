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


def test_bathy_fetch_samples_exactly_where_the_plan_says(monkeypatch):
    # The fetcher resolves its sampling *through* the plan, so the two can
    # never drift apart (the SSP pair already works this way).
    seen = {}

    def fake_depths(points, **kwargs):
        seen['lats'] = [la for la, _ in points]
        seen['lons'] = [lo for _, lo in points]
        return np.full(len(points), 1500.0)

    monkeypatch.setattr(bath, '_fetch_depths', fake_depths)
    a, b = (45.0, -2.5), (45.4, -2.5)
    plan = bath.bathy_transect_plan(a, b, n_points='auto', max_points=30)
    out = bath.fetch_bathy_transect(a, b, n_points='auto', max_points=30)
    assert out.shape == (plan['n_points'], 2)
    assert np.allclose(seen['lats'], plan['lats'])
    assert np.allclose(seen['lons'], plan['lons'])
    assert np.allclose(out[:, 0], plan['ranges_m'])


def test_bathy_plan_reports_the_uncapped_native_count(monkeypatch):
    # 'native_points' is what the max_points warning quotes, so it must survive
    # the cap rather than being recomputed at the warning site.
    monkeypatch.setattr(bath, '_fetch_depths',
                        lambda points, **kw: np.full(len(points), 1500.0))
    plan = bath.bathy_transect_plan((0.0, -40.0), (0.0, 40.0), n_points='auto',
                                    max_points=50)
    assert plan['n_points'] == 50
    assert plan['native_points'] > 50
    with pytest.warns(UserWarning, match=f"~{plan['native_points']} points"):
        bath.fetch_bathy_transect((0.0, -40.0), (0.0, 40.0), n_points='auto',
                                  max_points=50)


def test_bathy_plan_rejects_a_degenerate_count():
    with pytest.raises(ConfigurationError, match='n_points'):
        bath.bathy_transect_plan((0.0, 0.0), (1.0, 1.0), n_points=1)


def test_module_all_covers_what_the_package_reexports():
    # A name re-exported from uacpy.data (or imported by a sibling) but absent
    # from its own module's __all__ reads as private and drops out of `from
    # <module> import *` and the API docs.
    import uacpy.data as data
    from uacpy.data import seaice_local
    expected = {
        bath: ['fetch_bathy', 'fetch_bathy_transect', 'bathy_transect_plan',
               'fetch_bathy_grid', 'transect_length'],
        ss: ['fetch_ssp', 'fetch_ssp_transect', 'ssp_transect_plan',
             'fetch_ts_profile', 'assemble_range_dependent',
             'extend_ssp_below_data'],
        seaice_local: ['fetch_sea_ice_concentration', 'sea_ice_surface',
                       'fetch_sea_ice_surface', 'sea_ice_surface_transect'],
    }
    for module, names in expected.items():
        missing = [n for n in names if n not in module.__all__]
        assert not missing, f"{module.__name__}.__all__ omits {missing}"
    # sea_ice_surface_transect is the one of these the package re-exports and
    # fetch_environment calls, so it must also resolve on uacpy.data.
    assert 'sea_ice_surface_transect' in data.__all__
    assert callable(data.sea_ice_surface_transect)
