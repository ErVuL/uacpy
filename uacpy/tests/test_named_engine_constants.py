"""The engines' default factors hold their values, and the code that
applies each one reads the name (measured through the public resolvers)."""
import numpy as np
import pytest

import uacpy
from uacpy.models.bellhop._plan import resolve_ray_box
from uacpy.core.deck_limits import NO_RECEIVER_RANGE_FALLBACK
from uacpy.io.bellhop_writer import BELLHOP_RAY_BOX_FACTOR, bellhop_ray_box
from uacpy.models.bounce import _plan as bounce_plan
from uacpy.models.kraken import _grid as kraken_grid
from uacpy.models.oases import oasr
from uacpy.models.scooter import _plan as scooter_plan
from uacpy.models.sparc import _plan as sparc_plan


def test_the_factors_hold_their_values():
    assert scooter_plan.SCOOTER_RMAX_FACTOR == 2.0
    assert sparc_plan.SPARC_RMAX_TIME_MARGIN == 1.2
    assert sparc_plan.SPARC_WINDOW_TRAVEL_TIMES == 2.5
    assert sparc_plan.SPARC_C_HIGH_UNBOUNDED == 1.0e7
    # on the auto window the default RMax is 4 x the farthest range
    assert 1.0 + sparc_plan.SPARC_RMAX_TIME_MARGIN * \
        sparc_plan.SPARC_WINDOW_TRAVEL_TIMES == pytest.approx(4.0)
    assert kraken_grid.KRAKEN_RMAX_NO_RANGE == 100_000.0
    assert BELLHOP_RAY_BOX_FACTOR == 1.2
    assert NO_RECEIVER_RANGE_FALLBACK == 10_000.0
    assert oasr.OASR_DEFAULT_ANGLES_DEG == (0.0, 90.0, 181)


def test_scooter_resolves_its_named_multiplier():
    value, _ = scooter_plan.resolve_rmax_factor(rmax_factor=None)
    assert value == scooter_plan.SCOOTER_RMAX_FACTOR


def test_sparc_resolves_its_named_margin():
    rmax, _, _ = sparc_plan.resolve_rmax(1000.0, 1500.0, 2.0,
                                         rmax_factor=None)
    assert rmax == 1000.0 + sparc_plan.SPARC_RMAX_TIME_MARGIN * 1500.0 * 2.0


class _Rx:
    def __init__(self, ranges):
        self.ranges = np.asarray(ranges, dtype=float)
        self.range_max = float(np.max(self.ranges))


def test_kraken_resolves_the_farthest_receiver_range():
    rmax, _ = kraken_grid.resolve_rmax(_Rx([1000.0, 4000.0]),
                                       pinned_rmax=None)
    assert rmax == 4000.0


def test_kraken_falls_back_to_the_named_range():
    rmax, _ = kraken_grid.resolve_rmax(_Rx([0.0]), pinned_rmax=None)
    assert rmax == kraken_grid.KRAKEN_RMAX_NO_RANGE
    assert kraken_grid.compute_rmax(None) == \
        kraken_grid.KRAKEN_RMAX_NO_RANGE


def _env():
    return uacpy.Environment(name='box', bathymetry=200.0, ssp=1500.0)


def test_bellhop_pads_the_ray_box_by_the_named_factor():
    rx = uacpy.Receiver(depths=[50.0], ranges=[1000.0, 5000.0])
    z, r = bellhop_ray_box(_env(), rx)
    assert (z, r) == (BELLHOP_RAY_BOX_FACTOR * 200.0,
                      BELLHOP_RAY_BOX_FACTOR * 5000.0)
    _, z_origin, _, r_origin = resolve_ray_box(_env(), rx, z_box=None, r_box=None)
    assert z_origin == '1.2 x env.depth'
    assert r_origin == '1.2 x receiver.range_max'


def test_bellhop_falls_back_to_the_named_range():
    rx = uacpy.Receiver(depths=[50.0], ranges=[0.0])
    _, r = bellhop_ray_box(_env(), rx)
    assert r == NO_RECEIVER_RANGE_FALLBACK
    _, _, _, r_origin = resolve_ray_box(_env(), rx, z_box=None, r_box=None)
    assert r_origin == '10 km (every receiver range is 0)'


def test_bounce_falls_back_to_the_same_range():
    rmax, origin = bounce_plan.resolve_rmax(
        _Rx([0.0]), 100.0, 1400.0, n_angles=None, rmax=None, c_high=1e9)
    assert rmax == NO_RECEIVER_RANGE_FALLBACK
    assert origin == 'the 10 km fallback (no receiver range)'


def test_oasr_samples_the_named_default_grid():
    lo, hi, n = oasr.OASR_DEFAULT_ANGLES_DEG
    grid = oasr.default_grazing_angles()
    assert np.array_equal(grid, np.linspace(0, 90, 181))
    assert np.array_equal(grid, np.linspace(lo, hi, n))
    grid[0] = -1.0
    assert oasr.default_grazing_angles()[0] == 0.0
