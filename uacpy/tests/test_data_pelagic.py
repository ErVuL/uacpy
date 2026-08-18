"""Tests for the pelagic open-ocean sediment fallback (uacpy.data.pelagic)."""

import numpy as np
import pytest

from uacpy.core.environment import BoundaryProperties, Bottom
from uacpy.data import pelagic
from uacpy.data.sediment import bottom_from_grain_size


@pytest.mark.parametrize('depth_m, lat, litho', [
    (5500.0, 0.0, 'pelagic clay'),      # abyssal below the CCD, low latitude
    (3000.0, 10.0, 'calcareous ooze'),  # above the CCD, low–mid latitude
    (2000.0, 65.0, 'diatom ooze'),      # high-latitude siliceous belt
    (6000.0, -60.0, 'diatom ooze'),     # Southern Ocean wins over depth
])
def test_lithology_zones(depth_m, lat, litho):
    assert pelagic.pelagic_lithology(depth_m, lat) == litho


def test_grain_size_matches_lithology():
    assert pelagic.pelagic_grain_size(5500.0, 0.0) == 9.0      # clay
    assert pelagic.pelagic_grain_size(3000.0, 0.0) == 7.5      # calcareous


@pytest.mark.parametrize('lon, litho', [
    (175.0, 'diatom ooze'),     # western Bering Sea
    (-175.0, 'diatom ooze'),    # eastern Bering Sea (sector crosses 180°)
    (145.0, 'diatom ooze'),     # Sea of Okhotsk
    (-140.0, 'diatom ooze'),    # Gulf of Alaska
    (-30.0, 'calcareous ooze'),  # subpolar North Atlantic: carbonate, not silica
    (5.0, 'calcareous ooze'),    # Norwegian Sea
])
def test_northern_siliceous_belt_is_pacific_only(lon, litho):
    """Regression: Berger's northern siliceous belt is the subarctic Pacific;
    applied symmetrically it misclassified the North Atlantic as diatom ooze."""
    assert pelagic.pelagic_lithology(2000.0, 55.0, lon) == litho
    # The Southern Ocean belt stays circumglobal.
    assert pelagic.pelagic_lithology(2000.0, -55.0, lon) == 'diatom ooze'


def test_fetch_bottom_belt_is_pacific_only_in_the_north():
    pacific = pelagic.fetch_bottom_pelagic((55.0, -175.0), depth=2000.0)
    atlantic = pelagic.fetch_bottom_pelagic((55.0, -30.0), depth=2000.0)
    assert pacific.grain_size_phi == 9.0     # diatom ooze
    assert atlantic.grain_size_phi == 7.5    # calcareous ooze, not diatom


def test_fetch_bottom_with_explicit_depth():
    bp = pelagic.fetch_bottom_pelagic((0.0, -150.0), depth=5500.0)
    assert isinstance(bp, BoundaryProperties)
    assert bp.acoustic_type == 'half-space'
    assert bp.grain_size_phi == 9.0


def test_fetch_bottom_fetches_depth(monkeypatch):
    monkeypatch.setattr(pelagic, '_water_depth', lambda *a, **k: 3000.0)
    bp = pelagic.fetch_bottom_pelagic((10.0, 20.0))
    assert bp.grain_size_phi == 7.5                            # calcareous (above CCD)


def test_transect_classifies_each_waypoint_from_its_own_cell():
    # 30°W meridian, waypoints (-60, -30), (-30, -30), (0, -30). The depth
    # callable returns 3000 m south of 15°S and 5000 m north of it, so the
    # three waypoints span three provinces: Southern Ocean siliceous belt →
    # diatom ooze (ϕ 9.0), above-CCD mid-latitude → calcareous ooze (ϕ 7.5),
    # below-CCD equator → pelagic clay (ϕ 9.0).
    def depth(la, lo):
        return 5000.0 if la > -15.0 else 3000.0

    assert [pelagic.pelagic_lithology(depth(la, -30.0), la, -30.0)
            for la in (-60.0, -30.0, 0.0)] == \
        ['diatom ooze', 'calcareous ooze', 'pelagic clay']
    rdb = pelagic.fetch_bottom_pelagic_transect((-60.0, -30.0), (0.0, -30.0),
                                                n_points=3, depth=depth)
    assert isinstance(rdb, Bottom)
    assert rdb.ranges.shape == (3,) and rdb.ranges[0] == 0.0
    assert np.all(np.diff(rdb.ranges) > 0)
    # Each column carries its own waypoint's lithology → ϕ → Hamilton c_p:
    # ϕ 9.0 → 1494.90 m/s at the ends, ϕ 7.5 → 1513.71 m/s in the middle.
    expected = [bottom_from_grain_size(phi).sound_speed
                for phi in (9.0, 7.5, 9.0)]
    assert rdb.halfspace_sound_speed.tolist() == pytest.approx(expected)
    assert rdb.halfspace_sound_speed[0] == pytest.approx(1494.90, abs=0.01)
    assert rdb.halfspace_sound_speed[1] == pytest.approx(1513.71, abs=0.01)
