"""Tests for the pelagic open-ocean sediment fallback (uacpy.data.pelagic)."""

import numpy as np
import pytest

from uacpy.core.environment import BoundaryProperties, Bottom
from uacpy.data import pelagic


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


def test_fetch_bottom_with_explicit_depth():
    bp = pelagic.fetch_bottom_pelagic((0.0, -150.0), depth=5500.0)
    assert isinstance(bp, BoundaryProperties)
    assert bp.acoustic_type == 'half-space'
    assert bp.grain_size_phi == 9.0


def test_fetch_bottom_fetches_depth(monkeypatch):
    monkeypatch.setattr(pelagic, '_water_depth', lambda *a, **k: 3000.0)
    bp = pelagic.fetch_bottom_pelagic((10.0, 20.0))
    assert bp.grain_size_phi == 7.5                            # calcareous (above CCD)


def test_transect(monkeypatch):
    depths = iter([5500.0, 3000.0, 2000.0])
    monkeypatch.setattr(pelagic, '_water_depth', lambda *a, **k: next(depths))
    rdb = pelagic.fetch_bottom_pelagic_transect((0.0, 0.0), (2.0, 0.0), n_points=3)
    assert isinstance(rdb, Bottom)
    assert rdb.halfspace_sound_speed.shape == (3,)
    assert np.all(np.isfinite(rdb.halfspace_sound_speed)) and np.all(rdb.halfspace_sound_speed > 1400.0)
