"""Tests for the Francois-Garrison builder (uacpy.data.absorption)."""

import pytest

from uacpy.core.absorption import FrancoisGarrison
from uacpy.core.exceptions import ConfigurationError
from uacpy.data.absorption import build_francois_garrison


def test_builds_from_surface_by_default():
    fg = build_francois_garrison([0.0, 50.0, 100.0], [18.0, 16.0, 13.0],
                                 [36.0, 36.1, 36.2])
    assert isinstance(fg, FrancoisGarrison)
    assert fg.temperature_c == 18.0      # shallowest sample
    assert fg.salinity_psu == 36.0
    assert fg.z_bar_m == 0.0
    assert fg.pH == 8.1                   # default open-ocean


def test_reference_depth_picks_nearest_level():
    fg = build_francois_garrison([0.0, 50.0, 100.0], [18.0, 16.0, 13.0],
                                 [36.0, 36.1, 36.2], reference_depth=60.0, pH=7.9)
    assert fg.z_bar_m == 50.0             # nearest to 60 m
    assert fg.temperature_c == 16.0
    assert fg.pH == 7.9


def test_mismatched_or_empty_raises():
    with pytest.raises(ConfigurationError):
        build_francois_garrison([], [], [])
    with pytest.raises(ConfigurationError):
        build_francois_garrison([0.0, 10.0], [18.0], [36.0])
