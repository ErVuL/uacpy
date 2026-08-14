import numpy as np
"""Tests for the Francois-Garrison builder (uacpy.data.absorption)."""

import pytest

from uacpy.core.absorption import FrancoisGarrison
from uacpy.core.exceptions import ConfigurationError
from uacpy.data.absorption import build_francois_garrison


def test_builds_from_the_mid_depth_row_by_default():
    """The single row this picks sets the temperature the models use for the
    whole column (they vary only depth), so the representative choice is the
    middle of the profile, not its shallowest sample. See
    TestNominalRowIsColumnRepresentative for the dB consequence."""
    fg = build_francois_garrison([0.0, 50.0, 100.0], [18.0, 16.0, 13.0],
                                 [36.0, 36.1, 36.2])
    assert isinstance(fg, FrancoisGarrison)
    assert fg.temperature_c == 16.0      # mid-depth sample, not the surface
    assert fg.salinity_psu == 36.1
    assert fg.z_bar_m == 50.0
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


class TestNominalRowIsColumnRepresentative:
    """The models receive a **single** T/S row and re-evaluate the formula in
    depth only, so that row's temperature governs absorption for the whole
    column. Taking it at the surface carried the warmest water down the entire
    profile: on a mid-latitude column (22 C surface, 4 C at 2 km) that
    understated absorption by 34 % at 10 kHz and 20 % at 1 kHz against a
    mid-column reference."""

    Z = np.array([0.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0])
    T = np.array([22.0, 20.0, 16.0, 12.0, 8.0, 5.0, 4.0])
    S = np.full(7, 35.0)

    def _fg(self, ref=None):
        return build_francois_garrison(self.Z, self.T, self.S,
                                       reference_depth=ref)

    def test_default_takes_the_mid_depth_row(self):
        fg = self._fg()
        assert fg.z_bar_m == pytest.approx(1000.0)
        assert fg.temperature_c == pytest.approx(5.0)

    def test_explicit_reference_still_wins(self):
        # The discriminating counterpart: the default changed, the override
        # did not.
        assert self._fg(ref=0.0).temperature_c == pytest.approx(22.0)
        assert self._fg(ref=1000.0).temperature_c == pytest.approx(5.0)

    def test_surface_reference_understates_high_frequency_absorption(self):
        # Pins the reason the default moved, in dB rather than in degrees.
        zq = np.array([500.0])
        a_mid = float(np.ravel(self._fg().alpha_db_per_m(1e4, zq))[0])
        a_surf = float(np.ravel(self._fg(ref=0.0).alpha_db_per_m(1e4, zq))[0])
        assert a_surf < a_mid
        assert (a_mid - a_surf) / a_mid == pytest.approx(0.34, abs=0.05)

    def test_isothermal_column_is_unaffected_by_the_change(self):
        # Where there is no stratification there is nothing to choose, so the
        # new default must agree with the old one exactly.
        z = np.array([0.0, 100.0, 500.0])
        t = np.full(3, 12.0)
        s = np.full(3, 35.0)
        top = build_francois_garrison(z, t, s, reference_depth=0.0)
        mid = build_francois_garrison(z, t, s)
        assert mid.temperature_c == pytest.approx(top.temperature_c)
