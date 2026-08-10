"""Tests for SSP factory methods on :class:`SoundSpeedProfile`."""

import numpy as np
import pytest

from uacpy.core.environment import SoundSpeedProfile
from uacpy.core.exceptions import ConfigurationError


class TestIsovelocity:
    def test_constant_value(self):
        ssp = SoundSpeedProfile.from_isovelocity(depth_max=2000.0, sound_speed=1500.0)
        assert np.allclose(ssp.data, 1500.0)

    def test_default_value(self):
        ssp = SoundSpeedProfile.from_isovelocity(depth_max=5000.0)
        assert np.allclose(ssp.data, 1500.0)

    def test_value_returns_scalar_for_isovelocity(self):
        ssp = SoundSpeedProfile.from_isovelocity(depth_max=2000.0, sound_speed=1480.0)
        assert ssp.value == 1480.0

    def test_value_raises_when_profile_varies(self):
        ssp = SoundSpeedProfile.from_pairs([(0, 1500.0), (100, 1480.0)])
        with pytest.raises(ConfigurationError, match="varies"):
            _ = ssp.value


class TestMackenzie:
    def test_seawater_surface_value_at_the_mackenzie_reference_point(self):
        # T=15 °C, S=35 PSU, D=0 m is the reference point of Mackenzie (1981);
        # the nine-term polynomial collapses to its first four terms there
        # (every S-35 and D factor vanishes) and evaluates to 1506.692225 m/s.
        # ``abs=0.05`` only buys room to write the expectation to two decimals
        # — the closed form leaves no discretisation error to absorb.
        z = np.array([0.0])
        T = np.array([15.0])
        S = np.array([35.0])
        ssp = SoundSpeedProfile.from_mackenzie(z, T, S)
        assert float(ssp.data[0, 0]) == pytest.approx(1506.69, abs=0.05)

    def test_increasing_with_depth(self):
        z = np.linspace(0.0, 4000.0, 81)
        T = np.full_like(z, 4.0)
        S = np.full_like(z, 35.0)
        ssp = SoundSpeedProfile.from_mackenzie(z, T, S)
        assert np.all(np.diff(ssp.data[:, 0]) > 0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ConfigurationError, match="must share shape"):
            SoundSpeedProfile.from_mackenzie(
                depths=np.array([0.0, 100.0]),
                temperature_c=np.array([15.0]),
                salinity_psu=np.array([35.0, 35.0]),
            )


class TestIsovelocityShapeMustBeTrue:
    """``resolve_ssp_topopt`` turns ``shape='isovelocity'`` into AT
    ``TopOpt(1)='C'`` on the grounds that any connection scheme over constant
    data is constant. That reasoning only holds if the data *is* constant, so
    the declaration is checked rather than trusted — otherwise the deck
    silently flattens a gradient."""

    def test_a_gradient_declared_isovelocity_is_rejected(self):
        with pytest.raises(ConfigurationError, match='isovelocity'):
            SoundSpeedProfile.from_pairs(
                np.array([[0.0, 1500.0], [200.0, 1400.0]]),
                shape='isovelocity')

    def test_constant_data_is_accepted(self):
        ssp = SoundSpeedProfile.from_pairs(
            np.array([[0.0, 1500.0], [200.0, 1500.0]]), shape='isovelocity')
        assert ssp.shape == 'isovelocity'

    def test_the_shape_does_not_swallow_an_invalid_interp(self):
        """``resolve_ssp_topopt`` validates the model's ``interp_ssp`` before
        the ``shape='isovelocity'`` shortcut returns ``'C'``
        (``io/oalib_writer.py:360-368``), so an unrecognised scheme raises on
        an isovelocity env exactly as it does on any other."""
        from uacpy.core.environment import Environment
        from uacpy.io.oalib_writer import resolve_ssp_topopt
        env = Environment(bathymetry=200.0, ssp=SoundSpeedProfile.from_pairs(
            np.array([[0.0, 1500.0], [200.0, 1500.0]]), shape='isovelocity'))
        assert resolve_ssp_topopt(env, 'linear') == 'C'
        for bad in ('analytic', 'bogus'):
            with pytest.raises(ConfigurationError):
                resolve_ssp_topopt(env, bad)
