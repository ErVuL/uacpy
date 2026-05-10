"""Tests for SSP factory methods on :class:`SoundSpeedProfile`."""

import numpy as np
import pytest

from uacpy.core.environment import SoundSpeedProfile


class TestIsovelocity:
    def test_constant_value(self):
        ssp = SoundSpeedProfile.from_isovelocity(depth_max=2000.0, sound_speed=1500.0)
        assert np.allclose(ssp.data, 1500.0)

    def test_default_value(self):
        ssp = SoundSpeedProfile.from_isovelocity(depth_max=5000.0)
        assert np.allclose(ssp.data, 1500.0)


class TestMackenzie:
    def test_pure_water_surface_value(self):
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
        with pytest.raises(ValueError, match="must share shape"):
            SoundSpeedProfile.from_mackenzie(
                depths=np.array([0.0, 100.0]),
                temperature_c=np.array([15.0]),
                salinity_psu=np.array([35.0, 35.0]),
            )
