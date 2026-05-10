"""Tests for the parametric SSP factory methods on
:class:`SoundSpeedProfile` (``from_isothermal``,
``from_temperature_salinity``)."""

import numpy as np
import pytest

from uacpy.core.environment import SoundSpeedProfile


class TestIsothermal:
    def test_constant_value(self):
        ssp = SoundSpeedProfile.from_isothermal(c=1500.0, depth_max=2000.0)
        assert np.allclose(ssp.data, 1500.0)

    def test_default_value(self):
        ssp = SoundSpeedProfile.from_isothermal()
        assert np.allclose(ssp.data, 1500.0)


class TestTemperatureSalinity:
    def test_pure_water_surface_value(self):
        # T=15°C, S=35 PSU, z=0 → c ≈ 1506.69 m/s (Mackenzie 9-term).
        z = np.array([0.0])
        T = np.array([15.0])
        S = np.array([35.0])
        ssp = SoundSpeedProfile.from_temperature_salinity(z, T, S)
        assert float(ssp.data[0, 0]) == pytest.approx(1506.69, abs=0.05)

    def test_increasing_with_depth(self):
        z = np.linspace(0.0, 4000.0, 81)
        T = np.full_like(z, 4.0)         # cold deep water
        S = np.full_like(z, 35.0)
        ssp = SoundSpeedProfile.from_temperature_salinity(z, T, S)
        # In an isothermal column, c rises monotonically with z (pressure term)
        assert np.all(np.diff(ssp.data[:, 0]) > 0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="must share shape"):
            SoundSpeedProfile.from_temperature_salinity(
                depths=np.array([0.0, 100.0]),
                temperature_c=np.array([15.0]),
                salinity_psu=np.array([35.0, 35.0]),
            )
