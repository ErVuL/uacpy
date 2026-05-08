"""SSP-interpolation method-focused tests."""

import pytest
import numpy as np

import uacpy
from uacpy.core.environment import SoundSpeedProfile
from uacpy.models import Bellhop, RAM, Kraken, KrakenC, Scooter, SPARC
from uacpy.models.base import RunMode
from uacpy.core import Environment, BoundaryProperties, Source, Receiver
from uacpy.core.exceptions import ExecutableNotFoundError, UnsupportedFeatureError

pytestmark = pytest.mark.requires_binary

class TestSSPInterpolationMethods:
    """Test different SSP interpolation types."""


    @pytest.fixture
    def receiver(self):
        return Receiver(
            depths=np.array([25.0, 50.0, 75.0]),
            ranges=np.array([1000.0, 3000.0])
        )

    @pytest.mark.requires_binary
    def test_ssp_isovelocity(self, source, receiver):
        """Test isovelocity SSP."""
        env = Environment(
            name="iso_test",
            bathymetry=100.0,
            sound_speed=1500.0
        )

        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env=env, source=source, receiver=receiver)
        assert result.field_type == 'tl'

    @pytest.mark.requires_binary
    def test_ssp_linear(self, source, receiver):
        """Test linear SSP interpolation."""
        depths = np.array([0, 50, 100])
        speeds = np.array([1500, 1490, 1480])

        env = Environment(
            name="linear_test",
            bathymetry=100.0,
            ssp=SoundSpeedProfile.from_pairs(np.column_stack([depths, speeds]), interp='linear')
        )

        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env=env, source=source, receiver=receiver)
        assert result.field_type == 'tl'

    @pytest.mark.requires_binary
    def test_ssp_cubic(self, source, receiver):
        """Test cubic spline SSP interpolation."""
        depths = np.array([0, 25, 50, 75, 100])
        speeds = np.array([1500, 1495, 1490, 1485, 1480])

        env = Environment(
            name="cubic_test",
            bathymetry=100.0,
            ssp=SoundSpeedProfile.from_pairs(np.column_stack([depths, speeds]), interp='cubic')
        )

        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env=env, source=source, receiver=receiver)
        assert result.field_type == 'tl'
