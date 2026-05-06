"""RAM parabolic-equation-focused tests."""

import pytest
import numpy as np

import uacpy
from uacpy.core.environment import SoundSpeedProfile
from uacpy.models import Bellhop, RAM, Kraken, KrakenC, Scooter, SPARC
from uacpy.models.base import RunMode
from uacpy.core import Environment, BoundaryProperties, Source, Receiver
from uacpy.core.exceptions import ExecutableNotFoundError, UnsupportedFeatureError

pytestmark = pytest.mark.requires_binary

class TestRAMAdvancedParameters:
    """Test RAM Pade orders and stability parameters."""

    @pytest.fixture
    def ram_env(self):
        return Environment(
            name="ram_test",
            depth=100.0,
            sound_speed=1500.0
        )

    @pytest.fixture
    def ram_source(self):
        return Source(depth=50.0, frequency=50.0)

    @pytest.fixture
    def ram_receiver(self):
        return Receiver(
            depths=np.linspace(10, 90, 9),
            ranges=np.linspace(100, 5000, 11)
        )

    @pytest.mark.parametrize('np_pade', [2, 6, 8])
    def test_ram_pade_order(self, ram_env, ram_source, ram_receiver, np_pade):
        """RAM converges across the supported Padé-coefficient counts."""
        ram = RAM(verbose=False)
        result = ram.compute_tl(
            env=ram_env,
            source=ram_source,
            receiver=ram_receiver,
            np_pade=np_pade,
        )
        assert result.field_type == 'tl'
        assert np.all(np.isfinite(result.data))

    def test_ram_stability_parameter(self, ram_env, ram_source, ram_receiver):
        """Test RAM stability parameter."""
        ram = RAM(verbose=False)
        result = ram.compute_tl(
            env=ram_env,
            source=ram_source,
            receiver=ram_receiver,
            ns_stability=1
        )
        assert result.field_type == 'tl'
        assert np.all(np.isfinite(result.data))

    def test_ram_custom_dr_dz(self, ram_env, ram_source, ram_receiver):
        """Test RAM with custom range and depth steps."""
        ram = RAM(verbose=False)
        result = ram.compute_tl(
            env=ram_env,
            source=ram_source,
            receiver=ram_receiver,
            dr=10.0,  # 10m range step
            dz=0.5    # 0.5m depth step
        )
        assert result.field_type == 'tl'
        assert np.all(np.isfinite(result.data))
