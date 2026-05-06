"""SPARC time-domain-focused tests."""

import pytest
import numpy as np

import uacpy
from uacpy.core.environment import SoundSpeedProfile
from uacpy.models import Bellhop, RAM, Kraken, KrakenC, Scooter, SPARC
from uacpy.models.base import RunMode
from uacpy.core import Environment, BoundaryProperties, Source, Receiver
from uacpy.core.exceptions import ExecutableNotFoundError, UnsupportedFeatureError

pytestmark = pytest.mark.requires_binary

class TestSPARCBasic:
    """Basic tests for SPARC model (seismo-acoustic PE)."""

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_sparc_basic_tl(self):
        """Test basic SPARC TL computation."""
        env = Environment(
            name="sparc_test",
            depth=100.0,
            sound_speed=1500.0
        )
        source = Source(depth=50.0, frequency=50.0)
        receiver = Receiver(
            depths=np.linspace(10, 90, 9),
            ranges=np.linspace(100, 5000, 11)
        )

        sparc = SPARC(verbose=False)
        result = sparc.compute_tl(env=env, source=source, receiver=receiver)

        assert result.field_type == 'tl'
        assert np.all(np.isfinite(result.data))
