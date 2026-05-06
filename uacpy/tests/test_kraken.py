"""Kraken / KrakenC normal-mode-focused tests."""

import pytest
import numpy as np

import uacpy
from uacpy.core.environment import SoundSpeedProfile
from uacpy.models import Bellhop, RAM, Kraken, KrakenC, Scooter, SPARC
from uacpy.models.base import RunMode
from uacpy.core import Environment, BoundaryProperties, Source, Receiver
from uacpy.core.exceptions import ExecutableNotFoundError, UnsupportedFeatureError

pytestmark = pytest.mark.requires_binary

class TestKrakenCComplexModes:
    """Test KrakenC for complex modes with elastic bottom."""

    @pytest.fixture
    def elastic_env(self):
        """Create environment with elastic bottom."""
        bottom = BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600.0,
            shear_speed=400.0,
            density=1.8,
            attenuation=0.2,
            shear_attenuation=0.5
        )
        return Environment(
            name="krakenc_test",
            depth=100.0,
            sound_speed=1500.0,
            bottom=bottom
        )


    @pytest.fixture
    def receiver(self):
        return Receiver(depths=[25.0, 50.0, 75.0], ranges=[1000.0, 3000.0])

    @pytest.mark.requires_binary
    def test_krakenc_complex_modes(self, elastic_env, source, receiver):
        """Test KrakenC complex mode computation."""
        krakenc = KrakenC(verbose=False)

        modes = krakenc.run(
            env=elastic_env,
            source=source,
            receiver=receiver
        )

        assert modes.field_type == 'modes'
        assert 'k' in modes.metadata
        assert len(modes.metadata['k']) > 0

        # Complex modes should have complex wavenumbers
        k = modes.metadata['k']
        assert np.any(np.imag(k) != 0), "Should have complex wavenumbers for elastic bottom"
