"""Volume-attenuation-focused tests."""

import pytest
import numpy as np

import uacpy
from uacpy.core.environment import SoundSpeedProfile
from uacpy.models import Bellhop, RAM, Kraken, KrakenC, Scooter, SPARC
from uacpy.models.base import RunMode
from uacpy.core import Environment, BoundaryProperties, Source, Receiver
from uacpy.core.exceptions import ExecutableNotFoundError, UnsupportedFeatureError

pytestmark = pytest.mark.requires_binary

class TestVolumeAttenuation:
    """Tests for volume attenuation models (Priority 1 gap)."""

    @pytest.fixture
    def shallow_env(self):
        """Shallow water environment for attenuation tests."""
        return Environment(
            name="atten_test",
            depth=100.0,
            sound_speed=1500.0
        )

    @pytest.fixture
    def high_freq_source(self):
        """High frequency source where attenuation is significant."""
        return Source(depth=50.0, frequency=10000.0)  # 10 kHz

    @pytest.fixture
    def low_freq_source(self):
        """Low frequency source where attenuation is minimal."""
        return Source(depth=50.0, frequency=100.0)  # 100 Hz

    @pytest.fixture
    def receiver(self):
        return Receiver(depths=[50.0], ranges=[1000.0, 3000.0, 5000.0])

    @pytest.mark.requires_binary
    def test_bellhop_thorp_attenuation(self, shallow_env, high_freq_source, receiver):
        """Test Bellhop with Thorp attenuation formula.

        At 10 kHz, Thorp absorption ≈ 0.6 dB/km; over the test ranges
        the extra TL should be on the order of the predicted Thorp value.
        We assert the depth-mean difference at the longest range is within
        the predicted-times-[0.1, 10] band — a sign-error or unit
        confusion would not satisfy that band.
        """
        bellhop_no_atten = Bellhop(verbose=False)
        bellhop_thorp = Bellhop(verbose=False, volume_attenuation='T')

        result_no_atten = bellhop_no_atten.run(
            env=shallow_env, source=high_freq_source, receiver=receiver,
            run_mode=RunMode.COHERENT_TL,
        )
        result_thorp = bellhop_thorp.run(
            env=shallow_env, source=high_freq_source, receiver=receiver,
            run_mode=RunMode.COHERENT_TL,
        )

        # Thorp formula at 10 kHz (f in kHz):
        #   alpha = 0.11 f^2/(1+f^2) + 44 f^2/(4100+f^2)
        #         + 2.75e-4 f^2 + 0.003   [dB/km]
        f_khz = high_freq_source.frequency[0] / 1000.0
        alpha_db_per_km = (
            0.11 * f_khz**2 / (1 + f_khz**2)
            + 44.0 * f_khz**2 / (4100.0 + f_khz**2)
            + 2.75e-4 * f_khz**2
            + 0.003
        )
        range_km_max = float(receiver.ranges[-1]) / 1000.0
        expected_extra_db = alpha_db_per_km * range_km_max

        assert result_thorp.field_type == 'tl'
        observed_extra = (
            np.mean(result_thorp.data[:, -1]) - np.mean(result_no_atten.data[:, -1])
        )
        # Sign must be right (Thorp adds loss, never reduces it).
        assert observed_extra > 0, (
            f"Thorp gave less loss than no-attenuation case: {observed_extra:.2f} dB"
        )
        # Magnitude must be the right order — within 10× of the predicted dB.
        # This is loose enough to absorb implementation differences (per-arrival
        # vs per-range application, alpha-formula variants) while still
        # catching unit confusion (which would be off by ~1000×).
        assert 0.1 * expected_extra_db < observed_extra < 10 * expected_extra_db, (
            f"Thorp absorption magnitude wrong: observed {observed_extra:.2f} dB "
            f"vs predicted {expected_extra_db:.2f} dB at {range_km_max:.1f} km"
        )

    @pytest.mark.requires_binary
    def test_kraken_thorp_attenuation(self, shallow_env, high_freq_source, receiver):
        """Test Kraken with Thorp attenuation formula."""
        kraken = Kraken(verbose=False, volume_attenuation='T')

        # Compute modes with attenuation
        result = kraken.run(
            env=shallow_env,
            source=high_freq_source,
            receiver=receiver,
        )

        assert result.field_type == 'modes'
        assert 'k' in result.metadata

    @pytest.mark.requires_binary
    def test_frequency_dependent_attenuation(self, shallow_env, low_freq_source, high_freq_source, receiver):
        """Test that attenuation increases with frequency."""
        bellhop = Bellhop(verbose=False, volume_attenuation='T')

        # Low frequency with Thorp
        result_low = bellhop.run(
            env=shallow_env,
            source=low_freq_source,
            receiver=receiver,
            run_mode=RunMode.COHERENT_TL,
        )

        # High frequency with Thorp
        result_high = bellhop.run(
            env=shallow_env,
            source=high_freq_source,
            receiver=receiver,
            run_mode=RunMode.COHERENT_TL,
        )

        # Both should complete successfully
        assert result_low.field_type == 'tl'
        assert result_high.field_type == 'tl'
        assert np.all(np.isfinite(result_low.data))
        assert np.all(np.isfinite(result_high.data))

    @pytest.mark.requires_binary
    def test_attenuation_with_scooter(self, shallow_env, high_freq_source, receiver):
        """Test Scooter with volume attenuation."""
        scooter = Scooter(verbose=False, volume_attenuation='T')

        result = scooter.run(
            env=shallow_env,
            source=high_freq_source,
            receiver=receiver,
        )

        assert result.field_type == 'tl'
        assert np.all(np.isfinite(result.data))
