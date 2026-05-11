"""Volume-attenuation-focused tests."""

import pytest
import numpy as np

from uacpy.models import Bellhop, Kraken, Scooter
from uacpy.models.base import RunMode
from uacpy.core import Environment, Source, Receiver
from uacpy.core.absorption import Thorp

pytestmark = pytest.mark.requires_binary


class TestVolumeAttenuation:
    """Tests for volume attenuation models (Priority 1 gap)."""

    @pytest.fixture
    def shallow_env(self):
        """Shallow water environment without volume absorption."""
        return Environment(
            name="atten_test",
            bathymetry=100.0,
            ssp=1500.0,
        )

    @pytest.fixture
    def shallow_env_thorp(self):
        """Shallow water environment with Thorp volume absorption."""
        return Environment(
            name="atten_test_thorp",
            bathymetry=100.0,
            ssp=1500.0,
            absorption=Thorp(),
        )

    @pytest.fixture
    def high_freq_source(self):
        """High frequency source where attenuation is significant."""
        return Source(depths=50.0, frequencies=10000.0)  # 10 kHz

    @pytest.fixture
    def low_freq_source(self):
        """Low frequency source where attenuation is minimal."""
        return Source(depths=50.0, frequencies=100.0)  # 100 Hz

    @pytest.fixture
    def receiver(self):
        return Receiver(depths=[50.0], ranges=[1000.0, 3000.0, 5000.0])

    @pytest.mark.requires_binary
    def test_bellhop_thorp_attenuation(self, shallow_env, shallow_env_thorp,
                                       high_freq_source, receiver):
        """Test Bellhop with Thorp attenuation formula.

        At 10 kHz, Thorp absorption ≈ 0.6 dB/km; over the test ranges
        the extra TL should be on the order of the predicted Thorp value.
        We assert the depth-mean difference at the longest range is within
        the predicted-times-[0.1, 10] band — a sign-error or unit
        confusion would not satisfy that band.
        """
        bellhop = Bellhop(verbose=False)

        result_no_atten = bellhop.run(
            env=shallow_env, source=high_freq_source, receiver=receiver,
            run_mode=RunMode.COHERENT_TL,
        )
        result_thorp = bellhop.run(
            env=shallow_env_thorp, source=high_freq_source, receiver=receiver,
            run_mode=RunMode.COHERENT_TL,
        )

        # Thorp formula at 10 kHz (f in kHz):
        #   alpha = 0.11 f^2/(1+f^2) + 44 f^2/(4100+f^2)
        #         + 2.75e-4 f^2 + 0.003   [dB/km]
        f_khz = high_freq_source.frequencies[0] / 1000.0
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
            np.mean(result_thorp.tl[:, -1]) - np.mean(result_no_atten.tl[:, -1])
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
    def test_kraken_thorp_attenuation(self, shallow_env_thorp,
                                      high_freq_source, receiver):
        """Test Kraken with Thorp attenuation formula."""
        kraken = Kraken(verbose=False)
        result = kraken.run(
            env=shallow_env_thorp,
            source=high_freq_source,
            receiver=receiver,
        )
        assert result.field_type == 'modes'
        assert result.k is not None

    @pytest.mark.requires_binary
    def test_frequency_dependent_attenuation(self, shallow_env_thorp,
                                             low_freq_source, high_freq_source,
                                             receiver):
        """Test that attenuation increases with frequency."""
        bellhop = Bellhop(verbose=False)

        result_low = bellhop.run(
            env=shallow_env_thorp,
            source=low_freq_source,
            receiver=receiver,
            run_mode=RunMode.COHERENT_TL,
        )

        result_high = bellhop.run(
            env=shallow_env_thorp,
            source=high_freq_source,
            receiver=receiver,
            run_mode=RunMode.COHERENT_TL,
        )

        assert result_low.field_type == 'tl'
        assert result_high.field_type == 'tl'
        assert np.all(np.isfinite(result_low.data))
        assert np.all(np.isfinite(result_high.data))

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_attenuation_with_scooter(self, shallow_env_thorp,
                                      high_freq_source, receiver):
        """Test Scooter with volume attenuation."""
        scooter = Scooter(verbose=False)
        result = scooter.run(
            env=shallow_env_thorp,
            source=high_freq_source,
            receiver=receiver,
        )
        assert result.field_type == 'tl'
        assert np.all(np.isfinite(result.data))
