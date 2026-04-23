"""
Test that modal models (Kraken, Scooter, OAST) produce similar results.

Modal models should agree well for range-independent environments since they
use the same underlying physics (normal mode theory).
"""

import pytest
import numpy as np

import uacpy
from uacpy.models import Kraken, KrakenField, Scooter, OAST

# All tests in this module spawn model binaries (Kraken, Scooter, OAST)
pytestmark = pytest.mark.requires_binary


class TestModalModelAgreement:
    """Test agreement between different modal models"""

    @pytest.fixture
    def simple_environment(self):
        """Create a simple Pekeris waveguide for testing"""
        ssp_data = np.array([[0, 1500], [100, 1480]])
        env = uacpy.Environment(
            name='Pekeris',
            depth=100,
            ssp_type='linear',
            ssp_data=ssp_data,
            sound_speed=1500
        )
        return env

    @pytest.fixture
    def simple_source(self):
        """Create a simple source"""
        return uacpy.Source(depth=[50], frequency=[100])

    @pytest.fixture
    def single_receiver(self):
        """Create a single receiver at 50m depth, 5km range"""
        return uacpy.Receiver(
            depths=np.array([50]),
            ranges=np.array([5000])
        )

    @pytest.fixture
    def multi_range_receiver(self):
        """Create receivers at multiple ranges"""
        return uacpy.Receiver(
            depths=np.array([50]),
            ranges=np.array([1000, 2000, 3000, 5000, 7000, 10000])
        )

    def test_kraken_modes_valid(self, simple_environment, simple_source):
        """Test that Kraken computes valid modes (wavenumbers non-zero)"""
        # Create receiver grid for mode computation
        mode_depths = np.linspace(0, simple_environment.depth * 0.999, 150)
        receiver = uacpy.Receiver(depths=mode_depths, ranges=np.array([5000]))

        # Compute modes
        kraken = Kraken(verbose=False)
        result = kraken.run(simple_environment, simple_source, receiver)

        k = result.metadata['k']

        # Check that we have modes
        assert len(k) > 0, "No modes computed"

        # Check that Mode 1 is non-zero (this was the bug that was fixed)
        assert np.abs(k[0]) > 0.1, f"Mode 1 wavenumber is zero or near-zero: {k[0]}"

        # Count valid modes (non-zero with non-positive imaginary part)
        n_valid = sum(1 for k_val in k if np.abs(k_val) >= 1e-10 and np.imag(k_val) <= 0)

        # For 100 Hz in 100m water, expect approximately 6 modes
        assert n_valid >= 5, f"Expected at least 5 valid modes, got {n_valid}"
        assert n_valid <= 7, f"Expected at most 7 valid modes, got {n_valid}"

    def test_krakenfield_vs_scooter_single_point(
        self, simple_environment, simple_source, single_receiver
    ):
        """Test KrakenField vs Scooter at a single point"""
        # KrakenField
        kf = KrakenField(verbose=False)
        kf_result = kf.run(simple_environment, simple_source, single_receiver)
        kf_tl = kf_result.data[0, 0]

        # Scooter
        scooter = Scooter(verbose=False)
        scooter_result = scooter.run(simple_environment, simple_source, single_receiver)
        scooter_tl = scooter_result.data[0, 0]

        # Check agreement
        diff = np.abs(kf_tl - scooter_tl)

        assert diff < 5.0, (
            f"KrakenField and Scooter disagree by {diff:.2f} dB "
            f"(KF={kf_tl:.2f}, Scooter={scooter_tl:.2f}). "
            "Modal models should agree within 5 dB."
        )

    def test_krakenfield_vs_scooter_multiple_ranges(
        self, simple_environment, simple_source, multi_range_receiver
    ):
        """Test KrakenField vs Scooter across multiple ranges"""
        # KrakenField
        kf = KrakenField(verbose=False)
        kf_result = kf.run(simple_environment, simple_source, multi_range_receiver)
        kf_tl = kf_result.data[0, :]

        # Scooter
        scooter = Scooter(verbose=False)
        scooter_result = scooter.run(simple_environment, simple_source, multi_range_receiver)
        scooter_tl = scooter_result.data[0, :]

        # Check agreement at each range
        diffs = np.abs(kf_tl - scooter_tl)
        mean_diff = np.mean(diffs)
        max_diff = np.max(diffs)

        assert mean_diff < 2.0, (
            f"KrakenField and Scooter mean difference is {mean_diff:.2f} dB. "
            "Modal models should agree with mean difference < 2 dB."
        )

        assert max_diff < 5.0, (
            f"KrakenField and Scooter max difference is {max_diff:.2f} dB. "
            "Modal models should agree with max difference < 5 dB."
        )

    @pytest.mark.requires_oases
    def test_krakenfield_vs_oast_single_point(
        self, simple_environment, simple_source, single_receiver
    ):
        """Test KrakenField vs OAST at a single point"""
        # KrakenField
        kf = KrakenField(verbose=False)
        kf_result = kf.run(simple_environment, simple_source, single_receiver)
        kf_tl = kf_result.data[0, 0]

        # OAST
        oast = OAST(verbose=False)
        oast_result = oast.run(simple_environment, simple_source, single_receiver)
        oast_tl = oast_result.data[0, 0]

        # Check agreement
        # Note: OAST may have larger differences due to different numerical methods
        diff = np.abs(kf_tl - oast_tl)

        assert diff < 10.0, (
            f"KrakenField and OAST disagree by {diff:.2f} dB "
            f"(KF={kf_tl:.2f}, OAST={oast_tl:.2f}). "
            "Modal models should agree within 10 dB."
        )

    @pytest.mark.requires_oases
    def test_scooter_vs_oast_single_point(
        self, simple_environment, simple_source, single_receiver
    ):
        """Test Scooter vs OAST at a single point"""
        # Scooter
        scooter = Scooter(verbose=False)
        scooter_result = scooter.run(simple_environment, simple_source, single_receiver)
        scooter_tl = scooter_result.data[0, 0]

        # OAST
        oast = OAST(verbose=False)
        oast_result = oast.run(simple_environment, simple_source, single_receiver)
        oast_tl = oast_result.data[0, 0]

        # Check agreement
        diff = np.abs(scooter_tl - oast_tl)

        assert diff < 10.0, (
            f"Scooter and OAST disagree by {diff:.2f} dB "
            f"(Scooter={scooter_tl:.2f}, OAST={oast_tl:.2f}). "
            "Modal models should agree within 10 dB."
        )

    @pytest.mark.requires_oases
    def test_all_modal_models_agreement(
        self, simple_environment, simple_source, single_receiver
    ):
        """Test that all modal models (Kraken, Scooter, OAST) agree"""
        # Run all models
        kf = KrakenField(verbose=False)
        kf_result = kf.run(simple_environment, simple_source, single_receiver)
        kf_tl = kf_result.data[0, 0]

        scooter = Scooter(verbose=False)
        scooter_result = scooter.run(simple_environment, simple_source, single_receiver)
        scooter_tl = scooter_result.data[0, 0]

        oast = OAST(verbose=False)
        oast_result = oast.run(simple_environment, simple_source, single_receiver)
        oast_tl = oast_result.data[0, 0]

        # Compute pairwise differences
        kf_scooter_diff = np.abs(kf_tl - scooter_tl)
        kf_oast_diff = np.abs(kf_tl - oast_tl)
        scooter_oast_diff = np.abs(scooter_tl - oast_tl)

        # All models should be within reasonable range of each other
        max_diff = max(kf_scooter_diff, kf_oast_diff, scooter_oast_diff)

        assert max_diff < 15.0, (
            f"Modal models disagree significantly:\n"
            f"  KrakenField: {kf_tl:.2f} dB\n"
            f"  Scooter:     {scooter_tl:.2f} dB\n"
            f"  OAST:        {oast_tl:.2f} dB\n"
            f"  Max difference: {max_diff:.2f} dB\n"
            "All modal models should agree within 15 dB."
        )

    def test_mode_count_consistency(self, simple_environment, simple_source):
        """Test that mode count is consistent for the environment"""
        mode_depths = np.linspace(0, simple_environment.depth * 0.999, 150)
        receiver = uacpy.Receiver(depths=mode_depths, ranges=np.array([5000]))

        # Compute modes with different resolutions
        resolutions = [100, 150, 200]
        mode_counts = []

        for n_points in resolutions:
            mode_depths = np.linspace(0, simple_environment.depth * 0.999, n_points)
            receiver = uacpy.Receiver(depths=mode_depths, ranges=np.array([5000]))

            kraken = Kraken(verbose=False)
            result = kraken.run(simple_environment, simple_source, receiver)

            k = result.metadata['k']
            n_valid = sum(1 for k_val in k if np.abs(k_val) >= 1e-10 and np.imag(k_val) <= 0)
            mode_counts.append(n_valid)

        # All resolutions should give same number of valid modes
        assert len(set(mode_counts)) == 1, (
            f"Mode count varies with resolution: {mode_counts}. "
            "Should be consistent across resolutions."
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
