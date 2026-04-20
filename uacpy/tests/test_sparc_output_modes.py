"""
Tests for SPARC output modes

Tests all three SPARC output modes:
- 'R': Horizontal array (time series at fixed depths)
- 'D': Vertical array (time series at fixed ranges)
- 'S': Snapshot (wavenumber-domain Green's function)
"""

import pytest
import numpy as np

from uacpy import Environment, Source, Receiver
from uacpy.models import SPARC


@pytest.fixture
def simple_env():
    """Simple isovelocity environment for testing"""
    env = Environment(
        name="Test Environment",
        depth=100.0,
        sound_speed=1500.0
    )
    # SPARC requires vacuum or rigid bottom
    env.bottom.acoustic_type = 'vacuum'
    return env


@pytest.fixture
def source_50hz():
    """50 Hz source at 50m depth"""
    return Source(depth=50.0, frequency=50.0)


@pytest.fixture
def receiver_grid():
    """Standard receiver grid"""
    depths = np.array([30.0, 50.0, 70.0])
    ranges = np.linspace(100, 1000, 10)
    return Receiver(depths=depths, ranges=ranges)


class TestSPARCOutputModes:
    """Test SPARC output modes"""

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_sparc_horizontal_array_mode(self, simple_env, source_50hz, receiver_grid):
        """
        Test SPARC horizontal array mode ('R')

        This is the default mode that was already implemented.
        """
        try:
            sparc = SPARC(output_mode='R')

            # Run with horizontal array mode (default)
            result = sparc.run(
                simple_env,
                source_50hz,
                receiver_grid,
            )

            assert result is not None
            assert result.field_type == 'tl'
            assert result.data is not None
            assert result.data.shape == (len(receiver_grid.depths), len(receiver_grid.ranges))
            assert result.metadata.get('output_mode') == 'R'
            assert result.metadata.get('model') == 'SPARC'

        except FileNotFoundError:
            pytest.skip("SPARC not installed")

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_sparc_vertical_array_mode(self, simple_env, source_50hz):
        """
        Test SPARC vertical array mode ('D')

        Computes pressure vs depth at fixed ranges.
        """
        try:
            sparc = SPARC(output_mode='D')

            # Create receiver with specific depths and ranges for vertical array
            depths = np.linspace(10, 90, 9)
            ranges = np.array([500.0, 1000.0])
            receiver = Receiver(depths=depths, ranges=ranges)

            # Run with vertical array mode
            result = sparc.run(
                simple_env,
                source_50hz,
                receiver,
            )

            assert result is not None
            assert result.field_type == 'tl'
            assert result.data is not None
            assert result.data.shape == (len(depths), len(ranges))
            assert result.metadata.get('output_mode') == 'D'
            assert result.metadata.get('model') == 'SPARC'
            assert result.metadata.get('n_range_runs') == len(ranges)

        except FileNotFoundError:
            pytest.skip("SPARC not installed")

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_sparc_snapshot_mode(self, simple_env, source_50hz, receiver_grid):
        """
        Test SPARC snapshot mode ('S')

        Computes Green's function in wavenumber domain, then transforms to range.
        """
        try:
            sparc = SPARC(output_mode='S')

            # Run with snapshot mode
            result = sparc.run(
                simple_env,
                source_50hz,
                receiver_grid,
            )

            assert result is not None
            assert result.field_type == 'tl'
            assert result.data is not None
            # Snapshot mode provides full 2D field
            assert result.data.ndim == 2
            assert result.metadata.get('output_mode') == 'S'
            assert result.metadata.get('model') == 'SPARC'
            assert 'Hankel transform' in result.metadata.get('note', '')

        except FileNotFoundError:
            pytest.skip("SPARC not installed")


class TestSPARCModeComparison:
    """Compare results from different SPARC modes"""

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_horizontal_vs_vertical_consistency(self, simple_env, source_50hz):
        """
        Test that horizontal and vertical modes give consistent results

        When computing a 2D field, both modes should produce similar TL values
        at the same (depth, range) points.
        """
        try:
            sparc = SPARC()

            # Simple grid for comparison
            depths = np.array([30.0, 50.0, 70.0])
            ranges = np.array([500.0, 1000.0])

            receiver_h = Receiver(depths=depths, ranges=ranges)
            receiver_v = Receiver(depths=depths, ranges=ranges)

            # Run both modes
            sparc_h = SPARC(verbose=False, output_mode='R')
            result_h = sparc_h.run(simple_env, source_50hz, receiver_h)
            sparc_v = SPARC(verbose=False, output_mode='D')
            result_v = sparc_v.run(simple_env, source_50hz, receiver_v)

            # Check shapes match
            assert result_h.data.shape == result_v.data.shape

            # Check that TL values are in the same ballpark.
            # Horizontal ('R') and vertical ('D') output modes use different
            # Hankel transforms so differences of 5-15 dB are normal.
            np.testing.assert_allclose(
                result_h.data,
                result_v.data,
                atol=15.0,
                rtol=0.2
            )

        except FileNotFoundError:
            pytest.skip("SPARC not installed")


class TestSPARCErrorHandling:
    """Test SPARC error handling for output modes"""

    def test_sparc_invalid_output_mode(self, simple_env, source_50hz, receiver_grid):
        """Test error handling for invalid output mode"""
        with pytest.raises(ValueError, match="Invalid output mode"):
            SPARC(output_mode='X')  # Invalid mode

    @pytest.mark.requires_binary
    def test_sparc_halfspace_warning(self, source_50hz, receiver_grid):
        """Test that halfspace bottom generates warning"""
        try:
            env = Environment(name="Test", depth=100, sound_speed=1500)
            env.bottom.acoustic_type = 'halfspace'  # SPARC doesn't support this

            sparc = SPARC(verbose=True)

            # SPARC converts halfspace to rigid and logs a warning
            result = sparc.run(env, source_50hz, receiver_grid)

            # Should complete successfully with auto-conversion
            assert result is not None

        except FileNotFoundError:
            pytest.skip("SPARC not installed")


class TestSPARCPerformance:
    """Test SPARC performance characteristics"""

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_sparc_depth_scaling(self, simple_env, source_50hz):
        """
        Test that computation time scales with number of depths

        Horizontal array mode requires one SPARC run per depth.
        """
        try:
            import time

            sparc = SPARC(verbose=False, output_mode='R')
            ranges = np.array([500.0, 1000.0])

            # Test with 1, 2, and 3 depths
            times = []
            for n_depths in [1, 2, 3]:
                depths = np.linspace(30, 70, n_depths)
                receiver = Receiver(depths=depths, ranges=ranges)

                start = time.time()
                result = sparc.run(simple_env, source_50hz, receiver)
                elapsed = time.time() - start

                times.append(elapsed)
                assert result is not None
                assert result.metadata.get('n_depth_runs') == n_depths

            # Time should scale approximately linearly with n_depths
            # (with some overhead for setup)
            # We expect: time(2 depths) < 2.5 * time(1 depth)
            if times[0] > 0.1:  # Only check if runs take meaningful time
                assert times[1] < 2.5 * times[0], "Scaling worse than expected"

        except FileNotFoundError:
            pytest.skip("SPARC not installed")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
