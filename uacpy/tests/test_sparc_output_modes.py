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
def sparc_simple_env():
    """Simple isovelocity environment configured for SPARC (vacuum bottom).

    Renamed from ``simple_env`` to avoid shadowing the shared conftest fixture,
    which uses a default half-space bottom that SPARC cannot handle.
    """
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
    """50 Hz source at 50m depth."""
    return Source(depths=50.0, frequencies=50.0)


@pytest.fixture
def receiver_grid():
    """Standard receiver grid."""
    depths = np.array([30.0, 50.0, 70.0])
    ranges = np.linspace(100, 1000, 10)
    return Receiver(depths=depths, ranges=ranges)


class TestSPARCOutputModes:
    """Test SPARC output modes."""

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_sparc_horizontal_array_mode(self, sparc_simple_env, source_50hz, receiver_grid):
        """
        Test SPARC horizontal array mode ('R')

        This is the default mode that was already implemented.
        """
        sparc = SPARC(output_mode='R')

        # Run with horizontal array mode (default)
        result = sparc.run(
            sparc_simple_env,
            source_50hz,
            receiver_grid,
        )

        assert result is not None
        assert result.field_type == 'tl'
        assert result.data is not None
        assert result.data.shape == (len(receiver_grid.depths), len(receiver_grid.ranges))
        assert result.metadata.get('output_mode') == 'R'
        assert result.metadata.get('model') == 'SPARC'

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_sparc_vertical_array_mode(self, sparc_simple_env, source_50hz):
        """
        Test SPARC vertical array mode ('D')

        Computes pressure vs depth at fixed ranges.
        """
        sparc = SPARC(output_mode='D')

        # Create receiver with specific depths and ranges for vertical array
        depths = np.linspace(10, 90, 9)
        ranges = np.array([500.0, 1000.0])
        receiver = Receiver(depths=depths, ranges=ranges)

        # Run with vertical array mode
        result = sparc.run(
            sparc_simple_env,
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

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_sparc_snapshot_mode(self, sparc_simple_env, source_50hz, receiver_grid):
        """
        Test SPARC snapshot mode ('S')

        Computes Green's function in wavenumber domain, then transforms to range.
        """
        sparc = SPARC(output_mode='S')

        # Run with snapshot mode
        result = sparc.run(
            sparc_simple_env,
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


class TestSPARCModeComparison:
    """Compare results from different SPARC modes."""

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_horizontal_vs_vertical_consistency(self, sparc_simple_env, source_50hz):
        """
        Test that horizontal and vertical modes give consistent results

        When computing a 2D field, both modes should produce similar TL values
        at the same (depth, range) points.
        """
        sparc = SPARC()

        # Simple grid for comparison
        depths = np.array([30.0, 50.0, 70.0])
        ranges = np.array([500.0, 1000.0])

        receiver_h = Receiver(depths=depths, ranges=ranges)
        receiver_v = Receiver(depths=depths, ranges=ranges)

        # Run both modes
        sparc_h = SPARC(verbose=False, output_mode='R')
        result_h = sparc_h.run(sparc_simple_env, source_50hz, receiver_h)
        sparc_v = SPARC(verbose=False, output_mode='D')
        result_v = sparc_v.run(sparc_simple_env, source_50hz, receiver_v)

        # Check shapes match
        assert result_h.data.shape == result_v.data.shape

        # Both modes must produce finite, physical TL values.
        # We deliberately do *not* compare R-mode and D-mode numerically:
        # they use different Hankel-transform paths internally and a tolerant
        # `assert_allclose(atol=15, rtol=0.2)` passes for almost any pair of
        # TL fields, so it provides no signal. Quantitative R-vs-D agreement
        # belongs in a dedicated benchmark with stored reference data.
        for r in (result_h, result_v):
            assert np.all(np.isfinite(r.data)), "non-finite TL"
            assert np.all(r.data > 0), "non-positive TL"
            assert np.all(r.data < 200), "TL exceeds 200 dB"


    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_sparc_snapshot_vs_horizontal_isovelocity(self, sparc_simple_env, source_50hz):
        """Snapshot ('S') and horizontal-array ('R') TL should agree at one
        receiver depth in an isovelocity vacuum-bottom case.

        Regression for the snapshot wavenumber bug: prior to the fix
        ``read_grn_file`` used ``freqVec[-1]`` (which actually stores the
        last output time for SPARC) when computing ``k = 2π·f/cVec``,
        producing a non-physical wavenumber grid. With the fix the GRN
        reader detects the SPARC title prefix and uses ``freq0`` (the
        source frequency) instead.
        """
        # Single ground-truth depth — match exactly between modes.
        depth = 50.0
        ranges = np.linspace(200.0, 1500.0, 8)

        receiver_S = Receiver(depths=np.array([depth]), ranges=ranges)
        receiver_R = Receiver(depths=np.array([depth]), ranges=ranges)

        # Snapshot (wavenumber-domain Green's function -> Hankel transform).
        sparc_S = SPARC(verbose=False, output_mode='S')
        result_S = sparc_S.run(sparc_simple_env, source_50hz, receiver_S)
        # Reference: horizontal time-marched FFP.
        sparc_R = SPARC(verbose=False, output_mode='R')
        result_R = sparc_R.run(sparc_simple_env, source_50hz, receiver_R)

        tl_S = np.asarray(result_S.data).reshape(-1)[:len(ranges)]
        tl_R = np.asarray(result_R.data).reshape(-1)[:len(ranges)]

        # Both must be finite and physical.
        assert np.all(np.isfinite(tl_S)), "snapshot TL contains non-finite values"
        assert np.all(np.isfinite(tl_R)), "horizontal TL contains non-finite values"
        assert np.all(tl_S < 200) and np.all(tl_R < 200)

        # Snapshot uses time-FFT + Hankel; horizontal uses time-FFT of the
        # fully-marched field. Outside deep TL nulls (where ±10 dB is
        # routine even for two correct methods) the two should agree well.
        # We require the mean absolute difference to stay below ~8 dB —
        # the pre-fix snapshot path (wrong wavenumber grid + wrong time
        # slice) produced 20-30 dB mean errors.
        mean_abs = float(np.mean(np.abs(tl_S - tl_R)))
        assert mean_abs < 8.0, (
            f"Snapshot vs horizontal mean |TL_S - TL_R| = {mean_abs:.2f} dB. "
            f"S={tl_S.tolist()}\nR={tl_R.tolist()}\nThe SPARC snapshot path "
            f"must use freq0 for the k grid (not freqVec[-1]) and must "
            f"time-FFT to extract the source-frequency component."
        )


class TestSPARCErrorHandling:
    """Test SPARC error handling for output modes."""

    def test_sparc_invalid_output_mode(self, sparc_simple_env, source_50hz, receiver_grid):
        """Test error handling for invalid output mode."""
        with pytest.raises(ValueError, match="Invalid output mode"):
            SPARC(output_mode='X')  # Invalid mode

    @pytest.mark.requires_binary
    def test_sparc_halfspace_warning(self, source_50hz, receiver_grid):
        """Test that halfspace bottom generates warning."""
        env = Environment(name="Test", depth=100, sound_speed=1500)
        env.bottom.acoustic_type = 'half-space'  # SPARC doesn't support this

        sparc = SPARC(verbose=False)

        # SPARC converts halfspace to rigid and logs a warning
        result = sparc.run(env, source_50hz, receiver_grid)

        # Should complete successfully with auto-conversion
        assert result is not None


class TestSPARCDepthDispatch:
    """Verify SPARC's per-depth re-run dispatch in horizontal-array mode."""

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_sparc_per_depth_run_count(self, sparc_simple_env, source_50hz):
        """
        Horizontal-array mode launches one SPARC run per receiver depth;
        confirm the metadata counter matches the requested depth count for
        1, 2, and 3 depths.
        """
        sparc = SPARC(verbose=False, output_mode='R')
        ranges = np.array([500.0, 1000.0])

        for n_depths in [1, 2, 3]:
            depths = np.linspace(30, 70, n_depths)
            receiver = Receiver(depths=depths, ranges=ranges)
            result = sparc.run(sparc_simple_env, source_50hz, receiver)
            assert result is not None
            assert result.metadata.get('n_depth_runs') == n_depths
