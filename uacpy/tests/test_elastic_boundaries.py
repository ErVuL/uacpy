"""
Tests for Elastic Boundary Handling

Tests both workflows for elastic boundaries:
1. KrakenField Auto-Detection (→ KrakenC)
2. BOUNCE → Reflection Files → BELLHOP/SCOOTER/KRAKEN

According to Acoustics Toolbox documentation:
- BOUNCE generates .brc (bottom reflection coef) and .irc (internal reflection coef)
- BELLHOP, SCOOTER, KRAKENC: Use .brc files
- KRAKEN: Uses .irc files (NOT .brc)
- SPARC: Does not support reflection files
"""

import pytest
import numpy as np
from pathlib import Path

from uacpy.core import Environment, Source, Receiver, BoundaryProperties
from uacpy.models import KrakenField, Bounce, Scooter

# Tests in this module spawn KrakenField/Bounce/Scooter/Bellhop/Kraken/KrakenC binaries
pytestmark = pytest.mark.requires_binary


class TestElasticBoundaryAutoDetection:
    """Test automatic elastic boundary detection in KrakenField."""

    @pytest.fixture
    def elastic_env(self):
        """Create environment with elastic bottom (shear speed > 0)."""
        bottom = BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600.0,
            shear_speed=400.0,  # THIS makes it elastic!
            density=1.8,
            attenuation=0.2,
            shear_attenuation=0.5
        )
        return Environment(
            name="Elastic Test",
            depth=100.0,
            sound_speed=1500.0,
            ssp_type='isovelocity',
            bottom=bottom
        )

    @pytest.fixture
    def fluid_env(self):
        """Create environment with fluid bottom (no shear)."""
        bottom = BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600.0,
            shear_speed=0.0,  # Fluid bottom
            density=1.5,
            attenuation=0.5
        )
        return Environment(
            name="Fluid Test",
            depth=100.0,
            sound_speed=1500.0,
            ssp_type='isovelocity',
            bottom=bottom
        )

    @pytest.fixture
    def source(self):
        return Source(depth=50.0, frequency=100.0)

    @pytest.fixture
    def receiver_small(self):
        return Receiver(
            depths=np.linspace(10, 90, 10),
            ranges=np.linspace(1000, 5000, 10)
        )

    def test_krakenfield_detects_elastic_bottom(self, elastic_env, source, receiver_small):
        """Test that KrakenField detects elastic bottom and uses KrakenC."""
        krakenfield = KrakenField(verbose=True)

        # This should automatically detect elastic boundary and use KrakenC
        result = krakenfield.compute_tl(elastic_env, source, receiver_small)

        assert result is not None
        assert result.data.shape == (len(receiver_small.depths), len(receiver_small.ranges))
        assert np.all(np.isfinite(result.data))
        assert result.field_type == 'tl'

        # TL should be reasonable (not all zeros, not all inf)
        assert np.any(result.data > 0)
        assert np.all(result.data < 200)  # Reasonable TL range

    def test_krakenfield_fluid_bottom(self, fluid_env, source, receiver_small):
        """Test that KrakenField works with fluid bottom (uses regular Kraken)."""
        krakenfield = KrakenField(verbose=False)

        result = krakenfield.compute_tl(fluid_env, source, receiver_small)

        assert result is not None
        assert result.data.shape == (len(receiver_small.depths), len(receiver_small.ranges))
        assert np.all(np.isfinite(result.data))

    def test_elastic_vs_fluid_difference(self, elastic_env, fluid_env, source, receiver_small):
        """Test that elastic and fluid bottoms produce different results."""
        krakenfield = KrakenField(verbose=False)

        result_elastic = krakenfield.compute_tl(elastic_env, source, receiver_small)
        result_fluid = krakenfield.compute_tl(fluid_env, source, receiver_small)

        # Should have different TL values
        diff = np.abs(result_elastic.data - result_fluid.data)
        mean_diff = np.nanmean(diff)

        # Elastic bottom should have some different loss characteristics
        assert mean_diff > 0.5, "Elastic and fluid bottoms should produce different results"


class TestBounceReflectionCoefficients:
    """Test BOUNCE reflection coefficient computation."""

    @pytest.fixture
    def elastic_env(self):
        """Create environment with elastic bottom for BOUNCE."""
        bottom = BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600.0,
            shear_speed=400.0,
            density=1.8,
            attenuation=0.2,
            shear_attenuation=0.5
        )
        return Environment(
            name="BOUNCE Test",
            depth=100.0,
            sound_speed=1500.0,
            ssp_type='isovelocity',
            bottom=bottom
        )

    @pytest.fixture
    def source(self):
        return Source(depth=50.0, frequency=100.0)

    @pytest.fixture
    def receiver_bounce(self):
        # BOUNCE doesn't need spatial receivers, just placeholder
        return Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))

    def test_bounce_basic(self, elastic_env, source, receiver_bounce, tmp_path):
        """Test basic BOUNCE execution."""
        bounce = Bounce(verbose=False, cmin=1400.0, cmax=10000.0, rmax_km=10.0)

        result = bounce.run(
            env=elastic_env,
            source=source,
            receiver=receiver_bounce,
            output_dir=tmp_path,
        )

        assert result is not None
        assert 'brc_file' in result.metadata
        assert Path(result.metadata['brc_file']).exists()

    def test_bounce_output_files(self, elastic_env, source, receiver_bounce, tmp_path):
        """Test that BOUNCE creates both .brc and .irc files."""
        bounce = Bounce(verbose=False, cmin=1400.0, cmax=10000.0, rmax_km=10.0)

        result = bounce.run(
            env=elastic_env,
            source=source,
            receiver=receiver_bounce,
            output_dir=tmp_path,
        )

        # Check .brc file exists
        assert 'brc_file' in result.metadata
        brc_file = Path(result.metadata['brc_file'])
        assert brc_file.exists()
        assert brc_file.suffix == '.brc'

        # Check .irc file exists
        assert 'irc_file' in result.metadata
        irc_file = Path(result.metadata['irc_file'])
        assert irc_file.exists()
        assert irc_file.suffix == '.irc'

    def test_bounce_reflection_coefficient_data(self, elastic_env, source, receiver_bounce):
        """Test that BOUNCE returns valid reflection coefficient data."""
        bounce = Bounce(verbose=False, cmin=1400.0, cmax=10000.0, rmax_km=10.0)

        result = bounce.run(
            env=elastic_env,
            source=source,
            receiver=receiver_bounce,
        )

        # Check metadata contains reflection coefficient data
        assert 'theta' in result.metadata  # Angles
        assert 'R' in result.metadata      # Magnitudes
        assert 'phi' in result.metadata    # Phases

        angles = result.metadata['theta']
        R_mag = result.metadata['R']
        phases = result.metadata['phi']

        assert len(angles) > 0
        assert len(R_mag) == len(angles)
        assert len(phases) == len(angles)

        # Check magnitudes are in valid range [0, 1]
        assert np.all(R_mag >= 0)
        assert np.all(R_mag <= 1.0)

        # Check angles are in valid range [0, 90]
        assert np.all(angles >= 0)
        assert np.all(angles <= 90)


class TestBounceToScooterWorkflow:
    """Test BOUNCE → SCOOTER workflow using .brc files."""

    @pytest.fixture
    def elastic_env(self):
        bottom = BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600.0,
            shear_speed=400.0,
            density=1.8,
            attenuation=0.2,
            shear_attenuation=0.5
        )
        return Environment(
            name="Workflow Test",
            depth=100.0,
            sound_speed=1500.0,
            ssp_type='isovelocity',
            bottom=bottom
        )

    @pytest.fixture
    def source(self):
        return Source(depth=50.0, frequency=100.0)

    @pytest.fixture
    def receiver_small(self):
        return Receiver(
            depths=np.linspace(10, 90, 10),
            ranges=np.linspace(1000, 5000, 10)
        )

    @pytest.fixture
    def receiver_bounce(self):
        return Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))

    def test_bounce_to_scooter(self, elastic_env, source, receiver_small, receiver_bounce, tmp_path):
        """Test complete BOUNCE → SCOOTER workflow."""
        # Step 1: Run BOUNCE to generate .brc file
        bounce = Bounce(verbose=False, cmin=1400.0, cmax=10000.0, rmax_km=10.0)
        bounce_result = bounce.run(
            env=elastic_env,
            source=source,
            receiver=receiver_bounce,
            output_dir=tmp_path,
        )

        brc_file = bounce_result.metadata['brc_file']
        assert Path(brc_file).exists()

        # Step 2: Create environment with .brc file
        bottom_with_file = BoundaryProperties(
            acoustic_type='file',
            reflection_file=brc_file,
            depth=100,
            sound_speed=1600.0,
            density=1.8,
            attenuation=0.2,
            reflection_cmin=1400.0,
            reflection_cmax=10000.0,
            reflection_rmax_km=10.0
        )

        env_with_rc = Environment(
            name="SCOOTER with BRC",
            depth=100.0,
            sound_speed=1500.0,
            ssp_type='isovelocity',
            bottom=bottom_with_file
        )

        # Step 3: Run SCOOTER with .brc file
        scooter = Scooter(verbose=False)
        result = scooter.compute_tl(env_with_rc, source, receiver_small)

        assert result is not None
        assert result.data.shape == (len(receiver_small.depths), len(receiver_small.ranges))
        assert np.all(np.isfinite(result.data))
        assert np.any(result.data > 0)

    def test_bounce_scooter_vs_direct_elastic(self, elastic_env, source, receiver_small, receiver_bounce, tmp_path):
        """Test that BOUNCE→SCOOTER gives similar results to direct elastic."""
        # Workflow 1: BOUNCE → SCOOTER
        bounce = Bounce(verbose=False, cmin=1400.0, cmax=10000.0, rmax_km=10.0)
        bounce_result = bounce.run(
            env=elastic_env,
            source=source,
            receiver=receiver_bounce,
            output_dir=tmp_path,
        )

        bottom_with_file = BoundaryProperties(
            acoustic_type='file',
            reflection_file=bounce_result.metadata['brc_file'],
            depth=100,
            sound_speed=1600.0,
            density=1.8,
            attenuation=0.2,
            reflection_cmin=1400.0,
            reflection_cmax=10000.0,
            reflection_rmax_km=10.0
        )

        env_with_rc = Environment(
            name="SCOOTER with BRC",
            depth=100.0,
            sound_speed=1500.0,
            ssp_type='isovelocity',
            bottom=bottom_with_file
        )

        scooter = Scooter(verbose=False)
        result_with_file = scooter.compute_tl(env_with_rc, source, receiver_small)

        # Workflow 2: Direct elastic
        result_direct = scooter.compute_tl(elastic_env, source, receiver_small)

        # Compare results
        diff = np.abs(result_with_file.data - result_direct.data)
        mean_diff = np.nanmean(diff)
        max_diff = np.nanmax(diff)

        # Should be similar (within 10 dB tolerance, ideally much closer)
        assert mean_diff < 10.0, f"Mean difference {mean_diff:.2f} dB is too large"
        assert max_diff < 50.0, f"Max difference {max_diff:.2f} dB is too large"


class TestWorkflowComparison:
    """Compare KrakenField auto-detection vs BOUNCE→SCOOTER workflows."""

    @pytest.fixture
    def elastic_env(self):
        bottom = BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600.0,
            shear_speed=400.0,
            density=1.8,
            attenuation=0.2,
            shear_attenuation=0.5
        )
        return Environment(
            name="Comparison Test",
            depth=100.0,
            sound_speed=1500.0,
            ssp_type='isovelocity',
            bottom=bottom
        )

    @pytest.fixture
    def source(self):
        return Source(depth=50.0, frequency=100.0)

    @pytest.fixture
    def receiver_small(self):
        return Receiver(
            depths=np.linspace(10, 90, 10),
            ranges=np.linspace(1000, 5000, 10)
        )

    def test_krakenfield_vs_bounce_scooter(self, elastic_env, source, receiver_small, tmp_path):
        """Compare results from both elastic boundary workflows."""
        # Approach 1: KrakenField auto-detection
        krakenfield = KrakenField(verbose=False)
        result_krakenfield = krakenfield.compute_tl(elastic_env, source, receiver_small)

        # Approach 2: BOUNCE → SCOOTER
        bounce = Bounce(verbose=False, cmin=1400.0, cmax=10000.0, rmax_km=10.0)
        receiver_bounce = Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))

        bounce_result = bounce.run(
            env=elastic_env,
            source=source,
            receiver=receiver_bounce,
            output_dir=tmp_path,
        )

        bottom_with_file = BoundaryProperties(
            acoustic_type='file',
            reflection_file=bounce_result.metadata['brc_file'],
            depth=100,
            sound_speed=1600.0,
            density=1.8,
            attenuation=0.2,
            reflection_cmin=1400.0,
            reflection_cmax=10000.0,
            reflection_rmax_km=10.0
        )

        env_with_rc = Environment(
            name="SCOOTER with BRC",
            depth=100.0,
            sound_speed=1500.0,
            ssp_type='isovelocity',
            bottom=bottom_with_file
        )

        scooter = Scooter(verbose=False)
        result_scooter = scooter.compute_tl(env_with_rc, source, receiver_small)

        # Both should produce valid results
        assert result_krakenfield is not None
        assert result_scooter is not None
        assert np.all(np.isfinite(result_krakenfield.data))
        assert np.all(np.isfinite(result_scooter.data))

        # Compare - different numerical methods, so allow some difference
        diff = np.abs(result_krakenfield.data - result_scooter.data)
        mean_diff = np.nanmean(diff)

        # Both workflows should produce reasonable TL fields
        # Allow larger tolerance since methods are fundamentally different
        assert mean_diff < 15.0, f"Mean difference {mean_diff:.2f} dB between workflows"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
