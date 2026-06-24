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

from uacpy.core import Environment, Receiver, BoundaryProperties
from uacpy.core.environment import SeabedColumn, SedimentLayer
from uacpy.core.exceptions import UnsupportedFeatureError
from uacpy import Field
from uacpy.models import Kraken, Bounce, Scooter

# Tests in this module spawn KrakenField/Bounce/Scooter/Bellhop/Kraken/KrakenC binaries
pytestmark = pytest.mark.requires_binary


@pytest.fixture
def elastic_env():
    """Module-shared Pekeris-style env with an elastic half-space bottom
    (cp=1600, cs=400, ρ=1.8, αp=0.2, αs=0.5). Distinct from conftest's
    ``elastic_env`` (cp=1700, αp=0.5, αs=0.8); the four test classes here
    were all written against this specific tune of the parameters."""
    bottom = BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1600.0,
        shear_speed=400.0,
        density=1.8,
        attenuation=0.2,
        shear_attenuation=0.5,
    )
    return Environment(
        name="Elastic Test",
        bathymetry=100.0,
        ssp=1500.0,
        bottom=bottom,
    )


class TestElasticOverFluidHalfspaceGuard:
    """`krakenc.exe` spins forever in setup on a solid-over-liquid bottom
    interface (an elastic ``SeabedColumn`` layer over a fluid halfspace).
    `_KrakenBase` rejects it up front so all three Kraken-family models fail
    fast with a typed error instead of hanging for ``timeout`` seconds."""

    @staticmethod
    def _src_rcv():
        from uacpy.core import Source
        return (Source(depths=25.0, frequencies=75.0),
                Receiver(depths=np.linspace(2, 48, 20),
                         ranges=np.linspace(100, 4000, 20)))

    @staticmethod
    def _elastic_over_fluid():
        return Environment(
            name='elastic-over-fluid', bathymetry=50.0, ssp=1500.0,
            bottom=SeabedColumn(
                layers=[SedimentLayer(thickness=10.0, sound_speed=1700.0,
                                      density=1.8, attenuation=0.3, shear_speed=300.0)],
                halfspace=BoundaryProperties(acoustic_type='half-space',
                                             sound_speed=2000.0, density=2.0,
                                             attenuation=0.4)),  # cs=0 ⇒ fluid
        )

    @pytest.mark.parametrize('model_cls', [Kraken])
    def test_rejects_fast_with_alternatives(self, model_cls):
        src, rcv = self._src_rcv()
        with pytest.raises(UnsupportedFeatureError) as exc:
            model_cls().run(self._elastic_over_fluid(), src, rcv)
        assert 'Scooter' in str(exc.value)

    @staticmethod
    def _elastic_over_elastic():
        return Environment(
            name='elastic-over-elastic', bathymetry=50.0, ssp=1500.0,
            bottom=SeabedColumn(
                layers=[SedimentLayer(thickness=10.0, sound_speed=1700.0,
                                      density=1.8, attenuation=0.3, shear_speed=300.0)],
                halfspace=BoundaryProperties(acoustic_type='half-space',
                                             sound_speed=2200.0, density=2.2,
                                             attenuation=0.4, shear_speed=600.0)))

    def test_elastic_over_elastic_is_allowed(self):
        """Guard is specific: an elastic layer over an *elastic* halfspace is
        not rejected (krakenc handles solid-solid)."""
        src, _ = self._src_rcv()
        modes = Kraken(backend='krakenc').compute_modes(self._elastic_over_elastic(), src)
        assert modes.k.shape[0] > 0

    def test_receiver_in_elastic_subbottom_returns_nan(self):
        """KrakenField: where the modes solve succeeds (elastic-over-elastic),
        receivers below the water column — which field.exe cannot evaluate in
        elastic media — are returned as NaN, the water-column receivers as
        finite values. No raise (the harmonized below-domain policy)."""
        src, _ = self._src_rcv()
        # 20 m water, 55 m in the elastic layer, 70 m in the elastic halfspace
        rcv = Receiver(depths=np.array([20.0, 55.0, 70.0]),
                       ranges=np.linspace(100, 3000, 20))
        with pytest.warns(UserWarning, match='sub-bottom'):
            field = Kraken().run(self._elastic_over_elastic(), src, rcv)
        tl = field.tl
        assert np.isfinite(tl[0]).any()          # water column: real values
        assert not np.isfinite(tl[1]).any()      # in elastic layer: all NaN
        assert not np.isfinite(tl[2]).any()      # in elastic halfspace: all NaN


class TestElasticBoundaryAutoDetection:
    """Test automatic elastic boundary detection in KrakenField."""

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
            bathymetry=100.0,
            ssp=1500.0,
            bottom=bottom
        )

    @pytest.fixture
    def receiver_small(self):
        return Receiver(
            depths=np.linspace(10, 90, 10),
            ranges=np.linspace(1000, 5000, 10)
        )

    def test_krakenfield_detects_elastic_bottom(self, elastic_env, source, receiver_small):
        """Test that KrakenField detects elastic bottom and uses KrakenC."""
        krakenfield = Kraken(verbose=False)

        # This should automatically detect elastic boundary and use KrakenC
        result = krakenfield.compute_tl(elastic_env, source, receiver_small)

        assert result is not None
        assert result.data.shape == (len(receiver_small.depths), len(receiver_small.ranges))
        assert np.all(np.isfinite(result.data))
        assert isinstance(result, Field)

        # TL should be reasonable (not all zeros, not all inf)
        assert np.any(result.tl > 0)
        assert np.all(result.tl < 200)  # Reasonable TL range

    def test_krakenfield_fluid_bottom(self, fluid_env, source, receiver_small):
        """Test that KrakenField works with fluid bottom (uses regular Kraken)."""
        krakenfield = Kraken(verbose=False)

        result = krakenfield.compute_tl(fluid_env, source, receiver_small)

        assert result is not None
        assert result.data.shape == (len(receiver_small.depths), len(receiver_small.ranges))
        assert np.all(np.isfinite(result.data))

    def test_elastic_vs_fluid_difference(self, elastic_env, fluid_env, source, receiver_small):
        """Test that elastic and fluid bottoms produce different results."""
        krakenfield = Kraken(verbose=False)

        result_elastic = krakenfield.compute_tl(elastic_env, source, receiver_small)
        result_fluid = krakenfield.compute_tl(fluid_env, source, receiver_small)

        # Should have different TL values (compare in dB; .tl works regardless
        # of underlying units storage)
        diff = np.abs(result_elastic.tl - result_fluid.tl)
        mean_diff = np.nanmean(diff)

        # Elastic bottom should have some different loss characteristics
        assert mean_diff > 0.5, "Elastic and fluid bottoms should produce different results"


class TestBounceReflectionCoefficients:
    """Test BOUNCE reflection coefficient computation."""

    @pytest.fixture
    def receiver_bounce(self):
        # BOUNCE doesn't need spatial receivers, just placeholder
        return Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))

    def test_bounce_basic(self, elastic_env, source, receiver_bounce, tmp_path):
        """Test basic BOUNCE execution."""
        bounce = Bounce(verbose=False, c_low=1400.0, c_high=10000.0, rmax=10000.0, work_dir=tmp_path)

        result = bounce.run(
            env=elastic_env,
            source=source,
            receiver=receiver_bounce,
        )

        assert result is not None
        assert 'brc_file' in result.metadata
        assert Path(result.metadata['brc_file']).exists()

    def test_bounce_output_files(self, elastic_env, source, receiver_bounce, tmp_path):
        """Test that BOUNCE creates both .brc and .irc files."""
        bounce = Bounce(verbose=False, c_low=1400.0, c_high=10000.0, rmax=10000.0, work_dir=tmp_path)

        result = bounce.run(
            env=elastic_env,
            source=source,
            receiver=receiver_bounce,
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

    def test_bounce_reflection_coefficient_data(
        self, elastic_env, source, receiver_bounce, tmp_path,
    ):
        """Test that BOUNCE returns valid reflection coefficient data."""
        bounce = Bounce(verbose=False, c_low=1400.0, c_high=10000.0, rmax=10000.0, work_dir=tmp_path)

        result = bounce.run(
            env=elastic_env,
            source=source,
            receiver=receiver_bounce,
        )

        angles = result.theta
        R_mag = result.R
        phases = result.phi

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
    def receiver_small(self):
        return Receiver(
            depths=np.linspace(10, 90, 10),
            ranges=np.linspace(1000, 5000, 10)
        )

    @pytest.fixture
    def receiver_bounce(self):
        return Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))

    def test_bounce_scooter_vs_direct_elastic(self, elastic_env, source, receiver_small, receiver_bounce, tmp_path):
        """Test that BOUNCE→SCOOTER gives similar results to direct elastic."""
        # Workflow 1: BOUNCE → SCOOTER
        bounce = Bounce(verbose=False, c_low=1400.0, c_high=10000.0, rmax=10000.0, work_dir=tmp_path)
        bounce_result = bounce.run(
            env=elastic_env,
            source=source,
            receiver=receiver_bounce,
        )

        bottom_with_file = BoundaryProperties(
            acoustic_type='file',
            reflection_file=bounce_result.metadata['brc_file'],
            sound_speed=1600.0,
            density=1.8,
            attenuation=0.2,
        )

        env_with_rc = Environment(
            name="SCOOTER with BRC",
            bathymetry=100.0,
            ssp=1500.0,
            bottom=bottom_with_file
        )

        scooter = Scooter(verbose=False, c_low=1400.0, c_high=10000.0)
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
    def receiver_small(self):
        return Receiver(
            depths=np.linspace(10, 90, 10),
            ranges=np.linspace(1000, 5000, 10)
        )

    def test_krakenfield_vs_bounce_scooter(self, elastic_env, source, receiver_small, tmp_path):
        """Compare results from both elastic boundary workflows."""
        # Approach 1: KrakenField auto-detection
        krakenfield = Kraken(verbose=False)
        result_krakenfield = krakenfield.compute_tl(elastic_env, source, receiver_small)

        # Approach 2: BOUNCE → SCOOTER
        bounce = Bounce(verbose=False, c_low=1400.0, c_high=10000.0, rmax=10000.0, work_dir=tmp_path)
        receiver_bounce = Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))

        bounce_result = bounce.run(
            env=elastic_env,
            source=source,
            receiver=receiver_bounce,
        )

        bottom_with_file = BoundaryProperties(
            acoustic_type='file',
            reflection_file=bounce_result.metadata['brc_file'],
            sound_speed=1600.0,
            density=1.8,
            attenuation=0.2,
        )

        env_with_rc = Environment(
            name="SCOOTER with BRC",
            bathymetry=100.0,
            ssp=1500.0,
            bottom=bottom_with_file
        )

        scooter = Scooter(verbose=False, c_low=1400.0, c_high=10000.0)
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
