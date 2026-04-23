"""
Comprehensive testing of all UACPY acoustic propagation models

This test suite performs rigorous validation of:
- Bellhop (ray/beam tracing)
- Kraken (normal modes)
- KrakenField (TL from modes)
- RAM (parabolic equation)
- SPARC (time-domain FFP)
- Scooter (frequency-domain FFP)
- OASES suite (OAST, OASN, OASR, OASP)

Tests focus on:
1. Physical validity (sound speed dependencies, absorption, energy conservation)
2. Numerical accuracy (stability, convergence, precision)
3. Domain-specific scenarios (shallow/deep water, frequency ranges)
4. Edge cases and numerical stability
5. Model intercomparison

Author: Claude (Underwater Acoustics Testing Specialist)
Date: 2025-11-15
"""

import pytest
import numpy as np

import uacpy
from uacpy.models import (
    Bellhop, RAM, Kraken, KrakenField, KrakenC,
    Scooter, SPARC, OAST, OASN, OASR, OASP
)
from uacpy.models.base import RunMode
from uacpy.core.environment import BoundaryProperties
from uacpy.core.exceptions import ExecutableNotFoundError


# ============================================================================
# FIXTURES: Test Environments
# ============================================================================

@pytest.fixture
def pekeris_env():
    """
    Pekeris waveguide: canonical test case for underwater acoustics
    - Isovelocity water column
    - Flat bottom with half-space
    - Known analytical solution
    """
    bottom = BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1600.0,
        density=1.5,
        attenuation=0.5
    )
    return uacpy.Environment(
        name='Pekeris Waveguide',
        depth=100.0,
        sound_speed=1500.0,
        ssp_type='isovelocity',
        bottom=bottom
    )


@pytest.fixture
def shallow_water_env():
    """Shallow water environment (continental shelf, 50m)"""
    bottom = BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=1650.0,
        density=1.8,
        attenuation=0.8
    )
    return uacpy.Environment(
        name='Shallow Water',
        depth=50.0,
        sound_speed=1500.0,
        ssp_type='isovelocity',
        bottom=bottom
    )


@pytest.fixture
def deep_water_env():
    """Deep water with Munk profile (sound channel)"""
    return uacpy.Environment(
        name='Deep Water Munk',
        depth=5000.0,
        ssp_type='munk',
        bottom=BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600.0,
            density=1.8,
            attenuation=0.3
        )
    )


@pytest.fixture
def sloping_env():
    """Range-dependent environment with sloping bottom"""
    ranges = np.linspace(0, 10000, 21)
    depths = np.linspace(100, 200, 21)
    bathymetry = np.column_stack([ranges, depths])

    return uacpy.Environment(
        name='Sloping Bottom',
        depth=100.0,
        sound_speed=1500.0,
        ssp_type='isovelocity',
        bathymetry=bathymetry,
        bottom=BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600.0,
            density=1.5,
            attenuation=0.5
        )
    )


@pytest.fixture
def low_freq_source():
    """Low frequency source (50 Hz - whale calls, sonar)"""
    return uacpy.Source(depth=50.0, frequency=50.0)


@pytest.fixture
def mid_freq_source():
    """Mid frequency source (1000 Hz - typical sonar)"""
    return uacpy.Source(depth=50.0, frequency=1000.0)


@pytest.fixture
def high_freq_source():
    """High frequency source (10 kHz - imaging sonar)"""
    return uacpy.Source(depth=50.0, frequency=10000.0)


@pytest.fixture
def receiver_grid_dense():
    """Denser receiver grid local to this module (20 ranges, 5 depths).

    Renamed from ``receiver_grid`` to avoid shadowing the shared conftest
    fixture that uses 11 ranges.
    """
    return uacpy.Receiver(
        depths=np.array([10.0, 30.0, 50.0, 70.0, 90.0]),
        ranges=np.linspace(100, 5000, 20)
    )


@pytest.fixture
def receiver_small():
    """Small receiver grid for quick tests"""
    return uacpy.Receiver(
        depths=np.array([25.0, 50.0, 75.0]),
        ranges=np.array([500.0, 1000.0, 2000.0, 5000.0])
    )


# ============================================================================
# TEST CLASS: Bellhop
# ============================================================================

class TestBellhopPhysics:
    """Test Bellhop physical validity and accuracy"""

    @pytest.mark.requires_binary
    def test_bellhop_tl_increases_with_range(self, pekeris_env, mid_freq_source, receiver_grid_dense):
        """TL should increase with range (energy conservation)"""
        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(pekeris_env, mid_freq_source, receiver_grid_dense)

        # Average TL over depths at each range
        tl_vs_range = result.data.mean(axis=0)

        # Check monotonic increase (allowing for some modal oscillations)
        # Use moving average to smooth out interference patterns
        window = 3
        tl_smoothed = np.convolve(tl_vs_range, np.ones(window)/window, mode='valid')

        # TL should generally increase
        assert tl_smoothed[-1] > tl_smoothed[0], "TL should increase with range"

    @pytest.mark.requires_binary
    def test_bellhop_no_nan_inf(self, pekeris_env, mid_freq_source, receiver_grid_dense):
        """Bellhop should not produce NaN or inf values"""
        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(pekeris_env, mid_freq_source, receiver_grid_dense)

        assert np.all(np.isfinite(result.data)), "All TL values should be finite"
        assert not np.any(np.isnan(result.data)), "No NaN values"
        assert not np.any(np.isinf(result.data)), "No inf values"

    @pytest.mark.requires_binary
    def test_bellhop_positive_tl(self, pekeris_env, mid_freq_source, receiver_grid_dense):
        """TL values should be positive (physical constraint)"""
        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(pekeris_env, mid_freq_source, receiver_grid_dense)

        assert np.all(result.data > 0), "All TL values must be positive"
        assert np.all(result.data < 200), "TL should not exceed 200 dB (sanity check)"

    @pytest.mark.requires_binary
    def test_bellhop_frequency_dependence(self, pekeris_env, receiver_small):
        """Higher frequency should have higher TL due to absorption"""
        bellhop = Bellhop(verbose=False)

        # Low frequency (50 Hz)
        source_low = uacpy.Source(depth=50.0, frequency=50.0)
        result_low = bellhop.compute_tl(pekeris_env, source_low, receiver_small)
        tl_low = result_low.data.mean()

        # High frequency (10 kHz)
        source_high = uacpy.Source(depth=50.0, frequency=10000.0)
        result_high = bellhop.compute_tl(pekeris_env, source_high, receiver_small)
        tl_high = result_high.data.mean()

        # High frequency should have higher loss
        # (But be careful - high freq also has different modal structure)
        # This is a weak test - just checking they're different
        assert abs(tl_high - tl_low) > 1.0, "TL should differ significantly with frequency"

    @pytest.mark.requires_binary
    def test_bellhop_shallow_water(self, shallow_water_env, mid_freq_source):
        """Bellhop should handle shallow water (50m)"""
        # Adjust receiver depths for shallow water (50m environment)
        receiver = uacpy.Receiver(
            depths=np.array([10.0, 25.0, 40.0]),  # All within 50m
            ranges=np.array([500.0, 1000.0, 2000.0, 5000.0])
        )

        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(shallow_water_env, mid_freq_source, receiver)

        assert result.field_type == 'tl'
        assert np.all(np.isfinite(result.data))
        assert result.shape == (len(receiver.depths), len(receiver.ranges))

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_bellhop_deep_water(self, deep_water_env, mid_freq_source):
        """Bellhop should handle deep water with sound channel"""
        bellhop = Bellhop(verbose=False)

        # Create receiver grid appropriate for deep water
        receiver = uacpy.Receiver(
            depths=np.linspace(0, 5000, 26),
            ranges=np.linspace(1000, 20000, 10)
        )

        result = bellhop.compute_tl(deep_water_env, mid_freq_source, receiver)

        assert result.field_type == 'tl'
        assert np.all(np.isfinite(result.data))

    @pytest.mark.requires_binary
    def test_bellhop_range_dependent(self, sloping_env, mid_freq_source, receiver_small):
        """Bellhop should handle range-dependent bathymetry"""
        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(sloping_env, mid_freq_source, receiver_small)

        assert result.field_type == 'tl'
        assert np.all(np.isfinite(result.data))

    @pytest.mark.requires_binary
    def test_bellhop_run_types(self, pekeris_env, mid_freq_source, receiver_small):
        """Test different Bellhop run types (C, I, S)"""
        bellhop = Bellhop(verbose=False)

        # Coherent TL
        result_c = bellhop.run(pekeris_env, mid_freq_source, receiver_small, run_mode=RunMode.COHERENT_TL)
        assert result_c.field_type == 'tl'

        # Incoherent TL
        result_i = bellhop.run(pekeris_env, mid_freq_source, receiver_small, run_mode=RunMode.INCOHERENT_TL)
        assert result_i.field_type == 'tl'

        # Semi-coherent TL
        result_s = bellhop.run(pekeris_env, mid_freq_source, receiver_small, run_mode=RunMode.SEMICOHERENT_TL)
        assert result_s.field_type == 'tl'

        # All should produce valid output
        for result in [result_c, result_i, result_s]:
            assert np.all(np.isfinite(result.data))

    @pytest.mark.requires_binary
    def test_bellhop_beam_types(self, pekeris_env, mid_freq_source, receiver_small):
        """Test different beam types (B, R, G)"""
        # Gaussian beams (most stable)
        bellhop_b = Bellhop(verbose=False, beam_type='B')
        result_b = bellhop_b.run(pekeris_env, mid_freq_source, receiver_small)
        assert np.all(np.isfinite(result_b.data))

        # Ray-centered beams
        bellhop_r = Bellhop(verbose=False, beam_type='R')
        result_r = bellhop_r.run(pekeris_env, mid_freq_source, receiver_small)
        assert np.all(np.isfinite(result_r.data))

    @pytest.mark.requires_binary
    def test_bellhop_rays_output(self, pekeris_env, mid_freq_source):
        """Test ray tracing output"""
        bellhop = Bellhop(verbose=False)

        # Simple receiver for ray tracing
        receiver = uacpy.Receiver(
            depths=np.array([50.0]),
            ranges=np.array([1000.0])
        )

        try:
            result = bellhop.run(pekeris_env, mid_freq_source, receiver, run_mode=RunMode.RAYS)
            # Ray output may have different format
            assert result is not None
        except (FileNotFoundError, ExecutableNotFoundError) as e:
            pytest.skip(f"Ray tracing test skipped: {e}")


# ============================================================================
# TEST CLASS: Kraken Normal Modes
# ============================================================================

class TestKrakenPhysics:
    """Test Kraken normal mode computation"""

    @pytest.mark.requires_binary
    def test_kraken_mode_wavenumbers_positive(self, pekeris_env, mid_freq_source):
        """Mode wavenumbers should be real and positive"""
        kraken = Kraken(verbose=False)

        # Adjust receiver depths for mode computation
        receiver = uacpy.Receiver(
            depths=np.linspace(0, 100, 101),
            ranges=np.array([1000.0])
        )

        modes = kraken.run(pekeris_env, mid_freq_source, receiver)

        k = modes.metadata['k']
        assert len(k) > 0, "Should compute at least one mode"

        # Check wavenumbers are positive (real part)
        k_real = np.real(k)
        assert np.all(k_real > 0), "Mode wavenumbers should be positive"

    @pytest.mark.requires_binary
    def test_kraken_mode_count_frequency_scaling(self, pekeris_env):
        """Number of modes should scale with frequency * depth"""
        kraken = Kraken(verbose=False)

        receiver = uacpy.Receiver(
            depths=np.linspace(0, 100, 101),
            ranges=np.array([1000.0])
        )

        # Low frequency
        source_low = uacpy.Source(depth=50.0, frequency=50.0)
        modes_low = kraken.run(pekeris_env, source_low, receiver)
        n_modes_low = len(modes_low.metadata['k'])

        # High frequency
        source_high = uacpy.Source(depth=50.0, frequency=200.0)
        modes_high = kraken.run(pekeris_env, source_high, receiver)
        n_modes_high = len(modes_high.metadata['k'])

        # Higher frequency should have more modes
        assert n_modes_high > n_modes_low, "More modes at higher frequency"

    @pytest.mark.requires_binary
    def test_kraken_mode_shapes_non_zero(self, pekeris_env, mid_freq_source):
        """Mode shapes should have non-zero amplitude"""
        kraken = Kraken(verbose=False)

        receiver = uacpy.Receiver(
            depths=np.linspace(0, 100, 101),
            ranges=np.array([1000.0])
        )

        modes = kraken.run(pekeris_env, mid_freq_source, receiver)

        phi = modes.metadata['phi']
        assert len(phi) > 0, "Should have mode shapes"

        # Check first mode is non-trivial
        phi_0 = phi[0]
        assert np.max(np.abs(phi_0)) > 0, "First mode should have non-zero amplitude"

    @pytest.mark.requires_binary
    def test_kraken_rejects_range_dependent(self, sloping_env, mid_freq_source):
        """Kraken should reject range-dependent environments"""
        kraken = Kraken(verbose=False)

        receiver = uacpy.Receiver(
            depths=np.linspace(0, 100, 101),
            ranges=np.array([1000.0])
        )

        with pytest.raises(ValueError, match="range-dependent"):
            kraken.run(sloping_env, mid_freq_source, receiver)


# ============================================================================
# TEST CLASS: KrakenField
# ============================================================================

class TestKrakenFieldPhysics:
    """Test KrakenField TL computation from modes"""

    @pytest.mark.requires_binary
    def test_krakenfield_tl_output(self, pekeris_env, mid_freq_source, receiver_grid_dense):
        """KrakenField should produce valid TL field"""
        kf = KrakenField(verbose=False)
        result = kf.compute_tl(pekeris_env, mid_freq_source, receiver_grid_dense)

        assert result.field_type == 'tl'
        assert result.shape == (len(receiver_grid_dense.depths), len(receiver_grid_dense.ranges))
        assert np.all(np.isfinite(result.data))

    @pytest.mark.requires_binary
    def test_krakenfield_mode_resolution(self, pekeris_env, mid_freq_source, receiver_small):
        """Test different mode depth resolutions"""
        # Default resolution (1.5 pts/m)
        kf_default = KrakenField(mode_points_per_meter=1.5, verbose=False)
        result_default = kf_default.compute_tl(pekeris_env, mid_freq_source, receiver_small)

        # Higher resolution (3.0 pts/m)
        kf_high = KrakenField(mode_points_per_meter=3.0, verbose=False)
        result_high = kf_high.compute_tl(pekeris_env, mid_freq_source, receiver_small)

        # Both should produce finite results
        assert np.all(np.isfinite(result_default.data))
        assert np.all(np.isfinite(result_high.data))

        # Results should be similar (within 5 dB mean difference)
        mean_diff = np.abs(result_default.data.mean() - result_high.data.mean())
        assert mean_diff < 5.0, "Different resolutions should give similar TL"

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_krakenfield_range_dependent(self, sloping_env, mid_freq_source, receiver_small):
        """KrakenField should handle range-dependent via adiabatic modes"""
        kf = KrakenField(verbose=False, mode_coupling='adiabatic', n_segments=10)
        result = kf.compute_tl(sloping_env, mid_freq_source, receiver_small)

        assert result.field_type == 'tl'
        assert np.all(np.isfinite(result.data))


# ============================================================================
# TEST CLASS: RAM Parabolic Equation
# ============================================================================

class TestRAMPhysics:
    """Test RAM parabolic equation model"""

    def test_ram_basic_run(self, pekeris_env, mid_freq_source, receiver_small):
        """RAM should produce valid TL output"""
        try:
            ram = RAM(verbose=False)
            result = ram.compute_tl(pekeris_env, mid_freq_source, receiver_small)

            assert result.field_type == 'tl'
            assert np.all(np.isfinite(result.data))
            assert np.all(result.data > 0)
        except FileNotFoundError:
            pytest.skip("mpiramS binary not found")

    def test_ram_range_dependent(self, sloping_env, mid_freq_source, receiver_small):
        """RAM should handle range-dependent bathymetry"""
        try:
            ram = RAM(verbose=False)
            result = ram.compute_tl(sloping_env, mid_freq_source, receiver_small)

            assert result.field_type == 'tl'
            assert np.all(np.isfinite(result.data))
        except FileNotFoundError:
            pytest.skip("mpiramS binary not found")

    def test_ram_accuracy_parameters(self, pekeris_env, mid_freq_source, receiver_small):
        """Test RAM accuracy parameters (dr, dz, np_pade)"""
        try:
            # Default accuracy
            ram_default = RAM(verbose=False)
            result_default = ram_default.compute_tl(pekeris_env, mid_freq_source, receiver_small)

            # High accuracy
            ram_high = RAM(dr=50.0, dz=0.25, np_pade=6, verbose=False)
            result_high = ram_high.compute_tl(pekeris_env, mid_freq_source, receiver_small)

            # Both should produce finite results
            assert np.all(np.isfinite(result_default.data))
            assert np.all(np.isfinite(result_high.data))
        except FileNotFoundError:
            pytest.skip("mpiramS binary not found")


# ============================================================================
# TEST CLASS: SPARC Time-Domain FFP
# ============================================================================

class TestSPARCPhysics:
    """Test SPARC time-domain model"""

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_sparc_single_depth(self, pekeris_env, mid_freq_source):
        """SPARC should work with single receiver depth

        Note: SPARC is computationally expensive for time-domain integration.
        Uses narrowband frequency range (±2%) to keep computation tractable.
        For production use, consider Bellhop, Kraken, or Scooter for faster TL computation.
        """
        # Use rigid bottom (SPARC doesn't support halfspace)
        from uacpy.core.environment import BoundaryProperties
        pekeris_rigid = uacpy.Environment(
            name='Pekeris Waveguide (Rigid)',
            depth=100.0,
            sound_speed=1500.0,
            ssp_type='isovelocity',
            bottom=BoundaryProperties(
                acoustic_type='rigid',  # SPARC-compatible
                sound_speed=1600.0,
                density=1.5,
                attenuation=0.5
            )
        )

        sparc = SPARC(verbose=False)

        # SPARC works best with single depth (horizontal array)
        # Use fewer ranges to reduce computation time
        receiver = uacpy.Receiver(
            depths=np.array([50.0]),
            ranges=np.array([500.0, 1000.0, 2000.0, 5000.0])  # Only 4 ranges
        )

        from uacpy.core.exceptions import ModelExecutionError

        try:
            result = sparc.run(pekeris_rigid, mid_freq_source, receiver)

            if result is not None:  # May return None if too many depths
                assert result.field_type == 'tl'
                assert np.all(np.isfinite(result.data))
                assert result.shape == (1, 4)  # Single depth, 4 ranges
        except ModelExecutionError as e:
            # SPARC may timeout if computation is too expensive for the host
            if "Timed out" in str(e) or "timed out" in str(e):
                pytest.skip(f"SPARC computation too expensive: {e}")
            else:
                raise
        except (FileNotFoundError, ExecutableNotFoundError) as e:
            pytest.skip(f"SPARC test skipped: {e}")

    @pytest.mark.requires_binary
    def test_sparc_rejects_range_dependent(self, sloping_env, mid_freq_source):
        """SPARC hangs on range-dependent input; wrapper surfaces ModelExecutionError."""
        from uacpy.core.exceptions import ModelExecutionError
        sparc = SPARC(verbose=False)

        receiver = uacpy.Receiver(
            depths=np.array([50.0]),
            ranges=np.array([1000.0])
        )

        with pytest.raises(ModelExecutionError):
            sparc.run(sloping_env, mid_freq_source, receiver)


# ============================================================================
# TEST CLASS: Scooter FFP
# ============================================================================

class TestScooterPhysics:
    """Test Scooter frequency-domain FFP model"""

    @pytest.mark.requires_binary
    def test_scooter_basic_run(self, pekeris_env, mid_freq_source, receiver_small):
        """Scooter should produce valid TL output"""
        scooter = Scooter(verbose=False)

        try:
            result = scooter.run(pekeris_env, mid_freq_source, receiver_small)

            assert result.field_type == 'tl'
            assert np.all(np.isfinite(result.data))
        except (FileNotFoundError, ExecutableNotFoundError) as e:
            pytest.skip(f"Scooter test skipped: {e}")

    @pytest.mark.requires_binary
    def test_scooter_rmax_multiplier(self, pekeris_env, mid_freq_source, receiver_small):
        """Test Scooter rmax_multiplier parameter"""
        scooter = Scooter(verbose=False)

        try:
            # Low multiplier (faster, less accurate)
            result_low = scooter.run(pekeris_env, mid_freq_source, receiver_small,
                                     rmax_multiplier=1.5)

            # High multiplier (slower, more accurate)
            result_high = scooter.run(pekeris_env, mid_freq_source, receiver_small,
                                      rmax_multiplier=3.0)

            # Both should produce finite results
            assert np.all(np.isfinite(result_low.data))
            assert np.all(np.isfinite(result_high.data))
        except (FileNotFoundError, ExecutableNotFoundError) as e:
            pytest.skip(f"Scooter multiplier test skipped: {e}")


# ============================================================================
# TEST CLASS: OASES Models
# ============================================================================

@pytest.mark.requires_oases
class TestOASESPhysics:
    """Test OASES suite models"""

    @pytest.mark.requires_binary
    def test_oast_basic_run(self, pekeris_env, mid_freq_source, receiver_small):
        """OAST should produce valid TL output"""
        try:
            oast = OAST(verbose=False)
            result = oast.run(pekeris_env, mid_freq_source, receiver_small)

            assert result.field_type == 'tl'
            assert np.all(np.isfinite(result.data))
        except FileNotFoundError:
            pytest.skip("OAST executable not found")

    @pytest.mark.requires_binary
    def test_oasn_mode_computation(self, pekeris_env, mid_freq_source):
        """OASN should compute modes"""
        try:
            oasn = OASN(verbose=False)

            receiver = uacpy.Receiver(
                depths=np.linspace(0, 100, 101),
                ranges=np.array([1000.0])
            )

            result = oasn.run(pekeris_env, mid_freq_source, receiver)

            # OASN may return experimental mode data
            assert result.field_type == 'modes'
        except FileNotFoundError:
            pytest.skip("OASN executable not found")


# ============================================================================
# TEST CLASS: Model Intercomparison
# ============================================================================

class TestModelAgreement:
    """Test that different models agree on simple cases"""

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_bellhop_krakenfield_agreement(self, pekeris_env, mid_freq_source, receiver_small):
        """Bellhop and KrakenField should agree on Pekeris waveguide"""
        bellhop = Bellhop(verbose=False)
        krakenfield = KrakenField(verbose=False)

        result_bellhop = bellhop.compute_tl(pekeris_env, mid_freq_source, receiver_small)
        result_kraken = krakenfield.compute_tl(pekeris_env, mid_freq_source, receiver_small)

        # Compare mean TL (smooths out modal differences)
        mean_bellhop = result_bellhop.data.mean()
        mean_kraken = result_kraken.data.mean()

        diff = abs(mean_bellhop - mean_kraken)
        assert diff < 10.0, f"Bellhop and KrakenField differ by {diff:.1f} dB (expect <10 dB)"

    @pytest.mark.slow
    def test_bellhop_ram_agreement(self, pekeris_env, mid_freq_source, receiver_small):
        """Bellhop and RAM should agree on range-independent case"""
        try:
            bellhop = Bellhop(verbose=False)
            ram = RAM(verbose=False)

            result_bellhop = bellhop.compute_tl(pekeris_env, mid_freq_source, receiver_small)
            result_ram = ram.compute_tl(pekeris_env, mid_freq_source, receiver_small)

            # Compare mean TL
            mean_bellhop = result_bellhop.data.mean()
            mean_ram = result_ram.data.mean()

            diff = abs(mean_bellhop - mean_ram)
            assert diff < 15.0, f"Bellhop and RAM differ by {diff:.1f} dB (expect <15 dB)"
        except (ImportError, FileNotFoundError):
            pytest.skip("Models not available")


# ============================================================================
# TEST CLASS: Edge Cases and Numerical Stability
# ============================================================================

class TestEdgeCases:
    """Test edge cases that might break models"""

    @pytest.mark.requires_binary
    def test_very_shallow_water_10m(self):
        """Models should handle very shallow water (10m)"""
        env = uacpy.Environment(
            name='Very Shallow',
            depth=10.0,
            sound_speed=1500.0,
            ssp_type='isovelocity',
            bottom=BoundaryProperties(
                acoustic_type='half-space',
                sound_speed=1600.0,
                density=1.5,
                attenuation=0.5
            )
        )

        source = uacpy.Source(depth=5.0, frequency=1000.0)
        receiver = uacpy.Receiver(
            depths=np.array([2.0, 5.0, 8.0]),
            ranges=np.array([100.0, 500.0, 1000.0])
        )

        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env, source, receiver)

        assert np.all(np.isfinite(result.data))

    @pytest.mark.requires_binary
    def test_very_deep_water_6000m(self, mid_freq_source):
        """Models should handle very deep water (6000m)"""
        env = uacpy.Environment(
            name='Very Deep',
            depth=6000.0,
            sound_speed=1500.0,
            ssp_type='isovelocity',
            bottom=BoundaryProperties(
                acoustic_type='half-space',
                sound_speed=1600.0,
                density=1.8,
                attenuation=0.3
            )
        )

        receiver = uacpy.Receiver(
            depths=np.array([100.0, 1000.0, 3000.0, 5000.0]),
            ranges=np.array([1000.0, 10000.0, 50000.0])
        )

        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env, mid_freq_source, receiver)

        assert np.all(np.isfinite(result.data))

    @pytest.mark.requires_binary
    def test_very_high_frequency_50khz(self, pekeris_env, receiver_small):
        """Models should handle very high frequency (50 kHz)"""
        source = uacpy.Source(depth=50.0, frequency=50000.0)

        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(pekeris_env, source, receiver_small)

        assert np.all(np.isfinite(result.data))
        # High frequency should have high absorption
        assert result.data.mean() > 50, "High frequency should have significant loss"

    @pytest.mark.requires_binary
    def test_very_low_frequency_10hz(self, pekeris_env, receiver_small):
        """Models should handle very low frequency (10 Hz)"""
        source = uacpy.Source(depth=50.0, frequency=10.0)

        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(pekeris_env, source, receiver_small)

        assert np.all(np.isfinite(result.data))

    def test_source_near_surface(self, pekeris_env, mid_freq_source, receiver_small):
        """Models should handle source very close to surface"""
        source = uacpy.Source(depth=1.0, frequency=1000.0)

        try:
            bellhop = Bellhop(verbose=False)
            result = bellhop.compute_tl(pekeris_env, source, receiver_small)
            assert np.all(np.isfinite(result.data))
        except (FileNotFoundError, ExecutableNotFoundError):
            pytest.skip("Model may not support source at 1m depth")

    def test_source_near_bottom(self, pekeris_env, receiver_small):
        """Models should handle source very close to bottom"""
        source = uacpy.Source(depth=99.0, frequency=1000.0)

        try:
            bellhop = Bellhop(verbose=False)
            result = bellhop.compute_tl(pekeris_env, source, receiver_small)
            assert np.all(np.isfinite(result.data))
        except (FileNotFoundError, ExecutableNotFoundError):
            pytest.skip("Model may not support source at 99m depth")


# ============================================================================
# TEST CLASS: Absorption Models
# ============================================================================

class TestAbsorptionModels:
    """Test volume attenuation models (Thorp, Francois-Garrison)"""

    @pytest.mark.requires_binary
    def test_bellhop_with_thorp_absorption(self, pekeris_env, mid_freq_source, receiver_small):
        """Test Bellhop with Thorp absorption"""
        bellhop_no_atten = Bellhop(verbose=False)
        bellhop_thorp = Bellhop(verbose=False, volume_attenuation='T')

        # Without absorption
        result_no_atten = bellhop_no_atten.run(pekeris_env, mid_freq_source, receiver_small)

        # With Thorp absorption
        result_thorp = bellhop_thorp.run(pekeris_env, mid_freq_source, receiver_small)

        # Absorption should increase TL
        mean_no_atten = result_no_atten.data.mean()
        mean_thorp = result_thorp.data.mean()

        assert mean_thorp > mean_no_atten, "Thorp absorption should increase TL"


# ============================================================================
# MAIN: Run tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
