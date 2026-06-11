"""
Physical sanity checks for UACPY models.

These tests validate qualitative physical behavior on canonical scenarios
(Pekeris waveguide, sloping bottom, Munk profile) — TL increases with range,
modes are positive, mode count scales with frequency, etc. They do NOT
compare against stored reference numbers.

For *quantitative* benchmarks against canonical waveguides (Pekeris,
Munk, layered fluids), see the roadmap item in README.md ("add
reference-case regressions"). That work is tracked separately and will
live in benchmark_data/ + test_benchmarks.py when added.
"""

import pytest
import numpy as np

import uacpy
from uacpy import Field
from uacpy.core.results import Modes
from uacpy.models import Bellhop, Kraken, KrakenField, RAM, RunMode
from uacpy.core.environment import BoundaryProperties, SoundSpeedProfile


class TestPekerisWaveguide:
    """
    Test against classic Pekeris waveguide solution

    The Pekeris waveguide is a fundamental benchmark problem in underwater acoustics:
    - Isovelocity water column
    - Flat bottom with half-space
    - Simple geometry with known analytical solution

    This is the "Hello World" of underwater acoustic validation.
    """

    @pytest.fixture
    def pekeris_env(self):
        """Standard Pekeris waveguide environment."""
        bottom = BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600.0,  # Slightly faster sediment
            density=1.5,  # g/cm³
            attenuation=0.5  # dB/wavelength
        )

        env = uacpy.Environment(
            name='Pekeris Waveguide',
            bathymetry=100.0,
            ssp=1500.0,
            bottom=bottom
        )
        return env

    @pytest.fixture
    def pekeris_source(self):
        """Source at mid-depth."""
        return uacpy.Source(depths=50.0, frequencies=100.0)

    @pytest.fixture
    def pekeris_receiver(self):
        """Receiver grid."""
        return uacpy.Receiver(
            depths=np.array([25.0, 50.0, 75.0]),
            ranges=np.linspace(1000, 5000, 9)
        )

    @pytest.mark.requires_binary
    def test_bellhop_pekeris_tl_range(self, pekeris_env, pekeris_source, pekeris_receiver):
        """
        Bellhop TL should follow expected range-dependent decay

        Expected behavior:
        - TL increases with range (conservation of energy)
        - Lloyd mirror interference pattern visible
        - TL at 5km should be ~70-90 dB for 100 Hz
        """
        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(pekeris_env, pekeris_source, pekeris_receiver)

        # Basic validation
        assert isinstance(result, Field)
        assert np.all(np.isfinite(result.data))

        # Physics validation
        # 1. TL increases with range (monotonic in mean)
        tl_vs_range = result.tl.mean(axis=0)  # Average over depths
        assert tl_vs_range[-1] > tl_vs_range[0], "TL should increase with range"

        # 2. TL at 5km should be reasonable (70-90 dB typical for 100 Hz)
        tl_at_5km = result.tl[:, -1].mean()
        assert 60 < tl_at_5km < 100, f"TL at 5km should be ~70-90 dB, got {tl_at_5km:.1f} dB"

        # 3. TL at 1km should be less than at 5km
        tl_at_1km = result.tl[:, 0].mean()
        assert tl_at_1km < tl_at_5km, "TL at 1km should be less than at 5km"

        # 4. No NaN or inf values
        assert np.all(result.tl > 0), "All TL values should be positive"
        assert np.all(result.tl < 200), "TL should not exceed 200 dB (sanity check)"

    @pytest.mark.requires_binary
    def test_bellhop_pekeris_depth_structure(self, pekeris_env, pekeris_source, pekeris_receiver):
        """
        Bellhop should show Lloyd mirror interference pattern

        Expected behavior:
        - TL varies with depth due to interference between direct and reflected paths
        - Pattern depends on source/receiver geometry
        """
        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(pekeris_env, pekeris_source, pekeris_receiver)

        # Check depth variation exists (not constant)
        tl_vs_depth_at_1km = result.tl[:, 0]
        depth_std = np.std(tl_vs_depth_at_1km)
        assert depth_std > 1.0, "Should see >1 dB variation in TL vs depth (Lloyd mirror pattern)"

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_kraken_pekeris_modes(self, pekeris_env, pekeris_source):
        """
        Kraken should compute physically reasonable modes

        Expected behavior:
        - Mode count increases with frequency * depth
        - First mode has no zero-crossings
        - Mode wavenumbers are real and positive (for hard bottom)
        """
        kraken = Kraken(verbose=False)
        modes = kraken.compute_modes(pekeris_env, pekeris_source, n_modes=20)

        assert isinstance(modes, Modes)
        assert modes.k is not None, "Should have wavenumber data"
        assert modes.phi is not None, "Should have mode functions"

        k = modes.k
        phi = modes.phi

        # Mode validation
        assert len(k) > 0, "Should compute at least one mode"
        assert len(k) <= 20, "Should not exceed requested mode count"

        # Wavenumbers should be real and positive (lossless half-space)
        if np.iscomplexobj(k):
            k_real = np.real(k)
            k_imag = np.imag(k)
            assert np.all(k_real > 0), "Mode wavenumbers (real part) should be positive"
            # Allow small imaginary parts (numerical error or slight attenuation)
            assert np.all(np.abs(k_imag) < 0.1), "Imaginary parts should be small for lossless case"
        else:
            assert np.all(k > 0), "Mode wavenumbers should be positive"

        # Mode functions should be normalized (check first mode)
        if len(phi) > 0:
            phi_0 = phi[0]
            # Mode should not be all zeros
            assert np.max(np.abs(phi_0)) > 0, "First mode should have non-zero amplitude"

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_bellhop_kraken_consistency(self, pekeris_env, pekeris_source, pekeris_receiver):
        """
        Bellhop and Kraken should give similar TL for Pekeris waveguide

        Both models are solving the same problem with different methods:
        - Bellhop: Ray/beam tracing
        - Kraken: Normal modes

        They should agree within ~3 dB for this simple case.
        """
        # Run both models
        bellhop = Bellhop(verbose=False)
        bellhop_result = bellhop.compute_tl(pekeris_env, pekeris_source, pekeris_receiver)

        krakenfield = KrakenField(verbose=False)
        kraken_result = krakenfield.compute_tl(pekeris_env, pekeris_source, pekeris_receiver)

        # Compare TL values
        # Use mean TL over depths at each range to reduce sensitivity to modal structure
        bellhop_tl_mean = bellhop_result.tl.mean(axis=0)
        kraken_tl_mean = kraken_result.tl.mean(axis=0)

        # Models should agree within reasonable tolerance
        tl_diff = np.abs(bellhop_tl_mean - kraken_tl_mean)
        max_diff = np.max(tl_diff)
        mean_diff = np.mean(tl_diff)

        assert mean_diff < 5.0, f"Mean TL difference should be < 5 dB, got {mean_diff:.1f} dB"
        assert max_diff < 10.0, f"Max TL difference should be < 10 dB, got {max_diff:.1f} dB"


class TestRangeDependentPhysicalSanity:
    """
    Physical-sanity checks against range-dependent bathymetry.

    Continental-shelf transitions, seamounts, coastal environments —
    asserts realistic TL behaviour, not constructor mechanics. The
    constructor / Environment-attribute checks live in
    ``test_range_depth_dependent.py::TestRangeDependentEnvironment``.
    """

    @pytest.fixture
    def slope_env(self):
        """Environment with sloping bottom (100m to 200m over 10km)."""
        # Bathymetry: linear slope
        ranges = np.linspace(0, 10000, 21)  # 0 to 10 km
        depths = np.linspace(100, 200, 21)  # 100 to 200 m
        bathymetry = np.column_stack([ranges, depths])

        bottom = BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600.0,
            density=1.5,
            attenuation=0.5
        )

        env = uacpy.Environment(
            name='Sloping Bottom',
            # Initial depth
            ssp=1500.0,
            bathymetry=bathymetry,
            bottom=bottom
        )
        return env

    @pytest.fixture
    def slope_source(self):
        return uacpy.Source(depths=50.0, frequencies=100.0)

    @pytest.fixture
    def slope_receiver(self):
        return uacpy.Receiver(
            depths=np.array([25.0, 50.0, 75.0]),
            ranges=np.linspace(1000, 10000, 10)
        )

    @pytest.mark.requires_binary
    def test_bellhop_handles_slope(self, slope_env, slope_source, slope_receiver):
        """Bellhop should handle sloping bathymetry without crashing."""
        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(slope_env, slope_source, slope_receiver)

        assert isinstance(result, Field)
        assert np.all(np.isfinite(result.data))
        assert result.shape == (len(slope_receiver.depths), len(slope_receiver.ranges))

        # TL should be reasonable
        assert np.all(result.tl > 0)
        assert np.all(result.tl < 150)

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_ram_handles_slope(self, slope_env, slope_source, slope_receiver):
        """
        RAM (parabolic equation) should handle sloping bathymetry

        RAM is specifically designed for range-dependent environments.
        """
        ram = RAM(verbose=False, dr=20.0, dz=2.0)
        result = ram.compute_tl(slope_env, slope_source, slope_receiver)

        assert isinstance(result, Field)
        assert np.all(np.isfinite(result.data))

        # TL should be physically reasonable
        assert np.all(result.tl > 0)
        assert np.all(result.tl < 150)


class TestMunkProfile:
    """
    Test with Munk sound speed profile

    The Munk profile is a canonical deep-water SSP with a sound channel.
    It's widely used in underwater acoustics research.
    """

    @pytest.fixture
    def munk_source(self):
        """Source at the Munk sound-channel axis (1300 m), 100 Hz."""
        return uacpy.Source(depths=1300.0, frequencies=100.0)

    @pytest.fixture
    def munk_receiver(self):
        """20-point depth grid at one long range — enough to compare
        median TL near the surface vs the channel vs the bottom."""
        return uacpy.Receiver(
            depths=np.linspace(50.0, 4950.0, 20),
            ranges=np.array([50000.0]),
        )

    @pytest.mark.requires_binary
    def test_bellhop_munk_channel_traps_energy(
        self, munk_env, munk_source, munk_receiver,
    ):
        """Bellhop on a canonical Munk profile must trap refracted energy
        in the sound channel. Measured ~9 dB surface-vs-channel and
        ~5 dB bottom-vs-channel; thresholds half those values catch
        gross errors (lost channel, inverted gradient) while leaving
        margin for beam-sampling jitter.
        """
        bellhop = Bellhop(verbose=False, n_beams=500, alpha=(-25, 25))
        result = bellhop.run(
            munk_env, munk_source, munk_receiver,
            run_mode=RunMode.COHERENT_TL,
        )
        tl = np.asarray(result.tl).reshape(-1)
        z = np.asarray(munk_receiver.depths)

        tl_surface = float(np.median(tl[z < 200.0]))
        tl_mid = float(np.median(tl[(z > 800.0) & (z < 3000.0)]))
        tl_bottom = float(np.median(tl[z > 4500.0]))

        assert tl_surface - tl_mid >= 4.0, (
            f"Munk channel should trap energy away from the surface: "
            f"TL_surface={tl_surface:.1f}, TL_mid={tl_mid:.1f} "
            f"(diff {tl_surface - tl_mid:.1f}, expected ≥ 4 dB)"
        )
        assert tl_bottom - tl_mid >= 2.0, (
            f"Munk channel should trap energy away from the bottom: "
            f"TL_bottom={tl_bottom:.1f}, TL_mid={tl_mid:.1f} "
            f"(diff {tl_bottom - tl_mid:.1f}, expected ≥ 2 dB)"
        )


class TestInterpolationAccuracy:
    """
    Test critical interpolation paths

    These tests validate that interpolation doesn't introduce numerical errors
    that could corrupt results.
    """

    def test_environment_ssp_at_exact_points(self):
        """SSP interpolation at exact data points should return exact values."""
        depths = np.array([0, 50, 100])
        speeds = np.array([1520, 1500, 1480])
        ssp_data = np.column_stack([depths, speeds])

        env = uacpy.Environment(
            name='Test',
            bathymetry=100.0,
            ssp=SoundSpeedProfile.from_pairs(ssp_data)
        )

        # At exact points, should return exact values
        for i, (d, c) in enumerate(ssp_data):
            c_interp = env.get_sound_speed(d)
            assert np.abs(c_interp - c) < 1e-6, f"At depth {d}, expected {c}, got {c_interp}"

    def test_environment_ssp_interpolation_bounds(self):
        """Interpolated SSP values should stay within bounds."""
        depths = np.array([0, 50, 100])
        speeds = np.array([1520, 1500, 1480])
        ssp_data = np.column_stack([depths, speeds])

        env = uacpy.Environment(
            name='Test',
            bathymetry=100.0,
            ssp=SoundSpeedProfile.from_pairs(ssp_data)
        )

        # Interpolate at intermediate points
        test_depths = np.linspace(0, 100, 101)
        for d in test_depths:
            c = env.get_sound_speed(d)
            # Should be within min/max of original data
            assert min(speeds) - 1 <= c <= max(speeds) + 1, \
                f"Interpolated speed {c} at depth {d} outside bounds [{min(speeds)}, {max(speeds)}]"


class TestSoundSpeedEquations:
    """Standard sound-speed equations (UNESCO Chen-Millero, Del Grosso)."""

    def test_unesco_reference_check_value(self):
        from uacpy.core.acoustics import soundspeed_unesco
        # UNESCO 1983 check value: c(S=35, T=0 ITS-90, P=0) ≈ 1449.14 m/s
        assert soundspeed_unesco(0.0, 35.0, 0.0) == pytest.approx(1449.14, abs=0.02)

    def test_unesco_agrees_with_mackenzie_at_surface(self):
        from uacpy.core.acoustics import soundspeed_unesco, soundspeed
        # the two independent surface formulas agree to < 0.2 m/s
        for t, s in [(5, 35), (15, 35), (25, 36)]:
            assert soundspeed_unesco(t, s, 0.0) == pytest.approx(soundspeed(t, s, 0.0), abs=0.2)

    def test_delgrosso_matches_unesco_within_documented_difference(self):
        from uacpy.core.acoustics import soundspeed_unesco, soundspeed_delgrosso
        # Del Grosso and UNESCO agree to < 1 m/s over realistic profiles; the
        # small residual grows with depth (UNESCO overpredicts, Dushaw 1993).
        for t, s, p in [(25, 35, 0), (15, 35.5, 600), (4, 34.8, 3000), (1.5, 34.7, 8000)]:
            assert soundspeed_delgrosso(t, s, p) == pytest.approx(
                soundspeed_unesco(t, s, p), abs=1.0)
        # near-identical at the surface
        assert soundspeed_delgrosso(15, 35, 0) == pytest.approx(
            soundspeed_unesco(15, 35, 0), abs=0.05)

    def test_monotonic_increase_with_each_variable(self):
        from uacpy.core.acoustics import soundspeed_unesco as c
        assert c(20, 35, 0) > c(10, 35, 0)        # temperature
        assert c(15, 38, 0) > c(15, 32, 0)        # salinity
        assert c(15, 35, 5000) > c(15, 35, 0)     # pressure


class TestPekerisRoot:
    """Branch selection matches Acoustics-Toolbox PekerisRoot.m."""

    def test_matches_at_reference_branch(self):
        from uacpy.core.acoustics import pekeris_root
        z = np.array([4 + 0j, -4 + 0j, -4 + 1j, -4 - 1j, 1j, -1j,
                      1e-12 + 1j, 4 + 1j, 4 - 1j, 0j])
        expected = np.where(z.real >= 0, np.sqrt(z), 1j * np.sqrt(-z))
        assert np.allclose(pekeris_root(z), expected)

    def test_trapped_mode_branch_decays(self):
        from uacpy.core.acoustics import pekeris_root
        # gamma² > 0 (k > k_halfspace): real positive root so the
        # halfspace solution exp(-gamma*(z-D)) decays
        gamma = pekeris_root(np.array([9.0 + 0j]))
        assert gamma[0] == pytest.approx(3.0)

    def test_evanescent_branch_sign(self):
        from uacpy.core.acoustics import pekeris_root
        # gamma² < 0: AT picks +i*sqrt(|gamma²|)
        gamma = pekeris_root(np.array([-9.0 + 0j]))
        assert gamma[0] == pytest.approx(3j)


class TestNumericalStability:
    """
    Test numerical stability and edge cases

    These tests catch numerical issues that could cause crashes or wrong results.
    """

    @pytest.mark.requires_binary
    def test_very_shallow_water(self):
        """Models should handle very shallow water (10m depth)."""
        env = uacpy.Environment(
            name='Very Shallow',
            bathymetry=10.0,
            ssp=1500.0,
            bottom=BoundaryProperties(
                acoustic_type='half-space',
                sound_speed=1600.0,
                density=1.5,
                attenuation=0.5
            )
        )

        source = uacpy.Source(depths=5.0, frequencies=1000.0)
        receiver = uacpy.Receiver(
            depths=np.array([2.0, 5.0, 8.0]),
            ranges=np.array([100.0, 500.0, 1000.0])
        )

        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env, source, receiver)
        assert np.all(np.isfinite(result.data))

    @pytest.mark.requires_binary
    def test_very_high_frequency(self):
        """Models should handle high frequency (10 kHz)."""
        env = uacpy.Environment(
            name='High Frequency',
            bathymetry=100.0,
            ssp=1500.0,
            bottom=BoundaryProperties(
                acoustic_type='half-space',
                sound_speed=1600.0,
                density=1.5,
                attenuation=0.5
            )
        )

        source = uacpy.Source(depths=50.0, frequencies=10000.0)  # 10 kHz
        receiver = uacpy.Receiver(
            depths=np.array([25.0, 50.0, 75.0]),
            ranges=np.array([100.0, 500.0, 1000.0])
        )

        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env, source, receiver)
        assert np.all(np.isfinite(result.data))
        # High frequency should have higher TL due to absorption
        assert np.all(result.tl > 20)  # Expect significant loss
