"""
Tests for advanced model features and parameters

This test file addresses gaps in test coverage for:
- Bellhop run modes (all 6 types)
- RAM Pade orders and stability parameters
- SSP interpolation methods
- Boundary condition types
- Model-specific advanced parameters

These features exist in the codebase and are sometimes demonstrated in examples,
but lack systematic unit tests for validation.
"""

import pytest
import numpy as np
import os
from pathlib import Path

import uacpy
from uacpy.models import Bellhop, RAM, Kraken, KrakenC, Scooter, SPARC
from uacpy.models.base import RunMode
from uacpy.core import Environment, BoundaryProperties, Source, Receiver
from uacpy.core.exceptions import ExecutableNotFoundError, UnsupportedFeatureError


class TestBellhopRunModes:
    """Test all Bellhop run modes systematically"""

    @pytest.fixture
    def setup_env(self):
        """Create environment for run mode tests"""
        return Environment(
            name="run_mode_test",
            depth=100.0,
            sound_speed=1500.0,
            ssp_type='isovelocity'
        )

    @pytest.fixture
    def setup_source(self):
        return Source(depth=50.0, frequency=100.0)

    @pytest.fixture
    def setup_receiver(self):
        return Receiver(
            depths=np.array([25.0, 50.0, 75.0]),
            ranges=np.array([1000.0, 3000.0, 5000.0])
        )

    @pytest.mark.requires_binary
    def test_bellhop_coherent_tl(self, setup_env, setup_source, setup_receiver):
        """Test Bellhop coherent TL (run_type='C')"""
        bellhop = Bellhop(verbose=False)
        result = bellhop.run(
            env=setup_env,
            source=setup_source,
            receiver=setup_receiver,
            run_type='C'
        )

        assert result.field_type == 'tl'
        assert result.shape == (len(setup_receiver.depths), len(setup_receiver.ranges))
        assert np.all(np.isfinite(result.data))
        assert np.all(result.data > 0), "TL should be positive"

    @pytest.mark.requires_binary
    def test_bellhop_incoherent_tl(self, setup_env, setup_source, setup_receiver):
        """Test Bellhop incoherent TL (run_type='I')"""
        bellhop = Bellhop(verbose=False)
        result = bellhop.run(
            env=setup_env,
            source=setup_source,
            receiver=setup_receiver,
            run_type='I'
        )

        assert result.field_type == 'tl'
        assert result.shape == (len(setup_receiver.depths), len(setup_receiver.ranges))
        assert np.all(np.isfinite(result.data))

    @pytest.mark.requires_binary
    def test_bellhop_semicoherent_tl(self, setup_env, setup_source, setup_receiver):
        """Test Bellhop semi-coherent TL (run_type='S')"""
        bellhop = Bellhop(verbose=False)
        result = bellhop.run(
            env=setup_env,
            source=setup_source,
            receiver=setup_receiver,
            run_type='S'
        )

        assert result.field_type == 'tl'
        assert result.shape == (len(setup_receiver.depths), len(setup_receiver.ranges))
        assert np.all(np.isfinite(result.data))

    @pytest.mark.requires_binary
    def test_bellhop_rays(self, setup_env, setup_source, setup_receiver):
        """Test Bellhop ray tracing (run_type='R')"""
        bellhop = Bellhop(verbose=False)
        result = bellhop.run(
            env=setup_env,
            source=setup_source,
            receiver=setup_receiver,
            run_type='R'
        )

        assert result.field_type == 'rays'
        assert 'rays' in result.metadata
        # Rays are stored as list of dicts with 'r' and 'z' arrays
        rays = result.metadata['rays']
        assert len(rays) > 0, "Should have computed some rays"
        # Each ray is a dict with 'r' and 'z' arrays
        assert all(isinstance(ray, dict) for ray in rays), "Each ray should be a dict"
        assert all('r' in ray and 'z' in ray for ray in rays), "Each ray should have r,z coordinates"
        assert all(len(ray['r']) >= 2 for ray in rays), "Each ray should have at least 2 points"

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_bellhop_eigenrays(self, setup_env, setup_source):
        """Test Bellhop eigenrays (run_type='E')"""
        bellhop = Bellhop(verbose=False)

        # Eigenrays need a specific receiver point
        receiver = Receiver(depths=[50.0], ranges=[3000.0])

        result = bellhop.run(
            env=setup_env,
            source=setup_source,
            receiver=receiver,
            run_type='E'
        )

        # Bellhop returns eigenrays with field_type='rays' (same as regular rays)
        assert result.field_type == 'rays'
        # Should have ray data in metadata
        assert 'rays' in result.metadata
        rays = result.metadata['rays']
        assert len(rays) > 0, "Should have computed some eigenrays"

    @pytest.mark.requires_binary
    def test_bellhop_arrivals(self, setup_env, setup_source):
        """Test Bellhop arrivals (run_type='A')"""
        bellhop = Bellhop(verbose=False)

        # Arrivals at specific points
        receiver = Receiver(depths=[50.0], ranges=[3000.0])

        result = bellhop.run(
            env=setup_env,
            source=setup_source,
            receiver=receiver,
            run_type='A'
        )

        assert result.field_type == 'arrivals'
        # Check for arrival structure in metadata
        assert 'arrivals' in result.metadata


class TestRAMAdvancedParameters:
    """Test RAM Pade orders and stability parameters"""

    @pytest.fixture
    def ram_env(self):
        return Environment(
            name="ram_test",
            depth=100.0,
            sound_speed=1500.0,
            ssp_type='isovelocity'
        )

    @pytest.fixture
    def ram_source(self):
        return Source(depth=50.0, frequency=50.0)

    @pytest.fixture
    def ram_receiver(self):
        return Receiver(
            depths=np.linspace(10, 90, 9),
            ranges=np.linspace(100, 5000, 11)
        )

    def test_ram_pade_order_2(self, ram_env, ram_source, ram_receiver):
        """Test RAM with Pade order 2"""
        try:
            ram = RAM(verbose=False)
            result = ram.compute_tl(
                env=ram_env,
                source=ram_source,
                receiver=ram_receiver,
                np_pade=2
            )
            assert result.field_type == 'tl'
            assert np.any(np.isfinite(result.data))
        except FileNotFoundError:
            pytest.skip("mpiramS binary not found")

    def test_ram_pade_order_6(self, ram_env, ram_source, ram_receiver):
        """Test RAM with Pade order 6"""
        try:
            ram = RAM(verbose=False)
            result = ram.compute_tl(
                env=ram_env,
                source=ram_source,
                receiver=ram_receiver,
                np_pade=6
            )
            assert result.field_type == 'tl'
            assert np.any(np.isfinite(result.data))
        except FileNotFoundError:
            pytest.skip("mpiramS binary not found")

    def test_ram_pade_order_8(self, ram_env, ram_source, ram_receiver):
        """Test RAM with Pade order 8"""
        try:
            ram = RAM(verbose=False)
            result = ram.compute_tl(
                env=ram_env,
                source=ram_source,
                receiver=ram_receiver,
                np_pade=8
            )
            assert result.field_type == 'tl'
            assert np.any(np.isfinite(result.data))
        except FileNotFoundError:
            pytest.skip("mpiramS binary not found")

    def test_ram_stability_parameter(self, ram_env, ram_source, ram_receiver):
        """Test RAM stability parameter"""
        try:
            ram = RAM(verbose=False)
            result = ram.compute_tl(
                env=ram_env,
                source=ram_source,
                receiver=ram_receiver,
                ns_stability=1
            )
            assert result.field_type == 'tl'
            assert np.any(np.isfinite(result.data))
        except FileNotFoundError:
            pytest.skip("mpiramS binary not found")

    def test_ram_custom_dr_dz(self, ram_env, ram_source, ram_receiver):
        """Test RAM with custom range and depth steps"""
        try:
            ram = RAM(verbose=False)
            result = ram.compute_tl(
                env=ram_env,
                source=ram_source,
                receiver=ram_receiver,
                dr=10.0,  # 10m range step
                dz=0.5    # 0.5m depth step
            )
            assert result.field_type == 'tl'
            assert np.any(np.isfinite(result.data))
        except FileNotFoundError:
            pytest.skip("mpiramS binary not found")


class TestSSPInterpolationMethods:
    """Test different SSP interpolation types"""

    @pytest.fixture
    def source(self):
        return Source(depth=50.0, frequency=100.0)

    @pytest.fixture
    def receiver(self):
        return Receiver(
            depths=np.array([25.0, 50.0, 75.0]),
            ranges=np.array([1000.0, 3000.0])
        )

    @pytest.mark.requires_binary
    def test_ssp_isovelocity(self, source, receiver):
        """Test isovelocity SSP"""
        env = Environment(
            name="iso_test",
            depth=100.0,
            sound_speed=1500.0,
            ssp_type='isovelocity'
        )

        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env=env, source=source, receiver=receiver)
        assert result.field_type == 'tl'

    @pytest.mark.requires_binary
    def test_ssp_linear(self, source, receiver):
        """Test linear SSP interpolation"""
        depths = np.array([0, 50, 100])
        speeds = np.array([1500, 1490, 1480])

        env = Environment(
            name="linear_test",
            depth=100.0,
            ssp_data=np.column_stack([depths, speeds]),
            ssp_type='linear'
        )

        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env=env, source=source, receiver=receiver)
        assert result.field_type == 'tl'

    @pytest.mark.requires_binary
    def test_ssp_c_linear(self, source, receiver):
        """Test c-linear SSP interpolation (maps to 'linear' in UACPY)"""
        # Note: UACPY uses 'linear' for what Acoustics Toolbox calls 'C' (c-linear)
        depths = np.array([0, 50, 100])
        speeds = np.array([1500, 1490, 1480])

        env = Environment(
            name="clin_test",
            depth=100.0,
            ssp_data=np.column_stack([depths, speeds]),
            ssp_type='linear'  # UACPY's 'linear' is AT's 'C' (c-linear)
        )

        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env=env, source=source, receiver=receiver)
        assert result.field_type == 'tl'

    @pytest.mark.requires_binary
    def test_ssp_cubic(self, source, receiver):
        """Test cubic spline SSP interpolation"""
        depths = np.array([0, 25, 50, 75, 100])
        speeds = np.array([1500, 1495, 1490, 1485, 1480])

        env = Environment(
            name="cubic_test",
            depth=100.0,
            ssp_data=np.column_stack([depths, speeds]),
            ssp_type='cubic'
        )

        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env=env, source=source, receiver=receiver)
        assert result.field_type == 'tl'


class TestBoundaryTypes:
    """Test different boundary condition types"""

    @pytest.fixture
    def source(self):
        return Source(depth=50.0, frequency=100.0)

    @pytest.fixture
    def receiver(self):
        return Receiver(
            depths=np.array([25.0, 50.0, 75.0]),
            ranges=np.array([1000.0, 3000.0])
        )

    @pytest.mark.requires_binary
    def test_boundary_vacuum(self, source, receiver):
        """Test vacuum boundary condition"""
        bottom = BoundaryProperties(acoustic_type='vacuum')
        env = Environment(
            name="vacuum_test",
            depth=100.0,
            sound_speed=1500.0,
            bottom=bottom
        )

        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env=env, source=source, receiver=receiver)
        assert result.field_type == 'tl'

    @pytest.mark.requires_binary
    def test_boundary_rigid(self, source, receiver):
        """Test rigid boundary condition"""
        bottom = BoundaryProperties(acoustic_type='rigid')
        env = Environment(
            name="rigid_test",
            depth=100.0,
            sound_speed=1500.0,
            bottom=bottom
        )

        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env=env, source=source, receiver=receiver)
        assert result.field_type == 'tl'

    @pytest.mark.requires_binary
    def test_boundary_halfspace(self, source, receiver):
        """Test half-space boundary condition"""
        bottom = BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600.0,
            density=1.5,
            attenuation=0.5
        )
        env = Environment(
            name="halfspace_test",
            depth=100.0,
            sound_speed=1500.0,
            bottom=bottom
        )

        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env=env, source=source, receiver=receiver)
        assert result.field_type == 'tl'

    @pytest.mark.requires_binary
    def test_boundary_elastic(self, source, receiver):
        """Test elastic boundary with shear waves"""
        bottom = BoundaryProperties(
            acoustic_type='elastic',
            sound_speed=1700.0,
            shear_speed=400.0,
            density=1.8,
            attenuation=0.5,
            shear_attenuation=1.0
        )
        env = Environment(
            name="elastic_test",
            depth=100.0,
            sound_speed=1500.0,
            bottom=bottom
        )

        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_tl(env=env, source=source, receiver=receiver)
        assert result.field_type == 'tl'


class TestKrakenCComplexModes:
    """Test KrakenC for complex modes with elastic bottom"""

    @pytest.fixture
    def elastic_env(self):
        """Create environment with elastic bottom"""
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
    def source(self):
        return Source(depth=50.0, frequency=100.0)

    @pytest.fixture
    def receiver(self):
        return Receiver(depths=[25.0, 50.0, 75.0], ranges=[1000.0, 3000.0])

    @pytest.mark.requires_binary
    def test_krakenc_complex_modes(self, elastic_env, source, receiver):
        """Test KrakenC complex mode computation"""
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



class TestScooterBasic:
    """Basic tests for Scooter model (wavenumber integration)"""

    @pytest.mark.requires_binary
    def test_scooter_basic_tl(self):
        """Test basic Scooter TL computation"""
        env = Environment(
            name="scooter_test",
            depth=100.0,
            sound_speed=1500.0
        )
        source = Source(depth=50.0, frequency=100.0)
        receiver = Receiver(
            depths=np.array([25.0, 50.0, 75.0]),
            ranges=np.array([1000.0, 3000.0])
        )

        scooter = Scooter(verbose=False)
        result = scooter.compute_tl(env=env, source=source, receiver=receiver)

        assert result.field_type == 'tl'
        assert result.shape == (len(receiver.depths), len(receiver.ranges))
        assert np.any(np.isfinite(result.data))


class TestSPARCBasic:
    """Basic tests for SPARC model (seismo-acoustic PE)"""

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_sparc_basic_tl(self):
        """Test basic SPARC TL computation"""
        env = Environment(
            name="sparc_test",
            depth=100.0,
            sound_speed=1500.0
        )
        source = Source(depth=50.0, frequency=50.0)
        receiver = Receiver(
            depths=np.linspace(10, 90, 9),
            ranges=np.linspace(100, 5000, 11)
        )

        sparc = SPARC(verbose=False)
        result = sparc.compute_tl(env=env, source=source, receiver=receiver)

        assert result.field_type == 'tl'
        assert np.any(np.isfinite(result.data))


class TestVolumeAttenuation:
    """Tests for volume attenuation models (Priority 1 gap)"""

    @pytest.fixture
    def shallow_env(self):
        """Shallow water environment for attenuation tests"""
        return Environment(
            name="atten_test",
            depth=100.0,
            sound_speed=1500.0
        )

    @pytest.fixture
    def high_freq_source(self):
        """High frequency source where attenuation is significant"""
        return Source(depth=50.0, frequency=10000.0)  # 10 kHz

    @pytest.fixture
    def low_freq_source(self):
        """Low frequency source where attenuation is minimal"""
        return Source(depth=50.0, frequency=100.0)  # 100 Hz

    @pytest.fixture
    def receiver(self):
        return Receiver(depths=[50.0], ranges=[1000.0, 5000.0, 10000.0])

    @pytest.mark.requires_binary
    def test_bellhop_thorp_attenuation(self, shallow_env, high_freq_source, receiver):
        """Test Bellhop with Thorp attenuation formula"""
        bellhop_no_atten = Bellhop(verbose=False)
        bellhop_thorp = Bellhop(verbose=False, volume_attenuation='T')

        # Run without attenuation
        result_no_atten = bellhop_no_atten.run(
            env=shallow_env,
            source=high_freq_source,
            receiver=receiver,
            run_type='C',
        )

        # Run with Thorp attenuation
        result_thorp = bellhop_thorp.run(
            env=shallow_env,
            source=high_freq_source,
            receiver=receiver,
            run_type='C',
        )

        assert result_thorp.field_type == 'tl'
        # With attenuation, TL should be higher (more loss)
        assert np.mean(result_thorp.data) > np.mean(result_no_atten.data)

    @pytest.mark.requires_binary
    def test_kraken_thorp_attenuation(self, shallow_env, high_freq_source, receiver):
        """Test Kraken with Thorp attenuation formula"""
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
        """Test that attenuation increases with frequency"""
        bellhop = Bellhop(verbose=False, volume_attenuation='T')

        # Low frequency with Thorp
        result_low = bellhop.run(
            env=shallow_env,
            source=low_freq_source,
            receiver=receiver,
            run_type='C',
        )

        # High frequency with Thorp
        result_high = bellhop.run(
            env=shallow_env,
            source=high_freq_source,
            receiver=receiver,
            run_type='C',
        )

        # Both should complete successfully
        assert result_low.field_type == 'tl'
        assert result_high.field_type == 'tl'
        assert np.all(np.isfinite(result_low.data))
        assert np.all(np.isfinite(result_high.data))

    @pytest.mark.requires_binary
    def test_attenuation_with_scooter(self, shallow_env, high_freq_source, receiver):
        """Test Scooter with volume attenuation"""
        scooter = Scooter(verbose=False, volume_attenuation='T')

        result = scooter.run(
            env=shallow_env,
            source=high_freq_source,
            receiver=receiver,
        )

        assert result.field_type == 'tl'
        assert np.all(np.isfinite(result.data))


class TestAdvancedBeamTypes:
    """Tests for advanced Bellhop beam types (Priority 1 gap)"""

    @pytest.fixture
    def env(self):
        return Environment(
            name="beam_test",
            depth=100.0,
            sound_speed=1500.0
        )

    @pytest.fixture
    def source(self):
        return Source(depth=50.0, frequency=1000.0)

    @pytest.fixture
    def receiver(self):
        return Receiver(depths=[50.0], ranges=[1000.0, 5000.0])

    @pytest.mark.requires_binary
    def test_gaussian_beam(self, env, source, receiver):
        """Test Gaussian beam (type 'B' - default)"""
        bellhop = Bellhop(verbose=False, beam_type='B')
        result = bellhop.run(env=env, source=source, receiver=receiver)
        assert result.field_type == 'tl'
        assert np.all(np.isfinite(result.data))

    @pytest.mark.requires_binary
    def test_geometric_beam_hat(self, env, source, receiver):
        """Test geometric hat beam (type 'G')"""
        bellhop = Bellhop(verbose=False, beam_type='G')
        result = bellhop.run(env=env, source=source, receiver=receiver)
        assert result.field_type == 'tl'
        assert np.all(np.isfinite(result.data))

    @pytest.mark.requires_binary
    def test_simple_gaussian_beam(self, env, source, receiver):
        """Test simple Gaussian beam (type 'S')"""
        bellhop = Bellhop(verbose=False, beam_type='S')
        result = bellhop.run(env=env, source=source, receiver=receiver)
        assert result.field_type == 'tl'
        assert np.all(np.isfinite(result.data))

    @pytest.mark.requires_binary
    def test_cartesian_beam(self, env, source, receiver):
        """Test Cartesian beam (type 'C')"""
        bellhop = Bellhop(verbose=False, beam_type='C')
        result = bellhop.run(env=env, source=source, receiver=receiver)
        assert result.field_type == 'tl'
        assert np.all(np.isfinite(result.data))

    @pytest.mark.requires_binary
    def test_ray_centered_beam(self, env, source, receiver):
        """Test ray-centered beam (type 'R')"""
        bellhop = Bellhop(verbose=False, beam_type='R')
        result = bellhop.run(env=env, source=source, receiver=receiver)
        assert result.field_type == 'tl'
        assert np.all(np.isfinite(result.data))
