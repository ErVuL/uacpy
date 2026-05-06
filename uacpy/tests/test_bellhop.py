"""Bellhop ray/beam-focused tests."""

import pytest
import numpy as np

import uacpy
from uacpy.core.environment import SoundSpeedProfile
from uacpy.models import Bellhop, RAM, Kraken, KrakenC, Scooter, SPARC
from uacpy.models.base import RunMode
from uacpy.core import Environment, BoundaryProperties, Source, Receiver
from uacpy.core.exceptions import ExecutableNotFoundError, UnsupportedFeatureError

pytestmark = pytest.mark.requires_binary

class TestBellhopRunModes:
    """Test all Bellhop run modes systematically."""

    @pytest.fixture
    def setup_env(self):
        """Create environment for run mode tests."""
        return Environment(
            name="run_mode_test",
            depth=100.0,
            sound_speed=1500.0
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
        """Test Bellhop coherent TL (run_mode=RunMode.COHERENT_TL)."""
        bellhop = Bellhop(verbose=False)
        result = bellhop.run(
            env=setup_env,
            source=setup_source,
            receiver=setup_receiver,
            run_mode=RunMode.COHERENT_TL
        )

        assert result.field_type == 'tl'
        assert result.shape == (len(setup_receiver.depths), len(setup_receiver.ranges))
        assert np.all(np.isfinite(result.data))
        assert np.all(result.data > 0), "TL should be positive"

    @pytest.mark.requires_binary
    def test_bellhop_incoherent_tl(self, setup_env, setup_source, setup_receiver):
        """Test Bellhop incoherent TL (run_mode=RunMode.INCOHERENT_TL)."""
        bellhop = Bellhop(verbose=False)
        result = bellhop.run(
            env=setup_env,
            source=setup_source,
            receiver=setup_receiver,
            run_mode=RunMode.INCOHERENT_TL
        )

        assert result.field_type == 'tl'
        assert result.shape == (len(setup_receiver.depths), len(setup_receiver.ranges))
        assert np.all(np.isfinite(result.data))

    @pytest.mark.requires_binary
    def test_bellhop_semicoherent_tl(self, setup_env, setup_source, setup_receiver):
        """Test Bellhop semi-coherent TL (run_mode=RunMode.SEMICOHERENT_TL)."""
        bellhop = Bellhop(verbose=False)
        result = bellhop.run(
            env=setup_env,
            source=setup_source,
            receiver=setup_receiver,
            run_mode=RunMode.SEMICOHERENT_TL
        )

        assert result.field_type == 'tl'
        assert result.shape == (len(setup_receiver.depths), len(setup_receiver.ranges))
        assert np.all(np.isfinite(result.data))

    @pytest.mark.requires_binary
    def test_bellhop_rays(self, setup_env, setup_source, setup_receiver):
        """Test Bellhop ray tracing (run_mode=RunMode.RAYS)."""
        bellhop = Bellhop(verbose=False)
        result = bellhop.run(
            env=setup_env,
            source=setup_source,
            receiver=setup_receiver,
            run_mode=RunMode.RAYS
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
        """Test Bellhop eigenrays (run_mode=RunMode.EIGENRAYS)."""
        bellhop = Bellhop(verbose=False)

        # Eigenrays need a specific receiver point
        receiver = Receiver(depths=[50.0], ranges=[3000.0])

        result = bellhop.run(
            env=setup_env,
            source=setup_source,
            receiver=receiver,
            run_mode=RunMode.EIGENRAYS
        )

        # Bellhop returns eigenrays with field_type='rays' (same as regular rays)
        assert result.field_type == 'rays'
        # Should have ray data in metadata
        assert 'rays' in result.metadata
        rays = result.metadata['rays']
        assert len(rays) > 0, "Should have computed some eigenrays"

    @pytest.mark.requires_binary
    def test_bellhop_arrivals(self, setup_env, setup_source):
        """Test Bellhop arrivals (run_mode=RunMode.ARRIVALS)."""
        bellhop = Bellhop(verbose=False)

        # Arrivals at specific points
        receiver = Receiver(depths=[50.0], ranges=[3000.0])

        result = bellhop.run(
            env=setup_env,
            source=setup_source,
            receiver=receiver,
            run_mode=RunMode.ARRIVALS
        )

        assert result.field_type == 'arrivals'
        # Check for arrival structure in metadata (nested per-receiver format)
        assert 'arrivals_by_receiver' in result.metadata


class TestAdvancedBeamTypes:
    """Tests for advanced Bellhop beam types (Priority 1 gap)."""

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
        """Test Gaussian beam (type 'B' - default)."""
        bellhop = Bellhop(verbose=False, beam_type='B')
        result = bellhop.run(env=env, source=source, receiver=receiver)
        assert result.field_type == 'tl'
        assert np.all(np.isfinite(result.data))

    @pytest.mark.requires_binary
    def test_geometric_beam_hat(self, env, source, receiver):
        """Test geometric hat beam (type 'G')."""
        bellhop = Bellhop(verbose=False, beam_type='G')
        result = bellhop.run(env=env, source=source, receiver=receiver)
        assert result.field_type == 'tl'
        assert np.all(np.isfinite(result.data))

    @pytest.mark.requires_binary
    def test_simple_gaussian_beam(self, env, source, receiver):
        """Test simple Gaussian beam (type 'S')."""
        bellhop = Bellhop(verbose=False, beam_type='S')
        result = bellhop.run(env=env, source=source, receiver=receiver)
        assert result.field_type == 'tl'
        assert np.all(np.isfinite(result.data))

    @pytest.mark.requires_binary
    def test_cartesian_beam(self, env, source, receiver):
        """Test Cartesian beam (type 'C')."""
        bellhop = Bellhop(verbose=False, beam_type='C')
        result = bellhop.run(env=env, source=source, receiver=receiver)
        assert result.field_type == 'tl'
        assert np.all(np.isfinite(result.data))

    @pytest.mark.requires_binary
    def test_ray_centered_beam(self, env, source, receiver):
        """Test ray-centered beam (type 'R')."""
        bellhop = Bellhop(verbose=False, beam_type='R')
        result = bellhop.run(env=env, source=source, receiver=receiver)
        assert result.field_type == 'tl'
        assert np.all(np.isfinite(result.data))
