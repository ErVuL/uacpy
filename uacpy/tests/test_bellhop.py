"""Bellhop ray/beam-focused tests."""

import pytest
import numpy as np

from uacpy.models import Bellhop
from uacpy.models.base import RunMode
from uacpy.core import Environment, Source, Receiver

pytestmark = pytest.mark.requires_binary


class TestBellhopRunModes:
    """Test all Bellhop run modes systematically."""

    @pytest.fixture
    def setup_env(self):
        """Create environment for run mode tests."""
        return Environment(
            name="run_mode_test",
            bathymetry=100.0,
            ssp=1500.0
        )

    @pytest.fixture
    def setup_source(self):
        return Source(depths=50.0, frequencies=100.0)

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
        assert np.all(result.tl > 0), "TL should be positive"

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
        assert result.is_eigen is False
        # Receiver / source geometry attached by the wrapper.
        assert result.receiver_depths is not None
        assert result.receiver_ranges is not None
        assert result.source_depths is not None and len(result.source_depths) > 0

        rays = result.rays
        assert len(rays) > 0, "Should have computed some rays"
        assert all(isinstance(ray, dict) for ray in rays)
        assert all('r' in ray and 'z' in ray for ray in rays)
        assert all(len(ray['r']) >= 2 for ray in rays)

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_bellhop_eigenrays(self, setup_env, setup_source):
        """Test Bellhop eigenrays (run_mode=RunMode.EIGENRAYS)."""
        bellhop = Bellhop(verbose=False)

        receiver = Receiver(depths=[50.0], ranges=[3000.0])

        result = bellhop.run(
            env=setup_env,
            source=setup_source,
            receiver=receiver,
            run_mode=RunMode.EIGENRAYS
        )

        assert result.field_type == 'rays'
        # Wrapper must mark this as solver-computed eigenrays.
        assert result.is_eigen is True
        assert len(result.rays) > 0

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_bellhop_compute_eigenrays(self, setup_env, setup_source):
        """Test Bellhop.compute_eigenrays one-call API."""
        bellhop = Bellhop(verbose=False)

        rays = bellhop.compute_eigenrays(
            setup_env, setup_source,
            range=3000.0, depth=50.0,
        )
        assert rays.field_type == 'rays'
        assert rays.is_eigen is True
        # Receiver positions reflect the single target point.
        assert rays.receiver_ranges is not None
        assert float(rays.receiver_ranges[0]) == 3000.0
        assert float(rays.receiver_depths[0]) == 50.0
        # Filtering happens on Rays, not on the model.
        top4 = rays.top_n_by_miss(4)
        assert len(top4.rays) <= 4
        miss = [r['miss_distance_m'] for r in top4.rays]
        if len(miss) > 1:
            assert miss == sorted(miss)

    @pytest.mark.requires_binary
    def test_rays_filter_helpers_preserve_is_eigen(self, setup_env, setup_source, setup_receiver):
        """Rays.filter / filter_by_bounces / filter_by_launch_angle preserve is_eigen."""
        bellhop = Bellhop(verbose=False)
        rays = bellhop.run(
            env=setup_env, source=setup_source, receiver=setup_receiver,
            run_mode=RunMode.RAYS,
        )
        assert rays.is_eigen is False

        custom = rays.filter(lambda r: True)
        assert custom.is_eigen is False
        assert len(custom.rays) == len(rays.rays)

        sub = rays.filter_by_launch_angle(min_deg=-5.0, max_deg=5.0)
        assert sub.is_eigen is False
        assert all(-5.0 <= r['alpha'] <= 5.0 for r in sub.rays)

        direct = rays.filter_by_bounces(kind='direct')
        assert direct.is_eigen is False
        assert all(r.get('n_top_bounces', 0) == 0
                   and r.get('n_bot_bounces', 0) == 0
                   for r in direct.rays)

        # Exact-count form: bot=0 is "no bottom bounces".
        no_bot = rays.filter_by_bounces(bot=0)
        assert all(r.get('n_bot_bounces', 0) == 0 for r in no_bot.rays)

        # Range form: bot=(1, None) is "at least one bottom bounce".
        with_bot = rays.filter_by_bounces(bot=(1, None))
        assert all(r.get('n_bot_bounces', 0) >= 1 for r in with_bot.rays)

        # Closed range: top=(0, 1) is "0 or 1 surface bounces".
        few_top = rays.filter_by_bounces(top=(0, 1))
        assert all(0 <= r.get('n_top_bounces', 0) <= 1 for r in few_top.rays)

        import pytest as _pytest
        with _pytest.raises(ValueError):
            rays.filter_by_bounces(kind='bogus')

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
            bathymetry=100.0,
            ssp=1500.0
        )

    @pytest.fixture
    def source(self):
        return Source(depths=50.0, frequencies=1000.0)

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


class TestRunWithBounceConstructorPlumbing:
    """Verify Bellhop.run_with_bounce passes through volume-attenuation /
    c_low / c_high to the spawned Bounce instance."""

    def test_bounce_inherits_parent_volume_attenuation(self, monkeypatch):
        """Parent Bellhop's volume_attenuation / FG params reach Bounce."""
        from uacpy.models import bounce as bounce_mod

        captured = {}
        original_init = bounce_mod.Bounce.__init__

        def spy_init(self_, *args, **kwargs):
            captured.update(kwargs)
            original_init(self_, *args, **kwargs)
            # Stop early so we don't actually run BOUNCE.
            raise RuntimeError("stop after Bounce __init__")

        monkeypatch.setattr(bounce_mod.Bounce, '__init__', spy_init)

        bellhop = Bellhop(
            verbose=False,
            volume_attenuation='F',
            francois_garrison_params=(10.0, 35.0, 8.0, 1000.0),
            bio_layers=None,
        )
        env = Environment(name='b', bathymetry=100.0, ssp=1500.0)
        source = Source(depths=50.0, frequencies=100.0)
        receiver = Receiver(depths=[50.0], ranges=[1000.0])
        with pytest.raises(RuntimeError, match='stop after Bounce __init__'):
            bellhop.run_with_bounce(
                env=env, source=source, receiver=receiver,
                c_low=1450.0, c_high=20000.0, rmax_m=42000.0,
            )
        assert captured.get('volume_attenuation') == 'F'
        assert captured.get('francois_garrison_params') == (10.0, 35.0, 8.0, 1000.0)
        assert captured.get('c_low') == 1450.0
        assert captured.get('c_high') == 20000.0
        assert captured.get('rmax_m') == 42000.0


class TestBounceConstructorAcceptsFGParams:
    """B2: Bounce.__init__ now accepts francois_garrison_params / bio_layers."""

    def test_bounce_accepts_fg_params_kwarg(self):
        from uacpy.models import Bounce
        b = Bounce(
            volume_attenuation='F',
            francois_garrison_params=(10.0, 35.0, 8.0, 1000.0),
        )
        assert b.francois_garrison_params == (10.0, 35.0, 8.0, 1000.0)
        assert b.volume_attenuation == 'F'

    def test_bounce_validates_F_requires_fg(self):
        from uacpy.models import Bounce
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError):
            Bounce(volume_attenuation='F')
