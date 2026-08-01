"""Bellhop ray/beam-focused tests."""

import pytest
import numpy as np

from uacpy.models import Bellhop
from uacpy import Field
from uacpy.core.results import Rays, Arrivals
from uacpy.models.base import RunMode
from uacpy.core import Environment, Source, Receiver
from uacpy.core.exceptions import ConfigurationError

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

        assert isinstance(result, Field)
        assert result.shape == (len(setup_receiver.depths), len(setup_receiver.ranges))
        assert np.all(np.isfinite(result.data))
        assert np.all(result.tl > 0), "TL should be positive"

    @pytest.mark.requires_binary
    def test_r0_column_is_no_data_nan(self, setup_env, setup_source):
        """Bellhop writes zero pressure at r=0 (no ray travels zero distance);
        the SHD reader surfaces those cells as NaN no-data."""
        rcv = Receiver(depths=np.array([25.0, 50.0, 75.0]),
                       ranges=np.array([0.0, 1000.0, 3000.0]))
        result = Bellhop(verbose=False).run(
            env=setup_env, source=setup_source, receiver=rcv,
            run_mode=RunMode.COHERENT_TL)
        tl = np.asarray(result.tl)
        assert np.all(np.isnan(tl[:, 0]))
        assert np.all(np.isfinite(tl[:, 1:]))

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

        assert isinstance(result, Field)
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

        assert isinstance(result, Field)
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

        assert isinstance(result, Rays)
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

        assert isinstance(result, Rays)
        # Wrapper must mark this as solver-computed eigenrays.
        assert result.is_eigen is True
        assert len(result.rays) > 0

    @pytest.mark.requires_binary
    @pytest.mark.slow
    def test_bellhop_compute_eigenrays(self, setup_env, setup_source):
        """Test Bellhop.compute_eigenrays one-call API."""
        bellhop = Bellhop(verbose=False)

        from uacpy import Receiver
        rcv = Receiver(depths=[50.0], ranges=[3000.0])
        rays = bellhop.compute_eigenrays(setup_env, setup_source, rcv)
        assert isinstance(rays, Rays)
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

        with pytest.raises(ConfigurationError):
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

        assert isinstance(result, Arrivals)
        assert result.by_receiver is not None

    @pytest.mark.requires_binary
    def test_broadband_multi_frequency_source(self, setup_env, setup_receiver):
        """Bellhop BROADBAND with a multi-frequency Source traces rays at the
        band centre and synthesizes H(f) over the band (the arrivals sub-run
        uses a single-frequency source)."""
        bellhop = Bellhop(verbose=False)
        source = Source(depths=50.0, frequencies=np.array([100.0, 200.0, 300.0]))
        result = bellhop.run(
            env=setup_env, source=source, receiver=setup_receiver,
            run_mode=RunMode.BROADBAND,
        )
        assert isinstance(result, Field)
        assert 'frequency' in result.coords
        assert np.iscomplexobj(result.data)


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
        assert isinstance(result, Field)
        assert np.all(np.isfinite(result.data))

    @pytest.mark.requires_binary
    def test_geometric_beam_hat(self, env, source, receiver):
        """Test geometric hat beam (type 'G')."""
        bellhop = Bellhop(verbose=False, beam_type='G')
        result = bellhop.run(env=env, source=source, receiver=receiver)
        assert isinstance(result, Field)
        assert np.all(np.isfinite(result.data))

    @pytest.mark.requires_binary
    def test_simple_gaussian_beam(self, env, source, receiver):
        """Test simple Gaussian beam (type 'S')."""
        bellhop = Bellhop(verbose=False, beam_type='S')
        result = bellhop.run(env=env, source=source, receiver=receiver)
        assert isinstance(result, Field)
        assert np.all(np.isfinite(result.data))

    @pytest.mark.requires_binary
    def test_cartesian_beam(self, env, source, receiver):
        """Test Cartesian beam (type 'C').

        Cerveny-style beams leave cells outside every beam's footprint
        unwritten — those come back as NaN no-data cells, so assert real
        data arrives somewhere rather than everywhere."""
        bellhop = Bellhop(verbose=False, beam_type='C')
        result = bellhop.run(env=env, source=source, receiver=receiver)
        assert isinstance(result, Field)
        assert np.any(np.isfinite(result.data))

    @pytest.mark.requires_binary
    def test_ray_centered_beam(self, env, source, receiver):
        """Test ray-centered beam (type 'R'); no-data cells as for 'C'."""
        bellhop = Bellhop(verbose=False, beam_type='R')
        result = bellhop.run(env=env, source=source, receiver=receiver)
        assert isinstance(result, Field)
        assert np.any(np.isfinite(result.data))

    @pytest.mark.requires_binary
    def test_beam_type_changes_tl(self, env, source, receiver):
        """beam_type must actually reach the solver: different beam models
        give measurably different TL. (Guards against the wrapper silently
        ignoring beam_type — the per-beam smoke tests above would all still
        pass in that case.)"""
        tl_b = Bellhop(verbose=False, beam_type='B').run(
            env=env, source=source, receiver=receiver).tl
        tl_s = Bellhop(verbose=False, beam_type='S').run(
            env=env, source=source, receiver=receiver).tl
        assert not np.allclose(tl_b, tl_s, atol=1e-2), (
            "Gaussian ('B') and simple-Gaussian ('S') beams produced identical "
            "TL — beam_type may not be reaching the Bellhop input."
        )


class TestRunWithBounceConstructorPlumbing:
    """Verify Bellhop.run_with_bounce passes through volume-attenuation /
    c_low / c_high to the spawned Bounce instance."""

    def test_bounce_sees_env_absorption(self, monkeypatch):
        """env.absorption (Francois-Garrison) flows through Bellhop's
        auto-BOUNCE call into the Bounce subprocess."""
        from uacpy.models import bounce as bounce_mod
        from uacpy.core.absorption import FrancoisGarrison

        captured = {}

        def spy_run(self_, env, source, receiver, **kwargs):
            captured['absorption'] = env.absorption
            captured.update(kwargs)
            raise RuntimeError("stop after Bounce.run capture")

        monkeypatch.setattr(bounce_mod.Bounce, 'run', spy_run)

        fg = FrancoisGarrison(
            temperature_c=10.0, salinity_psu=35.0, pH=8.0, z_bar_m=1000.0,
        )
        bellhop = Bellhop(verbose=False)
        env = Environment(
            name='b', bathymetry=100.0, ssp=1500.0, absorption=fg,
        )
        source = Source(depths=50.0, frequencies=100.0)
        receiver = Receiver(depths=[50.0], ranges=[1000.0])
        with pytest.raises(RuntimeError, match='stop after Bounce.run capture'):
            bellhop.run_with_bounce(
                env=env, source=source, receiver=receiver,
                c_low=1450.0, c_high=20000.0, rmax=42000.0,
            )
        assert isinstance(captured.get('absorption'), FrancoisGarrison)
        assert captured['absorption'].temperature_c == 10.0
        assert captured['absorption'].salinity_psu == 35.0


class TestFrancoisGarrisonValidation:
    """FrancoisGarrison validates its own params at construction."""

    def test_francois_garrison_constructs(self):
        from uacpy.core.absorption import FrancoisGarrison
        fg = FrancoisGarrison(
            temperature_c=10.0, salinity_psu=35.0, pH=8.0, z_bar_m=1000.0,
        )
        assert fg.topopt_code() == 'F'
        assert fg.as_at_tuple() == (10.0, 35.0, 8.0, 1000.0)


class TestBellhopRangeDependentSSP:
    """Bellhop with ``interp_ssp='quad'`` reads the ``.ssp`` file emitted
    by uacpy: ``Npts`` on its own line, range vector on the next, depth
    rows after — matching the AT LDIFile record convention."""

    @pytest.fixture
    def rd_ssp_env(self):
        from uacpy.core.environment import SoundSpeedProfile, BoundaryProperties
        z = np.linspace(0.0, 100.0, 21)
        # SSP range must extend past the receiver max range or Bellhop's
        # rays exit the "soundspeed box" with a FATAL ERROR.
        r = np.array([0.0, 4000.0, 8000.0, 12000.0])
        c2d = (1500.0
               + 5.0 * np.sin(np.pi * z[:, None] / 100.0)
               - 5e-5 * r[None, :])
        return Environment(
            name='rd-ssp-e2e',
            bathymetry=100.0,
            ssp=SoundSpeedProfile.from_2d(z, r, c2d),
            bottom=BoundaryProperties(
                acoustic_type='half-space',
                sound_speed=1700.0, density=1.8, attenuation=0.5,
            ),
        )

    @pytest.mark.requires_binary
    @pytest.mark.parametrize('backend', [None, 'fortran'])
    def test_rd_ssp_quad_runs_end_to_end(self, rd_ssp_env, backend):
        src = Source(depths=20.0, frequencies=200.0)
        rcv = Receiver(
            depths=np.linspace(5.0, 95.0, 19),
            ranges=np.linspace(100.0, 8000.0, 41),
        )
        bh = Bellhop(verbose=False, interp_ssp='quad', backend=backend)
        res = bh.run(rd_ssp_env, src, rcv, run_mode=RunMode.COHERENT_TL)
        tl = np.asarray(res.tl)
        # Cells no ray reached (shadow zones) are NaN no-data cells.
        real = tl[np.isfinite(tl)]
        assert real.size > tl.size * 0.5, (
            'most cells should carry real TL values'
        )
        assert real.min() > 0
        assert real.max() < 200

    @pytest.mark.requires_binary
    @pytest.mark.parametrize('backend', [None, 'fortran'])
    def test_rd_ssp_extended_past_box(self, rd_ssp_env, backend):
        """A receiver past the RD-SSP range grid must not crash: the writer
        holds the last profile constant out beyond the ray box (with a
        UserWarning) instead of letting rays exit the soundspeed box."""
        src = Source(depths=20.0, frequencies=200.0)
        # rd_ssp_env defines SSP only to 12 km; receiver at 14 km → box 16.8 km.
        rcv = Receiver(
            depths=np.linspace(5.0, 95.0, 19),
            ranges=np.linspace(100.0, 14000.0, 30),
        )
        bh = Bellhop(verbose=False, interp_ssp='quad', backend=backend)
        # Two warnings fire for this geometry: the writer holds the last SSP
        # profile constant past the box, and validate_inputs flags the same
        # range shortfall. Capture both so neither leaks to the summary.
        with pytest.warns(UserWarning) as record:
            res = bh.run(rd_ssp_env, src, rcv, run_mode=RunMode.COHERENT_TL)
        msgs = [str(w.message) for w in record]
        assert any('range-dependent SSP spans' in m for m in msgs)
        assert any('constant-extrapolated' in m for m in msgs)
        tl = np.asarray(res.tl)
        real = tl[(tl > 0) & (tl < 500)]
        assert real.size > tl.size * 0.5
        assert real.max() < 200


# ---------------------------------------------------------------------
# Bellhop multi-source-depth: result-type dispatch and stack semantics.
# ---------------------------------------------------------------------

class TestBellhopMultiSourceDepth:
    """Bellhop's ``.shd`` carries one slab per source depth. Single-source
    runs return a :class:`Field`; multi-source runs return a
    :class:`ResultStack` of :class:`Field` slabs keyed by
    source depth."""

    @pytest.mark.requires_binary
    def test_multi_source_returns_result_stack(self):
        from uacpy.core.results import ResultStack
        env = Environment(
            name='multi-src-shape', bathymetry=100.0, ssp=1500.0,
        )
        source = Source(depths=[30.0, 50.0, 70.0], frequencies=100.0)
        receiver = Receiver(
            depths=np.linspace(10.0, 90.0, 9),
            ranges=np.linspace(100.0, 5000.0, 11),
        )
        bh = Bellhop(verbose=False)
        result = bh.run(env, source, receiver, run_mode=RunMode.COHERENT_TL)
        assert isinstance(result, ResultStack)
        assert result.slab_type is Field
        np.testing.assert_allclose(
            np.sort(result.coordinate), np.array([30.0, 50.0, 70.0])
        )
        assert result.coordinate_name == 'source_depth'
        assert result.n_slabs == 3
        assert len(result) == 3
        # Each slab is a 2-D Field on the same receiver grid.
        for sd, slab in result:
            assert isinstance(slab, Field)
            assert slab.data.shape == (9, 11)

    @pytest.mark.requires_binary
    def test_single_source_returns_pressure_field(self):
        from uacpy.core.results import ResultStack
        env = Environment(
            name='single-src-shape', bathymetry=100.0, ssp=1500.0,
        )
        source = Source(depths=50.0, frequencies=100.0)
        receiver = Receiver(
            depths=np.linspace(10.0, 90.0, 9),
            ranges=np.linspace(100.0, 5000.0, 11),
        )
        bh = Bellhop(verbose=False)
        result = bh.run(env, source, receiver, run_mode=RunMode.COHERENT_TL)
        assert isinstance(result, Field)
        assert not isinstance(result, ResultStack)
        assert result.data.shape == (9, 11)

    @pytest.mark.requires_binary
    def test_multi_source_per_slab_tl_physically_plausible(self):
        """Each source slab in the stack produces finite,
        physically-sensible TL (positive, < 200 dB)."""
        env = Environment(
            name='multi-src-physics', bathymetry=100.0, ssp=1500.0,
        )
        source = Source(depths=[30.0, 50.0, 70.0], frequencies=100.0)
        receiver = Receiver(
            depths=np.linspace(10.0, 90.0, 9),
            ranges=np.linspace(500.0, 5000.0, 10),
        )
        bh = Bellhop(verbose=False)
        stack = bh.run(env, source, receiver, run_mode=RunMode.COHERENT_TL)
        for sd_value, slab in stack:
            assert slab.data.shape == (9, 10)
            tl = slab.tl
            real = tl[(tl > 0) & (tl < 500)]
            assert real.size > tl.size * 0.5
            assert real.min() > 0
            assert real.max() < 200

    @pytest.mark.requires_binary
    def test_at_source_depth_recovers_middle_slab(self):
        """``stack.at(source_depth=z)`` returns the matching
        :class:`Field` slab, and that slab's TL matches a
        single-source run at the same depth (within Bellhop's beam-
        partitioning round-off)."""
        env = Environment(
            name='multi-src-at', bathymetry=100.0, ssp=1500.0,
        )
        receiver = Receiver(
            depths=np.linspace(10.0, 90.0, 9),
            ranges=np.linspace(500.0, 5000.0, 10),
        )
        bh = Bellhop(verbose=False)
        stack = bh.run(
            env,
            Source(depths=[30.0, 50.0, 70.0], frequencies=100.0),
            receiver, run_mode=RunMode.COHERENT_TL,
        )
        slab = stack.at(source_depth=50.0)
        assert isinstance(slab, Field)
        assert slab.data.shape == (9, 10)

        single = bh.run(
            env, Source(depths=50.0, frequencies=100.0),
            receiver, run_mode=RunMode.COHERENT_TL,
        )
        np.testing.assert_allclose(slab.tl, single.tl, rtol=1e-4, atol=1e-3)

    @pytest.mark.requires_binary
    def test_multi_source_rays_returns_stack(self):
        """RAYS mode: one binary call, reader splits the deterministic
        ``NSz × Nalpha`` block layout into one :class:`Rays` slab per
        source depth."""
        from uacpy.core.results import Rays, ResultStack
        env = Environment(name='multi-rays', bathymetry=100.0, ssp=1500.0)
        receiver = Receiver(depths=np.array([30.0, 60.0]),
                            ranges=np.array([200.0, 1000.0]))
        bh = Bellhop(verbose=False, n_beams=20, alpha=(-30, 30))
        stack = bh.run(
            env, Source(depths=[20.0, 50.0, 80.0], frequencies=100.0),
            receiver, run_mode=RunMode.RAYS,
        )
        assert isinstance(stack, ResultStack)
        assert stack.slab_type is Rays
        assert stack.n_slabs == 3
        np.testing.assert_allclose(stack.coordinate,
                                   np.array([20.0, 50.0, 80.0]))
        # Every slab carries the same Nalpha = 20 rays.
        for sd, slab in stack:
            assert len(slab.rays) == 20
            assert slab.is_eigen is False

    @pytest.mark.requires_binary
    def test_multi_source_arrivals_returns_stack(self):
        """ARRIVALS mode: one binary call, reader splits the
        ``by_receiver[isd]`` axis it already parses."""
        from uacpy.core.results import Arrivals, ResultStack
        env = Environment(name='multi-arr', bathymetry=100.0, ssp=1500.0)
        receiver = Receiver(depths=np.array([30.0, 60.0]),
                            ranges=np.array([500.0, 1000.0]))
        bh = Bellhop(verbose=False, n_beams=20, alpha=(-30, 30))
        stack = bh.run(
            env, Source(depths=[20.0, 50.0, 80.0], frequencies=100.0),
            receiver, run_mode=RunMode.ARRIVALS,
        )
        assert isinstance(stack, ResultStack)
        assert stack.slab_type is Arrivals
        assert stack.n_slabs == 3
        np.testing.assert_allclose(stack.coordinate,
                                   np.array([20.0, 50.0, 80.0]))
        # Each slab's by_receiver is shaped (1, n_rd, n_rr) — one
        # source-depth axis, two receiver depths, two ranges.
        for sd, slab in stack:
            assert len(slab.by_receiver) == 1
            assert len(slab.by_receiver[0]) == 2
            assert len(slab.by_receiver[0][0]) == 2

    @pytest.mark.requires_binary
    def test_multi_source_eigenrays_returns_stack(self):
        """EIGENRAYS mode: Bellhop reorders α for its bracketing
        heuristic, so the ``.ray`` file isn't splittable per source.
        The wrapper loops in Python — N binary calls — and bundles the
        result into a :class:`ResultStack` with the right slab type."""
        from uacpy.core.results import Rays, ResultStack
        env = Environment(name='multi-eig', bathymetry=100.0, ssp=1500.0)
        receiver = Receiver(depths=50.0, ranges=1000.0)
        bh = Bellhop(verbose=False, n_beams=20, alpha=(-30, 30))
        stack = bh.run(
            env, Source(depths=[20.0, 50.0, 80.0], frequencies=100.0),
            receiver, run_mode=RunMode.EIGENRAYS,
        )
        assert isinstance(stack, ResultStack)
        assert stack.slab_type is Rays
        assert stack.n_slabs == 3
        np.testing.assert_allclose(stack.coordinate,
                                   np.array([20.0, 50.0, 80.0]))
        for sd, slab in stack:
            assert slab.is_eigen is True


def test_bellhop_backend_fallback_warns(monkeypatch):
    """An explicitly requested backend that isn't built must fall back to the
    Fortran binary with a ``UserWarning`` (not a hard error). Verified by
    hiding the cxx/cuda variants so only ``bellhop`` resolves."""
    from uacpy.core.exceptions import ExecutableNotFoundError
    fortran = Bellhop(backend='fortran', verbose=False)._exe   # real bellhop path

    def fake_find(self, names, **kw):
        names = list(names) if isinstance(names, (list, tuple)) else [names]
        if 'bellhop' in names:
            return fortran
        raise ExecutableNotFoundError('Bellhop', repr(names))

    monkeypatch.setattr(Bellhop, '_find_executable_in_paths', fake_find)
    with pytest.warns(UserWarning, match='falling back'):
        bh = Bellhop(backend='cuda', verbose=False)
    assert bh.version == 'fortran'
    assert bh.backend == 'cuda'                # requested value retained for copy()


def test_bellhop_compute_arrivals_and_transfer_function():
    """Smoke-cover the ``compute_arrivals`` / ``compute_transfer_function``
    convenience wrappers (capability-check + kwarg-forwarding layer), which
    are otherwise only reached via ``run(run_mode=...)``."""
    env = Environment(name='cf', bathymetry=200.0, ssp=1500.0)
    src = Source(depths=50.0, frequencies=200.0)
    rcv = Receiver(depths=np.array([60.0, 120.0]),
                   ranges=np.array([2000.0, 5000.0]))
    bh = Bellhop(verbose=False)
    arr = bh.compute_arrivals(env, src, rcv)
    assert isinstance(arr, Arrivals)
    hf = bh.compute_transfer_function(env, src, rcv,
                                      frequencies=np.linspace(180.0, 220.0, 5))
    assert isinstance(hf, Field) and np.iscomplexobj(hf.data)


class TestConstructorValidation:
    """Binary-affecting ctor args are validated up front (audit H2 + the
    related ctor-validation cross-cutting theme): a bad value must raise a
    ``ConfigurationError`` rather than silently mis-drive the binary."""

    def test_dimensionality_3d_raises(self):
        # '3D' would emit --3D against a 2D-only env file (silent 2D on
        # Fortran, abort on cxx/cuda).
        with pytest.raises(ConfigurationError):
            Bellhop(dimensionality='3D')

    def test_dimensionality_arbitrary_raises(self):
        with pytest.raises(ConfigurationError):
            Bellhop(dimensionality='foo')

    def test_dimensionality_2d_ok(self):
        assert Bellhop(dimensionality='2D').dimensionality == '2D'

    def test_invalid_beam_type_raises(self):
        # Unknown letters silently map to geometric-hat in the Fortran reader.
        with pytest.raises(ConfigurationError):
            Bellhop(beam_type='Q')

    @pytest.mark.parametrize('bt', ['B', 'R', 'C', 'b', 'g', 'G', 'S'])
    def test_valid_beam_types_ok(self, bt):
        assert Bellhop(beam_type=bt).beam_type == bt

    def test_invalid_grid_type_raises(self):
        with pytest.raises(ConfigurationError):
            Bellhop(grid_type='Q')

    def test_alpha_wrong_length_raises(self):
        with pytest.raises(ConfigurationError):
            Bellhop(alpha=(-80, 0, 80))

    def test_alpha_scalar_raises(self):
        with pytest.raises(ConfigurationError):
            Bellhop(alpha=80)


class TestBellhopSourceGeometry:
    """Source geometry is read off the Source carrier (spec 2026-07-25)."""

    @staticmethod
    def _env():
        return Environment(bathymetry=200.0, ssp=1500.0)

    def test_constructor_no_longer_accepts_source_type(self):
        with pytest.raises(TypeError):
            Bellhop(source_type='R')

    def test_constructor_no_longer_accepts_beam_pattern_file(self):
        with pytest.raises(TypeError):
            Bellhop(source_beam_pattern_file=None)

    def test_line_source_differs_from_point_source(self):
        # The finding this refactor exists for: identical before the fix.
        env = self._env()
        rcv = Receiver(depths=100.0, ranges=np.linspace(100, 5000, 60))
        model = Bellhop(verbose=False)
        pt = model.run(env, Source(depths=50, frequencies=200,
                                   source_type='point'), rcv)
        ln = model.run(env, Source(depths=50, frequencies=200,
                                   source_type='line'), rcv)
        delta = np.nanmax(np.abs(np.asarray(pt.tl) - np.asarray(ln.tl)))
        assert delta > 10.0, f"source_type is still inert (max dTL={delta})"

    def test_line_vs_point_matches_influence_f90_ratio(self):
        # influence.f90:783 — line: factor = -4*sqrt(pi)*const;
        # point: factor = const/sqrt(r). The 1/sqrt(r) is the only
        # range-dependent term, so the slope of TL_point - TL_line over
        # range is 10*log10(r2/r1).
        env = self._env()
        ranges = np.array([1000.0, 2000.0, 4000.0])
        rcv = Receiver(depths=100.0, ranges=ranges)
        model = Bellhop(verbose=False)
        pt = np.asarray(model.run(env, Source(depths=50, frequencies=200,
                                              source_type='point'), rcv).tl).ravel()
        ln = np.asarray(model.run(env, Source(depths=50, frequencies=200,
                                              source_type='line'), rcv).tl).ravel()
        diff = pt - ln
        measured = diff[-1] - diff[0]
        expected = 10 * np.log10(ranges[-1] / ranges[0])
        assert measured == pytest.approx(expected, abs=1.0)

    def test_beam_pattern_array_writes_sbp(self, tmp_path):
        env = self._env()
        rcv = Receiver(depths=100.0, ranges=np.linspace(100, 2000, 20))
        pat = np.array([[-90.0, -30.0], [0.0, 0.0], [90.0, -30.0]])
        model = Bellhop(verbose=False, work_dir=tmp_path, cleanup=False)
        model.run(env, Source(depths=50, frequencies=200,
                              beam_pattern=pat), rcv)
        sbp = list(tmp_path.rglob('*.sbp'))
        assert sbp, "no .sbp written for Source(beam_pattern=...)"
        assert sbp[0].read_text().split()[0] == '3'


class TestBeamPatternReachesEveryPath:
    """``Source.beam_pattern`` must survive every internal ``Source`` rebuild.

    The broadband paths and multi-depth EIGENRAYS construct their own Source
    from the caller's; dropping the pattern there launches an omnidirectional
    fan while ``validate_inputs`` still accepts the request.
    """

    PATTERN = np.array([[-180.0, -100.0], [-5.0, -100.0],
                        [-2.0, 0.0], [2.0, 0.0],
                        [5.0, -100.0], [180.0, -100.0]])

    @staticmethod
    def _env():
        return Environment(bathymetry=200.0, ssp=1500.0)

    @staticmethod
    def _rcv():
        return Receiver(depths=np.array([100.0]),
                        ranges=np.linspace(500, 2000, 4))

    def _run(self, tmp_path, run_mode, **kwargs):
        model = Bellhop(verbose=False, work_dir=tmp_path, cleanup=False)
        src = Source(depths=50.0, frequencies=200.0,
                     beam_pattern=self.PATTERN)
        result = model.run(self._env(), src, self._rcv(),
                           run_mode=run_mode, **kwargs)
        sbp = list(tmp_path.rglob('*.sbp'))
        assert sbp, f"no .sbp staged for run_mode={run_mode}"
        return result

    def test_broadband_stages_sbp(self, tmp_path):
        self._run(tmp_path, RunMode.BROADBAND)

    def test_time_series_stages_sbp(self, tmp_path):
        self._run(
            tmp_path, RunMode.TIME_SERIES,
            source_waveform=np.sin(2 * np.pi * 200 * np.arange(256) / 4000.0),
            sample_rate=4000.0,
        )

    def test_multi_depth_eigenrays_stages_sbp(self, tmp_path):
        model = Bellhop(verbose=False, work_dir=tmp_path, cleanup=False)
        src = Source(depths=[30.0, 60.0], frequencies=200.0,
                     beam_pattern=self.PATTERN)
        model.run(self._env(), src,
                  Receiver(depths=[100.0], ranges=[1500.0]),
                  run_mode=RunMode.EIGENRAYS)
        assert list(tmp_path.rglob('*.sbp')), \
            "no .sbp staged for multi-depth EIGENRAYS"

    def test_broadband_pattern_changes_the_field(self, tmp_path):
        """The pattern must reach the physics, not just the file: a ±2°
        aperture cannot produce the omnidirectional transfer function."""
        model = Bellhop(verbose=False)
        env, rcv = self._env(), self._rcv()
        plain = model.run(env, Source(depths=50.0, frequencies=200.0), rcv,
                          run_mode=RunMode.BROADBAND)
        shaded = model.run(
            env,
            Source(depths=50.0, frequencies=200.0, beam_pattern=self.PATTERN),
            rcv, run_mode=RunMode.BROADBAND,
        )
        assert not np.allclose(np.abs(plain.data), np.abs(shaded.data)), \
            "beam_pattern did not change BROADBAND H(f)"


class TestAutoBounceWithBeamPattern:
    """``Bellhop(auto_bounce=True)`` + a beam-pattern Source must not raise
    from inside the spawned Bounce: BOUNCE reads no source geometry, and
    Bellhop — the model the user called — supports beam patterns."""

    def test_elastic_bottom_auto_route_accepts_beam_pattern(self):
        from uacpy.core import BoundaryProperties
        env = Environment(
            name='elastic', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(
                acoustic_type='half-space', sound_speed=1600.0, density=1.8,
                attenuation=0.2, shear_speed=400.0, shear_attenuation=0.5,
            ),
        )
        src = Source(depths=50.0, frequencies=100.0,
                     beam_pattern=np.array([[-180.0, -20.0], [0.0, 0.0],
                                            [180.0, -20.0]]))
        rcv = Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))
        result = Bellhop(verbose=False).run(env, src, rcv)
        assert 'bounce_result' in result.metadata

    def test_bounce_accepts_beam_pattern_source_directly(self):
        """The guard lives in ``_validate_geometry``, which Bounce no-ops —
        so a Source can be reused across models without stripping it."""
        from uacpy.models import Bounce
        env = Environment(name='e', bathymetry=100.0, ssp=1500.0)
        src = Source(depths=50.0, frequencies=100.0,
                     beam_pattern=np.array([[-90.0, -10.0], [90.0, -10.0]]))
        rcv = Receiver(depths=np.array([50.0]), ranges=np.array([1000.0]))
        Bounce(verbose=False).validate_inputs(
            env, src, rcv, run_mode=RunMode.REFLECTION)
