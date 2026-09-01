"""Bellhop ray/beam-focused tests.

Run modes, beam types and source geometry, the env-record ordering the binary
demands, and the validity floor below which ray theory stops describing the
field at all.

Everything in this file is marked ``requires_binary``: constructing ``Bellhop``
resolves its executable, so even the tests that never launch a run need the
install to be present.
"""

import warnings

import pytest
import numpy as np

import uacpy
from uacpy.models import Bellhop
from uacpy import Field
from uacpy.core.results import Rays, Arrivals
from uacpy.models.base import RunMode
from uacpy.models.bellhop import (_RAY_VALIDITY_D_OVER_LAMBDA,
                                  _WARNED_RAY_VALIDITY)
from uacpy.core import (
    Environment, Source, Receiver, BoundaryProperties,
)
from uacpy.core.exceptions import (
    ConfigurationError, UnsupportedFeatureError,
)

pytestmark = pytest.mark.requires_binary

_D_OVER_LAMBDA = 'D/lambda'


def _messages(fn, needle):
    """Run ``fn`` and return the warning messages containing ``needle``."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        fn()
    return [str(w.message) for w in rec if needle in str(w.message)]


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
        assert np.all(result.db > 0), "TL should be positive"

    @pytest.mark.requires_binary
    def test_r0_column_is_no_data_nan(self, setup_env, setup_source):
        """``Bellhop/influence.f90:786-787`` short-circuits the cylindrical
        spreading factor to 0 at ``r = 0`` to avoid dividing by zero, so the
        binary writes exact zero pressure there. The SHD reader surfaces those
        cells as NaN no-data rather than the -inf dB a literal reading gives."""
        rcv = Receiver(depths=np.array([25.0, 50.0, 75.0]),
                       ranges=np.array([0.0, 1000.0, 3000.0]))
        result = Bellhop(verbose=False).run(
            env=setup_env, source=setup_source, receiver=rcv,
            run_mode=RunMode.COHERENT_TL)
        tl = np.asarray(result.db)
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
        # AT parks the incoherent magnitude sum in the complex .shd slot, so
        # the payload stays complex with an identically zero imaginary part
        # (docs/guide/results.md §9 "An incoherent field has no phase",
        # DOCUMENTATION.md §7 "its phase an artefact of AT's storage") — the
        # phase carries no information and .db is the cross-engine surface.
        assert np.iscomplexobj(result.data)
        assert np.all(np.imag(result.data) == 0.0)

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
        # 3 km to a single point with the auto fan: the eigenray bracketing
        # lands the best ray within metres of the receiver. Misses of tens of
        # metres are the signature of a fan that never refined, so bound the
        # closest ray well below that scale.
        # miss_distance_m is attached by the filter helper (target defaults
        # to the run's receiver point), not by the reader itself.
        close = result.filter_by_miss_distance(50.0)
        assert len(close.rays) > 0
        assert min(r['miss_distance_m'] for r in close.rays) < 50.0

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
        cell = result.by_receiver[0][0][0]
        assert int(cell['n_arrivals']) > 0
        # 100 m of isovelocity 1500 m/s water with source and receiver both
        # at 50 m: the direct path to r = 3000 m is horizontal, so the
        # earliest arrival is 3000/1500 = 2.0 s exactly, and the one-bounce
        # paths (sqrt(3000^2 + 100^2)/1500) trail it by only ~1 ms. 10 ms
        # absorbs the beam-window delay spread while still catching a wrong
        # sound speed or a seconds/milliseconds units error outright.
        first = float(np.min(np.asarray(cell['delays'], dtype=float)))
        assert first == pytest.approx(3000.0 / 1500.0, abs=0.01)

    @pytest.mark.requires_binary
    @pytest.mark.parametrize('grid_type,n_depth_blocks', [('R', 3), ('I', 1)])
    def test_arrivals_honour_the_receiver_grid_type(self, setup_env,
                                                    setup_source, grid_type,
                                                    n_depth_blocks):
        """The ``.arr`` header reports the full ``Pos%NRz``
        (``ReadEnvironmentBell.f90:591``) but its body carries only
        ``NRz_per_range`` depth blocks — ``Pos%NRz`` for ``'R'`` and 1 for
        ``'I'`` (``bellhop.f90:202-206,329``, ``ArrMod.f90:101-102``). Nothing
        in the file distinguishes them, so the run has to tell the reader
        which it wrote; unpassed, an irregular run cannot be parsed at all."""
        receiver = Receiver(depths=[50.0, 100.0, 150.0],
                            ranges=[1000.0, 2000.0, 3000.0])
        result = Bellhop(verbose=False, grid_type=grid_type).run(
            env=setup_env, source=setup_source, receiver=receiver,
            run_mode=RunMode.ARRIVALS)

        assert isinstance(result, Arrivals)
        assert len(result.by_receiver[0]) == n_depth_blocks
        assert len(result.by_receiver[0][0]) == 3

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
        # A multi-element source IS the band, verbatim — no expansion, no
        # resampling (base.py _resolve_broadband_frequencies): 3 depths x
        # 3 ranges x exactly the 3 requested bins.
        assert result.data.shape == (3, 3, 3)
        np.testing.assert_array_equal(
            np.asarray(result.coords['frequency'], dtype=float),
            [100.0, 200.0, 300.0])


class TestBroadbandAttenuationWarningNamesTheTracedCarrier:
    """The volume-attenuation warning is anchored at the carrier the single
    ray trace actually ran at — the source-derived fc — not the resolved
    band's midpoint: an explicit off-centre ``frequencies=`` band moves the
    midpoint away from the traced carrier, and a bound anchored there
    understates the true band-edge error."""

    def test_off_centre_explicit_band_reports_the_traced_fc_bound(self):
        from uacpy.core.absorption import Thorp
        env = Environment(
            name='thorp', bathymetry=100.0, ssp=1500.0, absorption=Thorp(),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.8,
                                      attenuation=0.5))
        source = Source(depths=50.0, frequencies=10000.0)
        receiver = Receiver(depths=[50.0], ranges=[200.0])
        # Traced carrier 10 kHz; band midpoint 12 kHz. Anchored at the
        # midpoint the Thorp bound reads 0.577 dB/km; at the traced
        # carrier it is 0.872 dB/km.
        with pytest.warns(UserWarning,
                          match=r'traced once at 1e\+04 Hz'
                                r'(?s:.*)0\.872 dB/km'):
            Bellhop(verbose=False).run(
                env, source, receiver, run_mode=RunMode.BROADBAND,
                frequencies=np.array([8000.0, 16000.0]))


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
        tl = np.asarray(res.db)
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
        tl = np.asarray(res.db)
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
            tl = slab.db
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
        np.testing.assert_allclose(slab.db, single.db, rtol=1e-4, atol=1e-3)

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
    related ctor-validation cross-cutting theme): a bad value must raise
    rather than silently mis-drive the binary. Which type it raises follows
    the split in ``core/exceptions``: an unrecognised argument is a
    ``ConfigurationError``, a recognised one this wrapper cannot produce is
    an ``UnsupportedFeatureError``."""

    def test_dimensionality_3d_raises(self):
        # '3D' would emit --3D against a 2D-only env file (silent 2D on
        # Fortran, abort on cxx/cuda).
        with pytest.raises(UnsupportedFeatureError):
            Bellhop(dimensionality='3D')

    def test_dimensionality_arbitrary_raises(self):
        with pytest.raises(UnsupportedFeatureError):
            Bellhop(dimensionality='foo')

    def test_dimensionality_2d_ok(self):
        assert Bellhop(dimensionality='2D').dimensionality == '2D'

    def test_invalid_beam_type_raises(self):
        # Unknown letters silently map to geometric-hat in the Fortran reader.
        with pytest.raises(ConfigurationError):
            Bellhop(beam_type='Q')

    @pytest.mark.parametrize('bt', ['B', 'R', 'C', 'g', 'G', 'S'])
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

    def test_constructor_rejects_source_type(self):
        with pytest.raises(TypeError):
            Bellhop(source_type='R')

    def test_constructor_rejects_beam_pattern_file(self):
        with pytest.raises(TypeError):
            Bellhop(source_beam_pattern_file=None)

    def test_line_source_differs_from_point_source(self):
        env = self._env()
        rcv = Receiver(depths=100.0, ranges=np.linspace(100, 5000, 60))
        model = Bellhop(verbose=False)
        pt = model.run(env, Source(depths=50, frequencies=200,
                                   source_type='point'), rcv)
        ln = model.run(env, Source(depths=50, frequencies=200,
                                   source_type='line'), rcv)
        delta = np.nanmax(np.abs(np.asarray(pt.db) - np.asarray(ln.db)))
        assert delta > 10.0, f"source_type is still inert (max dTL={delta})"

    def test_line_vs_point_matches_influence_f90_ratio(self):
        # Bellhop/influence.f90:783-790 — line: factor = -4*sqrt(pi)*const;
        # point: factor = const/sqrt(r). That 1/sqrt(r) is the only
        # range-dependent term in ScalePressure, so the slope of
        # TL_point - TL_line over range is 10*log10(r2/r1).
        #
        # The two fields are not otherwise identical: RunType(4:4)=='R' also
        # applies a per-ray Ratio1 = sqrt(|cos(alpha)|) at
        # Bellhop/influence.f90:52,185,321,423,534, so the beam sums differ
        # before scaling and the residual drifts as the dominant ray family
        # changes with range. abs=1.0 dB covers that drift; the 10*log10 term
        # itself spans 6 dB over this range span.
        env = self._env()
        ranges = np.array([1000.0, 2000.0, 4000.0])
        rcv = Receiver(depths=100.0, ranges=ranges)
        model = Bellhop(verbose=False)
        pt = np.asarray(model.run(env, Source(depths=50, frequencies=200,
                                              source_type='point'), rcv).db).ravel()
        ln = np.asarray(model.run(env, Source(depths=50, frequencies=200,
                                              source_type='line'), rcv).db).ravel()
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
        lines = sbp[0].read_text().splitlines()
        assert lines[0].split()[0] == '3'
        # The rows are the pattern verbatim (angle, dB re peak) — refl_io
        # writes both columns at %.6f, so 5e-7 is pure format rounding.
        rows = np.array([[float(v) for v in ln.split()] for ln in lines[1:4]])
        np.testing.assert_allclose(rows, pat, rtol=0, atol=5e-7)


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

    def test_layered_bottom_auto_route_accepts_beam_pattern(self):
        from uacpy.core import BoundaryProperties
        from uacpy.core.bottom import Bottom, SeabedColumn, SedimentLayer
        env = Environment(
            name='layered', bathymetry=100.0, ssp=1500.0,
            bottom=Bottom(columns=[SeabedColumn(
                layers=[SedimentLayer(thickness=5.0, sound_speed=1550.0,
                                      density=1.4, attenuation=0.2)],
                halfspace=BoundaryProperties(
                    acoustic_type='half-space', sound_speed=1600.0,
                    density=1.8, attenuation=0.2, shear_speed=400.0,
                    shear_attenuation=0.5))]),
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


@pytest.mark.requires_binary
def test_time_series_every_range_cell_carries_energy():
    """One clock is locked for the whole receiver grid.

    A window sized from a single cell covers only that cell's delays, so every
    farther receiver convolves to EXACTLY zero — on this 6-range grid that
    silences 5 of the 6 cells. Zero energy, not merely small energy, is the
    signature, which is why the assertion is ``> 0`` rather than a threshold.
    """
    from uacpy.core.environment import Environment
    from uacpy.core.source import Source
    from uacpy.core.receiver import Receiver
    env = Environment(name='ts', bathymetry=200.0, ssp=1500.0)
    src = Source(depths=50.0, frequencies=500.0)
    rcv = Receiver(depths=np.array([100.0]),
                   ranges=np.linspace(1000., 8000., 6))
    fs = 20000.0
    wf = np.sin(2 * np.pi * 500 * np.arange(400) / fs) * np.hanning(400)
    ts = Bellhop(verbose=False).run(env, src, rcv,
                                    run_mode=RunMode.TIME_SERIES,
                                    source_waveform=wf, sample_rate=fs)
    energy = (np.asarray(ts.data) ** 2).sum(axis=-1).ravel()
    assert np.all(energy > 0.0), f"silent range cells: {np.where(energy == 0)[0]}"
    # energy must also fall with range, not merely be non-zero
    assert energy[0] > energy[-1]


class TestEchoesLandAtTheDelayTheyWereGiven:
    """COA eq. 8.30 places each echo at ``S[t - tau]`` — a CONTINUOUS shift.

    Rounding tau to the nearest sample is a numerical convenience the
    formulation does not license: it moves every echo by up to half a
    sample, which is a phase error of tens of degrees at any carrier a
    modem would use, and ``delayandsum`` sums the echoes coherently, so
    that error lands in the interference pattern.
    """

    FS = 200000.0
    FC = 20000.0

    def _cell(self, delay):
        return dict(n_arrivals=1,
                    amplitudes=np.array([1.0]),
                    phases=np.array([0.0]),
                    delays=np.array([float(delay)]),
                    delays_imag=np.array([0.0]),
                    n_top_bounces=np.array([0]),
                    n_bot_bounces=np.array([0]),
                    src_angles=np.array([0.0]),
                    rcv_angles=np.array([0.0]))

    @staticmethod
    def _waveform(fs, fc, n=256):
        k = np.arange(n)
        return np.hanning(n) * np.sin(2 * np.pi * fc * k / fs)

    def _trace(self, delay, **kw):
        from uacpy.models.bellhop import delayandsum
        rts, _ = delayandsum(
            rcv_arrivals=self._cell(delay),
            source_timeseries=self._waveform(self.FS, self.FC),
            sample_rate=self.FS, fc=self.FC,
            t_start=0.0, time_window=0.02, **kw)
        return rts

    def test_a_half_sample_echo_is_not_snapped_onto_a_sample(self):
        """An echo half a sample late must land between the two samples, not
        on one of them. Rounding makes the half-sample trace bit-identical
        to a whole-sample one, which is how the error hides."""
        dt = 1.0 / self.FS
        base = 1000 * dt
        half = self._trace(base + 0.5 * dt)
        assert not np.allclose(half, self._trace(base), atol=1e-12), (
            "the half-sample echo was snapped back onto sample 1000")
        assert not np.allclose(half, self._trace(base + dt), atol=1e-12), (
            "the half-sample echo was snapped forward onto sample 1001")

    def test_the_echo_arrives_where_it_was_told_to(self):
        """Measured as the energy centroid of the trace, which reads the
        group delay whatever the waveform. Rounding quantises it to the
        sample grid; the requested delay does not sit on that grid."""
        dt = 1.0 / self.FS
        for frac in (0.25, 0.5, 0.75):
            want = (1000 + frac) * dt
            rts = self._trace(want)
            power = np.abs(rts) ** 2
            centroid = float((power * np.arange(power.size)).sum()
                             / power.sum()) * dt
            reference = self._trace(1000 * dt)
            p0 = np.abs(reference) ** 2
            c0 = float((p0 * np.arange(p0.size)).sum() / p0.sum()) * dt
            shift_samples = (centroid - c0) / dt
            assert shift_samples == pytest.approx(frac, abs=0.02), (
                f"asked for +{frac} samples, the trace moved "
                f"{shift_samples:.3f}")

    def test_nearest_sample_placement_stays_available(self):
        """The old behaviour is a keyword, not a removal: a caller who wants
        every echo on a sample boundary can still have it. 0.6 rather than
        0.5 of a sample, because numpy rounds a half to the nearest EVEN
        sample, so 1000.5 snaps down to 1000 and would not show the move."""
        dt = 1.0 / self.FS
        snapped = self._trace(1000.6 * dt, fractional=False)
        assert np.allclose(snapped, self._trace(1001 * dt, fractional=False),
                           atol=1e-12)
        assert not np.allclose(snapped, self._trace(1000 * dt,
                                                   fractional=False),
                               atol=1e-12)


class TestVolumeAttenuationFromImaginaryDelay:
    """``Im(tau)`` is the only carrier of Bellhop's volume attenuation.

    ``ArrMod.f90:118-125`` writes ``REAL(delay)`` and ``AIMAG(delay)`` as
    separate fields and the amplitude column does **not** include the loss;
    ``delayandsum.m:134`` applies ``exp(omega*Im(tau))`` as its own factor.
    Dropping it returns a silently lossless field — -23 dB at 20 kHz.
    """

    TAU_I = -1.83e-5          # s; -20.0 dB at 20 kHz
    FC = 20000.0

    def _cell(self):
        return dict(n_arrivals=1,
                    amplitudes=np.array([1.0]),
                    phases=np.array([0.0]),
                    delays=np.array([1.0]),
                    delays_imag=np.array([self.TAU_I]),
                    n_top_bounces=np.array([0]),
                    n_bot_bounces=np.array([0]),
                    src_angles=np.array([0.0]),
                    rcv_angles=np.array([0.0]))

    def test_transfer_function_applies_it(self):
        H = Bellhop(verbose=False)._arrivals_to_tf(
            self._cell(), np.array([self.FC]))
        expected = np.exp(2 * np.pi * self.FC * self.TAU_I)
        assert np.abs(np.asarray(H).ravel()[0]) == pytest.approx(expected,
                                                                 rel=1e-9)
        assert 20 * np.log10(expected) < -19.0

    def test_delay_and_sum_applies_it(self):
        from uacpy.models.bellhop import delayandsum
        fs = 200000.0
        wf = np.sin(2 * np.pi * self.FC * np.arange(64) / fs)
        loud, _ = delayandsum(rcv_arrivals=self._cell(), source_timeseries=wf,
                              sample_rate=fs, fc=self.FC)
        lossless = dict(self._cell(), delays_imag=np.array([0.0]))
        ref, _ = delayandsum(rcv_arrivals=lossless, source_timeseries=wf,
                             sample_rate=fs, fc=self.FC)
        ratio = np.max(np.abs(loud)) / np.max(np.abs(ref))
        assert ratio == pytest.approx(np.exp(2 * np.pi * self.FC * self.TAU_I),
                                      rel=1e-6)


class TestEnvRecordOrder:
    """``.env`` record order against ``ReadEnvironmentBell.f90``.

    ``:59 CALL ReadTopOpt`` reads the Francois-Garrison / biological rows
    itself (``:308-320``); the top half-space row is read only afterwards by
    ``:69 CALL TopBot`` (``:474``). Emitting them the other way round feeds the
    F-G row to the half-space reader and kills the run in ``AttenMod : CRCI``.
    """

    @staticmethod
    def _ice_env():
        from uacpy.core.absorption import FrancoisGarrison
        from uacpy.core.environment import BoundaryProperties
        return Environment(
            name='ice-fg', bathymetry=100.0, ssp=1500.0,
            surface=BoundaryProperties(
                acoustic_type='half-space', sound_speed=3500.0, density=0.9,
                attenuation=0.1, shear_speed=1800.0, shear_attenuation=0.2),
            absorption=FrancoisGarrison(temperature_c=4.0, salinity_psu=34.0,
                                        pH=8.0, z_bar_m=50.0),
        )

    def test_absorption_block_precedes_top_halfspace(self, tmp_path):
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        path = tmp_path / 'order.env'
        write_bellhop_env_file(
            path, self._ice_env(), Source(depths=25.0, frequencies=100.0),
            Receiver(depths=np.array([50.0]),
                     ranges=np.linspace(100.0, 5000.0, 10)))
        lines = path.read_text().splitlines()
        topopt = next(i for i, ln in enumerate(lines) if ln.startswith("'CAWF"))
        # F-G record (T S pH z_bar) first, then the elastic half-space row.
        assert lines[topopt + 1].split() == ['4.0000', '34.0000', '8.0000',
                                             '50.0000']
        assert lines[topopt + 2].split()[:3] == ['0.00', '3500.000000',
                                                 '1800.000000']

    @pytest.mark.requires_binary
    def test_ice_surface_with_francois_garrison_runs(self, tmp_path):
        model = Bellhop(verbose=False, backend='fortran',
                        work_dir=tmp_path, cleanup=False)
        result = model.run(
            self._ice_env(), Source(depths=25.0, frequencies=100.0),
            Receiver(depths=np.array([50.0]),
                     ranges=np.linspace(100.0, 5000.0, 10)),
            run_mode=RunMode.COHERENT_TL)
        prt = next(iter(tmp_path.rglob('*.prt'))).read_text()
        assert '*** FATAL ERROR ***' not in prt
        assert 'CRCI' not in prt
        assert np.any(np.isfinite(np.asarray(result.db)))


class TestQuadSSPMatrixAlignment:
    """``Bellhop/sspMod.f90:427-431`` pairs row ``iz2`` of the ``.ssp`` matrix with
    ``SSP%z(iz2)`` from the ``.env`` and stops after ``SSP%NPts`` rows, so a
    matrix built from the raw profile silently mis-assigns sound speeds once
    the ``.env`` block is truncated or extended to the medium depth."""

    Z = np.array([0.0, 50.0, 100.0, 200.0, 500.0, 1000.0])
    C = np.array([1500.0, 1495.0, 1490.0, 1492.0, 1495.0, 1500.0])
    DEPTH = 150.0

    @classmethod
    def _env(cls):
        from uacpy.core.ssp import SoundSpeedProfile
        data = np.column_stack([cls.C, cls.C + 1.0, cls.C + 2.0])
        return Environment(
            name='quad-align', bathymetry=cls.DEPTH,
            ssp=SoundSpeedProfile(depths=cls.Z, data=data,
                                  ranges=np.array([0.0, 10000.0, 20000.0])))

    @staticmethod
    def _env_ssp_block(path):
        """Return ``(n_pts, depths, speeds)`` from the ``.env`` SSP block."""
        lines = path.read_text().splitlines()
        header = next(i for i, ln in enumerate(lines) if ln.rstrip().endswith(','))
        n_pts = int(lines[header].split()[0])
        rows = [ln.split() for ln in lines[header + 1:header + 1 + n_pts]]
        return (n_pts,
                np.array([float(r[0]) for r in rows]),
                np.array([float(r[1]) for r in rows]))

    def _write(self, tmp_path):
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        path = tmp_path / 'quad.env'
        write_bellhop_env_file(
            path, self._env(), Source(depths=25.0, frequencies=100.0),
            Receiver(depths=np.array([50.0]),
                     ranges=np.linspace(1000.0, 10000.0, 10)),
            interp_ssp='quad')
        return path

    def test_ssp_matrix_rows_match_env_npts(self, tmp_path):
        path = self._write(tmp_path)
        n_pts, depths, speeds = self._env_ssp_block(path)
        ssp_lines = path.with_suffix('.ssp').read_text().splitlines()
        # line 0 = profile count, line 1 = range vector, then one row per depth
        matrix = np.array([[float(v) for v in ln.split()]
                           for ln in ssp_lines[2:] if ln.strip()])
        assert matrix.shape[0] == n_pts
        assert depths[-1] == pytest.approx(self.DEPTH)
        # The deepest .env sample is the 150 m interpolant of 1490 @ 100 m and
        # 1492 @ 200 m. A matrix built from the raw profile instead of the
        # truncated .env block would put the 200 m sample (1492) on this row.
        assert speeds[-1] == pytest.approx(1491.0, abs=1e-6)
        # Every guard column holds a copy of an interior profile, so the
        # range-0 column of the matrix must reproduce the .env block exactly.
        n_ranges = int(ssp_lines[0].split()[0])
        r_km = np.array([float(v) for v in ssp_lines[1].split()])
        assert r_km.size == n_ranges == matrix.shape[1]
        col0 = matrix[:, int(np.argmin(np.abs(r_km)))]
        np.testing.assert_allclose(col0, speeds, rtol=0, atol=1e-6)

    @pytest.mark.requires_binary
    def test_quad_env_runs(self, tmp_path):
        model = Bellhop(verbose=False, backend='fortran', interp_ssp='quad',
                        work_dir=tmp_path, cleanup=False)
        result = model.run(
            self._env(), Source(depths=25.0, frequencies=100.0),
            Receiver(depths=np.array([50.0]),
                     ranges=np.linspace(1000.0, 10000.0, 10)),
            run_mode=RunMode.COHERENT_TL)
        prt = next(iter(tmp_path.rglob('*.prt'))).read_text()
        assert '*** FATAL ERROR ***' not in prt
        assert np.any(np.isfinite(np.asarray(result.db)))


class TestMeshDepthCoversBathymetry:
    """``Bellhop/bdryMod.f90:211`` aborts on any ``.bty`` depth below the
    medium depth on the ``.env`` SSP header line, and that header carries one
    decimal place. A fractional bathymetry maximum (e.g. GEBCO's 100.04 m
    below) therefore has to round the header UP: rounding to nearest puts the
    header under the deepest ``.bty`` point and the run dies."""

    BATHY = [(0.0, 90.0), (2500.0, 100.04), (5000.0, 95.0)]

    @staticmethod
    def _receiver():
        return Receiver(depths=np.array([50.0]),
                        ranges=np.linspace(100.0, 5000.0, 10))

    def test_header_depth_covers_every_bty_point(self, tmp_path):
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        env = Environment(name='gebco', bathymetry=self.BATHY, ssp=1500.0)
        path = tmp_path / 'mesh.env'
        write_bellhop_env_file(path, env,
                               Source(depths=25.0, frequencies=100.0),
                               self._receiver())
        header = next(ln for ln in path.read_text().splitlines()
                      if ln.rstrip().endswith(','))
        z_max = float(header.split()[2].rstrip(','))
        bty = path.with_suffix('.bty').read_text().splitlines()
        depths = np.array([float(ln.split()[1]) for ln in bty[2:] if ln.strip()])
        assert depths.max() <= z_max
        # The mesh sits on the deepest bathymetry point itself. bdryMod.f90:211
        # aborts only when a .bty point is STRICTLY deeper than the SSP's last
        # depth, so equality is safe and no margin is needed — and a margin
        # would put Bellhop on a different water column from the other models.
        assert z_max == pytest.approx(max(d for _r, d in self.BATHY))

    @pytest.mark.requires_binary
    def test_fractional_bathymetry_runs(self, tmp_path):
        env = Environment(name='gebco', bathymetry=self.BATHY, ssp=1500.0)
        model = Bellhop(verbose=False, backend='fortran',
                        work_dir=tmp_path, cleanup=False)
        result = model.run(env, Source(depths=25.0, frequencies=100.0),
                           self._receiver(), run_mode=RunMode.COHERENT_TL)
        prt = next(iter(tmp_path.rglob('*.prt'))).read_text()
        assert '*** FATAL ERROR ***' not in prt
        assert list(tmp_path.rglob('*.shd'))
        assert np.any(np.isfinite(np.asarray(result.db)))


class TestRayCenteredGaussianRejected:
    """``bellhop.f90:403`` — ``PickEpsilon`` calls ``ERROUT`` for ``'b'``, and
    ``bellhopcuda/src/runtype.hpp:54`` leaves ``'b'`` out of ``IsRayCen()`` so
    the ports silently run the Cartesian beam instead."""

    def test_constructor_rejects_b(self):
        with pytest.raises(ConfigurationError, match="not implemented"):
            Bellhop(beam_type='b')

    def test_writer_rejects_b(self, tmp_path):
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        with pytest.raises(ConfigurationError, match="not implemented"):
            write_bellhop_env_file(
                tmp_path / 'b.env',
                Environment(bathymetry=100.0, ssp=1500.0),
                Source(depths=25.0, frequencies=100.0),
                Receiver(depths=50.0, ranges=1000.0), beam_type='b')

    @pytest.mark.requires_binary
    def test_solver_aborts_on_b(self, tmp_path):
        """Ground truth for the rejection: patch 'b' into an otherwise valid
        .env and watch the Fortran binary abort."""
        import subprocess
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        path = tmp_path / 'raycen.env'
        write_bellhop_env_file(
            path, Environment(bathymetry=100.0, ssp=1500.0),
            Source(depths=25.0, frequencies=100.0),
            Receiver(depths=np.array([50.0]),
                     ranges=np.linspace(100.0, 2000.0, 5)))
        path.write_text(path.read_text().replace("'CB ", "'Cb "))
        subprocess.run([str(Bellhop(backend='fortran', verbose=False)._exe),
                        'raycen'], cwd=tmp_path, capture_output=True,
                       text=True, timeout=300)
        prt = path.with_suffix('.prt').read_text()
        assert '*** FATAL ERROR ***' in prt
        assert 'not implemented in BELLHOP' in prt


class TestBeamShapeValidation:
    """``component`` is ``P``/``V``/``H`` (``influence.f90:120-130``); an
    unknown letter falls through to pressure. An unknown ``beam_width_type``
    leaves ``epsilonOpt`` at zero (``bellhop.f90:372-390``). Both are silent,
    so they are rejected up front. Which beam types honour a ``V``/``H``
    component — only the ray-centred Cerveny one — is a separate contract,
    tested by ``TestBellhopComponentIsRayCentredOnly`` in this file."""

    @pytest.mark.parametrize('component', ['P', 'V', 'H'])
    def test_valid_components_ok(self, component):
        with warnings.catch_warnings():
            # 'C' reads no Component, so V/H warn that the letter is inert.
            warnings.simplefilter('ignore', UserWarning)
            model = Bellhop(beam_type='C', component=component)
        assert model.component == component

    def test_displacement_component_rejected(self):
        with pytest.raises(ConfigurationError, match='component'):
            Bellhop(beam_type='C', component='D')

    def test_invalid_beam_width_type_rejected(self):
        with pytest.raises(ConfigurationError, match='beam_width_type'):
            Bellhop(beam_type='C', beam_width_type='Q')

    def test_invalid_beam_curvature_rejected(self):
        with pytest.raises(ConfigurationError, match='beam_curvature'):
            Bellhop(beam_type='C', beam_curvature='Q')

    def test_negative_n_beams_rejected(self):
        with pytest.raises(ConfigurationError, match='n_beams'):
            Bellhop(n_beams=-1)


class TestLineSourceArrivalsPhase:
    """``ArrMod.f90:103-104`` scales a line source by a purely real
    ``4*sqrt(pi)``, exactly as ``influence.f90:784`` does on the ``.shd``
    path, so the arrivals-derived BROADBAND / TIME_SERIES results need the
    same ``exp(-i*pi/4)`` the ``.shd`` path applies."""

    @staticmethod
    def _cell():
        return dict(n_arrivals=2,
                    amplitudes=np.array([1.0, 0.6]),
                    phases=np.array([0.0, 35.0]),
                    delays=np.array([0.7, 0.9]),
                    delays_imag=np.array([0.0, 0.0]),
                    n_top_bounces=np.array([0, 1]),
                    n_bot_bounces=np.array([0, 0]),
                    src_angles=np.array([0.0, 5.0]),
                    rcv_angles=np.array([0.0, 5.0]))

    def test_transfer_function_phase_offset(self):
        f = np.linspace(90.0, 110.0, 8)
        base = Bellhop(verbose=False)._arrivals_to_tf(self._cell(), f)
        shifted = Bellhop(verbose=False)._arrivals_to_tf(
            self._cell(), f, phase_offset=-np.pi / 4.0)
        np.testing.assert_allclose(shifted, base * np.exp(-1j * np.pi / 4.0),
                                   rtol=1e-12, atol=1e-12)

    @pytest.mark.requires_binary
    def test_line_and_point_share_the_shd_phase_residual(self):
        """The source-type factor is a scalar applied at the very end of both
        codes, so the arrivals-vs-.shd phase residual — the beam-model bias —
        must be identical for a line and a point source. A pi/4 gap between the
        two means only one of the paths carries the 2-D correction."""
        env = Environment(name='line-phase', bathymetry=200.0, ssp=1500.0)
        rcv = Receiver(depths=np.array([100.0]),
                       ranges=np.array([2000.0, 4000.0]))
        fc = 200.0
        model = Bellhop(verbose=False, backend='fortran')
        residuals = {}
        for stype in ('point', 'line'):
            src = Source(depths=50.0, frequencies=fc, source_type=stype)
            shd = np.asarray(model.run(env, src, rcv,
                                       run_mode=RunMode.COHERENT_TL).data)
            hf = np.asarray(model.run(env, src, rcv,
                                      run_mode=RunMode.BROADBAND,
                                      frequencies=np.array([fc])).data)
            residuals[stype] = np.angle(hf[..., 0].ravel() / shd.ravel())
        delta = np.angle(np.exp(1j * (residuals['line'] - residuals['point'])))
        assert np.max(np.abs(delta)) < 0.1, (
            f"line/point arrivals phase differ by {np.rad2deg(delta)} deg")


class TestIrregularGridBroadband:
    """``ArrMod.f90:101-102`` writes ``NRz_per_range`` depth blocks, which
    ``bellhop.f90:202-206`` sets to 1 for an irregular grid — its entries are
    the paired receivers ``(Rz(i), Rr(i))``. So broadband/time-series synthesis
    cannot index ``[0][ird][irr]`` over ``len(receiver_depths)`` blocks; only
    one block exists and the walk runs off the end. The depth axis has to
    collapse onto the range axis, as ``read_shd_file`` already does for the TL
    path."""

    @staticmethod
    def _fixture():
        return (Environment(name='irr', bathymetry=200.0, ssp=1500.0),
                Receiver(depths=[50.0, 100.0, 150.0],
                         ranges=[1000.0, 2000.0, 3000.0]))

    def test_broadband_collapses_the_depth_axis(self):
        env, rcv = self._fixture()
        result = Bellhop(verbose=False, grid_type='I').run(
            env, Source(depths=25.0, frequencies=[150.0, 200.0, 250.0]),
            rcv, RunMode.BROADBAND)
        assert list(result.coords) == ['range', 'frequency']
        assert result.data.shape == (3, 3)
        assert np.asarray(result.metadata['receiver_depths']) == pytest.approx(
            np.asarray(rcv.depths))

    def test_rectilinear_broadband_keeps_its_depth_axis(self):
        env, rcv = self._fixture()
        result = Bellhop(verbose=False, grid_type='R').run(
            env, Source(depths=25.0, frequencies=[150.0, 200.0, 250.0]),
            rcv, RunMode.BROADBAND)
        assert list(result.coords) == ['depth', 'range', 'frequency']
        assert result.data.shape == (3, 3, 3)

    def test_time_series_collapses_the_depth_axis(self):
        env, rcv = self._fixture()
        result = Bellhop(verbose=False, grid_type='I').run(
            env, Source(depths=25.0, frequencies=200.0), rcv,
            RunMode.TIME_SERIES, source_waveform=np.hanning(64),
            sample_rate=8000.0)
        assert list(result.coords) == ['range', 'time']
        assert result.data.shape[0] == 3

    def test_the_writer_refuses_a_mismatched_irregular_grid(self, tmp_path):
        """``ReadEnvironmentBell.f90:414`` ERROUTs on ``NRz != NRr``; the
        public writer must refuse it rather than emit a rejected deck."""
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        env, _ = self._fixture()
        with pytest.raises(ConfigurationError, match='irregular'):
            write_bellhop_env_file(
                tmp_path / 'b.env', env, Source(depths=25.0, frequencies=200.0),
                Receiver(depths=[50.0, 100.0, 150.0], ranges=[1000.0, 2000.0]),
                grid_type='I')


class TestBeamPatternMustSpanTheLaunchFan:
    """``bellhop.f90:269-270`` clamps the beam-pattern table index but not the
    interpolation weight at ``:273``, so ``Amp0`` (``:274``) extrapolates past
    both ends. ``misc/beampattern.f90:59`` has already converted the levels to
    linear amplitude, so extrapolating a roll-off drives ``Amp0`` negative — the
    outer beams launch louder than declared and phase-inverted, and the field
    returns partly NaN with no warning on any backend.
    ``third_party/MODIFICATIONS.md`` records the site as deliberately left
    unclamped so the three backends stay identical, so the input check is the
    only guard available.
    """

    @staticmethod
    def _env():
        from uacpy.core.environment import (SoundSpeedProfile,
                                            BoundaryProperties)
        return Environment(
            bathymetry=100.0,
            ssp=SoundSpeedProfile.from_pairs([(0.0, 1500.0), (100.0, 1500.0)]),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.8,
                                      attenuation=0.5))

    @staticmethod
    def _lobe(span_deg):
        return np.array([[-span_deg, -40.0], [0.0, 0.0], [span_deg, -40.0]])

    def _run(self, span_deg, fan):
        rcv = Receiver(depths=[10.0, 50.0, 90.0],
                       ranges=[50.0, 100.0, 200.0, 400.0, 800.0, 1600.0])
        src = Source(depths=25.0, frequencies=200.0,
                     beam_pattern=self._lobe(span_deg))
        return Bellhop(beam_type='G', n_beams=501, alpha=fan,
                       backend='fortran').compute_tl(self._env(), src, rcv)

    def test_a_pattern_narrower_than_the_fan_is_refused(self):
        with pytest.raises(ConfigurationError, match='launch fan'):
            self._run(30.0, (-80.0, 80.0))

    def test_the_default_fan_is_what_makes_this_reachable(self):
        """``alpha`` defaults to ±80°, so an ordinary narrow lobe trips it."""
        assert Bellhop().alpha == (-80, 80)

    @pytest.mark.requires_binary
    def test_a_pattern_covering_the_fan_runs_and_is_finite(self):
        tl = np.asarray(self._run(80.0, (-80.0, 80.0)).db)
        assert np.isfinite(tl).all()

    @pytest.mark.requires_binary
    def test_narrowing_the_fan_to_the_pattern_also_works(self):
        """The other remedy the error offers must work too."""
        tl = np.asarray(self._run(60.0, (-60.0, 60.0)).db)
        assert np.isfinite(tl).all()


class TestBeamTypeCapabilities:
    """``Bellhop/influence.f90`` implements a different set of ``RunType(1:1)``
    letters per influence routine, and uacpy used to write any pairing. The
    unsupported ones are not merely unsupported: the arrivals/eigenray pairs
    write ``U( iz, ir )`` into the ``U(1,1)`` dummy ``bellhop.f90:216``
    allocated for them (glibc aborts, exit -6), and ``beam_type='S'`` with an
    incoherent run returns a field that was measured 70 % NaN. The C++/CUDA
    ports implemented some of the missing branches, so an ungated pair made the
    *answer* depend on which binaries ``install.sh`` built."""

    @staticmethod
    def _fixture():
        return (Environment(name='caps', bathymetry=100.0, ssp=1500.0),
                Source(depths=25.0, frequencies=200.0),
                Receiver(depths=50.0, ranges=np.linspace(500.0, 5000.0, 10)))

    # (beam_type, run_mode) pairs whose influence routine has no such branch.
    _UNSUPPORTED = [
        ('S', RunMode.INCOHERENT_TL), ('S', RunMode.SEMICOHERENT_TL),
        ('S', RunMode.ARRIVALS), ('S', RunMode.BROADBAND),
        ('S', RunMode.TIME_SERIES),
        ('C', RunMode.EIGENRAYS), ('C', RunMode.ARRIVALS),
        ('C', RunMode.BROADBAND), ('C', RunMode.TIME_SERIES),
        ('R', RunMode.EIGENRAYS), ('R', RunMode.ARRIVALS),
        ('R', RunMode.BROADBAND), ('R', RunMode.TIME_SERIES),
    ]

    @pytest.mark.parametrize('beam_type,run_mode', _UNSUPPORTED)
    def test_unsupported_pairs_are_refused(self, beam_type, run_mode):
        env, src, rcv = self._fixture()
        kw = ({'source_waveform': np.hanning(64), 'sample_rate': 8000.0}
              if run_mode == RunMode.TIME_SERIES else {})
        with pytest.raises(ConfigurationError, match='influence.f90'):
            Bellhop(verbose=False, beam_type=beam_type).run(
                env, src, rcv, run_mode, **kw)

    @pytest.mark.parametrize('beam_type', ['G', 'B', 'g', 'S', 'C', 'R'])
    def test_rays_is_safe_for_every_beam_type(self, beam_type):
        """``bellhop.f90:288`` writes the trajectory and returns before the
        influence dispatch, so no routine is entered at all."""
        env, src, rcv = self._fixture()
        result = Bellhop(verbose=False, beam_type=beam_type).run(
            env, src, rcv, RunMode.RAYS)
        assert isinstance(result, Rays)

    @pytest.mark.parametrize('beam_type', ['G', 'B', 'g', 'C', 'R'])
    def test_incoherent_runs_where_the_branch_exists(self, beam_type):
        """Both Cerveny routines do implement ``CASE ( 'I', 'S' )``
        (``influence.f90:137-140`` and :279-283), so only ``'S'`` is gated."""
        env, src, rcv = self._fixture()
        result = Bellhop(verbose=False, beam_type=beam_type).run(
            env, src, rcv, RunMode.INCOHERENT_TL)
        assert isinstance(result, Field)

    def test_simple_gaussian_beam_runs_coherent_and_eigenrays(self):
        """``InfluenceSGB`` has a ``'E'`` branch and a coherent ``CASE
        DEFAULT``; the gate must not take those away."""
        env, src, rcv = self._fixture()
        model = Bellhop(verbose=False, beam_type='S')
        assert isinstance(model.run(env, src, rcv, RunMode.COHERENT_TL), Field)
        assert isinstance(model.run(env, src, rcv, RunMode.EIGENRAYS), Rays)

    def test_the_table_matches_the_vendored_source(self):
        """Only the routines calling ``ApplyContribution`` carry the ``'E'`` and
        ``'A'``/``'a'`` branches, so a vendor refresh that changes which ones do
        must fail here rather than silently invalidate the table."""
        from pathlib import Path
        from uacpy.models.bellhop import (_BEAM_TYPE_RUN_TYPES,
                                          _INFLUENCE_ROUTINE)
        src_file = (Path(__file__).resolve().parent.parent / 'third_party' /
                    'Acoustics-Toolbox' / 'Bellhop' / 'influence.f90')
        text = src_file.read_text()
        for beam_type, routine in _INFLUENCE_ROUTINE.items():
            start = text.index(f'SUBROUTINE {routine}(')
            end = text.index('END SUBROUTINE', start)
            body = text[start:end]
            calls_apply = 'ApplyContribution' in body
            claims_arrivals = 'A' in _BEAM_TYPE_RUN_TYPES[beam_type]
            assert calls_apply == claims_arrivals, (
                f"{routine} ({beam_type}): calls ApplyContribution="
                f"{calls_apply} but the table claims arrivals support="
                f"{claims_arrivals}")
            has_eigenray_branch = calls_apply or "CASE ( 'E' )" in body
            assert ('E' in _BEAM_TYPE_RUN_TYPES[beam_type]) is has_eigenray_branch, (
                f"{routine} ({beam_type}): eigenray branch in the source="
                f"{has_eigenray_branch}, table says "
                f"{'E' in _BEAM_TYPE_RUN_TYPES[beam_type]}")


class TestReceiverGridMatchesTheBeamType:
    """Two receiver-grid choices are silently wrong for the routines that do not
    implement them — no NaN, no warning, up to 30 dB of error."""

    @staticmethod
    def _env_src():
        return (Environment(name='grid', bathymetry=100.0, ssp=1500.0),
                Source(depths=25.0, frequencies=200.0))

    @pytest.mark.parametrize('beam_type', ['g', 'S', 'C', 'R'])
    def test_irregular_grid_is_refused(self, beam_type):
        """``bellhop.f90:202-204`` pins ``NRz_per_range`` to 1 for
        ``RunType(5:5)=='I'`` and only ``InfluenceGeoHatCart`` (:461-465) and
        ``InfluenceGeoGaussianCart`` (:581-585) re-read ``Pos%Rz(ir)``; the rest
        evaluate every paired receiver at ``receiver.depths[0]``."""
        env, src = self._env_src()
        rcv = Receiver(depths=[10.0, 30.0, 50.0, 70.0],
                       ranges=[500.0, 1400.0, 2300.0, 3200.0])
        with pytest.raises(ConfigurationError, match='NRz_per_range'):
            Bellhop(verbose=False, beam_type=beam_type, grid_type='I').run(
                env, src, rcv, RunMode.COHERENT_TL)

    @pytest.mark.parametrize('beam_type', ['G', 'B'])
    def test_irregular_grid_is_allowed_where_implemented(self, beam_type):
        env, src = self._env_src()
        rcv = Receiver(depths=[10.0, 30.0, 50.0, 70.0],
                       ranges=[500.0, 1400.0, 2300.0, 3200.0])
        result = Bellhop(verbose=False, beam_type=beam_type,
                         grid_type='I').run(env, src, rcv, RunMode.COHERENT_TL)
        assert np.asarray(result.db).shape == (4,)

    @pytest.mark.parametrize('beam_type', ['g', 'C', 'R'])
    def test_non_uniform_ranges_are_refused(self, beam_type):
        """These form the range index as
        ``INT( ( r - Pos%Rr(1) ) / Pos%Delta_r ) + 1`` (``influence.f90:92``,
        :223-224, :339, :351) and ``SourceReceiverPositions.f90:160`` sets
        ``Delta_r`` from the last gap alone."""
        env, src = self._env_src()
        rcv = Receiver(depths=50.0,
                       ranges=[500.0, 700.0, 1000.0, 1500.0, 2200.0, 3200.0])
        with pytest.raises(ConfigurationError, match='Delta_r'):
            Bellhop(verbose=False, beam_type=beam_type).run(
                env, src, rcv, RunMode.COHERENT_TL)

    @pytest.mark.parametrize('beam_type', ['G', 'B', 'S'])
    def test_arbitrary_range_spacing_is_allowed_where_implemented(self,
                                                                 beam_type):
        """``doc/bellhop.htm``'s list is wrong twice: it includes
        ``CervenyRayCen`` and omits ``S``. ``InfluenceSGB`` (:683) compares
        ``rB > Pos%Rr( ir )`` directly and never touches ``Delta_r``."""
        env, src = self._env_src()
        rcv = Receiver(depths=50.0,
                       ranges=[500.0, 700.0, 1000.0, 1500.0, 2200.0, 3200.0])
        result = Bellhop(verbose=False, beam_type=beam_type).run(
            env, src, rcv, RunMode.COHERENT_TL)
        assert np.isfinite(np.asarray(result.db)).all()

    @pytest.mark.parametrize('beam_type', ['g', 'C', 'R'])
    def test_uniform_ranges_accepted(self, beam_type):
        env, src = self._env_src()
        rcv = Receiver(depths=50.0, ranges=np.linspace(500.0, 5000.0, 10))
        result = Bellhop(verbose=False, beam_type=beam_type).run(
            env, src, rcv, RunMode.COHERENT_TL)
        assert isinstance(result, Field)


class TestBeamCountGuard:
    """``bellhop.f90:176-178`` leaves ``Angles%Dalpha = 0`` when
    ``Nalpha == 1``, so ``q0 = c/Dalpha`` gives every beam zero width and the
    influence sum contributes nothing — the run exits 0 with an all-NaN field
    and no diagnostic. Two beams have a finite Dalpha and do return a field,
    so the floor is at two. Ray modes are unaffected:
    ``bellhop.f90:288`` skips influence, and one traced ray is legitimate."""

    @staticmethod
    def _env():
        return Environment(bathymetry=100.0, ssp=1500.0,
                           bottom=BoundaryProperties(
                               acoustic_type='half-space', sound_speed=1600.0,
                               density=1.5, attenuation=0.5))

    @pytest.mark.parametrize('run_mode', [RunMode.COHERENT_TL, RunMode.ARRIVALS])
    def test_degenerate_fan_is_refused_for_influence_modes(self, run_mode):
        # One beam is the whole degenerate case: bellhop.f90:176-178 leaves
        # Dalpha at 0 there, giving every beam zero width.
        with pytest.raises(ConfigurationError, match='Dalpha'):
            Bellhop(n_beams=1).run(
                self._env(), Source(depths=25.0, frequencies=200.0),
                Receiver(depths=[50.0], ranges=[1000.0]), run_mode=run_mode)

    @pytest.mark.parametrize('run_mode', [RunMode.COHERENT_TL, RunMode.ARRIVALS])
    def test_a_two_beam_fan_is_under_resolved_not_degenerate(self, run_mode):
        # Two beams have a finite Dalpha and return a field wherever the pair
        # reaches the receiver — 69.1 dB against a converged 66.1 dB on a
        # deep-water direct path — see
        # ``test_bellhop_two_beam_fan_returns_a_usable_field`` in this file.
        # Sparse is the caller's
        # choice; only Dalpha = 0 is the model's to refuse.
        Bellhop(n_beams=2).run(
            self._env(), Source(depths=25.0, frequencies=200.0),
            Receiver(depths=[50.0], ranges=[1000.0]), run_mode=run_mode)

    @pytest.mark.parametrize('n_beams', [1, 2])
    def test_sparse_fan_is_allowed_for_ray_modes(self, n_beams):
        # The discriminating counterpart: beam width never enters a ray trace.
        rays = Bellhop(n_beams=n_beams).run(
            self._env(), Source(depths=25.0, frequencies=200.0),
            Receiver(depths=[50.0], ranges=[1000.0]), run_mode=RunMode.RAYS)
        assert len(rays.rays) == n_beams

    @pytest.mark.parametrize('n_beams', [0, 3, 51])
    def test_usable_fan_produces_a_finite_field(self, n_beams):
        # 0 lets Bellhop auto-pick; the guard must not touch either case.
        tl = np.squeeze(Bellhop(n_beams=n_beams).run(
            self._env(), Source(depths=25.0, frequencies=200.0),
            Receiver(depths=[50.0], ranges=[1000.0]),
            run_mode=RunMode.COHERENT_TL).db)
        assert np.all(np.isfinite(tl))


class TestPrecalcBoundaryIsRefused:
    """``ReadEnvironmentBell.f90:459`` accepts the ``'P'`` option letter and
    prints "reading PRECALCULATED IRC", but ``bellhop.f90:681``'s
    ``SELECT CASE ( HS%BC )`` implements only 'R', 'V', 'F' and 'A'/'G' —
    there is no 'P' branch, so the run failed with a bare exit code instead of
    naming the boundary. The ``Bounce`` docstring's Model Support list already
    records it."""

    def test_precalc_bottom_raises_and_names_the_cause(self):
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          bottom=BoundaryProperties(acoustic_type='precalc'))
        with pytest.raises(UnsupportedFeatureError, match="'P' reflection branch"):
            Bellhop().run(env, Source(depths=25.0, frequencies=200.0),
                          Receiver(depths=[50.0], ranges=[1000.0]),
                          run_mode=RunMode.COHERENT_TL)

    def test_precalc_column_at_positive_range_raises(self):
        # A range-dependent Bottom carries SeabedColumn entries whose
        # acoustic_type lives on ``.halfspace``; a precalc column anywhere
        # along the transect must be refused, not only one at r = 0.
        from uacpy.core.bottom import Bottom, SeabedColumn
        bot = Bottom(columns=[
            SeabedColumn(layers=[], halfspace=BoundaryProperties(
                acoustic_type='half-space', sound_speed=1600.0,
                density=1.5, attenuation=0.5)),
            SeabedColumn(layers=[],
                         halfspace=BoundaryProperties(acoustic_type='precalc')),
        ], ranges=[0.0, 5000.0])
        env = Environment(bathymetry=100.0, ssp=1500.0, bottom=bot)
        with pytest.raises(UnsupportedFeatureError,
                           match="'P' reflection branch"):
            Bellhop().run(env, Source(depths=25.0, frequencies=200.0),
                          Receiver(depths=[50.0], ranges=[1000.0]),
                          run_mode=RunMode.COHERENT_TL)

    def test_ordinary_halfspace_is_unaffected(self):
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          bottom=BoundaryProperties(
                              acoustic_type='half-space', sound_speed=1600.0,
                              density=1.5, attenuation=0.5))
        tl = np.squeeze(Bellhop().run(
            env, Source(depths=25.0, frequencies=200.0),
            Receiver(depths=[50.0], ranges=[1000.0]),
            run_mode=RunMode.COHERENT_TL).db)
        assert np.all(np.isfinite(tl))


class TestSourceMustBeInsideTheMedium:
    """``bellhop.f90:488-492`` tests ``DistBegTop <= 0 .OR. DistBegBot <= 0``
    and terminates every ray at step 1 — *"source must be within the medium"*.
    The run then exits 0 with an all-NaN field, zero arrivals and 1-point
    rays, warning about none of it. Measured: 99.99 m gives a perfectly good
    field, 100.00 m gives nanfrac 1.0000. All three backends fail identically.

    The test is against the seafloor at **r = 0** (``bellhop.f90:237`` launches
    from ``xs = [0.0, sz]``), not ``env.depth`` — on a sloping bottom a source
    can be buried at its own range while well above ``env.depth``."""

    BOT = BoundaryProperties(acoustic_type='half-space', sound_speed=1600.0,
                             density=1.8, attenuation=0.5)

    def _flat(self):      # env.depth == r=0 floor == 100
        return Environment(bathymetry=100.0, ssp=1500.0, bottom=self.BOT)

    def _slope(self):     # r=0 floor 50, env.depth 150
        return Environment(bathymetry=[(0.0, 50.0), (5000.0, 150.0)],
                           ssp=1500.0, bottom=self.BOT)

    def _rcv(self):
        return Receiver(depths=[30.0, 50.0, 75.0],
                        ranges=np.linspace(100.0, 5000.0, 20))

    def _run(self, env, zs):
        return Bellhop().run(env, Source(depths=zs, frequencies=200.0),
                             self._rcv(), run_mode=RunMode.COHERENT_TL)

    def test_source_on_a_flat_seafloor_raises(self):
        with pytest.raises(ConfigurationError, match='at or below the seafloor'):
            self._run(self._flat(), 100.0)

    @pytest.mark.parametrize('zs', [50.0, 100.0])
    def test_sloping_bottom_is_judged_at_the_source_range(self, zs):
        # 50 m is the r=0 floor; 100 m is buried at r=0 but still above
        # env.depth=150. A guard written against env.depth misses both.
        with pytest.raises(ConfigurationError, match='at or below the seafloor'):
            self._run(self._slope(), zs)

    @pytest.mark.parametrize('env_name,zs', [('_flat', 99.99), ('_slope', 40.0)])
    def test_a_source_inside_the_medium_runs(self, env_name, zs):
        # The knife edge: 99.99 m is a good field, so the guard must not
        # creep upward. Water-column cells are finite; on the sloping bottom
        # the receivers that sit below the local seafloor at short range are
        # NaN by the below-domain mask — masked cells, not a rejection.
        env = getattr(self, env_name)()
        rcv = self._rcv()
        tl = np.atleast_2d(np.squeeze(self._run(env, zs).db))
        seafloor = np.asarray(env.bathymetry.eval(range=rcv.ranges),
                              dtype=float)
        above = (np.asarray(rcv.depths, dtype=float)[:, None]
                 <= seafloor[None, :])
        assert np.all(np.isfinite(tl[above]))
        assert np.all(np.isnan(tl[~above]))

    def test_the_guard_is_bellhop_only(self):
        # Kraken/Scooter/RAM answer a buried source (heavily attenuated, as
        # physics requires). Putting this on the shared funnel would reject
        # three correct answers.
        from uacpy.models.base import PropagationModel
        import inspect
        assert 'at or below the seafloor' not in inspect.getsource(
            PropagationModel._validate_geometry)


class TestSingleReceiverRangeBeamTypes:
    """``g``/``C``/``R`` clamp the receiver index to ``Pos%NRr``
    (``influence.f90:339,351``), so with one range ``irA == irB`` at every
    step and ``:354`` skips the whole ray — the run exits 0 with an all-NaN
    field, zero eigenrays and zero arrivals. ``G``/``B`` walk the index with a
    bracket test and are unaffected."""

    @staticmethod
    def _env():
        return Environment(bathymetry=100.0, ssp=1500.0,
                           bottom=BoundaryProperties(
                               acoustic_type='half-space', sound_speed=1600.0,
                               density=1.8, attenuation=0.5))

    @pytest.mark.parametrize('beam_type', ['g', 'C', 'R'])
    def test_single_range_is_refused(self, beam_type):
        with pytest.raises(ConfigurationError, match='single receiver range'):
            Bellhop(beam_type=beam_type).run(
                self._env(), Source(depths=25.0, frequencies=200.0),
                Receiver(depths=[50.0], ranges=[1000.0]),
                run_mode=RunMode.COHERENT_TL)

    @pytest.mark.parametrize('beam_type', ['G', 'B'])
    def test_bracket_testing_beam_types_accept_one_range(self, beam_type):
        # The discriminating counterpart: these index by bracket, not by
        # division, so one range is legitimate for them.
        tl = np.squeeze(Bellhop(beam_type=beam_type).run(
            self._env(), Source(depths=25.0, frequencies=200.0),
            Receiver(depths=[50.0], ranges=[1000.0]),
            run_mode=RunMode.COHERENT_TL).db)
        assert np.all(np.isfinite(tl))

    @pytest.mark.parametrize('beam_type', ['g', 'C', 'R'])
    def test_several_equally_spaced_ranges_work(self, beam_type):
        tl = np.squeeze(Bellhop(beam_type=beam_type).run(
            self._env(), Source(depths=25.0, frequencies=200.0),
            Receiver(depths=[50.0], ranges=np.linspace(500.0, 3000.0, 6)),
            run_mode=RunMode.COHERENT_TL).db)
        assert np.any(np.isfinite(tl))


class TestSolverWarningsReachTheCaller:
    """The AT binaries write both fatals and *non-fatal* diagnoses to the
    ``.prt``. Only the fatals were read (``_attach_prt_tail``, on the
    exception path), so a run the solver itself diagnosed came back at exit 0
    with a full-size result and nothing said — BELLHOP writes ``Warning in
    BELLHOP : Too few beams`` while uacpy returned the under-sampled TL field
    in silence."""

    @staticmethod
    def _env():
        return Environment(bathymetry=200.0, ssp=1500.0,
                           bottom=BoundaryProperties(
                               acoustic_type='half-space', sound_speed=1700.0,
                               density=1.8, attenuation=0.5))

    def _run(self, n_beams):
        return Bellhop(n_beams=n_beams, backend='fortran').run(
            self._env(), Source(depths=50.0, frequencies=500.0),
            Receiver(depths=[100.0], ranges=np.linspace(500.0, 5000.0, 10)),
            run_mode=RunMode.COHERENT_TL)

    def test_too_few_beams_is_reported_in_the_solvers_own_words(self):
        with pytest.warns(UserWarning, match='Too few beams'):
            self._run(5)

    def test_a_converged_run_is_silent(self):
        # The discriminating counterpart: n_beams=0 lets BELLHOP choose, the
        # .prt carries no Warning line, and the caller must not be nagged.
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            warnings.filterwarnings('ignore', message='.*not redistributable.*')
            self._run(0)


class TestSubSeafloorReceiversAreMasked:
    """BELLHOP clamps a receiver below the bottom boundary onto it
    (``misc/SourceReceiverPositions.f90:136-139``): the ``.shd`` then carries
    the clamped depth axis with the boundary row repeated, so uacpy used to
    return a constant TL plateau under the requested depths. The wrapper now
    restores the requested depth axis and NaNs every cell below the local
    seafloor — the same below-domain convention as Scooter, SPARC and RAM.
    Kraken and the OASES models still return their physical transmitted
    field; this mask is Bellhop's, whose ray tracer evaluates nothing in the
    sediment."""

    @staticmethod
    def _run(depths):
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          bottom=BoundaryProperties(
                              acoustic_type='half-space', sound_speed=1600.0,
                              density=1.5, attenuation=0.5))
        return Bellhop(verbose=False).run(
            env, Source(depths=25.0, frequencies=200.0),
            Receiver(depths=depths, ranges=np.array([500.0, 1000.0])),
            run_mode=RunMode.COHERENT_TL)

    def test_below_seafloor_cells_are_nan_on_the_requested_axis(self):
        depths = np.linspace(10.0, 300.0, 30)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            field = self._run(depths)
        np.testing.assert_allclose(field.coords['depth'], depths)
        below = depths > 100.0
        assert np.isnan(field.data[below, :]).all()
        assert np.isfinite(field.data[~below, :]).all()

    def test_water_column_only_grid_is_untouched(self):
        depths = np.linspace(10.0, 90.0, 9)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            field = self._run(depths)
        assert np.isfinite(field.data).all()


class TestBroadbandNoDataCellsAreNaN:
    """A zero-arrival cell carries no model output. ``H = 0`` (600 dB) reads
    as a real, perfectly quiet channel, so the broadband routes return NaN
    there — the same no-data convention as COHERENT_TL's empty .shd cells."""

    def test_transfer_function_zero_arrivals_is_all_nan(self):
        empty = dict(n_arrivals=0, amplitudes=np.array([]),
                     phases=np.array([]), delays=np.array([]),
                     delays_imag=np.array([]))
        H = Bellhop._arrivals_to_tf(empty, np.linspace(90.0, 110.0, 8))
        assert np.isnan(H).all()

    def test_broadband_r0_column_is_nan_and_warns(self):
        env = Environment(name='bb-r0', bathymetry=100.0, ssp=1500.0)
        src = Source(depths=25.0, frequencies=200.0)
        rcv = Receiver(depths=np.array([50.0]),
                       ranges=np.array([0.0, 2000.0]))
        with pytest.warns(UserWarning, match='r=0'):
            H = Bellhop(verbose=False).run(env, src, rcv,
                                           run_mode=RunMode.BROADBAND)
        assert np.isnan(np.asarray(H.data)[:, 0, :]).all()
        assert np.isfinite(np.asarray(H.data)[:, 1, :]).all()

    def test_r0_warning_fires_on_every_run(self):
        """Cadence matches Kraken / Scooter / RAM: per run, not per instance."""
        env = Environment(name='r0-cadence', bathymetry=100.0, ssp=1500.0)
        src = Source(depths=25.0, frequencies=200.0)
        rcv = Receiver(depths=np.array([50.0]),
                       ranges=np.array([0.0, 1000.0]))
        model = Bellhop(verbose=False)
        for _ in range(2):
            with pytest.warns(UserWarning, match='r=0'):
                model.run(env, src, rcv, run_mode=RunMode.COHERENT_TL)


class TestBroadbandRestoresRequestedDepthAxis:
    """BELLHOP clamps a below-bottom receiver onto the boundary
    (misc/SourceReceiverPositions.f90:136-139); the broadband routes used to
    return the clamped axis with the boundary field relabelled as the asked
    depth. The requested axis is restored with NaN there, matching run()'s
    TL-mode masking."""

    @staticmethod
    def _rig():
        env = Environment(
            name='bb-depths', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.5,
                                      attenuation=0.5))
        return (env, Source(depths=25.0, frequencies=200.0),
                Receiver(depths=np.array([50.0, 150.0]),
                         ranges=np.array([1000.0, 2000.0])))

    def test_broadband_masks_sub_bottom_depths(self):
        env, src, rcv = self._rig()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            H = Bellhop(verbose=False).run(env, src, rcv,
                                           run_mode=RunMode.BROADBAND)
        np.testing.assert_array_equal(np.asarray(H.coords['depth']),
                                      [50.0, 150.0])
        assert np.isnan(np.asarray(H.data)[1]).all()
        assert np.isfinite(np.asarray(H.data)[0]).all()

    def test_time_series_masks_and_stamps_the_derived_band(self):
        env, src, rcv = self._rig()
        fs = 2000.0
        t = np.arange(0, 0.05, 1 / fs)
        wf = np.sin(2 * np.pi * 200.0 * t) * np.hanning(t.size)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ts = Bellhop(verbose=False).run(
                env, src, rcv, run_mode=RunMode.TIME_SERIES,
                source_waveform=wf, sample_rate=fs)
        np.testing.assert_array_equal(np.asarray(ts.coords['depth']),
                                      [50.0, 150.0])
        assert np.isnan(np.asarray(ts.data)[1]).all()
        assert np.isfinite(np.asarray(ts.data)[0]).all()
        # The frequency stamp is the waveform-derived band, like the
        # IFFT-based engines'; fc stays on metadata['center_frequency'].
        freqs = np.atleast_1d(np.asarray(ts.frequencies, dtype=float))
        assert freqs.size > 1
        assert freqs[0] < 200.0 < freqs[-1]
        assert ts.metadata['center_frequency'] == 200.0


class TestAutoBounceTriggersOnLayeringOnly:
    """bellhop.f90:694-712 evaluates the exact acousto-elastic halfspace
    reflection coefficient natively (per range node on a long .bty), so only
    a LAYERED bottom — which the single-halfspace .env cannot carry — routes
    through BOUNCE. Routing a range-dependent elastic bottom through BOUNCE
    collapsed it to one column (measured 9.80 dB error)."""

    _src = staticmethod(lambda: Source(depths=25.0, frequencies=200.0))
    _rcv = staticmethod(lambda: Receiver(depths=np.array([50.0]),
                                         ranges=np.array([1000.0, 3000.0])))

    def test_elastic_halfspace_runs_natively(self):
        env = Environment(
            name='ri-elastic', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.6,
                                      attenuation=0.3, shear_speed=400.0,
                                      shear_attenuation=0.5))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            f = Bellhop(verbose=False).run(env, self._src(), self._rcv(),
                                           run_mode=RunMode.COHERENT_TL)
        assert not [w for w in caught if 'BOUNCE' in str(w.message)]
        assert 'bounce_result' not in f.metadata
        assert np.isfinite(np.asarray(f.db)).all()

    def test_rd_elastic_bottom_runs_natively(self):
        from uacpy.core.bottom import Bottom
        env = Environment(
            name='rd-elastic', bathymetry=100.0, ssp=1500.0,
            bottom=Bottom.from_halfspaces(
                ranges=[0.0, 1500.0, 3000.0],
                sound_speed=[1700.0, 2400.0, 1700.0],
                density=[1.6, 2.2, 1.6], attenuation=0.3,
                shear_speed=[400.0, 1200.0, 400.0],
                shear_attenuation=0.5))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            f = Bellhop(verbose=False).run(env, self._src(), self._rcv(),
                                           run_mode=RunMode.COHERENT_TL)
        assert not [w for w in caught if 'BOUNCE' in str(w.message)]
        assert 'bounce_result' not in f.metadata
        assert np.isfinite(np.asarray(f.db)).all()


class TestTLModeRelationships:
    """The three TL run modes against each other on one dense range line.

    ``influence.f90``'s 'I' branch sums per-beam intensity with the phase
    thrown away, so the coherent interference nulls are absent and the TL
    spread over range collapses; 'S' additionally pre-shades each launch
    amplitude by the Lloyd-mirror factor ``sqrt(2)|sin(omega z_s
    sin(alpha)/c)|`` (``bellhop.f90:276-278``), so it matches neither of the
    other two. The per-mode smokes above check shapes; the relationships here
    are what show RunType position 1 actually reached the influence
    dispatch."""

    @pytest.fixture(scope='class')
    def tl_fields(self):
        env = Environment(name='tl-mode-rel', bathymetry=100.0, ssp=1500.0)
        src = Source(depths=50.0, frequencies=100.0)
        rcv = Receiver(depths=np.array([50.0]),
                       ranges=np.linspace(1000.0, 5000.0, 41))
        model = Bellhop(verbose=False)
        return {mode: model.run(env, src, rcv, run_mode=mode)
                for mode in (RunMode.COHERENT_TL, RunMode.INCOHERENT_TL,
                             RunMode.SEMICOHERENT_TL)}

    def test_incoherent_smooths_the_interference_pattern(self, tl_fields):
        coh = np.asarray(tl_fields[RunMode.COHERENT_TL].db).ravel()
        inc = np.asarray(tl_fields[RunMode.INCOHERENT_TL].db).ravel()
        assert np.ptp(inc) < np.ptp(coh), (
            "incoherent TL is no smoother than coherent — RunType(1:1)='I' "
            "never took effect")

    def test_semicoherent_differs_from_both(self, tl_fields):
        coh = np.asarray(tl_fields[RunMode.COHERENT_TL].db).ravel()
        inc = np.asarray(tl_fields[RunMode.INCOHERENT_TL].db).ravel()
        semi = np.asarray(tl_fields[RunMode.SEMICOHERENT_TL].db).ravel()
        # The Lloyd shading redistributes several dB across a 4 km line, so
        # 0.1 dB separates a real third mode from either neighbour while
        # staying far above solver reproducibility.
        assert not np.allclose(semi, coh, atol=0.1)
        assert not np.allclose(semi, inc, atol=0.1)

    def test_incoherent_payload_is_complex_with_zero_imag(self, tl_fields):
        # docs/guide/results.md §9 "An incoherent field has no phase": the
        # incoherent sum rides in the complex .shd container with an
        # identically zero imaginary part.
        data = np.asarray(tl_fields[RunMode.INCOHERENT_TL].data)
        assert np.iscomplexobj(data)
        assert np.all(np.imag(data) == 0.0)


class TestBeamShiftRunTypePosition7:
    """``beam_shift=True`` writes 'S' into RunType position 7
    (bellhop.md §3 "turns on BELLHOP's own beam-displacement correction");
    ``ReadEnvironmentBell.f90:159`` copies that
    position into ``Beam%Type(4:4)``, which is what enables the
    beam-displacement correction on boundary reflections. Deck-token test —
    no binary in the loop."""

    @staticmethod
    def _run_type(tmp_path, **kwargs):
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        path = tmp_path / 'shift.env'
        write_bellhop_env_file(
            path, Environment(bathymetry=100.0, ssp=1500.0),
            Source(depths=25.0, frequencies=100.0),
            Receiver(depths=np.array([50.0]),
                     ranges=np.linspace(100.0, 2000.0, 5)),
            **kwargs)
        # The RunType record is a 7-character quoted line whose position 6
        # is the hard-coded dimension '2' — which is what tells it apart
        # from a 7-character title (the default title 'unnamed' is one).
        line = next(ln for ln in path.read_text().splitlines()
                    if len(ln) == 9 and ln[0] == ln[-1] == "'"
                    and ln[6] == '2')
        return line[1:8]

    def test_beam_shift_writes_S_in_position_7(self, tmp_path):
        assert self._run_type(tmp_path, beam_shift=True)[6] == 'S'

    def test_default_leaves_position_7_blank(self, tmp_path):
        assert self._run_type(tmp_path)[6] == ' '

    def test_the_model_carries_the_flag(self):
        assert Bellhop(beam_shift=True).beam_shift is True
        assert Bellhop().beam_shift is False


class TestInterpSspAutoPick:
    """``interp_ssp=None`` auto-picks quad for a range-dependent ``env.ssp``
    and C-linear otherwise (bellhop.md §5 "SSP connection scheme:", and
    bellhop.md §7 "Range-dependent SSP needs"), asserted on the
    ``TopOpt(1)`` character the deck carries: 'Q' opens ``<root>.ssp``
    unconditionally (``ReadEnvironmentBell.f90:262-268``), 'C' is AT's
    C-linear connection. Deck-token test — no binary in the loop."""

    @staticmethod
    def _topopt(tmp_path, env, receiver):
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        path = tmp_path / 'auto.env'
        write_bellhop_env_file(path, env,
                               Source(depths=25.0, frequencies=100.0),
                               receiver, interp_ssp=None)
        lines = path.read_text().splitlines()
        # Line 0 is the quoted title; the next quoted line is TopOpt.
        topopt = next(ln for ln in lines[1:] if ln.startswith("'"))
        return topopt[1], path

    def test_range_dependent_ssp_auto_picks_quad(self, tmp_path):
        from uacpy.core.ssp import SoundSpeedProfile
        z = np.linspace(0.0, 100.0, 5)
        r = np.array([0.0, 6000.0, 12000.0])
        c2d = 1500.0 + 0.01 * z[:, None] - 1e-4 * r[None, :]
        env = Environment(
            name='rd-auto', bathymetry=100.0,
            ssp=SoundSpeedProfile.from_2d(z, r, c2d))
        # Receivers stop at 8 km so the 1.2x ray box stays inside the 12 km
        # SSP span and no constant-extrapolation warning fires.
        char, path = self._topopt(
            tmp_path, env,
            Receiver(depths=np.array([50.0]),
                     ranges=np.linspace(100.0, 8000.0, 5)))
        assert char == 'Q'
        assert path.with_suffix('.ssp').exists(), (
            "TopOpt 'Q' was written but the .ssp it makes Bellhop open was "
            "not staged")

    def test_range_independent_ssp_auto_picks_c_linear(self, tmp_path):
        from uacpy.core.ssp import SoundSpeedProfile
        env = Environment(
            name='ri-auto', bathymetry=100.0,
            ssp=SoundSpeedProfile.from_pairs([(0.0, 1500.0), (100.0, 1480.0)]))
        char, path = self._topopt(
            tmp_path, env,
            Receiver(depths=np.array([50.0]),
                     ranges=np.linspace(100.0, 2000.0, 5)))
        assert char == 'C'
        assert not path.with_suffix('.ssp').exists()


class TestCervenyKnobsOnNonCervenyBeams:
    """The Cerveny beam knobs exist in the deck only for ``beam_type`` 'C' /
    'R' (``ReadEnvironmentBell.f90`` reads the two extra lines for those
    alone). On any other beam type a non-default knob warns at construction
    (bellhop.md §5 "written to the env file only for the two") and the writer
    emits no Cerveny rows at all."""

    def test_non_cerveny_beam_warns_on_a_set_knob(self):
        with pytest.warns(UserWarning, match='Cerveny'):
            Bellhop(beam_type='B', beam_width_type='M')

    def test_cerveny_beam_accepts_the_knob_silently(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            model = Bellhop(beam_type='C', beam_width_type='M')
        assert model.beam_width_type == 'M'

    def test_deck_omits_the_cerveny_rows_for_non_cerveny_beams(self, tmp_path):
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        common = (Environment(bathymetry=100.0, ssp=1500.0),
                  Source(depths=25.0, frequencies=100.0),
                  Receiver(depths=np.array([50.0]),
                           ranges=np.linspace(100.0, 2000.0, 5)))
        for beam_type, path in (('C', tmp_path / 'cerveny.env'),
                                ('G', tmp_path / 'hat.env')):
            write_bellhop_env_file(path, *common, beam_type=beam_type,
                                   beam_width_type='M', beam_curvature='D')
        # The first Cerveny row opens with the quoted width+curvature pair.
        assert "'MD'" in (tmp_path / 'cerveny.env').read_text()
        assert "'MD'" not in (tmp_path / 'hat.env').read_text()


def test_bandwidth_factor_above_one_warns():
    """bellhop.md §6 "amplitudes are computed once at the band centre":
    arrival amplitudes are held frequency-flat, so a band wider than
    +/-50% of fc degrades toward the
    edges — ``bandwidth_factor > 1`` warns at construction; 1.0 is the
    documented practical limit and stays silent."""
    with pytest.warns(UserWarning, match='bandwidth_factor'):
        Bellhop(bandwidth_factor=1.5)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        Bellhop(bandwidth_factor=1.0)


class TestBroadbandFrequencyGridResolution:
    """A single centre frequency expands to ``n_freqs`` bins spanning
    ``fc*(1 +/- bandwidth_factor/2)``
    (DOCUMENTATION.md §15 "Bellhop synthesizes", base.py
    ``_resolve_broadband_frequencies``); explicit ``frequencies=`` wins over
    everything. Resolver-level — nothing runs."""

    def test_default_is_128_bins_over_half_fc(self):
        from uacpy.core.constants import DEFAULT_BROADBAND_N_FREQS
        assert DEFAULT_BROADBAND_N_FREQS == 128
        model = Bellhop(verbose=False)
        freqs = model._resolve_broadband_frequencies(
            Source(depths=50.0, frequencies=200.0), None,
            n_freqs=model.n_freqs, bandwidth_factor=model.bandwidth_factor)
        assert freqs.shape == (128,)
        assert freqs[0] == pytest.approx(150.0)
        assert freqs[-1] == pytest.approx(250.0)
        assert np.allclose(np.diff(freqs), freqs[1] - freqs[0])

    def test_constructor_knobs_shape_the_grid(self):
        model = Bellhop(verbose=False, n_freqs=16, bandwidth_factor=0.2)
        freqs = model._resolve_broadband_frequencies(
            Source(depths=50.0, frequencies=200.0), None,
            n_freqs=model.n_freqs, bandwidth_factor=model.bandwidth_factor)
        assert freqs.shape == (16,)
        assert freqs[0] == pytest.approx(180.0)
        assert freqs[-1] == pytest.approx(220.0)

    def test_explicit_frequencies_win(self):
        model = Bellhop(verbose=False, n_freqs=16)
        freqs = model._resolve_broadband_frequencies(
            Source(depths=50.0, frequencies=200.0),
            np.array([90.0, 100.0, 110.0]),
            n_freqs=model.n_freqs, bandwidth_factor=model.bandwidth_factor)
        np.testing.assert_array_equal(freqs, [90.0, 100.0, 110.0])


class TestRayBoxDefaults:
    """``z_box``/``r_box`` left ``None`` reach the deck as 1.2x the max depth
    / receiver range (DOCUMENTATION.md §18 "Max depth of the ray box",
    ``write_bellhop_env_file``); the
    range column is written in km (``ReadEnvironmentBell.f90:154`` converts
    back). Deck-token test — no binary in the loop."""

    @staticmethod
    def _box_line(tmp_path, **kwargs):
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        path = tmp_path / 'box.env'
        write_bellhop_env_file(
            path, Environment(bathymetry=100.0, ssp=1500.0),
            Source(depths=25.0, frequencies=100.0),
            Receiver(depths=np.array([50.0]),
                     ranges=np.linspace(100.0, 5000.0, 5)),
            **kwargs)
        lines = path.read_text().splitlines()
        # RunType record (position 6 holds the hard-coded dimension '2',
        # distinguishing it from a 7-character title), then n_beams, alpha,
        # step, box.
        i = next(i for i, ln in enumerate(lines)
                 if len(ln) == 9 and ln[0] == ln[-1] == "'"
                 and ln[6] == '2')
        z_str, r_km_str = lines[i + 4].split()
        return float(z_str), float(r_km_str)

    def test_defaults_are_1_2x_the_grid(self, tmp_path):
        z_box, r_box_km = self._box_line(tmp_path)
        assert z_box == pytest.approx(1.2 * 100.0)
        assert r_box_km == pytest.approx(1.2 * 5000.0 / 1000.0)

    def test_pinned_boxes_pass_through_verbatim(self, tmp_path):
        z_box, r_box_km = self._box_line(tmp_path, z_box=333.0, r_box=7000.0)
        assert z_box == pytest.approx(333.0)
        assert r_box_km == pytest.approx(7.0)


@pytest.mark.requires_binary
def test_broadband_metadata_carries_c0_and_the_arrivals_field():
    """One 1-bin broadband run, two documented metadata contracts:
    ``metadata['c0']`` is the sea-surface sound speed of the first profile
    (DOCUMENTATION.md §8 "the sea-surface sound speed of the first profile" —
    a physical speed, deliberately not 1500), and
    ``metadata['arrivals_field']`` keeps the :class:`Arrivals` the band was
    synthesised from so a different waveform needs no re-run.
    results.md §8 "under `metadata['arrivals_field']`, so you can re-synthesise"
    """
    from uacpy.core.ssp import SoundSpeedProfile
    env = Environment(
        name='c0-meta', bathymetry=100.0,
        ssp=SoundSpeedProfile.from_pairs([(0.0, 1490.0), (100.0, 1500.0)]))
    result = Bellhop(verbose=False).run(
        env, Source(depths=25.0, frequencies=200.0),
        Receiver(depths=np.array([50.0]), ranges=np.array([1000.0])),
        run_mode=RunMode.BROADBAND, frequencies=np.array([200.0]))
    assert result.metadata['c0'] == pytest.approx(1490.0)
    assert isinstance(result.metadata['arrivals_field'], Arrivals)


class TestBackendAutoSelectionPriority:
    """``backend=None`` auto-selects cuda > cxx > fortran, silently
    (DOCUMENTATION.md §7 "auto-selects the fastest installed binary",
    bellhop.md §5 "auto-picks in the order cuda"). ``_find_bellhop_executable``
    expresses the priority as the name order handed to
    ``_find_executable_in_paths``, which returns the first name that
    resolves; the fake below reproduces exactly that first-name-wins search
    over a directory of stub files, so the priority is pinned by
    construction alone — no binary is ever launched."""

    @staticmethod
    def _fake_find(available_dir):
        from uacpy.core.exceptions import ExecutableNotFoundError

        def fake(self, names, **kwargs):
            for name in ([names] if isinstance(names, str) else names):
                candidate = available_dir / name
                if candidate.exists():
                    return candidate
            raise ExecutableNotFoundError('Bellhop', repr(names))
        return fake

    @pytest.mark.parametrize('installed,expected_version,expected_name', [
        (('bellhopcuda', 'bellhopcxx', 'bellhop'), 'cuda', 'bellhopcuda'),
        (('bellhopcxx', 'bellhop'), 'cxx', 'bellhopcxx'),
        (('bellhop',), 'fortran', 'bellhop'),
    ])
    def test_auto_pick_prefers_cuda_then_cxx_then_fortran(
            self, tmp_path, monkeypatch, installed, expected_version,
            expected_name):
        for name in installed:
            stub = tmp_path / name
            stub.write_text('')
            # The resolver requires the execute bit, so a stub without it
            # is rejected as unrunnable before the priority order is read.
            stub.chmod(0o755)
        monkeypatch.setattr(Bellhop, '_find_executable_in_paths',
                            self._fake_find(tmp_path))
        # bellhop.md §7 "auto-pick never warns at all" — only an explicit
        # backend= that falls back does.
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            model = Bellhop(verbose=False)
        assert model.version == expected_version
        assert model._exe.name == expected_name


@pytest.mark.requires_binary
def test_gaussian_beams_picket_the_arrivals_and_hat_beams_do_not():
    """bellhop.md §6 "every physical path is resolved into a picket":
    with ``beam_type='B'`` every physical path
    resolves into a picket of neighbouring beams inside the beam window
    (2639 entries against 61 in the doc's 4000-beam example), while 'G'
    returns one arrival per path — which is why the user guide prescribes
    'G' for arrivals. Pinned as a count ratio so the fan size stays free."""
    env = Environment(name='picket', bathymetry=100.0, ssp=1500.0)
    src = Source(depths=25.0, frequencies=200.0)
    rcv = Receiver(depths=[60.0], ranges=[3000.0])
    counts = {}
    for beam_type in ('B', 'G'):
        arr = Bellhop(verbose=False, beam_type=beam_type, n_beams=800,
                      alpha=(-45.0, 45.0)).run(env, src, rcv,
                                               run_mode=RunMode.ARRIVALS)
        counts[beam_type] = int(arr.by_receiver[0][0][0]['n_arrivals'])
    assert counts['G'] >= 1
    assert counts['B'] > 2 * counts['G'], (
        f"'B' should picket each path across the beam window: {counts}")


def _ray_validity_messages(depth_m, sound_speed, frequency_hz):
    """Warnings from the D/lambda guard for one (depth, c, f) — the guard is
    called directly, so no Bellhop binary runs."""
    env = uacpy.Environment(bathymetry=depth_m, ssp=float(sound_speed),
                            bottom='sand')
    src = uacpy.Source(depths=depth_m / 4.0, frequencies=frequency_hz)
    return _messages(
        lambda: Bellhop()._warn_if_below_ray_validity(env, src),
        _D_OVER_LAMBDA)


class TestBellhopRayValidityFloor:
    """Bellhop's ``D/lambda >= 5`` validity floor, documented in
    ``docs/models/bellhop.md`` and ``docs/models/README.md`` and for a long
    time unenforced: at ``D/lambda = 1.07`` Bellhop read 10.1 to 17.6 dB below
    Kraken, silently.

    The threshold is pinned on BOTH sides. A mutation campaign found that a
    guard test using values far from the boundary pins that the guard fires,
    never where.
    """

    @pytest.fixture(autouse=True)
    def _forget_ray_validity_warnings(self):
        """``_WARNED_RAY_VALIDITY`` deduplicates per PROCESS, so a test that
        did not clear it would pass or fail on the selection order (the lesson
        ``test_input_validation.py`` records for ``_WARNED_MODEL_SOURCES``).

        Scoped to this class rather than the module: the rest of this file has
        no stake in that set, and an autouse fixture reaching them would be
        clearing global state under 180 unrelated tests.
        """
        _WARNED_RAY_VALIDITY.clear()
        yield
        _WARNED_RAY_VALIDITY.clear()

    def test_depth_of_exactly_five_wavelengths_is_accepted(self):
        """100 m at 75 Hz in 1500 m/s is D/lambda = 5 exactly. The sources put
        ✗ BELOW 5, so the floor itself is the accepting side."""
        assert _ray_validity_messages(100.0, 1500.0, 75.0) == []

    def test_depth_just_under_five_wavelengths_warns(self):
        msgs = _ray_validity_messages(100.0, 1500.0, 74.85)
        assert len(msgs) == 1
        assert 'Kraken' in msgs[0]

    def test_depth_just_over_five_wavelengths_is_accepted(self):
        assert _ray_validity_messages(100.0, 1500.0, 75.15) == []

    def test_the_cross_check_band_is_accepted(self):
        """5-20 asks for a second opinion, not a different model, so nothing
        in it warns — including its own upper edge."""
        for frequency in (75.1, 100.0, 200.0, 300.0):
            assert _ray_validity_messages(100.0, 1500.0, frequency) == []

    def test_the_audited_case_names_its_ratio(self):
        """80 m at 20 Hz — the geometry the round-22 audit measured 10.1 to
        17.6 dB away from Kraken."""
        msgs = _ray_validity_messages(80.0, 1500.0, 20.0)
        assert len(msgs) == 1
        assert 'D/lambda = 1.07' in msgs[0]
        assert '80 m at 20 Hz' in msgs[0]

    def test_the_threshold_constant_is_the_documented_five(self):
        assert _RAY_VALIDITY_D_OVER_LAMBDA == 5.0

    def test_the_lowest_frequency_of_a_band_decides(self):
        """Ray theory fails at the LONGEST wavelength, so a band straddling
        the floor is judged by its bottom — the opposite end from a
        resolution criterion."""
        env = uacpy.Environment(bathymetry=100.0, ssp=1500.0, bottom='sand')
        src = uacpy.Source(depths=25.0, frequencies=[50.0, 1000.0])
        assert len(_messages(
            lambda: Bellhop()._warn_if_below_ray_validity(env, src),
            _D_OVER_LAMBDA)) == 1

    def test_one_geometry_warns_once_per_process(self):
        for expected in (1, 0, 0):
            assert len(_ray_validity_messages(80.0, 1500.0, 20.0)) == expected

    def test_each_geometry_gets_its_own_warning(self):
        assert len(_ray_validity_messages(80.0, 1500.0, 20.0)) == 1
        assert len(_ray_validity_messages(80.0, 1500.0, 30.0)) == 1

    def test_the_documented_pekeris_quick_start_is_silent(self):
        """DOCUMENTATION.md section 3: 100 m, 1500 m/s, 100 Hz."""
        env = uacpy.Environment(
            name='Pekeris', bathymetry=100.0, ssp=1500.0,
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1600.0,
                density=1.5, attenuation=0.5))
        src = uacpy.Source(depths=50.0, frequencies=100.0)
        assert _messages(
            lambda: Bellhop()._warn_if_below_ray_validity(env, src),
            _D_OVER_LAMBDA) == []

    def test_the_docs_readme_quick_start_is_silent(self):
        """docs/README.md "Start here": 100 m, surface 1500 m/s, 200 Hz."""
        env = uacpy.Environment(
            bathymetry=100.0, ssp=[(0.0, 1500.0), (100.0, 1490.0)],
            bottom=uacpy.BoundaryProperties(
                acoustic_type='half-space', sound_speed=1650.0,
                density=1.8, attenuation=0.6))
        src = uacpy.Source(depths=25.0, frequencies=200.0)
        assert _messages(
            lambda: Bellhop()._warn_if_below_ray_validity(env, src),
            _D_OVER_LAMBDA) == []

    @pytest.mark.requires_binary   # constructs Bellhop (resolves its binary)
    @pytest.mark.parametrize('frequency,warns', [(20.0, True), (200.0, False)])
    def test_run_reaches_the_guard_before_the_binary(self, monkeypatch,
                                                     frequency, warns):
        """The guard has to sit on the real entry point, not just be
        callable. ``_run_bellhop`` is replaced with a raise, so the deck is
        built and the binary never runs: whatever the guard says, it has said
        by then. 80 m at 20 Hz is D/lambda = 1.07, at 200 Hz it is 10.7."""
        class _Stop(RuntimeError):
            pass

        monkeypatch.setattr(
            Bellhop, '_run_bellhop',
            lambda self, base_name, work_dir: (_ for _ in ()).throw(_Stop()))
        env = uacpy.Environment(bathymetry=80.0, ssp=1500.0, bottom='sand')
        src = uacpy.Source(depths=10.0, frequencies=frequency)
        rcv = uacpy.Receiver(depths=np.array([40.0]),
                             ranges=np.linspace(100.0, 5000.0, 10))

        def _attempt():
            with pytest.raises(_Stop):
                Bellhop(verbose=False).run(env, src, rcv)

        assert (len(_messages(_attempt, _D_OVER_LAMBDA)) == 1) is warns

    def test_an_environment_without_a_usable_sound_speed_is_silent(self):
        """A diagnostic never decides whether a run happens; the deck-validity
        guards own that. Same geometry either side — only ``c`` changes."""
        env = uacpy.Environment(bathymetry=100.0, ssp=1500.0, bottom='sand')
        src = uacpy.Source(depths=25.0, frequencies=20.0)
        assert len(_messages(
            lambda: Bellhop()._warn_if_below_ray_validity(env, src),
            _D_OVER_LAMBDA)) == 1
        env.ssp.data[:] = np.nan
        _WARNED_RAY_VALIDITY.clear()
        assert _messages(
            lambda: Bellhop()._warn_if_below_ray_validity(env, src),
            _D_OVER_LAMBDA) == []


@pytest.mark.requires_binary
class TestBellhopConstructorGuards:
    """Round-19 constructor guards mirroring the RAM siblings: n_beams
    integrality (the deck writer emits ``int(n_beams)``), alpha limit
    ordering, and step sign/finiteness."""

    def test_a_fractional_n_beams_raises_the_integer_guard(self):
        with pytest.raises(ConfigurationError, match='integer beam count'):
            Bellhop(verbose=False, n_beams=250.7)

    def test_a_string_n_beams_raises_the_integer_guard(self):
        with pytest.raises(ConfigurationError, match='integer beam count'):
            Bellhop(verbose=False, n_beams='many')

    def test_a_numpy_integer_n_beams_is_accepted(self):
        assert Bellhop(verbose=False, n_beams=np.int64(300)).n_beams == 300

    def test_reversed_alpha_limits_raise_the_ordering_guard(self):
        with pytest.raises(ConfigurationError, match='min_deg < max_deg'):
            Bellhop(verbose=False, alpha=(80, -80))

    def test_equal_alpha_limits_raise_the_ordering_guard(self):
        with pytest.raises(ConfigurationError, match='min_deg < max_deg'):
            Bellhop(verbose=False, alpha=(45, 45))

    def test_a_nan_alpha_limit_raises_the_ordering_guard(self):
        with pytest.raises(ConfigurationError, match='min_deg < max_deg'):
            Bellhop(verbose=False, alpha=(float('nan'), 80))

    def test_non_numeric_alpha_entries_raise_a_typed_error(self):
        with pytest.raises(ConfigurationError, match='numbers'):
            Bellhop(verbose=False, alpha=('a', 'b'))

    def test_a_negative_step_raises(self):
        with pytest.raises(ConfigurationError, match='>= 0 and finite'):
            Bellhop(verbose=False, step=-5.0)

    def test_an_infinite_step_raises(self):
        with pytest.raises(ConfigurationError, match='>= 0 and finite'):
            Bellhop(verbose=False, step=float('inf'))

    def test_zero_step_is_the_automatic_step_sentinel(self):
        assert Bellhop(verbose=False, step=0.0).step == 0.0


class TestBellhopComponentIsRayCentredOnly:
    """``Beam%Component`` has one use site in the solver: ``influence.f90:120``
    inside ``InfluenceCervenyRayCen`` (``beam_type='R'``).
    ``InfluenceCervenyCart`` (``'C'``) never reads it, while the writer still
    emits the letter and the .prt echoes it back. Where it *is* honoured the
    .shd holds particle velocity, which a Field can only report as pressure."""

    @pytest.mark.parametrize('beam_type', ['C', 'G', 'B'])
    @pytest.mark.parametrize('component', ['V', 'H'])
    def test_an_inert_component_letter_warns(self, beam_type, component):
        with pytest.warns(UserWarning, match='ignored for beam_type'):
            Bellhop(beam_type=beam_type, component=component, verbose=False)

    @pytest.mark.parametrize('component', ['V', 'H'])
    def test_a_honoured_velocity_component_is_refused(self, component):
        with pytest.raises(UnsupportedFeatureError, match='particle velocity'):
            Bellhop(beam_type='R', component=component, verbose=False)

    @pytest.mark.parametrize('beam_type', ['C', 'R', 'G'])
    def test_pressure_is_silent_everywhere(self, beam_type):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            Bellhop(beam_type=beam_type, component='P', verbose=False)
        assert not [w for w in caught if 'component' in str(w.message)]


class TestBellhopMinimumBeamFan:
    """``bellhop.f90:176-178`` zeroes ``Dalpha`` for a single beam, so every
    beam has zero width and the field is all-NaN. Two beams have a finite
    ``Dalpha`` and do produce a field — 69.1 dB on a deep-water direct path
    against a converged 66.1 dB, under-resolved but not degenerate, and no
    more so than the five-beam fan that reads 100.1 dB on the same
    geometry."""

    def test_one_beam_is_refused(self):
        with pytest.raises(ConfigurationError, match='n_beams=1'):
            Bellhop(n_beams=1, verbose=False)._check_beam_count_supports_run_mode(
                RunMode.COHERENT_TL)

    def test_two_beams_are_allowed(self):
        Bellhop(n_beams=2, verbose=False)._check_beam_count_supports_run_mode(
            RunMode.COHERENT_TL)

    def test_a_single_ray_trace_is_allowed(self):
        for mode in (RunMode.RAYS, RunMode.EIGENRAYS):
            Bellhop(n_beams=1,
                    verbose=False)._check_beam_count_supports_run_mode(mode)


@pytest.mark.requires_binary
@pytest.mark.slow
def test_bellhop_two_beam_fan_returns_a_usable_field():
    """The measurement behind :data:`_MIN_INFLUENCE_BEAMS`: a two-beam fan on
    a deep-water direct path returns a finite TL a few dB off a converged
    51-beam run — not the all-NaN field the guard claimed — so refusing it
    was refusing a working, merely under-resolved, configuration."""
    env = Environment(
        name='deep', bathymetry=5000.0, ssp=1500.0,
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=1800.0, density=1.8,
                                  attenuation=0.5))
    source = Source(depths=1000.0, frequencies=100.0)
    receiver = Receiver(depths=np.array([1000.0]), ranges=np.array([2000.0]))

    def _tl(n_beams):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = Bellhop(n_beams=n_beams, alpha=(-10, 10),
                             verbose=False).run(env, source, receiver,
                                                run_mode=RunMode.COHERENT_TL)
        return float(np.asarray(result.db, dtype=float).ravel()[0])

    two, converged = _tl(2), _tl(51)
    assert np.isfinite(two), "a two-beam fan is not degenerate"
    assert two == pytest.approx(converged, abs=5.0), (
        f"two-beam TL {two:.2f} dB against a converged {converged:.2f} dB")


class TestBellhopResolvesItsOwnRayStep:
    """``bellhop.f90:170-174`` substitutes ``depth/10`` for a zero ``deltas``
    — a fraction of the water depth rather than of any wavelength or gradient
    scale — and ``Step.f90:138-146``'s ``hInt`` only shortens a step at an
    SSP-layer crossing, which a near-horizontal refracted ray never makes. So
    in deep water the ray integrated tens of km at depth/10. Measured on Munk
    5000 m at 100 Hz, source and receiver at 1000 m, 10-100 km, against a
    converged 5 m step: depth/10 is 26.56 dB max / 6.48 rms out, depth/50
    1.90 / 0.46. depth/50 is also what AT's own deep-water reference deck
    picks (``tests/MunkRot/Munk.env`` writes 100.0 m in a 5500 m box).
    """

    @staticmethod
    def _env(depth):
        from uacpy.core import BoundaryProperties, Environment
        return Environment(
            bathymetry=depth, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=1.8,
                                      attenuation=0.5))

    @pytest.mark.parametrize('depth, expected', [(5000.0, 100.0),
                                                 (100.0, 2.0)])
    def test_an_unpinned_step_scales_with_the_water_depth(self, depth,
                                                          expected):
        from uacpy.models import Bellhop
        assert Bellhop(verbose=False)._resolve_step(
            self._env(depth)) == pytest.approx(expected)

    def test_a_pinned_step_reaches_the_deck_unchanged(self):
        from uacpy.models import Bellhop
        assert Bellhop(step=250.0, verbose=False)._resolve_step(
            self._env(5000.0)) == pytest.approx(250.0)

    def test_the_resolved_step_is_finer_than_the_binarys_own_default(self):
        # The whole point: whatever uacpy sends must be positive, so
        # bellhop.f90:170-174 never substitutes depth/10.
        from uacpy.models import Bellhop
        for depth in (100.0, 1000.0, 5000.0):
            resolved = Bellhop(verbose=False)._resolve_step(self._env(depth))
            assert 0.0 < resolved < depth / 10.0


class TestBellhopReportsAFanThatCannotReachAReceiver:
    """``angleMod.f90:58-61`` fills the launch fan strictly between the two
    ``alpha`` values, and BELLHOP's only under-resolution diagnostic
    (``bellhop.f90:252-258``) tests the beam COUNT — never whether the span
    reaches the receivers. So a geometry needing a steeper launch than the fan
    carries loses those paths silently.

    Measured on a 100 m isovelocity guide, source 10 m, receiver 90 m at 2 kHz
    with the angular resolution matched so only the span differs: at r = 10 m
    the direct path needs 82.9 deg and the default +/-80 deg fan reads
    64.59 dB against 39.54 dB for +/-89.9 deg, a 25.05 dB error; where the
    required angle is inside the fan (r >= 50 m, needing <= 58 deg) the two
    agree to 0.33 dB.
    """

    @staticmethod
    def _check(ranges, alpha=(-80.0, 80.0)):
        from uacpy.core import Receiver, Source
        from uacpy.models import Bellhop
        Bellhop(alpha=alpha, verbose=False)._warn_if_fan_misses_receivers(
            Source(depths=10.0, frequencies=2000.0),
            Receiver(depths=[90.0], ranges=ranges))

    def test_a_receiver_needing_a_steeper_launch_is_reported(self):
        # atan2(90 - 10, 10) = 82.9 deg, past the +/-80 deg default.
        with pytest.warns(UserWarning, match='launch angle outside'):
            self._check([10.0])

    def test_a_receiver_inside_the_fan_is_silent(self):
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter('error')
            self._check([500.0])

    def test_widening_the_fan_silences_it(self):
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter('error')
            self._check([10.0], alpha=(-89.9, 89.9))

    def test_the_report_names_an_angle_the_fan_really_misses(self):
        # An asymmetric fan misses at one end only; the reported angle has to
        # be one of the misses, not merely the largest in magnitude.
        with pytest.warns(UserWarning, match='steepest is -8') as rec:
            from uacpy.core import Receiver, Source
            from uacpy.models import Bellhop
            Bellhop(alpha=(0.0, 80.0), verbose=False)._warn_if_fan_misses_receivers(
                Source(depths=90.0, frequencies=2000.0),
                Receiver(depths=[10.0], ranges=[10.0]))
        assert 'steepest is -8' in str(rec[0].message)


class TestFanMissReportEqualsTheDenseAngleGrid:
    """The report carries three numbers, and only the first is extremal.

    The guard no longer materialises ``degrees(arctan2(zr - zs, rr))`` over the
    whole (source depth, receiver depth, range) grid — 1.5 GiB at
    20 x 500 x 10000. The four corner angles bound that grid exactly, so they
    decide WHETHER anything is outside; they do not decide HOW MANY pairs are
    (a count is not an extremal quantity) nor WHICH angle is steepest among the
    misses (a corner only when the fan spans the horizontal). A fan that
    excludes zero, which ``alpha=(17, 74)`` is and ``Bellhop.__init__``
    accepts, is where a corner-only shortcut names the wrong angle, so it is
    the case pinned hardest here.
    """

    @staticmethod
    def _dense(zs, zr, rr, lo, hi):
        """``(count, worst)`` straight off the full angle grid."""
        zs, zr, rr = (np.atleast_1d(np.asarray(a, dtype=float))
                      for a in (zs, zr, rr))
        needed = np.degrees(np.arctan2(zr[None, :, None] - zs[:, None, None],
                                       rr[None, None, :]))
        outside = (needed < lo) | (needed > hi)
        if not outside.any():
            return 0, None
        return (int(outside.sum()),
                float(needed[outside].flat[
                    int(np.argmax(np.abs(needed[outside])))]))

    GEOMETRIES = [
        ([10.0], [90.0], [10.0, 50.0, 500.0]),
        ([5.0, 60.0, 120.0], [1.0, 90.0, 200.0], [8.0, 30.0, 120.0, 4000.0]),
        ([50.0], np.linspace(0.0, 300.0, 17), np.linspace(2.0, 900.0, 23)),
        ([90.0], [10.0], [10.0]),
        ([40.0], [40.0], [1.0, 20.0]),          # every angle exactly 0
    ]
    FANS = [(-80.0, 80.0), (-89.9, 89.9), (0.0, 80.0), (17.0, 74.0),
            (-74.0, -17.0), (-5.0, 5.0)]

    @pytest.mark.parametrize('zs,zr,rr', GEOMETRIES)
    @pytest.mark.parametrize('lo,hi', FANS)
    def test_count_and_steepest_match_the_dense_grid(self, zs, zr, rr, lo, hi):
        from uacpy.models.bellhop import _fan_miss_count_and_worst
        zs, zr, rr = (np.atleast_1d(np.asarray(a, dtype=float))
                      for a in (zs, zr, rr))
        count, worst = self._dense(zs, zr, rr, lo, hi)
        got_count, got_worst = _fan_miss_count_and_worst(zs, zr, rr, lo, hi)
        assert got_count == count
        if count:
            assert got_worst == pytest.approx(worst, abs=1e-9)

    @pytest.mark.parametrize('lo,hi', FANS)
    def test_the_early_out_fires_exactly_when_nothing_is_outside(self, lo, hi):
        import warnings as _w
        from uacpy.core import Receiver, Source
        from uacpy.models import Bellhop
        zs, zr, rr = [30.0], [5.0, 150.0], [12.0, 60.0, 3000.0]
        count, _ = self._dense(zs, zr, rr, lo, hi)
        model = Bellhop(alpha=(lo, hi), verbose=False)
        with _w.catch_warnings(record=True) as rec:
            _w.simplefilter('always')
            model._warn_if_fan_misses_receivers(
                Source(depths=zs, frequencies=2000.0),
                Receiver(depths=zr, ranges=rr))
        fired = [w for w in rec if 'launch angle outside' in str(w.message)]
        assert bool(fired) is bool(count)

    def test_the_message_reports_the_dense_numbers_on_a_one_sided_fan(self):
        import re
        from uacpy.core import Receiver, Source
        from uacpy.models import Bellhop
        zs, zr, rr = [50.0], [10.0, 95.0, 240.0], [15.0, 45.0, 700.0]
        lo, hi = 17.0, 74.0
        count, worst = self._dense(zs, zr, rr, lo, hi)
        model = Bellhop(alpha=(lo, hi), verbose=False)
        with pytest.warns(UserWarning, match='launch angle outside') as rec:
            model._warn_if_fan_misses_receivers(
                Source(depths=zs, frequencies=2000.0),
                Receiver(depths=zr, ranges=rr))
        msg = str(rec[0].message)
        n_out, total = re.search(
            r'(\d+) of (\d+) source/receiver', msg).groups()
        assert (int(n_out), int(total)) == (count, len(zr) * len(rr))
        steepest = float(
            re.search(r'steepest is (-?[\d.]+) deg', msg).group(1))
        assert steepest == pytest.approx(round(worst, 1), abs=1e-9)


# ── the launch-angle sign convention ─────────────────────────────────────────
#
# Which way ``alpha`` points is a contract other code reads: it is why
# ``uacpy.plot.plot_beam_pattern`` runs its polar axes clockwise from due east,
# so a .sbp lobe is drawn over the water it ensonifies. If the sign ever
# flipped, that plot would silently mirror against the field beside it.

@pytest.mark.convention
def test_positive_launch_angle_traces_a_downward_ray():
    """``alpha > 0`` goes deeper — ``ray2D(1)%t = [COS(alpha), SIN(alpha)]/c``
    over a depth axis that is positive downward."""
    env = uacpy.Environment(bathymetry=2000.0, ssp=1500.0)
    src = uacpy.Source(depths=1000.0, frequencies=200.0)
    rcv = uacpy.Receiver(depths=np.linspace(0.0, 2000.0, 21),
                         ranges=np.linspace(0.0, 5000.0, 11))
    rays = Bellhop(verbose=False, alpha=(-10.0, 10.0), n_beams=3).run(
        env, src, rcv, run_mode='rays')

    by_angle = {round(ray['alpha']): np.asarray(ray['z']) for ray in rays.rays}
    assert by_angle[10][1] > by_angle[10][0]      # +10° dives
    assert by_angle[-10][1] < by_angle[-10][0]    # -10° climbs


@pytest.mark.convention
def test_downward_only_pattern_ensonifies_below_the_source():
    """A ``.sbp`` passing only positive angles moves energy to the deeper
    receiver, which is what the polar plot claims when it draws that lobe
    below the horizontal."""
    env = uacpy.Environment(bathymetry=2000.0, ssp=1500.0)
    rcv = uacpy.Receiver(depths=np.array([500.0, 1500.0]),
                         ranges=np.array([2000.0]))
    downward = np.array([[-180.0, -40.0], [-0.001, -40.0],
                         [0.0, 0.0], [180.0, 0.0]])
    model = Bellhop(verbose=False, alpha=(-80.0, 80.0), n_beams=2001)

    def below_minus_above(beam_pattern):
        src = uacpy.Source(depths=1000.0, frequencies=200.0,
                           beam_pattern=beam_pattern)
        tl = np.squeeze(np.asarray(model.run(env, src, rcv).tl))
        return float(tl[1] - tl[0])          # TL, so lower = louder

    # Against the omni control, because the bare sign of the difference is an
    # interference detail of these two points, not a directivity effect.
    swing = below_minus_above(downward) - below_minus_above(None)
    assert swing < -10.0, f"downward pattern shifted below-above by {swing:+.2f} dB"
    assert below_minus_above(downward) < 0.0   # below is the louder one


class TestDelayAndSumSaysWhatTheWindowLeavesOut:
    """A delay-and-sum drops an echo the window does not hold — it never
    folds — so the record looks complete whatever it left out. The window
    report names the omitted and clipped echoes and the energy they carried."""

    FS = 100000.0
    FC = 10000.0

    def _cell(self, *delays):
        n = len(delays)
        return dict(n_arrivals=n,
                    amplitudes=np.ones(n), phases=np.zeros(n),
                    delays=np.asarray(delays, dtype=float),
                    delays_imag=np.zeros(n),
                    n_top_bounces=np.zeros(n, dtype=int),
                    n_bot_bounces=np.zeros(n, dtype=int),
                    src_angles=np.zeros(n), rcv_angles=np.zeros(n))

    def _waveform(self, n=200):
        # 2 ms hann-windowed tone burst
        k = np.arange(n)
        return np.hanning(n) * np.sin(2 * np.pi * self.FC * k / self.FS)

    def _run(self, cell, **kw):
        from uacpy.models.bellhop import delayandsum
        return _messages(lambda: delayandsum(
            rcv_arrivals=cell, source_timeseries=self._waveform(),
            sample_rate=self.FS, fc=self.FC, **kw),
            "does not hold every echo")

    def test_an_echo_beyond_the_window_is_counted_with_its_energy(self):
        # Two equal echoes; the window closes before the second begins, so
        # it is omitted and takes half the energy with it: -3 dB.
        msgs = self._run(self._cell(0.010, 0.050),
                         t_start=0.0, time_window=0.030)
        assert len(msgs) == 1, msgs
        assert "1 echo(es) fall entirely outside" in msgs[0]
        assert "-3 dB of the received energy" in msgs[0]

    def test_a_window_holding_every_echo_says_nothing(self):
        assert self._run(self._cell(0.010, 0.020),
                         t_start=0.0, time_window=0.040) == []

    def test_the_auto_window_holds_everything(self):
        assert self._run(self._cell(0.010, 0.050)) == []

    def test_a_cut_leading_edge_is_named_as_such(self):
        # The window opens 1 ms into the first echo's 2 ms waveform.
        msgs = self._run(self._cell(0.010, 0.020),
                         t_start=0.011, time_window=0.040)
        assert len(msgs) == 1, msgs
        assert "1 echo(es) begin before it and lose their leading edge" in msgs[0]

    def test_an_echo_running_past_the_end_loses_its_tail(self):
        # The second echo starts 1 ms before a window that ends at 30 ms.
        msgs = self._run(self._cell(0.010, 0.029),
                         t_start=0.0, time_window=0.030)
        assert len(msgs) == 1 and "run past its end" in msgs[0], msgs

    def test_a_report_dict_totals_the_cells_and_stays_silent(self):
        from uacpy.models.bellhop import delayandsum
        report = {}
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            for cell in (self._cell(0.010, 0.050), self._cell(0.012, 0.060)):
                delayandsum(rcv_arrivals=cell,
                            source_timeseries=self._waveform(),
                            sample_rate=self.FS, fc=self.FC,
                            t_start=0.0, time_window=0.030, report=report)
        assert not [w for w in rec if "does not hold every echo"
                    in str(w.message)]
        assert report['omitted'] == 2
        assert report['omitted_power'] == pytest.approx(2.0)
        assert report['total_power'] == pytest.approx(4.0)


@pytest.mark.requires_binary
class TestTheDefaultBroadbandGridReportsWhatItFolds:
    """The default BROADBAND grid — 128 bins over fc·(1 ± 0.25) — sets Δf
    from the carrier alone, a record 254/fc s long. The arrivals are in hand
    when H(f) is built, so the run measures them against that record."""

    def _geometry(self):
        # 300 m of water, source and receiver mid-column 1 km apart: the
        # surface and bottom bounces arrive ~29 ms after the direct path,
        # against a 254/40e3 = 6.35 ms default record.
        env = uacpy.Environment(name='fold', bathymetry=300.0, ssp=1500.0)
        source = Source(depths=150.0, frequencies=40e3)
        receiver = Receiver(depths=[150.0], ranges=[1000.0])
        return env, source, receiver

    def test_the_default_grid_names_the_arrivals_it_folds(self):
        env, source, receiver = self._geometry()
        msgs = _messages(lambda: Bellhop(verbose=False).run(
            env, source, receiver, run_mode=RunMode.BROADBAND),
            "fold back onto the early trace")
        assert len(msgs) == 1, msgs
        assert msgs[0].startswith(
            "Bellhop.run(run_mode=BROADBAND): a 0.00635 s record"), msgs[0]
        assert "Arrivals.synthesis_band" in msgs[0]

    def test_a_grid_the_caller_chose_is_not_second_guessed(self):
        env, source, receiver = self._geometry()
        grid = np.linspace(30e3, 50e3, 128)     # the same spacing, chosen
        assert _messages(lambda: Bellhop(verbose=False).run(
            env, source, receiver, run_mode=RunMode.BROADBAND,
            frequencies=grid), "fold back onto the early trace") == []
