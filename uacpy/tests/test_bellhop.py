"""Bellhop ray/beam-focused tests.

Run modes, beam types and source geometry, the env-record ordering the binary
demands, and the validity floor below which ray theory stops describing the
field at all.

Everything in this file is marked ``requires_binary``: constructing ``Bellhop``
resolves its executable, so even the tests that never launch a run need the
install to be present.
"""

import os
import warnings

import re
import pytest
import numpy as np

import uacpy
from uacpy.models import Bellhop
from uacpy.models.bellhop import _backend as bellhop_backend
from uacpy import Field
from uacpy.core.results import Rays, Arrivals, Result, ResultStack
from uacpy.core.run_settings import RunMode
from uacpy.models.bellhop._backend import (
    arrivals_need_merge, build_command, warn_on_engine_stdout_warnings,
)
from uacpy.models.bellhop._output import warn_if_arrival_table_filled
from uacpy.models.bellhop._plan import (
    eigenray_beam_count, fan_miss_notice, resolve_ray_step,
)
from uacpy.acoustic_signal import arrival_grid_transfer_function
from uacpy.models.bellhop._synthesis import warn_if_attenuation_extrapolates
from uacpy.tests.conftest import make_pekeris
from uacpy.tests.conftest import warning_messages
from uacpy.tests.conftest import recorded_warnings
from uacpy.core.acoustics import no_energy_mask


def _cell_tf(cell, frequencies, **kwargs):
    """``H(f)`` of one Bellhop arrival cell
    (:func:`~uacpy.acoustic_signal.arrival_grid_transfer_function`)."""
    return arrival_grid_transfer_function(frequencies, [[cell]],
                                          **kwargs)[0, 0]
from uacpy.core import (
    Environment, Source, Receiver, BoundaryProperties,
)
from uacpy.core.exceptions import (
    ConfigurationError, UnsupportedFeatureError,
)

pytestmark = pytest.mark.requires_binary

#: RunType(1:1) letters the writer can emit, and the source-geometry /
#: grid-type letters that sit at positions 4 and 5. Together they identify the
#: RunType record among the deck's quoted 7-character lines — position 6 does
#: NOT, because it is blank: ``io/bellhop_writer.py`` writes ' ' there rather
#: than '2', the one character that means plain 2-D to the Fortran engine AND
#: to bellhopcxx/bellhopcuda (a literal '2' makes those two warn about Nx2D).
#: The default title 'unnamed' is the other quoted 7-character line and is
#: rejected by every one of the three tests.
_RUN_TYPE_LETTERS = set('CSIEAaR')
_SOURCE_TYPE_LETTERS = set('RXS')
_GRID_TYPE_LETTERS = set('RI')


def _run_type_record(lines):
    """Index of the deck's RunType record, and its 7 characters."""
    hits = [(i, ln[1:8]) for i, ln in enumerate(lines)
            if len(ln) == 9 and ln[0] == ln[-1] == "'"
            and ln[1] in _RUN_TYPE_LETTERS
            and ln[4] in _SOURCE_TYPE_LETTERS
            and ln[5] in _GRID_TYPE_LETTERS]
    assert len(hits) == 1, (hits, lines)
    return hits[0]


def _say(notice):
    """Announce a stage-3 notice (``None``: nothing) as ``run`` does: one
    ``UserWarning``."""
    if notice is not None:
        warnings.warn(notice, UserWarning)


def _say_fan_miss(model, source, receiver):
    """Announce ``model``'s launch-fan notice for ``source``/``receiver``."""
    _say(fan_miss_notice(source, receiver, launch_angles_deg=model.launch_angles_deg,
                         grid_type=model.grid_type,
                         model_name=model.model_name))


def _merge(model):
    """Whether ``model`` reads its ``.arr`` with the pair-merge."""
    return arrivals_need_merge(backend=model._resolved_backend, exe=model._exe,
                               model_name=model.model_name)


def _command(model, base_name, **kw):
    """The argv ``model`` launches the deck ``base_name`` with."""
    return build_command(model._exe, base_name, backend=model._resolved_backend,
                         dimensionality=model.dimensionality, **kw)


def _scan_stdout(model, stdout):
    """``model``'s scan of an engine's stdout for the ports' diagnoses."""
    warn_on_engine_stdout_warnings(stdout, model_name=model.model_name,
                                   backend=model._resolved_backend)


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
        assert np.all(result.dB > 0), "TL should be positive"

    @pytest.mark.requires_binary
    def test_r0_column_is_no_data_nan(self, setup_env, setup_source):
        """``Bellhop/influence.f90:786-787`` sets a point source's
        cylindrical spreading factor to 0 at ``r = 0``, "avoid /0 at origin",
        so the binary writes exact zero pressure there that is no field
        value. The Bellhop wrapper returns those cells as NaN no-data rather
        than as the no-energy level a literal reading gives."""
        rcv = Receiver(depths=np.array([25.0, 50.0, 75.0]),
                       ranges=np.array([0.0, 1000.0, 3000.0]))
        result = Bellhop(verbose=False).run(
            env=setup_env, source=setup_source, receiver=rcv,
            run_mode=RunMode.COHERENT_TL)
        tl = np.asarray(result.dB)
        assert np.all(np.isnan(tl[:, 0]))
        assert np.all(np.isfinite(tl[:, 1:]))

    @staticmethod
    def _origin_rig():
        env = Environment(
            name='origin', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.5,
                                      attenuation=0.5))
        rcv = Receiver(depths=np.array([10.0, 25.0, 50.0, 90.0]),
                       ranges=np.linspace(0.0, 2000.0, 5))
        return env, rcv

    @pytest.mark.requires_binary
    @pytest.mark.parametrize('beam_type,written', [('G', False),
                                                   ('S', True)])
    def test_a_line_source_r0_column_is_nan_where_no_beam_loop_reaches_it(
            self, beam_type, written):
        """A line source's factor has no ``1/sqrt(r)``
        (``influence.f90:784``), so its ``r = 0`` column is a value exactly
        when the beam type's receiver loop reaches it: ``'S'`` starts at
        ``ir = 1`` (``:672``), ``'G'`` at the first receiver right of the
        ray start (``:419``) and never evaluates ``r = 0``."""
        env, rcv = self._origin_rig()
        field = Bellhop(verbose=False, beam_type=beam_type).run(
            env, Source(depths=25.0, frequencies=200.0, source_type='line'),
            rcv, run_mode=RunMode.COHERENT_TL)
        origin = np.asarray(field.data)[:, 0]
        if written:
            assert np.isfinite(origin).all()
            assert np.abs(origin).max() > 0.0
        else:
            assert np.isnan(origin).all()

    @pytest.mark.requires_binary
    def test_an_exact_zero_in_a_shadow_zone_is_the_engines_empty_beam_sum(
            self, tmp_path):
        """A cell no beam reaches keeps the ``U = 0.0`` the grid starts from
        (``bellhop.f90:240-241``): Bellhop's answer, not a skipped cell. Under
        a strong downward gradient the near-surface column at 4 km is such a
        shadow; the Field holds those zeros and their TL is the no-energy
        marker, not NaN."""
        from uacpy.core.constants import NO_ENERGY_DB
        from uacpy.io.oalib_reader import read_shd_bin
        env = Environment(
            name='shadow', bathymetry=1000.0,
            ssp=np.array([[0.0, 1540.0], [200.0, 1480.0], [1000.0, 1480.0]]),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.5,
                                      attenuation=0.5))
        rcv = Receiver(depths=np.linspace(0.0, 100.0, 4),
                       ranges=np.linspace(4000.0, 20000.0, 5))
        field = Bellhop(verbose=False, launch_angles_deg=(-20.0, 20.0),
                        work_dir=tmp_path, cleanup=False).run(
            env, Source(depths=20.0, frequencies=200.0), rcv,
            run_mode=RunMode.COHERENT_TL)
        shd = read_shd_bin(field.metadata['shd_file'])
        columns = [int(np.argmin(np.abs(shd.receiver_ranges - r)))
                   for r in rcv.ranges]
        empty = shd.pressure[0, 0][:, columns] == 0
        assert empty.any(), "the rig must hold a cell no beam reaches"
        data = np.asarray(field.data)
        assert not np.isnan(data).any()
        assert (data[empty] == 0).all()
        assert (np.asarray(field.dB)[empty] == NO_ENERGY_DB).all()

    @pytest.mark.requires_binary
    @pytest.mark.parametrize('source_type,beam_type,announced', [
        ('point', 'G', True), ('point', 'S', True),
        ('line', 'G', True), ('line', 'S', False)])
    def test_the_r0_notice_follows_whether_bellhop_writes_r0(
            self, source_type, beam_type, announced):
        env, rcv = self._origin_rig()
        src = Source(depths=25.0, frequencies=200.0, source_type=source_type)
        notices = [n.message for n in Bellhop(
            verbose=False, beam_type=beam_type).run_settings(
                env, src, rcv).engine.notices]
        assert any('starts at r=0 m' in n for n in notices) is announced

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
        # An incoherent magnitude sum has no phase, so it is stored as real
        # dB TL, as Kraken and Scooter store the same run mode.
        assert not np.iscomplexobj(result.data)
        assert result.unit == 'dB'

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
        # miss_distance is attached by the filter helper (target defaults
        # to the run's receiver point), not by the reader itself.
        close = result.filter_by_miss_distance(50.0)
        assert len(close.rays) > 0
        assert min(r['miss_distance'] for r in close.rays) < 50.0

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
        miss = [r['miss_distance'] for r in top4.rays]
        if len(miss) > 1:
            assert miss == sorted(miss)

    @pytest.mark.requires_binary
    def test_rays_filter_helpers_preserve_is_eigen(self, setup_env, setup_source, setup_receiver):
        """Rays.filter / filter_by_bounces / window preserve is_eigen."""
        bellhop = Bellhop(verbose=False)
        rays = bellhop.run(
            env=setup_env, source=setup_source, receiver=setup_receiver,
            run_mode=RunMode.RAYS,
        )
        assert rays.is_eigen is False

        custom = rays.filter(lambda r: True)
        assert custom.is_eigen is False
        assert len(custom.rays) == len(rays.rays)

        sub = rays.window(launch_angle_deg=(-5.0, 5.0))
        assert sub.is_eigen is False
        assert all(-5.0 <= r['launch_angle'] <= 5.0 for r in sub.rays)

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

        with pytest.raises(ConfigurationError,
                           match="bounce filter: kind='bogus' not in"):
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
        # resampling (models/_band.py broadband_band): 3 depths x
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
    understates the true error. Only a Biological law is scaled linearly
    (the separable laws scale exactly), so the layer carries the check."""

    def test_off_centre_explicit_band_reports_the_traced_fc_bound(self):
        from uacpy.core.absorption import (
            Biological, _band_absorption_error_dB_per_km)
        bio = Biological(layers=[(20.0, 80.0, 16000.0, 4.0, 0.05)])
        env = Environment(
            name='bio', bathymetry=100.0, ssp=1500.0, absorption=bio,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.8,
                                      attenuation=0.5))
        source = Source(depths=50.0, frequencies=10000.0)
        receiver = Receiver(depths=[50.0], ranges=[200.0])
        band = np.array([8000.0, 16000.0])
        depths = np.linspace(0.0, 100.0, 65)
        traced = _band_absorption_error_dB_per_km(bio, band, 10000.0,
                                                  depths=depths)
        midpoint = _band_absorption_error_dB_per_km(bio, band, 12000.0,
                                                    depths=depths)
        # Traced carrier 10 kHz: 0.768 dB/km; the 12 kHz midpoint would read
        # 0.700.
        assert f"{traced:.3g}" == '0.768' and f"{midpoint:.3g}" == '0.7'
        with pytest.warns(UserWarning,
                          match=r'traced once at 1e\+04 Hz'
                                r'(?s:.*)0\.768 dB/km'):
            Bellhop(verbose=False).run(
                env, source, receiver, run_mode=RunMode.BROADBAND,
                frequencies=band)


class TestRunWithBounceConstructorPlumbing:
    """Verify Bellhop.run_with_bounce passes through volume-attenuation /
    c_low / c_high to the spawned Bounce instance."""

    def test_bounce_sees_env_absorption(self, monkeypatch):
        """env.absorption (Francois-Garrison) flows through Bellhop's
        BOUNCE route into the deck the Bounce launch writes, with the knobs
        the call passed."""
        from uacpy.models import bounce as bounce_mod
        from uacpy.core.absorption import FrancoisGarrison

        captured = {}

        def spy_write(self_, inputs):
            captured['absorption'] = inputs.env.absorption
            captured['knobs'] = (self_.c_low, self_.c_high, self_.rmax)
            raise RuntimeError("stop after Bounce.run capture")

        monkeypatch.setattr(bounce_mod.Bounce, '_write_input', spy_write)

        fg = FrancoisGarrison(temperature=10.0, salinity=35.0, pH=8.0)
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
        assert captured['absorption'].temperature == 10.0
        assert captured['absorption'].salinity == 35.0
        assert captured['knobs'] == (1450.0, 20000.0, 42000.0)


def test_run_with_bounce_takes_rmax():
    import inspect
    params = inspect.signature(Bellhop.run_with_bounce).parameters
    assert 'rmax' in params and 'rmax_m' not in params


class TestFrancoisGarrisonValidation:
    """FrancoisGarrison validates its own params at construction."""

    def test_francois_garrison_goes_into_the_ssp_rows(self):
        from uacpy.core.absorption import FrancoisGarrison
        fg = FrancoisGarrison(temperature=10.0, salinity=35.0, pH=8.0)
        from uacpy.io.at_codes import (
            volume_attenuation_code, writes_alpha_per_ssp_row)
        assert volume_attenuation_code(fg) == ' '
        assert writes_alpha_per_ssp_row(fg)


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
        tl = np.asarray(res.dB)
        # Cells no beam reached carry the no-energy marker; below-seafloor
        # receivers are NaN.
        real = tl[np.isfinite(tl) & ~no_energy_mask(tl)]
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
        tl = np.asarray(res.dB)
        real = tl[(tl > 0) & (tl < 500)]
        assert real.size > tl.size * 0.5
        assert real.max() < 200


# ---------------------------------------------------------------------
# Bellhop multi-source-depth: the result each mode returns over several depths.
# ---------------------------------------------------------------------

class TestBellhopMultiSourceDepth:
    """Bellhop's ``.shd`` carries one field per source depth. A
    single-source TL run returns a :class:`Field`; a multi-source TL run
    returns one :class:`Field` whose leading ``source_depth`` axis holds
    the field of each depth. RAYS, ARRIVALS and EIGENRAYS over several
    depths return a :class:`ResultStack` of per-depth results."""

    @pytest.mark.requires_binary
    def test_multi_source_returns_one_field_over_source_depth(self):
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
        assert isinstance(result, Field)
        assert list(result.coords) == ['source_depth', 'depth', 'range']
        np.testing.assert_allclose(result.coords['source_depth'],
                                   [30.0, 50.0, 70.0])
        np.testing.assert_allclose(result.source_depths, [30.0, 50.0, 70.0])
        assert result.data.shape == (3, 9, 11)
        # Each depth's plane is a 2-D Field on the same receiver grid.
        for i in range(3):
            plane = result.isel(source_depth=i)
            assert isinstance(plane, Field)
            assert plane.data.shape == (9, 11)

    @pytest.mark.requires_binary
    def test_single_source_returns_pressure_field(self):
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
        assert 'source_depth' not in result.coords
        assert result.data.shape == (9, 11)

    @pytest.mark.requires_binary
    def test_multi_source_per_depth_tl_physically_plausible(self):
        """The plane of each source depth produces finite,
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
        field = bh.run(env, source, receiver, run_mode=RunMode.COHERENT_TL)
        assert field.data.shape == (3, 9, 10)
        for i in range(3):
            plane = field.isel(source_depth=i)
            assert plane.data.shape == (9, 10)
            tl = plane.dB
            real = tl[(tl > 0) & (tl < 500)]
            assert real.size > tl.size * 0.5
            assert real.min() > 0
            assert real.max() < 200

    @pytest.mark.requires_binary
    def test_at_source_depth_recovers_the_middle_depth(self):
        """``field.at(source_depth=z)`` returns the :class:`Field` of the
        matching source depth, and its TL matches a single-source run at
        the same depth (within Bellhop's beam-partitioning round-off)."""
        env = Environment(
            name='multi-src-at', bathymetry=100.0, ssp=1500.0,
        )
        receiver = Receiver(
            depths=np.linspace(10.0, 90.0, 9),
            ranges=np.linspace(500.0, 5000.0, 10),
        )
        bh = Bellhop(verbose=False)
        field = bh.run(
            env,
            Source(depths=[30.0, 50.0, 70.0], frequencies=100.0),
            receiver, run_mode=RunMode.COHERENT_TL,
        )
        plane = field.at(source_depth=50.0)
        assert isinstance(plane, Field)
        assert plane.data.shape == (9, 10)

        single = bh.run(
            env, Source(depths=50.0, frequencies=100.0),
            receiver, run_mode=RunMode.COHERENT_TL,
        )
        np.testing.assert_allclose(plane.dB, single.dB, rtol=1e-4, atol=1e-3)

    @pytest.mark.requires_binary
    def test_multi_source_rays_returns_stack(self):
        """RAYS mode: one binary call, reader splits the deterministic
        ``NSz × Nalpha`` block layout into one :class:`Rays` slab per
        source depth."""
        from uacpy.core.results import Rays, ResultStack
        env = Environment(name='multi-rays', bathymetry=100.0, ssp=1500.0)
        receiver = Receiver(depths=np.array([30.0, 60.0]),
                            ranges=np.array([200.0, 1000.0]))
        bh = Bellhop(verbose=False, n_beams=20, launch_angles_deg=(-30, 30))
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
        bh = Bellhop(verbose=False, n_beams=20, launch_angles_deg=(-30, 30))
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
        bh = Bellhop(verbose=False, n_beams=20, launch_angles_deg=(-30, 30))
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


def test_the_time_series_record_is_placed_by_run_alone():
    """``output_duration`` and ``t_start`` place the TIME_SERIES record per
    call, as on every engine; the constructor takes neither."""
    import inspect
    ctor = inspect.signature(Bellhop.__init__).parameters
    run = inspect.signature(Bellhop.run).parameters
    assert {'output_duration', 't_start'} <= set(run)
    assert not {'time_window', 't_start'} & set(ctor)


@pytest.mark.parametrize('backend', ['cuda', 'cxx'])
def test_an_explicit_bellhop_backend_without_its_binary_raises(monkeypatch,
                                                               backend):
    """An explicitly requested backend that isn't built raises, naming the
    ``install.sh`` flag that builds it; no other engine runs in its place.
    The cxx/cuda variants are hidden so only ``bellhop`` resolves, and the
    automatic pick on the same install resolves to that Fortran binary."""
    from uacpy.core.exceptions import ExecutableNotFoundError
    fortran = Bellhop(backend='fortran', verbose=False)._exe   # real bellhop path

    def fake_find(names, **kw):
        names = list(names) if isinstance(names, (list, tuple)) else [names]
        if 'bellhop' in names:
            return fortran
        raise ExecutableNotFoundError('Bellhop', repr(names))

    monkeypatch.setattr(bellhop_backend, 'find_executable_in_paths',
                        fake_find)
    with pytest.raises(ExecutableNotFoundError,
                       match=rf'install\.sh --bellhop {backend}'):
        Bellhop(backend=backend, verbose=False)
    auto = Bellhop(verbose=False)
    assert auto._resolved_backend == 'fortran'
    assert auto.backend is None


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
        with pytest.raises(UnsupportedFeatureError,
                           match="does not support: dimensionality='3D'"):
            Bellhop(dimensionality='3D')

    def test_dimensionality_arbitrary_raises(self):
        with pytest.raises(UnsupportedFeatureError,
                           match="does not support: dimensionality='foo'"):
            Bellhop(dimensionality='foo')

    def test_dimensionality_2d_ok(self):
        assert Bellhop(dimensionality='2D').dimensionality == '2D'

    def test_invalid_beam_type_raises(self):
        # Unknown letters silently map to geometric-hat in the Fortran reader.
        with pytest.raises(ConfigurationError,
                           match="is not a known beam type"):
            Bellhop(beam_type='Q')

    @pytest.mark.parametrize('bt', ['B', 'R', 'C', 'g', 'G', 'S'])
    def test_valid_beam_types_ok(self, bt):
        assert Bellhop(beam_type=bt).beam_type == bt

    def test_invalid_grid_type_raises(self):
        with pytest.raises(ConfigurationError,
                           match=r"grid_type='Q'\) is not valid"):
            Bellhop(grid_type='Q')

    def test_launch_angles_wrong_length_raises(self):
        with pytest.raises(ConfigurationError,
                           match="must be a 2-element sequence"):
            Bellhop(launch_angles_deg=(-80, 0, 80))

    def test_launch_angles_scalar_raises(self):
        with pytest.raises(ConfigurationError,
                           match="must be a 2-element sequence"):
            Bellhop(launch_angles_deg=80)


class TestBellhopSourceGeometry:
    """Source geometry is read off the Source carrier (spec 2026-07-25)."""

    @staticmethod
    def _env():
        return Environment(bathymetry=200.0, ssp=1500.0)

    def test_constructor_rejects_source_type(self):
        with pytest.raises(TypeError,
                           match="unexpected keyword argument 'source_type'"):
            Bellhop(source_type='R')

    def test_constructor_rejects_beam_pattern_file(self):
        with pytest.raises(
                TypeError,
                match="unexpected keyword argument 'source_beam_pattern_file'"):
            Bellhop(source_beam_pattern_file=None)

    def test_line_source_differs_from_point_source(self):
        env = self._env()
        rcv = Receiver(depths=100.0, ranges=np.linspace(100, 5000, 60))
        model = Bellhop(verbose=False)
        pt = model.run(env, Source(depths=50, frequencies=200,
                                   source_type='point'), rcv)
        ln = model.run(env, Source(depths=50, frequencies=200,
                                   source_type='line'), rcv)
        delta = np.nanmax(np.abs(np.asarray(pt.dB) - np.asarray(ln.dB)))
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
                                              source_type='point'), rcv).dB).ravel()
        ln = np.asarray(model.run(env, Source(depths=50, frequencies=200,
                                              source_type='line'), rcv).dB).ravel()
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
        from uacpy.core.boundary import SedimentLayer
        from uacpy.core.bottom import Bottom, SeabedColumn
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
        assert 'bounce' in result.components

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


class TestEnvRecordOrder:
    """``.env`` record order against ``ReadEnvironmentBell.f90``.

    ``:59 CALL ReadTopOpt`` reads the Francois-Garrison / biological rows
    itself (``:308-320``); the top half-space row is read only afterwards by
    ``:69 CALL TopBot`` (``:474``). Emitting them the other way round feeds the
    F-G row to the half-space reader and kills the run in ``AttenMod : CRCI``.
    """

    @staticmethod
    def _ice_env(absorption=None):
        from uacpy.core.absorption import FrancoisGarrison
        from uacpy.core.environment import BoundaryProperties
        return Environment(
            name='ice-fg', bathymetry=100.0, ssp=1500.0,
            surface=BoundaryProperties(
                acoustic_type='half-space', sound_speed=3500.0, density=0.9,
                attenuation=0.1, shear_speed=1800.0, shear_attenuation=0.2),
            absorption=(FrancoisGarrison(temperature=4.0, salinity=34.0,
                                         pH=8.0)
                        if absorption is None else absorption),
        )

    def _order_lines(self, tmp_path, env):
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        path = tmp_path / 'order.env'
        write_bellhop_env_file(
            path, env, Source(depths=25.0, frequencies=100.0),
            Receiver(depths=np.array([50.0]),
                     ranges=np.linspace(100.0, 5000.0, 10)))
        lines = path.read_text().splitlines()
        return lines, next(i for i, ln in enumerate(lines)
                           if ln.startswith("'CAW"))

    def test_absorption_block_precedes_top_halfspace(self, tmp_path):
        from uacpy.core.absorption import Biological
        lines, topopt = self._order_lines(tmp_path, self._ice_env(
            Biological(layers=[(10.0, 20.0, 400.0, 5.0, 0.1)])))
        assert lines[topopt][4] == 'B'
        # The bio-layer count and row first, then the elastic half-space row.
        assert lines[topopt + 1].strip() == '1'
        assert [float(v) for v in lines[topopt + 2].split()] == [
            10.0, 20.0, 400.0, 5.0, 0.1]
        assert lines[topopt + 3].split()[:3] == ['0.00', '3500.000000',
                                                 '1800.000000']

    def test_francois_garrison_writes_no_row_before_the_halfspace(
            self, tmp_path):
        lines, topopt = self._order_lines(tmp_path, self._ice_env())
        assert lines[topopt][4] == ' '
        assert lines[topopt + 1].split()[:3] == ['0.00', '3500.000000',
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
        assert np.any(np.isfinite(np.asarray(result.dB)))


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
            ssp=SoundSpeedProfile(depths=cls.Z, sound_speed=data,
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
    def test_a_pinned_quad_on_a_range_independent_ssp_runs_as_quad(self):
        """The caller's 'quad' is honoured, not swapped for 'linear': with
        one profile held in range, Quad's field is C-linear's."""
        env = Environment(name='ri', bathymetry=100.0,
                          ssp=np.array([[0.0, 1500.0], [100.0, 1480.0]]),
                          bottom=BoundaryProperties(
                              acoustic_type='half-space', sound_speed=1600.0,
                              density=1.5, attenuation=0.5))
        src = Source(depths=25.0, frequencies=200.0)
        rcv = Receiver(depths=np.array([20.0, 60.0]),
                       ranges=np.linspace(500.0, 3000.0, 6))
        fields = {}
        for interp in ('quad', 'linear'):
            with recorded_warnings() as caught:
                fields[interp] = Bellhop(
                    verbose=False, interp_ssp=interp, backend='fortran').run(
                    env, src, rcv, run_mode=RunMode.COHERENT_TL)
            assert not [w for w in caught if 'quad' in str(w.message)]
        assert fields['quad'].run_settings.engine.interp_ssp == 'quad'
        np.testing.assert_allclose(np.asarray(fields['quad'].dB),
                                   np.asarray(fields['linear'].dB), atol=0.01)

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
        assert np.any(np.isfinite(np.asarray(result.dB)))


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
        assert np.any(np.isfinite(np.asarray(result.dB)))


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
                     ranges=np.linspace(100.0, 2000.0, 5)), beam_type='B')
        deck = path.read_text()
        assert deck.count("'CB ") == 1, deck
        path.write_text(deck.replace("'CB ", "'Cb "))
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
                    source_angles_deg=np.array([0.0, 5.0]),
                    receiver_angles_deg=np.array([0.0, 5.0]))

    def test_transfer_function_phase_offset(self):
        f = np.linspace(90.0, 110.0, 8)
        base = _cell_tf(self._cell(), f)
        shifted = _cell_tf(self._cell(), f, phase_offset=-np.pi / 4.0)
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
        assert np.asarray(result.aux_coords['receiver_depth'][1]) == pytest.approx(
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

    def test_the_mismatch_error_names_the_sorted_diagonal_pairing(self, tmp_path):
        """BELLHOP sorts both receiver lists before pairing them
        (``SourceReceiverPositions.f90:224``), so the error says what an
        irregular grid can express instead of leaving the user to guess."""
        env = Environment(bathymetry=100.0, ssp=1500.0)
        src = Source(depths=10.0, frequencies=100.0)
        rcv = Receiver(depths=[10.0, 20.0, 30.0], ranges=[100.0, 200.0])
        with pytest.raises(ConfigurationError,
                           match=r"sorts both lists.*monotone diagonal"):
            Bellhop(verbose=False, grid_type='I', work_dir=tmp_path).run(
                env, src, rcv, RunMode.COHERENT_TL)

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
        return Bellhop(beam_type='G', n_beams=501, launch_angles_deg=fan,
                       backend='fortran').compute_field(self._env(), src, rcv)

    def test_a_pattern_narrower_than_the_fan_is_refused(self):
        with pytest.raises(ConfigurationError, match='launch fan'):
            self._run(30.0, (-80.0, 80.0))

    def test_a_pattern_file_is_held_to_the_fan_by_its_angles(self, tmp_path):
        """A ``.sbp`` path is checked on the angles its file holds, on both
        sides: the lobe's own span passes, one degree more is refused."""
        from uacpy.io import write_source_beam_pattern
        from uacpy.models.bellhop._checks import (
            check_beam_pattern_spans_the_fan)
        lobe = self._lobe(30.0)
        path = tmp_path / 'lobe.sbp'
        write_source_beam_pattern(path, lobe[:, 0], lobe[:, 1])
        check_beam_pattern_spans_the_fan(path, launch_angles_deg=(-30.0, 30.0))
        with pytest.raises(ConfigurationError, match=r'spans \[-30, 30\]°'):
            check_beam_pattern_spans_the_fan(path, launch_angles_deg=(-31.0, 30.0))

    def test_the_default_fan_is_what_makes_this_reachable(self):
        """``launch_angles_deg`` defaults to ±80°, so an ordinary narrow lobe trips it."""
        assert Bellhop().launch_angles_deg == (-80, 80)

    @pytest.mark.requires_binary
    def test_a_pattern_covering_the_fan_runs_and_is_finite(self):
        tl = np.asarray(self._run(80.0, (-80.0, 80.0)).dB)
        assert np.isfinite(tl).all()

    @pytest.mark.requires_binary
    def test_narrowing_the_fan_to_the_pattern_also_works(self):
        """The other remedy the error offers must work too."""
        tl = np.asarray(self._run(60.0, (-60.0, 60.0)).dB)
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

    @pytest.mark.parametrize('beam_type', ['G', 'B', 'g'])
    def test_incoherent_runs_where_the_branch_exists(self, beam_type):
        """The geometric beams implement ``'I'``; so do both Cerveny
        routines (``influence.f90:137-140`` and :279-283), but their level
        falls with the beam count, so those pairs are refused (see
        ``TestCervenyBeamsRefuseTheMagnitudeSumModes``)."""
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
        from uacpy.models.bellhop._tables import (_BEAM_TYPE_RUN_TYPES,
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
        assert np.asarray(result.dB).shape == (4,)

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
        assert np.isfinite(np.asarray(result.dB)).all()

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
        return make_pekeris(sound_speed=1600.0, density=1.5)

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
            run_mode=RunMode.COHERENT_TL).dB)
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
            run_mode=RunMode.COHERENT_TL).dB)
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
        tl = np.atleast_2d(np.squeeze(self._run(env, zs).dB))
        seafloor = np.asarray(env.bathymetry.eval(range=rcv.ranges),
                              dtype=float)
        above = (np.asarray(rcv.depths, dtype=float)[:, None]
                 <= seafloor[None, :])
        assert np.all(np.isfinite(tl[above]))
        assert np.all(np.isnan(tl[~above]))

    def test_the_guard_is_bellhop_only(self):
        # Kraken/Scooter answer a buried source (heavily attenuated, as
        # physics requires). Putting this on the shared funnel would reject
        # two correct answers.
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
        return make_pekeris(sound_speed=1600.0)

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
            run_mode=RunMode.COHERENT_TL).dB)
        assert np.all(np.isfinite(tl))

    @pytest.mark.parametrize('beam_type', ['g', 'C', 'R'])
    def test_several_equally_spaced_ranges_work(self, beam_type):
        tl = np.squeeze(Bellhop(beam_type=beam_type).run(
            self._env(), Source(depths=25.0, frequencies=200.0),
            Receiver(depths=[50.0], ranges=np.linspace(500.0, 3000.0, 6)),
            run_mode=RunMode.COHERENT_TL).dB)
        assert np.any(np.isfinite(tl))


class TestSolverWarningsReachTheCaller:
    """The AT binaries write both fatals and *non-fatal* diagnoses to the
    ``.prt``. Only the fatals were read (``_launch.attach_prt_tail``, on the
    exception path), so a run the solver itself diagnosed came back at exit 0
    with a full-size result and nothing said — BELLHOP writes ``Warning in
    BELLHOP : Too few beams`` while uacpy returned the under-sampled TL field
    in silence."""

    @staticmethod
    def _env():
        return make_pekeris(bathymetry=200.0)

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


class TestBroadbandEmptyCellsAndTheOrigin:
    """A cell whose arrival list is empty is Bellhop's empty sum: ``H = 0``
    and a silent trace, its TL the 600 dB no-energy marker, as COHERENT_TL
    reports a cell no beam reached. Only ``r = 0`` is NaN: the beam types
    that record arrivals ('G', 'B', 'g') never evaluate it
    (``influence.f90:373, :419, :528``)."""

    def test_an_empty_arrival_list_is_the_empty_sum(self):
        empty = dict(n_arrivals=0, amplitudes=np.array([]),
                     phases=np.array([]), delays=np.array([]),
                     delays_imag=np.array([]))
        H = _cell_tf(empty, np.linspace(90.0, 110.0, 8))
        assert (H == 0).all()

    @staticmethod
    def _shadow_run(mode, tmp_path, **kw):
        """The TL shadow rig (a near-surface column at 4 km no beam reaches)
        with an r = 0 column added."""
        env = Environment(
            name='shadow', bathymetry=1000.0,
            ssp=np.array([[0.0, 1540.0], [200.0, 1480.0], [1000.0, 1480.0]]),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.5,
                                      attenuation=0.5))
        rcv = Receiver(depths=np.linspace(0.0, 100.0, 4),
                       ranges=np.array([0.0, 4000.0, 8000.0, 12000.0]))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return Bellhop(verbose=False, launch_angles_deg=(-20.0, 20.0),
                           work_dir=tmp_path, cleanup=False).run(
                env, Source(depths=20.0, frequencies=200.0), rcv,
                run_mode=mode, **kw)

    @staticmethod
    def _empty_off_origin(result):
        arrivals = result.components['arrivals']
        counts = np.array([[cell['n_arrivals'] for cell in row]
                           for row in arrivals.by_receiver[0]])
        empty = counts == 0
        empty[:, 0] = False
        assert empty.any(), "the rig must hold a cell no arrival reaches"
        return empty

    @pytest.mark.requires_binary
    def test_a_broadband_cell_without_arrivals_is_zero_and_r0_is_nan(
            self, tmp_path):
        H = self._shadow_run(RunMode.BROADBAND, tmp_path)
        data = np.asarray(H.data)
        assert np.isnan(data[:, 0, :]).all()
        assert (data[self._empty_off_origin(H)] == 0).all()
        assert np.isfinite(data[:, 1:, :]).all()

    @pytest.mark.requires_binary
    def test_a_time_series_cell_without_arrivals_is_silent(self, tmp_path):
        p = self._shadow_run(RunMode.TIME_SERIES, tmp_path,
                             source_waveform=np.hanning(64),
                             sample_rate=2000.0)
        data = np.asarray(p.data)
        assert np.isnan(data[:, 0, :]).all()
        assert (data[self._empty_off_origin(p)] == 0).all()
        assert np.isfinite(data[:, 1:, :]).all()

    @pytest.mark.requires_binary
    @pytest.mark.parametrize('mode,kw', [
        (RunMode.BROADBAND, {}),
        (RunMode.TIME_SERIES, {'source_waveform': np.hanning(64),
                               'sample_rate': 2000.0})])
    def test_a_line_source_r0_column_is_nan_on_the_synthesis_routes(
            self, mode, kw):
        """A line source's factor carries no ``1/sqrt(r)``, but the beam
        types that record arrivals still never reach ``r = 0``
        (``influence.f90:373, :419, :528``), so its column is NaN there too."""
        env = Environment(name='bb-line-r0', bathymetry=100.0, ssp=1500.0)
        src = Source(depths=25.0, frequencies=200.0, source_type='line')
        rcv = Receiver(depths=np.array([50.0]),
                       ranges=np.array([0.0, 1000.0, 2000.0]))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = Bellhop(verbose=False).run(env, src, rcv, run_mode=mode,
                                                **kw)
        data = np.asarray(result.data)
        assert np.isnan(data[:, 0, :]).all()
        assert np.isfinite(data[:, 1:, :]).all()

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
        # IFFT-based engines'; fc stays on metadata['fc'].
        freqs = np.atleast_1d(np.asarray(ts.frequencies, dtype=float))
        assert freqs.size > 1
        assert freqs[0] < 200.0 < freqs[-1]
        assert ts.metadata['fc'] == 200.0


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
        with recorded_warnings() as caught:
            f = Bellhop(verbose=False).run(env, self._src(), self._rcv(),
                                           run_mode=RunMode.COHERENT_TL)
        assert not [w for w in caught if 'BOUNCE' in str(w.message)]
        assert 'bounce' not in f.components
        assert np.isfinite(np.asarray(f.dB)).all()

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
        with recorded_warnings() as caught:
            f = Bellhop(verbose=False).run(env, self._src(), self._rcv(),
                                           run_mode=RunMode.COHERENT_TL)
        assert not [w for w in caught if 'BOUNCE' in str(w.message)]
        assert 'bounce' not in f.components
        assert np.isfinite(np.asarray(f.dB)).all()


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
        coh = np.asarray(tl_fields[RunMode.COHERENT_TL].dB).ravel()
        inc = np.asarray(tl_fields[RunMode.INCOHERENT_TL].dB).ravel()
        assert np.ptp(inc) < np.ptp(coh), (
            "incoherent TL is no smoother than coherent — RunType(1:1)='I' "
            "never took effect")

    def test_semicoherent_differs_from_both(self, tl_fields):
        coh = np.asarray(tl_fields[RunMode.COHERENT_TL].dB).ravel()
        inc = np.asarray(tl_fields[RunMode.INCOHERENT_TL].dB).ravel()
        semi = np.asarray(tl_fields[RunMode.SEMICOHERENT_TL].dB).ravel()
        # The Lloyd shading redistributes several dB across a 4 km line, so
        # 0.1 dB separates a real third mode from either neighbour while
        # staying far above solver reproducibility.
        assert not np.allclose(semi, coh, atol=0.1)
        assert not np.allclose(semi, inc, atol=0.1)

    def test_incoherent_payload_is_a_real_level(self, tl_fields):
        # docs/guide/results.md §9 "An incoherent field has no phase": the
        # incoherent sum is stored as real dB TL on every engine.
        data = np.asarray(tl_fields[RunMode.INCOHERENT_TL].data)
        assert not np.iscomplexobj(data)


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
        return _run_type_record(path.read_text().splitlines())[1]

    def test_beam_shift_writes_S_in_position_7(self, tmp_path):
        assert self._run_type(tmp_path, beam_shift=True)[6] == 'S'

    def test_default_leaves_position_7_blank(self, tmp_path):
        assert self._run_type(tmp_path)[6] == ' '

    def test_the_model_carries_the_flag(self):
        assert Bellhop(beam_shift=True).beam_shift is True
        assert Bellhop().beam_shift is False


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

    @pytest.mark.parametrize('knob', ['n_image', 'ib_win'])
    @pytest.mark.parametrize('value', [2.5, -1])
    def test_a_count_that_is_not_a_whole_number_is_refused(self, knob, value):
        """ReadEnvironmentBell.f90:217 reads Nimage and iBeamWindow into
        INTEGERs (bellhopMod.f90:22), so a fraction or a negative count is
        refused."""
        with pytest.raises(ConfigurationError,
                           match=f'Bellhop: {knob} must be an integer'):
            Bellhop(beam_type='C', **{knob: value})

    @pytest.mark.parametrize('knob', ['n_image', 'ib_win'])
    def test_a_whole_count_is_kept_as_an_int(self, knob):
        model = Bellhop(beam_type='C', **{knob: np.int64(0)})
        assert getattr(model, knob) == 0
        assert type(getattr(model, knob)) is int


class TestAGaussianBeamBandWarnsPastHalfOfFc:
    """bellhop.md §6 "whether that costs anything": a
    Gaussian beam (``'B'``) takes its width from the one trace at fc, so its
    bins drift away from fc (measured +1.41 /
    -0.88 dB of mean energy at +/-50 % on a 30 m sand guide); the hat beams
    of the default ``'G'`` carry no frequency in their width and match a
    trace at each frequency to 0.05 dB at +/-95 %. So only a ``'B'`` band
    reaching past 50 % of fc warns, at run time, whichever way the band was
    given; a band at 50 % exactly stays silent."""

    @staticmethod
    def _notice(beam_type, frequencies, fc=1000.0,
                mode=RunMode.BROADBAND):
        from uacpy.models.bellhop._plan import gaussian_beam_band_notice
        return gaussian_beam_band_notice(mode, frequencies, fc,
                                         beam_type=beam_type,
                                         model_name='Bellhop')

    @pytest.mark.parametrize('beam_type, band, warns', [
        ('B', (499.0, 1000.0), True), ('B', (1000.0, 1501.0), True),
        ('B', (500.0, 1500.0), False), ('G', (50.0, 1950.0), False)])
    def test_only_a_gaussian_band_past_half_of_fc_is_named(
            self, beam_type, band, warns):
        text = self._notice(beam_type, np.array(band))
        assert (text is not None) is warns
        if warns:
            assert "beam_type='B'" in text and '+1.41' in text

    def test_a_time_series_band_is_held_to_the_same_rule(self):
        assert self._notice('B', np.array([400.0, 1600.0]),
                            mode=RunMode.TIME_SERIES) is not None
        assert self._notice('B', np.array([400.0, 1600.0]),
                            mode=RunMode.COHERENT_TL) is None

    def test_a_wide_bandwidth_factor_constructs_quietly(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            Bellhop(bandwidth_factor=1.5)

    @pytest.mark.requires_binary
    @pytest.mark.parametrize('beam_type, warns', [('B', True), ('G', False)])
    def test_a_run_over_a_wide_band_says_it_for_gaussian_beams_only(
            self, beam_type, warns):
        env = uacpy.Environment(bathymetry=30.0, ssp=1500.0, bottom='sand')
        f = np.arange(100.0, 1901.0, 50.0)
        with recorded_warnings() as caught:
            Bellhop(verbose=False, beam_type=beam_type).run(
                env, uacpy.Source(depths=15.0, frequencies=f),
                uacpy.Receiver(depths=[15.0], ranges=[500.0, 1000.0]),
                run_mode=RunMode.BROADBAND, frequencies=f)
        said = any("takes every Gaussian beam's width" in str(w.message)
                   for w in caught)
        assert said is warns


class TestBroadbandFrequencyGridResolution:
    """A single centre frequency expands to ``n_frequencies`` bins spanning
    ``fc*(1 +/- bandwidth_factor/2)``
    (DOCUMENTATION.md §15 "Bellhop synthesizes", base.py
    ``_band.broadband_band``); explicit ``frequencies=`` wins over
    everything. Resolver-level — nothing runs."""

    def test_default_is_128_bins_over_half_fc(self):
        from uacpy.models._defaults import DEFAULT_BROADBAND_N_FREQUENCIES
        assert DEFAULT_BROADBAND_N_FREQUENCIES == 128
        model = Bellhop(verbose=False)
        freqs = model._requested_frequencies(
            RunMode.BROADBAND, Source(depths=50.0, frequencies=200.0), None,
            None).frequencies
        assert freqs.shape == (128,)
        assert freqs[0] == pytest.approx(150.0)
        assert freqs[-1] == pytest.approx(250.0)
        assert np.allclose(np.diff(freqs), freqs[1] - freqs[0])

    def test_constructor_knobs_shape_the_grid(self):
        model = Bellhop(verbose=False, n_frequencies=16, bandwidth_factor=0.2)
        freqs = model._requested_frequencies(
            RunMode.BROADBAND, Source(depths=50.0, frequencies=200.0), None,
            None).frequencies
        assert freqs.shape == (16,)
        assert freqs[0] == pytest.approx(180.0)
        assert freqs[-1] == pytest.approx(220.0)

    def test_explicit_frequencies_win(self):
        model = Bellhop(verbose=False, n_frequencies=16)
        freqs = model._requested_frequencies(
            RunMode.BROADBAND, Source(depths=50.0, frequencies=200.0),
            np.array([90.0, 100.0, 110.0]), None).frequencies
        np.testing.assert_array_equal(freqs, [90.0, 100.0, 110.0])


@pytest.mark.requires_binary
def test_a_broadband_run_states_its_surface_speed_and_its_arrivals():
    """One 1-bin broadband run, two documented contracts:
    ``speeds.surface`` is the sea-surface sound speed of the first profile
    (DOCUMENTATION.md §8 "the sea-surface sound speed of the first profile" —
    a physical speed, deliberately not 1500), and
    ``components['arrivals']`` keeps the :class:`Arrivals` the band was
    synthesised from so a different waveform needs no re-run, and no nested
    result rides in ``metadata``.
    results.md §8 "`components['arrivals']`, so you can re-synthesise"
    """
    from uacpy.core.ssp import SoundSpeedProfile
    env = Environment(
        name='c0-meta', bathymetry=100.0,
        ssp=SoundSpeedProfile.from_pairs([(0.0, 1490.0), (100.0, 1500.0)]))
    result = Bellhop(verbose=False).run(
        env, Source(depths=25.0, frequencies=200.0),
        Receiver(depths=np.array([50.0]), ranges=np.array([1000.0])),
        run_mode=RunMode.BROADBAND, frequencies=np.array([200.0]))
    assert result.speeds.surface == pytest.approx(1490.0)
    assert isinstance(result.components['arrivals'], Arrivals)
    assert not [key for key, value in result.metadata.items()
                if isinstance(value, Result)]


class TestBackendAutoSelectionPriority:
    """``backend=None`` auto-selects cuda > cxx > fortran, silently
    (DOCUMENTATION.md §7 "auto-selects the fastest installed binary",
    bellhop.md §5 "auto-picks in the order cuda"). ``_find_bellhop_executable``
    expresses the priority as the name order handed to
    ``_launch.find_executable_in_paths``, which returns the first name that
    resolves; the fake below reproduces exactly that first-name-wins search
    over a directory of stub files, so the priority is pinned by
    construction alone — no binary is ever launched."""

    @staticmethod
    def _fake_find(available_dir):
        from uacpy.core.exceptions import ExecutableNotFoundError

        def fake(names, **kwargs):
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
        monkeypatch.setattr(bellhop_backend, 'find_executable_in_paths',
                            self._fake_find(tmp_path))
        # bellhop.md §7 "picks the first installed binary" — without a
        # warning.
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            model = Bellhop(verbose=False)
        assert model._resolved_backend == expected_version
        assert model._exe.name == expected_name

    @pytest.mark.parametrize('installed,expected_version', [
        (('bellhopcuda', 'bellhopcxx', 'bellhop'), 'cuda'),
        (('bellhopcxx', 'bellhop'), 'cxx'),
        (('bellhop',), 'fortran'),
    ])
    def test_the_repr_names_the_auto_selected_build(
            self, tmp_path, monkeypatch, installed, expected_version):
        """``backend=None`` prints as the build it resolved, and a copy
        still auto-selects."""
        for name in installed:
            stub = tmp_path / name
            stub.write_text('')
            stub.chmod(0o755)
        monkeypatch.setattr(bellhop_backend, 'find_executable_in_paths',
                            self._fake_find(tmp_path))
        model = Bellhop(verbose=False)
        assert repr(model) == f"Bellhop(backend={expected_version!r})"
        assert model.copy().backend is None

    def test_a_pinned_executable_named_for_no_build_shows_no_backend(
            self, tmp_path):
        stub = tmp_path / 'my_ray_tracer'
        stub.write_text('')
        stub.chmod(0o755)
        model = Bellhop(executable=stub, verbose=False)
        assert model.select_backend() == 'custom'
        assert repr(model) == 'Bellhop()'


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
                      launch_angles_deg=(-45.0, 45.0)).run(env, src, rcv,
                                               run_mode=RunMode.ARRIVALS)
        counts[beam_type] = int(arr.by_receiver[0][0][0]['n_arrivals'])
    assert counts['G'] >= 1
    assert counts['B'] > 2 * counts['G'], (
        f"'B' should picket each path across the beam window: {counts}")


class TestBellhopStatesNoRayValidityThreshold:
    """Bellhop's run carries no water-depth-in-wavelengths warning: no source
    gives an a-priori ray-theory criterion (JKPS §3.4.2), and the one that
    gives numbers, Stergiopoulos §10.2.2 (rays accurate at D/λ ≥ 100, modes
    used at ≲ 30), lives in the guidance of docs/models/bellhop.md and
    docs/models/README.md."""

    @pytest.mark.requires_binary   # constructs Bellhop (resolves its binary)
    def test_a_guide_one_wavelength_deep_runs_without_a_validity_notice(
            self):
        """80 m at 20 Hz in 1500 m/s is D/λ = 1.07."""
        env = uacpy.Environment(bathymetry=80.0, ssp=1500.0, bottom='sand')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            notices = Bellhop(verbose=False).run_settings(
                env, uacpy.Source(depths=10.0, frequencies=20.0),
                uacpy.Receiver(depths=[40.0],
                               ranges=np.linspace(100.0, 5000.0, 10))
            ).engine.notices
        assert not [n for n in notices
                    if 'wavelength' in n.message and 'D/' in n.message]


@pytest.mark.requires_binary
def test_the_geometry_notices_are_settings_the_preview_shows():
    """The launch-fan and r = 0 conditions are decided in stage 3:
    ``run_settings().engine.notices`` holds them, in the order they are
    announced, ``run_settings`` announces them, and ``validate_inputs``
    says nothing. A receiver 70 m below the source at 10 m range needs
    81.9 deg, outside the +/-80 deg fan."""
    env = uacpy.Environment(bathymetry=80.0, ssp=1500.0, bottom='sand')
    src = uacpy.Source(depths=10.0, frequencies=20.0)
    rcv = uacpy.Receiver(depths=[80.0 - 0.5], ranges=[0.0, 10.0, 500.0])
    model = Bellhop(verbose=False)
    with recorded_warnings() as rec:
        model.validate_inputs(env, src, rcv)
    assert rec == []
    with recorded_warnings() as rec:
        notices = [n.message for n in model.run_settings(env, src, rcv).engine.notices]
    assert ['launch angle outside' in n for n in notices[:2]] == [
        True, False]
    assert 'starts at r=0 m' in notices[1]
    assert [str(w.message) for w in rec][:2] == list(notices[:2])


@pytest.mark.requires_binary
class TestTheBeamCountTheBinaryChoseIsRecorded:
    """``n_beams=0`` leaves the count to BELLHOP (``angleMod.f90:38-51``),
    so the settings hold 0; the result's ``run_settings.notes`` carries the
    count the binary printed to its ``.prt`` (``angleMod.f90:70``)."""

    _ENV = uacpy.Environment(bathymetry=100.0, ssp=1500.0, bottom='sand')
    _SRC = uacpy.Source(depths=25.0, frequencies=200.0)
    _RCV = uacpy.Receiver(depths=[50.0], ranges=np.linspace(100.0, 5000.0, 10))

    @pytest.mark.parametrize('mode', [RunMode.COHERENT_TL, RunMode.ARRIVALS])
    def test_the_chosen_count_is_noted(self, mode, tmp_path):
        result = Bellhop(verbose=False, backend='fortran', work_dir=tmp_path,
                         cleanup=False).run(self._ENV, self._SRC, self._RCV,
                                            run_mode=mode)
        prt = next(tmp_path.rglob('*.prt')).read_text()
        count = int(re.search(r'Number of beams in elevation\s*=\s*(\d+)',
                              prt).group(1))
        assert result.run_settings.engine.n_beams == 0
        assert (f"n_beams: 0, so the binary chose {count} beams "
                f"(angleMod.f90:38-51), as its .prt reports"
                in result.run_settings.notes)

    def test_a_pinned_count_is_not_noted(self):
        result = Bellhop(verbose=False, backend='fortran', n_beams=100).run(
            self._ENV, self._SRC, self._RCV)
        assert not [n for n in result.run_settings.notes if 'n_beams' in n]


class TestBellhopTracesOneBeamOfTheFan:
    """``Bellhop(single_beam=k)``: ``TopOpt(6:6) = 'I'`` and ``NBeams k``
    (``angleMod.f90:32-33``), the Bellhop User Guide §6.2 single-beam
    option; ``bellhop.f90:265`` then sums beam ``k`` alone."""

    _ENV = uacpy.Environment(bathymetry=100.0, ssp=1500.0, bottom='sand')
    _SRC = uacpy.Source(depths=25.0, frequencies=200.0)
    _RCV = uacpy.Receiver(depths=np.linspace(5.0, 95.0, 19),
                          ranges=np.linspace(200.0, 3000.0, 15))

    def test_the_deck_carries_the_option_and_the_index(self, tmp_path):
        from uacpy.io import write_bellhop_env_file
        path = tmp_path / 'one.env'
        write_bellhop_env_file(path, self._ENV, self._SRC, self._RCV,
                               n_beams=21, single_beam=12)
        lines = path.read_text().splitlines()
        assert lines[3].strip("'")[5] == 'I'
        assert '21 12' in lines
        write_bellhop_env_file(path, self._ENV, self._SRC, self._RCV,
                               n_beams=21)
        lines = path.read_text().splitlines()
        assert lines[3].strip("'")[5] == ' ' and '21' in lines

    @pytest.mark.parametrize('single_beam, n_beams', [(0, 21), (22, 21)])
    def test_an_index_outside_the_fan_is_refused(self, single_beam, n_beams):
        with pytest.raises(ConfigurationError, match='single_beam'):
            Bellhop(n_beams=n_beams, single_beam=single_beam)
        Bellhop(n_beams=21, single_beam=21)
        Bellhop(single_beam=500)        # n_beams=0: the binary's count

    @pytest.mark.requires_binary
    def test_one_beam_carries_part_of_the_fans_field(self, tmp_path):
        """Beam 12 of 21 is summed alone: its field differs from the
        fan's, and BELLHOP's .prt names it."""
        fan = Bellhop(verbose=False, backend='fortran', n_beams=21).run(
            self._ENV, self._SRC, self._RCV)
        one = Bellhop(verbose=False, backend='fortran', n_beams=21,
                      single_beam=12, work_dir=tmp_path, cleanup=False).run(
            self._ENV, self._SRC, self._RCV)
        assert 'Trace only beam number' in next(
            tmp_path.rglob('*.prt')).read_text()
        p_fan, p_one = np.abs(fan.data), np.abs(one.data)
        assert np.nanmax(np.abs(p_fan - p_one)) > 0.1 * np.nanmax(p_fan)


@pytest.mark.requires_binary
class TestBellhopConstructorGuards:
    """Round-19 constructor guards mirroring the RAM siblings: n_beams
    integrality (the deck writer emits ``int(n_beams)``), alpha limit
    ordering, and step sign/finiteness."""

    def test_a_fractional_n_beams_raises_the_integer_guard(self):
        with pytest.raises(ConfigurationError, match='integer beam count'):
            Bellhop(verbose=False, n_beams=250.7)

    def test_a_string_n_beams_is_refused_as_text(self):
        with pytest.raises(ConfigurationError,
                           match='^Bellhop: n_beams must be a number, not '
                                 'text'):
            Bellhop(verbose=False, n_beams='many')

    def test_a_numpy_integer_n_beams_is_accepted(self):
        assert Bellhop(verbose=False, n_beams=np.int64(300)).n_beams == 300

    def test_reversed_launch_angles_limits_raise_the_ordering_guard(self):
        with pytest.raises(ConfigurationError, match='min_deg < max_deg'):
            Bellhop(verbose=False, launch_angles_deg=(80, -80))

    def test_equal_launch_angles_limits_raise_the_ordering_guard(self):
        with pytest.raises(ConfigurationError, match='min_deg < max_deg'):
            Bellhop(verbose=False, launch_angles_deg=(45, 45))

    def test_a_nan_launch_angles_limit_raises_the_ordering_guard(self):
        with pytest.raises(ConfigurationError, match='min_deg < max_deg'):
            Bellhop(verbose=False, launch_angles_deg=(float('nan'), 80))

    def test_non_numeric_alpha_entries_raise_a_typed_error(self):
        with pytest.raises(ConfigurationError, match='numbers'):
            Bellhop(verbose=False, launch_angles_deg=('a', 'b'))

    def test_a_negative_step_raises(self):
        with pytest.raises(ConfigurationError, match='>= 0 and finite'):
            Bellhop(verbose=False, ray_step=-5.0)

    def test_an_infinite_step_raises(self):
        with pytest.raises(ConfigurationError, match='>= 0 and finite'):
            Bellhop(verbose=False, ray_step=float('inf'))

    def test_zero_step_is_the_automatic_step_sentinel(self):
        assert Bellhop(verbose=False, ray_step=0.0).ray_step == 0.0


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
        with recorded_warnings() as caught:
            Bellhop(beam_type=beam_type, component='P', verbose=False)
        assert not [w for w in caught if 'component' in str(w.message)]


class TestBellhopMinimumBeamFan:
    """``bellhop.f90:176-178`` zeroes ``Dalpha`` for a single beam, so every
    beam has zero width and the field is all-NaN. The deck writer
    (``bellhop_writer.py``) refuses ``n_beams=1`` for every influence run
    type before the binary is launched. Two beams have a finite ``Dalpha``
    and do produce a field — 69.1 dB on a deep-water direct path against a
    converged 66.1 dB, under-resolved but not degenerate, and no more so
    than the five-beam fan that reads 100.1 dB on the same geometry."""

    @staticmethod
    def _run(n_beams, run_mode, tmp_path):
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          bottom=BoundaryProperties(
                              acoustic_type='half-space', sound_speed=1600.0,
                              density=1.5, attenuation=0.5))
        return Bellhop(n_beams=n_beams, verbose=False, work_dir=tmp_path).run(
            env, Source(depths=25.0, frequencies=200.0),
            Receiver(depths=[50.0], ranges=[1000.0]), run_mode=run_mode)

    def test_one_beam_is_refused(self, tmp_path):
        with pytest.raises(ConfigurationError, match='n_beams=1'):
            self._run(1, RunMode.COHERENT_TL, tmp_path)
        assert not (tmp_path / 'bellhop.shd').exists()

    def test_two_beams_are_allowed(self, tmp_path):
        self._run(2, RunMode.COHERENT_TL, tmp_path)

    def test_a_single_ray_trace_is_allowed_but_not_a_single_beam_eigenray_search(
            self, tmp_path):
        # bellhop.f90:288 skips influence for RAYS only; EIGENRAYS detect
        # receiver hits inside the influence step, where one beam has no width.
        self._run(1, RunMode.RAYS, tmp_path)
        with pytest.raises(ConfigurationError, match='single beam'):
            self._run(1, RunMode.EIGENRAYS, tmp_path)


@pytest.mark.requires_binary
@pytest.mark.slow
def test_bellhop_two_beam_fan_returns_a_usable_field():
    """The measurement behind the writer's floor of two beams: a two-beam
    fan on a deep-water direct path returns a finite TL a few dB off a
    converged 51-beam run — not an all-NaN field — so refusing it would
    refuse a working, merely under-resolved, configuration."""
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
            result = Bellhop(n_beams=n_beams, launch_angles_deg=(-10, 10),
                             verbose=False).run(env, source, receiver,
                                                run_mode=RunMode.COHERENT_TL)
        return float(np.asarray(result.dB, dtype=float).ravel()[0])

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
        assert resolve_ray_step(
            self._env(depth), ray_step=None) == pytest.approx(expected)

    @pytest.mark.parametrize('pinned', [250.0, 0.0])
    def test_a_pinned_step_reaches_the_deck_unchanged(self, pinned):
        """0 included: it is BELLHOP's own "choose for me"
        (bellhop.f90:170-174), which a caller may ask for."""
        assert resolve_ray_step(
            self._env(5000.0), ray_step=pinned) == pinned

    def test_the_default_step_is_finer_than_the_binarys_own_choice(self):
        # Unset, uacpy sends a positive step finer than the depth/10
        # bellhop.f90:170-174 substitutes for 0.
        assert Bellhop.__init__.__kwdefaults__['ray_step'] is None
        for depth in (100.0, 1000.0, 5000.0):
            resolved = resolve_ray_step(self._env(depth), ray_step=None)
            assert 0.0 < resolved < depth / 10.0

    @pytest.mark.requires_binary
    def test_a_pinned_zero_step_reaches_the_deck_settings_as_zero(self):
        """The deck's step is 0, so BELLHOP picks depth/10."""
        from uacpy.core import Receiver, Source
        settings = Bellhop(verbose=False, ray_step=0.0).run_settings(
            self._env(100.0), Source(depths=20.0, frequencies=200.0),
            Receiver(depths=[50.0], ranges=[1000.0]))
        assert settings.engine.ray_step == 0.0
        assert settings.engine.ray_step_origin == 'Bellhop(ray_step=…)'


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
    def _check(ranges, launch_angles_deg=(-80.0, 80.0)):
        from uacpy.core import Receiver, Source
        from uacpy.models import Bellhop
        _say_fan_miss(Bellhop(launch_angles_deg=launch_angles_deg, verbose=False),
            Source(depths=10.0, frequencies=2000.0),
            Receiver(depths=[90.0], ranges=ranges))

    def test_the_suggested_fan_covers_the_angle_actually_needed(self):
        """The advice was a hardcoded ``launch_angles_deg=(-89.9, 89.9)`` whatever the fan
        or the geometry. It has to be derived from the angle needed, and must
        never be NARROWER than the fan already in use."""
        with pytest.warns(UserWarning, match='launch angle outside') as rec:
            self._check([10.0], launch_angles_deg=(-80.0, 80.0))
        msg = str(rec[0].message)
        assert 'Widen launch_angles_deg' in msg, msg
        nums = [float(x) for x in re.findall(r'launch_angles_deg=\(-([0-9.]+)', msg)]
        assert nums and nums[0] >= 82.9, msg      # covers atan2(80, 10)
        assert nums[0] > 80.0, msg                # wider than what is in use

    def test_a_narrow_fan_is_told_to_widen_even_when_the_need_is_vertical(self):
        """The branch must key on the FAN's width, not on the angle needed. A
        +/-80 deg fan has room to widen whatever the geometry asks for; only a
        fan already near vertical does not. Both parts of the advice apply
        here: widening recovers most of these pairs, and a residual blind cone
        remains because 90 deg is a vertical ray."""
        with pytest.warns(UserWarning, match='launch angle outside') as rec:
            self._check([0.05], launch_angles_deg=(-80.0, 80.0))
        msg = str(rec[0].message)
        assert 'Widen launch_angles_deg' in msg, msg
        assert 'blind cone' in msg, msg

    def test_a_fan_already_at_the_vertical_limit_is_not_told_to_widen(self):
        """A fan of +/-89.95 deg is a hair off vertical, so widening chases an
        asymptote: the blind cone shrinks but never closes. The reachable
        remedy is the receiver range -- here r < 80/tan(89.95) = 0.07 m."""
        with pytest.warns(UserWarning, match='launch angle outside') as rec:
            self._check([0.05], launch_angles_deg=(-89.95, 89.95))
        msg = str(rec[0].message)
        assert 'Widen launch_angles_deg' not in msg, msg
        assert 'range' in msg.lower(), msg
        assert '0.07' in msg, msg

    def test_the_report_states_each_fans_own_cone_in_user_terms(self):
        """The cone quoted beside the current fan is that fan's (80 m of
        depth offset / tan 80 deg = 14.11 m), not one the suggested fan
        would leave; a residual cone is stated only when receivers remain
        inside the suggested fan's (80 / tan 89.9 deg = 0.14 m), and the text
        carries no Fortran source addresses."""
        with pytest.warns(UserWarning, match='launch angle outside') as rec:
            self._check([10.0], launch_angles_deg=(-80.0, 80.0))
        msg = str(rec[0].message)
        assert 'closer than 14.11 m in range' in msg, msg
        assert 'blind cone' not in msg and '.f90' not in msg, msg
        with pytest.warns(UserWarning, match='launch angle outside') as rec:
            self._check([0.05], launch_angles_deg=(-80.0, 80.0))
        msg = str(rec[0].message)
        assert 'still leaves a blind cone of 0.14 m' in msg, msg

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
            self._check([10.0], launch_angles_deg=(-89.9, 89.9))

    def test_the_report_names_an_angle_the_fan_really_misses(self):
        # An asymmetric fan misses at one end only; the reported angle has to
        # be one of the misses, not merely the largest in magnitude.
        with pytest.warns(UserWarning, match='steepest is -8') as rec:
            from uacpy.core import Receiver, Source
            from uacpy.models import Bellhop
            _say_fan_miss(Bellhop(launch_angles_deg=(0.0, 80.0), verbose=False),
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
    excludes zero, which ``launch_angles_deg=(17, 74)`` is and ``Bellhop.__init__``
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
        from uacpy.models.bellhop._plan import _fan_miss_count_and_worst
        zs, zr, rr = (np.atleast_1d(np.asarray(a, dtype=float))
                      for a in (zs, zr, rr))
        count, worst = self._dense(zs, zr, rr, lo, hi)
        got_count, got_worst = _fan_miss_count_and_worst(zs, zr, rr, lo, hi)
        assert got_count == count
        if count:
            assert got_worst == pytest.approx(worst, abs=1e-9)

    @pytest.mark.parametrize('lo,hi', FANS)
    def test_the_early_out_fires_exactly_when_nothing_is_outside(self, lo, hi):
        from uacpy.core import Receiver, Source
        from uacpy.models import Bellhop
        zs, zr, rr = [30.0], [5.0, 150.0], [12.0, 60.0, 3000.0]
        count, _ = self._dense(zs, zr, rr, lo, hi)
        model = Bellhop(launch_angles_deg=(lo, hi), verbose=False)
        with recorded_warnings() as rec:
            _say_fan_miss(model,
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
        model = Bellhop(launch_angles_deg=(lo, hi), verbose=False)
        with pytest.warns(UserWarning, match='launch angle outside') as rec:
            _say_fan_miss(model,
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

def test_positive_launch_angle_traces_a_downward_ray():
    """``alpha > 0`` goes deeper — ``ray2D(1)%t = [COS(alpha), SIN(alpha)]/c``
    over a depth axis that is positive downward."""
    env = uacpy.Environment(bathymetry=2000.0, ssp=1500.0)
    src = uacpy.Source(depths=1000.0, frequencies=200.0)
    rcv = uacpy.Receiver(depths=np.linspace(0.0, 2000.0, 21),
                         ranges=np.linspace(0.0, 5000.0, 11))
    rays = Bellhop(verbose=False, launch_angles_deg=(-10.0, 10.0), n_beams=3).run(
        env, src, rcv, run_mode='rays')

    by_angle = {round(ray['launch_angle']): np.asarray(ray['z']) for ray in rays.rays}
    assert by_angle[10][1] > by_angle[10][0]      # +10° dives
    assert by_angle[-10][1] < by_angle[-10][0]    # -10° climbs


def test_downward_only_pattern_ensonifies_below_the_source():
    """A ``.sbp`` passing only positive angles moves energy to the deeper
    receiver, which is what the polar plot claims when it draws that lobe
    below the horizontal."""
    env = uacpy.Environment(bathymetry=2000.0, ssp=1500.0)
    rcv = uacpy.Receiver(depths=np.array([500.0, 1500.0]),
                         ranges=np.array([2000.0]))
    downward = np.array([[-180.0, -40.0], [-0.001, -40.0],
                         [0.0, 0.0], [180.0, 0.0]])
    model = Bellhop(verbose=False, launch_angles_deg=(-80.0, 80.0), n_beams=2001)

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
        msgs = warning_messages(lambda: Bellhop(verbose=False).run(
            env, source, receiver, run_mode=RunMode.BROADBAND),
            "fold back onto the early trace")
        assert len(msgs) == 1, msgs
        assert msgs[0].startswith(
            "Bellhop.run(run_mode=BROADBAND): a 0.00635 s record"), msgs[0]
        assert "Arrivals.synthesis_band" in msgs[0]

    def test_a_grid_the_caller_chose_is_not_second_guessed(self):
        env, source, receiver = self._geometry()
        grid = np.linspace(30e3, 50e3, 128)     # the same spacing, chosen
        assert warning_messages(lambda: Bellhop(verbose=False).run(
            env, source, receiver, run_mode=RunMode.BROADBAND,
            frequencies=grid), "fold back onto the early trace") == []


@pytest.mark.requires_binary
class TestRayCentredBeamsFillEveryRequestedRangeColumn:
    """influence.f90 clamps the range index to 1 and steps from irA + 1 for
    'C', 'R' and 'g' ('C' also returns before the last range), so without help
    the first (and last) requested column comes back NaN with no notice. The
    wrapper writes one extra range at each end and trims them off."""

    def _run(self, beam_type, ranges, **kw):
        env = uacpy.Environment(name='pad', bathymetry=100.0, ssp=1500.0)
        source = Source(depths=50.0, frequencies=200.0)
        receiver = Receiver(depths=[30.0, 60.0], ranges=ranges)
        return Bellhop(verbose=False, beam_type=beam_type, n_beams=200,
                       **kw).run(env, source, receiver,
                                 run_mode=RunMode.COHERENT_TL)

    @pytest.mark.parametrize('beam_type', ['C', 'R', 'g'])
    def test_first_and_last_columns_are_finite(self, beam_type):
        ranges = np.linspace(400.0, 1200.0, 5)
        field = self._run(beam_type, ranges)
        assert np.allclose(np.asarray(field.coords['range']), ranges)
        data = np.asarray(field.data)
        assert data.shape[-1] == ranges.size
        assert np.all(np.isfinite(data[:, 0])), data[:, 0]
        assert np.all(np.isfinite(data[:, -1])), data[:, -1]

    def test_the_padding_does_not_move_the_interior(self):
        ranges = np.linspace(400.0, 1200.0, 5)
        # One ray box for both: the default (a factor on the farthest
        # receiver) differs between the two axes, and a ray-centred beam's
        # last column reads ray points past it.
        a = np.asarray(self._run('R', ranges, r_box=1500.0).data)
        b = np.asarray(self._run('R', np.linspace(200.0, 1400.0, 7),
                                 r_box=1500.0).data)
        np.testing.assert_allclose(a, b[:, 1:-1], rtol=1e-5, atol=1e-9)

    def test_a_first_range_within_one_step_of_the_source_is_declared(self):
        msgs = warning_messages(lambda: self._run('R', np.linspace(100.0, 500.0, 5)),
                         "never fills the first receiver-range column")
        assert len(msgs) == 1, msgs
        assert 'the 100 m column comes back NaN' in msgs[0], msgs[0]

    @pytest.mark.parametrize('beam_type, first_nan', [
        ('C', True), ('R', True), ('g', True), ('G', False), ('B', False)])
    def test_an_unpadded_first_column_is_nan_where_bellhop_never_evaluates(
            self, beam_type, first_nan):
        """100 m with a 100 m step cannot be padded ahead of: the ray-centred
        beams never evaluate that column (influence.f90:223-228), so it is
        NaN and every other column holds the engine's value; the
        bracket-walking beams evaluate it."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            data = np.asarray(self._run(beam_type,
                                        np.linspace(100.0, 500.0, 5)).data)
        assert bool(np.isnan(data[:, 0]).all()) is first_nan, data[:, 0]
        assert np.isfinite(data[:, 1:]).all(), data

    @pytest.mark.parametrize('mode', [RunMode.BROADBAND, RunMode.TIME_SERIES])
    @pytest.mark.parametrize('beam_type, first_range, first_nan', [
        ('g', 100.0, True), ('g', 400.0, False), ('G', 100.0, False)])
    def test_a_synthesised_result_follows_the_same_rule(
            self, mode, beam_type, first_range, first_nan):
        """BROADBAND and TIME_SERIES are built from the ARRIVALS run ('g' is
        the ray-centred beam with an arrivals branch), whose unevaluated
        first column holds no arrivals: it is NaN, not H = 0, on the
        unpadded route only."""
        env = uacpy.Environment(name='pad', bathymetry=100.0, ssp=1500.0)
        receiver = Receiver(depths=[30.0, 60.0],
                            ranges=first_range + 100.0 * np.arange(5))
        call = (dict(frequencies=np.array([180.0, 200.0, 220.0]))
                if mode == RunMode.BROADBAND else
                dict(source_waveform=np.hanning(64) * np.sin(
                    2 * np.pi * 200.0 * np.arange(64) / 2000.0),
                     sample_rate=2000.0))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            field = Bellhop(verbose=False, beam_type=beam_type,
                            n_beams=200).run(
                env, Source(depths=50.0, frequencies=200.0), receiver,
                run_mode=mode, **call)
        data = np.asarray(field.data)
        assert bool(np.isnan(data[:, 0]).all()) is first_nan, data[:, 0]
        assert not np.isnan(data[:, 1:]).any()

    def test_an_arrivals_run_says_the_column_holds_no_arrivals(self):
        env = uacpy.Environment(name='pad', bathymetry=100.0, ssp=1500.0)
        msgs = warning_messages(
            lambda: Bellhop(verbose=False, beam_type='g', n_beams=200).run(
                env, Source(depths=50.0, frequencies=200.0),
                Receiver(depths=[30.0, 60.0],
                         ranges=np.linspace(100.0, 500.0, 5)),
                run_mode=RunMode.ARRIVALS),
            "never fills the first receiver-range column")
        assert len(msgs) == 1, msgs
        assert 'the 100 m column holds no arrivals' in msgs[0], msgs[0]


@pytest.mark.requires_binary
class TestEigenraysDeclareWhatTheirBeamTypeCannotDeliver:
    """Two run/beam pairs that Bellhop accepts and answers wrongly without
    saying so. Both are EIGENRAYS: the output is a set of ray records, so
    neither failure leaves a NaN behind for anyone to notice — a wrong
    eigenray set just looks like a different eigenray set.
    """

    RANGES = np.array([400.0, 500.0, 600.0])

    @staticmethod
    def _rig(depths=(50.0,)):
        env = uacpy.Environment(name='eig', bathymetry=100.0, ssp=1500.0)
        return (env, Source(depths=50.0, frequencies=200.0),
                Receiver(depths=np.array(depths, dtype=float),
                         ranges=TestEigenraysDeclareWhatTheirBeamTypeCannotDeliver.RANGES))

    @classmethod
    def _eigenrays(cls, beam_type, depths=(50.0,)):
        env, source, receiver = cls._rig(depths)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return Bellhop(verbose=False, beam_type=beam_type,
                           n_beams=300).run(env, source, receiver,
                                            run_mode=RunMode.EIGENRAYS)

    # --- 'g' skips the first receiver-range column -----------------------

    def test_the_ray_centred_beam_says_it_returns_no_rays_at_the_first_range(self):
        env, source, receiver = self._rig()
        msgs = warning_messages(
            lambda: Bellhop(verbose=False, beam_type='g', n_beams=300).run(
                env, source, receiver, run_mode=RunMode.EIGENRAYS),
            "no rays at receiver.ranges[0]")
        assert len(msgs) == 1, msgs
        # The remedy has to be in the message, not just the diagnosis.
        assert "beam_type='G'" in msgs[0], msgs[0]

    def test_that_warning_describes_a_real_hole(self):
        # The claim the warning makes, measured: 'g' returns nothing at the
        # first range where 'G' returns a full set. If the Fortran ever
        # starts filling column 1, this fails and the warning comes out.
        def ends_at_first_range(beam_type):
            rays = self._eigenrays(beam_type).rays
            ends = np.array([np.asarray(r['r'], float)[-1] for r in rays])
            return int(np.sum(np.abs(ends - self.RANGES[0]) < 1.0))
        assert ends_at_first_range('g') == 0
        assert ends_at_first_range('G') > 0

    @pytest.mark.parametrize('beam_type', ['G', 'B'])
    def test_the_bracket_walking_beams_are_quiet(self, beam_type):
        env, source, receiver = self._rig()
        assert warning_messages(
            lambda: Bellhop(verbose=False, beam_type=beam_type,
                            n_beams=300).run(env, source, receiver,
                                             run_mode=RunMode.EIGENRAYS),
            "no rays at receiver.ranges[0]") == []

    def test_a_ray_centred_tl_run_is_padded_rather_than_warned_about(self):
        # The TL branch pads around the same hole, so it must NOT inherit
        # the eigenray warning.
        env, source, receiver = self._rig(depths=(30.0, 60.0))
        assert warning_messages(
            lambda: Bellhop(verbose=False, beam_type='g', n_beams=200).run(
                env, source, receiver, run_mode=RunMode.COHERENT_TL),
            "no rays at receiver.ranges[0]") == []

    # --- 'S' writes every ray at every receiver depth --------------------

    def test_the_simple_gaussian_beam_says_its_eigenrays_are_unscreened(self):
        env, source, receiver = self._rig()
        msgs = warning_messages(
            lambda: Bellhop(verbose=False, beam_type='S', n_beams=300).run(
                env, source, receiver, run_mode=RunMode.EIGENRAYS),
            "not screened for passing near a receiver")
        assert len(msgs) == 1, msgs
        assert 'filter_by_miss_distance' in msgs[0], msgs[0]

    def test_that_warning_describes_a_real_defect(self):
        # influence.f90:698-699 has the proximity test commented out, so the
        # written set is dominated by rays that miss. Measured against 'G',
        # whose ApplyContribution is reached only inside `n < RadiusMax`.
        def miss_profile(beam_type):
            rays = self._eigenrays(beam_type).rays
            worst = [min(np.min(np.hypot(np.asarray(ray['r'], float) - rr,
                                         np.asarray(ray['z'], float) - 50.0))
                         for rr in self.RANGES)
                     for ray in rays]
            return len(rays), float(np.median(worst))
        n_sgb, miss_sgb = miss_profile('S')
        n_hat, miss_hat = miss_profile('G')
        assert n_sgb > 3 * n_hat, (n_sgb, n_hat)
        assert miss_sgb > 3 * miss_hat, (miss_sgb, miss_hat)

    def test_a_simple_gaussian_tl_run_is_quiet(self):
        env, source, receiver = self._rig()
        assert warning_messages(
            lambda: Bellhop(verbose=False, beam_type='S', n_beams=300).run(
                env, source, receiver, run_mode=RunMode.COHERENT_TL),
            "not screened for passing near a receiver") == []

    def test_the_run_completes_and_e_stays_in_the_table(self):
        # The warning must not become a rejection by the back door: the
        # records are real rays and filter_by_miss_distance recovers the
        # eigenrays from them, so the capability stays. _BEAM_TYPE_RUN_TYPES
        # mirrors the Fortran's CASE branches and nothing else — dropping
        # 'E' from it would contradict influence.f90:700-703 and redden
        # test_the_table_matches_the_vendored_source.
        from uacpy.models.bellhop._tables import _BEAM_TYPE_RUN_TYPES
        assert 'E' in _BEAM_TYPE_RUN_TYPES['S']
        rays = self._eigenrays('S')
        assert isinstance(rays, Rays) and len(rays.rays) > 0
        near = rays.filter_by_miss_distance(5.0, target_range=500.0,
                                            target_depth=50.0)
        assert 0 < len(near.rays) < len(rays.rays)


@pytest.mark.requires_binary
class TestTheTimeSeriesEchoNoticeIsSaidOnce:
    def test_a_window_that_drops_every_echo_is_reported_once_as_all(self):
        env = uacpy.Environment(name='once', bathymetry=300.0, ssp=1500.0)
        source = Source(depths=150.0, frequencies=2e3)
        receiver = Receiver(depths=[150.0], ranges=[1000.0])
        fs = 8e3
        t = np.arange(int(0.005 * fs)) / fs
        wf = np.hanning(t.size) * np.sin(2 * np.pi * 2e3 * t)
        # Anchored at emission (t_start=0), a 0.3 s window closes before the
        # first echo at ~0.67 s: every echo is dropped.
        msgs = warning_messages(lambda: Bellhop(verbose=False).run(
            env, source, receiver, run_mode=RunMode.TIME_SERIES,
            source_waveform=wf, sample_rate=fs, output_duration=0.3,
            t_start=0.0),
            "does not hold every echo")
        assert len(msgs) == 1, msgs
        assert "all of the received energy" in msgs[0]


class TestArrivalsAreMergedExactlyOnce:
    """Sequential Fortran Bellhop merges bracketing pairs as it accumulates;
    the multithreaded C++ engine leaves them for the reader. Re-merging a
    merged file is not a no-op, so the reader reads as written by default and
    the wrapper asks for the merge only where the engine skipped it."""

    def test_the_reader_reads_as_written_by_default(self):
        import inspect
        from uacpy.io.oalib_reader import read_arr_file
        assert inspect.signature(read_arr_file).parameters['merge'].default is False

    def test_the_wrapper_keys_the_merge_on_the_engine(self):
        model = Bellhop(verbose=False)
        model._resolved_backend = 'fortran'
        assert _merge(model) is False
        model._resolved_backend = 'cuda'
        assert _merge(model) is True

    @pytest.mark.requires_binary
    def test_both_backends_return_the_same_arrivals(self):
        env = uacpy.Environment(name='inv', bathymetry=100.0, ssp=1500.0)
        source = Source(depths=50.0, frequencies=200.0)
        receiver = Receiver(depths=[30.0, 70.0], ranges=[3000.0])
        results = {}
        for backend in ('fortran', 'cxx'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model = Bellhop(verbose=False, backend=backend, n_beams=300)
            if model._resolved_backend != backend:
                pytest.skip(f"{backend} binary not installed")
            results[backend] = model.run(env, source, receiver,
                                         run_mode=RunMode.ARRIVALS)
        f, c = results['fortran'], results['cxx']
        assert len(f.arrivals) == len(c.arrivals)
        np.testing.assert_allclose(np.sort(f.delays), np.sort(c.delays),
                                   atol=2e-6, rtol=0)


class TestASingleBeamCannotCarryEigenrays:
    """The deck writer refuses ``n_beams=1`` for an eigenray run
    (``bellhop_writer.py``): eigenrays are detected inside the influence
    step, where a single beam has no width."""

    def test_eigenrays_with_one_beam_are_refused_before_the_run(self):
        env = uacpy.Environment(name='one', bathymetry=100.0, ssp=1500.0)
        source = Source(depths=50.0, frequencies=200.0)
        receiver = Receiver(depths=[50.0], ranges=[1000.0])
        with pytest.raises(ConfigurationError, match='single beam'):
            Bellhop(verbose=False, n_beams=1).run(
                env, source, receiver, run_mode=RunMode.EIGENRAYS)


class TestArrivalsKeepTheRequestedDepthAxis:
    """BELLHOP clamps a receiver below the deck's bottom boundary onto it
    (``misc/SourceReceiverPositions.f90:136-139``) and the ``.arr`` then
    carries the clamped depth with the boundary cell's arrivals. The wrapper
    hands the ARRIVALS result back on the depth axis that was asked for, with
    that cell empty — the same no-data convention the TL and broadband routes
    report as NaN — instead of relabelling the receiver and handing over
    arrivals evaluated somewhere else."""

    @staticmethod
    def _run(depths):
        from uacpy.tests.conftest import make_pekeris
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return Bellhop(verbose=False, n_beams=20, beam_type='G').run(
                make_pekeris(name='clamp', bathymetry=100.0),
                Source(depths=50.0, frequencies=1000.0),
                Receiver(depths=depths, ranges=[500.0]),
                run_mode=RunMode.ARRIVALS)

    def test_a_below_seafloor_receiver_comes_back_empty_under_its_own_depth(self):
        result = self._run([50.0, 150.0])
        np.testing.assert_array_equal(result.receiver_depths, [50.0, 150.0])
        cells = result.by_receiver[0]
        assert cells[0][0]['n_arrivals'] > 0
        assert cells[1][0]['n_arrivals'] == 0
        assert len(cells[1][0]['delays']) == 0
        assert {a['depth_idx'] for a in result.arrivals} == {0}

    def test_an_in_water_receiver_is_left_as_read(self):
        # The discriminating counterpart: nothing is emptied when every
        # receiver lies inside the deck, and the axis is the one asked for.
        result = self._run([50.0, 90.0])
        np.testing.assert_array_equal(result.receiver_depths, [50.0, 90.0])
        assert all(c[0]['n_arrivals'] > 0 for c in result.by_receiver[0])

    def test_a_padded_two_source_run_returns_a_slab_per_source_depth(self):
        # beam_type 'g' on a rectilinear grid with >= 2 ranges takes the
        # padded route (_pad_receiver_ranges), and two source depths make
        # the reader hand back a ResultStack; the trim and the restore walk
        # its slabs, each of which keeps the two requested ranges.
        from uacpy.tests.conftest import make_pekeris
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = Bellhop(verbose=False, n_beams=20, beam_type='g').run(
                make_pekeris(name='clamp-pad', bathymetry=100.0),
                Source(depths=[30.0, 50.0], frequencies=1000.0),
                Receiver(depths=[50.0, 150.0], ranges=[500.0, 600.0]),
                run_mode=RunMode.ARRIVALS)
        assert result.n_slabs == 2
        for slab in result.slabs:
            np.testing.assert_array_equal(slab.receiver_ranges, [500.0, 600.0])
            np.testing.assert_array_equal(slab.receiver_depths, [50.0, 150.0])
            assert all(c['n_arrivals'] == 0 for c in slab.by_receiver[0][1])
            assert {a['depth_idx'] for a in slab.arrivals} == {0}

    def test_a_paired_grid_empties_the_cell_of_the_clamped_pair(self):
        from uacpy.tests.conftest import make_pekeris
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = Bellhop(verbose=False, n_beams=20, beam_type='G',
                             grid_type='I').run(
                make_pekeris(name='clamp-i', bathymetry=100.0),
                Source(depths=50.0, frequencies=1000.0),
                Receiver(depths=[50.0, 150.0], ranges=[400.0, 500.0]),
                run_mode=RunMode.ARRIVALS)
        np.testing.assert_array_equal(result.receiver_depths, [50.0, 150.0])
        cells = result.by_receiver[0][0]
        assert cells[0]['n_arrivals'] > 0
        assert cells[1]['n_arrivals'] == 0


class TestAPinnedExecutableIsReadLikeTheAutoPickedOne:
    """``Bellhop(executable=...)`` names the binary by path; which engine it
    is — and so whether its ``.arr`` still needs the AddArr pair-merge and
    whether the CLI takes a ``--2D`` flag — is read off the basename with
    the same three-way test the auto-pick uses. A basename matching none of
    them stays ``'custom'``, and an arrivals read then says the merge state
    is unknown."""

    @staticmethod
    def _stub(tmp_path, name):
        stub = tmp_path / name
        stub.write_text('')
        stub.chmod(0o755)
        return stub

    @pytest.mark.parametrize('name,version', [
        ('bellhopcxx', 'cxx'), ('bellhopcuda', 'cuda'), ('bellhop', 'fortran'),
        ('BellhopCUDA.exe', 'cuda'), ('my-bellhopcxx-build', 'cxx'),
    ])
    def test_the_variant_is_inferred_from_the_basename(
            self, tmp_path, name, version):
        model = Bellhop(executable=self._stub(tmp_path, name), verbose=False)
        assert model._resolved_backend == version
        assert model.executable == tmp_path / name

    def test_a_pinned_cxx_path_merges_like_the_auto_picked_one(
            self, tmp_path, monkeypatch):
        model = Bellhop(executable=self._stub(tmp_path, 'bellhopcxx'),
                        verbose=False)
        monkeypatch.setattr(os, 'cpu_count', lambda: 4)
        assert _merge(model) is True
        monkeypatch.setattr(os, 'cpu_count', lambda: 1)
        assert _merge(model) is False
        assert _command(model, 'base')[1] == '--2D'

    def test_an_unrecognised_name_stays_custom_and_warns_on_arrivals(
            self, tmp_path):
        model = Bellhop(executable=self._stub(tmp_path, 'ray-solver'),
                        verbose=False)
        assert model._resolved_backend == 'custom'
        assert _command(model, 'base') == [str(tmp_path / 'ray-solver'),
                                                'base']
        with pytest.warns(UserWarning, match='merge'):
            assert _merge(model) is False

    def test_a_recognised_name_reads_arrivals_without_a_warning(
            self, tmp_path):
        model = Bellhop(executable=self._stub(tmp_path, 'bellhop'),
                        verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert _merge(model) is False


class TestALineSourceArrivalsCarryTheGreensFunctionPhase:
    """A 2-D line source carries the Green's function's ``exp(-i*pi/4)``
    against the ``e^{-ikR}`` carrier, and ``ArrMod.f90:103-104`` scales
    its arrivals by a purely real ``4*sqrt(pi)``. The ARRIVALS result the
    caller receives carries that phase itself — its ``phases`` sit 45 deg
    behind a point source's for the same rays — and the broadband routes
    synthesise from those arrivals as they are, so the offset enters
    exactly once: each arrival's ``phases`` entry is exactly −45° behind the
    point run's, and the transfer function is built from those arrivals
    with no further offset (the line/point H(f) ratio only approximates
    −45°, since a point source weights each beam by √cos α)."""

    FREQS = np.linspace(900.0, 1100.0, 5)

    @staticmethod
    def _model():
        return Bellhop(verbose=False, n_beams=20, beam_type='G')

    @classmethod
    def _run(cls, source_type, run_mode):
        from uacpy.tests.conftest import make_pekeris
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return cls._model().run(
                make_pekeris(name='line-' + source_type, bathymetry=100.0),
                Source(depths=50.0, frequencies=1000.0,
                       source_type=source_type),
                Receiver(depths=[50.0], ranges=[500.0]),
                run_mode=run_mode,
                # ARRIVALS runs at the Source frequency; only the broadband
                # routes take a grid.
                **({} if run_mode == RunMode.ARRIVALS
                   else {'frequencies': cls.FREQS}))

    def test_line_arrival_phases_sit_45_deg_behind_point_ones(self):
        point = self._run('point', RunMode.ARRIVALS)
        line = self._run('line', RunMode.ARRIVALS)
        assert len(line.arrivals) == len(point.arrivals) > 0
        i_p, i_l = np.argsort(point.delays), np.argsort(line.delays)
        np.testing.assert_allclose(line.delays[i_l], point.delays[i_p],
                                   rtol=0, atol=1e-9)
        diff = np.degrees(line.phases[i_l] - point.phases[i_p])
        np.testing.assert_allclose(diff, -45.0, rtol=0, atol=1e-6)

    def test_the_broadband_result_carries_the_phase_exactly_once(self):
        # A point run's phases carry no source-shape offset (influence.f90
        # weights its beams by sqrt(cos(alpha)), so only its amplitudes
        # differ); the line run's own amplitudes with those phases and ONE
        # explicit -pi/4 is the reference H(f). Applying the offset twice
        # or not at all lands 45 deg away from it.
        from uacpy.models.bellhop._output import _LINE_SOURCE_PHASE
        point = self._run('point', RunMode.ARRIVALS).by_receiver[0][0][0]
        line = self._run('line', RunMode.ARRIVALS).by_receiver[0][0][0]
        tf = self._run('line', RunMode.BROADBAND)
        i_p, i_l = np.argsort(point['delays']), np.argsort(line['delays'])
        reference = {k: (v[i_l] if isinstance(v, np.ndarray) else v)
                     for k, v in line.items()}
        reference['phases'] = point['phases'][i_p]
        expected = _cell_tf(reference, self.FREQS,
                                  phase_offset=_LINE_SOURCE_PHASE)
        np.testing.assert_allclose(np.asarray(tf.data)[0, 0, :], expected,
                                   rtol=1e-9, atol=0)

    def test_the_transfer_function_is_built_from_the_arrivals_as_they_are(self):
        # The same offset must not be added again on synthesis: H(f) is
        # ``arrival_grid_transfer_function`` of the returned cell with no extra phase.
        arr = self._run('line', RunMode.ARRIVALS)
        tf = self._run('line', RunMode.BROADBAND)
        expected = _cell_tf(arr.by_receiver[0][0][0], self.FREQS)
        np.testing.assert_allclose(np.asarray(tf.data)[0, 0, :], expected,
                                   rtol=1e-12, atol=0)


class TestTheFanMissCheckFollowsThePairedGrid:
    """``grid_type='I'`` pairs ``depths[i]`` with ``ranges[i]``
    (``bellhop.f90:202-206``), so the launch-angle check reasons over those
    pairs, not over the depth x range product a rectilinear deck spans."""

    @staticmethod
    def _check(grid_type, depths, ranges):
        with recorded_warnings() as rec:
            _say_fan_miss(
                Bellhop(launch_angles_deg=(-80.0, 80.0), grid_type=grid_type,
                        verbose=False),
                Source(depths=10.0, frequencies=2000.0),
                Receiver(depths=depths, ranges=ranges))
        return [str(w.message) for w in rec
                if 'launch angle outside' in str(w.message)]

    def test_a_paired_grid_whose_pairs_are_inside_the_fan_is_silent(self):
        # Pairs (90 m, 500 m) -> 9.1 deg and (10 m, 10 m) -> 0 deg are both
        # inside; only the product's (90 m, 10 m) at 82.9 deg would miss.
        assert self._check('I', [10.0, 90.0], [10.0, 500.0]) == []
        assert len(self._check('R', [10.0, 90.0], [10.0, 500.0])) == 1

    def test_a_paired_grid_counts_its_pairs(self):
        # (85 m, 10 m) at 82.4 deg misses, (90 m, 500 m) at 9.1 deg does
        # not: 1 of 2 pairs, where the product would say 2 of 4.
        said = self._check('I', [85.0, 90.0], [10.0, 500.0])
        assert len(said) == 1
        assert '1 of 2 source/receiver pairs' in said[0], said[0]


def _layered_bottom_env(name='layered'):
    """5 m of silt over a sand half-space: a layer stack the single-halfspace
    ``.env`` cannot carry, so ``run`` auto-routes through BOUNCE."""
    from uacpy.core.bottom import Bottom
    return Environment(
        name=name, bathymetry=100.0, ssp=1500.0,
        bottom=Bottom.from_presets([('silt', 5.0)], halfspace='sand'))


class TestBroadbandOverALayeredBottomWarnsOfTheFcTable:
    """BROADBAND/TIME_SERIES trace once at fc, and a layered bottom routes
    that trace through BOUNCE, which tabulates R(theta) at fc alone — while a
    layer stack's R(theta) moves with frequency (bounce.md). The run says so;
    a half-space bottom, or auto_bounce=False, takes no BOUNCE table."""

    def _warnings(self, env, auto_bounce=True, run_mode=RunMode.BROADBAND):
        # run_settings says what run would, and launches nothing.
        model = Bellhop(verbose=False, auto_bounce=auto_bounce)
        with recorded_warnings() as caught:
            model.run_settings(
                env, Source(depths=[10.0], frequencies=500.0),
                Receiver(depths=[30.0], ranges=[500.0]),
                run_mode,
                source_waveform=(np.hanning(64) if run_mode
                                 == RunMode.TIME_SERIES else None),
                sample_rate=(8000.0 if run_mode == RunMode.TIME_SERIES
                             else None))
        return [str(w.message) for w in caught
                if 'BOUNCE reflection table' in str(w.message)]

    @pytest.mark.parametrize('run_mode', [RunMode.BROADBAND,
                                          RunMode.TIME_SERIES])
    def test_a_layered_bottom_warns(self, run_mode):
        said = self._warnings(_layered_bottom_env(), run_mode=run_mode)
        assert len(said) == 1 and 'fc = 500 Hz' in said[0], said

    def test_a_half_space_bottom_is_silent(self):
        env = Environment(name='hs', bathymetry=100.0, ssp=1500.0,
                          bottom=BoundaryProperties(
                              acoustic_type='half-space', sound_speed=1700.0,
                              density=1.8, attenuation=0.5))
        assert not self._warnings(env)

    def test_auto_bounce_off_is_silent(self):
        assert not self._warnings(_layered_bottom_env(), auto_bounce=False)


class TestBounceTableRidesOnTheMultiDepthField:
    """A multi-depth Source joins the ``.shd`` into one Field over
    ``source_depth``. BOUNCE reads no source depth, so the in-memory table
    the constructor promises as ``result.components['bounce']`` is one
    table: the joined field holds it, and the field of every depth carries
    the same object."""

    def test_multi_depth_tl_field_carries_the_one_table_on_each_depth(self):
        from uacpy.core.results._superposition import split_source_depths
        rcv = Receiver(depths=[30.0], ranges=[500.0])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = Bellhop(verbose=False, beam_type='G').run(
                _layered_bottom_env(), Source(depths=[10.0, 20.0],
                                              frequencies=500.0),
                rcv, run_mode=RunMode.COHERENT_TL)
        assert isinstance(result, Field)
        assert result.coords['source_depth'].tolist() == [10.0, 20.0]
        table = result.components.get('bounce')
        assert table is not None
        assert 'bounce_result' not in result.metadata
        tables = [plane.components.get('bounce')
                  for plane in split_source_depths(result)]
        assert len(tables) == 2
        assert all(t is table for t in tables), tables
        # The table is propagated to the receiver's own range.
        assert table.run_settings.engine.rmax == pytest.approx(rcv.range_max)


class TestEigenraysMultiDepthSharesOneBounceTable:
    """The BOUNCE table depends on the bottom column and frequency only, so a
    multi-depth EIGENRAYS run on a layered bottom resolves it once, in the
    call's settings, and says so once; the per-depth loop launches it beside
    each depth's ``.ray`` run, and every depth reads the same table."""

    def test_three_source_depths_share_one_table_and_warn_once(self):
        with recorded_warnings() as rec:
            result = Bellhop(verbose=False, beam_type='G').run(
                _layered_bottom_env('eig'),
                Source(depths=[10.0, 20.0, 30.0], frequencies=500.0),
                Receiver(depths=[50.0], ranges=[500.0]),
                run_mode=RunMode.EIGENRAYS)
        routed = [w for w in rec
                  if 'auto-routing through BOUNCE' in str(w.message)]
        assert isinstance(result, ResultStack) and len(result) == 3
        assert len(routed) == 1, [str(w.message) for w in routed]
        tables = [slab.components['bounce'] for slab in result.slabs]
        for table in tables[1:]:
            np.testing.assert_array_equal(table.magnitude, tables[0].magnitude)
            np.testing.assert_array_equal(table.angles_deg, tables[0].angles_deg)
        assert all(slab.run_settings.engine.bounce
                   == result.slabs[0].run_settings.engine.bounce
                   for slab in result.slabs)


class TestAllZeroReceiverRangesResolveTheBounceRangeByFallback:
    """``rmax`` for the spawned Bounce is left to ``Bounce.run``, whose own
    rule reads ``receiver.range_max`` and falls back to 10 km when every
    receiver sits at r = 0 — so the hidden BOUNCE pass cannot fail on an
    input Bellhop itself accepts (its ``r_box`` uses the same fallback)."""

    def test_zero_range_rays_on_a_layered_bottom_uses_the_10_km_fallback(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = Bellhop(verbose=False).run(
                _layered_bottom_env('r0'),
                Source(depths=25.0, frequencies=500.0),
                Receiver(depths=[50.0], ranges=[0.0]),
                run_mode=RunMode.RAYS)
        assert isinstance(result, Rays)
        assert result.components['bounce'].run_settings.engine.rmax == \
            pytest.approx(10000.0)


class TestPairedGridReceiversBelowTheSeafloorAreNoData:
    """BELLHOP clamps a receiver below the deck bottom onto it
    (``misc/SourceReceiverPositions.f90:136-139``), and a receiver just under
    a shoaling ``.bty`` still collects a finite Gaussian-beam tail from the
    rays skimming the seafloor; both cells are no-data and come back NaN
    under the requested depth, on the paired ``grid_type='I'`` grid exactly as
    on the rectilinear one. Pair ``i`` is ``(depths[i], ranges[i])``."""

    _SRC = Source(depths=25.0, frequencies=500.0)

    @staticmethod
    def _run(env, rcv, run_mode, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return Bellhop(verbose=False, beam_type='G', grid_type='I').run(
                env, TestPairedGridReceiversBelowTheSeafloorAreNoData._SRC,
                rcv, run_mode=run_mode, **kwargs)

    def test_tl_cell_below_the_deck_bottom_is_nan_under_its_own_depth(self):
        env = Environment(name='flat', bathymetry=100.0, ssp=1500.0)
        rcv = Receiver(depths=[30.0, 120.0], ranges=[500.0, 1000.0])
        result = self._run(env, rcv, RunMode.COHERENT_TL)
        assert list(result.coords) == ['range']
        tl = np.asarray(result.dB, dtype=float)
        assert np.isfinite(tl[0]) and np.isnan(tl[1]), tl
        np.testing.assert_allclose(result.aux_coords['receiver_depth'][1],
                                   [30.0, 120.0])

    def test_tl_cell_below_a_shoaling_seafloor_is_nan(self):
        env = Environment(name='shoal', ssp=1500.0,
                          bathymetry=[(0.0, 100.0), (1000.0, 100.0),
                                      (2000.0, 40.0)])
        # (50 m, 500 m) sits in 100 m of water; (55 m, 1800 m) is 3 m under
        # the 52 m seafloor there — inside the beam tails, so the engine
        # alone returns a finite level — and above the 100 m deck bottom.
        rcv = Receiver(depths=[50.0, 55.0], ranges=[500.0, 1800.0])
        result = self._run(env, rcv, RunMode.COHERENT_TL)
        tl = np.asarray(result.dB, dtype=float)
        assert np.isfinite(tl[0]) and np.isnan(tl[1]), tl

    def test_broadband_cell_below_the_deck_bottom_is_nan(self):
        env = Environment(name='flat', bathymetry=100.0, ssp=1500.0)
        rcv = Receiver(depths=[30.0, 120.0], ranges=[500.0, 1000.0])
        result = self._run(env, rcv, RunMode.BROADBAND,
                           frequencies=np.array([450.0, 500.0]))
        assert list(result.coords) == ['range', 'frequency']
        assert np.isfinite(result.data[0]).all()
        assert np.isnan(result.data[1]).all()
        np.testing.assert_allclose(result.aux_coords['receiver_depth'][1],
                                   [30.0, 120.0])


class TestCxxRunTimeWarningsReachTheUser:
    """bellhopcxx / bellhopcuda report run-time conditions only on stdout
    (``bellhopcuda/src/util/errors.cpp:26-36, 113-137``): a header ``N
    warning(s) thrown of the following type(s):`` and one
    ``BHC_WARN_<NAME>: ...`` line per condition, nothing in the ``.prt``.
    ``_backend.warn_on_engine_stdout_warnings`` re-emits those lines as the
    same ``UserWarning`` the Fortran ``.prt`` scan raises, so ``n_beams=3``
    warns on every backend. The block below is the stdout captured from a
    real ``backend='cxx'`` run of that case."""

    _CAPTURED_CXX_STDOUT = (
        "setup: 0.624820 ms\n"
        "Preprocess: 0.140358 ms\n"
        "1 warning(s) thrown of the following type(s):\n"
        "BHC_WARN_TOO_FEW_BEAMS: Nalpha is too small; there may be gaps "
        "between the beams\n"
        "Run: 0.379620 ms\n"
        "Postprocess: 0.000781 ms\n"
        "writeout: 0.052021 ms\n"
    )

    @staticmethod
    def _model(backend):
        from uacpy.core.exceptions import ExecutableNotFoundError
        try:
            return Bellhop(backend=backend, n_beams=3, verbose=False)
        except ExecutableNotFoundError:
            pytest.skip(f"no {backend} Bellhop binary on this host")

    @pytest.mark.parametrize('backend', ['fortran', 'cxx'])
    def test_too_few_beams_warns_on_every_backend(self, backend):
        env = Environment(name='fewbeams', bathymetry=100.0, ssp=1500.0)
        src = Source(depths=50.0, frequencies=500.0)
        rcv = Receiver(depths=np.linspace(5.0, 95.0, 10),
                       ranges=np.linspace(100.0, 3000.0, 30))
        with pytest.warns(UserWarning,
                          match=r"Too few beams|BHC_WARN_TOO_FEW_BEAMS"):
            self._model(backend).run(env, src, rcv)

    def test_the_captured_block_is_re_emitted_verbatim(self):
        model = Bellhop(verbose=False)
        with pytest.warns(UserWarning) as record:
            _scan_stdout(model, self._CAPTURED_CXX_STDOUT)
        messages = [str(w.message) for w in record
                    if 'BHC_WARN' in str(w.message)]
        assert len(messages) == 1, messages
        assert "reported 1 non-fatal warning(s)" in messages[0]
        assert ("BHC_WARN_TOO_FEW_BEAMS: Nalpha is too small; there may be "
                "gaps between the beams") in messages[0]
        # The timing lines around the block are not diagnoses.
        assert 'Preprocess' not in messages[0]

    def test_the_moved_receiver_line_says_uacpy_returns_nan(self):
        """ReadSzRz moves a receiver below the bottom boundary onto it and
        says so in the .prt; uacpy NaNs that receiver instead of returning
        the moved one's field, and the forwarded line says which holds."""
        env = Environment(name='below', bathymetry=100.0, ssp=1500.0,
                          bottom='sand')
        with recorded_warnings() as record:
            field = self._model('fortran').run(
                env, Source(depths=50.0, frequencies=200.0),
                Receiver(depths=[50.0, 150.0], ranges=[500.0, 1000.0]))
        said = [str(w.message) for w in record if 'moved up' in str(w.message)]
        assert len(said) == 1 and 'uacpy returns NaN' in said[0], said
        assert np.all(np.isnan(field.data[1]))

    def test_a_stdout_without_the_block_warns_nothing(self):
        model = Bellhop(verbose=False)
        with recorded_warnings() as record:
            _scan_stdout(
                model, "setup: 0.6 ms\nPreprocess: 0.1 ms\nRun: 0.4 ms\n")
        assert [str(w.message) for w in record] == []

    @pytest.mark.requires_binary
    def test_the_arrivals_launcher_is_scanned_and_the_engine_stays_silent(
            self, monkeypatch):
        """One launcher serves every mode (``Bellhop._launch``), so an ARRIVALS
        run's stdout goes through the same scan as a TL run's. What differs
        is the engine: both ports raise ``Too few beams`` only on a coherent
        TL run — ``bellhop.f90:255`` tests ``RunType(1:1) == 'C'`` and
        ``bellhopcuda/src/trace.hpp:127`` tests ``IsCoherentRun(Beam)`` — so
        the same ``n_beams=3`` fan that warns in COHERENT_TL is silent in
        ARRIVALS on every backend, and uacpy passes that silence through
        rather than inventing a warning the engine did not make."""
        from uacpy.models.bellhop import _model as bellhop_model
        seen = []
        original = bellhop_model.warn_on_engine_stdout_warnings

        def spy(stdout, **kw):
            seen.append(stdout)
            return original(stdout, **kw)

        monkeypatch.setattr(bellhop_model, 'warn_on_engine_stdout_warnings',
                            spy)
        env = Environment(name='fewbeams', bathymetry=100.0, ssp=1500.0)
        src = Source(depths=50.0, frequencies=500.0)
        rcv = Receiver(depths=np.linspace(5.0, 95.0, 10),
                       ranges=np.linspace(100.0, 3000.0, 30))
        with recorded_warnings() as record:
            self._model('cxx').run(env, src, rcv, run_mode=RunMode.ARRIVALS)
        assert len(seen) == 1, "the ARRIVALS launcher skipped the scan"
        assert 'Run:' in seen[0], seen[0]
        assert 'BHC_WARN' not in seen[0], seen[0]
        assert not [w for w in record if 'BHC_WARN' in str(w.message)]
        # The discriminating side, same fan and grid: the TL run warns.
        with pytest.warns(UserWarning, match='BHC_WARN_TOO_FEW_BEAMS'):
            self._model('cxx').run(env, src, rcv,
                                   run_mode=RunMode.COHERENT_TL)


class TestPairedReceiverUnderTheSeabedHasNoTrace:
    """A paired-grid receiver below a shoaling seabed is clamped onto the
    boundary by BELLHOP, so what it reports is another receiver. TL and
    BROADBAND returned no-data there while TIME_SERIES handed back a real
    trace labelled with the buried depth."""

    @staticmethod
    def _run(run_mode, **kw):
        env = Environment(name='shoal', bathymetry=[(0.0, 100.0),
                                                    (6000.0, 40.0)],
                          ssp=1500.0)
        rcv = Receiver(depths=[20.0, 50.0, 80.0],
                       ranges=[1000.0, 2000.0, 5500.0])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return Bellhop(verbose=False, grid_type='I', n_beams=300).run(
                env, Source(depths=10.0, frequencies=500.0), rcv, run_mode,
                **kw)

    def test_time_series_is_no_data_where_tl_is(self):
        tl = self._run(RunMode.COHERENT_TL)
        ts = self._run(RunMode.TIME_SERIES,
                       source_waveform=np.hanning(64), sample_rate=8000.0)
        buried_tl = ~np.isfinite(np.asarray(tl.dB, dtype=float)).ravel()
        buried_ts = np.all(~np.isfinite(np.asarray(ts.data)), axis=-1)
        assert buried_tl.tolist() == [False, False, True]
        assert buried_ts.tolist() == buried_tl.tolist()
        assert np.asarray(ts.aux_coords['receiver_depth'][1]) == pytest.approx(
            [20.0, 50.0, 80.0])



class TestEveryEngineOpensItsRecordWhereAsked:
    """``output_duration`` is the record's length and ``t_start`` its first
    sample on every engine. Bellhop used to anchor an ``output_duration``
    record at emission, all zeros for a receiver 1 km out (first echo
    0.67 s) while the IFFT engines placed theirs at the arrival."""

    @staticmethod
    def _run(model, **kw):
        env = uacpy.Environment(name='once', bathymetry=300.0, ssp=1500.0)
        fs = 8e3
        t = np.arange(int(0.005 * fs)) / fs
        wf = np.hanning(t.size) * np.sin(2 * np.pi * 2e3 * t)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return model.run(env, Source(depths=150.0, frequencies=2e3),
                             Receiver(depths=[150.0], ranges=[1000.0]),
                             run_mode=RunMode.TIME_SERIES, source_waveform=wf,
                             sample_rate=fs, **kw)

    def test_an_output_duration_alone_holds_the_arrival(self):
        ts = self._run(Bellhop(verbose=False), output_duration=0.3)
        assert np.max(np.abs(np.asarray(ts.data))) > 0.0
        assert float(ts.coords['time'][0]) > 0.5

    def test_t_start_opens_the_record_there(self):
        ts = self._run(Bellhop(verbose=False), output_duration=0.3,
                       t_start=0.6)
        assert float(ts.coords['time'][0]) == pytest.approx(0.6, abs=1e-6)

    def test_a_mode_without_a_record_says_t_start_is_ignored(self):
        env = uacpy.Environment(name='tl', bathymetry=300.0, ssp=1500.0)
        with pytest.warns(UserWarning, match="t_start"):
            Bellhop(verbose=False).run(
                env, Source(depths=150.0, frequencies=2e3),
                Receiver(depths=[150.0], ranges=[1000.0]),
                run_mode=RunMode.COHERENT_TL, t_start=0.5)


class TestIncoherentTlIsARealLevelOnEveryEngine:
    """Bellhop kept INCOHERENT/SEMICOHERENT TL as complex Pa with a zero
    phase that looked real; Kraken and Scooter store real dB. One payload
    for magnitude sums (Theo's decision on MODELS_B-37)."""

    @pytest.mark.parametrize('mode', [RunMode.INCOHERENT_TL,
                                      RunMode.SEMICOHERENT_TL])
    def test_the_payload_is_real_dB(self, mode):
        env = uacpy.Environment(name='inc', bathymetry=100.0, ssp=1500.0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            coh = Bellhop(verbose=False).run(
                env, Source(depths=20.0, frequencies=200.0),
                Receiver(depths=[30.0, 60.0], ranges=[500.0, 1000.0]),
                run_mode=RunMode.COHERENT_TL)
            inc = Bellhop(verbose=False).run(
                env, Source(depths=20.0, frequencies=200.0),
                Receiver(depths=[30.0, 60.0], ranges=[500.0, 1000.0]),
                run_mode=mode)
        assert not np.iscomplexobj(inc.data)
        assert inc.unit == 'dB'
        with pytest.raises((AttributeError, ConfigurationError),
                           match='requires complex data'):
            inc.phase
        # The level is the same kind of number as the coherent one's .dB.
        assert np.all(np.abs(np.asarray(inc.dB) - np.asarray(coh.dB)) < 30.0)


class TestCoherentFieldDtypeMatchesTheOtherEngines:
    """The ``.shd`` payload is complex64; Kraken, Scooter and OASP return
    complex128, so Bellhop upcasts to the same dtype on every source type."""

    @pytest.mark.parametrize('source_type', ['point', 'line'])
    def test_coherent_tl_is_complex128(self, source_type):
        env = Environment(name='dtype', bathymetry=100.0, ssp=1500.0)
        field = Bellhop(verbose=False).compute_field(
            env, Source(depths=[20.0], frequencies=200.0,
                        source_type=source_type),
            Receiver(depths=[30.0, 50.0], ranges=[500.0, 1000.0]))
        assert field.data.dtype == np.complex128


def test_a_sub_200_hz_band_announces_francois_garrison_once():
    """The broadband synthesis and its band check each evaluate the band in
    one vectorised call, so Francois-Garrison's out-of-range notice (fitted
    from 200 Hz) names the band once, where a per-frequency loop printed one
    per frequency; the deck's water rows, written at the carrier, name it
    once more. The values are the same kernel either way."""
    from uacpy.core.absorption import FrancoisGarrison
    fg = FrancoisGarrison(temperature=10.0, salinity=35.0, pH=8.0)
    env = Environment(name='fg', bathymetry=100.0, ssp=1500.0,
                      absorption=fg,
                      bottom=BoundaryProperties(acoustic_type='half-space',
                                                sound_speed=1700.0,
                                                density=1.8, attenuation=0.5))
    with recorded_warnings() as caught:
        Bellhop(verbose=False).run(
            env, Source(depths=30.0, frequencies=100.0),
            Receiver(depths=[50.0], ranges=[2000.0]),
            run_mode=RunMode.BROADBAND,
            frequencies=np.linspace(60.0, 140.0, 9))
    notices = {str(w.message) for w in caught
               if str(w.message).startswith('FrancoisGarrison:')}
    assert len(notices) == 2, notices
    assert sum('1 of 1 frequency (100 Hz)' in n for n in notices) == 1
    assert sum('10 of 10 frequencies (60-140 Hz)' in n for n in notices) == 1
    band = fg.table([100.0, 60.0, 140.0], depths=0.0, units='dB/m').data
    single = [float(np.ravel(fg._alpha_dB_per_m_at_frequency(f, 0.0))[0])
              for f in (100.0, 60.0, 140.0)]
    np.testing.assert_array_equal(band, single)


@pytest.mark.parametrize('backend', ['fortran', None])
def test_an_accented_environment_name_reads_back_its_ray_file(backend):
    """The ``.ray`` header carries ``Title(1:70)`` after a 9-byte prefix
    (``Bellhop/ReadEnvironmentBell.f90:45-46,557``); measured before deck
    titles were folded to ASCII, ``'é'*79`` put half a character there and
    ``read_ray_file`` failed with ``UnicodeDecodeError``."""
    env = uacpy.Environment(name='é' * 79, bathymetry=100.0, ssp=1500.0)
    kw = {} if backend is None else {'backend': backend}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rays = Bellhop(**kw).run(
            env, uacpy.Source(depths=30.0, frequencies=200.0),
            uacpy.Receiver(depths=[50.0], ranges=[1000.0]),
            run_mode=RunMode.RAYS)
    assert isinstance(rays, Rays)


class TestPortsArrivalsMemoryBudget:
    """bellhopcxx / bellhopcuda size and zero-fill the ARRIVALS table from
    their whole memory budget (``src/mode/arr.hpp:83-105``), 4 GiB by
    default: measured peak RSS 4.17 GiB for a 10 x 20 grid that Fortran ran
    in 18 MB. ARRIVALS launches pass ``-mem=`` sized from the grid."""

    def test_budget_is_clamped_and_scales_with_the_grid(self):
        from uacpy.models.bellhop._backend import (
            _BHC_ARRIVALS_MEMORY_CAP, _BHC_ARRIVALS_MEMORY_FLOOR,
            _ports_arrivals_memory_bytes)
        assert _ports_arrivals_memory_bytes(1, 1) == _BHC_ARRIVALS_MEMORY_FLOOR
        assert (_ports_arrivals_memory_bytes(1, 4000)
                < _ports_arrivals_memory_bytes(2, 4000)
                <= _BHC_ARRIVALS_MEMORY_CAP)
        assert _ports_arrivals_memory_bytes(10, 200_000) \
            == _BHC_ARRIVALS_MEMORY_CAP

    def test_the_port_command_carries_the_budget_in_kib(self, tmp_path):
        stub = tmp_path / 'bellhopcxx'
        stub.write_text('')
        stub.chmod(0o755)
        model = Bellhop(executable=stub, verbose=False)
        cmd = _command(model, 'base', memory_bytes=64 * 1024 ** 2)
        assert cmd == [str(stub), '--2D', '-mem=65536KiB', 'base']
        assert _command(model, 'base') == [str(stub), '--2D', 'base']

    def test_the_budget_keeps_every_arrival_the_default_budget_keeps(
            self, monkeypatch):
        from uacpy.core.exceptions import ExecutableNotFoundError
        from uacpy.models.bellhop import _backend as bellhop_backend
        try:
            model = Bellhop(backend='cxx', verbose=False)
        except ExecutableNotFoundError:
            pytest.skip("no cxx Bellhop binary on this host")
        env = Environment(name='arrmem', bathymetry=100.0, ssp=1500.0)
        src = Source(depths=[30.0, 60.0], frequencies=500.0)
        rcv = Receiver(depths=np.linspace(5.0, 95.0, 10),
                       ranges=np.linspace(100.0, 3000.0, 30))

        def counts():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = model.run(env, src, rcv, run_mode=RunMode.ARRIVALS)
            slabs = (result.slabs if isinstance(result, ResultStack)
                     else [result])
            return sorted((i, a['depth_idx'], a['range_idx'])
                          for i, slab in enumerate(slabs)
                          for a in slab.arrivals)

        sized = counts()
        asked = []

        def no_budget(n_sources, n_receivers):
            asked.append((n_sources, n_receivers))
            return None

        monkeypatch.setattr(bellhop_backend, '_ports_arrivals_memory_bytes',
                            no_budget)
        assert sized == counts()
        # The equality is vacuous unless the second run read the patched
        # budget: two budgeted runs agree too.
        assert asked, "the arrivals run never asked for a memory budget"


class TestEigenraysWarnsBeforeALargeRayFile:
    """An EIGENRAYS ``.ray`` holds a whole trajectory per (beam, receiver)
    hit; with the automatic fan a 10 x 20 grid at 1 kHz passed 3 GB. The
    warning fires before launch, from the receiver count times the beam count
    the deck will get."""

    @staticmethod
    def _stopped_run(monkeypatch, n_beams, n_depths, n_ranges):
        class _Stop(RuntimeError):
            pass

        monkeypatch.setattr(
            Bellhop, '_launch',
            lambda self, inputs, deck: (_ for _ in ()).throw(_Stop()))
        env = Environment(name='eig', bathymetry=100.0, ssp=1500.0)
        rcv = Receiver(depths=np.linspace(10.0, 90.0, n_depths),
                       ranges=np.linspace(1000.0, 5000.0, n_ranges))
        with recorded_warnings() as record:
            with pytest.raises(_Stop):
                Bellhop(backend='fortran', n_beams=n_beams,
                        verbose=False).run(
                    env, Source(depths=30.0, frequencies=1000.0), rcv,
                    run_mode=RunMode.EIGENRAYS)
        return [w for w in record if 'ray-receiver pairs' in str(w.message)]

    def test_pairs_at_the_threshold_are_silent(self, monkeypatch):
        assert self._stopped_run(monkeypatch, 1000, 4, 5) == []

    def test_one_pair_past_the_threshold_warns(self, monkeypatch):
        assert self._stopped_run(monkeypatch, 1001, 4, 5)

    def test_the_automatic_fan_on_a_small_grid_warns(self, monkeypatch):
        assert self._stopped_run(monkeypatch, 0, 10, 20)

    def test_the_automatic_fan_count_follows_anglemod(self):
        env = Environment(name='eig', bathymetry=100.0, ssp=1500.0)
        rcv = Receiver(depths=[50.0], ranges=[5000.0])
        n = eigenray_beam_count(
            env, Source(depths=30.0, frequencies=1000.0), rcv, n_beams=0)
        assert n == int(np.pi / np.arctan(100.0 / 50000.0))


class TestAFullArrivalTableIsReported:
    """No engine reports dropping arrivals: Fortran replaces the weakest once
    a cell is full (``ArrMod.f90:49-59``), the ports keep the first to arrive
    (``arrivals.hpp:113-119``), and both log only the capacity to the
    ``.prt``. A cell at that capacity is reported."""

    @staticmethod
    def _check(tmp_path, n_in_cell):
        import types
        (tmp_path / 'm.prt').write_text('\n ( Maximum # of arrivals = 3 )\n')
        result = types.SimpleNamespace(
            by_receiver=[[[{'n_arrivals': 1}, {'n_arrivals': n_in_cell}]]])
        with recorded_warnings() as record:
            warn_if_arrival_table_filled(result, tmp_path, 'm')
        return [w for w in record if 'full capacity' in str(w.message)]

    def test_a_cell_below_capacity_is_silent(self, tmp_path):
        assert self._check(tmp_path, 2) == []

    def test_a_cell_at_capacity_warns(self, tmp_path):
        assert self._check(tmp_path, 3)

    def test_the_ports_capacity_notice_is_passed_through(self, tmp_path):
        stub = tmp_path / 'bellhopcxx'
        stub.write_text('')
        stub.chmod(0o755)
        model = Bellhop(executable=stub, verbose=False)
        with pytest.warns(UserWarning, match='Only enough memory'):
            _scan_stdout(
                model, 'setup: 1 ms\n'
                'Only enough memory to allocate up to 5 arrivals per '
                'receiver\nRun: 1 ms\n')


class TestAutoSelectedCudaWithoutADevice:
    """A CUDA build on a host, job or container with no visible GPU exits 1
    with ``cudaErrorNoDevice`` (``bellhopcuda/src/api.cpp:148``). With
    ``backend=None`` the run moves to the next installed binary, warns once
    and reports the backend that ran; ``backend='cuda'`` raises."""

    @staticmethod
    def _cuda_or_skip():
        from uacpy.core.exceptions import ExecutableNotFoundError
        try:
            model = Bellhop(verbose=False)
        except ExecutableNotFoundError:
            pytest.skip("no Bellhop binary on this host")
        if model._resolved_backend != 'cuda':
            pytest.skip("auto selection does not pick bellhopcuda here")
        return model

    @staticmethod
    def _case():
        env = Environment(name='nodev', bathymetry=100.0, ssp=1500.0)
        return (env, Source(depths=30.0, frequencies=200.0),
                Receiver(depths=[50.0], ranges=[1000.0, 2000.0]))

    def test_auto_selection_runs_the_next_binary(self, monkeypatch):
        from uacpy.models.bellhop import _backend as bellhop_backend
        monkeypatch.setattr(bellhop_backend, '_cuda_found_no_device', False)
        model = self._cuda_or_skip()
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
        with pytest.warns(UserWarning, match='found no CUDA device'):
            result = model.run(*self._case(), run_mode=RunMode.COHERENT_TL)
        assert model._resolved_backend == 'cxx'
        assert result.backend == 'cxx'
        # The finding lands in the flag this test reset, so the reset
        # reached the namespace the selection reads.
        assert bellhop_backend._cuda_found_no_device is True

    def test_an_explicit_cuda_backend_raises(self, monkeypatch):
        from uacpy.core.exceptions import (
            ExecutableNotFoundError, ModelExecutionError)
        from uacpy.models.bellhop import _backend as bellhop_backend
        monkeypatch.setattr(bellhop_backend, '_cuda_found_no_device', False)
        try:
            model = Bellhop(backend='cuda', verbose=False)
        except ExecutableNotFoundError:
            pytest.skip("no bellhopcuda binary on this host")
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
        with pytest.raises(ModelExecutionError, match='cudaErrorNoDevice'):
            model.run(*self._case(), run_mode=RunMode.COHERENT_TL)
        # An explicit backend raises without moving later auto-selections.
        assert bellhop_backend._cuda_found_no_device is False


# ── the protocol stages (w2-bellhop) ────────────────────────────────────────

def _pekeris_env(depth=100.0, ssp=1500.0, **kw):
    return make_pekeris(name='pekeris', bathymetry=depth, ssp=ssp,
                        sound_speed=1650.0, density=1.5, **kw)


def _layered_env(**kw):
    from uacpy.core.bottom import Bottom
    return Environment(
        name='layered', bathymetry=100.0, ssp=1500.0,
        bottom=Bottom.from_presets([('silt', 5.0)], halfspace='sand'), **kw)


_SRC = Source(depths=[40.0], frequencies=200.0)
_RCV = Receiver(depths=[30.0, 60.0], ranges=np.linspace(400.0, 2000.0, 5))


class TestRunWithBounceRunsThroughTheTemplate:
    """``run_with_bounce`` is ``run`` with the seabed routed through BOUNCE,
    so it applies the source weights, loops a multi-depth BROADBAND source
    and stamps the settings, as ``run`` does (RA-WAVE-1)."""

    @staticmethod
    def _model():
        return Bellhop(backend='fortran', verbose=False, beam_type='G',
                       n_beams=301)

    def test_a_weighted_source_is_scaled_by_its_weight(self):
        model = self._model()
        unit = model.run_with_bounce(_layered_env(), _SRC, _RCV,
                                     run_mode=RunMode.COHERENT_TL)
        weighted = model.run_with_bounce(
            _layered_env(),
            Source(depths=[40.0], frequencies=200.0, weights=[2.0 - 1.0j]),
            _RCV, run_mode=RunMode.COHERENT_TL)
        ok = np.isfinite(unit.data) & (np.abs(unit.data) > 0)
        ratio = np.asarray(weighted.data)[ok] / np.asarray(unit.data)[ok]
        np.testing.assert_allclose(ratio, 2.0 - 1.0j, rtol=1e-12)

    def test_a_two_depth_broadband_source_returns_one_field_over_source_depth(
            self):
        """The field of both depths on one ``source_depth`` axis, carrying
        the one BOUNCE table and, since each depth has arrivals of its own,
        a ``ResultStack`` of the per-depth Arrivals."""
        from uacpy.core.results._superposition import split_source_depths
        field = self._model().run_with_bounce(
            _layered_env(), Source(depths=[30.0, 60.0], frequencies=200.0),
            _RCV, run_mode=RunMode.BROADBAND,
            frequencies=np.array([180.0, 200.0, 220.0]))
        assert isinstance(field, Field)
        assert list(field.coords)[0] == 'source_depth'
        assert field.coords['source_depth'].tolist() == [30.0, 60.0]
        assert 'bounce' in field.components
        assert all('bounce' in plane.components
                   for plane in split_source_depths(field))
        arrivals = field.components['arrivals']
        assert isinstance(arrivals, ResultStack)
        assert arrivals.slab_type is Arrivals
        assert arrivals.coordinate.tolist() == [30.0, 60.0]

    def test_the_result_carries_the_route_and_bounce_settings(self):
        result = self._model().run_with_bounce(
            _pekeris_env(), _SRC, _RCV, run_mode=RunMode.COHERENT_TL,
            rmax=3000.0)
        engine = result.run_settings.engine
        assert result.run_mode == RunMode.COHERENT_TL
        assert engine.bounce_origin == 'run_with_bounce'
        assert engine.bounce.mode == RunMode.REFLECTION
        assert engine.bounce.engine.rmax == 3000.0
        assert result.components['bounce'].run_settings == engine.bounce

    def test_a_weighted_incoherent_run_is_refused_as_run_refuses_it(self):
        source = Source(depths=[40.0], frequencies=200.0, weights=[0.5j])
        refusals = []
        for call in (lambda m: m.run(_layered_env(), source, _RCV,
                                     RunMode.INCOHERENT_TL),
                     lambda m: m.run_with_bounce(
                         _layered_env(), source, _RCV,
                         run_mode=RunMode.INCOHERENT_TL)):
            with pytest.raises(ConfigurationError,
                               match='a weight cannot scale it') as info:
                call(self._model())
            refusals.append(str(info.value))
        assert refusals[0] == refusals[1] and "'dB'" in refusals[0]

    def test_run_without_the_route_carries_no_bounce_settings(self):
        engine = self._model().run_settings(_pekeris_env(), _SRC,
                                            _RCV).engine
        assert engine.bounce is None and engine.bounce_origin is None


class TestAPinnedRayBoxInsideTheDomainRunsWithANotice:
    """``bellhop.f90:571-572`` drops a ray whose depth exceeds ``z_box`` or
    whose range exceeds ``r_box``. The box is the caller's control, and
    BELLHOP honours it, so a pinned box inside the domain runs as given
    and a notice says which paths it drops (bellhop.md §7 "Ray-box fitting");
    the boundary is pinned on both sides."""

    @staticmethod
    def _said(model, mode=RunMode.COHERENT_TL):
        with recorded_warnings():
            settings = model.run_settings(_pekeris_env(), _SRC, _RCV, mode)
        return settings.engine, [n.message for n in settings.engine.notices
                                 if 'ray box' in n.message]

    @pytest.mark.parametrize('value', [0.0, -5.0, float('nan'), True, '50'])
    def test_a_non_positive_box_is_refused_at_construction(self, value):
        # A bool or text is no number at all: the knob's reading refuses it
        # before the box's own sign check.
        said = 'positive number|must be a number, not (a bool|text)'
        with pytest.raises(ConfigurationError, match=said):
            Bellhop(backend='fortran', verbose=False, z_box=value)
        with pytest.raises(ConfigurationError, match=said):
            Bellhop(backend='fortran', verbose=False, r_box=value)

    @pytest.mark.parametrize('z_box, noticed', [(100.0, True),
                                                (100.001, False)])
    def test_a_depth_box_at_the_seafloor_runs_with_a_notice(self, z_box,
                                                            noticed):
        engine, said = self._said(
            Bellhop(backend='fortran', verbose=False, z_box=z_box))
        assert engine.z_box == z_box
        assert bool(said) is noticed
        if noticed:
            assert 'seafloor' in said[0] and 'bellhop.f90:571-572' in said[0]

    @pytest.mark.parametrize('r_box, noticed', [(1999.999, True),
                                                (2000.0, False)])
    def test_a_range_box_short_of_the_farthest_receiver_runs_with_a_notice(
            self, r_box, noticed):
        engine, said = self._said(
            Bellhop(backend='fortran', verbose=False, r_box=r_box))
        assert engine.r_box == r_box
        assert bool(said) is noticed
        if noticed:
            assert 'farthest receiver' in said[0]

    def test_a_ray_trace_keeps_the_box_it_is_given_without_a_notice(self):
        """A RAYS run has no receiver to reach: the box is the trace extent
        the caller chose."""
        engine, said = self._said(
            Bellhop(backend='fortran', verbose=False, z_box=50.0,
                    r_box=500.0), RunMode.RAYS)
        assert (engine.z_box, engine.r_box) == (50.0, 500.0)
        assert said == []

    def test_the_same_box_is_noticed_on_a_field_mode(self):
        _, said = self._said(
            Bellhop(backend='fortran', verbose=False, z_box=50.0),
            RunMode.ARRIVALS)
        assert said and 'seafloor' in said[0]

    def test_the_default_box_is_recorded_with_its_origin(self):
        engine = Bellhop(backend='fortran', verbose=False).run_settings(
            _pekeris_env(), _SRC, _RCV).engine
        assert engine.z_box == pytest.approx(120.0)
        assert engine.r_box == pytest.approx(2400.0)
        assert engine.z_box_origin == '1.2 x env.depth'
        assert engine.r_box_origin == '1.2 x receiver.range_max'

    def test_the_default_box_leaves_room_for_returning_paths(self):
        """1.2 x, measured on the ideal wedge (3.06 dB median at 1.01 x,
        0.57 at 1.2 x; test_benchmarks_analytic)."""
        from uacpy.io.bellhop_writer import BELLHOP_RAY_BOX_FACTOR
        assert BELLHOP_RAY_BOX_FACTOR == 1.2


def _never_launch(*_a, **_k):
    raise AssertionError('the binary was launched')


class TestCervenyBeamsRefuseTheMagnitudeSumModes:
    """A Cerveny beam's INCOHERENT/SEMICOHERENT level falls as
    ``n_beams**-0.5`` (``influence.f90:140``, ``:282``, ``:772-779``), so
    those pairs are refused with the geometric beams as the remedy (B3,
    RA-WAVE-14); COHERENT_TL and the geometric beams stay legal."""

    @pytest.mark.parametrize('beam_type', ['C', 'R'])
    @pytest.mark.parametrize('mode', [RunMode.INCOHERENT_TL,
                                      RunMode.SEMICOHERENT_TL])
    def test_refused_at_every_entry_point(self, beam_type, mode,
                                          monkeypatch):
        model = Bellhop(backend='fortran', verbose=False,
                        beam_type=beam_type)
        monkeypatch.setattr(model, '_run_subprocess', _never_launch)
        for call in ('validate_inputs', 'run_settings', 'run'):
            with pytest.raises(ConfigurationError,
                               match=r"n_beams\*\*-0.5") as info:
                getattr(model, call)(_pekeris_env(), _SRC, _RCV, mode)
            assert "beam_type='G', 'B' or 'g'" in info.value.remediation

    @pytest.mark.parametrize('beam_type,mode', [
        ('C', RunMode.COHERENT_TL), ('R', RunMode.COHERENT_TL),
        ('G', RunMode.INCOHERENT_TL), ('B', RunMode.SEMICOHERENT_TL)])
    def test_the_other_pairs_resolve(self, beam_type, mode):
        model = Bellhop(backend='fortran', verbose=False,
                        beam_type=beam_type)
        assert model.run_settings(_pekeris_env(), _SRC, _RCV,
                                  mode).engine.beam_type == beam_type


class TestAReflectionTableIsStagedForTheBinaryThatReadsIt:
    """The writer stages a copied ``reflection_file=`` table for the backend
    the run records: bellhopcxx / bellhopcuda refuse a repeated angle
    (``src/module/reflcoef.hpp:135-141``), so their copy keeps the first row
    of each repeat and says so; the Fortran binary reads the table with no
    monotonicity test (``misc/RefCoef.f90:45-55``), so its copy keeps every
    row."""

    @staticmethod
    def _staged_rows(backend, tmp_path):
        from uacpy.io.refl_io import write_reflection_coefficient
        table = tmp_path / 'seabed.brc'
        angles = np.array([0.0, 10.0, 10.0, 30.0, 60.0, 90.0])
        write_reflection_coefficient(table, angles,
                                     np.full(angles.size, 0.5 + 0j))
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          bottom=BoundaryProperties(
                              acoustic_type='file',
                              reflection_file=str(table)))
        work = tmp_path / backend
        work.mkdir()
        with recorded_warnings() as rec:
            Bellhop(backend=backend, work_dir=str(work), cleanup=False,
                    verbose=False).run(
                env, Source(depths=30.0, frequencies=200.0),
                Receiver(depths=[50.0], ranges=np.linspace(200, 2000, 10)))
        rows = int((work / 'model.brc').read_text().split()[0])
        said = [str(w.message) for w in rec if 'reflcoef.hpp' in str(w.message)]
        return rows, said

    def test_the_fortran_copy_keeps_every_row(self, tmp_path):
        rows, said = self._staged_rows('fortran', tmp_path)
        assert rows == 6 and not said

    def test_a_port_copy_keeps_the_first_row_of_each_repeat(self, tmp_path):
        from uacpy.core.exceptions import ExecutableNotFoundError
        try:
            rows, said = self._staged_rows('cxx', tmp_path)
        except ExecutableNotFoundError:
            pytest.skip("bellhopcxx not installed")
        assert rows == 5 and len(said) == 1


class TestCervenyKnobsResolveFromNone:
    """Every Cerveny knob defaults to ``None``: the deck then carries
    BELLHOP's own curvature 'S', window 5, one image and multiplier 1
    (bellhopMod.f90:22-25) with uacpy's measured 'F' width, and a value set
    on a beam type that does not read it is said to be unused, whatever the
    value."""

    _KNOBS = ('beam_width_type', 'beam_curvature', 'eps_multiplier',
              'r_loop', 'n_image', 'ib_win')

    def test_every_cerveny_knob_defaults_to_none(self):
        defaults = Bellhop.__init__.__kwdefaults__
        assert all(defaults[k] is None for k in self._KNOBS)

    def test_an_unset_knob_writes_the_resolved_values(self, tmp_path):
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        path = tmp_path / 'c.env'
        write_bellhop_env_file(path, _pekeris_env(), _SRC, _RCV,
                               beam_type='C')
        lines = path.read_text().splitlines()
        assert "'FS' 1.000000 1.000000" in lines
        assert "1 5 'P'" in lines

    @pytest.mark.parametrize('knob, value', [('ib_win', 5),
                                             ('beam_curvature', 'S'),
                                             ('r_loop', 1000.0)])
    def test_a_value_set_on_a_hat_beam_is_said_unused_even_at_default(
            self, knob, value):
        with pytest.warns(UserWarning, match=f'Cerveny beam knobs {knob}'):
            Bellhop(beam_type='G', **{knob: value})

    def test_an_unset_knob_on_a_hat_beam_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            Bellhop(beam_type='G')

    @pytest.mark.parametrize('knob, value', [('n_image', 2.5),
                                             ('ib_win', -1),
                                             ('eps_multiplier', 'abc'),
                                             ('r_loop', float('nan'))])
    def test_a_reassigned_cerveny_knob_is_refused_by_the_run(
            self, knob, value):
        """``check_knobs`` runs at construction and again in stage 2, so a
        knob reassigned between the two is refused before any deck."""
        model = Bellhop(beam_type='C', backend='fortran')
        setattr(model, knob, value)
        with pytest.raises(ConfigurationError, match=knob):
            model.run_settings(_pekeris_env(), _SRC, _RCV)

    def test_a_reassigned_velocity_component_on_the_ray_centred_beam_is_refused(
            self):
        model = Bellhop(beam_type='R', backend='fortran')
        model.component = 'V'
        with pytest.raises(UnsupportedFeatureError, match='component'):
            model.run_settings(_pekeris_env(), _SRC, _RCV)

    @pytest.mark.parametrize('beam_type', ['C', 'R'])
    def test_the_record_holds_the_values_the_deck_carries(self, beam_type):
        """DEV.md §2.3: a knob the run derived is the field of the same
        name holding the derived value."""
        engine = Bellhop(beam_type=beam_type, backend='fortran',
                         r_loop=5000.0).run_settings(
            _pekeris_env(), _SRC, _RCV).engine
        assert [getattr(engine, k) for k in self._KNOBS] == \
            ['F', 'S', 1.0, 5000.0, 1, 5]
        assert all(engine.knobs[k] is None
                   for k in self._KNOBS if k != 'r_loop')

    def test_a_hat_beam_records_none_of_them(self):
        engine = Bellhop(beam_type='G', backend='fortran').run_settings(
            _pekeris_env(), _SRC, _RCV).engine
        assert all(getattr(engine, k) is None for k in self._KNOBS)

    def test_the_record_holds_the_ssp_interpolation_the_deck_carries(self):
        model = Bellhop(backend='fortran')
        flat = model.run_settings(_pekeris_env(), _SRC, _RCV).engine
        assert flat.interp_ssp == 'linear'
        assert flat.knobs['interp_ssp'] is None
        from uacpy.core.ssp import SoundSpeedProfile
        rd = Environment(
            name='rd-ssp', bathymetry=100.0,
            ssp=SoundSpeedProfile.from_2d(
                np.array([0.0, 100.0]), np.array([0.0, 12000.0]),
                np.array([[1500.0, 1505.0], [1490.0, 1495.0]])))
        assert model.run_settings(rd, _SRC, _RCV).engine.interp_ssp == 'quad'


class TestUnusedBandKnobsAreSaidUnused:
    """``n_frequencies`` / ``bandwidth_factor`` shape only the band a
    BROADBAND run expands from a one-frequency Source without
    ``frequencies=``; set on any other run, the run says it does not use
    them."""

    @staticmethod
    def _said(model, mode, **kw):
        with recorded_warnings() as caught:
            model.run_settings(_pekeris_env(), _SRC, _RCV, mode, **kw)
        said = [str(w.message) for w in caught]
        return [m for m in said if 'is not used' in m
                and ('n_frequencies=' in m or 'bandwidth_factor=' in m)]

    def test_both_default_to_none(self):
        defaults = Bellhop.__init__.__kwdefaults__
        assert defaults['n_frequencies'] is None
        assert defaults['bandwidth_factor'] is None

    @pytest.mark.parametrize('mode, kw, said', [
        (RunMode.COHERENT_TL, {}, True),
        (RunMode.BROADBAND, dict(frequencies=np.array([180.0, 200.0])), True),
        (RunMode.BROADBAND, {}, False)])
    def test_a_pin_off_its_band_is_said_unused(self, mode, kw, said):
        model = Bellhop(backend='fortran', verbose=False, n_frequencies=16)
        assert bool(self._said(model, mode, **kw)) is said

    def test_unset_knobs_say_nothing_and_expand_the_default_band(self):
        model = Bellhop(backend='fortran', verbose=False)
        assert self._said(model, RunMode.COHERENT_TL) == []
        settings = model.run_settings(_pekeris_env(), _SRC, _RCV,
                                      RunMode.BROADBAND)
        assert settings.frequencies.size == 128
        assert settings.frequencies[0] == pytest.approx(150.0)


class TestTheBandKnobsAreCheckedAtConstruction:
    """``n_frequencies`` and ``bandwidth_factor`` shape every BROADBAND run, so a
    value no run could use is refused when the model is built (RA-WAVE-6)."""

    @pytest.mark.parametrize('n_frequencies', [0, 1])
    def test_fewer_than_two_bins_are_refused(self, n_frequencies):
        with pytest.raises(ConfigurationError, match='n_frequencies'):
            Bellhop(backend='fortran', verbose=False, n_frequencies=n_frequencies)

    def test_two_bins_are_accepted(self):
        assert Bellhop(backend='fortran', verbose=False,
                       n_frequencies=2).n_frequencies == 2

    def test_a_fractional_count_is_refused_and_a_numpy_count_taken(self):
        """A fraction was truncated to the bins below it (2.7 -> 2)."""
        with pytest.raises(ConfigurationError, match='whole number'):
            Bellhop(backend='fortran', verbose=False, n_frequencies=2.7)
        count = Bellhop(backend='fortran', verbose=False,
                        n_frequencies=np.int64(5)).n_frequencies
        assert count == 5 and type(count) is int

    @pytest.mark.parametrize('factor', [0.0, -0.5, float('nan'), 2.0, 2.5])
    def test_a_width_outside_0_to_2_is_refused(self, factor):
        """From 2 up the band's lower edge fc*(1 - bw/2) reaches 0 Hz."""
        with pytest.raises(ConfigurationError, match=r'must lie in \(0, 2\)'):
            Bellhop(backend='fortran', verbose=False,
                    bandwidth_factor=factor)

    def test_a_width_just_below_2_is_taken(self):
        assert Bellhop(backend='fortran', verbose=False,
                       bandwidth_factor=1.999).bandwidth_factor == 1.999

    def test_a_reassigned_knob_is_refused_by_the_run(self):
        model = Bellhop(backend='fortran', verbose=False)
        model.n_frequencies = 1
        with pytest.raises(ConfigurationError, match='n_frequencies'):
            model.validate_inputs(_pekeris_env(), _SRC, _RCV)


class TestTheBroadbandFieldCarriesTheWaterColumnsFastestSpeed:
    """``Field.to_time_trace`` anchors its window at r / max(c_max, c0), so
    the H(f) Bellhop returns carries the fastest WATER speed: the rays
    travel in the water, and a faster seabed carries no head wave
    (RA-WAVE-11)."""

    def test_c_max_is_the_fastest_water_not_the_seabed(self):
        from uacpy.core.environment import SoundSpeedProfile
        env = _pekeris_env(ssp=SoundSpeedProfile.from_pairs(
            np.array([[0.0, 1490.0], [100.0, 1520.0]])))
        field = Bellhop(backend='fortran', verbose=False).run(
            env, _SRC, _RCV, RunMode.BROADBAND,
            frequencies=np.array([180.0, 200.0, 220.0]))
        assert field.speeds.surface == 1490.0
        assert field.speeds.water_max == 1520.0


class TestTheBroadbandArrivalsAreAnArrivalsResult:
    """BROADBAND / TIME_SERIES are built from one ARRIVALS launch at fc in
    the same call; the arrivals are kept on the result stamped as the
    ARRIVALS result they are."""

    def test_the_arrivals_carry_their_own_mode_and_settings(self):
        field = Bellhop(backend='fortran', verbose=False).run(
            _pekeris_env(), _SRC, _RCV, RunMode.BROADBAND,
            frequencies=np.array([180.0, 200.0, 220.0]))
        arrivals = field.components['arrivals']
        assert isinstance(arrivals, Arrivals)
        assert arrivals.run_mode == RunMode.ARRIVALS
        assert arrivals.run_settings.mode == RunMode.ARRIVALS
        np.testing.assert_array_equal(arrivals.run_settings.frequencies,
                                      [200.0])


class TestTheBounceRouteWarnsOfWhatIsModelled:
    """BOUNCE reads no surface, and it tabulates one seabed column against
    one seafloor water speed while the ray trace keeps the environment's
    surface, bathymetry and SSP (JRN-41): the route says so instead of
    relaying notices about losses the Bellhop field does not suffer."""

    @staticmethod
    def _messages(env):
        with recorded_warnings() as caught:
            Bellhop(backend='fortran', verbose=False).run_settings(
                env, _SRC, _RCV)
        return [str(w.message) for w in caught]

    def test_no_notice_about_the_surface(self):
        env = _layered_env(altimetry=[(0.0, 0.0), (1000.0, -2.0),
                                      (2000.0, 0.0)])
        said = self._messages(env)
        assert not any('Bounce does not support sea-surface altimetry' in m
                       for m in said), said

    def test_a_collapsed_bathymetry_is_said_to_concern_the_table(self):
        from uacpy.core.bottom import Bottom
        env = Environment(
            name='slope', bathymetry=np.array([[0.0, 100.0],
                                               [2000.0, 120.0]]),
            ssp=1500.0,
            bottom=Bottom.from_presets([('silt', 5.0)], halfspace='sand'))
        said = self._messages(env)
        table = [i for i, m in enumerate(said)
                 if 'describe that table only' in m]
        collapse = [i for i, m in enumerate(said)
                    if m.startswith('Bounce does not support range-dependent '
                                    'bathymetry')]
        assert table and collapse and table[0] < collapse[0], said
        assert 'keeps the bathymetry as given' in said[table[0]]

    def test_a_flat_environment_gets_no_such_notice(self):
        assert not any('describe that table only' in m
                       for m in self._messages(_layered_env()))


class TestTheSettingsRecordTheDeck:
    """``run_settings().engine`` is what the deck is written from: the step,
    the ray box and the beams read back from ``model.env`` equal it, and the
    record round-trips and pickles, BOUNCE's own settings included."""

    def test_the_deck_carries_the_settings(self, tmp_path):
        model = Bellhop(backend='fortran', verbose=False, work_dir=tmp_path,
                        cleanup=False, n_beams=201)
        result = model.run(_pekeris_env(), _SRC, _RCV)
        engine = result.run_settings.engine
        assert engine == model.run_settings(_pekeris_env(), _SRC,
                                            _RCV).engine
        lines = (tmp_path / 'model.env').read_text().splitlines()
        step_at = next(i for i, line in enumerate(lines)
                       if line.strip().startswith(f"{engine.ray_step:.6f}"))
        assert lines[step_at + 1].split() == [
            f"{engine.z_box:.6f}", f"{engine.r_box / 1000.0:.6f}"]
        assert f"{engine.launch_angles_deg[0]:.6f} {engine.launch_angles_deg[1]:.6f} /" in lines

    def test_the_record_round_trips_and_pickles(self):
        import pickle
        from uacpy.models import RunSettings
        from uacpy.models.bellhop import BellhopSettings
        model = Bellhop(backend='fortran', verbose=False, beam_type='C')
        engine = model.run_settings(_layered_env(), _SRC, _RCV).engine
        assert engine.bounce is not None and engine.range_trim is not None
        again = BellhopSettings.from_dict(engine.to_dict())
        assert again == engine
        # Equality compares the plain forms, so the nested record is checked
        # for what it is, not only for what it prints.
        assert isinstance(again.bounce, RunSettings)
        assert again.bounce.mode == RunMode.REFLECTION
        assert pickle.loads(pickle.dumps(engine)) == engine


class TestTheAutomaticFanUsesAngleModsReferenceSpeed:
    """angleMod's automatic beam count divides by its module constant
    ``c0 = 1500`` (``angleMod.f90:15``), whatever the water's speed, so the
    eigenray-file estimate does too (RA-WAVE-10)."""

    def test_a_slow_water_column_keeps_the_1500_divisor(self):
        env = Environment(name='deep', bathymetry=1000.0, ssp=1450.0)
        rcv = Receiver(depths=[50.0], ranges=[5000.0])
        n = eigenray_beam_count(
            env, Source(depths=30.0, frequencies=1000.0), rcv, n_beams=0)
        assert n == int(0.3 * 5000.0 * 1000.0 / 1500.0)


class TestTheBackendDocstringsStateTheRefusal:
    """An explicit backend that is not installed raises; nothing falls back
    to the Fortran binary (RA-WAVE-2)."""

    def test_the_backend_entry_names_the_error(self):
        # Each engine documents its constructor once (class or __init__).
        docs = [d for d in (Bellhop.__doc__, Bellhop.__init__.__doc__)
                if d and 'backend :' in d]
        assert docs, 'no docstring documents the backend parameter'
        for doc in docs:
            entry = doc[doc.index('backend :'):doc.index('dimensionality :')]
            assert 'ExecutableNotFoundError' in entry
            assert 'falls back' not in entry


class TestTheFortranSimpleGaussianBeamSaysItWeightsALineSourceAsAPoint:
    """The Fortran ``InfluenceSGB`` applies the point-source launch weight
    to a line source too (``influence.f90:665``); the ports do not, and the
    binary is left as built, so the run says so (RA-WAVE-9)."""

    @staticmethod
    def _said(beam_type, source_type):
        model = Bellhop(backend='fortran', verbose=False, beam_type=beam_type)
        with recorded_warnings() as caught:
            model.run_settings(
                _pekeris_env(),
                Source(depths=[40.0], frequencies=200.0,
                       source_type=source_type), _RCV)
        return [str(w.message) for w in caught
                if 'weights a line source as a point source' in str(w.message)]

    def test_a_line_source_is_told(self):
        assert len(self._said('S', 'line')) == 1

    @pytest.mark.parametrize('beam_type,source_type', [('S', 'point'),
                                                       ('G', 'line')])
    def test_the_other_pairs_are_silent(self, beam_type, source_type):
        assert self._said(beam_type, source_type) == []


@pytest.mark.requires_binary
class TestBellhop:
    """Tests for Bellhop model. Smoke TL coverage on ``simple_env`` lives
    in ``test_compute_wrappers.TestComputeAPI`` and ``test_bellhop`` —
    only model-specific scenarios live here."""

    def test_range_dependent_env_returns_full_receiver_grid(self, range_dependent_env, source, receiver_small):
        """Test Bellhop with range-dependent environment."""
        bellhop = Bellhop(verbose=False)
        result = bellhop.compute_field(
            env=range_dependent_env,
            source=source,
            receiver=receiver_small
        )

        assert isinstance(result, Field)
        assert result.shape[0] == len(receiver_small.depths)

    def test_bellhop_cuda_backend_compute_field(self, simple_env, source, receiver_small):
        """Smoke test for ``Bellhop(backend='cuda')``. Skipped when the
        bellhopcuda binary is not built, which that constructor reports as
        ``ExecutableNotFoundError``.
        """
        from uacpy.core.exceptions import ExecutableNotFoundError
        from uacpy.models import Bellhop
        try:
            bhc = Bellhop(backend='cuda', verbose=False)
        except ExecutableNotFoundError:
            pytest.skip("bellhopcuda binary not installed")
        result = bhc.compute_field(env=simple_env, source=source, receiver=receiver_small)
        assert isinstance(result, Field)
        assert result.shape == (len(receiver_small.depths), len(receiver_small.ranges))


class TestSeparableAbsorptionScalesExactlyAcrossTheBand:
    """Bellhop's ``Im tau`` is ``-integral of alpha(f_t) ds / omega_t`` along
    the ray (``Step.f90:73``, ``misc/AttenMod.f90:113``), so for Thorp the
    absorption at any f is ``omega_t Im tau * alpha(f)/alpha(f_t)`` exactly.
    Measured arrival by arrival against Bellhop traced at f itself:
    0.0011 dB, where the linear-in-f scaling missed by 2.7 dB (20-60 kHz,
    1 km). One Francois-Garrison row takes the same scaling by its surface
    ratio, which its pressure terms bend with depth; the band check measures
    that bend."""

    @staticmethod
    def _env(absorption, depth=100.0, bottom_speed=1700.0):
        return Environment(
            name='pek', bathymetry=depth, ssp=[(0.0, 1520.0), (depth, 1500.0)],
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=bottom_speed, density=1.8,
                                      attenuation=0.5),
            absorption=absorption)

    @staticmethod
    def _arrivals(env, f):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return Bellhop(verbose=False, backend='fortran', n_beams=2001).run(
                env, Source(depths=50.0, frequencies=f),
                Receiver(depths=[30.0], ranges=[1000.0]),
                run_mode=RunMode.ARRIVALS)

    def test_every_multipath_arrival_matches_its_own_trace(self):
        from uacpy.core.absorption import Thorp
        env = self._env(Thorp())
        traced = self._arrivals(env, 40000.0)
        assert type(traced.absorption).__name__ == 'Thorp'
        worst, matched = 0.0, 0
        for f in (20000.0, 60000.0):
            own = self._arrivals(env, f)
            scaled = traced._received_amplitudes_at(f)
            geometric = np.abs(traced.amplitudes)
            for a in own.arrivals:
                same = [i for i, b in enumerate(traced.arrivals)
                        if (b['n_top_bounces'], b['n_bot_bounces'])
                        == (a['n_top_bounces'], a['n_bot_bounces'])]
                if not same:
                    continue
                i = min(same, key=lambda i: abs(traced.arrivals[i]['delay']
                                                - a['delay']))
                if abs(traced.arrivals[i]['delay'] - a['delay']) > 1e-4:
                    continue
                loss_scaled = 20 * np.log10(np.abs(scaled[i]) / geometric[i])
                loss_own = 20 * np.log10(np.exp(2 * np.pi * f * a['delay_imag']))
                worst = max(worst, abs(loss_scaled - loss_own))
                matched += 1
        assert matched >= 20
        assert worst <= 0.01, worst

    def test_the_time_series_spectrum_follows_thorp(self):
        from uacpy.core.absorption import Thorp
        from uacpy.core.acoustics.attenuation import absorption_thorp
        env = Environment(
            name='deep', bathymetry=5000.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1600.0, density=1.8,
                                      attenuation=0.5))
        fs = 480000.0
        t = np.arange(int(0.002 * fs)) / fs
        pulse = np.hanning(t.size) * np.sin(
            2 * np.pi * (20000.0 * t + 0.5 * (40000.0 / t[-1]) * t ** 2))
        traces = []
        for absorption in (Thorp(), None):
            env.absorption = absorption
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                traces.append(np.asarray(Bellhop(
                    verbose=False, backend='fortran', launch_angles_deg=(-10, 10),
                    n_beams=401).run(
                        env, Source(depths=2500.0, frequencies=40000.0),
                        Receiver(depths=[2500.0], ranges=[1000.0]),
                        run_mode=RunMode.TIME_SERIES, source_waveform=pulse,
                        sample_rate=fs, output_duration=0.02).data,
                    dtype=float).ravel())
        f = np.fft.rfftfreq(traces[0].size, 1.0 / fs)
        lossy, lossless = (np.abs(np.fft.rfft(x)) for x in traces)
        for fk in (25000.0, 55000.0):
            k = int(np.argmin(np.abs(f - fk)))
            applied = 20 * np.log10(lossless[k] / lossy[k])
            assert applied == pytest.approx(
                float(absorption_thorp(f[k])), abs=0.05)

    @pytest.mark.parametrize('a0, warns', [(0.0010, True), (0.0006, False)])
    def test_a_subsurface_biological_layer_warns_above_the_threshold(
            self, a0, warns):
        """The linear scaling a Biological layer takes is checked over the
        band and the water column: a layer at 30-70 m, invisible at the
        surface, warns once its error passes 0.05 dB/km."""
        from uacpy.core.absorption import (
            _BAND_ABSORPTION_CHECK_DEPTHS, _BAND_ABSORPTION_WARN_DB_PER_KM,
            Biological, _band_absorption_error_dB_per_km)
        bio = Biological(layers=[(30.0, 70.0, 3000.0, 8.0, a0)])
        env = self._env(bio)
        band = np.linspace(1000.0, 4000.0, 7)
        err = _band_absorption_error_dB_per_km(
            bio, band, 2500.0,
            depths=np.linspace(0.0, 100.0, _BAND_ABSORPTION_CHECK_DEPTHS))
        assert (err >= _BAND_ABSORPTION_WARN_DB_PER_KM) is warns
        with recorded_warnings() as caught:
            warn_if_attenuation_extrapolates(
                env, band, 2500.0, model_name='Bellhop')
        fired = any('Biological absorption that varies with depth'
                    in str(w.message) for w in caught)
        assert fired is warns

    @pytest.mark.parametrize('law, depth, warns', [
        ('thorp', 5000.0, False), ('fg', 5000.0, True), ('fg', 100.0, False)])
    def test_the_surface_ratio_is_checked_over_the_column(
            self, law, depth, warns):
        """Over 5-15 kHz Thorp's surface ratio is its ratio at every depth
        (silent); one Francois-Garrison row's misses a 5 km column by
        0.058 dB/km (warns) and a 100 m one by 0.003 (silent)."""
        from uacpy.core.absorption import FrancoisGarrison, Thorp
        env = self._env(Thorp() if law == 'thorp' else FrancoisGarrison(),
                        depth=depth)
        with recorded_warnings() as caught:
            warn_if_attenuation_extrapolates(
                env, np.linspace(5000.0, 15000.0, 11), 10000.0,
                model_name='Bellhop')
        fired = [str(w.message) for w in caught
                 if "ratio alpha(f)/alpha(fc) at the surface"
                 in str(w.message)]
        assert bool(fired) is warns
        if warns:
            assert 'up to 0.0583 dB/km' in fired[0]

    def test_a_wide_broadband_run_warns_for_a_francois_garrison_profile(self):
        """A T/S profile does not scale from one trace frequency by one
        ratio (its frequency dependence follows the local water), so a
        5-15 kHz BROADBAND run applies it linearly and says so."""
        from uacpy.core.exceptions import NumericsWarning
        from uacpy.tests.conftest import two_layer_absorption
        env = Environment(name='band', bathymetry=100.0, ssp=1500.0,
                          bottom='sand', absorption=two_layer_absorption())
        with recorded_warnings() as rec:
            Bellhop(verbose=False).run(
                env, Source(depths=30.0, frequencies=10000.0),
                Receiver(depths=[20.0], ranges=[1000.0]),
                run_mode=RunMode.BROADBAND,
                frequencies=np.linspace(5000.0, 15000.0, 11))
        hits = [str(w.message) for w in rec
                if issubclass(w.category, NumericsWarning)
                and 'does not scale from one frequency' in str(w.message)]
        assert hits
        assert 'FrancoisGarrison' in hits[0]


class TestTheEigenrayFileInMemoryIsWeighedAsABound:
    """EIGENRAYS in a work directory held in memory weighs the .ray file's
    upper bound (pairs x RMax/step points x 53 B) with the memory budget and
    never refuses it; on disk only the pair count speaks."""

    @staticmethod
    def _notice(monkeypatch, free, memory_backed):
        from uacpy.core import BoundaryProperties, Environment, Receiver, Source
        from uacpy.models import _budget
        from uacpy.models.bellhop._plan import eigenray_size_notice
        monkeypatch.setattr(_budget, 'available_memory_bytes', lambda: free)
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          bottom=BoundaryProperties(
                              acoustic_type='half-space', sound_speed=1700.0,
                              density=1.8, attenuation=0.5))
        return eigenray_size_notice(
            env, Source(depths=[30.0], frequencies=[1000.0]),
            Receiver(depths=[50.0], ranges=[3000.0]), n_beams=0,
            grid_type='R', ray_step=2.0, memory_backed=memory_backed)

    # 942 beams x 1 receiver x 3000/2 points x 53 B
    BOUND = 942 * 1500 * 53

    def test_half_the_free_memory_is_silent(self, monkeypatch):
        assert self._notice(monkeypatch, 2 * self.BOUND, True) is None

    def test_over_half_the_free_memory_is_announced(self, monkeypatch):
        message = self._notice(monkeypatch, 2 * self.BOUND - 1, True)
        assert 'held in memory' in message and '942 ray-receiver' in message
        assert 'an upper bound' in message and 'work_dir=' in message

    def test_over_the_free_memory_is_announced_not_refused(
            self, monkeypatch):
        message = self._notice(monkeypatch, self.BOUND - 1, True)
        assert 'not refused' in message

    def test_on_disk_the_free_memory_is_not_weighed(self, monkeypatch):
        assert self._notice(monkeypatch, 1, False) is None
