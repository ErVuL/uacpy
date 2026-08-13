"""BOUNCE reflection-coefficient deck and dispatch tests.

Every numeric expectation here is anchored either on the vendored Fortran
(``Kraken/bounce.f90``, ``misc/ReadEnvironmentMod.f90``, ``misc/RefCoef.f90``)
or on closed-form plane-wave theory, never on another uacpy code path.
"""

import time

import numpy as np
import pytest

from uacpy.core import (
    Environment, Source, Receiver, BoundaryProperties,
)
from uacpy.core.bottom import Bottom, SeabedColumn, SedimentLayer
from uacpy.core.exceptions import ConfigurationError, UnsupportedFeatureError
from uacpy.models import Bounce

pytestmark = pytest.mark.requires_binary


def _src(freq=500.0):
    return Source(depths=[50.0], frequencies=[float(freq)])


def _rcv():
    return Receiver(depths=[50.0], ranges=[10000.0])


def _halfspace_env(**kw):
    props = dict(acoustic_type='half-space', sound_speed=1600.0,
                 shear_speed=400.0, density=1.8, attenuation=0.2,
                 shear_attenuation=0.5)
    props.update(kw)
    return Environment(name='bnc', bathymetry=100.0,
                       ssp=[(0.0, 1500.0), (100.0, 1500.0)],
                       bottom=BoundaryProperties(**props))


def _deck(work_dir):
    return (work_dir / 'bounce_run.env').read_text().splitlines()


class TestBareHalfspaceReferencePlane:
    """``doc/bounce.htm``: "If you only have a halfspace, you can set NMedia to
    0." With NMedia = 0 ``bounce.f90:76`` gives NPTS = 0, the medium loop does
    not run, ``FirstAcoustic`` stays 0 so ``AcousticLayers`` returns at :258,
    and the ``f``/``g`` that :201 turns into ``RCmplx`` are the ones
    ``BCImpedance('BOT')`` formed at the seafloor. Any padding medium moves that
    reference plane and rotates the phase of every ``.brc``/``.irc`` row by
    ``-2 k dz sin(theta)``."""

    def test_deck_declares_no_medium(self, tmp_path):
        Bounce(work_dir=tmp_path, cleanup=False).run(
            _halfspace_env(), _src(), _rcv())
        lines = _deck(tmp_path)
        assert lines[2].strip() == '0', (
            f"NMedia must be 0 for a bare half-space seabed; deck reads\n"
            + "\n".join(lines))
        # Nothing between the top half-space row and the BotOpt line.
        assert lines[5].strip().startswith("'A'"), (
            f"a medium block was written after the top half-space row:\n"
            + "\n".join(lines))

    def test_normal_incidence_matches_the_plane_wave_impedance_ratio(
            self, tmp_path):
        """At 90 deg grazing the shear branch decouples, so R is the
        closed-form P-wave impedance ratio with zero phase. A padding medium of
        thickness ``dz`` would show up as ``-2 k dz`` of phase here.

        The match is close but not exact: ``misc/AttenMod.f90:73,113-114`` turns
        ``alpha`` dB/wavelength into ``Im(c) = c * alpha / (2 pi * 8.6858896)``,
        so the seabed impedance the code forms is complex. At the 0.2 dB/lambda
        used here that displaces ``|R|`` by 8e-6 and rotates the phase by
        0.30 deg. Both tolerances sit an order of magnitude above that — loose
        enough not to track the loss model, tight enough that a shifted
        reference plane (tens of degrees of phase) still fails.
        """
        env = _halfspace_env()
        res = Bounce(work_dir=tmp_path, cleanup=False).run(env, _src(), _rcv())
        hs = env.bottom.halfspace_at(range=0.0)
        z1 = 1.0 * 1500.0
        z2 = hs.density * hs.sound_speed
        expected = (z2 - z1) / (z2 + z1)

        i = int(np.argmax(res.theta))
        assert res.theta[i] == pytest.approx(90.0, abs=1e-6)
        assert res.R[i] == pytest.approx(expected, abs=2e-4), (
            f"|R| at normal incidence is {res.R[i]}, not the impedance ratio "
            f"{expected}")
        phase_deg = np.degrees(res.phi[i])
        assert abs(phase_deg) < 2.0, (
            f"phase at normal incidence is {phase_deg:.3f} deg — the "
            f"reflection coefficient is referenced below the seafloor")


class TestElasticLayerMeshing:
    """``misc/ReadEnvironmentMod.f90:101-112``: ``c = alphaR``, then
    ``IF ( betaR > 0.0 ) c = betaR``, ``deltaz = c / freq0 / 20``,
    ``Nneeded = INT( thickness / deltaz )``, and *Mesh is too coarse* whenever
    the deck asks for ``NG < Nneeded / 2``. The meshing speed is the medium's
    shear speed, which for ordinary sand is ~8x slower than its compressional
    speed."""

    @staticmethod
    def _sand_env(shear=200.0):
        layer = SedimentLayer(thickness=10.0, sound_speed=1700.0, density=1.9,
                              attenuation=0.8, shear_speed=shear,
                              shear_attenuation=1.0)
        return Environment(
            name='sand', bathymetry=100.0,
            ssp=[(0.0, 1500.0), (100.0, 1500.0)],
            bottom=Bottom(columns=[SeabedColumn(
                layers=[layer],
                halfspace=BoundaryProperties(
                    'half-space', sound_speed=1800.0, density=2.0,
                    attenuation=0.1))]))

    @pytest.mark.parametrize('freq', [250.0, 500.0])
    def test_elastic_sediment_clears_the_at_mesh_floor(self, tmp_path, freq):
        env = self._sand_env()
        res = Bounce(c_low=1400.0, work_dir=tmp_path, cleanup=False).run(
            env, _src(freq), _rcv())
        assert len(res.theta) > 0

        layer = env.bottom.at(range=0.0).layers[0]
        needed = int(layer.thickness / (layer.shear_speed / freq / 20))
        # The medium mesh line is ``NG sigma Depth(Medium+1)``
        # (misc/ReadEnvironmentMod.f90:88) — the only three-token record.
        mesh_lines = [ln.split() for ln in _deck(tmp_path)
                      if len(ln.split()) == 3]
        n_g = int(mesh_lines[0][0])
        assert n_g >= needed // 2, (
            f"deck wrote NG={n_g}; ReadEnvironmentMod.f90:110 needs "
            f"{needed // 2} for a {layer.thickness} m / "
            f"{layer.shear_speed} m/s medium at {freq} Hz")

    def test_layer_thickness_is_written_exactly(self, tmp_path):
        """A BOUNCE deck reads no source or receiver records, so the 0.1 m
        interface grid that keeps them inside the mesh has nothing to protect
        here; ``misc/ReadEnvironmentMod.f90:88`` reads the depth column
        list-directed."""
        layer = SedimentLayer(thickness=2.37, sound_speed=1650.0, density=1.7,
                              attenuation=0.3)
        env = Environment(
            name='thin', bathymetry=100.0,
            ssp=[(0.0, 1500.0), (100.0, 1500.0)],
            bottom=Bottom(columns=[SeabedColumn(
                layers=[layer],
                halfspace=BoundaryProperties('half-space', sound_speed=1800.0,
                                             density=2.0, attenuation=0.1))]))
        Bounce(work_dir=tmp_path, cleanup=False).run(env, _src(), _rcv())
        depths = [float(ln.split()[0]) for ln in _deck(tmp_path)
                  if ln.strip().endswith('/') and len(ln.split()) == 7]
        assert 102.37 in depths, (
            f"the 2.37 m layer was quantised off the deck: {depths}")


class TestReflectionTableInput:
    """``misc/RefCoef.f90:39`` opens ``<root>.brc`` with ``STATUS='OLD'`` before
    ``ComputeReflectionCoefficient`` rewrites it at ``bounce.f90:230``, so a
    staged table is an *input* for the same launch that overwrites it."""

    @staticmethod
    def _basement_env(table, acoustic_type):
        layer = SedimentLayer(thickness=5.0, sound_speed=1650.0, density=1.7,
                              attenuation=0.3)
        return Environment(
            name='chain', bathymetry=100.0,
            ssp=[(0.0, 1500.0), (100.0, 1500.0)],
            bottom=Bottom(columns=[SeabedColumn(
                layers=[layer],
                halfspace=BoundaryProperties(
                    acoustic_type, reflection_file=str(table)))]))

    def test_a_staged_brc_survives_the_stale_output_sweep(self, tmp_path):
        ref = Bounce(work_dir=tmp_path / 'ref', cleanup=False).run(
            _halfspace_env(shear_speed=0.0), _src(200.0), _rcv())
        env = self._basement_env(ref.metadata['brc_file'], 'file')
        res = Bounce(work_dir=tmp_path / 'chain', cleanup=False).run(
            env, _src(200.0), _rcv())
        assert len(res.theta) > 0
        assert np.all(np.isfinite(res.R))

    def test_an_irc_seabed_is_refused(self, tmp_path):
        """``misc/RefCoef.f90:103-104`` leaves xTab/fTab/gTab/iTab allocated for
        the table it read, so ``bounce.f90:52`` cannot allocate them for the
        table it must write."""
        ref = Bounce(work_dir=tmp_path / 'ref2', cleanup=False).run(
            _halfspace_env(shear_speed=0.0), _src(200.0), _rcv())
        env = self._basement_env(ref.metadata['irc_file'], 'precalc')
        with pytest.raises(UnsupportedFeatureError, match='precalc'):
            Bounce(work_dir=tmp_path / 'chain2', cleanup=False).run(
                env, _src(200.0), _rcv())


class TestAngularCoverage:
    """``doc/bounce.htm``: "For a full 90 degree calculation set CMin to the
    lowest speed in the problem (say 1400.0) CMax to 1.0E9." Above the last
    tabulated angle every consumer silently returns R = 0, phi = 0
    (``misc/RefCoef.f90:144-149``, both warning WRITEs commented out)."""

    def test_default_c_high_reaches_grazing_90(self, tmp_path):
        res = Bounce(work_dir=tmp_path, cleanup=False).run(
            _halfspace_env(), _src(200.0), _rcv())
        assert res.theta.max() == pytest.approx(90.0, abs=1e-6), (
            f"table stops at {res.theta.max()} deg")

    def test_a_finite_c_high_stops_at_acos_c0_over_c_high(self, tmp_path):
        res = Bounce(c_high=10000.0, work_dir=tmp_path, cleanup=False).run(
            _halfspace_env(), _src(200.0), _rcv())
        expected = np.degrees(np.arccos(1500.0 / 10000.0))
        assert res.theta.max() == pytest.approx(expected, abs=1e-3)


class TestSamplingGuards:
    """``bounce.f90:49`` NkTab = INT( 1000 * RMax_km * ( kMax - kMin ) / 2 pi )
    and :172 Deltak = ( kMax - kMin ) / ( NkTab - 1 )."""

    def test_single_angle_is_refused_instead_of_hanging(self):
        env = _halfspace_env(shear_speed=0.0)
        t0 = time.monotonic()
        with pytest.raises(ConfigurationError, match='n_angles'):
            Bounce(c_low=1400.0, n_angles=1, timeout=30).run(
                env, _src(50.0), _rcv())
        assert time.monotonic() - t0 < 10.0, "the guard ran the binary"

    def test_rmax_below_one_tabulated_angle_is_refused(self):
        env = _halfspace_env(shear_speed=0.0)
        with pytest.raises(ConfigurationError, match='tabulated angle'):
            Bounce(c_low=1400.0, rmax=1.0, timeout=30).run(
                env, _src(50.0), _rcv())

    @pytest.mark.parametrize('rmax', [0.0, -5.0])
    def test_non_positive_rmax_is_refused(self, rmax):
        with pytest.raises(ConfigurationError, match='rmax > 0'):
            Bounce(rmax=rmax)

    def test_n_angles_is_honoured_at_high_frequency(self, tmp_path):
        """The requested count only survives if RMax reaches the deck at
        better than 10 m resolution — at 5 kHz, n_angles=50 needs
        RMax = 0.0134 km."""
        env = _halfspace_env(shear_speed=0.0)
        res = Bounce(c_low=1400.0, n_angles=50, work_dir=tmp_path,
                     cleanup=False).run(env, _src(5000.0), _rcv())
        # BOUNCE echoes the count it derived: bounce.f90:50.
        prt = (tmp_path / 'bounce_run.prt').read_text()
        n_ktab = int(prt.split('NkTab =')[1].split()[0])
        assert n_ktab == 50, (
            f"asked for 50 angles, deck produced {n_ktab}")
        assert len(res.theta) > 0


@pytest.mark.requires_binary
class TestReflectionPhaseIsUnwrapped:
    """``misc/RefCoef.f90:119`` states the table's contract — "Assumes phi has
    been unwrapped so that it varies smoothly" — and
    ``InterpolateReflectionCoefficient`` interpolates phi linearly between the
    bracketing abscissas. BOUNCE takes the principal value at
    ``Kraken/bounce.f90:203`` and attempts an unwrap at ``:213-222``, but its
    incrementing branch (``:219``) tests ``AIMAG(R1) > 0 .AND. AIMAG(R1) < 0``
    — a contradiction that can never fire — so the raw table steps by nearly a
    full turn and interpolating across such a step sweeps the phase the long
    way round. uacpy re-unwraps in ``io/refl_io.py``; the 180 deg bound below
    is the largest adjacent step a correctly unwrapped table can show."""

    @staticmethod
    def _env():
        from uacpy.core.bottom import SeabedColumn, SedimentLayer
        return Environment(
            name='layered', bathymetry=100.0, ssp=1500.0,
            bottom=SeabedColumn(
                layers=[SedimentLayer(thickness=10.0, sound_speed=1600.0,
                                      density=1.8, attenuation=0.2)],
                halfspace=BoundaryProperties(
                    'half-space', sound_speed=1800.0, density=2.0,
                    attenuation=0.5)))

    def test_the_written_table_has_no_principal_value_wraps(self, tmp_path):
        Bounce(work_dir=tmp_path, cleanup=False, verbose=False).run(
            self._env(), Source(depths=50.0, frequencies=500.0),
            Receiver(depths=[50.0], ranges=[1000.0]))
        table = np.loadtxt(sorted(tmp_path.glob('*.brc'))[0], skiprows=1)
        jumps = np.abs(np.diff(table[:, 2]))
        assert not np.any(jumps > 180.0), (
            f"phase still wraps: max adjacent jump {jumps.max():.1f} deg")
        # The dedup contract and the magnitudes must survive the rewrite.
        assert np.all(np.diff(table[:, 0]) > 0)
        assert table[:, 1].min() >= 0.0 and table[:, 1].max() <= 1.0 + 1e-6


class TestStagedTableGetsTheSameTreatment:
    """A table handed straight to ``reflection_file=`` reaches the engine
    through :func:`uacpy.io.refl_io.stage_reflection_file` and never through
    ``Bounce.run``, so the unwrap has to happen at the staging boundary — the
    one point every angle table crosses on its way to a run. Whatever produced
    the file, what ``InterpolateReflectionCoefficient`` reads must satisfy
    ``misc/RefCoef.f90:119``.

    The synthetic table below is what BOUNCE's broken unwrap leaves behind: a
    principal value that steps ~300 deg between adjacent angles, plus the
    duplicate 0-degree rows of the evanescent block (``bounce.f90:204-209``)
    that ``RefCoef.f90:165`` would divide by.
    """

    # Principal values of a phase falling smoothly past -180 deg.
    _WRAPPED = [(0.0, 1.0, 180.0), (0.0, 1.0, 180.0), (10.0, 0.9, 150.0),
                (20.0, 0.8, -170.0), (30.0, 0.7, -130.0), (40.0, 0.6, -90.0)]
    _UNWRAPPED = [180.0, 150.0, 190.0, 230.0, 270.0]

    @staticmethod
    def _write(path, rows):
        path.write_text(f"{len(rows):12d}\n" + ''.join(
            f"   {a}        {r}        {p}\n" for a, r, p in rows))
        return path

    def _env(self, table, boundary='bottom'):
        from uacpy.core.surface import Surface
        props = BoundaryProperties('file', reflection_file=str(table))
        if boundary == 'top':
            return Environment(name='stage', bathymetry=100.0, ssp=1500.0,
                               surface=Surface(properties=[props]))
        return Environment(name='stage', bathymetry=100.0, ssp=1500.0,
                           bottom=props)

    def _stage_via_writer(self, tmp_path, table, boundary='bottom'):
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        env_path = tmp_path / 'staged.env'
        write_bellhop_env_file(env_path, self._env(table, boundary),
                               _src(200.0), _rcv())
        suffix = '.trc' if boundary == 'top' else '.brc'
        return np.loadtxt(env_path.with_suffix(suffix), skiprows=1)

    @pytest.mark.parametrize('boundary', ['bottom', 'top'])
    def test_a_user_supplied_table_is_unwrapped_when_staged(self, tmp_path,
                                                            boundary):
        suffix = '.trc' if boundary == 'top' else '.brc'
        src = self._write(tmp_path / f'user{suffix}', self._WRAPPED)
        staged = self._stage_via_writer(tmp_path, src, boundary)
        assert staged[:, 2] == pytest.approx(self._UNWRAPPED)
        assert np.all(np.abs(np.diff(staged[:, 2])) <= 180.0)
        assert np.all(np.diff(staged[:, 0]) > 0)       # duplicate row collapsed
        assert staged[:, 1] == pytest.approx([1.0, 0.9, 0.8, 0.7, 0.6])

    def test_the_users_own_file_is_left_alone(self, tmp_path):
        """Staging copies before normalising, so the path the user passed keeps
        the bytes they wrote."""
        src = self._write(tmp_path / 'user.brc', self._WRAPPED)
        before = src.read_text()
        self._stage_via_writer(tmp_path, src)
        assert src.read_text() == before

    def test_an_already_smooth_table_is_unchanged(self, tmp_path):
        rows = [(0.0, 1.0, 180.0), (10.0, 0.9, 150.0), (20.0, 0.8, 190.0)]
        staged = self._stage_via_writer(
            tmp_path, self._write(tmp_path / 'smooth.brc', rows))
        assert staged == pytest.approx(np.array(rows))

    def test_an_irc_is_staged_untouched(self, tmp_path):
        """``boundary='internal'`` carries BOUNCE's six fixed-format columns,
        not the 3-column angle table — normalising it would destroy it."""
        from uacpy.io.refl_io import stage_reflection_file
        src = tmp_path / 'user.irc'
        src.write_text(" ' BOUNCE '  100.0\n 2\n"
                       "     0.1000000     1.0000000     0.0000000"
                       "     0.5000000     0.1000000    0\n"
                       "     0.2000000     0.9000000     0.1000000"
                       "     0.4000000     0.2000000    1\n")
        dest = stage_reflection_file(src, tmp_path / 'deck.env',
                                     boundary='internal')
        assert dest.read_text() == src.read_text()


class TestCLowCannotExceedTheWaterSpeed:
    """``bounce.f90:46`` sets ``kMax = omega/cLow`` but ``:195`` references the
    angles to ``k0 = omega/c0`` with ``c0 = HSTop%cP``, the water sound speed at
    the seafloor. ``:198-210`` only computes ``theta`` ``WHERE( k0 > kx )``, so
    ``cLow > c0`` makes the table start above 0 deg grazing, and
    ``misc/RefCoef.f90:137-141`` then hands every consumer ``R = 0`` below the
    first tabulated angle (its warning goes to the ``.prt`` only). Measured cost
    at 1.3 % above the water speed: mean 5.1 dB, max 25 dB."""

    @staticmethod
    def _fixture(c_water):
        from uacpy.core.environment import SoundSpeedProfile
        env = Environment(
            name='clow', bathymetry=100.0,
            ssp=SoundSpeedProfile.from_pairs(
                np.array([[0.0, c_water], [100.0, c_water]])),
            bottom=BoundaryProperties(sound_speed=1600.0, density=1.8,
                                      attenuation=0.5))
        return (env, Source(depths=25.0, frequencies=200.0),
                Receiver(depths=50.0, ranges=np.linspace(500.0, 5000.0, 20)))

    @pytest.mark.parametrize('c_low', [1520.0, 1560.0, 2000.0])
    def test_above_the_water_speed_is_refused(self, c_low):
        env, src, rcv = self._fixture(1500.0)
        with pytest.raises(ConfigurationError, match='grazing'):
            Bounce(verbose=False, c_low=c_low, rmax=5000.0).run(env, src, rcv)

    @pytest.mark.parametrize('c_low', [1400.0, 1450.0, 1500.0])
    def test_at_or_below_the_water_speed_still_tabulates_from_zero(self, c_low):
        env, src, rcv = self._fixture(1500.0)
        result = Bounce(verbose=False, c_low=c_low, rmax=5000.0).run(
            env, src, rcv)
        theta = np.asarray(result.theta)
        assert theta.min() == pytest.approx(0.0, abs=1e-6), (
            f"c_low={c_low} against 1500 m/s water left the grazing wedge out: "
            f"table starts at {theta.min():.3f} deg")
        assert theta.max() == pytest.approx(90.0, abs=1e-6)

    def test_the_default_is_refused_in_cold_fresh_water(self):
        """``DEFAULT_C_MIN`` is 1400 m/s, which cold fresh water undercuts — so
        the failure was reachable without the user setting anything."""
        env, src, rcv = self._fixture(1380.0)
        with pytest.raises(ConfigurationError, match='grazing'):
            Bounce(verbose=False, rmax=5000.0).run(env, src, rcv)


class TestRunWithBounceDerivesCLow:
    """``Bellhop.run_with_bounce`` and the ``_maybe_route_through_bounce`` auto
    route are uacpy's own choice, so they must not hand BOUNCE a ``c_low`` the
    environment rejects."""

    @staticmethod
    def _fixture(c_water):
        from uacpy.core.environment import SoundSpeedProfile
        env = Environment(
            name='derive', bathymetry=100.0,
            ssp=SoundSpeedProfile.from_pairs(
                np.array([[0.0, c_water], [100.0, c_water]])),
            bottom=BoundaryProperties(sound_speed=1600.0, density=1.8,
                                      attenuation=0.5))
        return (env, Source(depths=25.0, frequencies=200.0),
                Receiver(depths=[30.0, 50.0, 70.0],
                         ranges=np.linspace(500.0, 5000.0, 12)))

    @pytest.mark.parametrize('c_water', [1500.0, 1380.0])
    def test_matches_the_direct_halfspace_in_cold_and_ordinary_water(self,
                                                                    c_water):
        from uacpy.models import Bellhop
        from uacpy.models.base import RunMode
        env, src, rcv = self._fixture(c_water)
        model = Bellhop(verbose=False, beam_type='G', n_beams=2001,
                        alpha=(-80.0, 80.0), backend='fortran')
        reference = np.asarray(model.run(env, src, rcv,
                                         RunMode.COHERENT_TL).tl)
        routed = np.asarray(model.run_with_bounce(
            env, src, rcv, run_mode=RunMode.COHERENT_TL).tl)
        delta = np.abs(routed - reference)
        assert np.nanmax(delta) < 0.5, (
            f"water {c_water} m/s: BOUNCE round trip differs from the direct "
            f"half-space by up to {np.nanmax(delta):.2f} dB")

    def test_an_explicit_c_low_is_still_honoured(self):
        env, src, rcv = self._fixture(1500.0)
        from uacpy.models import Bellhop
        from uacpy.models.base import RunMode
        with pytest.raises(ConfigurationError, match='grazing'):
            Bellhop(verbose=False, backend='fortran').run_with_bounce(
                env, src, rcv, run_mode=RunMode.COHERENT_TL, c_low=1560.0)
