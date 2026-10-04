"""BOUNCE reflection-coefficient deck and dispatch tests.

Every numeric expectation here is anchored either on the vendored Fortran
(``Kraken/bounce.f90``, ``misc/ReadEnvironmentMod.f90``, ``misc/RefCoef.f90``)
or on closed-form plane-wave theory, never on another uacpy code path.
"""

import inspect
import time
import warnings

import numpy as np
import pytest

import uacpy
from uacpy.core import (
    Environment, Source, Receiver, BoundaryProperties,
)
from uacpy.core.boundary import SedimentLayer
from uacpy.core.bottom import Bottom, SeabedColumn
from uacpy.core.constants import DEFAULT_WATER_DENSITY_G_CM3
from uacpy.models._window import DEFAULT_C_MAX_UNBOUNDED
from uacpy.models._defaults import DEFAULT_C_MIN
from uacpy.core.environment import SoundSpeedProfile
from uacpy.core.exceptions import (
    ConfigurationError, ModelExecutionError, UnsupportedFeatureError,
)
from uacpy.models import Bounce
from uacpy.models.bounce._plan import (
    resolve_c_low, resolve_n_mesh, tabulated_angle_count,
)
from uacpy.core.results import ReflectionCoefficient
from uacpy.tests.conftest import recorded_warnings

pytestmark = pytest.mark.requires_binary


def _env(ssp_pairs, bathymetry=100.0):
    return Environment(
        name='r26', bathymetry=bathymetry,
        ssp=SoundSpeedProfile.from_pairs(np.asarray(ssp_pairs, dtype=float)),
        bottom=BoundaryProperties(sound_speed=1600.0, density=1.8,
                                  attenuation=0.5))


def _isothermal(c, bathymetry=100.0):
    return _env([[0.0, c], [bathymetry, c]], bathymetry)


_SRC = Source(depths=25.0, frequencies=200.0)
_RCV = Receiver(depths=50.0, ranges=np.linspace(500.0, 5000.0, 20))


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
        z1 = env.water_density * 1500.0
        z2 = hs.density * hs.sound_speed
        expected = (z2 - z1) / (z2 + z1)

        i = int(np.argmax(res.angles))
        assert res.angles[i] == pytest.approx(90.0, abs=1e-6)
        assert res.magnitude[i] == pytest.approx(expected, abs=2e-4), (
            f"|R| at normal incidence is {res.magnitude[i]}, not the impedance ratio "
            f"{expected}")
        phase_deg = np.degrees(res.phase[i])
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
        assert len(res.angles) > 0

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
        assert len(res.angles) > 0
        assert np.all(np.isfinite(res.magnitude))

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
        assert res.angles.max() == pytest.approx(90.0, abs=1e-6), (
            f"table stops at {res.angles.max()} deg")

    def test_a_finite_c_high_stops_at_acos_c0_over_c_high(self, tmp_path):
        res = Bounce(c_high=10000.0, work_dir=tmp_path, cleanup=False).run(
            _halfspace_env(), _src(200.0), _rcv())
        expected = np.degrees(np.arccos(1500.0 / 10000.0))
        assert res.angles.max() == pytest.approx(expected, abs=1e-3)


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
            Bounce(c_low=1400.0, rmax_m=1.0, timeout=30).run(
                env, _src(50.0), _rcv())

    def test_the_unitless_rmax_keyword_is_not_accepted(self):
        with pytest.raises(TypeError, match='rmax'):
            Bounce(rmax=1000.0)

    @pytest.mark.parametrize('rmax_m', [0.0, -5.0])
    def test_non_positive_rmax_is_refused(self, rmax_m):
        with pytest.raises(ConfigurationError, match='rmax_m > 0'):
            Bounce(rmax_m=rmax_m)

    def test_n_angles_is_honoured_at_high_frequency(self, tmp_path):
        """The requested count only survives if RMax reaches the deck at
        better than 10 m resolution — at 5 kHz, n_angles=50 needs
        RMax = 0.0134 km."""
        env = _halfspace_env(shear_speed=0.0)
        res = Bounce(c_low=1400.0, n_angles=50, work_dir=tmp_path,
                     cleanup=False).run(env, _src(5000.0), _rcv())
        # BOUNCE echoes the count it derived: bounce.f90:50.
        prt = (tmp_path / 'bounce_run.prt').read_text()
        n_angles = int(prt.split('NkTab =')[1].split()[0])
        assert n_angles == 50, (
            f"asked for 50 angles, deck produced {n_angles}")
        assert len(res.angles) > 0


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
        from uacpy.core.boundary import SedimentLayer
        from uacpy.core.bottom import SeabedColumn
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
                               surface=Surface(nodes=[props]))
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


def _lossless_sand_env():
    """The doc's reference case (bounce.md §4): 1650 m/s / 1.9 g/cm³ lossless
    sand under 100 m of 1500 m/s water."""
    return _halfspace_env(sound_speed=1650.0, density=1.9, attenuation=0.0,
                          shear_speed=0.0, shear_attenuation=0.0)


# Critical grazing angle of the reference case: arccos(c1/c2).
_SAND_THETA_C = float(np.degrees(np.arccos(1500.0 / 1650.0)))


def _interp_R(rc, theta_deg):
    theta = np.asarray(rc.angles, dtype=float)
    return float(np.interp(theta_deg, theta, np.asarray(rc.magnitude, dtype=float)))


def _steepest_descent_deg(rc):
    """Angle of the steepest |R| descent, on a uniform 0.25 deg resample of
    the run's own grid (10-40 deg brackets the reference case's critical
    angle)."""
    grid = np.arange(10.0, 40.0, 0.25)
    r = np.interp(grid, np.asarray(rc.angles, float), np.asarray(rc.magnitude, float))
    return float(grid[np.argmin(np.gradient(r, grid))])


class TestLosslessSandRayleighAnalytics:
    """COA §2.4's fluid-fluid Rayleigh reflection, on the doc's lossless sand
    half-space (bounce.md §4): total reflection below
    ``θc = arccos(c1/c2) = 24.62°``, the normal-incidence impedance-ratio
    plateau ``(ρ2c2−ρ1c1)/(ρ2c2+ρ1c1) ≈ 0.353``, and a phase walking from
    −180° at grazing to 0 at the critical angle (``e^{−iωt}`` convention,
    bounce.md §2), flat at 0 above it. All expectations are closed-form."""

    @pytest.fixture(scope='class')
    def rc(self):
        return Bounce(verbose=False).run(_lossless_sand_env(), _src(200.0),
                                         _rcv())

    def test_total_reflection_below_the_critical_angle(self, rc):
        theta = np.asarray(rc.angles, dtype=float)
        sub = np.asarray(rc.magnitude, dtype=float)[theta <= _SAND_THETA_C - 0.3]
        assert sub.size > 10
        np.testing.assert_allclose(sub, 1.0, atol=1e-4)

    def test_critical_angle_is_arccos_of_the_speed_ratio(self, rc):
        theta = np.asarray(rc.angles, dtype=float)
        r = np.asarray(rc.magnitude, dtype=float)
        first_loss = float(theta[r < 0.999].min())
        assert first_loss == pytest.approx(_SAND_THETA_C, abs=0.3), (
            f"|R| first drops below 1 at {first_loss:.2f} deg; "
            f"arccos(1500/1650) = {_SAND_THETA_C:.2f} deg")

    def test_normal_incidence_is_the_impedance_ratio(self, rc):
        # BOUNCE references R to a unit density and gets the seabed as a
        # ratio, so the water's real density enters the contrast.
        z1 = DEFAULT_WATER_DENSITY_G_CM3 * 1500.0
        z2 = 1.9 * 1650.0
        expected = (z2 - z1) / (z2 + z1)      # 0.3410
        i = int(np.argmax(rc.angles))
        assert rc.angles[i] == pytest.approx(90.0, abs=1e-6)
        assert rc.magnitude[i] == pytest.approx(expected, abs=5e-4)

    def test_phase_walks_from_minus_180_at_grazing_to_zero_at_critical(
            self, rc):
        theta = np.asarray(rc.angles, dtype=float)
        phi_deg = np.degrees(np.asarray(rc.phase, dtype=float))
        i0 = int(np.argmin(theta))
        assert theta[i0] == pytest.approx(0.0, abs=1e-6)
        # R = -1 at grazing: -180 deg and +180 deg are the same angle, and
        # np.angle reports it as +180.
        assert abs(phi_deg[i0]) == pytest.approx(180.0, abs=2.0)
        band = (theta > 0.5) & (theta < _SAND_THETA_C - 0.5)
        # rc.phase is BOUNCE's raw .brc phase (bounce.md: R*exp(1j*phi) is
        # exactly the coefficient BOUNCE computed): measured, it rides the
        # positive branch — +180 deg at grazing falling monotonically to 0
        # at the critical angle. Textbooks quoting the conjugate convention
        # show the mirror image, -180 -> 0.
        assert np.all(phi_deg[band] > -0.5)
        assert np.all(phi_deg[band] < 180.5)
        # Continuous, monotone sweep down to zero (0.2 deg of text-rounding
        # slack per step).
        assert np.all(np.diff(phi_deg[band]) < 0.2)
        above = theta > _SAND_THETA_C + 0.5
        assert np.all(np.abs(phi_deg[above]) < 1.0), (
            "R is real and positive above critical; phase must be 0")


class TestAttenuationSagsThePlateauNotTheCriticalAngle:
    """bounce.md §4 ('What absorption does to it'): the evanescent field
    samples the lossy sediment, so at α = 1.5 dB/λ a 10° ray gives up ~14 %
    per bounce (closed-form Rayleigh with the AT dB/λ convention: 0.860) —
    while the critical angle, set by the speed contrast alone, stays put."""

    @pytest.fixture(scope='class')
    def runs(self):
        lossless = Bounce(verbose=False).run(_lossless_sand_env(), _src(200.0),
                                             _rcv())
        lossy = Bounce(verbose=False).run(
            _halfspace_env(sound_speed=1650.0, density=1.9, attenuation=1.5,
                           shear_speed=0.0, shear_attenuation=0.0),
            _src(200.0), _rcv())
        return lossless, lossy

    def test_plateau_sags_about_14_percent_at_10_degrees(self, runs):
        lossless, lossy = runs
        assert _interp_R(lossless, 10.0) == pytest.approx(1.0, abs=1e-3)
        assert _interp_R(lossy, 10.0) == pytest.approx(0.86, abs=0.02)

    def test_the_sag_covers_the_whole_subcritical_plateau(self, runs):
        lossless, lossy = runs
        for deg in np.arange(5.0, 21.0, 2.5):
            assert _interp_R(lossy, deg) < _interp_R(lossless, deg) - 0.02, (
                f"no sag at {deg:g} deg")

    def test_the_critical_angle_stays_put(self, runs):
        # Closed form: steepest descent at 24.6 deg lossless, 25.6 deg at
        # α = 1.5; both inside ±2.5 deg of arccos(c1/c2).
        lossless, lossy = runs
        assert _steepest_descent_deg(lossless) == pytest.approx(
            _SAND_THETA_C, abs=2.5)
        assert _steepest_descent_deg(lossy) == pytest.approx(
            _SAND_THETA_C, abs=2.5)


class TestElasticGraniteShearWindow:
    """bounce.md §4 ('Shear'): between the shear (60°) and compressional
    (75°) critical angles the elastic granite radiates a shear wave and |R|
    drops to ~0.75 (closed-form fluid-solid coefficient: 0.72-0.77 across
    62-72°), while the shear-dropped fluid preset still reflects ~0.999."""

    @staticmethod
    def _run(elastic):
        bottom = BoundaryProperties.from_preset('granite', elastic=elastic)
        env = Environment(name='granite', bathymetry=100.0,
                          ssp=[(0.0, 1500.0), (100.0, 1500.0)],
                          bottom=bottom)
        return Bounce(verbose=False).run(env, _src(200.0), _rcv())

    def test_shear_radiation_cuts_R_in_the_window(self):
        rc = self._run(elastic=True)
        theta = np.asarray(rc.angles, dtype=float)
        window = np.asarray(rc.magnitude, dtype=float)[(theta >= 62.0)
                                               & (theta <= 72.0)]
        assert window.size > 5
        assert 0.65 < float(window.mean()) < 0.85, (
            f"elastic granite window mean |R| = {window.mean():.3f}")
        assert float(window.min()) < 0.80

    def test_the_fluid_preset_cannot_see_the_loss(self):
        rc = self._run(elastic=False)
        theta = np.asarray(rc.angles, dtype=float)
        window = np.asarray(rc.magnitude, dtype=float)[(theta >= 62.0)
                                               & (theta <= 72.0)]
        assert float(window.min()) > 0.99


class TestSandOverGraniteEtalonNulls:
    """bounce.md §4 ('A layer turns a mirror into a filter'): 8 m of sand on
    granite interferes its two returns into nulls near 34° and 60° at 200 Hz
    (closed-form two-interface recursion: |R| 0.178 at 34.0° and 0.609 at
    59.4°). Bounds, not exact values — the binary meshes the layer while the
    closed form does not."""

    @pytest.fixture(scope='class')
    def rc(self):
        stack = SeabedColumn.from_presets(layers=[('sand', 8.0)],
                                          halfspace='granite')
        env = Environment(name='etalon', bathymetry=100.0,
                          ssp=[(0.0, 1500.0), (100.0, 1500.0)],
                          bottom=Bottom(columns=[stack]))
        return Bounce(verbose=False).run(env, _src(200.0), _rcv())

    @staticmethod
    def _window_min(rc, lo, hi):
        theta = np.asarray(rc.angles, dtype=float)
        mask = (theta >= lo) & (theta <= hi)
        r = np.asarray(rc.magnitude, dtype=float)[mask]
        i = int(np.argmin(r))
        return float(theta[mask][i]), float(r[i])

    def test_first_null_near_34_degrees(self, rc):
        angle, depth = self._window_min(rc, 28.0, 40.0)
        assert depth < 0.30, f"|R| only reaches {depth:.3f}"
        assert 31.0 < angle < 37.0, f"null sits at {angle:.1f} deg"

    def test_second_null_near_60_degrees_is_partial(self, rc):
        angle, depth = self._window_min(rc, 54.0, 66.0)
        assert 0.40 < depth < 0.70, f"|R| at the null is {depth:.3f}"
        assert 56.0 < angle < 63.0, f"null sits at {angle:.1f} deg"


class TestHalfspaceReflectionIsFrequencyInvariant:
    """A half-space has no length scale, so R(θ) carries no frequency
    dependence (bounce.md §6): two runs at different frequencies must overlay.
    The angle grids differ (NkTab scales with ω), so both are interpolated
    onto one axis before comparing."""

    def test_two_frequencies_give_identical_curves(self):
        runs = [Bounce(verbose=False).run(_lossless_sand_env(), _src(f),
                                          _rcv())
                for f in (150.0, 600.0)]
        common = np.linspace(1.0, 89.0, 177)
        r = [np.interp(common, np.asarray(rc.angles, float),
                       np.asarray(rc.magnitude, float)) for rc in runs]
        phi = [np.interp(common, np.asarray(rc.angles, float),
                         np.degrees(np.asarray(rc.phase, float)))
               for rc in runs]
        assert np.max(np.abs(r[0] - r[1])) < 0.005
        assert np.max(np.abs(phi[0] - phi[1])) < 1.0


class TestAngleGridIsUniformInCosTheta:
    """bounce.md §7: the grid comes from a uniform sweep in horizontal
    slowness, so it is uniform in cos θ — coarse at grazing, dense at normal
    incidence."""

    def test_cos_theta_spacing_is_constant(self):
        rc = Bounce(verbose=False).run(_lossless_sand_env(), _src(200.0),
                                       _rcv())
        theta = np.asarray(rc.angles, dtype=float)
        spacing = -np.diff(np.cos(np.radians(theta)))
        assert np.all(spacing > 0)
        interior = (theta[:-1] > 1.0) & (theta[1:] < 89.0)
        assert np.allclose(spacing[interior], np.median(spacing[interior]),
                           rtol=0.05), (
            "the angle grid is not uniform in cos(theta)")


class TestRmaxResolutionAndReceiverInertness:
    """bounce.md §5/§7: ``rmax_m=None`` auto-derives from
    ``receiver.range_max`` (10 km when no positive range is available), and
    that single number is all the receiver contributes — the deck writer
    takes no receiver at all. The resolved value is read off
    ``run_settings``, which is what the deck is written from
    (``TestTheDeckIsWrittenFromTheResolvedSettings``)."""

    @staticmethod
    def _resolved_rmax(model, receiver):
        return model.run_settings(_halfspace_env(shear_speed=0.0),
                                  _src(200.0), receiver).engine.rmax_m

    def test_rmax_defaults_to_the_receiver_range_max(self):
        rmax = self._resolved_rmax(
            Bounce(verbose=False), Receiver(depths=[50.0], ranges=[4000.0]))
        assert rmax == pytest.approx(4000.0)

    def test_zero_receiver_range_falls_back_to_10_km(self):
        rmax = self._resolved_rmax(
            Bounce(verbose=False), Receiver(depths=[50.0], ranges=[0.0]))
        assert rmax == pytest.approx(10000.0)

    def test_pinned_rmax_ignores_the_receiver(self):
        rmax = self._resolved_rmax(
            Bounce(rmax_m=1234.0, verbose=False),
            Receiver(depths=[50.0], ranges=[99999.0]))
        assert rmax == pytest.approx(1234.0)

    def test_the_deck_writer_takes_no_receiver(self):
        import inspect
        from uacpy.io.oalib_writer import write_bounce_input_file
        params = inspect.signature(write_bounce_input_file).parameters
        assert 'receiver' not in params

    def test_receiver_none_without_rmax_is_refused(self):
        with pytest.raises(ConfigurationError, match='Receiver is required'):
            Bounce(verbose=False).run(_halfspace_env(shear_speed=0.0),
                                      _src(200.0), None)

    def test_receiver_none_with_n_angles_derives_rmax_from_n_angles(self):
        """``n_angles`` sizes the table without the receiver, so
        ``run(env, src, None)`` is accepted and rmax inverts bounce.f90:49."""
        model = Bounce(c_low=1400.0, n_angles=200, verbose=False)
        rmax = self._resolved_rmax(model, None)
        omega = 2.0 * np.pi * 200.0
        assert rmax == pytest.approx(200 * 2.0 * np.pi / (omega / 1400.0))


class TestGradientSSPAnchorsTheTopHalfspaceAtTheSeabed:
    """bounce.md §7: the critical angle uses the water speed at the seabed.
    The deck's top half-space row (the incident medium) must carry
    ``env.ssp.sound_speed_at(seafloor)``, not the surface speed."""

    def test_deck_top_halfspace_carries_the_seafloor_speed(self, tmp_path):
        env = Environment(
            name='grad', bathymetry=100.0,
            ssp=[(0.0, 1540.0), (100.0, 1500.0)],
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1800.0, density=2.0,
                                      attenuation=0.5))
        deck = tmp_path / 'bounce_run.env'
        Bounce(verbose=False, work_dir=tmp_path, cleanup=False).run(
            env, _src(200.0), _rcv())
        lines = deck.read_text().splitlines()
        assert lines[2].strip() == '0'
        # The top half-space parameter row directly follows the TopOpt line.
        top_speed = float(lines[4].split()[1])
        assert top_speed == pytest.approx(1500.0)
        assert '1540' not in deck.read_text()


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
            Bounce(verbose=False, c_low=c_low, rmax_m=5000.0).run(env, src, rcv)

    @pytest.mark.parametrize('c_low', [1400.0, 1450.0, 1500.0])
    def test_at_or_below_the_water_speed_tabulates_from_zero(self, c_low):
        env, src, rcv = self._fixture(1500.0)
        result = Bounce(verbose=False, c_low=c_low, rmax_m=5000.0).run(
            env, src, rcv)
        theta = np.asarray(result.angles)
        assert theta.min() == pytest.approx(0.0, abs=1e-6), (
            f"c_low={c_low} against 1500 m/s water left the grazing wedge out: "
            f"table starts at {theta.min():.3f} deg")
        assert theta.max() == pytest.approx(90.0, abs=1e-6)

    def test_the_default_tabulates_from_zero_in_cold_fresh_water(self):
        """``DEFAULT_C_MIN`` is 1400 m/s, which cold fresh water undercuts, so a
        fixed 1400 default put the failure one ``Bounce()`` away with nothing
        set by the user. ``c_low=None`` derives ``min(1400, min(env.ssp))``
        instead — AT ``bounce.htm``'s "lowest speed in the problem"."""
        env, src, rcv = self._fixture(1380.0)
        result = Bounce(verbose=False, rmax_m=5000.0).run(env, src, rcv)
        theta = np.asarray(result.angles)
        assert theta.min() == pytest.approx(0.0, abs=1e-6)
        assert theta.max() == pytest.approx(90.0, abs=1e-6)


class TestRunWithBounceDerivesCLow:
    """``Bellhop.run_with_bounce`` and the auto route
    (Bellhop's ``bounce_route``) are uacpy's own choice, so they must not hand
    BOUNCE a ``c_low`` the environment rejects."""

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
        from uacpy.core.run_settings import RunMode
        env, src, rcv = self._fixture(c_water)
        model = Bellhop(verbose=False, beam_type='G', n_beams=2001,
                        launch_angles=(-80.0, 80.0), backend='fortran')
        reference = np.asarray(model.run(env, src, rcv,
                                         RunMode.COHERENT_TL).dB)
        routed = np.asarray(model.run_with_bounce(
            env, src, rcv, run_mode=RunMode.COHERENT_TL).dB)
        delta = np.abs(routed - reference)
        assert np.nanmax(delta) < 0.5, (
            f"water {c_water} m/s: BOUNCE round trip differs from the direct "
            f"half-space by up to {np.nanmax(delta):.2f} dB")

    def test_an_explicit_c_low_is_honoured(self):
        env, src, rcv = self._fixture(1500.0)
        from uacpy.models import Bellhop
        from uacpy.core.run_settings import RunMode
        with pytest.raises(ConfigurationError, match='grazing'):
            Bellhop(verbose=False, backend='fortran').run_with_bounce(
                env, src, rcv, run_mode=RunMode.COHERENT_TL, c_low=1560.0)


class TestBounceCLowDerivesFromTheEnvironment:
    """AT ``doc/bounce.htm``: "For a full 90 degree calculation set CMin to the
    lowest speed in the problem (say 1400.0) CMax to 1.0E9." The rule is *the
    lowest speed in the problem*; the 1400 is that sentence's example, and a
    fixed 1400 default put a ``ConfigurationError`` one bare ``Bounce()`` away
    for any column of cold or brackish water. 8 of the other 9 AT/OASES model
    classes already declare ``c_low: Optional[float] = None``."""

    def test_the_default_is_none_not_a_fixed_speed(self):
        param = inspect.signature(Bounce.__init__).parameters['c_low']
        assert param.default is None

    @pytest.mark.parametrize('c_water', [1540.0, 1480.0, 1400.0])
    def test_ordinary_sea_water_resolves_to_the_1400_cap(self, c_water):
        """The half of the change that must move nothing: ``min()`` pins the
        resolved value at ``DEFAULT_C_MIN`` for every column that never drops
        below it, so the wavenumber grid is bit-identical to the old fixed
        default."""
        model = Bounce(verbose=False, rmax_m=10000.0)
        env = _isothermal(c_water)
        c_low = resolve_c_low(env, c_low=model.c_low)
        assert c_low == DEFAULT_C_MIN
        assert (tabulated_angle_count(10000.0, 200.0, c_low,
                                      c_high=DEFAULT_C_MAX_UNBOUNDED)
                == tabulated_angle_count(10000.0, 200.0, DEFAULT_C_MIN,
                                         c_high=DEFAULT_C_MAX_UNBOUNDED))

    @pytest.mark.parametrize('c_water, n_angles', [(1390.0, 1438),
                                                 (1350.0, 1481)])
    def test_water_below_the_cap_widens_the_wavenumber_grid(self, c_water,
                                                            n_angles):
        """The other half: below 1400 m/s the resolved value follows the water
        and the tabulated-angle count grows with it (1428 at ``DEFAULT_C_MIN``,
        which the assertion below pins, for this 10 km / 200 Hz deck)."""
        model = Bounce(verbose=False, rmax_m=10000.0)
        env = _isothermal(c_water)
        assert resolve_c_low(env, c_low=model.c_low) == c_water
        assert tabulated_angle_count(10000.0, 200.0, DEFAULT_C_MIN,
                                     c_high=DEFAULT_C_MAX_UNBOUNDED) == 1428
        assert tabulated_angle_count(10000.0, 200.0, c_water,
                                     c_high=DEFAULT_C_MAX_UNBOUNDED) == n_angles

    def test_the_whole_profile_is_read_not_only_the_seafloor_sample(self):
        """``min(SSP)``, the manual's rule, rather than
        ``Bellhop.run_with_bounce``'s seafloor sample: a cold surface layer over
        warmer deep water has its lowest speed at the top of the column.

        Below the seafloor speed the extra samples are evanescent — ``theta=0,
        R=1, phase=180`` from ``bounce.f90``'s ``ELSEWHERE`` branch — so they
        are duplicate head rows that ``dedupe_reflection_file`` removes, not
        extra angular coverage. The manual is still followed because it is
        ground truth for the deck and the cost is rows already dropped
        downstream."""
        env = _env([[0.0, 1380.0], [100.0, 1500.0]])
        assert float(np.atleast_1d(env.ssp.sound_speed_at(env.depth))[0]) == 1500.0
        assert resolve_c_low(env, c_low=Bounce(verbose=False).c_low) == 1380.0

    def test_every_range_column_is_read_not_only_the_one_at_range_zero(self):
        """``SoundSpeedProfile.to_pairs`` returns the **range-0 column** by
        contract, so resolving through it reads one column and misses a slower
        one further out; ``SoundSpeedProfile.data`` is the whole
        ``(n_depth, n_range)`` block. ``collapse['ssp']`` defaults to ``'r0'``,
        which hides the difference; under any other documented method a bare
        ``Bounce()`` with ``c_low`` unset then raised, telling the user to leave
        ``c_low`` None — which is exactly what they had done."""
        from uacpy.core.ssp import SoundSpeedProfile
        ssp = SoundSpeedProfile(
            depths=np.array([0.0, 100.0]),
            sound_speed=np.array([[1500.0, 1300.0], [1500.0, 1300.0]]),
            ranges=np.array([0.0, 5000.0]))
        env = Environment(
            name='rd', bathymetry=100.0, ssp=ssp,
            bottom=BoundaryProperties(sound_speed=1600.0, density=1.8,
                                      attenuation=0.5))
        assert float(ssp.to_pairs()[:, 1].min()) == 1500.0, (
            "fixture no longer exercises the range-0 shortcut")
        assert resolve_c_low(env, c_low=Bounce(verbose=False).c_low) == 1300.0
        # The invariant enforced, not asserted: with a non-default collapse the
        # projected column is the slow one, and the resolved c_low has to sit
        # at or below it or the wedge guard refuses the run.
        model = Bounce(verbose=False, rmax_m=5000.0, collapse={'ssp': 'rmax'})
        assert resolve_c_low(env, c_low=model.c_low) <= 1300.0
        result = model.run(env, _SRC, _RCV)
        assert np.asarray(result.angles).min() == pytest.approx(0.0, abs=1e-6)

    def test_an_explicit_c_low_is_used_unchanged(self):
        assert resolve_c_low(
            _isothermal(1500.0),
            c_low=Bounce(verbose=False, c_low=1234.0).c_low) == 1234.0

    def test_an_explicit_c_low_above_the_water_is_refused(self):
        with pytest.raises(ConfigurationError, match='grazing'):
            Bounce(verbose=False, c_low=1560.0, rmax_m=5000.0).run(
                _isothermal(1500.0), _SRC, _RCV)

    def test_a_bare_bounce_tabulates_from_zero_in_cold_fresh_water(self):
        """End to end through the binary: the deck a stock ``Bounce()`` writes
        for a 1390 m/s column now covers the grazing wedge."""
        result = Bounce(verbose=False, rmax_m=5000.0).run(
            _isothermal(1390.0), _SRC, _RCV)
        theta = np.asarray(result.angles)
        assert theta.min() == pytest.approx(0.0, abs=1e-6)
        assert theta.max() == pytest.approx(90.0, abs=1e-6)
        assert result.run_settings.engine.c_low == 1390.0

    def test_a_bare_bounce_in_ordinary_water_records_the_1400_cap(self):
        result = Bounce(verbose=False, rmax_m=5000.0).run(
            _isothermal(1500.0), _SRC, _RCV)
        assert result.run_settings.engine.c_low == DEFAULT_C_MIN
        assert np.asarray(result.angles).min() == pytest.approx(0.0, abs=1e-6)


class _StubReached(Exception):
    """Raised by the stub Bounce once it has recorded its arguments."""


def _cold_layer_fixture():
    """A column whose slowest water is *above* the seafloor, so the two rules
    give different numbers: ``min(env.ssp)`` is 1300 m/s while the water at
    the seafloor is 1490 m/s. The rule this replaced returns 1400 here."""
    env = Environment(
        name="cold-layer", bathymetry=100.0,
        ssp=[(0.0, 1500.0), (50.0, 1300.0), (100.0, 1490.0)],
        bottom=Bottom.from_halfspace(BoundaryProperties(
            "half-space", sound_speed=1800.0, density=2.0, attenuation=0.1)))
    return (env, uacpy.Source(depths=25.0, frequencies=200.0),
            Receiver(depths=[30.0, 50.0], ranges=np.linspace(500.0, 3000.0, 6)))


def test_the_bounce_route_is_read_off_the_checked_call() -> None:
    """``run_with_bounce`` hands its BOUNCE knobs to the stages on the
    checked call (``_RunCall.engine_request``): a call carrying knobs routes
    a halfspace seabed through BOUNCE; the same call without them, or no
    call, keeps it on the deck."""
    import dataclasses
    from uacpy.core.run_settings import RunMode
    from uacpy.models.base import _RunCall
    from uacpy.models.bellhop._bounce_route import _BounceKnobs, bounce_route
    env, _, _ = _cold_layer_fixture()
    model = uacpy.Bellhop(verbose=False)
    knobs = _BounceKnobs(c_low=1400.0)
    call = _RunCall(mode=RunMode.COHERENT_TL, kwargs={},
                    engine_request=knobs)
    auto = model.auto_bounce
    assert bounce_route(env, call, auto_bounce=auto) == ('run_with_bounce',
                                                         knobs)
    assert bounce_route(
        env, dataclasses.replace(call, engine_request=None),
        auto_bounce=auto) is None
    assert bounce_route(env, None, auto_bounce=auto) is None


@pytest.mark.parametrize('entry, refused_by', [
    ('run', 'Bellhop'), ('run_with_bounce', 'Bounce')])
def test_a_precalc_seabed_is_refused_by_the_engine_that_reads_it(
        tmp_path, entry, refused_by) -> None:
    """On the deck (``run``) Bellhop's stage 2 refuses a ``'precalc'``
    seabed; routed through BOUNCE (``run_with_bounce``) the seabed never
    reaches the deck, so stage 2 reads the route off the call and leaves the
    refusal to BOUNCE, which cannot read one either."""
    irc = tmp_path / 'table.irc'
    irc.write_text('1\n0 1 0\n')
    env = Environment(
        name='irc', bathymetry=100.0, ssp=1500.0,
        bottom=BoundaryProperties(acoustic_type='precalc',
                                  reflection_file=irc))
    with pytest.raises(UnsupportedFeatureError, match='precalc') as caught:
        getattr(uacpy.Bellhop(verbose=False), entry)(
            env, uacpy.Source(depths=25.0, frequencies=200.0),
            Receiver(depths=[30.0], ranges=[1000.0]))
    assert caught.value.model_name == refused_by


def test_run_with_bounce_hands_bounce_an_unresolved_c_low(monkeypatch) -> None:
    """``run_with_bounce`` passes ``c_low=None`` through to ``Bounce``.

    Asserted on the argument ``Bounce`` actually receives, not on the text of
    the caller: any local derivation — an assignment, a keyword-argument
    conditional, a ternary — makes this a float instead of ``None`` and fails
    here. The rule this replaced took ``min(1400, water speed at the
    seafloor)``, which on this column returns 1400 and misses the 1300 m/s
    layer above it; ``Bounce`` owning the rule is what the delegation buys, so
    the resolved value is checked too.
    """
    env, source, receiver = _cold_layer_fixture()
    seen = {}

    class _StubBounce:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            raise _StubReached

    import uacpy.models.bounce as bounce_module
    monkeypatch.setattr(bounce_module, "Bounce", _StubBounce)

    with pytest.raises(_StubReached):
        uacpy.Bellhop(verbose=False).run_with_bounce(
            env, source, receiver, run_mode=uacpy.RunMode.COHERENT_TL)

    assert "c_low" in seen, "run_with_bounce did not pass c_low to Bounce"
    assert seen["c_low"] is None, (
        f"run_with_bounce resolved c_low itself and handed Bounce "
        f"{seen['c_low']!r}; Bounce owns that rule"
    )

    # And the rule Bounce applies to the forwarded None is the faithful one:
    # the lowest speed in the column, not the speed at the seafloor.
    assert resolve_c_low(env, c_low=Bounce().c_low) == pytest.approx(1300.0)


def test_run_with_bounce_forwards_an_explicit_c_low_unchanged(monkeypatch) -> None:
    """An explicit ``c_low`` still reaches ``Bounce`` as given — delegating the
    ``None`` case did not start overriding the caller."""
    env, source, receiver = _cold_layer_fixture()
    seen = {}

    class _StubBounce:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            raise _StubReached

    import uacpy.models.bounce as bounce_module
    monkeypatch.setattr(bounce_module, "Bounce", _StubBounce)

    with pytest.raises(_StubReached):
        uacpy.Bellhop(verbose=False).run_with_bounce(
            env, source, receiver, run_mode=uacpy.RunMode.COHERENT_TL,
            c_low=1350.0)

    assert seen["c_low"] == pytest.approx(1350.0)


@pytest.mark.parametrize("ssp, expected", [
    ([(0.0, 1500.0), (100.0, 1500.0)], DEFAULT_C_MIN),
    ([(0.0, 1520.0), (50.0, 1500.0), (100.0, 1490.0)], DEFAULT_C_MIN),
    ([(0.0, 1480.0), (100.0, 1510.0)], DEFAULT_C_MIN),
    ([(0.0, 1500.0), (50.0, 1300.0), (100.0, 1490.0)], 1300.0),
    ([(0.0, 1380.0), (50.0, 1450.0), (100.0, 1500.0)], 1380.0),
])
def test_derived_c_low_never_exceeds_the_bounce_reference_speed(
        ssp, expected) -> None:
    """The delegated ``c_low`` is ``min(DEFAULT_C_MIN, min(env.ssp))`` and
    stays at or below the water speed at the seafloor.

    BOUNCE references its angles to that speed (``bounce.f90:186-195``), and
    ``Bounce`` rejects a ``c_low`` above it. ``min(env.ssp)`` is a minimum over
    the whole profile, so it can never be the value that trips the rejection —
    which is what makes the delegation safe. The first three rows are ordinary
    water, where the derived value is ``DEFAULT_C_MIN`` exactly and the deck is
    unchanged from the seafloor-only rule this replaced.
    """
    env = Environment(
        name="t", bathymetry=100.0, ssp=ssp,
        bottom=Bottom.from_halfspace(BoundaryProperties(
            "half-space", sound_speed=1800.0, density=2.0, attenuation=0.1)))

    derived = resolve_c_low(env, c_low=Bounce().c_low)
    seafloor_speed = float(np.atleast_1d(
        env.ssp.sound_speed_at(env.depth))[0])

    assert derived == pytest.approx(expected)
    assert derived <= seafloor_speed


def test_bounce_empty_table_from_a_legal_deck_is_a_run_failure(monkeypatch,
                                                               tmp_path):
    """A deck the ``NkTab`` guard accepts but whose binary still writes no
    angle rows is an outcome of the run, not a bad configuration — so it
    raises ModelExecutionError, carrying the .prt tail the message cites."""
    import uacpy.models.bounce._model as bounce_module

    monkeypatch.setattr(bounce_module, 'read_reflection_coefficient',
                        lambda path: ReflectionCoefficient(
                            angles=np.array([]), magnitude=np.array([]),
                            phase=np.array([])))
    env = Environment(
        name='elastic', bathymetry=100.0, ssp=1500.0,
        bottom=BoundaryProperties(acoustic_type='half-space',
                                  sound_speed=1600.0, shear_speed=400.0,
                                  density=1.8, attenuation=0.2,
                                  shear_attenuation=0.5))
    with pytest.raises(ModelExecutionError,
                       match='empty reflection-coefficient'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            Bounce(verbose=False, work_dir=tmp_path).run(
                env=env, source=Source(depths=50.0, frequencies=100.0),
                receiver=Receiver(depths=[50.0], ranges=[3000.0]))


class TestBounceMeshFollowsItsOwnManual:
    """``doc/bounce.htm``: "BOUNCE is very fast, there's no reason to skimp...
    I'll suggest perhaps 100 points/wavelength as a good balance between run
    time and accuracy. I have seen cases where 10 points/wavelength gave very
    poor accuracy in R( theta )."

    The wrapper previously used 20 — the density AT's GENERIC automatic mesh
    uses (``ReadEnvironmentMod.f90:103``), not BOUNCE's own recommendation, and
    the docstring wrongly said the manual "states the same rule". 10/wavelength
    is precisely the binary's acceptance floor (it sizes ``Nneeded`` at 20 and
    rejects below ``Nneeded/2``), so the coarsest deck BOUNCE accepts is the
    one its manual calls very poor.

    Measured on a 20 m sediment layer over a half-space at 200 Hz against a
    converged 400/wavelength reference: max |dR| = 0.0049 at 20/wavelength,
    0.0031 at 50, 0.00074 at 100, 0.00015 at 200. Cost at 2 kHz: 0.02 s
    against 0.03 s.
    """

    @staticmethod
    def _layered(thickness=20.0):
        import uacpy
        from uacpy.core.environment import SeabedColumn, SedimentLayer
        return uacpy.Environment(
            bathymetry=100.0, ssp=1500.0,
            bottom=SeabedColumn(
                layers=[SedimentLayer(thickness=thickness, sound_speed=1600.0,
                                      density=1.6, attenuation=0.3)],
                halfspace=uacpy.BoundaryProperties(
                    acoustic_type='half-space', sound_speed=1800.0,
                    density=2.0, attenuation=0.5)))

    def test_the_density_is_the_manuals_hundred_not_the_generic_twenty(self):
        from uacpy.models.bounce import _plan
        assert _plan._MESH_POINTS_PER_WAVELENGTH == 100
        assert _plan._AT_AUTO_MESH_POINTS_PER_WAVELENGTH == 20

    def test_a_layer_gets_a_hundred_points_per_shear_wavelength(self):
        counts = resolve_n_mesh(self._layered(), 2000.0)
        assert counts == [int(100 * 20.0 * 2000.0 / 1600.0)]

    def test_the_ceiling_clips_where_the_binary_accepts(self):
        # At 50 kHz the manual's density wants 62500 points, above the 20000
        # ceiling — but AT itself needs only 12500 and rejects below 6250, so
        # clipping keeps a deck the binary runs instead of refusing it.
        counts = resolve_n_mesh(self._layered(), 50000.0)
        assert counts == [20000]

    def test_a_stack_the_binary_would_reject_is_refused(self):
        # 400 m at 50 kHz needs 250000 points even at AT's own density, so the
        # 20000 ceiling sits below the binary's floor and refusing is right.
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError, match='mesh points'):
            resolve_n_mesh(
                self._layered(thickness=400.0), 50000.0)


class TestTheAnalyticCoefficientSharesTheEnginesConvention:
    """EXPERT-7: reflection_coeff returned Brekhovskikh & Lysanov's
    exp(-iωt) coefficient, the complex conjugate of what Bounce and OASR
    report. On a fluid seabed the two now agree in phase as well as |R|."""

    def test_phase_matches_bounce_on_a_fluid_seabed(self, tmp_path):
        from uacpy.core.acoustics import reflection_coeff
        env = _halfspace_env(shear_speed=0.0, shear_attenuation=0.0)
        res = Bounce(work_dir=tmp_path, cleanup=False).run(env, _src(), _rcv())
        hs = env.bottom.halfspace_at(range=0.0)
        pick = (res.angles > 5.0) & (res.angles < 60.0)
        R = reflection_coeff(res.angles[pick], sound_speed=hs.sound_speed,
                             density=hs.density, attenuation=hs.attenuation,
                             water_sound_speed=1500.0,
                             water_density=env.water_density)
        np.testing.assert_allclose(np.abs(R), res.magnitude[pick], atol=5e-3)
        dphi = np.angle(R * np.exp(-1j * res.phase[pick]))
        assert np.max(np.abs(dphi)) < 0.02


class TestIrcImpedanceIsOnTheAbsoluteDensityScale:
    """BOUNCE's deck states seabed densities relative to the water's, which is
    right for the ``.brc`` but leaves the ``.irc``'s ``g`` column (the seabed
    density itself for an acoustic half-space,
    ``Kraken/BCImpedanceMod.f90:85-87``) a factor ``water_density`` off the
    scale Kraken and Scooter apply it on. The wrapper rescales ``g`` once, as
    the file is made."""

    #: One ``( 5G15.7, I5 )`` row as gfortran writes it: G editing picks F
    #: form (four trailing blanks) inside 0.1..1e7 and E form outside.
    ROW = ('  0.1000000    ' '   2.000000    ' ' -0.3000000    '
           '   1.500000    ' '  0.4000000E-01' '    3')

    def test_only_the_g_columns_scale_and_the_row_layout_holds(self, tmp_path):
        from uacpy.io.refl_io import _scale_irc_impedance
        path = tmp_path / 't.irc'
        path.write_text("'BOUNCE- t'  100.0\n 1\n" + self.ROW + '\n')
        _scale_irc_impedance(path, 1.027)
        lines = path.read_text().splitlines()
        assert lines[:2] == ["'BOUNCE- t'  100.0", ' 1']
        row = lines[2]
        assert len(row) == 5 * 15 + 5
        values = [float(row[i * 15:(i + 1) * 15]) for i in range(5)]
        np.testing.assert_allclose(values, [0.1, 2.0, -0.3, 1.5 * 1.027,
                                            0.04 * 1.027], rtol=1e-7)
        assert row[75:] == '    3'

    def test_precalc_seabed_matches_the_direct_half_space_at_sea_water_density(
            self, tmp_path):
        """Measured before the rescale: mean |dTL| 0.206 dB at
        water_density=1.027 against 0.050 dB at 1.0 (tabulation residual)."""
        from uacpy.models import Scooter
        props = dict(acoustic_type='half-space', sound_speed=1600.0,
                     density=1.5, attenuation=0.5)
        env = Environment(name='irc', bathymetry=100.0, ssp=1500.0,
                          water_density=1.027,
                          bottom=BoundaryProperties(**props))
        src = Source(depths=50.0, frequencies=100.0)
        rcv = Receiver(depths=[30.0, 70.0],
                       ranges=np.linspace(1000.0, 10000.0, 46))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ref = Bounce(work_dir=tmp_path / 'b', cleanup=False).run(
                env, src, rcv)
            precalc = Environment(
                name='irc', bathymetry=100.0, ssp=1500.0, water_density=1.027,
                bottom=BoundaryProperties(
                    acoustic_type='precalc',
                    reflection_file=ref.metadata['irc_file']))
            tl_direct = np.asarray(Scooter().compute_tl(env, src, rcv).tl)
            tl_table = np.asarray(Scooter().compute_tl(precalc, src, rcv).tl)
        assert np.mean(np.abs(tl_direct - tl_table)) < 0.1


# ── the run protocol: knobs, one checking stage, the deck's settings ─────


class TestConstructorKnobsAreCheckedAtConstruction:
    """RA-WAVE-6: a knob no run could use is refused when the model is
    built, with a ``ConfigurationError`` naming it, and ``c_high=None`` is
    the default, as it is on every sibling engine."""

    def test_c_high_none_is_the_unbounded_default(self):
        model = Bounce(c_high=None)
        assert model.c_high is None
        assert repr(model) == 'Bounce()'

    @pytest.mark.parametrize('kw, name', [
        (dict(c_high='1e9'), 'c_high'),
        (dict(c_low='1400'), 'c_low'),
        (dict(c_low=float('nan')), 'c_low'),
        (dict(rmax_m=True), 'rmax_m'),
    ])
    def test_a_knob_that_is_not_a_number_is_refused(self, kw, name):
        with pytest.raises(ConfigurationError,
                           match=f'{name} must be a number'):
            Bounce(**kw)

    @pytest.mark.parametrize('n_angles, match', [
        (1, 'n_angles must be >= 2'),
        (2.5, 'whole number'),
    ])
    def test_an_unusable_angle_count_is_refused_at_construction(
            self, n_angles, match):
        with pytest.raises(ConfigurationError, match=match):
            Bounce(n_angles=n_angles)

    def test_a_whole_float_angle_count_is_accepted(self):
        assert Bounce(n_angles=50.0).n_angles == 50.0

    def test_a_knob_reassigned_after_construction_is_refused_by_the_run(
            self):
        model = Bounce(verbose=False)
        model.c_high = 'fast'
        with pytest.raises(ConfigurationError,
                           match='c_high must be a number'):
            model.validate_inputs(_halfspace_env(), _src(200.0), _rcv())

    def test_c_high_none_is_the_default_however_it_is_set(self):
        """RA-WAVE-6: ``None`` means the default at construction and when
        reassigned before a run — one spelling, one table."""
        reassigned = Bounce(verbose=False)
        reassigned.c_high = None
        for model in (reassigned, Bounce(verbose=False, c_high=None)):
            assert model.c_high is None
            assert (model.run_settings(_halfspace_env(), _src(200.0),
                                       _rcv()).engine.c_high
                    == DEFAULT_C_MAX_UNBOUNDED)


class _Launched(Exception):
    """Raised by the launch spy: the call reached the binary."""


class TestTheThreeEntryPointsRefuseAlike:
    """RA-CONTRACT-18 / ARCH-9 for Bounce's own refusals: they live in its
    ``_validate_engine`` (carriers) and ``_resolve_engine_settings`` (the
    deck), which ``validate_inputs``, ``run_settings`` and ``run`` all run,
    so the three raise the same exception with the same message and
    ``run`` launches nothing."""

    @staticmethod
    def _irc_seabed(tmp_path):
        ref = Bounce(work_dir=tmp_path / 'ref', cleanup=False).run(
            _halfspace_env(shear_speed=0.0), _src(200.0), _rcv())
        return TestReflectionTableInput._basement_env(
            ref.metadata['irc_file'], 'precalc')

    @pytest.mark.parametrize('label', [
        'a multi-frequency Source',
        'receiver=None with nothing sizing the table',
        "a 'precalc' seabed",
        'fewer than two tabulated angles',
        'c_low above the water speed',
        'c_high at or below the derived c_low',
        'a layer beyond the mesh ceiling',
    ])
    def test_validate_inputs_run_settings_and_run_raise_the_same(
            self, label, tmp_path, monkeypatch):
        env, src, rcv = _halfspace_env(shear_speed=0.0), _src(200.0), _rcv()
        model_kw = {}
        if label == 'a multi-frequency Source':
            src = Source(depths=50.0, frequencies=[100.0, 120.0])
        elif label == 'receiver=None with nothing sizing the table':
            rcv = None
        elif label == "a 'precalc' seabed":
            env = self._irc_seabed(tmp_path)
        elif label == 'fewer than two tabulated angles':
            model_kw = dict(c_low=1400.0, rmax_m=1.0)
        elif label == 'c_low above the water speed':
            model_kw = dict(c_low=1560.0, rmax_m=5000.0)
        elif label == 'c_high at or below the derived c_low':
            model_kw = dict(c_high=1300.0)
        elif label == 'a layer beyond the mesh ceiling':
            env = TestBounceMeshFollowsItsOwnManual._layered(thickness=400.0)
            src = _src(50000.0)
        model = Bounce(verbose=False, **model_kw)

        def _spy(*a, **k):
            raise _Launched
        monkeypatch.setattr(model, '_run_subprocess', _spy)
        outcomes = []
        for call in ('validate_inputs', 'run_settings', 'run'):
            try:
                getattr(model, call)(env, src, rcv)
            except _Launched:
                outcomes.append('launched')
            except Exception as exc:          # noqa: BLE001
                outcomes.append((type(exc).__name__, str(exc)))
            else:
                outcomes.append('accepted')
        assert outcomes[0] not in ('accepted', 'launched'), outcomes
        assert outcomes[0] == outcomes[1] == outcomes[2], outcomes

    def test_a_refused_deck_emits_no_weights_warning_first(self):
        """ARCH-9: the settings refusal comes before the notice that a
        REFLECTION run does not apply ``Source(weights=)``."""
        src = Source(depths=50.0, frequencies=200.0, weights=[2.0])
        with recorded_warnings() as caught:
            with pytest.raises(ConfigurationError, match='tabulated angle'):
                Bounce(verbose=False, c_low=1400.0, rmax_m=1.0).run(
                    _halfspace_env(shear_speed=0.0), src, _rcv())
        assert not caught, [str(w.message) for w in caught]

    def test_validate_inputs_checks_a_weighted_source_without_the_notice(
            self):
        """The weights-not-applied notice says how a run will go, so
        ``validate_inputs`` (which launches nothing) does not emit it;
        ``run_settings`` does, as ``run`` does."""
        src = Source(depths=50.0, frequencies=200.0, weights=[2.0])
        env, rcv = _halfspace_env(shear_speed=0.0), _rcv()
        with recorded_warnings() as caught:
            Bounce(verbose=False).validate_inputs(env, src, rcv)
        assert not caught, [str(w.message) for w in caught]
        with pytest.warns(UserWarning, match='not applied in the REFLECTION'):
            Bounce(verbose=False).run_settings(env, src, rcv)


class TestTheDeckIsWrittenFromTheResolvedSettings:
    """``run_settings(...).engine`` is what the run writes and the binary
    reads — ``cLow``/``cHigh`` and ``RMax`` on the deck, the ``NkTab`` the
    binary echoes — and the table carries the same record."""

    def test_the_deck_and_the_binary_agree_with_the_settings(self, tmp_path):
        """The receiver reaches 10 km while ``rmax_m`` is pinned at 5 km, so
        a deck step that re-derived the range from the receiver would
        disagree with the settings."""
        env, src, rcv = _halfspace_env(shear_speed=0.0), _src(200.0), _rcv()
        model = Bounce(verbose=False, rmax_m=5000.0, work_dir=tmp_path,
                       cleanup=False)
        settings = model.run_settings(env, src, rcv)
        result = model.run(env, src, rcv)
        engine = settings.engine
        assert result.run_settings == settings
        assert (engine.c_low, engine.rmax_m) == (1400.0, 5000.0)
        assert engine.c_low_origin == 'min(1400, min(env.ssp))'
        assert engine.rmax_origin == 'Bounce(rmax_m=…)'
        lines = _deck(tmp_path)
        c_low, c_high = (float(v) for v in lines[-2].split()[:2])
        assert (c_low, c_high) == (engine.c_low, engine.c_high)
        assert float(lines[-1].split()[0]) == pytest.approx(
            engine.rmax_m / 1000.0, abs=5e-7)
        prt = (tmp_path / 'bounce_run.prt').read_text()
        assert int(prt.split('NkTab =')[1].split()[0]) == engine.n_angles
        assert result.run_settings.engine.c_low == engine.c_low
        assert result.run_settings.engine.rmax_m == engine.rmax_m

    def test_the_settings_record_the_table_a_file_seabed_stages(self,
                                                                 tmp_path):
        ref = Bounce(work_dir=tmp_path / 'ref', cleanup=False).run(
            _halfspace_env(shear_speed=0.0), _src(200.0), _rcv())
        env = TestReflectionTableInput._basement_env(
            ref.metadata['brc_file'], 'file')
        settings = Bounce().run_settings(env, _src(200.0), _rcv())
        assert settings.engine.staged_table_suffix == '.brc'
        # 100 points per wavelength of a 5 m, 1650 m/s layer at 200 Hz is 61,
        # under the 100-point floor.
        assert settings.engine.n_mesh == (100,)
        from uacpy.core.run_settings import RunSettings
        back = RunSettings.from_dict(settings.to_dict())
        assert back == settings and back.engine.n_mesh == (100,)

    def test_c_low_reads_every_range_column_of_the_given_environment(self):
        """The settings resolve ``c_low`` off the environment as the caller
        gave it: its slow column sits at 5 km, which the default ``'r0'``
        projection drops before the deck is written."""
        from uacpy.core.ssp import SoundSpeedProfile
        ssp = SoundSpeedProfile(
            depths=np.array([0.0, 100.0]),
            sound_speed=np.array([[1500.0, 1300.0], [1500.0, 1300.0]]),
            ranges=np.array([0.0, 5000.0]))
        env = Environment(
            name='rd', bathymetry=100.0, ssp=ssp,
            bottom=BoundaryProperties(sound_speed=1600.0, density=1.8,
                                      attenuation=0.5))
        with pytest.warns(UserWarning, match='(?i)ssp'):
            settings = Bounce(verbose=False, rmax_m=5000.0).run_settings(
                env, _SRC, _RCV)
        assert settings.engine.c_low == 1300.0

    def test_the_output_contract_is_the_declared_one(self):
        settings = Bounce().run_settings(_halfspace_env(), _src(200.0),
                                         _rcv())
        assert settings.output == Bounce.outputs[uacpy.RunMode.REFLECTION]
        assert settings.output.result_type == 'ReflectionCoefficient'
        assert settings.output.phase_reference == 'travelling_wave'
        assert (settings.waveguide.c_min, settings.waveguide.c_max) == (
            1500.0, 1600.0)


@pytest.mark.requires_binary
class TestBounce:
    """Tests for Bounce model."""

    def test_bounce_compute_reflection_coefficient(self, simple_env, source, receiver_small, tmp_path):
        """Test Bounce reflection coefficient computation.

        Uses ``work_dir`` (with Bounce's default ``cleanup=False``) so
        the .brc/.irc files survive past the call for the consumer model.
        """
        bounce = Bounce(verbose=False, work_dir=tmp_path)

        # Bounce needs an environment with elastic bottom properties
        from uacpy.core import Environment, BoundaryProperties
        bottom = BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600,
            shear_speed=400,
            density=1.8,
            attenuation=0.2,
            shear_attenuation=0.5
        )
        env_elastic = Environment(
            name="elastic_test",
            bathymetry=simple_env.depth,
            ssp=float(simple_env.ssp.sound_speed[0, 0]),
            bottom=bottom
        )

        result = bounce.run(
            env=env_elastic,
            source=source,
            receiver=receiver_small,
        )

        assert isinstance(result, ReflectionCoefficient)
        assert 'brc_file' in result.metadata
        assert result.metadata['brc_file'] is not None

        # Check that .brc file persists in work_dir
        import os
        brc_file = result.metadata['brc_file']
        assert os.path.exists(brc_file), f"BRC file should exist: {brc_file}"

        # Check reflection coefficient data
        assert result.magnitude is not None
        assert result.angles is not None
        assert len(result.magnitude) > 0
        assert len(result.angles) > 0
        # |R| is a passive-boundary amplitude ratio, and the table is a
        # strictly increasing grazing-angle grid on [0, 90].
        R = np.asarray(result.magnitude, dtype=float)
        theta = np.asarray(result.angles, dtype=float)
        assert np.all(R >= 0.0) and np.all(R <= 1.0 + 1e-6)
        assert theta.min() >= 0.0 and theta.max() <= 90.0 + 1e-9
        assert np.all(np.diff(theta) > 0)

    def test_bounce_empty_table_raises(self, simple_env, source, tmp_path):
        """A degenerate RMax (sub-metre receiver range) makes BOUNCE emit a
        reflection table with no angle rows; the wrapper must raise a clear
        ConfigurationError, not silently return an empty ReflectionCoefficient
        (manual-test finding). Caught by the deck-level ``NkTab`` guard before
        the binary runs — the post-run empty-table check is a
        ModelExecutionError, since by then the binary has produced the table.
        """
        from uacpy.core import Environment, BoundaryProperties, Receiver
        bottom = BoundaryProperties(
            acoustic_type='half-space', sound_speed=1600, shear_speed=400,
            density=1.8, attenuation=0.2, shear_attenuation=0.5,
        )
        env_elastic = Environment(
            name="elastic_test", bathymetry=simple_env.depth,
            ssp=float(simple_env.ssp.sound_speed[0, 0]), bottom=bottom,
        )
        tiny = Receiver(depths=[50.0], ranges=[1.0])  # RMax = 1 m
        with pytest.raises(ConfigurationError, match="empty reflection-coefficient"):
            Bounce(verbose=False, work_dir=tmp_path).run(
                env=env_elastic, source=source, receiver=tiny)

    def test_bounce_compute_reflection_helper(self, simple_env, source, receiver_small, tmp_path):
        """Verify the convenience method ``Bounce.compute_reflection`` runs."""
        from uacpy.core import Environment, BoundaryProperties
        bottom = BoundaryProperties(
            acoustic_type='half-space',
            sound_speed=1600, shear_speed=400, density=1.8,
            attenuation=0.2, shear_attenuation=0.5,
        )
        env_elastic = Environment(
            name="elastic_test",
            bathymetry=simple_env.depth,
            ssp=float(simple_env.ssp.sound_speed[0, 0]),
            bottom=bottom,
        )
        bounce = Bounce(verbose=False, work_dir=tmp_path)
        result = bounce.compute_reflection(
            env=env_elastic, source=source, receiver=receiver_small,
        )
        assert isinstance(result, ReflectionCoefficient)
