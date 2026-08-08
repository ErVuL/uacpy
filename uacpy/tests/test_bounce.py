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
        thickness ``dz`` would show up as ``-2 k dz`` of phase here."""
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
    bracketing abscissas. BOUNCE writes the principal value, so its own output
    breaks that: a step of ~299 deg appeared between adjacent angles, and
    interpolating across it sweeps the phase the long way round."""

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
