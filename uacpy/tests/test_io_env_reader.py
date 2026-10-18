"""``read_env``: the decks uacpy writes, the decks other tools write, and
the Acoustics Toolbox's own decks, read back into an ``Environment``.
"""

import numpy as np
import pytest
import uacpy
import warnings
from uacpy.core import BoundaryProperties
from uacpy.core import Environment
from uacpy.core import Source
from uacpy.core.bottom import Bottom
from uacpy.core.bottom import SeabedColumn
from uacpy.core.boundary import SedimentLayer
from uacpy.core.environment import SoundSpeedProfile
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.exceptions import FileFormatError
from uacpy.tests.conftest import _AT_REF_DIR


class TestReadEnvInvertsTheDeckWriters:
    """``read_env`` reads back what the KRAKEN-family and Bellhop writers
    write: the Environment, Source and Receiver the deck was written from,
    and the solver options under the writer's own keyword names."""

    @staticmethod
    def _env(**kw):
        from uacpy.core.absorption import FrancoisGarrison
        return Environment(
            bathymetry=kw.pop('bathymetry', 100.0),
            ssp=SoundSpeedProfile(depths=[0.0, 50.0, 100.0],
                                  sound_speed=[1500.0, 1490.0, 1510.0]),
            bottom=SeabedColumn(
                layers=[SedimentLayer(thickness=10.0, sound_speed=1600.0,
                                      density=1.6, attenuation=0.2,
                                      roughness=0.25)],
                halfspace=BoundaryProperties(
                    acoustic_type='half-space', sound_speed=1800.0,
                    shear_speed=300.0, density=2.0, attenuation=0.5,
                    shear_attenuation=0.1)),
            absorption=FrancoisGarrison(temperature=12.0, salinity=34.0,
                                        pH=8.0),
            name='round trip', water_density=1.03, **kw)

    def test_a_kraken_deck_reads_back_its_environment(self, tmp_path):
        from uacpy.io import read_env, write_kraken_env_file
        env = self._env()
        freqs = np.array([50.0, 100.0, 150.0])
        src = Source(depths=[20.0, 30.0], frequencies=freqs)
        rcv = uacpy.Receiver(depths=np.linspace(5.0, 95.0, 10),
                             ranges=[1000.0])
        path = tmp_path / 'k.env'
        write_kraken_env_file(
            path, env, src, rcv, interp_ssp='linear', frequencies=freqs,
            n_mesh=0, rmax=5000.0, c_low=1400.0, c_high=2000.0)
        got, gsrc, grcv, opt = read_env(path)
        assert opt['model'] == 'kraken' and got.name == 'round trip'
        np.testing.assert_allclose(np.ravel(got.ssp.sound_speed),
                                   [1500.0, 1490.0, 1510.0])
        assert got.depth == 100.0 and got.water_density == 1.03
        (layer,) = got.bottom.columns[0].layers
        assert (layer.thickness, layer.sound_speed, layer.density,
                layer.attenuation, layer.roughness) == (10.0, 1600.0, 1.6,
                                                        0.2, 0.25)
        hs = got.bottom.halfspace_at(range=0.0)
        assert (hs.sound_speed, hs.shear_speed, hs.density, hs.attenuation,
                hs.shear_attenuation) == (1800.0, 300.0, 2.0, 0.5, 0.1)
        # A broadband deck carries one Francois-Garrison row as 'F'.
        assert got.absorption == env.absorption
        np.testing.assert_allclose(gsrc.depths, [20.0, 30.0])
        np.testing.assert_allclose(gsrc.frequencies, freqs)
        np.testing.assert_allclose(grcv.depths, rcv.depths)
        assert (opt['c_low'], opt['c_high'], opt['rmax']) == (
            1400.0, 2000.0, 5000.0)

    def test_a_bellhop_deck_reads_back_its_geometry_and_beams(self, tmp_path):
        from uacpy.io import read_env
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        env = Environment(
            bathymetry=[(0.0, 100.0), (3000.0, 150.0), (6000.0, 120.0)],
            ssp=SoundSpeedProfile(depths=[0.0, 150.0],
                                  sound_speed=[1500.0, 1520.0]),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.8,
                                      attenuation=0.4),
            name='bellhop trip')
        src = Source(depths=[25.0], frequencies=200.0, source_type='line')
        rcv = uacpy.Receiver(depths=[10.0, 50.0, 90.0],
                             ranges=np.linspace(500.0, 5000.0, 10))
        path = tmp_path / 'b.env'
        write_bellhop_env_file(path, env, src, rcv, beam_type='C',
                               source_type='X', n_beams=500,
                               launch_angles_deg=(-60.0, 60.0), r_loop=2000.0)
        got, gsrc, grcv, opt = read_env(path)
        assert opt['model'] == 'bellhop'
        bathy = np.asarray(got.bathymetry.to_pairs())
        assert bathy[0].tolist() == [0.0, 100.0]
        assert 150.0 in bathy[:, 1]
        np.testing.assert_allclose(grcv.ranges, rcv.ranges)
        np.testing.assert_allclose(grcv.depths, rcv.depths)
        assert gsrc.source_type == 'line'
        assert (opt['beam_type'], opt['source_type'], opt['n_beams'],
                opt['launch_angles_deg']) == ('C', 'X', 500, (-60.0, 60.0))
        assert opt['r_loop'] == pytest.approx(2000.0)

    @staticmethod
    def _settings(opt):
        """The writer keywords of ``read_env``'s options."""
        return {k: v for k, v in opt.items() if k not in ('model', 'title')}

    def test_a_kraken_deck_without_ranges_says_so_in_read_envs_name(
            self, tmp_path):
        """The deck holds no receiver ranges and no .flp is beside it: one
        IOWarning from read_env, naming the deck, and no Receiver warning
        about an argument the caller never passed."""
        from uacpy.io import read_env, write_kraken_env_file
        path = tmp_path / 'k.env'
        write_kraken_env_file(
            path, self._env(), Source(depths=[20.0], frequencies=100.0),
            uacpy.Receiver(depths=[10.0, 50.0], ranges=[1000.0]),
            interp_ssp='linear', rmax=5000.0, c_low=1400.0, c_high=2000.0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _env, _src, rcv, _opt = read_env(path)
        texts = [str(w.message) for w in caught]
        assert [t for t in texts if 'no ranges' in t and 'k.env' in t]
        assert not [t for t in texts if t.startswith('Receiver:')]
        assert issubclass(next(w.category for w in caught
                               if 'no ranges' in str(w.message)),
                          uacpy.IOWarning)
        np.testing.assert_array_equal(rcv.ranges, [0.0])

    def test_a_kraken_decks_settings_write_it_back_unchanged(self, tmp_path):
        """write -> read_env -> write with the returned settings is the
        identity: per-medium mesh counts included."""
        from uacpy.io import read_env, write_kraken_env_file
        src = Source(depths=[20.0], frequencies=100.0)
        rcv = uacpy.Receiver(depths=[10.0, 50.0, 90.0], ranges=[1000.0])
        first, second = tmp_path / 'a' / 'k.env', tmp_path / 'b' / 'k.env'
        first.parent.mkdir(), second.parent.mkdir()
        write_kraken_env_file(first, self._env(), src, rcv,
                              interp_ssp='linear', n_mesh=[60, 12],
                              rmax=5000.0, c_low=1400.0, c_high=2000.0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            env, gsrc, grcv, opt = read_env(first)
        assert opt['n_mesh'] == [60, 12]
        write_kraken_env_file(second, env, gsrc, grcv, **self._settings(opt))
        assert second.read_text() == first.read_text()

    def test_a_bellhop_decks_settings_write_it_back_unchanged(self, tmp_path):
        """The Bellhop options carry the writer's own keywords and no
        n_mesh, so they write the deck back byte for byte."""
        from uacpy.io import read_env
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        env = Environment(
            bathymetry=100.0,
            ssp=SoundSpeedProfile(depths=[0.0, 100.0],
                                  sound_speed=[1500.0, 1520.0]),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1700.0, density=1.8,
                                      attenuation=0.4),
            name='bellhop trip')
        src = Source(depths=[25.0], frequencies=200.0)
        rcv = uacpy.Receiver(depths=[10.0, 50.0, 90.0],
                             ranges=np.linspace(500.0, 5000.0, 10))
        first, second = tmp_path / 'a' / 'b.env', tmp_path / 'b' / 'b.env'
        first.parent.mkdir(), second.parent.mkdir()
        write_bellhop_env_file(first, env, src, rcv, n_beams=500,
                               launch_angles_deg=(-60.0, 60.0))
        env2, gsrc, grcv, opt = read_env(first)
        assert 'n_mesh' not in opt
        write_bellhop_env_file(second, env2, gsrc, grcv,
                               **self._settings(opt))
        assert second.read_text() == first.read_text()

    def test_a_sparc_deck_reads_its_ranges_and_writes_back_unchanged(
            self, tmp_path):
        """SPARC's tail — pulse, band, receiver ranges, output times, time
        march (sparc.f90:106-168) — reads into the Receiver and the writer's
        keywords, so write -> read_env -> write is the identity."""
        from uacpy.io import read_env
        from uacpy.io.oalib_writer import write_sparc_env_file
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          bottom=BoundaryProperties(acoustic_type='rigid'),
                          name='sparc trip')
        src = Source(depths=[20.0], frequencies=100.0)
        rcv = uacpy.Receiver(depths=[10.0, 50.0, 90.0],
                             ranges=[1000.0, 2000.0, 3000.0])
        first, second = tmp_path / 'a' / 's.env', tmp_path / 'b' / 's.env'
        first.parent.mkdir(), second.parent.mkdir()
        write_sparc_env_file(
            first, env, src, rcv, output_mode='R', c_low=1400.0,
            c_high=1e7, pulse_type='P', freq_min=50.0, freq_max=150.0,
            n_time_samples=101, time_max=3.0, march_start=-0.1,
            courant_factor=0.999)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            env2, gsrc, grcv, opt = read_env(first)
        assert not [w for w in caught if 'no ranges' in str(w.message)]
        np.testing.assert_allclose(grcv.ranges, [1000.0, 2000.0, 3000.0])
        assert opt['model'] == 'sparc'
        assert (opt['output_mode'], opt['pulse_type'], opt['freq_min'],
                opt['freq_max'], opt['n_time_samples'], opt['time_max'],
                opt['march_start'], opt['courant_factor']) == (
            'R', 'P', 50.0, 150.0, 101, 3.0, -0.1, 0.999)
        write_sparc_env_file(second, env2, gsrc, grcv, **self._settings(opt))
        assert second.read_text() == first.read_text()

    @pytest.mark.parametrize('edit, said', [
        (('101\n0.0 3.000000 /', '4\n0.0 0.1 0.2 0.5 /'), 'output times'),
        (('-0.100000 0.999000 0.0 0.0 0.0', '-0.100000 0.999000 0.25 0.0 0.0'),
         'alpha = 0.25'),
    ])
    def test_a_sparc_tail_the_writer_cannot_state_is_refused(
            self, tmp_path, edit, said):
        """write_sparc_env_file states the output times as 0 .. time_max
        evenly spaced and alpha = beta = V = 0 (sparc.f90:159-168), so a deck
        with uneven times or a non-zero integration record is refused rather
        than read into settings that would write another deck."""
        from uacpy.core.exceptions import UnsupportedFeatureError
        from uacpy.io import read_env
        from uacpy.io.oalib_writer import write_sparc_env_file
        path = tmp_path / 's.env'
        write_sparc_env_file(
            path, Environment(bathymetry=100.0, ssp=1500.0,
                              bottom=BoundaryProperties(acoustic_type='rigid')),
            Source(depths=[20.0], frequencies=100.0),
            uacpy.Receiver(depths=[10.0, 50.0], ranges=[1000.0, 2000.0]),
            output_mode='R', c_low=1400.0, c_high=1e7, pulse_type='P',
            freq_min=50.0, freq_max=150.0, n_time_samples=101, time_max=3.0,
            march_start=-0.1, courant_factor=0.999)
        text = path.read_text()
        assert text.count(edit[0]) == 1
        path.write_text(text.replace(*edit))
        with pytest.raises(UnsupportedFeatureError, match=said):
            read_env(path)

    def test_a_scooter_decks_topopt_extra_writes_it_back_unchanged(
            self, tmp_path):
        """TopOpt(7) '0' (SCOOTER's stabilising attenuation off,
        scooter.f90:130) is returned as topopt_extra, so the round trip
        keeps it."""
        from uacpy.io import read_env
        from uacpy.io.oalib_writer import write_scooter_env_file
        src = Source(depths=[20.0], frequencies=100.0)
        rcv = uacpy.Receiver(depths=[10.0, 50.0, 90.0], ranges=[1000.0])
        first, second = tmp_path / 'a' / 's.env', tmp_path / 'b' / 's.env'
        first.parent.mkdir(), second.parent.mkdir()
        write_scooter_env_file(first, self._env(), src, rcv,
                               topopt_extra='0', interp_ssp='linear',
                               n_mesh=[60, 12], rmax=5000.0,
                               c_low=1400.0, c_high=2000.0)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            env, gsrc, grcv, opt = read_env(first)
        assert opt['topopt_extra'] == '0'
        write_scooter_env_file(second, env, gsrc, grcv,
                               **self._settings(opt))
        assert second.read_text() == first.read_text()

    def test_a_kraken_deck_beside_its_field_flp_writes_back_on_its_mode_depths(
            self, tmp_path):
        """With a field.exe .flp beside it, the Receiver holds the .flp's
        receivers and mode_depths the .env's own; the settings without
        mode_depths and a Receiver on mode_depths write the .env back."""
        from uacpy.io import read_env, write_kraken_env_file
        from uacpy.io.oalib_writer import write_fieldflp
        src = Source(depths=[20.0], frequencies=100.0)
        rcv = uacpy.Receiver(depths=[10.0, 50.0, 90.0], ranges=[1000.0])
        first, second = tmp_path / 'a' / 'k.env', tmp_path / 'b' / 'k.env'
        first.parent.mkdir(), second.parent.mkdir()
        write_kraken_env_file(first, self._env(), src, rcv,
                              interp_ssp='linear', n_mesh=[60, 12],
                              rmax=5000.0, c_low=1400.0, c_high=2000.0)
        write_fieldflp(first.with_suffix('.flp'), 'RA  ',
                       source_depths=[20.0], receiver_depths=[30.0, 60.0],
                       receiver_ranges=[1000.0, 2000.0])
        env, gsrc, grcv, opt = read_env(first)
        np.testing.assert_allclose(grcv.depths, [30.0, 60.0])
        np.testing.assert_allclose(opt['mode_depths'], [10.0, 50.0, 90.0])
        settings = {k: v for k, v in self._settings(opt).items()
                    if k != 'mode_depths'}
        write_kraken_env_file(
            second, env, gsrc,
            uacpy.Receiver(depths=opt['mode_depths'], ranges=[1000.0]),
            **settings)
        assert second.read_text() == first.read_text()

    def test_a_multi_profile_kraken_deck_is_refused(self, tmp_path):
        """kraken.f90:42-46 reads one block per profile; reading the first
        alone would drop the range dependence, so read_env refuses."""
        from uacpy.core.exceptions import UnsupportedFeatureError
        from uacpy.io import read_env
        from uacpy.io.oalib_writer import write_multi_profile_env
        near = Environment(bathymetry=100.0, ssp=1500.0, name='p1')
        far = Environment(bathymetry=150.0, ssp=1490.0, name='p2')
        path = tmp_path / 'm.env'
        write_multi_profile_env(
            path, [(0.0, near), (5000.0, far)],
            Source(depths=[20.0], frequencies=100.0),
            uacpy.Receiver(depths=[10.0, 50.0], ranges=[1000.0]),
            c_low=1400.0, c_high=1700.0)
        with pytest.raises(UnsupportedFeatureError, match='multi-profile'):
            read_env(path)

    @pytest.mark.parametrize('near', [4999.996, 4999.99999])
    def test_a_medium_ends_on_the_row_at_its_base_by_ats_test(self, tmp_path,
                                                              near):
        """AT ends a medium on the row within 100*EPSILON(1.0) = 1.19e-5 m
        of its base (misc/sspMod.f90:353): a row 4 mm above a 5000 m base is
        read on to the base row; one 1e-5 m above it ends the medium."""
        from uacpy.io import read_env
        path = tmp_path / 'deep.env'
        path.write_text(
            "'deep'\n100.0\n1\n'CVW'\n0 0.0 5000.0\n"
            f" 0.0 1500.0 /\n {near} 1510.0 /\n"
            + (" 5000.0 1511.0 /\n" if near < 4999.999 else "")
            + "'A' 0.0\n 5000.0 1600.0 0.0 1.5 0.5 /\n"
            "1400 2000\n10\n1\n50 /\n1\n50 /\n")
        env, _src, _rcv, _opt = read_env(path)
        assert env.ssp.depths[-1] == pytest.approx(
            5000.0 if near < 4999.999 else near)
        assert env.bottom.halfspace_at(range=0.0).sound_speed == 1600.0

    def test_an_attenuation_unit_other_than_db_per_wavelength_is_refused(
            self, tmp_path):
        from uacpy.core.exceptions import UnsupportedFeatureError
        from uacpy.io import read_env
        path = tmp_path / 'n.env'
        path.write_text("'t'\n100.0\n1\n'CVN'\n0 0.0 100.0\n"
                        " 0.0 1500.0 /\n 100.0 1500.0 /\n'V' 0.0\n"
                        "1400 2000\n10\n1\n50 /\n1\n50 /\n")
        with pytest.raises(UnsupportedFeatureError, match="TopOpt\\(3\\)"):
            read_env(path)

    def test_a_gradient_sediment_medium_is_refused(self, tmp_path):
        from uacpy.core.exceptions import UnsupportedFeatureError
        from uacpy.io import read_env
        path = tmp_path / 'g.env'
        path.write_text("'t'\n100.0\n2\n'CVW'\n0 0.0 100.0\n"
                        " 0.0 1500.0 /\n 100.0 1500.0 /\n"
                        "0 0.0 110.0\n 100.0 1600.0 0.0 1.6 0.2 /\n"
                        " 110.0 1650.0 0.0 1.6 0.2 /\n'V' 0.0\n"
                        "1400 2000\n10\n1\n50 /\n1\n50 /\n")
        with pytest.raises(UnsupportedFeatureError, match='gradient'):
            read_env(path)
        # The same medium uniform reads.
        path.write_text(path.read_text().replace('1650.0', '1600.0'))
        env, *_ = read_env(path)
        assert env.bottom.columns[0].layers[0].thickness == 10.0


_KRAKEN_TAIL = "1400 2000\n10\n1\n50 /\n1\n50 /\n"


class TestReadEnvReadsDecksOtherToolsWrite:
    """``read_env`` reads a deck the way the Acoustics-Toolbox programs read
    it, not only the layout uacpy's writers emit."""

    def test_the_bellhop_step_and_ray_box_are_one_read(self, tmp_path):
        from uacpy.io import read_env
        path = tmp_path / 'b.env'
        path.write_text("'t'\n100.0\n1\n'CVW'\n0 0.0 100.0\n 0.0 1500.0 /\n"
                        " 100.0 1500.0 /\n'V' 0.0\n1\n25.0 /\n2\n10.0 50.0 /"
                        "\n3\n1.0 3.0 /\n'CG'\n11\n-10 10 /\n"
                        "5.0 120.0 4.0   ! STEP ZBOX RBOX\n")
        *_, opt = read_env(path)
        assert (opt['ray_step'], opt['z_box'], opt['r_box']) == (5.0, 120.0,
                                                             4000.0)
        assert opt['launch_angles_deg'] == (-10.0, 10.0)

    def test_a_bellhop_deck_with_no_beam_angles_is_refused(self, tmp_path):
        """A ``/`` straight after NBeams gives BELLHOP no fan to trace."""
        from uacpy.io import read_env
        path = tmp_path / 'b.env'
        path.write_text("'t'\n100.0\n1\n'CVW'\n0 0.0 100.0\n 0.0 1500.0 /\n"
                        " 100.0 1500.0 /\n'V' 0.0\n1\n25.0 /\n2\n10.0 50.0 /"
                        "\n3\n1.0 3.0 /\n'CG'\n11\n/\n"
                        "5.0 120.0 4.0   ! STEP ZBOX RBOX\n")
        with pytest.raises(FileFormatError,
                           match='no beam angles after NBeams'):
            read_env(path)

    @pytest.mark.parametrize('n_beams, angles, refused', [
        (5, '-20 -10 0 10 20', False),         # an evenly spaced list
        (5, '-20 -15 0 10 20', True),          # explicit, not a fan
        (5, '20 10 0 -10 -20', False),         # sorted by Bellhop first
        (7, '-20 -10 0 10 20', True),          # a fan, but not NBeams long
    ])
    def test_an_explicit_launch_angle_list_reads_only_as_one_fan(
            self, tmp_path, n_beams, angles, refused):
        """angleMod.f90:54-61 keeps an explicit list as given (then sorts
        it); the carriers hold a fan as its two ends, so a list that is not
        one evenly spaced fan of NBeams angles is refused rather than
        approximated."""
        from uacpy.core.exceptions import UnsupportedFeatureError
        from uacpy.io import read_env
        path = tmp_path / 'b.env'
        path.write_text("'t'\n100.0\n1\n'CVW'\n0 0.0 100.0\n 0.0 1500.0 /\n"
                        " 100.0 1500.0 /\n'V' 0.0\n1\n25.0 /\n2\n10.0 50.0 /"
                        f"\n3\n1.0 3.0 /\n'CG'\n{n_beams}\n{angles} /\n"
                        "5.0 120.0 4.0   ! STEP ZBOX RBOX\n")
        if refused:
            with pytest.raises(UnsupportedFeatureError,
                               match='explicit launch-angle list'):
                read_env(path)
        else:
            *_, opt = read_env(path)
            assert opt['launch_angles_deg'] == (-20.0, 20.0)

    def test_a_short_row_keeps_the_previous_reads_values(self, tmp_path):
        """AT READs every row into the same variables, so a ``/`` carries
        the last row's values across media and into the half-space."""
        from uacpy.io import read_env
        path = tmp_path / 'k.env'
        path.write_text("'t'\n100.0\n2\n'CVW'\n0 0.0 100.0\n"
                        " 0.0 1500.0 0.0 1.03 /\n 100.0 1510.0 /\n"
                        "0 0.0 110.0\n 100.0 1600.0 /\n 110.0 /\n"
                        "'A' 0.0\n 110.0 /\n" + _KRAKEN_TAIL)
        env, *_ = read_env(path)
        (layer,) = env.bottom.columns[0].layers
        assert (layer.sound_speed, layer.density) == (1600.0, 1.03)
        hs = env.bottom.halfspace_at(range=0.0)
        assert (hs.sound_speed, hs.density) == (1600.0, 1.03)

    def test_a_scooter_flp_gives_the_ranges_and_the_source(self, tmp_path):
        from uacpy.io import read_env
        path = tmp_path / 's.env'
        path.write_text("'t'\n100.0\n1\n'CVW'\n0 0.0 100.0\n 0.0 1500.0 /\n"
                        " 100.0 1500.0 /\n'R' 0.0\n" + _KRAKEN_TAIL)
        path.with_suffix('.flp').write_text(
            "'XP *'  ! option\n3\n1.0 2.0 5.0 /\n")
        _, src, rcv, opt = read_env(path)
        np.testing.assert_allclose(rcv.ranges, [1000.0, 2000.0, 5000.0])
        np.testing.assert_allclose(rcv.depths, [50.0])
        assert src.source_type == 'line'
        assert src.beam_pattern == path.with_suffix('.sbp')
        assert 'mode_depths' not in opt

    def test_an_unreadable_flp_leaves_the_decks_receivers(self, tmp_path):
        from uacpy.io import read_env
        path = tmp_path / 's.env'
        path.write_text("'t'\n100.0\n1\n'CVW'\n0 0.0 100.0\n 0.0 1500.0 /\n"
                        " 100.0 1500.0 /\n'R' 0.0\n" + _KRAKEN_TAIL)
        path.with_suffix('.flp').write_text("'RP'\n0.0 1.0 1000\n")
        with pytest.warns(UserWarning, match='neither a field.exe nor'):
            _, _, rcv, _ = read_env(path)
        np.testing.assert_allclose(rcv.depths, [50.0])

    def test_a_kraken_deck_ignores_bellhops_bathymetry_flag(self, tmp_path):
        from uacpy.io import read_env
        path = tmp_path / 'k.env'
        path.write_text("'t'\n100.0\n1\n'CVW'\n0 0.0 100.0\n 0.0 1500.0 /\n"
                        " 100.0 1500.0 /\n'A*' 0.0\n 100.0 1600.0 0.0 1.5 /\n"
                        + _KRAKEN_TAIL)
        env, *_ = read_env(path)
        assert env.depth == 100.0 and not env.is_range_dependent

    @pytest.mark.parametrize('unit,alpha_p,alpha_s,expected', [
        ('W', 0.5, 0.1, (0.5, 0.1)),
        ('F', 0.5, 0.1, (0.5 * 1600.0 / 1000.0, 0.1 * 400.0 / 1000.0)),
        ('Q', 50.0, 20.0, (8.6858896 * np.pi / 50.0,
                           8.6858896 * np.pi / 20.0)),
        ('L', 0.01, 0.02, (8.6858896 * 2 * np.pi * 0.01,
                           8.6858896 * 2 * np.pi * 0.02)),
    ])
    def test_a_frequency_free_unit_reads_as_db_per_wavelength(
            self, tmp_path, unit, alpha_p, alpha_s, expected):
        """CRCI's 'F', 'Q' and 'L' equal a dB/wavelength value at each
        row's own wave speed, whatever the frequency."""
        from uacpy.io import read_env
        path = tmp_path / 'k.env'
        path.write_text(f"'t'\n100.0\n1\n'CV{unit}'\n0 0.0 100.0\n"
                        " 0.0 1500.0 /\n 100.0 1500.0 /\n'A' 0.0\n"
                        f" 100.0 1600.0 400.0 1.5 {alpha_p} {alpha_s} /\n"
                        + _KRAKEN_TAIL)
        env, *_ = read_env(path)
        hs = env.bottom.halfspace_at(range=0.0)
        assert (hs.attenuation, hs.shear_attenuation) == pytest.approx(
            expected, rel=1e-12)

    def test_bellhops_grain_size_boundary_reads_as_its_half_space(
            self, tmp_path):
        from uacpy.io import read_env
        path = tmp_path / 'g.env'
        path.write_text("'t'\n100.0\n1\n'CVW'\n0 0.0 100.0\n 0.0 1500.0 /\n"
                        " 100.0 1500.0 /\n'G' 0.0\n100.0 1.5\n1\n25.0 /\n1\n"
                        "50.0 /\n1\n1.0 /\n'C'\n11\n-10 10 /\n0.0 120.0 2.0\n")
        env, *_ = read_env(path)
        hs = env.bottom.halfspace_at(range=0.0)
        vr = (-0.0014881 * 1.5 ** 3 + 0.0213937 * 1.5 ** 2
              - 0.1382798 * 1.5 + 1.3425)
        assert hs.sound_speed == pytest.approx(1500.0 * vr, rel=1e-12)
        # The UW-APL loss parameter for 0 <= Mz < 2.6, handed to CRCI as 'L'.
        loss = ((0.4556 + 0.0245 * 1.5) * (vr / 1000) * 1500.0
                * np.log(10.0) / (40.0 * np.pi))
        assert hs.attenuation == pytest.approx(
            8.6858896 * 2 * np.pi * loss, rel=1e-12)
        assert hs.grain_size_phi == 1.5

    def test_a_bounce_deck_is_refused_by_name(self, tmp_path):
        from uacpy.core.exceptions import UnsupportedFeatureError
        from uacpy.io import read_env
        path = tmp_path / 'r.env'
        path.write_text("'t'\n100.0\n1\n'CVW'\n0 0.0 100.0\n 0.0 1500.0 /\n"
                        " 100.0 1500.0 /\n'V' 0.0\n1400 2000\n10\n")
        with pytest.raises(UnsupportedFeatureError, match='BOUNCE'):
            read_env(path)

    def test_a_truncated_vector_names_the_file_and_the_vector(self, tmp_path):
        from uacpy.io import read_env
        path = tmp_path / 't.env'
        path.write_text("'t'\n100.0\n1\n'CVW'\n0 0.0 100.0\n 0.0 1500.0 /\n"
                        " 100.0 1500.0 /\n'V' 0.0\n1400 2000\n10\n1\n50 /\n")
        with pytest.raises(FileFormatError,
                           match=r"t\.env ends before the receiver depths"):
            read_env(path)

    def test_a_hexahedral_ssp_points_at_the_3d_reader(self, tmp_path):
        from uacpy.core.exceptions import UnsupportedFeatureError
        from uacpy.io import read_env
        path = tmp_path / 'h.env'
        path.write_text("'t'\n100.0\n1\n'HVW'\n0 0.0 100.0\n 0.0 1500.0 /\n")
        with pytest.raises(UnsupportedFeatureError, match='read_ssp_3d'):
            read_env(path)

    def test_a_3d_grid_bathymetry_points_at_the_3d_reader(self, tmp_path):
        from uacpy.io import read_bathymetry
        path = tmp_path / 'g.bty'
        path.write_text("'R'\n2\n0.0 1.0\n2\n0.0 1.0\n100 100\n100 100\n")
        with pytest.raises(FileFormatError, match='read_boundary_3d'):
            read_bathymetry(path)


class TestReadEnvReadsBackUacpysOwnBellhopDecks:
    """The guard rows and columns the Bellhop writer adds for the engine
    are dropped on read, and a range-dependent seabed comes back as one."""

    @staticmethod
    def _geometry():
        return (Source(depths=[20.0], frequencies=100.0),
                uacpy.Receiver(depths=[10.0, 50.0],
                               ranges=np.linspace(500.0, 4000.0, 8)))

    def test_a_quad_ssp_reads_back_without_its_guard_columns(self, tmp_path):
        from uacpy.io import read_env
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        speeds = np.array([[1500.0, 1505.0], [1490.0, 1495.0],
                           [1510.0, 1512.0]])
        env = Environment(bathymetry=100.0, ssp=SoundSpeedProfile(
            depths=[0.0, 50.0, 100.0], sound_speed=speeds,
            ranges=[0.0, 5000.0]))
        write_bellhop_env_file(tmp_path / 'q.env', env, *self._geometry(),
                               interp_ssp='quad')
        got, *_ = read_env(tmp_path / 'q.env')
        np.testing.assert_array_equal(got.ssp.ranges, [0.0, 5000.0])
        np.testing.assert_array_equal(np.asarray(got.ssp.sound_speed), speeds)

    def test_profiles_short_of_the_ray_box_read_back_without_the_hold(
            self, tmp_path):
        """Profiles ending at 3 km inside a 4.8 km box get a held copy of
        the last one at 1.1·r_box in the .ssp; the reader drops it."""
        from uacpy.io import read_env
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        speeds = np.array([[1500.0, 1505.0], [1510.0, 1512.0]])
        env = Environment(bathymetry=100.0, ssp=SoundSpeedProfile(
            depths=[0.0, 100.0], sound_speed=speeds, ranges=[0.0, 3000.0]))
        with pytest.warns(UserWarning, match='held constant'):
            write_bellhop_env_file(tmp_path / 'h.env', env, *self._geometry(),
                                   interp_ssp='quad')
        assert '5.280000' in (tmp_path / 'h.ssp').read_text()
        got, *_ = read_env(tmp_path / 'h.env')
        np.testing.assert_array_equal(got.ssp.ranges, [0.0, 3000.0])

    def test_an_altimetry_deck_reads_back_without_its_guard_row(
            self, tmp_path):
        from uacpy.io import read_env
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        env = Environment(
            bathymetry=100.0,
            ssp=SoundSpeedProfile(depths=[0.0, 100.0],
                                  sound_speed=[1500.0, 1510.0]),
            altimetry=[(0.0, 0.0), (2000.0, 1.0), (4000.0, 0.0)])
        write_bellhop_env_file(tmp_path / 'a.env', env, *self._geometry())
        got, *_ = read_env(tmp_path / 'a.env')
        np.testing.assert_array_equal(got.ssp.depths, [0.0, 100.0])

    def test_a_range_dependent_seabed_reads_back_step_for_step(
            self, tmp_path):
        from uacpy.io import read_env
        from uacpy.io.bellhop_writer import write_bellhop_env_file
        env = Environment(
            bathymetry=100.0,
            ssp=SoundSpeedProfile(depths=[0.0, 100.0],
                                  sound_speed=[1500.0, 1510.0]),
            bottom=Bottom.from_halfspaces(
                [0.0, 2000.0, 3000.0], sound_speed=[1600.0, 1700.0, 1650.0],
                density=[1.5, 1.8, 1.6], attenuation=[0.2, 0.5, 0.3]))
        write_bellhop_env_file(tmp_path / 'rd.env', env, *self._geometry())
        got, *_ = read_env(tmp_path / 'rd.env')
        for r in (0.0, 999.0, 1001.0, 2499.0, 2501.0, 5000.0):
            want = env.bottom.halfspace_at(range=r)
            have = got.bottom.halfspace_at(range=r)
            assert (have.sound_speed, have.density, have.attenuation) == (
                want.sound_speed, want.density, want.attenuation), r


class TestReadEnvReadsTheToolboxsOwnDecks:
    """Decks shipped with the Acoustics Toolbox, one per construct."""

    @staticmethod
    def _read(rel):
        from uacpy.io import read_env
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return read_env(_AT_REF_DIR / rel)

    def test_dickins_has_the_ray_box_its_step_line_states(self):
        *_, opt = self._read('Dickins/Dickins.env')
        assert (opt['ray_step'], opt['z_box'], opt['r_box']) == (10.0, 3100.0,
                                                             101000.0)

    def test_arctic_half_space_takes_the_last_water_row(self):
        env, *_ = self._read('arctic/arcticK.env')
        hs = env.bottom.halfspace_at(range=0.0)
        assert (hs.sound_speed, hs.density) == (1510.4, 1.0)

    def test_a_scooter_deck_reads_beside_its_fields_flp(self):
        _, src, rcv, opt = self._read('BeamPattern/MunkS.env')
        assert rcv.ranges.size == 501 and rcv.ranges[-1] == 10000.0
        assert src.beam_pattern is not None and opt['model'] == 'kraken'

    def test_a_sparc_deck_reads_its_ranges_and_time_settings(self):
        _, _, rcv, opt = self._read('iso.env')
        np.testing.assert_allclose(rcv.ranges, [50.0])
        assert opt['model'] == 'sparc' and opt['output_mode'] == 'D'
        assert (opt['pulse_type'], opt['freq_max'], opt['n_time_samples'],
                opt['time_max']) == ('PN+B', 300.0, 501, 0.1)

    def test_a_sparc_deck_the_writer_cannot_state_is_refused(self):
        """head.env's output times start at 0.002 s; write_sparc_env_file
        states them from 0, so the deck could not be written back."""
        from uacpy.core.exceptions import UnsupportedFeatureError
        with pytest.raises(UnsupportedFeatureError, match='output times'):
            self._read('head/head.env')

    def test_a_multi_profile_deck_is_refused(self):
        from uacpy.core.exceptions import UnsupportedFeatureError
        with pytest.raises(UnsupportedFeatureError, match='multi-profile'):
            self._read('wedge/wedge.env')

    def test_a_bellhop3d_deck_is_refused_as_one(self):
        from uacpy.core.exceptions import UnsupportedFeatureError
        with pytest.raises(UnsupportedFeatureError, match='BELLHOP3D'):
            self._read('Bellhop3DTests/Munk/slice2dGaussian.env')

    def test_the_readable_share_of_the_shipped_decks(self):
        """244 of the 409 decks: every one that an Environment can hold and
        the vendored programs themselves can read. The rest are listed with
        their reasons in the t-io log of 2026-09-26; five of those (VolAtt/
        freeB, freeB_Inc, freeS, free_gbtB, free_gbtB_Inc, a water
        attenuation varying with depth) read as a table at the deck
        frequency. Refused besides: the four multi-profile KRAKEN decks
        (Dickins/March and Dickins/Precalc DickinsK_rd, step/stepK_rd,
        wedge/wedge), head/head.env, a SPARC tail whose output times
        start at 0.002 s, and sduct/sductBray.env, an explicit non-uniform
        launch-angle list (-20:2:-10, -9:1:10, 12:2:20)."""
        from uacpy.io import read_env
        ok = 0
        decks = sorted(_AT_REF_DIR.rglob('*.env'))
        assert len(decks) == 409
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for deck in decks:
                try:
                    read_env(deck)
                except (ConfigurationError, FileFormatError,
                        uacpy.core.exceptions.UnsupportedFeatureError):
                    continue
                ok += 1
        assert ok == 244


def test_a_francois_garrison_deck_reads_as_the_law_with_its_water():
    """AT's ``'F'`` row ``T S pH z_bar`` (``tests/VolAtt/free_FGB.env``:
    19.3 33.5 7.5 4000.7) reads as the law with that water, evaluated at each
    depth; the deck's z_bar has no place in it and is announced as
    dropped."""
    from uacpy.core.absorption import FrancoisGarrison
    from uacpy.core.exceptions import FallbackWarning
    from uacpy.io import read_env
    from uacpy.tests.conftest import recorded_warnings
    with recorded_warnings() as rec:
        env = read_env(_AT_REF_DIR / 'VolAtt' / 'free_FGB.env')[0]
    assert env.absorption == FrancoisGarrison(temperature=19.3,
                                              salinity=33.5, pH=7.5)
    (msg,) = [str(w.message) for w in rec
              if issubclass(w.category, FallbackWarning)]
    assert 'z_bar=4000.7 m' in msg and 'z_bar is dropped' in msg


def test_the_broadband_f_deck_uacpy_writes_reads_back_as_its_law(tmp_path):
    """A broadband deck writes one Francois-Garrison row as 'F' at
    mid-water column, and reads back as that law with no notice: written
    again, it is the same deck."""
    from uacpy.core.absorption import FrancoisGarrison
    from uacpy.core.exceptions import FallbackWarning
    from uacpy.io import read_env
    from uacpy.io.oalib_writer import write_kraken_env_file
    from uacpy.tests.conftest import recorded_warnings
    env = Environment(name='band', bathymetry=100.0, ssp=1500.0,
                      bottom='sand', absorption=FrancoisGarrison(
                          temperature=12.0, salinity=34.0, pH=8.0))
    deck = tmp_path / 'band.env'
    write_kraken_env_file(deck, env, Source(depths=30.0, frequencies=200.0),
                          uacpy.Receiver(depths=[20.0], ranges=[1000.0]),
                          frequencies=np.array([200.0, 300.0, 400.0]),
                          c_low=1400.0, c_high=2000.0)
    with recorded_warnings() as rec:
        back = read_env(deck)[0]
    assert back.absorption == env.absorption
    assert not [w for w in rec if issubclass(w.category, FallbackWarning)
                and 'z_bar' in str(w.message)]
    # The same deck with z_bar moved off mid-water column is not one uacpy
    # writes: its z_bar is dropped, and said so.
    text = deck.read_text()
    assert text.count('12.0000 34.0000 8.0000 50.0000\n') == 1
    deck.write_text(text.replace('12.0000 34.0000 8.0000 50.0000\n',
                                 '12.0000 34.0000 8.0000 30.0000\n'))
    with recorded_warnings() as rec:
        assert read_env(deck)[0].absorption == env.absorption
    assert [w for w in rec if issubclass(w.category, FallbackWarning)
            and 'z_bar=30 m' in str(w.message)]


def test_a_depth_varying_attenuation_column_reads_back_as_a_table(tmp_path):
    """A law the water rows carry (here a measured table) is written as α
    per row at the deck frequency; the reader takes that column back as a
    measured table at the deck frequency (no model, dB/wavelength) that
    reproduces every row."""
    from uacpy.core.absorption import AbsorptionCoefficient, _absorption_law
    from uacpy.io import read_env
    from uacpy.io.oalib_writer import write_kraken_env_file
    from uacpy.tests.conftest import (
        at_deck_water_rows, measured_absorption_table)
    env = Environment(name='measured', bathymetry=100.0, ssp=1500.0,
                      bottom='sand', absorption=measured_absorption_table())
    deck = tmp_path / 'measured.env'
    write_kraken_env_file(deck, env, Source(depths=30.0, frequencies=4000.0),
                          uacpy.Receiver(depths=[20.0], ranges=[1000.0]),
                          c_low=1400.0, c_high=2000.0)
    rows = at_deck_water_rows(deck.read_text())
    expected = _absorption_law(env.absorption)._alpha_dB_per_wavelength_at_frequency(
        4000.0, rows[:, 0], rows[:, 1])
    np.testing.assert_allclose(rows[:, 2], expected, rtol=1e-9)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        back = read_env(deck)[0]
    assert type(back.absorption) is AbsorptionCoefficient
    assert back.absorption.model is None
    np.testing.assert_allclose(
        _absorption_law(back.absorption)._alpha_dB_per_wavelength_at_frequency(
            4000.0, rows[:, 0], rows[:, 1]),
        rows[:, 2], rtol=1e-12)
