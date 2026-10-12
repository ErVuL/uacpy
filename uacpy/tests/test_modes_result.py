"""``Modes``, the modal result: its phase speeds."""

import pickle

import numpy as np
import pytest
from uacpy.core.exceptions import ConfigurationError


class TestModesPhaseSpeeds:
    """``Modes.phase_speeds`` requires a frequency context;
    without one it raises :class:`ValueError`."""

    def _build_modes(self, *, frequencies):
        from uacpy.core.results import Modes
        depths = np.linspace(0, 100, 11)
        # Two trivial modes; numbers don't matter.
        k = np.array([0.4 + 0.0j, 0.3 + 0.0j])
        phi = np.zeros((len(depths), 2))
        return Modes(
            k=k, phi=phi, depths=depths,
            model='Test', frequencies=frequencies,
        )

    def test_phase_speeds_raises_without_frequency(self):
        modes = self._build_modes(frequencies=None)
        with pytest.raises(ConfigurationError, match='requires frequencies'):
            modes.phase_speeds

    def test_phase_speeds_with_frequency_is_omega_over_k(self):
        modes = self._build_modes(frequencies=100.0)
        v_p = modes.phase_speeds
        omega = 2.0 * np.pi * 100.0
        expected = omega / np.array([0.4, 0.3])
        np.testing.assert_allclose(v_p, expected, rtol=1e-12)

    def test_grazing_angles_without_frequency_names_grazing_angles(self):
        modes = self._build_modes(frequencies=None)
        with pytest.raises(ConfigurationError,
                           match='needs the mode set') as raised:
            modes.grazing_angles(1500.0)
        message = str(raised.value)
        assert message.startswith('Modes.grazing_angles:')
        assert 'phase_speeds' not in message
        assert message.endswith('.')


class TestTheMediaTableIsAnAttribute:
    """``Modes.media`` (a :class:`MediaTable`) holds the densities the modes
    were normalised in: a ``metadata`` carrying one of its members is
    refused, and every derived mode set carries it."""

    TABLE = dict(water_density=1.0, tops=[0.0, 100.0, 130.0],
                 densities=[1.0, 1.8, 1.9], bottom_depth=150.0,
                 halfspace_density=2.0)

    def _modes(self, table=None, **kw):
        from uacpy.core.results import MediaTable, Modes
        depths = np.linspace(0.0, 100.0, 11)
        return Modes(k=np.array([0.4 + 1e-6j, 0.3 + 2e-6j]),
                     phi=np.ones((depths.size, 2)), depths=depths,
                     model='Test', frequencies=100.0,
                     media=MediaTable(**(table or self.TABLE)), **kw)

    @pytest.mark.parametrize('key,member', [
        ('water_density', 'water_density'), ('media_depths', 'tops'),
        ('media_densities', 'densities'),
        ('media_bottom_depth', 'bottom_depth'),
        ('halfspace_density', 'halfspace_density')])
    def test_a_metadata_holding_a_member_is_refused(self, key, member):
        from uacpy.core.results import Modes
        with pytest.raises(ConfigurationError,
                           match=rf"metadata carries \['{key}'\]") as info:
            Modes(k=np.array([0.4 + 0j]), phi=np.ones((2, 1)),
                  depths=np.array([0.0, 50.0]), metadata={key: 1.0})
        assert f'media=MediaTable({member}=...)' in info.value.remediation

    def test_a_media_that_is_not_a_table_is_refused(self):
        from uacpy.core.results import Modes
        with pytest.raises(ConfigurationError, match='not a MediaTable'):
            Modes(k=np.array([0.4 + 0j]), phi=np.ones((2, 1)),
                  depths=np.array([0.0, 50.0]),
                  media={'water_density': 1.0})

    def test_tops_without_one_density_each_are_refused(self):
        from uacpy.core.results import MediaTable
        with pytest.raises(ConfigurationError, match='one density per'):
            MediaTable(water_density=1.0, tops=[0.0, 100.0],
                       densities=[1.0])
        with pytest.raises(ConfigurationError, match='one density per'):
            MediaTable(water_density=1.0, tops=[0.0, 100.0])

    def test_a_derived_mode_set_carries_it(self):
        modes = self._modes()
        assert modes.first_n(1).media == modes.media
        assert modes.with_attenuation(np.zeros(11)).media == modes.media

    def test_it_round_trips_through_the_dict(self):
        from uacpy.core.results import Modes
        modes = self._modes()
        assert Modes.from_dict(modes.to_dict()).media == modes.media


    @pytest.mark.parametrize('z_s,rho', [(50.0, 1.1), (100.0, 1.1)])
    def test_the_modal_sum_divides_by_the_density_the_table_gives(self, z_s,
                                                                  rho):
        modes = self._modes(table={**self.TABLE, 'water_density': 1.1})
        assert modes.media.density_at(z_s) == rho
        kw = dict(receiver_depths=np.array([50.0]),
                  ranges=np.array([1000.0]))
        read = modes.modal_pressure_field(source_depth=z_s, **kw)
        given = modes.modal_pressure_field(source_depth=z_s,
                                           source_density=rho, **kw)
        unit = modes.modal_pressure_field(source_depth=z_s,
                                          source_density=1.0, **kw)
        np.testing.assert_array_equal(read.data, given.data)
        assert not np.allclose(read.data, unit.data)


class TestTheModeCountSplitsAtTheSeabedSpeed:
    """``MediaTable.halfspace_sound_speed`` and ``halfspace_attenuation``
    are the fluid seabed's speed and loss: when the speed is known the repr
    counts the modes faster than it apart from the trapped ones, both travel
    with every saved form, and a table without them (no fluid half-space)
    keeps ``None`` in every saved form."""

    @staticmethod
    def _modes(speeds, c_bottom=1800.0, frequencies=100.0):
        from uacpy.core.results import MediaTable, Modes
        depths = np.linspace(0.0, 100.0, 5)
        k = 2.0 * np.pi * 100.0 / np.asarray(speeds, dtype=float) + 0j
        return Modes(k=k, phi=np.ones((depths.size, k.size)), depths=depths,
                     model='Test', frequencies=frequencies,
                     media=MediaTable(water_density=1.0, bottom_depth=100.0,
                                      halfspace_sound_speed=c_bottom,
                                      halfspace_attenuation=0.5))

    @pytest.mark.parametrize('speed,bit', [
        (1799.9, ' 3 modes, '),
        (1800.1, ' 3 modes (2 trapped, 1 above the 1800 m/s seabed), ')])
    def test_a_mode_counts_as_above_only_past_the_seabed_speed(self, speed,
                                                               bit):
        assert bit in repr(self._modes([1500.0, 1700.0, speed]))

    @pytest.mark.parametrize('kw', [{'c_bottom': None},
                                    {'frequencies': None}])
    def test_without_the_seabed_speed_or_a_frequency_the_count_is_plain(
            self, kw):
        assert ' 3 modes, ' in repr(self._modes([1500.0, 1700.0, 1850.0],
                                                 **kw))

    @pytest.mark.parametrize('speed', [0.0, -1500.0, np.inf, np.nan])
    def test_a_speed_that_is_not_positive_and_finite_is_refused(self, speed):
        from uacpy.core.results import MediaTable
        with pytest.raises(ConfigurationError,
                           match='halfspace_sound_speed must be a positive'):
            MediaTable(water_density=1.0, halfspace_sound_speed=speed)

    @pytest.mark.parametrize('loss', [-1e-9, np.inf, np.nan])
    def test_an_attenuation_that_is_negative_or_not_finite_is_refused(
            self, loss):
        from uacpy.core.results import MediaTable
        with pytest.raises(ConfigurationError,
                           match='halfspace_attenuation must be a non-neg'):
            MediaTable(water_density=1.0, halfspace_attenuation=loss)

    def test_a_lossless_half_space_is_kept(self):
        from uacpy.core.results import MediaTable
        assert MediaTable(water_density=1.0,
                          halfspace_attenuation=0.0).halfspace_attenuation \
            == 0.0

    def test_the_smallest_positive_speed_is_kept(self):
        from uacpy.core.results import MediaTable
        speed = MediaTable(water_density=1.0,
                           halfspace_sound_speed=1e-9).halfspace_sound_speed
        assert speed == 1e-9

    def test_it_round_trips_through_the_dict(self):
        from uacpy.core.results import Modes
        back = Modes.from_dict(self._modes([1500.0, 1850.0]).to_dict())
        assert back.media.halfspace_sound_speed == 1800.0
        assert back.media.halfspace_attenuation == 0.5
        assert '(1 trapped, 1 above the 1800 m/s seabed)' in repr(back)

    def test_it_round_trips_through_xarray_and_netcdf(self, tmp_path):
        pytest.importorskip('xarray')
        pytest.importorskip('h5netcdf')
        from uacpy.core.results import Modes
        modes = self._modes([1500.0, 1850.0])
        modes.to_netcdf(tmp_path / 'm.nc')
        for back in (Modes.from_xarray(modes.to_xarray()),
                     Modes.from_netcdf(tmp_path / 'm.nc')):
            assert back.media.halfspace_sound_speed == 1800.0
            assert back.media.halfspace_attenuation == 0.5
            assert '(1 trapped, 1 above the 1800 m/s seabed)' in repr(back)

    _FIVE_MEMBERS = dict(water_density=1.0, tops=[0.0, 100.0],
                         densities=[1.0, 1.8], bottom_depth=100.0,
                         halfspace_density=2.0)


    def test_a_table_without_them_keeps_none_through_every_saved_form(
            self, tmp_path):
        """A seabed with no fluid half-space leaves both members ``None``:
        the pickle, the dict and the NetCDF form keep them ``None``, and the
        repr counts the modes plainly."""
        pytest.importorskip('xarray')
        pytest.importorskip('h5netcdf')
        from uacpy.core.results import MediaTable, Modes
        table = MediaTable(**self._FIVE_MEMBERS)
        assert table.halfspace_sound_speed is None
        assert table.halfspace_attenuation is None
        assert pickle.loads(pickle.dumps(table)) == table
        modes = self._modes([1500.0, 1700.0, 1850.0])
        modes = Modes(k=modes.k, phi=modes.phi, depths=modes.depths,
                      model='Test', frequencies=100.0, media=table)
        modes.to_netcdf(tmp_path / 'm.nc')
        for back in (Modes.from_dict(modes.to_dict()),
                     Modes.from_netcdf(tmp_path / 'm.nc')):
            assert back.media == table
            assert ' 3 modes, ' in repr(back)
