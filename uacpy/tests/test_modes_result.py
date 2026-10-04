"""``Modes``, the modal result: its phase speeds."""

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


class TestTheMediaTableIsAnAttribute:
    """``Modes.media`` (a :class:`MediaTable`) holds the densities the modes
    were normalised in: a ``metadata`` carrying one of its members is
    refused, every derived mode set carries it, and a file that keeps the
    table in its metadata loads it."""

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

    def test_a_dict_that_keeps_it_in_metadata_loads_it(self):
        from uacpy.core.results import MediaTable, Modes
        d = self._modes(metadata={'title': 't'}).to_dict()
        del d['media']
        d['metadata'] = {**d['metadata'], 'water_density': 1.0,
                         'media_depths': [0.0, 100.0, 130.0],
                         'media_densities': [1.0, 1.8, 1.9],
                         'media_bottom_depth': 150.0,
                         'halfspace_density': 2.0}
        back = Modes.from_dict(d)
        assert back.media == MediaTable(**self.TABLE)
        assert back.metadata == {'title': 't'}

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
