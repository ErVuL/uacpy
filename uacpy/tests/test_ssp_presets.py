"""Tests for SSP factory methods on :class:`SoundSpeedProfile`."""

import numpy as np
import pytest

from uacpy.core.environment import SoundSpeedProfile
from uacpy.core.exceptions import ConfigurationError


class TestIsovelocity:
    def test_constant_value(self):
        ssp = SoundSpeedProfile.from_isovelocity(depth_max=2000.0, sound_speed=1500.0)
        assert np.allclose(ssp.sound_speed, 1500.0)

    def test_default_value(self):
        ssp = SoundSpeedProfile.from_isovelocity(depth_max=5000.0)
        assert np.allclose(ssp.sound_speed, 1500.0)

    def test_value_returns_scalar_for_isovelocity(self):
        ssp = SoundSpeedProfile.from_isovelocity(depth_max=2000.0, sound_speed=1480.0)
        assert ssp.value == 1480.0

    def test_value_raises_when_profile_varies(self):
        ssp = SoundSpeedProfile.from_pairs([(0, 1500.0), (100, 1480.0)])
        with pytest.raises(ConfigurationError, match="varies"):
            _ = ssp.value


class TestMackenzie:
    def test_seawater_surface_value_at_the_mackenzie_reference_point(self):
        # T=15 °C, S=35 PSU, D=0 m is the reference point of Mackenzie (1981);
        # the nine-term polynomial collapses to its first four terms there
        # (every S-35 and D factor vanishes) and evaluates to 1506.692225 m/s.
        # ``abs=0.05`` only buys room to write the expectation to two decimals
        # — the closed form leaves no discretisation error to absorb.
        z = np.array([0.0])
        T = np.array([15.0])
        S = np.array([35.0])
        ssp = SoundSpeedProfile.from_temperature_salinity(z, T, S)
        assert float(ssp.sound_speed[0, 0]) == pytest.approx(1506.69, abs=0.05)

    def test_increasing_with_depth(self):
        z = np.linspace(0.0, 4000.0, 81)
        T = np.full_like(z, 4.0)
        S = np.full_like(z, 35.0)
        ssp = SoundSpeedProfile.from_temperature_salinity(z, T, S)
        assert np.all(np.diff(ssp.sound_speed[:, 0]) > 0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ConfigurationError, match="must share shape"):
            SoundSpeedProfile.from_temperature_salinity(
                depths=np.array([0.0, 100.0]),
                temperature=np.array([15.0]),
                salinity=np.array([35.0, 35.0]),
            )

    def test_the_water_is_named_as_the_formulas_name_it(self):
        with pytest.raises(TypeError, match='salinity_psu'):
            SoundSpeedProfile.from_temperature_salinity(
                [0.0], temperature=10.0, salinity_psu=35.0)


class TestIsovelocityShapeMustBeTrue:
    """``resolve_ssp_topopt`` turns ``kind='isovelocity'`` into AT
    ``TopOpt(1)='C'`` on the grounds that any connection scheme over constant
    data is constant. That reasoning only holds if the data *is* constant, so
    the declaration is checked rather than trusted — otherwise the deck
    silently flattens a gradient."""

    def test_a_gradient_declared_isovelocity_is_rejected(self):
        with pytest.raises(ConfigurationError, match='isovelocity'):
            SoundSpeedProfile.from_pairs(
                np.array([[0.0, 1500.0], [200.0, 1400.0]]),
                kind='isovelocity')

    def test_constant_data_is_accepted(self):
        ssp = SoundSpeedProfile.from_pairs(
            np.array([[0.0, 1500.0], [200.0, 1500.0]]), kind='isovelocity')
        assert ssp.kind == 'isovelocity'

    def test_the_shape_does_not_swallow_an_invalid_interp(self):
        """``resolve_ssp_topopt`` validates the model's ``interp_ssp`` before
        the ``kind='isovelocity'`` shortcut returns ``'C'``, so an
        unrecognised scheme raises on an isovelocity env exactly as it does on
        any other."""
        from uacpy.core.environment import Environment
        from uacpy.io.oalib_writer import resolve_ssp_topopt
        env = Environment(bathymetry=200.0, ssp=SoundSpeedProfile.from_pairs(
            np.array([[0.0, 1500.0], [200.0, 1500.0]]), kind='isovelocity'))
        assert resolve_ssp_topopt(env, 'linear') == 'C'
        for bad in ('analytic', 'bogus'):
            with pytest.raises(ConfigurationError,
                               match="interp_ssp='(analytic|bogus)'"):
                resolve_ssp_topopt(env, bad)


class TestMackenzieIsEvaluatedOnTheDepthsGiven:
    """Mackenzie is stated in depth, so a profile built with it at any
    latitude equals ``sound_speed_mackenzie(T, S, z)`` exactly; the latitude
    only enters the pressure-stated equations (TEOS-10 moves by ~0.3 m/s at
    6 km between 0 and 80 deg)."""

    z = np.array([0.0, 1000.0, 3000.0, 6000.0])

    def _build(self, formula, lat):
        return SoundSpeedProfile.from_temperature_salinity(
            self.z, np.full(4, 2.0), np.full(4, 34.7), formula=formula,
            latitude_deg=lat).sound_speed[:, 0]

    @pytest.mark.parametrize('lat', [0.0, 45.0, 80.0])
    def test_mackenzie_ignores_the_latitude(self, lat):
        from uacpy.core.acoustics import sound_speed_mackenzie
        np.testing.assert_array_equal(
            self._build('mackenzie', lat),
            sound_speed_mackenzie(2.0, 34.7, self.z))

    def test_a_pressure_equation_reads_the_latitude(self):
        spread = self._build('teos10', 80.0) - self._build('teos10', 0.0)
        assert spread[-1] > 0.2

    def test_the_builder_wraps_the_array_level_function(self):
        from uacpy.acoustics import sound_speed_at_depth
        np.testing.assert_array_equal(
            self._build('unesco', 30.0),
            sound_speed_at_depth(2.0, 34.7, self.z, formula='unesco',
                                 latitude_deg=30.0))


class TestTwoColumnsAreNotTwoPairs:
    """``ssp=(depths, speeds)`` with columns longer than two cannot be pairs
    and is refused naming ``from_pairs(np.column_stack(...))``. A 2x2 input
    is the documented pairs form — a tuple of two ``(depth, speed)`` pairs —
    and builds the profile; both sides pinned."""

    def test_a_tuple_of_two_long_columns_is_refused_naming_from_pairs(self):
        import uacpy
        z = np.array([0.0, 50.0, 100.0])
        c = np.array([1500.0, 1490.0, 1495.0])
        with pytest.raises(ConfigurationError, match='from_pairs'):
            uacpy.Environment(bathymetry=100, ssp=(z, c))

    def test_a_tuple_of_two_pairs_builds_the_profile(self):
        import uacpy
        env = uacpy.Environment(bathymetry=100,
                                ssp=((0.0, 1500.0), (100.0, 1510.0)))
        np.testing.assert_array_equal(env.ssp.depths, [0.0, 100.0])
        np.testing.assert_array_equal(env.ssp.sound_speed[:, 0], [1500.0, 1510.0])

    def test_a_list_of_pairs_builds_the_profile(self):
        import uacpy
        env = uacpy.Environment(bathymetry=100,
                                ssp=[(0.0, 1500.0), (50.0, 1495.0),
                                     (100.0, 1490.0)])
        np.testing.assert_array_equal(env.ssp.depths, [0.0, 50.0, 100.0])

    def test_from_pairs_on_transposed_columns_names_column_stack(self):
        with pytest.raises(ConfigurationError, match='column_stack'):
            SoundSpeedProfile.from_pairs(np.array([[0.0, 50.0, 100.0],
                                                   [1500.0, 1490.0, 1495.0]]))


class TestCastsOnTheirOwnGridsBuildOneProfile:
    """``from_casts`` takes casts with different depth grids and lengths:
    the axis is the union of their depths, a short cast holds its deepest
    value, a raw cast's repeated depths are averaged and its order sorted,
    and the result equals ``from_2d`` on the hand-interpolated matrix."""

    shelf = [(0.0, 1500.0), (60.0, 1495.0)]
    slope = [(0.0, 1502.0), (100.0, 1490.0), (800.0, 1485.0)]

    def test_the_axis_is_the_union_and_a_short_cast_holds_its_last_value(self):
        ssp = SoundSpeedProfile.from_casts([0.0, 5000.0],
                                           [self.shelf, self.slope])
        np.testing.assert_array_equal(ssp.depths, [0.0, 60.0, 100.0, 800.0])
        np.testing.assert_array_equal(ssp.sound_speed[:, 0],
                                      [1500.0, 1495.0, 1495.0, 1495.0])
        np.testing.assert_allclose(ssp.sound_speed[:, 1],
                                   [1502.0, 1502.0 - 12.0 * 0.6, 1490.0,
                                    1485.0])
        np.testing.assert_array_equal(ssp.ranges, [0.0, 5000.0])

    def test_a_raw_cast_is_sorted_and_its_repeats_averaged(self):
        raw = [(10.0, 1499.0), (0.0, 1500.0), (10.0, 1497.0), (50.0, 1495.0)]
        ssp = SoundSpeedProfile.from_casts([0.0, 1000.0], [raw, self.shelf],
                                           depths=[0.0, 10.0, 50.0])
        np.testing.assert_array_equal(ssp.sound_speed[:, 0],
                                      [1500.0, 1498.0, 1495.0])

    def test_a_column_pair_cast_is_refused_naming_column_stack(self):
        z, c = np.array([0.0, 30.0, 60.0]), np.array([1500.0, 1497.0, 1495.0])
        with pytest.raises(ConfigurationError, match='column_stack'):
            SoundSpeedProfile.from_casts([0.0, 1000.0], [(z, c), self.shelf])
        ssp = SoundSpeedProfile.from_casts(
            [0.0, 1000.0], [np.column_stack([z, c]), self.shelf])
        np.testing.assert_array_equal(ssp.sound_speed[:, 0], c)

    def test_a_2x2_cast_is_two_rows(self):
        # (0, 1500), (60, 1495): never 60 m/s at the surface or a node at
        # 1500 m, the columns reading of the same numbers.
        ssp = SoundSpeedProfile.from_casts(
            [0.0, 1000.0], [((0.0, 1500.0), (60.0, 1495.0)), self.shelf])
        np.testing.assert_array_equal(ssp.depths, [0.0, 60.0])
        np.testing.assert_array_equal(ssp.sound_speed[:, 0], [1500.0, 1495.0])

    def test_one_range_per_cast_and_at_least_two(self):
        with pytest.raises(ConfigurationError, match='one range per cast'):
            SoundSpeedProfile.from_casts([0.0], [self.shelf, self.slope])
        with pytest.raises(ConfigurationError, match='from_pairs'):
            SoundSpeedProfile.from_casts([0.0], [self.shelf])
