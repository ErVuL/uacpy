"""Unit tests for the unified ``Bottom`` / ``SeabedColumn`` carrier.

Self-contained: expected values are hand-computed golden numbers (no coupling
to any other carrier type).
"""

import pytest

from uacpy.core.bottom import (
    SedimentLayer, BoundaryProperties, SeabedColumn, Bottom,
)
from uacpy.core.exceptions import ConfigurationError


def _hs(cp=1800.0, rho=1.9, a=0.3, cs=0.0, a_s=0.0):
    return BoundaryProperties(acoustic_type='half-space', sound_speed=cp,
                              density=rho, attenuation=a, shear_speed=cs,
                              shear_attenuation=a_s)


def _layers():
    return [SedimentLayer(thickness=10, sound_speed=1600, density=1.5,
                          attenuation=0.4),
            SedimentLayer(thickness=20, sound_speed=1700, density=1.8,
                          attenuation=0.3)]


# ─── construction / queries ────────────────────────────────────────────────

class TestBottomQueries:
    def test_halfspace_flags(self):
        b = Bottom.from_halfspace(_hs())
        assert not b.is_layered and not b.is_range_dependent
        assert b.n_ranges == 1 and b.ranges is None and not b.is_elastic
        assert b.acoustic_type == 'half-space'

    def test_layered_flag(self):
        b = Bottom.from_column(SeabedColumn(_layers(), _hs()))
        assert b.is_layered and not b.is_range_dependent

    def test_rd_halfspace_flags(self):
        b = Bottom.from_halfspaces([0, 5000, 10000], sound_speed=[1600, 1700, 1800],
                                   density=[1.5, 1.6, 1.7], attenuation=[.5, .4, .3])
        assert b.is_range_dependent and not b.is_layered and b.n_ranges == 3

    def test_rd_layered_flags(self):
        col = SeabedColumn(_layers(), _hs())
        b = Bottom.from_columns([col, col], ranges=[0, 8000])
        assert b.is_range_dependent and b.is_layered

    def test_elastic_detected_layer_and_halfspace(self):
        assert Bottom.from_column(SeabedColumn(_layers(), _hs(cs=600))).is_elastic
        assert Bottom.from_column(SeabedColumn(
            [SedimentLayer(5, 1600, 1.5, 0.3, shear_speed=200)], _hs())).is_elastic

    def test_validation(self):
        with pytest.raises(ConfigurationError):       # ranges=None needs 1 column
            Bottom(columns=[SeabedColumn([], _hs()), SeabedColumn([], _hs())])
        with pytest.raises(ConfigurationError):       # length mismatch
            Bottom(columns=[SeabedColumn([], _hs())], ranges=[0, 1])
        with pytest.raises(ConfigurationError):       # non-monotone ranges
            Bottom.from_columns([SeabedColumn([], _hs())] * 2, ranges=[5, 5])


# ─── half-space column ──────────────────────────────────────────────────────

class TestHalfspaceColumn:
    def test_breakpoints(self):
        col = SeabedColumn(layers=[], halfspace=_hs(cp=1800.0))
        bp = col.to_piecewise_breakpoints(seafloor_depth=100.0, zmax=400.0)
        assert bp['sound_speed'] == [(100.0, 1800.0), (400.0, 1800.0)]

    def test_collapse_degrades_to_halfspace(self):
        col = SeabedColumn(layers=[], halfspace=_hs(cp=1800.0))
        for m in ('halfspace', 'top_layer', 'volume_average'):
            assert col.collapse(m).sound_speed == 1800.0


# ─── layered column ─────────────────────────────────────────────────────────

class TestLayeredColumn:
    def test_breakpoints_step_function(self):
        col = SeabedColumn(_layers(), _hs(cp=1800))
        bp = col.to_piecewise_breakpoints(100.0, 400.0, ('sound_speed',))
        # layer1 10 m, layer2 20 m from seafloor 100 → deepest layer bottom 130
        assert bp['sound_speed'] == [
            (100.0, 1600.0), (110.0, 1600.0),     # layer 1 (10 m)
            (110.0, 1700.0), (130.0, 1700.0),     # layer 2 (20 m)
            (130.0, 1800.0), (400.0, 1800.0),     # half-space to zmax
        ]

    def test_total_thickness(self):
        assert SeabedColumn(_layers(), _hs()).total_thickness() == 30.0

    def test_collapse_top_layer(self):
        c = SeabedColumn(_layers(), _hs(cp=1800)).collapse('top_layer')
        assert c.sound_speed == 1600 and c.density == 1.5

    def test_collapse_volume_average(self):
        # weights: layer thk [10,20] + halfspace weight = last layer thk (20)
        # cp = (10*1600 + 20*1700 + 20*1800) / (10+20+20) = 86000/50 = 1720
        c = SeabedColumn(_layers(), _hs(cp=1800, rho=1.9, a=0.3)).collapse('volume_average')
        assert c.sound_speed == pytest.approx(1720.0)


# ─── range-dependent half-space: linear blend + reductions ──────────────────

class TestRDHalfspace:
    def _b(self):
        return Bottom.from_halfspaces(
            [0, 5000, 12000], sound_speed=[1600, 1700, 1800],
            density=[1.5, 1.6, 1.7], attenuation=[0.5, 0.4, 0.3])

    def test_halfspace_at_linear(self):
        b = self._b()
        assert b.halfspace_at(range=0).sound_speed == pytest.approx(1600)
        assert b.halfspace_at(range=2500).sound_speed == pytest.approx(1650)   # midpoint
        assert b.halfspace_at(range=2500).density == pytest.approx(1.55)
        # between 5000 and 12000 at 7000: t = 2000/7000
        assert b.halfspace_at(range=7000).sound_speed == pytest.approx(
            1700 + (2000 / 7000) * 100)
        assert b.halfspace_at(range=12000).sound_speed == pytest.approx(1800)

    def test_soa_views(self):
        b = self._b()
        assert b.halfspace_sound_speed.tolist() == [1600, 1700, 1800]
        assert b.halfspace_density.tolist() == [1.5, 1.6, 1.7]

    def test_select_range(self):
        b = self._b()
        assert b.select_range('r0').columns[0].halfspace.sound_speed == 1600
        assert b.select_range('rmax').columns[0].halfspace.sound_speed == 1800
        assert b.select_range('mean').columns[0].halfspace.sound_speed == pytest.approx(1700)
        assert b.select_range('median').columns[0].halfspace.sound_speed == pytest.approx(1700)


# ─── range-dependent layered: nearest only ──────────────────────────────────

class TestRDLayered:
    def _b(self):
        near = SeabedColumn(_layers(), _hs(cp=1900))
        far = SeabedColumn([SedimentLayer(5, 1650, 1.8, 0.3)], _hs(cp=2200))
        return Bottom.from_columns([near, far], ranges=[0, 10000])

    def test_column_at_nearest(self):
        b = self._b()
        assert b.column_at(range=0).halfspace.sound_speed == 1900
        assert b.column_at(range=4000).halfspace.sound_speed == 1900   # nearer 0
        assert b.column_at(range=7000).halfspace.sound_speed == 2200   # nearer 10000

    def test_halfspace_at_is_nearest_when_layered(self):
        b = self._b()
        assert b.halfspace_at(range=4000).sound_speed == 1900
        assert b.halfspace_at(range=7000).sound_speed == 2200

    def test_max_total_thickness(self):
        assert self._b().max_total_thickness() == 30.0   # near column 10+20

    def test_mean_range_collapse_raises(self):
        with pytest.raises(ConfigurationError):
            self._b().select_range('mean')

    def test_all_sound_speeds(self):
        speeds = self._b().all_sound_speeds()
        assert set(speeds) == {1600, 1700, 1900, 1650, 2200}


# ─── two-axis collapse (base-collapse semantics) ────────────────────────────

class TestCollapse:
    def test_layers_collapse_keeps_range_axis(self):
        near = SeabedColumn(_layers(), _hs(cp=1900))
        far = SeabedColumn([SedimentLayer(5, 1650, 1.8, 0.3)], _hs(cp=2200))
        flat = Bottom.from_columns([near, far], ranges=[0, 10000]).collapse(layers='halfspace')
        assert flat.is_range_dependent and not flat.is_layered
        assert flat.halfspace_sound_speed.tolist() == [1900, 2200]

    def test_range_then_layers(self):
        near = SeabedColumn(_layers(), _hs(cp=1900))
        far = SeabedColumn(_layers(), _hs(cp=2200))
        out = Bottom.from_columns([near, far], ranges=[0, 10000]).collapse(
            range='rmax', layers='halfspace')
        assert not out.is_range_dependent and not out.is_layered
        assert out.columns[0].halfspace.sound_speed == 2200

    def test_to_halfspace(self):
        b = Bottom.from_column(SeabedColumn(_layers(), _hs(cp=1800)))
        assert b.to_halfspace().sound_speed == 1800
