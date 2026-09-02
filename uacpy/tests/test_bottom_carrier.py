"""Unit tests for the unified ``Bottom`` / ``SeabedColumn`` carrier.

Self-contained: expected values are hand-computed golden numbers (no coupling
to any other carrier type).
"""

import pytest

import copy
import warnings

import numpy as np

import uacpy
from uacpy.core.bottom import (
    SedimentLayer, BoundaryProperties, SeabedColumn, Bottom,
)
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.environment import Environment
from uacpy.core.materials import get_material, list_materials
from uacpy.core.surface import Surface


def _hs(cp=1800.0, rho=1.9, a=0.3, cs=0.0, a_s=0.0):
    return BoundaryProperties(acoustic_type='half-space', sound_speed=cp,
                              density=rho, attenuation=a, shear_speed=cs,
                              shear_attenuation=a_s)


def _layers():
    return [SedimentLayer(thickness=10, sound_speed=1600, density=1.5,
                          attenuation=0.4),
            SedimentLayer(thickness=20, sound_speed=1700, density=1.8,
                          attenuation=0.3)]


# ─── acoustic_type inference ────────────────────────────────────────────────

class TestAcousticTypeInference:
    """acoustic_type inference keys on *explicitly passed* parameters, not on
    value-vs-default comparison — passing the documented default values must
    still mean 'half-space'."""

    def test_bare_construction_is_vacuum(self):
        assert BoundaryProperties().acoustic_type == 'vacuum'

    def test_default_valued_params_infer_halfspace(self):
        # 1600/1.5/0.5 are exactly the resolved defaults; explicitly passing
        # them must still build a half-space, never a silent vacuum.
        bp = BoundaryProperties(sound_speed=1600.0, density=1.5,
                                attenuation=0.5)
        assert bp.acoustic_type == 'half-space'
        assert bp.sound_speed == 1600.0

    def test_single_default_valued_param_infers_halfspace(self):
        assert BoundaryProperties(sound_speed=1600.0).acoustic_type == \
            'half-space'

    def test_unset_params_resolve_to_documented_defaults(self):
        bp = BoundaryProperties(sound_speed=1700.0)
        assert bp.density == 1.5 and bp.attenuation == 0.5
        assert bp.roughness == 0.0 and bp.shear_speed == 0.0

    def test_environment_scalar_bottom_1600_is_halfspace(self):
        # The documented scalar form with the textbook sand speed — which
        # coincides with the class default — must be a half-space.
        from uacpy.core.environment import Environment
        env = Environment(bathymetry=100.0, bottom=1600)
        assert env.bottom.acoustic_type == 'half-space'
        assert env.bottom.columns[0].halfspace.sound_speed == 1600.0

    def test_explicit_vacuum_with_params_raises(self):
        with pytest.raises(ConfigurationError):
            BoundaryProperties(acoustic_type='vacuum', sound_speed=1600.0)

    def test_vacuum_bottom_reductions_stay_parameter_free(self):
        cols = [SeabedColumn([], BoundaryProperties(acoustic_type='vacuum'))
                for _ in range(2)]
        b = Bottom.from_columns(cols, ranges=[0.0, 5000.0])
        assert b.select_range('mean').acoustic_type == 'vacuum'
        assert b.halfspace_at(range=2500.0).acoustic_type == 'vacuum'


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

    def test_bottom_rejects_column_count_range_mismatch_and_nonmonotone_ranges(self):
        with pytest.raises(ConfigurationError):       # ranges=None needs 1 column
            Bottom(columns=[SeabedColumn([], _hs()), SeabedColumn([], _hs())])
        with pytest.raises(ConfigurationError):       # length mismatch
            Bottom(columns=[SeabedColumn([], _hs())], ranges=[0, 1])
        with pytest.raises(ConfigurationError):       # non-monotone ranges
            Bottom.from_columns([SeabedColumn([], _hs())] * 2, ranges=[5, 5])


class TestSeabedColumnAccessors:
    """``SeabedColumn.at(depth=)`` → material at sub-bottom depth (step
    lookup); ``isel(layer=i)`` → the SedimentLayer; no ``eval`` (distinct
    materials can't blend)."""

    def _col(self):
        # layer0: 0-10 m cp1600 · layer1: 10-30 m cp1700 · half-space cp2200
        return SeabedColumn(layers=_layers(), halfspace=_hs(cp=2200.0))

    def test_at_depth_step_lookup(self):
        c = self._col()
        assert c.at(depth=5).sound_speed == 1600
        assert c.at(depth=25).sound_speed == 1700
        assert c.at(depth=100).sound_speed == 2200       # below layers → hs
        assert c.at(depth=10).sound_speed == 1600        # boundary → upper layer

    def test_isel_layer(self):
        c = self._col()
        assert c.isel(layer=0).sound_speed == 1600
        assert isinstance(c.isel(layer=1), SedimentLayer)
        with pytest.raises(IndexError, match="SeabedColumn.isel"):
            c.isel(layer=9)

    def test_no_eval(self):
        assert not hasattr(SeabedColumn, 'eval')

    def test_at_and_sample_agree_on_boundary_convention(self):
        # at() and sample_at_depths() must share one layer-boundary rule:
        # a depth exactly on an internal boundary maps to the UPPER layer.
        c = self._col()                                  # 0-10:1600, 10-30:1700
        cs, _, _ = c.sample_at_depths(n_points=4, max_thickness=30)
        # sample depths 0, 10, 20, 30 → upper layer at the 10 m interface
        assert cs[0] == c.at(depth=0).sound_speed == 1600
        assert cs[1] == c.at(depth=10).sound_speed == 1600   # boundary → upper
        assert cs[2] == c.at(depth=20).sound_speed == 1700
        assert cs[3] == c.at(depth=30).sound_speed == 1700   # boundary → upper


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

    def test_collapse_carries_halfspace_roughness(self):
        # roughness is an interface property held by the half-space; the
        # layer-derived collapses must preserve it, not reset to 0.
        hs = _hs(cp=1800)
        hs.roughness = 0.25
        col = SeabedColumn(_layers(), hs)
        assert col.collapse('top_layer').roughness == 0.25
        assert col.collapse('volume_average').roughness == 0.25
        assert col.collapse('halfspace').roughness == 0.25


# ─── carrier construction ───────────────────────────────────────────────────

class TestSedimentLayer:
    """Tests for the SedimentLayer dataclass."""

    def test_layer_stores_fields_and_defaults_attenuation_and_zero_shear(self):
        layer = SedimentLayer(thickness=10, sound_speed=1650, density=1.9)
        assert layer.thickness == 10
        assert layer.sound_speed == 1650
        assert layer.density == 1.9
        assert layer.attenuation == 0.5  # default
        assert layer.shear_speed == 0.0  # default

    def test_validation_negative_thickness(self):
        """A negative thickness is a bad user value, so ConfigurationError."""
        with pytest.raises(ConfigurationError, match="thickness"):
            SedimentLayer(thickness=-5, sound_speed=1650, density=1.9)

    def test_validation_negative_sound_speed(self):
        """A negative sound speed is a bad user value, so ConfigurationError."""
        with pytest.raises(ConfigurationError, match="sound_speed"):
            SedimentLayer(thickness=10, sound_speed=-100, density=1.9)

    def test_elastic_layer(self):
        """Test layer with shear properties."""
        layer = SedimentLayer(
            thickness=20, sound_speed=1700, density=2.0,
            shear_speed=400, shear_attenuation=1.0
        )
        assert layer.shear_speed == 400
        assert layer.shear_attenuation == 1.0


class TestLayeredBottom:
    """Tests for the SeabedColumn class."""

    def test_two_layer_column_sums_total_thickness(self):
        lb = SeabedColumn(
            layers=[
                SedimentLayer(thickness=10, sound_speed=1550, density=1.3, attenuation=0.5),
                SedimentLayer(thickness=50, sound_speed=1650, density=1.7, attenuation=0.3),
            ],
            halfspace=BoundaryProperties(
                acoustic_type='half-space',
                sound_speed=1800, density=2.0, attenuation=0.1
            )
        )
        assert len(lb.layers) == 2
        assert lb.total_thickness() == 60

    def test_layer_depths(self):
        """Test layer depth computation."""
        lb = SeabedColumn(
            layers=[
                SedimentLayer(thickness=10, sound_speed=1550, density=1.3),
                SedimentLayer(thickness=50, sound_speed=1650, density=1.7),
            ],
            halfspace=BoundaryProperties(acoustic_type='half-space', sound_speed=1800, density=2.0)
        )
        depths = lb.layer_depths(200)
        assert depths[0] == (200, 210)
        assert depths[1] == (210, 260)

    def test_empty_layers_is_halfspace(self):
        """A 0-layer SeabedColumn is a valid pure half-space."""
        col = SeabedColumn(
            layers=[],
            halfspace=BoundaryProperties(acoustic_type='half-space', sound_speed=1800, density=2.0)
        )
        assert not col.is_layered and col.total_thickness() == 0.0

    def test_environment_with_layered_bottom(self):
        """Test Environment coerces a SeabedColumn into a layered Bottom."""
        lb = SeabedColumn(
            layers=[SedimentLayer(thickness=10, sound_speed=1550, density=1.3)],
            halfspace=BoundaryProperties(acoustic_type='half-space', sound_speed=1800, density=2.0)
        )
        env = uacpy.Environment(name='test', bathymetry=100, bottom=lb)

        assert env.has_layered_bottom
        assert not env.has_range_dependent_bottom
        assert env.bottom.columns[0] is lb
        assert env.bottom.columns[0].halfspace.sound_speed == 1800

    def test_environment_plain_boundary_properties(self):
        """A half-space bottom is a non-layered, range-independent Bottom."""
        env = uacpy.Environment(name='test', bathymetry=100)
        assert not env.has_layered_bottom and not env.bottom.is_range_dependent
        assert isinstance(env.bottom, uacpy.Bottom)


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

    def test_select_range_mean_carries_roughness(self):
        lo = _hs(); lo.roughness = 0.5
        hi = _hs(); hi.roughness = 1.5
        b = Bottom.from_columns([SeabedColumn([], lo), SeabedColumn([], hi)],
                                ranges=[0, 5000])
        assert b.select_range('mean').columns[0].halfspace.roughness == pytest.approx(1.0)
        assert b.select_range('median').columns[0].halfspace.roughness == pytest.approx(1.0)


# ─── range-dependent layered: nearest only ──────────────────────────────────

class TestRDLayered:
    def _b(self):
        near = SeabedColumn(_layers(), _hs(cp=1900))
        far = SeabedColumn([SedimentLayer(5, 1650, 1.8, 0.3)], _hs(cp=2200))
        return Bottom.from_columns([near, far], ranges=[0, 10000])

    def test_at_nearest_column(self):
        b = self._b()
        assert b.at(range=0).halfspace.sound_speed == 1900
        assert b.at(range=4000).halfspace.sound_speed == 1900   # nearer 0
        assert b.at(range=7000).halfspace.sound_speed == 2200   # nearer 10000

    def test_isel_positional_column(self):
        b = self._b()
        assert b.isel(range=0).halfspace.sound_speed == 1900
        assert b.isel(range=1).halfspace.sound_speed == 2200
        with pytest.raises(IndexError, match="Bottom.isel"):
            b.isel(range=9)

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


def test_from_grain_size_builds_halfspace():
    hs = BoundaryProperties.from_grain_size(2.0)
    assert hs.acoustic_type == 'half-space'
    assert hs.grain_size_phi == 2.0          # retained as metadata
    assert hs.sound_speed > 1500.0           # sand is faster than water
    # model= selects the conversion; the two differ.
    assert (BoundaryProperties.from_grain_size(2.0, model='apl-uw').sound_speed
            != pytest.approx(hs.sound_speed))
    # Bottom-level convenience builds a range-independent half-space.
    b = Bottom.from_grain_size(2.0)
    assert b.halfspace_at(range=0.0).acoustic_type == 'half-space'


def test_seabedcolumn_copy_is_deep():
    c = SeabedColumn([], BoundaryProperties())
    assert c.copy() is not c and type(c.copy()) is SeabedColumn


# ─── from_halfspaces roughness ─────────────────────────────────────────────

def test_from_halfspaces_roughness_scalar():
    b = Bottom.from_halfspaces([0.0, 5000.0], sound_speed=[1600.0, 1700.0],
                               density=[1.5, 1.6], attenuation=[0.5, 0.4],
                               roughness=0.3)
    assert all(c.halfspace.roughness == pytest.approx(0.3) for c in b.columns)


def test_from_halfspaces_roughness_per_range():
    b = Bottom.from_halfspaces([0.0, 5000.0], sound_speed=[1600.0, 1700.0],
                               density=[1.5, 1.6], attenuation=[0.5, 0.4],
                               roughness=[0.1, 0.4])
    assert b.halfspace_roughness.tolist() == [
        pytest.approx(0.1), pytest.approx(0.4)]


def test_from_halfspaces_roughness_default_zero():
    b = Bottom.from_halfspaces([0.0, 5000.0], sound_speed=[1600.0, 1700.0],
                               density=[1.5, 1.6], attenuation=[0.5, 0.4])
    assert all(c.halfspace.roughness == 0.0 for c in b.columns)


def test_from_halfspaces_roughness_length_mismatch():
    with pytest.raises(ConfigurationError, match='roughness'):
        Bottom.from_halfspaces([0.0, 5000.0], sound_speed=[1600.0, 1700.0],
                               density=[1.5, 1.6], attenuation=[0.5, 0.4],
                               roughness=[0.1, 0.2, 0.3])


def test_from_halfspaces_single_break_keeps_ranges():
    """A single range break is still a range coordinate (it feeds
    ``env.max_range``), even though one column is not range-*dependent*."""
    b = Bottom.from_halfspaces([500.0], sound_speed=1600.0, density=1.8,
                               attenuation=0.5)
    assert b.ranges is not None
    assert b.ranges.tolist() == [500.0]
    assert not b.is_range_dependent


# ─── collapse over a parameter-free half-space ─────────────────────────────

@pytest.mark.parametrize('kind', ['vacuum', 'rigid'])
@pytest.mark.parametrize('method', ['halfspace', 'top_layer', 'volume_average'])
def test_collapse_over_parameter_free_halfspace(kind, method):
    """A layered column over a ``'vacuum'``/``'rigid'`` half-space collapses
    back to that parameter-free type instead of raising.

    Those types carry no acoustic parameters, so handing the reduced cp/rho/a
    to ``BoundaryProperties`` trips its explicit-conflict guard. Only the
    interfacial ``roughness`` survives.
    """
    col = SeabedColumn(
        layers=_layers(),
        halfspace=BoundaryProperties(acoustic_type=kind, roughness=1.5))
    out = col.collapse(method)
    assert out.acoustic_type == kind
    assert out.roughness == pytest.approx(1.5)


def test_collapse_over_file_halfspace_keeps_reflection_file():
    """A ``'file'`` half-space keeps its ``reflection_file`` — the type is
    meaningless without it."""
    col = SeabedColumn(
        layers=_layers(),
        halfspace=BoundaryProperties(acoustic_type='file',
                                     reflection_file='bottom.brc'))
    for method in ('halfspace', 'top_layer', 'volume_average'):
        out = col.collapse(method)
        assert out.acoustic_type == 'file'
        assert out.reflection_file == 'bottom.brc'


def test_each_collapse_mode_reduces_the_column_to_its_own_speed():
    """Regression guard on the numeric reduction itself: thickness-weighted
    mean over 1600(10 m), 1700(20 m) and the half-space at the deepest layer's
    weight (20 m) = 1720 m/s."""
    col = SeabedColumn(layers=_layers(), halfspace=_hs(cp=1800.0))
    assert col.collapse('top_layer').sound_speed == pytest.approx(1600.0)
    assert col.collapse('volume_average').sound_speed == pytest.approx(1720.0)
    assert col.collapse('halfspace').sound_speed == pytest.approx(1800.0)


@pytest.mark.parametrize('layers', [[], _layers()])
def test_collapse_rejects_unknown_method_in_both_branches(layers):
    """The method name is validated once on entry, so the no-layer shortcut
    rejects a typo just like the layered path."""
    col = SeabedColumn(layers=layers, halfspace=_hs())
    with pytest.raises(ConfigurationError, match='unknown method'):
        col.collapse('bogus')


def test_at_depth_carries_layer_name():
    col = SeabedColumn(
        layers=[SedimentLayer.from_preset('sand', thickness=10.0)],
        halfspace=_hs())
    assert col.at(depth=5.0).name == 'sand'


def test_at_depth_carries_layer_roughness():
    """``SedimentLayer.roughness`` and ``BoundaryProperties.roughness`` both
    describe the interface at the top of their material, so the step lookup
    hands the layer's own number back rather than a default 0."""
    col = SeabedColumn(
        layers=[SedimentLayer(thickness=10, sound_speed=1600, density=1.5,
                              attenuation=0.4, roughness=0.35)],
        halfspace=_hs())
    col.halfspace.roughness = 0.1
    assert col.at(depth=5.0).roughness == pytest.approx(0.35)
    assert col.at(depth=50.0).roughness == pytest.approx(0.1)


def test_collapse_reports_the_seabed_surface_roughness():
    """The counterpart of the line above: a reduction over the whole stack has
    only one interface to report, so it stays on the half-space's number."""
    col = SeabedColumn(
        layers=[SedimentLayer(thickness=10, sound_speed=1600, density=1.5,
                              attenuation=0.4, roughness=0.35)],
        halfspace=_hs())
    col.halfspace.roughness = 0.1
    assert col.collapse('volume_average').roughness == pytest.approx(0.1)


class TestAccessorsReturnCopies:
    """Every `Bottom` / `SeabedColumn` accessor hands back a copy, so a caller
    never has to know whether a result is safe to mutate. ``halfspace_at``
    already deep-copied; ``at`` and ``isel`` handed out the stored object."""

    def _bottom(self):
        return Bottom.from_column(SeabedColumn(layers=_layers(), halfspace=_hs()))

    def test_bottom_at_result_is_not_the_stored_column(self):
        b = self._bottom()
        got = b.at(range=0.0)
        got.halfspace.sound_speed = 9999.0
        assert b.columns[0].halfspace.sound_speed == pytest.approx(1800.0)

    def test_bottom_isel_result_is_not_the_stored_column(self):
        b = self._bottom()
        got = b.isel(range=0)
        got.layers[0].sound_speed = 9999.0
        assert b.columns[0].layers[0].sound_speed == pytest.approx(1600.0)

    def test_seabedcolumn_isel_result_is_not_the_stored_layer(self):
        col = SeabedColumn(layers=_layers(), halfspace=_hs())
        got = col.isel(layer=0)
        got.density = 9.9
        assert col.layers[0].density == pytest.approx(1.5)

    def test_the_copies_carry_the_stored_values(self):
        b = self._bottom()
        assert b.at(range=0.0).halfspace.sound_speed == pytest.approx(1800.0)
        assert b.isel(range=0).layers[1].sound_speed == pytest.approx(1700.0)
        assert b.columns[0].isel(layer=1).thickness == pytest.approx(20.0)


class TestReductionsReturnCopies:
    """The reductions that *pick* a column — ``select_range('r0'/'rmax')``,
    ``'median'`` on a layered bottom, and the one-column-in/one-column-out
    early return — put that column straight into the new ``Bottom``, so the
    result shared its ``SeabedColumn`` with the parent and a write through
    the reduction edited the carrier it came from. The averaging methods
    build a fresh half-space and never had the problem. Same contract as
    ``at`` / ``isel`` / ``halfspace_at`` above."""

    @staticmethod
    def _rd(layered=False):
        return Bottom.from_columns(
            [SeabedColumn(layers=_layers() if layered else [],
                          halfspace=_hs(cp=cp))
             for cp in (1600.0, 1700.0, 1800.0)],
            ranges=[0.0, 1000.0, 2000.0])

    @pytest.mark.parametrize('method, index', [('r0', 0), ('rmax', -1)])
    def test_a_picked_column_does_not_write_back(self, method, index):
        b = self._rd()
        b.select_range(method).columns[0].halfspace.sound_speed = 9999.0
        assert b.columns[index].halfspace.sound_speed == pytest.approx(
            1600.0 if index == 0 else 1800.0)

    def test_a_layered_median_does_not_write_back(self):
        b = self._rd(layered=True)
        b.select_range('median').columns[0].layers[0].sound_speed = 9999.0
        assert b.columns[1].layers[0].sound_speed == pytest.approx(1600.0)

    def test_collapse_on_the_range_axis_does_not_write_back(self):
        b = self._rd()
        b.collapse(range='rmax').columns[0].halfspace.density = 42.0
        assert b.columns[-1].halfspace.density == pytest.approx(1.9)

    def test_a_range_independent_bottom_does_not_write_back(self):
        b = Bottom.from_column(SeabedColumn(layers=[], halfspace=_hs()))
        b.select_range('r0').columns[0].halfspace.attenuation = 7.0
        assert b.columns[0].halfspace.attenuation == pytest.approx(0.3)

    def test_to_halfspace_does_not_write_back(self):
        b = self._rd()
        b.to_halfspace('r0').sound_speed = 9999.0
        assert b.columns[0].halfspace.sound_speed == pytest.approx(1600.0)

    @pytest.mark.parametrize('method, expected',
                             [('r0', 1600.0), ('rmax', 1800.0),
                              ('mean', 1700.0), ('median', 1700.0)])
    def test_the_copies_carry_the_reduced_values(self, method, expected):
        got = self._rd().select_range(method).columns[0].halfspace
        assert got.sound_speed == pytest.approx(expected)


class TestSingleNodeRangesSurvivesReduction:
    """``ranges=[r]`` is a coordinate at range r, not a range-*dependent*
    axis: ``from_halfspaces`` keeps it and ``env.max_range`` reads it. A
    reduction that leaves the column count alone must therefore leave it
    alone too — dropping it moved ``env.max_range`` from 5000 m to 0."""

    def _bottom(self):
        return Bottom.from_halfspaces([5000.0], sound_speed=1700.0,
                                      density=1.8, attenuation=0.5)

    @pytest.mark.parametrize('method', ['r0', 'rmax', 'mean', 'median'])
    def test_select_range_keeps_the_single_node(self, method):
        assert self._bottom().select_range(method).ranges.tolist() == [5000.0]

    def test_collapse_layers_keeps_the_single_node(self):
        out = self._bottom().collapse(layers='halfspace')
        assert out.ranges.tolist() == [5000.0]

    def test_env_max_range_survives(self):
        env = uacpy.Environment(name='t', bathymetry=100.0,
                                bottom=self._bottom().select_range('r0'))
        assert env.max_range == pytest.approx(5000.0)

    def test_a_real_reduction_drops_the_axis(self):
        """Two columns down to one: the range axis no longer describes the
        result, so it goes."""
        b = Bottom.from_halfspaces([0.0, 5000.0], sound_speed=[1700.0, 1800.0],
                                   density=1.8, attenuation=0.5)
        assert b.select_range('r0').ranges is None
        assert b.select_range('mean').ranges is None

    def test_a_range_independent_bottom_stays_range_independent(self):
        b = Bottom.from_halfspace(_hs())
        assert b.select_range('r0').ranges is None
        assert b.collapse(layers='halfspace').ranges is None


def test_halfspace_at_rejects_bad_interp_on_range_independent_bottom():
    """The ``interp`` guard runs before the range-independent shortcut, so a
    typo cannot be silently accepted."""
    b = Bottom.from_halfspace(BoundaryProperties(sound_speed=1700.0))
    with pytest.raises(ConfigurationError, match='interp'):
        b.halfspace_at(range=0.0, interp='bogus')


class TestBoundaryPropertiesRoughnessSign:
    """``roughness`` is an RMS magnitude, so it takes the same sign check as
    the acoustic parameters beside it.

    Its exclusion from the ``half_space_offenders`` list is about
    ``acoustic_type`` inference — roughness is an interface property every
    boundary type carries, so it must not make a vacuum surface infer
    ``'half-space'``. That is not a reason to accept a negative value: OASES
    reads the sign of column 7 as an encoding, and ``rough(m).lt.-1e-10``
    makes INENVI re-read the layer record as nine tokens
    (``oases/src/oaseun31.f:72-93``) where uacpy emits eight, shifting every
    later READ in the deck. ``SedimentLayer`` has always checked it.
    """

    def test_negative_roughness_is_rejected(self):
        with pytest.raises(ConfigurationError,
                           match='roughness must be non-negative'):
            BoundaryProperties(roughness=-0.5)

    def test_zero_and_positive_roughness_are_accepted(self):
        assert BoundaryProperties(roughness=0.0).roughness == 0.0
        assert BoundaryProperties(roughness=2.5).roughness == 2.5

    def test_roughness_does_not_infer_a_half_space(self):
        """The sign check must not turn roughness into a half-space
        parameter — a rough pressure-release sea surface stays a vacuum."""
        assert BoundaryProperties(roughness=2.0).acoustic_type == 'vacuum'

    def test_roughness_may_accompany_an_explicit_vacuum(self):
        """The explicit-conflict guard lists only cp/rho/alpha/cs, so a rough
        vacuum surface is still constructible."""
        assert BoundaryProperties(acoustic_type='vacuum',
                                  roughness=2.0).roughness == 2.0

    def test_sediment_layer_agrees(self):
        with pytest.raises(ConfigurationError,
                           match='roughness must be non-negative'):
            SedimentLayer(thickness=10, sound_speed=1600, density=1.5,
                          roughness=-1.0)


# ─── mixed-type blending guards (2026-08 audit) ────────────────────────────

def test_halfspace_at_mixed_types_defaults_to_nearest():
    """Mixed-type columns must not blend: the default reads the nearest
    column intact (its own type and values), and an explicit
    interp='linear' is refused instead of averaging placeholders."""
    bot = Bottom(
        columns=[
            SeabedColumn(layers=[], halfspace=BoundaryProperties(
                acoustic_type='vacuum')),
            SeabedColumn(layers=[], halfspace=BoundaryProperties(
                sound_speed=1800.0, density=1.9, attenuation=0.8)),
        ],
        ranges=[0.0, 10_000.0])
    far = bot.halfspace_at(range=10_000.0)
    assert far.acoustic_type == 'half-space'
    assert far.sound_speed == pytest.approx(1800.0)
    near = bot.halfspace_at(range=0.0)
    assert near.acoustic_type == 'vacuum'
    with pytest.raises(ConfigurationError, match="linear"):
        bot.halfspace_at(range=5_000.0, interp='linear')


def test_select_range_mean_refuses_mixed_types():
    """Averaging across boundary types would fold construction placeholders
    into the numbers — same guard Surface.collapse applies."""
    bot = Bottom(
        columns=[
            SeabedColumn(layers=[], halfspace=BoundaryProperties(
                acoustic_type='vacuum')),
            SeabedColumn(layers=[], halfspace=BoundaryProperties(
                sound_speed=1800.0, density=1.9, attenuation=0.8)),
        ],
        ranges=[0.0, 10_000.0])
    for method in ('mean', 'median'):
        with pytest.raises(ConfigurationError, match="single boundary type"):
            bot.select_range(method)


def test_all_sound_speeds_skips_precalc_placeholder():
    """A 'precalc' half-space carries the resolved 1600 m/s placeholder, not
    a seabed speed; aggregate speeds must skip it like vacuum/rigid/file so
    it never feeds a c0 or phase-speed bound."""
    col = SeabedColumn(
        [SedimentLayer(thickness=10, sound_speed=1550, density=1.6,
                       attenuation=0.4)],
        BoundaryProperties(acoustic_type='precalc',
                           reflection_file='bot.irc'))
    assert Bottom.from_column(col).all_sound_speeds() == [1550.0]
    bare = SeabedColumn([], BoundaryProperties(acoustic_type='precalc',
                                               reflection_file='bot.irc'))
    assert Bottom.from_column(bare).all_sound_speeds() == []


class TestSelectRangeFileColumns:
    """'mean'/'median' cannot numerically average reflection-table columns:
    a uniform axis sharing one table collapses to that table (only the
    roughness, the one genuine number such nodes carry, is reduced), and
    differing tables are refused instead of silently dropping one."""

    @staticmethod
    def _file_col(path, roughness=0.0):
        return SeabedColumn([], BoundaryProperties(
            acoustic_type='file', reflection_file=path, roughness=roughness))

    @pytest.mark.parametrize('method', ['mean', 'median'])
    def test_shared_spec_collapses_to_it(self, method):
        b = Bottom.from_columns(
            [self._file_col('bot.brc', 0.1), self._file_col('bot.brc', 0.3)],
            ranges=[0.0, 5000.0])
        out = b.select_range(method).columns[0].halfspace
        assert out.acoustic_type == 'file'
        assert out.reflection_file == 'bot.brc'
        assert out.roughness == pytest.approx(0.2)

    @pytest.mark.parametrize('method', ['mean', 'median'])
    def test_differing_specs_raise(self, method):
        b = Bottom.from_columns(
            [self._file_col('a.brc'), self._file_col('b.brc')],
            ranges=[0.0, 5000.0])
        with pytest.raises(ConfigurationError, match='reflection files'):
            b.select_range(method)


class TestBottomAndSurfaceShareOneNearestRule:
    """``Bottom`` and ``Surface`` select with the same nearest rule.

    Neither carrier can blend its entries, so both answer ``at`` by landing on
    a stored index — the rule lives once in :mod:`uacpy.core._grid` and each
    carrier's ``_nearest_index`` delegates to it. A private copy in either
    module is what this catches: two copies drift, and the drift shows up as
    one carrier accepting a label the other refuses.
    """

    def test_both_modules_bind_the_one_grid_helper(self):
        from uacpy.core import _grid
        from uacpy.core import bottom as bottom_module
        from uacpy.core import surface as surface_module
        assert (bottom_module._nearest_index_on_axis
                is _grid._nearest_index_on_axis)
        assert (surface_module._nearest_index_on_axis
                is _grid._nearest_index_on_axis)

    @pytest.mark.parametrize('bad', [float('nan'), float('inf'),
                                     float('-inf')])
    def test_both_carriers_refuse_the_same_non_finite_label(self, bad):
        from uacpy.core.surface import Surface
        bottom = Bottom.from_columns(
            [SeabedColumn(_layers(), _hs()), SeabedColumn(_layers(), _hs())],
            ranges=[0.0, 5000.0])
        surface = Surface(
            properties=[BoundaryProperties(acoustic_type='vacuum'),
                        BoundaryProperties(acoustic_type='vacuum')],
            ranges=[0.0, 5000.0])
        with pytest.raises(ConfigurationError, match='finite label'):
            bottom.at(range=bad)
        with pytest.raises(ConfigurationError, match='finite label'):
            surface.at(range=bad)

    def test_column_index_at_and_at_share_the_bottom_validation_path(self):
        """``column_index_at`` is ``at``'s read-only counterpart, so it must
        refuse every label ``at`` refuses rather than answering index 0."""
        bottom = Bottom.from_columns(
            [SeabedColumn(_layers(), _hs()), SeabedColumn(_layers(), _hs())],
            ranges=[0.0, 5000.0])
        for bad in (float('nan'), float('inf')):
            with pytest.raises(ConfigurationError):
                bottom.column_index_at(range=bad)
            with pytest.raises(ConfigurationError):
                bottom.at(range=bad)
        assert bottom.columns[bottom.column_index_at(range=5000.0)] \
            is bottom.columns[1]


class TestSedimentLayerPresetCopiesRoughness:
    """``SedimentLayer.from_preset`` carries the preset's ``roughness``,
    matching ``BoundaryProperties.from_preset`` on every material, and an
    explicit override wins."""

    def test_every_preset_roughness_matches_the_materials_table(self):
        for name in list_materials():
            layer = SedimentLayer.from_preset(name, thickness=1.0)
            assert layer.roughness == pytest.approx(
                get_material(name)['roughness']), name

    def test_layer_and_boundary_presets_agree_on_roughness(self):
        for name in list_materials():
            layer = SedimentLayer.from_preset(name, thickness=1.0)
            boundary = BoundaryProperties.from_preset(name)
            assert layer.roughness == pytest.approx(boundary.roughness), name

    def test_roughness_override_wins_over_the_preset(self):
        layer = SedimentLayer.from_preset('sand', thickness=1.0,
                                          roughness=0.3)
        assert layer.roughness == pytest.approx(0.3)

    def test_a_nonzero_preset_roughness_reaches_the_layer(self, monkeypatch):
        # Every shipped preset carries roughness 0.0, so the parity tests
        # above pass whether or not from_preset copies the field; a patched
        # preset makes the copy observable.
        from uacpy.core import materials
        rough = dict(materials.MATERIALS['sand'], roughness=0.05)
        monkeypatch.setitem(materials.MATERIALS, 'sand', rough)
        layer = SedimentLayer.from_preset('sand', thickness=1.0)
        assert layer.roughness == pytest.approx(0.05)


class TestSedimentLayerCoercesNumericFieldsToFloat:
    """The numeric fields are ``float()``-coerced in ``__post_init__``
    (as ``BoundaryProperties`` does), so a convertible string is stored as
    a float and a non-numeric one is refused at construction — the
    validators check a converted copy, so an unconverted string used to
    pass them and crash later in ``__repr__``."""

    def test_numeric_strings_are_stored_as_floats(self):
        layer = SedimentLayer(thickness='10', sound_speed='1650',
                              density='1.9')
        assert isinstance(layer.sound_speed, float)
        assert layer.sound_speed == pytest.approx(1650.0)
        assert 'cp=1650' in repr(layer)

    def test_a_non_numeric_string_is_refused_at_construction(self):
        with pytest.raises(ValueError):
            SedimentLayer(thickness=10.0, sound_speed='fast', density=1.9)

    def test_numpy_scalars_construct(self):
        layer = SedimentLayer(thickness=np.float64(10.0),
                              sound_speed=np.float64(1650.0),
                              density=np.float64(1.9))
        assert isinstance(layer.density, float)


def _halfspace(cp):
    return BoundaryProperties(acoustic_type='half-space', sound_speed=cp,
                              density=1.5, attenuation=0.5)


def _range_dependent_bottom():
    return Bottom(columns=[SeabedColumn(layers=[], halfspace=_halfspace(1600.0)),
                           SeabedColumn(layers=[], halfspace=_halfspace(1800.0))],
                  ranges=[0.0, 5000.0])


class TestBoundaryCarrierLabelsMustBeFiniteScalars:
    """``Bottom``/``SeabedColumn``/``Surface`` pick a node with
    ``argmin(|axis - label|)``, which ranks nothing when every distance is
    NaN and hands back index 0 — a real column, so the caller sees a
    plausible answer to an unanswerable query."""

    def test_bottom_at_nan_range_raises_a_typed_label_error(self):
        with pytest.raises(ConfigurationError,
                           match="range=nan is not a finite label"):
            _range_dependent_bottom().at(range=np.nan)

    def test_bottom_at_inf_range_raises(self):
        with pytest.raises(ConfigurationError, match="not a finite label"):
            _range_dependent_bottom().at(range=np.inf)

    def test_bottom_at_array_range_raises_a_typed_scalar_label_error(self):
        with pytest.raises(ConfigurationError, match="not a scalar label"):
            _range_dependent_bottom().at(range=np.array([0.0, 5000.0]))

    def test_bottom_at_a_finite_range_returns_the_nearest_column(self):
        bottom = _range_dependent_bottom()
        assert bottom.at(range=4900.0).halfspace.sound_speed == 1800.0

    def test_a_range_independent_bottom_also_rejects_a_nan_range(self):
        bottom = Bottom(columns=[SeabedColumn(layers=[],
                                              halfspace=_halfspace(1600.0))])
        with pytest.raises(ConfigurationError, match="not a finite label"):
            bottom.at(range=np.nan)

    def test_halfspace_at_nan_range_blames_the_label_not_the_density(self):
        # The blend path never reaches ``_nearest_index``: np.interp carried
        # the NaN into every blended property and BoundaryProperties then
        # reported a non-finite density.
        with pytest.raises(ConfigurationError) as exc:
            _range_dependent_bottom().halfspace_at(range=np.nan,
                                                   interp='linear')
        assert "range=nan is not a finite label" in str(exc.value)
        assert "density must be finite" not in str(exc.value)

    def test_halfspace_at_nearest_nan_range_raises(self):
        with pytest.raises(ConfigurationError, match="not a finite label"):
            _range_dependent_bottom().halfspace_at(range=np.nan,
                                                   interp='nearest')

    def test_halfspace_at_a_finite_range_blends_the_two_columns(self):
        bp = _range_dependent_bottom().halfspace_at(range=2500.0,
                                                    interp='linear')
        assert bp.sound_speed == pytest.approx(1700.0)

    def test_seabed_column_at_nan_depth_raises_a_typed_label_error(self):
        col = SeabedColumn(
            layers=[SedimentLayer(thickness=10.0, sound_speed=1550.0,
                                  density=1.4, attenuation=0.3)],
            halfspace=_halfspace(1800.0))
        with pytest.raises(ConfigurationError,
                           match="depth=nan is not a finite label"):
            col.at(depth=np.nan)

    def test_seabed_column_at_a_finite_depth_returns_the_containing_layer(self):
        col = SeabedColumn(
            layers=[SedimentLayer(thickness=10.0, sound_speed=1550.0,
                                  density=1.4, attenuation=0.3)],
            halfspace=_halfspace(1800.0))
        assert col.at(depth=5.0).sound_speed == 1550.0
        assert col.at(depth=50.0).sound_speed == 1800.0

    def test_sample_at_depths_returns_the_enclosing_layers_speed(self):
        col = SeabedColumn(
            layers=[SedimentLayer(thickness=10.0, sound_speed=1550.0,
                                  density=1.4, attenuation=0.3)],
            halfspace=_halfspace(1800.0))
        cp, rho, attn = col.sample_at_depths(4)
        assert np.allclose(cp, 1550.0)

    def test_surface_at_nan_range_raises(self):
        surface = Surface(
            properties=[BoundaryProperties(acoustic_type='vacuum'),
                        BoundaryProperties(acoustic_type='rigid')],
            ranges=[0.0, 5000.0])
        with pytest.raises(ConfigurationError,
                           match="range=nan is not a finite label"):
            surface.at(range=np.nan)

    def test_surface_at_a_finite_range_returns_the_nearest_node(self):
        surface = Surface(
            properties=[BoundaryProperties(acoustic_type='vacuum'),
                        BoundaryProperties(acoustic_type='rigid')],
            ranges=[0.0, 5000.0])
        assert surface.at(range=4000.0).acoustic_type == 'rigid'


def _column(thickness, speed):
    return SeabedColumn(
        layers=[SedimentLayer(thickness=thickness, sound_speed=speed,
                              density=1.5, attenuation=0.5,
                              shear_speed=400.0, shear_attenuation=1.0),
                SedimentLayer(thickness=2.0 * thickness, sound_speed=speed + 50,
                              density=1.7, attenuation=0.6)],
        halfspace=BoundaryProperties(
            acoustic_type='half-space', sound_speed=1900.0, density=2.0,
            attenuation=0.1, shear_speed=600.0, shear_attenuation=0.5),
    )


def _wide_range_dependent_env(n_columns=24, n_bathy=97, r_end=60000.0):
    """A bottom with many columns under a wavy seafloor with many nodes.

    The two axes deliberately do not line up, so most bathymetry nodes fall
    between bottom columns and the nearest-column rule actually has to choose.
    """
    ranges = np.linspace(0.0, r_end, n_columns)
    bottom = Bottom(
        columns=[_column(10.0 + 5.0 * (i % 7), 1650.0 + 10.0 * (i % 11))
                 for i in range(n_columns)],
        ranges=ranges)
    r_bathy = np.linspace(0.0, r_end, n_bathy)
    z_bathy = 120.0 + 60.0 * np.sin(r_bathy / r_end * 6.0 * np.pi)
    return Environment(name='wide-rd',
                       bathymetry=list(zip(r_bathy.tolist(), z_bathy.tolist())),
                       ssp=1500.0, bottom=bottom)


def _range_independent_env():
    return Environment(name='ri', bathymetry=[(0.0, 60.0), (5000.0, 400.0)],
                       ssp=1500.0, bottom=_column(4.0, 1600.0))


class TestColumnIndexAt:
    """``Bottom.column_index_at`` names the column ``Bottom.at`` would copy."""

    def test_it_indexes_the_column_at_returns(self):
        env = _wide_range_dependent_env()
        bottom = env.bottom
        for r in np.linspace(0.0, 60000.0, 61):
            i = bottom.column_index_at(range=float(r))
            live = bottom.columns[i]
            copied = bottom.at(range=float(r))
            assert [la.thickness for la in live.layers] == \
                   [la.thickness for la in copied.layers]
            assert live.halfspace.sound_speed == copied.halfspace.sound_speed

    def test_it_returns_the_only_index_for_a_range_independent_bottom(self):
        bottom = _range_independent_env().bottom
        assert bottom.n_ranges == 1
        assert bottom.column_index_at(range=0.0) == 0
        assert bottom.column_index_at(range=1e9) == 0

    def test_it_hands_back_the_live_column_not_a_copy(self):
        # The whole point of the accessor: no deep copy stands between the
        # caller and the carrier, which is why the docstring calls it
        # read-only and points mutation at ``at``.
        bottom = _wide_range_dependent_env().bottom
        i = bottom.column_index_at(range=30000.0)
        assert bottom.columns[i] is bottom.columns[i]
        assert bottom.columns[i] is not bottom.at(range=30000.0)

    @pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
    def test_it_rejects_a_non_finite_range_the_way_at_does(self, bad):
        bottom = _wide_range_dependent_env().bottom
        with pytest.raises(ConfigurationError):
            bottom.column_index_at(range=bad)
        with pytest.raises(ConfigurationError):
            bottom.at(range=bad)

    def test_it_rejects_an_array_range_the_way_at_does(self):
        bottom = _wide_range_dependent_env().bottom
        with pytest.raises(ConfigurationError):
            bottom.column_index_at(range=np.array([0.0, 1.0]))


class TestAtAndIselReturnCopies:
    """What ``at`` and ``isel`` hand back is a copy: mutating the result
    cannot reach the carrier's own column. That is the contract making
    ``column_index_at`` a read-only counterpart rather than a replacement."""

    def test_mutating_an_at_result_leaves_the_carrier_alone(self):
        bottom = _wide_range_dependent_env().bottom
        before = copy.deepcopy(bottom.columns[3])
        got = bottom.at(range=float(bottom.ranges[3]))
        got.layers[0].thickness = 12345.0
        got.halfspace.sound_speed = 4321.0
        got.layers.append(SedimentLayer(thickness=1.0, sound_speed=1500.0,
                                        density=1.0, attenuation=0.0))
        assert bottom.columns[3].layers[0].thickness == \
            before.layers[0].thickness
        assert bottom.columns[3].halfspace.sound_speed == \
            before.halfspace.sound_speed
        assert len(bottom.columns[3].layers) == len(before.layers)

    def test_isel_hands_back_a_copy_too(self):
        bottom = _wide_range_dependent_env().bottom
        before = float(bottom.columns[2].layers[0].thickness)
        bottom.isel(range=2).layers[0].thickness = 999.0
        assert float(bottom.columns[2].layers[0].thickness) == before


class TestBottomDelegatedWritesReachTheHalfspaces:
    """``bottom.sound_speed = 1610`` used to land in ``__dict__`` where
    nothing reads it: ``halfspace_at`` — and every writer and model behind
    it — reads ``columns``, so the attribute echoed 1610 back while every
    engine kept the stored half-spaces. The write now follows
    ``Surface.__setattr__``'s contract: it is validated, propagates to
    every column's half-space, and warns on a multi-column bottom that the
    broadcast flattens range dependence."""

    def test_a_write_lands_where_halfspace_at_reads(self):
        b = Bottom.from_halfspace(_hs())
        b.sound_speed = 1610.0
        assert b.columns[0].halfspace.sound_speed == pytest.approx(1610.0)
        assert b.halfspace_at(range=0.0).sound_speed == pytest.approx(1610.0)
        assert 'sound_speed' not in vars(b)

    def test_a_multi_column_write_warns_and_reaches_every_column(self):
        b = Bottom.from_halfspaces([0.0, 5000.0],
                                   sound_speed=[1600.0, 1700.0],
                                   density=1.8, attenuation=0.4)
        with pytest.warns(UserWarning, match=r"sets all 2 range columns"):
            b.sound_speed = 1650.0
        assert [c.halfspace.sound_speed for c in b.columns] == [1650.0, 1650.0]
        assert b.halfspace_at(
            range=5000.0).sound_speed == pytest.approx(1650.0)

    def test_the_multi_column_warning_points_at_columns(self):
        b = Bottom.from_halfspaces([0.0, 5000.0],
                                   sound_speed=[1600.0, 1700.0],
                                   density=1.8, attenuation=0.4)
        with pytest.warns(UserWarning, match=r"\.columns\[i\]\.halfspace"):
            b.roughness = 0.3

    def test_a_single_column_write_is_silent(self):
        b = Bottom.from_halfspace(_hs())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            b.roughness = 0.5
        assert not [w for w in caught if issubclass(w.category, UserWarning)]
        assert b.halfspace_at(range=0.0).roughness == pytest.approx(0.5)

    def test_type_fields_are_not_assignable(self):
        b = Bottom.from_halfspace(_hs())
        for name, value in (('acoustic_type', 'rigid'),
                            ('reflection_file', 'bottom.brc')):
            with pytest.raises(ConfigurationError, match='cannot be assigned'):
                setattr(b, name, value)

    def test_numeric_rules_mirror_the_constructor(self):
        b = Bottom.from_halfspace(_hs())
        with pytest.raises(ConfigurationError, match='must be positive'):
            b.density = -3.0
        with pytest.raises(ConfigurationError, match='must be positive'):
            b.sound_speed = 0.0
        with pytest.raises(ConfigurationError, match='exceeds'):
            b.attenuation = 500.0
        with pytest.raises(ConfigurationError, match='non-negative'):
            b.roughness = -1.0
        assert b.columns[0].halfspace.sound_speed == pytest.approx(1800.0)
        assert b.columns[0].halfspace.density == pytest.approx(1.9)

    def test_grain_size_phi_keeps_its_sign_and_its_unset_value(self):
        # phi = -log2(d/mm) is signed (gravel is negative), so the
        # non-negative rule the other fields take must not reach it, and
        # None is the field's own unset value.
        b = Bottom.from_halfspace(_hs())
        b.grain_size_phi = -1.5
        assert b.halfspace_at(range=0.0).grain_size_phi == pytest.approx(-1.5)
        b.grain_size_phi = None
        assert b.halfspace_at(range=0.0).grain_size_phi is None

    def test_a_layered_bottom_refuses_the_flat_write(self):
        b = Bottom.from_column(SeabedColumn(layers=_layers(), halfspace=_hs()))
        with pytest.raises(ConfigurationError, match='layer'):
            b.sound_speed = 1610.0
        assert b.columns[0].halfspace.sound_speed == pytest.approx(1800.0)
        assert 'sound_speed' not in vars(b)

    def test_a_rigid_bottom_rejects_halfspace_params_but_takes_roughness(self):
        b = Bottom.from_halfspace(BoundaryProperties(acoustic_type='rigid'))
        with pytest.raises(ConfigurationError, match='rigid'):
            b.sound_speed = 1700.0
        b.roughness = 0.5
        assert b.halfspace_at(range=0.0).roughness == pytest.approx(0.5)


class TestSeabedColumnDelegatedWritesReachTheHalfspace:
    """The same shadow-write hazard one level down: ``at`` / ``collapse`` /
    ``sample_at_depths`` read ``layers`` / ``halfspace``, never a flat
    instance attribute, so ``column.sound_speed = …`` delegates to the
    half-space under the same rules as the ``Bottom`` write."""

    def test_a_pure_halfspace_column_write_reaches_its_halfspace(self):
        col = SeabedColumn(layers=[], halfspace=_hs())
        col.sound_speed = 1610.0
        assert col.halfspace.sound_speed == pytest.approx(1610.0)
        assert col.at(depth=0.0).sound_speed == pytest.approx(1610.0)
        assert 'sound_speed' not in vars(col)

    def test_a_layered_column_refuses_the_flat_write(self):
        col = SeabedColumn(layers=_layers(), halfspace=_hs())
        with pytest.raises(ConfigurationError, match='layer'):
            col.sound_speed = 1610.0
        assert col.halfspace.sound_speed == pytest.approx(1800.0)
        assert 'sound_speed' not in vars(col)


class TestGrainSizeMustBeAFiniteNumberEverywhere:
    """ϕ is signed metadata, so it has no sign rule — but every carrier must
    agree that it is a finite NUMBER: a NaN/inf/str ϕ stored on one carrier and
    refused by another reads back as a measurement on the first."""

    @pytest.mark.parametrize('bad', [float('nan'), float('inf'), 'abc'])
    def test_the_constructor_refuses_it(self, bad):
        from uacpy.core.bottom import BoundaryProperties
        with pytest.raises((ConfigurationError, ValueError)):
            BoundaryProperties(acoustic_type='half-space', sound_speed=1600.0,
                               grain_size_phi=bad)

    def test_the_shared_delegated_write_validator_refuses_it(self):
        # The one validator behind the SeabedColumn/Bottom/Surface writes.
        from uacpy.core.bottom import BoundaryProperties, _validate_boundary_write
        node = BoundaryProperties(acoustic_type='half-space', sound_speed=1600.0)
        with pytest.raises(ConfigurationError, match='grain_size_phi'):
            _validate_boundary_write('Bottom', 'grain_size_phi', float('nan'),
                                     [node])
        assert _validate_boundary_write('Bottom', 'grain_size_phi', 2.5,
                                        [node]) == 2.5
        assert _validate_boundary_write('Surface', 'grain_size_phi', None,
                                        [node]) is None

    def test_a_surface_node_refuses_it_at_construction(self):
        from uacpy.core.bottom import BoundaryProperties
        from uacpy.core.surface import Surface
        with pytest.raises(ConfigurationError, match='grain_size_phi'):
            Surface(properties=[BoundaryProperties(
                acoustic_type='half-space', sound_speed=1600.0,
                grain_size_phi=float('inf'))])
