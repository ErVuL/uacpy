"""Unit tests for the unified ``Bottom`` / ``SeabedColumn`` carrier.

Self-contained: expected values are hand-computed golden numbers (no coupling
to any other carrier type).
"""

import pytest

import uacpy
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


def test_collapse_halfspace_values_unchanged():
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

    def test_roughness_still_does_not_infer_a_half_space(self):
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
