"""``Environment`` and the input carriers it holds.

Construction and coercion of ``Environment``, ``Bathymetry``, ``Bottom``,
``Surface`` and ``BoundaryProperties``: which spellings each ``ssp=``,
``bathymetry=`` and ``bottom=`` accepts, what a reassigned carrier becomes,
``copy()`` and geolocation, and the refusals every core carrier shares
(complex coordinates, non-finite queries, bool scalars).

A recurring subject is the label query. ``at()`` and its siblings pick a
node with ``argmin(|axis - label|)``, which ranks nothing when every
distance is NaN and hands back index 0, a real node, so the caller sees a
plausible answer to an unanswerable question. Those guards are pinned on
both sides: the label that must be refused, and the legitimate one next to
it that must still work.
"""

import dataclasses
import numpy as np
import pytest
import re
import tempfile
import uacpy
import warnings
from pathlib import Path
from uacpy.core import BoundaryProperties
from uacpy.core import Environment
from uacpy.tests.conftest import water_density_env
from uacpy.core.altimetry import Altimetry
from uacpy.core.bathymetry import Bathymetry
from uacpy.core.bottom import Bottom
from uacpy.core.bottom import SeabedColumn
from uacpy.io.at_codes import AttenuationUnits
from uacpy.core.boundary import BoundaryType
from uacpy.core.constants import DEFAULT_WATER_DENSITY_G_CM3
from uacpy.core.environment import SoundSpeedProfile
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import Field
from uacpy.core.source import Source
from uacpy.core.surface import Surface
from uacpy.models.kraken._segments import segment_environment_by_range
from uacpy.models.kraken import _checks


class TestEnvironment:
    """Tests for Environment class."""

    def test_create_simple_environment(self, simple_env):
        """Test creating a simple isovelocity environment."""
        assert simple_env.name == "Test Environment"
        assert simple_env.depth == 100.0
        assert float(simple_env.ssp.sound_speed[0, 0]) == 1500.0
        assert simple_env.ssp.kind == 'isovelocity'
        assert not simple_env.is_range_dependent

    def test_create_parabolic_ssp_environment(self, parabolic_ssp_env):
        """Construction of a 100-m parabolic-SSP env."""
        assert parabolic_ssp_env.name == "Parabolic SSP"
        assert parabolic_ssp_env.depth == 100.0
        assert parabolic_ssp_env.ssp.n_depths == 21
        assert parabolic_ssp_env.ssp.kind == 'measured'

    def test_create_munk_environment(self, munk_env):
        """Construction of a deep-water Munk env using from_munk()."""
        assert munk_env.name == "Munk Profile"
        assert munk_env.depth == 5000.0
        assert munk_env.ssp.kind == 'munk'

    def test_range_dependent_environment(self, range_dependent_env):
        """Test range-dependent environment."""
        assert range_dependent_env.is_range_dependent
        assert range_dependent_env.bathymetry.n_ranges == 11
        assert range_dependent_env.bathymetry.depths[0] == 80.0
        assert range_dependent_env.bathymetry.depths[-1] == 120.0

    def test_max_range(self, simple_env, range_dependent_env):
        """range_max is the range extent (0 when range-independent), symmetric
        with depth and matching the bathymetry range axis."""
        assert simple_env.range_max == 0.0
        assert range_dependent_env.range_max == pytest.approx(
            float(range_dependent_env.bathymetry.ranges.max()))

    def test_max_range_covers_single_node_ranged_carriers(self):
        """A carrier whose ranged axis holds one node still marks a range
        coordinate, even though it is not range-*dependent* (that is a
        more-than-one-node test); range_max must include it."""
        from uacpy.core.environment import (
            Bottom, SeabedColumn, BoundaryProperties, SoundSpeedProfile,
            Surface,
        )
        surf = Surface(nodes=[BoundaryProperties(acoustic_type='vacuum')],
                       ranges=[5000.0])
        assert uacpy.Environment(bathymetry=100.0,
                                 surface=surf).range_max == 5000.0
        ssp = SoundSpeedProfile(depths=[0.0, 100.0],
                                sound_speed=[[1500.0], [1500.0]], ranges=[7000.0])
        assert uacpy.Environment(bathymetry=100.0,
                                 ssp=ssp).range_max == 7000.0
        bot = Bottom(columns=[SeabedColumn(
            layers=[], halfspace=BoundaryProperties(sound_speed=1700.0))],
            ranges=[9000.0])
        assert uacpy.Environment(bathymetry=100.0,
                                 bottom=bot).range_max == 9000.0

    def test_ssp_pairs_shape(self, simple_env, parabolic_ssp_env):
        """SSP pairs view always has shape (N, 2)."""
        assert simple_env.ssp.to_pairs().shape[1] == 2
        assert parabolic_ssp_env.ssp.to_pairs().shape[1] == 2

    def test_bathymetry_collapse_range(self, range_dependent_env):
        """Median depth of the fixture's 11-node 80..120 m linspace is its
        middle node — exactly 100.0."""
        assert range_dependent_env.bathymetry.collapse_range('median') == 100.0
        bathymetry = range_dependent_env.bathymetry
        assert bathymetry.collapse_range() == float(bathymetry.depths.max())
        assert bathymetry.collapse_range('initial') == float(bathymetry.depths[0])
        with pytest.raises(ConfigurationError, match='Bathymetry.collapse_range'):
            bathymetry.collapse_range('deepest')

    def test_depth_is_read_only(self, simple_env):
        """``env.depth`` is derived from bathymetry (a getter-only property);
        assigning to it raises rather than silently shadowing the
        bathymetry."""
        with pytest.raises(
                AttributeError,
                match="property 'depth' of 'Environment' object has no setter"):
            simple_env.depth = 200.0

    def test_invalid_depth(self):
        """Test that negative depth raises error."""
        with pytest.raises(ConfigurationError,
                           match='Bathymetry depths must be positive'):
            uacpy.Environment(name="Test", bathymetry=-10, ssp=1500)

    def test_bathymetry_rejects_negative_range(self):
        """Bathymetry ranges are measured from the source; they cannot
        be negative."""
        with pytest.raises(ConfigurationError, match="ranges must be non-negative"):
            uacpy.Environment(
                name="Test",
                bathymetry=[[-100.0, 80.0], [5000.0, 90.0]],
                ssp=1500,
            )


class TestEnvironmentWaterDensity:
    """``Environment.water_density``: sea water by default (1.027 g/cm³),
    an explicit value kept through copies, range segmentation and
    Kraken's single-profile reduction; the wrong unit refused."""

    def test_default_is_sea_water_not_one(self):
        assert DEFAULT_WATER_DENSITY_G_CM3 == 1.027
        assert water_density_env().water_density == 1.027

    def test_an_explicit_value_is_kept(self):
        assert water_density_env(water_density=1.0).water_density == 1.0
        assert water_density_env(water_density=1.03).water_density == 1.03

    def test_kg_per_m3_is_refused_with_the_unit_named(self):
        with pytest.raises(ConfigurationError, match='g/cm³.*kg/m³'):
            water_density_env(water_density=1027.0)

    def test_a_non_number_is_refused(self):
        with pytest.raises(ConfigurationError, match='number in g/cm³'):
            water_density_env(water_density='sea water')

    def test_copy_keeps_it(self):
        assert water_density_env(water_density=1.02).copy().water_density == 1.02

    def test_range_segmentation_keeps_it(self):
        env = water_density_env(bathymetry=[(0.0, 100.0), (5000.0, 200.0)],
                   water_density=1.02)
        for _, seg in segment_environment_by_range(env, n_segments=3):
            assert seg.water_density == 1.02

    @pytest.mark.requires_binary  # constructs Kraken (resolves its binary)
    def test_krakens_single_profile_reduction_keeps_it(self):
        from uacpy.models import Kraken
        env = water_density_env(bathymetry=[(0.0, 100.0), (5000.0, 200.0)],
                   water_density=1.02)
        model = Kraken(verbose=False)
        assert _checks.modes_single_profile(
            env, collapse=model._collapse,
            model_name=model.model_name).water_density \
            == 1.02


@pytest.mark.requires_binary  # constructs models (resolves their binaries)
class TestCopyAndGeolocation:
    """`.copy()` is universal across carriers + results; Environment carries
    optional geolocation/date provenance that survives copy."""

    def test_copy_symmetry_across_carriers_and_io(self):
        import uacpy
        from uacpy import (Bathymetry, Altimetry, Surface, Bottom,
                           SoundSpeedProfile, BoundaryProperties)
        from uacpy.core.boundary import SedimentLayer
        from uacpy.core.bottom import SeabedColumn
        objs = [
            Bathymetry(ranges=[0, 1000.], depths=[100, 90.]),
            Altimetry(ranges=[0, 1000.], heights=[0, -2.]),
            Surface.coerce(BoundaryProperties(acoustic_type='vacuum')),
            Bottom.from_halfspace(BoundaryProperties()),
            SeabedColumn(layers=[], halfspace=BoundaryProperties()),
            SedimentLayer(thickness=5.0, sound_speed=1600.0, density=1.8,
                          attenuation=0.2),
            BoundaryProperties(),
            SoundSpeedProfile.from_pairs([(0, 1500), (100, 1490.)]),
            uacpy.Source(depths=50., frequencies=120.),
            uacpy.Receiver(depths=[100.], ranges=[2000.]),
            uacpy.Environment(bathymetry=200., ssp=1500.),
        ]
        for o in objs:
            c = o.copy()
            assert type(c) is type(o) and c is not o
            # Deep copy: no top-level array is shared, so mutating the
            # copy can never reach the original.
            for name, val in vars(o).items():
                if isinstance(val, np.ndarray):
                    assert not np.shares_memory(getattr(c, name), val), (
                        type(o).__name__, name)

    def test_every_carrier_docs_call_a_carrier_has_copy(self):
        """``copy()``'s docstring says "symmetric with the other carriers" at
        nine sites, and docs/DEV.md section 5 names the carrier set. Two of
        the classes it lists — both components of ``SeabedColumn``, which has
        ``copy()`` — reached that claim without the method."""
        import uacpy
        from uacpy.core.boundary import BoundaryProperties, SedimentLayer
        from uacpy.core.bottom import Bottom, SeabedColumn
        from uacpy.core.surface import Surface
        carriers = (uacpy.SoundSpeedProfile, uacpy.Bathymetry,
                    uacpy.Altimetry, SedimentLayer, SeabedColumn, Bottom,
                    BoundaryProperties, Surface, uacpy.Source,
                    uacpy.Receiver, uacpy.Environment)
        missing = sorted(c.__name__ for c in carriers
                         if not callable(getattr(c, 'copy', None)))
        assert not missing, (
            f"carrier(s) {missing} are named in docs/DEV.md section 5 but "
            f"have no copy()")

    def test_the_bottom_components_copy_deeply(self):
        """The two additions, driven: a mutation of the copy must not reach
        the original, which is the whole content of "deep"."""
        from uacpy.core.boundary import BoundaryProperties, SedimentLayer
        layer = SedimentLayer(thickness=5.0, sound_speed=1600.0, density=1.8,
                              attenuation=0.2)
        clone = layer.copy()
        assert clone is not layer and clone == layer
        clone.sound_speed = 1900.0
        assert layer.sound_speed == 1600.0

        props = BoundaryProperties(sound_speed=1700.0, density=1.9)
        twin = props.copy()
        assert twin is not props and twin == props
        twin.density = 2.5
        assert props.density == 1.9

    def test_source_and_receiver_copy_the_whole_attribute_surface(self):
        """``Source`` / ``Receiver`` deep-copy like every other carrier, so an
        attribute added to either is carried across without editing
        ``copy()``, and no array is shared with the original."""
        import uacpy
        objs = [
            uacpy.Source(depths=[10., 50.], frequencies=[100., 200.],
                         source_type='line',
                         beam_pattern=np.array([[-90., -20.], [90., 0.]])),
            uacpy.Receiver(depths=[100., 200.], ranges=[1000., 2000.]),
        ]
        for o in objs:
            c = o.copy()
            assert vars(c).keys() == vars(o).keys(), type(o).__name__
            for name, val in vars(o).items():
                new = getattr(c, name)
                if isinstance(val, np.ndarray):
                    np.testing.assert_array_equal(new, val)
                    assert not np.shares_memory(new, val), name
                else:
                    assert new == val, name

    def test_result_copy_is_independent(self):
        import uacpy
        env = uacpy.Environment(bathymetry=300., ssp=1500.)
        f = uacpy.Bellhop().compute_tl(
            env, uacpy.Source(depths=50., frequencies=150.),
            uacpy.Receiver(depths=[100., 200.], ranges=[2000., 4000.]))
        c = f.copy()
        assert type(c) is type(f) and c is not f
        # Mutating the copy's payload must not reach the original.
        assert not np.shares_memory(c.data, f.data)
        baseline = f.data.copy()
        c.data[...] = 999.0
        np.testing.assert_array_equal(f.data, baseline)

    def test_environment_geolocation_and_date(self):
        import datetime
        import uacpy
        e = uacpy.Environment(bathymetry=200., ssp=1500.,
                              location=(75., 12.5), date='2026-03-15')
        assert e.location == (75., 12.5)
        assert e.date == datetime.date(2026, 3, 15)
        assert e.transect is None
        # transect → location defaults to the midpoint; explicit overrides it
        t = uacpy.Environment(bathymetry=[(0, 200), (10000, 300)], ssp=1500.,
                              transect=((75., 0.), (77., 40.)))
        # the midpoint of the great-circle path, not the mean of the ends
        assert t.location == pytest.approx((76.8098, 18.5404), abs=1e-4)
        o = uacpy.Environment(bathymetry=200., ssp=1500., location=(75., 0.),
                              transect=((75., 0.), (77., 40.)))
        assert o.location == (75., 0.)
        # survives a deep copy
        c = t.copy()
        assert c.location == t.location
        assert c.transect == ((75., 0.), (77., 40.))
        # hand-built env carries none of it
        plain = uacpy.Environment(bathymetry=100., ssp=1500.)
        assert plain.location is None and plain.transect is None and plain.date is None

    @pytest.mark.parametrize("bad", [(91.0, 0.0), ('a', 'b'), 'not-a-date'])
    def test_environment_geolocation_typed_errors(self, bad):
        import uacpy
        with pytest.raises(ConfigurationError,
                           match='Environment: (date|location)'):
            if isinstance(bad, str):
                uacpy.Environment(bathymetry=100., ssp=1500., date=bad)
            else:
                uacpy.Environment(bathymetry=100., ssp=1500., location=bad)


class TestCarrierPredicatesAreProperties:
    """The ``is_*`` predicates the models and writers branch on must be
    properties, not methods.

    As bound methods they are always truthy, so ``if env.bottom.is_layered:``
    silently takes the True branch for every environment.
    """

    _NAMES = (
        ('bathymetry', 'is_range_dependent'), ('bathymetry', 'varies_with_range'),
        ('ssp', 'is_range_dependent'),
        ('bottom', 'is_range_dependent'), ('bottom', 'is_layered'),
        ('bottom', 'is_elastic'),
        ('surface', 'is_range_dependent'), ('surface', 'is_elastic'),
        (None, 'is_range_dependent'),
    )

    @pytest.mark.parametrize('carrier,name', _NAMES)
    def test_is_a_property(self, carrier, name):
        import inspect
        env = uacpy.Environment(bathymetry=100.0, ssp=1500.0)
        owner = type(env if carrier is None else getattr(env, carrier))
        found = inspect.getattr_static(owner, name)
        assert isinstance(found, property), (
            f"{owner.__name__}.{name} is a {type(found).__name__}; "
            f"as a method it is always truthy in a boolean test")

    def test_flat_environment_reports_false_not_truthy_method(self):
        env = uacpy.Environment(bathymetry=100.0, ssp=1500.0)
        for carrier, name in self._NAMES:
            owner = env if carrier is None else getattr(env, carrier)
            assert getattr(owner, name) is False, f"{carrier}.{name} should be False"


class TestZeroDimensionalArraysCoerceLikeTheirScalars:
    """A 0-d ndarray is a scalar in every respect except ``isinstance``, so
    the carriers have to see through it the way ``ssp=`` already does:
    ``np.array(True)`` is the bool a depth guard exists to refuse, and
    ``np.array(1600.0)`` is the scalar sound speed ``bottom=`` documents
    itself as taking."""

    def test_a_zero_dimensional_bool_is_refused_as_a_depth(self):
        for spelling in (True, np.bool_(True), np.array(True)):
            with pytest.raises(ConfigurationError, match='is a bool'):
                Bathymetry.coerce(spelling)

    def test_a_zero_dimensional_bool_is_refused_through_the_environment(self):
        with pytest.raises(ConfigurationError, match='is a bool'):
            Environment(bathymetry=np.array(True), ssp=1500.0)

    def test_a_zero_dimensional_depth_coerces_to_that_depth(self):
        assert float(Bathymetry.coerce(np.array(100.0)).depths[0]) == 100.0

    def test_a_zero_dimensional_sound_speed_builds_the_same_bottom(self):
        flat = Environment(bathymetry=100.0, ssp=1500.0, bottom=1600.0)
        zero_d = Environment(bathymetry=100.0, ssp=1500.0,
                             bottom=np.array(1600.0))
        assert (zero_d.bottom.halfspace_at(range=0.0).sound_speed
                == flat.bottom.halfspace_at(range=0.0).sound_speed)

    def test_a_zero_dimensional_bool_is_refused_as_a_sound_speed(self):
        with pytest.raises(ConfigurationError, match='is a bool'):
            Environment(bathymetry=100.0, ssp=1500.0,
                        bottom=np.array(True))


class TestBathymetryCoerceRefusesNonNumericScalars:
    """A numeric string is not a depth. ``_scalar_or_none`` returns for
    every numeric scalar, so a 0-d value reaching the fallback (``'100'``,
    ``b'100'``, a 0-d string array, ``None``) must be refused with a message
    that names it, the way ``SoundSpeedProfile.coerce`` refuses a string."""

    @pytest.mark.parametrize('spelling', ['100', b'100', np.array('100')],
                             ids=['str', 'bytes', '0-d str array'])
    def test_a_numeric_string_is_refused_and_named(self, spelling):
        with pytest.raises(ConfigurationError, match='non-numeric') as info:
            Bathymetry.coerce(spelling)
        assert repr(spelling) in str(info.value)

    def test_none_keeps_its_non_numeric_message(self):
        with pytest.raises(ConfigurationError, match='non-numeric None'):
            Bathymetry.coerce(None)

    def test_the_environment_refuses_a_string_bathymetry(self):
        with pytest.raises(ConfigurationError, match='non-numeric'):
            Environment(bathymetry='100', ssp=1500.0)

    def test_a_numeric_scalar_coerces_to_a_flat_seafloor(self):
        assert Bathymetry.coerce(np.float32(100.0)).depth == 100.0


class TestFromStringRefusesNonStrings:
    """``BoundaryType.from_string`` / ``AttenuationUnits.from_string`` refuse
    a non-string, non-enum input as a ``ConfigurationError`` naming the
    received type, and the carrier-level ``acoustic_type`` message keeps its
    own wording."""

    @pytest.mark.parametrize("bad", [2, None])
    def test_boundary_type_names_the_received_type(self, bad):
        with pytest.raises(ConfigurationError,
                           match=type(bad).__name__):
            BoundaryType.from_string(bad)

    @pytest.mark.parametrize("bad", [2, None])
    def test_attenuation_units_names_the_received_type(self, bad):
        with pytest.raises(ConfigurationError,
                           match=type(bad).__name__):
            AttenuationUnits.from_string(bad)

    def test_the_carrier_message_names_the_bad_acoustic_type(self):
        with pytest.raises(ConfigurationError,
                           match=r"acoustic_type=2 is not recognized"):
            BoundaryProperties(acoustic_type=2)

    def test_the_lowercase_m_is_refused_before_the_uppercasing_lookup(self):
        """``'m'`` and ``'M'`` are two different AT units, and the lookup
        below upper-cases — so only an explicit guard keeps ``'m'`` from
        silently becoming ``DB_PER_M``. Both sides of that case boundary."""
        with pytest.raises(ConfigurationError, match="has no enum member"):
            AttenuationUnits.from_string('m')
        assert AttenuationUnits.from_string('M') is AttenuationUnits.DB_PER_M

    def test_from_string_is_documented_as_having_no_in_package_caller(self):
        """The reader's question at an unreferenced public parser is "what
        calls this?", and the answer — nothing, because every writer hardwires
        ``TOPOPT(3:3)='W'`` — is the useful one. A ``grep`` that finds only
        this test is otherwise indistinguishable from dead code."""
        import ast
        from pathlib import Path

        import uacpy
        package = Path(uacpy.__file__).resolve().parent
        callers = []
        for path in sorted(package.rglob('*.py')):
            if 'third_party' in path.parts or 'tests' in path.parts:
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == 'from_string'
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == 'AttenuationUnits'):
                    callers.append(f"{path.relative_to(package)}:{node.lineno}")
        prose = AttenuationUnits.from_string.__doc__ or ''
        if callers:
            assert 'No uacpy API takes an attenuation unit' not in prose, (
                f"the docstring says nothing calls this, but {callers} do")
        else:
            assert 'No uacpy API takes an attenuation unit' in prose


class TestEnvironmentStoresItsCoordinatesAsGiven:
    """A geolocation longitude is accepted in either sign convention up to
    one full wrap and stored exactly as given; a lookup wraps it
    (``normalize_lon``). Beyond a full wrap it is refused."""

    def test_a_lon_beyond_180_is_stored_as_given(self):
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          location=(10.0, 250.0))
        assert env.location == (10.0, 250.0)

    @pytest.mark.parametrize('lon', [360.0, -360.0])
    def test_a_full_wrap_either_way_is_stored_as_given(self, lon):
        env = Environment(bathymetry=100.0, ssp=1500.0, location=(10.0, lon))
        assert env.location == (10.0, lon)

    @pytest.mark.parametrize('lon', [360.5, -360.5])
    def test_a_lon_past_a_full_wrap_is_refused_either_way(self, lon):
        with pytest.raises(ConfigurationError,
                           match=r"Environment: location: longitude must be "
                                 r"in \[-360, 360\]"):
            Environment(bathymetry=100.0, ssp=1500.0, location=(10.0, lon))

    def test_a_transect_across_the_antimeridian_gets_its_great_circle_midpoint(self):
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          transect=((0.0, 170.0), (0.0, -160.0)))
        assert env.location[0] == pytest.approx(0.0)
        assert env.location[1] == pytest.approx(-175.0)

    def test_an_in_range_longitude_is_stored_bit_exactly(self):
        # ==, not approx: ((lon + 180) % 360) - 180 returns -6.199999999999989
        # for -6.2, which is why the carrier never wraps what it stores.
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          location=(10.0, -6.2))
        assert env.location[1] == -6.2

    def test_a_datetime_is_stored_as_its_utc_calendar_date(self):
        import datetime as dt
        late = dt.datetime(2026, 3, 15, 23, 30,
                           tzinfo=dt.timezone(dt.timedelta(hours=-2)))
        env = Environment(bathymetry=100.0, ssp=1500.0, date=late)
        assert env.date == dt.date(2026, 3, 16)
        env = Environment(bathymetry=100.0, ssp=1500.0,
                          date=np.datetime64('2026-03-15T12:00'))
        assert env.date == dt.date(2026, 3, 15)

    def test_a_lon_beyond_a_full_wrap_is_rejected(self):
        with pytest.raises(ConfigurationError, match="longitude"):
            Environment(bathymetry=100.0, ssp=1500.0,
                        location=(10.0, 9999.0))


class TestEnvironmentCoerceDispatchesRejectBool:
    """``True``/``False`` reach the scalar dispatch arm as numbers (bool is
    an int subclass; ``np.ndim(True) == 0``), where they would mean a 1 or
    0 m/s ocean, a 1 or 0 m/s half-space, or a 1 m deep seafloor. All three
    coerce dispatches refuse bools with a typed error instead."""

    @pytest.mark.parametrize("value", [True, False])
    def test_ssp_bool_raises_a_typed_error(self, value):
        with pytest.raises(ConfigurationError, match="bool"):
            SoundSpeedProfile.coerce(value, depth_max=100.0)

    @pytest.mark.parametrize("value", [True, False])
    def test_bottom_bool_raises_a_typed_error(self, value):
        with pytest.raises(ConfigurationError, match="bool"):
            Bottom.coerce(value)

    @pytest.mark.parametrize("value", [True, False, np.True_])
    def test_bathymetry_bool_raises_a_typed_error(self, value):
        with pytest.raises(ConfigurationError, match="bool"):
            Bathymetry.coerce(value)

    def test_numeric_scalars_coerce_on_all_three_axes(self):
        assert SoundSpeedProfile.coerce(
            1480, depth_max=50.0).sound_speed[0, 0] == pytest.approx(1480.0)
        bottom = Bottom.coerce(1700)
        assert bottom.halfspace_at(
            range=0.0).sound_speed == pytest.approx(1700.0)
        assert Bathymetry.coerce(120).depth == pytest.approx(120.0)

    @pytest.mark.parametrize('spelling', [
        1500.0, 1500, np.float64(1500.0), np.int64(1500),
        np.array(1500.0), np.array(1500),
    ], ids=['float', 'int', 'np.float64', 'np.int64', '0d-float-array',
            '0d-int-array'])
    def test_every_scalar_spelling_reaches_isovelocity(self, spelling):
        """``Environment(ssp=np.array(1500.0))`` failed while
        ``Environment(bathymetry=np.array(200.0))`` succeeded in the same
        constructor call: a 0-d ndarray matches no ``isinstance`` in the
        scalar chain and fell through to ``from_pairs``, which complained
        about an ``(N, 2)`` shape the caller never asked for."""
        profile = SoundSpeedProfile.coerce(spelling, depth_max=50.0)
        assert profile.sound_speed[0, 0] == pytest.approx(1500.0)
        env = uacpy.Environment(bathymetry=200.0, ssp=spelling)
        assert env.ssp.sound_speed[0, 0] == pytest.approx(1500.0)

    @pytest.mark.parametrize('spelling', [
        np.array('abc'), np.array(object(), dtype=object),
    ], ids=['0d-string-array', '0d-object-array'])
    def test_a_non_numeric_zero_d_array_raises_a_typed_error(self, spelling):
        # The other side of the same branch: admitting the numeric 0-d
        # spelling must not route a string or an object into float().
        with pytest.raises(ConfigurationError, match='0-d array'):
            SoundSpeedProfile.coerce(spelling, depth_max=50.0)

    def test_a_zero_d_bool_array_is_refused_like_a_bool(self):
        with pytest.raises(ConfigurationError, match='is a bool'):
            SoundSpeedProfile.coerce(np.array(True), depth_max=50.0)

    def test_a_one_d_array_is_refused_and_a_two_d_array_is_pairs(self):
        # The dimension boundary: 0-d is a scalar, 1-d is refused with a
        # message naming both accepted forms, 2-d is (depth, c) pairs.
        with pytest.raises(ConfigurationError, match=r'shape \(N, 2\)'):
            SoundSpeedProfile.coerce(np.array([1500.0, 1490.0]),
                                     depth_max=50.0)
        profile = SoundSpeedProfile.coerce(
            np.array([[0.0, 1500.0], [50.0, 1490.0]]), depth_max=50.0)
        assert profile.sound_speed[0, 0] == pytest.approx(1500.0)


def test_surface_and_bottom_delegate_the_same_boundary_field_set():
    """``Surface`` and ``Bottom`` hold their nodes in the same
    ``BoundaryProperties``, so the names each follows through on a delegated
    write cannot legitimately differ. They were two separately written
    frozensets of the same nine names; ``Surface`` now imports ``Bottom``'s,
    which it already imported four other names from. Identity, not equality:
    equality would still pass on two copies that had been edited in step and
    then let drift."""
    from uacpy.core.boundary import _HALFSPACE_DELEGATED
    from uacpy.core.surface import _SURFACE_DELEGATED
    assert _SURFACE_DELEGATED is _HALFSPACE_DELEGATED
    # Every delegated name is a real BoundaryProperties field, which is what
    # makes one shared set correct rather than a coincidence.
    fields = {f.name for f in dataclasses.fields(BoundaryProperties)}
    assert _SURFACE_DELEGATED <= fields


def _two_vacuum_boundaries():
    return [BoundaryProperties(acoustic_type='vacuum') for _ in range(2)]


def _two_vacuum_columns():
    return [SeabedColumn([], BoundaryProperties(acoustic_type='vacuum'))
            for _ in range(2)]


# Every coordinate array a core carrier casts with
# ``np.array(..., dtype=float)``, as
# ``(message label, ctor, field, complex samples, real samples, siblings)``.
# The six carriers are ``Source``, ``Receiver``, ``SoundSpeedProfile``,
# ``_RangeProfile`` — the last through both of its concrete subclasses, since
# the guard reads ``_VALUE_FIELD`` and so builds a different label for each —
# plus the two range-dependent boundary carriers ``Surface`` and ``Bottom``.
_COMPLEX_COORDINATE_FIELDS = [
    ("source depths", uacpy.Source, 'depths',
     [50.0 + 2.0j], [50.0], dict(frequencies=100.0)),
    ("source frequencies", uacpy.Source, 'frequencies',
     [100.0 + 5.0j], [100.0], dict(depths=50.0)),
    ("receiver depths", uacpy.Receiver, 'depths',
     [50.0 + 2.0j], [50.0], dict(ranges=[1000.0])),
    ("receiver ranges", uacpy.Receiver, 'ranges',
     [1000.0 + 5.0j], [1000.0], dict(depths=[50.0])),
    ("SoundSpeedProfile.depths", SoundSpeedProfile, 'depths',
     [0.0, 100.0 + 1.0j], [0.0, 100.0], dict(sound_speed=[1500.0, 1490.0])),
    ("SoundSpeedProfile sound speeds", SoundSpeedProfile, 'sound_speed',
     [1500.0 + 2.0j, 1490.0], [1500.0, 1490.0], dict(depths=[0.0, 100.0])),
    ("SoundSpeedProfile.ranges", SoundSpeedProfile, 'ranges',
     [0.0 + 1.0j, 1000.0], [0.0, 1000.0],
     dict(depths=[0.0, 100.0],
          sound_speed=[[1500.0, 1500.0], [1490.0, 1490.0]])),
    ("Bathymetry ranges", Bathymetry, 'ranges',
     [0.0 + 1.0j, 1000.0], [0.0, 1000.0], dict(depths=[100.0, 120.0])),
    ("Bathymetry depths", Bathymetry, 'depths',
     [100.0 + 7.0j, 120.0], [100.0, 120.0], dict(ranges=[0.0, 1000.0])),
    ("Altimetry ranges", Altimetry, 'ranges',
     [0.0 + 1.0j, 1000.0], [0.0, 1000.0], dict(heights=[0.5, -0.5])),
    ("Altimetry heights", Altimetry, 'heights',
     [0.5 + 1.0j, -0.5], [0.5, -0.5], dict(ranges=[0.0, 1000.0])),
    ("Surface.ranges", Surface, 'ranges',
     [0.0 + 1.0j, 1000.0], [0.0, 1000.0],
     dict(nodes=_two_vacuum_boundaries())),
    ("Bottom.ranges", Bottom, 'ranges',
     [0.0 + 1.0j, 1000.0], [0.0, 1000.0],
     dict(columns=_two_vacuum_columns())),
]


_COMPLEX_FIELD_IDS = [case[0] for case in _COMPLEX_COORDINATE_FIELDS]


class TestEveryCoreCarrierRejectsComplexCoordinates:
    """One typed ``ConfigurationError`` for a complex coordinate, on every
    carrier and in every container spelling.

    The float64 cast each carrier applies destroys a complex input two
    different ways — an ndarray keeps only the real part under a
    ``ComplexWarning`` that the suite's ``ignore::UserWarning`` filter hides,
    a scalar or list raises a bare ``TypeError`` from ``float()`` naming no
    field — so the guard runs ahead of the cast and both spellings are pinned
    per field, together with the real construction it lets through.
    """

    @pytest.mark.parametrize("label,ctor,field,bad,good,siblings",
                             _COMPLEX_COORDINATE_FIELDS,
                             ids=_COMPLEX_FIELD_IDS)
    def test_a_complex_ndarray_raises_a_typed_error_naming_the_field(
            self, label, ctor, field, bad, good, siblings):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            with pytest.raises(ConfigurationError,
                               match=re.escape(f"{label} must be real numbers")):
                ctor(**{field: np.array(bad)}, **siblings)

    @pytest.mark.parametrize("label,ctor,field,bad,good,siblings",
                             _COMPLEX_COORDINATE_FIELDS,
                             ids=_COMPLEX_FIELD_IDS)
    def test_a_complex_list_raises_the_same_typed_error_as_an_ndarray(
            self, label, ctor, field, bad, good, siblings):
        with pytest.raises(ConfigurationError,
                           match=re.escape(f"{label} must be real numbers")):
            ctor(**{field: bad}, **siblings)

    @pytest.mark.parametrize("label,ctor,field,bad,good,siblings",
                             _COMPLEX_COORDINATE_FIELDS,
                             ids=_COMPLEX_FIELD_IDS)
    def test_a_complex_dtype_carrying_a_zero_imaginary_part_is_refused(
            self, label, ctor, field, bad, good, siblings):
        # The near side of the boundary: dtype is the criterion, because the
        # cast emits its ComplexWarning for any complex dtype regardless of
        # what the imaginary parts hold (measured on numpy 2.5).
        with pytest.raises(ConfigurationError,
                           match=re.escape(f"{label} must be real numbers")):
            ctor(**{field: np.array(good, dtype=complex)}, **siblings)

    @pytest.mark.parametrize("label,ctor,field,bad,good,siblings",
                             _COMPLEX_COORDINATE_FIELDS,
                             ids=_COMPLEX_FIELD_IDS)
    def test_a_real_float_array_constructs_and_is_stored_as_float64(
            self, label, ctor, field, bad, good, siblings):
        # The far side of the same boundary, and the guard's cost: a real
        # array reaches the cast untouched and raises no warning of any kind.
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            carrier = ctor(**{field: np.array(good, dtype=float)}, **siblings)
        assert np.asarray(getattr(carrier, field)).dtype == np.float64

    def test_a_complex_scalar_raises_a_typed_error(self):
        # The bare-scalar spelling, which only the single-value fields take.
        with pytest.raises(ConfigurationError,
                           match="source frequencies must be real"):
            Source(depths=50.0, frequencies=100.0 + 5.0j)

    def test_the_error_names_the_first_element_carrying_an_imaginary_part(self):
        # Not flat index 0: one complex element promotes the whole array, so
        # the leading sample prints as ``(10+0j)`` and names nothing the
        # caller can act on.
        with pytest.raises(
                ConfigurationError,
                match='must be real numbers; got complex value') as exc:
            Source(depths=[10.0, 50.0 + 2.0j, 90.0], frequencies=100.0)
        assert '(50+2j) at flat index 1 of 3 value(s)' in str(exc.value)

    def test_an_empty_complex_array_reports_its_dtype(self):
        # No element to point at, so the offence is the dtype itself; the
        # alternative is an IndexError out of the guard.
        with pytest.raises(ConfigurationError,
                           match="an empty array of dtype complex"):
            Source(depths=np.array([], dtype=complex), frequencies=100.0)

    # ``Field.coords`` arrives as a dict rather than a named field, so it
    # cannot ride the table above, but it casts the same way and is the one
    # place where a complex *axis* is easy to confuse with the complex
    # ``data`` a pressure field legitimately carries.
    @pytest.mark.parametrize('axis', [np.array([0.0 + 1.0j, 10.0]),
                                      [0.0 + 1.0j, 10.0],
                                      np.array([0.0, 10.0], dtype=complex)],
                             ids=['ndarray', 'list', 'zero_imaginary_dtype'])
    def test_a_complex_field_coordinate_axis_is_refused_by_name(self, axis):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            with pytest.raises(
                    ConfigurationError,
                    match=re.escape("Field.coords['range'] must be real")):
                Field(data=np.zeros(2), coords={'range': axis})

    def test_a_complex_field_data_array_is_accepted(self):
        """The far side of the guard, and the reason it is on the coords and
        not on the Field: pressure is complex on every frequency-domain
        result, and only the axis is required to be a real distance."""
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            f = Field(data=np.array([1.0 + 2.0j, 3.0 - 1.0j]),
                      coords={'range': [0.0, 10.0]})
        assert np.iscomplexobj(f.data)
        assert f.coords['range'].dtype == np.float64


def _bathymetry():
    return Bathymetry(ranges=[0.0, 1000.0, 2000.0],
                      depths=[100.0, 200.0, 300.0])


class TestRangeProfileQueriesMustBeFinite:
    """``Bathymetry``/``Altimetry`` query a profile by searchsorted rather
    than through ``collapse_axis``, so a NaN or inf range walked to an end
    node and returned that node's stored value as a successful lookup. The
    query stays array-capable — ``PropagationModel`` hands it the whole
    receiver range axis — so the check is element-wise."""

    def test_bathymetry_at_nan_range_raises_a_typed_label_error(self):
        with pytest.raises(ConfigurationError,
                           match="range=nan is not a finite label"):
            _bathymetry().at(range=np.nan)

    @pytest.mark.parametrize('bad', [np.inf, -np.inf])
    def test_bathymetry_at_infinite_range_raises(self, bad):
        with pytest.raises(ConfigurationError, match="not a finite label"):
            _bathymetry().at(range=bad)

    def test_bathymetry_eval_nan_range_raises(self):
        with pytest.raises(ConfigurationError, match="not a finite label"):
            _bathymetry().eval(range=float('nan'))

    def test_bathymetry_eval_rejects_an_array_containing_one_nan(self):
        with pytest.raises(ConfigurationError,
                           match='is not a finite label') as exc:
            _bathymetry().eval(range=np.array([0.0, np.nan, 2000.0]))
        assert "1 of 3 query ranges" in str(exc.value)

    def test_bathymetry_eval_accepts_a_finite_array_of_ranges(self):
        out = _bathymetry().eval(range=np.array([0.0, 1500.0]))
        assert isinstance(out, np.ndarray)
        assert np.allclose(out, [100.0, 250.0])

    def test_bathymetry_at_a_finite_range_returns_the_nearest_depth(self):
        assert _bathymetry().at(range=900.0) == 200.0

    def test_altimetry_at_nan_range_raises(self):
        alt = Altimetry(ranges=[0.0, 1000.0], heights=[0.0, 1.0])
        with pytest.raises(ConfigurationError, match="not a finite label"):
            alt.at(range=np.nan)


def _nearest_probe_field():
    return Field(data=np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                 coords={'depth': [0.0, 50.0],
                         'range': [0.0, 500.0, 1000.0]},
                 model='Test')


def _bottom():
    return Bottom.from_halfspaces([0.0, 1000.0, 2000.0],
                                  sound_speed=[1600.0, 1700.0, 1800.0],
                                  density=1.8, attenuation=0.5)


class TestSharedNearestPathRefusesAnAxisAbsorbingLabel:
    """``_grid.py``'s argmin-based nearest lookups refuse a finite label so
    far outside the axis that every ``|axis - label|`` rounds to the same
    value — the refusal ``Field.at`` makes on its own argmin path. A label
    that still ranks (1e9 against a metre-scale axis) answers with the
    correct end node."""

    def test_bottom_at_refuses_the_absorbing_label(self):
        with pytest.raises(ConfigurationError, match='same distance'):
            _bottom().at(range=1e300)

    def test_bottom_at_answers_a_rankable_out_of_span_label(self):
        assert _bottom().at(range=1e9).halfspace.sound_speed == \
            pytest.approx(1800.0, rel=1e-12)

    def test_bottom_at_matches_field_at_on_the_absorbing_label(self):
        with pytest.raises(
                ConfigurationError,
                match='every sample rounds to the same distance') as exc_field:
            _nearest_probe_field().at(depth=1e300)
        with pytest.raises(
                ConfigurationError,
                match='every sample rounds to the same distance') as exc_bottom:
            _bottom().at(range=1e300)
        assert 'same distance' in str(exc_field.value)
        assert 'same distance' in str(exc_bottom.value)

    def test_surface_at_refuses_the_absorbing_label(self):
        s = Surface.coerce(
            [(0.0, BoundaryProperties(acoustic_type='vacuum')),
             (1000.0, BoundaryProperties(acoustic_type='half-space',
                                         sound_speed=340.0, density=0.0012,
                                         attenuation=0.0))])
        with pytest.raises(ConfigurationError, match='same distance'):
            s.at(range=1e300)

    def test_ssp_at_refuses_the_absorbing_depth_label(self):
        ssp = SoundSpeedProfile(depths=[0.0, 50.0, 100.0],
                                sound_speed=[1500.0, 1490.0, 1495.0])
        with pytest.raises(ConfigurationError, match='same distance'):
            ssp.at(depth=1e300)

    def test_field_eval_nearest_matches_field_at_refusal(self):
        with pytest.raises(ConfigurationError, match='same distance'):
            _nearest_probe_field().eval(depth=1e300, method='nearest')

    def test_bathymetry_searchsorted_path_answers_the_far_node(self):
        """``Bathymetry.at`` brackets by searchsorted + midpoint compare,
        which is cancellation-immune: the huge label resolves to the correct
        end node on both sides."""
        bath = Bathymetry(ranges=[0.0, 1000.0, 2000.0],
                          depths=[100.0, 200.0, 300.0])
        assert bath.at(range=1e300) == pytest.approx(300.0, rel=1e-12)
        assert bath.at(range=-1e300) == pytest.approx(100.0, rel=1e-12)

    def test_a_single_node_axis_takes_any_finite_label(self):
        """One node cannot tie with another, so there is nothing to lose."""
        b1 = Bottom.from_halfspaces([5000.0], sound_speed=[1600.0],
                                    density=1.8, attenuation=0.5)
        assert b1.at(range=1e300).halfspace.sound_speed == \
            pytest.approx(1600.0, rel=1e-12)


class TestScalarSpellingsAreSharedByTheThreeCoercers:
    """``ssp=``, ``bathymetry=`` and ``bottom=`` accept a Python number and a
    numeric 0-d array as the same scalar, and refuse a bool (bare or 0-d)
    through one guard."""

    @pytest.mark.parametrize('spelling', [1500.0, np.array(1500.0), np.float32(1500.0)])
    def test_a_scalar_ssp_is_isovelocity(self, spelling):
        env = uacpy.Environment(name='t', bathymetry=100.0, ssp=spelling)
        assert np.allclose(env.ssp.sound_speed, 1500.0)

    @pytest.mark.parametrize('spelling', [100.0, np.array(100.0)])
    def test_a_scalar_bathymetry_is_flat(self, spelling):
        env = uacpy.Environment(name='t', bathymetry=spelling, ssp=1500.0)
        assert float(env.depth) == 100.0

    def test_a_zero_d_bottom_is_a_half_space(self):
        env = uacpy.Environment(name='t', bathymetry=100.0, ssp=1500.0,
                                bottom=np.array(1650.0))
        assert env.bottom.halfspace_at(range=0.0).sound_speed == 1650.0

    @pytest.mark.parametrize('kw', ['ssp', 'bathymetry', 'bottom'])
    @pytest.mark.parametrize('bad', [True, np.array(True)])
    def test_a_bool_is_refused_by_all_three(self, kw, bad):
        base = dict(name='t', bathymetry=100.0, ssp=1500.0)
        base[kw] = bad
        with pytest.raises(ConfigurationError, match='is a bool'):
            uacpy.Environment(**base)


class TestOneElementSspArrayIsToldTheAcceptedForms:
    """A 1-D ``ssp`` array with a single entry is neither a scalar nor a
    pair table; the refusal names ``ssp=`` and both accepted forms."""

    @pytest.mark.parametrize('value', [np.array([1500.0]), [1500.0]])
    def test_message_names_the_scalar_and_pair_forms(self, value):
        with pytest.raises(ConfigurationError,
                           match='ssp must be a scalar') as info:
            Environment(bathymetry=100.0, ssp=value)
        text = str(info.value)
        assert 'ssp=' in text
        assert 'scalar' in text
        assert '(depth, sound_speed)' in text

    def test_two_entry_flat_array_is_refused_as_a_pair_table(self):
        with pytest.raises(ConfigurationError,
                           match='ssp must be a scalar') as info:
            Environment(bathymetry=100.0, ssp=np.array([1500.0, 1520.0]))
        assert 'ssp=' in str(info.value)

    def test_scalar_and_pairs_coerce(self):
        assert Environment(bathymetry=100.0, ssp=1500.0).ssp is not None
        env = Environment(bathymetry=100.0,
                          ssp=[(0.0, 1500.0), (100.0, 1480.0)])
        assert env.ssp is not None


class TestAReassignedCarrierIsCoercedLikeAConstructedOne:
    """EXPERT-12: ``env.bottom = BoundaryProperties(...)`` stored the bare
    boundary and every wrapper then failed on ``bottom.columns``."""

    def test_bottom_surface_and_ssp_are_coerced_on_assignment(self):
        from uacpy import Environment, BoundaryProperties
        from uacpy.core.bottom import Bottom
        from uacpy.core.surface import Surface
        from uacpy.core.ssp import SoundSpeedProfile
        env = Environment(bathymetry=100.0, ssp=1500.0)
        env.bottom = BoundaryProperties(acoustic_type='half-space',
                                        sound_speed=1700.0, density=1.8,
                                        attenuation=0.5)
        assert isinstance(env.bottom, Bottom)
        assert env.bottom.halfspace_at(range=0.0).sound_speed == 1700.0
        env.surface = BoundaryProperties(acoustic_type='vacuum')
        assert isinstance(env.surface, Surface)
        env.ssp = 1490.0
        assert isinstance(env.ssp, SoundSpeedProfile)


class TestTheEnvironmentKeepsItsOwnCarriers:
    """D22: the environment holds its own copy of every carrier it is
    given, so an edit of the caller's object leaves it as built. D6: an
    assignment is completed as the constructor completes the same
    argument, and ``data_sources`` is read from the carriers each time."""

    @staticmethod
    def _prov(source_id):
        from uacpy.data.sources import SOURCES, DataProvenance
        return (DataProvenance(source=SOURCES[source_id]),)

    def test_construction_never_aliases_a_callers_object(self):
        from uacpy.core.absorption import Thorp
        from uacpy.core.ssp import SoundSpeedProfile
        from uacpy.core.surface import Surface
        given = dict(
            bathymetry=Bathymetry(ranges=np.array([0.0]),
                                  depths=np.array([100.0])),
            ssp=SoundSpeedProfile.from_pairs([(0.0, 1500.0),
                                              (200.0, 1490.0)]),
            bottom=Bottom.from_halfspace(BoundaryProperties(
                acoustic_type='half-space', sound_speed=1700.0,
                density=1.8)),
            surface=Surface(nodes=[
                BoundaryProperties(acoustic_type='vacuum')]),
            altimetry=Altimetry(ranges=np.array([0.0, 1000.0]),
                                heights=np.array([0.0, 1.0])),
            absorption=Thorp())
        env = Environment(**given)
        for name, obj in given.items():
            assert getattr(env, name) is not obj, name
        given['bathymetry'].depths = [300.0]
        given['ssp'].sound_speed = [1400.0, 1400.0]
        given['bottom'].sound_speed = 1600.0
        given['altimetry'].heights = [0.0, 2.0]
        assert env.depth == 100.0
        assert env.ssp.sound_speed[0, 0] == 1500.0
        assert env.bottom.halfspace_at(range=0.0).sound_speed == 1700.0
        assert env.altimetry.heights[-1] == 1.0

    def test_a_deeper_seafloor_extends_the_profile_on_assignment(self):
        from uacpy.core.ssp import SoundSpeedProfile
        env = Environment(bathymetry=100.0,
                          ssp=[(0.0, 1500.0), (100.0, 1490.0)])
        env.bathymetry = 500.0
        assert env.ssp.depths[-1] == 500.0
        env.ssp = SoundSpeedProfile.from_pairs([(0.0, 1500.0),
                                                (50.0, 1495.0)])
        assert env.ssp.depths[-1] == 500.0

    def test_data_sources_follow_a_reassigned_carrier(self):
        env = Environment(bathymetry=100.0, bottom=BoundaryProperties(
            acoustic_type='half-space', sound_speed=1700.0, density=1.8,
            data_sources=self._prov('crust1')))
        assert [p.source.id for p in env.data_sources] == ['crust1']
        env.bottom = BoundaryProperties(
            acoustic_type='half-space', sound_speed=1650.0, density=1.7,
            data_sources=self._prov('grainsize'))
        assert [p.source.id for p in env.data_sources] == ['grainsize']

    def test_data_sources_are_read_only(self):
        env = Environment(bathymetry=100.0)
        with pytest.raises(AttributeError, match='data_sources'):
            env.data_sources = self._prov('gebco')
        assert env.data_sources == ()

    def test_extra_records_sit_between_the_carriers_and_the_altimetry(
            self):
        env = Environment(
            bathymetry=Bathymetry(ranges=np.array([0.0]),
                                  depths=np.array([100.0]),
                                  data_sources=self._prov('gebco')),
            altimetry=Altimetry(ranges=np.array([0.0, 1000.0]),
                                heights=np.array([0.0, 1.0]),
                                data_sources=self._prov('waverys')),
            extra_data_sources=(self._prov('woa23')
                                + self._prov('gebco')))
        assert ([p.source.id for p in env.data_sources]
                == ['gebco', 'woa23', 'waverys'])

    def test_a_derived_location_follows_a_reassigned_transect(self):
        from uacpy.core.geo import great_circle_midpoint
        a, b, c = (43.0, 5.0), (43.0, 6.0), (44.0, 6.0)
        env = Environment(bathymetry=100.0, transect=(a, b))
        env.transect = (a, c)
        assert env.location == great_circle_midpoint(a, c)
        pinned = Environment(bathymetry=100.0, transect=(a, b),
                             location=(43.5, 5.5))
        pinned.transect = (a, c)
        assert pinned.location == (43.5, 5.5)

    def test_an_assignment_copies_only_the_assigned_value(self):
        """The other carriers are the environment's own already; an
        assignment leaves them in place, however large they are."""
        env = Environment(bathymetry=100.0,
                          ssp=[(0.0, 1500.0), (100.0, 1490.0)])
        ssp, surface, bathymetry = env.ssp, env.surface, env.bathymetry
        new = Bottom.from_halfspace(BoundaryProperties(
            acoustic_type='half-space', sound_speed=1700.0, density=1.8))
        env.bottom = new
        assert env.bottom is not new
        assert env.ssp is ssp and env.surface is surface
        assert env.bathymetry is bathymetry
        env.bathymetry = 50.0           # the profile already spans it
        assert env.ssp is ssp


class TestRoutedDocstringsStateTheCurrentContract:
    """Docstring contracts routed from the io and journey audits: the
    reflection table's producers, the SPARC snapshot-bin description and the
    default seabed."""

    def test_the_reflection_file_names_bounce_and_the_writer_not_oasr(self):
        from uacpy.core.boundary import BoundaryProperties
        doc = BoundaryProperties.__doc__
        assert 'write_reflection_coefficient' in doc
        assert 'Generated by BOUNCE or OASR' not in doc

    def test_the_snapshot_bin_says_the_field_is_at_the_source_frequency(self):
        from uacpy.core.results._base import _DOCUMENTED_METADATA
        text = _DOCUMENTED_METADATA[('SPARC', 'snapshot_freq_bin')][1]
        assert 'evaluated at the source frequency' in text

    def test_the_default_seabed_is_not_called_sand_like(self):
        assert 'sand-like' not in Environment.__doc__


class TestEnvironmentIO:
    """Tests for Environment I/O."""

    def test_environment_with_bathymetry_file(self):
        """Test loading environment with bathymetry from file."""
        # Create temporary bathymetry file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("# Range(m) Depth(m)\n")
            f.write("0 80\n")
            f.write("5000 100\n")
            f.write("10000 120\n")
            bathy_file = f.name

        try:
            # Load bathymetry
            bathymetry = np.loadtxt(bathy_file)

            env = uacpy.Environment(
                name="Test",
                ssp=1500.0,
                bathymetry=bathymetry
            )

            assert env.is_range_dependent
            assert env.bathymetry.n_ranges == 3
            assert env.bathymetry.depths[0] == 80
            assert env.bathymetry.depths[-1] == 120

        finally:
            Path(bathy_file).unlink()

    def test_environment_ssp_from_array(self):
        """Test creating environment with SSP from array."""
        depths = np.linspace(0, 100, 11)
        sound_speeds = 1500 + depths * 0.1  # Linear gradient

        ssp_data = np.column_stack([depths, sound_speeds])

        env = uacpy.Environment(
            name="Test",
            bathymetry=100.0,
            ssp=SoundSpeedProfile.from_pairs(ssp_data)
        )

        assert len(env.ssp.to_pairs()) == 11
        assert np.allclose(env.ssp.to_pairs()[:, 0], depths)


class TestAcousticFieldsAreAlwaysPresentOnTheirCarriers:
    """The contract the io writers read half-space properties under.

    The writers address ``hs.shear_speed``, ``layer.shear_attenuation``,
    ``env.surface.roughness`` and their siblings directly. That is only
    safe because every carrier fills all six acoustic fields at
    construction, so there is nothing for a ``getattr`` default to catch:

    * ``BoundaryProperties.__post_init__`` walks ``_ACOUSTIC_DEFAULTS``
      (density, sound_speed, attenuation, roughness, shear_speed,
      shear_attenuation) and either substitutes the default for a ``None``
      or coerces the given value with ``float()``, then requires each to
      be non-negative.
    * ``SedimentLayer`` declares shear_speed / shear_attenuation /
      roughness as plain floats defaulting to 0.0 and coerces them the
      same way.
    * ``Surface.__getattr__`` forwards each of those names to
      ``properties[0]``, and ``Surface.__post_init__`` refuses an empty
      node list and any node that is not a ``BoundaryProperties``.

    If any of that changes, the writers raise ``AttributeError`` instead
    of silently emitting a default into a deck — which is why this is
    pinned rather than left implicit.
    """

    FIELDS = ('density', 'sound_speed', 'attenuation', 'roughness',
              'shear_speed', 'shear_attenuation')

    @pytest.mark.parametrize('kwargs', [
        {},
        {'acoustic_type': 'vacuum'},
        {'acoustic_type': 'rigid'},
        {'acoustic_type': 'half-space', 'sound_speed': 1800.0,
         'density': 2.0, 'attenuation': 0.1},
        {'acoustic_type': 'half-space', 'sound_speed': 2400.0,
         'density': 2.5, 'attenuation': 0.2, 'shear_speed': 1000.0,
         'shear_attenuation': 0.4, 'roughness': 0.6},
    ])
    def test_boundary_properties_fills_every_acoustic_field(self, kwargs):
        bp = BoundaryProperties(**kwargs)
        for name in self.FIELDS:
            value = getattr(bp, name)
            assert isinstance(value, float), name
            assert value >= 0.0, name

    def test_sediment_layer_fills_every_optional_field(self):
        from uacpy.core.environment import SedimentLayer
        layer = SedimentLayer(thickness=5.0, sound_speed=1550.0, density=1.5)
        for name in ('attenuation', 'shear_speed', 'shear_attenuation',
                     'roughness'):
            value = getattr(layer, name)
            assert isinstance(value, float), name
            assert value >= 0.0, name

    def test_a_surface_forwards_every_acoustic_field_to_its_first_node(self):
        env = Environment(bathymetry=100.0, ssp=1500.0)
        for name in self.FIELDS:
            assert getattr(env.surface, name) == getattr(
                env.surface.nodes[0], name), name

    def test_a_surface_cannot_be_built_without_a_node_to_forward_to(self):
        from uacpy.core.surface import Surface
        with pytest.raises(ConfigurationError, match='at least one'):
            Surface(nodes=[])

    def test_a_halfspace_lookup_returns_a_boundary_properties(self):
        """``halfspace_at`` is the other carrier the writers address
        directly, and it must hand back the filled dataclass, not a
        duck-typed stand-in."""
        env = Environment(bathymetry=100.0, ssp=1500.0)
        hs = env.bottom.halfspace_at(range=0.0)
        assert isinstance(hs, BoundaryProperties)
        for name in self.FIELDS:
            assert isinstance(getattr(hs, name), float), name


class TestAnAbsorptionLawSurvivesExport:
    """Every absorption law — Thorp, a Francois-Garrison row and profile,
    Biological, a constant — and a measured table come back from
    ``to_dict`` and NetCDF as the same law, under the law's field names."""

    @staticmethod
    def _envs():
        from uacpy.core.absorption import (
            Biological, ConstantAbsorption, FrancoisGarrison, Thorp)
        from uacpy.tests.conftest import (
            measured_absorption_table, two_layer_absorption)
        laws = {'thorp': Thorp(), 'fg': FrancoisGarrison(4.0, 34.5, 7.9),
                'bio': Biological(layers=[(30, 45, 1000, 5, 1)]),
                'const': ConstantAbsorption(0.02),
                'p': two_layer_absorption(), 't': measured_absorption_table(),
                'axes': FrancoisGarrison(
                    temperature=[(0.0, 20.0), (60.0, 9.0), (100.0, 8.0)],
                    salinity=35.0, pH=[(0.0, 8.1), (100.0, 7.9)],
                    ph_scale='total')}
        return [Environment(name=name, bathymetry=100.0, absorption=law)
                for name, law in laws.items()]

    @staticmethod
    def _same_law(a, b):
        from uacpy.core.absorption import Absorption
        assert isinstance(a, Absorption)
        if not hasattr(a, 'measured'):
            return a == b
        return (np.array_equal(a.measured.data, b.measured.data)
                and np.array_equal(a.measured.depths, b.measured.depths)
                and np.array_equal(a.measured.frequencies,
                                   b.measured.frequencies)
                and a.measured.units == b.measured.units
                and a.measured.model == b.measured.model)

    def test_to_dict(self):
        for env in self._envs():
            back = Environment.from_dict(env.to_dict())
            assert type(back.absorption) is type(env.absorption)
            assert self._same_law(back.absorption, env.absorption)

    def test_netcdf(self, tmp_path):
        pytest.importorskip('xarray')
        pytest.importorskip('h5netcdf')
        for env in self._envs():
            path = tmp_path / f'{env.name}.nc'
            env.to_netcdf(path, engine='h5netcdf')
            back = Environment.from_netcdf(path, engine='h5netcdf')
            assert type(back.absorption) is type(env.absorption)
            assert self._same_law(back.absorption, env.absorption)
