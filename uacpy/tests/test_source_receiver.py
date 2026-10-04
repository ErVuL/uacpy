"""``Source`` and ``Receiver``.

Construction and its refusals, source geometry and beam pattern, distinct
frequencies, the line-receiver recipe, and the constructor annotations the
type checker reads.
"""

import numpy as np
import pytest
import uacpy
import warnings
from uacpy.core.boundary import BoundaryProperties
from uacpy.core.deck_limits import SBP_ANGLE_RESOLUTION_DEG
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.source import Source


class TestSource:
    """Tests for Source class."""

    def test_create_source(self, source):
        """Test creating a source."""
        assert source.depths[0] == 50.0
        assert source.frequencies[0] == 100.0

    def test_source_array_conversion(self):
        """Test that single values are converted to arrays."""
        source = uacpy.Source(depths=30.0, frequencies=200.0)
        assert isinstance(source.depths, np.ndarray)
        assert isinstance(source.frequencies, np.ndarray)
        assert len(source.depths) == 1
        assert len(source.frequencies) == 1

    def test_multiple_sources(self):
        """Test multiple source depths."""
        source = uacpy.Source(depths=[10.0, 20.0, 30.0], frequencies=100.0)
        assert len(source.depths) == 3
        assert np.allclose(source.depths, [10, 20, 30])

    def test_multiple_frequencies(self):
        """Test multiple frequencies."""
        source = uacpy.Source(depths=50.0, frequencies=[50.0, 100.0, 200.0])
        assert len(source.frequencies) == 3

    def test_source_depths_must_be_strictly_increasing(self):
        """Multi-element source depths must be sorted (matches Receiver), so
        output rows indexed by source depth stay unambiguous across models."""
        with pytest.raises(ConfigurationError, match="strictly increasing"):
            uacpy.Source(depths=[30.0, 10.0, 20.0], frequencies=100.0)


@pytest.mark.parametrize("ctor,kwargs", [
    # Source / Receiver reject NaN or inf in any
    # ``depths``/``frequencies``/``ranges`` array.
    (uacpy.Source, dict(depths=[10, np.nan], frequencies=100)),
    (uacpy.Source, dict(depths=10, frequencies=[100, np.nan])),
    (uacpy.Receiver, dict(depths=[10, 20], ranges=[100, np.nan])),
    (uacpy.Receiver, dict(depths=[np.nan], ranges=[100])),
    (uacpy.Source, dict(depths=[10, np.inf], frequencies=100)),
    (uacpy.Receiver, dict(depths=[10], ranges=[np.inf])),
])
def test_source_receiver_reject_non_finite(ctor, kwargs):
    """Source and Receiver reject NaN / inf at construction so
    non-finite values cannot leak into env-file writers."""
    with pytest.raises(ConfigurationError, match="finite"):
        ctor(**kwargs)


class TestReceiver:
    """Tests for Receiver class."""

    def test_create_receiver_grid(self, receiver_grid):
        """Test creating receiver grid."""
        assert len(receiver_grid.depths) == 9
        assert len(receiver_grid.ranges) == 11
        assert receiver_grid.depth_min == 10.0
        assert receiver_grid.depth_max == 90.0
        assert receiver_grid.range_min == 100.0
        assert receiver_grid.range_max == 5000.0

    def test_small_receiver_grid(self, receiver_small):
        """Test small receiver grid."""
        assert len(receiver_small.depths) == 3
        assert len(receiver_small.ranges) == 3

    def test_receiver_line_array(self):
        """Test line array receiver."""
        receiver = uacpy.Receiver(
            depths=[50.0],
            ranges=np.linspace(1000, 10000, 100)
        )
        assert len(receiver.depths) == 1
        assert len(receiver.ranges) == 100

    def test_receiver_has_no_sampling_layout_parameter(self):
        """A Receiver is the depth x range grid and nothing else: the
        one-value ``receiver_type`` slot is gone, so passing it is a
        TypeError."""
        with pytest.raises(TypeError, match='receiver_type'):
            uacpy.Receiver(depths=50, ranges=1000, receiver_type='grid')

    def test_omitted_ranges_default_to_source_point_with_warning(self):
        """``Receiver(depths=50)`` defaults ranges to a single point at 0 m
        (the source location) and warns, because r=0 is singular for
        TL/pressure runs."""
        with pytest.warns(UserWarning, match="ranges not given"):
            rx = uacpy.Receiver(depths=50.0)
        np.testing.assert_array_equal(rx.ranges, [0.0])
        np.testing.assert_array_equal(rx.depths, [50.0])


class TestSourceGeometryAndBeamPattern:
    """Source owns source geometry and directivity (spec 2026-07-25)."""

    def test_scaled_is_a_valid_source_type(self):
        src = uacpy.Source(depths=50, frequencies=100, source_type='scaled')
        assert src.source_type == 'scaled'

    def test_unknown_source_type_raises(self):
        with pytest.raises(ConfigurationError, match="source_type"):
            uacpy.Source(depths=50, frequencies=100, source_type='Z')

    def test_beam_pattern_defaults_to_none(self):
        assert uacpy.Source(depths=50, frequencies=100).beam_pattern is None

    def test_beam_pattern_accepts_angle_level_array(self):
        pat = np.array([[-90.0, -20.0], [0.0, 0.0], [90.0, -20.0]])
        src = uacpy.Source(depths=50, frequencies=100, beam_pattern=pat)
        assert src.beam_pattern.shape == (3, 2)

    def test_beam_pattern_wrong_shape_raises(self):
        with pytest.raises(ConfigurationError, match=r"N, 2"):
            uacpy.Source(depths=50, frequencies=100,
                         beam_pattern=np.array([1.0, 2.0, 3.0]))

    def test_beam_pattern_non_monotonic_angles_raise(self):
        # misc/beampattern.f90:56-57 rejects this with ERROUT, which gfortran
        # exits 0 on; catching it here is the point of validating in Python.
        pat = np.array([[0.0, 0.0], [-90.0, -20.0], [90.0, -20.0]])
        with pytest.raises(
                ConfigurationError,
                match='beam-pattern angles must be strictly increasing'):
            uacpy.Source(depths=50, frequencies=100, beam_pattern=pat)

    def test_beam_pattern_single_row_raises(self):
        # bellhop.f90:273 interpolates between rows IBP and IBP+1 after clamping
        # IBP to NSBPPts-1, so one row makes it read below the bound allocated at
        # misc/beampattern.f90:36 and return an all-NaN field with exit code 0.
        # misc/monotonicMod.f90:30-31 returns .TRUE. for N==1, so the engine's own
        # guard cannot catch it either.
        with pytest.raises(ConfigurationError, match="at least 2"):
            uacpy.Source(depths=50, frequencies=100,
                         beam_pattern=np.array([[0.0, 0.0]]))

    def test_beam_pattern_path_is_stored_as_path(self, tmp_path):
        from pathlib import Path
        sbp = tmp_path / 'pattern.sbp'
        sbp.write_text("2\n-90.0 0.0\n90.0 0.0\n")
        src = uacpy.Source(depths=50, frequencies=100, beam_pattern=sbp)
        assert isinstance(src.beam_pattern, Path)

    def test_copy_carries_geometry_and_pattern(self):
        pat = np.array([[-90.0, -20.0], [90.0, 0.0]])
        src = uacpy.Source(depths=50, frequencies=100,
                           source_type='line', beam_pattern=pat)
        dup = src.copy()
        assert dup.source_type == 'line'
        np.testing.assert_array_equal(dup.beam_pattern, pat)


class TestBeamPatternAnglesCarryTheSbpResolution:
    """``write_source_beam_pattern`` prints the angle column at ``%12.6f``
    and refuses a pair closer than 1e-6 deg; the carrier now refuses the
    same pattern at construction rather than at write."""

    def test_angles_closer_than_the_sbp_resolution_are_rejected(self):
        with pytest.raises(ConfigurationError,
                           match="must increase by more than"):
            Source(depths=[10.0], frequencies=[100.0],
                   beam_pattern=[[0.0, 0.0], [1e-9, -3.0], [10.0, -6.0]])

    def test_the_angle_step_is_reported_in_degrees(self):
        with pytest.raises(ConfigurationError,
                           match='angles must increase by more than') as exc:
            Source(depths=[10.0], frequencies=[100.0],
                   beam_pattern=[[0.0, 0.0], [1e-9, -3.0]])
        assert 'deg apart' in str(exc.value)
        assert 'm apart' not in str(exc.value)

    def test_a_metre_axis_keeps_reporting_metres(self):
        with pytest.raises(ConfigurationError,
                           match='depths must increase by more than') as exc:
            Source(depths=[25.0, 25.0 + 1e-7], frequencies=[200.0])
        assert 'm apart' in str(exc.value)

    def test_a_resolvable_pattern_is_accepted(self):
        src = Source(depths=[10.0], frequencies=[100.0],
                     beam_pattern=[[-90.0, -20.0], [0.0, 0.0], [90.0, -20.0]])
        assert src.beam_pattern.shape == (3, 2)

    def test_the_constant_matches_the_sbp_writers_bound(self):
        assert SBP_ANGLE_RESOLUTION_DEG == 1e-6

    def test_a_step_of_exactly_the_resolution_is_rejected_by_the_carrier(self):
        # Source.__post_init__ asks _require_strictly_increasing for steps
        # greater than the resolution, so the carrier refuses the one step
        # write_source_beam_pattern accepts (its own guard is `< resolution`).
        # The two conventions differ by exactly this value, deliberately: the
        # carrier is the stricter of the two.
        with pytest.raises(ConfigurationError,
                           match="must increase by more than"):
            Source(depths=[10.0], frequencies=[100.0],
                   beam_pattern=[[0.0, 0.0], [SBP_ANGLE_RESOLUTION_DEG, -3.0]])

    def test_a_step_just_above_the_resolution_is_accepted(self):
        src = Source(depths=[10.0], frequencies=[100.0],
                     beam_pattern=[[0.0, 0.0],
                                   [2 * SBP_ANGLE_RESOLUTION_DEG, -3.0]])
        assert src.beam_pattern[1, 0] == pytest.approx(2e-6)


# ── the two roles of a dataclass field annotation ───────────────────────────
#
# ``Source``, ``Receiver`` and ``BoundaryProperties`` all take a wide input and
# normalize it in ``__post_init__``, so the constructor parameter and the
# attribute have different types. A dataclass writes one annotation for both
# roles, and the wider of the two used to win: ``s.depths`` was declared
# ``float | list[float] | ndarray``, so ``len(s.depths)``, ``s.depths.shape``
# and ``for d in r.depths`` were reported as errors in downstream code that
# runs correctly — on a package that ships ``py.typed`` and therefore has its
# annotations believed. The field now declares what the attribute holds, and an
# ``if TYPE_CHECKING:`` ``__init__`` carries the input union.
#
# (class, field, attribute type, constructor annotation, constructor kwargs)
_SPLIT_ANNOTATION_FIELDS = [
    (Source, 'depths', np.ndarray,
     'Union[float, List[float], np.ndarray]',
     dict(depths=50.0, frequencies=100.0)),
    (Source, 'frequencies', np.ndarray,
     'Union[float, List[float], np.ndarray]',
     dict(depths=50.0, frequencies=[100.0, 200.0])),
    (uacpy.Receiver, 'depths', np.ndarray,
     'Union[float, List[float], np.ndarray]',
     dict(depths=[10.0, 20.0], ranges=1000.0)),
    (uacpy.Receiver, 'ranges', np.ndarray,
     'Optional[Union[float, List[float], np.ndarray]]',
     dict(depths=10.0, ranges=[1000.0, 2000.0])),
    (BoundaryProperties, 'acoustic_type', str,
     'Optional[str]',
     dict(sound_speed=1600.0, density=1.8)),
]


_SPLIT_IDS = [f'{cls.__name__}.{field}'
              for cls, field, _, _, _ in _SPLIT_ANNOTATION_FIELDS]


def _type_checking_init(cls):
    """The ``__init__`` that declares the class's constructor types.

    Either the one declared inside ``if TYPE_CHECKING:`` or one written in
    the class body. Read from source: it is what a static checker reads,
    while the runtime ``__init__`` is ``@carrier``'s, whose signature states
    the constructor types through ``init_annotations``."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(inspect.getmodule(cls)))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ClassDef) and node.name == cls.__name__):
            continue
        for child in node.body:
            if (isinstance(child, ast.FunctionDef)
                    and child.name == '__init__'):
                return child
            if not isinstance(child, ast.If):
                continue
            test = child.test
            guard = (test.id if isinstance(test, ast.Name)
                     else test.attr if isinstance(test, ast.Attribute) else '')
            if not guard.endswith('TYPE_CHECKING'):
                continue
            for inner in child.body:
                if (isinstance(inner, ast.FunctionDef)
                        and inner.name == '__init__'):
                    return inner
    return None


@pytest.mark.parametrize(
    'cls, field, attribute_type, constructor_annotation, kwargs',
    _SPLIT_ANNOTATION_FIELDS, ids=_SPLIT_IDS)
def test_a_normalized_field_is_annotated_as_what_the_attribute_holds(
        cls, field, attribute_type, constructor_annotation, kwargs):
    """Both halves of the split, so neither can drift back into the other.

    The attribute annotation has to name the normalized type *and* the
    constructed object has to carry it — an annotation narrowed without the
    normalization behind it is the same defect pointing the other way."""
    import typing

    declared = typing.get_type_hints(cls)[field]
    assert declared is attribute_type, (
        f"{cls.__name__}.{field} is annotated {declared!r}, not "
        f"{attribute_type!r}: every downstream read of the attribute is "
        f"checked against the constructor's input type")

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        carrier = cls(**kwargs)
    assert isinstance(getattr(carrier, field), attribute_type), (
        f"{cls.__name__}.{field} is annotated {attribute_type!r} but holds "
        f"{type(getattr(carrier, field))!r} after construction")


_SPLIT_CLASSES = [Source, uacpy.Receiver, BoundaryProperties]


@pytest.mark.parametrize('cls', _SPLIT_CLASSES,
                         ids=[c.__name__ for c in _SPLIT_CLASSES])
def test_the_type_checking_constructor_matches_the_one_python_compiles(cls):
    """The guarded ``__init__`` and the one the decorator compiles have to be
    the same function, parameter for parameter.

    Two ways they come apart, and this sees both. A field added to the
    dataclass and not to the block silently drops out of every static caller's
    view of the constructor. And the runtime ``__init__`` takes its
    annotations from the *fields*, so without ``@carrier``'s
    ``init_annotations`` ``inspect.signature`` and ``help()`` advertise a
    default the annotation refuses — ``ranges: np.ndarray = None``.
    """
    import ast
    import inspect

    initializer = _type_checking_init(cls)
    assert initializer is not None, cls.__name__
    guarded = [a.arg for a in initializer.args.args if a.arg != 'self']
    compiled = list(inspect.signature(cls).parameters)
    assert guarded == compiled, (
        f"{cls.__name__}'s `if TYPE_CHECKING:` __init__ takes {guarded} and "
        f"the compiled one takes {compiled}")

    namespace = vars(inspect.getmodule(cls))
    padding = [None] * (len(initializer.args.args)
                        - len(initializer.args.defaults))
    declared_defaults = padding + list(initializer.args.defaults)
    for argument, default in zip(initializer.args.args, declared_defaults):
        if argument.arg == 'self':
            continue
        parameter = inspect.signature(cls).parameters[argument.arg]
        assert parameter.annotation == eval(  # noqa: S307 — our own source
            ast.unparse(argument.annotation), namespace), (
            f"{cls.__name__}.__init__ advertises "
            f"{parameter.annotation!r} for {argument.arg}, not the "
            f"{ast.unparse(argument.annotation)!r} the guarded block declares")
        if default is None:
            assert parameter.default is inspect.Parameter.empty, (
                f"{cls.__name__}.__init__ gives {argument.arg} a default the "
                f"guarded block does not")
        else:
            assert repr(parameter.default) == ast.unparse(default), (
                f"{cls.__name__}.__init__ defaults {argument.arg} to "
                f"{parameter.default!r}, not {ast.unparse(default)}")


@pytest.mark.parametrize(
    'cls, field, attribute_type, constructor_annotation, kwargs',
    _SPLIT_ANNOTATION_FIELDS, ids=_SPLIT_IDS)
def test_the_constructor_declares_the_input_union_the_docstring_documents(
        cls, field, attribute_type, constructor_annotation, kwargs):
    """The other half. Narrowing the field annotation without this block would
    make ``Source(depths=50.0)`` — the spelling every docstring example and
    the whole test suite uses — a type error for a downstream caller."""
    import ast

    initializer = _type_checking_init(cls)
    assert initializer is not None, (
        f"{cls.__name__} declares no `if TYPE_CHECKING:` __init__ and writes "
        f"none out, so its constructor is typed from the fields and rejects "
        f"the input union its Parameters section documents")
    annotations = {arg.arg: ast.unparse(arg.annotation)
                   for arg in initializer.args.args if arg.annotation}
    assert annotations.get(field) == constructor_annotation, (
        f"{cls.__name__}.__init__ annotates {field} as "
        f"{annotations.get(field)!r}, not {constructor_annotation!r}")


class TestSourceFrequenciesAreDistinct:
    """Any order, never twice (CORE-4): a result axis follows the caller's
    order, and a duplicated bin is two identical slices under one label."""

    def test_unsorted_is_kept_in_order(self):
        from uacpy import Source
        assert Source(depths=50.0, frequencies=[300.0, 100.0]).frequencies \
            .tolist() == [300.0, 100.0]

    def test_a_repeated_frequency_is_refused(self):
        from uacpy import Source
        from uacpy.core.exceptions import ConfigurationError
        with pytest.raises(ConfigurationError, match=r"\[100\.0\]"):
            Source(depths=50.0, frequencies=[300.0, 100.0, 100.0])


class TestTheLineReceiverRecipeCoversATrack:
    """The Receiver docstring names a recipe for paired track samples.
    The recipe must work for the use case it names, a glider whose depth
    yo-yos along range, which the strictly-increasing ``Receiver`` axes
    cannot hold as paired arrays."""

    track_depths = np.array([10.0, 50.0, 20.0, 60.0, 50.0])
    track_ranges = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

    def test_the_docstring_names_the_unique_inverse_recipe(self):
        assert 'return_inverse' in uacpy.Receiver.__doc__

    def test_the_recipe_returns_one_value_per_track_point(self):
        zu, iz = np.unique(self.track_depths, return_inverse=True)
        ru, ir = np.unique(self.track_ranges, return_inverse=True)
        rcv = uacpy.Receiver(depths=zu, ranges=ru)
        grid = rcv.depths[:, None] * 1000.0 + rcv.ranges[None, :]
        paired = grid[iz, ir]
        np.testing.assert_array_equal(
            paired, self.track_depths * 1000.0 + self.track_ranges)

    def test_the_class_docstring_describes_only_the_grid(self):
        summary = uacpy.Receiver.__doc__.split('Parameters')[0]
        assert 'line-type' not in summary


class TestAxesAreOneDimensional:
    """A Receiver is the cross-product of two 1-D axes and a Source lists
    1-D depths and frequencies; a 2-D array is refused at construction
    rather than stored and crashing an engine later."""

    @pytest.mark.parametrize('axis', ['depths', 'ranges'])
    def test_a_receiver_axis_of_two_dimensions_is_refused(self, axis):
        kw = {'depths': [30.0, 60.0], 'ranges': [500.0, 1000.0]}
        kw[axis] = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ConfigurationError,
                           match=f'Receiver.{axis} must be a scalar or a '
                                 f'1-D axis; got shape \\(2, 2\\)'):
            uacpy.Receiver(**kw)

    def test_a_meshgrid_pair_is_refused_naming_the_axes(self):
        rr, zz = np.meshgrid([500.0, 1000.0], [30.0, 60.0])
        with pytest.raises(ConfigurationError, match='not a meshgrid'):
            uacpy.Receiver(depths=zz, ranges=rr)

    @pytest.mark.parametrize('axis', ['depths', 'frequencies'])
    def test_a_source_axis_of_two_dimensions_is_refused(self, axis):
        kw = {'depths': 10.0, 'frequencies': 100.0}
        kw[axis] = [[10.0], [20.0]]
        with pytest.raises(ConfigurationError,
                           match=f'Source {axis} must be a scalar or a 1-D '
                                 f'vector; got shape \\(2, 1\\)'):
            Source(**kw)

    def test_scalars_and_one_dimensional_axes_are_kept(self):
        r = uacpy.Receiver(depths=30.0, ranges=np.array([500.0, 1000.0]))
        assert r.depths.shape == (1,) and r.ranges.shape == (2,)
        s = Source(depths=[10.0, 20.0], frequencies=100.0)
        assert s.depths.shape == (2,) and s.frequencies.shape == (1,)
