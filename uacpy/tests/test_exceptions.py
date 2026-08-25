"""Exception handling, the public exception types, and how the type is chosen.

The first half is the surface the caller touches: direct construction of the
typed classes, the exports, what the messages have to say, and pickling
through ``run_parallel``.

The second half is the rule that decides *which* type — provenance, not the
call site. Three parts, each of which the package stated somewhere and broke
somewhere else:

1. A missing file the *user* named is a ``ConfigurationError``; a missing file
   a *model* should have written is a ``FileFormatError``
   (:class:`~uacpy.core.exceptions.FileFormatError` says so). A decorator
   wrapped around readers of both kinds cannot tell them apart, so it converts
   neither and each reader states its own provenance.
2. ``DataFetchError`` is the data layer's exception whatever the reason, so an
   unreadable file in the *cache* — which ``./install.sh`` wrote, not the user
   — is one too.
3. A model refusing a legal input it cannot represent raises
   ``UnsupportedFeatureError``, the type whose remediation offers somewhere
   else to go; a genuinely illegal argument stays a ``ConfigurationError``.
"""

import inspect
import pickle
import re
import types
from pathlib import Path

import numpy as np
import pytest

import uacpy
from uacpy.core import Environment
from uacpy.core.exceptions import (
    UACPYError, InvalidDepthError, ExecutableNotFoundError,
    ModelExecutionError, UnsupportedFeatureError, ConfigurationError,
    DataFetchError, FileFormatError,
)
from uacpy.acoustic_signal.active import ambiguity_function, matched_filter
from uacpy.acoustic_signal.arrays import steering_vectors
from uacpy.acoustic_signal.waveforms import nwave
from uacpy.comms.channel_models import apply_fading_channel
from uacpy.comms.coding import deinterleave, interleave
from uacpy.comms.metrics import evm
from uacpy.models import Kraken
from uacpy.noise.marine_mammal import auditory_weighting


class TestCustomExceptions:
    """Direct construction of the typed exception classes."""

    def test_invalid_depth_error(self):
        error = InvalidDepthError(depth=150, max_depth=100, context="Source")
        assert isinstance(error, UACPYError)
        assert "150" in str(error)
        assert "100" in str(error)
        assert "Source" in str(error)

    def test_invalid_depth_error_subclass_of_uacpyerror(self):
        try:
            raise InvalidDepthError(depth=200, max_depth=100, context="Receiver")
        except UACPYError as e:
            assert "200" in str(e)

    def test_unsupported_feature_error(self):
        error = UnsupportedFeatureError(
            "Bellhop", "normal mode computation",
            alternatives=['Kraken', 'OASN'],
        )
        assert isinstance(error, UACPYError)
        msg = str(error)
        assert "Bellhop" in msg
        assert "normal mode" in msg.lower()


class TestExceptionPublicExports:
    """Every exception must be reachable from `uacpy` and `uacpy.core`."""

    def test_uacpy_top_level_exports(self):
        for name in ('UACPYError', 'InvalidDepthError', 'UnsupportedFeatureError',
                     'ConfigurationError', 'ExecutableNotFoundError',
                     'ModelExecutionError'):
            assert hasattr(uacpy, name), f"uacpy missing {name}"

    def test_uacpy_core_exports(self):
        import uacpy.core as core
        for name in ('UACPYError', 'InvalidDepthError', 'UnsupportedFeatureError',
                     'ConfigurationError', 'ExecutableNotFoundError',
                     'ModelExecutionError'):
            assert hasattr(core, name), f"uacpy.core missing {name}"

    def test_isinstance_through_uacpy(self):
        err = uacpy.InvalidDepthError(depth=150, max_depth=100, context='Source')
        assert isinstance(err, uacpy.UACPYError)


class TestInputValidation:
    """Constructor-time validation on Source / Receiver / Environment."""

    def test_negative_source_depth(self):
        with pytest.raises(ConfigurationError, match="source depths must be"):
            uacpy.Source(depths=-10, frequencies=100)

    def test_negative_receiver_depth(self):
        with pytest.raises(ConfigurationError, match="receiver depths must be"):
            uacpy.Receiver(depths=[-10], ranges=[1000])

    def test_zero_frequency_rejected(self):
        with pytest.raises(ConfigurationError, match="frequencies"):
            uacpy.Source(depths=50, frequencies=0)

    def test_negative_frequency_rejected(self):
        with pytest.raises(ConfigurationError, match="frequencies"):
            uacpy.Source(depths=50, frequencies=-100)

    def test_zero_environment_depth_rejected(self):
        with pytest.raises(ConfigurationError):
            uacpy.Environment(name='bad', bathymetry=0, ssp=1500)


class TestUnsupportedOperations:
    """Asking a model for something it can't do raises UnsupportedFeatureError."""

    @pytest.mark.requires_binary  # constructs Kraken (resolves its binary)
    def test_kraken_does_not_support_rays(self):
        kraken = Kraken(verbose=False)
        env = uacpy.Environment(name='t', bathymetry=100, ssp=1500)
        source = uacpy.Source(depths=50, frequencies=100)
        receiver = uacpy.Receiver(depths=[10], ranges=[1000])
        with pytest.raises(UnsupportedFeatureError):
            kraken.compute_rays(env, source, receiver)


class TestFieldErrors:
    """Result classes refuse operations that don't apply to their shape."""

    def test_rays_has_no_sel(self):
        from uacpy.core.results import Rays
        r = Rays(rays=[], model='Bellhop')
        with pytest.raises(AttributeError):
            r.at(range=1000, depth=50)

    def test_rays_has_no_to_db(self):
        from uacpy.core.results import Rays
        r = Rays(rays=[], model='Bellhop')
        with pytest.raises(AttributeError):
            r.to_db()


class TestErrorMessages:
    """Error messages should include enough information to act on."""

    def test_invalid_depth_message_helpful(self):
        error = InvalidDepthError(depth=150, max_depth=100, context="Source")
        msg = str(error)
        assert "150" in msg and "100" in msg

    def test_unsupported_feature_lists_alternatives(self):
        error = UnsupportedFeatureError(
            'Kraken', 'ray-path computation', alternatives=['Bellhop'],
        )
        assert 'Bellhop' in str(error)

    def test_remediation_renders_as_the_documented_how_to_fix_block(self):
        """``remediation=`` renders under the message as the 'How to fix:'
        block documented in docs/guide/io.md; without one, ``str()`` is the
        bare message."""
        from uacpy.core.exceptions import ConfigurationError, UACPYError
        err = ConfigurationError("bad knob", remediation="Turn the good knob.")
        assert str(err) == "bad knob\n\nHow to fix:\nTurn the good knob."
        assert err.remediation == "Turn the good knob."
        assert str(UACPYError("plain message")) == "plain message"


class TestValidationHelpers:
    """validate_inputs raises typed errors, not bare ValueError."""

    def test_source_deeper_than_env_raises_typed(self, simple_env):
        from uacpy.models.base import PropagationModel

        from uacpy.models.base import ModelSpec, RunMode

        # ``spec`` and ``source`` are required of any subclass that defines
        # run(); this double spawns no binary, and 'acoustics_toolbox' is
        # unrestricted so it emits no licence warning.
        Model = type('M', (PropagationModel,), {
            'spec': ModelSpec(modes=(RunMode.COHERENT_TL,)),
            'source': 'acoustics_toolbox',
            'run': lambda self, env, source, receiver, run_mode=None: None,
        })
        m = Model()
        source_deep = uacpy.Source(depths=150, frequencies=100)
        receiver = uacpy.Receiver(depths=[50], ranges=[1000])
        with pytest.raises(InvalidDepthError):
            m.validate_inputs(simple_env, source_deep, receiver)

    def test_receiver_deeper_than_env_warns_not_raises(self, simple_env):
        """Receivers below the model's resolvable depth are accepted with a
        warning (the model returns its below-domain value there), not
        rejected — unlike the source, which is a hard error."""
        import warnings
        from uacpy.models.base import PropagationModel

        from uacpy.models.base import ModelSpec, RunMode

        # ``spec`` and ``source`` are required of any subclass that defines
        # run(); this double spawns no binary, and 'acoustics_toolbox' is
        # unrestricted so it emits no licence warning.
        Model = type('M', (PropagationModel,), {
            'spec': ModelSpec(modes=(RunMode.COHERENT_TL,)),
            'source': 'acoustics_toolbox',
            'run': lambda self, env, source, receiver, run_mode=None: None,
        })
        m = Model()
        source = uacpy.Source(depths=50, frequencies=100)
        receiver_deep = uacpy.Receiver(depths=[150], ranges=[1000])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m.validate_inputs(simple_env, source, receiver_deep)
        assert any("below the model's resolvable depth" in str(w.message)
                   for w in caught)


# Typed exceptions must survive pickling so run_parallel returns the real
# per-job error instead of a BrokenProcessPool (these override __init__ with
# multi-positional / keyword-only signatures and need __reduce__).
@pytest.mark.parametrize("exc", [
    InvalidDepthError(99999.0, 4482.0, "Source depth"),
    ExecutableNotFoundError("Bellhop", "bellhop.exe", ["/a", "/b"]),
    ModelExecutionError("Kraken", -6, stdout="o", stderr="e"),
    UnsupportedFeatureError("Kraken", "elastic ice surface", ["Bellhop"]),
    UnsupportedFeatureError("OASR", "freq override", ["OASP"],
                            alternatives_label='run modes'),
])
def test_typed_exceptions_pickle_roundtrip(exc):
    back = pickle.loads(pickle.dumps(exc))
    assert type(back) is type(exc) and str(back) == str(exc)


# ── 1. a missing file is typed by who was supposed to write it ───────────────

class TestMissingFileProvenance:
    """``.bty`` / ``.ati`` / ``.sbp`` are decks the caller authored — no uacpy
    model writes one a uacpy reader reads back — so an absent one is a bad
    argument. Everything else these readers open is model output."""

    def test_a_missing_bty_the_caller_named_is_a_configuration_error(
            self, tmp_path):
        from uacpy.io.bathy_io import read_bathymetry
        with pytest.raises(ConfigurationError, match='not found'):
            read_bathymetry(tmp_path / 'absent.bty')

    def test_a_missing_ati_the_caller_named_is_a_configuration_error(
            self, tmp_path):
        from uacpy.io.bathy_io import read_altimetry
        with pytest.raises(ConfigurationError, match='not found'):
            read_altimetry(tmp_path / 'absent.ati')

    def test_a_missing_sbp_the_caller_named_is_a_configuration_error(
            self, tmp_path):
        from uacpy.io.refl_io import read_source_beam_pattern
        with pytest.raises(ConfigurationError, match='not found'):
            read_source_beam_pattern(tmp_path / 'absent.sbp')

    def test_both_sbp_entry_points_type_the_same_missing_path_alike(
            self, tmp_path):
        """``read_source_beam_pattern`` and ``stage_source_beam_pattern`` take
        the same user-supplied path; one raising ``FileFormatError`` and the
        other ``ConfigurationError`` for the identical condition made the type
        depend on which door the caller came through."""
        from uacpy.io.refl_io import (
            read_source_beam_pattern, stage_source_beam_pattern)
        absent = tmp_path / 'absent.sbp'
        with pytest.raises(ConfigurationError) as read_exc:
            read_source_beam_pattern(absent)
        with pytest.raises(ConfigurationError) as stage_exc:
            stage_source_beam_pattern(absent, tmp_path / 'out.sbp')
        assert type(read_exc.value) is type(stage_exc.value)

    def test_a_missing_brc_bounce_should_have_written_is_a_format_error(
            self, tmp_path):
        """The other side of the rule, and the reason it is not "every
        reflection table is the user's": ``Bounce`` calls this reader on the
        ``.brc`` it just wrote, and the message says "Run Bounce or OASR
        first". Its ``reflection_file=`` sibling — a path the user did supply
        — raises ``ConfigurationError``; two provenances, two types."""
        from uacpy.io.refl_io import read_reflection_coefficient
        with pytest.raises(FileFormatError, match='not found'):
            read_reflection_coefficient(tmp_path / 'absent.brc')


class TestTypedFormatErrorConvertsParsesOnly:
    """``typed_format_error`` sees only the exception, never the provenance,
    so blanket-converting ``FileNotFoundError`` gave a user's own missing
    ``.bty`` the type reserved for a failed model run."""

    def test_a_missing_file_reaches_the_caller_from_the_reader(self, tmp_path):
        from uacpy.io._fortran_helpers import typed_format_error

        @typed_format_error
        def read_undecided(path):
            with open(path):
                pass

        with pytest.raises(FileNotFoundError):
            read_undecided(tmp_path / 'absent.dat')

    def test_a_parse_failure_is_converted(self, tmp_path):
        from uacpy.io._fortran_helpers import typed_format_error

        @typed_format_error
        def read_with_parse_error(path):
            raise ValueError('invalid literal')

        with pytest.raises(FileFormatError, match='could not parse'):
            read_with_parse_error(tmp_path / 'x.dat')


def _model_output_readers():
    """``(label, call)`` for every reader whose file a model writes.

    Dropping the decorator's blanket conversion must not leave any of these
    leaking a bare ``FileNotFoundError``: each now states the same provenance
    explicitly.
    """
    from uacpy.io.bathy_io import read_bathymetry            # noqa: F401
    from uacpy.io.grn_reader import read_grn_file
    from uacpy.io.oalib_reader import (
        read_arr_file, read_flp, read_ray_file, read_rts_file, read_shd_bin,
        read_ssp_2d, read_ts)
    from uacpy.io.ramsurf_reader import read_pcomplex_grid, read_tl_grid
    from uacpy.io.refl_io import read_reflection_coefficient
    grid = dict(dr=1.0, ndr=1, dz=1.0, ndz=1)
    return [
        ('read_shd_bin', lambda p: read_shd_bin(str(p / 'a.shd'))),
        ('read_arr_file', lambda p: read_arr_file(p / 'a.arr')),
        ('read_ray_file', lambda p: read_ray_file(p / 'a.ray')),
        ('read_ssp_2d', lambda p: read_ssp_2d(p / 'a.ssp')),
        ('read_flp', lambda p: read_flp(p / 'a.flp')),
        ('read_rts_file', lambda p: read_rts_file(p / 'a.rts')),
        ('read_ts', lambda p: read_ts(p / 'a.ts')),
        ('read_grn_file', lambda p: read_grn_file(p / 'a.grn')),
        ('read_tl_grid', lambda p: read_tl_grid(p / 'tl.line', **grid)),
        ('read_pcomplex_grid',
         lambda p: read_pcomplex_grid(p / 'p.line', **grid)),
        ('read_reflection_coefficient',
         lambda p: read_reflection_coefficient(p / 'a.brc')),
    ]


@pytest.mark.parametrize('label,call', _model_output_readers(),
                         ids=[lbl for lbl, _ in _model_output_readers()])
def test_a_missing_model_output_is_a_file_format_error(label, call, tmp_path):
    with pytest.raises(FileFormatError, match='not found'):
        call(tmp_path)


# ── 2. the data layer has one exception ──────────────────────────────────────

class TestUnreadableSedimentCacheIsADataFetchError:
    """``grainsize.csv`` is written by ``./install.sh``, never by the user, so
    neither an unreadable header nor an unreadable row is a bad *argument*.
    Three exception types out of one file forced callers into a three-way
    ``except`` for a single decision — can I get a sediment value here?"""

    @staticmethod
    def _read(path):
        from uacpy.data.sediment_db import _index, _PHI_COLS, _phi_from_float
        return _index(path, _PHI_COLS, _phi_from_float)

    def test_a_csv_missing_its_columns_is_a_data_fetch_error(self, tmp_path):
        p = tmp_path / 'grainsize.csv'
        p.write_text('a,b,c\n1,2,3\n')
        with pytest.raises(DataFetchError, match='missing expected columns'):
            self._read(p)

    def test_a_csv_with_a_non_numeric_coordinate_is_a_data_fetch_error(
            self, tmp_path):
        p = tmp_path / 'grainsize.csv'
        p.write_text('latitude,longitude,mean_phi\nqqq,0.0,5.5\n')
        with pytest.raises(DataFetchError, match='non-numeric coordinate'):
            self._read(p)

    def test_one_unreadable_file_yields_one_exception_type(self, tmp_path):
        """The point of the retype: both failures of the same file are now
        catchable by the same ``except`` the rest of ``uacpy.data`` uses."""
        header = tmp_path / 'no_cols.csv'
        header.write_text('a,b,c\n1,2,3\n')
        row = tmp_path / 'bad_row.csv'
        row.write_text('latitude,longitude,mean_phi\nqqq,0.0,5.5\n')
        seen = set()
        for path in (header, row):
            with pytest.raises(DataFetchError) as exc:
                self._read(path)
            seen.add(type(exc.value))
        assert seen == {DataFetchError}


@pytest.fixture
def sediment_cache(tmp_path, monkeypatch):
    """``(write_grainsize_csv)`` over a cache root the sediment DB will read."""
    from uacpy.data import sediment_db
    root = tmp_path / 'data_cache'
    (root / 'sediment').mkdir(parents=True)
    monkeypatch.setenv('UACPY_DATA_CACHE', str(root))
    sediment_db._SAMPLES.clear()
    yield lambda text: (root / 'sediment' / 'grainsize.csv').write_text(text)
    sediment_db._SAMPLES.clear()


class TestATransectReportsAnUnreadableCacheAsItself:
    """Typing the cache failures ``DataFetchError`` put them in reach of
    ``range_dependent_bottom_along``'s per-waypoint handler, which reads that
    exception as "this waypoint is not covered" and fills the gap from a
    neighbour. An unreadable file raises it at *every* waypoint, so the
    all-gaps guard fired and told the user to change their transect while the
    remediation that would actually fix it — re-run the installer — was
    discarded. The source is read once before the loop instead, so the
    per-waypoint exception keeps a single meaning.
    """

    START, END = (50.0, 0.0), (50.5, 0.5)

    @staticmethod
    def _transect(**kw):
        from uacpy.data.sediment_db import fetch_bottom_local_transect
        return fetch_bottom_local_transect(
            TestATransectReportsAnUnreadableCacheAsItself.START,
            TestATransectReportsAnUnreadableCacheAsItself.END,
            n_points=4, **kw)

    @pytest.mark.parametrize('csv,fragment', [
        ('a,b,c\n1,2,3\n', 'missing expected columns'),
        ('latitude,longitude,mean_phi\nqqq,0.0,5.5\n', 'non-numeric coordinate'),
    ])
    def test_a_corrupt_cache_keeps_its_message_and_remediation(
            self, sediment_cache, csv, fragment):
        """Asserting the *type* alone would pass against the bug — the wrong
        diagnosis was a ``DataFetchError`` too. The message and the How-to-fix
        line are the things that were destroyed, so they are what is pinned."""
        sediment_cache(csv)
        with pytest.raises(DataFetchError) as exc:
            self._transect()
        assert fragment in str(exc.value)
        assert 'grainsize.csv' in str(exc.value)
        assert './install.sh --data sediment' in exc.value.remediation
        assert 'along the transect' not in str(exc.value)

    def test_a_corrupt_cache_reads_alike_from_a_point_and_a_transect(
            self, sediment_cache):
        from uacpy.data.sediment_db import fetch_bottom_local
        sediment_cache('a,b,c\n1,2,3\n')
        with pytest.raises(DataFetchError) as point:
            fetch_bottom_local(self.START)
        with pytest.raises(DataFetchError) as transect:
            self._transect()
        assert str(point.value) == str(transect.value)

    def test_an_uncovered_transect_keeps_the_coverage_diagnosis(
            self, sediment_cache):
        """The other half of the split, and what stops the fix from being
        "re-raise everything": with a readable cache holding one sample an
        ocean away, no waypoint is covered and the transect says so."""
        sediment_cache('latitude,longitude,mean_phi\n-40.0,-30.0,5.5\n')
        with pytest.raises(DataFetchError) as exc:
            self._transect(max_distance_km=250.0)
        assert 'no seabed data anywhere along the transect' in str(exc.value)
        assert 'grainsize.csv' not in str(exc.value)

    def test_a_covered_transect_returns_a_bottom(self, sediment_cache):
        """A readable, covering cache is untouched by the pre-flight read."""
        sediment_cache('latitude,longitude,mean_phi\n50.0,0.0,5.5\n')
        bottom = self._transect(max_distance_km=250.0)
        assert bottom.halfspace_at(range=0.0).sound_speed > 1400.0


# ── 3. a capability limit is an UnsupportedFeatureError ──────────────────────

class TestCapabilityLimitsAreUnsupportedFeatureErrors:
    """``'quad'``, a 3-D run, an option letter whose block uacpy never writes,
    a Source geometry a model does not implement — every one of these is a
    legal input the *model* cannot take, which is the decision
    ``except UnsupportedFeatureError`` exists to make."""

    @pytest.mark.parametrize('model', ['Scooter', 'SPARC'])
    def test_quad_ssp_interp_is_refused_by_capability(self, model):
        from uacpy.io.oalib_writer import reject_unsupported_ssp_interp
        with pytest.raises(UnsupportedFeatureError, match="'quad'") as exc:
            reject_unsupported_ssp_interp(model, 'quad')
        # The hand-rolled "Pick one of …" sentence is now the typed field, so
        # it renders under a How-to-fix heading like every other remediation.
        assert exc.value.alternatives_label == 'SSP interpolations'
        assert "'pchip'" in exc.value.remediation

    def test_kraken_quad_ssp_interp_is_refused_by_capability(self):
        from uacpy.models.kraken import Kraken
        stub = types.SimpleNamespace(interp_ssp='quad')
        with pytest.raises(UnsupportedFeatureError, match="'quad'") as exc:
            Kraken._check_kraken_ssp_type(stub)
        assert "'pchip'" in exc.value.remediation

    def test_a_supported_ssp_interp_passes(self):
        """Both guards fire on 'quad' alone — a retype that started raising on
        the default would pass every test above."""
        from uacpy.io.oalib_writer import reject_unsupported_ssp_interp
        from uacpy.models.kraken import Kraken
        assert reject_unsupported_ssp_interp('Scooter', 'pchip') is None
        assert Kraken._check_kraken_ssp_type(
            types.SimpleNamespace(interp_ssp=None)) is None

    @pytest.mark.parametrize('writer,options,cite', [
        ('write_oast_input', 'N J T E', 'unoast31.f:299'),
        ('write_oasp_input', 'N J d', 'unoasp22.f:127'),
        ('write_oasn_input', 'N J Z', 'unoasn22.f:322-323'),
        ('write_oassp_input', 'N J G', 'unoassp30.f:397-400'),
        ('write_oass_input', 'N Z', 'unoass21.f:318-320'),
    ])
    def test_an_unwritten_option_block_is_refused_by_capability(
            self, writer, options, cite):
        from uacpy.io.oases_writer import _reject_unwritten_option_blocks
        with pytest.raises(UnsupportedFeatureError) as exc:
            _reject_unwritten_option_blocks(writer, options)
        assert cite in str(exc.value)
        # The offered alternative is the caller's own string with the letters
        # taken out — GETOPT reads the record character by character, so that
        # is the whole edit.
        assert exc.value.alternatives_label == 'option strings'

    def test_a_supported_option_string_passes(self):
        from uacpy.io.oases_writer import _reject_unwritten_option_blocks
        assert _reject_unwritten_option_blocks(
            'write_oast_input', 'N J T') is None

    def test_bellhop_3d_is_refused_by_capability(self):
        """The guard precedes binary resolution in ``__init__``, so no binary
        is needed to reach it."""
        from uacpy.models.bellhop import Bellhop
        with pytest.raises(UnsupportedFeatureError, match='3D'):
            Bellhop(dimensionality='3D')

    def test_an_unknown_bellhop_backend_stays_a_configuration_error(self):
        """The neighbouring guard in the same ``__init__``: ``'cpu'`` is not a
        backend uacpy has, so it is an illegal argument rather than a
        capability this build lacks. Nothing in this round moves it."""
        from uacpy.models.bellhop import Bellhop
        with pytest.raises(ConfigurationError, match='known backend'):
            Bellhop(backend='cpu')


@pytest.mark.requires_binary
class TestModelCapabilityLimitsNeedingAnInstance:
    """The same rule at sites reachable only through a constructed model. No
    binary is launched: each guard raises while validating or while writing
    the deck."""

    def test_an_unsupported_source_geometry_is_refused_by_capability(self):
        import uacpy
        from uacpy.core.exceptions import ExecutableNotFoundError
        from uacpy.models.ram import RAM
        try:
            model = RAM()
        except ExecutableNotFoundError:
            pytest.skip('RAM binary not available')
        env = uacpy.Environment(name='geom', bathymetry=200.0, ssp=1500.0)
        rcv = uacpy.Receiver(depths=[50.0], ranges=[1000.0])
        src = uacpy.Source(depths=25.0, frequencies=100.0, source_type='line')
        with pytest.raises(UnsupportedFeatureError,
                           match='source_type') as exc:
            model.validate_inputs(env, src, rcv)
        assert "'point'" in exc.value.remediation

    def test_a_reflection_table_bottom_is_refused_by_sparc(self, tmp_path):
        """``_validate_acoustic_type`` has already rejected unrecognised
        names and ``_sparc_rigidify_halfspace`` has already converted a
        halfspace, so what reaches SPARC's bottom branch is a table bottom —
        legal everywhere else, unrepresentable in a Vacuum/Rigid-only deck."""
        import uacpy
        from uacpy.core.bottom import BoundaryProperties
        from uacpy.core.exceptions import ExecutableNotFoundError
        from uacpy.models.sparc import SPARC
        try:
            model = SPARC()
        except ExecutableNotFoundError:
            pytest.skip('SPARC binary not available')
        env = uacpy.Environment(
            name='table_bottom', bathymetry=100.0, ssp=1500.0,
            bottom=BoundaryProperties(acoustic_type='file'))
        src = uacpy.Source(depths=50.0, frequencies=50.0)
        rcv = uacpy.Receiver(depths=[50.0], ranges=np.array([500.0]))
        with pytest.raises(UnsupportedFeatureError, match='file') as exc:
            model._write_sparc_env(tmp_path / 'x.env', env, src, rcv)
        assert 'Kraken' in exc.value.remediation


def test_data_fetch_error_docstring_covers_local_reads():
    doc = inspect.getdoc(DataFetchError).lower()
    assert 'data layer' in doc
    assert 'local' in doc


def test_a_local_only_reader_raises_data_fetch_error_without_a_network_import():
    """What the docstring now claims, checked against the code: the local
    GEBCO reader raises ``DataFetchError`` and imports nothing that fetches."""
    source = Path(uacpy.__file__).resolve().parent / 'data' / 'gebco_local.py'
    text = source.read_text(encoding='utf-8')
    assert 'DataFetchError' in text
    network = re.findall(
        r'^\s*(?:import|from)\s+(requests|urllib|http|aiohttp|ftplib)\b',
        text, flags=re.MULTILINE)
    assert network == []


def _bare_model():
    """A model that spawns no binary, with altimetry unsupported (the
    ``PropagationModel`` default)."""
    from uacpy.models.base import PropagationModel

    from uacpy.models.base import ModelSpec, RunMode

    class _Bare(PropagationModel):
        # ``spec`` and ``source`` are required of any subclass that defines
        # run(); this double spawns no binary, so the source id only has to
        # be a real one — 'acoustics_toolbox' is unrestricted, so nothing
        # here emits a licence warning.
        spec = ModelSpec(modes=(RunMode.COHERENT_TL,))
        source = 'acoustics_toolbox'

        def run(self, env, source, receiver, run_mode=None):
            return self._project_environment(env)

    return _Bare()


def test_unknown_altimetry_collapse_raises_when_the_private_dict_is_mutated():
    """The only way in — which is why the branch needs a comment saying so."""
    model = _bare_model()
    model._collapse['altimetry'] = 'flatten'
    env = Environment(name='alt', bathymetry=100.0, ssp=1500.0,
                      altimetry=[(0.0, -2.0)])
    with pytest.raises(ConfigurationError,
                       match=r"Unknown collapse\['altimetry'\]"):
        model._project_environment(env)


def test_the_unreachable_altimetry_branch_is_marked_defensive():
    """Without the comment the next reader finds an uncovered raise with no
    way to trigger it and deletes it — the branch is what makes adding a
    second altimetry method a ``ConfigurationError`` rather than a silent
    drop."""
    from uacpy.models.base import PropagationModel

    source = inspect.getsource(PropagationModel._project_environment)
    lines = source.splitlines()
    raise_at = next(i for i, line in enumerate(lines)
                    if "Unknown collapse['altimetry']" in line)
    preamble = '\n'.join(lines[max(0, raise_at - 12):raise_at])
    assert 'Defensive' in preamble, preamble


class TestOasesReadersRaiseTheirOwnErrorType:
    """Five of the six OASES readers check the path before opening it; the
    ``.rhs`` header reader leaked ``FileNotFoundError`` / ``IsADirectoryError``
    instead."""

    def test_a_missing_rhs_is_a_file_format_error(self, tmp_path):
        from uacpy.io.oases_reader import read_oases_rhs_header
        with pytest.raises(FileFormatError, match='not found'):
            read_oases_rhs_header(tmp_path / 'absent.rhs')

    def test_a_directory_is_a_file_format_error(self, tmp_path):
        from uacpy.io.oases_reader import read_oases_rhs_header
        with pytest.raises(FileFormatError, match='not found'):
            read_oases_rhs_header(tmp_path)


class TestTypedErrorsReplaceBareOnes:
    """Cases that used to surface as a numpy/Python error naming none of the
    arguments, or as a plausible-looking number."""

    def test_zero_energy_replica_raises_instead_of_returning_nan(self):
        with pytest.raises(ConfigurationError, match="energy"):
            matched_filter(np.ones(8), np.zeros(4))
        # normalize=False is the un-normalised correlation and stays valid.
        assert np.all(matched_filter(np.ones(8), np.zeros(4),
                                     normalize=False) == 0.0)

    def test_zero_energy_waveform_raises_instead_of_a_nan_surface(self):
        with pytest.raises(ConfigurationError, match="energy"):
            ambiguity_function(np.zeros(8), 1000.0, n_doppler=3)

    @pytest.mark.parametrize("depth", [0, -2])
    def test_interleaver_depth_below_one_is_typed(self, depth):
        with pytest.raises(ConfigurationError, match="depth"):
            interleave(np.ones(8, dtype=int), depth)
        with pytest.raises(ConfigurationError, match="depth"):
            deinterleave(np.ones(8, dtype=int), depth)

    def test_evm_against_a_zero_energy_reference_is_typed(self):
        with pytest.raises(ConfigurationError, match="energy"):
            evm(np.ones(4, complex), np.zeros(4, complex))

    def test_negative_tap_delay_is_rejected(self):
        # y[di:di+n] indexes from the end for a negative di: delays [10, -8]
        # on a 6-sample input placed the second echo at samples 8-13.
        x = np.ones(6, complex)
        taps = np.ones((2, 10), complex)
        with pytest.raises(ConfigurationError, match="delays_samples"):
            apply_fading_channel(x, taps, [10, -8])
        assert apply_fading_channel(x, taps, [0, 3]).size == 9

    @pytest.mark.parametrize("frequency", [-100.0, 0.0])
    def test_auditory_weighting_rejects_non_positive_frequency(self, frequency):
        # The weighting is even in f, so a negative frequency squared away and
        # came back as a plausible finite value; WenzNoise rejects f <= 0 for
        # exactly this reason.
        with pytest.raises(ConfigurationError, match="> 0 Hz"):
            auditory_weighting(frequency, 'LF')

    def test_steering_vectors_rejects_multidimensional_positions(self):
        # np.outer flattens, so an (N, 2) coordinate array silently produced a
        # (n_angles, 2N) manifold for an array that does not exist.
        with pytest.raises(ConfigurationError, match="1-D"):
            steering_vectors(np.arange(8.0).reshape(4, 2), [0.0, 10.0], 1000.0)
        assert steering_vectors(np.arange(4.0), [0.0, 10.0], 1000.0).shape == (2, 4)

    def test_nwave_accepts_a_scalar_time(self):
        # Boolean-mask assignment on a 0-d result raised a bare TypeError,
        # where ricker_wavelet and sparc_pulse both take scalars.
        assert float(nwave(0.001, 100.0)) == pytest.approx(
            np.sin(2 * np.pi * 100.0 * 0.001)
            - 0.5 * np.sin(4 * np.pi * 100.0 * 0.001))
        assert float(nwave(-0.001, 100.0)) == 0.0
        assert float(nwave(0.05, 100.0)) == 0.0
