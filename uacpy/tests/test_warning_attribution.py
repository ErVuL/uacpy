"""Which file a warning names, and why it has to be the caller's.

A warning raised inside a dataclass's ``__post_init__``, or inside a reader
several layers down, must name the line the *user* wrote. When it names a
library line instead — or the ``<string>`` pseudo-file a dataclass's generated
``__init__`` lives in — the warnings module's once-per-location dedup keys
every call site to that one line, and the second, tenth and hundredth caller
are told nothing at all.

So these are not cosmetic. Each test below compares ``warning.filename``
against *this file*: the test module is the caller, so its own path is the
answer the attribution machinery has to produce. That makes them portable —
the assertion is "the caller's file", not any particular filename.

The rest of the file holds the shared ``USER_FRAME_SKIP`` prefix set that the
attribution walk uses to decide which frames are library and which are the
user's. Its two failure modes both really happened: a prefix that lost its
trailing separator matched a *sibling* directory (``uacpy_venv/`` under a
``uacpy`` prefix), and a second copy of the set drifted out of step with the
first, so two call sites disagreed about where the library ended.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import tempfile
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

import uacpy
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.acoustic_signal.constant_q import constant_q_transform
from uacpy.core.absorption import Biological, BiologicalLayer
from uacpy.core.acoustics import soundspeed_delgrosso
from uacpy.core.receiver import Receiver
from uacpy.io import grn_reader

_THIS_FILE = Path(__file__).resolve()
_PACKAGE_DIR = Path(uacpy.__file__).resolve().parent


def _only_warning(record):
    """The single warning in ``record``, with the count asserted."""
    assert len(record) == 1, [str(w.message) for w in record]
    return record[0]


_TWO_FRAME_SOURCE = '''\
import warnings

SKIP = ()


def inner():
    warnings.warn('probe', UserWarning, skip_file_prefixes=SKIP)


def outer():
    inner()
'''


def _write_two_frame_module():
    """A throwaway module holding a two-frame chain above a ``warnings.warn``.

    Its own path stands in for a package module's, so the skip rule can be
    read off the interpreter with a chain deep enough to tell "skipped one
    frame" from "walked out to the caller"."""
    directory = tempfile.mkdtemp()
    path = Path(directory) / 'two_frame_probe.py'
    path.write_text(_TWO_FRAME_SOURCE, encoding='utf-8')
    spec = importlib.util.spec_from_file_location('two_frame_probe', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_receiver_default_ranges_warning_names_the_callers_file():
    """``Receiver(depths=…)`` with no ``ranges`` is a documented, supported
    path (docs/guide/source-receiver.md §"``Receiver`` without ``ranges``
    warns"), so its warning is raised on every such call and must land on the
    caller, not on the ``<string>`` frame of the generated ``__init__``."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        Receiver(depths=[10.0])
    warning = _only_warning(record)
    assert warning.filename != '<string>'
    assert Path(warning.filename).resolve() == _THIS_FILE


def test_biological_layer_ceiling_warning_names_the_callers_file():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        BiologicalLayer(10.0, 20.0, 100.0, 100.0, 100.0)
    warning = _only_warning(record)
    assert warning.filename != '<string>'
    assert Path(warning.filename).resolve() == _THIS_FILE


def test_a_layer_built_from_a_tuple_also_names_the_callers_file():
    """The layer's second door, and the one no frame count could reach.

    ``Biological`` turns a 5-tuple into a ``BiologicalLayer`` inside its own
    constructor, one frame below a direct ``BiologicalLayer(...)``. While both
    constructors were generated, the two paths stacked one and two ``<string>``
    frames respectively: ``stacklevel=3`` was right for the direct call and
    measured wrong here (it needed 5), and ``skip_file_prefixes`` could not
    stand in for either, because the walk stops on ``<string>`` — it matches no
    package prefix. Writing both ``__init__``s out removed the generated frames
    and let one mechanism serve both depths."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        Biological(layers=[(10.0, 20.0, 100.0, 100.0, 100.0)])
    warning = _only_warning(record)
    assert warning.filename != '<string>'
    assert Path(warning.filename).resolve() == _THIS_FILE


def test_each_biological_call_site_warns_under_the_once_per_location_filter():
    """The consequence of the misattribution, measured before the fix: with
    every tuple layer's warning keyed to the normalising loop's own line,
    ``'default'`` — the filter a user runs under — showed the first
    over-ceiling ``Biological(...)`` in a program and swallowed every later
    one, from anywhere."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('default')
        Biological(layers=[(10.0, 20.0, 100.0, 100.0, 100.0)])   # call site A
        Biological(layers=[(10.0, 30.0, 100.0, 100.0, 100.0)])   # call site B
    assert len(record) == 2, [(w.filename, w.lineno) for w in record]


def test_each_receiver_call_site_warns_under_the_once_per_location_filter():
    """The consequence of misattribution, pinned directly: with both call
    sites keyed to the same library line, ``'default'`` — the filter a user
    runs under — shows the first and swallows the second."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('default')
        Receiver(depths=[10.0])      # call site A
        Receiver(depths=[20.0])      # call site B
    assert len(record) == 2, [str(w.message) for w in record]

    with warnings.catch_warnings(record=True) as repeated:
        warnings.simplefilter('default')
        for _ in range(3):
            Receiver(depths=[30.0])  # one call site, three calls
    assert len(repeated) == 1


_PACKAGE_DISPATCH_SOURCE = '''\
def dispatch(formula, **kwargs):
    return formula(**kwargs)
'''


def _in_package_dispatcher():
    """A callable whose frame reports an in-package filename.

    ``fetch_ssp`` and ``fetch_ssp_argo`` reach the sound-speed formulas through
    a ``_FORMULAS[formula]`` lookup, so a warning raised inside a formula has
    one library frame above it. Both attribution mechanisms read
    ``frame.f_code.co_filename`` and nothing else, so compiling this dispatcher
    against a real package path reproduces that depth without a network fetch
    and without leaving a file in the package tree."""
    namespace = {}
    exec(compile(_PACKAGE_DISPATCH_SOURCE,
                 str(_PACKAGE_DIR / 'data' / 'sound_speed.py'), 'exec'),
         namespace)
    return namespace['dispatch']


@pytest.mark.parametrize('kwargs', [
    {'temperature': 99.0, 'salinity': 35.0, 'pressure': 0.0},
    {'temperature': 10.0, 'salinity': 20.0, 'pressure': 0.0},
    {'temperature': 10.0, 'salinity': 35.0, 'pressure': 1.0e5},
], ids=['temperature', 'salinity', 'pressure'])
def test_delgrosso_extrapolation_warnings_name_the_callers_file(kwargs):
    """Each of the three Del Grosso domain warnings, driven through the
    library frame ``fetch_ssp`` puts above them.

    A hand-counted ``stacklevel`` is right only for a direct call: through the
    formula dispatch it names ``data/sound_speed.py``, which is a uacpy line
    the user cannot act on and — worse — one dedup key for every call site in
    their program."""
    dispatch = _in_package_dispatcher()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        dispatch(soundspeed_delgrosso, **kwargs)
    warning = _only_warning(record)
    assert 'Del Grosso' in str(warning.message)
    assert Path(warning.filename).resolve() == _THIS_FILE


def test_each_delgrosso_call_site_warns_under_the_once_per_location_filter():
    """The consequence of the misattribution, pinned directly: keyed to the
    dispatch line inside ``data/sound_speed.py``, ``'default'`` — the filter a
    user runs under — shows the first extrapolating call in a program and
    swallows every later one, from anywhere."""
    dispatch = _in_package_dispatcher()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('default')
        dispatch(soundspeed_delgrosso, temperature=10.0, salinity=20.0,
                 pressure=0.0)                                  # call site A
        dispatch(soundspeed_delgrosso, temperature=10.0, salinity=21.0,
                 pressure=0.0)                                  # call site B
    assert len(record) == 2, [(w.filename, w.lineno) for w in record]
    assert {Path(w.filename).resolve() for w in record} == {_THIS_FILE}


def test_skip_prefixes_stop_at_a_directory_boundary():
    """A bare package-root prefix is a string match, not a path match: it
    swallows every sibling whose name starts with the package's own name."""
    assert str(_PACKAGE_DIR) not in USER_FRAME_SKIP
    for prefix in USER_FRAME_SKIP:
        is_directory = prefix.endswith(os.sep)
        is_module_stem = Path(prefix + '.py').is_file()
        assert is_directory or is_module_stem, prefix

    sibling = str(_PACKAGE_DIR) + '_venv' + os.sep + 'lib' + os.sep + 'x.py'
    assert not any(sibling.startswith(p) for p in USER_FRAME_SKIP)


def test_a_prefix_equal_to_a_whole_filename_does_not_skip_that_frame():
    """The CPython rule the module-stem entries are shaped around.

    A frame is skipped when its filename *starts with* an entry and is longer
    than it. An entry equal to the filename in full skips nothing — so a
    top-level module listed by its whole path would be inert, and a warning
    raised two frames deep inside one would land on the outer of those two
    frames instead of on the caller. Both readings are checked here against
    the interpreter rather than assumed, because the whole shape of
    ``USER_FRAME_SKIP``'s module entries follows from which one holds."""
    module = _write_two_frame_module()

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        module.SKIP = (module.__file__,)
        module.outer()                                  # whole-path entry
    assert Path(_only_warning(record).filename) == Path(module.__file__)

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        module.SKIP = (module.__file__[:-len('.py')],)
        module.outer()                                  # module-stem entry
    assert Path(_only_warning(record).filename).resolve() == _THIS_FILE


def test_no_module_entry_swallows_the_excluded_test_directories():
    """Dropping ``.py`` makes a module entry a prefix of anything under a
    directory of the same stem: a ``test.py`` beside the package's ``tests/``
    would start skipping the test tree, and every attribution assertion in
    this file would then walk past the test that made it."""
    for excluded in ('tests', 'examples'):
        inside = str(_PACKAGE_DIR / excluded) + os.sep + 'some_module.py'
        offenders = [p for p in USER_FRAME_SKIP if inside.startswith(p)]
        assert offenders == [], offenders


def test_test_and_example_files_are_not_skipped():
    """They play the caller role the attribution points at."""
    for name in ('tests', 'examples'):
        candidate = str(_PACKAGE_DIR / name / 'some_module.py')
        assert not any(candidate.startswith(p) for p in USER_FRAME_SKIP)


def test_two_call_sites_of_one_converted_site_keep_separate_dedup_keys():
    """The half of the fix a filename check alone would miss.

    ``warnings`` keys its once-per-location registry on the attributed file
    *and line*, so a site that names a uacpy line collapses every caller onto
    one entry: under ``'default'`` -- the filter a user runs under -- the first
    call warns and every later one from anywhere in the program is silent.

    ``_cq_setup`` is the shared helper four public constant-Q estimators reach
    at the same depth, and its hand-counted level named the estimator's own
    line in this package rather than either caller's."""
    signal = np.zeros(64)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('default')
        constant_q_transform(signal, 1000.0, fmin=1.0)   # call site A
        constant_q_transform(signal, 1000.0, fmin=1.0)   # call site B
    lines = {w.lineno for w in record}
    files = {Path(w.filename).resolve() for w in record}
    assert files == {_THIS_FILE}, files
    assert len(lines) == 2, [(w.filename, w.lineno) for w in record]


def test_grn_reader_and_models_base_share_one_skip_prefix_set():
    """Identity, not equality: two definitions that agree today are what
    produced the bug this pins — ``grn_reader``'s copy had lost the trailing
    separator and swallowed every sibling path. One object cannot diverge."""
    from uacpy.models.base import USER_FRAME_SKIP as base_skip

    assert grn_reader.USER_FRAME_SKIP is USER_FRAME_SKIP
    assert base_skip is USER_FRAME_SKIP
    assert not hasattr(grn_reader, '_UACPY_PACKAGE_ROOT')


def test_grn_zero_range_warning_names_the_callers_file():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        grn_reader._warn_zero_ranges(np.array([0.0, 100.0]), 'R')
    warning = _only_warning(record)
    assert Path(warning.filename).resolve() == _THIS_FILE


# ── the sites that walk out instead of counting ─────────────────────────────
#
# Every entry below raised its warning with a hand-counted ``stacklevel`` until
# its attribution was measured. A hand count names the frame of whoever called
# the function raising the warning, so it can only ever be right for ONE call
# depth; each of these functions is reachable at more than one, whether through
# a shared private helper, a decorator, or a second public entry point that
# adds a frame. Every entry was measured misattributing on at least one such
# path, or converted alongside a sibling in the same function that was.
#
# Measuring by driving each site directly is what MISSES these: a direct call
# is the one depth a hand count is tuned for. The second door has to be found
# in the call graph and driven through deliberately.
#
# The count is part of the entry: a file may hold several converted sites in
# one function, and a revert that leaves one behind still has to fail.
CONVERTED_SITES = [
    ('acoustic_signal/analysis.py', '_warn_two_sided', 1),
    ('acoustic_signal/arrays.py', '_powerless_covariance', 1),
    ('acoustic_signal/channel.py', 'impulse_response', 2),
    ('acoustic_signal/constant_q.py', '_cq_setup', 2),
    ('acoustic_signal/system_id.py', '_etfe_divide', 1),
    ('comms/channel_models.py', 'awgn', 1),
    ('comms/janus.py', 'JanusPacket.from_bits', 1),
    ('comms/transceiver.py', 'CommsReceiver.receive', 1),
    ('comms/transceiver.py', 'OFDMReceiver.receive', 1),
    # Not a converted hand count but the same rule, and it belongs under the
    # same guard: two construction depths (a direct ``BiologicalLayer(...)``
    # and the tuple normalisation inside ``Biological.__init__``) that no
    # single count covers. Both classes write their ``__init__`` out so no
    # ``<string>`` frame stops the walk; a ``stacklevel`` reappearing here
    # would mean one of those was regenerated.
    ('core/absorption.py', 'BiologicalLayer.__init__', 1),
    ('core/acoustics.py', 'soundspeed', 3),
    ('core/acoustics.py', 'soundspeed_delgrosso', 3),
    ('core/acoustics.py', 'soundspeed_unesco', 4),
    ('core/bottom.py', 'SeabedColumn.collapse', 1),
    ('core/environment.py', 'Environment.get_sound_speed', 1),
    ('core/results/field.py', 'Field._warn_if_frequency_axis_undersamples', 1),
    ('core/results/field.py', 'Field._warn_if_undersampled', 2),
    ('core/results/field.py', '_clean_cell_spectra', 1),
    ('core/results/field.py', '_ifft_to_trace', 2),
    ('core/results/field.py', '_synthesize_time_series', 2),
    ('core/results/field.py', '_taper', 1),
    ('core/results/modes.py', 'Modes._warn_if_depth_axis_underresolves', 1),
    ('core/results/reflection.py', 'ReflectionCoefficient._resolve_axes', 1),
    ('core/sediment.py', 'grain_size_to_geoacoustics', 1),
    ('core/ssp.py', 'generate_sea_surface', 1),
    ('data/_netcdf.py', 'NetcdfGrid._bounded', 1),
    ('data/bathymetry.py', 'fetch_bathy_transect', 2),
    ('data/crust1_local.py', '_warn_non_commercial', 1),
    ('data/environment.py', '_record_provenance', 1),
    ('data/gebco_local.py', '_grid', 1),
    ('data/mars.py', '_query_bbox', 1),
    ('data/pelagic.py', 'pelagic_lithology', 1),
    ('data/sea_surface.py', '_surface', 1),
    ('data/seaice_local.py', '_concentration', 1),
    ('data/seaice_local.py', 'sea_ice_surface_transect', 1),
    ('data/sediment.py', 'range_dependent_bottom_along', 1),
    ('data/sound_speed.py', '_ts_profile_with_cell', 1),
    ('data/sound_speed.py', 'extend_ssp_below_data', 1),
    ('data/sound_speed.py', 'ssp_transect_plan', 1),
    ('io/_fortran_helpers.py', '_warn_non_little_endian', 1),
    ('io/bathy_io.py', 'write_bty_long_format', 1),
    ('io/bellhop_writer.py', 'write_bellhop_env_file', 1),
    ('io/grn_reader.py', '_warn_zero_ranges', 1),
    ('io/oalib_reader.py', 'read_flp', 1),
    ('io/oalib_reader.py', 'read_shd_bin', 2),
    ('io/oalib_writer.py', 'write_ssp_section', 1),
    ('io/oases_reader.py', '_oast_curve_slots', 1),
    ('io/oases_reader.py', '_read_oasp_trf_binary', 1),
    ('io/oases_writer.py', '_check_n_time_samples', 1),
    ('io/oases_writer.py', '_check_nw_samples', 1),
    ('io/oases_writer.py', '_check_ssp_layer_count', 1),
    ('io/oases_writer.py', '_emit_bottom_layers', 1),
    ('io/oases_writer.py', '_format_upper_halfspace', 1),
    ('io/oases_writer.py', '_noise_nw', 1),
    ('io/oases_writer.py', '_warn_rough_gradient_surface', 1),
    ('io/oases_writer.py', '_warn_volume_attenuation_ignored', 2),
    ('io/oases_writer.py', '_write_oases_header', 1),
    ('io/oases_writer.py', 'write_oasn_input', 1),
    ('io/oases_writer.py', 'write_oassp_input', 1),
    ('io/refl_io.py', 'stage_reflection_file', 1),
    ('parallel.py', '_reap_scratch_root', 2),
    ('sonar/reverberation.py', '_warn_if_cell_is_not_short', 1),
    ('sonar/scattering.py', '_warn_outside_chapman_harris_fit', 3),
    ('sonar/sonar_equation.py', 'detection_range', 1),
    ('sonar/target_strength.py', '_warn_below_geometric', 1),
    ('visualization/plots/_common.py', '_plot_warn', 1),
]


def _warn_call_kinds(path):
    """Map enclosing qualified name to (skip-walk calls, hand-counted calls)."""
    skip = Counter()
    counted = Counter()

    def walk(node, chain):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)):
                walk(child, chain + [child.name])
            else:
                walk(child, chain)
            if isinstance(child, ast.Call):
                names = {kw.arg for kw in child.keywords}
                qual = '.'.join(chain)
                if 'skip_file_prefixes' in names:
                    skip[qual] += 1
                if 'stacklevel' in names:
                    counted[qual] += 1

    walk(ast.parse(path.read_text(encoding='utf-8')), [])
    return skip, counted


@pytest.mark.parametrize('relative_path,qualname,count', CONVERTED_SITES,
                         ids=[f'{f}:{s}' for f, s, _ in CONVERTED_SITES])
def test_site_attributes_by_walking_out_not_by_counting(relative_path, qualname,
                                                        count):
    """Each converted site still hands the walk its prefix set.

    A hand-counted ``stacklevel`` reappearing here is the regression: it reads
    as correct from the one call path its author had in mind, and the site goes
    back to naming a uacpy line from every other one."""
    skip, counted = _warn_call_kinds(_PACKAGE_DIR / relative_path)
    assert skip[qualname] == count
    assert counted[qualname] == 0


def test_hand_counted_stacklevel_is_two_except_inside_a_post_init():
    """The shape every surviving hand-count has, with no exception list.

    A ``stacklevel`` of 2 blames the caller of the function raising the
    warning, so it is right exactly when that function is the one the user
    called. Anything larger encodes a chain of intermediate frames, and a
    chain is what breaks when a check moves or gains a second entry point.

    ``__post_init__`` is the one shape that legitimately needs 3, and needs it
    for a reason that is not a chain: a dataclass reaches ``__post_init__``
    through the ``__init__`` the decorator compiles from a string, and level 2
    is that generated frame. ``skip_file_prefixes`` cannot stand in there
    either — the generated frame's ``<string>`` filename matches no package
    prefix, so the walk stops on it.

    **What this rule cannot see, which is most of it.** A count larger than 2
    is visibly wrong-shaped and this catches it. A count of *2* under a
    private helper, or under a decorator, or in a public function that a
    second public entry point also reaches, is equally wrong and looks exactly
    like a correct one from here — the shape carries no information about how
    many frames actually sit above it at run time. Those are caught by the
    site table above, which was built by driving each site through its
    in-package caller, and by nothing else. A green run here is not evidence
    that attribution is right."""
    offenders = []
    for path in sorted(_PACKAGE_DIR.rglob('*.py')):
        if 'tests' in path.parts or 'examples' in path.parts:
            continue

        def walk(node, chain):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.ClassDef)):
                    walk(child, chain + [child.name])
                else:
                    walk(child, chain)
                if not isinstance(child, ast.Call):
                    continue
                for keyword in child.keywords:
                    if keyword.arg != 'stacklevel':
                        continue
                    value = ast.unparse(keyword.value)
                    wanted = '3' if chain[-1:] == ['__post_init__'] else '2'
                    if value != wanted:
                        offenders.append(
                            f"{path.relative_to(_PACKAGE_DIR)}:{child.lineno}"
                            f" stacklevel={value} in {'.'.join(chain)}"
                            f" (wanted {wanted})")

        walk(ast.parse(path.read_text(encoding='utf-8')), [])
    assert offenders == [], offenders


def test_no_site_combines_the_skip_walk_with_a_raised_stacklevel():
    """The over-skip failure mode, which nothing else here would catch.

    The walk already stops on the first frame outside the package, so a
    ``stacklevel`` passed alongside it carries the attribution that many frames
    *further* — ``stacklevel=3`` names the caller's caller. Measured on a
    two-frame package chain: levels 1 and 2 both land on the user (CPython
    raises the level to 2 before walking), 3 lands one frame past, 4 lands two
    past. It is the mirror image of the counts this mechanism replaced, and
    just as silent, because the warning still names a file outside uacpy.

    No site combines them today. This keeps it that way."""
    offenders = []
    for path in sorted(_PACKAGE_DIR.rglob('*.py')):
        if 'tests' in path.parts or 'examples' in path.parts:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if not isinstance(node, ast.Call):
                continue
            names = {kw.arg for kw in node.keywords}
            if {'skip_file_prefixes', 'stacklevel'} <= names:
                offenders.append(
                    f"{path.relative_to(_PACKAGE_DIR)}:{node.lineno}")
    assert offenders == [], offenders


# A hand count is only ever right for one call depth, so the thing that breaks
# it is a SECOND in-package caller appearing above a site that had none. Each
# entry here is a site where that has already happened and the count still
# stands, with the reason it stands. Anything else showing up is a site that
# has silently grown a second door.
HAND_COUNTS_WITH_AN_IN_PACKAGE_CALLER = {
    # Reachable only from a user's constructor call: every in-package
    # ``Receiver(...)`` passes an explicit ``ranges=``.
    ('core/receiver.py', 'Receiver.__post_init__'),
    # ``bottom_loss_curve`` always passes an explicit ``c=``, so the fallback
    # this warns about cannot be reached from inside the package.
    ('core/acoustics.py', 'reflection_coeff'),
    # ``_wind_merklinger`` passes three positional arguments, so
    # ``band_integrate`` is always False on that path and the branch is dead.
    ('noise/noise.py', 'compute_windnoise'),
}


def _resolved_in_package_callers(target, defining_file, call_index, import_index):
    """Calls to *target* from elsewhere in the package, name collisions removed.

    A bare name match cannot tell this package's ``sel`` from xarray's, or
    ``ParallelResult.stack`` from ``np.stack``; a call is kept only when it
    sits in the defining module itself or in a module that imports the name
    from it."""
    out = []
    for caller_file, lineno, enclosing in call_index.get(target, []):
        if caller_file == defining_file:
            out.append((caller_file, lineno, enclosing))
            continue
        source = import_index.get(caller_file, {}).get(target)
        own = defining_file[:-len('.py')].replace(os.sep, '.')
        if source and own.endswith(source.split('.')[-1]):
            out.append((caller_file, lineno, enclosing))
    return out


def test_no_hand_counted_site_has_grown_a_second_in_package_caller():
    """The check that the first pass of this work did not make, and needed to.

    Attribution was originally measured by driving every site directly, which
    is the one depth a hand count is tuned for; 26 sites that read as correct
    that way were misattributing through a second, supported public path. The
    call graph is what exposes that, not the driver.

    BLIND SPOT, stated because a green run here is not coverage: callers are
    resolved through imports, so a call reached on an instance
    (``obj.method()``) resolves only inside the defining module. A site whose
    only second caller invokes it as a method from another module will not
    appear. This narrows to what can be resolved statically; it does not
    certify the rest."""
    call_index = defaultdict(list)
    import_index = {}
    site_files = {}

    package_files = [p for p in sorted(_PACKAGE_DIR.rglob('*.py'))
                     if 'tests' not in p.parts and 'examples' not in p.parts]
    for path in package_files:
        tree = ast.parse(path.read_text(encoding='utf-8'))
        rel = str(path.relative_to(_PACKAGE_DIR))
        import_index[rel] = {
            alias.asname or alias.name: node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
            for alias in node.names
        }

        def walk(node, chain):
            for child in ast.iter_child_nodes(node):
                nxt = (chain + [child.name]
                       if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                             ast.ClassDef)) else chain)
                walk(child, nxt)
                if not isinstance(child, ast.Call):
                    continue
                func = child.func
                name = (func.attr if isinstance(func, ast.Attribute)
                        else func.id if isinstance(func, ast.Name) else None)
                if name:
                    call_index[name].append((rel, child.lineno, '.'.join(chain)))
                if any(kw.arg == 'stacklevel' for kw in child.keywords):
                    leaf = chain[-1] if chain else '<module>'
                    owner = chain[-2] if len(chain) >= 2 else None
                    site_files[(rel, '.'.join(chain))] = (
                        owner if leaf == '__post_init__' else leaf)

        walk(tree, [])

    found = set()
    for (rel, qualname), target in site_files.items():
        callers = [c for c in _resolved_in_package_callers(
            target, rel, call_index, import_index) if c[2] != qualname]
        if callers:
            found.add((rel, qualname))

    assert found == HAND_COUNTS_WITH_AN_IN_PACKAGE_CALLER, {
        'newly reachable from inside the package': sorted(
            found - HAND_COUNTS_WITH_AN_IN_PACKAGE_CALLER),
        'no longer reachable, drop from the set': sorted(
            HAND_COUNTS_WITH_AN_IN_PACKAGE_CALLER - found),
    }


#: A floor on the sweep below. The rule it enforces is only as good as the
#: number of sites it reaches, and a walk that stopped matching would leave the
#: gate silently green on an empty set.
_MIN_ATTRIBUTED_WARN_SITES = 190


def test_every_warn_site_names_the_users_line_by_one_of_the_two_forms():
    """The convention DEV.md §6.2 publishes, enforced.

    Every other gate in this module inspects sites that *already* carry
    ``skip_file_prefixes`` or ``stacklevel`` — none of them asserts that a warn
    call carries either, so a bare ``warnings.warn(msg)`` in a new module is
    invisible to all of them and blames a uacpy frame at run time. This is the
    one that sees it.

    What it cannot see: whether the form chosen is the *right* one for that
    site's call depth. That is the site table earlier in this module."""
    offenders = []
    attributed = 0
    for path in sorted(_PACKAGE_DIR.rglob('*.py')):
        if {'tests', 'examples', 'third_party'} & set(path.parts):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == 'warn'
                    and getattr(func.value, 'id', None) == 'warnings'):
                continue
            keywords = {keyword.arg for keyword in node.keywords}
            if keywords & {'skip_file_prefixes', 'stacklevel'}:
                attributed += 1
            else:
                offenders.append(f"{path.relative_to(_PACKAGE_DIR)}:{node.lineno}")
    assert offenders == [], offenders
    assert attributed >= _MIN_ATTRIBUTED_WARN_SITES, (
        f"the warn-site walk found only {attributed} attributed sites, below "
        f"the {_MIN_ATTRIBUTED_WARN_SITES} floor — it has stopped matching")


def _dev_md_section(heading, next_heading):
    """The text of one DEV.md section, so a gate on §5 cannot be satisfied by
    §6.2's copy of the same word, or the other way round."""
    dev_md = _PACKAGE_DIR.parent / 'docs' / 'DEV.md'
    if not dev_md.is_file():
        pytest.skip('docs/DEV.md is not present')
    text = dev_md.read_text(encoding='utf-8')
    assert heading in text and next_heading in text, (heading, next_heading)
    start = text.index(heading)
    return text[start:text.index(next_heading, start)]


def test_dev_md_lists_the_module_every_warn_site_imports():
    """``core/_warn_frames.py`` is the most widely imported module in
    ``core/``, and DEV.md §5's core list is where a contributor looks for what
    is in that package."""
    importers = [
        path for path in _PACKAGE_DIR.rglob('*.py')
        if 'tests' not in path.parts
        and 'core._warn_frames' in path.read_text(encoding='utf-8')]
    assert len(importers) > 20, len(importers)      # the premise of the claim
    section = _dev_md_section('- `_beamforming.py`', '## 6. Support systems')
    assert '`_warn_frames.py`' in section
    assert 'USER_FRAME_SKIP' in section


def test_dev_md_states_the_convention_every_warn_site_follows():
    """§6.2 is the logging-and-warnings section. A contributor reading only
    the mechanism sentence there has no way to know a warn call takes a
    keyword at all, and a bare call is what the gate above then catches."""
    section = _dev_md_section('### 6.2 Logging',
                              '### 6.3 Stack-size bootstrapping')
    for phrase in ('skip_file_prefixes', 'USER_FRAME_SKIP', 'stacklevel'):
        assert phrase in section, phrase

