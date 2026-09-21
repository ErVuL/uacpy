"""A multi-depth ``Source`` runs once per depth on every field model and the
slabs add coherently through ``ResultStack.superpose``.

Two sources are driven by adding the complex field of each one: the engines
are linear in the source amplitude, so ``Σ wᵢ·pᵢ`` over unit-source slabs is
the field of the weighted array. The tests here pin (a) that the per-depth
loop changes nothing per slab, (b) that ``superpose`` is that linear sum,
(c) that the sum behaves like a field of two sources, (d) the time-domain
case, and (e) the guards on weights and on stacks that cannot add.
"""

from pathlib import Path

import warnings

import numpy as np
import pytest

import uacpy
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import Field, ResultStack
from uacpy.models import RunMode
from uacpy.tests.conftest import make_pekeris

# 100 m Pekeris guide at 100 Hz on a 3-depth × 20-range receiver grid: the
# smallest grid on which a Kraken, Scooter, RAM, OAST or Bellhop field is
# still a field.
F0 = 100.0
GUIDE_DEPTH = 100.0


def _env():
    return make_pekeris(name='multi-source', bathymetry=GUIDE_DEPTH)


def _receiver():
    return uacpy.Receiver(depths=[20.0, 50.0, 80.0],
                          ranges=np.linspace(100.0, 2000.0, 20))


def _layered_env():
    """A bottom Bellhop cannot write natively, so ``run()`` auto-routes
    through BOUNCE — the spawn the weights must not reach."""
    from uacpy.core import BoundaryProperties
    from uacpy.core.bottom import Bottom, SeabedColumn, SedimentLayer
    env = _env()
    env.bottom = Bottom(columns=[SeabedColumn(
        layers=[SedimentLayer(thickness=5.0, sound_speed=1550.0,
                              density=1.4, attenuation=0.2)],
        halfspace=BoundaryProperties(acoustic_type='half-space',
                                     sound_speed=1600.0, density=1.8,
                                     attenuation=0.2))])
    return env


def _kraken():
    from uacpy.models.kraken import Kraken
    return Kraken(verbose=False)


def _scooter():
    from uacpy.models.scooter import Scooter
    return Scooter(verbose=False)


def _ram():
    from uacpy.models.ram import RAM
    return RAM(verbose=False)


def _oast():
    from uacpy.models.oases import OAST
    return OAST(verbose=False)


def _bellhop():
    from uacpy.models.bellhop import Bellhop
    return Bellhop(verbose=False)


# (factory, slab tolerance). The looped engines write the same deck for a
# slab and for its stand-alone run, so the two agree to round-off. Bellhop
# writes both depths into one deck and its ``.shd`` is complex64, and the
# parallel beam sum lands in a different order per run, so its slabs agree
# to float32 round-off only.
_FIELD_MODELS = [
    pytest.param(_bellhop, 1e-5, id='Bellhop',
                 marks=pytest.mark.requires_binary),
    pytest.param(_kraken, 1e-9, id='Kraken',
                 marks=pytest.mark.requires_binary),
    pytest.param(_scooter, 1e-9, id='Scooter',
                 marks=pytest.mark.requires_binary),
    pytest.param(_ram, 1e-9, id='RAM', marks=pytest.mark.requires_binary),
    pytest.param(_oast, 1e-9, id='OAST',
                 marks=[pytest.mark.requires_binary,
                        pytest.mark.requires_oases]),
]


def _relative_error(actual, reference):
    """Largest ``|actual - reference| / |reference|`` over the cells where
    the reference is finite and non-zero."""
    ref = np.asarray(reference)
    ok = np.isfinite(ref) & (ref != 0)
    assert ok.any(), "the reference field is empty"
    return float(np.max(np.abs(np.asarray(actual)[ok] - ref[ok])
                        / np.abs(ref[ok])))


def _norm_relative_error(actual, reference):
    """``‖actual - reference‖ / ‖reference‖`` over the finite cells."""
    ref = np.asarray(reference)
    act = np.asarray(actual)
    ok = np.isfinite(ref) & np.isfinite(act)
    return float(np.linalg.norm(act[ok] - ref[ok]) / np.linalg.norm(ref[ok]))


@pytest.fixture(scope='module')
def kraken_stack():
    """One Kraken run over ``Source(depths=[30, 70], weights=[1, -1])``,
    shared by the linearity and guard tests below."""
    pytest.importorskip('uacpy.models.kraken')
    from uacpy.core.exceptions import ExecutableNotFoundError
    try:
        model = _kraken()
    except ExecutableNotFoundError:
        pytest.skip("Kraken binary not available")
    source = uacpy.Source(depths=[30.0, 70.0], frequencies=F0,
                          weights=[1.0, -1.0])
    return model.run(_env(), source, _receiver())


# ── (a) the loop changes nothing per slab ───────────────────────────────


@pytest.mark.parametrize('factory,tolerance', _FIELD_MODELS)
def test_every_field_model_stacks_a_two_depth_source(factory, tolerance):
    """Each slab of the stack is the single-source run at that depth: the
    same environment, receiver and mode go to every depth, so nothing but
    the source depth differs between a slab and its stand-alone run."""
    model = factory()
    env, receiver = _env(), _receiver()
    depths = [30.0, 70.0]
    stack = model.run(env, uacpy.Source(depths=depths, frequencies=F0),
                      receiver)
    assert isinstance(stack, ResultStack)
    assert stack.n_slabs == 2
    assert stack.slab_type is Field
    assert stack.coordinate_name == 'source_depth'
    np.testing.assert_array_equal(stack.coordinate, depths)
    np.testing.assert_array_equal(stack.metadata['source_weights'],
                                  [1.0, 1.0])
    for i, z in enumerate(depths):
        single = model.run(env, uacpy.Source(depths=z, frequencies=F0),
                           receiver)
        assert isinstance(single, Field)
        assert stack[i].data.shape == single.data.shape
        assert stack[i].is_complex == single.is_complex
        assert _relative_error(stack[i].data, single.data) < tolerance
        np.testing.assert_array_equal(stack[i].source_depths, [z])


@pytest.mark.requires_binary
def test_a_single_depth_source_returns_a_field_with_no_weight_stamp():
    """The loop only fires above one depth: the single-source contract is a
    plain ``Field`` with no weight stamp, and an explicit unit weight is
    the same run."""
    model = _kraken()
    field = model.run(_env(), uacpy.Source(depths=50.0, frequencies=F0),
                      _receiver())
    assert isinstance(field, Field)
    assert 'source_weights' not in field.metadata
    assert 'superposed_sources' not in field.metadata
    unit = model.run(_env(), uacpy.Source(depths=50.0, frequencies=F0,
                                          weights=1.0), _receiver())
    np.testing.assert_array_equal(unit.data, field.data)
    assert 'superposed_sources' not in unit.metadata


@pytest.mark.requires_binary
def test_a_single_depth_source_applies_its_weight():
    """One source is the ``n = 1`` case of the same rule: ``weights=2``
    scales the field by 2, ``weights=1j`` rotates it, and the metadata
    records the one depth and weight."""
    model = _kraken()
    unit = model.run(_env(), uacpy.Source(depths=50.0, frequencies=F0),
                     _receiver())
    for w in (2.0, 1j, -0.5 + 0.25j):
        scaled = model.run(_env(), uacpy.Source(depths=50.0, frequencies=F0,
                                                weights=w), _receiver())
        assert isinstance(scaled, Field)
        assert _relative_error(scaled.data, w * unit.data) < 1e-12
        assert scaled.metadata['superposed_sources'] == {
            'depths': [50.0], 'weights': [complex(w)]}


def test_every_concrete_model_implements_run_single_and_inherits_run():
    """``run`` is the base class's template method — validation, the
    per-depth loop, the weight — and every concrete model implements the
    one-source, one-mode body as ``_run_single`` with the same leading
    signature. No wrapper redefines ``run``, so a subclass that overrides
    it and delegates to ``super().run()`` passes through the template once.
    """
    import inspect
    from uacpy.models.base import PropagationModel
    from uacpy.models.bellhop import Bellhop
    from uacpy.models.kraken import Kraken
    from uacpy.models.ram import RAM
    from uacpy.models.scooter import Scooter
    from uacpy.models.sparc import SPARC
    from uacpy.models.bounce import Bounce
    from uacpy.models.oases import OAST, OASN, OASR, OASP, OASSP, OASS
    for cls in (Bellhop, Kraken, RAM, Scooter, SPARC, Bounce,
                OAST, OASN, OASR, OASP, OASSP, OASS):
        assert 'run' not in cls.__dict__, cls
        assert cls.run is PropagationModel.run, cls
        assert '_run_single' in cls.__dict__, cls
        names = list(inspect.signature(cls._run_single).parameters)
        assert names[:5] == ['self', 'env', 'source', 'receiver', 'run_mode']


def test_a_run_single_whose_run_mode_default_is_not_none_is_refused():
    """The template calls ``_run_single(..., run_mode)`` with the value it
    received, so a body declaring its own default for ``run_mode`` would
    never see it; ``__init_subclass__`` refuses the declaration."""
    from uacpy.models.ram import RAM
    with pytest.raises(TypeError, match='run_mode'):
        class DefaultedMode(RAM):
            def _run_single(self, env, source, receiver,
                            run_mode=RunMode.COHERENT_TL, *,
                            frequencies=None, source_waveform=None,
                            sample_rate=None, output_duration=None):
                return None


def test_a_run_single_with_a_var_positional_sink_is_refused():
    """The mirror of the ``**kwargs`` rule: a ``*args`` sink swallows
    exactly the unknown positional arguments the keyword-only rule below
    it exists to refuse."""
    from uacpy.models.ram import RAM
    with pytest.raises(TypeError, match=r'\*args'):
        class VarPositionalSink(RAM):
            def _run_single(self, env, source, receiver, run_mode=None,
                            *args, frequencies=None, source_waveform=None,
                            sample_rate=None, output_duration=None):
                return None


def test_a_run_single_that_requires_run_mode_is_accepted():
    """``run`` always passes ``run_mode`` positionally, so a body that
    declares no default for it is callable — only a non-None default is a
    lie, because the body could never see it."""
    from uacpy.models.ram import RAM

    class RequiredMode(RAM):
        def _run_single(self, env, source, receiver, run_mode, *,
                        frequencies=None, source_waveform=None,
                        sample_rate=None, output_duration=None):
            return None

    assert RequiredMode._run_single is not None


def test_a_class_that_declares_no_spec_is_told_which_body_it_defined():
    """The message names the body ``__init_subclass__`` actually saw, so a
    class that defined ``run`` is not told it defined ``_run_single``."""
    from uacpy.models.base import PropagationModel
    with pytest.raises(TypeError, match=r'defines run\(\) but declares no'):
        type('OnlyRun', (PropagationModel,), {
            'run': lambda self, env, source, receiver, run_mode=None: None})


@pytest.mark.requires_binary
def test_a_weight_on_a_db_mode_refuses_eagerly_and_waits_lazily():
    """The two paths differ because of *when* the weight is applied, not
    whether it could be.

    One source is weighted by ``run()`` itself, and there is no defined way
    to scale a field whose phase is gone at the moment it is handed back —
    so it refuses, naming the dB arithmetic that would do it. Several
    sources are weighted by ``superpose``, which has not been called yet and
    may be called either way: a dB-only stack cannot add coherently but adds
    perfectly well in intensity, so refusing it here would close the only
    route to N mutually incoherent sources at given levels."""
    model = _kraken()
    with pytest.raises(ConfigurationError, match='dB'):
        model.run(_env(), uacpy.Source(depths=50.0, frequencies=F0,
                                       weights=2.0), _receiver(),
                  run_mode=RunMode.INCOHERENT_TL)
    stack = model.run(_env(), uacpy.Source(depths=[30.0, 70.0],
                                           frequencies=F0, weights=[1.0, 2.0]),
                      _receiver(), run_mode=RunMode.INCOHERENT_TL)
    assert isinstance(stack, ResultStack)
    assert stack.superpose(coherent=False).unit == 'dB'
    with pytest.raises(ConfigurationError, match='coherent sum'):
        stack.superpose()


@pytest.mark.requires_binary
def test_a_subclass_that_delegates_to_super_run_applies_the_weight_once():
    """Overriding ``run`` and calling ``super().run()`` is how a wrapper is
    extended; the weight must reach the field once, not once per layer."""
    from uacpy.models.ram import RAM

    class Delegating(RAM):
        def run(self, env, source, receiver, run_mode=None, *,
                frequencies=None, source_waveform=None, sample_rate=None,
                output_duration=None):
            return super().run(env, source, receiver, run_mode,
                               frequencies=frequencies,
                               source_waveform=source_waveform,
                               sample_rate=sample_rate,
                               output_duration=output_duration)

    unit = RAM(verbose=False).run(
        _env(), uacpy.Source(depths=50.0, frequencies=F0), _receiver())
    scaled = Delegating(verbose=False).run(
        _env(), uacpy.Source(depths=50.0, frequencies=F0, weights=2.0),
        _receiver())
    assert _relative_error(scaled.data, 2.0 * unit.data) < 1e-12
    assert scaled.metadata['superposed_sources']['weights'] == [2.0 + 0j]


@pytest.mark.requires_binary
def test_bellhop_applies_a_single_depth_weight_in_its_native_tl_modes():
    """Bellhop writes every depth into one deck, so it never loops — and
    the one-source weight is a field-mode rule, not a loop rule: it scales
    a Bellhop TL field exactly as it scales Kraken's."""
    model = _bellhop()
    unit = model.run(_env(), uacpy.Source(depths=50.0, frequencies=F0),
                     _receiver())
    scaled = model.run(_env(), uacpy.Source(depths=50.0, frequencies=F0,
                                            weights=2.0), _receiver())
    assert _relative_error(scaled.data, 2.0 * unit.data) < 1e-5
    assert scaled.metadata['superposed_sources'] == {
        'depths': [50.0], 'weights': [2.0 + 0j]}


@pytest.mark.requires_binary
def test_a_weight_on_a_db_only_field_is_refused_as_superpose_refuses_it():
    """Kraken's ``INCOHERENT_TL`` returns real dB, whose phase is gone: a
    weight cannot scale it (it would multiply the decibels), so the
    one-source path refuses with the same reason ``superpose`` gives a
    dB-only stack, and names the hand formula for a magnitude weight."""
    model = _kraken()
    with pytest.raises(ConfigurationError,
                       match=r'dB.*phase.*20\*log10'):
        model.run(_env(), uacpy.Source(depths=50.0, frequencies=F0,
                                       weights=2.0), _receiver(),
                  run_mode=RunMode.INCOHERENT_TL)
    # The unit weight is no weight: the same run answers as before.
    plain = model.run(_env(), uacpy.Source(depths=50.0, frequencies=F0),
                      _receiver(), run_mode=RunMode.INCOHERENT_TL)
    unit = model.run(_env(), uacpy.Source(depths=50.0, frequencies=F0,
                                          weights=1.0), _receiver(),
                     run_mode=RunMode.INCOHERENT_TL)
    np.testing.assert_array_equal(unit.data, plain.data)


@pytest.mark.requires_binary
def test_a_pinned_work_dir_keeps_one_scratch_set_per_source_depth(tmp_path):
    """Each depth of the loop runs in its own ``source_depth_<z>m``
    subdirectory of the pinned ``work_dir``, so every slab's ``*_file``
    paths stay its own instead of all naming the last depth's files. RAM
    is the engine that genuinely marches once per depth."""
    from uacpy.models.ram import RAM
    depths = [30.0, 70.0]
    stack = RAM(verbose=False, work_dir=tmp_path, cleanup=False).run(
        _env(), uacpy.Source(depths=depths, frequencies=F0), _receiver())
    key = next(k for k in sorted(stack[0].metadata) if k.endswith('_file'))
    files = [s.metadata[key] for s in stack.slabs]
    assert files[0] != files[1], files
    for z, path in zip(depths, files):
        path = str(path)
        assert path.startswith(str(tmp_path)), path
        assert f'source_depth_{z:g}m' in path, path
        assert Path(path).exists(), path


@pytest.mark.requires_binary
@pytest.mark.parametrize('factory,mode,tolerance', [
    pytest.param(_scooter, None, 0.0, id='Scooter-COHERENT_TL'),
    pytest.param(_kraken, None, 0.0, id='Kraken-COHERENT_TL'),
    pytest.param(_kraken, RunMode.INCOHERENT_TL, 0.0, id='Kraken-INCOHERENT'),
])
def test_the_native_engines_solve_once_for_every_source_depth(
        factory, mode, tolerance, tmp_path):
    """SCOOTER's ``.env`` and KRAKEN's ``.flp`` take a source-depth vector:
    one wavenumber sweep / one mode solve serves every depth, and the
    ``.grn`` / ``.shd`` carries the ``NSz`` axis the reader splits. A
    multi-depth run in these modes therefore launches the binary once — one
    deck in the pinned ``work_dir``, no per-depth subdirectory — and each
    slab is still that depth's stand-alone run.

    Both are exact. Scooter's depth mesh comes from the media, so a source
    depth touches no other depth's answer. Kraken's receivers are placed on
    the mode-tabulation grid (``_write_field_env``) AND written to the
    ``.flp`` in full rather than through FIELD's subtabulate shortcut, which
    would recompute them in single precision and leave them a few ULPs off
    their own nodes — enough to put the interpolation weight at ~1e-7
    instead of 0 and let a source depth elsewhere in the grid move a
    near-null receiver by 6e-6."""
    model = factory()
    model.work_dir, model.cleanup = tmp_path, False
    env, receiver = _env(), _receiver()
    depths = [30.0, 70.0]
    stack = model.run(env, uacpy.Source(depths=depths, frequencies=F0),
                      receiver, run_mode=mode)
    assert isinstance(stack, ResultStack) and stack.n_slabs == 2
    decks = list(tmp_path.rglob('*.env'))
    assert len(decks) == 1, decks
    assert not list(tmp_path.glob('source_depth_*')), \
        "the run was split into one deck per depth"
    np.testing.assert_array_equal(stack.metadata['source_weights'], [1.0, 1.0])
    for i, z in enumerate(depths):
        single = factory().run(env, uacpy.Source(depths=z, frequencies=F0),
                               receiver, run_mode=mode)
        np.testing.assert_array_equal(stack[i].source_depths, [z])
        assert stack[i].is_complex == single.is_complex
        err = _relative_error(stack[i].data, single.data)
        assert err <= tolerance, f"slab {i} (z={z} m) drifted by {err:.2e}"


@pytest.mark.requires_binary
def test_a_line_source_stack_levels_each_slab_at_its_own_depth():
    """The line-source level is sqrt(k0) at the SOURCE depth, so on a
    sound-speed gradient each slab of a native multi-depth run must use
    its own depth's speed. A point source on an isovelocity guide makes
    both factors degenerate, which is why this uses neither."""
    grad = uacpy.Environment(
        name='gradient', bathymetry=GUIDE_DEPTH,
        ssp=[(0.0, 1450.0), (GUIDE_DEPTH, 1550.0)],
        bottom=_env().bottom,
    )
    receiver = _receiver()
    depths = [30.0, 70.0]
    model = _scooter()
    stack = model.run(grad, uacpy.Source(depths=depths, frequencies=F0,
                                         source_type='line'), receiver)
    for i, z in enumerate(depths):
        single = _scooter().run(
            grad, uacpy.Source(depths=z, frequencies=F0, source_type='line'),
            receiver)
        assert _relative_error(stack[i].data, single.data) < 1e-9, (
            f"slab {i} (z={z} m) is not the stand-alone line-source run")


@pytest.mark.requires_binary
def test_a_per_depth_scratch_dir_does_not_leak_into_the_model(tmp_path):
    """``run()`` moves the scratch into a per-depth subdirectory for the
    duration of one depth only. The model's own ``work_dir`` is what the
    user configured, before and after — a run must not leave the instance
    pointing inside a subdirectory, and a concurrent run on the same
    instance must not see one."""
    from uacpy.models.ram import RAM
    model = RAM(verbose=False, work_dir=tmp_path, cleanup=False)
    before = model.work_dir
    model.run(_env(), uacpy.Source(depths=[30.0, 70.0], frequencies=F0),
              _receiver())
    assert model.work_dir == before, model.work_dir


@pytest.mark.requires_binary
def test_two_threads_on_one_model_keep_their_own_scratch_dirs(tmp_path):
    """The per-depth redirect is per-run state, not instance state: a
    single-depth run on a shared instance must not be refused for a
    ``source_depth_*`` directory another thread's multi-depth run created.
    """
    import threading
    from uacpy.models.ram import RAM
    model = RAM(verbose=False, work_dir=tmp_path / 'shared', cleanup=False)
    errors = {}

    def go(tag, depths):
        try:
            model.run(_env(), uacpy.Source(depths=depths, frequencies=F0),
                      _receiver())
        except Exception as exc:          # noqa: BLE001 - recorded, not hidden
            errors[tag] = exc

    threads = [threading.Thread(target=go, args=('multi', [30.0, 70.0])),
               threading.Thread(target=go, args=('single', 50.0))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert 'source_depth' not in str(errors.get('single', '')), errors
    assert model.work_dir == tmp_path / 'shared', model.work_dir


@pytest.mark.requires_binary
def test_a_bellhop_eigenray_stack_keeps_one_ray_file_per_depth(tmp_path):
    """Bellhop's EIGENRAYS loop is the package's other per-depth Python
    loop; it gets the same per-depth scratch as the base's, so slab 0's
    ``ray_file`` is slab 0's rays and not the last depth's."""
    model = _bellhop()
    model.work_dir, model.cleanup = tmp_path, False
    stack = model.run(_env(), uacpy.Source(depths=[30.0, 70.0],
                                           frequencies=F0), _receiver(),
                      run_mode=RunMode.EIGENRAYS)
    files = [s.metadata['ray_file'] for s in stack.slabs]
    assert files[0] != files[1], files
    for z, path in zip((30.0, 70.0), files):
        assert f'source_depth_{z:g}m' in str(path), path


@pytest.mark.requires_binary
def test_an_internal_spawn_never_sees_the_callers_weights():
    """A model that runs another model inside itself hands it a unit
    source: the weight belongs to the run the user called. Otherwise the
    spawn warns — or refuses — about weights in a mode the user never
    asked for, naming a model they never built."""
    env, receiver = _env(), _receiver()
    weighted = uacpy.Source(depths=[30.0, 70.0], frequencies=F0,
                            weights=[1.0, -1.0])
    layered = _layered_env()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _bellhop().run(layered, weighted, receiver)
    spurious = [str(w.message) for w in caught
                if 'is not applied' in str(w.message)]
    assert not spurious, spurious


def test_a_complex_weight_needs_a_phase_reference_not_just_complex_data():
    """Bellhop's incoherent TL stores complex pressure whose phase is an
    artefact of AT's storage (``phase_reference`` is None). Rotating it by
    a complex weight would record a source phase the field cannot carry,
    so a complex weight is refused there; a real one still scales it."""
    from uacpy.core.results.field import _check_field_weightable
    artefact = Field(
        data=np.full((2, 3), 1.0 + 1.0j),
        coords={'depth': [10.0, 20.0], 'range': [100.0, 200.0, 300.0]},
        model='Test', frequencies=F0, source_depths=[10.0],
        phase_reference=None,
    )
    with pytest.raises(ConfigurationError, match='phase reference'):
        _check_field_weightable(artefact, np.array([1j]), where='probe')
    _check_field_weightable(artefact, np.array([2.0]), where='probe')
    referenced = _complex_slab(10.0, 1.0 + 1.0j)
    _check_field_weightable(referenced, np.array([1j]), where='probe')


def test_a_stack_with_unapplied_weights_warns_on_its_level_views():
    """A multi-depth run's stack holds unit-amplitude slabs whatever the
    Source's weights, so its dB / TL view and its panel plot show a field
    the weights never touched; they say so once, naming ``superpose``. A
    unit-weight stack is silent."""
    weighted = ResultStack([_complex_slab(10.0, 1.0), _complex_slab(20.0, 2.0)],
                           [10.0, 20.0])
    for slab in weighted.slabs:
        slab.metadata['source_weights'] = np.array([1.0, -1.0])
    with pytest.warns(UserWarning, match=r'weights.*superpose'):
        weighted.dB
    with pytest.warns(UserWarning, match=r'weights.*superpose'):
        weighted.tl
    unit = ResultStack([_complex_slab(10.0, 1.0), _complex_slab(20.0, 2.0)],
                       [10.0, 20.0])
    for slab in unit.slabs:
        slab.metadata['source_weights'] = np.array([1.0, 1.0])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        unit.dB
        unit.tl


# ── (b) superpose is the linear sum ─────────────────────────────────────


def test_superpose_is_the_weighted_sum_of_the_slabs(kraken_stack):
    """``superpose([w1, w2])`` is ``w1·p1 + w2·p2`` cell by cell, for real
    and complex weights, and ``[1, 0]`` returns slab 0 unchanged."""
    p1, p2 = kraken_stack[0].data, kraken_stack[1].data
    for w1, w2 in ((1.0, 1.0), (0.3, -2.0), (1.0, 1j), (0.5 + 0.5j, -1j)):
        summed = kraken_stack.superpose([w1, w2])
        assert isinstance(summed, Field)
        assert summed.is_complex
        assert _relative_error(summed.data, w1 * p1 + w2 * p2) < 1e-12
    only_first = kraken_stack.superpose([1.0, 0.0])
    np.testing.assert_array_equal(only_first.data, p1)


def test_superpose_reads_the_source_weights_by_default(kraken_stack):
    """With no argument the stack sums with the weights the ``Source``
    carried, here ``[1, -1]``: the same result as applying them by hand."""
    np.testing.assert_array_equal(kraken_stack.metadata['source_weights'],
                                  [1.0, -1.0])
    by_default = kraken_stack.superpose()
    by_hand = kraken_stack[0].data - kraken_stack[1].data
    assert _relative_error(by_default.data, by_hand) < 1e-12


def test_the_superposed_field_carries_the_stack_identity(kraken_stack):
    """The sum keeps the slabs' grid, phase reference and model, widens
    ``source_depths`` to every depth summed, and records what was added."""
    summed = kraken_stack.superpose([1.0, 1.0])
    first = kraken_stack[0]
    assert list(summed.coords) == list(first.coords)
    for axis in first.coords:
        np.testing.assert_array_equal(summed.coords[axis], first.coords[axis])
    assert summed.phase_reference == first.phase_reference
    assert summed.phase_reference is not None
    assert summed.model == first.model
    np.testing.assert_array_equal(summed.source_depths, [30.0, 70.0])
    assert summed.metadata['superposed_sources'] == {
        'depths': [30.0, 70.0], 'weights': [1.0 + 0j, 1.0 + 0j],
        'coherent': True}
    assert 'source_weights' not in summed.metadata
    assert 'source_depth' not in summed.pinned
    # The sum is complex pressure, so its dB view is defined.
    assert np.isfinite(summed.dB).any()


# ── (c) the sum behaves like the field of two sources ───────────────────


@pytest.mark.requires_binary
def test_two_sources_straddling_one_depth_add_to_twice_the_single_field():
    """In-phase sources at ``50 ± Δ`` m sum to ``2·p(50)`` up to a term of
    order ``(k_z·Δ)²``; at 100 Hz and Δ = 0.05 m that is below 1e-3. The
    antiphase pair is small but not zero — the odd combination is of order
    ``k_z·Δ`` — which pins that the two slabs really are different fields."""
    model = _kraken()
    env, receiver = _env(), _receiver()
    delta = 0.05
    pair = model.run(env, uacpy.Source(depths=[50.0 - delta, 50.0 + delta],
                                       frequencies=F0), receiver)
    single = model.run(env, uacpy.Source(depths=50.0, frequencies=F0),
                       receiver)
    even = pair.superpose([1.0, 1.0])
    assert _norm_relative_error(even.data, 2.0 * single.data) < 1e-3
    odd = pair.superpose([1.0, -1.0])
    odd_over_even = (np.linalg.norm(np.nan_to_num(odd.data))
                     / np.linalg.norm(np.nan_to_num(even.data)))
    assert 0.0 < odd_over_even < 0.05


@pytest.mark.requires_binary
def test_an_antiphase_pair_from_the_source_matches_the_hand_sum():
    """``Source(weights=[1, -1])`` at 30 m and 70 m: ``run().superpose()``
    is ``p(30) - p(70)`` built from two single-source runs, so the weights
    live in the sum and never reach the engine."""
    model = _kraken()
    env, receiver = _env(), _receiver()
    source = uacpy.Source(depths=[30.0, 70.0], frequencies=F0,
                          weights=[1.0, -1.0])
    summed = model.run(env, source, receiver).superpose()
    by_hand = (model.run(env, uacpy.Source(depths=30.0, frequencies=F0),
                         receiver).data
               - model.run(env, uacpy.Source(depths=70.0, frequencies=F0),
                           receiver).data)
    assert _relative_error(summed.data, by_hand) < 1e-9


# ── (d) time domain ─────────────────────────────────────────────────────


def _time_slab(z, trace):
    """A real time-domain slab: one receiver point, ``trace`` samples."""
    return Field(
        data=np.asarray(trace, dtype=float).reshape(1, 1, -1),
        coords={'depth': [50.0], 'range': [1000.0],
                'time': np.arange(len(trace)) / 1000.0},
        model='Test', frequencies=[90.0, 100.0, 110.0],
        source_depths=[z], phase_reference=None,
        metadata={'source_weights': np.array([1.0, -1.0])},
    )


def test_a_time_series_stack_superposes_to_the_summed_traces():
    """Real traces add sample by sample with real weights; the default
    weights come from the stamp and a complex weight is refused because a
    real trace has no phase to rotate."""
    a = np.sin(np.linspace(0.0, 6.0, 64))
    b = np.cos(np.linspace(0.0, 6.0, 64))
    stack = ResultStack([_time_slab(30.0, a), _time_slab(70.0, b)],
                        [30.0, 70.0])
    summed = stack.superpose([2.0, 0.5])
    assert not summed.is_complex
    assert 'time' in summed.coords
    np.testing.assert_allclose(summed.data[0, 0], 2.0 * a + 0.5 * b,
                               rtol=1e-12)
    np.testing.assert_allclose(stack.superpose().data[0, 0], a - b,
                               rtol=1e-12)
    with pytest.raises(ConfigurationError, match='real time-domain traces'):
        stack.superpose([1.0, 1j])


@pytest.mark.requires_binary
@pytest.mark.slow
def test_a_time_series_run_over_two_depths_stacks_real_traces():
    """Scooter TIME_SERIES over a two-depth ``Source`` returns two real
    time-domain slabs whose superposition is their sum."""
    env = uacpy.Environment(name='ts-pair', bathymetry=GUIDE_DEPTH,
                            ssp=1500.0)
    receiver = uacpy.Receiver(depths=[50.0], ranges=[2000.0])
    fs, n = 2000.0, 256
    t = np.arange(n) / fs
    waveform = np.sin(2 * np.pi * F0 * t) * np.hanning(n)
    stack = _scooter().run(
        env, uacpy.Source(depths=[30.0, 70.0], frequencies=F0), receiver,
        run_mode=RunMode.TIME_SERIES,
        frequencies=np.linspace(60.0, 140.0, 17),
        source_waveform=waveform, sample_rate=fs,
    )
    assert isinstance(stack, ResultStack)
    assert stack.n_slabs == 2
    for _, slab in stack:
        assert 'time' in slab.coords
        assert not slab.is_complex
    summed = stack.superpose()
    np.testing.assert_allclose(summed.data, stack[0].data + stack[1].data,
                               rtol=1e-12)


@pytest.mark.requires_binary
def test_a_bellhop_time_series_pair_shares_one_padded_time_axis():
    """Bellhop sizes its delay-and-sum window from each depth's own
    arrivals, so the two records differ in length; the loop pads the
    shorter one with silence at the end so the pair superposes, and the
    longer slab is the single run untouched."""
    env, receiver, waveform, fs = _bellhop_time_series_setup()
    model = _bellhop()
    stack = model.run(env, uacpy.Source(depths=[30.0, 70.0], frequencies=F0),
                      receiver, run_mode=RunMode.TIME_SERIES,
                      source_waveform=waveform, sample_rate=fs)
    singles = [model.run(env, uacpy.Source(depths=z, frequencies=F0),
                         receiver, run_mode=RunMode.TIME_SERIES,
                         source_waveform=waveform, sample_rate=fs)
               for z in (30.0, 70.0)]
    lengths = [f.coords['time'].size for f in singles]
    assert lengths[0] != lengths[1], "the pair no longer needs padding"
    n_max = max(lengths)
    for slab, single, n in zip(stack.slabs, singles, lengths):
        assert slab.coords['time'].size == n_max
        np.testing.assert_allclose(slab.coords['time'][:n],
                                   single.coords['time'], rtol=1e-12)
        np.testing.assert_array_equal(slab.data[..., :n], single.data)
        assert not np.any(slab.data[..., n:])
        assert slab.metadata['nt'] == n_max
    summed = stack.superpose()
    np.testing.assert_allclose(summed.data, stack[0].data + stack[1].data,
                               rtol=1e-12)


def _bellhop_time_series_setup():
    env = make_pekeris(name='ts-window', bathymetry=GUIDE_DEPTH)
    receiver = uacpy.Receiver(depths=[50.0], ranges=[2000.0])
    fs, n = 2000.0, 256
    t = np.arange(n) / fs
    return env, receiver, np.sin(2 * np.pi * F0 * t) * np.hanning(n), fs


@pytest.mark.requires_binary
def test_a_window_that_ends_before_the_first_echo_names_its_arrival():
    """``output_duration=1.0`` at 2 km ends before the 1.32 s first echo,
    so every trace is silence and the warning names that arrival time and
    the window; a window that holds the echoes raises no window notice."""
    env, receiver, waveform, fs = _bellhop_time_series_setup()
    model = _bellhop()
    source = uacpy.Source(depths=30.0, frequencies=F0)
    with pytest.warns(UserWarning,
                      match=r'\[0, 1\] s window.*earliest echo arrives at '
                            r'1\.3\d* s') as record:
        short = model.run(env, source, receiver,
                          run_mode=RunMode.TIME_SERIES,
                          source_waveform=waveform, sample_rate=fs,
                          output_duration=1.0)
    assert not np.any(short.data)
    assert any('all of the received energy' in str(w.message)
               for w in record)
    import warnings
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter('always')
        wide = model.run(env, source, receiver,
                         run_mode=RunMode.TIME_SERIES,
                         source_waveform=waveform, sample_rate=fs,
                         output_duration=2.0)
    assert np.any(wide.data)
    assert not any('window does not hold' in str(w.message) for w in record)


@pytest.mark.requires_binary
def test_a_complex_weight_on_a_time_domain_run_is_refused():
    env, receiver, waveform, fs = _bellhop_time_series_setup()
    with pytest.raises(ConfigurationError, match='complex weight cannot'):
        _bellhop().run(env, uacpy.Source(depths=30.0, frequencies=F0,
                                         weights=1j),
                       receiver, run_mode=RunMode.TIME_SERIES,
                       source_waveform=waveform, sample_rate=fs)
    real = _bellhop().run(env, uacpy.Source(depths=30.0, frequencies=F0,
                                            weights=-2.0),
                          receiver, run_mode=RunMode.TIME_SERIES,
                          source_waveform=waveform, sample_rate=fs)
    assert not real.is_complex
    assert real.metadata['superposed_sources'] == {'depths': [30.0],
                                                   'weights': [-2.0]}


def test_superpose_names_the_differing_lengths_of_a_time_axis_mismatch():
    short = _time_slab(30.0, np.ones(8))
    long = _time_slab(70.0, np.ones(10))
    with pytest.raises(ConfigurationError,
                       match=r"shape \(1, 1, 8\) vs \(1, 1, 10\), axis "
                             r"lengths \{'time': \(8, 10\)\}"):
        ResultStack([short, long], [30.0, 70.0]).superpose()


# ── (e) guards ──────────────────────────────────────────────────────────


def test_source_weights_default_to_ones_and_broadcast():
    src = uacpy.Source(depths=[10.0, 20.0, 30.0], frequencies=F0)
    np.testing.assert_array_equal(src.weights, [1.0, 1.0, 1.0])
    assert src.weights.dtype == np.complex128
    assert src.has_unit_weights
    scaled = uacpy.Source(depths=[10.0, 20.0], frequencies=F0, weights=2.0)
    np.testing.assert_array_equal(scaled.weights, [2.0, 2.0])
    assert not scaled.has_unit_weights
    one = uacpy.Source(depths=10.0, frequencies=F0, weights=[1j])
    np.testing.assert_array_equal(one.weights, [1j])


def test_a_length_one_weight_array_does_not_broadcast():
    """Only a scalar broadcasts; ``[2.0]`` on two depths is a length
    mismatch, while the same value as a scalar is accepted."""
    with pytest.raises(ConfigurationError,
                       match=r'2 depth\(s\) but 1 weight\(s\)'):
        uacpy.Source(depths=[10.0, 20.0], frequencies=F0, weights=[2.0])
    with pytest.raises(ConfigurationError,
                       match=r'2 depth\(s\) but 1 weight\(s\)'):
        uacpy.Source(depths=[10.0, 20.0], frequencies=F0,
                     weights=np.array([2.0]))
    ok = uacpy.Source(depths=[10.0, 20.0], frequencies=F0, weights=2.0)
    np.testing.assert_array_equal(ok.weights, [2.0, 2.0])
    both = uacpy.Source(depths=[10.0, 20.0], frequencies=F0,
                        weights=[2.0, 2.0])
    np.testing.assert_array_equal(both.weights, [2.0, 2.0])


def test_source_weights_length_must_match_the_depths():
    with pytest.raises(ConfigurationError,
                       match=r'2 depth\(s\) but 3 weight\(s\)'):
        uacpy.Source(depths=[10.0, 20.0], frequencies=F0,
                     weights=[1.0, 1.0, 1.0])
    with pytest.raises(ConfigurationError, match='scalar or a 1-D vector'):
        uacpy.Source(depths=[10.0, 20.0], frequencies=F0,
                     weights=[[1.0], [1.0]])


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf, np.nan * 1j])
def test_source_weights_must_be_finite(bad):
    with pytest.raises(ConfigurationError, match=r'weights\[1\]'):
        uacpy.Source(depths=[10.0, 20.0], frequencies=F0,
                     weights=[1.0, bad])


def test_source_repr_shows_weights_only_when_not_all_ones():
    plain = uacpy.Source(depths=[10.0, 20.0], frequencies=F0)
    assert 'weights' not in repr(plain)
    assert repr(uacpy.Source(depths=[10.0, 20.0], frequencies=F0,
                             weights=[1.0, 1.0])) == repr(plain)
    weighted = uacpy.Source(depths=[10.0, 20.0], frequencies=F0,
                            weights=[1.0, -1.0])
    assert 'weights=[(1+0j), (-1+0j)]' in repr(weighted)


def test_at_depth_copies_one_depth_with_unit_weight():
    pattern = np.array([[-90.0, -20.0], [0.0, 0.0], [90.0, -20.0]])
    src = uacpy.Source(depths=[10.0, 20.0], frequencies=[90.0, 110.0],
                       source_type='line', beam_pattern=pattern,
                       weights=[2.0, -1j])
    one = src.at_depth(1)
    np.testing.assert_array_equal(one.depths, [20.0])
    np.testing.assert_array_equal(one.frequencies, src.frequencies)
    assert one.source_type == 'line'
    np.testing.assert_array_equal(one.beam_pattern, pattern)
    assert one.has_unit_weights


def _complex_slab(z, value):
    return Field(
        data=np.full((2, 3), value, dtype=complex),
        coords={'depth': [10.0, 20.0], 'range': [100.0, 200.0, 300.0]},
        model='Test', frequencies=F0, source_depths=[z],
        phase_reference='travelling_wave',
    )


def test_superpose_refuses_a_wrong_length_or_non_finite_weight_vector():
    stack = ResultStack([_complex_slab(10.0, 1.0), _complex_slab(20.0, 2.0)],
                        [10.0, 20.0])
    with pytest.raises(ConfigurationError, match=r'2 slabs but 3 weight'):
        stack.superpose([1.0, 1.0, 1.0])
    with pytest.raises(ConfigurationError, match=r'2 slabs but 1 weight'):
        stack.superpose([1.0])
    with pytest.raises(ConfigurationError, match=r'weights\[0\]'):
        stack.superpose([np.nan, 1.0])
    # Both sides of the length guard: exactly one weight per slab adds.
    assert stack.superpose([1.0, 1.0]).data[0, 0] == 3.0 + 0j


def test_superpose_refuses_a_stack_that_lost_its_phase():
    """A dB-only stack cannot add coherently; the complex one it came from
    can, which pins that the refusal is about phase and not about the
    stack."""
    complex_stack = ResultStack(
        [_complex_slab(10.0, 1.0 + 1j), _complex_slab(20.0, 2.0)],
        [10.0, 20.0])
    assert complex_stack.superpose().data[0, 0] == 3.0 + 1j
    db_stack = ResultStack([s.to_dB() for s in complex_stack.slabs],
                           [10.0, 20.0])
    with pytest.raises(ConfigurationError,
                       match='not complex pressure.*coherent sum'):
        db_stack.superpose()


def test_an_incoherent_sum_adds_intensities_not_pressures():
    """Independent sources add in intensity: N identical slabs give
    10*log10(N) of gain, against the coherent 20*log10(N). The two answers
    differ, which is why the caller has to say which one they mean."""
    slabs = [_complex_slab(z, 1.0 + 0.0j) for z in (10.0, 20.0, 30.0)]
    stack = ResultStack(slabs, [10.0, 20.0, 30.0])
    one = -20.0 * np.log10(1.0)
    inc = stack.superpose(coherent=False)
    coh = stack.superpose()
    np.testing.assert_allclose(np.asarray(inc.data)[0, 0],
                               one - 10.0 * np.log10(3.0), atol=1e-12)
    np.testing.assert_allclose(np.asarray(coh.dB)[0, 0],
                               one - 20.0 * np.log10(3.0), atol=1e-12)


def test_an_incoherent_sum_returns_a_real_db_field_with_no_phase():
    """An incoherent sum has no phase, so it is stored the way every engine
    stores its own incoherent mode: real dB, ``phase_reference`` cleared."""
    stack = ResultStack([_complex_slab(10.0, 1.0), _complex_slab(20.0, 2.0)],
                        [10.0, 20.0])
    out = stack.superpose(coherent=False)
    assert not out.is_complex
    assert out.unit == 'dB'
    assert out.phase_reference is None
    assert out.metadata['superposed_sources']['coherent'] is False
    np.testing.assert_array_equal(out.source_depths, [10.0, 20.0])


def test_an_incoherent_sum_uses_only_the_weight_magnitudes():
    """Phase is meaningless in an intensity sum, so ``1j`` weighs the same
    as ``1`` — and the caller who wrote a complex weight is told so rather
    than having it quietly dropped."""
    stack = ResultStack([_complex_slab(10.0, 1.0), _complex_slab(20.0, 1.0)],
                        [10.0, 20.0])
    plain = stack.superpose([1.0, 1.0], coherent=False)
    with pytest.warns(UserWarning, match='magnitude'):
        rotated = stack.superpose([1.0, 1j], coherent=False)
    np.testing.assert_allclose(rotated.data, plain.data, atol=1e-12)
    # A quadrature weight is the one pair that CANNOT separate the two
    # sums: |p + i·p| is the quadrature sum, which is what adding
    # intensities computes. Show the difference with an in-phase pair,
    # where coherent gives 20*log10(2) and incoherent 10*log10(2).
    in_phase = stack.superpose([1.0, 1.0])
    np.testing.assert_allclose(np.asarray(in_phase.dB)[0, 0],
                               -20.0 * np.log10(2.0), atol=1e-12)
    np.testing.assert_allclose(np.asarray(plain.data)[0, 0],
                               -10.0 * np.log10(2.0), atol=1e-12)


def test_a_db_only_stack_adds_incoherently_though_it_cannot_add_coherently():
    """The asymmetry an intensity sum removes: a dB-only stack (Kraken's
    INCOHERENT_TL, OAST's TL) has lost its phase, so a coherent sum is
    undefined — but magnitudes are all an incoherent one needs."""
    complex_stack = ResultStack(
        [_complex_slab(10.0, 1.0), _complex_slab(20.0, 1.0)], [10.0, 20.0])
    db_stack = ResultStack([s.to_dB() for s in complex_stack.slabs],
                           [10.0, 20.0])
    with pytest.raises(ConfigurationError, match='coherent sum'):
        db_stack.superpose()
    out = db_stack.superpose(coherent=False)
    # Same two unit sources, so the same answer as adding them as pressure.
    np.testing.assert_allclose(
        out.data, complex_stack.superpose(coherent=False).data, atol=1e-9)


def test_an_incoherent_sum_refuses_time_domain_traces():
    """Adding intensities sample by sample does not produce a trace."""
    t = np.linspace(0.0, 1.0, 5)
    slabs = [Field(data=np.ones((2, 5)) * a,
                   coords={'depth': [10.0, 20.0], 'time': t},
                   model='Test', frequencies=F0, source_depths=[z],
                   phase_reference='time_domain_native')
             for z, a in ((10.0, 1.0), (20.0, 2.0))]
    with pytest.raises(ConfigurationError, match='time-domain'):
        ResultStack(slabs, [10.0, 20.0]).superpose(coherent=False)


def test_superpose_refuses_slabs_on_different_grids():
    a = _complex_slab(10.0, 1.0)
    b = Field(data=np.ones((2, 3), dtype=complex),
              coords={'depth': [10.0, 20.0],
                      'range': [100.0, 200.0, 400.0]},
              model='Test', frequencies=F0, source_depths=[20.0],
              phase_reference='travelling_wave')
    with pytest.raises(ConfigurationError, match='different grid'):
        ResultStack([a, b], [10.0, 20.0]).superpose()


def test_superpose_refuses_non_field_slabs():
    from uacpy.core.results import Rays
    rays = [Rays(rays=[], model='Test', frequencies=F0, source_depths=[z])
            for z in (10.0, 20.0)]
    with pytest.raises(ConfigurationError, match='not Field'):
        ResultStack(rays, [10.0, 20.0]).superpose()


@pytest.mark.requires_binary
def test_a_multi_depth_source_in_a_non_field_mode_keeps_its_refusal():
    """Mode shapes have no per-source sum, so ``MODES`` on a two-depth
    ``Source`` still raises and names the field modes that do stack."""
    model = _kraken()
    source = uacpy.Source(depths=[30.0, 70.0], frequencies=F0)
    with pytest.raises(ConfigurationError,
                       match='single source depth per MODES run'):
        model.run(_env(), source, _receiver(), run_mode=RunMode.MODES)
    with pytest.raises(ConfigurationError, match='field modes'):
        model.validate_inputs(_env(), source, _receiver(),
                              run_mode=RunMode.MODES)


@pytest.mark.requires_binary
def test_the_zero_range_refusal_fires_before_any_per_depth_run():
    """The loop runs each depth through the wrapper's own validation, so a
    receiver with no positive range is refused on the first depth."""
    model = _kraken()
    source = uacpy.Source(depths=[30.0, 70.0], frequencies=F0)
    with pytest.raises(ConfigurationError, match='largest range is 0 m'):
        model.run(_env(), source,
                  uacpy.Receiver(depths=[50.0], ranges=[0.0]))
