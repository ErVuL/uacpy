"""A multi-depth ``Source`` runs once per depth on every field model and
returns ONE ``Field`` whose leading ``source_depth`` axis holds each source's
field; ``Field.superpose`` adds them.

Two sources are driven by adding the complex field of each one: the engines
are linear in the source amplitude, so ``Σ wᵢ·pᵢ`` over the unit-source
planes is the field of the weighted array. That every field engine's plane is
its single-depth run, and that one weighted depth is ``w`` times the unit run,
is held for every registered engine by ``test_engine_conformance.py``. The
tests here pin (a) the single-depth contract and the joining machinery, (b)
that ``superpose`` is that linear sum, (c) that the sum behaves like a field
of two sources, (d) the time-domain case, and (e) the guards on weights and
on fields that cannot add.
"""

from pathlib import Path


import numpy as np
import pytest

import uacpy
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.results import Field, ResultStack, stack_fields
from uacpy.core.results._superposition import (_check_field_weightable,
                                               split_source_depths)
from uacpy.models import RunMode
from uacpy.tests.conftest import make_pekeris
from uacpy.tests.conftest import recorded_warnings

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
    from uacpy.core.boundary import SedimentLayer
    from uacpy.core.bottom import Bottom, SeabedColumn
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


def _bellhop():
    from uacpy.models.bellhop import Bellhop
    return Bellhop(verbose=False)


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


def _pair(*fields):
    """The fields of several one-depth sources as one Field over their
    ``source_depth`` axis, as a multi-depth run returns them."""
    return stack_fields(fields, [f.source_depths[0] for f in fields])


@pytest.fixture(scope='module')
def kraken_pair():
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


# ── (a) the single-depth contract and the stacking machinery ─────────


@pytest.mark.requires_binary
def test_a_single_depth_source_returns_a_field_with_no_weight_stamp():
    """The loop only fires above one depth: the single-source contract is a
    plain ``Field`` with no weight stamp, and an explicit unit weight is
    the same run."""
    model = _kraken()
    field = model.run(_env(), uacpy.Source(depths=50.0, frequencies=F0),
                      _receiver())
    assert isinstance(field, Field)
    assert field.source_weights is None
    assert 'superposed_sources' not in field.metadata
    unit = model.run(_env(), uacpy.Source(depths=50.0, frequencies=F0,
                                          weights=1.0), _receiver())
    np.testing.assert_array_equal(unit.data, field.data)
    assert 'superposed_sources' not in unit.metadata


def test_every_concrete_model_inherits_run_and_implements_the_stage_hooks():
    """``run`` is the base class's template method — validation, the
    per-depth loop, the weight — and no wrapper redefines it, so a subclass
    that overrides it and delegates to ``super().run()`` passes through the
    template once. Every engine implements the four stage hooks of
    :class:`PropagationModel`.
    """
    from uacpy.models._registry import engine_classes
    from uacpy.models.base import PropagationModel, _STAGE_HOOKS
    for cls in engine_classes().values():
        assert 'run' not in cls.__dict__, cls
        assert cls.run is PropagationModel.run, cls
        assert all(hook in cls.__dict__ for hook in _STAGE_HOOKS), cls


def test_a_run_override_whose_run_mode_default_is_not_none_is_refused():
    """``super().run()`` resolves ``run_mode=None`` through
    ``_default_run_mode()``, so an override declaring its own default for
    ``run_mode`` would be a second decider of the default mode;
    ``__init_subclass__`` refuses the declaration."""
    from uacpy.models.ram import RAM
    with pytest.raises(TypeError, match='run_mode'):
        class DefaultedMode(RAM):
            def run(self, env, source, receiver,
                    run_mode=RunMode.COHERENT_TL, *,
                    frequencies=None, source_waveform=None,
                    sample_rate=None, output_duration=None):
                return None


def test_a_run_override_with_a_var_positional_sink_is_refused():
    """The mirror of the ``**kwargs`` rule: a ``*args`` sink swallows
    exactly the unknown positional arguments the keyword-only rule below
    it exists to refuse."""
    from uacpy.models.ram import RAM
    with pytest.raises(TypeError, match=r'\*args'):
        class VarPositionalSink(RAM):
            def run(self, env, source, receiver, run_mode=None,
                    *args, frequencies=None, source_waveform=None,
                    sample_rate=None, output_duration=None):
                return None


def test_a_run_override_that_requires_run_mode_is_accepted():
    """An override that declares no default for ``run_mode`` decides no
    default mode, so it is accepted — only a non-None default is refused."""
    from uacpy.models.ram import RAM

    class RequiredMode(RAM):
        def run(self, env, source, receiver, run_mode, *,
                frequencies=None, source_waveform=None,
                sample_rate=None, output_duration=None):
            return None

    assert RequiredMode.run is not None


def test_a_class_that_declares_no_spec_is_told_what_it_defined():
    """The message names what ``__init_subclass__`` actually saw, so a
    class that defined ``run`` is not told it defined the stage hooks."""
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
    pair = model.run(_env(), uacpy.Source(depths=[30.0, 70.0],
                                          frequencies=F0, weights=[1.0, 2.0]),
                     _receiver(), run_mode=RunMode.INCOHERENT_TL)
    assert isinstance(pair, Field) and list(pair.coords)[0] == 'source_depth'
    assert pair.superpose(coherent=False).unit == 'dB'
    with pytest.raises(ConfigurationError, match='coherent sum'):
        pair.superpose()


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


@pytest.mark.requires_binary  # constructs Kraken (resolves its binary)
@pytest.mark.parametrize('mode, weight, refused', [
    (RunMode.INCOHERENT_TL, 2.0, True),
    (RunMode.INCOHERENT_TL, 1.0, False),
    (RunMode.COHERENT_TL, 2.0, False),
])
def test_a_one_depth_weight_on_a_db_mode_is_refused_before_the_engine_runs(
        mode, weight, refused):
    """``validate_inputs`` stops at stage 3 and launches nothing, so the
    dB refusal is reached there; a unit weight and a complex-pressure mode
    pass."""
    source = uacpy.Source(depths=50.0, frequencies=F0, weights=weight)
    if refused:
        with pytest.raises(ConfigurationError,
                           match='a weight cannot scale it'):
            _kraken().validate_inputs(_env(), source, _receiver(),
                                      run_mode=mode)
    else:
        _kraken().validate_inputs(_env(), source, _receiver(),
                                  run_mode=mode)


@pytest.mark.requires_binary
def test_a_pinned_work_dir_keeps_one_scratch_set_per_source_depth(tmp_path):
    """Each depth of the loop runs in its own ``source_depth_<z>m``
    subdirectory of the pinned ``work_dir``, so each depth's ``*_file``
    paths stay its own — one entry per source depth in the field's metadata
    — instead of all naming the last depth's files. RAM is the engine that
    genuinely marches once per depth."""
    from uacpy.models.ram import RAM
    depths = [30.0, 70.0]
    field = RAM(verbose=False, work_dir=tmp_path, cleanup=False).run(
        _env(), uacpy.Source(depths=depths, frequencies=F0), _receiver())
    key = next(k for k in sorted(field.metadata) if k.endswith('_file'))
    files = field.metadata[key]
    assert isinstance(files, list) and len(files) == 2, files
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
    ``.grn`` / ``.shd`` carries the ``NSz`` axis the reader puts on the
    Field's ``source_depth`` axis. A multi-depth run in these modes
    therefore launches the binary once — one deck in the pinned
    ``work_dir``, no per-depth subdirectory — and each plane is still that
    depth's stand-alone run.

    Both are exact. Scooter's depth mesh comes from the media, so a source
    depth touches no other depth's answer. Kraken's receivers are placed on
    the mode-tabulation grid (``_resolve_field_launch``) AND written to the
    ``.flp`` in full rather than through FIELD's subtabulate shortcut, which
    would recompute them in single precision and leave them a few ULPs off
    their own nodes — enough to put the interpolation weight at ~1e-7
    instead of 0 and let a source depth elsewhere in the grid move a
    near-null receiver by 6e-6."""
    model = factory()
    model.work_dir, model.cleanup = tmp_path, False
    env, receiver = _env(), _receiver()
    depths = [30.0, 70.0]
    field = model.run(env, uacpy.Source(depths=depths, frequencies=F0),
                      receiver, run_mode=mode)
    assert isinstance(field, Field)
    assert field.coords['source_depth'].tolist() == depths
    decks = list(tmp_path.rglob('*.env'))
    assert len(decks) == 1, decks
    assert not list(tmp_path.glob('source_depth_*')), \
        "the run was split into one deck per depth"
    np.testing.assert_array_equal(field.source_weights, [1.0, 1.0])
    np.testing.assert_array_equal(field.source_depths, depths)
    for i, (z, plane) in enumerate(zip(depths, split_source_depths(field))):
        single = factory().run(env, uacpy.Source(depths=z, frequencies=F0),
                               receiver, run_mode=mode)
        np.testing.assert_array_equal(plane.source_depths, [z])
        assert plane.is_complex == single.is_complex
        err = _relative_error(plane.data, single.data)
        assert err <= tolerance, f"plane {i} (z={z} m) drifted by {err:.2e}"


@pytest.mark.requires_binary
def test_a_line_source_field_levels_each_source_at_its_own_depth():
    """The line-source level is sqrt(k0) at the SOURCE depth, so on a
    sound-speed gradient each plane of a native multi-depth run must use
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
    field = model.run(grad, uacpy.Source(depths=depths, frequencies=F0,
                                         source_type='line'), receiver)
    for i, z in enumerate(depths):
        single = _scooter().run(
            grad, uacpy.Source(depths=z, frequencies=F0, source_type='line'),
            receiver)
        assert _relative_error(field.data[i], single.data) < 1e-9, (
            f"plane {i} (z={z} m) is not the stand-alone line-source run")


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
    with recorded_warnings() as caught:
        _bellhop().run(layered, weighted, receiver)
    spurious = [str(w.message) for w in caught
                if 'is not applied' in str(w.message)]
    assert not spurious, spurious


def test_a_complex_weight_needs_a_phase_reference_not_just_complex_data():
    """Bellhop's incoherent TL stores complex pressure whose phase is an
    artefact of AT's storage (``phase_reference`` is None). Rotating it by
    a complex weight would record a source phase the field cannot carry,
    so a complex weight is refused there; a real one still scales it."""
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


def test_a_multi_depth_field_with_unapplied_weights_warns_on_its_views():
    """A multi-depth run's field holds unit-amplitude planes whatever the
    Source's weights, so its dB / TL / pressure view and its panel plot show
    a field the weights never touched; they say so, naming ``superpose``. A
    unit-weight field, and the one-source field ``superpose`` returns, are
    silent."""
    import warnings
    weighted = _pair(_complex_slab(10.0, 1.0), _complex_slab(20.0, 2.0))
    weighted.source_weights = np.array([1.0, -1.0])
    for view in ('dB', 'tl', 'p'):
        with pytest.warns(UserWarning, match=r'weights.*superpose'):
            getattr(weighted, view)
    unit = _pair(_complex_slab(10.0, 1.0), _complex_slab(20.0, 2.0))
    unit.source_weights = np.array([1.0, 1.0])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        unit.dB
        unit.tl
        unit.p
        weighted.superpose().dB


# ── (b) superpose is the linear sum ─────────────────────────────────────


def test_superpose_is_the_weighted_sum_of_the_slabs(kraken_pair):
    """``superpose([w1, w2])`` is ``w1·p1 + w2·p2`` cell by cell, for real
    and complex weights, and ``[1, 0]`` returns slab 0 unchanged."""
    p1, p2 = kraken_pair.data[0], kraken_pair.data[1]
    for w1, w2 in ((1.0, 1.0), (0.3, -2.0), (1.0, 1j), (0.5 + 0.5j, -1j)):
        summed = kraken_pair.superpose([w1, w2])
        assert isinstance(summed, Field)
        assert summed.is_complex
        assert _relative_error(summed.data, w1 * p1 + w2 * p2) < 1e-12
    only_first = kraken_pair.superpose([1.0, 0.0])
    np.testing.assert_array_equal(only_first.data, p1)


def test_superpose_reads_the_source_weights_by_default(kraken_pair):
    """With no argument the field sums with the weights the ``Source``
    carried, here ``[1, -1]``: the same result as applying them by hand."""
    np.testing.assert_array_equal(kraken_pair.source_weights,
                                  [1.0, -1.0])
    by_default = kraken_pair.superpose()
    by_hand = kraken_pair.data[0] - kraken_pair.data[1]
    assert _relative_error(by_default.data, by_hand) < 1e-12


def test_the_superposed_field_carries_the_runs_identity(kraken_pair):
    """The sum keeps the receiver grid, phase reference and model, drops
    the ``source_depth`` axis, keeps ``source_depths`` as every depth
    summed, and records what was added."""
    summed = kraken_pair.superpose([1.0, 1.0])
    first = kraken_pair.isel(source_depth=0)
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
    assert summed.source_weights is None
    assert 'source_depth' not in summed.pinned
    # The sum is complex pressure, so its dB view is defined.
    assert np.isfinite(summed.dB).any()


# ── (c) the sum behaves like the field of two sources ───────────────────


@pytest.mark.requires_binary
def test_two_sources_straddling_one_depth_add_to_twice_the_single_field():
    """In-phase sources at ``50 ± Δ`` m sum to ``2·p(50)`` up to a term of
    order ``(k_z·Δ)²``; at 100 Hz and Δ = 0.05 m that is below 1e-3. The
    antiphase pair is small but not zero — the odd combination is of order
    ``k_z·Δ`` — which pins that the two planes really are different fields."""
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
    """A real time-domain field of one source: one receiver point,
    ``trace`` samples."""
    return Field(
        data=np.asarray(trace, dtype=float).reshape(1, 1, -1),
        coords={'depth': [50.0], 'range': [1000.0],
                'time': np.arange(len(trace)) / 1000.0},
        model='Test', frequencies=[90.0, 100.0, 110.0],
        source_depths=[z], phase_reference=None,
        source_weights=np.array([1.0, -1.0]),
    )


def test_a_time_series_field_superposes_to_the_summed_traces():
    """Real traces add sample by sample with real weights; the default
    weights come from the stamp and a complex weight is refused because a
    real trace has no phase to rotate."""
    a = np.sin(np.linspace(0.0, 6.0, 64))
    b = np.cos(np.linspace(0.0, 6.0, 64))
    stack = _pair(_time_slab(30.0, a), _time_slab(70.0, b))
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
def test_a_time_series_run_over_two_depths_returns_real_traces_per_depth():
    """Scooter TIME_SERIES over a two-depth ``Source`` returns one real
    time-domain Field, one trace set per depth, whose superposition is their
    sum."""
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
    assert isinstance(stack, Field)
    assert list(stack.coords) == ['source_depth', 'depth', 'range', 'time']
    assert not stack.is_complex
    summed = stack.superpose()
    np.testing.assert_allclose(summed.data, stack.data[0] + stack.data[1],
                               rtol=1e-12)


@pytest.mark.requires_binary
def test_a_bellhop_time_series_pair_shares_one_padded_time_axis():
    """Bellhop sizes its delay-and-sum window from each depth's own
    arrivals, so the two records differ in length; the loop pads the
    shorter one with silence at the end so the pair shares one time axis,
    and the longer plane is the single run untouched."""
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
    assert stack.coords['time'].size == n_max
    assert stack.metadata['nt'] == n_max
    for plane, single, n in zip(stack.data, singles, lengths):
        np.testing.assert_allclose(stack.coords['time'][:n],
                                   single.coords['time'], rtol=1e-12)
        np.testing.assert_array_equal(plane[..., :n], single.data)
        assert not np.any(plane[..., n:])
    summed = stack.superpose()
    np.testing.assert_allclose(summed.data, stack.data[0] + stack.data[1],
                               rtol=1e-12)


def _bellhop_time_series_setup():
    env = make_pekeris(name='ts-window', bathymetry=GUIDE_DEPTH)
    receiver = uacpy.Receiver(depths=[50.0], ranges=[2000.0])
    fs, n = 2000.0, 256
    t = np.arange(n) / fs
    return env, receiver, np.sin(2 * np.pi * F0 * t) * np.hanning(n), fs


@pytest.mark.requires_binary
def test_a_window_that_ends_before_the_first_echo_names_its_arrival():
    """A window opened at ``t_start=0`` with ``output_duration=1.0`` at 2 km
    ends before the 1.32 s first echo, so every trace is silence and the
    warning names that arrival time and the window; a window that holds the
    echoes raises no window notice."""
    env, receiver, waveform, fs = _bellhop_time_series_setup()
    model = _bellhop()
    source = uacpy.Source(depths=30.0, frequencies=F0)
    with pytest.warns(UserWarning,
                      match=r'\[0, 1\] s window.*earliest echo arrives at '
                            r'1\.3\d* s') as record:
        short = model.run(env, source, receiver,
                          run_mode=RunMode.TIME_SERIES,
                          source_waveform=waveform, sample_rate=fs,
                          t_start=0.0, output_duration=1.0)
    assert not np.any(short.data)
    assert any('all of the received energy' in str(w.message)
               for w in record)
    with recorded_warnings() as record:
        wide = model.run(env, source, receiver,
                         run_mode=RunMode.TIME_SERIES,
                         source_waveform=waveform, sample_rate=fs,
                         t_start=0.0, output_duration=2.0)
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


def test_joining_names_the_differing_lengths_of_a_time_axis_mismatch():
    short = _time_slab(30.0, np.ones(8))
    long = _time_slab(70.0, np.ones(10))
    with pytest.raises(ConfigurationError,
                       match=r"shape \(1, 1, 8\) vs \(1, 1, 10\), axis "
                             r"lengths \{'time': \(8, 10\)\}"):
        _pair(short, long)


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
    assert 'weights [1, -1]' in repr(weighted)


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
    pair = _pair(_complex_slab(10.0, 1.0), _complex_slab(20.0, 2.0))
    with pytest.raises(ConfigurationError,
                       match=r'2 source depths but 3 weight'):
        pair.superpose([1.0, 1.0, 1.0])
    with pytest.raises(ConfigurationError,
                       match=r'2 source depths but 1 weight'):
        pair.superpose([1.0])
    with pytest.raises(ConfigurationError, match=r'weights\[0\]'):
        pair.superpose([np.nan, 1.0])
    # Both sides of the length guard: exactly one weight per depth adds.
    assert pair.superpose([1.0, 1.0]).data[0, 0] == 3.0 + 0j


def test_superpose_refuses_a_field_that_lost_its_phase():
    """A dB-only field cannot add coherently; the complex one it came from
    can, which pins that the refusal is about phase and not about the
    field."""
    pair = _pair(_complex_slab(10.0, 1.0 + 1j), _complex_slab(20.0, 2.0))
    assert pair.superpose().data[0, 0] == 3.0 + 1j
    with pytest.raises(ConfigurationError,
                       match='not complex pressure.*coherent sum'):
        pair.to_dB().superpose()


def test_the_coherent_refusal_of_a_db_field_names_the_intensity_sum():
    """The sources of a dB-only field still add in intensity, so the
    coherent sum's refusal names superpose(coherent=False), which then
    runs; the one-depth weight a run applies has no such sum to offer."""
    from uacpy.core.results._superposition import _weighted_sum
    levels = _pair(_complex_slab(10.0, 1.0), _complex_slab(20.0, 2.0)).to_dB()
    with pytest.raises(ConfigurationError,
                       match=r"(?s)Field\.superpose:.*How to fix:.*"
                             r"superpose\(coherent=False\)"):
        levels.superpose()
    assert levels.superpose(coherent=False).unit == 'dB'
    one = _complex_slab(10.0, 1.0).to_dB()
    with pytest.raises(ConfigurationError,
                       match=r'run: the field is real .* not complex') as err:
        _weighted_sum(one, [one.data], [2.0], where='run')
    assert 'coherent=False' not in str(err.value)


def test_superpose_refuses_a_field_with_no_source_depth_axis():
    """A one-source field has nothing to sum: the refusal names the run
    that gives the axis; the same field with the axis sums."""
    one = _complex_slab(10.0, 1.0)
    with pytest.raises(ConfigurationError,
                       match="no 'source_depth' axis") as ei:
        one.superpose()
    assert 'Source(depths=[...]' in ei.value.remediation
    np.testing.assert_array_equal(_pair(one).superpose().data, one.data)


def test_superpose_sums_the_source_depth_axis_wherever_it_lies():
    """The axis is found by name: a field built with ``source_depth``
    after ``depth`` sums along it as one built with it first."""
    pair = _pair(_complex_slab(10.0, 1.0), _complex_slab(20.0, 2.0 - 1j))
    moved = Field(data=np.moveaxis(pair.data, 0, 1),
                  coords={'depth': pair.coords['depth'],
                          'source_depth': pair.coords['source_depth'],
                          'range': pair.coords['range']},
                  model='Test', frequencies=F0,
                  source_depths=pair.coords['source_depth'],
                  phase_reference='travelling_wave')
    np.testing.assert_array_equal(moved.superpose([1.0, 1j]).data,
                                  pair.superpose([1.0, 1j]).data)


def test_an_incoherent_sum_adds_intensities_not_pressures():
    """Independent sources add in intensity: N identical planes give
    10*log10(N) of gain, against the coherent 20*log10(N). The two answers
    differ, which is why the caller has to say which one they mean."""
    field = _pair(*[_complex_slab(z, 1.0 + 0.0j) for z in (10.0, 20.0, 30.0)])
    one = -20.0 * np.log10(1.0)
    inc = field.superpose(coherent=False)
    coh = field.superpose()
    np.testing.assert_allclose(np.asarray(inc.data)[0, 0],
                               one - 10.0 * np.log10(3.0), atol=1e-12)
    np.testing.assert_allclose(np.asarray(coh.dB)[0, 0],
                               one - 20.0 * np.log10(3.0), atol=1e-12)


def test_an_incoherent_sum_returns_a_real_db_field_with_no_phase():
    """An incoherent sum has no phase, so it is stored the way every engine
    stores its own incoherent mode: real dB, ``phase_reference`` cleared."""
    out = _pair(_complex_slab(10.0, 1.0),
                _complex_slab(20.0, 2.0)).superpose(coherent=False)
    assert not out.is_complex
    assert out.unit == 'dB'
    assert out.coherent is False
    assert out.phase_reference is None
    assert out.metadata['superposed_sources']['coherent'] is False
    np.testing.assert_array_equal(out.source_depths, [10.0, 20.0])
    assert 'source_depth' not in out.coords


def test_an_incoherent_sum_uses_only_the_weight_magnitudes():
    """Phase is meaningless in an intensity sum, so ``1j`` weighs the same
    as ``1`` — and the caller who wrote a complex weight is told so rather
    than having it quietly dropped."""
    pair = _pair(_complex_slab(10.0, 1.0), _complex_slab(20.0, 1.0))
    plain = pair.superpose([1.0, 1.0], coherent=False)
    with pytest.warns(UserWarning, match='magnitude'):
        rotated = pair.superpose([1.0, 1j], coherent=False)
    np.testing.assert_allclose(rotated.data, plain.data, atol=1e-12)
    # A quadrature weight is the one pair that CANNOT separate the two
    # sums: |p + i·p| is the quadrature sum, which is what adding
    # intensities computes. Show the difference with an in-phase pair,
    # where coherent gives 20*log10(2) and incoherent 10*log10(2).
    in_phase = pair.superpose([1.0, 1.0])
    np.testing.assert_allclose(np.asarray(in_phase.dB)[0, 0],
                               -20.0 * np.log10(2.0), atol=1e-12)
    np.testing.assert_allclose(np.asarray(plain.data)[0, 0],
                               -10.0 * np.log10(2.0), atol=1e-12)


def test_a_db_only_field_adds_incoherently_though_it_cannot_add_coherently():
    """The asymmetry an intensity sum removes: a dB-only field (Kraken's
    INCOHERENT_TL, OAST's TL) has lost its phase, so a coherent sum is
    undefined — but magnitudes are all an incoherent one needs."""
    complex_pair = _pair(_complex_slab(10.0, 1.0), _complex_slab(20.0, 1.0))
    db_pair = complex_pair.to_dB()
    with pytest.raises(ConfigurationError, match='coherent sum'):
        db_pair.superpose()
    out = db_pair.superpose(coherent=False)
    # Same two unit sources, so the same answer as adding them as pressure.
    np.testing.assert_allclose(
        out.data, complex_pair.superpose(coherent=False).data, atol=1e-9)


def test_an_incoherent_sum_refuses_time_domain_traces():
    """Adding intensities sample by sample does not produce a trace."""
    t = np.linspace(0.0, 1.0, 5)
    slabs = [Field(data=np.ones((2, 5)) * a,
                   coords={'depth': [10.0, 20.0], 'time': t},
                   model='Test', frequencies=F0, source_depths=[z],
                   phase_reference='time_domain_native')
             for z, a in ((10.0, 1.0), (20.0, 2.0))]
    with pytest.raises(ConfigurationError, match='time-domain'):
        _pair(*slabs).superpose(coherent=False)


class TestAnIntensitySumReadsEachSourceAsWhatItIs:
    """``superpose(coherent=False)`` adds the power behind each source's
    field: a dB level or loss through its sign, real linear magnitudes in
    ``'Pa'`` as they stand, and nothing that is not a power."""

    @staticmethod
    def _pair(**field):
        def one(z):
            return Field(coords={'depth': [10.0], 'range': [100.0, 200.0]},
                         source_depths=[z], **field)
        return _pair(one(10.0), one(20.0))

    def test_real_linear_magnitudes_add_as_magnitudes(self):
        out = self._pair(data=[[1.0, 2.0]], unit='Pa').superpose(
            coherent=False)
        # |p| = sqrt(2) * [1, 2], so TL = -20 log10 of that.
        np.testing.assert_allclose(
            out.data, -20.0 * np.log10(np.sqrt(2.0) * np.array([[1.0, 2.0]])),
            atol=1e-12)

    def test_dB_levels_and_losses_add_in_intensity(self):
        level = self._pair(data=[[120.0, 100.0]], kind='level', unit='dB')
        np.testing.assert_allclose(
            level.superpose(coherent=False).data,
            [[120.0 + 10 * np.log10(2.0), 100.0 + 10 * np.log10(2.0)]],
            atol=1e-12)
        loss = self._pair(data=[[60.0, 80.0]], unit='dB')
        np.testing.assert_allclose(
            loss.superpose(coherent=False).data,
            [[60.0 - 10 * np.log10(2.0), 80.0 - 10 * np.log10(2.0)]],
            atol=1e-12)

    def test_a_cell_no_energy_reached_keeps_the_no_energy_marker(self):
        """Two zero-pressure sources sum to the package's no-energy marker
        (600 dB loss), not to a finite loss 3 dB under it."""
        from uacpy.core.constants import NO_ENERGY_DB
        out = self._pair(data=[[0.0 + 0j, 1.0 + 0j]]).superpose(
            coherent=False)
        assert out.data[0, 0] == NO_ENERGY_DB
        np.testing.assert_allclose(out.data[0, 1], -10 * np.log10(2.0),
                                   atol=1e-12)

    @pytest.mark.parametrize('kind, unit, data', [
        ('difference', 'dB', [[3.0, -3.0]]),
        ('probability_of_detection', '1', [[0.5, 0.9]]),
        ('signal_excess', 'dB', [[3.0, -3.0]]),
        ('peak_pressure', 'dB', [[150.0, 140.0]]),
    ])
    def test_a_quantity_that_is_not_a_power_is_refused(self, kind, unit,
                                                       data):
        field = self._pair(data=data, kind=kind, unit=unit)
        with pytest.raises(ConfigurationError,
                           match=rf"superpose\(coherent=False\): the field "
                                 rf"is a '{kind}' field"):
            field.superpose(coherent=False)

    def test_a_snapshot_of_a_trace_is_refused(self):
        field = self._pair(data=[[0.3, -0.1]], unit='Pa',
                           pinned={'time': 0.5})
        with pytest.raises(ConfigurationError, match='snapshots of one'):
            field.superpose(coherent=False)


class TestOneSourceWeightIsTheSuperposeSum:
    """The n = 1 weight a run applies to a one-depth ``Source`` and
    ``Field.superpose`` form one sum, ``_weighted_sum``."""

    @staticmethod
    def _one():
        return Field(data=np.array([[1.0 + 2.0j, -3.0 + 0.5j]]),
                     coords={'depth': np.array([10.0]),
                             'range': np.array([100.0, 200.0])},
                     model='Test', frequencies=100.0,
                     source_depths=np.array([50.0]),
                     phase_reference='travelling_wave')

    def test_both_paths_call_the_one_sum(self, monkeypatch):
        from uacpy.core.results import _superposition
        from uacpy.models import _stacking
        calls = []
        real = _superposition._weighted_sum

        def spy(field, planes, weights, *, where, **kwargs):
            calls.append(where)
            return real(field, planes, weights, where=where, **kwargs)
        monkeypatch.setattr(_superposition, '_weighted_sum', spy)
        monkeypatch.setattr(_stacking, '_weighted_sum', spy)
        _pair(self._one()).superpose([2.0 - 1.0j])
        source = uacpy.Source(depths=50.0, frequencies=100.0,
                              weights=[2.0 - 1.0j])
        _stacking.apply_single_source_weight(self._one(), source,
                                             model_name='Test')
        assert len(calls) == 2

    def test_the_one_source_weight_equals_the_one_source_superpose(self):
        from uacpy.models import _stacking
        w = 2.0 - 1.0j
        summed = _pair(self._one()).superpose([w])
        source = uacpy.Source(depths=50.0, frequencies=100.0, weights=[w])
        weighted = _stacking.apply_single_source_weight(
            self._one(), source, model_name='Test')
        np.testing.assert_array_equal(weighted.data, summed.data)


# ── (f) joining the fields of several sources ──────────────────────────


class TestTheFieldsOfSeveralSourcesJoinOnOneAxis:
    """``stack_fields`` is how a run joins its per-depth fields: one grid,
    one quantity, one identity apart from the axis it adds; what the depths
    disagree on is kept per depth."""

    def test_the_axis_leads_and_lists_the_depths(self):
        pair = _pair(_complex_slab(10.0, 1.0), _complex_slab(20.0, 2.0))
        assert list(pair.coords) == ['source_depth', 'depth', 'range']
        np.testing.assert_array_equal(pair.coords['source_depth'],
                                      [10.0, 20.0])
        np.testing.assert_array_equal(pair.source_depths, [10.0, 20.0])
        assert pair.data.shape == (2, 2, 3)

    def test_a_split_field_joins_back_into_the_field(self):
        pair = _pair(_complex_slab(10.0, 1.0), _complex_slab(20.0, 2.0))
        planes = split_source_depths(pair)
        np.testing.assert_array_equal(planes[1].data, pair.data[1])
        np.testing.assert_array_equal(planes[1].source_depths, [20.0])
        assert 'source_depth' not in planes[1].coords
        assert 'source_depth' not in planes[1].pinned
        again = stack_fields(planes, pair.coords['source_depth'])
        np.testing.assert_array_equal(again.data, pair.data)

    def test_metadata_the_depths_disagree_on_is_kept_per_depth(self):
        a, b = _complex_slab(10.0, 1.0), _complex_slab(20.0, 2.0)
        a.metadata.update(prt_file='/x/source_depth_10m/a.prt', nt=8)
        b.metadata.update(prt_file='/x/source_depth_20m/a.prt', nt=8)
        pair = _pair(a, b)
        assert pair.metadata['prt_file'] == ['/x/source_depth_10m/a.prt',
                                             '/x/source_depth_20m/a.prt']
        assert pair.metadata['nt'] == 8

    def test_components_held_once_are_kept_and_differing_ones_stacked(self):
        from uacpy.tests._synthetic_fields import _bounce
        a, b = _complex_slab(10.0, 1.0), _complex_slab(20.0, 2.0)
        shared = _bounce()
        a._components = {'bounce': shared}
        b._components = {'bounce': shared}
        assert _pair(a, b).components['bounce'] is shared
        b._components = {'bounce': _bounce()}
        joined = _pair(a, b).components['bounce']
        assert isinstance(joined, ResultStack) and len(joined) == 2

    def test_selecting_a_source_depth_narrows_the_components_to_it(self):
        """A component that runs over source depth (each depth's own
        table, joined into a stack) follows a selection of the field: one
        depth gets its own table, a window keeps the stack of the kept
        depths, an evaluation between depths keeps none of it; a component
        every depth shares is kept as it is."""
        from uacpy.tests._synthetic_fields import _bounce
        a, b = _complex_slab(10.0, 1.0), _complex_slab(20.0, 2.0)
        shared, own_a, own_b = _bounce(), _bounce(), _bounce()
        a._components = {'bounce': own_a, 'mean_field': shared}
        b._components = {'bounce': own_b, 'mean_field': shared}
        pair = _pair(a, b)
        assert isinstance(pair.components['bounce'], ResultStack)
        assert pair.isel(source_depth=1).components['bounce'] is own_b
        assert pair.at(source_depth=10.0).components['bounce'] is own_a
        planes = split_source_depths(pair)
        assert [p.components['bounce'] for p in planes] == [own_a, own_b]
        kept = pair.window(source_depth=(15.0, 25.0)).components['bounce']
        assert isinstance(kept, ResultStack) and kept.slabs == [own_b]
        between = pair.eval(source_depth=15.0).components
        assert 'bounce' not in between
        assert between['mean_field'] is shared
        assert pair.isel(range=0).components['bounce'] is \
            pair.components['bounce']

    def test_selecting_source_depths_narrows_the_weights_to_them(self):
        """The Source weights follow the ``source_depth`` axis: a window
        keeps the weight of each depth it keeps, so ``superpose()`` is the
        weighted sum of the kept planes; pinning one depth (at, isel, eval,
        max, a split plane, an xarray plane) leaves one source's field,
        which carries no weight; an axis widened to a depth no weight was
        given for carries none. Joining the split planes again restores the
        run's weights."""
        from uacpy.core.results._superposition import restack_source_depths
        trio = _pair(_complex_slab(30.0, 1.0), _complex_slab(60.0, 2.0),
                     _complex_slab(90.0, 3.0))
        trio.source_weights = np.array([1.0, 0.5, -0.25])
        kept = trio.window(source_depth=(50.0, 95.0))
        np.testing.assert_array_equal(kept.source_weights, [0.5, -0.25])
        np.testing.assert_allclose(kept.superpose().data,
                                   0.5 * trio.data[1] - 0.25 * trio.data[2])
        for plane in (trio.at(source_depth=60.0), trio.isel(source_depth=1),
                      trio.eval(source_depth=45.0), trio.max(),
                      *split_source_depths(trio)):
            assert plane.source_weights is None
        widened = trio.reindex(source_depth=[30.0, 45.0, 60.0, 90.0])
        assert widened.source_weights is None
        again = restack_source_depths(split_source_depths(trio), trio,
                                      who='t')
        np.testing.assert_array_equal(again.source_weights,
                                      trio.source_weights)
        array = trio.to_xarray()
        assert Field.from_xarray(
            array.sel(source_depth=60.0)).source_weights is None
        sliced = Field.from_xarray(array.isel(source_depth=slice(1, 3)))
        np.testing.assert_array_equal(sliced.source_depths, [60.0, 90.0])
        np.testing.assert_array_equal(sliced.source_weights, [0.5, -0.25])
        np.testing.assert_array_equal(
            Field.from_xarray(array).source_weights, trio.source_weights)

    def test_a_reordered_source_axis_reorders_its_identity_and_weights(self):
        """An xarray step that permutes the source_depth axis (sortby, isel
        of a permutation) keeps each depth's weight with it, so superpose()
        is the same sum."""
        trio = _pair(_complex_slab(30.0, 1.0), _complex_slab(60.0, 2.0),
                     _complex_slab(90.0, 3.0))
        trio.source_weights = np.array([1.0, 0.5 + 0.5j, -0.25])
        array = trio.to_xarray()
        for permuted in (array.sortby('source_depth', ascending=False),
                         array.isel(source_depth=[1, 0, 2])):
            back = Field.from_xarray(permuted)
            labels = back.coords['source_depth']
            np.testing.assert_array_equal(back.source_depths, labels)
            order = [list(trio.coords['source_depth']).index(z)
                     for z in labels]
            np.testing.assert_array_equal(back.source_weights,
                                          trio.source_weights[order])
            np.testing.assert_allclose(back.superpose().data,
                                       trio.superpose().data, rtol=1e-12)

    def test_the_stacked_record_is_the_whole_stacks_or_none(self):
        """One-depth records that differ only in their source depth join
        into the multi-depth record (the axis, depth_loop 'per_depth');
        records that differ in a setting describe no one run, so the
        stack carries none and keeps the speeds they stated; one shared
        record is kept."""
        from uacpy.core.run_settings import RunSettings, WaveguideSpeeds

        def record(z, **changes):
            return RunSettings(model='Test', mode=RunMode.COHERENT_TL,
                               frequencies=np.array([F0]),
                               source_depths=np.array([z]),
                               waveguide=WaveguideSpeeds(1500.0, 1700.0),
                               **changes)

        def slab(z, settings):
            return _complex_slab(z, z).replace(run_settings=settings)

        joined = _pair(slab(10.0, record(10.0)), slab(20.0, record(20.0)))
        np.testing.assert_array_equal(joined.run_settings.source_depths,
                                      [10.0, 20.0])
        assert joined.run_settings.depth_loop == 'per_depth'
        assert joined.run_settings == record(10.0)._replace(
            source_depths=np.array([10.0, 20.0]), depth_loop='per_depth')
        swept = _pair(slab(10.0, record(10.0)),
                      slab(20.0, record(20.0, source_level_dB=170.0)))
        assert swept.run_settings is None
        assert swept.speeds.waveguide_max == 1700.0
        knob = stack_fields([slab(10.0, record(10.0)),
                             slab(10.0, record(10.0, source_type='line'))],
                            [1.0, 2.0], coordinate_name='case')
        assert knob.run_settings is None
        shared = record(10.0)
        same = stack_fields([slab(10.0, shared), slab(10.0, shared)],
                            [1.0, 2.0], coordinate_name='case')
        assert same.run_settings is shared

    def test_a_component_over_the_sources_carries_the_runs_weights(self):
        """Each source's own Field component (an OASES mean field) joins
        into one over source_depth that carries the weights its run
        stamped, so the component's superpose() drives the same sources;
        split and joined again it keeps them."""
        from uacpy.core.results._superposition import restack_source_depths
        slabs = []
        for z, value in ((80.0, 1.0), (100.0, 2.0j)):
            slab = _complex_slab(z, value).replace(
                components={'mean_field': _complex_slab(z, 3.0 * value)})
            slab.source_weights = np.array([1.0, 0.5])
            slabs.append(slab)
        pair = stack_fields(slabs, [80.0, 100.0])
        mean = pair.components['mean_field']
        np.testing.assert_array_equal(mean.source_weights, [1.0, 0.5])
        np.testing.assert_allclose(mean.superpose().data,
                                   3.0 + 0.5 * 6.0j)
        again = restack_source_depths(split_source_depths(pair), pair,
                                      who='t')
        np.testing.assert_array_equal(
            again.components['mean_field'].source_weights, [1.0, 0.5])

    def test_a_selection_keeps_a_declared_absence_of_coherence(self):
        """A field stamped with no coherence answer keeps it on every
        selection of its cells; complex planes re-decided from their storage
        would read as coherent. A stated answer is kept too."""
        pair = _pair(_complex_slab(10.0, 1.0), _complex_slab(20.0, 2.0))
        assert pair.coherent is True
        assert pair.isel(source_depth=0).coherent is True
        pair._coherent = None
        assert pair.isel(source_depth=0).coherent is None
        assert pair.at(source_depth=20.0).coherent is None
        assert pair.window(range=(100.0, 200.0)).coherent is None
        assert [p.coherent for p in split_source_depths(pair)] == [None, None]

    def test_the_public_join_is_exported_and_names_itself(self):
        """stack_fields is reachable where the result types are (uacpy,
        uacpy.core, uacpy.core.results), takes no caller label, and its
        refusal names it; a run names itself through the labelled form."""
        import inspect
        import uacpy.core
        from uacpy.core.results._superposition import _stack_fields
        assert uacpy.stack_fields is uacpy.core.stack_fields is stack_fields
        assert list(inspect.signature(stack_fields).parameters) == [
            'fields', 'coordinate', 'coordinate_name']
        a = _complex_slab(10.0, 1.0)
        b = _complex_slab(20.0, 1.0).window(range=(100.0, 200.0))
        with pytest.raises(ConfigurationError,
                           match=r'^stack_fields: fields\[1\] is on a '
                                 r'different grid'):
            stack_fields([a, b], [10.0, 20.0])
        with pytest.raises(ConfigurationError,
                           match=r'^Kraken\.run: fields\[1\] is on a '
                                 r'different grid'):
            _stack_fields([a, b], [10.0, 20.0], who='Kraken.run')

    def test_a_label_must_be_its_fields_own_and_used_once(self):
        """A one-depth field is the field of its depth, so its label must
        be that depth (within a deck's rounding): a swapped or made-up list
        is refused, as is a label used twice; the same holds on a frequency
        axis; a field with no one value there takes any label."""
        a, b = _complex_slab(30.0, 1.0), _complex_slab(60.0, 2.0)
        with pytest.raises(ConfigurationError,
                           match=r"(?s)^stack_fields: fields\[0\] is the "
                                 r"field of source_depth 30 but is labelled "
                                 r"60\..*f\.source_depths\[0\] for f in "
                                 r"fields"):
            stack_fields([a, b], [60.0, 30.0])
        with pytest.raises(ConfigurationError,
                           match=r"(?s)labels \[30\.0, 30\.0\] repeat "
                                 r"\[30\.0\].*its own label"):
            stack_fields([a, b], [30.0, 30.0])
        joined = stack_fields([a, b], [30.0, 60.0 * (1 + 1e-7)])
        assert joined.data.shape == (2, 2, 3)
        with pytest.raises(ConfigurationError, match='is labelled 60.0006'):
            stack_fields([a, b], [30.0, 60.0 * (1 + 1e-5)])
        low = _complex_slab(30.0, 1.0)
        high = _complex_slab(30.0, 2.0).replace(frequencies=120.0)
        with pytest.raises(ConfigurationError,
                           match='field of frequency 100 but is labelled 1'):
            stack_fields([low, high], [1.0, 2.0], coordinate_name='frequency')
        assert stack_fields([low, high], [F0, 120.0],
                            coordinate_name='frequency').data.shape[0] == 2
        anonymous = [a.replace(source_depths=[]), b.replace(source_depths=[])]
        assert stack_fields(anonymous, [1.0, 2.0]).data.shape[0] == 2

    def test_labels_are_finite_flat_and_apart(self):
        """On the source_depth axis labels within a deck's rounding of each
        other — 1e-6 of the largest of the axis span, the label and 1, so
        6e-5 m at 60 m — are one position and are refused as a repeat, as
        the identity check reads them: 60 and 60 + 3e-5 are one, 60 and
        60 + 1.2e-4 two. A sweep's labels, in any unit, repeat only when
        equal (1e-7, 2e-7, 3e-7 are three). A 2-D list and a non-finite
        label are refused by name with a remedy."""
        a = _complex_slab(60.0, 1.0)
        with pytest.raises(ConfigurationError,
                           match=r"(?s)^stack_fields: the source_depth labels "
                                 r"\[60\.0, 60\.00003\] repeat "
                                 r"\[60\.0\].*its own label"):
            stack_fields([a, _complex_slab(60.0 + 3e-5, 2.0)],
                         [60.0, 60.0 + 3e-5])
        apart = stack_fields([a, _complex_slab(60.0 + 1.2e-4, 2.0)],
                             [60.0, 60.0 + 1.2e-4])
        assert apart.data.shape[0] == 2
        anonymous = [_complex_slab(60.0, v).replace(source_depths=[])
                     for v in (1.0, 2.0, 3.0)]
        sweep = stack_fields(anonymous, [1e-7, 2e-7, 3e-7],
                             coordinate_name='sweep')
        assert sweep.data.shape[0] == 3
        with pytest.raises(ConfigurationError, match=r"repeat \[2e-07\]"):
            stack_fields(anonymous, [1e-7, 2e-7, 2e-7],
                         coordinate_name='sweep')
        with pytest.raises(ConfigurationError,
                           match=r"(?s)^stack_fields: the source_depth labels "
                                 r"must be one 1-D list; got shape "
                                 r"\(1, 2\)\..*flat list"):
            stack_fields([a, _complex_slab(30.0, 2.0)], [[60.0, 30.0]])
        with pytest.raises(ConfigurationError,
                           match=r"(?s)^stack_fields: the source_depth labels "
                                 r"\[60\.0, nan\] hold a non-finite value.*"
                                 r"finite label"):
            stack_fields([a, _complex_slab(30.0, 2.0)], [60.0, np.nan])

    @pytest.mark.parametrize('labels', [[True, False], [60.0 + 0j, 30.0],
                                        ['60', '30'], [True, 100.0],
                                        [60.0, np.True_]])
    def test_labels_that_are_not_real_numbers_are_refused(self, labels):
        # Read as depths, bools would be 1 m and 0 m, complex values their
        # real parts.
        with pytest.raises(ConfigurationError,
                           match=r"(?s)^stack_fields: the source_depth labels "
                                 r"must be real numbers.*one number per "
                                 r"field"):
            stack_fields([_complex_slab(60.0, 1.0), _complex_slab(30.0, 2.0)],
                         labels)

    def test_every_joining_refusal_says_what_to_do(self):
        """The refusals of fields that do not join carry a remedy: a sweep
        over the waveguide's sound speeds is iterated, not stacked."""
        from uacpy.core.results import SoundSpeeds
        a = _complex_slab(30.0, 1.0).replace(speeds=SoundSpeeds(
            water_max=1500.0))
        b = _complex_slab(60.0, 1.0).replace(speeds=SoundSpeeds(
            water_max=1520.0))
        with pytest.raises(ConfigurationError,
                           match=r"(?s)fields\[0\]\.speeds=.*How to fix:.*"
                                 r"ParallelResult\.results"):
            stack_fields([a, b], [30.0, 60.0])
        c = _complex_slab(60.0, 1.0).replace(pinned={'time': 1.0})
        with pytest.raises(ConfigurationError,
                           match=r"(?s)is pinned at.*How to fix:.*field\.at"):
            stack_fields([_complex_slab(30.0, 1.0), c], [30.0, 60.0])

    def test_fields_on_different_grids_are_refused(self):
        a = _complex_slab(10.0, 1.0)
        b = Field(data=np.ones((2, 3), dtype=complex),
                  coords={'depth': [10.0, 20.0],
                          'range': [100.0, 200.0, 400.0]},
                  model='Test', frequencies=F0, source_depths=[20.0],
                  phase_reference='travelling_wave')
        with pytest.raises(ConfigurationError, match='different grid'):
            _pair(a, b)

    def test_fields_holding_different_quantities_are_refused(self):
        a = _complex_slab(10.0, 1.0)
        with pytest.raises(ConfigurationError,
                           match="storage 'complex' vs 'real'"):
            _pair(a, _complex_slab(20.0, 1.0).to_dB())

    def test_fields_of_different_models_are_refused(self):
        a = _complex_slab(10.0, 1.0)
        b = _complex_slab(20.0, 1.0)
        b.model = 'Other'
        with pytest.raises(ConfigurationError, match=r"fields\[0\]\.model"):
            _pair(a, b)

    def test_only_fields_join(self):
        from uacpy.core.results import Rays
        rays = [Rays(rays=[], model='Test', frequencies=F0, source_depths=[z])
                for z in (10.0, 20.0)]
        with pytest.raises(ConfigurationError, match='only Fields stack'):
            stack_fields(rays, [10.0, 20.0])


@pytest.mark.requires_binary
def test_a_bellhop_time_series_whose_depths_open_apart_runs_every_depth_at_one_start():
    """Bellhop opens each depth's record just before that depth's own
    earliest arrival; on two ranges the two starts differ by less than a
    sample, so the depths share no time axis. The loop runs every depth
    again at the earlier start: each plane is the one-depth run at that
    ``t_start`` and the settings say so. A caller's own ``t_start`` is
    honoured with no second run."""
    env = make_pekeris(name='ts-start', bathymetry=GUIDE_DEPTH)
    receiver = uacpy.Receiver(depths=[30.0, 50.0], ranges=[1000.0, 3000.0])
    fs = 2000.0
    t = np.arange(100) / fs
    waveform = np.hanning(t.size) * np.sin(2 * np.pi * 200.0 * t)
    model = _bellhop()
    starts = [model.run(env, uacpy.Source(depths=z, frequencies=200.0),
                        receiver, run_mode=RunMode.TIME_SERIES,
                        source_waveform=waveform,
                        sample_rate=fs).coords['time'][0]
              for z in (20.0, 70.0)]
    assert starts[0] != starts[1], "the depths no longer open apart"
    field = model.run(env, uacpy.Source(depths=[20.0, 70.0],
                                        frequencies=200.0),
                      receiver, run_mode=RunMode.TIME_SERIES,
                      source_waveform=waveform, sample_rate=fs)
    t0 = min(starts)
    assert field.coords['time'][0] == t0
    assert any('t_start' in note for note in field.run_settings.notes)
    for plane, z in zip(field.data, (20.0, 70.0)):
        single = model.run(env, uacpy.Source(depths=z, frequencies=200.0),
                           receiver, run_mode=RunMode.TIME_SERIES,
                           source_waveform=waveform, sample_rate=fs,
                           t_start=t0)
        n = single.coords['time'].size
        np.testing.assert_array_equal(plane[..., :n], single.data)
    pinned = model.run(env, uacpy.Source(depths=[20.0, 70.0],
                                         frequencies=200.0),
                       receiver, run_mode=RunMode.TIME_SERIES,
                       source_waveform=waveform, sample_rate=fs,
                       t_start=0.5)
    assert pinned.coords['time'][0] == 0.5
    assert not any('t_start' in note for note in pinned.run_settings.notes)


@pytest.mark.requires_binary
def test_a_multi_depth_source_in_modes_solves_the_modes_once():
    """The modes do not depend on the source depth, so ``MODES`` on a
    two-depth ``Source`` is one solve and one mode set (not a stack),
    tabulated at both depths — what ``compute_modes`` returns too."""
    model = _kraken()
    source = uacpy.Source(depths=[30.0, 70.0], frequencies=F0)
    model.validate_inputs(_env(), source, _receiver(),
                          run_mode=RunMode.MODES)
    modes = model.run(_env(), source, _receiver(), run_mode=RunMode.MODES)
    assert isinstance(modes, uacpy.Modes)
    assert np.any(np.isclose(modes.depths, 70.0))


@pytest.mark.requires_binary
def test_the_zero_range_refusal_fires_before_any_per_depth_run():
    """The loop runs each depth through the wrapper's own validation, so a
    receiver with no positive range is refused on the first depth."""
    model = _kraken()
    source = uacpy.Source(depths=[30.0, 70.0], frequencies=F0)
    with pytest.raises(ConfigurationError, match='largest range is 0 m'):
        model.run(_env(), source,
                  uacpy.Receiver(depths=[50.0], ranges=[0.0]))
