"""The contract every registered engine is held to, with no per-engine code.

Every test here is parametrised over ``uacpy.models._registry.ENGINES``, so
an engine added to the registry is checked by the whole suite on its first
run. What an engine must satisfy:

* registry — its entry imports a concrete ``PropagationModel`` with a
  ``spec`` and a ``provenance_id``; every concrete engine in the registered modules
  is registered; the public name lists match the registry;
* install — the executable search starts in the package's own ``bin``
  directory, and an installed engine runs the binary found there;
* construction — every constructor knob is keyword-only, and every
  run checks the knobs again (``_check_knobs``, in stage 2);
* run mode — ``run_settings`` resolves ``run_mode=None`` to the model's
  default and every declared mode to itself;
* one checking stage — ``validate_inputs``, ``run_settings`` and ``run``
  answer the same calls alike (the same exception, or all accept), and a
  refused ``run`` launches nothing;
* settings — ``run_settings`` writes no file, is repeatable, pickles and
  round-trips ``to_dict``; a record saved to disk loads from the class path
  it names;
* stage hooks — every engine runs on the stage hooks of ``PropagationModel``:
  it declares what every mode returns and resolves its own frozen settings
  and the speed bounds of the environment it runs on;
* what a run returns — every mode returns what it declared, stamped with
  the engine and the source; a two-depth Source stacks the single-depth
  runs and a weight scales one; a field keeps the requested grid and masks
  the cells no engine can fill; a run cleans up after itself, a failed
  launch releases its directory, a timeout kills the launch and says so, a
  missing binary names the install step and an unwritable ``work_dir`` is
  refused;
* copies and parallel runs — ``copy`` carries every knob and refuses an
  unknown one; ``run_parallel`` returns what a direct run returns.

Engine physics (benchmarks, cross-engine agreement) stays in each engine's
own test file.
"""

import dataclasses
import importlib.resources
import inspect
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

import uacpy
from uacpy.core.exceptions import (ExecutableNotFoundError,
                                   ModelExecutionError)
from uacpy.models._registry import ENGINES
from uacpy.core.run_settings import EngineSettings, RunMode, RunSettings
from uacpy.models.base import _STAGE_HOOKS, PropagationModel
from uacpy.tests.conftest import LaunchReached

_KEYS = sorted(ENGINES)


def _engine(key, **kw):
    """An instance of the registered engine built with its registry
    ``example_kwargs``, or a skip when its binary is not installed."""
    entry = ENGINES[key]
    cls = entry.load()
    try:
        return cls(**dict(entry.example_kwargs), **kw)
    except ExecutableNotFoundError:
        pytest.skip(f"{cls.__name__} binary not available")


def _carriers(key):
    """A 100 m guide the registered engine ``key`` runs: over the Pekeris
    half-space, or over the seabed its registry entry names
    (``EngineEntry.example_bottom``)."""
    bottom = dict(ENGINES[key].example_bottom) or dict(
        acoustic_type='half-space', sound_speed=1700.0, density=1.8,
        attenuation=0.5)
    env = uacpy.Environment(
        bathymetry=100.0, ssp=1500.0,
        bottom=uacpy.BoundaryProperties(**bottom))
    src = uacpy.Source(depths=50.0, frequencies=100.0)
    rcv = uacpy.Receiver(depths=[20.0, 50.0], ranges=[200.0, 400.0])
    return env, src, rcv


class _Declared:
    """What the registered engine ``key`` declares on its class — its
    modes, traits, field modes and default mode — read without constructing
    it, so a parameter list is filtered at collection with no binary
    installed. It answers through the engine's own methods, which read only
    ``_supported_modes`` and ``_traits``;
    ``test_the_declared_engine_answers_what_the_built_engine_answers`` holds
    the two alike."""

    def __init__(self, key):
        self._cls = ENGINES[key].load()
        self.spec = self._cls.spec
        self._traits = self.spec.traits
        self._supported_modes = list(self.spec.modes)
        self.supported_modes = self._supported_modes
        self._FIELD_MODES = self._cls._FIELD_MODES

    def _default_run_mode(self):
        return self._cls._default_run_mode(self)

    def _default_run_mode_for(self, frequencies):
        return self._cls._default_run_mode_for(self, frequencies)

    def _run_keywords_never_consumed(self):
        return self._cls._run_keywords_never_consumed(self)


def _default_mode_runs_a_field(model):
    """Whether ``run(run_mode=None)`` on ``model`` runs a field mode."""
    return model._default_run_mode_for(None) in model._FIELD_MODES


#: The engines with a TIME_SERIES mode.
_TIME_SERIES_KEYS = [key for key in _KEYS
                     if RunMode.TIME_SERIES in _Declared(key).supported_modes]

#: The engines whose default mode runs a field.
_FIELD_DEFAULT_KEYS = [key for key in _KEYS
                       if _default_mode_runs_a_field(_Declared(key))]


def _declares_a_field_default(key):
    """Whether the default mode of engine ``key`` declares a ``Field``."""
    declared = _Declared(key)
    mode = declared._default_run_mode_for(None)
    return declared._cls.outputs[mode].result_type == 'Field'


#: The engines whose default mode returns a ``Field``: a depth-range field
#: on :func:`_carriers`.
_FIELD_RESULT_KEYS = [key for key in _KEYS if _declares_a_field_default(key)]


# ── registry ────────────────────────────────────────────────────────────


@pytest.mark.parametrize('key', _KEYS)
def test_a_registered_engine_is_a_concrete_model_with_spec_and_source(key):
    cls = ENGINES[key].load()
    assert inspect.isclass(cls) and issubclass(cls, PropagationModel)
    assert not inspect.isabstract(cls)
    assert cls.spec is not None and cls.spec.modes
    assert cls.provenance_id is not None
    assert cls.__name__.lower() == key


def test_every_concrete_engine_in_the_registered_modules_is_registered():
    """A new engine class a registered module exposes (defined there, or
    in a module of the registered engine package that re-exports it)
    without its registry line would silently miss this whole suite."""
    import importlib
    registered = {e.class_name for e in ENGINES.values()}
    found = set()
    for module in {e.module for e in ENGINES.values()}:
        mod = importlib.import_module(module)
        for name, obj in vars(mod).items():
            if (inspect.isclass(obj) and issubclass(obj, PropagationModel)
                    and (obj.__module__ == module
                         or obj.__module__.startswith(module + '.'))
                    and not inspect.isabstract(obj)):
                found.add(name)
    assert found == registered


def test_the_public_name_lists_match_the_registry():
    import uacpy.models as models
    registered = {e.class_name for e in ENGINES.values()}
    engine_names = {n for n in models.__all__
                    if inspect.isclass(getattr(models, n))
                    and issubclass(getattr(models, n), PropagationModel)
                    and not inspect.isabstract(getattr(models, n))}
    assert engine_names == registered
    lazy = {n for n, (mod, _) in uacpy._LAZY_ATTRS.items()
            if mod == 'uacpy.models'}
    assert registered <= lazy


# ── install ─────────────────────────────────────────────────────────────


def _package_bin_dir() -> Path:
    """``uacpy/bin``, found through ``importlib.resources`` and so from the
    package itself, never from where a module of it sits."""
    return Path(os.path.realpath(importlib.resources.files('uacpy'))) / 'bin'


def test_the_executable_search_starts_in_the_package_bin_dir():
    """The first place an engine looks for its binary is ``uacpy/bin/<sub>``.
    The search roots itself on a path computed from a module's own
    location, which a module moved one level deeper would shift."""
    cls = ENGINES['kraken'].load()
    model = cls.__new__(cls)        # only the search is exercised
    model.model_name = cls.__name__
    with pytest.raises(ExecutableNotFoundError,
                       match='executable not found: uacpy-no-such-binary') as err:
        model._find_executable_in_paths(['uacpy-no-such-binary'],
                                        bin_subdirs=['oalib'])
    first = Path(err.value.search_paths[0])
    assert Path(os.path.realpath(first.parent)) == \
        _package_bin_dir() / 'oalib'


@pytest.mark.parametrize('key', _KEYS)
def test_an_installed_engine_runs_the_binary_in_the_package_bin_dir(key):
    """An engine whose binary ``install.sh`` put in ``uacpy/bin`` runs that
    one, not a same-named file further down the search (the vendored
    sources, ``PATH``)."""
    exe = Path(_engine(key)._exe)
    if not any(_package_bin_dir().rglob(exe.name)):
        pytest.skip(f"{exe.name} is not installed in uacpy/bin")
    assert Path(os.path.realpath(exe.parent)).is_relative_to(
        _package_bin_dir()), exe


# ── construction ────────────────────────────────────────────────────────


@pytest.mark.parametrize('key', _KEYS)
def test_every_constructor_knob_is_keyword_only(key):
    """B5 / RA-CONTRACT-9: a positional constructor call cannot bind a
    value to the wrong knob (``Kraken('/opt/kraken.exe')`` used to set
    ``mode_points_per_meter``); it raises ``TypeError``."""
    cls = ENGINES[key].load()
    for klass in cls.__mro__:
        init = klass.__dict__.get('__init__')
        if init is None or klass is object:
            continue
        params = list(inspect.signature(init).parameters.values())[1:]
        positional = [p.name for p in params
                      if p.kind in (p.POSITIONAL_ONLY,
                                    p.POSITIONAL_OR_KEYWORD)]
        assert positional == [], f"{klass.__name__}: {positional}"


@pytest.mark.parametrize('key', _KEYS)
def test_the_class_signature_lists_the_constructor_knobs(key):
    """``inspect.signature`` (and so ``help`` and IDE completion) shows
    the constructor's parameters, not the construction metaclass's
    ``__call__(*args, **kwargs)``."""
    cls = ENGINES[key].load()
    init = list(inspect.signature(cls.__init__).parameters.values())[1:]
    shown = list(inspect.signature(cls).parameters.values())
    assert shown == init
    assert not any(p.kind is p.VAR_POSITIONAL for p in shown)


@pytest.mark.parametrize('cls_name, knob', [('Kraken', 'c_high'),
                                            ('OASS', 'correlation_length')])
def test_a_known_knob_is_in_the_class_signature(cls_name, knob):
    """The class signature carries the knob by name."""
    cls = getattr(uacpy.models, cls_name)
    assert knob in inspect.signature(cls).parameters


# ── run mode ────────────────────────────────────────────────────────────


@pytest.mark.parametrize('key', _KEYS)
def test_run_settings_resolves_every_declared_mode_to_itself(key):
    model = _engine(key)
    env, src, rcv = _carriers(key)
    default = model.run_settings(env, src, rcv)
    assert default.mode is model._default_run_mode_for(None)
    for mode in model.supported_modes:
        kw = {}
        if mode == RunMode.TIME_SERIES:
            kw = dict(source_waveform=np.hanning(40), sample_rate=400.0)
        assert model.run_settings(env, src, rcv, mode, **kw).mode is mode


# ── one checking stage ──────────────────────────────────────────────────


_ILLEGAL_CALLS = (
    'source below the domain',
    'env is not an Environment',
    'non-finite t_start',
    'unsupported mode',
    'frequencies= on a single-frequency mode',
    'multi-frequency Source on a single-frequency mode',
    'a keyword no mode reads',
)

def _illegal_call(key, model, label):
    """``(run_mode, kwargs, carriers)`` for the call ``label`` names, or
    ``None`` when the call is legal on this model."""
    env, src, rcv = _carriers(key)
    if label == 'source below the domain':
        return None, {}, (env, uacpy.Source(depths=5000.0,
                                            frequencies=100.0), rcv)
    if label == 'env is not an Environment':
        return None, {}, (None, src, rcv)
    if label == 'non-finite t_start':
        return None, dict(t_start=float('nan')), (env, src, rcv)
    if label == 'unsupported mode':
        unsupported = [m for m in RunMode if m not in model.supported_modes]
        return (unsupported[0], {}, (env, src, rcv)) if unsupported else None
    single = [m for m in model.supported_modes
              if m in model.spec.traits.single_frequency_modes]
    if label == 'frequencies= on a single-frequency mode':
        if not single or model.spec.traits.consumes_single_mode_frequencies:
            return None
        return single[0], dict(frequencies=[90.0, 110.0]), (env, src, rcv)
    if label == 'multi-frequency Source on a single-frequency mode':
        if not single:
            return None
        return single[0], {}, (env, uacpy.Source(
            depths=50.0, frequencies=[90.0, 110.0]), rcv)
    if label == 'a keyword no mode reads':
        if 'sample_rate' not in model._run_keywords_never_consumed():
            return None
        return None, dict(sample_rate=1000.0), (env, src, rcv)
    raise KeyError(label)


def _outcome(call):
    """``(exception type, message)`` of ``call()``, or ``'accepted'``."""
    try:
        call()
    except LaunchReached:
        return 'accepted'
    except Exception as exc:                 # noqa: BLE001
        return type(exc), str(exc)
    return 'accepted'


#: The (engine, call) pairs where the call is illegal: a call legal on an
#: engine is not collected for it.
_ILLEGAL_CASES = [pytest.param(key, label, id=f'{key}-{label}')
                  for label in _ILLEGAL_CALLS for key in _KEYS
                  if _illegal_call(key, _Declared(key), label) is not None]


@pytest.mark.parametrize('key,label', _ILLEGAL_CASES)
def test_the_three_entry_points_answer_a_call_alike(key, label, launch_spy):
    """ARCH-9 / RA-CONTRACT-18: ``validate_inputs`` is the checking stage of
    ``run``, and ``run_settings`` runs it too, so for each of these calls
    the three raise the same exception with the same message — or all
    three accept it — and a refused ``run`` launches no binary."""
    model = _engine(key)
    mode, kw, carriers = _illegal_call(key, model, label)
    launch_spy(model)
    outcomes = [_outcome(lambda c=call: getattr(model, c)(*carriers, mode,
                                                          **kw))
                for call in ('validate_inputs', 'run_settings', 'run')]
    assert len(set(outcomes)) == 1, outcomes


# ── settings ────────────────────────────────────────────────────────────


@pytest.mark.parametrize('key', _KEYS)
def test_run_settings_is_pure_repeatable_and_round_trips(key, tmp_path):
    model = _engine(key, work_dir=tmp_path / 'wd')
    env, src, rcv = _carriers(key)
    a = model.run_settings(env, src, rcv)
    b = model.run_settings(env, src, rcv)
    assert a == b
    assert not (tmp_path / 'wd').exists(), 'run_settings wrote a file'
    assert pickle.loads(pickle.dumps(a)) == a
    assert RunSettings.from_dict(a.to_dict()) == a
    assert a.model == type(model).__name__
    assert '\n' not in a.summary()


#: Every engine's default-mode ``run_settings(...).to_dict()`` on this
#: suite's example guide, as JSON: the record a result saved to disk
#: carries. Each names its engine settings' class by the module path it was
#: written with (``engine.__class__``), ``uacpy.models.<engine module>``.
#: ``python -m uacpy.tests._save_run_settings KEY`` adds a new engine's
#: record; a saved record is never rewritten, since it stands for the files
#: saved before.
_SAVED_SETTINGS = Path(__file__).parent / 'data' / 'run_settings_by_engine.json'


def _plain_default(field):
    """The plain-type value ``to_dict`` writes for ``field`` left at its
    default."""
    from uacpy.core._records import _plain
    if field.default is not dataclasses.MISSING:
        value = field.default
    else:
        value = field.default_factory()
    return json.loads(json.dumps(_plain(value)))


@pytest.mark.parametrize('key', _KEYS)
def test_a_saved_run_settings_loads_from_the_class_path_it_names(key):
    """A saved record keeps loading when an engine's settings class moves:
    its saved module path still reaches the class, and the record it
    rebuilds writes back what was saved. A settings field renamed or
    removed fails here too, as it fails for every file saved before.

    A field added after the record was saved is absent from it: the record
    loads with that field at its default, which therefore has to reproduce
    what the run computed before the field existed. A new field with no
    default fails here, as it would fail every file saved before."""
    saved_by_engine = json.loads(_SAVED_SETTINGS.read_text(encoding='utf-8'))
    if key not in saved_by_engine:
        pytest.fail(f"no saved run-settings record for {key!r}: run "
                    f"python -m uacpy.tests._save_run_settings {key} "
                    f"(docs/DEV.md section 3, step 5)")
    saved = saved_by_engine[key]
    try:
        settings = RunSettings.from_dict(saved)
    except TypeError as exc:
        pytest.fail(f"a record saved before a settings field existed no "
                    f"longer loads ({exc}); give the new field a default "
                    f"(docs/DEV.md section 13.1, step 3)")
    assert type(settings.engine).__name__ == \
        saved['engine']['__class__'].rpartition('.')[2]
    written = json.loads(json.dumps(settings.to_dict()))
    written['engine'].pop('__class__')
    expected = dict(saved, engine=dict(saved['engine']))
    expected['engine'].pop('__class__')
    for part, record in ((written, settings), (written['engine'],
                                               settings.engine)):
        saved_part = expected if part is written else expected['engine']
        fields = {f.name: f for f in dataclasses.fields(record)}
        for name in sorted(set(part) - set(saved_part)):
            assert part.pop(name) == _plain_default(fields[name]), (
                f"{type(record).__name__}.{name}: a record saved without "
                f"the field must load with its default")
    assert written == expected


@pytest.mark.parametrize('key', _KEYS)
def test_every_engine_re_checks_its_knobs_in_stage_2(key, monkeypatch):
    """Every run calls the engine's own ``_check_knobs`` again from the base
    (the attributes can be reassigned after construction), right before the
    engine's ``_validate_engine`` refusals (M-16)."""
    model = _engine(key)
    cls = type(model)
    assert cls._check_knobs is not PropagationModel._check_knobs
    calls = []
    monkeypatch.setattr(cls, '_check_knobs',
                        lambda self: calls.append('knobs'))
    monkeypatch.setattr(cls, '_validate_engine',
                        lambda self, *args, **kwargs: calls.append('engine'))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.validate_inputs(*_carriers(key))
    assert calls == ['knobs', 'engine']


@pytest.mark.parametrize('key', _KEYS)
def test_a_saved_run_settings_names_the_engine_package(key):
    """A record saved today names the engine settings' class by the module
    the registry imports the engine from, its one public home, and not by
    a module inside an engine package that defines the class."""
    settings = _engine(key).run_settings(*_carriers(key))
    assert settings.to_dict()['engine']['__class__'] == (
        f"{ENGINES[key].module}.{type(settings.engine).__qualname__}")
    # A registry entry naming a module inside a package would make that
    # private module the saved-file contract.
    assert not any(part.startswith('_')
                   for part in ENGINES[key].module.split('.'))


def test_a_saved_class_path_is_the_package_that_re_exports_the_class(
        tmp_path, monkeypatch):
    """A settings class defined inside a package is saved under the package
    that re-exports it, and loads back from there; a class the package does
    not re-export is saved under the module that defines it."""
    package = tmp_path / 'saved_path_engine'
    package.mkdir()
    (package / '_settings.py').write_text(
        "from dataclasses import dataclass\n"
        "from uacpy.core.run_settings import EngineSettings\n"
        "\n"
        "@dataclass(frozen=True, eq=False)\n"
        "class ExportedSettings(EngineSettings):\n"
        "    n: int = 1\n"
        "\n"
        "@dataclass(frozen=True, eq=False)\n"
        "class InternalSettings(EngineSettings):\n"
        "    n: int = 1\n")
    (package / '__init__.py').write_text(
        "from saved_path_engine._settings import ExportedSettings\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        defined = importlib.import_module('saved_path_engine._settings')

        def saved(engine):
            settings = RunSettings(model='X', mode=RunMode.COHERENT_TL,
                                   frequencies=np.array([100.0]),
                                   source_depths=np.array([10.0]),
                                   engine=engine)
            d = settings.to_dict()
            assert RunSettings.from_dict(d) == settings
            return d['engine']['__class__']

        assert saved(defined.ExportedSettings(n=2)) == \
            'saved_path_engine.ExportedSettings'
        assert saved(defined.InternalSettings(n=2)) == \
            'saved_path_engine._settings.InternalSettings'
    finally:
        for name in ('saved_path_engine', 'saved_path_engine._settings'):
            sys.modules.pop(name, None)


# ── the stamp on results ────────────────────────────────────────────────


@pytest.mark.parametrize('key', ['scooter', 'kraken'])
def test_a_result_its_slices_and_its_reload_carry_the_run_settings(key):
    """Every result carries the settings ``run`` resolved: the result, a
    slice of it and the result rebuilt from ``to_dict`` all answer
    ``run_settings`` equal to what ``run_settings()`` returned for the same
    call, and the attribute is read-only."""
    from uacpy.core.results import Field
    model = _engine(key)
    env, src, rcv = _carriers(key)
    want = model.run_settings(env, src, rcv)
    result = model.run(env, src, rcv)
    assert result.run_settings == want
    assert result.at(depth=20.0).run_settings == want
    assert Field.from_dict(result.to_dict()).run_settings == want
    with pytest.raises(AttributeError, match='has no setter'):
        result.run_settings = None


@pytest.mark.parametrize('key', _TIME_SERIES_KEYS)
def test_a_time_series_and_every_snapshot_of_it_answer_coherence_alike(
        key):
    """A TIME_SERIES trace takes the ``coherent`` its engine declares
    (``None``: the question does not arise for p(t)), and a selection keeps
    it; re-decided from the axes left, an instant read as coherent
    frequency-domain pressure (ARCH-4)."""
    model = _engine(key)
    env, src, rcv = _carriers(key)
    n = np.arange(64)
    pulse = np.hanning(64) * np.sin(2 * np.pi * 100.0 * n / 800.0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        trace = model.run(env, src, rcv, RunMode.TIME_SERIES,
                          source_waveform=pulse, sample_rate=800.0,
                          output_duration=1.0)
    declared = type(model).outputs[RunMode.TIME_SERIES].coherent
    assert declared is None
    t = float(trace.coords['time'][trace.coords['time'].size // 2])
    for view in (trace, trace.at(time=t), trace.isel(time=0),
                 trace.eval(time=t), trace.max()):
        assert view.coherent is declared


# ── stage hooks ─────────────────────────────────────────────────────────


def _staged(key):
    """The registered engine class, which runs on the stage hooks."""
    cls = ENGINES[key].load()
    assert not inspect.isabstract(cls), cls
    assert all(hook in vars(cls) for hook in _STAGE_HOOKS), cls
    return cls


@pytest.mark.parametrize('key', _KEYS)
def test_a_staged_engine_declares_what_every_mode_returns(key):
    import uacpy.core.results as results
    cls = _staged(key)
    assert set(cls.outputs) == set(cls.spec.modes)
    for spec in cls.outputs.values():
        assert isinstance(getattr(results, spec.result_type), type)


@pytest.mark.parametrize('key', _KEYS)
def test_a_staged_engine_resolves_its_settings_before_launching(key):
    """``run_settings`` holds the engine's own frozen settings, the speed
    bounds of the projected environment and the declared output of the
    mode."""
    _staged(key)
    model = _engine(key)
    env, src, rcv = _carriers(key)
    for mode in model.supported_modes:
        kw = {}
        if mode == RunMode.TIME_SERIES:
            kw = dict(source_waveform=np.hanning(40), sample_rate=400.0)
        settings = model.run_settings(env, src, rcv, mode, **kw)
        assert isinstance(settings.engine, EngineSettings)
        assert settings.output == type(model).outputs[mode]
        assert (settings.waveguide.c_min, settings.waveguide.c_max) == \
            model._speed_bounds(model._project_environment(env))


@pytest.mark.parametrize('key', _KEYS)
def test_every_mode_resolves_through_the_staged_stage_3(key, stage_spy):
    """Stage 3 of every declared mode runs :meth:`PropagationModel.
    _resolve_settings`, so a rule added there reaches every engine; an
    engine that runs its band otherwise declares it in
    ``spec.traits.announced_band_modes`` instead of skipping the method."""
    model = _engine(key)
    env, src, rcv = _carriers(key)
    seen = stage_spy(PropagationModel, 'settings')
    for mode in model.supported_modes:
        kw = {}
        if mode == RunMode.TIME_SERIES:
            kw = dict(source_waveform=np.hanning(40), sample_rate=400.0)
        before = len(seen)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.run_settings(env, src, rcv, mode, **kw)
        assert len(seen) > before, mode


# ── what a run returns ──────────────────────────────────────────────────
#
# One tiny run per engine and declared mode, kept for the rest of the
# worker's session: every check below reads it, and none changes it.

#: Slab tolerance of a re-run of the same deck. Bellhop's ``.shd`` is
#: complex64 and its parallel beam sum lands in a different order per run,
#: so its re-runs agree to float32 round-off only; every other engine writes
#: the same deck and reads the same numbers back.
_RERUN_TOLERANCE = {'bellhop': 1e-5}

_RUNS = {}


def _mode_kwargs(mode):
    """The run keywords a mode needs: TIME_SERIES takes a pulse."""
    if mode != RunMode.TIME_SERIES:
        return {}
    n = np.arange(64)
    return dict(source_waveform=np.hanning(64)
                * np.sin(2 * np.pi * 100.0 * n / 800.0),
                sample_rate=800.0, output_duration=1.0)


def _run(key, mode=None):
    """The registered engine's run of ``mode`` (its default when ``None``)
    on :func:`_carriers`, with the keywords the mode needs."""
    model = _engine(key)
    mode = mode or model._default_run_mode_for(None)
    if (key, mode) not in _RUNS:
        env, src, rcv = _carriers(key)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _RUNS[key, mode] = model.run(env, src, rcv, mode,
                                         **_mode_kwargs(mode))
    return _RUNS[key, mode]


def _slabs(result):
    """The results a run returned: the slabs of a stack, or the result."""
    from uacpy.core.results import ResultStack
    return list(result) if isinstance(result, ResultStack) else [result]


def _relative_error(actual, reference):
    """Largest ``|actual - reference| / |reference|`` over the cells where
    the reference is finite and non-zero."""
    ref = np.asarray(reference)
    ok = np.isfinite(ref) & (ref != 0)
    assert ok.any(), 'the reference is empty'
    return float(np.max(np.abs(np.asarray(actual)[ok] - ref[ok])
                        / np.abs(ref[ok])))


#: Checks an engine is known to break, each reported as a finding: the case
#: fails strictly, so it turns red again the day the engine is fixed.
_KNOWN_BREAKS = {}


def _known_break(check, case_id):
    reason = _KNOWN_BREAKS.get((check, case_id))
    return [pytest.mark.xfail(strict=True, reason=reason)] if reason else []


_MODE_CASES = [pytest.param(key, mode, id=f'{key}-{mode.name}')
               for key in _KEYS for mode in ENGINES[key].load().spec.modes]


def _identity(result):
    return tuple(getattr(result, name, None)
                 for name in ('kind', 'unit', 'phase_reference', 'coherent'))


@pytest.mark.parametrize('key,mode', _MODE_CASES)
def test_every_mode_returns_its_declared_output(key, mode):
    """The result class, and a Field's ``kind``, ``unit``,
    ``phase_reference`` (a :class:`PhaseReference` member, never a bare
    string) and ``coherent``, are those ``outputs[mode]`` declares; complex
    pressure is complex data and a dB field is real; a selection and a
    ``to_dict`` round trip keep them."""
    from uacpy.core.results import Field, PhaseReference
    declared = type(_engine(key)).outputs[mode]
    for slab in _slabs(_run(key, mode)):
        assert type(slab).__name__ == declared.result_type
        if not isinstance(slab, Field):
            continue
        for name in ('kind', 'unit', 'coherent'):
            if getattr(declared, name) is not None:
                assert getattr(slab, name) == getattr(declared, name), name
        if declared.phase_reference is not None:
            assert isinstance(slab.phase_reference, PhaseReference)
            assert slab.phase_reference.value == declared.phase_reference
        if declared.phase_reference == PhaseReference.TRAVELLING_WAVE.value:
            assert slab.is_complex
        if slab.unit == 'dB':
            assert not slab.is_complex
        first = next(iter(slab.coords))
        for view in (slab.isel(**{first: 0}),
                     Field.from_dict(slab.to_dict())):
            assert _identity(view) == _identity(slab), view


@pytest.mark.parametrize('key,mode', [
    pytest.param(key, mode, id=f'{key}-{mode.name}',
                 marks=_known_break('stamped_settings', f'{key}-{mode.name}'))
    for key in _KEYS for mode in ENGINES[key].load().spec.modes])
def test_every_mode_stamps_the_settings_run_settings_resolves(key, mode):
    model = _engine(key)
    env, src, rcv = _carriers(key)
    assert _run(key, mode).run_settings == model.run_settings(
        env, src, rcv, mode, **_mode_kwargs(mode))


@pytest.mark.parametrize('key', _KEYS)
def test_a_result_names_the_engine_and_the_source_it_ran(key):
    from uacpy.core.results import Field
    for slab in _slabs(_run(key)):
        assert slab.model == ENGINES[key].class_name
        assert isinstance(slab.backend, str) and slab.backend
        if isinstance(slab, Field):
            np.testing.assert_array_equal(slab.source_depths, [50.0])


@pytest.mark.parametrize('key', _FIELD_DEFAULT_KEYS)
def test_a_two_depth_source_stacks_the_single_depth_runs(key):
    """In a field mode a two-depth Source returns a stack over
    ``source_depth`` whose every slab is the single-depth run at that depth,
    carrying the Source's weights; a coherent stack superposes to their
    weighted sum."""
    from uacpy.core.results import ResultStack
    model = _engine(key)
    mode = model._default_run_mode_for(None)
    env, _, rcv = _carriers(key)
    depths, weights = [30.0, 70.0], [1.0, -1.0]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        stack = model.run(env, uacpy.Source(depths=depths, frequencies=100.0,
                                            weights=weights), rcv, mode,
                          **_mode_kwargs(mode))
        singles = [model.run(env, uacpy.Source(depths=z, frequencies=100.0),
                             rcv, mode, **_mode_kwargs(mode))
                   for z in depths]
    assert isinstance(stack, ResultStack)
    assert stack.coordinate_name == 'source_depth'
    np.testing.assert_array_equal(stack.coordinate, depths)
    np.testing.assert_array_equal(stack.source_weights, weights)
    tolerance = _RERUN_TOLERANCE.get(key, 1e-9)
    for i, single in enumerate(singles):
        slab = stack[i]
        assert slab.data.shape == single.data.shape
        assert _relative_error(slab.data, single.data) < tolerance
    if stack[0].unit != 'dB':
        assert _relative_error(stack.superpose().data,
                               singles[0].data - singles[1].data) < tolerance


@pytest.mark.parametrize('key', _FIELD_DEFAULT_KEYS)
def test_a_single_depth_weight_scales_the_unit_run(key):
    """``Source(weights=w)`` on one depth returns ``w`` times the unit
    field and records the depth and weight; a dB field has lost the phase a
    weight multiplies, so it is refused."""
    model = _engine(key)
    mode = model._default_run_mode_for(None)
    unit = _run(key)
    env, _, rcv = _carriers(key)
    w = -0.5 + 0.25j if unit.is_complex else -2.0
    weighted = uacpy.Source(depths=50.0, frequencies=100.0, weights=w)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if unit.unit == 'dB':
            with pytest.raises(uacpy.ConfigurationError, match='dB'):
                model.run(env, weighted, rcv, mode, **_mode_kwargs(mode))
            return
        scaled = model.run(env, weighted, rcv, mode, **_mode_kwargs(mode))
    assert _relative_error(scaled.data, w * unit.data) < \
        _RERUN_TOLERANCE.get(key, 1e-9)
    assert scaled.metadata['superposed_sources'] == {
        'depths': [50.0], 'weights': [complex(w)]}


#: A receiver grid holding the source axis (r = 0) and a depth below the
#: 100 m seabed.
_MASK_DEPTHS, _MASK_RANGES = [50.0, 110.0], [0.0, 200.0, 400.0]

_MASK_RUNS = {}


def _mask_run(key):
    """The default-mode run of the registered engine on the receiver grid
    of ``_MASK_DEPTHS`` × ``_MASK_RANGES``, and the messages it warned."""
    if key not in _MASK_RUNS:
        model = _engine(key)
        env, src, _ = _carriers(key)
        rcv = uacpy.Receiver(depths=_MASK_DEPTHS, ranges=_MASK_RANGES)
        mode = model._default_run_mode_for(None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            field = model.run(env, src, rcv, mode, **_mode_kwargs(mode))
        _MASK_RUNS[key] = field, [str(w.message) for w in caught]
    return _MASK_RUNS[key]


@pytest.mark.parametrize('key', [
    pytest.param(key, marks=_known_break('requested_grid', key))
    for key in _FIELD_RESULT_KEYS])
def test_a_field_keeps_the_requested_depth_and_range_axes(key):
    field, _ = _mask_run(key)
    np.testing.assert_array_equal(field.coords['depth'], _MASK_DEPTHS)
    np.testing.assert_array_equal(field.coords['range'], _MASK_RANGES)


@pytest.mark.parametrize('key', _FIELD_RESULT_KEYS)
def test_a_field_masks_the_cells_no_engine_can_fill(key):
    """source-receiver.md §7 "The source is at range zero": a point-source
    field is singular on its own axis, so the r = 0 column is NaN, with a
    warning naming it; the water-column receiver is finite at every r > 0;
    the receiver below the seabed is either masked or filled across the
    whole row, never in part."""
    field, messages = _mask_run(key)
    data = np.asarray(field.data)
    finite = np.isfinite(data).all(axis=tuple(range(2, data.ndim)))
    assert np.isnan(data[:, 0, ...]).all()
    assert any(re.search(r'r\s*<?=\s*0', m) for m in messages)
    assert finite[0, 1:].all()
    assert finite[1, 1:].all() or not finite[1, 1:].any(), finite[1]


# ── filtered parameter lists ────────────────────────────────────────────


def test_every_filtered_parameter_list_holds_a_case():
    """A filter that drops every engine, or a call illegal on none, would
    leave a test with nothing to run."""
    for cases in (_TIME_SERIES_KEYS, _FIELD_DEFAULT_KEYS, _FIELD_RESULT_KEYS):
        assert cases
    assert {p.values[1] for p in _ILLEGAL_CASES} == set(_ILLEGAL_CALLS)


@pytest.mark.parametrize('key', _KEYS)
def test_the_declared_engine_answers_what_the_built_engine_answers(key):
    """``_Declared`` answers every question the filters ask as the built
    engine does, and each filtered list holds the engine exactly where the
    built engine says the case applies."""
    model, declared = _engine(key), _Declared(key)
    assert declared.supported_modes == model.supported_modes
    assert declared._traits == model._traits
    assert declared._FIELD_MODES == model._FIELD_MODES
    assert (declared._default_run_mode_for(None)
            == model._default_run_mode_for(None))
    assert (declared._run_keywords_never_consumed()
            == model._run_keywords_never_consumed())
    assert (key in _TIME_SERIES_KEYS) == (
        RunMode.TIME_SERIES in model.supported_modes)
    assert (key in _FIELD_DEFAULT_KEYS) == _default_mode_runs_a_field(model)
    illegal = {p.values[1] for p in _ILLEGAL_CASES if p.values[0] == key}
    assert illegal == {label for label in _ILLEGAL_CALLS
                       if _illegal_call(key, model, label) is not None}


@pytest.mark.parametrize('key', _KEYS)
def test_an_engine_declaring_a_field_default_returns_a_depth_range_field(
        key):
    """``_FIELD_RESULT_KEYS`` holds the engines whose default run returns a
    Field over depth and range, and no other."""
    from uacpy.core.results import Field
    field, _ = _mask_run(key)
    assert (key in _FIELD_RESULT_KEYS) == (
        isinstance(field, Field) and {'depth', 'range'} <= set(field.coords))


@pytest.mark.parametrize('key', _KEYS)
def test_an_unpinned_run_leaves_no_directory_behind(key, tmp_path):
    """``tempfile`` points at ``tmp_path`` here (``conftest``), where an
    unpinned run makes its scratch directory; the run removes it."""
    model = _engine(key)
    assert model.work_dir is None
    env, src, rcv = _carriers(key)
    mode = model._default_run_mode_for(None)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.run(env, src, rcv, mode, **_mode_kwargs(mode))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('key', _KEYS)
def test_a_pinned_run_that_keeps_its_files_leaves_the_deck(key, tmp_path):
    work = tmp_path / 'wd'
    model = _engine(key, work_dir=work, cleanup=False)
    env, src, rcv = _carriers(key)
    mode = model._default_run_mode_for(None)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model.run(env, src, rcv, mode, **_mode_kwargs(mode))
    assert any(path.is_file() for path in work.rglob('*'))


@pytest.mark.parametrize('key', _KEYS)
def test_a_failed_launch_releases_its_work_dir(key, tmp_path, launch_spy):
    """A launch that raises removes an unpinned scratch directory and
    releases a pinned one, so the next run may claim it."""
    env, src, rcv = _carriers(key)
    unpinned = _engine(key)
    launch_spy(unpinned)
    with pytest.raises(LaunchReached):
        unpinned.run(env, src, rcv)
    assert list(tmp_path.iterdir()) == []
    pinned = _engine(key, work_dir=tmp_path / 'wd')
    launch_spy(pinned)
    for _ in range(2):
        with pytest.raises(LaunchReached):
            pinned.run(env, src, rcv)


@pytest.mark.parametrize('key', _KEYS)
def test_a_launch_past_its_timeout_is_killed_and_named(key, launch_spy):
    """Every engine's launch route hands the timeout on as
    ``ModelExecutionError(timed_out=True)``, after the launched process
    group is gone."""
    model = _engine(key)
    launch = type(model)._run_subprocess
    marker = f'{7 + os.getpid() % 997 / 1000:.3f}'

    def sleep_past_the_timeout(cmd, cwd, **kwargs):
        return launch(model, [shutil.which('sleep'), marker], cwd=cwd,
                      timeout=0.2)

    launch_spy(model, then=sleep_past_the_timeout)
    env, src, rcv = _carriers(key)
    with pytest.raises(ModelExecutionError,
                       match='execution timed out') as err:
        model.run(env, src, rcv)
    assert err.value.timed_out
    left = subprocess.run(['pgrep', '-f', f'sleep {marker}'],
                          capture_output=True, text=True)
    assert left.stdout == '', left.stdout


@pytest.mark.parametrize('key', _KEYS)
def test_a_missing_binary_names_the_install_step(key, tmp_path):
    entry = ENGINES[key]
    with pytest.raises(ExecutableNotFoundError, match='install.sh'):
        entry.load()(**dict(entry.example_kwargs),
                     executable=tmp_path / 'absent')


@pytest.mark.parametrize('key', _KEYS)
def test_a_work_dir_that_is_not_writable_is_refused_before_a_deck(
        key, tmp_path):
    if os.geteuid() == 0:
        pytest.skip('root writes to a read-only directory anyway')
    work = tmp_path / 'ro'
    work.mkdir()
    work.chmod(0o500)
    try:
        model = _engine(key, work_dir=work, cleanup=False)
        env, src, rcv = _carriers(key)
        with pytest.raises(uacpy.ConfigurationError,
                           match='work_dir is not writable'):
            model.run(env, src, rcv)
    finally:
        work.chmod(0o755)
    assert list(work.iterdir()) == []


# ── copies and parallel runs ────────────────────────────────────────────


@pytest.mark.parametrize('key', _KEYS)
def test_copy_carries_every_constructor_knob(key):
    """``copy`` is the parameter-sweep primitive: it rebuilds the model
    from every constructor knob along the MRO, each stored as
    ``self.<name>``, and a ``collapse`` override survives it."""
    from uacpy.models._introspect import _collect_init_params, _values_equal
    model = _engine(key, collapse={'bathymetry': 'min'})
    twin = model.copy()
    assert type(twin) is type(model)
    for name, _ in _collect_init_params(type(model)):
        assert hasattr(model, name), name
        assert _values_equal(getattr(model, name), getattr(twin, name)), name
    assert twin.collapse == {'bathymetry': 'min'}
    assert twin._collapse['bathymetry'] == 'min'


@pytest.mark.parametrize('key', _KEYS)
def test_copy_refuses_a_knob_the_model_does_not_have(key):
    with pytest.raises(uacpy.ConfigurationError,
                       match=r"copy: unknown override\(s\) \['n_beamz'\]"):
        _engine(key).copy(n_beamz=1)


@pytest.mark.parametrize('key', _KEYS)
def test_run_parallel_returns_what_a_direct_run_returns(key):
    from uacpy.parallel import Job, run_parallel
    model = _engine(key)
    env, src, rcv = _carriers(key)
    mode = model._default_run_mode_for(None)
    direct = _run(key)
    batch = run_parallel([Job(model, env, src, rcv, run_mode=mode,
                              run_kwargs=_mode_kwargs(mode))], n_workers=1)
    (result,) = batch.results
    assert type(result) is type(direct)
    assert result.run_settings == direct.run_settings
    if hasattr(direct, 'data'):
        assert _relative_error(result.data, direct.data) < \
            _RERUN_TOLERANCE.get(key, 1e-9)


class TestAKnobAssignedAfterConstructionIsChecked:
    """A constructed engine takes no public attribute it does not already
    carry: a misspelt knob is refused with the closest name, a real one is
    assigned (lead's finding 2026-09-28)."""

    @pytest.mark.parametrize('key', _KEYS)
    def test_a_misspelt_knob_is_refused_with_the_closest_name(self, key):
        from uacpy.core.exceptions import ConfigurationError
        model = _engine(key)
        with pytest.raises(ConfigurationError,
                           match="no knob 'verbos'.*Did you mean 'verbose'"):
            model.verbos = False
        assert 'verbos' not in vars(model)

    @pytest.mark.parametrize('key', _KEYS)
    def test_a_knob_the_engine_carries_is_assigned(self, key):
        model = _engine(key)
        model.verbose = True
        assert model.verbose is True

    def test_kraken_c_high_misspelt_is_refused(self):
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.models import Kraken
        k = Kraken()
        with pytest.raises(ConfigurationError,
                           match="Did you mean 'c_high'"):
            k.c_hig = 5
        k.c_high = 1600.0
        assert k.c_high == 1600.0

    def test_a_private_name_is_the_engines_own(self):
        from uacpy.models import Kraken
        k = Kraken()
        k._scratch_note = 1
        assert k._scratch_note == 1

    def test_a_copy_is_constructed_too(self):
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.models import Kraken
        clone = Kraken(c_high=1600.0).copy()
        with pytest.raises(ConfigurationError, match='c_high'):
            clone.c_hig = 5


@pytest.mark.parametrize('key', _KEYS)
def test_the_record_states_every_knob_as_given(key):
    """A run's settings record states every knob of the model by name,
    as given (in plain types), in ``engine.knobs`` — the resolved values
    are its fields — and never a host knob that cannot change a result
    (P4.6)."""
    from uacpy.core._records import _plain
    from uacpy.models._introspect import _collect_init_params
    model = _engine(key)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        engine = model.run_settings(*_carriers(key)).engine
    host = PropagationModel._HOST_KNOBS
    for name, _default in _collect_init_params(type(model)):
        if name in host:
            assert name not in engine.knobs, name
        else:
            assert engine.knobs[name] == _plain(getattr(model, name)), name


def test_a_recorded_knob_is_the_value_the_run_used():
    """A deck option the record held under no field before (Bellhop's
    ``beam_shift``) is in ``engine.knobs`` with the value set, and survives
    the record's save and load."""
    from uacpy.core.run_settings import RunSettings
    model = _engine('bellhop')
    model.beam_shift = True
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        settings = model.run_settings(*_carriers('bellhop'))
    assert settings.engine.knobs['beam_shift'] is True
    back = RunSettings.from_dict(settings.to_dict())
    assert back == settings
    assert back.engine.knobs['beam_shift'] is True


def test_a_record_saved_with_strings_and_notes_loads():
    """A record saved before notices carried their class: each string
    reads as a UACPYWarning notice; notes pair with the messages when
    every message had one, and are kept as note-only notices when they
    do not line up; an unknown class name reads as UACPYWarning."""
    from uacpy.core.run_settings import Notice
    from uacpy.models.scooter import ScooterSettings
    engine = json.loads(_SAVED_SETTINGS.read_text(
        encoding='utf-8'))['scooter']['engine']
    engine.pop('__class__')
    paired = ScooterSettings.from_dict(
        dict(engine, notices=['m1', 'm2'], notes=['n1', 'n2']))
    assert paired.notices == (Notice('n1', 'm1', uacpy.UACPYWarning),
                              Notice('n2', 'm2', uacpy.UACPYWarning))
    unpaired = ScooterSettings.from_dict(
        dict(engine, notices=['m1', 'm2'], notes=['n1']))
    assert unpaired.notices == (Notice(None, 'm1', uacpy.UACPYWarning),
                                Notice(None, 'm2', uacpy.UACPYWarning),
                                Notice('n1', 'n1', uacpy.UACPYWarning))
    unknown = ScooterSettings.from_dict(dict(engine, notices=[
        {'note': None, 'message': 'm', 'category': 'NotAWarning'}]))
    assert unknown.notices[0].category is uacpy.UACPYWarning



def _resolved(engine):
    """An engine record's resolved values, nested producer records
    included: everything but the knobs as given and the origins (a pinned
    knob reports its own origin)."""
    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items()
                    if k != 'knobs' and not k.endswith('_origin')}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value
    return strip(engine.to_dict())


@pytest.mark.parametrize('key', _KEYS)
def test_a_record_rebuilds_a_model_that_resolves_the_same_settings(key):
    """``from_run_settings`` gives back a model whose run resolves the
    recorded settings, from the record itself and from the record saved and
    read back (P4.6)."""
    model = _engine(key)
    carriers = _carriers(key)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        first = model.run_settings(*carriers)
        again = type(model).from_run_settings(first).run_settings(*carriers)
        back = RunSettings.from_dict(json.loads(json.dumps(first.to_dict())))
        loaded = PropagationModel.from_run_settings(back).run_settings(
            *carriers)
    assert _resolved(again.engine) == _resolved(first.engine)
    assert _resolved(loaded.engine) == _resolved(first.engine)


class TestFromRunSettingsRefusesWhatItCannotRerun:

    @staticmethod
    def _record(key='kraken'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return _engine(key).run_settings(*_carriers(key))

    def test_a_misspelt_knob_in_an_edited_record_is_refused(self):
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.models import Kraken
        d = self._record().to_dict()
        d['engine']['knobs']['c_hig'] = 5.0
        with pytest.raises(ConfigurationError,
                           match="'c_hig' \\(did you mean 'c_high'\\?\\)"):
            Kraken.from_run_settings(RunSettings.from_dict(d))

    def test_a_valid_edited_knob_is_taken(self):
        from uacpy.models import Kraken
        d = self._record().to_dict()
        d['engine']['knobs']['c_high'] = 1750.0
        model = Kraken.from_run_settings(RunSettings.from_dict(d))
        assert model.c_high == 1750.0

    def test_another_engines_record_is_refused(self):
        from uacpy.core.exceptions import ConfigurationError
        from uacpy.models import Scooter
        with pytest.raises(ConfigurationError, match='written by Kraken'):
            Scooter.from_run_settings(self._record())

    def test_a_leaky_mode_record_rebuilds_with_c_high_unset(self):
        """leaky_modes sets c_high to the unbounded sentinel and refuses a
        c_high beside it, so a re-run pins leaky_modes, not c_high."""
        from uacpy.models import Kraken
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            record = Kraken(leaky_modes=True, verbose=False).run_settings(
                *_carriers('kraken'))
        model = Kraken.from_run_settings(record)
        assert model.leaky_modes is True and model.c_high is None

    def test_replace_is_not_public(self):
        assert not hasattr(RunSettings, 'replace')


class TestAPerLaunchValueIsCheckedAgainstTheRecord:
    """Kraken's c_high is one value per profile of a range-dependent band, so
    no single knob holds it: the rebuilt model keeps the recorded values and
    its first run warns, naming each one the rule now derives differently
    (the lead's ruling on P4.6)."""

    @staticmethod
    def _carriers():
        from uacpy.core import (BoundaryProperties, Environment, Receiver,
                                Source)
        from uacpy.core.ssp import SoundSpeedProfile
        env = Environment(
            bathymetry=np.array([[0.0, 100.0], [4000.0, 100.0]]),
            ssp=SoundSpeedProfile(depths=[0.0, 100.0],
                                  sound_speed=[[1500.0, 1500.0],
                                               [1500.0, 1560.0]],
                                  ranges=[0.0, 4000.0]),
            bottom=BoundaryProperties(acoustic_type='half-space',
                                      sound_speed=1550.0, density=1.8,
                                      attenuation=0.3))
        return (env, Source(depths=30.0, frequencies=60.0),
                Receiver(depths=[30.0], ranges=[1000.0, 3000.0]))

    def _record(self):
        from uacpy.models import Kraken
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return Kraken(n_segments=3, verbose=False).run_settings(
                *self._carriers(), run_mode=RunMode.BROADBAND,
                frequencies=[50.0, 60.0, 70.0])

    def test_the_varying_values_are_kept_not_pinned(self):
        from uacpy.models import Kraken
        record = self._record()
        c_highs = {v for launch in record.engine.launches
                   for v in launch.c_high}
        assert len(c_highs) > 1
        model = Kraken.from_run_settings(record)
        assert model.c_high is None
        assert 'c_high' in model._recorded_per_launch

    def test_an_unchanged_rule_runs_silently(self):
        from uacpy.core.exceptions import ProvenanceWarning
        from uacpy.models import Kraken
        model = Kraken.from_run_settings(self._record())
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            warnings.simplefilter('error', ProvenanceWarning)
            model.run_settings(*self._carriers(), run_mode=RunMode.BROADBAND,
                               frequencies=[50.0, 60.0, 70.0])

    def test_a_changed_rule_warns_naming_the_value(self, monkeypatch):
        from uacpy.core.exceptions import ProvenanceWarning
        from uacpy.models import Kraken
        from uacpy.models.kraken import _window
        model = Kraken.from_run_settings(self._record())
        real = _window.profile_c_high

        def drifted(*args, **kwargs):
            c_high, origin = real(*args, **kwargs)
            return c_high + 1.0, origin
        monkeypatch.setattr(_window, 'profile_c_high', drifted)
        with pytest.warns(ProvenanceWarning,
                          match=r'recorded c_high .* re-derived as'):
            model.run_settings(*self._carriers(), run_mode=RunMode.BROADBAND,
                               frequencies=[50.0, 60.0, 70.0])
