"""Write every engine's declared spec and resolved capabilities to ``tests/data/engine_specs.json``.

``test_capability_flags.py`` compares, field by field, the record each
registered engine gives now with the one saved here: its ``ModelSpec`` (run
modes, capability flags, source geometries, collapse defaults, every
``EngineTraits`` field) and what a default instance resolves from it (its run
modes, every capability flag, its source geometries, its collapse policy, its
default run mode, and what a two-depth ``Source`` meets in that mode). A
change to any of them fails naming the engine and the field; a deliberate
one is recorded by rewriting the record. Command line, run from the
repository root (the directory holding ``uacpy/``)::

    python -m uacpy.tests._save_engine_specs              # every engine
    python -m uacpy.tests._save_engine_specs KEY [KEY ...]

writes the record of each registry key (``ENGINES``), replacing a saved
one, and exits 0. The instance half constructs each engine, so its binary
must be installed.
"""
import json
import re
import sys
import warnings
from pathlib import Path

SAVED_SPECS = Path(__file__).parent / 'data' / 'engine_specs.json'

#: The ``ConfigurationError`` a mode with no per-source sum raises for a
#: multi-depth ``Source`` (``PropagationModel._validate_geometry``).
_SINGLE_DEPTH_REFUSAL = re.compile('single source depth per .* run')


def _plain(value):
    """``value`` in JSON types: a set of run modes or of names as a sorted
    list of names, a tuple as a list."""
    from uacpy.core.run_settings import RunMode
    if isinstance(value, (set, frozenset)):
        return sorted(m.name if isinstance(m, RunMode) else m for m in value)
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    if isinstance(value, RunMode):
        return value.name
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in sorted(value.items())}
    return value


def spec_record(cls) -> dict:
    """What engine class ``cls`` declares in ``cls.spec``: its run modes in
    declaration order (the first is the default unless the class overrides
    ``_default_run_mode``), its capability flags, source geometries and
    collapse defaults, and every ``EngineTraits`` field."""
    from dataclasses import fields
    spec = cls.spec
    return {
        'modes': [m.name for m in spec.modes],
        'supports': _plain(frozenset(spec.supports)),
        'source_types': _plain(frozenset(spec.source_types)),
        'collapse': _plain(dict(spec.collapse)),
        'traits': {f.name: _plain(getattr(spec.traits, f.name))
                   for f in fields(spec.traits)},
    }


def _two_depth_call(key):
    """``(env, source, receiver)``: a 200 m isovelocity guide over the
    seabed the engine's registry entry names (``EngineEntry.example_bottom``,
    the default half-space otherwise), a two-depth source at 100 Hz and one
    receiver depth over 100-2000 m."""
    import numpy as np
    import uacpy
    from uacpy.models._registry import ENGINES
    bottom = dict(ENGINES[key].example_bottom)
    kw = {'bottom': uacpy.BoundaryProperties(**bottom)} if bottom else {}
    env = uacpy.Environment(
        bathymetry=200.0,
        ssp=uacpy.SoundSpeedProfile(depths=[0, 200], sound_speed=[1500, 1500]),
        **kw)
    source = uacpy.Source(depths=[30.0, 60.0], frequencies=100.0)
    receiver = uacpy.Receiver(depths=100.0,
                              ranges=np.linspace(100, 2000, 20))
    return env, source, receiver


def multi_depth_source_in_default_mode(key, model) -> str:
    """What a two-depth ``Source`` meets in ``model``'s default run mode,
    asked of ``validate_inputs``: ``'field'`` — the default mode is a field
    mode, where ``run()`` loops over the depths (or the deck carries them)
    and returns one ``Field`` over ``source_depth``; ``'ignored'`` — the
    call is accepted in a mode with no per-source sum, because the engine
    reads no source geometry (a REFLECTION run: Bounce, OASR) and the
    extra depth reaches no deck; ``'refused'`` — the default
    mode has no per-source sum and the depths are refused."""
    from uacpy.core.exceptions import ConfigurationError
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.validate_inputs(*_two_depth_call(key))
    except ConfigurationError as exc:
        if not _SINGLE_DEPTH_REFUSAL.search(str(exc)):
            raise
        return 'refused'
    return ('field' if model._default_run_mode() in model._FIELD_MODES
            else 'ignored')


def instance_record(key, model) -> dict:
    """What the default instance ``model`` of registry engine ``key``
    resolves: its run modes in order, every capability flag (read through
    ``supports_feature``), its source geometries, its whole collapse policy,
    its default run mode and :func:`multi_depth_source_in_default_mode`."""
    from uacpy.models._spec import _CAPABILITY_FLAGS
    return {
        'supported_modes': [m.name for m in model.supported_modes],
        'features': {name: model.supports_feature(name)
                     for name in sorted(_CAPABILITY_FLAGS)},
        'source_types': _plain(frozenset(model._supported_source_types)),
        'collapse': _plain(dict(model._collapse)),
        'default_run_mode': model._default_run_mode().name,
        'multi_depth_source_in_default_mode':
            multi_depth_source_in_default_mode(key, model),
    }


def default_engine(key):
    """Registry engine ``key`` built with its ``example_kwargs`` (the knobs
    it requires) and ``verbose=False``."""
    from uacpy.models._registry import ENGINES
    entry = ENGINES[key]
    return entry.load()(**{**dict(entry.example_kwargs), 'verbose': False})


def saved_record(key) -> dict:
    """The JSON form of registry engine ``key``'s record: its class name,
    :func:`spec_record` and :func:`instance_record`."""
    from uacpy.models._registry import ENGINES
    cls = ENGINES[key].load()
    record = {'class': cls.__name__, 'spec': spec_record(cls),
              'instance': instance_record(key, default_engine(key))}
    return json.loads(json.dumps(record, sort_keys=True))


def main(keys):
    from uacpy.models._registry import ENGINES
    saved = (json.loads(SAVED_SPECS.read_text(encoding='utf-8'))
             if SAVED_SPECS.exists() else {})
    keys = list(keys) or sorted(ENGINES)
    for key in keys:
        saved[key] = saved_record(key)
    out = json.dumps(dict(sorted(saved.items())), indent=1,
                     sort_keys=True) + '\n'
    tmp = SAVED_SPECS.with_suffix('.json.tmp')
    tmp.write_text(out, encoding='utf-8')
    tmp.replace(SAVED_SPECS)
    print(f"saved: {sorted(keys)}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
