"""Add an engine's saved run-settings record to ``tests/data/run_settings_by_engine.json``.

``test_a_saved_run_settings_loads_from_the_class_path_it_names`` loads one
saved record per registered engine: the default-mode
``run_settings(...).to_dict()`` on the conformance suite's example guide,
which is what a result saved to disk carries. A new engine needs its record
written once. Command line, run from the repository root (the directory
holding ``uacpy/``)::

    python -m uacpy.tests._save_run_settings KEY [KEY ...]

writes the record of each registry key (``ENGINES``) and exits 0. A key
already in the file is refused and nothing is written: a saved record stands
for every file saved before, so rewriting it would hide the change that
stops such files loading.
"""
import json
import sys
import warnings
from pathlib import Path

SAVED_SETTINGS = Path(__file__).parent / 'data' / 'run_settings_by_engine.json'


def saved_record(key):
    """The JSON form of engine ``key``'s default-mode run settings on the
    conformance guide."""
    from uacpy.models._registry import ENGINES
    from uacpy.tests.test_engine_conformance import _carriers
    entry = ENGINES[key]
    model = entry.load()(**dict(entry.example_kwargs))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        settings = model.run_settings(*_carriers(key))
    return json.loads(json.dumps(settings.to_dict(), sort_keys=True))


def main(keys):
    text = SAVED_SETTINGS.read_text(encoding='utf-8')
    saved = json.loads(text)
    present = sorted(set(keys) & set(saved))
    if present:
        print(f"already saved, not rewritten: {present}", file=sys.stderr)
        return 1
    for key in keys:
        saved[key] = saved_record(key)
    out = json.dumps(dict(sorted(saved.items())), indent=1) + '\n'
    tmp = SAVED_SETTINGS.with_suffix('.json.tmp')
    tmp.write_text(out, encoding='utf-8')
    tmp.replace(SAVED_SETTINGS)
    print(f"saved: {sorted(keys)}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
