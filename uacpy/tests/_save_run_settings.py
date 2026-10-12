"""Write engines' run-settings records to ``tests/data/run_settings_by_engine.json``.

``test_the_run_settings_record_keeps_its_saved_form`` compares the record
each registered engine writes — its default-mode ``run_settings(...).to_dict()``
on the conformance suite's example guide, which a result saved to disk
carries — with the one saved here: the fields at every level, so a field
added, renamed or removed is a change the suite reports, and a deliberate
one is recorded by rewriting the record. uacpy reads only records this
version writes, so the file holds the current form. Command line, run from
the repository root (the directory holding ``uacpy/``)::

    python -m uacpy.tests._save_run_settings              # every engine
    python -m uacpy.tests._save_run_settings KEY [KEY ...]

writes the record of each registry key (``ENGINES``), replacing a saved
one, and exits 0.
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
    from uacpy.models._registry import ENGINES
    saved = (json.loads(SAVED_SETTINGS.read_text(encoding='utf-8'))
             if SAVED_SETTINGS.exists() else {})
    keys = list(keys) or sorted(ENGINES)
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
