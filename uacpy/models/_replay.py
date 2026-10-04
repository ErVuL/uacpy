"""Re-running a recorded run: the model a run's settings record describes
(:meth:`PropagationModel.from_run_settings
<uacpy.models.base.PropagationModel.from_run_settings>`), and the check its
first run makes of the values the record holds per launch.

The record states every knob of the run: as given in
``run_settings.engine.knobs`` and, where the run derived it (the knob given
as ``None``), resolved in the field of the same name. The model rebuilt
from it takes each derived value as an explicit knob, so a later change of
the rule that derives it cannot change the re-run. A value the run derived
per launch — one per range segment or per frequency — is not one knob; the
model keeps the recorded values and, when its first run re-derives them,
warns naming each that differs.
"""

import dataclasses
import difflib
import warnings

from uacpy.core._records import _plain
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.exceptions import ConfigurationError, ProvenanceWarning
from uacpy.core.run_settings import RunSettings
from uacpy.models._introspect import _collect_init_params


def _engine_class(name: str, who: str):
    """The public engine class a record's ``model`` names."""
    import uacpy.models as models
    cls = getattr(models, name, None)
    if not (isinstance(cls, type) and hasattr(cls, 'from_run_settings')):
        raise ConfigurationError(
            f"{who}: the record names model {name!r}, which is no uacpy "
            f"engine.")
    return cls


def model_from_run_settings(cls, run_settings):
    """The model the run ``run_settings`` records, every value the run
    derived once pinned as an explicit knob (see the module docstring).
    ``cls`` is the class the call was made on: an engine, which must be the
    record's, or the abstract base, which takes the record's own."""
    who = f"{cls.__name__}.from_run_settings"
    if not isinstance(run_settings, RunSettings):
        raise ConfigurationError(
            f"{who} takes a RunSettings (result.run_settings); got "
            f"{type(run_settings).__name__}.")
    engine = run_settings.engine
    if engine is None:
        raise ConfigurationError(
            f"{who}: this record holds no engine settings, so it states no "
            f"knob to pin; it was not written by an engine's run.")
    target = _engine_class(run_settings.model, who)
    if not getattr(cls, '__abstractmethods__', None) and cls is not target:
        raise ConfigurationError(
            f"{who}: the record was written by {target.__name__}, not "
            f"{cls.__name__}.",
            remediation=f"Call {target.__name__}.from_run_settings(...).")
    params = [name for name, _ in _collect_init_params(target)
              if name not in target._HOST_KNOBS]
    unknown = sorted(set(engine.knobs) - set(params))
    if unknown:
        hints = [f"{name!r} (did you mean {close[0]!r}?)"
                 if (close := difflib.get_close_matches(name, params, n=1))
                 else repr(name) for name in unknown]
        raise ConfigurationError(
            f"{who}: the record names knob(s) {', '.join(hints)} that "
            f"{target.__name__} does not take, so it was edited or written "
            f"by another version; it cannot be re-run as written.",
            remediation=(f"Correct or remove the knob in "
                         f"run_settings.engine.knobs; {target.__name__} "
                         f"takes: {', '.join(params)}."))
    fields = {f.name for f in dataclasses.fields(engine)}
    kwargs = {}
    for name in params:
        if name not in engine.knobs:
            continue
        given = engine.knobs[name]
        value = getattr(engine, name) if name in fields else None
        if isinstance(value, RunSettings):
            # A producer the caller gave (OASS/OASSP's mean field) is
            # rebuilt from its own record; one the run built itself is
            # built again from this model's knobs.
            kwargs[name] = (None if given is None else
                            model_from_run_settings(
                                _engine_class(value.model, who), value))
        elif (given is None and name in fields
              and name not in target._UNPINNED_FIELDS):
            kwargs[name] = value
        else:
            kwargs[name] = given
    per_launch = {}
    for name, values in target._per_launch_knob_values(engine).items():
        if engine.knobs.get(name) is not None:
            continue
        plain = [_plain(v) for v in values]
        if plain and all(v == plain[0] for v in plain):
            kwargs[name] = values[0]
        else:
            per_launch[name] = plain
    model = target(**kwargs)
    model._recorded_per_launch = per_launch
    return model


def check_recorded_per_launch(model, engine) -> None:
    """At the first resolution of a model rebuilt by
    :func:`model_from_run_settings`, warn for each per-launch value the
    rule now derives differently from the record (old -> new); the run uses
    the re-derived values. Later resolutions check nothing."""
    recorded = vars(model).pop('_recorded_per_launch', None)
    if not recorded:
        return
    now = type(model)._per_launch_knob_values(engine)
    for name, old in recorded.items():
        new = [_plain(v) for v in now.get(name, ())]
        if new != old:
            warnings.warn(
                f"{model.model_name}: the recorded {name} {old} is "
                f"re-derived as {new} — the rule that derives it changed "
                f"since the record was written. The run uses {new}.",
                ProvenanceWarning, skip_file_prefixes=USER_FRAME_SKIP)
