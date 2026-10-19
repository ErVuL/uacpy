"""The constructor introspection behind ``PropagationModel.copy`` and
``__repr__`` (:func:`model_copy`, :func:`model_repr`)."""

from typing import Dict, List

import numpy as np

from uacpy.core._repr import knob
from uacpy.core.exceptions import ConfigurationError


_NO_DEFAULT = object()


def _collect_init_params(cls) -> List[tuple]:
    """Walk ``cls.__mro__`` for every ``__init__`` and collect named
    parameters (excluding ``self`` and ``**kwargs``) in declaration order,
    deduplicated by name (subclass declaration wins).

    Returns a list of ``(name, default_or_NO_DEFAULT)`` pairs. Read by
    :meth:`PropagationModel.__repr__`, by :func:`model_copy`, by the
    record a run's settings carry (``PropagationModel._given_knobs``) and by
    ``from_run_settings`` (``models/_replay.py``).
    """
    import inspect as _inspect

    seen: Dict[str, object] = {}
    order: List[str] = []
    for klass in cls.__mro__:
        if klass is object:
            continue
        init = klass.__dict__.get('__init__')
        if init is None:
            continue
        try:
            sig = _inspect.signature(init)
        except (TypeError, ValueError):
            continue
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            if param.kind == _inspect.Parameter.VAR_KEYWORD:
                continue
            if param.kind == _inspect.Parameter.VAR_POSITIONAL:
                continue
            if name in seen:
                continue
            seen[name] = (
                param.default if param.default is not _inspect.Parameter.empty
                else _NO_DEFAULT
            )
            order.append(name)
    return [(name, seen[name]) for name in order]


def _values_equal(a, b) -> bool:
    """Compare two configuration values, tolerating ndarray equality."""
    if a is b:
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(a), np.asarray(b)))
        except Exception:
            return False
    try:
        return bool(a == b)
    except Exception:
        return False


def model_copy(model, overrides: dict):
    """A new instance of ``model``'s class with the same configuration plus
    ``overrides`` (:meth:`PropagationModel.copy`): every constructor
    parameter along the MRO (:func:`_collect_init_params`) read off
    ``model.<name>``, ``overrides`` merged over them; an override that is
    not a constructor parameter is refused."""
    kwargs: Dict[str, object] = {}
    valid_names = set()
    for name, _default in _collect_init_params(type(model)):
        valid_names.add(name)
        if hasattr(model, name):
            kwargs[name] = getattr(model, name)

    unknown = set(overrides) - valid_names
    if unknown:
        raise ConfigurationError(
            f"{type(model).__name__}.copy: unknown override(s) "
            f"{sorted(unknown)}; valid parameters are "
            f"{sorted(valid_names)}."
        )

    kwargs.update(overrides)

    # An auto-derived ``cleanup`` describes *this* instance's work_dir, not
    # the clone's. Hand back the ``None`` sentinel so the new instance
    # re-resolves; otherwise ``model.copy(work_dir=d)`` off an unpinned
    # model would arrive at ``cleanup=True`` and rmtree the caller's ``d``.
    # An explicitly passed value is the caller's choice and is preserved.
    if (not model._cleanup_explicit and 'cleanup' not in overrides):
        kwargs['cleanup'] = None

    return type(model)(**kwargs)


def model_repr(model) -> str:
    """``ClassName(arg=val, …)`` of ``model`` (:meth:`PropagationModel.
    __repr__`): the constructor parameters along the MRO whose current
    value differs from the constructor default, read off
    ``model.<name>``; a ``backend=`` left ``None`` shown as the backend
    the instance resolved (``model._repr_backend()``), when it resolves
    one."""
    bits: List[str] = []
    for name, default in _collect_init_params(type(model)):
        if not hasattr(model, name):
            continue
        # Resolved binary paths are machine-specific and not copy-paste-
        # portable — hide them regardless of value.
        if name in ('executable', 'field_executable'):
            continue
        value = getattr(model, name)
        if name == 'backend' and value is None:
            value = model._repr_backend()
        # ``cleanup`` resolves to ``work_dir is None`` when left at its None
        # default; hide it while it carries that auto value, matching
        # ``copy()``, which hands the ``None`` sentinel back so the clone
        # re-resolves against its own work_dir.
        if name == 'cleanup' and not model._cleanup_explicit:
            continue
        if default is not _NO_DEFAULT and _values_equal(value, default):
            continue
        # The shared knob form keeps print(model) one short line even when
        # a knob holds a large array.
        bits.append(f"{name}={knob(value)}")
    return f"{type(model).__name__}({', '.join(bits)})"
