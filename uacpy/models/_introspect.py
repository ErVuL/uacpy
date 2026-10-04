"""The constructor introspection behind ``PropagationModel.copy`` and
``__repr__``."""

from typing import Dict, List

import numpy as np

from uacpy.core._repr import knob


_NO_DEFAULT = object()


def _collect_init_params(cls) -> List[tuple]:
    """Walk ``cls.__mro__`` for every ``__init__`` and collect named
    parameters (excluding ``self`` and ``**kwargs``) in declaration order,
    deduplicated by name (subclass declaration wins).

    Returns a list of ``(name, default_or_NO_DEFAULT)`` pairs. Used by
    :meth:`PropagationModel.__repr__` and parallels what
    :meth:`PropagationModel.copy` introspects.
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


def _short_repr(value) -> str:
    """A constructor value as :meth:`PropagationModel.__repr__` shows it:
    the shared :func:`~uacpy.core._repr.knob` form, so ``print(model)``
    stays one short line even when a knob holds a large array."""
    return knob(value)
