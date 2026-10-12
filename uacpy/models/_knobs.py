"""Shared validators for engine knobs. Each engine's
``_check_knobs`` hook runs at construction and again in stage 2 of every run
(:meth:`~uacpy.models.base.PropagationModel._check_carriers`), since the
attributes can be reassigned in between; a validator raises the
:class:`~uacpy.core.exceptions.ConfigurationError` that names the knob and
the value it got."""
import difflib
import functools
import inspect
import numbers
import typing

import numpy as np

from uacpy.core.exceptions import ConfigurationError
from uacpy.core._validate import real_scalar_or_none, require_real_scalar


def is_real_number(value) -> bool:
    """Whether ``value`` is one real number by the package's one rule
    (:func:`~uacpy.core._validate.real_scalar_or_none`: a bool, a complex
    value or text is not) that is not NaN."""
    number = real_scalar_or_none(value)
    return number is not None and not np.isnan(number)


@functools.lru_cache(maxsize=None)
def numeric_knob_names(cls) -> frozenset:
    """The constructor parameters of engine ``cls`` that take one number:
    those whose default is a number (not a bool) or whose annotation is
    ``int`` or ``float``, optional or not."""
    try:
        hints = typing.get_type_hints(cls.__init__)
    except (NameError, TypeError):
        hints = {}
    names = set()
    for name, p in inspect.signature(cls.__init__).parameters.items():
        if isinstance(p.default, numbers.Real) and not isinstance(
                p.default, bool):
            names.add(name)
            continue
        hint = hints.get(name)
        kinds = (set(typing.get_args(hint))
                 if typing.get_origin(hint) is typing.Union else {hint})
        kinds.discard(type(None))
        if kinds and kinds <= {int, float}:
            names.add(name)
    return frozenset(names)


def read_numeric_knob(model_name: str, name: str, value):
    """The numeric knob ``name`` of engine ``model_name`` as the Python
    number it reads as by the package's one rule — an ``int`` for an
    integer (Python, NumPy or a 0-d array of one), a ``float`` otherwise —
    or ``None`` left as ``None``; a bool, a complex value or text is
    refused, naming the engine and the knob, before any engine casts it."""
    if value is None:
        return None
    number = require_real_scalar(
        value, f"{model_name}: {name}",
        why=f" Pass {name} as a number.")
    integral = (isinstance(value, numbers.Integral)
                or np.asarray(value).dtype.kind in 'iu')
    return int(value) if integral else number


def positive_finite(name: str, value, *, optional: bool = False) -> None:
    """Refuse ``value`` unless it is a positive finite real number (a
    ``bool`` is not one); ``None`` passes when ``optional`` (the knob is
    unset)."""
    if optional and value is None:
        return
    if not (is_real_number(value) and np.isfinite(value) and value > 0):
        when = ' if set' if optional else ''
        raise ConfigurationError(
            f"{name} must be positive and finite{when}; got {value!r}.")


def read_numeric_knobs(cls, args, kwargs):
    """``(args, kwargs)`` of a constructor call of engine ``cls`` with each
    numeric knob it passes (:func:`numeric_knob_names`) read by
    :func:`read_numeric_knob`: refused before the engine's constructor
    runs, so before any cast of its own. A call that does not bind is left
    to raise its own ``TypeError``."""
    names = numeric_knob_names(cls)
    if not names:
        return args, kwargs
    try:
        bound = inspect.signature(cls.__init__).bind(None, *args, **kwargs)
    except TypeError:
        return args, kwargs
    for name in names & set(bound.arguments):
        bound.arguments[name] = read_numeric_knob(
            cls.__name__, name, bound.arguments[name])
    return bound.args[1:], bound.kwargs


def is_whole_number(value) -> bool:
    """Whether ``value`` is an integer: a Python ``int``, a NumPy integer
    or a 0-d array of one, not a ``bool`` (the package's one rule for a
    number, :func:`~uacpy.core._validate.real_scalar_or_none`)."""
    return (real_scalar_or_none(value) is not None
            and (isinstance(value, numbers.Integral)
                 or np.asarray(value).dtype.kind in 'iu'))


def whole_count(name: str, value, minimum: int) -> None:
    """Refuse ``value`` unless it is an integer (:func:`is_whole_number`)
    of at least ``minimum``."""
    if not is_whole_number(value) or value < minimum:
        raise ConfigurationError(
            f"{name} must be an integer >= {minimum}; got {value!r}.")


def refuse_unknown_knob(model, name: str) -> None:
    """Refuse assigning the public attribute ``name`` on a constructed
    engine ``model`` unless the engine already carries it (a constructor
    knob, a resolved attribute, a method or property of its class).

    A name it does not carry is read by no run, so a misspelt knob
    (``kraken.c_hig = 5``) would leave the run on the default; the refusal
    names the closest attribute it does carry. A name starting with ``_``
    passes, so private state stays the engine's own business."""
    if (name.startswith('_') or name in vars(model)
            or hasattr(type(model), name)):
        return
    carried = sorted(n for n in vars(model) if not n.startswith('_'))
    close = difflib.get_close_matches(name, carried, n=1)
    hint = f" Did you mean {close[0]!r}?" if close else ""
    raise ConfigurationError(
        f"{type(model).__name__} has no knob {name!r}: assigning it "
        f"would change nothing a run reads.{hint}",
        remediation=(f"Set one of the knobs help({type(model).__name__}) "
                     f"lists, or pass it to the constructor."))
