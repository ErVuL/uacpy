"""Shared validators for engine knobs (M-16). Each engine's
``_check_knobs`` hook runs at construction and again in stage 2 of every run
(:meth:`~uacpy.models.base.PropagationModel._check_carriers`), since the
attributes can be reassigned in between; a validator raises the
:class:`~uacpy.core.exceptions.ConfigurationError` that names the knob and
the value it got."""
import difflib
import numbers

import numpy as np

from uacpy.core.exceptions import ConfigurationError


def is_real_number(value) -> bool:
    """Whether ``value`` is a real number that is not NaN (a ``bool`` is
    not)."""
    return (isinstance(value, numbers.Real) and not isinstance(value, bool)
            and not np.isnan(float(value)))


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


def whole_count(name: str, value, minimum: int) -> None:
    """Refuse ``value`` unless it is an ``int`` (not a ``bool``) of at least
    ``minimum``."""
    if (not isinstance(value, int) or isinstance(value, bool)
            or value < minimum):
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
