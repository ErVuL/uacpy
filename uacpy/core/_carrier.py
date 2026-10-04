"""Behaviour every carrier shares: the :func:`carrier` class decorator, a deep
``copy()``, and the constructor's checks re-run on assignment.

``@carrier`` makes a dataclass whose ``__init__`` is an ordinary function of
this module rather than the one the dataclass decorator compiles from a string.
A generated ``__init__`` lives in the pseudo-file ``<string>``, where the
warning-attribution walk (:data:`uacpy.core._warn_frames.USER_FRAME_SKIP`)
stops; this one is a package frame the walk steps over, so a warning raised
while a carrier is built — from the user's call, from an in-package factory, or
from a rebuild on assignment — names the line the user wrote.
"""

import copy as _copy
import dataclasses
import inspect
import threading
from typing import Self, dataclass_transform

__all__ = ['carrier', 'DeepCopyMixin', 'RevalidateOnAssignMixin']

# The ids of the carriers this thread is building: a store to one of them is
# construction, a store to any other an assignment (RevalidateOnAssignMixin).
# Kept here rather than as a flag on the instance, so a carrier's ``__dict__``
# holds its fields and nothing else.
_constructing = threading.local()


def _under_construction() -> set:
    """This thread's set of carriers whose constructor has not returned."""
    ids = getattr(_constructing, 'ids', None)
    if ids is None:
        ids = _constructing.ids = set()
    return ids


def _names(names) -> str:
    """``'a'``, ``'a' and 'b'``, ``'a', 'b', and 'c'``: how CPython lists the
    arguments a call is missing."""
    quoted = [repr(n) for n in names]
    if len(quoted) == 1:
        return quoted[0]
    if len(quoted) == 2:
        return f"{quoted[0]} and {quoted[1]}"
    return ", ".join(quoted[:-1]) + f", and {quoted[-1]}"


def _carrier_init(cls):
    """The ``__init__`` of the carrier class ``cls``: the dataclass fields
    bound from the call as the generated ``__init__`` binds them (positional
    in field order, keyword-only fields by keyword, defaults and default
    factories for the rest), stored, then ``__post_init__`` run. A call the
    generated ``__init__`` would refuse raises the same ``TypeError`` with
    the same message."""
    fields = [f for f in dataclasses.fields(cls) if f.init]
    positional = [f for f in fields if not f.kw_only]
    names = {f.name for f in fields}
    qualname = f"{cls.__qualname__}.__init__"

    def has_default(f):
        return (f.default is not dataclasses.MISSING
                or f.default_factory is not dataclasses.MISSING)

    # Counted with ``self``, as CPython counts them.
    most = len(positional) + 1
    least = sum(1 for f in positional if not has_default(f)) + 1
    post_init = getattr(cls, '__post_init__', None)

    def __init__(self, *args, **kwargs):
        if len(args) >= most:
            given = len(args) + 1
            takes = (f"{most} positional argument{'s' if most != 1 else ''}"
                     if least == most else
                     f"from {least} to {most} positional arguments")
            raise TypeError(
                f"{qualname}() takes {takes} but {given} "
                f"{'were' if given != 1 else 'was'} given.")
        values = {f.name: arg for f, arg in zip(positional, args)}
        for name, value in kwargs.items():
            if name in values:
                raise TypeError(
                    f"{qualname}() got multiple values for argument {name!r}.")
            if name not in names:
                raise TypeError(
                    f"{qualname}() got an unexpected keyword argument "
                    f"{name!r}.")
            values[name] = value
        for kind, group in (('positional', positional),
                            ('keyword-only',
                             [f for f in fields if f.kw_only])):
            missing = [f.name for f in group
                       if f.name not in values and not has_default(f)]
            if missing:
                raise TypeError(
                    f"{qualname}() missing {len(missing)} required {kind} "
                    f"argument{'s' if len(missing) > 1 else ''}: "
                    f"{_names(missing)}.")
        building = _under_construction()
        building.add(id(self))
        try:
            for f in fields:
                if f.name in values:
                    value = values[f.name]
                elif f.default is not dataclasses.MISSING:
                    value = f.default
                else:
                    value = f.default_factory()
                object.__setattr__(self, f.name, value)
            if post_init is not None:
                self.__post_init__()
        finally:
            building.discard(id(self))

    __init__.__qualname__ = qualname
    __init__.__module__ = cls.__module__
    return __init__


@dataclass_transform()
def carrier(cls=None, /, *, init_annotations=None, **options):
    """Class decorator: ``@dataclass(**options)`` with the ``__init__``
    replaced by :func:`_carrier_init`'s, which keeps the generated one's
    signature (``inspect.signature`` and ``help()`` show the same
    parameters).

    ``init_annotations`` maps a parameter name to the type the constructor
    takes where it is wider than what the attribute holds (``Source(depths=
    50.0)`` stores an ndarray): the signature and ``__init__.__annotations__``
    state it, and the class annotations — what an attribute read is checked
    against — keep the field's.

    No ``__repr__`` is generated (``repr=False`` unless given): a carrier
    writes its own one-line repr, or inherits its base's (docs/DEV.md §5.2)."""
    def wrap(cls):
        options.setdefault('repr', False)
        cls = dataclasses.dataclass(**options)(cls)
        generated = cls.__init__
        signature = inspect.signature(generated)
        annotations = dict(generated.__annotations__)
        if init_annotations:
            unknown = set(init_annotations) - set(signature.parameters)
            if unknown:
                raise TypeError(f"{cls.__qualname__}: init_annotations names "
                                f"no parameter {sorted(unknown)}.")
            signature = signature.replace(parameters=[
                p.replace(annotation=init_annotations[p.name])
                if p.name in init_annotations else p
                for p in signature.parameters.values()])
            annotations.update(init_annotations)
        init = _carrier_init(cls)
        init.__signature__ = signature
        init.__annotations__ = annotations
        cls.__init__ = init
        return cls
    return wrap if cls is None else wrap(cls)


class DeepCopyMixin:
    """``copy()`` shared by the carriers and the results."""

    def copy(self) -> Self:
        """Deep copy: every field — nested carriers and arrays included — is
        duplicated, with no aliasing back to the original instance."""
        return _copy.deepcopy(self)


class RevalidateOnAssignMixin:
    """Re-runs a carrier's constructor checks when one of its fields is
    assigned after construction.

    A store made while the carrier is being built — by ``__init__`` and by
    ``__post_init__`` — passes straight through. Once the constructor has
    returned, a store to a field is an assignment: the carrier is rebuilt
    from its current fields with the new value, so the assignment gets the
    constructor's normalisation and refusals, and a refused assignment leaves
    the carrier unchanged. A copy or an unpickled carrier restores its
    ``__dict__`` without calling ``__setattr__``.
    """

    def _fields_for_assignment(self, name, value) -> dict:
        """The constructor arguments that assigning ``value`` to ``name``
        stands for. A carrier whose fields move together overrides it."""
        fields = {f: self.__dict__[f] for f in self.__dataclass_fields__}
        fields[name] = value
        return fields

    def __setattr__(self, name, value):
        if (name in self.__dataclass_fields__
                and id(self) not in _under_construction()):
            checked = type(self)(**self._fields_for_assignment(name, value))
            for f in self.__dataclass_fields__:
                object.__setattr__(self, f, checked.__dict__[f])
            return
        object.__setattr__(self, name, value)
