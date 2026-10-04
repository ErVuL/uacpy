"""The frozen-record machinery the run settings are built on.

A record here is a frozen dataclass whose arrays are read-only copies, whose
equality compares its plain-type form, and whose :meth:`FrozenRecord.to_dict`
writes plain Python types. :mod:`uacpy.core.run_settings` builds every
settings record on :class:`FrozenRecord`, and an engine builds the records
nested in its own settings (Kraken's launches, RAM's grids) on it too.

A package-internal module: :class:`FrozenRecord` and :func:`array_summary`
are shared by ``core`` and ``models`` and are not part of the user API.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import PurePath
from typing import Any, Dict, Optional, Tuple

import numpy as np

from uacpy.core._repr import FieldsRepr

__all__ = ['FrozenMapping', 'FrozenRecord', 'array_summary']


def _frozen_array(values, dtype=float) -> Optional[np.ndarray]:
    """A read-only 1-D copy of ``values``, or ``None``."""
    if values is None:
        return None
    arr = np.array(values, dtype=dtype).ravel()
    arr.setflags(write=False)
    return arr


def array_summary(arr: Optional[np.ndarray], unit: str) -> str:
    """``'100 Hz'``, ``'20, 50 m'``, ``'128 values, 75-125 Hz'`` or
    ``'none'``."""
    if arr is None:
        return 'none'
    if arr.size == 0:
        return 'empty'
    if np.iscomplexobj(arr):
        return (f"{arr.size} value{'s' if arr.size != 1 else ''}: "
                f"{[complex(v) for v in arr[:6]]}"
                + (' ...' if arr.size > 6 else ''))
    if arr.size <= 4:
        return (', '.join(f"{float(v):g}" for v in arr) + f" {unit}").rstrip()
    return (f"{arr.size} values, {float(arr.min()):g}-{float(arr.max()):g} "
            f"{unit}").rstrip()


def _plain(value):
    """``value`` as plain Python types: lists for arrays, ``[re, im]`` pairs
    for complex numbers, the value string for a ``str`` enum member, a dict
    for any mapping and a string for a filesystem path."""
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str):
            return str(getattr(value, 'value', value))
        return value
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return [[float(v.real), float(v.imag)] for v in value.ravel()]
        return [float(v) for v in value.ravel()]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, PurePath):
        return str(value)
    if dataclasses.is_dataclass(value):
        return value.to_dict()
    return repr(value)


class FrozenMapping(Mapping):
    """A read-only name -> value mapping a frozen record holds (it
    pickles and copies, as a ``MappingProxyType`` does not)."""

    def __init__(self, items=()):
        self._items = dict(items)

    def __getitem__(self, key):
        return self._items[key]

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __eq__(self, other):
        return isinstance(other, Mapping) and dict(self) == dict(other)

    def __hash__(self):
        return hash(repr(sorted(self._items.items(), key=lambda kv: kv[0])))

    def __repr__(self):
        return repr(self._items)


class FrozenRecord(FieldsRepr):
    """Shared behaviour of the frozen records: read-only arrays are
    restored after unpickling (numpy hands them back writeable), equality
    compares the ``to_dict`` forms, so two records holding equal arrays
    compare equal, and the repr is the one-line :class:`FieldsRepr`."""

    _ARRAY_FIELDS: Tuple[str, ...] = ()

    def _freeze_arrays(self) -> None:
        for name in self._ARRAY_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            dtype = complex if np.iscomplexobj(value) else float
            object.__setattr__(self, name, _frozen_array(value, dtype))

    def __setstate__(self, state):
        for name, value in state.items():
            object.__setattr__(self, name, value)
        self._freeze_arrays()

    def __eq__(self, other):
        if type(other) is not type(self):
            return NotImplemented
        return self.to_dict() == other.to_dict()

    def __hash__(self):
        return hash(repr(self.to_dict()))

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: _plain(getattr(self, f.name))
                for f in dataclasses.fields(self)}
