"""The formatting every public ``__repr__`` shares (docs/DEV.md §5.2).

A repr is one line, ``ClassName(bit, bit, ...)``: a quantity is its number in
``:g`` form, a space and its unit (``'20 m'``); an axis of more than
:data:`SHOWN_VALUES` values is its count and extent (``'12 depths 5–95 m'``),
a shorter one its values (``'depths [10, 50] m'``, ``'depth 20 m'``); a nested
object is summarised in its parent's words, never as a second constructor.

A package-internal module: its names are shared by every package that
defines a public class and are not part of the user API.
"""

from __future__ import annotations

import dataclasses
import inspect
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

__all__ = ['FieldsRepr', 'SettingsRepr', 'SHOWN_VALUES', 'knob', 'settings_repr', 'num', 'qty', 'axis', 'extent', 'array_bit', 'value_bit',
           'build', 'count', 'fields_repr', 'plural', 'unit_of']

#: The longest list a repr shows value by value; a longer one is a count
#: and an extent.
SHOWN_VALUES = 4

#: The unit a field name's suffix states, for a record that declares none.
_SUFFIX_UNITS = (('_dB', 'dB'), ('_db', 'dB'), ('_hz', 'Hz'), ('_Hz', 'Hz'),
                 ('_km', 'km'), ('_m', 'm'), ('_s', 's'), ('_c', '°C'),
                 ('_psu', 'psu'), ('_deg', '°'), ('_dbar', 'dbar'))


#: The unit of a field whose name is one of the package's axis names, or
#: the water's temperature and salinity.
_NAME_UNITS = {'depths': 'm', 'ranges': 'm', 'frequencies': 'Hz',
               'times': 's', 'lats': '°', 'lons': '°',
               'temperature': '°C', 'salinity': 'psu'}


def unit_of(name: str) -> str:
    """The unit the field name ``name`` states, by the axis name it is or
    the suffix it ends in, or ``''``."""
    if name in _NAME_UNITS:
        return _NAME_UNITS[name]
    return next((unit for suffix, unit in _SUFFIX_UNITS
                 if name.endswith(suffix)), '')


def num(value) -> str:
    """``value`` in ``:g`` form; an integer as itself, a complex number as
    ``a+bj``."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (complex, np.complexfloating)):
        value = complex(value)
        if value.imag == 0:
            return f"{value.real:g}"
        return f"{value.real:g}{value.imag:+g}j"
    return f"{float(value):g}"


def qty(value, unit: str = '') -> str:
    """``'20 m'``: the number, a space and the unit (no space without
    one)."""
    return f"{num(value)} {unit}" if unit else num(value)


def count(n: int, noun: str) -> str:
    """``'1 mode'``, ``'14 modes'``: ``n`` and the noun, plural unless
    ``n`` is one."""
    return f"{n} {noun if n == 1 else plural(noun)}"


def plural(word: str) -> str:
    """The plural of a coordinate noun: ``'depths'``, ``'frequencies'``."""
    if word.endswith('y') and not word.endswith(('ay', 'ey', 'oy')):
        return word[:-1] + 'ies'
    return word if word.endswith('s') else word + 's'


def _singular(plural: str) -> str:
    if plural.endswith('ies'):
        return plural[:-3] + 'y'
    if plural.endswith('s') and not plural.endswith(('ss', '_s')):
        return plural[:-1]
    return plural


def axis(values, plural: str, unit: str = '',
         singular: Optional[str] = None) -> str:
    """An axis: ``'depth 20 m'``, ``'depths [10, 50] m'`` or
    ``'12 depths 5–95 m'``."""
    arr = np.atleast_1d(np.asarray(values)).ravel()
    tail = f" {unit}" if unit else ''
    if arr.size == 0:
        return f"no {plural}"
    if arr.size == 1:
        return f"{singular or _singular(plural)} {num(arr[0])}{tail}"
    if arr.size <= SHOWN_VALUES:
        return f"{plural} [{', '.join(num(v) for v in arr)}]{tail}"
    return f"{arr.size} {plural} {extent(arr)}{tail}"


def extent(values, unit: str = '') -> str:
    """``'1500 m/s'`` for one value, else ``'1495–1520 m/s'``."""
    arr = np.asarray(values).real
    lo, hi = np.min(arr), np.max(arr)
    span = num(lo) if lo == hi else f"{num(lo)}–{num(hi)}"
    return f"{span} {unit}" if unit else span


def array_bit(name: str, value, unit: str = '') -> str:
    """A data array: a 1-D real one by its count and extent (or values,
    when few), any other by its shape (``'power 129×63 Pa²/Hz'``)."""
    arr = np.asarray(value)
    tail = f" {unit}" if unit else ''
    if arr.ndim == 0:
        return f"{name}={qty(arr.item(), unit)}"
    if arr.ndim == 1 and arr.dtype.kind in 'biuf':
        return axis(arr, name, unit, singular=name)
    kind = ' complex' if arr.dtype.kind == 'c' else ''
    return f"{name} {'×'.join(str(n) for n in arr.shape)}{kind}{tail}"


def value_bit(name: str, value, unit: str = '') -> Optional[str]:
    """One field of a record: ``None`` (left out) for ``None``, an array by
    :func:`array_bit`, a number with its unit, a string bare, a short
    sequence by its values, a longer one by its count, and any other
    object by its class name (never its constructor)."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return array_bit(name, value, unit)
    if isinstance(value, (bool, np.bool_)):
        return f"{name}={bool(value)}"
    if isinstance(value, (int, float, complex, np.number)):
        return f"{name}={qty(value, unit)}"
    if isinstance(value, str):
        return f"{name}={getattr(value, 'value', value)}"
    if isinstance(value, Enum):
        return f"{name}={value.value}"
    if type(value).__name__ == 'DataProvenance':
        return f"{name}={value.source.id}"
    if type(value).__name__ == 'DataSource':
        return f"{name}={value.id}"
    if isinstance(value, (set, frozenset)):
        value = tuple(sorted(value, key=str))
    if isinstance(value, Mapping):
        return f"{name} {len(value)} entries" if len(value) else None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        if len(value) > SHOWN_VALUES:
            return f"{len(value)} {name}"
        if all(isinstance(v, (int, float, np.number)) for v in value):
            return axis(value, name, unit, singular=name)
        return f"{name} [{', '.join(_word(v) for v in value)}]"
    return f"{name}={type(value).__name__}"


def _word(value) -> str:
    """A list element in repr words: an enum member by its value, a number
    in ``:g`` form, an object by its class name."""
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, (str, int, float, np.number)):
        return value if isinstance(value, str) else num(value)
    return type(value).__name__


def knob(value) -> str:
    """A constructor value as a model or tool repr shows it: a string
    quoted, a number in ``:g`` form, a short list by its values, a longer
    array by its count and extent, an object by its class name."""
    if isinstance(value, Enum):
        return repr(value.value)
    if isinstance(value, (str, bool, np.bool_)) or value is None:
        return repr(value if not isinstance(value, np.bool_) else bool(value))
    if isinstance(value, (int, float, complex, np.number)):
        return num(value)
    if hasattr(value, '__fspath__'):
        return repr(str(value))
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=object if not isinstance(
            value, np.ndarray) else None)
        if arr.ndim == 1 and len(arr) <= SHOWN_VALUES and all(
                isinstance(v, (int, float, np.number, str)) for v in arr):
            return f"[{', '.join(knob(v) for v in arr)}]"
        if (isinstance(value, np.ndarray) and value.ndim == 1
                and value.dtype.kind in 'biuf'):
            return f"[{len(value)} values {extent(value)}]"
        return f"[{len(value)} values]"
    if isinstance(value, Mapping):
        return f"{{{len(value)} entries}}"
    return type(value).__name__


def settings_repr(obj) -> str:
    """``ClassName(name=value, ...)`` over the constructor parameters of
    ``obj`` whose value, read back as ``obj.<name>``, differs from the
    parameter's default (a required parameter is always shown)."""
    bits = []
    for name, param in inspect.signature(type(obj).__init__).parameters.items():
        if name == 'self' or param.kind in (param.VAR_POSITIONAL,
                                            param.VAR_KEYWORD):
            continue
        if not hasattr(obj, name):
            continue
        value = getattr(obj, name)
        if param.default is not param.empty and _equal(value, param.default):
            continue
        bits.append(f"{name}={knob(value)}")
    return build(type(obj).__name__, bits)


class SettingsRepr:
    """Mixin: the :func:`settings_repr` of a configured tool (a modulator,
    an equaliser, a transceiver)."""

    __slots__ = ()

    def __repr__(self) -> str:
        return settings_repr(self)


def build(name: str, bits: Iterable[Optional[str]]) -> str:
    """``'Name(bit, bit)'`` from the bits that are not ``None`` or empty."""
    return f"{name}({', '.join(b for b in bits if b)})"


def fields_repr(obj, fields: Optional[Sequence[str]] = None,
                units: Optional[Mapping[str, Optional[str]]] = None) -> str:
    """The one-line repr of a record from its fields: each of ``fields``
    (default every dataclass or namedtuple field) by :func:`value_bit`, in
    the unit ``units`` gives it or its name's suffix states. A dataclass
    field still at its declared default is left out."""
    if fields is None:
        fields = (obj._fields if hasattr(obj, '_fields') else
                  [f.name for f in dataclasses.fields(obj) if f.repr])
    defaults = ({f.name: f.default for f in dataclasses.fields(obj)
                 if f.default is not dataclasses.MISSING}
                if dataclasses.is_dataclass(obj) else {})
    units = dict(units or {})
    bits = [value_bit(name, getattr(obj, name),
                      units.get(name) or unit_of(name))
            for name in fields
            if not (name in defaults
                    and _equal(getattr(obj, name), defaults[name]))]
    return build(type(obj).__name__, bits)


class FieldsRepr:
    """Mixin: the :func:`fields_repr` of a dataclass record, which every
    subclass keeps. ``@dataclass`` writes no ``__repr__`` over one the class
    body already holds, so each subclass is handed the nearest inherited
    one as it is created, before its decorator runs. A record names the
    fields its repr shows in ``_REPR_FIELDS`` (``None``: every field) and
    the unit of a field whose name states none in ``_REPR_UNITS``."""

    __slots__ = ()

    _REPR_FIELDS: Optional[Sequence[str]] = None
    _REPR_UNITS: Mapping[str, str] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if '__repr__' not in cls.__dict__:
            cls.__repr__ = cls.__repr__

    def __repr__(self) -> str:
        return fields_repr(self, self._REPR_FIELDS, self._REPR_UNITS)


def _equal(a: Any, b: Any) -> bool:
    if a is b:
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return False
    try:
        return bool(a == b)
    except (TypeError, ValueError):
        return False
