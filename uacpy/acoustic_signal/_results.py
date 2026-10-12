"""The behaviour every signal-processing result shares.

A result is a namedtuple: its fields are the measurement, so
``frequencies, power = welch(x, fs)`` unpacks it. What the measurement
MEANS (a scaling, a method, a moveout family) rides on attributes, which
take no part in equality. :class:`ResultTuple` is the mixin that gives every
such result the same copy, pickle, plot and unit behaviour.

A package-internal module: its names are public to the package, not to
users.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from uacpy.core._export import (Exportable, decode_attrs, encode_attrs,
                                join_complex, unwrap_0d)
from uacpy.core._plotting import plotter
from uacpy.core._repr import axis, build, value_bit
from uacpy.core.exceptions import ConfigurationError

#: The spelling of the reference pressure in a dB unit: the package's
#: underwater reference, 1 µPa, by name, and any other in Pa.
_MICROPASCAL = 1e-6


def reference_text(ref: float) -> str:
    """The reference pressure ``ref`` (Pa) as a unit string: ``'1 µPa'``
    for the underwater reference, else ``'<ref> Pa'``."""
    return '1 µPa' if float(ref) == _MICROPASCAL else f'{float(ref):g} Pa'


#: The unit of a linear spectral statistic of a pressure record in Pa, by the
#: ``scaling`` that names the statistic: a density per hertz, the power in
#: a bin, or an exposure (the time integral of the squared pressure).
POWER_UNITS = {'density': 'Pa²/Hz', 'spectrum': 'Pa²', 'exposure': 'Pa²·s'}


#: The ``attrs`` key :meth:`ResultTuple.to_xarray` lists the result's fields
#: under, comma-separated: a listed field the object does not hold was
#: ``None``, so :meth:`ResultTuple.from_xarray` can tell it from a field the
#: writer never had.
RESULT_FIELDS_ATTR = 'uacpy_result_fields'


def level_unit(ref: float, scaling: str) -> str:
    """The unit of a level in dB of the statistic ``scaling`` names, re the
    reference pressure ``ref`` (Pa): ``'dB re 1 µPa²/Hz'`` for a density
    re 1 µPa."""
    suffix = {'density': '²/Hz', 'spectrum': '²', 'exposure': '²·s'}[scaling]
    return f'dB re {reference_text(ref)}{suffix}'


class ResultTuple(Exportable):
    """Mixin placed before a namedtuple base.

    A subclass declares ``_attrs``, the attribute names that ride on the
    tuple: its ``__new__`` takes each by keyword, copying and pickling pass
    them back that way, and :meth:`_replace` carries every one not named
    across. It declares :meth:`_field_units`, the unit of each field: a string (``''`` for a
    dimensionless value), or ``None`` for a value calibrated to no unit (a
    relative distribution, or a transform that carries the input's samples
    unscaled). A field in the input's own unit is written for an input in
    Pa, the package's pressure unit.
    """

    __slots__ = ()

    _attrs: Tuple[str, ...] = ()

    def _replace(self, **changes):
        """A copy with ``changes`` applied to fields or attributes; every
        attribute not named carries across (``namedtuple._replace`` would
        rebuild through ``__new__`` with the fields only and silently reset
        what the result MEANS)."""
        kept = {name: changes.pop(name, getattr(self, name))
                for name in self._attrs}
        fields = [changes.pop(name, value)
                  for name, value in zip(self._fields, self)]
        if changes:
            # TypeError, as namedtuple._replace raises for a name it has not.
            raise TypeError(f"Got unexpected field names: {list(changes)!r}.")
        return type(self)(*fields, **kept)

    def __getnewargs_ex__(self):
        # Pickle and copy go back through __new__, which takes the
        # attributes by keyword; the plain __getnewargs__ a namedtuple
        # supplies would drop every one of them.
        return tuple(self), {name: getattr(self, name) for name in self._attrs}

    def _field_units(self) -> Dict[str, Optional[str]]:
        return {name: None for name in self._fields}

    def __repr__(self) -> str:
        """One line: each coordinate field as an axis, each other field by
        its size, in its unit, then each attribute that is a number or a
        word (an array or an object attribute is left out)."""
        units = self._field_units()
        coords = self._coords()
        bits = []
        for name, value in zip(self._fields, self):
            unit = units.get(name) or ''
            if name in coords:
                bits.append(axis(value, name, unit, singular=name))
            else:
                bits.append(value_bit(name, value, unit))
        for name in self._attrs:
            value = getattr(self, name)
            if isinstance(value, (str, bool, int, float, np.number)):
                bits.append(value_bit(name, value))
        return build(type(self).__name__, bits)

    @property
    def units(self) -> Dict[str, Optional[str]]:
        """The unit of each field, by field name: a string (``''`` for a
        dimensionless value), or ``None`` for a value calibrated to no
        unit."""
        return dict(self._field_units())

    # ── the export protocol ───────────────────────────────────────────

    def _arrays(self) -> Dict[str, np.ndarray]:
        """Every numeric field as an array, in field order; a field that
        is not numeric (a label, an object) is left out."""
        out = {}
        for name, value in zip(self._fields, self):
            if value is None:
                continue
            array = np.asarray(value)
            if array.dtype.kind in 'biufc':
                out[name] = array
        return out

    def _declared_dims(self) -> Dict[str, Tuple[str, ...]]:
        """The dimensions a result states for some of its fields, used in
        place of :meth:`_axes`' length matching — for a field whose data
        comes before its axes, or whose axes a length coincidence could
        swap. Empty here."""
        return {}

    def _axis_coords(self) -> Dict[str, Tuple[np.ndarray, str]]:
        """``dim -> (values, unit)`` for an axis a result carries as an
        attribute rather than a field (a beam scan's ``ranges``). Empty
        here."""
        return {}

    def _axes(self) -> Dict[str, Tuple[str, ...]]:
        """The dimensions of each numeric field. A field the result declares
        (:meth:`_declared_dims`) takes its declared names; for any other, an
        axis takes the name of the first earlier 1-D field of its length not
        already used, else ``<field>_dim<i>``. So ``power(frequencies,
        times)`` of a spectrogram is labelled by the two fields before it. A
        1-D field that labels another field's axis lies on its own name,
        which makes it that axis's coordinate; a 1-D field labelling nothing
        keeps ``<field>_dim0``."""
        arrays = self._arrays()
        declared = self._declared_dims()
        axes, seen = {}, []
        for name, array in arrays.items():
            if len(declared.get(name, ())) == array.ndim > 0:
                axes[name] = tuple(declared[name])
                seen.append(name)
                continue
            free = [f for f in seen if arrays[f].ndim == 1]
            dims = []
            for i, n in enumerate(array.shape):
                match = next((f for f in free if arrays[f].size == n), None)
                if match is not None:
                    free.remove(match)
                    dims.append(match)
                else:
                    dims.append(f'{name}_dim{i}')
            axes[name] = tuple(dims)
            seen.append(name)
        labels = {d for dims in axes.values() for d in dims}
        for name, array in arrays.items():
            if array.ndim == 1 and name in labels:
                axes[name] = (name,)
        return axes

    def _coords(self):
        units = self._field_units()
        axes = self._axes()
        used = {d for dims in axes.values() for d in dims}
        coords = {name: (array, units.get(name) or '')
                  for name, array in self._arrays().items()
                  if array.ndim == 1 and name in used
                  and axes[name] == (name,)}
        for dim, (values, unit) in self._axis_coords().items():
            if dim in used and dim not in coords:
                coords[dim] = (np.asarray(values), unit)
        return coords

    def _payload(self):
        units = self._field_units()
        coords = self._coords()
        axes = self._axes()
        return {name: (array, axes[name], units.get(name) or '')
                for name, array in self._arrays().items()
                if name not in coords}

    def _table(self):
        """The fields as columns, for a result whose fields are all 1-D
        and of one length (band levels, a BER curve, ...); ``None``
        otherwise."""
        values = [np.asarray(v) for v in self if v is not None]
        if (len(values) != len(self) or not values
                or any(v.ndim != 1 for v in values)
                or len({v.size for v in values}) != 1):
            return None
        return {name: np.asarray(v).copy()
                for name, v in zip(self._fields, self)}

    def _export_attrs(self) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {}
        encode_attrs({name: getattr(self, name) for name in self._attrs},
                     attrs, who=f"{type(self).__name__}.to_xarray")
        return attrs

    def to_xarray(self):
        """The result as an ``xarray.Dataset`` (optional extra
        ``uacpy[xarray]``): every numeric field as a variable or a
        coordinate in its unit, the attributes in ``attrs``, and the field
        names in ``attrs['uacpy_result_fields']`` — a field held as ``None``
        is listed there and not written. :meth:`from_xarray` reads it
        back."""
        dataset = super().to_xarray()
        dataset.attrs[RESULT_FIELDS_ATTR] = ','.join(self._fields)
        return dataset

    @classmethod
    def from_xarray(cls, obj):
        """The result :meth:`to_xarray` wrote: each field from the
        variable or coordinate of its name, each attribute from ``attrs``.
        A field the object's ``uacpy_result_fields`` lists and does not hold
        is ``None`` (:meth:`to_xarray` lists a field held as ``None``); one
        it neither holds nor lists is refused."""
        obj = join_complex(obj)
        held = {str(name) for name in obj.variables}
        listed = set(str(obj.attrs.get(RESULT_FIELDS_ATTR, '')).split(','))
        cls._refuse_missing_fields(
            [name for name in cls._fields
             if name not in held and name not in listed],
            held, 'from_xarray')
        fields = [np.asarray(obj[name].values) if name in held
                  else None for name in cls._fields]
        attrs = decode_attrs(obj.attrs, reserved=(RESULT_FIELDS_ATTR,))
        return cls(*fields, **{name: attrs[name] for name in cls._attrs
                               if name in attrs})

    @classmethod
    def _refuse_missing_fields(cls, missing, held, method: str) -> None:
        """Refuse a saved result that holds none of the fields ``missing``
        (``held``: the names it holds), naming the call ``method`` of this
        class."""
        if not missing:
            return
        raise ConfigurationError(
            f"{cls.__name__}.{method}: the saved result holds no "
            f"{', '.join(repr(name) for name in missing)} field (it holds "
            f"{', '.join(repr(name) for name in sorted(held)) or 'nothing'}), "
            f"so it is not a {cls.__name__} with the fields "
            f"{', '.join(cls._fields)}.",
            remediation=(f"Pass what {cls.__name__}.to_dict() or "
                         f"to_xarray() wrote, or rename its entries to the "
                         f"fields {', '.join(cls._fields)}."))

    def to_dict(self) -> Dict[str, Any]:
        """The fields and the attributes, by name (arrays copied).
        ``np.savez(path, **r.to_dict())`` saves it; :meth:`from_dict` of
        ``dict(np.load(path, allow_pickle=True))`` reads it back."""
        out = {name: (value.copy() if isinstance(value, np.ndarray)
                      else value)
               for name, value in zip(self._fields, self)}
        out.update({name: getattr(self, name) for name in self._attrs})
        return out

    @classmethod
    def from_dict(cls, d) -> 'ResultTuple':
        """The result :meth:`to_dict` wrote; ``d`` may be the mapping
        ``np.load(path, allow_pickle=True)`` returns. A mapping that holds
        no entry for a field is refused (:meth:`to_dict` writes every field,
        ``None`` included)."""
        d = {k: unwrap_0d(v) for k, v in dict(d).items()}
        cls._refuse_missing_fields(
            [name for name in cls._fields if name not in d], d, 'from_dict')
        return cls(*(d[name] for name in cls._fields),
                   **{name: d[name] for name in cls._attrs if name in d})


class PlottedResult(ResultTuple):
    """A :class:`ResultTuple` that draws itself. A subclass declares:

    ``_plotter``
        The name of the :mod:`uacpy.visualization` function that draws it;
        :meth:`_plotter_name` may choose it from an attribute instead.
    ``_plot_fields``
        The fields handed to the plotter positionally, or ``None`` to hand
        it the whole result.
    ``_plot_defaults``
        Attribute names set as plotter keywords unless the caller gives
        them.
    """

    __slots__ = ()

    _plotter: str = ''
    _plot_fields: Optional[Tuple[str, ...]] = None
    _plot_defaults: Tuple[str, ...] = ()

    def _plotter_name(self) -> str:
        return self._plotter

    def plot(self, **kwargs):
        """Draw this result through the :mod:`uacpy.visualization` plotter
        it calls for. ``kwargs`` reach the plotter; the attributes that say
        what the result holds are passed too unless given. Returns
        ``(fig, ax)``, as every plotter in the package does."""
        draw = plotter(self._plotter_name())
        for attr in self._plot_defaults:
            kwargs.setdefault(attr, getattr(self, attr))
        if self._plot_fields is None:
            return draw(self, **kwargs)
        return draw(*(getattr(self, f) for f in self._plot_fields),
                    **kwargs)
