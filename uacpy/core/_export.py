"""The export protocol: one engine for every value-bearing type.

A type declares its arrays once, through two hooks:

* ``_payload()`` — ``name -> (array, dims, unit)``, the values it holds, the
  primary one first;
* ``_coords()`` — ``name -> (values, unit)`` for a dimension coordinate, or
  ``name -> (values, unit, dim)`` for an auxiliary coordinate along ``dim``;

and, where its data are a table, a third:

* ``_table()`` — ``column -> 1-D array``, one row per record.

:class:`Exportable` derives the rest from them, identically on every type:
:meth:`~Exportable.values` (read-only views), ``to_xarray`` /
``from_xarray`` (dimensions, coordinates and CF ``units``),
:meth:`~Exportable.to_netcdf` and :meth:`~Exportable.to_dataframe`. The npz
round trip is ``np.savez(path, **x.to_dict())`` read back with
``cls.from_dict(dict(np.load(path, allow_pickle=True)))``; there is no
separate save method.

The xarray and pandas imports are deferred to the call: both come with the
optional ``uacpy[xarray]`` extra.
"""

from __future__ import annotations

import sys
import warnings
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from uacpy.core.exceptions import ConfigurationError, FallbackWarning
from uacpy.core.provenance import DataProvenance
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core._repr import FieldsRepr

__all__ = ['Exportable', 'CarrierExport', 'read_only', 'encode_attrs',
           'decode_attrs', 'units_attrs', 'require_extra', 'FrozenList',
           'FrozenDict', 'saved_class_path']


def read_only(array) -> np.ndarray:
    """A view of ``array`` that refuses writes; the stored array is untouched.

    ``np.asarray`` of the view still reads every value, and ``.copy()``
    gives a writeable array. A write through the view raises numpy's
    ``ValueError: assignment destination is read-only``.
    """
    view = np.asarray(array).view()
    view.flags.writeable = False
    return view


class _Frozen:
    """Refuses every in-place edit of a derived view: what it holds is
    rebuilt from the object it came from on each access, so an edit would
    change nothing there and must not look as if it had."""

    __slots__ = ()

    def _refuse(self, *args, **kwargs):
        raise TypeError(
            f"this {type(self).__name__} is a read-only view, rebuilt from its "
            f"result on every access; editing it would change nothing. Copy "
            f"it (list(...), dict(...)) to edit, or build a new result.")

    __setitem__ = __delitem__ = __iadd__ = __imul__ = __ior__ = _refuse
    append = extend = insert = pop = remove = clear = sort = reverse = _refuse
    popitem = setdefault = update = _refuse


class FrozenList(_Frozen, list):
    """A list that refuses in-place edits (see :class:`_Frozen`); a copy
    (``copy.copy``, pickle) is an ordinary list."""

    __slots__ = ()

    def __reduce__(self):
        return (list, (list(self),))


class FrozenDict(_Frozen, dict):
    """A dict that refuses in-place edits (see :class:`_Frozen`); a copy
    is an ordinary dict."""

    __slots__ = ()

    def __reduce__(self):
        return (dict, (dict(self),))


def require_extra(module: str, who: str):
    """Import ``module`` (``'xarray'`` or ``'pandas'``) or raise a
    ConfigurationError naming the extra that provides it."""
    try:
        if module == 'xarray':
            import xarray
            return xarray
        import pandas
        return pandas
    except ImportError as exc:
        raise ConfigurationError(
            f"{who}: {module} is not installed.",
            remediation="pip install 'uacpy[xarray]'") from exc


def encode_attrs(entries: Mapping[str, Any], attrs: Dict[str, Any], *,
                 skip: Iterable[str] = (), who: str) -> None:
    """Write every entry of ``entries`` a NetCDF attribute can hold into
    ``attrs``.

    A string, number, bool or real numeric array goes in as itself; a
    complex number or array as a ``<key>_real`` / ``<key>_imag`` pair named
    in ``attrs['complex_attrs']``; a tuple or list of numbers as a real array
    named in ``attrs['tuple_attrs']``. :func:`decode_attrs` restores all
    three. Keys already in ``attrs``, keys in ``skip`` and ``None`` values
    are passed over. Any other entry (a dict, an array of more than one
    dimension) is left out, with a warning that names it and ``who``.
    """
    skip = set(skip)
    complex_keys, tuple_keys, dropped = [], [], []
    for key, value in entries.items():
        if key in attrs or key in skip or value is None:
            continue
        if isinstance(value, (str, bool, int, float, np.integer,
                              np.floating)):
            attrs[key] = value
            continue
        numbers = None
        if isinstance(value, (np.ndarray, list, tuple, complex,
                              np.complexfloating)):
            try:
                numbers = np.asarray(value)
            except ValueError:          # a ragged list
                numbers = None
        if numbers is None or numbers.dtype.kind not in 'biufc' \
                or numbers.ndim > 1:
            dropped.append(key)
        elif numbers.dtype.kind == 'c':
            attrs[f'{key}_real'] = numbers.real.astype(float)
            attrs[f'{key}_imag'] = numbers.imag.astype(float)
            complex_keys.append(key)
        elif isinstance(value, (list, tuple)):
            attrs[key] = numbers.astype(float)
            tuple_keys.append(key)
        else:
            attrs[key] = value
    if complex_keys:
        attrs['complex_attrs'] = ','.join(complex_keys)
    if tuple_keys:
        attrs['tuple_attrs'] = ','.join(tuple_keys)
    if dropped:
        warnings.warn(
            f"{who}: metadata {sorted(dropped)} cannot be held "
            f"in a NetCDF attribute and is left out of the xarray object; "
            f"read it from the {who.split('.')[0]}'s metadata before writing.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)


def _attr_names(listing) -> List[str]:
    """The keys a ``complex_attrs`` / ``tuple_attrs`` attr names."""
    return [name for name in str(listing or '').split(',') if name]


def decode_attrs(attrs: Mapping[str, Any],
                 reserved: Iterable[str] = ()) -> Dict[str, Any]:
    """The entries :func:`encode_attrs` wrote, back as a plain dict: every
    attr outside ``reserved`` and the two listing attrs, with the complex
    pairs joined and the tuple entries made tuples again."""
    reserved = set(reserved) | {'complex_attrs', 'tuple_attrs'}
    out = {k: v for k, v in attrs.items() if k not in reserved}
    restore_bool_attrs(out)
    for key in _attr_names(attrs.get('complex_attrs')):
        real = out.pop(f'{key}_real')
        imag = out.pop(f'{key}_imag')
        value = np.asarray(real, dtype=float) + 1j * np.asarray(
            imag, dtype=float)
        out[key] = complex(value) if value.ndim == 0 else value
    for key in _attr_names(attrs.get('tuple_attrs')):
        out[key] = tuple(float(x) for x in np.atleast_1d(out[key]))
    return out


def units_attrs(unit: Optional[str]) -> Dict[str, str]:
    """``{'units': unit}``, CF's attribute, or ``{}`` for a unitless entry."""
    return {'units': unit} if unit else {}


class Exportable:
    """The derived half of the export protocol (see the module docstring).

    A subclass implements ``_payload`` and, where it has axes, ``_coords``;
    a tabular one implements ``_table``. ``to_xarray`` builds an
    ``xarray.Dataset`` with one data variable per payload entry; a type that
    exports a single ``DataArray`` overrides it. ``_export_attrs`` and
    ``_from_export`` connect the generic xarray round trip to the type's
    identity and constructor.
    """

    def _payload(self) -> Dict[str, Tuple[np.ndarray, Tuple[str, ...], str]]:
        raise NotImplementedError

    def _coords(self) -> Dict[str, Tuple]:
        return {}

    def _table(self) -> Optional[Dict[str, np.ndarray]]:
        return None

    def _export_attrs(self) -> Dict[str, Any]:
        return {}

    # ── values ────────────────────────────────────────────────────────

    def values(self, name: Optional[str] = None) -> np.ndarray:
        """A read-only view of the primary payload, or of the one named.

        The payload names are those ``to_dict`` writes; a write through the
        view raises, so the stored array cannot be changed by accident.
        ``.copy()`` gives a writeable array.
        """
        payload = self._payload()
        if name is None:
            name = next(iter(payload))
        elif name not in payload:
            raise ConfigurationError(
                f"{type(self).__name__}.values: no payload {name!r}; this "
                f"type holds {list(payload)}.")
        return read_only(payload[name][0])

    # ── tabular export ────────────────────────────────────────────────

    def to_dataframe(self):
        """One row per record as a ``pandas.DataFrame`` (optional extra
        ``uacpy[xarray]``), for the types whose data are a table.

        Raises
        ------
        ConfigurationError
            A type whose data are gridded rather than tabular: its labelled
            form is ``to_xarray``.
        """
        who = f"{type(self).__name__}.to_dataframe"
        table = self._table()
        if table is None:
            raise ConfigurationError(
                f"{who}: a {type(self).__name__} is gridded, not a table.",
                remediation="Use to_xarray() for the labelled form; "
                            "to_xarray().to_dataframe() flattens it.")
        pandas = require_extra('pandas', who)
        return pandas.DataFrame(table)

    # ── labelled export ───────────────────────────────────────────────

    def to_xarray(self):
        """This object as an ``xarray.Dataset`` (optional extra
        ``uacpy[xarray]``): one data variable per payload entry on its named
        dimensions, every coordinate on its axis, and each carrying its unit
        as the CF ``units`` attribute. ``attrs`` hold the identity and the
        metadata a NetCDF attribute can hold (:func:`encode_attrs`).
        :meth:`from_xarray` reads it back."""
        xarray = require_extra('xarray', f"{type(self).__name__}.to_xarray")
        data_vars = {name: (dims, np.asarray(array), units_attrs(unit))
                     for name, (array, dims, unit) in self._payload().items()}
        coords = {}
        for name, entry in self._coords().items():
            values, unit = entry[0], entry[1]
            dim = entry[2] if len(entry) > 2 else name
            coords[name] = (dim, np.asarray(values), units_attrs(unit))
        return xarray.Dataset(data_vars, coords=coords,
                              attrs=self._export_attrs())

    @classmethod
    def from_xarray(cls, obj):
        """An object of this type from the ``xarray.Dataset`` :meth:`to_xarray`
        writes, or ``xarray.open_dataset`` reads back from a NetCDF file
        (its complex values joined back, :func:`join_complex`)."""
        obj = join_complex(obj)
        arrays = {name: np.asarray(obj[name].values) for name in obj.data_vars}
        arrays.update({name: np.asarray(c.values)
                       for name, c in obj.coords.items()})
        return cls._from_export(arrays, dict(obj.attrs))

    @classmethod
    def _from_export(cls, arrays: Dict[str, np.ndarray],
                     attrs: Dict[str, Any]):
        raise NotImplementedError

    def to_netcdf(self, path, **kwargs) -> None:
        """Write :meth:`to_xarray` to a NetCDF file at ``path``
        (:func:`write_netcdf`): a complex payload or coordinate as its
        real and imaginary parts on a trailing ``_pfnc_complex``
        dimension, which every backend writes and any netCDF reader
        opens. ``kwargs`` go to xarray's ``to_netcdf``.
        """
        write_netcdf(self.to_xarray(), path,
                     who=f"{type(self).__name__}.to_netcdf", **kwargs)


def write_netcdf(obj, path, *, who: str = "write_netcdf",
                 **kwargs) -> None:
    """Write the xarray ``obj`` (a Dataset or a DataArray) to a NetCDF file
    at ``path``: what every ``to_netcdf`` of the export protocol does.

    Boolean attributes (a Field's ``coherent``) are written as ``int8``
    and listed in ``uacpy_bool_attrs`` (:func:`portable_attrs`), and
    complex values as real and imaginary parts on a trailing
    :data:`COMPLEX_DIM` (:func:`split_complex`): both what every
    backend stores and any netCDF reader opens, so no backend is
    special to a complex result. ``who`` names the caller. ``kwargs``
    go to xarray's ``to_netcdf``.
    """
    obj = split_complex(portable_attrs(obj))
    obj.to_netcdf(path, **kwargs)


def write_netcdf_groups(parts, path, *, who: str = "write_netcdf",
                        **kwargs) -> None:
    """Write several xarray objects to one NetCDF file at ``path``: each of
    ``parts`` is ``(group, obj)``, ``group`` ``None`` for the file's root
    (written first) or a ``/``-separated group path, each written as
    :func:`write_netcdf` writes one; ``who`` names the caller."""
    parts = [(group, split_complex(portable_attrs(obj)))
             for group, obj in parts]
    for i, (group, obj) in enumerate(parts):
        obj.to_netcdf(path, mode='w' if i == 0 else 'a', group=group,
                      **kwargs)


def complex_netcdf_backend():
    """The NetCDF backend that reads a file holding complex values
    natively (h5netcdf's ``invalid_netcdf``, netCDF4's compound type):
    ``'h5netcdf'`` when it can write (h5netcdf and its h5py backend are
    both installed; recent h5netcdf no longer pulls h5py in), else
    ``'netcdf4'`` when netCDF4 1.7.1 or later is, else ``None``. A file
    this package writes needs none (:func:`split_complex`)."""
    import importlib.metadata
    import importlib.util
    if (importlib.util.find_spec('h5netcdf') is not None
            and importlib.util.find_spec('h5py') is not None):
        return 'h5netcdf'
    if importlib.util.find_spec('netCDF4') is not None:
        try:
            version = importlib.metadata.version('netCDF4')
        except importlib.metadata.PackageNotFoundError:
            return None
        parts = tuple(int(p) for p in version.split('.')[:3]
                      if p.isdigit())
        if parts >= (1, 7, 1):
            return 'netcdf4'
    return None


def refuse_complex_without_backend(who: str, name: str) -> None:
    """Refuse ``name``'s natively stored complex values for want of a
    backend that reads them (:func:`complex_netcdf_backend` found
    none)."""
    raise ConfigurationError(
        f"{who}: {name!r} holds complex values, and neither h5netcdf nor "
        f"netCDF4 1.7.1 or later, the backends that read them as stored, "
        f"is installed.",
        remediation="pip install h5netcdf h5py — or install netCDF4 1.7.1 or "
                    "later, whose compound type holds complex values.")


# ── complex values in a netCDF file ────────────────────────────────────────

#: The trailing dimension a complex variable is written on: its real and
#: imaginary parts, in that order. The name is the nc-complex convention's,
#: which netCDF4 1.7 with ``auto_complex=True`` reads back as complex itself.
COMPLEX_DIM = '_pfnc_complex'


def _stacked(values) -> np.ndarray:
    """``values`` as ``(..., 2)`` real: the real part, then the imaginary."""
    values = np.asarray(values)
    return np.stack([values.real, values.imag], axis=-1)


def _joined(values) -> np.ndarray:
    """The complex array a ``(..., 2)`` real/imaginary pair holds, each part
    assigned as stored (so a NaN in one part leaves the other as it was)."""
    values = np.asarray(values)
    out = np.empty(values.shape[:-1],
                   dtype=np.result_type(values.dtype, np.complex64))
    out.real = values[..., 0]
    out.imag = values[..., 1]
    return out


def _is_split(variable) -> bool:
    return bool(variable.dims) and variable.dims[-1] == COMPLEX_DIM \
        and variable.shape[-1] == 2


def split_complex(obj):
    """``obj`` (a Dataset or a DataArray) with each complex variable and
    non-index coordinate as a real one on a trailing :data:`COMPLEX_DIM` of
    length 2, so every netCDF backend stores it and any netCDF reader opens
    it; ``obj`` itself when nothing is complex. :func:`join_complex` undoes
    it exactly."""
    if hasattr(obj, 'data_vars'):
        out = obj
        for name, variable in obj.variables.items():
            if np.iscomplexobj(variable.values):
                entry = (variable.dims + (COMPLEX_DIM,),
                         _stacked(variable.values), dict(variable.attrs))
                out = (out.assign_coords({name: entry}) if name in obj.coords
                       else out.assign({name: entry}))
        return out
    out = obj
    if np.iscomplexobj(obj.values):
        out = obj.expand_dims({COMPLEX_DIM: 2}, axis=-1).copy(
            data=_stacked(obj.values))
    for name, coord in obj.coords.items():
        if np.iscomplexobj(coord.values):
            out = out.assign_coords({name: (
                coord.dims + (COMPLEX_DIM,), _stacked(coord.values),
                dict(coord.attrs))})
    return out


def join_complex(obj):
    """``obj`` (a Dataset or a DataArray) with each variable and coordinate
    :func:`split_complex` wrote on :data:`COMPLEX_DIM` made complex again;
    ``obj`` itself when none is. What every reader of the protocol applies
    first, so a file opened with plain xarray reads back as written."""
    if hasattr(obj, 'data_vars'):
        out = obj
        for name, variable in obj.variables.items():
            if _is_split(variable):
                entry = (variable.dims[:-1], _joined(variable.values),
                         dict(variable.attrs))
                out = (out.assign_coords({name: entry}) if name in obj.coords
                       else out.assign({name: entry}))
        return out
    out = obj
    if _is_split(obj.variable):
        out = obj.isel({COMPLEX_DIM: 0}, drop=True).copy(
            data=_joined(obj.values))
    for name, coord in obj.coords.items():
        if _is_split(coord.variable):
            out = out.assign_coords({name: (
                coord.dims[:-1], _joined(coord.values), dict(coord.attrs))})
    return out


#: The attribute that names the attributes written as ``int8`` for a
#: boolean (:func:`portable_attrs`).
BOOL_ATTRS = 'uacpy_bool_attrs'


def portable_attrs(obj):
    """``obj`` (a Dataset or a DataArray) with each boolean attribute — of
    the object and of each of its variables — as an ``int8``, the names
    listed in that attrs' :data:`BOOL_ATTRS`, so every netCDF backend
    stores it (netCDF4 refuses a boolean); a shallow copy when any changes,
    ``obj`` itself otherwise. :func:`restore_bool_attrs` undoes it."""
    def convert(attrs):
        names = [k for k, v in attrs.items() if isinstance(v, (bool, np.bool_))]
        if not names:
            return None
        out = {k: (np.int8(v) if k in names else v) for k, v in attrs.items()}
        out[BOOL_ATTRS] = ','.join(names)
        return out
    variables = (obj.variables if hasattr(obj, 'variables') else obj.coords)
    if convert(obj.attrs) is None and all(convert(v.attrs) is None
                                          for v in variables.values()):
        return obj
    obj = obj.copy(deep=False)
    new = convert(obj.attrs)
    if new is not None:
        obj.attrs = new
    for name in list(variables):
        new = convert(obj[name].attrs)
        if new is not None:
            obj[name].attrs = new
    return obj


def restore_bool_attrs(attrs) -> None:
    """Make the attributes :data:`BOOL_ATTRS` names booleans again, in
    place, and drop the listing (the inverse of :func:`portable_attrs`)."""
    for key in _attr_names(attrs.pop(BOOL_ATTRS, None)):
        if key in attrs:
            attrs[key] = bool(attrs[key])


# ── carriers ───────────────────────────────────────────────────────────────


def saved_class_path(cls) -> str:
    """The ``module.qualname`` a saved record names ``cls`` by: the parent
    package of its module when that package re-exports the same class (an
    engine package's ``__init__``, the class's one public home), else the
    module that defines it. :meth:`RunSettings.from_dict` and the ``from_dict`` of a saved carrier or
    result import either."""
    parent = cls.__module__.rpartition('.')[0]
    package = sys.modules.get(parent)
    if package is not None and getattr(package, cls.__qualname__,
                                       None) is cls:
        return f"{parent}.{cls.__qualname__}"
    return f"{cls.__module__}.{cls.__qualname__}"


def _resolve_class(path: str, base: type) -> type:
    """The class a saved ``__class__`` path names, refused unless it is a
    uacpy class and a ``base``: a file names what to rebuild, and a path
    outside the package would import arbitrary code."""
    import importlib
    module, _, name = str(path).rpartition('.')
    if not module.startswith('uacpy.'):
        raise ConfigurationError(
            f"from_dict: {path!r} is not a uacpy class; a saved carrier names "
            f"one of the package's own.")
    klass = getattr(importlib.import_module(module), name, None)
    if not (isinstance(klass, type) and issubclass(klass, base)):
        raise ConfigurationError(
            f"from_dict: {path!r} is not a {base.__name__}.")
    return klass


def _encode(value):
    """One carrier field as plain data: a nested carrier as its own
    :meth:`CarrierExport.to_dict`, a provenance record by its source id, a
    date as ISO text, an enum by its value, a path as text; arrays copied,
    sequences element by element, everything else as it is."""
    import datetime
    import enum
    import pathlib
    if isinstance(value, CarrierExport):
        return value.to_dict()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, DataProvenance):
        # Every field the record defines, through its own to_dict, so a field
        # added to the record cannot be dropped here.
        fields = {k: v for k, v in value.to_dict().items() if v is not None}
        return {'__provenance__': str(fields.pop('source')), **fields}
    if isinstance(value, datetime.date):
        return {'__date__': value.isoformat()}
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, pathlib.PurePath):
        return str(value)
    if isinstance(value, (list, tuple)):
        return type(value)(_encode(v) for v in value)
    return value


def unwrap_0d(value):
    """``value``, or the object inside it when it is the 0-d object array
    ``np.load(path, allow_pickle=True)`` hands back for a saved scalar or
    mapping."""
    if isinstance(value, np.ndarray) and value.dtype == object \
            and value.ndim == 0:
        return value.item()
    return value


def _decode(value):
    """The inverse of :func:`_encode`."""
    value = unwrap_0d(value)
    if isinstance(value, dict):
        if '__class__' in value:
            return _resolve_class(value['__class__'],
                                  CarrierExport).from_dict(value)
        if '__provenance__' in value:
            fields = dict(value)
            fields['source'] = fields.pop('__provenance__')
            return DataProvenance.from_dict(fields)
        if '__date__' in value:
            import datetime
            return datetime.date.fromisoformat(value['__date__'])
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_decode(v) for v in value)
    return value


def _to_json(value):
    """Plain data (what :func:`_encode` returns) as JSON-ready values:
    an array as ``{'__array__': values, 'dtype': ...}`` (a complex one as
    ``[real, imag]`` pairs), a tuple as ``{'__tuple__': [...]}``, a numpy
    scalar as its Python value."""
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            data = np.stack([value.real, value.imag], axis=-1).tolist()
        else:
            data = value.tolist()
        return {'__array__': data, 'dtype': value.dtype.str}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {'__complex__': [value.real, value.imag]}
    if isinstance(value, dict):
        return {str(k): _to_json(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return {'__tuple__': [_to_json(v) for v in value]}
    if isinstance(value, list):
        return [_to_json(v) for v in value]
    return value


def _from_json(value):
    """The inverse of :func:`_to_json`."""
    if isinstance(value, dict):
        if '__array__' in value:
            dtype = np.dtype(value['dtype'])
            data = np.asarray(value['__array__'],
                              dtype=float if dtype.kind == 'c' else dtype)
            if dtype.kind == 'c':
                data = (data[..., 0] + 1j * data[..., 1]).astype(dtype)
            return data
        if '__tuple__' in value:
            return tuple(_from_json(v) for v in value['__tuple__'])
        if '__complex__' in value:
            return complex(*value['__complex__'])
        return {k: _from_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_json(v) for v in value]
    return value


class CarrierExport(Exportable):
    """The export protocol on a dataclass carrier.

    :meth:`to_dict` writes every constructor field (:func:`_encode`) under the
    carrier's public class path; :meth:`from_dict` rebuilds it through the
    constructor, so the result passes every check a new carrier does. A
    carrier with gridded values adds ``_payload`` / ``_coords`` for
    ``values()`` and ``to_xarray``.
    """

    def _payload(self):
        return {}

    #: For a gridded carrier: each payload / coordinate name of
    #: ``to_xarray`` -> the constructor field that holds it.
    _XARRAY_FIELDS: Dict[str, str] = {}

    #: The ``attrs`` key that holds the constructor fields ``to_xarray``
    #: does not write as variables, as JSON.
    _FIELDS_ATTR = 'uacpy_fields'

    def to_xarray(self):
        """This carrier as an ``xarray.Dataset`` (optional extra
        ``uacpy[xarray]``): its gridded values as variables on their named
        axes with CF ``units``, and every other constructor field — the
        nested carriers and the provenance included — as JSON in
        ``attrs['uacpy_fields']``, so :meth:`from_xarray` rebuilds it
        exactly."""
        import json
        dataset = super().to_xarray()
        rest = self.to_dict()
        for field in self._XARRAY_FIELDS.values():
            rest.pop(field, None)
        dataset.attrs[self._FIELDS_ATTR] = json.dumps(_to_json(rest))
        return dataset

    @classmethod
    def from_xarray(cls, obj):
        """The carrier :meth:`to_xarray` wrote, rebuilt through its
        constructor."""
        import json
        obj = join_complex(obj)
        d = _from_json(json.loads(obj.attrs[cls._FIELDS_ATTR]))
        path = d.get('__class__')
        klass = cls if path is None else _resolve_class(path, cls)
        for name, field in klass._XARRAY_FIELDS.items():
            if name in obj.variables:
                d[field] = np.asarray(obj[name].values)
        return klass.from_dict(d)

    def _constructor_fields(self) -> Dict[str, Any]:
        """The constructor arguments that rebuild this carrier as it is:
        every constructor field's current value. A carrier that stores a
        resolved default its constructor would read as an explicit choice
        overrides it."""
        import dataclasses
        return {f.name: getattr(self, f.name)
                for f in dataclasses.fields(self) if f.init}

    def to_dict(self) -> Dict[str, Any]:
        """Every constructor field as plain data, nested carriers as their
        own ``to_dict``, under ``'__class__'``, the carrier's public class
        path. ``np.savez(path, **c.to_dict())`` saves it; ``from_dict`` of
        ``dict(np.load(path, allow_pickle=True))`` reads it back."""
        return {'__class__': saved_class_path(type(self)),
                **{name: _encode(value)
                   for name, value in self._constructor_fields().items()}}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]):
        """The carrier :meth:`to_dict` wrote, rebuilt through its
        constructor; ``d`` may be the mapping ``np.load(path,
        allow_pickle=True)`` returns. ``'__class__'`` picks the class, which
        must be this one or a subclass."""
        d = {k: _decode(v) for k, v in dict(d).items()}
        path = d.pop('__class__', None)
        klass = cls if path is None else _resolve_class(path, cls)
        return klass(**d)


# ── records ────────────────────────────────────────────────────────────────


class ExportRecord(FieldsRepr, CarrierExport):
    """The export protocol on a record a reader or a fetcher returns: a
    ``@dataclass(frozen=True, eq=False)`` whose array fields, named in
    ``_ARRAY_FIELDS``, are held as read-only views of the arrays passed,
    with the one-line :class:`FieldsRepr`.

    :meth:`to_dict` / :meth:`from_dict` and :meth:`to_xarray` are the
    carriers' (:class:`CarrierExport`). A record whose data are a table
    names its columns in ``_TABLE_FIELDS``, and :meth:`to_dataframe` gives
    one row per entry.
    """

    _ARRAY_FIELDS: Tuple[str, ...] = ()
    _TABLE_FIELDS: Tuple[str, ...] = ()

    def __post_init__(self):
        self._freeze_arrays()

    def _freeze_arrays(self) -> None:
        for name in self._ARRAY_FIELDS:
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, read_only(value))

    def __setstate__(self, state):
        # numpy unpickles an array writeable; the record's stay read-only.
        for name, value in state.items():
            object.__setattr__(self, name, value)
        self._freeze_arrays()

    def _table(self):
        if not self._TABLE_FIELDS:
            return None
        return {name: np.asarray(getattr(self, name))
                for name in self._TABLE_FIELDS}
