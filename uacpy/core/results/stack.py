"""The :class:`ResultStack`: the sparse results of one run (rays,
arrivals), one per value of a coordinate."""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any, List, Mapping, Union

from uacpy.core._export import (_resolve_class, saved_class_path,
                                require_extra)
from uacpy.core._repr import build, count
from uacpy.core.exceptions import ConfigurationError
from uacpy.core._plotting import plotter
from uacpy.core._grid import nearest_index_on_axis
from uacpy.core.results._base import Result, _integer_index, coordinate_axis
from uacpy.core.results.field import Field, _AXIS_IDENTITY_ATTR


class ResultStack(Result):
    """Results that share no grid, one per value of a coordinate.

    A run over a ``Source`` with several depths returns one :class:`Field`
    in every mode that returns a field (TL, ``BROADBAND`` H(f),
    ``TIME_SERIES`` traces): the depths are its leading ``source_depth``
    axis and :meth:`Field.superpose` sums them. The sparse results cannot
    be one array — the :class:`Rays` of a ``RAYS`` / ``EIGENRAYS`` run and
    the :class:`Arrivals` of an ``ARRIVALS`` run hold a different number of
    paths per source — so a multi-depth run of those modes returns a
    ``ResultStack``: one slab per source depth.
    :meth:`ParallelResult.stack() <uacpy.parallel.ParallelResult.stack>`
    bundles the non-Field results of a sweep the same way. A Field is never
    a slab.

    Every slab has the same concrete type, ``model`` and ``backend``, and
    the same identity along every axis *except* the stacking one. The stack
    is a :class:`Result`: its identity is its first slab's (the stacking
    axis excepted), and it saves and loads as every result does —
    :meth:`to_dict` / :meth:`from_dict` (npz), :meth:`to_netcdf` /
    :meth:`~Result.from_netcdf` (each slab in its own NetCDF group), and
    :meth:`~uacpy.core._export.Exportable.to_dataframe` (the slabs' tables
    with the stacking coordinate as a column).

    Construction
    ------------
    ``ResultStack(slabs, coordinate, coordinate_name='source_depth')``

    Access
    ------
    ``stack[i]``                              i-th slab
    ``for c, slab in stack: …``               iterate ``(coordinate, slab)`` pairs
    ``stack.at(<coordinate_name>=value)``     nearest-label lookup
    ``stack.isel(<coordinate_name>=index)``   positional lookup
    ``len(stack)``                            number of slabs
    """

    def __init__(
        self,
        slabs: List[Result],
        coordinate: Union[List[float], np.ndarray],
        *,
        coordinate_name: str = 'source_depth',
    ):
        slabs = list(slabs)
        if len(slabs) == 0:
            raise ConfigurationError("ResultStack: requires at least one slab.")
        coord = np.atleast_1d(np.asarray(coordinate, dtype=float))
        if coord.size != len(slabs):
            raise ConfigurationError(
                f"ResultStack: coordinate length ({coord.size}) does not "
                f"match number of slabs ({len(slabs)})"
            )
        not_results = sorted({type(s).__name__ for s in slabs
                              if not isinstance(s, Result)})
        if not_results:
            raise ConfigurationError(
                f"ResultStack: every slab must be a Result; got "
                f"{not_results}.")
        if any(isinstance(s, Field) for s in slabs):
            raise ConfigurationError(
                "ResultStack: a Field is not a slab. The fields of several "
                "sources (or of any sweep on one grid) are one Field with a "
                "leading axis over them.",
                remediation=(
                    "Take the one Field a multi-depth run returns "
                    "(Field.superpose() sums it), or join separately "
                    "computed fields into it: stack_fields(fields, depths) "
                    "(coordinate_name= names another axis)."))
        types = {type(s) for s in slabs}
        if len(types) != 1:
            raise ConfigurationError(
                f"ResultStack: every slab must have the same concrete "
                f"type; got {sorted(t.__name__ for t in types)}."
            )

        varying_attr = _AXIS_IDENTITY_ATTR.get(str(coordinate_name))
        shared_attrs = ['model', 'backend']
        for attr in ('frequencies', 'source_depths'):
            if attr != varying_attr:
                shared_attrs.append(attr)

        first = slabs[0]

        def _equal(a, b):
            if a is None or b is None:
                return a is None and b is None
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                a = np.asarray(a)
                b = np.asarray(b)
                return a.shape == b.shape and np.array_equal(a, b)
            return bool(a == b)

        for attr in shared_attrs:
            ref = getattr(first, attr, None)
            for i, s in enumerate(slabs[1:], start=1):
                val = getattr(s, attr, None)
                if not _equal(ref, val):
                    # The two attributes a stack may vary along name their
                    # own stacking coordinate, so a sweep over one of them
                    # is told which coordinate_name admits it.
                    axis = {v: k for k, v in
                            _AXIS_IDENTITY_ATTR.items()}.get(attr)
                    raise ConfigurationError(
                        f"ResultStack: slabs[0].{attr}={ref!r} but "
                        f"slabs[{i}].{attr}={val!r} — every slab must "
                        f"share the same {attr} (stacking axis is "
                        f"{coordinate_name!r})",
                        remediation=(
                            "For a source-depth sweep, stack along "
                            "coordinate_name='source_depth', or pass "
                            "Source(depths=[...]) to one run."
                            if axis == 'source_depth' else
                            f"For a sweep over {attr}, stack along "
                            f"coordinate_name={axis!r}."
                            if axis is not None else
                            f"Stack only results that share {attr}; a "
                            f"result with a different {attr} goes in a "
                            f"separate ResultStack."),
                    )

        self.slabs: List[Result] = slabs
        self.coordinate: np.ndarray = coord
        self.coordinate_name: str = str(coordinate_name)

    @classmethod
    def from_slabs(cls, slabs, coordinate, *,
                   coordinate_name: str = 'source_depth'):
        """The lone slab when ``slabs`` holds one, else a
        :class:`ResultStack` of them along ``coordinate``.

        The one rule for a sparse result that may vary along an axis: a
        reader or a run that produced a single slab returns it as it is, and
        several are stacked. An empty ``slabs`` is refused as the
        constructor refuses it.

        Parameters
        ----------
        slabs : sequence of Result
            The slabs, at least one; none a :class:`Field`.
        coordinate : array_like
            The stacking label of each slab.
        coordinate_name : str, optional
            The stacking axis. Default ``'source_depth'``.
        """
        slabs = list(slabs)
        if len(slabs) == 1:
            return slabs[0]
        return cls(slabs, coordinate, coordinate_name=coordinate_name)

    @property
    def slab_type(self) -> type:
        return type(self.slabs[0])

    @property
    def n_slabs(self) -> int:
        return int(self.coordinate.size)

    # ── identity: the first slab's, the stacking axis excepted ─────────────

    @property
    def model(self) -> str:
        return self.slabs[0].model

    @property
    def backend(self) -> str:
        return self.slabs[0].backend

    @property
    def model_source(self):
        """Engine provenance of the first slab — what the plotters read for the
        model-credit footnote."""
        return self.slabs[0].model_source

    @property
    def phase_reference(self) -> Optional[str]:
        return self.slabs[0].phase_reference

    @property
    def run_mode(self) -> Optional[str]:
        return self.slabs[0].run_mode

    @property
    def run_settings(self):
        """The settings the run that produced the stack used (every slab
        carries the same record)."""
        return self.slabs[0].run_settings

    @property
    def frequencies(self) -> Optional[np.ndarray]:
        """Frequencies (Hz) the stack covers: the stacking coordinate when
        stacking by frequency, else the value every slab agrees on."""
        return self._identity_axis('frequencies')

    @property
    def source_depths(self) -> np.ndarray:
        """Source depths (m) the stack covers: the stacking coordinate when
        stacking by source depth, else the value every slab agrees on."""
        return self._identity_axis('source_depths')

    @property
    def source_level_dB(self) -> Optional[float]:
        """The drive level (dB re 1 µPa at 1 m) of the ``Source`` that
        produced the stack, as every slab records it."""
        return self.slabs[0].source_level_dB

    @property
    def source_weights(self) -> Optional[np.ndarray]:
        """The ``Source`` weights, one per source depth, every slab records,
        or ``None``. A sparse result has no sum to apply them to."""
        return self.slabs[0].source_weights

    def _identity_axis(self, attr: str):
        if _AXIS_IDENTITY_ATTR.get(self.coordinate_name) == attr:
            return self.coordinate.copy()
        return getattr(self.slabs[0], attr)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Metadata of the first slab, as a copy. Slabs are not required to
        agree on metadata — read a specific slab's dict via ``stack[i]``."""
        return dict(self.slabs[0].metadata)

    @property
    def components(self) -> Mapping[str, Result]:
        """The components of the first slab (:attr:`Result.components`):
        the results a run's slabs were built from, which one run puts on
        every slab."""
        return self.slabs[0].components

    # ── access ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.n_slabs

    def __getitem__(self, index: int) -> Result:
        return self.slabs[_integer_index(index, "ResultStack[index]")]

    def __iter__(self):
        for c, slab in zip(self.coordinate, self.slabs):
            yield float(c), slab

    def at(self, **kwargs) -> Result:
        """Select the slab nearest a value on the stacking axis.

        Pass exactly the stacking-axis keyword (``<coordinate_name>=<value>``);
        returns the slab whose coordinate is closest to ``value``. The value
        must be a finite scalar, the same label contract :meth:`Field.at`
        applies.
        """
        if len(kwargs) != 1 or self.coordinate_name not in kwargs:
            raise ConfigurationError(
                f"ResultStack.at(): pass exactly the stacking-axis "
                f"keyword ({self.coordinate_name}=<value>); got "
                f"{list(kwargs)}."
            )
        idx = nearest_index_on_axis(
            self.coordinate, kwargs[self.coordinate_name],
            self.coordinate_name, who="ResultStack.at",
            several=", or index positions with isel()")
        return self.slabs[idx]

    def isel(self, **kwargs) -> Result:
        """Select a slab by integer position on the stacking axis.

        Pass exactly the stacking-axis keyword (``<coordinate_name>=<index>``);
        the positional counterpart of :meth:`at` (and of ``stack[index]``),
        mirroring :meth:`Field.isel`, which also refuses a non-integer index.
        """
        if len(kwargs) != 1 or self.coordinate_name not in kwargs:
            raise ConfigurationError(
                f"ResultStack.isel(): pass exactly the stacking-axis "
                f"keyword ({self.coordinate_name}=<index>); got {list(kwargs)}."
            )
        return self.slabs[_integer_index(
            kwargs[self.coordinate_name],
            f"ResultStack.isel: {self.coordinate_name}")]

    # ── export ─────────────────────────────────────────────────────────────

    def values(self, name: Optional[str] = None) -> np.ndarray:
        """Refused: the slabs share no grid, so the stack holds no payload
        of its own; read a slab's (``stack[i].values()``).

        Parameters
        ----------
        name : str, optional
            A payload name, as :meth:`Exportable.values` takes it.

        Raises
        ------
        ConfigurationError
            Always.
        """
        raise ConfigurationError(
            f"ResultStack.values: a stack holds one "
            f"{self.slab_type.__name__} per {self.coordinate_name}, which "
            f"share no grid, so it has no array of its own.",
            remediation="Read one slab's: stack[i].values().")

    def to_xarray(self):
        """Refused: the slabs share no grid, so they are not one xarray
        object. :meth:`to_netcdf` writes them, each in its own group, and
        ``stack[i].to_xarray()`` is one slab.

        Raises
        ------
        ConfigurationError
            Always.
        """
        raise ConfigurationError(
            f"ResultStack.to_xarray: a stack holds one "
            f"{self.slab_type.__name__} per {self.coordinate_name}, which "
            f"share no grid, so it is not one xarray object.",
            remediation="Read one slab (stack[i].to_xarray()); to_netcdf() "
                        "writes every slab and from_netcdf() reads the "
                        "stack back; to_dataframe() lists every slab's "
                        "rows.")

    @classmethod
    def from_xarray(cls, obj):
        """Refused: a stack is not one xarray object; :meth:`from_netcdf`
        reads the file :meth:`to_netcdf` wrote.

        Parameters
        ----------
        obj : xarray.Dataset or xarray.DataArray
            The object a caller would read.

        Raises
        ------
        ConfigurationError
            Always.
        """
        raise ConfigurationError(
            "ResultStack.from_xarray: a stack is not one xarray object.",
            remediation="Read the file to_netcdf() wrote with "
                        "ResultStack.from_netcdf(path).")

    def _table(self) -> Optional[Dict[str, np.ndarray]]:
        """Every slab's rows, in stacking order, under a first column named
        by the stacking axis that holds each row's coordinate; ``None`` when
        the slabs are not tables."""
        tables = [slab._table() for slab in self.slabs]
        if any(t is None for t in tables):
            return None
        columns = list(tables[0])
        if self.coordinate_name in columns:
            raise ConfigurationError(
                f"ResultStack.to_dataframe: the slabs already carry a "
                f"{self.coordinate_name!r} column.")
        rows = [np.full(len(next(iter(t.values()))) if t else 0, c)
                for c, t in zip(self.coordinate, tables)]
        out = {self.coordinate_name: np.concatenate(rows)}
        for column in columns:
            out[column] = np.concatenate([np.asarray(t[column])
                                          for t in tables])
        return out

    def _netcdf_parts(self, group, parts) -> None:
        """The stack's own part — its coordinate, named by the stacking
        axis, and the class and slab count — then each slab (with its own
        components) in the group ``slab_<i>`` below it."""
        xarray = require_extra('xarray', 'ResultStack.to_netcdf')
        root = xarray.Dataset(
            coords={self.coordinate_name: (self.coordinate_name,
                                           np.asarray(self.coordinate,
                                                      dtype=float))},
            attrs={'uacpy_class': saved_class_path(type(self)),
                   'uacpy_components': '',
                   'coordinate_name': self.coordinate_name,
                   'n_slabs': int(self.n_slabs)})
        parts.append((group, root))
        for i, slab in enumerate(self.slabs):
            below = f'slab_{i}' if group is None else f'{group}/slab_{i}'
            slab._netcdf_parts(below, parts)

    @classmethod
    def _from_netcdf_part(cls, xarray, path, group, obj, kwargs,
                          who) -> 'ResultStack':
        """The stack :meth:`_netcdf_parts` wrote: the coordinate from the
        part ``obj``, each slab read back from its group below it."""
        name = str(obj.attrs['coordinate_name'])
        coordinate = np.asarray(obj.coords[name].values, dtype=float)
        slabs = [Result._from_netcdf_group(
                     xarray, path,
                     f'slab_{i}' if group is None else f'{group}/slab_{i}',
                     Result, kwargs, who)
                 for i in range(int(obj.attrs['n_slabs']))]
        return cls(slabs, coordinate, coordinate_name=name)

    def to_dict(self) -> Dict[str, Any]:
        """This stack as plain data: ``coordinate`` (copied),
        ``coordinate_name`` and ``slabs``, each slab its own
        :meth:`Result.to_dict` under ``'__class__'``, its public class path,
        as :meth:`Result.to_dict` nests a component.
        ``np.savez(path, **stack.to_dict())`` stores it; :meth:`from_dict` of
        ``dict(np.load(path, allow_pickle=True))`` reads it back. That load
        unpickles, so use it only on files you wrote; :meth:`to_netcdf` /
        :meth:`~Result.from_netcdf` is the route for a file from elsewhere.

        Returns
        -------
        dict
            ``'__class__'``, ``'coordinate'``, ``'coordinate_name'`` and
            ``'slabs'``.
        """
        return {
            '__class__': saved_class_path(type(self)),
            'coordinate': np.array(self.coordinate, dtype=float),
            'coordinate_name': self.coordinate_name,
            'slabs': [{'__class__': saved_class_path(type(slab)),
                       **slab.to_dict()} for slab in self.slabs],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'ResultStack':
        """The stack :meth:`to_dict` wrote, each slab rebuilt by its own
        class's ``from_dict``.

        Parameters
        ----------
        d : mapping
            What :meth:`to_dict` returned, or the mapping ``np.load(path,
            allow_pickle=True)`` returns for a file it was saved to (load
            such a file only if you wrote it: that load unpickles).

        Returns
        -------
        ResultStack

        Raises
        ------
        ConfigurationError
            A slab class path outside uacpy, or one that names no
            :class:`Result`; a :class:`Field` slab, which no stack holds.
        """
        d = cls._unwrap_saved(
            dict(d), required=('coordinate', 'coordinate_name', 'slabs'),
            identity=False)
        saved_slabs = [dict(saved) for saved in list(d['slabs'])]
        klasses = [_resolve_class(saved.pop('__class__'), Result)
                   for saved in saved_slabs]
        if any(issubclass(klass, Field) for klass in klasses):
            raise ConfigurationError(
                "ResultStack.from_dict: the saved stack holds Field slabs, "
                "and a ResultStack holds no Field; uacpy reads only files "
                "written by this version.",
                remediation="Re-run the computation: a multi-depth run "
                            "returns one Field over source_depth, saved and "
                            "read with Field.to_dict / Field.from_dict.")
        slabs = [klass.from_dict(saved)
                 for klass, saved in zip(klasses, saved_slabs)]
        coordinate = np.asarray(d['coordinate'], dtype=float)
        name = str(d['coordinate_name'])
        return cls(slabs, coordinate, coordinate_name=name)

    def plot(self, **kwargs):
        """Refused by :func:`uacpy.plot.plot_result`, naming this call and
        the slab to pick: the slabs are drawn one at a time."""
        return plotter('_draw_result')('ResultStack.plot', self, **kwargs)

    def __repr__(self) -> str:
        return build('ResultStack', [
            count(self.n_slabs, f"{self.slab_type.__name__} slab"),
            coordinate_axis(self.coordinate_name, self.coordinate)])
