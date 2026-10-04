"""The :class:`ResultStack`: typed :class:`Result` slabs along one coordinate,
and the checks its sums apply."""

from __future__ import annotations

import warnings

import numpy as np
from typing import Optional, Dict, Any, List, Mapping, Tuple, Union

from uacpy.core._carrier import DeepCopyMixin
from uacpy.core._export import (_resolve_class, saved_class_path, unwrap_0d,
                                join_complex, write_netcdf)
from uacpy.core.constants import PRESSURE_FLOOR
from uacpy.core._repr import build, count
from uacpy.core.exceptions import ConfigurationError, FallbackWarning
from uacpy.core._plotting import plotter
from uacpy.core._grid import nearest_index_on_axis
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.results import quantities as _quantities
from uacpy.core.results._base import (PhaseReference, Result, _integer_index,
                                      coordinate_axis)
from uacpy.core.results.field import Field, _RESULTSTACK_VARYING_ATTR


def _check_superpose_grids(slabs) -> None:
    """Refuse slabs that are not sampled at the same points.

    Either kind of sum adds cell to cell, so this belongs to both: a
    coherent one would add pressures from different places, an incoherent
    one their intensities, and the answer would carry the first slab's axes
    whichever it was.
    """
    first = slabs[0]
    for i, slab in enumerate(slabs[1:], start=1):
        same_axes = list(slab.coords) == list(first.coords) and all(
            np.array_equal(slab.coords[k], first.coords[k])
            for k in first.coords)
        if not same_axes or slab.data.shape != first.data.shape:
            sizes = {k: (first.coords[k].size, slab.coords[k].size)
                     for k in first.coords
                     if k in slab.coords
                     and first.coords[k].size != slab.coords[k].size}
            detail = (
                f"axes {list(first.coords)} vs {list(slab.coords)}"
                if list(slab.coords) != list(first.coords) else
                f"shape {first.data.shape} vs {slab.data.shape}"
                + (f", axis lengths {sizes}" if sizes else
                   "; same lengths, different coordinate values")
            )
            raise ConfigurationError(
                f"ResultStack.superpose: slabs[{i}] is on a different "
                f"grid from slabs[0] ({detail}); a sum needs every slab "
                f"sampled at the same points. A TIME_SERIES pair gets one "
                f"time axis from run(output_duration=…)."
            )


def _check_one_quantity(slabs) -> None:
    """Refuse Field slabs that do not hold the same quantity the same way.

    Every sum and every stacked view reads the first slab's storage, kind,
    unit and phase convention for all of them: a complex pressure slab and
    its own ``to_dB()`` would add a decibel to a pascal, and a loss and a
    level would be read with one sign."""
    def signature(slab):
        reference = slab.phase_reference
        return {'storage': 'complex' if slab.is_complex else 'real',
                'kind': slab.kind, 'unit': slab.unit,
                'phase_reference': (None if reference is None
                                    else PhaseReference(reference).value)}
    first = signature(slabs[0])
    for i, slab in enumerate(slabs[1:], start=1):
        this = signature(slab)
        differ = [k for k in first if this[k] != first[k]]
        if differ:
            detail = ', '.join(f"{k} {first[k]!r} vs {this[k]!r}"
                               for k in differ)
            raise ConfigurationError(
                f"ResultStack: slabs[{i}] holds a different quantity from "
                f"slabs[0] ({detail}); a stack is one quantity stored one "
                f"way, since every sum and view reads the first slab's.",
                remediation="Convert the slabs to one form first (e.g. "
                            "to_dB() on every one), or keep them in "
                            "separate stacks.")


def _check_stack_weightable(field: 'Field', weights, *, where: str) -> None:
    """The weaker check a *stack* has to pass: refuse only what no sum of
    it could ever use.

    A stack is not weighted when it is built — :meth:`ResultStack.superpose`
    applies the weights, and the caller has not yet said which sum they
    mean. A dB-only stack cannot add coherently but adds perfectly well in
    intensity, so refusing it here would close the only route to "N mutually
    incoherent sources at these levels". What is unusable either way is a
    complex weight on data with no phase to rotate; that is refused now,
    while the rest waits for :meth:`ResultStack.superpose` to judge against
    the sum actually asked for.
    """
    w = np.atleast_1d(np.asarray(weights, dtype=np.complex128))
    if not np.any(w.imag != 0.0):
        return
    if 'time' in field.coords or not field.is_complex:
        _check_field_weightable(field, w, where=where)
    elif field.phase_reference is None:
        _check_field_weightable(field, w, where=where)


def _dB_weight_refusal(unit, *, where: str) -> ConfigurationError:
    """The refusal of a weight on a real frequency-domain field of
    ``unit`` (dB): its phase is gone, so no weight scales it."""
    return ConfigurationError(
        f"{where}: the field is real {unit!r} values, not "
        f"complex pressure, so its phase is gone and a weight cannot "
        f"scale it (multiplying dB is not a coherent sum). Weight the "
        f"complex field the run returned (before to_dB(), or a mode "
        f"that keeps phase — phase_reference is not None); for the "
        f"level of one source at magnitude |w|, subtract "
        f"20*log10(|w|) from the dB view yourself."
    )


def _check_field_weightable(field: 'Field', weights, *, where: str) -> None:
    """Raise :class:`ConfigurationError` unless ``weights`` can scale
    ``field``. The one rule behind the n = 1 weight
    :meth:`PropagationModel.run` applies and the n-slab sum
    :meth:`ResultStack.superpose` forms:

    * a real frequency-domain field (dB) has lost its phase and takes no
      weight at all — multiplying decibels is not a scaling;
    * a real time-domain trace takes a real weight (a sign flip is -1),
      never a complex one;
    * complex data takes a complex weight only when it carries a
      ``phase_reference``. Complex data without one — a field built by
      hand, say — states no phase convention, so its phase cannot be told
      from an artefact of how it was stored; rotating it would record a
      source phase the field cannot carry. A real weight still scales such
      a field's level.
    """
    w = np.atleast_1d(np.asarray(weights, dtype=np.complex128))
    time_domain = 'time' in field.coords
    if not time_domain and not field.is_complex:
        raise _dB_weight_refusal(field.unit, where=where)
    if not field.is_complex and np.any(w.imag != 0.0):
        raise ConfigurationError(
            f"{where}: the data are real time-domain traces, which a "
            f"complex weight cannot scale; give a real weight (a sign flip "
            f"is -1), or run BROADBAND and weight the transfer function "
            f"before synthesising."
        )
    if (field.is_complex and field.phase_reference is None
            and np.any(w.imag != 0.0)):
        raise ConfigurationError(
            f"{where}: the data are complex but carry no phase reference, "
            f"so nothing says their phase is a propagation phase rather "
            f"than an artefact of how they were stored. Rotating it by a "
            f"complex weight would record a source phase the field cannot "
            f"carry; give a real weight to scale its level, or build the "
            f"field with a phase_reference (every coherent model mode "
            f"stamps one)."
        )


def _weighted_slab_sum(slabs, weights, *, where: str) -> Tuple[np.ndarray,
                                                              np.ndarray]:
    """``(Σ wᵢ·dataᵢ, w)``: the coherent weighted sum of ``slabs``' data,
    and the weights as applied — real for real data, complex otherwise —
    after :func:`_check_field_weightable` on the first slab. The one sum
    behind :meth:`ResultStack.superpose` and the n = 1 weight
    :meth:`PropagationModel.run` applies to a one-depth ``Source``."""
    first = slabs[0]
    w = np.atleast_1d(np.asarray(weights, dtype=np.complex128))
    _check_field_weightable(first, w, where=where)
    if not first.is_complex:
        w = w.real
    total = np.zeros(first.data.shape,
                     dtype=np.result_type(first.data.dtype, w.dtype))
    for wi, slab in zip(w, slabs):
        total += wi * slab.data
    return total, w


class ResultStack(DeepCopyMixin):
    """Stack of typed :class:`Result` slabs along one coordinate.

    Bundles a list of slabs together with the coordinate vector along
    which they are stacked. The coordinate can be a :class:`Result`
    field (``source_depth``, ``frequency``) or an external parameter
    the user varied. Every slab carries the same concrete type,
    ``model``, and ``backend``, and the same identification along
    every axis *except* the stacking axis.

    This is what a multi-source run returns, for gridded (``Field``) and
    sparse (``Rays`` / ``Arrivals``) results alike. Consumers that need one
    dense array — matched-field processing, say — accept either this stack or
    a single :class:`Field` carrying the varying axis in ``coords`` (e.g.
    ``coords={'source_depth', 'depth', 'range'}``).

    Construction
    ------------
    ``ResultStack(slabs, coordinate, coordinate_name='source_depth')``

    Access
    ------
    ``stack[i]``                              i-th slab
    ``for c, slab in stack: …``               iterate ``(coordinate, slab)`` pairs
    ``stack.at(<coordinate_name>=value)``     nearest-label lookup
    ``len(stack)``                            number of slabs
    ``stack.superpose(weights)``              coherent sum ``Σ wᵢ·pᵢ`` → Field
    """

    def __init__(
        self,
        slabs: List[Result],
        coordinate: Union[List[float], np.ndarray],
        *,
        coordinate_name: str = 'source_depth',
    ):
        if len(slabs) == 0:
            raise ConfigurationError("ResultStack: requires at least one slab.")
        coord = np.atleast_1d(np.asarray(coordinate, dtype=float))
        if coord.size != len(slabs):
            raise ConfigurationError(
                f"ResultStack: coordinate length ({coord.size}) does not "
                f"match number of slabs ({len(slabs)})"
            )
        types = {type(s) for s in slabs}
        if len(types) != 1:
            raise ConfigurationError(
                f"ResultStack: every slab must have the same concrete "
                f"type; got {sorted(t.__name__ for t in types)}."
            )

        varying_attr = _RESULTSTACK_VARYING_ATTR.get(str(coordinate_name))
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
                            _RESULTSTACK_VARYING_ATTR.items()}.get(attr)
                    raise ConfigurationError(
                        f"ResultStack: slabs[0].{attr}={ref!r} but "
                        f"slabs[{i}].{attr}={val!r} — every slab must "
                        f"share the same {attr} (stacking axis is "
                        f"{coordinate_name!r})",
                        remediation=(
                            "For a source-depth sweep, stack along "
                            "coordinate_name='source_depth', or pass "
                            "Source(depths=[...]) to one run — it returns a "
                            "ResultStack over source_depth."
                            if axis == 'source_depth' else
                            f"For a sweep over {attr}, stack along "
                            f"coordinate_name={axis!r}."
                            if axis is not None else
                            f"Stack only results that share {attr}; a "
                            f"result with a different {attr} goes in a "
                            f"separate ResultStack."),
                    )

        if isinstance(first, Field):
            _check_one_quantity(slabs)

        self.slabs: List[Result] = list(slabs)
        self.coordinate: np.ndarray = coord
        self.coordinate_name: str = str(coordinate_name)

    @classmethod
    def from_slabs(cls, slabs, coordinate, *,
                   coordinate_name: str = 'source_depth'):
        """The lone slab when ``slabs`` holds one, else a
        :class:`ResultStack` of them along ``coordinate``.

        The one rule for a result that may vary along an axis: a reader or
        a run that produced a single slab returns it as it is, and several
        are stacked. An empty ``slabs`` is refused as the constructor
        refuses it.

        Parameters
        ----------
        slabs : sequence of Result
            The slabs, at least one.
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
        """The ``Source`` weights, one per source depth, that
        :meth:`superpose` applies by default (every slab records them), or
        ``None``."""
        return self.slabs[0].source_weights

    def _identity_axis(self, attr: str):
        if _RESULTSTACK_VARYING_ATTR.get(self.coordinate_name) == attr:
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
            self.coordinate_name)
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

    @property
    def dB(self) -> np.ndarray:
        """Every slab's dB view stacked along the coordinate axis — shape
        ``(n_slabs, *slab.dB.shape)`` — so generic code can read ``result.dB``
        whether one or many source depths were requested. Requires Field slabs.
        """
        first = self.slabs[0]
        if not isinstance(first, Field):
            raise ConfigurationError(
                f"ResultStack.dB: slabs are {self.slab_type.__name__}, not "
                f"Field — no dB view. Pick a slab with stack[i] or "
                f"stack.at({self.coordinate_name}=...)."
            )
        self._warn_if_weights_unapplied('dB')
        if 'time' in first.coords:
            raise ConfigurationError(
                "ResultStack.dB: time-domain slabs are linear pressure, not "
                "a level; read the samples via stack[i].data, or recover a "
                "complex narrowband field first with stack[i].extract_tone(f)."
            )
        # Complex slabs derive their dB view (unit 'Pa', -20*log10|data|);
        # a real slab's data IS its dB view only when its unit says so, and
        # Field.dB refuses any other unit — pre-check it here so the stack
        # raises the same typed error as the time-domain case above.
        if not first.is_complex and first.unit != 'dB':
            raise ConfigurationError(
                f"ResultStack.dB: slabs are in {first.unit!r}, not dB, so "
                f"their values are not a level; read them via stack[i].data. "
                f"to_dB() returns a real slab unchanged, so if a dB view of "
                f"{first.unit!r} is meaningful, take "
                f"20*np.log10(np.abs(stack[i].data)) yourself and tag the "
                f"result unit='dB'."
            )
        return np.stack([s.dB for s in self.slabs], axis=0)

    @property
    def p(self) -> np.ndarray:
        """Every slab's complex pressure stacked along the coordinate axis —
        shape ``(n_slabs, *slab.data.shape)`` — :attr:`Field.p` per slab.

        Refused when the slabs are real (a dB field has discarded its phase),
        as :attr:`Field.p` refuses, but with the stack accessors'
        ConfigurationError. The per-source weights are not applied; use
        :meth:`superpose` for the weighted sum.
        """
        first = self.slabs[0]
        if not isinstance(first, Field):
            raise ConfigurationError(
                f"ResultStack.p: slabs are {self.slab_type.__name__}, not "
                f"Field — no pressure view. Pick a slab with stack[i] or "
                f"stack.at({self.coordinate_name}=...)."
            )
        if not first.is_complex:
            raise ConfigurationError(
                f"ResultStack.p: slabs are real ({first.unit!r}); complex "
                f"pressure is unavailable because the phase was discarded. "
                f"Run a complex mode (COHERENT_TL / BROADBAND) to keep it."
            )
        self._warn_if_weights_unapplied('p')
        return np.stack([s.p for s in self.slabs], axis=0)

    @property
    def tl(self) -> np.ndarray:
        """Every slab's transmission loss stacked along the coordinate
        axis — :attr:`dB` restricted to pressure slabs, mirroring
        :attr:`Field.tl` in values. The refusal type follows each class's
        own accessors: ``Field`` accessors raise :class:`AttributeError`,
        stack accessors raise ConfigurationError."""
        first = self.slabs[0]
        if isinstance(first, Field) and first.kind != 'pressure':
            raise ConfigurationError(
                f"ResultStack.tl: the slabs' kind is {first.kind!r}, not "
                f"'pressure', so their values are not a transmission loss; "
                f"their level view is stack.dB."
            )
        return self.dB

    def superpose(self, weights=None, *, coherent: bool = True) -> 'Field':
        """Coherent sum of the slabs: one :class:`Field` holding
        ``Σ wᵢ·pᵢ`` over the stack on the shared receiver grid.

        Adding the complex pressure of each source is how a multi-source
        array is driven: the engines are linear in the source amplitude, so
        the field of a weighted array is the weighted sum of the unit-source
        fields the stack holds. Every slab must therefore carry complex
        pressure (``phase_reference`` intact) or a time-domain trace — a
        dB-only slab has lost its phase and is refused.

        Parameters
        ----------
        weights : array-like, optional
            One coefficient per slab. ``None`` reads the weights the
            ``Source`` that produced the stack carried
            (:attr:`source_weights`, stamped by
            :meth:`PropagationModel.run`), else unit weights. Complex on a
            complex-pressure stack; real on a time-domain stack, whose
            traces are real samples. ``coherent=False`` uses only ``|w|``.
        coherent : bool, keyword-only
            How the sources combine, which is a statement about the sources
            and not about the arithmetic:

            ``True`` (default) adds complex pressure, ``Σ wᵢ·pᵢ`` — the
            sources are driven together with a fixed relative phase, as the
            elements of one array are.

            ``False`` adds intensity, ``√Σ|wᵢ·pᵢ|²`` — the sources are
            mutually incoherent (separate platforms, unrelated tones,
            random relative phase), so their phases carry no information and
            only ``|w|`` is read. N identical sources then give
            ``10·log10(N)`` where a coherent sum gives ``20·log10(N)``.
            The result has no phase, so it comes back the way every engine
            returns its own incoherent mode: real dB, ``phase_reference``
            cleared. A dB-only stack, which cannot add coherently, adds this
            way.

        Returns
        -------
        Field
            Same grid and identity as the slabs, with ``source_depths``
            widened to the stacking coordinate and
            ``metadata['superposed_sources']`` recording the ``depths``, the
            ``weights`` and whether the sum was ``coherent``. A coherent sum
            keeps the slabs' ``phase_reference``; an incoherent one clears
            it and returns real dB.

        Raises
        ------
        ConfigurationError
            Non-Field slabs; slabs on different grids; a weight vector of
            the wrong length or with a non-finite entry. A coherent sum also
            refuses a real frequency-domain (dB) stack, a complex weight on
            a time-domain stack, and a complex weight on data carrying no
            phase reference; an incoherent sum refuses time-domain traces,
            which do not add in intensity sample by sample.
        """
        first = self.slabs[0]
        if not isinstance(first, Field):
            raise ConfigurationError(
                f"ResultStack.superpose: slabs are "
                f"{self.slab_type.__name__}, not Field — only gridded "
                f"pressure adds. Pick a slab with stack[i]."
            )
        if self.coordinate_name != 'source_depth':
            raise ConfigurationError(
                f"ResultStack.superpose: this stack runs over "
                f"{self.coordinate_name!r}, not over sources. The sum is the "
                f"field of an array of sources driven together; summing "
                f"slabs over {self.coordinate_name!r} adds fields that "
                f"never coexist, and would record the "
                f"{self.coordinate_name} values as source depths.",
                remediation=f"Select a slab with stack[i] or "
                            f"stack.at({self.coordinate_name}=…).")
        if weights is None:
            weights = first.source_weights
        if weights is None:
            weights = np.ones(self.n_slabs)
        w = np.atleast_1d(np.asarray(weights, dtype=np.complex128))
        if w.ndim != 1 or w.size != self.n_slabs:
            raise ConfigurationError(
                f"ResultStack.superpose: {self.n_slabs} slabs but "
                f"{w.size} weight(s) (shape {w.shape}); give one weight "
                f"per slab."
            )
        if not np.all(np.isfinite(w)):
            bad = int(np.flatnonzero(~np.isfinite(w))[0])
            raise ConfigurationError(
                f"ResultStack.superpose: weights must be finite; "
                f"weights[{bad}] = {w[bad]}."
            )
        _check_superpose_grids(self.slabs)
        if not coherent:
            return self._superpose_incoherent(w)
        total, w = _weighted_slab_sum(self.slabs, w,
                                     where="ResultStack.superpose")

        id_kwargs = first.id_kwargs()
        id_kwargs['source_depths'] = self.coordinate.copy()
        id_kwargs['source_weights'] = None
        meta = id_kwargs['metadata']
        meta['superposed_sources'] = {
            'depths': self.coordinate.tolist(),
            'weights': w.tolist(),
            'coherent': True,
        }
        pinned = {k: v for k, v in first.pinned.items()
                  if k != 'source_depth'}
        return first.replace(data=total, pinned=pinned, **id_kwargs)

    def _superpose_incoherent(self, w) -> 'Field':
        """``√Σ|wᵢ·pᵢ|²`` over the slabs, as a real dB ``Field``.

        Reads magnitudes only, so it serves a dB-only stack as well as a
        complex one: ``|p| = 10**(-dB/20)`` recovers the magnitude a level
        already is. Time-domain traces are refused — intensity does not add
        sample by sample."""
        first = self.slabs[0]
        if 'time' in first.coords:
            raise ConfigurationError(
                "ResultStack.superpose(coherent=False): the slabs are "
                "time-domain traces, and intensity does not add sample by "
                "sample. Sum the traces coherently, or superpose the "
                "BROADBAND transfer function and synthesise afterwards."
            )
        if np.any(w.imag != 0.0):
            warnings.warn(
                "ResultStack.superpose(coherent=False): an intensity sum "
                "carries no phase, so only the weight magnitudes are used "
                f"({np.abs(w).tolist()}); the phases of {w.tolist()} are "
                "dropped. Pass coherent=True to drive the sources with a "
                "fixed relative phase.",
                FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
        amp = np.abs(w)

        # A loss counts DOWN from the source and a level counts UP, so the
        # stored numbers turn into an amplitude with opposite signs. Reading
        # every real dB field as a loss inverted a stack of levels: two
        # incoherent 120 dB sources came back 116.99 dB instead of 123.01.
        loss = _quantities.is_loss(first.kind)
        sign = -1.0 if loss else 1.0

        def magnitude(slab):
            data = np.asarray(slab.data)
            if slab.is_complex:
                return np.abs(data)
            return np.power(10.0, sign * np.asarray(data, dtype=float) / 20.0)

        total = np.zeros(np.asarray(first.data).shape, dtype=float)
        for a, slab in zip(amp, self.slabs):
            total += (a * magnitude(slab)) ** 2
        # Clamped like every other dB view (``transmission_loss_dB``), so a cell no
        # energy reached reads the package's floor instead of ``inf``.
        # A loss counts down (-20log10|p|), a level counts up (+20log10|p|),
        # which is the same ``sign`` the magnitudes were recovered with.
        level = sign * 20.0 * np.log10(
            np.maximum(np.sqrt(total), PRESSURE_FLOOR))

        id_kwargs = first.id_kwargs()
        id_kwargs['source_depths'] = self.coordinate.copy()
        id_kwargs['phase_reference'] = None
        id_kwargs['source_weights'] = None
        meta = id_kwargs['metadata']
        id_kwargs['unit'] = 'dB'
        # An intensity sum is incoherent whatever run mode produced the
        # slabs; the run mode it inherits would say otherwise.
        id_kwargs['coherent'] = False
        meta['superposed_sources'] = {
            'depths': self.coordinate.tolist(),
            'weights': amp.tolist(),
            'coherent': False,
        }
        pinned = {k: v for k, v in first.pinned.items()
                  if k != 'source_depth'}
        return first.replace(data=level, pinned=pinned, **id_kwargs)

    def _warn_if_weights_unapplied(self, view: str) -> None:
        """Every slab is the unit-amplitude field of one source; when the
        ``Source`` carried other weights, the level view or panel plot of
        the slabs shows a field those weights never touched. Say so once,
        naming :meth:`superpose`, which is where they apply."""
        weights = self.slabs[0].source_weights
        if weights is None or np.all(np.asarray(weights) == 1.0):
            return
        warnings.warn(
            f"ResultStack.{view}: the slabs are unit-amplitude fields; the "
            f"Source weights {np.asarray(weights).tolist()} this stack "
            f"carries are not applied to them. Call stack.superpose() for "
            f"the weighted field.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)

    def to_xarray(self):
        """This stack as one ``xarray.DataArray`` (optional extra
        ``uacpy[xarray]``), the slabs joined along a leading
        :attr:`coordinate_name` dimension whose coordinate is
        :attr:`coordinate`.

        Each slab is written by :meth:`Field.to_xarray`, so ``attrs`` carry
        the first slab's quantity and identity — every slab shares them except
        along the stacking axis, whose values are the new coordinate. The
        attribute that axis varies (``source_depths`` for a ``source_depth``
        stack, ``frequencies`` for a ``frequency`` stack) is therefore left
        out of ``attrs``. The Source weights are recorded, not applied (see
        :meth:`superpose`), as the complex pair :meth:`Field.to_xarray`
        writes. :meth:`from_xarray` reads the whole stack back;
        ``da.isel({coordinate_name: i})`` read with :meth:`Field.from_xarray`
        is slab ``i`` alone, the stacking label as a pinned coordinate.
        Requires :class:`Field` slabs.
        """
        first = self.slabs[0]
        if not isinstance(first, Field):
            raise ConfigurationError(
                f"ResultStack.to_xarray: slabs are {self.slab_type.__name__}, "
                f"not Field — only a gridded stack is one DataArray.")
        try:
            import xarray as xr
        except ImportError as exc:
            raise ConfigurationError(
                "ResultStack.to_xarray: xarray is not installed.",
                remediation="pip install 'uacpy[xarray]'") from exc
        arrays = [s.to_xarray() for s in self.slabs]
        stacked = xr.concat(arrays, dim=self.coordinate_name,
                            combine_attrs='override')
        stacked = stacked.assign_coords(
            {self.coordinate_name: np.asarray(self.coordinate, dtype=float)})
        varying = _RESULTSTACK_VARYING_ATTR.get(self.coordinate_name)
        if varying is not None:
            stacked.attrs.pop(varying, None)
        return stacked

    @classmethod
    def from_xarray(cls, array, coordinate_name: Optional[str] = None
                    ) -> 'ResultStack':
        """A stack of :class:`Field` slabs from an ``xarray.DataArray``, such
        as :meth:`to_xarray` writes: the inverse of that method.

        ``coordinate_name`` names the stacking dimension; ``None`` takes the
        leading one, where :meth:`to_xarray` puts it. Each slab is read by
        :meth:`Field.from_xarray` and carries the stacking label as its
        identity when the axis is one (``source_depths=[z]`` for a
        ``source_depth`` stack), not as a pinned coordinate, so the slabs
        come back as the stack held them.

        Parameters
        ----------
        array : xarray.DataArray
            An array as :meth:`to_xarray` writes it.
        coordinate_name : str, optional
            The stacking dimension; ``None`` is the leading one.
        """
        # A file's complex data, stored as real and imaginary parts.
        array = join_complex(array)
        name = str(array.dims[0]) if coordinate_name is None \
            else str(coordinate_name)
        if name not in array.dims or name not in array.coords:
            raise ConfigurationError(
                f"ResultStack.from_xarray: {name!r} is not a dimension with "
                f"a coordinate of this DataArray (dims {list(array.dims)}).",
                remediation="Pass coordinate_name= naming the stacking "
                            "dimension.")
        coordinate = np.asarray(array.coords[name].values, dtype=float)
        slabs = []
        for i in range(coordinate.size):
            slab = Field.from_xarray(array.isel({name: i}))
            slab.pinned.pop(name, None)
            slabs.append(slab)
        return cls(slabs, coordinate, coordinate_name=name)

    def to_netcdf(self, path, **kwargs) -> None:
        """Write :meth:`to_xarray` to a NetCDF file at ``path``
        (:func:`~uacpy.core._export.write_netcdf`); :meth:`from_netcdf`
        reads it back. Requires :class:`Field` slabs, as :meth:`to_xarray`
        does.

        Parameters
        ----------
        path : str or path-like
            The file to write (replaced if it exists).
        **kwargs
            Passed to xarray's ``to_netcdf``. Complex values are written as
            real and imaginary parts any backend stores.

        Raises
        ------
        ConfigurationError
            Non-Field slabs.
        """
        write_netcdf(self.to_xarray(), path,
                     who="ResultStack.to_netcdf", **kwargs)

    @classmethod
    def from_netcdf(cls, path, coordinate_name: Optional[str] = None,
                    **kwargs) -> 'ResultStack':
        """The stack :meth:`to_netcdf` wrote to ``path``: the file's one
        array read with ``xarray.open_dataarray`` and rebuilt by
        :meth:`from_xarray`.

        Parameters
        ----------
        path : str or path-like
            The file to read.
        coordinate_name : str, optional
            The stacking dimension; ``None`` is the leading one.
        **kwargs
            Passed to ``xarray.open_dataarray``.
        """
        from uacpy.core._export import require_extra
        xarray = require_extra('xarray', 'ResultStack.from_netcdf')
        with xarray.open_dataarray(path, **kwargs) as array:
            array = array.load()
        return cls.from_xarray(array, coordinate_name=coordinate_name)

    def to_dict(self) -> Dict[str, Any]:
        """This stack as plain data: ``coordinate`` (copied),
        ``coordinate_name`` and ``slabs``, each slab its own
        :meth:`Result.to_dict` under ``'__class__'``, its public class path,
        as :meth:`Result.to_dict` nests a component. Any slab type saves, not
        only :class:`Field`. ``np.savez(path, **stack.to_dict())`` stores it;
        :meth:`from_dict` of ``dict(np.load(path, allow_pickle=True))`` reads
        it back.

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
            allow_pickle=True)`` returns for a file it was saved to.

        Returns
        -------
        ResultStack

        Raises
        ------
        ConfigurationError
            A slab class path outside uacpy, or one that names no
            :class:`Result`.
        """
        d = {k: unwrap_0d(v) for k, v in dict(d).items()}
        slabs = []
        for saved in list(d['slabs']):
            saved = dict(saved)
            klass = _resolve_class(saved.pop('__class__'), Result)
            slabs.append(klass.from_dict(saved))
        return cls(slabs, np.asarray(d['coordinate'], dtype=float),
                   coordinate_name=str(d['coordinate_name']))

    def plot(self, **kwargs):
        """Plot every slab as a labelled panel grid (Field stacks), delegating
        to :func:`uacpy.plot.plot_result`."""
        if isinstance(self.slabs[0], Field):
            self._warn_if_weights_unapplied('plot')
        return plotter('plot_result')(self, **kwargs)

    def __repr__(self) -> str:
        return build('ResultStack', [
            count(self.n_slabs, f"{self.slab_type.__name__} slab"),
            coordinate_axis(self.coordinate_name, self.coordinate)])
