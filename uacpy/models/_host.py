"""The host options of an engine: where and how its runs execute on this
machine (``use_tmpfs``, ``verbose``, ``work_dir``, ``cleanup``, ``timeout``)
and the ``collapse`` policy of the environment features it does not read.

:meth:`~uacpy.models.base.PropagationModel.__init__`'s keyword parameters
declare them. Each engine's constructor repeats them as explicit keyword
parameters, so ``help()`` and ``inspect.signature`` list every knob and a
misspelt one is a ``TypeError``: :func:`check_host_parameters` refuses, when
the class is defined, a copy that differs from the declaration, and
:func:`set_host_options` validates and stores the values the constructor
forwards."""

import inspect
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from uacpy._log import _resolve_threshold
from uacpy.core.exceptions import ConfigurationError
from uacpy.models._knobs import is_real_number
from uacpy.models._projection import DEFAULT_COLLAPSE, VALID_COLLAPSE_METHODS


def check_host_parameters(cls, declared: inspect.Signature) -> None:
    """Refuse the model class ``cls`` when the ``__init__`` it defines does
    not repeat every host option of ``declared`` (the signature of
    ``PropagationModel.__init__``) as a keyword-only parameter with the same
    default and annotation, or takes a ``**kwargs`` sink. A class that
    defines no ``__init__`` inherits one that does."""
    init = cls.__dict__.get('__init__')
    if init is None:
        return
    own = inspect.signature(init).parameters
    label = f"{cls.__name__}.__init__()"
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in own.values()):
        raise TypeError(
            f"{label} must not use **kwargs: an unknown keyword has to "
            f"raise TypeError, and help() has to list every knob. Declare "
            f"the host options as keyword-only parameters.")
    for name, base in declared.parameters.items():
        if name == 'self':
            continue
        p = own.get(name)
        if p is None:
            raise TypeError(
                f"{label} lacks the host option {name!r}: every engine "
                f"constructor takes PropagationModel.__init__'s keyword "
                f"parameters, {name}={base.default!r} among them, and "
                f"forwards them with super().__init__(**_host_options("
                f"locals())).")
        if p.kind is not inspect.Parameter.KEYWORD_ONLY:
            raise TypeError(
                f"{label} parameter {name!r} must be keyword-only, as "
                f"PropagationModel.__init__ declares it.")
        if p.default != base.default or p.annotation != base.annotation:
            raise TypeError(
                f"{label} parameter {name!r} differs from "
                f"PropagationModel.__init__'s: {p} against {base}.")


def set_host_options(
    model,
    *,
    use_tmpfs: bool,
    verbose: Union[bool, str],
    work_dir: Optional[Path],
    cleanup: Optional[bool],
    timeout: Optional[float],
    collapse: Optional[Dict[str, str]],
) -> None:
    """Validate the host options a constructor of ``model`` forwards and
    store them: each as the attribute of its name, ``cleanup`` resolved,
    and the resolved collapse policy as ``model._collapse`` (the defaults
    under the user's keys, which ``model._user_collapse`` keeps so the
    spec's defaults cannot override them)."""
    _resolve_threshold(verbose)  # validate up front
    model.use_tmpfs = use_tmpfs
    model.verbose = verbose
    model.work_dir = work_dir
    # cleanup defaults to True only when uacpy owns the work dir. Whether
    # the caller passed it is kept so ``copy()`` can re-resolve against a
    # new work_dir instead of carrying an auto-derived True onto a
    # caller-supplied directory and deleting it.
    model._cleanup_explicit = cleanup is not None
    model.cleanup = (work_dir is None) if cleanup is None else bool(cleanup)
    # ``communicate(timeout=…)`` takes a positive finite number of seconds
    # (inf overflows it, nan is refused by it, 0 kills every launch) or
    # ``None``, its own "no limit", which every launch passes on as given.
    if timeout is not None and not (
            is_real_number(timeout) and np.isfinite(timeout) and timeout > 0):
        raise ConfigurationError(
            f"{type(model).__name__}: timeout (s) must be positive and "
            f"finite, or None for no limit; got {timeout!r}.")
    model.timeout = None if timeout is None else float(timeout)
    # Per-feature collapse policies applied by ``_project_environment``
    # when an env contains a feature this model doesn't support. Pass
    # ``collapse={'bathymetry': 'min', 'ssp': 'mean', ...}`` to override
    # any subset; missing keys keep the defaults.
    #
    # ``VALID_COLLAPSE_METHODS`` lists the methods each key takes, and is
    # itself read from the carriers that implement them, so the values are
    # not restated here. What each key reduces:
    #
    # 'bathymetry'    : a range-dependent depth profile to one depth
    # 'ssp'           : a range-dependent sound-speed field to one column
    # 'bottom_range'  : a range-dependent bottom to one column ('mean'
    #                   averages the half-spaces, so an all-half-space bottom
    #                   only — a layer stack cannot be averaged; 'median' on
    #                   a layered bottom takes the middle column, with a
    #                   notice, Bottom.collapse_range)
    # 'bottom_layers' : each column's layer stack to a half-space; see
    #                   SeabedColumn.collapse_layers for what each method keeps
    # 'altimetry'     : a rough sea surface, dropped entirely
    # 'surface'       : a range-dependent Surface to a single boundary
    # 'elastic'       : a shear-supporting boundary to one that does not
    #                   ('fluid' zeroes the shear, 'vacuum' replaces it)
    #
    # ``model._collapse`` is the resolved policy (defaults ← spec ← user);
    # ``model.collapse`` keeps the constructor argument verbatim, because
    # ``copy()`` and ``__repr__`` read every knob off ``model.<param>``.
    model._collapse = dict(DEFAULT_COLLAPSE)
    model._user_collapse = {}
    model.collapse = dict(collapse) if collapse else None
    if collapse:
        unknown = set(collapse) - set(DEFAULT_COLLAPSE)
        if unknown:
            raise ConfigurationError(
                f"Unknown collapse keys: {sorted(unknown)}. "
                f"Valid keys: {sorted(DEFAULT_COLLAPSE)}."
            )
        for key, value in collapse.items():
            if value not in VALID_COLLAPSE_METHODS[key]:
                raise ConfigurationError(
                    f"Invalid collapse value for {key!r}: {value!r}. "
                    f"Valid values: {sorted(VALID_COLLAPSE_METHODS[key])}."
                )
        model._collapse.update(collapse)
        model._user_collapse = dict(collapse)
