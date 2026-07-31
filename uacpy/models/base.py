"""Base class for acoustic propagation models."""

import copy as _copy
import os
import shutil
import signal
import subprocess
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

import uacpy._stack  # noqa: F401 — side-effect: raise RLIMIT_STACK
from uacpy.core.environment import Environment, Bathymetry
from uacpy.core.exceptions import (
    ConfigurationError,
    ExecutableNotFoundError,
    InvalidDepthError,
    ModelExecutionError,
    UnsupportedFeatureError,
)
from uacpy.core.receiver import Receiver
from uacpy.core.results import Result
from uacpy.core.source import Source
from uacpy.io.file_manager import FileManager
from uacpy.io.oalib_reader import read_prt


class RunMode(Enum):
    """
    Standard run modes for acoustic propagation models.

    Models may support a subset of these modes.
    """
    COHERENT_TL = 'coherent_tl'          # Coherent transmission loss
    INCOHERENT_TL = 'incoherent_tl'      # Incoherent (averaged) TL
    SEMICOHERENT_TL = 'semicoherent_tl'

    RAYS = 'rays'                        # Ray paths only
    EIGENRAYS = 'eigenrays'              # Eigenrays (specific paths)
    ARRIVALS = 'arrivals'                # Arrival structure

    MODES = 'modes'                      # Normal modes (Kraken depth eigenfunctions)

    # OASN frequency-domain array products: COVARIANCE → C(f, i, j) hydrophone
    # × hydrophone matrix; REPLICA → Green's-function samples at the array
    # elements per candidate source position. See core/results.Covariance and
    # core/results.Replicas.
    COVARIANCE = 'covariance'
    REPLICA = 'replica'

    # Time-domain pressure p(t) at the receiver(s). Models that compute a
    # broadband transfer function natively (Bellhop, RAM, Scooter,
    # Kraken, OASES) require ``source_waveform=`` + ``sample_rate=``;
    # SPARC computes p(t) directly from its source pulse and ignores them.
    TIME_SERIES = 'time_series'

    # Broadband complex transfer function H(f).
    BROADBAND = 'broadband'

    REFLECTION = 'reflection'            # Plane-wave reflection coefficients (Bounce, OASR)


DEFAULT_COLLAPSE: Dict[str, str] = {
    'bathymetry': 'max',
    'ssp': 'r0',
    'bottom_range': 'r0',
    'bottom_layers': 'halfspace',
    'altimetry': 'drop',
    'surface': 'r0',
    'elastic': 'fluid',
}

# Allowed method strings per collapse key (validated at construction in
# ``PropagationModel.__init__`` so bad values fail loudly rather than deep
# inside a writer at ``run()``-time).
VALID_COLLAPSE_METHODS: Dict[str, frozenset] = {
    'bathymetry':        frozenset({'max', 'median', 'mean', 'min', 'initial'}),
    'ssp':               frozenset({'r0', 'rmax', 'mean', 'median'}),
    'bottom_range':      frozenset({'r0', 'rmax', 'mean', 'median'}),
    'bottom_layers':     frozenset({'halfspace', 'top_layer', 'volume_average'}),
    'altimetry':         frozenset({'drop'}),
    'surface':           frozenset({'r0', 'rmax', 'mean', 'median'}),
    'elastic':           frozenset({'fluid', 'vacuum'}),
}
# Dev invariants on the collapse-policy constants (raise, not assert, so they
# survive `python -O`).
if set(VALID_COLLAPSE_METHODS) != set(DEFAULT_COLLAPSE):
    raise RuntimeError("VALID_COLLAPSE_METHODS keys must match DEFAULT_COLLAPSE keys")
if not all(DEFAULT_COLLAPSE[k] in VALID_COLLAPSE_METHODS[k] for k in DEFAULT_COLLAPSE):
    raise RuntimeError("DEFAULT_COLLAPSE values must satisfy VALID_COLLAPSE_METHODS")


# Source geometries a model may declare via ``ModelSpec.source_types``.
# 'point'  -> AT 'R', cylindrical spreading applied
# 'line'   -> AT 'X', Cartesian spreading
# 'scaled' -> AT 'S', point source with cylindrical spreading removed
VALID_SOURCE_TYPES: frozenset = frozenset({'point', 'line', 'scaled'})

# Fewest frequencies an auto-derived TIME_SERIES grid may carry. Δf is
# 1/waveform-duration, so a short pulse over a narrow band can derive 2-3 bins
# — below what the band-edge taper in ``_ifft_to_trace`` can act on, and too
# few to represent an arrival. Costs one model run per extra bin.
_MIN_TIMESERIES_FREQS = 8


def _max_roughness(boundaries) -> float:
    """Largest interfacial sigma over a list of ``BoundaryProperties``."""
    return max(
        (float(getattr(b, 'roughness', 0.0) or 0.0)
         for b in boundaries if b is not None),
        default=0.0,
    )


def _smooth_surface(surface):
    """``surface`` with every node's roughness zeroed.

    The nodes are rebuilt rather than assigned through: ``Surface`` serves
    ``roughness`` by ``__getattr__`` delegation to ``properties[0]``, and a
    plain attribute write would create an instance attribute that shadows the
    delegation while ``properties`` — what ``at()``, ``collapse()`` and the
    repr read — keeps the old value.
    """
    smoothed = _copy.deepcopy(surface)
    for node in smoothed.properties:
        # Assigned rather than rebuilt: ``dataclasses.replace`` re-runs
        # ``__post_init__``, which rejects explicit acoustic parameters on a
        # vacuum / rigid boundary even when they are the values it filled in.
        object.__setattr__(node, 'roughness', 0.0)
    return smoothed


def _smooth_bottom(bottom):
    """``bottom`` with every column's half-space roughness zeroed."""
    smoothed = _copy.deepcopy(bottom)
    for column in smoothed.columns:
        object.__setattr__(column.halfspace, 'roughness', 0.0)
    return smoothed


# Capability-flag names a model may advertise. Each maps to a
# ``_supports_<name>`` instance attribute; the question each answers is "does
# this env *shape* — or this source feature — work with this model?".
# Keep in lockstep with the ``self._supports_*`` block in
# ``PropagationModel.__init__``.
_CAPABILITY_FLAGS: frozenset = frozenset({
    'altimetry',
    'range_dependent_surface',
    'range_dependent_bathymetry',
    'range_dependent_ssp',
    'range_dependent_bottom',
    'layered_bottom',
    'range_dependent_layered_bottom',
    'elastic_media',
    'multi_source_depth',
    'source_beam_pattern',
    'rough_surface',
    'rough_bottom',
})


# Source ``id``s already warned about this process, so a licence-restricted
# engine (OASES) emits its one-time UserWarning once, not per instance.
_WARNED_MODEL_SOURCES: set = set()


@dataclass(frozen=True)
class ModelSpec:
    """Declarative per-model metadata read by :class:`PropagationModel`.

    Consolidates the *static* facts about a model — the run modes it
    emits, the environment shapes it handles natively, its physics-aware
    collapse defaults, and how to locate its binary — into one block the
    base class reads and **validates at class-definition time**, instead
    of scattering them across ``__init__``. The model stays a
    ``PropagationModel`` subclass, so all generic machinery (collapse
    application, validation, file manager, subprocess, ``copy()``) is
    inherited unchanged; the spec only supplies metadata.

    Precedence is preserved: ``collapse`` here layers on top of
    :data:`DEFAULT_COLLAPSE` but never overrides an explicit
    ``Model(collapse={...})`` user value (same rule as
    :meth:`PropagationModel._set_collapse_defaults`). A subclass may still
    set an instance-dependent flag in ``__init__`` *after* ``super().__init__``
    for the rare case a capability depends on a constructor argument.

    Fields
    ------
    modes : sequence of RunMode
        Run modes the model emits. Becomes ``self._supported_modes``; the
        first entry is the default when ``run_mode=None``.
    supports : iterable of str
        Capability-flag names (subset of :data:`_CAPABILITY_FLAGS`) the
        model honours natively. Every flag not listed defaults ``False``
        and its env feature is collapsed by ``_project_environment``.
    source_types : frozenset of str
        Source geometries (subset of :data:`VALID_SOURCE_TYPES`) the model
        honours. Becomes ``self._supported_source_types``; a ``Source``
        carrying anything else is rejected by ``validate_inputs``.
    collapse : dict
        Per-model collapse defaults overriding :data:`DEFAULT_COLLAPSE`.

    Provenance/licence is intentionally *not* here: it is an orthogonal axis
    (who wrote the engine, under what licence) from the run-behaviour fields
    above, and the codebase already keeps dataset provenance in a separate
    catalogue (:mod:`uacpy.data.sources`) rather than folding it into the
    carriers. Models declare it the same way — a ``source`` class attribute
    referencing :data:`uacpy.models.sources.MODEL_SOURCES`.

    Binary resolution is intentionally *not* here either: nothing generic reads it,
    its shape differs per model (single name vs. list of search dirs vs. the
    OASES helper vs. Bellhop's backend dispatch), and multi-binary models
    (Kraken's krakenc, RAM's Collins backends) pick the real executable at
    ``run()`` time. Each model resolves ``self._exe`` in its own ``__init__``.
    """

    modes: tuple = ()
    supports: frozenset = frozenset()
    source_types: frozenset = frozenset({'point'})
    collapse: dict = field(default_factory=dict)

    def validate(self, model_name: str) -> None:
        """Fail loudly at class-definition time on a malformed spec."""
        for m in self.modes:
            if not isinstance(m, RunMode):
                raise TypeError(
                    f"{model_name}.spec.modes must contain RunMode members; "
                    f"got {m!r}."
                )
        bad_flags = set(self.supports) - _CAPABILITY_FLAGS
        if bad_flags:
            raise ValueError(
                f"{model_name}.spec.supports has unknown capability flags: "
                f"{sorted(bad_flags)}. Valid: {sorted(_CAPABILITY_FLAGS)}."
            )
        bad_types = set(self.source_types) - VALID_SOURCE_TYPES
        if bad_types:
            raise ValueError(
                f"{model_name}.spec.source_types has unknown geometries: "
                f"{sorted(bad_types)}. Valid: {sorted(VALID_SOURCE_TYPES)}."
            )
        if not self.source_types:
            raise ValueError(
                f"{model_name}.spec.source_types is empty; every model must "
                f"accept at least one source geometry."
            )
        unknown = set(self.collapse) - set(DEFAULT_COLLAPSE)
        if unknown:
            raise ValueError(
                f"{model_name}.spec.collapse has unknown keys: "
                f"{sorted(unknown)}. Valid keys: {sorted(DEFAULT_COLLAPSE)}."
            )
        for key, value in self.collapse.items():
            if value not in VALID_COLLAPSE_METHODS[key]:
                raise ValueError(
                    f"{model_name}.spec.collapse[{key!r}] = {value!r} is "
                    f"invalid. Valid: {sorted(VALID_COLLAPSE_METHODS[key])}."
                )


class PropagationModel(ABC):
    """
    Abstract base class for acoustic propagation models.

    Provides the common interface and shared utilities (subprocess runner,
    executable lookup, input validation, range-dependent handling) for all
    propagation models.

    Parameters
    ----------
    use_tmpfs : bool, optional
        Use a RAM-backed filesystem for I/O. Default is False.
    verbose : bool or str, optional
        Status-output gate. ``False`` (default) prints only ``WARN`` and
        ``ERROR``. ``True`` or ``'info'`` also prints ``INFO``. ``'debug'``
        additionally prints ``DEBUG`` (per-subprocess command lines,
        grid-resolution choices, etc.). See :mod:`uacpy._log`.
    work_dir : str or Path, optional
        Working directory for files. If ``None``, a temporary directory is
        created per run.

    Attributes
    ----------
    model_name : str
        Name of the model (class name).
    use_tmpfs : bool
        Whether tmpfs is used.
    verbose : bool or str
        Verbose-output gate (see constructor).
    file_manager : FileManager
        File manager instance (populated during ``run``).
    """

    # The leading positional run() parameters every wrapper must carry, in
    # order. Anything a wrapper adds beyond these must be keyword-only (after
    # a bare ``*``) and no wrapper may use ``**kwargs`` — an unknown keyword
    # has to fail with TypeError at the call site, not be silently swallowed.
    _RUN_POSITIONAL = ('self', 'env', 'source', 'receiver', 'run_mode')

    # Declarative metadata. ``None`` keeps the legacy path (subclass sets
    # ``_supported_modes`` / ``_supports_*`` / collapse defaults by hand in
    # ``__init__``). When a subclass declares a :class:`ModelSpec`, the base
    # validates it at class-definition time and applies it in ``__init__``.
    spec: Optional['ModelSpec'] = None

    # Provenance/licence: ``id`` of this engine's entry in
    # :data:`uacpy.models.sources.MODEL_SOURCES`. Kept off :class:`ModelSpec`
    # (which is about run behaviour) because provenance is an orthogonal axis —
    # the same separation the data layer keeps between its carriers and
    # :mod:`uacpy.data.sources`. Surfaced via :attr:`provenance` / :attr:`citation`;
    # a ``commercial_use=False`` engine (OASES) warns once at construction.
    source: Optional[str] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        spec = cls.__dict__.get('spec')
        if spec is not None:
            if not isinstance(spec, ModelSpec):
                raise TypeError(
                    f"{cls.__name__}.spec must be a ModelSpec, got "
                    f"{type(spec).__name__}."
                )
            spec.validate(cls.__name__)
        if cls.source is not None:
            from uacpy.models.sources import MODEL_SOURCES
            if cls.source not in MODEL_SOURCES:
                raise ValueError(
                    f"{cls.__name__}.source = {cls.source!r} is not a known "
                    f"model source. Valid: {sorted(MODEL_SOURCES)}."
                )
        run = cls.__dict__.get('run')
        if run is None:
            return
        import inspect

        params = list(inspect.signature(run).parameters.values())
        for p in params:
            if p.kind is inspect.Parameter.VAR_KEYWORD:
                raise TypeError(
                    f"{cls.__name__}.run() must not use **kwargs: an unknown "
                    "keyword has to raise TypeError, not be swallowed. Declare "
                    "the accepted extras as keyword-only after a bare '*'."
                )

        leading = params[:len(cls._RUN_POSITIONAL)]
        names = tuple(p.name for p in leading)
        if names != cls._RUN_POSITIONAL:
            raise TypeError(
                f"{cls.__name__}.run() must begin with "
                f"{cls._RUN_POSITIONAL!r}; got {names!r}."
            )
        for p in leading:
            if p.kind is inspect.Parameter.KEYWORD_ONLY:
                raise TypeError(
                    f"{cls.__name__}.run() parameter {p.name!r} must be "
                    "positional-or-keyword, not keyword-only."
                )

        for p in params[len(cls._RUN_POSITIONAL):]:
            if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                raise TypeError(
                    f"{cls.__name__}.run() parameter {p.name!r} must be "
                    "keyword-only (place it after a bare '*') so unknown "
                    "positional args cannot reach it."
                )

    def __init__(
        self,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        timeout: float = 600.0,
        collapse: Optional[Dict[str, str]] = None,
    ):
        from uacpy._log import _resolve_threshold
        _resolve_threshold(verbose)  # validate up front
        self.model_name = self.__class__.__name__
        self.use_tmpfs = use_tmpfs
        self.verbose = verbose
        self.work_dir = work_dir
        # cleanup defaults to True only when uacpy owns the work dir. Whether
        # the caller passed it is kept so ``copy()`` can re-resolve against a
        # new work_dir instead of carrying an auto-derived True onto a
        # caller-supplied directory and deleting it.
        self._cleanup_explicit = cleanup is not None
        self.cleanup = (work_dir is None) if cleanup is None else bool(cleanup)
        self.timeout = float(timeout)
        # Per-feature collapse policies applied by ``_project_environment``
        # when an env contains a feature this model doesn't support. Pass
        # ``collapse={'bathymetry': 'min', 'ssp': 'mean', ...}`` to override
        # any subset; missing keys keep the defaults.
        #
        # 'bathymetry'    : 'max'|'median'|'mean'|'min'|'initial'
        # 'ssp'           : 'r0'|'rmax'|'mean'|'median'
        # 'bottom_range'  : 'r0'|'rmax'|'mean'|'median' — reduce a range-
        #                   dependent bottom to one column (mean/median
        #                   numeric only for an all-half-space bottom)
        # 'bottom_layers' : 'halfspace'|'top_layer'|'volume_average' — flatten
        #                   each column's layer stack to a half-space
        # 'altimetry'     : 'drop'
        # 'elastic'       : 'fluid' (zero shear) | 'vacuum'
        self._collapse: Dict[str, str] = dict(DEFAULT_COLLAPSE)
        self._user_collapse: Dict[str, str] = {}
        self.collapse = dict(collapse) if collapse else None
        if collapse:
            unknown = set(collapse) - set(DEFAULT_COLLAPSE)
            if unknown:
                raise ConfigurationError(
                    f"Unknown collapse keys: {sorted(unknown)}. "
                    f"Valid keys: {sorted(DEFAULT_COLLAPSE)}"
                )
            for key, value in collapse.items():
                if value not in VALID_COLLAPSE_METHODS[key]:
                    raise ConfigurationError(
                        f"Invalid collapse value for {key!r}: {value!r}. "
                        f"Valid values: {sorted(VALID_COLLAPSE_METHODS[key])}"
                    )
            self._collapse.update(collapse)
            self._user_collapse = dict(collapse)
        self.file_manager = None

        # Subclasses override to declare the run modes they support.
        self._supported_modes: List[RunMode] = [RunMode.COHERENT_TL]

        # Capability flags — one per axis of ``Environment`` shape. Subclasses
        # flip True for each feature they honour natively; anything left False
        # that's present in env on ``run()`` is collapsed by
        # ``_project_environment`` and triggers one ``UserWarning`` per dropped
        # feature.
        #
        # The flag list is intentionally bounded. Add a flag ONLY for a
        # question of the form "does this env shape work with this model?".
        # Niche numerical-method requirements (3-D, broadband, specific SSP
        # interp scheme, volume-attenuation formula) belong in run()-time
        # asserts, not here.
        self._supports_altimetry: bool = False
        self._supports_range_dependent_surface: bool = False
        self._supports_range_dependent_bathymetry: bool = False
        self._supports_range_dependent_ssp: bool = False
        self._supports_range_dependent_bottom: bool = False
        self._supports_layered_bottom: bool = False
        # Declared per model to document the axis as live; the combined
        # range-dependent-layered capability is the conjunction of the two
        # axes above (which is what _project_environment collapses against).
        self._supports_range_dependent_layered_bottom: bool = False
        self._supports_elastic_media: bool = False
        # Bellhop is the only model that runs one source-depth grid in
        # a single binary call; everyone else loops in Python.
        self._supports_multi_source_depth: bool = False
        self._supports_source_beam_pattern: bool = False
        # SPARC's GetPar and Bounce's elastic branch ERROUT on a
        # non-zero SSP%sigma; Kraken/Scooter consume it.
        self._supports_rough_surface: bool = False
        # Seabed sigma(NMedia+1). Kraken/Scooter apply the Kirchhoff
        # attenuation; Bellhop's solver ignores the value and RAM's PE
        # format has nowhere to put it.
        self._supports_rough_bottom: bool = False
        self._supported_source_types: frozenset = frozenset({'point'})

        # When the subclass declares a ModelSpec, apply it now (after the
        # defaults above and after ``_user_collapse`` is populated, so the
        # collapse precedence DEFAULT ← spec ← user override holds). A
        # subclass may still override an individual flag afterward for the
        # rare instance-dependent capability.
        if self.spec is not None:
            self._apply_spec()
        # Independent of spec: a model may declare ``source`` without one.
        self._warn_restricted_source()

    @property
    def provenance(self):
        """The engine's :class:`~uacpy.models.sources.ModelSource` (authorship
        + licence + citation), or ``None`` if the class declares no
        :attr:`source`."""
        from uacpy.models.sources import model_source
        return model_source(self.source)

    @property
    def citation(self) -> str:
        """The engine's bibliographic citation string (``''`` if unknown)."""
        src = self.provenance
        return src.citation if src is not None else ''

    def _warn_restricted_source(self) -> None:
        """Emit a one-time ``UserWarning`` for a licence-restricted engine.

        Mirrors the non-commercial fetch warning in ``data/`` (the audit's
        CRUST1.0 fix): a ``commercial_use=False`` source — OASES — must never
        be used silently. Deduplicated per source ``id`` per process so a
        parameter sweep warns once, not on every instance.
        """
        src = self.provenance
        if src is None or src.commercial_use or src.id in _WARNED_MODEL_SOURCES:
            return
        _WARNED_MODEL_SOURCES.add(src.id)
        warnings.warn(
            f"{self.model_name} uses {src.name} ({src.license}). {src.note} "
            f"Cite: {src.citation}",
            UserWarning, stacklevel=3,
        )

    def _apply_spec(self) -> None:
        """Install :attr:`spec`'s metadata onto this instance.

        Sets ``_supported_modes`` and every ``_supports_<flag>`` from the
        declarative manifest and layers the spec's collapse defaults via
        :meth:`_set_collapse_defaults` (so user overrides still win). The
        spec is already validated in ``__init_subclass__``.
        """
        spec = self.spec
        if spec.modes:
            self._supported_modes = list(spec.modes)
        for flag in _CAPABILITY_FLAGS:
            setattr(self, f'_supports_{flag}', flag in spec.supports)
        self._supported_source_types = frozenset(spec.source_types)
        if spec.collapse:
            self._set_collapse_defaults(spec.collapse)

    def _set_collapse_defaults(self, defaults: Dict[str, str]) -> None:
        """Subclass hook: install model-specific collapse defaults.

        Each ``(key, value)`` is applied only when the user did not pass
        an explicit value for ``key`` in ``Model(collapse={...})``, so
        user overrides always win. Use it to express physics-aware
        defaults that differ from the global ``DEFAULT_COLLAPSE``.
        """
        for key, value in defaults.items():
            if key not in self._user_collapse:
                self._collapse[key] = value

    def _resolve_run_mode(
        self,
        run_mode: Optional[Union['RunMode', str]],
        *,
        default: Optional['RunMode'] = None,
    ) -> 'RunMode':
        """Default ``None`` to the model's first supported mode and then
        validate that ``run_mode`` is in ``_supported_modes``. Raises
        :class:`UnsupportedFeatureError` otherwise.

        Strings matching a :class:`RunMode` value (e.g. ``'coherent_tl'``)
        are coerced to the corresponding enum member.

        Pass ``default=`` to override the auto-pick when the model has a
        smarter rule (e.g. Kraken picks BROADBAND when a frequency
        vector is supplied).
        """
        if run_mode is None:
            run_mode = default if default is not None else self._supported_modes[0]
        if isinstance(run_mode, str):
            try:
                run_mode = RunMode(run_mode)
            except ValueError:
                raise UnsupportedFeatureError(
                    self.model_name, repr(run_mode),
                    alternatives=[str(m) for m in self._supported_modes],
                    alternatives_label='run modes',
                )
        if not self.supports_mode(run_mode):
            raise UnsupportedFeatureError(
                self.model_name, str(run_mode),
                alternatives=[str(m) for m in self._supported_modes],
                alternatives_label='run modes',
            )
        return run_mode

    def _reject_unsupported_run_kwargs(self, **kwargs):
        """Guard the optional ``run()`` keywords a model does not consume.

        Every engine declares the full contract signature (``frequencies``,
        ``source_waveform``, ``sample_rate``, ``output_duration``) so a
        polymorphic ``model.run(...)`` never raises ``TypeError``. Frequency-
        domain engines (Bounce, OAST, OASN) ignore the broadband/waveform
        keywords — but ignoring them *silently* would hide a caller mistake, so
        any of these passed a non-``None`` value raises here instead."""
        supplied = sorted(name for name, value in kwargs.items() if value is not None)
        if supplied:
            raise UnsupportedFeatureError(
                self.model_name,
                f"run parameter(s): {', '.join(supplied)}",
                alternatives_label='run parameters',
            )

    def _warn_ignored_run_kwargs(self, run_mode, reason=None, **named_values):
        """Warn when the resolved ``run_mode`` will not consume some of the
        optional ``run()`` keywords the caller supplied.

        Complements :meth:`_reject_unsupported_run_kwargs`: that helper is
        for models that *never* consume a keyword (hard error); this one is
        for keywords the model does consume, just not on the resolved run
        mode's path. Only non-``None`` values are reported, in a single
        ``UserWarning``. Consuming paths (BROADBAND/TIME_SERIES, …) must not
        call it."""
        ignored = [
            f'{name}=' for name, value in named_values.items()
            if value is not None
        ]
        if not ignored:
            return
        if reason is None:
            reason = 'these apply to BROADBAND/TIME_SERIES only'
        warnings.warn(
            f"{self.model_name}.run(run_mode="
            f"{getattr(run_mode, 'name', run_mode)}): ignoring "
            f"{', '.join(ignored)} — {reason}.",
            UserWarning, stacklevel=3,
        )

    @property
    def supported_modes(self) -> List[RunMode]:
        """List of run modes supported by this model."""
        return self._supported_modes

    def supports_mode(self, mode: RunMode) -> bool:
        """Return True if the model supports ``mode``."""
        return mode in self._supported_modes

    def copy(self, **overrides) -> 'PropagationModel':
        """Return a new instance with the same configuration plus ``overrides``.

        Model configuration is constructor-only by design, which means
        every parameter sweep boils down to "instantiate the model again
        with one knob changed." This helper does that without forcing
        the caller to re-type every other argument::

            base = RAM(dr=2.0, dz=0.5, np_pade=8)
            for dr in (1.0, 2.0, 4.0):
                run_one(base.copy(dr=dr), env, source, receiver)

        Implementation: walks ``__init__`` along the MRO (so parameters
        defined on a parent constructor are included, since subclasses
        forward via ``super().__init__(**kwargs)``), pulls each parameter's
        current value off the instance (uacpy models store every
        constructor arg as ``self.<name>``), merges ``overrides``, and
        instantiates. ``**kwargs``-only sinks on the constructor are ignored.

        Parameters
        ----------
        **overrides
            Keyword arguments to override on the new instance.

        Returns
        -------
        PropagationModel
            A fresh instance of the same concrete class.

        Raises
        ------
        ConfigurationError
            If ``overrides`` includes a key that isn't a parameter of
            the constructor.
        """
        kwargs: Dict[str, object] = {}
        valid_names = set()
        for name, _default in _collect_init_params(type(self)):
            valid_names.add(name)
            if hasattr(self, name):
                kwargs[name] = getattr(self, name)

        unknown = set(overrides) - valid_names
        if unknown:
            raise ConfigurationError(
                f"{type(self).__name__}.copy: unknown override(s) "
                f"{sorted(unknown)}; valid parameters are "
                f"{sorted(valid_names)}."
            )

        kwargs.update(overrides)

        # An auto-derived ``cleanup`` describes *this* instance's work_dir, not
        # the clone's. Hand back the ``None`` sentinel so the new instance
        # re-resolves; otherwise ``model.copy(work_dir=d)`` off an unpinned
        # model would arrive at ``cleanup=True`` and rmtree the caller's ``d``.
        # An explicitly passed value is the caller's choice and is preserved.
        if (not self._cleanup_explicit and 'cleanup' not in overrides):
            kwargs['cleanup'] = None

        return type(self)(**kwargs)

    @abstractmethod
    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional['RunMode'] = None,
        *,
        frequencies=None,
        source_waveform=None,
        sample_rate=None,
        output_duration=None,
    ) -> Result:
        """Run the propagation model.

        Every wrapper takes the same first four parameters in the same
        order: the common ``Environment`` / ``Source`` / ``Receiver``
        triple, followed by the optional ``run_mode``.

        Model configuration is **constructor-only** — every model knob
        (e.g. ``RAM(dr=2.0, dz=0.5, np_pade=8)``,
        ``Bellhop(beam_type='B', n_beams=500)``) is set when the model
        instance is created. To sweep parameters, instantiate one model
        per parameter set.

        ``run()`` accepts a fixed keyword-only set: ``frequencies``,
        ``source_waveform``, ``sample_rate``, ``output_duration``. Every
        TIME_SERIES-capable wrapper (Bellhop, Scooter, Kraken,
        OASP, RAM) consumes ``source_waveform`` and ``sample_rate``;
        SPARC warns that they are ignored (it uses its constructor
        ``pulse_type``). ``output_duration`` is the desired output time
        window (seconds); when given, the IFFT-based wrappers zero-pad
        the source waveform internally so the auto-derived broadband
        grid is tight enough (``Δf = 1/output_duration``), and Bellhop
        maps it to ``time_window`` for delay-and-sum synthesis. Models
        with a broadband path consume ``frequencies`` as an explicit
        override for ``source.frequencies``. The Kraken family
        (Kraken) additionally takes ``n_modes``
        to cap the modal set used. No other kwargs are accepted —
        passing one raises :class:`TypeError`.

        Parameters
        ----------
        env : Environment
            Ocean environment.
        source : Source
            Acoustic source.
        receiver : Receiver
            Receiver grid.
        run_mode : RunMode, optional
            Output type to compute. ``None`` selects the model's natural
            default (typically ``RunMode.COHERENT_TL``). Each wrapper's
            :attr:`supported_modes` lists what it accepts.
        frequencies : array-like, optional
            Keyword-only. Explicit broadband frequency grid (Hz), overriding
            ``source.frequencies`` for models with a broadband path.
        source_waveform : array-like, optional
            Keyword-only. Source time series for ``RunMode.TIME_SERIES``
            synthesis (consumed with ``sample_rate``).
        sample_rate : float, optional
            Keyword-only. Sample rate (Hz) of ``source_waveform``.
        output_duration : float, optional
            Keyword-only. Desired output time-window length (s); IFFT-based
            wrappers zero-pad so ``Δf = 1/output_duration``.

        Returns
        -------
        result : Result
            One of the typed :mod:`uacpy.core.results` subclasses
            (``Field``, ``Arrivals``, ``Modes``, …) determined
            by ``run_mode`` and the model.
        """
        pass

    # Modes that consume exactly one source frequency. Multi-frequency
    # Source passed to one of these is a configuration error — the user
    # should pick BROADBAND/TIME_SERIES, or REFLECTION/COVARIANCE/REPLICA
    # for the OASES family that genuinely supports multi-freq sweeps.
    _SINGLE_FREQUENCY_MODES: 'frozenset[RunMode]' = frozenset({
        RunMode.COHERENT_TL, RunMode.INCOHERENT_TL, RunMode.SEMICOHERENT_TL,
        RunMode.RAYS, RunMode.EIGENRAYS, RunMode.ARRIVALS, RunMode.MODES,
    })

    def _require_timeseries_signal(
        self,
        run_mode: 'RunMode',
        source_waveform,
        sample_rate,
    ) -> None:
        """Raise :class:`ConfigurationError` when the caller asked for a
        :attr:`RunMode.TIME_SERIES` result but did not supply both
        ``source_waveform`` and ``sample_rate``.

        Used by every wrapper that synthesises p(t) from a broadband
        transfer function (Bellhop, RAM, Scooter, Kraken, OASP).
        SPARC has its own pulse mechanism (``pulse_type``) and does not
        call this helper.
        """
        if run_mode == RunMode.TIME_SERIES and (
            source_waveform is None or sample_rate is None
        ):
            raise ConfigurationError(
                f"{self.model_name}.run(run_mode=TIME_SERIES) requires "
                f"source_waveform and sample_rate. For the broadband "
                f"transfer function H(f), use run_mode=RunMode.BROADBAND."
            )
        if run_mode == RunMode.TIME_SERIES and source_waveform is not None:
            wf = np.asarray(source_waveform)
            if not np.all(np.isfinite(wf)):
                raise ConfigurationError(
                    f"{self.model_name}.run(run_mode=TIME_SERIES): "
                    "source_waveform contains non-finite values (NaN/inf)."
                )
            if np.iscomplexobj(wf) and not np.allclose(wf.imag, 0.0):
                raise ConfigurationError(
                    f"{self.model_name}.run(run_mode=TIME_SERIES): "
                    "source_waveform must be a real pressure pulse; got a "
                    "complex array with a non-zero imaginary part."
                )

    def _pad_waveform_to_duration(
        self, source_waveform, sample_rate, output_duration,
    ):
        """Zero-pad ``source_waveform`` so its duration is at least
        ``output_duration`` seconds. Returns the (possibly padded) array
        unchanged when ``output_duration`` is ``None`` or already met.

        Used by every IFFT-based TIME_SERIES wrapper so the user can
        request a longer output than the source pulse without having to
        pre-pad: ``Field.synthesize_time_series`` sets output duration
        = waveform duration, and the auto-derived broadband grid uses
        ``Δf = 1 / waveform_duration``.
        """
        if (
            output_duration is None
            or source_waveform is None
            or sample_rate is None
            or sample_rate <= 0
        ):
            return source_waveform
        wf = np.asarray(source_waveform, dtype=float).ravel()
        n_needed = int(np.ceil(float(output_duration) * float(sample_rate)))
        if wf.size >= n_needed:
            return source_waveform
        pad = np.zeros(n_needed - wf.size, dtype=wf.dtype)
        return np.concatenate([wf, pad])

    def _resolve_broadband_frequencies(
        self,
        source: 'Source',
        frequencies,
        *,
        n_freqs: Optional[int] = None,
        bandwidth_factor: Optional[float] = None,
    ) -> np.ndarray:
        """Resolve the BROADBAND frequency grid.

        Explicit ``frequencies=`` wins. Otherwise a multi-element
        ``source.frequencies`` *is* the band and is used as-is. A single
        centre frequency expands to ``n_freqs`` bins spanning
        ``fc·(1 ± bandwidth_factor/2)``.
        """
        from uacpy.core.constants import (
            DEFAULT_BROADBAND_N_FREQS,
            DEFAULT_BROADBAND_BANDWIDTH_FACTOR,
        )
        if frequencies is not None:
            return np.asarray(frequencies, dtype=float)
        src_f = np.atleast_1d(np.asarray(source.frequencies, dtype=float))
        if src_f.size > 1:
            return src_f
        if n_freqs is None:
            n_freqs = DEFAULT_BROADBAND_N_FREQS
        if bandwidth_factor is None:
            bandwidth_factor = DEFAULT_BROADBAND_BANDWIDTH_FACTOR
        fc = float(src_f[0])
        half_bw = 0.5 * float(bandwidth_factor)
        return np.linspace(
            max(1.0, fc * (1.0 - half_bw)),
            fc * (1.0 + half_bw),
            int(n_freqs),
        )

    def _resolve_time_series_frequencies(
        self,
        run_mode: 'RunMode',
        frequencies,
        source_waveform,
        sample_rate,
        threshold_db: float = -40.0,
    ):
        """Resolve the broadband frequency grid for TIME_SERIES dispatch.

        When ``run_mode == TIME_SERIES`` and the caller did not pass
        ``frequencies=``, derive one from the source waveform: Δf =
        ``sample_rate / n_samples`` (= 1 / waveform duration), band edges
        from the spectral support above ``threshold_db`` below the peak
        (default −40 dB). Pinning ``frequencies=`` skips derivation.
        Other run-modes pass through unchanged.

        Returns the resolved ndarray (uniformly spaced, Hz), or ``None``
        when no override applies. Emits a single ``UserWarning`` so the
        user sees what band/Δf were picked.
        """
        if (
            run_mode != RunMode.TIME_SERIES
            or frequencies is not None
            or source_waveform is None
            or sample_rate is None
        ):
            return frequencies
        wf = np.asarray(source_waveform, dtype=float).ravel()
        n = wf.size
        if n < 2 or sample_rate <= 0:
            return frequencies
        fs = float(sample_rate)
        df = fs / n
        spectrum = np.abs(np.fft.rfft(wf))
        src_freqs = np.fft.rfftfreq(n, 1.0 / fs)
        peak = spectrum.max()
        if peak <= 0:
            raise ConfigurationError(
                f"{self.model_name}.run(run_mode=TIME_SERIES): "
                f"source_waveform is identically zero."
            )
        threshold = peak * 10.0 ** (threshold_db / 20.0)
        significant = spectrum >= threshold
        if not significant.any():
            raise ConfigurationError(
                f"{self.model_name}.run(run_mode=TIME_SERIES): "
                f"source_waveform has no spectral content above "
                f"{threshold_db} dB."
            )
        i_lo = int(np.argmax(significant))
        i_hi = len(significant) - 1 - int(np.argmax(significant[::-1]))
        f_min = max(float(src_freqs[i_lo]), df)
        f_max = float(src_freqs[i_hi])
        if f_max <= f_min:
            f_max = f_min + df
        n_freqs = int(round((f_max - f_min) / df)) + 1
        # A short pulse gives a coarse Δf = 1/duration, so a narrow band can
        # derive only 2-3 bins — too few for the band-edge taper to leave an
        # interior, and too few to represent an arrival at all. Subdivide Δf
        # (equivalent to zero-padding the pulse) rather than hand back a grid
        # the synthesis cannot use.
        refined = max(n_freqs, _MIN_TIMESERIES_FREQS)
        derived = np.linspace(f_min, f_max, refined)
        note = ""
        if refined != n_freqs:
            note = (f" Δf refined to {(f_max - f_min) / (refined - 1):.4g} Hz "
                    f"so the {n_freqs}-bin band resolves an arrival.")
        warnings.warn(
            f"{self.model_name}.run(run_mode=TIME_SERIES): no "
            f"`frequencies=` passed; auto-derived {refined} freqs from "
            f"the source waveform ({f_min:.2f}-{f_max:.2f} Hz, "
            f"Δf={df:.4g} Hz, threshold {threshold_db:.0f} dB).{note} Pass "
            f"`frequencies=` to silence.",
            UserWarning, stacklevel=3,
        )
        return derived

    def _prepare_timeseries(
        self, run_mode, source, frequencies, source_waveform, sample_rate,
        output_duration=None,
    ):
        """Common TIME_SERIES/BROADBAND dispatch preamble for the IFFT-based
        synthesizers (RAM, Kraken, Scooter, OASP): validate the source
        pulse, zero-pad it to ``output_duration``, and derive the broadband
        frequency grid. Returns ``(source_waveform, frequencies)``.

        Bellhop validates the pulse directly and resolves its own grid, and
        SPARC uses ``pulse_type`` instead, so neither calls this.
        """
        self._require_timeseries_signal(run_mode, source_waveform, sample_rate)
        source_waveform = self._pad_waveform_to_duration(
            source_waveform, sample_rate, output_duration,
        )
        frequencies = self._resolve_time_series_frequencies(
            run_mode, frequencies, source_waveform, sample_rate,
        )
        return source_waveform, frequencies

    def _setup_file_manager(self) -> FileManager:
        """Build the FileManager. ``self.work_dir`` is used as-is (not a
        parent); when ``None``, a fresh temp dir is created.

        Auto-creates the user-pinned ``work_dir`` if it doesn't exist
        yet, so callers can construct ``Model(work_dir='./out')`` without
        a separate ``mkdir`` step.
        """
        if self.work_dir is not None:
            # base_dir is the *parent*: FileManager validates that it exists,
            # while ``adopt_work_dir`` keys ownership on whether work_dir
            # itself already existed, so it has to do that mkdir itself.
            parent = Path(self.work_dir).parent
            parent.mkdir(parents=True, exist_ok=True)
            fm = FileManager(
                use_tmpfs=False,
                base_dir=parent,
                prefix=f'{self.model_name.lower()}_',
                cleanup=getattr(self, 'cleanup', False),
            )
            # Adopted, not owned: cleanup may remove only what this run adds.
            fm.adopt_work_dir(self.work_dir)
        else:
            fm = FileManager(
                use_tmpfs=self.use_tmpfs,
                base_dir=None,
                prefix=f'{self.model_name.lower()}_',
                cleanup=True,
            )
            fm.create_work_dir()

        return fm

    def _log(self, message: str, level: str = "info"):
        """Emit a tagged line through :func:`uacpy._log.log_message`.
        ``WARN`` / ``ERROR`` always print; ``INFO`` / ``DEBUG`` only when
        ``self.verbose``."""
        from uacpy._log import log_message
        log_message(
            self.model_name, message,
            verbose=self.verbose, level=level,
        )

    def validate_inputs(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional['RunMode'] = None,
    ):
        """
        Validate inputs against the environment + the resolved run mode.

        Parameters
        ----------
        env : Environment
            Environment to validate against.
        source : Source
            Source to validate.
        receiver : Receiver
            Receiver to validate.
        run_mode : RunMode, optional
            Resolved run mode. When passed, single-frequency modes
            (``COHERENT_TL``, ``RAYS``, ``MODES``, …) refuse a Source
            with more than one frequency.

        Raises
        ------
        InvalidDepthError
            If source depths exceed the model's resolvable depth.
        ConfigurationError
            If source/receiver depths are negative, if ``run_mode`` is a
            single-frequency mode and ``source`` carries multiple
            frequencies, if ``source.source_type`` is outside
            ``_supported_source_types``, or if ``source.beam_pattern`` is set
            on a model that does not read one.
        """
        if (run_mode is not None
                and run_mode in self._SINGLE_FREQUENCY_MODES
                and len(source.frequencies) > 1):
            raise ConfigurationError(
                f"{self.model_name}.run(run_mode={run_mode.name}) takes a "
                f"single source frequency; got {len(source.frequencies)}: "
                f"{list(source.frequencies)}. For broadband H(f) use "
                f"RunMode.BROADBAND, and for time-domain p(t) use "
                f"RunMode.TIME_SERIES."
            )

        if (not self._supports_multi_source_depth
                and len(np.atleast_1d(source.depths)) > 1):
            raise ConfigurationError(
                f"{self.model_name} takes a single source depth per run; "
                f"got {len(source.depths)}: {list(source.depths)}. Loop "
                f"over Sources externally for multi-depth runs."
            )

        if source.source_type not in self._supported_source_types:
            raise ConfigurationError(
                f"{self.model_name} does not support "
                f"Source(source_type={source.source_type!r}); it supports "
                f"{sorted(self._supported_source_types)}."
            )

        if (source.beam_pattern is not None
                and not self._supports_source_beam_pattern):
            raise ConfigurationError(
                f"{self.model_name} does not read a source beam pattern; "
                f"drop Source(beam_pattern=...) or use Bellhop or Kraken."
            )

        # receiver_type='line' is honoured by the input-side checks (seafloor
        # comparison, depth clipping) but no model's result assembly collapses
        # the depth x range grid to the paired samples, so a 'line' request
        # silently returns the full cross-product. Refuse rather than hand back
        # a shape the caller did not ask for.
        if receiver.receiver_type == 'line':
            raise ConfigurationError(
                f"{self.model_name}: receiver_type='line' is not implemented — "
                f"every model returns the full depth x range grid, so the "
                f"paired (depths[i], ranges[i]) sampling you asked for would "
                f"be silently ignored. Use receiver_type='grid' and index the "
                f"diagonal yourself: "
                f"tl[np.arange(len(depths)), np.arange(len(ranges))]."
            )

        resolvable_depth = self._max_receiver_depth(env)

        # The source injects energy into the medium, so it must sit within
        # what the model resolves — placing it below is a hard error.
        # Receivers are outputs: ones below the resolvable depth are
        # accepted and return the model's below-domain value (transmitted /
        # evanescent field, or NaN inside a PE absorbing layer); the
        # warning helpers below surface that rather than rejecting it.
        if np.any(source.depths > resolvable_depth):
            raise InvalidDepthError(
                float(source.depths.max()), resolvable_depth, "Source",
            )

        if np.any(source.depths < 0):
            raise ConfigurationError("Source depths must be non-negative")

        # A source exactly at z = 0 sits on the pressure-release sea surface,
        # where the boundary forces p ≈ 0: a field run then returns a
        # degenerate result (a null / saturated-TL sentinel, or — in RAM — an
        # unphysical normalisation) that a valid-looking ``Source(depths=0)``
        # hides. Reflection coefficients and mode shapes don't propagate a
        # source field, so the warning doesn't apply to those.
        if (run_mode not in (RunMode.REFLECTION, RunMode.MODES)
                and np.any(np.asarray(source.depths) == 0.0)):
            warnings.warn(
                f"{self.model_name}: a source at depth 0 m is on the "
                f"pressure-release sea surface, where the field is ~0 — the "
                f"result is degenerate (null / saturated TL, model-dependent). "
                f"Use a small positive depth (e.g. 1 m).",
                UserWarning, stacklevel=3,
            )

        if receiver.depth_min < 0:
            raise ConfigurationError("Receiver depths must be non-negative")

        self._warn_receiver_below_resolvable(env, receiver, resolvable_depth)
        self._check_per_range_receiver_depth(env, receiver)
        self._warn_on_range_coverage(env, receiver)

    def _warn_receiver_below_resolvable(
        self, env: 'Environment', receiver: 'Receiver',
        resolvable_depth: float,
    ) -> None:
        """Flat-bathymetry counterpart to
        :meth:`_check_per_range_receiver_depth`: warn — never raise — when a
        receiver lies below the depth this model resolves the field at.
        Such receivers are accepted; every model degrades gracefully there
        (Bellhop transmitted field, Kraken evanescent tail, RAM NaN in the
        absorbing layer, Scooter/SPARC clipped to the sediment). The
        range-dependent case is handled per-range by
        :meth:`_check_per_range_receiver_depth`.
        """
        if env.has_range_dependent_bathymetry:
            return
        if receiver.depth_max > resolvable_depth:
            warnings.warn(
                f"{self.model_name}: receiver depth "
                f"{float(receiver.depth_max):.1f} m is below the model's "
                f"resolvable depth ({resolvable_depth:.1f} m). It is "
                f"accepted; the result there reflects the model's "
                f"below-domain behaviour (transmitted / evanescent field, "
                f"or NaN inside a PE absorbing layer).",
                UserWarning, stacklevel=3,
            )

    def _check_per_range_receiver_depth(
        self, env: 'Environment', receiver: 'Receiver',
    ) -> None:
        """Emit a ``UserWarning`` if any receiver sits below the local
        seafloor in a range-dependent bathymetry. Below-seafloor receivers
        are accepted, not rejected: several models (Bellhop, RAM) resolve
        them natively (transmitted field, PE absorbing region). The
        flat-bathy case is handled by
        :meth:`_warn_receiver_below_resolvable`.

        ``receiver_type='line'`` pairs depths[i] with ranges[i] (1-D
        compare); ``'grid'`` evaluates the depth × range cross-product.
        """
        if not env.has_range_dependent_bathymetry:
            return
        depths = np.atleast_1d(receiver.depths).astype(float)
        ranges = np.atleast_1d(receiver.ranges).astype(float)
        seafloor = np.asarray(env.bathymetry.eval(range=ranges), dtype=float)

        if receiver.receiver_type == 'line':
            mask = depths > seafloor
            row_ranges = ranges
            row_floors = seafloor
        else:
            mask = depths[:, None] > seafloor[None, :]
            row_ranges = np.broadcast_to(ranges[None, :], mask.shape)
            row_floors = np.broadcast_to(seafloor[None, :], mask.shape)
            depths = np.broadcast_to(depths[:, None], mask.shape)

        if np.any(mask):
            flat = int(np.argmax(mask))
            r = float(row_ranges.ravel()[flat])
            z = float(depths.ravel()[flat])
            sf = float(row_floors.ravel()[flat])
            warnings.warn(
                f"{self.model_name}: receiver at "
                f"(range={r:.1f} m, depth={z:.1f} m) sits below the local "
                f"seafloor ({sf:.1f} m). Results at that point will "
                f"reflect the model's below-bottom behaviour (e.g. "
                f"infinite TL, PE absorbing layer).",
                UserWarning, stacklevel=3,
            )

    def _warn_on_range_coverage(
        self, env: 'Environment', receiver: 'Receiver',
    ) -> None:
        """Emit one ``UserWarning`` per range-dependent axis whose extent
        falls short of ``receiver.range_max``. Constant extrapolation is
        what every downstream writer / interpolator does in that case;
        this surfaces it instead of leaving it silent.
        """
        r_target = float(receiver.range_max)
        if r_target <= 0:
            return

        def _check(axis_name: str, axis_max: float) -> None:
            if axis_max < r_target:
                warnings.warn(
                    f"{self.model_name}: {axis_name} extent "
                    f"({axis_max:.1f} m) is shorter than receiver.range_max "
                    f"({r_target:.1f} m); values beyond {axis_max:.1f} m are "
                    f"constant-extrapolated from the last sample.",
                    UserWarning, stacklevel=3,
                )

        if env.has_range_dependent_bathymetry:
            _check("env.bathymetry", float(env.bathymetry.ranges[-1]))
        if env.ssp.is_range_dependent:
            _check("env.ssp.ranges", float(env.ssp.ranges[-1]))
        if env.bottom.is_range_dependent:
            _check("env.bottom.ranges", float(env.bottom.ranges[-1]))
        if env.altimetry is not None and env.altimetry.n_ranges > 1:
            _check("env.altimetry ranges", float(env.altimetry.ranges[-1]))

    def compute_tl(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        *,
        run_mode: 'RunMode' = None,
    ) -> Result:
        """Compute transmission loss (thin wrapper around ``run``).

        Parameters
        ----------
        env : Environment
            Ocean environment.
        source : Source
            Acoustic source.
        receiver : Receiver
            Receiver grid. Required — depth/range resolution is a physical
            decision and is not auto-generated.
        run_mode : RunMode, optional
            ``COHERENT_TL`` (default), ``INCOHERENT_TL`` or
            ``SEMICOHERENT_TL``. Other modes raise — call ``model.run()``
            directly for those.

        Returns
        -------
        result : Result
            Transmission loss field.

        Examples
        --------
        >>> bellhop = Bellhop()
        >>> rcv = uacpy.Receiver(depths=np.linspace(0, env.depth, 50),
        ...                       ranges=np.linspace(100, 10_000, 100))
        >>> tl = bellhop.compute_tl(env, source, rcv)
        """
        if not self.supports_mode(RunMode.COHERENT_TL):
            raise UnsupportedFeatureError(
                self.model_name,
                "transmission loss computation",
                alternatives=['Bellhop', 'Kraken', 'RAM', 'Scooter', 'OAST'],
            )
        if run_mode is None:
            run_mode = RunMode.COHERENT_TL
        if run_mode not in (
            RunMode.COHERENT_TL, RunMode.INCOHERENT_TL, RunMode.SEMICOHERENT_TL,
        ):
            raise ConfigurationError(
                f"compute_tl() got run_mode={run_mode}; only COHERENT_TL / "
                f"INCOHERENT_TL / SEMICOHERENT_TL are accepted. Call "
                f"{self.model_name}.run(run_mode=…) for other modes."
            )
        return self.run(env, source, receiver, run_mode=run_mode)

    def compute_rays(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
    ) -> Result:
        """Compute ray paths (thin wrapper around ``run``).

        ``receiver`` is required — the receiver grid defines the ray-box
        extent and recording locations and is not auto-generated.

        Examples
        --------
        >>> bellhop = Bellhop()
        >>> rcv = uacpy.Receiver(depths=np.array([env.depth / 2]),
        ...                       ranges=np.linspace(0, 10_000, 50))
        >>> rays = bellhop.compute_rays(env, source, rcv)
        """
        if not self.supports_mode(RunMode.RAYS):
            raise UnsupportedFeatureError(
                self.model_name,
                "ray path computation",
                alternatives=['Bellhop'],
            )
        return self.run(env, source, receiver, run_mode=RunMode.RAYS)

    def compute_arrivals(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
    ) -> Result:
        """
        Compute the arrival structure (convenience wrapper around ``run``).

        Parameters
        ----------
        env : Environment
            Ocean environment.
        source : Source
            Acoustic source.
        receiver : Receiver
            Receiver array.

        Returns
        -------
        result : Result
            Arrival data.

        Raises
        ------
        UnsupportedFeatureError
            If the model does not support arrival computation.

        Examples
        --------
        >>> bellhop = Bellhop()
        >>> arrivals = bellhop.compute_arrivals(env, source, receiver)
        """
        if not self.supports_mode(RunMode.ARRIVALS):
            raise UnsupportedFeatureError(
                self.model_name,
                "arrival computation",
                alternatives=['Bellhop']
            )
        return self.run(env, source, receiver, run_mode=RunMode.ARRIVALS)

    def compute_modes(
        self,
        env: Environment,
        source: Source,
        n_modes: int = None,
    ) -> Result:
        """
        Compute normal modes (convenience wrapper around ``run``).

        Parameters
        ----------
        env : Environment
            Ocean environment. A range-dependent env is collapsed to range-
            independent (with a warning) before the mode solve.
        source : Source
            Acoustic source (used for frequency).
        n_modes : int, optional
            Number of modes to compute. If ``None``, all modes are computed.

        Returns
        -------
        result : Result
            :class:`Modes` instance.

        Raises
        ------
        UnsupportedFeatureError
            If the model does not support mode computation.

        Notes
        -----
        This is the one ``compute_*`` wrapper that takes no ``receiver``:
        normal modes are receiver-independent depth eigenfunctions, so there is
        nothing for the caller to position. A placeholder receiver is fabricated
        internally only to satisfy the shared ``run()`` recipe (the Kraken
        backend overrides it with its own dense depth grid).

        Examples
        --------
        >>> kraken = Kraken()
        >>> modes = kraken.compute_modes(env, source, n_modes=50)
        >>> wavenumbers = modes.k
        >>> mode_shapes = modes.phi
        """
        if not self.supports_mode(RunMode.MODES):
            raise UnsupportedFeatureError(
                self.model_name,
                "normal mode computation",
                alternatives=['Kraken']
            )

        if n_modes is not None and not isinstance(
                n_modes, (int, float, np.integer, np.floating)):
            raise ConfigurationError(
                f"compute_modes: n_modes must be an int or None; got "
                f"{type(n_modes).__name__}. Unlike the other compute_* "
                f"wrappers, compute_modes takes no receiver (normal modes are "
                f"receiver-independent) — its third argument is n_modes. Pass "
                f"it by keyword: compute_modes(env, source, n_modes=...).")

        if env.is_range_dependent:
            # Mode solvers (the Kraken backends) collapse the environment via
            # ``collapse={'bathymetry': …}`` and warn, rather than reject —
            # same pattern as OAST/OASP/Scooter/SPARC.
            env = self._project_environment(env)

        return self._compute_modes_impl(env, source, n_modes)

    def _compute_modes_impl(self, env, source, n_modes):
        """
        Model-specific mode computation implementation.

        Caller (compute_modes) already guaranteed RunMode.MODES is supported.
        """
        dummy_receiver = Receiver(depths=[0.0], ranges=[0.0])
        return self.run(
            env, source, dummy_receiver,
            run_mode=RunMode.MODES, n_modes=n_modes,
        )

    def compute_eigenrays(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
    ) -> Result:
        """Compute eigenrays — rays that arrive at the receiver(s).

        Thin wrapper around ``run(run_mode=RunMode.EIGENRAYS)``. Returns
        the raw :class:`Rays` from the solver. For a single-point target
        build a 1-point ``Receiver`` first:

        >>> receiver = uacpy.Receiver(depths=[30.0], ranges=[2000.0])
        >>> rays = bellhop.compute_eigenrays(env, source, receiver)
        >>> close = rays.top_n_by_miss(8).truncate_at_receiver()
        >>> direct = rays.filter_by_bounces(kind='direct')
        >>> within = rays.filter_by_miss_distance(max_miss=15.0)
        """
        if not self.supports_mode(RunMode.EIGENRAYS):
            raise UnsupportedFeatureError(
                self.model_name,
                "eigenray computation",
                alternatives=['Bellhop'],
            )
        return self.run(env, source, receiver, run_mode=RunMode.EIGENRAYS)

    def compute_reflection(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
    ) -> Result:
        """Compute plane-wave reflection coefficients.

        Dispatches to ``run(run_mode=RunMode.REFLECTION)``. Models that
        do not declare ``RunMode.REFLECTION`` in ``supported_modes``
        (everything except Bounce, OASR, OASES) raise
        :class:`UnsupportedFeatureError`.
        """
        if not self.supports_mode(RunMode.REFLECTION):
            raise UnsupportedFeatureError(
                self.model_name,
                "reflection coefficient computation",
                alternatives=['Bounce', 'OASR'],
            )
        return self.run(env, source, receiver, run_mode=RunMode.REFLECTION)

    def compute_time_series(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        *,
        source_waveform=None,
        sample_rate=None,
        output_duration=None,
    ) -> Result:
        """Compute time-domain pressure p(t) at the receiver(s).

        Forwards ``source_waveform``, ``sample_rate`` and
        ``output_duration`` to ``run(run_mode=RunMode.TIME_SERIES)``.
        ``output_duration`` (seconds) sets the synthesised time window for
        the broadband synthesizers (Bellhop / RAM / Scooter / Kraken /
        OASP) — e.g. the length of an animation; SPARC ignores all three
        (it builds p(t) from its native ``pulse_type`` and ``t_max``), and
        every other TIME_SERIES model requires the waveform/rate.
        """
        if not self.supports_mode(RunMode.TIME_SERIES):
            raise UnsupportedFeatureError(
                self.model_name,
                "time-series computation",
                alternatives=['Bellhop', 'Kraken', 'RAM', 'Scooter',
                              'OASP', 'SPARC'],
            )
        return self.run(
            env, source, receiver,
            run_mode=RunMode.TIME_SERIES,
            source_waveform=source_waveform,
            sample_rate=sample_rate,
            output_duration=output_duration,
        )

    def compute_transfer_function(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        *,
        frequencies=None,
    ) -> Result:
        """Compute broadband complex transfer function H(f).

        Dispatches to ``run(run_mode=RunMode.BROADBAND)``. Pass
        ``frequencies=`` to override ``source.frequencies`` for the
        sweep.
        """
        if not self.supports_mode(RunMode.BROADBAND):
            raise UnsupportedFeatureError(
                self.model_name,
                "broadband transfer-function computation",
                alternatives=['Bellhop', 'Kraken', 'RAM',
                              'Scooter', 'OASP'],
            )
        return self.run(
            env, source, receiver,
            run_mode=RunMode.BROADBAND,
            frequencies=frequencies,
        )

    def compute_covariance(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
    ) -> Result:
        """Compute hydrophone-array covariance matrix C(f, i, j).

        Dispatches to ``run(run_mode=RunMode.COVARIANCE)``. Currently
        OASN is the only model declaring this mode.
        """
        if not self.supports_mode(RunMode.COVARIANCE):
            raise UnsupportedFeatureError(
                self.model_name,
                "covariance-matrix computation",
                alternatives=['OASN'],
            )
        return self.run(env, source, receiver, run_mode=RunMode.COVARIANCE)

    def compute_replicas(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
    ) -> Result:
        """Compute replica fields at the array elements per candidate
        source position (matched-field-processing templates).

        Dispatches to ``run(run_mode=RunMode.REPLICA)``. Currently
        OASN is the only model declaring this mode.
        """
        if not self.supports_mode(RunMode.REPLICA):
            raise UnsupportedFeatureError(
                self.model_name,
                "replica-field computation",
                alternatives=['OASN'],
            )
        return self.run(env, source, receiver, run_mode=RunMode.REPLICA)

    def _find_executable_in_paths(
        self,
        names,
        bin_subdirs=None,
        dev_subdir: Optional[str] = None,
        try_exe_suffix: bool = True,
    ) -> Path:
        """
        Find a model executable by searching standard locations.

        Search order:
            1. uacpy/bin/<bin_subdir>/<name>[+.exe] for each combination
            2. uacpy/third_party/<dev_subdir>/bin (development location)
            3. System PATH

        Parameters
        ----------
        names : str or list of str
            Executable name(s) to try, in preference order.
        bin_subdirs : str or list of str, optional
            Subdirectory/ies under uacpy/bin/. Default 'oalib'.
        dev_subdir : str, optional
            Subdirectory under uacpy/third_party/ (e.g. 'Acoustics-Toolbox/Kraken',
            'oases'). If given, also checks <dev_subdir>/bin and <dev_subdir>/.
        try_exe_suffix : bool, optional
            If True, also try "<name>.exe". Default True.

        Raises
        ------
        ExecutableNotFoundError
        """
        if isinstance(names, str):
            names = [names]
        if bin_subdirs is None:
            bin_subdirs = ['oalib']
        elif isinstance(bin_subdirs, str):
            bin_subdirs = [bin_subdirs]

        base_dir = Path(__file__).parent.parent
        candidates = []
        for name in names:
            variants = [name]
            if try_exe_suffix and not name.endswith('.exe'):
                variants.append(name + '.exe')
            for v in variants:
                for sd in bin_subdirs:
                    candidates.append(base_dir / 'bin' / sd / v)
                if dev_subdir:
                    candidates.append(base_dir / 'third_party' / dev_subdir / 'bin' / v)
                    candidates.append(base_dir / 'third_party' / dev_subdir / v)

        for path in candidates:
            if path.exists():
                return path

        for name in names:
            variants = [name]
            if try_exe_suffix and not name.endswith('.exe'):
                variants.append(name + '.exe')
            for v in variants:
                found = shutil.which(v)
                if found:
                    return Path(found)

        raise ExecutableNotFoundError(
            self.model_name,
            names[0],
            search_paths=[str(p) for p in candidates],
        )

    def _run_subprocess(
        self,
        cmd,
        cwd,
        timeout: Optional[float] = None,
        stdin_input: Optional[str] = None,
        env: Optional[dict] = None,
        check: bool = True,
    ):
        """
        Run an external binary and raise ModelExecutionError on failure.

        All Fortran acoustic binaries are spawned through this helper so that
        failures surface as ``ModelExecutionError`` with stdout/stderr
        attached, and so every child inherits a raised ``RLIMIT_STACK``.
        Several Acoustics-Toolbox binaries (notably SPARC) statically
        allocate large COMPLEX arrays on the stack; the default 8 MB Linux
        soft stack segfaults them on first use.

        Parameters
        ----------
        cmd : list
            Command argv (str-able elements).
        cwd : path-like
            Working directory for the subprocess.
        timeout : float, optional
            Max seconds before raising.
        stdin_input : str, optional
            Text fed to the subprocess's stdin.
        env : dict, optional
            Environment variables for the subprocess.
        check : bool, optional
            If True (default), raise on non-zero return code.

        Returns
        -------
        subprocess.CompletedProcess
        """
        if timeout is None:
            timeout = getattr(self, 'timeout', 600.0)
        cmd_str = ' '.join(str(c) for c in cmd)
        self._log(f"Running: {cmd_str}", level='debug')

        # start_new_session puts the child in its own process group so a
        # timeout can SIGTERM the whole tree, not just the direct child.
        proc = None
        try:
            proc = subprocess.Popen(
                [str(c) for c in cmd],
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE if stdin_input is not None else None,
                text=True,
                env=env,
                start_new_session=(os.name == 'posix'),
            )
            stdout, stderr = proc.communicate(input=stdin_input, timeout=timeout)
            result = subprocess.CompletedProcess(
                proc.args, proc.returncode, stdout, stderr,
            )
        except FileNotFoundError as e:
            raise ModelExecutionError(
                self.model_name, return_code=-1,
                stdout=None, stderr=f"Executable not found: {e}",
            ) from e
        except subprocess.TimeoutExpired as e:
            # Kill the whole process group, not just the direct child.
            if proc is not None and os.name == 'posix':
                try:
                    pgid = os.getpgid(proc.pid)
                    os.killpg(pgid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(pgid, signal.SIGKILL)
                        proc.wait()
                except (ProcessLookupError, PermissionError):
                    proc.kill()
                    proc.wait()
            elif proc is not None:
                proc.kill()
                proc.wait()
            raise ModelExecutionError(
                self.model_name, return_code=-1,
                stdout=(e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout),
                stderr=f"Timed out after {timeout}s",
                timed_out=True,
            ) from e

        if check and result.returncode != 0:
            raise ModelExecutionError(
                self.model_name,
                return_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        return result

    @staticmethod
    def _has_shear(boundary) -> bool:
        """True if ``boundary`` carries any non-zero shear speed. Accepts a
        :class:`Bottom` or a surface :class:`BoundaryProperties`."""
        from uacpy.core.bottom import Bottom
        from uacpy.core.surface import Surface
        if boundary is None:
            return False
        if isinstance(boundary, (Bottom, Surface)):
            return boundary.is_elastic
        return getattr(boundary, 'shear_speed', 0.0) > 0

    @staticmethod
    def _collapse_elastic_boundary(boundary, method: str):
        """Collapse elastic shear on ``boundary`` per ``method``.

        ``'fluid'``  : zero shear_speed and shear_attenuation; keep cp / ρ / α.
        ``'vacuum'`` : replace with a vacuum boundary.

        Accepts a :class:`Bottom` (every column's layers + half-space) or a
        surface :class:`BoundaryProperties`, returning the same kind.
        """
        from uacpy.core.bottom import BoundaryProperties, Bottom
        from uacpy.core.surface import Surface

        def _zero_shear(b):
            if hasattr(b, 'shear_speed'):
                b.shear_speed = 0.0
            if hasattr(b, 'shear_attenuation'):
                b.shear_attenuation = 0.0

        if method not in ('fluid', 'vacuum'):
            raise ConfigurationError(
                f"Unknown elastic collapse method {method!r}. Use "
                "'fluid' or 'vacuum'."
            )
        if isinstance(boundary, Bottom):
            if method == 'vacuum':
                return Bottom.from_halfspace(
                    BoundaryProperties(acoustic_type='vacuum'))
            b = _copy.deepcopy(boundary)
            for col in b.columns:
                for layer in col.layers:
                    _zero_shear(layer)
                _zero_shear(col.halfspace)
            return b
        if isinstance(boundary, Surface):
            if method == 'vacuum':
                return Surface(properties=[
                    BoundaryProperties(acoustic_type='vacuum')])
            b = _copy.deepcopy(boundary)
            for p in b.properties:
                _zero_shear(p)
            return b
        if method == 'vacuum':
            return BoundaryProperties(acoustic_type='vacuum')
        b = _copy.deepcopy(boundary)
        _zero_shear(b)
        return b

    def _project_environment(self, env: 'Environment') -> 'Environment':
        """Return a copy of ``env`` with every unsupported feature collapsed.

        Each per-feature axis is checked against the matching
        ``_supports_*`` flag and reduced via the matching key in the
        configured ``collapse={…}`` dict. Emits one ``UserWarning`` per
        dropped feature.
        """
        e = env.copy()

        if e.altimetry is not None and not self._supports_altimetry:
            method = self._collapse["altimetry"]
            if method != 'drop':
                raise ConfigurationError(
                    f"Unknown collapse['altimetry']={method!r}. "
                    "Currently only 'drop' is supported."
                )
            e.altimetry = None
            warnings.warn(
                f"{self.model_name} does not support sea-surface altimetry; "
                f"using flat surface (collapse['altimetry']={method!r}).",
                UserWarning, stacklevel=3,
            )

        if e.surface.is_range_dependent and not self._supports_range_dependent_surface:
            method = self._collapse["surface"]
            e.surface = e.surface.collapse(method)
            warnings.warn(
                f"{self.model_name} does not support range-dependent surface "
                f"properties (e.g. a marginal ice zone); collapsed to a single "
                f"boundary (collapse['surface']={method!r}).",
                UserWarning, stacklevel=3,
            )

        surf_sigma = _max_roughness(
            getattr(e.surface, 'properties', None) or [e.surface])
        if surf_sigma and not self._supports_rough_surface:
            e.surface = _smooth_surface(e.surface)
            warnings.warn(
                f"{self.model_name} cannot model a rough sea surface "
                f"(its solver rejects a non-zero interfacial sigma); "
                f"env.surface.roughness={surf_sigma:g} m dropped. Use Kraken "
                f"or Scooter to keep it.",
                UserWarning, stacklevel=3,
            )

        bot_sigma = _max_roughness(
            [c.halfspace for c in e.bottom.columns]) if e.bottom else 0.0
        if bot_sigma and not self._supports_rough_bottom:
            e.bottom = _smooth_bottom(e.bottom)
            warnings.warn(
                f"{self.model_name} does not model seabed interfacial "
                f"roughness; env.bottom roughness={bot_sigma:g} m dropped "
                f"(its solver either ignores the value or has no place to put "
                f"it). Use Kraken or Scooter to keep it.",
                UserWarning, stacklevel=3,
            )

        if e.has_range_dependent_bathymetry and not self._supports_range_dependent_bathymetry:
            method = self._collapse["bathymetry"]
            new_depth = e.get_representative_depth(method)
            min_d = float(e.bathymetry.depths.min())
            max_d = float(e.bathymetry.depths.max())
            e.bathymetry = Bathymetry(
                ranges=np.array([0.0]), depths=np.array([new_depth]))
            if e.ssp.depths[-1] < new_depth:
                e.ssp = e.ssp.extend_to(new_depth)
            warnings.warn(
                f"{self.model_name} does not support range-dependent "
                f"bathymetry; collapsed to {new_depth:.1f} m "
                f"(method={method!r}, range {min_d:.1f}–{max_d:.1f} m). "
                f"Override via `collapse={{'bathymetry': "
                f"'min'|'median'|'mean'|'max'|'initial'}}`.",
                UserWarning, stacklevel=3,
            )

        if e.has_range_dependent_ssp and not self._supports_range_dependent_ssp:
            method = self._collapse["ssp"]
            e.ssp = e.ssp.collapse(method)
            warnings.warn(
                f"{self.model_name} does not support range-dependent SSP; "
                f"collapsed to 1-D (collapse['ssp']={method!r}).",
                UserWarning, stacklevel=3,
            )

        # Bottom: two orthogonal axes. Collapse the range axis first (to a
        # single column, keeping its layers), then flatten the layer axis if
        # the model can't take layers — leaving, for a model that supports RD
        # but not layers (Bellhop), a range-dependent half-space bottom.
        if e.bottom.is_range_dependent and not self._supports_range_dependent_bottom:
            method = self._collapse["bottom_range"]
            e.bottom = e.bottom.select_range(method)
            warnings.warn(
                f"{self.model_name} does not support range-dependent bottoms; "
                f"reduced to a single column "
                f"(collapse['bottom_range']={method!r}).",
                UserWarning, stacklevel=3,
            )

        if e.bottom.is_layered and not self._supports_layered_bottom:
            method = self._collapse["bottom_layers"]
            e.bottom = e.bottom.collapse(layers=method)
            warnings.warn(
                f"{self.model_name} does not support layered (depth-dependent) "
                f"bottoms; flattened each column to a half-space "
                f"(collapse['bottom_layers']={method!r}).",
                UserWarning, stacklevel=3,
            )

        if not self._supports_elastic_media:
            collapsed_at = []
            if e.surface is not None and self._has_shear(e.surface):
                e.surface = self._collapse_elastic_boundary(
                    e.surface, self._collapse["elastic"],
                )
                collapsed_at.append('surface')
            if e.bottom.is_elastic:
                e.bottom = self._collapse_elastic_boundary(
                    e.bottom, self._collapse["elastic"],
                )
                collapsed_at.append('bottom')
            if collapsed_at:
                method = self._collapse["elastic"]
                where = '/'.join(collapsed_at)
                warnings.warn(
                    f"{self.model_name} does not support elastic media; "
                    f"collapsed shear properties on {where} "
                    f"(collapse['elastic']={method!r}).",
                    UserWarning, stacklevel=3,
                )

        return e

    def _attach_output_paths(
        self,
        result: 'Result',
        work_dir: Path,
        base_name: str,
        *,
        primary_files: tuple = (),
    ) -> None:
        """Attach work-dir output paths to ``result.metadata``.

        When ``self.cleanup`` is True the work dir will be wiped immediately
        after ``run()`` returns, so no keys are written: the absence of a
        ``*_file`` / ``prt_file`` key is the documented signal that the
        directory has been cleaned up (DOCUMENTATION.md §6).

        Otherwise, for each ``(key, suffix)`` in ``primary_files`` set
        ``result.metadata[key] = str(work_dir / f'{base_name}{suffix}')``
        when the file exists. Also set ``'prt_file'`` from the binary's
        diagnostic log when present.
        """
        if self.cleanup:
            return
        for key, suffix in primary_files:
            path = work_dir / f'{base_name}{suffix}'
            if path.exists():
                result.metadata[key] = str(path)
        prt_path = work_dir / f'{base_name}.prt'
        if prt_path.exists():
            result.metadata['prt_file'] = str(prt_path)

    @staticmethod
    def _attach_prt_tail(exc, work_dir, base_name, n_chars: int = 2000):
        """Append the tail of the binary's ``<base>.prt`` log to ``exc``.

        Acoustics-Toolbox binaries dump fatal errors (``*** FATAL ERROR ***``)
        to ``.prt`` instead of stderr; surface that tail on the raised
        ``ModelExecutionError`` so the user sees the actual cause, not just a
        "check the .prt file" pointer.

        Updates ``exc.message`` (what ``UACPYError.__str__`` renders) **and**
        ``exc.stderr`` (a constructor arg, so the tail survives the pickle
        round-trip that ``run_parallel`` relies on — the rebuilt message is
        re-derived from ``stderr``), keeping ``exc.args`` in sync.
        """
        from pathlib import Path as _Path
        tail = read_prt(_Path(work_dir) / f"{base_name}.prt", tail_bytes=n_chars)
        if tail is None:
            return
        block = f"\n\n.prt tail:\n{tail}"
        if getattr(exc, 'message', None) is not None:
            exc.message += block
        if hasattr(exc, 'stderr'):
            exc.stderr = (exc.stderr + block) if exc.stderr else f".prt tail:\n{tail}"
        head = getattr(exc, 'message', None) or (
            f"{exc.args[0]}{block}" if exc.args else f"{exc}{block}")
        exc.args = (head,) + exc.args[1:]

    # ERROUT messages that describe a physical outcome uacpy models
    # explicitly, not a run failure. Below the modal cutoff KRAKEN calls
    # ERROUT, but "no trapped modes at this frequency" is a real answer and is
    # surfaced downstream as a NaN field plus a warning.
    _BENIGN_FORTRAN_FATALS: tuple = (
        'No modes for given phase speed interval',
    )

    def _raise_on_fortran_fatal(self, result, work_dir, base_name):
        """Raise when a binary reported a fatal error but exited 0.

        Acoustics-Toolbox errors funnel through ``ERROUT``
        (``misc/FatalError.f90:18,30``), which writes ``*** FATAL ERROR ***``
        to the ``.prt`` and then ends in ``STOP '<string>'`` — a *character*
        stop code, which gfortran exits **0** for. A return-code test therefore
        never fires, and the binary leaves its output file untouched: with a
        pinned ``work_dir`` the previous run's ``.mod`` / ``.shd`` is still on
        disk and would be read as this run's answer.

        stderr is the authoritative signal because it belongs to this process;
        the ``.prt`` is checked too since it names the actual cause.
        """
        stderr = result.stderr or ''
        prt = read_prt(Path(work_dir) / f"{base_name}.prt")
        # AT ends at ``STOP 'Fatal Error: …'`` and mirrors ``*** FATAL ERROR ***``
        # into the .prt; OASES writes no .prt at all and stops with its own
        # banner, e.g. ``STOP *** CONTOURS REQUIRE NRFR>1 ***``. Both exit 0.
        fatal_stderr = ('Fatal Error' in stderr
                        or ('STOP' in stderr and '***' in stderr))
        if not fatal_stderr and (
                prt is None or '*** FATAL ERROR ***' not in prt):
            return
        if prt and any(m in prt for m in self._BENIGN_FORTRAN_FATALS):
            return
        exc = ModelExecutionError(
            self.model_name, return_code=0, stdout=result.stdout,
            stderr=stderr or "binary reported a fatal error and exited 0",
        )
        self._attach_prt_tail(exc, work_dir, base_name)
        raise exc

    def _run_and_attach_prt(self, cmd, work_dir, base_name, *,
                            timeout: Optional[float] = None,
                            env: Optional[dict] = None):
        """Run a Fortran/AT binary via :meth:`_run_subprocess`, appending the
        ``<base>.prt`` tail to a :class:`ModelExecutionError` on failure and
        logging stdout when verbose. Returns the ``CompletedProcess``. Shared
        by every model's binary-launch wrapper."""
        # A leftover .prt from an earlier run in a pinned work_dir would be
        # read as this run's log, so clear it before launching.
        Path(work_dir).joinpath(f"{base_name}.prt").unlink(missing_ok=True)
        try:
            result = self._run_subprocess(cmd, cwd=work_dir, timeout=timeout, env=env)
        except ModelExecutionError as exc:
            self._attach_prt_tail(exc, work_dir, base_name)
            raise
        self._raise_on_fortran_fatal(result, work_dir, base_name)
        if result.stdout:
            self._log(f"{self.model_name} output:\n{result.stdout}", level='debug')
        return result

    def _result_kwargs(
        self,
        source: 'Source',
        *,
        backend: Optional[str] = None,
        frequencies: Optional[Union[float, np.ndarray]] = None,
        phase_reference: Optional[str] = None,
        **extra,
    ) -> dict:
        """Pre-built kwargs for any :mod:`uacpy.core.results` constructor.

        ``frequencies`` is auto-wrapped to a 1-D ndarray (length ≥ 1) when
        scalar; ``None`` is preserved for time-domain results. Anything in
        ``extra`` is stored on the result's ``metadata`` ad-hoc bag.
        """
        kw = dict(
            model=self.model_name,
            backend=backend or self.model_name,
            source_depths=np.atleast_1d(np.asarray(
                getattr(source, 'depths', []), dtype=float
            )),
            frequencies=(np.atleast_1d(np.asarray(frequencies, dtype=float))
                         if frequencies is not None else None),
            model_source=self.provenance,
            metadata=dict(extra),
        )
        if phase_reference is not None:
            kw['phase_reference'] = phase_reference
        return kw

    def _stamp_result(self, result, source: 'Source', *,
                      backend: Optional[str] = None,
                      frequencies: Optional[Union[float, np.ndarray]] = None,
                      phase_reference: Optional[str] = None):
        """Stamp the cross-model identification fields onto a ``Result`` that an
        io reader already constructed (so it couldn't take them as kwargs).
        Mirrors :meth:`_result_kwargs` for the reader-built path (Scooter,
        SPARC); leaves ``result.metadata`` untouched. Returns ``result``."""
        kw = self._result_kwargs(source, backend=backend,
                                 frequencies=frequencies,
                                 phase_reference=phase_reference)
        for attr in ('model', 'backend', 'source_depths', 'frequencies',
                     'model_source'):
            setattr(result, attr, kw[attr])
        if phase_reference is not None:
            result.phase_reference = phase_reference
        return result

    def _max_receiver_depth(self, env: 'Environment') -> float:
        """Deepest receiver depth this model can resolve the field at.

        Ray/mode models stop at the seafloor; full-waveguide spectral
        solvers override this to include the sediment layers they mesh
        through (see :meth:`_total_media_depth`).
        """
        return float(env.depth)

    def _total_media_depth(self, env: 'Environment') -> float:
        """
        Deepest modelled interface (m): the water column plus all sediment
        layer thicknesses. Below this lies the semi-infinite halfspace.

        Full-waveguide spectral solvers (Scooter, SPARC) resolve the field
        through the water and every fluid/elastic sediment layer, so the
        valid receiver range extends to this depth — not merely to
        ``env.depth`` (the seafloor).
        """
        depth = float(env.depth)
        if env.bottom.is_layered:
            depth += env.bottom.max_total_thickness()
        return depth

    def _clip_receiver_depths(
        self, receiver: 'Receiver', media_depth: float, margin: float = 3.0
    ) -> 'Receiver':
        """
        Clip receiver depths to the modelled media, with a safety margin.

        Parameters
        ----------
        receiver : Receiver
            Input receiver array
        media_depth : float
            Deepest modelled interface (m); see :meth:`_total_media_depth`.
            Only receivers *below* this boundary lie inside the semi-infinite
            halfspace, where the field is not resolved; those are pulled up to
            ``media_depth - margin``. Receivers anywhere in the water column or
            sediment layers (``depth <= media_depth``) are kept untouched.
        margin : float
            Landing margin above the halfspace boundary (m) for the receivers
            that must be pulled out of the halfspace. Default 3.0.

        Returns
        -------
        Receiver
            Receiver with clipped depths (unchanged if all depths are valid)
        """
        max_receiver_depth = receiver.depths.max()
        if max_receiver_depth > media_depth:
            # Only the sub-halfspace receivers are unresolvable; pull just
            # those up to a margin above the boundary and leave every
            # in-medium receiver alone. A full-column receiver grid
            # (depths up to env.depth) is therefore returned unchanged, so
            # its depth axis matches the ray/normal-mode models rather than
            # silently losing the deepest sample to a clip-and-dedup.
            ceiling = media_depth - margin
            clipped = np.where(
                receiver.depths > media_depth, ceiling, receiver.depths,
            )
            if receiver.receiver_type == 'grid':
                new_depths = np.unique(clipped)
            else:
                new_depths = clipped
            receiver = Receiver(
                depths=new_depths,
                ranges=receiver.ranges,
                receiver_type=receiver.receiver_type,
            )
            warnings.warn(
                f"{self.model_name}: receiver depths below the deepest "
                f"modelled interface ({media_depth:.1f} m) pulled up to "
                f"{ceiling:.1f} m (the field is not resolved inside the "
                f"semi-infinite halfspace). Add sediment layers (a layered "
                f"SeabedColumn) to place receivers below the seafloor.",
                UserWarning,
                stacklevel=2,
            )
        return receiver

    def __repr__(self) -> str:
        """``ClassName(arg=val, …)`` showing only constructor params whose
        current value differs from the constructor default.

        Walks ``__init__`` along the MRO (subclasses forward to
        ``super().__init__(**kwargs)``, so the union of named parameters
        across the chain is the full configuration surface). Reads each
        param off ``self.<name>`` — the same contract that powers
        ``model.copy``. Ndarrays and long sequences are summarised so
        the result stays one-line-readable even when a model has many
        knobs.
        """
        bits: List[str] = []
        for name, default in _collect_init_params(type(self)):
            if not hasattr(self, name):
                continue
            # Resolved binary paths are machine-specific and not copy-paste-
            # portable — hide them regardless of value.
            if name in ('executable', 'field_executable'):
                continue
            value = getattr(self, name)
            # ``cleanup`` resolves to ``work_dir is None`` when left at its None
            # default; hide it while it carries that auto value, matching
            # ``copy()``, which hands the ``None`` sentinel back so the clone
            # re-resolves against its own work_dir.
            if name == 'cleanup' and not self._cleanup_explicit:
                continue
            if default is not _NO_DEFAULT and _values_equal(value, default):
                continue
            bits.append(f"{name}={_short_repr(value)}")
        return f"{type(self).__name__}({', '.join(bits)})"


_NO_DEFAULT = object()


def _collect_init_params(cls) -> List[tuple]:
    """Walk ``cls.__mro__`` for every ``__init__`` and collect named
    parameters (excluding ``self`` and ``**kwargs``) in declaration order,
    deduplicated by name (subclass declaration wins).

    Returns a list of ``(name, default_or_NO_DEFAULT)`` pairs. Used by
    :meth:`PropagationModel.__repr__` and parallels what
    :meth:`PropagationModel.copy` introspects.
    """
    import inspect as _inspect

    seen: Dict[str, object] = {}
    order: List[str] = []
    for klass in cls.__mro__:
        if klass is object:
            continue
        init = klass.__dict__.get('__init__')
        if init is None:
            continue
        try:
            sig = _inspect.signature(init)
        except (TypeError, ValueError):
            continue
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            if param.kind == _inspect.Parameter.VAR_KEYWORD:
                continue
            if param.kind == _inspect.Parameter.VAR_POSITIONAL:
                continue
            if name in seen:
                continue
            seen[name] = (
                param.default if param.default is not _inspect.Parameter.empty
                else _NO_DEFAULT
            )
            order.append(name)
    return [(name, seen[name]) for name in order]


def _values_equal(a, b) -> bool:
    """Compare two configuration values, tolerating ndarray equality."""
    if a is b:
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(a), np.asarray(b)))
        except Exception:
            return False
    try:
        return bool(a == b)
    except Exception:
        return False


def _short_repr(value, *, list_limit: int = 6) -> str:
    """Compact ``repr`` for a constructor value, summarising big arrays.

    Used by :meth:`PropagationModel.__repr__` so ``print(model)`` stays
    short even when a knob holds a large ndarray or list.
    """
    if isinstance(value, np.ndarray):
        if value.size <= list_limit:
            return repr(value.tolist())
        return f"ndarray(shape={tuple(value.shape)}, dtype={value.dtype})"
    if isinstance(value, (list, tuple)):
        if len(value) <= list_limit:
            return repr(value)
        return f"{type(value).__name__}(len={len(value)})"
    if isinstance(value, Path):
        return repr(str(value))
    return repr(value)
