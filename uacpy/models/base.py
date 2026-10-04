"""Base class for acoustic propagation models.

:class:`PropagationModel` holds the contracts every wrapper in
:mod:`uacpy.models` inherits, grouped as: construction + spec application,
run-mode / run-kwarg resolution, broadband and time-series grid derivation,
work-dir setup and logging, input validation, the ``compute_*`` convenience
wrappers, executable lookup and subprocess launch, environment projection,
output-path and ``.prt`` bookkeeping, result stamping, depth policy,
``__repr__``.

The module-level code these read lives beside this module, one concern per
module: the collapse policy and the roughness helpers
(:mod:`~uacpy.models._projection`), :class:`~uacpy.models._spec.ModelSpec`
(:mod:`~uacpy.models._spec`), the TIME_SERIES grid a pulse implies
(:mod:`~uacpy.models._band`), the work directory and its
:class:`~uacpy.models._workspace.FileManager` (:mod:`~uacpy.models._workspace`),
the binary launch (:mod:`~uacpy.models._launch`), the source-depth stack
(:mod:`~uacpy.models._stacking`), the level conventions
(:mod:`~uacpy.models._conventions`), the notices a run states once
(:mod:`~uacpy.models._notices`), what a result records of how it was
computed (:mod:`~uacpy.models._extract`), and the introspection behind
``copy`` / ``__repr__`` (:mod:`~uacpy.models._introspect`).

``run()`` itself belongs to :class:`PropagationModel`: it is the template
method that validates the carrier triple, splits a multi-depth ``Source``
into one run per depth (or lets an engine that stacks them natively through),
applies a single source's ``weights`` to the field that comes back, and
stamps the weights on a stack.

Every engine subclasses :class:`PropagationModel` and supplies only the
stage hooks: ``run()`` projects the environment and checks the call (stages 1-2,
the engine's refusals in ``_validate_engine``), resolves every setting once
(stage 3, the engine's in ``_resolve_engine_settings``), then writes the
deck, launches the binary, reads its output and builds the result
(``_write_input``, ``_launch``, ``_read_output``, ``_to_result``) and stamps
it.
"""

import contextlib
import inspect
import warnings
from abc import ABC, ABCMeta, abstractmethod
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import (Any, Dict, List, Mapping, Optional, Self,
                    Tuple, Union)

import numpy as np

from uacpy._log import _log_enabled, _resolve_threshold, log_message
from uacpy.core.environment import Environment
from uacpy.core.exceptions import (
    ConfigurationError, ExecutableNotFoundError, FallbackWarning,
    InvalidDepthError, ProvenanceWarning,
    UnsupportedFeatureError, ValidityWarning,
)
from uacpy.core.receiver import Receiver
from uacpy.core.results import Result, ResultStack
from uacpy.core.source import Source
from uacpy.models.provenance import MODEL_PROVENANCE, model_provenance
from uacpy.core.run_settings import (
    OutputSpec, RunMode, RunSettings, TimeSettings, WaveguideSpeeds,
)
from uacpy.models._registry import engines_running
from uacpy.models._band import (
    BandResolution, _positive_finite, _real_pulse, frequencies_text,
    pad_waveform_to_duration, resolve_band,
)
from uacpy.models._checks import (
    check_carrier_types, check_per_range_receiver_depth,
    reject_malformed_irc_bottom, speed_bounds, total_media_depth,
    warn_on_range_coverage, warn_on_ssp_start,
    warn_receiver_below_resolvable,
)
from uacpy.models._extract import (
    _settings_as_run, attach_output_paths, check_output_contract,
    mask_source_axis, mask_unresolvable_depths, result_kwargs, stamp_result,
)
from uacpy.core._records import _plain
from uacpy.models._knobs import refuse_unknown_knob
from uacpy.models._replay import (check_recorded_per_launch,
                                  model_from_run_settings)
from uacpy.models._introspect import (
    _NO_DEFAULT, _collect_init_params, _short_repr, _values_equal,
)
from uacpy.models._launch import (
    Launch, _is_runnable, attach_prt_tail, find_executable_in_paths,
    raise_on_fortran_fatal, require_output, run_launch, run_subprocess,
    warn_on_prt_warnings,
)
from uacpy.models._notices import (
    _INSIDE_RUN, _WARNED_MODEL_PROVENANCE, _warn_if_volume_absorption_is_missing,
)
from uacpy.models._projection import (
    DEFAULT_COLLAPSE, VALID_COLLAPSE_METHODS, environment_projection,
)
from uacpy.models._spec import _CAPABILITY_FLAGS, EngineTraits, ModelSpec

#: s — how long one engine launch may run before it is stopped; every
#: engine's ``timeout`` default.
DEFAULT_RUN_TIMEOUT_S = 600.0
from uacpy.models._stacking import (
    _slabs_of, refuse_a_one_depth_weight_on_dB, run_depth_loop,
)
from uacpy.models._workspace import (
    FileManager, ScratchPolicy, _SCRATCH_REDIRECT, _effective_work_dir,
    setup_file_manager,
)

# ``warnings.warn(..., skip_file_prefixes=USER_FRAME_SKIP)`` reports the first
# frame outside the uacpy library — a warning raised from a nested helper in a
# model, or in an io / core layer a model delegates to, still points at the
# user's ``run()`` / constructor call. Hand-counted ``stacklevel`` cannot do
# that: it breaks the moment a check moves one frame deeper, and collapses
# distinct call sites onto one uacpy line in the warnings module's dedup key.
# The set is defined in ``core`` so io and core layers warn the same way
# without importing a model; :mod:`uacpy.core._warn_frames` carries the rest —
# why ``tests`` / ``examples`` are excluded, and why each directory entry ends
# at a separator.
from uacpy.core._warn_frames import USER_FRAME_SKIP

#: The traits of a class that declares no spec.
_DEFAULT_TRAITS = EngineTraits()

#: The optional keywords of ``run()`` / ``run_settings()`` /
#: ``validate_inputs()``, in signature order.
_RUN_KEYWORDS = ('frequencies', 'source_waveform', 'sample_rate',
                 'output_duration', 't_start')

#: The keywords only a TIME_SERIES run reads.
_TIME_SERIES_KEYWORDS = frozenset({'source_waveform', 'sample_rate',
                                   'output_duration', 't_start'})

#: The stage-4/5 hooks every engine implements, in the order
#: :meth:`PropagationModel._run_engine` calls them.
_STAGE_HOOKS = ('_write_input', '_launch', '_read_output', '_to_result')


@dataclass(frozen=True)
class _RunCall:
    """A run call after its own arguments are checked: the run mode,
    resolved once, the keywords the engine is handed (``t_start`` dropped
    where the mode cannot place a record), and the environment the engine
    runs on: the caller's, projected onto what the engine reads. ``env`` is
    ``None`` until stage 1 projects it
    (:meth:`PropagationModel._check_carriers_of_call`). ``engine_request``
    is what an engine's own entry point asked for beyond :meth:`run`'s
    keywords (the BOUNCE knobs of ``Bellhop.run_with_bounce``), ``None``
    from ``run``."""
    mode: 'RunMode'
    kwargs: Dict[str, object]
    env: Optional['Environment'] = None
    engine_request: object = None


@dataclass(frozen=True)
class ModesRequest:
    """The engine request of :meth:`PropagationModel.compute_modes`: the
    call's mode cap, ``None`` to keep the engine's own ``n_modes``."""
    n_modes: Optional[int] = None


@dataclass(frozen=True)
class StageInputs:
    """What the stage-4/5 hooks of a :class:`PropagationModel` read for one
    launch of its engine.

    Attributes
    ----------
    work_dir : Path
        The launch's work directory: a pinned ``work_dir`` (or its
        ``source_depth_*`` subdirectory), or a fresh temporary one.
    env : Environment
        The environment the engine runs on: the caller's, projected onto
        what the engine reads.
    source : Source
        The source of this launch: one depth of a per-depth loop, the unit
        copy of a weighted one-depth source, or the call's own.
    receiver : Receiver or None
        The call's receiver.
    settings : RunSettings
        The call's resolved settings; ``settings.engine`` holds the
        engine's own.
    launch : int
        Which of the call's :meth:`PropagationModel._n_launches` launches these
        inputs are for, from 0.
    earlier : tuple
        What :meth:`PropagationModel._read_output` returned for the launches
        before this one, in order — a deck that depends on an earlier
        launch's output reads it here. Every launch runs in the same
        ``work_dir``.
    prepared : object
        What :meth:`PropagationModel._prepare_launches` built once for every
        launch of the call (``None`` by default).
    """
    work_dir: Path
    env: 'Environment'
    source: 'Source'
    receiver: Optional['Receiver']
    settings: RunSettings
    launch: int = 0
    earlier: Tuple = ()
    prepared: Any = None


class _ConstructedModel(ABCMeta):
    """Marks an engine constructed once its whole constructor chain has
    returned, so :meth:`PropagationModel.__setattr__` lets the constructors
    create attributes and refuses an unknown public one afterwards."""

    def __call__(cls, *args, **kwargs):
        model = super().__call__(*args, **kwargs)
        vars(model)['_constructed'] = True
        return model

    @property
    def __signature__(cls):
        """The constructor's own parameters, so ``inspect.signature`` and
        ``help`` show the engine's knobs instead of this ``__call__``'s
        ``(*args, **kwargs)``."""
        params = list(inspect.signature(cls.__init__).parameters.values())
        return inspect.Signature(params[1:])


class PropagationModel(ABC, metaclass=_ConstructedModel):
    """
    Abstract base class for acoustic propagation models.

    Provides the common interface and shared utilities (subprocess runner,
    executable lookup, input validation, range-dependent handling) for all
    propagation models.

    :meth:`run` owns the order of a call, and the engine supplies only
    what is its own:

    1. normalise — :meth:`run` checks the call's own arguments and projects
       the environment onto what the engine reads
       (:meth:`_project_environment`, driven by ``spec``), once per call;
    2. validate — the carrier refusals of :meth:`validate_inputs` on the
       projected environment, the engine's own in :meth:`_validate_engine`;
    3. resolve settings — :meth:`_resolve_engine_settings` returns the
       engine's frozen :class:`~uacpy.core.run_settings.EngineSettings` and
       raises the refusals of settings the engine cannot run;
    4. execute — in a fresh work directory, :meth:`_write_input` writes the
       deck with the io writers, :meth:`_launch` runs the binary and
       :meth:`_read_output` reads what it wrote with the io readers;
    5. extract — :meth:`_to_result` builds the result, whose class and
       conventions :meth:`_check_output_contract` holds to the engine's
       :attr:`outputs` entry for the mode;
    6. stamp — :meth:`run` records the settings on the result.

    :meth:`validate_inputs` runs stages 1-3 and :meth:`run_settings`
    returns their record, so both refuse exactly what :meth:`run` refuses
    before it launches. The source-depth loop and the weights are
    :func:`~uacpy.models._stacking.run_depth_loop`'s; each source it hands
    on is one :meth:`_run_engine` call, whose launches share one work
    directory. The four stage hooks are abstract, so a class that does not
    implement them (``PropagationModel``, ``OASES``) is a base, not a model.

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
    cleanup : bool, optional
        Delete the run's scratch files when ``run()`` returns. ``None``
        (default) resolves to ``work_dir is None``, i.e. uacpy removes only
        directories it created. ``cleanup=False`` with an unpinned
        ``work_dir`` keeps the temp directory (and the ``*_file`` metadata
        paths that point into it); the caller then owns its removal.
    timeout : float, optional
        Wall-clock limit (s) on each binary launch; a run that exceeds it is
        killed and raises :class:`ModelExecutionError` with ``timed_out``
        set. Default 600.0.
    collapse : dict, optional
        Per-feature policies applied when an environment carries a feature
        this model does not read (``'bathymetry'``, ``'ssp'``,
        ``'bottom_range'``, ``'bottom_layers'``, ``'altimetry'``,
        ``'surface'``, ``'elastic'``); any subset overrides
        ``DEFAULT_COLLAPSE``, and the resolved policy is what
        ``_project_environment`` applies. See :meth:`_project_environment`.

    Attributes
    ----------
    model_name : str
        Name of the model (class name).
    use_tmpfs : bool
        Whether tmpfs is used.
    verbose : bool or str
        Verbose-output gate (see constructor).
    """

    # The leading positional run() parameters every wrapper must carry, in
    # order. Anything a wrapper adds beyond these must be keyword-only (after
    # a bare ``*``) and no wrapper may use ``**kwargs`` — an unknown keyword
    # has to fail with TypeError at the call site, not be silently swallowed.
    _RUN_POSITIONAL = ('self', 'env', 'source', 'receiver', 'run_mode')

    # Declarative metadata. When a subclass declares a :class:`ModelSpec`, the
    # base validates it at class-definition time and applies it in
    # ``__init__``. A subclass without one keeps the base defaults and sets
    # ``_supported_modes`` / ``_supports_*`` / collapse by hand in ``__init__``.
    # What the engine does with the parts of a call the base decides is the
    # spec's ``traits`` (:class:`EngineTraits`), read through :attr:`_traits`.
    spec: Optional['ModelSpec'] = None

    # Provenance/licence: ``id`` of this engine's entry in
    # :data:`uacpy.models.provenance.MODEL_PROVENANCE`. Kept off :class:`ModelSpec`
    # (which is about run behaviour) because provenance is an orthogonal axis —
    # the same separation the data layer keeps between its carriers and
    # :mod:`uacpy.data.sources`. Surfaced via :attr:`provenance` / :attr:`citation`;
    # a ``commercial_use=False`` engine (OASES) warns once at construction.
    provenance_id: Optional[str] = None

    #: What each run mode returns — result class, quantity, unit, phase
    #: convention, coherence — declared once per engine as
    #: ``{RunMode: OutputSpec}``. ``run_settings(...).output`` reads it, and a
    #: run checks every result it builds against it. Empty
    #: on the base.
    outputs: 'Mapping[RunMode, OutputSpec]' = MappingProxyType({})

    @property
    def _traits(self) -> EngineTraits:
        """This class's :class:`EngineTraits`: ``spec.traits``, or the
        defaults on a class that declares no spec."""
        return self.spec.traits if self.spec is not None else _DEFAULT_TRAITS

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
        if cls.provenance_id is not None:
            if cls.provenance_id not in MODEL_PROVENANCE:
                raise ConfigurationError(
                    f"{cls.__name__}.provenance_id = {cls.provenance_id!r} is not a known "
                    f"model source. Valid: {sorted(MODEL_PROVENANCE)}."
                )
        # (attribute name, function) — the attribute name is what the
        # messages below quote, because a ``run`` declared as a lambda has
        # ``__name__ == '<lambda>'``.
        bodies = ([('run', cls.__dict__['run'])] if 'run' in cls.__dict__
                  else [])
        hooks = [name for name in _STAGE_HOOKS if name in cls.__dict__]
        declares_abstract = any(getattr(value, '__isabstractmethod__', False)
                                for value in cls.__dict__.values())
        if declares_abstract or not (bodies or hooks):
            # Abstract: an intermediate base that leaves the stage hooks to
            # its own subclasses (OASES). Nothing below
            # applies, and the two declarations required of a concrete
            # wrapper are its subclasses' to make.
            return

        # A subclass that defines the stage hooks (or its own ``run``) is
        # a model a user can hold — and both declarations below are
        # load-bearing at that point. Without
        # ``spec`` the class silently takes the base defaults
        # (COHERENT_TL only, no env-shape support, point sources), which are
        # nobody's real answer; without ``provenance_id`` the licence and citation
        # machinery is skipped entirely, so an engine that must warn on use
        # would not (``_warn_restricted_provenance`` returns on ``source is
        # None``). Both are checked here rather than at ``__init__`` so a
        # missing declaration fails on import.
        missing = [name for name in ('spec', 'provenance_id')
                   if getattr(cls, name, None) is None]
        if missing:
            seen = ' or '.join([f"{name}()" for name, _ in bodies]
                               + [f"{name}()" for name in hooks])
            raise TypeError(
                f"{cls.__name__} defines {seen} but declares no "
                f"{' or '.join(missing)}. A concrete model must set both: "
                f"``spec = ModelSpec(...)`` for its run modes, env-shape "
                f"support and source geometries, and ``provenance_id = '<id>'`` "
                f"naming its engine in MODEL_PROVENANCE so licence and citation "
                f"metadata reach the user."
            )

        for name, body in bodies:
            cls._check_run_signature(name, body)

    @classmethod
    def _check_run_signature(cls, name: str, body) -> None:
        """Refuse a subclass's own ``run`` whose signature breaks the one
        every model shares: the four carriers by position, every extra as
        a keyword, no ``**kwargs`` sink, and no default of its own for
        ``run_mode``."""
        import inspect

        label = f"{cls.__name__}.{name}()"
        params = list(inspect.signature(body).parameters.values())
        for p in params:
            if p.kind is inspect.Parameter.VAR_KEYWORD:
                raise TypeError(
                    f"{label} must not use **kwargs: an unknown "
                    "keyword has to raise TypeError, not be swallowed. Declare "
                    "the accepted extras as keyword-only after a bare '*'."
                )
            # The mirror of the rule below: a *args sink accepts exactly the
            # unknown positional arguments that rule exists to refuse.
            if p.kind is inspect.Parameter.VAR_POSITIONAL:
                raise TypeError(
                    f"{label} must not use *args: an unknown positional "
                    "argument has to raise TypeError, not be swallowed. The "
                    "four carriers are the whole positional contract."
                )

        leading = params[:len(cls._RUN_POSITIONAL)]
        names = tuple(p.name for p in leading)
        if names != cls._RUN_POSITIONAL:
            raise TypeError(
                f"{label} must begin with "
                f"{cls._RUN_POSITIONAL!r}; got {names!r}."
            )
        for p in leading:
            if p.kind is inspect.Parameter.KEYWORD_ONLY:
                raise TypeError(
                    f"{label} parameter {p.name!r} must be "
                    "positional-or-keyword, not keyword-only."
                )
        # The override hands ``run_mode`` to ``super().run()``, which
        # resolves ``None`` through ``_default_run_mode()``: a default of the
        # override's own would be a second decider of the default mode.
        # ``inspect.Parameter.empty`` (no default at all) decides nothing, so
        # only a non-None default is refused.
        run_mode = leading[-1]
        if run_mode.default not in (None, inspect.Parameter.empty):
            raise TypeError(
                f"{label} declares run_mode={run_mode.default!r}; the "
                f"default must be None — run() resolves it through "
                f"_default_run_mode(), which is the place to change it."
            )

        for p in params[len(cls._RUN_POSITIONAL):]:
            if p.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                raise TypeError(
                    f"{label} parameter {p.name!r} must be "
                    "keyword-only (place it after a bare '*') so unknown "
                    "positional args cannot reach it."
                )

    def __setattr__(self, name: str, value) -> None:
        """Assign ``name``; on a constructed engine, a public name it does
        not already carry is refused
        (:func:`~uacpy.models._knobs.refuse_unknown_knob`), since no run
        would read it."""
        if vars(self).get('_constructed'):
            refuse_unknown_knob(self, name)
        super().__setattr__(name, value)

    def __init__(
        self,
        *,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        cleanup: Optional[bool] = None,
        timeout: float = DEFAULT_RUN_TIMEOUT_S,
        collapse: Optional[Dict[str, str]] = None,
    ):
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
        # ``VALID_COLLAPSE_METHODS`` above lists the methods each key
        # takes, and is itself read from the carriers that implement them, so
        # the values are not restated here. What each key reduces:
        #
        # 'bathymetry'    : a range-dependent depth profile to one depth
        # 'ssp'           : a range-dependent sound-speed field to one column
        # 'bottom_range'  : a range-dependent bottom to one column ('mean' and
        #                   'median' are numeric, so an all-half-space bottom
        #                   only — a layer stack cannot be averaged)
        # 'bottom_layers' : each column's layer stack to a half-space; see
        #                   SeabedColumn.collapse_layers for what each method keeps
        # 'altimetry'     : a rough sea surface, dropped entirely
        # 'surface'       : a range-dependent Surface to a single boundary
        # 'elastic'       : a shear-supporting boundary to one that does not
        #                   ('fluid' zeroes the shear, 'vacuum' replaces it)
        #
        # ``self._collapse`` is the resolved policy (defaults ← spec ← user);
        # ``self.collapse`` keeps the constructor argument verbatim, because
        # ``copy()`` and ``__repr__`` read every knob off ``self.<param>``.
        self._collapse: Dict[str, str] = dict(DEFAULT_COLLAPSE)
        self._user_collapse: Dict[str, str] = {}
        self.collapse = dict(collapse) if collapse else None
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
            self._collapse.update(collapse)
            self._user_collapse = dict(collapse)

        # Subclasses override to declare the run modes they support.
        self._supported_modes: List[RunMode] = [RunMode.COHERENT_TL]

        # Capability flags — one per axis of ``Environment`` shape. Subclasses
        # flip True for each feature they honour natively; anything left False
        # that's present in env on ``run()`` is collapsed by
        # ``_project_environment`` and triggers one ``FallbackWarning`` per dropped
        # feature.
        #
        # The flag list is intentionally bounded. Add a flag ONLY for a
        # question of the form "does this env shape work with this model?".
        # Niche numerical-method requirements (3-D, broadband, specific SSP
        # interp scheme, volume-attenuation formula) belong in run()-time
        # asserts, not here.
        self._supports_altimetry: bool = False
        self._supports_range_dependent_bathymetry: bool = False
        self._supports_range_dependent_ssp: bool = False
        self._supports_range_dependent_bottom: bool = False
        self._supports_layered_bottom: bool = False
        self._supports_elastic_media: bool = False
        # An engine writes one source-depth grid into a single deck in the
        # modes its ``spec.traits.native_multi_depth_modes`` lists, and this flag is
        # True exactly when that table is non-empty (``_apply_spec``). Every other engine
        # and mode reads one source depth per deck: in a field mode ``run()`` loops over the
        # depths in Python and stacks the slabs
        # (:func:`~uacpy.models._stacking.run_per_source_depth`); in any other mode the ten models that
        # read source geometry raise from ``_validate_geometry``. Bounce
        # accepts a multi-depth ``Source`` without raising because it reads
        # no source geometry at all and overrides ``_validate_geometry`` to
        # a no-op (``bounce/_model.py``); the extra depths reach no deck.
        self._supports_multi_source_depth: bool = False
        self._supports_source_beam_pattern: bool = False
        # Surface sigma(1). SPARC's GetPar (Scooter/sparc.f90:177) and
        # Bounce's elastic branch (Kraken/bounce.f90:104) ERROUT on a
        # non-zero SSP%sigma; Kraken consumes it (Kraken/kraken.f90:902) and
        # Scooter in its vacuum-boundary impedance (Scooter/scooter.f90:309).
        self._supports_rough_surface: bool = False
        # Seabed sigma(NMedia+1). Kraken/KrakenC feed it to the
        # Kuperman-Ingenito interfacial-roughness perturbation
        # (Kraken/Scattering.f90:8, Kraken/kraken.f90:902); Bellhop's solver
        # ignores the value and RAM's PE format has nowhere to put it.
        self._supports_rough_bottom: bool = False
        # Mirrors the class attribute; see _CAPABILITY_FLAGS.
        self._supports_volume_attenuation: bool = \
            self._traits.consumes_volume_absorption
        self._supported_source_types: frozenset = frozenset({'point'})

        # When the subclass declares a ModelSpec, apply it now (after the
        # defaults above and after ``_user_collapse`` is populated, so the
        # collapse precedence DEFAULT ← spec ← user override holds). A
        # subclass may still override an individual flag afterward for the
        # rare instance-dependent capability.
        if self.spec is not None:
            self._apply_spec()
        # Independent of spec: a model may declare ``provenance_id`` without one.
        self._warn_restricted_provenance()

    @property
    def provenance(self):
        """The engine's :class:`~uacpy.models.provenance.ModelProvenance` (authorship
        + licence + citation), or ``None`` if the class declares no
        :attr:`provenance_id`."""
        return model_provenance(self.provenance_id)

    @property
    def citation(self) -> str:
        """The engine's bibliographic citation string (``''`` if unknown):
        this engine's own reference where its source entry is shared by
        several engines (:meth:`ModelProvenance.citation_for`)."""
        src = self.provenance
        return src.citation_for(self.model_name) if src is not None else ''

    def _warn_restricted_provenance(self) -> None:
        """Emit a one-time ``ProvenanceWarning`` for a licence-restricted engine.

        Mirrors the non-commercial fetch warning in ``data/`` (the audit's
        CRUST1.0 fix): a ``commercial_use=False`` source — OASES — must never
        be used silently. Deduplicated per source ``id`` per process so a
        parameter sweep warns once, not on every instance.
        """
        src = self.provenance
        if src is None or src.commercial_use or src.id in _WARNED_MODEL_PROVENANCE:
            return
        _WARNED_MODEL_PROVENANCE.add(src.id)
        warnings.warn(
            f"{self.model_name} uses {src.name} ({src.license}). {src.note} "
            f"Cite: {src.citation}",
            ProvenanceWarning, skip_file_prefixes=USER_FRAME_SKIP,
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
        if 'volume_attenuation' in spec.supports:
            raise ConfigurationError(
                f"{self.model_name}.spec.supports must not list "
                f"'volume_attenuation': it mirrors the class attribute "
                f"spec.traits.consumes_volume_absorption, so declaring it here would "
                f"give the same question two answers."
            )
        if 'multi_source_depth' in spec.supports:
            raise ConfigurationError(
                f"{self.model_name}.spec.supports must not list "
                f"'multi_source_depth': it is read off "
                f"spec.traits.native_multi_depth_modes (the modes whose deck carries "
                f"every source depth), so declaring it here would give the "
                f"same question two answers."
            )
        derived = {'volume_attenuation', 'multi_source_depth'}
        for flag in _CAPABILITY_FLAGS - derived:
            setattr(self, f'_supports_{flag}', flag in spec.supports)
        self._supports_multi_source_depth = bool(
            self._traits.native_multi_depth_modes)
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
        are coerced to the corresponding enum member; any other string is
        refused with :class:`ConfigurationError` (naming the member when it
        is a member's name, ``'COHERENT_TL'``).

        Pass ``default=`` to override the auto-pick when the model has a
        smarter rule (e.g. Kraken picks BROADBAND when a frequency
        vector is supplied).

        A mode the model does not run goes to :meth:`_refuse_run_mode`
        first, so an engine can refuse it with a message of its own.
        """
        if run_mode is None:
            run_mode = default if default is not None else self._supported_modes[0]
        if isinstance(run_mode, str):
            try:
                run_mode = RunMode(run_mode)
            except ValueError:
                message = (
                    f"{self.model_name}: run_mode={run_mode!r} must be a "
                    f"RunMode member or its string value, one of "
                    f"{[m.value for m in RunMode]}.")
                named = RunMode.__members__.get(run_mode.upper())
                if named is not None:
                    message += (
                        f" {run_mode!r} is the name of RunMode.{named.name}; "
                        f"pass RunMode.{named.name} or {named.value!r}.")
                raise ConfigurationError(message) from None
        if not self.supports_mode(run_mode):
            self._refuse_run_mode(run_mode)
            raise UnsupportedFeatureError(
                self.model_name, str(run_mode),
                alternatives=[str(m) for m in self._supported_modes],
                alternatives_label='run modes',
            )
        return run_mode

    def _refuse_run_mode(self, run_mode: 'RunMode') -> None:
        """Engine hook: raise a refusal of ``run_mode`` more specific than
        the generic :class:`UnsupportedFeatureError`, which follows when this
        returns. Called only for a mode this model does not run. The base
        adds nothing."""

    def _reject_unsupported_run_kwargs(self, **kwargs):
        """Refuse the optional ``run()`` keywords a model never consumes.

        Every model takes the full keyword set (``frequencies``,
        ``source_waveform``, ``sample_rate``, ``output_duration``,
        ``t_start``) so a polymorphic ``model.run(...)`` never raises
        ``TypeError``. A keyword that no mode of this model reads — the
        waveform keywords on a model without ``TIME_SERIES``, ``frequencies=``
        on one without a broadband path — would be ignored, and ignoring it
        *silently* would hide a caller mistake, so any of these passed a
        non-``None`` value raises here instead.
        :meth:`_run_keywords_never_consumed` names them; a keyword the model
        reads on another mode is warned about and dropped instead."""
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
        ``FallbackWarning``. Consuming paths (BROADBAND/TIME_SERIES, …) must not
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
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )

    @property
    def supported_modes(self) -> List[RunMode]:
        """List of run modes supported by this model."""
        return self._supported_modes

    def supports_mode(self, mode: RunMode) -> bool:
        """Return True if the model supports ``mode``.

        Parameters
        ----------
        mode : RunMode
            The run mode asked about.
        """
        return mode in self._supported_modes

    @property
    def supported_features(self) -> List[str]:
        """Environment-shape features this *instance* carries into its deck.

        The env-shape twin of :attr:`supported_modes`. Names are the
        ``_CAPABILITY_FLAGS`` vocabulary (``'range_dependent_ssp'``,
        ``'layered_bottom'``, …); a feature listed here survives
        ``_project_environment`` untouched, one that is not is collapsed or
        dropped with a warning.

        Read from the instance, not from ``spec.supports``: a model may
        resolve a flag from its own constructor arguments (``Bellhop`` turns
        ``range_dependent_ssp`` off for ``interp_ssp='linear'``), and the
        class-level declaration cannot know that.
        """
        return sorted(name for name in _CAPABILITY_FLAGS
                      if getattr(self, f'_supports_{name}'))

    def supports_feature(self, name: str) -> bool:
        """Return True if this instance carries env-shape feature ``name``.

        Parameters
        ----------
        name : str
            An env-shape capability of ``_CAPABILITY_FLAGS``.

        Raises
        ------
        ConfigurationError
            For a name outside ``_CAPABILITY_FLAGS`` — a typo would
            otherwise answer ``False``, which reads as a real "no".
        """
        if name not in _CAPABILITY_FLAGS:
            raise ConfigurationError(
                f"{type(self).__name__}.supports_feature: unknown capability "
                f"{name!r}; expected one of {sorted(_CAPABILITY_FLAGS)}."
            )
        return bool(getattr(self, f'_supports_{name}'))

    def copy(self, **overrides) -> Self:
        """Return a new instance with the same configuration plus ``overrides``.

        Model configuration is constructor-only by design, which means
        every parameter sweep boils down to "instantiate the model again
        with one knob changed." This helper does that without forcing
        the caller to re-type every other argument::

            base = RAM(dr=2.0, dz=0.5, n_pade=8)
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
        Self
            A fresh instance of the same concrete class. ``Self`` rather than
            ``PropagationModel``: ``OASP(...).copy()`` is an ``OASP``, and
            declaring the base class discards that at every call site.

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

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[RunMode] = None,
        *,
        frequencies: Optional[np.ndarray] = None,
        source_waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        output_duration: Optional[float] = None,
        t_start: Optional[float] = None,
    ) -> Union[Result, ResultStack]:
        """Run the propagation model.

        Every wrapper takes the same first four parameters in the same
        order: the common ``Environment`` / ``Source`` / ``Receiver``
        triple, followed by the optional ``run_mode``.

        Model configuration is **constructor-only** — every model knob
        (e.g. ``RAM(dr=2.0, dz=0.5, n_pade=8)``,
        ``Bellhop(beam_type='B', n_beams=500)``) is set when the model
        instance is created. To sweep parameters, instantiate one model
        per parameter set.

        ``run()`` accepts a fixed keyword-only set: ``frequencies``,
        ``source_waveform``, ``sample_rate``, ``output_duration``,
        ``t_start``. ``output_duration`` and ``t_start`` describe the
        TIME_SERIES record the caller wants — its length and the time of its
        first sample, in seconds after emission — and mean that on every
        engine: ``t_start=None`` opens the record just before the earliest
        arrival (Bellhop from the arrivals, the IFFT engines from the range
        and the fastest sound speed). An engine that cannot place its record
        (SPARC, whose output always starts at emission) warns that
        ``t_start`` is ignored. Every
        TIME_SERIES-capable wrapper (Bellhop, Scooter, Kraken,
        OASP, RAM) consumes ``source_waveform`` and ``sample_rate``;
        SPARC marches ``source_waveform`` (as its STSFIL series) when its
        ``pulse_type`` is unpinned, and otherwise marches its own pulse.
        ``output_duration`` is the desired output time
        window (seconds); when given, the IFFT-based wrappers zero-pad
        the source waveform internally so the auto-derived broadband
        grid is tight enough (``Δf = 1/output_duration``), Bellhop
        maps it to ``time_window`` for delay-and-sum synthesis, and SPARC
        takes it as ``time_max`` when ``time_max`` is unpinned. Models
        with a broadband path consume ``frequencies`` as an explicit
        override for ``source.frequencies``. No other kwargs are accepted —
        passing one raises :class:`TypeError`.

        One rule decides what a keyword the resolved mode does not read
        costs. A keyword that no mode of this model reads (the waveform
        keywords and ``t_start`` on a model without ``TIME_SERIES``,
        ``frequencies`` on one without a broadband path) raises
        :class:`UnsupportedFeatureError`. A keyword the model reads on
        another mode is dropped with a ``FallbackWarning`` — except
        ``frequencies`` passed with a single-frequency ``run_mode`` (the
        TL modes, ``RAYS`` / ``EIGENRAYS`` / ``ARRIVALS``, ``MODES``), which
        raises :class:`ConfigurationError` on every engine except OASP, which
        consumes it — the value pins the frequency sweep its solver always
        runs. A single-frequency run takes its frequency from
        ``source.frequencies``.

        Every setting the call resolves — the run mode, the frequency grid,
        the source-depth loop, the weights, the time-series request — is
        decided once, before anything is launched, and recorded on the
        returned result as ``result.run_settings``.
        :meth:`run_settings` returns the same record without running, and
        :meth:`validate_inputs` raises what this call would raise before
        launching.

        A subclass may override ``run`` to extend it, but the override must
        call ``super().run(...)`` and let it do the running: the argument
        checks, the resolution of the settings, the source-depth loop, the
        weights and the stamp all happen here, and an override that goes
        around them returns results those guarantees do not cover.

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
            wrappers zero-pad so ``Δf = 1/output_duration``. A TIME_SERIES
            result's ``frequencies`` stamp is that same grid, so it changes
            when ``output_duration`` does — it records the frequencies the
            engine actually propagated for this call, not a property of the
            environment.
        t_start : float, optional
            Keyword-only. Time (s after emission) of the first sample of the
            TIME_SERIES record; ``None`` opens it just before the earliest
            arrival.

        Returns
        -------
        result : Result or ResultStack
            One of the typed :mod:`uacpy.core.results` subclasses
            (``Field``, ``Arrivals``, ``Modes``, …) determined
            by ``run_mode`` and the model — or a ``ResultStack`` of them
            over ``source_depth`` when ``source`` carries several depths.
            In a field mode (the TL modes, ``BROADBAND``, ``TIME_SERIES``)
            every model returns that stack: the engine runs once per depth
            with the same environment, receiver and keywords, and each
            slab is the unit-amplitude field of one source, so
            ``stack.superpose()`` adds them with ``source.weights``
            (see DOCUMENTATION.md §ResultStack). Outside the field modes
            only Bellhop stacks (``RAYS`` / ``ARRIVALS`` / ``EIGENRAYS``);
            the other engines raise ``ConfigurationError``.
            ``ResultStack`` is not a ``Result`` subclass, so a caller that
            annotates the result has to name both.
        """
        call = self._check_call(
            env, source, receiver, run_mode, frequencies=frequencies,
            source_waveform=source_waveform, sample_rate=sample_rate,
            output_duration=output_duration, t_start=t_start)
        return self._run_call(env, source, receiver, call)

    def run_settings(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[RunMode] = None,
        *,
        frequencies: Optional[np.ndarray] = None,
        source_waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        output_duration: Optional[float] = None,
        t_start: Optional[float] = None,
    ) -> RunSettings:
        """The settings :meth:`run` would use for this call, without running.

        Takes exactly the arguments of :meth:`run` and runs the same code up
        to the point where ``run`` would launch the engine: the argument
        checks, :meth:`validate_inputs`, and the resolution of the run mode,
        the frequency grid, the source-depth loop, the source weights and the
        time-series request. It raises what ``run`` raises before launching,
        and emits the same warnings.

        Parameters
        ----------
        env : Environment
            The environment.
        source : Source
            The source.
        receiver : Receiver
            The receiver grid.
        run_mode : RunMode, optional
            Output type; ``None`` is the model's natural default.
        source_waveform : ndarray, optional
            Source time series, sampled at ``sample_rate``.
        sample_rate : float, optional
            Sample rate (Hz) of ``source_waveform``.
        output_duration : float, optional
            Record length (s).
        frequencies : array_like, optional
            The synthesis frequency grid (Hz), in place of the derived one.
        t_start : float, optional
            Time (s after emission) of the first sample; ``None`` opens the
            record just before the earliest arrival.

        Returns
        -------
        RunSettings
            Immutable; ``print()`` shows one line per setting. The result of
            the matching ``run`` carries the settings it was produced with as
            ``result.run_settings``. Where the engine's result is computed on
            a frequency grid of its own (a lattice of its own, say),
            ``result.run_settings`` records what the engine used and says so
            in its ``notes``.

        Examples
        --------
        >>> print(Kraken().run_settings(env, source, receiver,
        ...                             frequencies=[90.0, 100.0, 110.0]))
        """
        call = self._check_call(
            env, source, receiver, run_mode, frequencies=frequencies,
            source_waveform=source_waveform, sample_rate=sample_rate,
            output_duration=output_duration, t_start=t_start)
        call = self._check_carriers_of_call(env, source, receiver, call)
        return self._resolve_settings(env, source, receiver, call)

    # ── a producer another engine runs ──────────────────────────────────

    def _producer_settings(self, env, source, receiver, mode, *,
                           frequencies=None):
        """``(settings, notices)`` of this model run by another engine as a
        producer (Bellhop's BOUNCE table, the OASES mean field): stages 1-3
        of a call with no TIME_SERIES keywords, saying nothing. The
        lossless-water notice is skipped (``announce=False``), the notices
        of the environment projection are returned for the caller to
        record, and the stage-3 notices stay on ``settings.engine`` for the
        caller's :meth:`_announce_engine_settings`. The projection is the
        pure :meth:`_environment_projection`; no producer engine overrides
        :meth:`_project_environment`."""
        call = self._check_call(
            env, source, receiver, mode, frequencies=frequencies,
            source_waveform=None, sample_rate=None, output_duration=None,
            t_start=None, announce=False)
        normalised = self._normalise_env(env, call.mode)
        self._check_time_series_keywords(call)
        projected, notices = self._environment_projection(normalised)
        self._check_carriers(projected, source, receiver, call.mode,
                             request=call)
        settings = self._resolve_settings(
            env, source, receiver, dataclasses.replace(call, env=projected),
            announce=False)
        return settings, notices

    def _write_producer_deck(self, inputs: StageInputs) -> Path:
        """Stage 4 of a producer's launch: its deck (:meth:`_write_input`)."""
        return self._write_input(inputs)

    def _launch_producer(self, inputs: StageInputs, deck) -> None:
        """Stage 4 of a producer's launch: its binary on ``deck``
        (:meth:`_launch`)."""
        self._launch(inputs, deck)

    def _run_producer_launch(self, inputs: StageInputs, deck):
        """Stages 4-6 of a producer's launch after its binary: the output
        read, the result assembled, checked against the output contract and
        stamped with ``inputs.settings`` — what the producer's own
        :meth:`run` returns for that launch."""
        raw = self._read_output(inputs, deck)
        result = self._to_result(inputs, deck, raw)
        self._check_output_contract(result, inputs.settings)
        return self._stamp_run_settings(result, inputs.settings)

    # ── stages of a run ─────────────────────────────────────────────────

    def _run_call(self, env, source, receiver, call: _RunCall):
        """:meth:`run` after :meth:`_check_call`: stages 1-6 of ``call``.
        An engine's own entry point (``Bellhop.run_with_bounce``) checks its
        call with :meth:`_check_call`, sets its ``engine_request`` and runs
        it here."""
        call = self._check_carriers_of_call(env, source, receiver, call)
        token = _INSIDE_RUN.set(True)
        try:
            settings = self._resolve_settings(env, source, receiver, call)
            result = self._execute_call(env, source, receiver, call, settings)
            return self._stamp_run_settings(result, settings)
        finally:
            _INSIDE_RUN.reset(token)

    def _check_carriers_of_call(self, env, source, receiver,
                                call: _RunCall) -> _RunCall:
        """Stages 1-2 after :meth:`_check_call`, for :meth:`run`,
        :meth:`run_settings` and :meth:`validate_inputs`: the engine's own
        reduction of the environment for the mode (:meth:`_normalise_env`),
        the TIME_SERIES keywords (:meth:`_check_time_series_keywords`), then
        the projection of the environment and the carriers checked against
        it, both handed ``request=call``. Returns ``call`` carrying the
        projected environment."""
        env = self._normalise_env(env, call.mode)
        self._check_time_series_keywords(call)
        projected = self._project_environment(env, request=call)
        self._check_carriers(projected, source, receiver, call.mode,
                             request=call)
        return dataclasses.replace(call, env=projected)

    def _normalise_env(self, env, mode):
        """Engine hook, stage 1: the environment ``mode`` reads, before the
        shared projection. The base returns ``env``; Kraken's MODES reduces a
        range-dependent environment to its r = 0 profile."""
        return env

    def _check_time_series_keywords(self, call: _RunCall) -> None:
        """Stage 2 on a model that declares ``TIME_SERIES``: in
        ``TIME_SERIES`` the pulse pair is refused when missing or unusable
        (:meth:`_require_timeseries_signal`); in any other mode the
        TIME_SERIES keywords are dropped with one warning, ``BROADBAND``
        naming its own reason. Once per call, whatever the source-depth
        loop."""
        if RunMode.TIME_SERIES not in self._supported_modes:
            return
        kw = call.kwargs
        if call.mode == RunMode.TIME_SERIES:
            self._check_time_series_request(call)
            return
        reason = None
        if call.mode == RunMode.BROADBAND:
            reason = ('BROADBAND returns the transfer function H(f); the '
                      'source pulse and time-axis keywords apply to '
                      'TIME_SERIES only')
        self._warn_ignored_run_kwargs(
            call.mode, reason=reason,
            source_waveform=kw['source_waveform'],
            sample_rate=kw['sample_rate'],
            output_duration=kw['output_duration'])

    def _check_time_series_request(self, call: _RunCall) -> None:
        """Engine hook, stage 2 of a ``TIME_SERIES`` call: refuse a missing
        or unusable pulse pair (:meth:`_require_timeseries_signal`). SPARC,
        which can march a pulse of its own, decides otherwise."""
        kw = call.kwargs
        self._require_timeseries_signal(
            call.mode, kw['source_waveform'], kw['sample_rate'])

    def _check_call(self, env, source, receiver, run_mode, *, frequencies,
                    source_waveform, sample_rate, output_duration,
                    t_start, announce: bool = True) -> _RunCall:
        """Check a call's own arguments and resolve its run mode, once.

        The part of a run that needs no environment projection: the carrier
        types, the lossless-water notice, ``t_start`` / ``output_duration``,
        the run mode (resolved here and nowhere else in the call), and the
        one keyword rule (see :meth:`run`). Shared by :meth:`run`,
        :meth:`run_settings` and :meth:`validate_inputs`, so all three refuse
        the same calls with the same exceptions. ``announce=False`` (the
        :meth:`validate_inputs` path, and the producers an engine runs
        itself) skips
        the lossless-water notice, which states how a run will go rather than
        refusing it.
        """
        check_carrier_types(
            self.model_name, env, source, receiver,
            allow_none_receiver=bool(self._traits.none_receiver_modes))
        if announce and self._traits.consumes_volume_absorption:
            _warn_if_volume_absorption_is_missing(env, source, receiver)
        if t_start is not None:
            try:
                t_start = float(t_start)
            except (TypeError, ValueError):
                t_start = float('nan')
            if not np.isfinite(t_start):
                raise ConfigurationError(
                    f"{self.model_name}.run: t_start must be a finite time in "
                    f"seconds after emission.")
        if output_duration is not None:
            # A NaN reached int() as a raw ValueError, and a zero or negative
            # duration was silently dropped (padding only lengthens), so the
            # record came back shorter than asked with nothing said.
            try:
                duration = float(output_duration)
            except (TypeError, ValueError):
                duration = float('nan')
            if not (np.isfinite(duration) and duration > 0.0):
                raise ConfigurationError(
                    f"{self.model_name}.run: output_duration must be a "
                    f"positive number of seconds; got {output_duration!r}.")
        mode = self._resolve_run_mode(
            run_mode, default=self._default_run_mode_for(frequencies))
        if receiver is None and mode not in self._traits.none_receiver_modes:
            # The type check above admitted None for the modes that take
            # it; any other mode refuses it as the wrong carrier.
            check_carrier_types(self.model_name, env, source, receiver)
        kwargs = dict(frequencies=frequencies, source_waveform=source_waveform,
                      sample_rate=sample_rate, output_duration=output_duration,
                      t_start=t_start)
        # A keyword no mode of this model reads is refused as such first:
        # the single-mode refusal below points at the modes that read
        # frequencies=, which such a model does not have.
        never = self._run_keywords_never_consumed()
        self._reject_unsupported_run_kwargs(
            **{name: value for name, value in kwargs.items() if name in never})
        if (frequencies is not None
                and not self._traits.consumes_single_mode_frequencies
                and mode in self._traits.single_frequency_modes):
            raise ConfigurationError(
                f"{self.model_name}.run(run_mode={mode.name}) takes its "
                f"frequency from source.frequencies; frequencies= "
                f"applies to BROADBAND and TIME_SERIES runs.",
                remediation=(
                    "Drop frequencies=, or set the frequency on the "
                    "Source, e.g. Source(depths=..., frequencies=f)."))
        if t_start is not None and mode != RunMode.TIME_SERIES:
            self._warn_ignored_run_kwargs(
                mode, reason='it places the TIME_SERIES record',
                t_start=t_start)
            kwargs['t_start'] = None
        elif t_start is not None and not self._traits.consumes_run_t_start:
            self._warn_ignored_run_kwargs(
                mode, reason=(f"{self.model_name} cannot place its "
                              f"output record; it starts at emission"),
                t_start=t_start)
            kwargs['t_start'] = None
        return _RunCall(mode=mode, kwargs=kwargs)

    def _run_keywords_never_consumed(self) -> frozenset:
        """The optional run keywords no mode of this model reads: the
        TIME_SERIES keywords without ``TIME_SERIES``, and ``frequencies``
        without a broadband path (``BROADBAND`` / ``TIME_SERIES``) or OASP's
        single-mode sweep."""
        modes = set(self._supported_modes)
        consumed = set()
        if RunMode.TIME_SERIES in modes:
            consumed |= _TIME_SERIES_KEYWORDS
        if (RunMode.TIME_SERIES in modes or RunMode.BROADBAND in modes
                or self._traits.consumes_single_mode_frequencies):
            consumed.add('frequencies')
        return frozenset(_RUN_KEYWORDS) - consumed

    def _check_carriers(self, env, source, receiver, mode, *,
                        request=None) -> None:
        """The carrier-level refusals and warnings of :meth:`validate_inputs`
        for a resolved ``mode``: the source's frequencies and geometry type,
        the source/receiver geometry (:meth:`_validate_geometry`), the
        engine's knobs again (:meth:`_check_knobs`: the attributes can be
        reassigned after construction), and the engine's own refusals
        (:meth:`_validate_engine`, handed ``request``)."""
        if (mode in self._traits.single_frequency_modes
                and len(source.frequencies) > 1):
            raise self._multi_frequency_refusal(mode, source)

        if source.source_type not in self._supported_source_types:
            # ``Source`` has already validated the geometry name, so what
            # fails here is this model's coverage of a legal Source — the
            # "try another model" case ``UnsupportedFeatureError`` names.
            raise UnsupportedFeatureError(
                self.model_name,
                f"Source(source_type={source.source_type!r})",
                alternatives=[repr(t)
                              for t in sorted(self._supported_source_types)],
                alternatives_label='source geometries',
            )

        self._validate_geometry(env, source, receiver, mode)
        self._check_knobs()
        self._validate_engine(env, source, receiver, mode, request=request)

    def _multi_frequency_refusal(self, mode, source) -> Exception:
        """Engine hook: the exception :meth:`_check_carriers` raises for a
        multi-frequency Source on a single-frequency ``mode``. The base
        decides when the rule applies; this names the remedy — here
        ``BROADBAND`` / ``TIME_SERIES``, and an engine without those modes
        names its own alternative."""
        return ConfigurationError(
            f"{self.model_name}.run(run_mode={mode.name}) takes a "
            f"single source frequency; got {len(source.frequencies)}: "
            f"{frequencies_text(source.frequencies)}. For broadband H(f) use "
            f"RunMode.BROADBAND, and for time-domain p(t) use "
            f"RunMode.TIME_SERIES."
        )

    def _check_knobs(self) -> None:
        """Engine hook: refuse a knob no run could use. The engine's
        constructor calls it, and :meth:`_check_carriers` calls it again on
        every run, before :meth:`_validate_engine`, since the attributes can
        be reassigned in between (shared validators:
        :mod:`uacpy.models._knobs`). The base has no knob to check."""

    def _validate_engine(self, env, source, receiver, run_mode, *,
                         request=None) -> None:
        """Engine hook: the refusals that need only the carriers and the
        resolved ``run_mode`` — a bottom type the engine's deck cannot spell,
        a configuration this engine cannot run. Called last by
        :meth:`_check_carriers`, so a batch checked up front with
        :meth:`validate_inputs` fails where ``run`` would. ``env`` is the
        projected environment, and the checks that need the engine's
        settings (its grid, its window) are refusals of
        :meth:`_resolve_engine_settings`, which ``validate_inputs`` runs too.
        ``request`` is the checked call (see :meth:`_resolve_engine_settings`),
        ``None`` when the hook is called on its own. The base adds
        nothing."""

    def _resolve_settings(self, env, source, receiver, call: _RunCall, *,
                          announce: bool = True) -> RunSettings:
        """Resolve every setting of a checked call, once: the frequency grid,
        the time-series request, the source-depth loop, the weights and the
        speed bounds of the environment the engine runs on (``call.env``,
        projected in stage 1). The output contract is the engine's
        :attr:`outputs` entry for the mode, and the engine's own part comes
        from :meth:`_resolve_engine_settings`, whose refusals come before the
        weights-not-applied notice. ``announce=False`` (the
        :meth:`validate_inputs` path) skips that notice, which states how a
        run will go rather than refusing it.

        First, on an engine whose TIME_SERIES ignores ``frequencies=``
        (``spec.traits.time_series_ignores_frequencies``), the warning that
        it is dropped; then the band, resolved once
        (:meth:`_requested_frequencies`), whose notice is given on the modes
        the engine announces (``spec.traits.announced_band_modes``) and
        whose refusals come whatever ``announce`` says."""
        kw = call.kwargs
        ignored = self._traits.time_series_ignores_frequencies
        if (announce and call.mode == RunMode.TIME_SERIES and ignored
                and kw['frequencies'] is not None):
            warnings.warn(
                f"{self.model_name}.run(run_mode=TIME_SERIES) {ignored}; the "
                f"supplied frequencies= is ignored. Use run_mode=BROADBAND for "
                f"an explicit H(f) grid.",
                FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
        mode = call.mode
        n_depths = int(source.depths.size)
        if n_depths == 1:
            depth_loop = 'single'
        elif ((mode in self._FIELD_MODES
               and mode not in self._traits.native_multi_depth_modes)
              or mode in self._traits.python_stacked_modes):
            depth_loop = 'per_depth'
        else:
            depth_loop = 'engine'
        weighted = not source.has_unit_weights
        weights_applied = mode in self._FIELD_MODES
        time = None
        if mode == RunMode.TIME_SERIES:
            time = self._time_settings(kw)
        band = self._requested_frequencies(
            mode, source, self._call_frequencies(call), time)
        if (announce and band.notice is not None
                and mode in self._traits.announced_band_modes):
            warnings.warn(band.notice, FallbackWarning,
                          skip_file_prefixes=USER_FRAME_SKIP)
        level = getattr(source, 'source_level_dB', None)
        settings = RunSettings(
            model=self.model_name,
            mode=mode,
            frequencies=band.frequencies,
            source_depths=source.depths,
            source_type=source.source_type,
            source_level_dB=None if level is None else float(level),
            depth_loop=depth_loop,
            source_weights=source.weights if weighted else None,
            weights_applied=weights_applied,
            time=time,
            waveguide=WaveguideSpeeds(*self._speed_bounds(call.env)),
            output=self.outputs.get(mode),
        )
        refuse_a_one_depth_weight_on_dB(source, settings,
                                        model_name=self.model_name)
        engine = self._resolve_engine_settings(
            call.env, source, receiver, settings, given_env=env,
            request=call)
        if engine is not None:
            engine = dataclasses.replace(
                engine, knobs=self._given_knobs())
            check_recorded_per_launch(self, engine)
            settings = settings._replace(engine=engine)
            if announce:
                self._announce_engine_settings(
                    call.env, source, receiver, settings)
        if announce and weighted and not weights_applied:
            # Rays, arrivals, mode shapes, a reflection table: there is no
            # pressure field for a complex amplitude to scale, and silently
            # dropping the weights would let the same Source mean two things
            # on two modes.
            warnings.warn(
                f"{self.model_name}: Source(weights="
                f"{source.weights.tolist()}) is not applied in the "
                f"{mode.name} mode — a weight scales a pressure field, "
                f"and this mode returns none. Run a TL, BROADBAND or "
                f"TIME_SERIES mode for a weighted field.",
                FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
        return settings._replace(
            frequencies=self._marched_frequencies(settings))

    @classmethod
    def from_run_settings(cls, run_settings) -> 'PropagationModel':
        """The model that re-runs the run ``run_settings`` records.

        Every knob the record states is taken as it was given, and every
        value the run derived once (a phase-speed window, a grid step, a
        backend) is pinned as an explicit knob, so the re-run does not
        depend on the rules that derived them. A value derived per launch —
        one per range segment or per frequency, which no single knob holds
        — is left to the same rule; the model keeps the recorded values
        and its first run warns (:class:`~uacpy.core.exceptions.
        ProvenanceWarning`) naming each it now derives differently.

        A few recorded values are not pinned, because the run takes them
        from the call or composes them from other knobs, and as explicit
        knobs they would conflict with what they come from
        (:data:`_UNPINNED_FIELDS`); the re-run derives them again from its
        own call:

        - every OASES model's ``options``: the option line is composed from
          the other knobs, which are pinned;
        - OASR's ``reflection_type``: read from a raw ``options=`` line when
          one was given;
        - OASP's ``freq_min`` / ``freq_max``: the band edges follow
          ``run(frequencies=…)``;
        - RAM's ``stability_range_m``: by default the farthest receiver of
          the call;
        - Kraken's ``c_high`` under ``leaky_modes``: the unbounded upper
          speed ``leaky_modes`` itself sets.

        Parameters
        ----------
        run_settings : RunSettings
            ``result.run_settings`` of the run to repeat (or a record read
            back with ``RunSettings.from_dict``). Called on
            ``PropagationModel``, it builds the engine the record names.

        Returns
        -------
        PropagationModel
            A new model of the engine that wrote the record. Host settings
            that cannot change a result (``work_dir``, ``verbose``,
            ``cleanup``, ``timeout``, the binary's path) are this machine's
            defaults.

        Raises
        ------
        ConfigurationError
            A record of another engine, one with no engine settings, or one
            naming a knob the engine does not take.

        Examples
        --------
        >>> tl = Kraken().run(env, source, receiver)
        >>> again = Kraken.from_run_settings(tl.run_settings).run(
        ...     env, source, receiver)
        """
        return model_from_run_settings(cls, run_settings)

    #: Settings fields named after a knob that :meth:`from_run_settings`
    #: does not pin: values the run takes from the call itself (a band edge
    #: ``run(frequencies=…)`` sets, a range from the receiver) or composes
    #: from other knobs (an OASES option line), which as an explicit knob
    #: would conflict with what they were derived from.
    _UNPINNED_FIELDS: frozenset = frozenset()

    @classmethod
    def _per_launch_knob_values(cls, engine) -> Dict[str, list]:
        """Engine hook: the values each knob the run derives per launch
        took, in launch order, from the settings record ``engine``: one per
        range segment (Kraken), one per frequency (RAM). A knob whose
        values all agree is pinned as one; the base has none."""
        return {}

    #: Knobs that cannot change a result — where scratch files go, how loud
    #: the log is, the timeout, the path of the binary (its kind is the
    #: settings' ``backend``) — and so are not recorded in the settings.
    _HOST_KNOBS = frozenset({'use_tmpfs', 'verbose', 'work_dir', 'cleanup',
                             'timeout', 'executable', 'field_executable'})

    def _given_knobs(self) -> Dict[str, Any]:
        """Every constructor knob of this model by name, as given (``None``
        where the run derives it), in plain types: the ``knobs`` its
        settings record carries beside the resolved fields. Host knobs
        (:data:`_HOST_KNOBS`) are left out."""
        return {name: _plain(getattr(self, name))
                for name, _default in _collect_init_params(type(self))
                if name not in self._HOST_KNOBS and hasattr(self, name)}

    def _resolve_engine_settings(self, env, source, receiver,
                                 settings: RunSettings, *, given_env,
                                 request: Optional[_RunCall] = None):
        """Engine hook: the engine's own resolved settings (its grids,
        windows, beams, the binary it launches) as an
        :class:`~uacpy.core.run_settings.EngineSettings`, computed from the
        carriers and the shared ``settings``, and the refusals of settings
        the engine cannot run (a grid with too few points, say). ``env`` is
        the environment the engine runs on — projected in stage 1 — and
        ``given_env`` the one the caller passed, for a rule that reads the
        problem as given. ``request`` is the checked call (its run mode and
        the keywords the user passed), the one way the hook sees what the
        user asked for rather than what the base resolved from it. Pure:
        writes no file and starts no process. ``None`` (the base) for an
        engine with no settings of its own."""
        return None

    def _announce_engine_settings(self, env, source, receiver,
                                  settings: RunSettings) -> None:
        """Stage 3, after :meth:`_resolve_engine_settings`: one warning per
        notice the engine's settings recorded, its ``message`` as the
        ``category`` it carries (:class:`~uacpy.core.run_settings.Notice`)
        (``settings.engine.notices``: a window that drops paths the geometry
        needs, a knob the binary overrides). Called by :meth:`run` and
        :meth:`run_settings`, never by :meth:`validate_inputs`, which states
        only what is refused. The conditions themselves are decided by
        :meth:`_resolve_engine_settings`; this hook only presents them. An
        engine overrides it only to announce a nested producer's settings
        before its own."""
        for notice in settings.engine.notices:
            warnings.warn(notice.message, notice.category,
                          skip_file_prefixes=USER_FRAME_SKIP)

    def _time_settings(self, kwargs) -> TimeSettings:
        """The TIME_SERIES request of a checked call. The pulse is stored as
        the synthesis uses it (real, float64, zero-padded to
        ``output_duration`` when ``spec.traits.pads_pulse_to_output_duration``)
        when it is a usable pulse, the float64 1-D array :func:`_real_pulse`
        admits. A pulse the engine will refuse is stored as ``None`` and the
        refusal is the engine's (:meth:`_require_timeseries_signal`)."""
        sample_rate = kwargs['sample_rate']
        output_duration = kwargs['output_duration']
        pulse = None
        if kwargs['source_waveform'] is not None:
            clean, problem = _real_pulse(kwargs['source_waveform'])
            if problem is None:
                pulse = np.asarray(clean, dtype=float).ravel()
                if (_positive_finite(sample_rate)
                        and self._traits.pads_pulse_to_output_duration):
                    pulse = np.asarray(pad_waveform_to_duration(
                        pulse, float(sample_rate), output_duration),
                        dtype=float)
        return TimeSettings(
            source_waveform=pulse,
            sample_rate=(float(sample_rate) if _positive_finite(sample_rate)
                         else None),
            output_duration=(None if output_duration is None
                             else float(output_duration)),
            t_start=kwargs['t_start'])

    def _call_frequencies(self, call: _RunCall):
        """The call's ``frequencies=`` as the band rules read it: ``None`` on
        a TIME_SERIES run of an engine whose TIME_SERIES ignores it
        (``spec.traits.time_series_ignores_frequencies``)."""
        if (call.mode == RunMode.TIME_SERIES
                and self._traits.time_series_ignores_frequencies):
            return None
        return call.kwargs['frequencies']

    def _requested_frequencies(self, mode, source, frequencies,
                               time: Optional[TimeSettings]) -> BandResolution:
        """Engine hook, stage 3: the frequency grid (Hz) the call asks for
        and the notice that states it, as a
        :class:`~uacpy.models._band.BandResolution` — by the base rules
        (:func:`~uacpy.models._band.resolve_band`, with this model's band
        knobs, :meth:`_broadband_band_knobs`), resolved once. ``frequencies``
        is the call's keyword as the band rules read it
        (:meth:`_call_frequencies`). Stage 3 gives the notice on the modes the
        engine announces (``spec.traits.announced_band_modes``), resolves the
        engine's own settings from the grid, and :meth:`_marched_frequencies`
        records the grid the engine propagates."""
        n_freqs, bandwidth_factor = self._broadband_band_knobs()
        return resolve_band(mode, source, frequencies, time,
                            model_name=self.model_name, n_freqs=n_freqs,
                            bandwidth_factor=bandwidth_factor)

    def _marched_frequencies(self, settings: RunSettings):
        """Engine hook, the end of stage 3: the frequency grid (Hz) the
        engine propagates for the resolved ``settings`` — what
        ``settings.frequencies`` records. The base returns the requested grid;
        an engine that propagates a grid of its own (SPARC's deck frequency,
        OASP's FFT ladder, OASSP's mean-field bins) returns that."""
        return settings.frequencies

    def _execute_call(self, env, source, receiver, call: _RunCall,
                      settings: RunSettings):
        """Stages 4-5 of a checked call with its resolved ``settings``, on
        the projected environment: one :meth:`_run_engine` per call of the
        engine, inside the source-depth loop and the weights of
        :func:`~uacpy.models._stacking.run_depth_loop`."""
        return run_depth_loop(
            source, settings,
            lambda single: self._run_engine(call.env, single, receiver,
                                            settings),
            pad_time_axes=call.kwargs.get('output_duration') is None,
            model_name=self.model_name, verbose=self.verbose,
            scratch_subdir=self._scratch_subdir)

    def _run_engine(self, env, source, receiver, settings: RunSettings):
        """Stages 4-5 for one call of the engine: in a fresh work directory
        (a pinned one is claimed), for each of the :meth:`_n_launches`
        launches write the deck, launch the binary and read its output —
        launch ``i`` seeing the outputs of the launches before it as
        ``inputs.earlier`` — then build the result and check it against the
        declared output contract. Every launch shares the one directory,
        which is let go once, on every exit path (wiped iff ``cleanup``).

        ``_to_result`` receives the deck and the output of the one launch,
        or the lists of every launch's when there are more; its ``inputs``
        are the first launch's."""
        n_launches = self._n_launches(settings)
        prepared = self._prepare_launches(env, settings)
        fm = self._setup_file_manager()
        try:
            first = StageInputs(work_dir=fm.work_dir, env=env, source=source,
                                receiver=receiver, settings=settings,
                                prepared=prepared)
            decks, raws = [], []
            for launch in range(n_launches):
                inputs = dataclasses.replace(first, launch=launch,
                                             earlier=tuple(raws))
                deck = self._write_input(inputs)
                self._launch(inputs, deck)
                decks.append(deck)
                raws.append(self._read_output(inputs, deck))
            if n_launches == 1:
                result = self._to_result(first, decks[0], raws[0])
            else:
                result = self._to_result(first, decks, raws)
        finally:
            fm.finish()
        self._check_output_contract(result, settings)
        return result

    def _n_launches(self, settings: RunSettings) -> int:
        """How many binary launches one call makes, all in one work
        directory (:meth:`_run_engine`), decided from the resolved
        ``settings`` — a band marched one frequency per launch, a producer
        run and the run that reads its output. One by default."""
        return 1

    def _prepare_launches(self, env, settings: RunSettings):
        """What every launch of the call reads and is built once, before
        the first (a deck shared by a band's launches, say): handed to each
        launch as ``inputs.prepared``. ``None`` by default."""
        return None

    @abstractmethod
    def _write_input(self, inputs: StageInputs) -> Path:
        """Stage 4: write the input deck(s) into ``inputs.work_dir`` with
        the io writers, from ``inputs.settings``, and return the path of the
        deck :meth:`_launch` hands the binary."""

    @abstractmethod
    def _launch(self, inputs: StageInputs, deck: Path) -> None:
        """Stage 4: run the binary on ``deck`` in ``inputs.work_dir``,
        as a :class:`~uacpy.models._launch.Launch` through
        :meth:`_launch_binary` (:meth:`_run_and_attach_prt` for a binary that
        writes a ``.prt``)."""

    @abstractmethod
    def _read_output(self, inputs: StageInputs, deck: Path):
        """Stage 4: read what the binary wrote with the io readers and
        return it as they return it; raise :class:`ModelExecutionError`
        when the binary wrote nothing usable."""

    @abstractmethod
    def _to_result(self, inputs: StageInputs, deck: Path, raw):
        """Stage 5: the result built from ``raw`` in the package's
        conventions — its class and conventions those of
        ``inputs.settings.output``, its identity from
        :meth:`_result_kwargs` — with the output paths attached
        (:meth:`_attach_output_paths`)."""

    def _check_output_contract(self, result, settings: RunSettings) -> None:
        """:func:`~uacpy.models._extract.check_output_contract` for this
        model."""
        check_output_contract(self.model_name, result, settings)

    def _stamp_run_settings(self, result, settings: RunSettings):
        """Stage 6: record on ``result`` (every slab of a ``ResultStack``)
        the run mode and the settings this call ran with, and on a
        ``Field`` the coherence its mode declares
        (``settings.output.coherent``, ``None`` where the question has no
        answer, as on a time trace). Returns ``result``.

        Where the engine's body chose the frequency grid itself, the
        stamped settings carry the grid the result was computed on, with a
        note naming what the base rules had resolved."""
        slabs = _slabs_of(result)
        if slabs:
            settings = _settings_as_run(settings, slabs[0])
        spec = settings.output
        for slab in slabs:
            slab.run_mode = settings.mode
            slab._run_settings = settings
            if spec is not None and spec.result_type == 'Field':
                slab._coherent = (None if slab.kind != 'pressure'
                                  or spec.coherent is None
                                  else bool(spec.coherent))
        return result

    # Modes that evaluate nothing on the receiver range axis, so a receiver
    # whose ranges are all 0 m is a legal input: mode shapes and plane-wave
    # reflection coefficients read no receiver range, a ray fan is bounded
    # by ``r_box`` (which falls back to 10 km), and the OASN array products
    # ignore ``receiver.ranges`` outright.
    _NO_RANGE_AXIS_MODES: 'frozenset[RunMode]' = frozenset({
        RunMode.RAYS, RunMode.MODES, RunMode.REFLECTION,
        RunMode.COVARIANCE, RunMode.REPLICA,
    })

    # Modes whose result is a gridded ``Field`` of pressure. A multi-depth
    # ``Source`` in one of these runs once per depth through
    # ``_stacking.run_per_source_depth`` and returns a ``ResultStack`` whose slabs
    # ``ResultStack.superpose`` can add; every other mode returns something
    # (mode shapes, rays, a reflection table, an array product) that has no
    # per-source linear sum, so ``_validate_geometry`` refuses the extra
    # depths there unless the model stacks them itself.
    _FIELD_MODES: 'frozenset[RunMode]' = frozenset({
        RunMode.COHERENT_TL, RunMode.INCOHERENT_TL, RunMode.SEMICOHERENT_TL,
        RunMode.BROADBAND, RunMode.TIME_SERIES,
    })

    def _default_run_mode(self) -> RunMode:
        """The mode ``run(run_mode=None)`` resolves to: the first declared
        mode. Kraken overrides it, because its list opens with ``MODES``
        while its ``run`` defaults to a TL mode."""
        return self._supported_modes[0]

    def _default_run_mode_for(self, frequencies) -> RunMode:
        """The mode ``run(run_mode=None, frequencies=...)`` resolves to. The
        base ignores ``frequencies``; Kraken overrides it, because a
        multi-element ``frequencies=`` makes its default ``BROADBAND``."""
        return self._default_run_mode()

    def _stacks_source_depths(self, mode) -> bool:
        """Whether a multi-depth ``Source`` is legal in ``mode``: a field
        mode (the base loops, or the engine batches them), a mode the engine
        batches into one deck, or one this model stacks by its own loop."""
        return (mode in self._FIELD_MODES
                or mode in self._traits.native_multi_depth_modes
                or mode in self._traits.python_stacked_modes)

    @contextlib.contextmanager
    def _scratch_subdir(self, name: str):
        """Run the block with a pinned ``work_dir`` redirected into its
        subdirectory ``name``; an unpinned one (a fresh temp dir per run)
        needs no such split.

        The redirect is recorded in a thread-local map keyed by this
        instance rather than written to ``self.work_dir``: the attribute is
        shared state that :meth:`_setup_file_manager` reads on every run, so
        mutating it made a concurrent single-depth run on the same model
        claim — and be refused for — a ``source_depth_*`` directory it never
        asked for, and left the instance pointing inside a subdirectory if a
        run raised between the two writes. Keying by ``id(self)`` keeps a
        model a hook runs from inheriting the redirect of the model that runs
        it."""
        if self.work_dir is None:
            yield
            return
        by_model = getattr(_SCRATCH_REDIRECT, 'by_model', None)
        if by_model is None:
            by_model = _SCRATCH_REDIRECT.by_model = {}
        saved = by_model.get(id(self))
        by_model[id(self)] = name
        try:
            yield
        finally:
            if saved is None:
                by_model.pop(id(self), None)
            else:
                by_model[id(self)] = saved

    def _require_timeseries_signal(
        self,
        run_mode: 'RunMode',
        source_waveform,
        sample_rate,
    ) -> None:
        """Refuse an unusable TIME_SERIES signal pair; returns nothing.

        Raises :class:`ConfigurationError` when the caller asked for a
        :attr:`RunMode.TIME_SERIES` result but did not supply both
        ``source_waveform`` and ``sample_rate``, when the waveform is not a
        real finite pressure pulse, or when ``sample_rate`` is not a
        positive finite number.

        The waveform rule is the one every waveform entry point applies
        (:func:`_real_pulse`): a real, finite, 1-D array. A complex waveform
        is refused whatever its imaginary part, and a ``(time, signal)`` pair
        or a 2-D array with the pair hint.

        Used by every wrapper that synthesises p(t) from a broadband
        transfer function (Bellhop, RAM, Scooter, Kraken, OASP), and by
        SPARC for the series a ``pulse_type`` opening with ``'F'`` / ``'B'``
        stages as ``STSFIL``.
        """
        if run_mode == RunMode.TIME_SERIES and (
            source_waveform is None or sample_rate is None
        ):
            raise ConfigurationError(
                f"{self.model_name}.run(run_mode=TIME_SERIES) requires "
                f"source_waveform and sample_rate. For the broadband "
                f"transfer function H(f), use run_mode=RunMode.BROADBAND."
            )
        if run_mode == RunMode.TIME_SERIES and sample_rate is not None:
            # Refuse a sample rate that is not a positive finite number. The
            # test is NaN-closed (``not (sr > 0)`` rather than ``sr <= 0``)
            # because nan compares False both ways and would slip through:
            # nothing downstream stops it either — pad_waveform_to_duration
            # and time_series_band (models/_band.py) no-op on a
            # non-positive rate — so it surfaces as a ZeroDivisionError in
            # the delay-and-sum, a raw ValueError from the deep guard in
            # acoustic_signal/_synthesis.py (``waveform_synthesis_setup``),
            # or a trace on a descending time axis.
            try:
                sr = float(sample_rate)
            except (TypeError, ValueError):
                sr = float('nan')
            if not np.isfinite(sr) or not (sr > 0.0):
                raise ConfigurationError(
                    f"{self.model_name}.run(run_mode=TIME_SERIES): "
                    f"sample_rate must be a positive finite number of Hz; "
                    f"got {sample_rate!r}."
                )
        if run_mode == RunMode.TIME_SERIES and source_waveform is not None:
            _, problem = _real_pulse(source_waveform)
            if problem is not None:
                raise ConfigurationError(
                    f"{self.model_name}.run(run_mode=TIME_SERIES): {problem}")

    def _broadband_band_knobs(self):
        """``(n_freqs, bandwidth_factor)`` a single BROADBAND carrier expands
        with; ``None`` takes the package default
        (:data:`~uacpy.models._defaults.DEFAULT_BROADBAND_N_FREQS`,
        :data:`~uacpy.models._defaults.DEFAULT_BROADBAND_BANDWIDTH_FACTOR`).
        A model with constructor knobs for the band returns them."""
        return None, None

    def _finish_broadband(self, result, settings: RunSettings):
        """Return the BROADBAND transfer function as computed or, for
        TIME_SERIES, its synthesis with the pulse of ``settings.time``
        (validated in stage 2, :meth:`_check_time_series_keywords`). Every
        IFFT-based wrapper ends its broadband route here.

        The transfer function carries ``settings`` while it is synthesised,
        so the synthesis reads the waveguide the run resolved
        (``run_settings.waveguide``) as every Field does; stage 6 stamps
        the returned result.
        """
        if settings.mode != RunMode.TIME_SERIES:
            return result
        for slab in _slabs_of(result):
            slab._run_settings = settings
        time = settings.time
        return result.synthesize_time_series(
            source_waveform=time.source_waveform,
            sample_rate=time.sample_rate, t_start=time.t_start)

    def scratch_policy(self) -> ScratchPolicy:
        """Where this model's runs put their scratch files and what happens
        to them: the pinned ``work_dir`` (``None`` for a fresh temporary
        directory per run), whether the files are kept (``cleanup=False``)
        and whether a RAM-backed directory was asked for (``use_tmpfs``).
        ``run_parallel`` reads it to decide which work dirs a batch leaves
        behind."""
        return ScratchPolicy(
            pinned_dir=(None if self.work_dir is None
                        else Path(self.work_dir)),
            keeps_files=self.cleanup is False,
            on_tmpfs=bool(self.use_tmpfs))

    def _setup_file_manager(self) -> FileManager:
        """This run's :class:`~uacpy.models._workspace.FileManager`
        (:func:`~uacpy.models._workspace.setup_file_manager`): the pinned
        ``work_dir`` — or its per-depth subdirectory while a per-depth loop
        is inside one depth (:func:`~uacpy.models._workspace.
        _effective_work_dir`) — claimed for this thread, or a fresh temp dir,
        wiped at the end iff ``cleanup``."""
        return setup_file_manager(
            _effective_work_dir(self), model_name=self.model_name,
            use_tmpfs=self.use_tmpfs, cleanup=self.cleanup)

    def _log(self, message: str, level: str = "info"):
        """Emit a tagged line through :func:`uacpy._log.log_message`.
        ``WARN`` / ``ERROR`` always print; ``INFO`` / ``DEBUG`` only when
        ``self.verbose``."""
        log_message(
            self.model_name, message,
            verbose=self.verbose, level=level,
        )

    def validate_inputs(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[Union['RunMode', str]] = None,
        *,
        frequencies: Optional[np.ndarray] = None,
        source_waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        output_duration: Optional[float] = None,
        t_start: Optional[float] = None,
    ) -> None:
        """
        Raise what :meth:`run` would raise for this call before launching.

        Takes exactly the arguments of :meth:`run` and runs the same checks,
        in the same order: the carrier types, ``t_start`` and
        ``output_duration``, the run mode (resolved as ``run()`` resolves it,
        ``frequencies=`` included), the keyword rule, and then the carriers
        against the resolved mode — the source's frequencies and geometry
        type, the source/receiver geometry, and the engine's own refusals of
        a carrier it cannot run (a bottom type its deck cannot spell, say).
        So a batch checked with it up front fails where ``run`` would.

        Parameters
        ----------
        env : Environment
            Environment to validate against.
        source : Source
            Source to validate.
        receiver : Receiver
            Receiver to validate.
        run_mode : RunMode or str, optional
            Run mode, resolved as ``run()`` resolves it: ``None`` stands for
            the model's default mode and a value string (``'coherent_tl'``)
            for its enum member. Single-frequency modes (``COHERENT_TL``,
            ``RAYS``, ``MODES``, …) refuse a Source with more than one
            frequency.
        frequencies, source_waveform, sample_rate, output_duration, t_start
            Keyword-only, as :meth:`run` takes them.

        Raises
        ------
        UnsupportedFeatureError
            If ``run_mode`` is not a mode this model runs, or a keyword is
            one no mode of this model reads.
        InvalidDepthError
            If source depths exceed the model's resolvable depth.
        ConfigurationError
            If source/receiver depths are negative, if ``run_mode`` is a
            single-frequency mode and ``source`` carries multiple
            frequencies (or ``frequencies=`` is passed), if
            ``source.source_type`` is outside ``_supported_source_types``,
            or if ``source.beam_pattern`` is set on a model that does not
            read one.

        Notes
        -----
        The source/receiver geometry checks — including the source's angular
        geometry, i.e. its beam pattern — live in :meth:`_validate_geometry`,
        which a model that reads no geometry
        (:class:`~uacpy.models.bounce.Bounce`) overrides to a no-op; an
        engine's own refusals live in :meth:`_validate_engine`.

        The checks read the environment projected from ``env``, as
        ``run()``'s do, and the run's settings are resolved too, so the
        refusals of settings the engine cannot run (a grid too coarse, a
        window too narrow) come from here as well.
        """
        call = self._check_call(
            env, source, receiver, run_mode, frequencies=frequencies,
            source_waveform=source_waveform, sample_rate=sample_rate,
            output_duration=output_duration, t_start=t_start, announce=False)
        call = self._check_carriers_of_call(env, source, receiver, call)
        self._resolve_settings(env, source, receiver, call, announce=False)

    def _validate_geometry(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional['RunMode'] = None,
    ) -> None:
        """Check the source/receiver geometry against what the model resolves.

        Split out of :meth:`validate_inputs` so a model that reads no
        geometry at all can opt out wholesale. The source's angular geometry
        (its beam pattern) belongs here for that reason: a reflection-only
        engine never launches a ray fan, so rejecting a pattern it simply
        ignores would break reusing one ``Source`` across models.
        """
        if (source.beam_pattern is not None
                and not self._supports_source_beam_pattern):
            raise ConfigurationError(
                f"{self.model_name} does not read a source beam pattern; "
                f"drop Source(beam_pattern=...) or use Bellhop or Kraken."
            )

        # Whether a multi-depth Source is legal is a question about the
        # MODE, not about the model: ``_supports_multi_source_depth`` says
        # the engine batches depths into one deck *somewhere*, which is the
        # declarative capability the matrix prints, and would wave through a
        # mode of that same engine which stacks nothing (Kraken MODES).
        n_depths = len(np.atleast_1d(source.depths))
        if n_depths > 1:
            mode = (run_mode if run_mode is not None
                    else self._default_run_mode())
            if not self._stacks_source_depths(mode):
                raise ConfigurationError(
                    f"{self.model_name} takes a single source depth per "
                    f"{mode.name} run; got {n_depths}: "
                    f"{np.asarray(source.depths).tolist()}. A multi-depth "
                    f"Source stacks "
                    f"only in the field modes (COHERENT_TL / INCOHERENT_TL "
                    f"/ SEMICOHERENT_TL / BROADBAND / TIME_SERIES); for "
                    f"{mode.name} loop over single-depth Sources externally."
                )

        resolvable_depth = self._check_source_depths(env, source)

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
                ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )

        if receiver is None:
            # A mode with no receiver geometry (Kraken MODES builds its own
            # depth grid): nothing below reads one.
            return

        if receiver.depth_min < 0:
            raise ConfigurationError("Receiver depths must be non-negative.")

        # A point source's field carries a 1/sqrt(r) factor that is singular
        # on the source axis, so a grid whose largest range is 0 m has no
        # cell any engine can fill: every wrapper NaNs the r = 0 column, and
        # the PE grid chooser divides by a range step it sizes from r_max.
        # Modes that evaluate nothing on the receiver range axis are exempt
        # (see ``_NO_RANGE_AXIS_MODES``); a mixed grid such as
        # ``ranges=[0, 1000]`` keeps its NaN column and the masking warning.
        if (run_mode not in self._NO_RANGE_AXIS_MODES
                and receiver.range_max <= 0.0):
            raise ConfigurationError(
                f"{self.model_name}: the receiver's largest range is 0 m; "
                f"give at least one positive receiver range (a field at "
                f"r = 0 is undefined for a point source).",
                remediation="Receiver(depths=<your depths>, ranges=[1000.0])",
            )

        warn_receiver_below_resolvable(self.model_name, env, receiver,
                                       resolvable_depth)
        self._check_per_range_receiver_depth(env, receiver)
        warn_on_range_coverage(self.model_name, env, receiver)
        warn_on_ssp_start(self.model_name, env)

    def _check_source_depths(self, env: 'Environment',
                             source: 'Source') -> float:
        """Refuse a source depth this model cannot resolve; return the
        resolvable depth (:meth:`_max_receiver_depth`).

        The source injects energy into the medium, so it must sit within
        what the model resolves — placing it below is a hard error, and so
        is a negative depth. Receivers are outputs: ones below the
        resolvable depth are accepted and return the model's below-domain
        value (transmitted / evanescent field, or NaN inside a PE absorbing
        layer); :meth:`_validate_geometry`'s warning helpers surface that
        rather than rejecting it.
        """
        resolvable_depth = self._max_receiver_depth(env)
        if np.any(source.depths > resolvable_depth):
            exc = InvalidDepthError(
                float(source.depths.max()), resolvable_depth, "Source",
            )
            note = self._source_below_domain_note(env, resolvable_depth)
            if note:
                # Message enrichment only. ``InvalidDepthError.__reduce__``
                # rebuilds from the three constructor arguments, so a copy
                # that crosses a process boundary (``run_parallel``) carries
                # the base remediation.
                exc.remediation = f"{exc.remediation}.\n\n{note}"
            raise exc
        if np.any(source.depths < 0):
            raise ConfigurationError("Source depths must be non-negative.")
        return resolvable_depth

    def _receiver_grid_is_paired(self, receiver: 'Receiver') -> bool:
        """Whether this model's deck pairs ``receiver.depths[i]`` with
        ``receiver.ranges[i]`` instead of spanning their Cartesian product.

        Returns ``False`` here because every wrapper but one writes a
        rectilinear grid; Bellhop overrides it for ``grid_type='I'``
        (``RunType(5:5)='I'``). The paired flag lives on the model rather
        than on :class:`Receiver` because the same Receiver written to a
        rectilinear deck genuinely does span the product.
        """
        return False

    def _check_per_range_receiver_depth(
        self, env: 'Environment', receiver: 'Receiver',
    ) -> None:
        """:func:`~uacpy.models._checks.check_per_range_receiver_depth` for
        this model, on the (depth, range) pairs its deck carries
        (:meth:`_receiver_grid_is_paired`)."""
        check_per_range_receiver_depth(
            self.model_name, env, receiver,
            paired=self._receiver_grid_is_paired(receiver),
            verbose=self.verbose)

    def compute_tl(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        *,
        run_mode: Optional['RunMode'] = None,
    ) -> Union[Result, ResultStack]:
        """Compute the coherent (or incoherent) complex pressure field on
        the receiver grid; ``.tl`` / ``.dB`` give the transmission loss
        (thin wrapper around ``run``).

        Parameters
        ----------
        env : Environment
            Ocean environment.
        source : Source
            Acoustic source.
        receiver : Receiver
            Receiver grid. Required — depth/range resolution is a physical
            decision and is not auto-generated.
        run_mode : RunMode or str, optional
            ``COHERENT_TL`` (default), ``INCOHERENT_TL`` or
            ``SEMICOHERENT_TL``, or its value string (``'coherent_tl'``)
            as ``run()`` accepts. Other modes raise — call ``model.run()``
            directly for those.

        Returns
        -------
        result : Result or ResultStack
            Transmission loss field, or a stack of them over source depth:
            a ``Source`` with more than one depth hands back a
            ``ResultStack`` from every model. An engine whose
            ``spec.traits.native_multi_depth_modes`` lists the mode writes every depth
            into one deck and its reader splits the slabs; every other engine
            runs once per depth (:func:`~uacpy.models._stacking.run_per_source_depth`).
            ``stack.superpose()`` adds
            the slabs' complex pressure with ``source.weights``.
            ``ResultStack`` is not a ``Result`` subclass, so a caller that
            annotates the result has to name both.

        Examples
        --------
        >>> bellhop = Bellhop()
        >>> rcv = uacpy.Receiver(depths=np.linspace(0, env.depth, 50),
        ...                       ranges=np.linspace(100, 10_000, 100))
        >>> tl = bellhop.compute_tl(env, source, rcv)
        """
        # Every ``compute_*`` names the engines that run its mode, read from
        # the engine registry when the refusal is raised (every engine module
        # is importable by then), so a new engine joins the advice without
        # an edit here.
        self._require_mode(RunMode.COHERENT_TL, "transmission loss computation")
        if run_mode is None:
            run_mode = RunMode.COHERENT_TL
        tl_modes = (
            RunMode.COHERENT_TL, RunMode.INCOHERENT_TL, RunMode.SEMICOHERENT_TL,
        )
        # ``run()`` accepts a RunMode value string (``'coherent_tl'``), so
        # the membership test runs on the coerced enum.
        if isinstance(run_mode, str):
            try:
                run_mode = RunMode(run_mode)
            except ValueError:
                pass
        if run_mode not in tl_modes:
            got = run_mode.name if isinstance(run_mode, RunMode) else repr(run_mode)
            raise ConfigurationError(
                f"compute_tl() got run_mode={got}; only COHERENT_TL / "
                f"INCOHERENT_TL / SEMICOHERENT_TL (or their value strings "
                f"{[m.value for m in tl_modes]}) are accepted. Call "
                f"{self.model_name}.run(run_mode=…) for other modes."
            )
        return self.run(env, source, receiver, run_mode=run_mode)

    def compute_rays(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
    ) -> Union[Result, ResultStack]:
        """Compute ray paths (thin wrapper around ``run``).

        ``receiver`` is required — the receiver grid defines the ray-box
        extent and recording locations and is not auto-generated.

        Parameters
        ----------
        env : Environment
            The environment.
        source : Source
            The source.
        receiver : Receiver
            The receiver grid.

        Returns
        -------
        result : Result or ResultStack
            Ray paths, or a stack of them over source depth.
            The stacking is done by the readers, not by the caller: a
            multi-depth ``.shd`` / ``.arr`` / ``.ray`` is split into one
            slab per source depth and bundled
            (``uacpy/io/oalib_reader.py``), so a run over a ``Source`` with
            more than one depth hands back a ``ResultStack`` over
            ``source_depth``. ``ResultStack`` is not a ``Result`` subclass,
            so a caller that annotates the result has to name both.

        Examples
        --------
        >>> bellhop = Bellhop()
        >>> rcv = uacpy.Receiver(depths=np.array([env.depth / 2]),
        ...                       ranges=np.linspace(0, 10_000, 50))
        >>> rays = bellhop.compute_rays(env, source, rcv)
        """
        self._require_mode(RunMode.RAYS, "ray path computation")
        return self.run(env, source, receiver, run_mode=RunMode.RAYS)

    def compute_arrivals(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
    ) -> Union[Result, ResultStack]:
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
        result : Result or ResultStack
            Arrival data, or a stack of it over source depth.
            The stacking is done by the readers, not by the caller: a
            multi-depth ``.shd`` / ``.arr`` / ``.ray`` is split into one
            slab per source depth and bundled
            (``uacpy/io/oalib_reader.py``), so a run over a ``Source`` with
            more than one depth hands back a ``ResultStack`` over
            ``source_depth``. ``ResultStack`` is not a ``Result`` subclass,
            so a caller that annotates the result has to name both.

        Raises
        ------
        UnsupportedFeatureError
            If the model does not support arrival computation.

        Examples
        --------
        >>> bellhop = Bellhop()
        >>> arrivals = bellhop.compute_arrivals(env, source, receiver)
        """
        self._require_mode(RunMode.ARRIVALS, "arrival computation")
        return self.run(env, source, receiver, run_mode=RunMode.ARRIVALS)

    def compute_modes(
        self,
        env: Environment,
        source: Source,
        n_modes: Optional[int] = None,
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
        nothing for the caller to position. It is ``run(env, source, None,
        run_mode=RunMode.MODES)`` with ``n_modes`` as the call's mode cap: the
        engine builds the depth grid the modes are tabulated on, and
        ``run_settings`` with the same arguments previews it.

        Examples
        --------
        >>> kraken = Kraken()
        >>> modes = kraken.compute_modes(env, source, n_modes=50)
        >>> wavenumbers = modes.k
        >>> mode_shapes = modes.phi
        """
        self._require_mode(RunMode.MODES, "normal mode computation")

        if n_modes is not None and (
                isinstance(n_modes, bool)
                or not isinstance(
                    n_modes, (int, float, np.integer, np.floating))):
            raise ConfigurationError(
                f"compute_modes: n_modes must be an int or None; got "
                f"{type(n_modes).__name__}. Unlike the other compute_* "
                f"wrappers, compute_modes takes no receiver (normal modes are "
                f"receiver-independent) — its third argument is n_modes. Pass "
                f"it by keyword: compute_modes(env, source, n_modes=...).")
        # Refuse a cap that is not a whole number. The backend applies it as
        # int(n_modes), which truncates toward zero, so a fractional value
        # would silently run a different cap than asked for and a non-finite
        # one would raise from int() with no context.
        if n_modes is not None and not (
                np.isfinite(n_modes) and float(n_modes).is_integer()):
            raise ConfigurationError(
                f"compute_modes: n_modes must be a whole number of modes or "
                f"None; got {n_modes!r}. The cap is applied as int(n_modes), "
                f"which truncates toward zero.")
        if n_modes is not None and n_modes < 1:
            raise ConfigurationError(
                f"compute_modes: n_modes must be >= 1 or None; got "
                f"{n_modes!r}, a cap that keeps no mode.")

        # Every source depth is checked here, as run() checks it: the modes
        # are solved once for the whole Source (they do not depend on the
        # source depth; Kraken tabulates them at every depth), and a depth
        # outside the domain must fail before the solve.
        check_carrier_types(self.model_name, env, source, None,
                            allow_none_receiver=True)
        self._check_source_depths(env, source)
        # A unit copy of the source: MODES applies no weight, and the
        # caller's weights would only be announced as unused.
        unit = Source(depths=source.depths, frequencies=source.frequencies,
                      source_type=source.source_type,
                      beam_pattern=source.beam_pattern,
                      source_level_dB=source.source_level_dB)
        # The env is passed through as-is: the run projects it exactly once
        # (Kraken's MODES path reduces a range-dependent env to its r=0
        # profile and then projects).
        call = self._check_call(
            env, unit, None, RunMode.MODES, frequencies=None,
            source_waveform=None, sample_rate=None, output_duration=None,
            t_start=None)
        return self._run_call(env, unit, None, dataclasses.replace(
            call, engine_request=ModesRequest(
                n_modes=None if n_modes is None else int(n_modes))))

    def compute_eigenrays(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
    ) -> Union[Result, ResultStack]:
        """Compute eigenrays — rays that arrive at the receiver(s).

        Thin wrapper around ``run(run_mode=RunMode.EIGENRAYS)``. Returns
        the raw :class:`Rays` from the solver — or, for a multi-depth
        ``Source``, a ``ResultStack`` of them over ``source_depth``
        (Bellhop runs one eigenray deck per depth in
        ``_run_eigenrays_multi_depth``).
        For a single-point target build a 1-point ``Receiver`` first:

        >>> receiver = uacpy.Receiver(depths=[30.0], ranges=[2000.0])
        >>> rays = bellhop.compute_eigenrays(env, source, receiver)
        >>> close = rays.top_n_by_miss(8).truncate_at_receiver()
        >>> direct = rays.filter_by_bounces(kind='direct')
        >>> within = rays.filter_by_miss_distance(max_miss=15.0)

        Parameters
        ----------
        env : Environment
            The environment.
        source : Source
            The source.
        receiver : Receiver
            The receiver grid.
        """
        self._require_mode(RunMode.EIGENRAYS, "eigenray computation")
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
        (everything except Bounce and OASR) raise
        :class:`UnsupportedFeatureError`.

        Parameters
        ----------
        env : Environment
            The environment.
        source : Source
            The source.
        receiver : Receiver
            The receiver grid.
        """
        self._require_mode(RunMode.REFLECTION, "reflection coefficient computation")
        return self.run(env, source, receiver, run_mode=RunMode.REFLECTION)

    def compute_time_series(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        *,
        source_waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        output_duration: Optional[float] = None,
        frequencies: Optional[np.ndarray] = None,
        t_start: Optional[float] = None,
    ) -> Union[Result, ResultStack]:
        """Compute time-domain pressure p(t) at the receiver(s).

        Forwards ``source_waveform``, ``sample_rate``, ``output_duration``,
        ``frequencies`` and ``t_start`` to
        ``run(run_mode=RunMode.TIME_SERIES)``. ``output_duration`` is the
        record's length and ``t_start`` the time of its first sample
        (seconds after emission; ``None`` opens it just before the earliest
        arrival) — on every engine that can place its record.
        ``frequencies`` sets the synthesis grid outright — its spacing
        ``Δf`` makes the record ``1/Δf`` long — which is the remedy the
        auto-grid warning names when the default grid, sized by the pulse,
        is too short for the channel.
        ``output_duration`` (seconds) sets the synthesised time window for
        the broadband synthesizers (Bellhop / RAM / Scooter / Kraken /
        OASP) — e.g. the length of an animation. SPARC marches
        ``source_waveform`` when its ``pulse_type`` is unpinned (its own
        pulse otherwise) and takes ``output_duration`` as ``time_max`` when
        ``time_max`` is unpinned; it cannot place ``t_start``. Every other
        TIME_SERIES model requires the waveform/rate.

        ``sample_rate`` is the rate of ``source_waveform`` (Hz), not a
        request for the output rate. The returned field's own
        ``sample_rate`` (``1/dt``) is the rate of its synthesis: Bellhop
        returns the input rate, while Kraken, Scooter and RAM return
        ``nfft·Δf`` of their frequency grid. Measured on a 100 m Pekeris
        channel with a 0.2 s pulse at 2000 Hz: Bellhop 2000 Hz, and
        Kraken / Scooter / RAM 2560 Hz (2048 samples at Δf = 1.25 Hz).
        Resample the trace if a fixed output rate matters.

        Parameters
        ----------
        env : Environment
            The environment.
        source : Source
            The source.
        receiver : Receiver
            The receiver grid.
        source_waveform : ndarray, optional
            Source time series, sampled at ``sample_rate``.
        sample_rate : float, optional
            Sample rate (Hz) of ``source_waveform``.
        output_duration : float, optional
            Record length (s).
        frequencies : array_like, optional
            The synthesis frequency grid (Hz), in place of the derived one.
        t_start : float, optional
            Time (s after emission) of the first sample; ``None`` opens the
            record just before the earliest arrival.

        Returns
        -------
        result : Result or ResultStack
            The time-domain ``Field``, or a ``ResultStack`` of them over
            ``source_depth`` for a multi-depth ``Source`` (one run per
            depth); ``stack.superpose()`` sums the traces with
            ``source.weights``.
        """
        self._require_mode(RunMode.TIME_SERIES, "time-series computation")
        return self.run(
            env, source, receiver,
            run_mode=RunMode.TIME_SERIES,
            source_waveform=source_waveform,
            sample_rate=sample_rate,
            output_duration=output_duration,
            frequencies=frequencies,
            t_start=t_start,
        )

    def compute_transfer_function(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        *,
        frequencies: Optional[np.ndarray] = None,
    ) -> Union[Result, ResultStack]:
        """Compute broadband complex transfer function H(f).

        Dispatches to ``run(run_mode=RunMode.BROADBAND)``. Pass
        ``frequencies=`` to override ``source.frequencies`` for the
        sweep.

        Parameters
        ----------
        env : Environment
            The environment.
        source : Source
            The source.
        receiver : Receiver
            The receiver grid.
        frequencies : array_like, optional
            The sweep (Hz), in place of ``source.frequencies``.

        Returns
        -------
        result : Result or ResultStack
            The ``H(f)`` ``Field``, or a ``ResultStack`` of them over
            ``source_depth`` for a multi-depth ``Source`` (one run per
            depth); ``stack.superpose()`` adds them with
            ``source.weights``.
        """
        self._require_mode(RunMode.BROADBAND, "broadband transfer-function computation")
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

        Dispatches to ``run(run_mode=RunMode.COVARIANCE)``. Two models
        declare this mode: OASN returns the noise-field covariance, OASS the
        reverberant-field covariance.

        Parameters
        ----------
        env : Environment
            The environment.
        source : Source
            The source.
        receiver : Receiver
            The receiver grid.
        """
        self._require_mode(RunMode.COVARIANCE, "covariance-matrix computation")
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

        Parameters
        ----------
        env : Environment
            The environment.
        source : Source
            The source.
        receiver : Receiver
            The receiver grid.
        """
        self._require_mode(RunMode.REPLICA, "replica-field computation")
        return self.run(env, source, receiver, run_mode=RunMode.REPLICA)

    def compute_reverberation(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
    ) -> Result:
        """Compute the scattered (reverberant) field from rough interfaces.

        Dispatches to ``run(run_mode=RunMode.REVERBERATION)``; OASS is the
        model declaring this mode. The environment must carry the rough
        interface the scattering kernel acts on.

        Parameters
        ----------
        env : Environment
            The environment.
        source : Source
            The source.
        receiver : Receiver
            The receiver grid.
        """
        self._require_mode(RunMode.REVERBERATION, "reverberation computation")
        return self.run(env, source, receiver,
                        run_mode=RunMode.REVERBERATION)

    def _resolve_executable(self, executable, find, *,
                            label: Optional[str] = None) -> Path:
        """Store the user's ``executable`` arg verbatim and return the binary
        as an absolute path.

        Keeps ``self.executable`` exactly as passed (``None`` when
        auto-detected) so ``copy()`` / ``__repr__`` — which read every
        constructor knob off ``self.<param>`` — round-trip the *intent*: a
        clone re-resolves the binary instead of re-pinning an
        already-resolved absolute path.

        Parameters
        ----------
        executable : str or Path, optional
            The user's constructor argument, verbatim.
        find : callable
            Zero-argument callable that locates the binary when
            ``executable`` is ``None`` (typically a
            :meth:`_find_executable_in_paths` closure).
        label : str, optional
            Model label for :class:`ExecutableNotFoundError`. Defaults to
            ``self.model_name``.
        """
        self.executable = Path(executable) if executable is not None else None
        # Every launch runs with ``cwd=`` a scratch directory, and POSIX
        # resolves a relative program path against the child's cwd, so the
        # returned launch path is absolute.
        exe = (self.executable.expanduser().resolve()
               if self.executable is not None else find())
        if not _is_runnable(exe):
            raise ExecutableNotFoundError(
                label or self.model_name, str(exe),
                reason=(None if not exe.exists() else
                        'not an executable file' if not exe.is_file() else
                        'the execute permission is not set'),
            )
        return exe

    def _find_executable_in_paths(self, names, bin_subdirs=None,
                                  dev_subdir: Optional[str] = None) -> Path:
        """:func:`~uacpy.models._launch.find_executable_in_paths` for this
        model: the first runnable ``names`` under ``uacpy/bin/<bin_subdirs>``,
        the development tree ``third_party/<dev_subdir>``, then ``PATH``."""
        return find_executable_in_paths(names, bin_subdirs, dev_subdir,
                                        model_name=self.model_name)

    def _run_subprocess(
        self,
        cmd,
        cwd,
        timeout: Optional[float] = None,
        env: Optional[dict] = None,
    ):
        """Run a binary through :func:`~uacpy.models._launch.run_subprocess`
        with this model's name, verbosity and ``timeout`` (unless one is
        given). Every launch of every engine passes here, the one place a
        run can be stopped before its binary starts."""
        return run_subprocess(
            self.model_name, cmd, cwd,
            timeout=self.timeout if timeout is None else timeout,
            verbose=self.verbose, env=env)

    def _project_environment(self, env: 'Environment', *,
                             request=None) -> 'Environment':
        """Return :meth:`_environment_projection`'s copy of ``env``, with
        every unsupported feature collapsed, after one ``FallbackWarning`` per
        dropped feature, in the order the features are collapsed.
        ``request`` is the checked call (Bellhop reads its
        ``engine_request``), ``None`` when the hook is called on its own.

        A subclass that overrides this must call ``super()`` — Kraken does —
        rather than reimplement the sequence.
        """
        projected, notices = self._environment_projection(env)
        for message in notices:
            warnings.warn(message, FallbackWarning,
                          skip_file_prefixes=USER_FRAME_SKIP)
        return projected

    def _environment_projection(self, env: 'Environment'):
        """``(projected, notices)``: :func:`~uacpy.models._projection.
        environment_projection` of ``env`` for this model — its capabilities
        (:attr:`supported_features`) and its resolved collapse policy. Pure:
        warns nothing, so a nested model's projection can be computed again
        without repeating what was said."""
        return environment_projection(
            env, model_name=self.model_name,
            supported=frozenset(self.supported_features),
            collapse=self._collapse)

    def _attach_output_paths(
        self,
        result: 'Result',
        work_dir: Path,
        base_name: str,
        *,
        primary_files: tuple = (),
    ) -> None:
        """:func:`~uacpy.models._extract.attach_output_paths` under this
        model's ``cleanup``."""
        attach_output_paths(result, work_dir, base_name, cleanup=self.cleanup,
                            primary_files=primary_files)

    def _require_output(self, candidates, *, what: str, process=None,
                        hint: str = '', prt_base: Optional[str] = None,
                        work_dir=None) -> Path:
        """:func:`~uacpy.models._launch.require_output` for this model."""
        return require_output(self.model_name, candidates, what=what,
                              process=process, hint=hint, prt_base=prt_base,
                              work_dir=work_dir)

    @staticmethod
    def _attach_prt_tail(exc, work_dir, base_name, tail_bytes: int = 2000):
        """:func:`~uacpy.models._launch.attach_prt_tail`."""
        attach_prt_tail(exc, work_dir, base_name, tail_bytes)

    def _raise_on_fortran_fatal(self, result, work_dir, base_name):
        """:func:`~uacpy.models._launch.raise_on_fortran_fatal` for this
        model, with the ERROUT messages its traits call benign
        (``spec.traits.benign_fortran_fatals``)."""
        raise_on_fortran_fatal(self.model_name, result, work_dir, base_name,
                               benign=self._traits.benign_fortran_fatals)

    def _launch_binary(self, launch: Launch):
        """:func:`~uacpy.models._launch.run_launch` for this model: through
        :meth:`_run_subprocess`, the child's stdout logged when the
        ``debug`` level prints."""
        return run_launch(
            launch, run=self._run_subprocess,
            log=self._log if _log_enabled(self.verbose, 'debug') else None)

    def _prt_launch(self, cmd, work_dir, base_name, *,
                    timeout: Optional[float] = None,
                    env: Optional[dict] = None,
                    stale_outputs: tuple = (),
                    report_prt_warnings: bool = True) -> Launch:
        """The :class:`~uacpy.models._launch.Launch` of a binary that writes
        ``<base_name>.prt`` (the Acoustics Toolbox, OASES): its outputs
        ``<base_name><suffix>`` for each of ``stale_outputs`` and its
        ``.prt`` cleared first, the ``.prt`` tail attached to a failure, then
        :meth:`_raise_on_fortran_fatal` and, with ``report_prt_warnings``,
        :meth:`_warn_on_prt_warnings`. See :meth:`_run_and_attach_prt`."""
        checks = [lambda result: self._raise_on_fortran_fatal(
            result, work_dir, base_name)]
        if report_prt_warnings:
            checks.append(lambda result: self._warn_on_prt_warnings(
                work_dir, base_name))
        return Launch(
            argv=tuple(cmd), cwd=work_dir, env=env, timeout=timeout,
            stale_outputs=tuple(f"{base_name}{suffix}"
                                for suffix in stale_outputs),
            prt_root=base_name, checks=tuple(checks),
            stdout_label=self.model_name)

    def _run_and_attach_prt(self, cmd, work_dir, base_name, *,
                            timeout: Optional[float] = None,
                            env: Optional[dict] = None,
                            stale_outputs: tuple = (),
                            report_prt_warnings: bool = True):
        """Run a Fortran/AT binary as :meth:`_prt_launch` describes it
        (:meth:`_launch_binary`), appending the ``<base>.prt`` tail to a
        :class:`ModelExecutionError` on failure and logging stdout when
        verbose. Returns the ``CompletedProcess``. Shared by every model's
        binary-launch wrapper.

        The AT engines hold the file root in ``CHARACTER(LEN=80)`` buffers,
        so a root longer than 80 characters is silently truncated; launching
        with ``work_dir`` as cwd and a short relative ``base_name`` keeps
        every root far inside that limit.

        ``stale_outputs`` lists the suffixes (``'.shd'``, ``'.arr'``, …) this
        binary may write under ``base_name``. They are removed before launch
        so a pinned work dir cannot hand an earlier run's output back as this
        run's answer — the same reason the ``.prt`` is cleared.

        ``report_prt_warnings=False`` leaves the binary's non-fatal ``.prt``
        warnings unsaid (:meth:`_warn_on_prt_warnings`), for a launch whose
        ``.prt`` describes a side computation rather than the run (Kraken's
        mode-count check solve).
        """
        return self._launch_binary(self._prt_launch(
            cmd, work_dir, base_name, timeout=timeout, env=env,
            stale_outputs=stale_outputs,
            report_prt_warnings=report_prt_warnings))

    def _warn_on_prt_warnings(self, work_dir, base_name) -> None:
        """:func:`~uacpy.models._launch.warn_on_prt_warnings` for this
        model."""
        warn_on_prt_warnings(self.model_name, work_dir, base_name)

    def _result_kwargs(
        self,
        source: 'Source',
        *,
        backend: Optional[str] = None,
        frequencies: Optional[Union[float, np.ndarray]] = None,
        phase_reference: Optional[str] = None,
        **extra,
    ) -> dict:
        """:func:`~uacpy.models._extract.result_kwargs` for this model: its
        name and provenance identify the result."""
        return result_kwargs(self.model_name, self.provenance, source,
                             backend=backend, frequencies=frequencies,
                             phase_reference=phase_reference, **extra)

    def _stamp_result(self, result, source: 'Source', *,
                      backend: Optional[str] = None,
                      frequencies: Optional[Union[float, np.ndarray]] = None,
                      phase_reference: Optional[str] = None):
        """:func:`~uacpy.models._extract.stamp_result` for this model."""
        return stamp_result(self.model_name, self.provenance, result, source,
                            backend=backend, frequencies=frequencies,
                            phase_reference=phase_reference)

    @staticmethod
    def _speed_bounds(env: 'Environment'):
        """:func:`~uacpy.models._checks.speed_bounds` of ``env``."""
        return speed_bounds(env)

    def _max_receiver_depth(self, env: 'Environment') -> float:
        """Deepest receiver depth this model can resolve the field at: the
        seafloor, or the deepest modelled interface when
        ``spec.traits.receivers_reach_sediment`` (see :meth:`_total_media_depth`).

        The value gates source and receiver depths asymmetrically in
        :meth:`_validate_geometry`: a source below it raises, a receiver
        below it only warns. Changing it therefore changes which source
        placements are legal, not just which receivers warn.
        """
        if self._traits.receivers_reach_sediment:
            return self._total_media_depth(env)
        return float(env.depth)

    def _source_below_domain_note(self, env: 'Environment',
                                  resolvable_depth: float):
        """Extra paragraph for the source-below-``_max_receiver_depth``
        error, or ``None``.

        The generic message states the depth and the limit; a model whose
        limit sits above what its engine can compute overrides this to say
        which of the two it is, so the user is not left reading a numerical
        limit as a physical one.
        """
        return None

    def _mask_unresolvable_depths(self, result, receiver, media_depth):
        """:func:`~uacpy.models._extract.mask_unresolvable_depths` for this
        model."""
        return mask_unresolvable_depths(self.model_name, result, receiver,
                                        media_depth)

    def _require_mode(self, mode: RunMode, what: str) -> None:
        """Refuse ``what``, a ``compute_*`` wrapper's task, on a model that
        does not run ``mode``, naming the engines that do."""
        if not self.supports_mode(mode):
            raise UnsupportedFeatureError(
                self.model_name, what, alternatives=engines_running(mode))

    def _mask_source_axis(self, field, source, *, warn: bool = True):
        """:func:`~uacpy.models._extract.mask_source_axis` for this model."""
        return mask_source_axis(self.model_name, field, source, warn=warn)

    def _reject_malformed_irc_bottom(self, env: 'Environment') -> None:
        """:func:`~uacpy.models._checks.reject_malformed_irc_bottom` for this
        model."""
        reject_malformed_irc_bottom(self.model_name, env)

    def _total_media_depth(self, env: 'Environment') -> float:
        """:func:`~uacpy.models._checks.total_media_depth` of ``env``."""
        return total_media_depth(env)

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
