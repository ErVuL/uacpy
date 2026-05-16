"""Base class for acoustic propagation models."""

import copy as _copy
import os
import shutil
import signal
import subprocess
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

import uacpy._stack  # noqa: F401 — side-effect: raise RLIMIT_STACK
from uacpy.core.environment import Environment
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

    MODES = 'modes'                      # Normal modes (Kraken/KrakenC depth eigenfunctions)

    # OASN frequency-domain array products: COVARIANCE → C(f, i, j) hydrophone
    # × hydrophone matrix; REPLICA → Green's-function samples at the array
    # elements per candidate source position. See core/results.Covariance and
    # core/results.Replicas.
    COVARIANCE = 'covariance'
    REPLICA = 'replica'

    # Time-domain pressure p(t) at the receiver(s). Models that compute a
    # broadband transfer function natively (Bellhop, RAM, Scooter,
    # KrakenField, OASES) require ``source_waveform=`` + ``sample_rate=``;
    # SPARC computes p(t) directly from its source pulse and ignores them.
    TIME_SERIES = 'time_series'

    # Broadband complex transfer function H(f).
    BROADBAND = 'broadband'

    REFLECTION = 'reflection'            # Plane-wave reflection coefficients (Bounce, OASR)


DEFAULT_COLLAPSE: Dict[str, str] = {
    'bathymetry': 'max',
    'ssp': 'r0',
    'bottom': 'r0',
    'layered': 'halfspace',
    'rd_layered_range': 'median',
    'rd_layered_layers': 'halfspace',
    'altimetry': 'drop',
    'elastic': 'fluid',
}

# Allowed method strings per collapse key (validated at construction in
# ``PropagationModel.__init__`` so bad values fail loudly rather than deep
# inside a writer at ``run()``-time).
VALID_COLLAPSE_METHODS: Dict[str, frozenset] = {
    'bathymetry':        frozenset({'max', 'median', 'mean', 'min', 'initial'}),
    'ssp':               frozenset({'r0', 'rmax', 'mean', 'median'}),
    'bottom':            frozenset({'r0', 'rmax', 'mean', 'median'}),
    'layered':           frozenset({'halfspace', 'top_layer', 'volume_average'}),
    'rd_layered_range':  frozenset({'r0', 'rmax', 'median'}),
    'rd_layered_layers': frozenset({'preserve', 'halfspace', 'top_layer', 'volume_average'}),
    'altimetry':         frozenset({'drop'}),
    'elastic':           frozenset({'fluid', 'vacuum'}),
}
assert set(VALID_COLLAPSE_METHODS) == set(DEFAULT_COLLAPSE), (
    "VALID_COLLAPSE_METHODS keys must match DEFAULT_COLLAPSE keys"
)
assert all(DEFAULT_COLLAPSE[k] in VALID_COLLAPSE_METHODS[k] for k in DEFAULT_COLLAPSE), (
    "DEFAULT_COLLAPSE values must satisfy VALID_COLLAPSE_METHODS"
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
        # cleanup defaults to True only when uacpy owns the work dir.
        self.cleanup = (work_dir is None) if cleanup is None else bool(cleanup)
        self.timeout = float(timeout)
        # Per-feature collapse policies applied by ``_project_environment``
        # when an env contains a feature this model doesn't support. Pass
        # ``collapse={'bathymetry': 'min', 'ssp': 'mean', ...}`` to override
        # any subset; missing keys keep the defaults.
        #
        # 'bathymetry'        : 'max'|'median'|'mean'|'min'|'initial'
        # 'ssp'               : 'r0'|'rmax'|'mean'|'median'
        # 'bottom'            : 'r0'|'rmax'|'mean'|'median'
        # 'layered'           : 'halfspace'|'top_layer'|'volume_average'
        # 'rd_layered_range'  : 'r0'|'rmax'|'median' — which range to
        #                       sample for an RDLB env
        # 'rd_layered_layers' : 'preserve'|'halfspace'|'top_layer'|
        #                       'volume_average' — how to flatten the
        #                       layer stack at that range. 'preserve'
        #                       keeps the LayeredBottom and requires
        #                       the model to support layered bottoms.
        # 'altimetry'         : 'drop'
        # 'elastic'           : 'fluid' (zero shear) | 'vacuum'
        self._collapse: Dict[str, str] = dict(DEFAULT_COLLAPSE)
        self._user_collapse: Dict[str, str] = {}
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
        self._supports_range_dependent_bathymetry: bool = False
        self._supports_range_dependent_ssp: bool = False
        self._supports_range_dependent_bottom: bool = False
        self._supports_layered_bottom: bool = False
        self._supports_range_dependent_layered_bottom: bool = False
        self._supports_elastic_media: bool = False
        # Bellhop is the only model that runs one source-depth grid in
        # a single binary call; everyone else loops in Python.
        self._supports_multi_source_depth: bool = False

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
        smarter rule (e.g. KrakenField picks BROADBAND when a frequency
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

        Implementation: walks ``inspect.signature(type(self).__init__)``,
        pulls each parameter's current value off the instance (uacpy
        models store every constructor arg as ``self.<name>``), merges
        ``overrides``, and instantiates. ``**kwargs``-only sinks on the
        constructor are ignored.

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
        import inspect

        sig = inspect.signature(type(self).__init__)
        kwargs: Dict[str, object] = {}
        valid_names = set()
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                continue
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
        return type(self)(**kwargs)

    @abstractmethod
    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional['RunMode'] = None,
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
        ``source_waveform``, ``sample_rate``. Every TIME_SERIES-capable
        wrapper (Bellhop, Scooter, KrakenField, OASP, RAM) consumes
        ``source_waveform`` and ``sample_rate``; SPARC warns that they
        are ignored (it uses its constructor ``pulse_type``). Models
        with a broadband path consume ``frequencies`` as an explicit
        override for ``source.frequencies``. KrakenField additionally
        takes ``n_modes`` for the field reconstruction limit. No
        other kwargs are accepted — passing one raises
        :class:`TypeError`.

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

        Returns
        -------
        result : Result
            One of the typed :mod:`uacpy.core.results` subclasses
            (``Field``, ``Field``, ``Modes``, …) determined
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
        transfer function (Bellhop, RAM, Scooter, KrakenField, OASP).
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

    def _setup_file_manager(self) -> FileManager:
        """Build the FileManager. ``self.work_dir`` is used as-is (not a
        parent); when ``None``, a fresh temp dir is created.

        Auto-creates the user-pinned ``work_dir`` if it doesn't exist
        yet, so callers can construct ``Model(work_dir='./out')`` without
        a separate ``mkdir`` step.
        """
        if self.work_dir is not None:
            Path(self.work_dir).mkdir(parents=True, exist_ok=True)
            fm = FileManager(
                use_tmpfs=False,
                base_dir=self.work_dir,
                prefix=f'{self.model_name.lower()}_',
                cleanup=getattr(self, 'cleanup', False),
            )
            fm.work_dir = Path(self.work_dir)
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
            If source/receiver depths exceed the environment's maximum depth.
        ValueError
            If source/receiver depths are negative.
        ConfigurationError
            If ``run_mode`` is a single-frequency mode and ``source`` carries
            multiple frequencies.
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
        max_depth = env.depth

        if np.any(source.depths > max_depth):
            raise InvalidDepthError(
                float(source.depths.max()), max_depth, "Source",
            )

        if receiver.depth_max > max_depth:
            raise InvalidDepthError(
                float(receiver.depth_max), max_depth, "Receiver",
            )

        if np.any(source.depths < 0):
            raise ConfigurationError("Source depths must be positive")

        if receiver.depth_min < 0:
            raise ConfigurationError("Receiver depths must be positive")

        self._check_per_range_receiver_depth(env, receiver)
        self._warn_on_range_coverage(env, receiver)

    def _check_per_range_receiver_depth(
        self, env: 'Environment', receiver: 'Receiver',
    ) -> None:
        """Emit a ``UserWarning`` if any receiver sits below the local
        seafloor in a range-dependent bathymetry. The flat-bathy case is
        already a hard ``InvalidDepthError`` via the global
        ``receiver.depth_max <= env.depth`` check above. We only warn
        here because several models (Bellhop, RAM) accept below-seafloor
        receivers natively (infinite TL, PE absorbing region).

        ``receiver_type='line'`` pairs depths[i] with ranges[i] (1-D
        compare); ``'grid'`` evaluates the depth × range cross-product.
        """
        if not env.has_range_dependent_bathymetry():
            return
        depths = np.atleast_1d(receiver.depths).astype(float)
        ranges = np.atleast_1d(receiver.ranges).astype(float)
        seafloor = np.asarray(env.bathymetry_at_range(ranges), dtype=float)

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
        from uacpy.core.environment import (
            RangeDependentBottom, RangeDependentLayeredBottom,
        )

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

        if env.has_range_dependent_bathymetry():
            _check("env.bathymetry", float(env.bathymetry[-1, 0]))
        if env.ssp.is_range_dependent:
            _check("env.ssp.ranges", float(env.ssp.ranges[-1]))
        if isinstance(env.bottom, (RangeDependentBottom,
                                   RangeDependentLayeredBottom)):
            _check("env.bottom.ranges", float(env.bottom.ranges[-1]))
        if env.altimetry is not None and len(env.altimetry) > 1:
            _check("env.altimetry ranges", float(env.altimetry[-1, 0]))

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
                alternatives=['Bellhop', 'KrakenField', 'RAM', 'Scooter', 'OAST'],
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
            Ocean environment. Must be range-independent for most models;
            only ``KrakenField`` handles range-dependent mode computation.
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
                alternatives=['Kraken', 'KrakenC', 'KrakenField']
            )

        if env.is_range_dependent and self.model_name != 'KrakenField':
            # Range-independent mode solvers (Kraken, KrakenC) collapse
            # the environment via ``collapse={'bathymetry': …}`` and warn,
            # rather than reject — same pattern as OAST/OASP/Scooter/SPARC.
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
    ) -> Result:
        """Compute time-domain pressure p(t) at the receiver(s).

        Forwards ``source_waveform`` and ``sample_rate`` to
        ``run(run_mode=RunMode.TIME_SERIES)``. SPARC ignores both (it
        builds p(t) from its native ``pulse_type``); every other
        TIME_SERIES model requires them.
        """
        if not self.supports_mode(RunMode.TIME_SERIES):
            raise UnsupportedFeatureError(
                self.model_name,
                "time-series computation",
                alternatives=['Bellhop', 'KrakenField', 'RAM', 'Scooter',
                              'OASP', 'SPARC'],
            )
        return self.run(
            env, source, receiver,
            run_mode=RunMode.TIME_SERIES,
            source_waveform=source_waveform,
            sample_rate=sample_rate,
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
                alternatives=['Bellhop', 'KrakenField', 'RAM',
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
        """True if ``boundary`` carries any non-zero shear speed."""
        from uacpy.core.environment import _boundary_has_shear
        return _boundary_has_shear(boundary)

    @staticmethod
    def _collapse_elastic_boundary(boundary, method: str):
        """Collapse elastic shear properties on ``boundary`` per ``method``.

        ``'fluid'``  : zero shear_speed and shear_attenuation; keep cp / ρ / α.
        ``'vacuum'`` : replace with default vacuum BoundaryProperties.

        Walks layer/profile structure for :class:`LayeredBottom` and
        :class:`RangeDependentLayeredBottom`.
        """
        from uacpy.core.environment import (
            BoundaryProperties, LayeredBottom, RangeDependentLayeredBottom,
        )

        def _zero_shear(b):
            if hasattr(b, 'shear_speed'):
                cs = b.shear_speed
                b.shear_speed = (
                    np.zeros_like(cs) if isinstance(cs, np.ndarray) else 0.0
                )
            if hasattr(b, 'shear_attenuation'):
                cas = b.shear_attenuation
                b.shear_attenuation = (
                    np.zeros_like(cas) if isinstance(cas, np.ndarray) else 0.0
                )

        if method == 'vacuum':
            return BoundaryProperties(acoustic_type='vacuum')
        if method != 'fluid':
            raise ConfigurationError(
                f"Unknown elastic collapse method {method!r}. Use "
                "'fluid' or 'vacuum'."
            )
        b = _copy.deepcopy(boundary)
        if isinstance(b, RangeDependentLayeredBottom):
            for prof in b.profiles:
                for layer in prof.layers:
                    _zero_shear(layer)
                _zero_shear(prof.halfspace)
        elif isinstance(b, LayeredBottom):
            for layer in b.layers:
                _zero_shear(layer)
            _zero_shear(b.halfspace)
        else:
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

        if e.has_range_dependent_bathymetry() and not self._supports_range_dependent_bathymetry:
            method = self._collapse["bathymetry"]
            new_depth = e.get_representative_depth(method)
            min_d = float(e.bathymetry[:, 1].min())
            max_d = float(e.bathymetry[:, 1].max())
            e.bathymetry = np.array([[0.0, new_depth]], dtype=np.float64)
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

        if e.has_range_dependent_ssp() and not self._supports_range_dependent_ssp:
            method = self._collapse["ssp"]
            e.ssp = e.ssp.collapse(method)
            warnings.warn(
                f"{self.model_name} does not support range-dependent SSP; "
                f"collapsed to 1-D (collapse['ssp']={method!r}).",
                UserWarning, stacklevel=3,
            )

        if e.has_range_dependent_bottom() and not self._supports_range_dependent_bottom:
            method = self._collapse["bottom"]
            e.bottom = e.bottom.collapse(method)
            warnings.warn(
                f"{self.model_name} does not support range-dependent bottom "
                f"geoacoustics; collapsed to single profile "
                f"(collapse['bottom']={method!r}).",
                UserWarning, stacklevel=3,
            )

        if e.has_range_dependent_layered_bottom() and not self._supports_range_dependent_layered_bottom:
            range_step = self._collapse["rd_layered_range"]
            layers_method = self._collapse["rd_layered_layers"]
            range_methods = ('r0', 'rmax', 'median')
            layers_methods = ('preserve', 'halfspace', 'top_layer', 'volume_average')
            if range_step not in range_methods:
                raise ConfigurationError(
                    f"Unknown collapse['rd_layered_range']={range_step!r}. "
                    f"Valid: {range_methods}."
                )
            if layers_method not in layers_methods:
                raise ConfigurationError(
                    f"Unknown collapse['rd_layered_layers']={layers_method!r}. "
                    f"Valid: {layers_methods}."
                )
            if layers_method == 'preserve':
                if not self._supports_layered_bottom:
                    raise ConfigurationError(
                        f"{self.model_name} does not support layered bottoms, "
                        f"so collapse['rd_layered_layers']='preserve' would "
                        f"leave a LayeredBottom the model cannot consume. "
                        f"Pick one of "
                        f"{layers_methods[1:]!r} to also flatten the layers."
                    )
                e.bottom = e.bottom.to_profile(range_step)
                warnings.warn(
                    f"{self.model_name} does not support range-dependent "
                    f"layered bottoms; selected the {range_step!r} layered "
                    f"profile (collapse['rd_layered_range']={range_step!r}, "
                    f"collapse['rd_layered_layers']='preserve').",
                    UserWarning, stacklevel=3,
                )
            else:
                e.bottom = e.bottom.to_profile(range_step).collapse(layers_method)
                warnings.warn(
                    f"{self.model_name} does not support range-dependent "
                    f"layered bottoms; collapsed to single boundary "
                    f"(collapse['rd_layered_range']={range_step!r}, "
                    f"collapse['rd_layered_layers']={layers_method!r}).",
                    UserWarning, stacklevel=3,
                )

        if e.has_layered_bottom() and not self._supports_layered_bottom:
            method = self._collapse["layered"]
            e.bottom = e.bottom.collapse(method)
            warnings.warn(
                f"{self.model_name} does not support layered (depth-"
                f"dependent) bottoms; collapsed to single boundary "
                f"(collapse['layered']={method!r}).",
                UserWarning, stacklevel=3,
            )

        if not self._supports_elastic_media:
            collapsed_at = []
            if e.surface is not None and self._has_shear(e.surface):
                e.surface = self._collapse_elastic_boundary(
                    e.surface, self._collapse["elastic"],
                )
                collapsed_at.append('surface')
            if e.bottom is not None and self._has_shear(e.bottom):
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

        Acoustics-Toolbox binaries dump fatal errors to ``.prt`` instead
        of stderr; this surfaces them in the raised ``ModelExecutionError``
        message.
        """
        from pathlib import Path as _Path
        prt = _Path(work_dir) / f"{base_name}.prt"
        if not prt.exists():
            return
        try:
            size = prt.stat().st_size
            with prt.open('rb') as fh:
                if size > n_chars:
                    fh.seek(size - n_chars)
                tail = fh.read().decode('utf-8', errors='replace')
        except OSError:
            return
        exc.args = (
            f"{exc.args[0] if exc.args else exc}\n\n.prt tail:\n{tail}",
        ) + exc.args[1:]

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
            metadata=dict(extra),
        )
        if phase_reference is not None:
            kw['phase_reference'] = phase_reference
        return kw

    def _clip_receiver_depths(
        self, receiver: 'Receiver', env_depth: float, margin: float = 3.0
    ) -> 'Receiver':
        """
        Clip receiver depths to stay within the environment, with a safety margin.

        Parameters
        ----------
        receiver : Receiver
            Input receiver array
        env_depth : float
            Maximum environment depth (m)
        margin : float
            Safety margin below the seafloor (m). Default 3.0.

        Returns
        -------
        Receiver
            Receiver with clipped depths (unchanged if all depths are valid)
        """
        max_receiver_depth = receiver.depths.max()
        if max_receiver_depth > env_depth - margin:
            clipped = np.clip(
                receiver.depths, receiver.depths.min(), env_depth - margin,
            )
            unique = np.unique(clipped)
            if receiver.receiver_type == 'grid':
                new_depths = unique
            else:
                new_depths = clipped
            receiver = Receiver(
                depths=new_depths,
                ranges=receiver.ranges,
                receiver_type=receiver.receiver_type,
            )
            if self.verbose:
                self._log(
                    f"Clipped receiver depths to {env_depth - margin:.1f}m "
                    f"(environment depth: {env_depth:.1f}m)"
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
            value = getattr(self, name)
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
