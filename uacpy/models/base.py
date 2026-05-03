"""Base class for acoustic propagation models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List
from enum import Enum
import warnings

import numpy as np

from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.results import Result
from uacpy.io.file_manager import FileManager


# Sentinel for "not provided" in run() overrides (distinct from None which means auto-compute)
_UNSET = object()


def _resolve_overrides(obj, **overrides):
    """Resolve per-call overrides as a context manager.

    For each override: if the supplied value is _UNSET, the attribute on
    ``obj`` is left untouched; otherwise it is temporarily overwritten for
    the duration of the ``with`` block so internal methods that read
    ``self.<param>`` see the overridden value.  On exit the original
    attribute values are restored unconditionally.

    Returns
    -------
    context manager
        A context manager that installs the overrides on entry and
        restores the originals on exit.
    """
    from contextlib import contextmanager

    @contextmanager
    def override_context():
        originals = {}
        for name, value in overrides.items():
            originals[name] = getattr(obj, name)
            if value is not _UNSET:
                setattr(obj, name, value)
        try:
            yield
        finally:
            for name, orig_value in originals.items():
                setattr(obj, name, orig_value)

    return override_context()


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

    MODES = 'modes'                      # Normal modes only

    # Time-domain pressure p(t) at the receiver(s). Models that compute a
    # broadband transfer function natively (Bellhop, RAM, Scooter,
    # KrakenField, OASES) require ``source_waveform=`` + ``sample_rate=``;
    # SPARC computes p(t) directly from its source pulse and ignores them.
    TIME_SERIES = 'time_series'

    # Broadband complex transfer function H(f).
    BROADBAND = 'broadband'

    REFLECTION = 'reflection'            # Plane-wave reflection coefficients (Bounce, OASR)


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
    verbose : bool, optional
        Print verbose output. Default is False.
    work_dir : str or Path, optional
        Working directory for files. If ``None``, a temporary directory is
        created per run.

    Attributes
    ----------
    model_name : str
        Name of the model (class name).
    use_tmpfs : bool
        Whether tmpfs is used.
    verbose : bool
        Verbose output flag.
    file_manager : FileManager
        File manager instance (populated during ``run``).
    """

    def __init__(
        self,
        use_tmpfs: bool = False,
        verbose: bool = False,
        work_dir: Optional[Path] = None,
    ):
        self.model_name = self.__class__.__name__
        self.use_tmpfs = use_tmpfs
        self.verbose = verbose
        self.work_dir = work_dir
        self.file_manager = None

        # Subclasses override to declare the run modes they support.
        self._supported_modes: List[RunMode] = [RunMode.COHERENT_TL]

        # Capability flags — subclasses set True for each Environment feature
        # they honour natively. Anything left False that's present in env on
        # ``run()`` triggers a warning via ``_warn_on_unsupported_env_features``
        # rather than silently dropping the data.
        #
        # Default model-suggestion strings appended to each warning. Override
        # ``_unsupported_env_alternatives`` per subclass to point users at
        # the right alternative (e.g., RAM for range-dependent fluid PE).
        self._supports_altimetry: bool = False
        self._supports_range_dependent_ssp: bool = False
        self._supports_range_dependent_bottom: bool = False
        self._supports_layered_bottom: bool = False
        self._supports_range_dependent_layered_bottom: bool = False
        self._unsupported_env_alternatives: str = (
            "Bellhop / KrakenField / RAM"
        )

    @property
    def supported_modes(self) -> List[RunMode]:
        """List of run modes supported by this model."""
        return self._supported_modes

    def supports_mode(self, mode: RunMode) -> bool:
        """Return True if the model supports ``mode``."""
        return mode in self._supported_modes

    @abstractmethod
    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional['RunMode'] = None,
        **kwargs
    ) -> Result:
        """Run the propagation model.

        Every wrapper takes the same first four parameters in the same
        order: the common ``Environment`` / ``Source`` / ``Receiver``
        triple, followed by the optional ``run_mode``.

        Model-specific settings come from two places:

        1. **Constructor kwargs** — the long-lived configuration of the
           model instance (e.g. ``RAM(dr=2.0, dz=0.5, np_pade=8)``).
        2. **Per-call overrides via ``**kwargs``** — any keyword whose
           name matches a constructor attribute is temporarily applied
           for the duration of this call. This lets you reuse one
           configured instance with localised tweaks.

        Mode-specific kwargs (e.g. ``source_waveform`` and ``sample_rate``
        for ``RunMode.TIME_SERIES``) are also accepted via ``**kwargs``;
        each model documents its own.

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
        **kwargs
            Per-call overrides (matching constructor attribute names) and
            mode-specific extras. See subclass docstrings.

        Returns
        -------
        result : Result
            One of the typed :mod:`uacpy.core.results` subclasses
            (``TLField``, ``TransferFunction``, ``Modes``, …) determined
            by ``run_mode`` and the model.
        """
        pass

    def _setup_file_manager(self) -> FileManager:
        """
        Create and configure the FileManager for this run.

        Returns
        -------
        fm : FileManager
            Configured file manager (auto-cleanup when ``work_dir`` is None).
        """
        cleanup = self.work_dir is None

        fm = FileManager(
            use_tmpfs=self.use_tmpfs,
            base_dir=self.work_dir,
            prefix=f'{self.model_name.lower()}_',
            cleanup=cleanup
        )
        fm.create_work_dir()

        return fm

    def _log(self, message: str, level: str = "info"):
        """
        Forward ``message`` to the model logger at ``level``.

        Parameters
        ----------
        message : str
            Message to log.
        level : str, optional
            Log level: 'info', 'warn', 'error', or 'debug'. Default is
            'info'. Unknown levels fall back to info.
        """
        if not hasattr(self, '_logger'):
            from uacpy.core.logger import Logger
            self._logger = Logger(self.model_name, verbose=self.verbose)

        level = level.lower()
        if level == 'info':
            self._logger.info(message)
        elif level == 'warn' or level == 'warning':
            self._logger.warn(message)
        elif level == 'error':
            self._logger.error(message)
        elif level == 'debug':
            self._logger.debug(message)
        else:
            self._logger.info(message)

    def validate_inputs(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver
    ):
        """
        Validate inputs against the environment's maximum depth.

        Parameters
        ----------
        env : Environment
            Environment to validate against.
        source : Source
            Source to validate.
        receiver : Receiver
            Receiver to validate.

        Raises
        ------
        ValueError
            If source/receiver depths are negative or exceed the
            environment's maximum depth.
        """
        # For flat environments, env.depth may differ from env.bathymetry[0, 1]
        # (e.g. Environment(depth=100, bathymetry=[(0, 80)]) reports 100m but
        # the seafloor is at 80m). Use the deeper of the two to avoid false
        # depth-violation errors on the shallower side.
        if env.is_range_dependent and env.bathymetry is not None:
            max_depth = float(np.max(env.bathymetry[:, 1]))
        else:
            bathy_depth = float(env.bathymetry[0, 1]) if env.bathymetry is not None and len(env.bathymetry) > 0 else env.depth
            max_depth = max(env.depth, bathy_depth)

        if np.any(source.depth > max_depth):
            error_msg = f"Source depth ({source.depth.max():.1f}m) exceeds maximum environment depth ({max_depth:.1f}m)"
            self._log(error_msg, level='error')
            raise ValueError(error_msg)

        if receiver.depth_max > max_depth:
            error_msg = f"Receiver depth ({receiver.depth_max:.1f}m) exceeds maximum environment depth ({max_depth:.1f}m)"
            self._log(error_msg, level='error')
            raise ValueError(error_msg)

        if np.any(source.depth < 0):
            self._log("Source depths must be positive", level='error')
            raise ValueError("Source depths must be positive")

        if receiver.depth_min < 0:
            self._log("Receiver depths must be positive", level='error')
            raise ValueError("Receiver depths must be positive")

        # Warn on env features the model doesn't honour (covers altimetry,
        # range-dep SSP / bottom, layered, range-dep layered). Bellhop has
        # its own custom warnings inside ``run()`` and sets the relevant
        # supports flags so this helper doesn't double-warn.
        self._warn_on_unsupported_env_features(env)

    def compute_tl(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver = None,
        **kwargs
    ) -> Result:
        """
        Compute transmission loss (convenience wrapper around ``run``).

        Parameters
        ----------
        env : Environment
            Ocean environment.
        source : Source
            Acoustic source.
        receiver : Receiver, optional
            Receiver array. If ``None``, an automatic grid is built using
            ``ReceiverGridBuilder.build_tl_grid`` and ``max_range``
            (default 10 000 m) from ``kwargs``.
        **kwargs
            Additional model-specific parameters.

        Returns
        -------
        result : Result
            Transmission loss field.

        Examples
        --------
        >>> bellhop = Bellhop()
        >>> result = bellhop.compute_tl(env, source, receiver)

        >>> # Auto-generate receiver grid
        >>> result = bellhop.compute_tl(env, source, max_range=10000)
        """
        if receiver is None:
            from uacpy.core.model_utils import ReceiverGridBuilder
            max_range = kwargs.pop('max_range', 10000.0)
            depths, ranges = ReceiverGridBuilder.build_tl_grid(env.depth, max_range)
            receiver = Receiver(depths=depths, ranges=ranges)

        return self._compute_tl_impl(env, source, receiver, **kwargs)

    def _compute_tl_impl(self, env, source, receiver, **kwargs):
        """
        Model-specific TL computation implementation.

        Models can override this to provide optimized TL computation.
        Default implementation calls run() with RunMode.COHERENT_TL. Models
        that do not support COHERENT_TL (e.g., Bounce, OASR) must override
        _compute_tl_impl to raise UnsupportedFeatureError.
        """
        from uacpy.core.exceptions import UnsupportedFeatureError

        if not self.supports_mode(RunMode.COHERENT_TL):
            raise UnsupportedFeatureError(
                self.model_name,
                "transmission loss computation",
                alternatives=['Bellhop', 'KrakenField', 'RAM', 'Scooter', 'OAST'],
            )
        return self.run(env, source, receiver, run_mode=RunMode.COHERENT_TL, **kwargs)

    def compute_rays(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver = None,
        **kwargs
    ) -> Result:
        """
        Compute ray paths (convenience wrapper around ``run``).

        Parameters
        ----------
        env : Environment
            Ocean environment.
        source : Source
            Acoustic source.
        receiver : Receiver, optional
            Receiver array. If ``None``, an automatic grid is built.
        **kwargs
            Additional model-specific parameters.

        Returns
        -------
        result : Result
            Ray path data.

        Raises
        ------
        UnsupportedFeatureError
            If the model does not support ray computation.

        Examples
        --------
        >>> bellhop = Bellhop()
        >>> rays = bellhop.compute_rays(env, source)
        """
        from uacpy.core.exceptions import UnsupportedFeatureError

        if not self.supports_mode(RunMode.RAYS):
            raise UnsupportedFeatureError(
                self.model_name,
                "ray path computation",
                alternatives=['Bellhop']
            )

        if receiver is None:
            from uacpy.core.model_utils import ReceiverGridBuilder
            max_range = kwargs.pop('max_range', 10000.0)
            depths, ranges = ReceiverGridBuilder.build_ray_grid(env.depth, max_range)
            receiver = Receiver(depths=depths, ranges=ranges)

        return self.run(env, source, receiver, run_mode=RunMode.RAYS, **kwargs)

    def compute_arrivals(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        **kwargs
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
        **kwargs
            Additional model-specific parameters.

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
        from uacpy.core.exceptions import UnsupportedFeatureError

        if not self.supports_mode(RunMode.ARRIVALS):
            raise UnsupportedFeatureError(
                self.model_name,
                "arrival computation",
                alternatives=['Bellhop']
            )

        return self.run(env, source, receiver, run_mode=RunMode.ARRIVALS, **kwargs)

    def compute_modes(
        self,
        env: Environment,
        source: Source,
        n_modes: int = None,
        **kwargs
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
        **kwargs
            Additional model-specific parameters.

        Returns
        -------
        result : Result
            Mode data (``field_type='modes'``).

        Raises
        ------
        UnsupportedFeatureError
            If the model does not support mode computation.
        EnvironmentError
            If the environment is range-dependent and the model does not
            support it.

        Examples
        --------
        >>> kraken = Kraken()
        >>> modes = kraken.compute_modes(env, source, n_modes=50)
        >>> wavenumbers = modes.metadata['k']
        >>> mode_shapes = modes.metadata['phi']
        """
        from uacpy.core.exceptions import UnsupportedFeatureError, EnvironmentError

        if not self.supports_mode(RunMode.MODES):
            raise UnsupportedFeatureError(
                self.model_name,
                "normal mode computation",
                alternatives=['Kraken', 'OASN']
            )

        if env.is_range_dependent:
            # Only KrakenField handles range-dependent modes (coupled mode theory).
            if self.model_name not in ['KrakenField']:
                raise EnvironmentError(
                    f"{self.model_name} does not support range-dependent environments for mode computation. "
                    f"Environment has bathymetry ranging from {env.bathymetry[:, 1].min():.1f}m to "
                    f"{env.bathymetry[:, 1].max():.1f}m.",
                    "Use a range-independent environment or try KrakenField with adiabatic coupling"
                )

        return self._compute_modes_impl(env, source, n_modes, **kwargs)

    def _compute_modes_impl(self, env, source, n_modes, **kwargs):
        """
        Model-specific mode computation implementation.

        Caller (compute_modes) already guaranteed RunMode.MODES is supported.
        """
        dummy_receiver = Receiver(depths=[0.0], ranges=[0.0])
        return self.run(
            env, source, dummy_receiver,
            run_mode=RunMode.MODES, n_modes=n_modes, **kwargs
        )

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
        import shutil
        from uacpy.core.exceptions import ExecutableNotFoundError

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
        import subprocess
        from uacpy.core.exceptions import ModelExecutionError

        cmd_str = ' '.join(str(c) for c in cmd)
        self._log(f"Running: {cmd_str}", level='debug')

        def _raise_stack_limit():
            # Best-effort: raise the child's stack to the hard limit. If this
            # fails the child will segfault loudly on the first large alloc.
            try:
                import resource
                _soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
                target = (
                    resource.RLIM_INFINITY
                    if hard == resource.RLIM_INFINITY else hard
                )
                resource.setrlimit(resource.RLIMIT_STACK, (target, hard))
            except Exception:
                pass

        try:
            result = subprocess.run(
                [str(c) for c in cmd],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
                input=stdin_input,
                env=env,
                preexec_fn=_raise_stack_limit,
            )
        except FileNotFoundError as e:
            raise ModelExecutionError(
                self.model_name, return_code=-1,
                stdout=None, stderr=f"Executable not found: {e}",
            ) from e
        except subprocess.TimeoutExpired as e:
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

    def _validate_volume_attenuation_params(self):
        """
        Validate that `volume_attenuation='F'`/`'B'` is accompanied by its
        follow-up params. Intended for models whose __init__ stores
        `volume_attenuation`, `francois_garrison_params`, `bio_layers` as
        instance attributes. No-op if `volume_attenuation` is None.
        """
        from uacpy.core.exceptions import ConfigurationError
        va = getattr(self, 'volume_attenuation', None)
        if va is None:
            return
        code = va.upper() if isinstance(va, str) else va
        if code == 'F' and getattr(self, 'francois_garrison_params', None) is None:
            raise ConfigurationError(
                f"{self.model_name}: volume_attenuation='F' requires "
                "francois_garrison_params=(T, S, pH, z_bar)",
            )
        if code == 'B' and not getattr(self, 'bio_layers', None):
            raise ConfigurationError(
                f"{self.model_name}: volume_attenuation='B' requires "
                "bio_layers=[(Z1, Z2, f0, Q, a0), ...]",
            )

    def _warn_on_unsupported_env_features(self, env: 'Environment') -> None:
        """Warn for every Environment feature this model does not support.

        Replaces the older pattern where each model's ``run()`` either
        warned with custom wording or silently dropped data. Subclasses
        declare what they honour by setting the ``_supports_*`` flags in
        ``__init__``; this helper inspects ``env`` and emits one
        ``UserWarning`` per unsupported feature actually present.

        Notes
        -----
        Bellhop keeps its existing custom warnings (``run_with_bounce`` is
        the right alternative there, not the generic suggestion). Other
        subclasses should call this helper instead of writing their own.
        """
        import warnings as _w

        alts = self._unsupported_env_alternatives

        if (not self._supports_altimetry
                and getattr(env, 'altimetry', None) is not None):
            _w.warn(
                f"{self.model_name} does not support sea-surface altimetry. "
                f"Altimetry data will be ignored and a flat surface used. "
                f"For rough-surface support use Bellhop or RAM.",
                UserWarning, stacklevel=3,
            )

        if (not self._supports_range_dependent_ssp
                and getattr(env, 'has_range_dependent_ssp', None) is not None
                and env.has_range_dependent_ssp()):
            _w.warn(
                f"{self.model_name} does not support range-dependent SSP. "
                f"The range-0 profile will be used. "
                f"For range-dependent SSP use {alts}.",
                UserWarning, stacklevel=3,
            )

        if (not self._supports_range_dependent_bottom
                and getattr(env, 'has_range_dependent_bottom', None) is not None
                and env.has_range_dependent_bottom()):
            _w.warn(
                f"{self.model_name} does not support range-dependent bottom "
                f"properties. Using the range-0 (or median) bottom only. "
                f"For range-dependent bottoms use {alts}.",
                UserWarning, stacklevel=3,
            )

        if (not self._supports_layered_bottom
                and getattr(env, 'has_layered_bottom', None) is not None
                and env.has_layered_bottom()):
            _w.warn(
                f"{self.model_name} does not support layered (depth-"
                f"dependent) bottoms. Using halfspace properties only. "
                f"For layered bottoms use Kraken/KrakenC, Scooter, or OASES.",
                UserWarning, stacklevel=3,
            )

        if (not self._supports_range_dependent_layered_bottom
                and getattr(env, 'has_range_dependent_layered_bottom', None)
                is not None
                and env.has_range_dependent_layered_bottom()):
            _w.warn(
                f"{self.model_name} does not support range-dependent "
                f"layered bottoms. Using halfspace properties only. "
                f"For range+depth-dependent bottoms use RAM.",
                UserWarning, stacklevel=3,
            )

    def _handle_range_dependent_environment(
        self, env: 'Environment', alternatives: str = 'Bellhop or RAM'
    ) -> 'Environment':
        """
        Handle range-dependent environments for range-independent models.

        Warns the user and creates a range-independent approximation using
        the maximum bathymetry depth (to ensure source/receiver depths remain valid).

        Parameters
        ----------
        env : Environment
            Input environment (may be range-dependent)
        alternatives : str
            Models to suggest as alternatives for range-dependent modeling

        Returns
        -------
        Environment
            Range-independent environment (unchanged if already range-independent)
        """
        if env.is_range_dependent:
            import warnings
            min_depth = env.bathymetry[:, 1].min()
            max_depth = env.bathymetry[:, 1].max()
            warning_msg = (
                f"{self.model_name} does not support range-dependent environments. "
                f"Using MAXIMUM bathymetry depth ({max_depth:.1f}m) as approximation. "
                f"Bathymetry range: {min_depth:.1f}m - {max_depth:.1f}m. "
                f"This ensures source/receiver depths remain valid (prevents depth violations). "
                f"For accurate range-dependent modeling, use {alternatives}."
            )
            warnings.warn(warning_msg, UserWarning, stacklevel=3)
            self._log(warning_msg, level='warn')
            return env.get_range_independent_approximation(method='max')
        return env

    def _build_base_metadata(
        self,
        source: 'Source',
        *,
        backend: Optional[str] = None,
        frequency: Optional[float] = None,
        frequencies: Optional[np.ndarray] = None,
        phase_reference: Optional[str] = None,
        **extra,
    ) -> dict:
        """Construct the standard ``metadata`` dict every Field should carry.

        Common keys
        -----------
        ``model``           : str — class name (e.g. 'Bellhop', 'KrakenField').
        ``backend``         : str — concrete binary that ran (e.g. 'kraken.exe',
                              'mpiramS'). Defaults to ``model_name`` when the
                              wrapper is not a dispatcher.
        ``source_depths``   : ndarray — every source depth in the run.
        ``frequency``       : float — single (centre) frequency, when applicable.
        ``frequencies``     : ndarray — frequency vector, when broadband.
        ``phase_reference`` : str, optional — describes the phase convention
                              of the stored complex pressure (e.g.
                              ``'travelling_wave'``, ``'porter_negated'``,
                              ``'psif_envelope'``). Lets ``synthesize_time_series``
                              and downstream consumers correctly interpret H(f).

        Extra model-specific keys are merged via ``**extra``; existing
        per-model keys (``Q``, ``Nsam``, ``mode_coupling``, etc.) keep
        their names.
        """
        meta = {
            'model': self.model_name,
            'backend': backend or self.model_name,
            'source_depths': np.atleast_1d(np.asarray(
                getattr(source, 'depth', []), dtype=float
            )),
        }
        if frequency is not None:
            meta['frequency'] = float(frequency)
        if frequencies is not None:
            meta['frequencies'] = np.asarray(frequencies, dtype=float)
        if phase_reference is not None:
            meta['phase_reference'] = phase_reference
        meta.update(extra)
        return meta

    def _result_kwargs(
        self,
        source: 'Source',
        *,
        backend: Optional[str] = None,
        frequency: Optional[float] = None,
        frequencies: Optional[np.ndarray] = None,
        phase_reference: Optional[str] = None,
        **extra,
    ) -> dict:
        """Pre-built kwargs for any :mod:`uacpy.core.results` constructor.

        Returns a dict with the harmonised identification fields
        (``model``, ``backend``, ``source_depths``, ``frequency``,
        ``frequencies``) plus a ``metadata`` dict carrying the rest
        (``phase_reference`` and the model-specific ``**extra``).

        Spread it into a typed-Result constructor:

        >>> kw = self._result_kwargs(source, backend='ram', frequency=fc,
        ...                          phase_reference='psif_envelope', dr=2.0)
        >>> tlf = TLField(data=tl, depths=…, ranges=…, **kw)
        """
        md = dict(extra)
        if phase_reference is not None:
            md['phase_reference'] = phase_reference
        return dict(
            model=self.model_name,
            backend=backend or self.model_name,
            source_depths=np.atleast_1d(np.asarray(
                getattr(source, 'depth', []), dtype=float
            )),
            frequency=(float(frequency) if frequency is not None else None),
            frequencies=(np.asarray(frequencies, dtype=float)
                         if frequencies is not None else None),
            metadata=md,
        )

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
            receiver = Receiver(
                depths=np.clip(receiver.depths, receiver.depths.min(), env_depth - margin),
                ranges=receiver.ranges
            )
            if self.verbose:
                self._log(
                    f"Clipped receiver depths to {env_depth - margin:.1f}m "
                    f"(environment depth: {env_depth:.1f}m)"
                )
        return receiver

    def __repr__(self) -> str:
        tmpfs_str = "tmpfs" if self.use_tmpfs else "disk"
        return f"{self.model_name}(io={tmpfs_str}, verbose={self.verbose})"
