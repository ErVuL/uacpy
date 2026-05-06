"""Base class for acoustic propagation models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict
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
        bathymetry_collapse_method: str = 'max',
        ssp_collapse_method: str = 'r0',
        bottom_collapse_method: str = 'r0',
        layered_collapse_method: str = 'halfspace',
        rd_layered_collapse_method: str = 'halfspace',
        altimetry_collapse_method: str = 'drop',
        elastic_collapse_method: str = 'fluid',
    ):
        self.model_name = self.__class__.__name__
        self.use_tmpfs = use_tmpfs
        self.verbose = verbose
        self.work_dir = work_dir
        # Per-feature collapse policies applied by ``_project_environment``
        # when an env contains a feature this model doesn't support.
        # Defaults preserve historical behaviour (range-0 / halfspace / drop)
        # so existing callers are unaffected.
        #
        # bathymetry_collapse_method : 'max'|'median'|'mean'|'min'|'initial'
        #     Single representative water depth for range-independent models.
        # ssp_collapse_method : 'r0'|'rmax'|'mean'|'median'
        #     Reduce 2-D SSP to 1-D.
        # bottom_collapse_method : 'r0'|'rmax'|'mean'|'median'
        #     Reduce ``RangeDependentBottom`` to a single ``BoundaryProperties``.
        # layered_collapse_method : 'halfspace'|'top_layer'|'volume_average'
        #     Reduce ``LayeredBottom`` to a single ``BoundaryProperties``.
        # rd_layered_collapse_method : 'halfspace'|'top_layer'|'volume_average'
        #     Layered method applied after picking a single profile from a
        #     ``RangeDependentLayeredBottom``.
        # altimetry_collapse_method : 'drop'
        # elastic_collapse_method   : 'fluid' (zero shear) | 'vacuum'
        #     Applied to BOTH env.surface and env.bottom whenever the
        #     interface has shear_speed > 0 and the model does not support
        #     elastic media.
        self.bathymetry_collapse_method = bathymetry_collapse_method
        self.ssp_collapse_method = ssp_collapse_method
        self.bottom_collapse_method = bottom_collapse_method
        self.layered_collapse_method = layered_collapse_method
        self.rd_layered_collapse_method = rd_layered_collapse_method
        self.altimetry_collapse_method = altimetry_collapse_method
        self.elastic_collapse_method = elastic_collapse_method
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
        # Per-feature suggestion strings appended to each warning. Override
        # individual entries per subclass (e.g. Bellhop adds a
        # ``run_with_bounce()`` hint to ``layered_bottom``).
        self._unsupported_env_alternatives: Dict[str, str] = {
            'altimetry': 'Bellhop or RAM (ramsurf backend)',
            'range_dependent_bathymetry': (
                'Bellhop, BellhopCUDA, KrakenField, or RAM'
            ),
            'range_dependent_ssp': (
                'Bellhop (with ssp.interp=\'quad\'), KrakenField, or RAM'
            ),
            'range_dependent_bottom': 'Bellhop (long .bty) or RAM',
            'layered_bottom': 'Kraken/KrakenC, Scooter, or OASES',
            'range_dependent_layered_bottom': 'RAM',
            'elastic_media': (
                'Bellhop, KrakenC, KrakenField, Scooter, OASES, Bounce, '
                'or RAM (auto-routes to rams0.5 for elastic bottom)'
            ),
        }

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

        Mode-specific kwargs follow a fixed convention: every
        TIME_SERIES-capable wrapper accepts ``source_waveform=`` and
        ``sample_rate=`` as explicit keyword arguments on ``run()``
        (Bellhop, Scooter, KrakenField, OASP, RAM); SPARC computes
        p(t) from its native source pulse and ignores both. Models with
        a broadband transfer-function path also accept ``frequencies=``
        as an explicit override for ``source.frequency``.

        Any kwarg not consumed by an explicit signature arg, an override,
        or a documented writer/mode pass-through is reported via
        ``_warn_unknown_kwargs`` (typo guard).

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

    def _resolve(self, value, attr: str):
        """Resolve a per-call override against the constructor default.

        Returns ``value`` unchanged unless it is the ``_UNSET`` sentinel,
        in which case ``getattr(self, attr)`` is returned. Used by
        ``run()`` paths that pass resolved values straight to writers
        (where the ``_resolve_overrides`` context manager doesn't apply).
        """
        return getattr(self, attr) if value is _UNSET else value

    def _warn_unknown_kwargs(self, kwargs: dict, allowed: tuple = ()):
        """Emit a ``UserWarning`` for any kwarg not consumed by ``run()``.

        Catches typos like ``Bellhop().run(env, src, rcv, n_beam=10)``
        (missing 's') that would otherwise silently use the default.

        ``kwargs`` is the leftover ``**kwargs`` dict from a model's
        ``run()`` after explicit named args have been peeled off.
        ``allowed`` is the tuple of writer- or mode-specific kwarg names
        the model legitimately forwards downstream (e.g. Bellhop's
        Cerveny beam params); these are silently passed through.
        """
        unknown = sorted(k for k in kwargs if k not in allowed)
        if unknown:
            warnings.warn(
                f"{self.model_name}.run received unknown kwargs (ignored): "
                f"{unknown}",
                UserWarning,
                stacklevel=3,
            )

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

        Examples
        --------
        >>> kraken = Kraken()
        >>> modes = kraken.compute_modes(env, source, n_modes=50)
        >>> wavenumbers = modes.metadata['k']
        >>> mode_shapes = modes.metadata['phi']
        """
        from uacpy.core.exceptions import UnsupportedFeatureError

        if not self.supports_mode(RunMode.MODES):
            raise UnsupportedFeatureError(
                self.model_name,
                "normal mode computation",
                alternatives=['Kraken', 'OASN']
            )

        if env.is_range_dependent and self.model_name != 'KrakenField':
            # Range-independent mode solvers (Kraken, KrakenC, OASN) collapse
            # the environment via ``bathymetry_collapse_method`` and warn,
            # rather than reject — same pattern as OAST/OASP/Scooter/SPARC.
            env = self._project_environment(env)

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

    @staticmethod
    def _has_shear(boundary) -> bool:
        """True if a ``BoundaryProperties`` carries a non-zero shear speed."""
        cs = getattr(boundary, 'shear_speed', None)
        return cs is not None and float(cs) > 0.0

    @staticmethod
    def _collapse_elastic_boundary(boundary, method: str, default_depth: float):
        """Collapse an elastic ``BoundaryProperties`` per ``method``.

        ``'fluid'``  : zero shear_speed and shear_attenuation; keep cp / ρ / α.
        ``'vacuum'`` : replace with default vacuum BoundaryProperties.
        """
        from uacpy.core.environment import BoundaryProperties
        import copy as _copy
        if method == 'fluid':
            b = _copy.deepcopy(boundary)
            b.shear_speed = 0.0
            b.shear_attenuation = 0.0
            return b
        if method == 'vacuum':
            return BoundaryProperties(
                acoustic_type='vacuum', depth=float(default_depth),
            )
        raise ValueError(
            f"Unknown elastic collapse method {method!r}. Use "
            "'fluid' or 'vacuum'."
        )

    def _project_environment(self, env: 'Environment') -> 'Environment':
        """Return a copy of ``env`` with every unsupported feature collapsed.

        Walks the eight feature axes (altimetry, range-dependent
        bathymetry, range-dependent SSP, range-dependent bottom, layered
        bottom, range-dependent layered bottom, elastic surface, elastic
        bottom) and applies the per-feature ``*_collapse_method``
        configured on the model. Emits one ``UserWarning`` per dropped
        feature, citing the chosen method and the alternative-model hint
        from ``self._unsupported_env_alternatives``.

        Models call this once at the top of ``run()`` instead of
        inspecting the env themselves.
        """
        import warnings as _w

        e = env.copy()
        alts = self._unsupported_env_alternatives

        if env.altimetry is not None and not self._supports_altimetry:
            method = self.altimetry_collapse_method
            if method != 'drop':
                raise ValueError(
                    f"Unknown altimetry_collapse_method {method!r}. "
                    "Currently only 'drop' is supported."
                )
            e.altimetry = None
            _w.warn(
                f"{self.model_name} does not support sea-surface altimetry; "
                f"using flat surface (altimetry_collapse_method={method!r}). "
                f"For rough-surface support use {alts['altimetry']}.",
                UserWarning, stacklevel=3,
            )
            self._log(
                f"{self.model_name}: altimetry dropped ({method}).",
                level='warn',
            )

        if (env.has_range_dependent_bathymetry()
                and not self._supports_range_dependent_bathymetry):
            method = self.bathymetry_collapse_method
            new_depth = env.get_representative_depth(method)
            e.bathymetry = np.array([[0.0, new_depth]], dtype=np.float64)
            e.depth = float(new_depth)
            if e.ssp.depths[-1] < new_depth:
                e.ssp = e.ssp.extend_to(new_depth)
            min_d = float(env.bathymetry[:, 1].min())
            max_d = float(env.bathymetry[:, 1].max())
            _w.warn(
                f"{self.model_name} does not support range-dependent "
                f"bathymetry; collapsed to {new_depth:.1f} m "
                f"(method={method!r}, range {min_d:.1f}–{max_d:.1f} m). "
                f"For RD bathymetry use "
                f"{alts['range_dependent_bathymetry']}. "
                f"Override with `bathymetry_collapse_method='min'|'median'|"
                f"'mean'|'max'|'initial'`.",
                UserWarning, stacklevel=3,
            )
            self._log(
                f"{self.model_name}: bathymetry collapsed ({method} → "
                f"{new_depth:.1f} m).",
                level='warn',
            )

        if (env.has_range_dependent_ssp()
                and not self._supports_range_dependent_ssp):
            method = self.ssp_collapse_method
            e.ssp = e.ssp.collapse(method)
            _w.warn(
                f"{self.model_name} does not support range-dependent SSP; "
                f"collapsed to 1-D (ssp_collapse_method={method!r}). "
                f"For RD SSP use {alts['range_dependent_ssp']}.",
                UserWarning, stacklevel=3,
            )
            self._log(
                f"{self.model_name}: SSP collapsed ({method}).", level='warn',
            )

        if (env.has_range_dependent_bottom()
                and not self._supports_range_dependent_bottom):
            method = self.bottom_collapse_method
            e.bottom = e.bottom_rd.collapse(method)
            e.bottom.depth = e.depth
            e.bottom_rd = None
            _w.warn(
                f"{self.model_name} does not support range-dependent bottom "
                f"geoacoustics; collapsed to single profile "
                f"(bottom_collapse_method={method!r}). "
                f"For RD bottoms use {alts['range_dependent_bottom']}.",
                UserWarning, stacklevel=3,
            )
            self._log(
                f"{self.model_name}: bottom_rd collapsed ({method}).",
                level='warn',
            )

        if (env.has_range_dependent_layered_bottom()
                and not self._supports_range_dependent_layered_bottom):
            layered_method = self.rd_layered_collapse_method
            e.bottom = e.bottom_rd_layered.collapse(
                layered_method=layered_method, range_method='middle',
            )
            e.bottom.depth = e.depth
            e.bottom_rd_layered = None
            _w.warn(
                f"{self.model_name} does not support range-dependent layered "
                f"bottoms; collapsed to single boundary "
                f"(rd_layered_collapse_method={layered_method!r}). "
                f"For RD-layered use "
                f"{alts['range_dependent_layered_bottom']}.",
                UserWarning, stacklevel=3,
            )
            self._log(
                f"{self.model_name}: bottom_rd_layered collapsed "
                f"({layered_method}).",
                level='warn',
            )

        if (env.has_layered_bottom()
                and not self._supports_layered_bottom):
            method = self.layered_collapse_method
            e.bottom = e.bottom_layered.collapse(method)
            e.bottom.depth = e.depth
            e.bottom_layered = None
            _w.warn(
                f"{self.model_name} does not support layered (depth-"
                f"dependent) bottoms; collapsed to single boundary "
                f"(layered_collapse_method={method!r}). "
                f"For layered bottoms use {alts['layered_bottom']}.",
                UserWarning, stacklevel=3,
            )
            self._log(
                f"{self.model_name}: layered bottom collapsed ({method}).",
                level='warn',
            )

        if not self._supports_elastic_media:
            collapsed_at = []
            if e.surface is not None and self._has_shear(e.surface):
                e.surface = self._collapse_elastic_boundary(
                    e.surface, self.elastic_collapse_method,
                    default_depth=0.0,
                )
                collapsed_at.append('surface')
            if e.bottom is not None and self._has_shear(e.bottom):
                e.bottom = self._collapse_elastic_boundary(
                    e.bottom, self.elastic_collapse_method,
                    default_depth=e.depth,
                )
                collapsed_at.append('bottom')
            if collapsed_at:
                method = self.elastic_collapse_method
                where = '/'.join(collapsed_at)
                _w.warn(
                    f"{self.model_name} does not support elastic media; "
                    f"collapsed shear properties on {where} "
                    f"(elastic_collapse_method={method!r}). "
                    f"For elastic media use {alts['elastic_media']}.",
                    UserWarning, stacklevel=3,
                )
                self._log(
                    f"{self.model_name}: elastic {where} collapsed ({method}).",
                    level='warn',
                )

        return e

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
                              ``'travelling_wave'``, ``'psif_envelope'``,
                              ``'time_domain_native'``). Lets
                              ``synthesize_time_series`` and downstream
                              consumers correctly interpret H(f).

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
