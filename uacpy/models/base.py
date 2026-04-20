"""
Base class for acoustic propagation models
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List
from enum import Enum
import warnings

from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.field import Field
from uacpy.io.file_manager import FileManager


# Sentinel for "not provided" in run() overrides (distinct from None which means auto-compute)
_UNSET = object()


def _resolve_overrides(obj, **overrides):
    """Resolve per-call overrides: return dict of param_name -> effective value.

    For each override, if the value is not _UNSET, use it; otherwise use self.<param>.
    Returns a dict and also temporarily sets the attributes on obj so that internal
    methods reading self.<param> get the overridden values.
    Returns a context manager that restores original values on exit.
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
    Standard run modes for acoustic propagation models

    Models may support a subset of these modes.
    """
    # Field computation modes
    COHERENT_TL = 'coherent_tl'          # Coherent transmission loss
    INCOHERENT_TL = 'incoherent_tl'      # Incoherent (averaged) TL
    SEMICOHERENT_TL = 'semicoherent_tl'  # Semi-coherent TL

    # Ray-based modes (mainly Bellhop)
    RAYS = 'rays'                        # Ray paths only
    EIGENRAYS = 'eigenrays'              # Eigenrays (specific paths)
    ARRIVALS = 'arrivals'                # Arrival structure

    # Mode-based (mainly Kraken)
    MODES = 'modes'                      # Normal modes only
    MODES_FIELD = 'modes_field'          # Field from modes

    # Time-domain (mainly SPARC, OASES)
    TIME_SERIES = 'time_series'          # Time-domain response


class PropagationModel(ABC):
    """
    Abstract base class for acoustic propagation models

    Provides common interface and utilities for all propagation models.

    Parameters
    ----------
    use_tmpfs : bool, optional
        Use RAM-based filesystem for I/O. Default is False.
    verbose : bool, optional
        Print verbose output. Default is False.
    work_dir : str or Path, optional
        Working directory for files. If None, creates temporary.

    Attributes
    ----------
    model_name : str
        Name of the model
    use_tmpfs : bool
        Whether tmpfs is used
    verbose : bool
        Verbose output flag
    file_manager : FileManager
        File manager instance
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

        # Models should override this to declare supported modes
        self._supported_modes: List[RunMode] = [RunMode.COHERENT_TL]
        # Models that support sea surface altimetry should set this to True
        self._supports_altimetry: bool = False

    @property
    def supported_modes(self) -> List[RunMode]:
        """List of run modes supported by this model"""
        return self._supported_modes

    def supports_mode(self, mode: RunMode) -> bool:
        """Check if model supports a specific run mode"""
        return mode in self._supported_modes

    @abstractmethod
    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        **kwargs
    ) -> Field:
        """
        Run the propagation model

        Parameters
        ----------
        env : Environment
            Environment definition
        source : Source
            Source definition
        receiver : Receiver
            Receiver definition
        **kwargs
            Model-specific parameters

        Returns
        -------
        field : Field
            Simulation results
        """
        pass

    def _setup_file_manager(self) -> FileManager:
        """
        Create and setup file manager

        Returns
        -------
        fm : FileManager
            Configured file manager
        """
        if self.work_dir is not None:
            cleanup = False
        else:
            cleanup = True

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
        Log message with timestamp and level

        Parameters
        ----------
        message : str
            Message to log
        level : str, optional
            Log level: 'info', 'warn', 'error', 'debug'. Default is 'info'.
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
            self._logger.info(message)  # Default to info

    def validate_inputs(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver
    ):
        """
        Validate input parameters

        Parameters
        ----------
        env : Environment
            Environment to validate
        source : Source
            Source to validate
        receiver : Receiver
            Receiver to validate

        Raises
        ------
        ValueError
            If inputs are invalid
        """
        # Determine maximum depth (consider range-dependent bathymetry)
        if env.is_range_dependent and env.bathymetry is not None:
            max_depth = np.max(env.bathymetry[:, 1])
        else:
            max_depth = env.depth

        # Check source depth within environment
        if np.any(source.depth > max_depth):
            error_msg = f"Source depth ({source.depth.max():.1f}m) exceeds maximum environment depth ({max_depth:.1f}m)"
            self._log(error_msg, level='error')
            raise ValueError(error_msg)

        # Check receiver depth within environment
        if receiver.depth_max > max_depth:
            error_msg = f"Receiver depth ({receiver.depth_max:.1f}m) exceeds maximum environment depth ({max_depth:.1f}m)"
            self._log(error_msg, level='error')
            raise ValueError(error_msg)

        # Check positive depths
        if np.any(source.depth < 0):
            self._log("Source depths must be positive", level='error')
            raise ValueError("Source depths must be positive")

        if receiver.depth_min < 0:
            self._log("Receiver depths must be positive", level='error')
            raise ValueError("Receiver depths must be positive")

        # Warn if altimetry provided but model doesn't support it
        if getattr(env, 'altimetry', None) is not None and not self._supports_altimetry:
            warnings.warn(
                f"{self.model_name} does not support sea surface altimetry. "
                "Altimetry data will be ignored and a flat surface will be used. "
                "Use Bellhop for altimetry/sea surface roughness support.",
                UserWarning,
                stacklevel=3,
            )

    def compute_tl(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver = None,
        **kwargs
    ) -> Field:
        """
        Compute transmission loss (convenience method)

        This is a user-friendly wrapper around run() that automatically
        configures the model for TL computation.

        Parameters
        ----------
        env : Environment
            Ocean environment
        source : Source
            Acoustic source
        receiver : Receiver, optional
            Receiver array. If None, creates automatic grid.
        **kwargs
            Additional model-specific parameters

        Returns
        -------
        field : Field
            Transmission loss field

        Examples
        --------
        >>> bellhop = Bellhop()
        >>> result = bellhop.compute_tl(env, source, receiver)
        >>> # Much simpler than: bellhop.run(env, source, receiver, run_type='C')

        >>> # Auto-generate receiver grid
        >>> result = bellhop.compute_tl(env, source, max_range=10000)
        """
        # Auto-generate receiver if not provided
        if receiver is None:
            from uacpy.core.model_utils import ReceiverGridBuilder
            max_range = kwargs.pop('max_range', 10000.0)
            depths, ranges = ReceiverGridBuilder.build_tl_grid(env.depth, max_range)
            receiver = Receiver(depths=depths, ranges=ranges)

        # Call model-specific implementation
        return self._compute_tl_impl(env, source, receiver, **kwargs)

    def _compute_tl_impl(self, env, source, receiver, **kwargs):
        """
        Model-specific TL computation implementation

        Models can override this to provide optimized TL computation.
        Default implementation calls run() with appropriate parameters.
        """
        # Determine appropriate run parameters for this model
        if self.supports_mode(RunMode.COHERENT_TL):
            # Model supports RunMode - use it
            return self.run(env, source, receiver, run_mode=RunMode.COHERENT_TL, **kwargs)
        else:
            # Fallback to basic run()
            return self.run(env, source, receiver, **kwargs)

    def compute_rays(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver = None,
        **kwargs
    ) -> Field:
        """
        Compute ray paths (convenience method)

        Parameters
        ----------
        env : Environment
            Ocean environment
        source : Source
            Acoustic source
        receiver : Receiver, optional
            Receiver array. If None, creates automatic grid.
        **kwargs
            Additional model-specific parameters

        Returns
        -------
        field : Field
            Ray path data

        Examples
        --------
        >>> bellhop = Bellhop()
        >>> rays = bellhop.compute_rays(env, source)
        >>> # Much simpler than: bellhop.run(env, source, receiver, run_type='R')

        Raises
        ------
        UnsupportedFeatureError
            If model doesn't support ray computation
        """
        from uacpy.core.exceptions import UnsupportedFeatureError

        if not self.supports_mode(RunMode.RAYS):
            raise UnsupportedFeatureError(
                self.model_name,
                "ray path computation",
                alternatives=['Bellhop']
            )

        # Auto-generate receiver if not provided
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
    ) -> Field:
        """
        Compute arrival structure (convenience method)

        Parameters
        ----------
        env : Environment
            Ocean environment
        source : Source
            Acoustic source
        receiver : Receiver
            Receiver array
        **kwargs
            Additional model-specific parameters

        Returns
        -------
        field : Field
            Arrival data

        Examples
        --------
        >>> bellhop = Bellhop()
        >>> arrivals = bellhop.compute_arrivals(env, source, receiver)

        Raises
        ------
        UnsupportedFeatureError
            If model doesn't support arrival computation
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
    ) -> Field:
        """
        Compute normal modes (convenience method)

        Parameters
        ----------
        env : Environment
            Ocean environment (must be range-independent for most models)
        source : Source
            Acoustic source (for frequency specification)
        n_modes : int, optional
            Number of modes to compute. If None, computes all modes.
        **kwargs
            Additional model-specific parameters

        Returns
        -------
        field : Field
            Mode data (field_type='modes')

        Examples
        --------
        >>> # Kraken normal modes
        >>> kraken = Kraken()
        >>> modes = kraken.compute_modes(env, source, n_modes=50)
        >>> print(f"Computed {len(modes.metadata['k'])} modes")
        >>> print(f"Field type: {modes.field_type}")  # 'modes'
        >>> print(f"Mode shapes shape: {modes.data.shape}")

        >>> # OASN also supports mode computation
        >>> oasn = OASN()
        >>> modes = oasn.compute_modes(env, source, n_modes=30)

        >>> # Access mode data from Field object
        >>> wavenumbers = modes.metadata['k']
        >>> mode_shapes = modes.metadata['phi']  # or modes.data

        Raises
        ------
        UnsupportedFeatureError
            If model doesn't support mode computation
        EnvironmentError
            If environment is range-dependent and model doesn't support it
        """
        from uacpy.core.exceptions import UnsupportedFeatureError, EnvironmentError

        if not self.supports_mode(RunMode.MODES):
            raise UnsupportedFeatureError(
                self.model_name,
                "normal mode computation",
                alternatives=['Kraken', 'OASN']
            )

        # Check for range-dependent environment
        if env.is_range_dependent:
            # Only certain models can handle range-dependent modes
            if self.model_name not in ['KrakenField']:  # KrakenField has coupled mode theory
                raise EnvironmentError(
                    f"{self.model_name} does not support range-dependent environments for mode computation. "
                    f"Environment has bathymetry ranging from {env.bathymetry[:, 1].min():.1f}m to "
                    f"{env.bathymetry[:, 1].max():.1f}m.",
                    "Use a range-independent environment or try KrakenField with adiabatic coupling"
                )

        # Call model-specific implementation
        return self._compute_modes_impl(env, source, n_modes, **kwargs)

    def _compute_modes_impl(self, env, source, n_modes, **kwargs):
        """
        Model-specific mode computation implementation

        Models can override this to provide optimized mode computation.
        Default implementation calls run() with appropriate parameters.
        """
        # Create dummy receiver (modes don't need receiver grid)
        dummy_receiver = Receiver(depths=[0.0], ranges=[0.0])

        if self.supports_mode(RunMode.MODES):
            return self.run(env, source, dummy_receiver, run_mode=RunMode.MODES, n_modes=n_modes, **kwargs)
        else:
            # Fallback - should not reach here due to check in compute_modes()
            return self.run(env, source, dummy_receiver, n_modes=n_modes, **kwargs)

    def _find_executable_in_paths(self, name: str, bin_subdir: str) -> Path:
        """
        Find a model executable by searching standard locations.

        Searches in order:
        1. uacpy/bin/<bin_subdir>/<name>
        2. System PATH

        Parameters
        ----------
        name : str
            Executable name (e.g., 'bellhop.exe', 'kraken.exe')
        bin_subdir : str
            Subdirectory under uacpy/bin/ (e.g., 'oalib', 'mpirams')

        Returns
        -------
        Path
            Path to the executable

        Raises
        ------
        FileNotFoundError
            If executable not found in any location
        """
        import shutil

        # Check in uacpy/bin/<subdir>/
        bin_path = Path(__file__).parent.parent / 'bin' / bin_subdir / name
        if bin_path.exists():
            return bin_path

        # Check system PATH
        result = shutil.which(name)
        if result:
            return Path(result)

        raise FileNotFoundError(
            f"Could not find {name}. Run install.sh to compile, "
            f"or ensure {name} is on your PATH."
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


# Import numpy for validation
import numpy as np
