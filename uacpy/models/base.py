"""Base class for acoustic propagation models."""

import copy as _copy
import shutil
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, List, Dict, Set, Union
from enum import Enum
import warnings

import numpy as np

from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.results import Result
from uacpy.core.constants import DEFAULT_SOUND_SPEED
from uacpy.core.exceptions import (
    ConfigurationError,
    ExecutableNotFoundError,
    InvalidDepthError,
    ModelExecutionError,
    UnsupportedFeatureError,
)
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
        cleanup: Optional[bool] = None,
        timeout: float = 600.0,
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
        # cleanup defaults to True only when uacpy owns the work dir.
        self.cleanup = (work_dir is None) if cleanup is None else bool(cleanup)
        self.timeout = float(timeout)
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
        as an explicit override for ``source.frequencies``.

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

    # Modes that consume exactly one source frequency. Multi-frequency
    # Source passed to one of these is a configuration error — the user
    # should pick BROADBAND/TIME_SERIES, or REFLECTION/COVARIANCE/REPLICA
    # for the OASES family that genuinely supports multi-freq sweeps.
    _SINGLE_FREQUENCY_MODES: 'frozenset[RunMode]' = frozenset({
        RunMode.COHERENT_TL, RunMode.INCOHERENT_TL, RunMode.SEMICOHERENT_TL,
        RunMode.RAYS, RunMode.EIGENRAYS, RunMode.ARRIVALS, RunMode.MODES,
    })

    _supports_multi_source_depth: bool = False

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
        """Build the FileManager. ``self.work_dir`` is used as-is (not a
        parent); when ``None``, a fresh temp dir is created."""
        if self.work_dir is not None:
            fm = FileManager(
                use_tmpfs=False,
                base_dir=self.work_dir,
                prefix=f'{self.model_name.lower()}_',
                cleanup=getattr(self, 'cleanup', False),
            )
            fm.work_dir = Path(self.work_dir)
            fm.work_dir.mkdir(parents=True, exist_ok=True)
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
        """Print ``message`` tagged with model + level. WARN/ERROR always
        print; INFO/DEBUG only when ``self.verbose``."""
        from datetime import datetime, timezone
        lvl = level.lower()
        if lvl in ('warn', 'warning'):
            label = 'WARN'
        elif lvl == 'error':
            label = 'ERROR'
        elif lvl == 'debug':
            if not self.verbose:
                return
            label = 'DEBUG'
        else:
            if not self.verbose:
                return
            label = 'INFO'
        ts = datetime.now(timezone.utc).strftime("%Y/%m/%d %H:%M:%S UTC")
        print(f"[{ts}] [{label}] [{self.model_name}] {message}")

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
        # For flat environments, env.depth may differ from env.bathymetry[0, 1]
        # (e.g. Environment(depth=100, bathymetry=[(0, 80)]) reports 100m but
        # the seafloor is at 80m). Use the deeper of the two to avoid false
        # depth-violation errors on the shallower side.
        if env.is_range_dependent and env.bathymetry is not None:
            max_depth = float(np.max(env.bathymetry[:, 1]))
        else:
            bathy_depth = float(env.bathymetry[0, 1]) if env.bathymetry is not None and len(env.bathymetry) > 0 else env.depth
            max_depth = max(env.depth, bathy_depth)

        if np.any(source.depths > max_depth):
            self._log(
                f"Source depth ({source.depths.max():.1f}m) exceeds maximum "
                f"environment depth ({max_depth:.1f}m)",
                level='error',
            )
            raise InvalidDepthError(
                float(source.depths.max()), max_depth, "Source",
            )

        if receiver.depth_max > max_depth:
            self._log(
                f"Receiver depth ({receiver.depth_max:.1f}m) exceeds maximum "
                f"environment depth ({max_depth:.1f}m)",
                level='error',
            )
            raise InvalidDepthError(
                float(receiver.depth_max), max_depth, "Receiver",
            )

        if np.any(source.depths < 0):
            self._log("Source depths must be positive", level='error')
            raise ValueError("Source depths must be positive")

        if receiver.depth_min < 0:
            self._log("Receiver depths must be positive", level='error')
            raise ValueError("Receiver depths must be positive")

    def compute_tl(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        **kwargs,
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
        **kwargs
            Forwarded to :meth:`run`.

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
        # Allow callers to echo run_mode=COHERENT_TL/INCOHERENT_TL/SEMICOHERENT_TL
        # but reject anything off-the-TL-family — the explicit method choice
        # already pins the intent.
        run_mode = kwargs.pop('run_mode', RunMode.COHERENT_TL)
        if run_mode not in (
            RunMode.COHERENT_TL, RunMode.INCOHERENT_TL, RunMode.SEMICOHERENT_TL,
        ):
            raise ValueError(
                f"compute_tl() got run_mode={run_mode}; only COHERENT_TL / "
                f"INCOHERENT_TL / SEMICOHERENT_TL are accepted. Call "
                f"{self.model_name}.run(run_mode=…) for other modes."
            )
        return self.run(env, source, receiver, run_mode=run_mode, **kwargs)

    def compute_rays(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        **kwargs,
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
            :class:`Modes` instance.

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

    def compute_eigenrays(
        self,
        env: Environment,
        source: Source,
        receiver: Optional[Receiver] = None,
        range_m: Optional[float] = None,
        depth_m: Optional[float] = None,
        tolerance_m: Optional[float] = None,
        max_rays: Optional[int] = None,
        truncate: bool = True,
        **kwargs,
    ) -> Result:
        """Compute eigenrays — rays that arrive at the receiver(s).

        Runs the model's eigenray solver (Bellhop's ``RunType='E'``) and,
        for a single-point query, sorts the returned rays by closest-
        approach miss distance, drops any beyond ``tolerance_m`` (default:
        one acoustic wavelength), caps to ``max_rays``, and (when
        ``truncate=True``) trims each kept polyline at its closest-
        approach index for clean display.

        Two usage patterns:

        * Single point — pass ``range_m`` and ``depth_m``. A 1-point
          ``Receiver`` is built internally and the cosmetic post-filter
          fires.
        * Multi-receiver — pass a ``Receiver`` directly. The solver
          targets every receiver point; the post-filter is skipped (no
          single anchor) and ``tolerance_m`` / ``max_rays`` / ``truncate``
          are ignored. To narrow afterwards, call ``compute_eigenrays``
          again for the specific point of interest.

        ``**kwargs`` forwards to :meth:`run`, so the full per-model
        configuration surface (beam type, step size, etc.) is available.

        Examples
        --------
        >>> bellhop = Bellhop(verbose=False, alpha=(-20, 20), n_beams=51)
        >>> rays = bellhop.compute_eigenrays(env, source,
        ...                                   range_m=2000, depth_m=30,
        ...                                   tolerance_m=15, max_rays=8)
        >>> for r in rays.rays:
        ...     print(r['miss_distance_m'])
        """
        from uacpy.core.results import Rays as _Rays

        if not self.supports_mode(RunMode.EIGENRAYS):
            raise UnsupportedFeatureError(
                self.model_name,
                "eigenray computation",
                alternatives=['Bellhop'],
            )

        single_point = range_m is not None and depth_m is not None
        if receiver is None:
            if not single_point:
                raise ValueError(
                    "compute_eigenrays requires either receiver=… or both "
                    "range_m=… and depth_m=…"
                )
            receiver = Receiver(
                depths=np.array([float(depth_m)]),
                ranges=np.array([float(range_m)]),
            )
        elif single_point:
            raise ValueError(
                "Pass either receiver=… OR (range_m, depth_m), not both."
            )

        result = self.run(env, source, receiver,
                          run_mode=RunMode.EIGENRAYS, **kwargs)

        if not single_point:
            return result

        if tolerance_m is None:
            f = float(np.atleast_1d(source.frequencies)[0])
            tolerance_m = DEFAULT_SOUND_SPEED / f if f > 0 else float('inf')

        rr_km = range_m / 1000.0
        scored = []
        for ray in result.rays:
            r_km = np.asarray(ray.get('r', [])) / 1000.0
            z = np.asarray(ray.get('z', []))
            if len(r_km) == 0:
                continue
            d2 = (r_km - rr_km) ** 2 + ((z - depth_m) / 1000.0) ** 2
            k = int(np.argmin(d2))
            miss_m = float(np.sqrt(d2[k]) * 1000.0)
            if miss_m > tolerance_m:
                continue
            ray = dict(ray)
            if truncate and k + 1 < len(r_km):
                ray['r'] = np.asarray(ray['r'])[: k + 1]
                ray['z'] = np.asarray(ray['z'])[: k + 1]
            ray['miss_distance_m'] = miss_m
            scored.append((miss_m, ray))

        scored.sort(key=lambda t: t[0])
        if max_rays is not None:
            scored = scored[:max_rays]

        meta = dict(result.metadata)
        meta['receiver_range_m'] = float(range_m)
        meta['receiver_depth_m'] = float(depth_m)
        meta['tolerance_m'] = tolerance_m
        return _Rays(
            rays=[r for _, r in scored],
            is_eigen=True,
            receiver_depths=np.array([float(depth_m)]),
            receiver_ranges=np.array([float(range_m)]),
            model=result.model,
            backend=result.backend,
            source_depths=result.source_depths,
            frequencies=result.frequencies,
            metadata=meta,
        )

    def compute_reflection(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        **kwargs,
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
        return self.run(
            env, source, receiver, run_mode=RunMode.REFLECTION, **kwargs,
        )

    def compute_time_series(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        source_waveform=None,
        sample_rate=None,
        **kwargs,
    ) -> Result:
        """Compute time-domain pressure p(t) at the receiver(s).

        Forwards ``source_waveform`` and ``sample_rate`` to
        ``run(run_mode=RunMode.TIME_SERIES)``. SPARC ignores both (it
        builds p(t) from its native pulse); every other TIME_SERIES
        model requires them.
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
            **kwargs,
        )

    def compute_transfer_function(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        **kwargs,
    ) -> Result:
        """Compute broadband complex transfer function H(f).

        Dispatches to ``run(run_mode=RunMode.BROADBAND)``.
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
            run_mode=RunMode.BROADBAND, **kwargs,
        )

    def compute_covariance(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        **kwargs,
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
        return self.run(
            env, source, receiver,
            run_mode=RunMode.COVARIANCE, **kwargs,
        )

    def compute_replicas(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        **kwargs,
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
        return self.run(
            env, source, receiver,
            run_mode=RunMode.REPLICA, **kwargs,
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
        if timeout is None:
            timeout = getattr(self, 'timeout', 600.0)
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

        cmd_str = ' '.join(str(c) for c in cmd)
        self._log(f"Running: {cmd_str}", level='debug')

        def _raise_stack_limit():
            # Raise the child's stack to the hard limit. If this fails the
            # SPARC-class binaries segfault on first large alloc, so surface
            # the failure rather than swallow it.
            try:
                import resource
                _soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
                target = (
                    resource.RLIM_INFINITY
                    if hard == resource.RLIM_INFINITY else hard
                )
                resource.setrlimit(resource.RLIMIT_STACK, (target, hard))
            except (ImportError, ValueError, OSError) as exc:
                warnings.warn(
                    f"Could not raise child stack limit ({exc}); "
                    f"large-stack binaries may segfault.",
                    RuntimeWarning, stacklevel=2,
                )

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

    # Each axis: (alt_key, support_flag_attr, method_attr, detector,
    #             collapser, log_label). detector(env) → bool. collapser
    #             takes (self, env_in, e_out, method) → warning_message
    #             (str). Walked once by ``_project_environment``.
    @staticmethod
    def _detect_altimetry(env):
        return env.altimetry is not None

    @staticmethod
    def _detect_rd_bathy(env):
        return env.has_range_dependent_bathymetry()

    @staticmethod
    def _detect_rd_ssp(env):
        return env.has_range_dependent_ssp()

    @staticmethod
    def _detect_rd_bottom(env):
        return env.has_range_dependent_bottom()

    @staticmethod
    def _detect_rd_layered(env):
        return env.has_range_dependent_layered_bottom()

    @staticmethod
    def _detect_layered(env):
        return env.has_layered_bottom()

    def _collapse_altimetry(self, env_in, e, method, alts):
        if method != 'drop':
            raise ValueError(
                f"Unknown altimetry_collapse_method {method!r}. "
                "Currently only 'drop' is supported."
            )
        e.altimetry = None
        return (
            f"{self.model_name} does not support sea-surface altimetry; "
            f"using flat surface (altimetry_collapse_method={method!r}). "
            f"For rough-surface support use {alts['altimetry']}."
        ), 'altimetry'

    def _collapse_rd_bathy(self, env_in, e, method, alts):
        new_depth = env_in.get_representative_depth(method)
        e.bathymetry = np.array([[0.0, new_depth]], dtype=np.float64)
        e.depth = float(new_depth)
        if e.ssp.depths[-1] < new_depth:
            e.ssp = e.ssp.extend_to(new_depth)
        min_d = float(env_in.bathymetry[:, 1].min())
        max_d = float(env_in.bathymetry[:, 1].max())
        msg = (
            f"{self.model_name} does not support range-dependent "
            f"bathymetry; collapsed to {new_depth:.1f} m "
            f"(method={method!r}, range {min_d:.1f}–{max_d:.1f} m). "
            f"For RD bathymetry use "
            f"{alts['range_dependent_bathymetry']}. "
            f"Override with `bathymetry_collapse_method='min'|'median'|"
            f"'mean'|'max'|'initial'`."
        )
        return msg, f"bathymetry ({method} → {new_depth:.1f} m)"

    def _collapse_rd_ssp(self, env_in, e, method, alts):
        e.ssp = e.ssp.collapse(method)
        return (
            f"{self.model_name} does not support range-dependent SSP; "
            f"collapsed to 1-D (ssp_collapse_method={method!r}). "
            f"For RD SSP use {alts['range_dependent_ssp']}."
        ), f"SSP ({method})"

    def _collapse_rd_bottom(self, env_in, e, method, alts):
        e.bottom = e.bottom_rd.collapse(method)
        e.bottom.depth = e.depth
        e.bottom_rd = None
        return (
            f"{self.model_name} does not support range-dependent bottom "
            f"geoacoustics; collapsed to single profile "
            f"(bottom_collapse_method={method!r}). "
            f"For RD bottoms use {alts['range_dependent_bottom']}."
        ), f"bottom_rd ({method})"

    def _collapse_rd_layered(self, env_in, e, method, alts):
        e.bottom = e.bottom_rd_layered.collapse(
            layered_method=method, range_method='middle',
        )
        e.bottom.depth = e.depth
        e.bottom_rd_layered = None
        return (
            f"{self.model_name} does not support range-dependent layered "
            f"bottoms; collapsed to single boundary "
            f"(rd_layered_collapse_method={method!r}). "
            f"For RD-layered use "
            f"{alts['range_dependent_layered_bottom']}."
        ), f"bottom_rd_layered ({method})"

    def _collapse_layered(self, env_in, e, method, alts):
        e.bottom = e.bottom_layered.collapse(method)
        e.bottom.depth = e.depth
        e.bottom_layered = None
        return (
            f"{self.model_name} does not support layered (depth-"
            f"dependent) bottoms; collapsed to single boundary "
            f"(layered_collapse_method={method!r}). "
            f"For layered bottoms use {alts['layered_bottom']}."
        ), f"layered bottom ({method})"

    @property
    def _PROJECTION_AXES(self):
        # (alt_key, support_flag_attr, method_attr, detector, collapser)
        return (
            ('altimetry', '_supports_altimetry',
             'altimetry_collapse_method',
             self._detect_altimetry, self._collapse_altimetry),
            ('range_dependent_bathymetry',
             '_supports_range_dependent_bathymetry',
             'bathymetry_collapse_method',
             self._detect_rd_bathy, self._collapse_rd_bathy),
            ('range_dependent_ssp', '_supports_range_dependent_ssp',
             'ssp_collapse_method',
             self._detect_rd_ssp, self._collapse_rd_ssp),
            ('range_dependent_bottom',
             '_supports_range_dependent_bottom',
             'bottom_collapse_method',
             self._detect_rd_bottom, self._collapse_rd_bottom),
            ('range_dependent_layered_bottom',
             '_supports_range_dependent_layered_bottom',
             'rd_layered_collapse_method',
             self._detect_rd_layered, self._collapse_rd_layered),
            ('layered_bottom', '_supports_layered_bottom',
             'layered_collapse_method',
             self._detect_layered, self._collapse_layered),
        )

    def _project_environment(self, env: 'Environment') -> 'Environment':
        """Return a copy of ``env`` with every unsupported feature collapsed.

        Walks the feature axes and applies the per-feature
        ``*_collapse_method`` configured on the model. Emits one
        ``UserWarning`` per dropped feature, citing the chosen method and
        the alternative-model hint from
        ``self._unsupported_env_alternatives``.

        Models call this once at the top of ``run()`` instead of
        inspecting the env themselves.
        """
        e = env.copy()
        alts = self._unsupported_env_alternatives

        for _key, support_attr, method_attr, detector, collapser in (
            self._PROJECTION_AXES
        ):
            if detector(env) and not getattr(self, support_attr):
                method = getattr(self, method_attr)
                msg, log_label = collapser(env, e, method, alts)
                warnings.warn(msg, UserWarning, stacklevel=3)
                self._log(
                    f"{self.model_name}: {log_label} dropped/collapsed.",
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
                warnings.warn(
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
            tail = prt.read_text()[-n_chars:]
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
        scalar; ``None`` is preserved for time-domain results.
        """
        md = dict(extra)
        if phase_reference is not None:
            md['phase_reference'] = phase_reference
        return dict(
            model=self.model_name,
            backend=backend or self.model_name,
            source_depths=np.atleast_1d(np.asarray(
                getattr(source, 'depths', []), dtype=float
            )),
            frequencies=(np.atleast_1d(np.asarray(frequencies, dtype=float))
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
