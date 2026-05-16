"""
OASES - Ocean Acoustics and Seismic Exploration Synthesis

OASES is a comprehensive acoustic/seismic propagation modeling suite developed
by Henrik Schmidt at MIT. It includes multiple executables for different scenarios:

- **OAST**: Transmission Loss via wavenumber integration
- **OASN**: Noise covariance matrices and signal replicas (matched-field processing)
- **OASR**: Reflection coefficients at stratified interfaces
- **OASP**: Broadband transfer-function / pulse synthesis (wideband wavenumber integration)

This module provides Python wrappers for all OASES executables following
the UACPY propagation model architecture.

Usage
-----
```python
from uacpy.models import OAST, OASN, OASR, OASP

# Transmission loss using OAST
oast = OAST()
result = oast.run(env, source, receiver)

# Normal modes using OASN
oasn = OASN()
modes = oasn.run(env, source, receiver)

# Reflection coefficients using OASR
oasr = OASR()
refl = oasr.run(env, source, receiver, angles=np.linspace(0, 90, 100))

# Parabolic equation using OASP
oasp = OASP()
result = oasp.run(env, source, receiver)
```
"""

import warnings

import numpy as np
from pathlib import Path
from typing import Optional, Union

from uacpy.models.base import PropagationModel, RunMode
from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.results import (
    Result, Field,
    Covariance, Replicas, ReflectionCoefficient,
)
from uacpy.core.exceptions import (
    ConfigurationError, ExecutableNotFoundError, ModelExecutionError,
    UnsupportedFeatureError,
)
from uacpy.io.oases_writer import write_oast_input, write_oasn_input, write_oasp_input, write_oasr_input
from uacpy.io.units import m_to_km
from uacpy.io.oases_reader import (
    read_oast_tl,
    read_oasn_covariance,
    read_oasn_replicas,
    read_oasp_trf,
    read_oasr_reflection_coefficients,
)


def _oases_resample_frequencies(
    freqs: np.ndarray, model_name: str,
) -> tuple[float, float, int, bool]:
    """Convert an arbitrary user ``frequencies=`` vector to the
    ``(fmin, fmax, N)`` triple OASR/OASP write into the input file.

    OASR's ``.dat`` and OASP's broadband kernel both express their
    frequency axis as a min/max/count, which implies equispaced
    sampling. If the user passes a non-equispaced vector we warn and
    auto-resample onto ``np.linspace(fmin, fmax, N)`` so the result's
    ``Result.frequencies`` reflects what the model actually saw.

    Returns
    -------
    fmin, fmax : float
    n : int
        Number of equispaced bins.
    resampled : bool
        True iff the user vector was non-equispaced and got resampled.
    """
    freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
    fmin = float(freqs.min())
    fmax = float(freqs.max())
    n = int(freqs.size)
    resampled = False
    if n > 1:
        diffs = np.diff(freqs)
        # Equispaced if all diffs match the mean diff within tolerance
        # (rtol scaled to the band; atol is a tiny absolute floor).
        target = (fmax - fmin) / (n - 1)
        if not np.allclose(diffs, target, rtol=1e-6, atol=1e-9):
            resampled = True
            warnings.warn(
                f"{model_name}: frequencies= vector is non-equispaced; "
                f"OASES expresses the frequency axis as (fmin, fmax, N) "
                f"so the vector has been resampled onto "
                f"np.linspace({fmin}, {fmax}, {n}). "
                f"Result.frequencies will be the resampled grid, not "
                f"your input. Pass an equispaced vector (or "
                f"freq_min/freq_max/n_frequencies) to suppress.",
                UserWarning, stacklevel=3,
            )
    return fmin, fmax, n, resampled


def _stack_oasr_data(data: dict):
    """Convert per-frequency lists from :func:`read_oasr_reflection_coefficients`
    into typed-Result arrays.

    Returns
    -------
    theta : ndarray, shape ``(n_angles,)``
    R : ndarray, shape ``(n_angles,)`` (single freq) or ``(n_angles, n_frequencies)``
    phi : ndarray, same shape as ``R``  — phase in radians (reader stores degrees)
    freqs : ndarray, shape ``(n_frequencies,)``
    """
    freqs = np.asarray(data.get('frequencies', []), dtype=float)
    angle_lists = data.get('angles_or_slowness', [])
    R_lists = data.get('magnitude', [])
    phi_lists = data.get('phase', [])
    if not angle_lists:
        raise ConfigurationError("OASR reader returned no frequency samples")
    theta = np.asarray(angle_lists[0], dtype=float)
    n_angles = len(theta)
    n_frequencies = len(freqs)
    for i, (a, m, p) in enumerate(zip(angle_lists, R_lists, phi_lists)):
        if len(a) != n_angles:
            raise ConfigurationError(
                f"OASR: angle grid mismatch between frequencies "
                f"(freq[{i}] has {len(a)} angles, freq[0] has {n_angles}). "
                f"Multi-frequency stacking requires a shared angle grid."
            )
        if len(m) != n_angles or len(p) != n_angles:
            raise ConfigurationError(
                f"OASR: payload length mismatch at frequency index {i}: "
                f"angles={len(a)}, magnitude={len(m)}, phase={len(p)}"
            )
    if n_frequencies == 1:
        R = np.asarray(R_lists[0], dtype=float)
        phi_deg = np.asarray(phi_lists[0], dtype=float)
    else:
        R = np.column_stack([np.asarray(m, dtype=float) for m in R_lists])
        phi_deg = np.column_stack([np.asarray(p, dtype=float) for p in phi_lists])
    return theta, R, np.deg2rad(phi_deg), freqs


def _oasn_freq_axis(data: dict) -> np.ndarray:
    """Reconstruct the frequency vector from an OASN reader payload.

    The OASN ``.xsm`` / ``.rpo`` headers carry ``freq_min``, ``freq_max``,
    ``n_frequencies`` (and a ``freq_delta`` cross-check). We rebuild the
    axis with ``np.linspace`` since the spacing OASN advertises is
    equispaced between FREQ1 and FREQ2.
    """
    n = int(data.get('n_frequencies', 1))
    f1 = float(data.get('freq_min', 0.0))
    f2 = float(data.get('freq_max', f1))
    if n <= 1:
        return np.array([f1], dtype=float)
    return np.linspace(f1, f2, n, dtype=float)


def _oases_find_executable(model: PropagationModel, name: str) -> Path:
    """Locate an OASES binary, preferring the ``<name>_bash`` wrapper.

    Searches ``uacpy/bin/oases``, ``uacpy/bin/oalib``, and
    ``uacpy/third_party/oases/bin``.
    """
    return model._find_executable_in_paths(
        [f'{name}_bash', name],
        bin_subdirs=['oases', 'oalib'],
        dev_subdir='oases',
    )


def _oases_subprocess_env(base_name: str, **extras: str) -> dict:
    """Build the FORnnn env-var dict the OASES csh wrappers set.

    Common keys (FOR001 input, FOR019/FOR020 plot files, FOR028/FOR029/
    FOR045 scratch) are populated for every sub-model; ``extras`` supplies
    per-binary unit numbers (e.g. FOR002='src' for OAST, FOR016='xsm' for
    OASN) — pass the suffix without the ``base_name + '.'`` prefix.
    ``base_name`` is the stem of the input file (no extension).
    """
    import os
    env = os.environ.copy()
    env['FOR001'] = f'{base_name}.dat'
    env['FOR019'] = f'{base_name}.plp'
    env['FOR020'] = f'{base_name}.plt'
    env['FOR028'] = f'{base_name}.028'
    env['FOR029'] = f'{base_name}.029'
    env['FOR045'] = f'{base_name}.045'
    for key, suffix in extras.items():
        env[key] = f'{base_name}.{suffix}'
    return env


class OAST(PropagationModel):
    """
    OAST - OASES Transmission Loss Model

    Computes transmission loss using wavenumber integration. Best for
    range-independent environments with depth-dependent sound speed profiles.

    Parameters
    ----------
    executable : Path, optional
        Path to OAST binary. Auto-detected if ``None``.
    compute_contour : bool, optional
        Add ``'C'`` option (range-depth contour plot). Default ``False``.
    compute_depth_average : bool, optional
        Add ``'A'`` option (depth-averaged TL). Default ``False``.
    complex_contour : bool, optional
        ``'J'`` option (complex integration contour). Default ``True``.
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    Range-independent wavenumber integration; consumes ``LayeredBottom``
    natively. **Collapse defaults (overrides of :data:`DEFAULT_COLLAPSE`).**
    Per-model: ``'ssp': 'mean'``, ``'bottom': 'median'``,
    ``'rd_layered_layers': 'preserve'``.

    Examples
    --------
    >>> from uacpy.models import OAST
    >>> oast = OAST()
    >>> result = oast.run(env, source, receiver)
    """

    def __init__(
        self,
        executable: Optional[Path] = None,
        compute_contour: bool = False,
        compute_depth_average: bool = False,
        complex_contour: bool = True,
        options: Optional[str] = None,
        integration_offset: float = 0.0,
        nw_samples: int = -1,
        plot_rmin: Optional[float] = None,
        plot_rmax: Optional[float] = None,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        **kwargs,
    ):
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            **kwargs,
        )
        self.compute_contour = bool(compute_contour)
        self.compute_depth_average = bool(compute_depth_average)
        self.complex_contour = bool(complex_contour)
        # Raw OASES option string (e.g. ``'N J T C'``). ``None`` lets the
        # wrapper derive it from compute_contour / compute_depth_average /
        # complex_contour.
        self.options = options
        self.integration_offset = float(integration_offset)
        # ``-1`` lets the OASES kernel pick its own wavenumber sample count.
        self.nw_samples = int(nw_samples)
        # Plot-axis bounds in metres. ``None`` → 0 / receiver.range_max.
        self.plot_rmin = float(plot_rmin) if plot_rmin is not None else None
        self.plot_rmax = float(plot_rmax) if plot_rmax is not None else None

        self._supported_modes = [RunMode.COHERENT_TL]
        # OAST: range-independent wavenumber integration; multi-layer
        # fluid + elastic bottom honored.
        self._supports_altimetry = False
        self._supports_range_dependent_bathymetry = False
        self._supports_range_dependent_ssp = False
        self._supports_range_dependent_bottom = False
        self._supports_layered_bottom = True
        self._supports_range_dependent_layered_bottom = False
        self._supports_elastic_media = True
        self._supports_multi_source_depth = False
        # Single spectral solve over the full wavenumber axis;
        # median/mean samples represent the path the field describes.
        self._set_collapse_defaults({
            'ssp': 'mean',
            'bottom': 'median',
            'rd_layered_layers': 'preserve',
        })
        if executable is None:
            self.executable = _oases_find_executable(self, 'oast')
        else:
            self.executable = Path(executable)

        if not self.executable.exists():
            raise ExecutableNotFoundError('OAST', str(self.executable))

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[RunMode] = None,
    ) -> Result:
        """
        Run OAST transmission loss computation

        Parameters
        ----------
        env : Environment
            Ocean environment (must be range-independent)
        source : Source
            Acoustic source
        receiver : Receiver
            Receiver array

        Returns
        -------
        result : Result
            Transmission loss field
        """
        run_mode = self._resolve_run_mode(run_mode)

        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)
        fm = self._setup_file_manager()

        try:
            base_name = 'oast_run'
            input_file = fm.get_path(f'{base_name}.dat')

            # OAST option letters: raw ``self.options`` wins; otherwise
            # derive from the typed compute_contour / compute_depth_average
            # / complex_contour flags.
            if self.options is not None:
                user_options = self.options
            else:
                opt = ['N', 'T']                            # Normal stress + TL table
                if self.complex_contour:
                    opt.append('J')
                if self.compute_contour:
                    opt.append('C')
                if self.compute_depth_average:
                    opt.append('A')
                user_options = ' '.join(opt)

            self._log(f"Writing OAST input file: {input_file} (options={user_options})")
            writer_kwargs: dict = {
                'integration_offset': self.integration_offset,
                'nw_samples': self.nw_samples,
            }
            if self.plot_rmin is not None:
                writer_kwargs['plot_rmin'] = self.plot_rmin
            if self.plot_rmax is not None:
                writer_kwargs['plot_rmax'] = self.plot_rmax
            write_oast_input(
                filepath=input_file,
                env=env,
                source=source,
                receiver=receiver,
                options=user_options,
                **writer_kwargs,
            )

            self._execute(input_file, fm.work_dir)

            # Read output - OAST creates .plt (data) and .plp (metadata) files
            # Per OASES documentation: FOR019 -> .plp, FOR020 -> .plt
            plt_file = fm.get_path(f'{base_name}.plt')

            # Check if output files exist
            if not plt_file.exists():
                exc = ModelExecutionError(
                    self.model_name, return_code=0, stdout=None,
                    stderr=(
                        f"OAST did not produce {plt_file} (FOR020 .plt). "
                        f"Check {fm.work_dir}/{base_name}.prt and the "
                        "examples in third_party/oases/tloss/."
                    ),
                )
                self._attach_prt_tail(exc, fm.work_dir, base_name)
                raise exc

            output_file = plt_file

            self._log(f"Reading OAST output: {output_file}")
            tl_data, native_depths, native_ranges, metadata = read_oast_tl(
                filepath=output_file,
                receiver_depths=receiver.depths,
            )

            kw = self._result_kwargs(
                source,
                backend='oast',
                frequencies=float(np.atleast_1d(source.frequencies)[0]),
            )
            kw['metadata'].update(metadata)
            native = Field(
                data=tl_data,
                coords={'depth': native_depths, 'range': native_ranges},
                **kw,
            )
            receiver_ranges = np.atleast_1d(np.asarray(receiver.ranges, dtype=float))
            receiver_depths = np.atleast_1d(np.asarray(receiver.depths, dtype=float))
            ranges_match = (
                len(native_ranges) == len(receiver_ranges)
                and np.allclose(native_ranges, receiver_ranges)
            )
            if ranges_match:
                result = native
            else:
                result = native.resample_to(receiver_ranges, receiver_depths)
                result.metadata['oast_native_ranges'] = native_ranges
                result.metadata['interpolated'] = True

            self._attach_output_paths(
                result, fm.work_dir, base_name,
                primary_files=(('plt_file', '.plt'),),
            )

            self._log("OAST simulation complete")
            return result

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _execute(self, input_file: Path, work_dir: Path):
        """Execute OAST binary. FOR005 stays as stdin per OASES docs."""
        env = _oases_subprocess_env(
            input_file.stem,
            FOR002='src',  # Source file
            FOR023='trc',  # Optional reflection-coef table
        )
        try:
            result = self._run_subprocess(
                [str(self.executable)], cwd=work_dir, env=env,
            )
        except ModelExecutionError as exc:
            self._attach_prt_tail(exc, work_dir, input_file.stem)
            raise
        if self.verbose and result.stdout:
            self._log(f"OASES output:\n{result.stdout}", level='debug')


class OASN(PropagationModel):
    """
    OASN — OASES Noise, Covariance Matrices and Signal Replicas.

    Per the OASES manual (``third_party/oases/doc/oasn.tex``) OASN produces
    two frequency-domain array products:

    - **Covariance matrices** (option ``N`` → ``.xsm``): hydrophone ×
      hydrophone correlation per frequency. Used for ambient-noise
      characterisation or as an MFP measurement covariance.
    - **Replica fields** (option ``R`` → ``.rpo``): array response per
      candidate source position per frequency. Used as MFP templates.

    For depth-eigenfunction normal modes use :class:`Kraken` or
    :class:`KrakenC`.

    Parameters
    ----------
    executable : Path, optional
        Path to OASN binary. Auto-detected if ``None``.
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    Supported run modes: ``RunMode.COVARIANCE`` (``.xsm`` →
    :class:`Covariance`) and ``RunMode.REPLICA`` (``.rpo`` →
    :class:`Replicas`). The convenience methods
    ``compute_covariance(...)`` / ``compute_replicas(...)`` route to
    these.

    **Collapse defaults (overrides of :data:`DEFAULT_COLLAPSE`).**
    Per-model: ``'ssp': 'mean'``, ``'bottom': 'median'``,
    ``'rd_layered_layers': 'preserve'``.

    Examples
    --------
    >>> from uacpy.models import OASN
    >>> oasn = OASN()
    >>> cov = oasn.compute_covariance(env, source, receiver)
    >>> # cov.covariance has shape (n_frequencies, n_rcv, n_rcv)
    """

    def __init__(
        self,
        executable: Optional[Path] = None,
        # Output / option control
        options: Optional[str] = None,
        # Noise field (Block VIII): broad-area sources expressed as
        # spectral levels (dB re 1 µPa²/Hz) at three depths.
        surface_noise_level: float = 0.0,
        white_noise_level: float = 0.0,
        deep_noise_level: float = 0.0,
        deep_source_depth: Optional[float] = None,
        # Discrete (point) sources — list of dicts; each may carry
        # 'depth' (m), 'x' (m), 'y' (m), 'level' (dB), 'phase' (rad).
        discrete_sources: Optional[list] = None,
        # Replica candidate-position grid (Block X). x/y/z in metres on
        # the public API, converted to km at write time. ``None`` lets
        # the writer apply OASES defaults (10 / depth-10 in z, 100 /
        # 10000 in x, 0 / 0 in y).
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
        nx: int = 50,
        ymin: Optional[float] = None,
        ymax: Optional[float] = None,
        ny: int = 1,
        zmin: Optional[float] = None,
        zmax: Optional[float] = None,
        nz: int = 20,
        # Phase-speed bounds for the wavenumber integrations (m/s).
        # Applied identically to OASN Block VIII (noise / discrete
        # sources) and Block X (replica generator). ``None`` → writer
        # derives c_water_min * 0.95 (cmin) and 1e8 (cmax).
        cmin: Optional[float] = None,
        cmax: Optional[float] = None,
        # Wavenumber-axis sampling & TL plot axes.
        integration_offset: float = 0.0,
        nw_samples: int = -1,
        plot_rmin: Optional[float] = None,
        plot_rmax: Optional[float] = None,
        # Frequency-line extras: vrec (m/s, vertical receiver velocity
        # for Doppler), offdb (single-mode horizontal offset).
        vrec: float = 0.0,
        offdb: Optional[float] = None,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        **kwargs,
    ):
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir, **kwargs,
        )
        self.options = options
        self.surface_noise_level = float(surface_noise_level)
        self.white_noise_level = float(white_noise_level)
        self.deep_noise_level = float(deep_noise_level)
        self.deep_source_depth = (
            float(deep_source_depth) if deep_source_depth is not None else None
        )
        self.discrete_sources = list(discrete_sources) if discrete_sources else None
        self.xmin = xmin
        self.xmax = xmax
        self.nx = int(nx)
        self.ymin = ymin
        self.ymax = ymax
        self.ny = int(ny)
        self.zmin = zmin
        self.zmax = zmax
        self.nz = int(nz)
        self.cmin = cmin
        self.cmax = cmax
        self.integration_offset = float(integration_offset)
        self.nw_samples = int(nw_samples)
        self.plot_rmin = float(plot_rmin) if plot_rmin is not None else None
        self.plot_rmax = float(plot_rmax) if plot_rmax is not None else None
        self.vrec = float(vrec)
        self.offdb = float(offdb) if offdb is not None else None

        self._supported_modes = [RunMode.COVARIANCE, RunMode.REPLICA]
        # OASN: range-independent covariance / replica field; multi-layer
        # bottom honored.
        self._supports_altimetry = False
        self._supports_range_dependent_bathymetry = False
        self._supports_range_dependent_ssp = False
        self._supports_range_dependent_bottom = False
        self._supports_layered_bottom = True
        self._supports_range_dependent_layered_bottom = False
        self._supports_elastic_media = True
        self._supports_multi_source_depth = False
        # Single spectral solve at each frequency; median/mean samples
        # represent the path the array sees.
        self._set_collapse_defaults({
            'ssp': 'mean',
            'bottom': 'median',
            'rd_layered_layers': 'preserve',
        })
        if executable is None:
            self.executable = _oases_find_executable(self, 'oasn2_bin')
        else:
            self.executable = Path(executable)

        if not self.executable.exists():
            raise ExecutableNotFoundError('OASN', str(self.executable))

    def _build_writer_kwargs(self, run_mode: 'RunMode') -> dict:
        """Materialise the writer's expected kwarg dict from ``self.*``.

        Translates uacpy's public API (singular names, metres) to the
        OASES file format the writer consumes (plural names for
        cmins/cmaxs, km for x/y axes). Only keys whose stored value is
        non-default are emitted, so the writer falls back to its own
        defaults for everything the user did not set.
        """
        kw: dict = {
            'integration_offset': self.integration_offset,
            'nw_samples': self.nw_samples,
            'surface_noise_level': self.surface_noise_level,
            'white_noise_level': self.white_noise_level,
            'deep_noise_level': self.deep_noise_level,
            'vrec': self.vrec,
        }
        if self.deep_source_depth is not None:
            kw['deep_source_depth'] = self.deep_source_depth
        if self.offdb is not None:
            kw['offdb'] = self.offdb
        if self.plot_rmin is not None:
            kw['plot_rmin'] = self.plot_rmin
        if self.plot_rmax is not None:
            kw['plot_rmax'] = self.plot_rmax

        # Phase-speed bounds applied identically to OASN's noise and
        # replica integrations (singular on the public API, plural-with-
        # suffix in the writer per OASES's CMINS/CMAXS variable names).
        if self.cmin is not None:
            kw['cmins_discrete'] = self.cmin
            kw['cmins_replica'] = self.cmin
        if self.cmax is not None:
            kw['cmaxs_discrete'] = self.cmax
            kw['cmaxs_replica'] = self.cmax

        # Replica grid x/y in km on disk; z in metres. nx/ny/nz always
        # forwarded so the writer reuses uacpy defaults rather than its
        # own OASES-doc ones when the user did not override.
        kw['replica_nx'] = self.nx
        kw['replica_ny'] = self.ny
        kw['replica_nz'] = self.nz
        if self.xmin is not None:
            kw['replica_xmin'] = float(m_to_km(self.xmin))
        if self.xmax is not None:
            kw['replica_xmax'] = float(m_to_km(self.xmax))
        if self.ymin is not None:
            kw['replica_ymin'] = float(m_to_km(self.ymin))
        if self.ymax is not None:
            kw['replica_ymax'] = float(m_to_km(self.ymax))
        if self.zmin is not None:
            kw['replica_zmin'] = self.zmin
        if self.zmax is not None:
            kw['replica_zmax'] = self.zmax

        # ``discrete_sources`` carry per-source x/y in metres (uacpy
        # API); OASES writes them in km (oasn.tex:343).
        if self.discrete_sources:
            converted = []
            for ds in self.discrete_sources:
                ds_km = dict(ds)
                if 'x' in ds_km:
                    ds_km['x'] = float(m_to_km(ds_km['x']))
                if 'y' in ds_km:
                    ds_km['y'] = float(m_to_km(ds_km['y']))
                converted.append(ds_km)
            kw['discrete_sources'] = converted

        return kw

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[RunMode] = None,
    ) -> Result:
        """
        Run OASN.

        Parameters
        ----------
        run_mode : RunMode, optional
            ``RunMode.COVARIANCE`` (default) → :class:`Covariance`.
            ``RunMode.REPLICA`` → :class:`Replicas` (build the OASN
            instance with ``xmin``/``xmax``/``zmin``/``zmax``/… already
            set on the constructor; sweeps go through ``model.copy(...)``).

        Returns
        -------
        Covariance or Replicas
        """
        run_mode = self._resolve_run_mode(run_mode)

        # OASN expresses the frequency axis as (fmin, fmax, N) like OASR
        # and OASP, so a non-equispaced source.frequencies vector would
        # be silently regridded by _resolve_freq_sweep inside the writer.
        # Warn (and resample) here so the user sees what is happening.
        source_freqs = np.atleast_1d(np.asarray(source.frequencies, dtype=float))
        if source_freqs.size > 1:
            _oases_resample_frequencies(source_freqs, self.model_name)

        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)

        writer_kwargs = self._build_writer_kwargs(run_mode)

        # The OASN options block must include the file output the run
        # mode produces. The user-supplied options (or default 'J') win
        # for additional letters (F, …); N (covariance) or R (replica)
        # is added automatically based on run_mode.
        user_options = self.options or 'J'
        opt_tokens = set(user_options.split())
        if run_mode == RunMode.COVARIANCE:
            opt_tokens.add('N')
        else:
            # Replica generation reuses OASN's noise-integration kernel,
            # so 'N' must accompany 'R' even when the user only wants
            # the replica fields.
            opt_tokens.add('R')
            opt_tokens.add('N')
        options = ' '.join(sorted(opt_tokens))

        fm = self._setup_file_manager()
        try:
            base_name = 'oasn_run'
            input_file = fm.get_path(f'{base_name}.dat')
            self._log(f"Writing OASN input file: {input_file} (options={options})")
            write_oasn_input(
                filepath=input_file,
                env=env,
                source=source,
                receiver=receiver,
                options=options,
                **writer_kwargs,
            )
            self._execute(base_name, fm.work_dir)

            if run_mode == RunMode.COVARIANCE:
                xsm_file = fm.get_path(f'{base_name}.xsm')
                fort16_file = fm.get_path('fort.16')
                cov_path = xsm_file if xsm_file.exists() else fort16_file
                if not cov_path.exists():
                    exc = ModelExecutionError(
                        self.model_name, return_code=0, stdout=None,
                        stderr=(
                            f"OASN did not produce a covariance file. "
                            f"Checked: {xsm_file}, {fort16_file}. "
                            f"Inspect {fm.work_dir}/{base_name}.prt."
                        ),
                    )
                    self._attach_prt_tail(exc, fm.work_dir, base_name)
                    raise exc
                self._log(f"Reading OASN covariance file: {cov_path}")
                cov_data = read_oasn_covariance(cov_path)

                rcv_pos = None
                rcv_depths = np.atleast_1d(np.asarray(receiver.depths, dtype=float))
                if rcv_depths.size == cov_data['covariance'].shape[1]:
                    # OASES OASN config places receivers along the array;
                    # only depth varies in the typical uacpy usage.
                    rcv_pos = np.zeros((rcv_depths.size, 3), dtype=float)
                    rcv_pos[:, 2] = rcv_depths

                freqs = _oasn_freq_axis(cov_data)
                kw = self._result_kwargs(
                    source,
                    backend='oasn',
                    frequencies=freqs,
                    n_receivers=cov_data['n_receivers'],
                    title=cov_data.get('title', ''),
                )
                cov_result = Covariance(
                    covariance=cov_data['covariance'],
                    receiver_positions=rcv_pos,
                    **kw,
                )
                self._attach_output_paths(
                    cov_result, fm.work_dir, base_name,
                    primary_files=(('xsm_file', '.xsm'),),
                )
                return cov_result

            # RunMode.REPLICA
            rpo_file = fm.get_path(f'{base_name}.rpo')
            fort14_file = fm.get_path('fort.14')
            rep_path = rpo_file if rpo_file.exists() else fort14_file
            if not rep_path.exists():
                exc = ModelExecutionError(
                    self.model_name, return_code=0, stdout=None,
                    stderr=(
                        f"OASN did not produce a replica file. Checked: "
                        f"{rpo_file}, {fort14_file}. Set the replica grid "
                        "via OASN(xmin=…, xmax=…, nx=…, zmin=…, zmax=…, "
                        "nz=…) on the constructor."
                    ),
                )
                self._attach_prt_tail(exc, fm.work_dir, base_name)
                raise exc
            self._log(f"Reading OASN replica file: {rep_path}")
            rep_data = read_oasn_replicas(rep_path)
            # x/y come back in km from the .rpo file; expose metres.
            replica_z = np.linspace(rep_data['z_min'], rep_data['z_max'], rep_data['n_z'])
            replica_x = np.linspace(rep_data['x_min'], rep_data['x_max'], rep_data['n_x']) * 1000.0
            replica_y = np.linspace(rep_data['y_min'], rep_data['y_max'], rep_data['n_y']) * 1000.0
            freqs = _oasn_freq_axis(rep_data)
            kw = self._result_kwargs(
                source,
                backend='oasn',
                frequencies=freqs,
                n_receivers=rep_data['n_receivers'],
                title=rep_data.get('title', ''),
            )
            rep_result = Replicas(
                replicas=rep_data['replicas'],
                replica_z=replica_z,
                replica_x=replica_x,
                replica_y=replica_y,
                receiver_positions=rep_data.get('receiver_positions'),
                **kw,
            )
            self._attach_output_paths(
                rep_result, fm.work_dir, base_name,
                primary_files=(('rpo_file', '.rpo'),),
            )
            return rep_result

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    # ``compute_covariance`` / ``compute_replicas`` come from the base class
    # (RunMode.COVARIANCE / RunMode.REPLICA dispatch).

    def _execute(self, base_name: str, work_dir: Path):
        """Execute OASN binary. FOR005 stays as stdin per OASES docs."""
        env = _oases_subprocess_env(
            base_name,
            FOR014='rpo',  # Replica vectors (unit 14)
            FOR016='xsm',  # Covariance matrix (unit 16)
            FOR026='chk',  # Checkpoint file (per oasn wrapper)
        )
        try:
            result = self._run_subprocess(
                [str(self.executable)], cwd=work_dir, env=env,
            )
        except ModelExecutionError as exc:
            self._attach_prt_tail(exc, work_dir, base_name)
            raise
        if self.verbose and result.stdout:
            self._log(f"OASES output:\n{result.stdout}", level='debug')


class OASR(PropagationModel):
    """
    OASR - OASES Reflection Coefficients Model

    Computes plane wave reflection coefficients at the bottom interface.

    Parameters
    ----------
    executable : Path, optional
        Path to OASR binary. Auto-detected if ``None``.
    angles : ndarray, optional
        Angle grid (deg). Default ``linspace(0, 90, 181)``.
    angle_type : str, optional
        ``'grazing'`` (OASES native, default) | ``'incidence'``
        (converted via ``grazing = 90 - incidence``).
    reflection_type : str, optional
        ``'P-P'`` (default) | ``'P-SV'`` | ``'P-Slow'`` (Biot only) |
        ``'transmission'``. Translates to OASR option letter
        (``'N'`` / ``'S'`` / ``'B'`` / ``'t'``).
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    Range-independent reflection-vs-angle/freq solver. Supports
    elastic / poro-elastic bottom layers. Returns reflection magnitude
    and phase per ``(frequency, angle)``. Pass ``freq_min``,
    ``freq_max``, ``n_frequencies`` on ``run()`` for a broadband sweep.

    **Collapse defaults (overrides of :data:`DEFAULT_COLLAPSE`).**
    Per-model: ``'bottom': 'median'``, ``'rd_layered_layers':
    'preserve'``. SSP collapse left at the global ``'r0'`` because the
    SSP boundary speed is essentially irrelevant to the reflection
    coefficient.

    Examples
    --------
    >>> from uacpy.models import OASR
    >>> oasr = OASR(angles=np.linspace(0, 90, 100))
    >>> refl = oasr.run(env, source, receiver)
    """

    def __init__(
        self,
        executable: Optional[Path] = None,
        angles: Optional[np.ndarray] = None,
        angle_type: str = 'grazing',
        reflection_type: str = 'P-P',
        options: Optional[str] = None,
        angle_output_increment: Optional[int] = None,
        interface_roughness: Optional[list] = None,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        executable : Path, optional
            Path to OASR executable. Auto-detected if None.
        angles : ndarray, optional
            Angle grid in degrees. Default: ``np.linspace(0, 90, 181)``.
            Pass a non-uniform ndarray for adaptive sampling.
        angle_type : str, optional
            'grazing' (OASES native) or 'incidence' (converted via
            ``grazing = 90 - incidence`` before being written to the input
            file). Default: 'grazing'.
        reflection_type : str, optional
            One of 'P-P' (default), 'P-SV', 'P-Slow' (Biot only), or
            'transmission'. Translates to the OASR option letter
            ('N' / 'S' / 'B' / 't').
        options : str, optional
            Raw OASES option string. ``None`` lets the wrapper derive
            it from ``reflection_type``.
        angle_output_increment : int, optional
            Decimation factor for the output angle table. ``None`` keeps
            every sample.
        interface_roughness : list, optional
            Per-interface RMS roughness in metres (one entry per layer
            interface, ordered top → bottom).
        """
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir, **kwargs,
        )
        self.angles = (
            np.asarray(angles, dtype=float) if angles is not None else None
        )
        self.angle_type = angle_type
        self.reflection_type = reflection_type
        self.options = options
        self.angle_output_increment = (
            int(angle_output_increment)
            if angle_output_increment is not None else None
        )
        self.interface_roughness = (
            list(interface_roughness) if interface_roughness else None
        )

        # OASR is strictly a boundary-reflection solver; it does not produce
        # transmission loss. Declare that explicitly.
        self._supported_modes = [RunMode.REFLECTION]
        # OASR: range-independent reflection vs angle/freq; layered bottom honored.
        self._supports_altimetry = False
        self._supports_range_dependent_bathymetry = False
        self._supports_range_dependent_ssp = False
        self._supports_range_dependent_bottom = False
        self._supports_layered_bottom = True
        self._supports_range_dependent_layered_bottom = False
        self._supports_elastic_media = True
        self._supports_multi_source_depth = False
        # Pure boundary reflection — only the bottom stack matters.
        # SSP collapse left at the global default ('r0') because the
        # SSP is essentially irrelevant to the reflection coefficient.
        self._set_collapse_defaults({
            'bottom': 'median',
            'rd_layered_layers': 'preserve',
        })
        if executable is None:
            self.executable = _oases_find_executable(self, 'oasr')
        else:
            self.executable = Path(executable)

        if not self.executable.exists():
            raise ExecutableNotFoundError('OASR', str(self.executable))

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[RunMode] = None,
        *,
        frequencies: Optional[np.ndarray] = None,
    ) -> Result:
        """Run OASR.

        Returns a :class:`ReflectionCoefficient` with ``theta`` (1-D angle
        grid in degrees) and ``R``/``phi`` shaped ``(n_angles, n_frequencies)``
        when more than one frequency was requested. Single-frequency runs
        return 1-D ``R``/``phi``.

        OASES uses grazing angles natively. When ``angle_type='incidence'``,
        input angles are converted via ``grazing = 90 - incidence`` before
        being written to the OASR input file.

        Parameters
        ----------
        frequencies : ndarray, optional
            Explicit frequency vector (Hz). When provided, overrides
            ``source.frequencies``. OASES treats the sweep as
            equispaced — uacpy resamples and warns if your vector is not.
        """
        run_mode = self._resolve_run_mode(run_mode)

        angles = self.angles if self.angles is not None else np.linspace(0, 90, 181)

        writer_kwargs: dict = {}
        if frequencies is not None:
            freqs_arr = np.atleast_1d(np.asarray(frequencies, dtype=float))
            if freqs_arr.size == 0:
                raise ConfigurationError(
                    "OASR.run(frequencies=…) requires at least one "
                    "positive frequency."
                )
            fmin, fmax, n_freq, _ = _oases_resample_frequencies(
                freqs_arr, 'OASR',
            )
            writer_kwargs['freq_min'] = fmin
            writer_kwargs['freq_max'] = fmax
            writer_kwargs['n_frequencies'] = n_freq

        if self.options is not None:
            writer_kwargs['options'] = self.options
        if self.angle_output_increment is not None:
            writer_kwargs['angle_output_increment'] = self.angle_output_increment
        if self.interface_roughness is not None:
            writer_kwargs['interface_roughness'] = self.interface_roughness

        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)

        fm = self._setup_file_manager()

        try:
            base_name = 'oasr_run'
            input_file = fm.get_path(f'{base_name}.dat')

            self._log(f"Writing OASR input file: {input_file}")
            write_oasr_input(
                filepath=input_file,
                env=env,
                source=source,
                receiver=receiver,
                angles=angles,
                angle_type=self.angle_type,
                reflection_type=self.reflection_type,
                **writer_kwargs,
            )

            self._execute(input_file, fm.work_dir)

            # OASR writes a grazing-angle table to .trc and a slowness
            # table to .rco. Both files may be present even though the
            # user only requested one — pick the one matching the
            # requested sampling. uacpy emits angles by default (no 'p'
            # option in the OASR options string), so .trc wins; the user
            # can pass options='p ...' to switch to slowness sampling, in
            # which case we prefer .rco.
            requested_opts = (self.options or '').split()
            wants_slowness = 'p' in requested_opts
            search = ['.rco', '.trc'] if wants_slowness else ['.trc', '.rco']
            search += ['.023', 'fort.023']

            output_file = None
            for ext in search:
                candidate = (
                    fm.get_path(f'{base_name}{ext}')
                    if not ext.startswith('fort')
                    else fm.get_path(ext)
                )
                if candidate.exists() and candidate.stat().st_size > 0:
                    output_file = candidate
                    break

            if output_file is None:
                exc = ModelExecutionError(
                    self.model_name, return_code=0, stdout=None,
                    stderr=f"OASR did not produce an output file. Checked: {search}",
                )
                self._attach_prt_tail(exc, fm.work_dir, base_name)
                raise exc

            self._log(f"Reading OASR output: {output_file}")
            data = read_oasr_reflection_coefficients(output_file)

            theta_arr, R_arr, phi_arr, freqs_arr = _stack_oasr_data(data)
            field = ReflectionCoefficient(
                theta=theta_arr, R=R_arr, phi=phi_arr,
                **self._result_kwargs(
                    source,
                    backend='oasr',
                    frequencies=freqs_arr if len(freqs_arr) else None,
                    sampling_type=data.get('sampling_type', 'angle'),
                    reflection_type=self.reflection_type,
                ),
            )
            self._attach_output_paths(
                field, fm.work_dir, base_name,
                primary_files=(
                    ('trc_file', '.trc'),
                    ('rco_file', '.rco'),
                ),
            )

            self._log("OASR simulation complete")
            return field

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _execute(self, input_file: Path, work_dir: Path):
        """Execute OASR binary. FOR005 stays as stdin per OASES docs."""
        env = _oases_subprocess_env(
            input_file.stem,
            FOR002='src',
            FOR004='trf',
            FOR022='rco',  # Reflection-coef table (slowness)
            FOR023='trc',  # Reflection-coef table (angle)
        )
        try:
            result = self._run_subprocess(
                [str(self.executable)], cwd=work_dir, env=env,
            )
        except ModelExecutionError as exc:
            self._attach_prt_tail(exc, work_dir, input_file.stem)
            raise
        if self.verbose and result.stdout:
            self._log(f"OASES output:\n{result.stdout}", level='debug')


class OASP(PropagationModel):
    """
    OASP - OASES Pulse / Broadband Transfer-Function Model

    Computes broadband acoustic transfer functions via wavenumber integration
    followed by FFT to produce time-series / pulse responses. OASP is the
    "Pulse" variant of SAFARI, using the same wavenumber-integration kernel
    as OAST but evaluated across a frequency sweep.

    Parameters
    ----------
    executable : Path, optional
        Path to OASP binary. Auto-detected if ``None``.
    n_time_samples : int, optional
        Number of FFT time samples. Default ``4096``.
    freq_max : float, optional
        Maximum FFT frequency (Hz). Default ``250.0``.
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    Broadband-oriented; supports ``RunMode.BROADBAND`` (returns a
    :class:`Field`) and ``RunMode.TIME_SERIES`` (returns a
    :class:`Field` after ``synthesize_time_series``). For a
    single-cell trace use ``tf.to_time_trace(depth, range)``. Can be
    expensive (reduce ``n_time_samples`` for speed). For range-dependent
    problems, RAM is recommended.

    **Collapse defaults (overrides of :data:`DEFAULT_COLLAPSE`).**
    Per-model: ``'ssp': 'mean'``, ``'bottom': 'median'``,
    ``'rd_layered_layers': 'preserve'``.

    Examples
    --------
    >>> from uacpy.models import OASP
    >>> oasp = OASP(n_time_samples=256, freq_max=120)
    >>> result = oasp.run(env, source, receiver)
    """

    def __init__(
        self,
        executable: Optional[Path] = None,
        n_time_samples: int = 4096,
        freq_max: float = 250.0,
        freq_min: float = 0.0,
        center_frequency: Optional[float] = None,
        freq_output_increment: Optional[int] = None,
        options: Optional[str] = None,
        range_start: Optional[float] = None,
        integration_offset: float = 0.0,
        nw_samples: int = -1,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        executable : Path, optional
            Path to OASP executable. Auto-detected if None.
        n_time_samples : int, optional
            Power-of-two FFT length (samples per receiver trace).
            Default 4096.
        freq_max : float, optional
            Upper edge of the OASP broadband sweep (Hz). Default 250.
        freq_min : float, optional
            Lower edge of the OASP broadband sweep (Hz). Default 0.0.
        center_frequency : float, optional
            Carrier frequency for the pulse (Hz). ``None`` defaults to
            ``source.frequencies[0]`` at ``run()`` time.
        freq_output_increment : int, optional
            Decimation factor for the .trf frequency axis. ``None`` →
            ``max(1, n_frequencies // 10)``.
        options : str, optional
            Raw OASES option string. ``None`` defers to the writer's
            default (``'N'``).
        range_start : float, optional
            First receiver range (m). ``None`` defaults to
            ``receiver.ranges.min()``.
        integration_offset : float, optional
            Wavenumber-contour offset (dB/wavelength). Default 0.
        nw_samples : int, optional
            Wavenumber sample count. ``-1`` lets OASP auto-pick.
        """
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            **kwargs,
        )
        self.n_time_samples = int(n_time_samples)
        self.freq_max = float(freq_max)
        self.freq_min = float(freq_min)
        self.center_frequency = (
            float(center_frequency) if center_frequency is not None else None
        )
        self.freq_output_increment = (
            int(freq_output_increment)
            if freq_output_increment is not None else None
        )
        self.options = options
        self.range_start = (
            float(range_start) if range_start is not None else None
        )
        self.integration_offset = float(integration_offset)
        self.nw_samples = int(nw_samples)

        self._supported_modes = [
            RunMode.COHERENT_TL,
            RunMode.BROADBAND,
            RunMode.TIME_SERIES,
        ]
        self._supports_altimetry = False
        self._supports_range_dependent_bathymetry = False
        self._supports_range_dependent_ssp = False
        self._supports_range_dependent_bottom = False
        self._supports_layered_bottom = True
        self._supports_range_dependent_layered_bottom = False
        self._supports_elastic_media = True
        self._supports_multi_source_depth = False
        # Range-independent broadband wavenumber-integration / pulse
        # synthesis. Single spectral solve per frequency, transformed
        # to range and time. Median/mean samples represent the path.
        self._set_collapse_defaults({
            'ssp': 'mean',
            'bottom': 'median',
            'rd_layered_layers': 'preserve',
        })

        if executable is None:
            self.executable = _oases_find_executable(self, 'oasp')
        else:
            self.executable = Path(executable)

        if not self.executable.exists():
            raise ExecutableNotFoundError('OASP', str(self.executable))

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[RunMode] = None,
        *,
        frequencies=None,
        source_waveform=None,
        sample_rate=None,
    ) -> Result:
        """
        Run OASP parabolic equation computation

        Parameters
        ----------
        env : Environment
            Ocean environment (supports range-dependent)
        source : Source
            Acoustic source
        receiver : Receiver
            Receiver array
        run_mode : RunMode, optional
            ``COHERENT_TL`` (default) — extract TL at source frequency.
            ``BROADBAND`` — full H(f).
            ``TIME_SERIES`` — real pressure p(t); requires
            ``source_waveform`` + ``sample_rate``.
        source_waveform : ndarray, optional
            Source pulse for ``TIME_SERIES`` mode.
        sample_rate : float, optional
            Sampling rate of ``source_waveform`` in Hz.

        Returns
        -------
        result : Result
            :class:`Field` for COHERENT_TL, :class:`Field`
            for BROADBAND, :class:`Field` for TIME_SERIES.
        """
        n_time_samples = self.n_time_samples
        freq_max = self.freq_max

        run_mode = self._resolve_run_mode(run_mode)

        # The .trf reader collapses MSUFT / ISROW / NOUT axes onto the
        # first slot. Refuse option letters that would produce
        # multi-axis output rather than silently discard data.
        if self.options:
            multi_axis = {'V', 'H', 'R', 'U', 'F'} & set(str(self.options).split())
            if multi_axis:
                raise ConfigurationError(
                    f"OASP.run: options {sorted(multi_axis)} request "
                    "multi-component / decomposed output, which the .trf "
                    "reader currently flattens. Pass options without "
                    "these letters (default 'N J' returns scalar pressure) "
                    "or read the .trf directly."
                )

        if frequencies is not None:
            freqs_arr = np.atleast_1d(np.asarray(frequencies, dtype=float))
            if freqs_arr.size == 0:
                raise ConfigurationError(
                    "OASP.run(frequencies=…) requires at least one positive "
                    "frequency."
                )
            # OASP rebuilds its own (n_time_samples, freq_max) grid; an
            # explicit ``frequencies=`` vector overrides ``freq_max`` to the
            # user's max and ``n_time_samples`` so the resulting OASP bins
            # span at least the requested band. The (fmin, fmax, N) triple
            # is OASP's internal language — warn if the user vector is
            # non-equispaced so they know the resulting bins won't match
            # their input one-to-one.
            fmin_user, freq_max, n_user, _ = _oases_resample_frequencies(
                freqs_arr, 'OASP',
            )
            if n_user > 1:
                # df implied by the equispaced (fmin, fmax, N) grid that
                # OASES will actually run.
                df_user = (freq_max - fmin_user) / (n_user - 1)
            else:
                df_user = float(freqs_arr[0])
            if df_user > 0:
                # OASP requires NT = 2^M (oasp.tex:129); round up.
                target = max(int(n_time_samples or 0),
                             int(np.ceil(2.0 * freq_max / df_user)))
                if target > 1:
                    n_time_samples = 1 << (target - 1).bit_length()
                else:
                    n_time_samples = 2

        self._require_timeseries_signal(run_mode, source_waveform, sample_rate)

        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)
        fm = self._setup_file_manager()

        writer_kwargs: dict = {
            'integration_offset': self.integration_offset,
            'nw_samples': self.nw_samples,
            'freq_min': self.freq_min,
        }
        if self.center_frequency is not None:
            writer_kwargs['center_frequency'] = self.center_frequency
        if self.freq_output_increment is not None:
            writer_kwargs['freq_output_increment'] = self.freq_output_increment
        if self.options is not None:
            writer_kwargs['options'] = self.options
        if self.range_start is not None:
            writer_kwargs['range_start'] = self.range_start

        try:
            base_name = 'oasp_run'
            input_file = fm.get_path(f'{base_name}.dat')
            self._log(f"Writing OASP input file: {input_file}")
            write_oasp_input(
                filepath=input_file,
                env=env,
                source=source,
                receiver=receiver,
                n_time_samples=n_time_samples,
                freq_max=freq_max,
                **writer_kwargs,
            )

            # Run executable
            self._execute(input_file, fm.work_dir)
            trf_file = fm.get_path(f'{base_name}.trf')
            plt_file = fm.get_path(f'{base_name}.plt')

            if trf_file.exists():
                self._log(f"Reading OASP output: {trf_file}")
                trf_data = read_oasp_trf(trf_file)
            elif plt_file.exists():
                self._log(f"Reading OASP output: {plt_file}")
                trf_data = read_oasp_trf(plt_file)
            else:
                exc = ModelExecutionError(
                    self.model_name, return_code=0, stdout=None,
                    stderr=(
                        f"OASP did not produce {trf_file} or {plt_file}. "
                        "Consider using RAM for parabolic equation modeling: "
                        "RAM().run(env, source, receiver)."
                    ),
                )
                self._attach_prt_tail(exc, fm.work_dir, base_name)
                raise exc

            transfer_func = trf_data['transfer_function']  # shape: (n_frequencies, n_range, n_depth)

            if run_mode in (RunMode.BROADBAND, RunMode.TIME_SERIES):
                # Convention: (n_depth, n_range, n_frequencies) — trailing
                # axis is the variable dim. Source axes: (freq, range, depth).
                tf_reordered = np.transpose(transfer_func, (2, 1, 0))

                result = Field(
                    data=tf_reordered,
                    coords={
                        'depth': trf_data['depths'],
                        'range': trf_data['ranges'],
                        'frequency': trf_data['freq'],
                    },
                    phase_reference='travelling_wave',
                    **self._result_kwargs(
                        source,
                        backend='oasp',
                        frequencies=trf_data['freq'],
                        center_frequency=trf_data['center_frequency'],
                        n_time_samples=n_time_samples,
                        freq_max=freq_max,
                    ),
                )
                if run_mode == RunMode.TIME_SERIES:
                    result = result.synthesize_time_series(
                        source_waveform=source_waveform,
                        sample_rate=sample_rate,
                    )
            else:
                # COHERENT_TL: pick the bin nearest the source frequency
                # and return the complex narrowband pressure (transposed
                # to the (n_depth, n_range) layout). Users get TL via
                # ``field.tl`` or ``.to_tl()``.
                freq_idx = 0
                if len(trf_data['freq']) > 1:
                    freq_diff = np.abs(trf_data['freq'] - source.frequencies[0])
                    freq_idx = np.argmin(freq_diff)

                p_at_freq = transfer_func[freq_idx, :, :].T  # (n_d, n_r)

                result = Field(
                    data=p_at_freq,
                    coords={
                        'depth': trf_data['depths'],
                        'range': trf_data['ranges'],
                    },
                    **self._result_kwargs(
                        source,
                        backend='oasp',
                        frequencies=float(trf_data['freq'][freq_idx]),
                        frequencies_available=trf_data['freq'],
                        source_depth=trf_data['source_depth'],
                        center_frequency=trf_data['center_frequency'],
                        n_time_samples=n_time_samples,
                        freq_max=freq_max,
                    ),
                )

            self._attach_output_paths(
                result, fm.work_dir, base_name,
                primary_files=(('trf_file', '.trf'),),
            )

            self._log("OASP simulation complete")
            return result

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _execute(self, input_file: Path, work_dir: Path):
        """Execute OASP binary. FOR005 stays as stdin per OASES docs."""
        env = _oases_subprocess_env(
            input_file.stem,
            FOR002='src',
        )
        try:
            result = self._run_subprocess(
                [str(self.executable)], cwd=work_dir, env=env,
            )
        except ModelExecutionError as exc:
            self._attach_prt_tail(exc, work_dir, input_file.stem)
            raise
        if self.verbose and result.stdout:
            self._log(f"OASES output:\n{result.stdout}", level='debug')


def OASES(
    run_mode: RunMode = RunMode.COHERENT_TL,
    *,
    broadband: bool = False,
    **kwargs,
) -> PropagationModel:
    """Factory: instantiate the OASES sub-class that handles ``run_mode``.

    Returns an ``OAST``/``OASN``/``OASR``/``OASP`` instance configured with
    ``**kwargs``. Kwargs not accepted by the chosen sub-class raise
    ``TypeError`` — pick the right sub-class directly if you need
    sub-class-specific options.

    Parameters
    ----------
    run_mode : RunMode, optional
        Selects the sub-class:
        ``COHERENT_TL`` → ``OAST`` (or ``OASP`` when ``broadband=True``);
        ``COVARIANCE`` / ``REPLICA`` → ``OASN``;
        ``REFLECTION`` → ``OASR``;
        ``BROADBAND`` / ``TIME_SERIES`` → ``OASP``.
    broadband : bool, optional
        When True with ``COHERENT_TL``, route to ``OASP`` (broadband
        transfer function) instead of ``OAST``. Needed for range-dependent
        envs where OAST's range-independent kernel is inappropriate.
    **kwargs
        Forwarded verbatim to the sub-class constructor.

    Returns
    -------
    PropagationModel
        ``OAST``/``OASN``/``OASR``/``OASP`` instance ready to ``.run(...)``.
    """
    dispatch = {
        RunMode.COHERENT_TL: OASP if broadband else OAST,
        RunMode.COVARIANCE: OASN,
        RunMode.REPLICA: OASN,
        RunMode.REFLECTION: OASR,
        RunMode.BROADBAND: OASP,
        RunMode.TIME_SERIES: OASP,
    }
    cls = dispatch.get(run_mode)
    if cls is None:
        raise UnsupportedFeatureError(
            'OASES', str(run_mode),
            alternatives=[str(m) for m in dispatch],
            alternatives_label='run modes',
        )
    return cls(**kwargs)
