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

import numpy as np
from pathlib import Path
from typing import Optional

from uacpy.models.base import PropagationModel, RunMode
from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.results import (
    Result, PressureField, TransferFunction,
    Covariance, Replicas, ReflectionCoefficient,
)
from uacpy.core.exceptions import (
    ConfigurationError, ExecutableNotFoundError, ModelExecutionError,
    UnsupportedFeatureError,
)
from uacpy.io.oases_writer import write_oast_input, write_oasn_input, write_oasp_input, write_oasr_input
from uacpy.io.oases_reader import (
    read_oast_tl,
    read_oasn_covariance,
    read_oasn_replicas,
    read_oasp_trf,
    read_oasr_reflection_coefficients,
)


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
    volume_attenuation : str, optional
        ``None`` | ``'T'`` Thorp | ``'F'`` Francois–Garrison | ``'B'`` Biological.
    francois_garrison_params, bio_layers : optional
        Required when ``volume_attenuation`` is ``'F'`` / ``'B'``.
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
        volume_attenuation: Optional[str] = None,
        francois_garrison_params: Optional[tuple] = None,
        bio_layers: Optional[list] = None,
        compute_contour: bool = False,
        compute_depth_average: bool = False,
        complex_contour: bool = True,
        use_tmpfs: bool = False,
        verbose: bool = False,
        work_dir: Optional[Path] = None,
        **kwargs,
    ):
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            **kwargs,
        )
        self.volume_attenuation = volume_attenuation
        self.francois_garrison_params = francois_garrison_params
        self.bio_layers = bio_layers
        self.compute_contour = bool(compute_contour)
        self.compute_depth_average = bool(compute_depth_average)
        self.complex_contour = bool(complex_contour)
        self._validate_volume_attenuation_params()

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
        **kwargs
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
        **kwargs
            Additional OAST parameters (e.g. ``options='N J T C'``).

        Returns
        -------
        result : Result
            Transmission loss field
        """
        run_mode = self._resolve_run_mode(run_mode)

        if kwargs.pop('broadband', False):
            raise ConfigurationError(
                "OAST does not support broadband TRF; call OASP directly or "
                "use OASES(broadband=True) (the factory dispatches to OASP)."
            )

        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)
        fm = self._setup_file_manager()

        try:
            base_name = 'oast_run'
            input_file = fm.get_path(f'{base_name}.dat')

            # Translate the typed kwargs into OAST option letters. The user
            # can still pass a raw `options=` string (e.g. 'N J T C') via
            # **kwargs as an escape hatch — it wins over the typed flags.
            user_options = kwargs.pop('options', None)
            if user_options is None:
                opt = ['N', 'T']                            # Normal stress + TL table
                if self.complex_contour:
                    opt.append('J')
                if self.compute_contour:
                    opt.append('C')
                if self.compute_depth_average:
                    opt.append('A')
                user_options = ' '.join(opt)

            self._log(f"Writing OAST input file: {input_file} (options={user_options})")
            write_oast_input(
                filepath=input_file,
                env=env,
                source=source,
                receiver=receiver,
                volume_attenuation=self.volume_attenuation,
                francois_garrison_params=self.francois_garrison_params,
                bio_layers=self.bio_layers,
                options=user_options,
                **kwargs,
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
            native = PressureField(
                units="dB",
                data=tl_data,
                depths=native_depths,
                ranges=native_ranges,
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
    volume_attenuation : str, optional
        ``None`` | ``'T'`` | ``'F'`` | ``'B'``.
    francois_garrison_params, bio_layers : optional
        Required when ``volume_attenuation`` is ``'F'`` / ``'B'``.
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
        volume_attenuation: Optional[str] = None,
        francois_garrison_params: Optional[tuple] = None,
        bio_layers: Optional[list] = None,
        use_tmpfs: bool = False,
        verbose: bool = False,
        work_dir: Optional[Path] = None,
        **kwargs,
    ):
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir, **kwargs,
        )
        self.volume_attenuation = volume_attenuation
        self.francois_garrison_params = francois_garrison_params
        self.bio_layers = bio_layers
        self._validate_volume_attenuation_params()

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

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[RunMode] = None,
        **kwargs
    ) -> Result:
        """
        Run OASN.

        Parameters
        ----------
        run_mode : RunMode, optional
            ``RunMode.COVARIANCE`` (default) → :class:`Covariance`.
            ``RunMode.REPLICA`` → :class:`Replicas` (requires Block-X
            replica-grid kwargs: ``replica_zmin``, ``replica_zmax``,
            ``replica_nz``, ``replica_xmin``, ``replica_xmax``,
            ``replica_nx``).
        **kwargs
            Additional OASN parameters forwarded to :func:`write_oasn_input`.

        Returns
        -------
        Covariance or Replicas
        """
        run_mode = self._resolve_run_mode(run_mode)

        # Public API uses metres for ranges; OASES expects km. Build a
        # shallow copy with the four range-axis kwargs converted so we
        # don't mutate the caller's dict (z-axis stays metres — OASES
        # accepts those). Only meaningful for RunMode.REPLICA.
        if run_mode == RunMode.REPLICA:
            kwargs = dict(kwargs)
            for k in ('replica_xmin', 'replica_xmax',
                      'replica_ymin', 'replica_ymax'):
                if k in kwargs:
                    kwargs[k] = float(kwargs[k]) / 1000.0

        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)

        # Force the OASN options block to include the file output we need.
        # The user can pass a custom 'options' to add other letters (J, F, …),
        # but N (covariance) or R (replica) is added automatically based on
        # run_mode so the wrapper's contract (return type) is honoured.
        user_options = kwargs.pop('options', None) or 'J'
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
                volume_attenuation=self.volume_attenuation,
                francois_garrison_params=self.francois_garrison_params,
                bio_layers=self.bio_layers,
                options=options,
                **kwargs,
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
                        f"{rpo_file}, {fort14_file}. Pass "
                        "replica_zmin/zmax/nz and replica_xmin/xmax/nx kwargs."
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
    volume_attenuation : str, optional
        ``None`` | ``'T'`` | ``'F'`` | ``'B'``.
    francois_garrison_params, bio_layers : optional
        Required when ``volume_attenuation`` is ``'F'`` / ``'B'``.
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
        volume_attenuation: Optional[str] = None,
        francois_garrison_params: Optional[tuple] = None,
        bio_layers: Optional[list] = None,
        use_tmpfs: bool = False,
        verbose: bool = False,
        work_dir: Optional[Path] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        executable : Path, optional
            Path to OASR executable. Auto-detected if None.
        angles : ndarray, optional
            Angles to compute (degrees). Default: np.linspace(0, 90, 181).
        angle_type : str, optional
            'grazing' (OASES native) or 'incidence' (converted via
            ``grazing = 90 - incidence`` before being written to the input
            file). Default: 'grazing'.
        reflection_type : str, optional
            One of 'P-P' (default), 'P-SV', 'P-Slow' (Biot only), or
            'transmission'. Translates to the OASR option letter
            ('N' / 'S' / 'B' / 't').
        volume_attenuation : str, optional
            'T' (Thorp), 'F' (Francois-Garrison), 'B' (Biological). Default: None.
        francois_garrison_params : tuple, optional
            (T, S, pH, z_bar) required when ``volume_attenuation='F'``.
        bio_layers : list, optional
            [(Z1, Z2, f0, Q, a0), ...] required when ``volume_attenuation='B'``.
        """
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir, **kwargs,
        )
        self.angles = angles
        self.angle_type = angle_type
        self.reflection_type = reflection_type
        self.volume_attenuation = volume_attenuation
        self.francois_garrison_params = francois_garrison_params
        self.bio_layers = bio_layers
        self._validate_volume_attenuation_params()

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
        frequencies: Optional[np.ndarray] = None,
        **kwargs
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
            Explicit frequency vector (Hz) — preferred multi-frequency
            interface. The min, max and length are forwarded as
            ``freq_min`` / ``freq_max`` / ``n_frequencies`` to the writer.
        **kwargs
            Forwarded to :func:`write_oasr_input`. ``freq_min``,
            ``freq_max``, ``n_frequencies`` are still accepted as
            alternatives but ``frequencies=`` is the documented path.
        """
        run_mode = self._resolve_run_mode(run_mode)

        angles = self.angles
        angle_type = self.angle_type
        reflection_type = self.reflection_type
        volume_attenuation = self.volume_attenuation

        if frequencies is not None:
            freqs_arr = np.atleast_1d(np.asarray(frequencies, dtype=float))
            if freqs_arr.size == 0:
                raise ConfigurationError(
                    "OASR.run(frequencies=…) requires at least one "
                    "positive frequency."
                )
            for sweep_key in ('freq_min', 'freq_max', 'n_frequencies'):
                if sweep_key in kwargs:
                    raise ConfigurationError(
                        f"OASR.run: pass either frequencies=… or "
                        f"{sweep_key}=…, not both."
                    )
            kwargs['freq_min'] = float(freqs_arr.min())
            kwargs['freq_max'] = float(freqs_arr.max())
            kwargs['n_frequencies'] = int(freqs_arr.size)

        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)

        # Only synthesize a default angle grid when the user passed neither an
        # explicit ``angles=`` array nor angle_min/angle_max/n_angles via kwargs.
        if angles is None:
            user_specified_angle_kwargs = (
                'angle_min' in kwargs or 'angle_max' in kwargs or 'n_angles' in kwargs
            )
            if not user_specified_angle_kwargs:
                angles = np.linspace(0, 90, 181)

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
                angle_type=angle_type,
                reflection_type=reflection_type,
                volume_attenuation=volume_attenuation,
                francois_garrison_params=self.francois_garrison_params,
                bio_layers=self.bio_layers,
                **kwargs,
            )

            self._execute(input_file, fm.work_dir)

            # OASR writes a grazing-angle table to .trc and a slowness
            # table to .rco. Both files may be present even though the
            # user only requested one — pick the one matching the
            # requested sampling. uacpy emits angles by default (no 'p'
            # option in the OASR options string), so .trc wins; the user
            # can pass options='p ...' to switch to slowness sampling, in
            # which case we prefer .rco.
            requested_opts = (kwargs.get('options') or '').split()
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
                    reflection_type=reflection_type,
                ),
            )
            self._attach_output_paths(
                field, fm.work_dir, base_name,
                primary_files=(
                    ('plt_file', '.plt'),
                    ('plp_file', '.plp'),
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
    volume_attenuation : str, optional
        ``None`` | ``'T'`` | ``'F'`` | ``'B'``.
    francois_garrison_params, bio_layers : optional
        Required when ``volume_attenuation`` is ``'F'`` / ``'B'``.
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    Broadband-oriented; supports ``RunMode.BROADBAND`` (returns a
    :class:`TransferFunction`) and ``RunMode.TIME_SERIES`` (returns a
    :class:`TimeSeriesField` after ``synthesize_time_series``). For a
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
        volume_attenuation: Optional[str] = None,
        francois_garrison_params: Optional[tuple] = None,
        bio_layers: Optional[list] = None,
        use_tmpfs: bool = False,
        verbose: bool = False,
        work_dir: Optional[Path] = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        executable : Path, optional
            Path to OASP executable. Auto-detected if None.
        n_time_samples : int, optional
            Number of time samples for FFT. Default: 4096.
        freq_max : float, optional
            Maximum frequency for FFT (Hz). Default: 250.
        volume_attenuation : str, optional
            'T' (Thorp), 'F' (Francois-Garrison), 'B' (Biological). Default: None.
        francois_garrison_params : tuple, optional
            (T, S, pH, z_bar) required when ``volume_attenuation='F'``.
        bio_layers : list, optional
            [(Z1, Z2, f0, Q, a0), ...] required when ``volume_attenuation='B'``.
        """
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            **kwargs,
        )
        self.n_time_samples = n_time_samples
        self.freq_max = freq_max
        self.volume_attenuation = volume_attenuation
        self.francois_garrison_params = francois_garrison_params
        self.bio_layers = bio_layers
        self._validate_volume_attenuation_params()

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
        frequencies=None,
        source_waveform=None,
        sample_rate=None,
        **kwargs
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
        **kwargs
            Additional OASP parameters

        Returns
        -------
        result : Result
            :class:`PressureField` for COHERENT_TL, :class:`TransferFunction`
            for BROADBAND, :class:`TimeSeriesField` for TIME_SERIES.
        """
        n_time_samples = self.n_time_samples
        freq_max = self.freq_max
        volume_attenuation = self.volume_attenuation

        run_mode = self._resolve_run_mode(run_mode)

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
            # span at least the requested band.
            freq_max = float(freqs_arr.max())
            df_user = (
                float(np.diff(freqs_arr).min())
                if freqs_arr.size > 1
                else float(freqs_arr[0])
            )
            if df_user > 0:
                # OASP requires NT = 2^M (oasp.tex:129); round up.
                target = max(int(n_time_samples or 0),
                             int(np.ceil(2.0 * freq_max / df_user)))
                if target > 1:
                    n_time_samples = 1 << (target - 1).bit_length()
                else:
                    n_time_samples = 2

        if run_mode == RunMode.TIME_SERIES and (
            source_waveform is None or sample_rate is None
        ):
            raise ConfigurationError(
                "OASP.run(run_mode=TIME_SERIES) requires source_waveform "
                "and sample_rate. For the broadband transfer function "
                "H(f), use run_mode=RunMode.BROADBAND."
            )

        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)
        fm = self._setup_file_manager()

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
                volume_attenuation=volume_attenuation,
                francois_garrison_params=self.francois_garrison_params,
                bio_layers=self.bio_layers,
                **kwargs
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

                result = TransferFunction(
                    data=tf_reordered,
                    depths=trf_data['depths'],
                    ranges=trf_data['ranges'],
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

                result = PressureField(
                    units="complex",
                    data=p_at_freq,
                    depths=trf_data['depths'],
                    ranges=trf_data['ranges'],
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
    ``**kwargs``. Sub-class-specific kwargs that the chosen class doesn't
    consume (e.g. ``angles=`` when routing to ``OASN``) are silently
    dropped per uacpy convention.

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
        Forwarded to the sub-class constructor.

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
        )
    return cls(**kwargs)
