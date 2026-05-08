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

from uacpy.models.base import PropagationModel, RunMode, _UNSET, _resolve_overrides
from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.results import (
    Result, TLField, TransferFunction,
    Modes, Covariance, Replicas, ReflectionCoefficient,
)
from uacpy.core.constants import PRESSURE_FLOOR
from uacpy.core.exceptions import (
    ConfigurationError, ExecutableNotFoundError, UnsupportedFeatureError,
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
    R : ndarray, shape ``(n_angles,)`` (single freq) or ``(n_angles, n_freq)``
    phi : ndarray, same shape as ``R``  — phase in radians (reader stores degrees)
    freqs : ndarray, shape ``(n_freq,)``
    """
    freqs = np.asarray(data.get('frequencies', []), dtype=float)
    angle_lists = data.get('angles_or_slowness', [])
    R_lists = data.get('magnitude', [])
    phi_lists = data.get('phase', [])
    if not angle_lists:
        raise ValueError("OASR reader returned no frequency samples")
    theta = np.asarray(angle_lists[0], dtype=float)
    n_angles = len(theta)
    n_freq = len(freqs)
    for i, (a, m, p) in enumerate(zip(angle_lists, R_lists, phi_lists)):
        if len(a) != n_angles:
            raise ValueError(
                f"OASR: angle grid mismatch between frequencies "
                f"(freq[{i}] has {len(a)} angles, freq[0] has {n_angles}). "
                f"Multi-frequency stacking requires a shared angle grid."
            )
        if len(m) != n_angles or len(p) != n_angles:
            raise ValueError(
                f"OASR: payload length mismatch at frequency index {i}: "
                f"angles={len(a)}, magnitude={len(m)}, phase={len(p)}"
            )
    if n_freq == 1:
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


class _OASESBase(PropagationModel):
    """Base class for all OASES model wrappers with shared functionality."""

    def _find_executable(self, name: str) -> Path:
        """Find an OASES executable using the base-class helper.

        Preference order: ``<name>_bash`` wrapper, then the raw binary.
        Searches ``uacpy/bin/oases``, ``uacpy/bin/oalib``, and
        ``uacpy/third_party/oases/bin``.
        """
        return self._find_executable_in_paths(
            [f'{name}_bash', name],
            bin_subdirs=['oases', 'oalib'],
            dev_subdir='oases',
        )

    def _oases_subprocess_env(self, base_name: str, **extras: str) -> dict:
        """Build the FORnnn env-var dict the OASES csh wrappers set.

        Common keys (FOR001 input, FOR019/FOR020 plot files, FOR028/FOR029/
        FOR045 scratch) are populated for every sub-model; ``extras`` supplies
        per-binary unit numbers (e.g. FOR002='.src' for OAST, FOR016='.xsm'
        for OASN). Pass values without the ``base_name + '.'`` prefix — only
        the suffix, e.g. ``FOR002='src'``. ``base_name`` is the stem of the
        input file (no extension).
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


class OAST(_OASESBase):
    """
    OAST - OASES Transmission Loss Model

    Computes transmission loss using wavenumber integration. Best for
    range-independent environments with depth-dependent sound speed profiles.

    Parameters
    ----------
    executable : Path, optional
        Path to OAST executable. If None, searches standard locations.
    volume_attenuation : str, optional
        'T' (Thorp), 'F' (Francois-Garrison), 'B' (Biological). Default: None.
    use_tmpfs : bool, optional
        Use tmpfs for I/O operations (faster). Default False.
    verbose : bool, optional
        Print detailed execution information. Default False.
    work_dir : Path, optional
        Working directory for I/O files. Default uses temp directory.

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
        self._supports_layered_bottom = True
        self._supports_elastic_media = True
        if executable is None:
            self.executable = self._find_executable('oast')
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
        *,
        volume_attenuation=_UNSET,
        compute_contour=_UNSET,
        compute_depth_average=_UNSET,
        complex_contour=_UNSET,
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
        volume_attenuation : str, optional
            Per-call override for constructor default.
        **kwargs
            Additional OAST parameters:
            - integration_method : str
                'gauss' (Gauss-Legendre), 'simpson', 'trapz'
            - attenuation_units : str
                'db_per_m', 'db_per_wavelength', 'nepers_per_m'

        Returns
        -------
        result : Result
            Transmission loss field
        """
        # OAST emits coherent TL only — guard against other modes.
        if run_mode is not None and run_mode != RunMode.COHERENT_TL:
            raise UnsupportedFeatureError(
                self.model_name, str(run_mode),
                alternatives=["RunMode.COHERENT_TL", "OASP for broadband"],
            )

        # ``broadband`` is only meaningful on the unified OASES wrapper (where
        # it routes to OASP for range-dependent broadband TRF). If OAST is used
        # directly, fail loudly rather than silently dropping.
        if kwargs.pop('broadband', False):
            raise ConfigurationError(
                "OAST does not support broadband TRF; call OASP directly or "
                "use OASES(compute_tl(broadband=True))."
            )

        self._warn_unknown_kwargs(kwargs, allowed=('options',))

        with _resolve_overrides(
            self,
            volume_attenuation=volume_attenuation,
            compute_contour=compute_contour,
            compute_depth_average=compute_depth_average,
            complex_contour=complex_contour,
        ):
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

            # Run executable
                self._execute(input_file, fm.work_dir)

                # Read output - OAST creates .plt (data) and .plp (metadata) files
                # Per OASES documentation: FOR019 -> .plp, FOR020 -> .plt
                plt_file = fm.get_path(f'{base_name}.plt')

                # Check if output files exist
                if not plt_file.exists():
                    raise FileNotFoundError(
                        f"OAST plot data file not found: {plt_file}\n"
                        f"OAST should create .plt file via FOR020 environment variable.\n"
                        "Check input file and compare with examples in third_party/oases/tloss/"
                    )

                output_file = plt_file

                self._log(f"Reading OAST output: {output_file}")
                tl_data, metadata = read_oast_tl(
                    filepath=output_file,
                    receiver_depths=receiver.depths,
                    receiver_ranges=receiver.ranges
                )

                # Build a typed TLField. OAST stashes the rest in metadata.
                metadata.pop('model', None)
                metadata.pop('backend', None)
                result = TLField(
                    data=tl_data,
                    depths=receiver.depths,
                    ranges=receiver.ranges,
                    model='OAST',
                    backend='oast',
                    source_depths=np.atleast_1d(np.asarray(source.depths, dtype=float)),
                    frequencies=float(np.atleast_1d(source.frequencies)[0]),
                    metadata=metadata,
                )

                self._log("OAST simulation complete")
                return result

            finally:
                if fm.cleanup:
                    fm.cleanup_work_dir()

    def _execute(self, input_file: Path, work_dir: Path):
        """Execute OAST binary. FOR005 stays as stdin per OASES docs."""
        env = self._oases_subprocess_env(
            input_file.stem,
            FOR002='src',  # Source file
            FOR023='trc',  # Optional reflection-coef table
        )
        result = self._run_subprocess(
            [str(self.executable)], cwd=work_dir, env=env,
        )
        if self.verbose and result.stdout:
            self._log(f"OASES output:\n{result.stdout}", level='debug')


class OASN(_OASESBase):
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
        Path to OASN executable. If None, searches standard locations.
    use_tmpfs : bool, optional
        Use tmpfs for I/O operations. Default False.
    verbose : bool, optional
        Print detailed execution information. Default False.
    work_dir : Path, optional
        Working directory for I/O files. Default uses temp directory.

    Examples
    --------
    >>> from uacpy.models import OASN
    >>> oasn = OASN()
    >>> cov = oasn.compute_covariance(env, source, receiver)
    >>> # cov.covariance has shape (n_freq, n_rcv, n_rcv)
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
        self._supports_layered_bottom = True
        self._supports_elastic_media = True
        if executable is None:
            self.executable = self._find_executable('oasn2_bin')
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
        *,
        volume_attenuation=_UNSET,
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
        volume_attenuation : str, optional
            Per-call override for constructor default.
        **kwargs
            Additional OASN parameters forwarded to :func:`write_oasn_input`.

        Returns
        -------
        Covariance or Replicas
        """
        if run_mode is None:
            run_mode = RunMode.COVARIANCE
        if run_mode not in (RunMode.COVARIANCE, RunMode.REPLICA):
            raise UnsupportedFeatureError(
                self.model_name, str(run_mode),
                alternatives=["RunMode.COVARIANCE", "RunMode.REPLICA"],
            )

        self._warn_unknown_kwargs(kwargs, allowed=(
            'options',
            'surface_noise_level', 'white_noise_level', 'deep_noise_level',
            'discrete_sources', 'n_modes', 'integration_offset',
            'replica_zmin', 'replica_zmax', 'replica_nz',
            'replica_xmin', 'replica_xmax', 'replica_nx',
            'replica_ymin', 'replica_ymax', 'replica_ny',
        ))

        with _resolve_overrides(self, volume_attenuation=volume_attenuation):
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
                opt_tokens.add('R')
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
                        raise FileNotFoundError(
                            f"OASN covariance file not found. Checked: "
                            f"{xsm_file}, {fort16_file}"
                        )
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
                    return Covariance(
                        covariance=cov_data['covariance'],
                        receiver_positions=rcv_pos,
                        **kw,
                    )

                # RunMode.REPLICA
                rpo_file = fm.get_path(f'{base_name}.rpo')
                fort14_file = fm.get_path('fort.14')
                rep_path = rpo_file if rpo_file.exists() else fort14_file
                if not rep_path.exists():
                    raise FileNotFoundError(
                        f"OASN replica file not found. Checked: "
                        f"{rpo_file}, {fort14_file}\n"
                        "Pass replica_zmin/zmax/nz and replica_xmin/xmax/nx kwargs."
                    )
                self._log(f"Reading OASN replica file: {rep_path}")
                rep_data = read_oasn_replicas(rep_path)
                replica_z = np.linspace(rep_data['z_min'], rep_data['z_max'], rep_data['n_z'])
                replica_x = np.linspace(rep_data['x_min'], rep_data['x_max'], rep_data['n_x'])
                replica_y = np.linspace(rep_data['y_min'], rep_data['y_max'], rep_data['n_y'])
                freqs = _oasn_freq_axis(rep_data)
                kw = self._result_kwargs(
                    source,
                    backend='oasn',
                    frequencies=freqs,
                    n_receivers=rep_data['n_receivers'],
                    title=rep_data.get('title', ''),
                )
                return Replicas(
                    replicas=rep_data['replicas'],
                    replica_z=replica_z,
                    replica_x=replica_x,
                    replica_y=replica_y,
                    receiver_positions=rep_data.get('receiver_positions'),
                    **kw,
                )

            finally:
                if fm.cleanup:
                    fm.cleanup_work_dir()

    # ``compute_covariance`` / ``compute_replicas`` come from the base class
    # (RunMode.COVARIANCE / RunMode.REPLICA dispatch).

    def _execute(self, base_name: str, work_dir: Path):
        """Execute OASN binary. FOR005 stays as stdin per OASES docs."""
        env = self._oases_subprocess_env(
            base_name,
            FOR014='rpo',  # Replica vectors (unit 14)
            FOR016='xsm',  # Covariance matrix (unit 16)
            FOR026='chk',  # Checkpoint file (per oasn wrapper)
        )
        result = self._run_subprocess(
            [str(self.executable)], cwd=work_dir, env=env,
        )
        if self.verbose and result.stdout:
            self._log(f"OASES output:\n{result.stdout}", level='debug')


class OASR(_OASESBase):
    """
    OASR - OASES Reflection Coefficients Model

    Computes plane wave reflection coefficients at the bottom interface.

    Parameters
    ----------
    executable : Path, optional
        Path to OASR executable. If None, searches standard locations.
    use_tmpfs : bool, optional
        Use tmpfs for I/O operations. Default False.
    verbose : bool, optional
        Print detailed execution information. Default False.
    work_dir : Path, optional
        Working directory for I/O files. Default uses temp directory.

    Examples
    --------
    >>> from uacpy.models import OASR
    >>> oasr = OASR()
    >>> refl = oasr.run(env, source, receiver, angles=np.linspace(0, 90, 100))

    Notes
    -----
    - Computes bottom reflection coefficients
    - Supports elastic/poro-elastic bottom layers
    - Returns reflection loss in dB
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
        self._supports_layered_bottom = True
        self._supports_elastic_media = True
        if executable is None:
            self.executable = self._find_executable('oasr')
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
        angles=_UNSET,
        angle_type=_UNSET,
        reflection_type=_UNSET,
        volume_attenuation=_UNSET,
        frequencies: Optional[np.ndarray] = None,
        **kwargs
    ) -> Result:
        """Run OASR.

        Returns a :class:`ReflectionCoefficient` with ``theta`` (1-D angle
        grid in degrees) and ``R``/``phi`` shaped ``(n_angles, n_freqs)``
        when more than one frequency was requested. Single-frequency runs
        return 1-D ``R``/``phi``.

        OASES uses grazing angles natively. When ``angle_type='incidence'``,
        input angles are converted via ``grazing = 90 - incidence`` before
        being written to the OASR input file.

        Parameters
        ----------
        angles, angle_type, reflection_type, volume_attenuation : optional
            Per-call overrides for constructor defaults.
        frequencies : ndarray, optional
            Explicit frequency vector (Hz) — preferred multi-frequency
            interface. The min, max and length are forwarded as
            ``freq_min`` / ``freq_max`` / ``n_frequencies`` to the writer.
        **kwargs
            Forwarded to :func:`write_oasr_input`. ``freq_min``,
            ``freq_max``, ``n_frequencies`` are still accepted as
            alternatives but ``frequencies=`` is the documented path.
        """
        if run_mode is not None and run_mode != RunMode.REFLECTION:
            raise UnsupportedFeatureError(
                self.model_name, str(run_mode),
                alternatives=["RunMode.REFLECTION"],
            )

        angles = self._resolve(angles, 'angles')
        angle_type = self._resolve(angle_type, 'angle_type')
        reflection_type = (
            self._resolve(reflection_type, 'reflection_type')
        )
        volume_attenuation = (
            self._resolve(volume_attenuation, 'volume_attenuation')
        )

        if frequencies is not None:
            freqs_arr = np.atleast_1d(np.asarray(frequencies, dtype=float))
            if freqs_arr.size == 0:
                raise ValueError(
                    "OASR.run(frequencies=…) requires at least one "
                    "positive frequency."
                )
            for sweep_key in ('freq_min', 'freq_max', 'n_frequencies'):
                if sweep_key in kwargs:
                    raise ValueError(
                        f"OASR.run: pass either frequencies=… or "
                        f"{sweep_key}=…, not both."
                    )
            kwargs['freq_min'] = float(freqs_arr.min())
            kwargs['freq_max'] = float(freqs_arr.max())
            kwargs['n_frequencies'] = int(freqs_arr.size)

        self._warn_unknown_kwargs(kwargs)

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
                raise FileNotFoundError(
                    f"OASR output file not found. Checked: {search}"
                )

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

            self._log("OASR simulation complete")
            return field

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _execute(self, input_file: Path, work_dir: Path):
        """Execute OASR binary. FOR005 stays as stdin per OASES docs."""
        env = self._oases_subprocess_env(
            input_file.stem,
            FOR002='src',
            FOR004='trf',
            FOR022='rco',  # Reflection-coef table (slowness)
            FOR023='trc',  # Reflection-coef table (angle)
        )
        result = self._run_subprocess(
            [str(self.executable)], cwd=work_dir, env=env,
        )
        if self.verbose and result.stdout:
            self._log(f"OASES output:\n{result.stdout}", level='debug')

class OASP(_OASESBase):
    """
    OASP - OASES Pulse / Broadband Transfer-Function Model

    Computes broadband acoustic transfer functions via wavenumber integration
    followed by FFT to produce time-series / pulse responses. OASP is NOT a
    parabolic-equation solver (that was an earlier mislabel in this wrapper);
    it is the "Pulse" variant of SAFARI, using the same wavenumber-integration
    kernel as OAST but evaluated across a frequency sweep.

    Parameters
    ----------
    executable : Path, optional
        Path to OASP executable. If None, searches standard locations.
    use_tmpfs : bool, optional
        Use tmpfs for I/O operations. Default False.
    verbose : bool, optional
        Print detailed execution information. Default False.
    work_dir : Path, optional
        Working directory for I/O files. Default uses temp directory.

    Examples
    --------
    >>> from uacpy.models import OASP
    >>> oasp = OASP()
    >>> result = oasp.run(env, source, receiver, n_time_samples=256, freq_max=120)

    Notes
    -----
    - Broadband-oriented, outputs transfer functions
    - Supports range-dependent bathymetry and bottom
    - Can be computationally expensive (reduce n_time_samples for speed)
    - For most range-dependent problems, RAM is recommended
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
        # OASP: broadband wavenumber-integration PE; supports layered fluid
        # / elastic bottoms and limited range dependence (bathymetry only).
        self._supports_layered_bottom = True
        self._supports_range_dependent_bottom = False
        self._supports_elastic_media = True

        if executable is None:
            self.executable = self._find_executable('oasp')
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
        n_time_samples=_UNSET,
        freq_max=_UNSET,
        volume_attenuation=_UNSET,
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
        n_time_samples, freq_max, volume_attenuation : optional
            Per-call overrides for constructor defaults.
        **kwargs
            Additional OASP parameters

        Returns
        -------
        result : Result
            :class:`TLField` for COHERENT_TL, :class:`TransferFunction`
            for BROADBAND, :class:`TimeSeriesField` for TIME_SERIES.
        """
        # Resolve per-call overrides
        n_time_samples = self._resolve(n_time_samples, 'n_time_samples')
        freq_max = self._resolve(freq_max, 'freq_max')
        volume_attenuation = self._resolve(volume_attenuation, 'volume_attenuation')

        if run_mode is None:
            run_mode = RunMode.COHERENT_TL

        if frequencies is not None:
            freqs_arr = np.atleast_1d(np.asarray(frequencies, dtype=float))
            if freqs_arr.size == 0:
                raise ValueError(
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
                n_time_samples = max(int(n_time_samples or 0), int(np.ceil(2.0 * freq_max / df_user)))

        if run_mode == RunMode.TIME_SERIES and (
            source_waveform is None or sample_rate is None
        ):
            raise ValueError(
                "OASP.run(run_mode=TIME_SERIES) requires source_waveform "
                "and sample_rate. For the broadband transfer function "
                "H(f), use run_mode=RunMode.BROADBAND."
            )

        self._warn_unknown_kwargs(kwargs)

        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)
        fm = self._setup_file_manager()

        try:
            base_name = 'oasp_run'
            input_file = fm.get_path(f'{base_name}.dat')

            # Write input file
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

            # Read output
            trf_file = fm.get_path(f'{base_name}.trf')
            plt_file = fm.get_path(f'{base_name}.plt')

            if trf_file.exists():
                self._log(f"Reading OASP output: {trf_file}")
                trf_data = read_oasp_trf(trf_file)
            elif plt_file.exists():
                self._log(f"Reading OASP output: {plt_file}")
                trf_data = read_oasp_trf(plt_file)
            else:
                raise FileNotFoundError(
                    f"OASP output files not found: {trf_file} or {plt_file}\n\n"
                    "Consider using RAM for parabolic equation modeling:\n"
                    "  >>> from uacpy.models import RAM\n"
                    "  >>> ram = RAM()\n"
                    "  >>> result = ram.run(env, source, receiver)"
                )

            transfer_func = trf_data['transfer_function']  # shape: (n_freq, n_range, n_depth)

            if run_mode in (RunMode.BROADBAND, RunMode.TIME_SERIES):
                # Convention: (n_depth, n_range, n_freq) — trailing
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
                # COHERENT_TL: extract TL at source frequency.
                freq_idx = 0
                if len(trf_data['freq']) > 1:
                    freq_diff = np.abs(trf_data['freq'] - source.frequencies[0])
                    freq_idx = np.argmin(freq_diff)

                tl_at_freq = transfer_func[freq_idx, :, :]
                magnitude = np.abs(tl_at_freq)
                magnitude[magnitude == 0] = PRESSURE_FLOOR
                tl_db = -20 * np.log10(magnitude)
                # Transpose to (n_depth, n_range).
                tl_db = tl_db.T

                result = TLField(
                    data=tl_db,
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

            self._log("OASP simulation complete")
            return result

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _execute(self, input_file: Path, work_dir: Path):
        """Execute OASP binary. FOR005 stays as stdin per OASES docs."""
        env = self._oases_subprocess_env(
            input_file.stem,
            FOR002='src',
        )
        result = self._run_subprocess(
            [str(self.executable)], cwd=work_dir, env=env,
        )
        if self.verbose and result.stdout:
            self._log(f"OASES output:\n{result.stdout}", level='debug')


class OASES(PropagationModel):
    """
    OASES - Unified interface to the OASES suite

    This class provides a convenient unified interface to all OASES models.
    It automatically delegates to the appropriate specialized model (OAST, OASN,
    OASR, OASP) based on the computation requested.

    Parameters
    ----------
    use_tmpfs : bool, optional
        Use tmpfs for I/O operations. Default False.
    verbose : bool, optional
        Print detailed execution information. Default False.
    work_dir : Path, optional
        Working directory for I/O files. Default uses temp directory.

    Examples
    --------
    >>> from uacpy.models import OASES
    >>> oases = OASES()
    >>>
    >>> # Transmission loss (uses OAST)
    >>> result = oases.compute_tl(env, source, receiver)
    >>>
    >>> # Hydrophone-array covariance (uses OASN)
    >>> cov = oases.compute_covariance(env, source, receiver)
    >>>
    >>> # MFP replica fields (uses OASN)
    >>> rep = oases.compute_replicas(env, source, receiver)
    >>>
    >>> # Or use run() with run_mode parameter
    >>> from uacpy.models.base import RunMode
    >>> result = oases.run(env, source, receiver, run_mode=RunMode.COHERENT_TL)

    Notes
    -----
    This is a convenience wrapper. You can also use the specialized classes
    directly (OAST, OASN, OASR, OASP) for more control.
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
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
        )

        # ``executable`` only makes sense per sub-binary; OASES is a
        # dispatcher so it cannot accept a single executable path.
        if executable is not None:
            raise ConfigurationError(
                "The unified OASES wrapper does not accept a single "
                "executable= path; instantiate OAST/OASN/OASR/OASP "
                "directly to override their binaries."
            )

        # Sub-model-specific kwargs (e.g. ``angles`` for OASR,
        # ``n_time_samples`` for OASP) cannot be applied uniformly here —
        # passing them to all four would break the constructors that don't
        # accept them. Reject anything outside the standard plumbing set.
        if kwargs:
            raise ConfigurationError(
                f"Unified OASES wrapper received sub-model-specific "
                f"kwargs {sorted(kwargs)}. Set them on OAST/OASN/OASR/OASP "
                "directly."
            )

        common = dict(
            use_tmpfs=use_tmpfs,
            verbose=verbose,
            work_dir=work_dir,
            volume_attenuation=volume_attenuation,
            francois_garrison_params=francois_garrison_params,
            bio_layers=bio_layers,
        )
        self._oast = OAST(**common)
        self._oasn = OASN(**common)
        self._oasr = OASR(**common)
        self._oasp = OASP(**common)

        # Aggregate supported modes from all submodels
        self._supported_modes = list(set(
            self._oast._supported_modes +
            self._oasn._supported_modes +
            self._oasr._supported_modes +
            self._oasp._supported_modes
        ))

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: RunMode = RunMode.COHERENT_TL,
        **kwargs
    ) -> Result:
        """
        Run OASES computation

        Automatically delegates to the appropriate OASES model based on run_mode.

        Parameters
        ----------
        env : Environment
            Ocean environment
        source : Source
            Acoustic source
        receiver : Receiver
            Receiver array
        run_mode : RunMode, optional
            Computation mode. Default is COHERENT_TL.
        **kwargs
            Additional parameters passed to the underlying model.
            Notable keyword arguments:

            broadband : bool, optional
                When True and the environment is range-dependent, delegate
                to OASP (broadband transfer function) instead of OAST. OASP
                handles range-dependent bathymetry via repeated OAST calls
                internally. Default False.

        Returns
        -------
        result : Result
            Simulation results.
        """
        if run_mode == RunMode.COHERENT_TL:
            broadband = kwargs.pop('broadband', False)
            if env.is_range_dependent and broadband:
                return self._oasp.run(env, source, receiver, **kwargs)
            return self._oast.run(env, source, receiver, **kwargs)
        if run_mode == RunMode.MODES:
            raise UnsupportedFeatureError(
                self.model_name, str(RunMode.MODES),
                alternatives=[
                    "RunMode.COVARIANCE (OASN hydrophone-array covariance)",
                    "RunMode.REPLICA (OASN replica field at array elements)",
                    "Kraken/KrakenC for explicit normal-mode eigenfunctions",
                ],
            )
        if run_mode == RunMode.REFLECTION:
            return self._oasr.run(env, source, receiver, **kwargs)
        if run_mode in (RunMode.COVARIANCE, RunMode.REPLICA):
            return self._oasn.run(env, source, receiver,
                                  run_mode=run_mode, **kwargs)
        if run_mode in (
            RunMode.BROADBAND, RunMode.TIME_SERIES,
        ):
            return self._oasp.run(env, source, receiver,
                                  run_mode=run_mode, **kwargs)
        raise ValueError(f"Run mode {run_mode} not supported by OASES")

    def compute_tl(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        broadband: bool = False,
        **kwargs,
    ) -> Result:
        """Compute transmission loss using OAST (wavenumber integration)
        for range-independent envs, or OASP for range-dependent envs when
        ``broadband=True``. Both routes return a single-frequency
        :class:`TLField` at ``source.frequencies[0]``.

        Parameters
        ----------
        env : Environment
            Ocean environment.
        source : Source
            Acoustic source.
        receiver : Receiver
            Receiver grid (required).
        broadband : bool, optional
            When True, delegate to OASP instead of OAST. Needed for
            range-dependent environments where OAST's range-independent
            wavenumber-integration kernel is inappropriate. Default False.
        **kwargs
            Forwarded to OAST / OASP.
        """
        if broadband:
            return self._oasp.run(env, source, receiver, **kwargs)
        return self._oast.run(env, source, receiver, **kwargs)


    # ``compute_reflection`` / ``compute_time_series`` /
    # ``compute_transfer_function`` come from PropagationModel and route
    # through OASES.run, which dispatches to OASR / OASP.
