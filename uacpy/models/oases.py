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
import warnings

from uacpy.models.base import PropagationModel, RunMode, _UNSET
from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.results import (
    Result, TLField, TransferFunction,
    Modes, OASNCovariance, ReflectionCoefficient,
)
from uacpy.io.oases_writer import write_oast_input, write_oasn_input, write_oasp_input, write_oasr_input
from uacpy.io.oases_reader import (
    read_oast_tl,
    read_oases_modes,
    read_oasn_covariance,
    read_oasp_trf,
    read_oasr_reflection_coefficients,
)


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
        use_tmpfs: bool = False,
        verbose: bool = False,
        work_dir: Optional[Path] = None,
    ):
        super().__init__(use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir)
        self.volume_attenuation = volume_attenuation
        self.francois_garrison_params = francois_garrison_params
        self.bio_layers = bio_layers
        self._validate_volume_attenuation_params()

        self._supported_modes = [RunMode.COHERENT_TL]
        # OAST: range-independent wavenumber integration; multi-layer
        # fluid + elastic bottom honored.
        self._supports_layered_bottom = True
        self._unsupported_env_alternatives = "OASP (range-dep) or RAM"

        if executable is None:
            self.executable = self._find_executable('oast')
        else:
            self.executable = Path(executable)

        if not self.executable.exists():
            raise FileNotFoundError(
                f"OAST executable not found: {self.executable}\n"
                "Please compile OASES using the installation scripts."
            )

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
            from uacpy.core.exceptions import UnsupportedFeatureError
            raise UnsupportedFeatureError(
                self.model_name, str(run_mode),
                alternatives=["RunMode.COHERENT_TL", "OASP for broadband"],
            )

        # Resolve per-call overrides
        volume_attenuation = volume_attenuation if volume_attenuation is not _UNSET else self.volume_attenuation

        # ``broadband`` is only meaningful on the unified OASES wrapper (where
        # it routes to OASP for range-dependent broadband TRF). If OAST is used
        # directly, fail loudly rather than silently dropping.
        if kwargs.pop('broadband', False):
            from uacpy.core.exceptions import ConfigurationError
            raise ConfigurationError(
                "OAST does not support broadband TRF; call OASP directly or "
                "use OASES(compute_tl(broadband=True))."
            )

        # Handle range-dependent environments
        if env.is_range_dependent:
            min_depth = env.bathymetry[:, 1].min()
            max_depth = env.bathymetry[:, 1].max()
            warning_msg = (
                f"OAST does not support range-dependent environments. "
                f"Using MAXIMUM bathymetry depth ({max_depth:.1f}m) as approximation. "
                f"Bathymetry range: {min_depth:.1f}m - {max_depth:.1f}m. "
                f"This ensures source/receiver depths remain valid (prevents depth violations). "
                f"For accurate range-dependent modeling, use OASP, Bellhop, or RAM."
            )
            # Emit Python warning for test detection
            warnings.warn(warning_msg, UserWarning, stacklevel=2)
            # Also log via logger for verbose output
            self._log(warning_msg, level='warn')
            env = env.get_range_independent_approximation(method='max')

        self.validate_inputs(env, source, receiver)
        fm = self._setup_file_manager()

        try:
            base_name = 'oast_run'
            input_file = fm.get_path(f'{base_name}.dat')

            # Write input file
            self._log(f"Writing OAST input file: {input_file}")
            write_oast_input(
                filepath=input_file,
                env=env,
                source=source,
                receiver=receiver,
                volume_attenuation=volume_attenuation,
                francois_garrison_params=self.francois_garrison_params,
                bio_layers=self.bio_layers,
                **kwargs
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
                source_depths=np.atleast_1d(np.asarray(source.depth, dtype=float)),
                frequency=float(np.atleast_1d(source.frequency)[0]),
                metadata=metadata,
            )

            self._log("OAST simulation complete")
            return result

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _execute(self, input_file: Path, work_dir: Path):
        """Execute OAST binary via the base-class subprocess helper.

        FOR005 is Fortran-standard stdin and MUST NOT be assigned to an
        output plot file (doing so collides with FOR020=.plt). Only the
        real output units documented in ``oast.tex`` are exposed.
        """
        import os
        base_name = input_file.stem
        env = os.environ.copy()
        # Only env vars that the OAST Fortran actually reads (via OPFILB's
        # getenv or equivalent) are set here. FOR003/FOR004 were decorative —
        # no Fortran source references them. Canonical mapping follows the
        # ``oast`` csh wrapper in third_party/oases/bin/oast.
        env['FOR001'] = f'{base_name}.dat'  # Input file
        env['FOR002'] = f'{base_name}.src'  # Source file
        env['FOR019'] = f'{base_name}.plp'  # Plot parameter file
        env['FOR020'] = f'{base_name}.plt'  # Plot data file (TL values)
        env['FOR023'] = f'{base_name}.trc'  # Optional reflection-coef table
        env['FOR028'] = f'{base_name}.028'
        env['FOR029'] = f'{base_name}.029'
        env['FOR045'] = f'{base_name}.045'

        result = self._run_subprocess(
            [str(self.executable)], cwd=work_dir, timeout=300, env=env,
        )
        if self.verbose and result.stdout:
            self._log(f"OASES output:\n{result.stdout}", level='debug')


class OASN(_OASESBase):
    """
    OASN - OASES Normal Modes Model

    Computes normal modes for range-independent environments.

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
    >>> modes = oasn.run(env, source, receiver)

    Notes
    -----
    - Mode file reading is experimental
    - For production use, consider Kraken for normal mode analysis
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
    ):
        super().__init__(use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir)
        self.volume_attenuation = volume_attenuation
        self.francois_garrison_params = francois_garrison_params
        self.bio_layers = bio_layers
        self._validate_volume_attenuation_params()

        self._supported_modes = [RunMode.MODES]
        # OASN: range-independent covariance / replica field; multi-layer
        # bottom honored.
        self._supports_layered_bottom = True
        self._unsupported_env_alternatives = "Kraken/KrakenC for full mode shapes"

        if executable is None:
            self.executable = self._find_executable('oasn2_bin')
        else:
            self.executable = Path(executable)

        if not self.executable.exists():
            raise FileNotFoundError(
                f"OASN executable not found: {self.executable}\n"
                "Please compile OASES using the installation scripts."
            )

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
        Run OASN covariance / replica computation.

        .. note::
            OASN produces covariance matrices (.xsm) and matched-field
            replicas (.rpo); it does NOT emit explicit mode shapes. For
            modal analysis use Kraken or KrakenC instead.

        Parameters
        ----------
        env : Environment
            Ocean environment (must be range-independent).
        source : Source
            Acoustic source (for frequency).
        receiver : Receiver
            Receiver array.
        volume_attenuation : str, optional
            Per-call override for constructor default.
        **kwargs
            Additional OASN parameters.

        Returns
        -------
        result : OASNCovariance
            Replica vectors and spatial covariance matrices.
        """
        if run_mode is not None and run_mode != RunMode.MODES:
            from uacpy.core.exceptions import UnsupportedFeatureError
            raise UnsupportedFeatureError(
                self.model_name, str(run_mode),
                alternatives=["RunMode.MODES (replica/covariance)"],
            )
        volume_attenuation = volume_attenuation if volume_attenuation is not _UNSET else self.volume_attenuation
        self.validate_inputs(env, source, receiver)
        fm = self._setup_file_manager()

        try:
            base_name = 'oasn_run'
            input_file = fm.get_path(f'{base_name}.dat')

            # Write input file
            self._log(f"Writing OASN input file: {input_file}")
            write_oasn_input(
                filepath=input_file,
                env=env,
                source=source,
                receiver=receiver,
                volume_attenuation=volume_attenuation,
                francois_garrison_params=self.francois_garrison_params,
                bio_layers=self.bio_layers,
                **kwargs
            )

            # Run executable
            self._execute(base_name, fm.work_dir)

            # Read XSM file - OASN's primary output is a direct-access
            # covariance file on unit 16 (see oasmun21_bin.f:270-346 — WRITE
            # records of fixed 8-byte length, NOT Fortran-sequential). The
            # previous implementation piped the .xsm through read_oases_modes
            # → _read_oasp_trf_binary which expects SEQUENTIAL unformatted
            # records, yielding silent empty-mode results.
            xsm_file = fm.get_path(f'{base_name}.xsm')
            fort16_file = fm.get_path('fort.16')

            if xsm_file.exists():
                self._log(f"Reading OASN covariance file: {xsm_file}")
                cov_data = read_oasn_covariance(xsm_file)
            elif fort16_file.exists():
                self._log(f"Reading OASN covariance file: {fort16_file}")
                cov_data = read_oasn_covariance(fort16_file)
            else:
                raise FileNotFoundError(
                    f"OASN covariance file not found. Checked: {xsm_file}, {fort16_file}\n\n"
                    "OASN produces covariance matrices (.xsm), not explicit mode "
                    "shapes. For modal analysis, use Kraken/KrakenC instead."
                )

            # Opportunistic mode-wavenumber recovery from the OASN stdout/log
            # if one is sitting alongside the xsm file.
            try:
                mode_aux = read_oases_modes(xsm_file if xsm_file.exists() else fort16_file)
            except Exception:
                mode_aux = {'k': np.array([]), 'phi': np.array([]), 'z': np.array([]), 'n_modes': 0}

            # Package as Field object. OASN outputs covariance matrices, not
            # mode shapes; we expose both the covariance and any wavenumbers
            # scraped from the print log in metadata.
            return OASNCovariance(
                phi=mode_aux.get('phi', np.array([])),
                z=mode_aux.get('z', receiver.depths),
                covariance=cov_data.get('covariance'),
                n_modes=mode_aux.get('n_modes', 0),
                model='OASN',
                backend='oasn',
                source_depths=np.atleast_1d(np.asarray(source.depth, dtype=float)),
                frequency=float(np.atleast_1d(source.frequency)[0]),
                metadata={
                    'k': mode_aux.get('k', np.array([])),
                    'n_receivers': cov_data.get('n_receivers'),
                    'n_frequencies': cov_data.get('n_frequencies'),
                    'freq_min': cov_data.get('freq_min'),
                    'freq_max': cov_data.get('freq_max'),
                    'experimental': True,
                },
            )

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _execute(self, base_name: str, work_dir: Path):
        """Execute OASN binary (FOR005 left as stdin per OASES docs).

        Canonical env-var mapping follows the ``oasn`` csh wrapper in
        third_party/oases/bin/oasn: FOR014=.rpo (replica vectors, OPFILB at
        oasmun21_bin.f:456), FOR016=.xsm (covariance, getenv at :335),
        FOR026=.chk (checkpoint). The prior implementation wrote the replica
        output to FOR026 and left FOR014 unset, which — on gfortran — causes
        the REPLICA write to fall back to fort.14 instead of the expected
        ``.rpo`` location.
        """
        import os
        env = os.environ.copy()
        env['FOR001'] = f'{base_name}.dat'
        env['FOR014'] = f'{base_name}.rpo'     # Replica vectors (unit 14)
        env['FOR016'] = f'{base_name}.xsm'     # Covariance matrix (unit 16)
        env['FOR019'] = f'{base_name}.plp'
        env['FOR020'] = f'{base_name}.plt'
        env['FOR026'] = f'{base_name}.chk'     # Checkpoint file (per oasn wrapper)
        env['FOR028'] = f'{base_name}.028'
        env['FOR029'] = f'{base_name}.029'
        env['FOR045'] = f'{base_name}.045'

        result = self._run_subprocess(
            [str(self.executable)], cwd=work_dir, timeout=300, env=env,
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
        volume_attenuation: Optional[str] = None,
        francois_garrison_params: Optional[tuple] = None,
        bio_layers: Optional[list] = None,
        use_tmpfs: bool = False,
        verbose: bool = False,
        work_dir: Optional[Path] = None,
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
        volume_attenuation : str, optional
            'T' (Thorp), 'F' (Francois-Garrison), 'B' (Biological). Default: None.
        francois_garrison_params : tuple, optional
            (T, S, pH, z_bar) required when ``volume_attenuation='F'``.
        bio_layers : list, optional
            [(Z1, Z2, f0, Q, a0), ...] required when ``volume_attenuation='B'``.
        """
        super().__init__(use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir)
        self.angles = angles
        self.angle_type = angle_type
        self.volume_attenuation = volume_attenuation
        self.francois_garrison_params = francois_garrison_params
        self.bio_layers = bio_layers
        self._validate_volume_attenuation_params()

        # OASR is strictly a boundary-reflection solver; it does not produce
        # transmission loss. Declare that explicitly.
        self._supported_modes = [RunMode.REFLECTION]
        # OASR: range-independent reflection vs angle/freq; layered bottom honored.
        self._supports_layered_bottom = True

        if executable is None:
            self.executable = self._find_executable('oasr')
        else:
            self.executable = Path(executable)

        if not self.executable.exists():
            raise FileNotFoundError(
                f"OASR executable not found: {self.executable}\n"
                "Please compile OASES using the installation scripts."
            )

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[RunMode] = None,
        *,
        angles=_UNSET,
        angle_type=_UNSET,
        volume_attenuation=_UNSET,
        **kwargs
    ) -> Result:
        """
        Run OASR reflection coefficient computation.

        OASES uses grazing angles natively. When ``angle_type='incidence'``,
        input angles are converted via ``grazing = 90 - incidence`` before
        being written to the OASR input file.

        Parameters
        ----------
        env : Environment
            Ocean environment
        source : Source
            Acoustic source
        receiver : Receiver
            Receiver array
        angles, angle_type, volume_attenuation : optional
            Per-call overrides for constructor defaults.

        Returns
        -------
        result : Result
            Reflection coefficients field
        """
        if run_mode is not None and run_mode != RunMode.REFLECTION:
            from uacpy.core.exceptions import UnsupportedFeatureError
            raise UnsupportedFeatureError(
                self.model_name, str(run_mode),
                alternatives=["RunMode.REFLECTION"],
            )

        # Resolve per-call overrides
        angles = angles if angles is not _UNSET else self.angles
        angle_type = angle_type if angle_type is not _UNSET else self.angle_type
        volume_attenuation = volume_attenuation if volume_attenuation is not _UNSET else self.volume_attenuation

        self.validate_inputs(env, source, receiver)

        angles = angles if angles is not None else np.linspace(0, 90, 181)

        fm = self._setup_file_manager()

        try:
            base_name = 'oasr_run'
            input_file = fm.get_path(f'{base_name}.dat')

            # Write input file
            self._log(f"Writing OASR input file: {input_file}")
            write_oasr_input(
                filepath=input_file,
                env=env,
                source=source,
                receiver=receiver,
                angles=angles,
                angle_type=angle_type,
                volume_attenuation=volume_attenuation,
                francois_garrison_params=self.francois_garrison_params,
                bio_layers=self.bio_layers,
                **kwargs
            )

            # Run executable
            self._execute(input_file, fm.work_dir)

            # Read output - check multiple possible file extensions
            # OASR can output to .rco, .trc, or fort.023 depending on options
            output_file = None
            for ext in ['.rco', '.trc', '.023', 'fort.023']:
                candidate = fm.get_path(f'{base_name}{ext}') if not ext.startswith('fort') else fm.get_path(ext)
                if candidate.exists() and candidate.stat().st_size > 0:
                    output_file = candidate
                    break

            if output_file is None:
                raise FileNotFoundError(
                    f"OASR output file not found. Checked: {base_name}.rco, {base_name}.trc, {base_name}.023"
                )

            self._log(f"Reading OASR output: {output_file}")
            data = read_oasr_reflection_coefficients(output_file)

            # Build a typed ReflectionCoefficient. OASR stashed everything in
            # ``data`` (theta, R, phi, …); we lift the spectrum-defining keys
            # to typed attributes and pass the rest as metadata.
            theta = data.pop('theta', np.array([]))
            R = data.pop('R', np.array([]))
            phi = data.pop('phi', np.zeros_like(np.atleast_1d(theta)))
            field = ReflectionCoefficient(
                theta=theta, R=R, phi=phi,
                model='OASR',
                backend='oasr',
                source_depths=np.atleast_1d(np.asarray(source.depth, dtype=float)),
                frequency=float(np.atleast_1d(source.frequency)[0]),
                metadata=data,
            )

            self._log("OASR simulation complete")
            return field

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _execute(self, input_file: Path, work_dir: Path):
        """Execute OASR binary (FOR005 left as stdin per OASES docs)."""
        import os
        base_name = input_file.stem
        env = os.environ.copy()
        env['FOR001'] = f'{base_name}.dat'
        env['FOR002'] = f'{base_name}.src'
        env['FOR004'] = f'{base_name}.trf'
        env['FOR019'] = f'{base_name}.plp'
        env['FOR020'] = f'{base_name}.plt'
        env['FOR022'] = f'{base_name}.rco'   # Reflection-coef table (slowness)
        env['FOR023'] = f'{base_name}.trc'   # Reflection-coef table (angle)
        env['FOR028'] = f'{base_name}.028'
        env['FOR029'] = f'{base_name}.029'
        env['FOR045'] = f'{base_name}.045'

        result = self._run_subprocess(
            [str(self.executable)], cwd=work_dir, timeout=300, env=env,
        )
        if self.verbose and result.stdout:
            self._log(f"OASES output:\n{result.stdout}", level='debug')

    def _compute_tl_impl(self, env, source, receiver, **kwargs):
        """OASR does not compute transmission loss — raise UnsupportedFeatureError."""
        from uacpy.core.exceptions import UnsupportedFeatureError
        raise UnsupportedFeatureError(
            model_name='OASR',
            feature='transmission loss (computes plane-wave reflection coefficients instead)',
            alternatives=['OAST (wavenumber integration)', 'OASP (parabolic equation)'],
        )


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
        super().__init__(use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir)
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
        self._unsupported_env_alternatives = "RAM (PE) for full range-dep"

        if executable is None:
            self.executable = self._find_executable('oasp')
        else:
            self.executable = Path(executable)

        if not self.executable.exists():
            raise FileNotFoundError(
                f"OASP executable not found: {self.executable}\n"
                "Please compile OASES using the installation scripts."
            )

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[RunMode] = None,
        n_time_samples=_UNSET,
        freq_max=_UNSET,
        volume_attenuation=_UNSET,
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
            ``field_type='tl'`` for COHERENT_TL,
            ``'transfer_function'`` for BROADBAND,
            ``'time_series'`` for TIME_SERIES.
        """
        # Resolve per-call overrides
        n_time_samples = n_time_samples if n_time_samples is not _UNSET else self.n_time_samples
        freq_max = freq_max if freq_max is not _UNSET else self.freq_max
        volume_attenuation = volume_attenuation if volume_attenuation is not _UNSET else self.volume_attenuation

        if run_mode is None:
            run_mode = RunMode.COHERENT_TL

        if run_mode == RunMode.TIME_SERIES and (
            source_waveform is None or sample_rate is None
        ):
            raise ValueError(
                "OASP.run(run_mode=TIME_SERIES) requires source_waveform "
                "and sample_rate. For the broadband transfer function "
                "H(f), use run_mode=RunMode.BROADBAND."
            )
        self.validate_inputs(env, source, receiver)
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
                    freq_diff = np.abs(trf_data['freq'] - source.frequency[0])
                    freq_idx = np.argmin(freq_diff)

                tl_at_freq = transfer_func[freq_idx, :, :]
                magnitude = np.abs(tl_at_freq)
                magnitude[magnitude == 0] = 1e-30
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
                        frequency=float(trf_data['freq'][freq_idx]),
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
        """Execute OASP binary (FOR005 left as stdin per OASES docs).

        Canonical env-var mapping follows the ``oasp`` csh wrapper in
        third_party/oases/bin/oasp. FOR003/FOR004 were decorative — no OASES
        Fortran source references those unit numbers via getenv. The real TRF
        output comes from OPFILW(unit, ...); the ``.trf`` file is produced by
        that path alongside a standard base-name-derived file name when the
        environment variable is unset.
        """
        import os
        base_name = input_file.stem
        env = os.environ.copy()
        env['FOR001'] = f'{base_name}.dat'
        env['FOR002'] = f'{base_name}.src'
        env['FOR019'] = f'{base_name}.plp'
        env['FOR020'] = f'{base_name}.plt'
        env['FOR028'] = f'{base_name}.028'
        env['FOR029'] = f'{base_name}.029'
        env['FOR045'] = f'{base_name}.045'

        result = self._run_subprocess(
            [str(self.executable)], cwd=work_dir, timeout=300, env=env,
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
    >>> # Normal modes (uses OASN)
    >>> modes = oases.compute_modes(env, source)
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
        use_tmpfs: bool = False,
        verbose: bool = False,
        work_dir: Optional[Path] = None,
    ):
        super().__init__(use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir)

        # Create instances of all OASES models
        self._oast = OAST(use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir)
        self._oasn = OASN(use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir)
        self._oasr = OASR(use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir)
        self._oasp = OASP(use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir)

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
            return self._oasn.run(env, source, receiver, **kwargs)
        if run_mode == RunMode.REFLECTION:
            return self._oasr.run(env, source, receiver, **kwargs)
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
        receiver: Receiver = None,
        broadband: bool = False,
        **kwargs
    ) -> Result:
        """
        Compute transmission loss using OAST (wavenumber integration) or OASP
        (wideband transfer function) when ``broadband=True``.

        Parameters
        ----------
        env : Environment
            Ocean environment
        source : Source
            Acoustic source
        receiver : Receiver, optional
            Receiver array
        broadband : bool, optional
            When True, delegate to OASP instead of OAST. Needed for
            range-dependent environments where OAST's range-independent
            wavenumber-integration kernel is inappropriate. Default False.
        **kwargs
            Additional OAST / OASP parameters.

        Returns
        -------
        result : Result
            Transmission loss field.
        """
        if receiver is None:
            from uacpy.core.model_utils import ReceiverGridBuilder
            max_range = kwargs.pop('max_range', 10000.0)
            depths, ranges = ReceiverGridBuilder.build_tl_grid(env.depth, max_range)
            receiver = Receiver(depths=depths, ranges=ranges)
        if broadband:
            return self._oasp.run(env, source, receiver, **kwargs)
        return self._oast.run(env, source, receiver, **kwargs)

    def compute_modes(
        self,
        env: Environment,
        source: Source,
        n_modes: Optional[int] = None,
        **kwargs
    ) -> Result:
        """
        Compute OASN covariance / replica field.

        .. note::
            OASN does not compute explicit mode shapes. ``n_modes`` is
            accepted for API symmetry with other model wrappers but has
            no effect inside OASN (the Fortran has no mode-truncation
            parameter). For true modal analysis use Kraken/KrakenC.

        Parameters
        ----------
        env : Environment
            Ocean environment.
        source : Source
            Acoustic source.
        n_modes : int, optional
            Ignored (see note). Accepted for API compatibility only.
        **kwargs
            Additional OASN parameters.

        Returns
        -------
        result : Result
            Covariance / replica field.
        """
        # n_modes is deliberately dropped — OASN has no mode-truncation knob.
        _ = n_modes
        # OASN needs a receiver to validate — synthesize a single-depth one if
        # the caller didn't provide one (compute_modes in the base class does
        # the same trick).
        dummy_receiver = Receiver(depths=[float(source.depth[0])], ranges=[1.0])
        return self._oasn.run(env, source, dummy_receiver, **kwargs)

    def compute_reflection(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        angles: Optional[np.ndarray] = None,
        angle_type: str = 'grazing',
        **kwargs
    ) -> Result:
        """
        Compute reflection coefficients using OASR.

        Parameters
        ----------
        env : Environment
            Ocean environment.
        source : Source
            Acoustic source.
        receiver : Receiver
            Receiver array.
        angles : array_like, optional
            Angles to compute (degrees). Interpretation depends on
            ``angle_type``.
        angle_type : str, optional
            'grazing' (OASES native) or 'incidence'. When 'incidence' is
            passed, ``OASR.run`` converts via ``grazing = 90 - incidence``
            before writing the OASR input file. Default 'grazing'.
        **kwargs
            Additional OASR parameters.

        Returns
        -------
        result : Result
            Reflection coefficients field.
        """
        return self._oasr.run(
            env, source, receiver,
            angles=angles, angle_type=angle_type, **kwargs,
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
        """
        Compute time-domain pressure p(t) using OASP.

        Internally runs the OASP broadband H(f) and convolves with the
        given source waveform.

        Parameters
        ----------
        env, source, receiver : see :class:`OASP`.
        source_waveform : ndarray
            1-D source pulse.
        sample_rate : float
            Source-waveform sampling rate (Hz).

        Returns
        -------
        result : Result
            ``field_type='time_series'``.
        """
        return self._oasp.run(
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
        **kwargs
    ) -> Result:
        """
        Compute broadband transfer function H(f) via OASP.

        Parameters
        ----------
        env : Environment
            Ocean environment.
        source : Source
            Acoustic source.
        receiver : Receiver
            Receiver array.
        **kwargs
            Additional OASP parameters.

        Returns
        -------
        result : Result
            ``field_type='transfer_function'``.
        """
        kwargs.pop('run_mode', None)
        return self._oasp.run(
            env, source, receiver,
            run_mode=RunMode.BROADBAND, **kwargs,
        )
