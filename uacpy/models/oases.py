"""
OASES - Ocean Acoustics and Seismic Exploration Synthesis

OASES is a comprehensive acoustic/seismic propagation modeling suite developed
by Henrik Schmidt at MIT. It includes multiple executables for different scenarios:

- **OAST**: Transmission Loss via wavenumber integration
- **OASN**: Normal modes extraction
- **OASR**: Reflection coefficients
- **OASP**: Parabolic equation for range-dependent/broadband problems

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

import subprocess
import numpy as np
from pathlib import Path
from typing import Optional
import warnings

from uacpy.models.base import PropagationModel, RunMode, _UNSET
from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.field import Field
from uacpy.io.oases_writer import write_oast_input, write_oasn_input, write_oasp_input, write_oasr_input
from uacpy.io.oases_reader import read_oast_tl, read_oases_modes, read_oasp_trf, read_oasr_reflection_coefficients


class _OASESBase(PropagationModel):
    """Base class for all OASES models with shared functionality"""

    def _find_executable(self, name: str) -> Path:
        """Find OASES executable in standard locations"""
        # Try bash version first (more compatible than csh)
        bash_name = f"{name}_bash"

        # Check in uacpy/bin/oases or bin/oalib
        for subdir in ['oases', 'oalib']:
            bash_path = Path(__file__).parent.parent / 'bin' / subdir / bash_name
            if bash_path.exists():
                return bash_path
            bin_path = Path(__file__).parent.parent / 'bin' / subdir / name
            if bin_path.exists():
                return bin_path

        # Check in third_party/oases/bin (development location)
        bash_dev_path = Path(__file__).parent.parent / 'third_party' / 'oases' / 'bin' / bash_name
        if bash_dev_path.exists():
            return bash_dev_path

        dev_path = Path(__file__).parent.parent / 'third_party' / 'oases' / 'bin' / name
        if dev_path.exists():
            return dev_path

        # Check in PATH
        import shutil
        result = shutil.which(bash_name)
        if result:
            return Path(result)
        result = shutil.which(name)
        if result:
            return Path(result)

        raise FileNotFoundError(
            f"Could not find {name} executable.\n"
            "Please run ./install.sh to download and compile OASES."
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
        use_tmpfs: bool = False,
        verbose: bool = False,
        work_dir: Optional[Path] = None,
    ):
        super().__init__(use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir)
        self.volume_attenuation = volume_attenuation

        self._supported_modes = [RunMode.COHERENT_TL]

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
        volume_attenuation=_UNSET,
        **kwargs
    ) -> Field:
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
        field : Field
            Transmission loss field
        """
        # Resolve per-call overrides
        volume_attenuation = volume_attenuation if volume_attenuation is not _UNSET else self.volume_attenuation

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
                **kwargs
            )

            # Run executable
            self._execute(input_file, fm.work_dir)

            # Read output - OAST creates .plt (data) and .plp (metadata) files
            # Per OASES documentation: FOR019 -> .plp, FOR020 -> .plt
            plt_file = fm.get_path(f'{base_name}.plt')
            plp_file = fm.get_path(f'{base_name}.plp')

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

            # Package as Field object
            result = Field(
                field_type='tl',
                data=tl_data,
                depths=receiver.depths,
                ranges=receiver.ranges,
                metadata=metadata
            )

            self._log("OAST simulation complete")
            return result

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _execute(self, input_file: Path, work_dir: Path):
        """Execute OAST binary"""
        try:
            base_name = input_file.stem

            # OASES uses Fortran unit environment variables to specify I/O files
            # Per OASES documentation (doc/oast.tex lines 586-593)
            import os
            env = os.environ.copy()
            env['FOR001'] = f'{base_name}.dat'  # Input file
            env['FOR002'] = f'{base_name}.src'  # Source file
            env['FOR003'] = f'{base_name}.cdr'  # Complex depth response
            env['FOR004'] = f'{base_name}.trf'  # Transfer function
            env['FOR005'] = f'{base_name}.plt'  # Plot file (main TL output)
            env['FOR019'] = f'{base_name}.plp'  # Plot parameter file (grid metadata)
            env['FOR020'] = f'{base_name}.plt'  # Plot data file (TL values)

            result = subprocess.run(
                [str(self.executable)],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=300,
                env=env
            )

            if result.returncode != 0:
                error_msg = f"OAST failed with return code {result.returncode}\n"
                error_msg += f"stdout: {result.stdout}\n"
                error_msg += f"stderr: {result.stderr}"
                raise RuntimeError(error_msg)

            if self.verbose and result.stdout:
                self._log(f"OASES output:\n{result.stdout}", level='debug')

        except subprocess.TimeoutExpired:
            raise RuntimeError("OAST execution timed out after 5 minutes")


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
        use_tmpfs: bool = False,
        verbose: bool = False,
        work_dir: Optional[Path] = None,
    ):
        super().__init__(use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir)
        self.volume_attenuation = volume_attenuation

        self._supported_modes = [RunMode.MODES]

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
        n_modes: Optional[int] = None,
        volume_attenuation=_UNSET,
        **kwargs
    ) -> Field:
        """
        Run OASN normal mode computation.

        .. note::
            Mode file reading from OASN is experimental. For production
            normal mode analysis, consider using Kraken or KrakenC instead.

        Parameters
        ----------
        env : Environment
            Ocean environment (must be range-independent).
        source : Source
            Acoustic source (for frequency).
        receiver : Receiver
            Receiver array.
        n_modes : int, optional
            Number of modes to compute. If None, computes all.
        volume_attenuation : str, optional
            Per-call override for constructor default.
        **kwargs
            Additional OASN parameters.

        Returns
        -------
        field : Field
            Normal modes field.
        """
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
                n_modes=n_modes,
                **kwargs
            )

            # Run executable
            self._execute(base_name, fm.work_dir)

            # Read mode file - OASN creates XSM on unit 16
            # Check multiple possible locations
            mode_file = fm.get_path(f'{base_name}.xsm')
            fort16_file = fm.get_path('fort.16')

            if mode_file.exists():
                self._log(f"Reading OASN mode file: {mode_file}")
                modes = read_oases_modes(mode_file)
            elif fort16_file.exists():
                # gfortran may create fort.16 instead of respecting FOR016
                self._log(f"Reading OASN mode file: {fort16_file}")
                modes = read_oases_modes(fort16_file)
            else:
                raise FileNotFoundError(
                    f"OASN mode file not found. Checked: {mode_file}, {fort16_file}\n\n"
                    "Consider using Kraken for more reliable mode computation."
                )

            if not modes or modes.get('n_modes', 0) == 0:
                self._log(
                    "OASN mode file reader is experimental and returned empty modes. "
                    "Consider using Kraken for reliable normal mode analysis",
                    level='warn'
                )

            # Package as Field object
            receiver_depths = receiver.depths if hasattr(receiver, 'depths') else np.array([source.depth[0]])
            return Field(
                field_type='modes',
                data=modes.get('phi', np.array([])),
                depths=modes.get('z', receiver_depths),
                metadata={
                    'model': 'OASN',
                    'frequency': source.frequency[0],
                    'n_modes': modes.get('n_modes', 0),
                    'wavenumbers': modes.get('k', np.array([])),
                    'experimental': True,
                }
            )

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _execute(self, base_name: str, work_dir: Path):
        """Execute OASN binary"""
        try:
            # OASES uses Fortran unit environment variables to specify I/O files
            import os
            env = os.environ.copy()
            env['FOR001'] = f'{base_name}.dat'  # Input file
            env['FOR002'] = f'{base_name}.src'  # Source file
            env['FOR016'] = f'{base_name}.xsm'  # Mode file (XSM on unit 16)
            env['FOR004'] = f'{base_name}.trf'  # Transfer function
            env['FOR005'] = f'{base_name}.plt'  # Plot file
            env['FOR019'] = f'{base_name}.019'  # Optional output
            env['FOR020'] = f'{base_name}.020'  # Optional output
            env['FOR021'] = f'{base_name}.021'  # Optional output
            env['FOR026'] = f'{base_name}.026'  # Replica vectors

            result = subprocess.run(
                [str(self.executable)],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=300,
                env=env
            )

            if result.returncode != 0:
                error_msg = f"OASN failed with return code {result.returncode}\n"
                error_msg += f"stdout: {result.stdout}\n"
                error_msg += f"stderr: {result.stderr}"
                raise RuntimeError(error_msg)

            if self.verbose and result.stdout:
                self._log(f"OASES output:\n{result.stdout}", level='debug')

        except subprocess.TimeoutExpired:
            raise RuntimeError("OASN execution timed out after 5 minutes")


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
            'grazing' or 'incidence'. Default: 'grazing'.
        volume_attenuation : str, optional
            'T' (Thorp), 'F' (Francois-Garrison), 'B' (Biological). Default: None.
        """
        super().__init__(use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir)
        self.angles = angles
        self.angle_type = angle_type
        self.volume_attenuation = volume_attenuation

        self._supported_modes = [RunMode.COHERENT_TL]

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
        angles=_UNSET,
        angle_type=_UNSET,
        volume_attenuation=_UNSET,
        **kwargs
    ) -> Field:
        """
        Run OASR reflection coefficient computation

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
        field : Field
            Reflection coefficients field
        """
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

            # Convert dict to Field object
            from uacpy.core.field import Field
            field = Field(
                field_type='reflection_coefficients',
                data=None,  # Reflection coefficients stored in metadata
                metadata=data
            )

            self._log("OASR simulation complete")
            return field

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _execute(self, input_file: Path, work_dir: Path):
        """Execute OASR binary"""
        try:
            base_name = input_file.stem

            # OASES uses Fortran unit environment variables to specify I/O files
            import os
            env = os.environ.copy()
            env['FOR001'] = f'{base_name}.dat'  # Input file
            env['FOR002'] = f'{base_name}.src'  # Source file
            env['FOR003'] = f'{base_name}.rco'  # Reflection coefficient (slowness)
            env['FOR004'] = f'{base_name}.trf'  # Transfer function
            env['FOR005'] = f'{base_name}.plt'  # Plot file
            env['FOR019'] = f'{base_name}.019'  # Optional output
            env['FOR020'] = f'{base_name}.020'  # Optional output
            env['FOR022'] = f'{base_name}.rco'  # Reflection coefficient table (slowness)
            env['FOR023'] = f'{base_name}.trc'  # Reflection coefficient table (angle)

            result = subprocess.run(
                [str(self.executable)],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=300,
                env=env
            )

            if result.returncode != 0:
                error_msg = f"OASR failed with return code {result.returncode}\n"
                error_msg += f"stdout: {result.stdout}\n"
                error_msg += f"stderr: {result.stderr}"
                raise RuntimeError(error_msg)

            if self.verbose and result.stdout:
                self._log(f"OASES output:\n{result.stdout}", level='debug')

        except subprocess.TimeoutExpired:
            raise RuntimeError("OASR execution timed out after 5 minutes")


class OASP(_OASESBase):
    """
    OASP - OASES Parabolic Equation Model

    Computes broadband acoustic propagation using parabolic equation method.
    Supports range-dependent environments.

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
        """
        super().__init__(use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir)
        self.n_time_samples = n_time_samples
        self.freq_max = freq_max
        self.volume_attenuation = volume_attenuation

        self._supported_modes = [RunMode.COHERENT_TL, RunMode.TIME_SERIES]

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
        **kwargs
    ) -> Field:
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
            COHERENT_TL (default): extract TL at source frequency.
            TIME_SERIES: return full broadband transfer function.
        n_time_samples, freq_max, volume_attenuation : optional
            Per-call overrides for constructor defaults.
        **kwargs
            Additional OASP parameters

        Returns
        -------
        field : Field
            TL field (COHERENT_TL) or transfer function field (TIME_SERIES)
        """
        # Resolve per-call overrides
        n_time_samples = n_time_samples if n_time_samples is not _UNSET else self.n_time_samples
        freq_max = freq_max if freq_max is not _UNSET else self.freq_max
        volume_attenuation = volume_attenuation if volume_attenuation is not _UNSET else self.volume_attenuation

        if run_mode is None:
            run_mode = RunMode.COHERENT_TL
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

            if run_mode == RunMode.TIME_SERIES:
                # Return full broadband transfer function
                # Reshape to (n_depth, n_freq, n_range) for consistency with RAM
                tf_reordered = np.transpose(transfer_func, (2, 0, 1))

                result = Field(
                    field_type='transfer_function',
                    data=tf_reordered,
                    depths=trf_data['depths'],
                    ranges=trf_data['ranges'],
                    metadata={
                        'model': 'OASP',
                        'frequencies': trf_data['freq'],
                        'source_depth': trf_data['source_depth'],
                        'center_frequency': trf_data['center_frequency'],
                        'n_time_samples': n_time_samples,
                        'freq_max': freq_max,
                    }
                )
            else:
                # COHERENT_TL: extract TL at source frequency
                freq_idx = 0
                if len(trf_data['freq']) > 1:
                    freq_diff = np.abs(trf_data['freq'] - source.frequency[0])
                    freq_idx = np.argmin(freq_diff)

                tl_at_freq = transfer_func[freq_idx, :, :]

                magnitude = np.abs(tl_at_freq)
                magnitude[magnitude == 0] = 1e-30
                tl_db = -20 * np.log10(magnitude)

                # Transpose to (n_depth, n_range)
                tl_db = tl_db.T

                result = Field(
                    field_type='tl',
                    data=tl_db,
                    depths=trf_data['depths'],
                    ranges=trf_data['ranges'],
                    metadata={
                        'model': 'OASP',
                        'frequency': trf_data['freq'][freq_idx],
                        'frequencies_available': trf_data['freq'],
                        'source_depth': trf_data['source_depth'],
                        'center_frequency': trf_data['center_frequency'],
                        'n_time_samples': n_time_samples,
                        'freq_max': freq_max,
                    }
                )

            self._log("OASP simulation complete")
            return result

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _execute(self, input_file: Path, work_dir: Path):
        """Execute OASP binary"""
        try:
            base_name = input_file.stem

            # OASES uses Fortran unit environment variables to specify I/O files
            import os
            env = os.environ.copy()
            env['FOR001'] = f'{base_name}.dat'  # Input file
            env['FOR002'] = f'{base_name}.src'  # Source file
            env['FOR003'] = f'{base_name}.cdr'  # Complex depth response
            env['FOR004'] = f'{base_name}.trf'  # Transfer function (main output)
            env['FOR005'] = f'{base_name}.plt'  # Plot file
            env['FOR019'] = f'{base_name}.019'  # Optional output
            env['FOR020'] = f'{base_name}.020'  # Optional output

            result = subprocess.run(
                [str(self.executable)],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=300,
                env=env
            )

            if result.returncode != 0:
                error_msg = f"OASP failed with return code {result.returncode}\n"
                error_msg += f"stdout: {result.stdout}\n"
                error_msg += f"stderr: {result.stderr}"
                raise RuntimeError(error_msg)

            if self.verbose and result.stdout:
                self._log(f"OASES output:\n{result.stdout}", level='debug')

        except subprocess.TimeoutExpired:
            raise RuntimeError("OASP execution timed out after 5 minutes")


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
    ) -> Field:
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

            use_pe : bool, optional
                When True and the environment is range-dependent, delegate
                to OASP (parabolic equation) instead of OAST for TL
                computation. Default False.

        Returns
        -------
        field : Field
            Simulation results.
        """
        if run_mode == RunMode.COHERENT_TL:
            # Use OASP for range-dependent PE, OAST otherwise
            if env.is_range_dependent and kwargs.get('use_pe', False):
                return self._oasp.run(env, source, receiver, **kwargs)
            else:
                return self._oast.run(env, source, receiver, **kwargs)
        elif run_mode == RunMode.MODES:
            return self._oasn.run(env, source, receiver, **kwargs)
        else:
            raise ValueError(f"Run mode {run_mode} not supported by OASES")

    def compute_tl(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver = None,
        **kwargs
    ) -> Field:
        """
        Compute transmission loss using OAST

        This method uses OAST (wavenumber integration) for TL computation.

        Parameters
        ----------
        env : Environment
            Ocean environment
        source : Source
            Acoustic source
        receiver : Receiver, optional
            Receiver array
        **kwargs
            Additional OAST parameters

        Returns
        -------
        field : Field
            Transmission loss field
        """
        return self._oast.compute_tl(env, source, receiver, **kwargs)

    def compute_modes(
        self,
        env: Environment,
        source: Source,
        n_modes: Optional[int] = None,
        **kwargs
    ) -> Field:
        """
        Compute normal modes using OASN

        Parameters
        ----------
        env : Environment
            Ocean environment
        source : Source
            Acoustic source
        n_modes : int, optional
            Number of modes to compute
        **kwargs
            Additional OASN parameters

        Returns
        -------
        field : Field
            Normal modes field
        """
        return self._oasn.compute_modes(env, source, n_modes=n_modes, **kwargs)

    def compute_reflection(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        angles: Optional[np.ndarray] = None,
        **kwargs
    ) -> Field:
        """
        Compute reflection coefficients using OASR

        Parameters
        ----------
        env : Environment
            Ocean environment
        source : Source
            Acoustic source
        receiver : Receiver
            Receiver array
        angles : array_like, optional
            Angles to compute (degrees)
        **kwargs
            Additional OASR parameters

        Returns
        -------
        field : Field
            Reflection coefficients field
        """
        return self._oasr.run(env, source, receiver, angles=angles, **kwargs)

    def compute_pe(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        **kwargs
    ) -> Field:
        """
        Compute using parabolic equation (OASP)

        Parameters
        ----------
        env : Environment
            Ocean environment
        source : Source
            Acoustic source
        receiver : Receiver
            Receiver array
        **kwargs
            Additional OASP parameters

        Returns
        -------
        field : Field
            Parabolic equation field
        """
        return self._oasp.run(env, source, receiver, **kwargs)
