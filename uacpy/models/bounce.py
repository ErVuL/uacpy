"""
BOUNCE - Reflection Coefficient Computation Module

BOUNCE computes reflection coefficients for a stack of acoustic/elastic layers.
Part of the Acoustics Toolbox (OALIB).

Outputs:
- .BRC file: Bottom Reflection Coefficient
  → Used by: BELLHOP, SCOOTER, KRAKENC (experimental)
- .IRC file: Internal Reflection Coefficient
  → Used by: KRAKEN (NOT KRAKENC - use .BRC with KRAKENC)

Note: SPARC does not support reflection coefficient files.
"""

import subprocess
import numpy as np
from pathlib import Path
from typing import Optional

from uacpy.models.base import PropagationModel, RunMode, _UNSET
from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.field import Field
from uacpy.core.constants import (
    DEFAULT_SOUND_SPEED, DEFAULT_C_MIN, DEFAULT_C_MAX,
    AttenuationUnits, parse_ssp_type, parse_boundary_type,
)
from uacpy.io.output_reader import read_reflection_coefficient
from uacpy.io.at_env_writer import ATEnvWriter


class Bounce(PropagationModel):
    """
    BOUNCE - Reflection Coefficient Model (Acoustics Toolbox)

    Computes plane wave reflection coefficients for a stack of acoustic/elastic
    layers. The reflection coefficient is written to both .BRC (Bottom Reflection
    Coefficient) and .IRC (Internal Reflection Coefficient) files.

    Model Support:
    - .BRC files: BELLHOP ✓, SCOOTER ✓, KRAKENC ⚠️ (experimental)
    - .IRC files: KRAKEN ✓ (uses internal reflection coefficient)
    - SPARC: ❌ (does not support reflection files)

    Parameters
    ----------
    executable : Path, optional
        Path to bounce executable. If None, searches standard locations.
    use_tmpfs : bool, optional
        Use tmpfs for I/O operations. Default False.
    verbose : bool, optional
        Print detailed execution information. Default False.
    work_dir : Path, optional
        Working directory for I/O files. Default uses temp directory.

    Examples
    --------
    Compute reflection coefficients for use in other models:

    >>> from uacpy.models import Bounce
    >>> from uacpy.core import Environment, Source, Receiver, BoundaryProperties
    >>> import numpy as np
    >>>
    >>> # Define environment with elastic bottom
    >>> bottom = BoundaryProperties(
    ...     acoustic_type='half-space',
    ...     sound_speed=1600,
    ...     shear_speed=400,
    ...     density=1.8,
    ...     attenuation=0.2,
    ...     shear_attenuation=0.5
    ... )
    >>> env = Environment(name="test", depth=100, bottom=bottom)
    >>> source = Source(depth=50, frequency=50)
    >>> receiver = Receiver(depths=np.array([50]))
    >>>
    >>> # Compute reflection coefficients
    >>> bounce = Bounce()
    >>> result = bounce.run(env, source, receiver,
    ...                     cmin=1400, cmax=10000, rmax_km=10)
    >>>
    >>> # Output files can be used by different models:
    >>> # - .brc file → Use with BELLHOP, SCOOTER, KRAKENC (experimental)
    >>> # - .irc file → Use with KRAKEN
    >>>
    >>> # Example: Use .brc with SCOOTER
    >>> from uacpy.models import Scooter
    >>> bottom_with_rc = BoundaryProperties(
    ...     acoustic_type='file',
    ...     reflection_file=result.metadata['brc_file']
    ... )
    >>> env_with_rc = Environment(name="test", depth=100, bottom=bottom_with_rc)
    >>> scooter = Scooter()
    >>> tl = scooter.compute_tl(env_with_rc, source, receiver)

    Notes
    -----
    - BOUNCE uses the same environmental file format as KRAKEN
    - The reflection coefficient depends on impedance contrast
    - Supports acoustic, elastic, and poro-elastic layers
    - Tabulated reflection coefficients cover angles from phase velocities [cmin, cmax]
    - **Recommended workflow**: BOUNCE → .brc → SCOOTER (most reliable)
    - KRAKENC support for .brc is experimental and may fail
    - For KRAKEN, use .irc files (internal reflection coefficient)

    References
    ----------
    - Porter, M.B., "The KRAKEN Normal Mode Program", SACLANT Undersea Research
      Centre Memorandum SM-245, 1991
    - Acoustics Toolbox: http://oalib.hlsresearch.com/
    """

    def __init__(
        self,
        executable: Optional[Path] = None,
        cmin: float = DEFAULT_C_MIN,
        cmax: float = DEFAULT_C_MAX,
        rmax_km: float = 10.0,
        volume_attenuation: Optional[str] = None,
        use_tmpfs: bool = False,
        verbose: bool = False,
        work_dir: Optional[Path] = None,
    ):
        """
        Parameters
        ----------
        executable : Path, optional
            Path to bounce executable. Auto-detected if None.
        cmin : float, optional
            Minimum phase velocity (m/s) for tabulation. Default: 1400.
        cmax : float, optional
            Maximum phase velocity (m/s) for tabulation. Default: 10000.
        rmax_km : float, optional
            Maximum range (km) for angular sampling. Default: 10.
        volume_attenuation : str, optional
            'T' (Thorp), 'F' (Francois-Garrison), 'B' (Biological). Default: None.
        """
        super().__init__(use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir)

        self.cmin = cmin
        self.cmax = cmax
        self.rmax_km = rmax_km
        self.volume_attenuation = volume_attenuation

        self._supported_modes = [RunMode.COHERENT_TL]

        if executable is None:
            self.executable = self._find_executable('bounce')
        else:
            self.executable = Path(executable)

        if not self.executable.exists():
            raise FileNotFoundError(
                f"BOUNCE executable not found: {self.executable}\n"
                "Please compile the Acoustics Toolbox:\n"
                "cd uacpy/third_party/Acoustics-Toolbox && make bounce"
            )

    def _find_executable(self, name: str) -> Path:
        """Find BOUNCE executable in standard locations"""
        # Try with .exe suffix first, then without
        import shutil
        for exe_name in [f'{name}.exe', name]:
            bin_path = Path(__file__).parent.parent / 'bin' / 'oalib' / exe_name
            if bin_path.exists():
                return bin_path
            result = shutil.which(exe_name)
            if result:
                return Path(result)

        raise FileNotFoundError(
            f"Could not find {name} executable. "
            "Run install.sh to compile Acoustics Toolbox."
        )

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        output_brc: bool = True,
        output_irc: bool = True,
        cmin=_UNSET,
        cmax=_UNSET,
        rmax_km=_UNSET,
        volume_attenuation=_UNSET,
        **kwargs
    ) -> Field:
        """
        Run BOUNCE reflection coefficient computation

        Parameters
        ----------
        env : Environment
            Ocean environment (bottom properties define the stack)
        source : Source
            Source definition (frequency is used)
        receiver : Receiver
            Receiver definition (not used by BOUNCE, but required for API)
        output_brc : bool, optional
            Output .BRC file (for BELLHOP/SCOOTER). Default True.
        output_irc : bool, optional
            Output .IRC file (for KRAKEN). Default True.
        **kwargs
            Additional parameters passed to ENV file writer

        Returns
        -------
        field : Field
            Field containing reflection coefficient data with:
            - angles: grazing angles (degrees)
            - R_magnitude: reflection coefficient magnitudes
            - R_phase: reflection coefficient phases (degrees)
            - brc_file: path to .BRC file (if output_brc=True)
            - irc_file: path to .IRC file (if output_irc=True)

        Notes
        -----
        - For full 90-degree coverage, set cmin=1400, cmax=1e9
        - Larger rmax_km gives finer angular resolution
        - The output .BRC file can be used directly by other models
        """
        # Resolve per-call overrides
        cmin = cmin if cmin is not _UNSET else self.cmin
        cmax = cmax if cmax is not _UNSET else self.cmax
        rmax_km = rmax_km if rmax_km is not _UNSET else self.rmax_km
        volume_attenuation = volume_attenuation if volume_attenuation is not _UNSET else self.volume_attenuation

        self.validate_inputs(env, source, receiver)

        # BOUNCE produces output files (.brc, .irc) that need to persist
        # So we disable cleanup if work_dir is not explicitly set
        if self.work_dir is None:
            # Use a persistent temp directory
            import tempfile
            work_dir = Path(tempfile.mkdtemp(prefix='bounce_'))
            from uacpy.io.file_manager import FileManager
            fm = FileManager(use_tmpfs=False, cleanup=False)
            fm.work_dir = work_dir
            fm.cleanup = False
        else:
            fm = self._setup_file_manager()

        try:
            base_name = 'bounce_run'
            input_file = fm.get_path(f'{base_name}.env')

            # Write ENV file (BOUNCE uses KRAKEN format)
            self._log(f"Writing BOUNCE input file: {input_file}", level='info')
            self._write_bounce_input(
                filepath=input_file,
                env=env,
                source=source,
                receiver=receiver,
                cmin=cmin,
                cmax=cmax,
                rmax_km=rmax_km,
                **kwargs
            )

            # Run BOUNCE executable
            self._log("Running BOUNCE...", level='info')
            self._execute(input_file, fm.work_dir)

            # Read output
            brc_file = fm.get_path(f'{base_name}.brc')
            irc_file = fm.get_path(f'{base_name}.irc')

            if not brc_file.exists():
                raise FileNotFoundError(
                    f"BOUNCE output file not found: {brc_file}\n"
                    "BOUNCE execution may have failed."
                )

            self._log(f"Reading BOUNCE output: {brc_file}", level='info')
            result = read_reflection_coefficient(str(brc_file), boundary='bottom')

            # Add file paths to result
            result['brc_file'] = str(brc_file)
            if irc_file.exists():
                result['irc_file'] = str(irc_file)

            # Convert to Field object
            frequency = source.frequency[0] if hasattr(source.frequency, '__len__') else source.frequency

            # For BOUNCE, the data is the full result dictionary
            # We use a custom field type for reflection coefficients
            field = Field(
                field_type='reflection_coefficients',
                data=result.get('R', np.array([])),  # Magnitude array
                ranges=result.get('theta', np.array([])),  # Using angles as "ranges"
                depths=np.array([0.0]),  # Not applicable for BOUNCE
                frequencies=np.array([frequency]),
                metadata={
                    'theta': result.get('theta', np.array([])),
                    'R': result.get('R', np.array([])),
                    'phi': result.get('phi', np.array([])),
                    'brc_file': result.get('brc_file'),
                    'irc_file': result.get('irc_file'),
                    'n_pts': result.get('n_pts', 0),
                    # Store parameters for later use in Bellhop/Kraken/Scooter
                    'cmin': self.cmin,
                    'cmax': self.cmax,
                    'rmax_km': self.rmax_km
                }
            )

            # Store the full result in metadata for easy access
            field.metadata['full_result'] = result

            self._log("BOUNCE simulation complete", level='info')
            return field

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _write_bounce_input(
        self,
        filepath: Path,
        env: Environment,
        source: Source,
        receiver: Receiver,
        cmin: float,
        cmax: float,
        rmax_km: float,
        **kwargs
    ):
        """
        Write BOUNCE input file using ATEnvWriter

        BOUNCE uses ENV format similar to KRAKEN with additional sections:
        - cmin, cmax (phase velocity bounds)
        - rmax_km (for angular sampling)
        """
        # Parse types
        ssp_type = parse_ssp_type(env.ssp_type)
        surface_type = parse_boundary_type(env.surface.acoustic_type)
        bottom_type = parse_boundary_type(env.bottom.acoustic_type)

        # Calculate dense mesh for BOUNCE (needs ~20 points per wavelength)
        frequency = source.frequency[0] if hasattr(source.frequency, '__len__') else source.frequency
        c_water = env.sound_speed if hasattr(env, 'sound_speed') else DEFAULT_SOUND_SPEED
        wavelength = c_water / frequency
        n_mesh = max(100, int(20 * env.depth / wavelength))

        with open(filepath, 'w') as f:
            # Write standard ENV sections using ATEnvWriter
            ATEnvWriter.write_header(
                f, env, source,
                ssp_type=ssp_type,
                surface_type=surface_type,
                attenuation_unit=AttenuationUnits.DB_PER_WAVELENGTH,
                volume_attenuation=None
            )

            ATEnvWriter.write_ssp_section(
                f, env, env.depth,
                n_mesh=n_mesh,
                roughness=0.0
            )

            ATEnvWriter.write_bottom_section(
                f, env,
                bottom_type=bottom_type,
                filepath=filepath,
                verbose=self.verbose
            )

            # BOUNCE-SPECIFIC SECTIONS

            # Phase velocity bounds (define angular coverage)
            f.write(f"{cmin:.2f} {cmax:.2f}\n")

            # Maximum range (for angular sampling resolution)
            f.write(f"{rmax_km:.2f}\n")

            # Source and receiver depths (placeholders - BOUNCE doesn't use them)
            ATEnvWriter.write_source_depths(f, source)
            ATEnvWriter.write_receiver_depths(f, receiver)

    def _execute(self, input_file: Path, work_dir: Path):
        """Execute BOUNCE binary"""
        try:
            base_name = input_file.stem
            result = subprocess.run(
                [str(self.executable), base_name],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=300
            )

            if result.returncode != 0:
                error_msg = f"BOUNCE failed with return code {result.returncode}\n"
                error_msg += f"stdout: {result.stdout}\n"
                error_msg += f"stderr: {result.stderr}"
                raise RuntimeError(error_msg)

            if self.verbose and result.stdout:
                self._log(f"Bounce output:\n{result.stdout}", level='debug')

        except subprocess.TimeoutExpired:
            raise RuntimeError("BOUNCE execution timed out after 5 minutes")
