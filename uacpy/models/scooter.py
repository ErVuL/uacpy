"""
Scooter finite-element FFP (Fast Field Program) model.

Computes the acoustic field in the frequency-wavenumber domain using a
finite-element discretization, then transforms to range via FFT. Supports
coherent TL and broadband time-series output.
"""

import subprocess
from pathlib import Path
from typing import Optional

from uacpy.models.base import PropagationModel, _UNSET, _resolve_overrides
from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.field import Field
from uacpy.core.constants import (
    AttenuationUnits, VolumeAttenuation,
    parse_ssp_type, parse_boundary_type,
    C_LOW_FACTOR, C_HIGH_FACTOR, DEFAULT_C_MIN, DEFAULT_C_MAX,
)
from uacpy.io.grn_reader import read_grn_file, grn_to_field, grn_to_transfer_function
from uacpy.io.at_env_writer import ATEnvWriter


class Scooter(PropagationModel):
    """
    Scooter finite element FFP (Fast Field Program) model

    Frequency-domain solver for underwater acoustics.
    Developed by Michael B. Porter.

    Parameters
    ----------
    executable : str or Path, optional
        Path to scooter executable. If None, searches automatically.
    use_tmpfs : bool, optional
        Use RAM filesystem. Default is False.
    verbose : bool, optional
        Verbose output. Default is False.

    Examples
    --------
    >>> scooter = Scooter()
    >>> result = scooter.run(env, source, receiver)
    """

    def __init__(
        self,
        executable: Optional[Path] = None,
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        n_mesh: int = 0,
        roughness: float = 0.0,
        rmax_multiplier: float = 2.0,
        volume_attenuation: Optional[str] = None,
        use_tmpfs: bool = False,
        verbose: bool = False,
        work_dir: Optional[Path] = None,
    ):
        """
        Parameters
        ----------
        executable : Path, optional
            Path to scooter executable. Auto-detected if None.
        c_low : float, optional
            Lower phase speed limit (m/s). None = auto (0.95 * min SSP speed).
        c_high : float, optional
            Upper phase speed limit (m/s). None = auto (1.05 * max of SSP and bottom speed).
        n_mesh : int, optional
            Mesh points per wavelength. 0 = auto. Default: 0.
        roughness : float, optional
            Bottom roughness (m). Default: 0.0.
        rmax_multiplier : float, optional
            Multiply max receiver range for wavenumber resolution. Default: 2.0.
        volume_attenuation : str, optional
            'T' (Thorp), 'F' (Francois-Garrison), 'B' (Biological). Default: None.
        """
        super().__init__(use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir)

        self.c_low = c_low
        self.c_high = c_high
        self.n_mesh = n_mesh
        self.roughness = roughness
        self.rmax_multiplier = rmax_multiplier
        self.volume_attenuation = volume_attenuation

        # Declare supported modes for Scooter
        from uacpy.models.base import RunMode
        self._supported_modes = [RunMode.COHERENT_TL, RunMode.TIME_SERIES]

        if executable is None:
            self.executable = self._find_executable('scooter.exe')
        else:
            self.executable = Path(executable)

        if not self.executable.exists():
            raise FileNotFoundError(
                f"Scooter executable not found: {self.executable}\n"
                "Please run install.sh to build Fortran models."
            )

    def _find_executable(self, name: str) -> Path:
        """Find executable (cross-platform)"""
        return self._find_executable_in_paths(name, 'oalib')

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode=None,
        frequencies: Optional['np.ndarray'] = None,
        c_low=_UNSET,
        c_high=_UNSET,
        n_mesh=_UNSET,
        roughness=_UNSET,
        rmax_multiplier=_UNSET,
        volume_attenuation=_UNSET,
        **kwargs
    ) -> Field:
        """
        Run Scooter simulation

        Parameters
        ----------
        env : Environment
            Ocean environment (range-dependent environments will be approximated)
        source : Source
            Acoustic source
        receiver : Receiver
            Receiver array
        run_mode : RunMode, optional
            COHERENT_TL (default): single-frequency TL.
            TIME_SERIES: broadband transfer function over frequency vector.
        frequencies : ndarray, optional
            Frequency vector for broadband (TIME_SERIES) mode. If not provided,
            a default vector spanning fc/2 to 2*fc is generated.
        c_low, c_high, n_mesh, roughness, rmax_multiplier, volume_attenuation : optional
            Per-call overrides for constructor defaults.
        **kwargs
            Additional Scooter parameters

        Returns
        -------
        field : Field
            TL field (COHERENT_TL) or transfer function field (TIME_SERIES)
        """
        import numpy as np
        from uacpy.models.base import RunMode

        if run_mode is None:
            run_mode = RunMode.COHERENT_TL

        with _resolve_overrides(self, c_low=c_low, c_high=c_high, n_mesh=n_mesh,
                                roughness=roughness, rmax_multiplier=rmax_multiplier,
                                volume_attenuation=volume_attenuation):
            # Handle range-dependent environments
            env = self._handle_range_dependent_environment(
                env, alternatives='Bellhop, RAM, or KrakenField'
            )

            # Clip receiver depths to environment depth (with safety margin)
            receiver = self._clip_receiver_depths(receiver, env.depth)

            self.validate_inputs(env, source, receiver)

            # For broadband mode, generate frequency vector if not provided
            broadband_freqs = None
            if run_mode == RunMode.TIME_SERIES:
                if frequencies is not None:
                    broadband_freqs = np.asarray(frequencies, dtype=float)
                else:
                    fc = float(source.frequency[0])
                    broadband_freqs = np.linspace(fc * 0.5, fc * 2.0, 64)
                self._log(f"Broadband mode: {len(broadband_freqs)} frequencies, "
                          f"{broadband_freqs[0]:.1f}-{broadband_freqs[-1]:.1f} Hz")

            fm = self._setup_file_manager()
            self.file_manager = fm

            try:
                base_name = 'model'

                # Write environment file
                env_file = fm.get_path(f'{base_name}.env')
                self._log(f"Writing environment file: {env_file}")

                self._write_scooter_env(
                    env_file, env, source, receiver,
                    frequencies=broadband_freqs,
                    **kwargs
                )

                # Run Scooter
                self._log("Running Scooter...")
                self._run_scooter(base_name, fm.work_dir)

                # Read Green's function output
                grn_file = fm.get_path(f'{base_name}.grn')
                if not grn_file.exists():
                    self._log(f"Green's function file not found: {grn_file}", level='error')
                    raise FileNotFoundError(f"Green's function file not found: {grn_file}")

                self._log("Reading Green's function...", level='info')
                grn_data = read_grn_file(grn_file)

                if grn_data['nk'] == 0:
                    self._log("Scooter produced empty Green's function (nk=0)", level='error')
                    raise RuntimeError("Scooter produced empty Green's function (nk=0)")

                if run_mode == RunMode.TIME_SERIES:
                    self._log(f"Transforming {grn_data['nfreq']} frequencies to range domain...")
                    result = grn_to_transfer_function(grn_data, receiver.ranges)
                else:
                    self._log("Transforming to range domain (FFT-based Hankel transform)...")
                    result = grn_to_field(grn_data, receiver.ranges, method='fft_hankel')

                self._log("Simulation complete")
                return result

            finally:
                if fm.cleanup:
                    fm.cleanup_work_dir()

    def _write_scooter_env(self, filepath, env, source, receiver, **kwargs):
        """
        Write Scooter environment file using shared ATEnvWriter

        Scooter uses ReadEnvironmentMod format (same as Kraken) with additional sections:
        - Phase speed limits (cLow, cHigh)
        - Maximum range with multiplier (RMax)
        - Receiver ranges (not in standard Kraken format)
        - Supports shear wave parameters in bottom halfspace
        """
        import numpy as np

        # Parse types with backward compatibility
        ssp_type = parse_ssp_type(env.ssp_type)
        surface_type = parse_boundary_type(env.surface.acoustic_type)
        bottom_type = parse_boundary_type(env.bottom.acoustic_type)

        # Parse volume attenuation from instance attribute
        vol_atten = None
        if self.volume_attenuation:
            vol_atten = VolumeAttenuation.from_string(self.volume_attenuation)

        # Get kwargs
        frequencies = kwargs.get('frequencies', None)

        with open(filepath, 'w') as f:
            # Write standard ENV sections using ATEnvWriter
            ATEnvWriter.write_header(
                f, env, source,
                ssp_type=ssp_type,
                surface_type=surface_type,
                attenuation_unit=AttenuationUnits.DB_PER_WAVELENGTH,
                volume_attenuation=vol_atten,
                frequencies=frequencies
            )

            ATEnvWriter.write_ssp_section(
                f, env, env.depth,
                n_mesh=self.n_mesh,
                roughness=self.roughness
            )

            # Write sediment layers if layered bottom
            ATEnvWriter.write_layer_sections(f, env, env.depth)

            # Bottom section with shear wave support for Scooter
            bottom_code = bottom_type.to_acoustics_toolbox_code()
            sigma = getattr(env.bottom, 'roughness', 0.0)

            # Check for range-dependent bathymetry
            if hasattr(env, 'bathymetry') and len(env.bathymetry) > 1:
                f.write(f"'{bottom_code}~' {sigma:.1f}\n")
            else:
                f.write(f"'{bottom_code}' {sigma:.1f}\n")

            # Handle reflection coefficient file (type 'F')
            if bottom_code == 'F':
                # Copy reflection file to working directory
                if env.bottom.reflection_file:
                    import shutil
                    from pathlib import Path
                    brc_source = Path(env.bottom.reflection_file)
                    if brc_source.exists():
                        # Copy to same directory as .env file with matching base name
                        brc_dest = Path(filepath).with_suffix('.brc')
                        shutil.copy(brc_source, brc_dest)
                        self._log(f"Copied reflection file: {brc_source} -> {brc_dest}", level='info')
                    else:
                        self._log(f"Reflection coefficient file not found: {env.bottom.reflection_file}", level='error')
                        raise FileNotFoundError(
                            f"Reflection coefficient file not found: {env.bottom.reflection_file}\n"
                            f"Generate this file using BOUNCE or OASR models."
                        )
                else:
                    self._log("acoustic_type='file' requires reflection_file parameter", level='error')
                    raise ValueError(
                        "acoustic_type='file' requires reflection_file parameter.\n"
                        "Example: BoundaryProperties(acoustic_type='file', reflection_file='path/to/file.brc')"
                    )

                # For 'F' type, write phase velocity bounds and rmax (not bottom properties)
                # These define the range of angles covered by the reflection coefficient table
                cmin = getattr(env.bottom, 'reflection_cmin', DEFAULT_C_MIN)
                cmax = getattr(env.bottom, 'reflection_cmax', DEFAULT_C_MAX)
                rmax_km = getattr(env.bottom, 'reflection_rmax_km', 10.0)
                f.write(f"{cmin:.2f}  {cmax:.2f}\n")
                f.write(f"{rmax_km:.2f}\n")

            # Write halfspace parameters with shear wave support (type 'A')
            elif bottom_code == 'A':  # Half-space
                z_bottom = env.bottom.depth if hasattr(env.bottom, 'depth') else env.depth
                cp = env.bottom.sound_speed
                cs = getattr(env.bottom, 'shear_speed', 0.0)
                rho = env.bottom.density
                alpha_p = env.bottom.attenuation
                alpha_s = getattr(env.bottom, 'shear_attenuation', 0.0)
                f.write(f"  {z_bottom:.2f}  {cp:.2f}  {cs:.1f}  "
                       f"{rho:.2f}  {alpha_p:.2f}  {alpha_s:.2f} /\n")

            # SCOOTER-SPECIFIC SECTIONS

            # Phase speed limits (cLow, cHigh) and RMax
            # NOTE: Skip these if using reflection coefficient file ('F' type)
            # because they were already written as part of the boundary specification
            if bottom_code != 'F':
                # Phase speed limits (cLow, cHigh)
                c_min = min([c for _, c in env.ssp_data])
                c_max = max([c for _, c in env.ssp_data] + [env.bottom.sound_speed])
                c_low = self.c_low if self.c_low is not None else c_min * C_LOW_FACTOR
                c_high = self.c_high if self.c_high is not None else c_max * C_HIGH_FACTOR
                f.write(f"{c_low:.1f} {c_high:.1f}\n")

                # RMax (km) with multiplier for wavenumber resolution
                # deltak = pi / RMax, so larger RMax gives finer k-spacing
                rmax = float(receiver.ranges.max() / 1000.0) * self.rmax_multiplier
                f.write(f"{rmax:.6f}\n")

            # Source depths
            source_depth = float(source.depth[0]) if hasattr(source.depth, '__len__') else float(source.depth)
            f.write(f"1\n")  # NSD
            f.write(f"{source_depth:.1f} /\n")

            # Receiver depths
            f.write(f"{len(receiver.depths)}\n")  # NRD
            if len(receiver.depths) == 1:
                f.write(f"{receiver.depths[0]:.1f} /\n")
            else:
                f.write(f"{receiver.depths.min():.1f} {receiver.depths.max():.1f} /\n")

            # Broadband frequency vector (ReadfreqVec reads after source/receiver depths)
            if frequencies is not None and len(frequencies) > 1:
                ATEnvWriter.write_broadband_freqs(f, frequencies)

    def _run_scooter(self, base_name: str, work_dir: Path):
        """Execute Scooter"""
        cmd = [str(self.executable), base_name]

        result = subprocess.run(
            cmd,
            cwd=str(work_dir),
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            error_msg = f"Scooter failed with return code {result.returncode}\n"
            error_msg += f"stdout: {result.stdout}\n"
            error_msg += f"stderr: {result.stderr}"
            raise RuntimeError(error_msg)

        if self.verbose and result.stdout:
            self._log(f"Scooter output:\n{result.stdout}", level='debug')
