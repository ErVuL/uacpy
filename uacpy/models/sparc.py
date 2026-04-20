"""
SPARC - Seismo-Acoustic Propagation in Realistic oCeans

SPARC is an FFP (Fast Field Program) model that includes elastic bottom effects
and seismo-acoustic coupling. It uses the same wavenumber integration approach
as Scooter but with support for elastic media.
"""

from pathlib import Path
import subprocess
from typing import Optional
import numpy as np

from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.field import Field
from uacpy.core.constants import (
    BoundaryType, AttenuationUnits, VolumeAttenuation,
    parse_ssp_type, parse_boundary_type,
    C_LOW_FACTOR, C_HIGH_FACTOR, DEFAULT_SOUND_SPEED,
)
from uacpy.models.base import PropagationModel, RunMode, _UNSET, _resolve_overrides
from uacpy.io.rts_reader import read_rts_file, rts_to_tl
from uacpy.io.at_env_writer import ATEnvWriter


class SPARC(PropagationModel):
    """
    SPARC - Seismo-Acoustic Propagation in Realistic oCeans

    Time-domain FFP model that computes transient pressure fields and converts
    to frequency-domain transmission loss via FFT.

    SPARC is fundamentally time-domain (unlike Scooter which is frequency-domain).
    It computes pressure time series at receiver locations, then uses FFT to
    extract amplitude at the target frequency for TL calculation.

    Note: For elastic bottom analysis in frequency domain, Scooter is recommended
    as it directly computes frequency-domain solutions.

    Limitations:
    - Only supports Vacuum or Rigid boundary conditions (no halfspace)
    - Single receiver depth per run (horizontal array mode)
    - Longer computation time due to time-domain integration

    Parameters
    ----------
    verbose : bool, optional
        Enable verbose output. Default is True.

    Examples
    --------
    >>> import uacpy
    >>> from uacpy.models.sparc import SPARC
    >>>
    >>> # Create simple environment
    >>> env = uacpy.Environment(depth=100.0, sound_speed=1500.0)
    >>>
    >>> source = uacpy.Source(depth=50.0, frequency=100.0)
    >>> receiver = uacpy.Receiver(
    ...     depths=[50.0],  # Single depth for SPARC
    ...     ranges=np.linspace(100, 5000, 50)
    ... )
    >>>
    >>> sparc = SPARC(verbose=False)
    >>> result = sparc.run(env, source, receiver)
    >>> print(f"TL range: {result.data.min():.1f} to {result.data.max():.1f} dB")
    """

    def __init__(
        self,
        executable: Optional[Path] = None,
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        n_mesh: int = 0,
        roughness: float = 0.0,
        output_mode: str = 'R',
        pulse_type: str = 'PN+B',
        n_t_out: int = 501,
        t_max: Optional[float] = None,
        t_start: float = -0.1,
        t_mult: float = 0.999,
        max_depths: int = 20,
        volume_attenuation: Optional[str] = None,
        use_tmpfs: bool = False,
        verbose: bool = False,
        work_dir: Optional[Path] = None,
    ):
        """
        Parameters
        ----------
        executable : Path, optional
            Path to sparc.exe. Auto-detected if None.
        c_low : float, optional
            Lower phase speed limit (m/s). None = auto. Default: None.
        c_high : float, optional
            Upper phase speed limit (m/s). None = auto. Default: None.
        n_mesh : int, optional
            Mesh points per wavelength. 0 = auto. Default: 0.
        roughness : float, optional
            Bottom roughness (m). Default: 0.0.
        output_mode : str, optional
            'R' (horizontal array), 'D' (vertical array), 'S' (snapshot). Default: 'R'.
        pulse_type : str, optional
            Pulse type string. Default: 'PN+B'.
        n_t_out : int, optional
            Number of time samples. Default: 501.
        t_max : float, optional
            Maximum time (s). None = auto (2.5x travel time). Default: None.
        t_start : float, optional
            Integration start time. Default: -0.1.
        t_mult : float, optional
            Integration time multiplier. Default: 0.999.
        max_depths : int, optional
            Maximum number of depths before warning. Default: 20.
        volume_attenuation : str, optional
            'T' (Thorp), 'F' (Francois-Garrison), 'B' (Biological). Default: None.
        """
        super().__init__(use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir)

        self.c_low = c_low
        self.c_high = c_high
        self.n_mesh = n_mesh
        self.roughness = roughness
        if output_mode not in ('R', 'D', 'S'):
            raise ValueError(
                f"Invalid output mode '{output_mode}'. "
                f"Valid modes: 'R' (horizontal array), 'D' (vertical array), 'S' (snapshot)"
            )
        self.output_mode = output_mode
        self.pulse_type = pulse_type
        self.n_t_out = n_t_out
        self.t_max = t_max
        self.t_start = t_start
        self.t_mult = t_mult
        self.max_depths = max_depths
        self.volume_attenuation = volume_attenuation

        # Declare supported modes for SPARC
        self._supported_modes = [
            RunMode.COHERENT_TL,
            RunMode.TIME_SERIES,
        ]

        if executable is None:
            self.executable = self._find_executable('sparc.exe')
        else:
            self.executable = Path(executable)

        if not self.executable.exists():
            from uacpy.core.exceptions import ExecutableNotFoundError
            raise ExecutableNotFoundError('SPARC', 'sparc.exe')

    def _find_executable(self, name: str) -> Path:
        """Find SPARC executable"""
        return self._find_executable_in_paths(name, 'oalib')

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[RunMode] = None,
        output_mode=_UNSET,
        pulse_type=_UNSET,
        n_t_out=_UNSET,
        t_max=_UNSET,
        t_start=_UNSET,
        t_mult=_UNSET,
        max_depths=_UNSET,
        c_low=_UNSET,
        c_high=_UNSET,
        n_mesh=_UNSET,
        roughness=_UNSET,
        volume_attenuation=_UNSET,
        **kwargs
    ) -> Field:
        """
        Run SPARC simulation (range-dependent environments will be approximated)

        Parameters
        ----------
        env : Environment
            Ocean environment (should include elastic bottom properties)
        source : Source
            Acoustic source
        receiver : Receiver
            Receiver array
        run_mode : RunMode, optional
            COHERENT_TL (default): compute transmission loss via FFT of time-series.
            TIME_SERIES: return raw pressure time-series.
        output_mode, pulse_type, n_t_out, t_max, t_start, t_mult, max_depths,
        c_low, c_high, n_mesh, roughness, volume_attenuation : optional
            Per-call overrides for constructor defaults.
        **kwargs
            Additional parameters for ENV file generation

        Returns
        -------
        field : Field
            TL field (COHERENT_TL) or time-series field (TIME_SERIES)

        Notes
        -----
        SPARC is range-independent. If a range-dependent environment is provided,
        it will automatically use a median-depth approximation with a warning.
        """
        # Validate output_mode override if provided
        if output_mode is not _UNSET and output_mode not in ('R', 'D', 'S'):
            raise ValueError(
                f"Invalid output mode '{output_mode}'. "
                f"Valid modes: 'R' (horizontal array), 'D' (vertical array), 'S' (snapshot)"
            )

        if run_mode is None:
            run_mode = RunMode.COHERENT_TL

        # Apply per-call overrides (temporarily sets self.* for internal methods)
        _overrides = _resolve_overrides(
            self, output_mode=output_mode, pulse_type=pulse_type,
            n_t_out=n_t_out, t_max=t_max, t_start=t_start, t_mult=t_mult,
            max_depths=max_depths, c_low=c_low, c_high=c_high, n_mesh=n_mesh,
            roughness=roughness, volume_attenuation=volume_attenuation
        )
        with _overrides:
            return self._run_impl(env, source, receiver, run_mode, **kwargs)

    def _run_impl(self, env, source, receiver, run_mode, **kwargs):
        """Internal run implementation (called within override context)"""
        # Handle range-dependent environments
        env = self._handle_range_dependent_environment(env, alternatives='Bellhop or RAM')

        # Clip receiver depths to environment depth (with safety margin)
        receiver = self._clip_receiver_depths(receiver, env.depth)

        # SPARC limitation: horizontal array mode requires one run per depth
        # For large depth grids, this becomes computationally expensive
        if len(receiver.depths) > self.max_depths:
            self._log(
                f"SPARC horizontal array mode requires one simulation per depth. "
                f"Requested {len(receiver.depths)} depths exceeds limit of {self.max_depths}. "
                f"To run SPARC with many depths:\n"
                f"  1. Use fewer receiver depths\n"
                f"  2. Increase max_depths: SPARC(max_depths={len(receiver.depths)})\n"
                f"  3. Use alternative models (Bellhop, RAM, Kraken, Scooter, OASN) for 2D fields\n"
                f"Skipping SPARC for this configuration.",
                level='warn'
            )
            # Return empty field to signal skip
            return None

        self.validate_inputs(env, source, receiver)

        fm = self._setup_file_manager()
        self.file_manager = fm

        try:
            base_name = 'model'
            output_mode = self.output_mode
            freq = source.frequency[0]

            if output_mode == 'R':
                # SPARC computes horizontal arrays (one depth at a time)
                # For 2D fields, we need to run SPARC for each receiver depth

                if len(receiver.depths) == 1:
                    # Single depth - run once
                    self._log(f"Computing at depth {receiver.depths[0]:.1f}m...")

                    # Write environment file
                    env_file = fm.get_path(f'{base_name}.env')
                    self._write_sparc_env(env_file, env, source, receiver, **kwargs)

                    # Run SPARC
                    self._run_sparc(base_name, fm.work_dir)

                    # Read output
                    rts_file = fm.get_path(f'{base_name}.rts')
                    if not rts_file.exists():
                        raise FileNotFoundError(f"RTS file not found: {rts_file}")

                    rts_data = read_rts_file(rts_file)

                    if run_mode == RunMode.TIME_SERIES:
                        # Return raw time-series pressure data
                        result = Field(
                            field_type='time_series',
                            data=rts_data['p'],  # shape: (nt, nr)
                            ranges=rts_data['ranges'],
                            depths=receiver.depths,
                            metadata={
                                'model': 'SPARC',
                                'frequency': freq,
                                'time': rts_data['time'],
                                'dt': rts_data['dt'],
                                'nt': rts_data['nt'],
                                'receiver_depth': receiver.depths[0],
                            }
                        )
                        self._log("SPARC simulation complete (time-series mode)")
                        return result

                    tl, ranges_out = rts_to_tl(rts_data, freq, method='fft')
                    tl_field = tl.reshape(1, -1)

                    time_series_data = {
                        'time': rts_data['time'],
                        'pressure': rts_data['p'],
                        'dt': rts_data['dt'],
                        'nt': rts_data['nt'],
                        'receiver_depth': receiver.depths[0],
                    }

                else:
                    # Multiple depths - run SPARC for each depth
                    self._log(f"Computing for {len(receiver.depths)} depths (SPARC horizontal array mode)...")

                    tl_field = []
                    ranges_out = receiver.ranges
                    time_series_data = None  # Only store for single depth to save memory

                    pressure_all = [] if run_mode == RunMode.TIME_SERIES else None

                    for idx, depth in enumerate(receiver.depths):
                        # Create single-depth receiver
                        single_receiver = Receiver(depths=np.array([depth]), ranges=receiver.ranges)

                        # Write environment file for this depth
                        depth_base = f'{base_name}_d{idx}'
                        env_file = fm.get_path(f'{depth_base}.env')
                        self._write_sparc_env(env_file, env, source, single_receiver, **kwargs)

                        # Run SPARC for this depth
                        if self.verbose:
                            self._log(f"  Depth {idx+1}/{len(receiver.depths)}: {depth:.1f}m")
                        self._run_sparc(depth_base, fm.work_dir)

                        # Read output
                        rts_file = fm.get_path(f'{depth_base}.rts')
                        if not rts_file.exists():
                            self._log(f"RTS file not found: {rts_file}", level='error')
                            raise FileNotFoundError(f"RTS file not found: {rts_file}")

                        rts_data = read_rts_file(rts_file)

                        if run_mode == RunMode.TIME_SERIES:
                            pressure_all.append(rts_data['p'])  # (nt, nr)
                        else:
                            tl_single, ranges_out = rts_to_tl(rts_data, freq, method='fft')
                            tl_field.append(tl_single)

                    if run_mode == RunMode.TIME_SERIES:
                        # Stack: (n_depths, nt, nr)
                        pressure_stack = np.stack(pressure_all, axis=0)
                        result = Field(
                            field_type='time_series',
                            data=pressure_stack,
                            ranges=receiver.ranges,
                            depths=receiver.depths,
                            metadata={
                                'model': 'SPARC',
                                'frequency': freq,
                                'time': rts_data['time'],
                                'dt': rts_data['dt'],
                                'nt': rts_data['nt'],
                            }
                        )
                        self._log("SPARC simulation complete (time-series mode)")
                        return result

                    # Stack all depths into 2D array
                    tl_field = np.vstack(tl_field)  # shape: (n_depths, n_ranges)

                result = Field(
                    field_type='tl',
                    data=tl_field,
                    ranges=ranges_out,
                    depths=receiver.depths,
                    metadata={
                        'model': 'SPARC',
                        'frequency': freq,
                        'conversion_method': 'fft',
                        'output_mode': 'R',
                        'n_depth_runs': len(receiver.depths),
                        'time_series': time_series_data,  # Only available for single-depth runs
                        'note': 'Time-series data only preserved for single-depth runs due to memory constraints'
                    }
                )

            elif output_mode == 'D':
                # Vertical array mode: pressure vs depth at fixed ranges
                # SPARC outputs .rts file with depth values
                self._log(f"Computing vertical array at {len(receiver.ranges)} ranges...")

                if len(receiver.ranges) == 1:
                    # Single range - run once
                    env_file = fm.get_path(f'{base_name}.env')
                    self._write_sparc_env(env_file, env, source, receiver, **kwargs)

                    self._run_sparc(base_name, fm.work_dir)

                    # Read vertical array output
                    rts_file = fm.get_path(f'{base_name}.rts')
                    if not rts_file.exists():
                        raise FileNotFoundError(f"RTS file not found: {rts_file}")

                    rts_data = read_rts_file(rts_file)
                    # In vertical mode, 'ranges' in RTS file actually contains depths
                    depths_out = rts_data['ranges']  # These are actually depths
                    tl, _ = rts_to_tl(rts_data, freq, method='fft')
                    tl_field = tl.reshape(-1, 1)  # shape: (n_depths, 1)

                else:
                    # Multiple ranges - run SPARC for each range
                    tl_field = []
                    depths_out = receiver.depths

                    for idx, range_m in enumerate(receiver.ranges):
                        # Create single-range receiver
                        single_receiver = Receiver(depths=receiver.depths, ranges=np.array([range_m]))

                        range_base = f'{base_name}_r{idx}'
                        env_file = fm.get_path(f'{range_base}.env')
                        self._write_sparc_env(env_file, env, source, single_receiver, **kwargs)

                        if self.verbose:
                            self._log(f"  Range {idx+1}/{len(receiver.ranges)}: {range_m:.1f}m")
                        self._run_sparc(range_base, fm.work_dir)

                        rts_file = fm.get_path(f'{range_base}.rts')
                        if not rts_file.exists():
                            self._log(f"RTS file not found: {rts_file}", level='error')
                            raise FileNotFoundError(f"RTS file not found: {rts_file}")

                        rts_data = read_rts_file(rts_file)
                        tl_single, _ = rts_to_tl(rts_data, freq, method='fft')
                        tl_field.append(tl_single)

                    # Stack all ranges into 2D array
                    tl_field = np.column_stack(tl_field)  # shape: (n_depths, n_ranges)

                result = Field(
                    field_type='tl',
                    data=tl_field,
                    ranges=receiver.ranges,
                    depths=depths_out,
                    metadata={
                        'model': 'SPARC',
                        'frequency': freq,
                        'conversion_method': 'fft',
                        'output_mode': 'D',
                        'n_range_runs': len(receiver.ranges),
                    }
                )

            elif output_mode == 'S':
                # Snapshot mode: wavenumber-domain Green's function
                # SPARC outputs .grn file which needs Hankel transform
                from uacpy.io.grn_reader import read_grn_file, grn_to_field

                self._log("Computing snapshot (wavenumber domain)...")

                # Write environment file
                env_file = fm.get_path(f'{base_name}.env')
                self._write_sparc_env(env_file, env, source, receiver, **kwargs)

                # Run SPARC
                self._run_sparc(base_name, fm.work_dir)

                # Read Green's function file
                grn_file = fm.get_path(f'{base_name}.grn')
                if not grn_file.exists():
                    raise FileNotFoundError(
                        f"GRN file not found: {grn_file}\n"
                        "SPARC snapshot mode should produce a .grn file.\n"
                        "Check SPARC output for errors."
                    )

                self._log("Reading Green's function and transforming to range domain...")
                grn_data = read_grn_file(grn_file)

                # Transform to range domain using Hankel transform
                result = grn_to_field(grn_data, receiver.ranges, method='fft_hankel')

                # Update metadata
                result.metadata.update({
                    'model': 'SPARC',
                    'frequency': freq,
                    'output_mode': 'S',
                    'note': 'Snapshot mode: Green function transformed via Hankel transform'
                })

            else:
                raise ValueError(
                    f"Invalid output mode '{output_mode}'. "
                    f"Valid modes: 'R' (horizontal array), 'D' (vertical array), 'S' (snapshot)"
                )

            self._log("Simulation complete")
            return result

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _write_sparc_env(self, filepath, env, source, receiver, **kwargs):
        """
        Write SPARC environment file using shared ATEnvWriter

        SPARC extends ReadEnvironmentMod format with:
        - Output mode in TopOpt (5th character: R=horizontal, D=vertical, S=snapshot)
        - Limited bottom types (only vacuum and rigid, no halfspace)
        - Time-domain pulse parameters
        - Time output parameters
        - Integration parameters
        """
        import numpy as np

        # Parse types with backward compatibility
        ssp_type = parse_ssp_type(env.ssp_type)
        surface_type = parse_boundary_type(env.surface.acoustic_type)

        # SPARC limitation: only supports Vacuum and Rigid boundaries
        bottom_acoustic_type = env.bottom.acoustic_type.lower()
        if bottom_acoustic_type in ['half-space', 'halfspace', 'a']:
            self._log(
                "SPARC does not support elastic halfspace bottom boundaries. "
                "Automatically converting to 'rigid' boundary. "
                "For full elastic bottom support, use Bellhop, Kraken, OASES, or Scooter. "
                "To suppress this warning, explicitly set env.bottom.acoustic_type='rigid' before running SPARC.",
                level='warn'
            )
            bottom_acoustic_type = 'rigid'

        # Map to BoundaryType
        if bottom_acoustic_type == 'vacuum':
            bottom_type = BoundaryType.VACUUM
        elif bottom_acoustic_type == 'rigid':
            bottom_type = BoundaryType.RIGID
        else:
            raise ValueError(
                f"Invalid bottom boundary type '{env.bottom.acoustic_type}' for SPARC. "
                f"Only 'vacuum' and 'rigid' are supported."
            )

        # Parse volume attenuation from instance attribute
        vol_atten = None
        if self.volume_attenuation:
            vol_atten = VolumeAttenuation.from_string(self.volume_attenuation)

        with open(filepath, 'w') as f:
            # SPARC-specific header with output mode in TopOpt
            # Write title, frequency, media count
            f.write(f"'{env.name}'\n")
            f.write(f"{source.frequency[0]:.6f}\n")
            f.write("1\n")

            # SPARC TopOpt: [SSP][BC][AttenUnit(2 chars)][OutputMode]
            ssp_code = ssp_type.to_acoustics_toolbox_code()
            surface_code = surface_type.to_acoustics_toolbox_code()
            atten_code = AttenuationUnits.DB_PER_WAVELENGTH.to_char()
            vol_atten_code = vol_atten.to_char() if vol_atten else ' '
            topopt = f"{ssp_code}{surface_code}{atten_code}{vol_atten_code}{self.output_mode}".ljust(6)
            f.write(f"'{topopt}'\n")

            # Write SSP section
            ATEnvWriter.write_ssp_section(
                f, env, env.depth,
                n_mesh=self.n_mesh,
                roughness=self.roughness
            )

            # Write sediment layers if layered bottom
            ATEnvWriter.write_layer_sections(f, env, env.depth)

            # Write bottom section (SPARC only supports V and R)
            bottom_code = bottom_type.to_acoustics_toolbox_code()
            sigma = getattr(env.bottom, 'roughness', 0.0)
            f.write(f"'{bottom_code}' {sigma:.1f}\n")
            # Note: No halfspace parameters since SPARC doesn't support them

            # SPARC-SPECIFIC SECTIONS

            # Phase speed limits (cLow, cHigh)
            c_min = min([c for _, c in env.ssp_data])
            c_max = max([c for _, c in env.ssp_data] + [env.bottom.sound_speed])
            c_low = self.c_low if self.c_low is not None else c_min * C_LOW_FACTOR
            c_high = self.c_high if self.c_high is not None else c_max * C_HIGH_FACTOR
            f.write(f"{c_low:.1f} {c_high:.1f}\n")

            # RMax (maximum range in km)
            rmax = float(receiver.ranges.max() / 1000.0)
            f.write(f"{rmax:.6f}\n")

            # Source depth
            source_depth = float(source.depth[0]) if hasattr(source.depth, '__len__') else float(source.depth)
            f.write(f"1\n")
            f.write(f"{source_depth:.1f} /\n")

            # Receiver depths (SPARC format: count then start/end)
            f.write(f"{len(receiver.depths)}\n")
            if len(receiver.depths) == 1:
                # Single depth - write it twice for SPARC interpolation
                f.write(f"{receiver.depths[0]:.1f} {receiver.depths[0]:.1f} /\n")
            else:
                f.write(f"{receiver.depths.min():.1f} {receiver.depths.max():.1f} /\n")

            # Time-domain pulse parameters (SPARC-specific, come BEFORE ranges!)
            f.write(f"'{self.pulse_type}'\n")

            # Frequency range for pulse (Hz)
            # IMPORTANT: Narrow frequency band to prevent excessive wavenumber computation
            # SPARC computes Nk wavenumbers spanning [f_min, f_max]
            # Wide bands (e.g., 500-1500 Hz) cause 3000+ wavenumbers and hang
            # Use ±2% band for fast computation while maintaining accuracy
            # Example: For 1000 Hz, use 980-1020 Hz (40 Hz bandwidth)
            freq = source.frequency[0]
            f_min = kwargs.get('f_min', freq * 0.98)  # ±2% band for speed
            f_max = kwargs.get('f_max', freq * 1.02)
            f.write(f"{f_min:.1f} {f_max:.1f}\n")

            # Receiver ranges (come AFTER pulse info in SPARC)
            ranges_km = receiver.ranges / 1000.0
            f.write(f"{len(ranges_km)}\n")

            # Write ranges
            if len(ranges_km) <= 10:
                # If few ranges, write all on one line
                ranges_str = ' '.join([f"{r:.6f}" for r in ranges_km])
                f.write(f"{ranges_str} /\n")
            else:
                # If many ranges, use start/end (SPARC will interpolate)
                f.write(f"{ranges_km.min():.6f} {ranges_km.max():.6f} /\n")

            # Time output parameters
            f.write(f"{self.n_t_out}\n")

            # Time output range (s)
            max_range_m = rmax * 1000.0  # Convert km to m
            c_water = kwargs.get('sound_speed', DEFAULT_SOUND_SPEED)
            travel_time = max_range_m / c_water
            t_max = self.t_max if self.t_max is not None else travel_time * 2.5
            f.write(f"0.0 {t_max:.6f} /\n")

            # Integration parameters: TSTART, TMULT, ALPHA, BETA, V
            f.write(f"{self.t_start:.6f} {self.t_mult:.6f} 0.0 0.0 0.0\n")

    def _run_sparc(self, base_name: str, work_dir: Path):
        """
        Run SPARC executable with timeout to prevent hangs

        SPARC can hang indefinitely if the ENV file format is incorrect
        or if boundary conditions are incompatible. A timeout ensures
        the test suite doesn't hang.
        """
        try:
            result = subprocess.run(
                [str(self.executable), base_name],
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=30.0  # 30 second timeout to prevent indefinite hangs
            )
        except subprocess.TimeoutExpired:
            error_msg = (
                f"SPARC executable timed out after 30 seconds.\n"
                f"This usually indicates:\n"
                f"  1. Incorrect ENV file format (check {work_dir}/{base_name}.env)\n"
                f"  2. Incompatible boundary conditions (SPARC only supports vacuum/rigid)\n"
                f"  3. Malformed input parameters\n"
                f"Check the .env file and ensure bottom boundary is 'rigid' or 'vacuum'."
            )
            raise RuntimeError(error_msg)

        if result.returncode != 0:
            error_msg = f"SPARC failed with return code {result.returncode}\n"
            error_msg += f"stdout: {result.stdout}\n"
            error_msg += f"stderr: {result.stderr}\n"
            # Try to read .prt file for more details
            prt_file = work_dir / f"{base_name}.prt"
            if prt_file.exists():
                try:
                    with open(prt_file) as f:
                        prt_contents = f.read()
                        error_msg += f"\n.prt file contents:\n{prt_contents[-2000:]}"  # Last 2000 chars
                except Exception:
                    pass
            raise RuntimeError(error_msg)

        if self.verbose and result.stdout:
            self._log(f"SPARC output:\n{result.stdout}", level='debug')
