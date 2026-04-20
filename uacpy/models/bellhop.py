"""
Bellhop and BellhopCUDA ray tracing models

Supports broadband time-series generation via the arrivals-based approach
described in the Bellhop User Guide (Section 9). The workflow:
1. Run Bellhop in arrivals mode ('A') at the center frequency
2. Build frequency-domain transfer function H(f) from arrival
   amplitudes, phases, and delays
3. IFFT to time domain, optionally convolved with a source waveform

This is a key advantage of ray/beam models: broadband results from a
single run, since ray travel times are frequency-independent (geometric).
"""

import os
import subprocess
import warnings
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from scipy.signal import hilbert

from uacpy.models.base import PropagationModel, RunMode, _UNSET
from uacpy.core.environment import Environment, BoundaryProperties
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.field import Field
from uacpy.core.constants import DEFAULT_SOUND_SPEED, DEFAULT_C_MIN, DEFAULT_C_MAX
from uacpy.io.bellhop_writer import BellhopEnvWriter
from uacpy.io.output_reader import read_shd_file, read_arr_file, read_ray_file


# ---------------------------------------------------------------------------
# Bellhop-specific signal processing helpers
# ---------------------------------------------------------------------------

def delayandsum(
    rcv_arrivals: dict,
    source_timeseries: np.ndarray,
    sample_rate: float,
    fc: float,
    time_window: Optional[float] = None,
    t_start: Optional[float] = None,
    c0: float = DEFAULT_SOUND_SPEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convolve source waveform with channel impulse response from Bellhop arrivals.

    Places a phase-shifted, amplitude-scaled copy of the source waveform at
    each arrival time.  Uses the Hilbert transform (analytic signal) to apply
    arbitrary phase rotations from caustics and boundary reflections, as
    described in the Bellhop User Guide (Sec. 9.3).

    Parameters
    ----------
    rcv_arrivals : dict
        Per-receiver arrival data (from ``read_arr_file``'s
        ``metadata['arrivals_by_receiver'][isd][ird][irr]``) with keys:
        amplitudes, phases, delays, delay_imag, n_arrivals.
    source_timeseries : ndarray
        Source waveform (1-D).  Normalized internally to 0 dB.
    sample_rate : float
        Sample rate in Hz.
    fc : float
        Center frequency in Hz (for volume-attenuation scaling).
    time_window : float, optional
        Output time window in seconds.  If *None*, estimated from the
        latest arrival plus the source waveform duration plus margin.
    t_start : float, optional
        Start time for the output.  If *None*, set to just before the
        earliest arrival.
    c0 : float, optional
        Reference sound speed in m/s.

    Returns
    -------
    rts : ndarray
        Received time series, shape ``(n_samples,)``.
    time_vector : ndarray
        Time vector in seconds.

    References
    ----------
    Bellhop User Guide, Section 9.3
    Original MATLAB code: delayandsum.m by M. B. Porter, 8/96
    """
    n_arr = rcv_arrivals['n_arrivals']
    if n_arr == 0:
        nrts = int(0.1 * sample_rate)
        return np.zeros(nrts), np.arange(nrts) / sample_rate

    amps = rcv_arrivals['amplitudes']
    phases_deg = rcv_arrivals['phases']
    delays = rcv_arrivals['delays']
    delay_imag = rcv_arrivals['delay_imag']

    # Normalize source waveform
    sts = np.asarray(source_timeseries, dtype=float)
    sts_max = np.max(np.abs(sts))
    if sts_max > 0:
        sts = sts / sts_max
    nsts = len(sts)

    # Compute analytic signal via Hilbert transform
    sts_analytic = hilbert(sts)

    deltat = 1.0 / sample_rate
    src_duration = nsts * deltat

    # Determine time window
    min_delay = float(np.min(delays))
    max_delay = float(np.max(delays))

    if t_start is None:
        t_start = max(0.0, min_delay - 0.1 * src_duration)

    if time_window is None:
        time_window = (max_delay - t_start) + 2.0 * src_duration

    nrts = int(np.ceil(time_window * sample_rate))
    rts = np.zeros(nrts)

    omega_c = 2.0 * np.pi * fc

    for ia in range(n_arr):
        phase_rad = np.deg2rad(phases_deg[ia])
        phase_factor = np.exp(1j * phase_rad)

        if delay_imag[ia] != 0.0:
            atten = np.exp(-delay_imag[ia] * omega_c / (2.0 * np.pi * fc))
        else:
            atten = 1.0

        scaled_amp = amps[ia] * atten

        delay_samples = (delays[ia] - t_start) / deltat
        i_start = int(np.round(delay_samples))

        for k in range(nsts):
            idx = i_start + k
            if 0 <= idx < nrts:
                rts[idx] += scaled_amp * np.real(
                    sts_analytic[k] * phase_factor
                )

    time_vector = t_start + np.arange(nrts) * deltat
    return rts, time_vector


def compute_channel_tf(
    arrivals: dict,
    freq_vec: np.ndarray,
    fc: float,
) -> np.ndarray:
    """
    Compute channel transfer function from Bellhop arrival data.

    Builds frequency-domain transfer function H(f) from ray arrivals.
    Each arrival contributes a complex exponential weighted by its
    amplitude and geometric phase.

    Parameters
    ----------
    arrivals : dict
        Per-receiver arrival data (from ``read_arr_file``'s
        ``metadata['arrivals_by_receiver'][isd][ird][irr]``) with keys:
        amplitudes, phases, delays, delay_imag, n_arrivals.
    freq_vec : ndarray
        Frequency vector in Hz.
    fc : float
        Center frequency in Hz (for attenuation scaling).

    Returns
    -------
    H : ndarray
        Complex channel transfer function, shape ``(n_freq,)``.

    References
    ----------
    Bellhop User Guide, Section 9
    Original MATLAB code: stackarr.m by mbp, 8/96
    """
    return Bellhop._arrivals_to_tf(arrivals, freq_vec, fc)


class Bellhop(PropagationModel):
    """
    Bellhop Gaussian beam/ray tracing model

    High-fidelity underwater acoustic ray tracing model developed by
    Michael B. Porter. Automatically detects and uses the fastest available
    version (bellhopcuda > bellhopcxx > Fortran).

    Performance comparison:
    - Fortran: Baseline single-threaded
    - bellhopcxx (C++): 10-30x faster (CPU multithreaded)
    - bellhopcuda (CUDA): 20-100x+ faster (GPU accelerated)

    Parameters
    ----------
    executable : str or Path, optional
        Path to bellhop executable. If None, auto-detects best version.
    prefer_cuda : bool, optional
        Prefer CUDA version if available. Default is True.
    use_tmpfs : bool, optional
        Use RAM filesystem for I/O. Default is False.
    verbose : bool, optional
        Print verbose output. Default is False.

    Examples
    --------
    Run transmission loss calculation:

    >>> bellhop = Bellhop()
    >>> result = bellhop.run(env, source, receiver, run_type='C')

    Force Fortran version:

    >>> bellhop = Bellhop(prefer_cuda=False)
    """

    def __init__(
        self,
        executable: Optional[Path] = None,
        prefer_cuda: bool = True,
        beam_type: str = 'B',
        n_beams: int = 0,
        alpha: tuple = (-80, 80),
        step: float = 0.0,
        z_box: Optional[float] = None,
        r_box: Optional[float] = None,
        source_type: str = 'R',
        grid_type: str = 'R',
        beam_shift: bool = False,
        volume_attenuation: Optional[str] = None,
        use_tmpfs: bool = False,
        verbose: bool = False,
        work_dir: Optional[Path] = None,
    ):
        """
        Parameters
        ----------
        executable : Path, optional
            Path to bellhop executable. Auto-detected if None.
        prefer_cuda : bool
            Prefer CUDA version if available. Default: True.
        beam_type : str
            Beam type: 'B' (Gaussian), 'R' (ray-centered), 'C' (Cartesian),
            'b' (geometric Gaussian), 'g' (geometric hat), 'G' (geometric hat Cartesian),
            'S' (simple Gaussian). Default: 'B'.
        n_beams : int
            Number of beams. 0 = use source.n_angles. Default: 0.
        alpha : tuple
            Launch angle limits (min, max) in degrees. Default: (-80, 80).
        step : float
            Ray step size in meters. 0 = automatic. Default: 0.0.
        z_box : float, optional
            Maximum depth for ray box. None = 1.2 * max depth. Default: None.
        r_box : float, optional
            Maximum range for ray box. None = 1.2 * max range. Default: None.
        source_type : str
            Source type: 'R' (point, cylindrical), 'X' (line, Cartesian). Default: 'R'.
        grid_type : str
            Receiver grid: 'R' (rectilinear), 'I' (irregular). Default: 'R'.
        beam_shift : bool
            Enable beam shift on boundary reflection. Default: False.
        volume_attenuation : str, optional
            'T' (Thorp), 'F' (Francois-Garrison), 'B' (Biological). Default: None.
        """
        super().__init__(use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir)

        # Declare supported modes for Bellhop
        self._supported_modes = [
            RunMode.COHERENT_TL,
            RunMode.INCOHERENT_TL,
            RunMode.SEMICOHERENT_TL,
            RunMode.RAYS,
            RunMode.EIGENRAYS,
            RunMode.ARRIVALS,
            RunMode.TIME_SERIES,
        ]
        self._supports_altimetry = True

        self.prefer_cuda = prefer_cuda
        self.beam_type = beam_type
        self.n_beams = n_beams
        self.alpha = alpha
        self.step = step
        self.z_box = z_box
        self.r_box = r_box
        self.source_type = source_type
        self.grid_type = grid_type
        self.beam_shift = beam_shift
        self.volume_attenuation = volume_attenuation
        self.version = "unknown"

        if executable is None:
            self.executable = self._find_executable()
        else:
            self.executable = Path(executable)
            self.version = "custom"

        if not self.executable.exists():
            from uacpy.core.exceptions import ExecutableNotFoundError
            raise ExecutableNotFoundError("Bellhop", str(self.executable))

        if verbose and self.version != "custom":
            self._log(f"Using Bellhop {self.version}: {self.executable}", level='info')

    def _find_executable(self) -> Path:
        """
        Find executable in PATH or uacpy/bin

        Search order (if prefer_cuda=True):
        1. bellhopcuda in bin/bellhopcuda/
        2. bellhopcxx in bin/bellhopcuda/
        3. bellhop.exe in bin/oalib/
        4. PATH
        """
        base_dir = Path(__file__).parent.parent

        # Search order based on preference
        if self.prefer_cuda:
            # Try CUDA/C++ versions first
            cuda_paths = [
                (base_dir / 'bin' / 'bellhopcuda' / 'bellhopcuda.exe', 'cuda'),
                (base_dir / 'bin' / 'bellhopcuda' / 'bellhopcuda', 'cuda'),
                (base_dir / 'bin' / 'bellhopcuda' / 'bellhopcxx.exe', 'cxx'),
                (base_dir / 'bin' / 'bellhopcuda' / 'bellhopcxx', 'cxx'),
            ]

            for path, version in cuda_paths:
                if path.exists():
                    self.version = version
                    return path

        # Try Fortran version
        fortran_paths = [
            (base_dir / 'bin' / 'oalib' / 'bellhop.exe', 'fortran'),
            (base_dir / 'bin' / 'bellhop' / 'bellhop.exe', 'fortran'),
        ]

        for path, version in fortran_paths:
            if path.exists():
                self.version = version
                return path

        # Check PATH for any bellhop variant (cross-platform)
        import shutil
        for name in ['bellhopcuda', 'bellhopcxx', 'bellhop', 'bellhop.exe']:
            result = shutil.which(name)
            if result:
                if 'cuda' in name:
                    self.version = 'cuda'
                elif 'cxx' in name:
                    self.version = 'cxx'
                else:
                    self.version = 'fortran'
                return Path(result)

        raise FileNotFoundError(
            "Could not find bellhop executable.\n"
            "Please run install_oalib.sh (Linux/Mac) or install_oalib.bat (Windows) to install."
        )

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_type: str = 'C',
        beam_type=_UNSET,
        n_beams=_UNSET,
        alpha=_UNSET,
        step=_UNSET,
        z_box=_UNSET,
        r_box=_UNSET,
        source_type=_UNSET,
        grid_type=_UNSET,
        beam_shift=_UNSET,
        volume_attenuation=_UNSET,
        **kwargs
    ) -> Field:
        """
        Run Bellhop simulation

        Parameters
        ----------
        env : Environment
            Ocean environment
        source : Source
            Acoustic source
        receiver : Receiver
            Receiver array
        run_type : str, optional
            Run type: 'C' (coherent TL), 'I' (incoherent TL), 'S' (semi-coherent),
            'A' (arrivals), 'E' (eigenrays), 'R' (ray trace).
            Default is 'C'.
        beam_type : str, optional
            Override constructor beam_type for this run.
        n_beams : int, optional
            Override constructor n_beams for this run.
        alpha : tuple, optional
            Override constructor alpha for this run.
        step : float, optional
            Override constructor step for this run.
        z_box : float, optional
            Override constructor z_box for this run.
        r_box : float, optional
            Override constructor r_box for this run.
        source_type : str, optional
            Override constructor source_type for this run.
        grid_type : str, optional
            Override constructor grid_type for this run.
        beam_shift : bool, optional
            Override constructor beam_shift for this run.
        volume_attenuation : str, optional
            Override constructor volume_attenuation for this run.
        **kwargs
            Advanced Cerveny beam parameters (for beam_type 'C' or 'R'):
            - beam_width_type (str): 'F' (filling), 'M' (minimum), 'W' (WKB)
            - beam_curvature (str): 'D' (double), 'S' (single), 'Z' (zero)
            - eps_multiplier (float): Epsilon multiplier for beam width
            - r_loop (float): Range for beam width selection (km)
            - n_image (int): Number of surface/bottom images
            - ib_win (int): Beam windowing parameter
            - component (str): 'P' for pressure (default), 'D' for displacement

        Returns
        -------
        field : Field
            Simulation results
        """
        # Resolve per-call overrides
        beam_type = beam_type if beam_type is not _UNSET else self.beam_type
        n_beams = n_beams if n_beams is not _UNSET else self.n_beams
        alpha = alpha if alpha is not _UNSET else self.alpha
        step = step if step is not _UNSET else self.step
        z_box = z_box if z_box is not _UNSET else self.z_box
        r_box = r_box if r_box is not _UNSET else self.r_box
        source_type = source_type if source_type is not _UNSET else self.source_type
        grid_type = grid_type if grid_type is not _UNSET else self.grid_type
        beam_shift = beam_shift if beam_shift is not _UNSET else self.beam_shift
        volume_attenuation = volume_attenuation if volume_attenuation is not _UNSET else self.volume_attenuation

        # Handle run_mode if provided (for compute_*() methods)
        run_mode_enum = None
        if 'run_mode' in kwargs:
            from uacpy.core.model_utils import ParameterMapper
            run_mode_enum = kwargs.pop('run_mode')
            if run_mode_enum == RunMode.TIME_SERIES:
                # Delegate to broadband pipeline
                return self._run_broadband(env, source, receiver, **kwargs)
            run_type = ParameterMapper.map_run_mode_to_bellhop(run_mode_enum)

        # Warn about unsupported range/depth-dependent bottom features
        if env.has_range_dependent_bottom():
            warnings.warn(
                "Bellhop does not support range-dependent bottom properties. "
                "Using median-range approximation via env.bottom. "
                "For range-dependent bottoms, consider RAM or KrakenField. "
                "For elastic bottoms, use run_with_bounce().",
                UserWarning, stacklevel=2
            )
            self._log(
                "Bellhop: range-dependent bottom properties ignored, "
                "using median-range approximation.", level='warn'
            )
        if env.has_layered_bottom():
            warnings.warn(
                "Bellhop does not support layered (depth-dependent) bottoms. "
                "Using halfspace properties only. "
                "For layered bottoms, use Kraken, Scooter, or OASES. "
                "For elastic bottoms, use run_with_bounce().",
                UserWarning, stacklevel=2
            )
            self._log(
                "Bellhop: layered bottom ignored, using halfspace only.",
                level='warn'
            )
        if env.has_range_dependent_layered_bottom():
            warnings.warn(
                "Bellhop does not support range-dependent layered bottoms. "
                "Using halfspace properties only. "
                "For range+depth-dependent bottoms, use RAM.",
                UserWarning, stacklevel=2
            )
            self._log(
                "Bellhop: range-dependent layered bottom ignored, using halfspace only.",
                level='warn'
            )

        self.validate_inputs(env, source, receiver)

        # Setup file manager
        fm = self._setup_file_manager()
        self.file_manager = fm

        try:
            # Generate base filename
            base_name = 'model'

            # Write environment file using shared writer
            env_file = fm.get_path(f'{base_name}.env')
            self._log(f"Writing environment file: {env_file}", level='info')

            BellhopEnvWriter.write_env_file(
                filepath=env_file,
                env=env,
                source=source,
                receiver=receiver,
                run_type=run_type,
                beam_type=beam_type,
                source_type=source_type,
                grid_type=grid_type,
                beam_shift=beam_shift,
                volume_attenuation=volume_attenuation,
                verbose=self.verbose,
                n_beams=n_beams,
                alpha=alpha,
                step=step,
                z_box=z_box,
                r_box=r_box,
                **kwargs
            )

            # Run Bellhop
            self._log(f"Running Bellhop...", level='info')
            self._run_bellhop(base_name, fm.work_dir)

            # Read output based on run type
            if run_type.upper() in ['C', 'I', 'S']:
                # TL output
                output_file = fm.get_path(f'{base_name}.shd')
                if output_file.exists():
                    result = read_shd_file(output_file)
                else:
                    self._log(f"Output file not found: {output_file}", level='error')
                    raise FileNotFoundError(f"Output file not found: {output_file}")

            elif run_type.upper() == 'A':
                # Arrivals output
                output_file = fm.get_path(f'{base_name}.arr')
                if output_file.exists():
                    result = read_arr_file(output_file)
                else:
                    self._log(f"Output file not found: {output_file}", level='error')
                    raise FileNotFoundError(f"Output file not found: {output_file}")

            elif run_type.upper() in ['R', 'E']:
                # Ray trace output
                output_file = fm.get_path(f'{base_name}.ray')
                if output_file.exists():
                    result = read_ray_file(output_file)
                else:
                    self._log(f"Output file not found: {output_file}", level='error')
                    raise FileNotFoundError(f"Output file not found: {output_file}")
            else:
                self._log(f"Unknown run_type: {run_type}", level='error')
                raise ValueError(f"Unknown run_type: {run_type}")

            self._log("Simulation complete", level='info')
            return result

        finally:
            # Cleanup if configured
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _compute_tl_impl(self, env, source, receiver, **kwargs):
        """Bellhop-specific TL computation"""
        # Map RunMode if provided, otherwise default to coherent
        from uacpy.core.model_utils import ParameterMapper
        run_mode = kwargs.pop('run_mode', 'C')

        if hasattr(run_mode, 'value'):  # Is RunMode enum
            run_type = ParameterMapper.map_run_mode_to_bellhop(run_mode)
        else:
            run_type = run_mode

        # Ensure we're computing TL (C, I, or S)
        if run_type not in ['C', 'I', 'S']:
            run_type = 'C'

        return self.run(env, source, receiver, run_type=run_type, **kwargs)

    def run_with_bounce(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        cmin: float = DEFAULT_C_MIN,
        cmax: float = DEFAULT_C_MAX,
        rmax_km: float = 10.0,
        **kwargs
    ) -> Field:
        """
        Run Bellhop using BOUNCE-generated reflection coefficients.

        Automatically runs BOUNCE first to compute bottom reflection
        coefficients for the environment's bottom properties, then runs
        Bellhop using the resulting .brc file. This provides accurate
        handling of elastic/layered bottoms that Bellhop cannot model
        directly.

        Parameters
        ----------
        env : Environment
            Ocean environment (bottom properties define the layer stack)
        source : Source
            Acoustic source
        receiver : Receiver
            Receiver array
        cmin : float, optional
            Minimum phase velocity for reflection table (m/s). Default 1400.
        cmax : float, optional
            Maximum phase velocity for reflection table (m/s). Default 10000.
        rmax_km : float, optional
            Maximum range for angular resolution (km). Default 10.
        **kwargs
            Additional parameters passed to Bellhop.run()

        Returns
        -------
        field : Field
            Bellhop simulation results using reflection coefficients
        """
        from uacpy.models.bounce import Bounce
        import copy

        self._log("Running BOUNCE to compute reflection coefficients...", level='info')
        bounce = Bounce(verbose=self.verbose)
        bounce_result = bounce.run(
            env, source, receiver,
            cmin=cmin, cmax=cmax, rmax_km=rmax_km
        )

        brc_file = bounce_result.metadata.get('brc_file')
        if not brc_file:
            raise RuntimeError("BOUNCE did not produce a .brc file")

        # Create environment copy with reflection file bottom
        env_bounce = copy.deepcopy(env)
        env_bounce.bottom = BoundaryProperties(
            acoustic_type='file',
            reflection_file=brc_file,
            reflection_cmin=cmin,
            reflection_cmax=cmax,
            reflection_rmax_km=rmax_km,
            depth=env.bottom.depth
        )
        env_bounce.bottom_rd = None
        env_bounce.bottom_layered = None

        self._log("Running Bellhop with BOUNCE reflection coefficients...", level='info')
        return self.run(env_bounce, source, receiver, **kwargs)

    def _run_broadband(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        frequencies: Optional[np.ndarray] = None,
        source_waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        time_window: Optional[float] = None,
        n_freqs: int = 128,
        bandwidth_factor: float = 1.0,
        **kwargs
    ) -> Field:
        """
        Run Bellhop in broadband mode to produce a time-series or
        transfer function result.

        Uses the ray/beam approach described in the Bellhop User Guide (Sec 9):
        run arrivals at center frequency, then either:

        1. **Without source_waveform**: build frequency-domain transfer function
           H(f) from arrivals → returns Field(field_type='transfer_function').
           Use field.to_time_domain() for subsequent IFFT.

        2. **With source_waveform**: perform delay-and-sum convolution of the
           time-domain waveform with each arrival (amplitude, phase, delay)
           → returns Field(field_type='time_series') directly.

        This is a key advantage of ray tracing: the ray geometry (paths, travel
        times) is frequency-independent, so a single arrivals calculation at fc
        provides the impulse response.

        Parameters
        ----------
        env : Environment
            Ocean environment
        source : Source
            Acoustic source (source.frequency is the center frequency)
        receiver : Receiver
            Receiver array
        frequencies : ndarray, optional
            Explicit frequency vector in Hz for the transfer function.
            If None, generates n_freqs points over [fc*(1-bw_factor),
            fc*(1+bw_factor)]. Ignored when source_waveform is provided.
        source_waveform : ndarray, optional
            Time-domain source waveform (1D array). When provided, the
            delay-and-sum method is used: the waveform is convolved with
            each arrival using proper phase shifts via the Hilbert
            transform (analytic signal). Requires sample_rate.
        sample_rate : float, optional
            Sample rate in Hz. Required when source_waveform is provided.
            For transfer function mode, ignored.
        time_window : float, optional
            Duration of the output time window in seconds. If None,
            estimated automatically from arrival delays.
        n_freqs : int, optional
            Number of frequency points for transfer function mode.
            Default: 128. Ignored when source_waveform is provided.
        bandwidth_factor : float, optional
            Fractional bandwidth around fc. bandwidth_factor=1.0 means
            [0, 2*fc]. Default: 1.0. Ignored when source_waveform is provided.
        **kwargs
            Additional Bellhop parameters (beam_type, etc.)

        Returns
        -------
        field : Field
            If source_waveform is None: field_type='transfer_function'
                (call field.to_time_domain() to get time series)
            If source_waveform is provided: field_type='time_series'
                with data shape (n_depths, n_samples, n_ranges) and
                metadata containing 'time', 'dt', 'fs'.
        """
        fc = float(np.atleast_1d(source.frequency)[0])

        # Step 1: Run Bellhop in arrivals mode
        self._log("Broadband mode: running Bellhop in arrivals mode...", level='info')
        arr_field = self.run(env, source, receiver, run_type='A', **kwargs)

        arrivals_by_rcv = arr_field.metadata['arrivals_by_receiver']
        sz = arr_field.metadata['source_depths']
        rz = arr_field.metadata['receiver_depths']
        rr = arr_field.metadata['receiver_ranges']  # in meters

        nrd = len(rz)
        nrr = len(rr)

        # ── Path A: time-domain delay-and-sum with source waveform ──
        if source_waveform is not None:

            if sample_rate is None:
                raise ValueError(
                    "sample_rate is required when source_waveform is provided"
                )

            # Find the single receiver position
            rcv_depth = kwargs.pop('depth', None)
            rcv_range = kwargs.pop('range_m', None)

            if rcv_depth is None:
                ird = 0
                rcv_depth = float(rz[0])
            else:
                ird = int(np.argmin(np.abs(rz - rcv_depth)))
                rcv_depth = float(rz[ird])

            if rcv_range is None:
                irr = 0
                rcv_range = float(rr[0])
            else:
                irr = int(np.argmin(np.abs(rr - rcv_range)))
                rcv_range = float(rr[irr])

            self._log(f"Broadband: delay-and-sum at depth={rcv_depth:.1f} m, "
                      f"range={rcv_range:.1f} m", level='info')

            rcv_arr = arrivals_by_rcv[0][ird][irr]
            rts, t_vec = delayandsum(
                rcv_arrivals=rcv_arr,
                source_timeseries=source_waveform,
                sample_rate=sample_rate,
                fc=fc,
                time_window=time_window,
            )

            dt = 1.0 / sample_rate

            return Field(
                field_type='time_series',
                data=rts,
                ranges=np.array([rcv_range]),
                depths=np.array([rcv_depth]),
                frequencies=np.array([fc]),
                metadata={
                    'model': 'bellhop',
                    'time': t_vec,
                    'dt': dt,
                    'fs': sample_rate,
                    'nt': len(rts),
                    't_start': float(t_vec[0]),
                    'depth': rcv_depth,
                    'range': rcv_range,
                    'center_frequency': fc,
                    'source_depths': sz,
                },
            )

        # ── Path B: frequency-domain transfer function ──
        if frequencies is None:
            f_min = max(1.0, fc * (1.0 - bandwidth_factor))
            f_max = fc * (1.0 + bandwidth_factor)
            frequencies = np.linspace(f_min, f_max, n_freqs)

        frequencies = np.asarray(frequencies, dtype=float)
        n_freq = len(frequencies)

        # Build H(f) for each (source, receiver_depth, receiver_range)
        # Use first source depth (most common case)
        # Output shape: (n_depths, n_freq, n_ranges) to match Field convention
        H = np.zeros((nrd, n_freq, nrr), dtype=complex)

        for ird in range(nrd):
            for irr in range(nrr):
                rcv_arr = arrivals_by_rcv[0][ird][irr]
                H[ird, :, irr] = self._arrivals_to_tf(
                    rcv_arr, frequencies, fc
                )

        self._log(f"Broadband: built transfer function "
                  f"({nrd} depths x {n_freq} freqs x {nrr} ranges)",
                  level='info')

        c0 = float(env.ssp_data[0, 1]) if hasattr(env, 'ssp_data') and env.ssp_data is not None else DEFAULT_SOUND_SPEED

        return Field(
            field_type='transfer_function',
            data=H,
            ranges=rr,
            depths=rz,
            frequencies=frequencies,
            metadata={
                'model': 'bellhop',
                'center_frequency': fc,
                'source_depths': sz,
                'arrivals_field': arr_field,
                'c0': c0,
            },
        )

    @staticmethod
    def _arrivals_to_tf(
        rcv_arrivals: dict,
        frequencies: np.ndarray,
        fc: float,
    ) -> np.ndarray:
        """
        Build frequency-domain transfer function from per-receiver arrivals.

        For each arrival with amplitude A, phase phi (deg), travel time tau,
        and imaginary delay tau_i (volume attenuation), the contribution to
        H(f) is:

            H(f) += A * exp(-tau_i * 2*pi*f/fc) * exp(i*(phi_rad - 2*pi*f*tau))

        The phase from Bellhop already includes the geometric phase (number of
        caustics, boundary reflections). The exponential delay term shifts the
        arrival in the frequency domain. The imaginary delay encodes
        frequency-dependent volume attenuation scaled from the center frequency.

        Parameters
        ----------
        rcv_arrivals : dict
            Per-receiver arrival data with keys: amplitudes, phases, delays,
            delay_imag, n_arrivals.
        frequencies : ndarray
            Frequency vector in Hz.
        fc : float
            Center frequency in Hz (used for attenuation scaling).

        Returns
        -------
        H : ndarray
            Complex transfer function, shape (n_freq,).
        """
        n_arr = rcv_arrivals['n_arrivals']
        if n_arr == 0:
            return np.zeros(len(frequencies), dtype=complex)

        amps = rcv_arrivals['amplitudes']
        phases_deg = rcv_arrivals['phases']
        delays = rcv_arrivals['delays']
        delay_imag = rcv_arrivals['delay_imag']

        phases_rad = np.deg2rad(phases_deg)
        omega = 2.0 * np.pi * frequencies  # (n_freq,)

        H = np.zeros(len(frequencies), dtype=complex)

        for ia in range(n_arr):
            # Complex amplitude with geometric phase
            A_complex = amps[ia] * np.exp(1j * phases_rad[ia])

            # Volume attenuation: imaginary delay scales with frequency
            # relative to center frequency (Bellhop convention)
            if delay_imag[ia] != 0.0:
                atten = np.exp(-delay_imag[ia] * omega / (2.0 * np.pi * fc))
            else:
                atten = 1.0

            # Phase delay: exp(-i * 2*pi*f * tau)
            phase_shift = np.exp(-1j * omega * delays[ia])

            H += A_complex * atten * phase_shift

        return H

    def _run_bellhop(self, base_name: str, work_dir: Path):
        """
        Execute bellhop

        Parameters
        ----------
        base_name : str
            Base filename (without extension)
        work_dir : Path
            Working directory
        """
        # Build command based on version
        if self.version in ['cuda', 'cxx']:
            # bellhopcxx/cuda need dimensionality flag
            cmd = [str(self.executable), '--2D', base_name]
        else:
            # Fortran version
            cmd = [str(self.executable), base_name]

        result = subprocess.run(
            cmd,
            cwd=str(work_dir),
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            error_msg = f"Bellhop ({self.version}) failed with return code {result.returncode}\n"
            error_msg += f"stdout: {result.stdout}\n"
            error_msg += f"stderr: {result.stderr}"
            self._log(error_msg, level='error')
            raise RuntimeError(error_msg)

        # Log detailed output at debug level
        if self.verbose and result.stdout:
            self._log(f"Bellhop output:\n{result.stdout}", level='debug')


class BellhopCUDA(PropagationModel):
    """
    BellhopCUDA - GPU-accelerated ray tracing model

    C++/CUDA port of Bellhop with massive speedups on NVIDIA GPUs.

    Parameters
    ----------
    use_gpu : bool, optional
        Use GPU if available. Default is True.
    executable : str or Path, optional
        Path to bellhopcuda executable. If None, searches automatically.
    use_tmpfs : bool, optional
        Use RAM filesystem. Default is False.
    verbose : bool, optional
        Verbose output. Default is False.

    Examples
    --------
    >>> bhc = BellhopCUDA(use_gpu=True)
    >>> result = bhc.run(env, source, receiver, run_type='C')
    """

    def __init__(
        self,
        use_gpu: bool = True,
        executable: Optional[Path] = None,
        beam_type: str = 'R',
        n_beams: int = 0,
        alpha: tuple = (-80, 80),
        step: float = 0.0,
        z_box: Optional[float] = None,
        r_box: Optional[float] = None,
        source_type: str = 'R',
        grid_type: str = 'R',
        beam_shift: bool = False,
        volume_attenuation: Optional[str] = None,
        dimensionality: str = '2D',
        use_tmpfs: bool = False,
        verbose: bool = False,
    ):
        super().__init__(use_tmpfs=use_tmpfs, verbose=verbose)

        # Declare supported modes for BellhopCUDA (same as Bellhop)
        self._supported_modes = [
            RunMode.COHERENT_TL,
            RunMode.INCOHERENT_TL,
            RunMode.SEMICOHERENT_TL,
            RunMode.RAYS,
            RunMode.EIGENRAYS,
            RunMode.ARRIVALS,
            RunMode.TIME_SERIES,
        ]
        self._supports_altimetry = True

        self.use_gpu = use_gpu
        self.beam_type = beam_type
        self.n_beams = n_beams
        self.alpha = alpha
        self.step = step
        self.z_box = z_box
        self.r_box = r_box
        self.source_type = source_type
        self.grid_type = grid_type
        self.beam_shift = beam_shift
        self.volume_attenuation = volume_attenuation
        self.dimensionality = dimensionality

        if executable is None:
            if use_gpu:
                self.executable = self._find_executable('bellhopcuda')
            else:
                self.executable = self._find_executable('bellhopcxx')
        else:
            self.executable = Path(executable)

        if not self.executable.exists():
            raise FileNotFoundError(
                f"BellhopCUDA executable not found: {self.executable}\n"
                "Please run install.sh to build BellhopCUDA."
            )

    def _find_executable(self, name: str) -> Path:
        """Find BellhopCUDA executable"""
        base_dir = Path(__file__).parent.parent

        # Check uacpy/bin/bellhopcuda (new location)
        bin_paths = [
            base_dir / 'bin' / 'bellhopcuda' / f'{name}.exe',
            base_dir / 'bin' / 'bellhopcuda' / name,
            base_dir / 'bin' / 'bellhop' / name,
        ]

        for bin_path in bin_paths:
            if bin_path.exists():
                return bin_path

        # Check PATH (cross-platform)
        import shutil
        result = shutil.which(name)
        if result:
            return Path(result)

        # Try reference_code build
        ref_path = Path('reference_code') / 'bellhopcuda' / 'build' / name
        if ref_path.exists():
            return ref_path

        raise FileNotFoundError(
            f"Could not find {name}.\n"
            "Please run install_oalib.sh (Linux/Mac) or install_oalib.bat (Windows) with CUDA option."
        )

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_type: str = 'C',
        beam_type=_UNSET,
        n_beams=_UNSET,
        alpha=_UNSET,
        step=_UNSET,
        z_box=_UNSET,
        r_box=_UNSET,
        source_type=_UNSET,
        grid_type=_UNSET,
        beam_shift=_UNSET,
        volume_attenuation=_UNSET,
        **kwargs
    ) -> Field:
        """
        Run BellhopCUDA simulation

        Parameters
        ----------
        env : Environment
            Ocean environment
        source : Source
            Acoustic source
        receiver : Receiver
            Receiver array
        run_type : str, optional
            Run type: 'C'/'I'/'S'/'A'/'E'/'R'. Default is 'C'.
        beam_type, n_beams, alpha, step, z_box, r_box, source_type, grid_type,
        beam_shift, volume_attenuation : optional
            Per-call overrides for constructor defaults.
        **kwargs
            Additional parameters (same as Bellhop class)

        Returns
        -------
        field : Field
            Simulation results
        """
        # Resolve per-call overrides
        beam_type = beam_type if beam_type is not _UNSET else self.beam_type
        n_beams = n_beams if n_beams is not _UNSET else self.n_beams
        alpha = alpha if alpha is not _UNSET else self.alpha
        step = step if step is not _UNSET else self.step
        z_box = z_box if z_box is not _UNSET else self.z_box
        r_box = r_box if r_box is not _UNSET else self.r_box
        source_type = source_type if source_type is not _UNSET else self.source_type
        grid_type = grid_type if grid_type is not _UNSET else self.grid_type
        beam_shift = beam_shift if beam_shift is not _UNSET else self.beam_shift
        volume_attenuation = volume_attenuation if volume_attenuation is not _UNSET else self.volume_attenuation

        # Handle run_mode if provided (for compute_*() methods)
        if 'run_mode' in kwargs:
            from uacpy.core.model_utils import ParameterMapper
            run_mode = kwargs.pop('run_mode')
            if run_mode == RunMode.TIME_SERIES:
                return self._run_broadband(env, source, receiver, **kwargs)
            run_type = ParameterMapper.map_run_mode_to_bellhop(run_mode)

        self.validate_inputs(env, source, receiver)

        # Setup file manager
        fm = self._setup_file_manager()
        self.file_manager = fm

        try:
            base_name = 'model'

            # Write environment file using shared writer
            env_file = fm.get_path(f'{base_name}.env')
            self._log(f"Writing environment file: {env_file}", level='info')

            BellhopEnvWriter.write_env_file(
                filepath=env_file,
                env=env,
                source=source,
                receiver=receiver,
                run_type=run_type,
                beam_type=beam_type,
                source_type=source_type,
                grid_type=grid_type,
                beam_shift=beam_shift,
                volume_attenuation=volume_attenuation,
                verbose=self.verbose,
                n_beams=n_beams,
                alpha=alpha,
                step=step,
                z_box=z_box,
                r_box=r_box,
                **kwargs
            )

            # Run BellhopCUDA
            self._log(f"Running BellhopCUDA ({self.dimensionality})...", level='info')
            self._run_bellhopcuda(base_name, fm.work_dir, self.dimensionality)

            # Read output (same format as Bellhop)
            if run_type.upper() in ['C', 'I', 'S']:
                output_file = fm.get_path(f'{base_name}.shd')
                result = read_shd_file(output_file)
            elif run_type.upper() == 'A':
                output_file = fm.get_path(f'{base_name}.arr')
                result = read_arr_file(output_file)
            elif run_type.upper() in ['R', 'E']:
                output_file = fm.get_path(f'{base_name}.ray')
                result = read_ray_file(output_file)
            else:
                raise ValueError(f"Unknown run_type: {run_type}")

            self._log("Simulation complete", level='info')
            return result

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _run_broadband(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        frequencies: Optional[np.ndarray] = None,
        source_waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        time_window: Optional[float] = None,
        n_freqs: int = 128,
        bandwidth_factor: float = 1.0,
        dimensionality: str = '2D',
        **kwargs
    ) -> Field:
        """
        Run BellhopCUDA in broadband mode. See Bellhop._run_broadband for details.

        Uses BellhopCUDA for the arrivals computation, then delegates to
        Bellhop's broadband synthesis (transfer function or delay-and-sum).
        """
        fc = float(np.atleast_1d(source.frequency)[0])

        self._log("Broadband mode: running BellhopCUDA in arrivals mode...",
                  level='info')
        arr_field = self.run(env, source, receiver, run_type='A',
                            dimensionality=dimensionality, **kwargs)

        arrivals_by_rcv = arr_field.metadata['arrivals_by_receiver']
        sz = arr_field.metadata['source_depths']
        rz = arr_field.metadata['receiver_depths']
        rr = arr_field.metadata['receiver_ranges']

        nrd = len(rz)
        nrr = len(rr)

        # ── Path A: time-domain delay-and-sum ──
        if source_waveform is not None:

            if sample_rate is None:
                raise ValueError(
                    "sample_rate is required when source_waveform is provided"
                )

            rcv_depth = kwargs.pop('depth', None)
            rcv_range = kwargs.pop('range_m', None)

            if rcv_depth is None:
                ird = 0
                rcv_depth = float(rz[0])
            else:
                ird = int(np.argmin(np.abs(rz - rcv_depth)))
                rcv_depth = float(rz[ird])

            if rcv_range is None:
                irr = 0
                rcv_range = float(rr[0])
            else:
                irr = int(np.argmin(np.abs(rr - rcv_range)))
                rcv_range = float(rr[irr])

            self._log(f"Broadband: delay-and-sum at depth={rcv_depth:.1f} m, "
                      f"range={rcv_range:.1f} m", level='info')

            rcv_arr = arrivals_by_rcv[0][ird][irr]
            rts, t_vec = delayandsum(
                rcv_arrivals=rcv_arr,
                source_timeseries=source_waveform,
                sample_rate=sample_rate,
                fc=fc,
                time_window=time_window,
            )

            dt = 1.0 / sample_rate

            return Field(
                field_type='time_series',
                data=rts,
                ranges=np.array([rcv_range]),
                depths=np.array([rcv_depth]),
                frequencies=np.array([fc]),
                metadata={
                    'model': 'bellhopcuda',
                    'time': t_vec,
                    'dt': dt,
                    'fs': sample_rate,
                    'nt': len(rts),
                    't_start': float(t_vec[0]),
                    'depth': rcv_depth,
                    'range': rcv_range,
                    'center_frequency': fc,
                    'source_depths': sz,
                },
            )

        # ── Path B: frequency-domain transfer function ──
        if frequencies is None:
            f_min = max(1.0, fc * (1.0 - bandwidth_factor))
            f_max = fc * (1.0 + bandwidth_factor)
            frequencies = np.linspace(f_min, f_max, n_freqs)

        frequencies = np.asarray(frequencies, dtype=float)
        n_freq = len(frequencies)

        H = np.zeros((nrd, n_freq, nrr), dtype=complex)
        for ird in range(nrd):
            for irr in range(nrr):
                rcv_arr = arrivals_by_rcv[0][ird][irr]
                H[ird, :, irr] = Bellhop._arrivals_to_tf(
                    rcv_arr, frequencies, fc
                )

        self._log(f"Broadband: built transfer function "
                  f"({nrd} depths x {n_freq} freqs x {nrr} ranges)",
                  level='info')

        c0 = float(env.ssp_data[0, 1]) if hasattr(env, 'ssp_data') and env.ssp_data is not None else DEFAULT_SOUND_SPEED

        return Field(
            field_type='transfer_function',
            data=H,
            ranges=rr,
            depths=rz,
            frequencies=frequencies,
            metadata={
                'model': 'bellhopcuda',
                'center_frequency': fc,
                'source_depths': sz,
                'arrivals_field': arr_field,
                'c0': c0,
            },
        )

    def _run_bellhopcuda(self, base_name: str, work_dir: Path, dimensionality: str):
        """Execute BellhopCUDA"""
        cmd = [str(self.executable), f'--{dimensionality}', base_name]

        result = subprocess.run(
            cmd,
            cwd=str(work_dir),
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            error_msg = f"BellhopCUDA failed with return code {result.returncode}\n"
            error_msg += f"stdout: {result.stdout}\n"
            error_msg += f"stderr: {result.stderr}"
            self._log(error_msg, level='error')
            raise RuntimeError(error_msg)

        # Log detailed output at debug level
        if self.verbose and result.stdout:
            self._log(f"BellhopCUDA output:\n{result.stdout}", level='debug')
