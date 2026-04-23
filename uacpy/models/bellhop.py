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

import warnings
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

from scipy.signal import hilbert

from uacpy.models.base import PropagationModel, RunMode, _UNSET, _resolve_overrides
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
    >>> result = bellhop.run(env, source, receiver, run_mode=RunMode.COHERENT_TL)

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
        volume_attenuation: Optional[str] = None,
        attenuation_unit: str = 'W',
        francois_garrison_params: Optional[tuple] = None,
        bio_layers: Optional[list] = None,
        bty_interp_type: str = 'L',
        source_beam_pattern_file: Optional[Path] = None,
        arrivals_format: str = 'ascii',
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
            Number of beams. Passing 0 defers to Bellhop's conservative
            auto-selection (NBEAMS<=0 in the Fortran reader); uacpy
            writes the value through as-is rather than substituting
            ``source.n_angles``. Default: 0.
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
        volume_attenuation : str, optional
            'T' (Thorp), 'F' (Francois-Garrison), 'B' (Biological). Default: None.
        attenuation_unit : str, optional
            Attenuation unit code (TopOpt pos. 3). 'W' dB/wavelength (default),
            'N' nepers/m, 'F' dB/kmHz, 'M' dB/m, 'Q' Q-factor, 'L' loss
            tangent, 'm' dB/m with per-SSP BETA/fT pair.
        francois_garrison_params : tuple, optional
            Defaults used for ``volume_attenuation='F'``: (T, S, pH, z_bar).
        bio_layers : list, optional
            Defaults used for ``volume_attenuation='B'``: list of
            (Z1, Z2, f0, Q, a0).
        bty_interp_type : str, optional
            Interpolation type used for BOTH ``.bty`` (bathymetry) and
            ``.ati`` (altimetry) files. 'L' (linear, default) or 'C'
            (curvilinear).
        source_beam_pattern_file : Path or ndarray, optional
            Source beam pattern. Either a path to an existing ``.sbp`` file
            (copied to ``<work_dir>/<base>.sbp``) or a 2-column array of
            ``(angle_deg, level_dB)`` pairs (written via
            ``write_source_beam_pattern``; Bellhop converts dB -> linear
            internally, bellhop.f90:132). When set, RunType position 3 is
            set to ``'*'`` so Bellhop reads the file. Default: None
            (omnidirectional).
        arrivals_format : str, optional
            Format for ``RunMode.ARRIVALS`` output. ``'ascii'`` (default) maps
            to RunType 'A'; ``'binary'`` maps to 'a' (Fortran unformatted).
            The arrivals reader auto-detects format on read.
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
        self.volume_attenuation = volume_attenuation
        self.attenuation_unit = attenuation_unit
        self.francois_garrison_params = francois_garrison_params
        self.bio_layers = bio_layers
        self.bty_interp_type = bty_interp_type
        self.source_beam_pattern_file = (
            Path(source_beam_pattern_file)
            if isinstance(source_beam_pattern_file, (str, Path))
            else source_beam_pattern_file
        )
        if arrivals_format not in ('ascii', 'binary'):
            raise ValueError(
                f"arrivals_format must be 'ascii' or 'binary', got "
                f"{arrivals_format!r}"
            )
        self.arrivals_format = arrivals_format
        self.version = "unknown"

        if executable is None:
            self.executable = self._find_bellhop_executable()
        else:
            self.executable = Path(executable)
            self.version = "custom"

        if not self.executable.exists():
            from uacpy.core.exceptions import ExecutableNotFoundError
            raise ExecutableNotFoundError("Bellhop", str(self.executable))

        if verbose and self.version != "custom":
            self._log(f"Using Bellhop {self.version}: {self.executable}", level='info')

    def _find_bellhop_executable(self) -> Path:
        """Locate the Bellhop binary using the base class helper.

        Preference order honors ``prefer_cuda``: CUDA > C++ > Fortran.
        ``self.version`` is inferred from the name of the returned path.
        """
        if self.prefer_cuda:
            names = ['bellhopcuda', 'bellhopcxx', 'bellhop']
        else:
            names = ['bellhop']

        path = self._find_executable_in_paths(
            names,
            bin_subdirs=['bellhopcuda', 'oalib', 'bellhop'],
            dev_subdir='Acoustics-Toolbox/Bellhop',
        )
        lower = path.name.lower()
        if 'cuda' in lower:
            self.version = 'cuda'
        elif 'cxx' in lower:
            self.version = 'cxx'
        else:
            self.version = 'fortran'
        return path

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode=_UNSET,
        beam_type=_UNSET,
        n_beams=_UNSET,
        alpha=_UNSET,
        step=_UNSET,
        z_box=_UNSET,
        r_box=_UNSET,
        source_type=_UNSET,
        grid_type=_UNSET,
        volume_attenuation=_UNSET,
        attenuation_unit=_UNSET,
        francois_garrison_params=_UNSET,
        bio_layers=_UNSET,
        bty_interp_type=_UNSET,
        source_beam_pattern_file=_UNSET,
        arrivals_format=_UNSET,
        **kwargs
    ) -> Field:
        """
        Run Bellhop simulation

        Parameters
        ----------
        env, source, receiver : see ``PropagationModel.run``.
        run_mode : RunMode, optional
            Which Bellhop mode to run. One of ``RunMode.COHERENT_TL``,
            ``INCOHERENT_TL``, ``SEMICOHERENT_TL``, ``RAYS``, ``EIGENRAYS``,
            ``ARRIVALS``, ``TIME_SERIES``. Defaults to ``COHERENT_TL``.
        beam_type, n_beams, alpha, step, z_box, r_box, source_type,
        grid_type, volume_attenuation : optional
            Per-call overrides for the constructor defaults.
        attenuation_unit : str, optional
            Top-option position 3. 'N'/'F'/'M'/'W' (default)/'Q'/'L'/'m'.
        francois_garrison_params : tuple, optional
            Required when ``volume_attenuation='F'``:
            ``(T_C, salinity_ppt, pH, z_bar_m)``.
        bio_layers : list of tuples, optional
            Required when ``volume_attenuation='B'``. Each entry is
            ``(Z1, Z2, f0, Q, a0)``.
        bty_interp_type : str, optional
            '.bty'/'.ati' interpolation: 'L' (linear, default) or 'C'
            (curvilinear).
        source_beam_pattern_file : Path or ndarray, optional
            Per-call override of the constructor default. See
            :meth:`__init__`. Supply ``None`` to explicitly disable for this
            call.
        arrivals_format : str, optional
            Per-call override: ``'ascii'`` or ``'binary'``. Only takes effect
            for ``run_mode=RunMode.ARRIVALS``.
        **kwargs
            Advanced Cerveny/Simple-Gaussian beam parameters (used when
            ``beam_type`` is 'C', 'R' or 'S'):
            beam_width_type ('F'/'M'/'W'), beam_curvature ('D'/'S'/'Z'),
            eps_multiplier, r_loop, n_image, ib_win, component.

        Returns
        -------
        field : Field
            Simulation results
        """
        # ── Resolve run_mode → internal single-char Bellhop code ────────
        from uacpy.core.model_utils import ParameterMapper
        if run_mode is _UNSET:
            run_mode = RunMode.COHERENT_TL
        if run_mode == RunMode.TIME_SERIES:
            # Delegate to broadband pipeline (arrivals + synthesis)
            return self._run_broadband(env, source, receiver, **kwargs)
        run_type = ParameterMapper.map_run_mode_to_bellhop(run_mode)

        # ── arrivals_format: 'A' (ASCII) vs 'a' (binary) ────────────────
        # ParameterMapper returns 'A' for RunMode.ARRIVALS; swap to 'a'
        # when the user requests binary output.
        arr_fmt = (
            arrivals_format if arrivals_format is not _UNSET
            else self.arrivals_format
        )
        if arr_fmt not in ('ascii', 'binary'):
            raise ValueError(
                f"arrivals_format must be 'ascii' or 'binary', got {arr_fmt!r}"
            )
        if run_mode == RunMode.ARRIVALS and arr_fmt == 'binary':
            run_type = 'a'

        # ── Per-call overrides (shared primitive) ───────────────────────
        override_kwargs = dict(
            beam_type=beam_type, n_beams=n_beams, alpha=alpha,
            step=step, z_box=z_box, r_box=r_box,
            source_type=source_type, grid_type=grid_type,
            volume_attenuation=volume_attenuation,
        )

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

        # Irregular receiver grid ('I' in RunType position 5) requires the
        # receiver.depths and receiver.ranges arrays to have the same
        # length (they are paired point-by-point).  Rectilinear ('R')
        # takes the Cartesian product.  Catch the mismatch here so users
        # see a clear error instead of a confusing Bellhop .prt message.
        eff_grid_type = (
            grid_type if grid_type is not _UNSET else self.grid_type
        )
        if (
            eff_grid_type is not None
            and str(eff_grid_type).upper() == 'I'
            and len(receiver.depths) != len(receiver.ranges)
        ):
            from uacpy.core.exceptions import ConfigurationError
            raise ConfigurationError(
                f"Bellhop grid_type='I' (irregular) requires "
                f"len(receiver.depths) == len(receiver.ranges); got "
                f"{len(receiver.depths)} depths and "
                f"{len(receiver.ranges)} ranges. Use grid_type='R' for "
                f"a rectilinear (Cartesian-product) grid, or rebuild the "
                f"Receiver with matched arrays."
            )

        # Setup file manager
        fm = self._setup_file_manager()
        self.file_manager = fm

        # Writer kwargs that are not tied to the overrides context. Each is
        # pulled from the per-call override when supplied, otherwise from the
        # constructor default on self.
        extra_writer_kwargs = {
            'attenuation_unit': (
                attenuation_unit if attenuation_unit is not _UNSET
                else self.attenuation_unit
            ),
            'francois_garrison_params': (
                francois_garrison_params if francois_garrison_params is not _UNSET
                else self.francois_garrison_params
            ),
            'bio_layers': (
                bio_layers if bio_layers is not _UNSET
                else self.bio_layers
            ),
            'bty_interp_type': (
                bty_interp_type if bty_interp_type is not _UNSET
                else self.bty_interp_type
            ),
        }

        # Resolve source beam pattern (file path or (angle, level_dB) array).
        sbp_spec = (
            source_beam_pattern_file if source_beam_pattern_file is not _UNSET
            else self.source_beam_pattern_file
        )

        with _resolve_overrides(self, **override_kwargs):
            try:
                base_name = 'model'
                env_file = fm.get_path(f'{base_name}.env')
                self._log(f"Writing environment file: {env_file}", level='info')

                # Stage source beam pattern file. Bellhop reads by base name
                # (<base>.sbp) when RunType position 3 is '*'.
                use_sbp = False
                if sbp_spec is not None:
                    import shutil
                    sbp_dest = env_file.with_suffix('.sbp')
                    if isinstance(sbp_spec, (str, Path)):
                        src = Path(sbp_spec)
                        if not src.exists():
                            raise FileNotFoundError(
                                f"Source beam pattern file not found: {src}"
                            )
                        shutil.copy(src, sbp_dest)
                    else:
                        # Array-like: expect shape (N, 2) [angle_deg, level_dB]
                        from uacpy.io.env_writer import write_source_beam_pattern
                        arr = np.asarray(sbp_spec, dtype=float)
                        if arr.ndim != 2 or arr.shape[1] != 2:
                            raise ValueError(
                                "source_beam_pattern_file array must be shape "
                                "(N, 2): [angle_deg, level_dB]."
                            )
                        write_source_beam_pattern(
                            sbp_dest, arr[:, 0], arr[:, 1]
                        )
                    use_sbp = True
                    if self.verbose:
                        self._log(f"Wrote source beam pattern: {sbp_dest}",
                                  level='info')

                BellhopEnvWriter.write_env_file(
                    filepath=env_file,
                    env=env,
                    source=source,
                    receiver=receiver,
                    run_type=run_type,
                    beam_type=self.beam_type,
                    source_type=self.source_type,
                    grid_type=self.grid_type,
                    volume_attenuation=self.volume_attenuation,
                    verbose=self.verbose,
                    n_beams=self.n_beams,
                    alpha=self.alpha,
                    step=self.step,
                    z_box=self.z_box,
                    r_box=self.r_box,
                    source_beam_pattern=use_sbp,
                    **extra_writer_kwargs,
                    **kwargs,
                )

                # Run Bellhop
                self._log("Running Bellhop...", level='info')
                self._run_bellhop(base_name, fm.work_dir)

                # Read output based on run type. Uppercase covers 'A'
                # (ASCII arrivals) and 'a' (binary arrivals) identically
                # since the arrivals reader auto-detects the format.
                rt = run_type.upper()
                if rt in ('C', 'I', 'S'):
                    output_file = fm.get_path(f'{base_name}.shd')
                    reader = read_shd_file
                elif rt == 'A':
                    output_file = fm.get_path(f'{base_name}.arr')
                    reader = read_arr_file
                elif rt in ('R', 'E'):
                    output_file = fm.get_path(f'{base_name}.ray')
                    reader = read_ray_file
                else:
                    self._log(f"Unknown run_type: {run_type}", level='error')
                    raise ValueError(f"Unknown run_type: {run_type}")

                if not output_file.exists():
                    self._log(f"Output file not found: {output_file}", level='error')
                    raise FileNotFoundError(f"Output file not found: {output_file}")
                result = reader(output_file)

                self._log("Simulation complete", level='info')
                return result

            finally:
                if fm.cleanup:
                    fm.cleanup_work_dir()

    def _compute_tl_impl(self, env, source, receiver, **kwargs):
        """Bellhop-specific TL computation.

        ``run_mode`` must be a :class:`RunMode` member. If absent defaults to
        :attr:`RunMode.COHERENT_TL`; any non-TL mode is rejected.
        """
        run_mode = kwargs.pop('run_mode', RunMode.COHERENT_TL)
        if not isinstance(run_mode, RunMode):
            raise TypeError(
                f"run_mode must be a RunMode enum member, got "
                f"{type(run_mode).__name__}: {run_mode!r}"
            )
        if run_mode not in (RunMode.COHERENT_TL, RunMode.INCOHERENT_TL,
                            RunMode.SEMICOHERENT_TL):
            raise ValueError(
                f"compute_tl requires a TL RunMode "
                f"(COHERENT_TL/INCOHERENT_TL/SEMICOHERENT_TL); got {run_mode}."
            )
        return self.run(env, source, receiver, run_mode=run_mode, **kwargs)

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
        t_start: Optional[float] = None,
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
        t_start : float, optional
            Start time (s) of the delay-and-sum output window. Forwarded to
            :func:`delayandsum`. If None, set to just before the earliest
            arrival. Only used when ``source_waveform`` is provided.
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
        arr_field = self.run(env, source, receiver, run_mode=RunMode.ARRIVALS, **kwargs)

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
                t_start=t_start,
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

    def _build_command(self, base_name: str) -> list:
        """Build the argv used to launch the binary.

        Subclasses (BellhopCUDA) may override this to add flags.
        """
        if self.version in ('cuda', 'cxx'):
            # bellhopcxx/cuda require a dimensionality flag
            return [str(self.executable), '--2D', base_name]
        return [str(self.executable), base_name]

    def _run_bellhop(self, base_name: str, work_dir: Path):
        """Execute the Bellhop binary via the shared subprocess runner.

        Bellhop reports most fatal errors in ``<base>.prt`` rather than on
        stderr. If the child exits non-zero, we append the tail of the .prt
        file (up to 2000 chars) to the raised ``ModelExecutionError`` so the
        diagnostic surface to the user instead of a blank stderr.
        """
        from uacpy.core.exceptions import ModelExecutionError

        cmd = self._build_command(base_name)
        try:
            result = self._run_subprocess(cmd, cwd=work_dir)
        except ModelExecutionError as exc:
            prt_file = Path(work_dir) / f"{base_name}.prt"
            if prt_file.exists():
                try:
                    tail = prt_file.read_text()[-2000:]
                    exc.args = (
                        f"{exc.args[0] if exc.args else exc}\n\n"
                        f".prt tail:\n{tail}",
                    ) + exc.args[1:]
                except Exception:
                    pass
            raise

        if self.verbose and result.stdout:
            self._log(f"Bellhop output:\n{result.stdout}", level='debug')


class BellhopCUDA(Bellhop):
    """
    BellhopCUDA - C++/CUDA ray tracing model (thin ``Bellhop`` subclass).

    Shares all Environment/Source/Receiver plumbing, per-call overrides,
    broadband synthesis, and output parsing with the parent. Only the
    executable selection and the ``--<dim>`` invocation flag differ.

    Parameters
    ----------
    use_gpu : bool, optional
        Select ``bellhopcuda`` (True) vs ``bellhopcxx`` (False). Default True.
    executable : str or Path, optional
        Override path to the chosen binary. Auto-detected if None.
    dimensionality : str, optional
        Either '2D' or '3D'. Default '2D' (matches the CLI flag '--2D').
    **kwargs
        All other kwargs are forwarded to ``Bellhop.__init__`` unchanged.

    Examples
    --------
    >>> bhc = BellhopCUDA(use_gpu=True)
    >>> result = bhc.run(env, source, receiver, run_mode=RunMode.COHERENT_TL)
    """

    def __init__(
        self,
        use_gpu: bool = True,
        executable: Optional[Path] = None,
        dimensionality: str = '2D',
        **kwargs,
    ):
        # Stash CUDA-specific knobs before super() uses them in executable
        # discovery.
        self.use_gpu = use_gpu
        self.dimensionality = dimensionality
        super().__init__(
            executable=executable,
            prefer_cuda=True,  # always prefer GPU/CXX before Fortran
            **kwargs,
        )

    def _find_bellhop_executable(self) -> Path:
        """Override parent: pick only CUDA or CXX flavor, never Fortran."""
        names = ['bellhopcuda'] if self.use_gpu else ['bellhopcxx']
        path = self._find_executable_in_paths(
            names,
            bin_subdirs=['bellhopcuda'],
            dev_subdir='bellhopcuda',
        )
        self.version = 'cuda' if 'cuda' in path.name.lower() else 'cxx'
        return path

    def _build_command(self, base_name: str) -> list:
        """Always emit the ``--<dim>`` flag required by the CUDA/CXX CLI."""
        return [str(self.executable), f'--{self.dimensionality}', base_name]


class Bellhop3D(Bellhop):
    """
    Placeholder wrapper for BELLHOP3D (3D ray tracing).

    Full 3D support requires a separate env-file layout (NSx/NSy source
    grid, Ntheta/Nbeta bearing fan, 3D bathymetry via .bty3d, 3D SSP via
    .ssp hexahedral format, etc.) and a 3D-aware output reader path.
    See ``third_party/Acoustics-Toolbox/doc/bellhop3d.htm`` for the
    authoritative file format.

    This stub exists so users can discover the gap via ``from
    uacpy.models import Bellhop3D`` rather than silently falling back to
    2D. Constructing the class raises :class:`NotImplementedError`.

    Contributions welcome.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Bellhop3D wrapper is pending — the uacpy writer currently "
            "emits 2D env blocks only (BellhopEnvWriter hardcodes "
            "position 6 to '2'). Contributions welcome; see "
            "third_party/Acoustics-Toolbox/doc/bellhop3d.htm for the 3D "
            "env-file specification and bellhop3D.f90 for the Fortran "
            "reference."
        )
