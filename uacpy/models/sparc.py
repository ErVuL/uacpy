"""
SPARC - Seismo-Acoustic Propagation in Realistic oCeans

SPARC is an FFP (Fast Field Program) model that includes elastic bottom effects
and seismo-acoustic coupling. It uses the same wavenumber integration approach
as Scooter but with support for elastic media.
"""

import warnings
from pathlib import Path
from typing import Optional, Union
import numpy as np

from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.results import Result, Field
from uacpy.core.constants import (
    BoundaryType, AttenuationUnits,
    parse_boundary_type,
    DEFAULT_SOUND_SPEED,
)
from uacpy.core.exceptions import (
    ConfigurationError, ExecutableNotFoundError, ModelExecutionError,
    UnsupportedFeatureError,
)
from uacpy.io.grn_reader import read_grn_file, sparc_snapshot_to_field
from uacpy.models.base import PropagationModel, RunMode
from uacpy.io.oalib_reader import read_rts_file, rts_to_pressure
from uacpy.io.oalib_writer import (
    write_absorption_block, write_layer_sections,
    write_phase_speed_and_rmax, write_receiver_depths, write_source_depths,
    write_ssp_section,
)


# SPARC pulse_type alphabets (per Scooter/sparc.f90:126-148 GetPar SELECT CASE).
# Pos 1: pulse shape — the 10 letters accepted by sparc.f90's parser.
# ``T`` and ``C`` exist in tslib/cans.f90 (the pulse evaluator) but
# sparc.f90's GetPar rejects them with "Unknown source type" before
# cans.f90 is reached.
_PULSE_TYPE_POS1 = set('PRASHNGFBM')
# Pos 2: post-processing applied to the pulse samples.
#   'H' = pre-envelope (|analytic signal|), 'Q' = Hilbert transform.
#   Any other character (including ' ' or 'N') means "no transform".
_PULSE_TYPE_POS2 = {' ', 'N', 'H', 'Q'}
# Pos 3: sign flag. '-' inverts the pulse; any other character keeps it.
_PULSE_TYPE_POS3 = {' ', '+', '-'}
# Pos 4: filter option applied in march.f90 / Matlab march.m.
#   'N' / ' ' = no band-pass, 'L' = low-cut, 'H' = high-cut, 'B' = both.
_PULSE_TYPE_POS4 = {' ', 'N', 'L', 'H', 'B'}


def _validate_pulse_type(pulse_type: str) -> str:
    """
    Validate a 4-character SPARC pulse_type string.

    Parameters
    ----------
    pulse_type : str
        The raw string the user passed. Short strings are right-padded
        with spaces to length 4 (matching sparcM.m's handling).

    Returns
    -------
    pulse_type : str
        The normalized 4-character string.

    Raises
    ------
    ValueError
        If any character falls outside the alphabets documented in
        ``sparc.f90:140`` / ``tslib/sourceMod.f90`` / ``Matlab/Sparc/march.m``.
    """
    if not isinstance(pulse_type, str):
        raise ConfigurationError(
            f"pulse_type must be a string, got {type(pulse_type).__name__}"
        )
    if len(pulse_type) > 4:
        raise ConfigurationError(
            f"pulse_type must be at most 4 characters, got {pulse_type!r}"
        )
    pulse_type = pulse_type.ljust(4)

    def _bad(pos, char, allowed):
        return ConfigurationError(
            f"Invalid pulse_type character {char!r} at position {pos} "
            f"(must be one of {sorted(allowed)!r}). "
            f"See Acoustics-Toolbox/Scooter/sparc.f90."
        )

    if pulse_type[0] not in _PULSE_TYPE_POS1:
        raise _bad(1, pulse_type[0], _PULSE_TYPE_POS1)
    if pulse_type[1] not in _PULSE_TYPE_POS2:
        raise _bad(2, pulse_type[1], _PULSE_TYPE_POS2)
    if pulse_type[2] not in _PULSE_TYPE_POS3:
        raise _bad(3, pulse_type[2], _PULSE_TYPE_POS3)
    if pulse_type[3] not in _PULSE_TYPE_POS4:
        raise _bad(4, pulse_type[3], _PULSE_TYPE_POS4)
    return pulse_type


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
    - Horizontal-array mode runs one SPARC simulation per receiver depth
      (looped in the wrapper) — see ``max_depths`` for the safety cap.
    - Longer computation time due to time-domain integration

    Parameters
    ----------
    executable : Path, optional
        Path to ``sparc.exe``. Auto-detected if ``None``.
    c_low, c_high : float, optional
        Phase-speed bounds (m/s). ``None`` ⇒ auto. Default ``None``.
    n_mesh : int, optional
        Mesh points per medium. ``0`` ⇒ auto. Default ``0``.
    roughness : float, optional
        Bottom RMS roughness (m). Default ``0``.
    output_mode : str, optional
        ``'R'`` horizontal array (default) | ``'D'`` vertical array | ``'S'`` snapshot.
    pulse_type : str, optional
        4-character source-pulse code per ``sourceMod.f90``. Default ``'PN+B'``.
        Position 1 = pulse shape (``PRASHNGFBM`` — ``T``/``C`` are listed
        in ``cans.f90`` but rejected by ``sparc.f90``'s ``GetPar``),
        2 = post-process, 3 = sign, 4 = filter.

        ``cans.f90``'s ``'R'`` (Ricker) is defined on ``U = ω·T - 5``
        and therefore peaks at ``T ≈ 5/(2π·F) ≈ 0.796/F``, not at
        ``T = 0`` (the comment in ``cans.f90:32`` says "peak at F"
        meaning at *time* ``1/F``). When aligning expected vs measured
        arrivals downstream, treat the Ricker peak as offset by
        ``+5/(2π·F)`` from the source pulse origin.
    n_t_out : int, optional
        Number of output time samples. Default ``501``.
    t_max : float, optional
        Maximum simulated time (s). ``None`` ⇒ ``2.5 ×`` travel time.
    t_start : float, optional
        Integration start time (s). Default ``-0.1``.
    t_mult : float, optional
        Integration time multiplier. Default ``0.999``.
    max_depths : int, optional
        Cap on receiver depths (looped in wrapper). Default ``20``.
    rmax_safety_margin : float, optional
        Multiplier applied to ``receiver.ranges.max()`` to set SPARC's
        ``RMax``. Default ``1.0001`` — fine for ``COHERENT_TL`` because
        the downstream time-FFT averages the FFT-Hankel periodic image
        out. **For ``RunMode.TIME_SERIES`` use ~2 or more** so the
        periodic alias of the source falls outside the receiver array:
        the inverse Hankel transform (``TransformG.f90``) is FFT-based
        with period = ``RMax`` in range, so with the default margin the
        source at r=0 appears as a non-physical replica at
        r = receiver.ranges.max(). Increasing this knob roughly
        proportionally increases ``Nk`` and SPARC runtime.
    timeout : float, optional
        Subprocess timeout per run (s). Default ``180.0``.
    use_tmpfs, verbose, work_dir, cleanup, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    Range-independent time-marched FFP. Only ``Vacuum`` / ``Rigid``
    bottom interfaces are supported (the writer auto-converts
    halfspaces to rigid). ``RunMode.TIME_SERIES`` returns a
    :class:`Field` directly; SPARC drives its source pulse via the
    constructor ``pulse_type`` so passing ``source_waveform`` /
    ``sample_rate`` to ``run()`` emits a ``UserWarning`` (they have no
    effect on the SPARC simulation).

    ``output_mode='S'`` requires ``n_t_out`` large enough that the
    source frequency stays below the snapshot Nyquist (``0.5/dt``);
    the wrapper raises a ``ValueError`` otherwise.

    **Collapse defaults (overrides of :data:`DEFAULT_COLLAPSE`).**
    Per-model: ``'ssp': 'mean'``, ``'bottom': 'median'``,
    ``'rd_layered_layers': 'preserve'`` (SPARC consumes ``LayeredBottom``
    natively).

    Defaults auto-derived at ``run()`` time:

    - ``n_mesh=0`` → SPARC picks per frequency / wavelength.
    - ``c_low`` / ``c_high`` → from env SSP and bottom speed.
    - ``rmax`` written as ``receiver.range_max × rmax_safety_margin``.
    - ``dt`` / ``dr`` derived from CFL stability and the source pulse.
    - TopOpt position 4 reads ``env.absorption``.

    Examples
    --------
    >>> sparc = SPARC(verbose=False)
    >>> result = sparc.run(env, source, receiver)
    """

    def __init__(
        self,
        executable: Optional[Path] = None,
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        n_mesh: int = 0,
        roughness: float = 0.0,
        interp_ssp: Optional[str] = None,
        output_mode: str = 'R',
        pulse_type: str = 'PN+B',
        # Power-of-two so the downstream np.fft.fft on this length runs
        # in O(N log N) instead of the prime-factor 501=3*167 fallback.
        n_t_out: int = 512,
        t_max: Optional[float] = None,
        t_start: float = -0.1,
        t_mult: float = 0.999,
        max_depths: int = 20,
        rmax_safety_margin: Optional[float] = None,
        f_min: Optional[float] = None,
        f_max: Optional[float] = None,
        sound_speed: Optional[float] = None,
        timeout: float = 180.0,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        **kwargs,
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

            ``'S'`` mode time-FFTs the snapshot's tout axis and picks the
            source-frequency bin (``uacpy.io.grn_reader.sparc_snapshot_to_field``);
            ``n_t_out`` must be large enough that the source frequency
            stays below ``0.5/dt``.
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
        rmax_safety_margin : float, optional
            Multiplier on ``receiver.ranges.max()`` to set SPARC's RMax.
            SPARC's inverse Hankel transform (TransformG.f90) is FFT-based,
            so the r-domain output is periodic with period RMax — the
            source's image at r=RMax contaminates the receivers unless
            RMax is pushed well past them. Default: ``None`` → 1.0001
            (0.01%, round-off only) for COHERENT_TL; 3.0 (alias at 3×
            receiver max) for TIME_SERIES.
        timeout : float, optional
            Subprocess timeout (s) for each SPARC run. Default: 180.0.
        """
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            timeout=timeout, **kwargs,
        )

        self.c_low = c_low
        self.c_high = c_high
        if c_low is not None and c_high is not None and c_low >= c_high:
            raise ConfigurationError(
                f"SPARC spectral phase-velocity band requires "
                f"c_low < c_high; got c_low={c_low} m/s, c_high={c_high} m/s."
            )
        self.n_mesh = n_mesh
        self.roughness = roughness
        self.interp_ssp = interp_ssp
        if output_mode not in ('R', 'D', 'S'):
            raise ConfigurationError(
                f"Invalid output mode '{output_mode}'. "
                f"Valid modes: 'R' (horizontal array), 'D' (vertical array), 'S' (snapshot)"
            )
        self.output_mode = output_mode
        self.pulse_type = _validate_pulse_type(pulse_type)
        self.n_t_out = n_t_out
        self.t_max = t_max
        self.t_start = t_start
        self.t_mult = t_mult
        self.max_depths = max_depths
        self.rmax_safety_margin = rmax_safety_margin
        # Pulse band (``f_min``/``f_max``) and ``sound_speed`` (used for
        # the travel-time window when ``t_max`` is auto) default to
        # ``None`` and are resolved at ``run()`` time from
        # ``source.frequencies[0]`` and :data:`DEFAULT_SOUND_SPEED`.
        self.f_min = float(f_min) if f_min is not None else None
        self.f_max = float(f_max) if f_max is not None else None
        self.sound_speed = (
            float(sound_speed) if sound_speed is not None else None
        )

        # Declare supported modes for SPARC
        self._supported_modes = [
            RunMode.COHERENT_TL,
            RunMode.TIME_SERIES,
        ]
        # SPARC: range-independent time-marched FFP. Multi-layer fluid /
        # SPARC's run() auto-converts halfspace bottoms to rigid (line ~650),
        # so elastic_bottom flag is False — collapse to fluid up front so
        # the user gets a uniform warning instead of a silent rigidify.
        self._supports_altimetry = False
        self._supports_range_dependent_bathymetry = False
        self._supports_range_dependent_ssp = False
        self._supports_range_dependent_bottom = False
        self._supports_layered_bottom = True
        self._supports_range_dependent_layered_bottom = False
        self._supports_elastic_media = False
        self._supports_multi_source_depth = False
        # Range-independent time-marched FFP — single solve over the
        # whole spectrum. Median/mean samples represent the path.
        self._set_collapse_defaults({
            'ssp': 'mean',
            'bottom': 'median',
            'rd_layered_layers': 'preserve',
        })

        if executable is None:
            self.executable = self._find_executable_in_paths(
                'sparc.exe', bin_subdirs='oalib',
                dev_subdir='Acoustics-Toolbox/Scooter',
            )
        else:
            self.executable = Path(executable)

        if not self.executable.exists():
            raise ExecutableNotFoundError('SPARC', str(self.executable))

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[RunMode] = None,
        *,
        source_waveform=None,
        sample_rate=None,
    ) -> Result:
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

        Returns
        -------
        result : Result
            TL field (COHERENT_TL) or time-series field (TIME_SERIES)

        Notes
        -----
        SPARC is range-independent. If a range-dependent environment is provided,
        it will automatically use a median-depth approximation with a warning.
        """
        run_mode = self._resolve_run_mode(run_mode)

        # SPARC drives its source pulse via the constructor ``pulse_type``.
        # ``source_waveform`` / ``sample_rate`` exist on the signature for
        # API uniformity but cannot influence the run — warn loudly when
        # the caller supplies them so they don't expect their waveform to
        # propagate.
        if source_waveform is not None or sample_rate is not None:
            warnings.warn(
                "SPARC.run: source_waveform / sample_rate are ignored — "
                "SPARC builds p(t) from its native pulse_type. To shape "
                "the pulse, pass SPARC(pulse_type=...).",
                UserWarning, stacklevel=2,
            )
        env = self._project_environment(env)
        env = self._sparc_rigidify_halfspace(env)
        receiver = self._clip_receiver_depths(receiver, env.depth)

        # SPARC limitation: horizontal array mode requires one run per depth
        # For large depth grids, this becomes computationally expensive
        if len(receiver.depths) > self.max_depths:
            raise UnsupportedFeatureError(
                model_name='SPARC',
                feature=(
                    f"{len(receiver.depths)} receiver depths (SPARC horizontal "
                    f"array mode runs one simulation per depth; current limit is "
                    f"max_depths={self.max_depths})"
                ),
                alternatives=[
                    f"Reduce receiver.depths to at most {self.max_depths} entries",
                    f"Raise the limit explicitly: SPARC(max_depths={len(receiver.depths)})",
                    "Bellhop, RAM, Kraken, Scooter, or OASN for dense 2D fields",
                ],
            )

        self.validate_inputs(env, source, receiver, run_mode=run_mode)

        fm = self._setup_file_manager()
        self.file_manager = fm

        try:
            base_name = 'model'
            freq = source.frequencies[0]

            if self.output_mode == 'R':
                # SPARC computes horizontal arrays (one depth at a time)
                # For 2D fields, we need to run SPARC for each receiver depth

                if len(receiver.depths) == 1:
                    # Single depth - run once
                    self._log(f"Computing at depth {receiver.depths[0]:.1f}m...")

                    # Write environment file
                    env_file = fm.get_path(f'{base_name}.env')
                    self._write_sparc_env(env_file, env, source, receiver, run_mode)
                    self._run_sparc(base_name, fm.work_dir)
                    rts_file = fm.get_path(f'{base_name}.rts')
                    if not rts_file.exists():
                        exc = ModelExecutionError(
                            self.model_name, return_code=0, stdout=None,
                            stderr=f"SPARC did not produce {rts_file}",
                        )
                        self._attach_prt_tail(exc, fm.work_dir, base_name)
                        raise exc

                    rts_data = read_rts_file(rts_file)

                    if run_mode == RunMode.TIME_SERIES:
                        # rts_data['p'] is (nt, nr). New shape contract is
                        # (n_d, n_r, n_t) so swap axes 0↔1 and add the
                        # leading n_d=1 axis.
                        p_3d = np.asarray(rts_data['p']).T[None, :, :]
                        dt = rts_data['dt']
                        time = rts_data['time']
                        result = Field(
                            data=p_3d,
                            coords={
                                'depth': receiver.depths,
                                'range': rts_data['ranges'],
                                'time': time,
                            },
                            **self._result_kwargs(
                                source,
                                backend='sparc.exe',
                                frequencies=freq,
                                phase_reference='time_domain_native',
                                dt=float(dt),
                                fs=(1.0 / float(dt)) if dt else float('nan'),
                                nt=int(rts_data['nt']),
                                t_start=float(time[0]) if len(time) else 0.0,
                            ),
                        )
                        self._log("simulation complete (time-series mode)")
                        return result

                    p_at_freq, ranges_out = rts_to_pressure(
                        rts_data, freq, method='fft',
                    )
                    p_field = p_at_freq.reshape(1, -1)

                else:
                    # Multiple depths - run SPARC for each depth
                    self._log(f"Computing for {len(receiver.depths)} depths (SPARC horizontal array mode)...")

                    p_list = []
                    ranges_out = receiver.ranges
                    pressure_all = [] if run_mode == RunMode.TIME_SERIES else None
                    time_grid = None  # captured from first run; SPARC's grid is depth-independent

                    for idx, depth in enumerate(receiver.depths):
                        # Create single-depth receiver
                        single_receiver = Receiver(depths=np.array([depth]), ranges=receiver.ranges)

                        # Write environment file for this depth
                        depth_base = f'{base_name}_d{idx}'
                        env_file = fm.get_path(f'{depth_base}.env')
                        self._write_sparc_env(env_file, env, source, single_receiver, run_mode)

                        # Run SPARC for this depth
                        if self.verbose:
                            self._log(f"  Depth {idx+1}/{len(receiver.depths)}: {depth:.1f}m")
                        self._run_sparc(depth_base, fm.work_dir)
                        rts_file = fm.get_path(f'{depth_base}.rts')
                        if not rts_file.exists():
                            exc = ModelExecutionError(
                                self.model_name, return_code=0, stdout=None,
                                stderr=f"SPARC did not produce {rts_file}",
                            )
                            self._attach_prt_tail(exc, fm.work_dir, depth_base)
                            raise exc

                        rts_data = read_rts_file(rts_file)
                        if time_grid is None:
                            time_grid = {
                                'time': rts_data['time'],
                                'dt': rts_data['dt'],
                                'nt': rts_data['nt'],
                            }

                        if run_mode == RunMode.TIME_SERIES:
                            pressure_all.append(rts_data['p'])  # (nt, nr)
                        else:
                            p_single, ranges_out = rts_to_pressure(
                                rts_data, freq, method='fft',
                            )
                            p_list.append(p_single)

                    if run_mode == RunMode.TIME_SERIES:
                        # Each pressure_all[i] is (nt, nr); stack into
                        # (n_d, nt, nr) then transpose middle/last axes to
                        # match the (n_d, n_r, n_t) contract.
                        pressure_stack = np.moveaxis(
                            np.stack(pressure_all, axis=0), 1, 2,
                        )
                        time = time_grid['time']
                        dt = time_grid['dt']
                        result = Field(
                            data=pressure_stack,
                            coords={
                                'depth': receiver.depths,
                                'range': receiver.ranges,
                                'time': time,
                            },
                            **self._result_kwargs(
                                source,
                                backend='sparc.exe',
                                frequencies=freq,
                                phase_reference='time_domain_native',
                                dt=float(dt),
                                fs=(1.0 / float(dt)) if dt else float('nan'),
                                nt=int(time_grid['nt']),
                                t_start=float(time[0]) if len(time) else 0.0,
                            ),
                        )
                        self._log("simulation complete (time-series mode)")
                        return result

                    p_field = np.vstack(p_list)  # shape: (n_depths, n_ranges)

                result = Field(
                    data=p_field,
                    coords={'depth': receiver.depths, 'range': ranges_out},
                    **self._result_kwargs(
                        source,
                        backend='sparc.exe',
                        frequencies=freq,
                        phase_reference='travelling_wave',
                        conversion_method='fft',
                        output_mode='R',
                        n_depth_runs=len(receiver.depths),
                    ),
                )

            elif self.output_mode == 'D':
                # Vertical array mode: pressure vs depth at fixed ranges
                # SPARC outputs .rts file with depth values
                self._log(f"Computing vertical array at {len(receiver.ranges)} ranges...")

                if len(receiver.ranges) == 1:
                    # Single range - run once
                    env_file = fm.get_path(f'{base_name}.env')
                    self._write_sparc_env(env_file, env, source, receiver, run_mode)

                    self._run_sparc(base_name, fm.work_dir)

                    # Read vertical array output
                    rts_file = fm.get_path(f'{base_name}.rts')
                    if not rts_file.exists():
                        exc = ModelExecutionError(
                            self.model_name, return_code=0, stdout=None,
                            stderr=f"SPARC did not produce {rts_file}",
                        )
                        self._attach_prt_tail(exc, fm.work_dir, base_name)
                        raise exc

                    rts_data = read_rts_file(rts_file)
                    # In vertical mode, 'ranges' in RTS file actually contains depths
                    depths_out = rts_data['ranges']  # These are actually depths
                    p_at_freq, _ = rts_to_pressure(rts_data, freq, method='fft')
                    p_field = p_at_freq.reshape(-1, 1)  # shape: (n_depths, 1)

                else:
                    p_list = []
                    depths_out = receiver.depths

                    for idx, range in enumerate(receiver.ranges):
                        # Create single-range receiver
                        single_receiver = Receiver(depths=receiver.depths, ranges=np.array([range]))

                        range_base = f'{base_name}_r{idx}'
                        env_file = fm.get_path(f'{range_base}.env')
                        self._write_sparc_env(env_file, env, source, single_receiver, run_mode)

                        if self.verbose:
                            self._log(f"  Range {idx+1}/{len(receiver.ranges)}: {range:.1f}m")
                        self._run_sparc(range_base, fm.work_dir)

                        rts_file = fm.get_path(f'{range_base}.rts')
                        if not rts_file.exists():
                            exc = ModelExecutionError(
                                self.model_name, return_code=0, stdout=None,
                                stderr=f"SPARC did not produce {rts_file}",
                            )
                            self._attach_prt_tail(exc, fm.work_dir, range_base)
                            raise exc

                        rts_data = read_rts_file(rts_file)
                        p_single, _ = rts_to_pressure(rts_data, freq, method='fft')
                        p_list.append(p_single)

                    p_field = np.column_stack(p_list)  # shape: (n_depths, n_ranges)

                result = Field(
                    data=p_field,
                    coords={'depth': depths_out, 'range': receiver.ranges},
                    **self._result_kwargs(
                        source,
                        backend='sparc.exe',
                        frequencies=freq,
                        phase_reference='travelling_wave',
                        conversion_method='fft',
                        output_mode='D',
                        n_range_runs=len(receiver.ranges),
                    ),
                )

            elif self.output_mode == 'S':
                # Snapshot mode: wavenumber-domain Green's function
                # SPARC outputs .grn file holding G(itout, irz, ik). To
                # extract steady-state TL at the source frequency we
                # time-FFT then Hankel-transform (see grn_reader docstring).
                self._log("Computing snapshot (wavenumber domain)...")

                # Write environment file
                env_file = fm.get_path(f'{base_name}.env')
                self._write_sparc_env(env_file, env, source, receiver, run_mode)
                self._run_sparc(base_name, fm.work_dir)

                # Read Green's function file
                grn_file = fm.get_path(f'{base_name}.grn')
                if not grn_file.exists():
                    exc = ModelExecutionError(
                        self.model_name, return_code=0, stdout=None,
                        stderr=(
                            f"SPARC snapshot mode did not produce {grn_file}; "
                            f"check {fm.work_dir}/{base_name}.prt for diagnostics."
                        ),
                    )
                    self._attach_prt_tail(exc, fm.work_dir, base_name)
                    raise exc

                self._log("Reading snapshot Green's function and extracting source-freq TL...")
                grn_data = read_grn_file(grn_file)
                if not grn_data['is_sparc']:
                    exc = ModelExecutionError(
                        self.model_name, return_code=0, stdout=None,
                        stderr=(
                            "GRN title does not start with 'SPARC' — snapshot path "
                            "expects a SPARC-produced file."
                        ),
                    )
                    self._attach_prt_tail(exc, fm.work_dir, base_name)
                    raise exc

                # Time-FFT along the snapshot's tout axis, pick the source
                # frequency bin, then Hankel transform — recovers steady-state
                # TL despite SPARC being natively a transient solver.
                result = sparc_snapshot_to_field(
                    grn_data, receiver.ranges, frequency=freq,
                )
                result.model = self.model_name
                result.backend = 'sparc.exe'
                result.source_depths = np.atleast_1d(np.asarray(source.depths, dtype=float))
                result.frequencies = np.atleast_1d(np.asarray(freq, dtype=float))
                result.phase_reference = 'travelling_wave'
                result.metadata['output_mode'] = 'S'
                result.metadata['note'] = 'Snapshot mode: time-FFT then Hankel transform'

            else:
                raise ConfigurationError(
                    f"Invalid output mode '{self.output_mode}'. "
                    f"Valid modes: 'R' (horizontal array), 'D' (vertical array), 'S' (snapshot)"
                )

            # output_mode='S' writes a snapshot .grn; 'R'/'D' write
            # per-depth/per-range .rts files inside loops. Expose
            # whatever exists at the wrapper base_name.
            self._attach_output_paths(
                result, fm.work_dir, base_name,
                primary_files=(
                    ('grn_file', '.grn'),
                    ('rts_file', '.rts'),
                ),
            )

            self._log("Simulation complete")
            return result

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _sparc_rigidify_halfspace(self, env: Environment) -> Environment:
        """Rewrite an env's halfspace bottom to 'rigid' so SPARC's
        ``Vacuum`` / ``Rigid``-only writer accepts it. Emits one
        :class:`UserWarning` per run.

        For ``LayeredBottom`` / ``RangeDependentLayeredBottom`` the
        ``acoustic_type`` lives on the inner ``.halfspace`` (and per
        range profile for RDL), not on the outer container; the walk
        flips it everywhere.
        """
        from uacpy.core.environment import (
            LayeredBottom, RangeDependentLayeredBottom,
        )
        hs = env.halfspace_at_range(0.0)
        kind = (hs.acoustic_type or '').lower()
        if kind not in ('half-space', 'halfspace', 'a'):
            return env
        warnings.warn(
            "SPARC supports only 'vacuum' / 'rigid' bottom boundaries; "
            "auto-converting the env's halfspace to 'rigid'. For "
            "physically meaningful halfspace reflection (fluid or "
            "elastic), use Bellhop / Kraken / Scooter / OASES. To "
            "suppress this warning, set the bottom acoustic_type to "
            "'rigid' (or 'vacuum') before constructing the env.",
            UserWarning, stacklevel=2,
        )
        e = env.copy()
        if isinstance(e.bottom, RangeDependentLayeredBottom):
            for prof in e.bottom.profiles:
                prof.halfspace.acoustic_type = 'rigid'
        elif isinstance(e.bottom, LayeredBottom):
            e.bottom.halfspace.acoustic_type = 'rigid'
        elif hasattr(e.bottom, 'acoustic_type'):
            e.bottom.acoustic_type = 'rigid'
        return e

    def _resolve_rmax_safety_margin(self, run_mode: RunMode) -> float:
        """Pick the effective ``rmax_safety_margin`` for this run.

        SPARC's inverse Hankel transform (``TransformG.f90``) is FFT-based,
        so the r-domain output is periodic with period ``RMax``. In
        ``COHERENT_TL`` the time-FFT averages over the aliased image and
        a ppm-level margin suffices; in ``TIME_SERIES`` the alias is a
        visible non-physical wave at the far range edge unless ``RMax``
        is pushed well past the receivers. User-pinned values win.
        """
        if self.rmax_safety_margin is not None:
            return float(self.rmax_safety_margin)
        return 3.0 if run_mode == RunMode.TIME_SERIES else 1.0001

    def _write_sparc_env(self, filepath, env, source, receiver, run_mode):
        """
        Write SPARC environment file using shared ATEnvWriter

        SPARC extends ReadEnvironmentMod format with:
        - Output mode in TopOpt (5th character: R=horizontal, D=vertical, S=snapshot)
        - Limited bottom types (only vacuum and rigid, no halfspace)
        - Time-domain pulse parameters
        - Time output parameters
        - Integration parameters
        """
        from uacpy.io.oalib_writer import resolve_ssp_topopt
        ssp_code = resolve_ssp_topopt(env, self.interp_ssp)
        surface_type = parse_boundary_type(env.surface.acoustic_type)

        hs = env.halfspace_at_range(0.0)
        bottom_acoustic_type = hs.acoustic_type.lower()
        if bottom_acoustic_type == 'vacuum':
            bottom_type = BoundaryType.VACUUM
        elif bottom_acoustic_type == 'rigid':
            bottom_type = BoundaryType.RIGID
        else:
            raise ConfigurationError(
                f"Invalid bottom boundary type '{hs.acoustic_type}' for SPARC. "
                f"Only 'vacuum' and 'rigid' are supported."
            )

        with open(filepath, 'w') as f:
            # SPARC TopOpt: [SSP][BC][AttenUnit(2 chars)][OutputMode]
            f.write(f"'{env.name}'\n")
            f.write(f"{source.frequencies[0]:.6f}\n")
            # NMedia = water column + one medium per sediment layer.
            n_media = 1
            if env.has_layered_bottom():
                n_media += len(env.bottom.layers)
            f.write(f"{n_media}\n")

            surface_code = surface_type.to_acoustics_toolbox_code()
            atten_code = AttenuationUnits.DB_PER_WAVELENGTH.to_char()
            vol_atten_code = (
                env.absorption.topopt_code() if env.absorption is not None else ' '
            )
            topopt = f"{ssp_code}{surface_code}{atten_code}{vol_atten_code}{self.output_mode}".ljust(6)
            f.write(f"'{topopt}'\n")

            write_absorption_block(f, env)

            # Write SSP section
            write_ssp_section(
                f, env, env.depth,
                n_mesh=self.n_mesh,
                roughness=self.roughness
            )

            # Write sediment layers if layered bottom
            write_layer_sections(f, env, env.depth)

            # Write bottom section (SPARC only supports V and R)
            bottom_code = bottom_type.to_acoustics_toolbox_code()
            sigma = getattr(env.halfspace_at_range(0.0), 'roughness', 0.0)
            f.write(f"'{bottom_code}' {sigma:.1f}\n")
            # Note: No halfspace parameters since SPARC doesn't support them

            # SPARC-SPECIFIC SECTIONS

            # RMax is the period of SPARC's inverse Hankel FFT — too tight
            # leaks the source's r=RMax image into the receiver area.
            # Margin policy in ``_resolve_rmax_safety_margin``.
            margin = self._resolve_rmax_safety_margin(run_mode)
            rmax_m = float(receiver.ranges.max()) * margin
            write_phase_speed_and_rmax(
                f, env,
                rmax_m=rmax_m,
                c_low=self.c_low, c_high=self.c_high,
                rmax_format="{:.6f}",
            )

            # Source and receiver depths. Use the shared ATEnvWriter so
            # non-uniform arrays are written verbatim rather than collapsed
            # to "min max /" (which the Fortran reader expands to a
            # uniformly-spaced vector).
            write_source_depths(f, source)
            if len(receiver.depths) == 1:
                # Single depth — SPARC interpolates a depth vector from
                # (first, last); repeat the value so it stays constant.
                f.write("1\n")
                f.write(f"{receiver.depths[0]:.6f} {receiver.depths[0]:.6f} /\n")
            else:
                write_receiver_depths(f, receiver)

            # Time-domain pulse parameters (SPARC-specific, come BEFORE ranges!)
            f.write(f"'{self.pulse_type}'\n")

            # Pulse frequency band [f_min, f_max] (Hz).
            #
            # SPARC's work scales with Nk ≈ 1000 * Rmax_km * (k_max-k_min) /
            # (2π). A ±2% band makes the pulse near-CW and the FFT at the
            # analysis frequency picks up almost nothing. A 10× band
            # (100-10000 Hz for a 1 kHz source) is tractable for small Rmax
            # but blows Nk up to many-thousands for 10-20 km ranges, which
            # routinely times out.
            #
            # One-octave (freq/2 to freq*2) is the sweet spot: enough
            # bandwidth that the pulse retains structure, yet bounded Nk.
            # Callers can override via kwargs for special analyses.
            freq = source.frequencies[0]
            f_min = self.f_min if self.f_min is not None else max(freq / 2.0, 0.1)
            f_max = self.f_max if self.f_max is not None else freq * 2.0
            f.write(f"{f_min:.6f} {f_max:.6f}\n")

            # Receiver ranges (come AFTER pulse info in SPARC).
            #
            # SubTab (Scooter/subtabulate.f90) expands "rmin rmax /" into a
            # uniformly-spaced vector, which silently discards a caller's
            # non-uniform ranges. Always emit the full list — SubTab accepts
            # an N-entry list verbatim when the trailing '/' comes after
            # more than two values.
            ranges_km = receiver.ranges / 1000.0
            f.write(f"{len(ranges_km)}\n")
            ranges_str = ' '.join([f"{r:.6f}" for r in ranges_km])
            f.write(f"{ranges_str} /\n")

            # Time output parameters
            f.write(f"{self.n_t_out}\n")

            # Time output range (s)
            c_water = self.sound_speed if self.sound_speed is not None else DEFAULT_SOUND_SPEED
            travel_time = rmax_m / c_water
            t_max = self.t_max if self.t_max is not None else travel_time * 2.5
            f.write(f"0.0 {t_max:.6f} /\n")

            # Integration parameters: TSTART, TMULT, ALPHA, BETA, V
            f.write(f"{self.t_start:.6f} {self.t_mult:.6f} 0.0 0.0 0.0\n")

    def _run_sparc(self, base_name: str, work_dir: Path):
        """
        Run SPARC as a subprocess (180 s timeout by default).

        Delegates to ``PropagationModel._run_subprocess`` (which raises the
        child stack limit — required because SPARC statically allocates
        ~80 MB of COMPLEX arrays and would otherwise segfault). On failure,
        appends the ``.prt`` tail to the raised ``ModelExecutionError`` for
        easier diagnosis. Override via the ``timeout`` constructor kwarg.
        """
        try:
            result = self._run_subprocess(
                [str(self.executable), base_name],
                cwd=work_dir,
                timeout=self.timeout,
            )
        except ModelExecutionError as exc:
            self._attach_prt_tail(exc, work_dir, base_name)
            raise

        if self.verbose and result.stdout:
            self._log(f"SPARC output:\n{result.stdout}", level='debug')
