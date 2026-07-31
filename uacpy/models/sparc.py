"""
SPARC - Seismo-Acoustic Propagation in Realistic oCeans

SPARC is a time-domain FFP (Fast Field Program) model using the same wavenumber
integration approach as Scooter. The underlying ``sparc.f90`` reads shear and is
elastic-capable, but the uacpy writer currently restricts the bottom boundary to
vacuum / rigid (any halfspace is force-rigidified with a warning), so as wired
uacpy's SPARC is a rigid/vacuum-bounded fluid model.
"""

import warnings
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np

from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.results import Result, Field
from uacpy.core.constants import (
    BoundaryType,
    parse_boundary_type,
    DEFAULT_SOUND_SPEED,
)
from uacpy.core.exceptions import (
    ConfigurationError, ExecutableNotFoundError, ModelExecutionError,
    UnsupportedFeatureError,
)
from uacpy.io.grn_reader import read_grn_file, sparc_snapshot_to_time_field
from uacpy.models.base import PropagationModel, RunMode, ModelSpec
from uacpy.io.oalib_reader import read_rts_file, rts_to_pressure
from uacpy.io.oalib_writer import write_sparc_env_file


# CW transmission-loss extraction deconvolves the source pulse from the output
# time series, so the output sampling must resolve the pulse band: its Nyquist
# (fs/2) has to sit above ``f_max`` or the requested frequency aliases (and a
# pulse shorter than one output sample vanishes entirely → S(f0)=0). These size
# the output grid (``n_t_out``) for ``RunMode.COHERENT_TL`` when the user's value
# would undersample. ``OVERSAMPLE`` is the factor above the Nyquist minimum;
# ``MAX_N_T_OUT`` caps the auto-grow so a too-high CW request fails with a clear
# message instead of timing out.
_SPARC_CW_PULSE_OVERSAMPLE = 3.0
_SPARC_MAX_N_T_OUT = 16384


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
    ConfigurationError
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


# Source geometry -> fieldsco.m Opt(1:1), consumed by the Hankel transform.
_SOURCE_TYPE_CODE = {'point': 'R', 'line': 'X', 'scaled': 'S'}


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
        Number of output time samples. Default ``512``.
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
    Per-model: ``'ssp': 'mean'``, ``'bottom_range': 'median'`` (the layer
    stack is kept since SPARC consumes layered seabed columns natively).

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

    # Declarative metadata (see PropagationModel / ModelSpec). SPARC:
    # range-independent time-marched FFP. Honours a multi-layer fluid
    # bottom; elastic_media is False because run() auto-rigidifies a
    # halfspace bottom, so we collapse to fluid up front (uniform warning
    # rather than a silent rigidify). Single solve over the spectrum →
    # mean SSP / median bottom column represent the path.
    spec = ModelSpec(
        modes=(RunMode.TIME_SERIES,),
        supports={'layered_bottom'},
        collapse={'ssp': 'mean', 'bottom_range': 'median'},
    )
    source = 'acoustics_toolbox'

    def __init__(
        self,
        executable: Optional[Path] = None,
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        n_mesh: int = 0,
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
        cleanup: Optional[bool] = None,
        collapse: Optional[Dict[str, str]] = None,
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
        interp_ssp : str, optional
            SSP connection scheme. ``None`` (default) auto-picks
            ``'quad'`` for a range-dependent ``env.ssp`` and
            ``'linear'`` otherwise. Explicit values: ``'linear'``,
            ``'pchip'``, ``'cubic'``, ``'quad'``, ``'n2linear'``,
            ``'analytic'``. ``env.ssp.shape='isovelocity'`` always
            forces ``'C'`` regardless.
        output_mode : str, optional
            'R' (horizontal array), 'D' (vertical array), 'S' (snapshot). Default: 'R'.

            ``'S'`` mode time-FFTs the snapshot's tout axis and picks the
            source-frequency bin (``uacpy.io.grn_reader.sparc_snapshot_to_field``).
            All three modes recover absolute TL re 1 m by deconvolving the
            known source spectrum ``S(omega0)`` from the time-domain field
            (convolution theorem, Jensen COA Eq. 8.1), then divide by √(4π) to
            convert SPARC's native RTS convention to Scooter/Kraken's
            bare-Hankel TL pressure, so the three modes report the same field
            at the same level.

            **CW transmission loss from SPARC is not quantitative.** SPARC
            marches a pulse and the CW field is extracted from it, and three
            things fight that extraction: ``sparc.f90:313`` sets ``Atten = 0``
            so a lossless waveguide's real-axis poles are integrated with no
            stabilising attenuation (Scooter uses ``Atten = Deltak`` precisely
            to make its k-sum converge, ``scooter.f90:129``); ``Nk`` is spread
            over the whole *pulse* band, so only a fraction of the wavenumber
            samples land in the analysis frequency's window; and the default
            ``pulse_type='PN+B'`` applies a per-wavenumber band-pass that
            ``rts_to_pressure`` deconvolves with a single scalar ``S(omega0)``,
            which cannot represent a k-dependent filter. Measured on a guide
            supporting exactly **one** propagating mode — where the exact TL is
            smooth and monotone — SPARC returns 2.4 dB median error with 13 dB
            excursions, against 0.07 dB for Scooter on the same case, and it
            does not converge under any of ``rmax_safety_margin`` / ``t_max`` /
            ``n_t_out`` / ``t_mult`` / ``n_mesh`` / ``t_start``. Use
            :class:`~uacpy.Scooter` or :class:`~uacpy.Kraken` for CW TL; SPARC's
            value is its time-domain output.
        pulse_type : str, optional
            Pulse type string. Default: 'PN+B'.
        n_t_out : int, optional
            Number of time samples. Default: 512.
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
        f_min, f_max : float, optional
            Pulse frequency band (Hz). ``None`` (default) resolves at
            ``run()`` time to one octave around the source frequency
            (``max(f/2, 0.1)`` .. ``2f``) — SPARC's work scales with
            the band's wavenumber span, so a much wider band slows it
            sharply.
        sound_speed : float, optional
            Water sound speed (m/s) used for the travel-time window
            when ``t_max`` is auto. ``None`` (default) →
            :data:`DEFAULT_SOUND_SPEED`.
        timeout : float, optional
            Subprocess timeout (s) for each SPARC run. Default: 180.0.
        """
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            timeout=timeout, cleanup=cleanup, collapse=collapse,
        )

        self.c_low = c_low
        self.c_high = c_high
        if c_low is not None and c_high is not None and c_low >= c_high:
            raise ConfigurationError(
                f"SPARC spectral phase-velocity band requires "
                f"c_low < c_high; got c_low={c_low} m/s, c_high={c_high} m/s."
            )
        self.n_mesh = n_mesh
        self.interp_ssp = interp_ssp
        if output_mode not in ('R', 'D', 'S'):
            raise ConfigurationError(
                f"Invalid output_mode {output_mode!r}. Valid modes (sparc.f90 "
                f"TopOpt(5:5)): 'R' RTS horizontal array, 'D' RTS vertical "
                f"array, 'S' snapshot."
            )
        self.output_mode = output_mode
        # Only the snapshot runs a Hankel transform, so it is the only mode
        # that can honour a source geometry; 'R'/'D' are range-/depth-native.
        if output_mode == 'S':
            self._supported_source_types = frozenset(_SOURCE_TYPE_CODE)
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

        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        #
        # Keep the user's ``executable`` arg verbatim (``None`` when
        # auto-detected) so ``model.copy()`` re-resolves the binary instead of
        # re-pinning the already-resolved absolute path. The resolved path
        # lives in ``self._exe``.
        self.executable = Path(executable) if executable is not None else None
        if self.executable is None:
            self._exe = self._find_executable_in_paths(
                'sparc.exe', bin_subdirs='oalib',
                dev_subdir='Acoustics-Toolbox/Scooter',
            )
        else:
            self._exe = self.executable

        if not self._exe.exists():
            raise ExecutableNotFoundError('SPARC', str(self._exe))

    def run(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional[RunMode] = None,
        *,
        frequencies=None,
        source_waveform=None,
        sample_rate=None,
        output_duration=None,
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
        if run_mode == RunMode.COHERENT_TL:
            # Named explicitly so the refusal points at a model that does
            # compute CW TL; the generic _resolve_run_mode error can only
            # list SPARC's own modes.
            raise UnsupportedFeatureError(
                model_name='SPARC',
                feature=(
                    'RunMode.COHERENT_TL — SPARC marches a pulse, and the '
                    'CW field extracted from it is not quantitative '
                    '(no contour offset on the wavenumber sum, a grid sized '
                    'for the whole pulse band, and a per-wavenumber band-pass '
                    'that a scalar source-spectrum deconvolution cannot undo)'
                ),
                alternatives=[
                    'Scooter for wavenumber-integration CW transmission loss',
                    'Kraken for normal-mode CW transmission loss',
                    'SPARC with run_mode=TIME_SERIES for its native p(t)',
                ],
            )
        run_mode = self._resolve_run_mode(run_mode)

        # The native transient p(t) is assembled only on the 'R' (horizontal,
        # range-native) path; the 'D'/'S' branches return a frequency-domain
        # field. Reject TIME_SERIES for those rather than silently returning
        # the wrong result kind.
        # SPARC builds p(t) from its native pulse_type on its own time grid
        # at source.frequencies — none of the contract extras can influence
        # the run, in any run mode.
        self._warn_ignored_run_kwargs(
            run_mode,
            reason=(
                "SPARC builds p(t) from its native pulse_type over its own "
                "time grid at source.frequencies; pass SPARC(pulse_type=...) "
                "to shape the pulse"
            ),
            frequencies=frequencies,
            source_waveform=source_waveform,
            sample_rate=sample_rate,
            output_duration=output_duration,
        )
        env = self._project_environment(env)
        env = self._sparc_rigidify_halfspace(env)
        receiver = self._clip_receiver_depths(
            receiver, self._total_media_depth(env)
        )

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

            if self.output_mode == 'D':
                return self._run_vertical(
                    fm, env, source, receiver, run_mode, base_name, freq)
            if self.output_mode == 'S':
                return self._run_snapshot(
                    fm, env, source, receiver, run_mode, base_name, freq)
            return self._run_range_native(
                fm, env, source, receiver, run_mode, base_name, freq)

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _finalize_sparc_result(self, result, fm, base_name):
        """Attach output paths + finishing log; the shared R/D/S tail."""
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

    def _run_range_native(self, fm, env, source, receiver, run_mode,
                          base_name, freq):
        """output_mode='R': horizontal (range-native) array — one SPARC run per
        receiver depth, assembling the native p(t) for TIME_SERIES."""
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
                        backend='sparc',
                        frequencies=freq,
                        phase_reference='time_domain_native',
                        dt=float(dt),
                        fs=(1.0 / float(dt)) if dt else float('nan'),
                        nt=int(rts_data['nt']),
                        t_start=float(time[0]) if len(time) else 0.0,
                    ),
                )
                # Same tail as every other exit: with cleanup=False the
                # caller is promised the .rts/.grn paths in metadata.
                return self._finalize_sparc_result(result, fm, base_name)

            p_at_freq, ranges_out = rts_to_pressure(
                rts_data, freq, method='fft', pulse_type=self.pulse_type,
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
                        rts_data, freq, method='fft', pulse_type=self.pulse_type,
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
                # The range axis is SPARC's actual output grid (identical
                # across depths), matching the single-depth path. Field
                # validates this length against the data shape.
                result = Field(
                    data=pressure_stack,
                    coords={
                        'depth': receiver.depths,
                        'range': rts_data['ranges'],
                        'time': time,
                    },
                    **self._result_kwargs(
                        source,
                        backend='sparc',
                        frequencies=freq,
                        phase_reference='time_domain_native',
                        dt=float(dt),
                        fs=(1.0 / float(dt)) if dt else float('nan'),
                        nt=int(time_grid['nt']),
                        t_start=float(time[0]) if len(time) else 0.0,
                    ),
                )
                # Same tail as every other exit: with cleanup=False the
                # caller is promised the .rts/.grn paths in metadata.
                return self._finalize_sparc_result(result, fm, base_name)

            p_field = np.vstack(p_list)  # shape: (n_depths, n_ranges)

        # SPARC's native 'R' (horizontal) RTS field is written unscaled
        # (sparc.f90 KERNEL), so relative to Scooter/Kraken's TL pressure it
        # carries an extra √(4π) — the asymptotic-Hankel spreading √π times a
        # factor-2 wavenumber-integral convention. Divide by it so the CW field
        # is in the bare-Hankel convention (|g(1 m)|≈1), matching Kraken to
        # ~1-4 dB on a Pekeris benchmark. ('D'/'S' carry their own scale and
        # are normalised in their own paths.)
        p_field = p_field / np.sqrt(4.0 * np.pi)

        result = Field(
            data=p_field,
            coords={'depth': receiver.depths, 'range': ranges_out},
            **self._result_kwargs(
                source,
                backend='sparc',
                frequencies=freq,
                phase_reference='travelling_wave',
                conversion_method='fft',
                output_mode='R',
                n_depth_runs=len(receiver.depths),
            ),
        )
        return self._finalize_sparc_result(result, fm, base_name)

    def _run_vertical(self, fm, env, source, receiver, run_mode,
                      base_name, freq):
        """``output_mode='D'``: the vertical-array received time series.

        ``sparc.f90:593-606`` accumulates ``RTSrz(ir, Itout)`` — a time series
        at each *depth* for a fixed range — the same kind of output as ``'R'``
        sampled down a vertical array instead of along a horizontal one. SPARC
        runs one range at a time, so the wrapper loops.

        The ``.rts`` written in this mode stores the depth axis in the slot the
        horizontal mode uses for ranges, which is why ``rts_data['ranges']``
        is read as depths here.
        """
        self._log(f"Computing vertical array at {len(receiver.ranges)} "
                  f"range(s)...")

        traces, depths_out = [], None
        time = dt = nt = None

        for idx, rng in enumerate(np.atleast_1d(receiver.ranges)):
            single = Receiver(depths=receiver.depths, ranges=np.array([rng]))
            run_base = (base_name if len(np.atleast_1d(receiver.ranges)) == 1
                        else f'{base_name}_r{idx}')
            self._write_sparc_env(fm.get_path(f'{run_base}.env'), env, source,
                                  single, run_mode)
            self._log(f"  range {idx + 1}/{len(np.atleast_1d(receiver.ranges))}: "
                      f"{float(rng):.1f} m")
            self._run_sparc(run_base, fm.work_dir)

            rts_file = fm.get_path(f'{run_base}.rts')
            if not rts_file.exists():
                exc = ModelExecutionError(
                    self.model_name, return_code=0, stdout=None,
                    stderr=f"SPARC did not produce {rts_file}",
                )
                self._attach_prt_tail(exc, fm.work_dir, run_base)
                raise exc

            rts_data = read_rts_file(rts_file)
            if depths_out is None:
                depths_out = np.asarray(rts_data['ranges'], dtype=float)
                time = np.asarray(rts_data['time'], dtype=float)
                dt, nt = rts_data['dt'], rts_data['nt']
            # rts_data['p'] is (nt, n_depth) here; want (n_depth, nt).
            # sparc.f90:292 scales the 'D' branch by 1/sqrt(pi*Rr) where 'R'
            # carries 1/sqrt(r), so the vertical array comes out sqrt(pi)
            # quieter for the identical field (measured ratio 0.5642 =
            # 1/sqrt(pi) at every depth). Put both modes on one convention.
            traces.append(np.sqrt(np.pi) * np.asarray(rts_data['p']).T)

        # (n_depth, n_range, n_time) — the shared Field contract.
        data = np.stack(traces, axis=1)
        result = Field(
            data=data,
            coords={
                'depth': depths_out,
                'range': np.atleast_1d(receiver.ranges),
                'time': time,
            },
            **self._result_kwargs(
                source,
                backend='sparc',
                frequencies=freq,
                phase_reference='time_domain_native',
                dt=float(dt),
                fs=(1.0 / float(dt)) if dt else float('nan'),
                nt=int(nt),
                t_start=float(time[0]) if len(time) else 0.0,
            ),
        )
        return self._finalize_sparc_result(result, fm, base_name)

    def _run_snapshot(self, fm, env, source, receiver, run_mode,
                      base_name, freq):
        """``output_mode='S'``: the snapshot — the whole field at each output
        time.

        SPARC writes the wavenumber-domain field ``Green(Itout, irz, ik)`` to a
        ``.grn`` (``sparc.f90:580-591``); ``doc/sparc.htm`` says FIELDS must be
        run afterwards to turn it into a pressure field. uacpy does that step
        in-tree via one inverse Hankel transform per output time, giving
        ``p(z, r, t)`` on the requested receiver grid in a single SPARC run —
        unlike ``'R'``/``'D'``, which loop the binary per depth / per range.
        """
        self._log("Computing snapshot (whole field per output time)...")
        env_file = fm.get_path(f'{base_name}.env')
        self._write_sparc_env(env_file, env, source, receiver, run_mode)
        self._run_sparc(base_name, fm.work_dir)

        grn_file = fm.get_path(f'{base_name}.grn')
        if not grn_file.exists():
            exc = ModelExecutionError(
                self.model_name, return_code=0, stdout=None,
                stderr=f"SPARC did not produce {grn_file}",
            )
            self._attach_prt_tail(exc, fm.work_dir, base_name)
            raise exc

        result = sparc_snapshot_to_time_field(
            read_grn_file(grn_file),
            np.atleast_1d(receiver.ranges),
            frequency=freq,
            source_type=_SOURCE_TYPE_CODE[source.source_type],
        )
        self._stamp_result(result, source, backend='sparc',
                           frequencies=freq)
        return self._finalize_sparc_result(result, fm, base_name)

    def _max_receiver_depth(self, env: Environment) -> float:
        return self._total_media_depth(env)

    def _sparc_rigidify_halfspace(self, env: Environment) -> Environment:
        """Rewrite an env's halfspace bottom to 'rigid' so SPARC's
        ``Vacuum`` / ``Rigid``-only writer accepts it. Emits one
        :class:`UserWarning` per run.

        For a ``Bottom`` the ``acoustic_type`` lives on each column's
        ``.halfspace`` (per range when range-dependent), not on the outer
        container; the walk flips it everywhere.
        """
        hs = env.bottom.halfspace_at(range=0.0)
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
        for col in e.bottom.columns:
            col.halfspace.acoustic_type = 'rigid'
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

        hs = env.bottom.halfspace_at(range=0.0)
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

        # RMax is the period of SPARC's inverse Hankel FFT — too tight leaks
        # the source's r=RMax image into the receiver area. Margin policy in
        # ``_resolve_rmax_safety_margin``.
        margin = self._resolve_rmax_safety_margin(run_mode)
        rmax_m = float(receiver.ranges.max()) * margin

        # Pulse frequency band [f_min, f_max] (Hz). SPARC's work scales with
        # Nk ≈ 1000 * Rmax_km * (k_max-k_min)/(2π); a near-CW band makes the
        # FFT at the analysis frequency pick up almost nothing, while a 10×
        # band blows Nk up and times out. One octave (freq/2 .. freq*2) is the
        # sweet spot. Callers override via constructor kwargs.
        freq = source.frequencies[0]
        f_min = self.f_min if self.f_min is not None else max(freq / 2.0, 0.1)
        f_max = self.f_max if self.f_max is not None else freq * 2.0

        # Time output window (s).
        c_water = self.sound_speed if self.sound_speed is not None else DEFAULT_SOUND_SPEED
        travel_time = rmax_m / c_water
        t_max = self.t_max if self.t_max is not None else travel_time * 2.5

        n_t_out = self._resolve_n_t_out(run_mode, f_max, t_max)

        write_sparc_env_file(
            filepath, env, source, receiver,
            ssp_code=ssp_code,
            surface_type=surface_type,
            bottom_type=bottom_type,
            output_mode=self.output_mode,
            n_mesh=self.n_mesh,
            rmax_m=rmax_m,
            c_low=self.c_low, c_high=self.c_high,
            pulse_type=self.pulse_type,
            f_min=f_min, f_max=f_max,
            n_t_out=n_t_out,
            t_max=t_max,
            t_start=self.t_start, t_mult=self.t_mult,
        )

    def _resolve_n_t_out(self, run_mode, f_max, t_max):
        """Output time-sample count, grown if needed to resolve the pulse band.

        ``RunMode.COHERENT_TL`` recovers the CW field by deconvolving the source
        pulse from the output time series (all three ``output_mode``s), so the
        output sampling ``fs = n_t_out / (t_max - t_start)`` must keep its
        Nyquist above ``f_max`` — otherwise the requested frequency aliases, or
        (pulse shorter than one sample) vanishes and the deconvolution divides
        by zero. The fixed default ``n_t_out`` is sized for a multi-second
        propagation window, which under-samples CW frequencies above a few tens
        of Hz; grow it here. ``TIME_SERIES`` keeps the user's value verbatim (the
        native ``p(t)`` sampling is theirs to choose).
        """
        if run_mode != RunMode.COHERENT_TL:
            # TIME_SERIES keeps the caller's sampling by contract, but a grid
            # whose Nyquist sits below the source band aliases silently: the
            # returned p(t) looks perfectly plausible at the wrong frequency.
            # Say so, and name the n_t_out that fixes it.
            window = t_max - self.t_start
            if window > 0 and f_max > 0:
                fs = self.n_t_out / window
                if f_max > 0.5 * fs:
                    n_needed = int(np.ceil(
                        window * _SPARC_CW_PULSE_OVERSAMPLE * 2.0 * f_max))
                    warnings.warn(
                        f"SPARC TIME_SERIES: the output grid samples at "
                        f"{fs:.1f} Hz (Nyquist {fs / 2:.1f} Hz) over a "
                        f"{window:.2f} s window, below the {f_max:.0f} Hz "
                        f"source band — p(t) will alias. Set n_t_out>="
                        f"{n_needed} (or shorten the window via t_start / "
                        f"receiver range).",
                        UserWarning, stacklevel=3,
                    )
            return self.n_t_out
        window = t_max - self.t_start
        fs_required = _SPARC_CW_PULSE_OVERSAMPLE * 2.0 * f_max
        n_required = int(np.ceil(window * fs_required))
        if n_required <= self.n_t_out:
            return self.n_t_out
        if n_required > _SPARC_MAX_N_T_OUT:
            # Fail fast: clamping to the cap could not meet the sampling target,
            # so the run would either alias the CW frequency or proceed
            # under-resolved at a very large n_t_out whose wavenumber march
            # exceeds the subprocess timeout. A 0-second actionable error beats
            # a multi-minute wait that ends in a generic timeout.
            raise ConfigurationError(
                f"SPARC COHERENT_TL: resolving the {f_max:.0f} Hz pulse band "
                f"over a {window:.1f} s output window needs n_t_out≈"
                f"{n_required}, above the {_SPARC_MAX_N_T_OUT} cap — the run "
                f"would alias the CW frequency or be impractically slow.",
                remediation="Shorten the receiver range, narrow the pulse band "
                "via f_max=, or use Kraken/Scooter for this frequency.",
            )
        n_t_out = n_required
        self._log(
            f"raising n_t_out {self.n_t_out}→{n_t_out} so the output Nyquist "
            f"({0.5 * n_t_out / window:.0f} Hz) clears the pulse band f_max="
            f"{f_max:.0f} Hz (required for CW deconvolution)."
        )
        return n_t_out

    def _run_sparc(self, base_name: str, work_dir: Path):
        """
        Run SPARC as a subprocess (180 s timeout by default).

        Delegates to ``PropagationModel._run_subprocess`` (which raises the
        child stack limit — required because SPARC statically allocates
        ~80 MB of COMPLEX arrays and would otherwise segfault). On failure,
        appends the ``.prt`` tail to the raised ``ModelExecutionError`` for
        easier diagnosis. Override via the ``timeout`` constructor kwarg.
        """
        self._run_and_attach_prt(
            [str(self._exe), base_name], work_dir, base_name,
            timeout=self.timeout)
