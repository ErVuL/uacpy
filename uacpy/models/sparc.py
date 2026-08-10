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
from uacpy.io.grn_reader import (
    read_grn_file, sparc_snapshot_to_time_field, _zero_range_mask,
)
from uacpy.models.base import (
    PropagationModel, RunMode, ModelSpec, USER_FRAME_SKIP,
)
from uacpy.io.oalib_reader import read_rts_file
from uacpy.io.oalib_writer import write_sparc_env_file, reject_unsupported_ssp_interp


# Factor above the Nyquist minimum used when naming the ``n_t_out`` that would
# resolve the pulse band in the aliasing warning.
_SPARC_PULSE_OVERSAMPLE = 3.0


def _mask_zero_range_traces(data: np.ndarray, ranges: np.ndarray,
                            output_mode: str) -> np.ndarray:
    """Return ``data`` with the receivers on the source axis set to no-data.

    ``sparc.f90:622`` weights the ``'R'`` branch by ``SQRT( rkT / Pos%Rr )``
    and ``:292`` scales ``'D'`` by ``1 / SQRT( pi * Pos%Rr( 1 ) )``, both
    singular at ``r = 0``. Measured on a 100 m Pekeris guide, ``'D'`` comes
    back ``+Inf`` for the whole trace and ``'R'`` mixes ``NaN`` with exact
    zeros — an amplitude a caller would read as a real, quiet sample. The
    snapshot mode already returns no-data there (its Hankel transform runs
    in-tree, ``grn_reader._hankel_transform``), so masking puts all three
    output modes on one answer.
    """
    ranges = np.atleast_1d(np.asarray(ranges, dtype=float))
    zero = _zero_range_mask(ranges)
    n_zero = int(np.count_nonzero(zero))
    if not n_zero:
        return data

    masked = np.asarray(data, dtype=float).copy()
    masked[:, zero, :] = np.nan
    warnings.warn(
        f"{n_zero} receiver range(s) at r = 0: SPARC's output_mode="
        f"{output_mode!r} carries a 1/sqrt(r) cylindrical-spreading factor "
        f"that is singular there, so those cells are returned as NaN "
        f"(no data). Move the receiver off the source axis (e.g. r = 1 m) to "
        f"get a field value.",
        UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )
    return masked


# SPARC pulse_type alphabets (per Scooter/sparc.f90:126-148 GetPar SELECT CASE).
# Pos 1: pulse shape — the 10 letters accepted by sparc.f90's parser.
# ``T`` and ``C`` exist in tslib/cans.f90 (the pulse evaluator) but
# sparc.f90's GetPar rejects them with "Unknown source type" before
# cans.f90 is reached.
_PULSE_TYPE_POS1 = set('PRASHNGFBM')
# Pos 2: post-processing applied to the pulse samples
# (tslib/sourceMod.f90:69-70).
#   'H' = pre-envelope (|analytic signal|), 'Q' = Hilbert transform.
#   Any other character (including ' ' or 'N') means "no transform".
_PULSE_TYPE_POS2 = {' ', 'N', 'H', 'Q'}
# Pos 3: sign flag. '-' inverts the pulse (tslib/sourceMod.f90:178); any
# other character keeps it.
_PULSE_TYPE_POS3 = {' ', '+', '-'}
# Pos 4: per-wavenumber band-pass of the source pulse, set in sparc.f90's
# MARCH (Scooter/sparc.f90:391-401, mirrored in Matlab/Sparc/march.m:93-102).
#   'L' cuts below k·cLow/2π, 'H' cuts above k·cHigh/2π, 'B' does both.
#   'N' skips the filter entirely (tslib/sourceMod.f90:68); any other
#   character (including ' ') leaves the full band [0, 10·fMax] in place.
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
        If any character falls outside the alphabets read from
        ``Scooter/sparc.f90:126-148`` (shape) and ``:391-401`` (filter), and
        from ``tslib/sourceMod.f90:68-70,178`` (post-process, sign, filter).
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

    Time-domain FFP model: it marches a source pulse and its product is the
    transient pressure ``p(z, r, t)``. ``RunMode.TIME_SERIES`` is the only
    supported run mode; CW transmission loss is refused (see
    :meth:`run`) — use :class:`~uacpy.Scooter` or :class:`~uacpy.Kraken`
    for that.

    Limitations:
    - Only supports Vacuum or Rigid boundary conditions (no halfspace)
    - ``output_mode='R'`` runs one SPARC simulation per receiver depth and
      ``'D'`` one per receiver range (both looped in the wrapper) — see
      ``max_depths`` for the safety cap on the looped axis. ``'S'`` runs the
      binary once for the whole grid.
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

        ``cans.f90:31-35``'s ``'R'`` (Ricker) is defined on ``U = ω·T - 5``
        and therefore peaks at ``T = 5/(2π·F) ≈ 0.796/F``, not at ``T = 0``.
        (``cans.f90``'s own per-case "peak at …, support […]" notes describe
        the *spectrum*, not the waveform — its ``'C'`` sinc entry reads
        "uniform spectrum from [0, F]" and its Gaussian says "peak at 0"
        while peaking in time at ``T0 = 0.5/F``.) When aligning expected vs
        measured arrivals downstream, treat the Ricker peak as offset by
        ``+5/(2π·F)`` from the source pulse origin.
    n_t_out : int, optional
        Number of output time samples. Default ``512``.

        The deck writes the output times as the pair ``0.0 t_max /``, which
        ``ReadVector`` leaves for ``SubTab`` to expand — and ``SubTab`` acts
        only for ``Nx >= 3`` (``misc/subtabulate.f90:24,40``). ``n_t_out >= 3``
        therefore gets ``n_t_out`` uniform samples on ``[0, t_max]``;
        ``n_t_out == 2`` happens to land on the two endpoints unexpanded; and
        **``n_t_out == 1`` reads only the ``0.0``** — the trailing ``t_max`` is
        never consumed, so the run returns the single time ``t = 0`` and the
        requested window is silently dropped.
    t_max : float, optional
        Maximum simulated time (s). ``None`` ⇒ ``2.5 ×`` travel time.
    t_start : float, optional
        Time the march begins (s), ``time = t_start + (i-1)·Δt``
        (``sparc.f90:409``). ``cans.f90`` returns zero pulse amplitude for
        ``T <= 0``, so a negative value starts the field from rest before the
        source turns on. Default ``-0.1``.
    t_mult : float, optional
        Courant safety factor on the marching time step: ``sparc.f90:265``
        sets ``Δt = t_mult / sqrt(1/crossT² + (0.5·cMax·k)²)``, so ``1.0`` is
        exactly the stability limit and the default sits just inside it.
        Default ``0.999``.
    max_depths : int, optional
        Cap on the axis the wrapper loops the binary over — receiver
        depths for ``output_mode='R'``, receiver ranges for ``'D'``.
        ``'S'`` runs once and is uncapped. Default ``20``.
    rmax_safety_margin : float, optional
        Multiplier applied to ``receiver.ranges.max()`` to set SPARC's
        ``RMax``. ``None`` (default) ⇒ ``3.0``. ``RMax`` fixes the
        wavenumber sampling — ``sparc.f90:116`` takes
        ``Nk = INT(1000·RMax_km·(kMax−kMin)/2π)``, i.e.
        ``Δk ≈ 2π/RMax_m`` — and the range synthesis is a direct ``Δk``
        sum over that grid, so the output is periodic in range with
        period ``2π/Δk = RMax``. A tight margin therefore puts the
        source's non-physical replica at ``r = receiver.ranges.max()``.
        Increasing this knob raises ``Nk`` (and SPARC runtime) in
        proportion. ``sparc.f90:153-155`` rejects the run outright when
        the largest receiver range exceeds ``RMax``.
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

    All three ``output_mode``s share one output grid: ``n_t_out``
    samples over ``[t_start, t_max]``. A grid whose Nyquist sits below
    the pulse band ``f_max`` aliases and emits a ``UserWarning`` naming
    the ``n_t_out`` that would resolve it.

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
        # Power of two: p(t) comes back on this many samples and any
        # downstream FFT over the time axis then stays on the radix-2 path.
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
            SSP connection scheme written into ``TopOpt(1)``. ``None``
            (default) resolves to ``'linear'`` (C-linear): SPARC declares
            no range-dependent-SSP capability, so ``run()`` collapses
            ``env.ssp`` to 1-D before the deck is written and the
            range-dependent auto-pick never applies. Explicit values:
            ``'linear'``, ``'n2linear'``, ``'pchip'``, ``'cubic'`` /
            ``'spline'``. ``env.ssp.shape='isovelocity'`` always forces
            ``'C'`` regardless.
        output_mode : str, optional
            'R' (horizontal array), 'D' (vertical array), 'S' (snapshot). Default: 'R'.

            All three return the received time series ``p(z, r, t)`` on the
            same convention. ``'R'`` loops the binary per receiver depth,
            ``'D'`` per receiver range; ``'S'`` runs once and is converted
            in-tree by one inverse Hankel transform per output time
            (``uacpy.io.grn_reader.sparc_snapshot_to_time_field``).
            CW transmission loss is not offered — see the
            ``RunMode.COHERENT_TL`` refusal in :meth:`run` for why.
        pulse_type : str, optional
            Pulse type string. Default: 'PN+B'.
        n_t_out : int, optional
            Number of time samples. Default: 512.
        t_max : float, optional
            Maximum time (s). None = auto (2.5x travel time). Default: None.
        t_start : float, optional
            Time the march begins (s); negative starts from rest before the
            pulse turns on. Default: -0.1.
        t_mult : float, optional
            Courant safety factor on the marching time step (1.0 = the
            stability limit). Default: 0.999.
        max_depths : int, optional
            Cap on the looped axis: receiver depths for ``output_mode='R'``,
            receiver ranges for ``'D'``. ``'S'`` is uncapped. Default: 20.
        rmax_safety_margin : float, optional
            Multiplier on ``receiver.ranges.max()`` to set SPARC's RMax.
            RMax sets the wavenumber step (``Δk ≈ 2π/RMax_m``,
            ``sparc.f90:116``) and the range synthesis is a direct ``Δk``
            sum, so the r-domain output is periodic with period RMax — the
            source's image at r=RMax contaminates the receivers unless
            RMax is pushed well past them. Default: ``None`` → 3.0
            (alias at 3× receiver max).
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
        frequencies: Optional[np.ndarray] = None,
        source_waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        output_duration: Optional[float] = None,
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
            ``TIME_SERIES`` (the default and the only supported mode):
            return the native pressure time series. ``COHERENT_TL``
            raises :class:`UnsupportedFeatureError`.

        Returns
        -------
        result : Field
            Real ``p(depth, range, time)`` on SPARC's output time grid.

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

        # SPARC builds p(t) from its native pulse_type on its own time grid
        # at source.frequencies — none of the contract extras can influence
        # the run.
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
        media_depth = self._total_media_depth(env)

        self.validate_inputs(env, source, receiver, run_mode=run_mode)
        reject_unsupported_ssp_interp('SPARC', self.interp_ssp)

        fm = self._setup_file_manager()

        try:
            base_name = 'model'
            freq = source.frequencies[0]

            if self.output_mode == 'D':
                result = self._run_vertical(fm, env, source, receiver,
                                            base_name, freq)
            elif self.output_mode == 'S':
                result = self._run_snapshot(fm, env, source, receiver,
                                            base_name, freq)
            else:
                result = self._run_range_native(fm, env, source, receiver,
                                                base_name, freq)
            return self._mask_unresolvable_depths(
                result, receiver, media_depth)

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def _reject_oversized_loop_axis(self, n_runs: int, axis: str,
                                    mode_label: str) -> None:
        """Cap the axis the wrapper loops the binary over.

        ``output_mode='R'`` runs one SPARC subprocess per receiver depth and
        ``'D'`` one per receiver range; ``'S'`` runs once and never calls this.
        """
        if n_runs <= self.max_depths:
            return
        raise UnsupportedFeatureError(
            model_name='SPARC',
            feature=(
                f"{n_runs} receiver {axis} (SPARC {mode_label} runs one "
                f"simulation per {axis[:-1]}; current limit is "
                f"max_depths={self.max_depths})"
            ),
            alternatives=[
                f"Reduce receiver.{axis} to at most {self.max_depths} entries",
                f"Raise the limit explicitly: SPARC(max_depths={n_runs})",
                "SPARC(output_mode='S') computes the whole grid in one run",
                "Bellhop, RAM, Kraken, Scooter, or OASN for dense 2D fields",
            ],
        )

    def _finalize_sparc_result(self, result, fm, run_bases):
        """Attach output paths + finishing log; the shared R/D/S tail.

        ``run_bases`` is the list of per-run base names in dispatch order
        ('R' loops per depth, 'D' per range, 'S' runs once). The first entry
        supplies the ``rts_file`` / ``grn_file`` / ``prt_file`` paths; the
        sibling runs write next to it in the same work dir.
        """
        self._attach_output_paths(
            result, fm.work_dir, run_bases[0],
            primary_files=(
                ('grn_file', '.grn'),
                ('rts_file', '.rts'),
            ),
        )
        self._log("Simulation complete")
        return result

    def _run_and_read_rts(self, fm, run_base):
        """Run the binary for ``run_base`` and read back its ``.rts``."""
        self._run_sparc(run_base, fm.work_dir)
        rts_file = fm.get_path(f'{run_base}.rts')
        if not rts_file.exists():
            exc = ModelExecutionError(
                self.model_name, return_code=0, stdout=None,
                stderr=f"SPARC did not produce {rts_file}",
            )
            self._attach_prt_tail(exc, fm.work_dir, run_base)
            raise exc
        return read_rts_file(rts_file)

    def _require_shared_time_grid(self, rts_data, time, fm, run_base):
        """Every run of a loop must land on one output time grid.

        The per-run traces are stacked onto a single ``time`` coordinate, so a
        divergence would silently mislabel every trace but the first.
        """
        got = np.asarray(rts_data['time'], dtype=float)
        if got.shape == time.shape and np.allclose(got, time,
                                                   rtol=1e-9, atol=1e-12):
            return
        span = (float(got[-1] - got[0]) if got.size else 0.0)
        ref_span = (float(time[-1] - time[0]) if time.size else 0.0)
        exc = ModelExecutionError(
            self.model_name, return_code=0, stdout=None,
            stderr=(
                f"SPARC run {run_base} returned an output time grid "
                f"(n={got.size}, span={span:.6g} s) that differs from the "
                f"first run's (n={time.size}, span={ref_span:.6g} s); the "
                f"per-run traces share one time axis and would be mislabelled."
            ),
        )
        self._attach_prt_tail(exc, fm.work_dir, run_base)
        raise exc

    def _run_range_native(self, fm, env, source, receiver, base_name, freq):
        """``output_mode='R'``: the horizontal (range-native) received time
        series. ``sparc.f90`` writes one receiver depth per run, so the wrapper
        loops over depths and stacks the traces."""
        depths = np.atleast_1d(np.asarray(receiver.depths, dtype=float))
        self._reject_oversized_loop_axis(
            len(depths), 'depths', "horizontal-array mode (output_mode='R')")
        self._log(f"Computing {len(depths)} depth(s) "
                  f"(SPARC horizontal array mode)...")

        traces, run_bases = [], []
        ranges_out = time = dt = nt = None

        for idx, depth in enumerate(depths):
            single = Receiver(depths=np.array([depth]), ranges=receiver.ranges)
            run_base = base_name if len(depths) == 1 else f'{base_name}_d{idx}'
            run_bases.append(run_base)
            self._write_sparc_env(fm.get_path(f'{run_base}.env'), env, source,
                                  single)
            self._log(f"  depth {idx + 1}/{len(depths)}: {float(depth):.1f} m")

            rts_data = self._run_and_read_rts(fm, run_base)
            if time is None:
                ranges_out = np.asarray(rts_data['ranges'], dtype=float)
                time = np.asarray(rts_data['time'], dtype=float)
                dt, nt = rts_data['dt'], rts_data['nt']
            else:
                self._require_shared_time_grid(rts_data, time, fm, run_base)
            # rts_data['p'] is (nt, n_range) here; want (n_range, nt).
            #
            # The far-field Hankel kernel is H0(kr) ~ sqrt(2/(pi·k·r))·
            # e^{i(kr-pi/4)}, so inverting it weights each wavenumber sample by
            # dk·k·sqrt(2/(pi·k·r)) = dk·sqrt(2k/(pi·r)). sparc.f90's 'D' branch
            # carries exactly that (sqrt(2)·dk·sqrt(k) at :595 times the
            # 1/sqrt(pi·Rr) write scale at :292). The 'R' branch at :622-623
            # applies sqrt(2)·dk·sqrt(k/r) and no write scale, i.e. the same
            # weight without the 1/sqrt(pi) — so it is sqrt(pi) hot. Divide it
            # back onto the kernel every other mode uses.
            traces.append(np.asarray(rts_data['p']).T / np.sqrt(np.pi))

        # (n_depth, n_range, n_time) — the shared Field contract. The range
        # axis is SPARC's actual output grid; Field validates its length
        # against the data shape.
        result = Field(
            data=_mask_zero_range_traces(
                np.stack(traces, axis=0), ranges_out, 'R'),
            coords={'depth': depths, 'range': ranges_out, 'time': time},
            **self._result_kwargs(
                source,
                backend='sparc',
                frequencies=freq,
                phase_reference='time_domain_native',
                output_mode='R',
                n_depth_runs=len(depths),
                dt=float(dt),
                fs=(1.0 / float(dt)) if dt else float('nan'),
                nt=int(nt),
                t_start=float(time[0]) if len(time) else 0.0,
            ),
        )
        return self._finalize_sparc_result(result, fm, run_bases)

    def _run_vertical(self, fm, env, source, receiver, base_name, freq):
        """``output_mode='D'``: the vertical-array received time series.

        ``sparc.f90:593-606`` accumulates ``RTSrz(ir, Itout)`` — a time series
        at each *depth* for a fixed range — the same kind of output as ``'R'``
        sampled down a vertical array instead of along a horizontal one. SPARC
        runs one range at a time, so the wrapper loops.

        Every run is written against the *full* receiver-range extent so all of
        them share one ``RMax`` and therefore one output time grid; without
        that each run would size its own window from its own single range and
        the stacked traces would carry mismatched times.

        The ``.rts`` written in this mode stores the depth axis in the slot the
        horizontal mode uses for ranges, which is why ``rts_data['ranges']``
        is read as depths here.
        """
        ranges = np.atleast_1d(np.asarray(receiver.ranges, dtype=float))
        self._reject_oversized_loop_axis(
            len(ranges), 'ranges', "vertical-array mode (output_mode='D')")
        self._log(f"Computing vertical array at {len(ranges)} range(s)...")

        rmax_reference_m = float(np.max(ranges))
        traces, run_bases, depths_out = [], [], None
        time = dt = nt = None

        for idx, rng in enumerate(ranges):
            single = Receiver(depths=receiver.depths, ranges=np.array([rng]))
            run_base = base_name if len(ranges) == 1 else f'{base_name}_r{idx}'
            run_bases.append(run_base)
            self._write_sparc_env(fm.get_path(f'{run_base}.env'), env, source,
                                  single, rmax_reference_m=rmax_reference_m)
            self._log(f"  range {idx + 1}/{len(ranges)}: {float(rng):.1f} m")

            rts_data = self._run_and_read_rts(fm, run_base)
            if time is None:
                depths_out = np.asarray(rts_data['ranges'], dtype=float)
                time = np.asarray(rts_data['time'], dtype=float)
                dt, nt = rts_data['dt'], rts_data['nt']
            else:
                self._require_shared_time_grid(rts_data, time, fm, run_base)
            # rts_data['p'] is (nt, n_depth) here; want (n_depth, nt).
            # sparc.f90:595 + :292 give this branch the full inverse-Hankel
            # weight dk·sqrt(2k/(pi·r)), so it needs no correction — it is the
            # convention the other two modes are brought onto.
            traces.append(np.asarray(rts_data['p']).T)

        # (n_depth, n_range, n_time) — the shared Field contract.
        result = Field(
            data=_mask_zero_range_traces(
                np.stack(traces, axis=1), ranges, 'D'),
            coords={'depth': depths_out, 'range': ranges, 'time': time},
            **self._result_kwargs(
                source,
                backend='sparc',
                frequencies=freq,
                phase_reference='time_domain_native',
                output_mode='D',
                n_range_runs=len(ranges),
                dt=float(dt),
                fs=(1.0 / float(dt)) if dt else float('nan'),
                nt=int(nt),
                t_start=float(time[0]) if len(time) else 0.0,
            ),
        )
        return self._finalize_sparc_result(result, fm, run_bases)

    def _run_snapshot(self, fm, env, source, receiver, base_name, freq):
        """``output_mode='S'``: the snapshot — the whole field at each output
        time.

        SPARC accumulates the wavenumber-domain field ``Green(Itout, irz, ik)``
        (``sparc.f90:580-591``) and dumps it to a ``.grn``
        (``sparc.f90:282-290``); ``doc/sparc.htm`` says FIELDS must be
        run afterwards to turn it into a pressure field. uacpy does that step
        in-tree via one inverse Hankel transform per output time, giving
        ``p(z, r, t)`` on the requested receiver grid in a single SPARC run —
        unlike ``'R'``/``'D'``, which loop the binary per depth / per range.
        """
        self._log("Computing snapshot (whole field per output time)...")
        env_file = fm.get_path(f'{base_name}.env')
        self._write_sparc_env(env_file, env, source, receiver)
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
        result.metadata['output_mode'] = 'S'
        return self._finalize_sparc_result(result, fm, [base_name])

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
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
        e = env.copy()
        for col in e.bottom.columns:
            col.halfspace.acoustic_type = 'rigid'
        return e

    def _resolve_rmax_safety_margin(self) -> float:
        """Pick the effective ``rmax_safety_margin`` for this run.

        ``RMax`` fixes SPARC's wavenumber step (``sparc.f90:116``:
        ``Nk = INT(1000·RMax_km·(kMax−kMin)/2π)`` ⇒ ``Δk ≈ 2π/RMax_m``).
        The ``'R'`` / ``'D'`` modes synthesise range inline in ``EXTRACT``
        as a direct ``Δk`` sum over that grid (``sparc.f90:595,622``), and
        the ``'S'`` snapshot is transformed in-tree by the same kind of
        direct DFT (``grn_reader._hankel_transform``) — no FFT anywhere. A
        uniform-``Δk`` sum is periodic in range with period ``2π/Δk =
        RMax``, so the alias is a visible non-physical wave at the far range
        edge unless ``RMax`` is pushed well past the receivers. User-pinned
        values win.
        """
        if self.rmax_safety_margin is not None:
            return float(self.rmax_safety_margin)
        return 3.0

    def _write_sparc_env(self, filepath, env, source, receiver, *,
                         rmax_reference_m=None):
        """
        Write SPARC environment file using shared ATEnvWriter

        SPARC extends ReadEnvironmentMod format with:
        - Output mode in TopOpt (5th character: R=horizontal, D=vertical, S=snapshot)
        - Limited bottom types (only vacuum and rigid, no halfspace)
        - Time-domain pulse parameters
        - Time output parameters
        - Integration parameters

        ``rmax_reference_m`` overrides the range ``RMax`` (and hence the output
        time window) is derived from; ``None`` uses ``receiver.ranges.max()``.
        The ``'D'`` loop pins it to the full receiver extent so its per-range
        runs share one time grid.
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
        r_ref = (float(rmax_reference_m) if rmax_reference_m is not None
                 else float(receiver.ranges.max()))
        rmax_m = r_ref * self._resolve_rmax_safety_margin()

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

        n_t_out = self._resolve_n_t_out(f_max, t_max)

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

    def _resolve_n_t_out(self, f_max, t_max):
        """Output time-sample count — the caller's ``n_t_out``, with a warning
        when the resulting grid cannot resolve the pulse band.

        The native ``p(t)`` sampling is the caller's to choose, so the value is
        kept verbatim. But a grid whose Nyquist ``0.5 · n_t_out / t_max`` sits
        below ``f_max`` aliases silently: the returned p(t) looks perfectly
        plausible at the wrong frequency. Say so, and name the ``n_t_out``
        that fixes it.

        The window is ``[0, t_max]`` — the *output* window the writer emits
        (``oalib_writer``), not ``t_start``, which only sets where the
        integration begins.
        """
        window = float(t_max)
        if window > 0 and f_max > 0:
            fs = self.n_t_out / window
            if f_max > 0.5 * fs:
                n_needed = int(np.ceil(
                    window * _SPARC_PULSE_OVERSAMPLE * 2.0 * f_max))
                warnings.warn(
                    f"SPARC TIME_SERIES: the output grid samples at "
                    f"{fs:.1f} Hz (Nyquist {fs / 2:.1f} Hz) over a "
                    f"{window:.2f} s window, below the {f_max:.0f} Hz "
                    f"source band — p(t) will alias. Set n_t_out>="
                    f"{n_needed}, lower f_max, or shorten the window via "
                    f"t_max / receiver.ranges.max().",
                    UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
                )
        return self.n_t_out

    def _run_sparc(self, base_name: str, work_dir: Path):
        """
        Run SPARC as a subprocess (180 s timeout by default).

        Delegates to ``PropagationModel._run_subprocess`` (which raises the
        child stack limit — required because ``MARCH`` declares twelve
        automatic ``COMPLEX(NTot1)`` arrays, ``sparc.f90:354-355``, which
        gfortran puts on the stack). On failure, appends the ``.prt`` tail to
        the raised ``ModelExecutionError`` for easier diagnosis. Override via
        the ``timeout`` constructor kwarg.
        """
        self._run_and_attach_prt(
            [str(self._exe), base_name], work_dir, base_name,
            timeout=self.timeout, stale_outputs=('.grn', '.rts'))
