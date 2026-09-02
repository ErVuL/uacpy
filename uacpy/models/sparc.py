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
from typing import Dict, List, Optional, Union
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
    ConfigurationError, ModelExecutionError,
    UnsupportedFeatureError,
)
from uacpy.io.grn_reader import (
    read_grn_file, sparc_snapshot_to_time_field,
)
from uacpy.models.base import (
    PropagationModel, RunMode, ModelSpec, USER_FRAME_SKIP,
)
from uacpy.io.oalib_reader import read_rts_file
from uacpy.io.oalib_writer import (
    write_sparc_env_file, reject_unsupported_ssp_interp,
    resolve_phase_speed_bounds,
    SOURCE_TYPE_CODE as _SOURCE_TYPE_CODE,
    at_env_media,
)


# Factor above the Nyquist minimum used when naming the ``n_t_out`` that would
# resolve the pulse band in the aliasing warning.
_SPARC_PULSE_OVERSAMPLE = 3.0

# Tail of the output window measured by ``_warn_on_truncated_window``, and the
# peak-relative level (-20 dB) at which that tail counts as a truncated trace.
_TRUNCATION_TAIL_FRACTION = 0.1
_TRUNCATION_LEVEL = 0.1

# Ceiling on the complex64 Green's-function cube an ``output_mode='S'`` run
# holds (both in the binary and in the reader): 2 GiB.
_MAX_SNAPSHOT_GREEN_BYTES = 2 * 1024 ** 3


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


def _peak_ignoring_nan(data: np.ndarray) -> float:
    """``np.nanmax(data)`` as a float, returning NaN for an all-NaN input.

    ``np.nanmax`` reports that case by raising a RuntimeWarning, and the only
    way to mute one is ``warnings.catch_warnings()`` — whose filter stack is
    process-global, so the window would also swallow warnings other threads
    raise while it is open. Masking the NaNs decides the same thing without
    touching global state. ``where=``/``initial=`` keep ``nanmax``'s treatment
    of the infinities: they are values, not gaps, and propagate to the result.
    """
    present = ~np.isnan(data)
    if not present.any():
        return float('nan')
    return float(np.max(data, where=present, initial=-np.inf))


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


# The sampling ReadEnvironmentMod.f90:103 sizes an automatic mesh at:
# ``deltaz = c / freq0 / 20``. Written explicitly for SPARC because sparc.f90
# never rescales the mesh with frequency the way kraken.f90:75 and
# scooter.f90:106 do, so "automatic" means 20 points per wavelength at the
# deck's nominal frequency and only 10 at the top of the pulse band this
# wrapper writes.
_AT_POINTS_PER_WAVELENGTH = 20.0


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
        Maximum simulated time (s). ``None`` ⇒ ``2.5 ×`` the direct travel
        time at the slowest speed in the profile. That is a heuristic, not a
        bound on the last arrival — a waveguide's tail is set by the slowest
        modal *group* velocity, which falls to zero at cutoff — so a run
        whose trace is still ringing at the end of the window emits a
        ``UserWarning`` saying the trace is truncated.
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
        ``RMax``. ``None`` (default) ⇒ ``4.0``. ``RMax`` fixes the
        wavenumber sampling — ``sparc.f90:116`` takes
        ``Nk = INT(1000·RMax_km·(kMax−kMin)/2π)``, i.e.
        ``Δk ≈ 2π/RMax_m`` — and the range synthesis is a direct ``Δk``
        sum over that grid, so the output is periodic in range with
        period ``2π/Δk = RMax``. The replica that period puts at
        ``r = RMax`` is not the whole story: a receiver at range ``r``
        sees the fold from ``r − RMax``, i.e. the arrival that belongs at
        ``|RMax − r|``, so the alias reaches it at ``(RMax − r)/c`` and
        pushing ``RMax`` just past the receivers does *not* clear it. At
        margin ``m`` the alias lands at ``(m−1)·r/c`` while the auto
        window runs to ``2.5·r/c``, so the margin has to exceed ``3.5``;
        ``4.0`` is the first round value that does. Increasing this knob
        raises ``Nk`` (and SPARC runtime) in proportion. A pinned margin
        (or a pinned ``t_max``) that leaves the alias inside the output
        window emits a ``UserWarning`` naming the margin that clears it.
        ``sparc.f90:153-155`` rejects the run outright when the largest
        receiver range exceeds ``RMax``.
    timeout : float, optional
        Subprocess timeout per run (s). Default ``600.0`` — sized so the
        suite's longest SPARC runs survive a fully loaded CPU (an
        oversubscribed 8-worker pytest session slows an ~85 s march past
        3x), while a hung binary still dies.
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
    samples over ``[0, t_max]`` (``t_start`` only sets where the time
    integration begins, not the output window). A grid whose Nyquist
    sits below the pulse band ``f_max`` aliases and emits a
    ``UserWarning`` naming the ``n_t_out`` that would resolve it.

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

    # TopOpt position 4 carries env.absorption to the engine.
    _consumes_volume_absorption = True

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
        timeout: float = 600.0,
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
            Total mesh points per medium (not per wavelength). 0 = auto,
            which lets SPARC size the mesh from the frequency. Default: 0.
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
            Maximum time (s). None = auto (2.5x the direct travel time,
            which does not bound a waveguide's slow modal tail — a
            still-ringing trace is warned about after the run).
            Default: None.
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
            sum, so the r-domain output is periodic with period RMax — a
            receiver at range r sees the fold from ``|RMax − r|`` at time
            ``(RMax − r)/c``, which stays inside the output window until
            the margin exceeds 3.5. Default: ``None`` → 4.0.
        f_min, f_max : float, optional
            Pulse frequency band (Hz). ``None`` (default) resolves at
            ``run()`` time to one octave around the source frequency
            (``max(f/2, 0.1)`` .. ``2f``) — SPARC's work scales with
            the band's wavenumber span, so a much wider band slows it
            sharply. ``f_min=0`` is accepted (``sparc.f90:114`` clamps
            the resulting ``kMin`` to 1e-20).
        sound_speed : float, optional
            Water sound speed (m/s) used for the travel-time window
            when ``t_max`` is auto. ``None`` (default) → the slowest
            speed in ``env.ssp``, which puts the *direct* arrival well
            inside the window; :data:`DEFAULT_SOUND_SPEED` only when the
            profile carries none.
        timeout : float, optional
            Subprocess timeout (s) for each SPARC run. Default: 600.0.
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
        # The march runs time = t_start + (i-1)·Δt from a field at rest
        # (sparc.f90:409) while the deck always asks for the output window
        # [0, t_max]. A positive t_start therefore starts the solution after
        # cans.f90's pulse has already turned on (it is zero only for T <= 0),
        # and EXTRACT's DO WHILE (:571) empties every requested time below
        # t_start out of the first step at a negative interpolation weight.
        if float(t_start) > 0.0:
            raise ConfigurationError(
                f"SPARC t_start={t_start} s starts the time march after the "
                f"source pulse has turned on (t = 0) and after the start of "
                f"the [0, t_max] output window the deck writes, so the field "
                f"is marched from rest mid-pulse and the output times below "
                f"t_start are extrapolated backwards out of the first step. "
                f"Use t_start <= 0 (the default -0.1 s pre-rolls the march "
                f"before the pulse)."
            )
        self.t_start = t_start
        self.t_mult = t_mult
        self.max_depths = max_depths
        self.rmax_safety_margin = rmax_safety_margin
        # Pulse band (``f_min``/``f_max``) and ``sound_speed`` (used for
        # the travel-time window when ``t_max`` is auto) default to
        # ``None`` and are resolved at ``run()`` time from
        # ``source.frequencies[0]`` and the environment's own profile.
        self.f_min = float(f_min) if f_min is not None else None
        self.f_max = float(f_max) if f_max is not None else None
        # An inverted or negative band feeds sparc.f90:116 a negative
        # kMax-kMin, i.e. Nk < 0 — the same silent all-zero field the run()
        # Nk guard refuses — so fail here with the cause named. f_min = 0 is
        # legal and stays: sparc.f90:114 clamps kMin to 1e-20 for exactly
        # that case, and doc/sparc.htm's own example deck reads "0.0 15.0".
        if self.f_min is not None and self.f_min < 0.0:
            raise ConfigurationError(
                f"SPARC pulse band requires f_min >= 0 Hz; got {self.f_min}."
            )
        if self.f_max is not None and self.f_max <= 0.0:
            raise ConfigurationError(
                f"SPARC pulse band requires f_max > 0 Hz; got {self.f_max}."
            )
        if (self.f_min is not None and self.f_max is not None
                and self.f_min >= self.f_max):
            raise ConfigurationError(
                f"SPARC pulse band requires f_min < f_max; got "
                f"f_min={self.f_min} Hz, f_max={self.f_max} Hz."
            )
        self.sound_speed = (
            float(sound_speed) if sound_speed is not None else None
        )

        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        self._exe = self._resolve_executable(
            executable,
            lambda: self._find_executable_in_paths(
                'sparc.exe', bin_subdirs='oalib',
                dev_subdir='Acoustics-Toolbox/Scooter',
            ),
        )

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
        self._require_run_triple(env, source, receiver)
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
            freq = self._resolve_pulse_frequency(source)

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
            fm.finish()

    def _resolve_pulse_frequency(self, source: Source) -> float:
        """The single frequency SPARC's pulse band is derived from.

        ``TIME_SERIES`` is not in ``_SINGLE_FREQUENCY_MODES`` (the mode is
        broadband by nature), but SPARC's deck carries one pulse band, and
        the auto band is one octave around this frequency
        (:meth:`_write_sparc_env`) — a multi-frequency Source silently loses
        every entry but the first, so say which one is being read.
        """
        freqs = np.atleast_1d(np.asarray(source.frequencies, dtype=float))
        if freqs.size > 1:
            warnings.warn(
                f"SPARC drives one pulse band per run: reading "
                f"source.frequencies[0] = {float(freqs[0]):.6g} Hz and "
                f"ignoring the other {freqs.size - 1} "
                f"({list(freqs[1:])}). The auto band is one octave around "
                f"it; pass SPARC(f_min=..., f_max=...) to cover the whole "
                f"source band in the single run.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
        return float(freqs[0])

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

    def _reject_oversized_snapshot(self, receiver, nk: int,
                                   n_t_out: int) -> None:
        """Cap the wavenumber-domain table ``output_mode='S'`` materialises.

        ``'R'`` / ``'D'`` are bounded by ``max_depths`` because the wrapper
        loops the binary over them; ``'S'`` runs once and is bounded by the
        ``Green(Itout, irz, ik)`` cube instead (``sparc.f90:580-591``), which
        the binary holds whole and ``read_grn_file`` reads back as a
        ``complex64`` array of the same shape. Its size is set by the
        wavenumber count — which grows with ``rmax_safety_margin`` and the
        pulse band — so it can reach tens of GB without any single knob
        looking unreasonable.
        """
        if self.output_mode != 'S':
            return
        n_depth = int(np.atleast_1d(np.asarray(receiver.depths)).size)
        n_bytes = 8 * int(n_t_out) * n_depth * int(nk)
        if n_bytes <= _MAX_SNAPSHOT_GREEN_BYTES:
            return
        raise UnsupportedFeatureError(
            model_name='SPARC',
            feature=(
                f"a snapshot (output_mode='S') whose Green's-function table "
                f"is {n_bytes / 1024 ** 3:.1f} GiB (n_t_out={int(n_t_out)} × "
                f"{n_depth} receiver depth(s) × Nk={int(nk)} × 8 B), over the "
                f"{_MAX_SNAPSHOT_GREEN_BYTES / 1024 ** 3:.1f} GiB cap"
            ),
            alternatives=[
                "Reduce n_t_out (the table scales with it one-for-one)",
                "Reduce receiver.depths",
                "Narrow the pulse band (f_min/f_max) or lower "
                "rmax_safety_margin — both set Nk",
                "SPARC(output_mode='R') or 'D', which stream one trace per "
                "run instead of holding the whole cube",
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
        self._warn_on_truncated_window(result)
        self._log("Simulation complete")
        return result

    def _warn_on_truncated_window(self, result) -> None:
        """Warn when p(t) is still ringing at the end of the output window.

        ``t_max`` is sized from the direct travel time
        (:meth:`_write_sparc_env`), which does not bound the last arrival: in
        a waveguide that is set by the slowest modal *group* velocity, which
        falls to zero at cutoff, and SPARC's vacuum / rigid boundaries leave
        the near-cutoff tail undamped. An arrival past ``t_max`` is absent
        from p(t) with nothing in the output marking it — so measure the tail
        that did come back (the peak over the final tenth of the window
        against the peak over the whole of it) and say so while it is within
        ``_TRUNCATION_LEVEL`` of the peak.
        """
        time = np.asarray(result.coords.get('time', ()), dtype=float)
        data = np.abs(np.asarray(result.data))
        if time.size < 10 or data.ndim == 0 or data.shape[-1] != time.size:
            return
        n_tail = max(1, int(round(_TRUNCATION_TAIL_FRACTION * time.size)))
        # An all-NaN slice (every receiver masked) is not a finding, and
        # :func:`_peak_ignoring_nan` returns NaN for one without raising the
        # warning that would have to be muted process-wide.
        peak = _peak_ignoring_nan(data)
        tail = _peak_ignoring_nan(data[..., -n_tail:])
        if not (np.isfinite(peak) and np.isfinite(tail) and peak > 0.0):
            return
        ratio = tail / peak
        if ratio < _TRUNCATION_LEVEL:
            return
        warnings.warn(
            f"SPARC TIME_SERIES: p(t) is still at {100.0 * ratio:.0f}% of "
            f"its peak over the last {100.0 * _TRUNCATION_TAIL_FRACTION:.0f}%"
            f" of the [0, {float(time[-1]):.4g}] s output window, so arrivals "
            f"past it are missing from the returned trace. The auto window is "
            f"2.5 direct travel times, which does not bound the slow modal "
            f"tail of a waveguide — raise it with SPARC(t_max=...) (and keep "
            f"n_t_out / rmax_safety_margin in step with the longer window).",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )

    def _run_and_read_rts(self, fm, run_base):
        """Run the binary for ``run_base`` and read back its ``.rts``."""
        self._run_sparc(run_base, fm.work_dir)
        # A missing or empty .rts means the binary died silently; the raised
        # error carries the .prt tail with the actual cause.
        rts_file = self._require_output(
            [fm.get_path(f'{run_base}.rts')],
            what='a time-series file (.rts)',
            prt_base=run_base, work_dir=fm.work_dir,
        )
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
        # against the data shape. r = 0 columns are no-data:
        # ``sparc.f90:622`` weights this branch by ``SQRT( rkT / Pos%Rr )``,
        # singular there — measured, it mixes NaN with exact zeros a caller
        # would read as real, quiet samples. The snapshot mode's in-tree
        # Hankel transform masks the same cells, so all three output modes
        # give one answer.
        result = Field(
            data=self._mask_zero_range_columns(
                np.stack(traces, axis=0), ranges_out,
                "output_mode='R''s 1/sqrt(r) cylindrical-spreading factor"),
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

        # (n_depth, n_range, n_time) — the shared Field contract. r = 0
        # columns are no-data: ``sparc.f90:292`` scales this branch by
        # ``1 / SQRT( pi * Pos%Rr( 1 ) )``, singular there — measured, the
        # whole trace comes back +Inf.
        result = Field(
            data=self._mask_zero_range_columns(
                np.stack(traces, axis=1), ranges,
                "output_mode='D''s 1/sqrt(r) cylindrical-spreading factor"),
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

        # A missing or empty .grn means the binary died silently; the raised
        # error carries the .prt tail with the actual cause.
        grn_file = self._require_output(
            [fm.get_path(f'{base_name}.grn')],
            what='a snapshot file (.grn)',
            prt_base=base_name, work_dir=fm.work_dir,
        )

        result = sparc_snapshot_to_time_field(
            read_grn_file(grn_file),
            np.atleast_1d(receiver.ranges),
            frequency=freq,
            source_type=_SOURCE_TYPE_CODE[source.source_type],
        )
        self._stamp_result(result, source, backend='sparc',
                           frequencies=freq,
                           phase_reference='time_domain_native')
        result.metadata['output_mode'] = 'S'
        return self._finalize_sparc_result(result, fm, [base_name])

    _receivers_reach_sediment = True    # meshes through the sediment stack

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
        edge unless ``RMax`` is pushed well past the receivers.

        The fold is not only the replica sitting at ``r = RMax``: the
        receiver at range ``r`` also carries the arrival that belongs at
        ``r - RMax``, i.e. a copy of the source response from the distance
        ``|RMax - r|``, reaching it at ``(RMax - r)/c``. Pushing ``RMax``
        just past the receivers therefore leaves the alias *early*, not
        late. At margin ``m`` the farthest receiver gets it at
        ``(m-1)*r/c`` against an auto window of ``2.5*r/c``
        (:meth:`_write_sparc_env`), so the margin has to exceed 3.5.
        User-pinned values win; :meth:`_warn_on_range_alias` checks whichever
        margin is in force against the window actually written.

        Scooter needs no such margin: ``scooter.f90:69`` samples four times
        as finely (``Nk = 2000*RMax_km*(kMax-kMin)/pi``, i.e.
        ``Δk ≈ π/(2·RMax)``), putting its period at ``4·RMax``.
        """
        if self.rmax_safety_margin is not None:
            return float(self.rmax_safety_margin)
        return 4.0

    def _profile_speed_bounds(self, env: Environment) -> tuple:
        """``(slowest, fastest)`` sound speed (m/s) for the timing checks.

        The output window is anchored on the slowest speed (the latest the
        direct arrival can land) and the range-alias check on the fastest
        (the earliest the folded replica can). ``sound_speed`` pins the slow
        end; ``DEFAULT_SOUND_SPEED`` covers a profile that carries no usable
        speed at all.

        The two ends read different parts of the environment, on purpose.
        ``c_slow`` stays water-only: it anchors ``t_max``, which is a
        heuristic rather than a bound on the last arrival (see
        :meth:`_write_sparc_env`) and is backstopped after the run by
        :meth:`_warn_on_truncated_window` measuring the trace that came back.
        ``c_fast`` spans every medium the deck carries, water and seabed
        alike, because the replica :meth:`_warn_on_range_alias` looks for
        travels the fold distance through the whole waveguide: a fast
        sediment layer over a 1500 m/s column carries it in ahead of any
        water path. That is also what the binary itself does —
        ``Scooter/sparc.f90:207`` runs ``MediumLoop: DO medium = 1,
        SSP%NMedia`` taking ``cMax = MAX(cpR, cMax)`` over EVERY medium, and
        ``write_layer_sections`` emits each sediment layer as one more
        medium with ``NMEDIA`` incremented. ``all_sound_speeds`` is the
        matching accessor: compressional speeds only, skipping the
        vacuum / rigid / file / precalc half-spaces that carry no seabed
        speed — including the one ``_sparc_rigidify_halfspace`` has already
        converted, which the deck no longer marches as a medium.
        """
        speeds = np.asarray(env.ssp.data, dtype=float).ravel()
        speeds = speeds[np.isfinite(speeds) & (speeds > 0.0)]
        c_slow = self.sound_speed
        if c_slow is None:
            c_slow = (float(speeds.min()) if speeds.size
                      else DEFAULT_SOUND_SPEED)
        c_fast = float(speeds.max()) if speeds.size else float(c_slow)
        seabed = [c for c in env.bottom.all_sound_speeds()
                  if np.isfinite(c) and c > 0.0]
        if seabed:
            c_fast = max(c_fast, max(seabed))
        return float(c_slow), max(float(c_fast), float(c_slow))

    def _warn_on_range_alias(self, rmax_m: float, r_ref: float,
                             c_fast: float, t_max: float) -> None:
        """Warn when the ``Δk`` sum's range replica falls inside ``[0, t_max]``.

        The farthest receiver (``r_ref``) sees the fold from ``RMax - r_ref``
        (see :meth:`_resolve_rmax_safety_margin`); the earliest it can arrive
        is that distance over the fastest speed in the profile. Inside the
        output window it is indistinguishable from a real late arrival, so
        name the ``rmax_safety_margin`` that pushes it clear.
        """
        if not (r_ref > 0.0 and c_fast > 0.0 and t_max > 0.0):
            return
        t_alias = (rmax_m - r_ref) / c_fast
        if t_alias > t_max:
            return
        margin_needed = 1.0 + t_max * c_fast / r_ref
        warnings.warn(
            f"SPARC range alias: RMax = {rmax_m:.6g} m puts the "
            f"wavenumber sum's range replica at t = {t_alias:.4g} s at the "
            f"farthest receiver ({r_ref:.6g} m), inside the "
            f"{t_max:.4g} s output window — it will read as a real late "
            f"arrival. Use SPARC(rmax_safety_margin>"
            f"{margin_needed:.3g}) (runtime scales with it) or shorten the "
            f"window with t_max.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )

    def _resolve_n_mesh(self, env, f_max: float) -> Union[int, List[int]]:
        """Mesh count for the deck: the caller's if pinned, else sized at the
        band top this run actually marches. A pinned ``n_mesh`` comes back as
        a single ``int`` the writer applies to every medium; the automatic
        path returns a ``list`` with one count per medium.

        ``misc/ReadEnvironmentMod.f90:103`` sizes an automatic mesh as
        ``deltaz = c / freq0 / 20`` — 20 points per wavelength at the deck's
        NOMINAL frequency. KRAKEN and SCOOTER then rescale it per frequency
        (``kraken.f90:75`` and ``scooter.f90:106`` both multiply ``NG`` by
        ``freq/freq0``), which is what lets those wrappers check the mesh at
        ``freq0`` and be done. ``sparc.f90`` has no such rescaling anywhere: it
        takes ``N`` as written and sets ``h = (Depth(2) - Depth(1)) / N``
        (``:54``, ``:85``). But this wrapper writes a pulse band reaching
        ``2*freq`` by default, so an automatic mesh ran the top of that band at
        10 points per wavelength.

        Measured on a 100 m isovelocity guide, vacuum surface / rigid bottom,
        source and receiver at 50 m, TIME_SERIES, against a converged mesh:
        the automatic count is 0.239 relative rms at 20 Hz / 500 m and 0.375 at
        60 Hz / 300 m (peak -0.86 dB), with the coarse mesh also putting the
        arrival about 1.7 ms early over 300 m. Convergence is monotone in the
        count, so sizing at ``f_max`` rather than ``freq0`` is the fix rather
        than a tuning choice.
        """
        if self.n_mesh:
            return int(self.n_mesh)
        f = float(f_max)
        if not np.isfinite(f) or f <= 0.0:
            return int(self.n_mesh)
        # One count per MEDIUM, not one scalar: AT sizes each medium from its
        # OWN thickness and speed (ReadEnvironmentMod.f90:101-106), so the
        # water column's count broadcast to a 10 m sediment layer over-resolves
        # that layer by the thickness ratio and the march fails outright.
        return [max(int(_AT_POINTS_PER_WAVELENGTH * thickness * f / c), 10)
                for thickness, c in at_env_media(env)]

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
            # ``_validate_acoustic_type`` has already rejected unrecognised
            # names and ``_sparc_rigidify_halfspace`` has already converted a
            # halfspace, so what reaches here is a reflection-table bottom
            # ('file' / 'precalc') — valid everywhere else, unrepresentable in
            # SPARC's Vacuum/Rigid-only deck. Kraken and Scooter stage the
            # same table.
            raise UnsupportedFeatureError(
                'SPARC',
                f"a {hs.acoustic_type!r} bottom — its deck carries only the "
                f"'vacuum' and 'rigid' boundary conditions",
                alternatives=['Kraken', 'Scooter'],
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

        # Time output window (s), anchored on the travel time to the farthest
        # receiver (r_ref) — not on rmax_m, whose rmax_safety_margin factor is
        # a wavenumber-sampling knob (Δk ≈ 2π/RMax): folding the margin into
        # the window stretches [0, t_max] by that factor while n_t_out stays
        # fixed, aliasing the default output grid.
        #
        # The auto window is 2.5 direct travel times at the slowest speed the
        # profile carries. That is a heuristic, not a bound on the last
        # arrival: in a guide the
        # tail is set by the slowest modal GROUP velocity, which goes to zero
        # at cutoff, and SPARC's vacuum/rigid boundaries (sparc.f90:101-104)
        # leave that tail undamped. An arrival past t_max is simply absent
        # from p(t) with nothing in the output to say so, hence the post-run
        # ``_warn_on_truncated_window`` check on the trace that comes back.
        # ``sound_speed`` pins the speed; DEFAULT_SOUND_SPEED is only the
        # fallback for an environment that declares no usable speed at all.
        c_slow, c_fast = self._profile_speed_bounds(env)
        travel_time = r_ref / c_slow
        t_max = self.t_max if self.t_max is not None else travel_time * 2.5

        self._warn_on_range_alias(rmax_m, r_ref, c_fast, t_max)

        n_t_out = self._resolve_n_t_out(f_max, t_max)

        # Refuse a wavenumber count the binary cannot march. sparc.f90:116
        # computes Nk = INT( 1000.0*RMax_km*(kMax-kMin)/(2π) ) with
        # kMin = 2π·fMin/cHigh and kMax = 2π·fMax/cLow, which reduces to
        # rmax_m·(fMax/cLow − fMin/cHigh). Nk <= 0 runs the march loop
        # (sparc.f90:269) zero times, so every output comes back all-zero at
        # exit 0, and :265 reads k(Nk) out of bounds first. Nk = 1 is a
        # single-sample spectrum; require >= 2, the floor bounce.f90:49's own
        # tabulation enforces.
        c_low_res, c_high_res = resolve_phase_speed_bounds(
            env, self.c_low, self.c_high)
        nk = int(rmax_m * (f_max / c_low_res - f_min / c_high_res))
        if nk < 2:
            raise ConfigurationError(
                f"SPARC would march Nk = {nk} wavenumber(s) (sparc.f90:116 "
                f"with rmax = {rmax_m:.6g} m, band {f_min:.6g}–{f_max:.6g} Hz, "
                f"c_low = {c_low_res:.6g} m/s, c_high = {c_high_res:.6g} m/s): "
                f"the wavenumber loop would run empty and the field would come "
                f"back all-zero at exit 0.",
                remediation="Increase the receiver range or widen the pulse "
                            "band (f_min/f_max) so that "
                            "rmax·(f_max/c_low − f_min/c_high) ≥ 2.",
            )
        self._reject_oversized_snapshot(receiver, nk, n_t_out)

        write_sparc_env_file(
            filepath, env, source, receiver,
            ssp_code=ssp_code,
            surface_type=surface_type,
            bottom_type=bottom_type,
            output_mode=self.output_mode,
            n_mesh=self._resolve_n_mesh(env, f_max),
            rmax_m=rmax_m,
            # The pair resolved once above (for the Nk guard); the writer's
            # own resolver passes explicit values through unchanged.
            c_low=c_low_res, c_high=c_high_res,
            pulse_type=self.pulse_type,
            f_min=f_min, f_max=f_max,
            n_t_out=n_t_out,
            t_max=t_max,
            t_start=self.t_start, t_mult=self.t_mult,
        )

    def _resolve_n_t_out(self, f_max, t_max):
        """Output time-sample count — the caller's ``n_t_out``, with a warning
        when the resulting grid cannot resolve the pulse band.

        ``n_t_out < 2`` is refused: the deck writes the output times as the
        ``0.0 t_max /`` pair that ``misc/subtabulate.f90:24`` expands only
        for ``Nx >= 3``, and ``Nx = 1`` consumes the leading ``0.0`` alone —
        the whole window is discarded and the binary returns a single-sample
        p(t) at exit 0.

        The native ``p(t)`` sampling is the caller's to choose, so the value is
        kept verbatim. But a grid whose Nyquist sits below ``f_max`` aliases
        silently: the returned p(t) looks perfectly plausible at the wrong
        frequency. Say so, and name the ``n_t_out`` that fixes it. ``SubTab``
        expands ``0.0 t_max /`` inclusive of both endpoints, so the step is
        ``t_max/(n_t_out − 1)`` and the sample rate ``(n_t_out − 1)/t_max``
        — not ``n_t_out/t_max``, which overstates it by ``n/(n−1)``.

        The window is ``[0, t_max]`` — the *output* window the writer emits
        (``oalib_writer``), not ``t_start``, which only sets where the
        integration begins.
        """
        if int(self.n_t_out) < 2:
            raise ConfigurationError(
                f"SPARC: n_t_out={self.n_t_out} cannot express the "
                f"[0, t_max] output window: the deck writes the times as a "
                f"'0.0 t_max /' pair that misc/subtabulate.f90:24 expands "
                f"only for Nx >= 3, and Nx = 1 reads the leading 0.0 alone — "
                f"the run returns a single-sample p(t) at exit 0. Use "
                f"n_t_out >= 2."
            )
        window = float(t_max)
        if window > 0 and f_max > 0:
            fs = (int(self.n_t_out) - 1) / window
            if f_max > 0.5 * fs:
                n_needed = 1 + int(np.ceil(
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
        Run SPARC as a subprocess (600 s timeout by default).

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
