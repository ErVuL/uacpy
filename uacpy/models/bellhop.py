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

import copy
import shutil
import warnings
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union

from scipy.signal import hilbert

from uacpy.models.base import PropagationModel, RunMode
from uacpy.core.environment import Environment, BoundaryProperties
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.results import Result, Field
from uacpy.core.constants import (
    DEFAULT_SOUND_SPEED, DEFAULT_C_MIN, DEFAULT_C_MAX,
    DEFAULT_BROADBAND_N_FREQS, DEFAULT_BROADBAND_BANDWIDTH_FACTOR,
)
from uacpy.core.exceptions import (
    ConfigurationError, ExecutableNotFoundError, ModelExecutionError,
)
from uacpy.io.bellhop_writer import write_bellhop_env_file
from uacpy.io.oalib_reader import read_shd_file, read_arr_file, read_ray_file


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
    normalize_source: bool = False,
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
        One per-receiver arrival record from
        ``arrivals_field.by_receiver[isd][ird][irr]`` with keys:
        amplitudes, phases, delays, delays_imag, n_arrivals.
    source_timeseries : ndarray
        Source waveform (1-D). Used as-is unless ``normalize_source`` is
        set.
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
    normalize_source : bool, optional
        When ``True`` rescale the source waveform to peak ``|s| = 1``
        before convolution. Default ``False`` so the absolute amplitude
        calibration of the user-supplied waveform is preserved
        end-to-end.

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
        if time_window is not None:
            nrts = int(np.ceil(time_window * sample_rate))
        else:
            nrts = int(0.1 * sample_rate)
        t0 = 0.0 if t_start is None else float(t_start)
        return np.zeros(nrts), t0 + np.arange(nrts) / sample_rate

    amps = rcv_arrivals['amplitudes']
    phases_deg = rcv_arrivals['phases']
    delays = rcv_arrivals['delays']
    delays_imag = rcv_arrivals['delays_imag']

    sts = np.asarray(source_timeseries, dtype=float)
    if normalize_source:
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

        # ``delays_imag`` is Im(tau) in seconds; volume-attenuation factor
        # is exp(omega * Im(tau)) per delayandsum.m:134.
        atten = np.exp(omega_c * delays_imag[ia])

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


def _validate_arrivals_format(fmt: str) -> None:
    """Reject any arrivals_format other than ``'ascii'``.

    Bellhop's ``'binary'`` (FORTRAN unformatted) output is not parseable
    by uacpy's arrivals reader.
    """
    if fmt == 'binary':
        raise ConfigurationError(
            "Bellhop's binary arrivals format ('a' / FORTRAN unformatted) "
            "is not currently parseable by uacpy's arrivals reader. "
            "Use arrivals_format='ascii' instead."
        )
    if fmt != 'ascii':
        raise ConfigurationError(
            f"arrivals_format must be 'ascii', got {fmt!r}"
        )


_RUN_MODE_TO_BELLHOP_TYPE = {
    RunMode.COHERENT_TL: 'C',
    RunMode.INCOHERENT_TL: 'I',
    RunMode.SEMICOHERENT_TL: 'S',
    RunMode.RAYS: 'R',
    RunMode.EIGENRAYS: 'E',
    RunMode.ARRIVALS: 'A',
}


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
        Bellhop binary; auto-detected if ``None``.
    prefer_cuda : bool, optional
        Prefer CUDA → cxx → Fortran when ``executable=None``. Default ``True``.
    beam_type : str, optional
        ``B`` Gaussian (default) | ``R`` ray-centered | ``C`` Cartesian |
        ``b`` geometric Gaussian | ``g``/``G`` geometric hat | ``S`` simple Gaussian.
    n_beams : int, optional
        Number of beams; ``0`` lets Bellhop auto-pick. Default ``0``.
    alpha : tuple, optional
        Launch-angle limits ``(min, max)`` in degrees. Default ``(-80, 80)``.
    step : float, optional
        Ray step size (m); ``0`` = auto. Default ``0.0``.
    z_box, r_box : float, optional
        Trace bounding box (m). ``None`` ⇒ ``1.2 ×`` receiver extent.
    source_type : str, optional
        ``'R'`` point/cylindrical (default) | ``'X'`` line/Cartesian.
    grid_type : str, optional
        ``'R'`` rectilinear (default) | ``'I'`` irregular (paired depth/range).
    source_beam_pattern_file : Path or array, optional
        ``.sbp`` path or ``(angle_deg, level_dB)`` array; sets ``RunType(3)='*'``.
    arrivals_format : str, optional
        ``'ascii'`` (default). ``'binary'`` is rejected — uacpy can't parse it.
    beam_width_type : str, optional
        Cerveny / simple-Gaussian only. ``'F'`` filling | ``'M'`` match | ``'W'`` waveguide.
    beam_curvature : str, optional
        Cerveny / simple-Gaussian only. ``'D'`` double | ``'S'`` single | ``'Z'`` zero.
    eps_multiplier, r_loop, n_image, ib_win, component : optional
        Cerveny / simple-Gaussian advanced beam knobs (used when ``beam_type ∈ {C, R, S}``).
        ``r_loop`` is in metres.
    auto_bounce : bool, optional
        Default ``True``. When the env carries layered / RDLB / elastic
        bottoms that Bellhop's fluid ray-tracer can't model accurately,
        ``run(...)`` auto-routes through BOUNCE to derive a ``.brc``
        reflection-coefficient table. Set ``False`` to skip the auto-route
        — Bellhop then collapses the bottom via its own ``collapse={…}``
        policy and runs with fluid-approximated physics, with one
        ``UserWarning``. ``run_with_bounce(...)`` always uses BOUNCE
        regardless of this flag.
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    Defaults auto-derived from inputs (no need to override unless tuning):

    - ``n_beams=0`` → Bellhop auto-picks the beam count.
    - ``step=0.0`` → Bellhop auto-picks the ray step from geometry.
    - ``z_box=None`` → ``1.2 × env.depth``.
    - ``r_box=None`` → ``1.2 × receiver.range_max`` (or 10 km if 0).
    - ``TopOpt`` position 4 reads from ``env.absorption``
      (``Thorp`` → ``'T'``, ``FrancoisGarrison`` → ``'F'`` + params,
      ``Biological`` → ``'B'`` + layers, ``ConstantAbsorption`` /
      ``None`` → ``' '``).
    - Bottom reflection: when ``env.bottom`` is layered / elastic and
      ``auto_bounce=True``, BOUNCE is invoked transparently to derive
      the ``.brc`` reflection coefficient table.

    **Auto-route through BOUNCE.** ``Bellhop.run(...)`` detects
    ``LayeredBottom`` / ``RangeDependentLayeredBottom`` / elastic
    halfspace / ``RangeDependentBottom`` with non-zero ``shear_speed``
    anywhere along range, runs BOUNCE upstream to derive a ``.brc``
    reflection-coefficient table, and re-runs Bellhop against
    ``acoustic_type='file'`` (one ``UserWarning``). The user's
    ``collapse={…}`` dict is forwarded to the spawned Bounce. Use
    :meth:`run_with_bounce` for explicit control over BOUNCE parameters.

    Bellhop uses the global :data:`DEFAULT_COLLAPSE` policy without
    overrides — RD bathymetry / RD bottom / RD-SSP (when the model's
    ``interp_ssp='quad'``) are honoured natively.

    Examples
    --------
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
        interp_ssp: Optional[str] = None,
        interp_bathymetry: str = 'linear',
        interp_altimetry: str = 'linear',
        source_beam_pattern_file: Optional[Path] = None,
        arrivals_format: str = 'ascii',
        beam_width_type: str = 'F',
        beam_curvature: str = 'D',
        eps_multiplier: float = 1.0,
        r_loop: float = 1000.0,
        n_image: int = 1,
        ib_win: int = 4,
        component: str = 'P',
        beam_shift: bool = False,
        # Broadband synthesis knobs (BROADBAND / TIME_SERIES paths)
        n_freqs: int = DEFAULT_BROADBAND_N_FREQS,
        bandwidth_factor: float = DEFAULT_BROADBAND_BANDWIDTH_FACTOR,
        time_window: Optional[float] = None,
        t_start: Optional[float] = None,
        auto_bounce: bool = True,
        use_tmpfs: bool = False,
        verbose: Union[bool, str] = False,
        work_dir: Optional[Path] = None,
        **kwargs,
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
            auto-selection (NBEAMS<=0 in the Fortran reader). Default: 0.
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
        interp_ssp : str, optional
            SSP connection scheme. ``None`` (default) auto-picks
            ``'quad'`` for a range-dependent ``env.ssp`` and ``'linear'``
            otherwise. Explicit values: ``'linear'``, ``'pchip'``,
            ``'cubic'``, ``'quad'``, ``'n2linear'``, ``'analytic'``.
            ``env.ssp.shape='isovelocity'`` always forces ``'C'`` regardless.
        interp_bathymetry : str, optional
            ``.bty`` interpolation. ``'linear'`` (default) or
            ``'curvilinear'``.
        interp_altimetry : str, optional
            ``.ati`` interpolation. ``'linear'`` (default) or
            ``'curvilinear'``.
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
        beam_width_type : {'F', 'M', 'W'}, optional
            Cerveny / simple-Gaussian beam width type. 'F' = filling
            (default), 'M' = match, 'W' = waveguide. Only used when
            ``beam_type`` ∈ ('C', 'R', 'S').
        beam_curvature : {'D', 'S', 'Z'}, optional
            Beam curvature: 'D' = double (default), 'S' = single,
            'Z' = zero.
        eps_multiplier : float, optional
            Beam-width epsilon multiplier. Default: 1.0.
        r_loop : float, optional
            Range (m) at which to choose the beam width. Default: 1000.0.
        n_image : int, optional
            Number of images. Default: 1.
        ib_win : int, optional
            Beam-windowing parameter. Default: 4.
        component : {'P', 'D'}, optional
            Output component for displacement-receiver fields: 'P'
            pressure (default), 'D' displacement.
        auto_bounce : bool, optional
            Default ``True``. When ``env`` carries a ``LayeredBottom`` /
            ``RangeDependentLayeredBottom`` / elastic halfspace /
            ``RangeDependentBottom`` with non-zero shear, ``run(...)``
            auto-routes through BOUNCE to derive a ``.brc`` reflection-
            coefficient table and re-runs Bellhop against
            ``acoustic_type='file'``, attaching the in-memory
            :class:`ReflectionCoefficient` to
            ``result.metadata['bounce_result']``. Set ``False`` to skip
            the auto-route — Bellhop then collapses the bottom via its
            own ``collapse={…}`` policy and runs with fluid-approximated
            physics, with one ``UserWarning``.
            ``run_with_bounce(...)`` always uses BOUNCE regardless.
        """
        super().__init__(
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir, **kwargs,
        )

        # Declare supported modes for Bellhop
        self._supported_modes = [
            RunMode.COHERENT_TL,
            RunMode.INCOHERENT_TL,
            RunMode.SEMICOHERENT_TL,
            RunMode.RAYS,
            RunMode.EIGENRAYS,
            RunMode.ARRIVALS,
            RunMode.BROADBAND,
            RunMode.TIME_SERIES,
        ]
        self._supports_altimetry = True
        self._supports_range_dependent_bathymetry = True
        # RD-SSP honoured when Bellhop(interp_ssp='quad'); other interp
        # values trigger a warning and SSP-collapse in run().
        self._supports_range_dependent_ssp = True
        self._supports_range_dependent_bottom = True
        self._supports_layered_bottom = False
        self._supports_range_dependent_layered_bottom = False
        self._supports_elastic_media = True
        self._supports_multi_source_depth = True

        self.prefer_cuda = prefer_cuda
        self.beam_type = beam_type
        self.n_beams = n_beams
        self.alpha = alpha
        self.step = step
        self.z_box = z_box
        self.r_box = r_box
        self.source_type = source_type
        self.grid_type = grid_type
        self.interp_ssp = interp_ssp
        self.interp_bathymetry = interp_bathymetry
        self.interp_altimetry = interp_altimetry
        self.source_beam_pattern_file = (
            Path(source_beam_pattern_file)
            if isinstance(source_beam_pattern_file, (str, Path))
            else source_beam_pattern_file
        )
        self.beam_width_type = beam_width_type
        self.beam_curvature = beam_curvature
        self.eps_multiplier = float(eps_multiplier)
        self.r_loop = float(r_loop)
        self.n_image = int(n_image)
        self.ib_win = int(ib_win)
        self.component = component
        self.beam_shift = bool(beam_shift)
        self.n_freqs = int(n_freqs)
        self.bandwidth_factor = float(bandwidth_factor)
        # Broadband synthesis window. ``None`` lets ``_run_broadband``
        # auto-derive from the latest arrival + source waveform duration.
        self.time_window = (
            float(time_window) if time_window is not None else None
        )
        self.t_start = (
            float(t_start) if t_start is not None else None
        )
        self.auto_bounce = bool(auto_bounce)
        _validate_arrivals_format(arrivals_format)
        self.arrivals_format = arrivals_format
        self.version = "unknown"

        if executable is None:
            self.executable = self._find_bellhop_executable()
        else:
            self.executable = Path(executable)
            self.version = "custom"

        if not self.executable.exists():
            raise ExecutableNotFoundError("Bellhop", str(self.executable))

        if verbose and self.version != "custom":
            self._log(f"Using Bellhop {self.version}: {self.executable}")

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
        run_mode: Optional[RunMode] = None,
        *,
        frequencies: Optional[np.ndarray] = None,
        source_waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
    ) -> Result:
        """
        Run Bellhop simulation

        Parameters
        ----------
        env, source, receiver : see ``PropagationModel.run``.
        run_mode : RunMode, optional
            Which Bellhop mode to run. One of ``RunMode.COHERENT_TL``,
            ``INCOHERENT_TL``, ``SEMICOHERENT_TL``, ``RAYS``, ``EIGENRAYS``,
            ``ARRIVALS``, ``TIME_SERIES``. Defaults to ``COHERENT_TL``.
        frequencies : ndarray, optional
            Explicit frequency vector (Hz) for ``RunMode.BROADBAND`` /
            ``RunMode.TIME_SERIES``. When ``None`` and no
            ``source_waveform`` is given, the wrapper auto-synthesises
            ``DEFAULT_BROADBAND_N_FREQS`` (128) bins linearly spaced
            over ``[fc*(1 - bw/2), fc*(1 + bw/2)]`` (clipped to [1, ∞))
            with ``bw = DEFAULT_BROADBAND_BANDWIDTH_FACTOR`` (0.5 →
            half-octave band). Pass ``frequencies=`` explicitly to
            override.
        source_waveform : ndarray, optional
            Time-domain source waveform for delay-and-sum synthesis
            (``RunMode.TIME_SERIES``). Requires ``sample_rate``.
        sample_rate : float, optional
            Sample rate (Hz) accompanying ``source_waveform``.

        Returns
        -------
        result : Result
            Simulation results
        """
        # ── Resolve run_mode → internal single-char Bellhop code ────────
        run_mode = self._resolve_run_mode(run_mode)

        if run_mode in (RunMode.TIME_SERIES, RunMode.BROADBAND):
            # Both routes go through the arrivals → H(f) pipeline. Without
            # source_waveform → Field; with it → Field (1×1 grid).
            return self._run_broadband(
                env, source, receiver,
                frequencies=frequencies,
                source_waveform=source_waveform,
                sample_rate=sample_rate,
            )

        # Multi-source-depth EIGENRAYS: ``WriteRay2D`` fires only on
        # receiver hits AND Bellhop's eigenray search reorders ``alpha``
        # for its bracketing heuristic, so the ``.ray`` file has no
        # parseable per-source boundary. Loop in Python for this one
        # mode — TL / RAYS / ARRIVALS all split at the reader level
        # from the single binary call.
        if (
            run_mode == RunMode.EIGENRAYS
            and len(np.atleast_1d(source.depths)) > 1
        ):
            from uacpy.core.results import ResultStack
            slabs = []
            for sd in source.depths:
                single = Source(
                    depths=float(sd),
                    frequencies=source.frequencies,
                    source_type=source.source_type,
                )
                slabs.append(self.run(
                    env, single, receiver, run_mode=run_mode,
                    frequencies=frequencies,
                    source_waveform=source_waveform,
                    sample_rate=sample_rate,
                ))
            return ResultStack(
                slabs=slabs, coordinate=source.depths,
                coordinate_name='source_depth',
            )

        run_type = _RUN_MODE_TO_BELLHOP_TYPE[run_mode]

        _validate_arrivals_format(self.arrivals_format)

        # Auto-route through BOUNCE whenever Bellhop's fluid ray-tracer
        # cannot represent the bottom's full reflection physics natively:
        #   - LayeredBottom / RangeDependentLayeredBottom — Bellhop has
        #     no multi-medium .env format; without BOUNCE the layers are
        #     silently lost.
        #   - BoundaryProperties (or RangeDependentBottom) with non-zero
        #     shear — Bellhop's writer emits cs/alpha_s on the 'A' line
        #     (or per-range on the long .bty), but the ray tracer
        #     approximates the resulting reflection coefficient with
        #     fluid physics; BOUNCE pre-computes the exact elastic RC
        #     including shear-conversion and Bellhop consumes it via the
        #     'F' bottom type.
        # BOUNCE itself is range-independent; the spawned Bounce instance
        # collapses any range-dependent env via its own collapse policy
        # (Bounce defaults: ``bottom='median'``, ``rd_layered_range=
        # 'median'``, ``rd_layered_layers='preserve'`` → median range,
        # layer stack kept since BOUNCE consumes LayeredBottom natively).
        # Pass ``collapse={...}`` to Bellhop to override;
        # ``Bellhop.run_with_bounce(...)`` is the explicit form for
        # users who want to control the BOUNCE constructor.
        from uacpy.core.environment import (
            LayeredBottom, RangeDependentLayeredBottom,
        )
        bp = env.bottom
        is_layered = isinstance(bp, (LayeredBottom, RangeDependentLayeredBottom))
        is_elastic = env.has_elastic_bottom()
        if is_layered or is_elastic:
            tag = ' (elastic)' if is_elastic else ''
            kind = type(bp).__name__ + tag
            if self.auto_bounce:
                warnings.warn(
                    f"{self.model_name}: env.bottom is {kind}; auto-routing "
                    f"through BOUNCE to derive a reflection-coefficient table. "
                    f"BOUNCE is range-independent — Bounce's collapse policy "
                    f"reduces the env (defaults: bottom='median', "
                    f"rd_layered_range='median', rd_layered_layers='preserve'). "
                    f"Pass ``Bellhop(auto_bounce=False)`` to skip the auto-route "
                    f"(Bellhop will then collapse the bottom via its own "
                    f"collapse policy and run with fluid-approximated physics).",
                    UserWarning, stacklevel=2,
                )
                return self.run_with_bounce(
                    env, source, receiver,
                    run_mode=run_mode,
                    frequencies=frequencies,
                    source_waveform=source_waveform,
                    sample_rate=sample_rate,
                )
            # auto_bounce=False: fall through. ``_project_environment``
            # below will collapse the bottom via the user's
            # ``collapse={...}`` policy (default ``layered='halfspace'``,
            # ``rd_layered_layers='halfspace'``, ``elastic='fluid'``).
            warnings.warn(
                f"{self.model_name}: env.bottom is {kind}; auto_bounce=False "
                f"→ collapsing via the model's collapse policy and running "
                f"with fluid ray-tracer physics. Reflection-coefficient "
                f"accuracy near elastic / layered bottoms will be degraded. "
                f"Set auto_bounce=True (default) or call run_with_bounce() "
                f"for the elastic-correct path.",
                UserWarning, stacklevel=2,
            )

        from uacpy.io.oalib_writer import resolve_ssp_interp
        effective_interp = resolve_ssp_interp(env, self.interp_ssp)
        if self.interp_ssp is None:
            self._log(
                f"interp_ssp auto-picked = {effective_interp!r} "
                f"(env.has_range_dependent_ssp={env.has_range_dependent_ssp()})"
            )
        if env.has_range_dependent_ssp() and effective_interp != 'quad':
            method = self._collapse['ssp']
            env = env.copy()
            env.ssp = env.ssp.collapse(method)
            warnings.warn(
                f"Bellhop reads range-dependent SSP only when "
                f"interp_ssp='quad' (external .ssp file). With "
                f"interp_ssp={self.interp_ssp!r} (resolved to "
                f"{effective_interp!r}) the SSP is collapsed to 1-D "
                f"(collapse['ssp']={method!r}). Pass "
                f"``Bellhop(interp_ssp='quad')`` (or leave the default "
                f"``None`` for auto-detection) to enable the 2-D profile.",
                UserWarning, stacklevel=2,
            )

        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)

        # Bellhop fills the r=0 column with the 600 dB "no data" sentinel
        # because no rays have travelled distance zero. Newcomers using
        # ``np.linspace(0, R, N)`` for ``receiver.ranges`` hit a wall of
        # 600 dB values at r=0 and rightly wonder what is wrong. Warn once
        # per ``Bellhop`` instance to nudge them toward a non-zero start.
        if (
            run_mode in (RunMode.COHERENT_TL, RunMode.INCOHERENT_TL,
                         RunMode.SEMICOHERENT_TL)
            and len(receiver.ranges) > 0
            and float(receiver.ranges[0]) == 0.0
            and not getattr(self, '_warned_r0_sentinel', False)
        ):
            warnings.warn(
                f"{self.model_name}: receiver.ranges starts at r=0 m. "
                f"Bellhop fills that column with the 600 dB no-data "
                f"sentinel (no rays have travelled zero distance). "
                f"Start ranges at a small positive value "
                f"(e.g. ``np.linspace(eps, R, N)``) to avoid surprise.",
                UserWarning, stacklevel=2,
            )
            self._warned_r0_sentinel = True

        # Irregular receiver grid ('I' in RunType position 5) requires the
        # receiver.depths and receiver.ranges arrays to have the same
        # length (they are paired point-by-point).  Rectilinear ('R')
        # takes the Cartesian product.  Catch the mismatch here so users
        # see a clear error instead of a confusing Bellhop .prt message.
        if (
            self.grid_type is not None
            and str(self.grid_type).upper() == 'I'
            and len(receiver.depths) != len(receiver.ranges)
        ):
            raise ConfigurationError(
                f"Bellhop grid_type='I' (irregular) requires "
                f"len(receiver.depths) == len(receiver.ranges); got "
                f"{len(receiver.depths)} depths and "
                f"{len(receiver.ranges)} ranges. Use grid_type='R' for "
                f"a rectilinear (Cartesian-product) grid, or rebuild the "
                f"Receiver with matched arrays."
            )
        fm = self._setup_file_manager()
        self.file_manager = fm

        extra_writer_kwargs = {
            'interp_ssp': self.interp_ssp,
            'interp_bathymetry': self.interp_bathymetry,
            'interp_altimetry': self.interp_altimetry,
        }

        sbp_spec = self.source_beam_pattern_file

        try:
            base_name = 'model'
            env_file = fm.get_path(f'{base_name}.env')
            self._log(f"Writing environment file: {env_file}")

            # Stage source beam pattern file. Bellhop reads by base name
            # (<base>.sbp) when RunType position 3 is '*'.
            use_sbp = False
            if sbp_spec is not None:
                sbp_dest = env_file.with_suffix('.sbp')
                if isinstance(sbp_spec, (str, Path)):
                    src = Path(sbp_spec)
                    if not src.exists():
                        raise ConfigurationError(
                            f"Source beam pattern file not found: {src}"
                        )
                    shutil.copy(src, sbp_dest)
                else:
                    # Array-like: expect shape (N, 2) [angle_deg, level_dB]
                    from uacpy.io.refl_io import write_source_beam_pattern
                    arr = np.asarray(sbp_spec, dtype=float)
                    if arr.ndim != 2 or arr.shape[1] != 2:
                        raise ConfigurationError(
                            "source_beam_pattern_file array must be shape "
                            "(N, 2): [angle_deg, level_dB]."
                        )
                    write_source_beam_pattern(
                        sbp_dest, arr[:, 0], arr[:, 1]
                    )
                use_sbp = True
                self._log(f"Wrote source beam pattern: {sbp_dest}")

            write_bellhop_env_file(
                filepath=env_file,
                env=env,
                source=source,
                receiver=receiver,
                run_type=run_type,
                beam_type=self.beam_type,
                source_type=self.source_type,
                grid_type=self.grid_type,
                verbose=self.verbose,
                n_beams=self.n_beams,
                alpha=self.alpha,
                step=self.step,
                z_box=self.z_box,
                r_box=self.r_box,
                source_beam_pattern=use_sbp,
                beam_width_type=self.beam_width_type,
                beam_curvature=self.beam_curvature,
                eps_multiplier=self.eps_multiplier,
                r_loop=self.r_loop,
                n_image=self.n_image,
                ib_win=self.ib_win,
                component=self.component,
                beam_shift=self.beam_shift,
                **extra_writer_kwargs,
            )

            # Run Bellhop
            self._log("Running Bellhop...")
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
                raise ConfigurationError(f"Unknown run_type: {run_type}")

            if not output_file.exists():
                exc = ModelExecutionError(
                    self.model_name, return_code=0, stdout=None,
                    stderr=(
                        f"Bellhop did not produce {output_file}; "
                        f"check {output_file.with_suffix('.prt')} for diagnostics."
                    ),
                )
                self._attach_prt_tail(exc, fm.work_dir, base_name)
                raise exc
            result = reader(output_file)

            # Bellhop fills the r=0 column with zero pressure ⇒ TL=600 dB
            # (the AT "no data" sentinel: no rays have travelled distance
            # zero). Replace that column with NaN so downstream numerics
            # (.tl.min(), np.nanmean, plots) ignore it rather than read
            # 600 dB as a real value. Honest shadow zones elsewhere
            # keep the 600 dB marker.
            if rt in ('C', 'I', 'S'):
                from uacpy.core.results import ResultStack
                if isinstance(result, ResultStack):
                    pf_slabs = result.slabs
                else:
                    pf_slabs = [result]
                for slab in pf_slabs:
                    if (
                        getattr(slab, 'ranges', None) is not None
                        and slab.ranges.size > 0
                        and float(slab.ranges[0]) == 0.0
                    ):
                        slab.data[:, 0] = np.nan

            from uacpy.core.results import ResultStack

            # The .ray header records only NSz (count), not Pos%Sz; the
            # reader returns the stack with a placeholder coordinate.
            # Replace it with the real source.depths order (Bellhop's
            # SourceDepth loop iterates Pos%Sz in writer order).
            if isinstance(result, ResultStack):
                real_sds = np.atleast_1d(np.asarray(source.depths, dtype=float))
                if real_sds.size == result.n_slabs:
                    result.coordinate = real_sds

            if rt in ('R', 'E'):
                # The .ray file format is identical for fan and
                # eigenray runs; only the wrapper knows which one
                # produced it. Same goes for the receiver geometry.
                rcv_d = np.atleast_1d(np.asarray(receiver.depths, dtype=float))
                rcv_r = np.atleast_1d(np.asarray(receiver.ranges, dtype=float))
                ray_slabs = (
                    result.slabs if isinstance(result, ResultStack) else [result]
                )
                for slab in ray_slabs:
                    slab.is_eigen = (rt == 'E')
                    slab.receiver_depths = rcv_d
                    slab.receiver_ranges = rcv_r

            f0 = np.atleast_1d(np.asarray(
                float(np.atleast_1d(source.frequencies)[0]), dtype=float,
            ))
            slabs_to_set = (
                result.slabs if isinstance(result, ResultStack) else [result]
            )
            for i, slab in enumerate(slabs_to_set):
                slab.model = self.model_name
                slab.backend = self.model_name.lower()
                if isinstance(result, ResultStack):
                    slab.source_depths = np.array(
                        [float(result.coordinate[i])], dtype=float,
                    )
                else:
                    slab.source_depths = np.atleast_1d(np.asarray(
                        source.depths, dtype=float,
                    ))
                slab.frequencies = f0
                slab.phase_reference = 'travelling_wave'
                self._attach_output_paths(
                    slab, fm.work_dir, base_name,
                    primary_files=(
                        ('shd_file', '.shd'),
                        ('arr_file', '.arr'),
                        ('ray_file', '.ray'),
                    ),
                )

            self._log("Simulation complete")
            return result

        finally:
            if fm.cleanup:
                fm.cleanup_work_dir()

    def run_with_bounce(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        c_low: float = DEFAULT_C_MIN,
        c_high: float = DEFAULT_C_MAX,
        rmax: float = 10000.0,
        run_mode: Optional[RunMode] = None,
        frequencies: Optional[np.ndarray] = None,
        source_waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
    ) -> Result:
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
        c_low : float, optional
            Minimum phase velocity for reflection table (m/s). Default 1400.
        c_high : float, optional
            Maximum phase velocity for reflection table (m/s). Default 10000.
        rmax : float, optional
            Maximum range for angular resolution (m). Default 10000.

        Returns
        -------
        result : Result
            Bellhop simulation results using reflection coefficients
        """
        from uacpy.models.bounce import Bounce
        import tempfile

        self._log("Running BOUNCE to compute reflection coefficients...")
        bounce_work_dir = Path(tempfile.mkdtemp(prefix='bellhop_bounce_'))
        bounce = Bounce(
            verbose=self.verbose,
            c_low=c_low,
            c_high=c_high,
            rmax=rmax,
            collapse=dict(self._user_collapse) or None,
            work_dir=bounce_work_dir,
            cleanup=False,            # we own bounce_work_dir; cleaned up below
        )
        try:
            bounce_result = bounce.run(env, source, receiver)

            brc_file = bounce_result.metadata.get('brc_file')
            if not brc_file:
                raise ModelExecutionError(
                    "Bounce", return_code=-1, stdout=None,
                    stderr="BOUNCE did not produce a .brc file",
                )

            env_bounce = copy.deepcopy(env)
            env_bounce.bottom = BoundaryProperties(
                acoustic_type='file',
                reflection_file=brc_file,
            )

            self._log("Running Bellhop with BOUNCE reflection coefficients...")
            result = self.run(
                env_bounce, source, receiver,
                run_mode=run_mode,
                frequencies=frequencies,
                source_waveform=source_waveform,
                sample_rate=sample_rate,
            )

            # Strip the about-to-be-invalid file paths (work dir is wiped
            # in the finally block below) and attach the in-memory bounce
            # result so the user can plot R(θ) / inspect the BRC without
            # re-running BOUNCE.
            bounce_result.metadata.pop('brc_file', None)
            bounce_result.metadata.pop('irc_file', None)
            result.metadata['bounce_result'] = bounce_result
            return result
        finally:
            shutil.rmtree(bounce_work_dir, ignore_errors=True)

    def _run_broadband(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        frequencies: Optional[np.ndarray] = None,
        source_waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
    ) -> Result:
        """
        Run Bellhop in broadband mode to produce a time-series or
        transfer function result.

        Uses the ray/beam approach described in the Bellhop User Guide (Sec 9):
        run arrivals at center frequency, then either:

        1. **Without source_waveform**: build frequency-domain transfer function
           H(f) from arrivals → returns ``Field``. Use
           ``Field.to_time_trace()`` (raw IFFT) or
           ``Field.synthesize_time_series(source_waveform, sample_rate)``
           (windowed convolution) downstream.

        2. **With source_waveform**: perform delay-and-sum convolution of the
           time-domain waveform with each arrival (amplitude, phase, delay)
           → returns :class:`Field` directly.

        This is a key advantage of ray tracing: the ray geometry (paths, travel
        times) is frequency-independent, so a single arrivals calculation at fc
        provides the impulse response.

        Parameters
        ----------
        env : Environment
            Ocean environment
        source : Source
            Acoustic source (source.frequencies is the center frequency)
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

        Constructor-controlled (``Bellhop(time_window=…, t_start=…,
        n_freqs=…, bandwidth_factor=…)``):
            ``time_window`` (s), ``t_start`` (s), ``n_freqs`` (TF bin
            count, default ``DEFAULT_BROADBAND_N_FREQS=128``),
            ``bandwidth_factor`` (fractional band around fc, default
            ``DEFAULT_BROADBAND_BANDWIDTH_FACTOR=0.5``).

        Returns
        -------
        result : Result
            If source_waveform is None: ``Field``
                (call ``.to_time_trace()`` or
                ``.synthesize_time_series(...)`` to get a time series).
            If source_waveform is provided: ``Field``
                with data shape (n_depths, n_ranges, n_samples) and
                metadata containing 'time', 'dt', 'fs'.
        """
        if len(np.atleast_1d(source.depths)) > 1:
            raise ConfigurationError(
                f"Bellhop broadband synthesis (BROADBAND / TIME_SERIES) "
                f"runs at a single source depth; got "
                f"{len(source.depths)}: {list(source.depths)}. Loop in "
                f"Python over Source(depths=z, ...) and stack the "
                f"results, or pick one depth for this run."
            )
        fc = float(np.atleast_1d(source.frequencies)[0])

        # Step 1: Run Bellhop in arrivals mode
        self._log("Running in arrivals mode (broadband path)...")
        arr_field = self.run(env, source, receiver, run_mode=RunMode.ARRIVALS)

        arrivals_by_rcv = arr_field.by_receiver
        rz = arr_field.receiver_depths
        rr = arr_field.receiver_ranges  # in meters

        nrd = len(rz)
        nrr = len(rr)

        # ── Path A: time-domain delay-and-sum with source waveform ──
        if source_waveform is not None:

            if sample_rate is None:
                raise ConfigurationError(
                    "sample_rate is required when source_waveform is provided"
                )

            self._log(f"Delay-and-sum over {nrd}×{nrr} receiver grid")

            # Lock the time window using the first cell so all traces share
            # a clock; reuse it for every subsequent cell via ``t_start``.
            rts0, t_vec = delayandsum(
                rcv_arrivals=arrivals_by_rcv[0][0][0],
                source_timeseries=source_waveform,
                sample_rate=sample_rate,
                fc=fc,
                time_window=self.time_window,
                t_start=self.t_start,
            )
            t_start_locked = float(t_vec[0])
            time_window_locked = float(t_vec[-1] - t_vec[0]) + 1.0 / sample_rate
            n_t = len(rts0)

            data = np.zeros((nrd, nrr, n_t), dtype=float)
            data[0, 0, :] = rts0
            for ird in range(nrd):
                for irr in range(nrr):
                    if ird == 0 and irr == 0:
                        continue
                    rts, _ = delayandsum(
                        rcv_arrivals=arrivals_by_rcv[0][ird][irr],
                        source_timeseries=source_waveform,
                        sample_rate=sample_rate,
                        fc=fc,
                        time_window=time_window_locked,
                        t_start=t_start_locked,
                    )
                    # delayandsum may return a slightly different length on
                    # cells with no arrivals — pad/truncate to n_t.
                    m = min(len(rts), n_t)
                    data[ird, irr, :m] = np.asarray(rts[:m], dtype=float)

            return Field(
                data=data,
                coords={
                    'depth': np.asarray(rz, dtype=float),
                    'range': np.asarray(rr, dtype=float),
                    'time': t_vec,
                },
                **self._result_kwargs(
                    source, backend='bellhop', frequencies=fc,
                    dt=1.0 / sample_rate, fs=sample_rate, nt=n_t,
                    t_start=t_start_locked, center_frequency=fc,
                ),
            )

        # ── Path B: frequency-domain transfer function ──
        if frequencies is None:
            half_bw = 0.5 * self.bandwidth_factor
            f_min = max(1.0, fc * (1.0 - half_bw))
            f_max = fc * (1.0 + half_bw)
            frequencies = np.linspace(f_min, f_max, self.n_freqs)

        frequencies = np.asarray(frequencies, dtype=float)
        n_freq = len(frequencies)

        # Build H(d, r, f) for each (receiver_depth, receiver_range).
        # Use first source depth (most common case). Trailing-axis convention.
        H = np.zeros((nrd, nrr, n_freq), dtype=complex)
        for ird in range(nrd):
            for irr in range(nrr):
                rcv_arr = arrivals_by_rcv[0][ird][irr]
                H[ird, irr, :] = self._arrivals_to_tf(
                    rcv_arr, frequencies, fc
                )

        self._log(f"Built transfer function "
                  f"({nrd} depths x {n_freq} freqs x {nrr} ranges)")

        c0 = float(env.ssp.data[0, 0])

        return Field(
            data=H,
            coords={
                'depth': np.asarray(rz, dtype=float),
                'range': np.asarray(rr, dtype=float),
                'frequency': frequencies,
            },
            phase_reference='travelling_wave',
            **self._result_kwargs(
                source,
                backend='bellhop',
                frequencies=frequencies,
                center_frequency=fc,
                arrivals_field=arr_field,
                c0=c0,
            ),
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
            delays_imag, n_arrivals.
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
        delays_imag = rcv_arrivals['delays_imag']

        phases_rad = np.deg2rad(phases_deg)
        omega = 2.0 * np.pi * frequencies  # (n_freq,)

        H = np.zeros(len(frequencies), dtype=complex)

        for ia in range(n_arr):
            A_complex = amps[ia] * np.exp(1j * phases_rad[ia])
            # exp(-i*omega*tau) with tau = Re(tau) + i*Im(tau) gives a
            # phase-shift and an exp(omega*Im(tau)) attenuation. Im(tau)
            # is in seconds; omega is the per-frequency carrier.
            phase_shift = np.exp(-1j * omega * delays[ia])
            atten = np.exp(omega * delays_imag[ia])
            H += A_complex * atten * phase_shift

        return H

    def _build_command(self, base_name: str) -> list:
        """Build the argv used to launch the binary.

        Subclasses (BellhopCUDA) may override this to add flags.
        """
        if self.version in ('cuda', 'cxx'):
            # bellhopcxx/cuda require a dimensionality flag.
            dim = getattr(self, 'dimensionality', '2D')
            return [str(self.executable), f'--{dim}', base_name]
        return [str(self.executable), base_name]

    def _run_bellhop(self, base_name: str, work_dir: Path):
        """Execute the Bellhop binary via the shared subprocess runner.

        Bellhop reports most fatal errors in ``<base>.prt`` rather than on
        stderr. If the child exits non-zero, we append the tail of the .prt
        file (up to 2000 chars) to the raised ``ModelExecutionError`` so the
        diagnostic surface to the user instead of a blank stderr.
        """
        cmd = self._build_command(base_name)
        try:
            result = self._run_subprocess(cmd, cwd=work_dir)
        except ModelExecutionError as exc:
            self._attach_prt_tail(exc, work_dir, base_name)
            raise

        if self.verbose and result.stdout:
            self._log(f"Bellhop output:\n{result.stdout}", level='debug')


class BellhopCUDA(Bellhop):
    """
    BellhopCUDA - C++/CUDA ray tracing model (thin ``Bellhop`` subclass).

    Shares all Environment/Source/Receiver plumbing, broadband synthesis,
    and output parsing with the parent. Only the executable selection and
    the ``--<dim>`` invocation flag differ.

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
        """Override parent: pick CUDA or CXX flavor, never Fortran.

        With ``use_gpu=True`` we look for bellhopcuda first and fall back to
        bellhopcxx so a CPU-only install (where install.sh built only the C++
        variant) still satisfies BellhopCUDA — install.sh's `--bellhop cxx`
        path produces just bellhopcxx, and forcing the user to instantiate
        BellhopCUDA(use_gpu=False) on every CPU box would break the
        "auto-picks whichever install.sh built" contract.
        """
        names = ['bellhopcuda', 'bellhopcxx'] if self.use_gpu else ['bellhopcxx']
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
