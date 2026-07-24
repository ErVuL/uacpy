"""
Bellhop ray tracing model (Fortran / C++ / CUDA backends)

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
from typing import Dict, Optional, Tuple, Union

from scipy.signal import hilbert

from uacpy.models.base import PropagationModel, RunMode, ModelSpec
from uacpy.core.environment import Environment, BoundaryProperties, Bottom
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.results import Result, Field, ResultStack
from uacpy.core.constants import (
    DEFAULT_C_MIN, DEFAULT_C_MAX,
    DEFAULT_BROADBAND_N_FREQS, DEFAULT_BROADBAND_BANDWIDTH_FACTOR,
)
from uacpy.core.exceptions import (
    ConfigurationError, ExecutableNotFoundError, ModelExecutionError,
)
from uacpy.io.bellhop_writer import write_bellhop_env_file
from uacpy.io.file_manager import FileManager
from uacpy.io.oalib_reader import read_shd_file, read_arr_file, read_ray_file


# Beam-type letters honoured by the Bellhop env reader (case-significant).
# Anything else maps to the geometric-hat DEFAULT case in
# ReadEnvironmentBell.f90, so the constructor rejects it rather than letting a
# typo silently change the beam model.
_VALID_BEAM_TYPES = frozenset({'B', 'R', 'C', 'b', 'g', 'G', 'S'})


# ---------------------------------------------------------------------------
# Bellhop-specific signal processing helpers
# ---------------------------------------------------------------------------

# Length (s) of the all-zero trace returned when a receiver has no arrivals and
# no explicit time_window — just enough to give a usable, non-empty time axis.
_EMPTY_TRACE_SECONDS = 0.1


def delayandsum(
    rcv_arrivals: dict,
    source_timeseries: np.ndarray,
    sample_rate: float,
    fc: float,
    time_window: Optional[float] = None,
    t_start: Optional[float] = None,
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
            nrts = int(_EMPTY_TRACE_SECONDS * sample_rate)
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

        # Add this arrival's shifted, scaled copy of the source signal as a
        # single clipped slice-add (vectorised over the source samples).
        contrib = scaled_amp * np.real(sts_analytic * phase_factor)
        lo = max(0, i_start)
        hi = min(nrts, i_start + nsts)
        if lo < hi:
            rts[lo:hi] += contrib[lo - i_start:hi - i_start]

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
    backend : str, optional
        Force a binary variant: ``'fortran'`` (bellhop), ``'cxx'``
        (bellhopcxx), or ``'cuda'`` (bellhopcuda). ``None`` (default)
        auto-selects CUDA → cxx → Fortran among whatever ``install.sh``
        built. If an explicitly requested variant isn't installed, Bellhop
        falls back to the Fortran binary with a ``UserWarning``. Mirrors
        ``RAM(backend=...)``.
    dimensionality : str, optional
        Only ``'2D'`` (default) is supported — it is the ``--2D`` flag the
        bellhopcxx / bellhopcuda CLIs require (the Fortran binary ignores it).
        ``'3D'`` is rejected: the env writer cannot emit a 3D-format input
        file, so a 3D flag would mis-drive the binary.
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
        Ray-trace bounding box (m); rays are dropped once they leave it.
        ``None`` ⇒ ``1.2 ×`` the receiver extent (``z_box = 1.2 × env.depth``,
        ``r_box = 1.2 × range_max``, or 10 km when the receiver range is 0).
        Box%r is a horizontal-range cut-off, so the 1.2× pad already
        captures arrivals at the outer receivers; do not enlarge it past a
        range-dependent SSP's defined extent.
    source_type : str, optional
        ``'R'`` point/cylindrical (default) | ``'X'`` line/Cartesian.
    grid_type : str, optional
        ``'R'`` rectilinear (default) | ``'I'`` irregular (paired depth/range).
    source_beam_pattern_file : Path or array, optional
        ``.sbp`` path or ``(angle_deg, level_dB)`` array; sets ``RunType(3)='*'``.
    arrivals_format : str, optional
        ``'ascii'`` (default). ``'binary'`` is rejected — uacpy can't parse it.
    beam_width_type : str, optional
        Cerveny only. ``'F'`` filling | ``'M'`` match | ``'W'`` waveguide.
    beam_curvature : str, optional
        Cerveny only. ``'D'`` double | ``'S'`` single | ``'Z'`` zero.
    eps_multiplier, r_loop, n_image, ib_win, component : optional
        Cerveny advanced beam knobs (used when ``beam_type ∈ {C, R}``).
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

    **Auto-route through BOUNCE.** ``Bellhop.run(...)`` detects a layered
    or elastic ``Bottom`` (layered columns, an elastic halfspace, or
    non-zero ``shear_speed`` anywhere along range), runs BOUNCE upstream
    to derive a ``.brc``
    reflection-coefficient table, and re-runs Bellhop against
    ``acoustic_type='file'`` (one ``UserWarning``). The user's
    ``collapse={…}`` dict is forwarded to the spawned Bounce. Use
    :meth:`run_with_bounce` for explicit control over BOUNCE parameters.

    Bellhop uses the global :data:`DEFAULT_COLLAPSE` policy without
    overrides — RD bathymetry / RD bottom / RD-SSP (when the model's
    ``interp_ssp='quad'``) are honoured natively.

    **Broadband amplitude approximation (BROADBAND / TIME_SERIES).** The
    arrival set is computed by a *single* ray trace at the band centre
    ``fc`` and reused across the synthesised band
    ``[fc(1-bw/2), fc(1+bw/2)]`` (``bw = bandwidth_factor``). The travel
    time ``τ`` and volume-attenuation ``Im(τ)`` are applied exactly per
    frequency (``exp(-i2πf·τ)``), so the *timing* and spreading of every
    arrival are correct at all frequencies. What is held frequency-flat is
    the geometric beam **amplitude and caustic phase**: a Gaussian beam's
    half-width scales as ``√(c/f)``, so the amplitude is only first-order
    correct near ``fc`` and the error grows toward the band edges, largest
    at caustics and in tight ducts. Keep ``bandwidth_factor`` ≲ 1 (±50 %
    of ``fc``) for amplitude-faithful results; for wider bands run several
    sub-bands at different ``fc`` and stitch them (Bellhop User Guide §9).
    A ``UserWarning`` fires when ``bandwidth_factor > 1``.

    Examples
    --------
    >>> bellhop = Bellhop()
    >>> result = bellhop.run(env, source, receiver, run_mode=RunMode.COHERENT_TL)

    Force a specific backend:

    >>> bellhop = Bellhop(backend='fortran')
    >>> bellhop_gpu = Bellhop(backend='cuda')
    """

    # Declarative metadata (see PropagationModel / ModelSpec). Bellhop is the
    # ray engine: honours altimetry, range-dependent bathymetry/bottom,
    # elastic media and a native multi-source-depth grid. No collapse
    # override — uses the base DEFAULT_COLLAPSE unchanged. ``layered_bottom``
    # is False (ray model takes a single half-space per column).
    # ``range_dependent_ssp`` is *instance-dependent* (honoured only on the
    # 'quad' interp), so it is set in __init__ below, not here.
    spec = ModelSpec(
        modes=(
            RunMode.COHERENT_TL, RunMode.INCOHERENT_TL, RunMode.SEMICOHERENT_TL,
            RunMode.RAYS, RunMode.EIGENRAYS, RunMode.ARRIVALS,
            RunMode.BROADBAND, RunMode.TIME_SERIES,
        ),
        supports={
            'altimetry',
            'range_dependent_bathymetry',
            'range_dependent_bottom',
            'elastic_media',
            'multi_source_depth',
        },
    )
    source = 'acoustics_toolbox'

    def __init__(
        self,
        executable: Optional[Path] = None,
        backend: Optional[str] = None,
        dimensionality: str = '2D',
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
        cleanup: Optional[bool] = None,
        timeout: float = 600.0,
        collapse: Optional[Dict[str, str]] = None,
    ):
        """
        Parameters
        ----------
        executable : Path, optional
            Path to bellhop executable. Auto-detected if None.
        backend : str, optional
            Force a binary variant: ``'fortran'`` (bellhop), ``'cxx'``
            (bellhopcxx), or ``'cuda'`` (bellhopcuda). ``None`` (default)
            auto-selects, preferring CUDA > C++ > Fortran among whatever
            ``install.sh`` built. If an explicitly requested variant isn't
            installed, Bellhop falls back to the Fortran binary with a
            ``UserWarning``. Mirrors ``RAM(backend=...)``.
        dimensionality : str, optional
            Only ``'2D'`` (default) is supported — the ``--2D`` flag the
            bellhopcxx / bellhopcuda CLIs require (ignored by the Fortran
            binary, which has no such flag). ``'3D'`` raises: the env writer
            produces 2D-format input only.
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
            Cerveny beam width type. 'F' = filling
            (default), 'M' = match, 'W' = waveguide. Only used when
            ``beam_type`` ∈ ('C', 'R').
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
        beam_shift : bool, optional
            When True, sets RunType position 7 to 'S' enabling beam-shift
            on boundary reflections. Default: False.
        n_freqs : int, optional
            Number of frequency bins for BROADBAND / TIME_SERIES
            synthesis when the band is expanded from a single centre
            frequency. Default: :data:`DEFAULT_BROADBAND_N_FREQS`.
        bandwidth_factor : float, optional
            Fractional bandwidth of the synthesised band
            ``[fc·(1-bw/2), fc·(1+bw/2)]`` around a single centre
            frequency. Default:
            :data:`DEFAULT_BROADBAND_BANDWIDTH_FACTOR`.
        time_window : float, optional
            TIME_SERIES output window length (s). ``None``
            auto-derives from the latest arrival plus the source
            waveform duration; ``run(output_duration=…)`` overrides
            per call.
        t_start : float, optional
            TIME_SERIES output start time (s). ``None`` auto-derives
            from the earliest arrival.
        auto_bounce : bool, optional
            Default ``True``. When ``env`` carries a layered or elastic
            ``Bottom`` (layered columns, an elastic halfspace, or non-zero
            shear anywhere along range), ``run(...)``
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
            use_tmpfs=use_tmpfs, verbose=verbose, work_dir=work_dir,
            cleanup=cleanup, timeout=timeout, collapse=collapse,
        )

        # Run modes, capability flags and collapse defaults come from the
        # class-level ``spec`` (applied by PropagationModel.__init__).
        #
        # Instance-dependent override: RD-SSP is honoured natively only via the
        # external 2-D .ssp file, which Bellhop reaches on the 'quad' interp.
        # ``interp_ssp=None`` auto-picks 'quad' for a range-dependent SSP
        # (oalib_writer.resolve_ssp_interp), so the default path honours it. A
        # user who *pins* a non-quad interp ('linear', 'cubic', …) gets a 1-D
        # collapse — so the flag must be False for that instance, and
        # ``_project_environment`` does the collapse with the standard
        # one-warning-per-feature. Keeping the flag honest means the advertised
        # capability matches the run-time behaviour.
        self._supports_range_dependent_ssp = (
            interp_ssp is None or str(interp_ssp).lower() == 'quad'
        )

        if beam_type not in _VALID_BEAM_TYPES:
            raise ConfigurationError(
                f"Bellhop(beam_type={beam_type!r}) is not a known beam type. "
                f"Choose one of {sorted(_VALID_BEAM_TYPES)} "
                f"(case-significant; Bellhop would otherwise silently fall "
                f"back to a geometric-hat beam)."
            )
        if source_type not in ('R', 'X'):
            raise ConfigurationError(
                f"Bellhop(source_type={source_type!r}) is not valid. Use "
                f"'R' (point/cylindrical) or 'X' (line/Cartesian)."
            )
        if grid_type not in ('R', 'I'):
            raise ConfigurationError(
                f"Bellhop(grid_type={grid_type!r}) is not valid. Use "
                f"'R' (rectilinear) or 'I' (irregular paired depth/range)."
            )
        if not isinstance(alpha, (tuple, list, np.ndarray)) or len(alpha) != 2:
            raise ConfigurationError(
                f"Bellhop(alpha={alpha!r}) must be a 2-element sequence "
                f"(min_deg, max_deg) of launch-angle limits."
            )
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
        self._warn_on_ignored_cerveny_knobs()
        self.n_freqs = int(n_freqs)
        self.bandwidth_factor = float(bandwidth_factor)
        if self.bandwidth_factor > 1.0:
            warnings.warn(
                f"Bellhop: bandwidth_factor={self.bandwidth_factor} spans "
                f">±50% of fc; arrival amplitudes are computed at fc and held "
                f"frequency-flat, so broadband amplitudes degrade toward the "
                f"band edges (worst at caustics/ducts). Run sub-bands at "
                f"different fc for wide bandwidths (Bellhop User Guide §9).",
                UserWarning, stacklevel=2,
            )
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
        if backend is not None and backend not in ('fortran', 'cxx', 'cuda'):
            raise ConfigurationError(
                f"Bellhop(backend={backend!r}) is not a known backend. "
                f"Choose 'fortran', 'cxx', 'cuda', or None for automatic "
                f"selection (CUDA > C++ > Fortran)."
            )
        self.backend = backend
        if dimensionality != '2D':
            raise ConfigurationError(
                f"Bellhop(dimensionality={dimensionality!r}) is not supported. "
                f"Only '2D' is available: the env writer produces 2D-format "
                f"input files, so a '3D' (or any other) flag would mis-drive "
                f"the binary (silent 2D run on Fortran, abort on cxx/cuda)."
            )
        self.dimensionality = dimensionality
        self.version = "unknown"

        # Keep the user's ``executable`` arg verbatim (``None`` when
        # auto-detected) so ``model.copy()`` re-resolves the binary honoring
        # ``backend=`` instead of re-pinning the already-resolved path (which
        # would flip ``version`` to 'custom' and drop the cxx/cuda ``--<dim>``
        # flag). The resolved path lives in ``self._exe``.
        self.executable = Path(executable) if executable is not None else None
        if self.executable is None:
            self._exe = self._find_bellhop_executable()
        else:
            self._exe = self.executable
            self.version = "custom"

        if not self._exe.exists():
            raise ExecutableNotFoundError("Bellhop", str(self._exe))

        if self.version != "custom":
            self._log(f"Using Bellhop {self.version}: {self._exe}")

    def _warn_on_ignored_cerveny_knobs(self) -> None:
        """The Cerveny beam knobs are written only for ``beam_type`` in
        {'C', 'R'} (ReadEnvironmentBell.f90). Warn if any is set to a
        non-default value while a non-Cerveny beam is selected, since it
        would otherwise be silently ignored."""
        if self.beam_type.upper() in ('C', 'R'):
            return
        defaults = {
            'beam_width_type': 'F', 'beam_curvature': 'D',
            'eps_multiplier': 1.0, 'r_loop': 1000.0,
            'n_image': 1, 'ib_win': 4, 'component': 'P',
        }
        ignored = [name for name, default in defaults.items()
                   if getattr(self, name) != default]
        if ignored:
            warnings.warn(
                f"Bellhop: Cerveny beam knobs {', '.join(ignored)} are "
                f"ignored for beam_type={self.beam_type!r} (they apply only "
                f"to Cerveny beams, beam_type 'C' or 'R').",
                UserWarning, stacklevel=3,
            )

    def _find_bellhop_executable(self) -> Path:
        """Locate the Bellhop binary, keyed on ``self.backend``.

        ``None`` auto-selects in preference order CUDA > C++ > Fortran among
        whatever ``install.sh`` built. An explicitly requested variant is
        tried first; if it isn't installed the search falls back to the
        Fortran binary and emits a ``UserWarning``. ``self.version`` is
        inferred from the name of the returned path.
        """
        backend_names = {
            'fortran': ['bellhop'],
            'cxx': ['bellhopcxx'],
            'cuda': ['bellhopcuda'],
        }
        if self.backend in backend_names:
            # Requested variant first, then Fortran as a graceful fallback.
            names = backend_names[self.backend] + ['bellhop']
        else:
            names = ['bellhopcuda', 'bellhopcxx', 'bellhop']

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

        if self.backend is not None and self.version != self.backend:
            warnings.warn(
                f"Bellhop(backend={self.backend!r}): the {self.backend} binary "
                f"was not found; falling back to the {self.version} binary.",
                UserWarning, stacklevel=2,
            )
        return path

    def _run_eigenrays_multi_depth(self, env, source, receiver, run_mode,
                                   frequencies, source_waveform, sample_rate,
                                   output_duration):
        """EIGENRAYS with multiple source depths: Bellhop's eigenray search
        reorders ``alpha`` and ``WriteRay2D`` leaves no per-source boundary in
        the ``.ray`` file, so loop one run per source depth in Python and stack."""
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
                output_duration=output_duration,
            ))
        return ResultStack(
            slabs=slabs, coordinate=source.depths,
            coordinate_name='source_depth',
        )

    def _maybe_route_through_bounce(self, env, source, receiver, run_mode,
                                    frequencies, source_waveform, sample_rate):
        """Layered/elastic bottoms can't be represented by Bellhop's fluid ray
        tracer natively. With ``auto_bounce`` (default) route through BOUNCE for
        an exact reflection-coefficient table and return that Result; otherwise
        warn that the reflection will be fluid-approximated and return ``None``
        to continue the native run."""
        is_layered = env.bottom.is_layered
        is_elastic = env.has_elastic_bottom()
        if not (is_layered or is_elastic):
            return None
        tag = ' (elastic)' if is_elastic else ''
        kind = ('layered bottom' if is_layered else 'bottom') + tag
        if self.auto_bounce:
            warnings.warn(
                f"{self.model_name}: env.bottom is {kind}; auto-routing "
                f"through BOUNCE to derive a reflection-coefficient table. "
                f"BOUNCE is range-independent — Bounce's collapse policy "
                f"reduces the env (default: bottom_range='median', layer "
                f"stack kept). "
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
        # auto_bounce=False: fall through. A LAYERED bottom is collapsed to
        # a halfspace by ``_project_environment`` (collapse policy). A pure
        # ELASTIC halfspace is NOT collapsed — Bellhop sets
        # ``_supports_elastic_media=True``, so the writer emits cs/alpha_s and
        # the ray tracer fluid-approximates the elastic reflection internally.
        if is_layered:
            detail = (
                "collapsing the layered bottom to a halfspace via the "
                "model's collapse policy and running with fluid ray-tracer "
                "physics"
            )
        else:
            detail = (
                "writing the elastic halfspace directly (no collapse — "
                "Bellhop supports elastic media); its reflection coefficient "
                "is fluid-approximated by the ray tracer"
            )
        warnings.warn(
            f"{self.model_name}: env.bottom is {kind}; auto_bounce=False → "
            f"{detail}. Reflection-coefficient accuracy near elastic / "
            f"layered bottoms will be degraded. Set auto_bounce=True "
            f"(default) or call run_with_bounce() for the elastic-correct "
            f"path.",
            UserWarning, stacklevel=2,
        )
        return None

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
        output_duration : float, optional
            Desired output duration (seconds) for ``TIME_SERIES``.
            Bellhop maps this to ``time_window`` for the delay-and-sum
            synthesis and defaults ``t_start=0`` (so its time clock
            starts at source emission, matching the wave-equation
            solvers). When ``None``, falls back to the constructor's
            ``time_window`` / ``t_start`` (which auto-derive from the
            latest arrival and source waveform duration).

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
            self._require_timeseries_signal(run_mode, source_waveform, sample_rate)
            return self._run_broadband(
                env, source, receiver,
                run_mode=run_mode,
                frequencies=frequencies,
                source_waveform=source_waveform,
                sample_rate=sample_rate,
                output_duration=output_duration,
            )

        self._warn_ignored_run_kwargs(
            run_mode,
            frequencies=frequencies,
            source_waveform=source_waveform,
            sample_rate=sample_rate,
            output_duration=output_duration,
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
            return self._run_eigenrays_multi_depth(
                env, source, receiver, run_mode,
                frequencies, source_waveform, sample_rate, output_duration)

        run_type = _RUN_MODE_TO_BELLHOP_TYPE[run_mode]

        _validate_arrivals_format(self.arrivals_format)

        # Auto-route through BOUNCE whenever Bellhop's fluid ray-tracer
        # cannot represent the bottom's full reflection physics natively:
        #   - layered columns — Bellhop has no multi-medium .env format;
        #     without BOUNCE the layers are silently lost.
        #   - a halfspace with non-zero shear — Bellhop's writer emits
        #     cs/alpha_s on the 'A' line (or per-range on the long .bty),
        #     but the ray tracer approximates the resulting reflection
        #     coefficient with fluid physics; BOUNCE pre-computes the exact
        #     elastic RC including shear-conversion and Bellhop consumes it
        #     via the 'F' bottom type.
        # BOUNCE itself is range-independent; the spawned Bounce instance
        # collapses any range-dependent env via its own collapse policy
        # (Bounce default ``bottom_range='median'`` → median range, layer
        # stack kept since BOUNCE consumes layered columns natively).
        # Pass ``collapse={...}`` to Bellhop to override;
        # ``Bellhop.run_with_bounce(...)`` is the explicit form for
        # users who want to control the BOUNCE constructor.
        routed = self._maybe_route_through_bounce(
            env, source, receiver, run_mode,
            frequencies, source_waveform, sample_rate)
        if routed is not None:
            return routed

        from uacpy.io.oalib_writer import resolve_ssp_interp
        effective_interp = resolve_ssp_interp(env, self.interp_ssp)
        interp_for_writer = self.interp_ssp
        if self.interp_ssp is None:
            self._log(
                f"interp_ssp auto-picked = {effective_interp!r} "
                f"(env.has_range_dependent_ssp={env.has_range_dependent_ssp()})"
            )
        if effective_interp == 'quad' and not env.has_range_dependent_ssp():
            # 'quad' is Bellhop's external .ssp (2-D) interpolator; with a
            # range-independent SSP there is no .ssp file to write, so fall
            # back to the auto 1-D interp instead of letting Bellhop fail on
            # a missing model.ssp.
            fallback = resolve_ssp_interp(env, None)
            warnings.warn(
                f"Bellhop(interp_ssp='quad') needs a range-dependent env.ssp "
                f"(the external .ssp / 2-D profile); this environment's SSP is "
                f"range-independent, so falling back to interp_ssp={fallback!r}. "
                f"Provide a range-dependent SSP to use the quad profile.",
                UserWarning, stacklevel=2,
            )
            effective_interp = fallback
            interp_for_writer = fallback

        # When ``interp_ssp`` is pinned to a non-quad scheme, the RD-SSP
        # capability flag is False and ``_project_environment`` collapses the
        # SSP to 1-D (one UserWarning per dropped feature). On the 'quad' /
        # auto path the flag is True and the 2-D profile is written verbatim.
        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)

        # Bellhop never writes the r=0 column (no ray travels zero
        # distance), so it comes back as NaN no-data cells. Newcomers using
        # ``np.linspace(0, R, N)`` for ``receiver.ranges`` hit a wall of
        # NaN at r=0 and rightly wonder what is wrong. Warn once per
        # ``Bellhop`` instance to nudge them toward a non-zero start.
        if (
            run_mode in (RunMode.COHERENT_TL, RunMode.INCOHERENT_TL,
                         RunMode.SEMICOHERENT_TL)
            and len(receiver.ranges) > 0
            and float(receiver.ranges[0]) == 0.0
            and not getattr(self, '_warned_r0_sentinel', False)
        ):
            warnings.warn(
                f"{self.model_name}: receiver.ranges starts at r=0 m. "
                f"Bellhop writes no data there (no ray travels zero "
                f"distance), so that column is NaN. Start ranges at a "
                f"small positive value (e.g. ``np.linspace(eps, R, N)``) "
                f"to avoid surprise.",
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
            'interp_ssp': interp_for_writer,
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
                slab.model_source = self.provenance
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
        output_duration: Optional[float] = None,
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

        self._log("Running BOUNCE to compute reflection coefficients...")
        # Route the bounce scratch through FileManager so it honours
        # ``use_tmpfs`` like every other path and is cleaned up here.
        bounce_fm = FileManager(
            use_tmpfs=self.use_tmpfs,
            prefix='bellhop_bounce_',
            cleanup=True,
        )
        bounce_work_dir = bounce_fm.create_work_dir()
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
            env_bounce.bottom = Bottom.from_halfspace(BoundaryProperties(
                acoustic_type='file',
                reflection_file=brc_file,
            ))

            self._log("Running Bellhop with BOUNCE reflection coefficients...")
            result = self.run(
                env_bounce, source, receiver,
                run_mode=run_mode,
                frequencies=frequencies,
                source_waveform=source_waveform,
                sample_rate=sample_rate,
                output_duration=output_duration,
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
            bounce_fm.cleanup_work_dir()

    def _run_broadband(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: 'RunMode' = None,
        frequencies: Optional[np.ndarray] = None,
        source_waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        output_duration: Optional[float] = None,
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
        # fc is the single carrier frequency Bellhop runs the ray tracer
        # at. If the user passed a multi-element frequency array (band),
        # take the band centre — frequencies[0] would map a [50, 350]
        # band to fc=50, which is the same footgun RAM's
        # ``_resolve_broadband_grid`` avoids.
        src_freqs = np.atleast_1d(np.asarray(source.frequencies, dtype=float))
        fc = (
            float(0.5 * (src_freqs.min() + src_freqs.max()))
            if src_freqs.size > 1
            else float(src_freqs[0])
        )
        # Run-time ``output_duration`` overrides the constructor's
        # ``time_window`` for this call. ``delayandsum`` consumes
        # ``time_window`` to size the per-cell synthesis grid. When the
        # caller asks for a specific duration they typically want the
        # trace anchored at the source-emission instant (t=0) to align
        # with the other broadband solvers; default ``t_start`` to 0
        # in that case so each cell's clock starts at emission rather
        # than at "just before the earliest arrival" (delayandsum's
        # default), which would shift every cell by its own delay.
        effective_time_window = (
            float(output_duration)
            if output_duration is not None
            else self.time_window
        )
        effective_t_start = (
            self.t_start
            if self.t_start is not None
            else (0.0 if output_duration is not None else None)
        )

        # Step 1: Run Bellhop in arrivals mode. Bellhop traces rays at the
        # single carrier fc, so the arrivals run uses a single-frequency source
        # (ARRIVALS does not accept a multi-frequency band).
        self._log("Running in arrivals mode (broadband path)...")
        arr_source = Source(
            depths=source.depths,
            frequencies=fc,
            source_type=source.source_type,
        )
        arr_field = self.run(env, arr_source, receiver, run_mode=RunMode.ARRIVALS)

        arrivals_by_rcv = arr_field.by_receiver
        rz = arr_field.receiver_depths
        rr = arr_field.receiver_ranges  # in meters

        nrd = len(rz)
        nrr = len(rr)

        # ── Path A: time-domain delay-and-sum with source waveform ──
        # Branch on the contracted mode, not on the presence of a waveform:
        # BROADBAND must return H(f) even if a waveform is supplied (the
        # waveform is meaningful only for the p(t) synthesis of TIME_SERIES).
        if run_mode == RunMode.BROADBAND and source_waveform is not None:
            warnings.warn(
                "Bellhop.run(run_mode=BROADBAND) returns the complex transfer "
                "function H(f); the supplied source_waveform is ignored. Use "
                "run_mode=TIME_SERIES to synthesise p(t).",
                UserWarning, stacklevel=2,
            )
        if run_mode == RunMode.TIME_SERIES:

            if sample_rate is None:
                raise ConfigurationError(
                    "sample_rate is required when source_waveform is provided"
                )

            self._log(f"Delay-and-sum over {nrd}×{nrr} receiver grid")

            # Lock the time window from the first cell *with* arrivals so
            # all traces share a clock; an arrival-less cell would let
            # delayandsum fall back to its 0.1 s stub and truncate every
            # trace in the grid.
            lock_arrivals = arrivals_by_rcv[0][0][0]
            for cell in (
                arrivals_by_rcv[0][ird][irr]
                for ird in range(nrd) for irr in range(nrr)
            ):
                if int(cell.get('n_arrivals', 0)) > 0:
                    lock_arrivals = cell
                    break
            _, t_vec = delayandsum(
                rcv_arrivals=lock_arrivals,
                source_timeseries=source_waveform,
                sample_rate=sample_rate,
                fc=fc,
                time_window=effective_time_window,
                t_start=effective_t_start,
            )
            t_start_locked = float(t_vec[0])
            time_window_locked = float(t_vec[-1] - t_vec[0]) + 1.0 / sample_rate
            n_t = len(t_vec)

            data = np.zeros((nrd, nrr, n_t), dtype=float)
            for ird in range(nrd):
                for irr in range(nrr):
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
                    source, backend=self.model_name.lower(), frequencies=fc,
                    dt=1.0 / sample_rate, fs=sample_rate, nt=n_t,
                    t_start=t_start_locked, center_frequency=fc,
                ),
            )

        # ── Path B: frequency-domain transfer function ──
        frequencies = self._resolve_broadband_frequencies(
            source, frequencies,
            n_freqs=self.n_freqs, bandwidth_factor=self.bandwidth_factor,
        )
        n_freq = len(frequencies)

        # Build H(d, r, f) for each (receiver_depth, receiver_range).
        # Use first source depth (most common case). Trailing-axis convention.
        H = np.zeros((nrd, nrr, n_freq), dtype=complex)
        for ird in range(nrd):
            for irr in range(nrr):
                rcv_arr = arrivals_by_rcv[0][ird][irr]
                H[ird, irr, :] = self._arrivals_to_tf(rcv_arr, frequencies)

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
                backend=self.model_name.lower(),
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
    ) -> np.ndarray:
        """
        Build frequency-domain transfer function from per-receiver arrivals.

        For each arrival with amplitude A, phase phi (deg), travel time tau,
        and imaginary delay tau_i (volume attenuation), the contribution to
        H(f) is:

            H(f) += A * exp(tau_i * 2*pi*f) * exp(i*(phi_rad - 2*pi*f*tau))

        The phase from Bellhop already includes the geometric phase (number of
        caustics, boundary reflections). The exponential delay term shifts the
        arrival in the frequency domain. The imaginary delay ``tau_i`` (<= 0)
        encodes frequency-dependent volume attenuation, applied per frequency f.

        Parameters
        ----------
        rcv_arrivals : dict
            Per-receiver arrival data with keys: amplitudes, phases, delays,
            delays_imag, n_arrivals.
        frequencies : ndarray
            Frequency vector in Hz.

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

        # Vectorised over arrivals. For each arrival, tau = Re(tau)+i*Im(tau)
        # gives a phase-shift exp(-i*omega*Re(tau)) and an attenuation
        # exp(omega*Im(tau)); omega is the per-frequency carrier.
        A_complex = np.asarray(amps) * np.exp(1j * phases_rad)        # (n_arr,)
        omega_tau = np.outer(delays, omega)                          # (n_arr, n_freq)
        omega_taui = np.outer(delays_imag, omega)                    # (n_arr, n_freq)
        contrib = A_complex[:, None] * np.exp(omega_taui - 1j * omega_tau)
        return contrib.sum(axis=0)

    def _build_command(self, base_name: str) -> list:
        """Build the argv used to launch the binary.

        The bellhopcxx / bellhopcuda CLIs require a ``--<dim>`` flag; the
        Fortran binary takes none.
        """
        if self.version in ('cuda', 'cxx'):
            return [str(self._exe), f'--{self.dimensionality}', base_name]
        return [str(self._exe), base_name]

    def _run_bellhop(self, base_name: str, work_dir: Path):
        """Execute the Bellhop binary via the shared subprocess runner.

        Bellhop reports most fatal errors in ``<base>.prt`` rather than on
        stderr. If the child exits non-zero, we append the tail of the .prt
        file (up to 2000 chars) to the raised ``ModelExecutionError`` so the
        diagnostic surface to the user instead of a blank stderr.
        """
        cmd = self._build_command(base_name)
        self._run_and_attach_prt(cmd, work_dir, base_name)

