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
import warnings
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from scipy.signal import hilbert

from uacpy.models.base import (
    PropagationModel, RunMode, ModelSpec, USER_FRAME_SKIP,
)
from uacpy.core.environment import Environment, BoundaryProperties, Bottom
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.results import PhaseReference, Result, Field, ResultStack
from uacpy.core.constants import (
    DEFAULT_C_MAX_UNBOUNDED,
    DEFAULT_BROADBAND_N_FREQS, DEFAULT_BROADBAND_BANDWIDTH_FACTOR,
)
from uacpy.core.exceptions import (
    ConfigurationError, ModelExecutionError,
    UnsupportedFeatureError,
)
from uacpy.io.bellhop_writer import (
    validate_beam_shape, validate_beam_type, write_bellhop_env_file,
)
from uacpy.io.refl_io import stage_source_beam_pattern
from uacpy.io.file_manager import FileManager
from uacpy.io.oalib_reader import read_shd_file, read_arr_file, read_ray_file
from uacpy.io.utils import equally_spaced
from uacpy.io.oalib_writer import SOURCE_TYPE_CODE as _SOURCE_TYPE_CODE

# A line source carries the 2-D Green's function's exp(-i*pi/4): (i/4)*H0(kR)
# ~ exp(i(kR + pi/4)) for large kR, while both AT scale factors are purely
# real — ScalePressure's line branch (influence.f90:784,
# factor = -4*sqrt(pi)*const with const = -1) and WriteArrivals' line branch
# (ArrMod.f90:103-104, factor = 4*sqrt(pi)). Applying it on every path keeps
# COHERENT_TL, BROADBAND and TIME_SERIES on one phase reference.
_LINE_SOURCE_PHASE = -np.pi / 4.0

#: Water depths per wavelength below which uacpy's own model-validity table
#: (``docs/models/README.md``, ``docs/models/bellhop.md``) marks ray theory ✗ —
#: "rays are meaningless", use a modal or wavenumber-integral solver. The same
#: table calls 5-20 a cross-check band and >= 20 comfortable, so only the ✗
#: side warns: a cross-check is a suggestion, not a defect.
_RAY_VALIDITY_D_OVER_LAMBDA = 5.0

# (water depth, frequency) pairs already warned about in this process, so the
# ray-validity notice stays one-time: a run that re-enters ``run`` (BOUNCE
# routing, multi-depth EIGENRAYS, the nested ARRIVALS pass of a broadband run)
# warns once for one geometry rather than once per internal call.
_WARNED_RAY_VALIDITY: set = set()


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
    phase_offset: float = 0.0,
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
        Source waveform (1-D), used as-is.
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
    phase_offset : float, optional
        Constant phase (radians) added to every arrival, applied on the
        analytic signal. Carries the line-source ``exp(-i*pi/4)``; ``0.0``
        (default) for a point source.

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
    nsts = len(sts)

    # Compute analytic signal via Hilbert transform
    sts_analytic = hilbert(sts)

    deltat = 1.0 / sample_rate
    src_duration = nsts * deltat

    # Determine time window
    min_delay = float(np.min(delays))
    max_delay = float(np.max(delays))

    # Every arrival places a whole copy of the source waveform starting at its
    # own delay, so the window must reach ``max_delay + src_duration`` for the
    # last one to fit; ``2 *`` leaves a further source duration of tail. The
    # lead-in keeps the earliest arrival's leading edge inside the window, and
    # the max(0, ...) stops the clock from starting before source emission.
    if t_start is None:
        t_start = max(0.0, min_delay - 0.1 * src_duration)

    if time_window is None:
        time_window = (max_delay - t_start) + 2.0 * src_duration

    nrts = int(np.ceil(time_window * sample_rate))
    rts = np.zeros(nrts)

    omega_c = 2.0 * np.pi * fc
    for ia in range(n_arr):
        phase_rad = np.deg2rad(phases_deg[ia]) + phase_offset
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


#: uacpy run mode -> Bellhop ``RunType(1:1)``. The letters are the manual's
#: (``doc/bellhop.htm`` section 8, RUN TYPE): 'R' ray file, 'E' eigenray file,
#: 'A' amplitude-delay ascii, 'a' amplitude-delay binary, 'C' coherent TL,
#: 'I' incoherent TL, 'S' semicoherent TL.
#:
#: This is position 1 and shares its alphabet with position 2 (``beam_type``),
#: where the same characters name influence routines instead: 'C' is coherent
#: TL here but Cerveny-Cartesian there, 'S' is semicoherent here but Bucker's
#: simple Gaussian there, and 'R' is a ray trace here but Cerveny ray-centred
#: there. Never resolve a letter without knowing which position it sits in.
_RUN_MODE_TO_BELLHOP_TYPE = {
    RunMode.COHERENT_TL: 'C',
    RunMode.INCOHERENT_TL: 'I',
    RunMode.SEMICOHERENT_TL: 'S',
    RunMode.RAYS: 'R',
    RunMode.EIGENRAYS: 'E',
    RunMode.ARRIVALS: 'A',
}
# ``_run_broadband`` runs ARRIVALS to get the eigenray set, so both broadband
# routes inherit the arrivals row of _BEAM_TYPE_RUN_TYPES.
_RUN_MODE_TO_INFLUENCE_LETTER = {
    **_RUN_MODE_TO_BELLHOP_TYPE,
    RunMode.BROADBAND: 'A',
    RunMode.TIME_SERIES: 'A',
}

# Which RunType(1:1) letters each influence routine actually implements,
# enumerated from Bellhop/influence.f90. 'G', 'B' and 'g' funnel through
# ApplyContribution (:629-652), which holds the only CALL AddArr (:640) and so
# is the sole implementer of the 'A'/'a' (arrivals) branch. 'S' (InfluenceSGB,
# :656-725) has its own CASE('E') (:701-703) writing the ray, so it serves
# eigenrays too, but it has no 'A'/'a', 'I' or 'S' branch: those fall into its
# CASE DEFAULT (:704), which writes complex pressure into U. For an 'A'/'E' run
# bellhop.f90:216 allocated U as U(1,1), so on 'A'/'a' that write runs off the
# end of the heap block; on 'I'/'S' ScalePressure's SQRT( REAL( U ) ) (:779)
# takes the root of a signed coherent sum. The two Cerveny beam types
# (InfluenceCervenyCart :157-289, InfluenceCervenyRayCen :19-153) branch on
# 'C'/'I'/'S' only and write U unconditionally, so both 'E' and 'A'/'a' hit
# that same U(1,1) overrun there. A RAYS run is safe for every beam type
# because bellhop.f90:288 writes the trajectory and never calls an influence
# routine.
_BEAM_TYPE_RUN_TYPES = {
    'G': frozenset({'C', 'I', 'S', 'E', 'A', 'R'}),
    'B': frozenset({'C', 'I', 'S', 'E', 'A', 'R'}),
    'g': frozenset({'C', 'I', 'S', 'E', 'A', 'R'}),
    'S': frozenset({'C', 'E', 'R'}),
    'C': frozenset({'C', 'I', 'S', 'R'}),
    'R': frozenset({'C', 'I', 'S', 'R'}),
}
_INFLUENCE_ROUTINE = {
    'G': 'InfluenceGeoHatCart', 'B': 'InfluenceGeoGaussianCart',
    'g': 'InfluenceGeoHatRayCen', 'S': 'InfluenceSGB',
    'C': 'InfluenceCervenyCart', 'R': 'InfluenceCervenyRayCen',
}
# Beam types whose influence routine re-reads the paired receiver depth when
# RunType(5:5)=='I' (influence.f90:461-465 and :581-585). The other four index
# the depth by the depth-loop counter, which bellhop.f90:202-204 has pinned to
# NRz_per_range == 1, so every paired receiver is evaluated at Pos%Rz(1).
_IRREGULAR_GRID_BEAM_TYPES = frozenset({'G', 'B'})
# Beam types that form the receiver index as
# INT( ( r - Pos%Rr(1) ) / Pos%Delta_r ) + 1 — influence.f90:92 ('R'), :223-224
# ('C'), :339 and :351 ('g'), each flagged "assumes uniform spacing in Pos%r".
# Pos%Delta_r is only the *last* gap (SourceReceiverPositions.f90:160), so an
# unevenly spaced range axis sends every step to the wrong column. 'G', 'B' walk
# ir with a bracket test and 'S' compares rB > Pos%Rr(ir) directly, so all three
# take an arbitrary range vector — doc/bellhop.htm's list is wrong twice, both
# including 'R' and omitting 'S'.
_UNIFORM_RANGE_BEAM_TYPES = frozenset({'g', 'C', 'R'})

# RunType letter -> (metadata key, suffix, reader). A run writes exactly one of
# these, so only the resolved entry is read and attached; the other two would be
# an earlier run's leftovers in a pinned work_dir.
_BELLHOP_OUTPUT = {
    'C': ('shd_file', '.shd', read_shd_file),
    'I': ('shd_file', '.shd', read_shd_file),
    'S': ('shd_file', '.shd', read_shd_file),
    'A': ('arr_file', '.arr', read_arr_file),
    'R': ('ray_file', '.ray', read_ray_file),
    'E': ('ray_file', '.ray', read_ray_file),
}
_BELLHOP_OUTPUT_SUFFIXES = ('.shd', '.arr', '.ray')


# bellhop.f90:176-178 zeroes the angular spacing for a single beam, which
# gives every beam zero width and an all-NaN field at exit 0. Two is the
# smallest fan with a finite Dalpha, and it does produce a field wherever its
# rays reach the receiver: on a deep-water direct path (1 km source and
# receiver, 2 km range, +/-10 deg fan) it returns 69.1 dB against a converged
# 66.1 dB. Under-resolved, not degenerate — and not specially so, since a
# five-beam fan on the same geometry lands a beam edge on the receiver and
# reads 100.1 dB. How well a sparse fan covers a grid is the caller's problem.
_MIN_INFLUENCE_BEAMS = 2


# Volume-attenuation mismatch (dB per km of path) above which a BROADBAND run
# says so. The band inherits one trace's Im(tau), which applies attenuation
# linearly in frequency; 0.05 dB/km is about 0.5 dB over a 10 km path, below
# which the single-trace band is not what limits the answer.
_BROADBAND_ATTEN_WARN_DB_PER_KM = 0.05


def _fan_miss_count_and_worst(zs, zr, rr, fan_lo, fan_hi):
    """How many (source depth, receiver depth, range) triples need a launch
    angle outside ``[fan_lo, fan_hi]``, and the steepest angle among them.

    Returns exactly what ``needed = degrees(arctan2(zr - zs, rr))`` over the
    full grid gives for ``outside.sum()`` and for the largest-magnitude
    ``needed[outside]``, without holding the grid: at a fixed depth difference
    ``d`` the angle ``atan2(d, r)`` is monotone in ``r``, so on a sorted range
    axis the angles below the fan and the angles above it are each one
    contiguous run — ``searchsorted`` gives the two run lengths, and the
    steepest angle in a run is at one of its ends. Cost is the
    (n_sz x n_rz) depth-difference matrix and one pass over the ranges per
    depth pair, never their product.
    """
    r_sorted = np.sort(rr)
    n_out = 0
    worst = 0.0
    for d in (zr[None, :] - zs[:, None]).ravel():
        ang = np.degrees(np.arctan2(d, r_sorted))
        if d > 0.0:                 # falling in r — reverse it to ascending
            ang = ang[::-1]
        i_lo = int(np.searchsorted(ang, fan_lo, 'left'))    # ang[:i_lo] < lo
        i_hi = int(np.searchsorted(ang, fan_hi, 'right'))   # ang[i_hi:] > hi
        n_out += i_lo + (ang.size - i_hi)
        ends = ([ang[0], ang[i_lo - 1]] if i_lo > 0 else [])
        ends += ([ang[i_hi], ang[-1]] if i_hi < ang.size else [])
        for end in ends:
            if abs(end) > abs(worst):
                worst = float(end)
    return n_out, worst


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

    The backends are not the same code. ``'fortran'`` is Porter's
    ``Acoustics-Toolbox`` BELLHOP; ``'cxx'`` and ``'cuda'`` are ports of
    ``A-New-BellHope``, a fork that deliberately fixes bugs and edge cases in
    it (``bellhopcuda/doc/accuracy.md:38-39``: results are compared "to our
    modified Fortran version, not to the original BELLHOP"). Measured over
    two 2-D scenarios, the ports agree with the Fortran to ~0.3 dB at p99
    with excursions of a few dB at interference nulls, and **no bias**
    (|signed mean| <= 0.0007 dB); ``'cxx'`` and ``'cuda'`` are identical to
    each other. :attr:`Result.backend` records which binary ran.

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
        ``'3D'`` is rejected because 3-D running is not yet available: the
        env writer cannot emit a 3D-format input file, so a 3D flag would
        mis-drive the binary. The BELLHOP3D / FIELD3D *file* readers and
        writers are already in :mod:`uacpy.io`, retained for it —
        ``write_bty_3d`` / ``read_boundary_3d``, ``read_ssp_3d``,
        ``write_field3dflp`` / ``read_flp3d``.
    beam_type : str, optional
        ``B`` geometric Gaussian, Cartesian (default) | ``R`` Cerveny
        ray-centered | ``C`` Cerveny Cartesian | ``g``/``G`` geometric hat,
        ray-centered / Cartesian | ``S`` Bucker's simple Gaussian, Cartesian
        (``influence.f90:658``). Each letter selects one influence routine at
        ``bellhop.f90:296-311``; ``G`` is the ``CASE DEFAULT`` there rather
        than a case of its own, so an unrecognised letter would also run
        geometric-hat Cartesian — :func:`validate_beam_type` refuses one
        instead.

        **These letters are RunType(2:2) and share their alphabet with
        RunType(1:1), where the same characters mean something else entirely:**
        ``C`` is coherent TL in position 1 and Cerveny-Cartesian in position 2,
        ``S`` is semi-coherent TL against Bucker's simple Gaussian, and ``R``
        is a ray trace against Cerveny ray-centred. ``run_mode`` sets position
        1 and ``beam_type`` position 2; they are never interchangeable.

        ``B`` and ``G`` trade places by scenario, so neither is right
        everywhere; measured against Kraken as the wave-theoretic reference:

        * 200 m Pekeris guide, 150 Hz, five source/receiver pairs over
          2-20 km — ``G`` 1.36 dB rms, ``B`` 2.26 dB. Boundary-dominated
          guides with few strong eigenrays favour the hat beam.
        * Munk 5000 m, 50 and 150 Hz, 10-100 km — ``B`` 2.70 dB rms,
          ``G`` 3.22 dB. Shadow zones and caustics favour the Gaussian beam,
          which is why it is the default.

        Only ``G`` is exactly reciprocal. Measured on the Pekeris case above
        (30 m / 150 m, 5 km): ``G`` matched to 0.000 dB while ``B`` differed
        by 0.9-1.0 dB. That is not a discretisation error: over 73 ranges
        from 2-20 km the gap is 1.42 dB rms / 3.44 dB max and holds across an
        80x refinement of the ray step (4 m, 1 m, 0.25 m, 0.05 m) and across
        beam counts from 201 to 4001, while ``G`` on the same sweep converges
        to 0.000 dB. The wave engines are reciprocal to <= 0.004 dB.

        What the sources establish, and nothing beyond it:

        * Ray theory preserves reciprocity. JKPS Sect. 3.6.8 ('Reciprocity') proves the
          spreading function is symmetric under exchange of endpoints,
          ``q(s2; s1) = q(s1; s2)``, from the constancy of the Wronskian
          ``W(s)/c(s)``.
        * ``B`` and ``G`` are both GEOMETRIC beams. ``bellhop.f90:308-311``
          sends ``B`` to ``InfluenceGeoGaussianCart`` and ``G`` to
          ``InfluenceGeoHatCart``; the Cerveny routines are ``R`` (``:299``)
          and ``C`` (``:303``).
        * ``B`` replaces the hat shape function with a Gaussian of the same
          fan-derived width (JKPS Sect. 3.3.5.5, 'Geometric Beams'). Those beams "serve simply to
          interpolate the field and are not intended to approximate the
          physics of a true Gaussian beam".
        * The hat vanishes at the neighbouring rays, so its width is set by
          the ray-fan density (JKPS Sect. 3.3.5.5). The Gaussian "has an
          influence at any distance from the central ray", cut off where it is
          negligible (Bellhop user guide), so neighbouring beams overlap.

        The corpus does not state whether ``B`` is reciprocal, and no
        mechanism linking the above to the measurement is asserted here.

        Use ``G`` when a reciprocal field matters (e.g. filling a
        source-receiver matrix from one half).
    n_beams : int, optional
        Number of beams; ``0`` lets Bellhop auto-pick. Default ``0``.
    alpha : tuple, optional
        Launch-angle limits ``(min, max)`` in degrees. Default ``(-80, 80)``.
    step : float, optional
        Ray step size (m); ``0`` resolves to ``env.depth / 50``.
        Default ``0.0``.
    z_box, r_box : float, optional
        Ray-trace bounding box (m); rays are dropped once they leave it.
        ``None`` ⇒ ``1.2 ×`` the receiver extent (``z_box = 1.2 × env.depth``,
        ``r_box = 1.2 × range_max``, or 10 km when the receiver range is 0).
        Box%r is a horizontal-range cut-off, so the 1.2× pad already
        captures arrivals at the outer receivers; do not enlarge it past a
        range-dependent SSP's defined extent.
    grid_type : str, optional
        ``'R'`` rectilinear (default) | ``'I'`` irregular (paired depth/range).
    beam_width_type : str, optional
        Cerveny only (``bellhop.f90:373-390``). ``'F'`` space filling |
        ``'M'`` minimum width | ``'W'`` WKB.
    beam_curvature : str, optional
        Curvature condition applied on a boundary reflection
        (``ReadEnvironmentBell.f90:202-210``, ``bellhop.f90:667-672``).
        ``'D'`` curvature doubling | ``'S'`` standard curvature |
        ``'Z'`` curvature zeroing.
    eps_multiplier, r_loop, n_image, ib_win : optional
        Cerveny advanced beam knobs (used when ``beam_type ∈ {C, R}``).
        ``r_loop`` is in metres.
    component : optional
        ``'P'``/``'V'``/``'H'``, honoured by ``beam_type='R'`` alone —
        see the constructor's own entry.
    auto_bounce : bool, optional
        Default ``True``. When ``env.bottom`` is layered — a stack Bellhop's
        single-halfspace ``.env`` cannot carry — ``run(...)`` auto-routes
        through BOUNCE to derive a ``.brc`` reflection-coefficient table.
        Set ``False`` to skip the auto-route — Bellhop then collapses the
        layer stack to a halfspace via its own ``collapse={…}`` policy, with
        one ``UserWarning``. A non-layered elastic halfspace never routes:
        Bellhop computes its exact acousto-elastic reflection coefficient
        natively (``bellhop.f90:694-712``), per range node on a
        range-dependent bottom. ``run_with_bounce(...)`` always uses BOUNCE
        regardless of this flag.
    use_tmpfs, verbose, work_dir, cleanup, timeout, collapse : optional
        Standard plumbing (see :class:`PropagationModel`).

    Notes
    -----
    Defaults auto-derived from inputs (no need to override unless tuning):

    - ``n_beams=0`` → Bellhop auto-picks the beam count.
    - ``step=0.0`` → ``env.depth / 50``. Bellhop's own default for a zero
      step is ``depth/10`` (``bellhop.f90:170-174``), which under-resolves a
      near-horizontal refracted ray: measured 26.56 dB max against a
      converged step on Munk 5000 m at 100 Hz.
    - ``z_box=None`` → ``1.2 × env.depth``.
    - ``r_box=None`` → ``1.2 × receiver.range_max`` (or 10 km if 0).
    - ``TopOpt`` position 4 reads from ``env.absorption``
      (``Thorp`` → ``'T'``, ``FrancoisGarrison`` → ``'F'`` + params,
      ``Biological`` → ``'B'`` + layers, ``ConstantAbsorption`` /
      ``None`` → ``' '``).
    - Bottom reflection: when ``env.bottom`` is layered and
      ``auto_bounce=True``, BOUNCE is invoked transparently to derive
      the ``.brc`` reflection coefficient table. An elastic halfspace
      needs no table — ``bellhop.f90:694-712`` evaluates the exact
      acousto-elastic reflection coefficient at each boundary hit.

    **Auto-route through BOUNCE.** ``Bellhop.run(...)`` detects a layered
    ``Bottom`` (sediment layers over the halfspace, anywhere along range),
    runs BOUNCE upstream to derive a ``.brc``
    reflection-coefficient table, and re-runs Bellhop against
    ``acoustic_type='file'`` (one ``UserWarning``). The user's
    ``collapse={…}`` dict is forwarded to the spawned Bounce. Use
    :meth:`run_with_bounce` for explicit control over BOUNCE parameters.
    A non-layered elastic bottom runs natively: the halfspace's cp/cs pair
    is written to the deck (per range node on a long-format ``.bty``) and
    the ray tracer applies the exact acousto-elastic reflection
    coefficient, so no BOUNCE pass — which would also collapse a
    range-dependent bottom to one column — is interposed.

    Bellhop uses the global :data:`DEFAULT_COLLAPSE` policy without
    overrides — RD bathymetry / RD bottom / RD-SSP (when the model's
    ``interp_ssp='quad'``) are honoured natively.

    **Broadband amplitude approximation (BROADBAND / TIME_SERIES).** The
    arrival set is computed by a *single* ray trace at the band centre
    ``fc`` and reused across the synthesised band
    ``[fc(1-bw/2), fc(1+bw/2)]`` (``bw = bandwidth_factor``). On the
    **BROADBAND** route the travel time ``Re(τ)`` is applied exactly per
    frequency (``exp(-i2πf·τ)``, :meth:`_arrivals_to_tf`), so the *timing*
    and spreading of every arrival are correct at all frequencies.
    ``Im(τ)``, however, is frozen at ``fc`` — ``Step.f90:73`` accumulates
    ``tau += hw/CMPLX(c, cimag)`` with ``cimag = alphaT·c²/ω``
    (``misc/AttenMod.f90:113``) — so ``exp(2πf·Im τ)`` applies the volume
    attenuation **linearly in f**. Real absorption laws are not linear
    (Thorp goes as ≈ ``f^1.8``), so the band edges are over- and
    under-attenuated: measured with ``Thorp()`` at ``fc = 10`` kHz and the
    default ``bandwidth_factor=0.5``, +11.51 / −6.75 dB at 40 km against a
    trace run at the edge frequency itself. A run whose band incurs more
    than 0.05 dB/km of this says so. **TIME_SERIES** synthesises in
    the time domain instead (:func:`delayandsum`), which delays each
    arrival exactly but takes its volume attenuation as the single factor
    ``exp(2π·fc·Im(τ))`` — frequency-flat like the amplitude below. What is
    held frequency-flat on *both* routes is the geometric beam
    **amplitude and caustic phase**: a Gaussian beam's
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
            'source_beam_pattern',
        },
        source_types=frozenset({'point', 'line'}),
    )
    source = 'acoustics_toolbox'

    def __init__(
        self,
        # Binary selection
        executable: Optional[Path] = None,
        backend: Optional[str] = None,
        dimensionality: str = '2D',
        # Ray fan and trace bounding box
        beam_type: str = 'B',
        n_beams: int = 0,
        alpha: tuple = (-80, 80),
        step: float = 0.0,
        z_box: Optional[float] = None,
        r_box: Optional[float] = None,
        # Receiver grid and environment interpolation
        grid_type: str = 'R',
        interp_ssp: Optional[str] = None,
        interp_bathymetry: str = 'linear',
        interp_altimetry: str = 'linear',
        # Cerveny beam shape (written only for beam_type 'C' / 'R')
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
        # Standard plumbing
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
            binary, which has no such flag). ``'3D'`` raises: 3-D running is
            not yet available because the env writer produces 2D-format
            input only. The BELLHOP3D / FIELD3D *file* readers and writers
            are already in :mod:`uacpy.io` and retained for it
            (``write_bty_3d`` / ``read_boundary_3d``, ``read_ssp_3d``,
            ``write_field3dflp`` / ``read_flp3d``).
        beam_type : str
            Beam type: 'B' (geometric Gaussian, Cartesian), 'R' (Cerveny
            Gaussian, ray-centered), 'C' (Cerveny Gaussian, Cartesian),
            'g' (geometric hat, ray-centered), 'G' (geometric hat,
            Cartesian), 'S' (simple Gaussian). Default: 'B'. ``'b'``
            (geometric Gaussian, ray-centered) is rejected — the Fortran
            solver aborts on it and the C++/CUDA ports silently substitute
            the Cartesian beam.
        n_beams : int
            Number of beams, ``>= 0``. Passing 0 defers to Bellhop's own
            estimate — ``angleMod.f90:38`` tests ``Nalpha == 0`` exactly,
            then takes ``MAX(INT(0.3 * Rmax * f / c0), 300)`` raised further
            by a beam-width-versus-depth rule (``angleMod.f90:44-50``); a
            ray-trace run gets 50. Default: 0.
        alpha : tuple
            Launch angle limits (min, max) in degrees, min < max.
            Default: (-80, 80).
        step : float
            Ray step size in meters, ``>= 0``. 0 resolves to
            ``env.depth / 50`` (see :attr:`_STEP_PER_DEPTH`). Default: 0.0.
        z_box : float, optional
            Maximum depth for ray box. None = 1.2 * max depth. Default: None.
        r_box : float, optional
            Maximum range for ray box. None = 1.2 * max range. Default: None.
        grid_type : str
            Receiver grid: 'R' (rectilinear), 'I' (irregular). Default: 'R'.
        interp_ssp : str, optional
            SSP connection scheme. ``None`` (default) auto-picks
            ``'quad'`` for a range-dependent ``env.ssp`` and ``'linear'``
            otherwise. Explicit values: ``'linear'``, ``'pchip'``,
            ``'cubic'``, ``'quad'``, ``'n2linear'``.
            ``env.ssp.shape='isovelocity'`` always forces ``'C'`` regardless.
            ``'analytic'`` is **not** accepted: AT's ``'A'`` profile is a
            hard-coded Munk curve on a fixed 5000 m grid (``misc/munk.f90``)
            that ignores ``env.ssp``, so it is refused with that explanation
            (:func:`~uacpy.io.oalib_writer.resolve_ssp_topopt`).
        interp_bathymetry : str, optional
            ``.bty`` interpolation. ``'linear'`` (default) or
            ``'curvilinear'``.
        interp_altimetry : str, optional
            ``.ati`` interpolation. ``'linear'`` (default) or
            ``'curvilinear'``.
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
        component : {'P', 'V', 'H'}, optional
            Field component computed by the Cerveny **ray-centred**
            influence routine (influence.f90:120-130): 'P' pressure
            (default), 'V' vertical particle velocity, 'H' horizontal
            particle velocity. That routine is ``beam_type='R'`` alone —
            every other beam type ignores the letter (with a warning), and
            'V'/'H' on 'R' is refused, since the resulting .shd holds
            particle velocity in m/s and a :class:`Field` can only report it
            as pressure.
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
            Default ``True``. When ``env`` carries a *layered* ``Bottom``
            (sediment layers Bellhop's ray tracer cannot mesh), ``run(...)``
            auto-routes through BOUNCE to derive a ``.brc`` reflection-
            coefficient table and re-runs Bellhop against
            ``acoustic_type='file'``, attaching the in-memory
            :class:`ReflectionCoefficient` to
            ``result.metadata['bounce_result']``. Elastic half-spaces
            (shear, range-dependent included) never route: Bellhop applies
            the exact acousto-elastic reflection coefficient natively
            (``Bellhop/bellhop.f90:694-712``). Set ``False`` to skip the
            auto-route — Bellhop then collapses the layered bottom via its
            own ``collapse={…}`` policy, with one ``UserWarning``.
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

        validate_beam_type(beam_type, 'Bellhop')
        validate_beam_shape(beam_width_type, beam_curvature, component,
                            'Bellhop')
        if n_beams is not None and (
                isinstance(n_beams, bool)
                or not isinstance(n_beams, (int, np.integer))):
            raise ConfigurationError(
                f"Bellhop(n_beams={n_beams!r}) must be an integer beam "
                f"count. The deck writer emits int(n_beams), so a "
                f"fractional value would silently truncate toward zero."
            )
        if n_beams is not None and n_beams < 0:
            raise ConfigurationError(
                f"Bellhop(n_beams={n_beams}) must be >= 0. angleMod.f90:38 "
                f"auto-estimates the beam count only on an exact 0; a "
                f"negative value reaches ALLOCATE unchecked."
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
        try:
            alpha_lo, alpha_hi = float(alpha[0]), float(alpha[1])
        except (TypeError, ValueError):
            raise ConfigurationError(
                f"Bellhop(alpha={alpha!r}) entries must be numbers "
                f"(min_deg, max_deg) of launch-angle limits."
            ) from None
        if not (alpha_lo < alpha_hi):
            raise ConfigurationError(
                f"Bellhop(alpha={alpha!r}) limits must satisfy "
                f"min_deg < max_deg. SubTab (misc/subtabulate.f90:41-45) "
                f"fills the fan uniformly from alpha(1) to alpha(2); a "
                f"reversed pair traces a negative-width fan."
            )
        if not np.isfinite(step) or step < 0:
            raise ConfigurationError(
                f"Bellhop(step={step!r}) must be >= 0 and finite. 0 defers "
                f"to the automatic step (env.depth / "
                f"{self._STEP_PER_DEPTH:g}); "
                f"the deck writes the value as the ray-marching step size "
                f"in meters."
            )
        self.beam_type = beam_type
        self.n_beams = n_beams
        self.alpha = alpha
        self.step = step
        self.z_box = z_box
        self.r_box = r_box
        self.grid_type = grid_type
        self.interp_ssp = interp_ssp
        self.interp_bathymetry = interp_bathymetry
        self.interp_altimetry = interp_altimetry
        self.beam_width_type = beam_width_type
        self.beam_curvature = beam_curvature
        self.eps_multiplier = float(eps_multiplier)
        self.r_loop = float(r_loop)
        self.n_image = int(n_image)
        self.ib_win = int(ib_win)
        self.component = component
        self.beam_shift = bool(beam_shift)
        self._validate_component()
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
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
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
        if backend is not None and backend not in ('fortran', 'cxx', 'cuda'):
            raise ConfigurationError(
                f"Bellhop(backend={backend!r}) is not a known backend. "
                f"Choose 'fortran', 'cxx', 'cuda', or None for automatic "
                f"selection (CUDA > C++ > Fortran)."
            )
        self.backend = backend
        if dimensionality != '2D':
            raise UnsupportedFeatureError(
                'Bellhop',
                f"dimensionality={dimensionality!r} — 3-D running is not yet "
                f"available: the env writer produces 2D-format input files, "
                f"so a '3D' (or any other) flag would mis-drive the binary "
                f"(silent 2D run on Fortran, abort on cxx/cuda). The file "
                f"layer for it is already in the package and kept for when "
                f"3-D is wired up — uacpy.io.write_bty_3d / read_boundary_3d "
                f"(BELLHOP3D boundary grids), read_ssp_3d (hexahedral SSP) "
                f"and write_field3dflp / read_flp3d (FIELD3D decks)",
                alternatives=["'2D'"],
                alternatives_label='dimensionality values',
            )
        self.dimensionality = dimensionality
        self.version = "unknown"

        # A copy re-resolves the binary honoring ``backend=`` instead of
        # re-pinning the already-resolved path (which would flip ``version``
        # to 'custom' and drop the cxx/cuda ``--<dim>`` flag).
        self._exe = self._resolve_executable(
            executable, self._find_bellhop_executable,
        )
        if self.executable is not None:
            self.version = "custom"

        if self.version != "custom":
            self._log(f"Using Bellhop {self.version}: {self._exe}")

    def _validate_component(self) -> None:
        """``component`` is a Cerveny **ray-centred** knob only.

        ``Beam%Component`` has exactly one use site in the solver:
        ``influence.f90:120-130``, inside ``InfluenceCervenyRayCen`` —
        ``beam_type='R'``. ``InfluenceCervenyCart`` (``'C'``,
        ``influence.f90:157-289``) never reads it, and no geometric routine
        does either, while the writer still emits the letter and the ``.prt``
        echoes it back: the run looks configured and returns pressure.

        Where the letter *is* honoured the ``.shd`` holds particle velocity
        (m/s), which the :class:`~uacpy.core.results.Field` contract has no
        ``kind`` for — its pressure conventions (the point/line ``_shd_phase``
        correction, ``phase_reference='travelling_wave'``, ``unit='Pa'``,
        ``.db`` as transmission loss) would all be applied to it and every
        one of them would be wrong. So that pairing is refused rather than
        mislabelled.
        """
        component = str(self.component).upper()
        if component == 'P':
            return
        if self.beam_type.upper() != 'R':
            warnings.warn(
                f"Bellhop(component={self.component!r}) is ignored for "
                f"beam_type={self.beam_type!r}: Beam%Component is read only by "
                f"InfluenceCervenyRayCen (influence.f90:120-130), the "
                f"beam_type='R' routine. The letter still reaches the deck and "
                f"the .prt echoes it, but the field returned is pressure.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
            return
        raise UnsupportedFeatureError(
            model_name='Bellhop',
            feature=(
                f"component={self.component!r} on beam_type='R' — the .shd "
                f"would then hold particle velocity (m/s), and uacpy's Field "
                f"carries no such kind: the result would report unit 'Pa', "
                f"apply the pressure point/line phase correction and read its "
                f".db as transmission loss"
            ),
            alternatives=[
                "Bellhop(component='P') for the pressure field",
                "Derive the particle velocity from the pressure field "
                "(v = -grad(p) / (i·omega·rho))",
            ],
        )

    def _warn_on_ignored_cerveny_knobs(self) -> None:
        """The Cerveny beam knobs are written only for ``beam_type`` in
        {'C', 'R'} (ReadEnvironmentBell.f90). Warn if any is set to a
        non-default value while a non-Cerveny beam is selected, since it
        would otherwise be silently ignored.

        ``component`` is narrower still — ray-centred Cerveny only — and is
        handled by :meth:`_validate_component` on every beam type.
        """
        if self.beam_type.upper() in ('C', 'R'):
            return
        defaults = {
            'beam_width_type': 'F', 'beam_curvature': 'D',
            'eps_multiplier': 1.0, 'r_loop': 1000.0,
            'n_image': 1, 'ib_win': 4,
        }
        ignored = [name for name, default in defaults.items()
                   if getattr(self, name) != default]
        if ignored:
            warnings.warn(
                f"Bellhop: Cerveny beam knobs {', '.join(ignored)} are "
                f"ignored for beam_type={self.beam_type!r} (they apply only "
                f"to Cerveny beams, beam_type 'C' or 'R').",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )

    def _validate_geometry(self, env, source, receiver, run_mode=None):
        """Reject a source on or below the seabed at its own range.

        ``bellhop.f90:488-492`` (``TraceRay2D``) tests ``DistBegTop <= 0 .OR.
        DistBegBot <= 0`` and, when either holds, sets ``Beam%Nsteps = 1``,
        prints *"Terminating the ray trace because the source is on or outside
        the boundaries"* and returns — *"source must be within the medium"*.
        Every beam is then one point long, so the run exits 0 with an all-NaN
        field, zero arrivals and 1-point rays, and warns about none of it.
        The test is ``<= 0``, so **on** the boundary already counts as outside.

        Measured against the seafloor at **r = 0**, not ``env.depth``:
        ``bellhop.f90:237`` launches from ``xs = [0.0, Pos%sz(is)]``, so on a
        sloping bottom a source can be buried at its own range while sitting
        well above ``env.depth`` — a check against ``env.depth`` misses it.

        Bellhop-only. Kraken, Scooter and RAM all return finite, continuous
        fields for a source at or below the seabed (measured -96.8, -58.6 and
        -101.2 dB on the same environment), so this must not go on the shared
        funnel.
        """
        super()._validate_geometry(env, source, receiver, run_mode)
        seafloor = float(np.asarray(env.bathymetry.eval(range=0.0)).flat[0])
        zs = np.atleast_1d(np.asarray(source.depths, dtype=float))
        if np.any(zs >= seafloor):
            raise ConfigurationError(
                f"Bellhop: source depth {float(zs.max()):g} m is at or below "
                f"the seafloor at the source range (r = 0), which is "
                f"{seafloor:g} m. Bellhop terminates every ray at step 1 when "
                f"the source is on or outside a boundary "
                f"(Bellhop/bellhop.f90:488-492, 'source must be within the "
                f"medium'), so the run would return an all-NaN field at exit 0.",
                remediation="Move the source above the seafloor, or use "
                            "Kraken / Scooter / RAM, which resolve a source at "
                            "or below the seabed.")

        # Top half of the same Fortran test: bellhop.f90:488-492 rejects
        # DistBegTop <= 0 symmetrically with the bottom. The top boundary at
        # the launch range r = 0 sits at z = 0 for a flat surface and at
        # z = -height(0) with altimetry (``bellhop_writer`` writes the
        # .ati depth column as -heights: env.altimetry is positive-up,
        # the .ati positive-down), so a source at or above it has
        # every ray terminated at step 1 and the run returns an all-NaN
        # field at exit 0.
        top = 0.0
        if env.altimetry is not None:
            top = -float(np.asarray(env.altimetry.eval(range=0.0)).flat[0])
        if np.any(zs <= top):
            raise ConfigurationError(
                f"Bellhop: source depth {float(zs.min()):g} m is at or above "
                f"the top boundary at the source range (r = 0), which sits "
                f"at z = {top:g} m. Bellhop terminates every ray at step 1 "
                f"when the source is on or outside a boundary "
                f"(Bellhop/bellhop.f90:488-492, 'source must be within the "
                f"medium'), so the run would return an all-NaN field at "
                f"exit 0.",
                remediation="Move the source below the sea surface (below "
                            "the altimetry crest at r = 0, if any).")

    # Ray step as a fraction of the water depth, used when the caller pins
    # none. ``bellhop.f90:170-174`` substitutes ``depth/10`` for a zero
    # ``deltas``, which is a fraction of the WATER DEPTH rather than of any
    # wavelength or gradient scale, and ``Step.f90:138-146``'s ``hInt`` only
    # shortens a step at an SSP-layer crossing (``ReduceStep2D`` also shortens
    # at a top, bottom or bathymetry-segment crossing — ``:148-154``,
    # ``:156-162``, ``:172-180``, combined at ``:188``) — a near-horizontal
    # refracted ray in deep water crosses none of those, so it integrates at
    # depth/10. Measured on Munk 5000 m at 100 Hz, source and receiver at
    # 1000 m, 10-100 km, against a converged 5 m step: depth/10 (500 m) is
    # 26.56 dB max / 6.48 dB rms out, depth/50 (100 m) 1.90 / 0.46, depth/500
    # (10 m) 0.42 / 0.10. Shallow water was never badly served but improves
    # too: 100 m guide at 1 kHz, 0.151 dB max at depth/10 against 0.026 at
    # depth/50. depth/50 is also what AT's own deep-water reference deck picks
    # (``tests/MunkRot/Munk.env`` writes 100.0 m in a 5500 m box; its arctic
    # deck goes finer still, 10 m in 3800 m), and it costs 0.7 s against 0.6 s
    # on the Munk case. Ray sums near a caustic are not monotone in the step,
    # so a convergence check remains the only way to be sure (JKPS section
    # 3.3).
    _STEP_PER_DEPTH = 50.0

    def _resolve_step(self, env) -> float:
        """Ray step (m) for the deck: the caller's if pinned, else
        ``env.depth / _STEP_PER_DEPTH``.

        Returning a positive number keeps ``bellhop.f90:170-174`` from
        substituting its own ``depth/10``.
        """
        if self.step:
            return float(self.step)
        depth = float(env.depth)
        if not np.isfinite(depth) or depth <= 0.0:
            return float(self.step)      # no depth to scale by; binary decides
        return depth / self._STEP_PER_DEPTH


    def _receiver_grid_is_paired(self, receiver) -> bool:
        """``grid_type='I'`` writes RunType(5:5)='I', where BELLHOP walks the
        depth and range arrays together (one receiver per index) rather than
        over their Cartesian product — the pairing the ``'I'`` deck also
        enforces by requiring equal lengths (see ``run``).
        """
        return str(self.grid_type).upper() == 'I'

    def _warn_if_below_ray_validity(self, env, source) -> None:
        """Warn when the water column spans too few wavelengths for ray theory.

        uacpy's model-validity table (``docs/models/README.md``,
        ``docs/models/bellhop.md``) marks ``D/lambda < 5`` ✗ — rays are the
        wrong tool, take a modal or wavenumber-integral solver — but until now
        nothing in the code said so. The round-22 cold-start audit measured an
        80 m isovelocity guide at 20 Hz (``D/lambda = 1.07``): Bellhop read
        10.1 to 17.6 dB below Kraken over 1-10 km, with no warning at all.

        ``c`` is the sea-surface speed of the first profile, the same reference
        speed :meth:`_run_broadband` derives from ``env.ssp.data``. The
        *lowest* frequency in the source's band binds, because the ray
        approximation fails at the LONGEST wavelength — the opposite end from a
        resolution criterion such as ``_segmentation._highest_frequency``,
        which takes the highest.

        The 5-20 cross-check band stays silent: the table asks for a second
        opinion there, not for a different model.
        """
        speeds = np.asarray(env.ssp.data, dtype=float)
        freqs = np.atleast_1d(np.asarray(source.frequencies, dtype=float))
        if speeds.size == 0 or freqs.size == 0:
            return
        # ``ssp.data`` is (n_depths, n_ranges) in C order, so ``flat[0]`` is
        # ``[0, 0]`` — the surface row of the first profile — for the 1-D and
        # 2-D cases alike.
        c = float(speeds.flat[0])
        f = float(np.min(freqs))
        depth = float(env.depth)
        # A diagnostic never decides whether a run happens: an environment
        # whose depth or speed will not reduce to a positive finite number is
        # left to the deck-validity guards that do reject it.
        if not (np.isfinite(depth) and depth > 0.0):
            return
        if not (np.isfinite(c) and c > 0.0) or not (np.isfinite(f) and f > 0.0):
            return
        d_over_lambda = depth * f / c
        if d_over_lambda >= _RAY_VALIDITY_D_OVER_LAMBDA:
            return
        key = (round(depth, 6), round(f, 9))
        if key in _WARNED_RAY_VALIDITY:
            return
        _WARNED_RAY_VALIDITY.add(key)
        warnings.warn(
            f"{self.model_name}: the water column spans D/lambda = "
            f"{d_over_lambda:.2f} wavelengths ({depth:.0f} m at {f:g} Hz, "
            f"c = {c:.0f} m/s), below the D/lambda >= "
            f"{_RAY_VALIDITY_D_OVER_LAMBDA:g} floor uacpy's model-validity "
            f"table sets for ray theory (docs/models/README.md). Ray theory is "
            f"asymptotic in frequency and carries no error bound here — "
            f"measured 10 to 18 dB against Kraken on an 80 m guide at 20 Hz. "
            f"Use Kraken (normal modes) or Scooter / OASES (wavenumber "
            f"integral), or cross-check this run against one.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )

    def _warn_if_fan_misses_receivers(self, source, receiver) -> None:
        """Warn when a receiver's direct path lies outside the launch fan.

        ``angleMod.f90:58-61`` fills the fan strictly between the two ``alpha``
        values, and BELLHOP's only under-resolution diagnostic
        (``bellhop.f90:252-258``) tests the beam COUNT — never whether the span
        reaches the receivers. So a geometry needing a steeper launch than the
        fan carries loses those paths silently.

        The direct-path launch angle to a receiver at range ``r`` and depth
        ``zr`` from a source at ``zs`` is ``atan2(zr - zs, r)``, which is steep
        for a receiver close in range and far in depth. Measured on a 100 m
        isovelocity guide, source 10 m, receiver 90 m at 2 kHz with the angular
        resolution matched so only the span differs: at r = 10 m the direct
        path needs 82.9 deg and the default +/-80 deg fan reads 64.59 dB
        against 39.54 dB for +/-89.9 deg, a 25.05 dB error; where the required
        angle is inside the fan (r >= 50 m, needing <= 58 deg) the two agree to
        0.33 dB. Proximity to the edge also costs something — 76 deg against an
        80 deg edge is 3.71 dB — so clearing this check is necessary, not
        sufficient.
        """
        fan_lo, fan_hi = float(min(self.alpha)), float(max(self.alpha))
        zs = np.atleast_1d(np.asarray(source.depths, dtype=float))
        zr = np.atleast_1d(np.asarray(receiver.depths, dtype=float))
        rr = np.atleast_1d(np.asarray(receiver.ranges, dtype=float))
        rr = rr[rr > 0.0]                      # r = 0 carries no ray path
        if not zs.size or not zr.size or not rr.size:
            return
        # Whether ANY pair misses is decided by the extremes of the angle over
        # the (zs, zr, rr) box, and those sit at its corners: atan2(d, r) rises
        # with d at fixed r, and is monotone in r at fixed d (falling for
        # d > 0, rising for d < 0). So four angles settle the common case
        # without the (n_sz, n_rz, n_rr) cube the check used to build — 1.5 GiB
        # and 1.3 s of float64 at 20 x 500 x 10000, to emit nothing.
        d_ends = np.array([zr.min() - zs.max(), zr.max() - zs.min()])
        r_ends = np.array([rr.min(), rr.max()])
        corner = np.degrees(np.arctan2(d_ends[:, None], r_ends[None, :]))
        if corner.min() >= fan_lo and corner.max() <= fan_hi:
            return
        # The corners cannot answer the other two: a count is not an extremal
        # quantity, and the steepest angle among the MISSES is a corner only
        # when the fan spans the horizontal (measured wrong in 6,369 of 99,698
        # random cases on fans that exclude 0, which alpha=(17, 74) is —
        # :758 requires only alpha_lo < alpha_hi). Both come exactly off the
        # sorted range axis instead.
        n_out, worst = _fan_miss_count_and_worst(zs, zr, rr, fan_lo, fan_hi)
        warnings.warn(
            f"{self.model_name}: {n_out} of {zs.size * zr.size * rr.size} "
            f"source/receiver pairs need a direct-path launch angle outside "
            f"alpha = [{fan_lo:g}, {fan_hi:g}] deg — the steepest is "
            f"{worst:.1f} deg. angleMod.f90:58-61 launches nothing beyond the "
            f"fan and bellhop.f90:252-258 only checks the beam count, so"
            f" those "
            f"receivers lose their direct path with no diagnostic from the "
            f"binary. Widen alpha (e.g. alpha=(-89.9, 89.9)) if the near"
            f" field "
            f"matters.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)

    def _reject_precalc_boundary(self, env) -> None:
        """A ``'precalc'`` boundary has no reflection branch in BELLHOP.

        ``ReadEnvironmentBell.f90:459`` accepts the ``'P'`` option letter and
        prints "reading PRECALCULATED IRC", but ``bellhop.f90:681``'s
        ``SELECT CASE ( HS%BC )`` implements only ``'R'``, ``'V'``, ``'F'``
        and ``'A'``/``'G'`` — there is no ``'P'`` case, so the run fails with
        a bare exit code instead of naming the boundary. ``bounce.py``'s
        module docstring already records this; the guard belongs here,
        where the deck is built.
        KRAKENC and SCOOTER do read ``.irc``.
        """
        for boundary, where in ((env.bottom, 'bottom'), (env.surface, 'surface')):
            if boundary is None:
                continue
            props = getattr(boundary, 'properties', None) or getattr(
                boundary, 'columns', None) or []
            # A Surface entry is a BoundaryProperties; a Bottom entry is a
            # SeabedColumn whose acoustic_type lives on ``.halfspace``.
            types = {getattr(getattr(p, 'halfspace', p), 'acoustic_type', None)
                     for p in props}
            types.add(getattr(boundary, 'acoustic_type', None))
            if 'precalc' in types:
                raise UnsupportedFeatureError(
                    'Bellhop',
                    f"a {where} with acoustic_type='precalc' — "
                    f"ReadEnvironmentBell.f90:459 accepts the 'P' option but "
                    f"bellhop.f90:681 has no 'P' reflection branch, so the run "
                    f"fails without naming the cause. Use acoustic_type='file' "
                    f"with a .brc table instead",
                    ['KrakenC', 'Scooter'])

    def _check_beam_count_supports_run_mode(self, run_mode) -> None:
        """Reject a beam fan too sparse to carry an influence calculation.

        ``bellhop.f90:176-178`` leaves ``Angles%Dalpha = 0`` when
        ``Nalpha == 1``, so ``q0 = c / Dalpha`` gives every beam zero width
        and the influence sum contributes nothing: the field comes back
        all-NaN at exit 0, with no diagnostic. That is the whole of the
        degenerate case — a two-beam fan has a finite ``Dalpha`` and returns
        a field wherever its rays reach the receivers (see
        :data:`_MIN_INFLUENCE_BEAMS`); it is under-resolved, not degenerate,
        and an under-resolved fan is the caller's choice to make.

        Ray and eigenray runs are unaffected — ``bellhop.f90:288`` skips the
        influence step, and a single traced ray is a legitimate request.
        """
        if run_mode in (RunMode.RAYS, RunMode.EIGENRAYS):
            return
        n = self.n_beams
        if n is None or n == 0 or n >= _MIN_INFLUENCE_BEAMS:
            return                                  # 0 = Bellhop auto-picks
        raise ConfigurationError(
            f"Bellhop(n_beams={n}) cannot produce a {run_mode.name} field: "
            f"bellhop.f90:176-178 leaves the angular spacing Dalpha at 0 for a "
            f"single beam, so every beam has zero width and the influence sum "
            f"is empty — the run exits 0 with an all-NaN field.",
            remediation=f"Use n_beams >= {_MIN_INFLUENCE_BEAMS} (or 0 to let "
                        f"Bellhop choose), or run_mode=RunMode.RAYS to trace "
                        f"individual rays.")

    def _check_beam_type_supports_run_mode(self, run_mode) -> None:
        """Reject ``beam_type`` × ``run_mode`` pairs the influence routine cannot
        run — see :data:`_BEAM_TYPE_RUN_TYPES` for the enumeration and its
        authority. Untrapped, the arrivals/eigenray pairs corrupt the heap
        (``bellhop.exe`` aborts with SIGABRT) and ``beam_type='S'`` with an
        incoherent run returns a field that is mostly NaN."""
        letter = _RUN_MODE_TO_INFLUENCE_LETTER.get(run_mode)
        if letter is None or letter in _BEAM_TYPE_RUN_TYPES[self.beam_type]:
            return
        usable = sorted(
            mode.name for mode, code in _RUN_MODE_TO_INFLUENCE_LETTER.items()
            if code in _BEAM_TYPE_RUN_TYPES[self.beam_type])
        raise ConfigurationError(
            f"Bellhop(beam_type={self.beam_type!r}) cannot run "
            f"{run_mode.name}: {_INFLUENCE_ROUTINE[self.beam_type]} "
            f"(Bellhop/influence.f90) has no RunType(1:1)=='{letter}' branch, "
            f"so the run either corrupts the pressure matrix or returns NaN.",
            remediation=f"Use beam_type='G' or 'B' for {run_mode.name}, or "
                        f"keep beam_type={self.beam_type!r} and one of "
                        f"{usable}.",
        )

    def _check_beam_type_supports_receiver_grid(self, receiver) -> None:
        """Reject receiver grids the influence routine indexes incorrectly — see
        :data:`_IRREGULAR_GRID_BEAM_TYPES` and
        :data:`_UNIFORM_RANGE_BEAM_TYPES`. Both failures are silent and
        plausible-looking: up to 30 dB off with no NaN and no warning."""
        if (str(self.grid_type).upper() == 'I'
                and self.beam_type not in _IRREGULAR_GRID_BEAM_TYPES):
            raise ConfigurationError(
                f"Bellhop(grid_type='I', beam_type={self.beam_type!r}) would "
                f"evaluate every paired receiver at receiver.depths[0]: "
                f"bellhop.f90:202-204 pins NRz_per_range to 1 for an irregular "
                f"grid and {_INFLUENCE_ROUTINE[self.beam_type]} indexes the "
                f"depth by the depth-loop counter.",
                remediation="Use beam_type='G' or 'B' with grid_type='I', or "
                            "grid_type='R' for a rectilinear grid.",
            )
        ranges = np.atleast_1d(receiver.ranges)
        if self.beam_type in _UNIFORM_RANGE_BEAM_TYPES and ranges.size == 1:
            raise ConfigurationError(
                f"Bellhop(beam_type={self.beam_type!r}) cannot use a single "
                f"receiver range: {_INFLUENCE_ROUTINE[self.beam_type]} clamps "
                f"the receiver index to Pos%NRr (influence.f90:339,351), so "
                f"irA == irB at every step and influence.f90:354 skips the "
                f"whole ray — the run exits 0 with an all-NaN field, zero "
                f"eigenrays and zero arrivals.",
                remediation="Use beam_type='G' or 'B', which walk the range "
                            "index with a bracket test, or give "
                            "receiver.ranges at least two equally spaced "
                            "entries.",
            )
        if (self.beam_type in _UNIFORM_RANGE_BEAM_TYPES and ranges.size > 2
                and not equally_spaced(np.asarray(ranges, dtype=float))):
            raise ConfigurationError(
                f"Bellhop(beam_type={self.beam_type!r}) requires equally "
                f"spaced receiver.ranges: {_INFLUENCE_ROUTINE[self.beam_type]} "
                f"forms the range index by dividing by Pos%Delta_r, which "
                f"SourceReceiverPositions.f90:160 sets from the last gap "
                f"alone ({float(ranges[-1] - ranges[-2]):.6g} m here against a "
                f"first gap of {float(ranges[1] - ranges[0]):.6g} m).",
                remediation="Use np.linspace for receiver.ranges, or "
                            "beam_type='G', 'B' or 'S', which take an "
                            "arbitrary range vector.",
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
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
        return path

    def _run_eigenrays_multi_depth(self, env, source, receiver, run_mode,
                                   frequencies, source_waveform, sample_rate,
                                   output_duration):
        """EIGENRAYS with multiple source depths: the ``.ray`` file carries no
        per-source boundary. Its header holds only the counts ``Pos%NSz`` and
        ``Angles%Nalpha`` (``Bellhop/ReadEnvironmentBell.f90:559-560``) and each
        record is a take-off angle plus a point list
        (``Bellhop/WriteRay.f90:41-46``); on the eigenray path ``WriteRay2D``
        fires from ``ApplyContribution`` only where a ray reaches a receiver
        (``Bellhop/influence.f90:633-635``), so a source depth contributes a
        data-dependent number of records and those counts cannot split the file.
        Loop one run per source depth in Python and stack."""
        slabs = []
        for sd in source.depths:
            single = Source(
                depths=float(sd),
                frequencies=source.frequencies,
                source_type=source.source_type,
                beam_pattern=source.beam_pattern,
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
                                    frequencies, source_waveform, sample_rate,
                                    output_duration):
        """A layered bottom can't be represented in Bellhop's single-halfspace
        ``.env``. With ``auto_bounce`` (default) route through BOUNCE for an
        exact reflection-coefficient table over the layer stack and return
        that Result; otherwise warn that the stack will be collapsed to a
        halfspace and return ``None`` to continue the native run. A
        non-layered bottom — elastic included — always returns ``None``:
        ``bellhop.f90:694-712`` computes the exact acousto-elastic halfspace
        reflection coefficient natively (per range node on a range-dependent
        bottom), which a BOUNCE pass would only degrade by collapsing the
        range axis to one column."""
        if not env.bottom.is_layered:
            return None
        kind = ('layered bottom (elastic)' if env.has_elastic_bottom
                else 'layered bottom')
        if self.auto_bounce:
            warnings.warn(
                f"{self.model_name}: env.bottom is a {kind}; auto-routing "
                f"through BOUNCE to derive a reflection-coefficient table. "
                f"BOUNCE is range-independent — Bounce's collapse policy "
                f"reduces the env (default: bottom_range='median', layer "
                f"stack kept). "
                f"Pass ``Bellhop(auto_bounce=False)`` to skip the auto-route "
                f"(Bellhop will then collapse the layer stack to a halfspace "
                f"via its own collapse policy).",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
            return self.run_with_bounce(
                env, source, receiver,
                run_mode=run_mode,
                frequencies=frequencies,
                source_waveform=source_waveform,
                sample_rate=sample_rate,
                output_duration=output_duration,
            )
        # auto_bounce=False: fall through — the layer stack is collapsed to a
        # halfspace by ``_project_environment`` (collapse policy), and the
        # ray tracer reflects off that single halfspace (exactly, elastic or
        # fluid — what is lost is the stack's interference structure).
        warnings.warn(
            f"{self.model_name}: env.bottom is a {kind}; auto_bounce=False → "
            f"collapsing the layer stack to a halfspace via the model's "
            f"collapse policy. The stack's interference structure is lost "
            f"from the bottom reflection. Set auto_bounce=True (default) or "
            f"call run_with_bounce() to keep the layers via a BOUNCE table.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
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
    ) -> Union[Result, ResultStack]:
        """
        Run Bellhop simulation

        Parameters
        ----------
        env, source, receiver : see ``PropagationModel.run``.
        run_mode : RunMode, optional
            Which Bellhop mode to run. One of ``RunMode.COHERENT_TL``,
            ``INCOHERENT_TL``, ``SEMICOHERENT_TL``, ``RAYS``, ``EIGENRAYS``,
            ``ARRIVALS``, ``TIME_SERIES``. Defaults to ``COHERENT_TL``.

            For ``INCOHERENT_TL`` / ``SEMICOHERENT_TL`` the returned
            ``Field.data`` holds complex pressure (Pa re 1 m) as read from
            the ``.shd`` — its magnitude is the incoherent / semicoherent
            beam sum, its phase an artefact of AT's storage with no phase
            reference stamped. Kraken's ``INCOHERENT_TL`` instead stores
            real dB TL in ``.data``; the two engines agree on ``.db``,
            which is the uniform cross-engine surface for magnitude-sum
            results.
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
        result : Result or ResultStack
            Simulation results — a ``ResultStack`` stacked over
            ``source_depth`` for an ``EIGENRAYS`` run over a multi-depth
            ``Source`` (DOCUMENTATION.md §ResultStack), one of the typed
            :mod:`uacpy.core.results` subclasses otherwise. ``ResultStack``
            is not a ``Result`` subclass, so a caller that annotates the
            result has to name both.
        """
        self._require_run_triple(env, source, receiver)
        # ── Resolve run_mode → internal single-char Bellhop code ────────
        run_mode = self._resolve_run_mode(run_mode)
        # Before the backend is chosen, so all three backends agree: the ports
        # implemented branches the Fortran lacks, which otherwise makes the
        # answer depend on which binaries install.sh built.
        self._reject_precalc_boundary(env)
        self._warn_if_fan_misses_receivers(source, receiver)
        self._check_beam_type_supports_run_mode(run_mode)
        self._check_beam_count_supports_run_mode(run_mode)
        if run_mode != RunMode.RAYS:      # bellhop.f90:288 skips influence
            self._check_beam_type_supports_receiver_grid(receiver)

        # Bellhop never writes the r=0 column (no ray travels zero
        # distance), so it comes back as NaN no-data cells on the TL grids
        # and as zero-arrival cells — NaN after synthesis — on the
        # broadband routes. Newcomers using ``np.linspace(0, R, N)`` for
        # ``receiver.ranges`` hit a wall of NaN at r=0 and rightly wonder
        # what is wrong. Warn on every run, matching the r=0 cadence of the
        # other engines; the check sits ahead of the broadband dispatch so
        # both routes pass it (the nested ARRIVALS run warns no second
        # time — ARRIVALS is not in the gated set — and a layered TL run
        # about to auto-route through BOUNCE defers to the re-entrant run's
        # own pass through this gate; the broadband routes never re-enter
        # with a gated mode, so they warn here regardless).
        _defers_to_bounce_rerun = (
            self.auto_bounce and env.bottom.is_layered
            and run_mode not in (RunMode.BROADBAND, RunMode.TIME_SERIES)
        )
        if (
            run_mode in (RunMode.COHERENT_TL, RunMode.INCOHERENT_TL,
                         RunMode.SEMICOHERENT_TL, RunMode.BROADBAND,
                         RunMode.TIME_SERIES)
            and len(receiver.ranges) > 0
            and float(receiver.ranges[0]) == 0.0
            and not _defers_to_bounce_rerun
        ):
            warnings.warn(
                f"{self.model_name}: receiver.ranges starts at r=0 m. "
                f"Bellhop writes no data there (no ray travels zero "
                f"distance), so that column is NaN. Start ranges at a "
                f"small positive value (e.g. ``np.linspace(eps, R, N)``) "
                f"to avoid surprise.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )

        # Ray-theory validity, on the same gated set: the D/lambda floor is a
        # statement about the FIELD the beam sum produces, so RAYS / EIGENRAYS /
        # ARRIVALS — geometry, not a field — stay silent, and the nested
        # ARRIVALS pass a broadband run makes does not warn a second time.
        if run_mode in (RunMode.COHERENT_TL, RunMode.INCOHERENT_TL,
                        RunMode.SEMICOHERENT_TL, RunMode.BROADBAND,
                        RunMode.TIME_SERIES):
            self._warn_if_below_ray_validity(env, source)

        if run_mode in (RunMode.TIME_SERIES, RunMode.BROADBAND):
            # Both routes go through the arrivals → H(f) pipeline. Without
            # source_waveform → Field; with it → Field (1×1 grid).
            source_waveform = self._require_timeseries_signal(
                run_mode, source_waveform, sample_rate)
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

        # Multi-source-depth EIGENRAYS: ``WriteRay2D`` fires once per
        # ray-meets-receiver (influence.f90:633-635), so the number of .ray
        # records a source depth contributes is data-dependent and the header
        # counts give no parseable per-source boundary. A RAYS run instead
        # writes exactly Nalpha records per source depth (bellhop.f90:288-289,
        # inside the SourceDepth / DeclinationAngle loops at :236, :262), so it
        # — like TL and ARRIVALS — splits at the reader level from the single
        # binary call. Loop in Python for this one mode.
        if (
            run_mode == RunMode.EIGENRAYS
            and len(np.atleast_1d(source.depths)) > 1
        ):
            return self._run_eigenrays_multi_depth(
                env, source, receiver, run_mode,
                frequencies, source_waveform, sample_rate, output_duration)

        run_type = _RUN_MODE_TO_BELLHOP_TYPE[run_mode]

        # ── Bottom physics: BOUNCE for what the ray tracer cannot carry ──
        #
        # Auto-route through BOUNCE only for layered columns — Bellhop has
        # no multi-medium .env format, so without BOUNCE the layers are
        # silently lost. A halfspace with non-zero shear stays native:
        # the writer emits cs/alpha_s on the 'A' line (or per-range on the
        # long .bty) and bellhop.f90:694-712 evaluates the exact
        # acousto-elastic reflection coefficient at every boundary hit.
        # BOUNCE itself is range-independent; the spawned Bounce instance
        # collapses any range-dependent env via its own collapse policy
        # (Bounce default ``bottom_range='median'`` → median range, layer
        # stack kept since BOUNCE consumes layered columns natively).
        # Pass ``collapse={...}`` to Bellhop to override;
        # ``Bellhop.run_with_bounce(...)`` is the explicit form for
        # users who want to control the BOUNCE constructor.
        routed = self._maybe_route_through_bounce(
            env, source, receiver, run_mode,
            frequencies, source_waveform, sample_rate, output_duration)
        if routed is not None:
            return routed

        # ── Resolve SSP interpolation, project the env, validate ────────
        from uacpy.io.oalib_writer import resolve_ssp_interp
        effective_interp = resolve_ssp_interp(env, self.interp_ssp)
        interp_for_writer = self.interp_ssp
        if self.interp_ssp is None:
            self._log(
                f"interp_ssp auto-picked = {effective_interp!r} "
                f"(env.has_range_dependent_ssp={env.has_range_dependent_ssp})"
            )
        if effective_interp == 'quad' and not env.has_range_dependent_ssp:
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
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )
            effective_interp = fallback
            interp_for_writer = fallback

        # When ``interp_ssp`` is pinned to a non-quad scheme, the RD-SSP
        # capability flag is False and ``_project_environment`` collapses the
        # SSP to 1-D (one UserWarning per dropped feature). On the 'quad' /
        # auto path the flag is True and the 2-D profile is written verbatim.
        env = self._project_environment(env)
        self.validate_inputs(env, source, receiver, run_mode=run_mode)

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
        # ── Write the deck, run the binary, read the one output it wrote ──
        fm = self._setup_file_manager()

        extra_writer_kwargs = {
            'interp_ssp': interp_for_writer,
            'interp_bathymetry': self.interp_bathymetry,
            'interp_altimetry': self.interp_altimetry,
        }

        try:
            base_name = 'model'
            env_file = fm.get_path(f'{base_name}.env')
            self._log(f"Writing environment file: {env_file}")

            # Bellhop reads <base>.sbp when RunType position 3 is '*'.
            use_sbp = source.beam_pattern is not None
            if use_sbp:
                sbp_dest = env_file.with_suffix('.sbp')
                self._check_beam_pattern_spans_the_fan(source.beam_pattern)
                stage_source_beam_pattern(source.beam_pattern, sbp_dest)
                self._log(f"Wrote source beam pattern: {sbp_dest}")

            write_bellhop_env_file(
                filepath=env_file,
                env=env,
                source=source,
                receiver=receiver,
                run_type=run_type,
                beam_type=self.beam_type,
                # Letter lands in RunType(4:4), ReadEnvironmentBell.f90:398-
                # 406; spec.source_types keeps 'scaled' out of this deck.
                source_type=_SOURCE_TYPE_CODE[source.source_type],
                grid_type=self.grid_type,
                verbose=self.verbose,
                n_beams=self.n_beams,
                alpha=self.alpha,
                step=self._resolve_step(env),
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

            return self._read_and_assemble(
                env, source, receiver, fm, base_name, run_type)

        finally:
            fm.finish()

    def _read_and_assemble(self, env, source, receiver, fm, base_name,
                           run_type):
        """Read the output ``_BELLHOP_OUTPUT[run_type]`` names and turn it into
        the tagged :class:`Result` the caller gets.

        Split from :meth:`run` so the deck-write / launch half of the run does
        not share a scope with the phase correction, the depth-axis restore
        and the stack-aware tagging that follow it. Every branch here reads
        the file the binary has already written; nothing launches anything.
        """
        output_key, output_suffix, reader = _BELLHOP_OUTPUT[run_type]
        # A missing or empty output means the binary died silently; the
        # raised error carries the .prt tail with the actual cause.
        output_file = self._require_output(
            [fm.get_path(f'{base_name}{output_suffix}')],
            what=f'a {run_type} output ({output_suffix})',
            prt_base=base_name, work_dir=fm.work_dir,
        )
        # The .arr header reports the full ``Pos%NRz``
        # (``ReadEnvironmentBell.f90:591``) while its body carries only
        # ``NRz_per_range`` depth blocks — 1 for an irregular grid
        # (``bellhop.f90:202-206,329``, ``ArrMod.f90:101-102``). Nothing in
        # the file distinguishes the two, so the reader has to be told.
        result = (reader(output_file, grid_type=self.grid_type)
                  if run_type == 'A' else reader(output_file))

        # AT's ScalePressure (influence.f90:757-795) carries const = -1
        # into the point-source branch (factor = const/sqrt(r)), so the
        # .shd field is inverted relative to the e^{i(wt-kr)} convention
        # Kraken and Scooter report. Undo it here so every uacpy model
        # shares one phase reference. The line-source branch
        # (factor = -4*sqrt(pi)*const) already cancels the sign, and the
        # arrivals path computes its own positive factor
        # (ArrMod.f90:103-111), so only the line source's
        # _LINE_SOURCE_PHASE is left to apply there.
        # Measured against the exact 2-D solution that correction is pi/4
        # to 0.01 deg, and once applied the line-source residual equals
        # the point-source beam bias exactly (4.78 deg vs 4.79 deg).
        _shd_phase = {'point': -1.0 + 0j,
                      'line': np.exp(1j * _LINE_SOURCE_PHASE)}
        if run_type in ('C', 'I', 'S'):
            _corr = _shd_phase[source.source_type]
            for _slab in (result.slabs
                          if isinstance(result, ResultStack) else [result]):
                _slab.data = _slab.data * _corr

            # BELLHOP clamps any receiver below the deck's bottom
            # boundary onto it (misc/SourceReceiverPositions.f90:136-139)
            # and the .shd then carries the clamped depth axis with the
            # boundary row repeated — no field is evaluated at the asked
            # depth. Restore the requested depth axis with NaN there,
            # then NaN below the local seafloor too (a range-dependent
            # .bty leaves sub-seafloor receivers above the deck depth
            # unclamped but ray-free), matching the RAM / Scooter /
            # SPARC below-domain convention. The irregular grid
            # (RunType(5:5)='I') carries no depth axis and is left as
            # read.
            def _restore_depths_and_mask(slab):
                if list(slab.coords) != ['depth', 'range']:
                    return slab
                slab = self._mask_unresolvable_depths(
                    slab, receiver, float(env.depth))
                return slab.mask_below_seafloor(env.bathymetry)

            if isinstance(result, ResultStack):
                result.slabs = [_restore_depths_and_mask(s)
                                for s in result.slabs]
            else:
                result = _restore_depths_and_mask(result)

        # The .ray header records only NSz (count), not Pos%Sz; the
        # reader returns the stack with a placeholder coordinate.
        # Replace it with the real source.depths order (Bellhop's
        # SourceDepth loop iterates Pos%Sz in writer order).
        if isinstance(result, ResultStack):
            real_sds = np.atleast_1d(np.asarray(source.depths, dtype=float))
            if real_sds.size == result.n_slabs:
                # The reader names this axis 'source_index' because the
                # .ray file carries only the order; once the real depths
                # are substituted the axis is a source depth, and the name
                # has to say so for .at(source_depth=...) to reach it.
                result.coordinate = real_sds
                result.coordinate_name = 'source_depth'

        if run_type in ('R', 'E'):
            # The .ray file format is identical for fan and
            # eigenray runs; only the wrapper knows which one
            # produced it. Same goes for the receiver geometry.
            rcv_d = np.atleast_1d(np.asarray(receiver.depths, dtype=float))
            rcv_r = np.atleast_1d(np.asarray(receiver.ranges, dtype=float))
            ray_slabs = (
                result.slabs if isinstance(result, ResultStack) else [result]
            )
            for slab in ray_slabs:
                slab.is_eigen = (run_type == 'E')
                slab.receiver_depths = rcv_d
                slab.receiver_ranges = rcv_r

        # ── Stamp identity and provenance onto every slab ────────────
        f0 = np.atleast_1d(np.asarray(
            float(np.atleast_1d(source.frequencies)[0]), dtype=float,
        ))
        slabs_to_set = (
            result.slabs if isinstance(result, ResultStack) else [result]
        )
        for i, slab in enumerate(slabs_to_set):
            self._stamp_result(
                slab, source, backend=self.version, frequencies=f0,
                # Only coherent pressure carries a phase to reference;
                # incoherent/semicoherent TL, rays and arrivals do not.
                phase_reference=('travelling_wave'
                                 if run_type == 'C' else None),
            )
            # Each slab of a stack carries its own source depth.
            if isinstance(result, ResultStack):
                slab.source_depths = np.array(
                    [float(result.coordinate[i])], dtype=float,
                )
            self._attach_output_paths(
                slab, fm.work_dir, base_name,
                primary_files=((output_key, output_suffix),),
            )

        self._log("Simulation complete")
        return result


    def run_with_bounce(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        *,
        run_mode: Optional[RunMode] = None,
        c_low: Optional[float] = None,
        c_high: Optional[float] = None,
        rmax: Optional[float] = None,
        frequencies: Optional[np.ndarray] = None,
        source_waveform: Optional[np.ndarray] = None,
        sample_rate: Optional[float] = None,
        output_duration: Optional[float] = None,
    ) -> Union[Result, ResultStack]:
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
        run_mode : RunMode, optional
            Bellhop mode for the second (field) pass; same values ``run()``
            takes. Keyword-only, like every argument after ``receiver``, so
            it cannot bind to a BOUNCE knob by position.
        c_low : float, optional
            Minimum phase velocity for the reflection table (m/s). ``None``
            (default) is resolved by :class:`~uacpy.Bounce` itself, as
            ``min(DEFAULT_C_MIN, min(env.ssp))`` — bounce.htm's "lowest speed
            in the problem" — which keeps the table's grazing wedge intact in
            cold or fresh water too; ``Bounce.run`` rejects a larger value.
        c_high : float, optional
            Maximum phase velocity for the reflection table (m/s). ``None``
            (default) uses ``DEFAULT_C_MAX_UNBOUNDED``, which zeroes BOUNCE's
            ``kMin`` and so covers grazing angles down to 0.
        rmax : float, optional
            Maximum range for angular resolution (m). ``None`` (default) uses
            ``receiver.range_max``, the range the table is propagated to.

        Returns
        -------
        result : Result or ResultStack
            Bellhop simulation results using reflection coefficients. This
            forwards ``run_mode`` to :meth:`run`, so it hands back the same
            two shapes: a ``ResultStack`` for an ``EIGENRAYS`` run over a
            multi-depth ``Source``, one of the typed
            :mod:`uacpy.core.results` subclasses otherwise.
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
        # AT's bounce.htm: "If you are using the reflection coefficient for a
        # coherent TL calculation then RMax should be the maximum range to
        # which you are propagating", and CMax must be ~1e9 "for a full 90
        # degree calculation" — a finite CMax truncates the table at
        # asin(c_water/CMax) and RefCoef.f90:144-149 then silently returns
        # R = 0 for every steeper ray. Both default to the run's own geometry.
        if rmax is None:
            rmax = float(np.max(np.atleast_1d(receiver.ranges)))
        if c_high is None:
            c_high = DEFAULT_C_MAX_UNBOUNDED
        # ``c_low=None`` is forwarded rather than resolved here: Bounce's own
        # ``_resolve_c_low`` reads bounce.htm's rule — "the lowest speed in the
        # problem" — as ``min(DEFAULT_C_MIN, min(env.ssp))``, where this once
        # took the water speed at the seafloor alone and so missed a slower
        # layer higher in the column. The derived value never exceeds the
        # seafloor speed BOUNCE references its angles to (bounce.f90:186-195),
        # so it cannot trip Bounce's own grazing-wedge rejection. Pinned by the
        # test asserting Bounce receives ``c_low=None`` from this call site —
        # it reads the argument handed over, not this text.

        # Bounce validates its arguments in __init__, so a rejected c_low /
        # c_high / rmax raises before the cleanup guard below is entered; the
        # scratch directory this call owns is released first.
        try:
            bounce = Bounce(
                verbose=self.verbose,
                c_low=c_low,
                c_high=c_high,
                rmax=rmax,
                collapse=dict(self._user_collapse) or None,
                timeout=self.timeout,
                work_dir=bounce_work_dir,
                cleanup=False,        # we own bounce_work_dir; cleaned up below
            )
        except Exception:
            bounce_fm.cleanup_work_dir()
            raise
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
            bounce_result.metadata.pop('prt_file', None)
            result.metadata['bounce_result'] = bounce_result
            return result
        finally:
            bounce_fm.cleanup_work_dir()

    def _run_broadband(
        self,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_mode: Optional['RunMode'] = None,
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
            If None, generates n_freqs points over
            [fc*(1-bw_factor/2), fc*(1+bw_factor/2)]. Ignored when
            source_waveform is provided.
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

        # Report the keywords this branch will not consume before spending a
        # ray trace on the run. Branch on the contracted mode, not on the
        # presence of a waveform: BROADBAND must return H(f) even when one is
        # supplied (a waveform is meaningful only for TIME_SERIES synthesis).
        if run_mode == RunMode.BROADBAND:
            if source_waveform is not None:
                warnings.warn(
                    "Bellhop.run(run_mode=BROADBAND) returns the complex "
                    "transfer function H(f); the supplied source_waveform is "
                    "ignored. Use run_mode=TIME_SERIES to synthesise p(t).",
                    UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
                )
            # sample_rate / output_duration size the delay-and-sum time axis,
            # which only the TIME_SERIES branch builds.
            self._warn_ignored_run_kwargs(
                run_mode,
                reason=('BROADBAND returns the transfer function H(f); the '
                        'time-axis keywords apply to TIME_SERIES only'),
                sample_rate=sample_rate,
                output_duration=output_duration,
            )
        if run_mode == RunMode.TIME_SERIES and frequencies is not None:
            warnings.warn(
                "Bellhop.run(run_mode=TIME_SERIES) synthesises p(t) by "
                "delay-and-sum from a single arrivals run at fc; the supplied "
                "frequencies= is ignored. Use run_mode=BROADBAND for an "
                "explicit H(f) grid.",
                UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
            )

        # Step 1: Run Bellhop in arrivals mode. Bellhop traces rays at the
        # single carrier fc, so the arrivals run uses a single-frequency source
        # (ARRIVALS does not accept a multi-frequency band).
        self._log("Running in arrivals mode (broadband path)...")
        arr_source = Source(
            depths=source.depths,
            frequencies=fc,
            source_type=source.source_type,
            beam_pattern=source.beam_pattern,
        )
        arr_field = self.run(env, arr_source, receiver, run_mode=RunMode.ARRIVALS)
        # The synthesised result is built entirely from this inner run, so it
        # inherits its file paths: they are the only handle on the scratch dir
        # that survives when ``cleanup=False``.
        arr_paths = {
            key: value for key, value in arr_field.metadata.items()
            if key.endswith('_file')
        }

        arrivals_by_rcv = arr_field.by_receiver
        rz = arr_field.receiver_depths
        rr = arr_field.receiver_ranges  # in meters

        # ``ArrMod.f90:101-102`` writes ``NRz_per_range`` depth blocks, which
        # ``bellhop.f90:202-206`` sets to 1 for an irregular grid: the entries
        # of that single block are the paired receivers (Rz(i), Rr(i)). The
        # depth axis therefore collapses onto the range axis, and the paired
        # depths ride on ``metadata['receiver_depths']`` — the same shape
        # ``read_shd_file`` gives the TL path.
        irregular = str(self.grid_type).upper() == 'I'
        nrd = 1 if irregular else len(rz)
        nrr = len(rr)

        def _emit(data, axis_name, axis_values, **extra):
            coords = {'range': np.asarray(rr, dtype=float),
                      axis_name: axis_values}
            if irregular:
                extra['receiver_depths'] = np.asarray(rz, dtype=float)
            else:
                coords = {'depth': np.asarray(rz, dtype=float), **coords}
            return (data[0] if irregular else data), coords, extra

        # ArrMod.f90:103-104 scales a line source by a purely real
        # 4*sqrt(pi), so the arrivals need the same exp(-i*pi/4) the .shd
        # path applies.
        arr_phase = (_LINE_SOURCE_PHASE
                     if source.source_type == 'line' else 0.0)

        # ── Path A: time-domain delay-and-sum with source waveform ──
        if run_mode == RunMode.TIME_SERIES:
            self._log(f"Delay-and-sum over {nrd}×{nrr} receiver grid")

            # Lock one clock for the whole grid, spanning every cell's
            # arrivals: a window taken from a single cell covers only that
            # cell's delays, and every farther receiver — whose energy lands
            # later — would convolve to exactly zero. Mirrors delayandsum's
            # own auto-window rule over the global delay span.
            lock_arrivals = arrivals_by_rcv[0][0][0]
            for cell in (
                arrivals_by_rcv[0][ird][irr]
                for ird in range(nrd) for irr in range(nrr)
            ):
                if int(cell.get('n_arrivals', 0)) > 0:
                    lock_arrivals = cell
                    break
            if effective_time_window is None or effective_t_start is None:
                spans = [
                    np.asarray(arrivals_by_rcv[0][ird][irr]['delays'], dtype=float)
                    for ird in range(nrd) for irr in range(nrr)
                    if int(arrivals_by_rcv[0][ird][irr].get('n_arrivals', 0)) > 0
                ]
                if spans:
                    src_duration = len(source_waveform) / float(sample_rate)
                    grid_min = float(min(s.min() for s in spans))
                    grid_max = float(max(s.max() for s in spans))
                    if effective_t_start is None:
                        effective_t_start = max(
                            0.0, grid_min - 0.1 * src_duration)
                    if effective_time_window is None:
                        effective_time_window = (
                            (grid_max - effective_t_start) + 2.0 * src_duration)
            _, t_vec = delayandsum(
                rcv_arrivals=lock_arrivals,
                source_timeseries=source_waveform,
                sample_rate=sample_rate,
                fc=fc,
                time_window=effective_time_window,
                t_start=effective_t_start,
                phase_offset=arr_phase,
            )
            t_start_locked = float(t_vec[0])
            time_window_locked = float(t_vec[-1] - t_vec[0]) + 1.0 / sample_rate
            n_t = len(t_vec)

            data = np.zeros((nrd, nrr, n_t), dtype=float)
            for ird in range(nrd):
                for irr in range(nrr):
                    cell = arrivals_by_rcv[0][ird][irr]
                    if int(cell.get('n_arrivals', 0)) == 0:
                        # No ray reached this cell (shadow zone, or the r=0
                        # column Bellhop never writes): there is no data to
                        # synthesise, so the trace is NaN — the same no-data
                        # convention the TL modes report — not a silent
                        # all-zero record that reads as a real quiet cell.
                        data[ird, irr, :] = np.nan
                        continue
                    rts, _ = delayandsum(
                        rcv_arrivals=cell,
                        source_timeseries=source_waveform,
                        sample_rate=sample_rate,
                        fc=fc,
                        time_window=time_window_locked,
                        t_start=t_start_locked,
                        phase_offset=arr_phase,
                    )
                    # delayandsum may return a slightly different length —
                    # pad/truncate to n_t.
                    m = min(len(rts), n_t)
                    data[ird, irr, :m] = np.asarray(rts[:m], dtype=float)

            data, coords, extra = _emit(data, 'time', t_vec)
            # The stamped frequency axis is the band the synthesised p(t)
            # represents, derived from the (padded) source waveform the same
            # way the IFFT-based engines derive their broadband grid — so a
            # TIME_SERIES result names its band identically across engines.
            # The ray-trace carrier fc stays on metadata['center_frequency'].
            stamp_freqs = self._resolve_time_series_frequencies(
                run_mode, None,
                self._pad_waveform_to_duration(
                    source_waveform, sample_rate, output_duration),
                sample_rate, announce=False)
            field = Field(
                data=data,
                coords=coords,
                phase_reference=PhaseReference.TIME_DOMAIN_NATIVE,
                **self._result_kwargs(
                    source, backend=self.version,
                    frequencies=stamp_freqs if stamp_freqs is not None else fc,
                    dt=1.0 / sample_rate, fs=sample_rate, nt=n_t,
                    t_start=t_start_locked, center_frequency=fc,
                    arrivals_field=arr_field, **arr_paths, **extra,
                ),
            )
            if not irregular:
                field = self._restore_broadband_depth_axis(
                    field, receiver, env)
            return field

        # ── Path B: frequency-domain transfer function ──
        frequencies = self._resolve_broadband_frequencies(
            source, frequencies,
            n_freqs=self.n_freqs, bandwidth_factor=self.bandwidth_factor,
        )
        n_freq = len(frequencies)
        self._warn_if_attenuation_extrapolates(env, frequencies)

        # Build H(d, r, f) for each (receiver_depth, receiver_range).
        # Use first source depth (most common case). Trailing-axis convention.
        H = np.zeros((nrd, nrr, n_freq), dtype=complex)
        for ird in range(nrd):
            for irr in range(nrr):
                rcv_arr = arrivals_by_rcv[0][ird][irr]
                H[ird, irr, :] = self._arrivals_to_tf(
                    rcv_arr, frequencies, phase_offset=arr_phase)

        self._log(f"Built transfer function "
                  f"({nrd} depths x {nrr} ranges x {n_freq} freqs)")

        # Sea-surface sound speed of the first profile: ``ssp.data`` is
        # (n_depths, n_ranges) of speeds alone, the depths living on
        # ``ssp.depths``. Carried on the result for ``Field.to_time_trace`` /
        # ``Field.synthesize_time_series``, which use it as the reference speed
        # that anchors the synthesis window to r/c.
        c0 = float(env.ssp.data[0, 0])

        H, coords, extra = _emit(H, 'frequency', frequencies)
        field = Field(
            data=H,
            coords=coords,
            phase_reference=PhaseReference.TRAVELLING_WAVE,
            **self._result_kwargs(
                source,
                backend=self.version,
                frequencies=frequencies,
                center_frequency=fc,
                arrivals_field=arr_field,
                c0=c0,
                **arr_paths,
                **extra,
            ),
        )
        if not irregular:
            field = self._restore_broadband_depth_axis(field, receiver, env)
        return field

    def _restore_broadband_depth_axis(self, field, receiver, env):
        """Restore the caller's depth axis on a broadband / time-series Field
        and NaN its no-data cells.

        BELLHOP clamps any receiver below the deck's bottom boundary onto it
        (``misc/SourceReceiverPositions.f90:136-139``), so the ``.arr``
        carries the clamped depth axis with the boundary row repeated — no
        arrivals are evaluated at the asked depth. Reattach the requested
        axis with NaN there, then NaN below the local seafloor too (a
        range-dependent ``.bty`` leaves sub-seafloor receivers above the deck
        depth unclamped but ray-free) — the 3-D
        ``(depth, range, time|frequency)`` counterpart of the
        ``_restore_depths_and_mask`` step the TL modes apply in :meth:`run`.
        The irregular grid carries no depth axis and never reaches here.
        """
        field = self._mask_unresolvable_depths(
            field, receiver, float(env.depth))
        bathy = np.asarray(env.bathymetry.to_pairs(), dtype=float)
        depths = np.asarray(field.coords['depth'], dtype=float)
        ranges = np.asarray(field.coords['range'], dtype=float)
        seafloor = np.interp(ranges, bathy[:, 0], bathy[:, 1])
        data = np.array(field.data, copy=True)
        data[depths[:, None] > seafloor[None, :], ...] = np.nan
        field.data = data
        return field

    def _warn_if_attenuation_extrapolates(self, env, frequencies) -> None:
        """Report the volume-attenuation error the single-trace band incurs.

        The arrival set comes from one trace at ``fc``, so ``Im(tau)`` is
        frozen there. ``Step.f90:73`` accumulates ``tau += hw/CMPLX(c, cimag)``
        with ``cimag = alphaT*c**2/omega`` (``misc/AttenMod.f90:113``), so
        ``exp(2*pi*f*Im tau) = exp(-alpha(fc)*s*f/fc)``: the attenuation the
        band carries is forced LINEAR in frequency. Real absorption laws are
        not — Thorp (``AttenMod.f90:94``) goes roughly as ``f**1.8`` — so the
        band edges are over- and under-attenuated. Measured on a deep
        isovelocity guide with ``absorption=Thorp()`` at ``fc = 10`` kHz and
        the default ``bandwidth_factor=0.5`` (exactly 7500-12500 Hz): +2.87 /
        -1.69 dB at 10 km, +5.76 / -3.38 at 20 km, +11.51 / -6.75 at 40 km,
        against an ARRIVALS run at the band-edge frequency itself.
        """
        absorption = getattr(env, 'absorption', None)
        if absorption is None:
            return
        freqs = np.atleast_1d(np.asarray(frequencies, dtype=float))
        if freqs.size < 2 or freqs[0] <= 0.0:
            return
        # The band centre the single trace was run at.
        # `_resolve_broadband_frequencies` spans [fc(1-bw/2), fc(1+bw/2)], so
        # the ARITHMETIC mean of the end points recovers fc exactly; the
        # geometric mean does not (7500-12500 Hz would report 9682, not
        # 10000, and mis-site the straight line the error is measured against).
        fc = 0.5 * (float(freqs[0]) + float(freqs[-1]))

        def alpha_at(f: float) -> float:
            """Volume attenuation (dB/m) at the surface, at one frequency.

            ``alpha_db_per_m`` takes ``(frequency, depths)`` in that order and
            wants a SCALAR frequency; passing the band as an array raises.
            """
            return float(np.atleast_1d(np.asarray(
                absorption.alpha_db_per_m(float(f), 0.0), dtype=float))[0])

        alpha_c = alpha_at(fc)
        if not np.isfinite(alpha_c) or alpha_c <= 0.0:
            return
        # Only the band edges can be furthest from a straight line through fc.
        edges = (float(freqs[0]), float(freqs[-1]))
        err_db_per_km = 0.0
        for f in edges:
            true_alpha = alpha_at(f)
            if not np.isfinite(true_alpha):
                return
            applied = alpha_c * f / fc          # what the frozen Im(tau) gives
            err_db_per_km = max(err_db_per_km,
                                abs(true_alpha - applied) * 1000.0)
        if err_db_per_km < _BROADBAND_ATTEN_WARN_DB_PER_KM:
            return
        warnings.warn(
            f"{self.model_name}: the arrival set is traced once at "
            f"{fc:.4g} Hz, so its volume attenuation is applied linearly in "
            f"frequency across {freqs[0]:.4g}-{freqs[-1]:.4g} Hz "
            f"(Step.f90:73 with cimag = alphaT*c^2/omega, "
            f"misc/AttenMod.f90:113). {type(absorption).__name__} is not "
            f"linear in f, so the band edges are mis-attenuated by up to "
            f"{err_db_per_km:.3g} dB/km of path — about "
            f"{err_db_per_km * 10.0:.3g} dB over a 10 km path. Narrow "
            f"bandwidth_factor, or run each frequency separately, if the band "
            f"edges matter.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)

    @staticmethod
    def _arrivals_to_tf(
        rcv_arrivals: dict,
        frequencies: np.ndarray,
        phase_offset: float = 0.0,
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
        phase_offset : float, optional
            Constant phase (radians) added to every arrival. Carries the
            line-source ``exp(-i*pi/4)``; ``0.0`` (default) for a point
            source.

        Returns
        -------
        H : ndarray
            Complex transfer function, shape (n_freq,). All-NaN when the
            cell has no arrivals (shadow zone, or the r=0 column Bellhop
            never writes): a no-arrival cell carries no model output, and
            ``H = 0`` would read as a real, perfectly quiet channel — the
            same cell the TL modes report as NaN.
        """
        n_arr = rcv_arrivals['n_arrivals']
        if n_arr == 0:
            return np.full(len(frequencies), np.nan, dtype=complex)

        amps = rcv_arrivals['amplitudes']
        phases_deg = rcv_arrivals['phases']
        delays = rcv_arrivals['delays']
        delays_imag = rcv_arrivals['delays_imag']

        phases_rad = np.deg2rad(phases_deg) + phase_offset
        omega = 2.0 * np.pi * frequencies  # (n_freq,)

        # Vectorised over arrivals. For each arrival, tau = Re(tau)+i*Im(tau)
        # gives a phase-shift exp(-i*omega*Re(tau)) and an attenuation
        # exp(omega*Im(tau)); omega is the per-frequency carrier.
        A_complex = np.asarray(amps) * np.exp(1j * phases_rad)        # (n_arr,)
        omega_tau = np.outer(delays, omega)                          # (n_arr, n_freq)
        omega_taui = np.outer(delays_imag, omega)                    # (n_arr, n_freq)
        contrib = A_complex[:, None] * np.exp(omega_taui - 1j * omega_tau)
        return contrib.sum(axis=0)

    def _check_beam_pattern_spans_the_fan(self, pattern) -> None:
        """Require the beam pattern to cover every launch angle in ``alpha``.

        ``bellhop.f90:269-270`` clamps the table index but **not** the
        interpolation weight at ``:273``, so ``Amp0`` at ``:274`` extrapolates
        past both ends of the table. ``misc/beampattern.f90:59`` has already
        converted the levels to linear amplitude by then, so extrapolating a
        roll-off drives ``Amp0`` through zero and negative: the outermost beams
        are launched louder than the pattern declares and phase-inverted, and the
        field comes back partly NaN with no warning from any backend.
        ``third_party/MODIFICATIONS.md`` records that this site is deliberately
        left unclamped so the Fortran, C++ and CUDA backends stay identical,
        which makes this the only available guard.
        """
        from uacpy.io.refl_io import read_source_beam_pattern

        if isinstance(pattern, (str, Path)):
            angles = read_source_beam_pattern(pattern)[:, 0]
        else:
            angles = np.asarray(pattern, dtype=float)[:, 0]
        lo, hi = float(np.min(angles)), float(np.max(angles))
        fan_lo, fan_hi = float(min(self.alpha)), float(max(self.alpha))
        if lo > fan_lo + 1e-9 or hi < fan_hi - 1e-9:
            raise ConfigurationError(
                f"Bellhop: the source beam pattern spans "
                f"[{lo:g}, {hi:g}]° but the launch fan alpha spans "
                f"[{fan_lo:g}, {fan_hi:g}]°. Bellhop extrapolates the pattern "
                f"past its ends on linear amplitude "
                f"(Bellhop/bellhop.f90:273), which inverts the amplitude of "
                f"the outer beams and returns a partly-NaN field with no "
                f"warning.",
                remediation=(
                    f"Extend the pattern to cover [{fan_lo:g}, {fan_hi:g}]° — "
                    f"repeat the edge level to hold it flat — or narrow "
                    f"alpha= to the pattern's own span."
                ),
            )

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

        The three mutually-exclusive outputs are cleared first: a pinned
        work_dir carries the previous run's ``.shd`` / ``.arr`` / ``.ray``,
        and this run writes only one of them.
        """
        cmd = self._build_command(base_name)
        self._run_and_attach_prt(
            cmd, work_dir, base_name,
            stale_outputs=_BELLHOP_OUTPUT_SUFFIXES,
        )

