"""What one Bellhop deck is resolved to before it is written — the carrier
frequency and source, the ray step and box, the SSP interpolation, the
padded receiver ranges, where a band and a pulse came from — and the stage-3
notices of how the run will go."""

from typing import Optional, Tuple

import numpy as np

from uacpy.core.run_settings import RunMode
from uacpy.core.receiver import Receiver
from uacpy.core.source import Source
from uacpy.io.bellhop_writer import (
    BELLHOP_RAY_BOX_FACTOR, bellhop_ray_box,
)
from uacpy.core.deck_limits import NO_RECEIVER_RANGE_FALLBACK_M
from uacpy.io.oalib_writer import resolve_ssp_interp
from uacpy.models._band import _real_pulse
from uacpy.models._defaults import (
    DEFAULT_BROADBAND_BANDWIDTH_FACTOR, DEFAULT_BROADBAND_N_FREQUENCIES,
)
from uacpy.models._budget import memory_budget
from uacpy.models.base import StageInputs
from uacpy.models.bellhop._settings import _BAND_FROM_CALL, _BAND_FROM_SOURCE
from uacpy.models.bellhop._tables import (
    _INFLUENCE_ROUTINE, _RUN_MODE_TO_INFLUENCE_LETTER,
    _UNIFORM_RANGE_BEAM_TYPES, grid_is_paired, writes_origin_column,
)
from uacpy.core.exceptions import NumericsWarning, ValidityWarning
from uacpy.models._notices import message_notice

#: Water depths per wavelength below which uacpy's own model-validity table
#: (``docs/models/README.md``, ``docs/models/bellhop.md``) marks ray theory ✗ —
#: use a modal or wavenumber-integral solver. The number is uacpy's guidance:
#: JKPS §3.4.2 gives no a-priori criterion, only the rule that the wavelength
#: be substantially smaller than the water depth and every other physical
#: scale. The same table calls 5-20 a cross-check band and >= 20 comfortable,
#: so only the ✗ side warns: a cross-check is a suggestion, not a defect.
_RAY_VALIDITY_D_OVER_LAMBDA = 5.0


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


#: Above this launch angle, widening the fan stops being the remedy for a
#: receiver it cannot reach: 90 deg is a vertical ray, so the un-reachable
#: near-field cone shrinks with every extra tenth of a degree but never
#: closes. Below it the fan is the fix — the docstring of
#: ``fan_miss_notice`` measures 25.05 dB of error at +/-80 deg
#: on a geometry needing 82.9 deg.
_FAN_WIDENING_CEILING = 85.0

#: The reference speed angleMod's automatic fan divides by: the module
#: constant ``c0 = 1500.0`` (``Bellhop/angleMod.f90:15``), not a speed read
#: from the environment.
_ANGLEMOD_C0 = 1500.0


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
# depth/50. AT's MunkRot deck writes depth/50 (``tests/MunkRot/Munk.env``,
# 100.0 m in a 5500 m box; its arctic deck goes finer still, 10 m in
# 3800 m); most AT test decks write 0. It costs 0.7 s against 0.6 s on the
# Munk case. Ray sums near a caustic are not monotone in the step, so a
# convergence check remains the only way to be sure: "If you use STEP=0.0
# BELLHOP will use a default step-size and tell you what it picked. You
# should then halve the step size until the results are convergent to your
# required accuracy." (``doc/bellhop.htm``).
_STEP_PER_DEPTH = 50.0


#: Ray-receiver pairs above which an EIGENRAYS run is warned about. The
#: engine writes a whole trajectory once per (beam, receiver) hit
#: (``Bellhop/influence.f90:629-652``, every step at ``WriteRay.f90:28``)
#: and a Gaussian beam hits every receiver its window covers; measured on
#: a 100 m Pekeris guide at 1 kHz with the automatic fan: one receiver,
#: 12.9 MB (3.75 s to read back); 10 x 20 receivers passed 3.1 GB before
#: the run was stopped.
_EIGENRAY_PAIR_WARNING = 20000

#: Bytes a ``.ray`` file spends per trajectory point: ``WriteRay2D`` writes
#: each ``(r, z)`` pair list-directed (``WriteRay.f90:44-46``), measured at
#: 53.0 B per point on gfortran-built bellhop.
_RAY_FILE_BYTES_PER_POINT = 53


#: The run modes that return a pressure field (the beam sum).
_FIELD_RUN_MODES = (RunMode.COHERENT_TL, RunMode.INCOHERENT_TL,
                    RunMode.SEMICOHERENT_TL, RunMode.BROADBAND,
                    RunMode.TIME_SERIES)


def carrier_frequency(source, mode) -> float:
    """The one frequency (Hz) the rays are traced at. A BROADBAND /
    TIME_SERIES Source listing a band is traced at the band centre:
    ``frequencies[0]`` would map a [50, 350] band to fc = 50, the
    footgun ``ram._band.resolve_broadband_grid`` avoids too."""
    freqs = np.atleast_1d(np.asarray(source.frequencies, dtype=float))
    if mode in (RunMode.BROADBAND, RunMode.TIME_SERIES) and freqs.size > 1:
        return float(0.5 * (freqs.min() + freqs.max()))
    return float(freqs[0])


def carrier_source(source, mode, fc) -> Source:
    """The Source the deck is written for: ``source`` itself, or on
    BROADBAND / TIME_SERIES a copy at the one carrier frequency the
    arrivals are traced at (the ``'A'`` deck takes a single
    frequency)."""
    if mode not in (RunMode.BROADBAND, RunMode.TIME_SERIES):
        return source
    return Source(
        depths=source.depths,
        frequencies=fc,
        source_type=source.source_type,
        beam_pattern=source.beam_pattern,
    )


def launch_source(inputs: StageInputs) -> Source:
    """:func:`carrier_source` of the launch's own source."""
    return carrier_source(
        inputs.source, inputs.settings.mode,
        inputs.settings.engine.fc)


def band_origin(mode, source, kwargs, *, n_frequencies,
                bandwidth_factor) -> Optional[str]:
    """BROADBAND only: where the frequency grid came from — a
    ``frequencies=`` passed to the run (``kwargs``, the call's keywords), a
    multi-frequency Source that is the band, or the package's expansion of
    one carrier by the ``n_frequencies`` / ``bandwidth_factor`` knobs.
    ``None`` on every other mode."""
    if mode != RunMode.BROADBAND:
        return None
    if kwargs.get('frequencies') is not None:
        return _BAND_FROM_CALL
    if np.atleast_1d(source.frequencies).size > 1:
        return _BAND_FROM_SOURCE
    n = DEFAULT_BROADBAND_N_FREQUENCIES if n_frequencies is None else n_frequencies
    bw = (DEFAULT_BROADBAND_BANDWIDTH_FACTOR if bandwidth_factor is None
          else bandwidth_factor)
    return f"{n} bins over fc*(1 +/- {bw:g}/2)"


def unused_band_knobs_notice(mode, band, *, n_frequencies,
                             bandwidth_factor) -> Optional[str]:
    """A notice when ``n_frequencies`` / ``bandwidth_factor`` are set but
    the run does not expand a band with them — any mode but BROADBAND, or a
    BROADBAND band the call or the Source names (``band``, from
    :func:`band_origin`) — else ``None``."""
    pinned = [f"{name}={value:g}" for name, value in
              (('n_frequencies', n_frequencies),
               ('bandwidth_factor', bandwidth_factor)) if value is not None]
    if not pinned or (mode == RunMode.BROADBAND
                      and band not in (_BAND_FROM_CALL, _BAND_FROM_SOURCE)):
        return None
    why = (f"the band is the run's {band}" if band is not None
           else f"a {getattr(mode, 'name', mode)} run expands no band")
    return (f"Bellhop({', '.join(pinned)}) is not used: {why}. The knobs "
            f"shape only the band a BROADBAND run expands from a "
            f"one-frequency Source without frequencies=.")


def pulse_samples(mode, kwargs) -> Optional[int]:
    """TIME_SERIES only: the length of the source pulse the call passed
    (``kwargs['source_waveform']``), which the delay-and-sum convolves;
    ``None`` on every other mode, or when the call passed no usable
    pulse."""
    if mode != RunMode.TIME_SERIES or kwargs.get('source_waveform') is None:
        return None
    pulse, _ = _real_pulse(kwargs['source_waveform'])
    if pulse is None:
        return None
    return int(pulse.size)


def resolve_ray_step(env, *, ray_step) -> float:
    """Ray step (m) for the deck: the caller's when pinned, ``0``
    included — the value with which BELLHOP chooses its own ``depth/10``
    (``bellhop.f90:170-174``) — else ``env.depth / _STEP_PER_DEPTH``.
    Without a positive depth to scale by, the deck carries ``0`` and the
    binary chooses."""
    if ray_step is not None:
        return float(ray_step)
    depth = float(env.depth)
    if not np.isfinite(depth) or depth <= 0.0:
        return 0.0
    return depth / _STEP_PER_DEPTH


def ray_step_origin(resolved, *, ray_step) -> str:
    """Where the deck's ray step ``resolved`` (:func:`resolve_ray_step`) came
    from, given the ``ray_step`` knob: the constructor, ``env.depth /
    _STEP_PER_DEPTH``, or the binary's own when there is no positive depth
    to scale by."""
    if ray_step is not None:
        return 'Bellhop(ray_step=…)'
    if resolved:
        return f"env.depth / {_STEP_PER_DEPTH:g}"
    return "the binary's own (no positive env.depth)"


def resolve_ray_box(env, receiver, *, z_box,
                    r_box) -> Tuple[float, str, float, str]:
    """``(z_box, z_origin, r_box, r_origin)``: each pinned value, else the
    box :func:`~uacpy.io.bellhop_writer.bellhop_ray_box` gives the
    caller's environment and receiver — so a deck whose range axis is
    padded traces the same rays as an unpadded one."""
    box_z, box_r = bellhop_ray_box(env, receiver, z_box=z_box,
                                   r_box=r_box)
    if z_box is not None:
        z_origin = 'Bellhop(z_box=…)'
    else:
        z_origin = f'{BELLHOP_RAY_BOX_FACTOR:g} x env.depth'
    if r_box is not None:
        r_origin = 'Bellhop(r_box=…)'
    elif receiver.range_max > 0:
        r_origin = f'{BELLHOP_RAY_BOX_FACTOR:g} x receiver.range_max'
    else:
        r_origin = (f'{NO_RECEIVER_RANGE_FALLBACK_M / 1000.0:g} km '
                    f'(every receiver range is 0)')
    return float(box_z), z_origin, float(box_r), r_origin


def resolve_interp_ssp(env, *, interp_ssp, log) -> Optional[str]:
    """The SSP interpolation the deck writer gets: the caller's as given,
    logging what ``None`` resolves to. A pinned ``'quad'`` on a
    range-independent SSP is written as its one profile held over the ray
    box (:func:`~uacpy.io.bellhop_writer.write_bellhop_env_file`).
    On a pinned non-quad scheme the RD-SSP capability flag is False and
    ``_project_environment`` has already collapsed the SSP to 1-D; on the
    'quad' / automatic path the 2-D profile is written verbatim."""
    if interp_ssp is None:
        log(
            f"interp_ssp auto-picked = "
            f"{resolve_ssp_interp(env, interp_ssp)!r} "
            f"(env.ssp.is_range_dependent={env.ssp.is_range_dependent})"
        )
    return interp_ssp


def pad_receiver_ranges(receiver, run_type, *, beam_type, grid_type,
                        model_name, arrivals_result=False):
    """``(deck_ranges, trim, notice)``: the receiver ranges to write for
    this run, the ``(lo, hi)`` slice that recovers the caller's range
    axis from what the engine wrote — ``(None, None, notice)`` when
    nothing is padded — and the notice of a column the run cannot fill,
    or ``None``.

    ``_UNIFORM_RANGE_BEAM_TYPES`` index the receiver range by division
    and clamp the result to ``[1, NRr]`` (``influence.f90:223-224``, with
    Porter's own ``! should be ", 0 )" ?`` beside it), then step from
    ``irA + 1``: the first column is never written, and ``'C'`` returns
    before the last (``:216``). One extra range at each end, one step
    away so the grid stays uniform, gives the engine somewhere to skip.
    A first range within one step of the source cannot be padded ahead
    of: that column is NaN in a TL or synthesised result
    (:func:`~uacpy.models.bellhop._output.mask_unwritten_first_range`), an
    ARRIVALS result (``arrivals_result``) holds no arrivals there, and the
    run says so.

    EIGENRAYS ('E') skips the same first column but cannot be padded
    around, so its notice says so instead — see below.
    """
    ranges = np.atleast_1d(np.asarray(receiver.ranges, dtype=float))
    if (run_type == 'E'
            and beam_type in _UNIFORM_RANGE_BEAM_TYPES
            and not grid_is_paired(grid_type)
            and ranges.size >= 2):
        # An eigenray run reaches the SAME RcvrRanges loop: 'g' steps from
        # irA + 1 - II at influence.f90:373, after the clamps at :339
        # and :351, and ApplyContribution (:629-652) only picks its
        # CASE('E') branch (:632-636) once that loop has already chosen
        # the column. So the first receiver range is skipped for
        # eigenrays exactly as it is for a TL run — silently, with no
        # NaN to show for it, because a missing eigenray is just an
        # absent record.
        #
        # Padding cannot fix it here the way it does for the TL and
        # arrivals branches: undoing the padding means trimming a range
        # axis, and _BELLHOP_OUTPUT['E'] reads a .ray file, whose records
        # are trajectories carrying no receiver index. An extra column
        # would leave extra eigenrays in the result with nothing to
        # identify them by. So the run says what it will not deliver.
        return None, None, (
            f"{model_name}(beam_type={beam_type!r}): "
            f"{_INFLUENCE_ROUTINE[beam_type]} never fills the first "
            f"receiver-range column (influence.f90:373, after the clamps "
            f"at :339 and :351), so an EIGENRAYS run returns no rays at "
            f"receiver.ranges[0] = {ranges[0]:g} m. Unlike a TL run this "
            f"cannot be padded around — a .ray record carries no receiver "
            f"index, so the padding could not be trimmed off again. "
            f"Use beam_type='G' or 'B', which walk the range index with a "
            f"bracket test, or add a throwaway first range.")
    if (beam_type not in _UNIFORM_RANGE_BEAM_TYPES
            or run_type not in ('C', 'I', 'S', 'A')
            or grid_is_paired(grid_type)
            or ranges.size < 2):
        return None, None, None
    dr = float(ranges[1] - ranges[0])
    lead = ranges[0] - dr > 0.0
    notice = None
    if not lead:
        notice = (
            f"{model_name}(beam_type={beam_type!r}): "
            f"{_INFLUENCE_ROUTINE[beam_type]} never fills the first "
            f"receiver-range column (influence.f90:223-228), and "
            f"receiver.ranges starts at {ranges[0]:g} m, within one range "
            f"step ({dr:g} m) of the source, so no column can be written "
            f"ahead of it: the {ranges[0]:g} m column "
            + ("holds no arrivals, the cell never being evaluated. "
               if arrivals_result else "comes back NaN. ")
            + "Start receiver.ranges more than one step from the source "
              "to have it filled.")
    # Only 'C' also drops the LAST column (influence.f90:216).
    trail = [ranges[-1] + dr] if beam_type == 'C' else []
    padded = ([ranges[0] - dr] if lead else []) + list(ranges) + trail
    lo = 1 if lead else 0
    return np.asarray(padded), (lo, lo + ranges.size), notice


def deck_receiver(receiver, engine):
    """The receiver the deck carries: the caller's, or its ranges padded
    (``engine.deck_ranges``, see :func:`pad_receiver_ranges`)."""
    if engine.deck_ranges is None:
        return receiver
    return Receiver(depths=receiver.depths,
                    ranges=np.asarray(engine.deck_ranges))


def eigenray_beam_count(env, source, receiver, *, n_beams) -> int:
    """The launch count an EIGENRAYS deck gets: ``n_beams`` when set,
    else Bellhop's automatic fan (``Bellhop/angleMod.f90:38-51``):
    ``MAX(INT(0.3*Rmax*f/c0), 300)``, raised to ``pi/atan(D/(10*Rmax))``,
    where ``c0`` is angleMod's constant 1500 m/s (:data:`_ANGLEMOD_C0`),
    whatever the water's speed.
    """
    if n_beams:
        return int(n_beams)
    r_max = float(np.max(np.atleast_1d(receiver.ranges)))
    freq = float(np.atleast_1d(source.frequencies)[0])
    n = max(int(0.3 * r_max * freq / _ANGLEMOD_C0), 300)
    if r_max > 0.0:
        n = max(int(np.pi / np.arctan(float(env.depth) / (10.0 * r_max))), n)
    return n


def eigenray_size_notice(env, source, receiver, *, n_beams, grid_type,
                         ray_step, memory_backed: bool) -> Optional[str]:
    """The stage-3 notice of an EIGENRAYS run whose ``.ray`` output
    grows as receivers x beams x path length, or ``None``.

    In a work directory held in memory (``memory_backed``) the file is
    memory, so :func:`~uacpy.models._budget.memory_budget` weighs its upper
    bound: every ray-receiver pair written as one trajectory of
    ``RMax / step`` points (``WriteRay.f90:28`` keeps every step below
    ``MaxNRayPoints``) of :data:`_RAY_FILE_BYTES_PER_POINT`. Measured on a
    100 m Pekeris guide at 1 kHz, the written file was 1/2.9 to 1/5.3 of
    it, since only some beams reach each receiver — so the bound is
    announced, never refused. On disk, or below the budget's thresholds,
    more than :data:`_EIGENRAY_PAIR_WARNING` pairs is announced."""
    n_rz = len(np.atleast_1d(receiver.depths))
    n_rr = len(np.atleast_1d(receiver.ranges))
    n_receivers = (n_rr if grid_is_paired(grid_type)
                   else n_rz * n_rr)
    pairs = n_receivers * eigenray_beam_count(env, source, receiver,
                                              n_beams=n_beams)
    if memory_backed and float(ray_step) > 0.0:
        r_max = float(np.max(np.atleast_1d(receiver.ranges)))
        points = max(int(r_max / float(ray_step)), 1)
        bound = pairs * points * _RAY_FILE_BYTES_PER_POINT
        notice = memory_budget(
            bound, model_name='Bellhop EIGENRAYS', what='.ray file',
            detail=(
                f"the .ray file of {n_receivers} receiver(s) x "
                f"{pairs // n_receivers} beams = {pairs} ray-receiver pairs "
                f"of up to {points} points each is at most "
                f"{bound / 1024 ** 3:.1f} GiB — an upper bound, counting "
                f"every beam at every receiver, which the file written "
                f"usually stays well under — and goes to a work directory "
                f"held in memory."),
            remediation=(
                "Put the work directory on disk (work_dir=, or TMPDIR), or "
                "use fewer receivers or a small explicit n_beams."),
            upper_bound=True)
        if notice is not None:
            return notice.message
    if pairs <= _EIGENRAY_PAIR_WARNING:
        return None
    return (
        f"Bellhop EIGENRAYS: {n_receivers} receiver(s) x "
        f"{pairs // n_receivers} beams = {pairs} ray-receiver pairs. The "
        f"engine writes a full trajectory for every beam that reaches a "
        f"receiver, so the .ray file can run to gigabytes (a 10 x 20 grid "
        f"at 1 kHz passed 3 GB) in the work directory and take minutes "
        f"to read back. Use fewer receivers or a small explicit n_beams.")


def ray_validity_notice(env, source, *, model_name) -> Optional[str]:
    """The stage-3 notice of a water column that spans too few
    wavelengths for ray theory, or ``None``.

    uacpy's model-validity table (``docs/models/README.md``,
    ``docs/models/bellhop.md``) marks ``D/lambda < 5`` ✗ — rays are the
    wrong tool, take a modal or wavenumber-integral solver — and this
    check is where the code says so. The threshold is uacpy's guidance, not
    a published bound: Jensen et al. (JKPS §3.4.2) state that whether a ray
    solution meets an accuracy requirement cannot be predicted a priori and
    give the often-quoted rule that the wavelength be substantially smaller
    than the water depth and every other physical scale. As an illustration,
    on an 80 m isovelocity guide at 20 Hz (``D/lambda = 1.07``) Bellhop read
    10.1 to 17.6 dB below Kraken over 1-10 km.

    ``c`` is the sea-surface speed of the first profile, the same reference
    speed :func:`~uacpy.models.bellhop._synthesis.synthesise` stamps as
    ``c0``. The *lowest* frequency in the source's band binds, because the ray
    approximation fails at the LONGEST wavelength — the opposite end from a
    resolution criterion such as ``kraken._segments._highest_frequency``,
    which takes the highest.

    The 5-20 cross-check band stays silent: the table asks for a second
    opinion there, not for a different model.
    """
    speeds = np.asarray(env.ssp.sound_speed, dtype=float)
    freqs = np.atleast_1d(np.asarray(source.frequencies, dtype=float))
    if speeds.size == 0 or freqs.size == 0:
        return None
    # ``ssp.sound_speed`` is (n_depths, n_ranges) in C order, so ``flat[0]`` is
    # ``[0, 0]`` — the surface row of the first profile — for the 1-D and
    # 2-D cases alike.
    c = float(speeds.flat[0])
    f = float(np.min(freqs))
    depth = float(env.depth)
    # A diagnostic never decides whether a run happens: an environment
    # whose depth or speed will not reduce to a positive finite number is
    # left to the deck-validity guards that do reject it.
    if not (np.isfinite(depth) and depth > 0.0):
        return None
    if not (np.isfinite(c) and c > 0.0) or not (np.isfinite(f) and f > 0.0):
        return None
    d_over_lambda = depth * f / c
    if d_over_lambda >= _RAY_VALIDITY_D_OVER_LAMBDA:
        return None
    return (
        f"{model_name}: the water column spans D/lambda = "
        f"{d_over_lambda:.2f} wavelengths ({depth:.0f} m at {f:g} Hz, "
        f"c = {c:.0f} m/s), below the D/lambda >= "
        f"{_RAY_VALIDITY_D_OVER_LAMBDA:g} floor uacpy's model-validity "
        f"table sets for ray theory (docs/models/README.md). Ray theory is "
        f"a high-frequency approximation that needs the wavelength "
        f"substantially smaller than the water depth (Jensen et al., "
        f"Computational Ocean Acoustics §3.4.2) and carries no error bound "
        f"here. Use Kraken (normal modes) or Scooter / OASES (wavenumber "
        f"integral), or cross-check this run against one.")


def fan_miss_notice(source, receiver, *, launch_angles, grid_type,
                    model_name) -> Optional[str]:
    """The stage-3 notice of a receiver whose direct path lies outside
    the launch fan, or ``None``.

    ``angleMod.f90:58-61`` fills the fan strictly between the two ``launch_angles``
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
    fan_lo, fan_hi = float(min(launch_angles)), float(max(launch_angles))
    zs = np.atleast_1d(np.asarray(source.depths, dtype=float))
    zr = np.atleast_1d(np.asarray(receiver.depths, dtype=float))
    rr = np.atleast_1d(np.asarray(receiver.ranges, dtype=float))
    if grid_is_paired(grid_type):
        # An 'I' deck pairs sorted depth i with sorted range i
        # (bellhop.f90:202-206): one receiver per index, not the
        # depth x range product, so the angles are evaluated per pair.
        # Unequal lists are refused when the deck is written.
        if zr.size != rr.size:
            return None
        keep = rr > 0.0                    # r = 0 carries no ray path
        zr, rr = zr[keep], rr[keep]
        if not zs.size or not rr.size:
            return None
        needed = np.degrees(np.arctan2(zr[None, :] - zs[:, None],
                                       rr[None, :]))
        outside = (needed < fan_lo) | (needed > fan_hi)
        if not outside.any():
            return None
        n_out = int(outside.sum())
        n_pairs = zs.size * rr.size
        worst = float(needed[outside].flat[
            int(np.argmax(np.abs(needed[outside])))])
    else:
        rr = rr[rr > 0.0]                  # r = 0 carries no ray path
        if not zs.size or not zr.size or not rr.size:
            return None
        # Whether ANY pair misses is decided by the extremes of the angle
        # over the (zs, zr, rr) box, and those sit at its corners:
        # atan2(d, r) rises with d at fixed r, and is monotone in r at
        # fixed d (falling for d > 0, rising for d < 0). So four angles
        # settle the common case without forming the (n_sz, n_rz, n_rr)
        # cube — 1.5 GiB and 1.3 s of float64 at 20 x 500 x 10000, to emit
        # nothing.
        d_ends = np.array([zr.min() - zs.max(), zr.max() - zs.min()])
        r_ends = np.array([rr.min(), rr.max()])
        corner = np.degrees(np.arctan2(d_ends[:, None], r_ends[None, :]))
        if corner.min() >= fan_lo and corner.max() <= fan_hi:
            return None
        # The corners cannot answer the other two: a count is not an
        # extremal quantity, and the steepest angle among the MISSES is
        # a corner only when the fan spans the horizontal (measured
        # wrong in 6,369 of 99,698 random cases on fans that exclude 0,
        # which launch_angles=(17, 74) is — _checks.check_knobs requires
        # only alpha_lo < alpha_hi). Both come exactly off the sorted range
        # axis instead.
        n_out, worst = _fan_miss_count_and_worst(
            zs, zr, rr, fan_lo, fan_hi)
        n_pairs = zs.size * zr.size * rr.size
    # The remedy is DERIVED, not a constant: a fixed suggestion is the
    # fan already in use for anyone running near-vertical, and is
    # narrower than it for alpha beyond +/-89.9 (which is accepted).
    # Which remedy applies is decided by the FAN's width, not by the
    # angle needed: a narrow fan has room to widen whatever the geometry
    # asks for, while one already near vertical does not.
    needed = abs(worst)
    limit = max(abs(fan_lo), abs(fan_hi))
    dz = max(abs(float(zr.max()) - float(zs.min())),
             abs(float(zr.min()) - float(zs.max())))

    def cone(angle):
        """The range inside which a receiver at the largest depth offset
        ``dz`` needs a launch steeper than ``angle``."""
        return dz / np.tan(np.radians(angle)) if 0.0 < angle < 90.0 else 0.0

    where = (f"receivers closer than {cone(limit):.2f} m in range (at the "
             f"{dz:g} m depth offset)")
    if limit >= _FAN_WIDENING_CEILING:
        # Widening chases an asymptote: 90 deg is a vertical ray, so the
        # cone shrinks with every extra tenth of a degree but never
        # closes. Only the receiver ranges close it.
        remedy = (
            f"With launch_angles already at {limit:g} deg, {where} stay in a "
            f"blind cone: 90 deg is a vertical ray, so widening the fan only "
            f"chases an asymptote. Start the receiver ranges beyond "
            f"{cone(limit):.2f} m if the near field matters.")
    else:
        # Margin past the angle needed, because proximity to the edge
        # costs level too — 76 deg against an 80 deg edge is 3.71 dB, see
        # the function docstring.
        wider = min(max(needed, limit) + 5.0, 89.9)
        remedy = (f"With this fan, {where} miss it. Widen launch_angles "
                  f"(e.g. launch_angles=(-{wider:g}, {wider:g})) if the near "
                  f"field matters")
        if float(rr.min()) < cone(wider):
            remedy += (f"; that still leaves a blind cone of "
                       f"{cone(wider):.2f} m, which only receiver ranges "
                       f"beyond it close, since 90 deg is a vertical ray.")
        else:
            remedy += "."
    return (
        f"{model_name}: {n_out} of {n_pairs} "
        f"source/receiver pairs need a direct-path launch angle outside "
        f"launch_angles = [{fan_lo:g}, {fan_hi:g}] deg — the steepest is "
        f"{worst:.1f} deg — and Bellhop launches no ray beyond its fan, so "
        f"those receivers get no direct path, with no word from the binary. "
        f"{remedy}")


#: Fraction of ``fc`` a band may reach from it, on either side, before a
#: Gaussian-beam (``beam_type='B'``) BROADBAND / TIME_SERIES run is told
#: its beam sum is first-order away from ``fc`` (:func:`gaussian_beam_band_notice`).
GAUSSIAN_BEAM_BAND_FRACTION = 0.5


def gaussian_beam_band_notice(run_mode, frequencies, fc, *, beam_type,
                              model_name) -> Optional[str]:
    """The notice for a Gaussian-beam band reaching more than
    :data:`GAUSSIAN_BEAM_BAND_FRACTION` of ``fc`` from it, or ``None``.

    One trace at ``fc`` serves the whole band. ``InfluenceGeoGaussianCart``
    (``influence.f90:561-563``) floors each Gaussian beam's half-width at
    ``pi * c / freq`` with ``freq = fc``, so every other bin is summed with
    ``fc``'s beams. The hat-shaped beams (``'G'``, ``'g'``) carry no
    frequency in their width, and their bins equal a trace at each
    frequency. Measured against such traces, 2000 beams, 30 m of water over
    sand, 100 m-5 km: ``'G'`` 0.04-0.06 dB median |ΔTL| at every bin of a
    50-1950 Hz band (fc 1 kHz, ±95 %); ``'B'`` a mean-energy bias of
    +0.74 / -0.40 dB at ±25 % and +1.41 / -0.88 dB at ±50 % (median |ΔTL|
    1.14 dB at -50 %), and on a 5 km Munk profile, 25-475 Hz, ±0.13 dB at
    ±50 %."""
    if (run_mode not in (RunMode.BROADBAND, RunMode.TIME_SERIES)
            or beam_type != 'B' or frequencies is None or not fc):
        return None
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    if f.size < 2:
        return None
    reach = float(np.max(np.abs(f / fc - 1.0)))
    if reach <= GAUSSIAN_BEAM_BAND_FRACTION:
        return None
    return (
        f"{model_name}: beam_type='B' takes every Gaussian beam's width "
        f"from the one trace at fc = {fc:g} Hz, and this band "
        f"({f.min():g}-{f.max():g} Hz) reaches {100 * reach:.0f} % of fc "
        f"from it, so its bins away from fc are first-order: measured on a "
        f"30 m sand guide against a trace at each frequency, the mean "
        f"energy is off by +1.41 / -0.88 dB at ±50 % of fc. Run sub-bands "
        f"near their own fc, or the default beam_type='G', whose hat-shaped "
        f"beams carry no frequency in their width (0.05 dB at ±95 % on "
        f"the same guide).")


def beam_type_run_mode_notice(run_mode, *, beam_type) -> Optional[str]:
    """The notice for the one supported-but-misleading ``beam_type`` x
    ``run_mode`` pair, ``beam_type='S'`` with EIGENRAYS, or ``None``:
    the rays it writes are real rays, just not screened for passing near
    a receiver."""
    if (_RUN_MODE_TO_INFLUENCE_LETTER.get(run_mode) != 'E'
            or beam_type != 'S'):
        return None
    # InfluenceSGB writes the ray from inside its RcvrDepths loop
    # (influence.f90:696-703) with the proximity test COMMENTED OUT:
    # `! Adeltaz = ABS( deltaz )` and `! IF ( Adeltaz < RadiusMax )`
    # at :698-699, matched by the `! END IF` at :714. So every ray
    # that crosses a receiver-range column is written once for EVERY
    # receiver depth, however far it passes — the count scales with
    # the depth count rather than with the number of rays that
    # actually reach a receiver. The other beam types reach
    # WriteRay2D only through ApplyContribution (:629-652), which
    # the caller enters only inside its own `n < RadiusMax` test
    # (:475 for 'G', :595 for 'B').
    #
    # Measured on a 100 m Pekeris at 200 Hz, 300 beams, receivers at
    # 400/500/600 m: 736 rays against 132 for 'G' with one receiver
    # depth, and 78 % of them pass more than 5 m from any receiver
    # (median 12.7 m, worst 49 m — half the water column).
    #
    # This is a notice, not a rejection: the records are genuine
    # ray trajectories and Rays.filter_by_miss_distance turns them
    # back into eigenrays, so refusing the run would remove a usable
    # capability. 'E' therefore stays in _BEAM_TYPE_RUN_TYPES['S'],
    # which mirrors the Fortran's CASE branches and nothing else.
    return (
        "Bellhop(beam_type='S') EIGENRAYS are not screened for "
        "passing near a receiver: InfluenceSGB writes the ray from "
        "inside its receiver-depth loop with the proximity test "
        "commented out (influence.f90:698-699), so every ray "
        "crossing a receiver-range column is written once per "
        "receiver depth however far away it passes. Expect several "
        "times as many rays as beam_type='G' returns, many of them "
        "missing the receiver by a large fraction of the water "
        "depth. Use beam_type='G' or 'B' for true eigenrays, or "
        "filter the result with Rays.filter_by_miss_distance().")


def line_source_sgb_notice(source, run_mode, *, beam_type, backend,
                           model_name) -> Optional[str]:
    """The notice for ``beam_type='S'`` with a line Source on the
    Fortran backend, or ``None``: the Fortran ``InfluenceSGB`` sets the
    launch weight ``Ratio1 = SQRT(COS(alpha))`` unconditionally
    (``influence.f90:665``), where every other influence routine applies
    it only to a point source (``RunType(4:4) == 'R'``, e.g. :52, :185)
    and bellhopcxx / bellhopcuda set 1 for a line source
    (``bellhopcuda/src/influence.hpp:551-555``). The binary is left as
    built, so the backends differ here and the run says so."""
    if (beam_type != 'S' or source.source_type != 'line'
            or backend != 'fortran'
            or run_mode != RunMode.COHERENT_TL):
        return None
    return (
        f"{model_name}(beam_type='S', backend='fortran') weights a "
        f"line source as a point source: InfluenceSGB applies the "
        f"launch weight sqrt(cos(alpha)) unconditionally "
        f"(influence.f90:665), where every other beam type and the "
        f"bellhopcxx / bellhopcuda ports weight a line source by 1, so "
        f"a path launched at alpha comes back changed by "
        f"10*log10(cos(alpha)) dB (-0.3 dB at 20 deg, -3 dB at 60 deg). "
        f"Use beam_type='G' or "
        f"'B', or backend='cxx', for a line source.")


def geometry_notices(env, source, receiver, mode, *, launch_angles, grid_type,
                     beam_type, n_beams, model_name, ray_step, memory_backed):
    """The stage-3 notices of the geometry, in the order they are
    announced: a receiver the launch fan cannot reach, a first range at
    r = 0, a water column too shallow for ray theory, an eigenray file
    that will be large."""
    notices = [message_notice(fan_miss_notice(source, receiver, launch_angles=launch_angles,
                                           grid_type=grid_type, model_name=model_name),
                           NumericsWarning)]
    # Bellhop evaluates no field at r = 0 unless writes_origin_column says
    # so: the TL grids come back NaN there (_output.mask_unwritten_origin),
    # and so do the broadband routes, whose arrivals only 'G', 'B' and 'g'
    # record and none of them at r = 0. Newcomers using
    # ``np.linspace(0, R, N)`` for ``receiver.ranges`` hit a column of NaN
    # at r=0 and rightly wonder what is wrong. An all-zero grid is refused
    # in stage 2, so it gets the refusal alone.
    if (mode in _FIELD_RUN_MODES
            and not writes_origin_column(source.source_type, beam_type)
            and len(receiver.ranges) > 0
            and float(receiver.ranges[0]) == 0.0
            and receiver.range_max > 0.0):
        notices.append(
            message_notice(f"{model_name}: receiver.ranges starts at r=0 m. "
                        f"Bellhop evaluates no field there: its receiver loops "
                        f"start right of the source range and its point-source "
                        f"factor 1/sqrt(r) is set to 0 at r = 0 "
                        f"(influence.f90:419, :786-787), so that column is "
                        f"NaN. Start ranges at a small positive value (e.g. "
                        f"``np.linspace(eps, R, N)``).", ValidityWarning))
    # Ray-theory validity: the D/lambda floor is a statement about the
    # FIELD the beam sum produces, so RAYS / EIGENRAYS / ARRIVALS —
    # geometry, not a field — stay silent.
    if mode in _FIELD_RUN_MODES:
        notices.append(message_notice(ray_validity_notice(env, source,
                                                       model_name=model_name),
                                   ValidityWarning))
    if mode == RunMode.EIGENRAYS:
        notices.append(message_notice(eigenray_size_notice(
                        env, source, receiver, n_beams=n_beams, grid_type=grid_type,
                        ray_step=ray_step, memory_backed=memory_backed),
                                   NumericsWarning))
    return [n for n in notices if n is not None]
