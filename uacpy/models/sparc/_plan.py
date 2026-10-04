"""What one SPARC deck is resolved to before it is written: the refusals of
knobs and boundaries the binary cannot run, the deck root of each launch, the
spectral ``RMax`` and its margin, the speeds the timing checks read, the
output time grid, the mesh, the memory a snapshot holds, and the notices of
conditions the run warns about."""

from typing import List, Tuple, Union

import numpy as np

from uacpy.core.constants import DEFAULT_SOUND_SPEED
from uacpy.core.environment import Environment
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, NumericsWarning,
    UnsupportedFeatureError,
)
from uacpy.io.oalib_writer import at_env_media
from uacpy.models._budget import memory_budget
from uacpy.core.run_settings import Notice
from uacpy.models._window import check_pinned_window
from uacpy.models.sparc._pulse import _validate_pulse_type


#: SPARC's automatic time window, in direct travel times to the
#: farthest receiver at the slowest SSP speed.
SPARC_WINDOW_TRAVEL_TIMES = 2.5
#: SPARC's default ``rmax_factor``; it must exceed
#: ``1 + SPARC_WINDOW_TRAVEL_TIMES`` (see
#: :func:`resolve_rmax_factor`).
SPARC_RMAX_FACTOR = 4.0

# Factor above the Nyquist minimum used when naming the ``n_time_samples`` that would
# resolve the pulse band in the aliasing warning.
_SPARC_PULSE_OVERSAMPLE = 3.0

# The sampling ReadEnvironmentMod.f90:103 sizes an automatic mesh at:
# ``deltaz = c / freq0 / 20``. Written explicitly for SPARC because sparc.f90
# never rescales the mesh with frequency the way kraken.f90:75 and
# scooter.f90:106 do, so "automatic" means 20 points per wavelength at the
# deck's nominal frequency and only 10 at the top of the pulse band this
# wrapper writes.
_AT_POINTS_PER_WAVELENGTH = 20.0

#: ``Scooter/sparc.f90:31`` — ``MaxN = 17000`` mesh points across all
#: media: medium ``m`` occupies ``Loc(m) + 1 .. Loc(m) + N(m) + 1`` and
#: ``:204-207`` stops with 'Insufficient storage for mesh' past it.
_MAX_MESH_POINTS = 17000

#: The two looped time-series modes, which differ only in the axis
#: ``sparc.f90`` cannot write in one run and therefore the wrapper loops
#: over. Everything else — the deck writer, the ``.rts`` read, the shared
#: time-grid check, the zero-range mask and the Field contract — is one
#: body in :func:`~uacpy.models.sparc._extract.stack_traces`. Per entry:
#:
#: ``loop_attr`` / ``held_attr``
#:     Receiver fields: the looped axis is passed one value at a time, the
#:     held one whole.
#: ``loop_coord`` / ``other_coord``
#:     Which Field coordinate the looped axis is, and which one the
#:     ``.rts`` file's own axis becomes. The ``.rts`` written in vertical
#:     mode stores the DEPTH axis in the slot the horizontal mode uses for
#:     ranges, which is why one reader field feeds both.
#: ``stack_axis``
#:     Where the looped axis lands in the (depth, range, time) contract.
#: ``scale``
#:     The far-field Hankel kernel is
#:     ``H0(kr) ~ sqrt(2/(pi·k·r))·e^{i(kr-pi/4)}``, so inverting it
#:     weights each wavenumber sample by
#:     ``dk·k·sqrt(2/(pi·k·r)) = dk·sqrt(2k/(pi·r))``. ``sparc.f90``'s 'D'
#:     branch carries exactly that (``sqrt(2)·dk·sqrt(k)`` at ``:595``
#:     times the ``1/sqrt(pi·Rr)`` write scale at ``:292``), so it needs no
#:     correction and is the convention the other modes are brought onto.
#:     The 'R' branch at ``:622-623`` applies ``sqrt(2)·dk·sqrt(k/r)`` and
#:     no write scale, i.e. the same weight without the ``1/sqrt(pi)`` — so
#:     it is ``sqrt(pi)`` hot and is divided back here.
#: ``mask_reason``
#:     Why ``r = 0`` columns are no-data. ``sparc.f90:622`` weights the
#:     'R' branch by ``SQRT( rkT / Pos%Rr )``, singular there — measured,
#:     it mixes NaN with exact zeros a caller would read as real, quiet
#:     samples. ``sparc.f90:292`` scales the 'D' branch by
#:     ``1 / SQRT( pi * Pos%Rr( 1 ) )``, also singular — measured, the
#:     whole trace comes back +Inf. The snapshot mode's in-tree Hankel
#:     transform masks the same cells, so all three output modes give one
#:     answer.
#:
#: Every launch shares one ``RMax`` (the farthest receiver range times
#: the margin, ``SPARC._resolve_engine_settings``) and therefore one output
#: time grid.
_LOOPED_TIME_SERIES_MODES = {
    'R': dict(
        loop_attr='depths', held_attr='ranges',
        loop_coord='depth', other_coord='range',
        # Spelled out, not built from loop_coord: a metadata key composed
        # by an f-string is invisible to grep and to the registry gate
        # that checks every documented key is actually written.
        runs_key='n_depth_runs',
        run_tag='d', stack_axis=0,
        scale=1.0 / np.sqrt(np.pi),
        loop_axis_label='depths',
        mode_label="horizontal-array mode (output_mode='R')",
        opening='Computing {n} depth(s) (SPARC horizontal array mode)...',
        step='  depth {i}/{n}: {value:.1f} m',
        mask_reason=("output_mode='R''s 1/sqrt(r) cylindrical-spreading "
                     "factor"),
    ),
    'D': dict(
        loop_attr='ranges', held_attr='depths',
        loop_coord='range', other_coord='depth',
        runs_key='n_range_runs',
        run_tag='r', stack_axis=1,
        scale=1.0,
        loop_axis_label='ranges',
        mode_label="vertical-array mode (output_mode='D')",
        opening='Computing vertical array at {n} range(s)...',
        step='  range {i}/{n}: {value:.1f} m',
        mask_reason=("output_mode='D''s 1/sqrt(r) cylindrical-spreading "
                     "factor"),
    ),
}


def check_knobs(*, c_low, c_high, output_mode, pulse_type, march_start,
                freq_min, freq_max) -> None:
    """Refuse a constructor knob no run could use: two pinned phase-speed
    bounds with ``c_low >= c_high``, an unknown ``output_mode`` or
    ``pulse_type``, a positive ``march_start``, a negative ``freq_min``, a
    non-positive ``freq_max`` or a pinned band with ``freq_min >= freq_max``.

    Run at construction and again by every run (the attributes can be
    reassigned in between).
    """
    check_pinned_window('SPARC', c_low=c_low, c_high=c_high)
    if output_mode not in ('R', 'D', 'S'):
        raise ConfigurationError(
            f"Invalid output_mode {output_mode!r}. Valid modes "
            f"(sparc.f90 TopOpt(5:5)): 'R' RTS horizontal array, 'D' RTS "
            f"vertical array, 'S' snapshot."
        )
    if pulse_type is not None:
        _validate_pulse_type(pulse_type)
    # The march runs time = march_start + (i-1)·Δt from a field at rest
    # (sparc.f90:409) while the deck always asks for the output window
    # [0, time_max]. A positive march_start therefore starts the solution after
    # cans.f90's pulse has already turned on (it is zero only for T <= 0),
    # and EXTRACT's DO WHILE (sparc.f90:572) empties every requested time below
    # march_start out of the first step at a negative interpolation weight.
    if float(march_start) > 0.0:
        raise ConfigurationError(
            f"SPARC march_start={march_start} s starts the time march "
            f"after the source pulse has turned on (t = 0) and after the "
            f"start of the [0, time_max] output window the deck writes, so "
            f"the field is marched from rest mid-pulse and the output "
            f"times below march_start are extrapolated backwards out of "
            f"the first step. Use march_start <= 0 (the default -0.1 s "
            f"pre-rolls the march before the pulse)."
        )
    # An inverted or negative band feeds sparc.f90:116 a negative
    # kMax-kMin, i.e. Nk < 0 — the same silent all-zero field the
    # run() Nk guard refuses — so fail here with the cause named. freq_min =
    # 0 is legal and stays: sparc.f90:114 clamps kMin to 1e-20 for exactly
    # that case, and doc/sparc.htm's own example deck reads "0.0 15.0".
    if freq_min is not None and freq_min < 0.0:
        raise ConfigurationError(
            f"SPARC pulse band requires freq_min >= 0 Hz; got {freq_min}."
        )
    if freq_max is not None and freq_max <= 0.0:
        raise ConfigurationError(
            f"SPARC pulse band requires freq_max > 0 Hz; got {freq_max}."
        )
    if (freq_min is not None and freq_max is not None
            and freq_min >= freq_max):
        raise ConfigurationError(
            f"SPARC pulse band requires freq_min < freq_max; got "
            f"freq_min={freq_min} Hz, freq_max={freq_max} Hz."
        )


def reject_halfspace_bottom(env: Environment) -> None:
    """Raise :class:`ConfigurationError` when any seabed column ends in
    a half-space.

    ``sparc.f90:101-104`` accepts only vacuum or rigid boundaries, and
    a rigid floor put in the half-space's place is a different
    waveguide: Scooter on 100 m of water at 100 and 500 Hz puts the
    rigid bottom 7.2-10.5 dB above the default half-space over 1-9 km,
    6.5-10.4 dB above 'sand' and 12.6-33.9 dB above 'clay'. The caller
    chooses the boundary; the model does not substitute one.

    For a ``Bottom`` the ``acoustic_type`` lives on each column's
    ``.halfspace`` (per range when range-dependent), so every column is
    checked.
    """
    if not any(col.halfspace.acoustic_type == 'half-space'
               for col in env.bottom.columns):
        return
    raise ConfigurationError(
        "SPARC's deck carries only 'vacuum' and 'rigid' bottom "
        "boundaries (sparc.f90:101-104), and this environment's bottom "
        "is a half-space. A rigid floor in its place is a different "
        "waveguide — 7-10 dB louder over 1-9 km on the default seabed, "
        "up to 34 dB on clay — so SPARC does not substitute one.",
        remediation=(
            "Pass acoustic_type='rigid' (or 'vacuum') explicitly on the "
            "bottom's half-space if that is the waveguide you mean, or "
            "use Scooter for a half-space bottom (TIME_SERIES)."),
    )


def reject_reflection_table_bottom(env) -> None:
    """Refuse a reflection-table seabed (``'file'`` / ``'precalc'``):
    SPARC's deck carries only the vacuum and rigid boundaries."""
    hs = env.bottom.halfspace_at(range=0.0)
    if hs.acoustic_type.lower() not in ('vacuum', 'rigid'):
        # ``_validate_acoustic_type`` has already rejected unrecognised
        # names and ``reject_halfspace_bottom`` has already refused a
        # half-space, so what reaches here is a reflection-table bottom
        # ('file' / 'precalc') — valid everywhere else, unrepresentable in
        # SPARC's Vacuum/Rigid-only deck. Kraken and Scooter stage the
        # same table.
        raise UnsupportedFeatureError(
            'SPARC',
            f"a {hs.acoustic_type!r} bottom — its deck carries only the "
            f"'vacuum' and 'rigid' boundary conditions",
            alternatives=['Kraken', 'Scooter'],
        )


def resolve_run_bases(receiver, *, output_mode,
                      max_launches) -> Tuple[str, ...]:
    """The deck root of each launch, in launch order — one per value of
    the looped axis (:data:`_LOOPED_TIME_SERIES_MODES`), ``'model'``
    alone for a one-value axis or a snapshot — with the looped axis
    refused over ``max_launches``."""
    if output_mode == 'S':
        return ('model',)
    spec = _LOOPED_TIME_SERIES_MODES[output_mode]
    n = int(np.atleast_1d(np.asarray(
        getattr(receiver, spec['loop_attr']))).size)
    reject_oversized_loop_axis(
        n, spec['loop_axis_label'], spec['mode_label'],
        max_launches=max_launches)
    if n == 1:
        return ('model',)
    return tuple(f"model_{spec['run_tag']}{idx}" for idx in range(n))


def reject_oversized_loop_axis(n_runs: int, axis: str, mode_label: str, *,
                               max_launches) -> None:
    """Cap the axis the wrapper loops the binary over.

    ``output_mode='R'`` runs one SPARC subprocess per receiver depth and
    ``'D'`` one per receiver range; ``'S'`` runs once and never calls this.
    """
    if n_runs <= max_launches:
        return
    raise UnsupportedFeatureError(
        model_name='SPARC',
        feature=(
            f"{n_runs} receiver {axis} (SPARC {mode_label} runs one "
            f"simulation per {axis[:-1]}; current limit is "
            f"max_launches={max_launches})"
        ),
        alternatives=[
            f"Reduce receiver.{axis} to at most {max_launches} entries",
            f"Raise the limit explicitly: SPARC(max_launches={n_runs})",
            "SPARC(output_mode='S') computes the whole grid in one run",
            "Bellhop, RAM, Kraken, Scooter, or OASN for dense 2D fields",
        ],
    )


def reject_oversized_snapshot(receiver, nk: int, n_time_samples: int, *,
                              output_mode):
    """The :func:`~uacpy.models._budget.memory_budget` notice of the
    wavenumber-domain table ``output_mode='S'`` materialises, or ``None``;
    raises when the table is more than the host reports free.

    ``'R'`` / ``'D'`` are bounded by ``max_launches`` because the wrapper
    loops the binary over them; ``'S'`` runs once and is bounded by the
    ``Green(Itout, irz, ik)`` cube instead (``sparc.f90:580-591``), which
    the binary holds whole and ``read_grn_file`` reads back as a
    ``complex64`` array of the same shape. Its size is set by the
    wavenumber count — which grows with ``rmax_factor`` and the
    pulse band — so it can reach tens of GB without any single knob
    looking unreasonable.
    """
    if output_mode != 'S':
        return None
    n_depth = int(np.atleast_1d(np.asarray(receiver.depths)).size)
    n_bytes = 8 * int(n_time_samples) * n_depth * int(nk)
    # What is weighed is the snapshot table alone --
    # ``GreensFunction.snapshot_to_time_field`` transforms it one output time
    # at a time, and that scratch is not counted here, so this estimate is
    # the more optimistic of the two Green's-function estimates.
    return memory_budget(
        n_bytes, model_name='SPARC', what='snapshot table',
        detail=(
            f"a snapshot (output_mode='S') holds a Green's-function table "
            f"of {n_bytes / 1024 ** 3:.1f} GiB (n_time_samples={int(n_time_samples)} x "
            f"{n_depth} receiver depth(s) x Nk={int(nk)} x 8 B)."),
        remediation=(
            "Reduce n_time_samples (the table scales with it one-for-one) or "
            "receiver.depths; narrow the pulse band (freq_min/freq_max) or lower "
            "rmax_factor, which both set Nk; or use "
            "SPARC(output_mode='R') or 'D', which stream one trace per run "
            "instead of holding the whole cube."))


def resolve_rmax_factor(*, rmax_factor) -> float:
    """Pick the effective ``rmax_factor`` for this run.

    ``RMax`` fixes SPARC's wavenumber step (``sparc.f90:116``:
    ``Nk = INT(1000·RMax_km·(kMax−kMin)/2π)`` ⇒ ``Δk ≈ 2π/RMax_m``).
    The ``'R'`` / ``'D'`` modes synthesise range inline in ``EXTRACT``
    as a direct ``Δk`` sum over that grid (``sparc.f90:595,622``), and
    the ``'S'`` snapshot is transformed in-tree by the same kind of
    direct DFT (``core.acoustics.hankel_transform``) — no FFT anywhere. A
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
    (``SPARC._resolve_engine_settings``), so the margin has to exceed 3.5.
    User-pinned values win; :func:`range_alias_notice` checks whichever
    margin is in force against the window actually written.

    Scooter needs no such margin: ``scooter.f90:69`` samples four times
    as finely (``Nk = 2000*RMax_km*(kMax-kMin)/pi``, i.e.
    ``Δk ≈ π/(2·RMax)``), putting its period at ``4·RMax``.
    """
    if rmax_factor is not None:
        return float(rmax_factor)
    return SPARC_RMAX_FACTOR


def profile_speed_bounds(env: Environment, *, window_sound_speed) -> tuple:
    """``(slowest, fastest)`` sound speed (m/s) for the timing checks.

    The output window is anchored on the slowest speed (the latest the
    direct arrival can land) and the range-alias check on the fastest
    (the earliest the folded replica can). ``window_sound_speed`` pins the slow
    end; ``DEFAULT_SOUND_SPEED`` covers a profile that carries no usable
    speed at all.

    The two ends read different parts of the environment, on purpose.
    ``c_slow`` stays water-only: it anchors ``time_max``, which is a
    heuristic rather than a bound on the last arrival (see
    ``SPARC._resolve_engine_settings``) and is backstopped after the run
    by
    :func:`~uacpy.models.sparc._extract.warn_on_truncated_window`
    measuring the trace that came back.
    ``c_fast`` spans every medium the deck carries, water and seabed
    alike, because the replica :func:`range_alias_notice` looks for
    travels the fold distance through the whole waveguide: a fast
    sediment layer over a 1500 m/s column carries it in ahead of any
    water path. That is also what the binary itself does —
    ``Scooter/sparc.f90:202-216`` runs ``MediumLoop: DO medium = 1,
    SSP%NMedia`` taking ``cMax = MAX(cpR, cMax)`` over EVERY medium, and
    ``write_layer_sections`` emits each sediment layer as one more
    medium with ``NMEDIA`` incremented. ``all_sound_speeds`` is the
    matching accessor: compressional speeds only, skipping the
    vacuum / rigid / file / precalc half-spaces that carry no seabed
    speed (a half-space bottom never reaches here:
    ``reject_halfspace_bottom`` refuses it first).
    """
    speeds = np.asarray(env.ssp.sound_speed, dtype=float).ravel()
    speeds = speeds[np.isfinite(speeds) & (speeds > 0.0)]
    c_slow = window_sound_speed
    if c_slow is None:
        c_slow = (float(speeds.min()) if speeds.size
                  else DEFAULT_SOUND_SPEED)
    c_fast = float(speeds.max()) if speeds.size else float(c_slow)
    seabed = [c for c in env.bottom.all_sound_speeds()
              if np.isfinite(c) and c > 0.0]
    if seabed:
        c_fast = max(c_fast, max(seabed))
    return float(c_slow), max(float(c_fast), float(c_slow))


def lossless_march_notice(env):
    """``(note, warning)`` when ``env`` asks for any attenuation — a
    water ``absorption`` law or a non-zero compressional attenuation in
    a seabed layer — which SPARC's march ignores (``sparc.f90:221`` keeps
    the real part of the complex sound speed only), else ``None``."""
    asked = []
    if env.absorption is not None:
        asked.append(f"env.absorption ({type(env.absorption).__name__})")
    bottom = env.bottom
    for column in (getattr(bottom, 'columns', None) or []):
        for layer in (getattr(column, 'layers', None) or []):
            if float(layer.attenuation) != 0.0:
                asked.append(f"a seabed layer at "
                             f"{float(layer.attenuation):g} dB/wavelength")
                break
        else:
            continue
        break
    if not asked:
        return None
    return Notice(
        "attenuation ignored: the march is lossless (sparc.f90:221)",
        f"SPARC: {' and '.join(asked)} ignored — the time march is "
        f"lossless. sparc.f90:221 converts each complex sound speed to "
        f"single precision with REAL(cp, 4), keeping the real part only, "
        f"so no attenuation reaches the march (c2I = 0 at :223); that "
        f"keeps the explicit scheme consistent, whose step divides by "
        f"the diagonal of A2 alone. Use Scooter for the same waveguide "
        f"with its attenuation (a TIME_SERIES from the spectral "
        f"solution).", FallbackWarning)


def range_alias_notice(rmax_m: float, r_ref: float, c_fast: float,
                       time_max: float):
    """``(note, warning)`` when the ``Δk`` sum's range replica falls
    inside ``[0, time_max]``, else ``None``.

    The farthest receiver (``r_ref``) sees the fold from ``RMax - r_ref``
    (see :func:`resolve_rmax_factor`); the earliest it can arrive
    is that distance over the fastest speed in the profile. Inside the
    output window it is indistinguishable from a real late arrival, so
    name the ``rmax_factor`` that pushes it clear.
    """
    if not (r_ref > 0.0 and c_fast > 0.0 and time_max > 0.0):
        return None
    t_alias = (rmax_m - r_ref) / c_fast
    if t_alias > time_max:
        return None
    margin_needed = 1.0 + time_max * c_fast / r_ref
    return Notice(
        f"range alias at t = {t_alias:.4g} s inside the {time_max:.4g} s "
        f"window",
        f"SPARC range alias: RMax = {rmax_m:.6g} m puts the "
        f"wavenumber sum's range replica at t = {t_alias:.4g} s at the "
        f"farthest receiver ({r_ref:.6g} m), inside the "
        f"{time_max:.4g} s output window — it will read as a real late "
        f"arrival. Use SPARC(rmax_factor>"
        f"{margin_needed:.3g}) (runtime scales with it) or shorten the "
        f"window with time_max.", NumericsWarning,
    )


def resolve_n_time_samples(freq_max, time_max, *, n_time_samples):
    """``(n_time_samples, notice)``: the caller's ``n_time_samples``, with a
    ``(note, warning)`` when the resulting grid cannot resolve the pulse
    band (``None`` otherwise).

    ``n_time_samples < 2`` is refused: the deck writes the output times as the
    ``0.0 time_max /`` pair that ``misc/subtabulate.f90:24`` expands only
    for ``Nx >= 3``, and ``Nx = 1`` consumes the leading ``0.0`` alone —
    the whole window is discarded and the binary returns a single-sample
    p(t) at exit 0.

    The native ``p(t)`` sampling is the caller's to choose, so the value is
    kept verbatim. But a grid whose Nyquist sits below ``freq_max`` aliases
    silently: the returned p(t) looks perfectly plausible at the wrong
    frequency. Say so, and name the ``n_time_samples`` that fixes it. ``SubTab``
    expands ``0.0 time_max /`` inclusive of both endpoints, so the step is
    ``time_max/(n_time_samples − 1)`` and the sample rate ``(n_time_samples − 1)/time_max``
    — not ``n_time_samples/time_max``, which overstates it by ``n/(n−1)``.

    The window is ``[0, time_max]`` — the *output* window the writer emits
    (``oalib_writer``), not ``march_start``, which only sets where the
    integration begins.
    """
    if int(n_time_samples) < 2:
        raise ConfigurationError(
            f"SPARC: n_time_samples={n_time_samples} cannot express the "
            f"[0, time_max] output window: the deck writes the times as a "
            f"'0.0 time_max /' pair that misc/subtabulate.f90:24 expands "
            f"only for Nx >= 3, and Nx = 1 reads the leading 0.0 alone — "
            f"the run returns a single-sample p(t) at exit 0. Use "
            f"n_time_samples >= 2."
        )
    window = float(time_max)
    notice = None
    if window > 0 and freq_max > 0:
        fs = (int(n_time_samples) - 1) / window
        if freq_max > 0.5 * fs:
            n_needed = 1 + int(np.ceil(
                window * _SPARC_PULSE_OVERSAMPLE * 2.0 * freq_max))
            notice = Notice(
                f"output grid Nyquist {fs / 2:.1f} Hz below freq_max "
                f"{freq_max:.0f} Hz",
                f"SPARC TIME_SERIES: the output grid samples at "
                f"{fs:.1f} Hz (Nyquist {fs / 2:.1f} Hz) over a "
                f"{window:.2f} s window, below the {freq_max:.0f} Hz "
                f"source band — p(t) will alias. Set n_time_samples>="
                f"{n_needed}, lower freq_max, or shorten the window via "
                f"time_max / receiver.ranges.max().", NumericsWarning,
            )
    return n_time_samples, notice


def refuse_too_few_wavenumbers(nk: int, rmax_m: float, freq_min: float,
                               freq_max: float, c_low_res: float,
                               c_high_res: float) -> None:
    """Refuse a wavenumber count the binary cannot march: ``Nk <= 0``
    runs the march loop zero times and ``Nk = 1`` is a single-sample
    spectrum."""
    if nk >= 2:
        return
    raise ConfigurationError(
        f"SPARC would march Nk = {nk} wavenumber(s) (sparc.f90:116 "
        f"with rmax = {rmax_m:.6g} m, band {freq_min:.6g}–{freq_max:.6g} Hz, "
        f"c_low = {c_low_res:.6g} m/s, c_high = {c_high_res:.6g} m/s): "
        f"the wavenumber loop would run empty and the field would come "
        f"back all-zero at exit 0.",
        remediation="Increase the receiver range or widen the pulse "
                    "band (freq_min/freq_max) so that "
                    "rmax·(freq_max/c_low − freq_min/c_high) ≥ 2.",
    )


def checked_n_mesh(env, freq_max: float, *,
                   pinned_n_mesh) -> Tuple[int, ...]:
    """:func:`resolve_n_mesh` as one count per medium (a pinned count on
    every medium), refused when the deck's mesh would overrun SPARC's
    static storage (:data:`_MAX_MESH_POINTS`)."""
    n_mesh = resolve_n_mesh(env, freq_max, n_mesh=pinned_n_mesh)
    counts = (list(n_mesh) if isinstance(n_mesh, list)
              else [int(n_mesh)] * len(at_env_media(env)))
    total = sum(int(n) + 1 for n in counts)
    if total <= _MAX_MESH_POINTS:
        return tuple(counts)
    origin = ("the pinned n_mesh" if pinned_n_mesh else
              f"the automatic mesh ({_AT_POINTS_PER_WAVELENGTH:g} points "
              f"per wavelength at the band top freq_max = {float(freq_max):g} "
              f"Hz)")
    raise ConfigurationError(
        f"SPARC's mesh would need {total} points over {len(counts)} "
        f"medium/media from {origin}, but the binary stores at most "
        f"{_MAX_MESH_POINTS} (Scooter/sparc.f90:31 MaxN).",
        remediation=("Lower freq_max (default: the source waveform's band, "
                     "or 2x the source frequency for a canned pulse), "
                     "reduce the modelled depth, or pin a smaller "
                     "n_mesh."),
    )


def resolve_n_mesh(env, freq_max: float, *, n_mesh) -> Union[int, List[int]]:
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
    (``:54``, ``:85``). But the pulse band this wrapper writes reaches
    above ``freq0`` (to ``2*freq`` for a canned pulse), so an automatic
    mesh ran the top of that band at 10 points per wavelength.

    Measured on a 100 m isovelocity guide, vacuum surface / rigid bottom,
    source and receiver at 50 m, TIME_SERIES, against a converged mesh:
    the automatic count is 0.239 relative rms at 20 Hz / 500 m and 0.375 at
    60 Hz / 300 m (peak -0.86 dB), with the coarse mesh also putting the
    arrival about 1.7 ms early over 300 m. Convergence is monotone in the
    count, so sizing at ``freq_max`` rather than ``freq0`` is the fix rather
    than a tuning choice.
    """
    if n_mesh:
        return int(n_mesh)
    f = float(freq_max)
    if not np.isfinite(f) or f <= 0.0:
        return int(n_mesh)
    # One count per MEDIUM, not one scalar: AT sizes each medium from its
    # OWN thickness and speed (ReadEnvironmentMod.f90:101-106), so the
    # water column's count broadcast to a 10 m sediment layer over-resolves
    # that layer by the thickness ratio and the march fails outright.
    return [max(int(_AT_POINTS_PER_WAVELENGTH * thickness * f / c), 10)
            for thickness, c in at_env_media(env)]
