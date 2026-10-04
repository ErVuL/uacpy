"""The Collins backends (ramgeo, rams, ramsurf): their deck and range
segments, domain and grid, the array limits they are compiled with, the
stop conditions they report, and the field read from their output."""

import numpy as np
import warnings
from pathlib import Path
from typing import Optional
from uacpy.models.base import StageInputs
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.models._notices import give_notice
from uacpy.models.ram._seabed import piecewise_breakpoints
from uacpy.models.ram._pe_phase import psi_to_travelling_wave
from uacpy.models.pe_grid import rams_dz_shear_cap
from uacpy.core.environment import (
    Environment,
)
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.results import Field, PhaseReference
from uacpy.core.constants import NO_ENERGY_DB
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, FileFormatError, ModelExecutionError,
    NumericsWarning, ValidityWarning,
)
from uacpy.io.ramsurf_writer import write_ramin
from uacpy.io._parsers import parse_pcomplex_grid, parse_tl_grid
from uacpy.models.ram._settings import RamGrid
from uacpy.core.bathymetry import mask_below_seafloor
from uacpy.models.ram._interp import interp_envelope_to_receiver_grid
from uacpy.models.ram._domain import (
    COLLINS_SAMPLES_PER_BEAT,
    absorbing_layer_thickness,
    check_source_row_is_solved,
    compute_zmax,
    cut_water_ssp_at_zmax,
    deck_block,
    deck_convention,
    deck_depth,
    deck_depth_inverse,
    deck_water_column,
    depth_index_base,
    is_seafloor_relative,
    launch_grid,
    max_shear_speed,
    min_shear_speed,
    modal_beat_wavenumber,
    resolve_c0,
    ssp_range_axis,
    warn_if_seafloor_outside_grid,
    water_attenuation_active,
    water_attenuation_block,
)
from uacpy.models.ram._band import (
    broadband_frequencies,
    report_band_grid_cost,
    requested_broadband_bins,
    resolve_broadband_grid,
)
from uacpy.models.ram._stability import (
    collins_carrier_rate,
    collins_stability_range,
    rams_rotation_remedy,
    rams_stability,
    resolve_collins_thetas,
    resolve_rams_rotation_angle,
    warn_if_rams_step_unstable,
)
from uacpy.models.ram.grid import (
    align_dz_with_seafloor,
    anchored_at_the_origin,
    collins_output_spacing,
    compute_grid_lytaev,
    ramsurf_surface_nodes,
    warn_if_trapped_modes_unresolved,
)


# Collins-family PE numerics constants.
#
# LAMBDA_PER_DZ_FLOOR — depth samples per acoustic wavelength at which the
#   automatic depth search STARTS (``dz = c_min/(16·f)`` where the optimiser
#   asked for finer); it bounds the cost of the depth grid on a seabed whose
#   trapped modes it resolves, which Lytaev's error model on its own does
#   not. It is not where the search stops: when the steepest trapped mode
#   scores at or above ``TRAPPED_MODE_SCORE_LIMIT`` on the floored grid, dz
#   is refined until it passes, within ``MAX_DEPTH_POINTS``. The value is
#   uacpy's — no samples-per-wavelength floor is prescribed by the RAM
#   sources, their readme, Collins 1993 or Lytaev 2023 §4.
# RAMS_DR_LAMBDA_CAP — empirical upper stability bound on dr for rams0.5's
#   rotated Padé elastic march, expressed as a divisor of c_min/freq.
#   ``dr ≤ c_min / (RAMS_DR_LAMBDA_CAP·f)`` ≈ 0.2 λ per step.
# Fortran array dimensions of the Collins binaries (``parameter (mr=…,mz=…)``
# at the top of each source): rams0.5 and ramgeo carry uacpy's enlargement,
# ramsurf1.5 keeps upstream's stock mz=20002, the size ramgeo is matched to — see
# third_party/MODIFICATIONS.md. ``mr`` bounds the bathymetry arrays. ``mz``
# bounds the depth arrays, but the codes do not consume it at the same rate:
# rams0.5 interleaves the elastic field vector and indexes 2*nz+4
# (rams0.5.f:154 ``do 3 i=1,2*nz+4``, and the 2*nz solve loops at :795, :811,
# :813, :842), while the fluid codes index nz+2 (ramgeo1.5.f:138). The matching
# bounds check at rams0.5.f:141 is a uacpy addition — see
# third_party/MODIFICATIONS.md. ``mz`` is set so all three reach nz=20000.
# Each binary stops with a diagnostic on an overrun but writes no output, so it
# would otherwise surface as a truncated-file read — guard before launching.
_COLLINS_ARRAY_LIMITS = {
    'rams':    {'mr': 505, 'mz': 40004, 'nz_factor': 2, 'nz_pad': 4},
    'ramsurf': {'mr': 505, 'mz': 20002, 'nz_factor': 1, 'nz_pad': 2},
    'ramgeo':  {'mr': 505, 'mz': 20002, 'nz_factor': 1, 'nz_pad': 2},
}


# ``zread`` pins each sediment-block point to the node ``i = 1.5 + z/dz`` and
# remembers only the *immediately* preceding index, so its collision push-down
# (``ramsurf1.5.f:208``, identical in ramgeo1.5.f:223 and rams0.5.f:226) protects
# one duplicate depth and no more. Two consecutive distinct depths landing in the
# same cell make the later point overwrite the earlier, and the fill loop at
# ramsurf1.5.f:218-219 then ramps linearly across the whole gap it left. A gap of at least
# ``BLOCK_GAP_PER_DZ * dz`` keeps the nodes distinct, since
# ``floor(x + g/dz) - floor(x) >= floor(g/dz)``.
BLOCK_GAP_PER_DZ = 2.0


# Ceiling on the number of range records a Collins binary writes. The stride
# ``ndr`` is otherwise lowered until the first written range reaches the
# nearest receiver, which for a near-field receiver on a long run would write
# one record per ``dr``.
_COLLINS_MAX_OUTPUT_RANGES = 20000


#: The Collins codes' marker table: the stdout lines of the two stop
#: conditions they diagnose themselves and end on a bare Fortran ``stop``
#: (exit 0) — the array-size checks (``ramgeo1.5.f:138-149``, the same block
#: in ``rams0.5.f`` and ``ramsurf1.5.f``) and the Padé root finder
#: (``ramgeo1.5.f:767-771``). See :func:`raise_on_collins_stop`.
_COLLINS_STOP_MARKERS = ('Laguerre method not converging',
                         'Need to increase parameter',
                         'Try a different combination')


# Cap on profile sections added so a layered elastic seabed follows sloping
# bathymetry under rams0.5. Each section costs six blocks in the deck.
MAX_BATHY_SECTIONS = 64


def resolve_collins_tl(env, source, receiver, kind: str, *,
                        notices=None, knobs, log, speed_bounds) -> dict:
    """Stage 3 of a Collins COHERENT_TL run: one launch at the source
    frequency."""
    fc = float(np.atleast_1d(source.frequencies)[0])
    theta = resolve_collins_thetas(env, kind, [fc],
                                   notices=notices, knobs=knobs,
                                   speed_bounds=speed_bounds)[0]
    grid = resolve_collins_launch(
        env, source, receiver, kind=kind, freq=fc, theta=theta,
        notices=notices, knobs=knobs, log=log, speed_bounds=speed_bounds)
    return dict(fc=None, q_factor=None, record_duration=None, bandwidth_hz=None, df_hz=None,
                marched_frequencies=[fc], requested_frequencies=None,
                zmax_frequency=fc,
                stability_range_m=collins_stability_range(kind, knobs=knobs),
                grids=(grid,))


def resolve_collins_band(env, source, receiver, kind: str, *,
                          require_uniform: bool = True,
                          notices=None, knobs, log, speed_bounds) -> dict:
    """Stage 3 of a Collins BROADBAND / TIME_SERIES run: one launch per
    frequency, every one on the band's grid.

    A caller-supplied frequency array is marched bin for bin; otherwise
    the vector matches mpiramS's convention: ``fc`` from
    :func:`_band.resolve_broadband_grid`, half-bandwidth ``fc/Q`` (the band
    spans ``2·fc/Q``) and frequency resolution ``df = 1/T``.
    ``require_uniform=False`` (BROADBAND) accepts a non-uniform array,
    which only the TIME_SERIES synthesis cannot use.

    ``rams_rotation_angle`` may be a callable; when it is, ``theta`` is
    resolved per frequency by ``_stability.theta_for_freq`` — useful when the
    elastic stability angle has to vary across the band.
    """
    fc, Q_used, T_used = resolve_broadband_grid(
        source, require_uniform=require_uniform, notices=notices, knobs=knobs,
        log=log)
    # The Python-side loop has no uniform-count constraint (unlike the
    # Fortran sweep), so when the caller supplied an explicit frequency
    # array march exactly the requested bins and no others. A
    # non-uniform array has no (fc, Q, T) description (Q_used is None).
    target = requested_broadband_bins(source, knobs=knobs)
    if target is not None:
        frequencies = np.asarray(target, dtype=float)
    else:
        frequencies = broadband_frequencies(fc, Q_used, T_used)
    if Q_used is not None:
        bw = fc / Q_used
        df = 1.0 / T_used
    else:
        bw = 0.5 * float(frequencies[-1] - frequencies[0])
        df = None

    # Pick numerics ONCE for the whole broadband loop, both at freq_max —
    # the smallest wavelength in the band, which is the end that binds. A
    # step sized lower in the band is under-resolved for every bin above
    # it, which `_resolve_mpirams_grid` states and the mpiramS path already
    # honours for both steps.
    #
    # EXCEPT on rams0.5, where dr stays at freq_min. That is a STABILITY
    # measure, not an accuracy one: a finer dr means more range steps, and
    # rams0.5's rotated-Pade elastic march is only marginally stable
    # (|G| ~ 1 near the evanescent boundary), so per-step floating-point
    # noise compounds — sizing dr at freq_max (~3x more steps) injects a
    # spurious acausal precursor into its broadband synthesis, verified
    # against the fluid baseline and confirmed by a rotation-angle sweep
    # (Milinazzo 1997).
    #
    # That argument is specific to the elastic march. Applying it to the
    # FLUID Collins backends only under-resolved them: measured on a 100 m
    # Pekeris guide (1700/1.7/0.5) against Scooter at 500 Hz, ramgeo with
    # dr sized at freq_min = 100 Hz (54.05 m) is 8.85 dB rms / 16.86 dB max
    # out, against 1.64 / 5.24 with dr sized at freq_max (12.81 m).
    freq_min = float(frequencies[0])
    freq_max = float(frequencies[-1])
    dr_sizing_freq = freq_min if kind == 'rams' else freq_max
    rmax_band = float(np.max(np.atleast_1d(receiver.ranges)))

    dr_band = float(knobs.dr) if knobs.dr is not None else None
    dz_band = float(knobs.dz) if knobs.dz is not None else None
    zs_band = float(np.min(source.depths))
    if dr_band is None:
        dr_band, _ = compute_grid_lytaev(
            env, dr_sizing_freq, max_range=rmax_band, kind=kind,
            warn_dz=False, zs=zs_band, notices=notices, knobs=knobs, log=log,
            speed_bounds=speed_bounds
        )
    if dz_band is None:
        _, dz_band = compute_grid_lytaev(
            env, freq_max, max_range=rmax_band, kind=kind, zs=zs_band,
            notices=notices, knobs=knobs, log=log, speed_bounds=speed_bounds
        )
    zmax_band = (float(knobs.zmax) if knobs.zmax is not None
                 else compute_zmax(env, freq_min, max_range=rmax_band,
                                   notices=notices, knobs=knobs,
                                   speed_bounds=speed_bounds))
    if knobs.dz is None:
        # Coarsen once for the whole band so every frequency marches the
        # same grid and the warning is emitted once, not per frequency.
        dz_band = fit_dz_to_mz(env, kind, dz_band, zmax_band,
                               freq=freq_max, notices=notices, knobs=knobs)
    warn_if_trapped_modes_unresolved(env, freq_max, kind, dr_band,
                                     dz_band, rmax_band,
                                     notices=notices, knobs=knobs, log=log,
                                     speed_bounds=speed_bounds)

    log(
        f"{kind} broadband: {len(frequencies)} frequencies, "
        f"{frequencies[0]:.2f}-{frequencies[-1]:.2f} Hz, "
        f"df={'non-uniform' if df is None else f'{df:.2f} Hz'}, "
        f"bw={bw:.2f} Hz, "
        f"dr={dr_band:.2f}m, dz={dz_band:.3f}m, zmax={zmax_band:.0f}m"
    )
    thetas = resolve_collins_thetas(env, kind, frequencies,
                                    notices=notices, knobs=knobs,
                                    speed_bounds=speed_bounds)
    # One profile deck for the whole sweep: only the absorbing ramp
    # inside it moves with frequency (``collins_deck_base``), and
    # ``zmax_band`` is fixed for the band, so the sections are cut once
    # instead of once per bin — on a 400-column bottom that is 45 ms a
    # bin.
    deck_base = collins_deck_base(env, kind, zmax_band, knobs=knobs)
    grids = tuple(
        resolve_collins_launch(
            env, source, receiver,
            kind=kind, freq=float(freq), theta=theta,
            dr_override=dr_band, dz_override=dz_band,
            zmax_override=zmax_band,
            # dr_band is the band's one-grid-for-all override; it is
            # caller-set only when self.dr is.
            dr_override_pinned=(knobs.dr is not None),
            deck_base=deck_base, notices=notices, knobs=knobs, log=log,
            speed_bounds=speed_bounds
        )
        for freq, theta in zip(frequencies, thetas))
    report_band_grid_cost(env, kind, grids[0], freq_min, freq_max,
                          notices=notices, knobs=knobs, log=log,
                          speed_bounds=speed_bounds)
    return dict(fc=fc, q_factor=Q_used, record_duration=T_used, bandwidth_hz=2.0 * bw,
                df_hz=df, marched_frequencies=frequencies,
                requested_frequencies=None, zmax_frequency=freq_min,
                stability_range_m=collins_stability_range(kind, knobs=knobs),
                grids=grids)


def resolve_collins_launch(
    
    env: Environment,
    source: Source,
    receiver: Receiver,
    *,
    kind: str,
    freq: float,
    theta: float,
    dr_override: Optional[float] = None,
    dz_override: Optional[float] = None,
    zmax_override: Optional[float] = None,
    dr_override_pinned: Optional[bool] = None,
    deck_base: Optional[dict] = None, notices=None, knobs, log, speed_bounds
) -> RamGrid:
    """Stage 3 of one Collins launch at ``freq``: its grid, stride and
    output depths, with the notices and the refusals of the deck they
    make — resolved on the deck geometry the launch writes
    (:func:`collins_deck_geometry`).

    ``dr_override_pinned`` says whether ``dr_override`` traces back to a
    caller-set value: the broadband loop always passes an override (one
    grid for the whole band), so the override's mere presence cannot
    distinguish a user's dr from an auto-derived one. ``None`` — the
    narrowband path — falls back to "any override is caller-set".

    ``deck_base`` lets the band share one :func:`collins_deck_base`
    cut for every bin; ``None`` cuts it for this launch alone.
    """
    # The Collins binaries handle one (zs, fc) per call; mpiramS does
    # the same. Range-dependent SSP and (layered) bottom ARE threaded
    # through — one ``ram.in`` profile section per range break, built in
    # ``collins_range_segments``.
    fc = float(freq)
    zs_all = [deck_depth(float(z), knobs=knobs)
              for z in np.atleast_1d(source.depths)]
    zs = min(zs_all)

    max_range = float(np.max(np.atleast_1d(receiver.ranges)))
    dr, dz, zmax = resolve_collins_grid(
        env, fc, kind, max_range,
        dr_override, dz_override, zmax_override, zs=zs, notices=notices,
        knobs=knobs, log=log, speed_bounds=speed_bounds
    )
    if kind == 'ramsurf':
        warn_if_ramsurf_crests(env, notices=notices)
    # Built before the stride because the section spacing bounds ``dr``:
    # the binary consumes at most one profile section per range step.
    # Only the section markers are read here, so the water block (which
    # the deck writer adds) is left out.
    bathymetry, surface, range_segments = collins_deck_geometry(
        env, kind, zmax, fc, dz, max_range, deck_base=deck_base,
        water=False, knobs=knobs, speed_bounds=speed_bounds)
    # The WRITTEN node ranges bound dr, the origin node included: the
    # binary consumes at most one bathymetry node per range step.
    bathy_r = [r for r, _ in bathymetry]
    alti_r = [r for r, _ in surface] if surface is not None else None
    # A dr pinned on the CONSTRUCTOR is just as user-set as one passed
    # to run(); testing only the override rewrites it in silence.
    dr_pinned = (dr_override_pinned if dr_override_pinned is not None
                 else dr_override is not None)
    dr = constrain_dr_to_sections(
        dr, range_segments,
        pinned=(dr_pinned or knobs.dr is not None),
        bathymetry_ranges=bathy_r, altimetry_ranges=alti_r,
        notices=notices, log=log)
    beat_k = modal_beat_wavenumber(env, fc, speed_bounds=speed_bounds)
    ndr, rmax_march = collins_output_stride(
        dr, max_range, receiver.ranges, beat_wavenumber=beat_k)
    log(
        f"{kind}: output stride ndr={ndr} (dr·ndr={dr * ndr:.4g} m; "
        f"cap {collins_output_spacing(beat_k):.4g} m = modal beat / "
        f"{COLLINS_SAMPLES_PER_BEAT:.0f})."
    )
    ndz = max(1, int(knobs.depth_decimation))

    # ``zmax`` stays geometric for the grid arithmetic above; the deck
    # is written in the mapped frame, ``zmax_deck`` being its floor
    # (mpiramS: ``zmax = maxval(zw)`` after its own transform).
    zmax_deck = deck_depth(zmax, knobs=knobs)
    rcv_d = np.atleast_1d(receiver.depths).astype(float)
    target_depth = deck_depth(float(np.max(rcv_d)), knobs=knobs)
    zmplt = collins_zmplt(
        max(target_depth, deck_depth(float(env.depth), knobs=knobs)),
        dz, zmax_deck, ndz, kind)
    z_deepest = collins_deepest_output(zmplt, dz, ndz, kind)
    # Receivers below the deepest stored output sample come back NaN from
    # ``interp_to_receiver_grid`` (fill_value=nan); warn so the empty
    # rows are attributable. Reachable only when ``zmax`` clamps ``zmplt``.
    if target_depth > z_deepest:
        # expected; not in filterwarnings — emerges to user
        give_notice(notices,
            f"RAM:{kind}: receiver depths up to {target_depth:.1f} m "
            f"exceed the PE domain (zmax={zmax:.1f} m, deepest stored "
            f"output sample {z_deepest:.1f} m); samples below it "
            f"are returned as NaN. Increase zmax to cover all receiver "
            f"depths.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP
        )

    r_first = dr * ndr
    near = np.atleast_1d(np.asarray(receiver.ranges, dtype=float))
    near = near[near > 0.0]
    if near.size and float(near.min()) < r_first:
        # expected; not in filterwarnings — emerges to user
        give_notice(notices,
            f"RAM:{kind}: the binary writes its first output range at "
            f"{r_first:.3f} m (dr={dr:.3f} m × ndr={ndr}); receiver ranges "
            f"below that are returned as NaN. Pin a smaller dr to move the "
            f"first output range in.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP
        )
    if ndz > 1:
        # rams0.5 writes from grid index 1+ndz, the fluid codes from ndz.
        z_first = (ndz if kind == 'rams' else ndz - 1) * dz
        in_gap = rcv_d[(rcv_d > 0.0) & (rcv_d < z_first)]
        if in_gap.size:
            give_notice(notices,
                f"RAM:{kind}: depth_decimation={ndz} makes the shallowest "
                f"computed output depth {z_first:.3f} m; {in_gap.size} "
                f"receiver depth(s) below it are interpolated between the "
                f"pressure-release surface and that sample. Set "
                f"depth_decimation=1 to resolve them.",
                NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP
            )

    if surface is not None:
        for z in zs_all:
            check_source_below_depressed_surface(surface, z, dz,
                                                 notices=notices)

    # Checked for every launch, the broadband sweep's included, and on
    # the profiles as written so the bound sees exactly what the binary
    # consumes.
    check_collins_array_limits(kind, dz, zmax_deck, bathymetry,
                               surface)
    check_rams_seafloor_index_floor(kind, dz, bathymetry)
    for z in zs_all:
        check_source_row_is_solved(z, dz)

    # ``tl.line``'s receiver depth indexes u(ir)/f3(ir+1) with no bounds
    # check (ramsurf1.5.f:427, rams0.5.f:251), so keep it inside the
    # binary's own nz = zmax/dz - 0.5 depth arrays.
    nz_march = int(zmax_deck / dz - 0.5)
    zr_line = min(target_depth, max(0.0, (nz_march - 1) * dz))
    return RamGrid(frequency=fc, dr=float(dr), dz=float(dz),
                   zmax=float(zmax), n_depth_points=nz_march,
                   theta=float(theta), ndr=int(ndr),
                   rmax_march=float(rmax_march), zmplt=float(zmplt),
                   zr_line=float(zr_line))


def resolve_collins_grid(env, fc, kind, max_range,
                          dr_override, dz_override, zmax_override,
                          zs=None, *, notices=None, knobs, log, speed_bounds):
    """Resolve the PE numerics grid ``(dr, dz, zmax)`` for one Collins run.

    Explicit overrides (from the broadband loop, which picks one set for the
    whole band — matching mpiramS) take priority, then user-set ``self.*``,
    then the Lytaev Padé-error optimizer for whatever is still ``None``. The
    rams shear-stability dz floor and the 5× rams ``dr`` safety factor are
    applied to the optimizer's output (see ``grid.compute_grid_lytaev``).

    ``self.dz`` alone decides whether ``dz`` counts as caller-pinned: the
    broadband override is the band's Lytaev grid whenever the caller left
    ``dz`` unset, and equals ``self.dz`` when they did set it.
    """
    dr = float(dr_override) if dr_override is not None else (
        float(knobs.dr) if knobs.dr is not None else None
    )
    dz = float(dz_override) if dz_override is not None else (
        float(knobs.dz) if knobs.dz is not None else None
    )
    dz_pinned = knobs.dz is not None

    dr_pinned = dr is not None
    if dr is None or dz is None:
        dr_auto, dz_auto = compute_grid_lytaev(
            env, fc, max_range=max_range, kind=kind, zs=zs, notices=notices,
            knobs=knobs, log=log, speed_bounds=speed_bounds
        )
        if dr is None:
            dr = dr_auto
        if dz is None:
            dz = dz_auto
    if kind == 'rams' and dr_pinned:
        warn_if_rams_step_unstable(env, fc, dr, notices=notices, knobs=knobs,
                                   speed_bounds=speed_bounds)
    if zmax_override is not None:
        zmax = float(zmax_override)
    elif knobs.zmax is not None:
        zmax = float(knobs.zmax)
    else:
        zmax = compute_zmax(env, fc, max_range=max_range, notices=notices,
                            knobs=knobs, speed_bounds=speed_bounds)
    if (zmax_override is not None or knobs.zmax is not None):
        warn_if_seafloor_outside_grid(zmax, env, dz=dz,
                                      kind=kind, freq=fc,
                                      notices=notices, knobs=knobs,
                                      speed_bounds=speed_bounds)

    # ``mz`` bounds the DECK's depth grid, which is written in the mapped
    # frame (``_domain.deck_depth``, up to ~1e-4 relative deeper under
    # ``earth_curvature``), so the array budget is measured there; the physics
    # above keeps the geometric ``zmax``.
    zmax_deck = deck_depth(zmax, knobs=knobs)
    if not dz_pinned:
        dz = fit_dz_to_mz(env, kind, dz, zmax_deck, freq=fc,
                          notices=notices, knobs=knobs)

    # Resolving the sediment block outranks every coarsening above,
    # including the mz budget: a block zread cannot represent is not a
    # coarser answer, it is a different environment. Only the three Collins
    # backends pin block points to grid nodes — mpiramS carries no such
    # arithmetic in any of its sources and interpolates the profile onto the
    # grid with ``interpolators.f90``'s ``interp1``, so it is exempt.
    if (kind in _COLLINS_ARRAY_LIMITS
            and block_loses_a_point(env, dz, zmax, kind, fc, knobs=knobs,
                                    speed_bounds=speed_bounds)):
        block_cap = block_dz_cap(env, zmax, kind, fc, knobs=knobs,
                                 speed_bounds=speed_bounds)
        if dz_pinned:
            raise ConfigurationError(
                f"RAM(dz={dz:.4f}) cannot represent this sediment block: "
                f"zread ({kind}) pins block points to nodes 1.5 + z/dz and "
                f"two of them collide at this dz, so the deeper value "
                f"overwrites the shallower one and the fill loop replaces "
                f"the layer with a linear ramp across the whole sub-bottom. "
                f"The thinnest step is "
                f"{block_cap * BLOCK_GAP_PER_DZ:.4f} m.",
                remediation=f"Use dz <= {block_cap:.4f} m, leave dz=None to "
                            f"have it derived, or merge the step into its "
                            f"neighbour if it is not physically meant to be "
                            f"resolved.",
            )
        # Keep the seafloor placed across the tightening, but only
        # where the aligned value still resolves the block and still fits
        # the depth arrays: the cap comes from an exact predicate rather
        # than a monotone bound, so a finer dz is not automatically clean.
        budget = collins_mz_budget(kind, zmax_deck)
        block_aligned = align_dz_with_seafloor(env, block_cap,
                                               kind=kind, knobs=knobs)
        if (block_aligned > 0
                and not block_loses_a_point(
                    env, block_aligned, zmax, kind, fc, knobs=knobs,
                    speed_bounds=speed_bounds)
                and (budget is None
                     or budget[0](block_aligned) <= budget[1])):
            block_cap = block_aligned
        if budget is not None and budget[0](block_cap) > budget[1]:
            raise ConfigurationError(
                f"RAM:{kind}: resolving this sediment block needs dz <= "
                f"{block_cap:.4f} m, i.e. {budget[0](block_cap)} depth "
                f"slots against the binary's mz={budget[1]} over a "
                f"zmax={zmax:.1f} m domain. A coarser grid would silently "
                f"replace the block with a linear ramp, so this cannot be "
                f"met by coarsening.",
                remediation="Lower zmax, use backend='mpirams' (no fixed "
                            "depth-array bound), or thicken/merge the "
                            "thinnest sediment step.",
            )
        log(
            f"RAM:{kind}: tightened dz from {dz:.4f} m to {block_cap:.4f} m "
            f"so zread resolves the sediment block (thinnest step "
            f"{block_cap * BLOCK_GAP_PER_DZ:.4f} m)."
        )
        dz = block_cap
    # The broadband sweep resolves its band grid once and hands it in as
    # the overrides; it scores that grid itself, once, not per bin.
    if dr_override is None and dz_override is None:
        warn_if_trapped_modes_unresolved(env, fc, kind, dr, dz,
                                         max_range, notices=notices,
                                         knobs=knobs, log=log,
                                         speed_bounds=speed_bounds)
    return dr, dz, zmax


def write_collins_deck(inputs: StageInputs, *, knobs, speed_bounds) -> Path:
    """Stage 4 of one Collins launch: ``ram.in`` / ``rams.in`` /
    ``ramgeo.in`` written by :func:`~uacpy.io.ramsurf_writer.write_ramin`
    from the launch's :class:`RamGrid` and the deck geometry
    (:func:`collins_deck_geometry`) at its frequency."""
    engine = inputs.settings.engine
    grid = launch_grid(inputs)
    kind = engine.backend
    env = inputs.env
    fc = grid.frequency
    zs = deck_depth(float(np.atleast_1d(inputs.source.depths)[0]), knobs=knobs)
    max_range = float(np.max(np.atleast_1d(inputs.receiver.ranges)))
    bathymetry, surface, range_segments = collins_deck_geometry(
        env, kind, grid.zmax, fc, grid.dz, max_range,
        deck_base=inputs.prepared, knobs=knobs, speed_bounds=speed_bounds)
    ram_in = inputs.work_dir / collins_in_name(kind)
    write_ramin(
        str(ram_in),
        kind=kind,
        frequency=fc, zs=zs, zr_line=grid.zr_line,
        rmax_march=grid.rmax_march, dr=grid.dr, ndr=grid.ndr,
        zmax=deck_depth(grid.zmax, knobs=knobs), dz=grid.dz,
        depth_decimation=max(1, int(knobs.depth_decimation)), zmplt=grid.zmplt,
        c0=engine.c0, n_pade=int(knobs.n_pade),
        n_stability=int(knobs.n_stability),
        stability_range_m=float(engine.stability_range_m or 0.0),
        rams_rotation=int(knobs.rams_rotation),
        rams_rotation_angle=float(grid.theta),
        bathymetry=bathymetry,
        surface=surface,
        range_segments=range_segments,
        title=f"uacpy {kind} run @ {fc:.1f} Hz"
    )
    return ram_in


def read_collins_output(inputs: StageInputs, *, knobs, speed_bounds) -> dict:
    """Stage 4 of one Collins launch: its output on the binary's grid
    (:func:`read_collins_grid`), the diverged samples marked no-data
    (:func:`mark_diverged_collins_samples`, reported rather than warned
    so a band warns once), and the envelope resampled onto the receiver
    grid (:func:`interp_envelope_to_receiver_grid`) — so a band holds
    one receiver-grid slice per bin, not the binary's whole grid.

    Keys: ``psi`` (receiver depth × range, complex envelope), ``report``
    (the diverged-sample report, or ``None``), and the launch's
    ``frequency`` / ``dr`` / ``dz`` / ``zmax``.
    """
    grid = launch_grid(inputs)
    env = inputs.env
    kind = inputs.settings.engine.backend
    receiver = inputs.receiver
    raw = read_collins_grid(inputs, knobs=knobs, speed_bounds=speed_bounds)
    reports = []
    psi_marked = mark_diverged_collins_samples(
        raw, env, kind, reports=reports, knobs=knobs,
        speed_bounds=speed_bounds)
    # Receivers outside the PE output grid get NaN so pcolormesh and
    # downstream consumers render them transparent rather than as a
    # saturated edge band.
    psi = interp_envelope_to_receiver_grid(
        raw['depths'], raw['ranges'], psi_marked,
        np.atleast_1d(receiver.depths).astype(float),
        np.atleast_1d(receiver.ranges).astype(float),
        carrier_rate=collins_carrier_rate(
            env, kind, grid.frequency, grid.theta, knobs=knobs,
            speed_bounds=speed_bounds))
    return {'psi': psi, 'report': reports[0] if reports else None,
            'frequency': grid.frequency, 'dr': grid.dr, 'dz': grid.dz,
            'zmax': grid.zmax}


def read_collins_grid(inputs: StageInputs, *, knobs, speed_bounds) -> dict:
    """One Collins launch's ``tl.grid`` and ``pcomplex.bin`` on the
    binary's own output grid, as pressure (rams0.5's dilatation
    converted, :func:`rams_pressure_from_dilatation`) with the surface
    node prepended and the depth axis geometric.

    Keys: ``tl`` / ``pcomplex`` (depth × range), ``depths`` /
    ``ranges`` (the binary's grid), and the launch's ``frequency`` /
    ``theta`` / ``dr`` / ``dz`` / ``zmax``.
    """
    engine = inputs.settings.engine
    kind = engine.backend
    grid = launch_grid(inputs)
    env = inputs.env
    receiver = inputs.receiver
    work_dir = inputs.work_dir
    dr, dz, ndr = grid.dr, grid.dz, grid.ndr
    ndz = max(1, int(knobs.depth_decimation))
    max_range = float(np.max(np.atleast_1d(receiver.ranges)))
    tlgrid = work_dir / 'tl.grid'
    pcgrid = work_dir / 'pcomplex.bin'
    # rams0.5 writes its output grid from index 1+ndz, ramsurf1.5
    # from ndz (third_party/ramsurf/{rams0.5,ramsurf1.5}.f outpt).
    depth_index_offset = depth_index_base(kind)
    ranges, depths, tl = parse_tl_grid(
        tlgrid, dr=dr, ndr=ndr, dz=dz, ndz=ndz,
        depth_index_offset=depth_index_offset
    )
    _, _, pcomplex = parse_pcomplex_grid(
        pcgrid, dr=dr, ndr=ndr, dz=dz, ndz=ndz,
        depth_index_offset=depth_index_offset
    )
    # The same ``outpt`` call writes both files on one (z, r) grid, so
    # unequal shapes mean one is truncated. Left to itself the
    # mismatch first bites as a numpy broadcast error inside
    # ``mark_diverged_collins_samples``, naming neither file.
    if tl.shape != pcomplex.shape:
        raise FileFormatError(
            f"RAM:{kind}: {tlgrid} decodes to a {tl.shape} grid but "
            f"{pcgrid} to {pcomplex.shape}; the binary writes both on "
            f"the same grid, so one of them is truncated.",
            remediation=(
                "Delete the work directory and re-run. If it repeats, "
                "the binary is being stopped mid-write (disk full, "
                "timeout, or an external kill)."
            ),
        )
    # ``interp_to_receiver_grid`` NaN-fills out-of-grid ranges, so a
    # march that ended short of the farthest receiver — a truncated
    # tl.grid, or a binary that exited 0 without finishing — would
    # otherwise return a partly-NaN Field that ``np.nanmean`` reduces
    # to a plausible wrong number. Third sibling of the depth and
    # near-range warnings of stage 3. Relative epsilon: the axis is
    # rebuilt as ``k·dr·ndr``, which can miss an equal receiver range by
    # ulps.
    r_last = float(ranges[-1]) if ranges.size else 0.0
    if max_range - r_last > 1e-9 * max_range:
        # expected; not in filterwarnings — emerges to user
        warnings.warn(
            f"RAM:{kind}: the binary's output grid stops at "
            f"{r_last:.1f} m but receiver ranges extend to "
            f"{max_range:.1f} m; samples beyond it are returned as "
            f"NaN. The march ended early — check for a truncated "
            f"{tlgrid.name} or a binary stopped before rmax.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP
        )
    if kind == 'rams':
        # The sections the deck carried; the sound-speed, shear and
        # density blocks the conversion reads are frequency-invariant.
        bathymetry, _, range_segments = collins_deck_geometry(
            env, kind, grid.zmax, grid.frequency, dz, max_range,
            deck_base=inputs.prepared, water=False, knobs=knobs,
            speed_bounds=speed_bounds)
        zs = deck_depth(
            float(np.atleast_1d(inputs.source.depths)[0]), knobs=knobs)
        tl, pcomplex = rams_pressure_from_dilatation(
            depths, ranges, tl, pcomplex, range_segments,
            bathymetry, zs)
    depths, tl, pcomplex = prepend_surface_node(
        depths, tl, pcomplex)
    # The binaries' depth axis is the deck's frame; receivers are
    # interpolated at geometric depths, as mpiramS's output is.
    depths = np.asarray(deck_depth_inverse(depths, knobs=knobs), dtype=float)

    return {'tl': tl, 'pcomplex': pcomplex, 'depths': depths,
            'ranges': ranges, 'dr': dr, 'dz': dz, 'zmax': grid.zmax,
            'frequency': grid.frequency, 'theta': grid.theta}


def assemble_collins_tl_field(inputs: StageInputs,
                               raw: dict, *, attach_output_paths,
                               mask_source_axis, result_kwargs) -> Field:
    """Stage 5 of a Collins COHERENT_TL run: the receiver-grid envelope
    of :func:`read_collins_output` as a :class:`Field` of
    travelling-wave pressure."""
    engine = inputs.settings.engine
    kind = engine.backend
    env = inputs.env
    source = inputs.source
    fc = raw['frequency']
    warn_on_diverged_collins_samples(
        kind, [raw['report']] if raw['report'] else [], 1)
    rcv_d = np.atleast_1d(inputs.receiver.depths).astype(float)
    rcv_r = np.atleast_1d(inputs.receiver.ranges).astype(float)

    # Same per-backend phase bookkeeping as the broadband loop; the
    # Collins files already carry the 1/√r scaling, so apply_radial=False.
    pressure = psi_to_travelling_wave(
        raw['psi'],
        convention=deck_convention(kind),
        ranges_m=rcv_r,
        range_axis=1,
        k0=2.0 * np.pi * fc / engine.c0,
        apply_radial=False,
    ).astype(np.complex128)

    # Mask sub-seafloor samples with NaN (same semantics as every backend).
    pressure = mask_below_seafloor(pressure, rcv_d, rcv_r, env.bathymetry)

    field = Field(
        data=pressure,
        coords={'depth': rcv_d, 'range': rcv_r},
        **result_kwargs(
            source,
            phase_reference=PhaseReference.TRAVELLING_WAVE.value,
            backend=kind,
            frequencies=fc,
        )
    )
    field = mask_source_axis(field, source)
    attach_output_paths(
        field, inputs.work_dir, '',
        primary_files=(
            ('tl_grid_file', 'tl.grid'),
            ('tl_line_file', 'tl.line'),
            ('pcomplex_file', 'pcomplex.bin'),
            ('in_file', collins_in_name(kind))
        )
    )
    return field


def assemble_collins_band_field(inputs: StageInputs,
                                 raws, *, attach_output_paths,
                                 mask_source_axis, result_kwargs) -> Field:
    """Stage 5 of a Collins band: the per-bin receiver-grid envelopes
    stacked into ``(n_d, n_r, n_f)`` and converted to the engineering
    travelling-wave H(f) every other broadband-capable model returns
    (the carrier ``exp(-i k0 r)`` baked in), with one warning for the
    diverged samples of the whole band."""
    settings = inputs.settings
    engine = settings.engine
    kind = engine.backend
    env = inputs.env
    source = inputs.source
    frequencies = np.array(engine.marched_frequencies, dtype=float)
    warn_on_diverged_collins_samples(
        kind, [r['report'] for r in raws if r['report']], len(raws))
    rcv_d = np.atleast_1d(inputs.receiver.depths).astype(float)
    rcv_r = np.atleast_1d(inputs.receiver.ranges).astype(float)

    # Convention: trailing axis is the variable dim (frequency).
    H = np.zeros((rcv_d.size, rcv_r.size, frequencies.size), dtype=complex)
    for k, raw in enumerate(raws):
        H[:, :, k] = raw['psi']

    # Convert each backend's raw output to the engineering travelling-
    # wave form. See ``models/ram/_pe_phase.py`` for the per-convention
    # math. H is shaped (n_d, n_r, n_f) here; the Collins binaries
    # already include the 1/√r radial scaling in the file they write,
    # so ``apply_radial=False``.
    c0 = engine.c0
    omega = 2.0 * np.pi * np.asarray(frequencies, dtype=np.float64)
    # ramgeo's UACPY envelope dump (u·f3/√r, carrier factored out) is
    # identical to ramsurf1.5's, so it uses the same phase convention.
    H = psi_to_travelling_wave(
        H,
        convention=deck_convention(kind),
        ranges_m=rcv_r,
        range_axis=1,
        k0=omega / c0,
        freq_axis=2,
        apply_radial=False,
    )

    # Mask sub-seafloor samples with NaN (same semantics as every backend).
    H = mask_below_seafloor(H, rcv_d, rcv_r, env.bathymetry)

    field = Field(
        data=H,
        coords={'depth': rcv_d, 'range': rcv_r, 'frequency': frequencies},
        **result_kwargs(
            source,
            phase_reference=PhaseReference.TRAVELLING_WAVE.value,
            backend=kind,
            frequencies=frequencies,
        )
    )
    field = mask_source_axis(field, source)
    # Every frequency ran in the one work directory, so the paths
    # describe the sweep as a whole rather than any one launch.
    attach_output_paths(
        field, inputs.work_dir, '',
        primary_files=(
            ('tl_grid_file', 'tl.grid'),
            ('tl_line_file', 'tl.line'),
            ('pcomplex_file', 'pcomplex.bin'),
            ('in_file', collins_in_name(kind))
        )
    )
    return field


def mark_diverged_collins_samples(raw: dict, env: Environment,
                                   kind: str, *,
                                   reports: Optional[list] = None, knobs,
                                   speed_bounds
                                   ) -> np.ndarray:
    """Return one Collins run's envelope with diverged samples marked
    no-data, warning on what was marked — or, given a ``reports`` list,
    appending what was marked to it for one warning over a whole band
    (:func:`warn_on_diverged_collins_samples`).

    A sample is no-data when it is NaN/inf, or when its TL is below what
    the range allows — the rotated-Padé march on a fast-shear seabed
    diverges into finite but hugely negative values. Those samples come
    back as NaN, the marker the wrapper already uses for receivers
    outside the output grid and below the seafloor, so they read as
    absent rather than as a deep shadow zone. Every other sample is the
    engine's own number: uacpy does not substitute a level for one the
    model produced.
    """
    # How negative a TL may legitimately be is set by the RANGE, not by a
    # constant: TL is referenced to 1 m, so beyond that radius a passive
    # medium cannot return more pressure than the source put out and TL
    # cannot go below 0, while inside it the sample is closer than the
    # reference and free-field spreading gives 20*log10(r) — -2.5 dB at
    # dr = 0.75 m, but -44.7 dB where the lambda cap drives dr to 5.8 mm
    # at 50 kHz. Measured on a 1 kHz ramgeo march, the minimum TL at each
    # range tracks that bound to 0.06 dB. The allowance below it is one
    # coherent boundary image, which doubles the pressure.
    #
    # Only the inside-1 m branch is a spreading law: past 1 m the bound
    # stays at 0 dB rather than following 20*log10(r), because a
    # waveguide spreads cylindrically and a real field at 10 km sits
    # ~40 dB below the spherical value — bounding on spreading there
    # would reject the whole far field.
    IMAGE_GAIN_DB = 6.02
    tl_raw = np.asarray(raw['tl'], dtype=float)
    ranges = np.asarray(raw['ranges'], dtype=float)
    with np.errstate(divide='ignore'):
        floor_dB = 20.0 * np.log10(
            np.minimum(np.maximum(ranges, np.finfo(float).tiny), 1.0)
        ) - IMAGE_GAIN_DB
    # A blow-up cannot be found by a NaN test: rams0.5.f:265 takes TL from
    # ``alog10(cabs(ur))``, so a march that overflows the field but stays
    # under the REAL ceiling (8-byte in the shipped build, 4-byte in a
    # stock one) writes finite, hugely negative samples.
    invalid = ~np.isfinite(tl_raw) | (tl_raw < floor_dB)
    n_invalid = int(np.count_nonzero(invalid))
    if n_invalid:
        note = ""
        c0_pe = resolve_c0(env, knobs=knobs, speed_bounds=speed_bounds)
        advice = "Try a larger n_pade or a finer dz."
        if kind == 'rams' and raw.get('dr') is not None:
            theta = raw.get('theta')
            if theta is None:
                theta = resolve_rams_rotation_angle(env,
                                           float(raw['frequency']),
                                           knobs=knobs,
                                           speed_bounds=speed_bounds)
            stab = rams_stability(
                env, float(raw['frequency']), dr=float(raw['dr']),
                theta=theta, knobs=knobs, speed_bounds=speed_bounds)
            if stab is not None and stab['margin']['excess'] > 0.0:
                m = stab['margin']
                advice = (
                    f"The rotated Crank-Nicolson step at dr="
                    f"{float(raw['dr']):.4g} m amplifies the steepest "
                    f"propagating components at {m['growth']:.2e} Np/m "
                    f"against the {m['leak']:.2e} Np/m this seabed leaks "
                    f"them at. "
                    + (f"Use dr <= {stab['dr']:.4g} m (dr=None picks it)."
                       if stab['dr'] is not None
                       else rams_rotation_remedy(stab, knobs=knobs)))
        if kind == 'rams' and max_shear_speed(env) > c0_pe:
            # A shear speed above the reference speed puts the shear band next
            # to the branch point of the rotated square root (ξ = (c0/c_s)² − 1
            # → −1), where the Crank-Nicolson step rams0.5 marches (rpade,
            # ``rams0.5.f:859-892``) has the least accuracy margin; a
            # zero-shear top layer is refused at dispatch
            # (:func:`_dispatch.check_rams_top_layer_carries_shear`).
            note = (f" A shear speed of {max_shear_speed(env):.0f} "
                    f"m/s exceeds the PE reference speed c0="
                    f"{c0_pe:.0f} m/s, where the Collins rams0.5 rotated "
                    f"march loses accuracy fastest; converge dz and dr "
                    f"by halving, or use OAST / Scooter for a fast "
                    f"elastic seabed.")
        report = {'n_invalid': n_invalid, 'size': int(tl_raw.size),
                  'frequency': float(raw['frequency']),
                  'detail': f"{advice}{note}"}
        if reports is None:
            warn_on_diverged_collins_samples(kind, [report], 1)
        else:
            reports.append(report)
    psi_raw = np.asarray(raw['pcomplex'], dtype=np.complex128)
    # Every surviving sample is the engine's own value, bit for bit:
    # the exact zero at the z = 0 pressure-release node the fluid codes
    # emit when ndz = 1 is a valid boundary value, not divergence (the
    # shared ``transmission_loss_dB`` floor reports it as the one no-energy
    # level, and it is not counted in the warning above), and inside the
    # 1 m reference radius |p/p0| > 1 is what the field is.
    return np.where(invalid, complex(np.nan, np.nan), psi_raw)


def warn_on_diverged_collins_samples(kind: str, reports,
                                      n_launches: int) -> None:
    """One warning for the diverged samples
    :func:`mark_diverged_collins_samples` reported over a run: the
    frequency's own when one launch diverged, else the count and span of
    the affected bins with the first one's diagnosis."""
    if not reports:
        return
    head = ("are NaN/inf or below the level their range allows (Padé "
            "instability or PE divergence) and are returned as NaN — no "
            "data there, not a shadow zone. Every other sample is the "
            "march's own value.")
    first = reports[0]
    if len(reports) == 1:
        text = (f"RAM:{kind}: {first['n_invalid']}/{first['size']} TL "
                f"samples at f={first['frequency']:.2f} Hz {head} "
                f"{first['detail']}")
    else:
        freqs = [r['frequency'] for r in reports]
        text = (f"RAM:{kind}: at {len(reports)} of {n_launches} "
                f"frequencies ({min(freqs):.2f}-{max(freqs):.2f} Hz), "
                f"{sum(r['n_invalid'] for r in reports)}/"
                f"{sum(r['size'] for r in reports)} TL samples {head} "
                f"At {first['frequency']:.2f} Hz: {first['detail']}")
    warnings.warn(text, NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)


def raise_on_collins_stop(kind: str, binary, proc_result, *,
                           dr: float, knobs) -> None:
    """Raise :class:`ModelExecutionError` when a Collins binary reported
    a stop condition on stdout and exited 0.

    The three Collins codes end their two self-diagnosed failures on a
    bare Fortran ``stop``, which exits 0: the array-size checks
    (``ramgeo1.5.f:138-149`` "Need to increase parameter mz/mp/mr to N",
    the same block in ``rams0.5.f`` and ``ramsurf1.5.f``) fire before
    the march and leave an empty ``tl.grid``; the Padé root finder
    (``ramgeo1.5.f:767-771`` "Laguerre method not converging. Try a
    different combination of DR and NP.") fires per range step and
    leaves a partial one. Measured: an over-limit Padé order gave
    rc 0, stdout ``Need to increase parameter mp to 11`` and an empty
    ``tl.grid``. Both streams are scanned so a build that routes the
    message to stderr is caught too. ``dr`` is the range step the run
    marched with (auto-chosen or pinned), so the remedy names the pair
    the binary saw.
    """
    text = "\n".join(t for t in (proc_result.stdout, proc_result.stderr)
                     if t)
    quoted = [line.strip() for line in text.splitlines()
              if any(m in line for m in _COLLINS_STOP_MARKERS)]
    if not quoted:
        return
    if any('Laguerre' in line for line in quoted):
        remediation = (
            f"The Padé root finder diverged at this (dr, n_pade) pair; "
            f"the run marched with dr={float(dr):.6g} m and "
            f"n_pade={int(knobs.n_pade)}. Change one of them, e.g. "
            f"RAM(dr={float(dr):.6g}, n_pade="
            f"{max(2, int(knobs.n_pade) - 2)}) or "
            f"RAM(dr={0.5 * float(dr):.6g}, n_pade={int(knobs.n_pade)})."
        )
    else:
        remediation = (
            "The deck exceeds a compiled array bound that "
            "uacpy.models.ram.collins.check_collins_array_limits "
            "sizes from the grid; coarsen "
            "it with RAM(dz=..., zmax=...) or lower RAM(n_pade=...) "
            "until the named parameter fits."
        )
    exc = ModelExecutionError(
        f"RAM:{kind}", proc_result.returncode,
        stdout=proc_result.stdout, stderr=proc_result.stderr)
    exc.message = (
        f"RAM:{kind}: {Path(binary).name} stopped on its own diagnosis "
        f"(a bare Fortran STOP, exit code {proc_result.returncode}):\n  "
        + "\n  ".join(quoted)
    )
    exc.remediation = remediation
    raise exc


def collins_deck_geometry(env: Environment, kind: str,
                           zmax: float, freq: float, dz: float,
                           max_range: float, *,
                           deck_base: Optional[dict] = None,
                           water: bool = True, knobs, speed_bounds):
    """``(bathymetry, surface, range_segments)`` one Collins deck
    carries at ``freq``: the profile sections
    (:func:`collins_range_segments`, or a band's ``deck_base`` ramped
    at ``freq``), the bathymetry anchored at the origin and reaching
    ``max_range``, and ramsurf's surface — each cut to what a march to
    ``max_range`` consumes (:func:`within_march`). ``water=False``
    leaves the water-attenuation block out, for callers that read only
    the section markers or the sediment blocks."""
    range_segments = (
        collins_range_segments(env, kind, zmax, freq, dz=dz,
                               water=water, knobs=knobs,
                               speed_bounds=speed_bounds)
        if deck_base is None
        else ramp_range_segments(env, deck_base, freq, kind=kind,
                                 zmax=zmax, dz=dz, water=water, knobs=knobs,
                                 speed_bounds=speed_bounds)
    )
    bathymetry = anchored_at_the_origin(
        [(float(r), deck_depth(float(d), knobs=knobs))
         for r, d in env.bathymetry.to_pairs().tolist()])
    if bathymetry[-1][0] < max_range:
        bathymetry.append((float(max_range), bathymetry[-1][1]))
    surface = (ramsurf_surface_nodes(env, max_range, knobs=knobs)
               if kind == 'ramsurf' else None)
    # Only what the march reaches is written, so the array-size guards,
    # the dr bound and the seafloor checks see what the binary consumes
    # rather than the whole track.
    bathymetry = within_march(bathymetry, max_range)
    if surface is not None:
        surface = within_march(surface, max_range)
    range_segments = [seg for i, seg in enumerate(range_segments)
                      if i == 0 or float(seg['range']) <= max_range]
    return bathymetry, surface, range_segments


def collins_in_name(kind: str) -> str:
    """Input-deck filename the Collins binary of ``kind`` hardcodes:
    rams0.5 reads 'rams.in', ramgeo 'ramgeo.in', ramsurf1.5 'ram.in'."""
    return {'rams': 'rams.in', 'ramgeo': 'ramgeo.in'}.get(kind, 'ram.in')


def collins_zmplt(target_depth: float, dz: float, zmax: float,
                   ndz: int = 1, kind: str = 'ramgeo') -> float:
    """Row-4 ``zmplt`` that makes the binary store an output sample at or
    below ``target_depth``.

    The Collins codes assign the real expression ``zmplt/dz - 0.5`` to the
    integer ``nzplt`` — a truncation (``ramgeo1.5.f:131``,
    ``ramsurf1.5.f:112``, ``rams0.5.f:133``) — and run their output loop up
    to grid index ``nzplt``, stepping by ``ndz`` from ``ndz`` on the fluid
    codes (``ramgeo1.5.f:429``, ``ramsurf1.5.f:437``) and from ``1+ndz`` on
    rams0.5 (``:262``). Grid index ``i`` sits at depth ``(i-1)·dz``
    (``ri = 1 + zr/dz``, ``ramgeo1.5.f:126-127``), so reaching
    ``target_depth`` needs the loop to visit an index of at least
    ``ceil(target_depth/dz) + 1``. The extra ``0.75·dz`` above the exact
    ``(nzplt + 0.5)·dz`` threshold keeps the binaries' REAL evaluation of
    ``zmplt/dz - 0.5`` (8-byte in the shipped build, 4-byte in a stock
    one) from truncating one grid point short.
    Clamped at ``zmax``: ``nzplt`` beyond ``nz = zmax/dz - 0.5`` would
    index past the marched arrays.
    """
    dz = float(dz)
    ndz = max(1, int(ndz))
    n = int(np.ceil(max(float(target_depth), 0.0) / dz)) + 1
    # Round up onto an index the output loop actually visits; it starts at
    # ``base + ndz``, so at least one stride is always needed.
    base = depth_index_base(kind)
    n = base + max(1, int(np.ceil((n - base) / ndz))) * ndz
    return min((n + 0.75) * dz, float(zmax))


def collins_deepest_output(zmplt: float, dz: float,
                            ndz: int = 1, kind: str = 'ramgeo') -> float:
    """Deepest depth (m) a Collins binary stores for ``zmplt`` — the
    inverse of :func:`collins_zmplt`. ``-1.0`` when the output loop
    visits no index at all."""
    dz = float(dz)
    ndz = max(1, int(ndz))
    nzplt = int(float(zmplt) / dz - 0.5)
    base = depth_index_base(kind)
    # Loop indices are base + k·ndz for k >= 1, up to nzplt.
    k = (nzplt - base) // ndz
    return (base + k * ndz - 1) * dz if k >= 1 else -1.0


def within_march(nodes, max_range: float) -> list:
    """The ``(range, value)`` nodes a march to ``max_range`` consumes.

    The Collins binaries advance to the next bathymetry / altimetry node
    only once the march reaches it (``updat``: ``if(r.ge.rb(ib+1))``,
    ``ramgeo1.5.f:348``) and interpolate towards it until then, so every
    node up to ``max_range`` plus the first one beyond it shapes the field
    out to ``max_range``, and none after that does. Writing the whole
    track instead, a 600-node, 60 km track with receivers to 3 km is
    refused by the 505-node array guard, and nodes 1 m apart at 3.1 km
    cut ``dr`` from 27.5 m to 1.0 m on a 3 km run.
    """
    kept = [n for n in nodes if float(n[0]) <= max_range]
    beyond = [n for n in nodes if float(n[0]) > max_range]
    return kept + beyond[:1]


def rams_pressure_from_dilatation(depths, ranges, tl, pcomplex,
                                   range_segments, bathymetry, zs):
    """rams0.5's ``tl.grid`` / ``pcomplex.bin`` as pressure.

    rams marches the (dilatation, vertical displacement) pair of Collins'
    elastic PE and writes the dilatation ``u(2i-1)`` raw
    (``rams0.5.f:251,263,270``), its starter placing the delta in that
    same component at ``z_s`` (``rams0.5.f:360-361``). Pressure in a
    fluid is ``-lambda*Delta`` with ``lambda = rho*c**2`` (Collins 1991,
    continuity of ``lambda*Delta`` across a fluid interface), and the
    fluid codes multiply their field variable back by ``f3`` at output
    (``ramgeo1.5.f:420,430``), so rams alone returns ``p(z)/lambda(z)``
    on a source-normalised scale. Multiplying each sample by
    ``lambda(z, r) / lambda(z_s, 0)`` puts it on the fluid codes' scale:
    measured on a 100 m channel with a 1450->1550 m/s linear profile, raw
    rams minus ramgeo was +0.59 dB near the surface to -0.44 dB near the
    bed against -40*log10(c(z)/c(z_s)) predicted, and iso-speed water
    left 0.02-0.09 dB.

    ``lambda`` is the deck's own (``rams0.5.f:201-204``): ``c_w(z)**2``
    in the water (unit density), and below the seafloor the bulk modulus
    ``rho_b*(c_p**2 - 4/3*c_s**2)`` — ``lamb + 2/3*mub`` — so the
    sub-bottom rows carry minus the mean normal stress, which is the
    pressure when ``c_s = 0``. Each output range takes the profile
    section rams is marching at that range (``if(r.ge.rp)``,
    ``rams0.5.f:332``). The attenuation factors in rams' complex moduli
    are left out of the ratio: they change it by ``O(eta*alpha)``, below
    1e-3 dB for attenuations under 1 dB/lambda.
    """
    def interp(pairs, z):
        zz = np.array([float(a) for a, _ in pairs])
        vv = np.array([float(b) for _, b in pairs])
        return np.interp(z, zz, vv)

    depths = np.asarray(depths, dtype=float)
    ranges = np.asarray(ranges, dtype=float)
    markers = np.array([float(s['range']) for s in range_segments])
    bathy_r = np.array([float(r) for r, _ in bathymetry])
    bathy_z = np.array([float(d) for _, d in bathymetry])

    def modulus(seg, z, seafloor):
        water = interp(seg['water_ssp'], z) ** 2
        cp = interp(seg['bottom_c'], z)
        cs = interp(seg['bottom_cs'], z)
        rho = interp(seg['bottom_rho'], z)
        bulk = rho * (cp ** 2 - (4.0 / 3.0) * cs ** 2)
        return np.where(z <= seafloor, water, bulk)

    lam_source = float(modulus(range_segments[0], np.array([zs]),
                               bathy_z[0])[0])
    section = np.clip(np.searchsorted(markers, ranges, side='right') - 1,
                      0, len(range_segments) - 1)
    factor = np.empty((depths.size, ranges.size), dtype=float)
    for j, r in enumerate(ranges):
        seafloor = float(np.interp(r, bathy_r, bathy_z))
        factor[:, j] = modulus(range_segments[section[j]], depths,
                               seafloor) / lam_source
    pcomplex = np.asarray(pcomplex) * factor
    with np.errstate(divide='ignore', invalid='ignore'):
        tl = np.asarray(tl) - 20.0 * np.log10(np.abs(factor))
    return tl, pcomplex


def prepend_surface_node(depths, tl, pcomplex):
    """Prepend the ``z = 0`` node to a Collins output grid.

    ``rams0.5`` starts its output loop at grid index ``1+ndz`` and the
    fluid codes at ``ndz`` (``outpt`` in each source), so the shallowest
    stored sample sits at ``ndz·dz`` / ``(ndz-1)·dz`` and a receiver at
    the sea surface falls outside the interpolator's grid. The surface is
    pressure-release in every Collins backend, so the node carries no
    energy — an exact boundary value, not an extrapolation. The pressure
    is written as a literal zero and the TL as what the shared
    ``transmission_loss_dB`` floor turns that zero into, so this row reports
    the same no-energy level as every other model's, and no wrapper
    invents one of its own.
    """
    depths = np.asarray(depths, dtype=float)
    if depths.size == 0 or depths[0] <= 0.0:
        return depths, tl, pcomplex
    n_r = np.asarray(tl).shape[1]
    no_energy_dB = NO_ENERGY_DB
    return (
        np.concatenate([[0.0], depths]),
        np.vstack([np.full((1, n_r), no_energy_dB), np.asarray(tl)]),
        np.vstack([np.zeros((1, n_r), dtype=np.complex128),
                   np.asarray(pcomplex)]),
    )


def sediment_blocks(env: 'Environment', kind: str, zmax: float,
                     freq: float, *, knobs, speed_bounds):
    """Exactly the ``(depth, value)`` blocks the deck will carry.

    Taken from :func:`collins_range_segments` rather than rebuilt, because
    three of the deck builder's behaviours change the node arithmetic and a
    hand-rolled copy got all three wrong: the depths are written **relative
    to the seafloor** for ramgeo/ramsurf (:func:`collins_deck_base`'s
    ``z_top``) so an absolute-depth copy runs the arithmetic in the wrong
    frame; a pure half-space column is its two breakpoints and nothing
    else; and :func:`ramp_absorbing_attenuation` **adds** points to the
    attenuation block. One section per range break, each with its own
    seafloor.

    This is the one deck build that stays per-frequency: the ramp is what
    moves the block points around, and this method exists to see them.
    """
    blocks = []
    for segment in collins_range_segments(env, kind, zmax, freq,
                                          water=False, knobs=knobs,
                                          speed_bounds=speed_bounds):
        for key in ('bottom_c', 'bottom_rho', 'bottom_attn',
                    'bottom_cs', 'bottom_attns'):
            block = segment.get(key)
            if block:
                blocks.append([(float(z), float(v)) for z, v in block])
    return blocks


def block_loses_a_point(env: 'Environment', dz: float,
                         zmax: float, kind: str, freq: float, *, knobs,
                         speed_bounds) -> bool:
    """Whether ``zread`` would lose a sediment-block point at this ``dz``.

    This runs the vendored node assignment (``ramsurf1.5.f:200-211``) rather
    than a bound on it, because the natural sufficient bound
    (``gap >= BLOCK_GAP_PER_DZ * dz``) is far from necessary and would reject
    grids that are in fact clean: a 3 m step on ``dz = 2 m`` assigns nodes 1,
    3 and 4, and the node it skips is filled between two *equal* values.

    What is not clean is an **overwrite**. A 0.6 m layer over an 1800 m/s
    basement on the auto grid ``dz = 1.887 m`` puts the half-space value on
    node 1 — the seafloor — and the fill loop at :218-219 then ramps across
    the whole sub-bottom: the layer was marched as a 692 m gradient
    1500 → 1800 m/s, 22.6 dB from Scooter, on the default dispatch for any
    layered fluid bottom.
    """
    if dz <= 0:
        return False
    for block in sediment_blocks(env, kind, zmax, freq, knobs=knobs,
                                 speed_bounds=speed_bounds):
        assigned, previous = {}, None
        for depth, value in block:
            node = int(1.5 + depth / dz)
            if previous is not None and node == previous:
                node += 1                       # :208 collision push-down
            if node in assigned and assigned[node] != value:
                return True
            assigned[node] = value
            previous = node
    return False


def block_dz_cap(env: 'Environment', zmax: float, kind: str,
                  freq: float, *, knobs, speed_bounds) -> float:
    """A ``dz`` that ``zread`` is *guaranteed* to represent — the smallest
    positive block gap over :data:`BLOCK_GAP_PER_DZ`. Used only to pick a
    replacement once :func:`block_loses_a_point` has said the current grid
    fails, never to judge a grid: see that method for why the bound is
    sufficient but not necessary. ``0.0`` when there is no block to resolve.
    """
    gaps = set()
    for block in sediment_blocks(env, kind, zmax, freq, knobs=knobs,
                                 speed_bounds=speed_bounds):
        depths = [z for z, _ in block]
        gaps |= {b - a for a, b in zip(depths, depths[1:]) if b > a}
    if not gaps:
        return 0.0
    # The smallest gap always clears the collision, but it is often far finer
    # than needed: a gap whose two points carry the *same* value loses nothing
    # when they collide, and the absorbing-attenuation ramp routinely places
    # such a point within a decimetre of the sediment base. Take the coarsest
    # gap-derived candidate the exact predicate accepts — for a 0.9 m layer
    # that is 0.45 m rather than the 0.05 m the ramp's gap would have forced.
    candidates = sorted((g / BLOCK_GAP_PER_DZ for g in gaps), reverse=True)
    for candidate in candidates:
        if not block_loses_a_point(env, candidate, zmax, kind, freq,
                                   knobs=knobs, speed_bounds=speed_bounds):
            return candidate
    return candidates[-1]


def fit_dz_to_mz(env: 'Environment', kind: str, dz: float,
                  zmax: float, freq: Optional[float] = None, *,
                  notices=None, knobs) -> float:
    """Coarsen an auto-picked ``dz`` so the depth grid fits ``mz``.

    Mirrors the ``MAX_DEPTH_POINTS`` clamp in ``grid.compute_grid_lytaev``: an
    auto grid is uacpy's own choice, so a hard array bound it cannot meet
    is coarsened here rather than raised at the caller. A ``dz`` the caller
    pinned is left alone and rejected by ``check_collins_array_limits``.

    The replacement comes back through :func:`grid.align_dz_with_seafloor` so
    it keeps the seafloor where the backend wants it in its cell; the
    aligned value is only taken when it still fits ``mz``, since that
    bound is a hard array dimension.

    **The rams shear cap outranks this coarsening.**
    ``grid.compute_grid_lytaev`` tightens an elastic ``dz`` to ``λ_s/14``
    because a coarser grid does not merely lose accuracy — the elastic march
    diverges (measured 134 dB against OASES at ``0.55 λ_s`` on Collins 1991's
    own example D, against 0.83 dB at ``λ_s/14``). Coarsening past that cap
    here to fit an array bound therefore hands back a grid that produces no
    usable answer, which is why this refuses instead: at 3000 m / 300 Hz /
    ``c_s`` 300 m/s the cap is 0.0714 m against a fitted ``dz`` of 0.1564 m.
    Same policy, and the same shape, as the sediment-block refusal in
    :func:`resolve_collins_grid`. ``freq`` is what sizes the cap; without
    it (a caller with no frequency in scope) the cap cannot be evaluated
    and the auto ``dz`` is coarsened to fit.
    """
    budget = collins_mz_budget(kind, zmax)
    if budget is None or dz <= 0 or zmax <= 0:
        return dz
    needed, mz, dz_min = budget
    if needed(dz) <= mz:
        return dz
    dz_aligned = align_dz_with_seafloor(env, dz_min, kind=kind,
                                        coarsen=True, knobs=knobs)
    if dz_aligned >= dz_min and needed(dz_aligned) <= mz:
        dz_min = dz_aligned
    shear_cap = 0.0
    if kind == 'rams' and freq is not None:
        shear_cap = rams_dz_shear_cap(min_shear_speed(env),
                                      float(freq))
    if shear_cap > 0.0 and dz_min > shear_cap:
        raise ConfigurationError(
            f"RAM:rams: fitting the binary's depth arrays needs dz >= "
            f"{dz_min:.4f} m ({needed(dz)} slots needed, mz={mz}) over a "
            f"zmax={zmax:.1f} m domain, but the elastic march needs dz <= "
            f"{shear_cap:.4f} m to resolve the shear wavelength "
            f"(λ_s/14 at c_s={min_shear_speed(env):.0f} m/s, "
            f"f={float(freq):.0f} Hz). A grid coarser than that cap does "
            f"not lose accuracy, it diverges — measured 134 dB against "
            f"OASES at 0.55 λ_s — so this cannot be met by coarsening.",
            remediation=("Lower zmax, pin a dz at or below "
                         f"{shear_cap:.4f} m and shrink the domain to fit "
                         "mz, or model a shallower depth range."),
        )
    give_notice(notices,
        f"RAM:{kind}: raised dz from {dz:.4f} m to {dz_min:.3f} m to fit "
        f"the binary's depth arrays ({needed(dz)} slots needed, mz={mz}) "
        f"over a zmax={zmax:.1f} m domain. Lytaev accuracy budget "
        f"eps={knobs.accuracy:.0e} is no longer met — lower zmax, or use "
        f"backend='mpirams' (no fixed limit) to keep the finer grid.",
        FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )
    return dz_min


def check_source_below_depressed_surface(surface, zs: float,
                                          dz: float, *,
                                          notices=None) -> None:
    """Refuse (or warn about) a source at or above the depressed surface.

    ``matrc`` forces ``r2=1, r1=r3=s1=s2=s3=0`` on rows ``1..izsrf``
    (``ramsurf1.5.f:281-290``) with ``izsrf = 1.0 + zsrf/dz`` — a hard
    Dirichlet zero at and above the surface index. ``selfs`` plants the
    source delta at ``is = ifix(1 + zs/dz)`` and splits it across
    ``u(is)``, ``u(is+1)`` (``:396-400``) **without consulting ``izsrf``**,
    so a source inside the depressed region is zeroed the moment the
    march starts.

    The result is a field that is identically zero — TL ``inf``
    everywhere — with nothing to flag it:
    :func:`mark_diverged_collins_samples` cannot fire because ``|u| = 0`` gives
    a large POSITIVE TL, not a negative or NaN one.

    Two cases, deliberately handled differently:

    * the surface is at or below the source **at the source's own range**
      — the field is dead from the first step, so this is a bad
      configuration and raises;
    * a keel deeper than ``zs`` only further along the track — the field
      dies partway and reads as a plausible shadow zone, which is the more
      dangerous variant precisely because it looks like physics. That
      warns, since the near field before the keel is still meaningful.
    """
    if not surface:
        return

    def zeroed_to(zsrf: float) -> float:
        """Depth of the deepest row matrc actually zeroes for ``zsrf``.

        ``izsrf = 1.0 + zsrf/dz`` (``ramsurf1.5.f:115``) truncates on
        assignment to an integer, and ``matrc`` zeroes rows ``1..izsrf``
        (``:282``). Row ``i`` sits at ``(i-1)*dz``, so the zeroed region
        ends up to one ``dz`` **above** ``zsrf``. Comparing against
        ``zsrf`` itself refuses a source in that band, where the field is
        measurably alive — 84-91 dB with ``dz=0.7``, ``zsrf=30``,
        ``zs=29.7``.
        """
        return (int(1.0 + float(zsrf) / float(dz)) - 1) * float(dz)

    zsrf_at_source = zeroed_to(surface[0][1])
    deepest = max(zeroed_to(z) for _, z in surface)
    if zs <= zsrf_at_source:
        raise ConfigurationError(
            f"ramsurf: source at {zs:.4g} m is at or above the depressed "
            f"surface at r=0 ({zsrf_at_source:.4g} m). matrc zeroes every "
            f"row down to izsrf (ramsurf1.5.f:281-290) while selfs plants "
            f"the source without checking it (:396-400), so the field is "
            f"identically zero. outpt adds eps=1e-20 before the log "
            f"(ramsurf1.5.f:101), so this reports as ~414-437 dB rather "
            f"than inf — there is no NaN or inf to test for.",
            remediation=("Put the source below the deepest surface "
                         "depression, or reduce env.altimetry's depth."),
        )
    if zs <= deepest:
        give_notice(notices,
            f"ramsurf: source at {zs:.4g} m is shallower than the deepest "
            f"surface depression ({deepest:.4g} m). Where the keel reaches "
            f"below the source the field is forced to zero "
            f"(ramsurf1.5.f:281-290), which reads as a shadow zone rather "
            f"than as a configuration error. The dead cells report "
            f"~414-437 dB, not inf: outpt adds eps=1e-20 before the log "
            f"(ramsurf1.5.f:101), so no isnan/isinf check will find them.",
            ValidityWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )


def warn_if_ramsurf_crests(env, *, notices=None) -> None:
    """Warn that altimetry samples above z = 0 (wave crests) are clamped
    to z = 0: ramsurf1.5 models surface depressions only."""
    if env.altimetry is None:
        return
    crests = [(r, h) for r, h in env.altimetry.to_pairs() if float(h) > 0]
    if crests:
        give_notice(notices,
            f"ramsurf1.5 only models pressure-release surfaces at or "
            f"below z=0 (zsrf >= 0). {len(crests)} altimetry sample(s) "
            f"with height > 0 (wave crests above mean sea level) "
            f"clamped to z=0. For two-sided wave fields use Bellhop.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP
        )


def constrain_dr_to_sections(dr, range_segments, *, pinned,
                              bathymetry_ranges=None,
                              altimetry_ranges=None, notices=None, log):
    """Bound ``dr`` by the closest pair of profile-section markers.

    ``profl`` reads exactly ONE section marker per call
    (``ramgeo1.5.f:195`` ``read(1,*,end=1)rp``, defaulted to ``2.0*rmax`` at
    ``:194`` so an exhausted deck parks the marker past the end of the
    march), ``updat`` re-enters it
    only ``if(r.ge.rp)`` (``:359``), and the march calls ``updat`` once
    per range step (``:78-84``). So the binary can consume at most one
    section per ``dr`` — and because ``profl`` reads *sequentially* from
    the deck, sections written closer together than ``dr`` are not merely
    ignored, they are never reached. The march falls permanently behind
    and every later section is lost.

    Nothing downstream can detect this: the run exits 0 and writes a full
    grid, computed from a truncated environment. Measured on two decks
    differing only in range sampling of the same physics, the finer one
    was wrong by up to 8.75 dB — the deck carried 101 sections and the
    march could reach 19.

    The manual states the rule directly (``ram.pdf`` p.8): "The size of
    the smallest region is an upper bound on Delta-r."
    """
    markers = sorted({float(seg['range']) for seg in range_segments})

    # `updat` advances THREE indices the same way, one step at a time:
    # the profile marker (`if(r.ge.rp)`), the bathymetry index
    # (`ramgeo1.5.f:348` `if(r.ge.rb(ib+1))ib=ib+1`) and, on ramsurf, the
    # altimetry index (`ramsurf1.5.f:346`). Bounding only the first leaves
    # the other two to fall behind, after which the seafloor is linearly
    # extrapolated from a pair of points far astern for the rest of the
    # march — range dependence silently lost. mpiramS is immune to all
    # three: it interpolates (`ram.f90:211`) rather than consuming.
    for stream in (bathymetry_ranges, altimetry_ranges):
        if stream:
            markers = sorted(set(markers) | {float(r) for r in stream})

    if len(markers) < 2:
        return dr
    min_gap = min(b - a for a, b in zip(markers, markers[1:]))
    if min_gap <= 0.0 or dr <= min_gap:
        return dr
    # Strictly inside the gap, by more than the march's rounding can
    # lose: the binaries accumulate ``r = r + dr`` in implicit REAL
    # (``ramgeo1.5.f:56-60``), which the shipped build promotes to 8
    # bytes (``-fdefault-real-8``, ``install.sh``), so at ``dr == min_gap``
    # the running range can land a few ulps below its marker and the
    # bathymetry index (``:348``) then trails by one segment for the
    # whole march. A 1e-4 relative margin clears the drift of any march
    # the arrays allow, in either precision; the step count grows by
    # < 0.01 %.
    dr_out = min_gap * (1.0 - 1e-4)
    if pinned:
        give_notice(notices,
            f"RAM: dr={dr:.4g} m exceeds the closest profile-section "
            f"spacing ({min_gap:.4g} m), so the binary could consume only "
            f"part of the {len(markers)}-section environment and the rest "
            f"would be silently dropped (one section per range step; "
            f"ramgeo1.5.f:194-195, :359, :78-84). dr has been reduced to "
            f"{dr_out:.6g} m. Coarsen the environment's range axis to "
            f"keep the dr you asked for.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    else:
        log(
            f"dr reduced {dr:.4g} -> {dr_out:.6g} m so every one of the "
            f"{len(markers)} profile sections is reachable "
            f"(ram.pdf p.8: the smallest region bounds dr)."
        )
    return dr_out


def collins_range_segments(
    env: Environment, kind: str, zmax: float, freq: float,
    dz: Optional[float] = None, *, water: bool = True, knobs, speed_bounds
) -> list:
    """Build the Collins ``range_segments`` list — one ``ram.in`` profile
    section per range break — from the environment's range-dependent SSP
    and (layered) bottom, at one frequency.

    Two stages, so a broadband sweep pays for the expensive one once:
    :func:`collins_deck_base` builds everything that does not move with
    frequency and owns the deck's format, :func:`ramp_range_segments`
    finishes it at ``freq`` by ramping each section's attenuation block
    into the absorbing layer. Single-frequency callers — and the
    grid-resolution helpers, which run before ``zmax`` is settled — come
    through here and get both.
    """
    return ramp_range_segments(
        env, collins_deck_base(env, kind, zmax, knobs=knobs), freq,
        kind=kind, zmax=zmax, dz=dz, water=water, knobs=knobs,
        speed_bounds=speed_bounds
    )


def collins_deck_base(env: Environment, kind: str,
                       zmax: float, *, knobs) -> dict:
    """The part of the Collins deck that does not move with frequency.

    rams0.5 / ramsurf1.5 read a piecewise range-dependent ``ram.in``: an
    initial profile plus one extra ``(range, profile blocks)`` section per
    break. Sections carry the environment at the union of the bottom's and
    the SSP's range breakpoints; a range-independent axis contributes its
    single column at every break.

    A section takes effect from its marker range on — ``if(r.ge.rp)``
    (``ramgeo1.5.f:359``, ``ramsurf1.5.f:364``, ``rams0.5.f:332``) — so the
    marker is written at the midpoint between consecutive breakpoints. That
    makes the switch happen where ``Bottom.at`` / mpiramS put it: mpiramS
    marches with the profile nearest the current range
    (``minloc(abs(rp-rint))``, ``mpiramS/src/ram.f90:231`` for the SSP and
    ``:241`` for the sediment), and ``uacpy.core.bottom.Bottom.at`` is
    documented nearest. The bottom switches midway between ITS OWN
    breakpoints, which is not a union midpoint when SSP breaks fall
    between two bottom breaks, so each such switch also opens a section
    of its own, carrying the column past it; all four backends then
    transition at the same range for the same ``Environment``.

    Bottom-profile depth reference differs per binary: ramgeo/ramsurf
    ``matrc`` restarts the profile index at the grid point below the local
    seafloor, so their ``z cb/rhob/attn`` blocks are **depth below the
    seafloor** (layers track the bathymetry — RAMGEO's defining feature);
    rams0.5 indexes absolutely from z=0. Water-SSP blocks are absolute for
    all three.

    Frequency enters the deck at two places — the absorbing layer's width
    (:func:`_domain.absorbing_layer_thickness`), which reaches it only through
    each section's ramped attenuation block
    (:func:`ramp_absorbing_attenuation`), and the water-attenuation block
    (:func:`_domain.water_attenuation_block`), which is ``alpha(f)`` outright.
    Every other quantity above is a function of the environment and ``zmax``
    alone, so a broadband sweep cuts this once for the band and re-ramps per
    bin — the whole cost of the deck minus the ramp.

    Returns ``{'kind', 'zmax', 'segments', 'ramps'}``. ``segments``
    holds each section's finished frequency-invariant blocks; ``ramps``
    holds, per section, the *unramped* attenuation block, the two
    depths that bound its ramp and the section's seafloor (all in the
    geometric frame) — kept rather than the ramped result
    because the ramp changes the block's LENGTH with frequency (see
    :func:`ramp_range_segments`). ``kind`` and ``zmax`` are carried so a
    caller reusing the payload across a sweep is checked, not trusted:
    every section's depth extent is cut against ``zmax``.
    """
    properties = (
        ('sound_speed', 'shear_speed', 'density',
         'attenuation', 'shear_attenuation')
        if kind == 'rams'
        else ('sound_speed', 'density', 'attenuation')
    )
    b = env.bottom
    seafloor_relative = is_seafloor_relative(kind)
    rho_w = float(env.water_density)

    breaks = {0.0}
    if b.is_range_dependent:
        breaks.update(float(r) for r in b.ranges)
    if env.ssp.is_range_dependent:
        breaks.update(float(r) for r in ssp_range_axis(env))
    # rams0.5 reads the bottom arrays at absolute depth indices
    # (``rams0.5.f:490-516``, ``do 6 i=iz+1,ib+2`` reading ``lamb(i)``),
    # so a layered elastic column stays where the first section put it
    # and stops following the seafloor. ramgeo/ramsurf re-anchor at the
    # local seafloor in ``matrc`` (``ramgeo1.5.f:262-268``, ``ii=1 …
    # ii=ii+1``) and need no extra sections. A half-space column is
    # immune either way — both of its breakpoints carry the same value.
    if (not seafloor_relative and b.is_layered
            and env.bathymetry.varies_with_range):
        breaks.update(bathy_anchor_ranges(env, b))
    ranges = sorted(breaks)

    # Where Bottom.at switches columns: midway between two of its
    # breakpoints.
    switches = []
    if b.is_range_dependent:
        b_ranges = sorted(float(r) for r in b.ranges)
        switches = [0.5 * (a + c) for a, c in zip(b_ranges, b_ranges[1:])]

    # Each union breakpoint owns the cell between the midpoints around
    # it; a bottom switch strictly inside a cell splits it into
    # sections, all on the cell's seafloor and SSP. A section that holds
    # its breakpoint samples the bottom there, as an unsplit cell does;
    # any other — or one whose breakpoint is itself a switch, where
    # ``Bottom.at`` ties — at its own middle (just past its start when it
    # runs to the end of the track), so each carries the column nearest
    # to every range it covers.
    sections = []
    for i, rng in enumerate(ranges):
        # Section 0 is the initial profile; write_ramin ignores its range.
        start = rng if i == 0 else 0.5 * (ranges[i - 1] + rng)
        end = (0.5 * (rng + ranges[i + 1]) if i + 1 < len(ranges)
               else float('inf'))
        cuts = [start] + [x for x in switches if start < x < end] + [end]
        for lo, hi in zip(cuts, cuts[1:]):
            if lo <= rng < hi and rng not in switches:
                at = rng
            elif np.isfinite(hi):
                at = 0.5 * (lo + hi)
            else:
                at = lo + 1.0
            sections.append((lo, rng, at))

    segments, ramps = [], []
    for marker, rng, bottom_at in sections:
        seafloor = float(np.asarray(env.bathymetry.eval(range=rng)).flat[0])
        # A pure half-space column is written as its two breakpoints:
        # ``zread`` interpolates any block onto the grid and ``matrc``
        # reads per-node values (``ramgeo1.5.f:209-311``), so no layer is
        # needed above the half-space.
        col = b.at(range=bottom_at)
        z_top = 0.0 if seafloor_relative else seafloor
        z_bottom = (zmax - seafloor) if seafloor_relative else zmax
        bp = piecewise_breakpoints(
            col,
            seafloor_depth=z_top,
            zmax=z_bottom,
            properties=properties,
        )
        ssp_pairs = (
            env.ssp.eval(range=rng).to_pairs()
            if env.ssp.is_range_dependent else env.ssp.to_pairs()
        )
        # Every block is cut in the geometric frame and mapped to the
        # deck's frame as the last step (``_domain.deck_block``), so the two
        # frames never mix inside one deck.
        def block(pairs):
            return deck_block(pairs, seafloor, seafloor_relative, knobs=knobs)
        ssp_cut = cut_water_ssp_at_zmax(ssp_pairs, zmax)
        seg = dict(
            range=float(marker),
            water_ssp=deck_water_column(ssp_cut, knobs=knobs),
            # Kept geometric for the water-attenuation block, whose
            # local wavelength is c(z)/f; not written to the deck.
            ssp_geo=ssp_cut,
            bottom_c=block(bp['sound_speed']),
            # Every RAM code fixes the water density at 1 and reads
            # the seabed's as a ratio to it (RAM guide: 'the density
            # is assigned the value 1 g/cc'), so the absolute g/cm³
            # the Bottom carries is divided by the water's.
            bottom_rho=block([(z, v / rho_w) for z, v in bp['density']]),
        )
        if kind == 'rams':
            seg['bottom_cs'] = block(bp['shear_speed'])
            seg['bottom_attns'] = block(bp['shear_attenuation'])
        segments.append(seg)
        ramps.append((bp['attenuation'], z_top + col.total_thickness(),
                      z_bottom, seafloor))
    return dict(kind=kind, zmax=float(zmax), segments=segments,
                ramps=ramps)


def ramp_range_segments(env: Environment, base: dict,
                         freq: float, *, kind: str, zmax: float,
                         dz: Optional[float] = None,
                         water: bool = True, knobs, speed_bounds) -> list:
    """Finish a :func:`collins_deck_base` payload at one frequency.

    The ramp is recomputed per frequency rather than rescaled because the
    attenuation block changes **length**, not just values: it keeps every
    control point at or above the ramp's start depth
    ``max(z_sediment_base, zmax - absorbing_width)``, and that count moves
    as ``absorbing_width ∝ 1/f`` slides the start across the sediment
    base. On the range-dependent elastic case the block-length tests
    count over, the block is 4 points at 50 Hz and 5 at 800 Hz — and
    *both* at 105 Hz, in the same deck, because the seafloor (hence
    ``z_sediment_base``) moves with range. Anything that pre-allocates one
    shape for the band is wrong.

    Sections come back as fresh dicts over copied blocks, so the base
    survives whatever a single bin's deck is handed to.

    ``kind`` / ``zmax`` are the grid this deck is about to be written for; they
    must match the ones the base was cut against. ``dz`` thins the
    water-attenuation block to one point per depth cell when known.
    ``water=False`` leaves the water block out, for callers that read only the
    sediment blocks; a broadband base carrying ``water_alpha``
    (:func:`_domain.water_alpha_band`) supplies the block's alpha for the
    sweep.
    """
    if base['kind'] != kind or base['zmax'] != float(zmax):
        raise ConfigurationError(
            f"RAM: a Collins deck cut for kind={base['kind']!r}, "
            f"zmax={base['zmax']!r} m was reused at kind={kind!r}, "
            f"zmax={float(zmax)!r} m. Section depths, the half-space "
            f"extent and the absorbing ramp are all cut against zmax, so "
            f"the deck describes a different domain.",
            f"Rebuild the base with collins_deck_base(env, {kind!r}, "
            f"{float(zmax)!r}) for the grid the deck is written for.")
    absorbing_width = absorbing_layer_thickness(env, freq, knobs=knobs,
                                                speed_bounds=speed_bounds)
    water_attn = water and water_attenuation_active(env)
    out = []
    seafloor_relative = is_seafloor_relative(kind)
    for seg, (attn, z_sediment_base, z_bottom, seafloor) in zip(
            base['segments'], base['ramps']):
        done = dict(
            range=seg['range'],
            water_ssp=list(seg['water_ssp']),
            bottom_c=list(seg['bottom_c']),
            bottom_rho=list(seg['bottom_rho']),
            bottom_attn=deck_block(
                ramp_absorbing_attenuation(
                    attn, z_sediment_base, z_bottom, absorbing_width,
                    knobs=knobs),
                seafloor, seafloor_relative, knobs=knobs),
        )
        if 'bottom_cs' in seg:
            done['bottom_cs'] = list(seg['bottom_cs'])
            done['bottom_attns'] = list(seg['bottom_attns'])
        if water_attn:
            # Absolute depth on every backend (``wattn`` fills the
            # water rows 1..iz from it), so the mapping is the
            # absolute-frame branch of ``_domain.deck_block``.
            done['water_attn'] = deck_block(
                water_attenuation_block(
                    env, freq, seg['ssp_geo'], zmax, dz,
                    band=base.get('water_alpha')),
                seafloor, False, knobs=knobs)
        out.append(done)
    return out


def bathy_anchor_ranges(env, bottom) -> list:
    """Ranges at which to re-anchor an absolute-depth bottom profile.

    Sections are emitted so the seafloor never moves by more than half the
    thinnest sediment layer between two of them — the bound at which the
    layer would start to slide off its own interval. The bathymetry's own
    control points are not enough: a two-point linear slope has none in
    between, and that is exactly where the layer detaches.
    """
    r_axis = np.atleast_1d(np.asarray(env.bathymetry.ranges, dtype=float))
    r_end = float(np.max(r_axis))
    if not r_end > 0.0:
        return []
    # Only layer thicknesses are read here, so the columns are read live
    # through ``Bottom.column_index_at``: ``Bottom.at`` would deep-copy a
    # whole layer stack per bathymetry node for one float per layer.
    # Which column a node resolves to is still ``Bottom``'s nearest rule
    # to own, not this method's to re-derive. Several nodes resolving to
    # one column contribute it once — the set is a de-duplication, and
    # ``min`` over the thicknesses cannot see the difference.
    indices = (sorted({bottom.column_index_at(range=float(r))
                       for r in r_axis})
               if bottom.is_range_dependent else [0])
    thicknesses = [
        float(layer.thickness)
        for i in indices
        for layer in bottom.columns[i].layers
        if float(layer.thickness) > 0.0
    ]
    if not thicknesses:
        return []
    tol = 0.5 * min(thicknesses)
    probe = np.linspace(0.0, r_end, 1024)
    # One vectorised query, not 1024 scalar ones: ``Bathymetry.eval`` takes a
    # whole range axis and ``_query_profile`` has no separate scalar path
    # for any of its methods, so the samples are the same doubles.
    floor = np.asarray(env.bathymetry.eval(range=probe), dtype=float)
    out, anchor = [], floor[0]
    for r, z in zip(probe[1:], floor[1:]):
        if abs(z - anchor) >= tol:
            out.append(float(r))
            anchor = z
    if len(out) > MAX_BATHY_SECTIONS:
        idx = np.linspace(0, len(out) - 1, MAX_BATHY_SECTIONS)
        out = [out[int(round(i))] for i in idx]
    return out


def ramp_absorbing_attenuation(pairs, z_sediment_base, z_bottom,
                                absorbing_width, *, knobs):
    """Ramp a Collins attenuation block into the artificial absorbing layer.

    ``zread`` (``rams0.5.f:212``, and the same routine in ramgeo1.5.f /
    ramsurf1.5.f) linearly interpolates a ``(depth, value)`` block onto the
    depth grid, so replacing the block's tail with two points
    ``(z_abs, attn_local)`` and ``(z_bottom, absorber_attenuation)`` gives
    the linear ramp Collins' own readme prescribes
    (``third_party/ramsurf/readme.orig:127-134``) — without it the flat
    half-space attenuation runs to the domain floor and energy reaching
    it reflects back into the field. This mirrors what mpiramS gets from
    ``attn[-1] = absorber_attenuation`` at ``zmax`` over its own linearly
    interpolated sediment profile.

    The ramp starts at ``max(z_sediment_base, z_bottom - absorbing_width)``
    so it never eats into the modelled sediment column. Depths are in
    whichever frame the backend reads (seafloor-relative for
    ramgeo/ramsurf, absolute for rams0.5).
    """
    pairs = [(float(d), float(v)) for d, v in pairs]
    if not pairs or absorbing_width <= 0.0:
        return pairs
    z_abs = max(float(z_sediment_base),
                float(z_bottom) - float(absorbing_width))
    if not z_abs < float(z_bottom):
        return pairs
    # Keep every control point down to and including z_abs. The block
    # carries duplicated abscissae at each interface — ``(base, layer)``
    # then ``(base, half-space)`` at the deepest layer's base — and the
    # value entering the ramp is the LAST pair at or above z_abs. When
    # the ramp is clamped to the sediment base that is the half-space's
    # value, which is the medium the ramp starts in; a strict `<` would
    # drop both pairs and lose the step.
    head = [p for p in pairs if p[0] <= z_abs]
    attn_local = (head[-1][1] if head
                  else float(np.interp(z_abs, [d for d, _ in pairs],
                                       [v for _, v in pairs])))
    if not head or head[-1][0] < z_abs:
        head = head + [(z_abs, attn_local)]      # pin the ramp's start
    attn_floor = max(attn_local, float(knobs.absorber_attenuation))
    return head + [(float(z_bottom), attn_floor)]


def check_collins_array_limits(kind: str, dz: float, zmax: float,
                                bathymetry, surface=None) -> None:
    """Reject a run that would overrun a Collins binary's fixed arrays.

    ``mr`` bounds the range arrays ``rb``/``zb`` (and ``rsrf``/``zsrf`` on
    ramsurf); ``mz`` bounds the depth arrays, at the per-backend rate
    recorded in :data:`_COLLINS_ARRAY_LIMITS`.

    The profiles actually written are passed in rather than re-derived from
    ``env``: the writer appends a point to reach the last receiver, and the
    Fortran read loop stores the ``-1 -1`` terminator at index ``N+1``
    before testing ``i.gt.mr`` (``ramgeo1.5.f:146``, ``ramsurf1.5.f:131``),
    so the true capacity is ``mr - 1`` written points.

    Each binary does test its own depth bound and ``stop`` with a
    diagnostic, but exits before writing ``tl.grid``, so the failure would
    otherwise reach the caller as a ``FileFormatError`` about a truncated
    file with the real message lost. ``ramsurf`` does not even check its
    surface arrays — ``ramsurf1.5.f:131`` is fed by the *bathymetry*
    counter — so an over-long altimetry corrupts memory instead. Only a
    ``dz`` the caller pinned can reach the depth check; an auto-picked grid
    is coarsened to fit in ``resolve_collins_grid``.
    """
    limits = _COLLINS_ARRAY_LIMITS.get(kind)
    if limits is None:
        return
    mr = limits['mr']
    for label, profile in (('bathymetry', bathymetry),
                           ('altimetry', surface)):
        if profile is None:
            continue
        # +1 for the terminator the Fortran stores past the last point.
        if len(profile) + 1 > mr:
            raise ConfigurationError(
                f"RAM(backend={kind!r}): the run writes {len(profile)} "
                f"{label} points but the binary's arrays hold {mr} "
                f"(mr={mr}, one slot taken by the list terminator), so it "
                f"would overrun them. Decimate env.{label} to at most "
                f"{mr - 1} points, or use backend='mpirams' (no fixed "
                f"limit)."
            )
    needed, mz, dz_min = collins_mz_budget(kind, zmax)
    if needed(dz) > mz:
        raise ConfigurationError(
            f"RAM(backend={kind!r}): dz={dz:g} m over a zmax={zmax:.1f} m "
            f"domain needs {needed(dz)} depth slots but the binary's "
            f"arrays hold {mz} (mz={mz}). Coarsen to dz>={dz_min:.4g} m, "
            f"lower zmax, or use backend='mpirams' (no fixed limit)."
        )


def check_rams_seafloor_index_floor(kind: str, dz: float,
                                     bathymetry) -> None:
    """Reject a track that drives ``rams0.5``'s seafloor index below 2.

    ``rams0.5.f:135`` and ``:305`` both assign ``iz=z/dz`` with no clamp,
    where ``ramgeo1.5.f:134`` and ``ramsurf1.5.f:119`` apply
    ``max(2,iz)``. Below 2 the binary indexes before the start of its own
    arrays: ``matrc:416`` runs ``do 2 i=ia-1,iz`` with ``ia=min(iz,jz)=1``
    and reads ``lamw(0)``, and at ``iz=1`` ``matrc:717`` forms
    ``i0=2*iz-3=-1`` and reads ``r5(0)``.

    The reason this is worth refusing rather than warning: at ``iz=1`` the
    run exits 0 and returns a **fully finite, entirely plausible** field.
    A bounds-checked build of the same deck aborts. Only ``iz=0`` is loud
    (all-NaN).

    ``updat`` recomputes the index from the *interpolated* bathymetry at
    every range step (``:305``), so the shallowest point anywhere on the
    track decides this — not ``env.depth``, which is the deepest. Every
    other depth guard in this class is written against the maximum, which
    is why this one is separate.
    """
    if kind != 'rams' or not bathymetry:
        return
    shallowest = min(float(d) for _, d in bathymetry)
    if shallowest >= 2.0 * float(dz):
        return
    raise ConfigurationError(
        f"rams: the track shoals to {shallowest:.4g} m, which is less "
        f"than 2*dz ({2.0 * float(dz):.4g} m), so the seafloor index "
        f"iz=z/dz falls below 2. rams0.5 does not clamp it "
        f"(rams0.5.f:135, :305 — unlike ramgeo1.5.f:134 / "
        f"ramsurf1.5.f:119), and matrc then reads lamw(0) at :418 and "
        f"r5(0) at :717. At iz=1 the run still exits 0 and returns a "
        f"plausible finite field, so nothing downstream can catch it.",
        remediation=(f"Set dz <= {shallowest / 2.0:.4g} m, or use "
                     f"backend='ramgeo'/'ramsurf' (which clamp) if the "
                     f"seabed needs no shear."),
    )


def collins_output_stride(dr: float, max_range: float, rcv_ranges,
                           beat_wavenumber: float = 0.0):
    """``(ndr, rmax_march)`` — the Collins range-output stride and the
    ``rmax`` to write into the input deck.

    The binaries write only at ``r = k·dr·ndr`` and test ``r < rmax``
    *after* writing (``rams0.5.f:79-80``, ``ramsurf1.5.f:52-53``,
    ``ramgeo1.5.f:83-84``), so a decimated march handed ``rmax =
    max_range`` verbatim stops up to ``(ndr-1)·dr`` short of the farthest
    receiver and the outermost receiver column comes back NaN.
    ``rmax_march`` extends the march to the first written range at or
    beyond ``max_range``, and is set half a range step short of it so the
    stop test fires exactly on that write — rounding drift in the
    binary's ``r = r + dr`` sum (REAL*8 in the shipped build, REAL*4 in a
    stock one) cannot then cost the final record.

    ``ndr`` is capped so the *first* written range ``dr·ndr`` is not past
    the nearest receiver, and so the output spacing ``dr·ndr`` stays at
    or below :func:`collins_output_spacing` of ``beat_wavenumber``
    (:func:`_domain.modal_beat_wavenumber`) — the receiver interpolation in
    :func:`interp_envelope_to_receiver_grid` reads the modulus between
    written ranges (0.7 dB at 1 kHz / 20 km between ``ndr`` 1 and 2 on
    one march). Both caps yield to the ceiling on the number of output
    ranges.
    """
    ndr = max(1, int((max_range / dr) / 1000.0))
    rr = np.atleast_1d(np.asarray(rcv_ranges, dtype=float))
    near = rr[rr > 0.0]
    if near.size:
        ndr = max(1, min(ndr, int(np.floor(float(near.min()) / dr))))
    spacing = collins_output_spacing(beat_wavenumber)
    if np.isfinite(spacing):
        ndr = min(ndr, max(1, int(np.floor(spacing / dr))))
    ndr = max(ndr, int(np.ceil(max_range / dr / _COLLINS_MAX_OUTPUT_RANGES)))
    block = dr * ndr
    # The epsilon absorbs the rounding of ``max_range / block`` when the
    # two divide exactly: ceil() on a value a few ulps above the integer
    # would buy a whole extra output block.
    n_blocks = max(1, int(np.ceil(max_range / block - 1e-9)))
    return ndr, float((n_blocks * ndr - 0.5) * dr)


def collins_mz_budget(kind: str, zmax: float):
    """``(needed, mz, min_dz)`` for one Collins depth grid.

    ``needed`` is a callable ``needed(dz) -> int`` giving how many ``mz``
    slots the binary's own ``nz = zmax/dz - 0.5`` consumes at that ``dz``,
    per the backend's indexing; ``min_dz`` is the coarsest-resolution
    bound that fits, with half a grid point of headroom against the
    Fortran truncation.
    """
    limits = _COLLINS_ARRAY_LIMITS.get(kind)
    if limits is None:
        return None
    nz_max = (limits['mz'] - limits['nz_pad']) // limits['nz_factor']

    def needed(dz):
        nz = int(zmax / dz - 0.5)
        return limits['nz_factor'] * nz + limits['nz_pad']

    return needed, limits['mz'], zmax / (nz_max - 0.5)
