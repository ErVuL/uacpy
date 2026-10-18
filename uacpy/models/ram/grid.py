"""The (dz, dr) grid of a RAM run: the Lytaev optimiser driver and the
constraints it applies, the seafloor and surface placement of the depth
nodes, and the notices of what the grid cannot resolve."""

import numpy as np
from typing import Optional
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.models._notices import give_notice
from uacpy.models._window import steep_path_clear_range, steep_path_cut
# The Padé optimiser's own Δz ladder floor, imported (not duplicated) so the
# relaxation warning can contrast it with the wrapper's binding cost floor
# ``LAMBDA_PER_DZ_FLOOR`` below and the two cannot drift apart.
from uacpy.models.pe_grid import (
    dz_ladder_end, grid_error, GridInfeasibleError,
    optimize_grid, rams_dz_shear_cap,
)
from uacpy.core.environment import (
    Environment,
)
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, NumericsWarning,
)
from uacpy.models.ram._domain import (
    COLLINS_SAMPLES_PER_BEAT,
    MAX_DEPTH_POINTS,
    RAMS_CN_DRIFT_BUDGET_DB,
    band_speeds,
    deck_depth,
    min_shear_speed,
    modal_beat_wavenumber,
    resolve_angle_max,
    steepest_slope_deg,
    water_speed_bounds,
)
from uacpy.models.ram._stability import (
    rams_dr_for_level_drift,
    rams_rotation_remedy,
    rams_stability,
    resolve_rams_rotation_angle,
)
from uacpy.models._checks import speed_bounds


#: Depth samples per wavelength of the slowest medium at which the
#: automatic depth search starts, ``dz = c_min/(15·f)`` where the optimiser
#: asks for finer: the Acoustics Toolbox's RAM writers' depth step
#: (``Matlab/Slicer/write_ram_in.m:18`` ``nseg_lambda = 15``,
#: ``Matlab/ReadWrite/write_env_RAM.m:24`` ``dz = 1500 / freq / 15``; Collins'
#: sample deck, 2 m at 50 Hz in 1480 m/s water, is λ/14.8). Measured where it
#: binds (100 m of water, 9 × 19 receivers to 5 km, rms dB from
#: ``Scooter(c_high=1e9)``): silt 1550 m/s at 50 Hz 0.865 → 0.914 (mpiramS)
#: and 1.384 → 1.404 (ramgeo) from λ/16 to λ/15, clay 1480 m/s at 100 Hz
#: unchanged (0.102, 0.251), for 7 % fewer depth points.
LAMBDA_PER_DZ_FLOOR = 15.0
#: Lytaev score (accumulated per-step error bound of the steepest scored
#: component) at or above which a grid does not carry the seabed's trapped
#: modes: the score saturates at τ ≤ 2 per step, so from 1 up it ranks
#: nothing and the mode's phase is lost over the march. Measured on a 2400 m/s
#: rock at 100 / 200 Hz over 5 km, 9 × 19 receivers: the λ/16 grid scores
#: 8 / 16 and reads 2.8 / 6.3 dB rms from Kraken; the coarsest grid under 1
#: (λ/45 / λ/64), with the seafloor placed a quarter cell below its node,
#: reads 0.8 / 1.4 dB, where sand (score 0.44) reads 0.8.
TRAPPED_MODE_SCORE_LIMIT = 1.0
#: Lytaev ε the automatic grid's relaxation ladder stays strictly below: the
#: same saturation point as :data:`TRAPPED_MODE_SCORE_LIMIT`, past which the
#: score ranks nothing. Every rung below it is a grid that runs, with its
#: rescored error reported; only a band no rung under it reaches is refused.
EPS_LADDER_CEILING = TRAPPED_MODE_SCORE_LIMIT
#: Lytaev ε up to which a relaxation is routine (forced at and above ~500 Hz
#: by the optimiser's own 0.01 m depth-ladder end) and reported as a log line
#: at the default accuracy; a grid that needed a rung above it warns.
ROUTINE_EPS_RELAXATION = 0.5


#: Where the automatic grid puts the seafloor inside a depth cell on the
#: fluid backends (mpiramS, ramgeo, ramsurf): the fraction of a cell the
#: seafloor sits BELOW its last water node, ``h/dz = n + offset``. Those
#: codes take nodes ``1..iz = int(1 + h/dz)`` as water and split the
#: properties between ``iz`` and ``iz+1`` (``matrc``), so the Galerkin
#: interface of an ON-node seafloor sits half a cell below it. Measured on
#: the 100 m channel (source 25 m, 9 receiver depths × 19 ranges over
#: 0.5-5 km), rms dB from Kraken over 1-5 km (mpiramS, n_pade 8) at
#: placements 0 / 0.25 / 0.5 / 0.75 of a cell: sand 200 Hz on the λ/15 grid
#: 1.04 / 0.38 / 0.74 / 1.55; rock 100 Hz at λ/45 3.15 / 0.56 / 2.67 / 4.19;
#: rock 200 Hz at λ/64 2.24 / 0.69 / 3.13 / 4.14; hard rock 100 Hz at λ/104
#: 3.99 / 0.97 / 2.14 / 4.61 and at λ/250 2.19 / 0.71 / 0.82 / 1.87; hard
#: rock 200 Hz at λ/148 3.32 / 0.66 / 3.30 / 5.16 and at λ/250 3.55 / 0.96 /
#: 1.59 / 3.70. On these seabeds the node is never the best placement and
#: a quarter cell wins in every case; on slower seabeds another fraction
#: can win (0.375 on an 80 m / 150 Hz / cb 1600 channel, see
#: :data:`SEAFLOOR_QUARTER_TOLERANCE`, and on a 50 m / 150 Hz / cb 1650
#: lossless guide, docs/models/ram.md §6). The value is uacpy guidance read
#: off these measurements; no cited source explains why one placement
#: wins. rams0.5 is different: ``iz = int(h/dz)`` and the fluid–solid interface
#: conditions are applied at node ``iz+1``, on the seafloor node itself,
#: so it keeps the on-node placement (:func:`snap_dz_to_seafloor`).
SEAFLOOR_CELL_OFFSET = 0.25
#: How far (a fraction of a cell, either way round the cell) a pinned
#: ``dz`` may put the shallowest seafloor from the quarter-cell placement
#: :data:`SEAFLOOR_CELL_OFFSET` before the fluid backends warn
#: (:func:`warn_if_pinned_dz_misplaces_the_seafloor`). uacpy's guidance, not
#: a property of the binaries: they march the same seafloor for every depth
#: inside one cell, and the tolerance is read off measurements — mpiramS
#: against Kraken, rms dB over 9 x 16 receivers to 8 km, the seafloor ``d``
#: of a cell from the quarter:
#: 120 m / 120 Hz / cb 1680 reads 0.49 at the quarter, 0.78 / 0.97 at
#: d = 0.075 (above / below it), 1.26 / 1.40 at d = 0.125, 2.84 on the node
#: and up to 4.56 at frac 0.95; 100 m / 200 Hz / cb 1650 reads 0.96, 0.98 /
#: 1.33, 1.30 / 1.64, 1.92, 4.59. Within 0.1 cell the excess over the
#: quarter stays under 0.7 dB, and past it grows about 5-7 dB per cell of
#: distance. It is a rule of thumb, not a bound: on an 80 m / 150 Hz / cb
#: 1600 channel the best placement sat at frac 0.375 (0.40 against 0.63 at
#: the quarter), which it warns on.
SEAFLOOR_QUARTER_TOLERANCE = 0.1
#: ramsurf1.5 puts its pressure-release surface at ``floor(zsrf/dz)·dz``
#: (:func:`surface_node_misfit`): the fraction of a cell the automatic
#: grid leaves between the surface asked for at ``r = 0`` and the one
#: marched, found by moving ``dz`` by at most ``SURFACE_ALIGN_WINDOW``
#: (:func:`surface_aligned_layers`).
SURFACE_NODE_TOLERANCE = 0.05
SURFACE_ALIGN_WINDOW = 0.25
#: Depth cells below which ramsurf1.5 staircases the altimetry's deepest
#: depression too coarsely to leave it unsaid.
MIN_SURFACE_RELIEF_CELLS = 4.0


#: Steps per wavelength a rams0.5 range step is held to,
#: ``dr ≤ c_min/(RAMS_DR_LAMBDA_CAP·f)`` ≈ 0.2 λ: the truncation requirement
#: of the rotated-Padé march's O(dr²) step, uacpy guidance, measured — τ·n ≈
#: 0.1 there against 14.6 at the Lytaev ``dr`` (:func:`tighten_rams_dr`).
RAMS_DR_LAMBDA_CAP = 5.0
#: Divisor an unset ``rams_dr_factor`` applies to the Lytaev ``dr`` on rams
#: (:func:`tighten_rams_dr`): uacpy guidance, chosen with the λ/5 cap and
#: the stability rule beside it.
DEFAULT_RAMS_DR_FACTOR = 5.0


def collins_output_spacing(beat_wavenumber: float) -> float:
    """Largest Collins output spacing (m) that samples the modal beat
    ``COLLINS_SAMPLES_PER_BEAT`` times per period; ``inf`` without a beat."""
    if beat_wavenumber <= 0.0:
        return float('inf')
    return float(2.0 * np.pi / beat_wavenumber / COLLINS_SAMPLES_PER_BEAT)


# Narrowest source aperture the auto-loosening below will fall back to.
# 30° is the aperture of the worked example in Lytaev (2023) §5.1, and the
# default here; the 15° floor is uacpy's own choice: under 15° the
# operator is essentially paraxial and "wide-angle" stops meaning
# anything, so a caller who genuinely wants narrow-angle physics has to
# say so via ``angle_max_deg=`` rather than reach it by relaxation. A pinned
# ``angle_max_deg`` is never narrowed.
_THETA_MAX_FLOOR = 15.0


def optimize_grid_relaxing(*, frequency, c_min, c_max, max_range,
                            c0_pe, eps0, theta0, kind, dz_floor=0.0,
                            c_min_all=None, c_max_all=None, knobs):
    """Run the Lytaev optimizer, loosening its inputs until one converges.

    ``c_min`` / ``c_max`` bound the accuracy band, ``c_min_all`` /
    ``c_max_all`` the whole medium (the stability band).

    A hard environment (deep ocean, high ``c_max``, wide ``θ_max``) admits
    no grid meeting ``eps0`` under the Collins second-order Numerov, and
    :func:`~uacpy.models.pe_grid.optimize_grid` signals that with a
    ``GridInfeasibleError``. Two nested fallbacks, tried in this order because
    a looser error budget still describes the requested physics whereas a
    narrower aperture no longer does:

    1. ``ε`` is tripled while the next rung stays below
       :data:`EPS_LADDER_CEILING` (from the default 1e-3 that is seven
       rungs, the last at 0.729). The ceiling is where the score stops
       meaning anything: ``τ·n_steps`` bounds the steepest scored
       component's accumulated error, and at 1 that is an O(1) error the
       score saturates on. Below it a relaxed grid still runs, with the
       rescored error reported (the field error is 3-10× under the
       bound; at 2 kHz over 50 m of water to 2 km the 0.729 rung marches
       0.16-0.38 dB median from Scooter where the 0.243 ceiling refused);
    2. only if the whole ε ladder failed and ``angle_max_deg`` is unset,
       ``θ_max`` steps down through 20° and :data:`_THETA_MAX_FLOOR`,
       restarting the ε ladder each time; steps at or above ``theta0`` are
       skipped. A pinned ``angle_max_deg`` is the caller's aperture and is
       never narrowed.

    Returns ``(result, eps_used, theta_used)`` — the optimizer payload
    plus the inputs that produced it, so the caller can warn when they
    differ from what was asked for. Raises :class:`ConfigurationError`
    when even the loosest combination is infeasible.

    ``dz_floor`` is the wrapper's own depth-step floor. It is NOT the
    ladder's low end — it is named in the refusal so an infeasible search
    says which limit made it infeasible. The distinction matters: the
    ladder starts at :func:`~uacpy.models.pe_grid.dz_ladder_end` (0.01 m,
    finer above 750 Hz) while the binding limit on every
    ordinary run is the cost floor ``c_min/(15 f)`` (0.1233 m at 800 Hz,
    0.0493 at 2 kHz, 0.0197 at 5 kHz in 1480 m/s), so ``eps_used`` describes a Δz an
    order of magnitude finer than the one that will be marched. Flooring
    the LADDER at ``dz_floor`` was tried and measured to make the marched
    field worse — the extra relaxation it forces licenses a coarser Δx
    (800 Hz over 2 km moved further from a converged grid, and 2 kHz was
    refused outright where it had run). The caller therefore keeps this
    selection and reports the truth instead:
    :func:`compute_grid_lytaev` recomputes the error on the grid it
    actually returns and puts BOTH numbers in the relaxation warning.

    Every rung scores the same candidate grids: ``ε`` moves the
    acceptance threshold, not the per-step error τ(Δx, Δz) the optimiser
    compares against it. One ``tau_cache`` spans the whole relaxation so a
    candidate is scored once rather than once per rung — a full ladder
    retries the search up to 24 times. Its keys carry ``θ`` (through the
    spectrum bounds it sets), so the θ steps below share the cache without
    reading each other's numbers.
    """
    tau_cache = {}
    eps_used, theta_used, res, last_exc = eps0, theta0, None, None
    eps_last = eps0
    trials = ((theta0,) if knobs.angle_max_deg is not None
              else (theta0, 20.0, _THETA_MAX_FLOOR))
    for theta_trial in trials:
        if theta_trial > theta0:
            continue
        theta_used = theta_trial
        eps_used = eps0
        for _ in range(8):
            eps_last = eps_used
            try:
                res = optimize_grid(
                    frequency=float(frequency),
                    c_min=c_min, c_max=c_max,
                    x_max=float(max_range),
                    c0=c0_pe,
                    angle_max_deg=float(theta_used),  # degrees
                    eps=eps_used,
                    p=int(knobs.n_pade),
                    alpha=0.0,
                    tau_cache=tau_cache,
                    c_min_all=c_min_all, c_max_all=c_max_all,
                )
                break
            except GridInfeasibleError as exc:
                last_exc = exc
                eps_used *= 3.0
                if eps_used >= EPS_LADDER_CEILING:
                    break
        if res is not None:
            break
    if res is None:
        if dz_floor > dz_ladder_end(frequency, c_min):
            # The floor bites above the ladder's own end, so the search
            # was already scoring grids finer than anything marchable.
            floor_note = (
                f"; the marched grid is floored at dz >= {dz_floor:.4g} m "
                f"(c_min/{LAMBDA_PER_DZ_FLOOR:.0f}f), above the "
                f"optimiser's dz={dz_ladder_end(frequency, c_min):g} m ladder end, so "
                f"a finer search would not have helped"
            )
        else:
            floor_note = (
                f"; the search bottomed out at the optimiser's own "
                f"dz={dz_ladder_end(frequency, c_min):g} m ladder end, which is "
                f"already finer than this grid's "
                f"dz>={dz_floor:.4g} m floor"
            )
        c_top = c_max if c_max_all is None else max(c_max_all, c_max)
        raise ConfigurationError(
            f"RAM:{kind}: no Lytaev grid feasible even at "
            f"ε={eps_last:.3g} (the last rung tried below "
            f"{EPS_LADDER_CEILING:g}), "
            f"θ_max={theta_used:.0f}° for f={frequency:.1f} Hz, "
            f"x_max={max_range:.0f} m{floor_note}. The accuracy band is "
            f"the water column [{c_min:.0f}, {c_max:.0f}] m/s out to the "
            f"wider of the aperture and the critical angle of the "
            f"fastest medium ({c_top:.0f} m/s), and no (dr, dz) on the "
            f"search ladders holds the steepest scored component "
            f"within ε over this range. Optimiser said: {last_exc}",
            remediation=(
                "Pin dr and dz and converge them yourself — a pinned "
                "grid is unscored on every backend beyond the "
                "trapped-mode check, and no accuracy warning follows. "
                "Two steps, dz first: halve dz until the field "
                "stops moving, keeping the seafloor a quarter cell "
                "below a node on the fluid backends (h/dz = n + 1/4; "
                "a round dz puts it on a node), then halve dr the "
                "same way. Check against Kraken or OASES; leave c0 at "
                "its default. Or pin accuracy= at the error bound you "
                "accept: a pinned accuracy is met as given, on the "
                "aperture you name with angle_max_deg."),
        ) from last_exc
    return res, eps_used, theta_used


def snap_dz_to_seafloor(h: float, n_layers: int) -> float:
    """``h / n_layers``, lowered onto a 12-significant-digit decimal so
    the value the input deck carries still satisfies ``h/dz >= n_layers``
    after the binary reads it back — the ON-node placement, which only
    rams0.5 uses on the automatic grid (:func:`dz_for_water_layers`;
    the fluid backends place the seafloor ``SEAFLOOR_CELL_OFFSET`` of a
    cell below the node, where no round-off can move the truncation).

    Every backend places the seafloor by truncating ``zb/dz``, with a
    cliff exactly at integer ``zb/dz``: ``ramgeo1.5.f:133``,
    ``ramsurf1.5.f:118`` and ``mpiramS/src/ram.f90:101`` take
    ``iz = int(1 + zb/dz)`` clamped into ``[2, nz]``, while
    ``rams0.5.f:135`` takes the bare ``iz = int(zb/dz)`` — one node
    lower and unclamped (which is why
    :func:`_domain.warn_if_seafloor_outside_grid` singles it out). The
    spelling differs; the alignment requirement does not.
    ``dz = h/n`` alone does not survive the trip
    through the deck: the Collins deck carries 12 significant digits
    (:func:`~uacpy.io.ramsurf_writer.write_ram_in`) and even a full-``repr``
    value can parse to a double a fraction of an ulp *above* the exact
    rational ``h/n`` (e.g. ``double(0.4) > 2/5``), which puts ``h/dz``
    just under ``n`` and the seafloor a whole cell up. Flooring onto the
    12-digit grid leaves ``dz`` below ``h/n`` by ~1e-12 relative — far
    above read-back noise, far below any physical scale — and the loop
    re-checks every spelling pair the decks can carry: the *depth* rides
    through the deck at the same precision as ``dz`` (``%.12g`` in
    :func:`~uacpy.io.ramsurf_writer.write_ram_in`, full ``repr`` in
    ``write_mpirams_inpe``), so the binaries' ``iz = int(1 + zb/dz)`` divides the
    read-back ``h`` by the read-back ``dz`` — both quotients must clear
    ``n`` (all binaries are built with ``-fdefault-real-8``).
    """
    h = float(h)
    n = max(1, int(n_layers))
    h_spellings = (h, float(f"{h:.12g}"))
    dz = h / n
    # One unit in the 12th significant digit of dz.
    quantum = 10.0 ** (int(np.floor(np.log10(dz))) - 11)
    dz = np.floor(dz / quantum) * quantum
    while any(hs / dzs < n
              for hs in h_spellings
              for dzs in (dz, float(f"{dz:.12g}"))):
        dz -= quantum
    return float(dz)


def dz_for_water_layers(h: float, n_layers: int, kind: str) -> float:
    """``dz`` that puts the seafloor ``SEAFLOOR_CELL_OFFSET`` of a cell
    below water node ``n_layers`` on a fluid backend, and on that node
    for rams0.5 (:func:`snap_dz_to_seafloor`, whose 12-digit guard is
    what an exact-node placement needs; a quarter-cell one sits far from
    the truncation cliff at integer ``h/dz`` and needs none).
    """
    n = max(1, int(n_layers))
    if kind == 'rams':
        return snap_dz_to_seafloor(h, n)
    return float(h) / (n + SEAFLOOR_CELL_OFFSET)


def seafloor_cell_offset(kind: str) -> float:
    """The fraction of a cell the seafloor sits below its last water
    node on ``kind`` (0 on rams0.5)."""
    return 0.0 if kind == 'rams' else SEAFLOOR_CELL_OFFSET


def seafloor_snap_depth(env: 'Environment') -> float:
    """The water depth the depth grid is aligned to: the shallowest
    bathymetry point. A range-dependent grid cannot be aligned at every
    range at once, and the shallowest column is the most demanding
    (fewest points in the water, so the largest relative interface
    displacement when the seafloor sits off-node). Every Environment
    carries a Bathymetry with at least one positive depth, so this is
    always > 0.
    """
    return float(np.min(env.bathymetry.depths))


def align_dz_with_seafloor(env: 'Environment', dz: float, *,
                            kind: str, coarsen: bool = False, knobs) -> float:
    """Move ``dz`` onto the nearest value that places the seafloor where
    ``kind`` wants it in its cell (:data:`SEAFLOOR_CELL_OFFSET`) without
    crossing the bound the caller is enforcing: the default returns a
    ``dz`` at or below the one passed in, ``coarsen=True`` one at or
    above it.

    Every site that overrides an already-placed ``dz`` — a cap, a floor,
    an array-bound coarsening — has to come back through here. A raw
    value leaves the seafloor anywhere in its cell, where each backend's
    ``iz`` truncation (``ramgeo1.5.f:133``, ``ramsurf1.5.f:118``,
    ``rams0.5.f:135``, ``mpiramS/src/ram.f90:101``) puts the interface
    up to a whole cell off.
    """
    h = seafloor_snap_depth(env)
    if not float(dz) > 0.0:
        return float(dz)
    ratio = h / float(dz) - seafloor_cell_offset(kind)
    # The 1e-9 keeps a ratio that is integral up to round-off from buying
    # a whole extra (or one fewer) layer.
    n = (int(np.floor(ratio + 1e-9)) if coarsen
         else int(np.ceil(ratio - 1e-9)))
    n = max(1, n)
    if kind == 'ramsurf':
        n = surface_aligned_layers(env, h, n, coarsen=coarsen, knobs=knobs)
    return dz_for_water_layers(h, n, kind)


def ramsurf_origin_depth(env: 'Environment', *, knobs) -> float:
    """Depth (m, deck frame) of ramsurf1.5's pressure-release surface at
    ``r = 0``: the altimetry's depression there, 0 without one."""
    if env.altimetry is None:
        return 0.0
    return float(ramsurf_surface_nodes(env, knobs=knobs)[0][1])


def surface_node_misfit(zsrf: float, dz: float) -> float:
    """How far (a fraction of a cell) ramsurf1.5 puts a surface asked at
    ``zsrf`` above it: it zeroes rows down to ``int(1 + zsrf/dz)``
    (``ramsurf1.5.f:115``, ``:281-290``), so the surface sits at
    ``floor(zsrf/dz)·dz``, and the misfit is ``frac(zsrf/dz)``. A ratio
    within 1e-6 under an integer counts as a whole cell: the deck's
    twelve digits can carry it either side of the integer."""
    x = float(zsrf) / float(dz)
    frac = x - np.floor(x)
    return 1.0 if frac > 1.0 - 1e-6 else float(frac)


def surface_aligned_layers(env: 'Environment', h: float, n: int, *,
                            coarsen: bool = False, knobs) -> int:
    """The seafloor layer count nearest ``n`` — more layers (a finer
    ``dz``) by default, fewer with ``coarsen`` — that also puts ramsurf's
    surface at ``r = 0`` within :data:`SURFACE_NODE_TOLERANCE` of a cell
    below a depth node, searched while ``dz`` moves by at most
    :data:`SURFACE_ALIGN_WINDOW`; the count with the smallest misfit
    when none in the window reaches the tolerance, and ``n`` itself for a
    surface at ``z = 0``. A range-dependent surface can be aligned at one
    range only; ``r = 0`` is where the source is planted."""
    zsrf = ramsurf_origin_depth(env, knobs=knobs)
    if not zsrf > 0.0:
        return n
    step = -1 if coarsen else 1
    best, best_misfit, k = n, 2.0, n
    while k >= 1 and (max(k, n) + 0.25) <= (1.0 + SURFACE_ALIGN_WINDOW) * (
            min(k, n) + 0.25):
        misfit = surface_node_misfit(
            zsrf, dz_for_water_layers(h, k, 'ramsurf'))
        if misfit <= SURFACE_NODE_TOLERANCE:
            return k
        if misfit < best_misfit:
            best, best_misfit = k, misfit
        k += step
    return best


def depth_floors(env: 'Environment', freq: float, kind: str,
                 c_min_all: float) -> 'tuple[float, float, float]':
    """``(dz_floor, cs_min, dz_shear_cap)`` for :func:`compute_grid_lytaev`:
    the depth-grid cost floor, the slowest shear speed (rams) and the
    ``λ_s/14`` cap it sets, which also bounds the floor."""
    # Depth-grid cost floor λ_p/15 (``LAMBDA_PER_DZ_FLOOR``) on every
    # backend, so the optimizer cannot demand an absurdly fine depth
    # grid. Override via ``dr=…``/``dz=…``.
    dz_floor = c_min_all / (LAMBDA_PER_DZ_FLOOR * max(freq, 1.0))
    cs_min = min_shear_speed(env) if kind == 'rams' else 0.0
    # The shear wavelength is the binding physical scale for the elastic
    # march, and resolving it is a correctness requirement rather than a
    # cost preference — so it caps dz, and it also bounds how far the cost
    # floor above may coarsen it.
    dz_shear_cap = rams_dz_shear_cap(cs_min, freq) if kind == 'rams' else 0.0
    if dz_shear_cap > 0:
        dz_floor = min(dz_floor, dz_shear_cap)
    return dz_floor, cs_min, dz_shear_cap


def tighten_rams_dr(env: 'Environment', freq: float, dr_opt: float,
                    kind: str, *, c0_pe, c_min, c_min_all, c_max_all,
                    max_range, notices=None, knobs, log) -> float:
    """Constraint of :func:`compute_grid_lytaev` on rams: the optimiser's
    ``dr`` tightened for the rotated Crank-Nicolson step rams0.5
    marches; ``dr_opt`` for every other backend.

    When the rotation itself amplifies faster than the seabed leaks, no
    step holds the march: uacpy's default angle (no stable angle on the
    ladder, :func:`_stability.resolve_rams_rotation_angle`) is refused,
    naming what to change, and a pinned ``rams_rotation_angle_deg`` is marched
    as given with a notice carrying the same remedy."""
    # With ``rams_rotation=True`` rams0.5 does not march the split-step Padé
    # exponential the optimiser scores: ``rpade`` (``rams0.5.f:859-892``,
    # "The Crank-Nicolson coefficients") builds ``pd1 = rot2 +
    # 0.5i·k0·dr·rot1`` / ``pd2 = rot2 − 0.5i·k0·dr·rot1`` and ``solve``
    # (``:780-850``) applies ``Π(1+pd1·X)/(1+pd2·X)`` — a Crank-Nicolson
    # step in range of the rotated rational-linear square root
    # (Milinazzo, Zala & Brooke 1997), O(dr²) in truncation with real
    # amplification ``|G| > 1`` on the propagating band (1.028 at the
    # Lytaev dr of 2.46 λ on a 1500/1800 case, where the march diverges;
    # 0.41 / 2.45 / 14.1 dB rms against krakenc at 0.2 / 0.6 / 1.8 λ).
    # So the Lytaev ``dr`` is not a valid step for this operator and two
    # constraints replace it, the tighter one winning:
    #   1. ``rams_dr_factor`` divides Lytaev's ``dr``;
    #   2. a wavelength cap ``dr ≤ c_min / (5·f)`` ≈ 0.2 λ per step —
    #      the truncation requirement of the O(dr²) step (τ·n ≈ 0.1
    #      there against 14.6 at the Lytaev dr).
    #   3. the stability rule of ``pe_grid.rams_growth_margin``:
    #      the Crank-Nicolson step's amplification of the steep
    #      propagating components just above cutoff must stay under the
    #      rate at which the seabed leaks them. The λ cap is a fixed
    #      fraction of a wavelength, and at fixed k0·dr the growth per
    #      metre scales with k0, so above a few kHz — or in deep water,
    #      where the leak per metre falls as 1/h — the λ cap alone lets
    #      the march diverge (5 kHz over 100 m of sand: 0.050 m stable
    #      against a 0.060 m λ cap; 1 kHz over 1000 m: 0.14 m against
    #      0.30 m). When the rotation's own growth already exceeds the
    #      leak no step helps (see the docstring for what follows).
    if kind == 'rams':
        dr_pre = dr_opt
        dr_safety = dr_opt / (DEFAULT_RAMS_DR_FACTOR
                              if knobs.rams_dr_factor is None
                              else float(knobs.rams_dr_factor))
        dr_cap = c_min_all / (RAMS_DR_LAMBDA_CAP * freq)
        dr_opt = min(dr_safety, dr_cap)
        limit = 'safety factor' if dr_safety <= dr_cap else 'λ cap'
        theta = resolve_rams_rotation_angle(env, freq, knobs=knobs)
        stab = rams_stability(
            env, freq, dr_max=dr_opt, theta=theta, knobs=knobs)
        dr_stab = None if stab is None else stab['dr']
        if stab is not None and dr_stab is None:
            if knobs.rams_rotation_angle_deg is None:
                raise ConfigurationError(
                    f"RAM:rams at f={freq:.1f} Hz: "
                    + rams_rotation_remedy(stab, knobs=knobs),
                    remediation="rams_rotation_angle_deg is a constructor knob (a "
                                "float, or a callable f -> degrees across a "
                                "band); see docs/models/ram.md §6 "
                                "constraint 3.",
                )
            give_notice(notices,
                f"RAM:rams at f={freq:.1f} Hz: "
                + rams_rotation_remedy(stab, knobs=knobs)
                + " The pinned angle is marched as given.",
                NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
        if dr_stab is not None and dr_stab < dr_opt:
            dr_opt = dr_stab
            limit = 'stability rule'
        dr_drift = rams_dr_for_level_drift(
            dr_opt, freq=freq, c0=c0_pe, c_min=c_min,
            c_max_all=c_max_all, max_range=max_range,
            theta=theta, knobs=knobs)
        if dr_drift < dr_opt:
            dr_opt = dr_drift
            limit = (f'level-drift budget of '
                     f'{RAMS_CN_DRIFT_BUDGET_DB:g} dB over the march')
        log(
            f"rams: tightened dr from {dr_pre:.2f} m to "
            f"{dr_opt:.4g} m (safety={dr_safety:.4g}, "
            f"λ-cap={dr_cap:.4g}, stability={dr_stab if dr_stab is None else round(dr_stab, 6)}; "
            f"{limit} active)."
        )
    return dr_opt


def cap_dr_at_the_collins_output_stride(env: 'Environment', freq: float,
                                        dr_opt: float, kind: str, *,
                                        log) -> float:
    """Constraint of :func:`compute_grid_lytaev` on the Collins codes:
    ``dr`` capped at the output spacing that samples the modal beat;
    ``dr_opt`` for mpiramS."""
    # The Collins binaries write the field only every ndr·dr and the
    # receiver modulus is interpolated between writes, a resolution the
    # Lytaev score does not see: an automatic dr is held to the output
    # spacing that also bounds ndr (mpiramS marches onto every receiver
    # range; a pinned dr is the caller's).
    if kind in ('ramgeo', 'rams', 'ramsurf'):
        beat_k = modal_beat_wavenumber(env, freq)
        cap = collins_output_spacing(beat_k)
        if dr_opt > cap:
            log(
                f"RAM:{kind}: capped dr from {dr_opt:.2f} m to {cap:.2f} m "
                f"so the output stride dr·ndr, which the receiver modulus "
                f"is interpolated across, samples the modal beat "
                f"({2 * np.pi / beat_k:.1f} m) {COLLINS_SAMPLES_PER_BEAT:.0f} "
                f"times per period (ndr is bounded by the same spacing)."
            )
            dr_opt = cap
    return dr_opt


def place_the_seafloor_in_its_cell(h: float, offset: float,
                                   dz_opt: float, kind: str) -> float:
    """Constraint of :func:`compute_grid_lytaev`: the ``dz`` nearest
    ``dz_opt`` that puts the seafloor ``h`` at ``offset`` of a cell
    below its last water node (:func:`seafloor_cell_offset`)."""
    n_layers = max(1, int(round(h / dz_opt - offset)))
    return dz_for_water_layers(h, n_layers, kind)


def cap_the_depth_points(h: float, dz_opt: float, dz_floor: float,
                        kind: str, *, warn_dz: bool, notices, knobs,
                        log) -> float:
    """Constraint of :func:`compute_grid_lytaev`: ``dz`` raised to keep
    the depth grid under ``MAX_DEPTH_POINTS``."""
    # Practical depth-grid cap (``MAX_DEPTH_POINTS``) — Lytaev's
    # optimizer at very low freq / deep ocean / wide θ_max can
    # demand dz ≈ λ/300 (5 cm at 25 Hz) → 100k+ depth points and
    # very slow per-step compute. Raise via dr/dz override for
    # accuracy-sensitive runs. The cap is measured against the coarser
    # of the optimiser's dz and the floor applied below: where the
    # floor alone brings the count under the cap the cap does not
    # bind, and a warning naming a dz that never runs would be noise.
    if h > 0 and h / max(dz_opt, dz_floor) > MAX_DEPTH_POINTS:
        dz_pre = dz_opt
        n_layers = MAX_DEPTH_POINTS
        dz_opt = dz_for_water_layers(h, n_layers, kind)
        cap_msg = (
            f"RAM:{kind}: raised dz from {dz_pre:.4f} m to {dz_opt:.3f} m "
            f"to keep the depth grid under {MAX_DEPTH_POINTS} points "
            f"(seafloor depth {h:.0f} m). The default Lytaev accuracy "
            f"budget ε={knobs.accuracy:.0e} is no longer met. Pin "
            f"accuracy= to march the grid that meets it, or set dr/dz "
            f"explicitly."
        )
        if warn_dz:
            give_notice(notices, cap_msg, NumericsWarning,
                        skip_file_prefixes=USER_FRAME_SKIP)
        else:
            log(cap_msg, level="info")
    return dz_opt


def floor_dz(env: 'Environment', kind: str, dr_opt: float,
             dz_opt: float, h: float, offset: float, dz_floor: float,
             score_kw: dict, *, knobs, log) -> 'tuple[float, str]':
    """Constraint of :func:`compute_grid_lytaev`: ``(dz, note)``, ``dz``
    raised to the cost floor and refined from there for the band's
    steepest component, ``note`` saying how when it was refined."""
    refined_note = ''
    if dz_floor > 0 and dz_opt < dz_floor:
        if h > 0:
            n_layers = max(1, int(np.floor(h / dz_floor - offset)))
            dz_opt = dz_for_water_layers(h, n_layers, kind)
        else:
            dz_opt = dz_floor
        # The floor is where the depth search starts, not where it
        # stops: a band whose steepest component scores at or above
        # ``TRAPPED_MODE_SCORE_LIMIT`` on the floored grid gets dz
        # refined until it passes, within the depth budget. Sand, silt
        # and a sloping sand wedge already pass on the floored grid and
        # keep it; rock and the 5500 m/s basement do not (measured in
        # :func:`refine_dz_for_the_steepest_component`).
        refined = refine_dz_for_the_steepest_component(
            env, kind, dr_opt, dz_opt, h, score_kw, knobs=knobs, log=log)
        if refined is not None:
            dz_opt, refined_note = refined
    return dz_opt, refined_note


def resolve_the_shear_wavelength(env: 'Environment', dz_opt: float,
                                 h: float, kind: str,
                                 dz_shear_cap: float, cs_min: float, *,
                                 warn_dz: bool, notices, knobs,
                                 log) -> float:
    """Constraint of :func:`compute_grid_lytaev` on rams: ``dz`` held at
    or under ``λ_s/14``, seafloor-aligned."""
    # Resolving the shear wavelength outranks every coarsening above: a
    # rams0.5 march on a grid coarser than λ_s/14 does not merely lose
    # accuracy, it diverges (measured 134 dB against OASES on Collins
    # 1991's own example D at 0.55 λ_s, versus 0.83 dB at λ_s/14).
    if dz_shear_cap > 0 and dz_opt > dz_shear_cap:
        dz_pre_cap = dz_opt
        # At or below the cap, and still placing the seafloor: the cap
        # is a bound on dz, not a grid, and h/λ_s is not an integer for
        # any ordinary seabed, so assigning it raw undoes the placement
        # above for essentially every auto-grid elastic run.
        dz_opt = align_dz_with_seafloor(env, dz_shear_cap, kind=kind,
                                        knobs=knobs)
        log(
            f"rams: tightened dz from {dz_pre_cap:.3f} m to "
            f"{dz_opt:.3f} m to resolve the shear wavelength "
            f"(λ_s/14 = {dz_shear_cap:.3f} m, c_s = {cs_min:.0f} m/s)."
        )
        # This cap outranks the MAX_DEPTH_POINTS budget applied above, since
        # a coarser grid diverges rather than merely costing accuracy. Say so
        # when it bites, so a slow run has a stated cause.
        if h > 0 and h / dz_opt > MAX_DEPTH_POINTS:
            shear_msg = (
                f"RAM:{kind}: resolving the shear wavelength needs dz="
                f"{dz_opt:.4f} m, i.e. {h / dz_opt:.0f} depth points — past "
                f"the {MAX_DEPTH_POINTS}-point runtime budget. A coarser "
                f"grid makes the elastic march diverge, so accuracy wins "
                f"here; expect a slow run."
            )
            if warn_dz:
                give_notice(notices, shear_msg, NumericsWarning,
                            skip_file_prefixes=USER_FRAME_SKIP)
            else:
                log(shear_msg, level="info")
    return dz_opt


def keep_the_source_below_row_one(env: 'Environment', dz_opt: float,
                                  zs: Optional[float], kind: str, *,
                                  knobs, log) -> float:
    """Constraint of :func:`compute_grid_lytaev`: ``dz`` no deeper than
    the source ``zs``, so it lands on a solved row."""
    if zs is not None and 0.0 < float(zs) < dz_opt:
        dz_pre_src = dz_opt
        dz_opt = align_dz_with_seafloor(env, float(zs), kind=kind, knobs=knobs)
        if dz_opt > float(zs):
            # The alignment's 1e-9 ratio slop rounded down one layer.
            dz_opt = align_dz_with_seafloor(env, dz_opt * (1 - 1e-8),
                                            kind=kind, knobs=knobs)
        log(
            f"RAM:{kind}: dz capped from {dz_pre_src:.4g} m to "
            f"{dz_opt:.4g} m so the {float(zs):.4g} m source lands at "
            f"depth index 2 or deeper (row 1 is never solved)."
        )
    return dz_opt


def put_the_ramsurf_surface_on_a_node(env: 'Environment',
                                      dz_opt: float, kind: str, *,
                                      knobs, log) -> float:
    """Constraint of :func:`compute_grid_lytaev` on ramsurf: ``dz``
    refined until the surface at ``r = 0`` sits on a depth node."""
    if kind == 'ramsurf':
        # ramsurf1.5 puts its pressure-release surface on the last row
        # it zeroes, ``floor(zsrf/dz)·dz``, up to a cell above the one
        # asked for: the seafloor-aligned dz is refined until the
        # surface at r = 0 sits on a node too
        # (:func:`surface_aligned_layers`).
        dz_pre_surface = dz_opt
        dz_opt = align_dz_with_seafloor(env, dz_opt, kind=kind, knobs=knobs)
        if dz_opt != dz_pre_surface:
            log(
                f"RAM:ramsurf: dz lowered from {dz_pre_surface:.4g} m "
                f"to {dz_opt:.4g} m so the surface at r=0 "
                f"({ramsurf_origin_depth(env, knobs=knobs):.4g} m) sits on a "
                f"depth node.")
    return dz_opt


def grid_for_pinned_accuracy(
    env: 'Environment', freq: float,
    *, max_range: float, kind: str, warn_dz: bool = True,
    zs: Optional[float] = None, notices=None, knobs, log
) -> 'tuple[float, float]':
    """``(dr, dz)`` of :func:`compute_grid_lytaev` when ``accuracy`` is
    pinned: the coarsest grid on the Lytaev (2023) ladders whose score
    stays under that ``accuracy`` (``pe_grid.optimize_grid`` at exactly
    ``ε``, on the aperture :func:`_domain.resolve_angle_max` gives), a
    pinned ``dr`` or ``dz`` held while the other is searched.

    No relaxation, cost floor or depth-point cap applies: the free step
    only takes the constraints that refine it — on ``dr`` the rams
    tightening (:func:`tighten_rams_dr`) and the Collins output stride
    (:func:`cap_dr_at_the_collins_output_stride`); on ``dz`` the seafloor
    placement rounded to the finer side (:func:`align_dz_with_seafloor`,
    which also puts ramsurf's surface on a node), the ``λ_s/14`` shear cap
    and the source row — and the score falls as either step is refined. A
    grid past :data:`MAX_DEPTH_POINTS` is marched, with a notice of its
    cost. ``ConfigurationError`` when no candidate meets the ``accuracy``.
    """
    c0_pe, c_min, c_max, c_min_all, c_max_all = band_speeds(
        env, knobs=knobs)
    theta = resolve_angle_max(env, knobs=knobs)
    score_kw = dict(
        frequency=float(freq), c_min=c_min, c_max=c_max,
        x_max=float(max_range), c0=c0_pe, angle_max_deg=float(theta),
        p=int(knobs.n_pade), alpha=0.0,
        c_min_all=c_min_all, c_max_all=c_max_all,
    )
    held = tuple(None if v is None else float(v)
                 for v in (knobs.dr, knobs.dz))
    held_text = ''.join(f", {name}={v:g} m held" for name, v
                        in zip(('dr', 'dz'), held) if v is not None)
    try:
        res = optimize_grid(eps=float(knobs.accuracy), grid=held,
                            **score_kw)
    except GridInfeasibleError as exc:
        if held_text:
            pinned = ' and '.join(f"{name}={v:g} m" for name, v
                                  in zip(('dr', 'dz'), held) if v is not None)
            raise ConfigurationError(
                f"RAM:{kind}: the pinned accuracy={knobs.accuracy:g} and the "
                f"pinned {pinned} contradict each other: no "
                f"{'dz' if held[0] is not None else 'dr'} on the Lytaev "
                f"search ladder meets that accuracy at f={freq:.1f} Hz over "
                f"{max_range:.0f} m (θ_max={theta:g}°).",
                remediation=(
                    f"Unpin {pinned.split('=')[0]} to have it chosen for "
                    f"the accuracy, pass a looser accuracy=, or unpin "
                    f"accuracy to march the pinned step as given."),
            ) from exc
        raise ConfigurationError(
            f"RAM:{kind}: no (dr, dz) on the Lytaev search ladders meets "
            f"the pinned accuracy={knobs.accuracy:g} at f={freq:.1f} Hz over "
            f"{max_range:.0f} m (θ_max={theta:g}°).",
            remediation=(
                "Pass a looser accuracy=, leave accuracy unset to have the "
                "grid chosen (its relaxation is reported), or pin dr and "
                "dz and converge them yourself."),
        ) from exc
    dr, dz = float(res.dr), float(res.dz)
    if knobs.dr is None:
        dr = tighten_rams_dr(
            env, freq, dr, kind, c0_pe=c0_pe, c_min=c_min,
            c_min_all=c_min_all, c_max_all=c_max_all,
            max_range=max_range, notices=notices, knobs=knobs, log=log)
        dr = cap_dr_at_the_collins_output_stride(
            env, freq, dr, kind, log=log)
    if knobs.dz is None:
        h = seafloor_snap_depth(env)
        dz = align_dz_with_seafloor(env, dz, kind=kind, knobs=knobs)
        _, cs_min, dz_shear_cap = depth_floors(env, freq, kind, c_min_all)
        dz = resolve_the_shear_wavelength(
            env, dz, h, kind, dz_shear_cap, cs_min, warn_dz=warn_dz,
            notices=notices, knobs=knobs, log=log)
        dz = keep_the_source_below_row_one(env, dz, zs, kind, knobs=knobs,
                                           log=log)
        if warn_dz and h / dz > MAX_DEPTH_POINTS:
            give_notice(notices,
                f"RAM:{kind}: the grid meeting accuracy={knobs.accuracy:g} "
                f"at f={freq:.1f} Hz has dz={dz:.4g} m, {h / dz:.0f} depth "
                f"points over the {h:.0f} m water column — past the "
                f"{MAX_DEPTH_POINTS} at which the automatic grid stops; "
                f"expect a slow run.",
                NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)
    err = optimize_grid(grid=(dr, dz), **score_kw).predicted_error
    log(
        f"{kind}: Lytaev grid for the pinned accuracy={knobs.accuracy:g} → "
        f"dr={dr:.4g} m, dz={dz:.4g} m (predicted error {err:.2e}, "
        f"θ_max={theta:g}°{held_text})."
    )
    return dr, dz


def compute_grid_lytaev(
    env: 'Environment', freq: float,
    *, max_range: float, kind: str, warn_dz: bool = True,
    zs: Optional[float] = None, notices=None, knobs, log
) -> 'tuple[float, float]':
    """Padé-error-based ``(dr, dz)`` selection following Lytaev
    (2023, https://doi.org/10.3390/jmse11030496).

    Picks the coarsest ``(dr, dz)`` whose accumulated single-step
    Padé error stays under ``accuracy`` over the marched range.
    The PE reference speed ``c₀`` comes from ``_domain.resolve_c0`` (Lytaev
    Eq. (15) by default, the user's value when pinned).

    The optimizer minimises Lytaev's error model alone. The
    constraints it does not represent are applied to its output
    afterwards: the rams ``dr`` stability tightening, the Collins
    output-stride cap on ``dr`` (``COLLINS_SAMPLES_PER_BEAT``), the
    seafloor's placement in its cell (``SEAFLOOR_CELL_OFFSET``), the
    ``MAX_DEPTH_POINTS`` runtime cap, and
    the shear/acoustic ``dz`` floor — where the depth search starts;
    a seabed whose trapped modes the floored grid cannot carry gets
    ``dz`` refined from there (:func:`refine_dz_for_the_steepest_component`).
    The accuracy that gets logged is therefore recomputed on the grid
    that is actually marched, which can be orders of magnitude above
    ``accuracy`` once a floor has bound.

    The constraints apply in this order, one function each:
    :func:`tighten_rams_dr`, :func:`cap_dr_at_the_collins_output_stride`,
    :func:`place_the_seafloor_in_its_cell`, :func:`cap_the_depth_points`,
    :func:`floor_dz`, :func:`resolve_the_shear_wavelength`,
    :func:`keep_the_source_below_row_one` and
    :func:`put_the_ramsurf_surface_on_a_node`.

    ``warn_dz=False`` demotes the depth-step warnings (the
    ``MAX_DEPTH_POINTS`` cap and the shear-wavelength budget) to log
    lines. Broadband
    callers size ``dr`` and ``dz`` at different band edges, so the call
    that keeps only ``dr`` would otherwise warn about a ``dz`` it throws
    away.

    ``zs`` (the source depth) caps ``dz`` from above: every binary plants
    the source at row ``1 + zs/dz`` and no solver writes row 1
    (:func:`_domain.check_source_row_is_solved`), so the cell has to be no
    deeper than the source. The cap outranks every coarsening above it
    and keeps the seafloor placed in its cell.

    Raises ``ConfigurationError`` if no candidate ``(dr, dz)`` pair
    meets the accuracy budget even after auto-loosening.

    All of the above is the default chooser, for an ``accuracy`` left
    unset. A pinned ``accuracy`` is met as given instead
    (:func:`grid_for_pinned_accuracy`).
    """
    if knobs.accuracy_pinned:
        return grid_for_pinned_accuracy(
            env, freq, max_range=max_range, kind=kind, warn_dz=warn_dz,
            zs=zs, notices=notices, knobs=knobs, log=log)
    c0_pe, c_min, c_max, c_min_all, c_max_all = band_speeds(
        env, knobs=knobs)

    dz_floor, cs_min, dz_shear_cap = depth_floors(
        env, freq, kind, c_min_all)

    eps0 = knobs.accuracy
    theta0 = resolve_angle_max(env, knobs=knobs)
    res, eps_used, theta_used = optimize_grid_relaxing(
        frequency=freq, c_min=c_min, c_max=c_max, max_range=max_range,
        c0_pe=c0_pe, eps0=eps0, theta0=theta0, kind=kind,
        dz_floor=dz_floor, c_min_all=c_min_all, c_max_all=c_max_all,
        knobs=knobs
    )
    # The relaxation warning is deferred to the end of this method, where
    # the error of the grid ACTUALLY marched is known. ``eps_used`` is only
    # the threshold at which the search stopped, and the Δz that met it is
    # routinely finer than the cost floor below will allow — at 800 Hz the
    # floor is 0.1156 m against the optimiser's own 0.01 m ladder end, so
    # the ε reported here belonged to a grid that never runs.
    relaxed = (eps_used > eps0 or theta_used < theta0)

    dr_opt, dz_opt = float(res.dr), float(res.dz)

    dr_opt = tighten_rams_dr(
        env, freq, dr_opt, kind, c0_pe=c0_pe, c_min=c_min,
        c_min_all=c_min_all, c_max_all=c_max_all,
        max_range=max_range, notices=notices, knobs=knobs, log=log)

    dr_opt = cap_dr_at_the_collins_output_stride(
        env, freq, dr_opt, kind, log=log)

    # Place the shallowest seafloor where this backend wants it in its
    # depth cell (``SEAFLOOR_CELL_OFFSET``): the interface's position to
    # a fraction of a cell sets the trapped modes on a fast seabed.
    h = seafloor_snap_depth(env)
    offset = seafloor_cell_offset(kind)
    dz_opt = place_the_seafloor_in_its_cell(h, offset, dz_opt, kind)

    dz_opt = cap_the_depth_points(
        h, dz_opt, dz_floor, kind, warn_dz=warn_dz, notices=notices,
        knobs=knobs, log=log)

    # The optimizer knows nothing about the adjustments above, so its
    # own ``predicted_error`` describes a grid that may never be
    # marched; every score below is recomputed on the grid in hand.
    score_kw = dict(
        frequency=float(freq), c_min=c_min, c_max=c_max,
        x_max=float(max_range), c0=c0_pe, angle_max_deg=float(theta_used),
        p=int(knobs.n_pade), alpha=0.0,
        c_min_all=c_min_all, c_max_all=c_max_all,
    )

    dz_pre_floor = dz_opt
    dz_opt, refined_note = floor_dz(
        env, kind, dr_opt, dz_opt, h, offset, dz_floor, score_kw,
        knobs=knobs, log=log)

    dz_opt = resolve_the_shear_wavelength(
        env, dz_opt, h, kind, dz_shear_cap, cs_min,
        warn_dz=warn_dz, notices=notices, knobs=knobs, log=log)

    dz_opt = keep_the_source_below_row_one(
        env, dz_opt, zs, kind, knobs=knobs, log=log)

    dz_opt = put_the_ramsurf_surface_on_a_node(
        env, dz_opt, kind, knobs=knobs, log=log)

    scores = optimize_grid(grid=(dr_opt, dz_opt), **score_kw)
    err, growth = scores.predicted_error, scores.growth
    theta_c = (np.degrees(np.arccos(c_min / c_max_all))   # steepest mode
               if scores.trapped_end_binds else 0.0)

    if dz_opt > dz_pre_floor:
        if cs_min > 0:
            reason = 'shear-wavelength resolution (λ_s / 14)'
        else:
            reason = (f'depth-grid cost floor '
                      f'(λ_p / {LAMBDA_PER_DZ_FLOOR:g})')
        msg = (
            f"RAM:{kind}: raised dz from {dz_pre_floor:.3f} m to "
            f"{dz_opt:.3f} m for {reason} "
            f"(floor={dz_floor:.3f} m at cs_min={cs_min:.0f} m/s, "
            f"f={freq:.0f} Hz{refined_note}). The default Lytaev "
            f"accuracy budget ε={knobs.accuracy:.0e} is not met on this "
            f"grid — its predicted error is {err:.2e}. Pin accuracy= to "
            f"march a grid that meets it, or set dr/dz explicitly. A "
            f"broadband sweep marches one grid for the whole band and "
            f"sizes dz at the *highest* frequency in it, so this floor "
            f"already covers every bin below f={freq:.0f} Hz."
        )
        # The stability floor sits above the default Lytaev dz for every
        # ordinary frequency: a status line on uacpy's own target, never
        # a warning that fires on essentially every run.
        log(msg, level="info")

    if relaxed:
        # Both numbers, in this order: the threshold the SEARCH accepted,
        # and the error of the grid this method returns. They differ
        # whenever a floor bound the answer, which is the ordinary case.
        if dz_floor <= dz_ladder_end(freq, c_min):
            floor_note = ""
        elif refined_note:
            floor_note = (
                f" The search ran down to the optimiser's "
                f"dz={dz_ladder_end(freq, c_min):g} m ladder end; this grid "
                f"starts at the dz={dz_floor:.4g} m floor "
                f"(c_min/{LAMBDA_PER_DZ_FLOOR:.0f}f) and is refined to "
                f"dz={dz_opt:.4g} m for the band's steepest component, so "
                f"ε={eps_used:.0e} describes a finer grid than the one "
                f"marched."
            )
        else:
            floor_note = (
                f" The search ran down to the optimiser's "
                f"dz={dz_ladder_end(freq, c_min):g} m ladder end, but this grid "
                f"is floored at dz={dz_floor:.4g} m "
                f"(c_min/{LAMBDA_PER_DZ_FLOOR:.0f}f), so "
                f"ε={eps_used:.0e} describes a finer grid than the one "
                f"marched."
            )
        # Only the input that moved is named: a "ε=1e-03→1e-03" or a
        # "θ_max=30°→30°" clause describes no relaxation.
        theta_narrowed = theta_used < theta0
        moved = []
        if eps_used > eps0:
            moved.append(f"ε={eps0:.0e}→{eps_used:.0e}")
        if theta_narrowed:
            moved.append(f"θ_max={theta0:.0f}°→{theta_used:.0f}°")
        # The score saturates at τ ≤ 2 per step, so a value at or above
        # 1 ranks nothing: it says the model cannot resolve this grid.
        err_note = (f"predicted error of {err:.2e}" if err < 1.0 else
                    f"predicted error of {err:.2e}, i.e. not resolved "
                    f"by the model")
        msg = (
            f"RAM:{kind}: Lytaev relaxed {', '.join(moved)} to find a "
            f"feasible grid at f={freq:.1f} Hz, "
            f"x_max={max_range:.0f} m. The grid returned "
            f"(dr={dr_opt:.3g} m, dz={dz_opt:.3g} m) has a {err_note}, "
            f"against the default target of ε={eps0:.0e}.{floor_note}"
        )
        # An ε-only relaxation is routine at and above ~500 Hz (the
        # second-order Δξ term forces it at the ladder's 0.01 m end over
        # a few km, and the marched dz is then the floor anyway), so it
        # follows the floor's own policy: a status line on uacpy's own
        # target. Narrowing the aperture changes the physics asked for
        # and always warns.
        # Past ``ROUTINE_EPS_RELAXATION`` the relaxation is no longer
        # routine: the returned grid's score bounds an error on the
        # steepest component that the caller should see.
        past_routine = eps_used > ROUTINE_EPS_RELAXATION
        if past_routine:
            edge = (f"critical angle {theta_c:.0f}°"
                    if scores.trapped_end_binds
                    else f"θ_max={theta_used:.0f}°")
            msg += (f" That bound is on the steepest scored component "
                    f"(the {edge} edge of the band).")
        if theta_narrowed or past_routine:
            give_notice(notices, msg, NumericsWarning,
                        skip_file_prefixes=USER_FRAME_SKIP)
        else:
            log(msg, level="info")

    c0_origin = 'user' if knobs.c0 is not None else 'Lytaev Eq.15'
    if kind == 'rams' and knobs.rams_rotation:
        # The score is of the split-step Padé exponential; rams marches
        # a Crank-Nicolson step of the rotated square root instead (see
        # the dr cap above), so the number says nothing about this run.
        score = (f"split-step Padé score {err:.2e}, not the error of the "
                 f"rotated Crank-Nicolson step rams marches — dr is held "
                 f"at ≤ λ/{RAMS_DR_LAMBDA_CAP:.0f} for that")
    else:
        score = f"predicted error {err:.2e}"
    band = (f"critical angle {theta_c:.0f}°" if scores.trapped_end_binds
            else f"θ_max={theta_used:.0f}°")
    log(
        f"{kind}: Lytaev grid → dr={dr_opt:.2f} m, "
        f"dz={dz_opt:.3f} m ({score} on the water band "
        f"ξ∈[{scores.xi_min:.3f}, {scores.xi_max:.3f}] out to "
        f"the {band}; stability-band growth {growth:.1e} per step on "
        f"ξ∈[{scores.xi_stab_min:.3f}, {scores.xi_min:.3f}); "
        f"c₀={c0_pe:.1f} m/s [{c0_origin}], ε={knobs.accuracy:.0e})."
    )
    return dr_opt, dz_opt

def dz_for_trapped_modes(dr, dz, score_kw):
    """The coarsest ``dz`` (to 1 %, three significant digits) at which
    the grid ``(dr, dz)`` scores under :data:`TRAPPED_MODE_SCORE_LIMIT`,
    and that score;
    ``None`` when ``dz`` already passes or when no ``dz`` down to
    ``dz / 2**20`` does (the range error at this ``dr`` is then what
    binds). The score falls as ``dz²`` (Lytaev's second-order Δξ term),
    so a geometric bisection between the failing ``dz`` and a passing
    one below it converges in a dozen 0.8 ms evaluations.
    """
    dr, hi = float(dr), float(dz)
    if grid_error(dr=dr, dz=hi, **score_kw) < TRAPPED_MODE_SCORE_LIMIT:
        return None
    lo = hi
    for _ in range(20):
        lo *= 0.5
        if grid_error(dr=dr, dz=lo, **score_kw) < TRAPPED_MODE_SCORE_LIMIT:
            break
    else:
        return None
    for _ in range(10):
        mid = float(np.sqrt(hi * lo))
        if grid_error(dr=dr, dz=mid, **score_kw) < TRAPPED_MODE_SCORE_LIMIT:
            lo = mid
        else:
            hi = mid
    # Rounded DOWN to three significant digits, so the value a log line
    # prints is the value that passes (a rounded-up print would not).
    quantum = 10.0 ** (int(np.floor(np.log10(lo))) - 2)
    lo = float(np.floor(lo / quantum) * quantum)
    return lo, grid_error(dr=dr, dz=lo, **score_kw)


def refine_dz_for_the_steepest_component(env, kind, dr, dz, h,
                                          score_kw, *, knobs, log):
    """``(dz, note)``: ``dz`` refined below the floored value so the
    band's steepest scored component — the seabed's critical-angle mode
    or the ``angle_max_deg`` aperture edge, whichever binds — scores under
    :data:`TRAPPED_MODE_SCORE_LIMIT` at this ``dr``, aligned with the
    seafloor and held to :data:`MAX_DEPTH_POINTS` over the ``h`` m water
    column, with a clause for the status line saying which; ``None`` when
    the floored grid already scores under the limit.

    Only ``dz`` moves: the search that chose ``dr`` already held the
    same band under its ε at the optimiser's own (finer) step, so a
    passing ``dz`` exists at this ``dr`` unless the budget forbids it —
    and then the budget's ``dz`` is returned and the marched-grid check
    (:func:`warn_if_trapped_modes_unresolved`) names both the need and
    the budget in its warning. Measured on the 100 m channel,
    9 × 19 receivers to 5 km, mpiramS, rms dB from Kraken (all columns),
    λ/16 floor → refined grid with the seafloor placed in its cell
    (``SEAFLOOR_CELL_OFFSET``): rock 100 Hz 2.8 → 0.8 (λ/45), rock 200 Hz
    6.3 → 1.4 (λ/64), hard rock 100 Hz 5.0 → 0.8 (λ/108), hard rock
    200 Hz 7.4 → 1.5 (λ/162); wall 0.3–0.7 s → 0.7–3.0 s. Sand, silt
    and a sloping sand wedge keep the floored grid. The aperture edge
    binds on a band relaxed at kHz: 2 kHz over 50 m of water on a
    1700 m/s seabed to 2 km scores 6.5 on the λ/16 floor and reads
    2.5 dB median |dTL| from Scooter there.
    """
    scores = optimize_grid(grid=(float(dr), float(dz)), **score_kw)
    if scores.predicted_error < TRAPPED_MODE_SCORE_LIMIT:
        return None
    need = dz_for_trapped_modes(dr, dz, score_kw)
    if need is None:
        return None
    dz_need, score_need = need
    c_min = score_kw['c_min']
    lam = c_min / max(float(score_kw['frequency']), 1.0)
    if h > 0 and h / dz_need > MAX_DEPTH_POINTS:
        dz_out = dz_for_water_layers(h, MAX_DEPTH_POINTS, kind)
        log(
            f"RAM:{kind}: the band's steepest component needs dz <= "
            f"{dz_need:.3g} m "
            f"(λ/{lam / dz_need:.0f}) at dr={dr:.3g} m, i.e. "
            f"{h / dz_need:.0f} depth points over the {h:.0f} m water "
            f"column — past the {MAX_DEPTH_POINTS}-point budget; dz stops "
            f"at the budget's {dz_out:.4g} m.", level="info")
        return dz_out, (f", then held at the {MAX_DEPTH_POINTS}-point depth "
                        f"budget short of the dz <= {dz_need:.3g} m the "
                        f"band's steepest component needs")
    dz_out = (align_dz_with_seafloor(env, dz_need, kind=kind, knobs=knobs)
              if h > 0 else dz_need)
    log(
        f"RAM:{kind}: refined dz from {dz:.4g} m (λ/{lam / dz:.0f}, the "
        f"cost floor) to {dz_out:.4g} m (λ/{lam / dz_out:.0f}) so the "
        f"band's steepest component scores {score_need:.2g} instead of "
        f"{scores.predicted_error:.2g} at dr={dr:.3g} m.", level="info")
    return dz_out, (f", then refined to λ/{lam / dz_out:.0f} so the band's "
                    f"steepest component scores under "
                    f"{TRAPPED_MODE_SCORE_LIMIT:g}")


def warn_if_trapped_modes_unresolved(env, freq, kind, dr, dz,
                                      max_range, *, notices=None, knobs, log):
    """Score the grid about to be marched — pinned or automatic — and
    warn when it cannot carry the seabed's trapped modes, naming the
    ``dz`` that would (:func:`dz_for_trapped_modes`) and, when that
    ``dz`` is past :data:`MAX_DEPTH_POINTS`, the budget that stopped
    the automatic refinement there.

    When the steepest scored component is the mode grazing the seabed at
    its critical angle (``trapped_end_binds``) it carries the far field,
    and a score at or above :data:`TRAPPED_MODE_SCORE_LIMIT` means its
    phase is lost over the march — 2.8 / 6.3 dB rms against Kraken on a
    2400 m/s rock at 100 / 200 Hz at λ/16, where sand is 1.0 dB. The
    automatic grid refines ``dz`` past that, so this fires on a pinned
    grid that cannot carry the modes, or on an automatic one the depth
    budget cut short; a grid that scores under the limit is silent.
    """
    c0, c_min, c_max, c_min_all, c_max_all = band_speeds(
        env, knobs=knobs)
    kw = dict(frequency=float(freq), c_min=c_min, c_max=c_max,
              x_max=float(max_range), c0=c0,
              angle_max_deg=resolve_angle_max(env, knobs=knobs),
              p=int(knobs.n_pade),
              alpha=0.0, c_min_all=c_min_all, c_max_all=c_max_all)
    scores = optimize_grid(grid=(float(dr), float(dz)), **kw)
    err = scores.predicted_error
    if not scores.trapped_end_binds or err < TRAPPED_MODE_SCORE_LIMIT:
        return
    need = dz_for_trapped_modes(dr, dz, kw)
    theta_c = np.degrees(np.arccos(c_min / c_max_all))
    lam = c_min / max(float(freq), 1.0)
    h = seafloor_snap_depth(env)
    if need is None:
        remedy = (f" No dz down to {float(dz) / 2 ** 20:.2g} m carries it "
                  f"at this dr: refine dr as well.")
    else:
        dz_need, score_need = need
        points = h / dz_need
        remedy = (f" Carrying it at this dr needs dz <= {dz_need:.3g} m "
                  f"(λ/{lam / dz_need:.0f}), {points:.0f} depth points "
                  f"over the {h:.0f} m water column")
        if points > MAX_DEPTH_POINTS and knobs.dz is None:
            remedy += (f" — past the {MAX_DEPTH_POINTS}-point budget "
                       f"(MAX_DEPTH_POINTS) at which the automatic dz "
                       f"stopped; pin dz to march it, and expect a slow "
                       f"run.")
        else:
            remedy += "; pin dz to march it."
    give_notice(notices,
        f"RAM:{kind}: the {c_max_all:.0f} m/s seabed traps modes out to "
        f"a {theta_c:.0f}° grazing angle, and on the grid being marched "
        f"(dr={dr:.3g} m, dz={dz:.3g} m = λ/{lam / dz:.0f}) the steepest "
        f"of them accumulates a depth-operator error of {err:.2g} over "
        f"{max_range:.0f} m — at or above {TRAPPED_MODE_SCORE_LIMIT:g}, "
        f"its phase is lost and the field at range is wrong, not merely a "
        f"decibel off.{remedy}",
        NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )
    if need is not None:
        log(
            f"RAM:{kind}: dz <= {dz_need:.3g} m (λ/{lam / dz_need:.0f}) "
            f"brings the steepest trapped mode's score to "
            f"{score_need:.2g} at dr={dr:.3g} m; pin dz to march it.",
            level="info")


def warn_if_receivers_see_paths_steeper_than_the_band(
        env: Environment, source, receiver, *, notices=None, knobs) -> None:
    """``NumericsWarning`` when a receiver's direct or surface-reflected
    path is steeper than the PE's angular band reaches, naming the first
    range clear of it.

    The band is the wider of the ``angle_max_deg`` aperture
    (:func:`~uacpy.models.ram._domain.resolve_angle_max`) and the critical
    angle of the fastest medium against the fastest water — the band the
    grid is scored on. A steeper component is damped or garbled at the
    marched ``dr`` rather than propagated, so the field there is wrong with
    no sign of it in the result: on a 100 m sand channel at 200 Hz
    (source 25 m, receiver 60 m) RAM read 18, 10 and 20 dB from
    ``Scooter(c_high=1e9)`` at 5, 20 and 50 m. The geometry is the one
    Kraken's and Scooter's ``c_high`` notice uses
    (:func:`~uacpy.models._window.steep_path_cut`)."""
    _, c_water = water_speed_bounds(env)
    c_fast = float(speed_bounds(env)[1])
    aperture = resolve_angle_max(env, knobs=knobs)
    origin = f"angle_max_deg={aperture:g}°"
    if c_fast > c_water:
        critical = float(np.degrees(np.arccos(c_water / c_fast)))
        if critical > aperture:
            aperture = critical
            origin = (f"the critical angle of the fastest medium, "
                      f"{c_fast:.0f} m/s")
    c_band = c_water / np.cos(np.radians(aperture))
    reach = steep_path_cut(c_water, c_band, source.depths, receiver.depths,
                           receiver.ranges)
    if reach is None:
        return
    cut, steepest = reach
    clear = steep_path_clear_range(source.depths, receiver.depths, cut)
    give_notice(
        notices,
        f"RAM: the PE's angular band reaches {cut:.1f}° ({origin}), but "
        f"the closest receiver sees a direct or surface-reflected path at "
        f"{steepest:.1f}°. A component steeper than the band is damped or "
        f"garbled at the marched range step rather than propagated, so the "
        f"near-range field can be wrong; from {clear} m on no direct or "
        f"surface-reflected "
        f"path is steeper than the band. For the closer receivers run "
        f"Scooter(c_high=1e9), which integrates from k = 0.",
        NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)


def warn_if_pinned_angle_max_is_below_the_slope(env: Environment, *,
                                                 notices=None,
                                                 knobs) -> None:
    """One notice when a pinned ``angle_max_deg`` is narrower than the
    steepest bathymetry slope: Lytaev (2023) §5.5 bounds the spectrum by
    ``θ_max = max(θ_max^src, θ_max^bottom)``, the bottom term being that
    slope, which the unset default takes (:func:`_domain.resolve_angle_max`);
    the pinned value is used as given."""
    if knobs.angle_max_deg is None:
        return
    slope = steepest_slope_deg(env)
    if slope <= float(knobs.angle_max_deg):
        return
    give_notice(notices,
        f"RAM: the pinned angle_max_deg={float(knobs.angle_max_deg):g}° is narrower "
        f"than the bathymetry's steepest slope, {slope:.3g}°. Lytaev (2023) "
        f"§5.5 bounds the Padé spectrum by θ_max = max(θ_src, θ_bottom), the "
        f"bottom term being that slope, which angle_max_deg=None takes; "
        f"{float(knobs.angle_max_deg):g}° is used as given.",
        NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)


def warn_if_pinned_rams_dr_factor_is_unused(backend: str, *, notices=None,
                                            knobs) -> None:
    """One notice when ``rams_dr_factor`` is pinned on a rams run whose
    ``dr`` is pinned too: it divides the automatic ``dr`` only
    (:func:`tighten_rams_dr`). Other backends' runs are named by
    ``RAM._warn_on_mpirams_only_overrides``."""
    if (backend == 'rams' and knobs.rams_dr_factor is not None
            and knobs.dr is not None):
        give_notice(notices,
            f"RAM:rams: rams_dr_factor={float(knobs.rams_dr_factor):g} "
            f"divides the automatic range step, and dr={float(knobs.dr):g} m "
            f"is pinned, so it is not used.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)


def warn_if_pinned_accuracy_is_unused(*, notices=None, knobs) -> None:
    """One notice when ``accuracy`` is pinned beside a pinned ``dr`` and
    ``dz``: it chooses the grid, and both steps are already the
    caller's."""
    if (knobs.accuracy_pinned and knobs.dr is not None
            and knobs.dz is not None):
        give_notice(notices,
            f"RAM: accuracy={knobs.accuracy:g} chooses the (dr, dz) grid, "
            f"and dr={float(knobs.dr):g} m and dz={float(knobs.dz):g} m are "
            f"both pinned, so it is not used.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)


def warn_if_pinned_dz_misplaces_the_seafloor(env: Environment,
                                              backend: str, *,
                                              notices=None, knobs) -> None:
    """Warn when a pinned ``dz`` puts the shallowest seafloor more than
    :data:`SEAFLOOR_QUARTER_TOLERANCE` of a cell from the quarter-cell
    placement :data:`SEAFLOOR_CELL_OFFSET` on a fluid backend, and name
    the aligned value; ``dz`` is marched as given.

    The fluid codes take nodes ``1..iz``, ``iz = int(1 + h/dz)``, as water
    and start the seabed at node ``iz + 1`` (``ramgeo1.5.f:133``, matrc
    ``:255-268``; ``ramsurf1.5.f:118``, ``:242-250``; mpiramS
    ``ram.f90:101``, ``matrc.f90:50``), so the march sees the same
    seafloor for every depth inside one cell and its Galerkin interface
    lies in the element between those nodes. The quarter-cell placement
    and the tolerance round it are uacpy's guidance, measured against
    Kraken (docs/models/ram.md). Round values — ``dz = 1, 0.5, 0.25`` m
    on a 100 m or 200 m channel — all land on a node, and so does every
    rung of a halving ladder from them. rams0.5 applies its interface
    conditions on the seafloor node itself and is not checked.
    """
    if knobs.dz is None or backend == 'rams':
        return
    h = seafloor_snap_depth(env)
    dz = float(knobs.dz)
    ratio = h / dz
    frac = ratio - np.floor(ratio)
    offset = abs(frac - SEAFLOOR_CELL_OFFSET)
    distance = min(offset, 1.0 - offset)
    if distance <= SEAFLOOR_QUARTER_TOLERANCE:
        return
    aligned = align_dz_with_seafloor(env, dz, kind=backend, knobs=knobs)
    last_water = np.floor(ratio) * dz
    give_notice(notices,
        f"RAM:{backend}: the pinned dz={dz:g} m puts the {h:g} m "
        f"seafloor {frac:.3g} of a cell below its last water node "
        f"(h/dz = {ratio:.6g}), {distance:.3g} of a cell from the "
        f"quarter cell the automatic grid uses. The fluid codes take the "
        f"nodes down to int(1 + h/dz) as water and start the seabed at "
        f"the next one (ramgeo1.5.f:133, :255-268), so the march sees "
        f"the same seafloor, between its nodes at {last_water:.6g} m and "
        f"{last_water + dz:.6g} m, for every depth in that cell. A "
        f"quarter cell below a node, within {SEAFLOOR_QUARTER_TOLERANCE:g} "
        f"of a cell, is uacpy's guidance (docs/models/ram.md). dz is "
        f"marched as given; dz={aligned:.6g} m puts the seafloor a "
        f"quarter cell below a node.",
        NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP,
    )


def dz_putting_the_surface_on_a_node(env: Environment, dz: float, *,
                                     knobs) -> 'tuple[float, Optional[float]]':
    """``(finer, coarser)``: the ``dz`` nearest ``dz`` at or below it, and
    the nearest above it (``None`` when there is none), that put
    ramsurf1.5's surface at ``r = 0`` within
    :data:`SURFACE_NODE_TOLERANCE` of a cell below a depth node
    (:func:`surface_node_misfit`), each rounded to the six digits the notice
    prints and checked again there.

    The seafloor-placed values ``h/(n + SEAFLOOR_CELL_OFFSET)`` are searched
    first, every layer count from one to four times the current one; on a
    side where none qualifies, the value putting the surface half the
    tolerance below the nearest node that side
    (``zsrf/(m + SURFACE_NODE_TOLERANCE/2)``, the seafloor where it falls).
    Either clears :func:`warn_if_ramsurf_surface_unresolved`'s condition by
    construction, so the remedy it names is a ``dz`` that silences it."""
    zsrf = ramsurf_origin_depth(env, knobs=knobs)
    h = seafloor_snap_depth(env)
    dz = float(dz)

    def clears(value):
        return surface_node_misfit(zsrf, value) <= SURFACE_NODE_TOLERANCE

    def printed(value):
        return float(f"{value:.6g}")

    n_now = max(1, int(round(h / dz - SEAFLOOR_CELL_OFFSET)))
    placed = [printed(dz_for_water_layers(h, n, 'ramsurf'))
              for n in range(1, 4 * n_now + 1)]
    finer = [v for v in placed if v <= dz and clears(v)]
    coarser = [v for v in placed if v > dz and clears(v)]
    if finer:
        finer_dz = max(finer)
    else:
        m = int(np.floor(zsrf / dz)) + 1
        finer_dz = printed(zsrf / (m + 0.5 * SURFACE_NODE_TOLERANCE))
    if coarser:
        coarser_dz = min(coarser)
    else:
        m = int(np.floor(zsrf / dz))
        coarser_dz = (printed(zsrf / (m + 0.5 * SURFACE_NODE_TOLERANCE))
                      if m >= 1 else None)
        if coarser_dz is not None and not (coarser_dz > dz
                                           and clears(coarser_dz)):
            coarser_dz = None
    return finer_dz, coarser_dz


def warn_if_ramsurf_surface_unresolved(env: Environment,
                                        dz: float, *,
                                        notices=None, knobs) -> None:
    """Warn when ramsurf1.5 will not march the surface asked for: the
    surface at ``r = 0`` more than :data:`SURFACE_NODE_TOLERANCE` of a
    cell below a depth node (a pinned ``dz``, or an automatic one the
    search window could not align), naming the ``dz`` that aligns it;
    and a deepest depression under :data:`MIN_SURFACE_RELIEF_CELLS`
    cells, which the staircase ``floor(zsrf/dz)·dz`` barely resolves."""
    zsrf = ramsurf_origin_depth(env, knobs=knobs)
    misfit = surface_node_misfit(zsrf, dz) if zsrf > 0.0 else 0.0
    if misfit > SURFACE_NODE_TOLERANCE:
        finer, coarser = dz_putting_the_surface_on_a_node(env, dz,
                                                          knobs=knobs)
        marched = float(np.floor(zsrf / dz)) * dz
        named = (f"dz={finer:.6g} m" if coarser is None else
                 f"dz={finer:.6g} m, or the coarser dz={coarser:.6g} m,")
        give_notice(notices,
            f"RAM:ramsurf: ramsurf1.5 zeroes every row down to "
            f"int(1 + zsrf/dz) (ramsurf1.5.f:115, :281-290), so with "
            f"dz={dz:.4g} m the pressure-release surface asked at "
            f"{zsrf:.4g} m at r=0 sits at {marched:.4g} m, "
            f"{zsrf - marched:.3g} m above it. Pinning {named} puts it "
            f"within {SURFACE_NODE_TOLERANCE:g} of a cell below a depth "
            f"node.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    nodes = ramsurf_surface_nodes(env, knobs=knobs)
    deepest = max(float(z) for _, z in nodes)
    if deepest > 0.0 and deepest / dz < MIN_SURFACE_RELIEF_CELLS:
        give_notice(notices,
            f"RAM:ramsurf: the altimetry's deepest depression, "
            f"{deepest:.3g} m, spans {deepest / dz:.2g} depth cells at "
            f"dz={dz:.4g} m. ramsurf1.5 resolves the surface to "
            f"floor(zsrf/dz)·dz, so its relief is staircased by up to a "
            f"cell; pin a finer dz to resolve it.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )


def ramsurf_surface_nodes(env, *, knobs):
    """The ramsurf1.5 surface profile, as ``(range, zsrf)`` nodes, from
    ``env.altimetry``.

    Sign convention: env.altimetry is (range, height) with height positive
    UP from sea level (the Bellhop ``.ati`` holds the opposite, a depth
    positive down, which its writer negates). ramsurf1.5 expects
    (range, zsrf) with zsrf >= 0 = depth BELOW z=0 (the pressure-release
    surface drops by zsrf at that range). So negate, then clamp wave crests
    (height > 0 → would imply zsrf < 0) to 0 — ramsurf only models surface
    depressions / ice keels, not crests above z=0. The clamp's notice is
    :func:`collins.warn_if_ramsurf_crests`, recorded when stage 3 resolves the
    launch; the deck writer re-reads these nodes without it."""
    if env.altimetry is None:
        raise ConfigurationError(
            "ramsurf backend requires env.altimetry to be set; "
            "got env.altimetry=None. Use the mpiramS backend "
            "(no altimetry) or supply an altimetry profile."
        )
    zsrf = [(float(r), deck_depth(-float(z), knobs=knobs))
            for r, z in env.altimetry.to_pairs()]
    if any(float(h) > 0 for _, h in env.altimetry.to_pairs()):
        zsrf = [(r, max(0.0, z)) for r, z in zsrf]
    return anchored_at_the_origin(zsrf)


def anchored_at_the_origin(pairs):
    """``pairs`` with an ``(0, first value)`` node in front when the first
    sample sits past the source.

    The Collins decks carry no node at r=0 of their own, and ``updat``
    (``ramgeo1.5.f:348-350``, ``rams0.5.f:304``, ``ramsurf1.5.f:349-354``
    for both the seafloor and the surface) interpolates every range below
    the second node along the FIRST segment — so a profile whose first
    sample is at r > 0 is extrapolated backward along its opening slope,
    against the constant extension ``env.bathymetry`` / ``env.altimetry``
    evaluate and against the mpiramS deck, which
    :func:`mpirams.prepare_bathymetry` anchors at the origin.
    """
    pairs = list(pairs)
    if pairs and float(pairs[0][0]) > 0.0:
        pairs.insert(0, (0.0, pairs[0][1]))
    return pairs
