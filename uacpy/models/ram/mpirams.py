"""The mpiramS backend: its deck (SSP, bathymetry, sediment profiles,
water attenuation), its domain and grid, the memory its field needs, and the
TL and broadband fields read from its output."""

import numpy as np
import warnings
from pathlib import Path
from typing import Optional, List
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.models._budget import memory_budget
from uacpy.models._notices import give_notice
from uacpy.models.ram._pe_phase import psi_to_travelling_wave
from uacpy.core.environment import (
    Environment,
)
from uacpy.core.receiver import Receiver
from uacpy.core.results import Field, SoundSpeeds
from uacpy.core.exceptions import ConfigurationError, NumericsWarning
from uacpy.io.mpirams_writer import (
    write_inpe, write_ssp_file, write_bth_file, write_ranges_file,
    write_sediment_file, write_water_attenuation_file,
)
from uacpy.models.ram._settings import RamGrid
from uacpy.models.ram._seabed import sample_at_depths
from uacpy.core.bathymetry import mask_below_seafloor
from uacpy.models.ram._interp import interp_to_receiver_grid
from uacpy.models.ram._domain import (
    absorber_span,
    check_source_row_is_solved,
    compute_zmax,
    deck_depth,
    flat_earth_depth,
    flat_earth_depth_inverse,
    launch_grid,
    ssp_column,
    ssp_range_axis,
    water_alpha_dB_per_wavelength,
    water_attenuation_active,
    water_attenuation_depths,
)
from uacpy.models.ram._band import (
    broadband_frequencies,
    report_band_grid_cost,
    requested_broadband_bins,
    resolve_broadband_grid,
)
from uacpy.models.ram._stability import (
    warn_stability_range_inert_on_a_multi_range_grid,
)
from uacpy.models.ram.grid import (
    compute_grid_lytaev,
    warn_if_trapped_modes_unresolved,
)


# Ceiling on the number of mpiramS sediment profiles written for a
# range-independent bottom whose seafloor water speed varies with range, and
# the speed spread (m/s) below which that range dependence is ignored.
_MAX_SED_PROFILES = 128
_CWG_RANGE_TOL = 0.01
#: Half the gap (m) of the pair of mpiramS sediment profiles written either
#: side of a bottom column switch that the other profiles would displace
#: (:func:`sediment_profile_ranges`): far below any march step, so the
#: pair moves the switch onto its range and nothing else.
_SWITCH_PAIR_HALF_GAP_M = 0.5


# Resident copies of the mpiramS field ``psif(nzo, nf, nr)`` (complex(8),
# ``peramx.f90:399``) over one run: the binary's own, the reader's
# (``mpirams_reader.py``), and the conjugate, ×4π and ×radial copies
# ``psi_to_travelling_wave`` makes before the depth interpolation.
_MPIRAMS_FIELD_COPIES = 5
_MPIRAMS_FIELD_BYTES_PER_SAMPLE = 16


#: How far an mpiramS output range may legitimately sit from the range it
#: stands for. The march stops stepping toward an output range once it is
#: inside 10 cm of it — ``if (abs(rnow-rend)<0.1_wp) exit`` (``ram.f90:169``,
#: "If we're within 10 cm, call it a day. Avoid rounding issues...") — and then
#: records that ``rnow`` as ``rout(irr)`` (``:271``). So a march that reached
#: every requested range can still report ranges up to this much short of them.
#: Beyond it, ``rout[-1]`` is not the requested range under another name; it is
#: a different position, and reading the field there under the requested
#: label would be a substitution rather than a rounding.
MPIRAMS_RANGE_TOL_M = 0.1


def mpirams_zmax(env: Environment, freq: float, dz: float, *,
                  max_range: float, notices=None, knobs, log,
                  speed_bounds) -> float:
    """PE domain depth for an mpiramS march, snapped onto the ``deltaz``
    grid.

    ``peramx.f90:391,404`` sizes the depth grid as
    ``icount = floor(zmax/deltaz - 0.5) + 2`` and then fills it with
    ``linspace(zg, 0, zmax, icount)``, so its actual spacing is
    ``zmax/(icount-1)`` while the depth operator (``ram.f90:51``
    ``cfact = 0.5/deltaz**2``, consumed at ``matrc.f90:60-62``) and the
    seafloor index (``ram.f90:101`` ``iz = floor(1 + zbc/deltaz)``) both
    assume ``deltaz``. The two agree only when ``zmax`` is an exact
    multiple of ``deltaz``.

    The value snapped has to be the one ``:388`` actually reads. Under
    ``earth_curvature`` ``:268-274`` rescales the whole depth axis
    *before* ``zmax = maxval(zw)``, so snapping the geometric depth leaves
    the transformed grid off the multiple again. Snapping in the
    transformed frame and mapping back is what makes the spacing exact on
    both paths.
    """
    zmax = compute_zmax(env, freq, max_range=max_range, notices=notices,
                        knobs=knobs, speed_bounds=speed_bounds)
    transformed = (flat_earth_depth(zmax) if knobs.earth_curvature
                   else zmax)
    snapped = (int(np.floor(transformed / float(dz) - 0.5)) + 1) * float(dz)
    result = (flat_earth_depth_inverse(snapped) if knobs.earth_curvature
              else snapped)
    if knobs.zmax is not None and abs(result - float(knobs.zmax)) > 1e-9:
        log(
            f"zmax {float(knobs.zmax):.6g} m snapped to {result:.6g} m so "
            f"mpiramS's depth grid reproduces dz={float(dz):g} m exactly"
        )
    return result


def prepare_ssp(env: Environment, work_dir: Path,
                 zmax_pe: float, dz: float, *, log) -> str:
    """
    Write SSP file from environment. Returns filename.

    The SSP is extended to ``zmax_pe`` (:func:`mpirams_zmax`), which is
    what mpiramS reads back as its domain depth (``zmax = maxval(zw)``,
    ``peramx.f90:388``).

    The column carries the user's true profile at every depth, sub-bottom
    included. ``matrc.f90:43-55`` reads ``cwg`` as a water property only
    above the seafloor index, and below it ``cwg`` is merely the reference
    the sediment speed is rebuilt against (``csg = cwg + cs``,
    ``mpiramS/src/ram.f90:345-346``) — which
    :func:`sediment_offsets` cancels per control point. Holding the column
    flat below the seabed instead would corrupt real water: mpiramS picks
    this column by nearest neighbour (``ram.f90:308-309``) while
    interpolating the seafloor continuously (``ram.f90:328-330``), so on a
    slope every range between two written columns takes a flat-start depth
    that is not its own seafloor.

    The range axis is the SSP's own breaks densified by
    :func:`_domain.ssp_range_axis`: the column is not keyed to the seabed, so
    it depends on the bathymetry only through the profiles the caller declared,
    and the intermediate columns make mpiramS's nearest-profile march follow
    the linear gradient between them.
    """
    n_points = max(50, int(float(zmax_pe) / float(dz) / 2))
    depths = np.linspace(0, zmax_pe, n_points)

    def _column(rng: float) -> np.ndarray:
        return ssp_column(env, rng, depths)

    ssp_filename = 'ssp.dat'
    ranges_m = ssp_range_axis(env)
    if len(ranges_m) > 1:
        log(f"Writing {len(ranges_m)} SSP profiles")
        speeds_2d = np.column_stack([_column(r) for r in ranges_m])
        write_ssp_file(work_dir / ssp_filename, depths, speeds_2d, ranges_m)
    else:
        write_ssp_file(work_dir / ssp_filename, depths, _column(0.0))

    return ssp_filename


def prepare_bathymetry(env: Environment, rmax: float,
                        work_dir: Path) -> str:
    """Write the bathymetry file, extended to ``r = 0`` and ``rmax``
    with its end values. Returns the filename.

    The ``rmax`` padding is load-bearing for the profile blocks:
    ``profl`` lays the sediment out from the bathymetry depth at the
    current range (``mpiramS/src/ram.f90:329``), and every ``profl`` call
    sits at ``rnow + dr/2 <= rmax``, so a table that reaches ``rmax``
    keeps each block anchored on the local seafloor rather than on the
    interpolator's out-of-range default. ``ram.f90:329`` falls back to
    the LAST depth past the table's end, so the padding is
    belt-and-braces there; the march's own extension of the table
    (``:92-94``) assumes it covers the track.
    """
    bth_filename = 'bathy.dat'

    bathy = env.bathymetry.to_pairs()
    if bathy[0, 0] > 0.0:
        bathy = np.vstack([[0.0, bathy[0, 1]], bathy])
    if bathy[-1, 0] < rmax:
        bathy = np.vstack([bathy, [rmax, bathy[-1, 1]]])
    write_bth_file(work_dir / bth_filename, bathy[:, 0], bathy[:, 1])
    return bth_filename


def control_point_depths(seafloor: float, sedlayer: float, nzs: int,
                          zmax: float) -> np.ndarray:
    """Absolute depths of ``profl``'s ``nzs`` sediment control points.

    ``zwork = [0, d, d + k·sedlayer/(nzs-3) (k = 1..nzs-3),
    max(zg(n), zwork(nzs-1)+1e-6)]`` (``mpiramS/src/ram.f90:334-342``), so
    points ``2..nzs-1`` are ``linspace(0, sedlayer, nzs-2)`` below the
    local seafloor ``d``, point 1 sits at the sea surface and point ``nzs``
    at the domain floor.
    """
    z = np.empty(int(nzs), dtype=float)
    z[0] = 0.0
    z[1:nzs - 1] = float(seafloor) + np.linspace(0.0, float(sedlayer),
                                                 int(nzs) - 2)
    z[nzs - 1] = max(float(zmax), z[nzs - 2] + 1e-6)
    return z


def sediment_offsets(env, rng, cp_abs, nzs, sedlayer, zmax):
    """``cs`` for one range: the offset that rebuilds ``cp_abs`` from the
    water column mpiramS reads at each control point.

    ``profl`` reconstructs the sediment speed as ``csg = cwg + cs``
    (``ram.f90:345-346``) with ``cwg`` the water profile splined onto the
    *whole* depth grid, sub-bottom included — so a single scalar offset
    only reproduces ``cp_abs`` where ``cwg`` happens to be flat. Taking the
    offset per control point makes the sub-bottom speed exact for any
    water column, which is what lets ``prepare_ssp`` write the true SSP
    instead of holding it flat below the seabed.
    """
    seafloor = float(np.asarray(env.bathymetry.eval(range=rng)).flat[0])
    z_ctrl = control_point_depths(seafloor, sedlayer, nzs, zmax)
    cs = np.asarray(cp_abs, dtype=float) - ssp_column(env, rng,
                                                      z_ctrl)
    # Control point 1 sits at the sea surface, and ``matrc`` reads the
    # sediment arrays only below the seafloor index
    # (``mpiramS/src/matrc.f90:43-55``), so its value never enters the
    # field. It is clamped non-negative because it is also the first
    # token of a ``.sed`` record, which the profile counter tests for
    # ``< 0`` as the ``-1 range`` header sentinel (``peramx.f90:128-131``)
    # — a seabed slower than the surface water would otherwise be
    # unwritable. Points at and below the seafloor keep their signed
    # offsets.
    cs[0] = max(cs[0], 0.0)
    return cs


def water_speed_at_seafloor(env: Environment,
                             range: float = 0.0) -> float:
    """
    Water-column sound speed at the *local* seafloor for range ``range``.

    This is what mpiramS's ``cwg`` is at the water-sediment interface.
    ``profl`` builds the sediment speed as ``csg = cwg + cs``
    (``ram.f90:345-346``) on a depth grid anchored at the seafloor depth
    *at that range*, so the offset that reproduces an absolute bottom
    speed ``cb`` is ``cb - cwg(z_seafloor(range))``. Referencing it to the
    deepest bathymetry point instead leaves the sediment fast wherever the
    seabed rises above it.
    """
    seafloor = float(np.asarray(env.bathymetry.eval(range=range)).flat[0])
    if env.ssp.is_range_dependent:
        ssp = env.ssp.eval(range=range).to_pairs()
    else:
        ssp = env.ssp.to_pairs()
    return float(np.interp(seafloor, ssp[:, 0], ssp[:, 1]))


def seafloor_speed_ranges(env: Environment) -> List[float]:
    """Ranges at which the sediment offsets change.

    ``cs`` is referenced to the water column at control points that sit
    below the *local* seafloor (:func:`sediment_offsets`), so it moves
    with the bathymetry — and it moves **continuously**, because
    ``ram.f90:328-330`` interpolates the seafloor between breakpoints while
    ``:308-309`` selects the nearest written profile. Sampling only the
    declared breaks would therefore leave every range between them
    referenced to a seafloor that is not its own.

    The bathymetry span is sampled uniformly and unioned with the declared
    SSP breaks, then decimated to ``_MAX_SED_PROFILES``.
    """
    breaks = {0.0}
    bathy = env.bathymetry
    if bathy is not None and bathy.n_ranges > 1:
        breaks.update(float(r) for r in bathy.ranges)
        r = np.asarray(bathy.ranges, dtype=float)
        breaks.update(float(v) for v in np.linspace(
            r.min(), r.max(), _MAX_SED_PROFILES))
    if env.ssp.is_range_dependent:
        breaks.update(float(r) for r in ssp_range_axis(env))
    ranges = sorted(breaks)
    if len(ranges) > _MAX_SED_PROFILES:
        keep = np.unique(
            np.linspace(0, len(ranges) - 1, _MAX_SED_PROFILES).astype(int))
        ranges = [ranges[i] for i in keep]
    return ranges


def prepare_bottom_properties(env: Environment, work_dir: Path,
                               absorber_span: float, zmax: float,
                               dz: float, *, knobs, log):
    """
    Extract bottom properties from environment and convert to mpiramS format.

    mpiramS's sediment model (profl in ram.f90) uses an N-point profile
    (``n_sediment_points``) interpolated over depth points
    [0, seafloor, ..interior.., seafloor+sedlayer, zmax]:

    - cs: sediment sound speed *perturbation* relative to water column,
          taken per control point (:func:`sediment_offsets`).
    - rho: sediment density relative to the water's
          (``env.water_density``): ``profl`` fixes the water at 1.
    - attn: sediment attenuation (dB/wavelength).
          The last point is set to absorbing-layer attenuation.

    ``absorber_span`` (:func:`_domain.absorber_span`) is the depth below the
    seafloor where the absorbing layer starts; :func:`sediment_profiles`
    stretches ``sedlayer`` to it, floored by the modelled sediment
    thickness.
    ``zmax`` locates the final control point. ``dz`` is the depth step the
    march will actually use; it sizes ``nzs`` (see below) and floors
    ``sedlayer``.

    Returns (sedlayer, nzs, cs, rho, attn, isedrd, sed_filename) from
    :func:`sediment_profiles`.
    """
    # ``profl`` spreads the interior control points at
    # ``dz_sed = sedlayer/(nzs-3)`` (``mpiramS/src/ram.f90:337``) and
    # interpolates the sediment arrays linearly between them
    # (``gorp``, ``:373-403``), so ``dz_sed`` — not ``nzs`` — is what
    # resolves an interface. With a fixed ``nzs`` that interval grows with
    # ``sedlayer``, which itself grows with the domain depth: measured on a
    # 100 m guide over a 60 m layer at 800 Hz, zmax=700 m against zmax=400 m
    # moved the field by 1.04 dB rms / 4.84 dB max at the default 1000
    # points, and equalising the control-point INTERVAL instead collapsed
    # that to 0.16 / 0.67 — so the interval is the mechanism, not the depth.
    # At the default the interval is already 0.263 m, i.e. 2.2 depth cells,
    # at zmax=400 m. Hold it to one depth cell: ``nzs-3 >= sedlayer/dz``.
    # ``n_sediment_points`` stays the floor, so a caller who raised it still
    # gets what they asked for.
    dz_grid = float(dz)
    # ``absorber_span`` is where the automatic domain puts the ramp start
    # (``_domain.adequate_zmax``: the real-seabed pad below the stack); one
    # depth cell is the floor that keeps control point ``nzs-1`` below the
    # seafloor point. ``sediment_profiles`` stretches it to the modelled
    # stack.
    sedlayer = max(dz_grid, float(absorber_span))

    # :func:`sediment_profiles` stretches ``sedlayer`` again, to the
    # deepest modelled stack (``Bottom.total_thickness_max``). Size
    # ``nzs`` against the span it will end up with, not the one it is
    # handed.
    nzs = max(int(knobs.n_sediment_points),
              int(np.ceil(max(sedlayer,
                              float(env.bottom.total_thickness_max()))
                          / dz_grid)) + 3)
    return sediment_profiles(env, work_dir, nzs, sedlayer, zmax, knobs=knobs,
                             log=log)


def sample_layered_column(col, nzs: int, sedlayer: float):
    """Sample ``col`` onto mpiramS's ``nzs`` sediment control points.

    ``profl`` places them at ``zwork = [0, d, d + k·sedlayer/(nzs-3)
    (k = 1..nzs-3), max(zg(n), zwork(nzs-1)+1e-6)]``
    (``mpiramS/src/ram.f90:334-342``) and interpolates the supplied arrays
    between them linearly (``gorp``, ``ram.f90:345-350`` and ``:373-403``).
    Points ``2..nzs-1`` are therefore ``linspace(0, sedlayer, nzs-2)`` below
    the local seafloor, point 1 sits at the sea surface and point ``nzs`` at
    the domain floor — the same depths :func:`control_point_depths`
    returns, which is what lets :func:`sediment_offsets` subtract the
    water column control point by control point.

    Point ``nzs-1`` carries the half-space, so the layer stack ends in a
    step resolved to ``sedlayer/(nzs-3)`` and every depth from there to the
    domain floor is constant.

    Constant is the physically required profile there, not merely the
    convenient one: everything below the half-space top is the absorbing
    layer, whose purpose is to swallow downgoing energy so it cannot
    reflect off the truncated grid at ``zmax``. Collins states the design
    directly — "the bottom of the computational grid (the depth zmax) is
    placed well below the ocean bottom interface and the attenuation is
    increased over the lower few wavelengths of the grid" (RAM manual;
    Collins & Siegmann, *Parabolic Wave Equations with Applications*,
    §on absorbing layers) — and Jensen, Kuperman, Porter & Schmidt,
    *Computational Ocean Acoustics* §7.3.4.3 (Fig. 7.5; the PE's own
    sponge is set up in §6.5.3) adds the constraint that "the
    sponge layer must be designed such that the internal reflections are
    insignificant". A sound-speed gradient inside the sponge is itself a
    refracting, partially reflecting feature, which is exactly what the
    layer exists to prevent; only the attenuation may vary through it.
    Spreading the layer/half-space contrast linearly to ``zmax`` puts that
    gradient across the whole absorber.

    Point 1 repeats the top-of-sediment value: ``matrc`` reads the bottom
    arrays only below the seafloor index
    (``mpiramS/src/matrc.f90:43-55``), so it never enters the water column
    and the water/sediment interface stays a step.
    """
    cp, rho, attn = sample_at_depths(col, nzs - 2,
                                     max_thickness=sedlayer)
    cp[-1] = col.halfspace.sound_speed
    rho[-1] = col.halfspace.density
    attn[-1] = col.halfspace.attenuation
    return (np.concatenate(([cp[0]], cp, [cp[-1]])),
            np.concatenate(([rho[0]], rho, [rho[-1]])),
            np.concatenate(([attn[0]], attn, [attn[-1]])))


def sediment_profile_ranges(env: Environment, *, log) -> List[float]:
    """Ranges the sediment deck carries a profile at: the bottom's own
    breaks plus every range where the water speed at the seafloor moves
    (:func:`varying_seafloor_speeds`), since ``cs`` is referenced to
    the local water column. ``[0.0]`` when one profile serves the whole
    march.

    mpiramS marches the profile nearest the current range
    (``minloc(abs(rp_sed-rint))``, ``ram.f90:241``), so a column switch
    sits midway between two written profiles, and ``Bottom.at`` switches
    midway between two of the bottom's breaks. Where other profiles fall
    between two bottom breaks the two midpoints differ; the switch is
    then put back on the bottom's own by a pair of profiles
    :data:`_SWITCH_PAIR_HALF_GAP_M` either side of it (each nearest to
    its own column), replacing a profile that sat on the switch itself,
    where ``Bottom.at`` ties."""
    breaks = {0.0}
    if env.bottom.is_range_dependent:
        breaks.update(float(r) for r in env.bottom.ranges)
    varying, _cwg = varying_seafloor_speeds(env, log=log)
    if varying is not None:
        breaks.update(float(r) for r in varying)
    if not env.bottom.is_range_dependent:
        return sorted(breaks)
    b_ranges = sorted(float(r) for r in env.bottom.ranges)
    for switch in (0.5 * (a + c) for a, c in zip(b_ranges, b_ranges[1:])):
        below = [r for r in breaks if r < switch]
        above = [r for r in breaks if r > switch]
        if (switch not in breaks and below and above
                and 0.5 * (max(below) + min(above)) == switch):
            continue
        half = min(_SWITCH_PAIR_HALF_GAP_M,
                   0.25 * (switch - max(below)) if below else np.inf,
                   0.25 * (min(above) - switch) if above else np.inf)
        breaks.discard(switch)
        breaks.update((switch - half, switch + half))
    return sorted(breaks)


def sediment_profiles(env, work_dir, nzs, sedlayer, zmax, *, knobs, log):
    """The 7-tuple :func:`prepare_bottom_properties` returns, for every
    seabed shape.

    One profile per range of :func:`sediment_profile_ranges`, each
    sampling the column nearest its range (``Bottom.column_index_at``, the
    same nearest rule ``ram.f90:316-320`` applies when it marches with
    the nearest written profile — so a column switch sits midway between
    the two written samples straddling the bottom break) and referencing
    ``cs`` to the water column at its own seafloor
    (:func:`sediment_offsets`). ``sedlayer`` is stretched to the deepest
    modelled stack so every layer is sampled; a pure half-space samples
    to the same value at every point. One range → the inline profile
    (``isedrd = 0``); more → the ``.sed`` deck.
    """
    bottom = env.bottom
    sedlayer = max(float(bottom.total_thickness_max()), float(sedlayer))
    ranges = sediment_profile_ranges(env, log=log)
    cs = np.zeros((nzs, len(ranges)))
    rho = np.zeros((nzs, len(ranges)))
    attn = np.zeros((nzs, len(ranges)))
    for i, rng in enumerate(ranges):
        col = bottom.columns[bottom.column_index_at(range=rng)]
        cp_abs, rho[:, i], attn[:, i] = sample_layered_column(
            col, nzs, sedlayer)
        cs[:, i] = sediment_offsets(env, rng, cp_abs, nzs,
                                    sedlayer, zmax)
    # The absorber is a floor on the half-space value, not a replacement:
    # RAM's guide has the attenuation "increased over the lower few
    # wavelengths", the Collins ramp takes the same max
    # (:func:`collins.ramp_absorbing_attenuation`), and ``gorp`` interpolates
    # linearly between control points (``ram.f90:373-405``), so a seabed
    # above ``absorber_attenuation`` would otherwise ramp DOWN into it.
    attn[-1, :] = np.maximum(attn[-2, :], knobs.absorber_attenuation)
    # mpiramS fixes the water density at 1 (``profl``: rhob is the
    # seabed's alone), so the absolute g/cm³ become ratios here, as
    # ``collins.collins_deck_base`` does for the Collins decks.
    rho = rho / float(env.water_density)
    log(f"Sediment: {len(ranges)} profile(s), nzs={nzs}, "
              f"sedlayer={sedlayer:.1f} m")
    if len(ranges) == 1:
        return sedlayer, nzs, cs[:, 0], rho[:, 0], attn[:, 0], 0, ''
    sed_filename = write_sediment_profiles(work_dir, ranges, cs,
                                           rho, attn)
    return (sedlayer, nzs, cs[:, 0].copy(), rho[:, 0].copy(),
            attn[:, 0].copy(), 1, sed_filename)


def varying_seafloor_speeds(env: Environment, *, log):
    """``(ranges, cwg)`` when the water speed at the seafloor varies with
    range, else ``(None, None)``.

    A range-independent bottom still needs one sediment profile per range
    whenever ``cwg`` moves, because mpiramS reconstructs the sediment
    speed as ``cwg + cs`` against the *local* water column.
    """
    ranges = seafloor_speed_ranges(env)
    if len(ranges) < 2:
        return None, None
    cwg = np.array([water_speed_at_seafloor(env, r) for r in ranges])
    if float(np.ptp(cwg)) <= _CWG_RANGE_TOL:
        return None, None
    log(
        f"Seafloor water speed varies {cwg.min():.1f}-{cwg.max():.1f} m/s "
        f"over range; writing {len(ranges)} sediment profiles so the "
        f"sediment speed stays referenced to the local seafloor."
    )
    return ranges, cwg


def write_sediment_profiles(work_dir, ranges, cs_profiles,
                             rho_profiles, attn_profiles) -> str:
    """Write the mpiramS ``.sed`` deck and return its filename."""
    sed_filename = 'sediment.sed'
    write_sediment_file(work_dir / sed_filename, np.asarray(ranges, float),
                        cs_profiles, rho_profiles, attn_profiles)
    return sed_filename


def resolve_mpirams_tl(env, source, receiver, *,
                        notices=None, knobs, log, speed_bounds) -> dict:
    """Stage 3 of an mpiramS COHERENT_TL run: the grid at the source
    frequency, the deck's ``(fc, Q, T)`` sweep collapsed to the one bin
    at fc (Q→∞, T=1) whatever ``self.q_factor`` / ``self.record_duration`` hold — the TL field
    keeps only the centre bin, so a wider band pinned for broadband runs
    would be marched and thrown away."""
    freq = float(source.frequencies[0])
    zsrc = float(np.min(source.depths))
    ranges = receiver.ranges
    rmax = float(np.max(ranges))

    dr, dz = resolve_mpirams_grid(env, freq, rmax, zs=zsrc,
                                  notices=notices, knobs=knobs, log=log,
                                  speed_bounds=speed_bounds)

    # COHERENT_TL collapses the mpiramS broadband window to one bin
    # (Q→∞, T=1): a band narrower than one bin marches fc alone
    # (``peramx.f90:370``).
    Q_tl = 1e6
    T_tl = 1.0
    # The deck carries (fc, Q, T) and the serial binary derives the
    # marched bins from it with no positivity guard of its own; the
    # same check the broadband path runs refuses the deck here, before
    # any file exists.
    marched = broadband_frequencies(freq, Q_tl, T_tl)
    log(
        f"mpiramS (TL mode): freq={freq:.1f} Hz, zs={zsrc:.1f} m, "
        f"dr={dr:.1f} m, dz={dz:.3f} m, Q={Q_tl:g}, T={T_tl:g}s"
    )
    log(f"Output grid: {len(ranges)} ranges x {len(receiver.depths)} depths")
    grid, rs = resolve_mpirams_deck(
        env, source, receiver, freq, Q_tl, T_tl, dr, dz, zmax_freq=freq,
        notices=notices, knobs=knobs, log=log, speed_bounds=speed_bounds)
    return dict(fc=freq, q_factor=Q_tl, record_duration=T_tl, bandwidth_hz=None, df_hz=None,
                marched_frequencies=marched, requested_frequencies=None,
                zmax_frequency=freq, stability_range_m=rs, grids=(grid,))


def resolve_mpirams_band(env, source, receiver, *,
                          notices=None, knobs, log, speed_bounds) -> dict:
    """Stage 3 of an mpiramS BROADBAND / TIME_SERIES run.

    The band the result carries is the caller's own bins when they
    define the grid, otherwise the ``(fc, Q, T)`` sweep itself. mpiramS
    marches every bin of it on ONE grid (deltaz/deltar are read once,
    peramx.f90:78-79, and the depth grid is sized at :391 before the
    frequency loop opens at :414-417), so the grid is sized at the band
    EDGES, not at fc: both steps at freq_max, where the wavelength is
    shortest and the Padé error per step largest, and the absorbing
    layer at freq_min, where it is longest.

    This is the same one-grid-for-the-band split as
    :func:`collins.resolve_collins_band` except for dr, which that path
    deliberately sizes at freq_min to hold rams0.5's rotated-Padé elastic
    march to few range steps. mpiramS has no elastic march to
    destabilise, so dr follows accuracy here: over a 50-350 Hz band an
    freq_min dr is 6x coarser and its Lytaev error at 350 Hz is ~19x the
    freq_max value, while the freq_max grid costs only ~15% more range steps
    than fc did.
    """
    freq, Q_bb, T_bb = resolve_broadband_grid(source,
                                              notices=notices, knobs=knobs,
                                              log=log)
    rmax = float(np.max(receiver.ranges))
    target = requested_broadband_bins(source, knobs=knobs)
    marched = broadband_frequencies(freq, Q_bb, T_bb)
    band = target if target is not None else marched
    freq_min = float(np.min(band))
    freq_max = float(np.max(band))

    dr, dz = resolve_mpirams_grid(
        env, freq_max, rmax, zs=float(np.min(source.depths)), notices=notices,
        knobs=knobs, log=log, speed_bounds=speed_bounds)
    log(
        f"mpiramS (broadband): fc={freq:.1f} Hz, Q={Q_bb}, T={T_bb}s, "
        f"band={freq_min:.2f}-{freq_max:.2f} Hz, "
        f"dr={dr:.1f} m, dz={dz:.3f} m (both at freq_max), "
        f"absorber at freq_min"
    )
    log(f"Bandwidth: {2.0 * freq / Q_bb:.2f} Hz "
              f"(fc ± {freq / Q_bb:.2f} Hz)")
    grid, rs = resolve_mpirams_deck(
        env, source, receiver, freq, Q_bb, T_bb, dr, dz, zmax_freq=freq_min,
        notices=notices, knobs=knobs, log=log, speed_bounds=speed_bounds)
    report_band_grid_cost(env, 'mpirams', grid, freq_min, freq_max,
                          notices=notices, knobs=knobs, log=log,
                          speed_bounds=speed_bounds)
    return dict(fc=freq, q_factor=Q_bb, record_duration=T_bb,
                bandwidth_hz=2.0 * freq / Q_bb, df_hz=1.0 / T_bb,
                marched_frequencies=marched, requested_frequencies=target,
                zmax_frequency=freq_min, stability_range_m=rs, grids=(grid,))


def resolve_mpirams_deck(env, source, receiver, freq: float,
                          q_factor: float, record_duration: float, dr: float, dz: float, *,
                          zmax_freq: float, notices=None, knobs, log,
                          speed_bounds):
    """``(grid, rs)``: the domain depth and the stability range one
    mpiramS deck carries, with the checks the deck needs before any file
    is written.

    mpiramS plants the source identically to the Collins backends
    (ram.f90:110-114) and its solver also starts at row 2
    (solvetri.f90:47), so the row-1 check applies to every source depth
    here too. ``zmax_freq`` sizes the domain depth and the absorbing
    layer, both of which scale with wavelength; a broadband caller
    passes ``freq_min`` so the layer is ``absorber_width_wavelengths``
    wavelengths deep at the longest wavelength in the band rather than at
    fc.
    """
    for zs in np.atleast_1d(source.depths):
        check_source_row_is_solved(float(zs), dz)
    rmax = float(np.max(receiver.ranges))
    zmax_pe = mpirams_zmax(env, float(zmax_freq), dz, max_range=rmax,
                           notices=notices, knobs=knobs, log=log,
                           speed_bounds=speed_bounds)
    check_mpirams_field_memory(
        zmax_pe, dz, freq, q_factor, record_duration, len(np.atleast_1d(receiver.ranges)),
        notices=notices, knobs=knobs)
    rs = (knobs.stability_range_m if knobs.stability_range_m is not None
          else rmax)
    warn_stability_range_inert_on_a_multi_range_grid(receiver,
                                                     notices=notices,
                                                     knobs=knobs)
    transformed = (flat_earth_depth(zmax_pe) if knobs.earth_curvature
                   else float(zmax_pe))
    return RamGrid(
        frequency=float(freq), dr=float(dr), dz=float(dz), zmax=zmax_pe,
        n_depth_points=int(np.floor(transformed / float(dz) - 0.5)) + 2,
    ), rs


def resolve_mpirams_grid(env, freq: float, rmax: float,
                          zs: Optional[float] = None, *, notices=None, knobs,
                          log, speed_bounds):
    """``(dr, dz)`` for an mpiramS march: user values where pinned, the
    Lytaev Padé-error optimizer for whatever is still ``None``.

    mpiramS reads ``deltaz``/``deltar`` once (``peramx.f90:78-79``) and
    sizes its depth grid once (``icount = floor(zmax/deltaz - 0.5) + 2``,
    ``:391``) *before* the frequency loop opens at ``:414-417``, so the one
    grid this returns marches every bin of a broadband band. Broadband
    callers therefore pass the band's **highest** frequency: both steps
    have to resolve the shortest wavelength in the band, and a step sized
    lower in the band is under-resolved for every bin above it.

    ``dr`` is the *longest* step mpiramS takes, not the step it takes
    everywhere. The march lands exactly on each requested output range
    (``ram.f90``, the ``abs(rend-rnow)<abs(deltar)`` branch), so a leg
    shorter than ``dr`` is marched in one step of that leg's length and
    the effective step is ``min(dr, leg)``. Nothing here needs to know
    the receiver grid for that: overshooting a requested range is
    impossible at any ``dr`` on any grid, which is a property of the
    Fortran and is pinned by
    ``test_ram_backends.TestMarchLandsOnEachOutputRange``.
    """
    dr = float(knobs.dr) if knobs.dr is not None else None
    dz = float(knobs.dz) if knobs.dz is not None else None
    if dr is None or dz is None:
        dr_auto, dz_auto = compute_grid_lytaev(
            env, freq, max_range=rmax, kind='mpirams', zs=zs,
            notices=notices, knobs=knobs, log=log, speed_bounds=speed_bounds
        )
        dr = dr_auto if dr is None else dr
        dz = dz_auto if dz is None else dz
    warn_if_trapped_modes_unresolved(env, freq, 'mpirams', dr, dz,
                                     rmax, notices=notices, knobs=knobs,
                                     log=log, speed_bounds=speed_bounds)
    return dr, dz


def check_mpirams_output_range_spacing(receiver: Receiver) -> None:
    """Refuse an output range grid mpiramS would march onto fewer
    positions than it has entries.

    The march steps toward each requested range until it is within
    ``MPIRAMS_RANGE_TOL_M`` of it and then records the position it
    stopped at as that range's ``rout`` entry. A pair of requested ranges
    closer together than the tolerance therefore shares one entry: the
    COHERENT_TL range axis stops being strictly increasing and the
    interpolator built on it refuses the grid, while BROADBAND labels its
    Field with ``rout`` and so returns one column twice under a range
    that appears twice. Raised from ``validate_inputs`` so both run modes
    get the same answer before a deck is written.
    """
    ranges = np.atleast_1d(np.asarray(receiver.ranges, dtype=float))
    gaps = np.diff(ranges)
    tight = np.flatnonzero(gaps < MPIRAMS_RANGE_TOL_M)
    if tight.size == 0:
        return
    i = int(tight[0])
    raise ConfigurationError(
        f"RAM:mpirams: receiver.ranges[{i}]={ranges[i]:.6g} m and "
        f"receiver.ranges[{i + 1}]={ranges[i + 1]:.6g} m are "
        f"{float(gaps[i]):.6g} m apart — closer than the "
        f"{MPIRAMS_RANGE_TOL_M} m the mpiramS march resolves. It stops "
        f"stepping once it is within that distance of a requested range "
        f"and reports the position it stopped at, so the two would come "
        f"back as one output range. {int(tight.size)} pair(s) in this "
        f"grid are that close.",
        remediation=(
            f"Space receiver.ranges at least {MPIRAMS_RANGE_TOL_M} m "
            f"apart, or pass RAM(backend='ramgeo'), whose output grid is "
            f"the binary's own uniform range axis and is interpolated "
            f"onto receiver.ranges at any spacing."
        ),
    )


def mpirams_field_shape(zmax_pe: float, dz: float, freq: float,
                         q_factor: float, record_duration: float, n_ranges: int, *, knobs):
    """``(nzo, nf, nr)`` of the field ``psif`` this deck makes mpiramS
    allocate, from the binary's own arithmetic: ``icount =
    floor(zmax/deltaz - 0.5) + 2`` on the transformed ``zmax``
    (``peramx.f90:391``, after the ``:268-274`` flat-earth rescale),
    ``nzo`` = every ``dzm``-th of those (``:393-396``), ``nf`` the
    ``(fc, Q, T)`` sweep and ``nr`` the ranges file."""
    transformed = (flat_earth_depth(zmax_pe) if knobs.earth_curvature
                   else float(zmax_pe))
    icount = int(np.floor(transformed / float(dz) - 0.5)) + 2
    dzm = max(1, int(knobs.depth_decimation))
    nzo = len(range(1, icount + 1, dzm))
    nf = int(broadband_frequencies(freq, q_factor, record_duration).size)
    return nzo, nf, int(n_ranges)


def check_mpirams_field_memory(zmax_pe, dz, freq, q_factor, record_duration, n_ranges, *,
                               notices=None, knobs) -> None:
    """Weigh the field mpiramS will allocate, times the copies the wrapper
    then holds of it, with :func:`~uacpy.models._budget.memory_budget`
    before the deck is written: its notice goes to ``notices``, and a field
    over what the host reports free is refused.

    The binary has no estimate of its own (``allocate(psif(nzo,nf,nr))``
    at ``peramx.f90:399`` is the first thing that fails), and the Python
    side holds about four more copies, so a many-range broadband run at
    high ``q_factor`` would die of a MemoryError or an OOM kill after the march.
    """
    nzo, nf, nr = mpirams_field_shape(zmax_pe, dz, freq, q_factor, record_duration,
                                      n_ranges, knobs=knobs)
    field = _MPIRAMS_FIELD_BYTES_PER_SAMPLE * nzo * nf * nr
    total = _MPIRAMS_FIELD_COPIES * field
    gib = 1024.0 ** 3
    notice = memory_budget(
        total, model_name='RAM:mpirams', what='mpiramS field copies',
        detail=(
            f"this deck makes the binary allocate a field of nzo={nzo} "
            f"depths × nf={nf} frequencies × nr={nr} ranges, "
            f"{field / gib:.2f} GiB of complex(8) (peramx.f90:399), and the "
            f"wrapper then holds about {_MPIRAMS_FIELD_COPIES} copies of it "
            f"({total / gib:.2f} GiB)."),
        remediation=(
            "Thin the field with depth_decimation (every n-th depth), "
            "fewer receiver ranges, or fewer frequency bins (a larger q_factor or "
            "a smaller record_duration)."))
    if notice is not None:
        give_notice(notices, notice.message, notice.category,
                    skip_file_prefixes=USER_FRAME_SKIP)


def write_mpirams_deck(inputs, *, knobs, log, speed_bounds) -> Path:
    """Stage 4 of an mpiramS run: every input file the march reads —
    ``ssp.dat``, the bathymetry and sediment files, the water-attenuation
    table, ``ranges.dat`` — and ``in.pe``, whose path it returns, all from
    ``inputs.settings.engine``.

    One writer for the narrowband and broadband decks, which differ only
    in the ``(fc, Q, T)`` sweep they carry — keeping one writer is what
    stops the two from drifting apart. The domain depth and the absorbing
    layer are the ones stage 3 sized at ``zmax_frequency`` (the band's
    lowest frequency, so the layer is ``absorber_width_wavelengths``
    wavelengths deep at the longest wavelength in the band rather than
    at fc).
    """
    env = inputs.env
    source = inputs.source
    receiver = inputs.receiver
    work_dir = inputs.work_dir
    engine = inputs.settings.engine
    grid = launch_grid(inputs)
    freq, Q, T = engine.fc, engine.q_factor, engine.record_duration
    dz = grid.dz
    zmax_pe = grid.zmax
    rmax = float(np.max(receiver.ranges))
    ssp_filename = prepare_ssp(env, work_dir, zmax_pe, dz, log=log)
    bth_filename = prepare_bathymetry(env, rmax, work_dir)
    sedlayer, nzs, cs, rho_arr, attn_arr, isedrd, sed_filename = \
        prepare_bottom_properties(
            env, work_dir,
            absorber_span(env, engine.zmax_frequency, zmax_pe, knobs=knobs,
                          speed_bounds=speed_bounds),
            zmax_pe, dz=dz, knobs=knobs, log=log)
    water_attn_filename = (
        write_mpirams_water_attenuation(env, work_dir, freq, Q, T,
                                        zmax_pe, knobs=knobs)
        if water_attenuation_active(env) else '')

    write_ranges_file(work_dir / 'ranges.dat', receiver.ranges)

    # mpiramS's horizontal-interpolation branch (ihorz=1) resamples the
    # SSP onto a uniform grid of nrp=nint(rmax/10000) points
    # (peramx.f90:253). For rmax < 5 km that rounds to nrp=0 -> a
    # zero-length allocate, an all-NaN sound-speed field, IEEE
    # divide-by-zero and a SIGABRT (exit -6); for rmax < 15 km it
    # collapses range dependence to 1-2 coarse 10-km samples. Always use
    # ihorz=0 so mpiramS steps directly between the per-range profiles
    # uacpy already builds in _prepare_ssp (_ssp_range_axis) — both
    # crash-free and more faithful than the buggy 10-km resample.
    deck = work_dir / 'in.pe'
    write_inpe(
        filepath=deck,
        fc=freq,
        q_factor=Q,
        record_duration=T,
        zsrc=float(source.depths[0]),
        dz=dz,
        dr=grid.dr,
        n_pade=knobs.n_pade,
        n_stability=knobs.n_stability,
        stability_range_m=engine.stability_range_m,
        depth_decimation=knobs.depth_decimation,
        ssp_filename=ssp_filename,
        earth_curvature=bool(knobs.earth_curvature),
        horizontal_interpolation=False,
        bathymetry_from_file=True,
        bth_filename=bth_filename,
        sedlayer=sedlayer,
        n_sediment_points=nzs,
        cs=cs,
        rho=rho_arr,
        attn=attn_arr,
        range_dependent_sediment=bool(isedrd),
        sed_filename=sed_filename,
        c0=engine.c0,
        water_attn_filename=water_attn_filename,
    )
    return deck


def write_mpirams_water_attenuation(env: Environment,
                                     work_dir: Path, freq: float,
                                     q_factor: float, record_duration: float,
                                     zmax_pe: float, *, knobs) -> str:
    """Write mpiramS's water-attenuation table and return its name.

    One column per bin of the ``(fc, Q, T)`` sweep the binary marches
    (:func:`_band.broadband_frequencies`; the binary refuses a column set that
    does not match), each the block :func:`_domain.water_attenuation_block`
    would give a Collins deck at that bin, on the r = 0 sound-speed
    column — the table is range-independent, and the local wavelength
    moves by under a percent across any SSP's range dependence. Depths
    are written in the deck's frame (:func:`_domain.deck_depth`), which is the
    grid ``wksqw`` interpolates onto.
    """
    frequencies = broadband_frequencies(float(freq), float(q_factor),
                                        float(record_duration))
    depths = water_attenuation_depths(env, zmax_pe)
    c = ssp_column(env, 0.0, depths)
    if env.absorption._needs_node_sound_speed:
        table = np.column_stack([
            water_alpha_dB_per_wavelength(env, float(f), depths, c)
            for f in frequencies
        ])
    else:
        # One ``alpha`` call over the whole sweep: one out-of-range
        # notice for the band, not one per bin.
        alpha_m = np.asarray(env.absorption.table(
            frequencies, depths=depths, units='dB/m').data,
            dtype=float).reshape(depths.size, np.size(frequencies))
        table = alpha_m * c[:, None] / np.asarray(frequencies)[None, :]
    name = 'water_attn.dat'
    write_water_attenuation_file(work_dir / name, deck_depth(depths,
                                                             knobs=knobs),
                                 frequencies, table)
    return name


def assemble_tl_field(result, env, source, receiver, work_dir,
                       freq, dr, *, attach_output_paths, knobs,
                       mask_source_axis, result_kwargs):
    """Stage 5 of an mpiramS COHERENT_TL run: the TL :class:`Field` from a
    finished ``psif`` result — pick the centre-frequency bin, interpolate
    complex pressure to the receiver grid (NaN-safe), convert to
    travelling-wave pressure and mask below the seafloor. ``dr`` (m) is
    the range a receiver at r = 0 is scaled at. The grid marched is the
    run settings' (``run_settings.engine.grids``)."""
    psif = result.pe_field  # (nzo, nf, nr)
    zg = result.depths
    rout = result.ranges

    # Center-frequency bin: nearest to fc. mpiramS sweeps a band
    # symmetric about fc, so odd nf has an exact middle; for an even
    # custom (Q,T) band this picks the closest bin (nf//2 would bias to
    # fc+Δf/2).
    center_idx = int(np.argmin(np.abs(result.frequencies - freq)))
    # pressure at center freq for all depths and ranges: (nzo, nr)
    pressure = psif[:, center_idx, :]

    # Interpolate COMPLEX PRESSURE from PE grid to receiver grid
    # BEFORE computing TL. Interpolating in dB destroys interference
    # nulls because linear interpolation of log-scale values smooths
    # out the sharp zeros in the field. Valid here — mpiramS writes its
    # output on the receiver range grid (write_ranges_file), so psif is
    # not undersampled in range. The Collins backends need the
    # modulus/phasor split instead (interp_envelope_to_receiver_grid).

    rcv_depths = receiver.depths
    # uacpy writes receiver.ranges themselves into ranges.dat, so rout is
    # the receiver grid back again and the two normally agree to the bit.
    # Where they do not, only ``MPIRAMS_RANGE_TOL_M`` of the difference is
    # the march calling a range reached; the rest is a range it did not
    # reach. Snap the first kind onto rout[-1] so the outermost receiver
    # still interpolates, and leave the second kind alone so it lands
    # off-grid and comes back NaN from interp_to_receiver_grid. A
    # tolerance of half the RECEIVER spacing would say nothing about the
    # march: on a short rout it stretches wide enough to pull unmarched
    # receivers onto rout[-1] and hand back that position's field under
    # their own range labels.
    rcv_ranges = np.asarray(receiver.ranges, dtype=float)
    beyond = rcv_ranges > rout[-1] + MPIRAMS_RANGE_TOL_M
    if np.any(beyond):
        warnings.warn(
            f"{knobs.model_name}: receiver ranges {rcv_ranges[beyond]} "
            f"exceed the PE marched range {rout[-1]}; those columns are "
            f"returned as NaN.",
            NumericsWarning,
            skip_file_prefixes=USER_FRAME_SKIP
        )
    # The near side needs no tolerance of its own, and its silence is not
    # the asymmetry it looks like. ``rout[-1]`` can fall arbitrarily short
    # of the last requested range when the march stops early, which is why
    # that side has to separate "reached" from "not reached". ``rout[0]``
    # cannot: the march starts at ``rnow=0``, runs to each requested range
    # in turn under ``ram.f90:169``'s exit test, and records that ``rnow``
    # as ``rout(irr)`` (``:271``); and write_ranges_file
    # writes ``receiver.ranges`` verbatim and in order, so ``rout[0]`` is
    # the first receiver's *own* achieved range. Receiver.ranges is
    # strictly increasing, so no receiver can sit below ``rout[0]`` by more
    # than the march's own 10 cm exit test — the same band the far side
    # snaps. This clip is therefore a no-op outside that band, and inside
    # it does what MPIRAMS_RANGE_TOL_M does out there. Pinned by the test
    # asserting every near-side receiver already lies within the march's
    # exit tolerance of ``rout[0]``, so the clip moves none of them.
    rcv_ranges = np.clip(rcv_ranges, rout[0], None)
    rcv_ranges = np.where(np.logical_and(~beyond, rcv_ranges > rout[-1]),
                          rout[-1], rcv_ranges)

    # Interpolate real and imaginary parts separately. A NaN sample in
    # the centre-frequency slice (PE divergence, or a depth the march did
    # not resolve) stays NaN through the interpolation and reaches the
    # user as no data, together with the receiver cells that read it.
    n_nan_p = int(np.count_nonzero(~np.isfinite(pressure)))
    if n_nan_p > 0:
        # expected; not in filterwarnings — emerges to user
        warnings.warn(
            f"RAM:mpirams: {n_nan_p}/{pressure.size} complex samples are "
            f"NaN/inf — the march did not solve there. They are returned "
            f"as no data, along with every receiver cell that "
            f"interpolates one, rather than as a level.",
            NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP
        )
    # Receivers outside the PE output domain return NaN pressure
    # so the resulting TL row is NaN (transparent in pcolormesh)
    # instead of saturating to the PRESSURE_FLOOR no-energy level.
    pressure_rcv = interp_to_receiver_grid(
        zg, rout, pressure, rcv_depths, rcv_ranges)

    # Compute TL from interpolated pressure.
    #
    # Collins' RAM convention (ram1.5.f, User Guide eq 4):
    #   psi = uu * f3 has r^{-1/2} removed from the actual pressure.
    #   TL = -20*log10(|psi|) + 10*log10(r)
    #
    # In mpiramS, psif = psi * exp(i*(k0*r + pi/4)) / (4*pi),
    # so |psi| = |psif| * 4*pi.
    #
    # A receiver at r = 0 is scaled at dr so the rest of the row still
    # computes; _mask_source_axis NaNs that column after assembly.
    log_ranges = rcv_ranges.astype(np.float64).copy()
    log_ranges[log_ranges <= 0.0] = dr

    # Convert the mpiramS .psif output to engineering travelling-
    # wave pressure (see ``models/ram/_pe_phase.py``). ``Field.dB``
    # only needs |p|, but downstream consumers that do coherent
    # integration get a meaningful phase.
    with np.errstate(divide='ignore', invalid='ignore'):
        pressure_field = psi_to_travelling_wave(
            pressure_rcv,
            convention='mpirams',
            ranges_m=log_ranges,
            range_axis=1,
        ).astype(np.complex128)
    # An exactly-zero sample is the pressure-release surface node (z = 0,
    # where mpiramS's field is identically zero). It is left at zero: the
    # shared ``transmission_loss_dB`` floors it to the one no-energy level every
    # model reports, rather than this wrapper writing a level of its own.

    field = Field(
        data=pressure_field,
        coords={'depth': receiver.depths, 'range': receiver.ranges},
        **result_kwargs(
            source,
            phase_reference='travelling_wave',
            backend='mpirams',
            frequencies=float(freq),
        )
    )
    field = mask_source_axis(field, source)
    field = field.mask_below_seafloor(env.bathymetry)
    attach_output_paths(
        field, work_dir, '',
        primary_files=(('psif_file', 'psif.dat'),)
    )
    return field


def assemble_broadband_field(result, env, source, receiver,
                              work_dir, engine, grid, *, attach_output_paths,
                              log, mask_source_axis, result_kwargs):
    """Stage 5 of an mpiramS BROADBAND / TIME_SERIES run: the
    transfer-function :class:`Field` from a finished ``psif`` result —
    convert to travelling-wave pressure, interpolate onto the receiver
    depth grid, trim the marched band onto the bins the settings ask for
    (``engine.requested_frequencies``), mask below the seafloor, and tag
    with the sweep and the grid marched.
    """
    dr, dz, zmax = grid.dr, grid.dz, grid.zmax
    # mpiramS stores psif = ψ·exp(+i(k0 r + π/4)) / (4π) under the
    # exp(+iωt) (engineering) carrier sign opposite to the
    # outgoing-wave convention every other uacpy model uses.
    # ``psi_to_travelling_wave`` conjugates — which flips the carrier
    # sign and turns the baked-in exp(+iπ/4) into exp(-iπ/4) — and
    # then applies exactly two factors, 4π and 1/√r. It applies no
    # π/4 of its own; doing so would double-count the one
    # peramx.f90:429 already wrote. The result is Collins' p(f,r,z)
    # in the engineering travelling-wave form
    # p ∝ ψ̄·exp(-ik0 r)·exp(-iπ/4)/√r.
    psif = result.pe_field  # (nzo, nf, nr)
    rout = np.asarray(result.ranges, dtype=np.float64)  # (nr,)
    zg = result.depths
    # A receiver at r = 0 is scaled at dr so the rest of the row still
    # computes; _mask_source_axis NaNs that column after assembly.
    rout_safe = rout.copy()
    rout_safe[rout_safe <= 0.0] = float(dr) if dr and dr > 0 else 1.0
    # psif shape: (nzo, nf, nr) — convert to engineering
    # travelling-wave pressure via models/ram/_pe_phase.py.
    pressure = psi_to_travelling_wave(
        psif,
        convention='mpirams',
        ranges_m=rout_safe,
        range_axis=2,
    )
    # An exactly-zero sample is the pressure-release surface node
    # (z = 0, where mpiramS's field is identically zero). It is left at
    # zero for the shared ``transmission_loss_dB`` floor to report, so no
    # wrapper writes a no-energy level of its own.

    # Map to receiver depth grid. PE domain extends below the
    # seafloor; output only the requested receiver depths.
    out_depths = receiver.depths
    if not np.array_equal(zg, out_depths):
        # zg is monotone but NOT uniform: with earth_curvature=1 peramx
        # un-transforms it by zg/(1 + eps/2 + eps²/3), eps = zg/Re
        # (peramx.f90:444-449), a quadratic map that stretches by
        # metres over a deep column. Bracket against the real axis.
        idx_lo = np.clip(
            np.searchsorted(zg, out_depths, side='right') - 1,
            0, len(zg) - 2)
        span = zg[idx_lo + 1] - zg[idx_lo]
        with np.errstate(divide='ignore', invalid='ignore'):
            w = np.where(span > 0.0,
                         (out_depths - zg[idx_lo]) / span, 0.0)
        w = np.clip(w, 0.0, 1.0)
        # Vectorized interpolation: (n_out, nf, nr)
        pressure = (pressure[idx_lo, :, :] * (1.0 - w[:, None, None]) +
                    pressure[idx_lo + 1, :, :] * w[:, None, None])
        # Depths outside the PE grid are NaN, matching the
        # COHERENT_TL below-domain convention — never a
        # plausible-looking edge extrapolation.
        outside = (out_depths < zg[0]) | (out_depths > zg[-1])
        pressure[outside, :, :] = np.nan
    else:
        out_depths = zg

    # Trim the marched (fc, Q, T) superset onto the caller's exact
    # frequency grid: pick the nearest marched bin for each requested
    # one and label the axis with the requested values, so H(f)
    # round-trips the request bin for bin.
    frq_out = np.atleast_1d(np.asarray(result.frequencies, dtype=float))
    target = engine.requested_frequencies
    if target is not None and not np.array_equal(frq_out, target):
        idx = np.array([int(np.argmin(np.abs(frq_out - f)))
                        for f in target])
        df_grid = 1.0 / float(engine.record_duration)
        if not np.allclose(frq_out[idx], target, rtol=0.0,
                           atol=0.51 * df_grid):
            raise ConfigurationError(
                f"RAM broadband: the marched frequency vector "
                f"({frq_out[0]:.4g}-{frq_out[-1]:.4g} Hz, "
                f"{frq_out.size} bins) does not contain the requested "
                f"bins ({float(target[0]):.4g}-"
                f"{float(target[-1]):.4g} Hz, {np.size(target)} bins) "
                f"within Δf/2 = {0.5 * df_grid:.4g} Hz."
            )
        pressure = pressure[:, idx, :]
        frq_out = np.asarray(target, dtype=float)

    log(f"Output: {len(out_depths)} depths x {frq_out.size} "
              f"freqs x {result.ranges.size} ranges")

    # (n_d, n_r, n_f).
    pressure = np.moveaxis(pressure, 1, 2)

    tf = Field(
        data=pressure,
        coords={
            'depth': out_depths,
            'range': rout,
            'frequency': frq_out,
        },
        # The binary's own header ``cmin`` is the WATER minimum
        # (``peramx.f90:295``, ``cmin=minval(cw)`` over the water grid),
        # carried for the time window rather than for the grid.
        speeds=SoundSpeeds(water_min=result.water_min),
        # The stock driver's time-sample count Nsam = 4·fc·T
        # (peramx.f90:354-356): the synthesis floors its FFT length at it.
        synthesis_floor=result.n_samples,
        **result_kwargs(
            source,
            phase_reference='travelling_wave',
            backend='mpirams',
            frequencies=frq_out,
        )
    )
    tf = mask_source_axis(tf, source)
    # Mask sub-seafloor samples with NaN (same semantics as every
    # backend), against the ranges the Field advertises.
    tf.data = mask_below_seafloor(tf.data, out_depths, rout, env.bathymetry)
    attach_output_paths(
        tf, work_dir, '',
        primary_files=(('psif_file', 'psif.dat'),)
    )
    return tf
