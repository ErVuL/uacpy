"""
Bellhop environment-file writer.

``write_bellhop_env_file`` writes the full ``.env`` (plus auxiliary ``.bty``,
``.ati``, ``.ssp``, ``.brc``/``.trc``, ``.sbp`` files) for a Bellhop run
(any backend). Handles SSP interpolation types, range-dependent
bathymetry/altimetry, surface/bottom boundary conditions, source/receiver
specifications, and run-type / beam-parameter configuration.
"""

import warnings
import numpy as np
from pathlib import Path
from typing import Optional, Union

from uacpy._log import log_message
from uacpy.core.absorption import ConstantAbsorption, _absorption_law
from uacpy.io.at_codes import (
    BELLHOP_GRID_TYPES, BELLHOP_RUN_TYPES, GEOMETRY_INTERP_CODES,
    boundary_code, writes_alpha_per_ssp_row,
)
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, UnsupportedFeatureError,
)
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.io.bathy_io import (
    write_bty_file,
    write_bty_long_format,
    write_ati_file,
)
from uacpy.core.deck_limits import (
    DECK_DEPTH_FMT, DECK_DEPTH_RESOLUTION_M, NO_RECEIVER_RANGE_FALLBACK_M,
)
from uacpy.io.oalib_writer import (
    SOURCE_TYPE_CODE,
    _biological_edge_nodes, _refuse_attenuation_past_at_ceiling,
    compose_topopt, deck_depth, format_halfspace_row,
    get_top_bc_code, quote_fortran_title, resolve_ssp_topopt,
    ssp_row_attenuation_texts, write_absorption_block, write_receiver_depths,
    write_receiver_ranges, write_source_depths, write_ssp,
    write_surface_halfspace,
)
from uacpy.io.refl_io import stage_reflection_file, stage_source_beam_pattern
from uacpy.core.units import m_to_km
from uacpy.io.input_checks import reject_unknown_kwargs
from uacpy.core.engine_defaults import (
    BELLHOP_BEAM_SHIFT,
    BELLHOP_BEAM_TYPE,
    BELLHOP_GRID_TYPE,
    BELLHOP_INTERP_ALTIMETRY,
    BELLHOP_INTERP_BATHYMETRY,
    BELLHOP_LAUNCH_ANGLES_DEG,
    BELLHOP_N_BEAMS,
    BELLHOP_RAY_STEP,
)


#: Height (m) of the SSP guard row above the highest altimetry crest.
#: ``bdryMod.f90:113-114`` needs the first SSP depth only at or above every
#: ``.ati`` point; the margin is uacpy guidance that keeps the row strictly
#: above it.
_ALTIMETRY_GUARD_MARGIN_M = 0.5

#: The guard columns of a range-dependent ``.ssp``, at ±this factor
#: times ``r_box``; :mod:`uacpy.io.env_reader` recognises them by it.
BELLHOP_SSP_GUARD_RANGE_FACTOR = 1.1


#: The ray box past the seafloor and the farthest receiver, as a factor on
#: each (:func:`bellhop_ray_box`). uacpy guidance, measured: BELLHOP drops a
#: ray once ABS(x(1)) > Box%r (bellhop.f90:571-572), and paths that turn back
#: from beyond the farthest receiver need room to turn. On the ASA ideal wedge
#: (25 Hz, receivers to 3 km, apex at 4 km), Bellhop against the analytic
#: field reads median |dTL| 3.06 dB with r_box = 1.01 x 3 km, whose up-slope
#: rays are killed before they turn, and 0.57 dB with 1.2 x; a ray-centred
#: 'R' beam's last column at 1.2 km moves 0.1-0.3 dB between the two. The
#: documented alternative is doc/bellhop.htm's "Make ZBOX, RBOX a bit (say
#: 1%) roomy too make sure rays are not killed the moment they hit the
#: bottom or are just reaching your furthest receiver." (the toolbox's decks
#: write 1.01-1.03).
BELLHOP_RAY_BOX_FACTOR = 1.2


#: The advanced Cerveny beam knobs (``beam_type`` 'C' or 'R') and the
#: values an unset one takes: the only ``**kwargs`` the beam block of
#: :func:`write_bellhop_env_file` reads (``None`` reads this table), and what
#: :class:`~uacpy.models.Bellhop`'s ``None`` defaults resolve to.
#: ``eps_multiplier`` 1, ``n_image`` 1, ``ib_win`` 5, ``component`` 'P' and
#: the curvature 'S' are BELLHOP's own built-in values (bellhopMod.f90:22-25,
#: ``Type = 'G S '``). The width 'F' (space filling) is uacpy guidance,
#: measured: the documented alternative is the 'M' (minimum width) every
#: Cerveny deck of the toolbox writes, whose width is set at the range
#: ``r_loop`` (bellhop.f90:378-380). On a 100 m Pekeris guide at 100 Hz to
#: 5 km, median TL off KRAKENC ('C' / 'R' beams): 'F' 5.90 / 5.46 dB; 'M' with
#: ``r_loop`` 1 km 11.1 / 10.5 dB, 5 km 6.97 / 6.30, 100 km 5.93 / 5.48 — 'M'
#: needs ``r_loop`` near 100 km to match 'F'; the curvature and window change
#: nothing there. ``r_loop`` has no built-in value (read in km; used only by
#: 'M'): 1 km
#: is uacpy guidance, the toolbox's calib decks (5 km receivers) — its 100 km
#: Munk / Dickins decks write 25-100 km.
CERVENY_DEFAULTS = {
    'beam_width_type': 'F', 'beam_curvature': 'S', 'component': 'P',
    'eps_multiplier': 1.0, 'r_loop': 1000.0, 'n_image': 1, 'ib_win': 5,
}


def cerveny_knobs(**given) -> dict:
    """The Cerveny knobs a deck carries: each given value, and
    :data:`CERVENY_DEFAULTS` for every one given as ``None`` or not at all."""
    return dict(CERVENY_DEFAULTS,
                **{k: v for k, v in given.items() if v is not None})

# Beam-type letters honoured by the Bellhop env reader (case-significant).
# Anything else maps to the geometric-hat DEFAULT case in
# ReadEnvironmentBell.f90:387-395.
_VALID_BEAM_TYPES = frozenset({'B', 'R', 'C', 'g', 'G', 'S'})

# RunType(4:4) and (5:5) (BELLHOP_GRID_TYPES, at_codes). Both have a silent
# CASE DEFAULT (ReadEnvironmentBell.f90:403-404 and :416-419), so a typo is
# never reported by the binary — it is overridden, and the deck no longer says
# what ran.
_VALID_SOURCE_TYPES = frozenset({'R', 'X'})

# Cerveny beam-shape letters (ReadEnvironmentBell.f90:178-185).
_VALID_BEAM_WIDTH_TYPES = frozenset({'F', 'M', 'W'})
_VALID_BEAM_CURVATURES = frozenset({'D', 'S', 'Z'})

# Field component selected in InfluenceCerveny* (influence.f90:120-130).
_VALID_COMPONENTS = frozenset({'P', 'V', 'H'})


def validate_beam_type(beam_type: str, who: str) -> None:
    """Reject beam-type letters the solver cannot honour.

    ``'b'`` (geometric Gaussian in ray-centred coordinates) is advertised by
    ``ReadEnvironmentBell.f90:387-395`` but ``bellhop.f90``'s ``PickEpsilon``
    (:403) calls ``ERROUT`` for it, and the C++/CUDA ports drop ``'b'`` from
    ``IsRayCen()`` (``bellhopcuda/src/runtype.hpp:54``) so they silently run
    the Cartesian variant instead.
    """
    if beam_type == 'b':
        raise ConfigurationError(
            f"{who}(beam_type='b') is not implemented: Bellhop's "
            f"PickEpsilon aborts on 'b' (geometric Gaussian beams in "
            f"ray-centered coordinates), and the C++/CUDA ports silently "
            f"substitute the Cartesian beam. Use beam_type='B' (geometric "
            f"Gaussian, Cartesian)."
        )
    if beam_type not in _VALID_BEAM_TYPES:
        raise ConfigurationError(
            f"{who}(beam_type={beam_type!r}) is not a known beam type. "
            f"Choose one of {sorted(_VALID_BEAM_TYPES)} "
            f"(case-significant; Bellhop would otherwise silently fall "
            f"back to a geometric-hat beam)."
        )


def validate_beam_shape(
    beam_width_type: str, beam_curvature: str, component: str, who: str,
) -> None:
    """Reject Cerveny beam-shape letters the solver does not implement.

    An unknown width letter leaves ``epsilonOpt`` at zero in ``PickEpsilon``
    (``bellhop.f90:372-390``) and an unknown component falls through to
    pressure (``influence.f90:120-130``) — both silent.
    """
    if beam_width_type not in _VALID_BEAM_WIDTH_TYPES:
        raise ConfigurationError(
            f"{who}(beam_width_type={beam_width_type!r}) is not valid. "
            f"Use 'F' (space filling), 'M' (minimum width) or 'W' (WKB); "
            f"any other letter leaves the Cerveny beam width at zero."
        )
    if beam_curvature not in _VALID_BEAM_CURVATURES:
        raise ConfigurationError(
            f"{who}(beam_curvature={beam_curvature!r}) is not valid. "
            f"Use 'D' (double), 'S' (single) or 'Z' (zero)."
        )
    if component not in _VALID_COMPONENTS:
        raise ConfigurationError(
            f"{who}(component={component!r}) is not valid. Use 'P' "
            f"(pressure), 'V' (vertical) or 'H' (horizontal); any other "
            f"letter silently returns pressure."
        )


def _bathymetry_within_mesh(bathymetry, z_max: float) -> np.ndarray:
    """Return ``(N, 2)`` bathymetry pairs quantised onto the mesh bottom.

    ``bdryMod.f90:211-212`` aborts on any ``.bty`` depth below the SSP mesh
    bottom, and ``z_max`` reaches the deck rounded to the ``.6f`` depth
    quantum, so a point sitting exactly on the seafloor can land a rounding
    residue beneath it. Pull those onto the mesh; anything more than a whole
    quantum deeper is a genuine bathymetry/mesh inconsistency and raises.
    """
    if hasattr(bathymetry, 'to_pairs'):
        bathymetry = bathymetry.to_pairs()
    pairs = np.asarray(bathymetry, dtype=float).copy()
    deepest = float(np.max(pairs[:, 1]))
    if deepest > z_max + DECK_DEPTH_RESOLUTION_M:
        raise ConfigurationError(
            f"Bellhop: bathymetry reaches {deepest:.6f} m but the SSP mesh "
            f"bottom is {z_max:.6f} m; Bellhop aborts with 'Bathymetry drops "
            f"below lowest point in the sound speed profile'."
        )
    pairs[:, 1] = np.minimum(pairs[:, 1], z_max)
    return pairs


def _write_altimetry_beside(filepath: Path, env, receiver, *,
                            interp_code: str, verbose: Union[bool, str]) -> None:
    """Write the ``.ati`` Bellhop opens under ``filepath``'s base name.

    ``env.altimetry`` is positive-up and the ``.ati`` is positive-down, so the
    height column is negated on the way out. Called only when the deck's
    ``TopOpt(5:5)`` says ``'~'``; the caller writes that letter.
    """
    ati_filepath = filepath.with_suffix(".ati")
    ati_data = env.altimetry.to_pairs()
    if ati_data.shape[0] == 1:
        # Single sample = constant offset; expand to a 2-point profile
        # spanning the receiver range so it isn't dropped. bdryMod.f90:131
        # tests the range axis with the strict monotonic() of
        # misc/monotonicMod.f90:32, so the second point has to sit strictly
        # beyond the first — hence the 1 m floor when the receivers do not
        # reach past it.
        r_last = max(float(np.max(receiver.ranges)),
                     float(ati_data[0, 0]) + 1.0)
        ati_data = np.array([[0.0, ati_data[0, 1]],
                             [r_last, ati_data[0, 1]]])
    ati_data[:, 1] = -ati_data[:, 1]
    write_ati_file(ati_filepath, ati_data, interp_type=interp_code)
    log_message('bellhop_writer',
                f"wrote altimetry file: {ati_filepath}",
                verbose=verbose)


def _write_bathymetry_beside(filepath: Path, env, receiver, *, z_max: float,
                             interp_code: str) -> bool:
    """Write the ``.bty`` when the seabed varies with range; report whether
    one was written, since the deck's ``BotOpt(2:2)`` has to say so.

    The ``.bty`` TYPE is auto-selected by the writer: ``write_bty_long_format``
    emits ``'LL'``/``'CL'``, ``write_bty_file`` the bare ``'L'``/``'C'`` —
    Bellhop compares the whole ``CHARACTER(LEN=2) btyType``
    (``bellhop.f90:552``), so ``'CS'`` never matches ``'C '`` and the blank
    second character selects the short format (``bdryMod.f90:179-181``). The
    first char is the interpolation the caller resolved.
    """
    is_range_dependent_bathy = env.bathymetry.n_ranges > 1
    rd_bottom = (env.bottom.is_range_dependent
                 and len(env.bottom.ranges) > 0)
    if not (is_range_dependent_bathy or rd_bottom):
        return False

    bty_filepath = filepath.with_suffix(".bty")
    if rd_bottom:
        # The long-format .bty is the only vehicle for per-range
        # geoacoustics; a flat bathymetry becomes a 2-point constant profile
        # so the property breaks still reach Bellhop.
        bathy_for_bty = env.bathymetry
        if not is_range_dependent_bathy:
            r_last = max(float(np.max(env.bottom.ranges)),
                         float(np.max(receiver.ranges)))
            bathy_for_bty = np.array(
                [[0.0, env.depth], [r_last, env.depth]])
        write_bty_long_format(
            bty_filepath, _bathymetry_within_mesh(bathy_for_bty, z_max),
            env.bottom, interp_type=interp_code,
        )
    else:
        write_bty_file(
            bty_filepath,
            _bathymetry_within_mesh(env.bathymetry, z_max),
            interp_type=interp_code,
        )
    return True


def bellhop_ray_box(env, receiver, *, z_box: Optional[float] = None,
                    r_box: Optional[float] = None):
    """``(z_box, r_box)``: the ray box a Bellhop deck traces rays in
    (``Box%z`` in m, ``Box%r`` in m), each pinned value kept as given.

    ``z_box`` defaults to :data:`BELLHOP_RAY_BOX_FACTOR` (1.2) x
    ``env.depth``. ``r_box`` defaults to 1.2 x the receiver range extent, or
    10 km when every receiver range is 0. Box%r is a horizontal-range cut-off
    for ray integration: the 20 % pad leaves room for the paths that turn
    back from beyond the farthest receiver (measured on an up-slope wedge,
    :data:`BELLHOP_RAY_BOX_FACTOR`); enlarging it further only risks rays
    leaving the range window where a range-dependent .ssp defines the sound
    speed (BHC_ERR_OUTSIDE_SSP).
    """
    if z_box is None:
        z_box = BELLHOP_RAY_BOX_FACTOR * env.depth
    if r_box is None:
        r_box = (BELLHOP_RAY_BOX_FACTOR * receiver.range_max
                 if receiver.range_max > 0
                 else NO_RECEIVER_RANGE_FALLBACK_M)
    return z_box, r_box


def _check_bellhop_options(run_type, beam_type, source_type, grid_type,
                           n_beams) -> int:
    """Refuse RunType letters and beam counts Bellhop cannot honour; return
    the beam count (``None`` is 0, Bellhop's own estimate)."""
    validate_beam_type(beam_type, 'write_bellhop_env_file')
    for _value, _allowed, _name, _cite in (
        (run_type, BELLHOP_RUN_TYPES, 'run_type',
         "ReadEnvironmentBell.f90:376-377 ERROUTs any other RunType(1:1)"),
        (source_type, _VALID_SOURCE_TYPES, 'source_type',
         "ReadEnvironmentBell.f90:403-404 silently overrides RunType(4:4) to 'R'"),
        (grid_type, BELLHOP_GRID_TYPES, 'grid_type',
         "ReadEnvironmentBell.f90:416-419 silently overrides RunType(5:5) to 'R'"),
    ):
        if str(_value) not in _allowed:
            raise ConfigurationError(
                f"write_bellhop_env_file({_name}={_value!r}): must be one "
                f"character from {sorted(_allowed)}. {_cite}, so a longer or "
                f"unknown value shifts or silently changes the RunType record "
                f"and the deck stops describing the run.")

    if n_beams is None:
        n_beams = 0
    if n_beams < 0:
        raise ConfigurationError(
            f"write_bellhop_env_file(n_beams={n_beams}) must be >= 0. "
            f"angleMod.f90:38 auto-estimates only on an exact 0; a negative "
            f"count is never rejected — ALLOCATE clamps it to MAX(3, Nalpha) "
            f"(:54) and the beam loop (bellhop.f90:262) then traces nothing, "
            f"leaving an all-zero field."
        )
    if n_beams == 1 and str(run_type)[:1] != 'R':
        raise ConfigurationError(
            f"write_bellhop_env_file(n_beams=1, run_type={run_type!r}): a "
            f"single beam cannot carry an influence calculation — "
            f"bellhop.f90:176-178 leaves Dalpha = 0, so every beam has zero "
            f"width and the field, arrivals or eigenrays come back empty at "
            f"exit 0 (the .prt says 'Too few beams' only for RunType 'C'). "
            f"Use n_beams >= 2, or 0 to let Bellhop choose; one beam is "
            f"meaningful only for run_type 'R'.")
    return n_beams


def _check_bellhop_env(env, interp_ssp) -> str:
    """Refuse an environment the deck cannot describe to Bellhop; return the
    ``TopOpt(1)`` SSP letter."""
    _refuse_attenuation_past_at_ceiling(env, 'write_bellhop_env_file')
    interp_char = resolve_ssp_topopt(env, interp_ssp)
    # ReadEnvironmentBell.f90:459 accepts the 'P' (precalculated IRC) letter
    # but bellhop.f90:681's boundary SELECT CASE has no 'P' branch, so its
    # CASE DEFAULT (:779) aborts the run at the first bottom bounce.
    if boundary_code(env.bottom.halfspace_at(range=0.0).acoustic_type) == 'P':
        raise UnsupportedFeatureError(
            'Bellhop', "a 'precalc' (.irc) bottom — bellhop.f90:681's "
            "boundary SELECT CASE has no 'P' branch, so the run aborts at "
            "the first bottom bounce ('Unknown boundary condition type')",
            alternatives=["acoustic_type='file' (a .brc table)",
                          'Kraken / Scooter, which read .irc natively'])
    return interp_char


def _boundary_interp_codes(interp_bathymetry, interp_altimetry):
    """``(bty_code, ati_code)``: the ``.bty`` and ``.ati`` interpolation
    letters of ``interp_bathymetry`` / ``interp_altimetry``."""
    bty_code = GEOMETRY_INTERP_CODES.get(str(interp_bathymetry).lower())
    ati_code = GEOMETRY_INTERP_CODES.get(str(interp_altimetry).lower())
    if bty_code is None:
        raise ConfigurationError(
            f"interp_bathymetry must be 'linear' or 'curvilinear'; "
            f"got {interp_bathymetry!r}."
        )
    if ati_code is None:
        raise ConfigurationError(
            f"interp_altimetry must be 'linear' or 'curvilinear'; "
            f"got {interp_altimetry!r}."
        )
    return bty_code, ati_code


def _check_irregular_grid(receiver, grid_type) -> None:
    """``ReadEnvironmentBell.f90:414`` ERROUTs an irregular grid whose
    receiver-depth and receiver-range counts differ, so refuse it here
    rather than emit a deck bellhop.exe rejects at READIN."""
    if (grid_type == 'I'
            and len(receiver.depths) != len(receiver.ranges)):
        raise ConfigurationError(
            f"grid_type='I' (irregular) requires len(receiver.depths) == "
            f"len(receiver.ranges); got {len(receiver.depths)} depths and "
            f"{len(receiver.ranges)} ranges.",
            remediation="Use grid_type='R' for a rectilinear "
                        "(Cartesian-product) grid, or rebuild the "
                        "Receiver with matched arrays.",
        )


def _write_top_block(f, filepath: Path, env, source, receiver, *,
                     interp_char: str, ati_code: str, verbose,
                     drop_repeated_angles: bool,
                     single_beam: bool = False) -> bool:
    """Title, frequency, NMedia, TopOpt, the absorption rows and the top
    half-space row, staging the ``.ati`` and ``.trc`` the surface needs;
    return whether the surface carries an altimetry."""
    f.write(f"{quote_fortran_title(env.name)}\n")
    f.write(f"{source.frequencies[0]:.6f}\n")

    # Number of media. 1 is the only legal value —
    # ReadEnvironmentBell.f90:56 ERROUTs on anything else ("sediment
    # layers must be handled using a reflection coefficient").
    f.write("1\n")

    top_bc = get_top_bc_code(env)

    # Position 5: Altimetry flag ('~' = read .ati file, ' ' = flat surface)
    has_altimetry = (env.altimetry is not None
                     and env.altimetry.n_ranges >= 1)
    alti_char = '~' if has_altimetry else ' '
    # Position 6: 'I' traces one beam of the fan, whose index follows NBeams
    # (ReadEnvironmentBell.f90:337-342, angleMod.f90:32-33).
    top_opt = compose_topopt(interp_char, top_bc, env, pos5=alti_char,
                             pos6='I' if single_beam else ' ')
    f.write(f"'{top_opt}'\n")

    # ReadEnvironmentBell.f90:59 calls ReadTopOpt, which consumes the
    # Francois-Garrison / biological records itself (:308-320); only then
    # does :69 CALL TopBot read the top half-space row (:474). Emitting
    # the half-space first feeds it the absorption record.
    write_absorption_block(f, env)

    write_surface_halfspace(f, env)

    if has_altimetry:
        _write_altimetry_beside(filepath, env, receiver,
                                interp_code=ati_code, verbose=verbose)

    # A surface reflection file ('F') is read by base name, so it has to
    # sit next to the .env as <base>.trc.
    if top_bc == 'F':
        stage_reflection_file(
            getattr(env.surface, 'reflection_file', None),
            filepath, boundary='top', verbose=verbose,
            drop_repeated_angles=drop_repeated_angles,
        )
    return has_altimetry


def _bellhop_ssp(env, *, has_altimetry: bool, interp_char: str):
    """``(ssp_depths, ssp_matrix, z_max)``: the one depth-aligned profile the
    ``.env`` SSP rows and the ``.ssp`` matrix are both built from, and the
    deck's quantised seafloor depth."""
    # Extend SSP above MSL to cover any wave crests (env.altimetry
    # positive-up, SSP z-axis positive-down). bdryMod.f90:113-114 aborts
    # with 'Altimetry rises above highest point in the sound speed
    # profile' unless SSP%z(1) — the .env's first SSP depth, which
    # ReadEnvironmentBell.f90:88 adopts as the top boundary — is at or
    # above every .ati point, so the guard row clears the highest crest.
    z_min = 0.0
    if has_altimetry:
        max_alti_above_msl = env.altimetry.heights.max()
        if max_alti_above_msl > 0:
            z_min = -max_alti_above_msl - _ALTIMETRY_GUARD_MARGIN_M

    z_max = deck_depth(env.depth)

    # Bellhop/sspMod.f90:922 ends the .env SSP block at the first row whose
    # depth equals the Depth field of the NPts line, so the profile has to
    # reach z_max exactly or the reader runs on into the bottom block.
    # Bellhop/sspMod.f90:427-431 then pairs row iz2 of the .ssp matrix with
    # SSP%z(iz2) from the .env and stops after SSP%NPts rows, so both blocks
    # are built from this one depth-aligned profile. A guard row above the
    # shallowest sample covers a raised surface or an SSP starting below
    # 0 m; it goes into both.
    ssp_aligned = env.ssp.extend_to(z_max)
    ssp_depths = np.asarray(ssp_aligned.depths, dtype=float)
    ssp_matrix = np.asarray(ssp_aligned.sound_speed, dtype=float)
    if z_min < ssp_depths[0]:
        ssp_depths = np.insert(ssp_depths, 0, z_min)
        ssp_matrix = np.vstack([ssp_matrix[0, :], ssp_matrix])
    # A Biological law is evaluated at SSP nodes only
    # (Bellhop/sspMod.f90:906): a node pair across each layer edge, in
    # the .env rows and the .ssp matrix alike, confines it to the layer.
    ssp_depths, ssp_matrix = _biological_edge_nodes(
        ssp_depths, ssp_matrix, _absorption_law(env.absorption), interp_char,
        who='write_bellhop_env_file')
    return ssp_depths, ssp_matrix, z_max


def _warn_if_quad_rows_hold_range0_absorption(env, ssp_matrix) -> None:
    """Say so when a law the SSP rows carry (Francois-Garrison, a table)
    meets a range-dependent ``'Q'`` profile whose speeds leave the
    range-0 column: Bellhop takes the imaginary sound speed from the
    ``.env`` rows alone (``Bellhop/sspMod.f90:520``), so the range-0 α(z),
    converted at the range-0 speed, is used at every range."""
    absorption = _absorption_law(env.absorption)
    if (not writes_alpha_per_ssp_row(absorption)
            or isinstance(absorption, ConstantAbsorption)):
        return
    c = np.asarray(ssp_matrix, dtype=float)
    if np.all(c == c[:, :1]):
        return
    # Step.f90 accumulates Im tau as -cimag/c**2 per metre, so a loss set
    # at c0 is applied as (c0/c)**2 of itself where the speed is c.
    scale = (c[:, :1] / c) ** 2
    worst = float(np.max(np.abs(scale - 1.0)))
    warnings.warn(
        f"write_bellhop_env_file: the {absorption._short()} absorption varies "
        f"with depth and the sound-speed profile is range-dependent ('Q'). "
        f"Bellhop takes the imaginary sound speed from the .env rows only "
        f"(Bellhop/sspMod.f90:520), which hold the range-0 column, so the "
        f"range-0 alpha(z) is used at every range; where the profile's speed "
        f"differs from range 0 the loss is off by up to {100.0 * worst:.3g} %.",
        FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)


def _write_quad_ssp_beside(filepath: Path, env, ssp_matrix, *, r_box,
                           receiver, verbose) -> None:
    """The ``.ssp`` a ``TopOpt(1) == 'Q'`` deck needs, which Bellhop opens
    unconditionally (``ReadEnvironmentBell.f90:262-268``), padded so its
    range axis brackets the ray box on both sides. A range-independent
    profile is its one column at ``r = 0``, held over the box by the same
    padding."""
    ssp_file = filepath.with_suffix('.ssp')
    ssp_ranges = np.asarray(
        env.ssp.ranges if env.ssp.is_range_dependent else [0.0], dtype=float)
    ssp_data = ssp_matrix.copy()
    # Bellhop drops a ray once it passes Box%r, and a ray landing on
    # the last SSP range is still flagged "outside the soundspeed box",
    # so the .ssp must extend *strictly past* r_box. When the profile
    # grid stops at or before the box, hold the last profile constant
    # out to 1.1·r_box (range-independent beyond the last profile).
    ssp_max = float(ssp_ranges.max()) if ssp_ranges.size else 0.0
    if ssp_ranges.size and ssp_max <= r_box:
        ssp_ranges = np.append(ssp_ranges,
                               BELLHOP_SSP_GUARD_RANGE_FACTOR * r_box)
        ssp_data = np.column_stack([ssp_data, ssp_data[:, -1]])
    # Warn only where RECEIVERS sit beyond the last profile: there
    # the held-constant column is the sound speed a returned value
    # was computed in. The padding out to r_box above is not such a
    # choice — no receiver lies in that margin.
    if (env.ssp.is_range_dependent and ssp_ranges.size
            and ssp_max < receiver.range_max):
        warnings.warn(
            f"Bellhop: the range-dependent SSP spans to {ssp_max:.0f} m "
            f"but receivers reach {receiver.range_max:.0f} m; the last "
            f"profile is held constant beyond {ssp_max:.0f} m, so TL at "
            f"the outer receivers is computed in a range-independent "
            f"column. Define SSP profiles out to at least "
            f"{receiver.range_max:.0f} m, or shorten receiver.ranges. "
            f"Rays are traced to the box at {r_box:.0f} m "
            f"({BELLHOP_RAY_BOX_FACTOR:g} x receiver.range_max unless r_box= "
            f"says otherwise); "
            f"profiles beyond the box are never read.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    # The source sits at range 0 == Seg.r[0]; a ray back-scattered off
    # a steep up-slope can step to negative range, which Quad's box
    # check (x < Seg.r[0]) flags as BHC_ERR_OUTSIDE_SSP under
    # bellhopcuda. Prepend a guard column at -1.1·r_box holding the
    # first profile constant so the box brackets that excursion (no
    # warning: negative range is not a user-controllable axis).
    if ssp_ranges.size and ssp_ranges.min() >= 0.0:
        ssp_ranges = np.insert(ssp_ranges, 0,
                               -BELLHOP_SSP_GUARD_RANGE_FACTOR * r_box)
        ssp_data = np.column_stack([ssp_data[:, 0], ssp_data])
    write_ssp(ssp_file, ssp_ranges, ssp_data)
    log_message('bellhop_writer',
                f"wrote range-dependent SSP file: {ssp_file}",
                verbose=verbose)


def _write_ssp_block(f, env, ssp_depths, ssp_matrix, z_max, *,
                     frequency: float) -> None:
    """The ``NPts Sigma Depth`` line and one six-column row per SSP node,
    each carrying the row law's ``alphaI`` at the deck ``frequency``
    (:func:`~uacpy.io.oalib_writer.ssp_row_attenuation_texts`). Under a
    range-dependent ``'Q'`` profile the ``.ssp`` matrix holds the real
    speeds only and the attenuation is taken from these rows
    (``Bellhop/sspMod.f90:520``), so it is evaluated on the range-0
    column."""
    # Bellhop reads this line as (NPts, Sigma, Depth) per
    # ReadEnvironmentBell.f90:73. NPts is informational — the SSP block
    # ends on Depth, not on a count ("NPts, Sigma not used by BELLHOP",
    # bellhop.f90:50; "LP: ignored", bellhopcuda/src/module/ssp.hpp:80-81
    # against the Depth match at :115). Sigma is the top-boundary RMS
    # roughness, NOT z_min; always emit 0.0, altimetry crests go through
    # the .ati file.
    f.write(f"{ssp_depths.size}  0.0  {z_max:{DECK_DEPTH_FMT}},\n")

    # SSP row: z alphaR betaR rhoR alphaI betaI /
    # Bellhop/sspMod.f90:903 READs all six into the module variables
    # ReadEnvironmentBell.f90:474 (TopBot) has already filled, and a short
    # row leaves the missing ones untouched — the C++ port models this
    # explicitly as `RecycledHS` (bellhopcuda/src/module/ssp.hpp:105-109).
    # Emit all six or an elastic top halfspace leaks its ice properties
    # into the water column.
    alpha_i = ssp_row_attenuation_texts(env, frequency, ssp_depths,
                                        ssp_matrix[:, 0])
    for depth, c, alpha in zip(ssp_depths, ssp_matrix[:, 0], alpha_i):
        # rhoR is the water density Bellhop's own reflection uses
        # (ReflectMod.f90:153 forms R from the SSP's rho against
        # HS%rho), so it carries env.water_density like the AT rows.
        f.write(
            f"{depth:.6f} {c:.6f} 0.0 {env.water_density:.6f} "
            f"{alpha} 0.0 /\n"
        )


def _write_bottom_block(f, filepath: Path, env, receiver, *, z_max,
                        bty_code: str, verbose,
                        drop_repeated_angles: bool) -> None:
    """``BotOpt Sigma`` and the bottom half-space row, staging the ``.bty``
    and ``.brc`` the seabed needs."""
    hs = env.bottom.halfspace_at(range=0.0)
    bottom_type = boundary_code(hs.acoustic_type)

    wrote_bty = _write_bathymetry_beside(
        filepath, env, receiver, z_max=z_max, interp_code=bty_code)
    # BotOpt(2:2) = '~' tells ReadEnvironmentBell.f90:97-99 to expect the
    # .bty, so the letter and the file are decided together. The 2nd field
    # is the sigma slot (:93); Bellhop echoes it and drops it — "NPts,
    # Sigma not used by BELLHOP" (bellhop.f90:50).
    f.write(f"'{bottom_type}{'~' if wrote_bty else ''}' "
            f"{hs.roughness:.6f}\n")

    if bottom_type == "F":
        # Bellhop finds the .brc by name convention (same base name as the
        # .env), so staging it beside the file is the whole job — no extra
        # lines go into the env.
        stage_reflection_file(hs.reflection_file, filepath,
                              boundary='bottom', verbose=verbose,
                              drop_repeated_angles=drop_repeated_angles)
    elif bottom_type == "A":  # Acousto-elastic halfspace
        # Per ReadEnvironmentBell.f90:474 the row is
        # READ(ENVFile,*) zTemp, alphaR, betaR, rhoR, alphaI, betaI
        # i.e. depth cp cs rho alpha_p alpha_s — the 6th column is
        # SHEAR attenuation, NOT roughness. Roughness (sigma) lives
        # on the preceding BOT line ('A' sigma).
        # The read is list-directed, so no column width is imposed (the
        # F10.2 at :475 is the .prt echo). Six decimals on every column
        # is what the AT writers share, and it is what keeps one
        # Environment from handing Bellhop and Kraken different seabeds:
        # at .2f a 1572.348 m/s / 112.607 m/s / 1.4449 g/cm3 halfspace
        # reaches Bellhop as 1572.35 / 112.61 / 1.44, moving |R| at 20 deg
        # grazing by 0.022 dB per bounce.
        f.write(format_halfspace_row(f"{z_max:{DECK_DEPTH_FMT}}", hs))


def _bellhop_run_type(run_type: str, beam_type: str, *,
                      source_beam_pattern: bool, source_type: str,
                      grid_type: str, beam_shift: bool) -> str:
    """The seven-character ``RunType`` record.

    Positions (ReadEnvironmentBell.f90:358-429):

    1. ``run_type`` (C/I/S/A/a/E/R) and 2. ``beam_type`` (B/R/C/S/g/G), both
       case significant: 'A' vs 'a' picks ASCII vs binary arrivals
       (:372-375) and lowercase 'g' the ray-centred geometric-hat beam
       (:387-395), so neither is upper-cased.
    3. ``'*'`` for a source beam pattern: bellhop.f90:137 copies it to
       SBPFlag and misc/beampattern.f90:22 opens <base>.sbp only on '*'.
    4. ``source_type`` (R/X) and 5. ``grid_type`` (R/I), each checked to be
       one of its letters before the file opened.
    6. dimensionality, a blank: see below.
    7. ``'S'`` for beam shift (Beam%Type(4:4), :159), else blank.

    Position 6 is the one character that means plain 2-D to BOTH engines.
    Fortran Bellhop SELECTs on RunType(6:6) (:422-429): '2' prints "N x 2D
    calculation", '3' prints "3D calculation", and the CASE DEFAULT, where a
    blank lands, assigns '2', so ' ' and '2' are the same run there.
    bellhopcxx/bellhopcuda are not indifferent: their 2-D default RunType is
    "CG RR  " with position 6 blank
    (third_party/bellhopcuda/src/module/runtype.hpp:36-41), and a literal
    '2' means Nx2D — runtype.hpp:92-100 warns "Environment file specifies
    dimensionality 2, which usually means Nx2D, but you are running <prog>
    in 2D mode" and rewrites the character to ' '. That warning goes to
    stdout (src/util/errors.cpp:26-36), which uacpy logs only at debug
    level. BELLHOP3D is not available: it would set '3', plug into the
    ``--3D`` _build_command path and change several downstream blocks
    (bearings, 3D bty, beam fan); the file layer it needs ships already
    (``uacpy.io.write_bty_3d`` / ``read_boundary_3d``, ``read_ssp_3d``,
    ``write_field3dflp`` / ``read_flp3d``). Nx2D is the '2'.
    """
    position_3 = '*' if source_beam_pattern else ' '
    position_6 = ' '
    position_7 = 'S' if beam_shift else ' '
    return (f"{run_type}{beam_type}{position_3}"
            f"{source_type}{grid_type}{position_6}{position_7}")


def _write_beam_block(f, beam_type: str, *, n_beams: int, launch_angles_deg, ray_step,
                      z_box, r_box, cerveny, single_beam=None) -> None:
    """The beam count (and, with ``single_beam``, the index of the one beam
    to trace, ``angleMod.f90:32-33``), the launch-angle fan, the step and
    ray box, and the two Cerveny lines when ``beam_type`` is 'C' or 'R'.
    ``cerveny`` holds the Cerveny knobs given; the rest take
    :data:`CERVENY_DEFAULTS`."""
    f.write(f"{int(n_beams)}\n" if single_beam is None
            else f"{int(n_beams)} {int(single_beam)}\n")

    # Launch angles. ``angleMod.f90:58`` READs the whole ``launch_angles_deg`` array;
    # the trailing '/' terminates list-directed input after two values, so
    # alpha(3) keeps the -999.9 sentinel planted at :57 and SubTab
    # (:60, misc/subtabulate.f90:41-45) fills the fan uniformly between
    # alpha(1) and alpha(2). Dropping the '/' would make Bellhop read the
    # step-size and box lines as further launch angles.
    f.write(f"{launch_angles_deg[0]:.6f} {launch_angles_deg[1]:.6f} /\n")

    # Step size (0 for automatic), then the ray box: z in m, r in km
    # (converted back at ReadEnvironmentBell.f90:154). All three are one
    # list-directed READ (:146), which spans records until its list is
    # satisfied — the split across two lines is cosmetic.
    f.write(f"{ray_step:.6f}\n")
    f.write(f"{z_box:.6f} {float(m_to_km(r_box)):.6f}\n")

    # Cerveny beam parameters.  ReadEnvironmentBell.f90 reads the two
    # extra lines only for 'R'/'C'; 'S' (simple Gaussian) shares the
    # no-extra-read case with 'G'/'g'/'^'/'B'.  Test case-insensitively
    # without destroying the original beam_type ('g' is a distinct
    # ray-centered variant).
    if beam_type.upper() not in ('C', 'R'):
        return
    knobs = cerveny_knobs(**cerveny)
    validate_beam_shape(knobs['beam_width_type'], knobs['beam_curvature'],
                        knobs['component'], 'write_bellhop_env_file')
    # Bellhop's RLoop column expects km; uacpy keeps everything in
    # metres at the API surface, so convert here.
    r_loop_km = float(m_to_km(knobs['r_loop']))
    f.write(f"'{knobs['beam_width_type']}{knobs['beam_curvature']}' "
            f"{knobs['eps_multiplier']:.6f} {r_loop_km:.6f}\n")
    # Line 2: n_image, ib_win, component
    f.write(f"{int(knobs['n_image'])} {int(knobs['ib_win'])} "
            f"'{knobs['component']}'\n")


def _source_letters(source: Source, source_type: Optional[str],
                    source_beam_pattern: Optional[bool]):
    """RunType positions 4 and 3 as the Source states them: the letter of
    ``source.source_type`` (:data:`~uacpy.io.oalib_writer.SOURCE_TYPE_CODE`)
    and whether it carries a beam pattern. A value given that contradicts
    the Source is refused naming both, since the deck would describe another
    source than the one handed in."""
    from_source = SOURCE_TYPE_CODE.get(source.source_type, source.source_type)
    if source_type is not None and source_type != from_source:
        raise ConfigurationError(
            f"write_bellhop_env_file(source_type={source_type!r}) "
            f"contradicts the Source (source_type={source.source_type!r}, "
            f"letter {from_source!r}).",
            remediation="Leave source_type unset to write the Source's "
                        "letter, or pass a Source of the type you mean.")
    has_pattern = source.beam_pattern is not None
    if source_beam_pattern is not None and bool(source_beam_pattern) != has_pattern:
        raise ConfigurationError(
            f"write_bellhop_env_file(source_beam_pattern="
            f"{source_beam_pattern!r}) contradicts the Source, which "
            f"{'carries' if has_pattern else 'carries no'} beam pattern.",
            remediation="Leave source_beam_pattern unset to follow the "
                        "Source, or pass a Source with (or without) "
                        "beam_pattern=.")
    return from_source, has_pattern


def write_bellhop_env_file(
    filepath: Union[str, Path],
    env: Environment,
    source: Source,
    receiver: Receiver,
    run_type: str = "C",
    beam_type: Optional[str] = None,
    source_type: Optional[str] = None,
    grid_type: Optional[str] = None,
    interp_ssp: Optional[str] = None,
    interp_bathymetry: Optional[str] = None,
    interp_altimetry: Optional[str] = None,
    source_beam_pattern: Optional[bool] = None,
    beam_shift: Optional[bool] = None,
    n_beams: Optional[int] = None,
    single_beam: Optional[int] = None,
    launch_angles_deg: Optional[tuple] = None,
    ray_step: Optional[float] = None,
    z_box: Optional[float] = None,
    r_box: Optional[float] = None,
    verbose: Union[bool, str] = False,
    backend: Optional[str] = None,
    **kwargs
):
    """
    Write Bellhop environment file (.env)

    This method generates a properly formatted Acoustics Toolbox environment file
    for the Bellhop ray tracing model (any backend).

    Parameters
    ----------
    filepath : Path
        Output file path for .env file
    env : Environment
        Environment definition (SSP, bathymetry, boundaries)
    source : Source
        Source definition (depths, frequencies, source geometry)
    receiver : Receiver
        Receiver definition (depths, ranges)
    run_type : str, optional
        Run type (position 1): 'C'/'I'/'S'/'A'/'a'/'E'/'R'. Default is 'C'.
        Case is *significant*: 'A' is ASCII arrivals, 'a' is binary
        arrivals (ReadEnvironmentBell.f90:372-375).
        - C: Coherent TL
        - I: Incoherent TL
        - S: Semi-coherent TL
        - A: ASCII Arrivals
        - a: Binary Arrivals
        - E: Eigenrays
        - R: Ray trace
    beam_type : str, optional
        Beam type (position 2): 'B'/'R'/'C'/'g'/'G'/'S'. Default is 'G',
        the geometric hat beam that is also Bellhop's own default
        (:data:`~uacpy.core.engine_defaults.BELLHOP_BEAM_TYPE`;
        ReadEnvironmentBell.f90:394-396). Case is significant: lowercase
        'g' is the ray-centered variant (ReadEnvironmentBell.f90:387-395),
        uppercase are Cartesian.
        - B: Geometric Gaussian, Cartesian
        - R: Cerveny Gaussian, ray-centered
        - C: Cerveny Gaussian, Cartesian
        - g: Geometric hat, ray-centered
        - G: Geometric hat, Cartesian
        - S: Simple Gaussian
    source_type : str, optional
        Source type (position 4): 'R' (point), 'X' (line). ``None`` takes
        the letter of ``source.source_type`` (``'point'`` → 'R', ``'line'``
        → 'X'); a letter that contradicts the Source is refused naming
        both.
    grid_type : str, optional
        Grid type (position 5): 'R' (rectilinear), 'I' (irregular). Default is 'R'.
    interp_ssp : str, optional
        SSP connection scheme (drives ``TopOpt(1)``): ``'linear'``,
        ``'pchip'``, ``'spline'``, ``'quad'``, ``'n2linear'``. Default
        ``None`` is automatic: ``'quad'`` (a ``.ssp`` file is written) for a
        range-dependent ``env.ssp``, ``'linear'`` otherwise. An isovelocity
        ``env.ssp`` writes ``'C'`` whatever is asked.
    interp_bathymetry : str, optional
        ``.bty`` interpolation: ``'linear'`` (default) or ``'curvilinear'``.
    interp_altimetry : str, optional
        ``.ati`` interpolation: ``'linear'`` (default) or ``'curvilinear'``.
    source_beam_pattern : bool, optional
        '*' in RunType position 3, so Bellhop reads ``<base>.sbp`` (the
        source beam pattern). ``None`` follows the Source: a
        ``source.beam_pattern`` sets it and is staged as ``<base>.sbp``
        beside the deck (:func:`~uacpy.io.refl_io.stage_source_beam_pattern`);
        none leaves position 3 blank. A value that contradicts the Source is
        refused naming both.
    beam_shift : bool, optional
        When True, sets RunType position 7 to 'S' enabling beam-shift
        on boundary reflections per Beam%Type(4:4)
        (ReadEnvironmentBell.f90:159-166). Default: False (no shift).
    n_beams : int, optional
        Number of beams. ``0`` defers to Bellhop's own estimate
        (``angleMod.f90:38`` tests ``Nalpha == 0`` exactly), which is
        ``MAX(INT(0.3 * Rmax * f / c0), 300)`` raised further by a
        beam-width-versus-depth rule (``angleMod.f90:44-50``); a ray-trace
        run gets 50. Negative values are rejected.
    single_beam : int, optional
        Index (1-based, lowest launch angle first) of the one beam of the
        fan to trace and sum: ``TopOpt(6:6) = 'I'`` and the index after
        ``NBeams`` (``angleMod.f90:32-33``; Bellhop User Guide §6.2,
        "Plotting a single beam"). ``None`` (default) sums every beam. An
        index below 1 or past a positive ``n_beams`` is refused
        (``angleMod.f90:80-83`` stops on it).
    launch_angles_deg : tuple, optional
        Launch angle limits (min, max) in degrees. Default is (-80, 80).
    ray_step : float, optional
        Step size in meters. If 0, uses automatic. Default is 0.
    z_box : float, optional
        Maximum depth for ray box. If None, uses 1.2 * env.depth
        (:data:`BELLHOP_RAY_BOX_FACTOR`).
    r_box : float, optional
        Maximum range for ray box. If None, uses 1.2 * receiver.range_max.
    verbose : bool or str, optional
        Print verbose output. Default is False.
    backend : {'fortran', 'cxx', 'cuda'}, optional
        The Bellhop binary the deck is for, as ``Bellhop(backend=)`` names
        it. For ``'cxx'`` / ``'cuda'`` a copied ``reflection_file=`` table
        keeps the first row of each repeated angle, which bellhopcxx and
        bellhopcuda refuse (``src/module/reflcoef.hpp:135-141``), with an
        ``IOWarning`` naming the rows dropped
        (:func:`~uacpy.io.refl_io.stage_reflection_file`). ``None`` or
        ``'fortran'`` stages every row, which the Fortran binary reads with
        no monotonicity test (``misc/RefCoef.f90:45-55``).
    **kwargs
        Advanced Cerveny beam parameters (for beam_type 'C' or 'R'); one
        absent or ``None`` takes :data:`CERVENY_DEFAULTS`:
        - beam_width_type (str): 'F'/'M'/'W' (default 'F')
        - beam_curvature (str): 'D'/'S'/'Z' (default 'S')
        - eps_multiplier (float): Epsilon multiplier (default 1)
        - r_loop (float): Range (m) for choosing beam width (default 1000)
        - n_image (int): Number of images (default 1)
        - ib_win (int): Beam windowing (default 5)
        - component (str): 'P' pressure (default), 'V' vertical,
          'H' horizontal

    Notes
    -----
    Shared by every Bellhop backend. The ``.env`` is positional; blocks are
    emitted in the order below, each against the Fortran READ that consumes
    it (``Bellhop/ReadEnvironmentBell.f90`` unless another file is named)
    and, for the optional ones, the option letter that gates it:

    1.  title (:46), frequency (:51), NMedia — must be 1 (:54, ERROUT :56)
    2.  ``TopOpt``, 6 chars (ReadTopOpt :243): (1:1) SSP interpolation,
        (2:2) top BC, (3:4) attenuation units, (5:5) altimetry flag,
        (6:6) development options
    3.  bio-layer block (:313-317) — gated by ``TopOpt(4:4)`` 'B' (uacpy
        writes no 'F', whose Francois-Garrison record :308 reads)
    4.  top half-space row (TopBot :474) — gated by ``TopOpt(2:2)``
    5.  ``NPts, Sigma, Depth`` (:73), then the SSP rows
        (``Bellhop/sspMod.f90:903``)
    6.  ``BotOpt, Sigma`` (:93)
    7.  bottom half-space row (TopBot :474) — gated by ``BotOpt(1:1) == 'A'``
    8.  source depths and receiver depths (ReadSzRz), receiver ranges
        (ReadRcvrRanges)
    9.  ``RunType``, 7 chars (ReadRunType :358)
    10. beam count (``Bellhop/angleMod.f90:35``), then the launch-angle
        line (:58)
    11. ``deltas, Box%z, Box%r`` (:146) — ``Box%r`` in km (:154)
    12. two Cerveny beam lines (:197, :217) — gated by ``RunType(2:2)``
        in 'C' / 'R'

    Auxiliary files are written beside the ``.env`` and opened by Bellhop
    under its base name, never by a path in the deck: ``.ati``
    (``TopOpt(5:5) == '~'``), ``.bty`` (``BotOpt(2:2) == '~'``), ``.ssp``
    (``TopOpt(1:1) == 'Q'``), ``.trc`` / ``.brc`` (the BC letter 'F'),
    ``.sbp`` (``RunType(3:3) == '*'``).
    """
    beam_type = BELLHOP_BEAM_TYPE if beam_type is None else beam_type
    n_beams = BELLHOP_N_BEAMS if n_beams is None else n_beams
    launch_angles_deg = (BELLHOP_LAUNCH_ANGLES_DEG if launch_angles_deg is None
                         else launch_angles_deg)
    ray_step = BELLHOP_RAY_STEP if ray_step is None else ray_step
    grid_type = BELLHOP_GRID_TYPE if grid_type is None else grid_type
    interp_bathymetry = BELLHOP_INTERP_BATHYMETRY if interp_bathymetry is None else interp_bathymetry
    interp_altimetry = BELLHOP_INTERP_ALTIMETRY if interp_altimetry is None else interp_altimetry
    beam_shift = BELLHOP_BEAM_SHIFT if beam_shift is None else beam_shift
    filepath = Path(filepath)

    # The only ``**kwargs`` this deck reads are the Cerveny beam knobs
    # consumed by the beam block; anything else is a typo'd option and is
    # rejected instead of silently dropped.
    reject_unknown_kwargs('write_bellhop_env_file', kwargs, CERVENY_DEFAULTS)
    source_type, source_beam_pattern = _source_letters(
        source, source_type, source_beam_pattern)
    n_beams = _check_bellhop_options(run_type, beam_type, source_type,
                                     grid_type, n_beams)
    if single_beam is not None and (
            int(single_beam) != single_beam or single_beam < 1
            or (n_beams and single_beam > n_beams)):
        raise ConfigurationError(
            f"write_bellhop_env_file(single_beam={single_beam!r}) must be a "
            f"beam index in 1..n_beams (n_beams={n_beams}): angleMod.f90:80-83 "
            f"stops with 'Selected beam, iSingle_alpha not in "
            f"[ 1, Angles%Nalpha ]'.")
    z_box, r_box = bellhop_ray_box(env, receiver, z_box=z_box, r_box=r_box)

    # Deck-validity guards that depend only on the environment run BEFORE the
    # file opens, so a refused deck never leaves a truncated .env behind.
    interp_char = _check_bellhop_env(env, interp_ssp)
    bty_code, ati_code = _boundary_interp_codes(interp_bathymetry,
                                                interp_altimetry)
    _check_irregular_grid(receiver, grid_type)
    # bellhopcxx / bellhopcuda test a reflection table's angle column with
    # '<=' (src/module/reflcoef.hpp:135-141, src/common_setup.hpp:243-253);
    # the Fortran binary reads it with no monotonicity test
    # (misc/RefCoef.f90:45-55).
    drop_repeated_angles = backend in ('cxx', 'cuda')
    if source_beam_pattern:
        # Bellhop opens <base>.sbp when RunType(3:3) is '*'.
        stage_source_beam_pattern(source.beam_pattern,
                                  filepath.with_suffix('.sbp'))

    with open(filepath, "w") as f:
        has_altimetry = _write_top_block(
            f, filepath, env, source, receiver, interp_char=interp_char,
            ati_code=ati_code, verbose=verbose,
            drop_repeated_angles=drop_repeated_angles,
            single_beam=single_beam is not None)
        ssp_depths, ssp_matrix, z_max = _bellhop_ssp(
            env, has_altimetry=has_altimetry, interp_char=interp_char)
        if interp_char == 'Q':
            _write_quad_ssp_beside(filepath, env, ssp_matrix, r_box=r_box,
                                   receiver=receiver, verbose=verbose)
            _warn_if_quad_rows_hold_range0_absorption(env, ssp_matrix)
        _write_ssp_block(f, env, ssp_depths, ssp_matrix, z_max,
                         frequency=float(source.frequencies[0]))
        _write_bottom_block(f, filepath, env, receiver, z_max=z_max,
                            bty_code=bty_code, verbose=verbose,
                            drop_repeated_angles=drop_repeated_angles)
        write_source_depths(f, source)
        write_receiver_depths(f, receiver)
        write_receiver_ranges(f, receiver)
        run_type_record = _bellhop_run_type(
            run_type, beam_type, source_beam_pattern=source_beam_pattern,
            source_type=source_type, grid_type=grid_type,
            beam_shift=beam_shift)
        f.write(f"'{run_type_record}'\n")
        _write_beam_block(f, beam_type, n_beams=n_beams, launch_angles_deg=launch_angles_deg,
                          ray_step=ray_step, z_box=z_box, r_box=r_box,
                          cerveny=kwargs, single_beam=single_beam)
