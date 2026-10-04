"""
Acoustics Toolbox / OALIB environment-file writers.

Each function writes one logical block of the AT ``.env`` format onto an
open text handle, plus the ``.flp`` field-parameter writer used by
Kraken. ``write_multi_profile_env`` and ``write_fieldflp`` write the
full file.

``write_field3dflp`` writes the FIELD3D deck. **It is deliberately retained
and is not dead code**: no uacpy model runs ``field3d`` yet, so nothing in
the 2-D public API calls it, but it is the deck writer a future 3-D
implementer builds on — with :func:`~uacpy.io.oalib_reader.read_flp3d`,
:func:`~uacpy.io.oalib_reader.read_ssp_3d`,
:func:`~uacpy.io.bathy_io.read_boundary_3d` and
:func:`~uacpy.io.bathy_io.write_bty_3d`.
``uacpy/tests/test_io_public_names_without_callers.py`` pins the five against a
dead-code sweep proposing their removal a second time.

Top-block record order (the one contract every AT ``.env`` here obeys)
---------------------------------------------------------------------
``misc/ReadEnvironmentMod.f90`` reads the records after NMedia in this
order, and a deck that emits them in any other order feeds the wrong row
to the wrong ``READ``:

1. the TopOpt line (``ReadEnvironment:68`` → ``ReadTopOpt``);
2. the volume-attenuation rows *inside* ``ReadTopOpt`` — the
   Francois-Garrison ``T S pH z_bar`` row for ``TopOpt(4)='F'``
   (``:215``; written only for a deck covering several frequencies) or the
   bio-layer count + rows for ``'B'`` (``:220-235``);
3. only then the top half-space row ``z cP cS rho alphaI betaI`` for
   ``TopOpt(2)='A'`` (``:75`` → ``TopBot`` ``:285``).

:func:`write_header` owns all three, plus staging the ``.trc`` table for
``TopOpt(2)='F'`` (``misc/RefCoef.f90:64-76`` opens ``<root>.trc``). Its
callers must not emit anything of their own between the TopOpt line and
the SSP mesh.

Adoption across uacpy model wrappers:

- ``write_header``: Kraken, Scooter, Bounce.
  SPARC writes its own title/freq/NMedia line (`SPARC` has a 5th TopOpt
  position for ``output_mode``); Bellhop has its own writer entirely.
- ``write_bottom_section``: Kraken, Scooter, Bounce.
  SPARC open-codes the bottom block because it accepts only a vacuum or
  rigid seabed.
- ``write_source_depths`` / ``write_receiver_depths`` /
  ``write_receiver_ranges``: every AT-family wrapper, including Bellhop.
- ``write_absorption_block`` (calls ``write_fg_params`` / ``write_bio_layers``):
  emitted by ``write_header`` for the AT-family writers here; Bellhop and
  SPARC call it directly from their own header code. Drives output from
  ``env.absorption``.
- ``get_top_bc_code`` / ``write_surface_halfspace``: all AT-family
  wrappers including Bellhop.
"""

import warnings

import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple, Union

from uacpy.core.absorption import (
    BAND_ABSORPTION_CHECK_DEPTHS, BAND_ABSORPTION_WARN_DB_PER_KM,
    Biological, ConstantAbsorption, warn_if_band_absorption_frozen,
)
from uacpy.core.environment import Environment
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.boundary import BoundaryProperties, BoundaryType
from uacpy.core.surface import Surface
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.core.deck_limits import (
    DECK_DEPTH_FMT, DECK_DEPTH_RESOLUTION_M, DECK_RANGE_RESOLUTION_M,
)
from uacpy.io.at_codes import (
    BOUNDARY_CODES, SSP_INTERP_CODES, AttenuationUnits,
    biological_records, boundary_code, francois_garrison_record,
    parse_boundary_type, volume_attenuation_code, writes_alpha_per_ssp_row,
    writes_francois_garrison_letter,
)
from uacpy.core._validate import equally_spaced
from uacpy.io.input_checks import (
    _collapsed_pair_index,
    reject_unknown_kwargs,
)
from uacpy.core.units import m_to_km
from uacpy.io.refl_io import stage_reflection_file
from uacpy.core._validate import sanitize_title
from uacpy.io._fortran_helpers import deck_title
from uacpy.core.exceptions import (
    ConfigurationError, FallbackWarning, UnsupportedFeatureError,
)
from uacpy.core.engine_defaults import (
    KRAKEN_N_MESH,
    SCOOTER_N_MESH,
    SPARC_N_MESH,
    SPARC_OUTPUT_MODE,
)


#: AT source-geometry letters keyed by uacpy ``source.source_type``, written
#: into the field/Hankel option string (Opt(1:1) of ``fieldsco.m:120-140`` /
#: ``field.flp``): ``'R'`` point source with cylindrical spreading, ``'X'``
#: line source with Cartesian spreading, ``'S'`` scaled point source with the
#: cylindrical spreading removed. One table for every AT-family wrapper; each
#: model's ``spec.source_types`` restricts which keys its ``validate_inputs``
#: lets through.
SOURCE_TYPE_CODE = {'point': 'R', 'line': 'X', 'scaled': 'S'}

#: Decimals in every depth column these decks write, and the resolution that
#: implies. The Acoustics-Toolbox manual states the format requirement outright —
#: *"All user input in all modules is read using list-directed I/O. Thus data can
#: be typed in free-format"* (``doc/index.htm``) — and the readers bear that out:
#: ``misc/ReadEnvironmentMod.f90:88`` and ``misc/sspMod.f90:334`` are both
#: ``READ( ENVFile, * )``. So no column width is imposed on uacpy; what the decks
#: DO require is that a depth uacpy compares in Python is the depth the Fortran
#: parses, which means one format for every depth written. Six decimals sits two
#: orders below the tightest tolerance any reader applies
#: (``sspMod.f90:353``'s ``100 * EPSILON( 1.0e0 )`` = 1.19e-05 m).
#: The format is ``core.deck_limits.DECK_DEPTH_FMT``, not restated: the
#: carriers admit a step down to ``DECK_DEPTH_RESOLUTION_M`` on the strength of
#: this column format, so a deck printed coarser than the carriers admit would
#: collapse two admitted samples onto one token.

#: Thickness of the transparent pad media that equalise ``NMedia`` across the
#: profiles of a range-dependent deck. This is uacpy's own construct, not
#: anything AT prescribes: the pad repeats the half-space it sits in, so it is
#: acoustically inert, and it only has to be thick enough to mesh
#: (``misc/sspMod.f90:356-358`` rejects a medium with fewer than 2 SSP points).
_PAD_MEDIUM_THICKNESS_M = 0.1


# misc/AttenMod.f90:10,18 — ``MaxBioLayers = 200`` sizes the static
# ``bio( MaxBioLayers )`` array shared by every AT program.
# misc/ReadEnvironmentMod.f90:222-225 bounds the count before filling it;
# Bellhop/ReadEnvironmentBell.f90:316-317 loops straight to NBioLayers, so a
# longer block walks off the array.
_MAX_BIO_LAYERS = 200

# Compiled media bound shared by kraken/krakenc/scooter/sparc:
# misc/ReadEnvironmentMod.f90:63-66 ERROUTs 'Too many Media' when the deck's
# NMedia exceeds MaxMedium — a bare Fortran fatal, so the writers refuse
# first.  MaxMedium is a dummy argument of that subroutine
# (ReadEnvironmentMod.f90:12,21); its value 500 is a PARAMETER declared
# separately by each calling program — Kraken/KrakenMod.f90:6,
# Kraken/KrakencMod.f90:6, Scooter/scooter.f90:32, Scooter/sparc.f90:31 —
# and they all agree on 500.
# NOT misc/sspMod.f90:11: that line declares ``MaxSSP = 20001,
# MaxMedia = 501``, a different name that nothing tests here.  (Upstream
# knows: ReadEnvironmentMod.f90:16 is a FIXME saying "MaxMedium and
# MaxMedia are essentially the same thing".)  sspMod.f90:11 IS the right
# address for the other bound taken from that line, MaxSSP = 20001, which
# ``write_ssp_section`` enforces.
_AT_MAX_MEDIA = 500


def _reject_media_overrun(n_media) -> None:
    """Raise before writing an NMedia the AT binaries cannot compile in."""
    if int(n_media) <= _AT_MAX_MEDIA:
        return
    raise ConfigurationError(
        f"AT deck: NMedia = {int(n_media)} (water column + sediment layers) "
        f"exceeds the compiled bound MaxMedium = {_AT_MAX_MEDIA} "
        f"(Acoustics-Toolbox misc/ReadEnvironmentMod.f90:63-66 ERROUTs "
        f"'Too many Media'; the 500 is each program's own PARAMETER, "
        f"e.g. Kraken/KrakenMod.f90:6).",
        remediation="Collapse the layered seabed below "
                    f"{_AT_MAX_MEDIA} media, e.g. merge thin layers.",
    )


def quote_fortran_title(name) -> str:
    """``name`` as the quoted deck-title literal, re-sanitized.

    ``Environment`` already strips apostrophes and control characters from
    names at construction (``_sanitize_title``: the Fortran ``''`` escape
    is rejected by bellhopcxx's title parser, so apostrophes are removed,
    not doubled). Re-running the same sanitizer here closes the
    post-construction ``env.name = ...`` mutation path, which otherwise
    writes a quote that silently truncates the Fortran list-directed READ.
    The result is ASCII (:func:`~uacpy.io._fortran_helpers.deck_title`),
    because the engines cut titles in bytes.
    """
    return "'" + deck_title(sanitize_title(name)) + "'"



def deck_depth(depth_m: float) -> float:
    """A depth as the deck writes it — the value the Fortran will parse back.

    The single point of truth for every Acoustics-Toolbox depth uacpy writes:
    the water-column mesh bottom, each sediment interface, the half-space top,
    and Bellhop's SSP header depth (``bellhop_writer``). Routing them all through
    one function is what keeps a fractional ``env.depth`` from putting two models
    on different water columns.

    It round-trips through :data:`DECK_DEPTH_FMT` rather than snapping to a grid.
    The three invariants the decks actually impose are all satisfied by writing
    the *same* number everywhere, not by coarsening it:

    * ``misc/sspMod.f90:353`` ends a medium when its SSP sample matches the mesh
      line's ``Depth`` to within ``100 * EPSILON( 1.0e0 )`` = 1.19e-05 m. Writing
      one value in both places makes that difference exactly zero.
    * ``misc/SourceReceiverPositions.f90:126-139`` moves a source or receiver
      **strictly** below ``zMax`` up onto it. A receiver at exactly ``env.depth``
      is not below a mesh bottom written from the same ``env.depth``.
    * ``Bellhop/bdryMod.f90:211`` aborts when a ``.bty`` point is **strictly**
      deeper than the SSP's last depth. The ``.bty`` is written at the same
      resolution (``io/bathy_io.py``), so a seafloor point equal to ``env.depth``
      is safe.

    Six decimals is two orders finer than the tightest of those tolerances, so
    quantising onto a 0.1 m grid — which rounds every interface up, and costs
    an 11 dB water-depth error for any bathymetry not already on a decimetre —
    is not needed to satisfy any of them.
    """
    return float(f"{float(depth_m):{DECK_DEPTH_FMT}}")


def writable_layers(bottom):
    """Sediment layers of a range-independent ``Bottom`` or a ``SeabedColumn``.

    Every layer is written at its own thickness. Dropping a sub-decimetre layer
    cost 0.212 in |R| — 4.7 dB per bottom bounce — against an exact plane-wave
    impedance recursion on a 0.08 m layer at 5 kHz, and nothing in the readers
    asks for it: ``misc/ReadEnvironmentMod.f90:88`` and ``misc/sspMod.f90:334``
    are list-directed, as the manual states for all AT input
    (``doc/index.htm``). ``SedimentLayer`` refuses a non-positive thickness at
    construction, so there is no degenerate case left to guard here either.
    """
    col = bottom.columns[0] if hasattr(bottom, 'columns') else bottom
    return list(col.layers)


def at_mesh_floor(media, frequency: float) -> int:
    """Smallest ``NG`` the coarsest of ``media`` will be accepted at.

    ``misc/ReadEnvironmentMod.f90:101-112`` sizes **each medium separately** at
    ``deltaz = c / freq0 / 20`` — with ``c`` that medium's shear speed wherever
    ``betaR > 0``, else its last compressional sample — takes
    ``Nneeded = MAX( INT( thickness / deltaz ), 10 )`` from its own
    ``SSP%Depth( m+1 ) - SSP%Depth( m )``, and stops with *Mesh is too coarse*
    whenever the deck asks for ``NG < Nneeded / 2`` (Fortran integer division).
    ``NG = 0`` means auto and is never checked, so this bound only constrains a
    pinned count.

    Per medium is the whole point: lumping a stack into one span at the slowest
    speed in it overstates ``Nneeded`` and rejects decks the reader accepts.

    ``media`` is an iterable of ``(thickness_m, speed_m_s)``.
    """
    needed = [max(int(20.0 * thickness * frequency / c), 10)
              for thickness, c in media]
    # ``NG >= Nneeded / 2`` passes, so the floor is the truncated half of the
    # most demanding medium. With no media at all the ``MAX( …, 10 )`` inside
    # Nneeded still puts the smallest floor AT can ever impose at 10 // 2.
    return max(needed) // 2 if needed else 5


def at_env_media(env):
    """``(thickness, speed)`` per medium of a range-independent AT deck."""
    seafloor = deck_depth(env.depth)
    # ``write_ssp_section`` anchors medium 1 at z = 0 and at ``seafloor``;
    # ``c`` is its last sample, the one ``alphaR`` holds after EvaluateSSP.
    water = env.ssp.extend_to(seafloor).to_pairs()
    media = [(seafloor, float(water[-1, 1]))]
    if env.bottom.is_layered and not env.bottom.is_range_dependent:
        top = seafloor
        for layer in writable_layers(env.bottom):
            bot = deck_depth(top + layer.thickness)
            shear = float(layer.shear_speed)
            media.append((bot - top,
                          shear if shear > 0.0 else float(layer.sound_speed)))
            top = bot
    return media


def reject_unsupported_ssp_interp(model: str, interp_ssp) -> None:
    """Reject an SSP interpolation the shared AT ``EvaluateSSP`` cannot read.

    ``misc/sspMod.f90:61-89`` accepts codes A (analytic), N (N^2-linear),
    C (C-linear), P (PCHIP) and S (spline); anything else falls to
    ``CASE DEFAULT`` and stops with ``ERROUT('EvaluateSSP', 'Unknown profile
    option')``. ``'Q'`` is Bellhop's external 2-D ``.ssp`` scheme, so every
    model that reads its deck through ``EvaluateSSP`` has to refuse it here —
    otherwise it surfaces as a bare Fortran fatal with no diagnostic.
    """
    if interp_ssp is None:
        return
    if str(interp_ssp).lower() in ('q', 'quad', 'quadratic'):
        # A model-capability refusal, not a bad argument: 'quad' is a real
        # uacpy interpolation that Bellhop honours, so the caller's recourse
        # is another interpolation or another model — the choice
        # ``except UnsupportedFeatureError`` exists to make.
        raise UnsupportedFeatureError(
            model,
            "the 'quad' SSP interpolation — it is Bellhop-only, the "
            "external 2-D .ssp scheme the shared EvaluateSSP has no case for",
            alternatives=["'linear' (C-linear)", "'n2linear'", "'pchip'",
                          "'spline'"],
            alternatives_label='SSP interpolations',
        )


#: Spacing (m) of the SSP node pair written across each edge of a
#: :class:`~uacpy.core.absorption.Biological` layer. AT evaluates the ``'B'``
#: law only at SSP nodes — ``CRCI`` tests the node depth against the layer
#: (``misc/AttenMod.f90:103-104``), called per node at
#: ``misc/sspMod.f90:388-393`` and ``Bellhop/sspMod.f90:906`` — and
#: interpolates ``Im c`` between them, so a layer holding no node is lossless
#: and a layer with one node per edge bleeds into the neighbouring intervals.
#: A node on the edge and one ``_BIOLOGICAL_EDGE_PAIR_M`` outside it confine
#: the ramp to that spacing. Measured on a 30-70 m layer (2 dB/km peak at
#: 300 Hz) over 10 km, Kraken BROADBAND: 0.01 m and 0.1 m give identical
#: absorption losses, which fall short of RAM's (RAM samples the edges
#: itself) by at most 0.22 dB, at the 300 Hz peak (8.777 against 8.999 dB);
#: 1 m adds 1-2 % to the loss.
_BIOLOGICAL_EDGE_PAIR_M = 0.01

#: SSP interpolations that build their curve from neighbouring nodes. Both
#: interpolate the complex ``c`` (``misc/sspMod.f90:397-415``,
#: ``Bellhop/sspMod.f90:281``, ``:339-344``), so an ``Im c`` step across a
#: layer edge rings under the spline (measured: NaN, or tens to hundreds of dB
#: of gain), and an inserted node reshapes the real PCHIP curve (measured: up
#: to 0.5 dB of range-averaged level on a curved profile).
_NEIGHBOUR_INTERP_CODES = frozenset({'P', 'S'})


def biological_interior_edges(absorption, z_min: float,
                              z_max: float) -> List[float]:
    """Depths of every :class:`~uacpy.core.absorption.Biological` layer edge
    strictly inside ``(z_min, z_max)``, sorted; empty for any other law."""
    if not isinstance(absorption, Biological):
        return []
    edges = {float(z) for layer in biological_records(absorption)
             for z in layer[:2]}
    return sorted(z for z in edges if z_min < z < z_max)


def reject_biological_edges_under_neighbour_interp(
        model: str, env: Environment, interp_ssp) -> None:
    """Refuse a Biological layer edge inside the water column when the
    model's ``interp_ssp`` resolves to a PCHIP or spline profile.

    The edge needs its own node pair (:func:`biological_edge_nodes`), which
    only a piecewise-linear interpolation takes without changing the profile
    or ringing on the ``Im c`` step (see :data:`_NEIGHBOUR_INTERP_CODES`).
    ``interp_ssp`` is resolved (:func:`resolve_ssp_topopt`) only when there
    is an interior edge, so every other environment is untouched here.
    """
    edges = biological_interior_edges(env.absorption, 0.0, float(env.depth))
    if not edges:
        return
    ssp_code = resolve_ssp_topopt(env, interp_ssp)
    if ssp_code not in _NEIGHBOUR_INTERP_CODES:
        return
    name = 'pchip' if ssp_code == 'P' else 'spline'
    raise ConfigurationError(
        f"{model}: env.absorption is Biological with layer edges at "
        f"{', '.join(f'{z:g}' for z in edges)} m inside the water column, "
        f"and interp_ssp resolves to {name!r}. The Acoustics Toolbox "
        f"evaluates the biological law only at SSP nodes "
        f"(misc/AttenMod.f90:103-104), so each edge needs a node pair; a "
        f"{name} interpolation of the complex sound speed "
        f"{'rings on that step into unphysical gain' if ssp_code == 'S' else 'is reshaped by the inserted nodes'}.",
        remediation="Pass interp_ssp='linear' (or 'n2linear'), which takes "
                    "the node pairs exactly.")


def biological_edge_nodes(depths, sound_speed, absorption,
                          ssp_code: str, *, who: str):
    """``depths`` and ``sound_speed`` with a node pair across every
    Biological layer edge strictly inside the profile.

    For a layer top ``z1`` the pair is ``(z1 - ε, z1)``; for a layer bottom
    ``z2`` it is ``(z2, z2 + ε)``, with ε = :data:`_BIOLOGICAL_EDGE_PAIR_M`:
    ``'B'`` counts both edges as inside (``misc/AttenMod.f90:104``), so the
    node on the edge carries the layer and the other one the water beside it.
    Nodes already present are kept as they are. ``sound_speed`` is 1-D, or
    ``(n_depths, n_ranges)`` for Bellhop's quad profile; each added row is
    interpolated the way the deck interpolates — linear in ``c`` for ``'C'``
    and ``'Q'`` (``Bellhop/sspMod.f90`` ``Quad``: linear in depth), linear
    in ``1/c²`` for ``'N'``. A PCHIP or spline deck with an interior edge is
    refused (:func:`reject_biological_edges_under_neighbour_interp`).
    Returns the arrays unchanged when there is no interior edge.
    """
    z = np.asarray(depths, dtype=float)
    c = np.asarray(sound_speed, dtype=float)
    edges = biological_interior_edges(absorption, float(z[0]), float(z[-1]))
    if not edges:
        return z, c
    if ssp_code in _NEIGHBOUR_INTERP_CODES:
        raise ConfigurationError(
            f"{who}: a Biological layer edge at "
            f"{', '.join(f'{e:g}' for e in edges)} m lies inside the profile "
            f"and the SSP interpolation is {ssp_code!r}; only a "
            f"piecewise-linear profile takes the node pairs the biological "
            f"law needs (see reject_biological_edges_under_neighbour_interp).",
            remediation="Pass interp_ssp='linear' (or 'n2linear').")
    tops = {float(layer[0]) for layer in biological_records(absorption)}
    added = []
    for e in edges:
        # An edge can be the top of one layer and the bottom of another:
        # then both sides are inside, and the node on the edge suffices.
        is_top = e in tops
        is_bottom = any(float(layer[1]) == e
                        for layer in biological_records(absorption))
        added.append(e)
        if is_top and not is_bottom:
            added.append(e - _BIOLOGICAL_EDGE_PAIR_M)
        elif is_bottom and not is_top:
            added.append(e + _BIOLOGICAL_EDGE_PAIR_M)
    new = np.array([a for a in added
                    if z[0] < a < z[-1]
                    and np.min(np.abs(z - a)) > DECK_DEPTH_RESOLUTION_M])
    if new.size == 0:
        return z, c
    columns = c.reshape(z.size, -1)
    if ssp_code == 'N':
        rows = np.column_stack([
            1.0 / np.sqrt(np.interp(new, z, 1.0 / col ** 2))
            for col in columns.T])
    else:
        rows = np.column_stack([np.interp(new, z, col) for col in columns.T])
    order = np.argsort(np.concatenate([z, new]), kind='stable')
    z_out = np.concatenate([z, new])[order]
    c_out = np.vstack([columns, rows])[order]
    return z_out, (c_out if c.ndim == 2 else c_out[:, 0])


#: ``misc/sspMod.f90:11`` declares ``MaxSSP = 20001`` and dimensions every
#: profile array to it. The read loop at ``:331-332`` counts *per medium* but
#: writes at a *cumulative* index (``SSP%Loc`` accumulates at ``:325``), so the
#: clean ``ERROUT`` at ``:368`` only fires for a single medium — with sediment
#: layers the index runs off the end of the array and gfortran dies inside the
#: READ instead. Bellhop's private ``Bellhop/sspMod.f90:16`` sets 100001, so
#: the same environment can be fine for Bellhop and fatal for Kraken.
_AT_MAX_SSP_POINTS = 20001


def reject_oversized_at_ssp(model: str, n_points: int) -> None:
    """Reject an SSP with more rows than the shared AT reader can hold."""
    if n_points <= _AT_MAX_SSP_POINTS:
        return
    raise ConfigurationError(
        f"{model}: the environment writes {n_points} SSP rows across all "
        f"media, over the {_AT_MAX_SSP_POINTS} that misc/sspMod.f90:11 "
        f"dimensions its profile arrays to. The run would stop inside the "
        f"reader — cleanly for a single medium, and with an unrelated "
        f"'Bad real number in item 1 of list input' once sediment layers "
        f"push the cumulative index past the end.",
        remediation="Thin the sound-speed profile (AT re-meshes it anyway — "
                    "the mesh line, not the tabulated point count, sets the "
                    "solver's resolution), or use Bellhop, whose own reader "
                    "holds 100001.")


def reject_coarse_at_mesh(model: str, n_mesh: int, env,
                          frequency: float) -> None:
    """Reject a pinned ``n_mesh`` the shared AT env reader will refuse.

    KRAKEN and SCOOTER both read their deck through
    ``misc/ReadEnvironmentMod.f90``, so both hit the same *Mesh is too coarse*
    stop — a bare Fortran fatal unless it is caught here.
    """
    if n_mesh <= 0:
        return
    floor = at_mesh_floor(at_env_media(env), frequency)
    if n_mesh < floor:
        raise ConfigurationError(
            f"{model}(n_mesh={n_mesh}) is below the {floor} mesh points "
            f"misc/ReadEnvironmentMod.f90:110-112 requires for the coarsest "
            f"medium of this environment at {frequency:.4g} Hz; the run would "
            f"stop with 'Mesh is too coarse'.",
            remediation=f"Pass n_mesh >= {floor}, or n_mesh=0 to let the "
                        f"model size each medium itself.",
        )


def _profile_n_media(env_seg) -> int:
    """AT media a single profile carries naturally: water plus its sediment
    layers."""
    n = 1
    if env_seg.bottom.is_layered and not env_seg.bottom.is_range_dependent:
        n += len(writable_layers(env_seg.bottom))
    return n


def _profile_media(env_seg) -> List[Tuple]:
    """Media 2..N of one profile: its sediment layers, nothing invented.

    Each entry is ``(top, bot, cp, cs, rho, alpha_p, alpha_s, sigma)`` with
    both interfaces already quantised to the deck's depth resolution
    (:func:`deck_depth`), so the depths compared in Python are the depths the
    Fortran parses back.
    """
    current = deck_depth(env_seg.depth)
    media: List[Tuple] = []
    if env_seg.bottom.is_layered and not env_seg.bottom.is_range_dependent:
        for layer in writable_layers(env_seg.bottom):
            top = current
            current = deck_depth(current + layer.thickness)
            media.append((top, current, layer.sound_speed,
                          layer.shear_speed, layer.density,
                          layer.attenuation,
                          layer.shear_attenuation,
                          layer.roughness))
    return media


def _plan_unpadded_media(segments, acoustic_types
                         ) -> Tuple[int, float, List[List[Tuple]]]:
    """Media plan for profiles whose half-space carries no material.

    ``vacuum``, ``rigid`` and the two reflection-table types
    (:attr:`~uacpy.core.boundary.BoundaryType.is_geoacoustic` false) are boundary
    conditions, not media: the ``sound_speed`` / ``density`` /
    ``attenuation`` a parameter-free ``BoundaryProperties`` carries are the
    constructor's placeholders, which is why the engines' default
    phase-speed window (``uacpy.models._window``) does not cap cHigh on them
    either.
    A pad medium built from those placeholders would put metres of invented
    sediment between the water and a boundary the user asked to be
    pressure-release, so no pad is emitted here.

    Without a pad there is nothing to stretch onto a common bottom, so the
    profiles have to already agree on both their media count and their total
    depth — the ``z( NR ) == depthB`` equality ``EvaluateCMMod.f90:313``
    enforces per profile against one shared mode-tabulation grid. A
    range-dependent bathymetry over a boundary-condition seabed cannot
    satisfy it in any deck, so it is refused rather than approximated.
    """
    plans = [_profile_media(env_seg) for _range_m, env_seg in segments]
    bottoms = {(media[-1][1] if media else deck_depth(env_seg.depth))
               for media, (_range_m, env_seg) in zip(plans, segments)}
    counts = {len(media) for media in plans}

    if len(bottoms) > 1 or len(counts) > 1:
        types = ', '.join(repr(t) for t in acoustic_types)
        raise ConfigurationError(
            f"range-dependent KRAKEN deck: the profiles end at "
            f"{sorted(bottoms)} m with {sorted(counts)} sub-bottom "
            f"medium/media, and the half-space acoustic_type is {types}. "
            f"Profiles of unequal geometry are normally equalised with pad "
            f"media repeating the half-space, but a boundary-condition "
            f"seabed carries no material to repeat — the pads would be "
            f"invented sediment between the water and the boundary.",
            remediation="Give every profile the same total depth and the "
                        "same sediment-layer count, or model the seabed as "
                        "acoustic_type='half-space' / 'acousto-elastic' so "
                        "the profiles can be padded with real material.",
        )

    return len(plans[0]) + 1, bottoms.pop(), plans


def plan_multi_profile_media(segments) -> Tuple[int, float, List[List[Tuple]]]:
    """Lay out the sub-bottom media of a multi-profile KRAKEN ``.env``.

    Returns ``(n_media, bottom_depth_m, plans)``. Every profile is written with
    the same ``n_media`` and the same ``bottom_depth_m``; ``plans[i]`` holds
    media 2..``n_media`` of ``segments[i]`` as
    ``(top, bot, cp, cs, rho, alpha_p, alpha_s, sigma)``, already quantised to
    the ``.6f`` depth resolution the deck is written at (:func:`deck_depth`),
    with the last entry stretched to ``bottom_depth_m``. A profile whose
    sediment stack already reaches the common bottom gets no extra medium, so
    ``plans[i]`` is empty when the deck needs none.

    This is the single source of truth for the deck's geometry. ``.env``
    writing takes ``plans`` verbatim, and the mode-tabulation grid must span
    ``[0, bottom_depth_m]`` — ``EvaluateCMMod.f90:313`` rejects a coupled run
    unless ``z( 1 ) == depthT`` and ``z( NR ) == depthB`` **exactly**, where
    ``z`` is the merged source/receiver depth vector the deck asks for
    (``kraken.f90:573,598``) and ``depthB`` is ``SSP%Depth( NMedia + 1 )``,
    this function's ``bottom_depth_m``. Recomputing either side independently
    is what broke coupled modes: an 0.1 m disagreement is a fatal stop, not a
    rounding nuisance. AT's own coupled deck holds the same invariant —
    ``tests/wedge/wedge.env`` gives all 51 profiles ``NMedia=2``, a common
    total depth of 2000 m, and ``NRz`` spanning ``0.0 2000.0``.

    Over a **geoacoustic** half-space one medium beyond the deepest layer
    stack is reserved. The stretch onto the common bottom must land on a
    transparent pad carrying the halfspace properties — never on a real
    ``SedimentLayer``, whose thickness is physical. Reserving it also
    satisfies AT multi-profile kraken's ``NMedia >= 2`` for range-dependent
    environments. Over a boundary-condition seabed there are no halfspace
    properties to repeat, so :func:`_plan_unpadded_media` takes over.
    """
    non_geoacoustic = sorted(
        {kind for kind in {env_seg.bottom.halfspace_at(range=0.0).acoustic_type
                           for _range_m, env_seg in segments}
         if not BoundaryType.from_string(kind).is_geoacoustic}
    )
    if non_geoacoustic:
        return _plan_unpadded_media(segments, non_geoacoustic)

    n_media = max(_profile_n_media(seg) for _, seg in segments) + 1

    plans = []
    for _range_m, env_seg in segments:
        hs = env_seg.bottom.halfspace_at(range=0.0)
        media = _profile_media(env_seg)
        current = media[-1][1] if media else deck_depth(env_seg.depth)

        hs_cs = hs.shear_speed
        hs_as = hs.shear_attenuation
        for _ in range(n_media - 1 - len(media)):
            top = current
            current = deck_depth(current + _PAD_MEDIUM_THICKNESS_M)
            # A pad is a transparent slice of the half-space, so its top is
            # not a real interface and must stay smooth.
            media.append((top, current, hs.sound_speed, hs_cs, hs.density,
                          hs.attenuation, hs_as, 0.0))
        plans.append(media)

    bottom_depth = max(media[-1][1] for media in plans)
    for media in plans:
        last = media[-1]
        if last[1] < bottom_depth:
            media[-1] = (last[0], bottom_depth) + last[2:]

    return n_media, bottom_depth, plans


#: Every ``interp_ssp`` name the writers accept -> ``TopOpt(1:1)`` letter:
#: :data:`~uacpy.io.at_codes.SSP_INTERP_CODES`.
_AT_INTERP_TO_CODE = dict(SSP_INTERP_CODES)


def resolve_ssp_interp(env: Environment, model_interp) -> str:
    """Return the user-facing ``interp_ssp`` value after auto-resolution.

    ``None`` means *auto*: pick ``'quad'`` when the env has a
    range-dependent SSP (matching Bellhop's ``.ssp`` quad-file path),
    otherwise ``'linear'``. Explicit values pass through unchanged.
    """
    if model_interp is None:
        return 'quad' if env.ssp.is_range_dependent else 'linear'
    return str(model_interp).lower()


def resolve_ssp_topopt(env: Environment, model_interp) -> str:
    """Pick the AT ``TopOpt(1)`` character for an env / model pair.

    The model's ``interp_ssp`` (``None`` → auto / ``'linear'`` /
    ``'pchip'`` / ``'spline'`` / ``'quad'`` / ``'n2linear'`` /
    ``'analytic'`` / …) drives the character via :data:`_AT_INTERP_TO_CODE`.
    The only env-side override is ``kind='isovelocity'`` which forces
    ``'C'`` (any connection scheme over constant data is constant). All
    other kind values (``'munk'``, ``'analytic'``, ``'n2linear'``,
    ``'measured'``) are informational — the model decides how to connect
    the samples.
    """
    key = resolve_ssp_interp(env, model_interp)
    if key == 'analytic':
        raise ConfigurationError(
            "interp_ssp='analytic' selects the Acoustics-Toolbox 'A' profile, "
            "which is a hard-coded Munk curve on a fixed 5000 m grid "
            "(misc/munk.f90) — it ignores env.ssp entirely, so the run would "
            "not model the environment you supplied. Pass the Munk profile as "
            "data via SoundSpeedProfile if you want it, and pick an "
            "interpolation of 'linear', 'n2linear', 'pchip' or 'spline'."
        )
    if key not in _AT_INTERP_TO_CODE:
        raise ConfigurationError(
            f"interp_ssp={model_interp!r} not recognised. Valid: "
            f"{sorted(set(_AT_INTERP_TO_CODE))} (or None for auto)"
        )
    # Checked after the model's knob so an isovelocity env cannot swallow an
    # invalid ``interp_ssp``.
    if env.ssp.kind == 'isovelocity':
        return 'C'
    return _AT_INTERP_TO_CODE[key]


def get_top_bc_code(env: Environment) -> str:
    """Return the single-character AT top boundary condition code.

    An ``acoustic_type`` no :class:`~uacpy.core.boundary.BoundaryType`
    covers raises :class:`ConfigurationError` — silently falling back to a
    vacuum would model a different surface than the one asked for.
    """
    return boundary_code(env.surface.acoustic_type)


def compose_topopt(ssp_code: str, surface_code: str, env: Environment, *,
                   pos5: str = ' ', pos6: str = ' ', extra: str = '',
                   multi_frequency: bool = False) -> str:
    """The Acoustics-Toolbox ``TopOpt`` string of a deck.

    Position 1 is the SSP interpolation letter and 2 the top boundary
    letter; 3:4 are the attenuation pair ``TopOpt(3:4)``
    (``misc/ReadEnvironmentMod.f90:167``): ``'W'`` (dB/wavelength, uacpy's
    unit for every attenuation field) and the volume-attenuation letter of
    ``env.absorption`` (blank for none). Positions 5 and 6 differ by program
    and are the caller's (``pos5``, ``pos6``); ``extra`` follows them.
    ``multi_frequency`` says the deck covers several frequencies, where one
    Francois-Garrison row takes ``'F'``
    (:func:`~uacpy.io.at_codes.writes_francois_garrison_letter`).
    """
    vol_atten_code = volume_attenuation_code(env.absorption,
                                             multi_frequency=multi_frequency)
    return (f"{ssp_code}{surface_code}"
            f"{AttenuationUnits.DB_PER_WAVELENGTH.to_char()}{vol_atten_code}"
            f"{pos5}{pos6}{extra}")


def format_halfspace_row(depth: str, hs) -> str:
    """The half-space row ``depth cp cs rho alpha_p alpha_s /`` the top
    (``TopBot``, ``misc/ReadEnvironmentMod.f90:285``) and the Bellhop bottom
    (``Bellhop/ReadEnvironmentBell.f90:474``) read, from the boundary ``hs``;
    ``depth`` is the depth column as the deck spells it."""
    return (f" {depth}  {hs.sound_speed:.6f} {hs.shear_speed:.6f}"
            f" {hs.density:.6f}"
            f" {hs.attenuation:.6f} {hs.shear_attenuation:.6f} /\n")


def ssp_row_attenuations(env: Environment, frequency: Optional[float],
                         depths, sound_speeds, *,
                         multi_frequency: bool = False) -> np.ndarray:
    """The compressional attenuation (dB/wavelength) of each water SSP row at
    ``depths`` / ``sound_speeds``: ``env.absorption`` at the deck
    ``frequency`` in dB per local wavelength when the rows carry the law
    (:func:`~uacpy.io.at_codes.writes_alpha_per_ssp_row` — a
    :class:`ConstantAbsorption`, :class:`FrancoisGarrison`, a tabulated
    α(f, z)), else 0 (a law with a ``TopOpt(4)`` letter is applied by the
    solver itself).

    The solver turns ``alphaI`` back into a loss at the row's own sound speed
    (``misc/AttenMod.f90:73``, ``alphaT = alpha*freq/(8.6858896*c)``), so
    the row reproduces ``α(f, z)`` exactly at each node and interpolates
    between nodes. ``frequency`` may be ``None`` only when no row law
    depends on it (none, or a constant). ``multi_frequency`` as in
    :func:`compose_topopt`."""
    z = np.atleast_1d(np.asarray(depths, dtype=float))
    absorption = env.absorption
    if not writes_alpha_per_ssp_row(absorption,
                                    multi_frequency=multi_frequency):
        return np.zeros(z.shape)
    if frequency is None and not isinstance(absorption, ConstantAbsorption):
        raise ConfigurationError(
            f"ssp_row_attenuations: env.absorption ({absorption._short()}) is "
            f"written into the SSP rows at the deck frequency, and none was "
            f"given.",
            remediation="Pass frequency= (Hz), the frequency the deck "
                        "header carries.")
    return np.asarray(absorption.alpha_dB_per_wavelength(
        frequency, z, sound_speeds), dtype=float)


def write_surface_halfspace(f, env: Environment, code: Optional[str] = None) -> None:
    """Write surface halfspace properties line if the top BC is 'A'.

    Position in the deck: after the TopOpt line *and* the volume-attenuation
    block, which ``ReadTopOpt`` consumes first, and before the SSP mesh line
    — ``TopBot`` reads this row at ``ReadEnvironmentMod.f90:285``, between
    ``ReadTopOpt`` (``:68``) and the medium loop (``:79``). See the module
    docstring for the full contract. Format:
    ``depth cp cs rho attn_p attn_s /``.

    ``code`` is the top BC letter actually written on the TopOpt line;
    it defaults to :func:`get_top_bc_code`. Pass it whenever the letter was
    resolved independently, so the row can never disagree with the letter it
    belongs to.
    """
    if (code if code is not None else get_top_bc_code(env)) != 'A':
        return
    f.write(format_halfspace_row('0.00', env.surface))


def write_ssp(filepath: Union[str, Path], ranges: np.ndarray, sound_speed: np.ndarray) -> None:
    """
    Write sound speed profile matrix to file.

    Parameters
    ----------
    filepath : str or Path
        SSP file path
    ranges : ndarray
        Range vector in metres, shape (N,), converted to the km the
        ``.ssp`` format expects at this boundary (``Bellhop/sspMod.f90:422``).
    sound_speed : ndarray
        Sound speed profiles in m/s, shape (n_depth, N)
        Each column is the SSP at the corresponding range

    Notes
    -----
    File format:
    - Line 1: Number of profiles (N)
    - Line 2: Range vector in km (space-separated)
    - Following lines: Sound speed values row by row
      (each row is SSP values at all ranges for one depth)

    This format is used for range-dependent SSP input to acoustic models.

    Translated from OALIB writessp.m

    Examples
    --------
    A range-dependent SSP: one column per range, so ``sound_speed`` is
    ``(n_depth, len(ranges))``. Written to a temporary directory so running
    the example leaves nothing behind:

    >>> import os, tempfile
    >>> ranges = np.array([0.0, 10000.0, 20000.0, 30000.0])
    >>> z = np.linspace(0, 100, 11)
    >>> gradient = 1500 - 0.1 * z[:, np.newaxis]      # (11, 1)
    >>> sound_speed = np.tile(gradient, (1, len(ranges)))     # (11, 4)
    >>> with tempfile.TemporaryDirectory() as d:
    ...     write_ssp(os.path.join(d, 'test.ssp'), ranges, sound_speed)
    ...     print(open(os.path.join(d, 'test.ssp')).readline().strip())
    4
    """
    filepath = Path(filepath)
    sound_speed = np.asarray(sound_speed, dtype=float)
    ranges = np.atleast_1d(np.asarray(ranges, dtype=float))
    r_km = m_to_km(ranges)
    Npts = len(r_km)

    # Validate range vector vs SSP matrix shape — each column of ``sound_speed``
    # is the profile at the corresponding range. Mismatched shapes will
    # otherwise produce a silently-malformed .ssp file that Bellhop
    # rejects deep in its run.
    if sound_speed.ndim != 2:
        raise ConfigurationError(
            f"write_ssp: sound_speed must be 2-D (n_depth, n_ranges); got shape {sound_speed.shape}."
        )
    if sound_speed.shape[1] != Npts:
        raise ConfigurationError(
            f"write_ssp: len(ranges) = {Npts} does not match sound_speed.shape[1] = "
            f"{sound_speed.shape[1]} (each column of sound_speed must be one profile)"
        )
    if Npts < 2:
        # Bellhop/sspMod.f90:410-412 — "You must have a least two profiles in
        # your 2D SSP field". A 1-column .ssp is rejected inside the Bellhop run.
        raise ConfigurationError(
            f"write_ssp: a Quad .ssp needs at least 2 range profiles; got "
            f"{Npts}.",
            remediation="Give the range-dependent SSP two or more range "
                        "nodes, or use a range-independent interp_ssp.",
        )
    # Bellhop's Quad segment search needs SSP%Seg%r strictly increasing
    # (Bellhop/sspMod.f90), and it reads the km tokens, not this array: the
    # check runs on the tokens the file will hold, so a decreasing, NaN or
    # sub-millimetre pair is caught here.
    tokens = [f"{r:.6f}" for r in r_km]
    bad = _collapsed_pair_index(tokens)
    if bad is not None:
        raise ConfigurationError(
            f"write_ssp: profile ranges {ranges[bad]:g} m and "
            f"{ranges[bad + 1]:g} m write as {tokens[bad]} and "
            f"{tokens[bad + 1]} km; the range axis must increase strictly at "
            f"the deck's 1 mm resolution.",
            remediation="Give strictly increasing, finite profile ranges "
                        "more than 1 mm apart.",
        )

    # AT/bellhopcuda's LDIFile reader treats each line as a separate
    # list-directed record (`LIST(SSPFile)` resets to the next line before
    # each read), so Npts and the range vector must live on different lines.
    with open(filepath, "w") as fid:
        fid.write(f"{Npts}\n")
        # 6 decimals of km = mm on the range axis. Bellhop's Quad segment
        # search needs SSP%Seg%r strictly increasing (Bellhop/sspMod.f90), so a
        # coarser format would collapse neighbouring profiles into duplicates.
        for token in tokens:
            fid.write(f"{token}  ")
        fid.write("\n")
        # Four decimals (0.1 mm/s), so a Munk-style speed such as
        # 1502.345 m/s is written unrounded; Bellhop reads the .ssp
        # list-directed (Bellhop/sspMod.f90:428), so no width applies. The
        # .env SSP rows carry the first profile at six decimals.
        for i in range(sound_speed.shape[0]):
            for j in range(sound_speed.shape[1]):
                fid.write(f"{sound_speed[i, j]:8.4f} ")
            fid.write("\n")


def write_header(
    f: TextIO,
    env: Environment,
    source: Source,
    ssp_topopt: str,
    surface_type: BoundaryType,
    frequencies: Optional[np.ndarray] = None,
    n_media_override: Optional[int] = None,
    topopt_extra: str = '',
    filepath: Optional[Union[str, Path]] = None,
    verbose: Union[bool, str] = False,
    pos5: str = ' ',
) -> None:
    """
    Write the whole top block: title, frequency, NMedia, TopOpt, the
    volume-attenuation rows, and the top half-space row.

    TopOpt position 3 is hardwired to ``'W'`` (dB/wavelength) — uacpy's
    documented unit convention for every attenuation field. Position 4 is
    taken from ``env.absorption``: ``Thorp`` → ``'T'``, ``Biological`` →
    ``'B'``, one ``FrancoisGarrison`` row on a broadband deck → ``'F'``
    (:func:`warn_if_francois_garrison_depth_frozen`), ``None`` and the laws
    the SSP rows carry in ``alphaI`` (``ConstantAbsorption``,
    ``FrancoisGarrison`` on a one-frequency deck or as a profile, a table;
    :func:`ssp_row_attenuations`) →
    ``' '``, and a broadband deck says what freezing a row law at the deck
    frequency costs (:func:`warn_if_row_absorption_frozen`); the per-formula
    follow-up rows are emitted here, before the half-space row, in the order
    ``ReadEnvironmentMod.f90`` reads them (module docstring). A
    ``TopOpt(2)='F'`` surface has its ``.trc`` table staged beside the
    ``.env``. Callers write the SSP mesh next and nothing in between.

    Parameters
    ----------
    f : TextIO
        Open file handle
    env : Environment
        Environment configuration (``env.absorption`` drives TopOpt(4))
    source : Source
        Source configuration
    ssp_topopt : str
        Pre-resolved single-character ``TopOpt(1)`` code (typically
        from :func:`resolve_ssp_topopt`).
    surface_type : BoundaryType
        Surface boundary condition
    frequencies : ndarray, optional
        Frequency vector for broadband runs. If provided, TopOpt(6) is set
        to ``'B'``; this function writes no frequency vector itself — the
        family writer emits it after the receiver depths, where
        ``ReadfreqVec`` reads it.
    n_media_override : int, optional
        Override NMedia value. Used by multi-profile writer to ensure
        all profiles have the same NMedia.
    topopt_extra : str, optional
        Extra characters appended to TopOpt beyond position 6 (e.g.
        Scooter's TopOpt(7:7)='0' to zero out stabilising attenuation —
        see ``scooter.f90:81``). Default: empty.
    filepath : str or Path, optional
        Path of the ``.env`` being written; required for a ``'file'``
        surface so its ``.trc`` table can be staged beside it.
    verbose : bool or str, optional
        Log the reflection-table staging step.
    pos5 : str, optional
        ``TopOpt(5:5)``: blank for KRAKEN, KRAKENC, SCOOTER and BOUNCE
        (krakenc tests it for ``'.'``); SPARC's output mode.
    """
    f.write(f"{quote_fortran_title(env.name)}\n")
    f.write(f"{source.frequencies[0]:.6f}\n")

    if n_media_override is not None:
        n_media = n_media_override
    else:
        n_media = 1
        if env.bottom.is_layered and not env.bottom.is_range_dependent:
            n_media += len(writable_layers(env.bottom))
    _reject_media_overrun(n_media)
    f.write(f"{int(n_media)}\n")

    surface_code = BOUNDARY_CODES[surface_type]
    broadband_code = (
        'B' if frequencies is not None and len(np.atleast_1d(frequencies)) > 1
        else ' '
    )

    # The literal blank default of ``pos5`` holds TopOpt(5:5) — krakenc tests
    # it for '.' (more root-finder restarts, Kraken/krakenc.f90:323), which a
    # blank leaves off; the restarts draw from an unseeded RANDOM_NUMBER, so a
    # deck carrying '.' does not give the same answer twice — so that
    # ``broadband_code`` lands on TopOpt(6:6) where kraken/krakenc/scooter
    # pick up the broadband flag
    # (Kraken/kraken.f90:52, Kraken/krakenc.f90:52, Scooter/scooter.f90:172).
    multi = broadband_code == 'B'
    topopt = compose_topopt(ssp_topopt, surface_code, env, pos5=pos5,
                            pos6=broadband_code, extra=topopt_extra,
                            multi_frequency=multi)
    f.write(f"'{topopt}'\n")

    write_absorption_block(f, env, multi_frequency=multi)
    if multi and writes_francois_garrison_letter(env.absorption,
                                                 multi_frequency=True):
        warn_if_francois_garrison_depth_frozen(env, frequencies)
    elif multi:
        warn_if_row_absorption_frozen(env, frequencies,
                                      float(source.frequencies[0]))

    if surface_code == 'F':
        if filepath is None:
            raise ConfigurationError(
                "write_header: acoustic_type='file' on the surface needs "
                "filepath= so the .trc table can be staged beside the .env; "
                "the deck would otherwise declare 'F' with no reflection file "
                "for AT to open.",
                remediation="Pass filepath= (the path of the .env being "
                            "written) so the .trc table lands next to it.",
            )
        stage_reflection_file(env.surface.reflection_file, filepath,
                              boundary='top', verbose=verbose)
    else:
        write_surface_halfspace(f, env, code=surface_code)


#: How a law the SSP rows carry is written into ``alphaI``: a Thorp-sized
#: loss at 1 kHz is 9e-5 dB/wavelength, which six decimals would keep to two
#: digits. A :class:`ConstantAbsorption` and a zero column keep the six
#: decimals every other attenuation field of the deck is written with.
_ROW_LAW_ALPHA_FORMAT = '.9e'


def ssp_row_attenuation_texts(env: Environment, frequency: Optional[float],
                              depths, sound_speeds, *,
                              multi_frequency: bool = False) -> List[str]:
    """:func:`ssp_row_attenuations` as the deck spells each row's
    ``alphaI``: nine significant digits for a law that varies with depth or
    frequency, six decimals for a constant or no law."""
    alpha = ssp_row_attenuations(env, frequency, depths, sound_speeds,
                                 multi_frequency=multi_frequency)
    fmt = ('.6f' if (not writes_alpha_per_ssp_row(
                         env.absorption, multi_frequency=multi_frequency)
                     or isinstance(env.absorption, ConstantAbsorption))
           else _ROW_LAW_ALPHA_FORMAT)
    return [format(float(a), fmt) for a in alpha]


def warn_if_row_absorption_frozen(env: Environment, frequencies,
                                  deck_frequency: float) -> None:
    """Say what a multi-frequency deck costs a law the SSP rows carry
    (:func:`ssp_row_attenuations`): the rows hold dB/wavelength at the deck
    frequency, and the solver re-applies that at every frequency of the
    vector, so the water absorption is linear in frequency
    (:func:`~uacpy.core.absorption.warn_if_band_absorption_frozen`). A
    :class:`ConstantAbsorption` is exactly that line and is not checked."""
    absorption = env.absorption
    if (not writes_alpha_per_ssp_row(absorption, multi_frequency=True)
            or isinstance(absorption, ConstantAbsorption)):
        return
    warn_if_band_absorption_frozen(
        'AT env writer', absorption, frequencies, float(deck_frequency),
        water_depth=float(env.depth),
        mechanism=(
            f"the SSP rows carry the absorption as dB/wavelength at the "
            f"deck frequency {float(deck_frequency):.4g} Hz, and the solver "
            f"re-applies it at every frequency of the broadband vector "
            f"(misc/AttenMod.f90:73, alphaT = alpha*freq/(8.6858896*c)), so "
            f"the water absorption is linear in frequency."),
        remediation=("Run one deck per frequency, narrow the band, or use "
                     "RAM, which evaluates the law at every frequency."))


def francois_garrison_deck_depth(env: Environment) -> float:
    """The ``z_bar`` (m) of a ``'F'`` deck: mid-water column, half the
    deck's water depth, where the one depth the solver evaluates the formula
    at departs least, at its worst, from the depths it is applied at."""
    return 0.5 * deck_depth(env.depth)


def warn_if_francois_garrison_depth_frozen(env: Environment,
                                           frequencies) -> Optional[float]:
    """Say what a broadband ``'F'`` deck costs one Francois-Garrison row:
    AT evaluates the formula at one ``z_bar``
    (:func:`francois_garrison_deck_depth`) and applies it at every depth
    (``misc/AttenMod.f90:148-160``), exact in frequency. A
    ``FallbackWarning`` when, somewhere in the band and the water column,
    that departs from the formula at the depth itself by
    :data:`~uacpy.core.absorption.BAND_ABSORPTION_WARN_DB_PER_KM` or more;
    the message also says that ``'F'`` adds the formula to the sediment
    layers and half-spaces too (``CRCI``, ``misc/AttenMod.f90:84-110``),
    where the water rows of a one-frequency deck carry it in the water
    only. Returns the error in dB/km."""
    freqs = np.atleast_1d(np.asarray(frequencies, dtype=float)).ravel()
    z_bar = francois_garrison_deck_depth(env)
    z = np.linspace(0.0, float(env.depth), BAND_ABSORPTION_CHECK_DEPTHS)
    grid = np.asarray(env.absorption.table(
        freqs, depths=np.concatenate([[z_bar], z]), units='dB/km').data,
        dtype=float).reshape(z.size + 1, freqs.size)
    err = float(np.max(np.abs(grid[1:] - grid[:1])))
    if err >= BAND_ABSORPTION_WARN_DB_PER_KM:
        warnings.warn(
            f"AT env writer: a deck covering {freqs.size} frequencies "
            f"carries the {env.absorption._short()} absorption as AT's 'F' "
            f"row, exact in frequency, which evaluates the formula at "
            f"z_bar = {z_bar:g} m (mid-water column) and applies it at every "
            f"depth (misc/AttenMod.f90:148-160). Across "
            f"{freqs.min():.4g}-{freqs.max():.4g} Hz and the water column "
            f"that departs from the formula at each depth by up to "
            f"{err:.3g} dB/km of path — about {err * 10.0:.3g} dB over "
            f"10 km. 'F' also adds the formula to the sediment layers and "
            f"half-spaces (misc/AttenMod.f90:84-110), where a one-frequency "
            f"deck carries it in the water rows only. Run one deck per "
            f"frequency, or RAM, for the formula at every depth.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
    return err


def write_absorption_block(f: TextIO, env: Environment, *,
                           multi_frequency: bool = False) -> None:
    """Emit the post-TopOpt absorption block (FG params or bio layers).

    For one :class:`FrancoisGarrison` row on a deck covering several
    frequencies (``multi_frequency``, letter ``'F'``) writes one record ``T
    S pH z_bar`` at :func:`francois_garrison_deck_depth`; for
    :class:`Biological` writes the layer count followed by one ``Z1 Z2 f0 Q
    a0`` record per layer. Every other law (Thorp, None, and the laws the
    SSP rows carry: a constant, Francois-Garrison on a one-frequency deck
    or as a profile, a table) emits nothing.

    ``ReadTopOpt`` consumes these rows before the top half-space row, so
    they belong immediately after the TopOpt line. :func:`write_header`
    already emits them; only a writer that formats its own TopOpt line
    (SPARC, Bellhop) calls this directly.
    """
    absorption = env.absorption
    code = volume_attenuation_code(absorption,
                                   multi_frequency=multi_frequency)
    if code == 'F':
        write_fg_params(f, francois_garrison_record(
            absorption, francois_garrison_deck_depth(env)))
    elif code == 'B':
        write_bio_layers(f, biological_records(absorption))


def write_fg_params(f: TextIO, params: Tuple[float, float, float, float]) -> None:
    """
    Write Francois-Garrison volume attenuation parameters.

    The AT Fortran ``ReadTopOpt`` routine reads one record of
    ``T, Salinity, pH, z_bar`` immediately after the TopOpt line when
    ``TopOpt(4)='F'``.

    Parameters
    ----------
    f : TextIO
        Open file handle
    params : tuple of 4 floats
        (T, S, pH, z_bar): temperature (degC), salinity (psu), pH,
        the depth (m) the formula is evaluated at.
    """
    if params is None or len(params) != 4:
        raise ConfigurationError(
            "write_fg_params: params must be a 4-tuple (T, S, pH, z_bar)"
        )
    T, S, pH, z_bar = params
    f.write(f"{T:.4f} {S:.4f} {pH:.4f} {z_bar:.4f}\n")


def write_bio_layers(f: TextIO, bio_layers) -> None:
    """
    Write biological attenuation layers.

    The AT Fortran ``ReadTopOpt`` routine reads one count line followed
    by ``NBioLayers`` records of ``Z1, Z2, f0, Q, a0`` when
    ``TopOpt(4)='B'``.

    Parameters
    ----------
    f : TextIO
        Open file handle
    bio_layers : list of 5-tuples
        [(Z1, Z2, f0, Q, a0), ...] per layer.
    """
    if not bio_layers:
        raise ConfigurationError("bio_layers must be a non-empty list of 5-tuples.")
    if len(bio_layers) > _MAX_BIO_LAYERS:
        raise ConfigurationError(
            f"{len(bio_layers)} biological attenuation layers exceed the "
            f"Acoustics Toolbox's MaxBioLayers = {_MAX_BIO_LAYERS} "
            f"(misc/AttenMod.f90:10), which sizes the static bio() array every "
            f"reader fills. Bellhop's reader does not bound the count "
            f"(Bellhop/ReadEnvironmentBell.f90:316-317) and segfaults on the "
            f"overrun.",
            remediation=f"Merge the layers down to at most {_MAX_BIO_LAYERS}.",
        )
    f.write(f"{len(bio_layers)}\n")
    for layer in bio_layers:
        if len(layer) != 5:
            raise ConfigurationError(
                "Each bio layer must be a 5-tuple (Z1, Z2, f0, Q, a0)"
            )
        Z1, Z2, f0, Q, a0 = layer
        f.write(f"{Z1:.4f} {Z2:.4f} {f0:.4f} {Q:.4f} {a0:.6f}\n")


def write_broadband_freqs(f: TextIO, frequencies: np.ndarray) -> None:
    """
    Write broadband frequency vector.

    In the AT file format, this is read by ReadfreqVec AFTER source/receiver
    depths (not immediately after TopOpt). Call this at the correct position
    in your env file writer.

    Parameters
    ----------
    f : TextIO
        Open file handle
    frequencies : ndarray
        Frequency vector in Hz
    """
    f.write(f"{len(frequencies)}\n")
    # AT reads this vector list-directed into REAL(KIND=8)
    # (SourceReceiverPositions.f90). 12 significant digits does NOT round-trip
    # an IEEE-754 double — that needs 17 ('%.17g') — but it puts the write
    # error at ~1e-9 Hz for a kHz-scale grid, orders below the spacing
    # uniformity the time-domain transforms need. Fixed 6-decimal text is what
    # fails: it quantises the spacing and the returned axis is no longer
    # uniform to that tolerance.
    freq_str = " ".join(f"{freq:.12g}" for freq in frequencies)
    f.write(f"{freq_str} /\n")


def write_phase_speed_and_rmax(
    f: TextIO,
    *,
    rmax_m: float,
    c_low: float,
    c_high: float,
) -> None:
    """Write the cLow/cHigh phase-speed line and the RMax (km) line.

    ``c_low`` / ``c_high`` are the resolved window (m/s): the engines
    derive it (``uacpy.models._window``) and the deck states it.

    ``rmax_m`` is converted to the km the deck expects and written at
    millimetre resolution. RMax is the range at which KRAKEN enforces
    eigenvalue accuracy — ``kraken.f90:80`` exits the Richardson
    mesh-refinement loop as soon as ``Error * 1000 * RMax < 1``, so RMax = 0
    skips every mesh doubling and returns the coarsest mesh. A coarser text
    format would round any run inside ~48 m onto that value.
    ``ReadEnvironmentMod.f90:138`` reads the field list-directed into a
    REAL(KIND=8), so there is no width constraint to respect.
    """
    f.write(f"{float(c_low):.1f} {float(c_high):.1f}\n")
    f.write(f"{float(m_to_km(rmax_m)):.6f}\n")


def write_ssp_section(
    f: TextIO,
    env: Environment,
    bottom_depth: float,
    n_mesh: int = 0,
    *,
    ssp_topopt: str,
    frequency: Optional[float] = None,
    multi_frequency: bool = False,
) -> None:
    """Write the SSP section spanning ``z = 0`` to ``bottom_depth``
    (quantised by :func:`deck_depth`).

    ``frequency`` is the deck frequency, at which a law the rows carry is
    written into each row's ``alphaI`` (:func:`ssp_row_attenuations`);
    ``multi_frequency`` as in :func:`compose_topopt`.

    ``ssp_topopt`` is the deck's ``TopOpt(1)`` letter
    (:func:`resolve_ssp_topopt`): a Biological ``env.absorption`` gets a
    node pair across every layer edge inside the column, interpolated the
    way that letter interpolates (:func:`biological_edge_nodes`).

    Both the header line and the SSP samples go through the same quantised
    depth so the AT parser sees ``ssp[-1].z == header.z_max`` exactly.
    Deep-end alignment delegates to :meth:`SoundSpeedProfile.extend_to`,
    which truncates with linear interpolation when the SSP runs past
    ``bottom_depth``, or extends by constant extrapolation when it falls
    short.

    The shallow end is anchored at ``z = 0`` the same way. AT takes the top
    of medium 1 from the first SSP row, not from the mesh line
    (``misc/sspMod.f90:355``: ``IF ( Medium == 1 ) SSP%Depth( 1 ) =
    SSP%z( 1 )``), and ``Kraken/kraken.f90:49-51`` then hands that depth to
    ``ReadSzRz`` as ``zMin``, which clamps every source and receiver above it
    (``misc/SourceReceiverPositions.f90:121-139``). A profile whose first
    sample is at 10 m would therefore move the pressure-release surface to
    10 m and model a different waveguide than the one ``env.depth``
    describes, so the shallowest speed is extrapolated up to the surface.
    """
    bottom_depth_rounded = deck_depth(bottom_depth)
    pairs = env.ssp.extend_to(bottom_depth_rounded).to_pairs()
    # Cumulative across every medium: sspMod.f90 indexes one shared array, so
    # the water rows and each sediment layer's rows share the MaxSSP budget.
    n_layers = 0
    bottom = env.bottom
    if bottom is not None:
        for col in (getattr(bottom, 'columns', None) or []):
            n_layers = max(n_layers, len(getattr(col, 'layers', ()) or ()))
    # The count is what the deck carries: the extended profile, the z = 0
    # guard row prepended below when the profile starts under the surface,
    # and two rows per sediment layer.
    # The profile the deck carries, surface guard row included, with the
    # Biological edge nodes added (the same arrays when there are none).
    guarded = (np.vstack([[0.0, pairs[0, 1]], pairs]) if pairs[0, 0] > 0.0
               else pairs)
    edge_z, edge_c = biological_edge_nodes(
        guarded[:, 0], guarded[:, 1], env.absorption, ssp_topopt,
        who='AT env writer')
    reject_oversized_at_ssp(
        'AT env writer',
        len(pairs) + int(pairs[0, 0] > 0.0)
        + (edge_z.size - guarded.shape[0]) + 2 * n_layers)
    # AT reads this mesh line as NG, SSP%sigma(Medium), Depth(Medium+1)
    # (misc/ReadEnvironmentMod.f90:81-88). For the water column that is sigma(1) —
    # the *sea surface* interface. Each sigma belongs to the interface at the top
    # of its own medium, so the seafloor is sigma(2) when a sediment layer
    # follows (write_layer_sections) and sigma(NMedia+1) on the bottom half-space
    # line when none does. Take each from its own carrier so none is mislabelled.
    surface_roughness = float(env.surface.roughness)
    f.write(f"{int(n_mesh)}  {surface_roughness:.6f}  {bottom_depth_rounded:{DECK_DEPTH_FMT}}\n")

    # A profile that starts under the surface carries its shallowest sound
    # speed up to z = 0 (the guard row above): AT places the pressure-release
    # surface on the first SSP sample, so without it the waveguide would be
    # that much thinner than env.depth and every source/receiver above it
    # would be moved down onto it. The run states it
    # (``models._checks.warn_on_ssp_start``).
    pairs = np.column_stack([edge_z, edge_c])
    alpha_i = ssp_row_attenuation_texts(env, frequency, pairs[:, 0],
                                        pairs[:, 1],
                                        multi_frequency=multi_frequency)

    # AT reads each SSP line as z, alphaR (cp), betaR (cs), rhoR, alphaI
    # (compressional attenuation), betaI (misc/sspMod.f90:334). All six are
    # pinned explicitly: Fortran's ``/`` terminator leaves unassigned items at
    # their previous value, and TopBot's 'A' branch
    # (misc/ReadEnvironmentMod.f90:285) reads the top half-space into those
    # very module variables first, so a short form donates the surface's
    # cs/rho/alphaI/betaI to the water.
    for (depth, c), alpha in zip(pairs, alpha_i):
        f.write(f"  {depth:.6f} {c:.6f} 0.000000 {env.water_density:.6f} "
                f"{alpha} 0.000000 /\n")


def write_layer_sections(
    f: TextIO,
    env: 'Environment',
    seafloor_depth: float,
    n_mesh: int = 0,
    density_reference: float = 1.0,
) -> float:
    """
    Write sediment layer SSP blocks for a layered SeabedColumn (NMEDIA > 1).

    Each SedimentLayer becomes an additional medium in the AT format.
    Each medium block has: mesh params line, then isovelocity SSP entries.

    Parameters
    ----------
    f : TextIO
        Open file handle
    env : Environment
        Environment with a layered SeabedColumn on env.bottom
    seafloor_depth : float
        Depth of the seafloor (bottom of water column)
    n_mesh : int or sequence of int, optional
        Mesh points on each medium's mesh line (0 = auto, which makes AT
        size that medium itself at 20 points per wavelength —
        ``misc/ReadEnvironmentMod.f90:107-109``). A scalar applies to every
        medium; a sequence gives one count per writable layer, which is what
        an elastic stack needs since ``ReadEnvironmentMod.f90:101-112`` sizes
        each medium from its own thickness and its own shear speed. For
        multi-profile runs, use a fixed scalar to keep NTotal consistent
        across profiles.
    density_reference : float, optional
        Every density is written divided by this (default 1, i.e. as
        carried). BOUNCE references its reflection coefficient to a unit
        water density, so its writer passes ``env.water_density`` here.

    Notes
    -----
    Every interface depth goes through :func:`deck_depth`, so the depth
    column carries exactly the value every other deck block writes — that
    shared value is what keeps a source or receiver sitting on an
    interface inside the mesh
    (``misc/SourceReceiverPositions.f90:126-139``). ``deck_depth``
    round-trips the deck's own ``%.6f`` format, so the written bytes are
    identical to printing the raw float.

    Returns
    -------
    float
        Depth of the bottom of the last sediment layer
        (i.e., top of the half-space)
    """
    if not (env.bottom.is_layered and not env.bottom.is_range_dependent):
        return seafloor_depth

    # ``deck_depth`` round-trips through the deck-wide depth format, so the
    # quantised value reprints exactly.
    interface = deck_depth
    zfmt = DECK_DEPTH_FMT

    layered = env.bottom
    current_depth = interface(seafloor_depth)
    layers = writable_layers(layered)

    if isinstance(n_mesh, (list, tuple, np.ndarray)):
        counts = [int(n) for n in n_mesh]
        if len(counts) != len(layers):
            raise ConfigurationError(
                f"write_layer_sections: n_mesh has {len(counts)} entries for "
                f"{len(layers)} writable sediment layer(s); pass one count per "
                f"layer or a single scalar."
            )
    else:
        counts = [int(n_mesh)] * len(layers)

    for n_layer_mesh, layer in zip(counts, layers):
        top_depth = current_depth
        bottom_depth = interface(current_depth + layer.thickness)
        # alphaI/betaI are list-directed REAL(KIND=8) reads
        # (misc/sspMod.f90:334) and feed CRCI unchanged (:390-393), so they
        # carry the same resolution as the water column's own attenuation
        # rather than a 0.01 dB/wavelength grid.
        # SSP%sigma(M) is the interface at the TOP of medium M
        # (ReadEnvironmentMod.f90:88 reads it on medium M's mesh line;
        # kraken.f90:902 pairs it with the media above and below), so the first
        # sediment layer's value is the seafloor and the half-space's own
        # roughness stays on the BotOpt line at the base of the stack.
        f.write(f"{n_layer_mesh}  {layer.roughness:.6f}  {bottom_depth:{zfmt}}\n")
        alpha_s = layer.shear_attenuation
        f.write(f"  {top_depth:{zfmt}} {layer.sound_speed:.6f} "
                f"{layer.shear_speed:.6f} {layer.density / density_reference:.6f} "
                f"{layer.attenuation:.6f} {alpha_s:.6f} /\n")
        f.write(f"  {bottom_depth:{zfmt}} {layer.sound_speed:.6f} "
                f"{layer.shear_speed:.6f} {layer.density / density_reference:.6f} "
                f"{layer.attenuation:.6f} {alpha_s:.6f} /\n")

        current_depth = bottom_depth

    return current_depth


def write_bottom_section(
    f: TextIO,
    env: Environment,
    bottom_type: Optional[BoundaryType] = None,
    filepath: Optional[Path] = None,
    verbose: Union[bool, str] = False,
    halfspace_depth: Optional[float] = None,
    density_reference: float = 1.0,
) -> None:
    """
    Write bottom boundary section

    Parameters
    ----------
    f : TextIO
        Open file handle
    env : Environment
        Environment configuration
    bottom_type : BoundaryType, optional
        Bottom boundary type (uses env.bottom.acoustic_type if None)
    filepath : Path, optional
        Path to the ENV file being written; required for a ``'file'``
        (``.brc``) or ``'precalc'`` (``.irc``) seabed so the table can be
        staged beside it.
    verbose : bool or str, optional
        Print verbose output
    halfspace_depth : float, optional
        Depth used for the 'A' halfspace line. Defaults to ``env.depth``
        plus stacked layered-bottom thicknesses.
    density_reference : float, optional
        Every density is written divided by this (default 1, i.e. as
        carried). BOUNCE references its reflection coefficient to a unit
        water density, so its writer passes ``env.water_density`` here.
    """
    hs = env.bottom.halfspace_at(range=0.0)
    if bottom_type is None:
        bottom_type = parse_boundary_type(hs.acoustic_type)

    cp = hs.sound_speed
    cs = hs.shear_speed
    rho = hs.density / density_reference
    alpha = hs.attenuation

    bottom_code = BOUNDARY_CODES[bottom_type]
    sigma = hs.roughness

    # misc/ReadEnvironmentMod.f90:121-129 reads only BotOpt(1:1); a bathymetry
    # flag in position 2 is Bellhop's alone, and Bellhop has its own writer.
    # sigma goes into the list-directed REAL(KIND=8) SSP%sigma(NMedia+1)
    # (:125), and the scatter approximation is only valid up to
    # sigma <= 60 / freq metres (:93-94) — millimetres at kHz frequencies —
    # so the column carries the value at the same resolution as the rest.
    f.write(f"'{bottom_code}' {sigma:.6f}\n")

    if bottom_code == 'F':
        if filepath is None:
            raise ConfigurationError(
                "write_bottom_section: acoustic_type='file' needs filepath= so "
                "the .brc table can be staged beside the .env; the model would "
                "otherwise declare 'F' with no reflection file for AT to open.",
                remediation="Pass filepath= (the path of the .env being "
                            "written) so the .brc table lands next to it.",
            )
        stage_reflection_file(hs.reflection_file, filepath,
                              boundary='bottom', verbose=verbose)

    elif bottom_code == 'P':
        if filepath is None:
            raise ConfigurationError(
                "write_bottom_section: acoustic_type='precalc' needs filepath= "
                "so the .irc table can be staged beside the .env; the model "
                "would otherwise declare 'P' with no table for AT to open.",
                remediation="Pass filepath= (the path of the .env being "
                            "written) so the .irc table lands next to it.",
            )
        # RefCoef.f90:92-96 opens <root>.irc — BOUNCE's (x, f, g, iPow)
        # table, a different format from the angle/magnitude/phase .brc.
        stage_reflection_file(hs.reflection_file, filepath,
                              boundary='internal', verbose=verbose)

    elif bottom_code == 'A':  # Half-space
        if halfspace_depth is not None:
            z_bottom = halfspace_depth
        else:
            # Same quantiser and same print width as the mesh line that
            # declares this interface, so the deck states one depth for it.
            # (KRAKEN itself discards this column — misc/ReadEnvironmentMod.f90
            # :257,285 read it into TopBot's local ``zTemp`` and never assign
            # it — but BELLHOP does not, and a deck that names one interface
            # three ways cannot be diffed against a reference.)
            z_bottom = deck_depth(env.depth)
            if env.bottom.is_layered and not env.bottom.is_range_dependent:
                for layer in writable_layers(env.bottom):
                    z_bottom = deck_depth(z_bottom + layer.thickness)
        # betaI on the 'A' line (misc/ReadEnvironmentMod.f90:285). Every AT program that
        # reads an elastic half-space uses it: krakenc and bounce apply it, and
        # real kraken.exe accepts the column and ignores it, so the
        # environment's value is written unconditionally.
        alpha_s = hs.shear_attenuation
        f.write(f"  {z_bottom:.6f}  {cp:.6f}  {cs:.6f}  "
                f"{rho:.6f}  {alpha:.6f}  {alpha_s:.6f} /\n")


def write_vector_record(f: TextIO, values) -> None:
    """Write an Acoustics Toolbox vector record: the count on one line, the
    values and the ``/`` terminator on the next (``ReadVector`` in
    ``misc/SourceReceiverPositions.f90``)."""
    values = np.asarray(values, dtype=float).ravel()
    f.write(f"{len(values)}\n")
    f.write(" ".join(f"{v:.6f}" for v in values) + " /\n")


def write_source_depths(f: TextIO, source) -> None:
    """Write the source-depth section of an Acoustics Toolbox ``.env`` file.

    Accepts either a ``Source`` instance or a 1-D depths array, like
    :func:`write_receiver_depths`.
    """
    depths = source.depths if isinstance(source, Source) else source
    write_vector_record(f, depths)


def write_receiver_depths(f: TextIO, receiver_or_depths) -> None:
    """Write the receiver-depth section of an Acoustics Toolbox ``.env`` file.

    Accepts either a ``Receiver`` instance or a 1-D depths array.
    """
    depths = (
        receiver_or_depths.depths if isinstance(receiver_or_depths, Receiver)
        else receiver_or_depths
    )
    write_vector_record(f, depths)


def write_receiver_ranges(f: TextIO, receiver_or_ranges) -> None:
    """Write the receiver-range section (ranges converted from m to km).

    Accepts either a ``Receiver`` instance or a 1-D ranges array (metres),
    like :func:`write_receiver_depths`.

    The check is on the **written** values, not on the ranges given: at
    ``.6f`` km the deck resolves 1 mm, so two ranges closer than that collapse
    to one token even though the carrier's own strictly-increasing guard
    passed. Bellhop reads the file, not the array.
    """
    ranges = np.atleast_1d(np.asarray(
        getattr(receiver_or_ranges, 'ranges', receiver_or_ranges),
        dtype=float))
    n_rr = len(ranges)
    tokens = [f"{float(m_to_km(r)):.6f}" for r in ranges]
    bad = _collapsed_pair_index(tokens)
    if bad is not None:
        raise ConfigurationError(
            f"receiver ranges {ranges[bad]:g} m and "
            f"{ranges[bad + 1]:g} m both write as "
            f"{tokens[bad]} km at the deck's 1 mm resolution, leaving the "
            f"range axis non-increasing.",
            remediation="Separate the receiver ranges by more than 1 mm.",
        )
    f.write(f"{n_rr}\n")
    f.write(f"{' '.join(tokens)} /\n")


def write_multi_profile_env(
    filepath: Union[str, Path],
    segments: List[Tuple[float, 'Environment']],
    source: Source,
    receiver: Receiver,
    **kwargs
) -> None:
    """
    Write multi-profile .env file for kraken.exe range-dependent mode.

    kraken.exe reads profile sections sequentially from a single .env file
    (via its ``Profile: DO iProf = 1, 9999`` loop), computing modes for
    each and writing them all into one .mod file.

    Each profile block contains: title, freq, NMedia, TopOpt, SSP,
    BotOpt, bottom halfspace, cLow/cHigh, RMax, source depths,
    receiver depths. Receiver ranges are NOT included (field.exe
    reads them from the .flp file).

    ``n_mesh`` is written as the ``NG`` mesh count on every medium line
    of every profile. The default 0 lets KRAKEN size each medium of each
    profile itself at 20 points per wavelength with a 10-point floor
    (``misc/ReadEnvironmentMod.f90:99-110``). Per-profile meshes are
    legal: the ``.mod`` record length is set once from the first profile
    as ``MAX(2*Nfreq, 2*NzTab, 32, 3*NMedia_acoustic)``
    (``Kraken/kraken.f90:587``, ``krakenc.f90:630``) and carries no mesh
    term. What the record length does depend on is held constant another
    way — every profile is padded to the same NMedia
    (:func:`plan_multi_profile_media`) and the source/receiver depth
    lines are written identically into each profile block.

    Parameters
    ----------
    filepath : Path
        Output .env file path
    segments : list of (range_m, Environment)
        Range segments (segment start in metres). Each Environment must be
        range-independent.
    source : Source
        Source configuration (frequency, depth)
    receiver : Receiver
        Receiver configuration (depths for mode computation)
    **kwargs
        n_mesh, c_low, c_high, rmax_m passed through. ``c_low`` and
        ``c_high`` are required: ``c_high`` is one value for every profile,
        or a sequence of one per profile (each profile's own window).
        ``n_mesh`` must be
        >= 0; ``rmax_m`` (metres) defaults to the farthest receiver range,
        as in :func:`write_kraken_env_file`. ``interp_ssp``
        (``'linear'`` when omitted) selects each block's SSP interpolation via
        :func:`resolve_ssp_topopt`, and ``verbose`` logs the bottom
        sections.
        TopOpt position 4 is taken from each segment env's ``absorption``
        field via :func:`write_header`.
    """
    # Reject any ``**kwargs`` key the blocks below never read, so a typo'd
    # option fails loudly instead of being silently dropped.
    reject_unknown_kwargs(
        'write_multi_profile_env', kwargs,
        {'n_mesh', 'c_low', 'c_high', 'rmax_m', 'interp_ssp', 'verbose'},
    )
    missing = [k for k in ('c_low', 'c_high') if k not in kwargs]
    if missing:
        raise TypeError(
            f"write_multi_profile_env: missing required keyword(s) "
            f"{', '.join(missing)} (the resolved phase-speed window, m/s).")
    c_low = kwargs['c_low']
    c_high = kwargs['c_high']
    c_highs = (list(c_high) if isinstance(c_high, (list, tuple, np.ndarray))
               else [c_high] * len(segments))
    if len(c_highs) != len(segments):
        raise ConfigurationError(
            f"write_multi_profile_env: c_high holds {len(c_highs)} values "
            f"for {len(segments)} profiles; pass one value, or one per "
            f"profile.")
    rmax_m = _rmax_or_farthest_receiver(kwargs.get('rmax_m'), receiver,
                                        'write_multi_profile_env')
    # NG = 0 on a mesh line asks the reader to size that medium of that
    # profile itself (misc/ReadEnvironmentMod.f90:105-110).
    n_mesh = int(kwargs.get('n_mesh', 0))
    if n_mesh < 0:
        raise ConfigurationError(
            f"write_multi_profile_env: n_mesh must be >= 0 (0 lets KRAKEN "
            f"size each medium); got {n_mesh}.")

    max_n_media, _bottom_depth, media_plans = plan_multi_profile_media(segments)

    interp_ssp = kwargs.get('interp_ssp', 'linear')

    with open(filepath, 'w') as f:
        for (_range_m, env_seg), all_extra_media, c_high_seg in zip(
                segments, media_plans, c_highs):
            ssp_topopt = resolve_ssp_topopt(env_seg, interp_ssp)
            surface_obj = getattr(env_seg, 'surface', None)
            if surface_obj is not None:
                surface_type = parse_boundary_type(
                    surface_obj.acoustic_type
                )
            else:
                surface_type = BoundaryType.VACUUM
            bottom_type = parse_boundary_type(env_seg.bottom.halfspace_at(range=0.0).acoustic_type)

            write_header(
                f, env_seg, source,
                ssp_topopt=ssp_topopt,
                surface_type=surface_type,
                n_media_override=max_n_media,
                filepath=filepath,
                verbose=kwargs.get('verbose', False),
            )

            # --- Water column (medium 1) ---
            write_ssp_section(
                f, env_seg, env_seg.depth,
                n_mesh=n_mesh, ssp_topopt=ssp_topopt,
                frequency=float(source.frequencies[0]),
            )

            # --- Sub-bottom media (2..max_n_media), from the shared plan ---
            for top, bot, cp, cs, rho_v, ap, as_, sigma in all_extra_media:
                f.write(f"{int(n_mesh)}  {sigma:.6f}  {bot:{DECK_DEPTH_FMT}}\n")
                f.write(f"  {top:{DECK_DEPTH_FMT}} {cp:.6f} "
                        f"{cs:.6f} {rho_v:.6f} "
                        f"{ap:.6f} {as_:.6f} /\n")
                f.write(f"  {bot:{DECK_DEPTH_FMT}} {cp:.6f} "
                        f"{cs:.6f} {rho_v:.6f} "
                        f"{ap:.6f} {as_:.6f} /\n")

            # Halfspace depth = bottom of all media; a profile with no
            # sub-bottom medium ends on its own seafloor.
            hs_depth = (all_extra_media[-1][1] if all_extra_media
                        else deck_depth(env_seg.depth))

            write_bottom_section(
                f, env_seg, bottom_type=bottom_type,
                filepath=filepath,
                verbose=kwargs.get('verbose', False),
                halfspace_depth=hs_depth,
            )

            write_phase_speed_and_rmax(
                f, rmax_m=rmax_m, c_low=c_low, c_high=c_high_seg,
            )

            write_source_depths(f, source)
            write_receiver_depths(f, receiver)


#: ``field.exe`` option columns it validates itself and ERROUTs on
#: (``KrakenField/field.f90:70-99``; column 2 at ``:125-136``, which the
#: program examines only when NProf > 1).
#:
#: The same letter means different things in different COLUMNS of this one
#: string — column 2 ``'C'`` is Coupled, column 4 ``'C'`` is Coherent — so the
#: alphabets are keyed by column and never merged. A blank is meaningful in
#: columns 3 and 4 (``CASE ( 'O', ' ' )`` and ``CASE ( 'C', ' ' )``), which is
#: why those sets carry a space; columns 1 and 2 have no blank case and
#: ERROUT on one.
_FLP_OPTION_ALPHABET = {
    1: (set('XRS'), 'source type (X line / R point / S scaled cylindrical)'),
    2: (set('CA'), 'mode coupling (C coupled / A adiabatic)'),
    3: (set('*O '), 'source beam pattern (* file / O or blank omnidirectional)'),
    4: (set('CI '), 'mode addition (C coherent / I incoherent)'),
}


def _validate_flp_option(option: str, n_profiles: int = 1) -> None:
    """Reject a ``.flp`` option string ``field.exe`` would ERROUT on.

    Column 2 is skipped for ``n_profiles == 1``: ``field.f90:125-136`` reaches
    its coupling SELECT CASE only for a range-dependent run (NProf > 1), so a
    blank there is legal in a single-profile deck.

    Without this the deck only fails inside the Fortran run, with the error
    buried in the ``.prt``.
    """
    padded = f"{option:<4s}"
    for col, (allowed, description) in _FLP_OPTION_ALPHABET.items():
        if col == 2 and n_profiles <= 1:
            continue
        char = padded[col - 1]
        if char not in allowed:
            raise ConfigurationError(
                f"write_fieldflp: option position {col} ({description}) must be "
                f"one of {sorted(allowed)}; got {char!r} in {option!r}."
            )
    # The per-column alphabets are not the whole contract: field.f90:126-129
    # ERROUTs on coupled modes asked for an incoherent sum. That matters more
    # than an ordinary deck error because misc/FatalError.f90:30 is
    # ``STOP '<string>'``, so every Acoustics-Toolbox fatal error exits with
    # status 0 — the run writes no .shd and a caller trusting the return code
    # reads a stale or missing output as success.
    if n_profiles > 1 and padded[1] == 'C' and padded[3] == 'I':
        raise ConfigurationError(
            f"write_fieldflp: option {option!r} asks for coupled modes "
            f"(position 2 = 'C') with an incoherent sum (position 4 = 'I'), "
            f"which field.f90:126-129 rejects outright. Use 'C' with a "
            f"coherent sum, or adiabatic modes ('A') for an incoherent one."
        )


def _write_flp_axis(f: TextIO, values, count_label: str, label: str,
                    *, subtabulate: bool = True) -> None:
    """One ``.flp`` axis: the count, then either the ``first last /``
    shortcut FIELD subtabulates (``misc/subtabulate.f90:24,40``, taken only
    for more than two equally spaced values) or every value in full.

    ``subtabulate=False`` always writes in full. The shortcut makes FIELD
    *recompute* the intermediate values in single precision, so they land a
    few ULPs from the ``%.6f`` the same numbers are written as elsewhere —
    harmless for an axis FIELD only evaluates on, and not for the receiver
    depths, which ``Kraken._write_field_env`` places on the mode
    tabulation grid so they are read off a tabulated point instead of
    interpolated between two. A few ULPs is enough to put the interpolation
    weight at ~1e-7 instead of 0, which let a source depth elsewhere in the
    grid move a receiver's answer by 6e-6.
    """
    f.write(f"{len(values):5d} \t \t \t \t ! {count_label} \n")
    if subtabulate and len(values) > 2 and equally_spaced(values):
        f.write(f"    {values[0]:.6f}  {values[-1]:.6f} ")
    else:
        for v in values:
            f.write(f"    {v:.6f}  ")
    f.write(f"/ \t ! {label} \n")


def write_fieldflp(
    filepath: Union[str, Path],
    option: str,
    pos: Dict[str, Any],
    title: str = "",
    n_modes: int = 999999,
    n_profiles: int = 1,
    profile_ranges: Any = None,
) -> None:
    """
    Write field parameters file (.flp) for FIELD/FIELDS programs.

    Parameters
    ----------
    filepath : str or Path
        Output file path. ``.flp`` is appended only when the path has no
        suffix (the same convention :func:`~uacpy.io.oalib_reader.read_flp`
        resolves with); a path that carries a suffix is written exactly as
        given.
    option : str
        4-character option string for field.exe. Column semantics per AT
        ``KrakenField/field.f90:70-99`` and
        ``KrakenField/ReadModes.f90:315-324``:

        - Pos 1 (source type):
          'R' = cylindrical point source, 'X' = line source (Cartesian),
          'S' = scaled-cylindrical point source.
        - Pos 2 (coupling, examined by field.exe only for NProf > 1,
          ``field.f90:125-136``): 'C' = coupled modes, 'A' = adiabatic.
        - Pos 3: either ``'*'`` to apply a ``.sbp`` source beam pattern
          or ``' '`` for omnidirectional. ``field.exe``
          (``KrakenField/field.f90:83-90``)
          only accepts ``{' ', 'O', '*'}`` through this writer; elastic
          component selectors (``'P'``/``'H'``/``'V'``/``'T'``/``'N'``)
          are not reachable from uacpy.
        - Pos 4 (summation): 'C' = coherent, 'I' = incoherent.
    pos : dict
        Position dictionary with:
        - 's': dict with 'z' (source depths in m)
        - 'r': dict with 'z' (receiver depths in m), 'r' (ranges in m)
    title : str, optional
        Title for the file. Default ``''``, which ``quote_fortran_title``
        writes as ``'unnamed'`` (an empty quoted title is legal to field.f90
        but useless as a label).
    n_modes : int, optional
        Maximum number of modes to include (default: 999999 = all)
    n_profiles : int, optional
        Number of range profiles (default: 1 for range-independent).
        For range-dependent, set > 1 and provide profile_ranges_m.
    profile_ranges : array-like, optional
        Profile boundary ranges in metres, converted to the km the
        ``.flp`` format expects at this boundary. Required when
        n_profiles > 1. First value must be 0.0. Length must equal
        n_profiles.

    Notes
    -----
    File format (.flp):
    - Line 1: Title
    - Line 2: Option (quoted, 4 chars)
    - Line 3: MLimit
    - Line 4: NProf (number of profiles)
    - Line 5: rProf (profile ranges in km)
    - Lines 6+: Receiver ranges, source depths, receiver depths, range offsets

    For range-dependent (NProf > 1), field.exe reads modes for each profile
    from a single .mod file produced by kraken.exe with multi-profile .env.

    See Also
    --------
    read_flp : Read field parameters file
    """
    filepath = Path(filepath)
    if not filepath.suffix:
        filepath = filepath.with_suffix(".flp")

    _validate_flp_option(option, n_profiles)

    r_ranges = m_to_km(pos["r"]["r"])
    s_depths = pos["s"]["z"]
    r_depths = pos["r"]["z"]

    # Validate profile parameters
    if n_profiles > 1:
        if profile_ranges is None:
            raise ConfigurationError("profile_ranges_m required when n_profiles > 1.")
        profile_ranges_km = m_to_km(profile_ranges)
        if len(profile_ranges_km) != n_profiles:
            raise ConfigurationError(
                f"profile_ranges_m length ({len(profile_ranges_km)}) "
                f"must equal n_profiles ({n_profiles})"
            )
        if abs(profile_ranges_km[0]) > 1e-9:
            raise ConfigurationError("First profile range must be 0.0 km.")
        # Check the values as WRITTEN. field.exe never tests rProf for
        # monotonicity — `grep monotonic KrakenField/field.f90` is empty, unlike
        # ReadRcvrRanges (misc/SourceReceiverPositions.f90:163-165) — so a
        # collapsed pair reaches EvaluateADMod.f90:75, whose
        # `(rProf(iProf+1) - rProf(iProf))` denominator is unguarded. The 0/0
        # then poisons a whole segment's interpolated wavenumbers and mode
        # functions, and nothing in AT reports it: the caller gets a partly-NaN
        # field with no error and no warning.
        tokens = [f"{float(r):.6f}" for r in profile_ranges_km]
        bad = _collapsed_pair_index(tokens)
        if bad is not None:
            raise ConfigurationError(
                f"profile ranges {profile_ranges[bad]:g} m and "
                f"{profile_ranges[bad + 1]:g} m both write as "
                f"{tokens[bad]} km at the deck's 1 mm resolution, leaving the "
                f"profile axis non-increasing.",
                remediation=(
                    f"Separate the profile ranges by more than "
                    f"{DECK_RANGE_RESOLUTION_M * 1e3:g} mm."
                ),
            )

    with open(filepath, "w") as f:
        f.write(f"{quote_fortran_title(title)} ! Title \n")

        # Option
        f.write(f"'{option:4s}'  ! Option \n")

        # Mode limit
        f.write(f"{int(n_modes)}   ! Mlimit (number of modes to include) \n")

        # Profile info
        f.write(f"{int(n_profiles)}        ! NProf  \n")
        if n_profiles == 1:
            f.write("0.0 /    ! rProf (km) \n")
        else:
            for r in profile_ranges_km:
                f.write(f"    {r:.6f}  ")
            f.write("/ \t ! rProf (km) \n")

        # Receiver ranges. The "first last /" shorthand below relies on
        # SubTab expanding the record, which it only does for Nx >= 3
        # (misc/subtabulate.f90:24,40) — hence the ``> 2`` gate on this and the
        # two depth blocks; shorter vectors are written out in full.
        _write_flp_axis(f, r_ranges, 'NRr', 'Rr(1)  ... (km)')
        _write_flp_axis(f, s_depths, 'NSz', 'Sz(1)  ... (m)')
        # In full: these depths are tabulation nodes, not just query
        # points — see ``_write_flp_axis``.
        _write_flp_axis(f, r_depths, 'NRz', 'Rz(1)  ... (m)',
                        subtabulate=False)

        # Receiver range offsets (array tilt) - default to zeros for every
        # receiver. field.exe ERROUTs unless ``NRro == NRz``
        # (KrakenField/field.f90:149-152), so the count stays NRz. The
        # sentinel ``/`` terminator paired with a single explicit value
        # lets AT's SubTab routine replicate it across the full vector
        # (see misc/subtabulate.f90 — when x(3) is left at its -999.9
        # default, the vector is filled by repeating x(1)). This matches
        # the canonical AT examples MunkK.flp / DickinsK_rd.flp.
        f.write(f"{len(r_depths):5d} \t \t \t \t ! NRro \n")
        if len(r_depths) >= 3:
            f.write("    0.0 /    \t \t \t \t ! Rro(1)  ... (m) \n")
        else:
            # SubTab only replicates for Nx >= 3 (misc/subtabulate.f90:24).
            # Below that the sentinel idiom leaves x(2) at ReadVector's
            # -999.9 pre-fill (SourceReceiverPositions.f90:219-221) and the
            # following Sort moves it to Rro(1), so the shallowest receiver
            # is evaluated at r - 999.9 m. Write the vector out in full.
            zeros = "  ".join(f"{0.0:.6f}" for _ in r_depths)
            f.write(f"    {zeros} /    \t \t \t \t ! Rro(1)  ... (m) \n")


def write_field3dflp(
    filepath: Union[str, Path],
    option: str,
    pos: Dict[str, Any],
    bathy: Dict[str, Any],
    mod_file_pattern: str = "'{}'",
    title: str = "",
    n_modes: int = 999999,
) -> None:
    """
    Write the FIELD3D field-parameter deck (``.flp``).

    **Retained for planned 3-D support — this is not dead code.** Nothing in
    the 2-D public API writes a FIELD3D deck; this is the writer a future
    3-D implementer builds on, and the round-trip partner of
    :func:`~uacpy.io.oalib_reader.read_flp3d`.

    Parameters
    ----------
    filepath : str or Path
        Output path. ``.flp`` is appended only when the path has no suffix —
        the convention :func:`write_fieldflp` and
        :func:`~uacpy.io.oalib_reader.read_flp` share; a path that carries a
        suffix is written exactly as given.
    option : str
        FIELD3D option word, ``CHARACTER(LEN=7)`` in the Fortran
        (``KrakenField/field3d.f90:152``). Columns 1-3 select the evaluator
        (``'STD'`` / ``'PAR'`` / ``'GBT'``, ``:96``), column 4 ``'T'``
        requests the tesselation check (``:210``), column 7 is the
        source-beam-pattern flag (``:54``). ``'STDFM'`` is what the shipped
        AT deck uses.
    pos : dict
        Positions, all in **metres** / degrees:

        - ``'s'``: ``{'x': (Nsx,), 'y': (Nsy,), 'z': (Nsz,)}``
        - ``'r'``: ``{'z': (Nrz,), 'r': (Nrr,), 'theta': (Ntheta,) degrees}``
        - optional ``'Nsx'`` / ``'Nsy'`` overriding the counts.

        ``x``, ``y`` and ``r`` are converted to the km the deck holds
        (``field3d.f90:174-175, 177``); depths and bearings are not.
    bathy : dict
        The node grid: ``{'X': (nx,) m, 'Y': (ny,) m, 'depth': (ny, nx) m}``.
        ``X``/``Y`` are converted to km on write. ``depth`` chooses which
        nodes get a real mode file and which get ``'DUMMY'``.
    mod_file_pattern : str, optional
        Mode-file name written beside each wet node. A pattern carrying
        ``{}`` / ``{:`` is formatted with ``(x_km, y_km)``; anything else is
        written verbatim. Default ``"'{}'"``.
    title : str, optional
        Deck title. Default ``''``, written as ``'unnamed'`` by
        ``quote_fortran_title``.
    n_modes : int, optional
        Mode-count cap. Default 999999.

    Raises
    ------
    ~uacpy.core.exceptions.ConfigurationError
        An axis is empty, ``depth`` does not match the node grid, or the
        grid is too small to triangulate (both ``nx`` and ``ny`` need at
        least two points for a single quad, hence two triangles).

    Notes
    -----
    Record order, matching ``field3d.f90:163-207`` exactly — title, option,
    ``Mlimit``, ``Sx``, ``Sy``, ``Sz``, ``Rz``, ``Rr``, ``theta``, the node
    table, the element table. Each axis is emitted as the AT
    ``first last /`` shorthand when it is uniformly sampled and longer than
    two points, which ``SubTab`` expands back to the declared count.

    **Element node indices are 1-based**, because the Fortran reads them
    straight into arrays it indexes from 1 (``field3d.f90:207``, then
    ``x( Node( 1, iElt ) )``): a 0-based table addresses ``x(0)`` on every
    triangle of the first row and runs one past ``x(NNodes)`` on the last.
    The shipped AT deck ``tests/3DAtlantic/lant.flp`` numbers its 397 nodes
    1..397, and ``field3d.exe`` echoes that count back on load.

    See Also
    --------
    ~uacpy.io.oalib_reader.read_flp3d : Read the deck back.
    write_fieldflp : The 2-D field-parameter writer.

    Examples
    --------
    >>> import tempfile, os
    >>> import numpy as np
    >>> from uacpy.io.oalib_reader import read_flp3d
    >>> pos = {
    ...     's': {'x': np.array([0.0]), 'y': np.array([0.0]),
    ...           'z': np.array([50.0])},
    ...     'r': {'z': np.linspace(0.0, 100.0, 11),
    ...           'r': np.linspace(0.0, 50000.0, 51),
    ...           'theta': np.linspace(0.0, 350.0, 36)},
    ... }
    >>> bathy = {'X': np.linspace(0.0, 100000.0, 11),
    ...          'Y': np.linspace(0.0, 100000.0, 11),
    ...          'depth': 100.0 * np.ones((11, 11))}
    >>> with tempfile.TemporaryDirectory() as d:
    ...     path = os.path.join(d, 'field3d.flp')
    ...     write_field3dflp(path, 'STDFM', pos, bathy,
    ...                      mod_file_pattern="'mode_{:07.1f}_{:07.1f}'",
    ...                      title='3D Test')
    ...     deck = read_flp3d(path)
    >>> deck.title, deck.method
    ('3D Test', 'STD')
    >>> len(deck.node_mode_files), deck.elements.shape
    (121, (200, 3))
    >>> int(deck.elements.min()), int(deck.elements.max())
    (1, 121)
    """
    filepath = Path(filepath)
    # Append only when the caller gave no suffix. Replacing an explicit one
    # silently renames the file the caller asked for -- the defect
    # write_fieldflp was corrected for, and read_flp3d resolves the same way.
    if not filepath.suffix:
        filepath = filepath.with_suffix(".flp")

    s_x = m_to_km(np.asarray(pos["s"]["x"], dtype=float))
    s_y = m_to_km(np.asarray(pos["s"]["y"], dtype=float))
    s_z = np.asarray(pos["s"]["z"], dtype=float)
    r_z = np.asarray(pos["r"]["z"], dtype=float)
    r_r = m_to_km(np.asarray(pos["r"]["r"], dtype=float))
    r_theta = np.asarray(pos["r"]["theta"], dtype=float)
    Nsx = pos.get("Nsx", s_x.size)
    Nsy = pos.get("Nsy", s_y.size)

    X = m_to_km(np.asarray(bathy["X"], dtype=float))
    Y = m_to_km(np.asarray(bathy["Y"], dtype=float))
    depth = np.asarray(bathy["depth"], dtype=float)
    nx = X.size
    ny = Y.size

    # ReadVector ERROUTs on a non-positive count
    # (misc/SourceReceiverPositions.f90:212), which is STOP at exit 0 with no
    # field written, so an empty axis is refused before the deck exists.
    for label, axis in (("source x", s_x), ("source y", s_y),
                        ("source depths", s_z), ("receiver depths", r_z),
                        ("receiver ranges", r_r),
                        ("receiver bearings", r_theta)):
        if axis.size == 0:
            raise ConfigurationError(
                f"write_field3dflp: {label} is empty; FIELD3D's ReadVector "
                f"ERROUTs on a count of 0 "
                f"(misc/SourceReceiverPositions.f90:212).",
                remediation=f"Give at least one {label} value.",
            )
    if depth.shape != (ny, nx):
        raise ConfigurationError(
            f"write_field3dflp: bathy['depth'] has shape {depth.shape}, but "
            f"bathy['X']/['Y'] describe a (ny={ny}, nx={nx}) node grid.",
            remediation="Pass depth[iy, ix] = depth at (X[ix], Y[iy]); "
                        "transpose an (nx, ny) array before calling.",
        )
    if nx < 2 or ny < 2:
        raise ConfigurationError(
            f"write_field3dflp: the node grid is {nx} x {ny}; a "
            f"triangulation needs at least 2 x 2 nodes to form one quad, and "
            f"FIELD3D ERROUTs on an element count of 0 "
            f"(field3d.f90:199-203).",
            remediation="Give at least two X and two Y node coordinates.",
        )

    def _write_axis(f, values, count, trailer):
        """One AT vector record: the ``first last /`` shorthand when SubTab
        can regenerate the axis, otherwise every value then ``/``."""
        f.write(f"{count}\n")
        if count > 2 and equally_spaced(values):
            f.write(f"{values[0]:.6f} {values[-1]:.6f} /{trailer}\n")
        else:
            f.write(" ".join(f"{v:.6f}" for v in values) + f" /{trailer}\n")

    with open(filepath, "w") as f:
        # Both records are Fortran character literals read list-directed
        # (field3d.f90:163 into CHARACTER(LEN=80), :166 into
        # CHARACTER(LEN=7)), so an apostrophe interpolated raw closes the
        # literal early and truncates the value. Route the title through
        # the same sanitizer every 2-D deck writer uses, and strip the
        # same character from the option word — losing an apostrophe from
        # a cosmetic title is harmless, while a truncated option word
        # shifts the columns the evaluator, tesselation-check and
        # beam-pattern flags live in.
        opt_literal = str(option).replace("'", "")
        f.write(f"{quote_fortran_title(title)} ! TITLE\n")
        f.write(f"'{opt_literal}' ! OPT\n")
        f.write(f"{n_modes} ! MLIMIT\n")

        _write_axis(f, s_x, Nsx, " ! Sx (km)")
        _write_axis(f, s_y, Nsy, " ! Sy (km)")
        _write_axis(f, s_z, s_z.size, " ! Sz (m)")
        _write_axis(f, r_z, r_z.size, " ! Rz (m)")
        _write_axis(f, r_r, r_r.size, " ! Rr (km)")
        _write_axis(f, r_theta, r_theta.size, " ! theta (degrees)")

        # Node table: x (km), y (km), mode-file name. Row-major over y, so
        # node (ix, iy) is number iy * nx + ix + 1.
        f.write(f"{nx * ny} ! NNODES\n")
        for iy in range(ny):
            for ix in range(nx):
                x_coord = X[ix]
                y_coord = Y[iy]
                if depth[iy, ix] > 0:
                    if "{}" in mod_file_pattern or "{:" in mod_file_pattern:
                        modfil = mod_file_pattern.format(x_coord, y_coord)
                    else:
                        modfil = mod_file_pattern
                else:
                    modfil = "'DUMMY'"
                # %.6f km = 1 mm, the resolution every other km column in
                # this package is written at. AT's own deck
                # (tests/3DAtlantic/lant.flp) prints 2 decimals, i.e. 10 m,
                # which silently moves a node the caller placed; the READ is
                # list-directed (field3d.f90:192) so the extra digits cost
                # nothing.
                f.write(f"{x_coord:14.6f} {y_coord:14.6f} {modfil}\n")

        # Element table: two triangles per quad, node indices 1-based
        # (field3d.f90:207 reads them into a Fortran array indexed from 1).
        f.write(f"{2 * (nx - 1) * (ny - 1)} ! NELTS\n")
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                n0 = iy * nx + ix + 1
                f.write(f"{n0:5d} {n0 + 1:5d} {n0 + nx:5d}\n")
                f.write(f"{n0 + 1:5d} {n0 + nx:5d} {n0 + nx + 1:5d}\n")


def _env_boundary_types(env: Environment) -> Tuple[BoundaryType, BoundaryType]:
    """The surface and bottom boundary types a deck writes for ``env``:
    its surface's, and its seabed half-space's at range 0."""
    return (parse_boundary_type(env.surface.acoustic_type),
            parse_boundary_type(
                env.bottom.halfspace_at(range=0.0).acoustic_type))


def _rmax_or_farthest_receiver(rmax_m: Optional[float], receiver,
                               who: str) -> float:
    """``rmax_m``, or the farthest receiver range when it is None."""
    if rmax_m is not None:
        return float(rmax_m)
    ranges = getattr(receiver, 'ranges', None)
    if ranges is None or np.size(ranges) == 0:
        raise ConfigurationError(
            f"{who}: rmax_m is None and the receiver carries no ranges to "
            f"take RMax from.",
            remediation="Pass rmax_m= (metres), or a Receiver with ranges.")
    return float(np.max(np.asarray(ranges, dtype=float)))


def _write_kraken_family_env_file(
    filepath: Union[str, Path],
    env: Environment,
    source: Source,
    receiver,
    *,
    who: str,
    interp_ssp: Optional[str],
    frequencies: Optional[np.ndarray],
    n_mesh: int,
    rmax_m: Optional[float],
    c_low: float,
    c_high: float,
    topopt_extra: str = '',
) -> None:
    """The KRAKEN ENV deck Kraken and Scooter share: header, SSP, sediment
    layers, bottom, cLow/cHigh/RMax, source and receiver depths, and the
    broadband frequency vector when there is more than one frequency.
    ``topopt_extra`` is the extra TopOpt character only Scooter reads.
    """
    reject_unsupported_ssp_interp(who, interp_ssp)
    ssp_topopt = resolve_ssp_topopt(env, interp_ssp)
    surface_type, bottom_type = _env_boundary_types(env)
    rmax_m = _rmax_or_farthest_receiver(rmax_m, receiver, who)
    with open(filepath, 'w') as f:
        write_header(
            f, env, source,
            ssp_topopt=ssp_topopt,
            surface_type=surface_type,
            frequencies=frequencies,
            topopt_extra=topopt_extra,
            filepath=Path(filepath),
        )
        write_ssp_section(f, env, env.depth, n_mesh=n_mesh,
                          ssp_topopt=ssp_topopt,
                          frequency=float(source.frequencies[0]),
                          multi_frequency=(
                              frequencies is not None
                              and len(np.atleast_1d(frequencies)) > 1))
        write_layer_sections(f, env, env.depth, n_mesh=n_mesh)
        # Both engines read the 'A' halfspace line as ``zTemp, alphaR, betaR,
        # rhoR, alphaI, betaI`` (misc/ReadEnvironmentMod.f90:285), shear
        # attenuation included, and take cLow/cHigh/RMax from the
        # write_phase_speed_and_rmax record that follows it.
        write_bottom_section(
            f, env,
            bottom_type=bottom_type,
            filepath=Path(filepath),
        )
        write_phase_speed_and_rmax(
            f, rmax_m=rmax_m, c_low=c_low, c_high=c_high,
        )
        write_source_depths(f, source)
        write_receiver_depths(f, receiver)
        if frequencies is not None and len(np.atleast_1d(frequencies)) > 1:
            write_broadband_freqs(f, np.asarray(frequencies))


def write_kraken_env_file(
    filepath: Union[str, Path], env: Environment, source: Source, receiver, *,
    interp_ssp: Optional[str] = None,
    frequencies: Optional[np.ndarray] = None, n_mesh: Optional[int] = None,
    rmax_m: Optional[float] = None,
    c_low: float, c_high: float,
) -> None:
    """Write a Kraken environment file (.env).

    Kraken extends the KRAKEN ENV format with phase-speed limits (cLow,
    cHigh), a maximum range (RMax), and an optional broadband frequency
    vector (``TopOpt(6)='B'``, read after the source/receiver depth blocks).
    ``receiver`` is whatever carries the receiver depths (a ``Receiver`` or
    a depth array).

    The deck states ``env``: the SSP letter comes from ``interp_ssp``
    (:func:`resolve_ssp_topopt`, the name :class:`~uacpy.models.Kraken`
    takes) and the boundary letters from ``env.surface`` and the seabed
    half-space at range 0. ``n_mesh=0`` lets KRAKEN size each medium;
    ``c_low``/``c_high`` are the phase-speed window the deck states (m/s,
    required); ``rmax_m`` of None is the farthest receiver range. :class:`~uacpy.models.Kraken` passes its own
    ``rmax_m`` (5 % past the farthest receiver, ×3 for a band) and phase
    speeds.

    Parameters
    ----------
    filepath : str or Path
        Output deck path.
    env : Environment
        The environment the deck states.
    source : Source
        Source depths and frequency.
    receiver : Receiver or array_like
        The receiver, or its depths.
    interp_ssp : str, optional
        SSP connection scheme (:func:`resolve_ssp_topopt`); ``None`` is the
        engine's own choice.
    frequencies : array_like, optional
        A broadband frequency vector (Hz), written after the depth blocks.
    n_mesh : int, optional
        Mesh points per medium; ``0`` lets the engine size each one.
    rmax_m : float, optional
        The deck's RMax (m); ``None`` is the farthest receiver range.
    c_low, c_high : float
        The phase-speed window the deck states (m/s).
    """
    n_mesh = KRAKEN_N_MESH if n_mesh is None else n_mesh
    _write_kraken_family_env_file(
        filepath, env, source, receiver, who='write_kraken_env_file',
        interp_ssp=interp_ssp, frequencies=frequencies, n_mesh=n_mesh,
        rmax_m=rmax_m, c_low=c_low, c_high=c_high)


def write_scooter_env_file(
    filepath: Union[str, Path], env: Environment, source: Source,
    receiver: Receiver, *,
    interp_ssp: Optional[str] = None,
    frequencies: Optional[np.ndarray] = None, topopt_extra: str = '',
    n_mesh: Optional[int] = None, rmax_m: Optional[float] = None,
    c_low: float, c_high: float,
) -> None:
    """Write a Scooter environment file (.env).

    Scooter uses the KRAKEN ENV format plus cLow/cHigh, RMax, and shear
    support on the bottom halfspace 'A' line. It reads no receiver ranges —
    ``scooter.f90:158-176`` (``GetPar``) stops at ``ReadfreqVec`` and the
    ranges come from the ``.grn`` post-processing instead.

    The SSP and boundary letters, ``n_mesh``, ``c_low``/``c_high`` and
    ``rmax_m`` default as in :func:`write_kraken_env_file`. RMax sets
    Scooter's wavenumber sampling; :class:`~uacpy.models.Scooter` passes
    the farthest receiver times its ``rmax_factor``. ``topopt_extra``
    ``'0'`` turns off the stabilising attenuation (TopOpt(7)).

    Parameters
    ----------
    filepath : str or Path
        Output deck path.
    env : Environment
        The environment the deck states.
    source : Source
        Source depths and frequency.
    receiver : Receiver
        The receiver depths.
    interp_ssp : str, optional
        SSP connection scheme (:func:`resolve_ssp_topopt`); ``None`` is the
        engine's own choice.
    frequencies : array_like, optional
        A broadband frequency vector (Hz), written after the depth blocks.
    topopt_extra : str, optional
        ``'0'`` turns off the stabilising attenuation (TopOpt(7)).
    n_mesh : int, optional
        Mesh points per medium; ``0`` lets the engine size each one.
    rmax_m : float, optional
        The deck's RMax (m); ``None`` is the farthest receiver range.
    c_low, c_high : float
        The phase-speed window the deck states (m/s).
    """
    n_mesh = SCOOTER_N_MESH if n_mesh is None else n_mesh
    _write_kraken_family_env_file(
        filepath, env, source, receiver, who='write_scooter_env_file',
        interp_ssp=interp_ssp, frequencies=frequencies,
        topopt_extra=topopt_extra, n_mesh=n_mesh, rmax_m=rmax_m,
        c_low=c_low, c_high=c_high)


def write_sparc_env_file(
    filepath: Union[str, Path],
    env: Environment,
    source: Source,
    receiver: Receiver,
    *,
    interp_ssp: Optional[str] = None,
    output_mode: Optional[str] = None,
    n_mesh: Optional[int] = None,
    rmax_m: Optional[float] = None,
    c_low: float,
    c_high: float,
    pulse_type: str,
    freq_min: float,
    freq_max: float,
    n_time_samples: int,
    time_max: float,
    march_start: float,
    courant_factor: float,
) -> None:
    """Write a SPARC environment file (.env).

    SPARC extends the KRAKEN ENV format with an output-mode TopOpt char
    (R/D/S), time-domain pulse parameters, and time-output/integration
    blocks. Both boundaries are restricted to vacuum or rigid
    (``sparc.f90:101-104``), so no half-space row is ever written — and a
    deck that declared one without writing the row would hand the SSP mesh
    line to ``TopBot``.

    The SSP and boundary letters, ``n_mesh``, ``c_low``/``c_high`` and
    ``rmax_m`` default as in :func:`write_kraken_env_file`; ``output_mode``
    defaults to ``'R'`` as :class:`~uacpy.models.SPARC` does. The pulse
    band and the time window have no answer in ``env`` and are required;
    :class:`~uacpy.models.SPARC` sizes them from the source frequency and
    the receiver ranges.

    Parameters
    ----------
    filepath : str or Path
        Output deck path.
    env : Environment
        The environment the deck states.
    source : Source
        Source depths and frequency.
    receiver : Receiver
        The receiver.
    interp_ssp : str, optional
        SSP connection scheme (:func:`resolve_ssp_topopt`); ``None`` is the
        engine's own choice.
    output_mode : {'R', 'D', 'S'}, optional
        Horizontal array, vertical array or snapshot.
    n_mesh : int, optional
        Mesh points per medium; ``0`` lets the engine size each one.
    rmax_m : float, optional
        The deck's RMax (m); ``None`` is the farthest receiver range.
    c_low, c_high : float
        The phase-speed window the deck states (m/s).
    pulse_type : str
        The 4-character pulse code (see :class:`~uacpy.models.SPARC`).
    freq_min, freq_max : float
        The pulse band (Hz).
    n_time_samples : int
        Output time samples.
    time_max : float
        End of the output window (s).
    march_start : float
        Time (s) the march begins.
    courant_factor : float
        Safety factor on the marching time step.
    """
    n_mesh = SPARC_N_MESH if n_mesh is None else n_mesh
    output_mode = SPARC_OUTPUT_MODE if output_mode is None else output_mode
    reject_unsupported_ssp_interp('write_sparc_env_file', interp_ssp)
    ssp_code = resolve_ssp_topopt(env, interp_ssp)
    surface_type, bottom_type = _env_boundary_types(env)
    rmax_m = _rmax_or_farthest_receiver(rmax_m, receiver,
                                        'write_sparc_env_file')
    # sparc.f90:177 stops on a non-zero roughness in the SSP block —
    # "Rough interfaces not allowed" — for the surface and every sediment
    # layer; the half-space's, on the BotOpt line, it reads and ignores.
    rough = [('surface', float(env.surface.roughness))]
    rough += [(f'sediment layer {i}', float(layer.roughness))
              for column in env.bottom.columns
              for i, layer in enumerate(column.layers)]
    offenders = [f"{what} ({sigma:g} m)" for what, sigma in rough if sigma]
    if offenders:
        raise UnsupportedFeatureError(
            'SPARC',
            f"a rough {', '.join(offenders)} — GETPAR stops with 'Rough "
            f"interfaces not allowed' (sparc.f90:177) at exit 0. Set the "
            f"roughness to 0 for SPARC (the SPARC wrapper does so, with a "
            f"warning).")
    hs_sigma = float(env.bottom.halfspace_at(range=0.0).roughness)
    if hs_sigma:
        warnings.warn(
            f"write_sparc_env_file: the half-space roughness ({hs_sigma:g} m) "
            f"goes on the BotOpt line, which SPARC reads and ignores — the "
            f"deck is valid but the seabed is smooth to SPARC.",
            FallbackWarning, skip_file_prefixes=USER_FRAME_SKIP)
    bottom_code = BOUNDARY_CODES[bottom_type]
    for btype, boundary, carrier in ((surface_type, 'surface', env.surface),
                                     (bottom_type, 'bottom', env.bottom)):
        if BOUNDARY_CODES[btype] not in ('V', 'R'):
            raise UnsupportedFeatureError(
                'SPARC',
                f"a {btype.value} {boundary} — "
                f"GETPAR stops with 'SPARC only allows Vacuum or Rigid "
                f"boundary conditions' (sparc.f90:101-104). Got "
                f"{carrier!r}",
                alternatives=[
                    f"set the {boundary} to 'vacuum' (pressure-release) or "
                    f"'rigid'",
                    "Scooter or Kraken for a broadband run that keeps the "
                    "half-space, then transform to the time domain",
                ],
                alternatives_label='options',
            )

    with open(filepath, 'w') as f:
        # SPARC's TopOpt(5:5) is its output mode. The V/R guard above means
        # there is no top half-space row and no .trc to stage.
        write_header(f, env, source, ssp_code, surface_type,
                     pos5=output_mode, filepath=filepath)
        # One count per MEDIUM when a sequence is given: AT sizes each medium
        # separately (ReadEnvironmentMod.f90:101-112 takes each one's own
        # thickness and speed), so a single scalar broadcast to a thin sediment
        # layer over-resolves it by the ratio of the layer's thickness to the
        # water column's. Element 0 is the water column, the rest the writable
        # sediment layers.
        if isinstance(n_mesh, (list, tuple, np.ndarray)):
            counts = [int(n) for n in n_mesh]
            write_ssp_section(f, env, env.depth, n_mesh=counts[0],
                              ssp_topopt=ssp_code,
                              frequency=float(source.frequencies[0]))
            write_layer_sections(f, env, env.depth, n_mesh=counts[1:])
        else:
            write_ssp_section(f, env, env.depth, n_mesh=n_mesh,
                              ssp_topopt=ssp_code,
                              frequency=float(source.frequencies[0]))
            write_layer_sections(f, env, env.depth, n_mesh=n_mesh)

        # Bottom section (V or R only, so no halfspace params follow).
        sigma = env.bottom.halfspace_at(range=0.0).roughness
        f.write(f"'{bottom_code}' {sigma:.6f}\n")

        write_phase_speed_and_rmax(
            f, rmax_m=rmax_m, c_low=c_low, c_high=c_high,
        )

        write_source_depths(f, source)
        # misc/subtabulate.f90:24 gates the whole (first, last) expansion on
        # ``Nx >= 3``, so a one-depth block is read verbatim and needs no
        # padding value.
        write_receiver_depths(f, receiver)

        # Time-domain pulse parameters (SPARC-specific, come BEFORE ranges).
        f.write(f"'{pulse_type}'\n")
        f.write(f"{freq_min:.6f} {freq_max:.6f}\n")

        # Receiver ranges (come AFTER pulse info in SPARC). SubTab expands
        # "rmin rmax /" into a uniform vector, silently discarding non-uniform
        # ranges — write_receiver_ranges emits the full list so an N-entry
        # list is read verbatim, and refuses two ranges sharing a km token.
        write_receiver_ranges(f, receiver)

        # Output times. Read through ReadVector (Scooter/sparc.f90:159), so the
        # "first last /" pair is expanded by SubTab into n_time_samples uniformly
        # spaced times — which needs n_time_samples >= 3
        # (misc/subtabulate.f90:24,40); below that the two values are taken
        # verbatim.
        f.write(f"{n_time_samples}\n")
        f.write(f"0.0 {time_max:.6f} /\n")
        # Integration parameters: TSTART, TMULT, ALPHA, BETA, V
        # (Scooter/sparc.f90:168). The trailing three pin the finite-element
        # time march to its standard scheme: ALPHA = 0 is a lumped mass matrix,
        # BETA = 0 a standard explicit step (Scooter/sparc.f90:161-165), and
        # V = 0 the convection velocity — a moving medium is not part of
        # uacpy's Environment. ``doc/sparc.htm`` names all three and its own
        # sample deck ends in the same three zeros.
        f.write(f"{march_start:.6f} {courant_factor:.6f} 0.0 0.0 0.0\n")


def write_sparc_source_time_series(filepath, source, waveform, sample_rate,
                                   rows: int) -> None:
    """Write SPARC's ``STSFIL`` in the layout
    ``tslib/sourceMod.f90:97-117`` reads: ``waveform`` zero-padded to
    ``rows`` rows (a power of two when the binary band-passes the pulse,
    ``tslib/bandpassc.f90:24-25``).

    Record layout (list-directed reads, so the title is quoted)::

        'uacpy source time series'          ! :99  PulseTitle
        Nsd  SD(1) ... SD(Nsd)              ! :100 count, source depths
        t    s(1) ... s(Nsd)                ! :107 one row per sample
        ...                                 !      until end of file

    ``t`` is in seconds from 0 at ``1/sample_rate`` steps; the binary
    interpolates linearly between rows (``:144-183``) and drives a zero
    source outside ``[t_first, t_last]`` (``:172-174``), so the march,
    which starts at ``march_start`` (default -0.1 s) from a field at rest,
    meets the waveform from t = 0. The same series is written under
    every source depth of the deck, whose count must match the row
    width (``:100`` reads ``SD`` with the deck's ``NSz``). A leading
    ``'B'`` makes the binary play the file backwards (``:125-136``);
    the file is the same.

    The band-pass is circular (an FFT over the whole record), so a
    series should lead with zeros rather than start on its pulse: a
    pulse placed at row 0 loses about a fifth of its peak level to the
    filter's wrap at the window edge (measured 19 %).

    Parameters
    ----------
    filepath : str or Path
        Output path.
    source : Source
        Source whose depths the file lists.
    waveform : array_like
        The source series.
    sample_rate : float
        Sample rate (Hz) of ``waveform``.
    rows : int
        Rows written; ``waveform`` is zero-padded to it.
    """
    waveform = np.asarray(waveform, dtype=float)
    n_depths = int(np.atleast_1d(np.asarray(source.depths)).size)
    depths = np.atleast_1d(np.asarray(source.depths, dtype=float))
    series = np.zeros(int(rows))
    series[:waveform.size] = waveform
    times = np.arange(int(rows)) / sample_rate
    with open(filepath, 'w') as f:
        f.write("'uacpy source time series'\n")
        f.write(f"{n_depths} " + " ".join(f"{d:.6f}" for d in depths)
                + "\n")
        for t, v in zip(times, series):
            f.write(f"{t:.9e} " + " ".join([f"{v:.9e}"] * n_depths)
                    + "\n")


def write_bounce_input_file(
    filepath: Union[str, Path],
    env: Environment,
    source: Source,
    *,
    interp_ssp: Optional[str] = None,
    n_mesh,
    c_low: float,
    c_high: float,
    rmax_m: float,
    verbose: Union[bool, str] = False,
) -> None:
    """Write a BOUNCE input file (.env).

    The SSP letter comes from ``interp_ssp`` (:func:`resolve_ssp_topopt`)
    and the bottom letter from the seabed half-space at range 0.

    BOUNCE uses the KRAKEN ENV format plus cLow/cHigh and RMax, and does
    NOT read source/receiver depth blocks — its Fortran driver stops after
    RMax (bounce.f90). ``rmax_m`` is in metres, written as the km the deck
    holds, like the ``rmax_m`` of the Kraken/Scooter/SPARC writers.

    **The water column is deliberately omitted.** ``bounce.f90:178-179`` shoots
    the impedance up from the bottom half-space through *every* acoustic medium
    (``AcousticLayers``, :245-288) and ``bounce.f90:201`` forms ``RCmplx`` from
    the ``f``/``g`` reached at the top of medium 1, so including the ocean
    would return the reflection coefficient of water + seabed seen from above
    the sea surface — which ``doc/bounce.htm`` warns against in those words,
    and which is detectable because it makes the result depend on water depth.
    The sediment stack is therefore medium 1, and the top boundary is an
    ``'A'`` half-space carrying the water sound speed at the seafloor:
    ``bounce.f90:186-187`` takes the reference speed ``c0`` from ``HSTop%cP``
    (falling back to a hardcoded 1500 at :189 otherwise). ``Initialize`` also
    folds the top half-space speed into ``cMin`` (:139-146), which raises the
    ``cLow`` floor at :149.

    A seabed that is a bare half-space carries **no** medium: the deck declares
    ``NMedia = 0``, which ``doc/bounce.htm`` documents ("If you only have a
    halfspace, you can set NMedia to 0") and which makes ``f``/``g`` come
    straight from ``BCImpedance('BOT')`` at the seafloor — ``NPTS = SUM(N(1:0))
    = 0``, the medium loop does not run, ``FirstAcoustic`` stays 0 and
    ``AcousticLayers`` returns at :258. Any padding medium would move the plane
    ``R`` is referenced to and rotate the phase of every ``.brc``/``.irc`` row.

    ``n_mesh`` is passed through to :func:`write_layer_sections` (a scalar for
    every medium, or one count per writable layer).

    **Densities reach BOUNCE as ratios to the water's.** ``bounce.f90:201``
    forms ``R = -(f - i kz g)/(f + i kz g)`` with ``g = P'/rho`` referenced
    to a unit density — no ``HSTop`` density appears anywhere in the
    program — so the reference row's density is written as ``1`` (the
    number it would ignore) and every seabed density below is divided by
    ``env.water_density`` (``density_reference``), the same treatment the
    RAM codes get. Measured: the water row alone moved ``R`` by 0.0, the
    direct Scooter run of the same seabed by 2 dB.

    Parameters
    ----------
    filepath : str or Path
        Output deck path.
    env : Environment
        The environment the deck states.
    source : Source
        Source depths and frequency.
    interp_ssp : str, optional
        SSP connection scheme (:func:`resolve_ssp_topopt`); ``None`` is the
        engine's own choice.
    n_mesh : int
        Mesh points per medium; ``0`` lets BOUNCE size each one.
    c_low, c_high : float
        The phase-speed window the deck states (m/s).
    rmax_m : float
        The deck's RMax (m).
    verbose : bool or str, optional
        Logging gate passed through to ``log_message``.
    """
    filepath = Path(filepath)
    ssp_topopt = resolve_ssp_topopt(env, interp_ssp)
    _surface_type, bottom_type = _env_boundary_types(env)
    seafloor = float(env.depth)
    layered = env.bottom.is_layered and not env.bottom.is_range_dependent
    layers = writable_layers(env.bottom) if layered else []

    # Reference medium for the incident wave: the water at the seafloor.
    water_top = BoundaryProperties(
        acoustic_type='half-space',
        sound_speed=float(np.atleast_1d(env.ssp.sound_speed_at(seafloor))[0]),
        density=1.0,
        attenuation=0.0,
    )
    bounce_env = env.copy()
    bounce_env.surface = Surface(nodes=[water_top])

    with open(filepath, 'w') as f:
        write_header(
            f, bounce_env, source,
            ssp_topopt=ssp_topopt,
            surface_type=BoundaryType.HALF_SPACE,
            n_media_override=len(layers),
            filepath=filepath,
            verbose=verbose,
        )
        if layers:
            halfspace_top = write_layer_sections(
                f, bounce_env, seafloor, n_mesh=n_mesh,
                density_reference=env.water_density)
        else:
            halfspace_top = seafloor
        write_bottom_section(
            f, bounce_env,
            bottom_type=bottom_type,
            filepath=filepath,
            halfspace_depth=halfspace_top,
            verbose=verbose,
            density_reference=env.water_density,
        )
        # Phase velocity bounds (define angular coverage) and RMax (km),
        # through the same writer the sibling decks use.
        # bounce.f90:49 makes the tabulated-angle count
        # NkTab = INT( 1000 * RMax * ( kMax - kMin ) / 2 pi ) directly
        # proportional to RMax, which the writer emits at the same millimetre
        # resolution as the rest of the deck's ranges.
        write_phase_speed_and_rmax(
            f, rmax_m=rmax_m, c_low=c_low, c_high=c_high,
        )
