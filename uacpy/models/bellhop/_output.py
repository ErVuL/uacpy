"""What one Bellhop run wrote, read back and assembled in the package's
conventions: the one output its run type writes, the line-source level and
phase, the sign of the ``.shd`` pressure, the caller's depth and range axes
restored with no-data cells NaN, provenance and output paths stamped."""

import re
import warnings
from pathlib import Path

import numpy as np

from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.bathymetry import mask_below_seafloor
from uacpy.core.exceptions import ModelExecutionError, NumericsWarning
from uacpy.core.results import ResultStack
from uacpy.io.oalib_reader import (
    read_arr_file, read_prt, read_ray_file, read_shd_file,
)
from uacpy.models._extract import (
    attach_output_paths, mask_unresolvable_depths, stamp_result,
)
from uacpy.models._launch import require_output
from uacpy.models.bellhop._backend import arrivals_need_merge
from uacpy.models.bellhop._tables import grid_is_paired

# A line source carries the 2-D Green's function's exp(-i*pi/4). In AT's
# e^{+i*omega*t} convention (index.htm:801) the free-space line field is
# (-i/4)*H0^(2)(kR) ~ (1/4)*sqrt(2/(pi*k*R))*exp(-i*(kR + pi/4)) for large
# kR — a phase of -pi/4 against the e^{-ikR} carrier — while both AT scale
# factors are purely real: ScalePressure's line branch (influence.f90:784,
# factor = -4*sqrt(pi)*const with const = -1) and WriteArrivals' line branch
# (ArrMod.f90:103-104, factor = 4*sqrt(pi)). Applied once, on the .shd field
# and on every arrival read from the .arr, it keeps COHERENT_TL, ARRIVALS,
# BROADBAND and TIME_SERIES on one phase reference.
_LINE_SOURCE_PHASE = -np.pi / 4.0
# The same 4*sqrt(pi) sets the LEVEL: Bellhop's free-space line field is
# 4*sqrt(pi)/sqrt(R), Kraken's and Scooter's 1/sqrt(k0*R) — a 4*sqrt(pi*k0)
# gap on one supported source type. The package normalises every engine's
# line source to unit amplitude at 1 m (JKPS §5.2.2's p/p0(1); see
# ``_conventions._line_source_unit_at_1m``): here that is dividing by 4*sqrt(pi),
# on the .shd field and on every arrival amplitude the arrivals routes use.
_LINE_SOURCE_LEVEL = 1.0 / (4.0 * np.sqrt(np.pi))


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


#: The arrivals capacity every engine logs to the ``.prt``
#: (``Bellhop/bellhop.f90:224``, ``bellhopcuda/src/mode/arr.hpp:99``).
_PRT_MAX_ARRIVALS = re.compile(r'Maximum # of arrivals\s*=\s*(\d+)')


def read_bellhop_output(work_dir, base_name, run_type, *, model_name,
                        grid_type, backend, exe):
    """Read the one output ``_BELLHOP_OUTPUT[run_type]`` names, as its io
    reader returns it. A missing or empty output means the binary died
    silently, and the raised error carries the ``.prt`` tail with the
    actual cause."""
    work_dir = Path(work_dir)
    output_key, output_suffix, reader = _BELLHOP_OUTPUT[run_type]
    output_file = require_output(
        model_name, [work_dir / f'{base_name}{output_suffix}'],
        what=f'a {run_type} output ({output_suffix})',
        prt_base=base_name, work_dir=work_dir,
    )
    # The .arr header reports the full ``Pos%NRz``
    # (``ReadEnvironmentBell.f90:591``) while its body carries only
    # ``NRz_per_range`` depth blocks — 1 for an irregular grid
    # (``bellhop.f90:202-206,329``, ``ArrMod.f90:101-102``). Nothing in
    # the file distinguishes the two, so the reader has to be told.
    result = (reader(output_file, grid_type=grid_type,
                     merge=arrivals_need_merge(backend=backend, exe=exe,
                                               model_name=model_name))
              if run_type == 'A' else reader(output_file))
    if run_type == 'A':
        warn_if_arrival_table_filled(result, work_dir, base_name)
    return result


def warn_if_arrival_table_filled(result, work_dir, base_name) -> None:
    """Warn when a receiver cell holds as many arrivals as the engine's
    table has room for, i.e. the table was full and later arrivals were
    dropped.

    No engine says so itself: Fortran replaces the weakest stored arrival
    once the cell is full (``Bellhop/ArrMod.f90:49-59``), the
    multithreaded ports keep whichever arrived first
    (``bellhopcuda/src/arrivals.hpp:113-119``), and both write only the
    capacity, ``( Maximum # of arrivals = N )``, to the ``.prt``.
    """
    prt = read_prt(Path(work_dir) / f'{base_name}.prt') or ''
    found = _PRT_MAX_ARRIVALS.search(prt)
    if not found:
        return
    capacity = int(found.group(1))
    fullest = 0
    for slab in (result.slabs if isinstance(result, ResultStack)
                 else [result]):
        for by_depth in (slab.by_receiver or []):
            for cells in by_depth:
                for cell in cells:
                    fullest = max(fullest, int(cell['n_arrivals']))
    if fullest < capacity:
        return
    warnings.warn(
        f"Bellhop ARRIVALS: a receiver cell holds {fullest} arrivals, the "
        f"engine's full capacity (Maximum # of arrivals = {capacity} in "
        f"the .prt); arrivals beyond it were dropped without notice. Use "
        f"fewer receivers per run (the capacity is the engine's storage "
        f"divided by the receiver count) or fewer beams.",
        NumericsWarning, skip_file_prefixes=USER_FRAME_SKIP)


def trim_padded_ranges(result, run_type, lo, hi):
    """Drop the range columns
    :func:`~uacpy.models.bellhop._plan.pad_receiver_ranges` added — on
    the raw result, before any assembly reads its axes."""
    slabs = result.slabs if isinstance(result, ResultStack) else [result]
    if run_type == 'A':
        for slab in slabs:
            slab._trim_ranges(lo, hi)
        return result
    for slab in slabs:
        if 'range' in slab.coords:
            axis = list(slab.coords).index('range')
            slab.data = np.take(slab.data, np.arange(lo, hi), axis=axis)
            slab.coords['range'] = np.asarray(slab.coords['range'])[lo:hi]
    return result


def assemble(env, source, receiver, result, work_dir, base_name, run_type,
             trim=None, *, model_name, provenance, backend, cleanup,
             grid_type, log):
    """Turn what :func:`read_bellhop_output` read into the tagged
    :class:`Result` the caller gets: the padded range columns trimmed,
    the depth axis restored, the source-type phase and level
    conventions applied, provenance stamped and the output paths
    attached. Nothing here launches anything."""
    output_key, output_suffix, _ = _BELLHOP_OUTPUT[run_type]
    if trim is not None:
        result = trim_padded_ranges(result, run_type, *trim)
    if run_type == 'A':
        result = restore_arrival_depths(result, receiver, env,
                                        model_name=model_name,
                                        grid_type=grid_type)
        # The law Im(tau) carries, so an Arrivals result scales its
        # absorption to other frequencies as the synthesis here does
        # (Arrivals.absorption).
        for slab in (result.slabs
                     if isinstance(result, ResultStack) else [result]):
            slab._record_absorption(env.absorption)
    if run_type == 'A' and source.source_type == 'line':
        # The .arr amplitudes carry ArrMod.f90:104's purely real
        # 4*sqrt(pi): bring them to the package's unit-at-1 m
        # line-source level and give every arrival the Green's
        # function's -pi/4 (in radians, the reader's unit) here, so the
        # Arrivals result, the broadband and time-series syntheses
        # built from it and the .shd field share one phase reference.
        for slab in (result.slabs
                     if isinstance(result, ResultStack) else [result]):
            slab._scale_paths(_LINE_SOURCE_LEVEL, _LINE_SOURCE_PHASE)

    # AT's ScalePressure (influence.f90:757-795) carries const = -1
    # into the point-source branch (factor = const/sqrt(r)), so the
    # .shd field is inverted relative to the e^{i(wt-kr)} convention
    # Kraken and Scooter report. Undo it here so every uacpy model
    # shares one phase reference. The line-source branch
    # (factor = -4*sqrt(pi)*const) already cancels the sign, and the
    # arrivals path computes its own positive factor
    # (ArrMod.f90:103-111) and takes the line source's
    # _LINE_SOURCE_PHASE in the loop above.
    # Measured against the exact 2-D solution that correction is pi/4
    # to 0.01 deg, and once applied the line-source residual equals
    # the point-source beam bias exactly (4.78 deg vs 4.79 deg).
    _shd_phase = {'point': -1.0 + 0j,
                  'line': _LINE_SOURCE_LEVEL * np.exp(1j * _LINE_SOURCE_PHASE)}
    if run_type in ('C', 'I', 'S'):
        _corr = _shd_phase[source.source_type]
        for _slab in (result.slabs
                      if isinstance(result, ResultStack) else [result]):
            # Upcast (the .shd payload is complex64) so every uacpy
            # engine returns one dtype, as Kraken and Scooter do.
            _slab.data = np.asarray(_slab.data, dtype=np.complex128) * _corr

        # BELLHOP clamps any receiver below the deck's bottom
        # boundary onto it (misc/SourceReceiverPositions.f90:136-139)
        # and the .shd then carries the clamped depth axis with the
        # boundary row repeated — no field is evaluated at the asked
        # depth. Restore the requested depth axis with NaN there,
        # then NaN below the local seafloor too (a range-dependent
        # .bty leaves sub-seafloor receivers above the deck depth
        # unclamped but ray-free), matching the RAM / Scooter /
        # SPARC below-domain convention. The irregular grid
        # (RunType(5:5)='I') carries no depth axis: its pairs are
        # masked in place.
        def _restore_depths_and_mask(slab):
            if list(slab.coords) == ['range']:
                return mask_paired_receivers_below_seafloor(
                    slab, receiver, env, model_name=model_name)
            if list(slab.coords) != ['depth', 'range']:
                return slab
            slab = mask_unresolvable_depths(
                model_name, slab, receiver, float(env.depth))
            return slab.mask_below_seafloor(env.bathymetry)

        if isinstance(result, ResultStack):
            result.slabs = [_restore_depths_and_mask(s)
                            for s in result.slabs]
        else:
            result = _restore_depths_and_mask(result)

        if run_type in ('I', 'S'):
            # An incoherent / semi-coherent beam sum is a magnitude
            # (ScalePressure takes SQRT(REAL(U)), influence.f90:779), so
            # its zero phase is an artefact. Store real dB TL, as Kraken
            # does for its INCOHERENT_TL, so the result claims only what
            # it has and `.phase` refuses on every engine.
            for _slab in (result.slabs if isinstance(result, ResultStack)
                          else [result]):
                _slab.data = np.asarray(_slab.dB, dtype=float)
                # The unit was fixed at construction, from the complex
                # payload; the payload is its dB view from here on.
                _slab._unit = 'dB'

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
        # produced it. Same goes for the receiver geometry: the
        # requested depths are stamped as read, so an eigenray found
        # at a clamped depth (misc/SourceReceiverPositions.f90:136-139)
        # is labelled with the depth that was asked for.
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
        stamp_result(
            model_name, provenance, slab, source, backend=backend,
            frequencies=f0,
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
        attach_output_paths(
            slab, work_dir, base_name, cleanup=cleanup,
            primary_files=((output_key, output_suffix),),
        )

    log("Simulation complete")
    return result


def mask_paired_receivers_below_seafloor(field, receiver, env, *,
                                         model_name):
    """NaN the no-data cells of a paired-grid (``grid_type='I'``) result
    and put the requested depths on ``aux_coords['receiver_depth']``, along
    the range axis that holds the pairs.

    Pair ``i`` is ``(depths[i], ranges[i])``. BELLHOP clamps a depth below
    the deck bottom onto it (``misc/SourceReceiverPositions.f90:136-139``)
    and reports the clamped depth, and a pair under a shoaling ``.bty`` is
    ray-free; both are no-data, exactly as on the rectilinear grid
    (``_restore_depths_and_mask`` in :func:`assemble` /
    :func:`restore_broadband_depth_axis`).
    The range axis (axis 0) is the pair index, so the one mask serves the
    1-D TL slab and the ``(range, frequency | time)`` broadband Field.
    """
    depths = np.atleast_1d(np.asarray(receiver.depths, dtype=float))
    ranges = np.asarray(field.coords['range'], dtype=float)
    if depths.size != ranges.size:
        raise ModelExecutionError(
            model_name, return_code=0, stdout=None,
            stderr=(f"{model_name} returned {ranges.size} receiver "
                    f"pairs for {depths.size} requested; the paired "
                    f"depths cannot be reattached."),
        )
    # A depth below the deck bottom (env.depth, the deepest seafloor) is
    # below the seafloor at every range, so the one mask covers both.
    field.data = mask_below_seafloor(field.data, depths, ranges,
                                     env.bathymetry, paired=True)
    return field.replace(aux_coords={**field.aux_coords,
                                     'receiver_depth': ('range', depths)})


def restore_arrival_depths(result, receiver, env, *, model_name,
                           grid_type):
    """Reattach the requested depth axis to an ARRIVALS result and empty
    (``n_arrivals = 0``) the cells BELLHOP clamped onto the deck bottom
    (``misc/SourceReceiverPositions.f90:136-139``) — the arrivals
    counterpart of the TL modes' ``_restore_depths_and_mask`` in
    :func:`assemble`. On a paired grid (``grid_type='I'``) depth ``i``
    is range cell ``i``."""
    depths = np.atleast_1d(np.asarray(receiver.depths, dtype=float))
    below = depths > float(env.depth)
    paired = grid_is_paired(grid_type)
    for slab in (result.slabs
                 if isinstance(result, ResultStack) else [result]):
        if slab.receiver_depths.size != depths.size:
            raise ModelExecutionError(
                model_name, return_code=0, stdout=None,
                stderr=(f"{model_name} returned "
                        f"{slab.receiver_depths.size} receiver depths "
                        f"for {depths.size} requested; the depth axis "
                        f"cannot be reattached."),
            )
        slab._empty_cells(below, by='range_idx' if paired else 'depth_idx')
        slab.receiver_depths = depths
    return result


def restore_broadband_depth_axis(field, receiver, env, *, model_name):
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
    ``_restore_depths_and_mask`` step the TL modes apply in
    :func:`assemble`. The irregular grid carries no depth axis and takes
    :func:`mask_paired_receivers_below_seafloor` instead.
    """
    field = mask_unresolvable_depths(
        model_name, field, receiver, float(env.depth))
    field.data = mask_below_seafloor(field.data, field.coords['depth'],
                                     field.coords['range'], env.bathymetry)
    return field
