"""The raw output-file parsers the engines read.

An engine reads its output file as the file holds it, through a parser here;
a user reads the record the public reader in :mod:`uacpy.io` builds from the
same file (:func:`~uacpy.io.read_oast_tl`, :func:`~uacpy.io.read_oasp_trf`,
:func:`~uacpy.io.read_tl_grid`, :func:`~uacpy.io.read_pcomplex_grid`). A
package-internal module (``tests/_internal_modules.py``): its names are
public across uacpy's packages and not part of the user API.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TypedDict, Union

import numpy as np

from uacpy.core.exceptions import FileFormatError
from uacpy.core.units import km_to_m
from uacpy.io._fortran_helpers import (
    PARSE_ERRORS, _bound_counts, typed_format_error,
)
from uacpy.io.oases_reader import (
    _OAST_TL_RANGE_TAG, _curve_values, _match_label_depths,
    _oast_curve_slots, _parse_oast_plp, _read_oasp_trf_binary,
    _read_plt_blocks,
)
from uacpy.io.ramsurf_reader import _read_grid

__all__ = ['OastTlFields', 'parse_oast_tl', 'parse_oasp_trf',
           'parse_tl_grid', 'parse_pcomplex_grid']


class OastTlFields(TypedDict):
    """The named fields :func:`parse_oast_tl` hands back; the function's
    own Returns section carries the shapes, which depend on ``NFREQ``.

    A ``Tuple[...]`` return type here is read by the caller in
    ``uacpy/models/oases/oast.py`` as ``oast_out['depths']`` — a string
    subscript of something declared a tuple."""

    tl: np.ndarray
    depths: np.ndarray
    ranges: np.ndarray
    metadata: Dict[str, Any]


def parse_oast_tl(
    filepath: Union[str, Path],
    receiver_depths: Optional[np.ndarray] = None,
) -> OastTlFields:
    """
    Read OAST transmission loss output on its native grid, as the OAST and
    OASS models read it.

    OAST outputs two files:
    - .plp: Plot metadata (ASCII with binary markers) - contains grid info
    - .plt: Actual curve data (pure ASCII, one value per line)

    OAST writes TL (in dB) directly to disk; this reader returns the
    native ``(n_depths, n_ranges)`` TL grid plus the depth and range
    axes. Resampling onto a user receiver grid is the caller's job —
    use :meth:`Field.resample_to` after wrapping.

    The TL curves are *not* necessarily first in the ``.plt``: options that
    add curves inside the same receiver loop (``'I'`` → PLINTGR, ``'a'`` →
    PLSPECT, unoast31.f:590-605) write theirs ahead of PLTLOS's, and ``'A'``
    / ``'D'`` add more afterwards. Every curve is described by a ``.plp``
    record whose 6-character tag names it (``<param>TLRAN`` for TL vs range,
    oasfun22.f:330), so the TL curves are selected by tag rather than by
    position.

    Parameters
    ----------
    filepath : str or Path
        Path to .plt or .plp file (base name works for both)
    receiver_depths : ndarray
        Receiver depth axis (m) the deck asked for. It is what the ``RD:``
        labels of the ``.plp`` are matched against, so the returned depths
        carry the caller's own precision rather than the labels' 0.1 m; it
        is *not* assumed to be the axis OAST plotted, which the deck's
        ``IDINC`` may decimate. Omitted, the depths are the ``RD:`` labels
        as printed (0.1 m); a ``.plp`` without labels then cannot be
        attributed to depths and raises.

    Returns
    -------
    result : dict
        Named fields, like the sibling OASES readers:

        - ``'tl'`` : ndarray — transmission loss on the OAST native grid.
          Shape ``(n_depths, n_ranges_native)`` for a single-frequency run.
          A multi-frequency deck (``NFREQ > 1``) writes one curve per
          plotted receiver *per frequency* and yields shape
          ``(n_freq, n_depths, n_ranges_native)``, the frequency axis
          ascending as the run swept it.
        - ``'depths'`` : ndarray — the depths OAST actually plotted, which
          is a subset of ``receiver_depths`` on a deck whose receiver record
          carries ``IDINC > 1``.
        - ``'ranges'`` : ndarray — OAST's native range grid in metres.
          **Shape ``(n_ranges_native,)`` for a single frequency and
          ``(n_freq, n_ranges_native)`` for a sweep**: OAST rebuilds the
          grid inside its frequency loop, so ``DX`` scales as ``1/f`` and
          the frequencies do not share one range axis.
        - ``'metadata'`` : dict — ``{'oast_grid_shape': …,
          'n_frequencies': n_f, 'plotted_frequencies': ndarray}``.
          ``'plotted_frequencies'`` comes from the ``Freq:`` labels (``F7.1``, oasfun22.f:368) and is
          absent from a ``.plp`` that carries no labels.

    Raises
    ------
    FileFormatError
        If the ``.plp`` file is missing or cannot be parsed (OAST chooses
        its own range grid via FFT-based sampling, so the native grid
        cannot be reconstructed without it), if it carries no TL-vs-range
        curve, if its ``Freq:``/``RD:`` labels do not form a full grid, or
        if the receivers of one frequency disagree about the range axis.
    """
    filepath = Path(filepath)

    # OAST writes the curve data on unit 20 and the plot description on unit
    # 19 (``bin/oast``: FOR019=.plp, FOR020=.plt); an unmapped unit 20 lands
    # in .020 instead.
    # Whichever of the three the caller named (or a bare root), the trio
    # is the same: each is the given path with its own suffix.
    plt_file = filepath.with_suffix('.plt')
    plp_file = filepath.with_suffix('.plp')
    f020_file = filepath.with_suffix('.020')

    # Try to find TL data file (prefer .plt, then .020)
    if plt_file.exists():
        tl_data_file = plt_file
    elif f020_file.exists():
        tl_data_file = f020_file
    else:
        raise FileFormatError(
            f"OAST TL data file not found. Checked: {plt_file}, {f020_file}.",
            remediation="Add the 'T' option (PLTL) to the OAST option string "
                        "and re-run: OAST writes its TL data file only when "
                        "TL plotting is enabled.",
        )

    # Parse .plp file to get OAST's native range grid. The grid is
    # mandatory: OAST chooses its own ranges via FFT-based sampling, so
    # without .plp we have no way to know what range each TL value
    # corresponds to. Raise rather than fabricate.
    if not plp_file.exists():
        raise FileFormatError(
            f"OAST .plp grid file not found: {plp_file}. "
            "Without it the native range grid cannot be reconstructed."
        )
    curves = _parse_oast_plp(plp_file)
    tl_curves = [c for c in curves if c['tag'].endswith(_OAST_TL_RANGE_TAG)]

    if not tl_curves:
        tags = sorted({c['tag'] for c in curves})
        raise FileFormatError(
            f"{plp_file} carries no TL-vs-range curve "
            f"(no '*{_OAST_TL_RANGE_TAG}' record); curves present: {tags}. "
            f"Add the 'T' option (PLTL) to the OAST option string."
        )

    # One TL curve per (output parameter, receiver). uacpy returns a single
    # TL grid, so more than one output parameter is ambiguous — say so rather
    # than pick a slab.
    params = sorted({c['tag'][0] for c in tl_curves})
    if len(params) > 1:
        raise FileFormatError(
            f"{plp_file} carries TL curves for {len(params)} output "
            f"parameters {params} (OASES optpar letters N,W,U,V,R,B,S); "
            f"uacpy returns a single TL grid. Request one output parameter "
            f"in the OAST option string."
        )

    freq_axis, depth_axis, slots = _oast_curve_slots(
        plp_file, tl_curves, receiver_depths)
    n_freq = len(freq_axis) if freq_axis is not None else (
        max(s[0] for s in slots) + 1)
    if depth_axis is None:
        depths_oast = np.asarray(receiver_depths, dtype=float)
    elif receiver_depths is None:
        depths_oast = np.asarray(depth_axis, dtype=float)
    else:
        depths_oast = _match_label_depths(depth_axis, receiver_depths)
    n_depths_oast = len(depths_oast)

    blocks = _read_plt_blocks(tl_data_file)
    n_expected = sum(1 for c in curves if c['index'] is not None)
    if len(blocks) != n_expected:
        raise FileFormatError(
            f"{tl_data_file} holds {len(blocks)} data blocks but "
            f"{plp_file} describes {n_expected}; the pair is inconsistent "
            f"(truncated or interleaved run).",
            remediation="Delete both files and re-run the case: the .plp and "
                        "its data file must come from one run, so a stale "
                        "file left in the working directory by an earlier "
                        "run produces exactly this mismatch.",
        )

    n_ranges_oast = tl_curves[0]['n']
    # The curve count sizes the TL grid allocation; one G13.6 value occupies
    # at least 2 bytes on disk, so no product of counts can exceed the .plt
    # size in half-bytes — reject a garbage 'N' before np.empty runs.
    _bound_counts(tl_data_file, tl_data_file.stat().st_size, 2,
                  n_freq=n_freq, n_depths=n_depths_oast,
                  n_ranges=n_ranges_oast)

    # OAST recomputes DLRAN = 2*pi/(NWVNO*DLWVNO) inside the frequency loop
    # (unoast31.f:480, DLWVNO proportional to FREQ at :475), so RSTEP scales
    # as 1/f and each frequency owns its own range axis. The point count LF
    # is clamped to NWVNO at :491, which is what makes the axes look alike:
    # equal N, halved DX. Within one frequency every receiver shares the
    # grid; across frequencies they need not.
    grids: Dict[int, Tuple[float, float]] = {}
    tl_oast = np.empty((n_freq, n_depths_oast, n_ranges_oast), dtype=float)
    filled = np.zeros((n_freq, n_depths_oast), dtype=bool)
    for i, (curve, (i_freq, i_depth)) in enumerate(zip(tl_curves, slots)):
        if curve['n'] != n_ranges_oast:
            raise FileFormatError(
                f"{plp_file}: TL curve {i} has {curve['n']} range samples, "
                f"curve 0 has {n_ranges_oast} — the curves do not share one "
                f"range grid."
            )
        grid = (curve['xoff'], curve['dx'])
        if grids.setdefault(i_freq, grid) != grid:
            raise FileFormatError(
                f"{plp_file}: TL curve {i} starts at XOFF={curve['xoff']} km "
                f"in steps of DX={curve['dx']} km, but another curve of the "
                f"same frequency uses {grids[i_freq]} — the receivers of one "
                f"frequency do not share a range grid."
            )
        if curve['index'] is None:
            raise FileFormatError(
                f"{plp_file}: TL curve {i} ({curve['tag']}) parameterises "
                f"both axes (DX={curve['dx']}, DY={curve['dy']}), so PLTWRI "
                f"wrote no {tl_data_file.suffix} block for it "
                f"(oasgun21.f:658-660) and it carries no TL."
            )
        tl_oast[i_freq, i_depth] = _curve_values(
            blocks[curve['index']], curve, tl_data_file)
        filled[i_freq, i_depth] = True

    # The label grid factors on its totals, so two curves sharing one
    # (Freq:, RD:) pair leave another slot untouched — and untouched here is
    # whatever np.empty allocated.
    if not filled.all():
        missing = [(freq_axis[j] if freq_axis is not None else j,
                    float(depths_oast[k]))
                   for j, k in zip(*np.nonzero(~filled))]
        raise FileFormatError(
            f"{plp_file}: no TL curve for {missing} (frequency Hz, depth m); "
            f"the run's curves do not cover every frequency at every plotted "
            f"receiver."
        )

    range_rows = np.array([
        km_to_m(grids[j][0] + np.arange(n_ranges_oast) * grids[j][1])
        for j in range(n_freq)
    ])
    if n_freq == 1:
        tl_oast = tl_oast[0]
        ranges_oast = range_rows[0]
    else:
        ranges_oast = range_rows

    metadata = {
        'oast_grid_shape': tl_oast.shape,
        'n_frequencies': n_freq,
    }
    if freq_axis is not None:
        metadata['plotted_frequencies'] = np.asarray(freq_axis, dtype=float)
    return {
        'tl': tl_oast,
        'depths': depths_oast,
        'ranges': ranges_oast,
        'metadata': metadata,
    }


def parse_oasp_trf(
    filepath: Union[str, Path],
    receiver_depths: np.ndarray,
) -> Dict:
    """
    Read an OASP transfer function file (.trf) as the file holds it: the
    output parameter's transfer function, as OASP and OASSP read it.

    OASP outputs transfer functions for postprocessing with PP module.
    These are complex frequency-domain responses.

    Only the Fortran-unformatted binary layout exists in practice: OASP's
    ``bintrf`` is a DATA-statement ``.true.`` (``unoasp22.f:1166``) that
    nothing in the tree reassigns, so ``TRFHEAD`` always takes its
    ``FORM='UNFORMATTED'`` branch (``oasiun23.f:844-846``).

    Parameters
    ----------
    filepath : str or Path
        Path to .trf file
    receiver_depths : ndarray
        Receiver depth axis (m), taken verbatim from the caller that wrote
        the deck. The ``.trf`` cannot supply it: TRFHEAD writes the explicit
        depth list only for ``IR < 0`` (oasiun23.f:870-877), but INREC has
        already flipped IR positive in COMMON /VARS1/ (oaseun31.f:1185,
        oases/src/compar.f:69-70) by the time TRFHEAD runs, so the header carries only
        RD, RDLOW and ``|IR|`` — a uniform grid — whatever the deck asked
        for. The header's count is cross-checked against this axis. It is
        required because the header cannot say whether the deck's array
        was uniform: a non-uniform ``[10, 20, 60, 90]`` would come back from
        the header as ``[10, 36.7, 63.3, 90]``.

    Returns
    -------
    data : dict
        Dictionary containing:
        - 'title': str, simulation title
        - 'option': str, output option used
        - 'freq': ndarray, frequencies (Hz)
        - 'ranges': ndarray, ranges (m)
        - 'depths': ndarray, receiver depths (m)
        - 'transfer_function': ndarray, complex transfer functions
                              shape (n_freq, n_range, n_depth)
        - 'source_depth': float, source depth (m)
        - 'center_frequency': float, center frequency (Hz)

    Notes
    -----
    Format follows the OASES PULSETRF binary specification from
    trford.f/oasiun23.f.

    Examples
    --------
    >>> data = parse_oasp_trf('pulse.trf', receiver_depths=[20., 50., 80.])
    >>> trf = data['transfer_function']  # shape: (n_freq, n_range, n_depth)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileFormatError(
            f"OASP transfer function file not found: {filepath}.",
            remediation="Check the OASP run completed: the .trf is its "
                        "primary output, so a missing one means the binary "
                        "stopped before writing it and its stdout carries "
                        "the reason.",
        )

    depths = np.atleast_1d(np.asarray(receiver_depths, dtype=float))

    try:
        return _read_oasp_trf_binary(filepath, depths)
    except PARSE_ERRORS as e:
        raise FileFormatError(
            f"Failed to read OASP transfer function file {filepath} as "
            f"Fortran-unformatted PULSETRF ({type(e).__name__}: {e}).",
            remediation="Verify the run finished writing the .trf; OASES "
                        "only ever writes the binary layout (bintrf is "
                        "hardwired .true., unoasp22.f:1166).",
        ) from e


@typed_format_error
def parse_tl_grid(
    filepath: Union[str, Path],
    *,
    dr: float,
    ndr: int,
    dz: float,
    ndz: int,
    depth_index_offset: int = 0,
):
    """A Collins ``tl.grid`` as ``(ranges, depths, tl)``: the range and
    depth axes (m) and the ``(n_depths, n_ranges)`` TL (dB); see
    :func:`~uacpy.io.read_tl_grid` for the grid parameters."""
    return _read_grid(filepath, 'read_tl_grid', 'f8', dr, ndr, dz, ndz,
                      depth_index_offset)


@typed_format_error
def parse_pcomplex_grid(
    filepath: Union[str, Path],
    *,
    dr: float,
    ndr: int,
    dz: float,
    ndz: int,
    depth_index_offset: int = 0,
):
    """A uacpy-patched ``pcomplex.bin`` as ``(ranges, depths, p)``: the
    axes (m) and the complex PE envelope, ``(n_depths, n_ranges)``, before
    the carrier and the Hankel phase; see
    :func:`~uacpy.io.read_pcomplex_grid`."""
    return _read_grid(filepath, 'read_pcomplex_grid', 'c16', dr, ndr, dz,
                      ndz, depth_index_offset)
