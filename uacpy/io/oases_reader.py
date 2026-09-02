"""
OASES Output File Readers

This module provides functions for reading output files from OASES models:
- OAST: .plp/.plt files (transmission loss)
- OASN: .xsm files (covariance matrices), .rpo files (replicas)
- OASP: .trf files (complex transfer functions)
- OASR: .rco/.trc files (reflection coefficients vs slowness / angle)
- OASS/OASSP: the mean-field .rhs header the scattering modules consume;
  their own outputs reuse the OAST .plt, OASN .xsm and OASP .trf readers

OASES (Ocean Acoustic and Seismic Exploration Synthetics, per the source
banner at ``src/oasgun21.f:4``) was developed by Henrik Schmidt at MIT.

References:
    Schmidt, H. OASES User Guide and Reference Manual, bundled under
    ``third_party/oases/doc``; its title page reads Version 3.1
    (``doc/oases.tex:53``) while the vendored source distribution is 2.1
    (``third_party/oases/README:2``).
"""

import struct
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple, TypedDict, Union

import numpy as np

from uacpy.core.exceptions import FileFormatError, UnsupportedFeatureError
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.io._fortran_helpers import (
    PARSE_ERRORS,
    _bound_counts,
    fortran_float,
    read_fortran_record as _read_fortran_record,
    detect_endian,
)
from uacpy.core.units import km_to_m


def _decode_fortran_title(raw: bytes) -> str:
    """Decode a Fortran ``CHARACTER`` title field to a clean string.

    Not every module initialises the title it writes. OASS declares it
    (``unoass21.f:34``) but leaves it unset before ``PUTXSM`` copies it into
    the ``.xsm`` header (``oasmun21_bin.f:364``), so the field arrives as NUL
    bytes rather than the blanks a Fortran ``CHARACTER`` is padded with when
    it *has* been assigned. ``str.strip()`` removes the blanks and keeps the
    NULs, which is how a title becomes 32 invisible bytes that then flow into
    ``Result.metadata['title']`` and print as nothing at all.

    NULs and blanks come off in one pass rather than two: a field that is
    part text, part NUL, part blank ends on whichever of the two the record
    happened to stop with, so stripping one class and then the other leaves
    the first class behind wherever it sits inside the second.

    Every OASES title read goes through here so the three readers cannot
    disagree about it.
    """
    return raw.decode('ascii', errors='ignore').strip('\x00 \t\r\n\v\f')


class OastTL(TypedDict):
    """The named fields :func:`read_oast_tl` hands back; the function's own
    Returns section carries the shapes, which depend on ``NFREQ``.

    A ``Tuple[...]`` return type here is read by the caller in
    ``uacpy/models/oases.py`` as ``oast_out['depths']`` — a string subscript
    of something declared a tuple."""

    tl: np.ndarray
    depths: np.ndarray
    ranges: np.ndarray
    metadata: Dict[str, Any]


def read_oast_tl(
    filepath: Union[str, Path],
    receiver_depths: np.ndarray,
) -> OastTL:
    """
    Read OAST transmission loss output on its native grid.

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
        ``IDINC`` may decimate.

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
          'n_frequencies': n_f, 'frequencies': ndarray}``. ``'frequencies'``
          comes from the ``Freq:`` labels (``F7.1``, oasfun22.f:368) and is
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
    if filepath.suffix == '.plt':
        plt_file = filepath
        plp_file = filepath.with_suffix('.plp')
        f020_file = filepath.with_suffix('.020')
    elif filepath.suffix == '.plp':
        plp_file = filepath
        plt_file = filepath.with_suffix('.plt')
        f020_file = filepath.with_suffix('.020')
    elif filepath.suffix == '.020':
        f020_file = filepath
        plt_file = filepath.with_suffix('.plt')
        plp_file = filepath.with_suffix('.plp')
    else:
        # No extension given, try all
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
            f"OAST TL data file not found. Checked: {plt_file}, {f020_file}",
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
    depths_oast = (_match_label_depths(depth_axis, receiver_depths)
                   if depth_axis is not None
                   else np.asarray(receiver_depths, dtype=float))
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
    # (unoast31.f:481, DLWVNO proportional to FREQ at :477), so RSTEP scales
    # as 1/f and each frequency owns its own range axis. The point count LF
    # is clamped to NWVNO at :492, which is what makes the axes look alike:
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
        metadata['frequencies'] = np.asarray(freq_axis, dtype=float)
    return {
        'tl': tl_oast,
        'depths': depths_oast,
        'ranges': ranges_oast,
        'metadata': metadata,
    }


def _unique_in_order(values: list) -> list:
    """The distinct entries of ``values``, first-appearance order kept.

    A ``dict`` keeps insertion order, so this is the hashed form of the
    membership scan the same walk would otherwise do per value. A ``set`` is
    not a substitute: the order is the receiver axis
    :func:`_match_label_depths` walks when it hands out depths greedily, so
    reordering it moves curves onto other receivers rather than raising.
    """
    return list(dict.fromkeys(values))


def _oast_curve_slots(plp_file: Path, tl_curves: list, receiver_depths):
    """Attribute each TL curve to its ``(frequency, depth)`` slot.

    Returns ``(freq_labels, depth_labels, slots)``. ``slots[i]`` is the
    ``(i_freq, i_depth)`` pair curve ``i`` belongs in; the two label lists
    are the axes those indices run over, or ``None`` when the ``.plp``
    carries no labels to key off.

    PLTLOS writes ``NLAB = 3`` labels into every plot block — ``Freq:``,
    ``SD:``, ``RD:`` (oasfun22.f:334-337) — which is the only per-curve
    record of *which* frequency and *which* receiver a curve came from.
    Counting instead is what the deck's ``IDINC`` breaks: PLTLOS runs only
    for ``MOD( NREC-1, INTF ) == 0`` (unoast31.f:630) with ``INTF`` the 4th
    field of the OASTL receiver record (oaseun31.f:1156), so a decimated run
    writes fewer curves than it has receivers and every curve past the first
    lands on the wrong depth.

    The labels are printed at ``F7.1`` / ``F9.1`` (oasfun22.f:368-370), so
    two frequencies within 0.05 Hz — or two receivers within 0.05 m —
    collapse onto one label. A label set that no longer factors into a full
    grid has collided that way, and a ``.plp`` carrying no labels at all did
    not come from PLTLOS. Both fall back to the positional walk
    (frequency-major: the NFREQ loop at unoast31.f:388 wraps the receiver
    loop at :584) behind a warning naming which reading was used, since that
    walk is right for every run that does not decimate.
    """
    freq_labels = [c['labels'].get('Freq') for c in tl_curves]
    depth_labels = [c['labels'].get('RD') for c in tl_curves]

    if all(v is not None for v in freq_labels + depth_labels):
        freq_axis = _unique_in_order(freq_labels)
        depth_axis = _unique_in_order(depth_labels)
        # Collisions only ever merge two curves onto one slot, never split
        # one, so a product below the curve count is the whole test.
        if len(tl_curves) == len(freq_axis) * len(depth_axis):
            # Both axes are already distinct, so a position lookup keyed on
            # the label is the ``.index()`` scan without the per-curve walk.
            freq_pos = {value: i for i, value in enumerate(freq_axis)}
            depth_pos = {value: i for i, value in enumerate(depth_axis)}
            return freq_axis, depth_axis, [
                (freq_pos[f], depth_pos[d])
                for f, d in zip(freq_labels, depth_labels)
            ]
        reason = (
            f"its {len(tl_curves)} TL curves carry {len(freq_axis)} distinct "
            f"'Freq:' and {len(depth_axis)} distinct 'RD:' labels, which do "
            f"not form a full grid — the F7.1 / F9.1 label fields "
            f"(oasfun22.f:368-370) cannot separate them")
    else:
        reason = ("its TL curves carry none of the 'Freq:'/'RD:' labels "
                  "PLTLOS writes (oasfun22.f:334-337)")

    n_depths = len(receiver_depths)
    n_freq, remainder = divmod(len(tl_curves), n_depths)
    if remainder or n_freq == 0:
        raise FileFormatError(
            f"{plp_file} carries {len(tl_curves)} TL-vs-range curves, "
            f"which is not a whole multiple of the {n_depths} receiver "
            f"depths requested, and {reason}. OAST writes one curve per "
            f"plotted receiver per frequency (unoast31.f:388, :629-640), so "
            f"the run does not correspond to this receiver array."
        )
    warnings.warn(
        f"{plp_file}: {reason}, so the curves are attributed to frequencies "
        f"and depths by position. That reading is wrong for a deck whose "
        f"receiver record decimates with IDINC > 1 (unoast31.f:630, "
        f"oaseun31.f:1156), which writes fewer curves than the deck has "
        f"receivers.",
        UserWarning,
        skip_file_prefixes=USER_FRAME_SKIP,
    )
    return None, None, [divmod(i, n_depths) for i in range(len(tl_curves))]


#: Half of the ``F9.1`` quantum PLTLOS prints the receiver depth label at
#: (``oasfun22.f:370``): the widest a label can sit from the depth it names.
_PLP_DEPTH_LABEL_TOLERANCE_M = 0.05


def _match_label_depths(depth_labels: list, receiver_depths) -> np.ndarray:
    """Full-precision depths for the ``RD:`` labels of the plotted curves.

    The label is an ``F9.1`` field, so it names a receiver to 0.1 m. Each
    label is matched back to the caller's own receiver axis and answered with
    that value where one lies within half a print quantum, so a decimated run
    returns the depths that were asked for rather than their printed
    roundings. A label with no such receiver — a ``.plp`` from a different
    deck — is returned as printed.
    """
    axis = np.asarray(receiver_depths, dtype=float).ravel()
    available = list(range(axis.size))
    matched = []
    for label in depth_labels:
        near = [k for k in available
                if abs(axis[k] - label) <= _PLP_DEPTH_LABEL_TOLERANCE_M]
        if near:
            available.remove(near[0])
            matched.append(float(axis[near[0]]))
        else:
            matched.append(float(label))
    return np.asarray(matched, dtype=float)


#: ``.plp`` curve tag PLTLOS writes for TL vs range: ``optpar(INR)//'TLRAN'``
#: (oasfun22.f:330). PLDAV/PTLDEP/PLINTGR/PLSPECT use TLDAV/TLDEP/INTGR/SPECT.
_OAST_TL_RANGE_TAG = 'TLRAN'

#: Column at which PLPWRI's ``FORMAT(1H ,I8,10X,A40)`` / ``(1H ,G15.6,3X,A40)``
#: put the field name; both leave the value in columns 0..18.
_PLP_LABEL_COL = 19

#: Records PLPWRI writes between the label list and ``NC``: XLEN, YLEN,
#: IGRID, XLEFT, XRIGHT, XINC, XDIV, XTXT, XTYP, YDOWN, YUP, YINC, YDIV,
#: YTXT, YTYP (oasgun21.f:616-630).
_PLP_AXIS_RECORDS = 15

#: ``OPTION(2)`` of the record that closes the file (unoast31.f:914-915).
_PLP_END_TAG = 'PLTEND'

#: The five PLTWRI fields, in the order oasgun21.f:653-657 writes them.
_PLTWRI_FIELDS = ('N', 'XOFF', 'DX', 'YOFF', 'DY')

#: The labelled quantities PLTLOS puts in the ``NLAB`` A16 records ahead of
#: the axis block — ``'Freq:',F7.1,' Hz$'`` / ``'SD:',F9.1,' m$'`` /
#: ``'RD:',F9.1,' m$'`` (oasfun22.f:335-337, :368-370).
_PLP_LABEL_KEYS = ('Freq', 'SD', 'RD')


def _plp_labels(lines: list, i: int, n_lab: int) -> Dict[str, float]:
    """Numeric ``Freq:`` / ``SD:`` / ``RD:`` labels of one plot block.

    PLPWRI writes each label as ``FORMAT(1H ,A16)`` (oasgun21.f:779), and
    the plot programs terminate the text with ``'$'``. A record that is not
    one of :data:`_PLP_LABEL_KEYS` followed by a number is skipped: the
    label list is free text that other routines fill differently.
    """
    labels: Dict[str, float] = {}
    for line in lines[i:i + n_lab]:
        key, sep, tail = line.strip().rstrip('$').partition(':')
        if not sep or key.strip() not in _PLP_LABEL_KEYS:
            continue
        tokens = tail.split()
        if not tokens:
            continue
        try:
            labels[key.strip()] = float(tokens[0])
        except ValueError:
            continue
    return labels


def _plp_value(plp_file: Path, lines: list, i: int, expect: str) -> float:
    """Numeric value of a labelled PLPWRI/PLTWRI record, checking its label.

    Every record PLPWRI and PLTWRI write with a numeric edit descriptor
    carries its own name in the A40 tail (oasgun21.f:662-663), so the label
    is a free check that the positional walk is still in step.
    """
    if i >= len(lines):
        raise FileFormatError(
            f"{plp_file}: file ends inside a plot block; expected the "
            f"{expect!r} record."
        )
    line = lines[i]
    label = line[_PLP_LABEL_COL:].strip()
    if label != expect:
        raise FileFormatError(
            f"{plp_file}: expected the {expect!r} record at line {i + 1}, "
            f"found {label!r}."
        )
    value = line[:_PLP_LABEL_COL].strip()
    try:
        return float(value)
    except ValueError as e:
        raise FileFormatError(
            f"{plp_file}: {expect} field is not numeric ({value!r})."
        ) from e


def _check_plp_count(plp_file: Path, i: int, name: str, value: int,
                     n_lines: int) -> None:
    """Reject a ``.plp`` DO-loop bound that cannot describe the file.

    The positional walk advances by counts the file supplies, so a negative
    one moves the cursor backwards and the walk never reaches ``PLTEND``.
    """
    if value < 0 or i + value > n_lines:
        raise FileFormatError(
            f"{plp_file}: line {i + 1} declares {name}={value}, which no "
            f"{n_lines}-line file can hold; the file is not an OASES .plp, "
            f"or the run was truncated."
        )


def _parse_oast_plp(plp_file: Path) -> list:
    """Parse an OASES ``.plp`` into its ordered list of curve descriptors.

    The file opens with the MODU record (unoast31.f:277) and then repeats a
    strictly positional plot block: PLPWRI writes the ``OPTION`` line, PTIT
    and TITLE as A80, ``NLAB``, ``NLAB`` A16 labels, 15 axis records and
    ``NC`` (oasgun21.f:610-631); PLTWRI then writes ``NC`` groups of five
    records — ``N``, ``XOFF``, ``DX``, ``YOFF``, ``DY`` (oasgun21.f:653-657).
    A record whose ``OPTION(2)`` is ``PLTEND`` closes the file.

    The walk is positional because it has to be: PTIT and TITLE are A80
    fields of arbitrary user text (``FORMAT(1H ,A80)``, oasgun21.f:634), so
    no column or substring distinguishes them from the records around them.
    The counts come from the file itself.

    Each PLTWRI record corresponds, in order, to one blank-line-separated
    block of the ``.plt``.

    Returns
    -------
    list of dict
        One entry per PLTWRI record, in file order, with keys ``tag`` (the
        6-character ``OPTION(2)``, e.g. ``'NTLRAN'``), ``labels`` (the
        block's numeric ``Freq:`` / ``SD:`` / ``RD:`` records, see
        :func:`_plp_labels`), ``index`` (position in
        the ``.plt`` block sequence), ``n``, ``xoff``, ``dx``, ``yoff``,
        ``dy``. Offsets/increments are in the plot's own x units — km for a
        TL-vs-range curve, whose axis PLTLOS labels ``'Range (km)$'``
        (oasfun22.f:338), hence the km→m conversion in :func:`read_oast_tl`.
    """
    # The xopt fields PLPWRI appends to the OPTION line come from COMMON
    # /PLXOPT/ (oasgun21.f:601-602), which nothing in the tree assigns, so
    # those CHARACTER*3 slots reach the file uninitialised — decode leniently.
    text = plp_file.read_bytes().decode('ascii', errors='replace')
    lines = text.split('\n')
    while lines and not lines[-1].strip():
        lines.pop()

    curves: list = []
    n_blocks = 0
    i = 1  # MODU record
    while i < len(lines):
        option = lines[i]
        # PLPWRI's OPTION line: ' ' + PROGNM(A6) + OPTION(2)(A6) + 6 x ',xxx'.
        tag = option[7:13].strip()
        if tag == _PLP_END_TAG:
            break
        if not tag:
            raise FileFormatError(
                f"{plp_file}: line {i + 1} carries no PLPWRI OPTION tag; the "
                f"file is not an OASES .plp, or the run was truncated."
            )
        i += 3                                    # OPTION, PTIT, TITLE
        n_lab = int(_plp_value(plp_file, lines, i, 'NUMBER OF LABELS'))
        # NLAB and NC are the file's own DO-loop bounds (oasgun21.f:614,
        # :653). A negative NLAB walks the cursor backwards over records it
        # has already read, which never terminates.
        _check_plp_count(plp_file, i, 'NLAB', n_lab, len(lines))
        labels = _plp_labels(lines, i + 1, n_lab)
        i += 1 + n_lab + _PLP_AXIS_RECORDS
        n_curves = int(_plp_value(plp_file, lines, i, 'NC'))
        _check_plp_count(plp_file, i, 'NC', n_curves, len(lines))
        i += 1
        for _ in range(n_curves):
            fields = {name: _plp_value(plp_file, lines, i + k, name)
                      for k, name in enumerate(_PLTWRI_FIELDS)}
            i += len(_PLTWRI_FIELDS)
            dx, dy = fields['DX'], fields['DY']
            # PLTWRI emits a .plt block only when it tabulates at least one
            # axis; a fully parameterised curve (DX and DY both non-zero)
            # contributes nothing but the blank terminator.
            writes_block = (dx == 0.0) or (dy == 0.0)
            curves.append({
                'tag': tag,
                'labels': labels,
                'index': n_blocks if writes_block else None,
                'n': int(fields['N']),
                'xoff': fields['XOFF'],
                'dx': dx,
                'yoff': fields['YOFF'],
                'dy': dy,
            })
            n_blocks += int(writes_block)
    else:
        raise FileFormatError(
            f"{plp_file}: no {_PLP_END_TAG} record — the run did not finish "
            f"writing its plot file.",
            remediation="Re-run the case and read the binary's stdout: a "
                        "plot file without its end record means the model "
                        "died mid-write, so the real failure is in that "
                        "output rather than in this file.",
        )

    if not curves:
        raise FileFormatError(
            f"{plp_file}: no PLTWRI curve records found — the file is not an "
            f"OASES .plp, or the run wrote no plot data (an OAST run plots "
            f"TL only under the 'T' option, PLTL)."
        )
    return curves


def _read_plt_blocks(plt_file: Path) -> list:
    """Split an OASES ``.plt`` into its blank-line-separated value blocks.

    PLTWRI writes each array with ``FORMAT(1H ,G13.6)`` and terminates the
    record with a bare ``write(20,*)`` (oasgun21.f:658-660), so one blank
    line closes each curve.
    """
    blocks: list = []
    current: list = []
    with open(plt_file, 'r') as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                if current:
                    blocks.append(np.array(current, dtype=float))
                    current = []
                continue
            try:
                current.append(float(stripped))
            except ValueError as e:
                raise FileFormatError(
                    f"{plt_file}:{lineno}: expected a single G13.6 value, "
                    f"got {stripped!r}."
                ) from e
    if current:
        blocks.append(np.array(current, dtype=float))
    return blocks


def _curve_values(block: np.ndarray, curve: Dict, plt_file: Path) -> np.ndarray:
    """The ordinate samples of one PLTWRI block.

    PLTWRI writes the abscissa only when ``DX == 0`` and the ordinate only
    when ``DY == 0`` (oasgun21.f:658-659), so a block holds N or 2N values
    depending on which axes are tabulated rather than parameterised.
    """
    n = curve['n']
    if curve['dy'] != 0.0:
        raise FileFormatError(
            f"{plt_file}: curve {curve['tag']} has DY={curve['dy']}, so no "
            f"ordinate was written."
        )
    n_arrays = int(curve['dx'] == 0.0) + 1
    if len(block) != n * n_arrays:
        raise FileFormatError(
            f"{plt_file}: curve {curve['index']} ({curve['tag']}) holds "
            f"{len(block)} values; .plp declares N={n} with "
            f"{n_arrays} tabulated array(s)."
        )
    return block[-n:]


def read_oasn_covariance(
    filepath: Union[str, Path]
) -> Dict:
    """
    Read OASN covariance matrix file (.xsm format)

    The .xsm file contains covariance matrices computed by OASN for
    ambient noise modeling and matched field processing.

    File format:
    - Binary direct-access file with 8-byte records
    - First 10 records: header (title, frequencies, array size)
    - Remaining records: complex covariance matrix data

    Parameters
    ----------
    filepath : str or Path
        Path to .xsm file

    Returns
    -------
    data : dict
        Dictionary containing:
        - 'title': str, simulation title
        - 'n_receivers': int, number of receivers
        - 'n_frequencies': int, number of frequencies
        - 'freq_min': float, minimum frequency (Hz)
        - 'freq_max': float, maximum frequency (Hz)
        - 'freq_delta': float, frequency increment (Hz)
        - 'surface_noise_level': float, surface noise level (dB)
        - 'white_noise_level': float, white noise level (dB)
        - 'covariance': ndarray, shape (n_freq, n_rcv, n_rcv)
                        Complex covariance matrices

    Notes
    -----
    Each record holds one COMPLEX: PUTXSM opens the file with ``RECL = 2 *
    ldaun`` (``oasmun21_bin.f:346``), where DASREC (``oashun21.f:667-693``)
    probes how many RECL units a 4-byte word takes on the host compiler, so
    the record is 8 bytes whether RECL counts words or bytes.

    Examples
    --------
    >>> data = read_oasn_covariance('test.xsm')
    >>> cov = data['covariance']  # shape: (n_freq, n_rcv, n_rcv)
    >>> print(f"Covariance for {data['n_receivers']} receivers")
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileFormatError(
            f"OASN covariance file not found: {filepath}",
            remediation="Re-run OASN with 'N' in the option string: the "
                        "covariance matrices reach the .xsm file only under "
                        "that letter.",
        )

    # One COMPLEX per record. PUTXSM opens the file with ``RECL = 2 * ldaun``
    # (oasmun21_bin.f:346), where DASREC (oashun21.f:667-693) probes how many
    # RECL units a 4-byte word takes on the host compiler — 1 where RECL
    # counts words, 4 where it counts bytes — so the record is two words, 8
    # bytes, either way.
    recl = 8

    try:
        with open(filepath, 'rb') as f:
            # Probe the byte order from the first int32 (n_rcv at record 5).
            f.seek(4 * recl)
            probe = f.read(4)
            endian = detect_endian(probe, source=f'read_oasn_covariance:{filepath.name}')

            # Read header (first 10 records)
            # Records 1-4 are four consecutive 8-character slices of one
            # CHARACTER*80 TITLE (oasmun21_bin.f:364-367 writes TITLE(1:8),
            # (9:16), (17:24), (25:32)) — slices, not tokens, so concatenate
            # the raw bytes and strip once. Stripping each slice would delete
            # any space that falls on a record boundary.
            f.seek(0)
            title = _decode_fortran_title(f.read(4 * recl))

            # Record 5: NRCV, NFREQ (2 integers)
            f.seek(4 * recl)
            n_rcv, n_freq = struct.unpack(endian + 'ii', f.read(8))

            # Bound the n_freq*n_rcv*n_rcv covariance allocation against the
            # file size before the strided read / ljust below.
            f.seek(0, 2)
            file_size = f.tell()
            _bound_counts(filepath, file_size, recl,
                          n_rcv=n_rcv, n_freq=n_freq, n_rcv2=n_rcv)

            # Record 6: ENSEM, IZERO (ensemble size, dummy) — skipped

            # Record 7: FREQ1, FREQ2 (2 floats)
            f.seek(6 * recl)
            freq1, freq2 = struct.unpack(endian + 'ff', f.read(8))

            # Record 8: DELFRQ, ZERO (frequency increment)
            f.seek(7 * recl)
            delfrq, _ = struct.unpack(endian + 'ff', f.read(8))

            # Record 9: SSLEV, WNLEV (surface and white noise levels)
            f.seek(8 * recl)
            sslev, wnlev = struct.unpack(endian + 'ff', f.read(8))

            # Record 10: ZERO, ZERO (reserved)
            # Skip

            # Read covariance matrices. Data starts at record 11 and runs
            # NRCV*NRCV records per frequency (``SRTREC = 11 + (IFR-1) *
            # NRCV*NRCV``, oasmun21_bin.f:390-399), one complex value
            # (re, im float32) at the head of each ``recl``-byte record. The
            # matrix is flattened column-major — OASES addresses element
            # (ircv, jrcv) as ``IRCV + (JRCV-1)*NRCV`` (oasnun22.f:1174) — so
            # the receiver index of the first subscript runs fastest. A
            # structured dtype with ``itemsize=recl`` strides over the records
            # in a single read.
            n_total = n_freq * n_rcv * n_rcv
            f.seek(10 * recl)
            rec_dt = np.dtype({
                'names': ['re', 'im'],
                'formats': [endian + 'f4', endian + 'f4'],
                'itemsize': recl,
            })
            # The final record may carry only its 8-byte payload (no
            # padding to ``recl``); pad the buffer so the strided view
            # still covers ``n_total`` records.
            buf = f.read(n_total * recl)
            if len(buf) < (n_total - 1) * recl + 8:
                raise FileFormatError(
                    f"{filepath}: truncated covariance data — expected "
                    f"{n_total} records of {recl} bytes, got {len(buf)} bytes"
                )
            buf = buf.ljust(n_total * recl, b'\x00')
            flat = np.frombuffer(buf, dtype=rec_dt, count=n_total)
            vals = (flat['re'] + 1j * flat['im']).astype(np.complex64)
            # C-order reshape puts the fastest axis last, i.e. ircv; transpose
            # to (ifreq, ircv, jrcv). The matrix is Hermitian
            # (``CORRNS(JI)=CONJG(CORRNS(IJ))``, oasnun22.f:1148), so getting
            # this backwards would silently conjugate every cross-spectrum.
            covariance = vals.reshape(n_freq, n_rcv, n_rcv).transpose(0, 2, 1).copy()

        return {
            'title': title,
            'n_receivers': n_rcv,
            'n_frequencies': n_freq,
            'freq_min': freq1,
            'freq_max': freq2,
            'freq_delta': delfrq,
            'surface_noise_level': sslev,
            'white_noise_level': wnlev,
            'covariance': covariance
        }

    except (FileFormatError, UnsupportedFeatureError):
        raise
    except PARSE_ERRORS as e:
        raise FileFormatError(f"Failed to read OASN covariance file {filepath}: {e}") from e


def read_oasn_replicas(
    filepath: Union[str, Path]
) -> Dict:
    """
    Read OASN replica field file (.rpo format)

    The .rpo file contains complex array responses (replicas) for
    matched field processing, computed over a grid of source positions.

    File format:
    - Binary sequential file
    - Header: title, frequencies, array geometry, replica grid
    - Data: complex replicas for each frequency and grid point

    Parameters
    ----------
    filepath : str or Path
        Path to .rpo file

    Returns
    -------
    data : dict
        Dictionary containing:
        - 'title': str, simulation title
        - 'n_receivers': int, number of receivers
        - 'n_frequencies': int, number of frequencies
        - 'freq_min': float, minimum frequency (Hz)
        - 'freq_max': float, maximum frequency (Hz)
        - 'freq_delta': float, frequency increment (Hz)
        - 'z_min', 'z_max', 'n_z': replica depth grid (m)
        - 'x_min', 'x_max', 'n_x': replica x-offset grid (km)
        - 'y_min', 'y_max', 'n_y': replica y-offset grid (km)
        - 'receiver_positions': ndarray, shape (n_rcv, 3) [x, y, z] in m
        - 'receiver_types': ndarray, receiver types
        - 'receiver_gains': ndarray, receiver gains (dB)
        - 'replicas': ndarray, shape (n_freq, n_z, n_x, n_y, n_rcv)
                     Complex replica fields

    Notes
    -----
    The grid fields are reported in the units the deck states them in, which
    are not the same for all three axes: depth in m, x/y offsets in km
    (``doc/oasn.tex:109-111``). Receiver coordinates are all in m
    (``doc/oasn.tex:47-51``).

    Examples
    --------
    >>> data = read_oasn_replicas('test.rpo')
    >>> replicas = data['replicas']  # shape: (n_freq, n_z, n_x, n_y, n_rcv)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileFormatError(
            f"OASN replica file not found: {filepath}",
            remediation="Re-run OASN with 'R' in the option string: the "
                        "replica fields reach the .rpo file only under that "
                        "letter.",
        )

    try:
        with open(filepath, 'rb') as f:
            head = f.read(4)
            f.seek(0)
            endian = detect_endian(
                head, source=f'read_oasn_replicas:{filepath.name}',
            )

            # Read title (CHARACTER*80, oasmun21_bin.f:506)
            title = _decode_fortran_title(
                _read_fortran_record(f, raw=True, endian=endian))

            # Read NRCV, NFREQ
            n_rcv, n_freq = _read_fortran_record(f, 'ii', endian=endian)

            # Read FREQ1, FREQ2, DELFRQ
            freq1, freq2, delfrq = _read_fortran_record(f, 'fff',
                                                        endian=endian)

            # Read replica grid: ZMINR, ZMAXR, NZR
            z_min, z_max, n_z = _read_fortran_record(f, 'ffi', endian=endian)

            # Read XMINR, XMAXR, NXR
            x_min, x_max, n_x = _read_fortran_record(f, 'ffi', endian=endian)

            # Read YMINR, YMAXR, NYR
            y_min, y_max, n_y = _read_fortran_record(f, 'ffi', endian=endian)

            # Bound every header count before the receiver / replica arrays
            # are sized off them. Each replica record is 16 bytes on disk
            # (2 markers + re + im); use that as the per-item floor for the
            # (n_freq, n_z, n_x, n_y, n_rcv) product.
            cur = f.tell()
            f.seek(0, 2)
            file_size = f.tell()
            f.seek(cur)
            _bound_counts(filepath, file_size, 16,
                          n_freq=n_freq, n_z=n_z, n_x=n_x, n_y=n_y, n_rcv=n_rcv)

            # Read receiver positions and properties
            receiver_positions = np.zeros((n_rcv, 3))
            receiver_types = np.zeros(n_rcv, dtype=int)
            receiver_gains = np.zeros(n_rcv)

            for i in range(n_rcv):
                # PUTREP writes RAN, TRAN, DEP, IRTYP, GAIN per receiver
                # (oasmun21_bin.f:515-516).
                x, y, z, itype, gain = _read_fortran_record(
                    f, 'fffif', endian=endian)
                receiver_positions[i] = [x, y, z]
                receiver_types[i] = itype
                # INPRCV converts the deck's dB column in place before any
                # output is written (oasnun22.f:99
                # `GAIN(I)=10.0**(GAIN(I)/20.0)`), so the file holds a linear
                # amplitude factor. Invert it to report dB, the unit the deck
                # and oasn.tex:51 use.
                receiver_gains[i] = 20.0 * np.log10(gain) if gain > 0 else -np.inf

            # Each replica is a Fortran sequential record
            # ``[marker][re im][marker]`` (16 bytes), written contiguously by
            # nested loops over (ifreq, iz, ix, iy, ircv) with ircv innermost
            # (``doc/oasn.tex:681-686``). Read the whole block in one strided
            # pass.
            n_total = n_freq * n_z * n_x * n_y * n_rcv
            rep_dt = np.dtype([
                ('m1', endian + 'i4'),
                ('re', endian + 'f4'),
                ('im', endian + 'f4'),
                ('m2', endian + 'i4'),
            ])
            flat = np.fromfile(f, dtype=rep_dt, count=n_total)
            if flat.size < n_total:
                raise FileFormatError(
                    f"{filepath}: truncated replica data — expected "
                    f"{n_total} records, got {flat.size}"
                )
            if np.any(flat['m1'] != 8) or np.any(flat['m2'] != 8):
                raise FileFormatError(
                    f"{filepath}: unexpected replica record layout — "
                    "Fortran record markers are not the expected 8-byte "
                    "payload length"
                )
            vals = (flat['re'] + 1j * flat['im']).astype(np.complex64)
            replicas = vals.reshape(n_freq, n_z, n_x, n_y, n_rcv)

        return {
            'title': title,
            'n_receivers': n_rcv,
            'n_frequencies': n_freq,
            'freq_min': freq1,
            'freq_max': freq2,
            'freq_delta': delfrq,
            'z_min': z_min,
            'z_max': z_max,
            'n_z': n_z,
            'x_min': x_min,
            'x_max': x_max,
            'n_x': n_x,
            'y_min': y_min,
            'y_max': y_max,
            'n_y': n_y,
            'receiver_positions': receiver_positions,
            'receiver_types': receiver_types,
            'receiver_gains': receiver_gains,
            'replicas': replicas
        }

    except (FileFormatError, UnsupportedFeatureError):
        raise
    except PARSE_ERRORS as e:
        raise FileFormatError(f"Failed to read OASN replica file {filepath}: {e}") from e


#: The option letter that selects each 1-based ``IOUT``/NPAR slot in OASP's
#: GETOPT (unoasp22.f:884-919: N→1 V→2 H→3 R→5 K→6 S→7). TRFHEAD stores the
#: slot numbers, not letters (oasiun23.f:855-861), so rendering them back as
#: something a user could type needs this table. Slot 4 has no OASP letter.
_OASP_OUTPUT_PARAM_LETTERS = {1: 'N', 2: 'V', 3: 'H', 5: 'R', 6: 'K', 7: 'S'}

#: The *physical-quantity* abbreviations PLTLOS builds its six-character .plp
#: curve tag from (``optpar``, oasfun22.f:322-328). A different table from the
#: option letters above — it agrees only at slots 1, 5 and 7 — and used only
#: to match ``.plp`` tags.
_OASES_OUTPUT_PARAM_LETTERS = ('N', 'W', 'U', 'V', 'R', 'B', 'S')


def read_oasp_trf(
    filepath: Union[str, Path],
    receiver_depths: np.ndarray,
) -> Dict:
    """
    Read OASP transfer function file (.trf format)

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
        for. The header's count is cross-checked against this axis.

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
    >>> data = read_oasp_trf('pulse.trf', receiver_depths=[20., 50., 80.])
    >>> trf = data['transfer_function']  # shape: (n_freq, n_range, n_depth)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileFormatError(
            f"OASP transfer function file not found: {filepath}",
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


def _read_oasp_trf_binary(filepath: Path, receiver_depths: np.ndarray) -> Dict:
    """Read OASES PULSETRF binary file (Fortran UNFORMATTED).

    Inferred record layout (oasiun23.f:844-898 + trford.f:159-192) — each
    record is bracketed by 4-byte length markers::

        1.  CHARACTER*8  FILEID         ('PULSETRF')
        2.  CHARACTER*6  PROGNM
        3.  INTEGER      NOUT
        4.  INTEGER      IPARM(1..NOUT)
        5.  CHARACTER*80 TITLE
        6.  CHARACTER*1  SIGNN
        7.  REAL*4       FREQS
        8.  REAL*4       SD
        9.  REAL*4 RD, REAL*4 RDLOW, INTEGER IR
        9a. (only if IR<0) REAL*4 RDC(|IR|)
        10. REAL*4 R0, REAL*4 RSPACE, INTEGER NPLOTS
        11. INTEGER NX, INTEGER LX, INTEGER MX, REAL*4 DT
        12. INTEGER      ICDR
        13. REAL*4       OMEGIM
        14. INTEGER      MSUFT
        15. INTEGER      ISROW
        16. INTEGER      INTTYP
        17-18. INTEGER   IDUMMY (x2)
        19-23. REAL*4    DUMMY  (x5)

    Data records (NF = MX-LX+1 frequency bins)::

        for ifr:     for is:      for m:
         for jrh:     for jrv:    REC: COMPLEX*8 CFFX(1..NOUT)

    Default TRF files use single-precision complex (COMPLEX*8). Little-endian
    on x86 by default.
    """
    with open(filepath, 'rb') as f:
        probe = f.read(4)
        if len(probe) < 4:
            raise FileFormatError(
                f"Cannot open {filepath} as Fortran-unformatted TRF: too short"
            )
        endian = detect_endian(probe, source=f'read_oasp_trf:{filepath}')
        f.seek(0)
        try:
            fileid_raw = _read_fortran_record(f, raw=True, endian=endian)
        except IOError as e:
            raise FileFormatError(
                f"Cannot open {filepath} as Fortran-unformatted TRF: {e}"
            ) from e
        fileid = fileid_raw.decode('ascii', errors='ignore').strip()
        if 'PULSETRF' not in fileid:
            raise FileFormatError(
                f"Expected 'PULSETRF' in first record, got {fileid!r}"
            )

        # prognm record consumed but not used
        _read_fortran_record(f, raw=True, endian=endian)
        (nout,) = _read_fortran_record(f, 'i', endian=endian)
        iparm_raw = _read_fortran_record(f, raw=True, endian=endian)
        iparm = list(struct.unpack(endian + f'{len(iparm_raw) // 4}i',
                                   iparm_raw))[:nout]

        title = _decode_fortran_title(
            _read_fortran_record(f, raw=True, endian=endian))
        # signn record consumed but not used
        _read_fortran_record(f, raw=True, endian=endian)

        (freqs,) = _read_fortran_record(f, 'f', endian=endian)
        (sd,) = _read_fortran_record(f, 'f', endian=endian)

        rd, rdlow, ir = _read_fortran_record(f, 'ffi', endian=endian)
        nrd = max(1, abs(ir))
        if ir < 0:
            # oasiun23.f:870-877 writes the explicit RDC list only for IR<0.
            # No OASES program reaches that branch — INREC negates IR in
            # COMMON /VARS1/ before TRFHEAD runs (oaseun31.f:1185) — but the
            # record must still be consumed to keep the stream aligned.
            _read_fortran_record(f, f'{nrd}f', endian=endian)
        if receiver_depths.size != nrd:
            raise FileFormatError(
                f"{filepath}: header declares {nrd} receiver depths "
                f"(RD={rd:g} m, RDLOW={rdlow:g} m) but {receiver_depths.size} "
                f"were supplied; the file does not belong to this receiver "
                f"array.",
                remediation=f"Pass the depths this file was written on — "
                            f"OASES lays them out uniformly, so "
                            f"receiver_depths=np.linspace({rd:g}, {rdlow:g}, "
                            f"{nrd}) reproduces them.",
            )

        r0, rspace, nplots = _read_fortran_record(f, 'ffi', endian=endian)
        cur = f.tell()
        f.seek(0, 2)
        _file_size = f.tell()
        f.seek(cur)
        _bound_counts(filepath, _file_size, 16, nplots=nplots)
        # r0/rspace are on disk in km (OASES convention); convert to metres at
        # the reader boundary so the Field carries ranges in metres — matching
        # read_oast_tl and the package-wide ranges-in-metres invariant.
        ranges = km_to_m(r0 + np.arange(nplots) * rspace)

        nx, lx, mx, dt = _read_fortran_record(f, 'iiif', endian=endian)
        # icdr record consumed but not used
        _read_fortran_record(f, 'i', endian=endian)
        (omegim,) = _read_fortran_record(f, 'f', endian=endian)

        (msuft,) = _read_fortran_record(f, 'i', endian=endian)
        (isrow,) = _read_fortran_record(f, 'i', endian=endian)
        # inttyp record consumed but not used
        _read_fortran_record(f, 'i', endian=endian)
        for _ in range(2):
            _read_fortran_record(f, 'i', endian=endian)
        for _ in range(5):
            _read_fortran_record(f, 'f', endian=endian)

        # --- Data records ---
        nf = max(1, mx - lx + 1)
        # OASES bin indices are 1-based: OASP sets DLFREQ = 1/(DT*NX) and
        # LX = FR1/DLFREQ + 1 (unoasp22.f:237-241; the pulse post-processor
        # repeats it at oasiun22.f:1256-1261), so bin k carries frequency
        # (k-1)*DLFREQ. Using k/(dt*nx) puts the whole axis one bin
        # (= 1/(dt*nx) Hz) too high.
        freq_array = np.array(
            [((k - 1) / (dt * nx)) for k in range(lx, mx + 1)],
            dtype=np.float64,
        )

        # Detect the data-record precision from the first record's length
        # marker: 2*nout float32 for the default COMPLEX*8 payload, 2*nout
        # float64 when the writer's CFFX was compiled with a promoted default
        # real. (Option '8' alone does not promote it — CFFX is declared plain
        # COMPLEX at oasiun23.f:18 and the '8' branch at oasiun23.f:309-312
        # writes the same kind; what '8' changes is the file's name, handled by
        # the caller.) Anything else is a wrong-endianness or truncated file.
        pos = f.tell()
        marker = f.read(4)
        data_bytes = 2 * nout * 4
        data_ftype = 'f4'
        out_dtype = np.complex64
        if len(marker) == 4:
            (rec_bytes,) = struct.unpack(endian + 'i', marker)
            if rec_bytes == 2 * nout * 8:
                data_bytes = 2 * nout * 8      # double-precision COMPLEX*16
                data_ftype = 'f8'
                out_dtype = np.complex128
            elif rec_bytes != 2 * nout * 4:
                raise FileFormatError(
                    f"OASP .trf data record is {rec_bytes} bytes; expected "
                    f"{2 * nout * 4} (COMPLEX*8) or {2 * nout * 8} "
                    f"(COMPLEX*16) for nout={nout} (wrong endianness or "
                    f"corrupt file)."
                )
        f.seek(pos)

        # uacpy reads the axisymmetric, single-output-parameter OASP case.
        # The writer nests DO IS=1,ISROW / DO M=1,MSUFT / DO JRH / DO JRV with
        # NOUT components per record (oasiun23.f:305-311); for isrow>1, msuft>1
        # or nout>1 the (nf, nplots, nrd)/first-component layout below would
        # silently collapse slabs/parameters, so reject those rather than
        # return wrong data.
        if isrow != 1 or msuft != 1:
            raise FileFormatError(
                f"OASP .trf has isrow={isrow}, msuft={msuft}: multi-source-row "
                f"or azimuthal (3D bearing) transfer functions are not "
                f"supported — uacpy reads the axisymmetric single-slab case "
                f"only."
            )
        if nout != 1:
            raise FileFormatError(
                f"OASP .trf carries nout={nout} output parameters; uacpy reads "
                f"a single component (normal stress / pressure). Use a "
                f"single-output OASP option string."
            )

        # Bound the (nf, nplots, nrd) allocation against the file size before
        # sizing it off these header-derived counts. Each data record holds
        # 2*nout floats plus two 4-byte Fortran markers, so ≥ 16 bytes.
        cur = f.tell()
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(cur)
        _bound_counts(filepath, file_size, 16, nf=nf, nplots=nplots, nrd=nrd)

        # The nout == 1 pin above fixes every data record at the same
        # ``[marker][re im][marker]`` frame, and CFFX writes them back to back
        # over (JRH, JRV) inside the frequency loop (oasiun23.f:305-311), so
        # the whole block reads in one strided pass rather than three
        # ``f.read``/``struct.unpack`` pairs per record. The .rpo replicas
        # at :932 can compare their marker against the constant 8 because
        # PUTREP writes COMPLEX*8 only; a .trf is COMPLEX*8 or COMPLEX*16, so
        # the marker checked here is the payload width detected above and
        # hardcoding 8 would reject every double-precision file.
        n_total = nf * nplots * nrd
        rec_dt = np.dtype([
            ('m1', endian + 'i4'),
            ('re', endian + data_ftype),
            ('im', endian + data_ftype),
            ('m2', endian + 'i4'),
        ])
        flat = np.fromfile(f, dtype=rec_dt, count=n_total)
        if flat.size < n_total:
            raise FileFormatError(
                f"{filepath}: truncated transfer function — expected "
                f"{n_total} data records of {data_bytes + 8} bytes, got "
                f"{flat.size}"
            )
        if np.any(flat['m1'] != data_bytes) or np.any(flat['m2'] != data_bytes):
            raise FileFormatError(
                f"Fortran record marker mismatch: .trf data records are not "
                f"all framed by the expected {data_bytes}-byte payload length "
                f"(wrong endianness or truncated file)"
            )
        transfer_function = np.empty(n_total, dtype=out_dtype)
        transfer_function.real = flat['re']
        transfer_function.imag = flat['im']
        transfer_function = transfer_function.reshape(nf, nplots, nrd)

    # IPARM holds slot numbers, so compare against the table's keys.
    if omegim != 0.0:
        warnings.warn(
            f"{filepath}: the .trf was integrated on a complex frequency "
            f"contour (OMEGIM = {float(omegim):.6g} rad/s, unoasp22.f:"
            f"372-376); a time series synthesized from this spectrum decays "
            f"by exp(OMEGIM*t) — 50x (34 dB) small at the end of the window "
            f"— because the synthesis does not undo the contour. Re-run "
            f"with option 'J' or pinned nw_samples >= 1 for a real "
            f"frequency axis.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )
    unnamed = sorted(set(iparm) - set(_OASP_OUTPUT_PARAM_LETTERS.keys()))
    if unnamed:
        raise FileFormatError(
            f"{filepath}: IPARM names output slot(s) {unnamed}, which no OASP "
            f"option letter selects (unoasp22.f:884-919)."
        )
    return {
        'title': title,
        'option': ''.join(_OASP_OUTPUT_PARAM_LETTERS[p] for p in iparm),
        'freq': freq_array,
        'ranges': ranges,
        'depths': receiver_depths,
        'transfer_function': transfer_function,
        'source_depth': float(sd),
        'center_frequency': float(freqs),
        'omegim': float(omegim),
    }


def read_oasr_reflection_coefficients(
    filepath: Union[str, Path],
    format_type: str = 'auto'
) -> Dict:
    """
    Read OASR reflection coefficient file (.rco or .trc format)

    OASR outputs reflection/transmission coefficients as a function of
    frequency and angle/slowness.

    Parameters
    ----------
    filepath : str or Path
        Path to .rco (slowness) or .trc (angle) file
    format_type : str, optional
        'slowness' or 'angle' to declare the abscissa, overriding whatever the
        file says; 'auto' (default) reads the sampling-type code OASR stamps
        into the header and warns if the file's extension disagrees with it.

    Returns
    -------
    data : dict
        Dictionary containing:
        - 'freq_min': float, minimum frequency (Hz)
        - 'freq_max': float, maximum frequency (Hz)
        - 'n_frequencies': int, number of frequencies
        - 'sampling_type': str, 'slowness' or 'angle'
        - 'frequencies': list of float, the frequency of each block (Hz)
        - 'angles_or_slowness': list of ndarray, angle (deg) or slowness (s/km)
        - 'magnitude': list of ndarray, reflection coefficient magnitude
        - 'phase': list of ndarray, reflection coefficient phase (degrees)

    Notes
    -----
    File format::

        freq_min freq_max n_freq sampling_type    (2F12.3, 2I4)
        for each frequency:
            frequency n_samples  # Frequency, # of slownesses|angles
            n_samples x (angle_or_slowness  magnitude  phase)   (3F15.6)

    ``sampling_type`` is 1 for the slowness table and 2 for the angle table:
    OASR opens both at once and stamps the codes on them (unoasr21.f:204-205),
    with unit 22 going to ``.rco`` and unit 23 to ``.trc`` (``bin/oasr:8-9``).
    The per-frequency header carries a trailing ``# …`` annotation
    (oasjun21.f:27-29), so only its first two tokens are values. The abscissa
    is written as ``slw*1e3`` for the ``.rco`` — slowness in s/km — and as
    ``degang`` for the ``.trc``; the phase is scaled by ``omr`` to degrees in
    both (oasjun21.f:102-104).

    Examples
    --------
    >>> data = read_oasr_reflection_coefficients('test.trc')
    >>> mag = data['magnitude'][0]  # First frequency
    >>> angles = data['angles_or_slowness'][0]
    >>> print(f"Reflection coefficient at 45°: {mag[np.argmin(np.abs(angles-45))]}")
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileFormatError(
            f"OASR reflection coefficient file not found: {filepath}",
            remediation="Re-run OASR with 'T' in the option string: the "
                        "reflection coefficient table reaches the .rco/.trc "
                        "files only under that letter.",
        )

    # Auto-detect from the file's own content: the header's fourth field is
    # the sampling-type code OASR stamps as it writes (unoasr21.f:204-205),
    # while the extension is only what the file is called. The extension is
    # kept as a hint so a disagreement can be reported rather than followed —
    # a renamed table used to come back with its abscissa relabelled.
    extension_hint = None
    if format_type == 'auto':
        if filepath.suffix == '.rco':
            extension_hint = 'slowness'
        elif filepath.suffix == '.trc':
            extension_hint = 'angle'
        format_type = None

    try:
        with open(filepath, 'r') as f:
            # Read header line
            header_line = f.readline().strip()
            header_parts = header_line.split()

            # fortran_float, not float(): every value column is a Fortran
            # real write, which can spell an exponent as D+00 or drop the
            # letter entirely when it needs three digits.
            if len(header_parts) >= 4:
                freq_min = fortran_float(header_parts[0])
                freq_max = fortran_float(header_parts[1])
                n_freq = int(header_parts[2])
                sampling_type_code = int(header_parts[3])

                # Decode sampling type
                if format_type is None:
                    format_type = ('slowness' if sampling_type_code == 1
                                   else 'angle')
                    if extension_hint not in (None, format_type):
                        warnings.warn(
                            f"read_oasr_reflection_coefficients: "
                            f"{filepath.name} is named {filepath.suffix} "
                            f"(OASR writes {extension_hint} tables there), "
                            f"but its header code {sampling_type_code} says "
                            f"{format_type} — following the header, so the "
                            f"abscissa is read as {format_type}. Pass "
                            f"format_type='{extension_hint}' to override.",
                            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)

                sampling_type = format_type
            else:
                raise FileFormatError(f"Invalid header format: {header_line}")

            # Read data for each frequency
            frequencies = []
            angles_or_slowness_list = []
            magnitude_list = []
            phase_list = []

            for i_freq in range(n_freq):
                raw_header = f.readline()
                freq_header = raw_header.strip().split()
                if len(freq_header) < 2:
                    # Wide fields can concatenate; slice the per-frequency
                    # header's own (1h ,f12.3,i6) layout (oasjun21.f:27-30)
                    # before giving up.
                    try:
                        freq = fortran_float(raw_header[1:13])
                        n_samples = int(raw_header[13:19])
                    except (ValueError, IndexError):
                        raise FileFormatError(
                            f"{filepath}: frequency header {i_freq + 1} of "
                            f"{n_freq} is malformed or missing "
                            f"({raw_header.strip()!r}) — the file is "
                            f"truncated or not a valid OASR "
                            f"{filepath.suffix or '.rco/.trc'} table.")
                else:
                    freq = fortran_float(freq_header[0])
                    n_samples = int(freq_header[1])

                # Read samples
                angles_or_slowness = []
                magnitude = []
                phase = []

                for i_row in range(n_samples):
                    parts = f.readline().split()
                    if len(parts) < 3:
                        raise FileFormatError(
                            f"{filepath}: reflection row {i_row + 1} of "
                            f"{n_samples} at {freq:g} Hz is short or missing "
                            f"— oasjun21.f:103-104 writes exactly NWVNO "
                            f"(3f15.6) rows, so the run died mid-write.")
                    angles_or_slowness.append(fortran_float(parts[0]))
                    magnitude.append(fortran_float(parts[1]))
                    phase.append(fortran_float(parts[2]))

                frequencies.append(freq)
                angles_or_slowness_list.append(np.array(angles_or_slowness))
                magnitude_list.append(np.array(magnitude))
                phase_list.append(np.array(phase))

        return {
            'freq_min': freq_min,
            'freq_max': freq_max,
            'n_frequencies': n_freq,
            'sampling_type': sampling_type,
            'frequencies': frequencies,
            'angles_or_slowness': angles_or_slowness_list,
            'magnitude': magnitude_list,
            'phase': phase_list,
        }

    except (FileFormatError, UnsupportedFeatureError):
        raise
    except PARSE_ERRORS as e:
        raise FileFormatError(f"Failed to read OASR reflection coefficient file {filepath}: {e}") from e


def read_oases_rhs_header(filepath: Union[str, Path]) -> Dict:
    """Read the header of a mean-field ``.rhs`` (OASES unit 45).

    A producer run with option ``'s'`` writes three kinds of record
    (``unoasp22.f:254``, ``oaseun31.f:1899``, ``:2395``)::

        WRITE(45) NX, FR1, FR2, DT                      ! once
        WRITE(45) FREQ, LAYS(1), NWVNO, NFLAG, FNI5     ! per frequency
        WRITE(45) WVNO, IN1, DBDZ(1..4), BLC(1..4)      ! per rough interface
                                                        ! and wavenumber

    **The first record's four slots mean different things per producer**, and
    the key names below are OASP's because that is the broadband path. OASP
    writes a genuine FFT grid, ``nx, fr1, fr2, dt`` (``unoasp22.f:254``); a
    single-frequency OAST or OASR writes ``nfreq, freq1, freq2,
    1/(nfreq*dlfreq)`` into the same four slots (``unoast31.f:164-165``,
    ``unoasr21.f:222-223``). So on the OAST/OASR path ``n_time_samples`` is the
    frequency-block count — which OASS requires to be 1, since it pins
    ``nfreq=1`` and reads exactly one per-frequency header
    (``unoass21.f:123``, ``:127``) — and ``freq_min`` is the single frequency
    the mean field ran at. The keys are not renamed per producer: the file
    layout is one layout, and a caller that knows which producer it drove knows
    which reading applies.

    OASSP reads exactly the same three records — the first at
    ``unoassp30.f:181``, the other two at ``:546-547`` — and then **replaces**
    its own deck's ``NT/FR1/FR2/DT`` with the first record's values, warning
    only when ``NX`` or ``DT`` differ (``:182-188``). A caller that writes the
    deck from these values instead of from a second guess makes that
    substitution a no-op, which is the only way to be sure the band that ran is
    the band that was asked for.

    ``interface`` is the third record's ``IN1``: ``SCTRHS`` loops over
    interfaces innermost (``oaseun31.f:2306-2308``), so the first such record
    carries the shallowest rough interface, and that is the single interface
    OASSP scatters from.

    Returns
    -------
    dict
        ``n_time_samples`` (int, ``NX`` — ``nfreq`` from OAST/OASR),
        ``freq_min`` / ``freq_max`` (Hz; ``freq_min`` is *the* frequency from
        OAST/OASR), ``time_step`` (s, ``DT`` — ``1/(nfreq*dlfreq)`` from
        OAST/OASR), ``frequency`` (Hz, the per-frequency header's own
        ``FREQ``), ``source_layer`` (``LAYS(1)``), ``n_wavenumbers``
        (``NWVNO``), ``interface`` (``IN1``, 1-based OASES layer index).
    """
    filepath = Path(filepath)

    if not filepath.is_file():
        raise FileFormatError(
            f"OASES .rhs mean-field file not found: {filepath}",
            remediation="Re-run the producer with 's' in the option string, "
                        "over an environment carrying a rough interface: "
                        "SCTRHS skips every interface with ROUGH2 < 1e-10 "
                        "(oaseun31.f:2310), so a smooth deck writes no .rhs "
                        "even under 's'.",
        )

    try:
        with open(filepath, 'rb') as f:
            probe = f.read(4)
            if len(probe) < 4:
                raise FileFormatError(
                    f"OASES .rhs {filepath} is empty or truncated. OPFILB "
                    f"opens unit 45 with STATUS='UNKNOWN' (oashun21.f:634), so "
                    f"a producer run that wrote no scattering right-hand sides "
                    f"still leaves a zero-length file behind."
                )
            endian = detect_endian(
                probe, source=f'read_oases_rhs_header:{filepath.name}')
            f.seek(0)
            nx, fr1, fr2, dt = _read_fortran_record(f, 'ifff', endian=endian)
            freq, lays1, nwvno, _nflag, _fni5 = _read_fortran_record(
                f, 'fiiif', endian=endian)
            # The SCTRHS record opens with a COMPLEX wavenumber (two reals)
            # followed by the interface index; the eight complex words after
            # it are the boundary operators DBDZ and BLC, which nothing here
            # needs. 19 four-byte words in all — checked, because a short
            # record here means the file is a per-frequency header rather
            # than a scattering record, i.e. the producer wrote none.
            try:
                payload = _read_fortran_record(f, raw=True, endian=endian)
            except FileFormatError as e:
                raise FileFormatError(
                    f"OASES .rhs {filepath} ends after its frequency header, "
                    f"so it holds no 76-byte SCTRHS record (WVNO, IN1, "
                    f"DBDZ(4), BLC(4) — oaseun31.f:2395) and names no "
                    f"scattering interface. SCTRHS skips every interface with "
                    f"ROUGH2 < 1e-10 (oaseun31.f:2310), so the producer run's "
                    f"environment was smooth. Underlying error: {e}"
                ) from e
            if len(payload) != 4 * (2 + 1 + 8 + 8):
                raise FileFormatError(
                    f"OASES .rhs {filepath}: expected a 76-byte SCTRHS record "
                    f"(WVNO, IN1, DBDZ(4), BLC(4) — oaseun31.f:2395), got "
                    f"{len(payload)} bytes."
                )
            interface = struct.unpack(endian + 'ffi', payload[:12])[2]
    except FileFormatError:
        raise
    except PARSE_ERRORS as e:
        raise FileFormatError(
            f"Failed to read OASES .rhs header {filepath}: {e}") from e

    if int(nx) < 1 or float(dt) <= 0.0:
        raise FileFormatError(
            f"OASES .rhs header {filepath} carries NX={nx}, DT={dt}, which "
            f"cannot describe an FFT grid; the producer run probably wrote no "
            f"scattering right-hand sides (option 's' missing, or every "
            f"interface smooth — SCTRHS skips ROUGH2 < 1e-10, "
            f"oaseun31.f:2310)."
        )
    return {
        'n_time_samples': int(nx),
        'freq_min': float(fr1),
        'freq_max': float(fr2),
        'time_step': float(dt),
        'frequency': float(freq),
        'source_layer': int(lays1),
        'n_wavenumbers': int(nwvno),
        'interface': int(interface),
    }
