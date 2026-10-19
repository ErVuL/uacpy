"""The raw output-file parsers the engines read.

An engine reads its output file as the file holds it, through a parser here;
a user reads the record the public reader in :mod:`uacpy.io` builds from the
same file (:func:`~uacpy.io.read_oast_tl`, :func:`~uacpy.io.read_oasp_trf`,
:func:`~uacpy.io.read_tl_grid`, :func:`~uacpy.io.read_pcomplex_grid`). A
package-internal module (``tests/_internal_modules.py``): its names are
public across uacpy's packages and not part of the user API.
"""

import struct
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TypedDict, Union

import numpy as np

from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.exceptions import (
    ConfigurationError, FileFormatError, IOWarning, MissingFileError,
    UnsupportedFeatureError,
)
from uacpy.core.results import Field
from uacpy.core.units import km_to_m
from uacpy.io._fortran_helpers import (
    _bound_counts, detect_endian, read_fortran_record as _read_fortran_record,
    require_model_output, typed_format_error,
)

__all__ = ['OastTlFields', 'parse_oast_tl', 'oast_depth_average_field',
           'parse_oasp_trf', 'oasp_trf_pressure', 'parse_tl_grid',
           'parse_pcomplex_grid']


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

    if receiver_depths is None:
        raise FileFormatError(
            f"{plp_file}: {reason}, so the file alone cannot say which depth "
            f"each curve belongs to.",
            remediation="Pass receiver_depths= (the deck's receiver axis) "
                        "to attribute the curves by position.")
    n_depths = len(receiver_depths)
    if n_depths == 0:
        raise ConfigurationError(
            f"receiver_depths is empty, and {plp_file}: {reason}, so the "
            f"curves can only be attributed to depths by position on the "
            f"deck's receiver axis.",
            remediation="Pass receiver_depths= (the deck's receiver depths, "
                        "m), or omit it for a .plp that carries the 'RD:' "
                        "labels.")
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
        IOWarning,
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

#: ``.plp`` curve tag PLDAV writes for the depth-averaged TL option ``'A'``
#: computes: ``optpar(INR)//'TLDAV'`` (oasfun22.f:403).
_OAST_DEPTH_AVERAGE_TAG = 'TLDAV'

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

    PLPWRI writes each label as ``FORMAT(1H ,A16)`` (oasgun21.f:615, FORMAT 779), and
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
        # NLAB and NC are the file's own record counts (oasgun21.f:614,
        # :631). A negative NLAB walks the cursor backwards over records it
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
                f"Cannot open {filepath} as Fortran-unformatted TRF: too short."
            )
        endian = detect_endian(probe, source=f'read_oasp_trf:{filepath}')
        f.seek(0)
        try:
            fileid_raw = _read_fortran_record(f, raw=True, endian=endian)
        except IOError as e:
            raise FileFormatError(
                f"Cannot open {filepath} as Fortran-unformatted TRF: {e}."
            ) from e
        fileid = fileid_raw.decode('ascii', errors='ignore').strip()
        if 'PULSETRF' not in fileid:
            raise FileFormatError(
                f"Expected 'PULSETRF' in first record, got {fileid!r}."
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
        # NOUT components per record (INTGR3, oasiun23.f:757-773); for isrow>1, msuft>1
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
        # over (JRH, JRV) inside the frequency loop (INTGR3, oasiun23.f:757-773), so
        # the whole block reads in one strided pass rather than three
        # ``f.read``/``struct.unpack`` pairs per record. The .rpo replicas
        # (``_read_oasn_replicas_payload``) can compare their marker against the constant 8 because
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
                f"{flat.size}."
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
            f"with option 'J' or pinned n_wavenumbers >= 1 for a real "
            f"frequency axis.",
            IOWarning, skip_file_prefixes=USER_FRAME_SKIP,
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
        'fc': float(freqs),
        'omegim': float(omegim),
    }



class OastTlFields(TypedDict):
    """The named fields :func:`parse_oast_tl` hands back; the function's
    own Returns section carries the shapes, which depend on ``NFREQ``.

    A ``Tuple[...]`` return type here is read by the caller in
    ``uacpy/models/oases/oast.py`` as ``oast_out['depths']`` — a string
    subscript of something declared a tuple."""

    tl: np.ndarray
    depths: np.ndarray
    ranges: np.ndarray
    depth_average: Optional[np.ndarray]
    metadata: Dict[str, Any]


def _oast_depth_average(plp_file: Path, plt_file: Path, curves: list,
                        blocks: list, param: str, freq_axis,
                        grids: Dict[int, Tuple[float, float]],
                        n_ranges: int) -> Optional[np.ndarray]:
    """The depth-averaged TL curves option ``'A'`` adds, one per frequency
    as ``(n_freq, n_ranges)``, or ``None`` for a ``.plp`` that carries none.

    GETTLAV (oasfun22.f:209-242) averages ``|p|²`` over all the deck's
    receivers with weight ``1/IR`` and writes ``-10·log10`` of the mean;
    PLDAV writes it once per frequency, after the receiver loop
    (unoast31.f:681-688), on the range grid of that frequency's TL curves
    (``PLTWRI(LF,RMIN,RSTEP,…)``, oasfun22.f:433). Its labels are ``Freq:``
    and ``SD:`` only (``NLAB=2``, oasfun22.f:406-408), so the curves are
    attributed to frequencies in file order, and a ``Freq:`` label that
    disagrees with the TL curves' is refused.
    """
    average_curves = [c for c in curves
                      if c['tag'].endswith(_OAST_DEPTH_AVERAGE_TAG)
                      and c['tag'][0] == param]
    if not average_curves:
        return None
    n_freq = len(grids)
    if len(average_curves) != n_freq:
        raise FileFormatError(
            f"{plp_file} carries {len(average_curves)} depth-averaged TL "
            f"curve(s) ('*{_OAST_DEPTH_AVERAGE_TAG}') for {n_freq} "
            f"frequenc{'y' if n_freq == 1 else 'ies'}; OAST writes one per "
            f"frequency (unoast31.f:681-688).")
    average = np.empty((n_freq, n_ranges), dtype=float)
    for j, curve in enumerate(average_curves):
        label = curve['labels'].get('Freq')
        if (freq_axis is not None and label is not None
                and label != freq_axis[j]):
            raise FileFormatError(
                f"{plp_file}: depth-averaged TL curve {j} is labelled "
                f"{label} Hz where the TL curves' frequency {j} is "
                f"{freq_axis[j]} Hz.")
        if (curve['n'], (curve['xoff'], curve['dx'])) != (n_ranges, grids[j]):
            raise FileFormatError(
                f"{plp_file}: depth-averaged TL curve {j} has N={curve['n']} "
                f"samples from XOFF={curve['xoff']} km in steps of "
                f"DX={curve['dx']} km, not the TL curves' range grid of its "
                f"frequency ({n_ranges} samples, {grids[j]}).")
        if curve['index'] is None:
            raise FileFormatError(
                f"{plp_file}: depth-averaged TL curve {j} parameterises both "
                f"axes, so PLTWRI wrote no {plt_file.suffix} block for it "
                f"(oasgun21.f:658-660).")
        average[j] = _curve_values(blocks[curve['index']], curve, plt_file)
    return average


@typed_format_error
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
        - ``'depth_average'`` : ndarray or None — the depth-averaged TL
          (dB) option ``'A'`` makes OAST write, ``-10·log10`` of ``|p|²``
          averaged over every receiver of the deck (oasfun22.f:209-242;
          "the averaging is performed over the specified number of
          receivers", oast.tex:144-146), on the TL curves' range grid:
          shape ``(n_ranges_native,)`` for a single frequency,
          ``(n_freq, n_ranges_native)`` for a sweep. ``None`` when the file
          carries no such curve.
        - ``'metadata'`` : dict — ``{'oast_grid_shape': …,
          'n_frequencies': n_f, 'plotted_frequencies': ndarray}``.
          ``'plotted_frequencies'`` comes from the ``Freq:`` labels (``F7.1``, oasfun22.f:368) and is
          absent from a ``.plp`` that carries no labels.

    Raises
    ------
    MissingFileError
        If the ``.plp`` file is missing (OAST chooses its own range grid via
        FFT-based sampling, so the native grid cannot be reconstructed
        without it) or the TL data file is.
    FileFormatError
        If the ``.plp`` file cannot be parsed, if it carries no TL-vs-range
        curve, if its ``Freq:``/``RD:`` labels do not form a full grid, if
        the receivers of one frequency disagree about the range axis, or if
        its depth-averaged TL curves do not match the TL curves'
        frequencies and range grids.
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
        raise MissingFileError(
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
        raise MissingFileError(
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

    depth_average = _oast_depth_average(
        plp_file, tl_data_file, curves, blocks, params[0], freq_axis, grids,
        n_ranges_oast)
    range_rows = np.array([
        km_to_m(grids[j][0] + np.arange(n_ranges_oast) * grids[j][1])
        for j in range(n_freq)
    ])
    if n_freq == 1:
        tl_oast = tl_oast[0]
        ranges_oast = range_rows[0]
        if depth_average is not None:
            depth_average = depth_average[0]
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
        'depth_average': depth_average,
        'metadata': metadata,
    }


@typed_format_error
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
        - 'fc': float, center frequency (Hz)

    Notes
    -----
    Format follows the OASES PULSETRF binary specification from
    trford.f/oasiun23.f.

    Examples
    --------
    The files are a run's own output, so these lines show the calls and
    do not run here::

        data = parse_oasp_trf('pulse.trf', receiver_depths=[20., 50., 80.])
        trf = data['transfer_function']  # shape: (n_freq, n_range, n_depth)
    """
    if not Path(filepath).exists():
        raise MissingFileError(
            f"OASP transfer function file not found: {filepath}.",
            remediation="Check the OASP run completed: the .trf is its "
                        "primary output, so a missing one means the binary "
                        "stopped before writing it and its stdout carries "
                        "the reason.",
        )
    return _read_oasp_trf_binary(
        Path(filepath), np.atleast_1d(np.asarray(receiver_depths, dtype=float)))


def oast_depth_average_field(depth_average: np.ndarray,
                             ranges: np.ndarray, **identity) -> Field:
    """The depth-averaged TL curve of an OAST run (option ``'A'``) as a
    real dB :class:`~uacpy.core.results.Field` on ``{'range'}``.

    The values are OAST's own: ``-10·log10`` of ``|p|²`` averaged with
    equal weights over every receiver depth of the deck
    (oasfun22.f:209-242, oast.tex:144-146), on the TL curves' native range
    grid ``ranges`` (m). The sum is over intensities, so the Field is
    ``coherent=False``, as a band average is. ``identity`` carries the
    producer's result keywords (``model``, ``backend``, ``frequencies`` …).
    """
    return Field(data=np.asarray(depth_average, dtype=float),
                 coords={'range': np.asarray(ranges, dtype=float)},
                 kind='pressure', unit='dB', coherent=False, **identity)


def oasp_trf_pressure(raw: Dict, who: str, filepath) -> np.ndarray:
    """The pressure a ``.trf`` holds, from :func:`parse_oasp_trf`'s
    ``raw``: complex128 on ``(depth, range, frequency)``.

    OASP's and OASSP's output parameter ``'K'`` (their default) is the bulk
    pressure, the acoustic pressure in a fluid (oasaun31.f:1448-1449,
    :1471-1472; oasp.tex:178-183), read as written. ``'N'`` is the normal
    stress, "N  Normal stress sigma_zz (= -p in fluids)" (oasp.tex:185), so
    the pressure is its negation. Both are on the ``e^{+i omega t}``
    convention the other engines share (a source delay is
    ``exp(-ai*dsq*sdelay)``, ``dsq = 2*pi*freq``, oaseun31.f:1681); the
    file's COMPLEX*8 is upcast so every engine returns one dtype. Any other
    output parameter is not a pressure and raises
    :class:`~uacpy.core.exceptions.UnsupportedFeatureError` naming ``who``
    and ``filepath``.
    """
    sign = {'K': 1.0, 'N': -1.0}.get(raw['option'])
    if sign is None:
        raise UnsupportedFeatureError(
            who,
            f"{filepath} holds OASP output parameter {raw['option']!r}; "
            f"uacpy reads the bulk pressure 'K' (oasp.tex:178-183) or the "
            f"normal stress 'N', whose negation is the pressure in a fluid "
            f"(oasp.tex:185), into a pressure Field")
    return sign * np.transpose(raw['transfer_function'], (2, 1, 0)).astype(
        np.complex128)


#: Double-precision kind → the single-precision kind a stock build writes.
_SINGLE_PRECISION = {'f8': 'f4', 'c16': 'c8'}


def _read_lz_records(
    filepath: Union[str, Path], *, dtype: str
) -> Tuple[int, np.ndarray]:
    """
    Read a Fortran-unformatted file whose record 1 is ``int32 lz`` and
    records 2..N each hold ``lz`` samples of ``dtype``. Returns ``(lz,
    matrix[lz, n_records])``.

    ``dtype`` is an endian-agnostic kind string (``'f8'``, ``'c16'`` — the
    Collins binaries are built with ``-fdefault-real-8``); this
    helper owns byte order entirely. Byte order is auto-detected from the
    first record marker and applied to ``dtype`` here, so callers must not
    pass a ``<``/``>`` prefix (any prefix is stripped defensively). A
    one-shot warning fires the first time a big-endian file is decoded.

    ``dtype`` names the double-precision kind; a first data record of half
    that length (``lz`` single-precision samples, a stock build) switches
    the file to ``'f4'``/``'c8'``, returned upcast to ``dtype``. Bytes left
    after the last whole record (a run cut short mid-write) warn with the
    count of records kept.
    """
    path = Path(filepath)
    with path.open('rb') as f:
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)
        # 12 bytes is the smallest possible file: the header record is a
        # 4-byte marker, one int32 ``lz``, and a 4-byte trailing marker.
        if file_size < 12:
            raise FileFormatError(f"{path}: too short to contain the header record.")

        probe = f.read(4)
        f.seek(0)
        endian = detect_endian(probe, source=f'Collins PE grid:{path.name}')
        # Re-anchor the caller-supplied dtype on the detected byte order so
        # the file decodes correctly regardless of host endianness.
        base_dtype = dtype.lstrip('<>=|')
        item_dtype = np.dtype(endian + base_dtype)

        (lz,) = _read_fortran_record(f, 'i', endian=endian)
        single = _SINGLE_PRECISION.get(base_dtype)
        if single is not None and lz > 0 and file_size - f.tell() >= 4:
            (first_len,) = np.frombuffer(f.read(4), dtype=endian + 'i4')
            f.seek(-4, 1)
            if first_len == lz * np.dtype(single).itemsize:
                item_dtype = np.dtype(endian + single)
        expected = lz * item_dtype.itemsize

        columns = []
        # Stop before a partial trailing record: each data record is its
        # ``lz``-sample payload framed by a leading and trailing 4-byte marker.
        while file_size - f.tell() >= 8 + expected:
            payload = _read_fortran_record(f, raw=True, endian=endian)
            if len(payload) != expected:
                raise FileFormatError(
                    f"{path}: expected {expected}-byte data record, "
                    f"got {len(payload)}."
                )
            col = np.frombuffer(payload, dtype=item_dtype).astype(
                base_dtype, copy=True
            )
            columns.append(col)
        leftover = file_size - f.tell()

    if not columns:
        raise FileFormatError(f"{path}: no data records found.")
    if leftover:
        warnings.warn(
            f"{path.name}: {leftover} byte(s) after the last whole "
            f"{expected}-byte record are not a complete range step and were "
            f"dropped; {len(columns)} range step(s) read. The run was cut "
            f"short while writing.",
            IOWarning, skip_file_prefixes=USER_FRAME_SKIP)

    return lz, np.stack(columns, axis=1)


def _grid_axes(lz, n_ranges, dr, ndr, dz, ndz, depth_index_offset):
    """Range and depth axes (m) shared by ``tl.grid`` and ``pcomplex.bin``,
    which are written on the same (z, r) grid."""
    ranges = np.arange(1, n_ranges + 1, dtype=float) * dr * ndr
    depths = (depth_index_offset
              + np.arange(1, lz + 1, dtype=float) * ndz - 1) * dz
    return ranges, depths


def _read_grid(filepath, reader, dtype, dr, ndr, dz, ndz, depth_index_offset):
    """``(ranges, depths, field)`` of a ``tl.grid``-shaped file; the field is
    native-endian ``dtype`` straight from :func:`_read_lz_records`."""
    require_model_output(filepath, reader)
    lz, field = _read_lz_records(filepath, dtype=dtype)
    ranges, depths = _grid_axes(lz, field.shape[1], dr, ndr, dz, ndz,
                                depth_index_offset)
    return ranges, depths, field


@typed_format_error
def parse_tl_grid(
    filepath: Union[str, Path],
    *,
    dr: float,
    range_decimation: int,
    dz: float,
    depth_decimation: int,
    depth_index_offset: int = 0,
):
    """A Collins ``tl.grid`` as ``(ranges, depths, tl)``: the range and
    depth axes (m) and the ``(n_depths, n_ranges)`` TL (dB); see
    :func:`~uacpy.io.read_tl_grid` for the grid parameters."""
    return _read_grid(filepath, 'read_tl_grid', 'f8', dr, range_decimation, dz, depth_decimation,
                      depth_index_offset)


@typed_format_error
def parse_pcomplex_grid(
    filepath: Union[str, Path],
    *,
    dr: float,
    range_decimation: int,
    dz: float,
    depth_decimation: int,
    depth_index_offset: int = 0,
):
    """A uacpy-patched ``pcomplex.bin`` as ``(ranges, depths, p)``: the
    axes (m) and the complex PE envelope, ``(n_depths, n_ranges)``, before
    the carrier and the Hankel phase; see
    :func:`~uacpy.io.read_pcomplex_grid`."""
    return _read_grid(filepath, 'read_pcomplex_grid', 'c16', dr, range_decimation, dz,
                      depth_decimation, depth_index_offset)
