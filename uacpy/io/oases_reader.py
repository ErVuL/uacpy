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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from uacpy.core._export import ExportRecord
from uacpy.core.exceptions import (ConfigurationError, FileFormatError,
                                   IOWarning, MissingFileError,
                                   UnsupportedFeatureError)
from uacpy.core.results import Field
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.io._fortran_helpers import (
    _bound_counts,
    fortran_float,
    read_fortran_record as _read_fortran_record,
    detect_endian,
    typed_format_error,
)
from uacpy.core.units import km_to_m
from uacpy.io._parsers import (
    _decode_fortran_title, oasp_trf_pressure, parse_oasp_trf, parse_oast_tl,
)


def read_oast_tl(
    filepath: Union[str, Path],
    receiver_depths: Optional[np.ndarray] = None,
    *,
    frequencies: Optional[np.ndarray] = None,
) -> List[Field]:
    """
    Read OAST transmission loss output on its native grids.

    OAST writes TL (dB) to a ``.plt`` and the plot description — the only
    record of its range sampling — to the ``.plp`` beside it. This reader
    returns the TL on OAST's own grids; resampling onto a receiver grid is
    the caller's :meth:`Field.resample_to`.

    Parameters
    ----------
    filepath : str or Path
        The ``.plt`` or ``.plp`` file, or their base name.
    receiver_depths : ndarray, optional
        The receiver depth axis (m) the deck asked for. The ``RD:`` labels
        of the ``.plp`` (printed to 0.1 m) are matched against it, so the
        depths carry the caller's precision; omitted, the depths are the
        labels as printed, and a ``.plp`` without labels raises.
    frequencies : array_like, optional
        The frequencies (Hz) of the plotted blocks, one each, in the order
        the run wrote them (the OAST deck's frequencies). Required for a
        ``.plp`` without ``Freq:`` labels. On a labelled ``.plp`` it is
        checked against the labels: within 0.05 Hz (half the ``F7.1``
        label step) it is kept at its own precision, as receiver depths are
        matched back to ``receiver_depths``; further away it is refused.
        Omitted, the labels are the frequencies.

    Returns
    -------
    tl : list of Field
        **Always** a list — of length 1 for a single-frequency run — of real
        TL :class:`~uacpy.core.results.Field` ``(n_depths, n_ranges)``, one
        per plotted frequency in the run's order, so a caller never
        branches on the type. Each carries its **own** native range axis:
        OAST rebuilds its range grid inside the frequency loop (``DX``
        scales as ``1/f``, ``unoast31.f:481``), so the frequencies share no
        axis and are not one Field. Each Field's ``frequencies`` is its
        frequency: ``frequencies`` when given (checked against the ``Freq:``
        labels to their 0.05 Hz print quantum and kept at its own precision,
        as the receiver depths are), else the label (``F7.1``,
        ``oasfun22.f:368``).

    Raises
    ------
    MissingFileError
        If the ``.plp`` or the TL data file is missing.
    FileFormatError
        If the ``.plp`` file cannot be parsed, if it carries
        no TL-vs-range curve, if its ``Freq:``/``RD:`` labels do not form a
        full grid, or if the receivers of one frequency disagree about the
        range axis, or if it carries no ``Freq:`` labels and no
        ``frequencies`` is given.
    ConfigurationError
        If ``frequencies`` does not hold one value per plotted block, or
        disagrees with the ``Freq:`` labels.
    """
    parsed = parse_oast_tl(filepath, receiver_depths)
    n_freq = int(parsed['metadata']['n_frequencies'])
    tl = parsed['tl'] if n_freq > 1 else parsed['tl'][None]
    ranges = parsed['ranges'] if n_freq > 1 else parsed['ranges'][None]
    labels = parsed['metadata'].get('plotted_frequencies')
    if frequencies is not None:
        frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
        if frequencies.shape != (n_freq,):
            raise ConfigurationError(
                f"read_oast_tl: {filepath} holds {n_freq} plotted frequency "
                f"block(s); frequencies= gives {frequencies.size}.",
                remediation="Pass one frequency (Hz) per plotted block, in "
                            "the order of the OAST deck's frequencies.")
    if labels is None:
        if frequencies is None:
            raise FileFormatError(
                f"{filepath} carries no 'Freq:' labels; pass "
                f"frequencies=[...] (the OAST deck's frequencies).")
    else:
        labels = np.asarray(labels, dtype=float)
        if frequencies is not None and \
                np.any(np.abs(frequencies - labels) > 0.05):
            raise ConfigurationError(
                f"read_oast_tl: frequencies={frequencies.tolist()} Hz "
                f"disagrees with the 'Freq:' labels {labels.tolist()} Hz "
                f"of {filepath}.",
                remediation="Omit frequencies= to use the labels, or pass "
                            "the deck the file came from.")
        # The labels are F7.1 print roundings: frequencies= that agrees
        # with them is kept at its own precision, as receiver depths are
        # matched back to the caller's axis (_match_label_depths).
        if frequencies is None:
            frequencies = labels
    return [Field(data=tl[i], coords={'depth': parsed['depths'],
                                      'range': ranges[i]},
                  model='', backend='', frequencies=float(frequencies[i]))
            for i in range(n_freq)]


@typed_format_error
def _read_oasn_covariance_payload(
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
        - 'surface_noise_level_dB': float, surface noise level (dB)
        - 'white_noise_level_dB': float, white noise level (dB)
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
    >>> data = _read_oasn_covariance_payload('test.xsm')
    >>> cov = data['covariance']  # shape: (n_freq, n_rcv, n_rcv)
    >>> print(f"Covariance for {data['n_receivers']} receivers")
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise MissingFileError(
            f"OASN covariance file not found: {filepath}.",
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

    with open(filepath, 'rb') as f:
        # Probe the byte order from the first int32 (n_rcv at record 5).
        f.seek(4 * recl)
        probe = f.read(4)
        endian = detect_endian(probe, source=f'_read_oasn_covariance_payload:{filepath.name}')

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
                f"{n_total} records of {recl} bytes, got {len(buf)} bytes."
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
        'surface_noise_level_dB': sslev,
        'white_noise_level_dB': wnlev,
        'covariance': covariance
    }


@typed_format_error
def _read_oasn_replicas_payload(
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
        - 'x_min', 'x_max', 'n_x': replica x-offset grid (m)
        - 'y_min', 'y_max', 'n_y': replica y-offset grid (m)
        - 'receiver_positions': ndarray, shape (n_rcv, 3) [x, y, z] in m
        - 'receiver_types': ndarray, receiver types
        - 'receiver_gains_dB': ndarray, receiver gains (dB)
        - 'replicas': ndarray, shape (n_freq, n_z, n_x, n_y, n_rcv)
                     Complex replica fields

    Notes
    -----
    Every grid field is returned in metres. The file holds depth in m but
    the x/y offsets in km (``doc/oasn.tex:109-111``), which are converted
    here. Receiver coordinates are all in m (``doc/oasn.tex:47-51``).

    Examples
    --------
    >>> data = _read_oasn_replicas_payload('test.rpo')
    >>> replicas = data['replicas']  # shape: (n_freq, n_z, n_x, n_y, n_rcv)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise MissingFileError(
            f"OASN replica file not found: {filepath}.",
            remediation="Re-run OASN with 'R' in the option string: the "
                        "replica fields reach the .rpo file only under that "
                        "letter.",
        )

    with open(filepath, 'rb') as f:
        head = f.read(4)
        f.seek(0)
        endian = detect_endian(
            head, source=f'_read_oasn_replicas_payload:{filepath.name}',
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
        receiver_gains_dB = np.zeros(n_rcv)

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
            receiver_gains_dB[i] = 20.0 * np.log10(gain) if gain > 0 else -np.inf

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
                f"{n_total} records, got {flat.size}."
            )
        if np.any(flat['m1'] != 8) or np.any(flat['m2'] != 8):
            raise FileFormatError(
                f"{filepath}: unexpected replica record layout — "
                "Fortran record markers are not the expected 8-byte "
                "payload length."
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
        'x_min': float(km_to_m(x_min)),
        'x_max': float(km_to_m(x_max)),
        'n_x': n_x,
        'y_min': float(km_to_m(y_min)),
        'y_max': float(km_to_m(y_max)),
        'n_y': n_y,
        'receiver_positions': receiver_positions,
        'receiver_types': receiver_types,
        'receiver_gains_dB': receiver_gains_dB,
        'replicas': replicas
    }


def read_oasp_trf(
    filepath: Union[str, Path],
    receiver_depths: np.ndarray,
) -> Field:
    """
    Read an OASP transfer function file (.trf) as a pressure Field.

    OASP's default output parameter ``'K'`` is the bulk pressure, the
    acoustic pressure in a fluid (``oases/doc/oasp.tex:178-183``), read as
    written; the normal stress ``'N'``, ``sigma_zz = -p`` in fluids
    (``oasp.tex:185``), is negated. Both are on the ``e^{+i omega t}``
    convention every engine shares — the conversion the OASP model applies
    to its own output (:func:`~uacpy.io._parsers.oasp_trf_pressure`). Inside
    an elastic layer ``'K'`` is the bulk pressure of the compressional
    potential alone, and ``-'N'`` a stress built from both potentials
    (``oasaun31.f:1489-1491``, ``:1509-1513``).

    Parameters
    ----------
    filepath : str or Path
        Path to the ``.trf`` file.
    receiver_depths : ndarray
        Receiver depth axis (m), taken verbatim from the caller that wrote
        the deck. The ``.trf`` cannot supply it: TRFHEAD writes the explicit
        depth list only for ``IR < 0`` (oasiun23.f:870-877), but INREC has
        already flipped IR positive in COMMON /VARS1/ (oaseun31.f:1185,
        oases/src/compar.f:69-70) by the time TRFHEAD runs, so the header
        carries only RD, RDLOW and ``|IR|`` — a uniform grid — whatever the
        deck asked for. The header's count is cross-checked against this
        axis.

    Returns
    -------
    field : Field
        Complex pressure, coords ``(depth, range, frequency)``,
        ``phase_reference='travelling_wave'``, the source depth as
        ``source_depths`` and the file's title and centre frequency in
        ``metadata``; ``model`` is empty, the file being the source.

    Raises
    ------
    MissingFileError
        The file is missing.
    FileFormatError
        The file is not a PULSETRF file.
    UnsupportedFeatureError
        The file holds another output parameter (``V``/``H`` velocity,
        ``R``/``S`` stress): a Field carries pressure, and labelling a
        velocity or a stress as one would misread it. uacpy has no reader
        for those records.
    """
    raw = parse_oasp_trf(filepath, receiver_depths)
    return Field(
        data=oasp_trf_pressure(raw, 'read_oasp_trf', filepath),
        coords={'depth': raw['depths'], 'range': raw['ranges'],
                'frequency': raw['freq']},
        model='', backend='',
        phase_reference='travelling_wave',
        frequencies=raw['freq'],
        source_depths=np.array([raw['source_depth']]),
        metadata={'title': raw['title'],
                  'fc': raw['fc']})


@typed_format_error
def _read_oasr_reflection_payload(
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
        - 'angles_or_slowness': list of ndarray, grazing angle (**degrees**)
          for a ``.trc`` or slowness (**s/m**) for a ``.rco``
        - 'magnitude': list of ndarray, reflection coefficient magnitude
        - 'phase': list of ndarray, reflection coefficient phase (**radians**)

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
    both (oasjun21.f:102-104). The reader converts once: slowness to s/m
    (``/ 1e3``) and phase to radians; the angle column stays in degrees.

    Examples
    --------
    >>> data = _read_oasr_reflection_payload('test.trc')
    >>> mag = data['magnitude'][0]  # First frequency
    >>> angles = data['angles_or_slowness'][0]
    >>> print(f"Reflection coefficient at 45°: {mag[np.argmin(np.abs(angles-45))]}")
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise MissingFileError(
            f"OASR reflection coefficient file not found: {filepath}.",
            remediation="Re-run OASR with 'T' in the option string: the "
                        "reflection coefficient table reaches the .rco/.trc "
                        "files only under that letter.",
        )

    # Auto-detect from the file's own content: the header's fourth field is
    # the sampling-type code OASR stamps as it writes (unoasr21.f:204-205),
    # while the extension is only what the file is called. The extension is
    # kept as a hint so a disagreement can be reported rather than followed —
    # a renamed table is reported, not relabelled.
    extension_hint = None
    if format_type == 'auto':
        if filepath.suffix == '.rco':
            extension_hint = 'slowness'
        elif filepath.suffix == '.trc':
            extension_hint = 'angle'
        format_type = None

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
            # OASR stamps 1 on the slowness table and 2 on the angle table
            # (unoasr21.f:204-205 with oasjun21.f:103-104).
            if sampling_type_code not in (1, 2):
                raise FileFormatError(
                    f"{filepath}: header sampling code "
                    f"{sampling_type_code}; OASR writes 1 (slowness) or 2 "
                    f"(angle), so this is not an OASR reflection table.")

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
                        IOWarning, skip_file_prefixes=USER_FRAME_SKIP)

            sampling_type = format_type
        else:
            raise FileFormatError(f"Invalid header format: {header_line}.")

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
            # File units are s/km (``slw*1e3``) and degrees (``omr*atan2z``),
            # oasjun21.f:102-104; the dict carries s/m and radians.
            abscissa = np.array(angles_or_slowness)
            if sampling_type == 'slowness':
                abscissa = abscissa / 1e3
            angles_or_slowness_list.append(abscissa)
            magnitude_list.append(np.array(magnitude))
            phase_list.append(np.deg2rad(np.array(phase)))

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


@dataclass(frozen=True, eq=False)
class OasesRhsHeader(ExportRecord):
    """The header of a mean-field ``.rhs`` as :func:`read_oases_rhs_header`
    reads it; the field names are OASP's reading of the first record (that
    function says what an OAST or OASR producer writes into the same slots).

    Attributes
    ----------
    n_time_samples : int
        ``NX`` (``nfreq`` from OAST/OASR).
    freq_min, freq_max : float
        ``FR1``, ``FR2`` (Hz); ``freq_min`` is *the* frequency from
        OAST/OASR.
    time_step : float
        ``DT`` (s; ``1/(nfreq*dlfreq)`` from OAST/OASR).
    frequency : float
        The per-frequency header's own ``FREQ`` (Hz).
    source_layer : int
        ``LAYS(1)``.
    n_wavenumbers : int
        ``NWVNO``.
    interface : int
        ``IN1``, the 1-based OASES layer index of the first rough
        interface.
    """

    n_time_samples: int
    freq_min: float
    freq_max: float
    time_step: float
    frequency: float
    source_layer: int
    n_wavenumbers: int
    interface: int

    _REPR_UNITS = {'freq_min': 'Hz', 'freq_max': 'Hz', 'time_step': 's',
                   'frequency': 'Hz'}


@typed_format_error
def read_oases_rhs_header(filepath: Union[str, Path]) -> OasesRhsHeader:
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
    (``unoass21.f:123``, ``:136``) — and ``freq_min`` is the single frequency
    the mean field ran at. The keys are not renamed per producer: the file
    layout is one layout, and a caller that knows which producer it drove knows
    which reading applies.

    OASSP reads exactly the same three records — the first at
    ``unoassp30.f:180``, the other two at ``:548-549`` — and then **replaces**
    its own deck's ``NT/FR1/FR2/DT`` with the first record's values, warning
    only when ``NX`` or ``DT`` differ (``:182-188``). A caller that writes the
    deck from these values instead of from a second guess makes that
    substitution a no-op, which is the only way to be sure the band that ran is
    the band that was asked for.

    ``interface`` is the third record's ``IN1``: ``SCTRHS`` loops over
    interfaces innermost (``oaseun31.f:2306-2308``), so the first such record
    carries the shallowest rough interface, and that is the single interface
    OASSP scatters from.

    Parameters
    ----------
    filepath : str or Path
        The file to read.

    Returns
    -------
    OasesRhsHeader
        ``n_time_samples`` (int, ``NX`` — ``nfreq`` from OAST/OASR),
        ``freq_min`` / ``freq_max`` (Hz; ``freq_min`` is *the* frequency from
        OAST/OASR), ``time_step`` (s, ``DT`` — ``1/(nfreq*dlfreq)`` from
        OAST/OASR), ``frequency`` (Hz, the per-frequency header's own
        ``FREQ``), ``source_layer`` (``LAYS(1)``), ``n_wavenumbers``
        (``NWVNO``), ``interface`` (``IN1``, 1-based OASES layer index).
    """
    filepath = Path(filepath)

    if not filepath.is_file():
        raise MissingFileError(
            f"OASES .rhs mean-field file not found: {filepath}.",
            remediation="Re-run the producer with 's' in the option string, "
                        "over an environment carrying a rough interface: "
                        "SCTRHS skips every interface with ROUGH2 < 1e-10 "
                        "(oaseun31.f:2310), so a smooth deck writes no .rhs "
                        "even under 's'.",
        )

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

    if int(nx) < 1 or float(dt) <= 0.0:
        raise FileFormatError(
            f"OASES .rhs header {filepath} carries NX={nx}, DT={dt}, which "
            f"cannot describe an FFT grid; the producer run probably wrote no "
            f"scattering right-hand sides (option 's' missing, or every "
            f"interface smooth — SCTRHS skips ROUGH2 < 1e-10, "
            f"oaseun31.f:2310)."
        )
    return OasesRhsHeader(
        n_time_samples=int(nx), freq_min=float(fr1), freq_max=float(fr2),
        time_step=float(dt), frequency=float(freq), source_layer=int(lays1),
        n_wavenumbers=int(nwvno), interface=int(interface))


# ---------------------------------------------------------------------------
# Public readers
#
# The private payload parsers above return each file as written. The public
# readers below return the result object the OASES models return —
# Covariance, Replicas, ReflectionCoefficient — with ``model`` left empty:
# the file, not a model run, is the source.
# ---------------------------------------------------------------------------

def _oasn_frequency_axis(data: Dict) -> np.ndarray:
    """The frequency axis (Hz) of an OASN ``.xsm`` / ``.rpo`` payload.

    The headers carry ``freq_min``, ``freq_max`` and ``n_frequencies``;
    OASN steps its frequency loop uniformly between the first two, so the
    axis is their ``linspace``.
    """
    n = int(data.get('n_frequencies', 1))
    f1 = float(data.get('freq_min', 0.0))
    f2 = float(data.get('freq_max', f1))
    if n <= 1:
        return np.array([f1], dtype=float)
    return np.linspace(f1, f2, n, dtype=float)


def _depth_positions(receiver_depths, n_receivers: int):
    """``(n, 3)`` ``[x, y, z]`` receiver positions for a vertical array of
    ``receiver_depths``, or ``None`` when the depths do not match the file's
    receiver count."""
    if receiver_depths is None:
        return None
    depths = np.atleast_1d(np.asarray(receiver_depths, dtype=float))
    if depths.size != n_receivers:
        return None
    positions = np.zeros((depths.size, 3), dtype=float)
    positions[:, 2] = depths
    return positions


def _header_metadata(data: Dict, arrays) -> Dict:
    """Every header field of a payload except the ``arrays`` the carrier
    holds as attributes."""
    return {key: value for key, value in data.items() if key not in arrays}


def read_oasn_covariance(
    filepath: Union[str, Path],
    receiver_depths: Optional[np.ndarray] = None,
    *,
    unit: str = '',
):
    """Read an OASN/OASS ``.xsm`` as a :class:`~uacpy.core.results.Covariance`.

    The file is Fortran direct access with 8-byte records: ten header
    records (title, receiver and frequency counts, ``FREQ1``/``FREQ2``/
    ``DELFRQ``, the surface and white noise levels), then the complex
    covariance matrix ``COVMAT(IRCV, JRCV, IFREQ)`` IRCV-fastest
    (``oasn.tex:583-618``), returned as ``covariance[ifreq, ircv, jrcv]``
    in the binary's own units: the square of the reference the deck's
    levels are in (µPa²/Hz for OASN noise levels re 1 µPa²/Hz). The OASN
    model converts it to Pa²/Hz.

    Parameters
    ----------
    filepath : str or Path
        The ``.xsm`` file.
    receiver_depths : ndarray, optional
        Depths (m) of the array's elements. The file carries no positions;
        given, and one per receiver, they fill ``receiver_positions`` as a
        vertical array at ``x = y = 0``.
    unit : str, optional
        The unit of the matrix as its producer states it, stamped on
        ``Covariance.unit``. The file carries none, and the two programs
        that write this layout differ: OASN writes µPa²/Hz (noise levels re
        1 µPa²/Hz), OASS the variance of a CW field in Pa² for a 1 Pa
        source (``oass.tex:216-218``). Default ``''``: not stated.

    Returns
    -------
    Covariance
        ``covariance`` ``(n_freq, n_rcv, n_rcv)`` in the file's own scale;
        ``unit`` as given; ``frequencies`` the header's uniform FREQ1…FREQ2
        axis; ``model`` empty. ``metadata``
        holds every header field: ``title``, ``n_receivers``,
        ``n_frequencies``, ``freq_min``, ``freq_max``, ``freq_delta``,
        ``surface_noise_level_dB``, ``white_noise_level_dB`` (dB).
    """
    from uacpy.core.results import Covariance
    data = _read_oasn_covariance_payload(filepath)
    return Covariance(
        covariance=data['covariance'],
        receiver_positions=_depth_positions(
            receiver_depths, data['covariance'].shape[1]),
        backend='oasn',
        frequencies=_oasn_frequency_axis(data),
        metadata=_header_metadata(data, ('covariance',)),
        unit=unit,
    )


def read_oasn_replicas(filepath: Union[str, Path]):
    """Read an OASN ``.rpo`` as a :class:`~uacpy.core.results.Replicas`.

    Sequential binary: the header (title, counts, the frequency sweep, the
    replica grid ``min max n`` per axis — z in m, x/y in km on disk,
    ``oasn.tex:109-111``), one record per array element (X, Y, Z in m, type,
    linear gain), then the replicas IRCV-innermost, then IYR, IXR, IZR,
    IFREQ (``oasn.tex:651-689``).

    Parameters
    ----------
    filepath : str or Path
        The file to read.

    Returns
    -------
    Replicas
        ``replicas`` ``(n_freq, n_z, n_x, n_y, n_rcv)``; ``candidates``
        ``{'depth', 'x', 'y'}``, the grid axes in metres (OASN interpolates
        each linearly between its endpoints, ``unoasn22.f:197-222``);
        ``receiver_positions`` from
        the file. ``metadata`` holds the header fields — ``title``,
        ``n_receivers``, the frequency sweep, the grid endpoints and counts
        (metres), ``receiver_types`` and ``receiver_gains_dB`` (dB).
    """
    from uacpy.core.results import Replicas
    data = _read_oasn_replicas_payload(filepath)
    return Replicas(
        replicas=data['replicas'],
        candidates={
            'depth': np.linspace(data['z_min'], data['z_max'], data['n_z']),
            'x': np.linspace(data['x_min'], data['x_max'], data['n_x']),
            'y': np.linspace(data['y_min'], data['y_max'], data['n_y']),
        },
        receiver_positions=data.get('receiver_positions'),
        backend='oasn',
        frequencies=_oasn_frequency_axis(data),
        metadata=_header_metadata(data, ('replicas', 'receiver_positions')),
    )


def _stack_oasr_blocks(data: Dict):
    """Stack a :func:`_read_oasr_reflection_payload` payload's
    per-frequency blocks into ``(theta, R, phi, frequencies)`` arrays.

    ``R`` and ``phi`` are ``(n_angles,)`` for one frequency and
    ``(n_angles, n_frequencies)`` otherwise; ``phi`` (radians) is unwrapped
    along the angle axis. OASR writes the phase as a principal value
    (``oases/src/oasjun21.f:102`` takes ``atan2z``), but every consumer
    interpolates it linearly between bracketing angles, which
    ``misc/RefCoef.f90:119`` states requires an unwrapped phase ("Assumes phi
    has been unwrapped so that it varies smoothly"); interpolating across a
    ±2π step sweeps the phase the long way round and returns a reflection
    coefficient of the wrong sign. Unwrapping is a no-op on an
    already-smooth table.

    Raises
    ------
    ConfigurationError
        No angle/slowness rows (OASR writes the table only under option
        ``'T'``).
    FileFormatError
        Blocks that do not share one angle grid, or a block whose magnitude
        or phase column is not as long as its angles.
    """
    from uacpy.core.exceptions import ConfigurationError
    freqs = np.asarray(data.get('frequencies', []), dtype=float)
    angle_lists = data.get('angles_or_slowness', [])
    R_lists = data.get('magnitude', [])
    phi_lists = data.get('phase', [])
    if not angle_lists:
        raise ConfigurationError(
            f"OASR reader returned no frequency samples: the file carries "
            f"{freqs.size} frequency value(s) and no angle/slowness rows. "
            f"Re-run OASR with 'T' in the option string — the reflection "
            f"coefficient table is written only under that letter.")
    theta = np.asarray(angle_lists[0], dtype=float)
    n_angles = len(theta)
    for i, (a, m, p) in enumerate(zip(angle_lists, R_lists, phi_lists)):
        if len(a) != n_angles:
            raise FileFormatError(
                f"OASR: angle grid mismatch between frequencies "
                f"(freq[{i}] has {len(a)} angles, freq[0] has {n_angles}). "
                f"Multi-frequency stacking requires a shared angle grid."
            )
        if len(m) != n_angles or len(p) != n_angles:
            raise FileFormatError(
                f"OASR: payload length mismatch at frequency index {i}: "
                f"angles={len(a)}, magnitude={len(m)}, phase={len(p)}."
            )
    if len(freqs) == 1:
        R = np.asarray(R_lists[0], dtype=float)
        phi_rad = np.asarray(phi_lists[0], dtype=float)
    else:
        R = np.column_stack([np.asarray(m, dtype=float) for m in R_lists])
        phi_rad = np.column_stack([np.asarray(p, dtype=float)
                                   for p in phi_lists])
    return theta, R, np.unwrap(phi_rad, axis=0), freqs


def read_oasr_reflection_coefficients(
    filepath: Union[str, Path],
    format_type: str = 'auto',
):
    """Read an OASR grazing-angle table (``.trc``) as a
    :class:`~uacpy.core.results.ReflectionCoefficient`.

    The per-frequency blocks are stacked into ``(n_angles[, n_frequencies])``
    arrays with the phase (radians) unwrapped along the angle axis. The
    header's fourth field is the abscissa code OASR stamps
    (``unoasr21.f:204-205``: 2 grazing angle, 1 slowness), and it decides
    the reading whatever the file is called; ``format_type`` is passed to
    the table parser.

    Parameters
    ----------
    filepath : str or Path
        The file to read.
    format_type : str, optional
        Passed to the table parser. Default ``'auto'``.

    Returns
    -------
    ReflectionCoefficient
        ``angles_deg`` (grazing), ``magnitude``, ``phase``; ``frequencies`` one
        per block; ``model`` empty; ``metadata`` the header fields
        (``sampling_type``, ``n_frequencies``, ``freq_min``, ``freq_max``).

    Raises
    ------
    UnsupportedFeatureError
        A slowness-sampled table (``.rco``): the carrier is indexed by
        grazing angle, and turning slowness into angle needs the incident
        medium's sound speed, which the file does not carry. OASR writes
        the angle table (``.trc``) beside it under option ``'T'``.
    ConfigurationError
        No angle rows (OASR writes the table only under ``'T'``), or
        blocks that do not share one angle grid.
    """
    from uacpy.core.results import ReflectionCoefficient
    data = _read_oasr_reflection_payload(filepath, format_type=format_type)
    if data.get('sampling_type') == 'slowness':
        raise UnsupportedFeatureError(
            'read_oasr_reflection_coefficients',
            f"{filepath} is sampled in slowness; a ReflectionCoefficient is "
            f"indexed by grazing angle and the file carries no sound speed "
            f"to convert with",
            alternatives=["the grazing-angle .trc OASR writes beside it "
                          "under option 'T'"],
            alternatives_label='files')
    theta, R, phi, freqs = _stack_oasr_blocks(data)
    return ReflectionCoefficient(
        angles_deg=theta, magnitude=R, phase=phi,
        backend='oasr',
        frequencies=freqs if len(freqs) else None,
        metadata=_header_metadata(
            data, ('frequencies', 'angles_or_slowness', 'magnitude',
                   'phase')),
    )
