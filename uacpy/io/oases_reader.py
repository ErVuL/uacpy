"""
OASES Output File Readers

This module provides functions for reading output files from OASES models:
- OAST: .plp/.plt files (transmission loss)
- OASN: .xsm files (covariance matrices), .rpo files (replicas)
- Mode files: .mod files (mode shapes and wavenumbers)

OASES (Ocean Acoustics and Seismic Exploration Synthesis) was developed by
Henrik Schmidt at MIT.

References:
    Schmidt, H. OASES Version 2.1 User Guide and Reference Manual (bundled
    under ``third_party/oases``). Public OASES is 3.1 but the distribution
    vendored here is 2.1 — see the bundled README.
"""

from pathlib import Path
from typing import Dict, Tuple, Union
import numpy as np
import struct

from uacpy.core.exceptions import FileFormatError, UnsupportedFeatureError
from uacpy.io._fortran_helpers import (
    PARSE_ERRORS,
    read_fortran_record_marker as _read_fortran_record_marker,
    read_fortran_record as _read_fortran_record,
    detect_endian,
)
from uacpy.io.units import km_to_m


def _bound_counts(filepath, file_size, min_item_bytes, **counts):
    """Reject header counts that cannot be satisfied by ``file_size``.

    Binary OASES readers size NumPy allocations directly off integer
    header fields (``n_rcv``, ``n_freq``, grid extents). A corrupt or
    hostile file with a garbage count (e.g. ``n_rcv = 0x7fffffff``) would
    otherwise drive a multi-GB/TB ``np.zeros`` before any data record is
    validated. The smallest a single data item can occupy on disk is
    ``min_item_bytes``, so no count — nor their product — can exceed
    ``file_size // min_item_bytes``. Raise :class:`FileFormatError` on
    overflow rather than attempting the allocation.
    """
    max_items = file_size // max(min_item_bytes, 1)
    product = 1
    for name, val in counts.items():
        if val < 0:
            raise FileFormatError(
                f"{filepath}: negative header count {name}={val}."
            )
        if val > max_items:
            raise FileFormatError(
                f"{filepath}: header count {name}={val} is implausible for "
                f"a {file_size}-byte file (max {max_items} items)."
            )
        product *= val
    if product > max_items:
        raise FileFormatError(
            f"{filepath}: header counts {dict(counts)} imply {product} data "
            f"items, implausible for a {file_size}-byte file "
            f"(max {max_items})."
        )


def read_oast_tl(
    filepath: Union[str, Path],
    receiver_depths: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Read OAST transmission loss output on its native grid.

    OAST outputs two files:
    - .plp: Plot metadata (ASCII with binary markers) - contains grid info
    - .plt: Actual TL data (pure ASCII, one value per line)

    OAST writes TL (in dB) directly to disk; this reader returns the
    native ``(n_depths, n_ranges)`` TL grid plus the depth and range
    axes. Resampling onto a user receiver grid is the caller's job —
    use :meth:`Field.resample_to` after wrapping.

    Parameters
    ----------
    filepath : str or Path
        Path to .plt or .plp file (base name works for both)
    receiver_depths : ndarray
        Receiver depth axis (m). OAST writes TL on this depth grid; the
        depth axis is taken verbatim from the user.

    Returns
    -------
    tl_data : ndarray
        Transmission loss data on the OAST native grid, shape
        ``(n_depths, n_ranges_native)``.
    depths : ndarray
        Depth axis (m), == ``receiver_depths``.
    ranges : ndarray
        OAST's native range grid in metres.
    metadata : dict
        ``{'model': 'OAST', 'oast_grid_shape': (n_d, n_r_native)}``.

    Raises
    ------
    IOError
        If the ``.plp`` file is missing or cannot be parsed (OAST chooses
        its own range grid via FFT-based sampling, so the native grid
        cannot be reconstructed without it).
    """
    filepath = Path(filepath)

    # Get file paths - OAST can output to .plt, .plp, or .020 (Fortran unit 20)
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
        raise FileFormatError(f"OAST TL data file not found. Checked: {plt_file}, {f020_file}")

    # Parse .plp file to get OAST's native range grid. The grid is
    # mandatory: OAST chooses its own ranges via FFT-based sampling, so
    # without .plp we have no way to know what range each TL value
    # corresponds to. Raise rather than fabricate.
    if not plp_file.exists():
        raise FileFormatError(
            f"OAST .plp grid file not found: {plp_file}. "
            "Without it the native range grid cannot be reconstructed."
        )
    oast_grid = _parse_oast_plp(plp_file)

    n_depths_oast = len(receiver_depths)  # OAST writes TL on this depth grid
    n_ranges_oast = oast_grid['n_ranges']
    ranges_oast = oast_grid['ranges']

    # Read all TL values from data file (.plt or .020)
    tl_values = []
    with open(tl_data_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip header lines and empty lines
            if line and not any(keyword in line for keyword in ['MODU', 'OASTL', '$', 'AXIS', 'TITLE']):
                try:
                    tl_values.append(float(line))
                except ValueError:
                    pass  # Skip non-numeric lines

    tl_values = np.array(tl_values)

    if len(tl_values) == 0:
        raise FileFormatError(f"No TL data found in {tl_data_file}")

    # OAST outputs data as: all ranges for depth 1, all ranges for depth 2, etc.
    expected_total = n_depths_oast * n_ranges_oast

    if len(tl_values) < expected_total:
        missing = expected_total - len(tl_values)
        raise FileFormatError(
            f"Truncated OAST output: got {len(tl_values)} TL values, "
            f"expected {expected_total} "
            f"(n_depths={n_depths_oast}, n_ranges={n_ranges_oast}); "
            f"{missing} cells missing. A truncated run (crash, disk-full, "
            f"killed job) cannot be completed without fabricating TL. "
            f"File: {tl_data_file}"
        )

    tl_oast = tl_values[:expected_total].reshape(n_depths_oast, n_ranges_oast)

    metadata = {
        'oast_grid_shape': (n_depths_oast, n_ranges_oast),
    }
    return tl_oast, np.asarray(receiver_depths, dtype=float), ranges_oast, metadata


def _parse_oast_plp(plp_file: Path) -> Dict:
    """
    Parse OAST .plp file to extract grid information

    Returns dictionary with:
    - n_ranges: number of ranges (N in OAST output)
    - ranges: array of range values in meters
    - range_offset: starting range (XOFF in km)
    - range_increment: range step (DX in km)
    """
    try:
        with open(plp_file, 'rb') as f:
            content = f.read()

        # Decode as ASCII, ignoring binary sections
        text = content.decode('ascii', errors='ignore')
        lines = text.split('\n')

        n_ranges = None
        xoff = None
        dx = None

        for i, line in enumerate(lines):
            # Look for key parameters
            if 'NC' in line and 'ZINC' not in line and 'INC' not in line:
                # Number of curves
                try:
                    # Next line should have N (number of points)
                    if i + 1 < len(lines) and n_ranges is None:
                        next_line = lines[i + 1]
                        if 'N' in next_line and 'NUMBER' not in next_line:
                            try:
                                n_ranges = int(next_line.split()[0])
                            except (ValueError, IndexError):
                                pass
                except (ValueError, IndexError):
                    pass
            elif 'XOFF' in line:
                try:
                    xoff = float(line.split()[0])  # in km
                except (ValueError, IndexError):
                    pass
            elif 'DX' in line and 'XDIV' not in line and 'XINC' not in line:
                try:
                    dx = float(line.split()[0])  # in km
                except (ValueError, IndexError):
                    pass

        if n_ranges is None or xoff is None or dx is None:
            raise FileFormatError(
                f"Could not parse grid from .plp file: "
                f"n_ranges={n_ranges}, xoff={xoff}, dx={dx}"
            )

        ranges = km_to_m(xoff + np.arange(n_ranges) * dx)

        return {
            'n_ranges': n_ranges,
            'ranges': ranges,
            'range_offset_km': xoff,
            'range_increment_km': dx
        }

    except (FileFormatError, UnsupportedFeatureError):
        raise
    except PARSE_ERRORS as e:
        raise FileFormatError(f"Failed to parse OAST .plp file: {e}") from e


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
    Record length is 8 bytes on most systems, but may be 2 words on some
    (e.g., DEC workstations). This function assumes 8-byte records.

    Examples
    --------
    >>> data = read_oasn_covariance('test.xsm')
    >>> cov = data['covariance']  # shape: (n_freq, n_rcv, n_rcv)
    >>> print(f"Covariance for {data['n_receivers']} receivers")
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileFormatError(f"OASN covariance file not found: {filepath}")

    # Record length in bytes (8 bytes = 1 complex64 = 2 float32)
    recl = 8

    try:
        with open(filepath, 'rb') as f:
            # Probe the byte order from the first int32 (n_rcv at record 5).
            f.seek(4 * recl)
            probe = f.read(4)
            endian = detect_endian(probe, source=f'read_oasn_covariance:{filepath.name}')

            # Read header (first 10 records)
            # Record 1-4: Title (4 x 8 bytes = 32 characters)
            title_parts = []
            for i in range(4):
                f.seek(i * recl)
                data_bytes = f.read(recl)
                title_parts.append(data_bytes.decode('ascii', errors='ignore').strip())
            title = ''.join(title_parts).strip()

            # Record 5: NRCV, NFREQ (2 integers)
            f.seek(4 * recl)
            n_rcv, n_freq = struct.unpack(endian + 'ii', f.read(8))

            # Bound the n_freq*n_rcv*n_rcv covariance allocation against the
            # file size before the strided read / ljust below.
            f.seek(0, 2)
            file_size = f.tell()
            _bound_counts(filepath, file_size, recl,
                          n_rcv=n_rcv, n_freq=n_freq, n_rcv2=n_rcv)

            # Record 6: IZERO, IZERO (dummy)
            # Skip

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

            # Read covariance matrices. Data starts at record 11 (offset
            # 10 * recl); one complex value (re, im float32) sits at the head
            # of each ``recl``-byte record, ordered (ifreq, jrcv, ircv) with
            # ircv innermost. A structured dtype with ``itemsize=recl`` strides
            # over the records in a single read.
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
            # Stored (ifreq, jrcv, ircv); the matrix wants (ifreq, ircv, jrcv).
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
        - 'z_min', 'z_max', 'n_z': replica depth grid
        - 'x_min', 'x_max', 'n_x': replica x-range grid
        - 'y_min', 'y_max', 'n_y': replica y-range grid
        - 'receiver_positions': ndarray, shape (n_rcv, 3) [x, y, z]
        - 'receiver_types': ndarray, receiver types
        - 'receiver_gains': ndarray, receiver gains (dB)
        - 'replicas': ndarray, shape (n_freq, n_z, n_x, n_y, n_rcv)
                     Complex replica fields

    Examples
    --------
    >>> data = read_oasn_replicas('test.rpo')
    >>> replicas = data['replicas']  # shape: (n_freq, n_z, n_x, n_y, n_rcv)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileFormatError(f"OASN replica file not found: {filepath}")

    try:
        with open(filepath, 'rb') as f:
            head = f.read(4)
            f.seek(0)
            endian = detect_endian(
                head, source=f'read_oasn_replicas:{filepath.name}',
            )

            # Read title (80 characters)
            _read_fortran_record_marker(f, endian=endian)
            title = f.read(80).decode('ascii', errors='ignore').strip()
            _read_fortran_record_marker(f, endian=endian)

            # Read NRCV, NFREQ
            _read_fortran_record_marker(f, endian=endian)
            n_rcv, n_freq = struct.unpack(endian + 'ii', f.read(8))
            _read_fortran_record_marker(f, endian=endian)

            # Read FREQ1, FREQ2, DELFRQ
            _read_fortran_record_marker(f, endian=endian)
            freq1, freq2, delfrq = struct.unpack(endian + 'fff', f.read(12))
            _read_fortran_record_marker(f, endian=endian)

            # Read replica grid: ZMINR, ZMAXR, NZR
            _read_fortran_record_marker(f, endian=endian)
            z_min, z_max, n_z = struct.unpack(endian + 'ffi', f.read(12))
            _read_fortran_record_marker(f, endian=endian)

            # Read XMINR, XMAXR, NXR
            _read_fortran_record_marker(f, endian=endian)
            x_min, x_max, n_x = struct.unpack(endian + 'ffi', f.read(12))
            _read_fortran_record_marker(f, endian=endian)

            # Read YMINR, YMAXR, NYR
            _read_fortran_record_marker(f, endian=endian)
            y_min, y_max, n_y = struct.unpack(endian + 'ffi', f.read(12))
            _read_fortran_record_marker(f, endian=endian)

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
                _read_fortran_record_marker(f, endian=endian)
                x, y, z, itype, gain = struct.unpack(
                    endian + 'fffif', f.read(20),
                )
                _read_fortran_record_marker(f, endian=endian)
                receiver_positions[i] = [x, y, z]
                receiver_types[i] = itype
                receiver_gains[i] = gain

            # Each replica is a Fortran sequential record
            # ``[marker][re im][marker]`` (16 bytes), written contiguously in
            # (ifreq, iz, ix, iy, ircv) order with ircv innermost. Read the
            # whole block in one strided pass.
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


def read_oasp_trf(
    filepath: Union[str, Path]
) -> Dict:
    """
    Read OASP transfer function file (.trf format)

    OASP outputs transfer functions for postprocessing with PP module.
    These are complex frequency-domain responses.

    Supports both binary (Fortran unformatted) and ASCII (formatted) TRF files.

    Parameters
    ----------
    filepath : str or Path
        Path to .trf file

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
        - 'model': str, 'OASP'

    Notes
    -----
    Transfer function files can be binary with Fortran record markers
    or ASCII formatted. The reader attempts binary first, then ASCII.
    Format follows OASES PULSETRF specification from trford.f/oasiun23.f.

    Examples
    --------
    >>> data = read_oasp_trf('pulse.trf')
    >>> trf = data['transfer_function']  # shape: (n_freq, n_range, n_depth)
    >>> print(f"Transfer functions for {data['n_depths']} depths")
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileFormatError(f"OASP transfer function file not found: {filepath}")

    # Try Fortran-unformatted binary first (current OASES default).
    errors = []
    try:
        return _read_oasp_trf_binary(filepath)
    except (FileFormatError,) + PARSE_ERRORS as e:
        errors.append(('fortran-unformatted', e))

    # ASCII path always raises NotImplemented, but wrap so the binary
    # error surfaces when both paths fail.
    try:
        return _read_oasp_trf_ascii(filepath)
    except (FileFormatError, UnsupportedFeatureError) + PARSE_ERRORS as e:
        errors.append(('ascii', e))

    err_msg = '\n'.join(f"  {k}: {v}" for k, v in errors)
    raise FileFormatError(
        f"Failed to read OASP transfer function file {filepath}.\n{err_msg}"
    )


def _read_oasp_trf_binary(filepath: Path) -> Dict:
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

        title = _read_fortran_record(f, raw=True, endian=endian).decode(
            'ascii', errors='ignore').strip()
        # signn record consumed but not used
        _read_fortran_record(f, raw=True, endian=endian)

        (freqs,) = _read_fortran_record(f, 'f', endian=endian)
        (sd,) = _read_fortran_record(f, 'f', endian=endian)

        rd, rdlow, ir = _read_fortran_record(f, 'ffi', endian=endian)
        if ir < 0:
            nrd = abs(ir)
            rdc = np.array(
                _read_fortran_record(f, f'{nrd}f', endian=endian),
                dtype=np.float64,
            )
            receiver_depths = rdc
        else:
            nrd = max(1, ir)
            if nrd > 1:
                receiver_depths = np.linspace(rd, rdlow, nrd)
            else:
                receiver_depths = np.array([rd])

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
        (icdr,) = _read_fortran_record(f, 'i', endian=endian)
        (omegim,) = _read_fortran_record(f, 'f', endian=endian)

        (msuft,) = _read_fortran_record(f, 'i', endian=endian)
        (isrow,) = _read_fortran_record(f, 'i', endian=endian)
        (inttyp,) = _read_fortran_record(f, 'i', endian=endian)
        for _ in range(2):
            _read_fortran_record(f, 'i', endian=endian)
        for _ in range(5):
            _read_fortran_record(f, 'f', endian=endian)

        # --- Data records ---
        nf = max(1, mx - lx + 1)
        # OASES bin indices are 1-based: oasiun22.f:1256-1261 sets
        # DLFRQP = 1/(DT*NX) and LX = nint(FMIN/DLFRQP + 1), so bin k carries
        # frequency (k-1)*DLFRQ. Using k/(dt*nx) puts the whole axis one bin
        # (= 1/(dt*nx) Hz) too high.
        freq_array = np.array(
            [((k - 1) / (dt * nx)) for k in range(lx, mx + 1)],
            dtype=np.float64,
        ) if nf >= 1 else np.array([freqs], dtype=np.float64)

        # Detect the data-record precision from the first record's length
        # marker so both OASES output modes are supported: default COMPLEX*8
        # (2*nout float32 = 2*nout*4 bytes) and the double-precision '8' option
        # COMPLEX*16 (2*nout float64 = 2*nout*8 bytes).
        pos = f.tell()
        marker = f.read(4)
        data_fmt = f'{2 * nout}f'
        out_dtype = np.complex64
        if len(marker) == 4:
            (rec_bytes,) = struct.unpack(endian + 'i', marker)
            if rec_bytes == 2 * nout * 8:
                data_fmt = f'{2 * nout}d'      # double-precision COMPLEX*16
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

        transfer_function = np.zeros((nf, nplots, nrd), dtype=out_dtype)
        for j in range(nf):
            for jrh in range(nplots):
                for jrv in range(nrd):
                    rec = _read_fortran_record(f, data_fmt, endian=endian)
                    transfer_function[j, jrh, jrv] = complex(rec[0], rec[1])

    return {
        'title': title,
        'option': ''.join(chr(ord('A') - 1 + p) for p in iparm if 0 < p < 27),
        'freq': freq_array,
        'ranges': ranges,
        'depths': receiver_depths,
        'transfer_function': transfer_function,
        'source_depth': float(sd),
        'center_frequency': float(freqs),
    }


def _read_oasp_trf_ascii(filepath: Path) -> Dict:
    """Read ASCII (formatted) TRF file.

    ASCII TRF reading is not implemented — the previous stub silently returned
    ``np.ones(...)`` for the transfer function, which produced bogus TL values
    downstream (uniform 0 dB). OASES is expected to be run with binary TRF
    output (the default); if users genuinely need ASCII TRF support they can
    open a PR with the proper payload reader.

    Raises
    ------
    UnsupportedFeatureError
        Always. Either re-run OASES with binary TRF (default) or extend this
        reader to parse the ASCII payload.
    """
    raise UnsupportedFeatureError(
        "oases_reader",
        f"ASCII TRF reader for {filepath}",
        alternatives=[
            "Re-run OASES with binary (Fortran-unformatted) TRF output — "
            "the default; do not pass any ASCII conversion option."
        ],
    )


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
        'slowness' for .rco files, 'angle' for .trc files, or 'auto' to detect
        from file extension or content (default: 'auto')

    Returns
    -------
    data : dict
        Dictionary containing:
        - 'freq_min': float, minimum frequency (Hz)
        - 'freq_max': float, maximum frequency (Hz)
        - 'n_frequencies': int, number of frequencies
        - 'sampling_type': str, 'slowness' or 'angle'
        - 'frequencies': list of ndarray, frequency array for each freq
        - 'angles_or_slowness': list of ndarray, angle (deg) or slowness (s/km)
        - 'magnitude': list of ndarray, reflection coefficient magnitude
        - 'phase': list of ndarray, reflection coefficient phase (degrees)
        - 'model': str, 'OASR'

    Notes
    -----
    File format (from OASES documentation):
    Line 1: freq_min freq_max n_freq sampling_type
            where sampling_type is 1 for slowness, 2 for angle
    For each frequency:
        Line: frequency n_samples
        Then n_samples lines of:
            angle/slowness magnitude phase

    Examples
    --------
    >>> data = read_oasr_reflection_coefficients('test.trc')
    >>> mag = data['magnitude'][0]  # First frequency
    >>> angles = data['angles_or_slowness'][0]
    >>> print(f"Reflection coefficient at 45°: {mag[np.argmin(np.abs(angles-45))]}")
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileFormatError(f"OASR reflection coefficient file not found: {filepath}")

    # Auto-detect format from extension
    if format_type == 'auto':
        if filepath.suffix == '.rco':
            format_type = 'slowness'
        elif filepath.suffix == '.trc':
            format_type = 'angle'
        else:
            # Will be detected from file content
            format_type = None

    try:
        with open(filepath, 'r') as f:
            # Read header line
            header_line = f.readline().strip()
            header_parts = header_line.split()

            if len(header_parts) >= 4:
                freq_min = float(header_parts[0])
                freq_max = float(header_parts[1])
                n_freq = int(header_parts[2])
                sampling_type_code = int(header_parts[3])

                # Decode sampling type
                if format_type is None:
                    format_type = 'slowness' if sampling_type_code == 1 else 'angle'

                sampling_type = format_type
            else:
                raise FileFormatError(f"Invalid header format: {header_line}")

            # Read data for each frequency
            frequencies = []
            angles_or_slowness_list = []
            magnitude_list = []
            phase_list = []

            for _ in range(n_freq):
                # Read frequency header
                freq_header = f.readline().strip().split()
                if len(freq_header) >= 2:
                    freq = float(freq_header[0])
                    n_samples = int(freq_header[1])
                else:
                    continue

                # Read samples
                angles_or_slowness = []
                magnitude = []
                phase = []

                for _ in range(n_samples):
                    line = f.readline().strip()
                    if not line:
                        break
                    parts = line.split()
                    if len(parts) >= 3:
                        angles_or_slowness.append(float(parts[0]))
                        magnitude.append(float(parts[1]))
                        phase.append(float(parts[2]))

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
