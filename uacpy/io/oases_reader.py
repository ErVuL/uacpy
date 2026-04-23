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
import warnings


def read_oast_tl(
    filepath: Union[str, Path],
    receiver_depths: np.ndarray,
    receiver_ranges: np.ndarray,
    interpolate: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Read OAST transmission loss output

    OAST outputs two files:
    - .plp: Plot metadata (ASCII with binary markers) - contains grid info
    - .plt: Actual TL data (pure ASCII, one value per line)

    OAST uses FFT-based automatic range sampling, so we:
    1. Parse .plp to get OAST's actual grid
    2. Read .plt data on that grid
    3. Optionally interpolate onto requested receiver grid

    Parameters
    ----------
    filepath : str or Path
        Path to .plt or .plp file (base name works for both)
    receiver_depths : ndarray
        Requested receiver depths in meters
    receiver_ranges : ndarray
        Requested receiver ranges in meters
    interpolate : bool, optional
        If True, interpolate OAST grid onto receiver grid. Default is True.

    Returns
    -------
    tl_data : ndarray
        Transmission loss data, shape (n_depths, n_ranges)
    metadata : dict
        Metadata including:
        - 'oast_native_ranges': OAST's range grid
        - 'oast_grid_shape': Shape of OAST's native grid
        - 'model': 'OAST'

    Examples
    --------
    >>> tl, meta = read_oast_tl('test.plt', depths, ranges)
    >>> print(tl.shape)
    (40, 100)
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
        raise FileNotFoundError(f"OAST TL data file not found. Checked: {plt_file}, {f020_file}")

    # Parse .plp file to get OAST's grid
    oast_grid = None
    if plp_file.exists():
        try:
            oast_grid = _parse_oast_plp(plp_file)
        except Exception as e:
            warnings.warn(
                f"Could not parse .plp file: {e}. "
                "Assuming grid matches receiver specification.",
                UserWarning, stacklevel=2
            )

    if oast_grid is None:
        # If .plp doesn't exist or couldn't be parsed, use receiver grid
        if not plp_file.exists():
            warnings.warn(
                f".plp file not found: {plp_file}. "
                "Cannot determine OAST's native grid. "
                "Assuming grid matches receiver specification.",
                UserWarning, stacklevel=2
            )
        oast_grid = {
            'n_ranges': len(receiver_ranges),
            'ranges': receiver_ranges,
            'range_offset_km': receiver_ranges[0] / 1000.0,
            'range_increment_km': (receiver_ranges[-1] - receiver_ranges[0]) / (len(receiver_ranges) - 1) / 1000.0 if len(receiver_ranges) > 1 else 0.0
        }

    n_depths_oast = len(receiver_depths)  # OAST uses our receiver depths
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
        raise IOError(f"No TL data found in {tl_data_file}")

    # OAST outputs data as: all ranges for depth 1, all ranges for depth 2, etc.
    expected_total = n_depths_oast * n_ranges_oast

    if len(tl_values) < expected_total:
        warnings.warn(
            f"Got {len(tl_values)} TL values, expected {expected_total}. "
            "Padding with last value.",
            UserWarning, stacklevel=2
        )
        tl_values = np.pad(tl_values, (0, expected_total - len(tl_values)),
                          mode='edge')

    # Take expected amount and reshape
    tl_oast = tl_values[:expected_total].reshape(n_depths_oast, n_ranges_oast)

    # Interpolate onto requested receiver grid if requested
    # Check if grids match (same shape and values)
    ranges_match = (len(ranges_oast) == len(receiver_ranges) and
                   np.allclose(ranges_oast, receiver_ranges))

    if interpolate and not ranges_match:
        from scipy.interpolate import RegularGridInterpolator

        # Create interpolator
        interp = RegularGridInterpolator(
            (receiver_depths, ranges_oast),
            tl_oast,
            method='linear',
            bounds_error=False,
            fill_value=None
        )

        # Create grid for requested points
        mesh_d, mesh_r = np.meshgrid(receiver_depths, receiver_ranges, indexing='ij')
        points = np.array([mesh_d.ravel(), mesh_r.ravel()]).T

        # Interpolate
        tl_interp = interp(points).reshape(len(receiver_depths), len(receiver_ranges))

        metadata = {
            'model': 'OAST',
            'oast_native_ranges': ranges_oast,
            'oast_grid_shape': (n_depths_oast, n_ranges_oast),
            'interpolated': True
        }

        return tl_interp, metadata
    else:
        metadata = {
            'model': 'OAST',
            'oast_native_ranges': ranges_oast,
            'oast_grid_shape': (n_depths_oast, n_ranges_oast),
            'interpolated': False
        }

        return tl_oast, metadata


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
            raise ValueError(
                f"Could not parse grid from .plp file: "
                f"n_ranges={n_ranges}, xoff={xoff}, dx={dx}"
            )

        # Calculate range array (convert km to m)
        ranges = (xoff + np.arange(n_ranges) * dx) * 1000.0

        return {
            'n_ranges': n_ranges,
            'ranges': ranges,
            'range_offset_km': xoff,
            'range_increment_km': dx
        }

    except Exception as e:
        raise IOError(f"Failed to parse OAST .plp file: {e}")


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
        raise FileNotFoundError(f"OASN covariance file not found: {filepath}")

    # Record length in bytes (8 bytes = 1 complex64 = 2 float32)
    recl = 8

    try:
        with open(filepath, 'rb') as f:
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
            n_rcv, n_freq = struct.unpack('ii', f.read(8))

            # Record 6: IZERO, IZERO (dummy)
            # Skip

            # Record 7: FREQ1, FREQ2 (2 floats)
            f.seek(6 * recl)
            freq1, freq2 = struct.unpack('ff', f.read(8))

            # Record 8: DELFRQ, ZERO (frequency increment)
            f.seek(7 * recl)
            delfrq, _ = struct.unpack('ff', f.read(8))

            # Record 9: SSLEV, WNLEV (surface and white noise levels)
            f.seek(8 * recl)
            sslev, wnlev = struct.unpack('ff', f.read(8))

            # Record 10: ZERO, ZERO (reserved)
            # Skip

            # Read covariance matrices
            # Data starts at record 11 (offset 10 * recl)
            # Format: for each frequency, for each column, for each row
            covariance = np.zeros((n_freq, n_rcv, n_rcv), dtype=np.complex64)

            for ifreq in range(n_freq):
                for jrcv in range(n_rcv):
                    for ircv in range(n_rcv):
                        irec = 10 + ircv + jrcv * n_rcv + ifreq * n_rcv * n_rcv
                        f.seek(irec * recl)
                        real, imag = struct.unpack('ff', f.read(8))
                        covariance[ifreq, ircv, jrcv] = complex(real, imag)

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

    except Exception as e:
        raise IOError(f"Failed to read OASN covariance file {filepath}: {e}")


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
        raise FileNotFoundError(f"OASN replica file not found: {filepath}")

    try:
        with open(filepath, 'rb') as f:
            # Read header
            # Fortran unformatted files have record markers (4-byte integers before/after each record)

            # Read title (80 characters)
            _read_fortran_record_marker(f)
            title = f.read(80).decode('ascii', errors='ignore').strip()
            _read_fortran_record_marker(f)

            # Read NRCV, NFREQ
            _read_fortran_record_marker(f)
            n_rcv, n_freq = struct.unpack('ii', f.read(8))
            _read_fortran_record_marker(f)

            # Read FREQ1, FREQ2, DELFRQ
            _read_fortran_record_marker(f)
            freq1, freq2, delfrq = struct.unpack('fff', f.read(12))
            _read_fortran_record_marker(f)

            # Read replica grid: ZMINR, ZMAXR, NZR
            _read_fortran_record_marker(f)
            z_min, z_max, n_z = struct.unpack('ffi', f.read(12))
            _read_fortran_record_marker(f)

            # Read XMINR, XMAXR, NXR
            _read_fortran_record_marker(f)
            x_min, x_max, n_x = struct.unpack('ffi', f.read(12))
            _read_fortran_record_marker(f)

            # Read YMINR, YMAXR, NYR
            _read_fortran_record_marker(f)
            y_min, y_max, n_y = struct.unpack('ffi', f.read(12))
            _read_fortran_record_marker(f)

            # Read receiver positions and properties
            receiver_positions = np.zeros((n_rcv, 3))
            receiver_types = np.zeros(n_rcv, dtype=int)
            receiver_gains = np.zeros(n_rcv)

            for i in range(n_rcv):
                _read_fortran_record_marker(f)
                x, y, z, itype, gain = struct.unpack('fffif', f.read(20))
                _read_fortran_record_marker(f)
                receiver_positions[i] = [x, y, z]
                receiver_types[i] = itype
                receiver_gains[i] = gain

            # Read replicas
            replicas = np.zeros((n_freq, n_z, n_x, n_y, n_rcv), dtype=np.complex64)

            for ifreq in range(n_freq):
                for iz in range(n_z):
                    for ix in range(n_x):
                        for iy in range(n_y):
                            for ircv in range(n_rcv):
                                _read_fortran_record_marker(f)
                                real, imag = struct.unpack('ff', f.read(8))
                                _read_fortran_record_marker(f)
                                replicas[ifreq, iz, ix, iy, ircv] = complex(real, imag)

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

    except Exception as e:
        raise IOError(f"Failed to read OASN replica file {filepath}: {e}")


def _read_fortran_record_marker(f) -> int:
    """
    Read Fortran unformatted record marker (4-byte integer)

    Fortran unformatted files have record markers before and after each record
    indicating the size of the record in bytes.
    """
    marker_bytes = f.read(4)
    if len(marker_bytes) < 4:
        raise IOError("Unexpected end of file while reading record marker")
    marker = struct.unpack('i', marker_bytes)[0]
    return marker


def read_oases_modes(
    filepath: Union[str, Path]
) -> Dict:
    """
    Best-effort recovery of OASN mode-related information from auxiliary files.

    OASN does not have a documented mode-shape output format. Its primary
    outputs are:
      - ``.xsm`` covariance matrix (direct-access unformatted, 8-byte records)
        — read via :func:`read_oasn_covariance`, NOT this function.
      - ``.rpo`` replica field — read via :func:`read_oasn_replicas`.

    Some deployments happen to write a TRF-style Fortran-SEQUENTIAL file
    alongside the .xsm (typically via a custom build); this helper tries a
    sequential PULSETRF parse first as a best-effort probe. It will fail
    cleanly on a real ``.xsm`` file (which is direct-access) and fall through
    to the log-scan strategy.

    Strategies, in order:

    1. Try :func:`_read_oasp_trf_binary` — only succeeds for Fortran-sequential
       PULSETRF-header files. ``.xsm`` files will fail at the record-marker
       check; that's expected.
    2. Scan companion ``.prt`` / ``.log`` files for WAVENUMBER / EIGENVALUE
       lines and extract real eigenvalues.
    3. If neither strategy yields data, warn and return empty arrays.

    Parameters
    ----------
    filepath : str or Path
        Path to mode file (``.xsm``, ``.mod``, or ``fort.16``).

    Returns
    -------
    modes : dict
        - ``k``: ndarray, mode wavenumbers (rad/m). Real eigenvalues when
          available from the log; empty otherwise.
        - ``phi``: ndarray, mode shapes (2D: n_z x n_modes). Often empty —
          OASN does not write explicit shapes to a documented format.
        - ``z``: ndarray, depth grid for ``phi`` (or empty).
        - ``n_modes``: int.
        - ``model``: 'OASN'.

    Notes
    -----
    OASES mode files remain under-documented in the public OASES distribution;
    the authoritative output is the covariance file (``read_oasn_covariance``).
    For reliable modal analysis, prefer Kraken/KrakenC.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"OASES mode file not found: {filepath}")

    k_list = []
    phi = np.array([])
    z = np.array([])

    # --- Strategy 1: try TRF-style Fortran unformatted ---
    try:
        trf = _read_oasp_trf_binary(filepath)
        # Transfer-function magnitude can be treated as a rough modal envelope
        # when OASN writes a .trf-style file. Mode wavenumbers are not directly
        # recoverable from that payload, so we only populate phi/z/depths.
        tf = trf.get('transfer_function')
        if tf is not None and tf.size > 0:
            phi = np.abs(tf[0, :, :]).T  # (n_depth, n_range) as pseudo-shapes
            z = np.asarray(trf.get('depths', np.array([])))
        # Fall through to also try log parsing for wavenumbers.
    except Exception:
        pass

    # --- Strategy 2: scan companion .prt / log file ---
    candidates = [
        filepath.with_suffix('.prt'),
        filepath.with_suffix('.log'),
        filepath.with_name(filepath.stem + '.prt'),
    ]
    for log_path in candidates:
        if log_path.exists():
            try:
                with open(log_path, 'r', errors='ignore') as lf:
                    for line in lf:
                        up = line.upper()
                        if 'WAVENUMBER' in up or 'EIGENVALUE' in up:
                            # Extract floats on the line
                            parts = line.replace(',', ' ').split()
                            for tok in parts:
                                try:
                                    v = float(tok)
                                    if 1e-4 < abs(v) < 1e3:
                                        k_list.append(v)
                                except ValueError:
                                    continue
                break
            except Exception:
                continue

    if not k_list and phi.size == 0:
        warnings.warn(
            f"OASN mode reader found no usable data in {filepath} "
            "(format remains under-documented; only wavenumber scan or "
            "TRF-style payloads are supported). Prefer Kraken for modes.",
            UserWarning, stacklevel=2,
        )

    k_arr = np.array(k_list, dtype=np.float64) if k_list else np.array([])

    return {
        'k': k_arr,
        'phi': phi,
        'z': z,
        'n_modes': int(max(len(k_arr), phi.shape[1] if phi.ndim == 2 else 0)),
        'model': 'OASN',
    }


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
        raise FileNotFoundError(f"OASP transfer function file not found: {filepath}")

    # Try Fortran-unformatted binary first (current OASES default).
    errors = []
    try:
        return _read_oasp_trf_binary(filepath)
    except Exception as e:
        errors.append(('fortran-unformatted', e))

    # ASCII path always raises NotImplemented, but wrap so the binary
    # error surfaces when both paths fail.
    try:
        return _read_oasp_trf_ascii(filepath)
    except Exception as e:
        errors.append(('ascii', e))

    err_msg = '\n'.join(f"  {k}: {v}" for k, v in errors)
    raise IOError(
        f"Failed to read OASP transfer function file {filepath}.\n{err_msg}"
    )


def _read_fortran_record(f, fmt=None, raw=False, endian='<'):
    """Read a single Fortran UNFORMATTED sequential record.

    Layout::

        [4-byte length N][N bytes payload][4-byte length N]

    Both length markers must match; mismatch indicates file corruption or
    wrong endianness and raises ``IOError``.

    Parameters
    ----------
    f : file object (binary mode)
    fmt : str, optional
        struct format string for the payload (excluding endian prefix).
    raw : bool, optional
        If True, return raw bytes. Default False.
    endian : str, optional
        '<' (little-endian, x86 default) or '>' (big-endian).

    Returns
    -------
    tuple | bytes
        Unpacked payload (or raw bytes).
    """
    head = f.read(4)
    if len(head) < 4:
        raise IOError("Unexpected EOF reading Fortran record head")
    (nbytes,) = struct.unpack(endian + 'i', head)
    if nbytes < 0 or nbytes > (1 << 28):  # magic-number sanity: <256 MB
        raise IOError(
            f"Unreasonable Fortran record length: {nbytes} (wrong endianness?)"
        )
    payload = f.read(nbytes)
    if len(payload) < nbytes:
        raise IOError(
            f"Short read: expected {nbytes} bytes, got {len(payload)}"
        )
    tail = f.read(4)
    if len(tail) < 4:
        raise IOError("Unexpected EOF reading Fortran record tail")
    (ntail,) = struct.unpack(endian + 'i', tail)
    if ntail != nbytes:
        raise IOError(
            f"Fortran record marker mismatch: head={nbytes} tail={ntail} "
            "(wrong endianness or truncated file)"
        )
    if raw or fmt is None:
        return payload
    expected = struct.calcsize(endian + fmt)
    if expected != nbytes:
        raise IOError(
            f"Fortran record payload {nbytes} != fmt '{fmt}' size {expected}"
        )
    return struct.unpack(endian + fmt, payload)


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
    endian = '<'
    with open(filepath, 'rb') as f:
        try:
            fileid_raw = _read_fortran_record(f, raw=True, endian=endian)
        except IOError as e:
            raise IOError(
                f"Cannot open {filepath} as Fortran-unformatted TRF: {e}"
            ) from e
        fileid = fileid_raw.decode('ascii', errors='ignore').strip()
        if 'PULSETRF' not in fileid:
            raise IOError(
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
        ranges = r0 + np.arange(nplots) * rspace

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
        freq_array = np.array(
            [(k / (dt * nx)) for k in range(lx, mx + 1)], dtype=np.float64
        ) if nf >= 1 else np.array([freqs], dtype=np.float64)

        transfer_function = np.zeros((nf, nplots, nrd), dtype=np.complex64)
        for j in range(nf):
            for _is in range(max(1, isrow)):
                for _m in range(max(1, msuft)):
                    for jrh in range(nplots):
                        for jrv in range(nrd):
                            rec = _read_fortran_record(
                                f, f'{2 * nout}f', endian=endian)
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
        'model': 'OASP',
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
    RuntimeError
        Always. Either re-run OASES with binary TRF (default) or extend this
        reader to parse the ASCII payload.
    """
    raise RuntimeError(
        f"ASCII TRF reader not implemented for {filepath}. "
        "OASES must be run with binary (Fortran-unformatted) TRF output — "
        "that is the default; do not pass any ASCII conversion option. "
        "If you have a mixed-format legacy file, use the binary reader."
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
        raise FileNotFoundError(f"OASR reflection coefficient file not found: {filepath}")

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
                raise ValueError(f"Invalid header format: {header_line}")

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
            'model': 'OASR'
        }

    except Exception as e:
        raise IOError(f"Failed to read OASR reflection coefficient file {filepath}: {e}")


def _trf_regression_selftest(tmp_path=None):
    """Write a synthetic PULSETRF-style file and round-trip through the reader.

    Produces a minimal single-frequency, single-depth, single-range file with
    known complex payload and validates the reader returns matching values.
    Used by the OASES reader test-suite and available as a quick sanity check::

        from uacpy.io.oases_reader import _trf_regression_selftest
        _trf_regression_selftest()
    """
    import tempfile
    endian = '<'

    def w(f, payload: bytes):
        f.write(struct.pack(endian + 'i', len(payload)))
        f.write(payload)
        f.write(struct.pack(endian + 'i', len(payload)))

    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp()) / 'synthetic.trf'
    else:
        tmp_path = Path(tmp_path)

    # --- Build synthetic header matching the documented layout ---
    title = 'synthetic-trf-test'
    prognm = 'OASP17'
    nout = 1
    iparm = [1]
    signn = '+'
    freqs = 100.0
    sd = 50.0
    rd, rdlow, ir = 20.0, 20.0, 1   # single receiver depth
    r0, rspace, nplots = 1000.0, 1000.0, 1  # single range
    nx, lx, mx, dt = 4, 1, 1, 0.01       # NX=power of 2, single freq bin
    icdr = 0
    omegim = 0.0
    msuft = 1
    isrow = 1
    inttyp = 0

    test_real, test_imag = 0.75, -0.25

    with open(tmp_path, 'wb') as f:
        w(f, b'PULSETRF')
        w(f, prognm.encode())
        w(f, struct.pack(endian + 'i', nout))
        w(f, struct.pack(endian + f'{nout}i', *iparm))
        w(f, title.ljust(80).encode())
        w(f, signn.encode())
        w(f, struct.pack(endian + 'f', freqs))
        w(f, struct.pack(endian + 'f', sd))
        w(f, struct.pack(endian + 'ffi', rd, rdlow, ir))
        w(f, struct.pack(endian + 'ffi', r0, rspace, nplots))
        w(f, struct.pack(endian + 'iiif', nx, lx, mx, dt))
        w(f, struct.pack(endian + 'i', icdr))
        w(f, struct.pack(endian + 'f', omegim))
        w(f, struct.pack(endian + 'i', msuft))
        w(f, struct.pack(endian + 'i', isrow))
        w(f, struct.pack(endian + 'i', inttyp))
        for _ in range(2):
            w(f, struct.pack(endian + 'i', 0))
        for _ in range(5):
            w(f, struct.pack(endian + 'f', 0.0))
        # Data records: 1 freq * 1 isrow * 1 msuft * 1 nplots * 1 ir
        w(f, struct.pack(endian + 'ff', test_real, test_imag))

    data = _read_oasp_trf_binary(tmp_path)
    tf = data['transfer_function']
    assert tf.shape == (1, 1, 1), f"shape mismatch: {tf.shape}"
    assert np.isclose(tf[0, 0, 0].real, test_real), tf[0, 0, 0]
    assert np.isclose(tf[0, 0, 0].imag, test_imag), tf[0, 0, 0]
    assert np.isclose(data['source_depth'], sd)
    assert np.isclose(data['center_frequency'], freqs)
    return True


if __name__ == '__main__':
    _trf_regression_selftest()
    print('TRF regression self-test: OK')
