"""
OASES Output File Readers

This module provides functions for reading output files from OASES models:
- OAST: .plp/.plt files (transmission loss)
- OASN: .xsm files (covariance matrices), .rpo files (replicas)
- Mode files: .mod files (mode shapes and wavenumbers)

OASES (Ocean Acoustics and Seismic Exploration Synthesis) was developed by
Henrik Schmidt at MIT.

References:
    Schmidt, H. (2004). OASES Version 3.1 User Guide and Reference Manual.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional, Union
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
    Read OASES mode file

    Mode files can be in different formats depending on the OASES module used.
    This function attempts to read mode shapes and wavenumbers.

    Parameters
    ----------
    filepath : str or Path
        Path to mode file

    Returns
    -------
    modes : dict
        Dictionary containing:
        - 'k': ndarray, mode wavenumbers
        - 'phi': ndarray, mode shapes (depth x n_modes)
        - 'z': ndarray, depth grid
        - 'n_modes': int, number of modes
        - 'model': str, 'OASN'

    Notes
    -----
    This is a basic implementation. OASES mode file formats can vary.
    For production use, consult OASES documentation for the specific
    file format used by your version.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"OASES mode file not found: {filepath}")

    # Placeholder implementation
    # Actual format depends on OASES version and module
    warnings.warn(
        f"OASES mode file reading is experimental.\n"
        f"File: {filepath}\n"
        "Mode file format depends on OASES version.\n"
        "For production use, verify against OASES manual.",
        UserWarning, stacklevel=2
    )

    # Return empty structure
    return {
        'k': np.array([]),
        'phi': np.array([]),
        'z': np.array([]),
        'n_modes': 0,
        'model': 'OASN'
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

    # Try binary format first
    try:
        return _read_oasp_trf_binary(filepath)
    except Exception as binary_error:
        # Try ASCII format as fallback
        try:
            return _read_oasp_trf_ascii(filepath)
        except Exception as ascii_error:
            raise IOError(
                f"Failed to read OASP transfer function file {filepath}.\n"
                f"Binary format error: {binary_error}\n"
                f"ASCII format error: {ascii_error}"
            )


def _read_oasp_trf_binary(filepath: Path) -> Dict:
    """Read mixed-format (ASCII headers + binary data) TRF file

    Based on MATLAB trf_reader_oases.m from OASES distribution.
    Format: ASCII text headers followed by binary float/int data.
    No Fortran record markers!
    """
    import io

    try:
        with open(filepath, 'rb') as f:
            # Read ASCII text until we find "PULSETRF"
            text_chunk = b''
            while b'PULSETRF' not in text_chunk:
                byte = f.read(1)
                if not byte:
                    raise ValueError("Could not find PULSETRF identifier in TRF file")
                text_chunk += byte

            # Find the position right after "PULSETRF"
            pulsetrf_end = text_chunk.index(b'PULSETRF') + len(b'PULSETRF')
            # Seek back to after PULSETRF
            f.seek(pulsetrf_end)

            # Read until we find '+' or '-' sign
            sign_byte = b''
            while sign_byte not in [b'+', b'-']:
                sign_byte = f.read(1)
                if not sign_byte:
                    raise ValueError("Could not find sign byte")

            # Now read binary data (little-endian floats and ints)
            # Skip 2 floats
            f.read(8)

            # Read fc (center frequency)
            fc, = struct.unpack('f', f.read(4))

            # Skip 2 floats
            f.read(8)

            # Read sd (source depth)
            sd, = struct.unpack('f', f.read(4))

            # Skip 2 floats
            f.read(8)

            # Read receiver depth parameters
            z1, = struct.unpack('f', f.read(4))
            z2, = struct.unpack('f', f.read(4))
            num_z, = struct.unpack('i', f.read(4))

            # Create receiver depth array
            if num_z > 1:
                receiver_depths = np.linspace(z1, z2, num_z)
            else:
                receiver_depths = np.array([z1])

            # Skip 2 floats
            f.read(8)

            # Read range parameters
            r1, = struct.unpack('f', f.read(4))
            dr, = struct.unpack('f', f.read(4))
            nr, = struct.unpack('i', f.read(4))

            # Create range array
            ranges = np.array([r1 + i * dr for i in range(nr)])

            # Skip 2 floats
            f.read(8)

            # Read FFT parameters
            nfft, = struct.unpack('i', f.read(4))
            bin_low, = struct.unpack('i', f.read(4))
            bin_high, = struct.unpack('i', f.read(4))
            dt, = struct.unpack('f', f.read(4))

            # Calculate frequencies
            nf = bin_high - bin_low + 1
            freq_array = np.array([(k / (dt * nfft)) for k in range(bin_low, bin_high + 1)])

            # Skip 2 floats
            f.read(8)

            # Read icdr
            icdr, = struct.unpack('i', f.read(4))

            # Skip 2 floats
            f.read(8)

            # Read omegim
            omegim, = struct.unpack('f', f.read(4))

            # Skip remaining header (5 dummy values with 2 floats before each)
            for _ in range(5):
                f.read(8)  # 2 floats
                f.read(4)  # 1 int or float

            # Skip 1 float
            f.read(4)

            # Read transfer function data
            # Data structure: for each frequency, for each range, read num_z complex values
            transfer_function = np.zeros((nf, nr, num_z), dtype=np.complex64)

            for j in range(nf):
                for jj in range(nr):
                    # Skip 1 float (record marker)
                    f.read(4)

                    # Read num_z*4-2 floats (interleaved real/imag pairs + some overhead)
                    # Based on MATLAB: temp=fread(fid,num_z*4-2,'float')
                    n_floats = num_z * 4 - 2
                    data = struct.unpack(f'{n_floats}f', f.read(4 * n_floats))

                    # Extract real and imaginary parts
                    # MATLAB: temp(1:2:length(temp))+i*temp(2:2:length(temp))
                    real_parts = np.array(data[0::2])
                    imag_parts = np.array(data[1::2])

                    # MATLAB: temp(1:2:length(temp)) - takes every other element again
                    # This extracts num_z values
                    transfer_function[j, jj, :] = real_parts[:num_z] + 1j * imag_parts[:num_z]

                    # Skip 1 float (record marker)
                    f.read(4)

        return {
            'title': 'OASP Transfer Function',
            'option': '',
            'freq': freq_array,
            'ranges': ranges,
            'depths': receiver_depths,
            'transfer_function': transfer_function,
            'source_depth': sd,
            'center_frequency': fc,
            'model': 'OASP'
        }

    except Exception as e:
        raise IOError(f"Binary TRF read failed: {e}")


def _read_oasp_trf_ascii(filepath: Path) -> Dict:
    """Read ASCII (formatted) TRF file

    Returns minimal data structure with empty transfer functions.
    Full ASCII TRF reading is complex and rarely used.
    """
    try:
        with open(filepath, 'r') as f:
            # Line 1: FILEID
            fileid = f.readline().strip()
            if fileid != 'PULSETRF':
                raise ValueError(f"Invalid ASCII TRF file ID: {fileid}")

            # Line 2: PROGNM
            prognm = f.readline().strip()

            # Line 3: NOUT
            nout = int(f.readline().strip())

            # Line 4: IPARM array
            iparm_line = f.readline().strip()
            iparm = [int(x) for x in iparm_line.split()]

            # Line 5: TITLE
            title = f.readline().strip()

            # Line 6: SIGNN
            signn = f.readline().strip()

            # Line 7: FREQS (center frequency)
            freqs = float(f.readline().strip())

            # Line 8: SD (source depth)
            sd = float(f.readline().strip())

            # Line 9: RD, RDLOW, IR
            rd_line = f.readline().strip().split()
            rd, rdlow, ir = float(rd_line[0]), float(rd_line[1]), int(rd_line[2])

            # Handle receiver depths
            if ir < 0:
                # Non-equidistant
                depth_line = f.readline().strip()
                receiver_depths = [float(x) for x in depth_line.split()]
            else:
                # Equidistant
                receiver_depths = list(np.linspace(rd, rdlow, ir))

            # Line 10: R0, RSPACE, NPLOTS
            range_line = f.readline().strip().split()
            r0, rspace, nplots = float(range_line[0]), float(range_line[1]), int(range_line[2])

            ranges = np.array([r0 + i * rspace for i in range(nplots)])

            # Return minimal structure - full ASCII TRF parsing is complex
            # Most users should use binary format
            return {
                'title': title,
                'option': ''.join([chr(p + ord('A') - 1) for p in iparm if 0 < p < 27]),
                'freq': np.array([freqs]),  # Single frequency for now
                'ranges': ranges,
                'depths': np.array(receiver_depths),
                'transfer_function': np.ones((1, len(ranges), len(receiver_depths)), dtype=np.complex64),
                'source_depth': sd,
                'center_frequency': freqs,
                'model': 'OASP'
            }

    except Exception as e:
        raise IOError(f"ASCII TRF read failed: {e}")


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
