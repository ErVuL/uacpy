"""
Low-level Fortran-record helpers shared by the AT/Bellhop output readers.

These functions read fragments of the binary ``.shd``/``.mod`` formats —
record-length-prefixed vectors, source/receiver depth blocks, bearing/angle
arrays. They are private to ``uacpy.io`` and not part of the public surface.
"""

import struct
from typing import Tuple

import numpy as np


def read_fortran_record_marker(f) -> int:
    """Read a 4-byte Fortran unformatted record-length marker (little-endian).

    Used by Fortran sequential-unformatted record framing
    ``[len][payload][len]``.
    """
    marker_bytes = f.read(4)
    if len(marker_bytes) < 4:
        raise IOError("Unexpected end of file while reading record marker")
    return struct.unpack('i', marker_bytes)[0]


def read_fortran_record(f, fmt=None, raw=False, endian='<'):
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
    if nbytes < 0 or nbytes > (1 << 28):
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


def read_vector(fid) -> Tuple[np.ndarray, int]:
    """
    Read a vector from BELLHOP environment file with Fortran-style input.

    This routine emulates Fortran capability that allows '/' to terminate input
    and create equally-spaced vectors. Supports three input formats:

    1. Explicit values: N / v1 v2 v3 ... vN
    2. Linear spacing: N / v_start v_end / (creates N points between start and end)
    3. Replicate value: N / v / (creates N copies of v)

    Parameters
    ----------
    fid : file object
        Open file handle positioned at vector specification

    Returns
    -------
    x : ndarray
        Vector of values
    Nx : int
        Number of values

    Notes
    -----
    Examples of input formats:

    Format 1 (linear spacing):
        5
        0 1000 /
    Creates: [0, 250, 500, 750, 1000]

    Format 2 (explicit values):
        5
        0 100 300 700 1000
    Creates: [0, 100, 300, 700, 1000]

    Format 3 (replicate):
        501
        0.0 /
    Creates: [0, 0, ..., 0] (501 zeros)

    The '/' character terminates reading and triggers vector generation.

    Translated from OALIB readvector.m

    Examples
    --------
    >>> # Create test file
    >>> with open('test_vec.txt', 'w') as f:
    ...     f.write('5\\n0 1000 /\\n')
    >>> with open('test_vec.txt', 'r') as f:
    ...     x, Nx = read_vector(f)
    >>> print(x)
    [   0.  250.  500.  750. 1000.]
    """
    # Read number of values
    line = fid.readline()
    Nx = int(line.strip())

    # Read values line
    line = fid.readline()

    if "/" in line:
        # Extract numbers before '/'
        nums_str = line.split("/")[0].strip()
        if nums_str:
            values = np.fromstring(nums_str, sep=" ")
        else:
            values = np.array([])

        if Nx == 1:
            x = values[0] if len(values) > 0 else 0.0
        elif Nx == 2:
            x = values[:2] if len(values) >= 2 else values
        elif Nx > 2:
            if len(values) > 1:
                # Generate linearly spaced vector
                x = np.linspace(values[0], values[1], Nx)
            elif len(values) == 1:
                # Replicate single value
                x = np.full(Nx, values[0])
            else:
                # No values provided, return zeros
                x = np.zeros(Nx)
        else:
            x = np.array([])
    else:
        # Read explicit values
        x = np.fromstring(line, sep=" ", count=Nx)

    # Ensure x is a 1D array
    x = np.atleast_1d(x)

    return x, Nx
