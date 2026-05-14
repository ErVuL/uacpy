"""
Low-level Fortran-record helpers shared by the AT/Bellhop output readers.

These functions read fragments of the binary ``.shd``/``.mod`` formats —
record-length-prefixed vectors, source/receiver depth blocks, bearing/angle
arrays. They are private to ``uacpy.io`` and not part of the public surface.
"""

import struct
import warnings
from typing import Tuple

import numpy as np


_ENDIAN_WARN_EMITTED = False


def _warn_non_little_endian(detected: str, source: str) -> None:
    """Emit a one-shot warning the first time we decode a non-little-endian
    Fortran file. uacpy CI runs little-endian; big-endian decode works but
    is unvalidated."""
    global _ENDIAN_WARN_EMITTED
    if detected == 'big' and not _ENDIAN_WARN_EMITTED:
        warnings.warn(
            f"{source}: detected big-endian Fortran record framing; "
            "uacpy decodes it correctly but this byte order is not "
            "validated by CI.",
            UserWarning, stacklevel=3,
        )
        _ENDIAN_WARN_EMITTED = True


def detect_endian(first4: bytes, source: str = '_fortran_helpers') -> str:
    """Detect Fortran-record byte order from the first 4 bytes of a file.

    The Fortran framing puts a 4-byte record-length prefix at the head of
    every record. On a well-formed file that integer is much smaller than
    ``2**31`` in the correct endianness and absurdly large in the wrong
    one. We pick the byte order that yields the smaller positive integer
    (with a sanity cap of ``2**28``) and warn once if it isn't little-endian.

    Returns ``'<'`` (little-endian) or ``'>'`` (big-endian).
    """
    if len(first4) < 4:
        raise OSError("detect_endian: need 4 bytes to probe.")
    little = struct.unpack('<i', first4)[0]
    big = struct.unpack('>i', first4)[0]
    cap = 1 << 28
    little_ok = 0 < little < cap
    big_ok = 0 < big < cap
    if little_ok and not big_ok:
        chosen = '<'
    elif big_ok and not little_ok:
        chosen = '>'
    elif little_ok and big_ok:
        chosen = '<' if little <= big else '>'
    else:
        raise OSError(
            f"detect_endian: cannot resolve byte order from first record "
            f"marker (little={little}, big={big}); file is probably corrupt."
        )
    _warn_non_little_endian('big' if chosen == '>' else 'little', source)
    return chosen


def read_fortran_record_marker(f, endian: str = '<') -> int:
    """Read a 4-byte Fortran unformatted record-length marker.

    Used by Fortran sequential-unformatted record framing
    ``[len][payload][len]``.

    Parameters
    ----------
    f : file object (binary mode)
    endian : str, optional
        '<' (little-endian, default) or '>' (big-endian). Pass the
        value returned by :func:`detect_endian` after a one-time probe
        of the file head.
    """
    marker_bytes = f.read(4)
    if len(marker_bytes) < 4:
        raise OSError("Unexpected end of file while reading record marker")
    return struct.unpack(endian + 'i', marker_bytes)[0]


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
        raise OSError("Unexpected EOF reading Fortran record head")
    (nbytes,) = struct.unpack(endian + 'i', head)
    if nbytes < 0 or nbytes > (1 << 28):
        raise OSError(
            f"Unreasonable Fortran record length: {nbytes} (wrong endianness?)"
        )
    payload = f.read(nbytes)
    if len(payload) < nbytes:
        raise OSError(
            f"Short read: expected {nbytes} bytes, got {len(payload)}"
        )
    tail = f.read(4)
    if len(tail) < 4:
        raise OSError("Unexpected EOF reading Fortran record tail")
    (ntail,) = struct.unpack(endian + 'i', tail)
    if ntail != nbytes:
        raise OSError(
            f"Fortran record marker mismatch: head={nbytes} tail={ntail} "
            "(wrong endianness or truncated file)"
        )
    if raw or fmt is None:
        return payload
    expected = struct.calcsize(endian + fmt)
    if expected != nbytes:
        raise OSError(
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
            if len(values) < 1:
                warnings.warn(
                    "read_vector: Nx=1 record had no values; using 0.0.",
                    UserWarning, stacklevel=2,
                )
                x = 0.0
            else:
                x = values[0]
        elif Nx == 2:
            if len(values) >= 2:
                x = values[:2]
            elif len(values) == 1:
                warnings.warn(
                    f"read_vector: Nx=2 record had 1 value; broadcasting "
                    f"{values[0]} to both slots.",
                    UserWarning, stacklevel=2,
                )
                x = np.array([values[0], values[0]])
            else:
                warnings.warn(
                    "read_vector: Nx=2 record had no values; using zeros.",
                    UserWarning, stacklevel=2,
                )
                x = np.zeros(2)
        elif Nx > 2:
            # AT semantics: either exactly 2 values (linspace shorthand)
            # or exactly Nx values (explicit list). Other lengths fall back
            # with a warning rather than silently producing wrong numbers.
            if len(values) == Nx:
                x = values
            elif len(values) == 2:
                x = np.linspace(values[0], values[1], Nx)
            elif len(values) > 1:
                warnings.warn(
                    f"read_vector: Nx={Nx} record had {len(values)} values "
                    f"(expected 2 or {Nx}); falling back to linspace "
                    f"between {values[0]} and {values[-1]}.",
                    UserWarning, stacklevel=2,
                )
                x = np.linspace(values[0], values[-1], Nx)
            elif len(values) == 1:
                warnings.warn(
                    f"read_vector: Nx={Nx} record had 1 value; broadcasting "
                    f"{values[0]} to all slots.",
                    UserWarning, stacklevel=2,
                )
                x = np.full(Nx, values[0])
            else:
                warnings.warn(
                    f"read_vector: Nx={Nx} record had no values; using zeros.",
                    UserWarning, stacklevel=2,
                )
                x = np.zeros(Nx)
        else:
            x = np.array([])
    else:
        # Read explicit values
        x = np.fromstring(line, sep=" ", count=Nx)

    # Ensure x is a 1D array
    x = np.atleast_1d(x)

    return x, Nx
