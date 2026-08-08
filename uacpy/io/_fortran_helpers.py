"""
Low-level Fortran helpers shared by the AT/Bellhop readers.

Two families, both private to ``uacpy.io`` and outside the public surface:

* binary record framing for the ``.shd``/``.mod`` formats — endianness
  detection and record-length-prefixed payload reads;
* list-directed text parsing for the ``.env``/``.flp``/``.bty`` decks —
  :func:`read_vector` and :func:`strip_fortran_comment`.

:func:`typed_format_error` and :data:`PARSE_ERRORS` give both families one
typed failure mode.
"""

import functools
import struct
import warnings
from typing import Tuple

import numpy as np

from uacpy.core.exceptions import FileFormatError


#: Exceptions a malformed or truncated file can legitimately raise out of a
#: reader's ``int()``/``float()``/``struct``/indexing operations. Deliberately
#: excludes ``AttributeError``/``TypeError``/``NameError``: those signal a uacpy
#: defect, and rewrapping them as :class:`FileFormatError` sends users to debug
#: a file that is fine.
PARSE_ERRORS = (ValueError, IndexError, KeyError, StopIteration, EOFError,
                ZeroDivisionError, OverflowError, struct.error)


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


def typed_format_error(reader):
    """Decorator: surface a reader's raw parse/truncation exceptions as a typed
    :class:`~uacpy.core.exceptions.FileFormatError`.

    The file-format readers do ``int()``/``float()`` on tokens, ``next()`` on
    line iterators, and index binary records — so a malformed or truncated file
    leaks a bare ``ValueError``/``IndexError``/``StopIteration``/``struct.error``
    instead of the typed, remediated error the rest of ``io`` raises. This
    converts :data:`PARSE_ERRORS` (and only those) while letting an
    already-typed ``FileFormatError``/``ConfigurationError`` pass through
    unchanged.
    """
    @functools.wraps(reader)
    def wrapper(*args, **kwargs):
        try:
            return reader(*args, **kwargs)
        except PARSE_ERRORS as exc:
            target = args[0] if args else '<stream>'
            raise FileFormatError(
                f"{reader.__name__}: could not parse {target} — the file is "
                f"malformed, truncated, or not the expected format "
                f"({type(exc).__name__}: {exc}).",
                remediation="Verify the file was produced by the matching "
                            "model/writer and downloaded completely; a partial "
                            "file or a wrong format triggers this.",
            ) from exc
    return wrapper


def strip_fortran_comment(line: str) -> str:
    """Drop a Fortran trailing comment (``! …``) and surrounding whitespace.

    AT's list-directed ``READ`` stops at the first ``!``, so its own example
    files and uacpy's writers annotate every scalar line (``999999   ! Mlimit``).
    Readers must do the same before ``int()``/``float()``.
    """
    return line.split('!', 1)[0].strip()


def strip_fortran_quotes(line: str) -> str:
    """Return the contents of a Fortran quoted string, or the bare line.

    AT writes titles and option words as ``'…'`` character literals, but its
    list-directed ``READ`` also accepts an unquoted token — so readers must
    handle both. Falls back to :func:`strip_fortran_comment` when the line
    carries no quotes.
    """
    line = line.strip()
    start = line.find("'")
    if start >= 0:
        end = line.find("'", start + 1)
        if end > start:
            return line[start + 1:end]
    return strip_fortran_comment(line)


_ENDIAN_WARN_EMITTED = False


def _warn_non_little_endian(detected: str, source: str) -> None:
    """Log a one-shot notice the first time we decode a non-little-endian
    Fortran file. uacpy CI runs little-endian; big-endian decode works but
    is unvalidated."""
    global _ENDIAN_WARN_EMITTED
    if detected == 'big' and not _ENDIAN_WARN_EMITTED:
        warnings.warn(
            f"{source}: detected big-endian Fortran record framing; uacpy "
            "decodes it correctly but this byte order is not validated by CI.",
            UserWarning, stacklevel=2,
        )
        _ENDIAN_WARN_EMITTED = True


def detect_endian(first4: bytes, source: str = '_fortran_helpers') -> str:
    """Detect Fortran-record byte order from the first 4 bytes of a file.

    The Fortran framing puts a 4-byte record-length prefix at the head of
    every record. On a well-formed file that integer is much smaller than
    ``2**31`` in the correct endianness and absurdly large in the wrong
    one. We pick the byte order that yields the smaller positive integer
    (with a sanity cap of ``2**28``) and log a one-shot notice if it isn't
    little-endian.

    Returns ``'<'`` (little-endian) or ``'>'`` (big-endian).
    """
    if len(first4) < 4:
        raise FileFormatError("detect_endian: need 4 bytes to probe.")
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
        raise FileFormatError(
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
        raise FileFormatError("Unexpected end of file while reading record marker")
    return struct.unpack(endian + 'i', marker_bytes)[0]


def read_fortran_record(f, fmt=None, raw=False, endian='<'):
    """Read a single Fortran UNFORMATTED sequential record.

    Layout::

        [4-byte length N][N bytes payload][4-byte length N]

    Both length markers must match; mismatch indicates file corruption or
    wrong endianness and raises :class:`~uacpy.core.exceptions.FileFormatError`.

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
        raise FileFormatError("Unexpected EOF reading Fortran record head")
    (nbytes,) = struct.unpack(endian + 'i', head)
    if nbytes < 0 or nbytes > (1 << 28):
        raise FileFormatError(
            f"Unreasonable Fortran record length: {nbytes} (wrong endianness?)"
        )
    payload = f.read(nbytes)
    if len(payload) < nbytes:
        raise FileFormatError(
            f"Short read: expected {nbytes} bytes, got {len(payload)}"
        )
    tail = f.read(4)
    if len(tail) < 4:
        raise FileFormatError("Unexpected EOF reading Fortran record tail")
    (ntail,) = struct.unpack(endian + 'i', tail)
    if ntail != nbytes:
        raise FileFormatError(
            f"Fortran record marker mismatch: head={nbytes} tail={ntail} "
            "(wrong endianness or truncated file)"
        )
    if raw or fmt is None:
        return payload
    expected = struct.calcsize(endian + fmt)
    if expected != nbytes:
        raise FileFormatError(
            f"Fortran record payload {nbytes} != fmt '{fmt}' size {expected}"
        )
    return struct.unpack(endian + fmt, payload)


def _read_vector_values(fid, Nx: int) -> Tuple[np.ndarray, bool]:
    """Collect the value tokens of one AT ``ReadVector`` record.

    ``READ( ENVFile, * ) x( 1 : Nx )`` is list-directed: it keeps consuming
    records until ``Nx`` values have been read, or until a ``/`` terminates the
    read early. Returns the values collected and whether a ``/`` ended them.
    """
    values = []
    while len(values) < Nx:
        line = fid.readline()
        if line == '':
            break
        line = strip_fortran_comment(line)
        if '/' in line:
            head = line.split('/', 1)[0]
            values.extend(float(tok) for tok in head.split())
            return np.array(values[:Nx], dtype=float), True
        values.extend(float(tok) for tok in line.split())
    return np.array(values[:Nx], dtype=float), False


def read_vector(fid) -> Tuple[np.ndarray, int]:
    """
    Read a vector from an Acoustics Toolbox / Bellhop input file.

    Mirrors AT's ``ReadVector`` (``misc/SourceReceiverPositions.f90:221``)
    followed by ``SubTab`` (``misc/subtabulate.f90``): a list-directed
    ``READ`` of ``Nx`` values that continues across records until ``Nx`` are
    consumed, with ``/`` terminating early to trigger a generated vector.

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
    Format 1 (linear spacing) — two values then ``/``::

        5
        0 1000 /

    Creates: [0, 250, 500, 750, 1000]

    Format 2 (explicit values) — ``Nx`` values, on one record or wrapped
    across several::

        5
        0 100 300
        700 1000

    Creates: [0, 100, 300, 700, 1000]

    Format 3 (replicate) — one value then ``/``::

        501
        0.0 /

    Creates: [0, 0, ..., 0] (501 zeros). ``SubTab`` reaches this through
    ``x(2) == x(1)``, so the generated spacing is zero.

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
    Nx = int(strip_fortran_comment(fid.readline()))
    if Nx <= 0:
        return np.array([]), Nx

    values, slash_terminated = _read_vector_values(fid, Nx)

    if len(values) == Nx:
        x = values
    elif not slash_terminated:
        # No '/' and the file ran out before Nx values: the record is truncated.
        raise FileFormatError(
            f"read_vector: expected {Nx} values but the file ended after "
            f"{len(values)}.",
            remediation="Verify the file was written completely by the "
                        "matching model/writer.",
        )
    elif Nx == 1:
        warnings.warn(
            "read_vector: Nx=1 record had no values; using 0.0.",
            UserWarning, stacklevel=2,
        )
        x = 0.0
    elif Nx == 2:
        # SubTab only generates for Nx >= 3, so a '/' before both values is a
        # malformed record — AT leaves x(2) at ReadVector's -999.9 pre-fill.
        if len(values) == 1:
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
    elif len(values) == 2:
        # SubTab's two-value branch: equally spaced from x(1) to x(2).
        x = np.linspace(values[0], values[1], Nx)
    elif len(values) == 1:
        # SubTab's replicate branch (x(2) defaults to x(1) ⇒ zero spacing).
        x = np.full(Nx, values[0])
    elif len(values) == 0:
        warnings.warn(
            f"read_vector: Nx={Nx} record had no values; using zeros.",
            UserWarning, stacklevel=2,
        )
        x = np.zeros(Nx)
    else:
        # More than 2 values but fewer than Nx before the '/'. SubTab does not
        # generate for this case and AT leaves x(4:Nx) uninitialised, so there
        # is no correct reading to recover.
        raise FileFormatError(
            f"read_vector: Nx={Nx} record was terminated by '/' after "
            f"{len(values)} values; only 1 (replicate) or 2 (equally spaced) "
            f"values generate a vector.",
            remediation="Give either 2 values before the '/' or all "
                        f"{Nx} values.",
        )

    return np.atleast_1d(x), Nx
