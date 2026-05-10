"""
Readers for the output of the Collins-style RAM family binaries
uacpy dispatches to (``rams0.5``, ``ramsurf1.5``).

Two output files are produced per run:

- ``tl.line`` — ASCII ``range  TL`` rows at the receiver depth ``zr_line``
  configured in row 2 of ``ram.in``. One row per range step.
- ``tl.grid`` — unformatted Fortran binary. Record 1 is a single int32
  ``lz`` (number of stored depth points). Records 2..N hold ``lz``
  ``real*4`` TL samples each, one record per range output step.

The reader returns a regular ``Field`` of ``field_type='tl'`` so the rest
of uacpy (visualization, max-finding, comparisons) handles the output
without special cases.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Tuple, Union

import numpy as np


def read_tl_line(filepath: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a Collins ``tl.line`` (ASCII range, TL).

    Parameters
    ----------
    filepath : str or Path
        Path to the file.

    Returns
    -------
    ranges, tl : (ndarray, ndarray)
        Range (m) and transmission loss (dB), shape ``(N,)``.
    """
    data = np.loadtxt(str(filepath))
    if data.ndim == 1:
        data = data[None, :]
    return data[:, 0].astype(float), data[:, 1].astype(float)


def _read_lz_records(
    filepath: Union[str, Path], *, dtype: str
) -> Tuple[int, np.ndarray]:
    """
    Read a Fortran-unformatted file whose record 1 is ``int32 lz`` and
    records 2..N each hold ``lz`` samples of ``dtype``. Returns ``(lz,
    matrix[lz, n_records])``.
    """
    path = Path(filepath)
    raw = path.read_bytes()
    pos = 0
    item_size = np.dtype(dtype).itemsize

    if len(raw) < 12:
        raise ValueError(f"{path}: too short to contain the header record")
    rec_len = struct.unpack('<i', raw[pos:pos + 4])[0]
    pos += 4
    if rec_len != 4:
        raise ValueError(
            f"{path}: expected 4-byte header record, got {rec_len}"
        )
    lz = struct.unpack('<i', raw[pos:pos + 4])[0]
    pos += 4
    rec_len_close = struct.unpack('<i', raw[pos:pos + 4])[0]
    pos += 4
    if rec_len_close != 4:
        raise ValueError(
            f"{path}: malformed header record marker {rec_len_close}"
        )

    columns = []
    expected = lz * item_size
    while pos < len(raw):
        if len(raw) - pos < 8 + expected:
            break
        rec_len = struct.unpack('<i', raw[pos:pos + 4])[0]
        pos += 4
        if rec_len != expected:
            raise ValueError(
                f"{path}: expected {expected}-byte data record, got {rec_len}"
            )
        col = np.frombuffer(raw[pos:pos + expected], dtype=dtype).copy()
        pos += expected
        rec_len_close = struct.unpack('<i', raw[pos:pos + 4])[0]
        pos += 4
        if rec_len_close != expected:
            raise ValueError(
                f"{path}: malformed data-record closing marker {rec_len_close}"
            )
        columns.append(col)

    if not columns:
        raise ValueError(f"{path}: no data records found")

    return lz, np.stack(columns, axis=1)


def read_tl_grid(
    filepath: Union[str, Path],
    *,
    dr: float,
    ndr: int,
    dz: float,
    ndz: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a Collins ``tl.grid`` (unformatted Fortran binary).

    Parameters
    ----------
    filepath : str or Path
        Path to the file.
    dr, ndr : float, int
        Range step (m) and output stride from ``ram.in``. Output ranges
        are at ``r = k * dr * ndr`` for ``k = 1, 2, ...``.
    dz, ndz : float, int
        Depth step (m) and output stride from ``ram.in``. Output depths
        are at ``z = k * dz * ndz`` for ``k = 1, 2, ..., lz``.

    Returns
    -------
    ranges, depths, tl : (ndarray, ndarray, ndarray)
        Range axis (m), depth axis (m), and TL field of shape
        ``(n_depths, n_ranges)``.
    """
    lz, tl = _read_lz_records(filepath, dtype='<f4')
    tl = tl.astype(float)
    n_ranges = tl.shape[1]
    ranges = np.arange(1, n_ranges + 1, dtype=float) * dr * ndr
    depths = np.arange(1, lz + 1, dtype=float) * dz * ndz
    return ranges, depths, tl


def read_pcomplex_grid(
    filepath: Union[str, Path],
    *,
    dr: float,
    ndr: int,
    dz: float,
    ndz: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a uacpy-patched ``pcomplex.bin`` (unformatted Fortran binary).

    Format (added to rams0.5 / ramsurf1.5 by uacpy — see
    ``third_party/MODIFICATIONS.md``): record 1 holds a single int32 ``lz``
    (number of stored depth points, identical to the ``tl.grid`` header).
    Records 2..N each hold ``lz`` ``complex*8`` samples — the envelope
    ``u·f3 / sqrt(r)`` evaluated at the same (z, r) grid as ``tl.grid``.
    The carrier ``exp(+i k0 r)`` has been factored out by the PE march;
    the RAM wrapper bakes the engineering travelling-wave carrier
    ``exp(-i k0 r)`` back in before tagging the result.

    Parameters
    ----------
    filepath : str or Path
        Path to ``pcomplex.bin``.
    dr, ndr, dz, ndz : as in :func:`read_tl_grid`.

    Returns
    -------
    ranges, depths, p : (ndarray, ndarray, ndarray)
        Range axis (m), depth axis (m), complex envelope of shape
        ``(n_depths, n_ranges)``.
    """
    lz, p = _read_lz_records(filepath, dtype='<c8')
    p = p.astype(complex)
    n_ranges = p.shape[1]
    ranges = np.arange(1, n_ranges + 1, dtype=float) * dr * ndr
    depths = np.arange(1, lz + 1, dtype=float) * dz * ndz
    return ranges, depths, p
