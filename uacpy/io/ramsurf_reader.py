"""
Readers for the output of the Collins-style RAM family binaries
uacpy dispatches to (``rams0.5``, ``ramsurf1.5``, ``ramgeo1.5``).

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

from pathlib import Path
from typing import Tuple, Union

import numpy as np

from uacpy.io._fortran_helpers import detect_endian, read_fortran_record
from uacpy.core.exceptions import FileFormatError


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

    ``dtype`` is an endian-agnostic kind string (``'f8'``, ``'c16'`` — the
    Collins binaries are built with ``-fdefault-real-8``); this
    helper owns byte order entirely. Byte order is auto-detected from the
    first record marker and applied to ``dtype`` here, so callers must not
    pass a ``<``/``>`` prefix (any prefix is stripped defensively). A
    one-shot warning fires the first time a big-endian file is decoded.
    """
    path = Path(filepath)
    with path.open('rb') as f:
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)
        # 12 bytes is the smallest possible file: the header record is a
        # 4-byte marker, one int32 ``lz``, and a 4-byte trailing marker.
        if file_size < 12:
            raise FileFormatError(f"{path}: too short to contain the header record")

        probe = f.read(4)
        f.seek(0)
        endian = detect_endian(probe, source=f'ramsurf_reader:{path.name}')
        # Re-anchor the caller-supplied dtype on the detected byte order so
        # the file decodes correctly regardless of host endianness.
        base_dtype = dtype.lstrip('<>=|')
        item_dtype = np.dtype(endian + base_dtype)

        (lz,) = read_fortran_record(f, 'i', endian=endian)
        expected = lz * item_dtype.itemsize

        columns = []
        # Stop before a partial trailing record: each data record is its
        # ``lz``-sample payload framed by a leading and trailing 4-byte marker.
        while file_size - f.tell() >= 8 + expected:
            payload = read_fortran_record(f, raw=True, endian=endian)
            if len(payload) != expected:
                raise FileFormatError(
                    f"{path}: expected {expected}-byte data record, "
                    f"got {len(payload)}"
                )
            col = np.frombuffer(payload, dtype=item_dtype).astype(
                base_dtype, copy=True
            )
            columns.append(col)

    if not columns:
        raise FileFormatError(f"{path}: no data records found")

    return lz, np.stack(columns, axis=1)


def _grid_axes(lz, n_ranges, dr, ndr, dz, ndz, depth_index_offset):
    """Range and depth axes (m) shared by ``tl.grid`` and ``pcomplex.bin``,
    which are written on the same (z, r) grid."""
    ranges = np.arange(1, n_ranges + 1, dtype=float) * dr * ndr
    depths = (depth_index_offset
              + np.arange(1, lz + 1, dtype=float) * ndz - 1) * dz
    return ranges, depths


def read_tl_grid(
    filepath: Union[str, Path],
    *,
    dr: float,
    ndr: int,
    dz: float,
    ndz: int,
    depth_index_offset: int = 0,
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
        Depth step (m) and output stride from ``ram.in``. The PE grid maps
        grid index ``i`` to depth ``(i - 1) * dz`` (from ``ri = 1 + zr/dz``
        in the Collins binaries), so the ``k``-th stored sample sits at
        ``z = (depth_index_offset + k * ndz - 1) * dz`` for ``k = 1..lz``.
    depth_index_offset : int
        Grid-index marker of the first stored depth sample. ``ramsurf1.5``
        and ``ramgeo1.5`` write from grid index ``ndz`` (offset 0, first
        sample at ``z = (ndz-1)·dz``, i.e. the ``z = 0`` surface node only
        when ``ndz = 1``); ``rams0.5`` writes from ``1 + ndz`` (offset 1,
        first sample at ``z = ndz·dz`` — it never stores ``z = 0``). See
        the ``outpt`` loops in ``third_party/ramsurf/{rams0.5,ramsurf1.5}.f``
        and ``third_party/ramgeo/ramgeo1.5.f``.

    Returns
    -------
    ranges, depths, tl : (ndarray, ndarray, ndarray)
        Range axis (m), depth axis (m), and TL field of shape
        ``(n_depths, n_ranges)``.
    """
    lz, tl = _read_lz_records(filepath, dtype='f8')
    tl = tl.astype(float)
    ranges, depths = _grid_axes(lz, tl.shape[1], dr, ndr, dz, ndz,
                                depth_index_offset)
    return ranges, depths, tl


def read_pcomplex_grid(
    filepath: Union[str, Path],
    *,
    dr: float,
    ndr: int,
    dz: float,
    ndz: int,
    depth_index_offset: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read a uacpy-patched ``pcomplex.bin`` (unformatted Fortran binary).

    Format (added to rams0.5 / ramsurf1.5 / ramgeo1.5 by uacpy — see
    ``third_party/MODIFICATIONS.md``): record 1 holds a single int32 ``lz``
    (number of stored depth points, identical to the ``tl.grid`` header).
    Records 2..N each hold ``lz`` ``complex*8`` samples: whatever ``outpt``
    takes the magnitude of for ``tl.grid``, divided by ``sqrt(r)``. That is
    ``u·f3`` for the fluid codes (``ramsurf1.5.f:438``, ``ramgeo1.5.f:430``)
    and the odd-indexed elastic component ``u(2i-1)`` for RAMS
    (``rams0.5.f:263``), so the two grids stay consistent per backend.
    The carrier ``exp(+i k0 r)`` has been factored out by the PE march;
    the RAM wrapper bakes the engineering travelling-wave carrier
    ``exp(-i k0 r)`` back in before tagging the result.

    Parameters
    ----------
    filepath : str or Path
        Path to ``pcomplex.bin``.
    dr, ndr, dz, ndz, depth_index_offset : as in :func:`read_tl_grid`.

    Returns
    -------
    ranges, depths, p : (ndarray, ndarray, ndarray)
        Range axis (m), depth axis (m), complex envelope of shape
        ``(n_depths, n_ranges)``.
    """
    lz, p = _read_lz_records(filepath, dtype='c16')
    p = p.astype(complex)
    ranges, depths = _grid_axes(lz, p.shape[1], dr, ndr, dz, ndz,
                                depth_index_offset)
    return ranges, depths, p
