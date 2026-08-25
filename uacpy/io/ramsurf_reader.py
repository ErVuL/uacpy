"""
Readers for the output of the Collins-style RAM family binaries
uacpy dispatches to (``rams0.5``, ``ramsurf1.5``, ``ramgeo1.5``).

The files read here are:

- ``tl.line`` — ASCII ``range  TL`` rows at the single receiver depth
  ``zr_line`` from row 2 of ``ram.in``, one row per output range step
  (:func:`read_tl_line`). No uacpy model consumes it — the RAM wrappers
  build their ``Field`` from ``tl.grid`` — but it is the run's own
  single-depth trace and the cheapest cross-check on the grid.
- ``tl.grid`` — unformatted Fortran binary. Record 1 is a single int32
  ``lz`` (number of stored depth points). Records 2..N hold ``lz``
  ``real*8`` TL samples each, one record per range output step.
  uacpy builds the Collins binaries with ``-fdefault-real-8``
  (``install.sh``, both Makefiles), so these are 8 bytes, not the 4 of a
  stock build. The uacpy-patched builds additionally write
  ``pcomplex.bin`` on the same grid.

The readers return plain ``(ranges, depths, values)`` arrays; the RAM
wrapper builds a regular ``Field`` from them so the rest of uacpy
(visualization, max-finding, comparisons) handles the output without
special cases.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Tuple, Union

import numpy as np

from uacpy.io._fortran_helpers import (
    detect_endian, read_fortran_record, require_model_output,
    typed_format_error,
)
from uacpy.core.exceptions import FileFormatError


@typed_format_error
def read_tl_line(filepath: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a Collins ``tl.line`` — the ASCII ``range  TL`` trace at the single
    receiver depth ``zr_line`` from row 2 of ``ram.in``.

    Parameters
    ----------
    filepath : str or Path
        Path to the file.

    Returns
    -------
    ranges : ndarray, shape ``(N,)``
        Range in **metres**. The Fortran works in metres throughout and
        writes the range verbatim (``rams0.5.f:253`` /
        ``ramsurf1.5.f:429``: ``write(2,*) r, tl``), so no conversion is
        applied.
    tl : ndarray, shape ``(N,)``
        Transmission loss in dB.

    Raises
    ------
    ~uacpy.core.exceptions.FileFormatError
        The file is absent, holds no rows, or does not carry two columns.
        An empty ``tl.line`` is what a run that died before its first output
        range leaves behind, so it is named rather than indexed into.

    Notes
    -----
    One row is written per output range step (every ``ndr``-th march step),
    so this is :func:`read_tl_grid`'s single-depth sibling on the same range
    axis, and the two agree row for row where ``zr_line`` falls on a stored
    depth. It is unaffected by the ``-fdefault-real-8`` build the binary
    ``tl.grid`` reader has to account for: this file is text.

    See Also
    --------
    read_tl_grid : The full range-depth TL grid from the same run.
    """
    filepath = Path(filepath)
    require_model_output(filepath, 'read_tl_line')

    # ``ndmin=2`` because the default squeezes both a one-row file and a
    # one-column file to the same 1-D array: a stray single-column file would
    # then read as one row of (range, TL) instead of being refused below.
    with warnings.catch_warnings():
        # An empty file is the one case this reader answers with its own
        # typed error below; numpy's UserWarning about it would arrive first
        # and name a numpy frame, so it is dropped here rather than shown
        # alongside the FileFormatError that actually says what to do.
        warnings.filterwarnings('ignore', message='.*input contained no data',
                                category=UserWarning)
        data = np.loadtxt(str(filepath), ndmin=2)
    if data.size == 0:
        raise FileFormatError(
            f"{filepath}: the file holds no rows. RAM writes one "
            f"(range, TL) row per output range step, so an empty tl.line "
            f"means the run stopped before its first output range.",
            remediation="Check the run log and ram.in's rmax / dr / ndr "
                        "row — an ndr larger than the number of march steps "
                        "produces no output.",
        )
    if data.shape[1] < 2:
        raise FileFormatError(
            f"{filepath}: rows carry {data.shape[1]} column(s); a tl.line "
            f"row is 'range TL' (rams0.5.f:253).",
            remediation="Verify this is the tl.line of a Collins RAM run, "
                        "not another output file.",
        )
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


@typed_format_error
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
    require_model_output(filepath, 'read_tl_grid')
    lz, tl = _read_lz_records(filepath, dtype='f8')
    tl = tl.astype(float)
    ranges, depths = _grid_axes(lz, tl.shape[1], dr, ndr, dz, ndz,
                                depth_index_offset)
    return ranges, depths, tl


@typed_format_error
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
    Records 2..N each hold ``lz`` ``complex*16`` samples (8-byte reals — see
    the module docstring on the double-precision build): whatever ``outpt``
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
    require_model_output(filepath, 'read_pcomplex_grid')
    lz, p = _read_lz_records(filepath, dtype='c16')
    p = p.astype(complex)
    ranges, depths = _grid_axes(lz, p.shape[1], dr, ndr, dz, ndz,
                                depth_index_offset)
    return ranges, depths, p
