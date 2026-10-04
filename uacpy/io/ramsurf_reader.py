"""
Readers for the output of the Collins-style RAM family binaries
uacpy dispatches to (``rams0.5``, ``ramsurf1.5``, ``ramgeo1.5``).

The files read here are:

- ``tl.line`` — ASCII ``range  TL`` rows at the single receiver depth
  ``zr_line`` from row 2 of ``ram.in``, one row per **march** step
  (:func:`read_tl_line`). No uacpy model consumes it — the RAM wrappers
  build their ``Field`` from ``tl.grid`` — but it is the run's own
  single-depth trace, on a finer range axis than the grid, and the cheapest
  cross-check on it. The Collins results carry its path as
  ``metadata['tl_line_file']`` when the work dir survives.
- ``tl.grid`` — unformatted Fortran binary. Record 1 is a single int32
  ``lz`` (number of stored depth points). Records 2..N hold ``lz``
  ``real*8`` TL samples each, one record per range output step.
  uacpy builds the Collins binaries with ``-fdefault-real-8``
  (``install.sh``, both Makefiles), so these are 8 bytes, not the 4 of a
  stock build. The readers take the precision from the first data record's
  length (``4·lz`` or ``8·lz`` bytes), so a stock single-precision
  ``tl.grid`` reads too, returned as float64. The uacpy-patched builds
  additionally write ``pcomplex.bin`` on the same grid.

The readers return a :class:`PeGrid`, which names the quantity its values
are (transmission loss in dB, or the complex PE envelope before the RAM
wrapper's carrier and Hankel phase). The RAM wrapper reads the same files
through :mod:`uacpy.io._parsers` and builds a regular ``Field`` from them.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from uacpy.core._export import ExportRecord

from uacpy.io._fortran_helpers import (
    detect_endian, read_fortran_record, require_model_output,
    typed_format_error,
)
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.core.exceptions import ConfigurationError, FileFormatError, IOWarning


#: The quantities a Collins output file holds, and their units.
PE_QUANTITY_UNITS = {'transmission_loss': 'dB', 'pe_envelope': ''}


@dataclass(frozen=True, eq=False)
class PeGrid(ExportRecord):
    """A Collins RAM output file as :func:`read_tl_line`,
    :func:`read_tl_grid` or :func:`read_pcomplex_grid` reads it.

    Attributes
    ----------
    quantity : {'transmission_loss', 'pe_envelope'}
        What ``data`` is: transmission loss (``tl.line``, ``tl.grid``),
        or the complex PE envelope ``pcomplex.bin`` holds — the field the
        march carries, *before* the RAM wrapper multiplies in the carrier
        and the Hankel phase. An envelope is not a pressure.
    unit : str
        ``'dB'`` for transmission loss; ``''`` for the envelope, which is
        in the binary's own normalisation.
    ranges : ndarray
        The output ranges (m).
    depths : ndarray or None
        The output depths (m); ``None`` for ``tl.line``, whose single depth
        is ``zr_line`` on row 2 of ``ram.in``, not in the file.
    data : ndarray
        Shape ``(n_depths, n_ranges)``, or ``(n_ranges,)`` for ``tl.line``.
    """

    quantity: str
    unit: str
    ranges: np.ndarray
    depths: Optional[np.ndarray]
    data: np.ndarray

    _ARRAY_FIELDS = ('ranges', 'depths', 'data')
    _XARRAY_FIELDS = {'data': 'data', 'range': 'ranges',
                      'depth': 'depths'}

    def __post_init__(self):
        if PE_QUANTITY_UNITS.get(self.quantity) != self.unit:
            raise ConfigurationError(
                f"PeGrid: quantity {self.quantity!r} with unit "
                f"{self.unit!r}; the quantities and their units are "
                f"{PE_QUANTITY_UNITS}.")
        super().__post_init__()

    def _payload(self):
        dims = ('range',) if self.depths is None else ('depth', 'range')
        return {'data': (self.data, dims, self.unit)}

    def _coords(self):
        coords = {'range': (self.ranges, 'm')}
        if self.depths is not None:
            coords['depth'] = (self.depths, 'm')
        return coords


@typed_format_error
def read_tl_line(filepath: Union[str, Path]) -> PeGrid:
    """
    Read a Collins ``tl.line`` — the ASCII ``range  TL`` trace at the single
    receiver depth ``zr_line`` from row 2 of ``ram.in``.

    Parameters
    ----------
    filepath : str or Path
        Path to the file.

    Returns
    -------
    tl : PeGrid
        ``quantity='transmission_loss'`` in ``'dB'``, ``depths`` ``None``
        (the depth is ``zr_line`` in ``ram.in``), ``ranges`` in **metres**
        — the Fortran works in metres throughout and writes the range
        verbatim (``rams0.5.f:253`` / ``ramsurf1.5.f:429``:
        ``write(2,*) r, tl``), so no conversion is applied — and
        ``data``, shape ``(N,)``.

    Raises
    ------
    ~uacpy.core.exceptions.FileFormatError
        The file is absent, holds no rows, or does not carry two columns.
        An empty ``tl.line`` is what a run that died before its first output
        range leaves behind, so it is named rather than indexed into.

    Notes
    -----
    One row is written per **march** step, i.e. every ``dr``, while
    ``tl.grid`` is written every ``ndr``-th step: the ``write(2,*)r,tl`` sits
    above the ``if(mdr.eq.ndr)`` block that gates the grid record
    (``ramgeo1.5.f:420-425``, ``rams0.5.f``, ``ramsurf1.5.f`` alike). So the
    two share a starting range and a single-depth quantity but **not** a
    range axis: with ``ndr=2`` on a 5 km run at ``dr=2`` m this file holds
    2500 rows spaced 2 m against ``tl.grid``'s 1250 spaced 4 m. It is
    unaffected by the ``-fdefault-real-8`` build the binary ``tl.grid``
    reader has to account for: this file is text.

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
    return PeGrid(quantity='transmission_loss', unit='dB',
                  ranges=data[:, 0].astype(float), depths=None,
                  data=data[:, 1].astype(float))


#: Double-precision kind → the single-precision kind a stock build writes.
_SINGLE_PRECISION = {'f8': 'f4', 'c16': 'c8'}


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

    ``dtype`` names the double-precision kind; a first data record of half
    that length (``lz`` single-precision samples, a stock build) switches
    the file to ``'f4'``/``'c8'``, returned upcast to ``dtype``. Bytes left
    after the last whole record (a run cut short mid-write) warn with the
    count of records kept.
    """
    path = Path(filepath)
    with path.open('rb') as f:
        f.seek(0, 2)
        file_size = f.tell()
        f.seek(0)
        # 12 bytes is the smallest possible file: the header record is a
        # 4-byte marker, one int32 ``lz``, and a 4-byte trailing marker.
        if file_size < 12:
            raise FileFormatError(f"{path}: too short to contain the header record.")

        probe = f.read(4)
        f.seek(0)
        endian = detect_endian(probe, source=f'ramsurf_reader:{path.name}')
        # Re-anchor the caller-supplied dtype on the detected byte order so
        # the file decodes correctly regardless of host endianness.
        base_dtype = dtype.lstrip('<>=|')
        item_dtype = np.dtype(endian + base_dtype)

        (lz,) = read_fortran_record(f, 'i', endian=endian)
        single = _SINGLE_PRECISION.get(base_dtype)
        if single is not None and lz > 0 and file_size - f.tell() >= 4:
            (first_len,) = np.frombuffer(f.read(4), dtype=endian + 'i4')
            f.seek(-4, 1)
            if first_len == lz * np.dtype(single).itemsize:
                item_dtype = np.dtype(endian + single)
        expected = lz * item_dtype.itemsize

        columns = []
        # Stop before a partial trailing record: each data record is its
        # ``lz``-sample payload framed by a leading and trailing 4-byte marker.
        while file_size - f.tell() >= 8 + expected:
            payload = read_fortran_record(f, raw=True, endian=endian)
            if len(payload) != expected:
                raise FileFormatError(
                    f"{path}: expected {expected}-byte data record, "
                    f"got {len(payload)}."
                )
            col = np.frombuffer(payload, dtype=item_dtype).astype(
                base_dtype, copy=True
            )
            columns.append(col)
        leftover = file_size - f.tell()

    if not columns:
        raise FileFormatError(f"{path}: no data records found.")
    if leftover:
        warnings.warn(
            f"{path.name}: {leftover} byte(s) after the last whole "
            f"{expected}-byte record are not a complete range step and were "
            f"dropped; {len(columns)} range step(s) read. The run was cut "
            f"short while writing.",
            IOWarning, skip_file_prefixes=USER_FRAME_SKIP)

    return lz, np.stack(columns, axis=1)


def _grid_axes(lz, n_ranges, dr, ndr, dz, ndz, depth_index_offset):
    """Range and depth axes (m) shared by ``tl.grid`` and ``pcomplex.bin``,
    which are written on the same (z, r) grid."""
    ranges = np.arange(1, n_ranges + 1, dtype=float) * dr * ndr
    depths = (depth_index_offset
              + np.arange(1, lz + 1, dtype=float) * ndz - 1) * dz
    return ranges, depths


def _read_grid(filepath, reader, dtype, dr, ndr, dz, ndz, depth_index_offset):
    """``(ranges, depths, field)`` of a ``tl.grid``-shaped file; the field is
    native-endian ``dtype`` straight from :func:`_read_lz_records`."""
    require_model_output(filepath, reader)
    lz, field = _read_lz_records(filepath, dtype=dtype)
    ranges, depths = _grid_axes(lz, field.shape[1], dr, ndr, dz, ndz,
                                depth_index_offset)
    return ranges, depths, field


def read_tl_grid(
    filepath: Union[str, Path],
    *,
    dr: float,
    ndr: int,
    dz: float,
    ndz: int,
    depth_index_offset: int = 0,
) -> PeGrid:
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
    tl : PeGrid
        ``quantity='transmission_loss'`` in ``'dB'``: the range axis (m),
        the depth axis (m), and ``data`` of shape
        ``(n_depths, n_ranges)``.
    """
    ranges, depths, tl = _read_grid(filepath, 'read_tl_grid', 'f8', dr, ndr,
                                    dz, ndz, depth_index_offset)
    return PeGrid(quantity='transmission_loss', unit='dB', ranges=ranges,
                  depths=depths, data=tl)


def read_pcomplex_grid(
    filepath: Union[str, Path],
    *,
    dr: float,
    ndr: int,
    dz: float,
    ndz: int,
    depth_index_offset: int = 0,
) -> PeGrid:
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
    The travelling-wave carrier differs per backend. The fluid codes
    (ramsurf1.5 / ramgeo1.5) absorb it into the operator function, so
    their envelope carries no ``exp(+i k0 r)`` and the RAM wrapper
    multiplies ``exp(-i k0 r)`` back in; rams0.5's ``g0`` march step
    bakes ``exp(+i k0 r*rot0)`` into ``u``, so its envelope arrives
    with the carrier and the wrapper adds none. The wrapper conjugates
    both and applies the shared ``exp(-i pi/4)`` Hankel phase
    (``psi_to_travelling_wave`` in ``models/ram/_pe_phase.py``) before tagging
    the result.

    Parameters
    ----------
    filepath : str or Path
        Path to ``pcomplex.bin``.
    dr, ndr, dz, ndz, depth_index_offset : as in :func:`read_tl_grid`.

    Returns
    -------
    envelope : PeGrid
        ``quantity='pe_envelope'`` (unit ``''``, the binary's own
        normalisation): the range axis (m), the depth axis (m), and the
        complex envelope of shape ``(n_depths, n_ranges)`` — *before* the
        carrier and the Hankel phase the RAM wrapper applies, so not a
        pressure.
    """
    ranges, depths, p = _read_grid(filepath, 'read_pcomplex_grid', 'c16', dr,
                                   ndr, dz, ndz, depth_index_offset)
    return PeGrid(quantity='pe_envelope', unit='', ranges=ranges,
                  depths=depths, data=p)
