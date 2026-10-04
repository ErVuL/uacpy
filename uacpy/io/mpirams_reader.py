"""
I/O reader for mpiramS output (psif.dat).

Parses Fortran sequential unformatted output produced by the modified
``peramx.f90``. All values are double precision (real(wp), kind=
kind(1.0d0)). Each Fortran sequential-unformatted record carries a
4-byte length marker before and after the data; ``scipy.io.FortranFile``
parses this format directly.

Output records:
    1. Header   : Nsam, nf, nzo, nr, c0, cmin, fs, Q  (8 reals)
    2. Frequency: frq(1:nf)                            (nf reals)
    3. Ranges   : rout(1:nr)                           (nr reals)
    4..3+nzo*nr: for each range ir, for each depth ii:
        zg1(ii), re(psi(ii,1,ir)), im(psi(ii,1,ir)), …,
                 re(psi(ii,nf,ir)), im(psi(ii,nf,ir))
        = 1 + 2*nf reals per record

Stock mpiramS writes a different file under the same name: a direct-access
``psif.dat`` whose record length it saves to ``recl.dat``
(``third_party/mpiramS/README.RECL``). uacpy's patched build writes the
sequential layout above (``third_party/MODIFICATIONS.md``), and that is the
one this reader parses; a stock file is recognised by its ``recl.dat`` and
refused by name.
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Union
from uacpy.core._export import ExportRecord
from uacpy.core.exceptions import FileFormatError


@dataclass(frozen=True, eq=False)
class PsifFile(ExportRecord):
    """An mpiramS ``psif.dat`` as :func:`read_psif` reads it.

    Attributes
    ----------
    n_samples : float
        The header's ``Nsam = fs·T`` (``peramx.f90:356``, with
        ``fs = 4·fc`` at ``:354``), the driver's time-sample count, as the
        real the file holds; :class:`~uacpy.Field`'s ``synthesis_floor``
        applies the least-whole-count rule to it.
    c0 : float
        The reference sound speed (m/s), the RAM settings' ``c0``.
    water_min : float
        The header's ``cmin``: the water's minimum sound speed (m/s,
        ``peramx.f90:295``), :class:`~uacpy.core.results.SoundSpeeds`'
        ``water_min``.
    sample_rate : float
        The header's ``fs`` (Hz).
    q_factor : float
        The band's quality factor (the deck's ``Q``).
    frequencies : ndarray
        The marched frequencies (Hz), shape ``(nf,)``.
    depths : ndarray
        The output depth grid (m), shape ``(nzo,)``.
    ranges : ndarray
        The output ranges (m), shape ``(nr,)``.
    pe_field : ndarray
        The PE field ``ψ`` as peramx writes it, complex, shape
        ``(nzo, nf, nr)``; the RAM wrapper turns it into pressure.
    """

    n_samples: float
    c0: float
    water_min: float
    sample_rate: float
    q_factor: float
    frequencies: np.ndarray
    depths: np.ndarray
    ranges: np.ndarray
    pe_field: np.ndarray

    _REPR_FIELDS = ('c0', 'sample_rate', 'q_factor', 'frequencies', 'depths',
                    'ranges', 'pe_field')
    _REPR_UNITS = {'c0': 'm/s', 'sample_rate': 'Hz', 'frequencies': 'Hz',
                   'depths': 'm', 'ranges': 'm'}

    _ARRAY_FIELDS = ('frequencies', 'depths', 'ranges', 'pe_field')
    _XARRAY_FIELDS = {'pe_field': 'pe_field', 'depth': 'depths',
                      'frequency': 'frequencies', 'range': 'ranges'}

    def _payload(self):
        return {'pe_field': (self.pe_field, ('depth', 'frequency', 'range'),
                             '')}

    def _coords(self):
        return {'depth': (self.depths, 'm'),
                'frequency': (self.frequencies, 'Hz'),
                'range': (self.ranges, 'm')}


def read_psif(filepath: Union[str, Path]) -> PsifFile:
    """
    Read mpiramS output file (``psif.dat``).

    Parameters
    ----------
    filepath : str or Path
        The ``psif.dat`` file itself, or the directory containing it.

    Returns
    -------
    psif : PsifFile
        The header scalars (``n_samples`` ← Fortran ``Nsam``,
        ``water_min`` ← ``cmin``, ``sample_rate`` ← ``fs``, ``c0``,
        ``q_factor`` ← ``Q``), the
        ``frequencies`` (Hz), ``depths`` and ``ranges`` (m), and the
        complex ``pe_field``, shape ``(nzo, nf, nr)``.

    Raises
    ------
    FileFormatError
        No ``psif.dat`` at the path, a malformed or truncated file, or a
        stock mpiramS direct-access file (a ``recl.dat`` beside it), which
        is a different layout from the sequential one uacpy's patched build
        writes.
    """
    filepath = Path(filepath)
    psif_file = filepath / 'psif.dat' if filepath.is_dir() else filepath

    if not psif_file.exists():
        raise FileFormatError(f"mpiramS output not found: {psif_file}.")

    # ``psif.dat`` is written by mpiramS on this host during the same run, so
    # its byte order is the host's; ``FortranFile`` reads native endianness,
    # which is therefore correct here. (Unlike the vendored/cross-host binaries
    # read elsewhere, this is never a foreign-endian file — so it does not go
    # through ``_fortran_helpers.detect_endian``.)
    # Imported here rather than at module level so ``import uacpy.io`` does
    # not pull scipy in; only this reader needs it.
    from scipy.io import FortranEOFError, FortranFormattingError
    stock = (psif_file.parent / 'recl.dat').exists()
    try:
        return _read_psif_records(psif_file)
    except (FortranEOFError, FortranFormattingError, ValueError,
            FileFormatError) as exc:
        if stock:
            raise FileFormatError(
                f"{psif_file}: not the sequential layout uacpy's patched "
                f"mpiramS writes; the recl.dat beside it marks a stock "
                f"mpiramS direct-access psif.dat "
                f"({type(exc).__name__}: {exc}).",
                remediation="Re-run the case through uacpy.RAM "
                            "(backend='mpirams'), whose patched binary writes "
                            "the sequential file, or read the direct-access "
                            "records yourself with the record length in "
                            "recl.dat (third_party/mpiramS/README.RECL).",
            ) from exc
        if isinstance(exc, FileFormatError):
            raise
        # scipy's FortranFile raises TypeError-derived FortranEOFError /
        # FortranFormattingError on a truncated or mis-framed record, and
        # ValueError on a garbage length marker; all mean the same thing
        # here — psif.dat is not a complete mpiramS output.
        raise FileFormatError(
            f"{psif_file}: malformed or truncated mpiramS output "
            f"({type(exc).__name__}: {exc}).",
            remediation="The mpiramS run may have been killed mid-write; "
                        "re-run it, or check the path points at a "
                        "completed run.",
        ) from exc


def _read_psif_records(psif_file: Path) -> PsifFile:
    """Walk the sequential-unformatted records of one ``psif.dat``."""
    from scipy.io import FortranFile
    with FortranFile(str(psif_file), 'r') as f:
        header = f.read_reals(dtype=np.float64)
        if header.size != 8:
            raise FileFormatError(
                f"{psif_file}: header has {header.size} reals, expected 8."
            )
        Nsam = float(header[0])
        nf = int(header[1])
        nzo = int(header[2])
        nr = int(header[3])
        c0 = float(header[4])
        cmin = float(header[5])
        fs = float(header[6])
        Q = float(header[7])

        frq = f.read_reals(dtype=np.float64).copy()
        rout = f.read_reals(dtype=np.float64).copy()
        if frq.size != nf or rout.size != nr:
            raise FileFormatError(
                f"{psif_file}: header says nf={nf}, nr={nr}; got frq.size="
                f"{frq.size}, rout.size={rout.size}."
            )

        # ``nf`` and ``nr`` are validated against the frq/rout records above;
        # ``nzo`` is a raw header field driving the (nzo, nf, nr) psif
        # allocation. Each depth record holds 1 + 2*nf float64 (+ two Fortran
        # length markers), so nzo*nr records cannot occupy more than the file;
        # bound nzo against the remaining bytes before allocating to reject a
        # corrupt/garbage header (e.g. nzo = 0x7fffffff) that would otherwise
        # drive a multi-GB np.zeros.
        if nzo < 0:
            raise FileFormatError(f"{psif_file}: negative nzo={nzo}.")
        rec_bytes = (1 + 2 * nf) * 8 + 8  # payload + two 4-byte markers
        file_size = psif_file.stat().st_size
        max_records = file_size // max(rec_bytes, 1)
        if nr > 0 and nzo > max_records // nr:
            raise FileFormatError(
                f"{psif_file}: header counts (nzo={nzo}, nf={nf}, nr={nr}) "
                f"imply more depth records than a {file_size}-byte file holds."
            )

    # Depth records: 1 + 2*nf reals each, nzo records per range, nr ranges.
    # Each record is [z, Re_1, Im_1, ..., Re_nf, Im_nf], so the real parts
    # are the odd slots and the imaginary parts the even ones after z. Every
    # record has the same length, so the body is read as one structured array
    # (one Python call, not one per record) behind the three header records:
    # 8 reals, nf reals, nr reals, each framed by two 4-byte length markers.
    payload = 8 * (1 + 2 * nf)
    record = np.dtype([('head', '=i4'), ('z', '=f8'),
                       ('re_im', '=f8', (2 * nf,)), ('tail', '=i4')])
    offset = (8 + 8 * 8) + (8 + 8 * nf) + (8 + 8 * nr)
    n_records = nzo * nr
    body = np.fromfile(psif_file, dtype=record, count=n_records,
                       offset=offset)
    # Framing first: a record of the wrong length shifts every later one, so
    # the first bad marker names the disagreement where a bare count would
    # only say "short". Exact length, not a minimum: a record longer than the
    # header implies is as much a header/file disagreement as a short one.
    bad = np.flatnonzero((body['head'] != payload) | (body['tail'] != payload))
    if bad.size:
        ir, ii = divmod(int(bad[0]), nzo) if nzo else (0, 0)
        raise FileFormatError(
            f"{psif_file}: depth record (ir={ir}, iz={ii}) holds "
            f"{int(body['head'][bad[0]]) // 8} reals, expected {1 + 2 * nf} "
            f"(z + {nf} complex values); the file does not match its own "
            f"header."
        )
    if body.size != n_records:
        raise FileFormatError(
            f"{psif_file}: holds {body.size} of the {n_records} depth records "
            f"its header announces (nzo={nzo} x nr={nr}); the file is "
            f"truncated."
        )
    # The depth axis repeats across ranges; take it from the first only.
    zg = body['z'][:nzo].copy()
    re_im = body['re_im'].reshape(nr, nzo, 2 * nf) if n_records else \
        np.zeros((nr, nzo, 2 * nf))
    psif = np.transpose(re_im[:, :, 0::2] + 1j * re_im[:, :, 1::2],
                        (1, 2, 0)).astype(np.complex128)

    return PsifFile(n_samples=Nsam, c0=c0, water_min=cmin, sample_rate=fs,
                    q_factor=Q,
                    frequencies=frq, depths=zg, ranges=rout, pe_field=psif)
