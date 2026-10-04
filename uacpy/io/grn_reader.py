"""
Green's function reader for SCOOTER (and SPARC snapshot mode).

The wavenumber-domain Green's function file is the SHD-format binary written
by SCOOTER (``scooter.f90``) or SPARC in snapshot mode (``sparc.f90``).
:func:`read_grn_file` parses it into a
:class:`~uacpy.core.results.GreensFunction`, whose methods do the
Hankel/Fourier transform that ``Acoustics-Toolbox/Matlab/Scooter/fieldsco.m``
does.

Convention summary
------------------
* SCOOTER — the record-4 vector holds the frequencies, and per-frequency
  wavenumbers are recovered from the stored phase-speed grid as
  ``k(f) = 2π·f/c``.
* SPARC snapshot — the record-4 vector holds the output *times*
  (``sparc.f90:320``), not frequencies, and the wavenumber grid is
  **frequency-independent**: it maps through the source frequency ``freq0``
  of the header (``fieldsco.m:100-102``).

A file is a SPARC snapshot when its title starts with ``'SPARC'`` (set at
``sparc.f90:84``).
"""

import numpy as np
from pathlib import Path
from typing import Union

from uacpy.core.results import GreensFunction
from uacpy.io._fortran_helpers import (
    DirectAccessFile, _bound_counts, require_model_output, typed_format_error,
)
from uacpy.io.oalib_reader import read_shd_header
from uacpy.core.exceptions import FileFormatError


@typed_format_error
def read_grn_file(filepath: Union[str, Path]) -> GreensFunction:
    """
    Read a SCOOTER / SPARC Green's function file (``.grn``).

    The format is the same Fortran direct-access binary as ``.shd`` (record
    length stored in 4-byte words; see ``misc/RWSHDFile.f90``).

    Parameters
    ----------
    filepath : str or Path
        The file to read.

    Returns
    -------
    GreensFunction
        ``data`` is the complex64 ``G`` of shape ``(nfreq, nsd, nrd, nk)``
        as stored; ``phase_speeds`` the grid in REC=10 (the ``Pos%Rr``
        slot), monotonically decreasing; ``source_depths`` and
        ``receiver_depths`` in metres; ``stabilizing_attenuation`` the REC=3
        ``atten`` (for SCOOTER ``Δk`` unless TopOpt(7)='0', then 0; for a
        SPARC snapshot 0, ``sparc.f90:313``); ``title`` the title line.
        A SCOOTER file carries the record-4 vector as ``frequencies``; a
        SPARC snapshot carries it as ``times`` and its REC=3 source
        frequency ``freq0`` as ``frequencies``. ``model`` is ``'SPARC'``,
        ``'Scooter'`` or ``''`` by the title's prefix.
    """
    filepath = Path(filepath)
    require_model_output(filepath, 'read_grn_file')

    with open(filepath, "rb") as f:
        daf = DirectAccessFile(f, source=f'read_grn_file:{filepath.name}')
        f4, f8 = daf.f4, daf.f8

        # Records 1-3: title, PlotType, then Nfreq Ntheta NSx NSy NSz NRz NRr
        # with freq0 and atten; Ntheta, NSx and NSy are unused here.
        header = read_shd_header(daf)
        title = header.title
        nfreq, _ntheta, _nsx, _nsy, nsd, nrd, nk = header.counts
        freq0, atten = header.freq0, header.atten

        # File-size-aware sanity bound on the header counts before any
        # vector or the (nfreq, nsd, nrd, nk) cube is sized off them. A
        # corrupt/hostile header (e.g. nk=0x3ffffff0, or all counts = 1000)
        # would otherwise drive a multi-GB/TB allocation before a single
        # data record is validated. The G cube holds nfreq*nsd*nrd*nk
        # complex samples, each stored on disk as 2 float32 (8 bytes).
        _bound_counts(str(filepath), daf.file_size, 8,
                      nfreq=nfreq, nsd=nsd, nrd=nrd, nk=nk)

        # Record 4: frequency vector (or time vector for SPARC snapshot)
        freqVec = daf.vector(3, f8, nfreq)

        # Records 5-7: theta / sx / sy — skipped.
        # Records 8-10 hold Pos%Sz, Pos%Rz and Pos%Rr (RWSHDFile.f90:111-114).
        # The depth vectors are REAL(KIND=4) and the range vector — whose slot
        # the phase speeds occupy — is REAL(KIND=8)
        # (SourceReceiverPositions.f90:23-25); the record-length formula counts
        # them the same way, ``Pos%NSz, Pos%NRz, 2 * Pos%NRr`` words
        # (RWSHDFile.f90:100).
        sd = daf.vector(7, f4, nsd)          # Record 8: source depths
        rd = daf.vector(8, f4, nrd)          # Record 9: receiver depths
        cVec = daf.vector(9, f8, nk)         # Record 10: phase speeds, Nk
        # A vector cut short by the end of the file would otherwise reach
        # the GreensFunction as axes that disagree with the header counts.
        for what, record, got, want in (
                ('frequency/time', 4, freqVec, nfreq),
                ('source-depth', 8, sd, nsd),
                ('receiver-depth', 9, rd, nrd),
                ('phase-speed', 10, cVec, nk)):
            if got.size < want:
                raise FileFormatError(
                    f"read_grn_file: truncated {what} record {record} "
                    f"(expected {want} values, got {got.size})")

        # Records 11+: complex Green's function, one record per
        # (freq, source_depth, receiver_depth) tuple, receiver depth fastest —
        # ``REC = 10 + (ifreq-1)*NSz*NRz + (iS-1)*NRz + iR`` (scooter.f90:588).
        # ``irec`` is the 0-based record index (record ``irec + 1`` in the
        # Fortran count), so starting at 9 and incrementing before the first
        # read lands the first slab on record 11.
        #
        # SPARC's snapshot indexes ``iG = (Itout-1)*Pos%NRz + ir``
        # (sparc.f90:286-287) — output time stands in for the frequency axis
        # (``WriteHeaderSparc`` sets ``Nfreq = Ntout``, sparc.f90:318-320) and
        # there is NO source-depth factor, because ``Green`` carries no
        # source-depth axis in a snapshot run. That matches the sequential walk
        # below only while ``NSz == 1``. ``WriteHeaderSparc`` never rewrites
        # ``Pos%NSz``, so a snapshot header still reports whatever the deck
        # asked for: were a deck to carry two source depths, the walk would
        # read time slot 2's slabs into slot 1's second source and run off the
        # end of the file. `SPARC.run` refuses a multi-depth Source for exactly
        # this reason, so the case is unreachable from the public API — this
        # check states that coupling here, where the layout assumption lives.
        if title.upper().startswith('SPARC') and nsd != 1:
            raise FileFormatError(
                f"read_grn_file: SPARC snapshot header reports NSz={nsd}, but "
                f"sparc.f90:286-287 writes its records with no source-depth "
                f"factor, so only NSz=1 is readable. The Green's function of a "
                f"snapshot run carries no source-depth axis.",
            )
        G = np.zeros((nfreq, nsd, nrd, nk), dtype=np.complex64)
        irec = 9
        for ifreq in range(nfreq):
            for isd in range(nsd):
                for ird in range(nrd):
                    irec += 1
                    data = daf.vector(irec, f4, 2 * nk)
                    if data.size < 2 * nk:
                        raise FileFormatError(
                            f"read_grn_file: truncated Green's-function record "
                            f"at ifreq={ifreq}, isd={isd}, ird={ird} "
                            f"(expected {2 * nk} float32 values, got {data.size})"
                        )
                    G[ifreq, isd, ird, :] = data[0::2] + 1j * data[1::2]

    is_sparc = title.upper().startswith('SPARC')
    model = ('SPARC' if is_sparc
             else 'Scooter' if title.upper().startswith('SCOOTER') else '')
    return GreensFunction(
        data=G,
        phase_speeds=cVec,
        receiver_depths=rd,
        source_depths=sd,
        frequencies=float(freq0) if is_sparc else freqVec,
        times=freqVec if is_sparc else None,
        stabilizing_attenuation=float(atten),
        title=title,
        model=model,
    )
