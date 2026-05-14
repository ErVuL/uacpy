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
"""

import numpy as np
from pathlib import Path
from typing import Union, Dict
from scipy.io import FortranFile


def read_psif(work_dir: Union[str, Path]) -> Dict:
    """
    Read mpiramS output file (``psif.dat``).

    Parameters
    ----------
    work_dir : str or Path
        Directory containing ``psif.dat``.

    Returns
    -------
    dict with keys:
        Nsam, nf, nzo, nr  : ints / floats from the header
        c0, cmin, fs, Q    : float scalars from the header
        rout : ndarray, shape (nr,)        — output ranges (m)
        frq  : ndarray, shape (nf,)        — frequency vector (Hz)
        zg   : ndarray, shape (nzo,)       — output depth grid (m)
        psif : ndarray, shape (nzo, nf, nr), complex128 — acoustic field
    """
    work_dir = Path(work_dir)
    psif_file = work_dir / 'psif.dat'

    if not psif_file.exists():
        raise FileNotFoundError(f"mpiramS output not found: {psif_file}")

    with FortranFile(str(psif_file), 'r') as f:
        header = f.read_reals(dtype=np.float64)
        if header.size != 8:
            raise ValueError(
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
            raise ValueError(
                f"{psif_file}: header says nf={nf}, nr={nr}; got frq.size="
                f"{frq.size}, rout.size={rout.size}."
            )

        # Depth records: 1 + 2*nf reals each, nzo records per range, nr ranges.
        zg = np.zeros(nzo, dtype=np.float64)
        psif = np.zeros((nzo, nf, nr), dtype=np.complex128)
        for ir in range(nr):
            for ii in range(nzo):
                rec = f.read_reals(dtype=np.float64)
                if ir == 0:
                    zg[ii] = rec[0]
                psif[ii, :, ir] = rec[1::2][:nf] + 1j * rec[2::2][:nf]

    return {
        'Nsam': Nsam,
        'nf': nf,
        'nzo': nzo,
        'nr': nr,
        'rout': rout,
        'c0': c0,
        'cmin': cmin,
        'fs': fs,
        'Q': Q,
        'frq': frq,
        'zg': zg,
        'psif': psif,
    }
