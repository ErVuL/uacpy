"""
I/O reader for mpiramS output files (psif.dat, recl.dat)

Reads the Fortran direct-access binary output produced by the modified
peramx.f90 program. All values are double precision (float64) per
kinds.f90: wp = kind(1.0d0).

Output format:
  Record 1: Nsam, nf, nzo, nr, c0, cmin, fs, Q  (8 x real(wp))
  Record 2: frq(1:nf)                            (nf x real(wp))
  Record 3: rout(1:nr)                            (nr x real(wp))
  Records 4..3+nzo*nr: for each range ir, for each depth ii:
    zg1(ii), re(psi(ii,1,ir)), im(psi(ii,1,ir)), ..., re(psi(ii,nf,ir)), im(psi(ii,nf,ir))
    = 1 + 2*nf real(wp) values per record
"""

import numpy as np
from pathlib import Path
from typing import Union, Dict


def read_psif(work_dir: Union[str, Path]) -> Dict:
    """
    Read mpiramS output files (psif.dat and recl.dat).

    Parameters
    ----------
    work_dir : str or Path
        Directory containing psif.dat and recl.dat

    Returns
    -------
    dict with keys:
        Nsam : float
            Number of time samples (fs * T)
        nf : int
            Number of frequencies
        nzo : int
            Number of output depth points
        nr : int
            Number of output ranges
        rout : ndarray, shape (nr,)
            Output ranges (m)
        c0 : float
            Mean sound speed (m/s)
        cmin : float
            Minimum sound speed (m/s)
        fs : float
            Sampling frequency (Hz)
        Q : float
            Q value
        frq : ndarray, shape (nf,)
            Frequency vector (Hz)
        zg : ndarray, shape (nzo,)
            Output depth grid (m)
        psif : ndarray, shape (nzo, nf, nr), complex128
            Complex acoustic field
    """
    work_dir = Path(work_dir)
    recl_file = work_dir / 'recl.dat'
    psif_file = work_dir / 'psif.dat'

    if not psif_file.exists():
        raise FileNotFoundError(f"mpiramS output not found: {psif_file}")

    # Read record length from recl.dat
    # gfortran writes recl in bytes (same units as iolength and recl= in open)
    with open(recl_file, 'r') as f:
        recl_bytes = int(f.read().strip())

    # Read the binary file (all values are float64 / complex128)
    with open(psif_file, 'rb') as f:
        # Record 1: header (8 float64 values)
        f.seek(0)
        header = np.fromfile(f, dtype=np.float64, count=8)
        Nsam = float(header[0])
        nf = int(header[1])
        nzo = int(header[2])
        nr = int(header[3])
        c0 = float(header[4])
        cmin = float(header[5])
        fs = float(header[6])
        Q = float(header[7])

        # Record 2: frequency vector
        f.seek(1 * recl_bytes)
        frq = np.fromfile(f, dtype=np.float64, count=nf).copy()

        # Record 3: output ranges
        f.seek(2 * recl_bytes)
        rout = np.fromfile(f, dtype=np.float64, count=nr).copy()

        # Records 4+: depth data, organized as nr blocks of nzo records
        zg = np.zeros(nzo, dtype=np.float64)
        psif = np.zeros((nzo, nf, nr), dtype=np.complex128)

        values_per_record = 1 + 2 * nf  # 1 depth + nf complex values

        for ir in range(nr):
            for ii in range(nzo):
                rec_index = 3 + ir * nzo + ii  # 0-based record
                f.seek(rec_index * recl_bytes)
                record = np.fromfile(f, dtype=np.float64, count=values_per_record)

                if ir == 0:
                    zg[ii] = record[0]

                # Extract complex field: interleaved real, imag pairs
                real_parts = record[1::2][:nf]
                imag_parts = record[2::2][:nf]
                psif[ii, :, ir] = real_parts + 1j * imag_parts

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
