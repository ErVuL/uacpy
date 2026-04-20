"""
I/O writers for mpiramS (Fortran PE model)

Generates input files for the mpiramS binary:
- in.pe: main configuration file
- SSP file: sound speed profiles
- BTH file: bathymetry
- ranges file: output ranges
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union


def write_inpe(
    filepath: Union[str, Path],
    fc: float,
    Q: float,
    T: float,
    zsrc: float,
    deltaz: float,
    deltar: float,
    np_pade: int,
    nss: int,
    rs: float,
    dzm: int,
    ssp_filename: str,
    iflat: int,
    ihorz: int,
    ibot: int,
    bth_filename: str,
    ranges_filename: str,
    sedlayer: float = 300.0,
    nzs: int = 50,
    cs: Optional[np.ndarray] = None,
    rho: Optional[np.ndarray] = None,
    attn: Optional[np.ndarray] = None,
    isedrd: int = 0,
    sed_filename: str = '',
):
    """
    Write mpiramS input configuration file (in.pe).

    Uses list-directed (free-format) reads matching the modified peramx.f90.

    Parameters
    ----------
    filepath : str or Path
        Output file path
    fc : float
        Center frequency (Hz)
    Q : float
        Q value (bandwidth = fc/Q). Use large Q (e.g. 1e6) for narrowband/TL mode.
    T : float
        Time window width (s)
    zsrc : float
        Source depth (m)
    deltaz : float
        Depth accuracy parameter (m). Typical: 0.5
    deltar : float
        Range accuracy parameter (m). Typical: 250.0
    np_pade : int
        Number of Pade coefficients (2-8, typical: 4)
    nss : int
        Number of stability terms (0 for short range, 1-2 for long range)
    rs : float
        Stability range (m)
    dzm : int
        Output depth decimation factor
    ssp_filename : str
        Sound speed profile filename
    iflat : int
        Flat earth transformation flag (0=no, 1=yes)
    ihorz : int
        Horizontal linear interpolation flag (0=no, 1=yes)
    ibot : int
        Ocean bottom flag (0=no bottom file, 1=read bathymetry)
    bth_filename : str
        Bathymetry filename (used only if ibot=1)
    ranges_filename : str
        Output ranges filename
    sedlayer : float, optional
        Sediment layer thickness (m). Default: 300.0
    nzs : int, optional
        Number of sediment depth control points. Default: 50.
    cs : ndarray, optional
        Sediment sound speed perturbation relative to water, shape (nzs,).
    rho : ndarray, optional
        Sediment density in g/cm^3, shape (nzs,).
    attn : ndarray, optional
        Sediment attenuation in dB/wavelength, shape (nzs,).
    isedrd : int, optional
        Range-dependent sediment flag. 0=range-independent (default),
        1=range-dependent from external file.
    sed_filename : str, optional
        Sediment profile filename (used when isedrd=1).
    """
    if cs is None:
        cs = np.zeros(nzs)
        cs[2:] = 200.0
    if rho is None:
        rho = np.full(nzs, 1.2)
    if attn is None:
        attn = np.full(nzs, 0.5)
        attn[-1] = 5.0

    with open(filepath, 'w') as f:
        f.write(f"0.0\n")  # dummy line (read and discarded by peramx.f90)
        f.write(f"{fc}  {Q}\n")
        f.write(f"{T}\n")
        f.write(f"{zsrc}\n")
        f.write(f"{deltaz}\n")
        f.write(f"{deltar}\n")
        f.write(f"{np_pade} {nss}\n")
        f.write(f"{rs}\n")
        f.write(f"{dzm}\n")
        f.write(f"{ssp_filename}\n")
        f.write(f"{iflat}\n")
        f.write(f"{ihorz}\n")
        f.write(f"{ibot}\n")
        f.write(f"{bth_filename}\n")
        f.write(f"{ranges_filename}\n")
        # Bottom properties
        f.write(f"{sedlayer}\n")
        f.write(f"{nzs}\n")
        f.write(f"{isedrd}\n")
        if isedrd == 1:
            # Range-dependent: write sediment filename
            f.write(f"{sed_filename}\n")
        else:
            # Range-independent: write nzs-element arrays
            f.write("  ".join(f"{v}" for v in cs) + "\n")
            f.write("  ".join(f"{v}" for v in rho) + "\n")
            f.write("  ".join(f"{v}" for v in attn) + "\n")


def write_sediment_file(
    filepath: Union[str, Path],
    ranges_km: np.ndarray,
    cs_profiles: np.ndarray,
    rho_profiles: np.ndarray,
    attn_profiles: np.ndarray,
):
    """
    Write range-dependent sediment profile file for mpiramS.

    Same format as SSP: each profile starts with ``-1 range_km``,
    followed by 3 lines of nzs values each (cs, rho, attn).

    Parameters
    ----------
    filepath : str or Path
        Output file path
    ranges_km : ndarray
        Range points in km, shape (N,)
    cs_profiles : ndarray
        Sound speed perturbation profiles, shape (nzs, N).
    rho_profiles : ndarray
        Density profiles, shape (nzs, N).
    attn_profiles : ndarray
        Attenuation profiles, shape (nzs, N).
    """
    ranges_km = np.asarray(ranges_km)
    n_profiles = len(ranges_km)

    with open(filepath, 'w') as f:
        for ip in range(n_profiles):
            f.write(f"-1 {ranges_km[ip]}\n")
            f.write("  ".join(f"{v}" for v in cs_profiles[:, ip]) + "\n")
            f.write("  ".join(f"{v}" for v in rho_profiles[:, ip]) + "\n")
            f.write("  ".join(f"{v}" for v in attn_profiles[:, ip]) + "\n")


def write_ssp_file(
    filepath: Union[str, Path],
    depths: np.ndarray,
    speeds: np.ndarray,
    ranges_km: Optional[np.ndarray] = None,
):
    """
    Write sound speed profile file for mpiramS.

    Format: Each profile starts with a header line ``-1 range_km``,
    followed by ``depth speed`` pairs. All profiles must have the
    same number of depth points.

    Parameters
    ----------
    filepath : str or Path
        Output file path
    depths : ndarray
        Depth points (m), shape (nz,)
    speeds : ndarray
        Sound speed values. Either 1D (nz,) for range-independent,
        or 2D (nz, n_profiles) for range-dependent.
    ranges_km : ndarray, optional
        Range of each profile in km. Required if speeds is 2D.
        If None and speeds is 1D, writes a single profile at range 0.
    """
    depths = np.asarray(depths)

    if speeds.ndim == 1:
        # Range-independent: single profile
        n_profiles = 1
        speeds = speeds.reshape(-1, 1)
        if ranges_km is None:
            ranges_km = np.array([0.0])
    else:
        n_profiles = speeds.shape[1]
        if ranges_km is None:
            raise ValueError("ranges_km required for range-dependent SSP")
        ranges_km = np.asarray(ranges_km)

    with open(filepath, 'w') as f:
        for ip in range(n_profiles):
            f.write(f"-1 {ranges_km[ip]}\n")
            for iz in range(len(depths)):
                f.write(f"{depths[iz]} {speeds[iz, ip]}\n")


def write_bth_file(
    filepath: Union[str, Path],
    ranges_m: np.ndarray,
    depths_m: np.ndarray,
):
    """
    Write bathymetry file for mpiramS.

    Format: ``range(m) depth(m)`` pairs, one per line.

    Parameters
    ----------
    filepath : str or Path
        Output file path
    ranges_m : ndarray
        Bathymetry range points (m)
    depths_m : ndarray
        Water depth at each range point (m)
    """
    ranges_m = np.asarray(ranges_m)
    depths_m = np.asarray(depths_m)
    with open(filepath, 'w') as f:
        for r, d in zip(ranges_m, depths_m):
            f.write(f"{r} {d}\n")


def write_ranges_file(
    filepath: Union[str, Path],
    ranges_m: np.ndarray,
):
    """
    Write output ranges file for mpiramS.

    One range per line, in meters.

    Parameters
    ----------
    filepath : str or Path
        Output file path
    ranges_m : ndarray
        Output ranges (m)
    """
    ranges_m = np.asarray(ranges_m)
    with open(filepath, 'w') as f:
        for r in ranges_m:
            f.write(f"{r}\n")
