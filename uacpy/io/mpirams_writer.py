"""
I/O writers for mpiramS (Fortran PE model)

Generates input files for the mpiramS binary:
- in.pe: main configuration file
- SSP file: sound speed profiles
- BTH file: bathymetry
- ranges file: output ranges

Every format here is fixed by ``mpiramS/src/peramx.f90``, which is uacpy's
own I/O rewrite of the vendored source (``third_party/MODIFICATIONS.md``):
it is authority on what this build reads and none at all on the physics.
The stock ``mpiramS/in.pe`` carries a different layout.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union

from uacpy.core.exceptions import ConfigurationError
from uacpy.io.units import m_to_km


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
    sedlayer: float = 300.0,
    nzs: int = 50,
    cs: Optional[np.ndarray] = None,
    rho: Optional[np.ndarray] = None,
    attn: Optional[np.ndarray] = None,
    isedrd: int = 0,
    sed_filename: str = '',
    *,
    c0_user: float,
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
    sedlayer : float, optional
        Sediment layer thickness (m). Default: 300.0
    nzs : int, optional
        Number of sediment depth control points. Default: 50. Must be at
        least 4: ``profl`` lays the points out as [surface, seafloor,
        nzs-3 interior, domain floor] (``mpiramS/src/ram.f90:334-342``).
    cs : ndarray, optional
        Sediment sound speed perturbation relative to water, shape (nzs,).
    rho : ndarray, optional
        Sediment density in g/cm^3, shape (nzs,).
    attn : ndarray, optional
        Sediment attenuation in dB/wavelength, shape (nzs,). Enters the
        sediment wavenumber as ``k = (omega/c)(1 + i*eta*attn)`` with
        ``eta = 1/(40*pi*log10(e))`` — the dB/wavelength-to-nepers factor
        (``mpiramS/src/ram.f90:289``, ``:361``).
    isedrd : int, optional
        Range-dependent sediment flag. 0=range-independent (default),
        1=range-dependent from external file.
    sed_filename : str, optional
        Sediment profile filename (used when isedrd=1).
    c0_user : float, keyword-only
        PE reference sound speed (m/s). The Fortran binary requires a
        positive value (e.g. Lytaev Eq. (15) optimum) and stops with
        an error otherwise.

    Notes
    -----
    ``in.pe`` is positional: ``mpiramS/src/peramx.f90:74-105`` consumes one
    record per line in exactly the order written here — a dummy line (``:74``),
    then ``fc Q``, ``T``, ``zsrc``, ``deltaz``, ``deltar``, ``np_pade nss``,
    ``rs``, ``dzm``, ``c0_user``, ``ssp_filename``, ``iflat``, ``ihorz``,
    ``ibot``, ``bth_filename``, the output-ranges filename (``:91``),
    ``sedlayer``, ``nzs``, ``isedrd`` (``:105``). ``isedrd = 1`` is followed by
    the sediment filename (``:109``); otherwise by three ``nzs``-value
    records — ``cs``, ``rho``, ``attn`` (``:151-153``).

    That order comes from uacpy's own I/O rewrite of ``peramx.f90`` (see the
    module docstring): authority on the file format, none on the physics.
    """
    # profl lays the sediment control points out as [surface, seafloor,
    # nzs-3 interior, domain floor] (mpiramS/src/ram.f90:334-342): fewer
    # than 4 points cannot express that layout, and the binary stops on
    # nzs < 4 (peramx.f90:101-104) because nzs=1 would write zwork(2)
    # past the end of a 1-element array.
    if int(nzs) < 4:
        raise ConfigurationError(
            f"write_inpe: nzs must be at least 4 (got {int(nzs)}). profl "
            f"builds the sub-bottom from [surface, seafloor, nzs-3 interior, "
            f"domain floor] control points (mpiramS/src/ram.f90:334-342), so "
            f"a deck with nzs < 4 cannot express the layout and is rejected "
            f"by the binary (peramx.f90:101-104)."
        )

    # Placeholder sediment column for a caller that supplies none — uacpy's
    # RAM wrapper always passes arrays built from env.bottom. ``profl`` puts
    # control point 1 at the sea surface, point 2 at the seafloor, points
    # 3..nzs-1 through the sediment and point nzs on the domain floor
    # (mpiramS/src/ram.f90:334-342 — uacpy's own rewrite of ``profl``, see
    # MODIFICATIONS.md, so it defines what this build expects and is no
    # authority on the physics). ``cs`` is added to the water profile at
    # :346, so this is water speed down to the seafloor and +200 m/s below
    # it, with a raised attenuation on the floor point that damps the
    # absorbing layer.
    if cs is None:
        cs = np.zeros(nzs)
        cs[2:] = 200.0
    if rho is None:
        rho = np.full(nzs, 1.2)
    if attn is None:
        attn = np.full(nzs, 0.5)
        attn[-1] = 5.0

    if isedrd != 1:
        for name, row in (('cs', cs), ('rho', rho), ('attn', attn)):
            n_row = len(np.atleast_1d(np.asarray(row)))
            if n_row != int(nzs):
                raise ConfigurationError(
                    f"write_inpe: nzs={int(nzs)} but the {name} row holds "
                    f"{n_row} value(s). peramx.f90:151-153 reads exactly "
                    f"nzs values per row, so a longer row is truncated "
                    f"without a diagnostic and a shorter one runs the "
                    f"read off the end of the deck."
                )

    with open(filepath, 'w') as f:
        # Dummy line, read and discarded at peramx.f90:74. It has to carry a
        # value: the list-directed read would skip a blank record and eat the
        # fc/Q line instead.
        f.write("0.0\n")
        f.write(f"{fc}  {Q}\n")
        f.write(f"{T}\n")
        f.write(f"{zsrc}\n")
        f.write(f"{deltaz}\n")
        f.write(f"{deltar}\n")
        f.write(f"{int(np_pade)} {int(nss)}\n")
        f.write(f"{rs}\n")
        f.write(f"{int(dzm)}\n")
        f.write(f"{c0_user}\n")
        f.write(f"{ssp_filename}\n")
        f.write(f"{int(iflat)}\n")
        f.write(f"{int(ihorz)}\n")
        f.write(f"{int(ibot)}\n")
        f.write(f"{bth_filename}\n")
        # peramx.f90:91-92 takes the output-ranges filename from this line,
        # so this writer is what pins it to 'ranges.dat'. Keep in sync with
        # :func:`write_ranges_file`.
        f.write("ranges.dat\n")
        # Bottom properties. ``peramx.f90:151-153`` reads the three sediment
        # rows as ``read (nunit,*) (cs(jj,1), jj=1,nzs)`` — a list-directed
        # read of exactly ``nzs`` values, which stops mid-record and discards
        # the rest. A row longer than ``nzs`` is therefore truncated in
        # silence, taking the absorbing-layer attenuation that lives at the
        # end of it with it: measured with nzs=5 against 10-element rows,
        # s_mpiram exits 0, echoes "Sediment speed (cs): 0.00 0.00 200.00
        # 200.00 200.00", and the water-column TL moves by a median 0.82 dB /
        # max 29.5 dB against the deck that honours all ten. The mirror case
        # (nzs larger than the rows) is loud — "End of file", exit 2 — so only
        # the truncating direction needs catching. Same guard as
        # :func:`write_bth_file`.
        f.write(f"{sedlayer}\n")
        f.write(f"{int(nzs)}\n")
        f.write(f"{int(isedrd)}\n")
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
    ranges_m: np.ndarray,
    cs_profiles: np.ndarray,
    rho_profiles: np.ndarray,
    attn_profiles: np.ndarray,
    nzs: Optional[int] = None,
):
    """
    Write range-dependent sediment profile file for mpiramS.

    Same format as SSP: each profile starts with ``-1 range_km``,
    followed by 3 lines of nzs values each (cs, rho, attn). The negative
    first column is the profile-header sentinel ``peramx.f90:129-130``
    counts on; ``:140`` multiplies the second column by 1000 to get metres.

    Parameters
    ----------
    filepath : str or Path
        Output file path
    ranges_m : ndarray
        Range points in metres, shape (N,). Converted to the km the
        ``.sed`` format expects at this boundary.
    cs_profiles : ndarray
        Sound speed perturbation profiles, shape (nzs, N).
    rho_profiles : ndarray
        Density profiles, shape (nzs, N).
    attn_profiles : ndarray
        Attenuation profiles, shape (nzs, N).
    nzs : int, optional
        The ``nzs`` written into the ``.inpe`` deck by :func:`write_inpe`.
        When given, every profile row is checked against it: ``peramx.f90:
        141-143`` reads each row as ``read (nunit,*) (cs(jj,1), jj=1,nzs)``, a
        list-directed read of exactly ``nzs`` values, so a longer row is
        truncated with no diagnostic. Omitting it trusts the caller to keep
        the two decks consistent, which is what ``models/ram.py`` does by
        building both from one plan.
    """
    ranges_km = m_to_km(ranges_m)
    n_profiles = len(ranges_km)

    if nzs is not None:
        for name, arr in (('cs', cs_profiles), ('rho', rho_profiles),
                          ('attn', attn_profiles)):
            n_row = int(np.asarray(arr).shape[0])
            if n_row != int(nzs):
                raise ConfigurationError(
                    f"write_sediment_file: the deck declares nzs={int(nzs)} "
                    f"but {name}_profiles holds {n_row} depth point(s). "
                    f"peramx.f90:141-143 reads exactly nzs values per row, so "
                    f"a longer row is truncated without a diagnostic."
                )

    # peramx.f90:128-131 counts profiles by testing the first token of every
    # record for < 0 (the "-1 range" header sentinel); a data row whose first
    # value is negative is miscounted as a header and the second read pass
    # aborts on EOF. cs is a signed offset (cp - surface water speed), so a
    # seabed slower than the surface water hits this format limitation.
    for name, arr in (('cs', cs_profiles), ('rho', rho_profiles),
                      ('attn', attn_profiles)):
        first_row = np.asarray(arr, dtype=float)[0, :]
        if np.any(first_row < 0):
            raise ConfigurationError(
                f"mpiramS sediment file: {name} profile(s) start with a "
                f"negative value (min {float(first_row.min()):g}), which the "
                f"binary's profile counter reads as a '-1 range' header "
                f"sentinel (peramx.f90:128-131) — the deck cannot express "
                f"it.",
                remediation="Use a Collins backend (RAM(backend='ramgeo' or "
                            "'ramsurf')), which writes sediment speeds "
                            "absolutely.",
            )

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
    ranges_m: Optional[np.ndarray] = None,
):
    """
    Write sound speed profile file for mpiramS.

    Format: Each profile starts with a header line ``-1 range_km``,
    followed by ``depth speed`` pairs. The negative first column is the
    profile-header sentinel ``peramx.f90:228-232`` counts on, and it takes
    the depth count from the first profile alone — hence all profiles must
    have the same number of depth points. ``:240`` multiplies the second
    column by 1000 to get metres.

    Parameters
    ----------
    filepath : str or Path
        Output file path
    depths : ndarray
        Depth points (m), shape (nz,). Must be non-negative — see below.
    speeds : ndarray
        Sound speed values. Either 1D (nz,) for range-independent,
        or 2D (nz, n_profiles) for range-dependent.
    ranges_m : ndarray, optional
        Range of each profile in metres, converted to the km the
        ``.ssp`` format expects at this boundary. Required if speeds
        is 2D, and must hold one range per profile. If None and speeds
        is 1D, writes a single profile at 0.
    """
    depths = np.asarray(depths)
    speeds = np.asarray(speeds)

    if speeds.ndim == 1:
        # Range-independent: single profile
        n_profiles = 1
        speeds = speeds.reshape(-1, 1)
        ranges_km = np.array([0.0])
    else:
        n_profiles = speeds.shape[1]
        if ranges_m is None:
            raise ConfigurationError("ranges_m required for range-dependent SSP")
        ranges_m = np.asarray(ranges_m)
        if ranges_m.size != n_profiles:
            raise ConfigurationError(
                f"write_ssp_file: ranges_m holds {ranges_m.size} range(s) but "
                f"speeds declares {n_profiles} profile(s); the two are written "
                f"as one header per profile, so a mismatch either indexes past "
                f"the range vector or silently drops the trailing profiles."
            )
        ranges_km = m_to_km(ranges_m)

    # peramx.f90:228-232 makes both counts from the sign of each record's first
    # token: a first token < 0 is the "-1 range" profile header, and only the
    # non-negative ones inside the first profile are counted as depths. A
    # negative depth is therefore counted as a profile AND missed as a depth,
    # so the second read pass walks records the file does not have.
    depth_values = np.asarray(depths, dtype=float)
    if np.any(depth_values < 0):
        raise ConfigurationError(
            f"mpiramS SSP file: {int(np.count_nonzero(depth_values < 0))} of "
            f"{depth_values.size} depth point(s) are negative (min "
            f"{float(depth_values.min()):g}), and the binary's counter reads a "
            f"negative first column as a '-1 range' profile-header sentinel "
            f"(peramx.f90:228-232) — inflating the profile count and deflating "
            f"the depth count. The deck cannot express it.",
            remediation="Give depths as non-negative metres below the surface.",
        )

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

    Format: ``range(m) depth(m)`` pairs, one per line. Metres, not the km
    the ``.ssp`` / ``.sed`` range headers use — ``peramx.f90:323`` reads
    this file with no scaling and marches ``rb`` against ``r`` in metres.

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
    if ranges_m.shape != depths_m.shape:
        raise ConfigurationError(
            f"write_bth_file: ranges_m ({ranges_m.size}) and depths_m "
            f"({depths_m.size}) must have the same length; a mismatch would "
            f"silently truncate the bathymetry."
        )
    with open(filepath, 'w') as f:
        for r, d in zip(ranges_m, depths_m):
            f.write(f"{r} {d}\n")


def write_ranges_file(
    filepath: Union[str, Path],
    ranges_m: np.ndarray,
):
    """
    Write output ranges file for mpiramS.

    One range per line, in metres (``peramx.f90:168-170`` reads them
    unscaled). There is no count header: ``:160-164`` reads to EOF to size
    the array first, then rewinds and fills it.

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
