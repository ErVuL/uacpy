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
from uacpy.core.units import m_to_km


def write_inpe(
    filepath: Union[str, Path],
    fc: float,
    q_factor: float,
    record_duration: float,
    zsrc: float,
    dz: float,
    dr: float,
    n_pade: int,
    n_stability: int,
    stability_range_m: float,
    depth_decimation: int,
    ssp_filename: str,
    earth_curvature: bool,
    horizontal_interpolation: bool,
    bathymetry_from_file: bool,
    bth_filename: str,
    sedlayer: float = 300.0,
    n_sediment_points: int = 50,
    cs: Optional[np.ndarray] = None,
    rho: Optional[np.ndarray] = None,
    attn: Optional[np.ndarray] = None,
    range_dependent_sediment: bool = False,
    sed_filename: str = '',
    *,
    c0: float,
    water_attn_filename: str = '',
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
    q_factor : float
        Q value (bandwidth = fc/Q). Use large Q (e.g. 1e6) for narrowband/TL mode.
    record_duration : float
        Time window width (s)
    zsrc : float
        Source depth (m)
    dz : float
        Depth accuracy parameter (m). Typical: 0.5
    dr : float
        Range accuracy parameter (m). Typical: 250.0
    n_pade : int
        Number of Pade coefficients (2-10, typical: 4)
    n_stability : int
        Number of stability terms (0 for short range, 1-2 for long range)
    stability_range_m : float
        Stability range (m)
    depth_decimation : int
        Output depth decimation factor
    ssp_filename : str
        Sound speed profile filename
    earth_curvature : bool
        Apply the flat-earth transform (peramx.f90:86, :268, :329, :444); written 0/1.
    horizontal_interpolation : bool
        Interpolate the profiles linearly in range (peramx.f90:87, :251); written 0/1.
    bathymetry_from_file : bool
        Read the bathymetry from ``bth_filename``; False uses a flat bottom 400 m above the deepest SSP depth (peramx.f90:88, :301-312); written 0/1.
    bth_filename : str
        Bathymetry filename (used only if bathymetry_from_file=1)
    sedlayer : float, optional
        Sediment layer thickness (m). Default: 300.0
    n_sediment_points : int, optional
        Number of sediment depth control points. Default: 50. Must be at
        least 4: ``profl`` lays the points out as [surface, seafloor,
        n_sediment_points-3 interior, domain floor] (``mpiramS/src/ram.f90:334-342``).
    cs : ndarray, optional
        Sediment sound speed perturbation relative to water, shape (n_sediment_points,).
    rho : ndarray, optional
        Sediment density relative to the water column (dimensionless),
        shape (n_sediment_points,).
    attn : ndarray, optional
        Sediment attenuation in dB/wavelength, shape (n_sediment_points,). Enters the
        sediment wavenumber as ``k = (omega/c)(1 + i*eta*attn)`` with
        ``eta = 1/(40*pi*log10(e))`` — the dB/wavelength-to-nepers factor
        (``mpiramS/src/ram.f90:289``, ``:361``).
    range_dependent_sediment : bool, optional
        Read the sediment from ``sed_filename`` instead of the three rows (peramx.f90:105-107); written 0/1.
    sed_filename : str, optional
        Sediment profile filename (used when range_dependent_sediment=1).
    c0 : float, keyword-only
        PE reference sound speed (m/s). The Fortran binary requires a
        positive value (e.g. Lytaev Eq. (15) optimum) and stops with
        an error otherwise.
    water_attn_filename : str, keyword-only, optional
        Name of a water-column attenuation table written by
        :func:`write_water_attenuation_file`. When given it is the deck's
        last line, after the sediment block; the uacpy-patched binary
        reads it and applies the table to the water wavenumber
        (``read_wattn`` in ``peramx.f90``, ``wksqw`` in ``ram.f90``).
        Empty (default) ends the deck with the sediment block, which the
        binary reads as a lossless water column.

    Notes
    -----
    ``in.pe`` is positional: ``mpiramS/src/peramx.f90:74-105`` consumes one
    record per line in exactly the order written here — a dummy line (``:74``),
    then ``fc Q``, ``record_duration``, ``zsrc``, ``dz``, ``dr``, ``np_pade nss``,
    ``stability_range_m``, ``depth_decimation``, ``c0``, ``ssp_filename``, ``earth_curvature``, ``horizontal_interpolation``,
    ``bathymetry_from_file``, ``bth_filename``, the output-ranges filename (``:91``),
    ``sedlayer``, ``n_sediment_points``, ``range_dependent_sediment`` (``:105``). ``range_dependent_sediment = 1`` is followed by
    the sediment filename (``:109``); otherwise by three ``n_sediment_points``-value
    records — ``cs``, ``rho``, ``attn`` (``:151-153``).

    That order comes from uacpy's own I/O rewrite of ``peramx.f90`` (see the
    module docstring): authority on the file format, none on the physics.
    """
    # profl lays the sediment control points out as [surface, seafloor,
    # n_sediment_points-3 interior, domain floor] (mpiramS/src/ram.f90:334-342): fewer
    # than 4 points cannot express that layout, and the binary stops on
    # n_sediment_points < 4 (peramx.f90:101-104) because n_sediment_points=1 would write zwork(2)
    # past the end of a 1-element array.
    if int(n_sediment_points) < 4:
        raise ConfigurationError(
            f"write_inpe: n_sediment_points must be at least 4 (got {int(n_sediment_points)}). profl "
            f"builds the sub-bottom from [surface, seafloor, n_sediment_points-3 interior, "
            f"domain floor] control points (mpiramS/src/ram.f90:334-342), so "
            f"a deck with n_sediment_points < 4 cannot express the layout and is rejected "
            f"by the binary (peramx.f90:101-104)."
        )

    # Placeholder sediment column for a caller that supplies none — uacpy's
    # RAM wrapper always passes arrays built from env.bottom. ``profl`` puts
    # control point 1 at the sea surface, point 2 at the seafloor, points
    # 3..n_sediment_points-1 through the sediment and point n_sediment_points on the domain floor
    # (mpiramS/src/ram.f90:334-342 — uacpy's own rewrite of ``profl``, see
    # MODIFICATIONS.md, so it defines what this build expects and is no
    # authority on the physics). ``cs`` is added to the water profile at
    # :346, so this is water speed down to the seafloor and +200 m/s below
    # it, with a raised attenuation on the floor point that damps the
    # absorbing layer.
    if cs is None:
        cs = np.zeros(n_sediment_points)
        cs[2:] = 200.0
    if rho is None:
        rho = np.full(n_sediment_points, 1.2)
    if attn is None:
        attn = np.full(n_sediment_points, 0.5)
        attn[-1] = 5.0

    if range_dependent_sediment != 1:
        for name, row in (('cs', cs), ('rho', rho), ('attn', attn)):
            n_row = len(np.atleast_1d(np.asarray(row)))
            if n_row != int(n_sediment_points):
                raise ConfigurationError(
                    f"write_inpe: n_sediment_points={int(n_sediment_points)} but the {name} row holds "
                    f"{n_row} value(s). peramx.f90:151-153 reads exactly "
                    f"n_sediment_points values per row, so a longer row is truncated "
                    f"without a diagnostic and a shorter one runs the "
                    f"read off the end of the deck."
                )

    with open(filepath, 'w') as f:
        # Dummy line, read and discarded at peramx.f90:74. It has to carry a
        # value: the list-directed read would skip a blank record and eat the
        # fc/Q line instead.
        f.write("0.0\n")
        f.write(f"{fc}  {q_factor}\n")
        f.write(f"{record_duration}\n")
        f.write(f"{zsrc}\n")
        f.write(f"{dz}\n")
        f.write(f"{dr}\n")
        f.write(f"{int(n_pade)} {int(n_stability)}\n")
        f.write(f"{stability_range_m}\n")
        f.write(f"{int(depth_decimation)}\n")
        f.write(f"{c0}\n")
        f.write(f"{ssp_filename}\n")
        f.write(f"{int(earth_curvature)}\n")
        f.write(f"{int(horizontal_interpolation)}\n")
        f.write(f"{int(bathymetry_from_file)}\n")
        f.write(f"{bth_filename}\n")
        # peramx.f90:91-92 takes the output-ranges filename from this line,
        # so this writer is what pins it to 'ranges.dat'. Keep in sync with
        # :func:`write_ranges_file`.
        f.write("ranges.dat\n")
        # Bottom properties. ``peramx.f90:151-153`` reads the three sediment
        # rows as ``read (nunit,*) (cs(jj,1), jj=1,nzs)`` — a list-directed
        # read of exactly ``n_sediment_points`` values, which stops mid-record and discards
        # the rest. A row longer than ``n_sediment_points`` is therefore truncated in
        # silence, taking the absorbing-layer attenuation that lives at the
        # end of it with it: measured with n_sediment_points=5 against 10-element rows,
        # s_mpiram exits 0, echoes "Sediment speed (cs): 0.00 0.00 200.00
        # 200.00 200.00", and the water-column TL moves by a median 0.82 dB /
        # max 29.5 dB against the deck that honours all ten. The mirror case
        # (n_sediment_points larger than the rows) is loud — "End of file", exit 2 — so only
        # the truncating direction needs catching. Same guard as
        # :func:`write_bth_file`.
        f.write(f"{sedlayer}\n")
        f.write(f"{int(n_sediment_points)}\n")
        f.write(f"{int(range_dependent_sediment)}\n")
        if range_dependent_sediment == 1:
            # Range-dependent: write sediment filename
            f.write(f"{sed_filename}\n")
        else:
            # Range-independent: write n_sediment_points-element arrays
            f.write("  ".join(f"{v}" for v in cs) + "\n")
            f.write("  ".join(f"{v}" for v in rho) + "\n")
            f.write("  ".join(f"{v}" for v in attn) + "\n")
        if water_attn_filename:
            f.write(f"{water_attn_filename}\n")


def write_water_attenuation_file(
    filepath: Union[str, Path],
    depths: np.ndarray,
    frequencies: np.ndarray,
    attenuation_dB_per_wavelength: np.ndarray,
) -> None:
    """Write the water-column attenuation table mpiramS reads.

    Layout (``read_wattn`` in ``peramx.f90``, list-directed): a ``nzaw nfaw``
    header, one row of the ``nfaw`` bin frequencies (Hz), then one row per
    depth — the depth (m) followed by its ``nfaw`` attenuations in
    dB/wavelength. The binary looks the marched bin up in the frequency row
    and stops if no entry sits within 1e-7 of it, so ``frequencies`` must be
    the sweep :func:`~uacpy.models.ram._band.broadband_frequencies` reproduces.
    Between depths the binary interpolates linearly and holds the end values
    beyond the table (``wksqw`` in ``ram.f90``).

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    depths : ndarray, shape (nz,)
        Increasing depths (m), in the frame the deck's other depths use.
    frequencies : ndarray, shape (nf,)
        The bin frequencies (Hz).
    attenuation_dB_per_wavelength : ndarray, shape (nz, nf)
        ``alpha(z, f)`` in dB per wavelength: one row per depth, one column
        per frequency. A ``(nz,)`` vector is accepted when ``nf == 1``.
        Any other shape — the transpose ``(nf, nz)`` included — raises
        rather than being reshaped.

    Raises
    ------
    ConfigurationError
        An empty table, a table whose shape is not ``(nz, nf)``, depths
        that do not increase strictly, or a negative / non-finite value.
    """
    z = np.atleast_1d(np.asarray(depths, dtype=float))
    f = np.atleast_1d(np.asarray(frequencies, dtype=float))
    a = np.asarray(attenuation_dB_per_wavelength, dtype=float)
    if z.size == 0 or f.size == 0:
        raise ConfigurationError(
            "write_water_attenuation_file: the table needs at least one depth "
            "and one frequency.")
    if f.size == 1 and a.shape == (z.size,):
        a = a[:, None]
    if a.shape != (z.size, f.size):
        raise ConfigurationError(
            f"write_water_attenuation_file: attenuation_dB_per_wavelength has shape "
            f"{a.shape}; the table is (n_depths, n_frequencies) = "
            f"({z.size}, {f.size}).",
            remediation="One row per depth, one column per frequency; "
                        "transpose an (n_frequencies, n_depths) array.")
    if np.any(np.diff(z) <= 0.0):
        raise ConfigurationError(
            "write_water_attenuation_file: depths must increase strictly; the "
            "binary walks the table assuming monotone depths.")
    if not np.all(np.isfinite(a)) or np.any(a < 0.0):
        raise ConfigurationError(
            "write_water_attenuation_file: attenuation must be finite and "
            "non-negative dB/wavelength.")
    with open(filepath, 'w') as fh:
        fh.write(f"{z.size} {f.size}\n")
        fh.write(" ".join(f"{v:.12g}" for v in f) + "\n")
        for zi, row in zip(z, a):
            fh.write(f"{zi:.12g} " + " ".join(f"{v:.12g}" for v in row) + "\n")


def write_sediment_file(
    filepath: Union[str, Path],
    ranges: np.ndarray,
    cs_profiles: np.ndarray,
    rho_profiles: np.ndarray,
    attn_profiles: np.ndarray,
    n_sediment_points: Optional[int] = None,
):
    """
    Write range-dependent sediment profile file for mpiramS.

    Same format as SSP: each profile starts with ``-1 range_km``,
    followed by 3 lines of n_sediment_points values each (cs, rho, attn). The negative
    first column is the profile-header sentinel ``peramx.f90:129-130``
    counts on; ``:140`` multiplies the second column by 1000 to get metres.

    Parameters
    ----------
    filepath : str or Path
        Output file path
    ranges : ndarray
        Range points in metres, shape (N,). Converted to the km the
        ``.sed`` format expects at this boundary.
    cs_profiles : ndarray
        Sound speed perturbation profiles, shape (n_sediment_points, N).
    rho_profiles : ndarray
        Density profiles, shape (n_sediment_points, N).
    attn_profiles : ndarray
        Attenuation profiles, shape (n_sediment_points, N).
    n_sediment_points : int, optional
        The ``n_sediment_points`` written into the ``.inpe`` deck by :func:`write_inpe`.
        When given, every profile row is checked against it: ``peramx.f90:
        141-143`` reads each row as ``read (nunit,*) (cs(jj,1), jj=1,nzs)``, a
        list-directed read of exactly ``n_sediment_points`` values, so a longer row is
        truncated with no diagnostic. Omitting it trusts the caller to keep
        the two decks consistent, which is what
        ``models/ram/mpirams.py`` does by building both from one plan.
    """
    ranges_km = m_to_km(ranges)
    n_profiles = len(ranges_km)

    if n_sediment_points is not None:
        for name, arr in (('cs', cs_profiles), ('rho', rho_profiles),
                          ('attn', attn_profiles)):
            n_row = int(np.asarray(arr).shape[0])
            if n_row != int(n_sediment_points):
                raise ConfigurationError(
                    f"write_sediment_file: the deck declares n_sediment_points={int(n_sediment_points)} "
                    f"but {name}_profiles holds {n_row} depth point(s). "
                    f"peramx.f90:141-143 reads exactly n_sediment_points values per row, so "
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
    sound_speed: np.ndarray,
    ranges: Optional[np.ndarray] = None,
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
    sound_speed : ndarray
        Sound speed values. Either 1D (nz,) for range-independent,
        or 2D (nz, n_profiles) for range-dependent.
    ranges : ndarray, optional
        Range of each profile in metres, converted to the km the
        ``.ssp`` format expects at this boundary. Required if speeds
        is 2D, and must hold one range per profile. If None and speeds
        is 1D, writes a single profile at 0.

    Raises
    ------
    ConfigurationError
        ``sound_speed`` does not hold one row per depth, the profile count and
        ``ranges`` disagree, or a depth is negative.
    """
    depths = np.asarray(depths)
    sound_speed = np.asarray(sound_speed)
    if sound_speed.ndim not in (1, 2) or sound_speed.shape[0] != depths.size:
        raise ConfigurationError(
            f"write_ssp_file: speeds has shape {sound_speed.shape} for "
            f"{depths.size} depth(s); it must be (n_depths,) or "
            f"(n_depths, n_profiles).",
            remediation="One row per depth; transpose an (n_profiles, "
                        "n_depths) array.")

    if sound_speed.ndim == 1:
        # Range-independent: single profile
        n_profiles = 1
        sound_speed = sound_speed.reshape(-1, 1)
        ranges_km = np.array([0.0])
    else:
        n_profiles = sound_speed.shape[1]
        if ranges is None:
            raise ConfigurationError("ranges_m required for range-dependent SSP.")
        ranges = np.asarray(ranges)
        if ranges.size != n_profiles:
            raise ConfigurationError(
                f"write_ssp_file: ranges_m holds {ranges.size} range(s) but "
                f"speeds declares {n_profiles} profile(s); the two are written "
                f"as one header per profile, so a mismatch either indexes past "
                f"the range vector or silently drops the trailing profiles."
            )
        ranges_km = m_to_km(ranges)

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
                f.write(f"{depths[iz]} {sound_speed[iz, ip]}\n")


def write_bth_file(
    filepath: Union[str, Path],
    ranges: np.ndarray,
    depths: np.ndarray,
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
    ranges : ndarray
        Bathymetry range points (m)
    depths : ndarray
        Water depth at each range point (m)
    """
    ranges = np.asarray(ranges)
    depths = np.asarray(depths)
    if ranges.shape != depths.shape:
        raise ConfigurationError(
            f"write_bth_file: ranges_m ({ranges.size}) and depths_m "
            f"({depths.size}) must have the same length; a mismatch would "
            f"silently truncate the bathymetry."
        )
    with open(filepath, 'w') as f:
        for r, d in zip(ranges, depths):
            f.write(f"{r} {d}\n")


def write_ranges_file(
    filepath: Union[str, Path],
    ranges: np.ndarray,
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
    ranges : ndarray
        Output ranges (m)
    """
    ranges = np.asarray(ranges)
    with open(filepath, 'w') as f:
        for r in ranges:
            f.write(f"{r}\n")
