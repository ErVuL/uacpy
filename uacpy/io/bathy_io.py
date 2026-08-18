"""Bathymetry / altimetry file I/O.

Readers and writers for the geometric-boundary auxiliary files attached to
a Bellhop or Acoustics-Toolbox env description:

* ``.bty`` — bathymetry (:func:`read_bathymetry`, :func:`write_bty_file`,
  :func:`write_bty_long_format`)
* ``.ati`` — altimetry (:func:`read_altimetry`, :func:`write_ati_file`)

Reflection coefficients (`.brc`, `.irc`, `.trc`) and source beam patterns
(`.sbp`) live in :mod:`uacpy.io.refl_io`.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Union

from uacpy._log import log_message
from uacpy.core.exceptions import (
    ConfigurationError, FileFormatError,
)
from uacpy.io.units import km_to_m, m_to_km
from uacpy.io._fortran_helpers import (
    list_directed_int, read_list_directed_values,
    strip_fortran_quotes, typed_format_error,
)


def _summarize_axis(arr, head: int = 10, fmt: str = "{:9.5g}") -> str:
    """Compact one-line preview of a numeric axis for debug logging."""
    n = len(arr)
    if n <= head + 1:
        return "[" + ", ".join(fmt.format(v).strip() for v in arr) + f"] ({n} pts)"
    body = ", ".join(fmt.format(v).strip() for v in arr[:head])
    tail = fmt.format(arr[-1]).strip()
    return f"[{body}, …, {tail}] ({n} pts)"


def _read_boundary_2d(
    filepath: Union[str, Path], suffix: str, kind: str, verbose: bool,
) -> Tuple[np.ndarray, str]:
    """Read a BELLHOP 2-D boundary file (``.bty`` or ``.ati``).

    ``ReadATI`` and ``ReadBTY`` (``bdryMod.f90:61-110`` / ``:161-215``) are the
    same parser on two units, down to the long-format branch, so one reader
    serves both.
    """
    # Use the path as given when it already exists (an explicit path is never
    # shadowed by a sibling .bty/.ati); the suffix is appended only as a
    # fallback so a bare base name still resolves to the conventional file.
    filepath = Path(filepath)
    if not filepath.exists() and not filepath.suffix:
        filepath = filepath.with_suffix(suffix)

    with open(filepath, "r") as fid:
        log_message('bathy_io', f"Reading {kind} file", verbose=verbose)

        type_str = strip_fortran_quotes(fid.readline())

        # TYPE is up to 2 chars: TYPE(1:1) sets the interpolation, TYPE(2:2)
        # selects short (geometry only) or long (geometry + geoacoustics).
        # The comparison is case-sensitive because the Fortran's is:
        # bdryMod.f90's SELECT CASE blocks test only 'C'/'L' (:162-165) and
        # 'S'/''/'L' (:193-199) and ERROUT on anything else, so a lowercase
        # letter that parsed here would still abort the binary.
        interp_type = type_str[:1]
        format_char = type_str[1:2].strip()
        if interp_type not in ("L", "C"):
            raise FileFormatError(
                f"Unknown {kind} type: {interp_type!r} (must be 'L' or 'C'; "
                f"the Fortran SELECT CASE is case-sensitive, bdryMod.f90:162-"
                f"165, so 'l'/'c' abort the engine with ERROUT)"
            )
        if format_char not in ("", "S", "L"):
            raise FileFormatError(
                f"Unknown {kind} format flag TYPE(2:2) = {format_char!r} "
                f"(must be 'S', 'L' or absent; case-sensitive in the Fortran, "
                f"bdryMod.f90:193-199)"
            )
        is_long = format_char == "L"

        approx = "Piecewise-linear" if interp_type == "L" else "Curvilinear"
        log_message('bathy_io', f"{approx} approximation to {kind}",
                    verbose=verbose)

        # The count is a list-directed scalar READ (bdryMod.f90:71 / :171), so
        # '3, ! npts' and '3   ! number of points' are both valid records.
        n_pts = list_directed_int(fid.readline())
        log_message('bathy_io', f"Number of {kind} points = {n_pts}",
                    verbose=verbose)

        # A long row is the 7 numbers of bdryMod.f90:200-201 — range, depth
        # and the five geoacoustic columns; a short row is range, depth.
        # Each point is ONE list-directed READ (bdryMod.f90:98 / :195), which
        # keeps consuming records until its n_cols values are read and then
        # skips the remainder of its final record — so a row may wrap across
        # any number of lines, and this reader accepts the same files the
        # binary does.
        n_cols = 7 if is_long else 2
        rows = [
            read_list_directed_values(
                fid, n_cols,
                f"{kind} point {i + 1} ({n_cols} columns for "
                f"TYPE {type_str!r})", filepath)
            for i in range(n_pts)
        ]

        data = np.array(rows).T

        log_message('bathy_io', f"range (km): {_summarize_axis(data[0])}",
                    verbose=verbose, level='debug')
        log_message('bathy_io', f"depth (m): {_summarize_axis(data[1])}",
                    verbose=verbose, level='debug')

    data[0, :] = km_to_m(data[0, :])

    # Extend to ±infinity by holding every row constant. The ±1e50 sentinel is
    # the one AT's own reader uses — "extend the bathymetry to +/- infinity in
    # a piecewise constant fashion", Matlab/ReadWrite/readbty.m:66-71 — so a
    # segment search over this array brackets any range without a special case.
    out = np.zeros((n_cols, n_pts + 2))
    out[:, 1:-1] = data
    out[:, 0] = data[:, 0]
    out[:, -1] = data[:, -1]
    out[0, 0] = -1e50
    out[0, -1] = 1e50

    return out, interp_type


@typed_format_error
def read_bathymetry(filepath: Union[str, Path], verbose: bool = False) -> Tuple[np.ndarray, str]:
    """
    Read bathymetry data from BELLHOP .bty file.

    Reads 2D range-depth bathymetry profile with optional interpolation type.
    Extends the bathymetry to ±infinity for computational purposes.

    Parameters
    ----------
    filepath : str or Path
        Path to bathymetry file. An existing path is read exactly as
        given, whatever its extension; a bare root without an extension
        resolves to ``<root>.bty``.
    verbose : bool, optional
        If True, print bathymetry information. Default is False.

    Returns
    -------
    bty : ndarray
        Bathymetry data array of shape (n_cols, N+2) where:
        - bty[0, :] = range in meters (extended to ±1e50 at endpoints)
        - bty[1, :] = depth in meters

        A long-format file (``TYPE(2:2) == 'L'``, written by
        :func:`write_bty_long_format`) carries per-range geoacoustics in the
        remaining rows, in ``bdryMod.f90:200-201`` column order:
        - bty[2, :] = compressional speed (m/s)
        - bty[3, :] = shear speed (m/s)
        - bty[4, :] = density (g/cm³)
        - bty[5, :] = compressional attenuation
        - bty[6, :] = shear attenuation

        First and last points are extended to -infinity and +infinity, every
        row held constant across the extension.
    bty_type : str
        Interpolation type:
        - 'L' : Piecewise-linear
        - 'C' : Curvilinear (cubic spline)

    Notes
    -----
    - Input file ranges are in km, converted to meters on output
    - Bathymetry is extended to ±infinity using constant extrapolation
    - File format:
        Line 1: TYPE in quotes — position 1 is 'L' or 'C' (interpolation),
        position 2 is 'S' (short) or 'L' (long, with geoacoustics)
        Line 2: Number of points
        Lines 3+: range(km) depth(m) [cp cs rho alpha_p alpha_s]

    References
    ----------
    Based on BELLHOP/readbty.m
    """
    return _read_boundary_2d(filepath, ".bty", "bathymetry", verbose)


@typed_format_error
def read_altimetry(filepath: Union[str, Path], verbose: bool = False) -> Tuple[np.ndarray, str]:
    """
    Read altimetry data from BELLHOP .ati file.

    Reads 2D range-depth altimetry (surface) profile with optional
    interpolation type. Extends the altimetry to ±infinity.

    Parameters
    ----------
    filepath : str or Path
        Path to altimetry file. An existing path is read exactly as
        given, whatever its extension; a bare root without an extension
        resolves to ``<root>.ati``.
    verbose : bool, optional
        If True, print altimetry information. Default is False.

    Returns
    -------
    ati : ndarray
        Altimetry data array of shape (n_cols, N+2) where:
        - ati[0, :] = range in meters (extended to ±1e50 at endpoints)
        - ati[1, :] = depth in meters

        ``ReadATI`` accepts the same long format as ``ReadBTY``
        (``bdryMod.f90:80-110``), so a ``TYPE(2:2) == 'L'`` file carries
        per-range top geoacoustics — an ice cover, typically — in rows 2..6
        with the column order documented on :func:`read_bathymetry`.
    ati_type : str
        Interpolation type:
        - 'L' : Piecewise-linear
        - 'C' : Curvilinear (cubic spline)

    Notes
    -----
    - Input file ranges are in km, converted to meters on output
    - Altimetry is extended to ±infinity using constant extrapolation
    - File format identical to bathymetry (.bty) files

    References
    ----------
    Based on BELLHOP/readati.m
    """
    return _read_boundary_2d(filepath, ".ati", "altimetry", verbose)


def _validate_interp_type(interp_type: str) -> str:
    """Return a single-character interpolation code, or raise
    :class:`~uacpy.core.exceptions.ConfigurationError`.

    Only the first character of TYPE ('L' piecewise linear or 'C'
    curvilinear) is user-selectable; the second character (format flag)
    is chosen by the writer depending on the number of columns it emits
    (see ``write_bty_file`` vs ``write_bty_long_format``).
    """
    t = str(interp_type).strip().upper()
    if t not in ("L", "C"):
        raise ConfigurationError(
            f"Invalid interpolation type {interp_type!r}; expected 'L' or 'C'."
        )
    return t


def _validate_boundary_axis(label: str, axis: np.ndarray) -> None:
    """Reject a boundary axis the engines cannot interpolate on.

    Bellhop brackets a coordinate by searching the axis in ascending order
    and interpolates within the bracketing segment
    (``bdryMod.f90 GetTopSeg/GetBotSeg``, :256-282); a repeated value makes
    a zero-length segment whose tangent normalisation divides by zero, and
    a decreasing value breaks the segment search silently. Non-finite
    values pass straight into the geometry and poison every tangent/normal
    derived from it.
    """
    axis = np.asarray(axis, dtype=float)
    if not np.all(np.isfinite(axis)):
        raise ConfigurationError(
            f"{label}: contains non-finite values "
            f"({np.count_nonzero(~np.isfinite(axis))} of {axis.size}); the "
            f"engines interpolate the boundary from these coordinates and a "
            f"NaN/Inf poisons the segment tangents."
        )
    if axis.size > 1 and np.diff(axis).min() <= 0.0:
        i = int(np.argmin(np.diff(axis)))
        raise ConfigurationError(
            f"{label}: values must be strictly increasing, but "
            f"index {i} -> {i + 1} goes {axis[i]:g} -> {axis[i + 1]:g}. The "
            f"engines' segment search assumes an ascending axis and a "
            f"repeated value creates a zero-length segment.",
            remediation="Sort the axis and remove duplicate coordinates "
                        "before writing.",
        )


def _write_boundary_2d(
    filepath: Union[str, Path], data: np.ndarray, interp_type: str,
) -> None:
    """Emit the short (2-column) ATI/BTY record: TYPE, count, ``(range_km,
    value_m)`` rows.

    ``.ati`` and ``.bty`` are the same writer on two units — the counterpart
    of :func:`_read_boundary_2d`. ``data`` is an ``(N, 2)`` array of
    ``(range_m, value_m)`` or any carrier exposing ``to_pairs()``; ranges are
    converted to km for the file.

    ``bdryMod.f90:98`` (.ati) and ``:195`` (.bty) read the rows list-directed,
    so the field widths do not matter; the 6 decimals do — on a km axis they
    are what keeps a metre-domain range exact to the millimetre.
    """
    filepath = Path(filepath)
    type_str = f"{_validate_interp_type(interp_type)}S"

    if hasattr(data, "to_pairs"):
        data = data.to_pairs()
    rows = np.asarray(data, dtype=float).copy()
    _validate_boundary_axis(f"{filepath.suffix or '.bty/.ati'} range column",
                            rows[:, 0])
    rows[:, 0] = m_to_km(rows[:, 0])

    with open(filepath, "w") as f:
        f.write(f"'{type_str}'\n")
        f.write(f"{rows.shape[0]}\n")
        for r, value in rows:
            f.write(f"{r:.6f} {value:.6f}\n")
        f.write("\n")


def write_bty_file(filepath: Union[str, Path], bathymetry: np.ndarray, interp_type: str = "L") -> None:
    """
    Write bathymetry file (short format, bathymetry only).

    Parameters
    ----------
    filepath : str or Path
        Bathymetry file path (typically .bty extension)
    bathymetry : ndarray
        Bathymetry data, shape (N, 2):
        - Column 0: Range in meters
        - Column 1: Depth in meters
    interp_type : str, optional
        Interpolation type (single character, TYPE(1:1)):
        - 'C': Curvilinear (with tangents/normals)
        - 'L': Linear interpolation (default)

    Notes
    -----
    File format (see third_party/Acoustics-Toolbox/doc/ATI_BTY_File.htm):
    - Line 1: 2-character TYPE in quotes — position 1 is interpolation
      ('L' or 'C'), position 2 is format ('S' short, bathymetry only)
    - Line 2: Number of points
    - Following lines: range (km), depth (m) pairs

    This writer always emits the short (2-column) format, so TYPE(2:2)
    is hardcoded to 'S'.

    The function automatically converts ranges from meters to kilometers
    for the output file.
    """
    _write_boundary_2d(filepath, bathymetry, interp_type)


def write_bty_long_format(
    filepath: Union[str, Path],
    bathymetry: np.ndarray,
    bottom_rd,
    interp_type: str = "L",
) -> None:
    """
    Write long-format bathymetry file (.bty) with per-segment geoacoustics.

    Unlike the 2-column ``write_bty_file`` output, the long format adds
    bottom compressional sound speed, density, attenuation and shear speed
    per range so Bellhop can use range-dependent bottom properties
    (``BOTY(:)`` handling in ReadEnvironmentBell.f90).

    Parameters
    ----------
    filepath : str or Path
        Output .bty path.
    bathymetry : ndarray
        Shape (N, 2): range (m), depth (m).
    bottom_rd : Bottom
        Range-dependent ``Bottom`` carrying per-range halfspace geoacoustics:
        ``ranges`` (metres) plus the ``halfspace_sound_speed`` /
        ``halfspace_density`` / ``halfspace_attenuation`` /
        ``halfspace_shear_speed`` / ``halfspace_shear_attenuation`` views.
    interp_type : str, optional
        'L' (linear, default) or 'C' (curvilinear).

    Notes
    -----
    File format (extended BTY — long format), matching the AT Fortran
    READ in ``Bellhop/bdryMod.f90:200-201``:
    ``READ(BTYFile,*) Bot(ii)%x, %HS%alphaR, %HS%betaR, %HS%rho,
    %HS%alphaI, %HS%betaI``  — i.e. 7 numbers per row.

    - Line 1: 2-character TYPE in quotes — position 1 is interpolation
      ('L' or 'C'), position 2 is 'L' (long format, bathymetry +
      geoacoustics). See ATI_BTY_File.htm.
    - Line 2: number of points
    - Following lines:
      ``range_km depth cp_m_s cs_m_s rho_g_cm3 alpha_p alpha_s``

    Rows are emitted on the **union** of the bathymetry and bottom range
    grids (depth and geoacoustics each interpolated onto it), so property
    breaks between bathymetry points survive rather than being blended away.
    """
    filepath = Path(filepath)
    interp_char = _validate_interp_type(interp_type)
    type_str = f"{interp_char}L"

    if hasattr(bathymetry, "to_pairs"):
        bathymetry = bathymetry.to_pairs()
    bathy_km = np.asarray(bathymetry, dtype=float).copy()
    _validate_boundary_axis(".bty (long) range column", bathy_km[:, 0])
    bathy_km[:, 0] = m_to_km(bathy_km[:, 0])

    rd_r_km = m_to_km(bottom_rd.ranges)
    r_km = np.union1d(bathy_km[:, 0], np.asarray(rd_r_km, dtype=float))
    depth = np.interp(r_km, bathy_km[:, 0], bathy_km[:, 1])
    cp = np.interp(r_km, rd_r_km, bottom_rd.halfspace_sound_speed)
    rho = np.interp(r_km, rd_r_km, bottom_rd.halfspace_density)
    alpha = np.interp(r_km, rd_r_km, bottom_rd.halfspace_attenuation)
    cs = np.interp(r_km, rd_r_km, bottom_rd.halfspace_shear_speed)
    alpha_s = np.interp(r_km, rd_r_km,
                        bottom_rd.halfspace_shear_attenuation)
    n_pts = r_km.size

    with open(filepath, "w") as f:
        f.write(f"'{type_str}'\n")
        f.write(f"{n_pts}\n")
        for i in range(n_pts):
            # Column order matches bdryMod.f90:200-201 (range_km depth cp cs rho alpha_p alpha_s).
            f.write(
                f"{r_km[i]:.6f} {depth[i]:.6f} "
                f"{cp[i]:.3f} {cs[i]:.3f} {rho[i]:.3f} "
                f"{alpha[i]:.6f} {alpha_s[i]:.6f}\n"
            )
        f.write("\n")


def write_ati_file(filepath: Union[str, Path], altimetry: np.ndarray, interp_type: str = "L") -> None:
    """
    Write altimetry (surface) file for acoustic models.

    Parameters
    ----------
    filepath : str or Path
        Altimetry file path (typically .ati extension)
    altimetry : ndarray
        Altimetry data, shape (N, 2):
        - Column 0: Range in meters
        - Column 1: Surface position in meters, **positive-down** in
          Bellhop's z-axis convention (i.e. +2 means the surface is 2 m
          *below* MSL — a wave trough). Wrappers that own the
          public positive-up convention (``Environment(altimetry=…)``)
          must negate column 1 before calling this writer.
    interp_type : str, optional
        Interpolation type (single character, TYPE(1:1)):
        - 'C': Curvilinear (with tangents/normals)
        - 'L': Linear interpolation (default)

    Notes
    -----
    File format identical to bathymetry (.bty) files. TYPE is a
    2-character string; this writer emits only the 2-column short format,
    so position 2 is hardcoded to 'S'. ``ReadATI`` itself also accepts
    the long format (``bdryMod.f90:80-110``) carrying per-range *top*
    geoacoustics — an ice cover — which this writer does not produce.

    Altimetry describes surface variations (ice keels, surface waves, etc.)
    """
    _write_boundary_2d(filepath, altimetry, interp_type)
