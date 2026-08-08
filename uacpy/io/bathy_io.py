"""Bathymetry / altimetry file I/O.

Readers and writers for the geometric-boundary auxiliary files attached to
a Bellhop or Acoustics-Toolbox env description:

* ``.bty`` — bathymetry (:func:`read_bathymetry`, :func:`write_bty_file`,
  :func:`write_bty_long_format`, :func:`write_bty_3d`)
* ``.ati`` — altimetry (:func:`read_altimetry`, :func:`write_ati_file`)
* 3-D boundary blocks (:func:`read_boundary_3d`)

Reflection coefficients (`.brc`, `.irc`, `.trc`) and source beam patterns
(`.sbp`) live in :mod:`uacpy.io.refl_io`.
"""

import re

import numpy as np
from pathlib import Path
from typing import Tuple, Union

from uacpy._log import log_message
from uacpy.core.exceptions import (
    ConfigurationError, FileFormatError,
)
from uacpy.io.units import km_to_m, m_to_km
from uacpy.io._fortran_helpers import (
    read_vector, strip_fortran_comment, strip_fortran_quotes,
    typed_format_error,
)


def _summarize_axis(arr, head: int = 10, fmt: str = "{:9.5g}") -> str:
    """Compact one-line preview of a numeric axis for debug logging."""
    n = len(arr)
    if n <= head + 1:
        return "[" + ", ".join(fmt.format(v).strip() for v in arr) + f"] ({n} pts)"
    body = ", ".join(fmt.format(v).strip() for v in arr[:head])
    tail = fmt.format(arr[-1]).strip()
    return f"[{body}, …, {tail}] ({n} pts)"


@typed_format_error
def read_boundary_3d(
    filename: Union[str, Path], verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Read 3D boundary (bathymetry/altimetry) file for BELLHOP3D.

    BELLHOP3D uses 3D boundary files to specify depth-varying bathymetry
    and altimetry on a 2D horizontal grid (x, y, z(x,y)).

    Parameters
    ----------
    filename : str or Path
        Boundary filename (.bty for bathymetry, .ati for altimetry)
        Extension should be included
    verbose : bool, optional
        Print file contents to console (default: False)

    Returns
    -------
    x_bot : ndarray
        X-coordinates in metres, shape (n_x,). The file stores km;
        ``read_boundary_3d`` converts to metres to match the rest of
        the uacpy I/O surface.
    y_bot : ndarray
        Y-coordinates in metres, shape (n_y,). Same km→m conversion
        as ``x_bot``.
    z_bot : ndarray
        Z-coordinates (depth/height) in metres, shape (n_y, n_x).
        Note: transposed to match (y, x) grid convention.
    n_x : int
        Number of boundary points in x-direction.
    n_y : int
        Number of boundary points in y-direction.

    Notes
    -----
    File format (.bty or .ati), per Bellhop3D ``bdry3DMod.f90``:
    - Line 1: Boundary type ('R' or 'C')
      - 'R': Piecewise-linear (ruled) approximation
      - 'C': Curvilinear approximation
    - Lines 2-3: X-vector specification in km (uses read_vector format)
    - Lines 4-5: Y-vector specification in km (uses read_vector format)
    - Remaining lines: Z-values as n_y rows of n_x values, depths in metres
      positive downward.
    """
    with open(filename, "r") as fid:
        bdry_type_line = fid.readline().strip()

        match = re.search(r"'(.)'", bdry_type_line)
        if match:
            bdry_type = match.group(1)
        else:
            raise FileFormatError(f"Cannot parse boundary type from: {bdry_type_line}")

        if bdry_type == "R":
            log_message('bathy_io',
                        "Piecewise-linear approximation to boundary",
                        verbose=verbose)
        elif bdry_type == "C":
            log_message('bathy_io',
                        "Curvilinear approximation to boundary",
                        verbose=verbose)
        else:
            raise FileFormatError(f"Unknown boundary type: {bdry_type}")

        x_bot, n_x = read_vector(fid)

        log_message('bathy_io',
                    f"Number of boundary points in x = {n_x}",
                    verbose=verbose)
        log_message('bathy_io', f"x (km): {_summarize_axis(x_bot)}",
                    verbose=verbose, level='debug')

        y_bot, n_y = read_vector(fid)

        log_message('bathy_io',
                    f"Number of boundary points in y = {n_y}",
                    verbose=verbose)
        log_message('bathy_io', f"y (km): {_summarize_axis(y_bot)}",
                    verbose=verbose, level='debug')

        z_values = []
        for line in fid:
            values = [float(v) for v in line.split() if v]
            z_values.extend(values)

        # Bellhop3D writes the depth grid as ny rows of nx values
        # each (bdry3DMod.f90: DO iy = 1, NbtyPts(2); READ Bot(:, iy)).
        z_bot = np.array(z_values).reshape(n_y, n_x)

    return km_to_m(x_bot), km_to_m(y_bot), z_bot, n_x, n_y


def _read_boundary_2d(
    filepath: Union[str, Path], suffix: str, kind: str, verbose: bool,
) -> Tuple[np.ndarray, str]:
    """Read a BELLHOP 2-D boundary file (``.bty`` or ``.ati``).

    ``ReadATI`` and ``ReadBTY`` (``bdryMod.f90:61-110`` / ``:161-215``) are the
    same parser on two units, down to the long-format branch, so one reader
    serves both.
    """
    filepath = Path(filepath)
    if filepath.suffix != suffix:
        filepath = filepath.with_suffix(suffix)

    with open(filepath, "r") as fid:
        log_message('bathy_io', f"Reading {kind} file", verbose=verbose)

        type_str = strip_fortran_quotes(fid.readline()).upper()

        # TYPE is up to 2 chars: TYPE(1:1) sets the interpolation, TYPE(2:2)
        # selects short (geometry only) or long (geometry + geoacoustics).
        interp_type = type_str[:1]
        is_long = type_str[1:2] == "L"
        if interp_type not in ("L", "C"):
            raise FileFormatError(
                f"Unknown {kind} type: {interp_type} (must be 'L' or 'C')"
            )

        approx = "Piecewise-linear" if interp_type == "L" else "Curvilinear"
        log_message('bathy_io', f"{approx} approximation to {kind}",
                    verbose=verbose)

        n_pts = int(strip_fortran_comment(fid.readline()))
        log_message('bathy_io', f"Number of {kind} points = {n_pts}",
                    verbose=verbose)

        n_cols = 7 if is_long else 2
        rows = []
        for _ in range(n_pts):
            parts = fid.readline().split()
            if len(parts) < n_cols:
                raise FileFormatError(
                    f"{kind.capitalize()} file {filepath}: a '{type_str}' row "
                    f"needs {n_cols} columns, got {len(parts)} "
                    f"({' '.join(parts)!r})."
                )
            rows.append([float(v) for v in parts[:n_cols]])

        data = np.array(rows).T

        log_message('bathy_io', f"range (km): {_summarize_axis(data[0])}",
                    verbose=verbose, level='debug')
        log_message('bathy_io', f"depth (m): {_summarize_axis(data[1])}",
                    verbose=verbose, level='debug')

    data[0, :] = km_to_m(data[0, :])

    # Extend to ±infinity by holding every row constant.
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
        Path to bathymetry file (.bty extension).
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
        Path to altimetry file (.ati extension).
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


def _write_boundary_2d(
    filepath: Union[str, Path], data: np.ndarray, interp_type: str,
) -> None:
    """Emit the short (2-column) ATI/BTY record: TYPE, count, ``(range_km,
    value_m)`` rows.

    ``.ati`` and ``.bty`` are the same writer on two units — the counterpart
    of :func:`_read_boundary_2d`. ``data`` is an ``(N, 2)`` array of
    ``(range_m, value_m)`` or any carrier exposing ``to_pairs()``; ranges are
    converted to km for the file.
    """
    filepath = Path(filepath)
    type_str = f"{_validate_interp_type(interp_type)}S"

    if hasattr(data, "to_pairs"):
        data = data.to_pairs()
    rows = np.asarray(data, dtype=float).copy()
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
    File format identical to bathymetry (.bty) files. Per the
    Acoustics-Toolbox ATI/BTY specification, TYPE is a 2-character
    string — position 2 is always 'S' (short format) for altimetry
    since no geoacoustic parameters apply to the surface.

    Altimetry describes surface variations (ice keels, surface waves, etc.)
    """
    _write_boundary_2d(filepath, altimetry, interp_type)


def write_bty_3d(filepath: Union[str, Path], X: np.ndarray, Y: np.ndarray,
                 depth: np.ndarray, interp_type: str = "R") -> None:
    """
    Write 3D bathymetry file for BELLHOP3D.

    Parameters
    ----------
    filepath : str or Path
        Bathymetry file path (typically .bty extension).
    X : ndarray
        X coordinates in metres, shape (nx,). Converted to km on disk
        to match the Bellhop3D file format.
    Y : ndarray
        Y coordinates in metres, shape (ny,). Same m→km conversion as
        ``X``.
    depth : ndarray
        Depth values in metres, shape (ny, nx).
        ``depth[iy, ix]`` is depth at ``Y[iy], X[ix]``.
    interp_type : str, optional
        Interpolation type:
        - 'R': Piecewise-linear (default)
        - 'C': Curvilinear

    Notes
    -----
    File format for 3D bathymetry:
    - Line 1: Interpolation type in quotes
    - Line 2: nx (number of X points)
    - Line 3: X coordinates (km, space-separated)
    - Line 4: ny (number of Y points)
    - Line 5: Y coordinates (km, space-separated)
    - Following lines: Depth matrix (ny lines, nx values per line)

    NaN values in depth array are replaced with 0.0.

    The coordinate system uses:
    - X: Eastings (m) - horizontal coordinate
    - Y: Northings (m) - vertical coordinate
    - depth: Positive downward (m)

    See Also
    --------
    write_bty_file : Write 2D bathymetry
    read_boundary_3d : Read 3D bathymetry
    """
    filepath = Path(filepath)

    if interp_type not in ['R', 'C']:
        raise ConfigurationError(f"Unknown interpolation type: {interp_type}. Use 'R' or 'C'")

    depth = depth.copy()
    depth[np.isnan(depth)] = 0.0

    nx = len(X)
    ny = len(Y)

    if depth.shape != (ny, nx):
        raise ConfigurationError(f"Depth array shape {depth.shape} doesn't match (ny={ny}, nx={nx})")

    X_km = m_to_km(X)
    Y_km = m_to_km(Y)

    with open(filepath, 'w') as f:
        f.write(f"'{interp_type}'\n")

        f.write(f"{nx}\n")
        for x in X_km:
            f.write(f"{x:.6f} ")
        f.write("\n")

        f.write(f"{ny}\n")
        for y in Y_km:
            f.write(f"{y:.6f} ")
        f.write("\n")

        for iy in range(ny):
            for ix in range(nx):
                f.write(f"{depth[iy, ix]:9.3f} ")
            f.write("\n")
