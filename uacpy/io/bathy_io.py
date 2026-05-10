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

import numpy as np
from pathlib import Path
from typing import Tuple, Union

from uacpy.core.exceptions import ModelExecutionError
from uacpy.io._fortran_helpers import read_vector


def read_boundary_3d(
    filename: str, verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Read 3D boundary (bathymetry/altimetry) file for BELLHOP3D.

    BELLHOP3D uses 3D boundary files to specify depth-varying bathymetry
    and altimetry on a 2D horizontal grid (x, y, z(x,y)).

    Parameters
    ----------
    filename : str
        Boundary filename (.bty for bathymetry, .ati for altimetry)
        Extension should be included
    verbose : bool, optional
        Print file contents to console (default: True)

    Returns
    -------
    x_bot : ndarray
        X-coordinates (typically in km), shape (n_x,)
    y_bot : ndarray
        Y-coordinates (typically in km), shape (n_y,)
    z_bot : ndarray
        Z-coordinates (depth/height), shape (n_y, n_x)
        Note: Transposed to match (y, x) grid convention
    n_x : int
        Number of boundary points in x-direction
    n_y : int
        Number of boundary points in y-direction

    Notes
    -----
    File format (.bty or .ati):
    - Line 1: Boundary type ('R' or 'C')
      - 'R': Piecewise-linear (ruled) approximation
      - 'C': Curvilinear approximation
    - Lines 2-3: X-vector specification (uses read_vector format)
    - Lines 4-5: Y-vector specification (uses read_vector format)
    - Remaining lines: Z-values in row-major order (n_x × n_y values)

    The Z-matrix is read as (n_x, n_y) and transposed to (n_y, n_x)
    to match standard grid indexing conventions.
    """
    try:
        with open(filename, "r") as fid:
            bdry_type_line = fid.readline().strip()

            import re

            match = re.search(r"'(.)'", bdry_type_line)
            if match:
                bdry_type = match.group(1)
            else:
                raise ValueError(f"Cannot parse boundary type from: {bdry_type_line}")

            if verbose:
                if bdry_type == "R":
                    print("Piecewise-linear approximation to boundary")
                elif bdry_type == "C":
                    print("Curvilinear approximation to boundary")
                else:
                    raise ValueError(f"Unknown boundary type: {bdry_type}")

            x_bot, n_x = read_vector(fid)

            if verbose:
                print(f"Number of boundary points in x = {n_x}\n")
                print(" x (km)")
                for i, x_val in enumerate(x_bot):
                    if i < 50 or i == n_x - 1:
                        print(f"{x_val:9.5g}")
                    elif i == 50:
                        print("   ...")

            y_bot, n_y = read_vector(fid)

            if verbose:
                print(f"Number of boundary points in y = {n_y}\n")
                print(" y (km)")
                for i, y_val in enumerate(y_bot):
                    if i < 50 or i == n_y - 1:
                        print(f"{y_val:9.5g}")
                    elif i == 50:
                        print("   ...")

            z_values = []
            for line in fid:
                values = [float(v) for v in line.split() if v]
                z_values.extend(values)

            z_bot = np.array(z_values).reshape(n_x, n_y).T

    except FileNotFoundError:
        raise FileNotFoundError(f"Boundary file not found: {filename}")
    except Exception as e:
        raise ModelExecutionError(
            'Bellhop3D', return_code=0, stdout=None,
            stderr=f"Error reading boundary file {filename}: {e}",
        ) from e

    return x_bot, y_bot, z_bot, n_x, n_y


def read_bathymetry(filepath: Union[str, Path], verbose: bool = True) -> Tuple[np.ndarray, str]:
    """
    Read bathymetry data from BELLHOP .bty file.

    Reads 2D range-depth bathymetry profile with optional interpolation type.
    Extends the bathymetry to ±infinity for computational purposes.

    Parameters
    ----------
    filepath : str or Path
        Path to bathymetry file (.bty extension).
    verbose : bool, optional
        If True, print bathymetry information. Default is True.

    Returns
    -------
    bty : ndarray
        Bathymetry data array of shape (2, N+2) where:
        - bty[0, :] = range in meters (extended to ±1e50 at endpoints)
        - bty[1, :] = depth in meters
        First and last points are extended to -infinity and +infinity.
    bty_type : str
        Interpolation type:
        - 'L' : Piecewise-linear
        - 'C' : Curvilinear (cubic spline)

    Notes
    -----
    - Input file ranges are in km, converted to meters on output
    - Bathymetry is extended to ±infinity using constant extrapolation
    - File format:
        Line 1: 'L' or 'C' (in quotes)
        Line 2: Number of points
        Lines 3+: range(km) depth(m) pairs

    References
    ----------
    Based on BELLHOP/readbty.m
    """
    filepath = Path(filepath)
    if filepath.suffix != ".bty":
        filepath = filepath.with_suffix(".bty")

    with open(filepath, "r") as fid:
        if verbose:
            print("\n_______________________")
            print("Using bottom-bathymetry file")

        line = fid.readline().strip()
        if "'" in line:
            start = line.index("'") + 1
            end = line.index("'", start)
            bty_type = line[start:end].strip()
        else:
            bty_type = line.strip()

        if bty_type not in ["L", "C"]:
            raise ValueError(
                f"Unknown bathymetry type: {bty_type} (must be 'L' or 'C')"
            )

        if verbose:
            if bty_type == "L":
                print("Piecewise-linear approximation to bathymetry")
            else:
                print("Curvilinear approximation to bathymetry")

        n_pts = int(fid.readline().strip())

        if verbose:
            print(f"Number of bathymetry points = {n_pts}\n")
            print(" Range (km)     Depth (m)")

        bty_data = []
        for i in range(n_pts):
            parts = fid.readline().split()
            range_km, depth = float(parts[0]), float(parts[1])
            bty_data.append([range_km, depth])

            if verbose and (i < 10 or i == n_pts - 1):
                print(f"{range_km:9.5g}    {depth:9.5g}")
            elif verbose and i == 10:
                print("    ...")

        bty_data = np.array(bty_data).T

    bty_data[0, :] *= 1000.0

    n_pts_extended = n_pts + 2
    bty = np.zeros((2, n_pts_extended))

    bty[0, 0] = -1e50
    bty[1, 0] = bty_data[1, 0]

    bty[:, 1:-1] = bty_data

    bty[0, -1] = 1e50
    bty[1, -1] = bty_data[1, -1]

    return bty, bty_type


def read_altimetry(filepath: Union[str, Path], verbose: bool = True) -> Tuple[np.ndarray, str]:
    """
    Read altimetry data from BELLHOP .ati file.

    Reads 2D range-depth altimetry (surface) profile with optional
    interpolation type. Extends the altimetry to ±infinity.

    Parameters
    ----------
    filepath : str or Path
        Path to altimetry file (.ati extension).
    verbose : bool, optional
        If True, print altimetry information. Default is True.

    Returns
    -------
    ati : ndarray
        Altimetry data array of shape (2, N+2) where:
        - ati[0, :] = range in meters (extended to ±1e50 at endpoints)
        - ati[1, :] = depth in meters
        First and last points are extended to -infinity and +infinity.
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
    filepath = Path(filepath)
    if filepath.suffix != ".ati":
        filepath = filepath.with_suffix(".ati")

    with open(filepath, "r") as fid:
        if verbose:
            print("\n_______________________")
            print("Using top-altimetry file")

        line = fid.readline().strip()
        if "'" in line:
            start = line.index("'") + 1
            end = line.index("'", start)
            ati_type = line[start:end].strip()
        else:
            ati_type = line.strip()

        if ati_type not in ["L", "C"]:
            raise ValueError(f"Unknown altimetry type: {ati_type} (must be 'L' or 'C')")

        if verbose:
            if ati_type == "L":
                print("Piecewise-linear approximation to altimetry")
            else:
                print("Curvilinear approximation to altimetry")

        n_pts = int(fid.readline().strip())

        if verbose:
            print(f"Number of altimetry points = {n_pts}\n")
            print(" Range (km)     Depth (m)")

        ati_data = []
        for i in range(n_pts):
            parts = fid.readline().split()
            range_km, depth = float(parts[0]), float(parts[1])
            ati_data.append([range_km, depth])

            if verbose and (i < 10 or i == n_pts - 1):
                print(f"{range_km:9.5g}    {depth:9.5g}")
            elif verbose and i == 10:
                print("    ...")

        ati_data = np.array(ati_data).T

    ati_data[0, :] *= 1000.0

    n_pts_extended = n_pts + 2
    ati = np.zeros((2, n_pts_extended))

    ati[0, 0] = -1e50
    ati[1, 0] = ati_data[1, 0]

    ati[:, 1:-1] = ati_data

    ati[0, -1] = 1e50
    ati[1, -1] = ati_data[1, -1]

    return ati, ati_type


def _validate_interp_type(interp_type: str) -> str:
    """Return a single-character interpolation code or raise ValueError.

    Only the first character of TYPE ('L' piecewise linear or 'C'
    curvilinear) is user-selectable; the second character (format flag)
    is chosen by the writer depending on the number of columns it emits
    (see ``write_bty_file`` vs ``write_bty_long_format``).
    """
    t = str(interp_type).strip().upper()
    if t not in ("L", "C"):
        raise ValueError(
            f"Invalid interpolation type {interp_type!r}; expected 'L' or 'C'."
        )
    return t


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
    filepath = Path(filepath)
    interp_char = _validate_interp_type(interp_type)
    type_str = f"{interp_char}S"

    bathy_km = bathymetry.copy()
    bathy_km[:, 0] = bathy_km[:, 0] / 1000.0

    n_pts = bathy_km.shape[0]

    with open(filepath, "w") as f:
        f.write(f"'{type_str}'\n")
        f.write(f"{n_pts}\n")
        for i in range(n_pts):
            f.write(f"{bathy_km[i, 0]:.6f} {bathy_km[i, 1]:.6f}\n")
        f.write("\n")


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
    bottom_rd : RangeDependentBottom
        Object carrying per-range geoacoustics with attributes
        ``ranges`` (metres), ``sound_speed``, ``density``, ``attenuation``,
        ``shear_speed``, ``shear_attenuation``.
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

    Ranges in ``bottom_rd.ranges`` (metres) are re-sampled onto the
    bathymetry range grid via ``numpy.interp`` so the two lengths match.
    """
    filepath = Path(filepath)
    interp_char = _validate_interp_type(interp_type)
    type_str = f"{interp_char}L"

    bathy_km = bathymetry.copy()
    bathy_km[:, 0] = bathy_km[:, 0] / 1000.0
    n_pts = bathy_km.shape[0]

    rd_r_km = np.asarray(bottom_rd.ranges, dtype=float) / 1000.0
    cp = np.interp(bathy_km[:, 0], rd_r_km, bottom_rd.sound_speed)
    rho = np.interp(bathy_km[:, 0], rd_r_km, bottom_rd.density)
    alpha = np.interp(bathy_km[:, 0], rd_r_km, bottom_rd.attenuation)
    cs_arr = bottom_rd.shear_speed if bottom_rd.shear_speed is not None \
        else np.zeros_like(rd_r_km)
    cs = np.interp(bathy_km[:, 0], rd_r_km, cs_arr)
    alpha_s_arr = getattr(bottom_rd, 'shear_attenuation', None)
    if alpha_s_arr is None:
        alpha_s_arr = np.zeros_like(rd_r_km)
    alpha_s = np.interp(bathy_km[:, 0], rd_r_km, alpha_s_arr)

    with open(filepath, "w") as f:
        f.write(f"'{type_str}'\n")
        f.write(f"{n_pts}\n")
        for i in range(n_pts):
            # Column order matches bdryMod.f90:200-201 (range_km depth cp cs rho alpha_p alpha_s).
            f.write(
                f"{bathy_km[i, 0]:.6f} {bathy_km[i, 1]:.6f} "
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
    filepath = Path(filepath)
    interp_char = _validate_interp_type(interp_type)
    type_str = f"{interp_char}S"

    alti_km = altimetry.copy()
    alti_km[:, 0] = alti_km[:, 0] / 1000.0

    n_pts = alti_km.shape[0]

    with open(filepath, "w") as f:
        f.write(f"'{type_str}'\n")
        f.write(f"{n_pts}\n")
        for i in range(n_pts):
            f.write(f"{alti_km[i, 0]:.6f} {alti_km[i, 1]:.6f}\n")
        f.write("\n")


def write_bty_3d(filepath: Union[str, Path], X: np.ndarray, Y: np.ndarray,
                 depth: np.ndarray, interp_type: str = "R") -> None:
    """
    Write 3D bathymetry file for BELLHOP3D.

    Parameters
    ----------
    filepath : str or Path
        Bathymetry file path (typically .bty extension)
    X : ndarray
        X coordinates in kilometers, shape (nx,)
    Y : ndarray
        Y coordinates in kilometers, shape (ny,)
    depth : ndarray
        Depth values in meters, shape (ny, nx)
        depth[iy, ix] is depth at Y[iy], X[ix]
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
    - X: Eastings (km) - horizontal coordinate
    - Y: Northings (km) - vertical coordinate
    - depth: Positive downward (m)

    See Also
    --------
    write_bty_file : Write 2D bathymetry
    read_boundary_3d : Read 3D bathymetry
    """
    filepath = Path(filepath)

    if interp_type not in ['R', 'C']:
        raise ValueError(f"Unknown interpolation type: {interp_type}. Use 'R' or 'C'")

    depth = depth.copy()
    depth[np.isnan(depth)] = 0.0

    nx = len(X)
    ny = len(Y)

    if depth.shape != (ny, nx):
        raise ValueError(f"Depth array shape {depth.shape} doesn't match (ny={ny}, nx={nx})")

    with open(filepath, 'w') as f:
        f.write(f"'{interp_type}'\n")

        f.write(f"{nx}\n")
        for x in X:
            f.write(f"{x:.6f} ")
        f.write("\n")

        f.write(f"{ny}\n")
        for y in Y:
            f.write(f"{y:.6f} ")
        f.write("\n")

        for iy in range(ny):
            for ix in range(nx):
                f.write(f"{depth[iy, ix]:9.3f} ")
            f.write("\n")
