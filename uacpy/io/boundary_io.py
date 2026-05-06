"""
Boundary / interface auxiliary file I/O.

Readers and writers for the auxiliary files attached to a Bellhop or
Acoustics-Toolbox env description:

* ``.bty`` — bathymetry (:func:`read_bathymetry`, :func:`write_bty_file`,
  :func:`write_bty_long_format`, :func:`write_bty_3d`)
* ``.ati`` — altimetry (:func:`read_altimetry`, :func:`write_ati_file`)
* ``.brc`` / ``.irc`` / ``.trc`` — precomputed reflection coefficients
  (:func:`read_reflection_coefficient`, :func:`write_reflection_coefficient`)
* ``.sbp`` — source beam pattern (:func:`read_source_beam_pattern`,
  :func:`write_source_beam_pattern`)
* 3-D boundary blocks (:func:`read_boundary_3d`)

Reader/writer pairs live next to each other for symmetry and to make the
auxiliary-file surface easy to discover.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Union

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

    Examples
    --------
    >>> # Read bathymetry file
    >>> x, y, z, nx, ny = read_boundary_3d('canyon.bty')
    >>> print(f"Grid: {nx} x {ny} points")
    >>> print(f"X range: {x[0]:.1f} to {x[-1]:.1f} km")
    >>> print(f"Y range: {y[0]:.1f} to {y[-1]:.1f} km")
    >>> print(f"Depth range: {z.min():.1f} to {z.max():.1f} m")

    >>> # Plot bathymetry
    >>> import matplotlib.pyplot as plt
    >>> x, y, z, _, _ = read_boundary_3d('seamount.bty', verbose=False)
    >>> X, Y = np.meshgrid(x, y)
    >>> plt.contourf(X, Y, z, levels=20)
    >>> plt.colorbar(label='Depth (m)')
    >>> plt.xlabel('X (km)')
    >>> plt.ylabel('Y (km)')

    >>> # Use in BELLHOP3D simulation
    >>> x_bty, y_bty, z_bty, _, _ = read_boundary_3d('domain.bty')
    >>> # Interpolate bathymetry to acoustic ray positions
    >>> from scipy.interpolate import RectBivariateSpline
    >>> interp = RectBivariateSpline(y_bty, x_bty, z_bty)
    >>> depth_at_ray = interp(ray_y, ray_x, grid=False)

    See Also
    --------
    read_vector : Read vector with shorthand notation
    scipy.interpolate.RectBivariateSpline : 2D interpolation
    """
    try:
        with open(filename, "r") as fid:
            # Read boundary type
            bdry_type_line = fid.readline().strip()

            # Extract character between quotes
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

            # Read x-coordinates
            x_bot, n_x = read_vector(fid)

            if verbose:
                print(f"Number of boundary points in x = {n_x}\n")
                print(" x (km)")
                for i, x_val in enumerate(x_bot):
                    if i < 50 or i == n_x - 1:
                        print(f"{x_val:9.5g}")
                    elif i == 50:
                        print("   ...")

            # Read y-coordinates
            y_bot, n_y = read_vector(fid)

            if verbose:
                print(f"Number of boundary points in y = {n_y}\n")
                print(" y (km)")
                for i, y_val in enumerate(y_bot):
                    if i < 50 or i == n_y - 1:
                        print(f"{y_val:9.5g}")
                    elif i == 50:
                        print("   ...")

            # Read z-values (depth/height matrix)
            z_values = []
            for line in fid:
                values = [float(v) for v in line.split() if v]
                z_values.extend(values)

            # Reshape and transpose
            z_bot = np.array(z_values).reshape(n_x, n_y).T

    except FileNotFoundError:
        raise FileNotFoundError(f"Boundary file not found: {filename}")
    except Exception as e:
        raise RuntimeError(f"Error reading boundary file {filename}: {e}")

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

    Examples
    --------
    >>> bty, bty_type = read_bathymetry('canyon.bty')
    >>> print(f"Bathymetry type: {bty_type}")
    >>> print(f"Range: {bty[0, 1]:.1f} to {bty[0, -2]:.1f} m")
    >>> print(f"Depth: {bty[1, 1]:.1f} to {bty[1, -2]:.1f} m")
    """
    filepath = Path(filepath)
    if filepath.suffix != ".bty":
        filepath = filepath.with_suffix(".bty")

    with open(filepath, "r") as fid:
        if verbose:
            print("\n_______________________")
            print("Using bottom-bathymetry file")

        # Read interpolation type
        line = fid.readline().strip()
        # Extract character between quotes
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

        # Read number of points
        n_pts = int(fid.readline().strip())

        if verbose:
            print(f"Number of bathymetry points = {n_pts}\n")
            print(" Range (km)     Depth (m)")

        # Read bathymetry points
        bty_data = []
        for i in range(n_pts):
            range_km = float(fid.readline().strip())
            depth_m = float(fid.readline().strip())
            bty_data.append([range_km, depth_m])

            if verbose and (i < 10 or i == n_pts - 1):
                print(f"{range_km:9.5g}    {depth_m:9.5g}")
            elif verbose and i == 10:
                print("    ...")

        bty_data = np.array(bty_data).T  # Shape: (2, n_pts)

    # Convert ranges from km to meters
    bty_data[0, :] *= 1000.0

    # Extend bathymetry to ±infinity
    n_pts_extended = n_pts + 2
    bty = np.zeros((2, n_pts_extended))

    # Left endpoint at -infinity with constant depth
    bty[0, 0] = -1e50
    bty[1, 0] = bty_data[1, 0]

    # Copy interior points
    bty[:, 1:-1] = bty_data

    # Right endpoint at +infinity with constant depth
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

    Examples
    --------
    >>> ati, ati_type = read_altimetry('ice.ati')
    >>> print(f"Altimetry type: {ati_type}")
    >>> print(f"Surface depth range: {ati[1, 1]:.1f} to {ati[1, -2]:.1f} m")
    """
    filepath = Path(filepath)
    if filepath.suffix != ".ati":
        filepath = filepath.with_suffix(".ati")

    with open(filepath, "r") as fid:
        if verbose:
            print("\n_______________________")
            print("Using top-altimetry file")

        # Read interpolation type
        line = fid.readline().strip()
        # Extract character between quotes
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

        # Read number of points
        n_pts = int(fid.readline().strip())

        if verbose:
            print(f"Number of altimetry points = {n_pts}\n")
            print(" Range (km)     Depth (m)")

        # Read altimetry points
        ati_data = []
        for i in range(n_pts):
            range_km = float(fid.readline().strip())
            depth_m = float(fid.readline().strip())
            ati_data.append([range_km, depth_m])

            if verbose and (i < 10 or i == n_pts - 1):
                print(f"{range_km:9.5g}    {depth_m:9.5g}")
            elif verbose and i == 10:
                print("    ...")

        ati_data = np.array(ati_data).T  # Shape: (2, n_pts)

    # Convert ranges from km to meters
    ati_data[0, :] *= 1000.0

    # Extend altimetry to ±infinity
    n_pts_extended = n_pts + 2
    ati = np.zeros((2, n_pts_extended))

    # Left endpoint at -infinity with constant depth
    ati[0, 0] = -1e50
    ati[1, 0] = ati_data[1, 0]

    # Copy interior points
    ati[:, 1:-1] = ati_data

    # Right endpoint at +infinity with constant depth
    ati[0, -1] = 1e50
    ati[1, -1] = ati_data[1, -1]

    return ati, ati_type


def read_reflection_coefficient(
    filename: str, boundary: str = "bottom"
) -> Dict[str, np.ndarray]:
    """
    Read reflection coefficient data from file (.trc or .brc).

    Reads angle-dependent reflection coefficient data used by BELLHOP
    for top or bottom boundary conditions. Data includes angle, magnitude,
    and phase.

    Parameters
    ----------
    filename : str
        Path to reflection coefficient file (.trc for top, .brc for bottom).
        Extension is added automatically if not present.
    boundary : str, optional
        Boundary type: 'top' or 'bottom'. Default is 'bottom'.

    Returns
    -------
    rc_data : dict
        Reflection coefficient data containing:
        - 'theta' : ndarray - Angles in degrees, shape (n,)
        - 'R' : ndarray - Reflection coefficient magnitudes, shape (n,)
        - 'phi' : ndarray - Phases in radians, shape (n,)
        - 'n_pts' : int - Number of data points

    Notes
    -----
    - Angles must be non-decreasing
    - Phase is converted from degrees (in file) to radians
    - File format:
        Line 1: N (number of points)
        Lines 2+: angle(deg) magnitude phase(deg)

    References
    ----------
    Based on BELLHOP/readrc.m

    Examples
    --------
    >>> # Read bottom reflection coefficient
    >>> rc = read_reflection_coefficient('sediment.brc', boundary='bottom')
    >>> print(f"Angles: {rc['theta']}")
    >>> print(f"Magnitudes: {rc['R']}")
    >>> print(f"Phases (rad): {rc['phi']}")

    >>> # Interpolate to specific angle
    >>> angle = 30.0  # degrees
    >>> R_interp = np.interp(angle, rc['theta'], rc['R'])
    >>> phi_interp = np.interp(angle, rc['theta'], rc['phi'])
    """
    # Add appropriate extension
    if boundary.lower() == "top":
        if not filename.endswith(".trc"):
            filename = filename + ".trc"
    else:
        if not filename.endswith(".brc"):
            filename = filename + ".brc"

    try:
        with open(filename, "r") as fid:
            # Read number of points
            n_pts = int(fid.readline().strip())

            if n_pts == 0:
                return {
                    "theta": np.array([0.0]),
                    "R": np.array([0.0]),
                    "phi": np.array([0.0]),
                    "n_pts": 0,
                }

            # Pre-allocate arrays
            theta = np.zeros(n_pts)
            R = np.zeros(n_pts)
            phi = np.zeros(n_pts)

            # Read data (angle magnitude phase per line)
            for i in range(n_pts):
                line = fid.readline().strip()
                values = line.split()
                if len(values) >= 3:
                    theta[i] = float(values[0])
                    R[i] = float(values[1])
                    phi[i] = float(values[2])
                else:
                    # Fallback: try old format (one value per line)
                    theta[i] = float(values[0]) if values else 0.0
                    R[i] = float(fid.readline().strip()) if i == 0 else R[i-1]
                    phi[i] = float(fid.readline().strip()) if i == 0 else phi[i-1]

            # Convert phase from degrees to radians
            phi = phi * np.pi / 180.0

            # Validate angles are non-decreasing
            if not np.all(np.diff(theta) >= 0):
                raise ValueError("Angles must be non-decreasing")

            return {"theta": theta, "R": R, "phi": phi, "n_pts": n_pts}

    except FileNotFoundError:
        # Return zero reflection if file doesn't exist
        return {
            "theta": np.array([0.0]),
            "R": np.array([0.0]),
            "phi": np.array([0.0]),
            "n_pts": 0,
        }


def read_source_beam_pattern(filepath: Union[str, Path], sbp_option: str = "O") -> np.ndarray:
    """
    Read source beam pattern from file.

    Parameters
    ----------
    filepath : str or Path
        Source beam pattern file root name (without .sbp extension)
    sbp_option : str, optional
        Option flag:
        - '*': Read beam pattern from file
        - 'O': Create omni-directional pattern (default)

    Returns
    -------
    beam_pattern : ndarray
        Beam pattern array, shape (N, 2):
        - Column 0: Angles in degrees
        - Column 1: Power (linear scale, not dB)

    Notes
    -----
    File format (.sbp):
    - Line 1: Number of points
    - Subsequent lines: angle (degrees), power (dB)

    Power values are converted from dB to linear scale on output:
        power_linear = 10^(power_dB / 20)

    For omni-directional pattern, creates [-180°, 180°] with 0 dB (=1.0).

    Translated from OALIB readpat.m

    Examples
    --------
    >>> # Omni-directional pattern (default)
    >>> pattern = read_source_beam_pattern('dummy', 'O')
    >>> print(pattern)
    [[-180.    1.]
     [ 180.    1.]]

    >>> # Read from file (if test.sbp exists):
    >>> # pattern = read_source_beam_pattern('test', '*')
    """
    if sbp_option == "*":
        print("-----------------------------------")
        print("Using source beam pattern file")

        filepath = Path(filepath)
        sbp_file = str(filepath) + ".sbp"
        with open(filepath, "r") as fid:
            # Read number of points
            line = fid.readline()
            NSBPPts = int(line.strip())
            print(f"Number of source beam pattern points = {NSBPPts}")

            # Read angle and power data
            beam_pattern = np.zeros((NSBPPts, 2))
            print(" ")
            print(" Angle (degrees)  Power (dB)")

            for i in range(NSBPPts):
                line = fid.readline()
                vals = np.fromstring(line, sep=" ", count=2)
                beam_pattern[i, :] = vals
                print(f" {vals[0]:7.2f}         {vals[1]:6.2f}")

    else:
        # Omni-directional pattern
        beam_pattern = np.array([[-180.0, 0.0], [180.0, 0.0]])

    # Convert dB to linear scale
    beam_pattern[:, 1] = 10.0 ** (beam_pattern[:, 1] / 20.0)

    return beam_pattern


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

    Examples
    --------
    >>> # Create simple sloping bottom
    >>> rng = np.array([0, 10000, 20000, 30000])  # meters
    >>> dep = np.array([100, 110, 120, 130])       # meters
    >>> bathymetry = np.column_stack([rng, dep])
    >>> write_bty_file('test.bty', bathymetry, interp_type='L')
    """
    filepath = Path(filepath)
    interp_char = _validate_interp_type(interp_type)
    # 2-char TYPE: position 1 = interpolation, position 2 = 'S' short format
    type_str = f"{interp_char}S"

    # Convert to km for range column (first column)
    bathy_km = bathymetry.copy()
    bathy_km[:, 0] = bathy_km[:, 0] / 1000.0  # Convert m to km

    n_pts = bathy_km.shape[0]

    with open(filepath, "w") as f:
        # Write 2-character TYPE string
        f.write(f"'{type_str}'\n")

        # Write number of points
        f.write(f"{n_pts}\n")

        # Write range-depth pairs (range in km, depth in m)
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
        ``shear_speed``.
    interp_type : str, optional
        'L' (linear, default) or 'C' (curvilinear).

    Notes
    -----
    File format (extended BTY — long format):
    - Line 1: 2-character TYPE in quotes — position 1 is interpolation
      ('L' or 'C'), position 2 is 'L' (long format, bathymetry +
      geoacoustics). See ATI_BTY_File.htm.
    - Line 2: number of points
    - Following lines: ``range_km depth_m cp_m_s rho_g_cm3 atten cs_m_s``

    Ranges in ``bottom_rd.ranges`` (metres) are re-sampled onto the
    bathymetry range grid via ``numpy.interp`` so the two lengths match.
    """
    filepath = Path(filepath)
    interp_char = _validate_interp_type(interp_type)
    # 2-char TYPE: position 1 = interpolation, position 2 = 'L' long format
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

    with open(filepath, "w") as f:
        f.write(f"'{type_str}'\n")
        f.write(f"{n_pts}\n")
        for i in range(n_pts):
            f.write(
                f"{bathy_km[i, 0]:.6f} {bathy_km[i, 1]:.6f} "
                f"{cp[i]:.3f} {rho[i]:.3f} {alpha[i]:.6f} {cs[i]:.3f}\n"
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
        - Column 1: Altitude/height in meters (positive up from sea level)
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

    Examples
    --------
    >>> # Create surface with ice keel
    >>> rng = np.array([0, 5000, 10000, 15000])  # meters
    >>> alt = np.array([0, -10, -5, 0])          # meters (negative = below surface)
    >>> altimetry = np.column_stack([rng, alt])
    >>> write_ati_file('surface.ati', altimetry, interp_type='L')
    """
    filepath = Path(filepath)
    interp_char = _validate_interp_type(interp_type)
    # 2-char TYPE: position 1 = interpolation, position 2 = 'S' short format
    type_str = f"{interp_char}S"

    # Convert to km for range column
    alti_km = altimetry.copy()
    alti_km[:, 0] = alti_km[:, 0] / 1000.0  # Convert m to km

    n_pts = alti_km.shape[0]

    with open(filepath, "w") as f:
        # Write 2-character TYPE string
        f.write(f"'{type_str}'\n")

        # Write number of points
        f.write(f"{n_pts}\n")

        # Write range-altitude pairs (range in km, altitude in m)
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

    Examples
    --------
    >>> # Create simple 3D bathymetry
    >>> X = np.linspace(0, 10, 11)  # 0 to 10 km, 11 points
    >>> Y = np.linspace(0, 5, 6)    # 0 to 5 km, 6 points
    >>> depth = 100 + 10 * np.outer(Y, X)  # Sloping bottom
    >>> write_bty_3d('test3d.bty', X, Y, depth, interp_type='R')

    >>> # With NaN handling
    >>> depth[0, 0] = np.nan  # Will be replaced with 0.0
    >>> write_bty_3d('test3d.bty', X, Y, depth)

    See Also
    --------
    write_bty_file : Write 2D bathymetry
    read_boundary_3d : Read 3D bathymetry
    """
    filepath = Path(filepath)

    # Validate interpolation type
    if interp_type not in ['R', 'C']:
        raise ValueError(f"Unknown interpolation type: {interp_type}. Use 'R' or 'C'")

    # Replace NaN with 0.0
    depth = depth.copy()
    depth[np.isnan(depth)] = 0.0

    nx = len(X)
    ny = len(Y)

    # Verify depth array shape
    if depth.shape != (ny, nx):
        raise ValueError(f"Depth array shape {depth.shape} doesn't match (ny={ny}, nx={nx})")

    with open(filepath, 'w') as f:
        # Write interpolation type
        f.write(f"'{interp_type}'\n")

        # Write X data
        f.write(f"{nx}\n")
        for x in X:
            f.write(f"{x:.6f} ")
        f.write("\n")

        # Write Y data
        f.write(f"{ny}\n")
        for y in Y:
            f.write(f"{y:.6f} ")
        f.write("\n")

        # Write depth matrix (row by row)
        for iy in range(ny):
            for ix in range(nx):
                f.write(f"{depth[iy, ix]:9.3f} ")
            f.write("\n")


def write_reflection_coefficient(
    filepath: Union[str, Path],
    angles: np.ndarray,
    coefficients: np.ndarray,
    file_type: str = "brc",
) -> None:
    """
    Write reflection coefficient file for Bellhop bottom/top boundary.

    Parameters
    ----------
    filepath : str or Path
        Output file path (typically .brc for bottom, .trc for top)
    angles : ndarray
        Grazing angles in degrees, shape (N,)
    coefficients : ndarray
        Complex reflection coefficients, shape (N,) or (N, 2) for [amplitude, phase]
        If complex, uses magnitude and phase. If real (N, 2), uses directly.
    file_type : str, optional
        File type: 'brc' (bottom) or 'trc' (top). Default is 'brc'.

    Notes
    -----
    File format (.brc/.trc):
    - Line 1: Number of angles
    - Following lines: angle (degrees), |R| (amplitude), phase (degrees)

    The reflection coefficient R = |R| * exp(i*phase)

    Examples
    --------
    >>> # Create simple reflection coefficient
    >>> angles = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    >>> # Full reflection with phase shift
    >>> R_complex = 0.9 * np.exp(1j * np.deg2rad(-20))
    >>> coeffs = np.full(len(angles), R_complex, dtype=complex)
    >>> write_reflection_coefficient('bottom.brc', angles, coeffs)

    >>> # Using amplitude and phase directly
    >>> amp_phase = np.column_stack([np.ones(10) * 0.9, np.ones(10) * -20])
    >>> write_reflection_coefficient('bottom.brc', angles, amp_phase)
    """
    filepath = Path(filepath)

    # Convert complex to amplitude/phase if needed
    if np.iscomplexobj(coefficients):
        amplitude = np.abs(coefficients)
        phase = np.angle(coefficients, deg=True)
    elif coefficients.ndim == 2 and coefficients.shape[1] == 2:
        amplitude = coefficients[:, 0]
        phase = coefficients[:, 1]
    else:
        # Assume real values are amplitudes with zero phase
        amplitude = coefficients
        phase = np.zeros_like(coefficients)

    n_angles = len(angles)

    with open(filepath, "w") as f:
        # Write number of angles
        f.write(f"{n_angles}\n")

        # Write angle, amplitude, phase triplets
        for i in range(n_angles):
            f.write(f"{angles[i]:8.2f} {amplitude[i]:12.6f} {phase[i]:12.6f}\n")


def write_source_beam_pattern(
    filepath: Union[str, Path], angles: np.ndarray, pattern: np.ndarray
) -> None:
    """
    Write source beam pattern file for Bellhop.

    Parameters
    ----------
    filepath : str or Path
        Output file path (typically .sbp extension)
    angles : ndarray
        Beam angles in degrees, shape (N,)
        Typically from -90 to +90 degrees
    pattern : ndarray
        Beam pattern level in dB relative to peak, shape (N,)
        (typically 0 dB at peak, negative elsewhere).  Bellhop
        converts dB -> linear via 10**(SrcBmPat(:,2)/20) at load time
        (bellhop.f90:132).

    Notes
    -----
    File format (.sbp):
    - Line 1: Number of angles
    - Following lines: angle (degrees), level (dB re peak)

    Used to specify directional source characteristics.

    Examples
    --------
    >>> # Create Gaussian beam pattern (levels in dB)
    >>> angles = np.linspace(-90, 90, 181)
    >>> width = 10  # degrees (half-power beamwidth)
    >>> # linear -> dB; 0 dB at peak
    >>> linear = np.exp(-(angles**2) / (2 * width**2))
    >>> pattern_db = 20 * np.log10(np.clip(linear, 1e-6, None))
    >>> write_source_beam_pattern('source.sbp', angles, pattern_db)

    >>> # Create omnidirectional source (0 dB everywhere)
    >>> angles = np.array([-90, 0, 90])
    >>> pattern = np.array([0.0, 0.0, 0.0])
    >>> write_source_beam_pattern('omni.sbp', angles, pattern)
    """
    filepath = Path(filepath)

    n_angles = len(angles)

    with open(filepath, "w") as f:
        # Write number of angles
        f.write(f"{n_angles}\n")

        # Write angle, amplitude pairs
        for i in range(n_angles):
            f.write(f"{angles[i]:8.2f} {pattern[i]:12.6f}\n")
