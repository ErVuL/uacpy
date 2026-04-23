"""
Bathymetry file writer for acoustic models
"""

import numpy as np
from pathlib import Path
from typing import Union


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
        ``ranges_km``, ``sound_speed``, ``density``, ``attenuation``,
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

    Ranges in ``bottom_rd.ranges_km`` are re-sampled onto the bathymetry
    range grid via ``numpy.interp`` so the two lengths always match.
    """
    filepath = Path(filepath)
    interp_char = _validate_interp_type(interp_type)
    # 2-char TYPE: position 1 = interpolation, position 2 = 'L' long format
    type_str = f"{interp_char}L"

    bathy_km = bathymetry.copy()
    bathy_km[:, 0] = bathy_km[:, 0] / 1000.0
    n_pts = bathy_km.shape[0]

    rd_r = np.asarray(bottom_rd.ranges_km, dtype=float)
    cp = np.interp(bathy_km[:, 0], rd_r, bottom_rd.sound_speed)
    rho = np.interp(bathy_km[:, 0], rd_r, bottom_rd.density)
    alpha = np.interp(bathy_km[:, 0], rd_r, bottom_rd.attenuation)
    cs_arr = bottom_rd.shear_speed if bottom_rd.shear_speed is not None \
        else np.zeros_like(rd_r)
    cs = np.interp(bathy_km[:, 0], rd_r, cs_arr)

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
