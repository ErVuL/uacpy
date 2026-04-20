"""
Bathymetry file writer for acoustic models
"""

import numpy as np
from pathlib import Path
from typing import Union


def write_bty_file(filepath: Union[str, Path], bathymetry: np.ndarray, interp_type: str = "L") -> None:
    """
    Write bathymetry file for acoustic models.

    Parameters
    ----------
    filepath : str or Path
        Bathymetry file path (typically .bty extension)
    bathymetry : ndarray
        Bathymetry data, shape (N, 2):
        - Column 0: Range in meters
        - Column 1: Depth in meters
    interp_type : str, optional
        Interpolation type (single character):
        - 'C': Curvilinear (with tangents/normals)
        - 'L': Linear interpolation (default)

    Notes
    -----
    File format:
    - Line 1: Interpolation type in quotes
    - Line 2: Number of points
    - Following lines: range (km), depth (m) pairs

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

    # Convert to km for range column (first column)
    bathy_km = bathymetry.copy()
    bathy_km[:, 0] = bathy_km[:, 0] / 1000.0  # Convert m to km

    n_pts = bathy_km.shape[0]

    with open(filepath, "w") as f:
        # Write interpolation type
        f.write(f"'{interp_type}'\n")

        # Write number of points
        f.write(f"{n_pts}\n")

        # Write range-depth pairs (range in km, depth in m)
        for i in range(n_pts):
            f.write(f"{bathy_km[i, 0]:.6f} {bathy_km[i, 1]:.6f}\n")

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
        Interpolation type (single character):
        - 'C': Curvilinear (with tangents/normals)
        - 'L': Linear interpolation (default)

    Notes
    -----
    File format identical to bathymetry (.bty) files.
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

    # Convert to km for range column
    alti_km = altimetry.copy()
    alti_km[:, 0] = alti_km[:, 0] / 1000.0  # Convert m to km

    n_pts = alti_km.shape[0]

    with open(filepath, "w") as f:
        # Write interpolation type
        f.write(f"'{interp_type}'\n")

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
