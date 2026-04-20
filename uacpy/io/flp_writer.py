"""
Writers for Acoustics Toolbox field parameter (.flp) files.

Generates .flp files consumed by field.exe and field3d.exe to specify
receiver grid geometry and output options.
"""

import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, Tuple

from uacpy.utils import equally_spaced


def write_fieldflp(
    filepath: Union[str, Path],
    option: str,
    pos: Dict[str, Any],
    title: str = "",
    M_limit: int = 999999,
    n_profiles: int = 1,
    profile_ranges_km: Any = None,
) -> None:
    """
    Write field parameters file (.flp) for FIELD/FIELDS programs.

    Parameters
    ----------
    filepath : str or Path
        Output file path (extension .flp added if missing)
    option : str
        Option string (4 chars max):
        - Pos 1: 'R' (point source, cylindrical), 'X' (line source, Cartesian)
        - Pos 2: 'C' (coupled modes), 'A' (adiabatic) — for NProf > 1
        - Pos 3: '*' (beam pattern) or ' ' (omnidirectional)
        - Pos 4: 'C' (coherent) or 'I' (incoherent)
    pos : dict
        Position dictionary with:
        - 's': dict with 'z' (source depths in m)
        - 'r': dict with 'z' (receiver depths in m), 'r' (ranges in m)
    title : str, optional
        Title for the file (default: empty)
    M_limit : int, optional
        Maximum number of modes to include (default: 999999 = all)
    n_profiles : int, optional
        Number of range profiles (default: 1 for range-independent).
        For range-dependent, set > 1 and provide profile_ranges_km.
    profile_ranges_km : array-like, optional
        Profile boundary ranges in km. Required when n_profiles > 1.
        First value must be 0.0. Length must equal n_profiles.

    Notes
    -----
    File format (.flp):
    - Line 1: Title
    - Line 2: Option (quoted, 4 chars)
    - Line 3: MLimit
    - Line 4: NProf (number of profiles)
    - Line 5: rProf (profile ranges in km)
    - Lines 6+: Receiver ranges, source depths, receiver depths, range offsets

    For range-dependent (NProf > 1), field.exe reads modes for each profile
    from a single .mod file produced by kraken.exe with multi-profile .env.

    See Also
    --------
    read_flp : Read field parameters file
    write_field3dflp : Write 3D field parameters
    """
    filepath = Path(filepath)
    if filepath.suffix != ".flp":
        filepath = filepath.with_suffix(".flp")

    # Extract position data
    r_ranges = pos["r"]["r"] / 1000.0  # Convert m to km
    s_depths = pos["s"]["z"]
    r_depths = pos["r"]["z"]

    # Validate profile parameters
    if n_profiles > 1:
        if profile_ranges_km is None:
            raise ValueError("profile_ranges_km required when n_profiles > 1")
        profile_ranges_km = np.asarray(profile_ranges_km, dtype=float)
        if len(profile_ranges_km) != n_profiles:
            raise ValueError(
                f"profile_ranges_km length ({len(profile_ranges_km)}) "
                f"must equal n_profiles ({n_profiles})"
            )
        if profile_ranges_km[0] != 0.0:
            raise ValueError("First profile range must be 0.0 km")

    with open(filepath, "w") as f:
        # Title
        f.write(f"'{title}' ! Title \n")

        # Option
        f.write(f"'{option:4s}'  ! Option \n")

        # Mode limit
        f.write(f"{M_limit}   ! Mlimit (number of modes to include) \n")

        # Profile info
        f.write(f"{n_profiles}        ! NProf  \n")
        if n_profiles == 1:
            f.write("0.0 /    ! rProf (km) \n")
        else:
            for r in profile_ranges_km:
                f.write(f"    {r:6f}  ")
            f.write("/ \t ! rProf (km) \n")

        # Receiver ranges
        f.write(f"{len(r_ranges):5d} \t \t \t \t ! NRr \n")
        if len(r_ranges) > 2 and equally_spaced(r_ranges):
            f.write(f"    {r_ranges[0]:6f}  {r_ranges[-1]:6f} ")
        else:
            for r in r_ranges:
                f.write(f"    {r:6f}  ")
        f.write("/ \t ! Rr(1)  ... (km) \n")

        # Source depths
        f.write(f"{len(s_depths):5d} \t \t \t \t ! NSz \n")
        if len(s_depths) > 2 and equally_spaced(s_depths):
            f.write(f"    {s_depths[0]:6f}  {s_depths[-1]:6f} ")
        else:
            for z in s_depths:
                f.write(f"    {z:6f}  ")
        f.write("/ \t ! Sz(1)  ... (m) \n")

        # Receiver depths
        f.write(f"{len(r_depths):5d} \t \t \t \t ! NRz \n")
        if len(r_depths) > 2 and equally_spaced(r_depths):
            f.write(f"    {r_depths[0]:6f}  {r_depths[-1]:6f} ")
        else:
            for z in r_depths:
                f.write(f"    {z:6f}  ")
        f.write("/ \t ! Rz(1)  ... (m) \n")

        # Receiver range offsets (array tilt) - default to zeros
        f.write(f"{len(r_depths):5d} \t \t \t \t ! NRro \n")
        f.write("    0.00  0.00 ")
        f.write("/ \t \t \t \t ! Rro(1)  ... (m) \n")


def write_field3dflp(
    filepath: Union[str, Path],
    option: str,
    pos: Dict[str, Any],
    bathy: Dict[str, Any],
    mod_file_pattern: str = "'{}'",
    title: str = "",
    M_limit: int = 999999,
) -> None:
    """
    Write 3D field parameters file (.flp) for FIELD3D program.

    This creates a more complex .flp file for 3D acoustic field computation
    that includes bathymetry nodes, elements, and mode file references.

    Parameters
    ----------
    filepath : str or Path
        Output file path (extension .flp added if missing)
    option : str
        Option string (e.g., 'STDFM' for standard field mode)
    pos : dict
        Position dictionary with:
        - 's': dict with 'x', 'y', 'z' (source coords in km, km, m)
        - 'r': dict with 'z' (receiver depths in m), 'r' (ranges in km),
                         'theta' (bearings in degrees)
        - 'Nsx', 'Nsy': Number of source x,y points
    bathy : dict
        Bathymetry dictionary with:
        - 'X': ndarray - X coordinates in km, shape (nx,)
        - 'Y': ndarray - Y coordinates in km, shape (ny,)
        - 'depth': ndarray - Depths in m, shape (ny, nx)
    mod_file_pattern : str, optional
        Pattern for mode file names. Can include format specifiers.
        Default: "'{}'" (single quoted)
    title : str, optional
        Title for the file (default: empty)
    M_limit : int, optional
        Maximum number of modes to include (default: 999999)

    Notes
    -----
    3D field parameters files specify:
    - Source/receiver positions in 3D
    - Bathymetry node locations (x, y, z)
    - Triangular mesh elements
    - Mode file for each node

    This is used for range and azimuth dependent propagation modeling.

    The bathymetry is represented as a triangulated surface where each
    node has an associated mode file containing normal modes for that
    location.

    Examples
    --------
    >>> import numpy as np
    >>> from uacpy.io import write_field3dflp
    >>>
    >>> # Set up 3D positions
    >>> pos = {
    ...     's': {'x': np.array([0]), 'y': np.array([0]), 'z': np.array([50])},
    ...     'r': {
    ...         'z': np.linspace(0, 100, 11),
    ...         'r': np.linspace(0, 50, 51),
    ...         'theta': np.linspace(0, 360, 37)
    ...     },
    ...     'Nsx': 1,
    ...     'Nsy': 1
    ... }
    >>>
    >>> # Set up bathymetry grid
    >>> X = np.linspace(0, 100, 11)
    >>> Y = np.linspace(0, 100, 11)
    >>> depth = 100 * np.ones((11, 11))
    >>> bathy = {'X': X, 'Y': Y, 'depth': depth}
    >>>
    >>> # Write 3D field parameters
    >>> write_field3dflp('field3d.flp', 'STDFM', pos, bathy,
    ...                  mod_file_pattern="'mode_{:07.1f}_{:07.1f}'",
    ...                  title='3D Test')

    See Also
    --------
    read_flp3d : Read 3D field parameters
    write_fieldflp : Write 2D field parameters
    """
    filepath = Path(filepath)
    if filepath.suffix != ".flp":
        filepath = filepath.with_suffix(".flp")

    # Extract data
    s_x = pos["s"]["x"]
    s_y = pos["s"]["y"]
    s_z = pos["s"]["z"]
    r_z = pos["r"]["z"]
    r_r = pos["r"]["r"]
    r_theta = pos["r"]["theta"]
    Nsx = pos.get("Nsx", len(s_x))
    Nsy = pos.get("Nsy", len(s_y))

    X = bathy["X"]
    Y = bathy["Y"]
    depth = bathy["depth"]
    nx = len(X)
    ny = len(Y)

    with open(filepath, "w") as f:
        # Header
        f.write(f"'{title}' ! Title\n")
        f.write(f"'{option}' \t ! OPT\n")
        f.write(f"{M_limit}   ! Mlimit (number of modes to include)\n")

        # Source x-coordinates
        f.write(f"{Nsx}                 ! Nsx\n")
        if Nsx > 2 and equally_spaced(s_x):
            f.write(f"{s_x[0]} {s_x[-1]}          /   ! Sx( 1 : Nsx ) (km)\n")
        else:
            for x in s_x:
                f.write(f"{x} ")
            f.write("/ ! Sx (km)\n")

        # Source y-coordinates
        f.write(f"{Nsy}                 ! Nsy\n")
        if Nsy > 2 and equally_spaced(s_y):
            f.write(f"{s_y[0]} {s_y[-1]}          /   ! Sy( 1 : Nsy ) (km)\n")
        else:
            for y in s_y:
                f.write(f"{y} ")
            f.write("/ ! Sy (km)\n")

        # Source depths
        f.write(f"{len(s_z):5d} \t \t \t \t ! NSD\n")
        if len(s_z) > 2 and equally_spaced(s_z):
            f.write(f"    {s_z[0]:6f}  {s_z[-1]:6f} ")
        else:
            for z in s_z:
                f.write(f"    {z:6f}  ")
        f.write("/ \t ! SD(1)  ... (m)\n")

        # Receiver depths
        f.write(f"{len(r_z):5d} \t \t \t \t ! NRD\n")
        if len(r_z) > 2 and equally_spaced(r_z):
            f.write(f"    {r_z[0]:6f}  {r_z[-1]:6f} ")
        else:
            for z in r_z:
                f.write(f"    {z:6f}  ")
        f.write("/ \t ! RD(1)  ... (m)\n")

        # Receiver ranges
        f.write(f"{len(r_r):5d} \t \t \t \t ! NRR\n")
        if len(r_r) > 2 and equally_spaced(r_r):
            f.write(f"    {r_r[0]:6f}  {r_r[-1]:6f} ")
        else:
            for r in r_r:
                f.write(f"    {r:6f}  ")
        f.write("/ \t ! RR(1)  ... (km)\n")

        # Receiver bearings
        f.write(f"{len(r_theta)}              \n")
        if len(r_theta) > 2 and equally_spaced(r_theta):
            f.write(f"{r_theta[0]:.1f} {r_theta[-1]:.1f} /")
        else:
            for theta in r_theta:
                f.write(f"{theta:.1f} ")
            f.write("/")
        f.write("        ! NTHETA THETA(1:NTHETA) (degrees)\n")

        # Nodes
        nnodes = nx * ny
        f.write(f"{nnodes:5d}\n")

        # Write node data (x, y, mode_file)
        for iy in range(ny):
            for ix in range(nx):
                x_coord = X[ix]
                y_coord = Y[iy]
                z_depth = depth[iy, ix]

                # Generate mode file name
                if z_depth > 0:
                    if "{}" in mod_file_pattern or "{:" in mod_file_pattern:
                        modfil = mod_file_pattern.format(x_coord, y_coord)
                    else:
                        modfil = mod_file_pattern
                else:
                    modfil = "'DUMMY'"

                f.write(f"{x_coord:8.2f} {y_coord:8.2f} {modfil}\n")

        # Elements (triangular mesh)
        nelts = 2 * (nx - 1) * (ny - 1)
        f.write(f"{nelts:5d}\n")

        inode = 0
        for iy in range(ny - 1):
            for ix in range(nx - 1):
                # Two triangles per grid cell
                f.write(f"{inode:5d} {inode + 1:5d} {inode + nx:5d}\n")
                f.write(f"{inode + 1:5d} {inode + nx:5d} {inode + nx + 1:5d}\n")
                inode += 1
            inode += 1
