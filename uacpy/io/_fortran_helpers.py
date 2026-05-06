"""
Low-level Fortran-record helpers shared by the AT/Bellhop output readers.

These functions read fragments of the binary ``.shd``/``.mod`` formats —
record-length-prefixed vectors, source/receiver depth blocks, bearing/angle
arrays. They are private to ``uacpy.io`` and not part of the public surface.
"""

from typing import Tuple, Dict, Any

import numpy as np


def read_vector(fid) -> Tuple[np.ndarray, int]:
    """
    Read a vector from BELLHOP environment file with Fortran-style input.

    This routine emulates Fortran capability that allows '/' to terminate input
    and create equally-spaced vectors. Supports three input formats:

    1. Explicit values: N / v1 v2 v3 ... vN
    2. Linear spacing: N / v_start v_end / (creates N points between start and end)
    3. Replicate value: N / v / (creates N copies of v)

    Parameters
    ----------
    fid : file object
        Open file handle positioned at vector specification

    Returns
    -------
    x : ndarray
        Vector of values
    Nx : int
        Number of values

    Notes
    -----
    Examples of input formats:

    Format 1 (linear spacing):
        5
        0 1000 /
    Creates: [0, 250, 500, 750, 1000]

    Format 2 (explicit values):
        5
        0 100 300 700 1000
    Creates: [0, 100, 300, 700, 1000]

    Format 3 (replicate):
        501
        0.0 /
    Creates: [0, 0, ..., 0] (501 zeros)

    The '/' character terminates reading and triggers vector generation.

    Translated from OALIB readvector.m

    Examples
    --------
    >>> # Create test file
    >>> with open('test_vec.txt', 'w') as f:
    ...     f.write('5\\n0 1000 /\\n')
    >>> with open('test_vec.txt', 'r') as f:
    ...     x, Nx = read_vector(f)
    >>> print(x)
    [   0.  250.  500.  750. 1000.]
    """
    # Read number of values
    line = fid.readline()
    Nx = int(line.strip())

    # Read values line
    line = fid.readline()

    if "/" in line:
        # Extract numbers before '/'
        nums_str = line.split("/")[0].strip()
        if nums_str:
            values = np.fromstring(nums_str, sep=" ")
        else:
            values = np.array([])

        if Nx == 1:
            x = values[0] if len(values) > 0 else 0.0
        elif Nx == 2:
            x = values[:2] if len(values) >= 2 else values
        elif Nx > 2:
            if len(values) > 1:
                # Generate linearly spaced vector
                x = np.linspace(values[0], values[1], Nx)
            elif len(values) == 1:
                # Replicate single value
                x = np.full(Nx, values[0])
            else:
                # No values provided, return zeros
                x = np.zeros(Nx)
        else:
            x = np.array([])
    else:
        # Read explicit values
        x = np.fromstring(line, sep=" ", count=Nx)

    # Ensure x is a 1D array
    x = np.atleast_1d(x)

    return x, Nx


def read_receiver_ranges(fid, verbose: bool = True) -> np.ndarray:
    """
    Read receiver ranges from BELLHOP environment file.

    Parameters
    ----------
    fid : file object
        Open file handle positioned at receiver range specification
    verbose : bool, optional
        Print information to console (default: True)

    Returns
    -------
    r_r : ndarray
        Receiver ranges in kilometers

    Notes
    -----
    Uses read_vector() to parse range specification, which supports:
    - Explicit ranges
    - Linear spacing with '/' terminator
    - Single value replication

    Output is returned in kilometers (as read from file).

    Translated from OALIB readr.m

    Examples
    --------
    >>> # Create test file
    >>> with open('test_r.txt', 'w') as f:
    ...     f.write('3\\n0 10 /\\n')
    >>> with open('test_r.txt', 'r') as f:
    ...     r = read_receiver_ranges(f, verbose=False)
    >>> print(r)
    [ 0.  5. 10.]
    """
    if verbose:
        print("\n_______________________")

    r_r, NRr = read_vector(fid)

    if verbose:
        print(f"\n Number of receiver ranges, NRr = {NRr}")
        print("\n Receiver ranges, Rr (km)")
        if NRr < 10:
            for r in r_r:
                print(f"{r:8.2f}")
        else:
            print(f"{r_r[0]:8.2f} ... {r_r[-1]:8.2f}")

    return r_r


def read_source_receiver_depths(fid, verbose: bool = True) -> Dict[str, Any]:
    """
    Read source and receiver depths from BELLHOP environment file.

    Parameters
    ----------
    fid : file object
        Open file handle positioned at depth specification
    verbose : bool, optional
        Print information to console (default: True)

    Returns
    -------
    pos : dict
        Dictionary with structure:
        - 's': dict with 'z' (source depths in m)
        - 'r': dict with 'z' (receiver depths in m)
        - 'Nsz': int (number of source depths)
        - 'Nrz': int (number of receiver depths)

    Notes
    -----
    Reads source depths followed by receiver depths using read_vector().

    Translated from OALIB readszrz.m

    Examples
    --------
    >>> # Create test file
    >>> with open('test_sz_rz.txt', 'w') as f:
    ...     f.write('2\\n10 50 /\\n')   # source depths
    ...     f.write('3\\n0 100 /\\n')   # receiver depths
    >>> with open('test_sz_rz.txt', 'r') as f:
    ...     pos = read_source_receiver_depths(f, verbose=False)
    >>> print(pos['s']['z'])
    [10. 50.]
    >>> print(pos['r']['z'])
    [  0.  50. 100.]
    """
    pos = {}

    # Source depths
    if verbose:
        print("\n_______________________")

    sz, Nsz = read_vector(fid)
    pos["s"] = {"z": sz}
    pos["Nsz"] = Nsz

    if verbose:
        print(f"\n Number of source depths, NSz = {Nsz}")
        print("\n Source depths, Sz (m)")
        if Nsz < 10:
            for z in sz:
                print(f"{z:8.2f}")
        else:
            print(f"{sz[0]:8.2f} ... {sz[-1]:8.2f}")

    # Receiver depths
    if verbose:
        print("\n_______________________")

    rz, Nrz = read_vector(fid)
    pos["r"] = {"z": rz}
    pos["Nrz"] = Nrz

    if verbose:
        print(f"\n Number of receiver depths, NRz = {Nrz}")
        print("\n Receiver depths, Rz (m)")
        if Nrz < 10:
            for z in rz:
                print(f"{z:8.2f}")
        else:
            print(f"{rz[0]:8.2f} ... {rz[-1]:8.2f}")

    return pos


def read_source_xy(
    fid, verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Read source x and y coordinates from BELLHOP3D environment file.

    Parameters
    ----------
    fid : file object
        Open file handle positioned at source coordinate specification
    verbose : bool, optional
        Print information to console (default: True)

    Returns
    -------
    sx : ndarray
        Source x coordinates in meters
    sy : ndarray
        Source y coordinates in meters
    Nsx : int
        Number of source x coordinates
    Nsy : int
        Number of source y coordinates

    Notes
    -----
    Reads x coordinates first, then y coordinates. Each uses read_vector().
    Input values are in kilometers and converted to meters on output.

    Translated from OALIB readsxsy.m

    Examples
    --------
    >>> # Create test file
    >>> with open('test_sxsy.txt', 'w') as f:
    ...     f.write('2\\n0 10 /\\n')   # x coords in km
    ...     f.write('3\\n0 5 /\\n')    # y coords in km
    >>> with open('test_sxsy.txt', 'r') as f:
    ...     sx, sy, Nsx, Nsy = read_source_xy(f, verbose=False)
    >>> print(sx)  # in meters
    [    0. 10000.]
    >>> print(sy)
    [   0. 2500. 5000.]
    """
    if verbose:
        print("\n_______________________")

    # x coordinates
    sx, Nsx = read_vector(fid)

    if verbose:
        print(f"\n Number of source x coordinates, NSx = {Nsx}")
        print("\n Source x coordinates (km)")
        for x in sx:
            print(f"{x:.6f}", end=" ")
        print()

    # y coordinates
    sy, Nsy = read_vector(fid)

    if verbose:
        print(f"\n Number of source y coordinates, NSy = {Nsy}")
        print("\n Source y coordinates (km)")
        for y in sy:
            print(f"{y:.6f}", end=" ")
        print()

    # Convert km to m
    sx = 1000.0 * sx
    sy = 1000.0 * sy

    return sx, sy, Nsx, Nsy


def read_receiver_bearings(fid, verbose: bool = True) -> np.ndarray:
    """
    Read receiver bearings from BELLHOP3D environment file.

    Parameters
    ----------
    fid : file object
        Open file handle positioned at bearing specification
    verbose : bool, optional
        Print information to console (default: True)

    Returns
    -------
    theta : ndarray
        Receiver bearings in degrees

    Notes
    -----
    If more than 2 bearings specified, first two values define endpoints
    and Ntheta linearly-spaced bearings are generated between them.

    If bearings form a full 360-degree sweep (last = first + 360),
    the duplicate angle is removed.

    Translated from OALIB readRcvrBearings.m

    Examples
    --------
    >>> # Create test file
    >>> with open('test_bearings.txt', 'w') as f:
    ...     f.write('5\\n0 90\\n')  # 5 bearings from 0 to 90 degrees
    >>> with open('test_bearings.txt', 'r') as f:
    ...     theta = read_receiver_bearings(f, verbose=False)
    >>> print(theta)
    [ 0.   22.5  45.   67.5  90. ]
    """
    # Read number of bearings
    line = fid.readline()
    Ntheta = int(line.strip())

    if verbose:
        print(f"\nNumber of receiver bearings = {Ntheta}")

    fid.readline()  # Skip to next line

    # Read bearing values
    line = fid.readline()
    theta = np.fromstring(line, sep=" ", count=Ntheta)

    if verbose:
        print("\nReceiver bearings (degrees)")
        for t in theta:
            print(f"{t:.6f}", end=" ")
        print()

    # Generate linearly-spaced bearings if more than 2 specified
    if Ntheta > 2:
        theta = np.linspace(theta[0], theta[1], Ntheta)

    # Remove duplicate if full 360-degree sweep
    if len(theta) > 1 and np.abs(theta[-1] - (theta[0] + 360.0)) < 1e-10:
        theta = theta[:-1]
        Ntheta = len(theta)

    fid.readline()  # Skip trailing line

    return theta


def read_receiver_angles(fid, verbose: bool = True) -> np.ndarray:
    """
    Read receiver angles from BELLHOP environment file.

    Parameters
    ----------
    fid : file object
        Open file handle positioned at angle specification
    verbose : bool, optional
        Print information to console (default: True)

    Returns
    -------
    theta : ndarray
        Receiver angles in degrees

    Notes
    -----
    Uses read_vector() to parse angle specification.
    Similar to read_receiver_bearings() but simpler (no special handling
    for full 360-degree sweep).

    Translated from OALIB readtheta.m

    Examples
    --------
    >>> # Create test file
    >>> with open('test_angles.txt', 'w') as f:
    ...     f.write('4\\n-30 30 /\\n')
    >>> with open('test_angles.txt', 'r') as f:
    ...     theta = read_receiver_angles(f, verbose=False)
    >>> print(theta)
    [-30. -10.  10.  30.]
    """
    if verbose:
        print("\n_______________________")

    theta, Ntheta = read_vector(fid)

    if verbose:
        print(f"\n Number of receiver angles = {Ntheta}")
        print("\n Receiver angles (degrees)")
        for t in theta:
            print(f"{t:.6f}", end=" ")
        print()

    return theta
