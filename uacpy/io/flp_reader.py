"""
Field parameters file reader for KRAKEN/FIELD programs
"""

import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, Tuple
from uacpy.utils import equally_spaced


def read_flp(fileroot: Union[str, Path]) -> Dict[str, Any]:
    """
    Read field parameters file (.flp) for KRAKEN/FIELD programs.

    Field parameters files specify how to compute acoustic fields from
    mode data, including receiver positions, profile ranges, and options.

    Parameters
    ----------
    fileroot : str or Path
        File root name (without .flp extension)

    Returns
    -------
    flp_data : dict
        Dictionary containing:
        - 'title': str - Title from file
        - 'opt': str - 4-character field.exe option string. Column
          semantics per AT ``field.f90:70-99`` / ``ReadModes.f90``:

          * ``opt[0]`` (source type):
            'R' = cylindrical point source (pressure),
            'X' = Cartesian line source,
            'S' = scaled-cylindrical point source.
          * ``opt[1]`` (profile mode for NProf > 1):
            'C' = coupled modes, 'A' = adiabatic.
          * ``opt[2]`` (elastic component selector / SBP flag):
            'P' = acoustic pressure (default), 'H' = horizontal velocity,
            'V' = vertical velocity, 'T' = tangential stress,
            'N' = normal stress, or '*' = apply .sbp beam pattern.
          * ``opt[3]`` (mode summation):
            'C' = coherent, 'I' = incoherent.
        - 'comp': str - Component selector (same as ``opt[2]``).
        - 'M_limit': int - Maximum number of modes to use
        - 'N_prof': int - Number of profiles
        - 'r_prof': ndarray - Profile ranges in meters
        - 'pos': dict - Position information
          - 's': dict with 'z' (source depths in m)
          - 'r': dict with 'z' (receiver depths in m), 'r' (ranges in m),
                              'ro' (range offsets in m)
          - 'Nro': int - Number of range offsets

    Notes
    -----
    File format (.flp):
    - Line 1: Title
    - Line 2: Options (quoted string)
    - Line 3: MLimit
    - Line 4+: Profile range vector (using / shorthand)
    - Receiver ranges
    - Source and receiver depths
    - Receiver range offsets (array tilt)

    The .flp file is used by FIELD/FIELDS programs to compute acoustic
    fields from KRAKEN mode data.

    Translated from OALIB read_flp.m

    Examples
    --------
    >>> flp = read_flp('test')
    >>> print(f"Options: {flp['opt']}")
    >>> print(f"Receiver depths: {flp['pos']['r']['z']}")
    >>> print(f"Receiver ranges: {flp['pos']['r']['r']}")

    See Also
    --------
    read_flp3d : Read 3D field parameters
    write_fieldflp : Write field parameters file
    """
    fileroot = Path(fileroot)
    if not fileroot.suffix:
        filepath = fileroot.with_suffix(".flp")
    else:
        filepath = fileroot

    with open(filepath, "r") as f:
        # Read title
        title = f.readline().strip()
        if "'" in title:
            # Extract text between quotes
            start = title.find("'") + 1
            end = title.find("'", start)
            title = title[start:end]
        print(f"Title: {title}")

        # Read options
        opt = f.readline().strip()
        if "'" in opt:
            start = opt.find("'") + 1
            end = opt.find("'", start)
            opt = opt[start:end]
        print(f"Options: {opt}")

        # Fill missing option columns with reasonable placeholders.
        # pos 3 is the elastic-component / beam-pattern column; we fill
        # it with ' ' (no beam pattern, no elastic component info) rather
        # than fabricating a default. pos 4 defaults to 'C' (coherent).
        if len(opt) <= 2:
            opt += " "
        if len(opt) <= 3:
            opt += "C"

        # Component selector lives in option column 3 (AT
        # ReadModes.f90:315-324). If the file didn't specify one, return
        # it verbatim rather than inventing a "P" default that wasn't in
        # the file — downstream code can distinguish ' ' vs 'P'.
        comp = opt[2]

        # Read MLimit
        M_limit = int(f.readline().strip())
        print(f"MLimit = {M_limit}\n")

        # Read profile ranges using _read_vector
        r_prof, N_prof = _read_vector(f)
        print(f"\nNumber of profiles, NProf = {N_prof}")
        print("Profile ranges, rProf (km)")
        if N_prof < 10:
            for r in r_prof:
                print(f"{r:8.2f}")
        else:
            print(f"{r_prof[0]:8.2f} ... {r_prof[-1]:8.2f}")

        # Read receiver ranges
        r_rcv, _ = _read_vector(f)
        r_rcv = r_rcv * 1000.0  # Convert km to m

        # Read source and receiver depths
        pos_temp = _read_sz_rz(f)

        # Read receiver range offsets (array tilt)
        r_offsets, N_offsets = _read_vector(f)

        print(f"\nNumber of receiver range offsets = {N_offsets}")
        print("Receiver range offsets, Rro (m)")
        if N_offsets < 10:
            for ro in r_offsets:
                print(f"{ro:8.2f}")
        else:
            print(f"{r_offsets[0]:8.2f} ... {r_offsets[-1]:8.2f}")

        if np.max(np.abs(r_offsets)) > 0.0:
            print("Warning: Receiver range offsets are not zero")

    return {
        "title": title,
        "opt": opt,
        "comp": comp,
        "M_limit": M_limit,
        "N_prof": N_prof,
        "r_prof": r_prof * 1000.0,  # Convert to meters
        "pos": {
            "s": {"z": pos_temp["sz"]},
            "r": {"z": pos_temp["rz"], "r": r_rcv, "ro": r_offsets},
            "Nro": N_offsets,
        },
    }


def read_flp3d(fileroot: Union[str, Path]) -> Dict[str, Any]:
    """
    Read 3D field parameters file (.flp) for FIELD3D program.

    Parameters
    ----------
    fileroot : str or Path
        File root name (without .flp extension)

    Returns
    -------
    flp3d_data : dict
        Dictionary containing 3D field parameters including:
        - 'title': str - Title
        - 'opt': str - Options
        - 'comp': str - Component
        - 'M_limit': int - Mode limit
        - 'N_prof': int - Number of profiles
        - 'r_prof': ndarray - Profile ranges (m)
        - 'theta_prof': ndarray - Profile bearings (degrees)
        - 'pos': dict - 3D position data

    Notes
    -----
    Similar to read_flp but for 3D cylindrical geometry used by FIELD3D.
    """
    fileroot = Path(fileroot)
    if not fileroot.suffix:
        filepath = fileroot.with_suffix(".flp")
    else:
        filepath = fileroot

    with open(filepath, "r") as f:
        # Read title
        title = f.readline().strip()
        if "'" in title:
            start = title.find("'") + 1
            end = title.find("'", start)
            title = title[start:end]

        # Read options
        opt = f.readline().strip()
        if "'" in opt:
            start = opt.find("'") + 1
            end = opt.find("'", start)
            opt = opt[start:end]

        if len(opt) <= 2:
            opt += " "
        if len(opt) <= 3:
            opt += "C"

        # See read_flp: column 3 is the elastic component / SBP flag.
        comp = opt[2]

        # Read MLimit
        M_limit = int(f.readline().strip())

        # Read profile info (ranges and bearings)
        r_prof, N_r_prof = _read_vector(f)
        theta_prof, N_theta_prof = _read_vector(f)

        # Read receiver ranges and bearings
        r_rcv, _ = _read_vector(f)
        theta_rcv, _ = _read_vector(f)

        # Read source and receiver depths
        pos_temp = _read_sz_rz(f)

        # Read range offsets
        r_offsets, N_offsets = _read_vector(f)

    return {
        "title": title,
        "opt": opt,
        "comp": comp,
        "M_limit": M_limit,
        "N_r_prof": N_r_prof,
        "N_theta_prof": N_theta_prof,
        "r_prof": r_prof * 1000.0,  # Convert to meters
        "theta_prof": theta_prof,
        "pos": {
            "s": {"z": pos_temp["sz"]},
            "r": {
                "z": pos_temp["rz"],
                "r": r_rcv * 1000.0,  # Convert to meters
                "theta": theta_rcv,
                "ro": r_offsets,
            },
            "Nro": N_offsets,
        },
    }


def _read_vector(fid) -> Tuple[np.ndarray, int]:
    """
    Read a vector from file with Fortran-style / shorthand.

    Helper function for read_flp.
    """
    # Read number of points
    n = int(fid.readline().strip())

    # Read data line
    line = fid.readline().strip()

    if "/" in line:
        # Parse values before '/'
        parts = line.split("/")[0].strip().split()
        values = [float(v) for v in parts if v]

        if n == 1:
            x = np.array([values[0]]) if values else np.array([0.0])
        elif n == 2:
            x = (
                np.array(values[:2])
                if len(values) >= 2
                else np.array([values[0], values[0]])
            )
        else:
            # Generate linearly spaced
            if len(values) >= 2:
                x = np.linspace(values[0], values[1], n)
            elif len(values) == 1:
                x = np.full(n, values[0])
            else:
                x = np.zeros(n)
    else:
        # Explicit values
        values = [float(v) for v in line.split() if v]
        x = np.array(values[:n])
        if len(x) < n:
            x = np.pad(x, (0, n - len(x)), constant_values=0)

    return x, n


def _read_sz_rz(fid) -> Dict[str, np.ndarray]:
    """
    Read source and receiver depths.

    Helper function for read_flp.
    """
    # Read source depths
    sz, _ = _read_vector(fid)

    # Read receiver depths
    rz, _ = _read_vector(fid)

    return {"sz": sz, "rz": rz}
