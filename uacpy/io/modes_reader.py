"""
Readers for Kraken normal-mode files (``.mod`` binary, ``.moa`` ASCII).

* ``read_modes`` — auto-detect binary/ASCII by extension.
* ``read_modes_bin`` — binary ``.mod``.
* ``read_modes_asc`` — ASCII ``.moa``.
* ``get_component`` — extract a column from a Kraken modes dict.
"""

import numpy as np
from typing import Any, Dict, Optional, Union

from uacpy.core.exceptions import ModelExecutionError


def get_component(modes_dict: Dict[str, Any], comp: str) -> np.ndarray:
    """
    Extract a single component from the stress-displacement vector in mode data.

    For elastic media, mode solutions contain multiple components (horizontal
    displacement, vertical displacement, stress components). This function
    extracts the requested component.

    Parameters
    ----------
    modes_dict : dict
        Modes structure containing:
        - 'phi': Mode shapes (ndarray)
        - 'z': Depth vector
        - 'Nmedia': Number of media layers
        - 'Mater': Material types ('ACOUSTIC' or 'ELASTIC ')
        - 'N': Number of mesh points per medium (optional)
    comp : str
        Component to extract:
        - 'H': Horizontal displacement
        - 'V': Vertical displacement
        - 'T': Tangential stress
        - 'N': Normal stress
        For acoustic media, any component returns the pressure field.

    Returns
    -------
    phi : ndarray
        Extracted component, shape (nz, nmodes)

    Notes
    -----
    Mode structure conventions:
    - Acoustic media: 1 component (pressure)
    - Elastic media: 4 components (H, V, T, N)

    KRAKEL tabulates modes at finite difference grid points.
    Other codes (KRAKEN, KRAKENC) tabulate at receiver depths.

    Translated from OALIB get_component.m by mbp

    Examples
    --------
    >>> # Extract horizontal displacement from elastic modes
    >>> phi_h = get_component(modes, 'H')

    >>> # Extract pressure from acoustic modes (any component)
    >>> phi_p = get_component(modes, 'H')  # Same as 'V', 'T', 'N'
    """
    phi = []
    jj = 0
    k = 0

    Nmedia = modes_dict.get("Nmedia", 1)
    phi_full = modes_dict["phi"]
    z = modes_dict["z"]
    Mater = modes_dict.get("Mater", [["ACOUSTIC"]])

    # Step through each medium
    for medium in range(Nmedia):
        for ii in range(len(z)):
            # Jump out if modes are not tabulated in elastic media
            if k >= phi_full.shape[0]:
                break

            # Get material type for this medium
            if medium < len(Mater):
                material = (
                    Mater[medium].strip()
                    if isinstance(Mater[medium], str)
                    else str(Mater[medium]).strip()
                )
            else:
                material = "ACOUSTIC"

            if material == "ACOUSTIC":
                # Acoustic: single component (pressure)
                if jj < len(z):
                    phi.append(phi_full[k, :])
                k += 1

            elif material == "ELASTIC":
                # Elastic: 4 components
                if jj < len(z):
                    if comp == "H":
                        phi.append(phi_full[k, :])
                    elif comp == "V":
                        phi.append(phi_full[k + 1, :])
                    elif comp == "T":
                        phi.append(phi_full[k + 2, :])
                    elif comp == "N":
                        phi.append(phi_full[k + 3, :])
                    else:
                        raise ValueError(f"Unknown component: {comp}")
                k += 4

            else:
                raise ValueError(f"Unknown material type: {material}")

            jj += 1

    return np.array(phi) if phi else np.array([])


def read_modes_asc(
    filename: str, modes: Optional[Union[int, list, np.ndarray]] = None
) -> Dict[str, Any]:
    """
    Read normal modes from KRAKEN ASCII format mode file.

    KRAKEN is a normal mode model for underwater acoustics that computes
    mode shapes and wavenumbers. This function reads the ASCII format
    output file (.mod extension).

    Parameters
    ----------
    filename : str
        Mode filename (should include .mod extension)
    modes : int, list, or ndarray, optional
        Mode indices to read (1-indexed like MATLAB).
        If None, reads all modes (default).
        Can be:
        - Single integer: read that mode
        - List/array: read specified modes

    Returns
    -------
    Modes : dict
        Dictionary containing:
        - 'pltitl': Plot title string
        - 'freq': Frequency (Hz)
        - 'Nmedia': Number of media layers
        - 'ntot': Total number of depth points
        - 'nmat': Number of material points
        - 'M': Total number of modes available
        - 'z': Depth vector (m), shape (ntot,)
        - 'k': Complex wavenumbers (1/m), shape (nmodes_read,)
        - 'phi': Mode shapes, shape (ntot, nmodes_read)

    Notes
    -----
    File format (.mod ASCII):
    - Record length (not used in ASCII)
    - Title line
    - freq, Nmedia, ntot, nmat, M
    - Nmedia lines of medium properties
    - Top halfspace line
    - Bottom halfspace line
    - Blank line
    - Depth vector
    - Wavenumbers (real and imaginary pairs)
    - Mode shapes (real and imaginary pairs for each mode)

    Mode indices are 1-indexed to match MATLAB convention.
    Python users may prefer 0-indexing, but this maintains compatibility.

    Wavenumbers are complex: k = k_real + 1j * k_imag
    - k_real: horizontal wavenumber (rad/m)
    - k_imag: attenuation (negative for decaying modes)

    Mode shapes are complex pressure fields normalized by KRAKEN.

    Examples
    --------
    >>> # Read all modes
    >>> modes = read_modes_asc('pekeris.mod')
    >>> print(f"Frequency: {modes['freq']} Hz")
    >>> print(f"Number of modes: {len(modes['k'])}")
    >>> print(f"Depth points: {len(modes['z'])}")

    >>> # Read specific modes
    >>> modes = read_modes_asc('pekeris.mod', modes=[1, 3, 5])
    >>> print(f"Read modes: {len(modes['k'])}")

    >>> # Plot first mode
    >>> import matplotlib.pyplot as plt
    >>> modes = read_modes_asc('pekeris.mod')
    >>> plt.plot(np.real(modes['phi'][:, 0]), modes['z'])
    >>> plt.gca().invert_yaxis()
    >>> plt.xlabel('Mode amplitude')
    >>> plt.ylabel('Depth (m)')
    >>> plt.title(f"Mode 1, k = {modes['k'][0]:.6f}")

    >>> # Compute horizontal wavenumbers
    >>> modes = read_modes_asc('pekeris.mod')
    >>> kr = np.real(modes['k'])  # horizontal wavenumber
    >>> ki = np.imag(modes['k'])  # attenuation
    >>> print(f"Mode 1: kr = {kr[0]:.6f}, alpha = {-ki[0]:.6e}")

    See Also
    --------
    read_modes_bin : Read binary format mode files
    get_component : Extract components from elastic modes
    """
    try:
        with open(filename, "r") as fid:
            # Read record length (not used in ASCII)
            int(fid.readline().strip())

            # Read title
            pltitl = fid.readline().strip()

            # Read parameters
            params_line = fid.readline().strip().split()
            freq = float(params_line[0])
            Nmedia = int(params_line[1])
            ntot = int(params_line[2])
            nmat = int(params_line[3])
            M = int(params_line[4])  # total number of modes

            # Skip medium property lines
            for _ in range(Nmedia):
                fid.readline()

            # Skip halfspace lines and blank line
            fid.readline()  # top halfspace
            fid.readline()  # bottom halfspace
            fid.readline()  # blank line

            # Read depth vector
            z_line = fid.readline().strip().split()
            z = np.array([float(x) for x in z_line])

            # Read wavenumbers (real and imaginary parts)
            k_real_line = fid.readline().strip().split()
            k_imag_line = fid.readline().strip().split()
            k_real = np.array([float(x) for x in k_real_line])
            k_imag = np.array([float(x) for x in k_imag_line])
            k_all = k_real + 1j * k_imag

            # Determine which modes to read
            if modes is None:
                modes_to_read = list(range(1, M + 1))  # 1-indexed
            elif isinstance(modes, (int, np.integer)):
                modes_to_read = [modes]
            else:
                modes_to_read = list(modes)

            # Filter out invalid mode numbers
            modes_to_read = [m for m in modes_to_read if 1 <= m <= M]

            # Select requested wavenumbers
            k_selected = k_all[[m - 1 for m in modes_to_read]]  # Convert to 0-indexed

            # Read mode shapes
            phi = np.zeros((ntot, len(modes_to_read)), dtype=complex)

            for mode_num in range(1, M + 1):
                # Skip blank line before each mode
                fid.readline()

                # Read real parts
                phi_real_line = fid.readline().strip().split()
                phi_real = np.array([float(x) for x in phi_real_line])

                # Read imaginary parts
                phi_imag_line = fid.readline().strip().split()
                phi_imag = np.array([float(x) for x in phi_imag_line])

                phi_mode = phi_real + 1j * phi_imag

                # Store if this mode is in the list
                if mode_num in modes_to_read:
                    idx = modes_to_read.index(mode_num)
                    phi[:, idx] = phi_mode

    except FileNotFoundError:
        raise FileNotFoundError(f"Mode file not found: {filename}")
    except Exception as e:
        raise ModelExecutionError(
            'Kraken', return_code=0, stdout=None,
            stderr=f"Error reading mode file {filename}: {e}",
        ) from e

    # Return as dictionary
    return {
        "pltitl": pltitl,
        "freq": freq,
        "Nmedia": Nmedia,
        "ntot": ntot,
        "nmat": nmat,
        "M": len(k_selected),  # number of modes read
        "z": z,
        "k": k_selected,
        "phi": phi,
    }


def read_modes_bin(
    filename: str,
    freq: float = 0.0,
    modes: Optional[Union[int, list, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Read mode data from KRAKEN binary format (.moA file).

    This function reads normal mode data including eigenvalues (wavenumbers),
    eigenfunctions (mode shapes), and environmental parameters from KRAKEN
    model output files.

    Parameters
    ----------
    filename : str
        Path to mode file (without extension, assumes '.moA').
    freq : float, optional
        Frequency in Hz for which to read modes. For broadband runs,
        selects the closest frequency. Use freq=0 if only one frequency.
        Default is 0.0.
    modes : int, list, or ndarray, optional
        Mode indices to read (1-indexed). If None, reads all modes.
        Can be:
        - int: read modes 1 through this number
        - list/array: read specific mode indices

    Returns
    -------
    modes_data : dict
        Dictionary containing mode information:
        - 'title' : str - Title from mode file
        - 'Nfreq' : int - Number of frequencies
        - 'Nmedia' : int - Number of media
        - 'N' : list - Number of depth points in each medium
        - 'Mater' : list - Material type of each medium ('ACOUSTIC' or 'ELASTIC')
        - 'depth' : ndarray - Depths of interfaces
        - 'rho' : ndarray - Densities in each medium
        - 'freqVec' : ndarray - Frequencies for which modes were calculated
        - 'z' : ndarray - Sample depths for modes
        - 'M' : int - Number of modes
        - 'phi' : ndarray - Mode shapes, shape (ntot, M) complex
        - 'k' : ndarray - Wavenumbers, shape (M,) complex
        - 'Top' : dict - Top boundary properties
            - 'BC' : str - Boundary condition
            - 'cp' : complex - P-wave speed
            - 'cs' : complex - S-wave speed
            - 'rho' : float - Density
            - 'depth' : float - Depth
        - 'Bot' : dict - Bottom boundary properties (same fields as Top)

    Notes
    -----
    - The file extension '.moA' is assumed and should not be included
    - This function uses persistent state to read sequentially from the same
      file across multiple calls in MATLAB. In Python, each call reopens the file.
    - Modes are stored in Fortran unformatted binary format
    - Record length (lrecl) is determined from first 4 bytes
    - Mode indices are 1-indexed (MATLAB/Fortran convention)

    References
    ----------
    Based on BELLHOP/read_modes_bin.m by Aaron Thode (1996)

    Examples
    --------
    >>> # Read all modes at 100 Hz
    >>> modes = read_modes_bin('pekeris', freq=100.0)
    >>> print(f"Number of modes: {modes['M']}")
    >>> print(f"Wavenumber of mode 1: {modes['k'][0]}")

    >>> # Read specific modes
    >>> modes = read_modes_bin('pekeris', freq=100.0, modes=[1, 2, 3])
    >>> print(f"Mode shapes: {modes['phi'].shape}")
    """
    # Add extension if not present
    if not filename.endswith(".moA"):
        filename = filename + ".moA"

    with open(filename, "rb") as fid:
        # Read record length (in bytes, converted from Fortran words)
        lrecl = 4 * np.fromfile(fid, dtype=np.int32, count=1)[0]

        # Read header record (rec 0)
        fid.seek(4, 0)  # Skip 4 bytes
        title_bytes = fid.read(80)
        title = title_bytes.decode("ascii", errors="ignore").strip()

        Nfreq = np.fromfile(fid, dtype=np.int32, count=1)[0]
        Nmedia = np.fromfile(fid, dtype=np.int32, count=1)[0]
        Ntot = np.fromfile(fid, dtype=np.int32, count=1)[0]
        NMat = np.fromfile(fid, dtype=np.int32, count=1)[0]

        if Ntot < 0:
            raise ValueError("Invalid mode file: Ntot < 0")

        # Read N and Mater (rec 1)
        fid.seek(lrecl, 0)
        N = []
        Mater = []
        for medium in range(Nmedia):
            n_val = np.fromfile(fid, dtype=np.int32, count=1)[0]
            N.append(n_val)
            mater_bytes = fid.read(8)
            Mater.append(mater_bytes.decode("ascii", errors="ignore").strip())

        # Read depth and density (rec 2)
        fid.seek(2 * lrecl, 0)
        bulk = np.fromfile(fid, dtype=np.float32, count=2 * Nmedia).reshape(
            (2, Nmedia), order="F"
        )
        depth = bulk[0, :]
        rho = bulk[1, :]

        # Read frequencies (rec 3)
        fid.seek(3 * lrecl, 0)
        freqVec = np.fromfile(fid, dtype=np.float64, count=Nfreq)

        # Read z (rec 4)
        fid.seek(4 * lrecl, 0)
        z = np.fromfile(fid, dtype=np.float32, count=Ntot)

        # Find closest frequency
        freq_diff = np.abs(freqVec - freq)
        freq_index = np.argmin(freq_diff)

        # Navigate to the correct frequency
        # Initialize iRecProfile to 5 (matches MATLAB after iRecProfile += 4)
        # Records 0-3: Header, N/Mater, depth/rho, freqVec
        # Record 4: z vector
        # Record 5: M (mode count) - this is where we start
        iRecProfile = 5
        for ifreq in range(freq_index + 1):
            fid.seek(iRecProfile * lrecl, 0)
            M = np.fromfile(fid, dtype=np.int32, count=1)[0]

            if ifreq < freq_index:
                # Advance to next frequency
                # Formula from MATLAB line 113: iRecProfile + 3 + M + floor(4*(2*M-1)/lrecl)
                iRecProfile = iRecProfile + 3 + M + (4 * (2 * M - 1)) // lrecl

        # Determine which modes to read
        if modes is None:
            modes = np.arange(1, M + 1)
        elif isinstance(modes, int):
            modes = np.arange(1, min(modes, M) + 1)
        else:
            modes = np.array(modes)
            modes = modes[modes <= M]

        # Read top and bottom halfspace info (rec iRecProfile + 2)
        fid.seek((iRecProfile + 2) * lrecl, 0)

        # Top boundary
        Top = {}
        Top["BC"] = chr(np.fromfile(fid, dtype=np.uint8, count=1)[0])
        cp_data = np.fromfile(fid, dtype=np.float32, count=2)
        Top["cp"] = complex(cp_data[0], cp_data[1])
        cs_data = np.fromfile(fid, dtype=np.float32, count=2)
        Top["cs"] = complex(cs_data[0], cs_data[1])
        Top["rho"] = np.fromfile(fid, dtype=np.float32, count=1)[0]
        Top["depth"] = np.fromfile(fid, dtype=np.float32, count=1)[0]

        # Bottom boundary
        Bot = {}
        Bot["BC"] = chr(np.fromfile(fid, dtype=np.uint8, count=1)[0])
        cp_data = np.fromfile(fid, dtype=np.float32, count=2)
        Bot["cp"] = complex(cp_data[0], cp_data[1])
        cs_data = np.fromfile(fid, dtype=np.float32, count=2)
        Bot["cs"] = complex(cs_data[0], cs_data[1])
        Bot["rho"] = np.fromfile(fid, dtype=np.float32, count=1)[0]
        Bot["depth"] = np.fromfile(fid, dtype=np.float32, count=1)[0]

        # Read eigenfunctions
        if M == 0:
            phi = np.array([])
            k = np.array([])
        else:
            phi = np.zeros((NMat, len(modes)), dtype=np.complex64)

            for ii, mode_idx in enumerate(modes):
                rec = iRecProfile + 1 + mode_idx
                fid.seek(rec * lrecl, 0)
                phi_data = np.fromfile(fid, dtype=np.float32, count=2 * NMat).reshape(
                    (2, NMat), order="F"
                )
                phi[:, ii] = phi_data[0, :] + 1j * phi_data[1, :]

            # Read wavenumbers
            rec = iRecProfile + 2 + M
            fid.seek(rec * lrecl, 0)
            k_data = np.fromfile(fid, dtype=np.float32, count=2 * M).reshape(
                (2, M), order="F"
            )
            k = k_data[0, :] + 1j * k_data[1, :]
            k = k[modes - 1]  # Convert to 0-indexed and select requested modes

    return {
        "title": title,
        "Nfreq": Nfreq,
        "Nmedia": Nmedia,
        "N": N,
        "Mater": Mater,
        "depth": depth,
        "rho": rho,
        "freqVec": freqVec,
        "z": z,
        "M": M,
        "phi": phi,
        "k": k,
        "Top": Top,
        "Bot": Bot,
    }


def read_modes(
    filename: str,
    freq: float = 0.0,
    modes: Optional[Union[int, list, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Read mode data from KRAKEN output file (wrapper for binary and ASCII readers).

    This is a convenience wrapper that automatically detects the file format
    and calls the appropriate reader (read_modes_bin for .mod files,
    read_modes_asc for .moa files).

    Parameters
    ----------
    filename : str
        Mode file path, with or without extension. Supported extensions:
        - '.mod': Binary format (default if no extension)
        - '.moa': ASCII format
        - '.mod.mat': MATLAB format (loads directly)
    freq : float, optional
        Frequency in Hz to select from multi-frequency files (default: 0)
    modes : int, list, or ndarray, optional
        Mode indices to extract (1-indexed). If None, all modes are returned.

    Returns
    -------
    modes_data : dict
        Mode data dictionary with fields from read_modes_bin or read_modes_asc,
        plus computed halfspace parameters:
        - 'Top': dict with top halfspace properties (k2, gamma, phi)
        - 'Bot': dict with bottom halfspace properties (k2, gamma, phi)

    Notes
    -----
    For acoustic halfspaces (boundary condition 'A'), computes:
    - k²: wavenumber squared in halfspace
    - γ: vertical wavenumber using Pekeris root
    - φ: mode value at interface

    The frequency index is found by searching for the closest match to
    the requested frequency in freqVec.

    Translated from OALIB read_modes.m

    Examples
    --------
    >>> # Read binary mode file
    >>> modes = read_modes('test.mod', freq=100.0)
    >>> print(f"Number of modes: {modes['M']}")
    >>> print(f"Wavenumbers shape: {modes['k'].shape}")

    >>> # Read specific modes
    >>> modes = read_modes('test.mod', freq=100.0, modes=[1, 2, 3])

    >>> # ASCII format
    >>> modes = read_modes('test.moa')
    """
    import os
    from ..utils import pekeris_root  # Import for halfspace calculations

    # Parse filename
    fileroot, ext = os.path.splitext(filename)

    if not ext:
        ext = ".mod"  # Default extension
    elif ext == ".mat":
        # Handle .mod.mat files
        fileroot2, ext2 = os.path.splitext(fileroot)
        if ext2 == ".mod":
            fileroot = fileroot2
            ext = ".mod.mat"

    filename = fileroot + ext

    # Read mode data based on extension
    if ext == ".mod":
        # Binary format
        if modes is None:
            Modes = read_modes_bin(filename, freq)
        else:
            Modes = read_modes_bin(filename, freq, modes)

    elif ext == ".mod.mat":
        # MATLAB format - load directly
        import scipy.io

        mat_data = scipy.io.loadmat(filename)
        Modes = {}
        # Extract variables from MATLAB struct
        # (This is a simplified version; actual implementation may need adjustment)
        for key in mat_data.keys():
            if not key.startswith("__"):
                Modes[key] = mat_data[key]

    elif ext == ".moa":
        # ASCII format
        if modes is None:
            Modes = read_modes_asc(filename)
        else:
            Modes = read_modes_asc(filename, modes)

    else:
        raise ValueError(f"read_modes: Unrecognized file extension {ext}")

    # Find frequency index closest to requested frequency
    freq_diff = np.abs(Modes["freqVec"] - freq)
    freq_index = np.argmin(freq_diff)

    # Calculate wavenumbers in halfspaces (if there are any modes)
    if Modes["M"] != 0:
        # Top halfspace
        if Modes["Top"]["BC"] == "A":
            # Acoustic halfspace
            k_top = 2.0 * np.pi * Modes["freqVec"][0] / Modes["Top"]["cp"]
            Modes["Top"]["k2"] = k_top**2
            gamma2 = Modes["k"] ** 2 - Modes["Top"]["k2"]
            Modes["Top"]["gamma"] = pekeris_root(gamma2)
            Modes["Top"]["phi"] = Modes["phi"][0, :]
        else:
            # Vacuum or rigid boundary
            Modes["Top"]["rho"] = 1.0
            Modes["Top"]["gamma"] = np.zeros_like(Modes["k"])
            Modes["Top"]["phi"] = np.zeros_like(Modes["phi"][0, :])

        # Bottom halfspace
        if Modes["Bot"]["BC"] == "A":
            # Acoustic halfspace
            k_bot = 2.0 * np.pi * Modes["freqVec"][freq_index] / Modes["Bot"]["cp"]
            Modes["Bot"]["k2"] = k_bot**2
            gamma2 = Modes["k"] ** 2 - Modes["Bot"]["k2"]
            Modes["Bot"]["gamma"] = pekeris_root(gamma2)
            Modes["Bot"]["phi"] = Modes["phi"][-1, :]
        else:
            # Vacuum or rigid boundary
            Modes["Bot"]["rho"] = 1.0
            Modes["Bot"]["gamma"] = np.zeros_like(Modes["k"])
            Modes["Bot"]["phi"] = np.zeros_like(Modes["phi"][-1, :])

    return Modes
