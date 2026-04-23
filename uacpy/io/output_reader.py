"""
Readers for acoustic model output files
"""

import numpy as np
import struct
from pathlib import Path
from typing import Union, Tuple, Dict, Optional, Any

from uacpy.core.field import Field
from uacpy.core.constants import PRESSURE_FLOOR, TL_MAX_DB


def read_shd_file(filepath: Union[str, Path]) -> Field:
    """
    Read shade file (.shd) - binary TL output from Bellhop/Kraken/Scooter

    Based on Acoustics Toolbox read_shd_bin.m format

    Parameters
    ----------
    filepath : str or Path
        Path to .shd file

    Returns
    -------
    field : Field
        Field object with TL data
    """
    filepath = Path(filepath)

    with open(filepath, "rb") as f:
        # Record 1: Title
        recl = struct.unpack("i", f.read(4))[0]  # record length in 4-byte words
        title = f.read(80).decode("utf-8", errors="ignore").strip()

        # Record 2: PlotType
        f.seek(4 * recl, 0)
        plot_type = f.read(10).decode("utf-8", errors="ignore").strip()

        # Record 3: Dimensions
        f.seek(2 * 4 * recl, 0)
        nfreq = struct.unpack("i", f.read(4))[0]
        ntheta = struct.unpack("i", f.read(4))[0]
        nsx = struct.unpack("i", f.read(4))[0]
        nsy = struct.unpack("i", f.read(4))[0]
        nsz = struct.unpack("i", f.read(4))[0]
        nrz = struct.unpack("i", f.read(4))[0]
        nrr = struct.unpack("i", f.read(4))[0]
        freq0 = struct.unpack("d", f.read(8))[0]
        atten = struct.unpack("d", f.read(8))[0]

        # Record 4: Frequency vector
        f.seek(3 * 4 * recl, 0)
        freqs = np.array([struct.unpack("d", f.read(8))[0] for _ in range(nfreq)])

        # Record 5: Theta
        f.seek(4 * 4 * recl, 0)
        theta = np.array([struct.unpack("d", f.read(8))[0] for _ in range(ntheta)])

        # Records 6-7: Source x,y (skip for now)
        # Record 8: Source depths
        f.seek(7 * 4 * recl, 0)
        sz = np.array([struct.unpack("f", f.read(4))[0] for _ in range(nsz)])

        # Record 9: Receiver depths
        f.seek(8 * 4 * recl, 0)
        rz = np.array([struct.unpack("f", f.read(4))[0] for _ in range(nrz)])

        # Record 10: Receiver ranges
        # NOTE: field.exe outputs ranges in METERS (not km as documented)
        # Bellhop/Scooter may output in km - need to detect and handle
        f.seek(9 * 4 * recl, 0)
        rr = np.array([struct.unpack("d", f.read(8))[0] for _ in range(nrr)])

        # Auto-detect units: if ranges are < 100, assume km and convert to m
        # This heuristic works for typical scenarios (ranges > 100m)
        if len(rr) > 0 and rr.max() < 100.0:
            rr = rr * 1000.0  # Convert km to m

        # Allocate pressure array
        # For rectilinear: pressure(ntheta, nsz, nrz, nrr)
        if "irregular" in plot_type.lower():
            nrcvrs_per_range = 1
        else:
            nrcvrs_per_range = nrz

        pressure = np.zeros((ntheta, nsz, nrcvrs_per_range, nrr), dtype=complex)

        # Read pressure data (Record 10+)
        # For first frequency, all theta, all source depths, all receiver depths
        ifreq = 0  # Read first frequency
        for itheta in range(ntheta):
            for isz in range(nsz):
                for irz in range(nrcvrs_per_range):
                    recnum = (
                        10
                        + (ifreq) * ntheta * nsz * nrcvrs_per_range
                        + (itheta) * nsz * nrcvrs_per_range
                        + (isz) * nrcvrs_per_range
                        + irz
                    )

                    f.seek(recnum * 4 * recl, 0)
                    temp = np.array(
                        [struct.unpack("f", f.read(4))[0] for _ in range(2 * nrr)]
                    )
                    # Interleaved real/imag
                    pressure[itheta, isz, irz, :] = temp[0::2] + 1j * temp[1::2]

        # Extract first source depth, all receiver depths
        # pressure shape: (ntheta, nsz, nrz, nrr) -> (nrz, nrr) for first theta, first source
        p = pressure[0, 0, :, :]

        # Convert to transmission loss with proper handling of shadow zones
        p_abs = np.abs(p)
        p_abs = np.maximum(p_abs, PRESSURE_FLOOR)
        tl_data = -20 * np.log10(p_abs)
        tl_data = np.clip(tl_data, None, TL_MAX_DB)

    return Field(
        field_type="tl",
        data=tl_data,
        ranges=rr,
        depths=rz,
        frequencies=freqs,
        metadata={
            "title": title,
            "plot_type": plot_type,
            "source_file": str(filepath),
            "freq0": freq0,
            "atten": atten,
            "source_depths": sz,
        },
    )


def read_arr_file(filepath: Union[str, Path]) -> Field:
    """
    Read arrivals file (.arr) from Bellhop

    Parameters
    ----------
    filepath : str or Path
        Path to .arr file

    Returns
    -------
    field : Field
        Field object with arrivals data. The metadata contains:
        - 'arrivals_by_receiver': nested list [isd][ird][irr] of per-receiver
          arrival dicts, each with keys: amplitudes, phases, delays,
          delay_imag, src_angles, rcv_angles, n_top_bounces, n_bot_bounces
        - 'frequency': center frequency in Hz
        - 'source_depths': array of source depths
        - 'receiver_depths': array of receiver depths
        - 'receiver_ranges': array of receiver ranges in meters
    """
    filepath = Path(filepath)

    # Fortran record marker length (in 4-byte words)
    # Most FORTRAN compilers use 1, some use 2
    marker_len = 1

    with open(filepath, "rb") as f:
        # Check if binary or ASCII format. The ASCII arrivals file always
        # begins with a quoted "'2D'" or "'3D'" tag at the very start of the
        # first line. The binary format is a Fortran unformatted stream whose
        # first bytes are a 4-byte record marker (typically \x04\x00\x00\x00
        # for a 4-byte record, but compilers may emit other marker lengths).
        # Prefer the positive ASCII test and fall back to binary otherwise.
        head = f.read(16)
        f.seek(0)
        try:
            head_text = head.decode('ascii')
        except UnicodeDecodeError:
            head_text = ''
        if head_text.lstrip().startswith(("'2D'", "'3D'")):
            is_binary = False
        else:
            is_binary = True

        if is_binary:
            # Binary arrivals format: each arrival is written as its own
            # Fortran unformatted sequential record (ArrMod.f90:164), so
            # each read requires bracketing-marker handling — the existing
            # implementation below treated arrivals as a single contiguous
            # block, producing wrong offsets and values.  Rather than ship
            # a half-verified rewrite, fail loudly and point users to the
            # ASCII path which is well-tested.
            from uacpy.core.exceptions import ConfigurationError
            raise ConfigurationError(
                "Binary arrivals format (.arr written by RunType 'a') is "
                "not fully supported yet. Re-run Bellhop with "
                "arrivals_format='ascii' (RunType 'A'). See "
                "ArrMod.f90:WriteArrivalsBinary for the authoritative "
                "record layout if you wish to contribute a reader."
            )
            # Unreachable — left as documentation of the historical code
            # path until a verified implementation lands.
            # Check if 2D or 3D format
            f.seek(4 * marker_len, 0)  # Skip first record marker
            flag = f.read(4).decode('ascii', errors='ignore')

            if flag not in ["'2D'", "'3D'"]:
                raise ValueError(f"Not a valid binary arrivals file: flag={repr(flag)}")

            is_3d = (flag == "'3D'")
        else:
            # ASCII format - read flag from text
            f.seek(0)
            text_content = f.read(10).decode('ascii', errors='ignore')

            if "'2D'" in text_content:
                is_3d = False
                flag = "'2D'"
            elif "'3D'" in text_content:
                is_3d = True
                flag = "'3D'"
            else:
                raise ValueError(f"Not a valid arrivals file: {repr(text_content)}")

        if not is_3d:
            # Read 2D format
            if is_binary:
                # Binary format reading
                # Frequency
                f.seek(8 * marker_len, 0)
                freq = struct.unpack('f', f.read(4))[0]

                # Number of source depths
                f.seek(8 * marker_len, 0)
                nsd = struct.unpack('i', f.read(4))[0]
                sz = np.frombuffer(f.read(4 * nsd), dtype='float32')

                # Number of receiver depths
                f.seek(8 * marker_len, 0)
                nrd = struct.unpack('i', f.read(4))[0]
                rz = np.frombuffer(f.read(4 * nrd), dtype='float32')

                # Number of receiver ranges
                f.seek(8 * marker_len, 0)
                nrr = struct.unpack('i', f.read(4))[0]
                rr = np.frombuffer(f.read(8 * nrr), dtype='float64')

                # Per-receiver structured arrivals
                arrivals_by_receiver = []

                # Read arrivals for each source depth
                for isd in range(nsd):
                    sd_list = []
                    # Maximum number of arrivals for this source
                    f.seek(8 * marker_len, 0)
                    narrmx = struct.unpack('i', f.read(4))[0]

                    for irz in range(nrd):
                        rd_list = []
                        for irr in range(nrr):
                            # Number of arrivals at this receiver
                            f.seek(8 * marker_len, 0)
                            narr = struct.unpack('i', f.read(4))[0]

                            rcv_arrivals = {
                                "amplitudes": np.array([], dtype='float64'),
                                "phases": np.array([], dtype='float64'),
                                "delays": np.array([], dtype='float64'),
                                "delay_imag": np.array([], dtype='float64'),
                                "src_angles": np.array([], dtype='float64'),
                                "rcv_angles": np.array([], dtype='float64'),
                                "n_top_bounces": np.array([], dtype='int32'),
                                "n_bot_bounces": np.array([], dtype='int32'),
                                "n_arrivals": 0,
                            }

                            if narr > 0:
                                # Read arrival data (8 floats per arrival + markers)
                                # Format: amp, phase, delay_real, delay_imag, src_angle, rcv_angle, n_top, n_bot
                                data = np.frombuffer(
                                    f.read(4 * (8 + 2 * marker_len) * narr),
                                    dtype='float32'
                                ).reshape(narr, 8 + 2 * marker_len)

                                # Discard record markers
                                data = data[:, 2 * marker_len:]

                                # Extract fields
                                amp = data[:, 0].astype('float64')
                                phase = data[:, 1].astype('float64')
                                delay_real = data[:, 2].astype('float64')
                                delay_imag = data[:, 3].astype('float64')
                                src_angle = data[:, 4].astype('float64')
                                rcv_angle = data[:, 5].astype('float64')
                                n_top = data[:, 6].astype('int32')
                                n_bot = data[:, 7].astype('int32')

                                rcv_arrivals = {
                                    "amplitudes": amp,
                                    "phases": phase,
                                    "delays": delay_real,
                                    "delay_imag": delay_imag,
                                    "src_angles": src_angle,
                                    "rcv_angles": rcv_angle,
                                    "n_top_bounces": n_top,
                                    "n_bot_bounces": n_bot,
                                    "n_arrivals": narr,
                                }

                            rd_list.append(rcv_arrivals)
                        sd_list.append(rd_list)
                    arrivals_by_receiver.append(sd_list)
            else:
                # ASCII format reading
                f.close()
                with open(filepath, 'r') as f:
                    # Read all lines
                    lines = f.readlines()
                    idx = 0

                    # Skip flag line
                    idx += 1

                    # Parse header - read all values from space-separated lines
                    freq = float(lines[idx].strip())
                    idx += 1

                    # Number of source depths and values
                    tokens = lines[idx].strip().split()
                    nsd = int(tokens[0])
                    sz = np.array([float(t) for t in tokens[1:1+nsd]])
                    idx += 1

                    # Number of receiver depths and values
                    tokens = lines[idx].strip().split()
                    nrd = int(tokens[0])
                    rz = np.array([float(t) for t in tokens[1:1+nrd]])
                    idx += 1

                    # Number of receiver ranges and values
                    tokens = lines[idx].strip().split()
                    nrr = int(tokens[0])
                    rr = np.array([float(t) for t in tokens[1:1+nrr]])
                    idx += 1

                    # Per-receiver structured arrivals
                    arrivals_by_receiver = []

                    # Read arrivals for each source depth
                    for isd in range(nsd):
                        sd_list = []
                        # Maximum number of arrivals for this source
                        narrmx = int(lines[idx].strip())
                        idx += 1

                        for irz in range(nrd):
                            rd_list = []
                            for irr in range(nrr):
                                # Number of arrivals at this receiver
                                narr = int(lines[idx].strip())
                                idx += 1

                                rcv_arrivals = {
                                    "amplitudes": np.array([], dtype='float64'),
                                    "phases": np.array([], dtype='float64'),
                                    "delays": np.array([], dtype='float64'),
                                    "delay_imag": np.array([], dtype='float64'),
                                    "src_angles": np.array([], dtype='float64'),
                                    "rcv_angles": np.array([], dtype='float64'),
                                    "n_top_bounces": np.array([], dtype='int32'),
                                    "n_bot_bounces": np.array([], dtype='int32'),
                                    "n_arrivals": 0,
                                }

                                if narr > 0:
                                    amps = []
                                    phases = []
                                    delays_r = []
                                    delays_i = []
                                    src_angs = []
                                    rcv_angs = []
                                    n_tops = []
                                    n_bots = []

                                    for ia in range(narr):
                                        values = lines[idx].strip().split()
                                        idx += 1

                                        amps.append(float(values[0]))
                                        phases.append(float(values[1]))
                                        delays_r.append(float(values[2]))
                                        delays_i.append(float(values[3]))
                                        src_angs.append(float(values[4]))
                                        rcv_angs.append(float(values[5]))
                                        n_tops.append(int(float(values[6])))
                                        n_bots.append(int(float(values[7])))

                                    rcv_arrivals = {
                                        "amplitudes": np.array(amps),
                                        "phases": np.array(phases),
                                        "delays": np.array(delays_r),
                                        "delay_imag": np.array(delays_i),
                                        "src_angles": np.array(src_angs),
                                        "rcv_angles": np.array(rcv_angs),
                                        "n_top_bounces": np.array(n_tops, dtype='int32'),
                                        "n_bot_bounces": np.array(n_bots, dtype='int32'),
                                        "n_arrivals": narr,
                                    }

                                rd_list.append(rcv_arrivals)
                            sd_list.append(rd_list)
                        arrivals_by_receiver.append(sd_list)
        else:
            # 3D format - similar structure but with more dimensions
            raise NotImplementedError("3D arrivals format not yet implemented")

    return Field(
        field_type="arrivals",
        data=np.array([]),  # Arrivals don't have regular grid data
        ranges=rr,
        depths=rz,
        frequencies=np.array([freq]),
        metadata={
            "arrivals_by_receiver": arrivals_by_receiver,
            "frequency": freq,
            "source_depths": sz,
            "receiver_depths": rz,
            "receiver_ranges": rr,
        },
    )


def read_ray_file(filepath: Union[str, Path]) -> Field:
    """
    Read ray file (.ray) from Bellhop

    Parameters
    ----------
    filepath : str or Path
        Path to .ray file

    Returns
    -------
    field : Field
        Field object with ray paths
    """
    filepath = Path(filepath)

    rays = []

    try:
        with open(filepath, "r") as f:
            # Read header
            # Line 1: Title (with quotes)
            f.readline()
            # Line 2: Frequency
            f.readline()
            # Line 3: NSx, NSy, NSz (number of sources)
            f.readline()
            # Line 4: Nalpha, Nbeta (number of angles)
            f.readline()
            # Line 5: Top depth
            f.readline()
            # Line 6: Bottom depth
            f.readline()
            # Line 7: Coordinate system ('rz' or 'xyz')
            coord_line = f.readline().strip()

            # Read rays - format is:
            # alpha (launch angle - float, no quotes)
            # n_points n_top_bounces n_bot_bounces (3 integers)
            # Then n_points lines of ray data
            while True:
                # Try to read angle line
                angle_line = f.readline()
                if not angle_line:
                    break

                # Skip empty lines
                if not angle_line.strip():
                    continue

                # Try to parse as angle (float)
                try:
                    alpha = float(angle_line.strip())
                except ValueError:
                    # Not a valid angle, skip
                    continue

                # Read n_points, n_top_bounces, n_bot_bounces
                counts_line = f.readline()
                if not counts_line:
                    break

                counts = counts_line.split()
                if len(counts) < 1:
                    continue

                n_points = int(counts[0])
                # Parse bounce counts if present
                n_top_bounces = int(counts[1]) if len(counts) > 1 else 0
                n_bot_bounces = int(counts[2]) if len(counts) > 2 else 0

                if n_points == 0:
                    continue

                ray_r = []
                ray_z = []

                for _ in range(n_points):
                    line = f.readline().strip()
                    if not line:
                        break
                    parts = line.split()
                    if len(parts) >= 2:
                        # Bellhop's WriteRay2D (WriteRay.f90:45) writes
                        # ray2D(is)%x directly in meters (the MATLAB
                        # plotray.m only divides by 1000 when the user
                        # requests km output). No unit conversion here.
                        ray_r.append(float(parts[0]))
                        ray_z.append(float(parts[1]))

                if len(ray_r) > 0:
                    rays.append(
                        {
                            "r": np.array(ray_r),
                            "z": np.array(ray_z),
                            "alpha": alpha,
                            "num_top_bounces": n_top_bounces,
                            "num_bottom_bounces": n_bot_bounces,
                        }
                    )

    except Exception as e:
        print(f"Warning: Error reading ray file: {e}")
        # Try binary format
        rays = _read_ray_file_binary(filepath)

    return Field(
        field_type="rays",
        data=np.array([]),  # Rays don't have regular grid data
        metadata={"rays": rays},
    )


def _read_ray_file_binary(filepath: Path) -> list:
    """
    Read binary ray file format

    Parameters
    ----------
    filepath : Path
        Path to binary .ray file

    Returns
    -------
    rays : list
        List of ray dictionaries
    """
    rays = []

    with open(filepath, "rb") as f:
        # Read header
        recl = struct.unpack("i", f.read(4))[0]
        # Skip header info
        f.seek(recl * 4)

        # Read rays
        try:
            while True:
                n_points = struct.unpack("i", f.read(4))[0]
                if n_points <= 0:
                    break

                ray_r = []
                ray_z = []

                for _ in range(n_points):
                    # WriteRay2D writes ray2D%x directly in meters
                    # (WriteRay.f90:45); no km conversion needed.
                    r = struct.unpack("f", f.read(4))[0]
                    z = struct.unpack("f", f.read(4))[0]
                    ray_r.append(r)
                    ray_z.append(z)

                rays.append(
                    {
                        "r": np.array(ray_r),
                        "z": np.array(ray_z),
                    }
                )

        except struct.error:
            pass  # End of file

    return rays


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
            lrecl = int(fid.readline().strip())

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
        raise RuntimeError(f"Error reading mode file {filename}: {e}")

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



def read_arrivals_asc(filepath: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read ASCII format arrivals data file written by BELLHOP or BELLHOP3D.

    This function reads ray arrival data including amplitude, phase, delay,
    angles, and bounce counts for each source-receiver combination.

    Parameters
    ----------
    filepath : str or Path
        Path to the ASCII format arrivals file (.arr extension).

    Returns
    -------
    arrivals : np.ndarray
        Structured array containing arrival data. Each element contains:
        - Narr : int
            Number of arrivals at this receiver
        - A : complex ndarray
            Complex amplitudes (amplitude * exp(1j * phase))
        - delay : complex ndarray
            Complex delays (real_delay + 1j * imag_delay) in seconds
        - SrcDeclAngle : float ndarray
            Source declination angles in degrees
        - RcvrDeclAngle : float ndarray
            Receiver declination angles in degrees
        - NumTopBnc : int ndarray
            Number of top boundary bounces
        - NumBotBnc : int ndarray
            Number of bottom boundary bounces
        For 3D files, also includes:
        - SrcAzimAngle : float ndarray
            Source azimuth angles in degrees
        - RcvrAzimAngle : float ndarray
            Receiver azimuth angles in degrees

        Array dimensions:
        - 2D: (Nrr, Nrz, Nsz)
        - 3D: (Nrr, Nrz, Ntheta, Nsz)

    pos : dict
        Position information containing:
        - 'freq' : float - Acoustic frequency in Hz
        - 's' : dict - Source positions
            - 'z' : ndarray - Source depths in meters
            - 'x' : ndarray - Source x-coordinates (3D only)
            - 'y' : ndarray - Source y-coordinates (3D only)
        - 'r' : dict - Receiver positions
            - 'z' : ndarray - Receiver depths in meters
            - 'r' : ndarray - Receiver ranges in km (2D) or meters (3D)
            - 'theta' : ndarray - Receiver bearings in degrees (3D only)

    Notes
    -----
    - The file must start with '2D' or '3D' flag to indicate format
    - Angles are stored in degrees
    - Complex amplitude: A = amplitude * exp(1j * phase_deg * pi/180)
    - Arrivals are organized by source, then receiver position

    References
    ----------
    Based on BELLHOP/read_arrivals_asc.m

    Examples
    --------
    >>> arr, pos = read_arrivals_asc('test.arr')
    >>> print(f"Frequency: {pos['freq']} Hz")
    >>> print(f"Arrivals at first receiver: {arr[0, 0, 0]['Narr']}")
    >>> if arr[0, 0, 0]['Narr'] > 0:
    ...     print(f"First arrival amplitude: {arr[0, 0, 0]['A'][0]}")
    ...     print(f"First arrival delay: {arr[0, 0, 0]['delay'][0]} s")
    """
    filepath = Path(filepath)
    with open(filepath, "r") as fid:
        # Read 2D/3D flag
        flag = fid.readline().strip().strip("'")

        if flag not in ["2D", "3D"]:
            raise ValueError(f"Not an ASCII format arrivals file (flag: {flag})")

        pos = {}

        if flag == "2D":
            # Read 2D format
            pos["freq"] = float(fid.readline().strip())

            # Source depths
            Nsz = int(fid.readline().strip())
            sz = np.array([float(fid.readline().strip()) for _ in range(Nsz)])
            pos["s"] = {"z": sz}

            # Receiver depths
            Nrz = int(fid.readline().strip())
            rz = np.array([float(fid.readline().strip()) for _ in range(Nrz)])

            # Receiver ranges
            Nrr = int(fid.readline().strip())
            rr = np.array([float(fid.readline().strip()) for _ in range(Nrr)])
            pos["r"] = {"z": rz, "r": rr}

            # Pre-allocate arrivals array
            arrivals = np.empty((Nrr, Nrz, Nsz), dtype=object)

            # Loop over sources
            for isd in range(Nsz):
                # Read max arrivals for this source (not used in Python)
                _ = int(fid.readline().strip())

                # Loop over receiver depths
                for irz in range(Nrz):
                    # Loop over receiver ranges
                    for irr in range(Nrr):
                        # Read number of arrivals at this receiver
                        Narr = int(fid.readline().strip())

                        arr_data = {
                            "Narr": Narr,
                            "A": np.array([], dtype=np.complex64),
                            "delay": np.array([], dtype=np.complex64),
                            "SrcDeclAngle": np.array([], dtype=np.float32),
                            "RcvrDeclAngle": np.array([], dtype=np.float32),
                            "NumTopBnc": np.array([], dtype=np.int16),
                            "NumBotBnc": np.array([], dtype=np.int16),
                        }

                        if Narr > 0:
                            # Read all arrivals (8 values per arrival)
                            data = []
                            for _ in range(Narr):
                                line_data = []
                                for _ in range(8):
                                    line_data.append(float(fid.readline().strip()))
                                data.append(line_data)
                            data = np.array(
                                data, dtype=np.float32
                            ).T  # Shape: (8, Narr)

                            # Parse arrival data
                            amplitude = data[0, :]
                            phase_deg = data[1, :]
                            arr_data["A"] = (
                                amplitude * np.exp(1j * phase_deg * np.pi / 180.0)
                            ).astype(np.complex64)

                            real_delay = data[2, :]
                            imag_delay = data[3, :]
                            arr_data["delay"] = (real_delay + 1j * imag_delay).astype(
                                np.complex64
                            )

                            arr_data["SrcDeclAngle"] = data[4, :]
                            arr_data["RcvrDeclAngle"] = data[5, :]
                            arr_data["NumTopBnc"] = data[6, :].astype(np.int16)
                            arr_data["NumBotBnc"] = data[7, :].astype(np.int16)

                        arrivals[irr, irz, isd] = arr_data

        else:  # 3D format
            # Read 3D format
            pos["freq"] = float(fid.readline().strip())

            # Source x-coordinates
            Nsx = int(fid.readline().strip())
            sx = np.array([float(fid.readline().strip()) for _ in range(Nsx)])

            # Source y-coordinates
            Nsy = int(fid.readline().strip())
            sy = np.array([float(fid.readline().strip()) for _ in range(Nsy)])

            # Source z-coordinates
            Nsz = int(fid.readline().strip())
            sz = np.array([float(fid.readline().strip()) for _ in range(Nsz)])
            pos["s"] = {"x": sx, "y": sy, "z": sz}

            # Receiver depths
            Nrz = int(fid.readline().strip())
            rz = np.array([float(fid.readline().strip()) for _ in range(Nrz)])

            # Receiver ranges
            Nrr = int(fid.readline().strip())
            rr = np.array([float(fid.readline().strip()) for _ in range(Nrr)])

            # Receiver bearings
            Ntheta = int(fid.readline().strip())
            theta = np.array([float(fid.readline().strip()) for _ in range(Ntheta)])
            pos["r"] = {"z": rz, "r": rr, "theta": theta}

            # Pre-allocate arrivals array
            arrivals = np.empty((Nrr, Nrz, Ntheta, Nsz), dtype=object)

            # Loop over sources
            for isd in range(Nsz):
                # Read max arrivals for this source (not used)
                _ = int(fid.readline().strip())

                # Loop over receiver bearings
                for itheta in range(Ntheta):
                    # Loop over receiver depths
                    for irz in range(Nrz):
                        # Loop over receiver ranges
                        for irr in range(Nrr):
                            # Read number of arrivals
                            Narr = int(fid.readline().strip())

                            arr_data = {
                                "Narr": Narr,
                                "A": np.array([], dtype=np.complex64),
                                "delay": np.array([], dtype=np.complex64),
                                "SrcDeclAngle": np.array([], dtype=np.float32),
                                "SrcAzimAngle": np.array([], dtype=np.float32),
                                "RcvrDeclAngle": np.array([], dtype=np.float32),
                                "RcvrAzimAngle": np.array([], dtype=np.float32),
                                "NumTopBnc": np.array([], dtype=np.int16),
                                "NumBotBnc": np.array([], dtype=np.int16),
                            }

                            if Narr > 0:
                                # Read all arrivals (10 values per arrival)
                                data = []
                                for _ in range(Narr):
                                    line_data = []
                                    for _ in range(10):
                                        line_data.append(float(fid.readline().strip()))
                                    data.append(line_data)
                                data = np.array(data, dtype=np.float32).T  # (10, Narr)

                                # Parse arrival data
                                amplitude = data[0, :]
                                phase_deg = data[1, :]
                                arr_data["A"] = (
                                    amplitude * np.exp(1j * phase_deg * np.pi / 180.0)
                                ).astype(np.complex64)

                                real_delay = data[2, :]
                                imag_delay = data[3, :]
                                arr_data["delay"] = (
                                    real_delay + 1j * imag_delay
                                ).astype(np.complex64)

                                arr_data["SrcDeclAngle"] = data[4, :]
                                arr_data["SrcAzimAngle"] = data[5, :]
                                arr_data["RcvrDeclAngle"] = data[6, :]
                                arr_data["RcvrAzimAngle"] = data[7, :]
                                arr_data["NumTopBnc"] = data[8, :].astype(np.int16)
                                arr_data["NumBotBnc"] = data[9, :].astype(np.int16)

                            arrivals[irr, irz, itheta, isd] = arr_data

    return arrivals, pos


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


def read_arrivals_bin(
    filename: str, marker_len: int = 1
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Read BINARY format arrivals data file written by BELLHOP or BELLHOP3D.

    This function reads ray arrival data from Fortran unformatted binary files,
    handling the record markers that different Fortran compilers insert.

    Parameters
    ----------
    filename : str
        Path to the binary format arrivals file (.arr extension).
    marker_len : int, optional
        Number of 4-byte words used as record markers at beginning and end
        of Fortran unformatted records. Default is 1 (standard for most
        compilers). If you get crashes or garbled results, try marker_len=2.

    Returns
    -------
    arrivals : np.ndarray
        Structured array containing arrival data. Each element contains:
        - Narr : int
            Number of arrivals at this receiver
        - A : complex ndarray
            Complex amplitudes (amplitude * exp(1j * phase))
        - delay : complex ndarray
            Complex delays (real_delay + 1j * imag_delay) in seconds
        - SrcDeclAngle : float ndarray
            Source declination angles in degrees
        - RcvrDeclAngle : float ndarray
            Receiver declination angles in degrees
        - NumTopBnc : int ndarray
            Number of top boundary bounces
        - NumBotBnc : int ndarray
            Number of bottom boundary bounces
        For 3D files, also includes:
        - SrcAzimAngle : float ndarray
            Source azimuth angles in degrees
        - RcvrAzimAngle : float ndarray
            Receiver azimuth angles in degrees

        Array dimensions:
        - 2D: (Nrr, Nrz, Nsz)
        - 3D: (Nrr, Nrz, Ntheta, Nsz)

    pos : dict
        Position information containing:
        - 'freq' : float - Acoustic frequency in Hz
        - 's' : dict - Source positions
            - 'z' : ndarray - Source depths in meters
            - 'x' : ndarray - Source x-coordinates (3D only)
            - 'y' : ndarray - Source y-coordinates (3D only)
        - 'r' : dict - Receiver positions
            - 'z' : ndarray - Receiver depths in meters
            - 'r' : ndarray - Receiver ranges in km (2D) or meters (3D)
            - 'theta' : ndarray - Receiver bearings in degrees (3D only)

    Notes
    -----
    - Binary files are more compact than ASCII format
    - The marker_len parameter depends on the Fortran compiler used:
        - Most compilers: marker_len=1 (default)
        - Some compilers (e.g., gfortran with certain flags): marker_len=2
    - File must start with '2D' or '3D' flag to indicate format
    - All data stored in single precision (float32) except ranges in 2D (float64)

    References
    ----------
    Based on BELLHOP/read_arrivals_bin.m

    Examples
    --------
    >>> arr, pos = read_arrivals_bin('test.arr')
    >>> print(f"Frequency: {pos['freq']} Hz")
    >>> print(f"Arrivals at first receiver: {arr[0, 0, 0]['Narr']}")

    If you get errors, try adjusting marker_len:
    >>> arr, pos = read_arrivals_bin('test.arr', marker_len=2)
    """
    with open(filename, "rb") as fid:
        # Skip first record marker and read 2D/3D flag
        fid.seek(4 * marker_len, 0)
        flag = fid.read(4).decode("ascii").strip("'")

        if flag not in ["2D", "3D"]:
            raise ValueError(f"Not a BINARY format arrivals file (flag: {flag})")

        pos = {}

        if flag == "2D":
            # Read 2D format
            fid.seek(8 * marker_len, 1)  # Skip two record markers (relative)
            pos["freq"] = np.fromfile(fid, dtype=np.float32, count=1)[0]

            # Source depths
            fid.seek(8 * marker_len, 1)
            Nsz = np.fromfile(fid, dtype=np.int32, count=1)[0]
            sz = np.fromfile(fid, dtype=np.float32, count=Nsz)
            pos["s"] = {"z": sz}

            # Receiver depths
            fid.seek(8 * marker_len, 1)
            Nrz = np.fromfile(fid, dtype=np.int32, count=1)[0]
            rz = np.fromfile(fid, dtype=np.float32, count=Nrz)

            # Receiver ranges (note: float64 in 2D)
            fid.seek(8 * marker_len, 1)
            Nrr = np.fromfile(fid, dtype=np.int32, count=1)[0]
            rr = np.fromfile(fid, dtype=np.float64, count=Nrr)
            pos["r"] = {"z": rz, "r": rr}

            # Pre-allocate arrivals array
            arrivals = np.empty((Nrr, Nrz, Nsz), dtype=object)

            # Loop over sources
            for isd in range(Nsz):
                # Read max arrivals for this source
                fid.seek(8 * marker_len, 1)
                _ = np.fromfile(fid, dtype=np.int32, count=1)[0]

                # Loop over receiver depths
                for irz in range(Nrz):
                    # Loop over receiver ranges
                    for irr in range(Nrr):
                        # Read number of arrivals
                        fid.seek(8 * marker_len, 1)
                        Narr = np.fromfile(fid, dtype=np.int32, count=1)[0]

                        arr_data = {
                            "Narr": Narr,
                            "A": np.array([], dtype=np.complex64),
                            "delay": np.array([], dtype=np.complex64),
                            "SrcDeclAngle": np.array([], dtype=np.float32),
                            "RcvrDeclAngle": np.array([], dtype=np.float32),
                            "NumTopBnc": np.array([], dtype=np.int16),
                            "NumBotBnc": np.array([], dtype=np.int16),
                        }

                        if Narr > 0:
                            # Read all arrivals (8 + 2*marker_len values per column)
                            data = np.fromfile(
                                fid, dtype=np.float32, count=(8 + 2 * marker_len) * Narr
                            ).reshape((8 + 2 * marker_len, Narr), order="F")

                            # Discard record markers
                            data = data[2 * marker_len :, :]

                            # Parse arrival data
                            amplitude = data[0, :]
                            phase_deg = data[1, :]
                            arr_data["A"] = (
                                amplitude * np.exp(1j * phase_deg * np.pi / 180.0)
                            ).astype(np.complex64)

                            real_delay = data[2, :]
                            imag_delay = data[3, :]
                            arr_data["delay"] = (real_delay + 1j * imag_delay).astype(
                                np.complex64
                            )

                            arr_data["SrcDeclAngle"] = data[4, :]
                            arr_data["RcvrDeclAngle"] = data[5, :]
                            arr_data["NumTopBnc"] = data[6, :].astype(np.int16)
                            arr_data["NumBotBnc"] = data[7, :].astype(np.int16)

                        arrivals[irr, irz, isd] = arr_data

        else:  # 3D format
            # Read 3D format
            fid.seek(8 * marker_len, 1)
            pos["freq"] = np.fromfile(fid, dtype=np.float32, count=1)[0]

            # Source x-coordinates
            fid.seek(8 * marker_len, 1)
            Nsx = np.fromfile(fid, dtype=np.int32, count=1)[0]
            sx = np.fromfile(fid, dtype=np.float32, count=Nsx)

            # Source y-coordinates
            fid.seek(8 * marker_len, 1)
            Nsy = np.fromfile(fid, dtype=np.int32, count=1)[0]
            sy = np.fromfile(fid, dtype=np.float32, count=Nsy)

            # Source z-coordinates
            fid.seek(8 * marker_len, 1)
            Nsz = np.fromfile(fid, dtype=np.int32, count=1)[0]
            sz = np.fromfile(fid, dtype=np.float32, count=Nsz)
            pos["s"] = {"x": sx, "y": sy, "z": sz}

            # Receiver depths
            fid.seek(8 * marker_len, 1)
            Nrz = np.fromfile(fid, dtype=np.int32, count=1)[0]
            rz = np.fromfile(fid, dtype=np.float32, count=Nrz)

            # Receiver ranges (float32 in 3D)
            fid.seek(8 * marker_len, 1)
            Nrr = np.fromfile(fid, dtype=np.int32, count=1)[0]
            rr = np.fromfile(fid, dtype=np.float32, count=Nrr)

            # Receiver bearings
            fid.seek(8 * marker_len, 1)
            Ntheta = np.fromfile(fid, dtype=np.int32, count=1)[0]
            theta = np.fromfile(fid, dtype=np.float32, count=Ntheta)
            pos["r"] = {"z": rz, "r": rr, "theta": theta}

            # Pre-allocate arrivals array
            arrivals = np.empty((Nrr, Nrz, Ntheta, Nsz), dtype=object)

            # Loop over sources
            for isd in range(Nsz):
                # Read max arrivals for this source
                fid.seek(8 * marker_len, 1)
                _ = np.fromfile(fid, dtype=np.int32, count=1)[0]

                # Loop over receiver bearings
                for itheta in range(Ntheta):
                    # Loop over receiver depths
                    for irz in range(Nrz):
                        # Loop over receiver ranges
                        for irr in range(Nrr):
                            # Read number of arrivals
                            fid.seek(8 * marker_len, 1)
                            Narr = np.fromfile(fid, dtype=np.int32, count=1)[0]

                            arr_data = {
                                "Narr": Narr,
                                "A": np.array([], dtype=np.complex64),
                                "delay": np.array([], dtype=np.complex64),
                                "SrcDeclAngle": np.array([], dtype=np.float32),
                                "SrcAzimAngle": np.array([], dtype=np.float32),
                                "RcvrDeclAngle": np.array([], dtype=np.float32),
                                "RcvrAzimAngle": np.array([], dtype=np.float32),
                                "NumTopBnc": np.array([], dtype=np.int16),
                                "NumBotBnc": np.array([], dtype=np.int16),
                            }

                            if Narr > 0:
                                # Read all arrivals (10 + 2*marker_len values per column)
                                data = np.fromfile(
                                    fid,
                                    dtype=np.float32,
                                    count=(10 + 2 * marker_len) * Narr,
                                ).reshape((10 + 2 * marker_len, Narr), order="F")

                                # Discard record markers
                                data = data[2 * marker_len :, :]

                                # Parse arrival data
                                amplitude = data[0, :]
                                phase_deg = data[1, :]
                                arr_data["A"] = (
                                    amplitude * np.exp(1j * phase_deg * np.pi / 180.0)
                                ).astype(np.complex64)

                                real_delay = data[2, :]
                                imag_delay = data[3, :]
                                arr_data["delay"] = (
                                    real_delay + 1j * imag_delay
                                ).astype(np.complex64)

                                arr_data["SrcDeclAngle"] = data[4, :]
                                arr_data["SrcAzimAngle"] = data[5, :]
                                arr_data["RcvrDeclAngle"] = data[6, :]
                                arr_data["RcvrAzimAngle"] = data[7, :]
                                arr_data["NumTopBnc"] = data[8, :].astype(np.int16)
                                arr_data["NumBotBnc"] = data[9, :].astype(np.int16)

                            arrivals[irr, irz, itheta, isd] = arr_data

    return arrivals, pos


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


def read_ssp_2d(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Read 2D sound speed profile file used by BELLHOP.

    Reads range-dependent SSP data where sound speed varies with both
    depth and range. Used for 2D propagation modeling.

    Parameters
    ----------
    filepath : str or Path
        Path to 2D SSP file (typically .ssp extension).

    Returns
    -------
    ssp_data : dict
        Dictionary containing:
        - 'n_prof' : int - Number of profiles (ranges)
        - 'r_prof' : ndarray - Range values in km, shape (n_prof,)
        - 'c_mat' : ndarray - Sound speed matrix in m/s, shape (n_depth, n_prof)
        - 'n_depth' : int - Number of depth points per profile

    Notes
    -----
    - File format:
        Line 1: NProf (number of range profiles)
        Line 2: r1 r2 ... rNProf (ranges in km)
        Lines 3+: Sound speeds for all profiles at each depth
                  (NProf values per line, NSSP lines total)

    - Sound speed matrix c_mat[i, j] gives speed at:
        - depth index i
        - range index j (profile j)

    References
    ----------
    Based on BELLHOP/readssp2d.m

    Examples
    --------
    >>> ssp = read_ssp_2d('range_dependent.ssp')
    >>> print(f"Number of profiles: {ssp['n_prof']}")
    >>> print(f"Ranges: {ssp['r_prof']} km")
    >>> print(f"SSP matrix shape: {ssp['c_mat'].shape}")
    >>> # Sound speed at depth index 10, range index 5
    >>> c = ssp['c_mat'][10, 5]
    """
    filepath = Path(filepath)
    with open(filepath, "r") as fid:
        # Read number of profiles
        n_prof = int(fid.readline().strip())

        # Read range values
        r_prof = np.array([float(fid.readline().strip()) for _ in range(n_prof)])

        # Read sound speed matrix
        # Read all remaining data and reshape
        remaining = fid.read().split()
        c_data = np.array([float(x) for x in remaining])

        # Reshape to (n_prof, n_depth), then transpose to (n_depth, n_prof)
        n_depth = len(c_data) // n_prof
        c_mat = c_data.reshape((n_prof, n_depth), order="F").T

    return {"n_prof": n_prof, "r_prof": r_prof, "c_mat": c_mat, "n_depth": n_depth}


def read_ssp_3d(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Read 3D sound speed profile file used by BELLHOP3D.

    Reads fully 3D SSP data where sound speed varies with x, y, and depth.
    Used for 3D propagation modeling.

    Parameters
    ----------
    filepath : str or Path
        Path to 3D SSP file (typically .ssp extension).

    Returns
    -------
    ssp_data : dict
        Dictionary containing:
        - 'Nx' : int - Number of x segments
        - 'Ny' : int - Number of y segments
        - 'Nz' : int - Number of depth points
        - 'Segx' : ndarray - X segment boundaries, shape (Nx,)
        - 'Segy' : ndarray - Y segment boundaries, shape (Ny,)
        - 'Segz' : ndarray - Depth values, shape (Nz,)
        - 'c_mat' : ndarray - Sound speed in m/s, shape (Nz, Ny, Nx)

    Notes
    -----
    - File format:
        Line 1: Nx
        Line 2: x1 x2 ... xNx (x segment boundaries)
        Line 3: Ny
        Line 4: y1 y2 ... yNy (y segment boundaries)
        Line 5: Nz
        Line 6: z1 z2 ... zNz (depth values)
        Lines 7+: For each depth iz=1:Nz:
                    Nx values for y1, then Nx values for y2, ..., Nx values for yNy

    - Sound speed accessed as c_mat[iz, iy, ix] for:
        - depth index iz
        - y index iy
        - x index ix

    References
    ----------
    Based on BELLHOP/readssp3d.m

    Examples
    --------
    >>> ssp = read_ssp_3d('seamount.ssp')
    >>> print(f"Grid size: {ssp['Nx']} x {ssp['Ny']} x {ssp['Nz']}")
    >>> print(f"X range: {ssp['Segx'][0]} to {ssp['Segx'][-1]}")
    >>> print(f"Y range: {ssp['Segy'][0]} to {ssp['Segy'][-1]}")
    >>> print(f"Depth range: {ssp['Segz'][0]} to {ssp['Segz'][-1]}")
    >>> # Sound speed at depth 10, y=5, x=3
    >>> c = ssp['c_mat'][10, 5, 3]
    """
    filepath = Path(filepath)
    with open(filepath, "r") as fid:
        # Read x segments
        Nx = int(fid.readline().strip())
        Segx = np.array([float(fid.readline().strip()) for _ in range(Nx)])

        # Read y segments
        Ny = int(fid.readline().strip())
        Segy = np.array([float(fid.readline().strip()) for _ in range(Ny)])

        # Read z (depth) values
        Nz = int(fid.readline().strip())
        Segz = np.array([float(fid.readline().strip()) for _ in range(Nz)])

        # Read sound speed data
        c_mat = np.zeros((Nz, Ny, Nx))

        for iz in range(Nz):
            # Read Nx*Ny values for this depth
            data = []
            for _ in range(Nx * Ny):
                data.append(float(fid.readline().strip()))

            # Reshape to (Nx, Ny) then transpose to (Ny, Nx)
            c_mat_2d = np.array(data).reshape((Nx, Ny), order="F").T
            c_mat[iz, :, :] = c_mat_2d

    return {
        "Nx": Nx,
        "Ny": Ny,
        "Nz": Nz,
        "Segx": Segx,
        "Segy": Segy,
        "Segz": Segz,
        "c_mat": c_mat,
    }


def read_shd_bin(
    filename: str,
    xs: Optional[float] = None,
    ys: Optional[float] = None,
    freq: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Read binary shade file (.shd) produced by acoustic models.

    Reads pressure field data from BELLHOP, KRAKEN, or other acoustic
    propagation models. Shade files contain 4D pressure fields with
    dimensions (theta, source_depth, receiver_depth, range).

    Parameters
    ----------
    filename : str
        Path to binary shade file (.shd extension).
    xs : float, optional
        Source x-coordinate in km. If provided with ys, reads only data
        for the source closest to (xs, ys). If None, reads first source.
    ys : float, optional
        Source y-coordinate in km. Required if xs is provided.
    freq : float, optional
        Frequency in Hz. If provided for broadband runs, selects closest
        frequency. If None, reads first frequency.

    Returns
    -------
    shd_data : dict
        Dictionary containing:
        - 'title' : str - Run title
        - 'PlotType' : str - Plot type ('rectilin', 'irregular', 'TL', etc.)
        - 'freqVec' : ndarray - Frequency vector in Hz
        - 'freq0' : float - Reference frequency
        - 'atten' : float - Attenuation parameter
        - 'Pos' : dict - Position data
            - 'theta' : ndarray - Bearing angles in degrees
            - 's' : dict - Source positions
                - 'x' : ndarray - X coordinates in meters
                - 'y' : ndarray - Y coordinates in meters
                - 'z' : ndarray - Depths in meters
            - 'r' : dict - Receiver positions
                - 'z' : ndarray - Depths in meters
                - 'r' : ndarray - Ranges in km
        - 'pressure' : ndarray - Complex pressure field
            Shape (Ntheta, Nsz, Nrz, Nrr) for rectilinear
            Shape (Ntheta, Nsz, 1, Nrr) for irregular

    Notes
    -----
    - File uses Fortran unformatted records with 4-byte record markers
    - Record length (recl) is read from first 4 bytes
    - Pressure is stored as interleaved real/imaginary pairs
    - For TL files from FIELD3D, source positions use compressed format
    - Coordinates: x,y in meters, z in meters, r in km, theta in degrees

    References
    ----------
    Based on BELLHOP/read_shd_bin.m by Chris Tiemann (2001)

    Examples
    --------
    >>> # Read first source
    >>> shd = read_shd_bin('pekeris.shd')
    >>> print(f"Title: {shd['title']}")
    >>> print(f"Pressure shape: {shd['pressure'].shape}")
    >>> print(f"Ranges: {shd['Pos']['r']['r']} km")

    >>> # Read specific source location
    >>> shd = read_shd_bin('field3d.shd', xs=5.0, ys=10.0)
    >>> # Pressure at first bearing, first source depth, first rcvr depth
    >>> p = shd['pressure'][0, 0, 0, :]

    >>> # Read specific frequency for broadband run
    >>> shd = read_shd_bin('broadband.shd', freq=100.0)
    """
    with open(filename, "rb") as fid:
        # Read record length
        recl = np.fromfile(fid, dtype=np.int32, count=1)[0]

        # Read title (80 characters)
        title_bytes = fid.read(80)
        title = title_bytes.decode("ascii", errors="ignore").strip()

        # Read plot type
        fid.seek(4 * recl, 0)
        plot_type_bytes = fid.read(10)
        PlotType = plot_type_bytes.decode("ascii", errors="ignore")

        # Read dimensions
        fid.seek(2 * 4 * recl, 0)
        Nfreq = np.fromfile(fid, dtype=np.int32, count=1)[0]
        Ntheta = np.fromfile(fid, dtype=np.int32, count=1)[0]
        Nsx = np.fromfile(fid, dtype=np.int32, count=1)[0]
        Nsy = np.fromfile(fid, dtype=np.int32, count=1)[0]
        Nsz = np.fromfile(fid, dtype=np.int32, count=1)[0]
        Nrz = np.fromfile(fid, dtype=np.int32, count=1)[0]
        Nrr = np.fromfile(fid, dtype=np.int32, count=1)[0]
        freq0 = np.fromfile(fid, dtype=np.float64, count=1)[0]
        atten = np.fromfile(fid, dtype=np.float64, count=1)[0]

        # Read frequency vector
        fid.seek(3 * 4 * recl, 0)
        freqVec = np.fromfile(fid, dtype=np.float64, count=Nfreq)

        # Read theta
        fid.seek(4 * 4 * recl, 0)
        theta = np.fromfile(fid, dtype=np.float64, count=Ntheta)

        # Read source positions
        if PlotType[:2] != "TL":
            # Full format
            fid.seek(5 * 4 * recl, 0)
            s_x = np.fromfile(fid, dtype=np.float64, count=Nsx)
            fid.seek(6 * 4 * recl, 0)
            s_y = np.fromfile(fid, dtype=np.float64, count=Nsy)
        else:
            # Compressed format for TL from FIELD3D
            fid.seek(5 * 4 * recl, 0)
            s_x_lim = np.fromfile(fid, dtype=np.float64, count=2)
            s_x = np.linspace(s_x_lim[0], s_x_lim[1], Nsx)
            fid.seek(6 * 4 * recl, 0)
            s_y_lim = np.fromfile(fid, dtype=np.float64, count=2)
            s_y = np.linspace(s_y_lim[0], s_y_lim[1], Nsy)

        # Read source and receiver depths
        fid.seek(7 * 4 * recl, 0)
        s_z = np.fromfile(fid, dtype=np.float32, count=Nsz)
        fid.seek(8 * 4 * recl, 0)
        r_z = np.fromfile(fid, dtype=np.float32, count=Nrz)

        # Read receiver ranges
        fid.seek(9 * 4 * recl, 0)
        r_r = np.fromfile(fid, dtype=np.float64, count=Nrr)

        # Determine pressure array shape
        if PlotType.strip() == "irregular":
            pressure = np.zeros((Ntheta, Nsz, 1, Nrr), dtype=np.complex64)
            Nrcvrs_per_range = 1
        else:
            pressure = np.zeros((Ntheta, Nsz, Nrz, Nrr), dtype=np.complex64)
            Nrcvrs_per_range = Nrz

        # Determine which source to read
        if xs is None:
            # Read all theta, sz, rz for first xs, ys
            idxX = 0
            idxY = 0

            # Get frequency index
            ifreq = 0
            if freq is not None:
                freq_diff = np.abs(freqVec - freq)
                ifreq = np.argmin(freq_diff)

            for itheta in range(Ntheta):
                for isz in range(Nsz):
                    for irz in range(Nrcvrs_per_range):
                        recnum = 10 + (
                            ifreq * Ntheta * Nsz * Nrcvrs_per_range
                            + itheta * Nsz * Nrcvrs_per_range
                            + isz * Nrcvrs_per_range
                            + irz
                        )

                        fid.seek(recnum * 4 * recl, 0)
                        temp = np.fromfile(fid, dtype=np.float32, count=2 * Nrr)
                        pressure[itheta, isz, irz, :] = temp[0::2] + 1j * temp[1::2]

        else:
            # Read for source at desired (xs, ys)
            if ys is None:
                raise ValueError("ys must be provided if xs is specified")

            # Find closest source
            x_diff = np.abs(s_x - xs * 1000.0)
            idxX = np.argmin(x_diff)
            y_diff = np.abs(s_y - ys * 1000.0)
            idxY = np.argmin(y_diff)

            for itheta in range(Ntheta):
                for isz in range(Nsz):
                    for irz in range(Nrcvrs_per_range):
                        recnum = 10 + (
                            idxX * Nsy * Ntheta * Nsz * Nrcvrs_per_range
                            + idxY * Ntheta * Nsz * Nrcvrs_per_range
                            + itheta * Nsz * Nrcvrs_per_range
                            + isz * Nrcvrs_per_range
                            + irz
                        )

                        fid.seek(recnum * 4 * recl, 0)
                        temp = np.fromfile(fid, dtype=np.float32, count=2 * Nrr)
                        pressure[itheta, isz, irz, :] = temp[0::2] + 1j * temp[1::2]

    return {
        "title": title,
        "PlotType": PlotType,
        "freqVec": freqVec,
        "freq0": freq0,
        "atten": atten,
        "Pos": {
            "theta": theta,
            "s": {"x": s_x, "y": s_y, "z": s_z},
            "r": {"z": r_z, "r": r_r},
        },
        "pressure": pressure,
    }


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


def read_shd_asc(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Read ASCII shade file produced by acoustic models.

    Parameters
    ----------
    filepath : str or Path
        Path to ASCII shade file (typically .shd.asc extension)

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'title': str, plot title
        - 'plot_type': str, plot type
        - 'freq_vec': ndarray, frequency vector in Hz
        - 'freq0': float, reference frequency in Hz
        - 'atten': float, stabilizing attenuation in dB/wavelength
        - 'pos': dict with position information:
          - 'theta': ndarray, bearing angles in degrees
          - 's': dict with 'z' (source depths in m)
          - 'r': dict with 'z' (receiver depths in m), 'r' (ranges in m)
        - 'pressure': ndarray, complex pressure field, shape (n_depth, n_range)

    Notes
    -----
    ASCII shade file format:
    - Line 1: Title
    - Line 2: Plot type
    - Line 3: Nfreq, Ntheta, Nsd, Nrd, Nrr, freq0, atten
    - Frequency vector
    - Bearing angles
    - Source depths
    - Receiver depths
    - Receiver ranges
    - Pressure data (interleaved real/imaginary)

    Currently only reads data for first source depth.

    Translated from OALIB read_shd_asc.m

    Examples
    --------
    >>> # Read ASCII shade file
    >>> # data = read_shd_asc('test.shd.asc')
    >>> # print(f"Pressure field shape: {data['pressure'].shape}")
    >>> # print(f"Frequency: {data['freq0']} Hz")
    """
    filepath = Path(filepath)
    # Open file
    try:
        fid = open(filepath, "r")
    except FileNotFoundError:
        raise FileNotFoundError(
            "No shade file with that name exists; you must run a model first"
        )

    # Read header
    title = fid.readline().strip()
    plot_type = fid.readline().strip()

    line = fid.readline()
    vals = np.fromstring(line, sep=" ")
    Nfreq = int(vals[0])
    Ntheta = int(vals[1])
    Nsd = int(vals[2])
    Nrd = int(vals[3])
    Nrr = int(vals[4])

    line = fid.readline()
    vals = np.fromstring(line, sep=" ", count=2)
    freq0 = vals[0]
    atten = vals[1]

    # Read vectors
    freq_vec = np.zeros(Nfreq)
    for i in range(Nfreq):
        freq_vec[i] = float(fid.readline().strip())

    theta = np.zeros(Ntheta)
    for i in range(Ntheta):
        theta[i] = float(fid.readline().strip())

    s_z = np.zeros(Nsd)
    for i in range(Nsd):
        s_z[i] = float(fid.readline().strip())

    r_z = np.zeros(Nrd)
    for i in range(Nrd):
        r_z[i] = float(fid.readline().strip())

    r_r = np.zeros(Nrr)
    for i in range(Nrr):
        r_r[i] = float(fid.readline().strip())

    # Read pressure data (only first source depth)
    # Data is interleaved: real1 imag1 real2 imag2 ...
    # Array is (2*Nrr, Nrd)
    temp = np.zeros((2 * Nrr, Nrd))
    for j in range(Nrd):
        for i in range(2 * Nrr):
            temp[i, j] = float(fid.readline().strip())

    fid.close()

    # Extract real and imaginary parts
    # temp[0::2, :] = real parts
    # temp[1::2, :] = imaginary parts
    pressure = temp[0::2, :].T + 1j * temp[1::2, :].T

    return {
        "title": title,
        "plot_type": plot_type,
        "freq_vec": freq_vec,
        "freq0": freq0,
        "atten": atten,
        "pos": {"theta": theta, "s": {"z": s_z}, "r": {"z": r_z, "r": r_r}},
        "pressure": pressure,
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


def read_ram_tlgrid(filepath: Union[str, Path] = "tl.grid") -> Dict[str, Any]:
    """
    Read binary transmission loss grid file from RAM (Range-dependent Acoustic Model).

    Parameters
    ----------
    filepath : str or Path, optional
        Path to RAM tl.grid file (default: 'tl.grid')

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'title': str, always 'RAM'
        - 'plot_type': str, always 'rectilin'
        - 'freq': float, frequency in Hz
        - 'atten': float, always 0.0
        - 'pos': dict with:
          - 's': dict with 'z' (source depth in m)
          - 'r': dict with 'z' (receiver depths in m), 'r' (ranges in m)
        - 'pressure': ndarray, complex pressure field, shape (1, 1, n_depth, n_range)
          converted from TL (dB) to pressure amplitude

    Notes
    -----
    RAM uses Fortran unformatted sequential files with record markers.
    This function auto-detects the record marker datatype (32-bit or 64-bit).

    File format:
    - Header record (60 bytes): freq, zs, zr, rmax, dr, ndr, zmax, dz, ndz,
      zmplt, c0, np, ns, rs, lz
    - TL matrix records: TL(lz, lr) in dB

    TL is converted to pressure amplitude:
        pressure = 10^(-TL/20)

    Translated from OALIB read_ram_tlgrid.m

    Examples
    --------
    >>> # Read RAM output
    >>> # data = read_ram_tlgrid('tl.grid')
    >>> # print(f"Frequency: {data['freq']} Hz")
    >>> # print(f"Pressure field shape: {data['pressure'].shape}")
    >>> # TL = -20 * np.log10(np.abs(data['pressure'][0, 0, :, :]))
    """
    filepath = Path(filepath)
    # Open binary file
    try:
        fid = open(filepath, "rb")
    except FileNotFoundError:
        raise FileNotFoundError(f"read_ram_tlgrid: unable to open {filepath}")

    # Auto-detect record marker datatype
    # Try 64-bit first
    rlen = np.fromfile(fid, dtype=np.uint64, count=1)

    if len(rlen) > 0 and rlen[0] == 60:
        rm_dtype = np.uint64
    else:
        # Try 32-bit
        fid.seek(0)
        rlen = np.fromfile(fid, dtype=np.uint32, count=1)
        if len(rlen) > 0 and rlen[0] == 60:
            rm_dtype = np.uint32
        else:
            fid.close()
            raise ValueError(
                "read_ram_tlgrid: error determining datatype of record markers"
            )

    # Read header (60 bytes after record marker)
    freq = np.fromfile(fid, dtype=np.float32, count=1)[0]
    zs = np.fromfile(fid, dtype=np.float32, count=1)[0]
    zr = np.fromfile(fid, dtype=np.float32, count=1)[0]
    rmax = np.fromfile(fid, dtype=np.float32, count=1)[0]
    dr = np.fromfile(fid, dtype=np.float32, count=1)[0]
    ndr = np.fromfile(fid, dtype=np.int32, count=1)[0]
    zmax = np.fromfile(fid, dtype=np.float32, count=1)[0]
    dz = np.fromfile(fid, dtype=np.float32, count=1)[0]
    ndz = np.fromfile(fid, dtype=np.int32, count=1)[0]
    zmplt = np.fromfile(fid, dtype=np.float32, count=1)[0]
    c0 = np.fromfile(fid, dtype=np.float32, count=1)[0]
    np_ram = np.fromfile(fid, dtype=np.int32, count=1)[0]
    ns = np.fromfile(fid, dtype=np.int32, count=1)[0]
    rs = np.fromfile(fid, dtype=np.float32, count=1)[0]
    lz = np.fromfile(fid, dtype=np.int32, count=1)[0]
    rlen = np.fromfile(fid, dtype=rm_dtype, count=1)  # Closing record marker

    # Read TL matrix
    lr = int(np.floor(rmax / (dr * ndr)))
    TL = np.zeros((lz, lr))

    for j in range(lr):
        rlen = np.fromfile(fid, dtype=rm_dtype, count=1)  # Opening marker
        TL[:, j] = np.fromfile(fid, dtype=np.float32, count=lz)
        rlen = np.fromfile(fid, dtype=rm_dtype, count=1)  # Closing marker

    fid.close()

    # Convert to standard Acoustics Toolbox format
    # Receiver depths
    r_z = np.arange(1, lz + 1) * ndz * dz

    # Receiver ranges
    r_r = np.arange(1, lr + 1) * ndr * dr

    # Convert TL (dB) to pressure amplitude
    # TL = -20 * log10(|p|)  =>  |p| = 10^(-TL/20)
    pressure = 10.0 ** (-TL / 20.0)

    # Reshape to 4D array (Ntheta=1, Nsd=1, Nrd=lz, Nrr=lr)
    pressure = pressure[np.newaxis, np.newaxis, :, :]

    return {
        "title": "RAM",
        "plot_type": "rectilin",
        "freq": freq,
        "atten": 0.0,
        "pos": {"s": {"z": zs}, "r": {"z": r_z, "r": r_r}},
        "pressure": pressure,
    }
