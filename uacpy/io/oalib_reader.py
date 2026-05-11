"""
Acoustics Toolbox / OALIB output-file readers.

One file per shared output format (Kraken, Scooter, SPARC, Bounce, KrakenField,
Bellhop output formats; modes are kept in their own ``modes_reader.py``).

Provides:

* ``.shd`` — :func:`read_shd_file`, :func:`read_shd_bin`, :func:`read_shd_asc`
* ``.arr`` — :func:`read_arr_file` (Bellhop arrivals, ASCII)
* ``.ray`` — :func:`read_ray_file` (Bellhop rays, ASCII or binary)
* ``.ssp`` — :func:`read_ssp_2d`, :func:`read_ssp_3d`
* ``.flp`` — :func:`read_flp`, :func:`read_flp3d`
* ``.rts`` — :func:`read_rts_file`, :func:`rts_to_pressure` (SPARC time series, binary)
* ``.ts``  — :func:`read_ts` (generic time series, ASCII)
"""

import numpy as np
import struct
import warnings
from pathlib import Path
from typing import Union, Tuple, Dict, Any, Optional

from uacpy._log import log_message
from uacpy.core.results import PressureField, Arrivals, Rays
from uacpy.io._fortran_helpers import read_vector as _read_vector


def read_shd_file(filepath: Union[str, Path]) -> PressureField:
    """Read a single-frequency ``.shd`` file as a complex :class:`PressureField`.

    Thin wrapper around :func:`read_shd_bin` that returns the first
    bearing / first source slice of the complex pressure cube as a
    ``units='complex'`` PressureField. Use ``field.tl`` (or
    ``field.to_tl()``) to materialise transmission loss in dB.
    Multi-frequency ``.shd`` files raise ``ValueError`` — call
    ``read_shd_bin`` directly and build a :class:`TransferFunction` from
    its complex pressure cube instead.
    """
    filepath = Path(filepath)
    shd = read_shd_bin(str(filepath))

    freqs = np.asarray(shd['freqVec'], dtype=float)
    nfreq = len(freqs)
    if nfreq == 0:
        raise ValueError(
            f"read_shd_file: {filepath} declares zero frequencies; the .shd "
            f"file is malformed (every Acoustics-Toolbox writer emits at "
            f"least one frequency record)."
        )
    if nfreq > 1:
        raise ValueError(
            f"read_shd_file: {filepath} contains {nfreq} frequencies; "
            "use read_shd_bin(filepath) for the full broadband payload "
            "and construct a TransferFunction from it."
        )

    pressure = shd['pressure']               # (Ntheta, Nsz, Nrz, Nrr)
    p = pressure[0, 0, :, :]                 # first bearing, first source

    pos = shd['Pos']
    return PressureField(
        units="complex",
        data=p,
        ranges=pos['r']['r'],
        depths=pos['r']['z'],
        model='', backend='',
        source_depths=pos['s']['z'],
        frequencies=freqs,
        metadata={
            'title': shd['title'],
            'plot_type': shd['PlotType'],
            'source_file': str(filepath),
            'freq0': shd['freq0'],
            'atten': shd['atten'],
        },
    )


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
                - 'r' : ndarray - Ranges in meters (Acoustics-Toolbox
                  converts km → m before WriteHeader; see
                  SourceReceiverPositions.f90:277)
        - 'pressure' : ndarray - Complex pressure field
            Shape (Ntheta, Nsz, Nrz, Nrr) for rectilinear
            Shape (Ntheta, Nsz, 1, Nrr) for irregular

    Notes
    -----
    - File uses Fortran unformatted records with 4-byte record markers
    - Record length (recl) is read from first 4 bytes
    - Pressure is stored as interleaved real/imaginary pairs
    - For TL files from FIELD3D, source positions use compressed format
    - Coordinates: x,y in meters, z in meters, r in meters, theta in degrees

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
        recl = np.fromfile(fid, dtype=np.int32, count=1)[0]
        title_bytes = fid.read(80)
        title = title_bytes.decode("ascii", errors="ignore").strip()
        fid.seek(4 * recl, 0)
        plot_type_bytes = fid.read(10)
        PlotType = plot_type_bytes.decode("ascii", errors="ignore")
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
        fid.seek(3 * 4 * recl, 0)
        freqVec = np.fromfile(fid, dtype=np.float64, count=Nfreq)
        fid.seek(4 * 4 * recl, 0)
        theta = np.fromfile(fid, dtype=np.float64, count=Ntheta)
        if PlotType[:2] != "TL":
            fid.seek(5 * 4 * recl, 0)
            s_x = np.fromfile(fid, dtype=np.float64, count=Nsx)
            fid.seek(6 * 4 * recl, 0)
            s_y = np.fromfile(fid, dtype=np.float64, count=Nsy)
        else:
            fid.seek(5 * 4 * recl, 0)
            s_x_lim = np.fromfile(fid, dtype=np.float64, count=2)
            s_x = np.linspace(s_x_lim[0], s_x_lim[1], Nsx)
            fid.seek(6 * 4 * recl, 0)
            s_y_lim = np.fromfile(fid, dtype=np.float64, count=2)
            s_y = np.linspace(s_y_lim[0], s_y_lim[1], Nsy)
        fid.seek(7 * 4 * recl, 0)
        s_z = np.fromfile(fid, dtype=np.float32, count=Nsz)
        fid.seek(8 * 4 * recl, 0)
        r_z = np.fromfile(fid, dtype=np.float32, count=Nrz)
        fid.seek(9 * 4 * recl, 0)
        r_r = np.fromfile(fid, dtype=np.float64, count=Nrr)
        if PlotType.strip() == "irregular":
            pressure = np.zeros((Ntheta, Nsz, 1, Nrr), dtype=np.complex64)
            Nrcvrs_per_range = 1
        else:
            pressure = np.zeros((Ntheta, Nsz, Nrz, Nrr), dtype=np.complex64)
            Nrcvrs_per_range = Nrz
        if xs is None:
            idxX = 0
            idxY = 0
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
            if ys is None:
                raise ValueError("ys must be provided if xs is specified")
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
    try:
        fid = open(filepath, "r")
    except FileNotFoundError:
        raise FileNotFoundError(
            "No shade file with that name exists; you must run a model first"
        )
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
    temp = np.zeros((2 * Nrr, Nrd))
    for j in range(Nrd):
        for i in range(2 * Nrr):
            temp[i, j] = float(fid.readline().strip())

    fid.close()
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


def read_arr_file(filepath: Union[str, Path]):
    """
    Read arrivals file (.arr) from Bellhop

    Parameters
    ----------
    filepath : str or Path
        Path to .arr file

    Returns
    -------
    Arrivals
        Typed result with:
        - ``by_receiver``: nested list ``[isd][ird][irr]`` of per-receiver
          arrival dicts (amplitudes, phases, delays, delay_imag,
          src_angles, rcv_angles, n_top_bounces, n_bot_bounces).
        - ``arrivals``: flat list of per-arrival records (same data,
          un-nested) for filter/top_n/in_window chain methods.
        - ``frequencies``, ``source_depths``, ``receiver_depths``,
          ``receiver_ranges`` as typed attributes.
    """
    filepath = Path(filepath)

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
            from uacpy.core.exceptions import ConfigurationError
            raise ConfigurationError(
                "Binary arrivals format (.arr written by RunType 'a') is "
                "not supported. Re-run Bellhop with "
                "arrivals_format='ascii' (RunType 'A'). See "
                "ArrMod.f90:WriteArrivalsBinary for the record layout."
            )

        f.seek(0)
        text_content = f.read(10).decode('ascii', errors='ignore')

        if "'2D'" in text_content:
            is_3d = False
        elif "'3D'" in text_content:
            is_3d = True
        else:
            raise ValueError(f"Not a valid arrivals file: {repr(text_content)}")

        if not is_3d:
            f.close()
            with open(filepath, 'r') as f:
                lines = f.readlines()
                idx = 1  # skip flag line

                freq = float(lines[idx].strip())
                idx += 1

                tokens = lines[idx].strip().split()
                nsd = int(tokens[0])
                sz = np.array([float(t) for t in tokens[1:1+nsd]])
                idx += 1

                tokens = lines[idx].strip().split()
                nrd = int(tokens[0])
                rz = np.array([float(t) for t in tokens[1:1+nrd]])
                idx += 1

                tokens = lines[idx].strip().split()
                nrr = int(tokens[0])
                rr = np.array([float(t) for t in tokens[1:1+nrr]])
                idx += 1

                arrivals_by_receiver = []

                for isd in range(nsd):
                    sd_list = []
                    # Skip the per-source max-narr line.
                    int(lines[idx].strip())
                    idx += 1

                    for irz in range(nrd):
                        rd_list = []
                        for irr in range(nrr):
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
            raise NotImplementedError("3D arrivals format not yet implemented")

    return Arrivals(
        by_receiver=arrivals_by_receiver,
        receiver_depths=rz,
        receiver_ranges=rr,
        model='', backend='',
        source_depths=sz,
        frequencies=float(freq),
        metadata={},
    )


def read_ray_file(filepath: Union[str, Path]):
    """
    Read ray file (.ray) from Bellhop

    Parameters
    ----------
    filepath : str or Path
        Path to .ray file

    Returns
    -------
    Rays
        Typed result with the parsed ray paths on ``.rays``.
    """
    filepath = Path(filepath)

    rays = []

    try:
        with open(filepath, "r") as f:
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            f.readline().strip()
            while True:
                angle_line = f.readline()
                if not angle_line:
                    break
                if not angle_line.strip():
                    continue
                try:
                    alpha = float(angle_line.strip())
                except ValueError:
                    continue
                counts_line = f.readline()
                if not counts_line:
                    break

                counts = counts_line.split()
                if len(counts) < 1:
                    continue

                n_points = int(counts[0])
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
                            "n_top_bounces": n_top_bounces,
                            "n_bot_bounces": n_bot_bounces,
                        }
                    )

    except (UnicodeDecodeError, ValueError) as e:
        # File is not valid ASCII or has malformed numeric content;
        # the binary reader is the legitimate fallback for these.
        warnings.warn(
            f"read_ray_file: ASCII parse failed ({type(e).__name__}: {e}); "
            f"falling back to binary reader for {filepath}",
            UserWarning, stacklevel=2,
        )
        rays = _read_ray_file_binary(filepath)

    return Rays(rays=rays, model='', backend='')


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
        recl = struct.unpack("i", f.read(4))[0]
        f.seek(recl * 4)

        truncated_after = None
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
            truncated_after = len(rays)

    if truncated_after is not None:
        warnings.warn(
            f"Ray file {filepath} appears truncated; recovered "
            f"{truncated_after} ray(s) before EOF. The producing model "
            "may have crashed mid-write — check its .prt log.",
            UserWarning,
            stacklevel=2,
        )
    return rays


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
        n_prof = int(fid.readline().strip())
        r_prof = np.array([float(fid.readline().strip()) for _ in range(n_prof)])
        remaining = fid.read().split()
        c_data = np.array([float(x) for x in remaining])
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
        Nx = int(fid.readline().strip())
        Segx = np.array([float(fid.readline().strip()) for _ in range(Nx)])
        Ny = int(fid.readline().strip())
        Segy = np.array([float(fid.readline().strip()) for _ in range(Ny)])
        Nz = int(fid.readline().strip())
        Segz = np.array([float(fid.readline().strip()) for _ in range(Nz)])
        c_mat = np.zeros((Nz, Ny, Nx))

        for iz in range(Nz):
            data = []
            for _ in range(Nx * Ny):
                data.append(float(fid.readline().strip()))
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


def read_flp(fileroot: Union[str, Path], verbose: bool = False) -> Dict[str, Any]:
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
        title = f.readline().strip()
        if "'" in title:
            start = title.find("'") + 1
            end = title.find("'", start)
            title = title[start:end]
        log_message('oalib_reader', f"Title: {title}", verbose=verbose)
        opt = f.readline().strip()
        if "'" in opt:
            start = opt.find("'") + 1
            end = opt.find("'", start)
            opt = opt[start:end]
        log_message('oalib_reader', f"Options: {opt}", verbose=verbose)

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
        M_limit = int(f.readline().strip())
        log_message('oalib_reader', f"MLimit = {M_limit}", verbose=verbose)
        r_prof, N_prof = _read_vector(f)
        log_message('oalib_reader', f"Number of profiles, NProf = {N_prof}",
                    verbose=verbose)
        if N_prof < 10:
            preview = ", ".join(f"{r:.2f}" for r in r_prof)
        else:
            preview = f"{r_prof[0]:.2f} … {r_prof[-1]:.2f}"
        log_message('oalib_reader', f"profile ranges rProf (km): {preview}",
                    verbose=verbose, level='debug')

        r_rcv, _ = _read_vector(f)
        r_rcv = r_rcv * 1000.0  # km → m
        pos_temp = _read_sz_rz(f)
        r_offsets, N_offsets = _read_vector(f)

        log_message('oalib_reader',
                    f"Number of receiver range offsets = {N_offsets}",
                    verbose=verbose)
        if N_offsets < 10:
            preview = ", ".join(f"{ro:.2f}" for ro in r_offsets)
        else:
            preview = f"{r_offsets[0]:.2f} … {r_offsets[-1]:.2f}"
        log_message('oalib_reader',
                    f"receiver range offsets Rro (m): {preview}",
                    verbose=verbose, level='debug')

        if np.max(np.abs(r_offsets)) > 0.0:
            warnings.warn(
                "read_flp: receiver range offsets are not zero — "
                "result includes array-tilt geometry.",
                UserWarning, stacklevel=2,
            )

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
        title = f.readline().strip()
        if "'" in title:
            start = title.find("'") + 1
            end = title.find("'", start)
            title = title[start:end]
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
        M_limit = int(f.readline().strip())
        r_prof, N_r_prof = _read_vector(f)
        theta_prof, N_theta_prof = _read_vector(f)
        r_rcv, _ = _read_vector(f)
        theta_rcv, _ = _read_vector(f)
        pos_temp = _read_sz_rz(f)
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


def _read_sz_rz(fid) -> Dict[str, np.ndarray]:
    """
    Read source and receiver depths.

    Helper function for read_flp.
    """
    sz, _ = _read_vector(fid)
    rz, _ = _read_vector(fid)

    return {"sz": sz, "rz": rz}


def read_rts_file(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Read SPARC time series file (.rts).

    SPARC computes pressure time series at receiver locations.
    This data must be transformed to frequency domain for TL calculations.

    Parameters
    ----------
    filepath : str or Path
        Path to .rts file

    Returns
    -------
    rts_data : dict
        Dictionary containing:
        - 'title': Run title
        - 'dt': Time step in seconds
        - 'nt': Number of time samples
        - 'nr': Number of ranges/depths
        - 'ranges': Range/depth vector (m)
        - 'time': Time vector (s)
        - 'p': Pressure time series, shape (nt, nr)

    Notes
    -----
    SPARC outputs time-domain pressure fields which must be FFT'd
    to extract a frequency-domain pressure. The RTS file does NOT
    store the analysis frequency; callers must pass it explicitly to
    :func:`rts_to_pressure`.

    File format is Fortran ASCII (FORMATTED), written by SPARC's output
    routine (``Scooter/sparc.f90``):

    - Line 1: Title, enclosed in single quotes.
    - Subsequent whitespace-separated token stream:
        * token 0: NRr (or NRz in vertical-array mode), an integer.
        * tokens 1..NRr: range (or depth) values in metres.
        * then repeating blocks of ``1 + NRr`` tokens:
          ``t, p(r_1, t), ..., p(r_NRr, t)``.

    Fortran writes these with ``12G15.6`` formatting, so the tokens wrap
    to a new line every 12 values. The parser tokenises the whole stream
    and is therefore insensitive to line wrapping.
    """
    filepath = Path(filepath)

    # Tokenize the entire file. Fortran's 12G15.6 format wraps at 12
    # values per line, so NRr > 12 causes the range vector to span
    # multiple lines. Flattening the whole stream and walking token by
    # token makes parsing independent of line wrapping.
    with open(filepath, "r") as f:
        line1 = f.readline().strip()
        if line1.startswith("'") and line1.endswith("'"):
            title = line1[1:-1]
        else:
            title = line1
        raw_tokens = []
        for line in f:
            raw_tokens.extend(line.strip().split())

    if not raw_tokens:
        raise ValueError(f"RTS file {filepath} appears empty after the title line")

    # First token is NRr/NRz, then exactly NRr range/depth floats.
    nr = int(raw_tokens[0])
    if len(raw_tokens) < 1 + nr:
        raise ValueError(
            f"RTS file {filepath} truncated: expected {nr} range/depth values, "
            f"only {len(raw_tokens) - 1} tokens available after count."
        )
    ranges = np.array([float(x) for x in raw_tokens[1:1 + nr]])

    # Remaining tokens are time-series records: (1 time + nr pressures) per step.
    rest = raw_tokens[1 + nr:]
    values_per_timestep = 1 + nr
    nt = len(rest) // values_per_timestep

    time_list = []
    pressure_list = []
    for i in range(nt):
        start_idx = i * values_per_timestep
        time_list.append(float(rest[start_idx]))
        pressure_list.append([float(x) for x in rest[start_idx + 1:start_idx + 1 + nr]])

    time = np.array(time_list)
    p = np.array(pressure_list)  # shape (nt, nr)
    if nt > 1:
        dt = time[1] - time[0]
    else:
        dt = 0.0

    return {
        "title": title,
        "dt": dt,
        "nt": nt,
        "nr": nr,
        "ranges": ranges,
        "time": time,
        "p": p,
    }


def rts_to_pressure(
    rts_data: Dict[str, Any], freq: float, method: str = "fft",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project SPARC time-series data onto complex pressure at one frequency.

    ``method='fft'`` extracts the spectral bin nearest ``freq`` from a
    Hanning-windowed FFT; ``method='goertzel'`` uses the Goertzel
    single-bin DFT. Both return ``(p_at_freq, ranges)`` where
    ``p_at_freq`` is the model-native, source-normalised complex pressure
    (suitable for ``PressureField(units='complex',
    phase_reference='travelling_wave')``).

    Used by :class:`uacpy.models.SPARC` to project the native
    time-domain pressure onto a steady-state field at the source frequency.
    """
    p = rts_data["p"]
    dt = rts_data["dt"]
    ranges = rts_data["ranges"]

    nt, nr = p.shape

    if method == "fft":
        window = np.hanning(nt)
        p_freq = np.fft.rfft(p * window[:, np.newaxis], axis=0)
        freqs = np.fft.rfftfreq(nt, dt)
        freq_idx = np.argmin(np.abs(freqs - freq))
        p_at_freq = 2.0 * p_freq[freq_idx, :] / np.sum(window)
    elif method == "goertzel":
        omega = 2 * np.pi * freq
        coeff = 2 * np.cos(omega * dt)
        p_at_freq = np.zeros(nr, dtype=complex)
        for ir in range(nr):
            s0 = 0.0
            s1 = 0.0
            s2 = 0.0
            for it in range(nt):
                s0 = p[it, ir] + coeff * s1 - s2
                s2 = s1
                s1 = s0
            p_at_freq[ir] = s0 - s1 * np.exp(-1j * omega * dt)
        p_at_freq = 2.0 * p_at_freq / nt
    else:
        raise ValueError(
            f"rts_to_pressure: unknown method {method!r}; "
            f"use 'fft' or 'goertzel'."
        )

    return p_at_freq, ranges


def read_ts(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Read time-series file from acoustic models.

    This is a simple ASCII time series format, different from the binary
    RTS format used by SPARC. Used by some AT models for time-domain output.

    Parameters
    ----------
    filepath : str or Path
        Path to time series file

    Returns
    -------
    ts_data : dict
        Dictionary containing:
        - 'PlotTitle': str - Plot title
        - 'pos': dict with 'r': {'z': receiver depths (m)}
        - 'tout': ndarray - Time vector (s), shape (nt,)
        - 'RTS': ndarray - Time series data, shape (nt, nrd)
          RTS[it, ird] is pressure at time tout[it], depth pos['r']['z'][ird]

    Notes
    -----
    File format:
    - Line 1: Plot title
    - Line 2: nrd (number of receiver depths)
    - Line 3: rd values (receiver depths in m)
    - Following lines: time RTS[0,:] RTS[1,:] ... (nt rows)
      First column is time, remaining columns are RTS values at each depth

    This format is simpler than the binary .rts format used by SPARC.

    Translated from OALIB read_ts.m

    Examples
    --------
    >>> ts = read_ts('timeseries.txt')
    >>> print(f"Time range: {ts['tout'][0]:.3f} to {ts['tout'][-1]:.3f} s")
    >>> print(f"Receiver depths: {ts['pos']['r']['z']}")
    >>> print(f"Time series shape: {ts['RTS'].shape}")

    >>> # Plot time series at first depth
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(ts['tout'], ts['RTS'][:, 0])
    >>> plt.xlabel('Time (s)')
    >>> plt.ylabel('Pressure')
    >>> plt.title(f"Depth = {ts['pos']['r']['z'][0]} m")

    See Also
    --------
    read_rts_file : Read binary RTS format from SPARC
    """
    filepath = Path(filepath)
    if filepath.suffix == '.mat':
        import scipy.io
        mat_data = scipy.io.loadmat(str(filepath))
        return {
            'PlotTitle': str(mat_data.get('PlotTitle', [''])[0]),
            'pos': {'r': {'z': mat_data['Pos'][0, 0]['r'][0, 0]['z'].ravel()}},
            'tout': mat_data['tout'].ravel(),
            'RTS': mat_data['RTS'].T  # MATLAB stores transposed
        }
    with open(filepath, 'r') as f:
        plot_title = f.readline().strip()
        nrd = int(f.readline().strip())
        rd = np.array([float(x) for x in f.readline().strip().split()])

        if len(rd) != nrd:
            raise ValueError(f"Expected {nrd} receiver depths, got {len(rd)}")
        data = []
        for line in f:
            line = line.strip()
            if line:
                values = [float(x) for x in line.split()]
                data.append(values)

    data = np.array(data)
    tout = data[:, 0]
    RTS = data[:, 1:nrd+1]

    return {
        'PlotTitle': plot_title,
        'pos': {'r': {'z': rd}},
        'tout': tout,
        'RTS': RTS
    }
