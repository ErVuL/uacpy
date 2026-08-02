"""
Acoustics Toolbox / OALIB output-file readers.

One file per shared output format (Kraken, Scooter, SPARC, Bounce,
Bellhop output formats; modes are kept in their own ``modes_reader.py``).

Provides:

* ``.shd`` — :func:`read_shd_file`, :func:`read_shd_bin`, :func:`read_shd_asc`
* ``.arr`` — :func:`read_arr_file` (Bellhop arrivals, ASCII)
* ``.ray`` — :func:`read_ray_file` (Bellhop rays, ASCII or binary)
* ``.ssp`` — :func:`read_ssp_2d`, :func:`read_ssp_3d`
* ``.flp`` — :func:`read_flp`, :func:`read_flp3d`
* ``.rts`` — :func:`read_rts_file`, :func:`rts_to_pressure` (SPARC time series, ASCII)
* ``.ts``  — :func:`read_ts` (generic time series, ASCII)
"""

import numpy as np
import warnings
from pathlib import Path
from typing import Union, Tuple, Dict, Any, Optional

from uacpy._log import log_message
from uacpy.acoustic_signal.waveforms import sparc_pulse
from uacpy.core.exceptions import ConfigurationError, FileFormatError
from uacpy.core.results import (
    Field, ResultStack, Arrivals, Rays,
)
from uacpy.io._fortran_helpers import (
    read_vector as _read_vector, detect_endian, typed_format_error,
    strip_fortran_comment as _strip_fortran_comment,
    strip_fortran_quotes as _strip_fortran_quotes)
from uacpy.io.units import km_to_m


def read_shd_file(filepath: Union[str, Path]):
    """Read a single-frequency ``.shd`` file as a typed pressure result.

    Thin wrapper around :func:`read_shd_bin`. Returns:

      * :class:`Field` (complex narrowband pressure, ``coords={'depth',
        'range'}``) when the file carries a single source depth — the
        common case.
      * :class:`ResultStack` of single-source :class:`Field` slabs when
        multiple source depths are present.

    Multi-frequency ``.shd`` files raise :class:`ValueError` — call
    :func:`read_shd_bin` directly and construct a broadband
    :class:`Field` from the cube instead.
    """
    filepath = Path(filepath)
    shd = read_shd_bin(str(filepath))

    freqs = np.asarray(shd['freqVec'], dtype=float)
    nfreq = len(freqs)
    if nfreq == 0:
        raise FileFormatError(
            f"read_shd_file: {filepath} declares zero frequencies; the .shd "
            f"file is malformed (every Acoustics-Toolbox writer emits at "
            f"least one frequency record)."
        )
    if nfreq > 1:
        raise FileFormatError(
            f"read_shd_file: {filepath} contains {nfreq} frequencies; "
            "use read_shd_bin(filepath) for the full broadband payload "
            "and construct a broadband Field from it."
        )

    pressure = shd['pressure']               # (Ntheta, Nsz, Nrz, Nrr)
    pos = shd['Pos']
    source_depths = np.atleast_1d(np.asarray(pos['s']['z'], dtype=float))

    def _slab(isz: int) -> Field:
        return Field(
            data=pressure[0, isz, :, :],
            coords={'depth': pos['r']['z'], 'range': pos['r']['r']},
            model='', backend='',
            source_depths=np.array([float(source_depths[isz])]),
            frequencies=freqs,
        )

    if len(source_depths) == 1:
        return _slab(0)
    return ResultStack(
        slabs=[_slab(i) for i in range(len(source_depths))],
        coordinate=source_depths,
        coordinate_name='source_depth',
    )


@typed_format_error
def read_shd_bin(
    filename: str,
    xs: Optional[float] = None,
    ys: Optional[float] = None,
    frequency: Optional[float] = None,
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
    frequency : float, optional
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
        - 'pressure' : ndarray - Complex pressure field for a SINGLE
            frequency (``pressure_freq``), never a multi-frequency cube.
            Shape (Ntheta, Nsz, Nrz, Nrr) for rectilinear
            Shape (Ntheta, Nsz, 1, Nrr) for irregular
            For a broadband file, pass ``frequency=`` per frequency (or iterate
            ``freqVec`` calling this once per entry) — do not treat
            ``pressure`` as spanning ``freqVec``.
            Cells the engine never wrote (exact zeros on disk — Bellhop's
            r=0 column and ray shadow zones, an empty KRAKEN modal sum)
            are returned as NaN, uacpy's no-data convention.
        - 'pressure_freq' : float - The frequency (Hz) the ``pressure``
            cube was sliced at (``frequency`` snapped to the nearest ``freqVec``
            entry, or ``freqVec[0]`` when ``frequency`` is None).

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
    >>> print(f"Ranges: {shd['Pos']['r']['r']} m")

    >>> # Read specific source location
    >>> shd = read_shd_bin('field3d.shd', xs=5.0, ys=10.0)
    >>> # Pressure at first bearing, first source depth, first rcvr depth
    >>> p = shd['pressure'][0, 0, 0, :]

    >>> # Read specific frequency for broadband run
    >>> shd = read_shd_bin('broadband.shd', frequency=100.0)
    """
    with open(filename, "rb") as fid:
        head = fid.read(4)
        fid.seek(0)
        endian = detect_endian(head, source=f'read_shd_bin:{filename}')
        i4 = np.dtype(endian + 'i4')
        f4 = np.dtype(endian + 'f4')
        f8 = np.dtype(endian + 'f8')

        recl = int(np.fromfile(fid, dtype=i4, count=1)[0])
        title_bytes = fid.read(80)
        title = title_bytes.decode("ascii", errors="ignore").strip()
        fid.seek(4 * recl, 0)
        plot_type_bytes = fid.read(10)
        PlotType = plot_type_bytes.decode("ascii", errors="ignore")
        fid.seek(2 * 4 * recl, 0)
        Nfreq = int(np.fromfile(fid, dtype=i4, count=1)[0])
        Ntheta = int(np.fromfile(fid, dtype=i4, count=1)[0])
        Nsx = int(np.fromfile(fid, dtype=i4, count=1)[0])
        Nsy = int(np.fromfile(fid, dtype=i4, count=1)[0])
        Nsz = int(np.fromfile(fid, dtype=i4, count=1)[0])
        Nrz = int(np.fromfile(fid, dtype=i4, count=1)[0])
        Nrr = int(np.fromfile(fid, dtype=i4, count=1)[0])
        freq0 = float(np.fromfile(fid, dtype=f8, count=1)[0])
        atten = float(np.fromfile(fid, dtype=f8, count=1)[0])
        fid.seek(3 * 4 * recl, 0)
        freqVec = np.fromfile(fid, dtype=f8, count=Nfreq)
        fid.seek(4 * 4 * recl, 0)
        theta = np.fromfile(fid, dtype=f8, count=Ntheta)
        if PlotType[:2] != "TL":
            fid.seek(5 * 4 * recl, 0)
            s_x = np.fromfile(fid, dtype=f8, count=Nsx)
            fid.seek(6 * 4 * recl, 0)
            s_y = np.fromfile(fid, dtype=f8, count=Nsy)
        else:
            fid.seek(5 * 4 * recl, 0)
            s_x_lim = np.fromfile(fid, dtype=f8, count=2)
            s_x = np.linspace(s_x_lim[0], s_x_lim[1], Nsx)
            fid.seek(6 * 4 * recl, 0)
            s_y_lim = np.fromfile(fid, dtype=f8, count=2)
            s_y = np.linspace(s_y_lim[0], s_y_lim[1], Nsy)
        fid.seek(7 * 4 * recl, 0)
        s_z = np.fromfile(fid, dtype=f4, count=Nsz)
        fid.seek(8 * 4 * recl, 0)
        r_z = np.fromfile(fid, dtype=f4, count=Nrz)
        fid.seek(9 * 4 * recl, 0)
        r_r = np.fromfile(fid, dtype=f8, count=Nrr)
        if PlotType.strip() == "irregular":
            pressure = np.zeros((Ntheta, Nsz, 1, Nrr), dtype=np.complex64)
            Nrcvrs_per_range = 1
        else:
            pressure = np.zeros((Ntheta, Nsz, Nrz, Nrr), dtype=np.complex64)
            Nrcvrs_per_range = Nrz
        # Select ONE frequency slice. The returned 'pressure' cube is always
        # single-frequency (the slice 'pressure_freq'); a broadband caller must
        # pass frequency= per frequency or iterate freqVec, never treat 'pressure' as
        # a multi-frequency cube. NB: only the standard 2D path (xs is None)
        # carries a frequency axis in the record stream (field.f90 stacks frequency
        # outermost). The 3D / irregular multi-source path (xs given) is written
        # one frequency per file (bellhop3D.f90: iRec has no frequency stride),
        # so there is no frequency to select there.
        ifreq = 0
        if frequency is not None:
            ifreq = int(np.argmin(np.abs(freqVec - frequency)))
        if xs is None:
            if Nsx > 1 or Nsy > 1:
                warnings.warn(
                    f"read_shd_bin: file has Nsx={Nsx}, Nsy={Nsy} source "
                    "positions but no xs=/ys= selector was given; returning "
                    "the (0, 0) slot only. Pass xs=, ys= to choose another.",
                    UserWarning, stacklevel=2,
                )
            idxX = 0
            idxY = 0

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
                        temp = np.fromfile(fid, dtype=f4, count=2 * Nrr)
                        pressure[itheta, isz, irz, :] = temp[0::2] + 1j * temp[1::2]
            freq_label = float(freqVec[ifreq]) if len(freqVec) else None

        else:
            if ys is None:
                raise ConfigurationError("ys must be provided if xs is specified")
            # 3D / irregular multi-source files are single-frequency (no frequency
            # stride in the record index), so frequency= cannot select a slice here.
            if frequency is not None and len(freqVec) > 1:
                warnings.warn(
                    "read_shd_bin: frequency selection (frequency=) is not supported "
                    "for multi-source-position (3D/irregular) shade files, which "
                    "carry a single frequency; returning freqVec[0].",
                    UserWarning, stacklevel=2,
                )
            x_diff = np.abs(s_x - km_to_m(xs))
            idxX = np.argmin(x_diff)
            y_diff = np.abs(s_y - km_to_m(ys))
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
                        temp = np.fromfile(fid, dtype=f4, count=2 * Nrr)
                        pressure[itheta, isz, irz, :] = temp[0::2] + 1j * temp[1::2]
            freq_label = float(freqVec[0]) if len(freqVec) else None

    # AT engines zero-initialise the pressure grid and never touch cells no
    # energy reached (Bellhop: the r=0 column and ray shadow zones; an empty
    # KRAKEN modal sum below cutoff), so an exact complex zero on disk means
    # "no data", not a field value — a computed field is never exactly 0 in
    # float. Surface those cells as NaN, uacpy's no-data convention (RAM uses
    # the same for absorbing-layer / outside-PE-grid cells), so TL reductions
    # and metrics exclude them via np.isfinite instead of reading a huge
    # pressure-floor dB level as real.
    pressure[pressure == 0] = np.nan

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
        "pressure_freq": freq_label,
    }


@typed_format_error
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
        Same keys as :func:`read_shd_bin`, so a caller can switch between
        the ASCII and binary readers without remapping:
        - 'title': str, plot title
        - 'PlotType': str, plot type
        - 'freqVec': ndarray, frequency vector in Hz
        - 'freq0': float, reference frequency in Hz
        - 'atten': float, stabilizing attenuation in dB/wavelength
        - 'Pos': dict with position information:
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
    if not filepath.exists():
        raise FileFormatError(
            f"ASCII shade file not found: {filepath}. Run a model first to "
            "produce it."
        )
    with open(filepath, "r") as fid:
        title = fid.readline().strip()
        plot_type = fid.readline().strip()

        line = fid.readline()
        vals = np.array(line.split(), dtype=float)
        Nfreq = int(vals[0])
        Ntheta = int(vals[1])
        Nsd = int(vals[2])
        Nrd = int(vals[3])
        Nrr = int(vals[4])

        line = fid.readline()
        vals = np.array(line.split()[:2], dtype=float)
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

    pressure = temp[0::2, :].T + 1j * temp[1::2, :].T
    # Exact complex zero = cell the engine never wrote (see read_shd_bin);
    # surface as NaN, uacpy's no-data convention.
    pressure[pressure == 0] = np.nan

    return {
        "title": title,
        "PlotType": plot_type,
        "freqVec": freq_vec,
        "freq0": freq0,
        "atten": atten,
        "Pos": {"theta": theta, "s": {"z": s_z}, "r": {"z": r_z, "r": r_r}},
        "pressure": pressure,
    }


@typed_format_error
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
          arrival dicts.
        - ``arrivals``: flat list of per-arrival records (same data,
          un-nested) for filter/top_n/in_window chain methods.
        - ``frequencies``, ``source_depths``, ``receiver_depths``,
          ``receiver_ranges`` as typed attributes.

        Per-arrival fields, with units (ArrMod.f90:WriteArrivalsASCII):

        - ``amplitudes`` : real linear pressure magnitude (dimensionless);
          combine with ``phases`` for the complex amplitude.
        - ``phases`` : degrees.
        - ``delays`` : real part of travel time in **seconds**.
        - ``delays_imag`` : imaginary part of travel time in **seconds**;
          carries volume-attenuation loss so that
          ``exp(ω · delays_imag) = exp(-α·r)`` reproduces the standard
          Nepers attenuation when summed by ``delayandsum``.
        - ``src_angles``, ``rcv_angles`` : ray angles in **degrees**,
          measured from the horizontal (positive downward).
        - ``n_top_bounces``, ``n_bot_bounces`` : integer bounce counts.

        Depths are in **m**, ranges in **m** (already metres on disk — AT
        converts km→m at env-read time, so ``Pos%Rr`` in the ``.arr`` is
        metres and the reader applies no further conversion), frequencies
        in **Hz**.
    """
    filepath = Path(filepath)

    # Check if binary or ASCII format. The ASCII arrivals file always begins
    # with a quoted "'2D'" or "'3D'" tag at the very start of the first line.
    # The binary format is a Fortran unformatted stream whose first bytes are a
    # 4-byte record marker (typically \x04\x00\x00\x00 for a 4-byte record, but
    # compilers may emit other marker lengths). Prefer the positive ASCII test
    # and fall back to binary otherwise.
    with open(filepath, "rb") as f:
        head = f.read(16)
    try:
        head_text = head.decode('ascii')
    except UnicodeDecodeError:
        head_text = ''

    if not head_text.lstrip().startswith(("'2D'", "'3D'")):
        raise ConfigurationError(
            "Binary arrivals format (.arr written by RunType 'a') is not "
            "supported; see ArrMod.f90:WriteArrivalsBinary for the record "
            "layout.",
            remediation="Re-run Bellhop with RunType 'A' (ASCII), which is "
                        "what uacpy's own runs emit.",
        )
    is_3d = head_text.lstrip().startswith("'3D'")

    if not is_3d:
        # ArrMod.f90 writes each Fortran record (freq, nsd+sz, nrd+rz,
        # nrr+rr, max-narr, narr, and each 8-tuple arrival) via
        # list-directed WRITE, which different Fortran runtimes may wrap
        # at different column widths. Walk the file as a token stream so
        # the parser is independent of how those records are line-broken.
        with open(filepath, 'r') as f:
            f.readline()  # skip the '2D' / '3D' flag line
            tokens = []
            for line in f:
                tokens.extend(line.split())

        def _next_floats(t_iter, n):
            return [float(next(t_iter)) for _ in range(n)]

        def _next_int(t_iter):
            # Some writers emit counts as floats; tolerate either.
            return int(float(next(t_iter)))

        t_iter = iter(tokens)
        freq = float(next(t_iter))

        nsd = _next_int(t_iter)
        sz = np.array(_next_floats(t_iter, nsd))

        nrd = _next_int(t_iter)
        rz = np.array(_next_floats(t_iter, nrd))

        nrr = _next_int(t_iter)
        rr = np.array(_next_floats(t_iter, nrr))

        arrivals_by_receiver = []

        for isd in range(nsd):
            sd_list = []
            # Skip the per-source max-narr value.
            _next_int(t_iter)

            for irz in range(nrd):
                rd_list = []
                for irr in range(nrr):
                    narr = _next_int(t_iter)

                    rcv_arrivals = {
                        "amplitudes": np.array([], dtype='float64'),
                        "phases": np.array([], dtype='float64'),
                        "delays": np.array([], dtype='float64'),
                        "delays_imag": np.array([], dtype='float64'),
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
                            values = _next_floats(t_iter, 8)

                            amps.append(values[0])
                            phases.append(values[1])
                            delays_r.append(values[2])
                            delays_i.append(values[3])
                            src_angs.append(values[4])
                            rcv_angs.append(values[5])
                            n_tops.append(int(values[6]))
                            n_bots.append(int(values[7]))

                        rcv_arrivals = {
                            "amplitudes": np.array(amps),
                            "phases": np.array(phases),
                            "delays": np.array(delays_r),
                            "delays_imag": np.array(delays_i),
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

    if nsd == 1:
        return Arrivals(
            by_receiver=arrivals_by_receiver,
            receiver_depths=rz,
            receiver_ranges=rr,
            model='', backend='',
            source_depths=sz,
            frequencies=float(freq),
            metadata={},
        )
    # Multi-source: one :class:`Arrivals` per source-depth slab,
    # bundled into a :class:`ResultStack`. Each slab carries the same
    # receiver grid and frequency; only the source-depth dimension is
    # split.
    slabs = [
        Arrivals(
            by_receiver=[arrivals_by_receiver[isd]],
            receiver_depths=rz,
            receiver_ranges=rr,
            model='', backend='',
            source_depths=np.array([float(sz[isd])]),
            frequencies=float(freq),
            metadata={},
        )
        for isd in range(nsd)
    ]
    return ResultStack(
        slabs=slabs, coordinate=sz, coordinate_name='source_depth',
    )


def read_ray_file(filepath: Union[str, Path]):
    """
    Read a Bellhop ``.ray`` file as a typed ray-bundle result.

    For ``RunType='R'`` (RAYS) the file holds ``NSz × Nalpha`` ray
    blocks in source-major order — return a :class:`Rays` for
    ``NSz == 1`` or a :class:`ResultStack` of :class:`Rays` slabs for
    ``NSz > 1``. ``EIGENRAYS`` files write a variable number of rays
    per source; this reader leaves them flat (the Bellhop wrapper
    loops Python-side for multi-source eigenrays to disambiguate).

    Parameters
    ----------
    filepath : str or Path
        Path to ``.ray`` file.

    Returns
    -------
    :class:`Rays` or :class:`ResultStack`
    """
    filepath = Path(filepath)

    rays = []
    n_sz = 1
    n_alpha = 0

    try:
        with open(filepath, "r") as f:
            f.readline()                  # title
            f.readline()                  # frequency
            # Line 3: NSx NSy NSz — the trailing token is the source-
            # depth count for 2-D Bellhop.
            sx_sy_sz_tokens = f.readline().split()
            if len(sx_sy_sz_tokens) >= 3:
                n_sz = int(sx_sy_sz_tokens[2])
            # Line 4: Nalpha Nbeta — first token is the launch-angle
            # count, used to split ray blocks per source-depth for
            # RAYS mode (deterministic NSz × Nalpha layout).
            alpha_beta_tokens = f.readline().split()
            if alpha_beta_tokens:
                n_alpha = int(alpha_beta_tokens[0])
            f.readline()                  # top depth
            f.readline()                  # bottom depth
            f.readline().strip()          # 'rz' / 'xyz' marker
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

    if n_sz <= 1 or n_alpha == 0 or len(rays) != n_sz * n_alpha:
        # Single source, or EIGENRAYS (which writes a non-deterministic
        # subset because ``WriteRay2D`` fires only on receiver hits,
        # and Bellhop's eigenray search reorders ``alpha`` for its
        # bracketing heuristic — the .ray file therefore has neither a
        # fixed block size nor a monotonic alpha pattern). The Bellhop
        # wrapper handles multi-source EIGENRAYS by looping in Python.
        return Rays(rays=rays, model='', backend='')

    # RAYS mode: every (source, alpha) pair writes one ray. The block
    # boundary is deterministic at index ``i * n_alpha``.
    slabs = [
        Rays(rays=rays[isz * n_alpha:(isz + 1) * n_alpha],
             model='', backend='')
        for isz in range(n_sz)
    ]
    return ResultStack(
        slabs=slabs,
        coordinate=np.arange(n_sz, dtype=float),
        coordinate_name='source_depth',
    )


def read_prt(prt_path: Union[str, Path], *, tail_bytes: Optional[int] = None) -> Optional[str]:
    """Read an Acoustics-Toolbox ``.prt`` log.

    AT binaries (Kraken/Scooter/Sparc/Bounce) dump fatal-error detail and
    run diagnostics to ``<base>.prt`` instead of stderr. Returns the log
    text, or ``None`` when the file is absent or unreadable.

    Parameters
    ----------
    prt_path : str or Path
        Path to the ``.prt`` file.
    tail_bytes : int, optional
        When given, return only the trailing ``tail_bytes`` of the file —
        used to append a short failure excerpt to error messages.
    """
    path = Path(prt_path)
    if not path.exists():
        return None
    try:
        if tail_bytes is not None:
            size = path.stat().st_size
            with path.open('rb') as fh:
                if size > tail_bytes:
                    fh.seek(size - tail_bytes)
                return fh.read().decode('utf-8', errors='replace')
        return path.read_text(encoding='utf-8', errors='ignore')
    except OSError:
        return None


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
        head = f.read(4)
        if len(head) < 4:
            return rays
        endian = detect_endian(head, source=f'read_ray_bin:{Path(filepath).name}')
        i4 = np.dtype(endian + 'i4')
        f4 = np.dtype(endian + 'f4')
        recl = int(np.frombuffer(head, dtype=i4, count=1)[0])
        f.seek(recl * 4)

        truncated_after = None
        while True:
            count_buf = f.read(4)
            if len(count_buf) < 4:
                # EOF where a ray-length record was expected.
                truncated_after = len(rays)
                break
            n_points = int(np.frombuffer(count_buf, dtype=i4, count=1)[0])
            if n_points <= 0:
                break

            # WriteRay2D writes ray2D%x (r, z interleaved) directly in metres
            # (WriteRay.f90:45); no km conversion needed.
            coords = np.fromfile(f, dtype=f4, count=2 * n_points)
            if coords.size < 2 * n_points:
                truncated_after = len(rays)
                break

            rays.append(
                {
                    "r": coords[0::2].astype(float),
                    "z": coords[1::2].astype(float),
                }
            )

    if truncated_after is not None:
        warnings.warn(
            f"read_ray_file: {filepath} appears truncated; recovered "
            f"{truncated_after} ray(s) before EOF. The producing model "
            "may have crashed mid-write — check its .prt log.",
            UserWarning,
            stacklevel=2,
        )
    return rays


@typed_format_error
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
        - 'r_prof' : ndarray - Range values in metres, shape (n_prof,).
          Stored on disk in km (``sspMod.f90:417-422``) and converted here.
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
    >>> print(f"Ranges: {ssp['r_prof']} m")
    >>> print(f"SSP matrix shape: {ssp['c_mat'].shape}")
    >>> # Sound speed at depth index 10, range index 5
    >>> c = ssp['c_mat'][10, 5]
    """
    filepath = Path(filepath)
    # Canonical AT/Bellhop layout (sspMod.f90:407,417,428):
    #   Line 1            : NProf (integer)
    #   Line 2            : NProf range values (single list-directed record)
    #   Lines 3..NSSP+2   : one SSP row per depth, each with NProf values
    with open(filepath, "r") as fid:
        n_prof = int(fid.readline().strip())
        r_prof = np.array(fid.readline().split()[:n_prof], dtype=float)
        if r_prof.size != n_prof:
            raise FileFormatError(
                f"SSP file {filepath}: expected {n_prof} range values on line 2, "
                f"parsed {r_prof.size}"
            )
        # Each remaining line is one depth row of NProf speed values. The
        # number of depth rows isn't stored here (it lives in the .env file)
        # so we read whatever the file contains and infer NSSP.
        rows = []
        for line in fid:
            tokens = line.split()
            if not tokens:
                continue
            row = [float(t) for t in tokens]
            if len(row) != n_prof:
                raise FileFormatError(
                    f"SSP file {filepath}: expected {n_prof} values per row, "
                    f"got {len(row)}"
                )
            rows.append(row)
        c_mat = np.array(rows)  # shape (n_depth, n_prof)
        n_depth = c_mat.shape[0]

    return {"n_prof": n_prof, "r_prof": km_to_m(r_prof), "c_mat": c_mat,
            "n_depth": n_depth}


@typed_format_error
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
        - 'Segx' : ndarray - X segment boundaries in metres, shape (Nx,).
          Stored on disk in km (``sspMod.f90:614-615``) and converted here.
        - 'Segy' : ndarray - Y segment boundaries in metres, shape (Ny,).
          Also km on disk.
        - 'Segz' : ndarray - Depth values in metres, shape (Nz,)
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

    # Bellhop3D (sspMod.f90:570-617) reads each Segx / Segy / Segz
    # vector and each per-(z, y) SSP row with a single list-directed
    # READ statement — one Fortran record per vector / row.
    def _read_vec(fid, n):
        vec = np.array(fid.readline().split()[:n], dtype=float)
        if vec.size != n:
            raise FileFormatError(
                f"3D SSP file {filepath}: expected {n} values on line, "
                f"parsed {vec.size}"
            )
        return vec

    with open(filepath, "r") as fid:
        Nx = int(fid.readline().strip())
        Segx = _read_vec(fid, Nx)
        Ny = int(fid.readline().strip())
        Segy = _read_vec(fid, Ny)
        Nz = int(fid.readline().strip())
        Segz = _read_vec(fid, Nz)
        c_mat = np.zeros((Nz, Ny, Nx))

        # Bellhop3D writes outermost-iz, inner-iy, then one record of Nx values.
        for iz in range(Nz):
            for iy in range(Ny):
                c_mat[iz, iy, :] = _read_vec(fid, Nx)

    return {
        "Nx": Nx,
        "Ny": Ny,
        "Nz": Nz,
        "Segx": km_to_m(Segx),
        "Segy": km_to_m(Segy),
        "Segz": Segz,
        "c_mat": c_mat,
    }


@typed_format_error
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
        title = _strip_fortran_quotes(f.readline())
        log_message('oalib_reader', f"Title: {title}", verbose=verbose)
        opt = _strip_fortran_quotes(f.readline())
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
        M_limit = int(_strip_fortran_comment(f.readline()))
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
        r_rcv = km_to_m(r_rcv)
        pos_temp = _read_sz_rz(f)
        # Rro is read by AT with Units='m' (KrakenField/field.f90:147), so the
        # column is already metres on disk — no conversion.
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
        "r_prof": km_to_m(r_prof),
        "pos": {
            "s": {"z": pos_temp["sz"]},
            "r": {"z": pos_temp["rz"], "r": r_rcv, "ro": r_offsets},
            "Nro": N_offsets,
        },
    }


@typed_format_error
def read_flp3d(fileroot: Union[str, Path]) -> Dict[str, Any]:
    """
    Read a 3-D field parameters file (.flp) for the FIELD3D program.

    Parameters
    ----------
    fileroot : str or Path
        File root name (``.flp`` appended when the path has no suffix).

    Returns
    -------
    flp3d_data : dict
        Dictionary containing:

        - 'title': str — Title.
        - 'opt': str — FIELD3D option string. ``opt[0:3]`` selects the
          field algorithm (``'STD'``/``'PAR'``/``'GBT'``), ``opt[3]`` is
          ``'T'`` to run the tesselation check, ``opt[6]`` is the ``.sbp``
          beam-pattern flag (``field3d.f90:54``, ``:96``, ``:210``).
        - 'M_limit': int — Maximum number of modes to use.
        - 'pos': dict — Source/receiver geometry:

          * ``'s'``: ``'x'``/``'y'`` source coordinates in m, ``'z'``
            source depths in m.
          * ``'r'``: ``'z'`` receiver depths in m, ``'r'`` receiver ranges
            in m, ``'theta'`` receiver bearings in degrees.
          * ``'Nsx'``/``'Nsy'``: source x/y counts.
        - 'nodes': dict — Triangulation nodes: ``'x'``/``'y'`` in m and
          ``'mode_files'`` (list of per-node mode-file names).
        - 'elements': ndarray, shape (n_elts, 3), int — 1-based node
          indices of each triangle, as FIELD3D indexes them
          (``field3d.f90:285-291``).

    Notes
    -----
    Read order follows ``KrakenField/field3d.f90`` ``READIN``: title,
    option, Mlimit, ``Sx`` (km), ``Sy`` (km), ``Sz``/``Rz`` (m), ``Rr``
    (km), ``theta`` (degrees), node block, element block. Axes on disk in
    km are returned in metres.

    See Also
    --------
    write_field3dflp : Write 3-D field parameters
    read_flp : Read 2-D field parameters
    """
    fileroot = Path(fileroot)
    if not fileroot.suffix:
        filepath = fileroot.with_suffix(".flp")
    else:
        filepath = fileroot

    with open(filepath, "r") as f:
        title = _strip_fortran_quotes(f.readline())
        opt = _strip_fortran_quotes(f.readline())
        M_limit = int(_strip_fortran_comment(f.readline()))

        s_x, n_sx = _read_vector(f)
        s_y, n_sy = _read_vector(f)
        pos_temp = _read_sz_rz(f)
        r_rcv, _ = _read_vector(f)
        theta_rcv, _ = _read_vector(f)

        n_nodes = int(_strip_fortran_comment(f.readline()))
        node_x = np.zeros(n_nodes)
        node_y = np.zeros(n_nodes)
        mode_files = []
        for i in range(n_nodes):
            parts = _strip_fortran_comment(f.readline()).split()
            node_x[i] = float(parts[0])
            node_y[i] = float(parts[1])
            mode_files.append(parts[2].strip("'\""))

        n_elts = int(_strip_fortran_comment(f.readline()))
        elements = np.zeros((n_elts, 3), dtype=int)
        for i in range(n_elts):
            elements[i] = [int(v) for v in
                           _strip_fortran_comment(f.readline()).split()[:3]]

    return {
        "title": title,
        "opt": opt,
        "M_limit": M_limit,
        "pos": {
            "s": {
                "x": km_to_m(s_x),
                "y": km_to_m(s_y),
                "z": pos_temp["sz"],
            },
            "r": {
                "z": pos_temp["rz"],
                "r": km_to_m(r_rcv),
                "theta": theta_rcv,
            },
            "Nsx": n_sx,
            "Nsy": n_sy,
        },
        "nodes": {
            "x": km_to_m(node_x),
            "y": km_to_m(node_y),
            "mode_files": mode_files,
        },
        "elements": elements,
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
        title = _strip_fortran_quotes(f.readline())
        raw_tokens = []
        for line in f:
            raw_tokens.extend(line.strip().split())

    if not raw_tokens:
        raise FileFormatError(f"RTS file {filepath} appears empty after the title line")

    # First token is NRr/NRz, then exactly NRr range/depth floats.
    nr = int(raw_tokens[0])
    if len(raw_tokens) < 1 + nr:
        raise FileFormatError(
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
    rts_data: Dict[str, Any], frequency: float, method: str = "fft",
    *, pulse_type: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project SPARC time-series data onto complex pressure at one frequency.

    ``method='fft'`` extracts the spectral bin nearest ``frequency`` from a
    Hanning-windowed FFT; ``method='goertzel'`` uses the Goertzel
    single-bin DFT. Both return ``(p_at_freq, ranges)`` where
    ``p_at_freq`` is the model-native, source-normalised complex pressure
    suitable for wrapping in a complex narrowband :class:`Field`
    (``coords={'depth', 'range'}``, ``phase_reference='travelling_wave'``).

    A post-processing utility for a ``.rts`` read by :func:`read_rts_file`;
    :class:`uacpy.models.SPARC` returns ``p(t)`` and does not call it. The
    projection is not a calibrated substitute for Kraken/Scooter TL — see the
    module docstring of ``uacpy/tests/test_sparc_output_modes.py``.
    """
    p = rts_data["p"]
    dt = rts_data["dt"]
    ranges = rts_data["ranges"]

    nt, nr = p.shape

    if pulse_type is not None:
        # Deconvolve the known source spectrum (convolution theorem): the range
        # time-series r(t) = s(t) ⊛ h(t), so rfft(r)/rfft(s) = h(w0) — the CW
        # transfer function ≈ absolute TL re 1 m (Jensen COA Eq. 8.1). uacpy
        # generated the pulse, so s(t) is known. The estimate is physical and
        # window/grid-independent once the output Nyquist clears the pulse band
        # (the SPARC model sizes n_t_out for this), but SPARC's discretised
        # pulse and band-pass leave a frequency-dependent bias of a few dB vs
        # Kraken/Scooter — it is not a calibrated replacement for them.
        # Use the RECTANGULAR DFT (no taper): a window breaks the convolution
        # theorem and would null the transient source pulse (first few samples).
        t = np.asarray(rts_data["time"], dtype=float)
        s_t, _ = sparc_pulse(t, 2.0 * np.pi * frequency, pulse_type[0])
        freqs = np.fft.rfftfreq(nt, dt)
        f_idx = int(np.argmin(np.abs(freqs - frequency)))
        S_at_f0 = np.fft.rfft(s_t)[f_idx]
        if S_at_f0 == 0:
            raise ConfigurationError(
                "rts_to_pressure: source spectrum is zero at "
                f"{frequency} Hz for pulse_type={pulse_type!r}; cannot "
                "deconvolve (check pulse / frequency).")
        p_freq = np.fft.rfft(p, axis=0)
        return p_freq[f_idx, :] / S_at_f0, ranges

    if method == "fft":
        window = np.hanning(nt)
        p_freq = np.fft.rfft(p * window[:, np.newaxis], axis=0)
        freqs = np.fft.rfftfreq(nt, dt)
        freq_idx = np.argmin(np.abs(freqs - frequency))
        p_at_freq = 2.0 * p_freq[freq_idx, :] / np.sum(window)
    elif method == "goertzel":
        omega = 2 * np.pi * frequency
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
        raise ConfigurationError(
            f"rts_to_pressure: unknown method {method!r}; "
            f"use 'fft' or 'goertzel'."
        )

    return p_at_freq, ranges


@typed_format_error
def read_ts(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Read time-series file from acoustic models.

    This is a simple ASCII time series format, different from the RTS
    format used by SPARC. Used by some AT models for time-domain output.

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

    This format is simpler than the .rts format used by SPARC.

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
    read_rts_file : Read the ASCII RTS format from SPARC
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
            raise FileFormatError(f"Expected {nrd} receiver depths, got {len(rd)}")
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
