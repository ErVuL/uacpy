"""
Acoustics Toolbox / OALIB output-file readers.

One file per shared output format (Kraken, Scooter, SPARC, Bounce,
Bellhop output formats; modes are kept in their own ``modes_reader.py``).

Provides:

* ``.shd`` — :func:`read_shd_file`, :func:`read_shd_bin`, :func:`read_shd_asc`
* ``.arr`` — :func:`read_arr_file` (Bellhop arrivals, ASCII)
* ``.ray`` — :func:`read_ray_file` (Bellhop rays, ASCII)
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
from uacpy.core.exceptions import (
    ConfigurationError, FileFormatError, UnsupportedFeatureError,
)
from uacpy.core.results import (
    Field, ResultStack, Arrivals, Rays,
)
from uacpy.io._fortran_helpers import (
    _bound_counts,
    read_vector as _read_vector, detect_endian, typed_format_error,
    fortran_float,
    list_directed_int,
    read_list_directed_values, take_tokens,
    strip_fortran_comment as _strip_fortran_comment,
    strip_fortran_quotes as _strip_fortran_quotes)
from uacpy.io.units import km_to_m


def read_shd_file(filepath: Union[str, Path]):
    """Read a single-frequency, single-bearing ``.shd`` file as a typed
    pressure result.

    Thin wrapper around :func:`read_shd_bin`. Returns:

      * :class:`Field` (complex narrowband pressure) when the file carries a
        single source depth — the common case.
      * :class:`ResultStack` of single-source :class:`Field` slabs when
        multiple source depths are present.

    A rectilinear file (``PlotType == 'rectilin  '``) gives
    ``coords={'depth', 'range'}``. An irregular file (``'irregular '``) gives
    ``coords={'range'}``: its receivers are the paired coordinates
    ``(Rz(i), Rr(i))`` and only one row of ``NRr`` samples is written per
    source depth (``Bellhop/bellhop.f90:202-206``, ``:323-326``), so the
    paired depths ride on ``metadata['receiver_depths']`` rather than forming
    a second axis.

    Multi-frequency and multi-bearing ``.shd`` files raise
    :class:`FileFormatError`: this wrapper carries neither axis. Call
    :func:`read_shd_bin` directly and build the result from its
    ``(Ntheta, Nsz, Nrz, Nrr)`` cube. A file with several source (x, y)
    positions (``Nsx``/``Nsy`` > 1, BELLHOP3D / FIELD3D) raises
    :class:`~uacpy.core.exceptions.UnsupportedFeatureError` for the same
    reason — use :func:`read_shd_bin` with ``xs=``/``ys=`` per source.
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
    # The receiver-bearing axis is a first-class .shd dimension
    # (misc/RWSHDFile.f90:105,107; BELLHOP3D writes Ntheta record blocks per
    # source, Bellhop/bellhop3D.f90:405-411). A Field carries no bearing axis,
    # so refuse rather than return plane 0 as if it were the whole file.
    n_theta = np.atleast_1d(np.asarray(pos['theta'], dtype=float)).size
    if n_theta > 1:
        raise FileFormatError(
            f"read_shd_file: {filepath} carries {n_theta} receiver bearings; "
            "use read_shd_bin(filepath) for the full (Ntheta, Nsz, Nrz, Nrr) "
            "cube and build the result from it."
        )

    # A file with several source (x, y) positions holds one pressure cube
    # per position; read_shd_bin without xs=/ys= returns only slot (0, 0),
    # so returning a Field here would silently present one source's field
    # as the whole file.
    n_sx = np.atleast_1d(np.asarray(pos['s']['x'], dtype=float)).size
    n_sy = np.atleast_1d(np.asarray(pos['s']['y'], dtype=float)).size
    if n_sx > 1 or n_sy > 1:
        raise UnsupportedFeatureError(
            'read_shd_file',
            f"{filepath} carries Nsx={n_sx} x Nsy={n_sy} source positions; "
            f"this wrapper returns fields for a single source position only",
            alternatives=[
                "read_shd_bin(filepath, xs=..., ys=...) once per source "
                "position and build the result from each cube",
            ],
            alternatives_label='readers',
        )

    source_depths = np.atleast_1d(np.asarray(pos['s']['z'], dtype=float))
    irregular = shd['PlotType'].strip() == 'irregular'
    receiver_depths = np.atleast_1d(np.asarray(pos['r']['z'], dtype=float))
    receiver_ranges = np.atleast_1d(np.asarray(pos['r']['r'], dtype=float))
    if irregular and receiver_depths.size != receiver_ranges.size:
        # Bellhop/ReadEnvironmentBell.f90:414 ERROUTs unless NRz == NRr for
        # RunType(5:5)='I', so the two header counts must pair up.
        raise FileFormatError(
            f"read_shd_file: {filepath} declares PlotType 'irregular' but "
            f"carries {receiver_depths.size} receiver depths against "
            f"{receiver_ranges.size} receiver ranges; an irregular grid pairs "
            f"them one-to-one."
        )

    def _slab(isz: int) -> Field:
        if irregular:
            return Field(
                data=pressure[0, isz, 0, :],
                coords={'range': receiver_ranges},
                model='', backend='',
                source_depths=np.array([float(source_depths[isz])]),
                frequencies=freqs,
                metadata={'receiver_depths': receiver_depths.copy()},
            )
        return Field(
            data=pressure[0, isz, :, :],
            coords={'depth': receiver_depths, 'range': receiver_ranges},
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

        # .shd is a Fortran DIRECT-access file whose logical record length is
        # counted in 4-byte words, not bytes: misc/RWSHDFile.f90:100 sets
        # LRecl = MAX( 41, 2*Nfreq, 2*Ntheta, 2*NSx, 2*NSy, NSz, NRz, 2*NRr )
        # ("words/record") and :102 opens the file with RECL = 4 * LRecl. Record
        # n therefore begins at byte (n - 1) * 4 * recl — the offset every seek
        # below computes. Header records (misc/RWSHDFile.f90:103-114):
        #   1  LRecl (i4) then Title, CHARACTER*80 (= 40 words, whence MAX(41,…))
        #   2  PlotType, CHARACTER*10
        #   3  Nfreq Ntheta NSx NSy NSz NRz NRr (i4) then freq0, atten (f8)
        #   4  freqVec  5  theta  6  Sx  7  Sy  8  Sz  9  Rz  10  Rr
        # The doubled terms of the LRecl formula are the REAL(KIND=8) vectors;
        # NSz and NRz enter undoubled because Sz and Rz are REAL(KIND=4)
        # (misc/SourceReceiverPositions.f90:22-27) — which is why those two
        # alone are read as f4 here. Records 11.. hold the pressure, one row of
        # NRr default-kind COMPLEX (two f4 words apiece) per receiver depth.
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
        # Receivers per range record. For PlotType 'irregular ' the receivers
        # are the paired coordinates (Rz(i), Rr(i)) and exactly one row of NRr
        # samples is written per source depth (Bellhop/bellhop.f90:202-206,
        # :323-326), so Nrz indexes the same paired list as Nrr and must not
        # enter the on-disk sample count.
        Nrcvrs_per_range = 1 if PlotType.strip() == "irregular" else Nrz
        file_size = Path(filename).stat().st_size
        # Record 9 holds Nrz REAL*4 receiver depths (misc/RWSHDFile.f90:113),
        # so that count alone is bounded by file_size // 4.
        _bound_counts(filename, file_size, 4, Nrz=Nrz)
        # The pressure cube is sized straight off the remaining header words;
        # one complex sample occupies 8 bytes on disk, so no count and no
        # product of counts can exceed file_size // 8.
        _bound_counts(filename, file_size, 8,
                      Nfreq=Nfreq, Ntheta=Ntheta, Nsx=Nsx, Nsy=Nsy,
                      Nsz=Nsz, Nrz_per_range=Nrcvrs_per_range, Nrr=Nrr)
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
            # Compressed FIELD3D 'TL' layout: records 6 and 7 carry only the
            # first and last source coordinate (misc/RWSHDFile.f90:126-127),
            # the grid between them being uniform.
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
        pressure = np.zeros((Ntheta, Nsz, Nrcvrs_per_range, Nrr),
                            dtype=np.complex64)
        # Select ONE frequency slice. The returned 'pressure' cube is always
        # single-frequency (the slice 'pressure_freq'); a broadband caller must
        # pass frequency= per frequency or iterate freqVec, never treat 'pressure' as
        # a multi-frequency cube. NB: only the standard 2D path (xs is None)
        # carries a frequency axis in the record stream (KrakenField/field.f90
        # stacks frequency
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
            # Records after the 10 header ones run frequency-major, then
            # bearing, then source depth, then receiver depth — the nesting the
            # writers step through one record at a time: KrakenField/field.f90
            # resets iRec to 10 on the first frequency (:179) and bumps it once
            # per (source depth, receiver depth) inside its frequency loop
            # (:215), and Bellhop/bellhop.f90:323-326 lands on the same index
            # via ``IRec = 10 + NRz_per_range * ( is - 1 )``. ``recnum`` is the
            # 0-based record index, i.e. Fortran REC - 1.
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
            # Sx/Sy are metres on disk: ReadSxSy reads them as 'km' and
            # ReadVector scales by 1000 before WriteHeader runs
            # (misc/SourceReceiverPositions.f90:87-88, :277).
            x_diff = np.abs(s_x - km_to_m(xs))
            idxX = np.argmin(x_diff)
            y_diff = np.abs(s_y - km_to_m(ys))
            idxY = np.argmin(y_diff)

            # Source x/y replace frequency as the outer strides here:
            # Bellhop/bellhop3D.f90:407-410 builds exactly this index.
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
        The same header keys as :func:`read_shd_bin`. ``pressure`` is 2-D
        here — one source depth, one frequency — and there is no
        ``pressure_freq``:
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
    ASCII shade file format (``Matlab/ReadWrite/read_shd_asc.m:13-38``):
    two title lines read whole, then one flat whitespace-delimited token
    stream — every scalar and vector there is consumed with ``fscanf``, so
    the values may be packed onto shared lines or spread one per line:

    - Line 1: Title
    - Line 2: Plot type
    - Tokens: Nfreq, Ntheta, Nsd, Nrd, Nrr, freq0, atten
    - Frequency vector, bearing angles, source depths, receiver depths,
      receiver ranges
    - Pressure data: ``Nrd`` groups of ``2 * Nrr`` interleaved
      real/imaginary values

    Reads the first source depth only, matching ``read_shd_asc.m:29-32``.

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
            f"ASCII shade file not found: {filepath}. No Acoustics-Toolbox "
            "program writes .shd.asc — supply the path to an existing ASCII "
            "shade file, or read a solver-written binary .shd with "
            "read_shd_bin."
        )
    with open(filepath, "r") as fid:
        title = fid.readline().strip()
        plot_type = fid.readline().strip()
        tokens = iter(fid.read().split())

    # Every vector is sized off an on-disk count, but each is filled straight
    # from the token stream: a garbage count exhausts the iterator (raising
    # StopIteration, which @typed_format_error types as FileFormatError)
    # instead of driving an allocation the file cannot back.
    def _take(n: int) -> np.ndarray:
        return np.array([float(next(tokens)) for _ in range(n)])

    Nfreq, Ntheta, Nsd, Nrd, Nrr = (int(v) for v in _take(5))
    freq0, atten = _take(2)
    freq_vec = _take(Nfreq)
    theta = _take(Ntheta)
    s_z = _take(Nsd)
    r_z = _take(Nrd)
    r_r = _take(Nrr)
    # One row of 2*Nrr interleaved re/im values per receiver depth.
    rows = np.array([_take(2 * Nrr) for _ in range(Nrd)])

    pressure = rows[:, 0::2] + 1j * rows[:, 1::2]
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
def read_arr_file(filepath: Union[str, Path], *, grid_type: str = 'R'):
    """
    Read arrivals file (.arr) from Bellhop

    Parameters
    ----------
    filepath : str or Path
        Path to .arr file
    grid_type : str, optional
        Receiver-grid option the run used — Bellhop ``RunType(5:5)``:
        ``'R'`` rectilinear (default), ``'I'`` irregular. The ``.arr``
        header always reports the full ``Pos%NRz``
        (``Bellhop/ReadEnvironmentBell.f90:591``) but the body carries only
        ``NRz_per_range`` depth blocks, and that is 1 for an irregular grid
        (``Bellhop/bellhop.f90:202-206``, ``:329``;
        ``Bellhop/ArrMod.f90:101-102``). The file records nothing that
        distinguishes the two, so the caller must say which it is.

    Returns
    -------
    Arrivals
        Typed result with:
        - ``by_receiver``: nested list ``[isd][ird][irr]`` of per-receiver
          arrival dicts. ``ird`` spans the receiver depths for
          ``grid_type='R'`` and is a single block for ``grid_type='I'``,
          whose receivers are the paired coordinates ``(Rz(i), Rr(i))``
          indexed by ``irr``.
        - ``arrivals``: flat list of per-arrival records (same data,
          un-nested) for filter/top_n/in_window chain methods.
        - ``frequencies``, ``source_depths``, ``receiver_depths``,
          ``receiver_ranges`` as typed attributes.

        Per-arrival fields, with units (ArrMod.f90:WriteArrivalsASCII):

        - ``amplitudes`` : real linear pressure magnitude (dimensionless);
          combine with ``phases`` for the complex amplitude. Bellhop has
          already folded the spreading factor in on write
          (``Bellhop/ArrMod.f90:103-110``): ``1/sqrt(r)`` for a point source,
          ``4·sqrt(pi)`` for a line source, and a fixed ``1e5`` at ``r == 0``
          to avoid the division — so an ``r = 0`` receiver carries an
          arbitrary magnitude, not a physical one.
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
    if grid_type not in ('R', 'I'):
        raise ConfigurationError(
            f"read_arr_file(grid_type={grid_type!r}) is not valid. Use 'R' "
            f"(rectilinear) or 'I' (irregular), matching Bellhop's "
            f"RunType(5:5).")

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
        raise FileFormatError(
            "Binary arrivals format (.arr written by RunType 'a') is not "
            "supported; see ArrMod.f90:WriteArrivalsBinary for the record "
            "layout.",
            remediation="Re-run Bellhop with RunType 'A' (ASCII), which is "
                        "what uacpy's own runs emit.",
        )
    if head_text.lstrip().startswith("'3D'"):
        raise FileFormatError(
            "3-D arrivals format (the '3D' tag written by BELLHOP3D) is not "
            "supported; its records carry ten values per arrival plus a "
            "receiver-bearing loop (Bellhop/ArrMod.f90:256-302).",
            remediation="Re-run in 2-D (bellhop.exe), which writes the "
                        "eight-value '2D' arrivals file this reader parses.",
        )

    # ArrMod.f90 writes each Fortran record (freq, nsd+sz, nrd+rz,
    # nrr+rr, max-narr, narr, and each 8-tuple arrival) via
    # list-directed WRITE, which different Fortran runtimes may wrap
    # at different column widths. Walk the file as a token stream so
    # the parser is independent of how those records are line-broken.
    with open(filepath, 'r') as f:
        f.readline()  # skip the '2D' flag line
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

    # Body bound: WriteArrivalsASCII loops DO id = 1, Nrd with Nrd bound to
    # NRz_per_range (Bellhop/ArrMod.f90:101-102, Bellhop/bellhop.f90:329),
    # which is 1 for RunType(5:5)='I' and Pos%NRz otherwise
    # (Bellhop/bellhop.f90:202-206).
    if grid_type == 'I':
        if nrd != nrr:
            raise FileFormatError(
                f"read_arr_file({filepath}, grid_type='I'): header declares "
                f"{nrd} receiver depths against {nrr} receiver ranges. An "
                f"irregular grid pairs them one-to-one — Bellhop refuses the "
                f"deck otherwise (Bellhop/ReadEnvironmentBell.f90:414).",
                remediation="Pass grid_type='R' if the run used a "
                            "rectilinear receiver grid.",
            )
        n_depth_blocks = 1
    else:
        n_depth_blocks = nrd

    arrivals_by_receiver = []

    for isd in range(nsd):
        sd_list = []
        # WriteArrivalsASCII is called once per source depth
        # (Bellhop/bellhop.f90:329, inside the source loop) and opens each call
        # with MAXVAL( NArr ) over the whole receiver grid
        # (Bellhop/ArrMod.f90:99) — a storage hint, not part of the data.
        _next_int(t_iter)

        for irz in range(n_depth_blocks):
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

                    # One record per arrival, in the column order of the single
                    # WRITE at Bellhop/ArrMod.f90:119-126: A, Phase,
                    # REAL(delay), AIMAG(delay), SrcDeclAngle, RcvrDeclAngle,
                    # NTopBnc, NBotBnc.
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


@typed_format_error
def read_ray_file(filepath: Union[str, Path]):
    """
    Read a Bellhop ``.ray`` file as a typed ray-bundle result.

    For ``RunType='R'`` (RAYS) the file holds ``NSz × Nalpha`` ray
    blocks in source-major order — return a :class:`Rays` for
    ``NSz == 1`` or a :class:`ResultStack` of :class:`Rays` slabs for
    ``NSz > 1``. ``EIGENRAYS`` files write a variable number of rays
    per source; this reader leaves them flat (the Bellhop wrapper
    loops Python-side for multi-source eigenrays to disambiguate).

    The ``.ray`` format is ASCII only — the only two ``.ray`` OPENs in the
    Acoustics-Toolbox tree, ``Bellhop/ReadEnvironmentBell.f90:556`` and
    ``KrakenField/EvaluateGBMod.f90:64``, are both ``FORM = 'FORMATTED'``.

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

    # Seven header records, one per line, written by
    # Bellhop/ReadEnvironmentBell.f90:557-568.
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
        # Coordinate-system marker: 'rz' (2-D, two columns per ray point,
        # take-off angle in degrees) or 'xyz' (BELLHOP3D / field3d, three
        # columns and radians) — Bellhop/ReadEnvironmentBell.f90:564-568,
        # Bellhop/WriteRay.f90:45 vs :89, Bellhop/bellhop.f90:263,289 vs
        # Bellhop/bellhop3D.f90:360.
        coord_system = _strip_fortran_quotes(f.readline())
        if coord_system != 'rz':
            raise FileFormatError(
                f"read_ray_file: {filepath} declares coordinate system "
                f"{coord_system!r}; only the 2-D 'rz' layout is supported. "
                f"An 'xyz' file stores (x, y, z) per point and its take-off "
                f"angle in radians, so reading it as (range, depth) would "
                f"return the y column as depth.",
                remediation="Re-run in 2-D (bellhop.exe), which writes the "
                            "'rz' ray file this reader parses.",
            )
        while True:
            angle_line = f.readline()
            if not angle_line:
                break
            if not angle_line.strip():
                continue
            alpha = float(angle_line.strip())
            counts_line = f.readline()
            if not counts_line or not counts_line.split():
                # A ray block is the angle record, the counts record, then
                # the points (Bellhop/WriteRay.f90:41-46); an angle with no
                # counts record is a run killed mid-write.
                raise FileFormatError(
                    f"read_ray_file: {filepath} ends after the take-off "
                    f"angle {alpha:g} with no point-count record; the file "
                    f"is truncated (the Bellhop run did not finish writing).",
                    remediation="Re-run Bellhop, or read only completed "
                                "rays from a fresh .ray file.",
                )

            # Each ray is three records (Bellhop/WriteRay.f90:41-46): the
            # take-off angle, then ``N2, NumTopBnc, NumBotBnc``, then N2 rows of
            # the coordinate pair. N2 counts the points kept after WriteRay2D's
            # subsampling, not the full step count.
            counts = counts_line.split()
            n_points = int(counts[0])
            n_top_bounces = int(counts[1]) if len(counts) > 1 else 0
            n_bot_bounces = int(counts[2]) if len(counts) > 2 else 0

            if n_points == 0:
                continue

            ray_r = []
            ray_z = []

            for _ in range(n_points):
                line = f.readline().strip()
                parts = line.split()
                if len(parts) < 2:
                    raise FileFormatError(
                        f"read_ray_file: {filepath} declares {n_points} "
                        f"points for the ray at {alpha:g} degrees but ends "
                        f"after {len(ray_r)}; the file is truncated (the "
                        f"Bellhop run did not finish writing).",
                        remediation="Re-run Bellhop, or read only completed "
                                    "rays from a fresh .ray file.",
                    )
                # Bellhop's WriteRay2D (WriteRay.f90:45) writes
                # ray2D(is)%x directly in meters (the MATLAB
                # plotray.m only divides by 1000 when the user
                # requests km output). No unit conversion here.
                ray_r.append(float(parts[0]))
                ray_z.append(float(parts[1]))

            rays.append(
                {
                    "r": np.array(ray_r),
                    "z": np.array(ray_z),
                    "alpha": alpha,
                    "n_top_bounces": n_top_bounces,
                    "n_bot_bounces": n_bot_bounces,
                }
            )

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
    # The .ray file carries no source depths, only their order, so the
    # coordinate is the index. The Bellhop wrapper substitutes the real
    # depths (and renames the coordinate) once it knows them.
    return ResultStack(
        slabs=slabs,
        coordinate=np.arange(n_sz, dtype=float),
        coordinate_name='source_index',
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
          Stored on disk in km (``Bellhop/sspMod.f90:417,422``) and converted here.
        - 'c_mat' : ndarray - Sound speed matrix in m/s, shape (n_depth, n_prof)
        - 'n_depth' : int - Number of depth points per profile

    Notes
    -----
    - File format:
        Record 1: NProf (number of range profiles)
        Record 2: r1 r2 ... rNProf (ranges in km)
        Then one record of NProf sound speeds per depth (NSSP records).
        Each record is a whole-vector list-directed READ
        (``Bellhop/sspMod.f90:417,428``), so its values may wrap across
        any number of lines; this parser accepts the same.

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
    # Canonical AT/Bellhop layout (Bellhop/sspMod.f90:407,417,428):
    #   READ 1 : NProf (integer)
    #   READ 2 : NProf range values (one whole-vector list-directed READ)
    #   then one whole-vector READ of NProf speeds per depth row.
    # Each whole-vector READ consumes records until its count is satisfied,
    # so a row may wrap across lines; the remainder of a READ's final line
    # is discarded, exactly as the Fortran does.
    with open(filepath, "r") as fid:
        n_prof = list_directed_int(fid.readline())
        r_prof = read_list_directed_values(
            fid, n_prof, f"{n_prof} profile ranges", filepath)
        # The number of depth rows isn't stored here (it lives in the .env
        # file) so we read rows until the file ends and infer NSSP.
        rows = []
        while True:
            line = fid.readline()
            if line == '':
                break
            if not _strip_fortran_comment(line).replace(',', ' ').split():
                continue
            rows.append(read_list_directed_values(
                fid, n_prof, f"SSP depth row {len(rows) + 1} "
                f"({n_prof} values)", filepath, first_line=line))
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
          Stored on disk in km (``Bellhop/sspMod.f90:621-622``) and converted here.
        - 'Segy' : ndarray - Y segment boundaries in metres, shape (Ny,).
          Also km on disk.
        - 'Segz' : ndarray - Depth values in metres, shape (Nz,)
        - 'c_mat' : ndarray - Sound speed in m/s, shape (Nz, Ny, Nx)

    Notes
    -----
    - File format (each vector/row is one whole-vector list-directed READ,
      ``Bellhop/sspMod.f90:570-617``, so its values may wrap across lines):
        Record 1: Nx
        Record 2: x1 x2 ... xNx (x segment boundaries)
        Record 3: Ny
        Record 4: y1 y2 ... yNy (y segment boundaries)
        Record 5: Nz
        Record 6: z1 z2 ... zNz (depth values)
        Then, for each depth iz=1:Nz:
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

    # Bellhop3D (Bellhop/sspMod.f90:570-617) reads each Segx / Segy / Segz
    # vector and each per-(z, y) SSP row with a single whole-vector
    # list-directed READ, which spans as many lines as it needs and
    # discards the remainder of its final line.
    def _read_vec(fid, n, what):
        return read_list_directed_values(fid, n, what, filepath)

    with open(filepath, "r") as fid:
        Nx = list_directed_int(fid.readline())
        Segx = _read_vec(fid, Nx, f"{Nx} x segment values")
        Ny = list_directed_int(fid.readline())
        Segy = _read_vec(fid, Ny, f"{Ny} y segment values")
        Nz = list_directed_int(fid.readline())
        Segz = _read_vec(fid, Nz, f"{Nz} depth values")
        c_mat = np.zeros((Nz, Ny, Nx))

        # Bellhop3D writes outermost-iz, inner-iy, then one record of Nx values.
        for iz in range(Nz):
            for iy in range(Ny):
                c_mat[iz, iy, :] = _read_vec(
                    fid, Nx, f"SSP row (iz={iz}, iy={iy}, {Nx} values)")

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
          semantics per AT ``KrakenField/field.f90:70-99`` /
          ``KrakenField/ReadModes.f90``:

          * ``opt[0]`` (source type):
            'R' = cylindrical point source (pressure),
            'X' = Cartesian line source,
            'S' = scaled-cylindrical point source.
          * ``opt[1]`` (profile mode for NProf > 1):
            'C' = coupled modes, 'A' = adiabatic.
          * ``opt[2]`` (source beam pattern, doubling as the elastic
            component selector): ``'*'`` reads a ``.sbp`` file, ``'O'`` or
            ``' '`` omnidirectional — the only three ``field.exe`` accepts
            (``KrakenField/field.f90:83-90``). The same character reaches
            ``ReadModes`` as ``Comp``, which picks one component of the
            stress-displacement vector in **elastic** media: ``'H'``
            horizontal displacement, ``'V'`` vertical displacement, ``'T'``
            tangential stress, ``'N'`` normal stress
            (``KrakenField/ReadModes.f90:315-324``). Any other letter —
            ``'P'`` by MATLAB convention — leaves acoustic pressure.
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

        # Fill missing option columns with reasonable placeholders:
        # positions 2-3 (coupling override, elastic component) default to
        # ' ' and position 4 to 'C' (coherent). Pad to three columns FIRST
        # so a one-letter option never lands the 'C' in the component
        # column (the old two-step pad turned 'R' into 'R C', i.e.
        # comp = 'C').
        opt = opt.ljust(3)
        if len(opt) <= 3:
            opt += "C"

        # Component selector lives in option column 3 (AT
        # ReadModes.f90:315-324). If the file didn't specify one, return
        # it verbatim rather than inventing a "P" default that wasn't in
        # the file — downstream code can distinguish ' ' vs 'P'.
        comp = opt[2]
        M_limit = list_directed_int(f.readline())
        log_message('oalib_reader', f"MLimit = {M_limit}", verbose=verbose)
        # ReadVector Sorts every vector it returns
        # (misc/SourceReceiverPositions.f90); rProf included (field.f90:106).
        r_prof, N_prof = _read_vector(f)
        r_prof = np.sort(r_prof)
        log_message('oalib_reader', f"Number of profiles, NProf = {N_prof}",
                    verbose=verbose)
        if N_prof < 10:
            preview = ", ".join(f"{r:.2f}" for r in r_prof)
        else:
            preview = f"{r_prof[0]:.2f} … {r_prof[-1]:.2f}"
        log_message('oalib_reader', f"profile ranges rProf (km): {preview}",
                    verbose=verbose, level='debug')

        # Receiver ranges pass through Sort in the Fortran
        # (misc/SourceReceiverPositions.f90:268).
        r_rcv, _ = _read_vector(f)
        r_rcv = np.sort(km_to_m(r_rcv))
        pos_temp = _read_sz_rz(f)
        # Rro is read by AT with Units='m' (KrakenField/field.f90:147), so the
        # column is already metres on disk — no conversion.
        # Rro passes through the same Sort (field.f90:147).
        r_offsets, N_offsets = _read_vector(f)
        r_offsets = np.sort(r_offsets)

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
          field algorithm (``'STD'``/``'PDQ'``/``'GBT'``), ``opt[3]`` is
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
        M_limit = list_directed_int(f.readline())

        s_x, n_sx = _read_vector(f)
        s_x = np.sort(s_x)
        s_y, n_sy = _read_vector(f)
        s_y = np.sort(s_y)
        pos_temp = _read_sz_rz(f)
        # Ranges and bearings pass through Sort in the Fortran
        # (misc/SourceReceiverPositions.f90).
        r_rcv, _ = _read_vector(f)
        r_rcv = np.sort(r_rcv)
        theta_rcv, _ = _read_vector(f)
        theta_rcv = np.sort(theta_rcv)
        # ReadRcvrBearings drops a duplicate closing radial (a 0..360 deg
        # fan lists the same bearing twice) —
        # misc/SourceReceiverPositions.f90:176-179.
        if (theta_rcv.size > 1
                and abs((theta_rcv[-1] - theta_rcv[0]) % 360.0) < 1e-9):
            theta_rcv = theta_rcv[:-1]

        n_nodes = list_directed_int(f.readline())
        node_x = np.zeros(n_nodes)
        node_y = np.zeros(n_nodes)
        mode_files = []
        for i in range(n_nodes):
            parts = _strip_fortran_comment(f.readline()).split()
            node_x[i] = fortran_float(parts[0])
            node_y[i] = fortran_float(parts[1])
            mode_files.append(parts[2].strip("'\""))

        n_elts = list_directed_int(f.readline())
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
    # SubTab is followed by Sort in the Fortran
    # (misc/SourceReceiverPositions.f90:224,268), so the solver computes on
    # ascending depth axes whatever order the deck listed them in.
    sz, _ = _read_vector(fid)
    rz, _ = _read_vector(fid)

    return {"sz": np.sort(sz), "rz": np.sort(rz)}


@typed_format_error
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

    # Remaining tokens are time-series records: (1 time + nr pressures) per step
    # (Scooter/sparc.f90:294,299 write ``tout( Itout ), values( 1 : nr )``).
    # Floor-divide so a run killed mid-record contributes no partial time step,
    # which would otherwise shift every later sample by one column.
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

    ``method='fft'`` (the only method) extracts the spectral bin nearest
    ``frequency`` from a Hanning-windowed FFT and returns
    ``(p_at_freq, ranges)`` where ``p_at_freq`` is the model-native,
    source-normalised complex pressure suitable for wrapping in a complex
    narrowband :class:`Field` (``coords={'depth', 'range'}``,
    ``phase_reference='travelling_wave'``).

    A post-processing utility for a ``.rts`` read by :func:`read_rts_file`;
    :class:`uacpy.models.SPARC` returns ``p(t)`` and does not call it. The
    projection is not a calibrated substitute for Kraken/Scooter TL — see the
    module docstring of ``uacpy/tests/test_sparc_output_modes.py``.
    """
    p = rts_data["p"]
    dt = rts_data["dt"]
    ranges = rts_data["ranges"]

    nt = p.shape[0]

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
        # Steady-tone amplitude from one rfft bin: the 2.0 restores the half of
        # the tone's energy that sits in the negative-frequency bin rfft drops,
        # and dividing by the window's coherent gain sum(w) undoes both the
        # 1/N of the unnormalised transform and the taper's amplitude loss. On
        # a pure tone at bin centre the pair returns the tone's own amplitude
        # and phase, whatever nt and whatever window.
        window = np.hanning(nt)
        p_freq = np.fft.rfft(p * window[:, np.newaxis], axis=0)
        freqs = np.fft.rfftfreq(nt, dt)
        freq_idx = np.argmin(np.abs(freqs - frequency))
        p_at_freq = 2.0 * p_freq[freq_idx, :] / np.sum(window)
    else:
        raise ConfigurationError(
            f"rts_to_pressure: unknown method {method!r}; only 'fft' is "
            f"supported (the 'goertzel' single-bin DFT was removed — it "
            f"reproduced the rfft bin to machine precision but skipped the "
            f"window, so its answer was not comparable)."
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
    - Then a free whitespace-separated token stream: nrd (number of
      receiver depths), nrd depth values (m), then repeating blocks of
      ``1 + nrd`` values — ``t, RTS(rd_1, t), ..., RTS(rd_nrd, t)``.

    ``read_ts.m`` reads everything after the title with ``fscanf`` —
    ``rz = fscanf(fid, '%f', nrz)`` then
    ``temp = fscanf(fid, '%f', [nrz + 1, inf])`` — which ignores line
    breaks entirely, so this parser tokenises the whole stream the same
    way. A trailing partial time-step block (a run killed mid-write) is
    dropped, matching the MATLAB column-major fill.

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
    # Everything after the title line is a free fscanf-style token stream
    # (read_ts.m:33-35): line breaks carry no meaning at all.
    with open(filepath, 'r') as f:
        plot_title = f.readline().strip()
        tokens = f.read().split()

    if not tokens:
        raise FileFormatError(
            f"read_ts: {filepath} carries no data after the title line."
        )
    nrd = int(fortran_float(tokens[0]))
    if nrd <= 0:
        raise FileFormatError(
            f"read_ts: {filepath} declares {nrd} receiver depths."
        )
    rd_toks, cursor = take_tokens(tokens, 1, nrd,
                                  f"{nrd} receiver depths", filepath)
    rd = np.array([fortran_float(t) for t in rd_toks])

    # Repeating (1 + nrd)-value time-step blocks; a partial trailing block
    # (run killed mid-write) contributes nothing, as in the MATLAB
    # column-major [nrz + 1, inf] fill.
    rest = tokens[cursor:]
    block = 1 + nrd
    nt = len(rest) // block
    data = np.array([fortran_float(t) for t in rest[:nt * block]],
                    dtype=float).reshape(nt, block)
    tout = data[:, 0]
    RTS = data[:, 1:]

    return {
        'PlotTitle': plot_title,
        'pos': {'r': {'z': rd}},
        'tout': tout,
        'RTS': RTS
    }
