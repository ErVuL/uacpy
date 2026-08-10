"""
Readers for Kraken normal-mode files (``.mod`` binary, ``.moa`` ASCII).

* ``read_modes`` — auto-detect binary/ASCII by extension.
* ``read_modes_bin`` — binary ``.mod``.
* ``read_modes_asc`` — ASCII ``.moa``.
* ``get_component`` — extract a column from a Kraken modes dict.
"""

import os
from typing import Any, Dict, Optional, Union

import numpy as np

from uacpy.core.acoustics import pekeris_root
from uacpy.core.exceptions import (
    ConfigurationError, FileFormatError,
)
from uacpy.io._fortran_helpers import PARSE_ERRORS, detect_endian


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
        - 'N': Number of mesh points per medium. Read only by the KRAKEL
          variant of the depth loop, which this does not take (see below).
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

    Mirrors ``Matlab/ReadWrite/get_component.m`` (mbp, 8/14/2010). No
    ``krakel`` binary ships in ``uacpy/bin/oalib``, so with the shipped
    solvers ``Mater`` never contains ``'ELASTIC'`` and the four-component
    branch is unreachable; it is kept because the file format admits it.

    Examples
    --------
    >>> # Extract horizontal displacement from elastic modes
    >>> phi_h = get_component(modes, 'H')

    >>> # Extract pressure from acoustic modes (any component)
    >>> phi_p = get_component(modes, 'H')  # Same as 'V', 'T', 'N'
    """
    # Mirrors ``Matlab/ReadWrite/get_component.m``, which differs from AT's
    # Fortran EXTRACT (``KrakenField/ReadModes.f90:307-331``) in exactly one
    # place: the depth-loop bound. EXTRACT walks ``N(Medium)+1`` grid points
    # per medium, which is right for KRAKEL — it tabulates modes on the
    # finite-difference grid — while KRAKEN/KRAKENC subtabulate to the
    # receiver depths, so the MATLAB walks ``length(Modes.z)`` and keeps
    # ``Modes.N`` only as a commented-out KRAKEL alternative
    # (``get_component.m:6-7,16-18``). Two cursors, because the axes advance
    # at different rates: ``k`` walks the packed stress-displacement vector,
    # 1 entry per depth point in an acoustic medium but 4 (H, V, T, N) in an
    # elastic one, while ``jj`` walks depth points. ``phi_full`` is trusted
    # only up to its own length: a KRAKEN run does not tabulate the elastic
    # media, so the stored block can be shorter than ``Nmedia``/``Mater``
    # imply (``get_component.m:19-23``).
    phi = []
    jj = 0
    k = 0

    Nmedia = modes_dict.get("Nmedia", 1)
    phi_full = modes_dict["phi"]
    z = modes_dict["z"]
    Mater = modes_dict.get("Mater", [["ACOUSTIC"]])
    for medium in range(Nmedia):
        for ii in range(len(z)):
            if k >= phi_full.shape[0]:
                break
            if medium < len(Mater):
                material = (
                    Mater[medium].strip()
                    if isinstance(Mater[medium], str)
                    else str(Mater[medium]).strip()
                )
            else:
                material = "ACOUSTIC"

            if material == "ACOUSTIC":
                if jj < len(z):
                    phi.append(phi_full[k, :])
                k += 1

            elif material == "ELASTIC":
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
                        raise ConfigurationError(f"Unknown component: {comp}")
                k += 4

            else:
                raise ConfigurationError(f"Unknown material type: {material}")

            jj += 1

    if not phi:
        raise FileFormatError(
            "get_component: the modes set contains no readable modes (M=0) — "
            "nothing to extract. The waveguide is likely below modal cutoff at "
            "this frequency; check the .mod record before requesting a component."
        )
    return np.array(phi)


class _TokenStream:
    """Whitespace-delimited token stream that also knows about line breaks.

    ``read_modes_asc.m`` alternates two MATLAB primitives: ``fscanf(fid,
    '%f', N)``, which is free-format and runs across line breaks, and
    ``fgetl(fid)``, which returns the remainder of the *current* line. This
    reproduces both against the same cursor, so a record that a Fortran
    runtime wrapped across several lines still parses.
    """

    def __init__(self, text: str, filename: str):
        self._lines = text.splitlines()
        self._filename = filename
        self._i = 0
        self._toks = self._lines[0].split() if self._lines else []
        self._j = 0

    def _next_line_start(self) -> None:
        self._i += 1
        self._j = 0
        self._toks = (
            self._lines[self._i].split() if self._i < len(self._lines) else []
        )

    def line(self) -> str:
        """Remainder of the current line; advances to the next one."""
        if self._i >= len(self._lines):
            raise FileFormatError(
                f"Modes file {self._filename}: EOF while reading a text line"
            )
        rest = ' '.join(self._toks[self._j:])
        self._next_line_start()
        return rest

    def floats(self, n: int) -> np.ndarray:
        """``n`` free-format floats, crossing line breaks as needed."""
        vals = []
        while len(vals) < n:
            if self._j >= len(self._toks):
                if self._i >= len(self._lines):
                    raise FileFormatError(
                        f"Modes file {self._filename}: EOF while reading "
                        f"{n} floats (got {len(vals)})"
                    )
                self._next_line_start()
                continue
            vals.append(float(self._toks[self._j]))
            self._j += 1
        return np.array(vals, dtype=float)

    def complex_pairs(self, n: int) -> np.ndarray:
        """``n`` complex values stored as ``re im re im …`` token pairs.

        ``read_modes_asc.m:35,52`` reads these with ``fscanf(fid, '%f',
        [2, n])``; MATLAB fills that 2×n matrix column-major, so row 1 is
        the real parts and row 2 the imaginary parts of *consecutive*
        tokens — the layout is interleaved, not two blocks.
        """
        vals = self.floats(2 * n)
        return vals[0::2] + 1j * vals[1::2]


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
        - 'M': Number of modes returned — ``len(k)`` (see Notes)
        - 'z': Depth vector (m), shape (ntot,)
        - 'k': Complex wavenumbers (1/m), shape (nmodes_read,)
        - 'phi': Mode shapes, shape (ntot, nmodes_read)

    Notes
    -----
    File format (.moa ASCII):
    - Record length (not used in ASCII)
    - Title line
    - freq, Nmedia, ntot, nmat, M
    - Nmedia lines of medium properties
    - Top halfspace line
    - Bottom halfspace line
    - Blank line
    - Depth vector (ntot floats)
    - Wavenumbers: ``M`` interleaved ``re im`` pairs
    - Per mode: a separator line then ``ntot`` interleaved ``re im`` pairs

    Both complex records store **interleaved pairs**, not a real block
    followed by an imaginary block: ``read_modes_asc.m:35,52`` reads them
    with ``fscanf(fid, '%f', [2, n])`` and MATLAB fills that matrix
    column-major. That script is a layout reference only — as vendored it
    reads counts from bare ``ntot``/``m`` names it never defines
    (``read_modes_asc.m:33,35``), so it does not run.

    ``'M'`` is the number of modes actually returned (always ``len(k)``
    and ``phi.shape[1]``), which is the file's total only when ``modes``
    is ``None``. :func:`read_modes_bin` reports it the same way.

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
            stream = _TokenStream(fid.read(), filename)

            int(stream.line().split()[0])  # lrecl (unused for ASCII)
            pltitl = stream.line().strip()
            params = stream.floats(5)
            freq = float(params[0])
            Nmedia = int(params[1])
            ntot = int(params[2])
            nmat = int(params[3])
            M = int(params[4])  # total number of modes in the file
            stream.line()       # rest of the parameter record's line
            for _ in range(Nmedia):
                stream.line()   # medium properties, unused

            def _parse_halfspace(line: str) -> Dict[str, Any]:
                """Parse a Kraken halfspace summary line.

                No Acoustics-Toolbox program writes an ASCII mode file, so
                the ASCII rendering of this record is inferred: the field
                order mirrors the binary halfspace record at
                ``Kraken/kraken.f90:603-605`` — boundary-condition character,
                then ``cP_real cP_imag cS_real cS_imag rho depth``. Missing
                trailing fields default to zero so vacuum / rigid halfspaces
                parse cleanly.
                """
                toks = line.split()
                if not toks:
                    return {"BC": " ", "cp": 0.0 + 0j, "cs": 0.0 + 0j,
                            "rho": 0.0, "depth": 0.0}
                bc = toks[0]
                vals = []
                for t in toks[1:]:
                    try:
                        vals.append(float(t))
                    except ValueError:
                        break
                vals = (vals + [0.0] * 6)[:6]
                return {
                    "BC": bc,
                    "cp": complex(vals[0], vals[1]),
                    "cs": complex(vals[2], vals[3]),
                    "rho": vals[4],
                    "depth": vals[5],
                }

            Top = _parse_halfspace(stream.line())
            Bot = _parse_halfspace(stream.line())
            stream.line()  # blank line
            z = stream.floats(ntot)
            stream.line()  # rest of the depth record's last line
            k_all = stream.complex_pairs(M)
            if modes is None:
                modes_to_read = list(range(1, M + 1))  # 1-indexed
            elif isinstance(modes, (int, np.integer)):
                modes_to_read = [int(modes)]
            else:
                modes_to_read = [int(m) for m in modes]
            modes_to_read = [m for m in modes_to_read if 1 <= m <= M]
            k_selected = k_all[[m - 1 for m in modes_to_read]]  # Convert to 0-indexed
            phi = np.zeros((ntot, len(modes_to_read)), dtype=complex)

            # Stop at the highest requested mode (read_modes_asc.m:50,
            # `for mode = 1: max( modes )`) — a file truncated after it still
            # satisfies the request.
            for mode_num in range(1, (max(modes_to_read) if modes_to_read else 0) + 1):
                stream.line()  # rest of the previous record's line
                phi_mode = stream.complex_pairs(ntot)
                if mode_num in modes_to_read:
                    idx = modes_to_read.index(mode_num)
                    phi[:, idx] = phi_mode

    except FileNotFoundError as e:
        raise ConfigurationError(
            f"ASCII mode file not found: {filename}. No Acoustics-Toolbox "
            "program writes .moa — supply the path to an existing ASCII mode "
            "file, or read a solver-written binary .mod with read_modes_bin."
        ) from e
    except PARSE_ERRORS as e:
        raise FileFormatError(
            f"Malformed Kraken mode file {filename}: {e}"
        ) from e
    return {
        "pltitl": pltitl,
        "title": pltitl,
        "freq": freq,
        "freqVec": np.asarray([freq], dtype=float),
        "Nfreq": 1,
        "Nmedia": Nmedia,
        "ntot": ntot,
        "nmat": nmat,
        "M": int(len(k_selected)),  # modes returned, not the file's total
        "z": z,
        "k": k_selected,
        "phi": phi,
        "Top": Top,
        "Bot": Bot,
    }


def _fortran_div(numerator: int, denominator: int) -> int:
    """Fortran integer division: truncation toward zero.

    ``kraken.f90:109,117`` form record counts with ``( 2 * M - 1 ) /
    LRecordLength``. For ``M == 0`` the numerator is negative and Fortran
    truncates to ``0`` where Python's ``//`` floors to ``-1``, which would
    misplace every record after a zero-mode frequency block.
    """
    if numerator < 0:
        return -((-numerator) // denominator)
    return numerator // denominator


def read_modes_bin(
    filename: str,
    frequency: float = 0.0,
    modes: Optional[Union[int, list, np.ndarray]] = None,
    profile: int = 1,
) -> Dict[str, Any]:
    """Read a KRAKEN binary ``.mod`` file, converting any malformed-file
    parse error into a typed :class:`FileFormatError` (a truncated/garbage
    file otherwise surfaces as a bare ``IndexError`` / ``struct.error`` from
    the record reads)."""
    try:
        return _read_modes_bin_impl(filename, freq=frequency, modes=modes,
                                    profile=profile)
    except FileFormatError:
        raise
    except FileNotFoundError as e:
        raise FileFormatError(f"Mode file not found: {filename}") from e
    except PARSE_ERRORS as e:
        raise FileFormatError(
            f"Malformed Kraken mode file {filename}: {e}"
        ) from e


def _read_modes_bin_impl(
    filename: str,
    freq: float = 0.0,
    modes: Optional[Union[int, list, np.ndarray]] = None,
    profile: int = 1,
) -> Dict[str, Any]:
    """
    Read mode data from KRAKEN binary format (.mod file).

    This function reads normal mode data including eigenvalues (wavenumbers),
    eigenfunctions (mode shapes), and environmental parameters from KRAKEN
    model output files.

    Parameters
    ----------
    filename : str
        Mode file path. If no extension is given, ``.mod`` is appended
        (this is the extension that Kraken actually emit per
        ``Kraken/kraken.f90`` — ``OPEN(FILE=TRIM(FileRoot)//'.mod', ...)``).
    freq : float, optional
        Frequency in Hz for which to read modes. For broadband runs,
        selects the closest frequency. Use freq=0 if only one frequency.
        Default is 0.0.
    modes : int, list, or ndarray, optional
        Mode indices to read (1-indexed). If None, reads all modes.
        Can be:
        - int: read that single mode (1-indexed)
        - list/array: read specific mode indices
    profile : int, optional
        Profile index to read, 1-indexed (default 1). A ``.mod`` holds one
        mode set per environmental profile — ``Kraken/kraken.f90:42`` loops
        ``Profile: DO iProf = 1, 9999`` and each profile restarts with its
        own five header records (``KrakenField/ReadModes.f90:19-25``). A
        range-dependent Kraken field run writes one profile per segment.

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
        - 'M' : int - Number of modes returned — always ``len(k)`` and
          ``phi.shape[1]``. Equals the number the solver found unless
          ``modes`` selected a subset. :func:`read_modes_asc` reports it
          the same way.
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
    - The canonical extension is ``.mod`` (binary direct-access produced by
      Kraken). Any explicit extension on ``filename`` is honoured;
      otherwise ``.mod`` is appended.
    - Modes are stored in Fortran unformatted direct-access binary.
    - Record length (lrecl) is determined from first 4 bytes.
    - Mode indices are 1-indexed (MATLAB/Fortran convention).
    - Record layout per profile (``KrakenField/ReadModes.f90:19-25``): five
      header records, then per frequency a mode-count record, a halfspace
      record, ``M`` eigenfunction records and the eigenvalues folded across
      ``1 + (2M-1)/LRecordLength`` records of ``LRecordLength/2`` complex
      values each (``Kraken/kraken.f90:106-117``).

    References
    ----------
    Mirrors ``Matlab/ReadWrite/read_modes_bin.m``, itself derived from
    ``readKRAKEN.m``, Aaron Thode, 1996 (``read_modes_bin.m:16``).

    Examples
    --------
    >>> # Read all modes at 100 Hz
    >>> modes = read_modes_bin('pekeris', frequency=100.0)
    >>> print(f"Number of modes: {modes['M']}")
    >>> print(f"Wavenumber of mode 1: {modes['k'][0]}")

    >>> # Read specific modes
    >>> modes = read_modes_bin('pekeris', frequency=100.0, modes=[1, 2, 3])
    >>> print(f"Mode shapes: {modes['phi'].shape}")
    """
    if profile < 1:
        raise ConfigurationError(
            f"read_modes_bin: profile must be >= 1 (got {profile}); mode-file "
            "profiles are numbered from 1 (Kraken/kraken.f90:42)."
        )
    if not os.path.splitext(filename)[1]:
        filename = filename + ".mod"

    with open(filename, "rb") as fid:
        head = fid.read(4)
        if len(head) < 4:
            raise FileFormatError(f"Invalid mode file (truncated header): {filename}")
        endian = detect_endian(
            head, source=f'read_modes_bin:{os.path.basename(filename)}')
        i4 = np.dtype(endian + 'i4')
        f4 = np.dtype(endian + 'f4')
        f8 = np.dtype(endian + 'f8')
        fid.seek(0, 2)
        file_size = fid.tell()
        max_items = file_size // 4
        fid.seek(0, 0)
        lrecl_words = int(np.fromfile(fid, dtype=i4, count=1)[0])
        # LRecordLength is a count of 4-byte `longwords' and each eigenvalue
        # record holds LRecordLength/2 complex values (kraken.f90:587,110), so
        # anything below 2 cannot carry a single eigenvalue.
        if lrecl_words < 2:
            raise FileFormatError(
                f"Invalid mode file: record length LRecordLength={lrecl_words} "
                f"words (must be a positive word-count of at least 2): {filename}"
            )
        lrecl = 4 * lrecl_words

        def _n_wavenumber_records(M: int) -> int:
            """Eigenvalue records for ``M`` modes (kraken.f90:109)."""
            return 1 + _fortran_div(2 * M - 1, lrecl_words)

        def _block_end_record(rec: int, M: int) -> int:
            """One past the last record this reader touches for the frequency
            block whose mode-count record is ``rec``.

            ``kraken.f90:106-113`` writes the count, the halfspace record,
            ``M`` eigenfunction records and the folded eigenvalue records.
            A zero-mode run never reaches that writer: ``kraken.f90:958-961``
            writes the profile header, puts ``M`` at ``iRecProfile + 6`` and
            calls ERROUT, which STOPs (``misc/FatalError.f90:30``). This
            reader reads the halfspace slot before it tests ``M``, so
            ``rec + 2`` still bounds what it touches.
            """
            if M == 0:
                return rec + 2
            return rec + 2 + M + _n_wavenumber_records(M)

        def _read_header(hdr: int) -> Dict[str, Any]:
            """Read the five descriptive records of the profile at record
            ``hdr`` (kraken.f90:593-599, ReadModes.f90:20)."""
            if (hdr + 5) * lrecl > file_size:
                raise FileFormatError(
                    f"Invalid mode file {filename}: profile header at record "
                    f"{hdr} needs {(hdr + 5) * lrecl} bytes but the file is "
                    f"{file_size} bytes."
                )
            fid.seek(hdr * lrecl + 4, 0)   # past this profile's LRecordLength
            title = fid.read(80).decode("ascii", errors="ignore").strip()
            Nfreq, Nmedia, Ntot, NMat = (
                int(v) for v in np.fromfile(fid, dtype=i4, count=4)
            )

            # File-size-aware sanity bound on the header counts before any
            # array is sized off them. A corrupt/hostile header (e.g.
            # NMat=0x7fffffff) would otherwise drive a multi-GB np.zeros below.
            # The smallest a single sample can occupy on disk is 4 bytes
            # (float32 / int32), so no count of 4-byte items can exceed the
            # remaining file size; use that as a generous upper bound.
            for _name, _val in (("Nfreq", Nfreq), ("Nmedia", Nmedia),
                                ("Ntot", Ntot), ("NMat", NMat)):
                if _val < 0 or _val > max_items:
                    raise FileFormatError(
                        f"Invalid mode file: header count {_name}={_val} is "
                        f"implausible for a {file_size}-byte file "
                        f"(max {max_items} 4-byte items)."
                    )
            if Nfreq < 1:
                raise FileFormatError(
                    f"Invalid mode file: Nfreq={Nfreq} (need at least one "
                    f"frequency block): {filename}"
                )

            # Records hdr+1..hdr+4 (kraken.f90:594-598, read back at
            # ReadModes.f90:185-188). The first two are implied-DO pair lists
            # over the media, so the two quantities interleave: int32 N with
            # CHARACTER*8 Material, then REAL*4 depth with REAL*4 rho — hence
            # the (2, Nmedia) Fortran-order reshape. freqVec is REAL(KIND=8)
            # (SourceReceiverPositions.f90:14) while zTab is a default REAL
            # and depth/rho are written through REAL(), so those are f4.
            fid.seek((hdr + 1) * lrecl, 0)
            N = []
            Mater = []
            for _ in range(Nmedia):
                N.append(int(np.fromfile(fid, dtype=i4, count=1)[0]))
                Mater.append(fid.read(8).decode("ascii", errors="ignore").strip())
            fid.seek((hdr + 2) * lrecl, 0)
            bulk = np.fromfile(fid, dtype=f4, count=2 * Nmedia).reshape(
                (2, Nmedia), order="F"
            )
            fid.seek((hdr + 3) * lrecl, 0)
            freqVec = np.fromfile(fid, dtype=f8, count=Nfreq)
            fid.seek((hdr + 4) * lrecl, 0)
            z = np.fromfile(fid, dtype=f4, count=Ntot)
            return {
                "title": title, "Nfreq": Nfreq, "Nmedia": Nmedia,
                "Ntot": Ntot, "NMat": NMat, "N": N, "Mater": Mater,
                "depth": bulk[0, :], "rho": bulk[1, :],
                "freqVec": freqVec, "z": z,
            }

        def _read_mode_count(rec: int) -> int:
            """Read ``M`` from record ``rec`` and bound it against the file.

            ``M`` is a plain header word (kraken.f90:106) that sizes every
            allocation below, so it gets the same file-size bound as the
            record-0 counts.
            """
            if (rec + 1) * lrecl > file_size:
                raise FileFormatError(
                    f"Invalid mode file {filename}: mode-count record {rec} "
                    f"starts past the end of a {file_size}-byte file."
                )
            fid.seek(rec * lrecl, 0)
            M = int(np.fromfile(fid, dtype=i4, count=1)[0])
            if M < 0 or _block_end_record(rec, M) * lrecl > file_size:
                raise FileFormatError(
                    f"Invalid mode file {filename}: mode count M={M} at record "
                    f"{rec} needs {_block_end_record(rec, M) * lrecl} bytes "
                    f"but the file is {file_size} bytes."
                )
            return M

        # Walk the preceding profiles: each is a five-record header followed
        # by one block per frequency (ReadModes.f90:19-25,125).
        hdr = 0
        for _ in range(profile - 1):
            rec = hdr + 5
            for _ in range(_read_header(hdr)["Nfreq"]):
                M_prev = _read_mode_count(rec)
                rec += 3 + M_prev + _fortran_div(2 * M_prev - 1, lrecl_words)
            hdr = rec

        header = _read_header(hdr)
        title = header["title"]
        Nfreq = header["Nfreq"]
        Nmedia = header["Nmedia"]
        NMat = header["NMat"]
        N = header["N"]
        Mater = header["Mater"]
        depth = header["depth"]
        rho = header["rho"]
        freqVec = header["freqVec"]
        z = header["z"]

        freq_diff = np.abs(freqVec - freq)
        freq_index = int(np.argmin(freq_diff))
        # Records hdr+0..hdr+3: header, N/Mater, depth/rho, freqVec
        # Record hdr+4: z vector
        # Record hdr+5: M (mode count) — where the first frequency block starts
        iRecProfile = hdr + 5
        for ifreq in range(freq_index + 1):
            M = _read_mode_count(iRecProfile)
            if ifreq < freq_index:
                # Advance to the next frequency block (kraken.f90:117).
                iRecProfile += 3 + M + _fortran_div(2 * M - 1, lrecl_words)
        if modes is None:
            modes = np.arange(1, M + 1)
        elif isinstance(modes, (int, np.integer)):
            modes = np.array([modes])      # single mode #N, matching read_modes (ASCII)
        else:
            modes = np.array(modes)
        # AT mode numbers are 1-based; keep only existing modes. read_modes_bin.m
        # filters `<= M` only because MATLAB's 1-based indexing rejects negatives;
        # the Python port must also guard the lower bound, else k[modes-1] silently
        # wraps for a negative index (returning the wrong mode).
        modes = modes[(modes >= 1) & (modes <= M)]
        # Top/Bot halfspace block sits at REC iRecProfile+1 per
        # kraken.f90:603 and read_modes_bin.m:129-131.
        fid.seek((iRecProfile + 1) * lrecl, 0)
        Top = {}
        Top["BC"] = chr(np.fromfile(fid, dtype=np.uint8, count=1)[0])
        cp_data = np.fromfile(fid, dtype=f4, count=2)
        Top["cp"] = complex(cp_data[0], cp_data[1])
        cs_data = np.fromfile(fid, dtype=f4, count=2)
        Top["cs"] = complex(cs_data[0], cs_data[1])
        Top["rho"] = np.fromfile(fid, dtype=f4, count=1)[0]
        Top["depth"] = np.fromfile(fid, dtype=f4, count=1)[0]
        Bot = {}
        Bot["BC"] = chr(np.fromfile(fid, dtype=np.uint8, count=1)[0])
        cp_data = np.fromfile(fid, dtype=f4, count=2)
        Bot["cp"] = complex(cp_data[0], cp_data[1])
        cs_data = np.fromfile(fid, dtype=f4, count=2)
        Bot["cs"] = complex(cs_data[0], cs_data[1])
        Bot["rho"] = np.fromfile(fid, dtype=f4, count=1)[0]
        Bot["depth"] = np.fromfile(fid, dtype=f4, count=1)[0]
        if M == 0:
            # Same shapes the M > 0 path yields for an empty selection, so a
            # zero-mode file stays consumable as (ntot, 0) / (0,).
            phi = np.zeros((NMat, 0), dtype=np.complex64)
            k = np.zeros(0, dtype=np.complex64)
        else:
            phi = np.zeros((NMat, len(modes)), dtype=np.complex64)

            # Mode ``m`` (1-based) sits one record past the halfspace record:
            # ReadModes.f90:243 reads REC = IRecProfile + 1 + Mode as NMat
            # COMPLEX*8 values, i.e. 2*NMat interleaved re/im float32.
            for ii, mode_idx in enumerate(modes):
                rec = iRecProfile + 1 + int(mode_idx)
                fid.seek(rec * lrecl, 0)
                phi_data = np.fromfile(fid, dtype=f4, count=2 * NMat).reshape(
                    (2, NMat), order="F"
                )
                phi[:, ii] = phi_data[0, :] + 1j * phi_data[1, :]
            # The eigenvalues are folded across records: kraken.f90:108-113
            # writes LRecordLength/2 complex values per record, each starting
            # on a record boundary, and KrakenField/ReadModes.f90:212-219
            # reads them back the same way. An odd LRecordLength leaves one
            # unwritten word at the end of every eigenvalue record, which a
            # single contiguous read would absorb as data. ``irec`` counts
            # from 0 where the Fortran IREC counts from 1, hence ``+ 2`` here
            # against kraken.f90:111's ``+ 1``.
            k_all = np.zeros(M, dtype=np.complex64)
            per_record = lrecl_words // 2
            ifirst = 0
            for irec in range(_n_wavenumber_records(M)):
                ilast = min(M, ifirst + per_record)
                fid.seek((iRecProfile + 2 + M + irec) * lrecl, 0)
                vals = np.fromfile(fid, dtype=f4, count=2 * (ilast - ifirst))
                k_all[ifirst:ilast] = vals[0::2] + 1j * vals[1::2]
                ifirst = ilast
            if ifirst < M:
                # kraken.f90:109 sizes the loop as 1 + (2M-1)/LRecordLength
                # while :110 fills only LRecordLength/2 values per record; for
                # an odd LRecordLength those disagree and the trailing
                # eigenvalues are never written.
                raise FileFormatError(
                    f"Mode file {filename} declares M={M} modes but its "
                    f"eigenvalue records hold only {ifirst}: LRecordLength="
                    f"{lrecl_words} is odd, so the writer's record count "
                    f"(kraken.f90:109) undershoots its own per-record payload "
                    f"of {per_record} values (kraken.f90:110).",
                    remediation="Re-run the solver with an even "
                                "LRecordLength: it is MAX(2*Nfreq, 2*NzTab, "
                                "32, 3*NAcoustic) (kraken.f90:587), so an "
                                "extra mode-tabulation depth or an even "
                                "number of acoustic media removes the fold "
                                "mismatch.",
                )
            k = k_all[modes - 1]  # 0-indexed; select the requested modes

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
        "M": int(len(k)),   # modes returned, not the file's total
        "phi": phi,
        "k": k,
        "Top": Top,
        "Bot": Bot,
    }


def read_modes(
    filename: str,
    frequency: float = 0.0,
    modes: Optional[Union[int, list, np.ndarray]] = None,
    profile: int = 1,
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
    frequency : float, optional
        Frequency in Hz to select from multi-frequency files (default: 0)
    modes : int, list, or ndarray, optional
        Mode indices to extract (1-indexed). If None, all modes are returned.
    profile : int, optional
        Profile index to read from a binary ``.mod`` (1-indexed, default 1).
        The ASCII format carries a single mode set, so any other value is a
        :class:`ConfigurationError` there.

    Returns
    -------
    modes_data : dict
        Mode data dictionary with fields from read_modes_bin or read_modes_asc,
        plus computed halfspace parameters:
        - 'Top': dict with top halfspace properties (k2, gamma, phi)
        - 'Bot': dict with bottom halfspace properties (k2, gamma, phi)

    Notes
    -----
    ``Modes['M']`` is the number of modes returned (``len(Modes['k'])``)
    from either reader; the halfspace parameters below are computed only
    when it is non-zero.

    For acoustic halfspaces (boundary condition 'A'), computes:
    - k²: wavenumber squared in halfspace
    - γ: vertical wavenumber using Pekeris root, from the full complex
      eigenvalue for a KRAKENC file and from ``Re(k)`` otherwise
    - φ: mode value at interface

    The frequency index is found by searching for the closest match to
    the requested frequency in freqVec.

    Translated from OALIB read_modes.m

    Examples
    --------
    >>> # Read binary mode file
    >>> modes = read_modes('test.mod', frequency=100.0)
    >>> print(f"Number of modes: {modes['M']}")
    >>> print(f"Wavenumbers shape: {modes['k'].shape}")

    >>> # Read specific modes
    >>> modes = read_modes('test.mod', frequency=100.0, modes=[1, 2, 3])

    >>> # ASCII format
    >>> modes = read_modes('test.moa')
    """
    fileroot, ext = os.path.splitext(filename)

    if not ext:
        ext = ".mod"  # Default extension

    filename = fileroot + ext
    if ext == ".mod":
        Modes = read_modes_bin(filename, frequency, modes, profile=profile)

    elif ext == ".moa":
        if profile != 1:
            raise ConfigurationError(
                f"read_modes: profile={profile} requested for an ASCII mode "
                "file; the ASCII layout holds a single mode set. Read a "
                "binary .mod to reach further profiles."
            )
        Modes = read_modes_asc(filename, modes)

    else:
        raise ConfigurationError(f"read_modes: Unrecognized file extension {ext}")
    freq_diff = np.abs(Modes["freqVec"] - frequency)
    freq_index = int(np.argmin(freq_diff))
    f_selected = float(Modes["freqVec"][freq_index])
    # KRAKENC keeps the full complex eigenvalue in the half-space vertical
    # wavenumber; KRAKEN discards the imaginary part, which is a first-order
    # perturbation there. KrakenField/ReadModes.f90:79 takes the model from
    # Title(1:7) and :254-272 switches on it.
    k_gamma = (Modes["k"] if str(Modes["title"])[:7].upper() == "KRAKENC"
               else np.real(Modes["k"]))
    if Modes["M"] != 0:
        if Modes["Top"]["BC"] == "A":
            k_top = 2.0 * np.pi * f_selected / Modes["Top"]["cp"]
            Modes["Top"]["k2"] = k_top**2
            gamma2 = k_gamma ** 2 - Modes["Top"]["k2"]
            Modes["Top"]["gamma"] = pekeris_root(gamma2)
            Modes["Top"]["phi"] = Modes["phi"][0, :]
        else:
            # A non-acoustic halfspace carries no evanescent tail, so its
            # interface terms are zeroed. The coupled-mode evaluator divides
            # the interface mode value by rho (``Matlab/Kraken/evalcm.m:192,
            # 197``); rho = 1 keeps that a well-defined zero whatever the file
            # stores for a vacuum or rigid boundary (read_modes.m:87,98).
            Modes["Top"]["rho"] = 1.0
            Modes["Top"]["gamma"] = np.zeros_like(Modes["k"])
            Modes["Top"]["phi"] = np.zeros_like(Modes["phi"][0, :])
        if Modes["Bot"]["BC"] == "A":
            k_bot = 2.0 * np.pi * f_selected / Modes["Bot"]["cp"]
            Modes["Bot"]["k2"] = k_bot**2
            gamma2 = k_gamma ** 2 - Modes["Bot"]["k2"]
            Modes["Bot"]["gamma"] = pekeris_root(gamma2)
            Modes["Bot"]["phi"] = Modes["phi"][-1, :]
        else:
            Modes["Bot"]["rho"] = 1.0
            Modes["Bot"]["gamma"] = np.zeros_like(Modes["k"])
            Modes["Bot"]["phi"] = np.zeros_like(Modes["phi"][-1, :])

    return Modes
