"""
Readers for Kraken normal-mode files (binary ``.mod``, ASCII ``.moa``).

* ``read_modes`` — read a ``.mod`` and attach the halfspace parameters.
* ``read_modes_bin`` — binary ``.mod``.
* ``read_modes_asc`` — ASCII ``.moa``.
* ``get_component`` — one component of the stress-displacement vector of an
  elastic-medium mode set.

The binary direct-access ``.mod`` is the only mode format any
Acoustics-Toolbox program *writes*, so :func:`read_modes` takes ``.mod``
only and a non-``.mod`` path raises
:class:`~uacpy.core.exceptions.FileFormatError`. :func:`read_modes_asc` is
the reader for an ASCII ``.moa`` produced elsewhere — the AT Matlab tools or
another OALIB-family code — and is called directly, not through
:func:`read_modes`.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from uacpy.core.acoustics import pekeris_root
from uacpy.core.exceptions import (
    ConfigurationError, FileFormatError,
)
from uacpy.io._fortran_helpers import (
    PARSE_ERRORS, detect_endian, list_directed_int, typed_format_error,
)


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


#: The four components of the stress-displacement vector KRAKEL tabulates in
#: an elastic medium, in the order it writes them
#: (``Matlab/ReadWrite/get_component.m:29-41``): horizontal displacement,
#: vertical displacement, tangential stress, normal stress.
_ELASTIC_COMPONENTS = ('H', 'V', 'T', 'N')


def get_component(modes_dict: Dict[str, Any], comp: str) -> np.ndarray:
    """
    Extract one component of the stress-displacement vector from a Kraken
    mode set.

    In an **elastic** medium a mode is a four-component vector at each mesh
    point — horizontal displacement, vertical displacement, tangential
    stress, normal stress — stacked into consecutive rows of ``phi``. In an
    acoustic medium a mode is the single pressure row. This walks the media
    in order and pulls the requested component out of each elastic block
    while copying each acoustic row through, so the result is one row per
    depth whatever the medium stack is.

    Parameters
    ----------
    modes_dict : dict
        A mode set as :func:`read_modes_bin` returns it. Uses:

        - ``'phi'`` : ndarray ``(nrows, nmodes)`` — the stacked rows.
        - ``'z'`` : ndarray — the depth axis.
        - ``'Nmedia'`` : int — how many media; one when the key is absent.
        - ``'Mater'`` : sequence of str, optional — ``'ACOUSTIC'`` or
          ``'ELASTIC'`` per medium. Absent, every medium is taken as
          acoustic, which is what the shipped solvers produce: without
          KRAKEL, ``Mater`` never contains ``'ELASTIC'`` and this reduces to
          a copy of ``phi``.
    comp : str
        One of ``'H'``, ``'V'``, ``'T'``, ``'N'``. Any of the four returns
        the pressure row for an acoustic medium; the choice only selects
        inside an elastic block. The value is validated up front, so a typo
        raises even on an all-acoustic mode set rather than being ignored.

    Returns
    -------
    phi : ndarray
        The extracted component, shape ``(nz, nmodes)``.

    Raises
    ------
    ~uacpy.core.exceptions.ConfigurationError
        ``comp`` is not one of the four components, or ``Mater`` names a
        material that is neither ``'ACOUSTIC'`` nor ``'ELASTIC'``.
    ~uacpy.core.exceptions.FileFormatError
        The mode set holds no modes (``M == 0``), so there is nothing to
        extract.

    Notes
    -----
    KRAKEL tabulates modes on its finite-difference grid; KRAKEN and KRAKENC
    subtabulate to the receiver depths and do **not** tabulate inside an
    elastic medium at all. The walk therefore stops as soon as it runs past
    the last row of ``phi`` (``get_component.m:20-22`` returns there), which
    is how a KRAKEN file with an elastic layer terminates cleanly instead of
    indexing off the end.

    References
    ----------
    Translated from ``Matlab/ReadWrite/get_component.m`` (mbp, 2010).

    Examples
    --------
    >>> import numpy as np
    >>> modes = {'phi': np.arange(12.0).reshape(6, 2), 'z': np.zeros(6),
    ...          'Nmedia': 1, 'Mater': ['ACOUSTIC']}
    >>> get_component(modes, 'N').shape
    (6, 2)

    An elastic medium stacks four rows per depth; ``'V'`` takes the second
    of each group:

    >>> modes = {'phi': np.arange(8.0).reshape(8, 1), 'z': np.zeros(2),
    ...          'Nmedia': 1, 'Mater': ['ELASTIC']}
    >>> get_component(modes, 'V').ravel()
    array([1., 5.])
    """
    if comp not in _ELASTIC_COMPONENTS:
        raise ConfigurationError(
            f"get_component(comp={comp!r}) is not a component of the "
            f"stress-displacement vector.",
            remediation="Use 'H' (horizontal displacement), 'V' (vertical "
                        "displacement), 'T' (tangential stress) or 'N' "
                        "(normal stress).",
        )

    phi_full = np.asarray(modes_dict["phi"])
    z = modes_dict["z"]
    n_media = int(modes_dict.get("Nmedia", 1))
    # A flat list of per-medium material names. The fallback is all-acoustic
    # rather than a nested placeholder: with the shipped solvers (no KRAKEL)
    # ``Mater`` never contains 'ELASTIC', so the acoustic path is the one a
    # caller without the key means.
    mater = list(modes_dict.get("Mater") or [])

    rows: List[np.ndarray] = []
    jj = 0    # depth index across all media
    k = 0     # row index into phi

    for medium in range(n_media):
        for _ in range(len(z)):
            # KRAKEN / KRAKENC do not tabulate inside an elastic medium, so
            # phi runs out before the media do (get_component.m:20-22).
            if k >= phi_full.shape[0]:
                break

            material = (str(mater[medium]).strip().upper()
                        if medium < len(mater) else "ACOUSTIC")

            if material == "ACOUSTIC":
                if jj < len(z):
                    rows.append(phi_full[k, :])
                k += 1
            elif material == "ELASTIC":
                if jj < len(z):
                    rows.append(phi_full[k + _ELASTIC_COMPONENTS.index(comp), :])
                k += 4
            else:
                raise ConfigurationError(
                    f"get_component: medium {medium + 1} has material "
                    f"{material!r}; a Kraken mode file describes each medium "
                    f"as 'ACOUSTIC' or 'ELASTIC' "
                    f"(Kraken/kraken.f90 writes the 8-character name).",
                    remediation="Check the mode file was read by "
                                "read_modes_bin; a hand-built dict must use "
                                "one of those two names.",
                )
            jj += 1

    if not rows:
        raise FileFormatError(
            "get_component: the modes set contains no readable modes (M=0) — "
            "nothing to extract. The waveguide is likely below modal cutoff "
            "at this frequency; check the .mod record before requesting a "
            "component."
        )
    return np.array(rows)


@typed_format_error
def read_modes_asc(
    filename: Union[str, Path],
    modes: Optional[Union[int, list, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Read a KRAKEN ASCII mode file (``.moa``) — the text sibling of the
    binary ``.mod`` :func:`read_modes_bin` parses.

    Parameters
    ----------
    filename : str or Path
        Path to the mode file, extension included.
    modes : int, list or ndarray, optional
        Mode indices to keep, **1-indexed** (the Fortran/MATLAB
        convention). An int selects that one mode; ``None`` (default) keeps
        all of them. Indices outside ``1..M`` are dropped rather than
        raising, matching ``read_modes_asc.m:41-43``.

    Returns
    -------
    Modes : dict
        - ``'pltitl'`` / ``'title'`` : str — the title line (both keys, so
          the dict reads like the binary reader's).
        - ``'freq'`` : float — frequency in Hz; ``'freqVec'`` is the same
          value as a one-element array and ``'Nfreq'`` is 1.
        - ``'Nmedia'``, ``'ntot'``, ``'nmat'`` : int — medium count, total
          depth points, total matrix rows.
        - ``'M'`` : int — the number of modes **returned**, i.e.
          ``len(k)``, the same meaning ``read_modes_bin`` gives it.
        - ``'z'`` : ndarray ``(ntot,)`` — depths in metres.
        - ``'k'`` : complex ndarray ``(M,)`` — horizontal wavenumbers in
          rad/m; the imaginary part is the attenuation.
        - ``'phi'`` : complex ndarray ``(ntot, M)`` — mode shapes.

    Raises
    ------
    ~uacpy.core.exceptions.FileFormatError
        The file is absent or malformed, or a declared count is
        non-positive.

    Notes
    -----
    Layout, in the order ``read_modes_asc.m`` scans it: the record length
    (unused in ASCII), the title line, ``freq Nmedia ntot nmat M``, one line
    per medium, the top and bottom halfspace lines, a blank line, the depth
    axis, the wavenumbers, then for each mode a separator line followed by
    the mode shape.

    **Complex values are stored as interleaved ``(Re, Im)`` pairs**, not as
    a real block followed by an imaginary block: the reference reader takes
    them with ``fscanf( fid, '%f', [ 2, N ] )``
    (``read_modes_asc.m:33,50``), which fills a 2-by-N array in column
    order. Both the wavenumber record and every mode-shape record follow
    that, and reading them as two blocks silently returns the first half of
    the file's values as real parts of everything.

    The medium and halfspace records are skipped, as the reference skips
    them (``read_modes_asc.m:26-32``). They are **not** parsed into a
    ``Top``/``Bot`` pair here: no shipped Acoustics-Toolbox program writes
    this format — every ``MODFile`` OPEN is ``ACCESS = 'DIRECT'``
    (``Kraken/kraken.f90:588``, ``Kraken/krakenc.f90:439,631``,
    ``Krakel/krakel.f90:493``) — so their column layout is not established
    by any producer, and inventing one would put fabricated halfspace speeds
    into a caller's modal sum. Read the binary ``.mod`` through
    :func:`read_modes` when the halfspace terms are needed.

    References
    ----------
    Translated from ``Matlab/ReadWrite/read_modes_asc.m``.

    See Also
    --------
    read_modes_bin : The binary ``.mod`` reader.
    get_component : Pull one component out of an elastic mode set.
    """
    filename = Path(filename)
    if not filename.exists():
        raise FileFormatError(
            f"read_modes_asc: mode file not found: {filename}.",
            remediation="Check the path; the ASCII .moa is written by the "
                        "AT Matlab tools, not by the shipped solvers.",
        )

    with open(filename, "r") as fid:
        # Each numeric record below is a token stream that may span lines,
        # mirroring the reference reader's ``fscanf( fid, '%f', N )``: a
        # Fortran runtime may wrap a long record and this must not care.
        def _read_floats(n: int, what: str) -> np.ndarray:
            values: list = []
            while len(values) < n:
                line = fid.readline()
                if line == '':
                    raise FileFormatError(
                        f"read_modes_asc: {filename} ended while reading "
                        f"{what} — expected {n} values, found {len(values)}.",
                        remediation="The file is truncated; verify it was "
                                    "written completely.",
                    )
                values.extend(float(tok) for tok in line.split())
            # A list-directed WRITE ends each record with a newline, so the
            # surplus tokens of the final line belong to this record's
            # padding, not to the next one.
            return np.array(values[:n], dtype=float)

        def _read_complex(n: int, what: str) -> np.ndarray:
            """``fscanf( fid, '%f', [ 2, n ] )``: interleaved (Re, Im)."""
            pairs = _read_floats(2 * n, what).reshape(n, 2)
            return pairs[:, 0] + 1j * pairs[:, 1]

        list_directed_int(fid.readline())          # lrecl, unused in ASCII
        pltitl = fid.readline().strip()
        params = _read_floats(5, "freq Nmedia ntot nmat M")
        freq = float(params[0])
        n_media = int(params[1])
        ntot = int(params[2])
        nmat = int(params[3])
        n_modes_total = int(params[4])

        if ntot <= 0 or n_media <= 0:
            raise FileFormatError(
                f"read_modes_asc: {filename} declares Nmedia={n_media}, "
                f"ntot={ntot}; both must be positive.",
                remediation="Check the 'freq Nmedia ntot nmat M' record — a "
                            "misaligned file reads the wrong line as it.",
            )

        for _ in range(n_media):
            fid.readline()                         # per-medium record
        fid.readline()                             # top halfspace
        fid.readline()                             # bottom halfspace
        fid.readline()                             # blank line

        z = _read_floats(ntot, f"{ntot} depths")
        k_all = _read_complex(n_modes_total,
                              f"{n_modes_total} wavenumbers")

        if modes is None:
            wanted = list(range(1, n_modes_total + 1))
        elif isinstance(modes, (int, np.integer)):
            wanted = [int(modes)]
        else:
            wanted = [int(m) for m in modes]
        wanted = [m for m in wanted if 1 <= m <= n_modes_total]

        k_selected = k_all[[m - 1 for m in wanted]]
        phi = np.zeros((ntot, len(wanted)), dtype=complex)

        for mode_num in range(1, n_modes_total + 1):
            fid.readline()                         # per-mode separator line
            shape = _read_complex(ntot, f"mode {mode_num} shape ({ntot} "
                                        f"depths)")
            if mode_num in wanted:
                phi[:, wanted.index(mode_num)] = shape

    return {
        "pltitl": pltitl,
        "title": pltitl,
        "freq": freq,
        "freqVec": np.asarray([freq], dtype=float),
        "Nfreq": 1,
        "Nmedia": n_media,
        "ntot": ntot,
        "nmat": nmat,
        # len(k), the same meaning read_modes_bin gives M.
        "M": len(k_selected),
        "z": z,
        "k": k_selected,
        "phi": phi,
    }



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
          ``modes`` selected a subset.
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
    - The stored ``phi`` are KRAKEN-normalised: the tabulated-span integral
      ``SUM(phi**2 / rho) dz`` plus the analytic top/bottom halfspace-tail
      terms equals 1 (``Kraken/kraken.f90:795-800``), so the tabulated span
      alone integrates to less than 1 and the shapes must **not** be
      re-normalised over ``z``. See :func:`read_modes` for the halfspace
      extension formula and the sign convention.

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
    Read mode data from a KRAKEN binary ``.mod`` file and attach the
    halfspace parameters the modal-sum evaluators need.

    Parameters
    ----------
    filename : str
        Mode file path; ``.mod`` is appended when no extension is given.
        Any other extension raises :class:`FileFormatError` — the binary
        direct-access ``.mod`` is the only mode format any
        Acoustics-Toolbox program writes.
    frequency : float, optional
        Frequency in Hz to select from multi-frequency files (default: 0)
    modes : int, list, or ndarray, optional
        Mode indices to extract (1-indexed). If None, all modes are returned.
    profile : int, optional
        Profile index to read (1-indexed, default 1).

    Returns
    -------
    modes_data : dict
        Mode data dictionary with fields from :func:`read_modes_bin`,
        plus computed halfspace parameters:
        - 'Top': dict with top halfspace properties (k2, gamma, phi)
        - 'Bot': dict with bottom halfspace properties (k2, gamma, phi)

    Notes
    -----
    ``Modes['M']`` is the number of modes returned (``len(Modes['k'])``);
    the halfspace parameters below are computed only when it is non-zero.

    For acoustic halfspaces (boundary condition 'A'), computes:
    - k²: wavenumber squared in halfspace
    - γ: vertical wavenumber using Pekeris root, from the full complex
      eigenvalue for a KRAKENC file and from ``Re(k)`` otherwise
    - φ: mode value at interface

    **Normalisation — do not re-normalise the mode shapes.** KRAKEN scales
    each mode so that the *full* norm equals 1: the discrete
    ``SUM(phi**2 / rho) dz`` over the tabulated span **plus** the analytic
    contribution of the top and bottom halfspace tails, carried by the
    admittance derivatives in ``RN = SqNorm - DrhoDx * Phi(1)**2 +
    DetaDx * Phi(NTotal1)**2`` (``Kraken/kraken.f90:795-800``). The
    tabulated span alone therefore integrates to *less* than 1 — the
    deficit grows with mode number as more energy sits in the evanescent
    tail (about 0.925 by mode 5 for a Pekeris case) — so re-normalising
    ``phi`` over ``z`` breaks the eigenfunction scaling.

    **Halfspace extension and sign convention.** Below the deepest
    tabulated depth ``D`` an 'A'-halfspace mode continues analytically as
    ``phi(z) = Bot['phi'] * exp(-Bot['gamma'] * (z - D))``, with
    ``gamma**2 = k**2 - Bot['k2']`` and the root chosen with
    ``Re(gamma) >= 0`` (decay into the halfspace; ``pekeris_root``, the
    branch ``KrakenField/ReadModes.f90:254-272`` uses). The overall sign
    of each mode follows KRAKEN's convention that ``phi`` is positive at
    the mode's turning point (``Kraken/kraken.f90:808-809`` flips the
    scale factor to enforce it). The mirrored ``Top`` entries extend
    above the shallowest tabulated depth the same way.

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
    """
    fileroot, ext = os.path.splitext(filename)

    if not ext:
        ext = ".mod"  # Default extension

    filename = fileroot + ext
    if ext != ".mod":
        raise FileFormatError(
            f"read_modes: {filename} is not a binary .mod mode file; the "
            f"binary direct-access .mod is the only mode format any "
            f"Acoustics-Toolbox program writes, and it is the only one this "
            f"dispatcher can attach halfspace terms to.",
            remediation="Read the solver's .mod output, or pass the root "
                        "name and let '.mod' be appended. An ASCII '.moa' "
                        "written by the AT Matlab tools is read by "
                        "read_modes_asc directly (it carries no halfspace "
                        "record this function could use).",
        )
    Modes = read_modes_bin(filename, frequency, modes, profile=profile)
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
