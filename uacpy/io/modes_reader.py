"""
Reader for Kraken normal-mode files (binary ``.mod``, ASCII ``.moa``).

:func:`read_modes` is the one public reader: it takes either format and
returns the :class:`~uacpy.core.results.Modes` carrier ``Kraken`` itself
returns. The file-layout parsers behind it (``_read_modes_payload`` for the
binary direct-access ``.mod`` — the only mode format any Acoustics-Toolbox
program writes — and ``_read_modes_asc_payload`` for an ASCII ``.moa`` from
the AT Matlab tools) return the raw OALIB-shaped dicts, and
``_get_component`` pulls one stress-displacement component out of an elastic
block, which ``read_modes(component=)`` exposes.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from uacpy.core.acoustics import pekeris_root
from uacpy.core.results import MediaTable, Modes
from uacpy.core.exceptions import (
    ConfigurationError, FileFormatError,
)
from uacpy.io._fortran_helpers import (
    DirectAccessFile, list_directed_int, read_list_directed_values,
    require_model_output, typed_format_error,
)


def _frequency_index(freq_vec, frequency, filename) -> int:
    """Index of the ``freqVec`` entry ``frequency`` names (the closest one).

    ``None`` names the only frequency of a single-frequency file; on a
    multi-frequency file it raises rather than silently take the first.
    """
    freq_vec = np.atleast_1d(np.asarray(freq_vec, dtype=float))
    if frequency is None:
        if freq_vec.size > 1:
            raise ConfigurationError(
                f"{filename}: the mode file holds {freq_vec.size} "
                f"frequencies ({', '.join(f'{f:g}' for f in freq_vec)} Hz); "
                f"say which one to read.",
                remediation="Pass frequency= (the closest entry is taken).")
        return 0
    return int(np.argmin(np.abs(freq_vec - float(frequency))))


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


def _get_component(modes_dict: Dict[str, Any], comp: str) -> np.ndarray:
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
        A mode set as :func:`_read_modes_payload` returns it. Uses:

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
    >>> _get_component(modes, 'N').shape
    (6, 2)

    An elastic medium stacks four rows per depth; ``'V'`` takes the second
    of each group:

    >>> modes = {'phi': np.arange(8.0).reshape(8, 1), 'z': np.zeros(2),
    ...          'Nmedia': 1, 'Mater': ['ELASTIC']}
    >>> _get_component(modes, 'V').ravel()
    array([1., 5.])
    """
    if comp not in _ELASTIC_COMPONENTS:
        raise ConfigurationError(
            f"read_modes(component={comp!r}) is not a component of the "
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
                    f"read_modes: medium {medium + 1} has material "
                    f"{material!r}; a Kraken mode file describes each medium "
                    f"as 'ACOUSTIC' or 'ELASTIC' "
                    f"(Kraken/kraken.f90 writes the 8-character name).",
                    remediation="Check the mode file was read by "
                                "_read_modes_payload; a hand-built dict must use "
                                "one of those two names.",
                )
            jj += 1

    if not rows:
        raise FileFormatError(
            "_get_component: the modes set contains no readable modes (M=0) — "
            "nothing to extract. The waveguide is likely below modal cutoff "
            "at this frequency; check the .mod record before requesting a "
            "component."
        )
    return np.array(rows)


@typed_format_error
def _read_modes_asc_payload(
    filename: Union[str, Path],
    modes: Optional[Union[int, list, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Read a KRAKEN ASCII mode file (``.moa``) — the text sibling of the
    binary ``.mod`` :func:`_read_modes_payload` parses.

    Parameters
    ----------
    filename : str or Path
        Path to the mode file, extension included.
    modes : int, list or ndarray, optional
        Mode indices to keep, **1-indexed** (the Fortran/MATLAB
        convention). An int selects that one mode; ``None`` (default) keeps
        all of them. Indices outside ``1..M`` are dropped rather than
        raising, matching ``read_modes_asc.m:44-46``.

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
          ``len(k)``, the same meaning ``_read_modes_payload`` gives it.
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
    _read_modes_payload : The binary ``.mod`` reader.
    _get_component : Pull one component out of an elastic mode set.
    """
    filename = Path(filename)
    if not filename.exists():
        raise FileFormatError(
            f"read_modes: mode file not found: {filename}.",
            remediation="Check the path; the ASCII .moa is written by the "
                        "AT Matlab tools, not by the shipped solvers.",
        )

    with open(filename, "r") as fid:
        # Each numeric record below is a token stream that may span lines,
        # mirroring the reference reader's ``fscanf( fid, '%f', N )``: a
        # Fortran runtime may wrap a long record and this must not care.
        # The shared list-directed reader: repeat counts (``2*0.0``),
        # E-less three-digit exponents and ``D`` exponents all parse, and
        # the surplus tokens of the final line are the record's padding.
        def _read_floats(n: int, what: str) -> np.ndarray:
            return np.asarray(
                read_list_directed_values(fid, n, what, filename),
                dtype=float)

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
                f"read_modes: {filename} declares Nmedia={n_media}, "
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
        # len(k), the same meaning _read_modes_payload gives M.
        "M": len(k_selected),
        "z": z,
        "k": k_selected,
        "phi": phi,
    }


class _ModFile(DirectAccessFile):
    """A KRAKEN ``.mod`` file open for reading.

    Per profile (``KrakenField/ReadModes.f90:19-25``): five header records,
    then one block per frequency — a mode-count record, a halfspace record,
    ``M`` eigenfunction records and the eigenvalues folded across
    ``1 + (2M-1)/LRecordLength`` records (``Kraken/kraken.f90:106-117``).
    ``LRecordLength`` is :attr:`record_words`, a count of 4-byte words.
    """

    def __init__(self, fid, filename: str):
        head = fid.read(4)
        fid.seek(0)
        if len(head) < 4:
            raise FileFormatError(
                f"Invalid mode file (truncated header): {filename}.")
        super().__init__(
            fid, source=f'_read_modes_payload:{os.path.basename(filename)}')
        self.filename = filename
        # LRecordLength is a count of 4-byte `longwords' and each eigenvalue
        # record holds LRecordLength/2 complex values (kraken.f90:587,110), so
        # anything below 2 cannot carry a single eigenvalue.
        if self.record_words < 2:
            raise FileFormatError(
                f"Invalid mode file: record length LRecordLength="
                f"{self.record_words} words (must be a positive word-count "
                f"of at least 2): {filename}."
            )

    def n_wavenumber_records(self, M: int) -> int:
        """Eigenvalue records for ``M`` modes (kraken.f90:109)."""
        return 1 + _fortran_div(2 * M - 1, self.record_words)

    def next_block(self, rec: int, M: int) -> int:
        """The mode-count record of the frequency block after the one at
        ``rec`` holding ``M`` modes (kraken.f90:117)."""
        return rec + 3 + M + _fortran_div(2 * M - 1, self.record_words)

    def block_end(self, rec: int, M: int) -> int:
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
        return rec + 2 + M + self.n_wavenumber_records(M)

    def profile_header(self, hdr: int) -> Dict[str, Any]:
        """The five descriptive records of the profile at record ``hdr``
        (kraken.f90:593-599, ReadModes.f90:20)."""
        lrecl, file_size = self.record_bytes, self.file_size
        if (hdr + 5) * lrecl > file_size:
            raise FileFormatError(
                f"Invalid mode file {self.filename}: profile header at record "
                f"{hdr} needs {(hdr + 5) * lrecl} bytes but the file is "
                f"{file_size} bytes."
            )
        self.seek(hdr, offset=4)       # past this profile's LRecordLength
        title = self.text(80).strip()
        Nfreq, Nmedia, Ntot, NMat = (
            int(v) for v in self.values(self.i4, 4)
        )

        # File-size-aware sanity bound on the header counts before any
        # array is sized off them. A corrupt/hostile header (e.g.
        # NMat=0x7fffffff) would otherwise drive a multi-GB np.zeros below.
        # The smallest a single sample can occupy on disk is 4 bytes
        # (float32 / int32), so no count of 4-byte items can exceed the
        # remaining file size; use that as a generous upper bound.
        max_items = file_size // 4
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
                f"frequency block): {self.filename}."
            )

        # Records hdr+1..hdr+4 (kraken.f90:594-598, read back at
        # ReadModes.f90:185-188). The first two are implied-DO pair lists
        # over the media, so the two quantities interleave: int32 N with
        # CHARACTER*8 Material, then REAL*4 depth with REAL*4 rho — hence
        # the (2, Nmedia) Fortran-order reshape. freqVec is REAL(KIND=8)
        # (SourceReceiverPositions.f90:14) while zTab is a default REAL
        # and depth/rho are written through REAL(), so those are f4.
        self.seek(hdr + 1)
        N = []
        Mater = []
        for _ in range(Nmedia):
            N.append(self.integer())
            Mater.append(self.text(8).strip())
        bulk = self.vector(hdr + 2, self.f4, 2 * Nmedia).reshape(
            (2, Nmedia), order="F"
        )
        freqVec = self.vector(hdr + 3, self.f8, Nfreq)
        z = self.vector(hdr + 4, self.f4, Ntot)
        return {
            "title": title, "Nfreq": Nfreq, "Nmedia": Nmedia,
            "Ntot": Ntot, "NMat": NMat, "N": N, "Mater": Mater,
            "depth": bulk[0, :], "rho": bulk[1, :],
            "freqVec": freqVec, "z": z,
        }

    def mode_count(self, rec: int) -> int:
        """Read ``M`` from record ``rec`` and bound it against the file.

        ``M`` is a plain header word (kraken.f90:106) that sizes every
        allocation below, so it gets the same file-size bound as the
        record-0 counts.
        """
        lrecl, file_size = self.record_bytes, self.file_size
        if (rec + 1) * lrecl > file_size:
            raise FileFormatError(
                f"Invalid mode file {self.filename}: mode-count record {rec} "
                f"starts past the end of a {file_size}-byte file."
            )
        self.seek(rec)
        M = self.integer()
        if M < 0 or self.block_end(rec, M) * lrecl > file_size:
            raise FileFormatError(
                f"Invalid mode file {self.filename}: mode count M={M} at "
                f"record {rec} needs {self.block_end(rec, M) * lrecl} bytes "
                f"but the file is {file_size} bytes."
            )
        return M

    def halfspace(self) -> Dict[str, Any]:
        """One halfspace description from the current position: the
        boundary-condition letter, ``cp``, ``cs``, ``rho`` and ``depth``
        (kraken.f90:603, read_modes_bin.m:132-141)."""
        side = {"BC": chr(self.values(np.uint8, 1)[0])}
        cp_data = self.values(self.f4, 2)
        side["cp"] = complex(cp_data[0], cp_data[1])
        cs_data = self.values(self.f4, 2)
        side["cs"] = complex(cs_data[0], cs_data[1])
        side["rho"] = self.values(self.f4, 1)[0]
        side["depth"] = self.values(self.f4, 1)[0]
        return side


@typed_format_error
def _read_modes_payload(
    filename: Union[str, Path],
    frequency: Optional[float] = None,
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
    frequency : float, optional
        Frequency in Hz for which to read modes; the closest entry of the
        file's ``freqVec`` is selected. ``None`` (default) reads the only
        frequency of a single-frequency file and raises
        :class:`~uacpy.core.exceptions.ConfigurationError` naming the
        frequencies of a multi-frequency one, rather than pick one.
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
        - 'freq' : float - The ``freqVec`` entry these modes belong to
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
    - An absent file (after that resolution) and a malformed or truncated
      one both raise a typed :class:`FileFormatError`
      (``require_model_output`` / ``typed_format_error``) rather than a
      bare ``FileNotFoundError`` / ``IndexError`` / ``struct.error``.
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
    >>> modes = _read_modes_payload('pekeris', frequency=100.0)
    >>> print(f"Number of modes: {modes['M']}")
    >>> print(f"Wavenumber of mode 1: {modes['k'][0]}")

    >>> # Read specific modes
    >>> modes = _read_modes_payload('pekeris', frequency=100.0, modes=[1, 2, 3])
    >>> print(f"Mode shapes: {modes['phi'].shape}")
    """
    if profile < 1:
        raise ConfigurationError(
            f"read_modes: profile must be >= 1 (got {profile}); mode-file "
            "profiles are numbered from 1 (Kraken/kraken.f90:42)."
        )
    filename = os.fspath(filename)
    if not os.path.splitext(filename)[1]:
        filename = filename + ".mod"
    require_model_output(filename, 'read_modes')

    with open(filename, "rb") as fid:
        mod = _ModFile(fid, filename)

        # Walk the preceding profiles: each is a five-record header followed
        # by one block per frequency (ReadModes.f90:19-25,125).
        hdr = 0
        for _ in range(profile - 1):
            rec = hdr + 5
            for _ in range(mod.profile_header(hdr)["Nfreq"]):
                rec = mod.next_block(rec, mod.mode_count(rec))
            hdr = rec

        header = mod.profile_header(hdr)
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

        freq_index = _frequency_index(freqVec, frequency, filename)
        # Records hdr+0..hdr+3: header, N/Mater, depth/rho, freqVec
        # Record hdr+4: z vector
        # Record hdr+5: M (mode count) — where the first frequency block starts
        iRecProfile = hdr + 5
        for ifreq in range(freq_index + 1):
            M = mod.mode_count(iRecProfile)
            if ifreq < freq_index:
                iRecProfile = mod.next_block(iRecProfile, M)
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
        mod.seek(iRecProfile + 1)
        Top = mod.halfspace()
        Bot = mod.halfspace()
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
                phi_data = mod.vector(
                    iRecProfile + 1 + int(mode_idx), mod.f4, 2 * NMat
                ).reshape((2, NMat), order="F")
                phi[:, ii] = phi_data[0, :] + 1j * phi_data[1, :]
            # The eigenvalues are folded across records: kraken.f90:108-113
            # writes LRecordLength/2 complex values per record, each starting
            # on a record boundary, and KrakenField/ReadModes.f90:212-219
            # reads them back the same way. An odd LRecordLength leaves one
            # unwritten word at the end of every eigenvalue record, which a
            # single contiguous read would absorb as data. ``irec`` counts
            # from 0 where the Fortran IREC counts from 1, hence ``+ 2`` here
            # against kraken.f90:111's ``+ 1``.
            lrecl_words = mod.record_words
            k_all = np.zeros(M, dtype=np.complex64)
            per_record = lrecl_words // 2
            ifirst = 0
            for irec in range(mod.n_wavenumber_records(M)):
                ilast = min(M, ifirst + per_record)
                vals = mod.vector(iRecProfile + 2 + M + irec, mod.f4,
                                  2 * (ilast - ifirst))
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
        "freq": float(freqVec[freq_index]),   # the frequency read
        "z": z,
        "M": int(len(k)),   # modes returned, not the file's total
        "phi": phi,
        "k": k,
        "Top": Top,
        "Bot": Bot,
    }


def _read_modes_with_halfspace(
    filename: Union[str, Path],
    frequency: Optional[float] = None,
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
        Frequency in Hz to select from a multi-frequency file (the closest
        ``freqVec`` entry). ``None`` (default) reads a single-frequency file
        and raises :class:`~uacpy.core.exceptions.ConfigurationError`
        naming the frequencies of a multi-frequency one.
    modes : int, list, or ndarray, optional
        Mode indices to extract (1-indexed). If None, all modes are returned.
    profile : int, optional
        Profile index to read (1-indexed, default 1).

    Returns
    -------
    modes_data : dict
        Mode data dictionary with fields from :func:`_read_modes_payload`,
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
    >>> payload = _read_modes_with_halfspace('test.mod', frequency=100.0)
    >>> print(payload['Bot']['gamma'])
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
                        "read_modes through its own parser (it carries no "
                        "halfspace record).",
        )
    Modes = _read_modes_payload(filename, frequency, modes, profile=profile)
    freq_index = _frequency_index(Modes["freqVec"], frequency, filename)
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



def _modes_from_payload(
    payload: Dict[str, Any],
    *,
    phi: Optional[np.ndarray] = None,
    **result_kwargs,
) -> Modes:
    """The :class:`~uacpy.core.results.Modes` carrier of a mode-file payload.

    ``k``, ``phi`` (KRAKEN's normalisation kept) and the depth axis ``z``
    are taken as they are; ``phi`` overrides the payload's rows (a component
    pulled out of an elastic block). ``backend`` defaults to the solver the
    title names, ``frequencies`` to the payload's ``'freq'``.
    """
    title = str(payload.get('title', ''))
    result_kwargs.setdefault(
        'backend', 'krakenc' if title[:7].upper() == 'KRAKENC' else 'kraken')
    if 'freq' in payload:
        result_kwargs.setdefault('frequencies', float(payload['freq']))
    return Modes(
        k=payload.get('k', np.array([])),
        phi=payload.get('phi', np.array([])) if phi is None else phi,
        depths=payload.get('z', np.array([])),
        **result_kwargs,
    )


def read_modes(
    filepath: Union[str, Path],
    frequency: Optional[float] = None,
    modes: Optional[Union[int, list, np.ndarray]] = None,
    profile: int = 1,
    *,
    component: Optional[str] = None,
    water_density: Optional[float] = None,
) -> Modes:
    """Read a KRAKEN mode file as a :class:`~uacpy.core.results.Modes` result.

    The carrier ``Kraken`` returns — plotting,
    :meth:`~uacpy.core.results.Modes.first_n`, the modal sums — from a
    ``.mod`` of any run, or an ASCII ``.moa``. ``model`` is empty (the file,
    not a model run, is the source) and ``backend`` names the solver the
    file's title records.

    Parameters
    ----------
    filepath : str or Path
        A binary ``.mod`` (the extension is appended to a bare root) or an
        ASCII ``.moa``. Any other extension raises
        :class:`~uacpy.core.exceptions.FileFormatError`.
    frequency : float, optional
        Hz; the closest entry of a multi-frequency ``.mod`` is read. ``None``
        (default) reads a single-frequency file and raises
        :class:`~uacpy.core.exceptions.ConfigurationError` naming the
        frequencies of a multi-frequency one.
    modes : int, list or ndarray, optional
        Mode numbers to keep, **1-based** as in the file and in
        ``read_modes_bin.m``; out-of-range numbers are dropped. ``None``
        keeps all.
    profile : int, optional
        Profile of a multi-profile ``.mod``, 1-based (``Kraken/kraken.f90:42``).
    component : {'H', 'V', 'T', 'N'}, optional
        The stress-displacement component to keep inside an **elastic**
        medium (horizontal / vertical displacement, tangential / normal
        stress, ``Matlab/ReadWrite/get_component.m``); acoustic rows pass
        through as pressure. Needed for a file with elastic media, whose
        rows otherwise stack four components per depth.
    water_density : float, optional
        g/cm³ the modes were normalised in, stored as
        ``media.water_density`` for
        :meth:`~uacpy.core.results.Modes.modal_pressure_field`. ``None``
        stores the density a ``.mod`` records for its first medium, the
        water column (``kraken.f90:595-596``); a ``.moa`` records none.

    Returns
    -------
    Modes
        ``k`` (rad/m, complex), ``phi`` ``(n_depths, n_modes)``, ``depths``
        (m), ``frequencies`` = the entry read. ``media`` is the
        :class:`~uacpy.core.results.MediaTable` of a ``.mod``: the top
        depth (m) and density (g/cm³) of every medium, the base of the last
        medium, the density below it and the water density (``None`` for a
        ``.moa`` read with no ``water_density``). ``metadata`` carries the
        file's ``title``, the ``.mod``'s ``top_halfspace`` /
        ``bottom_halfspace`` records (boundary code, ``cp``/``cs``/``rho``,
        and the ``k2``, ``gamma``, ``phi`` interface terms) and
        ``media_types``.
    """
    root, ext = os.path.splitext(os.fspath(filepath))
    metadata: Dict[str, Any] = {}
    media = None
    if ext == '.moa':
        payload = _read_modes_asc_payload(filepath, modes=modes)
    else:
        payload = _read_modes_with_halfspace(filepath, frequency, modes,
                                             profile=profile)
        metadata['top_halfspace'] = payload['Top']
        metadata['bottom_halfspace'] = payload['Bot']
        metadata['media_types'] = list(payload.get('Mater', []))
        # The medium table the modes were normalised in (kraken.f90:595-596:
        # the density at the top of every medium the file tabulates, the
        # water column first), from which Modes.modal_pressure_field takes
        # rho(z_s). The .mod carries the water density its modes were
        # normalised against; a .moa carries none, and keeps the package
        # default unless one is given.
        media = MediaTable(
            water_density=(float(payload['rho'][0]) if water_density is None
                           else float(water_density)),
            tops=[float(d) for d in payload['depth']],
            densities=[float(r) for r in payload['rho']],
            bottom_depth=float(payload['Bot']['depth']),
            halfspace_density=float(payload['Bot']['rho']))
    metadata['title'] = str(payload.get('title', payload.get('pltitl', '')))
    if media is None and water_density is not None:
        media = MediaTable(water_density=float(water_density))
    elastic = any(str(m).strip().upper() == 'ELASTIC'
                  for m in payload.get('Mater', []))
    if component is not None:
        phi = _get_component(payload, component)
    elif elastic:
        raise ConfigurationError(
            f"read_modes: {filepath} has elastic media, whose mode rows "
            f"stack four stress-displacement components per depth; say "
            f"which one to keep.",
            remediation="Pass component='H', 'V', 'T' or 'N'.")
    else:
        phi = None
    return _modes_from_payload(payload, phi=phi, metadata=metadata,
                               media=media)
