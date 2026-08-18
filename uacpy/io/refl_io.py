"""Reflection-coefficient + source-beam-pattern auxiliary file I/O.

* ``.brc`` / ``.trc`` — precomputed plane-wave reflection coefficients as
  ``(angle_deg, |R|, phase_deg)`` triples behind a count line
  (:func:`read_reflection_coefficient`)

``.irc`` is a different format and is **not** handled here: BOUNCE writes it
as a title/frequency line, a count, then fixed-format ``(5G15.7, I5)`` rows of
``x, Re f, Im f, Re g, Im g, iPower`` (``Kraken/bounce.f90:225-228``), read
back with the same fixed format by ``misc/RefCoef.f90:97-107``. It is consumed
only by the Kraken-family ``BotOpt='P'`` path, which passes it straight to the
Fortran.
* ``.sbp`` — source beam pattern (:func:`read_source_beam_pattern`,
  :func:`write_source_beam_pattern`)

Bathymetry / altimetry / 3-D boundary files live in
:mod:`uacpy.io.bathy_io`.
"""

import shutil

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union

from uacpy._log import log_message
from uacpy.io.units import deg_to_rad
from uacpy.core.exceptions import ConfigurationError, FileFormatError
from uacpy.io._fortran_helpers import _bound_counts
from uacpy.io._fortran_helpers import (
    fortran_float, list_directed_int, read_list_directed_values,
    typed_format_error,
)


@typed_format_error
def read_reflection_coefficient(
    filename: Union[str, Path]
) -> Dict[str, np.ndarray]:
    """
    Read reflection coefficient data from file (.trc or .brc).

    Reads angle-dependent reflection coefficient data used by BELLHOP
    for top or bottom boundary conditions. Data includes angle, magnitude,
    and phase.

    Parameters
    ----------
    filename : str or Path
        Path to a reflection coefficient file. An existing path is read as
        given whatever its suffix (.brc bottom / .trc top — both carry the
        same table layout); a path that does not exist has ``.brc``
        appended, so a bare base name resolves to the conventional bottom
        table.

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
        Then: 3·N values ``angle(deg) magnitude phase(deg)`` per point.
        The engine reads the whole table with a **single** list-directed
        READ (``misc/RefCoef.f90:53``), so the triples may be split or
        packed across lines any way at all; this parser accepts the same.

    References
    ----------
    Based on BELLHOP/readrc.m
    """
    # Use the path as given when it already exists (an explicit path is never
    # shadowed by a sibling file); a bare root without an extension resolves
    # to the conventional .brc bottom table. A missing path that carries an
    # extension is left untouched so the error names the file that was asked
    # for.
    filename = Path(filename)
    if not filename.exists() and not filename.suffix:
        filename = filename.with_name(filename.name + ".brc")

    try:
        with open(filename, "r") as fid:
            # The count is a list-directed scalar READ (RefCoef.f90:45), so
            # a comma or trailing annotation after the number is a valid
            # record.
            n_pts = list_directed_int(fid.readline())

            if n_pts == 0:
                return {
                    "theta": np.array([]),
                    "R": np.array([]),
                    "phi": np.array([]),
                    "n_pts": 0,
                }

            # Each point is three numbers of at least 2 bytes each (value +
            # separator) on disk; a count beyond that is a malformed header,
            # not a huge table.
            _bound_counts(filename, Path(filename).stat().st_size, 6,
                          n_pts=n_pts)
            # RefCoef.f90:53 reads the whole table with ONE list-directed
            # READ of n_pts (theta, R, phi) records, so the 3*n_pts values
            # can be packed or wrapped across lines arbitrarily.
            values = read_list_directed_values(
                fid, 3 * n_pts, f"{n_pts} (theta, R, phi) records", filename)
            table = values.reshape(n_pts, 3)
            theta = table[:, 0]
            R = table[:, 1]
            phi = deg_to_rad(table[:, 2])

            # Validate angles are non-decreasing
            if not np.all(np.diff(theta) >= 0):
                raise FileFormatError(
                    f"Reflection coefficient file {filename}: angles must be "
                    f"non-decreasing (got a decreasing step in the theta "
                    f"column). The file on disk is malformed."
                )

            return {"theta": theta, "R": R, "phi": phi, "n_pts": n_pts}

    except FileNotFoundError as e:
        raise FileFormatError(
            f"Reflection coefficient file not found: {filename}. "
            "Run Bounce or OASR first to generate the .brc/.trc file, "
            "or pass an explicit reflection_file= path to the model."
        ) from e


@typed_format_error
def read_source_beam_pattern(
    filepath: Union[str, Path],
    verbose: bool = False,
) -> np.ndarray:
    """
    Read source beam pattern from file.

    Parameters
    ----------
    filepath : str or Path
        Source beam pattern file; a root name without the ``.sbp`` extension
        also resolves.

    Returns
    -------
    beam_pattern : ndarray
        Beam pattern array, shape (N, 2):
        - Column 0: Angles in degrees
        - Column 1: level in dB re peak

    Raises
    ------
    FileFormatError
        Missing or malformed ``.sbp`` file.

    Notes
    -----
    File format (.sbp):
    - Line 1: Number of points
    - Subsequent lines: angle (degrees), level (dB)

    Levels stay in **dB**, the unit both the file and
    :attr:`uacpy.Source.beam_pattern` use, so this round-trips
    :func:`write_source_beam_pattern`. The engines convert to a linear
    amplitude factor (``10**(dB/20)``, ``beampattern.f90:59``) themselves
    after reading.

    Translated from OALIB readpat.m
    """
    log_message('refl_io', "Reading source beam pattern file",
                verbose=verbose)

    # Take the path as given when it already exists; a bare root without an
    # extension resolves to the conventional .sbp name (mirrors
    # read_reflection_coefficient). A missing path with an extension is left
    # untouched so the error names the file that was asked for.
    sbp_file = Path(filepath)
    if not sbp_file.exists() and not sbp_file.suffix:
        sbp_file = sbp_file.with_name(sbp_file.name + ".sbp")
    if not sbp_file.exists():
        raise FileFormatError(
            f"Source beam pattern file not found: {sbp_file}.",
            remediation="Provide the .sbp file next to the env, or pass "
                        "beam_pattern=None for an omni-directional source.",
        )
    with open(sbp_file, "r") as fid:
        # The count is a list-directed scalar READ (beampattern.f90:33), so
        # a comma or trailing annotation after the number is a valid record.
        NSBPPts = list_directed_int(fid.readline())
        log_message('refl_io',
                    f"Number of source beam pattern points = {NSBPPts}",
                    verbose=verbose)

        # Each point is two numbers of at least 2 bytes each (value +
        # separator) on disk; a larger count is a malformed header rather
        # than a huge table.
        _bound_counts(sbp_file, Path(sbp_file).stat().st_size, 4,
                      NSBPPts=NSBPPts)
        beam_pattern = np.zeros((NSBPPts, 2))
        # Each point is ONE list-directed READ of its (angle, level) pair
        # (beampattern.f90:44), which continues across records — including
        # blank lines — until both values are read, so a pair may wrap and
        # this reader accepts the same files the binary does.
        for i in range(NSBPPts):
            beam_pattern[i, :] = read_list_directed_values(
                fid, 2, f"beam-pattern point {i + 1} (angle, level)",
                sbp_file)
        log_message(
            'refl_io',
            f"angle (deg): {beam_pattern[:, 0].tolist()}",
            verbose=verbose, level='debug',
        )
        log_message(
            'refl_io',
            f"level (dB): {beam_pattern[:, 1].tolist()}",
            verbose=verbose, level='debug',
        )

    return beam_pattern


def write_source_beam_pattern(
    filepath: Union[str, Path], angles: np.ndarray, pattern: np.ndarray
) -> None:
    """
    Write source beam pattern file for Bellhop.

    Parameters
    ----------
    filepath : str or Path
        Output file path (typically .sbp extension)
    angles : ndarray
        Beam angles in degrees, shape (N,)
        Typically from -90 to +90 degrees
    pattern : ndarray
        Beam pattern level in dB relative to peak, shape (N,)
        (typically 0 dB at peak, negative elsewhere).  Bellhop
        converts dB -> linear via 10**(SrcBmPat(:,2)/20) at load time
        (beampattern.f90:59, bellhop.f90:131).

    Notes
    -----
    File format (.sbp):
    - Line 1: Number of angles
    - Following lines: angle (degrees), level (dB re peak)

    Used to specify directional source characteristics.

    The angle column is written at ``%.6f``: Bellhop's load-time guard
    rejects a pattern whose angles are not strictly increasing
    (``misc/beampattern.f90:56``), so two distinct angles that collide on
    the file's angle grid abort the run with ERROUT. Angles closer than
    the 1e-6-degree column resolution are rejected here for the same
    reason.
    """
    filepath = Path(filepath)

    angles = np.asarray(angles, dtype=float)
    steps = np.diff(angles)
    if angles.size > 1 and steps.min() < 1e-6:
        i = int(np.argmin(steps))
        raise ConfigurationError(
            f"write_source_beam_pattern: angles[{i}]={angles[i]!r} and "
            f"angles[{i + 1}]={angles[i + 1]!r} are closer than the file's "
            f"1e-6-degree angle resolution (or not increasing). Bellhop "
            f"requires strictly increasing beam-pattern angles "
            f"(misc/beampattern.f90:56) and aborts on a repeated value.",
            remediation="Pass strictly increasing angles with steps of at "
                        "least 1e-6 degrees.",
        )

    n_angles = len(angles)

    with open(filepath, "w") as f:
        # Write number of angles
        f.write(f"{n_angles}\n")

        # Write angle, amplitude pairs
        for i in range(n_angles):
            f.write(f"{angles[i]:12.6f} {pattern[i]:12.6f}\n")


_REFLECTION_SUFFIX = {'bottom': '.brc', 'top': '.trc', 'internal': '.irc'}


def stage_reflection_file(
    reflection_file: Optional[Union[str, Path]],
    env_path: Union[str, Path],
    boundary: str = 'bottom',
    verbose: bool = False,
) -> Path:
    """Place a reflection-coefficient table beside the ``.env`` that names it.

    AT and Bellhop open the table by base-name convention (``<env>.brc`` for
    the bottom, ``<env>.trc`` for the top, ``<env>.irc`` for a precalculated
    internal table), so a table produced elsewhere has to be copied next to
    the environment file. A table already at the destination — a BOUNCE run
    whose ``.brc`` sits in the same pinned ``work_dir`` the ``.env`` is
    written into — is copied over itself, i.e. left in place.

    The staged ``.brc`` / ``.trc`` then goes through
    :func:`dedupe_reflection_file`, so the table the engine reads satisfies
    ``InterpolateReflectionCoefficient``'s preconditions whatever produced it.
    Doing it here rather than at the producer is what makes a user-supplied
    ``reflection_file=`` equivalent to a :class:`~uacpy.Bounce` output: this is
    the single boundary every angle table crosses on its way to a run
    (``bellhop_writer``, ``oalib_writer``). An ``.irc`` is a different format
    and is staged untouched.

    Parameters
    ----------
    reflection_file : str or Path or None
        Source table; ``None`` is the caller's configuration error.
    env_path : str or Path
        Path of the ``.env`` being written.
    boundary : {'bottom', 'top', 'internal'}
        Which table this is, selecting the destination suffix. ``'internal'``
        is the ``BotOpt='P'`` precalculated table BOUNCE writes.
    verbose : bool, optional
        Log the staging step.

    Returns
    -------
    Path
        The staged destination path.
    """
    suffix = _REFLECTION_SUFFIX[boundary]
    internal = boundary == 'internal'
    acoustic_type = 'precalc' if internal else 'file'
    remediation = (
        "Run BOUNCE on the seabed first and pass "
        "result.metadata['irc_file'] as reflection_file=, or use "
        "acoustic_type='half-space' to model the seabed directly."
        if internal else
        "Generate the table via BOUNCE or OASR and pass its path as "
        "reflection_file=."
    )
    if not reflection_file:
        raise ConfigurationError(
            f"acoustic_type={acoustic_type!r} requires reflection_file= on the "
            f"{'bottom' if internal else boundary} BoundaryProperties "
            f"(path to a {suffix} table).",
            remediation=remediation,
        )
    src = Path(reflection_file)
    if not src.exists():
        raise ConfigurationError(
            f"{'Internal reflection' if internal else 'Reflection'} "
            f"coefficient file not found: {src}",
            remediation=remediation,
        )
    dest = Path(env_path).with_suffix(suffix)
    if src.resolve() != dest.resolve():
        shutil.copy(src, dest)
    if not internal:
        dedupe_reflection_file(dest)
    log_message('refl_io', f"staged {boundary} reflection file: {src} -> {dest}",
                verbose=verbose)
    return dest


def stage_source_beam_pattern(
    pattern: Union[np.ndarray, str, Path], dest: Union[str, Path]
) -> None:
    """Materialise a source beam pattern at ``dest`` as a ``.sbp`` file.

    ``pattern`` is either a path to an existing ``.sbp`` (copied verbatim)
    or an ``(N, 2)`` array of ``[angle_deg, level_dB]``.
    """
    if isinstance(pattern, (str, Path)):
        src = Path(pattern)
        if not src.exists():
            raise ConfigurationError(
                f"Source beam pattern file not found: {src}"
            )
        # ``bellhop.f90:273`` interpolates with
        # ``s = (SrcDeclAngle - SrcBmPat(IBP,1)) / (SrcBmPat(IBP+1,1) -
        # SrcBmPat(IBP,1))`` and never checks the denominator; what keeps a
        # repeated angle out of that division is the load-time guard —
        # ``misc/beampattern.f90:56`` (and ``bellhopcuda/src/module/sbp.hpp``
        # :114) rejects a pattern whose angles are not *strictly* increasing,
        # ``misc/monotonicMod.f90:32`` counting a repeated abscissa as
        # non-monotonic. Checking here turns that mid-run engine abort into a
        # Python-side error, and matches the check ``Source`` applies to the
        # array form so the guard does not depend on which way the caller
        # happened to supply the pattern.
        from uacpy.core._carrier_validate import _require_strictly_increasing
        angles = read_source_beam_pattern(src)[:, 0]
        # A one- or zero-row table survives every guard on both sides:
        # ``_require_strictly_increasing`` returns early for ``size <= 1``, and
        # ``misc/monotonicMod.f90:30`` pre-sets ``.TRUE.`` then returns for
        # ``N == 1`` (for ``N == 0`` its ``ANY`` runs over zero-size sections and
        # is also false), so ``misc/beampattern.f90:56`` passes it too. The
        # engines then index the table as a pair: ``bellhop.f90:270`` clamps
        # ``IBP`` to ``NSBPPts - 1`` and reads below the bound allocated at
        # ``beampattern.f90:36``, while ``field.f90:190-198`` brackets with
        # ``x(iseg + 1)``. Bellhop yields an all-NaN field and field.exe a
        # finite, plausible, wrong one — both at exit code 0 with nothing in the
        # print file.
        if angles.size < 2:
            raise ConfigurationError(
                f"source beam pattern {src.name} declares {angles.size} row(s); "
                f"the engines interpolate between adjacent rows and need at "
                f"least 2 (Bellhop/bellhop.f90:270).",
                remediation="Pass beam_pattern=None for an omnidirectional "
                            "source, or give at least two angles spanning the "
                            "launch fan.",
            )
        _require_strictly_increasing(
            angles, f"source beam-pattern angles in {src.name}")
        shutil.copy(src, dest)
        return
    arr = np.asarray(pattern, dtype=float)
    write_source_beam_pattern(dest, arr[:, 0], arr[:, 1])


def dedupe_reflection_file(filepath: Union[str, Path]) -> None:
    """Rewrite a ``.brc`` / ``.trc`` file with a strictly-increasing angle axis
    and an unwrapped phase column — the two preconditions
    ``misc/RefCoef.f90`` places on the table.

    BOUNCE sweeps the horizontal wavenumber uniformly from ``omega/c_high``
    to ``omega/c_low`` (``Kraken/bounce.f90:45-46``, ``:172-175``). Every
    sample whose phase speed falls below the reference speed — the top
    half-space speed, else 1500 m/s (``:186-190``) — is evanescent and takes
    the ``ELSEWHERE`` branch at ``:204-209``, which assigns ``theta = 0``,
    ``R = 1``, ``phase = 180`` identically. That block is ``1 - c_low/c0`` of
    the sweep — with uacpy's defaults (``c_low`` 1400 m/s, ``c_high``
    unbounded, so ``kMin`` collapses to 0 at ``:47``) about 7% of a
    several-thousand-row table, i.e. hundreds of byte-identical 0-degree rows
    at the head of the file. Bellhop tolerates them — ``misc/RefCoef.f90:45-55``
    reads the table with no monotonicity test — but bellhopcuda rejects them at
    ``bellhopcuda/src/module/reflcoef.hpp:135-141``: "Bottom reflection
    coefficients must be monotonically increasing".

    This loads the file, keeps only records whose angle strictly exceeds
    the previous kept record, and rewrites it as one
    ``(angle_deg, |R|, phase_deg)`` record per line. The parse is
    tokenised, not line-oriented: ``misc/RefCoef.f90:53`` reads the whole
    table with a single list-directed READ, so a legal input may pack
    several records on one line or wrap one across lines.

    ``.brc`` / ``.trc`` only. An ``.irc`` carries a title/frequency line ahead
    of its count and six fixed-format columns, so running this over one would
    strip the header and four of the six columns; a file whose first line is
    not a bare count is rejected.

    Discarding the evanescent block down to its first row is loss-free — the
    rows are equal by construction. Should two *propagating* samples ever
    print equal angles, the first (lowest-c, shallowest-angle) is kept and
    the rest dropped with no averaging or re-gridding, costing one R(θ)
    sample; ``Kraken/bounce.f90:235`` writes the angle list-directed from a
    ``REAL(KIND=8)``, so distinct wavenumbers do not normally collide there.
    """
    filepath = Path(filepath)
    with open(filepath, 'r') as fh:
        lines = fh.readlines()

    if not lines:
        return

    # The count is a list-directed scalar READ (RefCoef.f90:45), so a comma
    # or trailing annotation after the number is a valid record here too.
    try:
        n_declared = list_directed_int(lines[0])
    except (FileFormatError, ValueError, IndexError):
        raise FileFormatError(
            f"{filepath}: expected a .brc/.trc table, whose first line is the "
            f"row count; got {lines[0].strip()!r}.",
            remediation="An .irc has a title/frequency header and six "
                        "fixed-format columns — pass it through unmodified.",
        ) from None

    # RefCoef.f90:53 consumes the whole table with ONE list-directed READ of
    # 3*N values, so the (theta, R, phi) triples may be packed several per
    # line or wrapped across lines. Tokenise everything after the count line
    # and chunk by 3; a line-oriented walk would silently drop every record
    # after the first on a packed line.
    tokens = []
    for line in lines[1:]:
        tokens.extend(line.replace(',', ' ').split())
    try:
        n_triples = min(n_declared, len(tokens) // 3)
        triples = [
            (tokens[3 * i], tokens[3 * i + 1], tokens[3 * i + 2],
             fortran_float(tokens[3 * i]))
            for i in range(n_triples)
        ]
    except ValueError as exc:
        raise FileFormatError(
            f"{filepath}: the reflection table holds a non-numeric token "
            f"({exc}); the file is not a .brc/.trc angle table."
        ) from None

    kept_rows = []  # list of (angle, mag, phase_deg) as strings
    last_angle = -np.inf
    for a_tok, r_tok, p_tok, angle in triples:
        if angle > last_angle:
            kept_rows.append((a_tok, r_tok, p_tok))
            last_angle = angle

    # If nothing survived dedup (degenerate case), leave the file alone —
    # downstream reader will surface the real error.
    if not kept_rows:
        return

    # ``misc/RefCoef.f90:119`` states the contract for this table outright —
    # "Assumes phi has been unwrapped so that it varies smoothly" — and
    # ``InterpolateReflectionCoefficient`` then interpolates phi linearly
    # between the bracketing abscissas. BOUNCE takes the principal value at
    # ``Kraken/bounce.f90:203`` and attempts an unwrap at ``:212-223``, but the
    # incrementing branch (``:219``) tests ``AIMAG(R1) > 0 .AND. AIMAG(R1) < 0``
    # — a contradiction — so only the decrementing branch can fire and the
    # result is not smooth: measured on a 10 m sediment over a 1800 m/s
    # half-space at 500 Hz, the table jumps 298.8 deg between adjacent angles.
    # Interpolating across such a step sweeps the phase the long way round and
    # returns a reflection phase that is wrong through the whole interval.
    phases = np.degrees(np.unwrap(
        np.radians([fortran_float(p) for _a, _r, p in kept_rows])))

    with open(filepath, 'w') as fh:
        # BOUNCE pads the count with leading whitespace; match that so any
        # downstream tool expecting free-format reads happily.
        fh.write(f"{len(kept_rows):12d}\n")
        for (a, r, _p), phase in zip(kept_rows, phases):
            fh.write(f"   {a}        {r}        {phase:.6f}\n")
