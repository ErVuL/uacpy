"""Reflection-coefficient + source-beam-pattern auxiliary file I/O.

* ``.brc`` / ``.trc`` — precomputed plane-wave reflection coefficients as
  ``(angle_deg, |R|, phase_deg)`` triples behind a count line
  (:func:`read_reflection_coefficient`,
  :func:`write_reflection_coefficient`)

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
import warnings
from dataclasses import dataclass

import numpy as np
from pathlib import Path
from typing import Optional, Union

from uacpy._log import log_message
from uacpy.core._export import ExportRecord
from uacpy.core.units import deg_to_rad, rad_to_deg
from uacpy.core.deck_limits import SBP_ANGLE_RESOLUTION_DEG
from uacpy.core.results import ReflectionCoefficient
from uacpy.core.exceptions import (
    ConfigurationError, FileFormatError, IOWarning, MissingFileError,
)
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.io._fortran_helpers import _bound_counts
from uacpy.io.input_checks import _collapsed_pair_index
from uacpy.io._fortran_helpers import (
    expand_repeat_counts,
    fortran_float, list_directed_int, read_list_directed_values,
    typed_format_error,
)


@typed_format_error
def read_reflection_coefficient(
    filepath: Union[str, Path]
) -> ReflectionCoefficient:
    """
    Read reflection coefficient data from file (.trc or .brc).

    Reads angle-dependent reflection coefficient data used by BELLHOP
    for top or bottom boundary conditions. Data includes angle, magnitude,
    and phase.

    Parameters
    ----------
    filepath : str or Path
        Path to a reflection coefficient file. An existing path is read as
        given whatever its suffix (.brc bottom / .trc top — both carry the
        same table layout); a path that does not exist has ``.brc``
        appended, so a bare base name resolves to the conventional bottom
        table.

    Returns
    -------
    table : ReflectionCoefficient
        A single-frequency table: ``angles_deg`` (grazing),
        ``magnitude`` and ``phase`` (radians), one entry per
        record the header declared (``len(table.angles_deg)``), in the
        travelling-wave phase convention the Acoustics Toolbox tables
        carry. No frequency is stamped: the file holds none.

    Raises
    ------
    ~uacpy.core.exceptions.MissingFileError
        The file is absent.
    ~uacpy.core.exceptions.FileFormatError
        The file is malformed, its angles decrease, or a row is
        not a reflection coefficient (a negative magnitude or a non-finite
        angle, which :class:`~uacpy.core.results.ReflectionCoefficient`
        refuses).

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
    filepath = Path(filepath)
    if not filepath.exists() and not filepath.suffix:
        filepath = filepath.with_name(filepath.name + ".brc")

    try:
        with open(filepath, "r") as fid:
            # The count is a list-directed scalar READ (RefCoef.f90:45), so
            # a comma or trailing annotation after the number is a valid
            # record.
            n_pts = list_directed_int(fid.readline())

            # Each point is three numbers of at least 2 bytes each (value +
            # separator) on disk; a count beyond that is a malformed header,
            # not a huge table.
            _bound_counts(filepath, Path(filepath).stat().st_size, 6,
                          n_pts=n_pts)
            # RefCoef.f90:53 reads the whole table with ONE list-directed
            # READ of n_pts (theta, R, phi) records, so the 3*n_pts values
            # can be packed or wrapped across lines arbitrarily.
            values = read_list_directed_values(
                fid, 3 * n_pts, f"{n_pts} (theta, R, phi) records", filepath)
            table = values.reshape(n_pts, 3)
            theta = table[:, 0]
            R = table[:, 1]
            phi = deg_to_rad(table[:, 2])

            # Validate angles are non-decreasing
            if not np.all(np.diff(theta) >= 0):
                raise FileFormatError(
                    f"Reflection coefficient file {filepath}: angles must be "
                    f"non-decreasing (got a decreasing step in the theta "
                    f"column). The file on disk is malformed."
                )

            try:
                return ReflectionCoefficient(angles_deg=theta, magnitude=R, phase=phi)
            except ConfigurationError as exc:
                # The table's own guards (a negative magnitude, a
                # non-finite angle) judge the file, not a call argument.
                raise FileFormatError(
                    f"Reflection coefficient file {filepath}: "
                    f"{exc.message.rstrip('.')}.",
                    remediation="Every row needs a finite grazing angle "
                                "and a magnitude >= 0 (a sign belongs in "
                                "the phase column); regenerate the table "
                                "with Bounce or write_reflection_coefficient, "
                                "or correct the row.",
                ) from exc

    except FileNotFoundError as e:
        raise MissingFileError(
            f"Reflection coefficient file not found: {filepath}. "
            "Run Bounce first to generate the .brc file, or write a "
            "table with write_reflection_coefficient, and pass its path "
            "as reflection_file=. OASR's .trc is a different layout and "
            "is not read here."
        ) from e


@dataclass(frozen=True, eq=False)
class BeamPatternTable(ExportRecord):
    """A ``.sbp`` source beam pattern as :func:`read_source_beam_pattern`
    reads it (``misc/beampattern.f90:33-44``).

    Attributes
    ----------
    angles_deg : ndarray
        The declination angles (degrees), one per row of the file, in file
        order.
    levels_dB : ndarray
        The level at each angle (dB re peak), as written; the engines convert
        it to a linear amplitude (``10**(dB/20)``, ``beampattern.f90:59``)
        after reading.
    """

    angles_deg: np.ndarray
    levels_dB: np.ndarray

    _REPR_UNITS = {'angles_deg': 'deg'}

    _ARRAY_FIELDS = ('angles_deg', 'levels_dB')
    _TABLE_FIELDS = ('angles_deg', 'levels_dB')
    _XARRAY_FIELDS = {'levels_dB': 'levels_dB', 'angle': 'angles_deg'}

    def _payload(self):
        return {'levels_dB': (self.levels_dB, ('angle',), 'dB')}

    def _coords(self):
        return {'angle': (self.angles_deg, 'deg')}

    def __array__(self, dtype=None, copy=None):
        """The ``(N, 2)`` ``[angle_deg, level_dB]`` table this record holds:
        the array :attr:`uacpy.Source.beam_pattern` and
        :func:`~uacpy.plot.plot_beam_pattern` read, so either takes the record
        itself. Built on request, so ``copy=False`` cannot be honoured."""
        if copy is False:
            raise ValueError("BeamPatternTable: the (N, 2) table is built on "
                             "request, so it cannot be returned without a "
                             "copy.")
        table = np.column_stack([self.angles_deg, self.levels_dB])
        return table if dtype is None else table.astype(dtype)


@typed_format_error
def read_source_beam_pattern(filepath: Union[str, Path]) -> BeamPatternTable:
    """
    Read source beam pattern from file.

    Parameters
    ----------
    filepath : str or Path
        Source beam pattern file; a root name without the ``.sbp`` extension
        also resolves.

    Returns
    -------
    BeamPatternTable
        ``angles_deg`` and ``levels_dB`` (dB re peak), one entry per
        row of the file. ``Source(beam_pattern=t)`` and
        :func:`~uacpy.plot.plot_beam_pattern` take the record itself;
        ``np.asarray(t)`` is its ``(N, 2)`` ``[angle_deg, level_dB]`` table.

    Raises
    ------
    ConfigurationError
        No ``.sbp`` file at the given path.
    FileFormatError
        Malformed ``.sbp`` file.

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

    # Take the path as given when it already exists; a bare root without an
    # extension resolves to the conventional .sbp name (mirrors
    # read_reflection_coefficient). A missing path with an extension is left
    # untouched so the error names the file that was asked for.
    sbp_file = Path(filepath)
    if not sbp_file.exists() and not sbp_file.suffix:
        sbp_file = sbp_file.with_name(sbp_file.name + ".sbp")
    # ``stage_source_beam_pattern`` raises the same type for the identical
    # condition.
    if not sbp_file.exists():
        raise MissingFileError(
            f"Source beam pattern file not found: {sbp_file}.",
            remediation="Provide the .sbp file next to the env, or pass "
                        "beam_pattern=None for an omni-directional source.",
        )
    with open(sbp_file, "r") as fid:
        # The count is a list-directed scalar READ (beampattern.f90:33), so
        # a comma or trailing annotation after the number is a valid record.
        NSBPPts = list_directed_int(fid.readline())

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

    return BeamPatternTable(angles_deg=beam_pattern[:, 0],
                            levels_dB=beam_pattern[:, 1])


def write_reflection_coefficient(
    filepath: Union[str, Path],
    angles_deg: np.ndarray,
    coefficient: np.ndarray,
) -> None:
    """
    Write a ``.brc`` / ``.trc`` reflection-coefficient table — the writer
    side of :func:`read_reflection_coefficient`.

    Parameters
    ----------
    filepath : str or Path
        Output path, written exactly as given: ``.brc`` for a bottom table,
        ``.trc`` for a top one (the layout is identical; the extension is
        what tells the engine which boundary it belongs to). No suffix is
        added or replaced. :func:`stage_reflection_file` is what renames a
        table onto the ``<env>.brc`` / ``.trc`` the binary opens.
    angles_deg : ndarray
        Grazing angles in **degrees**, shape ``(N,)``, non-decreasing.
    coefficient : ndarray
        The table's other two columns, in any of three forms:

        - complex ``(N,)`` — ``|R|`` and ``np.angle`` (radians) are taken;
        - real ``(N, 2)`` — ``[amplitude, phase_radians]``, the amplitude
          ``>= 0`` (a sign belongs in the phase);
        - real ``(N,)`` — the real coefficient itself: phase zero, and a
          negative entry (``R = -1`` for a pressure-release boundary) is
          written as ``|R|`` with phase π, the same coefficient.

    Raises
    ------
    ~uacpy.core.exceptions.ConfigurationError
        The two columns disagree in length, the table is empty, an angle is
        not finite, the angles decrease, two distinct angles collide in the
        written column, or an ``(N, 2)`` amplitude is negative — each a
        table :func:`read_reflection_coefficient` would refuse.

    Notes
    -----
    File format (``misc/RefCoef.f90:45,53``): a count line, then ``N``
    ``THETA(deg)  RMAG  RPHASE(deg)`` triples read by a single list-directed
    ``READ``.

    **Phase is radians in, degrees on disk.** That direction matches
    :func:`read_reflection_coefficient`, whose ``phase`` is in radians,
    and the :class:`~uacpy.core.results.ReflectionCoefficient` convention —
    so a table read, modified and written back keeps its units.

    Guards this writer applies that the format does not:

    * **Non-decreasing angles.** ``read_reflection_coefficient`` rejects a
      decreasing theta column as malformed, so a writer that emitted one
      would produce a file uacpy itself refuses to read. Equal angles are
      allowed — BOUNCE produces them by construction, which is what
      :func:`dedupe_reflection_file` exists to collapse.
    * **Angles that survive the written column.** The angle column is
      written at ``%12.6f``, i.e.
      :data:`~uacpy.core.deck_limits.SBP_ANGLE_RESOLUTION_DEG` = 1e-6 degrees.
      Two *distinct* angles closer than that land on the same token, turning
      a resolved table into a duplicated one: ``bhc::setup()`` aborts on it
      ("Bottom reflection coefficients must be monotonically increasing")
      and Bellhop interpolates across a zero-width segment. Refused here,
      the way :func:`write_source_beam_pattern` refuses a colliding beam
      axis.

    See Also
    --------
    read_reflection_coefficient : Read the table back.
    stage_reflection_file : Copy a table onto the name a run opens.
    dedupe_reflection_file : Collapse BOUNCE's duplicate angle rows.

    Examples
    --------
    >>> import tempfile, os
    >>> import numpy as np
    >>> from uacpy.io.refl_io import read_reflection_coefficient
    >>> theta = np.array([0.0, 45.0, 90.0])
    >>> R = np.array([0.9, 0.5, 0.1]) * np.exp(1j * np.array([0.0, 1.0, 2.0]))
    >>> with tempfile.TemporaryDirectory() as d:
    ...     path = os.path.join(d, 'bottom.brc')
    ...     write_reflection_coefficient(path, theta, R)
    ...     table = read_reflection_coefficient(path)
    >>> table.n_angles
    3
    >>> bool(np.allclose(table.angles_deg, theta))
    True
    >>> bool(np.allclose(table.magnitude, np.abs(R)))
    True
    >>> bool(np.allclose(table.phase, np.angle(R)))
    True
    """
    filepath = Path(filepath)

    angles_deg = np.asarray(angles_deg, dtype=float)
    coefficient = np.asarray(coefficient)

    if np.iscomplexobj(coefficient):
        amplitude = np.abs(coefficient)
        phase_rad = np.angle(coefficient)
    elif coefficient.ndim == 2 and coefficient.shape[1] == 2:
        amplitude = np.asarray(coefficient[:, 0], dtype=float)
        phase_rad = np.asarray(coefficient[:, 1], dtype=float)
        negative = np.flatnonzero(amplitude < 0.0)
        if negative.size:
            i = int(negative[0])
            raise ConfigurationError(
                f"write_reflection_coefficient: coefficient[{i}, 0] = "
                f"{amplitude[i]:g} is a negative amplitude; the file's "
                f"magnitude column holds |R| >= 0 and "
                f"read_reflection_coefficient refuses a negative one.",
                remediation="Pass |R| in column 0 with π added to the phase "
                            "where R is negative, or pass the complex R.")
    else:
        # A real R is the coefficient itself; its magnitude column holds |R|,
        # so a negative entry is |R| at phase π, the same number.
        real = np.asarray(coefficient, dtype=float)
        amplitude = np.abs(real)
        phase_rad = np.where(real < 0.0, np.pi, 0.0)

    # The row loop is driven by len(angles), so a longer coefficient column
    # is silently truncated into a valid-looking table that describes a
    # different bottom, and a shorter one raises IndexError partway through
    # and leaves the truncated file behind. Checked before ``open``, the same
    # way write_source_beam_pattern pairs its two columns.
    if amplitude.shape != angles_deg.shape:
        raise ConfigurationError(
            f"write_reflection_coefficient: angles has shape {angles_deg.shape} "
            f"and the coefficient column has shape {amplitude.shape}; each "
            f"angle needs exactly one (|R|, phase) pair.",
            remediation="Pass one reflection coefficient per angle.",
        )
    if angles_deg.size == 0:
        raise ConfigurationError(
            "write_reflection_coefficient: no angles given; a reflection "
            "table declaring 0 rows leaves every grazing angle outside the "
            "tabulated domain, where InterpolateReflectionCoefficient "
            "returns R = 0 — a totally absorbing boundary "
            "(misc/RefCoef.f90:144-149).",
            remediation="Pass at least one (angle, coefficient) pair "
                        "spanning the grazing angles of the run.",
        )

    if not np.all(np.isfinite(angles_deg)):
        i = int(np.flatnonzero(~np.isfinite(angles_deg))[0])
        raise ConfigurationError(
            f"write_reflection_coefficient: angles_deg[{i}] = "
            f"{angles_deg[i]!r} is not a finite grazing angle; "
            f"read_reflection_coefficient refuses such a row.",
            remediation="Pass finite grazing angles in degrees.")
    steps = np.diff(angles_deg)
    if steps.size and steps.min() < 0.0:
        i = int(np.argmin(steps))
        raise ConfigurationError(
            f"write_reflection_coefficient: angles must be non-decreasing, "
            f"but index {i} -> {i + 1} goes {angles_deg[i]:g} -> "
            f"{angles_deg[i + 1]:g} degrees. read_reflection_coefficient rejects "
            f"a decreasing theta column as malformed, so this would write a "
            f"file uacpy cannot read back.",
            remediation="Sort the table by grazing angle before writing.",
        )
    # Distinct angles that collide once rounded into the written column.
    written = np.round(angles_deg, 6)
    i = _collapsed_pair_index(written, raw=angles_deg)
    if i is not None:
        raise ConfigurationError(
            f"write_reflection_coefficient: angles[{i}]={angles_deg[i]!r} and "
            f"angles[{i + 1}]={angles_deg[i + 1]!r} are distinct but both write "
            f"{written[i]:.6f}, the file's "
            f"{SBP_ANGLE_RESOLUTION_DEG:g}-degree angle resolution. The "
            f"table would carry a duplicated angle it does not have: "
            f"bellhopcuda aborts on a non-strictly-increasing table and "
            f"Bellhop interpolates across the zero-width segment.",
            remediation=f"Separate the angles by at least "
                        f"{SBP_ANGLE_RESOLUTION_DEG:g} degrees, or drop the "
                        f"one you do not need.",
        )

    phase_deg = rad_to_deg(phase_rad)

    with open(filepath, "w") as f:
        f.write(f"{angles_deg.size}\n")
        for theta, mag, phase in zip(angles_deg, amplitude, phase_deg):
            f.write(f"{theta:12.6f} {mag:12.6f} {phase:12.6f}\n")


def write_source_beam_pattern(
    filepath: Union[str, Path], angles_deg: np.ndarray, beam_pattern: np.ndarray
) -> None:
    """
    Write a source beam pattern file (``.sbp``), read by Bellhop and by
    KRAKEN's ``field.exe`` (``KrakenField/field.f90:101-102``).

    Parameters
    ----------
    filepath : str or Path
        Output file path (typically .sbp extension)
    angles_deg : ndarray
        Beam angles in degrees from the horizontal, positive downward
        (Bellhop's launch declination), shape (N,); typically from -90 to
        +90 degrees
    beam_pattern : ndarray
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

    At least two angles are required, and the angle column is written at
    ``%.6f``: Bellhop's load-time guard rejects a pattern whose angles are
    not strictly increasing (``misc/beampattern.f90:56``), so two distinct
    angles that collide on the file's angle grid abort the run with ERROUT.
    Angles closer than that column resolution are rejected here for the same
    reason, against the
    :data:`~uacpy.core.deck_limits.SBP_ANGLE_RESOLUTION_DEG` bound
    :class:`~uacpy.core.source.Source` validates a carrier's angles with.

    ``angles_deg`` and ``beam_pattern`` are paired row for row and must have the same
    shape; a mismatch raises :class:`~uacpy.core.exceptions.ConfigurationError`
    before the file is opened.
    """
    filepath = Path(filepath)

    angles_deg = np.asarray(angles_deg, dtype=float)
    # ``dtype=float`` up front so a non-numeric column raises a typed
    # ConfigurationError below rather than escaping the row loop as a bare
    # ``ValueError`` from the format spec, after the file is already open.
    try:
        beam_pattern = np.asarray(beam_pattern, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"write_source_beam_pattern: beam_pattern is not a numeric "
            f"array ({exc}); the level column is written at %12.6f.",
            remediation="Pass beam-pattern levels in dB re peak as a real "
                        "array of the same length as angles_deg.",
        ) from exc
    # A table shorter than two rows passes every guard on both sides and then
    # gets indexed as a pair: ``bellhop.f90:270`` clamps ``IBP`` to
    # ``NSBPPts - 1`` and reads ``SrcBmPat(IBP+1, :)`` past the bound allocated
    # at ``beampattern.f90:36``, yielding an all-NaN field at exit code 0 with
    # nothing in the print file. ``stage_source_beam_pattern`` refuses the path
    # form for the same reason, so refusing here makes the guard independent of
    # which way the caller supplied the pattern.
    if angles_deg.size < 2:
        raise ConfigurationError(
            f"write_source_beam_pattern: {angles_deg.size} angle(s) given; the "
            f"engines interpolate between adjacent beam-pattern rows and need "
            f"at least 2 (Bellhop/bellhop.f90:270).",
            remediation="Give at least two angles spanning the launch fan, or "
                        "omit the beam pattern for an omnidirectional source.",
        )
    i = _collapsed_pair_index(angles_deg, min_step=SBP_ANGLE_RESOLUTION_DEG)
    if i is not None:
        raise ConfigurationError(
            f"write_source_beam_pattern: angles_deg[{i}]={angles_deg[i]!r} "
            f"and angles_deg[{i + 1}]={angles_deg[i + 1]!r} are closer than "
            f"the file's "
            f"{SBP_ANGLE_RESOLUTION_DEG:g}-degree angle resolution (or not "
            f"increasing). Bellhop requires strictly increasing beam-pattern "
            f"angles (misc/beampattern.f90:56) and aborts on a repeated "
            f"value.",
            remediation=f"Pass strictly increasing angles with steps of at "
                        f"least {SBP_ANGLE_RESOLUTION_DEG:g} degrees.",
        )
    # The row loop is driven by ``len(angles)``, so a longer ``beam_pattern`` is
    # truncated to a valid-looking table that carries a different directivity
    # than the caller passed, and a shorter one raises ``IndexError`` partway
    # through the write and leaves the truncated file behind. Both columns
    # declare shape (N,), so requiring the shapes to agree — before ``open`` —
    # is what makes the written table the one the caller described.
    if beam_pattern.shape != angles_deg.shape:
        raise ConfigurationError(
            f"write_source_beam_pattern: angles_deg has shape "
            f"{angles_deg.shape} and beam_pattern has shape "
            f"{beam_pattern.shape}; each angle needs exactly one level, so "
            f"the two columns must have the same shape (N,).",
            remediation="Pass one beam-pattern level per angle.",
        )

    n_angles = len(angles_deg)

    with open(filepath, "w") as f:
        # Write number of angles
        f.write(f"{n_angles}\n")

        # Write angle, amplitude pairs
        for i in range(n_angles):
            f.write(f"{angles_deg[i]:12.6f} {beam_pattern[i]:12.6f}\n")


_REFLECTION_SUFFIX = {'bottom': '.brc', 'top': '.trc', 'internal': '.irc'}


def stage_reflection_file(
    reflection_file: Optional[Union[str, Path]],
    env_path: Union[str, Path],
    boundary: str = 'bottom',
    verbose: Union[bool, str] = False,
    *,
    drop_repeated_angles: bool = False,
) -> Path:
    """Place a reflection-coefficient table beside the ``.env`` that names it.

    AT and Bellhop open the table by base-name convention (``<env>.brc`` for
    the bottom, ``<env>.trc`` for the top, ``<env>.irc`` for a precalculated
    internal table), so a table produced elsewhere has to be copied next to
    the environment file. A table already at the destination — a BOUNCE run
    whose ``.brc`` sits in the same pinned ``work_dir`` the ``.env`` is
    written into — is copied over itself, i.e. left in place.

    A copied ``.brc`` / ``.trc`` keeps every row: the Fortran binaries read
    it with no monotonicity test (``misc/RefCoef.f90:45-55``). Only for a
    binary that refuses a repeated angle — bellhopcuda and bellhopcxx test
    the angle column with ``<=`` (``bellhopcuda/src/module/reflcoef.hpp:
    135-141``, ``src/common_setup.hpp:243-253``) and stop with
    "reflection coefficients must be monotonically increasing" — does the
    caller pass ``drop_repeated_angles=True``: the copy then keeps the first
    row of each repeated angle, and one ``IOWarning`` names the rows
    dropped. Either way the copy's phase column is unwrapped, which changes
    no coefficient and gives the engine the column ``misc/RefCoef.f90:119``
    assumes (:func:`_normalise_staged_table`). An ``.irc`` is a different
    format and is staged untouched. :class:`~uacpy.Bounce` dedupes and
    unwraps its own ``.brc`` where it writes it.

    A table already *at* the destination is never rewritten: with the source
    and the destination the same file, a dedupe would edit an input. Such a
    table is checked and warned about instead.

    Parameters
    ----------
    reflection_file : str or Path or None
        Source table; ``None`` is the caller's configuration error.
    env_path : str or Path
        Path of the ``.env`` being written.
    boundary : {'bottom', 'top', 'internal'}
        Which table this is, selecting the destination suffix. ``'internal'``
        is the ``BotOpt='P'`` precalculated table BOUNCE writes.
    verbose : bool or str, optional
        Log the staging step.
    drop_repeated_angles : bool, optional
        Drop the rows of a copied ``.brc`` / ``.trc`` that repeat the angle
        of the row before, for a binary that refuses them (bellhopcuda,
        bellhopcxx). Default ``False``: every row is staged.

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
        "Generate the table with BOUNCE (result.metadata['brc_file']) or "
        "write one with write_reflection_coefficient, and pass its path as "
        "reflection_file= (OASR's .trc is a different layout)."
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
        raise MissingFileError(
            f"{'Internal reflection' if internal else 'Reflection'} "
            f"coefficient file not found: {src}.",
            remediation=remediation,
        )
    dest = Path(env_path).with_suffix(suffix)
    in_place = src.resolve() == dest.resolve()
    if not in_place:
        shutil.copy(src, dest)
    if not internal and not in_place:
        dropped = _normalise_staged_table(dest, drop_repeated_angles)
        if dropped:
            warnings.warn(
                f"{src}: {len(dropped)} row(s) repeat the angle of the row "
                f"before (at {', '.join(f'{a:g}' for a in dropped[:5])}"
                f"{' …' if len(dropped) > 5 else ''} deg). bellhopcuda and "
                f"bellhopcxx refuse a repeated angle "
                f"(src/module/reflcoef.hpp:135-141), so the copy staged at "
                f"{dest} keeps the first row of each repeat; your file is "
                f"unchanged. The Fortran Bellhop reads the table as given.",
                IOWarning, skip_file_prefixes=USER_FRAME_SKIP)
    elif not internal:
        # The table the engine will read *is* the caller's own file — a pinned
        # work_dir holding a .brc next to the .env it names. Rewriting it would
        # drop rows from and reformat a file uacpy was given, not one it
        # produced, so the table is left exactly as supplied. Bellhop reads it
        # either way (misc/RefCoef.f90:45-55 applies no monotonicity test); it
        # is bellhopcuda that refuses a repeated angle
        # (src/module/reflcoef.hpp:135-141), so say so instead of editing.
        table = read_reflection_coefficient(dest)
        angles = table.angles_deg
        if angles.size > 1 and not np.all(np.diff(angles) > 0):
            warnings.warn(
                f"{dest} is both the reflection table you supplied and the "
                f"name the engine reads, so it is staged unmodified; its "
                f"angle column repeats a value, which bellhopcuda rejects "
                f"(src/module/reflcoef.hpp:135-141). Pass the table from a "
                f"path other than {dest} to have uacpy stage a cleaned copy.",
                IOWarning,
                skip_file_prefixes=USER_FRAME_SKIP,
            )
        # The engine interpolates phi linearly between bracketing rows and
        # assumes the column is unwrapped (misc/RefCoef.f90:119,157-160); a
        # step past a half turn is swept the long way round through the
        # whole interval. The copy path unwraps; here the table is not
        # edited, so the step is reported instead.
        phase_step_deg = np.abs(np.diff(np.degrees(table.phase)))
        if phase_step_deg.size and np.any(phase_step_deg > 180.0):
            warnings.warn(
                f"{dest} is both the reflection table you supplied and the "
                f"name the engine reads, so it is staged unmodified; its "
                f"phase column steps {phase_step_deg.max():.1f} deg between "
                f"adjacent angles, which the engine interpolates as written "
                f"(misc/RefCoef.f90:119 assumes an unwrapped phase). Unwrap "
                f"the phase column, or pass the table from a path other than "
                f"{dest} to have uacpy stage an unwrapped copy.",
                IOWarning,
                skip_file_prefixes=USER_FRAME_SKIP,
            )
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
            raise MissingFileError(
                f"Source beam pattern file not found: {src}."
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
        from uacpy.core._validate import require_strictly_increasing
        angles = read_source_beam_pattern(src).angles_deg
        # A one- or zero-row table survives every guard on both sides:
        # ``_require_strictly_increasing`` returns early for ``size <= 1``, and
        # ``misc/monotonicMod.f90:30`` pre-sets ``.TRUE.`` then returns for
        # ``N == 1`` (for ``N == 0`` its ``ANY`` runs over zero-size sections and
        # is also false), so ``misc/beampattern.f90:56`` passes it too. The
        # engines then index the table as a pair: ``bellhop.f90:270`` clamps
        # ``IBP`` to ``NSBPPts - 1`` and reads below the bound allocated at
        # ``beampattern.f90:36``, while ``field.f90:203-209`` brackets with
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
        require_strictly_increasing(
            angles, f"source beam-pattern angles in {src.name}")
        shutil.copy(src, dest)
        return
    arr = np.asarray(pattern, dtype=float)
    write_source_beam_pattern(dest, arr[:, 0], arr[:, 1])


#: Column layout of an ``.irc`` table row: ``( 5G15.7, I5 )``
#: (``Kraken/bounce.f90:228`` writes it, ``misc/RefCoef.f90:107`` reads it
#: back with the same fixed format) — kx, Re f, Im f, Re g, Im g, power.
_IRC_FIELD_WIDTH = 15
_IRC_POWER_WIDTH = 5


def _scale_irc_impedance(filepath: Union[str, Path], factor: float) -> None:
    """Multiply the ``g`` column of a BOUNCE ``.irc`` table by ``factor``, in
    place, keeping the fixed ``( 5G15.7, I5 )`` row layout.

    The table holds the Robin-condition pair ``(f, g)`` the consumer's
    bottom boundary uses (``Kraken/BCImpedanceMod.f90:118-125``). For an
    acoustic half-space ``g`` is the seabed density itself (``:85-87``), and
    for an elastic one it scales with the shear modulus, so ``g`` carries the
    density scale of the deck it was computed from while ``f`` does not.
    uacpy's BOUNCE deck states densities relative to the water's (right for
    the ``.brc``, whose ``R`` depends only on the ratio); multiplying ``g`` by
    ``water_density`` puts the table back on the absolute scale Kraken and
    Scooter apply it on. ``iPower`` scales ``f`` and ``g`` together and is
    left as written.
    """
    path = Path(filepath)
    lines = path.read_text().splitlines()
    w = _IRC_FIELD_WIDTH
    out = lines[:2]
    for line in lines[2:]:
        if not line.strip():
            out.append(line)
            continue
        fields = [line[i * w:(i + 1) * w] for i in range(5)]
        power = line[5 * w:5 * w + _IRC_POWER_WIDTH]
        # A G15.7 field drops the exponent letter when the exponent needs
        # three digits; fortran_float reads that form too.
        values = [fortran_float(x) for x in fields]
        values[3] *= factor
        values[4] *= factor
        out.append(''.join(f"{v:{w}.7E}" for v in values) + power)
    path.write_text('\n'.join(out) + '\n')


def _table_rows(filepath: Path):
    """The ``(angle_token, |R|_token, phase_token, angle)`` rows of a
    ``.brc`` / ``.trc`` as :func:`dedupe_reflection_file` documents its
    checks (a bare count first, as many complete triples as it declares,
    numeric tokens, a non-decreasing angle column), or ``None`` for an
    empty file."""
    filepath = Path(filepath)
    with open(filepath, 'r') as fh:
        lines = fh.readlines()

    if not lines:
        return None

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
        tokens.extend(expand_repeat_counts(line.replace(',', ' ').split()))
    # A short file is a truncated file, never a shorter table. Rewriting the
    # header down to what survived would hand Bellhop a table it reads without
    # complaint, and every grazing angle past the cut then falls outside the
    # tabulated domain, where ``InterpolateReflectionCoefficient`` returns
    # ``RInt%R = 0`` — a totally absorbing bottom — with its own warning
    # commented out (misc/RefCoef.f90:144-149). A BOUNCE run killed mid-write
    # and a partly-downloaded ``reflection_file=`` both land here. Extra
    # tokens past the count are fine: the single READ of 3*N values at :53
    # stops at N.
    n_available = len(tokens) // 3
    if n_available < n_declared:
        raise FileFormatError(
            f"{filepath}: the header declares {n_declared} rows but the file "
            f"holds only {n_available} complete (angle, |R|, phase) triples — "
            f"it is truncated.",
            remediation="Re-run BOUNCE or re-fetch the table; correcting the "
                        "count instead would silently give every angle past "
                        "the cut |R| = 0, a totally absorbing bottom "
                        "(misc/RefCoef.f90:144-149).",
        )
    try:
        triples = [
            (tokens[3 * i], tokens[3 * i + 1], tokens[3 * i + 2],
             fortran_float(tokens[3 * i]))
            for i in range(n_declared)
        ]
    except ValueError as exc:
        raise FileFormatError(
            f"{filepath}: the reflection table holds a non-numeric token "
            f"({exc}); the file is not a .brc/.trc angle table."
        ) from None

    # Only an equal angle is a duplicate to collapse; a *decreasing* one means
    # the table runs the other way (90 deg down to 0 is a common grazing
    # convention, and RefCoef.f90 applies no monotonicity test, so such a file
    # reaches here intact). Keeping "angle > last kept" would answer that with
    # a one-row table — one constant reflection coefficient at every angle —
    # so it is rejected on the same terms read_reflection_coefficient uses.
    angles = np.array([angle for *_toks, angle in triples], dtype=float)
    if angles.size > 1 and not np.all(np.diff(angles) >= 0):
        i = int(np.argmin(np.diff(angles)))
        raise FileFormatError(
            f"{filepath}: angles must be non-decreasing (row {i + 1} is "
            f"{angles[i]:g} deg, row {i + 2} is {angles[i + 1]:g} deg).",
            remediation="Write the table from the smallest angle to the "
                        "largest; RefCoef.f90 interpolates it in that order.",
        )

    return triples


def _normalise_staged_table(filepath: Path, drop_repeats: bool, *,
                            rewrite: bool = False) -> list:
    """Rewrite a copied ``.brc`` / ``.trc`` with its phase column unwrapped
    and, when ``drop_repeats``, the first row of each repeated angle kept;
    return the angles (deg) of the rows dropped.

    The unwrap changes no reflection coefficient (each phase moves by whole
    turns) and gives every binary the column ``misc/RefCoef.f90:119``
    assumes ("phi has been unwrapped so that it varies smoothly"). Angles
    and magnitudes keep their tokens; a table that needs neither change is
    left byte for byte unless ``rewrite``, which writes the kept rows in
    any case (dropping tokens past the declared count)."""
    triples = _table_rows(Path(filepath))
    if triples is None:
        return []
    kept, dropped = [], []
    last_angle = -np.inf
    for a_tok, r_tok, p_tok, angle in triples:
        if not drop_repeats or angle > last_angle:
            kept.append((a_tok, r_tok, p_tok))
            last_angle = angle
        else:
            dropped.append(float(angle))
    if not kept:
        return dropped
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
    written = np.array([fortran_float(p) for _a, _r, p in kept])
    phases = np.degrees(np.unwrap(np.radians(written)))
    if (rewrite or dropped
            or not np.allclose(phases, written, rtol=0.0, atol=1e-9)):
        with open(filepath, 'w') as fh:
            fh.write(f"{len(kept):12d}\n")
            for (a, r, _p), phase in zip(kept, phases):
                fh.write(f"   {a}        {r}        {phase:.6f}\n")
    return dropped


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
    the sweep (with ``c_high`` unbounded, so ``kMin`` collapses to 0 at
    ``:47``). Its ceiling is ~7% of a several-thousand-row table, reached
    when ``c_low`` sits at the 1400 m/s cap; it shrinks from there as
    ``c_low`` resolves lower, since an auto ``c_low`` takes
    ``min(1400, min(SSP))`` (``bounce._plan.resolve_c_low``) and the block
    is then ``1 - min(SSP)/c0``. Either way it is hundreds of byte-identical
    0-degree rows at the head of the file. Bellhop tolerates them —
    ``misc/RefCoef.f90:45-55``
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
    not a bare count is rejected. A table whose angle column *decreases*
    anywhere is rejected too, on the same terms
    :func:`read_reflection_coefficient` applies: collapsing duplicates is
    loss-free only on an ascending axis. So is a table holding fewer rows
    than its header declares — a truncated file, which this function must not
    turn into a well-formed shorter one, since the reader downstream would
    then never see the shortfall (the same checks run when
    :func:`stage_reflection_file` drops repeated angles for a binary that
    refuses them).

    Discarding the evanescent block down to its first row is loss-free — the
    rows are equal by construction. Should two *propagating* samples ever
    print equal angles, the first (lowest-c, shallowest-angle) is kept and
    the rest dropped with no averaging or re-gridding, costing one R(θ)
    sample; ``Kraken/bounce.f90:235`` writes the angle list-directed from a
    ``REAL(KIND=8)``, so distinct wavenumbers do not normally collide there.
    """
    _normalise_staged_table(Path(filepath), True, rewrite=True)
