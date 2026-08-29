"""Bathymetry / altimetry file I/O.

Readers and writers for the geometric-boundary auxiliary files attached to
a Bellhop or Acoustics-Toolbox env description:

* ``.bty`` — bathymetry (:func:`read_bathymetry`, :func:`write_bty_file`,
  :func:`write_bty_long_format`)
* ``.ati`` — altimetry (:func:`read_altimetry`, :func:`write_ati_file`)
* 3-D boundary grids (:func:`read_boundary_3d`, :func:`write_bty_3d`)

Reflection coefficients (`.brc`, `.irc`, `.trc`) and source beam patterns
(`.sbp`) live in :mod:`uacpy.io.refl_io`.

**The two 3-D entry points are deliberately retained and are not dead
code.** :func:`read_boundary_3d` and :func:`write_bty_3d` read and write the
BELLHOP3D boundary grid (``bdry3DMod.f90:216-330``). No uacpy model runs
``bellhop3d`` yet — Bellhop's RunType position 6 is hardwired ``'2'`` and
``Bellhop(dimensionality='3D')`` raises — so nothing in the 2-D public API
calls them. They are the foundation a future 3-D implementer builds on, and
``uacpy/tests/test_io_restored_capabilities.py`` pins them against a
dead-code sweep proposing their removal a second time.
"""

import warnings
import numpy as np
from pathlib import Path
from typing import Tuple, Union

from uacpy._log import log_message
from uacpy.core.exceptions import (
    ConfigurationError, FileFormatError,
)
from uacpy.core._warn_frames import USER_FRAME_SKIP
from uacpy.io.units import km_to_m, m_to_km
from uacpy.io.utils import _collapsed_pair_index
from uacpy.io._fortran_helpers import (
    list_directed_int, read_list_directed_values, read_vector,
    strip_fortran_quotes, typed_format_error,
)


def _summarize_axis(arr, head: int = 10, fmt: str = "{:9.5g}") -> str:
    """Compact one-line preview of a numeric axis for debug logging."""
    n = len(arr)
    if n <= head + 1:
        return "[" + ", ".join(fmt.format(v).strip() for v in arr) + f"] ({n} pts)"
    body = ", ".join(fmt.format(v).strip() for v in arr[:head])
    tail = fmt.format(arr[-1]).strip()
    return f"[{body}, …, {tail}] ({n} pts)"


def _read_boundary_2d(
    filepath: Union[str, Path], suffix: str, kind: str, verbose: bool,
) -> Tuple[np.ndarray, str]:
    """Read a BELLHOP 2-D boundary file (``.bty`` or ``.ati``).

    ``ReadATI`` and ``ReadBTY`` (``bdryMod.f90:61-110`` / ``:161-215``) are the
    same parser on two units, down to the long-format branch, so one reader
    serves both.
    """
    # Use the path as given when it already exists (an explicit path is never
    # shadowed by a sibling .bty/.ati); the suffix is appended only as a
    # fallback so a bare base name still resolves to the conventional file.
    filepath = Path(filepath)
    if not filepath.exists() and not filepath.suffix:
        filepath = filepath.with_suffix(suffix)
    # A .bty/.ati is a deck the *user* authored — no uacpy model writes one a
    # uacpy reader reads back — so an absent one is a bad argument, not a
    # failed run. ``ConfigurationError``/``FileFormatError`` split on exactly
    # that provenance (see :class:`~uacpy.core.exceptions.FileFormatError`).
    if not filepath.exists():
        raise ConfigurationError(
            f"{kind.capitalize()} file not found: {filepath}",
            remediation=f"Check the path passed for the {kind} file; a bare "
                        f"root name without an extension resolves to "
                        f"<root>{suffix}.",
        )

    with open(filepath, "r") as fid:
        log_message('bathy_io', f"Reading {kind} file", verbose=verbose)

        type_str = strip_fortran_quotes(fid.readline())

        # TYPE is up to 2 chars: TYPE(1:1) sets the interpolation, TYPE(2:2)
        # selects short (geometry only) or long (geometry + geoacoustics).
        # The comparison is case-sensitive because the Fortran's is:
        # bdryMod.f90's SELECT CASE blocks test only 'C'/'L' (:162-165) and
        # 'S'/''/'L' (:193-199) and ERROUT on anything else, so a lowercase
        # letter that parsed here would still abort the binary.
        interp_type = type_str[:1]
        format_char = type_str[1:2].strip()
        if interp_type not in ("L", "C"):
            raise FileFormatError(
                f"Unknown {kind} type: {interp_type!r} (must be 'L' or 'C'; "
                f"the Fortran SELECT CASE is case-sensitive, bdryMod.f90:162-"
                f"165, so 'l'/'c' abort the engine with ERROUT)"
            )
        if format_char not in ("", "S", "L"):
            raise FileFormatError(
                f"Unknown {kind} format flag TYPE(2:2) = {format_char!r} "
                f"(must be 'S', 'L' or absent; case-sensitive in the Fortran, "
                f"bdryMod.f90:193-199)"
            )
        is_long = format_char == "L"

        approx = "Piecewise-linear" if interp_type == "L" else "Curvilinear"
        log_message('bathy_io', f"{approx} approximation to {kind}",
                    verbose=verbose)

        # The count is a list-directed scalar READ (bdryMod.f90:71 / :171), so
        # '3, ! npts' and '3   ! number of points' are both valid records.
        n_pts = list_directed_int(fid.readline())
        log_message('bathy_io', f"Number of {kind} points = {n_pts}",
                    verbose=verbose)
        # Name the count here: without this the empty row list below indexes an
        # empty array and the typed_format_error wrapper reports a generic
        # IndexError, pointing the user at the data rows rather than the count
        # line that is actually wrong. The binary is no better off — with a
        # non-positive count bdryMod.f90:70-77 allocates NatiPts + 2 entries and
        # its ``DO ii = 2, NatiPts - 1`` fill runs zero times, leaving the
        # boundary uninitialised.
        if n_pts <= 0:
            raise FileFormatError(
                f"{filepath}: the {kind} deck declares {n_pts} points; at "
                f"least one boundary point is required.",
                remediation="Check the count line above the range/depth rows — "
                            "a truncated or misaligned deck reads the wrong "
                            "record as the count.",
            )

        # A long row is the 7 numbers of bdryMod.f90:200-201 — range, depth
        # and the five geoacoustic columns; a short row is range, depth.
        # Each point is ONE list-directed READ (bdryMod.f90:98 / :195), which
        # keeps consuming records until its n_cols values are read and then
        # skips the remainder of its final record — so a row may wrap across
        # any number of lines, and this reader accepts the same files the
        # binary does.
        n_cols = 7 if is_long else 2
        rows = [
            read_list_directed_values(
                fid, n_cols,
                f"{kind} point {i + 1} ({n_cols} columns for "
                f"TYPE {type_str!r})", filepath)
            for i in range(n_pts)
        ]

        data = np.array(rows).T

        log_message('bathy_io', f"range (km): {_summarize_axis(data[0])}",
                    verbose=verbose, level='debug')
        log_message('bathy_io', f"depth (m): {_summarize_axis(data[1])}",
                    verbose=verbose, level='debug')

    data[0, :] = km_to_m(data[0, :])

    # Extend to ±infinity by holding every row constant. The ±1e50 sentinel is
    # the one AT's own reader uses — "extend the bathymetry to +/- infinity in
    # a piecewise constant fashion", Matlab/ReadWrite/readbty.m:66-71 — so a
    # segment search over this array brackets any range without a special case.
    out = np.zeros((n_cols, n_pts + 2))
    out[:, 1:-1] = data
    out[:, 0] = data[:, 0]
    out[:, -1] = data[:, -1]
    out[0, 0] = -1e50
    out[0, -1] = 1e50

    return out, interp_type


@typed_format_error
def read_bathymetry(filepath: Union[str, Path], verbose: bool = False) -> Tuple[np.ndarray, str]:
    """
    Read bathymetry data from BELLHOP .bty file.

    Reads 2D range-depth bathymetry profile with optional interpolation type.
    Extends the bathymetry to ±infinity for computational purposes.

    Parameters
    ----------
    filepath : str or Path
        Path to bathymetry file. An existing path is read exactly as
        given, whatever its extension; a bare root without an extension
        resolves to ``<root>.bty``.
    verbose : bool, optional
        If True, print bathymetry information. Default is False.

    Returns
    -------
    bty : ndarray
        Bathymetry data array of shape (n_cols, N+2) where:
        - bty[0, :] = range in meters (extended to ±1e50 at endpoints)
        - bty[1, :] = depth in meters

        A long-format file (``TYPE(2:2) == 'L'``, written by
        :func:`write_bty_long_format`) carries per-range geoacoustics in the
        remaining rows, in ``bdryMod.f90:200-201`` column order:
        - bty[2, :] = compressional speed (m/s)
        - bty[3, :] = shear speed (m/s)
        - bty[4, :] = density (g/cm³)
        - bty[5, :] = compressional attenuation
        - bty[6, :] = shear attenuation

        First and last points are extended to -infinity and +infinity, every
        row held constant across the extension.
    bty_type : str
        Interpolation type:
        - 'L' : Piecewise-linear
        - 'C' : Curvilinear (cubic spline)

    Notes
    -----
    - Input file ranges are in km, converted to meters on output
    - Bathymetry is extended to ±infinity using constant extrapolation
    - File format:
        Line 1: TYPE in quotes — position 1 is 'L' or 'C' (interpolation),
        position 2 is 'S' (short) or 'L' (long, with geoacoustics)
        Line 2: Number of points
        Lines 3+: range(km) depth(m) [cp cs rho alpha_p alpha_s]

    References
    ----------
    Based on BELLHOP/readbty.m
    """
    return _read_boundary_2d(filepath, ".bty", "bathymetry", verbose)


@typed_format_error
def read_altimetry(filepath: Union[str, Path], verbose: bool = False) -> Tuple[np.ndarray, str]:
    """
    Read altimetry data from BELLHOP .ati file.

    Reads 2D range-depth altimetry (surface) profile with optional
    interpolation type. Extends the altimetry to ±infinity.

    Parameters
    ----------
    filepath : str or Path
        Path to altimetry file. An existing path is read exactly as
        given, whatever its extension; a bare root without an extension
        resolves to ``<root>.ati``.
    verbose : bool, optional
        If True, print altimetry information. Default is False.

    Returns
    -------
    ati : ndarray
        Altimetry data array of shape (n_cols, N+2) where:
        - ati[0, :] = range in meters (extended to ±1e50 at endpoints)
        - ati[1, :] = surface position in meters, **positive-down** on
          Bellhop's z axis exactly as the file carries it (+2 means the
          surface is 2 m *below* MSL — a wave trough). This is the same
          convention :func:`write_ati_file` documents on its input, so the
          pair round-trips as an identity; callers that own the public
          positive-up convention (``Environment(altimetry=…)``, whose
          heights ``bellhop_writer`` negates on the way out) must negate
          this column on the way back in.

        ``ReadATI`` accepts the same long format as ``ReadBTY``
        (``bdryMod.f90:80-110``), so a ``TYPE(2:2) == 'L'`` file carries
        per-range top geoacoustics — an ice cover, typically — in rows 2..6
        with the column order documented on :func:`read_bathymetry`.
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
    """
    return _read_boundary_2d(filepath, ".ati", "altimetry", verbose)


@typed_format_error
def read_boundary_3d(
    filepath: Union[str, Path], verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Read a BELLHOP3D 3-D boundary grid (``.bty`` bathymetry / ``.ati``
    altimetry).

    **Retained for planned 3-D support — this is not dead code.** No uacpy
    model runs ``bellhop3d`` yet, so nothing in the 2-D public API reaches
    this reader; it is the parser a 3-D implementer builds on, alongside
    :func:`write_bty_3d`, :func:`~uacpy.io.oalib_reader.read_ssp_3d`,
    :func:`~uacpy.io.oalib_reader.read_flp3d` and
    :func:`~uacpy.io.oalib_writer.write_field3dflp`. The 2-D readers refuse
    3-D files by name and point here.

    Parameters
    ----------
    filepath : str or Path
        Path to the boundary file, extension included (``.bty`` bathymetry,
        ``.ati`` altimetry — one format on two units, ``ReadBTY3D`` at
        ``bdry3DMod.f90:216`` and ``ReadATI3D`` at ``:54``). The path is
        used as given; no suffix is guessed, because the same grid is
        legal under either name.
    verbose : bool, optional
        Log the grid size and axis previews. Default False.

    Returns
    -------
    x_bot : ndarray
        X coordinates in **metres**, shape ``(n_x,)``. The file stores km
        and the engine scales by 1000 at ``bdry3DMod.f90:290``; this reader
        applies the same conversion so the returned axis follows uacpy's
        metres-unless-suffixed rule.
    y_bot : ndarray
        Y coordinates in **metres**, shape ``(n_y,)``. Same km->m conversion
        (``bdry3DMod.f90:291``).
    z_bot : ndarray
        Depths in metres, shape ``(n_y, n_x)``, positive downward.
        ``z_bot[iy, ix]`` is the depth at ``(x_bot[ix], y_bot[iy])`` — the
        file's own row order, one row of ``n_x`` depths per y
        (``bdry3DMod.f90:300-301``, ``DO iy … READ Bot( :, iy )``).
    n_x, n_y : int
        Grid sizes along x and y.

    Raises
    ------
    ~uacpy.core.exceptions.ConfigurationError
        The file does not exist. A 3-D boundary grid is a deck the *user*
        authors, so an absent one is a bad argument rather than a failed run.
    ~uacpy.core.exceptions.FileFormatError
        The TYPE record is not one the engine accepts, a declared count is
        non-positive, or the file ends inside the grid.

    Notes
    -----
    File layout (``bdry3DMod.f90:239-301``):

    1. TYPE, quoted. Position 1 is the interpolation — ``'R'`` regular grid
       or ``'C'`` curvilinear (``:241-247``); position 2 is the format flag,
       ``'S'``/absent short or ``'L'`` long (``:251-258``). **The 3-D
       interpolation codes are not the 2-D ones**: ``read_bathymetry``
       accepts ``'L'``/``'C'``, and ``'L'`` in position 1 here would be a
       different thing entirely.
    2. ``n_x``, then the x axis in km. The axis is one list-directed READ
       followed by ``SubTab`` (``:270-271``), so the AT shorthand
       ``first last /`` expands to ``n_x`` evenly spaced values — which is
       why this goes through :func:`~uacpy.io._fortran_helpers.read_vector`
       rather than a plain value read.
    3. ``n_y``, then the y axis in km, the same way (``:285-286``).
    4. ``n_y`` rows of ``n_x`` depths in metres, each row one whole-vector
       list-directed READ (``:300-301``) that may wrap across lines.

    The engine only *warns* about a NaN in the depth grid
    (``bdry3DMod.f90:310-312``) and then propagates it into every boundary
    tangent and normal, so this reader warns as well rather than refusing a
    file ``bellhop3d`` would run.

    A non-monotonic axis aborts the engine (``bdry3DMod.f90:324,328``); that
    is enforced on the writing side by :func:`write_bty_3d`, which is where a
    uacpy caller can still do something about it.

    Examples
    --------
    >>> import tempfile, os
    >>> import numpy as np
    >>> deck = ["'R'", '3', '0.0 1.0 2.0', '2', '0.0 1.0',
    ...         '100.0 90.0 100.0', '110.0 95.0 110.0']
    >>> with tempfile.TemporaryDirectory() as d:
    ...     path = os.path.join(d, 'seamount.bty')
    ...     with open(path, 'w') as fh:
    ...         for row in deck:
    ...             print(row, file=fh)
    ...     x, y, z, nx, ny = read_boundary_3d(path)
    >>> x
    array([   0., 1000., 2000.])
    >>> z.shape == (ny, nx)
    True
    >>> float(z[1, 1])
    95.0
    """
    filepath = Path(filepath)
    # Same provenance split as the 2-D reader: a boundary deck is authored by
    # the user, not written by a uacpy model, so an absent one is a
    # ConfigurationError (see FileFormatError's own docstring for the rule).
    if not filepath.exists():
        raise ConfigurationError(
            f"3-D boundary file not found: {filepath}",
            remediation="Pass the path with its extension (.bty bathymetry "
                        "or .ati altimetry); this reader guesses neither, "
                        "because the same 3-D grid is legal under both.",
        )

    with open(filepath, "r") as fid:
        log_message('bathy_io', "Reading 3-D boundary file", verbose=verbose)

        type_str = strip_fortran_quotes(fid.readline())
        interp_type = type_str[:1]
        format_char = type_str[1:2].strip()
        # Case-sensitive, because the Fortran SELECT CASE is: a lowercase
        # letter that parsed here would still ERROUT the binary.
        if interp_type not in ("R", "C"):
            raise FileFormatError(
                f"{filepath}: unknown 3-D boundary type {interp_type!r}; "
                f"BELLHOP3D accepts 'R' (regular grid) or 'C' (curvilinear) "
                f"and ERROUTs on anything else (bdry3DMod.f90:241-247). Note "
                f"these are not the 2-D codes — read_bathymetry's TYPE(1:1) "
                f"is 'L' or 'C'.",
                remediation="Fix the TYPE record, or read a 2-D boundary "
                            "with read_bathymetry / read_altimetry.",
            )
        if format_char not in ("", "S"):
            raise FileFormatError(
                f"{filepath}: TYPE(2:2) = {format_char!r} selects the long "
                f"3-D format, whose per-row bottom-province blocks and "
                f"province table (bdry3DMod.f90:330-350) this reader does "
                f"not parse; only the short format ('S' or absent) is read.",
                remediation="Re-export the grid in the short format, or "
                            "extend read_boundary_3d with the province "
                            "block when 3-D support lands.",
            )

        approx = "Regular-grid" if interp_type == "R" else "Curvilinear"
        log_message('bathy_io',
                    f"{approx} approximation to 3-D boundary",
                    verbose=verbose)

        # ReadVector semantics: count record, then a list-directed value
        # record expanded by SubTab (bdry3DMod.f90:270-271, :285-286).
        x_bot, n_x = read_vector(fid)
        if n_x <= 0:
            raise FileFormatError(
                f"{filepath}: the deck declares {n_x} boundary points in x; "
                f"at least one is required (bdry3DMod.f90:261-265 allocates "
                f"MAX(n_x, 3) and reads n_x of them).",
                remediation="Check the count line above the x axis — a "
                            "misaligned deck reads the wrong record as it.",
            )
        log_message('bathy_io', f"Number of boundary points in x = {n_x}",
                    verbose=verbose)
        log_message('bathy_io', f"x (km): {_summarize_axis(x_bot)}",
                    verbose=verbose, level='debug')

        y_bot, n_y = read_vector(fid)
        if n_y <= 0:
            raise FileFormatError(
                f"{filepath}: the deck declares {n_y} boundary points in y; "
                f"at least one is required (bdry3DMod.f90:276-280).",
                remediation="Check the count line above the y axis — a "
                            "misaligned deck reads the wrong record as it.",
            )
        log_message('bathy_io', f"Number of boundary points in y = {n_y}",
                    verbose=verbose)
        log_message('bathy_io', f"y (km): {_summarize_axis(y_bot)}",
                    verbose=verbose, level='debug')

        # One whole-vector READ per y row (bdry3DMod.f90:300-301). Reading row
        # by row rather than flattening the rest of the file is what makes a
        # truncated grid name the row it died on, and what lets a row wrap
        # across lines the way the Fortran allows.
        z_bot = np.array([
            read_list_directed_values(
                fid, n_x,
                f"3-D depth row {iy + 1} of {n_y} ({n_x} values)", filepath)
            for iy in range(n_y)
        ])

    if not np.all(np.isfinite(z_bot)):
        warnings.warn(
            f"read_boundary_3d: {filepath} contains "
            f"{np.count_nonzero(~np.isfinite(z_bot))} non-finite depth(s) of "
            f"{z_bot.size}; bellhop3d only warns about these "
            f"(bdry3DMod.f90:310-312) and then carries them into every "
            f"boundary tangent and normal derived from the grid.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP,
        )

    return km_to_m(x_bot), km_to_m(y_bot), z_bot, n_x, n_y


def _validate_interp_type(interp_type: str) -> str:
    """Return a single-character interpolation code, or raise
    :class:`~uacpy.core.exceptions.ConfigurationError`.

    Only the first character of TYPE ('L' piecewise linear or 'C'
    curvilinear) is user-selectable; the second character (format flag)
    is chosen by the writer depending on the number of columns it emits
    (see ``write_bty_file`` vs ``write_bty_long_format``).
    """
    t = str(interp_type).strip().upper()
    if t not in ("L", "C"):
        raise ConfigurationError(
            f"Invalid interpolation type {interp_type!r}; expected 'L' or 'C'."
        )
    return t


def _validate_boundary_axis(label: str, axis: np.ndarray) -> None:
    """Reject a boundary axis the engines cannot interpolate on.

    Bellhop brackets a coordinate by searching the axis in ascending order
    and interpolates within the bracketing segment
    (``bdryMod.f90 GetTopSeg/GetBotSeg``, :256-282); a repeated value makes
    a zero-length segment whose tangent normalisation divides by zero, and
    a decreasing value breaks the segment search silently. Non-finite
    values pass straight into the geometry and poison every tangent/normal
    derived from it.
    """
    axis = np.asarray(axis, dtype=float)
    # An empty axis is not a boundary. `write_bty_file(path, np.zeros((0, 2)))`
    # otherwise emitted a well-formed file declaring 0 points, which uacpy's
    # own `read_bathymetry` rejects but `bellhop.exe` accepts: it prints
    # "Number of bathymetry points = 0", terminates every beam, and writes an
    # all-zero .shd at exit 0. One point is legal — bdryMod.f90:174,224-225
    # extends a boundary to +/- infinity itself — so only the empty case is
    # refused here.
    if axis.size == 0:
        raise ConfigurationError(
            f"{label}: no points. A boundary file declaring 0 points is read "
            f"by bellhop.exe as a boundary that terminates every ray at step "
            f"1, returning an all-zero field at exit 0.",
            remediation="Pass at least one (range, value) pair; a single pair "
                        "is extended to +/- infinity by the engine.",
        )
    if not np.all(np.isfinite(axis)):
        raise ConfigurationError(
            f"{label}: contains non-finite values "
            f"({np.count_nonzero(~np.isfinite(axis))} of {axis.size}); the "
            f"engines interpolate the boundary from these coordinates and a "
            f"NaN/Inf poisons the segment tangents."
        )
    if axis.size > 1 and np.diff(axis).min() <= 0.0:
        i = int(np.argmin(np.diff(axis)))
        raise ConfigurationError(
            f"{label}: values must be strictly increasing, but "
            f"index {i} -> {i + 1} goes {axis[i]:g} -> {axis[i + 1]:g}. The "
            f"engines' segment search assumes an ascending axis and a "
            f"repeated value creates a zero-length segment.",
            remediation="Sort the axis and remove duplicate coordinates "
                        "before writing.",
        )


def _check_km_column_resolves(label: str, ranges_km, *,
                              engine_ref: str = "bdryMod.f90:230") -> None:
    """Refuse ranges that collide once written to the km column.

    The axis is validated in METRES but written in km at ``%.6f``, so the
    file's own resolution is 1e-6 km = 1 mm: two metre-domain ranges closer
    than that pass :func:`_validate_boundary_axis` and then land on the same
    token. Measured, ``[[0,200],[5000.0,190],[5000.0004,180],[10000,200]]``
    emits two ``5.000000`` rows and ``bellhop.exe`` stops with "Bathymetry
    ranges are not monotonically increasing" (``bdryMod.f90:230`` ->
    ``monotonicMod.f90:32``) — at exit 0, with no ``.shd``, because
    ``misc/FatalError.f90:30`` is ``STOP '<string>'``. Same guard as
    :func:`refl_io.write_source_beam_pattern`'s angle-resolution check.

    Tests what the file will hold rather than the axis handed in, so it cannot
    disagree with the format string it is protecting.

    ``engine_ref`` names the READ that aborts, because the caller knows which
    one it is: the 2-D writers reach ``bdryMod.f90:230`` and
    :func:`write_bty_3d` reaches ``bdry3DMod.f90:324,328``.
    """
    written = np.round(np.asarray(ranges_km, dtype=float), 6)
    i = _collapsed_pair_index(written)
    if i is not None:
        raise ConfigurationError(
            f"{label}: ranges {km_to_m(written[i]):.6g} m and "
            f"{km_to_m(written[i + 1]):.6g} m both write {written[i]:.6f} km, "
            f"so the file's 1 mm resolution cannot separate them. Bellhop"
            f" then "
            f"aborts on a non-increasing range axis ({engine_ref}) at exit "
            f"0, leaving no output.",
            remediation="Separate the ranges by at least 1 mm, or drop the "
                        "duplicate.",
        )


def _write_boundary_2d(
    filepath: Union[str, Path], data: np.ndarray, interp_type: str,
) -> None:
    """Emit the short (2-column) ATI/BTY record: TYPE, count, ``(range_km,
    value_m)`` rows.

    ``.ati`` and ``.bty`` are the same writer on two units — the counterpart
    of :func:`_read_boundary_2d`. ``data`` is an ``(N, 2)`` array of
    ``(range_m, value_m)`` or any carrier exposing ``to_pairs()``; ranges are
    converted to km for the file.

    ``bdryMod.f90:98`` (.ati) and ``:195`` (.bty) read the rows list-directed,
    so the field widths do not matter; the 6 decimals do — on a km axis they
    are what keeps a metre-domain range exact to the millimetre.
    """
    filepath = Path(filepath)
    # One character, not ``<interp>S``. ``atiType``/``btyType`` are
    # ``CHARACTER(LEN=2)`` (``bdryMod.f90:17``) and Bellhop tests them whole —
    # ``IF ( atiType == 'C' )`` at ``bellhop.f90:535`` and ``btyType == 'C'``
    # at ``:552`` — so Fortran blank-pads the literal to ``'C '`` and ``'CS'``
    # never matches. The node normals ARE built curvilinearly, because
    # ``bdryMod.f90:299`` slices ``(1:1)``, so a ``'CS'`` deck ran a hybrid
    # that is neither curvilinear nor flat: measured 26.59 dB max / 1.28 dB
    # mean against a bare ``'C'``, with both runs exiting 0 and both .prt
    # files printing "Curvilinear Interpolation". The two shipped backends
    # disagreed on it too — bellhopcuda tests ``type[0]`` only
    # (``src/trace.hpp:321``), so it honoured a deck bellhop.exe did not.
    # A bare character still selects the short format: ``bdryMod.f90:179-181``
    # switches on ``btyType(2:2)`` with ``CASE ( 'S', '' )``, and AT's own
    # decks write one character (``tests/ParaBot/ParaBot.bty`` is ``'C'``).
    type_str = _validate_interp_type(interp_type)

    if hasattr(data, "to_pairs"):
        data = data.to_pairs()
    rows = np.asarray(data, dtype=float).copy()
    _validate_boundary_axis(f"{filepath.suffix or '.bty/.ati'} range column",
                            rows[:, 0])
    rows[:, 0] = m_to_km(rows[:, 0])
    _check_km_column_resolves(
        f"{filepath.suffix or '.bty/.ati'} range column", rows[:, 0])

    with open(filepath, "w") as f:
        f.write(f"'{type_str}'\n")
        f.write(f"{rows.shape[0]}\n")
        for r, value in rows:
            f.write(f"{r:.6f} {value:.6f}\n")
        f.write("\n")


def write_bty_file(filepath: Union[str, Path], bathymetry: np.ndarray, interp_type: str = "L") -> None:
    """
    Write bathymetry file (short format, bathymetry only).

    Parameters
    ----------
    filepath : str or Path
        Bathymetry file path (typically .bty extension)
    bathymetry : ndarray
        Bathymetry data, shape (N, 2):
        - Column 0: Range in meters
        - Column 1: Depth in meters
    interp_type : str, optional
        Interpolation type (single character, TYPE(1:1)):
        - 'C': Curvilinear (with tangents/normals)
        - 'L': Linear interpolation (default)

    Notes
    -----
    File format (see third_party/Acoustics-Toolbox/doc/ATI_BTY_File.htm):
    - Line 1: 2-character TYPE in quotes — position 1 is interpolation
      ('L' or 'C'), position 2 is format ('S' short, bathymetry only)
    - Line 2: Number of points
    - Following lines: range (km), depth (m) pairs

    This writer always emits the short (2-column) format and writes TYPE
    as the bare interpolation character: Bellhop tests the whole
    CHARACTER(LEN=2) ``btyType`` (``bellhop.f90:552``), so ``'CS'`` never
    matches while ``'C'`` blank-pads to ``'C '`` and does, and the empty
    second character still selects the short format
    (``bdryMod.f90:179-181``, ``CASE ('S', '')``).

    The function automatically converts ranges from meters to kilometers
    for the output file.
    """
    _write_boundary_2d(filepath, bathymetry, interp_type)


def write_bty_long_format(
    filepath: Union[str, Path],
    bathymetry: np.ndarray,
    bottom_rd,
    interp_type: str = "L",
) -> None:
    """
    Write long-format bathymetry file (.bty) with per-segment geoacoustics.

    Unlike the 2-column ``write_bty_file`` output, the long format adds
    bottom compressional sound speed, density, attenuation and shear speed
    per range so Bellhop can use range-dependent bottom properties
    (``BOTY(:)`` handling in ReadEnvironmentBell.f90).

    Parameters
    ----------
    filepath : str or Path
        Output .bty path.
    bathymetry : ndarray
        Shape (N, 2): range (m), depth (m).
    bottom_rd : Bottom
        Range-dependent ``Bottom`` carrying per-range halfspace geoacoustics:
        ``ranges`` (metres) plus the ``halfspace_sound_speed`` /
        ``halfspace_density`` / ``halfspace_attenuation`` /
        ``halfspace_shear_speed`` / ``halfspace_shear_attenuation`` views.
    interp_type : str, optional
        'L' (linear, default) or 'C' (curvilinear).

    Notes
    -----
    File format (extended BTY — long format), matching the AT Fortran
    READ in ``Bellhop/bdryMod.f90:200-201``:
    ``READ(BTYFile,*) Bot(ii)%x, %HS%alphaR, %HS%betaR, %HS%rho,
    %HS%alphaI, %HS%betaI``  — i.e. 7 numbers per row.

    - Line 1: 2-character TYPE in quotes — position 1 is interpolation
      ('L' or 'C'), position 2 is 'L' (long format, bathymetry +
      geoacoustics). See ATI_BTY_File.htm.
    - Line 2: number of points
    - Following lines:
      ``range_km depth cp_m_s cs_m_s rho_g_cm3 alpha_p alpha_s``

    Rows are emitted on the **union** of the bathymetry and bottom range
    grids (depth and geoacoustics each interpolated onto it), so property
    breaks between bathymetry points survive rather than being blended away.
    """
    filepath = Path(filepath)
    interp_char = _validate_interp_type(interp_type)
    type_str = f"{interp_char}L"
    if interp_char == 'C':
        # The long format has no curvilinear spelling in this AT version.
        # Position 2 must be 'L' to select the geoacoustic columns
        # (``bdryMod.f90:183``), but Bellhop's curvilinear test compares the
        # whole LEN=2 string against 'C' (``bellhop.f90:552``), which 'CL'
        # cannot equal. So the reflection geometry is flat whatever is asked
        # for here, while the node normals are still built curvilinearly
        # (``bdryMod.f90:299`` slices ``(1:1)``) — measured 26.98 dB max /
        # 1.39 dB mean between the two shipped backends on such a deck.
        warnings.warn(
            "write_bty_long_format: interp_type='C' cannot be honoured in the "
            "long format. Bellhop selects curvilinear reflection on "
            "btyType == 'C' (bellhop.f90:552) but the long format needs 'L' "
            "in the second character, and 'CL' never matches, so the run uses "
            "flat segment normals. Use the short format (write_bty_file) for "
            "a genuinely curvilinear bottom, or pass interp_type='L' here to "
            "ask for what the deck will actually do.",
            UserWarning, skip_file_prefixes=USER_FRAME_SKIP)

    if hasattr(bathymetry, "to_pairs"):
        bathymetry = bathymetry.to_pairs()
    bathy_km = np.asarray(bathymetry, dtype=float).copy()
    _validate_boundary_axis(".bty (long) range column", bathy_km[:, 0])
    bathy_km[:, 0] = m_to_km(bathy_km[:, 0])
    _check_km_column_resolves('.bty (long) range column', bathy_km[:, 0])

    rd_r_km = m_to_km(bottom_rd.ranges)
    r_km = np.union1d(bathy_km[:, 0], np.asarray(rd_r_km, dtype=float))
    # Check the axis that is WRITTEN, not just the bathymetry axis checked
    # above. Each carrier enforces its own 1 mm minimum step within itself
    # (Bottom.__post_init__, _grid.py), but nothing enforces one ACROSS them,
    # and np.union1d de-dupes by exact float equality — so the same physical
    # range arrived at by different arithmetic survives twice and prints one
    # token. Measured with bathymetry ranges np.cumsum(np.full(16, 333.3))
    # against bottom ranges np.arange(1, 17)*333.3 — the same 16 ranges,
    # differing by at most 9.1e-13 m — the union is 23 rows carrying 7
    # duplicated '%.6f' tokens, after which bdryMod.f90:230 ->
    # monotonicMod.f90:32 aborts with "not monotonically increasing" and
    # FatalError.f90 STOPs at exit 0, leaving no .shd.
    _check_km_column_resolves('.bty (long) merged range axis', r_km)
    depth = np.interp(r_km, bathy_km[:, 0], bathy_km[:, 1])
    cp = np.interp(r_km, rd_r_km, bottom_rd.halfspace_sound_speed)
    rho = np.interp(r_km, rd_r_km, bottom_rd.halfspace_density)
    alpha = np.interp(r_km, rd_r_km, bottom_rd.halfspace_attenuation)
    cs = np.interp(r_km, rd_r_km, bottom_rd.halfspace_shear_speed)
    alpha_s = np.interp(r_km, rd_r_km,
                        bottom_rd.halfspace_shear_attenuation)
    n_pts = r_km.size

    with open(filepath, "w") as f:
        f.write(f"'{type_str}'\n")
        f.write(f"{n_pts}\n")
        for i in range(n_pts):
            # Column order matches bdryMod.f90:200-201 (range_km depth cp cs rho alpha_p alpha_s).
            # The read there is list-directed, so six decimals on every column
            # is a free choice — and it is the one the .env writers make for
            # the same halfspace, so a range-dependent bottom and a
            # range-independent one built from the same BoundaryProperties
            # describe the same seabed rather than two rounded off differently.
            f.write(
                f"{r_km[i]:.6f} {depth[i]:.6f} "
                f"{cp[i]:.6f} {cs[i]:.6f} {rho[i]:.6f} "
                f"{alpha[i]:.6f} {alpha_s[i]:.6f}\n"
            )
        f.write("\n")


def write_ati_file(filepath: Union[str, Path], altimetry: np.ndarray, interp_type: str = "L") -> None:
    """
    Write altimetry (surface) file for acoustic models.

    Parameters
    ----------
    filepath : str or Path
        Altimetry file path (typically .ati extension)
    altimetry : ndarray
        Altimetry data, shape (N, 2):
        - Column 0: Range in meters
        - Column 1: Surface position in meters, **positive-down** in
          Bellhop's z-axis convention (i.e. +2 means the surface is 2 m
          *below* MSL — a wave trough). Wrappers that own the
          public positive-up convention (``Environment(altimetry=…)``)
          must negate column 1 before calling this writer.
    interp_type : str, optional
        Interpolation type (single character, TYPE(1:1)):
        - 'C': Curvilinear (with tangents/normals)
        - 'L': Linear interpolation (default)

    Notes
    -----
    File format identical to bathymetry (.bty) files. This writer emits
    only the 2-column short format and writes TYPE as the bare
    interpolation character: Bellhop tests the whole CHARACTER(LEN=2)
    ``atiType`` (``bellhop.f90:535``), so ``'CS'`` never matches while
    ``'C'`` blank-pads to ``'C '`` and does. ``ReadATI`` itself also accepts
    the long format (``bdryMod.f90:80-110``) carrying per-range *top*
    geoacoustics — an ice cover — which this writer does not produce.

    Altimetry describes surface variations (ice keels, surface waves, etc.)
    """
    _write_boundary_2d(filepath, altimetry, interp_type)


def write_bty_3d(filepath: Union[str, Path], X: np.ndarray, Y: np.ndarray,
                 depth: np.ndarray, interp_type: str = "R") -> None:
    """
    Write a BELLHOP3D 3-D bathymetry grid.

    **Retained for planned 3-D support — this is not dead code.** Nothing in
    the 2-D public API writes a 3-D deck; this is the writer a 3-D
    implementer builds on, and the round-trip partner of
    :func:`read_boundary_3d`.

    Parameters
    ----------
    filepath : str or Path
        Output path, written exactly as given (typically ``.bty``).
    X : ndarray
        X coordinates in **metres**, shape ``(nx,)``, strictly increasing.
        Converted to the km the file format holds.
    Y : ndarray
        Y coordinates in **metres**, shape ``(ny,)``, strictly increasing.
        Same m->km conversion as ``X``.
    depth : ndarray
        Depths in metres, shape ``(ny, nx)``, positive downward.
        ``depth[iy, ix]`` is the depth at ``(X[ix], Y[iy])``.
    interp_type : str, optional
        ``'R'`` regular grid (default) or ``'C'`` curvilinear — TYPE(1:1) as
        BELLHOP3D reads it (``bdry3DMod.f90:241-247``). **Not the 2-D
        codes**: :func:`write_bty_file` takes ``'L'``/``'C'``.

    Raises
    ------
    ~uacpy.core.exceptions.ConfigurationError
        ``interp_type`` is not ``'R'``/``'C'``; an axis is empty, non-finite
        or not strictly increasing; an axis loses its ordering in the km
        column; ``depth`` does not have shape ``(ny, nx)``; or ``depth``
        holds a non-finite value.

    Notes
    -----
    File layout, matching :func:`read_boundary_3d`:

    - Line 1: TYPE in quotes.
    - Line 2: ``nx``; line 3: the x axis in km.
    - Line 4: ``ny``; line 5: the y axis in km.
    - Then ``ny`` lines of ``nx`` depths in metres.

    Three guards this writer applies that the file format itself does not,
    each because the engine's own failure is silent or destructive:

    * **Strictly increasing axes.** ``bdry3DMod.f90:324,328`` ERROUTs on a
      non-monotonic x or y axis, and ``misc/FatalError.f90:30`` is
      ``STOP '<string>'`` — exit 0, no output. Enforced through the same
      :func:`_validate_boundary_axis` the 2-D writers use.
    * **Axes that survive the km column.** The axes are validated in metres
      and written at ``%.6f`` km, i.e. 1 mm resolution, so two coordinates
      closer than that pass the axis check and then collide on disk into the
      non-monotonic abort above.
    * **Finite depths.** The engine warns about a NaN
      (``bdry3DMod.f90:310-312``) and keeps going, so it reaches every
      tangent and normal computed from the grid. Refused here rather than
      silently rewritten: substituting 0.0 for a NaN would hand the engine a
      sea surface where the caller had a gap, which is a different seabed
      run at exit 0 with nothing to show it happened.

    See Also
    --------
    read_boundary_3d : Read the grid back.
    write_bty_file : The 2-D short-format writer.

    Examples
    --------
    >>> import tempfile, os
    >>> import numpy as np
    >>> with tempfile.TemporaryDirectory() as d:
    ...     path = os.path.join(d, 'seamount.bty')
    ...     write_bty_3d(path, np.array([0.0, 1000.0, 2000.0]),
    ...                  np.array([0.0, 1000.0]),
    ...                  np.array([[100.0, 90.0, 100.0],
    ...                            [110.0, 95.0, 110.0]]))
    ...     x, y, z, nx, ny = read_boundary_3d(path)
    >>> (nx, ny)
    (3, 2)
    >>> float(z[1, 1])
    95.0
    """
    filepath = Path(filepath)

    type_str = str(interp_type).strip().upper()
    if type_str not in ("R", "C"):
        raise ConfigurationError(
            f"write_bty_3d(interp_type={interp_type!r}) is not a 3-D "
            f"interpolation code; BELLHOP3D accepts 'R' (regular grid) or "
            f"'C' (curvilinear) and ERROUTs on anything else "
            f"(bdry3DMod.f90:241-247).",
            remediation="Use 'R' or 'C'. The 2-D 'L'/'C' pair belongs to "
                        "write_bty_file, not here.",
        )

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    _validate_boundary_axis("3-D .bty x axis", X)
    _validate_boundary_axis("3-D .bty y axis", Y)

    nx = X.size
    ny = Y.size
    depth = np.asarray(depth, dtype=float)
    if depth.shape != (ny, nx):
        raise ConfigurationError(
            f"write_bty_3d: depth has shape {depth.shape}, but the axes "
            f"describe a (ny={ny}, nx={nx}) grid. The file holds one row of "
            f"nx depths per y (bdry3DMod.f90:300-301), so the row axis is y.",
            remediation="Pass depth[iy, ix] = depth at (X[ix], Y[iy]); "
                        "transpose an (nx, ny) array before calling.",
        )
    if not np.all(np.isfinite(depth)):
        raise ConfigurationError(
            f"write_bty_3d: depth contains "
            f"{np.count_nonzero(~np.isfinite(depth))} non-finite value(s) of "
            f"{depth.size}. bellhop3d only warns about a NaN in the grid "
            f"(bdry3DMod.f90:310-312) and then propagates it through every "
            f"boundary tangent and normal.",
            remediation="Interpolate or mask the gaps before writing; this "
                        "writer will not substitute a depth you did not "
                        "choose.",
        )

    X_km = m_to_km(X)
    Y_km = m_to_km(Y)
    _check_km_column_resolves("3-D .bty x axis", X_km,
                              engine_ref="bdry3DMod.f90:324")
    _check_km_column_resolves("3-D .bty y axis", Y_km,
                              engine_ref="bdry3DMod.f90:328")

    with open(filepath, "w") as f:
        # One character, like the 2-D writers: btyType is CHARACTER(LEN=2)
        # and TYPE(2:2) is tested with CASE ( 'S', '' ) at
        # bdry3DMod.f90:251-258, so a bare 'R' selects the short format.
        f.write(f"'{type_str}'\n")

        f.write(f"{nx}\n")
        f.write(" ".join(f"{x:.6f}" for x in X_km) + "\n")

        f.write(f"{ny}\n")
        f.write(" ".join(f"{y:.6f}" for y in Y_km) + "\n")

        # One record per y row, in the order bdry3DMod.f90:300-301 reads them.
        for iy in range(ny):
            f.write(" ".join(f"{z:.6f}" for z in depth[iy, :]) + "\n")
