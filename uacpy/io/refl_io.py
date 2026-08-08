"""Reflection-coefficient + source-beam-pattern auxiliary file I/O.

* ``.brc`` / ``.trc`` — precomputed plane-wave reflection coefficients as
  ``(angle_deg, |R|, phase_deg)`` rows behind a count line
  (:func:`read_reflection_coefficient`, :func:`write_reflection_coefficient`)

``.irc`` is a different format and is **not** handled here: BOUNCE writes it
as a title/frequency line, a count, then fixed-format ``(5G15.7, I5)`` rows of
``x, Re f, Im f, Re g, Im g, iPower`` (``bounce.f90:225-228``), read back with
the same fixed format by ``misc/RefCoef.f90:97-107``. It is consumed only by
the Kraken-family ``BotOpt='P'`` path, which passes it straight to the Fortran.
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
from uacpy.io.units import deg_to_rad, rad_to_deg
from uacpy.core.exceptions import ConfigurationError, FileFormatError
from uacpy.io._fortran_helpers import _bound_counts
from uacpy.io._fortran_helpers import typed_format_error


@typed_format_error
def read_reflection_coefficient(
    filename: Union[str, Path], boundary: str = "bottom"
) -> Dict[str, np.ndarray]:
    """
    Read reflection coefficient data from file (.trc or .brc).

    Reads angle-dependent reflection coefficient data used by BELLHOP
    for top or bottom boundary conditions. Data includes angle, magnitude,
    and phase.

    Parameters
    ----------
    filename : str or Path
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
    """
    # Use the path as given when it already exists (so a file written by
    # write_reflection_coefficient with a literal path round-trips); otherwise
    # apply the AT extension (.trc top / .brc bottom) so a bare base name still
    # resolves to the conventional file.
    filename = Path(filename)
    if not filename.exists():
        ext = ".trc" if boundary.lower() == "top" else ".brc"
        if filename.suffix != ext:
            filename = filename.with_name(filename.name + ext)

    try:
        with open(filename, "r") as fid:
            # Read number of points
            n_pts = int(fid.readline().strip())

            if n_pts == 0:
                return {
                    "theta": np.array([]),
                    "R": np.array([]),
                    "phi": np.array([]),
                    "n_pts": 0,
                }

            # Each point is one ASCII line of at least three numbers, so it
            # cannot occupy fewer than 6 bytes on disk; a count beyond that is
            # a malformed header, not a huge table.
            _bound_counts(filename, Path(filename).stat().st_size, 6,
                          n_pts=n_pts)
            # Pre-allocate arrays
            theta = np.zeros(n_pts)
            R = np.zeros(n_pts)
            phi = np.zeros(n_pts)

            for i in range(n_pts):
                line = fid.readline().strip()
                values = line.split()
                if len(values) < 3:
                    raise FileFormatError(
                        f"Reflection coefficient file {filename}: "
                        f"line {i + 2} has fewer than 3 tokens "
                        f"({line!r}); expected 'theta magnitude phase'."
                    )
                theta[i] = float(values[0])
                R[i] = float(values[1])
                phi[i] = float(values[2])

            phi = deg_to_rad(phi)

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


def read_source_beam_pattern(
    filepath: Union[str, Path],
    sbp_option: str = "O",
    verbose: bool = False,
) -> np.ndarray:
    """
    Read source beam pattern from file.

    Parameters
    ----------
    filepath : str or Path
        Source beam pattern file; a root name without the ``.sbp`` extension
        also resolves.
    sbp_option : str, optional
        Option flag:
        - '*': Read beam pattern from file
        - 'O': Create omni-directional pattern (default)

    Returns
    -------
    beam_pattern : ndarray
        Beam pattern array, shape (N, 2):
        - Column 0: Angles in degrees
        - Column 1: level in dB re peak

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

    For an omni-directional pattern, creates [-180°, 180°] at 0 dB.

    Translated from OALIB readpat.m
    """
    if sbp_option == "*":
        log_message('refl_io', "Reading source beam pattern file",
                    verbose=verbose)

        # Take the path as given when it already exists; otherwise apply the
        # AT extension so a bare base name still resolves (mirrors
        # read_reflection_coefficient).
        sbp_file = Path(filepath)
        if not sbp_file.exists():
            sbp_file = sbp_file.with_name(sbp_file.name + ".sbp")
        if not sbp_file.exists():
            raise ConfigurationError(
                f"Source beam pattern file not found: {sbp_file}. "
                "Provide the .sbp file next to the env, or pass "
                "sbp_option='O' for an omni-directional source."
            )
        with open(sbp_file, "r") as fid:
            line = fid.readline()
            NSBPPts = int(line.strip())
            log_message('refl_io',
                        f"Number of source beam pattern points = {NSBPPts}",
                        verbose=verbose)

            _bound_counts(sbp_file, Path(sbp_file).stat().st_size, 4,
                          NSBPPts=NSBPPts)
            beam_pattern = np.zeros((NSBPPts, 2))
            for i in range(NSBPPts):
                line = fid.readline()
                vals = np.array(line.split()[:2], dtype=float)
                beam_pattern[i, :] = vals
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

    else:
        # Omni-directional pattern
        beam_pattern = np.array([[-180.0, 0.0], [180.0, 0.0]])

    return beam_pattern


def write_reflection_coefficient(
    filepath: Union[str, Path],
    angles: np.ndarray,
    coefficients: np.ndarray,
) -> None:
    """
    Write reflection coefficient file for Bellhop bottom/top boundary.

    Parameters
    ----------
    filepath : str or Path
        Output file path (typically .brc for bottom, .trc for top).
    angles : ndarray
        Grazing angles in degrees, shape (N,).
    coefficients : ndarray
        - Complex (N,): ``np.angle`` is taken (radians) and converted
          to degrees on output.
        - Real (N, 2): ``[amplitude, phase_radians]``. Phase column is
          converted to degrees on output.
        - Real (N,): amplitudes only; phase is zero.
    Notes
    -----
    File format (.brc/.trc), per AT ReflectionCoefficientFile.htm:
        Line 1: NTHETA
        Lines 2+: THETA(deg)  RMAG  RPHASE(deg)

    Phase is stored in **degrees** on disk; this writer takes radians
    in to match :func:`read_reflection_coefficient` (which returns
    ``phi`` in radians) and the :class:`uacpy.ReflectionCoefficient`
    result convention.
    """
    filepath = Path(filepath)

    if np.iscomplexobj(coefficients):
        amplitude = np.abs(coefficients)
        phase_rad = np.angle(coefficients)
    elif coefficients.ndim == 2 and coefficients.shape[1] == 2:
        amplitude = coefficients[:, 0]
        phase_rad = coefficients[:, 1]
    else:
        amplitude = coefficients
        phase_rad = np.zeros_like(coefficients)

    phase_deg = rad_to_deg(phase_rad)

    n_angles = len(angles)

    with open(filepath, "w") as f:
        f.write(f"{n_angles}\n")
        for i in range(n_angles):
            f.write(
                f"{angles[i]:8.2f} {amplitude[i]:12.6f} {phase_deg[i]:12.6f}\n"
            )


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
    """
    filepath = Path(filepath)

    n_angles = len(angles)

    with open(filepath, "w") as f:
        # Write number of angles
        f.write(f"{n_angles}\n")

        # Write angle, amplitude pairs
        for i in range(n_angles):
            f.write(f"{angles[i]:8.2f} {pattern[i]:12.6f}\n")


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
    written into — is kept as is.

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
        # SrcBmPat(IBP,1))`` and never checks the denominator: a repeated
        # angle divides by zero and a decreasing one flips the weight, both
        # silently. ``Source`` applies this check to the array form, so the
        # file form has to get it too or the guard depends on which way the
        # caller happened to supply the pattern.
        from uacpy.core._carrier_validate import _require_strictly_increasing
        _require_strictly_increasing(
            read_source_beam_pattern(src, sbp_option='*')[:, 0],
            f"source beam-pattern angles in {src.name}")
        shutil.copy(src, dest)
        return
    arr = np.asarray(pattern, dtype=float)
    write_source_beam_pattern(dest, arr[:, 0], arr[:, 1])


def dedupe_reflection_file(filepath: Union[str, Path]) -> None:
    """Rewrite a ``.brc`` / ``.trc`` file with a strictly-increasing angle axis.

    BOUNCE's Fortran driver tabulates reflection coefficients by sweeping
    phase velocity (kx = omega/c), which — for the c_low/c_high defaults —
    produces many samples that round to the same grazing angle (hundreds of
    duplicate 0-degree rows are typical). Bellhop tolerates non-decreasing
    angles but bellhopcuda enforces strict monotonicity in ``bhc::setup()``
    and aborts with "Bottom reflection coefficients must be monotonically
    increasing".

    This loads the file, keeps only rows whose angle strictly exceeds the
    previous kept row, and rewrites it in the original 3-column BOUNCE
    format (angle_deg, |R|, phase_deg).

    ``.brc`` / ``.trc`` only. An ``.irc`` carries a title/frequency line ahead
    of its count and six fixed-format columns, so running this over one would
    strip the header and four of the six columns; a file whose first line is
    not a bare count is rejected.

    Precision caveat: when two physically distinct phase velocities map to
    grazing angles that round equal at the file's print precision, only the
    first (lowest-c, i.e. shallowest-angle) row of that collision group is
    kept and the rest are dropped — no averaging or re-gridding. This is
    loss-free for true duplicates (the dominant case near 0°) but discards
    one R(θ) sample per genuine collision, a slight under-resolution of the
    reflection table near grazing. Raising the BOUNCE angle/print resolution
    avoids the collisions.
    """
    filepath = Path(filepath)
    with open(filepath, 'r') as fh:
        lines = fh.readlines()

    if not lines:
        return

    try:
        int(lines[0].split()[0])
    except (ValueError, IndexError):
        raise FileFormatError(
            f"{filepath}: expected a .brc/.trc table, whose first line is the "
            f"row count; got {lines[0].strip()!r}.",
            remediation="An .irc has a title/frequency header and six "
                        "fixed-format columns — pass it through unmodified.",
        ) from None

    kept_rows = []  # list of (angle, mag, phase_deg) as strings
    last_angle = -np.inf
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 3:
            continue
        try:
            angle = float(parts[0])
        except ValueError:
            continue
        if angle > last_angle:
            kept_rows.append((parts[0], parts[1], parts[2]))
            last_angle = angle

    # If nothing survived dedup (degenerate case), leave the file alone —
    # downstream reader will surface the real error.
    if not kept_rows:
        return

    # ``misc/RefCoef.f90:119`` states the contract for this table outright —
    # "Assumes phi has been unwrapped so that it varies smoothly" — and
    # ``InterpolateReflectionCoefficient`` then interpolates phi linearly
    # between the bracketing abscissas. BOUNCE writes the principal value, so
    # its own output breaks that: measured on a 10 m sediment over a 1800 m/s
    # half-space at 500 Hz, the table jumps 298.8 deg between adjacent angles.
    # Interpolating across such a step sweeps the phase the long way round and
    # returns a reflection phase that is wrong through the whole interval.
    phases = np.degrees(np.unwrap(
        np.radians([float(p) for _a, _r, p in kept_rows])))

    with open(filepath, 'w') as fh:
        # BOUNCE pads the count with leading whitespace; match that so any
        # downstream tool expecting free-format reads happily.
        fh.write(f"{len(kept_rows):12d}\n")
        for (a, r, _p), phase in zip(kept_rows, phases):
            fh.write(f"   {a}        {r}        {phase:.6f}\n")
