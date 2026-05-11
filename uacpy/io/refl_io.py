"""Reflection-coefficient + source-beam-pattern auxiliary file I/O.

* ``.brc`` / ``.irc`` / ``.trc`` — precomputed reflection coefficients
  (:func:`read_reflection_coefficient`, :func:`write_reflection_coefficient`)
* ``.sbp`` — source beam pattern (:func:`read_source_beam_pattern`,
  :func:`write_source_beam_pattern`)

Bathymetry / altimetry / 3-D boundary files live in
:mod:`uacpy.io.bathy_io`.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Union

from uacpy._log import log_message


def read_reflection_coefficient(
    filename: str, boundary: str = "bottom"
) -> Dict[str, np.ndarray]:
    """
    Read reflection coefficient data from file (.trc or .brc).

    Reads angle-dependent reflection coefficient data used by BELLHOP
    for top or bottom boundary conditions. Data includes angle, magnitude,
    and phase.

    Parameters
    ----------
    filename : str
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
    # Add appropriate extension
    if boundary.lower() == "top":
        if not filename.endswith(".trc"):
            filename = filename + ".trc"
    else:
        if not filename.endswith(".brc"):
            filename = filename + ".brc"

    try:
        with open(filename, "r") as fid:
            # Read number of points
            n_pts = int(fid.readline().strip())

            if n_pts == 0:
                return {
                    "theta": np.array([0.0]),
                    "R": np.array([0.0]),
                    "phi": np.array([0.0]),
                    "n_pts": 0,
                }

            # Pre-allocate arrays
            theta = np.zeros(n_pts)
            R = np.zeros(n_pts)
            phi = np.zeros(n_pts)

            # Read data (angle magnitude phase per line)
            for i in range(n_pts):
                line = fid.readline().strip()
                values = line.split()
                if len(values) >= 3:
                    theta[i] = float(values[0])
                    R[i] = float(values[1])
                    phi[i] = float(values[2])
                else:
                    # Fallback: try old format (one value per line)
                    theta[i] = float(values[0]) if values else 0.0
                    R[i] = float(fid.readline().strip()) if i == 0 else R[i-1]
                    phi[i] = float(fid.readline().strip()) if i == 0 else phi[i-1]

            # Convert phase from degrees to radians
            phi = phi * np.pi / 180.0

            # Validate angles are non-decreasing
            if not np.all(np.diff(theta) >= 0):
                raise ValueError("Angles must be non-decreasing")

            return {"theta": theta, "R": R, "phi": phi, "n_pts": n_pts}

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Reflection coefficient file not found: {filename}. "
            "Run Bounce or OASR first to generate the .brc/.trc file, "
            "or pass an explicit reflection_file= path to the model."
        )


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
        Source beam pattern file root name (without .sbp extension)
    sbp_option : str, optional
        Option flag:
        - '*': Read beam pattern from file
        - 'O': Create omni-directional pattern (default)

    Returns
    -------
    beam_pattern : ndarray
        Beam pattern array, shape (N, 2):
        - Column 0: Angles in degrees
        - Column 1: Power (linear scale, not dB)

    Notes
    -----
    File format (.sbp):
    - Line 1: Number of points
    - Subsequent lines: angle (degrees), power (dB)

    Power values are converted from dB to linear scale on output:
        power_linear = 10^(power_dB / 20)

    For omni-directional pattern, creates [-180°, 180°] with 0 dB (=1.0).

    Translated from OALIB readpat.m
    """
    if sbp_option == "*":
        log_message('refl_io', "Reading source beam pattern file",
                    verbose=verbose)

        filepath = Path(filepath)
        sbp_file = str(filepath) + ".sbp"
        with open(sbp_file, "r") as fid:
            line = fid.readline()
            NSBPPts = int(line.strip())
            log_message('refl_io',
                        f"Number of source beam pattern points = {NSBPPts}",
                        verbose=verbose)

            beam_pattern = np.zeros((NSBPPts, 2))
            for i in range(NSBPPts):
                line = fid.readline()
                vals = np.fromstring(line, sep=" ", count=2)
                beam_pattern[i, :] = vals
            log_message(
                'refl_io',
                f"angle (deg): {beam_pattern[:, 0].tolist()}",
                verbose=verbose, level='debug',
            )
            log_message(
                'refl_io',
                f"power (dB): {beam_pattern[:, 1].tolist()}",
                verbose=verbose, level='debug',
            )

    else:
        # Omni-directional pattern
        beam_pattern = np.array([[-180.0, 0.0], [180.0, 0.0]])

    # Convert dB to linear scale
    beam_pattern[:, 1] = 10.0 ** (beam_pattern[:, 1] / 20.0)

    return beam_pattern


def write_reflection_coefficient(
    filepath: Union[str, Path],
    angles: np.ndarray,
    coefficients: np.ndarray,
    file_type: str = "brc",
) -> None:
    """
    Write reflection coefficient file for Bellhop bottom/top boundary.

    Parameters
    ----------
    filepath : str or Path
        Output file path (typically .brc for bottom, .trc for top)
    angles : ndarray
        Grazing angles in degrees, shape (N,)
    coefficients : ndarray
        Complex reflection coefficients, shape (N,) or (N, 2) for [amplitude, phase]
        If complex, uses magnitude and phase. If real (N, 2), uses directly.
    file_type : str, optional
        File type: 'brc' (bottom) or 'trc' (top). Default is 'brc'.

    Notes
    -----
    File format (.brc/.trc):
    - Line 1: Number of angles
    - Following lines: angle (degrees), |R| (amplitude), phase (degrees)

    The reflection coefficient R = |R| * exp(i*phase)
    """
    filepath = Path(filepath)

    # Convert complex to amplitude/phase if needed
    if np.iscomplexobj(coefficients):
        amplitude = np.abs(coefficients)
        phase = np.angle(coefficients, deg=True)
    elif coefficients.ndim == 2 and coefficients.shape[1] == 2:
        amplitude = coefficients[:, 0]
        phase = coefficients[:, 1]
    else:
        # Assume real values are amplitudes with zero phase
        amplitude = coefficients
        phase = np.zeros_like(coefficients)

    n_angles = len(angles)

    with open(filepath, "w") as f:
        # Write number of angles
        f.write(f"{n_angles}\n")

        # Write angle, amplitude, phase triplets
        for i in range(n_angles):
            f.write(f"{angles[i]:8.2f} {amplitude[i]:12.6f} {phase[i]:12.6f}\n")


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
        (bellhop.f90:132).

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
