"""
Writers for acoustic model environment files
"""

from pathlib import Path
from typing import Union

import numpy as np

from uacpy.core.environment import Environment
from uacpy.core.receiver import Receiver
from uacpy.core.source import Source
from uacpy.io.bty_writer import write_bty_file


# ── Shared AT surface helpers ──────────────────────────────────────────────

_SURFACE_TYPE_MAP = {
    "vacuum": "V", "rigid": "R",
    "halfspace": "A", "half-space": "A",
    "file": "F",
    "grain-size": "G", "grain_size": "G", "grain": "G",
}


def get_top_bc_code(env: Environment) -> str:
    """Return the single-character AT top boundary condition code."""
    return _SURFACE_TYPE_MAP.get(env.surface.acoustic_type.lower(), "V")


def write_surface_halfspace(f, env: Environment) -> None:
    """Write surface halfspace properties line if top BC is 'A'.

    Must be called right after writing the TopOpt line and before SSP data.
    Format: depth  cp  cs  rho  attn_p  attn_s /
    """
    if get_top_bc_code(env) != 'A':
        return
    s = env.surface
    f.write(
        f" 0.00  {s.sound_speed:.2f} {getattr(s, 'shear_speed', 0.0) or 0.0:.1f}"
        f" {s.density:.2f}"
        f" {s.attenuation:.4f} {getattr(s, 'shear_attenuation', 0.0) or 0.0:.4f} /\n"
    )


def write_env_file(
    filepath: Union[str, Path],
    env: Environment,
    source: Source,
    receiver: Receiver,
    run_type: str = "C",
    beam_type: str = "B",
    n_beams: int = 0,
    alpha: tuple = (-80, 80),
    step: float = 0.0,
    z_box: float = None,
    r_box: float = None,
):
    """
    Write Bellhop/Kraken environment file (.env)

    Parameters
    ----------
    filepath : str or Path
        Output file path
    env : Environment
        Environment definition
    source : Source
        Source definition
    receiver : Receiver
        Receiver definition
    run_type : str, optional
        Run type character:
        'C' = coherent TL, 'I' = incoherent TL, 'S' = semi-coherent TL,
        'A' = arrivals, 'E' = eigenrays, 'R' = ray trace
        Default is 'C'.
    beam_type : str, optional
        Beam type: 'R' = ray-centered, 'C' = Cartesian, 'g' = geometric,
        'B' = Gaussian. Default is 'R'.
    n_beams : int, optional
        Number of beams. If 0, uses source.n_angles.
    alpha : tuple, optional
        Launch angle limits (min, max) in degrees. Default is (-80, 80).
    step : float, optional
        Step size in meters. If 0, uses automatic. Default is 0.
    z_box : float, optional
        Maximum depth for ray box. If None, uses 1.2 * env.depth.
    r_box : float, optional
        Maximum range for ray box. If None, uses 1.2 * receiver.range_max.
    """
    filepath = Path(filepath)

    if n_beams == 0:
        n_beams = source.n_angles

    # For range-dependent bathymetry, ensure z_box accounts for max depth
    max_bathy_depth = (
        env.bathymetry[:, 1].max() if len(env.bathymetry) > 0 else env.depth
    )
    max_depth = max(env.depth, max_bathy_depth)

    if z_box is None:
        z_box = 1.2 * max_depth

    if r_box is None:
        r_box = 1.2 * receiver.range_max if receiver.range_max > 0 else 10000

    with open(filepath, "w") as f:
        # Title
        f.write(f"'{env.name}'\n")

        # Frequency
        f.write(f"{source.frequency[0]:.6f}\n")

        # Number of media (1 for simple case)
        f.write("1\n")

        # SSP interpolation type mapping
        # Maps user-facing ssp_type to Acoustics Toolbox interpolation codes
        # Format: '<ssp_type><top_bc><atten_unit>'
        # Position 1: N=N2-Linear, C=C-Linear, P=PCHIP, S=Spline, Q=Quad, A=Analytic
        # Position 2: Top boundary condition (V=vacuum, R=rigid, A=halfspace)
        # Position 3-4: Attenuation units (W=dB/wavelength)
        ssp_interp_map = {
            # Profile types → default to C-Linear
            "isovelocity": "C",  # Constant profile → C-Linear
            "munk": "C",  # Munk profile data → C-Linear
            "linear": "C",  # Linear profile → C-Linear
            "bilinear": "C",  # Bilinear profile → C-Linear
            # Interpolation types → AT codes
            "n2linear": "N",  # N2-Linear approximation
            "c-linear": "C",  # C-Linear approximation
            "clin": "C",  # C-Linear (short form)
            "pchip": "P",  # PCHIP approximation
            "spline": "S",  # Spline approximation
            "cubic": "S",  # Cubic spline (alias)
            "quad": "Q",  # Quad approximation (needs .ssp file)
            "analytic": "A",  # Analytic SSP option
        }
        interp_char = ssp_interp_map.get(env.ssp_type.lower(), "C")

        # Top boundary (surface)
        top_bc = get_top_bc_code(env)

        # Attenuation units
        atten_unit = "W "  # dB/wavelength (two characters)

        # SSP option string (pad to 6 chars for Fortran compatibility)
        ssp_options = f"{interp_char}{top_bc}{atten_unit}"
        ssp_options = ssp_options.ljust(6)  # Pad with spaces to 6 characters
        f.write(f"'{ssp_options}'\n")

        write_surface_halfspace(f, env)

        # SSP depth range and data
        # For range-dependent bathymetry, extend SSP to max bathymetry depth
        z_min = 0.0
        max_bathy_depth = (
            env.bathymetry[:, 1].max() if len(env.bathymetry) > 0 else env.depth
        )
        z_max = max(env.depth, max_bathy_depth)

        # Extend SSP if needed
        ssp_data_extended = env.ssp_data.copy()
        if z_max > env.ssp_data[-1, 0]:
            # Add point at max depth with same sound speed as last point
            last_c = env.ssp_data[-1, 1]
            ssp_data_extended = np.vstack([ssp_data_extended, [z_max, last_c]])

        n_ssp = len(ssp_data_extended)
        f.write(f"{n_ssp}  {z_min:.1f}  {z_max:.1f},\n")

        for depth, c in ssp_data_extended:
            # Format: depth(m) c(m/s) [attenuation(dB/wavelength)]
            if env.attenuation > 0:
                f.write(f"{depth:.6f} {c:.6f} {env.attenuation:.6f} /\n")
            else:
                f.write(f"{depth:.6f} {c:.6f} /\n")

        # Bottom properties
        # Map human-readable names to Bellhop codes
        bottom_type_map = {
            "vacuum": "V",
            "rigid": "R",
            "half-space": "A",  # Acousto-elastic halfspace
            "halfspace": "A",
            "file": "F",
            "precalc": "P",
        }
        bottom_type = bottom_type_map.get(env.bottom.acoustic_type.lower(), "A")

        # Check if bathymetry is range-dependent (more than 1 point)
        is_range_dependent_bathy = len(env.bathymetry) > 1

        if is_range_dependent_bathy:
            # Write .bty file for range-dependent bathymetry
            bty_filepath = filepath.with_suffix(".bty")
            write_bty_file(bty_filepath, env.bathymetry, interp_type="L")
            # Append '~' to bottom type to indicate bathymetry from file
            # Format: 'A~' = halfspace with range-dependent bathymetry
            bottom_type_with_bathy = f"{bottom_type}~"
            f.write(f"'{bottom_type_with_bathy}' 0.0\n")
        else:
            # Range-independent bottom
            f.write(f"'{bottom_type}' 0.0\n")  # Top of halfspace (usually 0)

        # Write halfspace parameters (for range-independent or as defaults)
        if bottom_type in ["A", "F"]:  # Half-space with parameters
            # Format: depth sound_speed shear_speed density attenuation roughness /
            shear_speed = 0.0  # Usually 0 for fluid bottom
            roughness = (
                env.bottom.roughness if hasattr(env.bottom, "roughness") else 0.0
            )
            f.write(
                f" {env.bottom.depth:.2f}  {env.bottom.sound_speed:.2f} "
                f"{shear_speed:.1f} {env.bottom.density:.1f} "
                f"{env.bottom.attenuation:.1f} {roughness:.1f} /\n"
            )

        # Source depths
        n_sources = len(source.depth)
        f.write(f"{n_sources}\n")
        depths_str = " ".join([f"{d:.6f}" for d in source.depth])
        f.write(f"{depths_str} /\n")

        # Receiver depths
        n_rd = len(receiver.depths)
        f.write(f"{n_rd}\n")
        depths_str = " ".join([f"{d:.6f}" for d in receiver.depths])
        f.write(f"{depths_str} /\n")

        # Receiver ranges
        n_rr = len(receiver.ranges)
        f.write(f"{n_rr}\n")
        ranges_str = " ".join(
            [f"{r:.6f}" for r in receiver.ranges / 1000.0]
        )  # Convert to km
        f.write(f"{ranges_str} /\n")

        # Run type and beam type combined
        f.write(f"'{run_type}{beam_type}'\n")

        # Number of beams
        f.write(f"{n_beams}\n")

        # Launch angles
        f.write(f"{alpha[0]:.6f} {alpha[1]:.6f} /\n")

        # Step size (0 for automatic)
        f.write(f"{step:.6f}\n")

        # Box parameters (z in m, r in km as per Bellhop documentation)
        f.write(f"{z_box:.6f} {r_box / 1000.0:.6f}\n")



def write_ssp(filepath: Union[str, Path], r_km: np.ndarray, c: np.ndarray) -> None:
    """
    Write sound speed profile matrix to file.

    Parameters
    ----------
    filepath : str or Path
        SSP file path
    r_km : ndarray
        Range vector in kilometers, shape (N,)
    c : ndarray
        Sound speed profiles in m/s, shape (n_depth, N)
        Each column is the SSP at the corresponding range

    Notes
    -----
    File format:
    - Line 1: Number of profiles (N)
    - Line 2: Range vector in km (space-separated)
    - Following lines: Sound speed values row by row
      (each row is SSP values at all ranges for one depth)

    This format is used for range-dependent SSP input to acoustic models.

    Translated from OALIB writessp.m

    Examples
    --------
    >>> # Create range-dependent SSP
    >>> r_km = np.array([0, 10, 20, 30])
    >>> z = np.linspace(0, 100, 11)
    >>> c = 1500 - 0.1 * z[:, np.newaxis]  # Simple gradient
    >>> write_ssp('test.ssp', r_km, c)
    """
    filepath = Path(filepath)
    Npts = len(r_km)

    with open(filepath, "w") as fid:
        # Write number of profiles
        fid.write(f"{Npts}")

        # Write range vector
        for r in r_km:
            fid.write(f"{r:6.3f}  ")
        fid.write("\n")

        # Write sound speed profiles (row by row)
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                fid.write(f"{c[i, j]:6.1f} ")
            fid.write("\n")


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

    Examples
    --------
    >>> # Create simple reflection coefficient
    >>> angles = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    >>> # Full reflection with phase shift
    >>> R_complex = 0.9 * np.exp(1j * np.deg2rad(-20))
    >>> coeffs = np.full(len(angles), R_complex, dtype=complex)
    >>> write_reflection_coefficient('bottom.brc', angles, coeffs)

    >>> # Using amplitude and phase directly
    >>> amp_phase = np.column_stack([np.ones(10) * 0.9, np.ones(10) * -20])
    >>> write_reflection_coefficient('bottom.brc', angles, amp_phase)
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

    Examples
    --------
    >>> # Create Gaussian beam pattern (levels in dB)
    >>> angles = np.linspace(-90, 90, 181)
    >>> width = 10  # degrees (half-power beamwidth)
    >>> # linear -> dB; 0 dB at peak
    >>> linear = np.exp(-(angles**2) / (2 * width**2))
    >>> pattern_db = 20 * np.log10(np.clip(linear, 1e-6, None))
    >>> write_source_beam_pattern('source.sbp', angles, pattern_db)

    >>> # Create omnidirectional source (0 dB everywhere)
    >>> angles = np.array([-90, 0, 90])
    >>> pattern = np.array([0.0, 0.0, 0.0])
    >>> write_source_beam_pattern('omni.sbp', angles, pattern)
    """
    filepath = Path(filepath)

    n_angles = len(angles)

    with open(filepath, "w") as f:
        # Write number of angles
        f.write(f"{n_angles}\n")

        # Write angle, amplitude pairs
        for i in range(n_angles):
            f.write(f"{angles[i]:8.2f} {pattern[i]:12.6f}\n")


