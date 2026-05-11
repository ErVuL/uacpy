"""
Bellhop environment-file writer.

``write_bellhop_env_file`` writes the full ``.env`` (plus auxiliary ``.bty``,
``.ati``, ``.ssp``, ``.brc``/``.trc``, ``.sbp`` files) for a Bellhop or
BellhopCUDA run. Handles SSP interpolation types, range-dependent
bathymetry/altimetry, surface/bottom boundary conditions, source/receiver
specifications, and run-type / beam-parameter configuration.
"""

import numpy as np
from pathlib import Path
from typing import Union

from uacpy._log import log_message
from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.io.bathy_io import (
    write_bty_file,
    write_bty_long_format,
    write_ati_file,
)
from uacpy.io.oalib_writer import (
    _BOUNDARY_TYPE_MAP, get_top_bc_code,
    write_receiver_depths, write_receiver_ranges, write_source_depths,
    write_surface_halfspace,
)


def write_bellhop_env_file(
    filepath: Union[str, Path],
    env: Environment,
    source: Source,
    receiver: Receiver,
    run_type: str = "C",
    beam_type: str = "B",
    source_type: str = "R",
    grid_type: str = "R",
    bty_interp_type: str = 'L',
    source_beam_pattern: bool = False,
    beam_shift: bool = False,
    n_beams: int = 0,
    alpha: tuple = (-80, 80),
    step: float = 0.0,
    z_box: float = None,
    r_box: float = None,
    verbose: bool = False,
    **kwargs
):
    """
    Write Bellhop environment file (.env)

    This method generates a properly formatted Acoustics Toolbox environment file
    for Bellhop/BellhopCUDA ray tracing models.

    Parameters
    ----------
    filepath : Path
        Output file path for .env file
    env : Environment
        Environment definition (SSP, bathymetry, boundaries)
    source : Source
        Source definition (depth, frequency, beam angles)
    receiver : Receiver
        Receiver definition (depths, ranges)
    run_type : str, optional
        Run type (position 1): 'C'/'I'/'S'/'A'/'a'/'E'/'R'. Default is 'C'.
        Case is *significant*: 'A' is ASCII arrivals, 'a' is binary
        arrivals (ReadEnvironmentBell.f90:372-375).
        - C: Coherent TL
        - I: Incoherent TL
        - S: Semi-coherent TL
        - A: ASCII Arrivals
        - a: Binary Arrivals
        - E: Eigenrays
        - R: Ray trace
    beam_type : str, optional
        Beam type (position 2): 'B'/'R'/'C'/'b'/'g'/'G'/'S'. Default is 'B'.
        Case is significant: lowercase 'b','g' are ray-centered variants
        (ReadEnvironmentBell.f90:387-395), uppercase are Cartesian.
        - B: Gaussian beams (recommended)
        - R: Ray-centered beams
        - C: Cartesian beams
        - b: Geometric Gaussian, ray-centered
        - g: Geometric hat, ray-centered
        - G: Geometric hat, Cartesian
        - S: Simple Gaussian
    source_type : str, optional
        Source type (position 4): 'R' (point), 'X' (line). Default is 'R'.
    grid_type : str, optional
        Grid type (position 5): 'R' (rectilinear), 'I' (irregular). Default is 'R'.
    bty_interp_type : str, optional
        Interpolation type for both the ``.bty`` (bathymetry) and
        ``.ati`` (altimetry) files: 'L' (linear, default) or 'C'
        (curvilinear). The same value is used for both files.
    source_beam_pattern : bool, optional
        When True, emits '*' in RunType position 3 so Bellhop reads
        ``<base>.sbp`` (source beam pattern file). The caller is
        responsible for staging that file next to the .env. Default: False.
    beam_shift : bool, optional
        When True, sets RunType position 7 to 'S' enabling beam-shift
        on boundary reflections per Beam%Type(4:4)
        (ReadEnvironmentBell.f90:159-166). Default: False (no shift).
    n_beams : int, optional
        Number of beams. If 0, uses source.n_angles. Bellhop picks a
        conservative default in that case.
    alpha : tuple, optional
        Launch angle limits (min, max) in degrees. Default is (-80, 80).
    step : float, optional
        Step size in meters. If 0, uses automatic. Default is 0.
    z_box : float, optional
        Maximum depth for ray box. If None, uses 1.2 * env.depth.
    r_box : float, optional
        Maximum range for ray box. If None, uses 1.2 * receiver.range_max.
    verbose : bool, optional
        Print verbose output. Default is False.
    **kwargs
        Advanced Cerveny beam parameters (for beam_type 'C' or 'R'):
        - beam_width_type (str): 'F'/'M'/'W'
        - beam_curvature (str): 'D'/'S'/'Z'
        - eps_multiplier (float): Epsilon multiplier
        - r_loop (float): Range (m) for choosing beam width
        - n_image (int): Number of images
        - ib_win (int): Beam windowing
        - component (str): 'P' for pressure (default), 'D' for displacement

    Notes
    -----
    This is a shared method used by both Bellhop and BellhopCUDA classes.
    It handles all the complexity of the Acoustics Toolbox .env format.

    For range-dependent bathymetry, automatically generates a .bty file.
    For range-dependent SSP with Quad interpolation, generates a .ssp file.
    """
    # Bellhop treats NBEAMS <= 0 as "pick conservatively" (it auto-
    # selects based on geometry). Honor the user's intent: passing
    # n_beams=0 defers to Bellhop rather than silently substituting
    # source.n_angles.
    if n_beams is None:
        n_beams = 0

    if z_box is None:
        z_box = 1.2 * env.depth

    if r_box is None:
        r_box = 1.2 * receiver.range_max if receiver.range_max > 0 else 10000

    with open(filepath, "w") as f:
        # Title
        f.write(f"'{env.name}'\n")

        # Frequency
        f.write(f"{source.frequencies[0]:.6f}\n")

        # Number of media (1 for simple case)
        f.write("1\n")

        from uacpy.core.constants import parse_ssp_type
        interp_char = parse_ssp_type(env.ssp.interp).to_acoustics_toolbox_code()

        # Top boundary (surface)
        top_bc = get_top_bc_code(env)

        # TopOpt(3) = 'W' (dB/wavelength, uacpy convention).
        # TopOpt(4) from env.absorption.
        from uacpy.core.constants import AttenuationUnits
        atten_unit_char = AttenuationUnits.DB_PER_WAVELENGTH.to_char()
        vol_atten_char = (
            env.absorption.topopt_code() if env.absorption is not None else ' '
        )
        atten_unit = f"{atten_unit_char}{vol_atten_char}"

        # Position 5: Altimetry flag ('~' = read .ati file, ' ' = flat surface)
        has_altimetry = getattr(env, 'altimetry', None) is not None and len(env.altimetry) > 1
        alti_char = '~' if has_altimetry else ' '

        # SSP option string (pad to 6 chars for Fortran compatibility)
        ssp_options = f"{interp_char}{top_bc}{atten_unit}{alti_char}"
        ssp_options = ssp_options.ljust(6)
        f.write(f"'{ssp_options}'\n")

        write_surface_halfspace(f, env)

        from uacpy.io.oalib_writer import write_absorption_block
        write_absorption_block(f, env)

        # env.altimetry is positive-up; Bellhop's .ati is positive-down.
        if has_altimetry:
            ati_filepath = filepath.with_suffix(".ati")
            ati_data = env.altimetry.copy()
            ati_data[:, 1] = -ati_data[:, 1]
            write_ati_file(ati_filepath, ati_data, interp_type=bty_interp_type)
            log_message('bellhop_writer',
                        f"wrote altimetry file: {ati_filepath}",
                        verbose=verbose)

        # If the surface boundary is a reflection file ('F'), copy the
        # .trc file alongside the .env so Bellhop can read it by base
        # name (mirroring the '.brc' flow for the bottom).
        if top_bc == 'F':
            refl_path = getattr(env.surface, 'reflection_file', None)
            if not refl_path:
                from uacpy.core.exceptions import ConfigurationError
                raise ConfigurationError(
                    "surface.acoustic_type='file' requires "
                    "reflection_file= on the surface BoundaryProperties."
                )
            import shutil
            src = Path(refl_path)
            if not src.exists():
                raise FileNotFoundError(
                    f"Top reflection file not found: {refl_path}"
                )
            dest = filepath.with_suffix('.trc')
            shutil.copy(src, dest)
            log_message('bellhop_writer',
                        f"copied top reflection file: {src} -> {dest}",
                        verbose=verbose)

        # Handle range-dependent SSP if using Quad interpolation
        if interp_char == 'Q' and env.has_range_dependent_ssp():
            from uacpy.io.oalib_writer import write_ssp
            ssp_file = filepath.with_suffix('.ssp')
            # write_ssp (.ssp file format) expects ranges in km.
            write_ssp(ssp_file, env.ssp.ranges / 1000.0, env.ssp.data)
            log_message('bellhop_writer',
                        f"wrote range-dependent SSP file: {ssp_file}",
                        verbose=verbose)

        # Extend SSP above MSL to cover any wave crests (env.altimetry
        # positive-up, SSP z-axis positive-down).
        z_min = 0.0
        if has_altimetry:
            max_alti_above_msl = env.altimetry[:, 1].max()
            if max_alti_above_msl > 0:
                z_min = -max_alti_above_msl - 0.5

        # Bellhop requires the last SSP sample to sit *exactly* at the
        # medium depth written in the SSP header — and the header uses
        # one decimal place. Round both sides through the same value so
        # the parser sees aligned depths.
        z_max = float(f"{env.depth:.1f}")

        if env.ssp.is_range_dependent:
            ssp_block = env.ssp.eval(range=0.0).extend_to(z_max)
        else:
            ssp_block = env.ssp.extend_to(z_max)
        ssp_data_extended = ssp_block.to_pairs()
        if z_min < ssp_data_extended[0, 0]:
            first_c = ssp_data_extended[0, 1]
            ssp_data_extended = np.vstack([[z_min, first_c], ssp_data_extended])

        n_ssp = len(ssp_data_extended)
        # Bellhop reads this line as (NPts, Sigma, Depth) per
        # ReadEnvironmentBell.f90:73 — Sigma is the top-boundary RMS
        # roughness, NOT z_min. Always emit Sigma=0.0; altimetry crests
        # are handled via the .ati file, not via this slot.
        f.write(f"{n_ssp}  0.0  {z_max:.1f},\n")

        # SSP row: z alphaR betaR rhoR alphaI betaI /
        # Emit all 6 columns so an elastic top halfspace doesn't leak ice
        # properties into the water column via Fortran's READ.
        from uacpy.core.absorption import ConstantAbsorption
        alpha_i = (
            env.absorption.value_db_per_wavelength
            if isinstance(env.absorption, ConstantAbsorption)
            else 0.0
        )
        for depth, c in ssp_data_extended:
            f.write(
                f"{depth:.6f} {c:.6f} 0.0 1.0 "
                f"{alpha_i:.6f} 0.0 /\n"
            )

        bottom_acoustic_type = env.halfspace_at_range(0.0).acoustic_type
        bottom_type = _BOUNDARY_TYPE_MAP.get(bottom_acoustic_type.lower(), "A")

        is_range_dependent_bathy = len(env.bathymetry) > 1

        from uacpy.core.environment import RangeDependentBottom
        hs = env.halfspace_at_range(0.0)
        if is_range_dependent_bathy:
            bty_filepath = filepath.with_suffix(".bty")
            # The 2nd TYPE char in the .bty (short 'S' vs long 'L') is
            # auto-selected by the writer: write_bty_long_format emits
            # 'LL'/'CL', write_bty_file emits 'LS'/'CS'. Callers only pick
            # the 1st char (interpolation) via bty_interp_type.
            if isinstance(env.bottom, RangeDependentBottom) and len(env.bottom.ranges) > 0:
                write_bty_long_format(
                    bty_filepath, env.bathymetry, env.bottom,
                    interp_type=bty_interp_type,
                )
            else:
                write_bty_file(
                    bty_filepath, env.bathymetry,
                    interp_type=bty_interp_type,
                )
            bottom_type_with_bathy = f"{bottom_type}~"
            # 2nd field on this BOT line is sigma (top-of-bottom RMS
            # roughness) per ReadEnvironmentBell.f90:466.
            roughness = getattr(hs, 'roughness', 0.0)
            f.write(f"'{bottom_type_with_bathy}' {roughness:.6f}\n")
        else:
            roughness = getattr(hs, 'roughness', 0.0)
            f.write(f"'{bottom_type}' {roughness:.6f}\n")

        # Write halfspace parameters (for range-independent or as defaults)
        if bottom_type == "G":  # Grain size
            # Format: depth Mz (mean grain size in phi units)
            hs = env.halfspace_at_range(0.0)
            grain_size = getattr(hs, 'grain_size_phi', 1.0)
            f.write(f" {env.depth:.2f}  {grain_size:.2f} /\n")
        elif bottom_type == "F":  # Reflection coefficient from file
            hs = env.halfspace_at_range(0.0)
            if hs.reflection_file:
                import shutil
                brc_source = Path(hs.reflection_file)
                if brc_source.exists():
                    brc_dest = filepath.with_suffix('.brc')
                    shutil.copy(brc_source, brc_dest)
                    log_message('bellhop_writer',
                                f"copied reflection file: {brc_source} -> {brc_dest}",
                                verbose=verbose)
                else:
                    raise FileNotFoundError(
                        f"bellhop_writer: reflection coefficient file not found: "
                        f"{hs.reflection_file}; "
                        f"generate it via BOUNCE or OASR first."
                    )
            else:
                raise ValueError(
                    "bellhop_writer: acoustic_type='file' requires reflection_file= "
                    "on the bottom BoundaryProperties (path to a .brc file)."
                )

            # For 'F' type, Bellhop finds the .brc file by name convention
            # (same base name as .env). No additional lines needed in the env file.
        elif bottom_type == "A":  # Acousto-elastic halfspace
            # Per ReadEnvironmentBell.f90:474 the row is
            # READ(ENVFile,*) zTemp, alphaR, betaR, rhoR, alphaI, betaI
            # i.e. depth cp cs rho alpha_p alpha_s — the 6th column is
            # SHEAR attenuation, NOT roughness. Roughness (sigma) lives
            # on the preceding BOT line ('A' sigma).
            hs = env.halfspace_at_range(0.0)
            shear_speed = getattr(hs, 'shear_speed', 0.0)
            shear_atten = getattr(hs, 'shear_attenuation', 0.0)
            f.write(
                f" {env.depth:.2f}  {hs.sound_speed:.2f} "
                f"{shear_speed:.1f} {hs.density:.1f} "
                f"{hs.attenuation:.6f} {shear_atten:.6f} /\n"
            )

        write_source_depths(f, source)
        write_receiver_depths(f, receiver)
        write_receiver_ranges(f, receiver)

        # Construct full RunType string per AT specification
        # Format: 'XXZZZZZ' (7 chars), positions:
        #   1 = run_type         (C/I/S/A/a/E/R)   — case significant
        #   2 = beam_type        (B/R/C/S/b/g/G/^) — case significant
        #   3 = reserved          ('*' for source beam pattern, else ' ')
        #   4 = source_type      (R/X) — uppercase
        #   5 = grid_type        (R/I) — uppercase
        #   6 = dimension        (always '2' for 2D Bellhop)
        #   7 = beam shift       ('S' to enable, else ' ') per
        #       ReadEnvironmentBell.f90:159  Beam%Type(4:4) = pos 7.
        # IMPORTANT: do NOT uppercase run_type or beam_type.  'A' vs
        # 'a' picks ASCII vs binary arrivals (ReadEnvironmentBell.f90
        # :372-375); lowercase 'b','g','^' pick ray-centered beam
        # variants (ReadEnvironmentBell.f90:387-395).  Uppercasing
        # destroys both user intents.
        # Position 3: '*' enables reading <base>.sbp source beam pattern
        # file; blank otherwise (see bellhop.htm → ReadRunType).
        position_3 = '*' if source_beam_pattern else ' '
        # Position 4/5 use uppercase (only 'R'/'X' and 'R'/'I' accepted).
        position_4 = source_type.upper()
        position_5 = grid_type.upper()
        # Position 6: dimensionality. Hardcoded to '2' (2D Bellhop).
        # 3D support (BELLHOP3D) would set this to '3' and plug into
        # BellhopCUDA._build_command / a future Bellhop3D class; 3D also
        # changes several downstream blocks (bearings, 3D bty, beam fan).
        position_6 = '2'
        # Position 7: 'S' for beam shift, otherwise blank.
        position_7 = 'S' if beam_shift else ' '
        full_run_type = (
            f"{run_type}{beam_type}{position_3}"
            f"{position_4}{position_5}{position_6}{position_7}"
        )
        f.write(f"'{full_run_type}'\n")

        # Number of beams
        f.write(f"{n_beams}\n")

        # Launch angles
        f.write(f"{alpha[0]:.6f} {alpha[1]:.6f} /\n")

        # Step size (0 for automatic)
        f.write(f"{step:.6f}\n")

        # Box parameters (z in m, r in km as per Bellhop documentation)
        f.write(f"{z_box:.6f} {r_box/1000.0:.6f}\n")

        # Cerveny / Simple Gaussian beam parameters
        # Required by Bellhop Fortran when beam_type is 'C', 'R' or 'S'
        # (ReadEnvironmentBell.f90:195-217).  Test case-insensitively
        # without destroying the original beam_type (lowercase 'b','g'
        # are distinct ray-centered variants).
        if beam_type.upper() in ('C', 'R', 'S'):
            beam_width_type = kwargs.get('beam_width_type', 'F')
            beam_curvature = kwargs.get('beam_curvature', 'D')
            eps_multiplier = kwargs.get('eps_multiplier', 1.0)
            # Bellhop's RLoop column expects km; uacpy keeps everything in
            # metres at the API surface, so convert here.
            r_loop_km = kwargs.get('r_loop', 1000.0) / 1000.0
            n_image = kwargs.get('n_image', 1)
            ib_win = kwargs.get('ib_win', 4)
            component = kwargs.get('component', 'P')

            beam_type_str = f"{beam_width_type}{beam_curvature}"
            f.write(f"'{beam_type_str}' {eps_multiplier:.6f} {r_loop_km:.6f}\n")

            # Line 2: n_image, ib_win, component
            f.write(f"{n_image} {ib_win} '{component}'\n")
