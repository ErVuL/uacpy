"""
Bellhop environment-file writer.

``write_bellhop_env_file`` writes the full ``.env`` (plus auxiliary ``.bty``,
``.ati``, ``.ssp``, ``.brc``/``.trc``, ``.sbp`` files) for a Bellhop run
(any backend). Handles SSP interpolation types, range-dependent
bathymetry/altimetry, surface/bottom boundary conditions, source/receiver
specifications, and run-type / beam-parameter configuration.
"""

import warnings
import numpy as np
from pathlib import Path
from typing import Union

from uacpy._log import log_message
from uacpy.core.absorption import ConstantAbsorption
from uacpy.core.constants import AttenuationUnits
from uacpy.core.exceptions import ConfigurationError
from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.io.bathy_io import (
    write_bty_file,
    write_bty_long_format,
    write_ati_file,
)
from uacpy.io.oalib_writer import (
    _BOUNDARY_TYPE_MAP, get_top_bc_code, resolve_ssp_topopt,
    write_absorption_block, write_receiver_depths, write_receiver_ranges,
    write_source_depths, write_ssp, write_surface_halfspace,
)
from uacpy.io.refl_io import stage_reflection_file
from uacpy.io.units import m_to_km


def write_bellhop_env_file(
    filepath: Union[str, Path],
    env: Environment,
    source: Source,
    receiver: Receiver,
    run_type: str = "C",
    beam_type: str = "B",
    source_type: str = "R",
    grid_type: str = "R",
    interp_ssp: str = 'linear',
    interp_bathymetry: str = 'linear',
    interp_altimetry: str = 'linear',
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
    for the Bellhop ray tracing model (any backend).

    Parameters
    ----------
    filepath : Path
        Output file path for .env file
    env : Environment
        Environment definition (SSP, bathymetry, boundaries)
    source : Source
        Source definition (depths, frequencies, source geometry)
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
    interp_ssp : str, optional
        SSP connection scheme when ``env.ssp.shape == 'measured'``
        (drives ``TopOpt(1)``): ``'linear'`` (default), ``'pchip'``,
        ``'cubic'``, ``'quad'``, ``'n2linear'``.
    interp_bathymetry : str, optional
        ``.bty`` interpolation: ``'linear'`` (default) or ``'curvilinear'``.
    interp_altimetry : str, optional
        ``.ati`` interpolation: ``'linear'`` (default) or ``'curvilinear'``.
    source_beam_pattern : bool, optional
        When True, emits '*' in RunType position 3 so Bellhop reads
        ``<base>.sbp`` (source beam pattern file). The caller is
        responsible for staging that file next to the .env. Default: False.
    beam_shift : bool, optional
        When True, sets RunType position 7 to 'S' enabling beam-shift
        on boundary reflections per Beam%Type(4:4)
        (ReadEnvironmentBell.f90:159-166). Default: False (no shift).
    n_beams : int, optional
        Number of beams. ``0`` defers to Bellhop's conservative
        auto-selection (NBEAMS<=0 in the Fortran reader).
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
    This is a shared method used across all Bellhop backends.
    It handles all the complexity of the Acoustics Toolbox .env format.

    For range-dependent bathymetry, automatically generates a .bty file.
    For range-dependent SSP with Quad interpolation, generates a .ssp file.
    """
    filepath = Path(filepath)

    if n_beams is None:
        n_beams = 0

    if z_box is None:
        z_box = 1.2 * env.depth

    # Box%r is the horizontal-range cut-off for ray integration. A ray
    # reaching a receiver at range_max has horizontal range == range_max, so
    # the 1.2× pad already captures every arrival at the outer receivers;
    # enlarging it further only risks rays leaving the range window where a
    # range-dependent .ssp defines the sound speed (BHC_ERR_OUTSIDE_SSP).
    if r_box is None:
        r_box = 1.2 * receiver.range_max if receiver.range_max > 0 else 10000

    with open(filepath, "w") as f:
        # Title
        f.write(f"'{env.name}'\n")

        # Frequency
        f.write(f"{source.frequencies[0]:.6f}\n")

        # Number of media (1 for simple case)
        f.write("1\n")

        interp_char = resolve_ssp_topopt(env, interp_ssp)

        _GEOM_INTERP_TO_CODE = {'linear': 'L', 'curvilinear': 'C'}
        bty_code = _GEOM_INTERP_TO_CODE.get(str(interp_bathymetry).lower())
        ati_code = _GEOM_INTERP_TO_CODE.get(str(interp_altimetry).lower())
        if bty_code is None:
            raise ConfigurationError(
                f"interp_bathymetry must be 'linear' or 'curvilinear'; "
                f"got {interp_bathymetry!r}"
            )
        if ati_code is None:
            raise ConfigurationError(
                f"interp_altimetry must be 'linear' or 'curvilinear'; "
                f"got {interp_altimetry!r}"
            )

        # Top boundary (surface)
        top_bc = get_top_bc_code(env)

        # TopOpt(3) = 'W' (dB/wavelength, uacpy convention).
        # TopOpt(4) from env.absorption.
        atten_unit_char = AttenuationUnits.DB_PER_WAVELENGTH.to_char()
        vol_atten_char = (
            env.absorption.topopt_code() if env.absorption is not None else ' '
        )
        atten_unit = f"{atten_unit_char}{vol_atten_char}"

        # Position 5: Altimetry flag ('~' = read .ati file, ' ' = flat surface)
        has_altimetry = (getattr(env, 'altimetry', None) is not None
                         and env.altimetry.n_ranges >= 1)
        alti_char = '~' if has_altimetry else ' '

        # SSP option string (pad to 6 chars for Fortran compatibility)
        ssp_options = f"{interp_char}{top_bc}{atten_unit}{alti_char}"
        ssp_options = ssp_options.ljust(6)
        f.write(f"'{ssp_options}'\n")

        write_surface_halfspace(f, env)

        write_absorption_block(f, env)

        # env.altimetry is positive-up; Bellhop's .ati is positive-down.
        if has_altimetry:
            ati_filepath = filepath.with_suffix(".ati")
            ati_data = env.altimetry.to_pairs()
            if ati_data.shape[0] == 1:
                # Single sample = constant offset; expand to a 2-point
                # profile spanning the receiver range so it isn't dropped.
                r_last = max(float(np.max(receiver.ranges)),
                             float(ati_data[0, 0]) + 1.0)
                ati_data = np.array([[0.0, ati_data[0, 1]],
                                     [r_last, ati_data[0, 1]]])
            ati_data[:, 1] = -ati_data[:, 1]
            write_ati_file(ati_filepath, ati_data, interp_type=ati_code)
            log_message('bellhop_writer',
                        f"wrote altimetry file: {ati_filepath}",
                        verbose=verbose)

        # A surface reflection file ('F') is read by base name, so it has to
        # sit next to the .env as <base>.trc.
        if top_bc == 'F':
            stage_reflection_file(
                getattr(env.surface, 'reflection_file', None),
                filepath, boundary='top', verbose=verbose,
            )

        # Handle range-dependent SSP if using Quad interpolation
        if interp_char == 'Q' and env.has_range_dependent_ssp:
            ssp_file = filepath.with_suffix('.ssp')
            ssp_ranges = np.asarray(env.ssp.ranges, dtype=float)
            ssp_data = np.asarray(env.ssp.data, dtype=float)
            # Bellhop drops a ray once it passes Box%r, and a ray landing on
            # the last SSP range is still flagged "outside the soundspeed box",
            # so the .ssp must extend *strictly past* r_box. When the profile
            # grid stops at or before the box, hold the last profile constant
            # out to 1.1·r_box (range-independent beyond the last profile) and
            # warn.
            if ssp_ranges.size and ssp_ranges.max() <= r_box:
                warnings.warn(
                    f"Bellhop: range-dependent SSP spans to "
                    f"{ssp_ranges.max():.0f} m but the ray box reaches "
                    f"{r_box:.0f} m; holding the last profile constant beyond "
                    f"{ssp_ranges.max():.0f} m. Define SSP profiles out to the "
                    f"receiver range to control the far-range sound speed.",
                    UserWarning, stacklevel=2,
                )
                ssp_ranges = np.append(ssp_ranges, 1.1 * r_box)
                ssp_data = np.column_stack([ssp_data, ssp_data[:, -1]])
            # The source sits at range 0 == Seg.r[0]; a ray back-scattered off
            # a steep up-slope can step to negative range, which Quad's box
            # check (x < Seg.r[0]) flags as BHC_ERR_OUTSIDE_SSP under
            # bellhopcuda. Prepend a guard column at -1.1·r_box holding the
            # first profile constant so the box brackets that excursion (no
            # warning: negative range is not a user-controllable axis).
            if ssp_ranges.size and ssp_ranges.min() >= 0.0:
                ssp_ranges = np.insert(ssp_ranges, 0, -1.1 * r_box)
                ssp_data = np.column_stack([ssp_data[:, 0], ssp_data])
            write_ssp(ssp_file, ssp_ranges, ssp_data)
            log_message('bellhop_writer',
                        f"wrote range-dependent SSP file: {ssp_file}",
                        verbose=verbose)

        # Extend SSP above MSL to cover any wave crests (env.altimetry
        # positive-up, SSP z-axis positive-down).
        z_min = 0.0
        if has_altimetry:
            max_alti_above_msl = env.altimetry.heights.max()
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

        bottom_acoustic_type = env.bottom.halfspace_at(range=0.0).acoustic_type
        bottom_type = _BOUNDARY_TYPE_MAP.get(bottom_acoustic_type.lower(), "A")

        is_range_dependent_bathy = env.bathymetry.n_ranges > 1
        rd_bottom = (env.bottom.is_range_dependent
                     and len(env.bottom.ranges) > 0)

        hs = env.bottom.halfspace_at(range=0.0)
        if is_range_dependent_bathy or rd_bottom:
            bty_filepath = filepath.with_suffix(".bty")
            # The 2nd TYPE char in the .bty (short 'S' vs long 'L') is
            # auto-selected by the writer: write_bty_long_format emits
            # 'LL'/'CL', write_bty_file emits 'LS'/'CS'. The first char
            # is the interpolation chosen via ``interp_bathymetry``.
            if rd_bottom:
                # The long-format .bty is the only vehicle for per-range
                # geoacoustics; a flat bathymetry becomes a 2-point constant
                # profile so the property breaks still reach Bellhop.
                bathy_for_bty = env.bathymetry
                if not is_range_dependent_bathy:
                    r_last = max(float(np.max(env.bottom.ranges)),
                                 float(np.max(receiver.ranges)))
                    bathy_for_bty = np.array(
                        [[0.0, env.depth], [r_last, env.depth]])
                write_bty_long_format(
                    bty_filepath, bathy_for_bty, env.bottom,
                    interp_type=bty_code,
                )
            else:
                write_bty_file(
                    bty_filepath, env.bathymetry,
                    interp_type=bty_code,
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
        if bottom_type == "F":
            # Bellhop finds the .brc by name convention (same base name as the
            # .env), so staging it beside the file is the whole job — no extra
            # lines go into the env.
            hs = env.bottom.halfspace_at(range=0.0)
            stage_reflection_file(hs.reflection_file, filepath,
                                  boundary='bottom', verbose=verbose)
        elif bottom_type == "A":  # Acousto-elastic halfspace
            # Per ReadEnvironmentBell.f90:474 the row is
            # READ(ENVFile,*) zTemp, alphaR, betaR, rhoR, alphaI, betaI
            # i.e. depth cp cs rho alpha_p alpha_s — the 6th column is
            # SHEAR attenuation, NOT roughness. Roughness (sigma) lives
            # on the preceding BOT line ('A' sigma).
            hs = env.bottom.halfspace_at(range=0.0)
            shear_speed = getattr(hs, 'shear_speed', 0.0)
            shear_atten = getattr(hs, 'shear_attenuation', 0.0)
            f.write(
                f" {env.depth:.2f}  {hs.sound_speed:.2f} "
                f"{shear_speed:.2f} {hs.density:.2f} "
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
        # the ``--3D`` _build_command path; 3D also changes several
        # downstream blocks (bearings, 3D bty, beam fan).
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
        f.write(f"{z_box:.6f} {float(m_to_km(r_box)):.6f}\n")

        # Cerveny beam parameters.  ReadEnvironmentBell.f90 reads the two
        # extra lines only for 'R'/'C'; 'S' (simple Gaussian) shares the
        # no-extra-read case with 'G'/'g'/'^'/'B'/'b'.  Test case-insensitively
        # without destroying the original beam_type (lowercase 'b','g'
        # are distinct ray-centered variants).
        if beam_type.upper() in ('C', 'R'):
            beam_width_type = kwargs.get('beam_width_type', 'F')
            beam_curvature = kwargs.get('beam_curvature', 'D')
            eps_multiplier = kwargs.get('eps_multiplier', 1.0)
            # Bellhop's RLoop column expects km; uacpy keeps everything in
            # metres at the API surface, so convert here.
            r_loop_km = float(m_to_km(kwargs.get('r_loop', 1000.0)))
            n_image = kwargs.get('n_image', 1)
            ib_win = kwargs.get('ib_win', 4)
            component = kwargs.get('component', 'P')

            beam_type_str = f"{beam_width_type}{beam_curvature}"
            f.write(f"'{beam_type_str}' {eps_multiplier:.6f} {r_loop_km:.6f}\n")

            # Line 2: n_image, ib_win, component
            f.write(f"{n_image} {ib_win} '{component}'\n")
