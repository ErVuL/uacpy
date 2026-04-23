"""
Bellhop environment file writer

Shared utility for writing Bellhop .env files, used by both Bellhop and BellhopCUDA models.
"""

import numpy as np
from pathlib import Path
from typing import Optional

from uacpy.core.environment import Environment
from uacpy.core.source import Source
from uacpy.core.receiver import Receiver
from uacpy.io.bty_writer import (
    write_bty_file,
    write_bty_long_format,
    write_ati_file,
)
from uacpy.io.env_writer import get_top_bc_code, write_surface_halfspace


class BellhopEnvWriter:
    """
    Writer for Bellhop environment files (.env)

    Handles all the complexity of the Acoustics Toolbox .env format including:
    - Sound speed profiles with multiple interpolation types
    - Range-dependent bathymetry via .bty files
    - Surface and bottom boundary conditions
    - Source and receiver specifications
    - Run type and beam parameter configuration
    """

    @staticmethod
    def write_env_file(
        filepath: Path,
        env: Environment,
        source: Source,
        receiver: Receiver,
        run_type: str = "C",
        beam_type: str = "B",
        source_type: str = "R",
        grid_type: str = "R",
        volume_attenuation: Optional[str] = None,
        attenuation_unit: str = 'W',
        francois_garrison_params: Optional[tuple] = None,
        bio_layers: Optional[list] = None,
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
        volume_attenuation : str, optional
            Volume attenuation formula: 'T' (Thorp), 'F' (Francois-Garrison),
            'B' (Biological), or None. Default is None.
        attenuation_unit : str, optional
            Attenuation unit (TopOpt position 3): 'N' (nepers/m), 'F' (dB/kmHz),
            'M' (dB/m), 'W' (dB/wavelength, default), 'Q' (quality factor),
            'L' (loss tangent). 'm' (dB/m with per-SSP power-law BETA/fT)
            is rejected because uacpy has no Environment field for the
            power-law exponent yet.
        francois_garrison_params : tuple, optional
            (T, S, pH, z_bar): temperature (degC), salinity (ppt), pH, and
            mean depth (m) for the Francois-Garrison volume-attenuation formula.
            Required when ``volume_attenuation='F'``.
        bio_layers : list of tuples, optional
            Biological attenuation layers when ``volume_attenuation='B'``.
            Each entry is (Z1, Z2, f0, Q, a0): top depth, bottom depth,
            resonance frequency, quality factor, absorption coefficient.
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
            Maximum depth for ray box. If None, uses 1.2 * max(env.depth, bathymetry).
        r_box : float, optional
            Maximum range for ray box. If None, uses 1.2 * receiver.range_max.
        verbose : bool, optional
            Print verbose output. Default is False.
        **kwargs
            Advanced Cerveny beam parameters (for beam_type 'C' or 'R'):
            - beam_width_type (str): 'F'/'M'/'W'
            - beam_curvature (str): 'D'/'S'/'Z'
            - eps_multiplier (float): Epsilon multiplier
            - r_loop (float): Range for beam width (km)
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
            # AT codes: N=N2-Linear, C=C-Linear, P=PCHIP, S=Spline, Q=Quad, A=Analytic
            ssp_interp_map = {
                # Profile types → default to C-Linear
                "isovelocity": "C",  # Constant profile → C-Linear
                "munk": "C",         # Munk profile data → C-Linear
                "linear": "C",       # Linear profile → C-Linear
                "bilinear": "C",     # Bilinear profile → C-Linear
                # Interpolation types → AT codes
                "n2linear": "N",     # N2-Linear approximation
                "c-linear": "C",     # C-Linear approximation
                "clin": "C",         # C-Linear (short form)
                "pchip": "P",        # PCHIP approximation
                "spline": "S",       # Spline approximation
                "cubic": "S",        # Cubic spline (alias)
                "quad": "Q",         # Quad approximation (needs .ssp file)
                "analytic": "A",     # Analytic SSP option
            }
            interp_char = ssp_interp_map.get(env.ssp_type.lower(), "C")

            # Top boundary (surface)
            top_bc = get_top_bc_code(env)

            # Attenuation units (2-character field)
            # Position 3: Unit type. Accepted Bellhop codes (ReadEnvironmentBell
            #   .f90 284-299): 'N' nepers/m, 'F' dB/kmHz, 'M' dB/m,
            #   'W' dB/wavelength, 'Q' quality factor, 'L' loss tangent.
            #   ('m' dB/m with per-SSP BETA/fT power-law is parsed by
            #   Bellhop but requires a separate exponent that uacpy's
            #   Environment does not expose; rejected below.)
            # Position 4: Volume attenuation formula: T/F/B/' '.
            atten_unit_char = attenuation_unit if attenuation_unit in (
                'N', 'F', 'M', 'W', 'Q', 'L', 'm') else 'W'
            # 'm' (dB/m with per-SSP BETA/fT power-law) requires a
            # separate power-law exponent (BETA) and reference frequency
            # (fT), which are distinct from env.attenuation.  uacpy's
            # Environment does not expose a power-law exponent field, so
            # emitting 'm' here would mis-use env.attenuation as BETA
            # (ssp_mod.f90 / EnvironmentalFile.html:411-424).  Fail loudly
            # until a dedicated field is added.
            if atten_unit_char == 'm':
                from uacpy.core.exceptions import ConfigurationError
                raise ConfigurationError(
                    "attenuation_unit='m' (dB/m with power-law BETA/fT) "
                    "requires a dedicated power-law exponent and "
                    "transition frequency that uacpy's Environment does "
                    "not yet expose. Please use one of 'N', 'F', 'M', "
                    "'W', 'Q', or 'L', or contribute an "
                    "Environment.attenuation_exponent field."
                )
            vol_atten_char = volume_attenuation.upper() if volume_attenuation else ' '
            atten_unit = f"{atten_unit_char}{vol_atten_char}"

            # Validate F/B required params up-front so the user sees a clear
            # error instead of a Bellhop parse failure.
            if vol_atten_char == 'F' and francois_garrison_params is None:
                from uacpy.core.exceptions import ConfigurationError
                raise ConfigurationError(
                    "volume_attenuation='F' (Francois-Garrison) requires "
                    "francois_garrison_params=(T, S, pH, z_bar)."
                )
            if vol_atten_char == 'B' and not bio_layers:
                from uacpy.core.exceptions import ConfigurationError
                raise ConfigurationError(
                    "volume_attenuation='B' (biological) requires a non-empty "
                    "bio_layers list of (Z1, Z2, f0, Q, a0) tuples."
                )

            # Position 5: Altimetry flag ('~' = read .ati file, ' ' = flat surface)
            has_altimetry = getattr(env, 'altimetry', None) is not None and len(env.altimetry) > 1
            alti_char = '~' if has_altimetry else ' '

            # SSP option string (pad to 6 chars for Fortran compatibility)
            ssp_options = f"{interp_char}{top_bc}{atten_unit}{alti_char}"
            ssp_options = ssp_options.ljust(6)  # Pad with spaces to 6 characters
            f.write(f"'{ssp_options}'\n")

            write_surface_halfspace(f, env)

            # Volume attenuation follow-up lines (consumed by Bellhop right
            # after the TopOpt/surface halfspace block).
            if vol_atten_char == 'F':
                T, S, pH, z_bar = francois_garrison_params
                f.write(f"{T:.3f} {S:.3f} {pH:.3f} {z_bar:.3f}\n")
            elif vol_atten_char == 'B':
                f.write(f"{len(bio_layers)}\n")
                for layer in bio_layers:
                    z1, z2, f0, Q, a0 = layer
                    f.write(
                        f"{z1:.3f} {z2:.3f} {f0:.3f} {Q:.3f} {a0:.6f}\n"
                    )

            # Write .ati file if altimetry provided
            if has_altimetry:
                ati_filepath = filepath.with_suffix(".ati")
                write_ati_file(ati_filepath, env.altimetry, interp_type=bty_interp_type)
                if verbose:
                    print(f"[BellhopEnvWriter] Wrote altimetry file: {ati_filepath}")

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
                if verbose:
                    print(f"[BellhopEnvWriter] Copied top reflection file: "
                          f"{src} -> {dest}")

            # Handle range-dependent SSP if using Quad interpolation
            if interp_char == 'Q' and env.has_range_dependent_ssp():
                # Write external .ssp file for Quad interpolation
                from uacpy.io.env_writer import write_ssp
                ssp_file = filepath.with_suffix('.ssp')
                write_ssp(ssp_file, env.ssp_2d_ranges, env.ssp_2d_matrix)
                if verbose:
                    print(f"[BellhopEnvWriter] Wrote range-dependent SSP file: {ssp_file}")

            # SSP depth range and data
            # For altimetry, extend SSP above surface to cover wave crests
            z_min = 0.0
            if has_altimetry:
                min_alti = env.altimetry[:, 1].min()
                if min_alti < 0:
                    z_min = min_alti - 0.5  # small margin below lowest wave trough

            max_bathy_depth = (
                env.bathymetry[:, 1].max() if len(env.bathymetry) > 0 else env.depth
            )
            z_max = max(env.depth, max_bathy_depth)

            # Extend SSP if needed
            ssp_data_extended = env.ssp_data.copy()
            # Extend upward for altimetry (same sound speed as surface)
            if z_min < ssp_data_extended[0, 0]:
                first_c = ssp_data_extended[0, 1]
                ssp_data_extended = np.vstack([[z_min, first_c], ssp_data_extended])
            # Extend downward for deep bathymetry
            if z_max > ssp_data_extended[-1, 0]:
                last_c = ssp_data_extended[-1, 1]
                ssp_data_extended = np.vstack([ssp_data_extended, [z_max, last_c]])

            n_ssp = len(ssp_data_extended)
            f.write(f"{n_ssp}  {z_min:.1f}  {z_max:.1f},\n")

            # Bellhop SSP format: z alphaR betaR rhoR alphaI betaI /
            # When top BC is elastic halfspace, Fortran READ keeps previous
            # values for unspecified columns — so we must write all 6 to
            # prevent ice properties from bleeding into the water column.
            # (attenuation_unit='m' would require an additional (BETA, fT)
            # pair; that path is rejected above with ConfigurationError.)
            for depth, c in ssp_data_extended:
                f.write(
                    f"{depth:.6f} {c:.6f} 0.0 1.0 "
                    f"{env.attenuation:.6f} 0.0 /\n"
                )

            # Bottom properties
            # Map human-readable names to Bellhop codes
            bottom_type_map = {
                "vacuum": "V",
                "rigid": "R",
                "half-space": "A",  # Acousto-elastic halfspace
                "halfspace": "A",
                "grain-size": "G",  # Grain size (UW-APL HF Handbook)
                "grain": "G",
                "file": "F",        # Reflection coefficient from file
                "precalc": "P",     # Precalculated
            }
            bottom_type = bottom_type_map.get(env.bottom.acoustic_type.lower(), "A")

            # Check if bathymetry is range-dependent (more than 1 point)
            is_range_dependent_bathy = len(env.bathymetry) > 1

            if is_range_dependent_bathy:
                # Write .bty file for range-dependent bathymetry.
                bty_filepath = filepath.with_suffix(".bty")
                # The 2nd TYPE char in the .bty (short 'S' vs long 'L') is
                # auto-selected by the writer: write_bty_long_format emits
                # 'LL'/'CL', write_bty_file emits 'LS'/'CS'. Callers only pick
                # the 1st char (interpolation) via bty_interp_type.
                bottom_rd = getattr(env, 'bottom_rd', None)
                if bottom_rd is not None and len(getattr(bottom_rd, 'ranges_km', [])) > 0:
                    write_bty_long_format(
                        bty_filepath, env.bathymetry, bottom_rd,
                        interp_type=bty_interp_type,
                    )
                else:
                    write_bty_file(
                        bty_filepath, env.bathymetry,
                        interp_type=bty_interp_type,
                    )
                # Append '~' to bottom type to indicate bathymetry from file
                # Format: 'A~' = halfspace with range-dependent bathymetry
                bottom_type_with_bathy = f"{bottom_type}~"
                f.write(f"'{bottom_type_with_bathy}' 0.0\n")
            else:
                # Range-independent bottom
                f.write(f"'{bottom_type}' 0.0\n")  # Top of halfspace (usually 0)

            # Write halfspace parameters (for range-independent or as defaults)
            if bottom_type == "G":  # Grain size
                # Format: depth Mz (mean grain size in phi units)
                grain_size = getattr(env.bottom, 'grain_size_phi', 1.0)
                f.write(f" {env.bottom.depth:.2f}  {grain_size:.2f} /\n")
            elif bottom_type == "F":  # Reflection coefficient from file
                # Copy reflection file to working directory
                if env.bottom.reflection_file:
                    import shutil
                    brc_source = Path(env.bottom.reflection_file)
                    if brc_source.exists():
                        # Copy to same directory as .env file with matching base name
                        brc_dest = filepath.with_suffix('.brc')
                        shutil.copy(brc_source, brc_dest)
                        if verbose:
                            print(f"[BellhopEnvWriter] Copied reflection file: {brc_source} -> {brc_dest}")
                    else:
                        raise FileNotFoundError(
                            f"Reflection coefficient file not found: {env.bottom.reflection_file}\n"
                            f"Generate this file using BOUNCE or OASR models."
                        )
                else:
                    raise ValueError(
                        "acoustic_type='file' requires reflection_file parameter.\n"
                        "Example: BoundaryProperties(acoustic_type='file', reflection_file='path/to/file.brc')"
                    )

                # For 'F' type, Bellhop finds the .brc file by name convention
                # (same base name as .env). No additional lines needed in the env file.
            elif bottom_type == "A":  # Acousto-elastic halfspace
                # Format: depth sound_speed shear_speed density attenuation roughness /
                shear_speed = getattr(env.bottom, 'shear_speed', 0.0)
                roughness = getattr(env.bottom, "roughness", 0.0)
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
                # Get parameters from kwargs with sensible defaults
                beam_width_type = kwargs.get('beam_width_type', 'F')  # Filling
                beam_curvature = kwargs.get('beam_curvature', 'D')    # Double
                eps_multiplier = kwargs.get('eps_multiplier', 1.0)
                r_loop = kwargs.get('r_loop', 1.0)  # Range for choosing beam width (km)
                n_image = kwargs.get('n_image', 1)  # Number of images
                ib_win = kwargs.get('ib_win', 4)    # Beam windowing parameter
                component = kwargs.get('component', 'P')  # 'P' for pressure, 'D' for displacement

                # ALWAYS write these two lines for beam_type 'C' or 'R'
                # Line 1: beam width/curvature type, epsilon multiplier, r_loop
                beam_type_str = f"{beam_width_type}{beam_curvature}"
                f.write(f"'{beam_type_str}' {eps_multiplier:.6f} {r_loop:.6f}\n")

                # Line 2: n_image, ib_win, component
                f.write(f"{n_image} {ib_win} '{component}'\n")
