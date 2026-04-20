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
from uacpy.io.bty_writer import write_bty_file, write_ati_file
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
        beam_shift: bool = False,
        volume_attenuation: Optional[str] = None,
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
            Run type (position 1): 'C'/'I'/'S'/'A'/'E'/'R'. Default is 'C'.
            - C: Coherent TL
            - I: Incoherent TL
            - S: Semi-coherent TL
            - A: Arrivals
            - E: Eigenrays
            - R: Ray trace
        beam_type : str, optional
            Beam type (position 2): 'B'/'R'/'C'/'b'/'g'/'G'/'S'. Default is 'B'.
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
        beam_shift : bool, optional
            Enable beam shift on reflection. Default is False.
        volume_attenuation : str, optional
            Volume attenuation formula: 'T' (Thorp), 'F' (Francois-Garrison),
            'B' (Biological), or None. Default is None.
        n_beams : int, optional
            Number of beams. If 0, uses source.n_angles.
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
            # Position 3: Unit type (W=dB/wavelength - most common)
            # Position 4: Volume attenuation (T=Thorp, F=Francois-Garrison, B=Biological, ' '=none)
            vol_atten_char = volume_attenuation.upper() if volume_attenuation else ' '
            atten_unit = f"W{vol_atten_char}"  # dB/wavelength + volume attenuation

            # Position 5: Altimetry flag ('~' = read .ati file, ' ' = flat surface)
            has_altimetry = getattr(env, 'altimetry', None) is not None and len(env.altimetry) > 1
            alti_char = '~' if has_altimetry else ' '

            # SSP option string (pad to 6 chars for Fortran compatibility)
            ssp_options = f"{interp_char}{top_bc}{atten_unit}{alti_char}"
            ssp_options = ssp_options.ljust(6)  # Pad with spaces to 6 characters
            f.write(f"'{ssp_options}'\n")

            write_surface_halfspace(f, env)

            # Write .ati file if altimetry provided
            if has_altimetry:
                ati_filepath = filepath.with_suffix(".ati")
                write_ati_file(ati_filepath, env.altimetry, interp_type="L")
                if verbose:
                    print(f"[BellhopEnvWriter] Wrote altimetry file: {ati_filepath}")

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
            for depth, c in ssp_data_extended:
                f.write(f"{depth:.6f} {c:.6f} 0.0 1.0 "
                        f"{env.attenuation:.6f} 0.0 /\n")

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
            # Format: 'XXYYZZ' where:
            #   XX = run_type + beam_type (positions 1-2)
            #   YY = reserved + source_type (positions 3-4)
            #   ZZ = grid_type + dimension (positions 5-6)
            #   + optional beam_shift (position 7)
            position_3 = ' '  # Reserved
            position_6 = '2'  # Dimension (always '2' for 2D Bellhop)
            beam_shift_char = 'S' if beam_shift else ''

            full_run_type = f"{run_type}{beam_type}{position_3}{source_type}{grid_type}{position_6}{beam_shift_char}"
            f.write(f"'{full_run_type}'\n")

            # Number of beams
            f.write(f"{n_beams}\n")

            # Launch angles
            f.write(f"{alpha[0]:.6f} {alpha[1]:.6f} /\n")

            # Step size (0 for automatic)
            f.write(f"{step:.6f}\n")

            # Box parameters (z in m, r in km as per Bellhop documentation)
            f.write(f"{z_box:.6f} {r_box/1000.0:.6f}\n")

            # Cerveny beam parameters (for beam_type 'C' or 'R')
            # These parameters are REQUIRED by Bellhop Fortran when beam_type is 'C' or 'R'
            # Reference: ReadEnvironmentBell.f90 lines 196-217
            if beam_type.upper() in ['C', 'R']:
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
